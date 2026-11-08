#!/usr/bin/env python3
"""Assert voice-loop harness results against each scenario's `expect` block.

home-ops#584. The harness (crispasr-voice --harness ...) writes
<out>/<scenario>/summary.json plus metrics.jsonl (one TurnMetrics record per
run, warm-up runs included). This script is the pass/fail gate: it reads the
scenarios and the summaries, applies the thresholds, prints one line per
scenario and exits non-zero on any failure.

Standard library only.

Usage:
  python3 tests/voice/check_metrics.py                     # default out dir
  python3 tests/voice/check_metrics.py --out tests/voice/out
  python3 tests/voice/check_metrics.py --scenario delegation
  python3 tests/voice/check_metrics.py --self-test         # gate the gate

Supported expectation keys
--------------------------
  eos_to_first_played_ms_p95 : float   p95(end of speech -> first played PCM)
  max_gap_ms                : float   largest playback gap across measured runs
  eos_to_first_audible_ms_p95: float  p95(WAV-derived first audible audio minus
                                      end of the input speech), see below
  min_filler_played         : int     minimum fillers_played in any measured run
  answer_played             : bool    the answer reached the sink in every run
  barge_in_detected         : bool    at least one barge-in per measured run
  barge_in_stop_ms          : float   p95(speech start -> silence)
  honest_failure_text       : bool    failure was spoken and the task was not
                                      dispatched twice (proxy, see NOTE below)
  max_task_dispatches       : int     task_dispatches per run must not exceed
                                      this (every actual adapter dispatch,
                                      including a speculative + corrective one)
  max_stale_results_dropped : int     stale_results_dropped per run <= this
  min_stale_results_dropped : int     stale_results_dropped per run >= this
                                      (proves the stale path was taken)
  max_tool_calls_valid      : int     tool_calls_valid per run <= this

WAV checks — the bookkeeping can lie, the audio cannot
-----------------------------------------------------
Every measured run's run<N>.wav is measured on every invocation, whether or not
the scenario asserts on it. The per-scenario line always prints:

  wav_gap        longest run of 20 ms frames below RMS 0.01 that sits inside
                 audio (leading silence is skipped, trailing silence counts:
                 it is still dead air for the listener)
  audible_p50/95 first audible sample in run<N>.wav (10 ms frame with RMS
                 > 0.01) minus the last audible sample in the scenario's input
                 WAV, in ms. The input WAV comes from summary.json's
                 `input_wav`; this is the WAV's own end-of-speech -> first
                 audible latency, independent of what the loop reports.
A note is printed when the self-reported p95(eos_to_first_played_ms) and the
WAV-derived p95 disagree by more than 100 ms, and when a self-reported gap
under-reports the WAV silence run.

A missing run<N>.wav or input WAV is a failure that names the file: a value is
never silently substituted with 0, because a zero always looks like a pass.
A run whose WAV contains no audible audio is noted (a barge-in turn may never
get the floor) but makes `eos_to_first_audible_ms_p95` unverifiable, so that
expectation fails when it is declared. `max_gap_ms` fails on
max(self-reported, WAV-derived).

The scenario directory is resolved from the scenario's `name` field (the
harness writes <out>/<name>/), and cross-checked against summary.json's own
`scenario` field.

NOTE on `honest_failure_text`: TurnMetrics has no text field (voice_types.h is a
contract header and is not edited for this experiment), so "the user heard an
honest failure" is asserted through what is observable: an answer clause was
synthesised and played (answer_first_played_ms > 0) and the failing task was
dispatched at most once (task_dispatches <= 1). A spoken transcript is the
missing signal; this is reported as a contract gap in the harness report.
"""

import argparse
import array
import glob
import json
import math
import os
import shutil
import sys
import tempfile
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
DEFAULT_OUT = os.path.join(HERE, "out")
DEFAULT_SCENARIOS = os.path.join(HERE, "scenarios")

# Thresholds that are asserted on the p95 aggregate (spiky-by-nature targets).
P95_KEYS = {"eos_to_first_played_ms_p95", "barge_in_stop_ms",
            "eos_to_first_audible_ms_p95"}

# WAV analysis windows. Silence runs use the harness's 20 ms playback quantum;
# audibility uses 10 ms so a soft word onset is not hidden inside one frame.
WAV_GAP_FRAME_MS = 20
WAV_AUDIBLE_FRAME_MS = 10
WAV_RMS_THRESHOLD = 0.01
# Report a self-reported vs WAV-derived latency disagreement above this.
LATENCY_MISMATCH_MS = 100.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def percentile(values, q):
    """Nearest-rank percentile, matching harness.cpp."""
    if not values:
        return 0.0
    v = sorted(values)
    idx = int(math.ceil(q * len(v))) - 1
    idx = max(0, min(idx, len(v) - 1))
    return float(v[idx])


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: not JSON ({exc})") from exc
    return records


def summary_metric(summary, key, stat="p95"):
    metrics = summary.get("metrics") or {}
    entry = metrics.get(key)
    if not isinstance(entry, dict):
        return None
    return entry.get(stat)


def run_value(record, key):
    """Reads a raw or derived field from one per-run metrics record."""
    if key in record:
        return record[key]
    # Derive what metrics_json() may not have emitted.
    if key == "eos_to_first_played_ms":
        return max(0, record.get("ack_first_played_ms", 0) - record.get("speech_end_ms", 0))
    if key == "eos_to_final_ms":
        return max(0, record.get("final_transcript_ms", 0) - record.get("speech_end_ms", 0))
    if key == "max_gap_ms":
        gaps = record.get("gaps") or []
        return max((g.get("duration_ms", 0) for g in gaps), default=0)
    return None


def measured_runs(records):
    runs = [r for r in records if not r.get("warmup")]
    return runs if runs else list(records)


def resolve_summary_path(value, run_dir):
    """Resolves a path recorded in summary.json.

    The harness records input WAVs relative to the working directory it ran
    in (normally the repo root), so try the value as written, relative to the
    scenario output directory and relative to the repo root.
    """
    if not value:
        return None
    candidates = [value]
    if not os.path.isabs(value):
        candidates.append(os.path.join(run_dir, value))
        candidates.append(os.path.join(REPO_ROOT, value))
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


# ---------------------------------------------------------------------------
# WAV ground truth: the metrics can lie, the audio cannot.
# ---------------------------------------------------------------------------
def read_wav_mono(path):
    """Returns (rate, array('h') samples) for a 16-bit PCM WAV, mixed to mono."""
    with wave.open(path, "rb") as w:
        rate = w.getframerate()
        channels = w.getnchannels()
        width = w.getsampwidth()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("unsupported sample width %d bit (need 16-bit PCM)"
                         % (width * 8))
    samples = array.array("h")
    samples.frombytes(raw[: len(raw) // 2 * 2])
    if sys.byteorder == "big":
        samples.byteswap()
    if channels > 1:
        samples = array.array(
            "h",
            (sum(samples[i:i + channels]) // channels
             for i in range(0, len(samples) - channels + 1, channels)))
    return rate, samples


def _frame_energies(samples, frame):
    """Sum of squares for each consecutive frame of `frame` samples."""
    return [sum(x * x for x in samples[i:i + frame])
            for i in range(0, len(samples) - frame + 1, frame)]


def _youngest_oldest(loud):
    first = next((i for i, on in enumerate(loud) if on), None)
    last = next((i for i, on in enumerate(reversed(loud)) if on), None)
    return first, None if last is None else len(loud) - last


def wav_profile(path, cache=None):
    """WAV-derived facts for one file.

    first_audible_ms   : start of the first 10 ms frame with RMS > 0.01 (None
                         when the file never gets above the threshold)
    last_audible_ms    : end of the last 10 ms frame with RMS > 0.01
    max_silence_run_ms : longest run of 20 ms frames at or below the threshold
                         inside audio; leading silence is skipped, trailing
                         silence counts
    duration_ms        : file length
    """
    key = os.path.abspath(path)
    if cache is not None and key in cache:
        return cache[key]
    rate, samples = read_wav_mono(path)
    frame10 = max(1, int(round(rate * WAV_AUDIBLE_FRAME_MS / 1000.0)))
    frame20 = max(1, int(round(rate * WAV_GAP_FRAME_MS / 1000.0)))
    thr2 = (WAV_RMS_THRESHOLD * 32768.0) ** 2
    loud10 = [e > thr2 * frame10 for e in _frame_energies(samples, frame10)]
    first10, last10 = _youngest_oldest(loud10)
    silence_ms = 0.0
    started = False
    run = 0
    for energy in _frame_energies(samples, frame20):
        if energy > thr2 * frame20:
            if run:
                silence_ms = max(silence_ms, run * frame20 * 1000.0 / rate)
                run = 0
            started = True
        elif started:
            run += 1
    if run:
        silence_ms = max(silence_ms, run * frame20 * 1000.0 / rate)
    profile = {
        "path": path,
        "rate": rate,
        "duration_ms": 1000.0 * len(samples) / rate if rate else 0.0,
        "first_audible_ms": None if first10 is None else first10 * frame10 * 1000.0 / rate,
        "last_audible_ms": None if last10 is None else last10 * frame10 * 1000.0 / rate,
        "max_silence_run_ms": silence_ms,
    }
    if cache is not None:
        cache[key] = profile
    return profile


def wav_evidence(name, run_dir, measured, summary, cache, problems):
    """Measures every WAV the scenario produced.

    Never fabricates a number: a missing file becomes a problem naming the
    file, so a broken harness cannot look like a passing one.
    """
    ev = {"run_dir": run_dir, "wav_gap": None, "audible": [], "runs": {},
          "input": None, "missing": [], "silent": []}

    input_path = resolve_summary_path(summary.get("input_wav"), run_dir)
    if not input_path:
        problems.append(
            f"{name}: summary.json input_wav {summary.get('input_wav')!r} cannot be "
            f"resolved; the WAV latency check has no end-of-speech reference")
    else:
        try:
            ev["input"] = wav_profile(input_path, cache)
        except (OSError, ValueError, wave.Error) as exc:
            problems.append(f"{name}: cannot read input WAV {input_path}: {exc}")

    for record in measured:
        run_no = record.get("run")
        if run_no is None:
            problems.append(
                f"{name}: measured record without a 'run' number — cannot find its WAV")
            continue
        path = os.path.join(run_dir, "run%s.wav" % run_no)
        if not os.path.exists(path):
            problems.append(
                f"{name}: {path} is missing — run {run_no} cannot be verified against "
                f"the audio; its self-reported numbers are unverified")
            ev["missing"].append(path)
            continue
        try:
            profile = wav_profile(path, cache)
        except (OSError, ValueError, wave.Error) as exc:
            problems.append(f"{name}: cannot read {path}: {exc}")
            continue
        ev["runs"][run_no] = profile
        if ev["wav_gap"] is None:
            ev["wav_gap"] = profile["max_silence_run_ms"]
        else:
            ev["wav_gap"] = max(ev["wav_gap"], profile["max_silence_run_ms"])
        if profile["first_audible_ms"] is None:
            # Legitimate for a barge-in turn that never gets the floor; it just
            # means a WAV latency cannot be measured for this run. A scenario
            # that asserts on it fails (see check_expect).
            ev["silent"].append(run_no)
            continue
        if ev["input"] and ev["input"]["last_audible_ms"] is not None:
            ev["audible"].append(profile["first_audible_ms"] - ev["input"]["last_audible_ms"])
    return ev


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------
def check_expect(name, expect, summary, records, wav=None):
    """Returns (problems, notes) for one scenario."""
    problems = []
    notes = []
    measured = measured_runs(records)
    if not measured:
        return [f"{name}: no measured runs in metrics.jsonl"], notes

    def p95_of(key):
        v = summary_metric(summary, key, "p95")
        if v is not None:
            return float(v)
        vals = [run_value(r, key) for r in measured]
        vals = [float(x) for x in vals if x is not None]
        return percentile(vals, 0.95) if vals else None

    def max_of(key):
        v = summary_metric(summary, key, "max")
        if v is not None:
            return float(v)
        vals = [run_value(r, key) for r in measured]
        vals = [float(x) for x in vals if x is not None]
        return max(vals) if vals else None

    def min_of(key):
        v = summary_metric(summary, key, "min")
        if v is not None:
            return float(v)
        vals = [run_value(r, key) for r in measured]
        vals = [float(x) for x in vals if x is not None]
        return min(vals) if vals else None

    def every(predicate, describe):
        bad = [r.get("run", i + 1) for i, r in enumerate(measured) if not predicate(r)]
        if bad:
            problems.append(f"{name}: {describe} failed on run(s) {bad}")
        return not bad

    wav_gap = (wav or {}).get("wav_gap")
    audible = (wav or {}).get("audible") or []

    for key in sorted(expect):
        want = expect[key]

        if key == "eos_to_first_played_ms_p95":
            got = p95_of("eos_to_first_played_ms")
            if got is None:
                problems.append(f"{name}: eos_to_first_played_ms missing from summary.json")
            elif got > float(want):
                problems.append(
                    f"{name}: eos_to_first_played_ms p95 {got:.0f} ms > {want} ms")

        elif key == "eos_to_first_audible_ms_p95":
            silent = (wav or {}).get("silent") or []
            if not audible:
                problems.append(
                    f"{name}: eos_to_first_audible_ms unavailable — <= {want} ms cannot "
                    f"be verified without both the run and the input WAVs")
            elif silent:
                problems.append(
                    f"{name}: run(s) {silent} produced no audible audio at all, so "
                    f"eos_to_first_audible_ms p95 cannot be trusted for <= {want} ms")
            else:
                got = percentile(audible, 0.95)
                if got > float(want):
                    problems.append(
                        f"{name}: eos_to_first_audible_ms p95 {got:.0f} ms > {want} ms "
                        f"(WAV-derived from {len(audible)} run(s))")

        elif key == "max_gap_ms":
            got = max_of("max_gap_ms")
            p95 = p95_of("max_gap_ms")
            if got is None:
                problems.append(f"{name}: max_gap_ms missing from summary.json")
            elif wav_gap is not None and max(got, wav_gap) > float(want):
                problems.append(
                    f"{name}: playback gap self={got:.0f} ms wav={wav_gap:.0f} ms > "
                    f"{want} ms (p95 {p95:.0f} ms)")

        elif key == "min_filler_played":
            got = min_of("fillers_played")
            if got is None:
                problems.append(f"{name}: fillers_played missing from summary.json")
            elif got < float(want):
                problems.append(
                    f"{name}: only {got:.0f} filler(s) played in the leanest run, "
                    f"expected >= {want}")

        elif key == "answer_played":
            if want:
                def played(r):
                    v = run_value(r, "answer_first_played_ms")
                    return v is not None and v > 0
                every(played, "answer_played (answer_first_played_ms > 0)")

        elif key == "barge_in_detected":
            if want:
                every(lambda r: int(r.get("barge_ins", 0) or 0) >= 1,
                      "barge_in detected (barge_ins >= 1)")

        elif key == "barge_in_stop_ms":
            got = p95_of("barge_in_stop_ms")
            if got is None:
                problems.append(f"{name}: barge_in_stop_ms missing from summary.json")
            elif got > float(want):
                problems.append(
                    f"{name}: barge-in stop p95 {got:.0f} ms > {want} ms")

        elif key == "honest_failure_text":
            if want:
                notes.append(
                    "honest_failure_text is asserted as: failure clause played and "
                    "task dispatched once (TurnMetrics has no text field)")
                every(lambda r: (run_value(r, "answer_first_played_ms") or 0) > 0,
                      "honest failure spoken (answer_first_played_ms > 0)")
                every(lambda r: int(r.get("task_dispatches", 0) or 0) <= 1,
                      "no duplicate task dispatch (task_dispatches <= 1)")

        elif key == "max_task_dispatches":
            # Adapter dispatches, not model attempts: a corrective redispatch is
            # a real second dispatch that `tool_calls` (attempts) does not see.
            limit = int(want)
            every(lambda r, limit=limit: int(r.get("task_dispatches", 0) or 0) <= limit,
                  f"task dispatched at most {limit}x (task_dispatches)")

        elif key == "max_stale_results_dropped":
            limit = int(want)
            every(lambda r, limit=limit: int(r.get("stale_results_dropped", 0) or 0) <= limit,
                  f"at most {limit} stale result(s) dropped")

        elif key == "min_stale_results_dropped":
            floor = int(want)
            every(lambda r, floor=floor: int(r.get("stale_results_dropped", 0) or 0) >= floor,
                  f"at least {floor} stale result(s) dropped")

        elif key == "max_tool_calls_valid":
            limit = int(want)
            every(lambda r, limit=limit: int(r.get("tool_calls_valid", 0) or 0) <= limit,
                  f"at most {limit} schema-valid tool call(s)")

        else:
            problems.append(f"{name}: unknown expectation key '{key}'")

    return problems, notes


# ---------------------------------------------------------------------------
# scenario runner
# ---------------------------------------------------------------------------
def _ms(value):
    return "-" if value is None else f"{value:.0f}ms"


def evaluate(scenarios_dir, out_dir, only=None):
    """Returns (problems, notes, checked_count)."""
    problems, notes = [], []
    entries = []
    for path in sorted(glob.glob(os.path.join(scenarios_dir, "*.json"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        try:
            scenario = load_json(path)
        except (OSError, ValueError) as exc:
            problems.append(f"{stem}: cannot read scenario {path}: {exc}")
            continue
        entries.append((path, stem, scenario))

    if only:
        wanted = set(only)
        entries = [e for e in entries
                   if e[1] in wanted or str(e[2].get("name") or e[1]) in wanted]

    wav_cache = {}
    checked = 0
    for path, stem, scenario in entries:
        # The harness writes out/<name>/, where <name> is the scenario's `name`
        # field (falling back to the file stem when the field is absent).
        declared = str(scenario.get("name") or stem)
        scen_out = os.path.join(out_dir, declared)
        summary_path = os.path.join(scen_out, "summary.json")
        if declared != stem:
            notes.append(
                f"{stem}: scenario file stem differs from its name field "
                f"('{declared}'); the harness writes {scen_out}")

        if not os.path.exists(summary_path):
            problems.append(
                f"{stem}: missing {summary_path} "
                f"(run: crispasr-voice --harness {os.path.relpath(path)})")
            continue

        try:
            summary = load_json(summary_path)
        except (OSError, ValueError) as exc:
            problems.append(f"{stem}: cannot read {summary_path}: {exc}")
            continue

        # Trust the name the harness itself recorded when it disagrees.
        run_name = str(summary.get("scenario") or declared)
        run_dir = os.path.join(out_dir, run_name)
        if run_name != declared:
            problems.append(
                f"{stem}: scenario name mismatch — scenario says '{declared}', "
                f"summary.json says '{run_name}'; using {run_dir}")

        try:
            records = load_jsonl(os.path.join(run_dir, "metrics.jsonl"))
        except (OSError, ValueError) as exc:
            problems.append(f"{stem}: cannot read {run_dir}/metrics.jsonl: {exc}")
            continue

        problem_start = len(problems)
        note_start = len(notes)
        expect = scenario.get("expect") or {}
        if not expect:
            notes.append(f"{stem}: no expect block, nothing asserted")

        # Sanity: the harness itself must not have reported failures.
        if not summary.get("ok", True):
            problems.append(
                f"{stem}: harness reported {summary.get('runs_failed')} failed run(s)")
        want_runs = int(scenario.get("runs", summary.get("runs", 0)) or 0)
        measured = measured_runs(records)
        got_runs = len(measured)
        if want_runs and got_runs != want_runs:
            problems.append(
                f"{stem}: {got_runs} measured run(s) recorded, scenario asks for {want_runs}")

        # WAV evidence for every measured run, asserted or not.
        wav = wav_evidence(stem, run_dir, measured, summary, wav_cache, problems)

        p, n = check_expect(stem, expect, summary, records, wav=wav)
        problems.extend(p)
        notes.extend(n)

        eos = summary_metric(summary, "eos_to_first_played_ms", "p95")
        gap = summary_metric(summary, "max_gap_ms", "max")
        fill = summary_metric(summary, "fillers_played", "min")
        wav_p50 = percentile(wav["audible"], 0.50) if wav["audible"] else None
        wav_p95 = percentile(wav["audible"], 0.95) if wav["audible"] else None

        if wav["silent"]:
            notes.append(
                f"{stem}: run(s) {wav['silent']} produced no audible audio, so the "
                f"WAV latency is undefined for them")

        self_gap = None if gap is None else float(gap)
        if wav["wav_gap"] is not None and self_gap is not None \
                and wav["wav_gap"] > self_gap + 50.0:
            notes.append(
                f"{stem}: self-reported max_gap {self_gap:.0f} ms under-reports the "
                f"WAV silence run ({wav['wav_gap']:.0f} ms)")
        if eos is not None and wav_p95 is not None:
            diff = wav_p95 - float(eos)
            if abs(diff) > LATENCY_MISMATCH_MS:
                notes.append(
                    f"{stem}: self-reported eos_to_first_played p95 {float(eos):.0f} ms vs "
                    f"WAV-derived eos_to_first_audible p95 {wav_p95:.0f} ms "
                    f"(diff {diff:+.0f} ms)")

        checked += 1
        scen_problems = problems[problem_start:]
        status = "FAIL" if scen_problems else "ok"
        print(
            f"  {stem:16s} {status:4s} "
            f"eos_p95={_ms(None if eos is None else float(eos))} "
            f"max_gap={_ms(self_gap)} "
            f"wav_gap={_ms(wav['wav_gap'])} "
            f"audible_p50={_ms(wav_p50)} audible_p95={_ms(wav_p95)} "
            f"fillers_min={'-' if fill is None else f'{float(fill):.0f}'} "
            f"runs={got_runs}"
        )
        for note in notes[note_start:]:
            print(f"    note: {note}")
        for prob in scen_problems:
            print(f"    {prob}")

    return problems, notes, checked


# ---------------------------------------------------------------------------
# self-test: the gate itself must fail when it should
# ---------------------------------------------------------------------------
def _write_tone_wav(path, rate, total_ms, spans):
    """Writes a 16-bit mono WAV: silence except inside the given (start, end) ms."""
    n = int(rate * total_ms / 1000.0)
    buf = array.array("h", [0]) * n
    for start_ms, end_ms in spans:
        a = max(0, int(rate * start_ms / 1000.0))
        b = min(n, int(rate * end_ms / 1000.0))
        for i in range(a, b):
            buf[i] = int(0.3 * 32767.0 * math.sin(2.0 * math.pi * 220.0 * (i - a) / rate))
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(buf.tobytes())


def _write_case(root, name, expect, runs, ok=True, runs_failed=0,
                declared=None, wav=None):
    """Writes a scenario + summary + metrics.jsonl + WAVs under root.

    `declared` is the scenario's `name` field and therefore its output
    directory. `wav` controls the synthetic audio:
      input_rate / input_speech_ms  input WAV: tone from 0 to input_speech_ms
      first_audible_ms             first audible sample in every run WAV
      run_ms                       run WAV length
      holes                        {run_no: (start_ms, duration_ms)} silent holes
      silent_runs                  run numbers whose run wav is all silence
      missing                      run numbers whose run wav is not written
    """
    declared = declared or name
    wav = dict(wav or {})
    input_speech_ms = wav.get("input_speech_ms", 1000)
    first_audible_ms = wav.get("first_audible_ms", input_speech_ms + 300)
    run_ms = wav.get("run_ms", 4000)
    holes = wav.get("holes") or {}
    silent_runs = set(wav.get("silent_runs") or [])
    missing = set(wav.get("missing") or [])
    rate = 16000

    scen_dir = os.path.join(root, "out", declared)
    os.makedirs(scen_dir, exist_ok=True)
    with open(os.path.join(root, "scenarios", f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"name": declared, "input": "input.wav", "runs": len(runs),
                   "warmup": 0, "expect": expect}, f)

    records = []
    for i, r in enumerate(runs, 1):
        rec = dict(r)
        rec.setdefault("run", i)
        rec.setdefault("warmup", False)
        records.append(rec)

    input_path = os.path.join(scen_dir, "input.wav")
    _write_tone_wav(input_path, rate, input_speech_ms + 500, [(0, input_speech_ms)])
    for rec in records:
        if rec["warmup"] or rec["run"] in missing:
            continue
        spans = [] if rec["run"] in silent_runs else [(first_audible_ms, run_ms)]
        hole = holes.get(rec["run"])
        if hole:
            hole_start, hole_ms = hole
            spans = [(first_audible_ms, hole_start), (hole_start + hole_ms, run_ms)]
        _write_tone_wav(os.path.join(scen_dir, "run%s.wav" % rec["run"]),
                        rate, run_ms, spans)

    measured = [r for r in records if not r["warmup"]]
    metrics = {}
    keys = ["eos_to_first_played_ms", "max_gap_ms", "fillers_played",
            "answer_first_played_ms", "barge_ins", "barge_in_stop_ms", "tool_calls",
            "tool_calls_valid", "task_dispatches", "stale_results_dropped"]
    for key in keys:
        vals = [float(r.get(key, 0) or 0) for r in measured]
        if not vals:
            continue
        metrics[key] = {"p50": percentile(vals, 0.50), "p95": percentile(vals, 0.95),
                        "min": min(vals), "max": max(vals), "n": len(vals)}
    summary = {"scenario": declared, "input_wav": input_path,
               "runs": len(measured), "warmup": 0,
               "runs_total": len(records), "runs_failed": runs_failed,
               "ok": ok and runs_failed == 0, "expect": expect, "metrics": metrics}
    with open(os.path.join(scen_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f)
    with open(os.path.join(scen_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def self_test():
    root = tempfile.mkdtemp(prefix="voice-check-selftest-")
    os.makedirs(os.path.join(root, "scenarios"), exist_ok=True)

    def base(eos=300, gap=0, fillers=1, answer=9000, barge=1, stop=120, tools=1,
             dispatches=None, valid=1, stale=0):
        return {"eos_to_first_played_ms": eos, "max_gap_ms": gap, "fillers_played": fillers,
                "answer_first_played_ms": answer, "barge_ins": barge,
                "barge_in_stop_ms": stop, "tool_calls": tools,
                "task_dispatches": tools if dispatches is None else dispatches,
                "tool_calls_valid": valid, "stale_results_dropped": stale}

    # (name, expect, per-run records, summarised ok, expect_problems, kwargs)
    cases = [
        ("pass_short", {"eos_to_first_played_ms_p95": 1800, "answer_played": True},
         [dict(base()), dict(base(eos=1000)), dict(base(eos=1100))], True, False, {}),
        ("fail_short_slow", {"eos_to_first_played_ms_p95": 1800, "answer_played": True},
         [dict(base(eos=1900)), dict(base(eos=2100)), dict(base(eos=2300))], True, True, {}),
        ("fail_no_answer", {"answer_played": True},
         [dict(base()), dict(base(answer=0)), dict(base())], True, True, {}),
        ("pass_delegation", {"max_gap_ms": 150, "min_filler_played": 1, "answer_played": True},
         [dict(base(gap=20, fillers=2)), dict(base(gap=60, fillers=1)),
          dict(base(gap=140, fillers=3))], True, False, {}),
        ("fail_gap", {"max_gap_ms": 150},
         [dict(base(gap=20)), dict(base(gap=420)), dict(base(gap=30))], True, True, {}),
        # Self-reported latency that disagrees with the audio is reported as a
        # note (the WAV value is the one that is asserted).
        ("pass_latency_mismatch_noted", {"answer_played": True},
         [dict(base(eos=40))], True, False, {}),
        # The WAV is the authority: self-reported gap 0, 320 ms hole in run 2.
        ("fail_wav_gap_only", {"max_gap_ms": 150},
         [dict(base(gap=0)), dict(base(gap=0)), dict(base(gap=0))], True, True,
         {"wav": {"holes": {2: (2200, 320)}}}),
        # Same audio, no hole: the WAV check must not fail a clean run.
        ("pass_wav_gap_clean", {"max_gap_ms": 150},
         [dict(base(gap=0)) for _ in range(3)], True, False, {}),
        ("fail_no_filler", {"min_filler_played": 1},
         [dict(base(fillers=2)), dict(base(fillers=0)), dict(base(fillers=1))], True, True, {}),
        ("pass_barge_in", {"barge_in_detected": True, "barge_in_stop_ms": 250},
         [dict(base(stop=90)), dict(base(stop=130)), dict(base(stop=180))], True, False, {}),
        ("fail_no_barge_in", {"barge_in_detected": True},
         [dict(base(barge=1)), dict(base(barge=0)), dict(base(barge=1))], True, True, {}),
        ("fail_slow_barge_in", {"barge_in_stop_ms": 250},
         [dict(base(stop=90)), dict(base(stop=300)), dict(base(stop=310))], True, True, {}),
        ("pass_task_failure", {"honest_failure_text": True, "max_task_dispatches": 1},
         [dict(base(tools=1)), dict(base(tools=1))], True, False, {}),
        ("fail_duplicate_dispatch", {"max_task_dispatches": 1},
         [dict(base(tools=1)), dict(base(tools=2))], True, True, {}),
        # A redispatch is invisible in tool_calls (one model attempt, two
        # dispatches); the gate has to read task_dispatches.
        ("fail_duplicate_redispatch", {"max_task_dispatches": 1},
         [dict(base(tools=1, dispatches=1)), dict(base(tools=1, dispatches=2))],
         True, True, {}),
        ("pass_redispatch_allowed", {"max_task_dispatches": 2},
         [dict(base(tools=1, dispatches=2))], True, False, {}),
        ("pass_stale_expected", {"min_stale_results_dropped": 1},
         [dict(base(stale=1)), dict(base(stale=2))], True, False, {}),
        ("fail_stale_missing", {"min_stale_results_dropped": 1},
         [dict(base(stale=1)), dict(base(stale=0))], True, True, {}),
        ("pass_no_stale", {"max_stale_results_dropped": 0},
         [dict(base(stale=0))], True, False, {}),
        ("fail_stale_dropped", {"max_stale_results_dropped": 0},
         [dict(base(stale=1))], True, True, {}),
        ("pass_no_valid_call", {"max_tool_calls_valid": 0},
         [dict(base(valid=0))], True, False, {}),
        ("fail_valid_call_unexpected", {"max_tool_calls_valid": 0},
         [dict(base(valid=1))], True, True, {}),
        ("fail_no_failure_text", {"honest_failure_text": True},
         [dict(base(answer=0))], True, True, {}),
        ("fail_harness_error", {"answer_played": True},
         [dict(base())], False, True, {}),
        ("fail_unknown_key", {"no_such_metric": 1}, [dict(base())], True, True, {}),
        # WAV-derived latency: input speech ends at 1000 ms, first audible at
        # 1300 ms -> 300 ms.
        ("pass_wav_audible", {"eos_to_first_audible_ms_p95": 500},
         [dict(base(eos=0)) for _ in range(3)], True, False, {}),
        ("fail_wav_audible", {"eos_to_first_audible_ms_p95": 200},
         [dict(base(eos=0)) for _ in range(3)], True, True, {}),
        ("pass_wav_audible_p50", {"eos_to_first_audible_ms_p95": 320},
         [dict(base(eos=0)) for _ in range(3)], True, False, {}),
        # A run with no audible audio: noted for a barge-in scenario, fatal when
        # the scenario asserts on the WAV latency.
        ("pass_silent_wav_no_assert", {"barge_in_detected": True},
         [dict(base()) for _ in range(3)], True, False,
         {"wav": {"silent_runs": [1, 2, 3]}}),
        ("fail_audible_silent_run", {"eos_to_first_audible_ms_p95": 500},
         [dict(base()) for _ in range(3)], True, True,
         {"wav": {"silent_runs": [2]}}),
        # A missing run WAV must be reported, not treated as 0 ms of silence.
        ("fail_wav_missing_run", {"max_gap_ms": 150},
         [dict(base(gap=0)) for _ in range(3)], True, True,
         {"wav": {"missing": [2]}}),
        # The output directory comes from the scenario's name field.
        ("pass_dir_from_name", {"answer_played": True},
         [dict(base())], True, False, {"declared": "renamed_scenario"}),
    ]
    for name, expect, runs, ok, _want, kwargs in cases:
        _write_case(root, name, expect, runs, ok=ok,
                    runs_failed=0 if ok else 1, **kwargs)

    # A missing summary must be reported, not crashed on.
    with open(os.path.join(root, "scenarios", "missing.json"), "w", encoding="utf-8") as f:
        json.dump({"name": "missing", "input": "input.wav", "runs": 1,
                   "expect": {"answer_played": True}}, f)

    failures = 0
    print(f"self-test in {root}")
    for name, expect, runs, ok, want_problem, _kwargs in cases:
        problems, _notes, _checked = evaluate(
            os.path.join(root, "scenarios"), os.path.join(root, "out"), only=[name])
        got_problem = bool(problems)
        if got_problem != want_problem:
            failures += 1
            print(f"  MISMATCH {name}: expected {'fail' if want_problem else 'pass'}, "
                  f"got {'fail' if got_problem else 'pass'}")
            for p in problems:
                print(f"      {p}")
        else:
            print(f"  ok       {name}: {'rejected' if got_problem else 'accepted'}")

    problems, _notes, _checked = evaluate(
        os.path.join(root, "scenarios"), os.path.join(root, "out"), only=["missing"])
    if not problems:
        failures += 1
        print("  MISMATCH missing: expected a missing-summary problem")
    else:
        print("  ok       missing: reported")

    shutil.rmtree(root, ignore_errors=True)
    if failures:
        print(f"self-test FAILED ({failures} mismatch(es))")
        return 1
    print("self-test OK")
    return 0


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="harness output root (default: tests/voice/out)")
    ap.add_argument("--scenarios", default=DEFAULT_SCENARIOS,
                    help="scenario directory (default: tests/voice/scenarios)")
    ap.add_argument("--scenario", action="append", metavar="NAME",
                    help="check only this scenario (repeatable)")
    ap.add_argument("--self-test", "--selftest", dest="selftest", action="store_true",
                    help="verify the checker itself against synthetic results")
    args = ap.parse_args(argv)

    if args.selftest:
        return self_test()

    if not os.path.isdir(args.out):
        print(f"no harness output at '{args.out}' — nothing to check.\n")
        print(__doc__.split("Usage:")[0].strip())
        print("\nUsage:")
        print(f"  python3 {os.path.relpath(__file__)} --out {args.out}"
              f" [--scenario NAME] [--self-test]")
        return 0

    summaries = glob.glob(os.path.join(args.out, "*", "summary.json"))
    if not summaries:
        print(f"no summary.json under '{args.out}' — run the harness first, e.g.:\n")
        for path in sorted(glob.glob(os.path.join(args.scenarios, "*.json"))):
            print(f"  crispasr-voice --harness {os.path.relpath(path)}")
        return 0

    print(f"checking {len(summaries)} scenario result(s) in {args.out}")
    problems, _notes, checked = evaluate(args.scenarios, args.out, only=args.scenario)
    if checked == 0:
        print("no scenarios matched — check --scenarios/--scenario")
        return 2
    if problems:
        print(f"\nFAIL: {len(problems)} problem(s)")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"\nPASS: {checked} scenario(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
