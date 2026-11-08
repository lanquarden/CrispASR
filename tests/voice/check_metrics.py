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
  min_filler_played         : int     minimum fillers_played in any measured run
  answer_played             : bool    the answer reached the sink in every run
  barge_in_detected         : bool    at least one barge-in per measured run
  barge_in_stop_ms          : float   p95(speech start -> silence)
  honest_failure_text       : bool    failure was spoken and the task was not
                                      dispatched twice (proxy, see NOTE below)
  max_task_dispatches       : int     tool_calls per run must not exceed this

NOTE on `honest_failure_text`: TurnMetrics has no text field (voice_types.h is a
contract header and is not edited for this experiment), so "the user heard an
honest failure" is asserted through what is observable: an answer clause was
synthesised and played (answer_first_played_ms > 0) and the failing task was
dispatched at most once (tool_calls <= 1). A spoken transcript is the missing
signal; this is reported as a contract gap in the harness report.
"""

import argparse
import glob
import json
import math
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "out")
DEFAULT_SCENARIOS = os.path.join(HERE, "scenarios")

# Thresholds that are asserted on the p95 aggregate (spiky-by-nature targets).
P95_KEYS = {"eos_to_first_played_ms_p95", "barge_in_stop_ms"}


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


# ---------------------------------------------------------------------------
# assertions
# ---------------------------------------------------------------------------
def check_expect(name, expect, summary, records, scen_dir=None):
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

    for key in sorted(expect):
        want = expect[key]

        if key == "eos_to_first_played_ms_p95":
            got = p95_of("eos_to_first_played_ms")
            if got is None:
                problems.append(f"{name}: eos_to_first_played_ms missing from summary.json")
            elif got > float(want):
                problems.append(
                    f"{name}: eos_to_first_played_ms p95 {got:.0f} ms > {want} ms")

        elif key == "max_gap_ms":
            got = max_of("max_gap_ms")
            p95 = p95_of("max_gap_ms")
            # Ground truth: measure silence runs in the rendered WAV. The
            # bookkeeping can miss drops/overruns; the audio cannot.
            wav_gap = 0
            if scen_dir:
                for r in measured:
                    run_no = r.get("run")
                    path = os.path.join(scen_dir, "run%s.wav" % run_no)
                    if os.path.exists(path):
                        wav_gap = max(wav_gap, wav_max_silence_run_ms(path))
            if got is None:
                problems.append(f"{name}: max_gap_ms missing from summary.json")
            elif max(got, wav_gap) > float(want):
                problems.append(
                    f"{name}: playback gap self={got:.0f} ms wav={wav_gap:.0f} ms > "
                    f"{want} ms (p95 {p95:.0f} ms)")
            notes.append(f"wav_max_gap={wav_gap}ms")

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
                every(lambda r: int(r.get("tool_calls", 0) or 0) <= 1,
                      "no duplicate task dispatch (tool_calls <= 1)")

        elif key == "max_task_dispatches":
            limit = int(want)
            every(lambda r, limit=limit: int(r.get("tool_calls", 0) or 0) <= limit,
                  f"task dispatched at most {limit}x (tool_calls)")

        else:
            problems.append(f"{name}: unknown expectation key '{key}'")

    return problems, notes


def evaluate(scenarios_dir, out_dir, only=None):
    """Returns (problems, notes, checked_count)."""
    problems, notes = [], []
    paths = sorted(glob.glob(os.path.join(scenarios_dir, "*.json")))
    if only:
        wanted = set(only)
        paths = [p for p in paths if os.path.splitext(os.path.basename(p))[0] in wanted]

    checked = 0
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        scenario = load_json(path)
        scen_out = os.path.join(out_dir, str(scenario.get("name", name)))
        summary_path = os.path.join(scen_out, "summary.json")
        jsonl_path = os.path.join(scen_out, "metrics.jsonl")

        if not os.path.exists(summary_path):
            problems.append(
                f"{name}: missing {summary_path} "
                f"(run: crispasr-voice --harness {os.path.relpath(path)})")
            continue

        summary = load_json(summary_path)
        try:
            records = load_jsonl(jsonl_path)
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: cannot read {jsonl_path}: {exc}")
            continue

        problem_start = len(problems)
        expect = scenario.get("expect") or {}
        if not expect:
            notes.append(f"{name}: no expect block, nothing asserted")

        # Sanity: the harness itself must not have reported failures.
        if not summary.get("ok", True):
            problems.append(f"{name}: harness reported {summary.get('runs_failed')} failed run(s)")
        want_runs = int(scenario.get("runs", summary.get("runs", 0)) or 0)
        got_runs = len(measured_runs(records))
        if want_runs and got_runs != want_runs:
            problems.append(
                f"{name}: {got_runs} measured run(s) recorded, scenario asks for {want_runs}")

        p, n = check_expect(name, expect, summary, records, scen_dir=os.path.join(out_dir, name))
        problems.extend(p)
        notes.extend(n)
        checked += 1
        # Status line reflects everything found for this scenario, including
        # the harness's own failed-run counter above.
        scen_problems = problems[problem_start:]

        eos = summary_metric(summary, "eos_to_first_played_ms", "p95")
        gap = summary_metric(summary, "max_gap_ms", "max")
        fill = summary_metric(summary, "fillers_played", "min")
        status = "FAIL" if scen_problems else "ok"
        print(
            f"  {name:14s} {status:4s} "
            f"eos_p95={'-' if eos is None else f'{eos:.0f}ms'} "
            f"max_gap={'-' if gap is None else f'{gap:.0f}ms'} "
            f"fillers_min={'-' if fill is None else f'{fill:.0f}'} "
            f"runs={got_runs}"
        )
        for note in n:
            print(f"    note: {note}")
        for prob in scen_problems:
            print(f"    {prob}")

    return problems, notes, checked


# ---------------------------------------------------------------------------
# self-test: the gate itself must fail when it should
# ---------------------------------------------------------------------------
def _write_case(root, name, expect, runs, ok=True, runs_failed=0):
    """Writes a scenario + summary + metrics.jsonl pair under root."""
    scen_dir = os.path.join(root, "out", name)
    os.makedirs(scen_dir, exist_ok=True)
    with open(os.path.join(root, "scenarios", f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump({"name": name, "input": "question_en.wav", "runs": len(runs),
                   "warmup": 0, "expect": expect}, f)

    measured = [r for r in runs if not r.get("warmup")]
    metrics = {}
    keys = ["eos_to_first_played_ms", "max_gap_ms", "fillers_played",
            "answer_first_played_ms", "barge_ins", "barge_in_stop_ms", "tool_calls"]
    for key in keys:
        vals = [float(r.get(key, 0) or 0) for r in measured]
        if not vals:
            continue
        metrics[key] = {"p50": percentile(vals, 0.50), "p95": percentile(vals, 0.95),
                        "min": min(vals), "max": max(vals), "n": len(vals)}
    summary = {"scenario": name, "runs": len(measured), "warmup": 0,
               "runs_total": len(runs), "runs_failed": runs_failed,
               "ok": ok and runs_failed == 0, "expect": expect, "metrics": metrics}
    with open(os.path.join(scen_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f)
    with open(os.path.join(scen_dir, "metrics.jsonl"), "w", encoding="utf-8") as f:
        for i, r in enumerate(runs, 1):
            rec = dict(r)
            rec.setdefault("run", i)
            rec.setdefault("warmup", False)
            f.write(json.dumps(rec) + "\n")


def self_test():
    root = tempfile.mkdtemp(prefix="voice-check-selftest-")
    os.makedirs(os.path.join(root, "scenarios"), exist_ok=True)

    def base(eos=900, gap=40, fillers=1, answer=9000, barge=1, stop=120, tools=1):
        return {"eos_to_first_played_ms": eos, "max_gap_ms": gap, "fillers_played": fillers,
                "answer_first_played_ms": answer, "barge_ins": barge,
                "barge_in_stop_ms": stop, "tool_calls": tools}

    cases = [
        # (name, expect, per-run records, summarised ok, expect_problems)
        ("pass_short", {"eos_to_first_played_ms_p95": 1800, "answer_played": True},
         [dict(base()), dict(base(eos=1000)), dict(base(eos=1100))], True, False),
        ("fail_short_slow", {"eos_to_first_played_ms_p95": 1800, "answer_played": True},
         [dict(base(eos=1900)), dict(base(eos=2100)), dict(base(eos=2300))], True, True),
        ("fail_no_answer", {"answer_played": True},
         [dict(base()), dict(base(answer=0)), dict(base())], True, True),
        ("pass_delegation", {"max_gap_ms": 150, "min_filler_played": 1, "answer_played": True},
         [dict(base(gap=20, fillers=2)), dict(base(gap=60, fillers=1)),
          dict(base(gap=140, fillers=3))], True, False),
        ("fail_gap", {"max_gap_ms": 150},
         [dict(base(gap=20)), dict(base(gap=420)), dict(base(gap=30))], True, True),
        ("fail_no_filler", {"min_filler_played": 1},
         [dict(base(fillers=2)), dict(base(fillers=0)), dict(base(fillers=1))], True, True),
        ("pass_barge_in", {"barge_in_detected": True, "barge_in_stop_ms": 250},
         [dict(base(stop=90)), dict(base(stop=130)), dict(base(stop=180))], True, False),
        ("fail_no_barge_in", {"barge_in_detected": True},
         [dict(base(barge=1)), dict(base(barge=0)), dict(base(barge=1))], True, True),
        ("fail_slow_barge_in", {"barge_in_stop_ms": 250},
         [dict(base(stop=90)), dict(base(stop=300)), dict(base(stop=310))], True, True),
        ("pass_task_failure", {"honest_failure_text": True, "max_task_dispatches": 1},
         [dict(base(tools=1)), dict(base(tools=1))], True, False),
        ("fail_duplicate_dispatch", {"max_task_dispatches": 1},
         [dict(base(tools=1)), dict(base(tools=2))], True, True),
        ("fail_no_failure_text", {"honest_failure_text": True},
         [dict(base(answer=0))], True, True),
        ("fail_harness_error", {"answer_played": True},
         [dict(base())], False, True),
        ("fail_unknown_key", {"no_such_metric": 1}, [dict(base())], True, True),
    ]
    for name, expect, runs, ok, _ in cases:
        _write_case(root, name, expect, runs, ok=ok, runs_failed=0 if ok else 1)

    # A missing summary must be reported, not crashed on.
    os.makedirs(os.path.join(root, "scenarios"), exist_ok=True)
    with open(os.path.join(root, "scenarios", "missing.json"), "w", encoding="utf-8") as f:
        json.dump({"name": "missing", "input": "question_en.wav", "runs": 1,
                   "expect": {"answer_played": True}}, f)

    failures = 0
    print(f"self-test in {root}")
    for name, expect, runs, ok, want_problem in cases:
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

# ---------------------------------------------------------------------------
# Ground truth from the rendered WAV: the metrics can lie, the audio cannot.
# ---------------------------------------------------------------------------
def wav_max_silence_run_ms(path, frame_ms=20, rms_threshold=0.01):
    import wave, struct, math
    with wave.open(path) as w:
        rate = w.getframerate()
        frames = w.readframes(w.getnframes())
    n = len(frames) // 2
    samples = struct.unpack("<%dh" % n, frames)
    frame = max(1, int(rate * frame_ms / 1000))
    runs, run_start, started = [], None, False
    for i in range(0, len(samples) - frame + 1, frame):
        seg = samples[i:i + frame]
        rms = math.sqrt(sum(x * x for x in seg) / len(seg)) / 32768.0
        loud = rms > rms_threshold
        if loud:
            if run_start is not None:
                runs.append((i - run_start) * 1000 // rate)
                run_start = None
            started = True
        elif started and run_start is None:
            run_start = i
    if run_start is not None:
        runs.append((len(samples) - run_start) * 1000 // rate)
    return max(runs) if runs else 0


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
