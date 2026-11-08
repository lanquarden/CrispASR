#!/usr/bin/env python3
"""Sweep the voice-loop filler policy over the fake `delegation` scenario.

home-ops#584, M2. The filler mechanism is only interesting if the numbers say
which knobs matter, so this script runs the deterministic harness over a grid
and writes one table:

    filler mode       {bank, llm, bank_then_llm}
    low_water_ms      {800, 1200, 1600}     ("bridge now", TTS gets priority)
    target_water_ms   {1500, 2000, 2500}    (desired playback depth)
    clause_max_words  {6, 20, 40}           (speech chunking)

Per cell: 2 measured runs + 1 warm-up, and for every measured run the WAV gap
(longest silent run inside the output audio), the WAV-derived end-of-speech ->
first-audible latency, the fillers played, the repeated filler phrases, and the
wall-clock runtime of the cell.

Everything the harness cannot see is measured from the audio: the WAV is the
authority, exactly as in tests/voice/check_metrics.py, whose WAV reader is
imported rather than reimplemented.

Outputs:
    tests/voice/out/sweep/summary.md      the table + the best cells
    tests/voice/out/sweep/summary.json    the same data, machine-readable
    tests/voice/out/sweep/cells/<slug>/   one harness output dir per cell
    tests/voice/out/sweep/cells/<slug>.log

Standard library only. The binary is built once under
`flock /tmp/voice-build.lock`; the harness calls run without the GPU lock
because these are the fake engines. `--gpu-lock` exists for a real-engine sweep
so that no measurement ever overlaps another GPU run.

Usage:
    python3 tests/voice/sweep.py                 # full grid, 4 workers
    python3 tests/voice/sweep.py --workers 1     # one cell at a time
    python3 tests/voice/sweep.py --gpu-lock      # real engines: serialize on GPU
    python3 tests/voice/sweep.py --only bank,llm # subset of filler modes
    python3 tests/voice/sweep.py --list          # print the grid and exit
"""

import argparse
import concurrent.futures
import itertools
import json
import os
import shlex
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
BINARY = os.path.join(REPO_ROOT, "build-vk", "bin", "crispasr-voice")
SCENARIO = os.path.join(HERE, "scenarios", "delegation.json")
OUT_ROOT = os.path.join(HERE, "out", "sweep")
BUILD_LOCK = "/tmp/voice-build.lock"
GPU_LOCK = "/tmp/voice-gpu.lock"

FILLER_MODES = ("bank", "llm", "bank_then_llm")
LOW_WATER = (800, 1200, 1600)
TARGET_WATER = (1500, 2000, 2500)
CLAUSE_WORDS = (6, 20, 40)

RUNS = 2
WARMUP = 1

sys.path.insert(0, HERE)
import check_metrics  # noqa: E402  (WAV ground truth lives in the gate)


def slug(mode, low, target, clause):
    return f"f{mode}_l{low}_t{target}_c{clause}"


def cell_grid(only=None):
    cells = []
    for mode, low, target, clause in itertools.product(
            FILLER_MODES, LOW_WATER, TARGET_WATER, CLAUSE_WORDS):
        if only and mode not in only:
            continue
        cells.append({"mode": mode, "low": low, "target": target, "clause": clause,
                      "slug": slug(mode, low, target, clause)})
    return cells


def build():
    """One build for the whole sweep, under the shared build lock."""
    cmd = ["flock", BUILD_LOCK, "-c", "python3 tools/build_voice_loop.py"]
    started = time.time()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return {
        "cmd": " ".join(cmd),
        "ok": res.returncode == 0,
        "seconds": round(time.time() - started, 2),
        "stdout_tail": res.stdout.strip().splitlines()[-3:],
        "stderr_tail": res.stderr.strip().splitlines()[-6:],
    }


def harness_cmd(cell, out_root, gpu_lock, engine_args=None):
    args = [
        BINARY,
        "--harness", SCENARIO,
        "--out", out_root,
        "--filler", cell["mode"],
        "--low-water-ms", str(cell["low"]),
        "--target-water-ms", str(cell["target"]),
        "--clause-max-words", str(cell["clause"]),
        "--runs", str(RUNS),
        "--warmup", str(WARMUP),
    ]
    if engine_args:
        args.extend(engine_args)
    if gpu_lock:
        # Real engines still need VOICE_REAL_ENGINES=1 and their model flags;
        # --engine-args carries those. What the lock guarantees is that no cell
        # overlaps another GPU user (a build or another real run).
        return ["flock", GPU_LOCK, "-c", " ".join(shlex.quote(a) for a in args)]
    return args


def run_cell(cell, gpu_lock=False, engine_args=None):
    """Runs one grid cell and returns its measurement record."""
    # One output root per cell: the harness writes <out>/<scenario>, and two
    # cells sharing it would overwrite each other's WAVs and metrics.
    out_root = os.path.join(OUT_ROOT, "cells", cell["slug"])
    log_path = os.path.join(OUT_ROOT, "cells", cell["slug"] + ".log")
    os.makedirs(out_root, exist_ok=True)
    cmd = harness_cmd(cell, out_root, gpu_lock, engine_args)
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        res = subprocess.run(cmd, cwd=REPO_ROOT, stdout=log,
                             stderr=subprocess.STDOUT, text=True)
    runtime = time.time() - started

    rec = dict(cell)
    rec["runtime_s"] = round(runtime, 1)
    rec["exit"] = res.returncode
    rec["log"] = os.path.relpath(log_path, REPO_ROOT)
    if res.returncode != 0:
        rec["status"] = "failed"
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            tail = f.read().strip().splitlines()[-6:]
        rec["log_tail"] = tail
        return rec

    scen_dir = os.path.join(out_root, "delegation")
    try:
        summary = check_metrics.load_json(os.path.join(scen_dir, "summary.json"))
        records = check_metrics.load_jsonl(os.path.join(scen_dir, "metrics.jsonl"))
    except (OSError, ValueError) as exc:
        rec["status"] = "failed"
        rec["log_tail"] = [f"cannot read cell output: {exc}"]
        return rec

    measured = [r for r in records if not r.get("warmup")]
    if not measured:
        rec["status"] = "failed"
        rec["log_tail"] = ["no measured runs recorded"]
        return rec

    # --- audio ground truth: the WAV, not the bookkeeping -------------------
    input_path = check_metrics.resolve_summary_path(summary.get("input_wav"), scen_dir)
    try:
        input_profile = check_metrics.wav_profile(input_path) if input_path else None
    except (OSError, ValueError) as exc:
        rec["status"] = "failed"
        rec["log_tail"] = [f"cannot read input wav: {exc}"]
        return rec
    wav_gaps, audible = [], []
    for run in measured:
        path = os.path.join(scen_dir, "run%s.wav" % run.get("run"))
        try:
            profile = check_metrics.wav_profile(path)
        except (OSError, ValueError) as exc:
            rec["status"] = "failed"
            rec["log_tail"] = [f"cannot read {path}: {exc}"]
            return rec
        wav_gaps.append(profile["max_silence_run_ms"])
        if (profile["first_audible_ms"] is not None and input_profile
                and input_profile["last_audible_ms"] is not None):
            audible.append(profile["first_audible_ms"] - input_profile["last_audible_ms"])

    # --- repeats: within one run (a run is one session) ---------------------
    repeats = []
    per_run_fillers = []
    try:
        with open(os.path.join(scen_dir, "timeline.csv"), "r", encoding="utf-8") as f:
            import csv
            rows = [r for r in csv.DictReader(f)
                    if r.get("stage") == "tts" and r.get("kind") == "filler"]
    except OSError:
        rows = []
    by_run = {}
    for row in rows:
        by_run.setdefault(row.get("run"), []).append(row.get("text", ""))
    for run_no, texts in sorted(by_run.items()):
        seen, dup = set(), []
        for text in texts:
            key = text.strip().lower()
            if key in seen:
                dup.append(text)
            seen.add(key)
        repeats.extend(dup)
        per_run_fillers.append(len(texts))

    def metric(key, stat="p50"):
        entry = (summary.get("metrics") or {}).get(key) or {}
        return entry.get(stat)

    rec.update({
        "status": "ok",
        "wav_gap_ms": max(wav_gaps),
        "audible_p50_ms": statistics.median(audible) if audible else None,
        "audible_p95_ms": check_metrics.percentile(audible, 0.95) if audible else None,
        "fillers_min": min(per_run_fillers) if per_run_fillers else 0,
        "fillers_max": max(per_run_fillers) if per_run_fillers else 0,
        "repeats": len(repeats),
        "repeat_texts": repeats[:6],
        "self_gap_max_ms": metric("max_gap_ms", "max"),
        "result_to_answer_p50_ms": metric("result_to_answer_played_ms", "p50"),
        "answer_first_played_p50_ms": metric("answer_first_played_ms", "p50"),
        "eos_to_first_played_p50_ms": metric("eos_to_first_played_ms", "p50"),
        "tts_pieces_p50": metric("tts_pieces", "p50"),
        "llm_calls_p50": metric("llm_calls", "p50"),
    })
    return rec


def fmt(value, digits=0, dash="-"):
    if value is None:
        return dash
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def rank_key(rec):
    # Fewest repeats, then the least dead air, then the earliest first audio,
    # then the fastest cell. Repeats come first because "no phrase twice" is a
    # hard policy, not a trade-off. `0` is a real measurement here, so missing
    # values are tested with `is None`, not with truthiness.
    gap = rec.get("wav_gap_ms")
    audible = rec.get("audible_p50_ms")
    return (rec.get("repeats", 1),
            gap if gap is not None else 1e9,
            audible if audible is not None else 1e9,
            rec.get("runtime_s") if rec.get("runtime_s") is not None else 1e9)


def write_outputs(cells, build_info, started_at):
    ok = [c for c in cells if c["status"] == "ok"]
    failed = [c for c in cells if c["status"] != "ok"]
    ranked = sorted(ok, key=rank_key)
    best = ranked[0] if ranked else None

    meta = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started_at)),
        "scenario": os.path.relpath(SCENARIO, REPO_ROOT),
        "binary": os.path.relpath(BINARY, REPO_ROOT),
        "runs_per_cell": RUNS,
        "warmup_per_cell": WARMUP,
        "grid": {"filler": list(FILLER_MODES), "low_water_ms": list(LOW_WATER),
                 "target_water_ms": list(TARGET_WATER),
                 "clause_max_words": list(CLAUSE_WORDS)},
        "build": build_info,
        "engine": "fake",
        "cells_ok": len(ok),
        "cells_failed": len(failed),
        "duration_s": round(time.time() - started_at, 1),
    }
    with open(os.path.join(OUT_ROOT, "summary.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "cells": cells}, f, indent=2)
        f.write("\n")

    def table(rows):
        out = [
            "| filler | low_water | target_water | clause_max_words | WAV gap | audible p50 | audible p95 | fillers (min-max) | repeats | answer after result | runtime |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for r in rows:
            out.append(
                f"| {r['mode']} | {r['low']} | {r['target']} | {r['clause']} | "
                f"{fmt(r.get('wav_gap_ms'))} ms | {fmt(r.get('audible_p50_ms'))} ms | "
                f"{fmt(r.get('audible_p95_ms'))} ms | "
                f"{fmt(r.get('fillers_min'))}-{fmt(r.get('fillers_max'))} | "
                f"{fmt(r.get('repeats'))} | "
                f"{fmt(r.get('result_to_answer_p50_ms'))} ms | "
                f"{fmt(r.get('runtime_s'), 1)} s |")
        return out

    lines = []
    lines.append("# M2 filler sweep — `delegation`, fake engines")
    lines.append("")
    lines.append(f"Generated {meta['generated_at']} by `python3 tests/voice/sweep.py`.")
    lines.append(f"Scenario `{meta['scenario']}` · {RUNS} measured runs + {WARMUP} warm-up per cell · "
                 f"{len(cells)} cells ({len(ok)} ok, {len(failed)} skipped) in "
                 f"{meta['duration_s']:.0f} s · build {build_info['seconds']:.1f} s "
                 f"({'ok' if build_info['ok'] else 'FAILED'}).")
    lines.append("")
    lines.append("WAV gap = longest silent run inside the rendered output audio (20 ms frames, "
                 "RMS <= 0.01). Audible = first audible sample of the output minus the last "
                 "audible sample of the input WAV, so it is end-of-speech -> first sound, not a "
                 "buffer timestamp. Fillers = TTS pieces of kind `filler` per run. Repeats = "
                 "filler phrases spoken twice inside one run. Answer after result = task result "
                 "-> answer's first played PCM: the cost of bridging too long.")
    lines.append("")
    if best:
        lines.append("Best cell: **"
                     f"{best['mode']} / low {best['low']} / target {best['target']} / "
                     f"clause {best['clause']}** — WAV gap {fmt(best.get('wav_gap_ms'))} ms, "
                     f"audible p50 {fmt(best.get('audible_p50_ms'))} ms, "
                     f"{fmt(best.get('fillers_min'))}-{fmt(best.get('fillers_max'))} fillers, "
                     f"{fmt(best.get('repeats'))} repeats.")
        lines.append("")
    lines.append("## All cells")
    lines.append("")
    lines.extend(table(ok))
    if failed:
        lines.append("")
        lines.append("## Skipped cells")
        lines.append("")
        lines.append("| cell | runtime | last log lines |")
        lines.append("|---|---|---|")
        for r in failed:
            tail = " / ".join(x.strip() for x in (r.get("log_tail") or [])[-2:])
            lines.append(f"| {r['slug']} | {fmt(r.get('runtime_s'), 1)} s | {tail[:240]} |")
    lines.append("")
    lines.append("## Best cells")
    lines.append("")
    lines.extend(table(ranked[:3]))
    lines.append("")
    with open(os.path.join(OUT_ROOT, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return meta, ok, failed, ranked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workers", type=int, default=4,
                    help="cells run concurrently (each is a lightly loaded process)")
    ap.add_argument("--gpu-lock", action="store_true",
                    help="wrap every harness call in flock /tmp/voice-gpu.lock "
                         "(real engines; never overlaps another GPU run)")
    ap.add_argument("--engine-args", default="",
                    help="extra flags appended to every harness call, e.g. "
                         "'--asr-model ... --tts-model ...' for --gpu-lock runs")
    ap.add_argument("--only", default="", help="comma-separated subset of filler modes")
    ap.add_argument("--list", action="store_true", help="print the grid and exit")
    ap.add_argument("--no-build", action="store_true", help="reuse the existing binary")
    args = ap.parse_args(argv)

    only = [m.strip() for m in args.only.split(",") if m.strip()]
    cells = cell_grid(only)
    if args.list:
        for c in cells:
            print(c["slug"])
        print(f"{len(cells)} cell(s); {RUNS}+{WARMUP} runs each")
        return 0

    os.makedirs(OUT_ROOT, exist_ok=True)
    started = time.time()

    build_info = {"cmd": "(skipped)", "ok": True, "seconds": 0.0,
                  "stdout_tail": [], "stderr_tail": []}
    if not args.no_build:
        print(f"sweep: building once under {BUILD_LOCK} ...", flush=True)
        build_info = build()
        print(f"sweep: build {'ok' if build_info['ok'] else 'FAILED'} "
              f"in {build_info['seconds']:.1f} s", flush=True)
        if not build_info["ok"]:
            print("\n".join(build_info["stderr_tail"]), file=sys.stderr)
            return 2
    if not os.path.exists(BINARY):
        print(f"sweep: {BINARY} not found (build first)", file=sys.stderr)
        return 2

    print(f"sweep: {len(cells)} cells, {args.workers} worker(s), "
          f"{RUNS} measured + {WARMUP} warm-up runs each", flush=True)

    results = []
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        engine_args = shlex.split(args.engine_args) if args.engine_args else None
        futures = {pool.submit(run_cell, c, args.gpu_lock, engine_args): c
                   for c in cells}
        for fut in concurrent.futures.as_completed(futures):
            cell = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001 - a crashed cell must not kill the sweep
                rec = dict(cell, status="failed", log_tail=[repr(exc)], runtime_s=None)
            results.append(rec)
            done += 1
            if rec["status"] == "ok":
                print(f"  [{done:3d}/{len(cells)}] {rec['slug']:32s} "
                      f"wav_gap={fmt(rec['wav_gap_ms']):>4s} ms "
                      f"audible_p50={fmt(rec['audible_p50_ms']):>4s} ms "
                      f"fillers={rec['fillers_min']}-{rec['fillers_max']} "
                      f"repeats={rec['repeats']} "
                      f"answer+{fmt(rec['result_to_answer_p50_ms']):>4s} ms "
                      f"({fmt(rec['runtime_s'], 1)} s)", flush=True)
            else:
                print(f"  [{done:3d}/{len(cells)}] {rec['slug']:32s} SKIPPED "
                      f"({rec.get('exit')}): {(rec.get('log_tail') or [''])[-1][:90]}",
                      flush=True)

    order = {c["slug"]: i for i, c in enumerate(cells)}
    results.sort(key=lambda r: order.get(r["slug"], 0))
    meta, ok, failed, ranked = write_outputs(results, build_info, started)

    print()
    print(f"sweep: {len(ok)} ok, {len(failed)} skipped, "
          f"{meta['duration_s']:.0f} s total")
    for r in ranked[:3]:
        print(f"  best: {r['slug']:32s} wav_gap={fmt(r.get('wav_gap_ms'))} ms "
              f"audible_p50={fmt(r.get('audible_p50_ms'))} ms "
              f"fillers={r['fillers_min']}-{r['fillers_max']} repeats={r['repeats']}")
    print(f"sweep: wrote {os.path.relpath(os.path.join(OUT_ROOT, 'summary.md'), REPO_ROOT)} "
          f"and summary.json")
    # Skipped cells are reported, not fatal: one unusable cell must not throw
    # away the other 80 measurements.
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
