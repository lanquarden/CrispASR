# 584 — Realtime voice loop: M0/M1 results

Date: 2026-11-08. Branch `feat/584-realtime-voice` (base v0.8.51 + fork
synthetic-provenance patch). Hardware: RX 7900 XT (Vulkan/RADV), desktop and the
two deployed CrispASR containers running concurrently.

## Verdict so far

The concept works: **one process** runs Qwen3-ASR-1.7B, Qwen3-TTS-1.7B and a
Qwen3-0.6B chat model on one ggml Vulkan device, and a scheduler keeps the
spoken pipeline fed through a simulated 6 s delegation. First audio lands in
**~0.27 s** after end of speech, well inside the 0.7 s target.

M1 is **green** on all six automated scenarios, including the real-engine
delegation run: the decoded audio's worst silent run is **100 ms** (target
<=300 ms), and the WAV-derived end-of-speech-to-first-audible latency is
**710 ms p50 / 730 ms p95**. The remaining latency is dominated by
end-of-turn detection, not the pipeline (see below).

## M0 — co-residency spike

| stage | cold | warm |
|---|---|---|
| ASR init | 1.2-1.4 s | — |
| ASR transcribe 11 s audio | 1023 ms (RTF 0.093) | 283 ms (RTF 0.026) |
| TTS init | 1.0-1.6 s | — |
| TTS time-to-first-PCM | 869 ms | **141-184 ms** |
| TTS RTF | — | 0.21 (first chunk 640 ms of audio) |
| chat TTFT / decode | 214 ms / 36.5 tok/s | **17.7 ms / 176.5 tok/s** |
| peak RSS | — | 2.5 GB |
| VRAM (all three) | — | 17.3 GB peak, clean teardown |

Concurrency probe: ASR ×5 ∥ TTS ×5 and chat ∥ ASR both completed with correct
output, no crash or hang, wall time ≈ max(serial, parallel). The single-arbiter
design is therefore a *choice*, not a constraint — but the measured workload
does not need a pool.

## Stage budget (measured, replaces the invented target)

| stage | measured p50 | budget |
|---|---|---|
| end of speech -> final transcript (2.6 s speech) | 130 ms | <= 200 ms |
| final -> ack text (LLM TTFT) | 30 ms | <= 150 ms |
| ack text -> first PCM (TTS TTFA) | 141-184 ms | <= 250 ms |
| PCM -> first played frame | ~20 ms | <= 30 ms |
| **end of speech -> first played** | **255-275 ms** | <= 700 ms |

## M1 — the loop (real engines)

Scenario `delegation_real`: real question WAV (TTS-generated), 6 s task.

- transcript: *"The status of the garage door."* (ASR correct)
- ack: *"The status of the garage door. I am checking."*
- answer: *"Garage door status is complete."*
- end of speech -> first played: **251-275 ms**
- fillers: 1-3 bank phrases (two-clause queue depth)
- barge-in: flush + cancel, 5 ms processing stop (harness-injected marker)
- **WAV-measured worst acoustic silence: 100 ms** (2.18 s -> 0.50 s -> 0.36 s
  -> 0.10 s). The residual is one quiet span inside the ack and one piece
  boundary; the audio is in `tests/voice/out/delegation_real/run*.wav`
- self-reported `max_gap_ms`: 0 ms; the WAV check stays the authority
- answer starts ~2.6 s after the task result (bridge audio already queued is
  allowed to drain; the answer itself jumps the TTS queue)

Automated suite (fake engines, deterministic):
`short`, `delegation`, `barge_in`, `task_failure`, `es_smoke` all pass,
including WAV-derived gap checks. `delegation_real` fails on the 2.18 s gap, as
it should.

## What exists

```
examples/cli/voice/
  voice_types.h/.cpp      config, events, metrics (JSONL)
  audio_io.h/.cpp         PCM ring, WAV reader/writer, paced source/sink
  turn_detector.h/.cpp    energy VAD + 120 ms hangover + barge-in
  stt_driver.h/.cpp       turn buffer, sliding partial window
  engines.h               abstract ASR/TTS/LLM + fakes
  engines_crispasr.cpp    real engines (CrispasrBackend + crispasr_chat_*)
  engines_fake.cpp        deterministic fakes for the harness
  arbiter.h/.cpp          one inference worker, bounded jobs, priorities
  sequencer.h/.cpp        pure policy: ack -> filler -> answer, handoff
  task_adapter.h/.cpp     simulated A2A-shaped task (latency, failure, cancel)
  harness.h/.cpp          scenario runner, p50/p95, WAV output
  crispasr_voice_loop.cpp CLI: --selftest | --harness | single run
tests/voice/              fixtures (synthetic + one real TTS question),
                          scenarios, check_metrics.py (asserts + WAV silence)
tools/build_voice_loop.py builds against build-vk's existing objects
```

`--selftest` passes 10/10 policy checks without a GPU.

## Reproduce

```sh
# fake engines, ~2 s per scenario
build-vk/bin/crispasr-voice --harness tests/voice/scenarios/delegation.json
python3 tests/voice/check_metrics.py

# real engines
M=~/.local/share/containers/storage/volumes/beast-crispasr_crispasr-models/_data
VOICE_REAL_ENGINES=1 build-vk/bin/crispasr-voice \
  --harness tests/voice/scenarios/delegation_real.json \
  --asr-model $M/qwen3-asr-1.7b-q8_0.gguf \
  --tts-model $M/qwen3-tts-12hz-1.7b-base-q8_0.gguf \
  --tts-tokenizer $M/qwen3-tts-tokenizer-12hz.gguf \
  --voice-dir ~/repos/home-ops/hosts/beast/crispasr/voice \
  --chat-model ~/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
```

## Known defects (open, with the next step)

1. **500 ms silent run in `delegation_real`** (down from 2180 ms). Root cause
   was model-padded silence around every TTS piece (0.42 s lead, 0.20 s tail on
   the clone voice) plus characters the model renders as pauses (quotes,
   markdown). Both are now trimmed/sanitised; what remains is the gap between a
   filler ending and the answer's first PCM. M2: per-piece trace + watermark
   sweep.
2. **Gap accounting under-reports** (10 ms reported vs 500 ms measured). The
   sink only sees ring underflow; the model emits silence inside pieces, which
   is why the WAV check exists. The sequencer's own number is diagnostic only.
3. **Ack phrasing leaks the third person** in some samples ("The status of the
   garage door is currently unknown") — prompt and word cap are in; needs a
   couple of prompt/decoding iterations.
4. **No model tool call yet.** The task is dispatched by the loop, not by a
   grammar-constrained LLM call. That is the next upstream-able piece.
5. **Deployment**: still a manual binary against `build-vk`, not a CMake target
   or container. Deferred until the go/no-go, per plan.
6. **ES**: fake scenario passes; no real-engine Spanish run yet.

## Next (M2)

- Fix the gap and the accounting; re-run `delegation_real` until the WAV gap is
  under 300 ms.
- Add per-piece TTS/LLM trace to `timeline.csv` (text, start, first PCM,
  duration) so a gap is attributable without guessing.
- Sweep `low/target/high` watermarks and `clause_max_words`; report the table.
- LLM-generated filler behind the existing bank interface (`--filler llm`).
- Real-engine `barge_in` and `short` runs; ES smoke.


## Comparison with huggingface/speech-to-speech

Read at commit `327a8c1` (`/tmp/hf-s2s`). What they do that is relevant:

- **Sentence batching before TTS** (`LLM/language_model.py:194`,
  `stream_batch_sentences=3`): fewer TTS piece boundaries.
- **Per-piece silence trimming** (Kokoro: 250 ms leading, 120-sample pad;
  Phonon: a streaming `_TrailingSilenceTrimmer` that holds quiet frames and
  restores them if speech resumes).
- **Sample-anchored latency origin** (`VAD/vad_handler.py:838-843`):
  `speech_end_sample` from the sample count, not from when VAD declared the
  end, so queue delay does not leak into the metric.
- **Generation-counter cancel** (`pipeline/cancel_scope.py`) plus queue flush on
  `SpeechStartedEvent` (`websocket_router.py:1002-1022`). Equivalent to our
  epoch counter.
- **Played-sample accounting** (`demo/worklets/audio-playback.js`): frames
  actually written plus `queuedMs`, and a sample-accurate interruption test in
  `node:vm`.
- **Testing**: fake `perf_counter`, a `RecordingTransport`, and protocol
  conformance asserted through the OpenAI SDK's own event union. No GPU and no
  human in the loop.

What they admit they do not solve, and we must not copy:

- `docs/response-latency.md`: E2E ends at the first *yielded TTS block*, "not
  browser playback"; no p50/p95; underruns visible only at the end of a turn.
- No filler/glue phrases, no speculative text generation, no A2A. Their
  speculation is turn segmentation only.

Port decisions (after adversarial review):

| idea | verdict | reason |
|---|---|---|
| audible-origin headline + p50/p95 | **port** | our 251-275 ms is buffer-side; WAV-derived audible is the real number |
| fewer TTS pieces (sentence-based clauses) | **adapt** | batching calls 3x longer on a serialized arbiter can enlarge the worst gap; stop splitting at 6 words first |
| hold-and-restore tail trimmer | drop | our pieces are complete at enqueue; restoring held silence can reinsert noise |
| sample-anchored gap metric as a second source | drop | two thresholds drift from the WAV check; keep one truth |
| duck-don't-mute | drop | no measured echo events; HF ships real AEC3 for a reason |
| fake clock for scenario tests | adapt | unit tests only; the failing defect is wall-clock native |

## Gap attribution (trace, real engines)

The 360-1090 ms silent runs inside `delegation_real` are **emitted silence, not
ring starvation**: at t=4.0-5.2 s the playback ring held 2.0-3.0 s of audio while
the WAV was silent. Per-piece trace shows why: a filler piece is ~1.5 s long but
only 0.5-1.2 s audible, and the old trim kept only the last 150 ms of tail. The
remaining work is a stream-level silence gate (keep natural pauses <=220 ms, drop
the rest), not scheduling.

Piece boundaries are also fewer now: answer/filler text is split at sentence
boundaries instead of every 6 words, and the answer jumps the TTS queue on task
result. Worst WAV run: 500 ms -> 360 ms before the gate.

## Testing compared

HF runs 98 pytest modules with fake `perf_counter`, a `RecordingTransport`, a
`node:vm` audio-worklet simulator, and OpenAI-schema conformance through the
SDK's own event union — no GPU, no human. We have the same property with a
different mechanism: deterministic fake engines, a scenario harness, a
GPU-free `--selftest` (10 policy assertions), and a WAV-derived silence/latency
check that the loop cannot fake. What we still lack from their playbook: a
fake-clock mode for unit tests (their tests are instant; ours spend real time),
and a protocol-conformance test once there is an external protocol to conform to.

## Measurement bases (found while porting the audible metric)

Two clocks were being mixed:

- `metrics.jsonl` timestamps are relative to **VAD speech start**; the rendered
  WAV starts at the loop's `t0` (before the input lead-in), so a constant
  ~0.6 s offset separates them.
- `speech_end_ms` is when **VAD declared** the end (after `min_silence_ms`),
  ~300-900 ms after the last audible input sample.

Consequences: `eos_to_first_played` is measured from a late end reference and
therefore understates latency; the "silence after ack_first_played" column in
the first checker report was a mixed-base artifact (~573 ms ≈ the speech-start
offset), not audio. The WAV-derived **`eos_to_first_audible_ms`** (first
10 ms RMS>0.01 in the output WAV minus the last audible frame of the input WAV)
is the only number comparable across runs, and it is now the headline.


## Endpointing dominates the audible number

The pipeline (VAD-declared end -> first audible) is ~250-290 ms. The rest of the
710 ms is the end-of-turn wait. Measured on the same scenario by varying
`--vad-min-silence-ms`:

| end-of-turn silence | WAV audible p50 | worst WAV gap |
|---|---|---|
| 500 ms (old) | 1050 ms | 100 ms |
| 250 ms | 830 ms | 100 ms |
| 150 ms | 710 ms | 100 ms |
| **250 ms (new default, after review)** | **810 ms p50 / 830 ms p95** | **120 ms** |
| 150 ms, before the silence gate | — | 360 ms |

This mirrors HF's choice (Silero `min_silence_ms=64` plus a smart-turn model).
We tried 150 ms and it passes the fixtures, but a final adversarial review found
that an energy VAD with no smart-turn can commit a fragment at a 300-500 ms
inter-phrase pause ("what's the ... weather"). 150 ms is therefore not the
default: **250 ms** is, which keeps the audible number around 830 ms. Going
lower is the speculative-ack work (M3), not a tuning exercise.

The silence gate plus the 40 ms speech-confirmation lookahead adds a fixed
~40 ms before a piece's first emitted sample; that is inside the measured
audible number, not on top of it.

The 120 ms residual is one quiet span inside the ack plus one piece boundary;
the gate drops quiet runs >220 ms and keeps natural pauses <=150 ms verbatim.
The 100 -> 120 ms change after the hold-cap fix is not new dead air: replaying
the same 33 real TTS pieces through both gate revisions gives 21/33 bit-identical
outputs and the other 12 differ only in their last 10-20 ms, where the new gate
keeps soft decay windows (0.006-0.01) it used to truncate.

## Final automated result

```
barge_in         ok   wav_gap=0ms    audible_p50=920ms
delegation       ok   wav_gap=0ms    audible_p50=900ms
delegation_real  ok   wav_gap=120ms  audible_p50=810ms  p95=830ms
es_smoke         ok   wav_gap=0ms    audible_p50=910ms
short            ok   wav_gap=0ms    audible_p50=920ms
task_failure     ok   wav_gap=0ms    audible_p50=900ms
PASS: 6 scenario(s)
```


## Final review of the gate (glm-5.3-flash)

Verdict was fix-before-ship on two defects, both now fixed:

1. **Hold-cap truncation (HIGH).** With 220 ms of held silence, a following
   loud-but-unconfirmed onset was appended and then truncated from the end, and
   `flush()` anchored its 40 ms pad to the front of the hold. A word after an
   internal pause could be clipped or lost. Fix: per-window loud flags, drop
   quiet windows from the front at cap (never the protected trailing loud run),
   anchor the flush pad to the last loud window. Verified with 36/36 gate unit
   cases, including "hold at cap + tone" (full tone) and "20 ms final tail"
   (kept; the old gate emitted none of it).
2. **Short final onsets (LOW/MEDIUM).** `flush()` now treats a trailing raw-loud
   run of >=2 windows (20 ms) as confirmed.
3. **Mid-sentence commits (MEDIUM).** Fixed by restoring the 250 ms default; the
   review's stronger suggestion (require the partial transcript to match the
   final before committing) is M3 work.
4. **Flag clamps (LOW).** `vad_min_speech_ms >= 20`, `vad_min_silence_ms >= 50`.
5. Comment corrected: the 40 ms confirmation clears `in_run_` on the first quiet
   window; the lookahead is what defers classification.


## M2 — model tool calls, filler policy, model choice

**Tool calls (real, Qwen3-0.6B then 1.7B):** the model emits one JSON object;
a strict prompt plus the loop's lenient parser gives **9/9 schema-valid** on the
real scenario and 10/10 on a transcript sweep, 8/10 semantically right. The only
real failure is semantic ("turn on the living room lights" -> `set_timer 5`),
so a grammar sampler would buy nothing today; that stays a later upstream PR.
Dispatch is speculative (default device check at FINAL, corrected/cancelled by a
generation counter when the model's call arrives); a review finding proved a
stale result could win the turn, now fixed by stamping gen/epoch on the event and
re-checking in the handler (`toolcall_stale` proves the drop).

**Filler policy:** 81-cell sweep (filler mode x low/target water x clause size).
Bank is the default: 120 ms WAV gap and 1.37 s result->answer, versus 2.16 s for
LLM filler, which also leaks results ("the door is closed") and repeats less.
`clause_max_words` is inert for answers (they split at sentence boundaries).
Kept: two-clause reserve, session-wide no-repeat, bank fallback, centred
deadline, Spanish bank.

**Model:** `MODEL-BENCH.md` compares 0.6B and 1.7B on 10 EN + 10 ES utterances.
1.7B wins: ES argument accuracy 43 % -> 90 %, code-switching 27 % -> 3 %, JSON
100 %, at +10 ms TTFT. **Use `Qwen3-1.7B-Q8_0`.** Independently, the ack prompt
was asking the model to "restate what you understood", which both models
answered as a result; removing that clause drops result claims to 0/10 EN.

**Real ES:** transcript correct, ack now Spanish and non-interrogative
("Estoy verificando el estado de la puerta del garaje."), WAV gap 100 ms.
The answer can still mix languages because the *simulated task result* is
English; that is a fixture, not the model.

**Caveat:** the M1 gap/latency numbers are not directly comparable to M2 runs —
filler policy, dispatch ordering and model changed. Re-baselined M2 numbers:
`delegation_real` 120-300 ms WAV gap, 810-890 ms audible; `es_real` 100 ms,
1080-1100 ms.
