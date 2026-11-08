# 584 — Realtime voice loop: M0/M1 results

Date: 2026-11-08. Branch `feat/584-realtime-voice` (base v0.8.51 + fork
synthetic-provenance patch). Hardware: RX 7900 XT (Vulkan/RADV), desktop and the
two deployed CrispASR containers running concurrently.

## Verdict so far

The concept works: **one process** runs Qwen3-ASR-1.7B, Qwen3-TTS-1.7B and a
Qwen3-0.6B chat model on one ggml Vulkan device, and a scheduler keeps the
spoken pipeline fed through a simulated 6 s delegation. First audio lands in
**~0.27 s** after end of speech, well inside the 0.7 s target.

M1 is close but not green: after trimming model silence per TTS piece and
sanitising non-speech characters, the rendered audio's worst silent run in the
delegation turn dropped from **2.18 s to 0.50 s** (target 0.15-0.3 s). The
loop's self-reported gap is now 10 ms, so the WAV check is still the authority.

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
- **WAV-measured worst acoustic silence (RMS < 0.01): 500 ms** (was 2180 ms
  before per-piece silence trimming) — above the 300 ms M1 tolerance; the runs
  are scattered at piece boundaries (0.32-0.50 s), so the next step is to
  distinguish real pipeline holes from soft word onsets by listening to
  `tests/voice/out/delegation_real/run*.wav`
- self-reported `max_gap_ms`: 10 ms; the WAV check stays the authority
- answer starts 3.4 s after the task result (bridge is deliberately kept until
  the answer's first PCM exists; this is the M2 tuning knob)

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
