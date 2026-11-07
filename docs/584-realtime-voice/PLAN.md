# 584 — Realtime voice loop in CrispASR

Tracking: home-ops#584 (parent voice workstream #278). Branch:
`feat/584-realtime-voice`. Status: **M0 (plan + co-residency spike)**.

Experiment, not a deployment. Judged by measured numbers from an automated
harness, not by ear. Upstream-clean: new files only, no edits to existing
request handlers, no model-specific hacks.

## Problem

The current HA Assist → OpenClaw path buffers every stage, so a turn takes
10–15 s to produce speech. Faster STT/TTS does not fix a sequential buffered
path. The experiment answers one question:

> Can a small local LLM keep a voice pipeline **continuously fed** while a slow
> backend task (6 s, A2A-shaped) runs, so the first audio lands in under a
> second and the output never falls silent?

## Decisions already settled (grilling, 2026-11-01/02)

1. Spike on a branch, new binary, clean module boundary. Ugly internals allowed.
2. Deployable like the current STT/TTS containers; upstream only after go/no-go.
3. Authority is an automated harness (replayable WAVs + scripted timeline).
   Targets live in config, re-run after every change.
4. First audio is an **intent-restating ack** from the final transcript.
   Never a result, never unknown progress.
5. A **scheduler owns the audio clock**; the LLM is a text producer behind it.
   Filler starts as a curated bank, LLM filler swaps in behind the same
   interface.
6. Handoff at clause boundaries (cap ~2 s). Never cut mid-word except barge-in.
7. Barge-in is a first-class event in M1: stop playback, cancel the utterance.
8. Tool calls are **simulated** (4–8 s, failure, cancel) behind an adapter.
9. Engines pinned: Qwen3-ASR-1.7B Q8_0 + Qwen3-TTS-1.7B Q8_0 (jarvis
   voicepack). Only the LLM varies.
10. EN first, ES smoke each milestone. GLM reviews architecture and diffs.

## Non-goals

- HA Assist / Wyoming realtime transport (M2+ deployment work; HA buffers).
- Real OpenClaw A2A, auth, task correlation, retention semantics.
- Choosing the production LLM. The bench informs it; it does not decide.
- Changing existing `/v1/*` endpoints or the deployed containers.

## Architecture

```
 mic/WAV ──► Capture ──► VAD/turn ──► STT driver ──► Transcript events
              │            │              (rolling re-transcribe)
              │            └──► BARGE_IN ──► sequencer.flush()
              ▼
        InferenceArbiter  (one thread, priority queue, owns all ggml calls)
          ├── TTS chunk  (highest; deadline = playback buffer low-water)
          ├── LLM clause (ack / filler / answer)
          └── ASR partial(lowest; droppable when stale)
              │
        SpeechSequencer ──► PCM ring ──► Playback pacer ──► sink (WAV/ALSA/WS)
              ▲                                   │
              └──────── played-frame callback ◄───┘
```

### Why one inference thread

The server serializes every inference behind `model_mutex`; the ggml Vulkan
device is process-global. Concurrent graphs from three threads are the highest
risk in the whole design. One arbiter thread is deterministic, makes VRAM
predictable and lets us reason about deadlines. Spike M0 verifies whether
sequential calls suffice; if concurrency is proven safe later, the arbiter
becomes a pool without changing the interfaces.

Priority rules: a TTS chunk is urgent when `buffer_ms < LOW_WATER`; ASR partials
are dropped if a newer partial exists or speech ended; LLM work is
deadline-free but must never starve TTS.

### Module layout (all new)

```
examples/cli/crispasr_voice_loop.cpp     argument parsing, wiring, main
examples/cli/voice/voice_types.h         events, config, metrics records
examples/cli/voice/audio_io.{h,cpp}      WAV source/sink, ring buffer, pacer
examples/cli/voice/turn_detector.{h,cpp} VAD, speech start/stop, barge-in
examples/cli/voice/stt_driver.{h,cpp}    rolling re-transcription + deltas
examples/cli/voice/llm_channels.{h,cpp}  chat ABI: ack, filler, answer, tools
examples/cli/voice/sequencer.{h,cpp}     scheduler, clause chunking, handoff
examples/cli/voice/task_adapter.{h,cpp}  simulated task, cancel, failure
examples/cli/voice/metrics.{h,cpp}       JSONL metrics + starvation accounting
tests/voice/*                            harness scenarios + assertions
docs/584-realtime-voice/PLAN.md          this file
```

`examples/cli/` is deliberate: TTS streaming is only exposed by the C++
`CrispasrBackend` vtable (`synthesize_streaming`), not the C ABI. The chat ABI
(`include/crispasr_chat.h`) is public and used directly.

Build: new executable `crispasr-voice` + object library so it shares
`crispasr-cli`'s adapter objects (no duplicate compile of ~100 adapters).

## State machine

```
IDLE ──SPEECH_START──► LISTENING ──SPEECH_END──► COMMITTING
                                                   │ FINAL
                                                   ▼
                                            DELEGATING ──TASK_RESULT──► ANSWERING
                                                   │                        │
                                                   └──── BARGE_IN ◄─────────┘
                                                            │
                                                    flush + cancel ──► LISTENING
```

Events: `SPEECH_START`, `PARTIAL`, `SPEECH_END`, `FINAL`, `TASK_RESULT`,
`TASK_FAIL`, `TTS_CHUNK_READY`, `PLAYBACK_LOW`, `BARGE_IN`, `TICK`.

`BARGE_IN` is emitted by the capture path when speech is detected while
`played_ms > 0` and the assistant owns the floor.

## Scheduler policy (defaults, configurable)

| knob | default | meaning |
|---|---|---|
| `sample_rate` | 16000 | mono float32 |
| `frame_ms` | 20 | capture/playback quantum |
| `stt_partial_ms` | 250 | rolling re-transcribe cadence while speech |
| `stt_min_speech_ms` | 250 | do not transcribe shorter blips |
| `low_water_ms` | 1200 | below this, TTS has priority and filler is requested |
| `target_water_ms` | 2000 | desired playback depth |
| `high_water_ms` | 3000 | stop queuing text above this |
| `clause_max_ms` | 2000 | max spoken length of one TTS clause |
| `ack_max_words` | 14 | ack length cap |
| `filler_max_words` | 18 | filler length cap |
| `task_ms` | 6000 | simulated delegation latency (jitter ±800) |
| `barge_in_max_ms` | 200 | speech start → silence |

Text queue entries carry `kind` (ack/filler/answer/clarify), `expiry_ms`, and
`seq`. The sequencer drops stale fillers before they reach TTS. Fillers are
never repeated within a session (bank index + recently-used set).

Handoff: answer arrival marks all pending fillers stale, lets the current
clause finish, then commits the answer. Barge-in flushes PCM, aborts the LLM
generation via `crispasr_chat_set_abort_callback`, and cancels the task only
when the turn is abandoned (cancel is explicit, never implicit on timeout).

## Partial transcripts

Qwen3-ASR does not implement native cross-append streaming
(`transcribe_streaming` falls back to one final result), so the STT driver owns
partials: while speech is active it re-transcribes the accumulated speech buffer
(minus trailing silence) every `stt_partial_ms`, emits the common prefix once
and a delta after that. Partials are advisory; only the final transcript may
trigger a tool call or an ack.

## Metrics (JSONL, one record per turn)

- `eos_to_final_ms` — end of speech → final transcript
- `eos_to_ack_text_ms`, `eos_to_first_pcm_ms`, `eos_to_first_played_ms`
- `task_result_ms`, `result_to_answer_text_ms`, `result_to_first_played_ms`
- `gap_events[]` — `{stage, at_ms, buffer_ms, duration_ms}` when the playback
  buffer hits zero while a turn is active
- `tts_ttfa_ms` per clause, `llm_ttft_ms` per channel, `asr_partial_cost_ms`
- `barge_in_stop_ms`, `dropped_partials`, `stale_fillers_dropped`
- `vram_peak_mb` (from the Vulkan allocator if exposed, else RSS + device query)

## Harness

`crispasr-voice --harness tests/voice/scenarios/<name>.json` replays WAV input
at real time, simulates the task adapter, writes `out.wav`, `metrics.jsonl`,
`timeline.csv`. Scenarios:

1. `short` — 2.5 s question, immediate answer (task 0 ms).
2. `delegation` — 2.5 s question, 6 s task; expects ack + filler + answer.
3. `barge_in` — delegation, user speaks at t=3.0 s.
4. `task_failure` — 6 s then error; expects honest failure, no duplicate action.
5. `es_smoke` — Spanish question, delegation.

Assertions live in `tests/voice/check_metrics.py` (no human in the loop).

## Milestones

**M0 — plan + co-residency spike.** One process: ASR + TTS backends + chat
session, sequential and (probe only) two-thread. Report VRAM/RSS, per-call
times. Verdict on the arbiter design. Exit: spike log committed under
`docs/584-realtime-voice/spike/`.

**M1 — the loop, canned filler.** Harness scenarios 1–4 pass; barge-in stops
playback; metrics emitted. Targets: ack first audio p50 ≤ 700 ms / p95 ≤ 1200 ms;
no gap > 150 ms during `delegation`; barge-in stop ≤ 250 ms.

**M2 — LLM filler + starvation instrumentation.** Filler bank behind the same
interface as LLM filler; sweep low/high water and partial cadence; report table
of gaps and TTFA. No phrase repeats.

**M3 — speculative opening clause + verdict.** `--speculate` on partials with
rollback; measure wrong-ack rate vs latency gain. ES smoke. Final report with
go/no-go recommendation and the deployment shape (third container vs
consolidated).

## Risks

- **Vulkan concurrency / VRAM spikes** → single arbiter; spike measures it.
- **Chat re-prefill cost** between filler and answer channels → measure tokens/s
  and TTFT per channel switch; fall back to a second session sharing weights.
- **TTS inter-sentence 200 ms pad** (server-side chunking) → add a flag, default
  unchanged; the loop sets it to 0.
- **qwen3-tts TTFA** is chunked (~1 s) → verify actual first-chunk latency in
  M0; if it exceeds the ack target, the ack clause gets shortened and the
  target is restated per-stage.
- **Voicepack policy**: jarvis is classified as synthetic on this fork; the
  upstream build would require consent + spoken disclosure. Do not build the
  experiment on the upstream consent path.
- **Thermals**: 20 GB card also drives the desktop; measure, don't assume.

## Reproduction

```
# toolchain (host): cmake>=3.31 (<4 required by the vendored Vulkan shaders),
# ninja, glslc, libvulkan, spirv-headers (brew), openblas
cmake -S . -B build-vk -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPIRV-Headers_DIR=$(brew --prefix spirv-headers)/share/cmake/SPIRV-Headers
ninja -C build-vk bin/crispasr-voice
build-vk/bin/crispasr-voice --harness tests/voice/scenarios/delegation.json
```

Models: ASR/TTS from the deployed podman volume, chat LLM from
`~/models/Qwen3-0.6B-GGUF/`.
