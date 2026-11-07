# 584 — Realtime voice loop in CrispASR

Tracking: home-ops#584 (parent voice workstream #278). Branch:
`feat/584-realtime-voice`, based on **v0.8.51** (latest release) with the
fork-local synthetic-provenance patch cherry-picked on top. The deployed
containers stay on 0.8.40; the experiment is upstream-clean so the branch can
become a PR.

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

Priorities are only meaningful at job boundaries — no inference call is
preemptible. Two hard rules follow (review finding #2):

- Every job class is bounded. ASR partials decode at most the last 6 s. TTS
  input is committed in pieces of <= ~6 words (~0.4 s of audio) so one
  `synthesize_streaming` call is a few hundred ms, not seconds of audio.
- The arbiter records `block_ms` per job; M1 fails if any TTS job blocks longer
  than 400 ms or any partial longer than 250 ms.

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
`played_ms > 0` and the assistant owns the floor. Additions required by review:

- `DELEGATING --SPEECH_START--> LISTENING` — a follow-up question spoken while
  the task runs takes the floor; the old task keeps running and its result is
  dropped if the new turn does not reference it (explicit rule, tested).
- `TASK_RESULT` arriving after its turn was abandoned is dropped and counted
  (`stale_results_dropped`), never spoken into a later conversation.
- `answer` entries carry `expiry_ms`; an expired answer is dropped, not spoken
  late.

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

Handoff (revised after review #4): answer arrival does **not** stop filler
immediately — the answer still needs LLM TTFT plus TTS TTFA to produce its
first PCM. Filler keeps flowing until the answer's first clause is buffered to
`target_water_ms`; only then queued fillers are dropped. The answer prompt is
conditioned on the filler text actually spoken, so the transition reads as
continuation, not as a restart. Barge-in flushes PCM, aborts the LLM generation
via `crispasr_chat_set_abort_callback`, lets the in-flight <=0.4 s TTS piece
finish (the vtable has no abort flag; adding one is a separate small upstream
PR), and cancels the task only when the turn is abandoned.

## Partial transcripts

Qwen3-ASR does not implement native cross-append streaming
(`transcribe_streaming` falls back to one final result), so the STT driver owns
partials: while speech is active it re-transcribes a **sliding last-6 s window**
every `stt_partial_ms`. Review #5 is right that a growing buffer is quadratic
and that re-decoding revises words — so partials are advisory only, never a
diff base: the driver emits the latest text and a monotonic timestamp, and the
speculator may only use words that have been stable across two consecutive
decodes. Only the final transcript may trigger a tool call or an ack. Cadence
is gated by measured `asr_partial_cost_ms` (skip a tick rather than queue).

## Stage budget (M0 must fill this in before M1 starts)

| stage | measured by | budget (p50) |
|---|---|---|
| end of speech -> final transcript | harness | <= 200 ms |
| final -> ack text (LLM TTFT) | harness | <= 150 ms |
| ack text -> first PCM (TTS TTFA) | M0 spike | **to be measured** |
| PCM -> first played frame | harness | <= 30 ms |

Review #1 blocks the headline target until the TTS TTFA is real: qwen3-tts
reports ~1 s time-to-first-audio in the backend comment, which would make
`eos_to_first_played <= 700 ms` impossible. M0 measures TTFA at the real
chunk size. The M1 target is then **the measured sum + 15 %**, published in
this file together with the measurement, not a number invented in advance.
If TTFA exceeds ~400 ms, the ack is shortened to its first clause and the
measured budget becomes the target.

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

**M0 — plan + co-residency spike + stage budget.** One process: ASR + TTS
backends + chat session, sequential and (probe only) two-thread. Report
VRAM/RSS, per-call times, **TTS TTFA at the real chunk size**, ASR final decode
for a 3 s utterance, chat TTFT. Fill the stage-budget table. Verdict on the
arbiter design. Exit: spike log committed under `docs/584-realtime-voice/spike/`.

**M1 — the loop, canned filler.** Harness scenarios 1–4 pass; barge-in stops
playback; metrics emitted. Targets: the **measured** stage budget from M0
(headline p50/p95 restated from it); no gap > 150 ms during `delegation`;
barge-in stop <= 250 ms; no stage overruns real time.

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
- **TTS inter-sentence pad**: review #9 is correct — the 200 ms pad lives in the
  server's chunking path, which the loop does not use. The sequencer owns
  inter-clause gaps; no server change is needed.
- **Echo**: with a real mic the loop hears its own TTS and fires `BARGE_IN`
  continuously. Mitigation: gate VAD while the assistant holds the floor and no
  user speech is plausible, plus a self-echo scenario in the harness.
- **Harness honesty**: WAV sink has no device latency. Warm-up runs are
  excluded, N >= 5 per scenario for p50/p95, a stage that overruns real time
  fails the scenario instead of silently compressing the timeline, and one
  scenario runs through a paced sink with an artificial device buffer.
- **qwen3-tts TTFA** is chunked (~1 s) → verify actual first-chunk latency in
  M0; if it exceeds the ack target, the ack clause gets shortened and the
  target is restated per-stage.
- **Voicepack policy**: jarvis is classified as synthetic on this fork; the
  upstream build would require consent + spoken disclosure. Do not build the
  experiment on the upstream consent path.
- **Thermals**: 20 GB card also drives the desktop; measure, don't assume.
- **Upstream debt** (review #10): the loop needs only two shared-surface
  changes — an abort flag on `synthesize_streaming` (barge-in mid-piece) and a
  grammar hook for chat tool calls. Both are small, separate PRs to the
  upstream vtable/ABI; do not bundle them into this experiment.

## Review

`zai/glm-5.3-flash` reviewed this plan before any code (findings 1-10 above are
incorporated). Re-review at each milestone boundary; the reviewer is prompted
to refute the numbers, not to approve them.

## Reproduction

```
# toolchain (host): cmake 3.31 (<4: the vendored Vulkan shaders reject it),
# ninja, glslc, libvulkan, vulkan-headers, spirv-headers (brew), openblas
# The cached build expects the Vulkan headers under /tmp/sdk/include:
#   /tmp/sdk/include/{vulkan,vk_video,spirv}
cmake -S . -B build-vk -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPIRV-Headers_DIR=$(brew --prefix spirv-headers)/share/cmake/SPIRV-Headers
ninja -C build-vk bin/crispasr-voice
build-vk/bin/crispasr-voice --harness tests/voice/scenarios/delegation.json
```

Base: `v0.8.51` + fork patch `8e09becac` (synthetic-provenance voicepacks).

Models: ASR/TTS from the deployed podman volume, chat LLM from
`~/models/Qwen3-0.6B-GGUF/`.
