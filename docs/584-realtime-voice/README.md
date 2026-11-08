# 584 — realtime voice loop

One process: streaming ASR -> small local LLM (model-emitted tool call) ->
streaming TTS, with a scheduler that keeps the spoken output continuous while a
slow backend task runs. Classic STT/TTS endpoints and the deployed containers
are untouched.

Branch: `feat/584-realtime-voice` (base v0.8.51 + the fork's synthetic
provenance patch). Tracking issue: home-ops#584.

## Status

- M0 co-residency spike: done (`spike/`).
- M1 loop + barge-in + WAV-truth harness: done, 6/6 scenarios green.
- M2 model tool calls, filler policy, ES, model choice: done.
- M3 speculative ack (`--speculate`, opt-in): done, -320 ms audible.
- Packaging: CMake target + VRAM budget documented; container/transport is next.

Headline numbers (real engines, RX 7900 XT, Vulkan, Qwen3-1.7B chat):

| metric | value |
|---|---|
| end of speech -> first audible | **590 ms p50** with `--speculate` (910 ms without) |
| worst silence inside a 6 s delegation | **100 ms** |
| ASR final (2.6 s utterance) | ~130 ms |
| LLM TTFT / decode | ~30 ms / ~176 tok/s |
| TTS TTFA | 141-184 ms |
| VRAM, voice stack alone | **8.95 GiB** incremental (11.6 GiB total with desktop) |
| VRAM with the deployed STT/TTS containers | 16.4 GiB peak (3.5 GiB headroom) |

## Read in this order

1. `PLAN.md` — design, decisions, scheduler policy, milestones.
2. `RESULTS.md` — measurements, HF comparison, review findings, final numbers.
3. `VRAM.md` — measured GPU budget and what it means for deployment.
4. `FILLERS.md` — 81-cell filler sweep; bank is the default.
5. `TOOLCALLS.md` — tool schema, prompt, parser, validity.
6. `MODEL-BENCH.md` — 0.6B vs 1.7B on EN/ES; 1.7B recommended.
7. `SPECULATION.md` — M3 design and its guard rails.
8. `spike/` — M0 co-residency evidence.

## Layout

```
examples/cli/voice/
  voice_types.*     config, events, metrics (JSONL)
  audio_io.*        PCM ring, WAV reader/writer, paced source/sink
  turn_detector.*   energy VAD + hangover + barge-in
  stt_driver.*      turn buffer, sliding partial window
  engines.h         abstract ASR/TTS/LLM + fakes
  engines_crispasr.* real engines (CrispasrBackend + crispasr_chat_*)
  engines_fake.*    deterministic fakes for tests
  arbiter.*         one inference worker, bounded jobs, priorities
  sequencer.*       pure policy: ack -> filler -> answer, handoff
  tool_call.*       schema, parser, dispatcher
  speculation.*     stable-partial ack preparation
  task_adapter.*    simulated A2A-shaped task
  harness.*         scenario runner, p50/p95, WAV output
  crispasr_voice_loop.cpp  CLI: --selftest | --harness | single run
tests/voice/        fixtures, scenarios, sweep, check_metrics.py
docs/584-realtime-voice/
```

## Build and run

```sh
# CMake target (consolidated)
cmake -S . -B build-vk -G Ninja -DCMAKE_MAKE_PROGRAM=/usr/bin/ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DSPIRV-Headers_DIR=$(brew --prefix spirv-headers)/share/cmake/SPIRV-Headers
ninja -C build-vk bin/crispasr-voice

# fast dev path (links the existing CLI objects)
python3 tools/build_voice_loop.py

# GPU-free policy checks
build-vk/bin/crispasr-voice --selftest

# automated scenarios (fake engines)
build-vk/bin/crispasr-voice --harness tests/voice/scenarios/delegation.json
python3 tests/voice/check_metrics.py
```

Real engines need the ASR/TTS GGUFs from the deployed volume plus a chat GGUF:

```sh
M=~/.local/share/containers/storage/volumes/beast-crispasr_crispasr-models/_data
VOICE_REAL_ENGINES=1 build-vk/bin/crispasr-voice \
  --harness tests/voice/scenarios/delegation_real.json --speculate \
  --asr-model $M/qwen3-asr-1.7b-q8_0.gguf \
  --tts-model $M/qwen3-tts-12hz-1.7b-base-q8_0.gguf \
  --tts-tokenizer $M/qwen3-tts-tokenizer-12hz.gguf \
  --voice-dir ~/repos/home-ops/hosts/beast/crispasr/voice \
  --chat-model ~/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf
```

## Testing philosophy

The rendered WAV is the authority. Metrics can be fooled by model-emitted
silence and mixed time bases; `check_metrics.py` measures silence runs and
end-of-speech -> first-audible latency from the audio itself and refuses to read
a missing file as zero. `--selftest` covers 52 policy/parser/speculation cases
without a GPU.
