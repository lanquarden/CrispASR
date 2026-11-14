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

# live session: microphone in, speaker out, one process, Ctrl-C to stop
# (half-duplex by default: the loop ignores its own voice; --barge-in opts out)
build-vk/bin/crispasr-voice --live --filler-max 1 \
  --asr-model ... --tts-model ... --tts-tokenizer ... \
  --voice-dir ... --chat-model ...

# chat-model measurement (no audio, no ASR/TTS)
build-vk/bin/crispasr-voice --bench-llm      --chat-model M --runs 15
build-vk/bin/crispasr-voice --eval-bridge    --chat-model M --runs 10
build-vk/bin/crispasr-voice --bench-context  --chat-model M --context-file home.txt [--chat-ctx N]
build-vk/bin/crispasr-voice --bench-prefill  --chat-model M --context-file home.txt [--chat-ctx N]
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

## LLM harness measurements (2026-11-18, RX 7900 XT, Vulkan)

Filler phrasing is a prompt problem, not a sampler problem. `--eval-bridge`
scores the raw model output over prompt variants x temperatures (10 samples
each); "clean" means no device noun, no condition word, no digit:

| variant | Qwen3-0.6B clean | words |
|---|---|---|
| bare instruction, transcript in the user turn | **0%** at t=0.0/0.4/0.8 | 5 |
| guidance in the system prompt | 100 / 70 / 70% | 15 |
| **three example phrases in the system prompt** | **100 / 100 / 100%** | 8 |

The bare prompt always invented the answer ("The garage door is open."), so the
old result-claim guard was masking a 100% failure rate rather than catching a
rare slip. Few-shot examples fix it; an explicit ban list measured identically,
so it is not carried. Qwen3-0.6B beats 1.7B on this narrow job (1.7B names the
device in its holding phrase and repeats it as the answer's first sentence).

Static guidance lives in the per-kind system prompt, so it sits in the chat
session's cached prefix (crispasr_chat.h reuses the KV cache when the prompt
prefix matches). Cost of the block, measured by the `guidance` bench case:
TTFT 3 ms p50 / 4 ms p95, identical to a bare 208-char prompt.

## Dynamic context cost (home snapshot)

`--bench-context` sweeps a real snapshot (475 entities, 16,672 chars compact)
behind the cached system prompt. TTFT p50, Qwen3-0.6B:

| block | ~tokens | cold | warm (identical) | fresh snapshot |
|---|---|---|---|---|
| 0 | 0 | 27 ms | 3 ms | 3 ms |
| 1,500 | 375 | 61 | 5 | 42 ms |
| 3,000 | 750 | 89 | 6 | 71 ms |
| 4,000 | 1,000 | 112 | 7 | 97 ms |
| 16,672 (full) | 4,168 | 454 | 17 | 468 ms |

The static prefix is free at any size. The dynamic part costs ~28 ms per
1,000 chars of *changed* context (~0.11 ms/token), because a real snapshot
differs throughout and the cache breaks at the first difference. The default
ctx 2048 fits ~4,000 chars (entity_ids tokenize badly, so chars/4 understates
the token count).

## Prefill pipelining

The snapshot does not depend on the STT result, so it can be decoded at speech
onset with `crispasr_chat`'s `prefill_only` (`LlmEngine::prefill`). The number
the user feels is the later `generate()`:

| snapshot | prefill (off-path) | baseline TTFT | prefetched TTFT |
|---|---|---|---|
| 0 chars | 4 ms | 24 ms | 27 ms (call floor) |
| 3,000 | 71 ms | 88 ms | **23 ms** |
| 6,500 | 160 ms | 174 ms | **24 ms** |
| 16,672 (full) | 419 ms | 457 ms | **44 ms** |

Visible TTFT flatlines at the ~25 ms call floor regardless of context size.
Under ~1.5 KB there is nothing to save. Prefilling with a *partial* transcript
that differs from the final one costs nothing extra, so the prefix can be
extended as partials arrive. The prefill still consumes GPU time and must yield
to ASR partials, so it belongs at low arbiter priority.

Curating the snapshot is deliberately not the loop's job: the service takes a
generic context block and the caller decides what goes in it.
