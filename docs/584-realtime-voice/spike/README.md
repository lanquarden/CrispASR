# Spike: ASR + TTS + text-LLM co-residency in one process

Throwaway feasibility spike for #584 (realtime voice loop). It answers three
questions before any realtime design is committed to:

1. Can a streaming-ASR backend and a streaming-TTS backend be alive in **one
   process** at the same time?
2. Can a `crispasr_chat_*` text-LLM session live in that **same process** and
   share the **same ggml/Vulkan device**?
3. Do **sequential** calls work, and what happens when **two threads** run
   inference at once — crash, hang, corruption, or fine?

Files: `spike.cpp` (the program), this README. Nothing is added to the CMake
tree; the program is compiled by hand against the existing `build-vk` objects.

Branch: `feat/584-realtime-voice` rebased on **v0.8.51**. Results below are from
that tree (`crispasr 0.8.51 (git 2193fbc6)`), Vulkan0 = AMD RX 7900 XT (RADV
NAVI31).

## Result: YES on all three

One process, one `ggml_backend` registry, one Vulkan device:

| stage | backend / model | cold (first call) | warm |
|---|---|---|---|
| ASR init (`qwen3-1.7b`, q8_0, 2.4 GB) | `qwen3` | 1204–1435 ms | — |
| ASR transcribe, 11.0 s jfk.wav | `qwen3` | 1023 ms (RTF 0.093) | 275–306 ms (mean 283 ms, RTF 0.026) |
| TTS init (`qwen3-tts-1.7b-base` q8_0 + tokenizer + `jarvis` pack) | `qwen3-tts` | 975–1576 ms | — |
| TTS time-to-first-PCM | `qwen3-tts` | 869 ms | 141–184 ms |
| TTS total synthesis | `qwen3-tts` | 1517 ms | 647–702 ms |
| TTS audio produced | — | 74880 samples @ 24 kHz = **3.12 s** | same |
| Chat model load (`Qwen3-0.6B-Q8_0`, `n_gpu_layers=-1`) | llama.cpp | 370 ms | — |
| Chat TTFT (16-token cap) | llama.cpp | 214 ms | 17.7 ms |
| Chat decode | llama.cpp | 36.5 tok/s (9 tok, cold) | 176.5 tok/s |

- **ASR transcript** (correct, 11 s JFK): *"And so, my fellow Americans, ask not
  what your country can do for you. Ask what you can do for your country."*
- **TTS**: 5 chunks, first chunk **15360 samples = 640 ms** of audio, max chunk
  15360, last 13440; 3.12 s of audio for *"Hello, this is a co-residency spike."*
- **Chat reply**: *"Hello! How can I assist you today?"* — 9 tokens.
- **Peak RSS (VmHWM, whole process, all three models resident)**: **2503 MB**.
- **VRAM in use (rocm-smi, whole card)**: 9481 MB at start (desktop + idle
  containers) → 12114 after ASR → 15084 after TTS → 17319 after chat →
  **back to 9491 MB after teardown** (all three freed cleanly).
- Device evidence from stderr: `crispasr_init_gpu_backend: using preferred GPU
  backend: Vulkan0` twice (ASR adapter + TTS adapter), and llama.cpp's
  `load_tensors: layer 0..27 assigned to device Vulkan0` for the chat model.
  All three ran on the same device in the same process.

Cold vs warm matters: the first inference in a fresh process pays Vulkan
pipeline/shader compilation (ASR 1023 → 283 ms, TTS TTFA 869 → 142 ms). A
realtime loop must warm up both backends before the first user turn.

## Concurrency probe

Two `std::thread`s, `join()`ed under a 90 s watchdog that `_Exit(99)`s:

| probe | shape | result |
|---|---|---|
| A | ASR `transcribe()` ×5 ∥ TTS `synthesize_streaming()` ×5 | **PASS** — 5/5 transcripts identical and correct, 5/5 syntheses produced 74880 samples, wall 4049 ms |
| B | chat `generate_stream` (16 tok) ∥ ASR ×3 | **PASS** — 3/3 transcripts correct, 9 chat tokens, wall 892 ms |

No crash, no hang, no corrupted or truncated output, no watchdog trip (ran
twice). Probe A also shows real overlap rather than lucky interleaving: ASR
summed 1415 ms, TTS summed 4048 ms, wall was 4049 ms — i.e. wall ≈ max of the
two, not the sum, and warm TTS latency did not degrade while ASR ran
(648 ms alone vs 647–649 ms concurrent).

Caveat: this proves *distinct backends, distinct contexts* may run on two
threads at once. It says nothing about two calls into the **same** session —
`crispasr_chat_*` holds a session mutex for a whole generation and `close()`
requires external ordering (see `include/crispasr_chat.h`). A single inference
thread per session/backend is still required; per-caller threads are not.

## Surprises

- **The voicepack needs no consent flag.** `jarvis.gguf` is stamped
  `general.architecture=qwen3tts.voicepack`, `crispasr.voice.source=voicedesign-synthetic`,
  `crispasr.voice.speaker_identity=synthetic`. The fork-local patch classifies
  that as a synthetic non-clone, so the whole-clip CLI run below succeeded
  **without `--i-have-rights`**, and **no spoken AI disclosure was prepended**
  (3.28 s of audio for a ~3.2 s sentence). Verified rather than assumed: that
  WAV transcribed back through `qwen3-1.7b` yields exactly *"Hello. This is a
  co-residency spike."* — no prefix, so no spoken disclosure was inserted. The
  spike bypasses the CLI gates
  entirely (`backend->synthesize_streaming()` takes no consent input); the gate
  lives in `crispasr_run.cpp` / `crispasr_server.cpp` around
  `crispasr_voice::classify()`.
- **The pack's `ref_text` is Spanish** ("A su servicio. El salón a veintiún
  grados…") while it is asked to speak English. It does, but the real loop
  should pin `target_lang` (or ship an English-reference pack) rather than rely
  on this.
- **Streaming vs whole-clip length differ**: streaming produced 74880 samples
  (3.12 s) for the sentence; the CLI's whole-clip `synthesize()` produced 78720
  (3.28 s). 80 ms of trailing silence/tail is not emitted by the streaming path
  (`chunk_frames=8, overlap_frames=96`). Not a bug for a voice loop, but don't
  compare the two numbers as if they were the same call.
- **`crispasr_chat_template_name()` returned the whole Jinja template**, not a
  short name, for `Qwen3-0.6B-Q8_0.gguf` (the model's GGUF carries a full
  `tokenizer.chat_template`). Do not print it in a UI or a log line.
- **`n_gpu_layers = -1` really does offload all 28 layers**, and llama's
  backend init is a `std::call_once` — the chat ABI can be opened after the
  ASR/TTS backends exist with no re-init and no conflict.
- `rocm-smi --showmeminfo vram --json` is a usable VRAM probe on this host
  (whole-card number, so it includes the desktop); `vulkaninfo --summary` is
  not needed.

## Init sequence to wire the real binary

```text
crispasr_set_gpu_backend_pref("vulkan");            // once, before any load

whisper_params asr = {};                            // struct defaults are fine
asr.backend = "qwen3-1.7b"; asr.model = <asr.gguf>;
asr.language = "en"; asr.n_threads = 8; asr.use_gpu = true; asr.gpu_backend = "vulkan";
auto A = crispasr_create_backend(asr.backend);  A->init(asr);   // 1.2-1.4 s

whisper_params tts = {};
tts.backend = "qwen3-tts-1.7b-base"; tts.model = <talker.gguf>;
tts.tts_codec_model = <qwen3-tts-tokenizer-12hz.gguf>;
tts.tts_voice = "jarvis"; tts.tts_voice_dir = <voice dir>;        // bare name -> <dir>/<name>.gguf
tts.language = tts.target_lang = "en"; tts.n_threads = 8;
tts.use_gpu = true; tts.gpu_backend = "vulkan";
auto T = crispasr_create_backend(tts.backend);  T->init(tts);   // 1.0-1.6 s

crispasr_chat_open_params cp; crispasr_chat_open_params_default(&cp);
cp.n_gpu_layers = -1; cp.n_ctx = 2048; cp.n_threads = 8;        // all layers -> Vulkan
auto S = crispasr_chat_open(<chat.gguf>, &cp, &err);             // 0.37 s

// warm: one transcribe(), one short synthesize_streaming(), one 1-token generate
```

Then per turn: `T->synthesize_streaming()` and `A->transcribe()` may run on
their own threads; keep one thread per `crispasr_chat_session_t` and close the
session before the backends. Teardown order used here: chat close → TTS → ASR →
`crispasr_audio_free`; VRAM returned to baseline.

## Reproduce

Requires the `build-vk` build tree (Vulkan, `-DGGML_VULKAN=ON`) already built,
including the `bin/crispasr` CLI target (its objects are the link basis):

```sh
cd /home/lanquarden/repos/CrispASR/build-vk
ninja -j12 bin/crispasr                      # provides examples/cli/CMakeFiles/crispasr-cli.dir/*.o

# 1. compile (flags copied from the crispasr-cli compile line; add examples/cli)
ninja -t commands examples/cli/CMakeFiles/crispasr-cli.dir/crispasr_backend_qwen3.cpp.o | tail -1
#    ... reuse its -D/-I list, then:
c++ <those flags> -O3 -DNDEBUG -fopenmp -std=gnu++17 \
    -o /tmp/voice_spike/spike.o -c \
    /home/lanquarden/repos/CrispASR/docs/584-realtime-voice/spike/spike.cpp

# 2. rename cli.cpp's main so it can be linked (it is the only object excluded
#    from the normal link; keeping it avoids every missing-symbol surprise)
objcopy --redefine-sym main=crispasr_cli_orig_main \
    examples/cli/CMakeFiles/crispasr-cli.dir/cli.cpp.o /tmp/voice_spike/cli_renamed.o

# 3. link: take the crispasr link line, swap cli.cpp.o for the renamed copy, add
#    spike.o, and redirect -o
ninja -t commands bin/crispasr | tail -1   # edit as above, then run it

# 4. run
/tmp/voice_spike/voice_spike
```

Model/wav paths are compiled in and can be overridden: `SPIKE_ASR_MODEL`,
`SPIKE_TTS_TALKER`, `SPIKE_TTS_CODEC`, `SPIKE_VOICE_DIR`, `SPIKE_CHAT_MODEL`,
`SPIKE_ASR_WAV`. Exit code 0 = both probes passed, 1 = degraded output,
99 = watchdog fired (hang), 2 = init/load failure.

Refreshing the numbers above: run it twice in a fresh process and read the first
("cold") block of each; the warm numbers come from the 5×-repeat probe.
