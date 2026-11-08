# VRAM budget for the voice stack

Measured 2026-11-09 on the RX 7900 XT (21.46 GB total, Vulkan/RADV), harness
`delegation_real` with 1 measured + 1 warm-up run, `rocm-smi --showmeminfo vram`
sampled every 0.5 s. "Incremental" is peak minus the idle baseline of the same
configuration.

| configuration | idle | peak total | incremental | notes |
|---|---|---|---|---|
| desktop only, STT/TTS containers stopped | 2.24 GiB | — | — | baseline |
| deployed STT+TTS containers, idle | 8.17 GiB | — | ~5.9 GiB | one ASR + one TTS process, shared volume |
| voice stack, Qwen3-0.6B chat | 2.24 GiB | 10.49 GiB | 7.87 GiB | ASR 1.7B + TTS 1.7B + chat 0.6B in one process |
| **voice stack, Qwen3-1.7B chat** | 2.24 GiB | **11.63 GiB** | **8.95 GiB** | recommended model |
| containers + voice stack (1.7B) + desktop | 8.17 GiB | **16.42 GiB** | — | 3.5 GiB headroom, run succeeds |

Process RSS is only 2.56 GB; the rest is Vulkan-resident weights, KV cache and
compute buffers.

## What this means

- Budget **~9 GiB** for `crispasr-voice` with the 1.7B chat model, or ~7.9 GiB
  with 0.6B. The three model files are 2.39 + 1.97 + 1.83 GB; the extra ~2.7 GiB
  is KV/compute/workspace.
- The stack **replaces** the two deployed containers (which need ~5.9 GiB) and
  adds the chat model: consolidation costs roughly +3 GiB over today's STT+TTS
  pair, but it is one process/one device and removes the HTTP hop.
- Running the voice stack **alongside** the deployed pair fits (16.42 GiB peak),
  so the experiment can be rolled out with the current HA path as a fallback —
  which is the desired rollback property. Do not expect room for a second
  large model.
- The numbers include the logical size the driver reports, not just the process
  footprint; the 0.6B -> 1.7B step is 1.08 GiB, matching the larger weight file.
