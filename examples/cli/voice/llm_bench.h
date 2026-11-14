// llm_bench.h — LLM-only latency bench for the voice loop (home-ops#584).
//
// Why a separate bench: in the loop, the ASR and TTS costs are stable and well
// measured (RESULTS.md), while the chat model is the one stage whose cost moves
// with what we ask it for. Answering "what does a filler prompt cost, and what
// does telling the model about the expected wait cost" needs numbers for the
// model alone, without the ASR/TTS/task timeline around it.
//
// The bench measures, per prompt kind: TTFT (prefill), decode tokens, decode
// throughput and total. It reports p50/p95 across reps, because TTFT on a
// shared Vulkan device has a tail that a mean hides.
//
// Usage: build-vk/bin/crispasr-voice --bench-llm --chat-model PATH [--reps N]
#pragma once

#include "voice_types.h"

namespace crispasr {
namespace voice {

// Returns 0 when every case produced output.
int run_llm_bench(const Config& cfg);

// Bridge-quality eval: sweeps prompt variants x sampling for the holding-phrase
// request and scores the RAW output (no guard, no bank fallback), so the number
// describes the model, not the safety net. Prints, per variant: clean rate
// (no device/condition claim), median words, TTFT and a sample.
int run_bridge_eval(const Config& cfg);

// Context-cost bench: how much does a dynamic context block (home sensor state)
// cost in TTFT? Sweeps block sizes and reports three numbers per size:
//   cold    - fresh session, everything must be prefilled
//   warm    - same session, identical prompt (should be ~free: cached prefix)
//   changed - same session, same size, one value different, which is the real
//             cost of a fresh sensor snapshot behind an unchanged system prompt
int run_context_bench(const Config& cfg, const std::string& context_file);

// Prefill pipelining: is the transcript-independent part of the prompt cheaper
// when it is decoded at speech onset (prefill_only) instead of inside the
// generating call? Reports prefill cost, baseline TTFT, and TTFT after a
// prefetch of the context and of a partial transcript.
int run_prefill_bench(const Config& cfg, const std::string& context_file);

}  // namespace voice
}  // namespace crispasr
