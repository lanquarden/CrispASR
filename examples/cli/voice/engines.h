// engines.h — abstract inference interfaces for the voice loop.
//
// The real implementations (engines_crispasr.cpp) sit on CrispASR's
// CrispasrBackend vtable and crispasr_chat_* C ABI. Fakes implement the same
// interfaces with deterministic timing so the sequencer, scheduler and harness
// can be tested without a GPU.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "voice_types.h"

namespace crispasr {
namespace voice {

// ---- ASR -----------------------------------------------------------------
struct AsrResult {
    bool ok = false;
    std::string text;
    int64_t cost_ms = 0;
};

class AsrEngine {
   public:
    virtual ~AsrEngine() = default;
    // Blocking. `samples` is mono float32 at cfg.sample_rate.
    virtual AsrResult transcribe(const float* samples, int n_samples,
                                 int64_t t_offset_ms) = 0;
    virtual bool ready() const = 0;
};

// ---- TTS -----------------------------------------------------------------
// Called once per PCM chunk. `is_final` marks the end of this piece.
using PcmCallback =
    std::function<void(const float* pcm, int n_samples, bool is_final)>;

class TtsEngine {
   public:
    virtual ~TtsEngine() = default;
    // Blocking, streaming. One call should stay under ~400 ms of wall time;
    // the caller keeps text pieces short.
    virtual void synthesize(const std::string& text, PcmCallback cb) = 0;
    // Best-effort abort for the *next* call; implementations without an abort
    // hook return false.
    virtual bool abort_pending() { return false; }
    virtual bool ready() const = 0;
};

// ---- Chat LLM ------------------------------------------------------------
struct ChatMessage {
    std::string role;  // "system" | "user" | "assistant"
    std::string content;
};

struct LlmResult {
    bool ok = false;
    std::string text;
    int64_t ttft_ms = 0;
    int64_t total_ms = 0;
    int tokens = 0;
    bool aborted = false;
};

using TokenCallback = std::function<bool(const std::string& piece)>;  // false = stop

class LlmEngine {
   public:
    virtual ~LlmEngine() = default;
    // Blocking; stops after max_tokens or when `on_token` returns false.
    virtual LlmResult generate(const std::vector<ChatMessage>& messages,
                               int max_tokens, TokenCallback on_token,
                               const std::string& grammar = std::string()) = 0;
    virtual void request_abort() = 0;
    virtual bool ready() const = 0;
};

// ---- Bundle --------------------------------------------------------------
struct Engines {
    AsrEngine* asr = nullptr;
    TtsEngine* tts = nullptr;
    LlmEngine* llm = nullptr;
    bool owns = false;
    ~Engines();
};

// Factory implemented in engines_crispasr.cpp; fakes in engines_fake.cpp.
Engines make_crispasr_engines(const Config& cfg);
Engines make_fake_engines(const Config& cfg);

}  // namespace voice
}  // namespace crispasr
