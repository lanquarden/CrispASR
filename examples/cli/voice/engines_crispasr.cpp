// engines_crispasr.cpp — real engine implementations on CrispASR's adapters.
//
// ASR/TTS go through the CrispasrBackend vtable that the CLI and server use
// (TTS streaming is not exposed by the C ABI); the chat LLM goes through the
// public crispasr_chat_* C ABI. All three share the process-wide ggml device.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "core/gpu_backend_pref.h"
#include "crispasr_backend.h"
#include "crispasr_chat.h"
#include "engines.h"
#include "whisper_params.h"

namespace crispasr {
namespace voice {
namespace {

int64_t mono_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch())
        .count();
}

std::string trim_copy(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// ---------------------------------------------------------------------------
class CrispasrAsrEngine : public AsrEngine {
   public:
    explicit CrispasrAsrEngine(const Config& cfg) : cfg_(cfg) {}

    bool init() {
        static std::once_flag gpu_once;
        std::call_once(gpu_once, [] { crispasr_set_gpu_backend_pref("vulkan"); });
        p_.backend = cfg_.asr_backend;
        p_.model = cfg_.asr_model;
        p_.no_warmup = true;
        p_.language = "auto";
        p_.n_threads = 8;
        p_.use_gpu = true;
        p_.gpu_backend = "vulkan";
        be_ = crispasr_create_backend(cfg_.asr_backend);
        if (!be_) return false;
        if (!be_->init(p_)) {
            be_.reset();
            return false;
        }
        return true;
    }

    AsrResult transcribe(const float* samples, int n_samples,
                         int64_t t_offset_ms) override {
        AsrResult r;
        if (!be_ || n_samples <= 0) return r;
        const int64_t t0 = mono_ms();
        auto segs = be_->transcribe(samples, n_samples, t_offset_ms / 10, p_);
        std::string text;
        for (auto& s : segs) {
            if (!text.empty()) text += ' ';
            text += s.text;
        }
        r.text = trim_copy(text);
        r.ok = !r.text.empty();
        r.cost_ms = mono_ms() - t0;
        return r;
    }

    bool ready() const override { return be_ != nullptr; }

   private:
    Config cfg_;
    whisper_params p_;
    std::unique_ptr<CrispasrBackend> be_;
};

// ---------------------------------------------------------------------------
class CrispasrTtsEngine : public TtsEngine {
   public:
    explicit CrispasrTtsEngine(const Config& cfg) : cfg_(cfg) {}

    bool init() {
        p_.backend = cfg_.tts_backend;
        p_.model = cfg_.tts_model;
        p_.tts_codec_model = cfg_.tts_tokenizer;
        p_.tts_voice_dir = cfg_.tts_voice_dir;
        p_.tts_voice = cfg_.tts_voice;
        p_.no_warmup = true;
        p_.n_threads = 8;
        p_.use_gpu = true;
        p_.gpu_backend = "vulkan";
        p_.target_lang = "en";
        be_ = crispasr_create_backend(cfg_.tts_backend);
        if (!be_) return false;
        if (!be_->init(p_)) {
            be_.reset();
            return false;
        }
        in_rate_ = be_->tts_sample_rate();
        return true;
    }

    bool ready() const override { return be_ != nullptr; }

    void synthesize(const std::string& text, PcmCallback cb) override {
        if (!be_ || text.empty()) return;
        resample_pending_.clear();
        resample_pos_ = 0.0;
        started_ = false;
        work_.clear();

        be_->synthesize_streaming(text, p_, [&](const float* pcm, int n, bool final) {
            std::vector<float> out;
            resample_push(pcm, n, &out);
            if (out.empty()) return;
            work_.insert(work_.end(), out.begin(), out.end());
            if (!started_) {
                trim_leading();
                if (!started_) {
                    // keep only a short pre-roll until speech actually begins
                    if (work_.size() > (size_t)cfg_.sample_rate / 10)
                        work_.erase(work_.begin(),
                                    work_.end() - cfg_.sample_rate / 10);
                    return;
                }
            }
            emit_steady(cb, final);
        });
        // flush: drop the trailing silence the model pads after the last word
        if (started_) {
            const int keep_pad = cfg_.sample_rate / 10;  // 100 ms
            size_t last = work_.size();
            const size_t keep = (size_t)cfg_.sample_rate / 4;  // 250 ms hold
            size_t i = work_.size() > keep ? work_.size() - keep : 0;
            while (i < work_.size()) {
                bool loud = false;
                for (size_t k = i; k < std::min(work_.size(), i + 160); ++k)
                    if (std::fabs(work_[k]) > 0.006f) { loud = true; break; }
                if (loud) last = std::min(work_.size(), i + keep);
                i += 160;
            }
            const size_t emit = std::min(work_.size(), last + keep_pad);
            if (emit > 0) cb(work_.data(), (int)emit, true);
            work_.clear();
        }
    }

   private:
    void resample_push(const float* in, int n, std::vector<float>* out) {
        if (in_rate_ == cfg_.sample_rate) {
            out->assign(in, in + n);
            return;
        }
        const double step = (double)cfg_.sample_rate / (double)in_rate_;
        resample_pending_.insert(resample_pending_.end(), in, in + n);
        double pos = resample_pos_;
        while (pos + 1.0 < (double)resample_pending_.size()) {
            const size_t i = (size_t)pos;
            const double f = pos - (double)i;
            out->push_back((float)(resample_pending_[i] * (1.0 - f) +
                                   resample_pending_[i + 1] * f));
            pos += step;
        }
        const size_t keep = (size_t)pos;
        if (keep > 0) {
            resample_pending_.erase(resample_pending_.begin(),
                                    resample_pending_.begin() + keep);
            pos -= (double)keep;
        }
        resample_pos_ = pos;
    }

    void trim_leading() {
        // 10 ms RMS window; skip model silence until speech starts.
        const int win = std::max(1, cfg_.sample_rate / 100);
        for (size_t i = 0; i + win <= work_.size(); i += win) {
            double acc = 0.0;
            for (size_t k = i; k < i + win; ++k) acc += (double)work_[k] * work_[k];
            const double rms = std::sqrt(acc / win);
            if (rms > 0.006) {
                work_.erase(work_.begin(), work_.begin() + i);
                started_ = true;
                return;
            }
        }
    }

    // Emit everything except a 250 ms hold so trailing silence can be trimmed
    // when the model ends the piece.
    void emit_steady(PcmCallback& cb, bool) {
        const size_t hold = (size_t)cfg_.sample_rate / 4;
        if (work_.size() <= hold) return;
        const size_t n = work_.size() - hold;
        cb(work_.data(), (int)n, false);
        work_.erase(work_.begin(), work_.begin() + n);
    }

    Config cfg_;
    whisper_params p_;
    std::unique_ptr<CrispasrBackend> be_;
    bool started_ = false;
    std::vector<float> work_;
    int in_rate_ = 24000;
    std::vector<float> resample_pending_;
    double resample_pos_ = 0.0;
};

// ---------------------------------------------------------------------------
class CrispasrLlmEngine : public LlmEngine {
   public:
    explicit CrispasrLlmEngine(const Config& cfg) : cfg_(cfg) {}

    bool init() {
        crispasr_chat_open_params p;
        crispasr_chat_open_params_default(&p);
        p.n_ctx = cfg_.chat_ctx;
        p.n_gpu_layers = cfg_.chat_gpu_layers;
        crispasr_chat_error err{};
        session_ = crispasr_chat_open(cfg_.chat_model.c_str(), &p, &err);
        if (!session_) {
            last_error_ = err.message;
            return false;
        }
        crispasr_chat_set_abort_callback(session_, &CrispasrLlmEngine::abort_cb,
                                        this);
        return true;
    }

    ~CrispasrLlmEngine() override {
        if (session_) crispasr_chat_close(session_);
    }

    LlmResult generate(const std::vector<ChatMessage>& messages,
                       int max_tokens, TokenCallback on_token,
                       const std::string& grammar) override {
        (void)grammar;
        LlmResult r;
        if (!session_) return r;
        abort_.store(false);
        std::vector<crispasr_chat_message> msgs;
        msgs.reserve(messages.size());
        for (auto& m : messages)
            msgs.push_back({m.role.c_str(), m.content.c_str()});

        crispasr_chat_generate_params gp;
        crispasr_chat_generate_params_default(&gp);
        gp.max_tokens = max_tokens;
        gp.temperature = 0.4f;

        struct Ctx {
            CrispasrLlmEngine* self;
            TokenCallback* cb;
            LlmResult* result;
            int64_t started;
        } ctx{this, &on_token, &r, mono_ms()};

        auto token_cb = [](const char* chunk, void* user) {
            Ctx* c = (Ctx*)user;
            if (c->result->tokens++ == 0)
                c->result->ttft_ms = mono_ms() - c->started;
            c->result->text += chunk;
            if (*c->cb && !(*c->cb)(std::string(chunk))) {
                c->self->abort_.store(true);
            }
        };
        crispasr_chat_error err{};
        const int32_t rc = crispasr_chat_generate_stream(
            session_, msgs.data(), msgs.size(), &gp, token_cb, &ctx, &err);
        r.total_ms = mono_ms() - ctx.started;
        r.aborted = rc == CRISPASR_CHAT_ERR_ABORTED;
        r.ok = rc == 0 && !r.text.empty();
        return r;
    }

    void request_abort() override { abort_.store(true); }
    bool ready() const override { return session_ != nullptr; }
    const std::string& last_error() const { return last_error_; }

   private:
    static bool abort_cb(void* user) {
        return !static_cast<CrispasrLlmEngine*>(user)->abort_.load();
    }

    Config cfg_;
    crispasr_chat_session_t session_ = nullptr;
    std::atomic<bool> abort_{true};
    std::string last_error_;
};

}  // namespace

Engines make_crispasr_engines(const Config& cfg) {
    Engines e;
    auto asr = std::make_unique<CrispasrAsrEngine>(cfg);
    auto tts = std::make_unique<CrispasrTtsEngine>(cfg);
    auto llm = std::make_unique<CrispasrLlmEngine>(cfg);
    if (!asr->init() || !tts->init() || !llm->init()) {
        return e;  // empty: not ready
    }
    e.asr = asr.release();
    e.tts = tts.release();
    e.llm = llm.release();
    e.owns = true;
    return e;
}

}  // namespace voice
}  // namespace crispasr
