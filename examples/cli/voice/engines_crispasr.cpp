// engines_crispasr.cpp — real engine implementations on CrispASR's adapters.
//
// ASR/TTS go through the CrispasrBackend vtable that the CLI and server use
// (TTS streaming is not exposed by the C ABI); the chat LLM goes through the
// public crispasr_chat_* C ABI. All three share the process-wide ggml device.
#include <algorithm>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

constexpr double kPi = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// Anti-aliased streaming resampler for the TTS path (model 24 kHz -> pipeline
// 16 kHz).
//
// Design: 63-tap Hamming-windowed sinc low-pass with the -6 dB cutoff at
// 0.45 * out_rate (7.2 kHz for 24 -> 16 kHz), evaluated on a 256-phase
// polyphase grid so any fractional output position is covered without a
// transcendental per sample. Every phase row is normalised to unity DC gain,
// which keeps the passband flat and deterministic across phases.
//
// 63 taps at 24 kHz is 2.6 ms of impulse response: the filter is symmetric,
// so the group delay (31 input samples) cancels against the start offset and
// the only added latency is the 31 samples of lookahead the kernel needs
// (~1.3 ms at 24 kHz), independent of the chunking. Each piece also loses its
// first and last ~1.3 ms of output (the kernel flanks it cannot fill), 2.6 ms
// in total: negligible against the >= 100 ms pieces the sequencer emits and
// well inside the 5 ms budget. State is carried across calls and cleared by
// reset() (once per synthesize()), so consecutive callback chunks are filtered
// as one continuous signal.
//
// The Hamming window gives ~-53 dB in the stopband, whose edge sits at
// ~0.5 * out_rate; the 8-16 kHz band is therefore attenuated before the
// decimation instead of folding back into 0-8 kHz (linear interpolation with
// no filter folded all of it back). Measured (24 -> 16 kHz): 0.0 dB to 6 kHz,
// -2.4 dB at 7 kHz, -6 dB at 7.2 kHz, -49 dB at 8 kHz, -67 dB for 10 kHz (the
// alias at 6 kHz), i.e. everything that could fold back is >= 49 dB down.
class SincDecimator {
   public:
    static constexpr int kTaps = 63;  // odd
    static constexpr int kHalf = kTaps / 2;
    static constexpr int kPhases = 256;

    // Sets the rates and (re)builds the polyphase table. A no-op when the same
    // rates are already configured, so init() may call it unconditionally.
    void configure(int in_rate, int out_rate) {
        if (ready_ && in_rate == in_rate_ && out_rate == out_rate_) return;
        in_rate_ = in_rate;
        out_rate_ = out_rate;
        ready_ = false;
        reset();
        if (in_rate_ <= 0 || out_rate_ <= 0 || in_rate_ == out_rate_) return;
        build_table();
        ready_ = true;
    }

    // Drops the filter state; call once per synthesize() so pieces never bleed
    // into each other. The first output position is kHalf so the symmetric
    // kernel has a full left flank: output sample k therefore represents input
    // time k * in_rate / out_rate, and the only latency is the kHalf-sample
    // lookahead.
    void reset() {
        buf_.clear();
        pos_ = (double)kHalf;
    }

    // Streams n input samples through the filter, appending every output
    // sample whose full kernel is available to *out. The only allocation is
    // the amortised growth of *out and of the internal input buffer.
    void push(const float* in, int n, std::vector<float>* out) {
        if (!out || !in || n <= 0) return;
        if (!ready_) {
            out->insert(out->end(), in, in + n);
            return;
        }
        buf_.insert(buf_.end(), in, in + n);
        const double step = (double)in_rate_ / (double)out_rate_;  // input samples per output
        const size_t avail = buf_.size();
        for (;;) {
            const long long centre = (long long)std::floor(pos_);
            if (centre - kHalf < 0) break;                   // need more history
            if ((size_t)(centre + kHalf) >= avail) break;    // need more lookahead
            int phase = (int)((pos_ - (double)centre) * kPhases);
            if (phase < 0) phase = 0;
            if (phase >= kPhases) phase = kPhases - 1;
            const float* h = table_.data() + (size_t)phase * kTaps;
            // tap t pairs h(t - kHalf + phase/N) with sample centre - (t - kHalf)
            const float* x = buf_.data() + (centre + kHalf);
            float acc = 0.0f;
            for (int t = 0; t < kTaps; ++t) acc += h[t] * x[-t];
            out->push_back(acc);
            pos_ += step;
        }
        // Forget input no future output position can reach again.
        const long long keep_from = (long long)std::floor(pos_) - kHalf;
        if (keep_from > 0) {
            const size_t drop = std::min<size_t>((size_t)keep_from, buf_.size());
            buf_.erase(buf_.begin(), buf_.begin() + drop);
            pos_ -= (double)drop;
        }
    }

   private:
    void build_table() {
        table_.assign((size_t)kPhases * kTaps, 0.0f);
        const double fs = (double)in_rate_;
        double fc = 0.45 * (double)out_rate_;  // low-pass just below out Nyquist
        if (fc > 0.45 * fs) fc = 0.45 * fs;    // never above the input Nyquist
        const double norm = 2.0 * fc / fs;     // sinc scaling in input samples
        for (int p = 0; p < kPhases; ++p) {
            const double frac = (double)p / (double)kPhases;
            float* row = table_.data() + (size_t)p * kTaps;
            double sum = 0.0;
            for (int t = 0; t < kTaps; ++t) {
                const double j = (double)(t - kHalf) + frac;  // distance, input samples
                double h = 0.0;
                if (std::fabs(j) <= (double)kHalf) {
                    const double x = norm * j;
                    const double sinc =
                        (std::fabs(x) < 1e-12) ? 1.0 : std::sin(kPi * x) / (kPi * x);
                    const double win = 0.54 + 0.46 * std::cos(kPi * j / (double)kHalf);
                    h = norm * sinc * win;
                }
                row[t] = (float)h;
                sum += h;
            }
            if (std::fabs(sum) > 1e-9)
                for (int t = 0; t < kTaps; ++t) row[t] = (float)(row[t] / sum);
        }
    }

    int in_rate_ = 0;
    int out_rate_ = 0;
    bool ready_ = false;
    std::vector<float> table_;  // kPhases * kTaps, unity-DC per row
    std::vector<float> buf_;    // input history + lookahead, consumed from front
    double pos_ = (double)kHalf;  // next output position, in buf_ samples
};

// ---------------------------------------------------------------------------
// Streaming silence gate for TTS output (gate-begin/gate-end markers delimit
// the block the unit test extracts verbatim).
//
// The model pads every piece: ~0.42 s of lead and ~0.20 s of tail on the clone
// voice, plus pauses it renders inside a piece (quotes, clause breaks, and
// plain padding between phrases). Those arrive as samples, land in the playback
// ring as dead air, and no ring underflow counter can see them. The gate drops
// them on the way out.
//
// Window model: the stream is tiled into 10 ms windows (160 samples at 16 kHz)
// classified by RMS against 0.006. Windows are never split, samples are copied
// verbatim (no gain, no reordering), and the output is a subsequence of the
// input.
//
// A window counts as speech only when it belongs to a *sustained* loud stretch:
// 40 ms (4 windows) of consecutive above-floor windows. Classification is
// therefore deferred by one 40 ms lookahead, so the answer is available in the
// same streaming pass. Windows inside the hangover of a confirmed stretch stay
// on the speech side; isolated 10-20 ms blips do not, which matters because the
// model's decay noise sits between 0.006 and 0.01 and would otherwise cut a
// pause into pieces shorter than the 220 ms cap (measurement and the rejected
// alternative are documented at classify()).
//
//   leading   drop everything before the first loud window, then prepend up to
//             20 ms of the silence immediately before it as pre-roll
//   mid       hold silence, bounded at 220 ms: when a loud window arrives the
//             held silence is emitted (a natural pause is preserved verbatim);
//             past 220 ms the surplus is dropped as it arrives, so an internal
//             pause can never exceed 220 ms
//   trailing  at flush, drop everything after the last loud window except a
//             40 ms pad
//
// Holding is what lets the trailing rule work: silence after the last loud
// window is never written ahead of the decision, so a pause the mid rule
// truncated to 220 ms can still be shortened to the 40 ms pad at the end of the
// piece instead of being emitted as a 220 ms gap. Worst-case added latency is
// the 40 ms hangover lookahead plus the 220 ms hold, and during both there is
// no audio that was cleared for playback anyway.
//
// --- gate-begin ---
class SilenceGate {
   public:
    // Derives the window and hold lengths from the pipeline rate.
    void configure(int sample_rate, double rms_floor = 0.006) {
        if (sample_rate <= 0) sample_rate = 16000;
        win_ = std::max(1, sample_rate / 100);  // 10 ms
        preroll_ = sample_rate / 50;            // 20 ms
        hold_cap_ = sample_rate * 220 / 1000;   // 220 ms
        pad_ = sample_rate * 40 / 1000;         // 40 ms
        hangover_ = std::max(1, (sample_rate * 40 / 1000) / win_);  // 40 ms
        floor_ = rms_floor;
        reset();
    }

    // Clears every buffer and decision, so pieces never bleed into each other.
    void reset() {
        started_ = false;
        in_run_ = false;
        in_.clear();
        lead_.clear();
        hold_.clear();
        ready_.clear();
    }

    // Streams n samples in and appends every sample cleared for emission to
    // *out. The caller drains *out per chunk, so nothing here grows without
    // bound: in_ < win_, lead_ <= 20 ms, hold_ <= 220 ms, pend_ <= 50 ms.
    //
    // Classification is deferred by one 40 ms hangover so "is this window
    // speech?" can be answered with the windows on both sides of it; samples
    // are buffered, never reordered.
    void feed(const float* pcm, int n, std::vector<float>* out) {
        if (!out) return;
        if (pcm != nullptr && n > 0) in_.insert(in_.end(), pcm, pcm + n);
        const size_t w = (size_t)win_;
        size_t i = 0;
        while (in_.size() - i >= w) {
            push_window(in_.data() + i, win_);
            i += w;
        }
        if (i > 0) in_.erase(in_.begin(), in_.begin() + (long)i);
        while (pend_.size() > (size_t)hangover_) emit_head();
        out->insert(out->end(), ready_.begin(), ready_.end());
        ready_.clear();
    }

    // End of piece: score the remaining windows, then apply the trailing rule.
    void flush(std::vector<float>* out) {
        if (!out) return;
        if (!in_.empty()) {
            push_window(in_.data(), (int)in_.size());
            in_.clear();
        }
        while (!pend_.empty()) emit_head();  // no lookahead left to confirm more
        lead_.clear();  // a piece that never gets loud emits nothing
        if (started_ && !hold_.empty()) {
            const size_t keep = std::min<size_t>((size_t)pad_, hold_.size());
            ready_.insert(ready_.end(), hold_.begin(), hold_.begin() + keep);
        }
        hold_.clear();
        out->insert(out->end(), ready_.begin(), ready_.end());
        ready_.clear();
    }

   private:
    bool raw_loud(const float* s, int n) const {
        if (s == nullptr || n <= 0) return false;
        double acc = 0.0;
        for (int i = 0; i < n; ++i) acc += (double)s[i] * (double)s[i];
        return std::sqrt(acc / n) > floor_;
    }

    // A window counts as speech only when it belongs to a *sustained* loud
    // stretch: a run of hangover_ (4 x 10 ms = 40 ms) consecutive above-floor
    // windows. The delay in the internal buffer is what makes this answerable
    // in one pass -- when the head window is classified, its three successors
    // are already buffered, and a run confirmed at an earlier window keeps the
    // whole stretch on the speech side through in_run_.
    //
    // Why sustained rather than "within 40 ms of a loud window": the model's
    // decay noise sits between 0.006 and 0.01, so a pause is dappled with
    // 10-20 ms blips. Extending every blip by 40 ms in both directions makes
    // the blips merge into one long loud stretch, which feeds the pause to the
    // lead/hold side as speech and preserves it in full (measured: a 400 ms
    // pause with 10 ms blips every 80 ms survived as 400 ms, and a 150 ms
    // pause was clipped to 80 ms). Requiring 40 ms of sustained loudness
    // instead keeps those blips on the silence side, so the pause stays one
    // silence run and the 220 ms cap truncates it. Genuine speech is >= 40 ms
    // above the floor, so the onset of every word is still classified from its
    // first window.
    bool classify() {
        if (pend_raw_.empty()) return false;
        if (pend_raw_.front() == 0) {
            in_run_ = false;
            return false;
        }
        if (in_run_) return true;
        if (pend_raw_.size() >= (size_t)hangover_) {
            bool sustained = true;
            for (size_t k = 1; k < (size_t)hangover_; ++k)
                if (pend_raw_[k] == 0) {
                    sustained = false;
                    break;
                }
            if (sustained) {
                in_run_ = true;
                return true;
            }
        }
        return false;  // blip shorter than the hangover: silence, not speech
    }

    // Acts on the oldest buffered window (loud -> speech, quiet -> lead/hold).
    void emit_head() {
        const bool is_loud = classify();
        const std::vector<float>& s = pend_.front();
        if (is_loud) {
            if (!started_) {
                started_ = true;
                ready_.insert(ready_.end(), lead_.begin(), lead_.end());
                lead_.clear();
            } else if (!hold_.empty()) {
                ready_.insert(ready_.end(), hold_.begin(), hold_.end());
                hold_.clear();
            }
            ready_.insert(ready_.end(), s.begin(), s.end());
        } else if (!started_) {
            // Keep only the 20 ms that ends where speech may start.
            lead_.insert(lead_.end(), s.begin(), s.end());
            if ((int)lead_.size() > preroll_)
                lead_.erase(lead_.begin(), lead_.end() - preroll_);
        } else {
            hold_.insert(hold_.end(), s.begin(), s.end());
            if ((int)hold_.size() > hold_cap_) hold_.resize((size_t)hold_cap_);
        }
        pend_.erase(pend_.begin());
        pend_raw_.erase(pend_raw_.begin());
    }

    void push_window(const float* s, int n) {
        pend_.emplace_back(s, s + n);
        pend_raw_.push_back(raw_loud(s, n) ? (unsigned char)1 : (unsigned char)0);
    }

    int win_ = 160;
    int preroll_ = 320;
    int hold_cap_ = 3520;
    int pad_ = 640;
    int hangover_ = 4;  // consecutive loud 10 ms windows (40 ms) = speech
    double floor_ = 0.006;
    bool started_ = false;
    bool in_run_ = false;  // current window sits inside a confirmed loud run
    std::vector<float> in_;              // input tail shorter than one window
    std::vector<float> lead_;            // silence before the first loud window
    std::vector<float> hold_;            // post-speech silence awaiting a decision
    std::vector<std::vector<float>> pend_;  // windows awaiting classification
    std::vector<unsigned char> pend_raw_;   // their above-floor flags
    std::vector<float> ready_;           // cleared by the caller, never reordered
};
// --- gate-end ---

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
        resampler_.configure(in_rate_, cfg_.sample_rate);
        return true;
    }

    bool ready() const override { return be_ != nullptr; }

    void synthesize(const std::string& text, PcmCallback cb) override {
        if (!be_ || text.empty()) return;
        resampler_.configure(in_rate_, cfg_.sample_rate);
        resampler_.reset();
        gate_.configure(cfg_.sample_rate);  // also resets: no state crosses pieces
        emitted_once_ = false;
        emitted_any_ = false;
        out_.clear();

        be_->synthesize_streaming(text, p_, [&](const float* pcm, int n, bool) {
            std::vector<float> res;
            resampler_.push(pcm, n, &res);
            if (res.empty()) return;
            out_.clear();
            gate_.feed(res.data(), (int)res.size(), &out_);
            if (out_.empty()) return;
            debug_first_emit();
            emitted_any_ = true;
            // Never final here: the gate still holds up to 220 ms and the tail
            // rule runs in flush(), which owns the is_final chunk.
            cb(out_.data(), (int)out_.size(), false);
        });

        out_.clear();
        gate_.flush(&out_);
        if (!out_.empty()) {
            debug_first_emit();
            emitted_any_ = true;
            cb(out_.data(), (int)out_.size(), true);
        } else if (emitted_any_) {
            // Piece ended on a loud window: keep is_final on the last chunk.
            float none = 0.0f;
            cb(&none, 0, true);
        }
    }

   private:
    void debug_first_emit() {
        if (emitted_once_ || std::getenv("VOICE_DEBUG") == nullptr) return;
        emitted_once_ = true;
        double acc = 0.0;
        const size_t k = std::min<size_t>(out_.size(), 160);
        for (size_t i = 0; i < k; ++i) acc += (double)out_[i] * (double)out_[i];
        std::fprintf(stderr, "[dbg] tts first emit n=%zu rms=%.4f\n", out_.size(),
                     k ? std::sqrt(acc / k) : 0.0);
    }

    Config cfg_;
    whisper_params p_;
    std::unique_ptr<CrispasrBackend> be_;
    bool emitted_once_ = false;  // first emit logged
    bool emitted_any_ = false;   // this piece produced audio
    std::vector<float> out_;     // per-chunk gate output
    int in_rate_ = 24000;
    SincDecimator resampler_;
    SilenceGate gate_;
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
