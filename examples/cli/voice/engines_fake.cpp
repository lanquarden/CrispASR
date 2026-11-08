// engines_fake.cpp — deterministic fake ASR / TTS / LLM for the voice-loop
// harness (home-ops#584).
//
// These exist so the sequencer, scheduler and harness can be exercised without
// a GPU, and so the harness stays replayable: identical interval, identical
// text, identical PCM. Every timing knob can be overridden from the
// environment, which is how the harness varies one scenario against another
// (Config has no fields for engine latencies).
//
// Environment (all optional, read per call except where noted):
//   VOICE_FAKE_TRANSCRIPT            final ASR text; when unset the fake
//                                    decides "speech" from frame energy
//   VOICE_FAKE_ASR_EXTRA_MS          added to every ASR decode cost
//   VOICE_FAKE_TTS_TTFA_MS           time to first PCM (default 350)
//   VOICE_FAKE_TTS_TTFA_ALWAYS=1     sleep TTFA on every call, not just the first
//   VOICE_FAKE_TTS_RATE              synthesis speed vs real time (default 4.0)
//   VOICE_FAKE_TTS_AMP               peak amplitude (default 0.15)
//   VOICE_FAKE_LLM_TTFT_MS           time to first token (default 80)
//   VOICE_FAKE_LLM_TPT_MS            per-token pacing (default 12)
//
// Role routing for the LLM is by system-prompt keyword, matching the channels
// the sequencer owns: "ack" -> a restating acknowledgement, "filler" -> a
// rotating holding phrase, "fail"/"error" -> an honest failure line, otherwise
// the answer. The canned texts are fixed by contract so metrics assertions can
// reason about them.

#include "engines.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "voice_types.h"

namespace crispasr {
namespace voice {
namespace {

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
int env_int(const char* name, int dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    long x = std::strtol(v, &end, 10);
    if (!end || *end != '\0') return dflt;
    return (int)x;
}

double env_double(const char* name, double dflt) {
    const char* v = std::getenv(name);
    if (!v || !*v) return dflt;
    char* end = nullptr;
    double x = std::strtod(v, &end);
    if (!end || *end != '\0' || !std::isfinite(x)) return dflt;
    return x;
}

std::string env_str(const char* name, const std::string& dflt) {
    const char* v = std::getenv(name);
    if (!v) return dflt;
    return std::string(v);
}

void sleep_ms(int64_t ms) {
    if (ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// Sleeps in small slices so an abort does not have to wait out a whole TTFT.
bool sleep_interruptible(int64_t ms, const std::atomic<bool>& abort) {
    const int64_t slice = 5;
    int64_t left = ms;
    while (left > 0) {
        if (abort.load()) return false;
        int64_t step = left < slice ? left : slice;
        std::this_thread::sleep_for(std::chrono::milliseconds(step));
        left -= step;
    }
    return !abort.load();
}

std::string lower_copy(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = (char)(c - 'A' + 'a');
    }
    return s;
}

int count_words(const std::string& s) {
    int n = 0;
    bool in_word = false;
    for (char c : s) {
        bool space = (c == ' ' || c == '\t' || c == '\n' || c == '\r');
        if (!space && !in_word) { ++n; in_word = true; }
        else if (space) in_word = false;
    }
    return n;
}

int64_t steady_now() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// 80 ms of decode per second of audio at 16 kHz, scaled to the real rate.
constexpr int kAsrMsPerSecondOfAudio = 80;
constexpr double kPi = 3.14159265358979323846;
// Energy gate for "is there speech in this buffer" (fixtures peak near 0.2).
constexpr double kSpeechRms = 0.01;

const char* kDefaultUtterance = "what is the state of the garage door";

// ---------------------------------------------------------------------------
// ASR
// ---------------------------------------------------------------------------
class FakeAsrEngine : public AsrEngine {
   public:
    explicit FakeAsrEngine(const Config& cfg) : rate_(cfg.sample_rate > 0 ? cfg.sample_rate : 16000) {}

    AsrResult transcribe(const float* samples, int n_samples, int64_t /*t_offset_ms*/) override {
        AsrResult r;
        r.ok = true;
        const int head = std::min(n_samples, rate_ / 2);
        double sum = 0.0;
        for (int i = 0; i < head; ++i) sum += (double)samples[i] * (double)samples[i];
        const double rms = head > 0 ? std::sqrt(sum / head) : 0.0;

        const std::string forced = env_str("VOICE_FAKE_TRANSCRIPT", std::string());
        if (!forced.empty()) {
            r.text = forced;
        } else if (rms >= kSpeechRms) {
            r.text = kDefaultUtterance;
        }
        // No speech -> empty transcript, still "ok": the loop decides what an
        // empty final means.
        r.cost_ms = (int64_t)n_samples * kAsrMsPerSecondOfAudio / rate_;
        sleep_ms(r.cost_ms + env_int("VOICE_FAKE_ASR_EXTRA_MS", 0));
        return r;
    }

    bool ready() const override { return true; }

   private:
    int rate_;
};

// ---------------------------------------------------------------------------
// TTS
// ---------------------------------------------------------------------------
class FakeTtsEngine : public TtsEngine {
   public:
    explicit FakeTtsEngine(const Config& cfg)
        : rate_(cfg.sample_rate > 0 ? cfg.sample_rate : 16000) {}

    void synthesize(const std::string& text, PcmCallback cb) override {
        if (abort_next_.exchange(false)) return;  // aborted before this piece began

        const bool first = (calls_ == 0);
        ++calls_;
        const int64_t ttfa = (int64_t)env_double("VOICE_FAKE_TTS_TTFA_MS", 350);
        const bool ttfa_every = env_int("VOICE_FAKE_TTS_TTFA_ALWAYS", 1) != 0;
        if (ttfa > 0 && (first || ttfa_every)) sleep_ms(ttfa);

        double speed = env_double("VOICE_FAKE_TTS_RATE", 4.0);
        if (speed <= 0.0) speed = 1.0;
        const double amp = env_double("VOICE_FAKE_TTS_AMP", 0.15);

        const int chunk = std::max(1, rate_ / 50);  // 20 ms
        int words = count_words(text);
        if (words <= 0) words = 2;
        int64_t duration_ms = (int64_t)words * 300;
        duration_ms = std::max<int64_t>(400, std::min<int64_t>(duration_ms, 8000));
        const int64_t chunk_ms = (int64_t)chunk * 1000 / rate_;
        const int64_t n_chunks = std::max<int64_t>(1, duration_ms / chunk_ms);
        const int64_t playback_sleep = (int64_t)((double)chunk_ms / speed);  // 4x => 5 ms

        const double w = 2.0 * kPi * 180.0 / (double)rate_;
        std::vector<float> pcm((size_t)chunk);
        const int64_t total = n_chunks * chunk;
        const int64_t fade = std::max<int64_t>(1, rate_ / 100);  // 10 ms
        for (int64_t c = 0; c < n_chunks; ++c) {
            for (int i = 0; i < chunk; ++i) {
                const int64_t n = c * chunk + i;
                // 180 Hz fundamental + 2nd harmonic, formant-ish and boring.
                double s = 0.7 * std::sin(phase_) + 0.3 * std::sin(2.0 * phase_);
                phase_ += w;
                if (phase_ > 2.0 * kPi) phase_ -= 2.0 * kPi;
                double env = 1.0;
                if (n < fade) env = (double)n / (double)fade;
                else if (total - n < fade) env = (double)(total - n) / (double)fade;
                pcm[(size_t)i] = (float)(amp * env * s);
            }
            const bool last = (c + 1 == n_chunks);
            if (cb) cb(pcm.data(), chunk, last);
            if (!last) sleep_ms(playback_sleep);
        }
    }

    // Arms the drop; returns true because this implementation does honour it
    // (the header's false return means "no abort hook"). The in-flight call is
    // never cut: the vtable has no abort flag, matching the real backend.
    bool abort_pending() override {
        abort_next_.store(true);
        return true;
    }

    bool ready() const override { return true; }

   private:
    int rate_;
    int calls_ = 0;
    double phase_ = 0.0;
    std::atomic<bool> abort_next_{false};
};

// ---------------------------------------------------------------------------
// Chat LLM
// ---------------------------------------------------------------------------
const char* kAckText = "Checking on that now.";
const char* kAnswerText = "The garage door is closed.";
const char* kFailureText = "I can't reach the garage service right now, so I can't confirm the door.";
const char* kFillers[] = {
    "One moment.",
    "Still working on it.",
    "Thanks for waiting.",
    "Nearly there.",
    "Let me keep checking.",
};
constexpr int kFillerCount = (int)(sizeof(kFillers) / sizeof(kFillers[0]));

class FakeLlmEngine : public LlmEngine {
   public:
    explicit FakeLlmEngine(const Config& /*cfg*/) {}

    LlmResult generate(const std::vector<ChatMessage>& messages, int max_tokens,
                       TokenCallback on_token,
                       const std::string& /*grammar*/) override {
        abort_.store(false);

        std::string system_text;
        std::string all_text;
        for (const auto& m : messages) {
            all_text += m.content;
            all_text.push_back('\n');
            if (m.role == "system") {
                system_text += m.content;
                system_text.push_back('\n');
            }
        }
        const std::string sys = lower_copy(system_text.empty() ? all_text : system_text);

        std::string text;
        if (sys.find("ack") != std::string::npos) {
            text = kAckText;
        } else if (sys.find("fail") != std::string::npos ||
                   sys.find("error") != std::string::npos) {
            text = kFailureText;
        } else if (sys.find("filler") != std::string::npos) {
            std::lock_guard<std::mutex> lock(mu_);
            text = kFillers[filler_cursor_ % kFillerCount];
            ++filler_cursor_;
        } else {
            text = kAnswerText;
        }

        LlmResult r;
        const int64_t t0 = steady_now();
        const int64_t ttft = (int64_t)env_double("VOICE_FAKE_LLM_TTFT_MS", 80);
        const int64_t tpt = (int64_t)env_double("VOICE_FAKE_LLM_TPT_MS", 12);

        if (!sleep_interruptible(ttft, abort_)) {
            r.aborted = true;
            r.ttft_ms = 0;
            r.total_ms = steady_now() - t0;
            return r;
        }
        r.ttft_ms = steady_now() - t0;

        // Split into word pieces carrying their trailing space.
        std::vector<std::string> pieces;
        size_t i = 0;
        while (i < text.size()) {
            size_t j = i;
            while (j < text.size() && text[j] != ' ') ++j;
            if (j < text.size()) ++j;  // include the separator
            pieces.push_back(text.substr(i, j - i));
            i = j;
        }

        const int limit = (max_tokens > 0) ? std::min<int>(max_tokens, (int)pieces.size())
                                           : (int)pieces.size();
        for (int k = 0; k < limit; ++k) {
            if (abort_.load()) { r.aborted = true; break; }
            if (tpt > 0) sleep_ms(tpt);
            r.text += pieces[(size_t)k];
            ++r.tokens;
            if (on_token && !on_token(pieces[(size_t)k])) { r.aborted = true; break; }
        }
        r.ok = !r.aborted;
        r.total_ms = steady_now() - t0;
        return r;
    }

    void request_abort() override { abort_.store(true); }

    bool ready() const override { return true; }

   private:
    std::atomic<bool> abort_{false};
    std::mutex mu_;
    int filler_cursor_ = 0;
};

}  // namespace

Engines make_fake_engines(const Config& cfg) {
    Engines e;
    e.asr = new FakeAsrEngine(cfg);
    e.tts = new FakeTtsEngine(cfg);
    e.llm = new FakeLlmEngine(cfg);
    e.owns = true;
    return e;
}

}  // namespace voice
}  // namespace crispasr
