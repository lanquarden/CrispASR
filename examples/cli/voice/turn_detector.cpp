// turn_detector.cpp — RMS energy VAD with start/stop hysteresis.
//
// Deterministic on purpose: the fixtures are clean speech with known silences,
// so a threshold plus two run-lengths is enough and every decision is
// reproducible. A model VAD (firered/silero) can replace the body later behind
// the same interface.

#include "turn_detector.h"

#include <cmath>

namespace crispasr {
namespace voice {

TurnDetector::TurnDetector(int sample_rate, int min_speech_ms, int min_silence_ms, float energy_threshold)
    : rate_(sample_rate > 0 ? sample_rate : 16000),
      min_speech_ms_(min_speech_ms < 0 ? 0 : min_speech_ms),
      min_silence_ms_(min_silence_ms < 0 ? 0 : min_silence_ms),
      threshold_(energy_threshold < 0.0f ? 0.0f : energy_threshold) {}

float TurnDetector::rms(const float* frame, int n) const {
    if (!frame || n <= 0) return 0.0f;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        const double v = (double)frame[i];
        acc += v * v;
    }
    return (float)std::sqrt(acc / (double)n);
}

void TurnDetector::push(const float* frame, int n_samples, int64_t t_ms, bool assistant_floor,
                        const TurnEvents& ev) {
    if (!frame || n_samples <= 0) return;

    // Duration of this frame, rounded: a 320-sample quantum at 16 kHz is 20 ms
    // exactly, but the loop must not silently treat an odd frame as 0 ms or the
    // hysteresis run-lengths would never expire.
    frame_ms_ = (int64_t)std::llround((double)n_samples * 1000.0 / (double)rate_);
    if (frame_ms_ < 1) frame_ms_ = 1;

    // Real speech (and the fixtures that mimic it) dips below the threshold
    // between syllables. A 120 ms hangover bridges those dips without letting
    // a genuine end-of-turn silence through: min_silence_ms is far longer.
    const bool raw_loud = rms(frame, n_samples) > threshold_;
    if (raw_loud) last_loud_ms_ = t_ms;
    const bool loud = raw_loud || (t_ms - last_loud_ms_ <= 120);

    if (!speech_) {
        if (!loud) {
            speech_run_ms_ = 0;
            return;
        }
        speech_run_ms_ += frame_ms_;
        if (speech_run_ms_ < (int64_t)min_speech_ms_) return;

        // Speech confirmed. This is also the only place a barge-in can be born:
        // the user started talking while the assistant owned the floor.
        speech_ = true;
        silence_run_ms_ = 0;
        speech_run_ms_ = 0;
        if (ev.on_speech_start) ev.on_speech_start(t_ms);
        if (assistant_floor && !barge_reported_) {
            barge_reported_ = true;
            if (ev.on_barge_in) ev.on_barge_in(t_ms);
        }
        return;
    }

    // Speech active: only sustained silence ends it.
    if (loud) {
        silence_run_ms_ = 0;
        return;
    }
    silence_run_ms_ += frame_ms_;
    if (silence_run_ms_ < (int64_t)min_silence_ms_) return;

    speech_ = false;
    silence_run_ms_ = 0;
    speech_run_ms_ = 0;
    // Re-arm barge-in: one report per utterance, and the next utterance gets a
    // fresh chance (the assistant may be talking again by then).
    barge_reported_ = false;
    if (ev.on_speech_end) ev.on_speech_end(t_ms);
}

void TurnDetector::reset() {
    speech_ = false;
    speech_run_ms_ = 0;
    silence_run_ms_ = 0;
    frame_ms_ = 0;
    barge_reported_ = false;
}

}  // namespace voice
}  // namespace crispasr
