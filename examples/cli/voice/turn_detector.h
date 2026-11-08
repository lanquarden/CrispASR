// turn_detector.h — energy VAD with speech start/stop hysteresis.
//
// Deliberately simple and deterministic: fixtures are clean speech with known
// silences. A model VAD (firered/silero) can replace it later behind the same
// interface; the loop only cares about the events.
#pragma once

#include <cstdint>
#include <functional>

namespace crispasr {
namespace voice {

struct TurnEvents {
    std::function<void(int64_t t_ms)> on_speech_start;
    std::function<void(int64_t t_ms)> on_speech_end;
    // Fired when speech is detected while the assistant owns the floor.
    std::function<void(int64_t t_ms)> on_barge_in;
};

class TurnDetector {
   public:
    TurnDetector(int sample_rate, int min_speech_ms, int min_silence_ms,
                 float energy_threshold);
    // Feed one frame. `assistant_floor` gates barge-in detection.
    void push(const float* frame, int n_samples, int64_t t_ms,
              bool assistant_floor, const TurnEvents& ev);
    void reset();
    bool speech_active() const { return speech_; }

   private:
    float rms(const float* frame, int n) const;

    int rate_;
    int min_speech_ms_, min_silence_ms_;
    float threshold_;

    bool speech_ = false;
    int64_t speech_run_ms_ = 0;
    int64_t silence_run_ms_ = 0;
    int64_t frame_ms_ = 0;
    bool barge_reported_ = false;
    int64_t last_loud_ms_ = -100000;  // hangover: bridge short dips inside words
};

}  // namespace voice
}  // namespace crispasr
