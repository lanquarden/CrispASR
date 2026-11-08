// stt_driver.h — speech buffer + partial/final snapshots.
//
// Qwen3-ASR has no native cross-append streaming, so partials are produced by
// re-decoding a sliding window. The driver owns the buffer and the windows;
// the orchestrator owns the cadence and the arbiter.
#pragma once

#include <cstdint>
#include <vector>

namespace crispasr {
namespace voice {

class SttDriver {
   public:
    explicit SttDriver(int sample_rate, int window_ms, int max_turn_ms);

    void begin_turn(int64_t t_ms);
    void push(const float* pcm, int n_samples);
    void end_turn(int64_t t_ms);

    // Last `window_ms` of speech (or everything if shorter).
    const std::vector<float>& partial_window();
    // Everything in the turn, capped at max_turn_ms (oldest dropped first).
    const std::vector<float>& final_audio();

    int64_t speech_samples() const { return speech_samples_; }
    int64_t speech_ms() const;
    bool active() const { return active_; }
    void reset();

   private:
    int rate_;
    int window_ms_;
    int max_turn_ms_;
    bool active_ = false;
    int64_t speech_samples_ = 0;
    std::vector<float> buf_;
    std::vector<float> view_;
};

}  // namespace voice
}  // namespace crispasr
