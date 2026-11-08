// stt_driver.cpp — the turn buffer behind rolling re-transcription.
//
// The buffer is bounded by max_turn_ms: a stuck VAD must not grow the process to
// gigabytes, and the final decode has to stay inside the ASR job budget. Older
// speech is dropped first, which is the right trade for a turn whose last words
// are the ones that matter.

#include "stt_driver.h"

#include <algorithm>
#include <cmath>

namespace crispasr {
namespace voice {

namespace {

void trim_front(std::vector<float>& v, size_t cap) {
    if (v.size() > cap) v.erase(v.begin(), v.begin() + (long)(v.size() - cap));
}

}  // namespace

SttDriver::SttDriver(int sample_rate, int window_ms, int max_turn_ms)
    : rate_(sample_rate > 0 ? sample_rate : 16000),
      window_ms_(window_ms > 0 ? window_ms : 1),
      max_turn_ms_(max_turn_ms > 0 ? max_turn_ms : 1) {
    const size_t cap = (size_t)(((int64_t)rate_ * (int64_t)max_turn_ms_) / 1000);
    buf_.reserve(cap > 0 ? cap : 1);
}

void SttDriver::begin_turn(int64_t t_ms) {
    (void)t_ms;  // the buffer is window-relative; absolute time lives in events
    active_ = true;
    speech_samples_ = 0;
    buf_.clear();
    view_.clear();
}

void SttDriver::push(const float* pcm, int n_samples) {
    if (!pcm || n_samples <= 0) return;
    // speech_samples_ counts everything offered in this turn, including the
    // samples the cap evicted: it is "how long did the user talk", not "how much
    // survived in the buffer".
    speech_samples_ += n_samples;
    buf_.insert(buf_.end(), pcm, pcm + n_samples);

    const size_t cap = (size_t)std::max<int64_t>(1, ((int64_t)rate_ * (int64_t)max_turn_ms_) / 1000);
    trim_front(buf_, cap);
}

void SttDriver::end_turn(int64_t t_ms) {
    (void)t_ms;
    active_ = false;
}

const std::vector<float>& SttDriver::partial_window() {
    const size_t want = (size_t)std::max<int64_t>(1, ((int64_t)rate_ * (int64_t)window_ms_) / 1000);
    const size_t have = std::min(want, buf_.size());
    view_.assign(buf_.end() - (long)have, buf_.end());
    // Stable reference: valid until the next partial_window()/reset() call, so
    // the caller can hand it to the arbiter without a copy.
    return view_;
}

const std::vector<float>& SttDriver::final_audio() {
    const size_t cap = (size_t)std::max<int64_t>(1, ((int64_t)rate_ * (int64_t)max_turn_ms_) / 1000);
    trim_front(buf_, cap);
    return buf_;
}

int64_t SttDriver::speech_ms() const {
    if (rate_ <= 0) return 0;
    return (int64_t)std::llround((double)speech_samples_ * 1000.0 / (double)rate_);
}

void SttDriver::reset() {
    buf_.clear();
    view_.clear();
    speech_samples_ = 0;
    active_ = false;
}

}  // namespace voice
}  // namespace crispasr
