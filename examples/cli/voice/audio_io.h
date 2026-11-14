// audio_io.h — PCM buffer, real-time pacing and file-backed sources/sinks.
//
// M1 uses WAV in / WAV out with wall-clock pacing, so the harness and a human
// get identical timing. ALSA and WebSocket endpoints plug into the same
// Source/Sink interfaces later without touching the loop.
#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <functional>
#include <vector>

namespace crispasr {
namespace voice {

int64_t now_ms();
int64_t steady_ms();

// A frame source. Two implementations exist: the paced WAV reader used by the
// harness (identical timing for the machine and the human) and a live
// microphone. `read` returns samples, 0 for end of stream; `overrun_ms` is
// pacing honesty and is only meaningful for a file-backed source.
struct FrameSource {
    virtual ~FrameSource() {}
    virtual int read(float* out, int n, int64_t* overrun_ms) = 0;
    virtual bool eof() const = 0;
};

// A PCM sink: a WAV file, or a speaker. The sink thread writes through this, so
// the loop never needs to know which one it is talking to.
struct AudioOut {
    virtual ~AudioOut() {}
    virtual void write(const float* pcm, int n) = 0;
};

// Single-producer / single-consumer PCM ring (float32 mono). Bounded: a push
// that would exceed capacity drops the oldest samples and reports it.
class PcmRing {
   public:
    explicit PcmRing(int capacity_samples);
    int push(const float* pcm, int n);        // returns samples written
    int pop(float* out, int n);               // returns samples read
    int available() const;
    int capacity() const;
    int64_t buffer_ms(int sample_rate) const;
    void clear();
    int64_t dropped_samples() const { return dropped_; }

   private:
    mutable std::mutex mu_;
    std::vector<float> buf_;
    size_t head_ = 0, tail_ = 0, size_ = 0;
    int64_t dropped_ = 0;
};

// Reads 16-bit PCM WAV (mono or stereo, any rate) and resamples to the target
// rate with a naive box filter (fixtures are authored at 16 kHz).
class WavReader {
   public:
    bool open(const std::string& path, int target_rate, std::string* err);
    // Returns samples read; 0 = EOF.
    int read(float* out, int n);
    int sample_rate() const { return rate_; }
    bool eof() const { return eof_; }
    void close();

   private:
    FILE* f_ = nullptr;
    int rate_ = 0, channels_ = 1, src_rate_ = 0;
    int64_t data_bytes_ = 0, data_read_ = 0;
    bool eof_ = false;
    std::vector<float> resample_pending_;
};

// Writes 16-bit mono WAV. finish() patches the RIFF sizes.
class WavWriter : public AudioOut {
   public:
    bool open(const std::string& path, int rate, std::string* err);
    void write(const float* pcm, int n) override;
    void finish();
    ~WavWriter() override;

   private:
    FILE* f_ = nullptr;
    int rate_ = 0;
    int64_t frames_ = 0;
    uint32_t data_bytes_ = 0;
};

// A real-time sink: pops `frame_samples` every `frame_ms`, writes to the
// writer, and tracks how much audio has been "played". The pacing clock is
// wall time, which is what the harness measures against.
class PacedSink {
   public:
    PacedSink(PcmRing* ring, AudioOut* out, int sample_rate, int frame_samples);
    // Runs until stop() is called. Calls on_starved(absent_ms) whenever a frame
    // could not be filled. Returns after stop.
    void run(const std::function<void(int64_t)>& on_starved);
    void stop() { running_ = false; }
    int64_t played_samples() const { return played_; }
    // Samples of real audio emitted (excludes inserted silence).
    int64_t real_played_samples() const { return real_played_; }
    int64_t real_played_ms() const { return real_played_ * 1000 / rate_; }
    int64_t played_ms() const;

   private:
    PcmRing* ring_;
    AudioOut* out_;
    int rate_, frame_;
    std::atomic<bool> running_{true};
    std::atomic<int64_t> played_{0};       // includes silence
    std::atomic<int64_t> real_played_{0};  // real audio only
};

// Input pacing: reads a frame every frame_ms from wall clock, sleeping as
// needed. Honest about overruns: reports them via on_overrun.
class PacedSource : public FrameSource {
   public:
    PacedSource(WavReader* reader, int sample_rate, int frame_samples);
    // Returns samples read (0 at EOF). Sleeps to keep real-time pace.
    int read(float* out, int n, int64_t* overrun_ms) override;
    bool eof() const override;

   private:
    WavReader* reader_;
    int rate_, frame_;
    int64_t started_ms_ = 0;
    int64_t samples_read_ = 0;
};

}  // namespace voice
}  // namespace crispasr
