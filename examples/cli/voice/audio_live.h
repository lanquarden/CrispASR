// audio_live.h — microphone input and speaker output for the live voice loop
// (home-ops#584).
//
// The device is not linked in: the loop spawns the system's PCM tools
// (arecord/aplay by default, pw-record/pw-cat or anything else via
// VOICE_MIC_CMD / VOICE_SPEAKER_CMD) and talks raw S16LE mono 16 kHz over a
// pipe. That keeps the loop free of ALSA/PipeWire headers, keeps the fake-engine
// selftest device-free, and makes the audio path inspectable with the exact same
// command a shell script would run.
//
// Both classes are deliberately thin: they own one child process and convert
// between int16 and the float32 the loop uses everywhere else.
#pragma once

#include <atomic>
#include <cstdint>
#include <string>
#include <sys/types.h>
#include <vector>

#include "audio_io.h"

namespace crispasr {
namespace voice {

// Live microphone. read() blocks until the device delivers the requested frame,
// so the device itself is the pacing clock: no sleeping, no overrun report.
class LiveMicSource : public FrameSource {
   public:
    LiveMicSource() = default;
    ~LiveMicSource() override;

    // `device` is an ALSA/PipeWire name ("default", "hw:2,0", ...); empty means
    // the system default. `frame` is the read quantum in samples.
    bool open(const std::string& device, int sample_rate, int frame_samples,
              std::string* err);

    // Returns samples read; 0 when the capture process has exited. Never
    // returns a short frame: a partial read is completed before returning, so
    // the detector always sees the frame_ms its hysteresis math assumes.
    int read(float* out, int n, int64_t* overrun_ms) override;
    bool eof() const override { return eof_.load(); }
    void close();

   private:
    int fd_ = -1;
    pid_t pid_ = -1;
    int rate_ = 16000;
    std::atomic<bool> eof_{false};
};

// Live speaker. write() converts to S16LE and pushes into the player's stdin.
// A dead player is not fatal: the loop keeps running and the audio is dropped,
// which is what a broken output device should look like.
class LiveSpeakerSink : public AudioOut {
   public:
    LiveSpeakerSink() = default;
    ~LiveSpeakerSink() override;

    bool open(const std::string& device, int sample_rate, int frame_samples,
              std::string* err);
    void write(const float* pcm, int n) override;
    void close();
    bool dead() const { return dead_.load(); }

   private:
    int fd_ = -1;
    pid_t pid_ = -1;
    std::atomic<bool> dead_{false};
    std::vector<int16_t> scratch_;
};

// The commands actually spawned, for the log line: the loop prints them once so
// a bad device name is diagnosable from the terminal.
std::string mic_command(const std::string& device, int rate);
std::string speaker_command(const std::string& device, int rate);

}  // namespace voice
}  // namespace crispasr
