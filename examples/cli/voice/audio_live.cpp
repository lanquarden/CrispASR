// audio_live.cpp — microphone capture and speaker playback as child processes.
//
// Why a child process instead of linking ALSA: the loop only needs one raw PCM
// stream per direction, the host tool handles device selection and PipeWire
// routing, and a wedged device kills a child instead of the experiment. The
// cost is one memcpy-free pipe hop, which is ~0.1 ms per 20 ms frame.

#include "audio_live.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

namespace crispasr {
namespace voice {
namespace {

bool debug() { return std::getenv("VOICE_DEBUG") != nullptr; }

// Spawns `command` with one pipe end wired to stdin or stdout. The command is
// run through /bin/sh -c so an override like VOICE_SPEAKER_CMD='tee out.raw' or
// a device name with spaces behaves the way a shell user expects. Returns the
// parent's fd, or -1 with *err filled.
int spawn_piped(const std::string& command, bool read_from_child, pid_t* pid_out,
                std::string* err) {
    int fds[2];
    if (pipe(fds) != 0) {
        if (err) *err = std::string("pipe: ") + std::strerror(errno);
        return -1;
    }
    // The child end must not leak into the other direction, and the parent end
    // must be close-on-exec so a later spawn does not inherit it.
    const int child_fd = read_from_child ? fds[1] : fds[0];
    const int parent_fd = read_from_child ? fds[0] : fds[1];
    fcntl(parent_fd, F_SETFD, FD_CLOEXEC);

    const pid_t pid = fork();
    if (pid < 0) {
        if (err) *err = std::string("fork: ") + std::strerror(errno);
        close(fds[0]);
        close(fds[1]);
        return -1;
    }
    if (pid == 0) {
        // Child: wire the pipe to the requested direction and exec.
        if (read_from_child) {
            if (dup2(child_fd, STDOUT_FILENO) < 0) _exit(127);
        } else {
            if (dup2(child_fd, STDIN_FILENO) < 0) _exit(127);
        }
        close(fds[0]);
        close(fds[1]);
        if (!debug()) {
            const int devnull = open("/dev/null", O_WRONLY);
            if (devnull >= 0) {
                dup2(devnull, STDERR_FILENO);
                if (devnull > STDERR_FILENO) close(devnull);
            }
        }
        std::vector<std::string> args = {"/bin/sh", "-c", command};
        std::vector<char*> cargv;
        cargv.reserve(args.size() + 1);
        for (const std::string& a : args) cargv.push_back(const_cast<char*>(a.c_str()));
        cargv.push_back(nullptr);
        execvp(cargv[0], cargv.data());
        _exit(127);  // exec failed: the parent sees EOF/EIO and reports it
    }
    close(child_fd);
    *pid_out = pid;
    return parent_fd;
}

// SIGTERM, then SIGKILL if the tool does not let go. The audio tools exit on
// EOF/SIGTERM promptly; the escalation is for a wedged device.
void reap(pid_t pid) {
    if (pid <= 0) return;
    int status = 0;
    if (waitpid(pid, &status, WNOHANG) == 0) {
        kill(pid, SIGTERM);
        for (int i = 0; i < 40; ++i) {
            if (waitpid(pid, &status, WNOHANG) == pid) return;
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        kill(pid, SIGKILL);
        waitpid(pid, &status, 0);
    }
}

int16_t to_s16(float v) {
    if (v > 1.0f) v = 1.0f;
    if (v < -1.0f) v = -1.0f;
    return (int16_t)std::lrintf(v * 32767.0f);
}

}  // namespace

std::string mic_command(const std::string& device, int rate) {
    if (const char* env = std::getenv("VOICE_MIC_CMD")) return env;
    std::string cmd = "arecord -q -t raw -f S16_LE -r " + std::to_string(rate) +
                      " -c 1";
    if (!device.empty()) cmd += " -D \"" + device + "\"";
    return cmd;
}

std::string speaker_command(const std::string& device, int rate) {
    if (const char* env = std::getenv("VOICE_SPEAKER_CMD")) return env;
    // -B/-F pin an 80 ms ALSA buffer with a 20 ms period. The writer paces at
    // exactly real time, so a deep default buffer would only add output latency
    // to every measurement the loop reports.
    std::string cmd = "aplay -q -t raw -f S16_LE -r " + std::to_string(rate) +
                      " -c 1 -B 80000 -F 20000";
    if (!device.empty()) cmd += " -D \"" + device + "\"";
    return cmd;
}

// ---------------------------------------------------------------------------
// LiveMicSource
// ---------------------------------------------------------------------------
bool LiveMicSource::open(const std::string& device, int sample_rate,
                         int frame_samples, std::string* err) {
    (void)frame_samples;
    rate_ = sample_rate > 0 ? sample_rate : 16000;
    const std::string command = mic_command(device, rate_);
    if (command.empty()) {
        if (err) *err = "empty microphone command";
        return false;
    }
    fd_ = spawn_piped(command, true, &pid_, err);
    if (fd_ < 0) return false;
    if (debug())
        std::fprintf(stderr, "[live] mic pid=%d cmd='%s'\n", (int)pid_,
                     command.c_str());
    return true;
}

int LiveMicSource::read(float* out, int n, int64_t* overrun_ms) {
    if (overrun_ms) *overrun_ms = 0;
    if (fd_ < 0 || n <= 0) return 0;
    // Complete the frame: the detector's hysteresis is expressed in whole
    // frames, and a short read would silently shorten the frame_ms it computes.
    int16_t raw[2048];
    if (n > (int)(sizeof(raw) / sizeof(raw[0]))) n = (int)(sizeof(raw) / sizeof(raw[0]));
    int done = 0;
    while (done < n) {
        const ssize_t got = ::read(fd_, raw + done, (size_t)(n - done) * sizeof(int16_t));
        if (got == 0) {  // capture process exited
            eof_.store(true);
            break;
        }
        if (got < 0) {
            if (errno == EINTR) continue;
            eof_.store(true);
            break;
        }
        done += (int)(got / (ssize_t)sizeof(int16_t));
    }
    for (int i = 0; i < done; ++i) out[i] = (float)raw[i] / 32768.0f;
    return done;
}

void LiveMicSource::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    reap(pid_);
    pid_ = -1;
}

LiveMicSource::~LiveMicSource() { close(); }

// ---------------------------------------------------------------------------
// LiveSpeakerSink
// ---------------------------------------------------------------------------
bool LiveSpeakerSink::open(const std::string& device, int sample_rate,
                           int frame_samples, std::string* err) {
    (void)sample_rate;
    // A player that dies must not take the loop down with it.
    signal(SIGPIPE, SIG_IGN);
    const std::string command =
        speaker_command(device, sample_rate > 0 ? sample_rate : 16000);
    if (command.empty()) {
        if (err) *err = "empty speaker command";
        return false;
    }
    fd_ = spawn_piped(command, false, &pid_, err);
    if (fd_ < 0) return false;
    scratch_.resize((size_t)(frame_samples > 0 ? frame_samples : 512));
    if (debug())
        std::fprintf(stderr, "[live] speaker pid=%d cmd='%s'\n", (int)pid_,
                     command.c_str());
    return true;
}

void LiveSpeakerSink::write(const float* pcm, int n) {
    if (fd_ < 0 || n <= 0 || dead_.load()) return;
    if ((int)scratch_.size() < n) scratch_.resize((size_t)n);
    for (int i = 0; i < n; ++i) scratch_[i] = to_s16(pcm[i]);
    const char* p = reinterpret_cast<const char*>(scratch_.data());
    size_t left = (size_t)n * sizeof(int16_t);
    while (left > 0) {
        const ssize_t put = ::write(fd_, p, left);
        if (put <= 0) {
            if (put < 0 && errno == EINTR) continue;
            // Broken player: drop the audio and let the loop keep its clock.
            dead_.store(true);
            if (debug()) std::fprintf(stderr, "[live] speaker died (write failed)\n");
            return;
        }
        p += put;
        left -= (size_t)put;
    }
}

void LiveSpeakerSink::close() {
    if (fd_ >= 0) {
        ::close(fd_);  // EOF: aplay drains its buffer and exits
        fd_ = -1;
    }
    reap(pid_);
    pid_ = -1;
}

LiveSpeakerSink::~LiveSpeakerSink() { close(); }

}  // namespace voice
}  // namespace crispasr
