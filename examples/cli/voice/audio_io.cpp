// audio_io.cpp — PCM ring, WAV source/sink, wall-clock pacing.
//
// No device, no network, no CrispASR: only <cstdio>/<chrono>/<thread>. The
// harness and the interactive CLI share this code so a human and an automated
// scenario observe identical timing.

#include <functional>  // audio_io.h uses std::function without including it

#include "audio_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <thread>
#include <utility>

namespace crispasr {
namespace voice {

namespace {

using Clock = std::chrono::steady_clock;

int64_t ms_of(int64_t samples, int rate) {
    if (rate <= 0) return 0;
    // Round rather than truncate: a 160-sample frame at 16 kHz is exactly 10 ms,
    // but odd quantum/rate combinations must not report 0 ms to the metric.
    return (int64_t)std::llround((double)samples * 1000.0 / (double)rate);
}

// Sleep until `deadline`, in small slices so stop() is honoured within a frame
// instead of after the full frame duration.
void sleep_until_cancellable(Clock::time_point deadline, const std::atomic<bool>& running) {
    const Clock::duration slice = std::chrono::milliseconds(2);
    for (;;) {
        if (!running.load()) return;
        Clock::time_point now = Clock::now();
        if (now >= deadline) return;
        Clock::duration left = deadline - now;
        std::this_thread::sleep_for(left < slice ? left : slice);
    }
}

void sleep_until_ms(int64_t deadline_ms) {
    int64_t now = steady_ms();
    if (now >= deadline_ms) return;
    std::this_thread::sleep_for(std::chrono::milliseconds(deadline_ms - now));
}

uint32_t rd_u32(const unsigned char* p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

uint16_t rd_u16(const unsigned char* p) { return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8)); }

void wr_u32(FILE* f, uint32_t v) {
    unsigned char b[4] = {(unsigned char)(v & 0xFF), (unsigned char)((v >> 8) & 0xFF),
                          (unsigned char)((v >> 16) & 0xFF), (unsigned char)((v >> 24) & 0xFF)};
    fwrite(b, 1, 4, f);
}

void wr_u16(FILE* f, uint16_t v) {
    unsigned char b[2] = {(unsigned char)(v & 0xFF), (unsigned char)((v >> 8) & 0xFF)};
    fwrite(b, 1, 2, f);
}

// Naive area (box) resampler over one fully buffered channel. Each output bin
// averages the source samples it overlaps, weighted by the overlap, which
// degrades gracefully to a plain average when downsampling and to a
// zero-order/linear blend when upsampling. Fixtures are authored at the target
// rate, so this only ever runs for ad-hoc WAVs.
void resample_box(const std::vector<float>& src, int src_rate, int dst_rate, std::vector<float>& dst) {
    dst.clear();
    if (src.empty()) return;
    if (src_rate <= 0 || dst_rate <= 0 || src_rate == dst_rate) {
        dst = src;
        return;
    }
    const double step = (double)src_rate / (double)dst_rate;
    const size_t n = src.size();
    const size_t out_n = (size_t)std::llround((double)n / step);
    if (out_n == 0) {
        dst.push_back(src[0]);
        return;
    }
    dst.resize(out_n, 0.0f);
    for (size_t k = 0; k < out_n; ++k) {
        const double pos0 = (double)k * step;
        const double pos1 = pos0 + step;
        int64_t i0 = (int64_t)std::floor(pos0);
        int64_t i1 = (int64_t)std::ceil(pos1);
        if (i0 < 0) i0 = 0;
        if (i1 > (int64_t)n) i1 = (int64_t)n;
        double acc = 0.0, wsum = 0.0;
        for (int64_t i = i0; i < i1; ++i) {
            const double lo = std::max(pos0, (double)i);
            const double hi = std::min(pos1, (double)(i + 1));
            const double w = hi - lo;
            if (w <= 0.0) continue;
            acc += w * (double)src[(size_t)i];
            wsum += w;
        }
        if (wsum <= 0.0) {
            int64_t idx = (int64_t)std::llround(pos0);
            if (idx < 0) idx = 0;
            if (idx >= (int64_t)n) idx = (int64_t)n - 1;
            dst[k] = src[(size_t)idx];
        } else {
            dst[k] = (float)(acc / wsum);
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// PcmRing — bounded SPSC.
//
// Overflow policy: drop-oldest. The producer never blocks, because blocking the
// capture thread on a starved playback path would create the very gap the
// metrics are trying to measure. Evicted samples are counted in dropped_ so the
// turn record can prove whether a gap was caused by PCM loss.
// ---------------------------------------------------------------------------
PcmRing::PcmRing(int capacity_samples) {
    if (capacity_samples < 1) capacity_samples = 1;
    buf_.assign((size_t)capacity_samples, 0.0f);
}

int PcmRing::push(const float* pcm, int n) {
    if (!pcm || n <= 0) return 0;
    std::lock_guard<std::mutex> lk(mu_);
    const size_t cap = buf_.size();

    // A push larger than the whole ring cannot be stored; keep its tail (newest)
    // and count the rest as dropped.
    if ((size_t)n > cap) {
        int64_t drop = (int64_t)n - (int64_t)cap;
        dropped_ += drop;
        pcm += drop;
        n = (int)cap;
    }

    // Make room by evicting the oldest queued samples.
    const size_t want = (size_t)n;
    if (size_ + want > cap) {
        size_t evict = size_ + want - cap;
        head_ = (head_ + evict) % cap;
        size_ -= evict;
        dropped_ += (int64_t)evict;
    }

    for (int i = 0; i < n; ++i) {
        buf_[tail_] = pcm[i];
        tail_ = (tail_ + 1) % cap;
    }
    size_ += want;
    return n;
}

int PcmRing::pop(float* out, int n) {
    if (!out || n <= 0) return 0;
    std::lock_guard<std::mutex> lk(mu_);
    const size_t cap = buf_.size();
    size_t take = std::min((size_t)n, size_);
    for (size_t i = 0; i < take; ++i) {
        out[i] = buf_[head_];
        head_ = (head_ + 1) % cap;
    }
    size_ -= take;
    return (int)take;
}

int PcmRing::available() const {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)size_;
}

int PcmRing::capacity() const {
    std::lock_guard<std::mutex> lk(mu_);
    return (int)buf_.size();
}

int64_t PcmRing::buffer_ms(int sample_rate) const {
    if (sample_rate <= 0) return 0;
    std::lock_guard<std::mutex> lk(mu_);
    return ms_of((int64_t)size_, sample_rate);
}

void PcmRing::clear() {
    std::lock_guard<std::mutex> lk(mu_);
    head_ = tail_ = size_ = 0;
    // dropped_ is a lifetime counter: a barge-in flush must not erase the
    // evidence that the ring overflowed earlier in the turn.
}

// ---------------------------------------------------------------------------
// WavReader
//
// The whole data chunk is decoded and resampled at open(): fixtures are a few
// seconds of 16-bit PCM, and doing it up front removes fractional resampler
// state from read(), which keeps replay exactly reproducible at any frame size.
// ---------------------------------------------------------------------------
bool WavReader::open(const std::string& path, int target_rate, std::string* err) {
    close();
    if (target_rate <= 0) {
        if (err) *err = "WavReader: target_rate must be > 0";
        return false;
    }
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open wav: " + path;
        return false;
    }

    auto fail = [&](const std::string& msg) {
        fclose(f);
        if (err) *err = path + ": " + msg;
        return false;
    };

    unsigned char hdr[12];
    if (fread(hdr, 1, 12, f) != 12) return fail("truncated RIFF header");
    if (memcmp(hdr, "RIFF", 4) != 0 || memcmp(hdr + 8, "WAVE", 4) != 0) return fail("not a RIFF/WAVE file");

    bool have_fmt = false;
    uint16_t fmt_code = 0, channels = 0, bits = 0;
    uint32_t src_rate = 0;
    long data_off = 0;
    uint32_t data_size = 0;

    for (;;) {
        unsigned char ch[8];
        if (fread(ch, 1, 8, f) != 8) break;  // no more chunks
        uint32_t size = rd_u32(ch + 4);
        if (memcmp(ch, "fmt ", 4) == 0) {
            std::vector<unsigned char> body(size < 16 ? 16 : size);
            size_t got = fread(body.data(), 1, body.size(), f);
            if (got < 16) return fail("truncated fmt chunk");
            fmt_code = rd_u16(body.data() + 0);
            channels = rd_u16(body.data() + 2);
            src_rate = rd_u32(body.data() + 4);
            bits = rd_u16(body.data() + 14);
            have_fmt = true;
            if ((size & 1u) && fseek(f, 1, SEEK_CUR) != 0) return fail("bad fmt padding");
        } else if (memcmp(ch, "data", 4) == 0) {
            data_off = ftell(f);
            data_size = size;
            if (data_size == 0xFFFFFFFFu) {
                // Streaming writers leave the size open; take what is on disk.
                if (fseek(f, 0, SEEK_END) != 0) return fail("seek failed");
                long end = ftell(f);
                data_off = data_off < 0 ? 0 : data_off;
                data_size = (uint32_t)std::max<long>(0, end - data_off);
            }
            if (fseek(f, data_off, SEEK_SET) != 0) return fail("seek to data failed");
            if (have_fmt) break;
        } else {
            // LIST/fact/bext/...: skip, honouring the RIFF pad byte.
            long skip = (long)size + (long)(size & 1u);
            if (fseek(f, skip, SEEK_CUR) != 0) return fail("truncated chunk");
        }
    }

    if (!have_fmt) return fail("missing fmt chunk");
    if (data_off <= 0) return fail("missing data chunk");
    // 1 = PCM, 0xFFFE = WAVE_FORMAT_EXTENSIBLE (16-bit PCM in practice).
    if (fmt_code != 1 && fmt_code != 0xFFFE) return fail("unsupported audio format (need PCM)");
    if (bits != 16) return fail("unsupported bit depth (need 16-bit PCM)");
    if (channels < 1 || channels > 8) return fail("unsupported channel count");
    if (src_rate < 1000 || src_rate > 384000) return fail("implausible sample rate");

    data_bytes_ = (int64_t)data_size;
    data_read_ = 0;

    const int64_t frames = data_bytes_ / (int64_t)(channels * 2);
    if (frames <= 0) {
        // Valid but empty: behave as an immediately-EOF reader.
        resample_pending_.clear();
        eof_ = true;
        f_ = f;
        rate_ = target_rate;
        src_rate_ = (int)src_rate;
        channels_ = (int)channels;
        return true;
    }

    std::vector<int16_t> raw((size_t)frames * channels);
    if (fseek(f, data_off, SEEK_SET) != 0) return fail("seek to data failed");
    size_t want = raw.size() * sizeof(int16_t);
    size_t got = fread(raw.data(), 1, want, f);
    if (got < want) raw.resize(got / sizeof(int16_t));
    fclose(f);
    f = nullptr;

    const int64_t usable = (int64_t)raw.size() / (int64_t)channels;
    std::vector<float> mono((size_t)usable, 0.0f);
    for (int64_t i = 0; i < usable; ++i) {
        int32_t acc = 0;
        for (int c = 0; c < (int)channels; ++c) acc += (int32_t)raw[(size_t)i * channels + c];
        mono[(size_t)i] = (float)acc / ((float)channels * 32768.0f);
    }

    resample_box(mono, (int)src_rate, target_rate, resample_pending_);

    rate_ = target_rate;
    src_rate_ = (int)src_rate;
    channels_ = (int)channels;
    f_ = nullptr;  // decoded; read() serves from resample_pending_
    eof_ = resample_pending_.empty();
    data_read_ = 0;
    return true;
}

int WavReader::read(float* out, int n) {
    if (!out || n <= 0) return 0;
    if (eof_) return 0;
    const size_t total = resample_pending_.size();
    size_t pos = (size_t)data_read_;
    if (pos >= total) {
        eof_ = true;
        return 0;
    }
    size_t take = std::min((size_t)n, total - pos);
    memcpy(out, resample_pending_.data() + pos, take * sizeof(float));
    data_read_ = (int64_t)(pos + take);
    if ((size_t)data_read_ >= total) eof_ = true;
    return (int)take;
}

void WavReader::close() {
    if (f_) {
        fclose(f_);
        f_ = nullptr;
    }
    resample_pending_.clear();
    data_bytes_ = 0;
    data_read_ = 0;
    eof_ = true;
    channels_ = 1;
    rate_ = 0;
    src_rate_ = 0;
}

// ---------------------------------------------------------------------------
// WavWriter — canonical 44-byte header, sizes patched in finish().
// ---------------------------------------------------------------------------
bool WavWriter::open(const std::string& path, int rate, std::string* err) {
    finish();
    if (f_) {
        fclose(f_);
        f_ = nullptr;
    }
    if (rate <= 0) {
        if (err) *err = "WavWriter: rate must be > 0";
        return false;
    }
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        if (err) *err = "cannot write wav: " + path;
        return false;
    }
    fwrite("RIFF", 1, 4, f);
    wr_u32(f, 36);  // patched in finish()
    fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f);
    wr_u32(f, 16);
    wr_u16(f, 1);                                        // PCM
    wr_u16(f, 1);                                        // mono
    wr_u32(f, (uint32_t)rate);
    wr_u32(f, (uint32_t)rate * 2);                       // byte rate
    wr_u16(f, 2);                                        // block align
    wr_u16(f, 16);                                       // bits per sample
    fwrite("data", 1, 4, f);
    wr_u32(f, 0);  // patched in finish()
    if (fflush(f) != 0) {
        fclose(f);
        if (err) *err = "cannot write wav header: " + path;
        return false;
    }
    f_ = f;
    rate_ = rate;
    frames_ = 0;
    data_bytes_ = 0;
    return true;
}

void WavWriter::write(const float* pcm, int n) {
    if (!f_ || !pcm || n <= 0) return;
    int16_t chunk[1024];
    int written = 0;
    while (written < n) {
        int m = std::min(n - written, (int)(sizeof(chunk) / sizeof(chunk[0])));
        for (int i = 0; i < m; ++i) {
            float v = pcm[written + i];
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            chunk[i] = (int16_t)std::lrintf(v * 32767.0f);
        }
        size_t bytes = (size_t)m * sizeof(int16_t);
        size_t put = fwrite(chunk, 1, bytes, f_);
        data_bytes_ += (uint32_t)put;
        written += m;
        if (put < bytes) break;  // disk full / closed: never lie about length
    }
    frames_ += written;
}

void WavWriter::finish() {
    if (!f_) return;
    fflush(f_);
    const uint32_t riff_size = 36u + data_bytes_;
    if (fseek(f_, 4, SEEK_SET) == 0) wr_u32(f_, riff_size);
    if (fseek(f_, 40, SEEK_SET) == 0) wr_u32(f_, data_bytes_);
    fflush(f_);
    fseek(f_, 0, SEEK_END);
}

WavWriter::~WavWriter() {
    finish();
    if (f_) {
        fclose(f_);
        f_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// PacedSink
//
// Starvation policy: a frame is always emitted at its wall-clock deadline. If
// the ring holds fewer than frame_samples, the missing tail is emitted as
// silence, and on_starved(missing_ms) reports exactly how much was absent. Every
// emitted sample — real or silence — counts towards played_, so
// eos_to_first_played_ms and the gap timeline describe what the listener heard,
// not what the scheduler hoped for. real_played_ counts only the samples that
// actually came out of the ring, which is what distinguishes "the loop kept
// talking" from "the loop kept emitting silence".
// ---------------------------------------------------------------------------
PacedSink::PacedSink(PcmRing* ring, WavWriter* writer, int sample_rate, int frame_samples)
    : ring_(ring), writer_(writer), rate_(sample_rate > 0 ? sample_rate : 16000),
      frame_(frame_samples > 0 ? frame_samples : 1) {}

void PacedSink::run(const std::function<void(int64_t)>& on_starved) {
    const int64_t frame_ms = std::max<int64_t>(1, ms_of(frame_, rate_));
    std::vector<float> buf((size_t)frame_, 0.0f);
    const Clock::time_point start = Clock::now();
    int64_t index = 0;

    while (running_.load()) {
        const Clock::time_point deadline =
            start + std::chrono::microseconds((int64_t)(frame_ms * 1000) * index);
        sleep_until_cancellable(deadline, running_);
        if (!running_.load()) break;

        int got = ring_ ? ring_->pop(buf.data(), frame_) : 0;
        if (got < frame_) {
            const int64_t missing_ms = ms_of((int64_t)(frame_ - got), rate_);
            std::fill(buf.begin() + got, buf.end(), 0.0f);
            if (on_starved) on_starved(missing_ms);
        }
        if (writer_) writer_->write(buf.data(), frame_);
        // played_ counts everything emitted (silence included) so the timeline is
        // honest; real_played_ counts only samples that came out of the ring.
        played_.store(played_.load() + frame_);
        real_played_.store(real_played_.load() + got);
        ++index;
    }
}

int64_t PacedSink::played_ms() const { return ms_of(played_.load(), rate_); }

// ---------------------------------------------------------------------------
// PacedSource — one frame per frame_ms of wall clock, overruns reported.
// ---------------------------------------------------------------------------
PacedSource::PacedSource(WavReader* reader, int sample_rate, int frame_samples)
    : reader_(reader), rate_(sample_rate > 0 ? sample_rate : 16000),
      frame_(frame_samples > 0 ? frame_samples : 1) {}

int PacedSource::read(float* out, int n, int64_t* overrun_ms) {
    if (overrun_ms) *overrun_ms = 0;
    if (!reader_) return 0;
    if (n <= 0) n = frame_;
    if (started_ms_ == 0) started_ms_ = steady_ms();

    const int got = reader_->read(out, n);
    if (got <= 0) return 0;

    samples_read_ += got;
    // Deadline for the audio delivered so far: the reader is on schedule when
    // the clock has not passed it. Falling behind is reported, never hidden by
    // compressing the timeline.
    const int64_t deadline = started_ms_ + ms_of(samples_read_, rate_);
    const int64_t now = steady_ms();
    if (now > deadline && overrun_ms) *overrun_ms = now - deadline;
    sleep_until_ms(deadline);
    return got;
}

bool PacedSource::eof() const { return !reader_ || reader_->eof(); }

}  // namespace voice
}  // namespace crispasr
