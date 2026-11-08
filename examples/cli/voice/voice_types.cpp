// voice_types.cpp — clocks, config normalisation, stringifiers, metrics JSON.
//
// Standard library only: this TU is the bottom of the voice module and is
// linked into the unit tests without any CrispASR object.

#include "voice_types.h"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <mutex>
#include <string>

namespace crispasr {
namespace voice {

namespace {

int64_t steady_clock_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

// Minimal JSON string escaping: the strings we serialise are stage labels and
// adapter text, so control characters are the only real hazard.
std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    static const char* kHex = "0123456789abcdef";
                    out += "\\u00";
                    out += kHex[(static_cast<unsigned char>(c) >> 4) & 0xF];
                    out += kHex[static_cast<unsigned char>(c) & 0xF];
                } else {
                    out += c;
                }
                break;
        }
    }
    return out;
}

void append_field(std::string& out, const char* name, int64_t value, bool& first) {
    if (!first) out += ',';
    first = false;
    out += '"';
    out += name;
    out += "\":";
    out += std::to_string(value);
}

void append_field(std::string& out, const char* name, const char* value, bool& first) {
    if (!first) out += ',';
    first = false;
    out += '"';
    out += name;
    out += "\":\"";
    out += json_escape(value ? value : "");
    out += '"';
}

}  // namespace

// Monotonic milliseconds. Wall-clock time is never used by the loop: every
// deadline in the plan is a real-time budget, not a calendar timestamp.
int64_t now_ms() { return steady_clock_ms(); }

// audio_io.h declares the same clock under a second name; both must observe the
// same monotonic source or the harness timeline and the metrics would drift.
int64_t steady_ms() { return steady_clock_ms(); }

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
void Config::normalise() {
    if (sample_rate <= 0) sample_rate = 16000;

    // frame_ms is the capture/playback quantum; 10 ms is the smallest useful
    // quantum for a 16 kHz mic path, 40 ms is the largest the VAD hysteresis and
    // the pacing loop stay honest about.
    if (frame_ms < 10) frame_ms = 10;
    if (frame_ms > 40) frame_ms = 40;

    frame_samples = (int)(((int64_t)sample_rate * (int64_t)frame_ms) / 1000);
    if (frame_samples < 1) frame_samples = 1;

    // Watermarks must be strictly ordered or the scheduler oscillates: low is
    // the "TTS has priority" line, target the fill goal, high the "stop queuing
    // text" line.
    if (low_water_ms < 1) low_water_ms = 1;
    if (target_water_ms <= low_water_ms) target_water_ms = low_water_ms + 1;
    if (high_water_ms <= target_water_ms) high_water_ms = target_water_ms + 1;

    if (vad_min_speech_ms < 0) vad_min_speech_ms = 0;
    if (vad_min_silence_ms < 0) vad_min_silence_ms = 0;
    if (stt_partial_ms < 1) stt_partial_ms = 1;
    if (stt_window_ms < 1) stt_window_ms = 1;
    if (clause_max_words < 1) clause_max_words = 1;
    if (ack_max_words < 1) ack_max_words = 1;
    if (filler_max_words < 1) filler_max_words = 1;
    if (answer_max_clause_tokens < 1) answer_max_clause_tokens = 1;
    if (task_ms < 0) task_ms = 0;
    if (task_jitter_ms < 0) task_jitter_ms = 0;
    if (task_fail_percent < 0) task_fail_percent = 0;
    if (task_fail_percent > 100) task_fail_percent = 100;
    if (max_turn_ms < frame_ms) max_turn_ms = frame_ms;
    if (barge_in_max_ms < 0) barge_in_max_ms = 0;
    if (runs < 1) runs = 1;
    if (warmup_runs < 0) warmup_runs = 0;
    if (filler_bank_size < 0) filler_bank_size = 0;
}

// ---------------------------------------------------------------------------
// Stringifiers (used by the timeline CSV and by the metrics JSON)
// ---------------------------------------------------------------------------
const char* to_string(TurnState s) {
    switch (s) {
        case TurnState::Idle: return "idle";
        case TurnState::Listening: return "listening";
        case TurnState::Committing: return "committing";
        case TurnState::Delegating: return "delegating";
        case TurnState::Answering: return "answering";
    }
    return "unknown";
}

const char* to_string(TextKind k) {
    switch (k) {
        case TextKind::Ack: return "ack";
        case TextKind::Filler: return "filler";
        case TextKind::Answer: return "answer";
        case TextKind::Clarify: return "clarify";
    }
    return "unknown";
}

const char* to_string(EventType e) {
    switch (e) {
        case EventType::SpeechStart: return "speech_start";
        case EventType::PartialTranscript: return "partial_transcript";
        case EventType::FinalTranscript: return "final_transcript";
        case EventType::BargeIn: return "barge_in";
        case EventType::PlaybackLow: return "playback_low";
        case EventType::PlaybackStarved: return "playback_starved";
        case EventType::TaskResult: return "task_result";
        case EventType::TaskFailed: return "task_failed";
        case EventType::Tick: return "tick";
        case EventType::Shutdown: return "shutdown";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------
int64_t TurnMetrics::max_gap_ms() const {
    int64_t m = 0;
    for (const GapEvent& g : gaps) {
        if (g.duration_ms > m) m = g.duration_ms;
    }
    return m;
}

int64_t TurnMetrics::total_gap_ms() const {
    int64_t total = 0;
    for (const GapEvent& g : gaps) total += g.duration_ms;
    return total;
}

// One compact JSON object, no spaces, no newline: metrics.jsonl is one of these
// per turn and check_metrics.py parses it line by line.
std::string metrics_json(const TurnMetrics& m) {
    std::string out;
    out.reserve(1024);
    out += '{';
    bool first = true;

    append_field(out, "turn", (int64_t)m.turn, first);
    append_field(out, "speech_start_ms", m.speech_start_ms, first);
    append_field(out, "speech_end_ms", m.speech_end_ms, first);
    append_field(out, "final_transcript_ms", m.final_transcript_ms, first);
    append_field(out, "ack_text_ms", m.ack_text_ms, first);
    append_field(out, "ack_first_pcm_ms", m.ack_first_pcm_ms, first);
    append_field(out, "ack_first_played_ms", m.ack_first_played_ms, first);
    append_field(out, "task_dispatch_ms", m.task_dispatch_ms, first);
    append_field(out, "task_result_ms", m.task_result_ms, first);
    append_field(out, "answer_text_ms", m.answer_text_ms, first);
    append_field(out, "answer_first_pcm_ms", m.answer_first_pcm_ms, first);
    append_field(out, "answer_first_played_ms", m.answer_first_played_ms, first);
    append_field(out, "answer_started_played_ms", m.answer_started_played_ms, first);
    append_field(out, "end_ms", m.end_ms, first);

    append_field(out, "stt_partials", (int64_t)m.stt_partials, first);
    append_field(out, "stt_partials_dropped", (int64_t)m.stt_partials_dropped, first);
    append_field(out, "stt_partial_cost_ms", m.stt_partial_cost_ms, first);
    append_field(out, "stt_final_cost_ms", m.stt_final_cost_ms, first);

    append_field(out, "llm_calls", (int64_t)m.llm_calls, first);
    append_field(out, "llm_ttft_ms", m.llm_ttft_ms, first);
    append_field(out, "llm_tokens", (int64_t)m.llm_tokens, first);
    append_field(out, "tool_calls", (int64_t)m.tool_calls, first);
    append_field(out, "tool_calls_valid", (int64_t)m.tool_calls_valid, first);

    append_field(out, "tts_pieces", (int64_t)m.tts_pieces, first);
    append_field(out, "tts_ttfa_ms", m.tts_ttfa_ms, first);
    append_field(out, "tts_block_max_ms", m.tts_block_max_ms, first);

    append_field(out, "fillers_played", (int64_t)m.fillers_played, first);
    append_field(out, "fillers_dropped_stale", (int64_t)m.fillers_dropped_stale, first);
    append_field(out, "stale_results_dropped", (int64_t)m.stale_results_dropped, first);
    append_field(out, "barge_ins", (int64_t)m.barge_ins, first);
    append_field(out, "barge_in_stop_ms", m.barge_in_stop_ms, first);

    append_field(out, "transcript", m.transcript.c_str(), first);
    append_field(out, "ack_text", m.ack_text.c_str(), first);
    append_field(out, "answer_text", m.answer_text.c_str(), first);

    if (!first) out += ',';
    first = false;
    out += "\"gaps\":[";
    for (size_t i = 0; i < m.gaps.size(); ++i) {
        const GapEvent& g = m.gaps[i];
        if (i) out += ',';
        out += '{';
        bool gfirst = true;
        append_field(out, "stage", g.stage.c_str(), gfirst);
        append_field(out, "at_ms", g.at_ms, gfirst);
        append_field(out, "buffer_ms", g.buffer_ms, gfirst);
        append_field(out, "duration_ms", g.duration_ms, gfirst);
        out += '}';
    }
    out += ']';

    // Derived stage-budget numbers. Names match the plan's metrics section so
    // check_metrics.py reads the same keys regardless of who computes them.
    append_field(out, "eos_to_final_ms", m.eos_to_final_ms(), first);
    append_field(out, "eos_to_ack_text_ms", m.eos_to_ack_text_ms(), first);
    append_field(out, "eos_to_first_played_ms", m.eos_to_first_played_ms(), first);
    append_field(out, "result_to_answer_played_ms", m.result_to_answer_played_ms(), first);
    append_field(out, "max_gap_ms", m.max_gap_ms(), first);
    append_field(out, "total_gap_ms", m.total_gap_ms(), first);

    out += '}';
    return out;
}

// ---------------------------------------------------------------------------
// Timeline CSV
// ---------------------------------------------------------------------------
const char* timeline_csv_header() {
    return "run,turn,stage,kind,text,t_request_ms,t_job_start_ms,"
           "t_first_pcm_ms,t_first_audible_ms,t_done_ms,samples,audible_ms\n";
}

namespace {

// CSV text field: always quoted, inner quotes doubled, one line, <= 60 chars.
// TTS text is user/model output, so both delimiters and control characters have
// to be neutralised or the file stops being machine-readable.
std::string csv_text(const std::string& in, size_t max_chars = 60) {
    std::string flat;
    flat.reserve(in.size());
    size_t chars = 0;
    bool pending_space = false;
    for (unsigned char c : in) {
        if (c == '\n' || c == '\r' || c == '\t' || c == ' ') {
            pending_space = !flat.empty();
            continue;
        }
        if (c < 0x20) continue;
        if (pending_space) {
            if (chars >= max_chars) break;
            flat.push_back(' ');
            ++chars;
            pending_space = false;
        }
        // Count a UTF-8 sequence once and never cut it in half.
        if ((c & 0xC0) != 0x80) {
            if (chars >= max_chars) break;
            ++chars;
        }
        flat.push_back((char)c);
    }
    std::string out;
    out.reserve(flat.size() + 2);
    out += '"';
    for (char c : flat) {
        if (c == '"') out += '"';
        out += c;
    }
    out += '"';
    return out;
}

}  // namespace

std::string timeline_csv_row(const TimelineRow& r) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), "%d,%d,%s,%s,", r.run, r.turn, r.stage.c_str(),
                  r.kind.c_str());
    std::string out(buf);
    out += csv_text(r.text);
    std::snprintf(buf, sizeof(buf), ",%lld,%lld,%lld,%lld,%lld,%lld,%lld\n",
                  (long long)r.t_request_ms, (long long)r.t_job_start_ms,
                  (long long)r.t_first_pcm_ms, (long long)r.t_first_audible_ms,
                  (long long)r.t_done_ms, (long long)r.samples,
                  (long long)r.audible_ms);
    out += buf;
    return out;
}

TimelineTrace& timeline_trace() {
    static TimelineTrace trace;
    return trace;
}

// ---------------------------------------------------------------------------
// Audibility scan
// ---------------------------------------------------------------------------
AudibilityMeter::AudibilityMeter(int rate, double threshold) {
    rate_ = rate > 0 ? rate : 16000;
    threshold_ = threshold;
    window_samples_ = rate_ / 100;  // 10 ms
    if (window_samples_ < 1) window_samples_ = 1;
    window_ms_ = (int64_t)window_samples_ * 1000 / rate_;
    if (window_ms_ < 1) window_ms_ = 1;
    pending_.reserve((size_t)window_samples_);
}

void AudibilityMeter::push(const float* pcm, int n) {
    if (pcm == nullptr || n <= 0) return;
    total_samples_ += n;
    for (int i = 0; i < n; ++i) {
        pending_.push_back(pcm[i]);
        if ((int)pending_.size() == window_samples_) score_window();
    }
}

void AudibilityMeter::finish() {
    if (!pending_.empty()) score_window();
}

void AudibilityMeter::score_window() {
    double sum = 0.0;
    for (float s : pending_) sum += (double)s * (double)s;
    const double rms = pending_.empty()
                           ? 0.0
                           : std::sqrt(sum / (double)pending_.size());
    if (rms >= threshold_) {
        if (first_window_ < 0) first_window_ = window_index_;
        audible_samples_ += (int64_t)pending_.size();
    }
    ++window_index_;
    pending_.clear();
}

int64_t AudibilityMeter::audible_ms() const {
    if (rate_ <= 0) return 0;
    return audible_samples_ * 1000 / rate_;
}

}  // namespace voice
}  // namespace crispasr
