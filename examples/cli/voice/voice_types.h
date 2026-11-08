// voice_types.h — shared data types for the realtime voice loop (home-ops#584).
//
// Pure data: no inference, no audio device, no CrispASR includes. The loop core
// can therefore be tested with fake engines and a fake clock.
#pragma once

#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

namespace crispasr {
namespace voice {

int64_t now_ms();

// ---------------------------------------------------------------------------
// Configuration. Every number is a knob because the experiment is a sweep.
// ---------------------------------------------------------------------------
struct Config {
    // audio
    int sample_rate = 16000;
    int frame_ms = 20;
    int frame_samples = 320;  // derived in normalise()

    // turn detection
    int vad_min_speech_ms = 250;
    int vad_min_silence_ms = 150;   // end-of-turn silence (budget-dominant; see RESULTS)
    float vad_energy_threshold = 0.006f;  // RMS on 0..1 float PCM
    bool vad_gate_during_playback = true;

    // partial transcripts
    int stt_partial_ms = 250;
    int stt_window_ms = 6000;       // sliding window re-decoded
    bool stt_partials_enabled = true;

    // playback / scheduling
    int low_water_ms = 1200;
    int target_water_ms = 2000;
    int high_water_ms = 3000;
    int clause_max_words = 6;
    int ack_max_words = 14;
    int filler_max_words = 18;
    int answer_max_clause_tokens = 48;

    // delegation (simulated A2A-shaped task)
    int task_ms = 6000;
    int task_jitter_ms = 800;
    int task_fail_percent = 0;

    // filler policy
    enum class FillerMode { Bank, Llm, BankThenLlm };
    FillerMode filler_mode = FillerMode::Bank;
    int filler_bank_size = 12;      // clips pre-rendered at startup
    bool prerender_filler = true;

    // behaviour
    bool barge_in_enabled = true;
    bool speculate = false;         // M3: start ack on stable partials
    int barge_in_max_ms = 200;
    int max_turn_ms = 30000;

    // engines
    std::string asr_backend = "qwen3-1.7b";
    std::string asr_model;
    std::string tts_backend = "qwen3-tts-1.7b-base";
    std::string tts_model;
    std::string tts_tokenizer;
    std::string tts_voice_dir;
    std::string tts_voice = "jarvis";
    std::string chat_model;
    int chat_gpu_layers = 99;
    int chat_ctx = 2048;

    // experiment
    int runs = 5;                   // warm-up runs excluded
    int warmup_runs = 1;
    std::string out_dir = ".";

    // CLI-only fields (parsed by config.cpp)
    std::string mode;          // "selftest" | "harness" | "run" | ""
    std::string harness_path;
    std::string input_wav;
    std::string out_wav;
    std::string metrics_path;

    void normalise();
};

// ---------------------------------------------------------------------------
// Events flowing into the sequencer (from capture, arbiter callbacks, playback,
// and the task adapter). All payloads are plain values.
// ---------------------------------------------------------------------------
enum class EventType {
    SpeechStart,
    PartialTranscript,
    FinalTranscript,
    BargeIn,
    PlaybackLow,
    PlaybackStarved,
    TaskResult,
    TaskFailed,
    Tick,
    Shutdown,
};

struct Event {
    EventType type = EventType::Tick;
    std::string text;       // transcripts, task results
    int64_t t_ms = 0;       // when the event happened
    int64_t aux = 0;        // buffer_ms for PlaybackLow/Starved, kind, marker
    int64_t aux2 = 0;       // ttfa / ttft carried by completion markers
    int64_t aux3 = 0;       // kind for completion markers
};

enum class TextKind { Ack, Filler, Answer, Clarify };

struct TextSegment {
    TextKind kind = TextKind::Filler;
    std::string text;
    uint64_t seq = 0;
    int64_t created_ms = 0;
    int64_t expire_ms = 0;   // 0 = never
    bool spoken = false;
};

enum class TurnState { Idle, Listening, Committing, Delegating, Answering };

const char* to_string(TurnState s);
const char* to_string(TextKind k);
const char* to_string(EventType e);

// ---------------------------------------------------------------------------
// Metrics. One TurnMetrics per turn, serialised as one JSON line.
// ---------------------------------------------------------------------------
struct GapEvent {
    std::string stage;     // "playback", "tts", "llm"
    int64_t at_ms = 0;
    int64_t buffer_ms = 0;
    int64_t duration_ms = 0;
};

struct TurnMetrics {
    int turn = 0;
    int64_t speech_start_ms = 0;
    int64_t speech_end_ms = 0;
    int64_t final_transcript_ms = 0;
    int64_t ack_text_ms = 0;
    int64_t ack_first_pcm_ms = 0;
    int64_t ack_first_played_ms = 0;
    int64_t task_dispatch_ms = 0;
    int64_t task_result_ms = 0;
    int64_t answer_text_ms = 0;
    int64_t answer_first_pcm_ms = 0;
    int64_t answer_first_played_ms = 0;
    int64_t answer_started_played_ms = 0;  // first played ms of the answer turn
    int64_t end_ms = 0;

    int stt_partials = 0;
    int stt_partials_dropped = 0;
    int64_t stt_partial_cost_ms = 0;
    int64_t stt_final_cost_ms = 0;

    int llm_calls = 0;
    int64_t llm_ttft_ms = 0;
    int llm_tokens = 0;
    int tool_calls = 0;
    int tool_calls_valid = 0;

    int tts_pieces = 0;
    int64_t tts_ttfa_ms = 0;
    int64_t tts_block_max_ms = 0;

    int fillers_played = 0;
    int fillers_dropped_stale = 0;
    int stale_results_dropped = 0;
    int barge_ins = 0;
    int64_t barge_in_stop_ms = 0;

    std::vector<GapEvent> gaps;
    // Text actually spoken, for verification (not aggregated by the harness).
    std::string transcript;
    std::string ack_text;
    std::string answer_text;

    // derived
    int64_t eos_to_final_ms() const { return max0(final_transcript_ms - speech_end_ms); }
    int64_t eos_to_ack_text_ms() const { return max0(ack_text_ms - speech_end_ms); }
    int64_t eos_to_first_played_ms() const { return max0(ack_first_played_ms - speech_end_ms); }
    int64_t result_to_answer_played_ms() const { return max0(answer_first_played_ms - task_result_ms); }
    int64_t max_gap_ms() const;
    int64_t total_gap_ms() const;

    static int64_t max0(int64_t v) { return v < 0 ? 0 : v; }
};

std::string metrics_json(const TurnMetrics& m);

// ---------------------------------------------------------------------------
// Per-piece timeline trace (timeline.csv).
//
// Diagnostics only: no policy reads this, and no existing metric changes
// meaning because of it. The point is attribution — when the rendered WAV holds
// a silent run, these rows say whether it was (a) a piece that starts with
// silence, (b) the gap between two pieces, or (c) the playback ring starving.
//
// One row per TTS piece, one per LLM generation, and one per 100 ms of an
// active turn for the playback ring. Times are milliseconds relative to the
// turn's speech_start (the same base as metrics.jsonl) or -1 when not
// applicable.
//
// Per-stage column meaning (the CSV keeps one fixed column set):
//   stage=tts       samples = emitted PCM samples, audible_ms = samples above
//                   threshold / rate, t_first_pcm = first emitted chunk,
//                   t_first_audible = first chunk + offset of the first loud
//                   10 ms window inside the piece (so model-padded leading
//                   silence shows up as t_first_audible > t_first_pcm).
//   stage=llm       t_first_pcm = first token (there is no PCM), samples =
//                   token count, the *audible fields are -1/0.
//   stage=playback  t_request_ms = sample time, samples = ring depth in ms,
//                   audible_ms = real_played_ms, t_* others = -1.
struct TimelineRow {
    int run = 1;
    int turn = 0;
    std::string stage;  // "tts" | "llm" | "playback"
    std::string kind;   // "ack" | "filler" | "answer" ("" for playback)
    std::string text;   // one line, quotes escaped by the writer, <= 60 chars
    int64_t t_request_ms = -1;
    int64_t t_job_start_ms = -1;
    int64_t t_first_pcm_ms = -1;
    int64_t t_first_audible_ms = -1;
    int64_t t_done_ms = -1;
    int64_t samples = 0;
    int64_t audible_ms = 0;
};

// Header line, including the trailing newline. Written once per file.
const char* timeline_csv_header();
// One CSV record, including the trailing newline.
std::string timeline_csv_row(const TimelineRow& r);

// Process-wide collector. Rows are produced from the arbiter thread and from
// the control thread, so every access is mutex-guarded. One run at a time: the
// orchestrator clears the rows when it starts, the harness stamps the run
// number and copies them out when it ends.
class TimelineTrace {
   public:
    void set_run(int run) {
        std::lock_guard<std::mutex> lk(mu_);
        run_ = run;
        rows_.clear();
    }
    void clear_rows() {
        std::lock_guard<std::mutex> lk(mu_);
        rows_.clear();
    }
    int run() const {
        std::lock_guard<std::mutex> lk(mu_);
        return run_;
    }
    void add(TimelineRow row) {
        std::lock_guard<std::mutex> lk(mu_);
        row.run = run_;
        rows_.push_back(std::move(row));
    }
    std::vector<TimelineRow> rows() const {
        std::lock_guard<std::mutex> lk(mu_);
        return rows_;
    }

   private:
    mutable std::mutex mu_;
    int run_ = 1;
    std::vector<TimelineRow> rows_;
};

TimelineTrace& timeline_trace();

// ---------------------------------------------------------------------------
// Audibility scan for one emitted TTS piece.
//
// Scores the PCM the engine actually emitted in 10 ms windows and counts the
// windows whose RMS reaches `threshold`. The first such window is the piece's
// audible onset, which is what separates "the model emitted silence first"
// from "the ring was empty".
// ---------------------------------------------------------------------------
class AudibilityMeter {
   public:
    explicit AudibilityMeter(int rate, double threshold = 0.01);

    void push(const float* pcm, int n);   // one emitted chunk
    void finish();                        // score a trailing partial window

    bool found() const { return first_window_ >= 0; }
    // Index of the first loud 10 ms window counted from the piece start.
    int64_t first_window() const { return first_window_; }
    int64_t window_ms() const { return window_ms_; }
    int64_t total_samples() const { return total_samples_; }
    int64_t audible_samples() const { return audible_samples_; }
    int64_t audible_ms() const;

   private:
    void score_window();

    double threshold_;
    int window_samples_ = 160;
    int64_t window_ms_ = 10;
    int rate_ = 16000;
    std::vector<float> pending_;
    int64_t window_index_ = 0;
    int64_t first_window_ = -1;
    int64_t total_samples_ = 0;
    int64_t audible_samples_ = 0;
};

}  // namespace voice
}  // namespace crispasr
