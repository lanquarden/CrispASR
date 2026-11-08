// sequencer.h — the speech policy. Pure state machine: no threads, no GPU, no
// clocks beyond the timestamps carried by events. Feed it events, get actions.
//
// This is the part of the experiment that decides whether the pipeline ever
// goes silent: it owns the playback watermark policy, filler selection, clause
// chunking, and the filler->answer handoff.
#pragma once

#include <deque>
#include <string>
#include <vector>

#include "voice_types.h"

namespace crispasr {
namespace voice {

enum class ActionType {
    RequestText,   // ask the LLM for text of `kind`; `prompt` is context
    Speak,         // synthesise and enqueue `segment`
    FlushAudio,    // barge-in: drop queued PCM, abort generation
    CancelTask,    // cancel in-flight delegation
    FinishTurn,    // turn complete (metrics flush point)
};

struct Action {
    ActionType type = ActionType::RequestText;
    TextKind kind = TextKind::Filler;
    std::string text;      // RequestText: prompt/context; Speak: clause text
    TextSegment segment;   // Speak
    int64_t arg = 0;       // CancelTask: task id
};

struct SequencerStats {
    int fillers_queued = 0;
    int fillers_dropped_stale = 0;
    int fillers_repeat_rejected = 0;
    int fillers_llm_empty = 0;
    int stale_results_dropped = 0;
    int clauses_spoken = 0;
    int barge_ins = 0;
};

class SpeechSequencer {
   public:
    SpeechSequencer(const Config& cfg, int sample_rate);

    // --- inputs -----------------------------------------------------------
    void on_speech_start(int64_t t_ms);
    void on_partial(const std::string& text, int64_t t_ms);
    void on_final(const std::string& text, int64_t t_ms);
    void on_task_result(bool ok, const std::string& text, int64_t t_ms);
    void on_text_delta(TextKind kind, const std::string& piece, int64_t t_ms);
    void on_text_done(TextKind kind, int64_t t_ms);
    void on_playback(int64_t buffer_ms, int64_t played_ms, int64_t t_ms);
    void on_starved(int64_t at_ms, int64_t buffer_ms);
    void on_barge_in(int64_t t_ms);
    void on_pcm_enqueued(int n_samples, int64_t t_ms);
    // True while a synthesised piece is queued or in flight; the orchestrator
    // owns this, because only it can see the TTS queue.
    void set_audio_in_flight(bool v) { audio_in_flight_ = v; }
    TurnState state() const { return state_; }
    const SequencerStats& stats() const { return stats_; }
    TurnMetrics& metrics() { return metrics_; }
    const TurnMetrics& metrics() const { return metrics_; }
    const std::string& transcript() const { return transcript_; }
    const std::string& spoken_filler_context() const { return spoken_context_; }

    // Diagnostics for the filler policy itself (not part of the metrics
    // contract): the sweep reads them to explain a thin bridge.
    int filler_repeats_rejected() const { return stats_.fillers_repeat_rejected; }
    int filler_llm_empty() const { return stats_.fillers_llm_empty; }

    // --- outputs ----------------------------------------------------------
    // Drain actions produced since the last call. Order is significant.
    std::vector<Action> take_actions();

    // Turn bookkeeping for the metrics writer.
    bool turn_active() const { return state_ != TurnState::Idle; }
    int turn_id() const { return turn_; }

   private:
    void emit(Action a);
    void emit_request(TextKind kind, const std::string& prompt);
    void feed_text(TextKind kind, const std::string& piece, int64_t t_ms, bool done);
    void maybe_emit_clause(int64_t t_ms);
    void flush_text();
    void maybe_request_filler(const std::string& reason, int64_t t_ms);
    void request_next_clause(int64_t t_ms);
    std::string next_bank_filler(bool shortest_first);
    bool queue_bank_filler(int64_t t_ms);
    bool accept_filler_text(const std::string& text);
    std::string recent_fillers_for_prompt() const;
    int64_t remaining_speech_ms() const;
    void end_turn(int64_t t_ms);
    bool text_stale(const TextSegment& s, int64_t t_ms) const;

    Config cfg_;
    int rate_;
    TurnState state_ = TurnState::Idle;
    int turn_ = 0;

    // current turn context
    std::string transcript_;
    std::string stable_transcript_;
    std::string task_text_;
    bool task_pending_ = false;
    bool task_done_ = false;
    bool answer_requested_ = false;
    int task_id_ = 0;
    int64_t speech_start_ms_ = 0;
    int64_t speech_end_ms_ = 0;

    // text being streamed by the LLM
    std::string pending_[4];       // indexed by TextKind
    bool generation_active_[4] = {false, false, false, false};
    std::deque<TextSegment> clauses_;  // ready to speak, in order

    // filler state
    int bank_cursor_ = 0;
    std::deque<std::string> spoken_bank_;   // bank phrases used this session
    // Every filler (bank or LLM) spoken this session, in original wording. The
    // policy is "never repeat within a session", so this outlives the turn; it
    // is capped and only ever searched linearly (a voice session is short).
    std::deque<std::string> spoken_fillers_;
    int llm_fillers_outstanding_ = 0;  // filler generations queued/in flight
    int llm_filler_empty_streak_ = 0;  // this turn; bounds a hot retry loop
    int llm_filler_speakable_ = 0;   // clauses this generation actually queued
    int64_t task_deadline_ms_ = 0;   // dispatch + task_ms + jitter (upper bound)
    int64_t last_filler_ms_ = 0;
    // Speech this turn has committed (ack + filler), from a word estimate, and
    // how much of it has actually been played since the ack started.
    int64_t speech_committed_ms_ = 0;
    int64_t play_origin_ms_ = -1;
    int64_t played_since_origin_ms_ = 0;

    // playback view
    int64_t buffer_ms_ = 0;
    int64_t played_ms_ = 0;
    bool starved_ = false;
    int64_t answer_text_ms_ = 0;
    bool answer_buffered_ = false;
    bool audio_in_flight_ = false;
    int ack_words_ = 0;  // total ack words emitted this turn

    std::vector<Action> actions_;
    SequencerStats stats_;
    std::string spoken_context_;  // filler text actually committed this turn

    // metrics
    TurnMetrics metrics_;
    bool ack_played_ = false;
    bool ack_emitted_ = false;  // the ack clause has been handed to TTS
    bool answer_played_ = false;
};

}  // namespace voice
}  // namespace crispasr
