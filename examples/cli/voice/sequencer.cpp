// sequencer.cpp — speech policy for the realtime voice loop.
//
// Pure: every decision is a function of the events fed in. No threads, no
// engines, no GPU. The orchestrator executes the emitted actions.
#include "sequencer.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace crispasr {
namespace voice {
namespace {

const char* const kFillerBank[] = {
    "One moment.",
    "Let me check that.",
    "Checking now.",
    "Bear with me.",
    "One second.",
    "Looking into that.",
    "I'm on it.",
    "Give me a moment.",
    "Checking the details.",
    "Just a moment.",
    "Hold on a second.",
    "Let me look that up.",
};
constexpr int kFillerCount = (int)(sizeof(kFillerBank) / sizeof(kFillerBank[0]));

bool is_sentence_end(char c) {
    return c == '.' || c == '!' || c == '?' || c == ';' || c == ':';
}

int word_count(const std::string& s) {
    int n = 0;
    bool in = false;
    for (char c : s) {
        if (std::isspace((unsigned char)c)) {
            in = false;
        } else if (!in) {
            in = true;
            ++n;
        }
    }
    return n;
}

// Splits off the first speakable clause: a sentence end, or clause_max words.
// Returns true and fills `out`/`rest` when a clause is available.
bool take_clause(std::string& pending, int max_words, std::string* out) {
    if (pending.empty()) return false;
    int words = 0;
    bool in_word = false;
    for (size_t i = 0; i < pending.size(); ++i) {
        const char c = pending[i];
        if (std::isspace((unsigned char)c)) {
            in_word = false;
        } else if (!in_word) {
            in_word = true;
            ++words;
        }
        const bool sentence = is_sentence_end(c) &&
                              (i + 1 == pending.size() ||
                               std::isspace((unsigned char)pending[i + 1]));
        if ((sentence && words >= 2) || words >= max_words) {
            // Never cut a word in half: back off to the last space if needed.
            size_t end = i + 1;
            while (end < pending.size() && !std::isspace((unsigned char)pending[end])) {
                if (is_sentence_end(pending[end])) { ++end; break; }
                ++end;
            }
            *out = pending.substr(0, end);
            pending.erase(0, end);
            // trim leading separators/space
            while (!pending.empty() && std::isspace((unsigned char)pending.front()))
                pending.erase(pending.begin());
            return true;
        }
    }
    return false;
}

// Quotes and markdown are not speech: qwen3-tts renders them as long pauses.
std::string sanitize_for_speech(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    bool space = false;
    for (char c : in) {
        if (c == '"' || c == '\'' || c == '`' || c == '*' || c == '#' ||
            c == '_' || c == '\n' || c == '\r' || c == '\t') {
            space = true;
            continue;
        }
        if (std::isspace((unsigned char)c)) {
            space = true;
            continue;
        }
        if (space && !out.empty()) out += ' ';
        space = false;
        out += c;
    }
    return out;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

}  // namespace

SpeechSequencer::SpeechSequencer(const Config& cfg, int sample_rate)
    : cfg_(cfg), rate_(sample_rate) {}

void SpeechSequencer::emit(Action a) { actions_.push_back(std::move(a)); }

std::vector<Action> SpeechSequencer::take_actions() {
    auto out = std::move(actions_);
    actions_.clear();
    return out;
}

void SpeechSequencer::emit_request(TextKind kind, const std::string& prompt) {
    Action a;
    a.type = ActionType::RequestText;
    a.kind = kind;
    a.text = prompt;
    emit(std::move(a));
}

bool SpeechSequencer::text_stale(const TextSegment& s, int64_t t_ms) const {
    return s.expire_ms > 0 && t_ms > s.expire_ms;
}

void SpeechSequencer::on_speech_start(int64_t t_ms) {
    if (state_ == TurnState::Delegating || state_ == TurnState::Answering) {
        // The orchestrator normally translates this into BARGE_IN first.
        on_barge_in(t_ms);
    }
    if (state_ != TurnState::Idle && state_ != TurnState::Listening) return;

    if (state_ == TurnState::Idle) {
        ++turn_;
        metrics_ = TurnMetrics{};
        metrics_.turn = turn_;
        metrics_.speech_start_ms = t_ms;
    }
    state_ = TurnState::Listening;
    speech_start_ms_ = t_ms;
    speech_end_ms_ = 0;
    transcript_.clear();
    stable_transcript_.clear();
    task_text_.clear();
    task_pending_ = false;
    task_done_ = false;
    task_id_ = 0;
    answer_requested_ = false;
    answer_text_ms_ = 0;
    answer_buffered_ = false;
    ack_played_ = false;
    ack_emitted_ = false;
    answer_played_ = false;
    spoken_context_.clear();
    for (int i = 0; i < 4; ++i) {
        pending_[i].clear();
        generation_active_[i] = false;
    }
    clauses_.clear();
    ack_words_ = 0;
}

void SpeechSequencer::on_partial(const std::string& text, int64_t t_ms) {
    (void)t_ms;
    stable_transcript_ = text;
}

void SpeechSequencer::on_final(const std::string& text, int64_t t_ms) {
    if (state_ == TurnState::Idle) return;
    transcript_ = trim(text);
    speech_end_ms_ = t_ms;
    metrics_.speech_end_ms = t_ms;
    metrics_.final_transcript_ms = t_ms;
    state_ = TurnState::Committing;

    if (transcript_.empty()) {
        end_turn(t_ms);
        return;
    }

    state_ = TurnState::Delegating;
    task_pending_ = true;
    metrics_.task_dispatch_ms = t_ms;
    // The orchestrator dispatches the task when it sees this request; the ack
    // is generated immediately and in parallel.
    emit_request(TextKind::Ack,
                 "The user said: \"" + transcript_ + "\". Reply with one short "
                 "acknowledgement that restates what you understood and says you "
                 "are checking. No result, no speculation.");
}

void SpeechSequencer::on_task_result(bool ok, const std::string& text,
                                     int64_t t_ms) {
    if (!task_pending_) {
        ++stats_.stale_results_dropped;
        metrics_.stale_results_dropped = stats_.stale_results_dropped;
        return;
    }
    task_pending_ = false;
    task_done_ = true;
    // Drop ack clauses that have not reached TTS yet: the answer preempts them.
    {
        std::deque<TextSegment> kept;
        for (auto& s : clauses_) {
            if (s.kind == TextKind::Ack) {
                ++stats_.fillers_dropped_stale;  // counted as stale speech
            } else {
                kept.push_back(std::move(s));
            }
        }
        clauses_.swap(kept);
    }
    task_text_ = ok ? text : ("The check failed: " + text);
    metrics_.task_result_ms = t_ms;
    state_ = TurnState::Answering;
    answer_requested_ = true;
    emit_request(TextKind::Answer,
                 "Conversation so far: user said \"" + transcript_ + "\". " +
                     (spoken_context_.empty()
                          ? std::string()
                          : "You already told them: \"" + spoken_context_ + "\". ") +
                     "Background result: \"" + task_text_ +
                     "\". Answer in one or two short spoken sentences, continuing "
                     "naturally from what you already said.");
}

void SpeechSequencer::on_text_delta(TextKind kind, const std::string& piece,
                                    int64_t t_ms) {
    const int idx = (int)kind;
    if (!generation_active_[idx]) {
        generation_active_[idx] = true;
        ++metrics_.llm_calls;
        if (kind == TextKind::Ack) metrics_.ack_text_ms = t_ms;
        if (kind == TextKind::Answer) {
            metrics_.answer_text_ms = t_ms;
            answer_text_ms_ = t_ms;
        }
    }
    pending_[idx] += piece;
    feed_text(kind, std::string(), t_ms, false);
}

void SpeechSequencer::on_text_done(TextKind kind, int64_t t_ms) {
    generation_active_[(int)kind] = false;
    if (kind == TextKind::Answer) answer_requested_ = false;
    feed_text(kind, std::string(), t_ms, true);
}

void SpeechSequencer::feed_text(TextKind kind, const std::string& piece,
                                int64_t t_ms, bool done) {
    if (!piece.empty()) pending_[(int)kind] += piece;

    const int max_words =
        kind == TextKind::Ack ? cfg_.ack_max_words
        : kind == TextKind::Filler ? cfg_.filler_max_words
                                   : cfg_.clause_max_words;

    std::string clause;
    while (take_clause(pending_[(int)kind], max_words, &clause)) {
        if (kind == TextKind::Ack) {
            // An ack is one short breath of speech. Two long clauses would push
            // the real answer behind ten seconds of acknowledgement.
            if (ack_words_ >= cfg_.ack_max_words) {
                pending_[(int)kind].clear();
                break;
            }
            ack_words_ += word_count(clause);
        }
        TextSegment s;
        s.kind = kind;
        s.text = sanitize_for_speech(trim(clause));
        s.seq = (uint64_t)clauses_.size() + 1;
        s.created_ms = t_ms;
        if (kind == TextKind::Filler) s.expire_ms = t_ms + 4000;
        clauses_.push_back(std::move(s));
    }
    if (done) {
        std::string rest = sanitize_for_speech(trim(pending_[(int)kind]));
        pending_[(int)kind].clear();
        if (!rest.empty() && word_count(rest) > 0) {
            TextSegment s;
            s.kind = kind;
            s.text = rest;
            s.seq = (uint64_t)clauses_.size() + 1;
            s.created_ms = t_ms;
            if (kind == TextKind::Filler) s.expire_ms = t_ms + 4000;
            clauses_.push_back(std::move(s));
        }
    }
    request_next_clause(t_ms);
}

void SpeechSequencer::request_next_clause(int64_t t_ms) {
    // Emit clauses while the playback buffer has room. Filler is dropped when
    // the answer is buffered; the answer always wins.
    while (!clauses_.empty()) {
        TextSegment& s = clauses_.front();
        if (text_stale(s, t_ms)) {
            if (s.kind == TextKind::Filler) ++stats_.fillers_dropped_stale;
            clauses_.pop_front();
            continue;
        }
        if (answer_buffered_ && s.kind == TextKind::Filler) {
            clauses_.pop_front();
            ++stats_.fillers_dropped_stale;
            continue;
        }
        if (buffer_ms_ >= cfg_.high_water_ms &&
            s.kind != TextKind::Ack) {  // ack always goes first
            return;
        }
        Action a;
        a.type = ActionType::Speak;
        a.segment = s;
        a.kind = s.kind;
        a.text = s.text;
        if (s.kind == TextKind::Ack) ack_emitted_ = true;
        if (s.kind == TextKind::Filler) {
            spoken_context_ += (spoken_context_.empty() ? "" : " ") + s.text;
            ++metrics_.fillers_played;
        }
        clauses_.pop_front();
        ++stats_.clauses_spoken;
        emit(std::move(a));
    }
}

void SpeechSequencer::maybe_request_filler(const std::string& reason,
                                           int64_t t_ms) {
    (void)reason;
    if (cfg_.filler_mode == Config::FillerMode::Bank ||
        cfg_.filler_mode == Config::FillerMode::BankThenLlm) {
        if (spoken_bank_.size() >= (size_t)kFillerCount) {
            // Bank exhausted in this session: fall through to the LLM only if
            // the mode allows it, otherwise stay silent.
            if (cfg_.filler_mode == Config::FillerMode::Bank) return;
        } else {
            TextSegment s;
            s.kind = TextKind::Filler;
            s.text = next_bank_filler();
            s.seq = 0;
            s.created_ms = t_ms;
            s.expire_ms = t_ms + 4000;
            clauses_.push_back(std::move(s));
            request_next_clause(t_ms);
            return;
        }
    }
    emit_request(TextKind::Filler,
                 "The user is waiting. You are still checking \"" + transcript_ +
                     "\". Say one short, natural holding phrase that does not "
                     "claim progress or a result, and does not repeat what you "
                     "already said" +
                     (spoken_context_.empty()
                          ? std::string(".")
                          : ": \"" + spoken_context_ + "\"."));
}

std::string SpeechSequencer::next_bank_filler() {
    if (bank_cursor_ >= kFillerCount) bank_cursor_ = 0;
    const std::string s = kFillerBank[bank_cursor_++ % kFillerCount];
    spoken_bank_.push_back(s);
    return s;
}

void SpeechSequencer::on_playback(int64_t buffer_ms, int64_t played_ms,
                                  int64_t t_ms) {
    buffer_ms_ = buffer_ms;
    played_ms_ = played_ms;

    // The answer's PCM existing is the signal, not a full target buffer: a
    // short answer may never fill target_water_ms, and waiting for it kept
    // requesting fillers that then played after the answer.
    if (metrics_.answer_first_pcm_ms > 0 && !answer_buffered_) {
        answer_buffered_ = true;
        // Drop queued fillers only now: the answer has enough audio buffered
        // that stopping the bridge will not create a gap.
        std::deque<TextSegment> kept;
        for (auto& s : clauses_) {
            if (s.kind == TextKind::Filler) {
                ++stats_.fillers_dropped_stale;
            } else {
                kept.push_back(std::move(s));
            }
        }
        clauses_.swap(kept);
    }

    if (buffer_ms < cfg_.target_water_ms) {
        // Never bridge before the ack itself has been heard: the ack is the
        // turn's first audio and must not queue behind instant bank fillers.
        // Keep bridging until the answer's own audio is buffered. Stopping the
        // moment the task returns leaves a hole exactly where TTS needs to
        // synthesise the answer.
        const bool awaiting_answer_audio =
            answer_text_ms_ > 0 && !answer_buffered_;
        if (clauses_.empty() && (task_pending_ || awaiting_answer_audio) &&
            metrics_.ack_first_played_ms > 0 && t_ms - last_filler_ms_ > 300) {
            last_filler_ms_ = t_ms;
            maybe_request_filler("low-water", t_ms);
        } else {
            request_next_clause(t_ms);
        }
    }

    if (state_ == TurnState::Answering && buffer_ms == 0 && clauses_.empty() &&
        !generation_active_[(int)TextKind::Answer] && !answer_requested_ &&
        !audio_in_flight_ && metrics_.answer_first_pcm_ms > 0) {
        end_turn(t_ms);
    }
}

void SpeechSequencer::on_starved(int64_t at_ms, int64_t buffer_ms) {
    if (!turn_active()) return;
    // Silence before the first audio is first-audio latency, not an underrun.
    if (metrics_.ack_first_played_ms == 0 && metrics_.answer_first_played_ms == 0)
        return;
    const int64_t missing = buffer_ms > 0 ? buffer_ms : 0;
    if (!metrics_.gaps.empty()) {
        GapEvent& last = metrics_.gaps.back();
        if (at_ms - (last.at_ms + last.duration_ms) <= 100) {
            last.duration_ms += missing;
            // keep the first depth
            return;
        }
    }
    GapEvent g;
    g.stage = "playback";
    g.at_ms = at_ms;
    g.buffer_ms = buffer_ms;
    g.duration_ms = missing;
    metrics_.gaps.push_back(g);
}

void SpeechSequencer::on_barge_in(int64_t t_ms) {
    if (!cfg_.barge_in_enabled) return;
    if (state_ != TurnState::Delegating && state_ != TurnState::Answering)
        return;
    ++stats_.barge_ins;
    metrics_.barge_ins = stats_.barge_ins;
    metrics_.barge_in_stop_ms = 0;  // measured by the orchestrator
    Action flush;
    flush.type = ActionType::FlushAudio;
    flush.arg = t_ms;
    emit(std::move(flush));
    if (task_pending_) {
        Action cancel;
        cancel.type = ActionType::CancelTask;
        cancel.arg = task_id_;
        emit(std::move(cancel));
        task_pending_ = false;
    }
    Action finish;
    finish.type = ActionType::FinishTurn;
    finish.arg = 1;  // interrupted
    emit(std::move(finish));
    flush_text();
    state_ = TurnState::Listening;
    speech_start_ms_ = t_ms;
    transcript_.clear();
}

void SpeechSequencer::flush_text() {
    for (int i = 0; i < 4; ++i) {
        pending_[i].clear();
        generation_active_[i] = false;
    }
    clauses_.clear();
    answer_text_ms_ = 0;
    answer_buffered_ = false;
}

void SpeechSequencer::on_pcm_enqueued(int n_samples, int64_t t_ms) {
    (void)n_samples;
    (void)t_ms;
    // Filler throttling is time-based; this hook exists for future watermarks.
}

void SpeechSequencer::end_turn(int64_t t_ms) {
    if (state_ == TurnState::Idle) return;
    metrics_.end_ms = t_ms;
    state_ = TurnState::Idle;
    Action a;
    a.type = ActionType::FinishTurn;
    a.arg = 0;
    emit(std::move(a));
}

}  // namespace voice
}  // namespace crispasr
