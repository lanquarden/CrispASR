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

// The bank is lip-synced to the TTS voice, so it is not translated at runtime:
// an English phrase read by the Spanish voice is exactly the accent failure the
// language flag exists to avoid. Same shape as the English bank (short, no
// digits, no result claims, one clause each).
const char* const kFillerBankEs[] = {
    "Un momento.",
    "Déjame comprobar.",
    "Estoy mirando.",
    "Un segundo.",
    "Dame un momento.",
    "Lo estoy revisando.",
    "Espera un segundo.",
    "Ahora mismo lo miro.",
};
constexpr int kFillerCountEs =
    (int)(sizeof(kFillerBankEs) / sizeof(kFillerBankEs[0]));

const char* bank_phrase(const std::string& language, int idx) {
    if (language == "es") return kFillerBankEs[idx % kFillerCountEs];
    return kFillerBank[idx % kFillerCount];
}

int bank_count(const std::string& language) {
    return language == "es" ? kFillerCountEs : kFillerCount;
}

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
// Emoji and symbol codepoints are the same class of noise (a smiley becomes a
// pause or a vocalisation), and the model emits them when the LLM answers with
// an emoji. Latin accents (es/fr/de) must survive, so only known symbol blocks
// are dropped.
bool is_symbol_codepoint(uint32_t cp) {
    return (cp >= 0x2190 && cp <= 0x21FF) ||  // arrows
           (cp >= 0x2300 && cp <= 0x23FF) ||  // technical (hourglass, etc.)
           (cp >= 0x25A0 && cp <= 0x25FF) ||  // geometric shapes
           (cp >= 0x2600 && cp <= 0x27BF) ||  // misc symbols / dingbats
           (cp >= 0x2B00 && cp <= 0x2BFF) ||  // misc symbols and arrows
           (cp >= 0x1F000 && cp <= 0x1FAFF) ||  // emoji, pictographs
           (cp == 0xFE0F) || (cp == 0xFE0E) ||  // variation selectors
           (cp == 0x200D) || (cp == 0x20E3);    // ZWJ, combining keycap
}

std::string sanitize_for_speech(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    bool space = false;
    for (size_t i = 0; i < in.size();) {
        const unsigned char c = (unsigned char)in[i];
        // Decode one UTF-8 sequence; malformed bytes are dropped as noise.
        uint32_t cp = c;
        size_t len = 1;
        if (c >= 0x80) {
            len = (c >= 0xF0) ? 4 : (c >= 0xE0) ? 3 : (c >= 0xC0) ? 2 : 0;
            if (len == 0 || i + len > in.size()) {
                ++i;
                space = true;
                continue;
            }
            cp = (uint32_t)(c & (0xFF >> (len + 1)));
            for (size_t k = 1; k < len; ++k)
                cp = (cp << 6) | (uint32_t)((unsigned char)in[i + k] & 0x3F);
        }
        i += len;

        if (c < 0x80) {
            if (c == '"' || c == '\'' || c == '`' || c == '*' || c == '#' ||
                c == '_' || c == '\n' || c == '\r' || c == '\t') {
                space = true;
                continue;
            }
            if (std::isspace(c)) {
                space = true;
                continue;
            }
        } else if (is_symbol_codepoint(cp)) {
            space = true;
            continue;
        }
        if (space && !out.empty()) out += ' ';
        space = false;
        out.append(in, i - len, len);
    }
    return out;
}

std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace((unsigned char)s[a])) ++a;
    while (b > a && std::isspace((unsigned char)s[b - 1])) --b;
    return s.substr(a, b - a);
}

// "No repeats within a session" is compared on lowercase words only, so
// "One moment." and "one moment" are the same phrase.
std::string normalize_phrase(const std::string& s) {
    std::string out;
    bool space = false;
    for (char c : s) {
        const unsigned char u = (unsigned char)c;
        if (std::isalnum(u)) {
            if (space && !out.empty()) out += ' ';
            space = false;
            out += (char)std::tolower(u);
        } else {
            space = true;
        }
    }
    return out;
}

// A filler that cannot be spoken is rejected and replaced by the bank. Digits
// are the mechanical proxy for "claims a result": "the door is 1" is never a
// holding phrase.
bool plausible_filler(const std::string& text) {
    const std::string t = trim(text);
    if (t.empty()) return false;
    if (word_count(t) > 40) return false;
    for (char c : t) {
        if (std::isdigit((unsigned char)c)) return false;
    }
    return !normalize_phrase(t).empty();
}

// Filler policy constants. They are deliberately conservative: the bridge only
// has to cover a gap, not fill the whole wait. A bank phrase lasts 0.7-1.3 s at
// kMsPerWord, so one estimate plus a short phrase accumulates ~1.5 s of speech.
constexpr int64_t kFillerEstimateMs = 1500;
// Head start the answer needs after the task result (LLM TTFT + TTS TTFA).
constexpr int64_t kResultReserveMs = 500;
// Bound on retries when the LLM produces an unusable filler.
constexpr int kMaxLlmFillerAttempts = 3;
constexpr size_t kMaxSpokenFillersRemembered = 64;
// Spoken-duration estimate, measured on the real engines (~325 ms/word for the
// English qwen3-tts clone voice, ~360 ms/word for Spanish; the longer figure is
// the conservative one, because over-estimating only shortens the bridge while
// under-estimating leaves dead air). Used as the filler budget: committed
// speech minus playback elapsed.
constexpr int64_t kMsPerWord = 325;
constexpr int64_t kMsPerWordEs = 360;

int64_t ms_per_word(const Config& cfg) {
    return cfg.language == "es" ? kMsPerWordEs : kMsPerWord;
}

int64_t estimate_speech_ms(const std::string& s, const Config& cfg) {
    return (int64_t)word_count(s) * ms_per_word(cfg);
}

}  // namespace

SpeechSequencer::SpeechSequencer(const Config& cfg, int sample_rate)
    : cfg_(cfg), rate_(sample_rate) {}

void SpeechSequencer::emit(Action a) { actions_.push_back(std::move(a)); }

bool task_event_is_stale(const Event& e, uint64_t current_gen,
                         uint64_t current_epoch) {
    return e.task_gen != current_gen || e.task_epoch != current_epoch;
}

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
    task_deadline_ms_ = 0;
    llm_fillers_outstanding_ = 0;
    llm_filler_empty_streak_ = 0;
    llm_filler_speakable_ = 0;
    speech_committed_ms_ = 0;
    play_origin_ms_ = -1;
    played_since_origin_ms_ = 0;
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
    // Expected arrival of the result. The adapter jitters by +/-task_jitter_ms;
    // the centre is used on purpose: over-covering costs a filler of answer
    // delay, under-covering is dead air, and dead air is the failure mode.
    task_deadline_ms_ = t_ms + cfg_.task_ms;
    // The orchestrator dispatches the task when it sees this request; the ack
    // is generated immediately and in parallel.
    // Do NOT ask for a restatement: both 0.6B and 1.7B then answer the
    // question instead of acknowledging it (bench: 10-20% result claims).
    emit_request(TextKind::Ack,
                 "The user said: \"" + transcript_ + "\". Reply with one short "
                 "acknowledgement that says you are checking. Do not restate the "
                 "request, do not state any status or result, do not ask a question.");
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
            if (s.kind == TextKind::Ack || s.kind == TextKind::Filler) {
                ++stats_.fillers_dropped_stale;  // counted as stale speech
                drop_committed(s);
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

void SpeechSequencer::on_stale_task_result() {
    ++stats_.stale_results_dropped;
    metrics_.stale_results_dropped = stats_.stale_results_dropped;
}

void SpeechSequencer::on_task_redispatch(int64_t deadline_ms) {
    if (!turn_active()) return;
    // Re-anchoring the deadline also moves the near-result window: the bank
    // picks its shortest phrase from `task_deadline_ms_ - t_ms`, so nothing
    // else has to be reset here.
    task_deadline_ms_ = std::max<int64_t>(0, deadline_ms);
}

void SpeechSequencer::drop_committed(const TextSegment& s) {
    speech_committed_ms_ =
        std::max<int64_t>(0, speech_committed_ms_ - estimate_speech_ms(s.text, cfg_));
}

void SpeechSequencer::on_text_delta(TextKind kind, const std::string& piece,
                                    int64_t t_ms) {
    const int idx = (int)kind;
    if (!generation_active_[idx]) {
        generation_active_[idx] = true;
        ++metrics_.llm_calls;
        if (kind == TextKind::Filler) llm_filler_speakable_ = 0;
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
    const int stale_before = stats_.fillers_dropped_stale;
    feed_text(kind, std::string(), t_ms, true);
    if (kind != TextKind::Filler) return;

    // The generation is over whichever way it went: never leave the queue
    // counter stuck, or the bridge stops asking for filler for the rest of the
    // turn.
    if (llm_fillers_outstanding_ > 0) --llm_fillers_outstanding_;

    // The result (or the answer's own audio) arrived while this filler was
    // still being generated. Its clauses were dropped as stale above; queuing a
    // fresh bank phrase now would put speech in front of the answer, which is
    // the delay this guard exists to prevent.
    if (task_done_ || answer_buffered_) return;
    if (llm_filler_speakable_ > 0) {
        llm_filler_empty_streak_ = 0;
        return;
    }
    // Clauses existed but were dropped as stale: that is not an empty
    // generation, so it must not count toward the retry limit.
    if (stats_.fillers_dropped_stale > stale_before) return;

    // Empty, unspoken, or a rejected repeat: the bridge must not stall. The bank
    // is the safety net in every mode (that is what a pre-written bank is for);
    // only if it is exhausted does a retry make sense, and only a bounded one.
    ++stats_.fillers_llm_empty;
    ++llm_filler_empty_streak_;
    if (queue_bank_filler(t_ms)) return;
    if (llm_filler_empty_streak_ < kMaxLlmFillerAttempts)
        maybe_request_filler("retry", t_ms);
}

void SpeechSequencer::feed_text(TextKind kind, const std::string& piece,
                                int64_t t_ms, bool done) {
    if (!piece.empty()) pending_[(int)kind] += piece;

    // Split at sentence boundaries, not every few words: each TTS piece
    // boundary costs model lead/tail silence. The ack stays short by its own
    // word cap; fillers are already short.
    const int max_words =
        kind == TextKind::Ack ? cfg_.ack_max_words
        : kind == TextKind::Filler ? cfg_.filler_max_words
                                   : std::max(cfg_.clause_max_words, 40);

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
            speech_committed_ms_ += estimate_speech_ms(clause, cfg_);
        }
        if (kind == TextKind::Filler) {
            // A filler that arrives after the result is in is stale speech: the
            // answer owns the floor from here on.
            if (task_done_) {
                ++stats_.fillers_dropped_stale;
                continue;
            }
            if (!accept_filler_text(clause)) continue;
            ++llm_filler_speakable_;
        }
        TextSegment s;
        s.kind = kind;
        s.text = sanitize_for_speech(trim(clause));
        s.seq = (uint64_t)clauses_.size() + 1;
        s.created_ms = t_ms;
        if (kind == TextKind::Filler) {
            s.expire_ms = t_ms + 4000;
            ++stats_.fillers_queued;
            speech_committed_ms_ += estimate_speech_ms(clause, cfg_);
        }
        clauses_.push_back(std::move(s));
    }
    if (done) {
        std::string rest = sanitize_for_speech(trim(pending_[(int)kind]));
        pending_[(int)kind].clear();
        const bool usable =
            !rest.empty() && word_count(rest) > 0 &&
            (kind != TextKind::Filler ||
             (!task_done_ && accept_filler_text(rest)));
        if (usable) {
            TextSegment s;
            s.kind = kind;
            s.text = rest;
            s.seq = (uint64_t)clauses_.size() + 1;
            s.created_ms = t_ms;
            if (kind == TextKind::Filler) {
                s.expire_ms = t_ms + 4000;
                ++llm_filler_speakable_;
                ++stats_.fillers_queued;
                speech_committed_ms_ += estimate_speech_ms(rest, cfg_);
            }
            clauses_.push_back(std::move(s));
        } else if (kind == TextKind::Filler && task_done_) {
            ++stats_.fillers_dropped_stale;
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
            drop_committed(s);
            clauses_.pop_front();
            continue;
        }
        if (answer_buffered_ && s.kind == TextKind::Filler) {
            drop_committed(s);
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
    if (!turn_active()) return;
    if (answer_buffered_) return;  // the answer's PCM exists: stop bridging
    if (task_done_ && !answer_requested_) return;
    // At most two filler generations outstanding, the same depth as the clause
    // queue. One is not enough for the LLM mode: each filler costs a generation
    // plus a TTS TTFA, so a single slot leaves the ring dry between phrases.
    // Unbounded is worse: a request on every low-water tick queued a pile of
    // fillers whose audio then played ahead of the answer it was covering.
    if (llm_fillers_outstanding_ >= 2) return;
    // Two regimes, both needed:
    //  - comfortable buffer: bridge only while the committed speech is short of
    //    the expected result (+reserve for the answer's TTFT/TTFA). Filler
    //    plays ahead of the answer and cannot be retracted, so a deep pile of
    //    it is exactly what delays the answer it was covering.
    //  - buffer below low_water: always bridge. This is the watermark's stated
    //    meaning ("below this, TTS has priority and filler is requested"), and
    //    it is what covers a task result that arrives earlier than expected,
    //    which any deadline prediction gets wrong half the time.
    if (task_deadline_ms_ > 0 && buffer_ms_ >= cfg_.low_water_ms) {
        const int64_t until_result = task_deadline_ms_ - t_ms;
        if (until_result > -kResultReserveMs &&
            remaining_speech_ms() >= until_result + kResultReserveMs)
            return;
    }

    // Bank first when the mode has one; a bank phrase costs no generation.
    if (cfg_.filler_mode != Config::FillerMode::Llm) {
        if (queue_bank_filler(t_ms)) return;
        if (cfg_.filler_mode == Config::FillerMode::Bank) return;  // stay silent
    }

    // A generation that keeps producing nothing usable must not turn into a
    // request loop; three empty results in a row retire the LLM filler for this
    // turn and leave the bank (if any) as the bridge.
    if (llm_filler_empty_streak_ >= kMaxLlmFillerAttempts) return;
    ++llm_fillers_outstanding_;
    llm_filler_speakable_ = 0;
    const std::string said = recent_fillers_for_prompt();
    emit_request(TextKind::Filler,
                 "The user is waiting. You are still checking \"" + transcript_ +
                     "\". Say one short, natural holding phrase that does not "
                     "claim progress or a result, and does not repeat what you "
                     "already said" +
                     (said.empty() ? std::string(".")
                                   : ": \"" + said + "\"."));
}

bool SpeechSequencer::queue_bank_filler(int64_t t_ms) {
    // Close to the expected result, take the shortest unused phrase: the bridge
    // lands on the answer more accurately when the last piece is ~0.6 s instead
    // of ~1.5 s, and a long last filler is what delays the answer.
    const bool near_result =
        task_deadline_ms_ > 0 && task_deadline_ms_ - t_ms < 2 * kFillerEstimateMs;
    std::string text = next_bank_filler(near_result);
    while (!text.empty()) {
        if (accept_filler_text(text)) {
            TextSegment s;
            s.kind = TextKind::Filler;
            s.text = sanitize_for_speech(text);
            s.seq = 0;
            s.created_ms = t_ms;
            s.expire_ms = t_ms + 4000;
            ++stats_.fillers_queued;
            speech_committed_ms_ += estimate_speech_ms(s.text, cfg_);
            clauses_.push_back(std::move(s));
            request_next_clause(t_ms);
            return true;
        }
        // A bank phrase the LLM already used this session: try the next one.
        text = next_bank_filler(near_result);
    }
    return false;
}

// Speech still ahead of the playback position, from what this turn has already
// committed (word-based estimate minus what the sink has actually played).
int64_t SpeechSequencer::remaining_speech_ms() const {
    return std::max<int64_t>(0, speech_committed_ms_ - played_since_origin_ms_);
}

bool SpeechSequencer::accept_filler_text(const std::string& text) {
    const std::string clean = sanitize_for_speech(trim(text));
    if (!plausible_filler(clean)) {
        ++stats_.fillers_repeat_rejected;  // unusable counts as rejected speech
        return false;
    }
    const std::string norm = normalize_phrase(clean);
    for (const std::string& said : spoken_fillers_) {
        if (normalize_phrase(said) == norm) {
            ++stats_.fillers_repeat_rejected;
            return false;
        }
    }
    for (const TextSegment& s : clauses_) {
        if (s.kind == TextKind::Filler && normalize_phrase(s.text) == norm) {
            ++stats_.fillers_repeat_rejected;
            return false;
        }
    }
    spoken_fillers_.push_back(clean);
    while (spoken_fillers_.size() > kMaxSpokenFillersRemembered)
        spoken_fillers_.pop_front();
    return true;
}

std::string SpeechSequencer::recent_fillers_for_prompt() const {
    std::string out;
    size_t n = spoken_fillers_.size();
    size_t start = n > 6 ? n - 6 : 0;
    for (size_t i = start; i < n; ++i) {
        if (!out.empty()) out += " ";
        out += spoken_fillers_[i];
    }
    return out;
}

std::string SpeechSequencer::next_bank_filler(bool shortest_first) {
    // Never repeat a bank phrase: walk the ring buffer and skip anything this
    // session already used. Returns "" when the bank is exhausted.
    const int count = bank_count(cfg_.language);
    int best = -1;
    for (int i = 0; i < count; ++i) {
        const int idx = (bank_cursor_ + i) % count;
        const std::string cand = bank_phrase(cfg_.language, idx);
        if (std::find(spoken_bank_.begin(), spoken_bank_.end(), cand) !=
            spoken_bank_.end())
            continue;
        if (!shortest_first) {
            best = idx;
            break;
        }
        if (best < 0 ||
            word_count(cand) < word_count(bank_phrase(cfg_.language, best)))
            best = idx;
    }
    if (best < 0) return std::string();
    bank_cursor_ = (best + 1) % count;
    spoken_bank_.push_back(bank_phrase(cfg_.language, best));
    return bank_phrase(cfg_.language, best);
}

void SpeechSequencer::on_playback(int64_t buffer_ms, int64_t played_ms,
                                  int64_t t_ms) {
    buffer_ms_ = buffer_ms;
    played_ms_ = played_ms;
    // Playback clock for the filler budget. The origin is the moment the ack
    // started playing (played_ms counts from the start of the run, not the
    // turn), so "committed speech minus elapsed" is the speech still to come.
    if (metrics_.ack_first_played_ms > 0) {
        if (play_origin_ms_ < 0) play_origin_ms_ = played_ms;
        played_since_origin_ms_ = std::max<int64_t>(0, played_ms - play_origin_ms_);
    }

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
                drop_committed(s);
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
        // Keep two clauses queued, not one: a single in-flight filler leaves
        // the buffer dry while the next one is still synthesising.
        if (turn_active() && clauses_.size() < 2 &&
            (task_pending_ || awaiting_answer_audio) &&
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
    llm_fillers_outstanding_ = 0;
    task_deadline_ms_ = 0;
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
