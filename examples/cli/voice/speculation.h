// speculation.h — M3: pre-render the ack on a stable partial transcript.
//
// Design (docs/584-realtime-voice/SPECULATION.md): the ack's LLM + TTS cost is
// paid while the user is still talking, on a *stable* partial transcript, into
// a side buffer that is never pushed into the playback ring. Only a committed
// turn (VAD end -> final transcript) may speak it, and only when the final
// transcript still agrees with the prefix the ack was generated for.
//
// This header is text logic and bookkeeping only: no audio, no clocks beyond
// the timestamps handed in, no engine. The orchestrator owns the PCM and the
// thread boundaries; everything here is pure so the selftest can pin the
// stability rule, the match guard and the attempt/hit/discard accounting
// without a GPU.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace crispasr {
namespace voice {
namespace speculation {

// A prepared ack may not be played later than this after it was created: what
// was said 3 s ago is not an acknowledgement of what was said now.
constexpr int64_t kMaxAgeMs = 3000;
// A prefix is stable only with this many identical leading words ...
constexpr size_t kMinStableWords = 3;
// ... and only after this much speech. An energy VAD has no smart turn, so a
// 250 ms inter-phrase pause can split an utterance; waiting 600 ms keeps the
// prefix from being the first phrase of a two-phrase question.
constexpr int64_t kMinSpeechMs = 600;

// lowercase ascii-alnum words joined by single spaces; punctuation, quotes,
// markdown and non-ascii bytes are separators (so "The status!" and
// "the status" are the same text).
std::string normalise(const std::string& text);

std::vector<std::string> tokens(const std::string& normalised_text);

// The word prefix shared by two consecutive partial transcripts, normalised.
// Empty when it is shorter than kMinStableWords or when less than kMinSpeechMs
// of speech has been heard.
std::string stable_prefix(const std::string& previous_text,
                          const std::string& current_text, int64_t speech_ms);

// True when the final transcript still answers the prepared prefix: a normalised
// prefix match, or at least 80 % of the prepared words present in the final
// (ASR revisions move words, and a strictly prefix-only check throws away acks
// that are still about the same request). This is the wrong-ack guard: a
// prepared ack that does not pass is never spoken.
bool prefix_matches(const std::string& prepared_prefix,
                    const std::string& final_transcript);

// Quotes, markdown and control characters are rendered by qwen3-tts as pauses
// or vocalisations. The prepared ack is synthesised before the sequencer's own
// sanitiser sees it, so the same noise class is stripped here first: the PCM we
// pre-render must carry the words we would have spoken.
std::string strip_speech_noise(const std::string& text);

// The speculative ack request must ask for the same thing the normal ack
// request asks for, or a speculative ack answers a differently phrased
// question than the path it replaces. Mirrors SpeechSequencer::on_final.
std::string ack_user_prompt(const std::string& said);

// ---------------------------------------------------------------------------
// The single prepared ack's bookkeeping.
//
// At most one prepared ack exists. A newer stable prefix replaces it only when
// it no longer answers the same request (a growth of the same utterance keeps
// the live one: replacing on every partial would re-run the LLM and the TTS
// every 250 ms and never finish anything). Every drop is counted, and only a
// *matching, fresh* commit is a hit, so a wrong prepared ack can never inflate
// the hit count.
// ---------------------------------------------------------------------------
class PreparedAckBook {
   public:
    enum class Start {
        None,     // nothing to do: the live prepared ack still answers this
        Begin,    // start the first prepared ack for this turn
        Replace,  // the live one was dropped (and counted) and is replaced
    };
    enum class Outcome {
        Miss,      // no prepared ack was waiting: the normal ack path runs
        Stale,     // older than kMaxAgeMs (counted as a discard)
        Mismatch,  // the final transcript no longer matches the prefix (discard)
        Hit,       // safe to play: the caller pushes the PCM and skips the ack
    };

    Start on_stable(const std::string& stable_prefix, int64_t t_ms);
    // Called once at the final transcript. `now_ms` is the commit time.
    Outcome commit(const std::string& final_transcript, int64_t now_ms);
    // A barge-in or a new turn: nothing pre-rendered survives it.
    void invalidate();

    bool live() const { return live_; }
    bool current(uint64_t generation) const {
        return live_ && generation_ == generation;
    }
    const std::string& prefix() const { return prefix_; }
    uint64_t generation() const { return generation_; }
    int64_t created_ms() const { return created_ms_; }

    int attempts() const { return attempts_; }
    int hits() const { return hits_; }
    int discards() const { return discards_; }

   private:
    void drop(bool count);
    void start(const std::string& stable_prefix, int64_t t_ms);

    bool live_ = false;
    std::string prefix_;
    int64_t created_ms_ = 0;
    uint64_t generation_ = 0;
    int attempts_ = 0;
    int hits_ = 0;
    int discards_ = 0;
};

}  // namespace speculation
}  // namespace voice
}  // namespace crispasr
