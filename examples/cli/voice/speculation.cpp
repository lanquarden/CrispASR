// speculation.cpp — M3 text logic and prepared-ack accounting. See speculation.h.
#include "speculation.h"

#include <cctype>

namespace crispasr {
namespace voice {
namespace speculation {
namespace {

bool is_separator(unsigned char c) {
    // Non-ascii bytes are separators: accents are not part of a word identity
    // here, and both sides of every comparison (partial prefix, final
    // transcript) are normalised the same way, so a Spanish or English run
    // matches itself.
    return c >= 0x80 || !std::isalnum(c);
}

bool is_speech_noise(unsigned char c) {
    if (c < 0x20 || c == 0x7F) return true;  // control characters
    switch (c) {
        case '"':
        case '\'':
        case '`':
        case '*':
        case '#':
        case '_':
            return true;
        default:
            return false;
    }
}

}  // namespace

std::string normalise(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool space = false;
    for (char ch : text) {
        const unsigned char c = (unsigned char)ch;
        if (is_separator(c)) {
            space = true;
            continue;
        }
        if (space && !out.empty()) out += ' ';
        space = false;
        out += (char)std::tolower(c);
    }
    return out;
}

std::vector<std::string> tokens(const std::string& normalised_text) {
    std::vector<std::string> out;
    size_t i = 0;
    const size_t n = normalised_text.size();
    while (i < n) {
        while (i < n && normalised_text[i] == ' ') ++i;
        size_t j = i;
        while (j < n && normalised_text[j] != ' ') ++j;
        if (j > i) out.push_back(normalised_text.substr(i, j - i));
        i = j;
    }
    return out;
}

std::string stable_prefix(const std::string& previous_text,
                          const std::string& current_text, int64_t speech_ms) {
    if (speech_ms < kMinSpeechMs) return std::string();
    const std::vector<std::string> prev = tokens(normalise(previous_text));
    const std::vector<std::string> cur = tokens(normalise(current_text));
    size_t shared = 0;
    while (shared < prev.size() && shared < cur.size() &&
           prev[shared] == cur[shared])
        ++shared;
    if (shared < kMinStableWords) return std::string();
    std::string out;
    for (size_t i = 0; i < shared; ++i) {
        if (i) out += ' ';
        out += cur[i];
    }
    return out;
}

bool prefix_matches(const std::string& prepared_prefix,
                    const std::string& final_transcript) {
    const std::vector<std::string> p = tokens(normalise(prepared_prefix));
    const std::vector<std::string> f = tokens(normalise(final_transcript));
    if (p.empty() || f.empty()) return false;
    if (p.size() <= f.size()) {
        bool equal_prefix = true;
        for (size_t i = 0; i < p.size(); ++i) {
            if (p[i] != f[i]) {
                equal_prefix = false;
                break;
            }
        }
        if (equal_prefix) return true;
    }
    // >= 80 % of the prepared words appear in the final transcript.
    std::vector<bool> used(f.size(), false);
    size_t hit = 0;
    for (const std::string& w : p) {
        for (size_t i = 0; i < f.size(); ++i) {
            if (!used[i] && f[i] == w) {
                used[i] = true;
                ++hit;
                break;
            }
        }
    }
    return hit * 5 >= p.size() * 4;
}

std::string strip_speech_noise(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    bool space = false;
    for (char ch : text) {
        const unsigned char c = (unsigned char)ch;
        if (is_speech_noise(c) || (c < 0x80 && std::isspace(c))) {
            space = true;
            continue;
        }
        if (space && !out.empty()) out += ' ';
        space = false;
        out += ch;
    }
    return out;
}

std::string ack_user_prompt(const std::string& said) {
    return "The user said: \"" + said + "\". Reply with one short "
           "acknowledgement that says you are checking. Do not restate the "
           "request, do not state any status or result, do not ask a question.";
}

// ---------------------------------------------------------------------------
// PreparedAckBook
// ---------------------------------------------------------------------------
void PreparedAckBook::start(const std::string& stable, int64_t t_ms) {
    live_ = true;
    prefix_ = stable;
    created_ms_ = t_ms;
    ++generation_;  // monotonic: a late job can never resurrect it
    ++attempts_;
}

void PreparedAckBook::drop(bool count) {
    if (count) ++discards_;
    live_ = false;
    prefix_.clear();
    created_ms_ = 0;
}

PreparedAckBook::Start PreparedAckBook::on_stable(
    const std::string& stable_prefix, int64_t t_ms) {
    if (stable_prefix.empty()) return Start::None;
    // A live prepared ack that already answers this prefix is kept: the partial
    // that made it stable only grew, so re-running the LLM and the TTS would
    // restart the work every cadence tick and never finish it.
    if (live_ && prefix_matches(prefix_, stable_prefix)) return Start::None;
    const bool replacing = live_;
    if (replacing) drop(true);  // superseded: its PCM is dropped
    start(stable_prefix, t_ms);
    return replacing ? Start::Replace : Start::Begin;
}

PreparedAckBook::Outcome PreparedAckBook::commit(
    const std::string& final_transcript, int64_t now_ms) {
    if (!live_) return Outcome::Miss;
    const bool stale = now_ms - created_ms_ >= kMaxAgeMs;
    const bool matches = prefix_matches(prefix_, final_transcript);
    if (stale || !matches) {
        drop(true);
        return stale ? Outcome::Stale : Outcome::Mismatch;
    }
    ++hits_;
    drop(false);  // consumed by a hit, so it is not also a discard
    return Outcome::Hit;
}

void PreparedAckBook::invalidate() {
    if (live_) drop(true);
}

}  // namespace speculation
}  // namespace voice
}  // namespace crispasr
