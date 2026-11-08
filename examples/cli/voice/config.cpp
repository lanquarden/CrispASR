// config.cpp — Config loading (flat key=value and flat/nested JSON) plus CLI
// parsing for crispasr-voice (home-ops#584).
//
// No third-party JSON library: examples/json.hpp exists in the tree, but the
// voice loop only needs a flat scalar object, and pulling a 25k-line header into
// a module that is compiled on every iteration of an experiment is not worth it.
// jsonlite flattens nested objects into dotted keys ("expect.max_gap_ms"), so
// scenario files can group their thresholds while Config loading stays flat.
//
// Precedence for config_from_args(): defaults < --config file < CLI flags.

#include "harness.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "voice_types.h"

namespace crispasr {
namespace voice {

// ---------------------------------------------------------------------------
// jsonlite — tiny reader for flat scalar objects.
// ---------------------------------------------------------------------------
namespace jsonlite {
namespace {

bool is_ws(char c) {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

void skip_ws(const std::string& s, size_t& i) {
    while (i < s.size() && is_ws(s[i])) ++i;
}

bool append_utf8(uint32_t cp, std::string* out) {
    if (cp <= 0x7f) {
        out->push_back((char)cp);
    } else if (cp <= 0x7ff) {
        out->push_back((char)(0xc0 | (cp >> 6)));
        out->push_back((char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0xffff) {
        out->push_back((char)(0xe0 | (cp >> 12)));
        out->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
        out->push_back((char)(0x80 | (cp & 0x3f)));
    } else if (cp <= 0x10ffff) {
        out->push_back((char)(0xf0 | (cp >> 18)));
        out->push_back((char)(0x80 | ((cp >> 12) & 0x3f)));
        out->push_back((char)(0x80 | ((cp >> 6) & 0x3f)));
        out->push_back((char)(0x80 | (cp & 0x3f)));
    } else {
        return false;
    }
    return true;
}

int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

struct Parser {
    const std::string& s;
    size_t i = 0;
    Object* out = nullptr;
    std::string err;

    Parser(const std::string& text, Object* obj) : s(text), out(obj) {}

    bool fail(const std::string& msg) {
        if (err.empty()) {
            char buf[64];
            std::snprintf(buf, sizeof(buf), " (at byte %zu)", i);
            err = msg + buf;
        }
        return false;
    }

    bool parse_string(std::string* dst) {
        if (i >= s.size() || s[i] != '"') return fail("expected string");
        ++i;
        dst->clear();
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return true;
            if (c == '\\') {
                if (i >= s.size()) return fail("unterminated escape");
                char e = s[i++];
                switch (e) {
                    case '"': dst->push_back('"'); break;
                    case '\\': dst->push_back('\\'); break;
                    case '/': dst->push_back('/'); break;
                    case 'b': dst->push_back('\b'); break;
                    case 'f': dst->push_back('\f'); break;
                    case 'n': dst->push_back('\n'); break;
                    case 'r': dst->push_back('\r'); break;
                    case 't': dst->push_back('\t'); break;
                    case 'u': {
                        if (i + 4 > s.size()) return fail("short \\u escape");
                        uint32_t cp = 0;
                        for (int k = 0; k < 4; ++k) {
                            int h = hex_val(s[i + k]);
                            if (h < 0) return fail("bad \\u escape");
                            cp = (cp << 4) | (uint32_t)h;
                        }
                        i += 4;
                        // Surrogate pair: combine when present, else emit U+FFFD.
                        if (cp >= 0xd800 && cp <= 0xdbff && i + 1 < s.size() &&
                            s[i] == '\\' && s[i + 1] == 'u') {
                            size_t save = i;
                            i += 2;
                            if (i + 4 <= s.size()) {
                                uint32_t lo = 0;
                                bool ok = true;
                                for (int k = 0; k < 4; ++k) {
                                    int h = hex_val(s[i + k]);
                                    if (h < 0) { ok = false; break; }
                                    lo = (lo << 4) | (uint32_t)h;
                                }
                                if (ok && lo >= 0xdc00 && lo <= 0xdfff) {
                                    i += 4;
                                    cp = 0x10000 + ((cp - 0xd800) << 10) + (lo - 0xdc00);
                                } else {
                                    i = save;
                                }
                            } else {
                                i = save;
                            }
                        }
                        if (!append_utf8(cp, dst)) return fail("bad codepoint");
                        break;
                    }
                    default: return fail("unknown escape");
                }
                continue;
            }
            dst->push_back(c);
        }
        return fail("unterminated string");
    }

    // Index just past a balanced [...] group, honouring strings.
    bool skip_array(size_t* end) {
        int depth = 0;
        while (i < s.size()) {
            char c = s[i];
            if (c == '"') {
                std::string ignored;
                if (!parse_string(&ignored)) return false;
                continue;
            }
            if (c == '[') ++depth;
            if (c == ']') {
                --depth;
                ++i;
                if (depth == 0) { *end = i; return true; }
                continue;
            }
            ++i;
        }
        return fail("unterminated array");
    }

    bool parse_scalar(const std::string& key) {
        if (i >= s.size()) return fail("expected value");
        if (s[i] == '{') return parse_object(key);
        if (s[i] == '"') {
            std::string v;
            if (!parse_string(&v)) return false;
            (*out)[key] = v;
            return true;
        }
        if (s[i] == '[') {
            size_t start = i, end = 0;
            if (!skip_array(&end)) return false;
            (*out)[key] = s.substr(start, end - start);
            return true;
        }
        size_t start = i;
        while (i < s.size() && s[i] != ',' && s[i] != '}' && s[i] != ']') ++i;
        size_t stop = i;
        while (stop > start && is_ws(s[stop - 1])) --stop;
        std::string tok = s.substr(start, stop - start);
        if (tok.empty()) return fail("expected value");
        // Reject garbage: numbers, true/false/null only.
        bool numeric = tok.size() > 0;
        size_t k = 0;
        if (k < tok.size() && (tok[k] == '-' || tok[k] == '+')) ++k;
        bool digits = false;
        while (k < tok.size() && std::isdigit((unsigned char)tok[k])) { ++k; digits = true; }
        if (k < tok.size() && tok[k] == '.') {
            ++k;
            while (k < tok.size() && std::isdigit((unsigned char)tok[k])) { ++k; digits = true; }
        }
        if (k < tok.size() && (tok[k] == 'e' || tok[k] == 'E')) {
            ++k;
            if (k < tok.size() && (tok[k] == '-' || tok[k] == '+')) ++k;
            while (k < tok.size() && std::isdigit((unsigned char)tok[k])) ++k;
        }
        numeric = numeric && digits && k == tok.size();
        if (!numeric && tok != "true" && tok != "false" && tok != "null") {
            return fail("unquoted value is not a number/bool/null: '" + tok + "'");
        }
        (*out)[key] = tok;
        return true;
    }

    bool parse_object(const std::string& prefix) {
        if (i >= s.size() || s[i] != '{') return fail("expected '{'");
        ++i;
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}') { ++i; return true; }
        while (true) {
            skip_ws(s, i);
            std::string key;
            if (!parse_string(&key)) return false;
            skip_ws(s, i);
            if (i >= s.size() || s[i] != ':') return fail("expected ':'");
            ++i;
            skip_ws(s, i);
            std::string full = prefix.empty() ? key : (prefix + "." + key);
            if (!parse_scalar(full)) return false;
            skip_ws(s, i);
            if (i < s.size() && s[i] == ',') { ++i; continue; }
            if (i < s.size() && s[i] == '}') { ++i; return true; }
            return fail("expected ',' or '}'");
        }
    }
};

}  // namespace

bool parse(const std::string& text, Object* out, std::string* err) {
    if (!out) return false;
    out->clear();
    size_t i = 0;
    skip_ws(text, i);
    if (i >= text.size() || text[i] != '{') {
        if (err) *err = "top level is not a JSON object";
        return false;
    }
    Parser p(text, out);
    p.i = i;
    if (!p.parse_object("")) {
        if (err) *err = p.err.empty() ? "parse error" : p.err;
        return false;
    }
    skip_ws(text, p.i);
    if (p.i != text.size()) {
        if (err) *err = "trailing data after top-level object";
        return false;
    }
    return true;
}

bool load_file(const std::string& path, Object* out, std::string* err) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        if (err) *err = "cannot open '" + path + "': " + std::strerror(errno);
        return false;
    }
    std::string text;
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    std::fclose(f);
    std::string perr;
    if (!parse(text, out, &perr)) {
        if (err) *err = path + ": " + perr;
        return false;
    }
    return true;
}

}  // namespace jsonlite

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
namespace {

std::string trim(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && std::isspace((unsigned char)s[b])) ++b;
    while (e > b && std::isspace((unsigned char)s[e - 1])) --e;
    return s.substr(b, e - b);
}

std::string lower(const std::string& s) {
    std::string r = s;
    for (char& c : r) c = (char)std::tolower((unsigned char)c);
    return r;
}

bool parse_bool(const std::string& v, bool* dst) {
    std::string s = lower(trim(v));
    if (s == "1" || s == "true" || s == "yes" || s == "on") { *dst = true; return true; }
    if (s == "0" || s == "false" || s == "no" || s == "off") { *dst = false; return true; }
    return false;
}

bool parse_int(const std::string& v, long long* dst) {
    std::string s = trim(v);
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    long long x = std::strtoll(s.c_str(), &end, 10);
    if (errno != 0 || !end || *end != '\0') return false;
    *dst = x;
    return true;
}

bool parse_double(const std::string& v, double* dst) {
    std::string s = trim(v);
    if (s.empty()) return false;
    errno = 0;
    char* end = nullptr;
    double x = std::strtod(s.c_str(), &end);
    if (errno != 0 || !end || *end != '\0' || !std::isfinite(x)) return false;
    *dst = x;
    return true;
}

bool parse_filler_mode(const std::string& v, Config::FillerMode* dst) {
    std::string s = lower(trim(v));
    if (s == "bank") { *dst = Config::FillerMode::Bank; return true; }
    if (s == "llm") { *dst = Config::FillerMode::Llm; return true; }
    if (s == "bank_then_llm" || s == "bank,llm" || s == "bankthenllm") {
        *dst = Config::FillerMode::BankThenLlm;
        return true;
    }
    return false;
}

void warn(std::vector<std::string>* warnings, const std::string& msg) {
    if (warnings) warnings->push_back(msg);
}

}  // namespace

// ---------------------------------------------------------------------------
// Config from a flat object (JSON keys or key=value lines)
// ---------------------------------------------------------------------------
bool apply_config_object(const jsonlite::Object& obj, Config* cfg,
                         std::vector<std::string>* warnings) {
    if (!cfg) return false;
    for (const auto& kv : obj) {
        const std::string& key = kv.first;
        const std::string& raw = kv.second;
        const std::string lk = lower(key);
        bool bad = false;

        auto as_int = [&](int* dst) {
            long long x = 0;
            if (!parse_int(raw, &x)) { bad = true; return; }
            *dst = (int)x;
        };
        auto as_bool = [&](bool* dst) {
            if (!parse_bool(raw, dst)) bad = true;
        };
        auto as_str = [&](std::string* dst) { *dst = raw; };

        if (lk == "sample_rate") as_int(&cfg->sample_rate);
        else if (lk == "frame_ms") as_int(&cfg->frame_ms);
        else if (lk == "vad_min_speech_ms") as_int(&cfg->vad_min_speech_ms);
        else if (lk == "vad_min_silence_ms") as_int(&cfg->vad_min_silence_ms);
        else if (lk == "vad_energy_threshold") {
            double d = 0;
            if (parse_double(raw, &d)) cfg->vad_energy_threshold = (float)d;
            else bad = true;
        } else if (lk == "vad_gate_during_playback") as_bool(&cfg->vad_gate_during_playback);
        else if (lk == "stt_partial_ms") as_int(&cfg->stt_partial_ms);
        else if (lk == "stt_window_ms") as_int(&cfg->stt_window_ms);
        else if (lk == "stt_partials_enabled") as_bool(&cfg->stt_partials_enabled);
        else if (lk == "low_water_ms") as_int(&cfg->low_water_ms);
        else if (lk == "target_water_ms") as_int(&cfg->target_water_ms);
        else if (lk == "high_water_ms") as_int(&cfg->high_water_ms);
        else if (lk == "clause_max_words") as_int(&cfg->clause_max_words);
        else if (lk == "ack_max_words") as_int(&cfg->ack_max_words);
        else if (lk == "filler_max_words") as_int(&cfg->filler_max_words);
        else if (lk == "answer_max_clause_tokens") as_int(&cfg->answer_max_clause_tokens);
        else if (lk == "task_ms") as_int(&cfg->task_ms);
        else if (lk == "task_jitter_ms") as_int(&cfg->task_jitter_ms);
        else if (lk == "task_fail_percent") as_int(&cfg->task_fail_percent);
        else if (lk == "filler_mode") {
            if (!parse_filler_mode(raw, &cfg->filler_mode)) bad = true;
        } else if (lk == "filler_bank_size") as_int(&cfg->filler_bank_size);
        else if (lk == "prerender_filler") as_bool(&cfg->prerender_filler);
        else if (lk == "barge_in_enabled") as_bool(&cfg->barge_in_enabled);
        else if (lk == "speculate") as_bool(&cfg->speculate);
        else if (lk == "barge_in_max_ms") as_int(&cfg->barge_in_max_ms);
        else if (lk == "max_turn_ms") as_int(&cfg->max_turn_ms);
        else if (lk == "asr_backend") as_str(&cfg->asr_backend);
        else if (lk == "asr_model") as_str(&cfg->asr_model);
        else if (lk == "tts_backend") as_str(&cfg->tts_backend);
        else if (lk == "tts_model") as_str(&cfg->tts_model);
        else if (lk == "tts_tokenizer") as_str(&cfg->tts_tokenizer);
        else if (lk == "tts_voice_dir" || lk == "voice_dir") as_str(&cfg->tts_voice_dir);
        else if (lk == "tts_voice" || lk == "voice") as_str(&cfg->tts_voice);
        else if (lk == "chat_model") as_str(&cfg->chat_model);
        else if (lk == "chat_gpu_layers") as_int(&cfg->chat_gpu_layers);
        else if (lk == "chat_ctx") as_int(&cfg->chat_ctx);
        else if (lk == "runs") as_int(&cfg->runs);
        else if (lk == "warmup_runs" || lk == "warmup") as_int(&cfg->warmup_runs);
        else if (lk == "out_dir" || lk == "out") as_str(&cfg->out_dir);
        else if (lk == "mode") as_str(&cfg->mode);
        else if (lk == "harness_path" || lk == "harness") as_str(&cfg->harness_path);
        else if (lk == "input_wav" || lk == "input") as_str(&cfg->input_wav);
        else if (lk == "out_wav") as_str(&cfg->out_wav);
        else if (lk == "metrics_path" || lk == "metrics") as_str(&cfg->metrics_path);
        else {
            warn(warnings, "config: ignoring unknown key '" + key + "'");
            continue;
        }

        if (bad) {
            warn(warnings, "config: bad value for '" + key + "': '" + raw + "'");
        }
    }
    return true;
}

bool load_config_file(const std::string& path, Config* cfg,
                      std::vector<std::string>* warnings) {
    if (!cfg) return false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        warn(warnings, "config: cannot open '" + path + "': " + std::strerror(errno));
        return false;
    }
    std::string text;
    char buf[4096];
    size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) text.append(buf, n);
    std::fclose(f);

    size_t first = 0;
    while (first < text.size() && std::isspace((unsigned char)text[first])) ++first;
    if (first < text.size() && text[first] == '{') {
        jsonlite::Object obj;
        std::string err;
        if (!jsonlite::parse(text, &obj, &err)) {
            warn(warnings, "config: " + path + ": " + err);
            return false;
        }
        return apply_config_object(obj, cfg, warnings);
    }

    // Flat key=value lines. '#' starts a comment line.
    jsonlite::Object obj;
    size_t pos = 0;
    int lineno = 0;
    while (pos <= text.size()) {
        size_t nl = text.find('\n', pos);
        std::string line = (nl == std::string::npos) ? text.substr(pos)
                                                    : text.substr(pos, nl - pos);
        ++lineno;
        pos = (nl == std::string::npos) ? text.size() + 1 : nl + 1;
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        size_t eq = t.find('=');
        if (eq == std::string::npos) {
            warn(warnings, "config: " + path + ":" + std::to_string(lineno) +
                               ": expected key=value");
            continue;
        }
        std::string key = trim(t.substr(0, eq));
        std::string val = trim(t.substr(eq + 1));
        if (key.empty()) {
            warn(warnings, "config: " + path + ":" + std::to_string(lineno) + ": empty key");
            continue;
        }
        // Strip matching quotes so `asr_model="x.gguf"` works.
        if (val.size() >= 2 && ((val.front() == '"' && val.back() == '"') ||
                                (val.front() == '\'' && val.back() == '\''))) {
            val = val.substr(1, val.size() - 2);
        }
        obj[key] = val;
    }
    return apply_config_object(obj, cfg, warnings);
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------
const char* cli_usage() {
    return
        "crispasr-voice — realtime voice-loop experiment harness (home-ops#584)\n"
        "\n"
        "usage: crispasr-voice --harness tests/voice/scenarios/<name>.json [flags]\n"
        "       crispasr-voice --input in.wav --out out.wav [flags]\n"
        "\n"
        "models:\n"
        "  --asr-model PATH         ASR GGUF model\n"
        "  --tts-model PATH         TTS GGUF model\n"
        "  --tts-tokenizer PATH     TTS tokenizer\n"
        "  --voice-dir DIR          voicepack directory\n"
        "  --chat-model PATH        chat LLM GGUF\n"
        "\n"
        "io:\n"
        "  --input PATH             input WAV (single run)\n"
        "  --out PATH|DIR           output WAV, or output directory in harness mode\n"
        "  --metrics PATH           metrics.jsonl destination\n"
        "  --harness PATH           scenario JSON; runs the automated harness\n"
        "\n"
        "experiment:\n"
        "  --runs N                 measured runs (warm-up excluded)\n"
        "  --warmup N               warm-up runs, not measured\n"
        "  --filler MODE            bank | llm | bank_then_llm\n"
        "  --speculate              start the ack on stable partials (M3)\n"
        "  --low-water-ms N         playback low-water mark\n"
        "  --target-water-ms N      target playback depth\n"
        "  --high-water-ms N        stop queuing text above this\n"
        "  --stt-partial-ms N       partial re-transcribe cadence\n"
        "  --task-ms N              simulated delegation latency\n"
        "  --config PATH            flat key=value (or JSON) defaults file\n"
        "  --selftest               GPU-free policy checks, no audio\n"
        "  --help                   this text\n";
}

namespace {
CliExtras g_extras;
}  // namespace

const CliExtras& config_extras() { return g_extras; }

Config config_from_args(int argc, char** argv) {
    Config cfg;
    g_extras = CliExtras();

    // Two passes: --config first so explicit flags win.
    std::string config_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i] ? argv[i] : "";
        if (a.rfind("--config", 0) == 0) {
            if (a.size() > 8 && a[8] == '=') {
                config_path = a.substr(9);
            } else if (i + 1 < argc) {
                config_path = argv[i + 1];
            }
        }
    }
    if (!config_path.empty()) {
        std::vector<std::string> warnings;
        load_config_file(config_path, &cfg, &warnings);
        for (const auto& w : warnings) std::fprintf(stderr, "warning: %s\n", w.c_str());
    }

    auto need_value = [&](int& i, const std::string& flag) -> std::string {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s needs a value\n", flag.c_str());
            return std::string();
        }
        ++i;
        return argv[i] ? std::string(argv[i]) : std::string();
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i] ? argv[i] : "";
        if (arg.empty() || arg[0] != '-') {
            std::fprintf(stderr, "warning: ignoring positional argument '%s'\n", arg.c_str());
            continue;
        }
        std::string flag = arg;
        std::string inline_val;
        bool has_inline = false;
        size_t eq = arg.find('=');
        if (eq != std::string::npos) {
            flag = arg.substr(0, eq);
            inline_val = arg.substr(eq + 1);
            has_inline = true;
        }
        auto value = [&]() { return has_inline ? inline_val : need_value(i, flag); };

        long long iv = 0;
        auto int_flag = [&](int* dst) {
            std::string v = value();
            if (!parse_int(v, &iv)) {
                std::fprintf(stderr, "warning: %s expects an integer, got '%s'\n",
                             flag.c_str(), v.c_str());
                return;
            }
            *dst = (int)iv;
        };

        if (flag == "--help" || flag == "-h") {
            g_extras.help_requested = true;
        } else if (flag == "--config") {
            (void)value();  // already handled
        } else if (flag == "--selftest") {
            cfg.mode = "selftest";
        } else if (flag == "--mode") {
            cfg.mode = value();
        } else if (flag == "--asr-model") cfg.asr_model = value();
        else if (flag == "--tts-model") cfg.tts_model = value();
        else if (flag == "--tts-tokenizer") cfg.tts_tokenizer = value();
        else if (flag == "--voice-dir") cfg.tts_voice_dir = value();
        else if (flag == "--chat-model") cfg.chat_model = value();
        else if (flag == "--input") {
            g_extras.input_wav = value();
            cfg.input_wav = g_extras.input_wav;
        } else if (flag == "--out") {
            // Harness mode treats it as a directory, single-run mode as a WAV;
            // set both so either reader sees it.
            cfg.out_dir = value();
            cfg.out_wav = cfg.out_dir;
        } else if (flag == "--metrics") {
            g_extras.metrics_path = value();
            cfg.metrics_path = g_extras.metrics_path;
        } else if (flag == "--harness") {
            g_extras.harness_path = value();
            cfg.harness_path = g_extras.harness_path;
            if (cfg.mode.empty()) cfg.mode = "harness";
        }
        else if (flag == "--runs") {
            int_flag(&cfg.runs);
            g_extras.runs_set = true;
        } else if (flag == "--warmup") {
            int_flag(&cfg.warmup_runs);
            g_extras.warmup_set = true;
        }
        else if (flag == "--filler") {
            std::string v = value();
            if (!parse_filler_mode(v, &cfg.filler_mode)) {
                std::fprintf(stderr, "warning: --filler expects bank|llm|bank_then_llm, got '%s'\n",
                             v.c_str());
            }
        } else if (flag == "--speculate") {
            bool b = true;
            if (has_inline) parse_bool(inline_val, &b);
            cfg.speculate = b;
        } else if (flag == "--low-water-ms") int_flag(&cfg.low_water_ms);
        else if (flag == "--target-water-ms") int_flag(&cfg.target_water_ms);
        else if (flag == "--high-water-ms") int_flag(&cfg.high_water_ms);
        else if (flag == "--stt-partial-ms") int_flag(&cfg.stt_partial_ms);
        else if (flag == "--vad-min-silence-ms") int_flag(&cfg.vad_min_silence_ms);
        else if (flag == "--vad-min-speech-ms") int_flag(&cfg.vad_min_speech_ms);
        else if (flag == "--clause-max-words") int_flag(&cfg.clause_max_words);
        else if (flag == "--language") cfg.language = value();
        else if (flag == "--task-ms") {
            int_flag(&cfg.task_ms);
            g_extras.task_ms_set = true;
        }
        else {
            std::fprintf(stderr, "warning: unknown flag '%s' (try --help)\n", flag.c_str());
        }
    }

    // A config file can carry the CLI-only paths too; mirror them back so
    // callers that read config_extras() see them regardless of where they
    // came from.
    if (g_extras.input_wav.empty()) g_extras.input_wav = cfg.input_wav;
    if (g_extras.harness_path.empty()) g_extras.harness_path = cfg.harness_path;
    if (g_extras.metrics_path.empty()) g_extras.metrics_path = cfg.metrics_path;
    if (!g_extras.harness_path.empty() && cfg.mode.empty()) cfg.mode = "harness";

    if (g_extras.help_requested) std::fputs(cli_usage(), stdout);
    return cfg;
}

}  // namespace voice
}  // namespace crispasr
