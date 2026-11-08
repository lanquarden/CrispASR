// harness.cpp — automated scenario harness for the realtime voice loop
// (home-ops#584).
//
// One process runs one scenario: warm-up runs first (unmeasured, they exist to
// load models and warm caches), then N measured runs of run_once(). Per run we
// write <out>/<scenario>/run<N>.wav, one JSONL metrics line and that run's
// per-piece timeline rows; at the end we aggregate p50/p95 over the measured
// runs into summary.json, print the table plus a per-kind piece summary, and
// write timeline.csv with every run's rows.
//
// timeline.csv is the attribution tool, not a scoring one: it holds one row per
// TTS piece, one per LLM generation and one per 100 ms of an active turn for
// the playback ring (columns in voice_types.h). metrics.jsonl stays the scoring
// record and check_metrics.py the gate.
//
// Engine timings are not Config fields, so they travel through the environment
// (see engines_fake.cpp). The barge-in marker is one of them:
// VOICE_BARGE_IN_AT_MS is read by the orchestrator, which fires BARGE_IN when
// the played audio clock reaches that offset. The harness sets it per scenario
// and restores the previous value afterwards.

#include "harness.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <sys/stat.h>
#include <sys/types.h>

#include "engines.h"
#include "voice_loop.h"
#include "voice_types.h"

namespace crispasr {
namespace voice {

namespace {

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------
std::string dir_name(const std::string& path) {
    size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return std::string(".");
    if (slash == 0) return std::string("/");
    return path.substr(0, slash);
}

std::string file_stem(const std::string& path) {
    size_t slash = path.find_last_of('/');
    std::string base = (slash == std::string::npos) ? path : path.substr(slash + 1);
    size_t dot = base.find_last_of('.');
    return (dot == std::string::npos) ? base : base.substr(0, dot);
}

bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool make_dirs(const std::string& path) {
    if (path.empty()) return false;
    std::string cur;
    size_t i = 0;
    if (path[0] == '/') { cur = "/"; i = 1; }
    while (i <= path.size()) {
        size_t slash = path.find('/', i);
        std::string part = (slash == std::string::npos) ? path.substr(i)
                                                        : path.substr(i, slash - i);
        if (!part.empty()) {
            if (cur.empty()) cur = part;
            else if (cur == "/") cur += part;
            else cur += "/" + part;
            struct stat st;
            if (::stat(cur.c_str(), &st) != 0) {
                if (::mkdir(cur.c_str(), 0775) != 0) {
                    std::fprintf(stderr, "harness: mkdir '%s' failed\n", cur.c_str());
                    return false;
                }
            } else if (!S_ISDIR(st.st_mode)) {
                std::fprintf(stderr, "harness: '%s' exists and is not a directory\n", cur.c_str());
                return false;
            }
        }
        if (slash == std::string::npos) break;
        i = slash + 1;
    }
    return true;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back((char)c);
                }
        }
    }
    return out;
}

// Emits `value` as a JSON literal when it already looks like one (number,
// true/false/null), otherwise as a quoted string.
std::string json_scalar(const std::string& value) {
    if (value == "true" || value == "false" || value == "null") return value;
    size_t i = 0;
    if (i < value.size() && (value[i] == '-' || value[i] == '+')) ++i;
    bool digits = false;
    while (i < value.size() && value[i] >= '0' && value[i] <= '9') { ++i; digits = true; }
    if (i < value.size() && value[i] == '.') {
        ++i;
        while (i < value.size() && value[i] >= '0' && value[i] <= '9') { ++i; digits = true; }
    }
    if (digits && i == value.size()) return value;
    return "\"" + json_escape(value) + "\"";
}

// Sets or clears an environment variable for the lifetime of the object and
// puts the previous value back on destruction. Empty value == unset.
class EnvGuard {
   public:
    EnvGuard(const char* name, const std::string& value) : name_(name) {
        const char* old = std::getenv(name_.c_str());
        if (old) { had_old_ = true; old_ = old; }
        apply(value);
    }
    ~EnvGuard() { apply(had_old_ ? old_ : std::string()); }
    EnvGuard(const EnvGuard&) = delete;
    EnvGuard& operator=(const EnvGuard&) = delete;

   private:
    static void apply(const std::string& name, const std::string& value) {
#if defined(_WIN32)
        if (value.empty()) _putenv_s(name.c_str(), "");
        else _putenv_s(name.c_str(), value.c_str());
#else
        if (value.empty()) ::unsetenv(name.c_str());
        else ::setenv(name.c_str(), value.c_str(), 1);
#endif
    }
    void apply(const std::string& value) { apply(name_, value); }

    std::string name_;
    std::string old_;
    bool had_old_ = false;
};

// ---------------------------------------------------------------------------
// scenario
// ---------------------------------------------------------------------------
struct Scenario {
    std::string path;
    std::string name;
    std::string input;      // as written in the scenario
    std::string input_wav;  // resolved
    int runs = 3;
    int warmup = 1;
    int task_ms = -1;
    std::string language;
    int task_fail_percent = -1;
    int barge_in_at_ms = -1;
    bool has_transcript = false;
    std::string transcript;
    std::string tts_ttfa_ms;  // raw strings: they go straight to the environment
    std::string tts_rate;
    std::string llm_ttft_ms;
    bool has_tts_ttfa_every = false;
    bool tts_ttfa_every = false;
    jsonlite::Object expect;
};

int to_int(const std::string& s, int dflt) {
    char* end = nullptr;
    long v = std::strtol(s.c_str(), &end, 10);
    if (!end || end == s.c_str() || *end != '\0') return dflt;
    return (int)v;
}

// Fixtures are generated by tests/voice/make_fixtures.py into
// tests/voice/fixtures/. Scenarios name the file and we resolve it against
// cwd, the scenario directory and the fixtures directory, in that order.
std::string resolve_input(const std::string& name, const std::string& scenario_path) {
    if (name.empty()) return std::string();
    if (name[0] == '/') return file_exists(name) ? name : std::string();
    const std::string candidates[] = {
        name,
        dir_name(scenario_path) + "/" + name,
        dir_name(scenario_path) + "/../fixtures/" + name,
        "tests/voice/fixtures/" + name,
        dir_name(scenario_path) + "/fixtures/" + name,
    };
    for (const auto& c : candidates) {
        if (file_exists(c)) return c;
    }
    return std::string();
}

bool load_scenario(const std::string& path, Scenario* sc, std::string* err) {
    jsonlite::Object obj;
    if (!jsonlite::load_file(path, &obj, err)) return false;
    sc->path = path;
    sc->name = file_stem(path);

    for (const auto& kv : obj) {
        const std::string& key = kv.first;
        const std::string& val = kv.second;
        if (key == "name") sc->name = val;
        else if (key == "input" || key == "input_wav") sc->input = val;
        else if (key == "runs") sc->runs = to_int(val, sc->runs);
        else if (key == "warmup" || key == "warmup_runs") sc->warmup = to_int(val, sc->warmup);
        else if (key == "task_ms") sc->task_ms = to_int(val, sc->task_ms);
        else if (key == "language") sc->language = val;
        else if (key == "task_fail_percent") {
            sc->task_fail_percent = to_int(val, sc->task_fail_percent);
        } else if (key == "barge_in_at_ms") {
            sc->barge_in_at_ms = to_int(val, sc->barge_in_at_ms);
        } else if (key == "transcript") {
            sc->has_transcript = true;
            sc->transcript = val;
        } else if (key == "tts_ttfa_every") {
            sc->has_tts_ttfa_every = true;
            sc->tts_ttfa_every = to_int(val, 0) != 0 || val == "true";
        } else if (key == "tts_ttfa_ms" || key == "tts_rate" || key == "llm_ttft_ms") {
            if (key == "tts_ttfa_ms") sc->tts_ttfa_ms = val;
            else if (key == "tts_rate") sc->tts_rate = val;
            else sc->llm_ttft_ms = val;
        } else if (key.rfind("expect.", 0) == 0) {
            sc->expect[key.substr(7)] = val;
        } else if (key == "expect") {
            // Scalar "expect" is meaningless; an object was flattened already.
            if (err) *err = path + ": 'expect' must be an object";
            return false;
        } else {
            std::fprintf(stderr, "harness: %s: ignoring unknown key '%s'\n", path.c_str(),
                         key.c_str());
        }
    }
    if (sc->input.empty()) {
        if (err) *err = path + ": missing \"input\" wav";
        return false;
    }
    sc->input_wav = resolve_input(sc->input, path);
    if (sc->input_wav.empty()) {
        if (err) {
            *err = path + ": cannot find input wav '" + sc->input +
                   "' (run python3 tests/voice/make_fixtures.py)";
        }
        return false;
    }
    if (sc->runs <= 0) sc->runs = 1;
    if (sc->warmup < 0) sc->warmup = 0;
    return true;
}

// ---------------------------------------------------------------------------
// aggregation
// ---------------------------------------------------------------------------
struct Agg {
    double p50 = 0;
    double p95 = 0;
    double min_v = 0;
    double max_v = 0;
    int n = 0;
};

double percentile(std::vector<double> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = std::ceil(q * (double)v.size()) - 1.0;  // nearest rank
    if (idx < 0.0) idx = 0.0;
    size_t i = (size_t)idx;
    if (i >= v.size()) i = v.size() - 1;
    return v[i];
}

using FieldFn = std::function<double(const TurnMetrics&)>;

// Everything check_metrics.py can assert on, grouped the way the table reads.
// Derived accessors call the methods on TurnMetrics, so the harness needs no
// extra plumbing from metrics.cpp.
const std::vector<std::pair<std::string, FieldFn>>& metric_fields() {
    static const std::vector<std::pair<std::string, FieldFn>> f = {
        {"eos_to_first_played_ms", [](const TurnMetrics& m) { return (double)m.eos_to_first_played_ms(); }},
        {"eos_to_final_ms", [](const TurnMetrics& m) { return (double)m.eos_to_final_ms(); }},
        {"eos_to_ack_text_ms", [](const TurnMetrics& m) { return (double)m.eos_to_ack_text_ms(); }},
        {"ack_first_pcm_ms", [](const TurnMetrics& m) { return (double)m.ack_first_pcm_ms; }},
        {"result_to_answer_played_ms", [](const TurnMetrics& m) { return (double)m.result_to_answer_played_ms(); }},
        {"max_gap_ms", [](const TurnMetrics& m) { return (double)m.max_gap_ms(); }},
        {"total_gap_ms", [](const TurnMetrics& m) { return (double)m.total_gap_ms(); }},
        {"final_transcript_ms", [](const TurnMetrics& m) { return (double)m.final_transcript_ms; }},
        {"ack_text_ms", [](const TurnMetrics& m) { return (double)m.ack_text_ms; }},
        {"ack_first_played_ms", [](const TurnMetrics& m) { return (double)m.ack_first_played_ms; }},
        {"task_dispatch_ms", [](const TurnMetrics& m) { return (double)m.task_dispatch_ms; }},
        {"task_result_ms", [](const TurnMetrics& m) { return (double)m.task_result_ms; }},
        {"answer_text_ms", [](const TurnMetrics& m) { return (double)m.answer_text_ms; }},
        {"answer_first_pcm_ms", [](const TurnMetrics& m) { return (double)m.answer_first_pcm_ms; }},
        {"answer_first_played_ms", [](const TurnMetrics& m) { return (double)m.answer_first_played_ms; }},
        {"end_ms", [](const TurnMetrics& m) { return (double)m.end_ms; }},
        {"stt_partials", [](const TurnMetrics& m) { return (double)m.stt_partials; }},
        {"stt_partials_dropped", [](const TurnMetrics& m) { return (double)m.stt_partials_dropped; }},
        {"stt_partial_cost_ms", [](const TurnMetrics& m) { return (double)m.stt_partial_cost_ms; }},
        {"stt_final_cost_ms", [](const TurnMetrics& m) { return (double)m.stt_final_cost_ms; }},
        {"llm_calls", [](const TurnMetrics& m) { return (double)m.llm_calls; }},
        {"llm_ttft_ms", [](const TurnMetrics& m) { return (double)m.llm_ttft_ms; }},
        {"llm_tokens", [](const TurnMetrics& m) { return (double)m.llm_tokens; }},
        {"tool_calls", [](const TurnMetrics& m) { return (double)m.tool_calls; }},
        {"tool_calls_valid", [](const TurnMetrics& m) { return (double)m.tool_calls_valid; }},
        {"tts_pieces", [](const TurnMetrics& m) { return (double)m.tts_pieces; }},
        {"tts_ttfa_ms", [](const TurnMetrics& m) { return (double)m.tts_ttfa_ms; }},
        {"tts_block_max_ms", [](const TurnMetrics& m) { return (double)m.tts_block_max_ms; }},
        {"fillers_played", [](const TurnMetrics& m) { return (double)m.fillers_played; }},
        {"fillers_dropped_stale", [](const TurnMetrics& m) { return (double)m.fillers_dropped_stale; }},
        {"stale_results_dropped", [](const TurnMetrics& m) { return (double)m.stale_results_dropped; }},
        {"barge_ins", [](const TurnMetrics& m) { return (double)m.barge_ins; }},
        {"barge_in_stop_ms", [](const TurnMetrics& m) { return (double)m.barge_in_stop_ms; }},
        {"gap_events", [](const TurnMetrics& m) { return (double)m.gaps.size(); }},
    };
    return f;
}

// Adds "run"/"warmup"/"scenario" to a serialised TurnMetrics object so the
// JSONL can be read run-by-run. metrics_json() owns the record itself;
// voice_types.h is a contract header, so no field is added to TurnMetrics.
std::string annotate_metrics_line(const std::string& line, const std::string& scenario,
                                  int run, bool warmup) {
    std::string base = line;
    while (!base.empty() && (base.back() == '\n' || base.back() == '\r' || base.back() == ' ')) {
        base.pop_back();
    }
    std::string extra = "\"scenario\":\"" + json_escape(scenario) + "\",\"run\":" +
                        std::to_string(run) + ",\"warmup\":" + (warmup ? "true" : "false");
    if (base.size() >= 2 && base.front() == '{' && base.back() == '}') {
        std::string inner = base.substr(1, base.size() - 2);
        bool empty = inner.find_first_not_of(" \t\r\n") == std::string::npos;
        return "{" + (empty ? std::string() : inner + ",") + extra + "}";
    }
    // Defensive: metrics_json() returned something unexpected.
    return "{" + extra + "}";
}

// Per-kind piece summary for the printed table: how many TTS pieces of each
// kind a run produced and the largest silence between the audible onsets of two
// consecutive pieces (attributed to the *later* piece). A big number against
// `filler` means the filler queue ran dry; a big number against `answer` means
// the answer's first PCM arrived late, which is a different fix.
struct KindSummary {
    int count = 0;
    int64_t max_gap_ms = -1;
    int64_t gap_run = 0;  // run whose gap is the max, for drilling into the CSV
};

void print_piece_summary(const std::vector<std::vector<TimelineRow>>& runs) {
    std::map<std::string, KindSummary> by_kind;
    KindSummary all;
    for (const auto& rows : runs) {
        std::vector<const TimelineRow*> tts;
        for (const TimelineRow& r : rows) {
            if (r.stage == "tts") tts.push_back(&r);
        }
        std::stable_sort(tts.begin(), tts.end(),
                         [](const TimelineRow* a, const TimelineRow* b) {
                             return a->t_first_audible_ms < b->t_first_audible_ms;
                         });
        int64_t prev = -1;
        for (const TimelineRow* r : tts) {
            KindSummary& s = by_kind[r->kind];
            ++s.count;
            ++all.count;
            // A piece that never became audible (flushed by a barge-in, or the
            // engine returned nothing) counts as a piece but has no onset, so
            // it cannot bound a gap.
            if (r->t_first_audible_ms < 0) continue;
            if (prev >= 0) {
                const int64_t gap = r->t_first_audible_ms - prev;
                if (gap > s.max_gap_ms) {
                    s.max_gap_ms = gap;
                    s.gap_run = r->run;
                }
                if (gap > all.max_gap_ms) {
                    all.max_gap_ms = gap;
                    all.gap_run = r->run;
                }
            }
            prev = r->t_first_audible_ms;
        }
    }
    if (by_kind.empty()) return;
    auto gap_str = [](int64_t gap) {
        return gap < 0 ? std::string("-") : std::to_string((long long)gap);
    };
    std::printf("timeline: tts pieces, audible-onset gap to the previous piece\n");
    std::printf("%-12s %6s %12s %6s\n", "kind", "count", "max_gap_ms", "run");
    for (const auto& kv : by_kind) {
        std::printf("%-12s %6d %12s %6lld\n", kv.first.c_str(), kv.second.count,
                    gap_str(kv.second.max_gap_ms).c_str(),
                    (long long)kv.second.gap_run);
    }
    std::printf("%-12s %6d %12s %6lld\n", "all", all.count,
                gap_str(all.max_gap_ms).c_str(), (long long)all.gap_run);
    std::fflush(stdout);
}

void print_table(const std::string& name, int runs, int warmup, int failed,
                 const std::vector<std::pair<std::string, Agg>>& rows) {
    std::printf("\nscenario %s: %d measured run(s), %d warm-up, %d failed\n", name.c_str(),
                runs, warmup, failed);
    std::printf("%-30s %10s %10s %10s %10s\n", "metric", "p50", "p95", "min", "max");
    std::printf("%-30s %10s %10s %10s %10s\n", "------------------------------", "----------",
                "----------", "----------", "----------");
    for (const auto& r : rows) {
        std::printf("%-30s %10.1f %10.1f %10.1f %10.1f\n", r.first.c_str(), r.second.p50,
                    r.second.p95, r.second.min_v, r.second.max_v);
    }
    std::printf("\n");
    std::fflush(stdout);
}

}  // namespace

// ---------------------------------------------------------------------------
// run_harness
// ---------------------------------------------------------------------------
int run_harness(const Config& cfg_in, const std::string& scenario_path) {
    Scenario sc;
    std::string err;
    if (!load_scenario(scenario_path, &sc, &err)) {
        std::fprintf(stderr, "harness: %s\n", err.c_str());
        return 2;
    }

    Config cfg = cfg_in;
    cfg.normalise();

    // Scenario values win unless the user overrode them on the command line.
    const CliExtras& extras = config_extras();
    if (!extras.runs_set) cfg.runs = sc.runs;
    if (!extras.warmup_set) cfg.warmup_runs = sc.warmup;
    if (!extras.task_ms_set && sc.task_ms >= 0) cfg.task_ms = sc.task_ms;
    if (!sc.language.empty()) cfg.language = sc.language;
    if (sc.task_fail_percent >= 0) cfg.task_fail_percent = sc.task_fail_percent;
    if (cfg.runs <= 0) cfg.runs = 1;
    if (cfg.warmup_runs < 0) cfg.warmup_runs = 0;

    const std::string out_root =
        (cfg.out_dir.empty() || cfg.out_dir == ".") ? std::string("tests/voice/out") : cfg.out_dir;
    const std::string scen_dir = out_root + "/" + sc.name;
    if (!make_dirs(scen_dir)) return 2;

    std::printf("harness: scenario '%s' (%s)\n", sc.name.c_str(), scenario_path.c_str());
    std::printf("harness: input '%s', task_ms=%d, fail=%d%%, runs=%d+%d warmup, out '%s'\n",
                sc.input_wav.c_str(), cfg.task_ms, cfg.task_fail_percent, cfg.runs,
                cfg.warmup_runs, scen_dir.c_str());

    // Fake-engine timings and the barge-in marker travel through the
    // environment; see the header comment of engines_fake.cpp.
    std::vector<std::unique_ptr<EnvGuard>> env;
    env.emplace_back(new EnvGuard("VOICE_BARGE_IN_AT_MS",
                                  sc.barge_in_at_ms >= 0 ? std::to_string(sc.barge_in_at_ms)
                                                         : std::string()));
    if (sc.has_transcript) {
        env.emplace_back(new EnvGuard("VOICE_FAKE_TRANSCRIPT", sc.transcript));
    }
    if (!sc.tts_ttfa_ms.empty()) {
        env.emplace_back(new EnvGuard("VOICE_FAKE_TTS_TTFA_MS", sc.tts_ttfa_ms));
    }
    if (!sc.tts_rate.empty()) {
        env.emplace_back(new EnvGuard("VOICE_FAKE_TTS_RATE", sc.tts_rate));
    }
    if (!sc.llm_ttft_ms.empty()) {
        env.emplace_back(new EnvGuard("VOICE_FAKE_LLM_TTFT_MS", sc.llm_ttft_ms));
    }
    if (sc.has_tts_ttfa_every) {
        // Per-clause TTFA instead of once-per-process model warm-up.
        env.emplace_back(new EnvGuard("VOICE_FAKE_TTS_TTFA_ALWAYS", sc.tts_ttfa_every ? "1" : ""));
    }

    const std::string metrics_target =
        !extras.metrics_path.empty() ? extras.metrics_path : cfg.metrics_path;

    const bool real_engines = std::getenv("VOICE_REAL_ENGINES") != nullptr;
    Engines engines = real_engines ? make_crispasr_engines(cfg)
                                   : make_fake_engines(cfg);
    if (!engines.asr || !engines.tts || !engines.llm) {
        std::fprintf(stderr, "harness: fake engine construction failed\n");
        return 2;
    }

    const int total_runs = cfg.warmup_runs + cfg.runs;
    std::vector<std::string> jsonl;
    std::string timeline = timeline_csv_header();
    std::vector<TurnMetrics> measured;
    // Timeline rows of the measured runs, for the printed piece summary.
    std::vector<std::vector<TimelineRow>> measured_rows;
    int failed = 0;

    for (int run = 1; run <= total_runs; ++run) {
        const bool warmup = run <= cfg.warmup_runs;
        const std::string out_wav = scen_dir + "/run" + std::to_string(run) + ".wav";
        std::printf("harness: run %d/%d%s ... ", run, total_runs, warmup ? " (warm-up)" : "");
        std::fflush(stdout);

        // The run number travels into the process-wide trace, so every row the
        // run produces carries it.
        timeline_trace().set_run(run);
        LoopRun result = run_once(cfg, engines, sc.input_wav, out_wav);
        std::vector<TimelineRow> run_rows = timeline_trace().rows();
        if (!result.ok) {
            ++failed;
            std::printf("FAILED: %s\n", result.error.c_str());
            std::fprintf(stderr, "harness: run %d failed: %s\n", run, result.error.c_str());
        } else {
            std::printf("ok  eos->first_played=%lldms  max_gap=%lldms  fillers=%d\n",
                        (long long)result.metrics.eos_to_first_played_ms(),
                        (long long)result.metrics.max_gap_ms(), result.metrics.fillers_played);
        }

        jsonl.push_back(annotate_metrics_line(metrics_json(result.metrics), sc.name, run, warmup));
        for (const TimelineRow& row : run_rows) timeline += timeline_csv_row(row);
        if (!warmup) {
            measured.push_back(result.metrics);
            measured_rows.push_back(std::move(run_rows));
        }
    }

    // ---- aggregate -------------------------------------------------------
    std::vector<std::pair<std::string, Agg>> rows;
    for (const auto& f : metric_fields()) {
        std::vector<double> vals;
        vals.reserve(measured.size());
        for (const auto& m : measured) vals.push_back(f.second(m));
        Agg a;
        a.n = (int)vals.size();
        if (!vals.empty()) {
            a.p50 = percentile(vals, 0.50);
            a.p95 = percentile(vals, 0.95);
            a.min_v = *std::min_element(vals.begin(), vals.end());
            a.max_v = *std::max_element(vals.begin(), vals.end());
        }
        rows.emplace_back(f.first, a);
    }

    print_table(sc.name, cfg.runs, cfg.warmup_runs, failed, rows);
    print_piece_summary(measured_rows);

    // ---- files -----------------------------------------------------------
    const std::string jsonl_path = scen_dir + "/metrics.jsonl";
    FILE* jf = std::fopen(jsonl_path.c_str(), "wb");
    if (!jf) {
        std::fprintf(stderr, "harness: cannot write '%s'\n", jsonl_path.c_str());
        return 2;
    }
    for (const auto& line : jsonl) std::fprintf(jf, "%s\n", line.c_str());
    std::fclose(jf);

    if (!metrics_target.empty() && metrics_target != jsonl_path) {
        make_dirs(dir_name(metrics_target));
        FILE* ef = std::fopen(metrics_target.c_str(), "wb");
        if (!ef) {
            std::fprintf(stderr, "harness: cannot write '%s'\n", metrics_target.c_str());
        } else {
            for (const auto& line : jsonl) std::fprintf(ef, "%s\n", line.c_str());
            std::fclose(ef);
        }
    }

    const std::string timeline_path = scen_dir + "/timeline.csv";
    if (FILE* tf = std::fopen(timeline_path.c_str(), "wb")) {
        std::fwrite(timeline.data(), 1, timeline.size(), tf);
        std::fclose(tf);
    }

    std::string summary;
    summary += "{\n";
    summary += "  \"scenario\": \"" + json_escape(sc.name) + "\",\n";
    summary += "  \"scenario_path\": \"" + json_escape(scenario_path) + "\",\n";
    summary += "  \"input_wav\": \"" + json_escape(sc.input_wav) + "\",\n";
    summary += "  \"task_ms\": " + std::to_string(cfg.task_ms) + ",\n";
    summary += "  \"task_fail_percent\": " + std::to_string(cfg.task_fail_percent) + ",\n";
    summary += "  \"runs\": " + std::to_string(cfg.runs) + ",\n";
    summary += "  \"warmup\": " + std::to_string(cfg.warmup_runs) + ",\n";
    summary += "  \"runs_total\": " + std::to_string(total_runs) + ",\n";
    summary += "  \"runs_failed\": " + std::to_string(failed) + ",\n";
    if (sc.barge_in_at_ms >= 0) {
        summary += "  \"barge_in_at_ms\": " + std::to_string(sc.barge_in_at_ms) + ",\n";
    }
    summary += std::string("  \"ok\": ") + (failed == 0 ? "true" : "false") + ",\n";
    summary += "  \"expect\": {";
    bool first = true;
    for (const auto& kv : sc.expect) {
        if (!first) summary += ", ";
        first = false;
        summary += "\"" + json_escape(kv.first) + "\": " + json_scalar(kv.second);
    }
    summary += "},\n";
    summary += "  \"metrics\": {\n";
    for (size_t i = 0; i < rows.size(); ++i) {
        const auto& r = rows[i];
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "    \"%s\": {\"p50\": %.1f, \"p95\": %.1f, \"min\": %.1f, \"max\": %.1f, \"n\": %d}",
                      r.first.c_str(), r.second.p50, r.second.p95, r.second.min_v,
                      r.second.max_v, r.second.n);
        summary += buf;
        summary += (i + 1 == rows.size()) ? "\n" : ",\n";
    }
    summary += "  }\n}\n";

    const std::string summary_path = scen_dir + "/summary.json";
    FILE* sf = std::fopen(summary_path.c_str(), "wb");
    if (!sf) {
        std::fprintf(stderr, "harness: cannot write '%s'\n", summary_path.c_str());
        return 2;
    }
    std::fwrite(summary.data(), 1, summary.size(), sf);
    std::fclose(sf);

    std::printf("harness: wrote %s, %s, %s\n", summary_path.c_str(), jsonl_path.c_str(),
                timeline_path.c_str());
    if (failed > 0) {
        std::fprintf(stderr, "harness: %d of %d run(s) failed\n", failed, total_runs);
        return 1;
    }
    return 0;
}

}  // namespace voice
}  // namespace crispasr
