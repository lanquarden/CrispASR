// harness.h — automated scenario harness and CLI config plumbing for the
// realtime voice-loop experiment (home-ops#584).
//
// The harness is the scoring authority: it replays fixture WAVs at real time
// through run_once() with the fake engines, records per-run metrics and
// aggregates p50/p95 across the non-warm-up runs. tests/voice/check_metrics.py
// turns expect{} thresholds into pass/fail.
//
// Scenario JSON (flat or one level of nesting; nested objects are flattened to
// dotted keys by jsonlite):
//
//   {
//     "name": "delegation",            // optional, defaults to the file stem
//     "input": "question_en.wav",      // resolved against the fixtures dir
//     "task_ms": 6000,
//     "task_fail_percent": 0,
//     "runs": 3,                        // measured runs
//     "warmup": 1,                      // unmeasured runs, executed first
//     "barge_in_at_ms": 3000,           // optional
//     "task_result_hold_ms": 300,       // optional: delay the task completion
//                                        // after its generation check (F1 test)
//     "transcript": "...",              // optional final ASR text for fakes
//     "language": "es",                 // optional en|es (CLI --language wins)
//     "tool_mode": "fake_valid",        // optional off|llm|fake_valid|fake_invalid
//                                       // (CLI --tool-mode wins)
//     "tts_ttfa_ms": 350,               // optional fake-engine timing override
//     "tts_rate": 4.0,                  // optional
//     "tts_ttfa_every": 1,              // optional: pay TTFA on every clause
//     "llm_ttft_ms": 80,                // optional
//     "expect": { "max_gap_ms": 150 }   // thresholds for check_metrics.py
//   }
//
// Timings that have no Config field are passed to the fake engines through the
// environment (documented in engines_fake.cpp):
//   VOICE_FAKE_TRANSCRIPT, VOICE_FAKE_TTS_TTFA_MS, VOICE_FAKE_TTS_RATE,
//   VOICE_FAKE_LLM_TTFT_MS, VOICE_BARGE_IN_AT_MS.
// The harness sets them for the duration of a scenario and restores the
// previous values afterwards.
//
// Output layout: <out_dir>/<scenario>/run<N>.wav, metrics.jsonl, timeline.csv,
// summary.json. run<N> counts warm-up runs first (run1..run<warmup>).

#pragma once

#include <map>
#include <string>
#include <vector>

#include "voice_types.h"

namespace crispasr {
namespace voice {

// ---------------------------------------------------------------------------
// jsonlite: minimal JSON reader for flat scalar objects (implementation in
// config.cpp). Nested objects flatten to dotted keys; arrays are kept as raw
// text under their key. Shared by config and scenario loading.
// ---------------------------------------------------------------------------
namespace jsonlite {

using Object = std::map<std::string, std::string>;

// Parses a single top-level JSON object. Returns false and fills `err` on
// malformed input.
bool parse(const std::string& text, Object* out, std::string* err);
bool load_file(const std::string& path, Object* out, std::string* err);

}  // namespace jsonlite

// CLI values that Config has no field for. voice_types.h is a contract header
// and is not edited; these three flags therefore live beside the Config.
struct CliExtras {
    std::string input_wav;     // --input
    std::string harness_path;  // --harness
    std::string metrics_path;  // --metrics
    bool help_requested = false;
    // Scenario files carry their own runs/warmup/task_ms; an explicit flag
    // overrides the scenario instead of being overridden by it.
    bool runs_set = false;
    bool warmup_set = false;
    bool task_ms_set = false;
    bool language_set = false;
    bool tool_mode_set = false;
};

const CliExtras& config_extras();
const char* cli_usage();

// Flags from PLAN.md: --asr-model --tts-model --tts-tokenizer --voice-dir
// --chat-model --input --out --metrics --harness --runs --warmup --filler
// --speculate --low-water-ms --target-water-ms --high-water-ms
// --stt-partial-ms --task-ms (plus --config/--help). Accepts both
// `--flag value` and `--flag=value`. Unknown flags warn on stderr.
Config config_from_args(int argc, char** argv);

// Applies a flat key=value file or a JSON object of Config fields.
bool load_config_file(const std::string& path, Config* cfg,
                      std::vector<std::string>* warnings);
bool apply_config_object(const jsonlite::Object& obj, Config* cfg,
                         std::vector<std::string>* warnings);

// Runs every run of one scenario, writes audio/metrics/summary, prints an
// aggregate table. Returns 0 when all measured runs succeed.
int run_harness(const Config& cfg, const std::string& scenario_path);

}  // namespace voice
}  // namespace crispasr
