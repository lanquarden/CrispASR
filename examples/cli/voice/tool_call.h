// tool_call.h — the model decides which (simulated) task runs.
//
// Until now the loop dispatched a task itself as soon as a final transcript
// existed. This module makes that decision a structured model output instead:
// the LLM is asked for exactly one JSON object, the orchestrator parses it
// leniently, and a schema-valid call selects the task's *latency/failure
// profile* in the existing TaskAdapter. Nothing else about the pipeline
// changes — the ack is still generated from the transcript, in parallel.
//
// Deliberately no grammar sampler: that is a later upstream PR (a sampler hook
// in src/chat.cpp plus one parameter field). Until then the honest question is
// "how often does a strict prompt produce a parseable call?", so the parser
// accepts what a sloppy small model actually emits and every attempt is counted
// (tool_calls / tool_calls_valid).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "engines.h"
#include "task_adapter.h"
#include "voice_types.h"

namespace crispasr {
namespace voice {

// The three tools, all simulated by TaskAdapter. Ids are stable because the
// dispatcher encodes them in the task id it hands back to the orchestrator.
enum class ToolId { DeviceStatus = 0, SetTimer = 1, Smalltalk = 2, Unknown = 3 };

const char* to_string(ToolId id);
// Lenient name lookup (lowercase, '-'/' ' folded to '_', a few aliases).
ToolId tool_from_name(const std::string& name);

// One parsed decision. `valid` means "schema-valid": a known tool with
// well-typed, in-range arguments. `reason` explains a rejection and is logged.
struct ToolCall {
    ToolId tool = ToolId::Unknown;
    std::string device;   // device_status: garage | lights | thermostat
    int minutes = 0;      // set_timer: 1..600
    bool valid = false;
    std::string reason;
    std::string raw;      // model output as received (empty in --tool-mode off)
};

// Lenient parse of one model reply. Accepts: a bare tool name, a JSON object
// wrapped in markdown fences, a JSON object with prose around it, a JSON object
// inside a JSON string, and numeric strings for `minutes`. Rejects: no object
// and no bare name, an unterminated object, an unknown tool, wrong argument
// types, out-of-range values.
ToolCall parse_tool_call(const std::string& text);

// ---------------------------------------------------------------------------
// Where the decision comes from.
//   off          no model call, dispatch device_status (the pre-M2 behaviour)
//   llm          ask the chat model (default)
//   fake_valid   deterministic valid JSON, no model call (harness)
//   fake_invalid deterministic prose, no model call (harness)
// Selected by --tool-mode on the CLI or VOICE_TOOL_MODE in the environment.
// ---------------------------------------------------------------------------
enum class ToolMode { Off, Llm, FakeValid, FakeInvalid };

const char* to_string(ToolMode m);
ToolMode tool_mode_from_env();
bool parse_tool_mode(const std::string& name, ToolMode* out);

// Tokens the tool decision may use. The JSON is ~20 tokens; 48 leaves room for
// a model that thinks out loud and still fits the TTS deadline budget.
constexpr int kToolCallMaxTokens = 48;

// The strict router prompt plus the user turn carrying the transcript.
std::vector<ChatMessage> tool_call_messages(const std::string& transcript);
// Deterministic replies for the fake modes: a keyword router (valid) and prose
// that no parser can accept (invalid).
std::string fake_tool_reply(ToolMode mode, const std::string& transcript);

// The request text handed to the task adapter. device_status keeps the raw
// transcript so its simulated result reads like the pre-M2 one; the other tools
// phrase the completed action, which is what the answer prompt quotes.
std::string tool_task_request(const ToolCall& call, const std::string& transcript);

// ---------------------------------------------------------------------------
// Per-tool latency/failure profile. device_status uses the configured 6 s task
// (with its jitter and failure injection); set_timer answers in 0-1 s;
// smalltalk needs no task at all (task_ms = 0).
// ---------------------------------------------------------------------------
struct ToolProfile {
    int latency_ms = 0;
    int jitter_ms = 0;
    int fail_percent = 0;
};

ToolProfile tool_profile(ToolId tool, const Config& cfg);

// Routes a decision to a SimulatedTaskAdapter with that tool's profile, behind
// the TaskAdapter interface the orchestrator already uses. Task ids are global
// (`local_id * 8 + tool_index`) so a cancel() needs no extra bookkeeping in the
// loop; ids restart per run because one dispatcher is created per run.
class ToolDispatcher {
   public:
    explicit ToolDispatcher(const Config& cfg);
    ~ToolDispatcher();
    ToolDispatcher(const ToolDispatcher&) = delete;
    ToolDispatcher& operator=(const ToolDispatcher&) = delete;

    // Non-blocking, like TaskAdapter::dispatch. Returns 0 when refused.
    int dispatch(const ToolCall& call, const std::string& transcript,
                 TaskCallback cb);
    void cancel(int id);
    void shutdown();

   private:
    struct Impl;
    Impl* impl_ = nullptr;
};

}  // namespace voice
}  // namespace crispasr
