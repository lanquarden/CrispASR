// crispasr_voice_loop.cpp — entry point for the realtime voice loop experiment
// (home-ops#584). Three modes:
//
//   --selftest                 GPU-free policy checks
//   --harness <scenario.json>  automated measurement run (fake or real engines)
//   --input in.wav --out out.wav   one turn with the real engines
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "arbiter.h"
#include "audio_io.h"
#include "config.h"
#include "engines.h"
#include "harness.h"
#include "sequencer.h"
#include "tool_call.h"
#include "voice_loop.h"
#include "voice_types.h"

using namespace crispasr::voice;

namespace {

int selftest_impl() {
    int failures = 0;
    auto check = [&](bool ok, const char* what) {
        std::printf("%s %s\n", ok ? "PASS" : "FAIL", what);
        if (!ok) ++failures;
    };

    Config cfg;
    cfg.normalise();

    // 1. final transcript asks for an ack and nothing else.
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("what is the garage status", 1000);
        auto a = s.take_actions();
        check(a.size() == 1 && a[0].type == ActionType::RequestText &&
                  a[0].kind == TextKind::Ack,
              "final transcript requests exactly one ack");
        check(s.state() == TurnState::Delegating, "state Delegating after final");
    }

    // 2. ack text is spoken as a clause; playback low cannot starve silently.
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("what is the garage status", 1000);
        s.take_actions();
        s.on_text_delta(TextKind::Ack, "Checking the garage now.", 1100);
        s.on_text_done(TextKind::Ack, 1100);
        auto a = s.take_actions();
        bool spoke = false;
        for (auto& x : a)
            if (x.type == ActionType::Speak && x.kind == TextKind::Ack) spoke = true;
        check(spoke, "ack clause is emitted for synthesis");
        s.metrics().ack_first_played_ms = 1150;  // orchestrator would set this
        s.on_playback(500, 0, 1200);
        auto b = s.take_actions();
        bool filler = false;
        for (auto& x : b)
            if ((x.type == ActionType::RequestText || x.type == ActionType::Speak) &&
                x.kind == TextKind::Filler)
                filler = true;
        check(filler, "low buffer while task pending requests filler");
    }

    // 3. task result asks for an answer and never claims a result early.
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("check the garage", 1000);
        s.take_actions();
        s.on_task_result(true, "the garage is closed", 7000);
        auto a = s.take_actions();
        bool answer = false;
        for (auto& x : a)
            if (x.type == ActionType::RequestText && x.kind == TextKind::Answer)
                answer = true;
        check(answer, "task result requests an answer");
        check(s.state() == TurnState::Answering, "state Answering after result");
    }

    // 4. barge-in flushes audio, cancels the task and finishes the turn.
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("check the garage", 1000);
        s.take_actions();
        s.on_barge_in(3000);
        auto a = s.take_actions();
        bool flush = false, cancel = false, finish = false;
        for (auto& x : a) {
            if (x.type == ActionType::FlushAudio) flush = true;
            if (x.type == ActionType::CancelTask) cancel = true;
            if (x.type == ActionType::FinishTurn) finish = true;
        }
        check(flush && cancel && finish, "barge-in flushes, cancels, finishes");
        check(s.state() == TurnState::Listening, "barge-in returns to Listening");
    }

    // 5. stale results are dropped, not spoken.
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("check the garage", 1000);
        s.take_actions();
        s.on_barge_in(2000);
        s.take_actions();
        s.on_task_result(true, "late result", 8000);
        auto a = s.take_actions();
        bool speaks = false;
        for (auto& x : a)
            if (x.type == ActionType::Speak) speaks = true;
        check(!speaks, "late result after barge-in is not spoken");
    }

    // 6. filler bank never repeats within a session.
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("check the garage", 1000);
        s.take_actions();
        std::vector<std::string> spoken;
        s.metrics().ack_first_played_ms = 1150;
        for (int i = 0; i < 8; ++i) {
            s.on_playback(200, 0, 1500 + i * 2000);
            for (auto& x : s.take_actions())
                if (x.type == ActionType::Speak) spoken.push_back(x.text);
        }
        bool dup = false;
        for (size_t i = 0; i < spoken.size(); ++i)
            for (size_t j = i + 1; j < spoken.size(); ++j)
                if (spoken[i] == spoken[j]) dup = true;
        check(!dup && spoken.size() >= 2, "bank fillers do not repeat");
    }

    // 7. tool-call parser: lenient about packaging, strict about content.
    {
        const ToolCall c = parse_tool_call(
            "```json\n{\"tool\":\"device_status\",\"args\":{\"device\":\"garage\"}}\n```");
        check(c.valid && c.tool == ToolId::DeviceStatus && c.device == "garage",
              "tool: fenced JSON parses");
    }
    {
        const ToolCall c = parse_tool_call("set_timer");
        check(!c.valid && c.tool == ToolId::SetTimer,
              "tool: bare tool name rejected (no args, nothing invented)");
    }
    {
        const ToolCall c =
            parse_tool_call("Sure, on it. {\"tool\":\"smalltalk\",\"args\":{}} Done.");
        check(c.valid && c.tool == ToolId::Smalltalk,
              "tool: extra prose around the object tolerated");
    }
    {
        const ToolCall c =
            parse_tool_call(
                "\"{\\\"tool\\\":\\\"device_status\\\",\\\"args\\\":{\\\"device\\\":\\\"lights\\\"}}\"");
        check(c.valid && c.tool == ToolId::DeviceStatus && c.device == "lights",
              "tool: JSON inside a quoted string unwrapped");
    }
    {
        const ToolCall c = parse_tool_call(
            "{\"tool\":\"set_timer\",\"args\":{\"minutes\":\"abc\"}}");
        check(!c.valid && c.tool == ToolId::SetTimer,
              "tool: non-numeric minutes rejected");
    }
    {
        const ToolCall c =
            parse_tool_call("{\"tool\":\"set_timer\",\"args\":{\"minutes\":\"7\"}}");
        check(c.valid && c.tool == ToolId::SetTimer && c.minutes == 7,
              "tool: numeric string coerced");
    }
    {
        const ToolCall c = parse_tool_call("{\"tool\":\"open_garage\",\"args\":{}}");
        check(!c.valid && c.tool == ToolId::Unknown, "tool: unknown tool rejected");
    }
    {
        const ToolCall c = parse_tool_call(
            "{\"tool\":\"device_status\",\"args\":{\"device\":\"gar");
        check(!c.valid, "tool: truncated JSON rejected");
    }
    {
        const ToolCall c =
            parse_tool_call("{\"tool\":\"device_status\",\"args\":{\"device\":7}}");
        check(!c.valid, "tool: wrong device type rejected");
    }
    {
        const ToolCall c =
            parse_tool_call("{\"tool\":\"set_timer\",\"args\":{\"minutes\":9999}}");
        check(!c.valid, "tool: out-of-range minutes rejected");
    }
    {
        const ToolCall c = parse_tool_call("I think I should check the garage door for you.");
        check(!c.valid && c.tool == ToolId::Unknown, "tool: plain prose rejected");
    }
    {
        const ToolProfile dev = tool_profile(ToolId::DeviceStatus, cfg);
        const ToolProfile timer = tool_profile(ToolId::SetTimer, cfg);
        const ToolProfile talk = tool_profile(ToolId::Smalltalk, cfg);
        check(dev.latency_ms == cfg.task_ms && timer.latency_ms > 0 &&
                  timer.latency_ms <= 1000 && talk.latency_ms == 0,
              "tool: latency profiles 6 s / <=1 s / 0");
    }

    // 8. a task completion is only current for the dispatch that posted it.
    {
        Event e;
        e.type = EventType::TaskResult;
        e.task_gen = 1;
        e.task_epoch = 7;
        check(!task_event_is_stale(e, 1, 7), "task event: current dispatch accepted");
        check(task_event_is_stale(e, 2, 7),
              "task event: completion from a superseded dispatch is stale");
        check(task_event_is_stale(e, 1, 8),
              "task event: completion from a flushed epoch is stale");
        Event legacy;  // no stamp at all
        legacy.type = EventType::TaskResult;
        check(task_event_is_stale(legacy, 1, 7),
              "task event: unstamped completion is stale");
    }

    // 9. a stale completion is counted, never spoken, and asks for nothing.
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("check the garage", 1000);
        s.take_actions();
        s.on_stale_task_result();
        check(s.metrics().stale_results_dropped == 1,
              "stale completion is counted");
        auto a = s.take_actions();
        check(a.empty(), "stale completion speaks nothing");
    }

    // 10. a redispatch re-anchors the expected result (F3).
    {
        SpeechSequencer s(cfg, cfg.sample_rate);
        s.on_speech_start(0);
        s.on_final("check the garage", 1000);
        s.take_actions();
        const int64_t spec_deadline = s.task_deadline_ms();
        s.on_task_redispatch(1000 + 900);  // set_timer answers in ~0.9 s
        check(spec_deadline == 1000 + cfg.task_ms && s.task_deadline_ms() == 1900,
              "redispatch moves the task deadline");
    }

    // 11. clauses that will never be spoken release their committed speech (F5),
    //     and a filler that arrives after the result is not bridged again (F4).
    {
        Config deep = cfg;
        deep.low_water_ms = 10;
        deep.target_water_ms = 50;
        deep.high_water_ms = 100;
        SpeechSequencer s(deep, deep.sample_rate);
        s.on_speech_start(0);
        s.on_final("check the garage", 1000);
        s.take_actions();
        s.metrics().ack_first_played_ms = 1100;
        s.on_playback(500, 0, 1200);  // ring above high water: clauses queue up
        s.on_text_delta(TextKind::Filler, "One moment.", 1300);
        s.on_text_done(TextKind::Filler, 1400);
        const int64_t committed = s.committed_speech_ms();
        s.on_task_result(true, "the garage is closed", 1500);
        check(committed > 0 && s.committed_speech_ms() == 0,
              "dropped clauses release their committed speech");
        s.take_actions();
        s.on_text_delta(TextKind::Filler, "Just a moment.", 1600);
        s.on_text_done(TextKind::Filler, 1650);
        bool bridged = false;
        for (auto& x : s.take_actions())
            if (x.kind == TextKind::Filler) bridged = true;
        check(!bridged, "filler finishing after the result does not queue a bank phrase");
    }

    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    // --tool-mode is a normal config flag now (config.cpp), so a scenario can
    // carry its own and the CLI still wins. Nothing to rewrite here.
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i] ? argv[i] : "", "--selftest") == 0)
            return selftest_impl();

    Config cfg = config_from_args(argc, argv);

    const CliExtras& x = config_extras();
    if (cfg.mode == "selftest")
        return selftest_impl();
    if (!x.harness_path.empty())
        return run_harness(cfg, x.harness_path);

    if (x.input_wav.empty() || cfg.out_dir.empty() || cfg.out_dir == ".") {
        std::fprintf(stderr,
                     "usage: crispasr-voice --selftest\n"
                     "       crispasr-voice --harness scenario.json "
                     "[--tool-mode off|llm|fake_valid|fake_invalid]\n"
                     "       crispasr-voice --input in.wav --out out.wav "
                     "--asr-model ... --tts-model ... --chat-model ...\n");
        return 2;
    }

    Engines e = make_crispasr_engines(cfg);
    LoopRun r = run_once(cfg, e, x.input_wav, cfg.out_dir);
    if (!r.ok) {
        std::fprintf(stderr, "run failed: %s\n", r.error.c_str());
        return 1;
    }
    std::printf("%s\n", metrics_json(r.metrics).c_str());
    return 0;
}
