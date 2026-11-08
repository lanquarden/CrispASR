// crispasr_voice_loop.cpp — entry point for the realtime voice loop experiment
// (home-ops#584). Three modes:
//
//   --selftest                 GPU-free policy checks
//   --harness <scenario.json>  automated measurement run (fake or real engines)
//   --input in.wav --out out.wav   one turn with the real engines
#include <cstdio>
#include <cstring>
#include <string>

#include "arbiter.h"
#include "audio_io.h"
#include "config.h"
#include "engines.h"
#include "harness.h"
#include "sequencer.h"
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

    std::printf("%d failure(s)\n", failures);
    return failures == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--selftest") == 0) return selftest_impl();

    Config cfg = config_from_args(argc, argv);

    const CliExtras& x = config_extras();
    if (cfg.mode == "selftest")
        return selftest_impl();
    if (!x.harness_path.empty())
        return run_harness(cfg, x.harness_path);

    if (x.input_wav.empty() || cfg.out_dir.empty() || cfg.out_dir == ".") {
        std::fprintf(stderr,
                     "usage: crispasr-voice --selftest\n"
                     "       crispasr-voice --harness scenario.json\n"
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
