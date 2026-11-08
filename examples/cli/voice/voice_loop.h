// voice_loop.h — the orchestrator entry point used by both the CLI and the
// automated harness.
#pragma once

#include <string>
#include <vector>

#include "engines.h"
#include "voice_types.h"

namespace crispasr {
namespace voice {

struct LoopRun {
    bool ok = false;
    std::string error;
    TurnMetrics metrics;
};

// Runs one turn: paced WAV in, paced WAV out, metrics collected.
// `engines` may be fake (no GPU) or real.
LoopRun run_once(const Config& cfg, Engines& engines,
                 const std::string& input_wav, const std::string& out_wav);

// Deterministic, GPU-free checks of the policy: fills a Config, drives the
// sequencer with synthetic events and asserts the action stream. Returns 0 on
// success.
int selftest();

}  // namespace voice
}  // namespace crispasr
