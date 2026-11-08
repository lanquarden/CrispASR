// config.h — CLI/config-file parsing for the voice-loop experiment.
#pragma once

#include "voice_types.h"

namespace crispasr {
namespace voice {

// Defaults < --config file < CLI flags. Recognised flags are listed in
// PLAN.md and implemented in config.cpp.
Config config_from_args(int argc, char** argv);

}  // namespace voice
}  // namespace crispasr
