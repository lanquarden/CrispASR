# M3 — Speculative opening clause

Status: design, waiting for M2 to land (voice_loop.cpp is owned by the tool-call
work until then).

## Why

Current audible latency is ~810 ms p50 at the 250 ms endpointing default:

- endpointing (true last audible sample -> VAD declared end): ~250 ms
- ack pipeline (declared end -> final transcript -> LLM -> TTS first PCM): ~260-290 ms
- playback/buffer: ~20-40 ms

M1 already removed the avoidable parts. The only two levers left are:

1. shorter endpointing (rejected: an energy VAD with no smart-turn commits
   fragments at inter-phrase pauses), and
2. start the ack pipeline *before* the endpoint.

M3 is (2), in the safest form: **pre-render, do not pre-speak**. The loop builds
the ack's audio on a stable partial transcript and only plays it when VAD
commits the turn. Nothing is said while the user is still talking.

## Design

On each partial transcript (250 ms cadence), compute a stability signal:

- take the last two partial texts;
- a prefix is *stable* once it has been identical (normalised: lowercase,
  punctuation stripped) in two consecutive partials;
- require >= 3 stable words and >= 600 ms of speech.

When stable and `--speculate` is on and no prepared ack exists for the current
speech segment:

- request an ack text for the stable prefix (separate LLM job, same prompt);
- synthesize it into a `PreparedAck { prefix, text, pcm, created_ms }` side
  buffer (no ring push);
- cap it at one prepared ack; a newer stable prefix replaces it (and the
  superseded PCM is dropped).

On `FinalTranscript`:

- if the final transcript shares the prepared prefix (normalised prefix match,
  or >= 80 % token overlap) and the prepared audio is younger than 3 s:
  push the PCM straight into the ring and mark the ack as played;
- otherwise discard it and run the normal ack path.

The answer path is unchanged. Fillers are unchanged. The tool call still comes
from the final transcript only.

## Metrics

- `eos_to_first_played_ms` and `eos_to_first_audible_ms` with/without
  `--speculate` (same fixtures, 3 runs each).
- `speculative_attempts`, `speculative_hits`, `speculative_discards`,
  `speculative_wrong_ack` (hit whose prefix did not match the final).
- A hit is only counted as a win if the audible time drops by >= 100 ms; the
  report must show the distribution, not just the mean.

## Rollback / no-go

- Revert if a wrong prepared ack is ever played (wrong-ack rate must be 0; the
  prefix check is the guard).
- Drop the feature if the p50 gain is < 100 ms, or if it raises the worst WAV
  gap by more than 20 ms.

## What this does not fix

Endpointing stays at 250 ms. M3 can hide the *pipeline* cost, not the silence
wait. Going below ~500 ms total needs a smart-turn model (HF uses one) or a
faster endpoint heuristic with a confidence signal; both are separate work.
