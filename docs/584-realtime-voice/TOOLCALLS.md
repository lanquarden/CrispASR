# 584 — M2: the model decides the task (tool calls)

Date: 2026-11-09. Branch `feat/584-realtime-voice`. Supersedes defect 4 in
`RESULTS.md` ("no model tool call yet"). No grammar sampler: this PR is the
strict prompt + lenient parse + repair path, and reports how often it works.

## What changed

| file | change |
|---|---|
| `examples/cli/voice/tool_call.{h,cpp}` | new: schema, parser, prompt, per-tool profiles, `ToolDispatcher` |
| `examples/cli/voice/voice_loop.cpp` | dispatch path only: ack unchanged, task now selected by the model (speculative + correction) |
| `examples/cli/voice/crispasr_voice_loop.cpp` | `--tool-mode`, 12 parser/profile selftest cases |
| `tests/voice/scenarios/toolcall_{valid,timer,invalid}.json` | new fake-engine scenarios (both decision paths) |
| `docs/584-realtime-voice/TOOLCALLS.md` | this file |

Untouched by this change: `sequencer.cpp`, `voice_types.*`, `engines*`,
`harness*`, `task_adapter.*`, `tests/voice/check_metrics.py`.

## Schema

One object, three tools, all simulated by the existing `TaskAdapter`:

```json
{"tool":"device_status","args":{"device":"garage"}}   // garage|lights|thermostat, ~6 s
{"tool":"set_timer","args":{"minutes":5}}             // 1..600, answers in 0.8±0.2 s
{"tool":"smalltalk","args":{}}                        // no task needed, task_ms = 0
```

`device_status` reuses the configured task profile (`task_ms`, jitter,
`task_fail_percent`), so `task_failure` still exercises the failure path.
`set_timer` and `smalltalk` get their own profiles; `tool_profile()` is the one
place that decides them.

## Prompt

Strict system prompt (`tool_call_system_prompt`, in `tool_call.cpp`): the three
JSON examples with their argument domains, then

> Rules: one object, double quotes, no prose, no markdown, no code fences, no
> explanation. Never invent a tool name. If the request is unclear, use
> device_status for the garage.

User turn: `Request: "<transcript>"` + `Reply with the one JSON object.`
`max_tokens = 48`. Generation temperature is **0.4**, which is as low as this
layer can ask: `engines_crispasr.cpp` hardcodes `gp.temperature = 0.4f` and is
not this milestone's file. `LlmEngine::generate` has no temperature parameter,
so the prompt does the work.

## Parser rules (lenient packaging, strict content)

Accepted:
- a bare tool name (`set_timer`, `device_status`, trailing punctuation allowed),
  with defaults: `device_status` → `garage`, `set_timer` → 5 minutes
- the first **balanced** `{...}` anywhere in the reply (prose before/after, code
  fences, stray backticks, a `json` language tag)
- a JSON object wrapped in one layer of quotes (`"{\"tool\":...}"`)
- `minutes` as a JSON number or a numeric string (`"7"`, `" 7 "`, `7.0`)
- device synonyms (`door`, `light`, `lamps`, `heat`, `temperature`), any case,
  `-`/`_`/space folded in names and values
- extra keys inside `args` for `smalltalk` (a chatty model often adds one)

Rejected (→ fallback, `tool_calls_valid = 0`):
- text with `{` but no balanced object (truncated JSON)
- prose with no object and no bare tool name
- unknown tool, `tool` not a string, `args` not an object
- `device` missing / not a string / not one of the three devices
- `minutes` missing, non-numeric, fractional, or outside 1..600

Rationale: packaging errors are the model fighting the format; content errors
are the model choosing wrongly. Only the second class is a real
tool-call failure, and only it should ever need a grammar.

## Dispatch

```
FINAL ──► seq.on_final()  ──► ack LLM request        (unchanged, first in FIFO)
      └─► dispatch device_status(garage)             (speculative, pre-M2 instant)
          └─► tool LLM request                       (queued behind the ack)
                ├─ valid & device_status(garage) → keep the running task  ("kept")
                ├─ valid, anything else          → cancel + dispatch that profile ("redispatch")
                └─ invalid / unknown / empty     → keep device_status + transcript ("fallback")
```

Why speculative: the sequencer anchors its filler reserve to the **final
transcript** (`task_deadline_ms_ = t + task_ms`), not to the real dispatch. If
dispatch waited ~150 ms for the JSON, the result would land after the reserve
and the pre-answer gap would grow. Measured on `delegation` (fake engines, 3
measured runs): dispatching after the decision gave `max_gap_ms` 340–740;
speculating restores 0 ms and `task_dispatch_ms` 1820, identical to
`--tool-mode off`. The correction path costs nothing on the dominant tool and a
cancel+redispatch on the other two.

A superseded task's outcome is dropped by a generation counter in
`dispatch_task`, so a cancelled device check is never spoken as a failure
(`stale_results_dropped = 0` in the timer scenario).

`task_calls` semantics: `tool_calls` counts **model attempts** (incremented when
the request is queued, so a turn flushed by a barge-in still counts);
`tool_calls_valid` counts schema-valid calls. Field names unchanged.

Nothing new is added to `metrics_json`. The chosen tool, the raw reply and the
action travel in a stderr line and in a `timeline.csv` row:

```
[tool] mode=llm valid=1 tool=device_status args=[garage] action=kept reason='' raw='{"tool":...}'
run,turn,stage,kind,text,...            stage=tool, kind=<tool>, text=<raw reply>
```

## Modes

`--tool-mode` (also `VOICE_TOOL_MODE`): `llm` (default), `off` (pre-M2 baseline:
dispatch `device_status` at the final transcript, `tool_calls = 0`),
`fake_valid` / `fake_invalid` (deterministic reply, no model call, same parse
and dispatch path — this is how the harness drives both branches without
touching `engines_fake.cpp`).

## Selftest

`build-vk/bin/crispasr-voice --selftest` → **22/22** (10 policy checks unchanged
+ 12 new): fenced JSON, bare name, prose around the object, JSON-in-quotes,
non-numeric `minutes`, numeric-string coercion, unknown tool, truncated JSON,
wrong device type, out-of-range minutes, plain prose, and the three latency
profiles (6 s / ≤1 s / 0).

## Fake scenarios (deterministic, GPU-free)

| scenario | mode | result |
|---|---|---|
| `toolcall_valid` | fake_valid | ok, `max_gap` 0, wav_gap 20 ms, fillers 6, tool 1/1 valid |
| `toolcall_timer` | fake_valid | ok, gap 0, wav_gap 0, fillers 3, `action=redispatch`, 1/1 valid |
| `toolcall_invalid` | fake_invalid | ok, gap 0, wav_gap 0, fillers 6, 1 attempt / 0 valid, fallback |

`delegation`, `short`, `es_smoke`, `task_failure`, `barge_in` are unchanged and
still pass with `--tool-mode llm` (the fake LLM replies prose, so they exercise
the invalid→fallback branch: `tool_calls = 1`, `tool_calls_valid = 0`).

Barge-in probe (scenario with `barge_in_at_ms: 2400`, i.e. mid-decision): the
tool job sees the bumped epoch, posts nothing, nothing is dispatched;
`tool_calls = 1`, `tool_calls_valid = 0`, `barge_ins = 1`, turn finishes.

## Real validity

`delegation_real` (Qwen3-ASR-1.7B + Qwen3-TTS-1.7B + Qwen3-0.6B, Q8_0):

```
transcript : The status of the garage door.
ack        : The status of the garage door. I am checking.
answer     : The garage door is now in good condition.
tool       : {"tool":"device_status","args":{"device":"garage"}}  action=kept
```

- **9/9 model attempts schema-valid** (3 runs of one invocation + 6 runs of a
  second, warm-up included) — 100 %, every reply the exact compact object, no
  prose, no fences. Decision cost 71–97 ms for 13 tokens.
- A wider prompt sweep over 10 realistic transcripts (same system prompt, same
  0.4 temperature, same 48-token cap, driven through `crispasr-chat`, parser
  mirrored in Python and cross-checked against the C++ selftest cases):
  **10/10 schema-valid, 8/10 semantically right**.
  - `turn on the lights in the living room` → `set_timer{minutes:5}` — syntactically
    fine, wrong tool (the one real failure mode observed).
  - `pick up the dry cleaning tomorrow morning` → `device_status{garage}` — the
    instructed fallback for "no better tool", so arguably correct.
  - `could you tell me a joke` → `smalltalk{"greetings":"hello"}` — extra key
    ignored by the parser, which is why `smalltalk` tolerates `args` noise.
- Validity is above the 80 % bar, so no grammar is required for *this* model.

## Regressions and attribution

`delegation_real` currently fails its 300 ms gap gate: measured
`max_gap_ms` 523 ms (WAV 600 ms), fillers 4, `eos_to_first_played` 269 ms p50.
A/B on the same binary:

| mode | fillers | self gap | WAV gap |
|---|---|---|---|
| `llm` (this PR) | 4 | 503–523 ms | 580–600 ms |
| `off` (pre-M2 dispatch, no model call at all) | 4 | 503–543 ms | 580–620 ms |

The gap is the sequencer's current filler policy, not the tool call: with the
decision removed entirely the number is the same, and the rendered WAV puts the
silent run (8120 ms, 600 ms) directly before the answer's first audio
(~8270 ms), i.e. the pre-answer hole the filler reserve is supposed to cover.
`es_real` with the same filler count passes (13 ms), so this is the reserve
landing badly for this scenario rather than "four fillers are always too few".
Two related notes for whoever owns `sequencer.cpp`:

1. The filler reserve is anchored to the final transcript. That is why this PR
   speculates instead of dispatching after the decision; if speculation is ever
   dropped, `on_final` needs a tool-decision budget (`+120 ms`).
2. `queued_filler_ms` estimates 1.5 s per filler but the last filler must also
   *finish* before the result, so a 1.9 s real piece still leaves dead air.

## Next (grammar hook), if it is ever needed

The observed error class is **wrong tool, valid JSON**, so a grammar buys
nothing for validity today. It is still cheap insurance against a weaker or
chattier model:

1. `src/chat.cpp`: add a grammar string to `crispasr_chat_generate_params`
   (already threaded through `LlmEngine::generate`'s `grammar` argument, which
   both engines ignore) and hand it to the sampler.
2. One GBNF production: `root ::= "{\"tool\":\"" ("device_status"|"set_timer"|"smalltalk") ... }`
   with per-tool argument alternatives.
3. Then the lenient parser becomes a safety net instead of the primary path,
   and `tool_calls_valid` should read 100 % by construction.
