# 584 — chat model bench: is 1.7B required for Spanish?

Date: 2026-11-09. Branch state: `feat/584-realtime-voice`, no code changed.
Question: the loop's Qwen3-0.6B-Q8_0 ack/answer text is poor in Spanish (ack asks
the user a question, or answers the question instead of acknowledging it; EN
looked fine). Decide: keep 0.6B, switch to 1.7B, or split by language — and check
that tool-call JSON survives the bigger model.

## Verdict

**Switch to Qwen3-1.7B-Q8_0 for both languages; do not split.** JSON stays 100%
parseable, ES tool arguments go 43% → 90%, and the ES ack stops code-switching
and stops interrogating the user. Cost is +1.2 GB of weights and +10 ms of TTFT,
against a 6 s device check and a 141-184 ms TTS TTFA.

**The 1.7B is not what fixes the reported ack defect.** Both models restate the
request as a *result* ("La puerta del garaje está cerrada", "The garage door is
closed") because the ack **user** prompt asks them to "restate what you
understood". Deleting that clause drops result claims to 0/10 EN on both models
(§5). Fix the ack prompt in the same PR; model size alone leaves the defect in
place.

## 1. What was measured

Both prompts are the loop's own, not lookalikes:

| | source |
|---|---|
| router system + user prompt | `tool_call_messages()` from `examples/cli/voice/tool_call.cpp` |
| router JSON acceptance | `parse_tool_call()` — the loop's real lenient parser, compiled unmodified into the bench |
| ack system prompt | `voice_loop.cpp` `TextKind::Ack` branch, incl. `lang_rule` |
| ack user prompt | `sequencer.cpp` `emit_request(TextKind::Ack, …)` |
| tokens / temperature | router 48 tok, ack 64 tok, `temperature = 0.4` — as `engines_crispasr.cpp` hardcodes |

Harness `/tmp/bench/bench` (not in the repo, not in the build): links the
already-built `build-vk/src/libcrispasr.so`, opens the model once, issues
`crispasr_chat_reset()` + one system+user generation per request, and records
TTFT (first streamed chunk), total, token count, raw text and the parser verdict.
Every run under `flock /tmp/voice-gpu.lock`. RX 7900 XT (Vulkan), loop idle.

Two conditions, 10 EN + 10 ES utterances each (`/tmp/bench/cases.tsv`), 3 repeats:

* **seed-varied** (`0x1001..0x1003`) — 30 samples per language per model. Used for
  the table below.
* **seed 0** — the loop's actual behaviour: `engines_crispasr.cpp` keeps the ABI
  default `seed = 0`, so a given transcript always produces the same text. Same
  verdicts, ±1-2 cases; numbers in §4.

Model loaded from `/home/lanquarden/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf`
(`Qwen/Qwen3-1.7B-GGUF`, 1 834 426 016 B, GGUF v3, 29/29 blocks offloaded
in 784 ms vs 399 ms for 0.6B). Kept on disk: 148 GB free, and both models are
worth keeping for the language-split fallback.

## 2. Results

Percentages over 30 route samples / 30 ack samples per cell.

| model | lang | JSON valid | tool acc | arg acc | ack lang | code-switch | claims result | asks question | ack >8 words | route p50 TTFT / total | ack p50 TTFT |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.6B | en | 100% | 90% | 80% | 100% | 0% | **20%** | 0% | 30% | 16 / 52 ms | 19 ms |
| 0.6B | es | **93%** | 93% | **43%** | 100% | **27%** | 13% | **17%** | 7% | 16 / 52 ms | 20 ms |
| 1.7B | en | 100% | 90% | 80% | 100% | 0% | 10% | 0% | **93%** | 26 / 81 ms | 30 ms |
| 1.7B | es | **100%** | 90% | **90%** | 100% | **3%** | 10% | **0%** | 40% | 26 / 81 ms | 30 ms |

`tool acc` compares only the 3-way tool; `arg acc` is the fully correct decision
(right tool *and* right device/minutes). The gap between those two columns for
0.6B ES is the Spanish-numeral failure below.

"ack lang" is gross language detection; "code-switch" is the sharper one — an
English verb inside an ES turn (`estoy checking`), or Spanish words inside an EN
turn. TTFT over a full re-prefill per request; a warm loop turn reuses the
history cache, so 1.7B's real in-loop TTFT is lower than 26 ms.

### Per-case tool decision (majority of 3 seeds)

`<<` = differs from the intended tool.

| case | expected | 0.6B | 1.7B |
|---|---|---|---|
| en01 is the garage door closed | device_status:garage | garage | garage |
| en02 turn on the living room lights | device_status:lights | **set_timer:5** | **set_timer:5** |
| en03 what is the thermostat set to | device_status:thermostat | thermostat | thermostat |
| en04 how warm is the house right now | device_status:thermostat | **garage** | **garage** |
| en05..en07 timers 5/10/20 | set_timer 5/10/20 | ok | ok |
| en08..en10 joke / greeting / thanks | smalltalk | ok | ok |
| es01 ¿está cerrada la puerta del garaje? | device_status:garage | garage | garage |
| es02 enciende las luces del salón | device_status:lights | **garage** | **set_timer:5** |
| es03 ¿a cuánto está el termostato? | device_status:thermostat | **garage** | thermostat |
| es04 ¿qué temperatura hace en casa? | device_status:thermostat | **garage** | thermostat |
| es05 pon un temporizador de cinco minutos | set_timer:5 | ok | ok |
| es06 ponme una alarma para diez minutos | set_timer:10 | **set_timer:5** | ok |
| es07 avísame en veinte minutos | set_timer:20 | **set_timer:5** | ok |
| es08 cuéntame un chiste | smalltalk | ok | ok |
| es09 hola, buenos días | smalltalk | **INVALID** | ok |
| es10 gracias, eso es todo | smalltalk | ok | ok |

EN is identical between the two models: the same 8/10 fully correct decisions
(9/10 on the tool alone — en04 picks the right tool, wrong device), the same two
failures, the same 80% argument accuracy. ES goes from **4 correct decisions out
of 10 (43% of samples) to 9 out of 10 (90%)**, and every 0.6B ES failure is
structural, not luck:

* **Spanish number words are not mapped.** `diez` and `veinte` both become
  `{"minutes":5}` — the example value in the prompt. 0.6B scores 43% on arguments
  because all three ES timer cases collapse to 5. 1.7B gets 10 and 20 right.
* **JSON breaks under Spanish text.** es09 emits
  `{"tool":"smalltalk","args":{"greetings":"hola","buenos_dias"}}` — a key with no
  value. That is the only invalid JSON anywhere in 240 generations; 1.7B never
  produced any.
* **The ack code-switches and asks questions.** 27% of 0.6B ES acks contain an
  English verb; one run emitted a Vietnamese token (`Sí, lo estoy kiểm tra.`);
  17% turn the ack into a question to the user — exactly the reported
  `¿Está la puerta del garaje en funcionamiento?`. 1.7B: 3% code-switch, 0%
  questions, and it keeps the third person out (`Entiendo que te preguntas por el
  estado del termostato. Estoy revisando la información.`).

## 3. Two failures both models share

**`turn on the X`** (en02/es02) routes to `set_timer`. Both models, both
languages. The tool set has no actuation tool: of the three, `device_status` is a
*read*. The models read "turn on" as an instruction with a duration and reach for
the only action tool. The bench case is the ambiguous one; fix it by adding an
actuation tool or by wording the profile ("device_status reports state only"), not
by buying a bigger model.

**`how warm is the house`** (en04) routes to `garage`, because `tool_call_system_prompt`
says "If the request is unclear, use device_status for the garage". 1.7B gets the
ES twin right and the EN twin wrong, so this is the fallback rule biting, not a
capability wall.

## 4. Seed 0 — what the loop actually says today

Same table, `seed = 0` (loop-faithful, 3 identical repeats so effectively 10
samples/cell): 0.6B en 100/90/80, es **100/100/50**, code-switch 40%, questions
20%; 1.7B en 100/90/80, es 100/90/90, code-switch 10%, questions 0%. The 0.6B ES
draw is luckier on *tool* choice here (100%, because an argument error still
counts as the right tool) and decides correctly on 15/30 samples instead of
13/30, while its acks are worse (40% code-switch). Hence the seed-varied table
above is the one to quote.

Because the seed is fixed, **a model change is the only way to change the ES
wording** short of editing prompts — the loop cannot sample its way out.

## 5. Probe: the ack defect is the prompt, not the weights

Same ack call, user prompt with `restates what you understood` removed
(`The user said: "…". Reply with one short acknowledgement that says you are
checking. No result, no speculation.`), 1 run:

| model | lang | claims result | asks question | code-switch | mean words |
|---|---|---|---|---|---|
| 0.6B | en | **0/10** (was 20%) | 0 | 0 | 3.0 |
| 0.6B | es | **0/10** (was 13%) | 1 | 2 | 3.0 |
| 1.7B | en | **0/10** (was 10%) | 0 | 0 | 4.6 |
| 1.7B | es | **1/10** (was 10%) | 0 | 3 | 4.2 |

`La puerta del garaje está cerrada. Estoy verificando.` disappears for 0.6B and
survives once for 1.7B. Cost: 0.6B degenerates to `I am checking.` / `You are
checking.` (second person, no content) and still code-switches in ES; 1.7B stays
informative (`I'm checking the thermostat setting.`) and only leaks `checking`
into ES. So the prompt fix is safe **with** 1.7B and marginal with 0.6B.

## 6. The 1.7B ack-length cost, and why it needs a sequencer change

1.7B ignores "at most eight words" more often than not: 93% of EN acks and 40% of
ES acks exceed it (mean 11.8 / 8.6 words vs 7.3 / 6.1 for 0.6B). `ack_max_words`
is 14, so the clause splitter in `sequencer.cpp` will *not* trim them — the whole
two-sentence ack ("I understand you're asking about the thermostat setting. I'm
checking the current temperature setting.") is spoken before the answer. At ~3
words/s that is roughly 1.5 s of extra ack audio in the way of the answer, which
is the exact thing `RESULTS.md`'s ack budget exists to prevent. Not measured
here (no voice harness run, per instruction); measure it with
`check_metrics.py` before shipping.

Cheapest fix if 1.7B is adopted: cut `ack_max_words` to 8-10, which the existing
trimmer already enforces, or add "one sentence, no restatement" to the ack system
prompt — the §5 probe shows the short-prompt 1.7B acks land at 4.6 words anyway.

## 7. Recommendation

1. **Adopt `Qwen3-1.7B-Q8_0` for both languages.** One model, one load path, one
   ctx budget; +1.2 GB and +10 ms TTFT in a 17.3 GB co-residency budget that
   already runs two 1.7B models. It fixes ES tool arguments (43%→90%), ES JSON
   (93%→100%), ES acks (27%→3% code-switch, 17%→0% questions) and costs EN
   nothing on tool choice.
2. **Change the ack user prompt in the same PR** — drop "restates what you
   understood". This, not the weights, is what stops the ack claiming a result.
3. **Trim the ack to <=8-10 words in the sequencer** so 1.7B's longer sentences
   do not delay the answer; verify with `check_metrics.py`.
4. **Language-split is the fallback, not the plan.** If VRAM or ack audio turns
   out to bind, 0.6B for EN + 1.7B for ES is coherent (EN verdicts are identical
   between the two), but it doubles the load path to save 1.2 GB.
5. **Do not touch the model to fix `turn on the lights`.** That is a missing tool.

Raw outputs, cases and harness: `/tmp/bench/` (`raw-*.tsv` seed 0, `vary-*.tsv`
seeded, `norestate-*.tsv` probe, `scored-*.txt`, `bench.cpp`, `score.py`).
Nothing in the repo was modified except this file.

## 8. Verification

The whole matrix was re-run after the branch's prompt files were edited again
(15:02: `tool_call.cpp`, `voice_loop.cpp`, `sequencer.cpp` all changed under a
parallel session) with `tool_call.cpp` recompiled from the new tree. Every
prompt above is unchanged by those edits, and the re-run reproduced the table
cell for cell (`recheck-*.tsv`, ±1 ms on a p50). The 1.7B stays on disk at
`/home/lanquarden/models/Qwen3-1.7B-GGUF/Qwen3-1.7B-Q8_0.gguf` — it is the
recommended model, not scratch data.
