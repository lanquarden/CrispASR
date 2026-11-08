# 584 — M2: filler policy, bank vs LLM, watermarks, clause sizing

Date: 2026-11-08. Branch `feat/584-realtime-voice`. Engine set unchanged from
M1: Qwen3-ASR-1.7B Q8_0 + Qwen3-TTS-1.7B Q8_0 (jarvis voice pack) +
Qwen3-0.6B Q8_0 chat, one ggml Vulkan device (RX 7900 XT). Measurements taken
with the tree as it stands, which also contains the tool-call work
(`VOICE_TOOL_MODE`, default `llm`): a tool decision now runs between the final
transcript and task dispatch, so `task_result_ms` is ~0.15 s later than in the
M1 numbers. This milestone owns `examples/cli/voice/sequencer.{h,cpp}` (filler
policy only), `tests/voice/sweep.py`, the two new scenarios and this document;
everything else in the tree belongs to the tool-call work or to M1.

## Verdict

**The bank stays the default.** With the real engines the three modes are close
on dead air (WAV worst silence: bank 120 ms, LLM 100 ms, bank_then_llm 120 ms,
target <= 300 ms) but the bank is cheaper and safer: 1.37 s vs 2.16 s from task
result to the answer being heard, 2 vs 6 LLM generations per turn, no network of
new failure modes, and Qwen3-0.6B's "holding phrases" leak the answer
("The garage door is currently closed." while the result is still pending).

**The watermarks that matter are `low_water_ms` and nothing else.** On the real
engines `low_water 1600 / target 2500` buys zero gap improvement over
`1200 / 2000` and delays the answer by a further 0.6-1.5 s.

**`clause_max_words` is currently a dead knob** (see "Sweep"): the 27 bank cells
collapse to three identical triples within measurement noise.

## The policy as it stands

Every decision is in `sequencer.cpp`; the orchestrator only executes actions.

| rule | value | why |
|---|---|---|
| filler queue depth | 2 outstanding generations | one slot leaves the ring dry between LLM phrases (0.3 s TTFT + 0.35 s TTFA each); more than two piles filler ahead of the answer |
| bridge budget | committed speech (words x 325 ms) minus played | estimates how much speech is still queued ahead of the answer, including text the sequencer handed to TTS and cannot retract |
| comfortable regime (`buffer >= low_water_ms`) | keep bridging only while committed speech is short of `dispatch + task_ms + 500 ms` | filler plays ahead of the answer; over-queueing is what delayed the answer in M1 |
| urgent regime (`buffer < low_water_ms`) | always bridge | this is the watermark's documented meaning, and it is what covers a task result that arrives earlier than the estimate (adapter jitter is +/-800 ms) |
| near the result | pick the shortest unused bank phrase | a 0.6 s last piece lands on the answer more accurately than a 1.5 s one |
| no repeats | session-wide normalised phrase set, bank **and** LLM | PLAN: "fillers are never repeated within a session" |
| unusable LLM filler | fall back to the bank, retire the LLM after 3 empty/invalid results in a turn | an empty generation must not stall the bridge |
| filler expiry | `created + 4000 ms`, checked in `request_next_clause` | unchanged; see "Defects" #8 for why it never fires |
| speech sanitising | quotes/markdown stripped, emoji and symbol codepoints stripped | qwen3-tts renders both as long pauses or vocalisations |

## Defects found and fixed (task 1)

| # | defect | evidence | fix |
|---|---|---|---|
| 1 | **LLM filler request loop.** `filler_in_flight_` was declared but never set, so a new filler generation was requested on every low-water tick (throttled only by 300 ms). | code; with the queue depth fixed at 2 the LLM mode still costs `llm_calls` 6 vs 2 for the bank on `delegation_real`, all of it filler generations. Unbounded, every 300 ms tick issued one. | counted outstanding generations, maximum 2 (the clause-queue depth). |
| 2 | **LLM fillers repeated.** Only the bank had a no-repeat cursor; generated text went straight to TTS, so the same phrase could be spoken on every request. | code; the fake engine returns one fixed sentence per request (confirmed: pre-fix `--filler llm` on the fake scenario was never observed to vary it). Dedupe now rejects it, and the bank takes over — the real run's filler list shows exactly that path. | one normalised (lowercase, punctuation-free) phrase set for the whole session, bank and LLM; a repeat is rejected and counted (`fillers_repeat_rejected`). |
| 3 | **An empty/unusable LLM filler stalled the bridge.** `on_text_done(Filler)` did nothing when the generation produced no clause; nothing re-requested filler until the next playback tick, and in `--filler llm` there was no fallback at all. | code path; the fake engine cannot return an empty string, so this was verified through the identical branch that the repeat rejection takes (see "how this was verified"). | a generation that queues nothing increments `fillers_llm_empty`, falls back to the bank in **every** mode, and after three in a row the LLM filler is retired for that turn (bounded retry, no loop). |
| 4 | **Fillers were queued far past the wait**, so answer audio was stuck behind them. | fake `delegation`, old policy: 12 fillers for a 6 s task, answer played at 10.86 s. | the committed-speech budget above: 12 -> 6 fillers, answer at 8.45 s, no gap. |
| 5 | **The bridge stopped ~0.6 s before the result** because the deadline used `task_ms + jitter` (upper bound). The ring went dry exactly where the answer's TTFA sits. | real `delegation_real`, intermediate policy: 600 ms WAV silence, `self 523 ms`. | deadline is the centre (`dispatch + task_ms`), the urgent regime overrides it, and the last phrase is short. Real WAV gap 600 -> 120 ms. |
| 6 | **Bank accounting could re-emit a phrase** (`bank_cursor_` wrapped with a modulo; `spoken_bank_` was appended to but never consulted for selection). | code; latent for `bank_then_llm`, where the transition to the LLM depends on the bank being genuinely exhausted. | selection walks the ring and skips anything already used, returns "" when exhausted, so `bank_then_llm` switches exactly once and a bank phrase the LLM already used is skipped. |
| 7 | **The sanitiser did not strip emoji/symbols.** | real ES answer ended in `... ya se ha confirmado. "😊`; qwen3-tts turns that into a pause or a vocalisation. | UTF-8 codepoint filter that drops emoji/pictograph/symbol blocks, ZWJ and variation selectors, and keeps Latin accents. Verified: the same run now yields `El estado de la puerta del garaje ya se ha confirmado.` |
| 8 | **Filler expiry never fires.** The mechanism is intact (`expire_ms = created + 4000`, `text_stale()` checked in `request_next_clause`), but a filler is only held while the ring is at `high_water_ms` (3 s), so it is spoken long before 4 s. | 0 stale fillers in every fake and real run, including a deliberately slowed fake TTS (`VOICE_FAKE_TTS_RATE=0.55`). | left as-is; it is a backstop for a stalled TTS queue. Lowering it to ~1.5 s would make it real, but it cannot trigger today, so it is not a measurement. |

## Sweep

`tests/voice/sweep.py` runs the deterministic fake `delegation` scenario over
the requested grid — filler {bank, llm, bank_then_llm} x low_water
{800, 1200, 1600} x target_water {1500, 2000, 2500} x clause_max_words
{6, 20, 40} — with 2 measured runs + 1 warm-up per cell. 81 cells, one build
(`flock /tmp/voice-build.lock`, 22.3 s), 4 cells at a time, **715.9 s** total
(~31 s per cell). Full machine-readable data:
`tests/voice/out/sweep/summary.json`; per-cell harness output and logs under
`tests/voice/out/sweep/cells/`.

### Condensed result

| filler | low_water | target_water | WAV gap | audible p50 | fillers/run | repeats | answer after result |
|---|---|---|---|---|---|---|---|
| bank | 800 / 1200 | any | 0 ms | 1230-1240 ms | 6 | 0 | ~1.35 s |
| bank | 1600 | any | 0-20 ms | 1240 ms | 8 | 0 | ~3.76 s |
| llm | 800 / 1200 | any | 0-20 ms | 1220-1240 ms | 5 | 0 | ~1.05 s |
| llm | 1600 | 1500 / 2000 | 0-20 ms | 1240 ms | 5 | 0 | ~1.05 s |
| llm | 1600 | 2500 | 0-40 ms | 1230-1240 ms | 4-5 | 0 | 0.5-1.07 s |
| bank_then_llm | 800 / 1200 | any | 0-20 ms | 1220-1240 ms | 6 | 0 | ~1.35 s |
| bank_then_llm | 1600 | any | 0-20 ms | 1240 ms | 8 | 0 | ~3.76 s |

What the grid says:

1. **`low_water_ms` is the only watermark with an effect.** 1600 keeps the
   "always bridge below low water" regime active for longer: 8 fillers instead
   of 6 and the answer arrives 2.4 s later. No gap improvement anywhere.
   `target_water_ms` moves nothing measurable on this scenario either: 26 of 27
   (filler, low, clause) triples are identical across 1500/2000/2500 within the
   same noise band.
2. **`clause_max_words` is inert.** 25 of the 27 (filler, low, target) triples
   agree across 6/20/40 within noise (WAV gap within 20 ms, answer delay within
   25 ms); the one 566 ms outlier is an `llm/1600/2500` cell where the fallback
   path landed one filler differently, not a clause effect. Cause: since M1 the
   sequencer splits at sentence
   boundaries, `Answer` uses `max(clause_max_words, 40)`, and `Filler` uses
   `filler_max_words` (18). Reverting that floor would re-open the M1
   piece-boundary gaps, so the knob is reported dead rather than wired.
3. **The `llm` rows measure the fallback, not LLM filler.** `engines_fake.cpp`
   picks its filler reply by looking for the word "filler" in the *system*
   prompt; the loop's filler system prompt says "holding phrase", so the fake
   returns the answer sentence instead. That sentence is accepted once, then
   rejected as a repeat, and the bank fallback takes over — which is exactly why
   the llm rows show one fewer filler than bank. The real comparison is in the
   next section.
4. **No cell repeated a phrase** (81/81), which is the property M2 had to
   establish; it is enforced by the policy, not by the fixture.

### Best cells

Ranked by repeats, then WAV gap, then audible p50 (all 0 repeats, so the order
is gap then latency):

| # | cell | WAV gap | audible p50 | fillers | answer after result |
|---|---|---|---|---|---|
| 1 | `llm / 800 / 2500 / 6` | 0 ms | 1220 ms | 5 | 1.04 s |
| 2 | `llm / 1200 / 1500 / 6` | 0 ms | 1220 ms | 5 | 1.04 s |
| 3 | `llm / 1200 / 1500 / 40` | 0 ms | 1220 ms | 5 | 1.05 s |

These are fake-engine cells and win only because the fallback queues one filler
fewer. They are not a recommendation: the same configuration with the real
engines is measured below, where `llm` costs 1.6x the answer delay.

## Real engines

`VOICE_REAL_ENGINES=1`, 2 measured runs + 1 warm-up, `check_metrics.py` gate.
Fields: WAV worst silence / WAV end-of-speech -> first audible (p50) / fillers
per run / task result -> answer heard.

| scenario | filler | WAV gap | audible p50 (p95) | fillers | result -> answer | gate |
|---|---|---|---|---|---|---|
| `delegation_real` | bank | 120 ms | 790 (810) ms | 6-7 | 1.37 s | ok |
| `delegation_real` | llm | 100 ms | 810 (810) ms | 4 | 2.16 s | ok |
| `delegation_real` | bank_then_llm | 120 ms | 830 (930) ms | 6-7 | 1.21 s | ok |
| `es_real` (new) | bank | 160 ms | 1040 (1040) ms | 5 | 1.19 s | ok |
| `barge_real` (new) | bank | 2380 ms* | 810 (810) ms | 3 | - | ok |

\* `barge_real`'s trailing silence is the interrupted turn: after the barge-in
nothing else is spoken and the WAV keeps recording, so the recording's last
2.4 s are silence by construction. It is not a bridge gap; the scenario asserts
barge-in detection and stop time instead.

Self-reported end-of-speech -> first played stays at 252-377 ms (p95) on the
four non-barge runs; the difference between that and the WAV-derived audible number is the
end-of-turn wait plus the input fixture's own silence, as established in
RESULTS.md.

### What was said (real runs)

- `delegation_real` transcript: *"The status of the garage door."*; ack:
  *"The status of the garage door. I am checking."*; answer:
  *"The garage door is now in good condition."* (run 2) /
  *"The garage door is now operational."* (run 3).
- `es_real` transcript: *"¿Es el estado de la puerta del garaje?"* (the fixture
  says *"¿Cuál es el estado de la puerta del garaje?"*; ASR drops the "Cuál").
  ack: *"El estado de la puerta del garaje. ¡Estoy checking!"* — the LLM code
  switches mid-sentence. Answer: *"El estado de la puerta del garaje ya se ha
  confirmado."* The tool call was valid (device_status, garage) in both runs.
- `barge_real`: same transcript and ack; the barge-in at 4000 ms fires inside
  the ack, playback is flushed, `barge_in_stop_ms` 5-6 ms, the task is
  cancelled, the stale result never reaches the floor, and no answer is spoken.
  Two of the three queued fillers were flushed before they reached TTS.

### Filler-mode comparison on the real engines

| | bank | llm | bank_then_llm |
|---|---|---|---|
| filler phrases (run 2) | One moment / Let me check that / Checking now / Bear with me / One second / Looking into that | The garage door is currently closed / One moment / The garage door is open / Let me check that | same as bank (bank never exhausted) |
| fillers played | 6-7 | 4 | 6-7 |
| TTS pieces | 9-10 | 8 | 9-10 |
| LLM generations | 2 | 6 | 2 |
| result -> answer | 1.37 s | 2.16 s | 1.21 s |
| WAV worst silence | 120 ms | 100 ms | 120 ms |

The LLM filler costs one generation plus one TTS round trip per phrase, so the
bridge is thinner (4 fillers, depth 2 is not enough to eliminate the extra
wait) and the answer queues behind whatever is in flight. It also *claims
results*: "The garage door is currently closed." and "The garage door is open."
are both result statements, and they contradict each other inside one turn. The
prompt says "do not claim progress or a result"; a 0.6B chat model does not obey
it. The bank cannot make that mistake.

### Watermark check on the real engines

To confirm the sweep's `low_water` finding on real audio (2+1 runs, bank filler):

| cell | WAV gap | audible p50 (p95) | fillers | result -> answer |
|---|---|---|---|---|
| low 800 / target 2000 | 120 ms | 810 (810) ms | 6 | 1.37 s |
| low 1600 / target 2500 | 120 ms | 810 (950) ms | 7-8 | 1.94 s (p95 2.89 s) |

Raising the watermarks buys nothing and pushes the answer 0.6-1.5 s further out.
The default `low_water 1200 / target 2000` sits on the same knee as 800.

## How this was verified

- `--selftest`: 22 policy assertions pass, including "bank fillers do not
  repeat" and the tool-call cases.
- The automated suite (`short`, `delegation`, `barge_in`, `task_failure`,
  `es_smoke`, plus the tool-call scenarios) passes unchanged; no threshold was
  touched.
- Fillers stop when the answer's PCM exists: over 88 `timeline.csv` files (all
  real runs and all 81 sweep cells), **no** `stage=tts, kind=filler` row starts
  at or after the first answer piece's `t_first_pcm_ms` — 0 of 88 files.
- No repeat within a session, both engines:
  `python3 - <<'EOF'` over `timeline.csv`, grouping filler rows by `run` and
  counting duplicate texts (0 in all 81 sweep cells, 0 in every real run).
- Empty/unusable LLM filler: exercised through the shared branch. The real
  0.6B produced two result claims that dedupe accepted (they differ), and the
  fake engine's fixed sentence is rejected as a repeat on every request after
  the first — the same `llm_filler_speakable_ == 0` branch an empty generation
  takes. In both cases the bank filler is queued instead (`One moment.`,
  `Let me check that.`) and the bridge continues with no stall.
- Filler expiry: 0 `fillers_dropped_stale` in every run, including a purposely
  slowed fake TTS, which is the finding in Defects #8.

## Remaining defects (not fixed here, with the reason)

1. **Queued filler cannot be retracted when the answer's PCM exists.** The
   sequencer stops *asking* for filler the moment the answer has PCM, but a
   filler already handed to the TTS pump still plays before the answer — that
   is the 1.2-1.4 s "answer after result" number. Fixing it needs a
   `DropQueuedFillers` action executed by the orchestrator's TTS queue
   (`voice_loop.cpp`, not owned here). It would cut the answer delay to the
   answer's own TTFA.
2. **The real-engine TTS language is pinned.** `engines_crispasr.cpp` sets
   `target_lang = "en"` for every piece, so the Spanish run is spoken through
   the English ICL path. The text is Spanish and intelligible, but this is a
   knob, not a decision.
3. **`barge_in.json` (fake) now lands before the ack has any PCM.** With the
   tool-decision stage in the tree the scenario's 3000 ms marker fires while the
   ack is still being synthesised: `ack_first_pcm_ms` and `ack_first_played_ms`
   stay 0 and the rendered WAV is silent, so the turn is flushed before it says
   anything. Its own assertions (barge detected, stop <= 250 ms) still pass and
   the checker notes the missing audio; its thresholds were not changed.
   `barge_real` uses 4000 ms for the same reason: the fixture's speech ends at
   2.51 s and the real ack is audible from ~3.2 s, so a 3000 ms marker would fire
   before the assistant owns the floor.
4. **`clause_max_words` is inert** (Sweep #2). Either delete the knob or wire it
   together with a reconsidered clause-splitting policy; wiring it alone
   re-introduces the M1 piece-boundary gaps.

## Reproduce

```sh
# fake sweep (builds once, ~12 min with 4 workers)
python3 tests/voice/sweep.py --workers 4

# real runs (GPU lock, one at a time; never concurrent with a build)
M=~/.local/share/containers/storage/volumes/beast-crispasr_crispasr-models/_data
VOICE_REAL_ENGINES=1 build-vk/bin/crispasr-voice \
  --harness tests/voice/scenarios/es_real.json \
  --asr-model $M/qwen3-asr-1.7b-q8_0.gguf \
  --tts-model $M/qwen3-tts-12hz-1.7b-base-q8_0.gguf \
  --tts-tokenizer $M/qwen3-tts-tokenizer-12hz.gguf \
  --voice-dir ~/repos/home-ops/hosts/beast/crispasr/voice \
  --chat-model ~/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf
python3 tests/voice/check_metrics.py --scenario es_real
# same for barge_real.json and delegation_real.json (--filler bank|llm|bank_then_llm)
```

`es_real`'s fixture was generated with the CLI TTS (24 kHz, `-tl es`, the
`question_es_real` text is *"¿Cuál es el estado de la puerta del garaje?"*) and
0.8 s of silence appended, matching `question_real.wav`:

```sh
build-vk/bin/crispasr --backend qwen3-tts-1.7b-base \
  -m $M/qwen3-tts-12hz-1.7b-base-q8_0.gguf --codec-model $M/qwen3-tts-tokenizer-12hz.gguf \
  --voice-dir ~/repos/home-ops/hosts/beast/crispasr/voice --voice jarvis -tl es \
  --tts "¿Cuál es el estado de la puerta del garaje?" --tts-output /tmp/es.wav
# then append 0.8 s of zero samples and save as tests/voice/fixtures/question_es_real.wav
```

## Appendix — all 81 sweep cells

| filler | low_water | target_water | clause_max_words | WAV gap (ms) | audible p50 (ms) | audible p95 (ms) | fillers/run | repeats | answer after result (ms) | runtime (s) |
|---|---|---|---|---|---|---|---|---|---|---|
| bank | 800 | 1500 | 6 | 0 | 1240 | 1240 | 6-6 | 0 | 1368 | 31.6 |
| bank | 800 | 1500 | 20 | 0 | 1230 | 1240 | 6-6 | 0 | 1348 | 31.6 |
| bank | 800 | 1500 | 40 | 0 | 1230 | 1240 | 6-6 | 0 | 1350 | 31.6 |
| bank | 800 | 2000 | 6 | 0 | 1230 | 1240 | 6-6 | 0 | 1350 | 31.6 |
| bank | 800 | 2000 | 20 | 0 | 1230 | 1240 | 6-6 | 0 | 1348 | 31.6 |
| bank | 800 | 2000 | 40 | 0 | 1240 | 1240 | 6-6 | 0 | 1368 | 31.6 |
| bank | 800 | 2500 | 6 | 0 | 1230 | 1240 | 6-6 | 0 | 1348 | 31.6 |
| bank | 800 | 2500 | 20 | 0 | 1230 | 1240 | 6-6 | 0 | 1349 | 31.6 |
| bank | 800 | 2500 | 40 | 0 | 1230 | 1240 | 6-6 | 0 | 1349 | 31.6 |
| bank | 1200 | 1500 | 6 | 0 | 1240 | 1240 | 6-6 | 0 | 1369 | 31.6 |
| bank | 1200 | 1500 | 20 | 0 | 1220 | 1220 | 6-6 | 0 | 1348 | 31.6 |
| bank | 1200 | 1500 | 40 | 0 | 1230 | 1240 | 6-6 | 0 | 1349 | 31.6 |
| bank | 1200 | 2000 | 6 | 0 | 1240 | 1240 | 6-6 | 0 | 1357 | 31.6 |
| bank | 1200 | 2000 | 20 | 0 | 1240 | 1240 | 6-6 | 0 | 1356 | 31.6 |
| bank | 1200 | 2000 | 40 | 0 | 1240 | 1240 | 6-6 | 0 | 1359 | 31.6 |
| bank | 1200 | 2500 | 6 | 0 | 1240 | 1240 | 6-6 | 0 | 1359 | 31.6 |
| bank | 1200 | 2500 | 20 | 0 | 1240 | 1240 | 6-6 | 0 | 1355 | 31.6 |
| bank | 1200 | 2500 | 40 | 0 | 1240 | 1240 | 6-6 | 0 | 1356 | 31.6 |
| bank | 1600 | 1500 | 6 | 0 | 1240 | 1240 | 8-8 | 0 | 3759 | 38.8 |
| bank | 1600 | 1500 | 20 | 0 | 1240 | 1240 | 8-8 | 0 | 3760 | 38.8 |
| bank | 1600 | 1500 | 40 | 0 | 1240 | 1240 | 8-8 | 0 | 3755 | 38.8 |
| bank | 1600 | 2000 | 6 | 0 | 1240 | 1240 | 8-8 | 0 | 3755 | 38.8 |
| bank | 1600 | 2000 | 20 | 20 | 1240 | 1240 | 8-8 | 0 | 3764 | 38.8 |
| bank | 1600 | 2000 | 40 | 0 | 1240 | 1240 | 8-8 | 0 | 3759 | 38.8 |
| bank | 1600 | 2500 | 6 | 0 | 1240 | 1240 | 8-8 | 0 | 3754 | 38.8 |
| bank | 1600 | 2500 | 20 | 0 | 1240 | 1240 | 8-8 | 0 | 3753 | 38.8 |
| bank | 1600 | 2500 | 40 | 0 | 1240 | 1240 | 8-8 | 0 | 3754 | 38.8 |
| llm | 800 | 1500 | 6 | 0 | 1240 | 1240 | 5-5 | 0 | 1058 | 30.7 |
| llm | 800 | 1500 | 20 | 0 | 1240 | 1240 | 5-5 | 0 | 1064 | 30.7 |
| llm | 800 | 1500 | 40 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 800 | 2000 | 6 | 0 | 1240 | 1240 | 5-5 | 0 | 1064 | 30.7 |
| llm | 800 | 2000 | 20 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 800 | 2000 | 40 | 0 | 1230 | 1240 | 5-5 | 0 | 1049 | 30.7 |
| llm | 800 | 2500 | 6 | 0 | 1220 | 1220 | 5-5 | 0 | 1044 | 30.7 |
| llm | 800 | 2500 | 20 | 0 | 1230 | 1240 | 5-5 | 0 | 1045 | 30.7 |
| llm | 800 | 2500 | 40 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1200 | 1500 | 6 | 0 | 1220 | 1220 | 5-5 | 0 | 1044 | 30.7 |
| llm | 1200 | 1500 | 20 | 0 | 1230 | 1240 | 5-5 | 0 | 1045 | 30.7 |
| llm | 1200 | 1500 | 40 | 0 | 1220 | 1220 | 5-5 | 0 | 1045 | 30.7 |
| llm | 1200 | 2000 | 6 | 20 | 1230 | 1240 | 5-5 | 0 | 1050 | 30.7 |
| llm | 1200 | 2000 | 20 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1200 | 2000 | 40 | 0 | 1240 | 1240 | 5-5 | 0 | 1064 | 30.7 |
| llm | 1200 | 2500 | 6 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1200 | 2500 | 20 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1200 | 2500 | 40 | 0 | 1240 | 1240 | 5-5 | 0 | 1064 | 30.7 |
| llm | 1600 | 1500 | 6 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1600 | 1500 | 20 | 0 | 1230 | 1240 | 5-5 | 0 | 1045 | 30.7 |
| llm | 1600 | 1500 | 40 | 0 | 1230 | 1240 | 5-5 | 0 | 1046 | 30.7 |
| llm | 1600 | 2000 | 6 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1600 | 2000 | 20 | 0 | 1240 | 1240 | 5-5 | 0 | 1063 | 30.7 |
| llm | 1600 | 2000 | 40 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1600 | 2500 | 6 | 0 | 1240 | 1240 | 5-5 | 0 | 1065 | 30.7 |
| llm | 1600 | 2500 | 20 | 0 | 1230 | 1240 | 5-5 | 0 | 1046 | 30.7 |
| llm | 1600 | 2500 | 40 | 40 | 1240 | 1240 | 4-5 | 0 | 499 | 30.2 |
| bank_then_llm | 800 | 1500 | 6 | 20 | 1240 | 1240 | 6-6 | 0 | 1353 | 31.6 |
| bank_then_llm | 800 | 1500 | 20 | 0 | 1240 | 1240 | 6-6 | 0 | 1370 | 31.6 |
| bank_then_llm | 800 | 1500 | 40 | 0 | 1230 | 1240 | 6-6 | 0 | 1349 | 31.6 |
| bank_then_llm | 800 | 2000 | 6 | 0 | 1240 | 1240 | 6-6 | 0 | 1368 | 31.6 |
| bank_then_llm | 800 | 2000 | 20 | 0 | 1220 | 1220 | 6-6 | 0 | 1349 | 31.6 |
| bank_then_llm | 800 | 2000 | 40 | 20 | 1220 | 1220 | 6-6 | 0 | 1333 | 31.6 |
| bank_then_llm | 800 | 2500 | 6 | 20 | 1240 | 1240 | 6-6 | 0 | 1368 | 31.6 |
| bank_then_llm | 800 | 2500 | 20 | 0 | 1240 | 1240 | 6-6 | 0 | 1369 | 31.6 |
| bank_then_llm | 800 | 2500 | 40 | 0 | 1230 | 1240 | 6-6 | 0 | 1350 | 31.6 |
| bank_then_llm | 1200 | 1500 | 6 | 20 | 1240 | 1240 | 6-6 | 0 | 1370 | 31.6 |
| bank_then_llm | 1200 | 1500 | 20 | 0 | 1230 | 1240 | 6-6 | 0 | 1348 | 31.6 |
| bank_then_llm | 1200 | 1500 | 40 | 0 | 1230 | 1240 | 6-6 | 0 | 1348 | 31.6 |
| bank_then_llm | 1200 | 2000 | 6 | 0 | 1230 | 1240 | 6-6 | 0 | 1349 | 31.6 |
| bank_then_llm | 1200 | 2000 | 20 | 20 | 1230 | 1240 | 6-6 | 0 | 1332 | 31.6 |
| bank_then_llm | 1200 | 2000 | 40 | 0 | 1230 | 1240 | 6-6 | 0 | 1349 | 31.6 |
| bank_then_llm | 1200 | 2500 | 6 | 0 | 1240 | 1240 | 6-6 | 0 | 1368 | 31.6 |
| bank_then_llm | 1200 | 2500 | 20 | 0 | 1220 | 1220 | 6-6 | 0 | 1348 | 31.6 |
| bank_then_llm | 1200 | 2500 | 40 | 0 | 1240 | 1240 | 6-6 | 0 | 1369 | 31.6 |
| bank_then_llm | 1600 | 1500 | 6 | 20 | 1240 | 1240 | 8-8 | 0 | 3754 | 38.8 |
| bank_then_llm | 1600 | 1500 | 20 | 20 | 1240 | 1240 | 8-8 | 0 | 3755 | 38.8 |
| bank_then_llm | 1600 | 1500 | 40 | 20 | 1240 | 1240 | 8-8 | 0 | 3754 | 38.8 |
| bank_then_llm | 1600 | 2000 | 6 | 20 | 1240 | 1240 | 8-8 | 0 | 3755 | 38.8 |
| bank_then_llm | 1600 | 2000 | 20 | 20 | 1240 | 1240 | 8-8 | 0 | 3754 | 38.8 |
| bank_then_llm | 1600 | 2000 | 40 | 20 | 1240 | 1240 | 8-8 | 0 | 3756 | 38.8 |
| bank_then_llm | 1600 | 2500 | 6 | 20 | 1240 | 1240 | 8-8 | 0 | 3755 | 38.8 |
| bank_then_llm | 1600 | 2500 | 20 | 20 | 1240 | 1240 | 8-8 | 0 | 3755 | 38.8 |
| bank_then_llm | 1600 | 2500 | 40 | 0 | 1240 | 1240 | 8-8 | 0 | 3757 | 38.8 |
