// voice_loop.cpp — orchestrator for the realtime voice loop.
//
// Threads: capture/control (this function), inference arbiter, playback, and
// one thread per simulated task. All inference goes through the arbiter, so
// the three models never run concurrently on the shared Vulkan device.
#include "voice_loop.h"

#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <mutex>
#include <cstdio>
#include <thread>
#include <vector>

#include "arbiter.h"
#include "audio_io.h"
#include "sequencer.h"
#include "stt_driver.h"
#include "task_adapter.h"
#include "tool_call.h"
#include "turn_detector.h"

namespace crispasr {
namespace voice {

Engines::~Engines() {
    if (owns) {
        delete asr;
        delete tts;
        delete llm;
    }
}

namespace {

struct EventQueue {
    std::mutex mu;
    std::deque<Event> q;

    void post(Event e) {
        std::lock_guard<std::mutex> lk(mu);
        q.push_back(std::move(e));
    }
    void drain(std::vector<Event>* out) {
        std::lock_guard<std::mutex> lk(mu);
        while (!q.empty()) {
            out->push_back(std::move(q.front()));
            q.pop_front();
        }
    }
};

// FIFO of enqueued audio, so first-played timestamps can be attributed.
struct AudioLedger {
    struct Piece {
        TextKind kind;
        int64_t samples;
    };
    std::mutex mu;
    std::deque<Piece> pieces;
    int64_t last_played = 0;

    void clear() {
        std::lock_guard<std::mutex> lk(mu);
        pieces.clear();
        last_played = 0;
    }
    void add(TextKind k, int64_t n) {
        std::lock_guard<std::mutex> lk(mu);
        if (std::getenv("VOICE_DEBUG"))
            std::fprintf(stderr, "[dbg] ledger add kind=%d n=%lld\n", (int)k,
                         (long long)n);
        pieces.push_back({k, n});
    }
    void consume(int64_t real_played, std::map<TextKind, int64_t>* first_play,
                 int64_t t_ms) {
        std::lock_guard<std::mutex> lk(mu);
        int64_t delta = real_played - last_played;
        last_played = real_played;
        while (delta > 0 && !pieces.empty()) {
            Piece& p = pieces.front();
            const int64_t take = std::min(delta, p.samples);
            if (take > 0 && first_play->find(p.kind) == first_play->end()) {
                (*first_play)[p.kind] = t_ms;
                if (std::getenv("VOICE_DEBUG"))
                    std::fprintf(stderr, "[dbg] ledger first kind=%d at=%lld\n",
                                 (int)p.kind, (long long)t_ms);
            }
            p.samples -= take;
            delta -= take;
            if (p.samples == 0) pieces.pop_front();
        }
    }
};

std::map<std::string, std::vector<float>> prerender_fillers(TtsEngine* tts,
                                                            const Config& cfg) {
    std::map<std::string, std::vector<float>> out;
    if (!cfg.prerender_filler || tts == nullptr) return out;
    static const char* const kPhrases[] = {
        "One moment.",       "Let me check that.", "Checking now.",
        "Bear with me.",     "One second.",        "Looking into that.",
        "I'm on it.",        "Give me a moment.",  "Checking the details.",
        "Just a moment.",    "Hold on a second.",  "Let me look that up.",
    };
    for (int i = 0; i < cfg.filler_bank_size && i < 12; ++i) {
        std::vector<float> pcm;
        tts->synthesize(kPhrases[i], [&](const float* p, int n, bool) {
            pcm.insert(pcm.end(), p, p + n);
        });
        if (!pcm.empty()) out[kPhrases[i]] = std::move(pcm);
    }
    return out;
}

constexpr int64_t kLlmDone = 100;  // + kind
constexpr int64_t kTtsDone = 200;
constexpr int64_t kToolDone = 300;  // tool decision finished; text = raw reply

// Timeline.csv cadence for the playback ring rows.
constexpr int64_t kPlaybackTraceMs = 100;

// A queued LLM request, with the turn context needed to write its timeline row
// relative to the same base as metrics.jsonl.
struct PendingText {
    TextKind kind = TextKind::Ack;
    std::string prompt;
    int64_t request_ms = 0;  // absolute steady_ms when the action ran
    int turn = 0;
    int64_t base_ms = 0;     // absolute speech_start of that turn
};

// A queued tool decision for one turn. The transcript is the model's only
// input and also the fallback task request, so it travels with the item.
struct PendingTool {
    std::string transcript;
    int64_t request_ms = 0;
    int turn = 0;
    int64_t base_ms = 0;
};

// A queued TTS piece: same idea. TextSegment alone cannot carry the base.
struct PendingSpeak {
    TextSegment segment;
    int64_t request_ms = 0;
    int turn = 0;
    int64_t base_ms = 0;
};

// `--out` is the output WAV in single-run mode and a directory in harness mode.
// The timeline file lives beside whichever one was given.
std::string timeline_path_for(const std::string& out) {
    struct stat st;
    if (!out.empty() && ::stat(out.c_str(), &st) == 0 && S_ISDIR(st.st_mode))
        return out + "/timeline.csv";
    const size_t slash = out.find_last_of('/');
    if (slash == std::string::npos) return std::string("timeline.csv");
    if (slash == 0) return std::string("/timeline.csv");
    return out.substr(0, slash) + "/timeline.csv";
}

void write_timeline(const std::string& path) {
    std::ofstream tf(path, std::ios::trunc);
    if (!tf) {
        std::fprintf(stderr, "voice: cannot write timeline '%s'\n", path.c_str());
        return;
    }
    tf << timeline_csv_header();
    for (const TimelineRow& row : timeline_trace().rows())
        tf << timeline_csv_row(row);
}

}  // namespace

LoopRun run_once(const Config& cfg_in, Engines& engines,
                 const std::string& input_wav, const std::string& out_wav) {
    LoopRun run;
    Config cfg = cfg_in;
    cfg.normalise();

    if (!engines.asr || !engines.tts || !engines.llm) {
        run.error = "engines incomplete";
        return run;
    }
    if (!engines.asr->ready() || !engines.tts->ready() || !engines.llm->ready()) {
        run.error = "engines not ready";
        return run;
    }

    const int rate = cfg.sample_rate;
    const int frame = cfg.frame_samples;

    // The timeline trace is process-wide (the harness stamps the run number and
    // copies the rows after each run); one call to run_once is one run's worth.
    timeline_trace().clear_rows();

    WavReader reader;
    std::string err;
    if (!reader.open(input_wav, rate, &err)) {
        run.error = "input: " + err;
        return run;
    }
    WavWriter writer;
    if (!writer.open(out_wav, rate, &err)) {
        run.error = "output: " + err;
        return run;
    }

    PcmRing ring(rate * 12);  // room for queued clauses + fillers; never drop
    PacedSink sink(&ring, &writer, rate, frame);
    SpeechSequencer seq(cfg, rate);
    SttDriver stt(rate, cfg.stt_window_ms, cfg.max_turn_ms);
    TurnDetector detector(rate, cfg.vad_min_speech_ms, cfg.vad_min_silence_ms,
                          cfg.vad_energy_threshold);
    InferenceArbiter arbiter;
    arbiter.start();
    // One dispatcher per run: the tool decides the task's latency/failure
    // profile, not the loop. --tool-mode (a Config field) wins, then
    // VOICE_TOOL_MODE, then llm; see tool_call.h.
    ToolDispatcher tools(cfg);
    ToolMode tool_mode = ToolMode::Llm;
    if (cfg.tool_mode.empty() || !parse_tool_mode(cfg.tool_mode, &tool_mode))
        tool_mode = tool_mode_from_env();
    std::fprintf(stderr, "[tool] mode=%s\n", to_string(tool_mode));

    EventQueue events;
    AudioLedger ledger;
    // Filler goes through the same serialised TTS queue as everything else:
    // instant pre-rendered pushes reordered audio behind the answer in the
    // first M1 runs. Pre-rendering stays available for a later watermark-aware
    // variant, but it is off in the measured path.
    std::map<std::string, std::vector<float>> filler_pcm;

    int64_t barge_inject_ms = -1;
    if (const char* env = std::getenv("VOICE_BARGE_IN_AT_MS"))
        barge_inject_ms = std::atoll(env);

    const int64_t t0 = steady_ms();
    int64_t first_played_ack = 0, first_played_answer = 0;
    std::string ack_buf, answer_buf;
    bool answer_pcm_dropped_fillers = false;
    int64_t current_task_id = 0;
    // Incremented per dispatch; the task callback drops outcomes from an older
    // generation (see dispatch_task).
    std::atomic<uint64_t> task_generation{0};
    std::atomic<uint64_t> epoch{0};
    std::atomic<int> work_in_flight{0};
    struct WorkGuard {
        std::atomic<int>& c;
        explicit WorkGuard(std::atomic<int>& v) : c(v) {}
        ~WorkGuard() { c.fetch_sub(1); }
    };

    std::thread playback([&] {
        sink.run([&](int64_t missing_ms) {
            Event e;
            e.type = EventType::PlaybackStarved;
            e.t_ms = steady_ms();
            e.aux = missing_ms;
            events.post(std::move(e));
        });
    });

    auto dispatch_task = [&](const ToolCall& call, const std::string& transcript) {
        const int64_t dispatch_ms = steady_ms();
        const ToolProfile profile = tool_profile(call.tool, cfg);
        const uint64_t my_epoch = epoch.load();
        // seq.on_final() stamps task_dispatch_ms speculatively; restamp it with
        // the real start (a corrective dispatch moves it too).
        seq.metrics().task_dispatch_ms = dispatch_ms;
        // Move the expected result with the dispatch: a corrective redispatch
        // to set_timer answers in ~0.8 s, not in the configured 6 s.
        seq.on_task_redispatch(dispatch_ms + profile.latency_ms);
        // Generation guard: a task superseded by a corrective dispatch must not
        // deliver its outcome (a cancelled device check would otherwise be
        // spoken as a task failure). The counter is published before dispatch,
        // so a zero-latency task cannot race its own start either.
        const uint64_t gen = task_generation.fetch_add(1) + 1;
        const int id = tools.dispatch(
            call, transcript, [&, my_epoch, gen](const TaskOutcome& o) {
                if (my_epoch != epoch.load()) return;
                if (gen != task_generation.load()) return;  // superseded
                Event e;
                e.type = o.cancelled
                             ? EventType::TaskFailed
                             : (o.ok ? EventType::TaskResult : EventType::TaskFailed);
                e.text = o.cancelled ? "cancelled" : o.text;
                e.t_ms = steady_ms();
                // The stamp the handler re-checks. The check above and this
                // post are not atomic: a harness-only fault injector can widen
                // the window (VOICE_TASK_RESULT_HOLD_MS), which is how the
                // stale-event path is tested end to end.
                if (const char* hold = std::getenv("VOICE_TASK_RESULT_HOLD_MS")) {
                    const long long ms = std::atoll(hold);
                    if (ms > 0)
                        std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                }
                e.task_gen = gen;
                e.task_epoch = my_epoch;
                events.post(std::move(e));
            });
        // Every actual adapter dispatch, so a redispatch is visible in the
        // metrics (tool_calls counts model attempts, not dispatches). Counted
        // only when the adapter accepted the request.
        if (id > 0) ++seq.metrics().task_dispatches;
        current_task_id = id;
        if (std::getenv("VOICE_DEBUG"))
            std::fprintf(stderr, "[dbg] dispatch tool=%s id=%d gen=%llu\n",
                         to_string(call.tool), id, (unsigned long long)gen);
    };

    // ---- LLM: one generation at a time, FIFO --------------------------------
    std::mutex llm_mu;
    std::deque<PendingText> llm_pending;
    std::atomic<bool> llm_busy{false};

    const std::string lang_rule =
        cfg.language == "es" ? " Responde siempre en español." : std::string();
    std::function<void()> pump_llm = [&] {
        PendingText item;
        {
            std::lock_guard<std::mutex> lk(llm_mu);
            if (llm_busy.load() || llm_pending.empty()) return;
            llm_busy = true;
            item = llm_pending.front();
            llm_pending.pop_front();
        }
        const TextKind kind = item.kind;
        const std::string prompt = item.prompt;
        const uint64_t my_epoch = epoch.load();
        work_in_flight.fetch_add(1);
        arbiter.post(Priority::Llm, "llm", [&, item, kind, prompt, my_epoch] {
            WorkGuard wg(work_in_flight);
            std::vector<ChatMessage> msgs;
            ChatMessage sys;
            sys.role = "system";
            if (kind == TextKind::Ack) {
                sys.content =
                    "You are a voice assistant speaking to the user. Reply in first "
                    "person, at most eight words. Acknowledge what was asked and say "
                    "you are checking. Never mention \"the user\" or restate the "
                    "request in the third person. No quotation marks, no markdown. "
                    "Never state a result you do not have." + lang_rule;
            } else if (kind == TextKind::Filler) {
                sys.content =
                    "You are a voice assistant. Say one short, natural holding "
                    "phrase. Do not claim progress or a result. Do not repeat "
                    "yourself." + lang_rule;
            } else {
                sys.content =
                    "You are a voice assistant. Answer in one or two short spoken "
                    "sentences, continuing naturally from what you already said. "
                    "No quotation marks, no markdown." + lang_rule;
            }
            ChatMessage user;
            user.role = "user";
            user.content = prompt;
            msgs.push_back(sys);
            msgs.push_back(user);

            const int64_t started = steady_ms();
            int64_t first_token = 0;
            int tokens = 0;
            std::string generated;
            TokenCallback on_token = [&](const std::string& piece) {
                if (my_epoch != epoch.load()) return false;
                if (first_token == 0) first_token = steady_ms();
                ++tokens;
                generated += piece;
                Event e;
                e.type = EventType::Tick;
                e.text = piece;
                e.aux = (int64_t)kind;
                e.t_ms = steady_ms();
                events.post(std::move(e));
                return true;
            };
            const int max_tokens = kind == TextKind::Answer
                                       ? cfg.answer_max_clause_tokens
                                       : 64;
            engines.llm->generate(msgs, max_tokens, on_token, std::string());
            const int64_t finished = steady_ms();
            {
                std::lock_guard<std::mutex> lk(llm_mu);
                llm_busy = false;
            }
            // The row is written even for an aborted generation: "text was
            // requested, produced this much, then the turn was flushed" is
            // exactly the event a barge-in timeline has to show.
            {
                const int64_t base = item.base_ms;
                TimelineRow row;
                row.turn = item.turn;
                row.stage = "llm";
                row.kind = to_string(kind);
                row.text = generated;
                row.t_request_ms = base > 0 ? item.request_ms - base : -1;
                row.t_job_start_ms = base > 0 ? started - base : -1;
                // No PCM here: the first output of an LLM job is its first
                // token, so that is what the column carries.
                row.t_first_pcm_ms = (first_token > 0 && base > 0) ? first_token - base : -1;
                row.t_first_audible_ms = -1;
                row.t_done_ms = base > 0 ? finished - base : -1;
                row.samples = tokens;
                row.audible_ms = 0;
                timeline_trace().add(std::move(row));
            }
            if (my_epoch != epoch.load()) return;  // aborted: no stale done event
            Event done;
            done.type = EventType::Tick;
            done.aux = kLlmDone + (int64_t)kind;
            done.aux2 = first_token > 0 ? first_token - started : 0;
            done.aux3 = (int64_t)kind;
            done.t_ms = steady_ms();
            events.post(std::move(done));
        });
    };

    // ---- tool decision: one per turn, queued *behind* the ack --------------─
    // Same arbiter, same chat session: the ack generation is already queued
    // when this runs, so the spoken ack never waits for the decision. The job
    // posts its parsed call back to the control thread; only that thread owns
    // the task id and the metrics.
    std::mutex tool_mu;
    std::deque<PendingTool> tool_pending;
    std::atomic<bool> tool_busy{false};

    std::function<void()> pump_tool = [&] {
        PendingTool item;
        {
            std::lock_guard<std::mutex> lk(tool_mu);
            if (tool_busy.load() || tool_pending.empty()) return;
            tool_busy = true;
            item = tool_pending.front();
            tool_pending.pop_front();
        }
        const uint64_t my_epoch = epoch.load();
        work_in_flight.fetch_add(1);
        arbiter.post(Priority::Llm, "tool", [&, item, my_epoch] {
            WorkGuard wg(work_in_flight);
            const int64_t started = steady_ms();
            int64_t first_token = 0;
            int tokens = 0;
            std::string raw;
            if (tool_mode == ToolMode::Llm) {
                const std::vector<ChatMessage> msgs = tool_call_messages(item.transcript);
                TokenCallback on_token = [&](const std::string& piece) {
                    if (my_epoch != epoch.load()) return false;
                    if (first_token == 0) first_token = steady_ms();
                    ++tokens;
                    raw += piece;
                    return true;
                };
                engines.llm->generate(msgs, kToolCallMaxTokens, on_token,
                                      std::string());
            } else {
                // Harness override: deterministic reply, no model call, same
                // parsing and dispatch path as the real thing.
                raw = fake_tool_reply(tool_mode, item.transcript);
            }
            const int64_t finished = steady_ms();
            {
                std::lock_guard<std::mutex> lk(tool_mu);
                tool_busy = false;
            }
            const ToolCall call = parse_tool_call(raw);
            // One timeline row per decision: which tool the model picked and
            // the reply it picked it from, on the same base as metrics.jsonl.
            {
                TimelineRow row;
                row.turn = item.turn;
                row.stage = "tool";
                row.kind = to_string(call.tool);
                row.text = raw;
                row.t_request_ms = item.base_ms > 0 ? item.request_ms - item.base_ms : -1;
                row.t_job_start_ms = item.base_ms > 0 ? started - item.base_ms : -1;
                row.t_done_ms = item.base_ms > 0 ? finished - item.base_ms : -1;
                row.samples = (tool_mode == ToolMode::Llm) ? tokens
                                                           : (int64_t)raw.size();
                timeline_trace().add(std::move(row));
            }
            if (my_epoch != epoch.load()) return;  // turn flushed: no dispatch
            Event done;
            done.type = EventType::Tick;
            done.aux = kToolDone;
            done.aux2 = call.valid ? 1 : 0;
            done.aux3 = (int64_t)call.tool;
            done.text = raw;
            done.t_ms = finished;
            events.post(std::move(done));
        });
    };

    // ---- TTS: one piece at a time, FIFO (ordering is audible) ---------------
    std::mutex tts_mu;
    std::deque<PendingSpeak> tts_pending;
    std::atomic<bool> tts_busy{false};

    std::function<void()> pump_tts = [&] {
        PendingSpeak item;
        {
            std::lock_guard<std::mutex> lk(tts_mu);
            if (tts_busy.load() || tts_pending.empty()) return;
            // Backpressure: synthesising 4x faster than playback must not
            // outrun the ring. The answer is exempt: it must be buffered the
            // moment it exists, or it waits for the watermark to drain.
            if (tts_pending.front().segment.kind != TextKind::Answer &&
                ring.buffer_ms(rate) >= cfg.high_water_ms)
                return;
            tts_busy = true;
            item = tts_pending.front();
            tts_pending.pop_front();
        }
        const uint64_t my_epoch = epoch.load();
        const Priority pr = ring.buffer_ms(rate) < cfg.low_water_ms
                                ? Priority::TtsUrgent
                                : Priority::TtsIdle;
        work_in_flight.fetch_add(1);
        arbiter.post(pr, "tts", [&, item, my_epoch] {
            WorkGuard wg(work_in_flight);
            const TextSegment& seg = item.segment;
            const int64_t base = item.base_ms;
            const int64_t job_started = steady_ms();
            int64_t first_pcm = 0;
            int64_t samples = 0;
            AudibilityMeter meter(rate);
            engines.tts->synthesize(seg.text, [&](const float* p, int n, bool) {
                if (my_epoch != epoch.load()) return;
                const int64_t t = steady_ms();
                if (first_pcm == 0) first_pcm = t;
                meter.push(p, n);
                samples += n;
                ring.push(p, n);
                ledger.add(seg.kind, n);
            });
            meter.finish();
            const int64_t job_done = steady_ms();
            {
                std::lock_guard<std::mutex> lk(tts_mu);
                tts_busy = false;
            }
            // Per-piece row: the whole point of the trace. A piece that emits
            // 400 ms of model silence before speech shows up as
            // t_first_audible_ms much later than t_first_pcm_ms; a piece that
            // simply arrives late shows up as a large t_job_start_ms.
            {
                TimelineRow row;
                row.turn = item.turn;
                row.stage = "tts";
                row.kind = to_string(seg.kind);
                row.text = seg.text;
                row.t_request_ms = base > 0 ? item.request_ms - base : -1;
                row.t_job_start_ms = base > 0 ? job_started - base : -1;
                row.t_first_pcm_ms = (first_pcm > 0 && base > 0) ? first_pcm - base : -1;
                // Emission time of the first chunk, plus the offset of the
                // first loud 10 ms window inside the piece: the emitted stream
                // gains that silence before it becomes audible.
                row.t_first_audible_ms =
                    (first_pcm > 0 && base > 0 && meter.found())
                        ? (first_pcm - base) + meter.first_window() * meter.window_ms()
                        : -1;
                row.t_done_ms = base > 0 ? job_done - base : -1;
                row.samples = samples;
                row.audible_ms = meter.audible_ms();
                timeline_trace().add(std::move(row));
            }
            if (my_epoch != epoch.load()) return;  // aborted: no stale done event
            Event done;
            done.type = EventType::Tick;
            done.aux = kTtsDone;
            done.aux2 = first_pcm > 0 ? first_pcm - job_started : 0;
            done.aux3 = (int64_t)seg.kind;
            done.t_ms = first_pcm;
            events.post(std::move(done));
        });
    };

    // Set on a final transcript, consumed after the sequencer's actions run so
    // the ack request is always ahead of the decision in the LLM queue.
    bool tool_request_pending = false;
    std::string tool_request_transcript;

    auto execute = [&](std::vector<Action>& actions) {
        for (Action& a : actions) {
            switch (a.type) {
                case ActionType::RequestText: {
                    PendingText item;
                    item.kind = a.kind;
                    item.prompt = a.text;
                    item.request_ms = steady_ms();
                    item.turn = seq.turn_id();
                    item.base_ms = seq.metrics().speech_start_ms;
                    {
                        std::lock_guard<std::mutex> lk(llm_mu);
                        llm_pending.push_back(std::move(item));
                    }
                    pump_llm();
                    break;
                }
                case ActionType::Speak: {
                    if (std::getenv("VOICE_DEBUG"))
                        std::fprintf(stderr, "[dbg] speak kind=%d len=%zu text='%.30s'\n",
                                     (int)a.segment.kind, a.segment.text.size(),
                                     a.segment.text.c_str());
                    auto it = filler_pcm.find(a.segment.text);
                    if (a.segment.kind == TextKind::Filler && it != filler_pcm.end()) {
                        ring.push(it->second.data(), (int)it->second.size());
                        ledger.add(a.segment.kind, (int64_t)it->second.size());
                    } else {
                        PendingSpeak item;
                        item.segment = a.segment;
                        item.request_ms = steady_ms();
                        item.turn = seq.turn_id();
                        item.base_ms = seq.metrics().speech_start_ms;
                        {
                            std::lock_guard<std::mutex> lk(tts_mu);
                            tts_pending.push_back(std::move(item));
                        }
                        pump_tts();
                    }
                    break;
                }
                case ActionType::FlushAudio: {
                    epoch.fetch_add(1);
                    ring.clear();
                    ledger.clear();
                    {
                        std::lock_guard<std::mutex> lk(tts_mu);
                        tts_pending.clear();
                        tts_busy = false;
                    }
                    {
                        std::lock_guard<std::mutex> lk(llm_mu);
                        llm_pending.clear();
                        llm_busy = false;
                    }
                    {
                        std::lock_guard<std::mutex> lk(tool_mu);
                        tool_pending.clear();
                        tool_busy = false;
                    }
                    tool_request_pending = false;
                    engines.llm->request_abort();
                    seq.metrics().barge_in_stop_ms =
                        std::max<int64_t>(0, steady_ms() - a.arg);
                    break;
                }
                case ActionType::CancelTask:
                    if (current_task_id > 0) tools.cancel(current_task_id);
                    break;
                case ActionType::FinishTurn: {
                    TurnMetrics m = seq.metrics();
                    const int64_t base = m.speech_start_ms;
                    auto rel = [&](int64_t& v) {
                        if (v > 0 && base > 0) v -= base;
                    };
                    rel(m.speech_end_ms); rel(m.final_transcript_ms);
                    rel(m.ack_text_ms); rel(m.ack_first_pcm_ms);
                    rel(m.ack_first_played_ms); rel(m.task_dispatch_ms);
                    rel(m.task_result_ms); rel(m.answer_text_ms);
                    rel(m.answer_first_pcm_ms); rel(m.answer_first_played_ms);
                    rel(m.end_ms);
                    for (auto& g : m.gaps) rel(g.at_ms);
                    m.speech_start_ms = 0;
                    if (std::getenv("VOICE_DEBUG"))
                        std::fprintf(stderr,
                                     "[dbg] finish ack_played=%lld ans_played=%lld ans_pcm=%lld fillers=%d\n",
                                     (long long)m.ack_first_played_ms,
                                     (long long)m.answer_first_played_ms,
                                     (long long)m.answer_first_pcm_ms,
                                     (int)m.fillers_played);
                    run.metrics = m;
                    if (cfg.out_dir != ".") {
                        std::ofstream mf(cfg.out_dir + "/metrics.jsonl",
                                         std::ios::app);
                        mf << metrics_json(run.metrics) << "\n";
                    }
                    run.ok = true;
                    break;
                }
            }
        }
    };

    PacedSource source(&reader, rate, frame);
    std::vector<float> frame_buf((size_t)frame);
    int64_t last_partial_ms = 0, last_tick_ms = 0, last_event_ms = steady_ms();
    int64_t last_playback_trace_ms = 0;
    bool speech_final_posted = false;
    bool at_eof = false;
    bool barge_done = false;
    int64_t deadline = 0;
    std::vector<Event> pending;

    while (true) {
        const int n = source.read(frame_buf.data(), frame, nullptr);
        const int64_t now = steady_ms();

        // Harness barge-in marker (VOICE_BARGE_IN_AT_MS). Checked every tick,
        // including after the input WAV has ended: a barge-in does not need
        // new microphone audio to be injected.
        if (barge_inject_ms >= 0 && !barge_done && now - t0 >= barge_inject_ms) {
            barge_done = true;
            Event e;
            e.type = EventType::BargeIn;
            e.t_ms = now;
            events.post(std::move(e));
        }

        if (n <= 0) {
            if (detector.speech_active() && !speech_final_posted) {
                speech_final_posted = true;
                std::vector<float> audio = stt.final_audio();
                const uint64_t my_epoch = epoch.load();
                work_in_flight.fetch_add(1);
                arbiter.post(Priority::SttFinal, "stt-final",
                             [&, audio, my_epoch] {
                    WorkGuard wg(work_in_flight);
                    if (my_epoch != epoch.load()) return;
                    AsrResult r = engines.asr->transcribe(
                        audio.data(), (int)audio.size(), 0);
                    if (std::getenv("VOICE_DEBUG"))
                        std::fprintf(stderr, "[dbg] final audio=%zu text='%.60s' cost=%lld\n",
                                     audio.size(), r.text.c_str(), (long long)r.cost_ms);
                    Event ev;
                    ev.type = EventType::FinalTranscript;
                    ev.text = r.ok ? r.text : std::string();
                    ev.t_ms = steady_ms();
                    ev.aux2 = r.cost_ms;
                    events.post(std::move(ev));
                });
            }
            if (deadline == 0) deadline = now + cfg.max_turn_ms;
            at_eof = true;
            if (now > deadline) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        } else {
            last_event_ms = now;
            TurnEvents te;
            te.on_speech_start = [&](int64_t t) {
                speech_final_posted = false;
                stt.begin_turn(t);
                Event e;
                e.type = EventType::SpeechStart;
                e.t_ms = t;
                events.post(std::move(e));
            };
            te.on_speech_end = [&](int64_t t) {
                // Snapshot before end_turn(): the driver resets its buffer.
                std::vector<float> audio = stt.final_audio();
                stt.end_turn(t);
                speech_final_posted = true;
                const uint64_t my_epoch = epoch.load();
                work_in_flight.fetch_add(1);
                arbiter.post(Priority::SttFinal, "stt-final",
                             [&, audio, my_epoch] {
                    WorkGuard wg(work_in_flight);
                    if (my_epoch != epoch.load()) return;
                    AsrResult r = engines.asr->transcribe(
                        audio.data(), (int)audio.size(), 0);
                    if (std::getenv("VOICE_DEBUG"))
                        std::fprintf(stderr, "[dbg] final audio=%zu text='%.60s' cost=%lld\n",
                                     audio.size(), r.text.c_str(), (long long)r.cost_ms);
                    Event ev;
                    ev.type = EventType::FinalTranscript;
                    ev.text = r.ok ? r.text : std::string();
                    ev.t_ms = steady_ms();
                    ev.aux2 = r.cost_ms;
                    events.post(std::move(ev));
                });
            };
            te.on_barge_in = [&](int64_t t) {
                Event e;
                e.type = EventType::BargeIn;
                e.t_ms = t;
                events.post(std::move(e));
            };
            const bool floor = seq.turn_active() || ring.buffer_ms(rate) > 0;
            detector.push(frame_buf.data(), n, now, floor, te);
            if (std::getenv("VOICE_DEBUG")) {
                static int dbg = 0;
                if ((dbg++ % 10) == 0)
                    std::fprintf(stderr, "[dbg] n=%d t=%lld speech=%d state=%d buf=%lld\n",
                                 n, (long long)(now - t0), (int)detector.speech_active(),
                                 (int)seq.state(), (long long)ring.buffer_ms(rate));
            }
            if (detector.speech_active()) stt.push(frame_buf.data(), n);

            if (detector.speech_active() && cfg.stt_partials_enabled &&
                now - last_partial_ms >= cfg.stt_partial_ms) {
                last_partial_ms = now;
                std::vector<float> win = stt.partial_window();
                const uint64_t my_epoch = epoch.load();
                arbiter.drop_queued_partials();
                work_in_flight.fetch_add(1);
                arbiter.post(Priority::SttPartial, "stt-partial",
                             [&, win, my_epoch] {
                    WorkGuard wg(work_in_flight);
                    if (my_epoch != epoch.load()) return;
                    AsrResult r =
                        engines.asr->transcribe(win.data(), (int)win.size(), 0);
                    Event ev;
                    ev.type = EventType::PartialTranscript;
                    ev.text = r.ok ? r.text : std::string();
                    ev.t_ms = steady_ms();
                    ev.aux2 = r.cost_ms;
                    events.post(std::move(ev));
                });
            }
        }

        events.drain(&pending);
        for (Event& e : pending) {
            switch (e.type) {
                case EventType::SpeechStart:
                    seq.on_speech_start(e.t_ms);
                    break;
                case EventType::PartialTranscript:
                    seq.on_partial(e.text, e.t_ms);
                    seq.metrics().stt_partials++;
                    seq.metrics().stt_partial_cost_ms += e.aux2;
                    break;
                case EventType::FinalTranscript:
                    if (std::getenv("VOICE_DEBUG"))
                        std::fprintf(stderr, "[dbg] final event text='%.40s' state=%d\n",
                                     e.text.c_str(), (int)seq.state());
                    seq.metrics().stt_final_cost_ms = e.aux2;
                    seq.metrics().transcript = e.text;
                    seq.on_final(e.text, e.t_ms);
                    if (!e.text.empty()) {
                        if (tool_mode == ToolMode::Off) {
                            // Pre-M2 behaviour, kept as the regression baseline.
                            ToolCall legacy;
                            legacy.tool = ToolId::DeviceStatus;
                            legacy.device = "garage";
                            legacy.reason = "tool_mode=off";
                            dispatch_task(legacy, e.text);
                        } else {
                            // Speculative dispatch: start the common device check
                            // now (the pre-M2 instant) and let the decision
                            // cancel it only when it chose something else. The
                            // alternative — waiting ~150 ms for the JSON —
                            // shifts the whole turn and widens the pre-answer
                            // gap, because the sequencer's filler reserve is
                            // anchored to the final transcript.
                            ToolCall spec;
                            spec.tool = ToolId::DeviceStatus;
                            spec.device = "garage";
                            spec.reason = "speculative";
                            dispatch_task(spec, e.text);
                            tool_request_transcript = e.text;
                            tool_request_pending = true;
                            // Counted now, not at completion: an attempt whose
                            // turn is flushed mid-generation still happened.
                            ++seq.metrics().tool_calls;
                        }
                    }
                    break;
                case EventType::BargeIn:
                    seq.on_barge_in(e.t_ms);
                    break;
                case EventType::TaskResult:
                case EventType::TaskFailed:
                    // Re-check the stamp: the callback's generation/epoch check
                    // ran before the post, so an outcome queued before a
                    // redispatch can still be sitting in the queue after it.
                    // Speaking it would start the answer from a superseded
                    // task, and the result we actually kept would then be
                    // dropped as stale (defect F1 of the M2 review).
                    if (task_event_is_stale(e, task_generation.load(),
                                            epoch.load())) {
                        std::fprintf(stderr,
                                     "[tool] stale task result dropped "
                                     "(gen %llu -> %llu, epoch %llu -> %llu)\n",
                                     (unsigned long long)e.task_gen,
                                     (unsigned long long)task_generation.load(),
                                     (unsigned long long)e.task_epoch,
                                     (unsigned long long)epoch.load());
                        seq.on_stale_task_result();
                        break;
                    }
                    seq.on_task_result(e.type == EventType::TaskResult, e.text,
                                       e.t_ms);
                    break;
                case EventType::PlaybackStarved:
                    seq.on_starved(e.t_ms, e.aux);
                    break;
                case EventType::Tick:
                    if (e.aux == kToolDone) {
                        // The event carries the raw reply; the parse is
                        // deterministic, so re-parse instead of shipping two
                        // encodings through one Event.
                        const ToolCall call = parse_tool_call(e.text);
                        // A speculative device_status(garage) is already
                        // running (see FinalTranscript). Only a disagreement
                        // costs a cancel + re-dispatch.
                        const bool agrees = call.valid &&
                                            call.tool == ToolId::DeviceStatus &&
                                            call.device == "garage";
                        ToolCall chosen = call;
                        const char* action = "kept";
                        if (call.valid) {
                            ++seq.metrics().tool_calls_valid;
                            if (!agrees) {
                                if (current_task_id > 0) tools.cancel(current_task_id);
                                action = "redispatch";
                                dispatch_task(chosen, seq.metrics().transcript);
                            }
                        } else {
                            // Unusable reply: keep the speculative device check
                            // and use the transcript as the request.
                            chosen.tool = ToolId::DeviceStatus;
                            chosen.device = "garage";
                            chosen.minutes = 0;
                            chosen.valid = false;
                            chosen.reason = "fallback (" + call.reason + ")";
                            action = "fallback";
                        }
                        std::string flat = e.text;
                        for (char& c : flat)
                            if (c == '\n' || c == '\r' || c == '\t') c = ' ';
                        const std::string args =
                            (chosen.device.empty() ? std::string() : chosen.device) +
                            (chosen.minutes > 0
                                 ? " minutes=" + std::to_string(chosen.minutes)
                                 : std::string());
                        std::fprintf(stderr,
                                     "[tool] mode=%s valid=%d tool=%s args=[%s] "
                                     "action=%s reason='%s' raw='%s'\n",
                                     to_string(tool_mode), (int)call.valid,
                                     to_string(chosen.tool), args.c_str(), action,
                                     call.reason.c_str(), flat.c_str());
                    } else if (e.aux >= kLlmDone && e.aux < kLlmDone + 4) {
                        seq.metrics().llm_ttft_ms = std::max(
                            seq.metrics().llm_ttft_ms, e.aux2);
                        if ((TextKind)e.aux3 == TextKind::Ack)
                            seq.metrics().ack_text = ack_buf;
                        if ((TextKind)e.aux3 == TextKind::Answer)
                            seq.metrics().answer_text = answer_buf;
                        seq.on_text_done((TextKind)e.aux3, e.t_ms);
                        pump_llm();
                    } else if (e.aux == kTtsDone) {
                        seq.metrics().tts_pieces++;
                        seq.metrics().tts_ttfa_ms =
                            std::max(seq.metrics().tts_ttfa_ms, e.aux2);
                        if ((TextKind)e.aux3 == TextKind::Ack &&
                            seq.metrics().ack_first_pcm_ms == 0)
                            seq.metrics().ack_first_pcm_ms = e.t_ms;
                        if ((TextKind)e.aux3 == TextKind::Answer &&
                            seq.metrics().answer_first_pcm_ms == 0)
                            seq.metrics().answer_first_pcm_ms = e.t_ms;
                        pump_tts();
                    } else if (e.aux >= 0 && e.aux < 4) {
                        if ((TextKind)e.aux == TextKind::Ack) ack_buf += e.text;
                        if ((TextKind)e.aux == TextKind::Answer) answer_buf += e.text;
                        seq.on_text_delta((TextKind)e.aux, e.text, e.t_ms);
                    }
                    break;
                default:
                    break;
            }
        }
        pending.clear();

        std::vector<Action> actions = seq.take_actions();
        if (!actions.empty()) execute(actions);

        // The ack (or clarify) request is in the LLM queue by now.
        if (tool_request_pending) {
            tool_request_pending = false;
            PendingTool item;
            item.transcript = tool_request_transcript;
            item.request_ms = steady_ms();
            item.turn = seq.turn_id();
            item.base_ms = seq.metrics().speech_start_ms;
            {
                std::lock_guard<std::mutex> lk(tool_mu);
                tool_pending.push_back(std::move(item));
            }
            pump_tool();
        }

        // Decide after draining events and executing their actions: a finished
        // ASR job may have queued the final transcript this very iteration.
        if (at_eof && seq.state() == TurnState::Idle && ring.available() == 0 &&
            !tts_busy && !llm_busy && !tool_busy && work_in_flight.load() == 0)
            break;

        if (now - last_tick_ms >= 20) {
            last_tick_ms = now;
            const int64_t real_played = sink.real_played_samples();
            std::map<TextKind, int64_t> first_play;
            ledger.consume(real_played, &first_play, now);
            if (std::getenv("VOICE_DEBUG") && !first_play.empty()) {
                for (auto& kv : first_play)
                    std::fprintf(stderr, "[dbg] first_play kind=%d at=%lld\n",
                                 (int)kv.first, (long long)(kv.second - t0));
            }
            for (auto& kv : first_play) {
                if (kv.first == TextKind::Ack && first_played_ack == 0) {
                    first_played_ack = kv.second;
                    seq.metrics().ack_first_played_ms = kv.second;
                }
                if (kv.first == TextKind::Answer && first_played_answer == 0) {
                    first_played_answer = kv.second;
                    seq.metrics().answer_first_played_ms = kv.second;
                }
            }
            {
                std::lock_guard<std::mutex> lk(tts_mu);
                seq.set_audio_in_flight(ring.available() > 0 || tts_busy ||
                                        !tts_pending.empty());
            }
            if (std::getenv("VOICE_DEBUG")) {
                static int64_t last_dbg = 0;
                if (now - last_dbg >= 500) {
                    last_dbg = now;
                    size_t tq;
                    { std::lock_guard<std::mutex> lk(tts_mu); tq = tts_pending.size(); }
                    std::fprintf(stderr,
                                 "[dbg] t=%lld buf=%lld avail=%d real=%lld ttsq=%zu busy=%d state=%d\n",
                                 (long long)(now - t0), (long long)ring.buffer_ms(rate),
                                 ring.available(), (long long)sink.real_played_samples(),
                                 tq, (int)tts_busy, (int)seq.state());
                }
            }
            pump_tts();  // deferred pieces resume as the ring drains
            seq.on_playback(ring.buffer_ms(rate), sink.real_played_ms(), now);

            // Ring-depth probe: one row per 100 ms while a turn is active. Read
            // next to the TTS arrivals, it separates "the model was quiet"
            // from "the ring ran dry".
            if (now - last_playback_trace_ms >= kPlaybackTraceMs) {
                last_playback_trace_ms = now;
                if (seq.turn_active()) {
                    const int64_t base = seq.metrics().speech_start_ms;
                    TimelineRow row;
                    row.turn = seq.turn_id();
                    row.stage = "playback";
                    row.t_request_ms = base > 0 ? now - base : now;
                    row.samples = ring.buffer_ms(rate);   // ring depth, ms
                    row.audible_ms = sink.real_played_ms();  // real played, ms
                    timeline_trace().add(std::move(row));
                }
            }

            // A barge-in that produced no follow-up speech still ends the turn.
            if (seq.state() == TurnState::Listening && now - last_event_ms > 3000)
                seq.on_final(std::string(), now);
        }

    }

    sink.stop();
    playback.join();
    writer.finish();
    tools.shutdown();
    arbiter.stop();

    // Harness mode owns the file (it truncates once per invocation and appends
    // every run); a single run writes its own timeline next to its WAV.
    if (cfg.mode != "harness") write_timeline(timeline_path_for(cfg.out_dir));

    // run.ok is set only by a FinishTurn action: a turn that never completed
    // is reported as a failure instead of being masked.
    return run;
}

}  // namespace voice
}  // namespace crispasr
