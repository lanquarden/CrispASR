// voice_loop.cpp — orchestrator for the realtime voice loop.
//
// Threads: capture/control (this function), inference arbiter, playback, and
// one thread per simulated task. All inference goes through the arbiter, so
// the three models never run concurrently on the shared Vulkan device.
#include "voice_loop.h"

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
    SimulatedTaskAdapter task(cfg.task_ms, cfg.task_jitter_ms,
                              cfg.task_fail_percent);

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
    int64_t current_task_id = 0;
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

    auto dispatch_task = [&](const std::string& transcript) {
        const uint64_t my_epoch = epoch.load();
        current_task_id =
            task.dispatch(transcript, [&, my_epoch](const TaskOutcome& o) {
                if (my_epoch != epoch.load()) return;
                Event e;
                e.type = o.cancelled
                             ? EventType::TaskFailed
                             : (o.ok ? EventType::TaskResult : EventType::TaskFailed);
                e.text = o.cancelled ? "cancelled" : o.text;
                e.t_ms = steady_ms();
                events.post(std::move(e));
            });
    };

    // ---- LLM: one generation at a time, FIFO --------------------------------
    std::mutex llm_mu;
    std::deque<std::pair<TextKind, std::string>> llm_pending;
    bool llm_busy = false;

    std::function<void()> pump_llm = [&] {
        std::pair<TextKind, std::string> item;
        {
            std::lock_guard<std::mutex> lk(llm_mu);
            if (llm_busy || llm_pending.empty()) return;
            llm_busy = true;
            item = llm_pending.front();
            llm_pending.pop_front();
        }
        const TextKind kind = item.first;
        const std::string prompt = item.second;
        const uint64_t my_epoch = epoch.load();
        work_in_flight.fetch_add(1);
        arbiter.post(Priority::Llm, "llm", [&, kind, prompt, my_epoch] {
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
                    "Never state a result you do not have.";
            } else if (kind == TextKind::Filler) {
                sys.content =
                    "You are a voice assistant. Say one short, natural holding "
                    "phrase. Do not claim progress or a result. Do not repeat "
                    "yourself.";
            } else {
                sys.content =
                    "You are a voice assistant. Answer in one or two short spoken "
                    "sentences, continuing naturally from what you already said.";
            }
            ChatMessage user;
            user.role = "user";
            user.content = prompt;
            msgs.push_back(sys);
            msgs.push_back(user);

            const int64_t started = steady_ms();
            int64_t first_token = 0;
            TokenCallback on_token = [&](const std::string& piece) {
                if (my_epoch != epoch.load()) return false;
                if (first_token == 0) first_token = steady_ms();
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
            {
                std::lock_guard<std::mutex> lk(llm_mu);
                llm_busy = false;
            }
            Event done;
            done.type = EventType::Tick;
            done.aux = kLlmDone + (int64_t)kind;
            done.aux2 = first_token > 0 ? first_token - started : 0;
            done.aux3 = (int64_t)kind;
            done.t_ms = steady_ms();
            events.post(std::move(done));
        });
    };

    // ---- TTS: one piece at a time, FIFO (ordering is audible) ---------------
    std::mutex tts_mu;
    std::deque<TextSegment> tts_pending;
    bool tts_busy = false;

    std::function<void()> pump_tts = [&] {
        TextSegment seg;
        {
            std::lock_guard<std::mutex> lk(tts_mu);
            if (tts_busy || tts_pending.empty()) return;
            // Backpressure: synthesising 4x faster than playback must not
            // outrun the ring. The answer is exempt: it must be buffered the
            // moment it exists, or it waits for the watermark to drain.
            if (tts_pending.front().kind != TextKind::Answer &&
                ring.buffer_ms(rate) >= cfg.high_water_ms)
                return;
            tts_busy = true;
            seg = tts_pending.front();
            tts_pending.pop_front();
        }
        const uint64_t my_epoch = epoch.load();
        const Priority pr = ring.buffer_ms(rate) < cfg.low_water_ms
                                ? Priority::TtsUrgent
                                : Priority::TtsIdle;
        work_in_flight.fetch_add(1);
        arbiter.post(pr, "tts", [&, seg, my_epoch] {
            WorkGuard wg(work_in_flight);
            const int64_t job_started = steady_ms();
            int64_t first_pcm = 0;
            engines.tts->synthesize(seg.text, [&](const float* p, int n, bool) {
                if (my_epoch != epoch.load()) return;
                if (first_pcm == 0) first_pcm = steady_ms();
                ring.push(p, n);
                ledger.add(seg.kind, n);
            });
            {
                std::lock_guard<std::mutex> lk(tts_mu);
                tts_busy = false;
            }
            Event done;
            done.type = EventType::Tick;
            done.aux = kTtsDone;
            done.aux2 = first_pcm > 0 ? first_pcm - job_started : 0;
            done.aux3 = (int64_t)seg.kind;
            done.t_ms = first_pcm;
            events.post(std::move(done));
        });
    };

    auto execute = [&](std::vector<Action>& actions) {
        for (Action& a : actions) {
            switch (a.type) {
                case ActionType::RequestText: {
                    {
                        std::lock_guard<std::mutex> lk(llm_mu);
                        llm_pending.emplace_back(a.kind, a.text);
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
                        {
                            std::lock_guard<std::mutex> lk(tts_mu);
                            tts_pending.push_back(a.segment);
                        }
                        pump_tts();
                    }
                    break;
                }
                case ActionType::FlushAudio: {
                    epoch.fetch_add(1);
                    ring.clear();
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
                    engines.llm->request_abort();
                    seq.metrics().barge_in_stop_ms =
                        std::max<int64_t>(0, steady_ms() - a.arg);
                    break;
                }
                case ActionType::CancelTask:
                    if (current_task_id > 0) task.cancel(current_task_id);
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
                    if (!e.text.empty()) dispatch_task(e.text);
                    break;
                case EventType::BargeIn:
                    seq.on_barge_in(e.t_ms);
                    break;
                case EventType::TaskResult:
                    seq.on_task_result(true, e.text, e.t_ms);
                    break;
                case EventType::TaskFailed:
                    seq.on_task_result(false, e.text, e.t_ms);
                    break;
                case EventType::PlaybackStarved:
                    seq.on_starved(e.t_ms, e.aux);
                    break;
                case EventType::Tick:
                    if (e.aux >= kLlmDone && e.aux < kLlmDone + 4) {
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

        // Decide after draining events and executing their actions: a finished
        // ASR job may have queued the final transcript this very iteration.
        if (at_eof && seq.state() == TurnState::Idle && ring.available() == 0 &&
            !tts_busy && !llm_busy && work_in_flight.load() == 0)
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

            // A barge-in that produced no follow-up speech still ends the turn.
            if (seq.state() == TurnState::Listening && now - last_event_ms > 3000)
                seq.on_final(std::string(), now);
        }

    }

    sink.stop();
    playback.join();
    writer.finish();
    task.shutdown();
    arbiter.stop();
    if (!run.ok) run.ok = true;
    return run;
}

}  // namespace voice
}  // namespace crispasr
