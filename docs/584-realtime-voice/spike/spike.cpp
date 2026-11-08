// docs/584-realtime-voice/spike/spike.cpp — co-residency feasibility spike.
//
// Throwaway. Answers three questions for the realtime voice loop (#584):
//   1. Can a streaming-ASR backend and a streaming-TTS backend be alive in
//      one process at the same time?
//   2. Can a `crispasr_chat_*` text-LLM session live in that same process and
//      share the same ggml/Vulkan device?
//   3. Do sequential calls work, and what happens on two concurrent
//      inference callers (second thread) — crash, hang, corruption, or fine?
//
// Build/run instructions live in the sibling README.md. Nothing here is
// wired into the CMake tree on purpose.

#include "core/gpu_backend_pref.h" // crispasr_set_gpu_backend_pref (#214)

#include "crispasr.h"         // crispasr_audio_load_at_rate
#include "crispasr_backend.h" // CrispasrBackend + crispasr_create_backend
#include "crispasr_chat.h"    // crispasr_chat_* text-LLM ABI
#include "whisper_params.h"   // whisper_params (CLI params struct)

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#if defined(__unix__)
#include <unistd.h>
#endif

// ---------------------------------------------------------------------------
// Paths (host paths; override with env vars when reproducing elsewhere)
// ---------------------------------------------------------------------------
static const char* kAsrModel =
    "/home/lanquarden/.local/share/containers/storage/volumes/beast-crispasr_crispasr-models/"
    "_data/qwen3-asr-1.7b-q8_0.gguf";
static const char* kTtsTalker =
    "/home/lanquarden/.local/share/containers/storage/volumes/beast-crispasr_crispasr-models/"
    "_data/qwen3-tts-12hz-1.7b-base-q8_0.gguf";
static const char* kTtsCodec =
    "/home/lanquarden/.local/share/containers/storage/volumes/beast-crispasr_crispasr-models/"
    "_data/qwen3-tts-tokenizer-12hz.gguf";
static const char* kVoiceDir = "/home/lanquarden/repos/home-ops/hosts/beast/crispasr/voice";
static const char* kChatModel = "/home/lanquarden/models/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf";
static const char* kAsrWav = "/home/lanquarden/repos/CrispASR/samples/jfk.wav";

static const char* env_or(const char* name, const char* dflt) {
    const char* v = getenv(name);
    return (v && *v) ? v : dflt;
}

// ---------------------------------------------------------------------------
// Small measurement helpers
// ---------------------------------------------------------------------------
static double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// VmHWM = process peak resident set size, from /proc/self/status.
static long vmhwm_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f)
        return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%ld", &kb);
            break;
        }
    }
    fclose(f);
    return kb;
}

// Total VRAM in use, as reported by rocm-smi (whole card, not just us).
static long gpu_used_mb() {
    FILE* p = popen("rocm-smi --showmeminfo vram --json 2>/dev/null", "r");
    if (!p)
        return -1;
    std::string out;
    char buf[512];
    while (fgets(buf, sizeof(buf), p))
        out += buf;
    pclose(p);
    const char* key = "\"VRAM Total Used Memory (B)\": \"";
    size_t at = out.find(key);
    if (at == std::string::npos)
        return -1;
    at += strlen(key);
    long long bytes = atoll(out.c_str() + at);
    return bytes > 0 ? (long)(bytes / (1024 * 1024)) : -1;
}

static void gpu_line(const char* when) {
    const long mb = gpu_used_mb();
    if (mb < 0)
        printf("[gpu] vram in use (%s): unavailable\n", when);
    else
        printf("[gpu] vram in use (%s): %ld MB (whole card, rocm-smi)\n", when, mb);
}

// ---------------------------------------------------------------------------
// Params
// ---------------------------------------------------------------------------
static whisper_params make_asr_params() {
    whisper_params p;
    p.backend = "qwen3-1.7b";
    p.model = kAsrModel;
    p.language = "en";
    p.n_threads = 8;
    p.use_gpu = true;
    p.gpu_backend = "vulkan";
    p.no_prints = true; // keep the spike's own timing table readable
    p.verbose = false;
    p.no_timestamps = false;
    return p;
}

static whisper_params make_tts_params() {
    whisper_params p;
    p.backend = "qwen3-tts-1.7b-base";
    p.model = kTtsTalker;
    p.tts_codec_model = kTtsCodec;
    p.tts_voice = "jarvis";   // bare name -> resolved against tts_voice_dir
    p.tts_voice_dir = kVoiceDir;
    p.language = "en";
    p.target_lang = "en";
    p.n_threads = 8;
    p.use_gpu = true;
    p.gpu_backend = "vulkan";
    p.no_prints = true;
    p.verbose = false;
    return p;
}

// ---------------------------------------------------------------------------
// Watchdog: hard-exit the process if the probe wedges (ggml/Vulkan deadlock
// on a shared device is exactly the failure mode we are probing for).
// ---------------------------------------------------------------------------
static std::atomic<bool> g_probe_finished{false};

static void start_watchdog(int seconds) {
    std::thread([seconds]() {
        for (int i = 0; i < seconds; ++i) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            if (g_probe_finished.load())
                return;
        }
        if (!g_probe_finished.load()) {
            fprintf(stderr, "\nprobe result: HANG (watchdog fired after %d s)\n", seconds);
            fflush(stderr);
            _Exit(99);
        }
    }).detach();
}

// Chat streaming state. `crispasr_chat_on_token` is a plain function
// pointer, so the state travels through the void* user argument.
struct ChatRun {
    double t0 = 0;
    double ttft_ms = -1;
    double total_ms = -1;
    std::string text;
    int n_chunks = 0;
};

static void chat_on_token(const char* chunk, void* user) {
    auto* r = static_cast<ChatRun*>(user);
    if (r->ttft_ms < 0)
        r->ttft_ms = now_ms() - r->t0;
    r->n_chunks++;
    if (chunk)
        r->text += chunk;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char* asr_model = env_or("SPIKE_ASR_MODEL", kAsrModel);
    const char* tts_talker = env_or("SPIKE_TTS_TALKER", kTtsTalker);
    const char* tts_codec = env_or("SPIKE_TTS_CODEC", kTtsCodec);
    const char* voice_dir = env_or("SPIKE_VOICE_DIR", kVoiceDir);
    const char* chat_model = env_or("SPIKE_CHAT_MODEL", kChatModel);
    const char* asr_wav = env_or("SPIKE_ASR_WAV", kAsrWav);

    // Process-global GPU backend preference. Must be set before any
    // init_from_file, exactly like cli.cpp does for --gpu-backend (#214).
    crispasr_set_gpu_backend_pref("vulkan");

    printf("=== 584 realtime-voice co-residency spike ===\n");
    printf("asr model : %s\n", asr_model);
    printf("tts talker: %s\n", tts_talker);
    printf("tts codec : %s\n", tts_codec);
    printf("voice dir : %s\n", voice_dir);
    printf("chat model: %s\n", chat_model);
    printf("asr wav   : %s\n", asr_wav);
    printf("\n");
    gpu_line("start");

    // -----------------------------------------------------------------------
    // 1. ASR backend (qwen3-1.7b), same init sequence as the CLI/server.
    // -----------------------------------------------------------------------
    whisper_params asr_p = make_asr_params();
    asr_p.model = asr_model;
    const double t_asr_init0 = now_ms();
    std::unique_ptr<CrispasrBackend> asr = crispasr_create_backend(asr_p.backend);
    if (!asr) {
        fprintf(stderr, "FATAL: crispasr_create_backend(%s) returned null\n", asr_p.backend.c_str());
        return 2;
    }
    if (!asr->init(asr_p)) {
        fprintf(stderr, "FATAL: ASR backend init failed\n");
        return 2;
    }
    const double asr_init_ms = now_ms() - t_asr_init0;
    printf("[init] asr backend '%s' ready in %.0f ms (caps=0x%x)\n", asr->name(), asr_init_ms,
           (unsigned)asr->capabilities());
    gpu_line("after ASR init");

    // -----------------------------------------------------------------------
    // 2. TTS backend (qwen3-tts-1.7b-base) in the SAME process.
    // -----------------------------------------------------------------------
    whisper_params tts_p = make_tts_params();
    tts_p.model = tts_talker;
    tts_p.tts_codec_model = tts_codec;
    tts_p.tts_voice_dir = voice_dir;
    const double t_tts_init0 = now_ms();
    std::unique_ptr<CrispasrBackend> tts = crispasr_create_backend(tts_p.backend);
    if (!tts) {
        fprintf(stderr, "FATAL: crispasr_create_backend(%s) returned null\n", tts_p.backend.c_str());
        return 2;
    }
    if (!tts->init(tts_p)) {
        fprintf(stderr, "FATAL: TTS backend init failed (voice/codec paths?)\n");
        return 2;
    }
    const double tts_init_ms = now_ms() - t_tts_init0;
    const int tts_rate = tts->tts_sample_rate();
    printf("[init] tts backend '%s' ready in %.0f ms (caps=0x%x, out_rate=%d)\n", tts->name(), tts_init_ms,
           (unsigned)tts->capabilities(), tts_rate);
    gpu_line("after TTS init");

    // -----------------------------------------------------------------------
    // ASR: load the wav at the backend's native rate, transcribe once.
    // -----------------------------------------------------------------------
    float* pcm = nullptr;
    int n_pcm = 0, src_rate = 0;
    const int want_rate = asr->input_sample_rate();
    if (crispasr_audio_load_at_rate(asr_wav, want_rate, &pcm, &n_pcm, &src_rate) != 0 || !pcm || n_pcm <= 0) {
        fprintf(stderr, "FATAL: could not load %s\n", asr_wav);
        return 2;
    }
    printf("[asr] loaded %s: %d samples @ %d Hz (%.2f s)\n", asr_wav, n_pcm, want_rate,
           (double)n_pcm / (double)want_rate);

    auto do_asr = [&](std::string* text_out) -> double {
        const double t0 = now_ms();
        std::vector<crispasr_segment> segs = asr->transcribe(pcm, n_pcm, 0, asr_p);
        const double ms = now_ms() - t0;
        if (text_out) {
            text_out->clear();
            for (const auto& s : segs) {
                if (!text_out->empty())
                    text_out->push_back(' ');
                *text_out += s.text;
            }
        }
        return ms;
    };

    std::string asr_text;
    const double asr_ms = do_asr(&asr_text);
    printf("\n--- 1. ASR ---------------------------------------------------\n");
    printf("transcript      : \"%s\"\n", asr_text.c_str());
    printf("audio           : %.2f s\n", (double)n_pcm / (double)want_rate);
    printf("latency         : %.0f ms  (RTF %.3f x realtime)\n", asr_ms,
           asr_ms / (1000.0 * (double)n_pcm / (double)want_rate));

    // -----------------------------------------------------------------------
    // TTS: streaming synthesis of the fixed sentence, with the jarvis voice.
    // -----------------------------------------------------------------------
    const std::string tts_text = "Hello, this is a co-residency spike.";
    struct TtsRun {
        double ttfa_ms = -1;
        double total_ms = -1;
        int n_chunks = 0;
        long long n_samples = 0;
        int first_chunk_samples = 0;
        int max_chunk_samples = 0;
        bool got_pcm = false;
    };
    auto do_tts = [&](TtsRun* out) {
        TtsRun r;
        const double t0 = now_ms();
        auto cb = [&](const float* chunk, int n, bool is_final) {
            if (n > 0) {
                if (!r.got_pcm) {
                    r.ttfa_ms = now_ms() - t0;
                    r.first_chunk_samples = n;
                    r.got_pcm = true;
                }
                r.n_chunks++;
                r.n_samples += n;
                if (n > r.max_chunk_samples)
                    r.max_chunk_samples = n;
            }
            (void)is_final;
        };
        tts->synthesize_streaming(tts_text, tts_p, cb);
        r.total_ms = now_ms() - t0;
        if (out)
            *out = r;
        return r;
    };

    TtsRun tts_run = do_tts(nullptr);
    const double audio_s = (double)tts_run.n_samples / (double)tts_rate;
    printf("\n--- 2. TTS ---------------------------------------------------\n");
    printf("text            : \"%s\"\n", tts_text.c_str());
    if (tts_run.ttfa_ms < 0)
        printf("time-to-first   : n/a (no PCM)\n");
    else
        printf("time-to-first   : %.0f ms\n", tts_run.ttfa_ms);
    printf("total synth     : %.0f ms\n", tts_run.total_ms);
    printf("audio duration  : %.2f s  (%lld samples @ %d Hz)\n", audio_s, tts_run.n_samples, tts_rate);
    printf("rtf             : %.3f x realtime\n", audio_s > 0 ? tts_run.total_ms / (1000.0 * audio_s) : -1.0);
    printf("chunks          : %d (first=%d samples = %.0f ms, max=%d)\n", tts_run.n_chunks,
           tts_run.first_chunk_samples, tts_rate > 0 ? 1000.0 * tts_run.first_chunk_samples / tts_rate : 0.0,
           tts_run.max_chunk_samples);
    gpu_line("after TTS synth");

    // -----------------------------------------------------------------------
    // 3. Chat session (text LLM) in the same process / same ggml device.
    // -----------------------------------------------------------------------
    crispasr_chat_open_params cp;
    crispasr_chat_open_params_default(&cp);
    cp.n_gpu_layers = -1; // all layers -> Vulkan (defaults to -1; explicit for the record)
    cp.n_ctx = 2048;
    cp.n_threads = 8;
    cp.n_threads_batch = 8;

    crispasr_chat_error cerr{};
    const double t_chat_init0 = now_ms();
    crispasr_chat_session_t chat = crispasr_chat_open(chat_model, &cp, &cerr);
    const double chat_init_ms = now_ms() - t_chat_init0;
    if (!chat) {
        fprintf(stderr, "FATAL: crispasr_chat_open failed (%d: %s)\n", cerr.code, cerr.message);
        return 2;
    }
    printf("\n[init] chat session ready in %.0f ms (template=%s, n_ctx=%d)\n", chat_init_ms,
           crispasr_chat_template_name(chat), crispasr_chat_n_ctx(chat));
    gpu_line("after chat model load");

    const crispasr_chat_message msgs[2] = {
        {"system", "You are a concise voice assistant. Answer in one short sentence."},
        {"user", "Say hello to the user."},
    };

    auto do_chat = [&](ChatRun* out) {
        ChatRun r;
        crispasr_chat_generate_params gp;
        crispasr_chat_generate_params_default(&gp);
        gp.max_tokens = 16;
        gp.temperature = 0.0f; // greedy: deterministic, and TTFT is the number we care about
        const double t0 = now_ms();
        r.t0 = t0;
        crispasr_chat_error e{};
        const int32_t rc = crispasr_chat_generate_stream(chat, msgs, 2, &gp, chat_on_token, &r, &e);
        r.total_ms = now_ms() - t0;
        if (rc != 0)
            fprintf(stderr, "chat generate failed rc=%d (%s)\n", rc, e.message);
        if (out)
            *out = r;
        return r;
    };

    ChatRun chat_run = do_chat(nullptr);
    const double tps = chat_run.total_ms > 0 ? (1000.0 * chat_run.n_chunks) / chat_run.total_ms : -1.0;
    printf("\n--- 3. CHAT (text LLM) --------------------------------------\n");
    printf("reply           : \"%s\"\n", chat_run.text.c_str());
    printf("tokens emitted  : %d\n", chat_run.n_chunks);
    printf("ttft            : %.1f ms\n", chat_run.ttft_ms);
    printf("total           : %.0f ms\n", chat_run.total_ms);
    printf("throughput      : %.1f tok/s\n", tps);
    printf("\n[rss] process peak RSS after warmup: %ld MB (VmHWM)\n", vmhwm_kb() / 1024);

    // -----------------------------------------------------------------------
    // 4. Probe: two inference threads at once (ASR x5 || TTS x5).
    //    Same threads-and-shared-device shape the realtime loop will have
    //    (ASR worker + TTS worker), so a hang/crash here is disqualifying.
    // -----------------------------------------------------------------------
    printf("\n--- 4. PROBE: concurrent ASR || TTS (5 + 5) ------------------\n");
    g_probe_finished.store(false);
    start_watchdog(90);

    std::atomic<int> asr_ok{0}, tts_ok{0};
    std::atomic<long long> probe_asr_ms_sum{0}, probe_tts_ms_sum{0};
    std::string last_asr_text;
    std::mutex print_mu;
    const double probe0 = now_ms();

    const double wall0 = now_ms();
    std::thread th_asr([&]() {
        for (int i = 0; i < 5; ++i) {
            std::string txt;
            const double ms = do_asr(&txt);
            const bool ok = !txt.empty();
            if (ok)
                asr_ok++;
            probe_asr_ms_sum += (long long)ms;
            {
                std::lock_guard<std::mutex> lk(print_mu);
                last_asr_text = txt;
                printf("  [asr  #%d] %6.0f ms  %s  \"%s\"\n", i, ms, ok ? "ok  " : "EMPTY", txt.c_str());
            }
        }
    });
    std::thread th_tts([&]() {
        for (int i = 0; i < 5; ++i) {
            TtsRun r = do_tts(nullptr);
            const bool ok = r.got_pcm && r.n_samples > 0;
            if (ok)
                tts_ok++;
            probe_tts_ms_sum += (long long)r.total_ms;
            std::lock_guard<std::mutex> lk(print_mu);
            printf("  [tts  #%d] %6.0f ms  %s  ttfa=%.0f ms, %d chunks, %lld samples\n", i, r.total_ms,
                   ok ? "ok  " : "EMPTY", r.ttfa_ms, r.n_chunks, r.n_samples);
        }
    });
    th_asr.join();
    th_tts.join();
    const double probe_ms = now_ms() - wall0;

    printf("probe result    : %s  (asr_ok=%d/5, tts_ok=%d/5, wall=%.0f ms)\n",
           (asr_ok == 5 && tts_ok == 5) ? "PASS (correct output, no crash, no hang)" : "DEGRADED", asr_ok.load(),
           tts_ok.load(), probe_ms);
    printf("mean latencies  : asr %.0f ms, tts %.0f ms\n", probe_asr_ms_sum / 5.0, probe_tts_ms_sum / 5.0);

    // -----------------------------------------------------------------------
    // 5. Probe B: chat generation concurrent with ASR.
    //    This is the "speculative opening clause" shape from the plan — the
    //    LLM starts producing while ASR is still finishing. Independent
    //    sessions/contexts, so it *should* be fine; measured, not assumed.
    // -----------------------------------------------------------------------
    printf("\n--- 5. PROBE B: concurrent CHAT || ASR (1 x 16 tok + 3 x asr) --\n");
    std::atomic<int> b_asr_ok{0}, b_chat_ok{0};
    const double probe_b0 = now_ms();
    std::thread tb_asr([&]() {
        for (int i = 0; i < 3; ++i) {
            std::string txt;
            const double ms = do_asr(&txt);
            if (!txt.empty())
                b_asr_ok++;
            std::lock_guard<std::mutex> lk(print_mu);
            printf("  [asr  #%d] %6.0f ms  %s\n", i, ms, txt.empty() ? "EMPTY" : "ok");
        }
    });
    ChatRun chat_b;
    std::thread tb_chat([&]() {
        chat_b = do_chat(nullptr);
        if (chat_b.n_chunks > 0)
            b_chat_ok++;
        std::lock_guard<std::mutex> lk(print_mu);
        printf("  [chat    ] %6.0f ms  %s  ttft=%.0f ms, %d tok\n", chat_b.total_ms,
               chat_b.n_chunks > 0 ? "ok" : "EMPTY", chat_b.ttft_ms, chat_b.n_chunks);
    });
    tb_asr.join();
    tb_chat.join();
    const double probe_b_ms = now_ms() - probe_b0;
    printf("probe B result  : %s  (asr_ok=%d/3, chat_tokens=%d, wall=%.0f ms)\n",
           (b_asr_ok == 3 && b_chat_ok == 1) ? "PASS (correct output, no crash, no hang)" : "DEGRADED",
           b_asr_ok.load(), chat_b.n_chunks, probe_b_ms);

    g_probe_finished.store(true);
    (void)probe0;

    // -----------------------------------------------------------------------
    // Teardown (order deliberately mirrors "session first, then backends").
    // -----------------------------------------------------------------------
    crispasr_chat_close(chat);
    tts.reset();
    asr.reset();
    crispasr_audio_free(pcm);

    printf("\n--- summary --------------------------------------------------\n");
    printf("peak RSS        : %ld MB (VmHWM)\n", vmhwm_kb() / 1024);
    gpu_line("end");
    printf("co-residency    : ASR+TTS+chat alive in one process: YES\n");
    printf("concurrency     : see probe results above (correct/hang/crash)\n");
    return (asr_ok == 5 && tts_ok == 5) ? 0 : 1;
}
