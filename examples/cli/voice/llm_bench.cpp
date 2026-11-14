// llm_bench.cpp — measure the chat model the voice loop actually uses.
//
// The cases mirror the four generation sites in the loop, so a number here can
// be attributed to a call site there:
//   tool     the strict router prompt (tool_call_messages())
//   ack      the "say you are checking" prompt (sequencer on_final)
//   filler   the bare holding-phrase prompt (sequencer maybe_request_filler)
//   fillerN  the same prompt plus the expected wait, which is the tuning
//            proposal: if the model knows the check takes ~N s it can size the
//            bridge itself instead of the sequencer stacking bank phrases
//   answer   the "background result ... answer in one or two sentences" prompt
//
// The prompt texts for ack/filler/answer are copied from sequencer.cpp on
// purpose: the sequencer builds them inline from per-turn state, and pulling
// them out would mean changing the policy module for a measurement. The `tool`
// case uses the real builder, and the copied ones are marked with their source.

#include "llm_bench.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engines.h"
#include "sequencer.h"
#include "tool_call.h"

namespace crispasr {
namespace voice {
namespace {

struct Case {
    std::string name;
    std::vector<ChatMessage> messages;
    int max_tokens;
};

struct Sample {
    int64_t ttft_ms = 0;
    int64_t total_ms = 0;
    int tokens = 0;
};

double pct(std::vector<int64_t> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const size_t i = (size_t)std::llround(p * (double)(v.size() - 1));
    return (double)v[std::min(i, v.size() - 1)];
}

int64_t median(std::vector<int64_t> v) {
    if (v.empty()) return 0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// A representative turn, so prompt sizes match what the loop sends.
constexpr const char* kTranscript = "What is the status of the garage door?";
constexpr const char* kTaskResult = "The garage door is closed and locked.";

std::string ack_prompt() {
    // sequencer.cpp: on_final()
    return "The user said: \"" + std::string(kTranscript) +
           "\". Reply with one short acknowledgement that says you are checking. "
           "Do not restate the request, do not state any status or result, do not "
           "ask a question.";
}

std::string filler_prompt() {
    // sequencer.cpp: maybe_request_filler()
    return "The user is waiting. You are still checking \"" +
           std::string(kTranscript) +
           "\". Say one short, natural holding phrase that does not claim "
           "progress or a result, and does not repeat what you already said.";
}

// The proposal: the same request, with the expected delay the sequencer already
// knows from the tool profile (tool_profile(): device_status is the 6 s case).
std::string filler_prompt_with_wait(int seconds, int words) {
    return "The user is waiting. You are still checking \"" +
           std::string(kTranscript) + "\". The check takes about " +
           std::to_string(seconds) +
           " seconds, so keep the user company for roughly that long. Reply with "
           "at most " + std::to_string(words) +
           " words of natural holding speech that does not claim progress or a "
           "result and does not repeat what you already said.";
}

std::string answer_prompt() {
    // sequencer.cpp: on_task_result()
    return "Conversation so far: user said \"" + std::string(kTranscript) +
           "\". You already told them: \"One moment, let me check that.\". "
           "Background result: \"" + std::string(kTaskResult) +
           "\". Answer in one or two short spoken sentences, continuing naturally "
           "from what you already said.";
}

std::vector<Case> build_cases(const Config& cfg) {
    std::vector<Case> cases;
    const int wait_s = std::max(1, cfg.task_ms / 1000);
    // ~325 ms/word for the English qwen3-tts voice (sequencer kMsPerWord).
    const int wait_words = std::max(4, (int)((int64_t)cfg.task_ms / 325));

    // The static block the loop puts in every text request's system prompt. It
    // is identical for the whole session, so on the second and later calls it
    // should come out of the chat session's cached prefix (crispasr_chat.h):
    // the `guidance` case measures the shape the loop actually sends.
    const std::string guidance = bridge_guidance_text(cfg);

    cases.push_back({"tool", tool_call_messages(kTranscript), kToolCallMaxTokens});
    cases.push_back({"ack", {{{"user", ack_prompt()}}}, 32});
    cases.push_back({"filler", {{{"user", filler_prompt()}}}, 32});
    cases.push_back({"filler+" + std::to_string(wait_s) + "s",
                     {{"user", filler_prompt_with_wait(wait_s, wait_words)}}, 64});
    cases.push_back({"answer", {{"user", answer_prompt()}}, 96});
    cases.push_back(
        {"guidance", {{"system", guidance}, {"user", filler_prompt()}}, 32});
    return cases;
}

}  // namespace

int run_llm_bench(const Config& cfg_in) {
    Config cfg = cfg_in;
    cfg.normalise();
    if (cfg.chat_model.empty()) {
        std::fprintf(stderr, "bench: --chat-model is required\n");
        return 2;
    }

    LlmEngine* llm = make_crispasr_llm(cfg);
    if (llm == nullptr || !llm->ready()) {
        std::fprintf(stderr, "bench: LLM not ready (%s)\n",
                     llm ? "load failed" : "init failed");
        delete llm;
        return 1;
    }

    const int reps = cfg.runs > 0 ? cfg.runs : 10;
    const std::vector<Case> cases = build_cases(cfg);

    std::printf("llm bench: model=%s ctx=%d reps=%d (first rep of each case is "
                "warm-up, not measured)\n\n",
                cfg.chat_model.c_str(), cfg.chat_ctx, reps);
    std::printf("%-12s %6s %8s %8s %8s %8s %8s\n", "case", "chars", "ttft_p50",
                "ttft_p95", "tokens", "decode_ms", "tok/s");

    int failures = 0;
    for (const Case& c : cases) {
        std::vector<int64_t> ttfts, totals, decodes;
        std::vector<int64_t> toks;
        int chars = 0;
        for (const ChatMessage& m : c.messages) chars += (int)m.content.size();

        for (int i = 0; i < reps + 1; ++i) {
            LlmResult r = llm->generate(c.messages, c.max_tokens,
                                        [](const std::string&) { return true; });
            if (!r.ok) {
                ++failures;
                std::fprintf(stderr, "bench: case '%s' rep %d produced no text\n",
                             c.name.c_str(), i);
                break;
            }
            if (i == 0) continue;  // warm-up
            ttfts.push_back(r.ttft_ms);
            totals.push_back(r.total_ms);
            decodes.push_back(std::max<int64_t>(0, r.total_ms - r.ttft_ms));
            toks.push_back(r.tokens);
        }
        if (ttfts.empty()) continue;

        const int64_t decode_ms = median(decodes);
        const int64_t tokens = median(toks);
        const double tps =
            decode_ms > 0 ? (double)(tokens - 1) * 1000.0 / (double)decode_ms : 0.0;
        std::printf("%-12s %6d %8.0f %8.0f %8lld %8lld %8.1f\n", c.name.c_str(),
                    chars, pct(ttfts, 0.5), pct(ttfts, 0.95), (long long)tokens,
                    (long long)decode_ms, tps);
    }

    delete llm;
    std::printf("\n%s: %d case(s) failed\n", failures ? "FAIL" : "OK", failures);
    return failures == 0 ? 0 : 1;
}

// ---------------------------------------------------------------------------
// Bridge-quality eval
//
// The loop's problem, stated as a measurement: when the LLM is asked for a
// holding phrase while a 6 s check runs, does it produce something neutral, or
// does it invent the answer? Everything deterministic (the guard, the bank
// fallback) is out of the loop here: the number has to describe the model.
//
// Each variant is a prompt shape; each is swept over temperatures and seeds so a
// lucky greedy sample cannot pass for a policy. "clean" = the text names no
// device and states no condition (the same predicate the loop's guard uses, but
// used as a score here, never as a filter).
// ---------------------------------------------------------------------------
namespace {

struct Variant {
    std::string name;
    std::string system;   // '' = no system message
    std::string user;
};

// Local word count: sequencer.cpp's is internal to that module and the eval
// needs its own copy rather than a header change for a measurement.
int count_words(const std::string& s) {
    int n = 0;
    bool in = false;
    for (char c : s) {
        const bool sep = std::isspace((unsigned char)c) != 0;
        if (!sep && !in) ++n;
        in = !sep;
    }
    return n;
}

// Prompt shapes, weakest to strongest constraint, all for the same 6 s device
// check on the same transcript.
std::vector<Variant> bridge_variants(const Config& cfg) {
    const std::string t = kTranscript;
    const std::string sys_base =
        "You are a voice assistant speaking to the user. Reply in first person, "
        "no quotation marks, no markdown.";
    std::vector<Variant> v;

    v.push_back({"v1-bare", "",
                 "The user is waiting. You are still checking \"" + t +
                     "\". Say one short, natural holding phrase that does not "
                     "claim progress or a result."});

    v.push_back({"v2-system",
                 sys_base +
                     " While a check runs, describe what you are doing without "
                     "naming the device or guessing its state.",
                 "Say one short holding phrase. The check is still running."});

    v.push_back({"v3-budget",
                 sys_base + bridge_guidance_text(cfg),
                 "Say one natural holding phrase. The check is still running. "
                 "About 6 seconds of the wait remain."});

    // Few-shot: show the shape instead of describing it.
    const std::string fewshot = bridge_guidance_text(cfg) +
                                " Good holding phrases: \"Let me take a proper "
                                "look at that.\" / \"I'm checking on that now, "
                                "one moment.\" / \"Give me a few seconds to "
                                "confirm.\" A holding phrase never names the "
                                "device, never says open or closed, never "
                                "states a result.";
    v.push_back({"v4-fewshot", sys_base + fewshot,
                 "Say one natural holding phrase. The check is still running. "
                 "About 6 seconds of the wait remain."});

    // Hardest: an explicit negative list plus a shape the model can copy.
    v.push_back(
        {"v5-negative",
         sys_base +
             " Never use the words garage, door, light, lights, thermostat, "
             "open, closed, locked, on, off, active, ready, or any device state. "
             "Talk only about your own action of checking.",
         "Say one sentence (max 12 words) about checking. Example shape: "
         "\"Let me take a look at that for you.\""});

    // v6/v7: the few-shot system block with the *production* user prompt, which
    // carries the transcript. v1 (transcript, no system) failed at 0% while v4
    // (clean user turn, few-shot) passed, so the two changes were conflated:
    // these variants isolate "does seeing the transcript reintroduce the leak".
    const std::string prod_user =
        "The user is waiting. You are still checking \"" + t +
        "\". Say one short, natural holding phrase that does not claim progress "
        "or a result, and does not repeat what you already said. About 6 seconds "
        "of the wait remain.";
    v.push_back({"v6-prod", sys_base + fewshot, prod_user});
    // Same again with the negative list added for good measure.
    v.push_back(
        {"v7-prod+neg",
         sys_base + fewshot +
             " Never use the words garage, door, light, lights, thermostat, "
             "open, closed, locked, active, ready.",
         prod_user});

    return v;
}

struct Score {
    int n = 0;
    int clean = 0;
    std::vector<int> words;
    std::vector<int64_t> ttfts;
    std::string example;
};

}  // namespace

int run_bridge_eval(const Config& cfg_in) {
    Config cfg = cfg_in;
    cfg.normalise();
    if (cfg.chat_model.empty()) {
        std::fprintf(stderr, "eval: --chat-model is required\n");
        return 2;
    }
    LlmEngine* llm = make_crispasr_llm(cfg);
    if (llm == nullptr || !llm->ready()) {
        std::fprintf(stderr, "eval: LLM not ready\n");
        delete llm;
        return 1;
    }

    const int samples = cfg.runs > 0 ? cfg.runs : 10;
    const std::vector<Variant> variants = bridge_variants(cfg);
    const std::vector<float> temps = {0.0f, 0.4f, 0.8f};

    std::printf("bridge eval: model=%s samples/variant=%d\n",
                cfg.chat_model.c_str(), samples);
    std::printf("clean = no device noun, no condition word, no digit in the text\n\n");
    std::printf("%-13s %5s %7s %7s %8s  %s\n", "variant", "temp", "clean%",
                "words", "ttft_p50", "sample");

    for (const Variant& v : variants) {
        for (float temp : temps) {
            Config c = cfg;
            c.chat_temperature = temp;
            c.chat_seed = -1;  // vary the seed: a policy, not a lucky sample
            LlmEngine* eng = make_crispasr_llm(c);
            if (eng == nullptr || !eng->ready()) {
                std::fprintf(stderr, "eval: cannot open model at t=%.2f\n", temp);
                delete eng;
                continue;
            }
            Score s;
            for (int i = 0; i < samples; ++i) {
                std::vector<ChatMessage> msgs;
                if (!v.system.empty()) msgs.push_back({"system", v.system});
                msgs.push_back({"user", v.user});
                LlmResult r = eng->generate(msgs, 48,
                                            [](const std::string&) { return true; });
                if (!r.ok) continue;
                ++s.n;
                s.ttfts.push_back(r.ttft_ms);
                s.words.push_back(count_words(r.text));
                const bool bad = filler_claims_result(r.text) ||
                                 r.text.find_first_of("0123456789") != std::string::npos;
                if (!bad) ++s.clean;
                if (s.example.empty() || bad == false) s.example = r.text;
            }
            if (s.n == 0) {
                delete eng;
                continue;
            }
            std::sort(s.words.begin(), s.words.end());
            const int med_words = s.words[s.words.size() / 2];
            std::string ex = s.example;
            for (char& ch : ex)
                if (ch == '\n' || ch == '\r' || ch == '\t') ch = ' ';
            if (ex.size() > 46) ex = ex.substr(0, 46) + "...";
            std::printf("%-13s %5.2f %6.0f%% %7d %8.0f  %s\n", v.name.c_str(),
                        temp, 100.0 * (double)s.clean / (double)s.n, med_words,
                        pct(s.ttfts, 0.5), ex.c_str());
            delete eng;
        }
    }

    delete llm;
    return 0;
}

// ---------------------------------------------------------------------------
// Context-cost bench
//
// The question: the system prompt and instruction block are static, so they sit
// in the cached prefix and cost nothing after the first call. A live home
// snapshot (sensor states) and the user's transcript are dynamic. What does the
// dynamic part cost in TTFT, and where does the context window run out?
//
// Three measurements per size, because they answer three different questions:
//   cold    -> what a session pays the first time (prefill of everything)
//   warm    -> identical prompt again: is the prefix actually reused?
//   changed -> same block, one sensor value perturbed: the per-turn cost of a
//              fresh snapshot in front of an unchanged system prompt
// ---------------------------------------------------------------------------
namespace {

int64_t now_ms_bench() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

std::string slurp(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return std::string();
    std::string out;
    char buf[4096];
    size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) out.append(buf, n);
    std::fclose(f);
    return out;
}

}  // namespace

int run_context_bench(const Config& cfg_in, const std::string& context_file) {
    Config cfg = cfg_in;
    cfg.normalise();
    if (cfg.chat_model.empty() || context_file.empty()) {
        std::fprintf(stderr, "context bench: --chat-model and --context-file are required\n");
        return 2;
    }
    const std::string full = slurp(context_file);
    if (full.empty()) {
        std::fprintf(stderr, "context bench: cannot read %s\n", context_file.c_str());
        return 2;
    }

    // Sizes in characters of the dynamic block. 0 = no context at all.
    const std::vector<size_t> sizes = {0, 500, 1500, 3000, 4000, 4800, 5500, 6500,
                                    12000, full.size()};
    const std::string question = "Make the living room warmer.";

    std::printf("context bench: model=%s ctx=%d block=%s (%zu chars total)\n",
                cfg.chat_model.c_str(), cfg.chat_ctx, context_file.c_str(),
                full.size());
    std::printf("static system prompt is the same for every row; ~tokens = chars/4\n\n");
    std::printf("%10s %8s %9s %9s %9s %9s\n", "block_chr", "~tokens", "cold_ms",
                "warm_ms", "tail_ms", "fresh_ms");

    for (size_t size : sizes) {
        const std::string block = full.substr(0, size);
        const std::string sys =
            std::string("You are a voice assistant with access to the home state. ") +
            bridge_guidance_text(cfg);
        auto messages_for = [&](const std::string& ctx) {
            std::vector<ChatMessage> m;
            m.push_back({"system", sys});
            if (!ctx.empty()) m.push_back({"user", "Home state:\n" + ctx});
            m.push_back({"user", question});
            return m;
        };

        LlmEngine* eng = make_crispasr_llm(cfg);
        if (eng == nullptr || !eng->ready()) {
            std::fprintf(stderr, "context bench: model failed to load\n");
            delete eng;
            return 1;
        }
        auto time_one = [&](const std::vector<ChatMessage>& msgs, int64_t* out_tokens) {
            const int64_t t0 = now_ms_bench();
            LlmResult r = eng->generate(msgs, 24, [](const std::string&) { return true; });
            const int64_t wall = now_ms_bench() - t0;
            if (out_tokens) *out_tokens = r.tokens;
            // TTFT is the number that matters; wall time equals it plus decode.
            return r.ok ? r.ttft_ms : -1;
        };

        int64_t out_tok = 0;
        const int64_t cold = time_one(messages_for(block), &out_tok);
        if (cold < 0) {
            // Context does not fit: that is the answer for this size, not an
            // error to hide.
            std::printf("%10zu %8zu %9s %9s %9s %9s\n", size, size / 4, "too-big",
                        "-", "-", "-");
            delete eng;
            continue;
        }
        // Two kinds of change, because they answer different questions:
        //   tail  - one char at the very end: the cached prefix survives, so this
        //           is the floor (re-decode of the last token + question).
        //   fresh - every line's value perturbed: a real snapshot, where the
        //           cache breaks at the FIRST differing token and the whole
        //           block plus the question has to be prefilled again.
        std::string tail = block;
        if (!tail.empty()) tail.back() = tail.back() == '1' ? '2' : '1';
        std::string fresh = block;
        for (size_t i = 0; i < fresh.size(); ++i) {
            if (fresh[i] == '\n' && i > 0) fresh[i - 1] = '9';
        }
        if (!fresh.empty()) fresh.back() = '7';

        // First-call numbers are single-shot (a session only has one first
        // call); the repeated ones are medians, because prefill on a shared
        // device has a tail and a single sample picked the wrong story once.
        std::vector<int64_t> warms, tails, freshs;
        const int reps = cfg.runs > 0 ? cfg.runs : 5;
        for (int i = 0; i < reps; ++i) {
            // The warm number must come from two CONSECUTIVE identical calls.
            // Interleaving the perturbed ones here was measured wrong: the warm
            // call then followed a fresh one, so its prefix was gone and it
            // looked like 94 ms instead of 12 ms.
            (void)time_one(messages_for(block), nullptr);  // clobber the cache
            warms.push_back(time_one(messages_for(block), nullptr));
            tails.push_back(time_one(messages_for(tail), nullptr));
            freshs.push_back(time_one(messages_for(fresh), nullptr));
        }

        std::printf("%10zu %8zu %9lld %9lld %9lld %9lld\n", size, size / 4,
                    (long long)cold, (long long)median(warms),
                    (long long)median(tails), (long long)median(freshs));
        delete eng;
    }
    return 0;
}


// ---------------------------------------------------------------------------
// Prefill pipelining bench
//
// The home snapshot does not depend on the STT result. Speech onset is therefore
// enough to start prefilling it, which puts the expensive, transcript-independent
// part of the prompt in the KV cache while the user is still talking. Three
// strategies are compared on the number the user feels (TTFT of the call that
// actually generates):
//   base    - one call with [system, context, question] on a cold session
//   prefetch- prefill([system, context]) at speech onset, then the same generate
//   partial - prefill([system, context, partial transcript]), then generate with
//             the final transcript: does the partial survive as a prefix?
// The prefill itself is measured too, but it is off the critical path.
// ---------------------------------------------------------------------------
int run_prefill_bench(const Config& cfg_in, const std::string& context_file) {
    Config cfg = cfg_in;
    cfg.normalise();
    if (cfg.chat_model.empty() || context_file.empty()) {
        std::fprintf(stderr, "prefill bench: --chat-model and --context-file are required\n");
        return 2;
    }
    const std::string full = slurp(context_file);
    if (full.empty()) {
        std::fprintf(stderr, "prefill bench: cannot read %s\n", context_file.c_str());
        return 2;
    }
    const std::string question = "Make the living room warmer.";
    // A plausible partial: the user has said most of the sentence. The last word
    // differs from the final, which is what costs the prefix.
    const std::string partial = "Make the living room";

    const std::vector<size_t> sizes = {0, 757, 3000, 6500, 12000, full.size()};
    const int reps = cfg.runs > 0 ? cfg.runs : 5;

    std::printf("prefill bench: model=%s ctx=%d block=%s (%zu chars)\n",
                cfg.chat_model.c_str(), cfg.chat_ctx, context_file.c_str(), full.size());
    std::printf("all times in ms; TTFT is the visible number, prefill is off the critical path\n\n");
    std::printf("%9s %10s %9s %11s %11s %9s\n", "block_chr", "prefill_ms", "base_ttft",
                "prefetch", "partial", "saved");

    for (size_t size : sizes) {
        const std::string ctx_block = full.substr(0, std::min(size, full.size()));
        const std::string sys = std::string("You are a voice assistant. ") +
                                bridge_guidance_text(cfg);
        auto msgs_base = [&]() {
            std::vector<ChatMessage> m{{"system", sys}};
            if (!ctx_block.empty()) m.push_back({"user", "Home state:\n" + ctx_block});
            m.push_back({"user", question});
            return m;
        };

        std::vector<int64_t> base, pipe, spec, pre;
        bool too_big = false;
        for (int i = 0; i < reps && !too_big; ++i) {
            // base: cold session, one call, everything on the critical path.
            {
                LlmEngine* e1 = make_crispasr_llm(cfg);
                if (!e1 || !e1->ready()) { delete e1; return 1; }
                LlmResult r = e1->generate(msgs_base(), 24, [](const std::string&) { return true; });
                if (!r.ok) too_big = true; else base.push_back(r.ttft_ms);
                delete e1;
            }
            if (too_big) break;
            // prefetch: the context is decoded at speech onset.
            {
                LlmEngine* e2 = make_crispasr_llm(cfg);
                if (!e2 || !e2->ready()) { delete e2; return 1; }
                std::vector<ChatMessage> ctx_msgs{{"system", sys}};
                if (!ctx_block.empty())
                    ctx_msgs.push_back({"user", "Home state:\n" + ctx_block});
                const int64_t t0 = now_ms_bench();
                e2->prefill(ctx_msgs);
                pre.push_back(now_ms_bench() - t0);
                LlmResult r = e2->generate(msgs_base(), 24, [](const std::string&) { return true; });
                if (r.ok) pipe.push_back(r.ttft_ms);
                delete e2;
            }
            // partial: the same, but the prefill already contains a partial
            // transcript that differs from the final one.
            {
                LlmEngine* e3 = make_crispasr_llm(cfg);
                if (!e3 || !e3->ready()) { delete e3; return 1; }
                std::vector<ChatMessage> ctx_msgs{{"system", sys}};
                if (!ctx_block.empty())
                    ctx_msgs.push_back({"user", "Home state:\n" + ctx_block});
                ctx_msgs.push_back({"user", partial});
                e3->prefill(ctx_msgs);
                LlmResult r = e3->generate(msgs_base(), 24, [](const std::string&) { return true; });
                if (r.ok) spec.push_back(r.ttft_ms);
                delete e3;
            }
        }
        if (too_big || base.empty()) {
            std::printf("%9zu %10s %9s %11s %11s %9s\n", size, "-", "too-big", "-", "-", "-");
            continue;
        }
        std::printf("%9zu %10lld %9lld %11lld %11lld %9lld\n", size,
                    (long long)median(pre), (long long)median(base),
                    (long long)median(pipe), (long long)median(spec),
                    (long long)(median(base) - median(pipe)));
    }
    return 0;
}

}  // namespace voice
}  // namespace crispasr
