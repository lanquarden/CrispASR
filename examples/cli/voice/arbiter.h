// arbiter.h — one inference worker for all three models.
//
// All backends share the process-wide ggml Vulkan device. One worker keeps GPU
// memory and ordering deterministic; jobs are bounded so a low-priority job
// cannot block an urgent one for long.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

namespace crispasr {
namespace voice {

enum class Priority {
    TtsUrgent = 0,   // playback below low-water
    SttFinal = 1,    // turn commit, blocks the ack
    Llm = 2,         // ack / filler / answer clause
    TtsIdle = 3,     // pre-render filler at startup, buffer already healthy
    SttPartial = 4,  // advisory, droppable
};

struct Job {
    Priority priority = Priority::Llm;
    uint64_t seq = 0;
    std::string label;
    std::function<void()> fn;
};

struct ArbiterStats {
    int64_t last_block_ms = 0;
    int64_t tts_block_max_ms = 0;
    int64_t stt_block_max_ms = 0;
    int64_t llm_block_max_ms = 0;
    int dropped_partials = 0;
    int jobs = 0;
};

class InferenceArbiter {
   public:
    InferenceArbiter() = default;
    ~InferenceArbiter();

    void start();
    void stop();  // drains nothing; workers exit at the next job boundary

    void post(Priority p, std::string label, std::function<void()> fn);
    // Drops queued SttPartial jobs (a newer partial supersedes them).
    void drop_queued_partials();

    const ArbiterStats& stats() const { return stats_; }
    bool busy() const { return busy_; }

   private:
    struct Compare {
        bool operator()(const Job& a, const Job& b) const {
            if (a.priority != b.priority) return (int)a.priority > (int)b.priority;
            return a.seq > b.seq;
        }
    };

    void run();

    std::priority_queue<Job, std::vector<Job>, Compare> q_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    std::atomic<bool> busy_{false};
    uint64_t seq_ = 0;
    ArbiterStats stats_;
};

}  // namespace voice
}  // namespace crispasr
