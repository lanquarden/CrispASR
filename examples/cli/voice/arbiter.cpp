// arbiter.cpp — one inference worker, one priority queue.
//
// Every backend shares the process-wide ggml device, so exactly one thread calls
// into them. Jobs are non-preemptible, therefore the queue is only re-evaluated
// at job boundaries and each job class must be bounded by its *caller* (short
// TTS pieces, short ASR windows). The arbiter's contribution is to measure how
// long each class actually blocked, so a violation is visible in the metrics.

#include "arbiter.h"

#include "voice_types.h"

#include <algorithm>
#include <thread>
#include <unordered_map>
#include <utility>

namespace crispasr {
namespace voice {

namespace {

// arbiter.h declares no std::thread member, but ~InferenceArbiter() must join
// the worker (a detached worker could touch a destroyed queue). The handle is
// therefore kept in a process-wide registry keyed by the arbiter instance. This
// is a workaround, not a design: the header wants a `std::thread worker_;`.
std::mutex& registry_mu() {
    static std::mutex mu;
    return mu;
}

std::unordered_map<const void*, std::thread>& registry() {
    static std::unordered_map<const void*, std::thread> map;
    return map;
}

bool is_tts(Priority p) { return p == Priority::TtsUrgent || p == Priority::TtsIdle; }
bool is_stt(Priority p) { return p == Priority::SttFinal || p == Priority::SttPartial; }

}  // namespace

InferenceArbiter::~InferenceArbiter() { stop(); }

void InferenceArbiter::start() {
    if (running_.exchange(true)) return;  // already running

    // Reap a stale handle from a previous start()/stop() cycle *outside* the
    // registry lock: joining while holding it could deadlock against a worker
    // that is calling stop() from inside a job.
    std::thread stale;
    {
        std::lock_guard<std::mutex> lk(registry_mu());
        auto& map = registry();
        auto it = map.find(this);
        if (it != map.end()) {
            stale = std::move(it->second);
            map.erase(it);
        }
    }
    if (stale.joinable()) {
        if (stale.get_id() == std::this_thread::get_id()) {
            stale.detach();
        } else {
            stale.join();
        }
    }

    std::lock_guard<std::mutex> lk(registry_mu());
    registry().emplace(this, std::thread(&InferenceArbiter::run, this));
}

void InferenceArbiter::stop() {
    running_.store(false);
    cv_.notify_all();

    std::thread worker;
    {
        std::lock_guard<std::mutex> lk(registry_mu());
        auto& map = registry();
        auto it = map.find(this);
        if (it != map.end()) {
            worker = std::move(it->second);
            map.erase(it);
        }
    }
    // The worker exits at the next job boundary: an in-flight inference is never
    // interrupted (the backends have no abort hook that is safe mid-graph).
    if (worker.joinable()) {
        if (worker.get_id() == std::this_thread::get_id()) {
            // stop() called from inside a job: joining ourselves would terminate.
            // The loop still observes running_ == false and exits after this job.
            worker.detach();
        } else {
            worker.join();
        }
    }

    // Idempotent: a second stop() has no registry entry and running_ is already
    // false, so it returns immediately.
}

void InferenceArbiter::post(Priority p, std::string label, std::function<void()> fn) {
    Job job;
    job.priority = p;
    job.label = std::move(label);
    job.fn = std::move(fn);
    {
        std::lock_guard<std::mutex> lk(mu_);
        job.seq = seq_++;
        q_.push(std::move(job));
    }
    cv_.notify_one();
}

void InferenceArbiter::drop_queued_partials() {
    std::lock_guard<std::mutex> lk(mu_);
    if (q_.empty()) return;

    std::vector<Job> keep;
    keep.reserve(q_.size());
    int dropped = 0;
    while (!q_.empty()) {
        if (q_.top().priority == Priority::SttPartial) {
            q_.pop();
            ++dropped;
        } else {
            keep.push_back(q_.top());  // copy: the heap entry dies on pop()
            q_.pop();
        }
    }
    // Compare is a total order (priority, then seq), so re-inserting the
    // survivors restores exactly the original order.
    for (Job& j : keep) q_.push(std::move(j));

    stats_.dropped_partials += dropped;
}

void InferenceArbiter::run() {
    for (;;) {
        Job job;
        {
            std::unique_lock<std::mutex> lk(mu_);
            cv_.wait(lk, [this] { return !running_.load() || !q_.empty(); });
            // "Drains nothing": on stop the queue is abandoned, not executed, so
            // shutdown latency stays bounded by one job.
            if (!running_.load()) return;
            if (q_.empty()) continue;
            job = q_.top();  // copy: no const_cast, fn/label are cheap here
            q_.pop();
        }

        const int64_t t0 = now_ms();
        busy_.store(true);
        try {
            if (job.fn) job.fn();
        } catch (...) {
            // A throwing job must not kill the single worker: the whole loop
            // would go silent for the rest of the session.
        }
        busy_.store(false);
        const int64_t block_ms = now_ms() - t0;

        std::lock_guard<std::mutex> lk(mu_);
        stats_.jobs += 1;
        stats_.last_block_ms = block_ms;
        if (is_tts(job.priority)) {
            stats_.tts_block_max_ms = std::max(stats_.tts_block_max_ms, block_ms);
        } else if (is_stt(job.priority)) {
            stats_.stt_block_max_ms = std::max(stats_.stt_block_max_ms, block_ms);
        } else {
            stats_.llm_block_max_ms = std::max(stats_.llm_block_max_ms, block_ms);
        }
    }
}

}  // namespace voice
}  // namespace crispasr
