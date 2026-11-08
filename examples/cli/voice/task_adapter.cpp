// task_adapter.cpp — simulated delegation with the shape of a real A2A call.
//
// Invariants the sequencer depends on:
//   * dispatch() never blocks and never invokes the callback before returning
//     is *not* promised — the callback may arrive on the task thread at any
//     time, including immediately for zero-latency tasks.
//   * The callback fires exactly once per task: task completion or explicit
//     cancellation, never both, never twice.
//   * A timeout is not a cancellation; only cancel()/shutdown() set the flag.
//
// Ownership: every Entry stays joinable in tasks_ until shutdown() joins it, so
// a cancel() racing completion can never observe a destroyed thread object. An
// Entry whose callback has already fired has an empty cb, which is what makes a
// late cancel() a no-op instead of a duplicate outcome.

#include "task_adapter.h"

#include "voice_types.h"

#include <algorithm>
#include <chrono>
#include <thread>
#include <utility>

namespace crispasr {
namespace voice {

namespace {

// Deterministic hash so a given task id always gets the same jitter and the
// harness timeline is reproducible run to run. Randomness here would make
// metrics incomparable between runs, which is the whole point of the harness.
uint32_t mix32(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

int64_t delay_ms_for(int id, int latency_ms, int jitter_ms) {
    const int64_t base = latency_ms > 0 ? latency_ms : 0;
    if (jitter_ms <= 0) return base;
    const uint32_t span = (uint32_t)(2 * jitter_ms + 1);
    const int64_t off = (int64_t)(mix32((uint32_t)id) % span) - (int64_t)jitter_ms;
    const int64_t d = base + off;
    return d < 0 ? 0 : d;
}

bool fails_for(int id, int fail_percent) {
    if (fail_percent <= 0) return false;
    if (fail_percent >= 100) return true;
    return (int)(mix32((uint32_t)id ^ 0x9e3779b9u) % 100u) < fail_percent;
}

}  // namespace

SimulatedTaskAdapter::SimulatedTaskAdapter(int latency_ms, int jitter_ms, int fail_percent)
    : latency_ms_(latency_ms < 0 ? 0 : latency_ms),
      jitter_ms_(jitter_ms < 0 ? 0 : jitter_ms),
      fail_percent_(std::min(100, std::max(0, fail_percent))) {}

SimulatedTaskAdapter::~SimulatedTaskAdapter() { shutdown(); }

int SimulatedTaskAdapter::dispatch(const std::string& request, TaskCallback cb) {
    // next_id_ doubles as the "open for business" flag: shutdown() stores a
    // non-positive value, so a dispatch that races shutdown returns 0 (invalid
    // task id) instead of starting a thread nobody will join.
    if (next_id_.load() <= 0) return 0;
    const int id = next_id_.fetch_add(1);
    if (id <= 0) return 0;

    Entry* e = new Entry();
    e->cb = std::move(cb);

    const int64_t delay = delay_ms_for(id, latency_ms_, jitter_ms_);
    const bool fail = fails_for(id, fail_percent_);
    const int64_t start = now_ms();

    try {
        e->th = std::thread([this, e, id, delay, fail, start, request]() {
            // Cooperative wait: cancel() is honoured within a few milliseconds
            // instead of after the whole simulated 6 s.
            while (!e->cancelled.load() && (now_ms() - start) < delay) {
                const int64_t left = delay - (now_ms() - start);
                const int64_t slice = std::min<int64_t>(left > 0 ? left : 1, 5);
                std::this_thread::sleep_for(std::chrono::milliseconds(slice));
            }

            TaskOutcome out;
            out.id = id;
            out.elapsed_ms = now_ms() - start;
            out.cancelled = e->cancelled.load();
            if (out.cancelled) {
                out.ok = false;
                out.text = "cancelled";
            } else if (fail) {
                out.ok = false;
                out.text = "task failed (simulated): " + request;
            } else {
                out.ok = true;
                out.text = "done: " + request;
            }

            // Claim the callback under the lock so a concurrent cancel() sees a
            // completed entry and cannot produce a second outcome.
            TaskCallback local;
            {
                std::lock_guard<std::mutex> lk(this->mu_);
                local = std::move(e->cb);
            }
            if (local) local(out);
            // Entry (and its joinable thread) stays in tasks_ until shutdown()
            // joins and frees it. A finished thread keeps only its handle and a
            // cached stack while parked here, and the harness runs a handful of
            // tasks per process, so reaping eagerly is not worth the risk of
            // joining a thread that is still inside the user callback.
        });
    } catch (...) {
        delete e;
        return 0;
    }

    bool closed = false;
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (next_id_.load() <= 0) {
            closed = true;  // shutdown() won the race; never register
        } else {
            tasks_[id] = e;
        }
    }
    if (closed) {
        e->cancelled.store(true);
        if (e->th.joinable()) e->th.join();
        delete e;
        return 0;
    }
    return id;
}

void SimulatedTaskAdapter::cancel(int id) {
    // The flag is set under the lock: shutdown() swaps tasks_ out under the same
    // lock, so an Entry pointer is only ever dereferenced while it is still
    // owned by the map.
    std::lock_guard<std::mutex> lk(mu_);
    auto it = tasks_.find(id);
    if (it != tasks_.end() && it->second) it->second->cancelled.store(true);
}

void SimulatedTaskAdapter::shutdown() {
    std::unordered_map<int, Entry*> local;
    {
        std::lock_guard<std::mutex> lk(mu_);
        next_id_.store(-1);  // refuse new dispatches
        local.swap(tasks_);
    }
    if (local.empty()) return;

    // Signal first, join second: a task that is mid-sleep sees the flag on its
    // next slice and reports cancellation, so the join is bounded.
    for (auto& kv : local) kv.second->cancelled.store(true);
    for (auto& kv : local) {
        if (kv.second->th.joinable()) kv.second->th.join();
        delete kv.second;
    }
}

}  // namespace voice
}  // namespace crispasr
