// task_adapter.h — background delegation, A2A-shaped but simulated.
//
// The interface is intentionally the shape a real OpenClaw/A2A adapter would
// need: dispatch returns a task id, completion/cancellation arrive
// asynchronously on another thread, and a timeout is *not* a cancellation.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>

namespace crispasr {
namespace voice {

struct TaskOutcome {
    int id = 0;
    bool ok = false;
    bool cancelled = false;
    std::string text;      // spoken result or error message
    int64_t elapsed_ms = 0;
};

using TaskCallback = std::function<void(const TaskOutcome&)>;

class TaskAdapter {
   public:
    virtual ~TaskAdapter() = default;
    // Non-blocking. Returns a task id (>0).
    virtual int dispatch(const std::string& request, TaskCallback cb) = 0;
    // Explicit cancellation; never implied by a timeout.
    virtual void cancel(int id) = 0;
    virtual void shutdown() = 0;
};

// Deterministic simulated backend: fixed latency + jitter, optional failure,
// cancellable. One thread per task is fine at this scale and keeps the
// simulation honest about asynchrony.
class SimulatedTaskAdapter : public TaskAdapter {
   public:
    SimulatedTaskAdapter(int latency_ms, int jitter_ms, int fail_percent);
    int dispatch(const std::string& request, TaskCallback cb) override;
    void cancel(int id) override;
    void shutdown() override;
    ~SimulatedTaskAdapter() override;

   private:
    struct Entry {
        std::thread th;
        std::atomic<bool> cancelled{false};
        TaskCallback cb;
    };
    int latency_ms_, jitter_ms_, fail_percent_;
    std::atomic<int> next_id_{1};
    std::mutex mu_;
    std::unordered_map<int, Entry*> tasks_;
};

}  // namespace voice
}  // namespace crispasr
