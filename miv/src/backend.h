#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <span>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "omlx_m5/types.h"

namespace omlx::m5::detail {

struct ExecutionResult {
  EventKind kind = EventKind::finished;
  std::string finish_reason = "length";
  std::string error;
  std::optional<RequestStats> request_stats;
};

using EmitTokens = std::function<void(std::vector<std::int32_t>)>;

// One non-preemptible operation per advance; nullopt yields to another ready
// request. The task owns all per-request progress and must clean up on cancel.
class BackendTask {
public:
  virtual ~BackendTask() = default;
  [[nodiscard]] virtual std::optional<ExecutionResult> advance() = 0;
  [[nodiscard]] virtual std::optional<RequestStats> request_stats() const {
    return std::nullopt;
  }
};

class Backend {
public:
  virtual ~Backend() = default;
  virtual void bind_owner(std::thread::id owner) = 0;
  [[nodiscard]] virtual ExecutionResult
  execute(std::uint64_t request_id, const SubmitRequest &request,
          const std::atomic<bool> &cancelled,
          const EmitTokens &emit_tokens) = 0;
  [[nodiscard]] virtual std::unique_ptr<BackendTask>
  start(SubmitRequest, std::shared_ptr<std::atomic<bool>>, EmitTokens) {
    throw std::logic_error("backend does not support stepped requests");
  }
  [[nodiscard]] virtual bool batch_enabled() const { return false; }
  [[nodiscard]] virtual std::vector<std::optional<ExecutionResult>>
  advance_batch(std::span<BackendTask *const> tasks) {
    std::vector<std::optional<ExecutionResult>> results;
    results.reserve(tasks.size());
    for (auto *task : tasks) results.push_back(task->advance());
    return results;
  }
  virtual void populate_stats(StatsReport &stats) const = 0;
  [[nodiscard]] virtual TrimCacheReport
  trim_cache(std::uint64_t /*target_free_bytes*/) {
    return {};
  }
  virtual void set_scheduler_hint(const SchedulerHint & /*hint*/) {}
  [[nodiscard]] virtual bool reload_lora(const std::string & /*lora_adapter*/,
                                         const std::string & /*lora_mode*/,
                                         std::string *error = nullptr) {
    if (error != nullptr) {
      *error = "backend does not support runtime LoRA reload";
    }
    return false;
  }
};

} // namespace omlx::m5::detail
