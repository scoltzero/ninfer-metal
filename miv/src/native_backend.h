#pragma once

#include <mutex>
#include <optional>
#include <thread>

#include "backend.h"

namespace omlx::m5::detail {

class NativeQwenState;
class NativeTask;

class NativeBackend final : public Backend {
public:
  explicit NativeBackend(const LoadOptions &options);
  ~NativeBackend() override;

  void bind_owner(std::thread::id owner) override;
  [[nodiscard]] ExecutionResult execute(std::uint64_t request_id,
                                        const SubmitRequest &request,
                                        const std::atomic<bool> &cancelled,
                                        const EmitTokens &emit_tokens) override;
  [[nodiscard]] std::unique_ptr<BackendTask>
  start(SubmitRequest, std::shared_ptr<std::atomic<bool>>, EmitTokens) override;
  [[nodiscard]] bool batch_enabled() const override;
  [[nodiscard]] std::vector<std::optional<ExecutionResult>>
  advance_batch(std::span<BackendTask *const>) override;
  void populate_stats(StatsReport &stats) const override;
  [[nodiscard]] TrimCacheReport
  trim_cache(std::uint64_t target_free_bytes) override;
  void set_scheduler_hint(const SchedulerHint &hint) override;

private:
  friend class NativeTask;
  void assert_owner() const;

  LoadOptions options_;
  std::unique_ptr<NativeQwenState> ar_program_;
  std::optional<std::thread::id> owner_;
  mutable std::mutex stats_mutex_;
  StatsReport backend_stats_;
};

} // namespace omlx::m5::detail
