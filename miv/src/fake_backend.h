#pragma once

#include <optional>
#include <thread>

#include "backend.h"

namespace omlx::m5::detail {

class FakeBackend final : public Backend {
public:
  explicit FakeBackend(FakeBackendOptions options);

  void bind_owner(std::thread::id owner) override;
  [[nodiscard]] ExecutionResult execute(std::uint64_t request_id,
                                        const SubmitRequest &request,
                                        const std::atomic<bool> &cancelled,
                                        const EmitTokens &emit_tokens) override;
  [[nodiscard]] std::unique_ptr<BackendTask>
  start(SubmitRequest, std::shared_ptr<std::atomic<bool>>, EmitTokens) override;
  void populate_stats(StatsReport &stats) const override;

private:
  void assert_owner() const;

  FakeBackendOptions options_;
  std::optional<std::thread::id> owner_;
};

} // namespace omlx::m5::detail
