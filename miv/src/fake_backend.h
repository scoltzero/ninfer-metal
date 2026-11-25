#pragma once

#include <mutex>
#include <optional>
#include <string>
#include <thread>

#include "backend.h"

namespace omlx::m5::detail {

class FakeBackend final : public Backend {
public:
  explicit FakeBackend(FakeBackendOptions options,
                       LoadOptions load_options = {});

  void bind_owner(std::thread::id owner) override;
  [[nodiscard]] ExecutionResult execute(std::uint64_t request_id,
                                        const SubmitRequest &request,
                                        const std::atomic<bool> &cancelled,
                                        const EmitTokens &emit_tokens) override;
  [[nodiscard]] std::unique_ptr<BackendTask>
  start(SubmitRequest, std::shared_ptr<std::atomic<bool>>, EmitTokens) override;
  void populate_stats(StatsReport &stats) const override;
  [[nodiscard]] bool reload_lora(const std::string &lora_adapter,
                                 const std::string &lora_mode,
                                 std::string *error = nullptr) override;

private:
  void assert_owner() const;

  FakeBackendOptions options_;
  std::optional<std::thread::id> owner_;
  mutable std::mutex lora_mutex_;
  std::string lora_adapter_;
  std::string lora_mode_ = "merge";
  std::uint64_t lora_buffer_bytes_ = 0;
  std::uint32_t lora_adapted_layers_ = 0;
};

} // namespace omlx::m5::detail
