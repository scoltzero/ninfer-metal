#pragma once

#include "serve/serve_backend.h"

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <sys/types.h>
#include <vector>

namespace omlx::m5::serve {

struct UdsSidecarOptions {
  std::string provider_id;
  std::filesystem::path binary_path;
  LoadOptions load_options;
  std::optional<bool> rolling_prefix_cache;
  std::optional<double> prefix_cache_gb;
  std::chrono::milliseconds startup_timeout{15'000};
  std::chrono::milliseconds rpc_timeout{300'000};
  std::uint32_t event_poll_interval_us = 50'000;
};

class UdsSidecarBackend final : public ServeBackend {
public:
  explicit UdsSidecarBackend(UdsSidecarOptions options);
  ~UdsSidecarBackend() override;

  UdsSidecarBackend(const UdsSidecarBackend &) = delete;
  UdsSidecarBackend &operator=(const UdsSidecarBackend &) = delete;

  [[nodiscard]] std::uint64_t submit(SubmitRequest request) override;
  [[nodiscard]] bool cancel(std::uint64_t request_id) override;
  [[nodiscard]] std::vector<TokenEvent>
  poll_events(std::chrono::microseconds max_wait) override;
  [[nodiscard]] CapacityReport capacity() const override;
  [[nodiscard]] StatsReport stats() const override;
  [[nodiscard]] TrimCacheReport
  trim_cache(std::uint64_t target_free_bytes = 0) override;
  void set_scheduler_hint(const SchedulerHint &hint) override;
  [[nodiscard]] std::uint64_t resident_memory_bytes() const override;
  void shutdown() override;

  [[nodiscard]] std::string provider_id() const override;
  [[nodiscard]] std::string backend_mode() const override { return "sidecar"; }

  [[nodiscard]] pid_t child_pid() const noexcept;
  [[nodiscard]] const std::filesystem::path &socket_path() const noexcept;
  [[nodiscard]] std::uint32_t protocol_version() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5::serve
