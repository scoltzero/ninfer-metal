#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <vector>

#include "omlx_m5/types.h"

namespace omlx::m5 {

class Engine final {
public:
  static std::unique_ptr<Engine> create(const LoadOptions &load_options = {});
  static std::unique_ptr<Engine>
  create_fake(const LoadOptions &load_options = {},
              const FakeBackendOptions &fake_options = {});

  Engine(const Engine &) = delete;
  Engine &operator=(const Engine &) = delete;
  Engine(Engine &&) = delete;
  Engine &operator=(Engine &&) = delete;
  ~Engine();

  [[nodiscard]] std::uint64_t submit(SubmitRequest request);
  [[nodiscard]] bool cancel(std::uint64_t request_id);
  [[nodiscard]] std::vector<TokenEvent>
  poll_events(std::chrono::microseconds max_wait);
  [[nodiscard]] CapacityReport capacity() const;
  [[nodiscard]] StatsReport stats() const;
  [[nodiscard]] TrimCacheReport trim_cache(std::uint64_t target_free_bytes = 0);
  void set_scheduler_hint(const SchedulerHint &hint);
  void shutdown();

private:
  class Impl;

  explicit Engine(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
