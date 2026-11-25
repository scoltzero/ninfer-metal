#pragma once

#include "omlx_m5/engine.h"
#include "omlx_m5/types.h"

#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/task_info.h>
#endif

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace omlx::m5::serve {

inline std::uint64_t current_process_phys_footprint_bytes() noexcept {
#ifdef __APPLE__
  task_vm_info_data_t vm_info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  const kern_return_t kr =
      task_info(mach_task_self(), TASK_VM_INFO,
                reinterpret_cast<task_info_t>(&vm_info), &count);
  if (kr == KERN_SUCCESS) {
    return static_cast<std::uint64_t>(vm_info.phys_footprint);
  }
#endif
  return 0;
}

class ServeBackend {
public:
  virtual ~ServeBackend() = default;

  [[nodiscard]] virtual std::uint64_t submit(SubmitRequest request) = 0;
  [[nodiscard]] virtual bool cancel(std::uint64_t request_id) = 0;
  [[nodiscard]] virtual std::vector<TokenEvent>
  poll_events(std::chrono::microseconds max_wait) = 0;
  [[nodiscard]] virtual CapacityReport capacity() const = 0;
  [[nodiscard]] virtual StatsReport stats() const = 0;
  [[nodiscard]] virtual TrimCacheReport
  trim_cache(std::uint64_t target_free_bytes = 0) = 0;
  virtual void set_scheduler_hint(const SchedulerHint &hint) = 0;
  [[nodiscard]] virtual std::uint64_t resident_memory_bytes() const = 0;
  virtual void shutdown() = 0;

  [[nodiscard]] virtual std::string provider_id() const = 0;
  [[nodiscard]] virtual std::string backend_mode() const = 0;
};

class LocalEngineBackend final : public ServeBackend {
public:
  explicit LocalEngineBackend(std::shared_ptr<Engine> engine,
                              std::uint64_t initial_footprint_bytes = 0,
                              std::string provider_id = "io.ninfer.metal")
      : engine_(std::move(engine)), provider_id_(std::move(provider_id)),
        resident_bytes_(initial_footprint_bytes) {
    if (engine_ == nullptr) {
      throw std::invalid_argument(
          "LocalEngineBackend requires a non-null Engine");
    }
  }

  ~LocalEngineBackend() override = default;

  [[nodiscard]] std::uint64_t submit(SubmitRequest request) override {
    return engine_->submit(std::move(request));
  }

  [[nodiscard]] bool cancel(std::uint64_t request_id) override {
    return engine_->cancel(request_id);
  }

  [[nodiscard]] std::vector<TokenEvent>
  poll_events(std::chrono::microseconds max_wait) override {
    return engine_->poll_events(max_wait);
  }

  [[nodiscard]] CapacityReport capacity() const override {
    return engine_->capacity();
  }

  [[nodiscard]] StatsReport stats() const override { return engine_->stats(); }

  [[nodiscard]] TrimCacheReport
  trim_cache(std::uint64_t target_free_bytes = 0) override {
    auto report = engine_->trim_cache(target_free_bytes);
    if (report.resident_bytes > 0) {
      resident_bytes_.store(report.resident_bytes, std::memory_order_relaxed);
    }
    return report;
  }

  void set_scheduler_hint(const SchedulerHint &hint) override {
    engine_->set_scheduler_hint(hint);
  }

  [[nodiscard]] std::uint64_t resident_memory_bytes() const override {
    const std::uint64_t tracked =
        resident_bytes_.load(std::memory_order_relaxed);
    if (tracked > 0) {
      return tracked;
    }
    return current_process_phys_footprint_bytes();
  }

  void shutdown() override { engine_->shutdown(); }

  [[nodiscard]] std::string provider_id() const override {
    return provider_id_;
  }

  [[nodiscard]] std::string backend_mode() const override { return "inproc"; }

private:
  std::shared_ptr<Engine> engine_;
  std::string provider_id_;
  std::atomic<std::uint64_t> resident_bytes_{0};
};

} // namespace omlx::m5::serve
