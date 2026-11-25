#pragma once

#include "omlx_m5/engine.h"
#include "omlx_m5/types.h"
#include "serve/serve_backend.h"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <vector>

namespace omlx::m5::serve {

struct RouterMetricsSnapshot {
  std::uint64_t requests_total = 0;
  std::uint64_t requests_succeeded = 0;
  std::uint64_t requests_failed = 0;
  std::uint64_t requests_cancelled = 0;
  std::uint64_t active_requests = 0;
  std::uint64_t generated_tokens_total = 0;
  std::uint64_t prompt_tokens_total = 0;
  CapacityReport capacity;
  StatsReport engine_stats;
  bool backend_healthy = true;
};

class EngineRouter;

class RequestStream final {
public:
  struct State;

  RequestStream(EngineRouter *router, std::uint64_t request_id,
                std::shared_ptr<State> state);
  ~RequestStream();

  RequestStream(const RequestStream &) = delete;
  RequestStream &operator=(const RequestStream &) = delete;
  RequestStream(RequestStream &&other) noexcept;
  RequestStream &operator=(RequestStream &&other) noexcept;

  [[nodiscard]] std::uint64_t request_id() const noexcept {
    return request_id_;
  }

  // Blocks until the next TokenEvent arrives or the stream terminates.
  // Returns std::nullopt after a terminal event has already been consumed.
  [[nodiscard]] std::optional<TokenEvent> next_event();

  // Requests cancellation on the underlying backend (idempotent).
  void cancel();

private:
  void release();

  EngineRouter *router_ = nullptr;
  std::uint64_t request_id_ = 0;
  std::shared_ptr<State> state_;
  bool terminal_delivered_ = false;
};

class EngineRouter final {
public:
  using ActivityCallback = std::function<void()>;

  EngineRouter(std::shared_ptr<ServeBackend> backend,
               std::vector<std::int32_t> default_eos_token_ids);
  EngineRouter(std::shared_ptr<Engine> engine,
               std::vector<std::int32_t> default_eos_token_ids);
  ~EngineRouter();

  EngineRouter(const EngineRouter &) = delete;
  EngineRouter &operator=(const EngineRouter &) = delete;

  [[nodiscard]] RequestStream submit(SubmitRequest request);
  bool cancel(std::uint64_t request_id);
  void stop();

  void set_activity_callback(ActivityCallback cb);

  [[nodiscard]] bool is_running() const noexcept {
    return !stopping_.load(std::memory_order_acquire);
  }
  [[nodiscard]] std::uint64_t active_requests() const noexcept {
    return active_streams_.load(std::memory_order_acquire);
  }
  [[nodiscard]] RouterMetricsSnapshot metrics_snapshot() const;
  [[nodiscard]] CapacityReport capacity() const;
  [[nodiscard]] StatsReport stats() const;
  [[nodiscard]] TrimCacheReport
  trim_cache(std::uint64_t target_free_bytes = 0);
  void set_scheduler_hint(const SchedulerHint &hint);
  [[nodiscard]] bool reload_lora(const std::string &lora_adapter,
                                 const std::string &lora_mode = "bypass",
                                 std::string *error = nullptr);
  [[nodiscard]] std::uint64_t resident_memory_bytes() const;
  [[nodiscard]] std::string provider_id() const;
  [[nodiscard]] std::string backend_mode() const;
  [[nodiscard]] std::shared_ptr<ServeBackend> backend() const noexcept {
    return backend_;
  }

private:
  friend class RequestStream;

  void unregister_stream(std::uint64_t request_id);
  void demux_loop();
  void dispatch_event(TokenEvent event);
  void fail_all_streams(std::string_view error_msg);
  void notify_activity_changed();

  std::shared_ptr<ServeBackend> backend_;
  std::vector<std::int32_t> default_eos_token_ids_;

  mutable std::mutex streams_mutex_;
  std::unordered_map<std::uint64_t, std::shared_ptr<RequestStream::State>>
      streams_;

  std::mutex callback_mutex_;
  ActivityCallback activity_callback_;

  std::mutex stop_mutex_;
  std::condition_variable stop_cv_;
  std::atomic<bool> stopping_{false};
  std::thread demux_thread_;

  std::atomic<std::uint64_t> requests_total_{0};
  std::atomic<std::uint64_t> requests_succeeded_{0};
  std::atomic<std::uint64_t> requests_failed_{0};
  std::atomic<std::uint64_t> requests_cancelled_{0};
  std::atomic<std::uint64_t> active_streams_{0};
  std::atomic<std::uint64_t> generated_tokens_total_{0};
  std::atomic<std::uint64_t> prompt_tokens_total_{0};
};

} // namespace omlx::m5::serve
