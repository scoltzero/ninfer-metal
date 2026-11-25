#include "serve/engine_router.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <utility>

namespace omlx::m5::serve {

struct RequestStream::State {
  std::mutex mutex;
  std::condition_variable cv;
  std::deque<TokenEvent> events;
  bool closed = false;
  bool local_stopped = false;
  bool metric_finalized = false;
  std::vector<std::vector<std::int32_t>> stop_token_seqs;
  std::size_t max_stop_len = 0;
  std::vector<std::int32_t> pending_tokens;
};

RequestStream::RequestStream(EngineRouter *router, std::uint64_t request_id,
                             std::shared_ptr<State> state)
    : router_(router), request_id_(request_id), state_(std::move(state)) {}

RequestStream::RequestStream(RequestStream &&other) noexcept
    : router_(std::exchange(other.router_, nullptr)),
      request_id_(std::exchange(other.request_id_, 0)),
      state_(std::move(other.state_)),
      terminal_delivered_(std::exchange(other.terminal_delivered_, true)) {}

RequestStream &RequestStream::operator=(RequestStream &&other) noexcept {
  if (this != &other) {
    release();
    router_ = std::exchange(other.router_, nullptr);
    request_id_ = std::exchange(other.request_id_, 0);
    state_ = std::move(other.state_);
    terminal_delivered_ = std::exchange(other.terminal_delivered_, true);
  }
  return *this;
}

RequestStream::~RequestStream() { release(); }

void RequestStream::release() {
  if (router_ == nullptr) {
    return;
  }
  if (!terminal_delivered_) {
    cancel();
  }
  router_->unregister_stream(request_id_);
  router_ = nullptr;
  request_id_ = 0;
  state_.reset();
}

void RequestStream::cancel() {
  if (router_ != nullptr && request_id_ != 0) {
    (void)router_->cancel(request_id_);
  }
}

std::optional<TokenEvent> RequestStream::next_event() {
  if (terminal_delivered_ || state_ == nullptr) {
    return std::nullopt;
  }
  std::unique_lock lock(state_->mutex);
  state_->cv.wait(
      lock, [&] { return !state_->events.empty() || state_->closed; });
  if (state_->events.empty()) {
    terminal_delivered_ = true;
    return std::nullopt;
  }
  TokenEvent ev = std::move(state_->events.front());
  state_->events.pop_front();
  if (ev.kind != EventKind::tokens) {
    terminal_delivered_ = true;
  }
  return ev;
}

EngineRouter::EngineRouter(std::shared_ptr<ServeBackend> backend,
                           std::vector<std::int32_t> default_eos_token_ids)
    : backend_(std::move(backend)),
      default_eos_token_ids_(std::move(default_eos_token_ids)) {
  if (backend_ == nullptr) {
    throw std::invalid_argument(
        "EngineRouter requires a non-null ServeBackend");
  }
  demux_thread_ = std::thread([this] { demux_loop(); });
}

EngineRouter::EngineRouter(std::shared_ptr<Engine> engine,
                           std::vector<std::int32_t> default_eos_token_ids)
    : EngineRouter(std::make_shared<LocalEngineBackend>(std::move(engine)),
                   std::move(default_eos_token_ids)) {}

EngineRouter::~EngineRouter() { stop(); }

void EngineRouter::set_activity_callback(ActivityCallback cb) {
  std::lock_guard lock(callback_mutex_);
  activity_callback_ = std::move(cb);
}

void EngineRouter::notify_activity_changed() {
  ActivityCallback cb;
  {
    std::lock_guard lock(callback_mutex_);
    cb = activity_callback_;
  }
  if (cb) {
    cb();
  }
}

RequestStream EngineRouter::submit(SubmitRequest request) {
  if (stopping_.load(std::memory_order_acquire)) {
    throw std::runtime_error("server is shutting down");
  }

  // Merge model EOS token IDs into stop_token_seqs (deduplicated).
  for (const std::int32_t eos : default_eos_token_ids_) {
    const std::vector<std::int32_t> seq{eos};
    if (std::find(request.stop_token_seqs.begin(),
                  request.stop_token_seqs.end(),
                  seq) == request.stop_token_seqs.end()) {
      request.stop_token_seqs.push_back(seq);
    }
  }

  auto state = std::make_shared<RequestStream::State>();
  state->stop_token_seqs = request.stop_token_seqs;
  for (const auto &seq : state->stop_token_seqs) {
    state->max_stop_len = std::max(state->max_stop_len, seq.size());
  }

  const std::uint64_t prompt_count = request.prompt_tokens.size();

  std::uint64_t request_id = 0;
  {
    // Synchronize submit + stream registration + active count increment
    // against demux_loop so the first TokenEvent can never arrive before
    // the stream map entry exists or decrement active_streams_ before add.
    std::lock_guard lock(streams_mutex_);
    if (stopping_.load(std::memory_order_acquire)) {
      throw std::runtime_error("server is shutting down");
    }
    request_id = backend_->submit(std::move(request));
    active_streams_.fetch_add(1, std::memory_order_acq_rel);
    streams_.emplace(request_id, state);
  }

  requests_total_.fetch_add(1, std::memory_order_relaxed);
  prompt_tokens_total_.fetch_add(prompt_count, std::memory_order_relaxed);
  notify_activity_changed();

  return RequestStream(this, request_id, std::move(state));
}

bool EngineRouter::cancel(std::uint64_t request_id) {
  return backend_->cancel(request_id);
}

void EngineRouter::unregister_stream(std::uint64_t request_id) {
  std::shared_ptr<RequestStream::State> state;
  {
    std::lock_guard lock(streams_mutex_);
    const auto it = streams_.find(request_id);
    if (it != streams_.end()) {
      state = std::move(it->second);
      streams_.erase(it);
    }
  }
  bool decremented = false;
  if (state != nullptr) {
    std::lock_guard lock(state->mutex);
    if (!state->metric_finalized) {
      state->metric_finalized = true;
      active_streams_.fetch_sub(1, std::memory_order_acq_rel);
      requests_cancelled_.fetch_add(1, std::memory_order_relaxed);
      decremented = true;
    }
    state->closed = true;
    state->cv.notify_all();
  }
  if (decremented) {
    notify_activity_changed();
  }
}

void EngineRouter::stop() {
  if (stopping_.exchange(true, std::memory_order_acq_rel)) {
    return;
  }

  // Wake and abort all currently active streams.
  std::vector<std::pair<std::uint64_t, std::shared_ptr<RequestStream::State>>>
      active;
  {
    std::lock_guard lock(streams_mutex_);
    active.reserve(streams_.size());
    for (auto &[id, state] : streams_) {
      active.emplace_back(id, state);
    }
  }
  bool any_decremented = false;
  for (auto &[id, state] : active) {
    (void)backend_->cancel(id);
    std::lock_guard lock(state->mutex);
    if (!state->closed) {
      state->events.push_back(TokenEvent{
          .request_id = id,
          .kind = EventKind::aborted,
          .finish_reason = "cancelled",
          .error = "server shutting down",
      });
      state->closed = true;
      if (!state->metric_finalized) {
        state->metric_finalized = true;
        active_streams_.fetch_sub(1, std::memory_order_acq_rel);
        requests_cancelled_.fetch_add(1, std::memory_order_relaxed);
        any_decremented = true;
      }
      state->cv.notify_all();
    }
  }

  if (any_decremented) {
    notify_activity_changed();
  }

  {
    std::lock_guard lock(stop_mutex_);
  }
  stop_cv_.notify_all();

  backend_->shutdown();
  if (demux_thread_.joinable()) {
    demux_thread_.join();
  }
}

void EngineRouter::fail_all_streams(std::string_view error_msg) {
  std::vector<std::pair<std::uint64_t, std::shared_ptr<RequestStream::State>>>
      active;
  {
    std::lock_guard lock(streams_mutex_);
    if (streams_.empty()) {
      return;
    }
    active.reserve(streams_.size());
    for (auto &[id, state] : streams_) {
      active.emplace_back(id, state);
    }
  }
  bool any_decremented = false;
  const std::string err = error_msg.empty()
                              ? "backend event stream disconnected"
                              : std::string(error_msg);
  for (auto &[id, state] : active) {
    std::lock_guard lock(state->mutex);
    if (!state->closed) {
      state->events.push_back(TokenEvent{
          .request_id = id,
          .kind = EventKind::failed,
          .finish_reason = "error",
          .error = err,
      });
      state->closed = true;
      if (!state->metric_finalized) {
        state->metric_finalized = true;
        active_streams_.fetch_sub(1, std::memory_order_acq_rel);
        requests_failed_.fetch_add(1, std::memory_order_relaxed);
        any_decremented = true;
      }
      state->cv.notify_all();
    }
  }
  if (any_decremented) {
    notify_activity_changed();
  }
}

void EngineRouter::demux_loop() {
  while (!stopping_.load(std::memory_order_acquire)) {
    std::vector<TokenEvent> events;
    try {
      events = backend_->poll_events(std::chrono::milliseconds(5));
    } catch (const std::exception &ex) {
      if (stopping_.load(std::memory_order_acquire)) {
        break;
      }
      fail_all_streams(ex.what());
      std::unique_lock lock(stop_mutex_);
      stop_cv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
        return stopping_.load(std::memory_order_acquire);
      });
      continue;
    } catch (...) {
      if (stopping_.load(std::memory_order_acquire)) {
        break;
      }
      fail_all_streams("backend event stream failed");
      std::unique_lock lock(stop_mutex_);
      stop_cv_.wait_for(lock, std::chrono::milliseconds(50), [this] {
        return stopping_.load(std::memory_order_acquire);
      });
      continue;
    }
    for (auto &event : events) {
      dispatch_event(std::move(event));
    }
  }
}

void EngineRouter::dispatch_event(TokenEvent event) {
  std::shared_ptr<RequestStream::State> state;
  {
    std::lock_guard lock(streams_mutex_);
    const auto it = streams_.find(event.request_id);
    if (it == streams_.end()) {
      return;
    }
    state = it->second;
  }

  bool trigger_cancel = false;
  bool decremented = false;
  {
    std::lock_guard lock(state->mutex);
    if (state->closed || state->local_stopped) {
      return;
    }

    if (event.kind == EventKind::tokens) {
      std::vector<std::int32_t> ready;
      ready.reserve(event.tokens.size());
      for (const std::int32_t tok : event.tokens) {
        state->pending_tokens.push_back(tok);

        bool matched_stop = false;
        std::size_t matched_len = 0;
        for (const auto &seq : state->stop_token_seqs) {
          if (seq.size() <= state->pending_tokens.size() &&
              std::equal(seq.rbegin(), seq.rend(),
                         state->pending_tokens.rbegin())) {
            matched_stop = true;
            matched_len = seq.size();
            break;
          }
        }
        if (matched_stop) {
          state->pending_tokens.resize(state->pending_tokens.size() -
                                       matched_len);
          ready.insert(ready.end(), state->pending_tokens.begin(),
                       state->pending_tokens.end());
          state->pending_tokens.clear();
          if (!ready.empty()) {
            generated_tokens_total_.fetch_add(ready.size(),
                                              std::memory_order_relaxed);
            state->events.push_back(TokenEvent{
                .request_id = event.request_id,
                .tokens = std::move(ready),
                .kind = EventKind::tokens,
            });
          }
          state->events.push_back(TokenEvent{
              .request_id = event.request_id,
              .kind = EventKind::finished,
              .finish_reason = "stop",
          });
          state->local_stopped = true;
          state->closed = true;
          if (!state->metric_finalized) {
            state->metric_finalized = true;
            active_streams_.fetch_sub(1, std::memory_order_acq_rel);
            requests_succeeded_.fetch_add(1, std::memory_order_relaxed);
            decremented = true;
          }
          trigger_cancel = true;
          break;
        }

        const std::size_t hold =
            state->max_stop_len > 0 ? (state->max_stop_len - 1) : 0;
        if (state->pending_tokens.size() > hold) {
          const std::size_t flush_n = state->pending_tokens.size() - hold;
          ready.insert(ready.end(), state->pending_tokens.begin(),
                       state->pending_tokens.begin() +
                           static_cast<std::ptrdiff_t>(flush_n));
          state->pending_tokens.erase(
              state->pending_tokens.begin(),
              state->pending_tokens.begin() +
                  static_cast<std::ptrdiff_t>(flush_n));
        }
      }

      if (!trigger_cancel && !ready.empty()) {
        generated_tokens_total_.fetch_add(ready.size(),
                                          std::memory_order_relaxed);
        state->events.push_back(TokenEvent{
            .request_id = event.request_id,
            .tokens = std::move(ready),
            .kind = EventKind::tokens,
        });
      }
    } else {
      // Terminal event from backend: flush any held-back non-matching tokens.
      if (!state->pending_tokens.empty()) {
        generated_tokens_total_.fetch_add(state->pending_tokens.size(),
                                          std::memory_order_relaxed);
        state->events.push_back(TokenEvent{
            .request_id = event.request_id,
            .tokens = std::move(state->pending_tokens),
            .kind = EventKind::tokens,
        });
        state->pending_tokens.clear();
      }
      if (!state->metric_finalized) {
        state->metric_finalized = true;
        active_streams_.fetch_sub(1, std::memory_order_acq_rel);
        decremented = true;
        switch (event.kind) {
        case EventKind::finished:
          requests_succeeded_.fetch_add(1, std::memory_order_relaxed);
          break;
        case EventKind::aborted:
          requests_cancelled_.fetch_add(1, std::memory_order_relaxed);
          break;
        case EventKind::failed:
          requests_failed_.fetch_add(1, std::memory_order_relaxed);
          break;
        case EventKind::tokens:
          break;
        }
      }
      state->events.push_back(std::move(event));
      state->closed = true;
    }
    state->cv.notify_all();
  }

  if (trigger_cancel) {
    (void)backend_->cancel(event.request_id);
  }
  if (decremented) {
    notify_activity_changed();
  }
}

CapacityReport EngineRouter::capacity() const { return backend_->capacity(); }

StatsReport EngineRouter::stats() const { return backend_->stats(); }

TrimCacheReport EngineRouter::trim_cache(std::uint64_t target_free_bytes) {
  return backend_->trim_cache(target_free_bytes);
}

void EngineRouter::set_scheduler_hint(const SchedulerHint &hint) {
  backend_->set_scheduler_hint(hint);
}

bool EngineRouter::reload_lora(const std::string &lora_adapter,
                               const std::string &lora_mode,
                               std::string *error) {
  return backend_->reload_lora(lora_adapter, lora_mode, error);
}

std::uint64_t EngineRouter::resident_memory_bytes() const {
  return backend_->resident_memory_bytes();
}

std::string EngineRouter::provider_id() const {
  return backend_->provider_id();
}

std::string EngineRouter::backend_mode() const {
  return backend_->backend_mode();
}

RouterMetricsSnapshot EngineRouter::metrics_snapshot() const {
  RouterMetricsSnapshot snap{
      .requests_total = requests_total_.load(std::memory_order_relaxed),
      .requests_succeeded = requests_succeeded_.load(std::memory_order_relaxed),
      .requests_failed = requests_failed_.load(std::memory_order_relaxed),
      .requests_cancelled = requests_cancelled_.load(std::memory_order_relaxed),
      .active_requests = active_streams_.load(std::memory_order_acquire),
      .generated_tokens_total =
          generated_tokens_total_.load(std::memory_order_relaxed),
      .prompt_tokens_total =
          prompt_tokens_total_.load(std::memory_order_relaxed),
  };
  try {
    snap.capacity = backend_->capacity();
    snap.engine_stats = backend_->stats();
    snap.backend_healthy = true;
  } catch (...) {
    snap.capacity.accepting = false;
    snap.backend_healthy = false;
  }
  return snap;
}

} // namespace omlx::m5::serve
