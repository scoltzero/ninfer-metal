#include "omlx_m5/engine.h"
#include "omlx_m5/position_sampling.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <utility>

#include "backend.h"
#include "fake_backend.h"
#include "native_backend.h"
#include "spec_controller.h"

namespace omlx::m5 {

namespace {

struct RequestItem {
  std::uint64_t id = 0;
  SubmitRequest request;
  std::shared_ptr<std::atomic<bool>> cancelled;
};

} // namespace

class Engine::Impl {
public:
  Impl(LoadOptions load_options, std::unique_ptr<detail::Backend> backend)
      : load_options_(std::move(load_options)), backend_(std::move(backend)) {
    if (load_options_.max_concurrency < 1 || load_options_.max_concurrency > 8)
      throw std::invalid_argument("max_concurrency must be in [1, 8]");
    if (load_options_.max_context <= 0) {
      throw std::invalid_argument("max_context must be positive");
    }
    switch (load_options_.speculative.backend) {
    case SpeculativeBackend::none:
      if (load_options_.speculative.draft_tokens != 0) {
        throw std::invalid_argument(
            "disabled speculative decoding requires draft_tokens=0");
      }
      break;
    case SpeculativeBackend::dflash2:
      if (load_options_.speculative.draft_tokens > kMaxDFlash2DraftTokens) {
        throw std::invalid_argument(
            "DFlash2 requires draft_tokens in [0, 7] (0 = auto)");
      }
      break;
    case SpeculativeBackend::mtp:
      if (load_options_.speculative.draft_tokens > kMaxMtpDraftTokens) {
        throw std::invalid_argument(
            "MTP requires draft_tokens in [0, 4] (0 = auto)");
      }
      break;
    case SpeculativeBackend::dflash:
      throw std::invalid_argument(
          "selected speculative backend is declared but not implemented in R1");
    }
    if (load_options_.kv_dtype != "bf16" &&
        load_options_.kv_dtype != "affine4_g128") {
      throw std::invalid_argument(
          "kv_dtype must be bf16 or affine4_g128");
    }
    if (load_options_.prefill_engine != "auto" &&
        load_options_.prefill_engine != "nax" &&
        load_options_.prefill_engine != "nax_ane" &&
        load_options_.prefill_engine != "flash_nax") {
      throw std::invalid_argument(
          "prefill_engine must be auto, nax, nax_ane, or flash_nax");
    }
    if (load_options_.lora_mode.empty()) {
      load_options_.lora_mode = "merge";
    }
    if (load_options_.lora_mode != "merge" &&
        load_options_.lora_mode != "bypass") {
      throw std::invalid_argument("lora_mode must be merge or bypass");
    }
    static_cast<void>(detail::resolve_spec_search_mode(load_options_));
    worker_ = std::thread([this] { worker_loop(); });
  }

  ~Impl() { shutdown(); }

  std::uint64_t submit(SubmitRequest request) {
    if (request.max_new_tokens <= 0) {
      throw std::invalid_argument("max_new_tokens must be positive");
    }
    if (request.prompt_tokens.size() +
            static_cast<std::size_t>(request.max_new_tokens) >
        static_cast<std::size_t>(load_options_.max_context)) {
      throw std::invalid_argument(
          "prompt plus generation budget exceeds max_context");
    }
    if (!std::isfinite(request.sampling.temperature) ||
        request.sampling.temperature < 0.0F) {
      throw std::invalid_argument(
          "temperature must be finite and non-negative");
    }
    if (!std::isfinite(request.sampling.top_p) ||
        request.sampling.top_p <= 0.0F || request.sampling.top_p > 1.0F) {
      throw std::invalid_argument("top_p must be finite and in (0, 1]");
    }
    if (!std::isfinite(request.sampling.min_p) ||
        request.sampling.min_p < 0.0F || request.sampling.min_p > 1.0F) {
      throw std::invalid_argument("min_p must be finite and in [0, 1]");
    }
    if (request.sampling.top_k < 0) {
      throw std::invalid_argument("top_k must not be negative");
    }
    const auto &sampling = request.sampling;
    if (!std::isfinite(sampling.repetition_penalty) ||
        sampling.repetition_penalty <= 0 ||
        !std::isfinite(sampling.presence_penalty) ||
        !std::isfinite(sampling.frequency_penalty) ||
        sampling.repetition_context_size <= 0 ||
        sampling.repetition_context_size > kMaxRepetitionContextSize) {
      throw std::invalid_argument("invalid sampling penalties or repetition_context_size");
    }
    if ((sampling.has_penalties() || request.thinking_budget) &&
        load_options_.sampling_contract != kDefaultServingSamplingContract) {
      throw std::invalid_argument(
          "sampling penalties and thinking_budget require position_coupled_v1");
    }
    if (const auto &thinking = request.thinking_budget) {
      const auto has_negative = [](const auto &ids) {
        return std::any_of(ids.begin(), ids.end(),
                           [](std::int32_t id) { return id < 0; });
      };
      if (thinking->budget < 0 || thinking->end_token_ids.empty() ||
          has_negative(thinking->start_token_ids) ||
          has_negative(thinking->end_token_ids) ||
          has_negative(thinking->force_token_ids) ||
          (!thinking->force_token_ids.empty() &&
           std::search(thinking->force_token_ids.begin(), thinking->force_token_ids.end(),
                       thinking->end_token_ids.begin(), thinking->end_token_ids.end()) ==
               thinking->force_token_ids.end())) {
        throw std::invalid_argument("invalid thinking_budget or closing token sequence");
      }
    }
    for (const auto &stop_sequence : request.stop_token_seqs) {
      if (stop_sequence.empty()) {
        throw std::invalid_argument("stop token sequences must not be empty");
      }
    }

    auto cancelled = std::make_shared<std::atomic<bool>>(false);
    std::lock_guard lock(request_mutex_);
    if (!accepting_) {
      throw std::runtime_error("engine is shut down");
    }
    if (load_options_.max_concurrency > 1 &&
        request_queue_.size() >= std::size_t(load_options_.max_concurrency) * 4)
      throw std::runtime_error("bounded request queue is full");
    const auto id = next_request_id_++;
    cancellations_.emplace(id, cancelled);
    request_queue_.push_back({
        .id = id,
        .request = std::move(request),
        .cancelled = std::move(cancelled),
    });
    submitted_.fetch_add(1, std::memory_order_relaxed);
    request_cv_.notify_one();
    return id;
  }

  bool cancel(std::uint64_t request_id) {
    std::lock_guard lock(request_mutex_);
    const auto it = cancellations_.find(request_id);
    if (it == cancellations_.end()) {
      return false;
    }
    it->second->store(true, std::memory_order_release);
    request_cv_.notify_one();
    return true;
  }

  std::vector<TokenEvent> poll_events(std::chrono::microseconds max_wait) {
    if (max_wait.count() < 0) {
      throw std::invalid_argument("max_wait must not be negative");
    }
    std::unique_lock lock(event_mutex_);
    event_cv_.wait_for(lock, max_wait, [this] {
      return !events_.empty() || worker_done_.load(std::memory_order_acquire);
    });
    std::vector<TokenEvent> result;
    result.reserve(events_.size());
    while (!events_.empty()) {
      result.push_back(std::move(events_.front()));
      events_.pop_front();
    }
    return result;
  }

  CapacityReport capacity() const {
    std::lock_guard lock(request_mutex_);
    return {
        .max_concurrency = load_options_.max_concurrency,
        .active_requests = active_count_,
        .queued_requests = static_cast<std::int32_t>(request_queue_.size()),
        .accepting = accepting_ &&
            (load_options_.max_concurrency == 1 ||
             request_queue_.size() <
                 std::size_t(load_options_.max_concurrency) * 4),
    };
  }

  StatsReport stats() const {
    StatsReport result{
        .submitted = submitted_.load(std::memory_order_relaxed),
        .completed = completed_.load(std::memory_order_relaxed),
        .aborted = aborted_.load(std::memory_order_relaxed),
        .failed = failed_.load(std::memory_order_relaxed),
        .emitted_tokens = emitted_tokens_.load(std::memory_order_relaxed),
    };
    result.max_context = load_options_.max_context;
    result.max_concurrency = load_options_.max_concurrency;
    result.kv_dtype = load_options_.kv_dtype;
    result.sampling_contract_id = load_options_.sampling_contract;
    populate_prefill_config_stats(load_options_, result);
    backend_->populate_stats(result);
    return result;
  }

  TrimCacheReport trim_cache(std::uint64_t target_free_bytes) {
    if (worker_done_.load(std::memory_order_acquire)) {
      return {};
    }
    std::future<TrimCacheReport> future;
    {
      std::lock_guard lock(request_mutex_);
      if (!accepting_ || stop_requested_ ||
          worker_done_.load(std::memory_order_acquire)) {
        return {};
      }
      PendingTrim pending{.target_free_bytes = target_free_bytes};
      future = pending.promise.get_future();
      trim_queue_.push_back(std::move(pending));
    }
    request_cv_.notify_one();
    return future.get();
  }

  void set_scheduler_hint(const SchedulerHint &hint) {
    backend_->set_scheduler_hint(hint);
  }

  bool reload_lora(const std::string &lora_adapter,
                   const std::string &lora_mode, std::string *error) {
    if (worker_done_.load(std::memory_order_acquire)) {
      if (error != nullptr) {
        *error = "engine is shut down";
      }
      return false;
    }
    std::future<std::pair<bool, std::string>> future;
    {
      std::lock_guard lock(request_mutex_);
      if (!accepting_ || stop_requested_ ||
          worker_done_.load(std::memory_order_acquire)) {
        if (error != nullptr) {
          *error = "engine is shut down";
        }
        return false;
      }
      PendingLoraReload pending{
          .lora_adapter = lora_adapter,
          .lora_mode = lora_mode,
      };
      future = pending.promise.get_future();
      lora_reload_queue_.push_back(std::move(pending));
    }
    request_cv_.notify_one();
    auto [ok, err_msg] = future.get();
    if (!ok && error != nullptr) {
      *error = std::move(err_msg);
    }
    return ok;
  }

  void shutdown() {
    std::lock_guard shutdown_lock(shutdown_mutex_);
    {
      std::lock_guard lock(request_mutex_);
      accepting_ = false;
      stop_requested_ = true;
      for (auto &[_, cancellation] : cancellations_) {
        cancellation->store(true, std::memory_order_release);
      }
    }
    request_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  struct PendingTrim {
    std::uint64_t target_free_bytes = 0;
    std::promise<TrimCacheReport> promise;
  };

  struct PendingLoraReload {
    std::string lora_adapter;
    std::string lora_mode;
    std::promise<std::pair<bool, std::string>> promise;
  };

  void process_pending_trims() {
    std::deque<PendingTrim> local;
    std::deque<PendingLoraReload> local_reloads;
    {
      std::lock_guard lock(request_mutex_);
      if (trim_queue_.empty() && lora_reload_queue_.empty()) {
        return;
      }
      local.swap(trim_queue_);
      local_reloads.swap(lora_reload_queue_);
    }
    for (auto &item : local) {
      try {
        item.promise.set_value(backend_->trim_cache(item.target_free_bytes));
      } catch (...) {
        item.promise.set_value(TrimCacheReport{});
      }
    }
    for (auto &item : local_reloads) {
      try {
        std::string err_msg;
        const bool ok =
            backend_->reload_lora(item.lora_adapter, item.lora_mode, &err_msg);
        item.promise.set_value({ok, std::move(err_msg)});
      } catch (const std::exception &ex) {
        item.promise.set_value({false, ex.what()});
      } catch (...) {
        item.promise.set_value({false, "unknown LoRA reload failure"});
      }
    }
  }

  void fail_pending_trims() {
    std::deque<PendingTrim> local;
    std::deque<PendingLoraReload> local_reloads;
    {
      std::lock_guard lock(request_mutex_);
      local.swap(trim_queue_);
      local_reloads.swap(lora_reload_queue_);
    }
    for (auto &item : local) {
      item.promise.set_value(TrimCacheReport{});
    }
    for (auto &item : local_reloads) {
      item.promise.set_value({false, "engine is shut down"});
    }
  }

  void push_event(TokenEvent event) {
    {
      std::lock_guard lock(event_mutex_);
      events_.push_back(std::move(event));
    }
    event_cv_.notify_all();
  }

  void finish_request(const RequestItem &item,
                      const detail::ExecutionResult &result) {
    switch (result.kind) {
    case EventKind::finished:
      completed_.fetch_add(1, std::memory_order_relaxed);
      break;
    case EventKind::aborted:
      aborted_.fetch_add(1, std::memory_order_relaxed);
      break;
    case EventKind::failed:
      failed_.fetch_add(1, std::memory_order_relaxed);
      break;
    case EventKind::tokens:
      throw std::logic_error("backend returned a non-terminal result");
    }
    {
      std::lock_guard lock(request_mutex_);
      cancellations_.erase(item.id);
      if (active_count_ == 0)
        throw std::logic_error("active request count underflow");
      --active_count_;
      active_request_id_ = 0;
    }
    push_event({
        .request_id = item.id,
        .kind = result.kind,
        .finish_reason = result.finish_reason,
        .error = result.error,
        .request_stats = result.request_stats,
    });
  }

  void worker_loop() {
    backend_->bind_owner(std::this_thread::get_id());
    if (load_options_.max_concurrency > 1) {
      worker_loop_interleaved();
      return;
    }
    for (;;) {
      process_pending_trims();
      RequestItem item;
      {
        std::unique_lock lock(request_mutex_);
        request_cv_.wait(lock, [this] {
          return stop_requested_ || !request_queue_.empty() ||
                 !trim_queue_.empty() || !lora_reload_queue_.empty();
        });
        if (!trim_queue_.empty() || !lora_reload_queue_.empty()) {
          continue;
        }
        if (request_queue_.empty()) {
          if (stop_requested_) {
            break;
          }
          continue;
        }
        item = std::move(request_queue_.front());
        request_queue_.pop_front();
        active_request_id_ = item.id;
        ++active_count_;
      }

      if (item.cancelled->load(std::memory_order_acquire)) {
        finish_request(item, {
                                 .kind = EventKind::aborted,
                                 .finish_reason = "cancelled",
                             });
        continue;
      }

      detail::ExecutionResult result;
      try {
        result = backend_->execute(
            item.id, item.request, *item.cancelled,
            [this, request_id = item.id](std::vector<std::int32_t> tokens) {
              emitted_tokens_.fetch_add(tokens.size(),
                                        std::memory_order_relaxed);
              push_event({
                  .request_id = request_id,
                  .tokens = std::move(tokens),
                  .kind = EventKind::tokens,
              });
            });
      } catch (const std::exception &error) {
        result = {
            .kind = EventKind::failed,
            .finish_reason = "error",
            .error = error.what(),
        };
      } catch (...) {
        result = {
            .kind = EventKind::failed,
            .finish_reason = "error",
            .error = "unknown backend failure",
        };
      }
      finish_request(item, result);
    }
    fail_pending_trims();
    worker_done_.store(true, std::memory_order_release);
    event_cv_.notify_all();
  }

  void worker_loop_interleaved() {
    struct Active {
      RequestItem item;
      std::unique_ptr<detail::BackendTask> task;
    };
    std::deque<Active> ready;
    for (;;) {
      process_pending_trims();
      while (ready.size() < std::size_t(load_options_.max_concurrency)) {
        RequestItem item;
        {
          std::unique_lock lock(request_mutex_);
          if (ready.empty())
            request_cv_.wait(lock, [this] {
              return stop_requested_ || !request_queue_.empty() ||
                     !trim_queue_.empty() || !lora_reload_queue_.empty();
            });
          if (request_queue_.empty()) break;
          item = std::move(request_queue_.front());
          request_queue_.pop_front();
          ++active_count_;
        }
        if (item.cancelled->load(std::memory_order_acquire)) {
          finish_request(item, {.kind = EventKind::aborted,
                                .finish_reason = "cancelled"});
          continue;
        }
        try {
          auto task = backend_->start(
              item.request, item.cancelled,
              [this, request_id = item.id](std::vector<std::int32_t> tokens) {
                emitted_tokens_.fetch_add(tokens.size(),
                                          std::memory_order_relaxed);
                push_event({.request_id = request_id,
                            .tokens = std::move(tokens),
                            .kind = EventKind::tokens});
              });
          ready.push_back({std::move(item), std::move(task)});
        } catch (const std::exception &error) {
          finish_request(item, {.kind = EventKind::failed,
                                .finish_reason = "error",
                                .error = error.what()});
        } catch (...) {
          finish_request(item, {.kind = EventKind::failed,
                                .finish_reason = "error",
                                .error = "unknown backend failure"});
        }
      }
      if (ready.empty()) {
        process_pending_trims();
        std::lock_guard lock(request_mutex_);
        if (stop_requested_ && request_queue_.empty()) break;
        continue;
      }
      if (backend_->batch_enabled() && ready.size() > 1) {
        std::vector<detail::BackendTask *> tasks;
        tasks.reserve(ready.size());
        for (auto &entry : ready) tasks.push_back(entry.task.get());
        std::vector<std::optional<detail::ExecutionResult>> outcomes;
        try {
          outcomes = backend_->advance_batch(tasks);
          if (outcomes.size() != ready.size())
            throw std::logic_error("backend batch result count mismatch");
        } catch (const std::exception &error) {
          outcomes.assign(ready.size(), detail::ExecutionResult{
                                            .kind = EventKind::failed,
                                            .finish_reason = "error",
                                            .error = error.what()});
        } catch (...) {
          outcomes.assign(ready.size(), detail::ExecutionResult{
                                            .kind = EventKind::failed,
                                            .finish_reason = "error",
                                            .error = "unknown batch failure"});
        }
        const auto count = outcomes.size();
        for (std::size_t index = 0; index < count; ++index) {
          auto current = std::move(ready.front());
          ready.pop_front();
          if (outcomes[index]) {
            outcomes[index]->request_stats = current.task->request_stats();
            current.task.reset();
            finish_request(current.item, *outcomes[index]);
          } else {
            ready.push_back(std::move(current));
          }
        }
        continue;
      }
      auto current = std::move(ready.front());
      ready.pop_front();
      std::optional<detail::ExecutionResult> result;
      try {
        result = current.item.cancelled->load(std::memory_order_acquire)
                     ? std::optional(detail::ExecutionResult{
                           .kind = EventKind::aborted,
                           .finish_reason = "cancelled"})
                     : current.task->advance();
      } catch (const std::exception &error) {
        result = detail::ExecutionResult{.kind = EventKind::failed,
                                         .finish_reason = "error",
                                         .error = error.what()};
      } catch (...) {
        result = detail::ExecutionResult{.kind = EventKind::failed,
                                         .finish_reason = "error",
                                         .error = "unknown backend failure"};
      }
      if (result) {
        result->request_stats = current.task->request_stats();
        current.task.reset(); // release slot before a new request starts
        finish_request(current.item, *result);
      } else {
        ready.push_back(std::move(current));
      }
    }
    fail_pending_trims();
    worker_done_.store(true, std::memory_order_release);
    event_cv_.notify_all();
  }

  LoadOptions load_options_;
  std::unique_ptr<detail::Backend> backend_;

  mutable std::mutex request_mutex_;
  std::condition_variable request_cv_;
  std::deque<RequestItem> request_queue_;
  std::deque<PendingTrim> trim_queue_;
  std::deque<PendingLoraReload> lora_reload_queue_;
  std::unordered_map<std::uint64_t, std::shared_ptr<std::atomic<bool>>>
      cancellations_;
  std::uint64_t next_request_id_ = 1;
  std::uint64_t active_request_id_ = 0;
  std::int32_t active_count_ = 0;
  bool accepting_ = true;
  bool stop_requested_ = false;

  std::mutex event_mutex_;
  std::condition_variable event_cv_;
  std::deque<TokenEvent> events_;
  std::atomic<bool> worker_done_ = false;

  std::atomic<std::uint64_t> submitted_ = 0;
  std::atomic<std::uint64_t> completed_ = 0;
  std::atomic<std::uint64_t> aborted_ = 0;
  std::atomic<std::uint64_t> failed_ = 0;
  std::atomic<std::uint64_t> emitted_tokens_ = 0;

  std::mutex shutdown_mutex_;
  std::thread worker_;
};

std::unique_ptr<Engine> Engine::create(const LoadOptions &load_options) {
  LoadOptions normalized = load_options;
  if (normalized.lora_mode.empty()) {
    normalized.lora_mode = "merge";
  }
  if (normalized.lora_mode != "merge" && normalized.lora_mode != "bypass") {
    throw std::invalid_argument("lora_mode must be merge or bypass");
  }
  auto backend = std::make_unique<detail::NativeBackend>(normalized);
  return std::unique_ptr<Engine>(
      new Engine(std::make_unique<Impl>(normalized, std::move(backend))));
}

std::unique_ptr<Engine>
Engine::create_fake(const LoadOptions &load_options,
                    const FakeBackendOptions &fake_options) {
  if(parse_sampling_contract(load_options.sampling_contract)!=SamplingContract::LegacyRejectionV1)
    throw std::invalid_argument("position-coupled sampling requires native backend");
  LoadOptions normalized = load_options;
  if (normalized.lora_mode.empty()) {
    normalized.lora_mode = "merge";
  }
  if (normalized.lora_mode != "merge" && normalized.lora_mode != "bypass") {
    throw std::invalid_argument("lora_mode must be merge or bypass");
  }
  auto backend = std::make_unique<detail::FakeBackend>(fake_options, normalized);
  return std::unique_ptr<Engine>(
      new Engine(std::make_unique<Impl>(normalized, std::move(backend))));
}

Engine::Engine(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

Engine::~Engine() = default;

std::uint64_t Engine::submit(SubmitRequest request) {
  return impl_->submit(std::move(request));
}

bool Engine::cancel(std::uint64_t request_id) {
  return impl_->cancel(request_id);
}

std::vector<TokenEvent>
Engine::poll_events(std::chrono::microseconds max_wait) {
  return impl_->poll_events(max_wait);
}

CapacityReport Engine::capacity() const { return impl_->capacity(); }

StatsReport Engine::stats() const { return impl_->stats(); }

TrimCacheReport Engine::trim_cache(std::uint64_t target_free_bytes) {
  return impl_->trim_cache(target_free_bytes);
}

void Engine::set_scheduler_hint(const SchedulerHint &hint) {
  impl_->set_scheduler_hint(hint);
}

bool Engine::reload_lora(const std::string &lora_adapter,
                         const std::string &lora_mode, std::string *error) {
  return impl_->reload_lora(lora_adapter, lora_mode, error);
}

void Engine::shutdown() { impl_->shutdown(); }

} // namespace omlx::m5
