#include "fake_backend.h"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

namespace omlx::m5::detail {

namespace {

std::int32_t next_fake_token(std::int32_t base, std::int32_t offset,
                             std::int32_t vocab_size) noexcept {
  const std::int64_t mod = static_cast<std::int64_t>(vocab_size);
  std::int64_t rem = (static_cast<std::int64_t>(base) + offset + 1) % mod;
  if (rem < 0) {
    rem += mod;
  }
  return static_cast<std::int32_t>(rem);
}

} // namespace

FakeBackend::FakeBackend(FakeBackendOptions options) : options_(options) {
  if (options_.event_batch_size <= 0) {
    throw std::invalid_argument("event_batch_size must be positive");
  }
  if (options_.step_delay_us < 0) {
    throw std::invalid_argument("step_delay_us must not be negative");
  }
  if (options_.fail_after_tokens < -1) {
    throw std::invalid_argument("fail_after_tokens must be -1 or non-negative");
  }
  if (options_.vocab_size <= 1) {
    throw std::invalid_argument("vocab_size must be greater than one");
  }
}

void FakeBackend::bind_owner(std::thread::id owner) {
  if (owner_ && *owner_ != owner) {
    throw std::logic_error("fake backend owner may only be bound once");
  }
  owner_ = owner;
}

void FakeBackend::assert_owner() const {
  if (!owner_ || *owner_ != std::this_thread::get_id()) {
    throw std::logic_error(
        "backend mutation attempted outside the worker thread");
  }
}

ExecutionResult FakeBackend::execute(std::uint64_t /*request_id*/,
                                     const SubmitRequest &request,
                                     const std::atomic<bool> &cancelled,
                                     const EmitTokens &emit_tokens) {
  assert_owner();
  if (options_.fail_before_first) {
    return {
        .kind = EventKind::failed,
        .finish_reason = "error",
        .error = "injected failure before first token",
    };
  }

  const std::int32_t base =
      request.prompt_tokens.empty() ? 0 : request.prompt_tokens.back();
  std::vector<std::int32_t> batch;
  batch.reserve(static_cast<std::size_t>(options_.event_batch_size));

  for (std::int32_t i = 0; i < request.max_new_tokens; ++i) {
    if (cancelled.load(std::memory_order_acquire)) {
      if (!batch.empty()) {
        emit_tokens(std::move(batch));
        batch.clear();
      }
      return {
          .kind = EventKind::aborted,
          .finish_reason = "cancelled",
      };
    }
    if (options_.fail_after_tokens >= 0 && i >= options_.fail_after_tokens) {
      if (!batch.empty()) {
        emit_tokens(std::move(batch));
        batch.clear();
      }
      return {
          .kind = EventKind::failed,
          .finish_reason = "error",
          .error = "injected failure after token emission",
      };
    }
    if (options_.step_delay_us > 0) {
      std::this_thread::sleep_for(
          std::chrono::microseconds(options_.step_delay_us));
    }
    const auto token = next_fake_token(base, i, options_.vocab_size);
    batch.push_back(token);
    if (batch.size() == static_cast<std::size_t>(options_.event_batch_size)) {
      emit_tokens(std::move(batch));
      batch.clear();
      batch.reserve(static_cast<std::size_t>(options_.event_batch_size));
    }
  }
  if (!batch.empty()) {
    emit_tokens(std::move(batch));
  }
  return {};
}

namespace {
class FakeTask final : public BackendTask {
public:
  FakeTask(FakeBackendOptions options, SubmitRequest request,
           std::shared_ptr<std::atomic<bool>> cancelled, EmitTokens emit)
      : options_(options), request_(std::move(request)),
        cancelled_(std::move(cancelled)), emit_(std::move(emit)),
        base_(request_.prompt_tokens.empty() ? 0 : request_.prompt_tokens.back()) {}
  std::optional<ExecutionResult> advance() override {
    if (cancelled_->load(std::memory_order_acquire))
      return ExecutionResult{.kind = EventKind::aborted,
                             .finish_reason = "cancelled"};
    if (options_.fail_before_first)
      return ExecutionResult{.kind = EventKind::failed,
                             .finish_reason = "error",
                             .error = "injected failure before first token"};
    std::vector<std::int32_t> batch;
    for (int count = 0; count < options_.event_batch_size &&
                        index_ < request_.max_new_tokens; ++count, ++index_) {
      if (cancelled_->load(std::memory_order_acquire)) {
        if (!batch.empty()) {
          emit_(std::move(batch));
          batch.clear();
        }
        return ExecutionResult{.kind = EventKind::aborted,
                               .finish_reason = "cancelled"};
      }
      if (options_.fail_after_tokens >= 0 && index_ >= options_.fail_after_tokens) {
        if (!batch.empty()) {
          emit_(std::move(batch));
          batch.clear();
        }
        return ExecutionResult{.kind = EventKind::failed,
                               .finish_reason = "error",
                               .error = "injected failure after token emission"};
      }
      if (options_.step_delay_us > 0)
        std::this_thread::sleep_for(
            std::chrono::microseconds(options_.step_delay_us));
      batch.push_back(next_fake_token(base_, index_, options_.vocab_size));
    }
    if (!batch.empty()) emit_(std::move(batch));
    if (index_ == request_.max_new_tokens) return ExecutionResult{};
    return std::nullopt;
  }
private:
  FakeBackendOptions options_;
  SubmitRequest request_;
  std::shared_ptr<std::atomic<bool>> cancelled_;
  EmitTokens emit_;
  std::int32_t base_ = 0, index_ = 0;
};
} // namespace
std::unique_ptr<BackendTask>
FakeBackend::start(SubmitRequest request,
                   std::shared_ptr<std::atomic<bool>> cancelled,
                   EmitTokens emit) {
  assert_owner();
  return std::make_unique<FakeTask>(options_, std::move(request),
                                    std::move(cancelled), std::move(emit));
}

void FakeBackend::populate_stats(StatsReport &stats) const {
  stats.backend_kind = "fake";
  stats.model_execution_kind = "fake";
  stats.kernel_family = "deterministic_fake";
}

} // namespace omlx::m5::detail
