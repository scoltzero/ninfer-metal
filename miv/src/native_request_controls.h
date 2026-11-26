#pragma once

#include "forward_plan.h"
#include "omlx_m5/types.h"

#include <deque>

namespace omlx::m5::detail {

struct RequestPenaltyToken {
  std::uint32_t token = 0;
  float repetition = 1.0F;
  float presence = 0.0F;
  float frequency = 0.0F;
};
static_assert(sizeof(RequestPenaltyToken) == 16);

// A request owns its history. Verification uses a copy advanced through only
// the draft prefix preceding each row and never commits a rejected suffix.
class RequestControlState {
public:
  RequestControlState(const SubmitRequest &request, std::uint32_t vocabulary)
      : sampling_(&request.sampling),
        budget_(request.thinking_budget ? &*request.thinking_budget : nullptr),
        window_(request.sampling.has_penalties()
                    ? std::max<std::size_t>(
                          20, request.sampling.repetition_penalty != 1.0F
                                  ? request.sampling.repetition_context_size
                                  : 20)
                    : 0),
        thinking_open_(budget_ && budget_->start_in_thinking) {
    if (budget_) {
      const auto valid = [vocabulary](const auto &ids) {
        return std::all_of(ids.begin(), ids.end(), [vocabulary](auto id) {
          return id >= 0 && std::uint32_t(id) < vocabulary;
        });
      };
      if (!valid(budget_->start_token_ids) || !valid(budget_->end_token_ids) ||
          !valid(budget_->force_token_ids)) {
        throw std::invalid_argument("thinking_budget token outside model vocabulary");
      }
      marker_width_ = std::max(budget_->start_token_ids.size(),
                               budget_->end_token_ids.size());
    }
    const auto size = request.prompt_tokens.size();
    const auto initial = std::min(size, window_);
    for (auto i = size - initial; i < size; ++i)
      history_.push_back(std::uint32_t(request.prompt_tokens[i]));
  }

  [[nodiscard]] std::optional<std::uint32_t> forced_token() const {
    if (!thinking_open_ ||
        thinking_count_ < std::max(0, budget_->budget - 1))
      return std::nullopt;
    const auto &sequence = force_sequence();
    return std::uint32_t(sequence.at(forced_index_));
  }

  void accept(std::int32_t token) {
    if (window_ != 0) {
      history_.push_back(std::uint32_t(token));
      if (history_.size() > window_)
        history_.pop_front();
    }
    if (!budget_)
      return;
    if (const auto forced = forced_token()) {
      // A mismatched draft invalidates all subsequent probe rows. On commit
      // only the deterministic closing token can advance this state.
      if (*forced != std::uint32_t(token))
        return;
      if (++forced_index_ == force_sequence().size())
        close_thinking();
      return;
    }
    recent_.push_back(token);
    if (recent_.size() > marker_width_)
      recent_.pop_front();
    if (thinking_open_) {
      if (matches(budget_->end_token_ids)) {
        close_thinking();
      } else {
        ++thinking_count_;
      }
    } else if (matches(budget_->start_token_ids)) {
      thinking_open_ = true;
      thinking_count_ = 0;
      recent_.clear();
    }
  }

  void append_penalties(std::vector<RequestPenaltyToken> &entries) const {
    if (history_.empty())
      return;
    const std::size_t n = history_.size();
    const std::size_t rep = sampling_->repetition_penalty == 1.0F
                                ? 0 : std::min<std::size_t>(
                                      n, sampling_->repetition_context_size);
    const std::size_t recent =
        sampling_->presence_penalty != 0.0F || sampling_->frequency_penalty != 0.0F
            ? std::min<std::size_t>(20, n) : 0;
    std::unordered_map<std::uint32_t, std::size_t> seen;
    seen.reserve(std::max(rep, recent));
    for (auto i = n - std::max(rep, recent); i < n; ++i) {
      const auto id = history_[i];
      auto [where, inserted] = seen.try_emplace(id, entries.size());
      if (inserted)
        entries.push_back({.token = id});
      auto &entry = entries[where->second];
      if (i >= n - rep)
        entry.repetition = sampling_->repetition_penalty;
      if (i >= n - recent) {
        entry.presence = sampling_->presence_penalty;
        entry.frequency += sampling_->frequency_penalty;
      }
    }
  }

  [[nodiscard]] std::size_t max_penalty_tokens() const { return window_; }

  // Returns true if this request may require penalty adjustment or a forced
  // thinking close within the next `lookahead` sampled tokens.
  [[nodiscard]] bool needs_intervention_within(std::size_t lookahead) const noexcept {
    if (window_ != 0)
      return true;
    if (!budget_)
      return false;
    const std::int64_t limit = std::max<std::int32_t>(0, budget_->budget - 1);
    if (thinking_open_)
      return std::int64_t(thinking_count_) + std::int64_t(lookahead) > limit;
    if (budget_->start_token_ids.empty())
      return false;
    return std::int64_t(lookahead) > limit + 1;
  }

private:
  [[nodiscard]] const std::vector<std::int32_t> &force_sequence() const {
    return budget_->force_token_ids.empty() ? budget_->end_token_ids
                                           : budget_->force_token_ids;
  }
  [[nodiscard]] bool matches(std::span<const std::int32_t> ids) const {
    return !ids.empty() && ids.size() <= recent_.size() &&
           std::equal(ids.rbegin(), ids.rend(), recent_.rbegin());
  }
  void close_thinking() {
    thinking_open_ = false;
    thinking_count_ = 0;
    forced_index_ = 0;
    recent_.clear();
  }

  const SamplingSpec *sampling_;
  const ThinkingBudget *budget_;
  std::size_t window_, marker_width_ = 0, forced_index_ = 0;
  bool thinking_open_;
  std::int32_t thinking_count_ = 0;
  std::deque<std::uint32_t> history_;
  std::deque<std::int32_t> recent_;
};

// The raw target logits stay untouched for checkpoints and future requests.
// This slow-feature-only path reads back selected tokens (or the existing
// small top-k tie data), never the full hot vocabulary.
class RequestGpuSampler {
public:
  RequestGpuSampler(const MetalContext &context,
                    const std::filesystem::path &libraries,
                    const MetalBuffer &raw, std::uint32_t vocabulary,
                    std::size_t max_penalties)
      : context_(context), libraries_(libraries), raw_(raw), vocabulary_(vocabulary),
        max_penalties_(max_penalties),
        plan_key_("position.request." +
                  std::to_string(reinterpret_cast<std::uintptr_t>(&raw))),
        scratch_(context.make_buffer(std::uint64_t(vocabulary) * 8U * 2U)),
        sampler_(context, scratch_, vocabulary, 8, libraries,
                 plan_key_ + ".sample") {
    if (max_penalties != 0) {
      penalties_.emplace(context.make_buffer(
          std::uint64_t(max_penalties) * 8U * sizeof(RequestPenaltyToken)));
    }
  }

  void ensure_penalty_capacity(std::size_t capacity) {
    if (capacity == 0 || (penalties_ && capacity <= max_penalties_))
      return;
    max_penalties_ = std::min<std::size_t>(
        std::max<std::size_t>(capacity, max_penalties_), vocabulary_);
    penalties_.emplace(context_.make_buffer(
        std::uint64_t(max_penalties_) * 8U * sizeof(RequestPenaltyToken)));
    for (auto &adjust : adjustments_)
      adjust.reset();
  }

  [[nodiscard]] PositionSamplingResult sample(
      std::span<const PositionSamplingRow> rows,
      std::span<const std::array<std::uint32_t, 2>> ranges,
      std::span<const RequestPenaltyToken> entries, std::uint32_t source_row) {
    auto &copy = copies_.at(rows.size() - 1);
    if (!copy)
      copy.emplace(make_copy(rows.size()));
    const auto width = std::uint32_t(rows.size());
    std::memcpy(copy->profile.parameter_data.data() + copy->source_offset,
                &source_row, sizeof(source_row));
    copy->profile.parameters.write("step", copy->profile.parameter_data);
    MetalSubmission pre = dispatch(copy->profile);
    if (!entries.empty()) {
      ensure_penalty_capacity(
          std::min<std::size_t>(kMaxRepetitionContextSize, vocabulary_));
      if (!penalties_ || entries.size_bytes() > penalties_->size())
        throw std::logic_error("request penalty staging exceeded");
      penalties_->write(std::as_bytes(entries));
      auto &adjust = adjustments_.at(width - 1);
      if (!adjust)
        adjust.emplace(make_adjust(width));
      std::memcpy(adjust->profile.parameter_data.data() + adjust->source_offset,
                  ranges.data(), ranges.size_bytes());
      adjust->profile.parameters.write("step", adjust->profile.parameter_data);
      const auto update = dispatch(adjust->profile);
      pre.dispatch_count += update.dispatch_count;
      pre.timing.gpu_ms += update.timing.gpu_ms;
      pre.timing.wall_ms += update.timing.wall_ms;
    }
    auto result = sampler_.sample(rows);
    result.submission.dispatch_count += pre.dispatch_count;
    result.submission.timing.gpu_ms += pre.timing.gpu_ms;
    result.submission.timing.wall_ms += pre.timing.wall_ms;
    return result;
  }

  [[nodiscard]] PositionSamplingResult sample_cached(
      const PositionSamplingRow &row, std::span<const std::byte> raw,
      std::span<const RequestPenaltyToken> entries) {
    if (raw.size_bytes() != std::uint64_t(vocabulary_) * 2U)
      throw std::logic_error("cached request anchor logits unavailable");
    scratch_.write(raw);
    MetalSubmission updated{};
    if (!entries.empty()) {
      ensure_penalty_capacity(
          std::min<std::size_t>(kMaxRepetitionContextSize, vocabulary_));
      if (!penalties_ || entries.size_bytes() > penalties_->size())
        throw std::logic_error("cached request penalty staging exceeded");
      penalties_->write(std::as_bytes(entries));
      auto &adjust = adjustments_.front();
      if (!adjust)
        adjust.emplace(make_adjust(1));
      const std::array<std::uint32_t, 2> range{0U, std::uint32_t(entries.size())};
      std::memcpy(adjust->profile.parameter_data.data() + adjust->source_offset,
                  range.data(), sizeof(range));
      adjust->profile.parameters.write("step", adjust->profile.parameter_data);
      updated = dispatch(adjust->profile);
    }
    auto result = sampler_.sample(std::span(&row, 1));
    result.submission.dispatch_count += updated.dispatch_count;
    result.submission.timing.gpu_ms += updated.timing.gpu_ms;
    result.submission.timing.wall_ms += updated.timing.wall_ms;
    return result;
  }

private:
  struct Kernel {
    forward::Profile profile;
    std::uint64_t source_offset;
  };

  [[nodiscard]] Kernel make_copy(std::size_t rows) {
    forward::Encoder program(context_, libraries_, std::uint32_t(rows),
                             plan_key_ + ".copy." + std::to_string(rows));
    auto &copy = program.emit("position_sampling", "omlx_m5_position_copy_raw_logits",
                              {.width = vocabulary_, .height = rows}, {.width = 256},
                              {{&raw_, 0, raw_.size()}, {&scratch_, 0, scratch_.size()}});
    program.constant(copy, 2, forward::words({vocabulary_, std::uint32_t(rows)}));
    const auto offset = program.dynamic(copy, 3, forward::words({0U}));
    return {program.finish(true), offset};
  }

  [[nodiscard]] Kernel make_adjust(std::size_t rows) {
    forward::Encoder program(context_, libraries_, std::uint32_t(rows),
                             plan_key_ + ".penalties." + std::to_string(rows));
    auto &update = program.emit("position_sampling", "omlx_m5_position_apply_sparse_penalties",
                                {.width = max_penalties_, .height = rows}, {.width = 32},
                                {{&scratch_, 0, scratch_.size()},
                                 {&*penalties_, 0, penalties_->size()}});
    const auto offset = program.dynamic(update, 2, std::vector<std::byte>(rows * 8U));
    program.constant(update, 3, forward::words({vocabulary_, std::uint32_t(rows)}));
    return {program.finish(true), offset};
  }

  [[nodiscard]] MetalSubmission dispatch(forward::Profile &profile) const {
    if (profile.icb)
      return context_.replay_icb(*profile.icb);
    const std::array calls{MetalCommandInvocation{
        .plan = &profile.plan, .bindings = profile.bindings, .parameters = &profile.parameters}};
    return context_.execute_batch(calls);
  }

  const MetalContext &context_;
  const std::filesystem::path &libraries_;
  const MetalBuffer &raw_;
  std::uint32_t vocabulary_;
  std::size_t max_penalties_;
  std::string plan_key_;
  MetalBuffer scratch_;
  std::optional<MetalBuffer> penalties_;
  PositionKeyedSampler sampler_;
  std::array<std::optional<Kernel>, 8> copies_{}, adjustments_{};
};

} // namespace omlx::m5::detail
