#include "omlx_m5/state.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "omlx_m5/quantization.h"

namespace omlx::m5 {

namespace {

float represented_bf16(float value) {
  return bf16_to_float(float_to_bf16(value));
}

void validate_record(const GdnGeometry &geometry,
                     const GdnTransitionRecord &record) {
  if (record.key_raw.size() != geometry.key_dim ||
      record.value.size() != geometry.value_dim) {
    throw std::invalid_argument("GDN transition record geometry mismatch");
  }
  if (!std::isfinite(record.log_decay) || !std::isfinite(record.beta)) {
    throw std::invalid_argument("GDN transition scalars must be finite");
  }
}

} // namespace

GdnState::GdnState(GdnGeometry geometry)
    : geometry_(geometry),
      values_(static_cast<std::size_t>(geometry.key_dim) * geometry.value_dim,
              0.0F) {
  if (geometry_.key_dim == 0 || geometry_.value_dim == 0 ||
      geometry_.epsilon <= 0.0F) {
    throw std::invalid_argument("invalid GDN geometry");
  }
}

GdnState::GdnState(GdnGeometry geometry, std::vector<float> values)
    : geometry_(geometry), values_(std::move(values)) {
  if (geometry_.key_dim == 0 || geometry_.value_dim == 0 ||
      geometry_.epsilon <= 0.0F ||
      values_.size() !=
          static_cast<std::size_t>(geometry_.key_dim) * geometry_.value_dim) {
    throw std::invalid_argument("invalid GDN state geometry or storage");
  }
}

const GdnGeometry &GdnState::geometry() const { return geometry_; }
const std::vector<float> &GdnState::values() const { return values_; }
std::vector<float> &GdnState::mutable_values() { return values_; }

void GdnState::apply(const GdnTransitionRecord &record) {
  validate_record(geometry_, record);
  float norm_squared = 0.0F;
  for (const auto value : record.key_raw) {
    norm_squared = std::fma(value, value, norm_squared);
  }
  const auto inverse_norm = 1.0F / std::sqrt(norm_squared + geometry_.epsilon);
  std::vector<float> key(geometry_.key_dim);
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = record.key_raw[index] * inverse_norm;
  }
  const auto alpha = std::exp(record.log_decay);
  for (std::uint32_t value_index = 0; value_index < geometry_.value_dim;
       ++value_index) {
    auto *row = values_.data() +
                static_cast<std::size_t>(value_index) * geometry_.key_dim;
    float correction = 0.0F;
    for (std::uint32_t key_index = 0; key_index < geometry_.key_dim;
         ++key_index) {
      row[key_index] *= alpha;
      correction = std::fma(row[key_index], key[key_index], correction);
    }
    const auto update = record.beta * (record.value[value_index] - correction);
    for (std::uint32_t key_index = 0; key_index < geometry_.key_dim;
         ++key_index) {
      row[key_index] = std::fma(update, key[key_index], row[key_index]);
    }
  }
}

GdnTransitionRecord make_represented_record(std::span<const float> key_raw,
                                            std::span<const float> value,
                                            float log_decay, float beta) {
  GdnTransitionRecord record;
  record.key_raw.reserve(key_raw.size());
  record.value.reserve(value.size());
  std::ranges::transform(key_raw, std::back_inserter(record.key_raw),
                         represented_bf16);
  std::ranges::transform(value, std::back_inserter(record.value),
                         represented_bf16);
  record.log_decay = log_decay;
  record.beta = beta;
  return record;
}

GdnState replay_prefix(const GdnState &checkpoint,
                       std::span<const GdnTransitionRecord> records,
                       std::size_t count) {
  if (count > records.size()) {
    throw std::out_of_range("ReplaySSM prefix exceeds record count");
  }
  auto state = checkpoint;
  for (std::size_t index = 0; index < count; ++index) {
    state.apply(records[index]);
  }
  return state;
}

SequenceState::SequenceState(std::uint64_t page_tokens, GdnState initial_state)
    : kv_page_tokens(page_tokens), gdn(std::move(initial_state)) {
  if (kv_page_tokens == 0) {
    throw std::invalid_argument("KV page size must be positive");
  }
}

void SequenceState::materialize_pending_context() {
  context_frontier = target_frontier;
}

RoundTransaction::RoundTransaction(SequenceState &sequence,
                                   std::uint32_t verify_width)
    : sequence_(&sequence), verify_width_(verify_width) {
  if (verify_width_ < 1 || verify_width_ > 8) {
    throw std::invalid_argument("verify width must be in [1, 8]");
  }
}

RoundTransaction::~RoundTransaction() {
  if (stage_ != RoundStage::committed && stage_ != RoundStage::aborted &&
      stage_ != RoundStage::cancelled) {
    abort();
  }
}

void RoundTransaction::set_verify_records(
    std::vector<GdnTransitionRecord> records) {
  if (stage_ != RoundStage::created || records.size() != verify_width_) {
    throw std::logic_error("verify records are invalid for this transaction");
  }
  for (const auto &record : records) {
    validate_record(sequence_->gdn.geometry(), record);
  }
  records_ = std::move(records);
  stage_ = RoundStage::verified;
}

void RoundTransaction::set_licensed_outputs(std::vector<std::int32_t> outputs) {
  if (stage_ != RoundStage::verified || outputs.empty() ||
      outputs.size() > records_.size()) {
    throw std::logic_error("licensed outputs are invalid for this transaction");
  }
  licensed_outputs_ = std::move(outputs);
  stage_ = RoundStage::licensed;
}

std::vector<std::int32_t> RoundTransaction::commit(std::size_t count) {
  if (stage_ != RoundStage::licensed || count > licensed_outputs_.size()) {
    throw std::logic_error("transaction commit length is invalid");
  }
  if (count == 0) {
    finish_without_commit(RoundStage::aborted);
    return {};
  }
  sequence_->gdn = replay_prefix(sequence_->gdn, records_, count);
  sequence_->target_frontier += count;
  sequence_->generated_tokens += count;
  sequence_->kv_page_count =
      (sequence_->target_frontier + sequence_->kv_page_tokens - 1) /
      sequence_->kv_page_tokens;
  sequence_->pending_anchor = licensed_outputs_[count - 1];
  stage_ = RoundStage::committed;
  return {licensed_outputs_.begin(),
          licensed_outputs_.begin() + static_cast<std::ptrdiff_t>(count)};
}

void RoundTransaction::finish_without_commit(RoundStage final_stage) {
  if (stage_ == RoundStage::committed || stage_ == RoundStage::aborted ||
      stage_ == RoundStage::cancelled) {
    throw std::logic_error("transaction is already terminal");
  }
  stage_ = final_stage;
}

void RoundTransaction::abort() { finish_without_commit(RoundStage::aborted); }

void RoundTransaction::cancel() {
  finish_without_commit(RoundStage::cancelled);
}

RoundStage RoundTransaction::stage() const { return stage_; }
std::uint32_t RoundTransaction::verify_width() const { return verify_width_; }
const std::vector<GdnTransitionRecord> &RoundTransaction::records() const {
  return records_;
}
const std::vector<std::int32_t> &RoundTransaction::licensed_outputs() const {
  return licensed_outputs_;
}

} // namespace omlx::m5
