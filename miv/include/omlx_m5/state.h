#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace omlx::m5 {

struct GdnGeometry {
  std::uint32_t key_dim = 0;
  std::uint32_t value_dim = 0;
  float epsilon = 1.0e-6F;
};

struct GdnTransitionRecord {
  std::vector<float> key_raw;
  std::vector<float> value;
  float log_decay = 0.0F;
  float beta = 0.0F;
};

class GdnState final {
public:
  explicit GdnState(GdnGeometry geometry);
  GdnState(GdnGeometry geometry, std::vector<float> values);

  [[nodiscard]] const GdnGeometry &geometry() const;
  [[nodiscard]] const std::vector<float> &values() const;
  [[nodiscard]] std::vector<float> &mutable_values();
  void apply(const GdnTransitionRecord &record);

private:
  GdnGeometry geometry_;
  std::vector<float> values_;
};

[[nodiscard]] GdnTransitionRecord
make_represented_record(std::span<const float> key_raw,
                        std::span<const float> value, float log_decay,
                        float beta);
[[nodiscard]] GdnState
replay_prefix(const GdnState &checkpoint,
              std::span<const GdnTransitionRecord> records, std::size_t count);

enum class RoundStage {
  created,
  verified,
  licensed,
  committed,
  aborted,
  cancelled,
};

struct SequenceState {
  std::uint64_t target_frontier = 0;
  std::uint64_t context_frontier = 0;
  std::uint64_t generated_tokens = 0;
  std::uint64_t kv_page_tokens = 64;
  std::uint64_t kv_page_count = 0;
  std::int32_t pending_anchor = -1;
  GdnState gdn;

  SequenceState(std::uint64_t page_tokens, GdnState initial_state);
  void materialize_pending_context();
};

class RoundTransaction final {
public:
  RoundTransaction(SequenceState &sequence, std::uint32_t verify_width);
  RoundTransaction(const RoundTransaction &) = delete;
  RoundTransaction &operator=(const RoundTransaction &) = delete;
  ~RoundTransaction();

  void set_verify_records(std::vector<GdnTransitionRecord> records);
  void set_licensed_outputs(std::vector<std::int32_t> outputs);
  [[nodiscard]] std::vector<std::int32_t> commit(std::size_t count);
  void abort();
  void cancel();

  [[nodiscard]] RoundStage stage() const;
  [[nodiscard]] std::uint32_t verify_width() const;
  [[nodiscard]] const std::vector<GdnTransitionRecord> &records() const;
  [[nodiscard]] const std::vector<std::int32_t> &licensed_outputs() const;

private:
  void finish_without_commit(RoundStage final_stage);

  SequenceState *sequence_;
  std::uint32_t verify_width_;
  RoundStage stage_ = RoundStage::created;
  std::vector<GdnTransitionRecord> records_;
  std::vector<std::int32_t> licensed_outputs_;
};

} // namespace omlx::m5
