#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/row_exact.h"
#include "omlx_m5/types.h"
#include "omlx_m5/vocab.h"
#include "omlx_m5/attention.h"

namespace omlx::m5 {

namespace detail {
class ChunkBlendCache;
struct BlendSegmentConfig;
} // namespace detail

class KvPagePool;
class QwenForward;
struct QwenMergedLoraOverrides;
struct QwenBypassLoraRuntime;

struct QwenPrefillOptions {
  PrefillLinearMode linear_mode = PrefillLinearMode::BFloat16;
  PrefillAttentionMode attention_mode = PrefillAttentionMode::ExactNax;
  bool sparse_minference = false;
  bool cache_blend = false;
  bool ane_offload = false;
  PrefillAneMode ane_mode = PrefillAneMode::Off;
  std::string artifact_digest;
  // Optional override for Dual-ANE fixed chunk tokens (0 => default 1024).
  std::uint32_t ane_chunk_tokens = 0;
};

struct QwenPrefillTelemetry {
  std::uint64_t prefill_w4a8_tokens = 0;
  std::uint64_t prefill_w4a4_tokens = 0;
  std::uint64_t prefill_sage2_tokens = 0;
  std::uint64_t prefill_minference_blocks_skipped = 0;
  std::uint64_t prefill_minference_blocks_total = 0;
  bool prefill_ane_active = false;
  double prefill_ane_split_alpha = 0.0;
  std::string prefill_ane_mode_effective = "off";
  std::string prefill_ane_inactive_reason;
  std::uint64_t prefill_ane_offload_tokens = 0;
  std::uint64_t prefill_ane_fallback_tokens = 0;
  std::uint64_t prefill_cache_blend_hit_tokens = 0;
  std::uint64_t prefill_cache_blend_recomputed_tokens = 0;
};

// P02-TF controls Linear arithmetic only. Attention/GDN/sampling qualification
// is separate; selecting this does not claim a whole-Program exact contract.
enum class LinearNumericPolicy { Legacy, RowExactPortableV1 };

enum class QwenKvDtype { BF16, Affine4Group128 };

struct QwenBufferPlan {
  std::string name;
  std::uint64_t bytes = 0;
  bool state = false;
  bool clear_on_reset = false;
};

struct QwenForwardMemoryPlan {
  std::vector<QwenBufferPlan> buffers;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t state_bytes = 0;
  std::uint64_t largest_buffer_bytes = 0;
  std::uint64_t sage2_kv_workspace_bytes = 0;
  std::uint32_t feature_ring_rows = 0;
  std::uint32_t workspace_rows = 1024;
  [[nodiscard]] std::uint64_t buffer_bytes() const;
};

// The same allocation list is consumed by the executor and the preflight.
[[nodiscard]] std::uint32_t
resolve_qwen_primary_workspace_rows(std::uint32_t max_context,
                                    bool inspect_tensors = false,
                                    bool sparse_moe = false) noexcept;

[[nodiscard]] QwenForwardMemoryPlan
plan_qwen_forward_memory(const QwenModelConfig &, std::uint32_t max_context,
                         std::uint32_t feature_count = 0,
                         bool inspect_routes = false, bool speculative = false,
                         bool split_attention = false,
                         bool position_sampling = false,
                         bool exact_attention = false,
                         bool inspect_tensors = false,
                         std::uint32_t feature_ring_rows = 0,
                         std::uint32_t workspace_rows = 1024,
                         bool affine4_kv = false,
                         std::uint32_t kv_pool_pages = 0,
                         bool flash_nax_prefill = false,
                         bool enable_w4a8_workspace = false,
                         bool enable_minference_workspace = false,
                         bool enable_sage2_kv_workspace = false,
                         bool enable_w4a4_workspace = false);

struct QwenLoadBudget {
  std::uint64_t required_bytes = 0;
  std::uint64_t available_bytes = 0;
  // Reservation, not a measured allocation: pipelines, ICBs and parameter
  // tables.
  std::uint64_t runtime_reserve_bytes = 64ULL * 1024 * 1024;
  bool fits = false;
};
[[nodiscard]] QwenLoadBudget qwen_load_budget(std::uint64_t weight_bytes,
                                              const QwenForwardMemoryPlan &,
                                              std::uint64_t recommended_bytes,
                                              std::uint64_t allocated_bytes);

struct QwenProfileStage {
  std::string name;
  std::uint32_t dispatch_count = 0;
};
struct QwenProfileWeightRoute {
  std::string stage, object, format, library, function, family, source,
      fallback_reason;
  std::uint32_t n = 0, k = 0, capacity = 0;
  MetalDispatchSize grid, threads_per_threadgroup;
  std::string numeric_contract_id = "legacy_route_v1";
  std::string row_exact_group = {};
  bool row_exact_eligible = false;
};
// P12 exact diagnostic only. Never a production fallback threshold.
struct QwenMoeRouteOverride {
  std::string object, format, layout, choice;
  std::uint32_t n = 0, k = 0, capacity = 0;
};
// Diagnostic A/B on one exact weight and one installed profile. The override
// is rejected off Apple M4 and cannot silently change another shape or width.
struct QwenLinearRouteOverride {
  std::string object, format, choice;
  std::uint32_t n = 0, k = 0, capacity = 0;
};
struct QwenProfileDescription {
  std::uint32_t capacity = 0;
  bool verify = false;
  std::uint64_t dispatch_count = 0;
  std::uint64_t barrier_count = 0;
  std::vector<QwenProfileStage> stages;
  std::vector<QwenProfileWeightRoute> weights;
  bool split_attention = false;
  std::uint32_t split_count = 0;
  std::uint32_t context_envelope = 0;
  std::uint64_t split_workspace_bytes = 0;
  std::string attention_kernel, attention_merge_kernel;
  MetalDispatchSize attention_grid, attention_merge_grid;
  std::string numeric_contract_id = "legacy_route_v1";
  bool whole_program_row_exact_qualified = false;
  std::string sampling_contract_id = "legacy_rejection_v1";
  std::string attention_numeric_contract_id = "legacy_attention_v1";
  bool attention_row_exact_eligible = false;
  std::string gdn_numeric_contract_id = "gdn.sequential.simd32.f32_state.conv_bf16.v1";
  bool gdn_row_exact_eligible = false;
};
struct QwenStageTiming {
  std::string stage;
  double gpu_ms = 0.0, wall_ms = 0.0;
  std::uint64_t dispatch_count = 0;
};

struct QwenMoeRouteSnapshot {
  std::uint32_t layer = 0, tokens = 0, start_position = 0;
  std::uint64_t readback_bytes = 0;
  std::vector<std::uint32_t> expert_ids;
  std::vector<std::uint16_t> scores_bf16;
};

struct QwenForwardResult {
  std::uint32_t token = 0;
  std::uint32_t tokens = 0;
  std::uint32_t executed_layers = 0;
  std::uint64_t host_read_bytes = 0;
  std::uint32_t short_matrix_calls = 0;
  std::uint32_t short_lane_calls = 0;
  MetalSubmission submission;
  // Diagnostic ICB range replay adds a commit per stage. These timings are not
  // interchangeable with the normal one-commit production profile.
  std::vector<QwenStageTiming> stage_timings;
  // Populated only by explicit numerical inspection; serving reads one token.
  std::vector<float> hidden;
  std::vector<float> logits;
  std::vector<float> layer_features;
  // Explicit inspection of the last GDN call's scratch tensors.
  std::map<std::string, std::vector<float>> gdn_inspection;
  std::vector<std::uint32_t> layer_routes;
  std::vector<float> router_logits;
  std::vector<float> router_scores;
  QwenPrefillTelemetry prefill_delta;
};

// A complete, committed C1 continuation. The KV lease pins GPU pages;
// recurrent and feature bytes are bounded host snapshots. Only the owning
// QwenForward may restore or release it via restore()/release(), before that
// owner is destroyed; cross-slot COW sharing uses restore_shared() or
// QwenSharedPrefixState.
struct QwenContinuation {
  std::string execution_identity;
  std::string payload_digest;
  std::uint64_t owner_id = 0;
  std::uint32_t position = 0;
  std::uint64_t kv_lease = 0;
  std::vector<std::vector<std::byte>> recurrent;
  std::vector<std::byte> feature_history;
  std::uint64_t accounted_bytes = 0;
};

// Zero-move UMA-resident C1 continuation: GDN recurrent states and active
// feature_history live in a single pooled MTLStorageModeShared GPU MetalBuffer
// copied via one GPU command buffer (zero CPU memcpy, zero software SHA-256),
// while KV pages are retained in KvPagePool by zero-copy physical page lease.
struct QwenUmaContinuation {
  std::string execution_identity;
  std::uint64_t owner_id = 0;
  std::uint64_t uma_epoch_id = 0;
  std::uint32_t position = 0;
  std::uint64_t kv_lease = 0;
  std::vector<std::uint32_t> kv_pages;
  std::uint64_t recurrent_bytes = 0;
  std::uint64_t feature_bytes = 0;
  std::uint64_t slab_bytes = 0;
  std::shared_ptr<MetalBuffer> uma_slab;
  std::uint64_t accounted_bytes = 0;
};

struct QwenSharedPrefixState {
  std::string execution_identity;
  std::uint32_t position = 0;
  std::vector<std::uint32_t> kv_pages;
  std::vector<std::vector<std::byte>> kv_blocks;
  std::vector<std::vector<std::byte>> recurrent;
  std::vector<std::byte> feature_history;
  std::uint64_t accounted_bytes = 0;
};

struct PackedSlotSlice {
  QwenForward *slot_program = nullptr;
  std::uint32_t slot_index = 0;
  std::span<const std::uint32_t> tokens{};
  GpuSamplingOptions sampling{};
  bool verify = false;
};

struct PackedBatchResult {
  MetalSubmission submission;
  std::vector<QwenForwardResult> slot_results;
  std::uint64_t sampling_host_read_bytes = 0;
  std::uint32_t compact_fallback_rows = 0;
};
using QwenPackedBatchResult = PackedBatchResult;

// One prepared row in a P10 multi-invocation command buffer. Destruction
// rolls back a reserved KV append if the batch was not completed.
class QwenPreparedStep final {
public:
  QwenPreparedStep(QwenPreparedStep &&) noexcept;
  QwenPreparedStep &operator=(QwenPreparedStep &&) noexcept;
  QwenPreparedStep(const QwenPreparedStep &) = delete;
  QwenPreparedStep &operator=(const QwenPreparedStep &) = delete;
  ~QwenPreparedStep();
  [[nodiscard]] MetalCommandInvocation invocation() const;
  [[nodiscard]] MetalIcbProfile *icb_profile() const;
  [[nodiscard]] std::uint32_t width() const;
  [[nodiscard]] bool verifying() const;
private:
  struct Impl;
  friend class QwenForward;
  explicit QwenPreparedStep(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

// Move-only handle for a pipelined T=1 pure-AR GPU decode step submitted via
// replay_icb_async; destructor safely drains the in-flight submission if unharvested.
class QwenAsyncArStep final {
public:
  QwenAsyncArStep() noexcept;
  QwenAsyncArStep(QwenAsyncArStep &&) noexcept;
  QwenAsyncArStep &operator=(QwenAsyncArStep &&) noexcept;
  QwenAsyncArStep(const QwenAsyncArStep &) = delete;
  QwenAsyncArStep &operator=(const QwenAsyncArStep &) = delete;
  ~QwenAsyncArStep();
  [[nodiscard]] bool valid() const noexcept;
private:
  struct Impl;
  friend class QwenForward;
  explicit QwenAsyncArStep(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

// Move-only handle for a double-buffered trunk-only prefill chunk submitted via
// replay_icb_async; destructor safely drains the in-flight submission if unharvested.
class QwenAsyncPrefillStep final {
public:
  QwenAsyncPrefillStep() noexcept;
  QwenAsyncPrefillStep(QwenAsyncPrefillStep &&) noexcept;
  QwenAsyncPrefillStep &operator=(QwenAsyncPrefillStep &&) noexcept;
  QwenAsyncPrefillStep(const QwenAsyncPrefillStep &) = delete;
  QwenAsyncPrefillStep &operator=(const QwenAsyncPrefillStep &) = delete;
  ~QwenAsyncPrefillStep();
  [[nodiscard]] bool valid() const noexcept;
  [[nodiscard]] std::uint32_t width() const noexcept;
private:
  struct Impl;
  friend class QwenForward;
  explicit QwenAsyncPrefillStep(std::unique_ptr<Impl>);
  std::unique_ptr<Impl> impl_;
};

// Fixed Qwen mathematics over stored weight planes. Context and weights must
// outlive this owner. Mutable state and all profiles are private to this
// object.
class QwenForward final {
public:
  QwenForward(const MetalContext &, const MaterializedModel &,
              const QwenLayerCatalog &, QwenModelConfig,
              std::uint32_t max_context,
              const std::filesystem::path &library_directory,
              std::span<const std::uint32_t> feature_layers = {},
              bool inspect_routes = false, bool speculative = false,
              bool enable_m4_short_routes = false,
              std::optional<QwenLinearRouteOverride> diagnostic_override = {},
              bool enable_attention_split = false,
              bool enable_batch = false,
              std::optional<QwenMoeRouteOverride> moe_override = {},
              LinearNumericPolicy linear_policy = LinearNumericPolicy::Legacy,
              SamplingContract sampling_contract = SamplingContract::LegacyRejectionV1,
              AttentionNumericPolicy attention_policy = AttentionNumericPolicy::Legacy,
              bool inspect_tensors_workspace = false,
              std::string_view kv_dtype = "bf16",
              QwenForward *shared_primary = nullptr,
              bool enable_flash_nax_prefill = false,
              std::uint32_t kv_pool_pages = 0,
              std::shared_ptr<const QwenMergedLoraOverrides> merged_lora = nullptr,
              std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora = nullptr,
              QwenPrefillOptions prefill_options = {});
  ~QwenForward();
  QwenForward(QwenForward &&) noexcept;
  QwenForward &operator=(QwenForward &&) noexcept;
  QwenForward(const QwenForward &) = delete;
  QwenForward &operator=(const QwenForward &) = delete;

  [[nodiscard]] QwenForwardResult run(std::span<const std::uint32_t>,
                                      const GpuSamplingOptions & = {},
                                      bool inspect_tensors = false,
                                      bool use_icb = true,
                                      bool profile_stages = false);
  [[nodiscard]] QwenForwardResult
  run_trunk(std::span<const std::uint32_t> tokens, bool use_icb = true);
  [[nodiscard]] QwenForwardResult run_blend_prefill(
      std::span<const std::uint32_t> prompt, std::uint32_t prefix_tokens,
      detail::ChunkBlendCache &blend_cache, std::uint64_t model_exec_hash,
      const GpuSamplingOptions &sampling = {}, bool inspect_tensors = false,
      const detail::BlendSegmentConfig *seg_cfg = nullptr);
  [[nodiscard]] QwenPrefillTelemetry prefill_telemetry() const noexcept;
  [[nodiscard]] bool ane_offload_active() const noexcept;
  [[nodiscard]] double ane_split_alpha() const noexcept;
  [[nodiscard]] std::uint32_t ane_chunk_tokens() const noexcept;
  [[nodiscard]] bool cache_blend_eligible() const noexcept;
  [[nodiscard]] bool can_pipeline_prefill(std::size_t count) const noexcept;
  [[nodiscard]] QwenAsyncPrefillStep
  enqueue_prefill_step(std::span<const std::uint32_t> tokens);
  [[nodiscard]] QwenForwardResult
  finish_prefill_step(QwenAsyncPrefillStep &&step);
  void drain_prefill_step(QwenAsyncPrefillStep &&step);
  [[nodiscard]] bool
  can_pipeline_ar(const GpuSamplingOptions &sampling) const noexcept;
  [[nodiscard]] std::uint32_t ar_virtual_position() const noexcept;
  [[nodiscard]] QwenAsyncArStep
  enqueue_ar_step(const GpuSamplingOptions &sampling, bool seed_first = false,
                  std::uint32_t first_token = 0);
  [[nodiscard]] QwenForwardResult finish_ar_step(QwenAsyncArStep &&step);
  void drain_ar_step(QwenAsyncArStep &&step);
  // Cross-slot packed shared-weight execution (M = sum(tokens.size())).
  // Reads all shared model weights once while updating each slice's slot state.
  // When scatter_slot_logits == false, callers can read contiguous packed
  // logits directly from packed_logits_buffer().
  [[nodiscard]] PackedBatchResult
  run_packed_batch(std::span<const PackedSlotSlice> slices,
                   bool use_icb = true,
                   bool scatter_slot_logits = true);
  // Single-CB cross-slot packed AR forward + in-ICB batched GPU sampling over
  // primary's contiguous logits[0..B-1] (skips per-slot last_logits scatter).
  [[nodiscard]] PackedBatchResult
  run_packed_batch_and_sample(std::span<const PackedSlotSlice> slices,
                              bool use_icb = true);
  [[nodiscard]] const MetalBuffer &packed_logits_buffer() const;
  // P10 multi-row submit: call prepare for every row, submit all invocations
  // once on the shared context, then finish each row. W>8 uses the old path.
  [[nodiscard]] QwenPreparedStep
  prepare_batch(std::span<const std::uint32_t>,
                const GpuSamplingOptions & = {}, bool verify = false);
  [[nodiscard]] QwenForwardResult finish_batch(QwenPreparedStep &&,
                                                std::uint32_t sampled_token = 0);
  [[nodiscard]] const MetalBuffer &batch_logits_buffer() const;
  // Internal diagnostic projection views over actual final-normalized rows.
  [[nodiscard]] const MetalBuffer &batch_projection_input_buffer() const;
  [[nodiscard]] const MetalBuffer &batch_projection_output_buffer() const;
  [[nodiscard]] std::uint32_t vocabulary_size() const;
  [[nodiscard]] QwenForwardResult verify(std::span<const std::uint32_t>,
                                         const GpuSamplingOptions & = {},
                                         bool inspect_tensors = false,
                                         bool profile_stages = false);
  [[nodiscard]] bool can_enqueue_verify(std::size_t count) const noexcept;
  [[nodiscard]] MetalPendingSubmission
  enqueue_verify(std::span<const std::uint32_t> tokens,
                 const GpuSamplingOptions &sampling = {});
  [[nodiscard]] QwenForwardResult
  finish_verify(MetalPendingSubmission &&pending, std::uint32_t count);
  [[nodiscard]] std::shared_ptr<const Q5High1Cache>
  q5_high1_cache() const noexcept;
  [[nodiscard]] std::optional<RowExactPlaneSlice>
  q5_high1_plane(std::uint32_t object_id) const;
  [[nodiscard]] std::optional<RowExactPlaneSlice>
  affine_sb_tiled_plane(std::uint32_t object_id) const;
  [[nodiscard]] std::optional<ResolvedKl32Mxfp4Planes>
  resolve_kl32_mxfp4(const MaterializedWeightView &w) const;
  [[nodiscard]] std::optional<ResolvedKl32AffinePlanes>
  resolve_kl32_affine(const MaterializedWeightView &w) const;
  [[nodiscard]] QwenProfileDescription
  inspect_profile(std::uint32_t tokens, bool verify = false) const;
  [[nodiscard]] QwenProfileDescription
  inspect_split_profile(std::uint32_t tokens, std::uint32_t context,
                        bool verify = false) const;
  // Explicit diagnostic readback for small fixture state comparisons only.
  [[nodiscard]] std::vector<std::byte>
  inspect_state(std::uint64_t max_bytes = 16ULL << 20) const;
  [[nodiscard]] MetalSubmission commit_prefix(std::uint32_t count);
  [[nodiscard]] MetalIcbProfile *prepare_commit_prefix(std::uint32_t count);
  [[nodiscard]] MetalSubmission finish_commit_prefix(std::uint32_t count,
                                                     bool already_executed);
  void abort_verify();
  [[nodiscard]] const MetalBuffer &feature_history() const;
  [[nodiscard]] std::uint64_t feature_epoch() const noexcept;
  [[nodiscard]] std::uint32_t feature_ring_rows() const noexcept;
  [[nodiscard]] KvPagePool *kv_page_pool() noexcept;
  [[nodiscard]] const KvPagePool *kv_page_pool() const noexcept;
  void share_prefill_workspace(QwenForward &donor);
  void share_kv_arena(QwenForward &primary);
  [[nodiscard]] const MaterializedWeightView &embedding_weight() const;
  [[nodiscard]] const MaterializedWeightView &head_weight() const;
  [[nodiscard]] std::vector<float> read_logits(std::uint32_t count) const;
  [[nodiscard]] std::vector<std::byte> read_last_logits_bf16() const;
  // Shared BF16 verify output; valid after verify and before commit/abort.
  [[nodiscard]] const MetalBuffer &verification_logits_buffer() const;
  [[nodiscard]] std::uint32_t last_logits_row() const;
  [[nodiscard]] const MetalBuffer &normed_hidden_buffer() const;
  [[nodiscard]] const MetalBuffer &trunk_hidden_buffer() const;
  [[nodiscard]] const MaterializedWeightView &final_norm_weight() const;
  [[nodiscard]] float normalization_weight_offset() const noexcept;
  // Test-only state inspection, never used by the serving implementation.
  [[nodiscard]] std::vector<std::byte> inspect_recurrent_state() const;
  [[nodiscard]] std::vector<std::uint32_t> inspect_kv_page_table() const;
  // Explicit route-only diagnostic after a committed run. Reads route IDs and
  // raw BF16 scores, never full hidden/router logits in the serving loop.
  [[nodiscard]] QwenMoeRouteSnapshot
  read_moe_route_trace(std::uint32_t layer) const;
  // Diagnostic KV-only lease. It does not snapshot GDN/conv/RNG and is not a
  // reusable prefix checkpoint; P08 owns that complete contract.
  [[nodiscard]] std::uint64_t diagnostic_retain_kv_pages();
  void diagnostic_release_kv_pages(std::uint64_t lease);
  [[nodiscard]] std::uint64_t checkpoint_bytes() const;
  [[nodiscard]] std::uint64_t non_kv_checkpoint_bytes() const;
  [[nodiscard]] std::uint64_t kv_page_all_layers_bytes() const;
  [[nodiscard]] QwenContinuation checkpoint() const;
  [[nodiscard]] QwenUmaContinuation checkpoint_uma() const;
  // Bind once at installation, before the first forward or checkpoint.
  void bind_execution_identity(std::string);
  [[nodiscard]] std::string_view execution_identity() const;
  void set_bypass_lora(std::shared_ptr<const QwenBypassLoraRuntime> bypass_lora,
                       std::string context_execution_identity = {});
  [[nodiscard]] std::uint64_t lora_scratch_bytes() const noexcept;
  void restore(const QwenContinuation &);
  void restore_shared(const QwenContinuation &,
                      QwenForward *source_owner = nullptr);
  void restore_uma(const QwenUmaContinuation &);
  void restore_shared_uma(const QwenUmaContinuation &,
                          QwenForward *source_owner = nullptr);
  [[nodiscard]] QwenSharedPrefixState
  export_shared_prefix(bool include_recurrent = true) const;
  void import_shared_prefix(const QwenSharedPrefixState &);
  void release(QwenContinuation &);
  void release_uma(QwenUmaContinuation &);
  void trim_uma_checkpoint_pool(std::size_t max_pooled_slabs = 0);
  void reset();
  [[nodiscard]] std::uint32_t position() const;
  [[nodiscard]] const QwenForwardMemoryPlan &memory_plan() const;

private:
  friend class QwenAsyncArStep;
  friend class QwenAsyncPrefillStep;
  QwenForwardResult execute(std::span<const std::uint32_t>,
                            const GpuSamplingOptions &, bool, bool, bool, bool);
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
