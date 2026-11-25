#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/topology_profile.h"

namespace omlx::m5 {

struct QwenProgramBuildOptions {
  std::uint32_t max_context = 0;
  std::uint32_t kv_page_tokens = 64;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t state_bytes = 0;
};

struct QwenProgramBuildPlan {
  QwenArchitecture architecture = QwenArchitecture::dense;
  QwenQuantProfile quant_profile = QwenQuantProfile::oq_affine;
  std::string topology_signature;
  std::uint32_t max_context = 0;
  std::uint32_t kv_page_tokens = 64;
  std::uint32_t layer_count = 0;
  std::uint32_t full_attention_layers = 0;
  std::uint32_t gdn_layers = 0;
  std::uint32_t hidden_size = 0;
  std::uint32_t vocabulary_size = 0;
  std::uint32_t expert_count = 0;
  std::uint32_t expert_top_k = 0;
  std::uint64_t stage_count = 0;
  std::uint64_t dispatch_count = 0;
  std::uint32_t profile_count = 0;
  std::uint64_t weight_segment_count = 0;
  std::uint64_t weight_bytes = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t state_bytes = 0;
  std::uint64_t parameter_bytes = 0;
  std::uint64_t residency_allocation_count = 0;
  std::vector<std::uint32_t> required_object_ids;
  DispatchTableStats dispatch_table;
};

// Builds the immutable, planned route catalog for every artifact-backed
// linear stage and every compiled topology capacity. Executors remain
// uninstalled until B4.4 wires the concrete B2/B3 plans.
[[nodiscard]] DispatchTable
build_qwen_program_dispatch_table(const ArtifactModel &artifact,
                                  const QwenLayerCatalog &layers,
                                  const TopologyProfileCatalog &profiles);

[[nodiscard]] QwenProgramBuildPlan plan_qwen_program(
    const ArtifactModel &artifact, const QwenModelConfig &config,
    const ModelMaterializationPlan &materialization,
    const QwenLayerCatalog &layers, const TopologyProfileCatalog &profiles,
    DispatchTableStats dispatch_table, QwenProgramBuildOptions options = {});

struct QwenProgramPlaneBinding {
  std::string kind;
  std::uint32_t segment_index = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t gpu_address = 0;

  bool operator==(const QwenProgramPlaneBinding &) const = default;
};

struct QwenProgramStagePlan {
  std::uint32_t layer = UINT32_MAX;
  LayerStageKind kind = LayerStageKind::linear;
  std::string role;
  std::uint32_t object_id = UINT32_MAX;
  std::uint32_t dispatches = 0;
  std::vector<QwenProgramPlaneBinding> planes;
};

struct QwenSequenceRuntimeState {
  std::uint32_t position = 0;
  std::uint32_t kv_frontier = 0;
  std::uint32_t page_count = 0;
  std::uint64_t generated_tokens = 0;
  std::uint64_t parameter_generation = 0;
};

struct QwenDynamicStep {
  std::span<const std::uint32_t> token_ids;
  std::span<const std::uint32_t> positions;
  std::uint32_t kv_frontier = 0;
  std::span<const std::uint32_t> page_table;
  std::span<const std::uint32_t> route_indices;
  std::span<const float> route_scores;
  std::span<const std::uint32_t> valid_mask;
  std::array<float, 4> sampling{1.0F, 1.0F, 1.0F, 64.0F};
  std::array<std::uint32_t, 16> kernel_params{};
};

struct QwenProgramAddressSnapshot {
  std::vector<std::uint64_t> weight_segments;
  std::uint64_t workspace = 0;
  std::uint64_t state = 0;
  std::vector<std::uint64_t> parameter_tables;

  bool operator==(const QwenProgramAddressSnapshot &) const = default;
};

struct QwenProgramStats {
  std::uint64_t static_stage_plan_count = 0;
  std::uint64_t static_plane_binding_count = 0;
  std::uint64_t profile_count = 0;
  std::uint64_t parameter_table_count = 0;
  std::uint64_t residency_allocations = 0;
  std::uint64_t residency_bytes = 0;
  std::uint64_t construction_artifact_lookups = 0;
  std::uint64_t token_loop_artifact_lookups = 0;
  std::uint64_t token_loop_pipeline_compiles = 0;
  std::uint64_t token_loop_weight_uploads = 0;
};

enum class QwenArExecutionRoute : std::uint8_t {
  execute_batch,
  icb_profile,
};

[[nodiscard]] std::string_view to_string(QwenArExecutionRoute route);

struct QwenArSamplingOptions {
  float temperature = 0.0F;
  std::uint32_t top_k = 0;
  float top_p = 1.0F;
  std::uint32_t seed = 0;
};

struct QwenArStepResult {
  std::uint32_t sampled_token = 0;
  std::vector<std::uint32_t> sampled_tokens;
  std::vector<float> hidden;
  std::vector<float> logits;
  std::uint32_t input_tokens = 0;
  std::uint32_t profile_index = UINT32_MAX;
  std::uint32_t profile_capacity = 0;
  QwenArExecutionRoute route = QwenArExecutionRoute::execute_batch;
  bool fallback = false;
  std::string fallback_reason;
  std::uint64_t kv_checksum = 0;
  std::uint64_t gdn_checksum = 0;
  float oracle_max_abs_error = 0.0F;
  MetalSubmission submission;
};

struct QwenArTokenEvent {
  std::uint32_t index = 0;
  std::uint32_t token = 0;
  std::uint32_t position = 0;
  std::string route;
};

struct QwenArRunResult {
  std::vector<QwenArTokenEvent> events;
  bool cancelled = false;
  bool failed = false;
  std::string error;
};

struct QwenArStats {
  bool installed = false;
  bool icb_installed = false;
  std::uint64_t profile_plan_count = 0;
  std::uint64_t batch_steps = 0;
  std::uint64_t icb_steps = 0;
  std::uint64_t fallback_steps = 0;
  std::uint64_t profile_switches = 0;
  std::uint64_t icb_recordings = 0;
  std::uint64_t icb_replays = 0;
  std::uint64_t token_loop_plan_builds = 0;
  std::uint64_t token_loop_pipeline_compiles = 0;
  std::uint64_t token_loop_weight_uploads = 0;
};

struct QwenArAddressSnapshot {
  std::uint64_t hidden = 0;
  std::uint64_t state = 0;
  std::uint64_t logits = 0;
  std::uint64_t sampled = 0;

  bool operator==(const QwenArAddressSnapshot &) const = default;
};

struct QwenArCheckpoint {
  std::vector<std::byte> state;
  std::vector<float> oracle_state;
  QwenSequenceRuntimeState sequence;
  std::uint32_t next_position = 0;
};

class QwenProgram final {
public:
  QwenProgram(QwenProgram &&) noexcept;
  QwenProgram &operator=(QwenProgram &&) noexcept;
  QwenProgram(const QwenProgram &) = delete;
  QwenProgram &operator=(const QwenProgram &) = delete;
  ~QwenProgram();

  [[nodiscard]] const QwenProgramBuildPlan &build_plan() const;
  [[nodiscard]] const MaterializedModel &weights() const;
  [[nodiscard]] const QwenLayerCatalog &layers() const;
  [[nodiscard]] const DispatchTable &dispatch_table() const;
  [[nodiscard]] const TopologyProfileCatalog &profiles() const;
  [[nodiscard]] const std::vector<QwenProgramStagePlan> &stage_plans() const;
  [[nodiscard]] const QwenProgramStats &stats() const;
  [[nodiscard]] const QwenSequenceRuntimeState &sequence_state() const;
  [[nodiscard]] const MetalResidencySet &residency() const;
  [[nodiscard]] const MetalParameterTable &
  parameter_table(std::uint32_t profile_index) const;
  [[nodiscard]] TopologyProfileSelection
  select_profile(TopologyProfileMode mode, std::uint32_t batch_size,
                 std::uint32_t tokens) const;
  void update_parameters(std::uint32_t profile_index,
                         const QwenDynamicStep &step);
  [[nodiscard]] QwenProgramAddressSnapshot address_snapshot() const;
  [[nodiscard]] MetalRegistryStats registry_stats() const;

  void install_autoregressive_executor(const std::filesystem::path &metallib);
  [[nodiscard]] bool autoregressive_installed() const;
  [[nodiscard]] const QwenArStats &autoregressive_stats() const;
  [[nodiscard]] QwenArAddressSnapshot autoregressive_address_snapshot() const;
  [[nodiscard]] QwenArCheckpoint checkpoint_autoregressive() const;
  void restore_autoregressive(const QwenArCheckpoint &checkpoint);
  [[nodiscard]] QwenArStepResult
  execute_ar(std::span<const std::uint32_t> tokens, TopologyProfileMode mode,
             QwenArExecutionRoute route,
             const QwenArSamplingOptions &sampling = {});
  [[nodiscard]] QwenArStepResult
  prefill(std::span<const std::uint32_t> prompt,
          const QwenArSamplingOptions &sampling = {});
  [[nodiscard]] QwenArRunResult
  generate(std::span<const std::uint32_t> prompt,
           std::uint32_t maximum_new_tokens,
           const QwenArSamplingOptions &sampling = {},
           const std::function<bool()> &cancelled = {});
  void reset_autoregressive();
  void request_cancel();
  void inject_failure_after(std::uint64_t completed_steps);
  void clear_failure_injection();

private:
  class Impl;
  friend QwenProgram assemble_qwen_program(QwenProgramBuildPlan,
                                           MetalContext &&,
                                           MaterializedModel &&,
                                           QwenLayerCatalog, DispatchTable &&,
                                           TopologyProfileCatalog);

  explicit QwenProgram(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] QwenProgram
assemble_qwen_program(QwenProgramBuildPlan plan, MetalContext &&context,
                      MaterializedModel &&weights, QwenLayerCatalog layers,
                      DispatchTable &&dispatch_table,
                      TopologyProfileCatalog profiles);

} // namespace omlx::m5
