#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/model_config.h"

namespace omlx::m5 {

enum class LayerStageKind {
  normalization,
  linear,
  attention_postprocess,
  attention,
  gdn,
  router,
  swiglu,
  reduce,
  combine,
};

[[nodiscard]] std::string_view to_string(LayerStageKind kind);

struct LayerStageRecord {
  LayerStageKind kind = LayerStageKind::linear;
  std::string role;
  std::string object_name;
  std::uint32_t object_id = UINT32_MAX;
  std::string format;
  std::uint32_t dispatches = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t state_bytes = 0;
};

struct QwenLayerSchedule {
  std::uint32_t layer = 0;
  LayerType type = LayerType::linear_attention;
  bool moe = false;
  std::string prefix;
  std::vector<LayerStageRecord> stages;
  std::uint64_t dispatches = 0;
  std::uint64_t workspace_bytes = 0;
  std::uint64_t state_bytes = 0;
};

struct QwenLayerCatalogStats {
  std::uint32_t layer_count = 0;
  std::uint32_t full_attention_layers = 0;
  std::uint32_t gdn_layers = 0;
  std::uint32_t dense_mlp_layers = 0;
  std::uint32_t moe_mlp_layers = 0;
  std::uint64_t stage_count = 0;
  std::uint64_t dispatches_per_decode_step = 0;
  std::uint64_t unbatched_command_buffers_per_decode_step = 0;
  std::uint64_t batched_command_buffers_per_decode_step = 0;
  std::uint64_t persistent_state_bytes = 0;
  std::uint64_t peak_layer_workspace_bytes = 0;
  std::uint64_t build_time_artifact_lookups = 0;
  std::uint64_t token_loop_artifact_lookups = 0;
  std::uint32_t global_stage_count = 0;
};

class QwenLayerCatalog final {
public:
  QwenLayerCatalog() = default;

  [[nodiscard]] const std::vector<QwenLayerSchedule> &layers() const;
  [[nodiscard]] const QwenLayerCatalogStats &stats() const;
  [[nodiscard]] const QwenLayerSchedule &layer(std::uint32_t index) const;
  [[nodiscard]] const std::vector<LayerStageRecord> &global_stages() const;

private:
  friend QwenLayerCatalog build_qwen_layer_catalog(const ArtifactModel &,
                                                   const QwenModelConfig &,
                                                   std::uint32_t,
                                                   std::uint32_t);
  std::vector<QwenLayerSchedule> layers_;
  std::vector<LayerStageRecord> global_stages_;
  QwenLayerCatalogStats stats_;
};

// Resolves artifact names/ids once during Program construction. The resulting
// schedule is immutable and performs zero artifact lookups in the token loop.
[[nodiscard]] QwenLayerCatalog build_qwen_layer_catalog(
    const ArtifactModel &artifact, const QwenModelConfig &config,
    std::uint32_t max_context, std::uint32_t kv_page_tokens = 64);

} // namespace omlx::m5
