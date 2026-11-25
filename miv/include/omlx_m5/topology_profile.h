#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"

namespace omlx::m5 {

enum class QwenArchitecture : std::uint8_t {
  dense,
  moe,
};

enum class QwenQuantProfile : std::uint8_t {
  oq_affine,
  mxfp4,
};

enum class TopologyProfileMode : std::uint8_t {
  decode,
  prefill,
};

[[nodiscard]] std::string_view to_string(QwenArchitecture value);
[[nodiscard]] std::string_view to_string(QwenQuantProfile value);
[[nodiscard]] std::string_view to_string(TopologyProfileMode value);

struct TopologyDispatchRecord {
  std::uint32_t index = 0;
  std::uint32_t layer = UINT32_MAX;
  LayerStageKind kind = LayerStageKind::linear;
  std::string role;
  std::uint32_t stage_dispatch = 0;
  std::uint32_t object_id = UINT32_MAX;
  std::string format;
  std::string layout;
  std::vector<std::uint64_t> shape;

  bool operator==(const TopologyDispatchRecord &) const = default;
};

// Artifact-resolved, request-independent command topology. Dynamic token,
// position, page, route, mask, and sampling values intentionally do not appear
// in this descriptor.
struct QwenTopologyDescriptor {
  QwenArchitecture architecture = QwenArchitecture::dense;
  QwenQuantProfile quant_profile = QwenQuantProfile::oq_affine;
  std::uint32_t layer_count = 0;
  std::uint32_t top_k = 0;
  std::uint32_t max_context = 0;
  std::uint32_t kv_page_tokens = 64;
  std::vector<TopologyDispatchRecord> dispatches;
  std::vector<MetalCommandRangeDescriptor> ranges;
  std::string signature;
};

[[nodiscard]] QwenTopologyDescriptor describe_qwen_topology(
    const ArtifactModel &artifact, const QwenModelConfig &config,
    const QwenLayerCatalog &layers, std::uint32_t max_context,
    std::uint32_t kv_page_tokens = 64);

struct TopologyProfileKey {
  TopologyProfileMode mode = TopologyProfileMode::decode;
  QwenArchitecture architecture = QwenArchitecture::dense;
  QwenQuantProfile quant_profile = QwenQuantProfile::oq_affine;
  std::uint32_t batch_size = 1;
  std::uint32_t token_capacity = 1;
  std::string topology_signature;

  bool operator==(const TopologyProfileKey &) const = default;
};

[[nodiscard]] std::string topology_profile_key(const TopologyProfileKey &key);

struct TopologyProfile {
  TopologyProfileKey key;
  std::string name;
  std::string structural_signature;
  MetalParameterTableDescriptor parameters;
  std::vector<TopologyDispatchRecord> dispatches;
  std::vector<MetalCommandRangeDescriptor> ranges;

  [[nodiscard]] std::uint64_t parameter_bytes() const {
    return parameters.byte_size;
  }
};

struct TopologyProfileCompilerOptions {
  std::uint32_t batch_size = 1;
  std::vector<std::uint32_t> prefill_capacities{16, 32, 64, 128, 512, 1024};
};

enum class TopologyFallbackReason : std::uint8_t {
  none,
  architecture_mismatch,
  quant_profile_mismatch,
  topology_mismatch,
  unsupported_batch,
  decode_requires_one_token,
  prefill_requires_multiple_tokens,
  unsupported_token_capacity,
};

[[nodiscard]] std::string_view to_string(TopologyFallbackReason value);

struct TopologyProfileRequest {
  TopologyProfileMode mode = TopologyProfileMode::decode;
  QwenArchitecture architecture = QwenArchitecture::dense;
  QwenQuantProfile quant_profile = QwenQuantProfile::oq_affine;
  std::uint32_t batch_size = 1;
  std::uint32_t tokens = 1;
  std::string topology_signature;
};

struct TopologyProfileSelection {
  std::uint32_t profile_index = UINT32_MAX;
  TopologyFallbackReason reason = TopologyFallbackReason::none;
  std::string execution_route = "compiled_profile";

  [[nodiscard]] bool matched() const {
    return reason == TopologyFallbackReason::none;
  }
};

struct TopologyProfileCatalogStats {
  std::uint32_t profile_count = 0;
  std::uint32_t decode_profile_count = 0;
  std::uint32_t prefill_profile_count = 0;
  std::uint32_t maximum_token_capacity = 0;
  std::uint64_t total_parameter_bytes = 0;
  std::uint64_t dispatch_records_per_profile = 0;
  bool finite = false;
};

class TopologyProfileCatalog final {
public:
  [[nodiscard]] const QwenTopologyDescriptor &topology() const;
  [[nodiscard]] const std::vector<TopologyProfile> &profiles() const;
  [[nodiscard]] const TopologyProfile &profile(std::uint32_t index) const;
  [[nodiscard]] const TopologyProfileCatalogStats &stats() const;
  [[nodiscard]] TopologyProfileSelection
  select(const TopologyProfileRequest &request) const;

private:
  friend TopologyProfileCatalog
      compile_topology_profiles(QwenTopologyDescriptor,
                                TopologyProfileCompilerOptions);

  QwenTopologyDescriptor topology_;
  std::vector<TopologyProfile> profiles_;
  TopologyProfileCatalogStats stats_;
};

[[nodiscard]] TopologyProfileCatalog
compile_topology_profiles(QwenTopologyDescriptor topology,
                          TopologyProfileCompilerOptions options = {});

} // namespace omlx::m5
