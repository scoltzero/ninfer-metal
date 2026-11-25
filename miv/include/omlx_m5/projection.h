#pragma once

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"

namespace omlx::m5 {

enum class ProjectionBundleKind {
  full_attention,
  gdn,
  dense_mlp,
};

[[nodiscard]] std::string_view to_string(ProjectionBundleKind kind);

struct ProjectionSliceSpec {
  std::string name;
  std::uint64_t feature_offset = 0;
  std::uint64_t feature_count = 0;
  std::uint64_t block_count = 1;
  std::uint64_t block_features = 0;
  std::uint64_t block_stride_features = 0;

  bool operator==(const ProjectionSliceSpec &) const = default;
};

struct ProjectionParentSpec {
  std::string role;
  std::string binding_name;
  std::uint64_t input_features = 0;
  std::uint64_t output_features = 0;
  std::vector<ProjectionSliceSpec> outputs;
};

struct ProjectionBundleSpec {
  ProjectionBundleKind kind = ProjectionBundleKind::full_attention;
  std::string prefix;
  std::vector<ProjectionParentSpec> parents;
};

[[nodiscard]] ProjectionBundleSpec
make_full_attention_projection_bundle(const QwenModelConfig &config,
                                      std::string prefix);
[[nodiscard]] ProjectionBundleSpec
make_gdn_projection_bundle(const QwenModelConfig &config, std::string prefix);
[[nodiscard]] ProjectionBundleSpec
make_dense_mlp_projection_bundle(const QwenModelConfig &config,
                                 std::string prefix);
void validate_projection_bundle(const ProjectionBundleSpec &bundle);
void validate_projection_parent(const ProjectionParentSpec &parent,
                                const ArtifactBinding &binding,
                                const MaterializedWeightView &weight);

struct ProjectionParentOutputView {
  std::string role;
  MetalBuffer *buffer = nullptr;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint32_t tokens = 0;
  std::uint64_t features = 0;
  std::uint64_t token_stride_bytes = 0;
};

struct ProjectionSplitOutputView {
  std::string name;
  std::string parent_role;
  MetalBuffer *buffer = nullptr;
  std::uint64_t offset = 0;
  std::uint64_t logical_bytes = 0;
  std::uint64_t span_bytes = 0;
  std::uint32_t tokens = 0;
  std::uint64_t features = 0;
  std::uint64_t token_stride_bytes = 0;
  std::uint64_t block_count = 1;
  std::uint64_t block_features = 0;
  std::uint64_t block_stride_bytes = 0;

  [[nodiscard]] std::uint64_t token_offset(std::uint32_t token) const;
  [[nodiscard]] std::uint64_t feature_offset(std::uint32_t token,
                                             std::uint64_t feature) const;
  [[nodiscard]] bool contiguous_per_token() const;
};

struct ProjectionOutputArenaStats {
  std::uint64_t capacity_bytes = 0;
  std::uint64_t logical_parent_bytes = 0;
  std::uint64_t alignment_padding_bytes = 0;
  std::uint64_t parent_count = 0;
  std::uint64_t split_count = 0;
  std::uint64_t split_copy_dispatches = 0;
};

class ProjectionOutputArena final {
public:
  ProjectionOutputArena(const MetalContext &context,
                        ProjectionBundleSpec bundle, std::uint32_t tokens,
                        std::uint64_t alignment = 256);
  ProjectionOutputArena(ProjectionOutputArena &&) noexcept;
  ProjectionOutputArena &operator=(ProjectionOutputArena &&) noexcept;
  ProjectionOutputArena(const ProjectionOutputArena &) = delete;
  ProjectionOutputArena &operator=(const ProjectionOutputArena &) = delete;
  ~ProjectionOutputArena();

  [[nodiscard]] const ProjectionBundleSpec &bundle() const;
  [[nodiscard]] std::uint32_t tokens() const;
  [[nodiscard]] const ProjectionOutputArenaStats &stats() const;
  [[nodiscard]] ProjectionParentOutputView parent(std::string_view role);
  [[nodiscard]] ProjectionSplitOutputView output(std::string_view name);
  [[nodiscard]] MetalBuffer &buffer();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
