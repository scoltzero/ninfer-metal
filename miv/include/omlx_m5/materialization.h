#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/metal.h"

namespace omlx::m5 {

enum class ModelComponent {
  text,
  vision,
  mtp,
  speculative,
};

[[nodiscard]] std::string_view to_string(ModelComponent component);
[[nodiscard]] ModelComponent classify_model_component(std::string_view name);

struct MaterializationOptions {
  std::vector<ModelComponent> components{ModelComponent::text};
  std::uint64_t max_segment_bytes = 0;
  std::uint64_t plane_alignment = 256;
  bool require_verified_payload = true;
};

struct WeightArenaSegmentPlan {
  std::uint32_t index = 0;
  std::uint64_t bytes = 0;
  std::uint64_t upload_bytes = 0;
  std::uint64_t padding_bytes = 0;
};

struct WeightPlanePlacement {
  std::uint32_t object_id = 0;
  std::string object_name;
  std::string plane_kind;
  std::uint32_t segment_index = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t alignment = 0;
};

struct ModelMaterializationPlan {
  const ArtifactModel *source = nullptr;
  MaterializationOptions options;
  std::uint64_t device_max_buffer_length = 0;
  std::uint64_t upload_bytes = 0;
  std::uint64_t resident_bytes = 0;
  std::uint64_t padding_bytes = 0;
  std::uint64_t selected_object_count = 0;
  std::uint64_t selected_binding_count = 0;
  std::vector<std::uint32_t> object_ids;
  std::vector<WeightArenaSegmentPlan> segments;
  std::vector<WeightPlanePlacement> placements;
};

[[nodiscard]] ModelMaterializationPlan
plan_model_materialization(const ArtifactModel &artifact,
                           const MetalContext &context,
                           MaterializationOptions options = {});

struct MaterializedPlaneView {
  std::string kind;
  const MetalBuffer *buffer = nullptr;
  std::uint32_t segment_index = 0;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  std::uint64_t row_stride = 0;
};

struct MaterializedWeightView {
  std::uint32_t object_id = 0;
  ModelComponent component = ModelComponent::text;
  std::string name;
  std::string format;
  std::string layout;
  std::vector<std::uint64_t> parent_shape;
  std::uint64_t padded_k = 0;
  std::uint64_t row_start = 0;
  std::uint64_t row_count = 0;
  std::uint64_t in_features = 0;
  std::vector<MaterializedPlaneView> planes;
  std::string input_transform = "none";

  [[nodiscard]] bool has_fwht64() const noexcept {
    return input_transform == "fwht64";
  }
};

struct MaterializedBindingView {
  ArtifactBinding binding;
  MaterializedWeightView weight;
};

struct MaterializationStats {
  std::uint64_t selected_object_count = 0;
  std::uint64_t selected_binding_count = 0;
  std::uint64_t segment_count = 0;
  std::uint64_t upload_operations = 0;
  std::uint64_t uploaded_bytes = 0;
  std::uint64_t resident_bytes = 0;
  std::uint64_t padding_bytes = 0;
};

class MaterializedModel final {
public:
  MaterializedModel();
  MaterializedModel(MaterializedModel &&) noexcept;
  MaterializedModel &operator=(MaterializedModel &&) noexcept;
  MaterializedModel(const MaterializedModel &) = delete;
  MaterializedModel &operator=(const MaterializedModel &) = delete;
  ~MaterializedModel();

  [[nodiscard]] bool valid() const;
  [[nodiscard]] const MaterializationStats &stats() const;
  [[nodiscard]] std::uint64_t segment_count() const;
  [[nodiscard]] const MetalBuffer &segment(std::uint32_t index) const;
  [[nodiscard]] bool has_object(std::uint32_t object_id) const;
  [[nodiscard]] bool has_object(std::string_view name) const;
  [[nodiscard]] MaterializedWeightView
  parent_view(std::uint32_t object_id) const;
  [[nodiscard]] MaterializedWeightView parent_view(std::string_view name) const;
  [[nodiscard]] MaterializedWeightView
  fused_parent_row_view(std::uint32_t object_id, std::uint64_t row_start,
                        std::uint64_t row_count) const;
  [[nodiscard]] MaterializedWeightView expert_view(std::uint32_t object_id,
                                                   std::uint64_t expert) const;
  [[nodiscard]] MaterializedWeightView expert_row_view(std::uint32_t object_id,
                                                       std::uint64_t expert,
                                                       std::uint64_t row) const;
  [[nodiscard]] MaterializedBindingView
  binding_view(std::string_view name) const;

private:
  class Impl;
  friend MaterializedModel materialize_model(const ArtifactModel &,
                                             ModelMaterializationPlan &&,
                                             const MetalContext &);

  explicit MaterializedModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] MaterializedModel
materialize_model(const ArtifactModel &artifact,
                  ModelMaterializationPlan &&plan, const MetalContext &context);

struct MutableMetalSlice {
  MetalBuffer *buffer = nullptr;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
};

struct MutableArenaStats {
  std::uint64_t capacity_bytes = 0;
  std::uint64_t used_bytes = 0;
  std::uint64_t peak_bytes = 0;
  std::uint64_t allocation_count = 0;
  std::uint64_t generation = 0;
};

class WorkspaceArena final {
public:
  WorkspaceArena(const MetalContext &context, std::uint64_t capacity_bytes);
  WorkspaceArena(WorkspaceArena &&) noexcept;
  WorkspaceArena &operator=(WorkspaceArena &&) noexcept;
  WorkspaceArena(const WorkspaceArena &) = delete;
  WorkspaceArena &operator=(const WorkspaceArena &) = delete;
  ~WorkspaceArena();

  [[nodiscard]] MutableMetalSlice allocate(std::uint64_t bytes,
                                           std::uint64_t alignment = 256);
  // Invalidates every prior slice. Reset does not zero the backing buffer.
  void reset();
  [[nodiscard]] const MutableArenaStats &stats() const;
  [[nodiscard]] MetalBuffer &buffer();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class StateArena final {
public:
  StateArena(const MetalContext &context, std::uint64_t capacity_bytes);
  StateArena(StateArena &&) noexcept;
  StateArena &operator=(StateArena &&) noexcept;
  StateArena(const StateArena &) = delete;
  StateArena &operator=(const StateArena &) = delete;
  ~StateArena();

  [[nodiscard]] MutableMetalSlice allocate(std::uint64_t bytes,
                                           std::uint64_t alignment = 256);
  // Invalidates every prior slice. State initialization remains a Program duty.
  void reset();
  [[nodiscard]] const MutableArenaStats &stats() const;
  [[nodiscard]] MetalBuffer &buffer();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace omlx::m5
