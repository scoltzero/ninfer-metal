#include "omlx_m5/materialization.h"

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace omlx::m5 {
namespace {

bool is_power_of_two(std::uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right,
                          std::string_view label) {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left + right;
}

std::uint64_t checked_multiply(std::uint64_t left, std::uint64_t right,
                               std::string_view label) {
  if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) {
    throw std::overflow_error(std::string(label) + " overflow");
  }
  return left * right;
}

std::uint64_t align_up(std::uint64_t value, std::uint64_t alignment,
                       std::string_view label) {
  if (!is_power_of_two(alignment)) {
    throw std::invalid_argument(std::string(label) + " alignment is invalid");
  }
  const auto mask = alignment - 1;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    throw std::overflow_error(std::string(label) + " alignment overflow");
  }
  return (value + mask) & ~mask;
}

std::uint64_t flattened_rows(const ArtifactObject &object) {
  if (object.shape.size() < 2) {
    return 1;
  }
  std::uint64_t rows = 1;
  for (std::size_t index = 0; index + 1 < object.shape.size(); ++index) {
    if (object.shape[index] == 0) {
      throw std::runtime_error("materialized object dimension is zero");
    }
    rows = checked_multiply(rows, object.shape[index], "materialized rows");
  }
  return rows;
}

std::array<bool, 4>
selected_component_mask(std::span<const ModelComponent> components) {
  if (components.empty()) {
    throw std::invalid_argument("materialization component set is empty");
  }
  std::array<bool, 4> selected{};
  for (const auto component : components) {
    const auto index = static_cast<std::size_t>(component);
    if (index >= selected.size() || selected[index]) {
      throw std::invalid_argument(
          "materialization component set is invalid or duplicated");
    }
    selected[index] = true;
  }
  return selected;
}

bool component_selected(const std::array<bool, 4> &selected,
                        ModelComponent component) {
  return selected.at(static_cast<std::size_t>(component));
}

bool same_segment(const WeightArenaSegmentPlan &left,
                  const WeightArenaSegmentPlan &right) {
  return left.index == right.index && left.bytes == right.bytes &&
         left.upload_bytes == right.upload_bytes &&
         left.padding_bytes == right.padding_bytes;
}

bool same_placement(const WeightPlanePlacement &left,
                    const WeightPlanePlacement &right) {
  return left.object_id == right.object_id &&
         left.object_name == right.object_name &&
         left.plane_kind == right.plane_kind &&
         left.segment_index == right.segment_index &&
         left.offset == right.offset && left.bytes == right.bytes &&
         left.alignment == right.alignment;
}

bool same_plan(const ModelMaterializationPlan &left,
               const ModelMaterializationPlan &right) {
  return left.source == right.source &&
         left.options.components == right.options.components &&
         left.options.max_segment_bytes == right.options.max_segment_bytes &&
         left.options.plane_alignment == right.options.plane_alignment &&
         left.options.require_verified_payload ==
             right.options.require_verified_payload &&
         left.device_max_buffer_length == right.device_max_buffer_length &&
         left.upload_bytes == right.upload_bytes &&
         left.resident_bytes == right.resident_bytes &&
         left.padding_bytes == right.padding_bytes &&
         left.selected_object_count == right.selected_object_count &&
         left.selected_binding_count == right.selected_binding_count &&
         left.object_ids == right.object_ids &&
         left.segments.size() == right.segments.size() &&
         left.placements.size() == right.placements.size() &&
         std::ranges::equal(left.segments, right.segments, same_segment) &&
         std::ranges::equal(left.placements, right.placements, same_placement);
}

class MutableArenaCore final {
public:
  MutableArenaCore(const MetalContext &context, std::uint64_t capacity_bytes)
      : buffer_(context.make_buffer(capacity_bytes)) {
    stats_.capacity_bytes = capacity_bytes;
  }

  MutableMetalSlice allocate(std::uint64_t bytes, std::uint64_t alignment) {
    if (bytes == 0 || !is_power_of_two(alignment)) {
      throw std::invalid_argument("arena allocation geometry is invalid");
    }
    const auto offset = align_up(stats_.used_bytes, alignment, "arena");
    const auto end = checked_add(offset, bytes, "arena allocation");
    if (end > stats_.capacity_bytes) {
      throw std::bad_alloc();
    }
    stats_.used_bytes = end;
    stats_.peak_bytes = std::max(stats_.peak_bytes, end);
    ++stats_.allocation_count;
    return {.buffer = &buffer_, .offset = offset, .bytes = bytes};
  }

  void reset() {
    stats_.used_bytes = 0;
    ++stats_.generation;
  }

  [[nodiscard]] const MutableArenaStats &stats() const { return stats_; }
  [[nodiscard]] MetalBuffer &buffer() { return buffer_; }

private:
  MetalBuffer buffer_;
  MutableArenaStats stats_;
};

} // namespace

std::string_view to_string(ModelComponent component) {
  switch (component) {
  case ModelComponent::text:
    return "text";
  case ModelComponent::vision:
    return "vision";
  case ModelComponent::mtp:
    return "mtp";
  case ModelComponent::speculative:
    return "speculative";
  }
  throw std::invalid_argument("unknown model component");
}

ModelComponent classify_model_component(std::string_view name) {
  const auto contains = [&](std::string_view needle) {
    return name.find(needle) != std::string_view::npos;
  };
  if (contains(".mtp.") || name.starts_with("mtp.")) {
    return ModelComponent::mtp;
  }
  if (contains("vision_model") || contains("vision_tower") ||
      contains("visual.")) {
    return ModelComponent::vision;
  }
  if (contains("draft_model") || contains(".draft.") || contains(".dflash") ||
      name.starts_with("dflash2.") || contains(".speculative.")) {
    return ModelComponent::speculative;
  }
  return ModelComponent::text;
}

ModelMaterializationPlan
plan_model_materialization(const ArtifactModel &artifact,
                           const MetalContext &context,
                           MaterializationOptions options) {
  if (options.require_verified_payload &&
      !artifact.summary().payload_verified) {
    throw std::invalid_argument(
        "materialization requires a checksum-verified artifact payload");
  }
  if (options.plane_alignment < 256 ||
      !is_power_of_two(options.plane_alignment)) {
    throw std::invalid_argument(
        "materialization plane alignment must be a power of two at least 256");
  }
  const auto selected = selected_component_mask(options.components);
  const auto device_limit = context.capabilities().max_buffer_length;
  // Apple Silicon Metal Compute ICBs (MTLIndirectComputeCommand
  // setKernelBuffer:offset:atIndex:) truncate buffer offsets >= 4 GiB without
  // triggering Metal API validation (observed on M5 Ultra 96GB with 14.2 GiB
  // Qwen3.8-27B in a single 58 GiB maxBufferLength arena). Cap default weight
  // arena segments at 2 GiB so every packed weight offset stays well within
  // the 32-bit ICB offset envelope on high-memory Apple Silicon.
  constexpr std::uint64_t kDefaultMaxWeightSegmentBytes = 2ULL << 30;
  const auto segment_limit =
      options.max_segment_bytes == 0
          ? std::min<std::uint64_t>(device_limit, kDefaultMaxWeightSegmentBytes)
          : options.max_segment_bytes;
  if (segment_limit == 0 || segment_limit > device_limit ||
      segment_limit < options.plane_alignment) {
    throw std::invalid_argument(
        "materialization segment limit is unsupported by the Metal device");
  }

  ModelMaterializationPlan plan;
  plan.source = &artifact;
  plan.options = std::move(options);
  plan.device_max_buffer_length = device_limit;
  std::unordered_set<std::uint32_t> selected_ids;
  for (const auto &object : artifact.objects()) {
    if (!component_selected(selected, classify_model_component(object.name))) {
      continue;
    }
    plan.object_ids.push_back(object.id);
    selected_ids.insert(object.id);
    for (const auto &plane : object.planes) {
      const auto alignment =
          std::max(plan.options.plane_alignment, plane.alignment);
      const bool oversize_single_plane =
          plan.options.max_segment_bytes == 0 && plane.bytes > segment_limit &&
          plane.bytes <= device_limit;
      if (plane.bytes > segment_limit && !oversize_single_plane) {
        throw std::length_error(
            "artifact plane exceeds the Metal segment limit: " + object.name +
            "/" + plane.kind);
      }
      if (plan.segments.empty()) {
        plan.segments.push_back({.index = 0});
      }
      auto *segment = &plan.segments.back();
      auto offset = align_up(segment->bytes, alignment, "weight segment");
      if (segment->bytes != 0 &&
          (oversize_single_plane ||
           checked_add(offset, plane.bytes, "weight segment") >
               segment_limit)) {
        if (plan.segments.size() >= std::numeric_limits<std::uint32_t>::max()) {
          throw std::overflow_error("weight segment count overflow");
        }
        plan.segments.push_back(
            {.index = static_cast<std::uint32_t>(plan.segments.size())});
        segment = &plan.segments.back();
        offset = 0;
      }
      const auto padding = offset - segment->bytes;
      segment->padding_bytes =
          checked_add(segment->padding_bytes, padding, "segment padding");
      segment->upload_bytes = checked_add(segment->upload_bytes, plane.bytes,
                                          "segment upload bytes");
      segment->bytes = checked_add(offset, plane.bytes, "segment bytes");
      plan.padding_bytes =
          checked_add(plan.padding_bytes, padding, "weight padding");
      plan.upload_bytes =
          checked_add(plan.upload_bytes, plane.bytes, "weight upload bytes");
      plan.placements.push_back({
          .object_id = object.id,
          .object_name = object.name,
          .plane_kind = plane.kind,
          .segment_index = segment->index,
          .offset = offset,
          .bytes = plane.bytes,
          .alignment = alignment,
      });
    }
  }
  if (plan.object_ids.empty() || plan.placements.empty()) {
    throw std::invalid_argument(
        "materialization selection contains no artifact objects");
  }
  plan.selected_object_count = plan.object_ids.size();
  plan.selected_binding_count =
      static_cast<std::uint64_t>(std::ranges::count_if(
          artifact.bindings(), [&](const ArtifactBinding &binding) {
            return selected_ids.contains(binding.object_id);
          }));
  for (const auto &segment : plan.segments) {
    plan.resident_bytes =
        checked_add(plan.resident_bytes, segment.bytes, "resident bytes");
  }
  if (plan.resident_bytes != checked_add(plan.upload_bytes, plan.padding_bytes,
                                         "resident accounting")) {
    throw std::logic_error("materialization resident accounting mismatch");
  }
  return plan;
}

class MaterializedModel::Impl {
public:
  struct PlaneStorage {
    std::string kind;
    std::uint32_t segment_index = 0;
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
  };

  struct ObjectStorage {
    ArtifactObject object;
    ModelComponent component = ModelComponent::text;
    std::vector<PlaneStorage> planes;
  };

  [[nodiscard]] const ObjectStorage &object(std::uint32_t id) const {
    if (id >= objects.size() || !objects[id].has_value()) {
      throw std::out_of_range("artifact object was not materialized");
    }
    return *objects[id];
  }

  [[nodiscard]] MaterializedWeightView view(std::uint32_t id,
                                            std::uint64_t row_start,
                                            std::uint64_t row_count,
                                            bool sliced) const {
    const auto &storage = object(id);
    const auto &object_record = storage.object;
    const auto rows = flattened_rows(object_record);
    if (row_count == 0 || row_start > rows || row_count > rows - row_start) {
      throw std::out_of_range("materialized row view is out of range");
    }
    if (sliced && (object_record.shape.size() < 2 ||
                   object_record.format.starts_with("direct_"))) {
      throw std::invalid_argument(
          "row views require a quantized rank-2 or rank-3 parent");
    }
    MaterializedWeightView result{
        .object_id = object_record.id,
        .component = storage.component,
        .name = object_record.name,
        .format = object_record.format,
        .layout = object_record.layout,
        .parent_shape = object_record.shape,
        .padded_k = object_record.padded_k,
        .row_start = row_start,
        .row_count = row_count,
        .in_features =
            object_record.shape.empty() ? 0 : object_record.shape.back(),
        .planes = {},
        .input_transform = object_record.input_transform,
    };
    result.planes.reserve(storage.planes.size());
    for (const auto &plane : storage.planes) {
      std::uint64_t stride = 0;
      std::uint64_t offset = plane.offset;
      std::uint64_t bytes = plane.bytes;
      if (object_record.shape.size() >= 2 && plane.bytes % rows == 0) {
        stride = plane.bytes / rows;
      }
      if (sliced) {
        if (stride == 0) {
          throw std::runtime_error(
              "materialized plane has no integral row stride");
        }
        offset = checked_add(
            offset, checked_multiply(row_start, stride, "plane row offset"),
            "plane view offset");
        bytes = checked_multiply(row_count, stride, "plane view bytes");
      }
      result.planes.push_back({
          .kind = plane.kind,
          .buffer = &segments.at(plane.segment_index),
          .segment_index = plane.segment_index,
          .offset = offset,
          .bytes = bytes,
          .row_stride = stride,
      });
    }
    return result;
  }

  std::vector<MetalBuffer> segments;
  std::vector<std::optional<ObjectStorage>> objects;
  std::unordered_map<std::string, std::uint32_t> object_ids;
  std::vector<ArtifactBinding> bindings;
  std::unordered_map<std::string, std::size_t> binding_indices;
  MaterializationStats stats;
};

MaterializedModel::MaterializedModel() = default;
MaterializedModel::MaterializedModel(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
MaterializedModel::MaterializedModel(MaterializedModel &&) noexcept = default;
MaterializedModel &
MaterializedModel::operator=(MaterializedModel &&) noexcept = default;
MaterializedModel::~MaterializedModel() = default;

bool MaterializedModel::valid() const { return impl_ != nullptr; }

const MaterializationStats &MaterializedModel::stats() const {
  if (impl_ == nullptr) {
    throw std::logic_error("materialized model is empty");
  }
  return impl_->stats;
}

std::uint64_t MaterializedModel::segment_count() const {
  return impl_ == nullptr ? 0 : impl_->segments.size();
}

const MetalBuffer &MaterializedModel::segment(std::uint32_t index) const {
  if (impl_ == nullptr || index >= impl_->segments.size()) {
    throw std::out_of_range("materialized weight segment is out of range");
  }
  return impl_->segments[index];
}

bool MaterializedModel::has_object(std::uint32_t object_id) const {
  return impl_ != nullptr && object_id < impl_->objects.size() &&
         impl_->objects[object_id].has_value();
}

bool MaterializedModel::has_object(std::string_view name) const {
  return impl_ != nullptr && impl_->object_ids.contains(std::string(name));
}

MaterializedWeightView
MaterializedModel::parent_view(std::uint32_t object_id) const {
  if (impl_ == nullptr) {
    throw std::logic_error("materialized model is empty");
  }
  const auto &object = impl_->object(object_id).object;
  return impl_->view(object_id, 0, flattened_rows(object), false);
}

MaterializedWeightView
MaterializedModel::parent_view(std::string_view name) const {
  if (impl_ == nullptr) {
    throw std::logic_error("materialized model is empty");
  }
  const auto found = impl_->object_ids.find(std::string(name));
  if (found == impl_->object_ids.end()) {
    throw std::out_of_range("unknown materialized object: " +
                            std::string(name));
  }
  return parent_view(found->second);
}

MaterializedWeightView
MaterializedModel::fused_parent_row_view(std::uint32_t object_id,
                                         std::uint64_t row_start,
                                         std::uint64_t row_count) const {
  if (impl_ == nullptr) {
    throw std::logic_error("materialized model is empty");
  }
  return impl_->view(object_id, row_start, row_count, true);
}

MaterializedWeightView
MaterializedModel::expert_view(std::uint32_t object_id,
                               std::uint64_t expert) const {
  if (impl_ == nullptr) {
    throw std::logic_error("materialized model is empty");
  }
  const auto &object = impl_->object(object_id).object;
  if (object.shape.size() != 3 || expert >= object.shape[0]) {
    throw std::invalid_argument(
        "expert view requires a valid rank-3 expert parent");
  }
  const auto rows_per_expert = object.shape[1];
  return impl_->view(
      object_id, checked_multiply(expert, rows_per_expert, "expert row start"),
      rows_per_expert, true);
}

MaterializedWeightView MaterializedModel::expert_row_view(
    std::uint32_t object_id, std::uint64_t expert, std::uint64_t row) const {
  if (impl_ == nullptr) {
    throw std::logic_error("materialized model is empty");
  }
  const auto &object = impl_->object(object_id).object;
  if (object.shape.size() != 3 || expert >= object.shape[0] ||
      row >= object.shape[1]) {
    throw std::invalid_argument(
        "expert row view requires valid expert and row indices");
  }
  const auto row_start =
      checked_add(checked_multiply(expert, object.shape[1], "expert row start"),
                  row, "expert row start");
  return impl_->view(object_id, row_start, 1, true);
}

MaterializedBindingView
MaterializedModel::binding_view(std::string_view name) const {
  if (impl_ == nullptr) {
    throw std::logic_error("materialized model is empty");
  }
  const auto found = impl_->binding_indices.find(std::string(name));
  if (found == impl_->binding_indices.end()) {
    throw std::out_of_range("unknown materialized binding: " +
                            std::string(name));
  }
  const auto &binding = impl_->bindings.at(found->second);
  return {.binding = binding, .weight = parent_view(binding.object_id)};
}

MaterializedModel materialize_model(const ArtifactModel &artifact,
                                    ModelMaterializationPlan &&plan,
                                    const MetalContext &context) {
  if (!artifact.summary().payload_verified) {
    throw std::invalid_argument(
        "materialization requires a checksum-verified artifact payload");
  }
  if (plan.source != &artifact) {
    throw std::invalid_argument(
        "materialization plan belongs to another artifact load session");
  }
  const auto expected =
      plan_model_materialization(artifact, context, plan.options);
  if (!same_plan(plan, expected)) {
    throw std::invalid_argument(
        "materialization plan was modified or is stale");
  }

  auto impl = std::make_unique<MaterializedModel::Impl>();
  impl->objects.resize(artifact.objects().size());
  impl->segments.reserve(plan.segments.size());
  for (const auto &segment : plan.segments) {
    impl->segments.push_back(context.make_buffer(segment.bytes));
  }
  for (const auto object_id : plan.object_ids) {
    const auto &object = artifact.object(object_id);
    MaterializedModel::Impl::ObjectStorage storage;
    storage.object = object;
    storage.component = classify_model_component(object.name);
    storage.planes.reserve(object.planes.size());
    impl->object_ids.emplace(object.name, object.id);
    impl->objects[object.id] = std::move(storage);
  }
  for (const auto &placement : plan.placements) {
    const auto &object = artifact.object(placement.object_id);
    const auto &plane = artifact.plane(object, placement.plane_kind);
    if (object.name != placement.object_name ||
        plane.bytes != placement.bytes ||
        placement.segment_index >= impl->segments.size()) {
      throw std::logic_error("materialization placement contract mismatch");
    }
    auto &storage = *impl->objects.at(object.id);
    context.upload_into(impl->segments[placement.segment_index],
                        artifact.plane_bytes(object, plane), placement.offset);
    storage.planes.push_back({
        .kind = plane.kind,
        .segment_index = placement.segment_index,
        .offset = placement.offset,
        .bytes = placement.bytes,
    });
  }
  for (const auto object_id : plan.object_ids) {
    const auto &storage = *impl->objects.at(object_id);
    if (storage.planes.size() != storage.object.planes.size()) {
      throw std::logic_error(
          "materialized object plane coverage is incomplete");
    }
  }
  for (const auto &binding : artifact.bindings()) {
    if (!impl->objects.at(binding.object_id).has_value()) {
      continue;
    }
    const auto index = impl->bindings.size();
    if (!impl->binding_indices.emplace(binding.name, index).second) {
      throw std::logic_error("materialized binding is duplicated");
    }
    impl->bindings.push_back(binding);
  }
  impl->stats = {
      .selected_object_count = plan.selected_object_count,
      .selected_binding_count = plan.selected_binding_count,
      .segment_count = plan.segments.size(),
      .upload_operations = plan.placements.size(),
      .uploaded_bytes = plan.upload_bytes,
      .resident_bytes = plan.resident_bytes,
      .padding_bytes = plan.padding_bytes,
  };
  return MaterializedModel(std::move(impl));
}

class WorkspaceArena::Impl {
public:
  Impl(const MetalContext &context, std::uint64_t capacity_bytes)
      : arena(context, capacity_bytes) {}

  MutableArenaCore arena;
};

WorkspaceArena::WorkspaceArena(const MetalContext &context,
                               std::uint64_t capacity_bytes)
    : impl_(std::make_unique<Impl>(context, capacity_bytes)) {}
WorkspaceArena::WorkspaceArena(WorkspaceArena &&) noexcept = default;
WorkspaceArena &WorkspaceArena::operator=(WorkspaceArena &&) noexcept = default;
WorkspaceArena::~WorkspaceArena() = default;

MutableMetalSlice WorkspaceArena::allocate(std::uint64_t bytes,
                                           std::uint64_t alignment) {
  return impl_->arena.allocate(bytes, alignment);
}

void WorkspaceArena::reset() { impl_->arena.reset(); }

const MutableArenaStats &WorkspaceArena::stats() const {
  return impl_->arena.stats();
}

MetalBuffer &WorkspaceArena::buffer() { return impl_->arena.buffer(); }

class StateArena::Impl {
public:
  Impl(const MetalContext &context, std::uint64_t capacity_bytes)
      : arena(context, capacity_bytes) {}

  MutableArenaCore arena;
};

StateArena::StateArena(const MetalContext &context,
                       std::uint64_t capacity_bytes)
    : impl_(std::make_unique<Impl>(context, capacity_bytes)) {}
StateArena::StateArena(StateArena &&) noexcept = default;
StateArena &StateArena::operator=(StateArena &&) noexcept = default;
StateArena::~StateArena() = default;

MutableMetalSlice StateArena::allocate(std::uint64_t bytes,
                                       std::uint64_t alignment) {
  return impl_->arena.allocate(bytes, alignment);
}

void StateArena::reset() { impl_->arena.reset(); }

const MutableArenaStats &StateArena::stats() const {
  return impl_->arena.stats();
}

MetalBuffer &StateArena::buffer() { return impl_->arena.buffer(); }

} // namespace omlx::m5
