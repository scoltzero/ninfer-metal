#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"

namespace {

using Json = nlohmann::json;

std::uint64_t parse_u64(std::string_view value, std::string_view label) {
  std::size_t parsed = 0;
  const auto result = std::stoull(std::string(value), &parsed, 0);
  if (parsed != value.size()) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return result;
}

omlx::m5::ModelComponent parse_component(std::string_view value) {
  using omlx::m5::ModelComponent;
  if (value == "text") {
    return ModelComponent::text;
  }
  if (value == "vision") {
    return ModelComponent::vision;
  }
  if (value == "mtp") {
    return ModelComponent::mtp;
  }
  if (value == "speculative") {
    return ModelComponent::speculative;
  }
  throw std::invalid_argument("unknown model component: " + std::string(value));
}

std::vector<omlx::m5::ModelComponent> parse_components(std::string_view value) {
  std::vector<omlx::m5::ModelComponent> result;
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const auto end = value.find(',', begin);
    const auto part =
        value.substr(begin, end == std::string_view::npos ? value.size() - begin
                                                          : end - begin);
    if (part.empty()) {
      throw std::invalid_argument("component list contains an empty item");
    }
    result.push_back(parse_component(part));
    if (end == std::string_view::npos) {
      break;
    }
    begin = end + 1;
  }
  return result;
}

Json plan_json(const omlx::m5::ModelMaterializationPlan &plan) {
  Json segments = Json::array();
  for (const auto &segment : plan.segments) {
    segments.push_back({
        {"index", segment.index},
        {"bytes", segment.bytes},
        {"upload_bytes", segment.upload_bytes},
        {"padding_bytes", segment.padding_bytes},
    });
  }
  Json components = Json::array();
  for (const auto component : plan.options.components) {
    components.push_back(omlx::m5::to_string(component));
  }
  return {
      {"components", std::move(components)},
      {"device_max_buffer_length", plan.device_max_buffer_length},
      {"max_segment_bytes", plan.options.max_segment_bytes},
      {"plane_alignment", plan.options.plane_alignment},
      {"selected_object_count", plan.selected_object_count},
      {"selected_binding_count", plan.selected_binding_count},
      {"segment_count", plan.segments.size()},
      {"placement_count", plan.placements.size()},
      {"upload_bytes", plan.upload_bytes},
      {"resident_bytes", plan.resident_bytes},
      {"padding_bytes", plan.padding_bytes},
      {"segments", std::move(segments)},
  };
}

Json stats_json(const omlx::m5::MaterializationStats &stats) {
  return {
      {"selected_object_count", stats.selected_object_count},
      {"selected_binding_count", stats.selected_binding_count},
      {"segment_count", stats.segment_count},
      {"upload_operations", stats.upload_operations},
      {"uploaded_bytes", stats.uploaded_bytes},
      {"resident_bytes", stats.resident_bytes},
      {"padding_bytes", stats.padding_bytes},
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_materialization ARTIFACT "
          "[--components text,mtp] [--max-segment-bytes N] "
          "[--materialize] [--verify-resident]");
    }
    const std::string artifact_path = argv[1];
    auto components =
        std::vector<omlx::m5::ModelComponent>{omlx::m5::ModelComponent::text};
    std::uint64_t max_segment_bytes = 0;
    bool materialize = false;
    bool verify_resident = false;
    for (int index = 2; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--components" && index + 1 < argc) {
        components = parse_components(argv[++index]);
      } else if (argument == "--max-segment-bytes" && index + 1 < argc) {
        max_segment_bytes = parse_u64(argv[++index], "max segment bytes");
      } else if (argument == "--materialize") {
        materialize = true;
      } else if (argument == "--verify-resident") {
        verify_resident = true;
      } else {
        throw std::invalid_argument("unknown materialization argument: " +
                                    std::string(argument));
      }
    }
    if (verify_resident && !materialize) {
      throw std::invalid_argument("--verify-resident requires --materialize");
    }

    auto context = omlx::m5::MetalContext::create();
    std::optional<omlx::m5::ArtifactModel> artifact;
    artifact.emplace(omlx::m5::ArtifactModel::open(artifact_path, materialize));
    omlx::m5::MaterializationOptions options{
        .components = std::move(components),
        .max_segment_bytes = max_segment_bytes,
        .plane_alignment = 256,
        .require_verified_payload = materialize,
    };
    auto plan =
        omlx::m5::plan_model_materialization(*artifact, context, options);
    Json result{
        {"schema", "omlx-m5-b1-2-materialization-v1"},
        {"evidence_scope", "M4 correctness; not M5 Ultra performance"},
        {"artifact", artifact_path},
        {"source_repo", artifact->summary().source_repo},
        {"source_revision", artifact->summary().source_revision},
        {"payload_verified", artifact->summary().payload_verified},
        {"materialized", false},
        {"plan", plan_json(plan)},
    };
    if (!materialize) {
      std::cout << result.dump(2) << '\n';
      return 0;
    }

    const auto registry_before = context.registry_stats();
    auto model =
        omlx::m5::materialize_model(*artifact, std::move(plan), context);
    const auto registry_after = context.registry_stats();
    bool resident_verified = true;
    std::string first_binding;
    std::optional<std::uint32_t> rank_three_object;
    for (const auto &binding : artifact->bindings()) {
      if (model.has_object(binding.object_id) && first_binding.empty()) {
        first_binding = binding.name;
      }
    }
    for (const auto &object : artifact->objects()) {
      if (!model.has_object(object.id)) {
        continue;
      }
      const auto parent = model.parent_view(object.id);
      if (object.shape.size() == 3 && !rank_three_object.has_value()) {
        rank_three_object = object.id;
      }
      if (!verify_resident) {
        continue;
      }
      for (const auto &view : parent.planes) {
        const auto &plane = artifact->plane(object, view.kind);
        const auto source = artifact->plane_bytes(object, plane);
        const auto resident = view.buffer->read(view.offset, view.bytes);
        if (!std::ranges::equal(source, resident)) {
          resident_verified = false;
        }
      }
    }

    Json view_checks{
        {"resident_bytes_verified", resident_verified},
        {"binding_view_verified", false},
        {"fused_parent_row_view_verified", false},
        {"expert_view_verified", false},
        {"reader_released", false},
        {"concurrent_readers_verified", false},
    };
    std::string stable_object;
    if (!first_binding.empty()) {
      const auto binding = model.binding_view(first_binding);
      stable_object = binding.weight.name;
      view_checks["binding_view_verified"] =
          binding.binding.object_id == binding.weight.object_id &&
          !binding.weight.planes.empty();
      if (binding.weight.row_count >= 2) {
        const auto rows = model.fused_parent_row_view(
            binding.weight.object_id, 1, binding.weight.row_count - 1);
        view_checks["fused_parent_row_view_verified"] =
            rows.row_start == 1 &&
            std::ranges::all_of(rows.planes, [](const auto &plane) {
              return plane.row_stride != 0 && plane.bytes != 0;
            });
      }
    }
    if (rank_three_object.has_value()) {
      const auto expert = model.expert_view(*rank_three_object, 0);
      const auto row = model.expert_row_view(*rank_three_object, 0, 0);
      view_checks["expert_view_verified"] =
          expert.row_start == 0 && expert.row_count > 0 && row.row_count == 1 &&
          expert.planes.size() == row.planes.size();
    }

    artifact.reset();
    view_checks["reader_released"] =
        !stable_object.empty() &&
        model.parent_view(stable_object).name == stable_object;
    std::atomic<bool> readers_ok{true};
    std::vector<std::jthread> readers;
    for (std::size_t index = 0; index < 8; ++index) {
      readers.emplace_back([&] {
        try {
          for (std::size_t iteration = 0; iteration < 100; ++iteration) {
            const auto view = model.parent_view(stable_object);
            if (view.name != stable_object || view.planes.empty()) {
              readers_ok.store(false);
            }
          }
        } catch (...) {
          readers_ok.store(false);
        }
      });
    }
    readers.clear();
    view_checks["concurrent_readers_verified"] = readers_ok.load();
    const auto registry_final = context.registry_stats();
    view_checks["repeated_view_upload_free"] =
        registry_final.upload_count == registry_after.upload_count &&
        registry_final.uploaded_bytes == registry_after.uploaded_bytes;

    result["materialized"] = true;
    result["stats"] = stats_json(model.stats());
    result["view_checks"] = std::move(view_checks);
    result["registry_delta"] = {
        {"buffer_count", registry_after.current_buffer_count -
                             registry_before.current_buffer_count},
        {"buffer_bytes", registry_after.current_buffer_bytes -
                             registry_before.current_buffer_bytes},
        {"upload_count",
         registry_after.upload_count - registry_before.upload_count},
        {"uploaded_bytes",
         registry_after.uploaded_bytes - registry_before.uploaded_bytes},
        {"post_materialization_upload_count",
         registry_final.upload_count - registry_after.upload_count},
        {"post_materialization_uploaded_bytes",
         registry_final.uploaded_bytes - registry_after.uploaded_bytes},
    };
    std::cout << result.dump(2) << '\n';
    return resident_verified ? 0 : 1;
  } catch (const std::exception &error) {
    std::cerr << "materialization inspection failed: " << error.what() << '\n';
    return 1;
  }
}
