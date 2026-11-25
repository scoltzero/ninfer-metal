#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/model_config.h"

namespace {

std::string_view layer_type(omlx::m5::LayerType type) {
  return type == omlx::m5::LayerType::full_attention ? "full_attention"
                                                     : "linear_attention";
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_layer_schedule ARTIFACT CONFIG MAX_CONTEXT");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    const auto max_context = static_cast<std::uint32_t>(std::stoul(argv[3]));
    const auto catalog =
        omlx::m5::build_qwen_layer_catalog(artifact, config, max_context, 64);
    nlohmann::json layers = nlohmann::json::array();
    for (const auto &layer : catalog.layers()) {
      nlohmann::json stages = nlohmann::json::array();
      for (const auto &stage : layer.stages) {
        if (stage.object_id != UINT32_MAX) {
          static_cast<void>(artifact.object(stage.object_id));
        }
        stages.push_back({
            {"kind", omlx::m5::to_string(stage.kind)},
            {"role", stage.role},
            {"object", stage.object_name},
            {"object_id", stage.object_id == UINT32_MAX
                              ? nlohmann::json(nullptr)
                              : nlohmann::json(stage.object_id)},
            {"format", stage.format},
            {"dispatches", stage.dispatches},
            {"workspace_bytes", stage.workspace_bytes},
            {"state_bytes", stage.state_bytes},
        });
      }
      layers.push_back({
          {"layer", layer.layer},
          {"type", layer_type(layer.type)},
          {"moe", layer.moe},
          {"prefix", layer.prefix},
          {"stage_count", layer.stages.size()},
          {"dispatches", layer.dispatches},
          {"workspace_bytes", layer.workspace_bytes},
          {"state_bytes", layer.state_bytes},
          {"stages", std::move(stages)},
      });
    }
    nlohmann::json global = nlohmann::json::array();
    for (const auto &stage : catalog.global_stages()) {
      if (stage.object_id != UINT32_MAX) {
        static_cast<void>(artifact.object(stage.object_id));
      }
      global.push_back({
          {"kind", omlx::m5::to_string(stage.kind)},
          {"role", stage.role},
          {"object", stage.object_name},
          {"object_id", stage.object_id == UINT32_MAX
                            ? nlohmann::json(nullptr)
                            : nlohmann::json(stage.object_id)},
          {"format", stage.format},
          {"dispatches", stage.dispatches},
      });
    }
    const auto &stats = catalog.stats();
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-10-layer-schedule-v1"},
        {"evidence_scope", "static artifact-resolved layer schedule"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"max_context", max_context},
        {"kv_page_tokens", 64},
        {"stats",
         {{"layer_count", stats.layer_count},
          {"full_attention_layers", stats.full_attention_layers},
          {"gdn_layers", stats.gdn_layers},
          {"dense_mlp_layers", stats.dense_mlp_layers},
          {"moe_mlp_layers", stats.moe_mlp_layers},
          {"stage_count", stats.stage_count},
          {"dispatches_per_decode_step", stats.dispatches_per_decode_step},
          {"unbatched_command_buffers_per_decode_step",
           stats.unbatched_command_buffers_per_decode_step},
          {"batched_command_buffers_per_decode_step",
           stats.batched_command_buffers_per_decode_step},
          {"persistent_state_bytes", stats.persistent_state_bytes},
          {"peak_layer_workspace_bytes", stats.peak_layer_workspace_bytes},
          {"build_time_artifact_lookups", stats.build_time_artifact_lookups},
          {"token_loop_artifact_lookups", stats.token_loop_artifact_lookups},
          {"global_stage_count", stats.global_stage_count}}},
        {"layers", std::move(layers)},
        {"global_stages", std::move(global)},
    };
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "layer schedule inspection failed: " << error.what() << '\n';
    return 1;
  }
}
