#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_program.h"
#include "omlx_m5/topology_profile.h"

int main(int argc, char **argv) {
  try {
    if (argc < 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_qwen_program ARTIFACT CONFIG MAX_CONTEXT "
          "[--materialize] [--iterations N]");
    }
    bool should_materialize = false;
    std::uint32_t iterations = 1000;
    for (int index = 4; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--materialize") {
        should_materialize = true;
      } else if (argument == "--iterations" && index + 1 < argc) {
        iterations = static_cast<std::uint32_t>(std::stoul(argv[++index]));
      } else {
        throw std::invalid_argument("unknown Qwen Program inspector argument");
      }
    }
    if (iterations == 0) {
      throw std::invalid_argument("Qwen Program iteration count is zero");
    }
    const auto max_context = static_cast<std::uint32_t>(std::stoul(argv[3]));
    auto artifact = omlx::m5::ArtifactModel::open(argv[1], should_materialize);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    auto context = omlx::m5::MetalContext::create();
    auto materialization = omlx::m5::plan_model_materialization(
        artifact, context,
        {.components = {omlx::m5::ModelComponent::text},
         .require_verified_payload = should_materialize});
    auto layers =
        omlx::m5::build_qwen_layer_catalog(artifact, config, max_context, 64);
    auto profiles =
        omlx::m5::compile_topology_profiles(omlx::m5::describe_qwen_topology(
            artifact, config, layers, max_context, 64));
    auto dispatch =
        omlx::m5::build_qwen_program_dispatch_table(artifact, layers, profiles);
    const auto plan = omlx::m5::plan_qwen_program(
        artifact, config, materialization, layers, profiles, dispatch.stats(),
        {.max_context = max_context, .kv_page_tokens = 64});

    nlohmann::json result = {
        {"schema", "omlx-m5-b4-3-qwen-program-v1"},
        {"evidence_scope",
         should_materialize
             ? "small artifact real Metal ownership; not full-model or M5 "
               "Ultra performance"
             : "full-model static resource plan; weights not resident"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"materialized", should_materialize},
        {"plan",
         {{"architecture", omlx::m5::to_string(plan.architecture)},
          {"quant_profile", omlx::m5::to_string(plan.quant_profile)},
          {"topology_signature", plan.topology_signature},
          {"max_context", plan.max_context},
          {"kv_page_tokens", plan.kv_page_tokens},
          {"layer_count", plan.layer_count},
          {"stage_count", plan.stage_count},
          {"dispatch_count", plan.dispatch_count},
          {"profile_count", plan.profile_count},
          {"weight_segment_count", plan.weight_segment_count},
          {"weight_bytes", plan.weight_bytes},
          {"workspace_bytes", plan.workspace_bytes},
          {"state_bytes", plan.state_bytes},
          {"parameter_bytes", plan.parameter_bytes},
          {"residency_allocation_count", plan.residency_allocation_count},
          {"required_object_count", plan.required_object_ids.size()},
          {"dispatch_table_sealed", plan.dispatch_table.sealed},
          {"dispatch_candidate_count", plan.dispatch_table.candidate_count},
          {"installed_dispatch_candidate_count",
           plan.dispatch_table.installed_candidate_count},
          {"planned_dispatch_candidate_count",
           plan.dispatch_table.planned_candidate_count}}},
    };
    if (should_materialize) {
      auto weights = omlx::m5::materialize_model(
          artifact, std::move(materialization), context);
      auto program = omlx::m5::assemble_qwen_program(
          plan, std::move(context), std::move(weights), std::move(layers),
          std::move(dispatch), std::move(profiles));
      const auto address_before = program.address_snapshot();
      const auto registry_before = program.registry_stats();
      const auto decode =
          program.select_profile(omlx::m5::TopologyProfileMode::decode, 1, 1);
      if (!decode.matched()) {
        throw std::runtime_error("fixture decode profile did not match");
      }
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const std::array<std::uint32_t, 1> tokens = {iteration % 63U};
        const std::array<std::uint32_t, 1> positions = {iteration};
        const std::array<std::uint32_t, 4> pages = {0, 1, 2, 3};
        program.update_parameters(decode.profile_index,
                                  {.token_ids = tokens,
                                   .positions = positions,
                                   .kv_frontier = iteration,
                                   .page_table = pages});
      }
      const auto registry_after = program.registry_stats();
      const auto &stats = program.stats();
      result["runtime"] = {
          {"iterations", iterations},
          {"addresses_stable", program.address_snapshot() == address_before},
          {"weight_segment_count", address_before.weight_segments.size()},
          {"parameter_table_count", address_before.parameter_tables.size()},
          {"workspace_address_nonzero", address_before.workspace != 0},
          {"state_address_nonzero", address_before.state != 0},
          {"residency_committed", program.residency().committed()},
          {"residency_allocations", program.residency().allocation_count()},
          {"residency_bytes", program.residency().allocated_size()},
          {"static_stage_plan_count", stats.static_stage_plan_count},
          {"static_plane_binding_count", stats.static_plane_binding_count},
          {"construction_artifact_lookups",
           stats.construction_artifact_lookups},
          {"token_loop_artifact_lookups", stats.token_loop_artifact_lookups},
          {"token_loop_pipeline_compiles", stats.token_loop_pipeline_compiles},
          {"token_loop_weight_uploads", stats.token_loop_weight_uploads},
          {"registry_delta",
           {{"buffer_count", registry_after.current_buffer_count -
                                 registry_before.current_buffer_count},
            {"upload_count",
             registry_after.upload_count - registry_before.upload_count},
            {"pipeline_compiles", registry_after.pipelines_compiled -
                                      registry_before.pipelines_compiled},
            {"dynamic_updates",
             registry_after.dynamic_parameter_updates -
                 registry_before.dynamic_parameter_updates}}},
      };
    }
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Qwen Program inspection failed: " << error.what() << '\n';
    return 1;
  }
}
