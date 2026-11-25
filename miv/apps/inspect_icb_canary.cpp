#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "omlx_m5/metal.h"

#ifndef OMLX_M5_ICB_CANARY_METALLIB_PATH
#error "OMLX_M5_ICB_CANARY_METALLIB_PATH is required"
#endif

namespace {

using Json = nlohmann::json;

Json report_json(const omlx::m5::MetalIcbCanaryReport &report) {
  return {
      {"execution_route", report.execution_route},
      {"fallback_reason", report.fallback_reason},
      {"compute_icb_eligible", report.compute_icb_eligible},
      {"metal4_icb_eligible", report.metal4_icb_eligible},
      {"pipeline_icb_enabled", report.pipeline_icb_enabled},
      {"metal4_pipeline_icb_enabled", report.metal4_pipeline_icb_enabled},
      {"dependency_barrier_encoded", report.dependency_barrier_encoded},
      {"bit_exact", report.bit_exact},
      {"changing_parameters_observed", report.changing_parameters_observed},
      {"resources_released", report.resources_released},
      {"iterations", report.iterations},
      {"parameter_updates", report.parameter_updates},
      {"compared_iterations", report.compared_iterations},
      {"icb_recordings", report.icb_recordings},
      {"icb_resets", report.icb_resets},
      {"icb_execute_calls", report.icb_execute_calls},
      {"token_loop_icb_recordings", report.token_loop_icb_recordings},
      {"replay_command_buffer_commits", report.replay_command_buffer_commits},
      {"replay_dispatches", report.replay_dispatches},
      {"oracle_execute_batch_commits", report.oracle_execute_batch_commits},
      {"oracle_dispatches", report.oracle_dispatches},
      {"output_checksum", report.output_checksum},
      {"timing",
       {
           {"icb_record_cpu_ms", report.icb_record_cpu_ms},
           {"replay_cpu_encode_ms", report.replay_cpu_encode_ms},
           {"replay_wall_ms", report.replay_wall_ms},
           {"replay_gpu_ms", report.replay_gpu_ms},
           {"oracle_cpu_encode_ms", report.oracle_cpu_encode_ms},
           {"oracle_wall_ms", report.oracle_wall_ms},
           {"oracle_gpu_ms", report.oracle_gpu_ms},
       }},
  };
}

std::string resolve_metallib_path(const std::string &compiled,
                                  const char *argv0) {
  if (!compiled.empty() && std::filesystem::is_regular_file(compiled)) {
    return compiled;
  }
  if (!compiled.empty() && argv0 && *argv0) {
    const auto name = std::filesystem::path(compiled).filename();
    const auto exe_dir =
        std::filesystem::weakly_canonical(argv0).parent_path();
    for (const auto &candidate : {exe_dir / "metallib" / name,
                                  exe_dir.parent_path() / "metallib" / name}) {
      if (std::filesystem::is_regular_file(candidate)) {
        return candidate.string();
      }
    }
  }
  return compiled;
}

} // namespace

int main(int argc, const char *const *argv) {
  try {
    const std::uint64_t iterations = argc == 2 ? std::stoull(argv[1]) : 10'000;
    if (argc > 2 || iterations == 0) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_icb_canary [ITERATIONS]");
    }
    auto context = omlx::m5::MetalContext::create();
    context.register_library({
        .key = "b4.0.icb.canary.library",
        .path = resolve_metallib_path(OMLX_M5_ICB_CANARY_METALLIB_PATH, argv[0]),
    });
    const auto stage_a = context.warmup_pipeline({
        .key = "b4.0.icb.stage_a.pipeline",
        .library_key = "b4.0.icb.canary.library",
        .function_name = "omlx_m5_icb_stage_a",
        .variant = "icb",
        .support_indirect_command_buffers = true,
    });
    const auto stage_b = context.warmup_pipeline({
        .key = "b4.0.icb.stage_b.pipeline",
        .library_key = "b4.0.icb.canary.library",
        .function_name = "omlx_m5_icb_stage_b",
        .variant = "icb",
        .support_indirect_command_buffers = true,
    });
    const auto before = context.registry_stats();
    const auto replay = context.run_icb_canary(
        stage_a.descriptor.key, stage_b.descriptor.key,
        {.iterations = iterations, .element_count = 1024});
    const auto after_replay = context.registry_stats();
    const auto missing_barrier =
        context.run_icb_canary(stage_a.descriptor.key, stage_b.descriptor.key,
                               {.iterations = 8,
                                .element_count = 1024,
                                .omit_dependency_barrier = true});
    const auto forced_fallback =
        context.run_icb_canary(stage_a.descriptor.key, stage_b.descriptor.key,
                               {.iterations = 8,
                                .element_count = 1024,
                                .force_execute_batch_fallback = true});
    const auto &capabilities = context.capabilities();
    const auto &probe = capabilities.profile.probe;
    const Json output = {
        {"schema", "omlx-m5-b4-0-icb-canary-v1"},
        {"evidence_scope",
         "M4/macOS runtime ICB correctness and fallback; not M5 Ultra "
         "performance"},
        {"hardware",
         {
             {"device_name", capabilities.device_name},
             {"architecture_name", capabilities.architecture_name},
             {"apple_gpu_family", probe.apple_gpu_family},
         }},
        {"software",
         {
             {"runtime_os", capabilities.runtime_os},
             {"sdk_max_allowed", probe.sdk_max_allowed},
         }},
        {"capability",
         {
             {"compute_icb_api", probe.compute_icb_api},
             {"compute_icb_barrier", probe.compute_icb_barrier},
             {"residency_sets", probe.residency_sets},
             {"metal4_headers", probe.metal4_headers},
             {"metal4_command_queue", probe.metal4_command_queue},
             {"metal4_command_allocator", probe.metal4_command_allocator},
             {"metal4_command_buffer", probe.metal4_command_buffer},
             {"metal4_compute_encoder", probe.metal4_compute_encoder},
             {"metal4_compiler", probe.metal4_compiler},
             {"metal4_argument_table", probe.metal4_argument_table},
             {"compute_icb_eligible",
              capabilities.profile.compute_icb_eligible},
             {"metal4_icb_eligible", capabilities.profile.metal4_icb_eligible},
             {"metal4_direct_eligible",
              capabilities.profile.metal4_direct_eligible},
             {"icb_fallback_reasons",
              capabilities.profile.icb_fallback_reasons},
         }},
        {"pipeline",
         {
             {"stage_a_icb_enabled", stage_a.indirect_command_buffers_enabled},
             {"stage_b_icb_enabled", stage_b.indirect_command_buffers_enabled},
             {"stage_a_metal4_icb_enabled",
              stage_a.metal4_indirect_command_buffers_enabled},
             {"stage_b_metal4_icb_enabled",
              stage_b.metal4_indirect_command_buffers_enabled},
         }},
        {"replay", report_json(replay)},
        {"missing_barrier_gate", report_json(missing_barrier)},
        {"forced_fallback", report_json(forced_fallback)},
        {"registry_delta",
         {
             {"setup_uploads", after_replay.upload_count - before.upload_count},
             {"token_loop_pipeline_compiles",
              after_replay.pipelines_compiled - before.pipelines_compiled},
             {"command_plans_built",
              after_replay.command_plans_built - before.command_plans_built},
             {"command_buffers_submitted",
              after_replay.command_buffers_submitted -
                  before.command_buffers_submitted},
             {"metal4_command_buffers_submitted",
              after_replay.metal4_command_buffers_submitted -
                  before.metal4_command_buffers_submitted},
             {"classic_command_buffers_submitted",
              after_replay.classic_command_buffers_submitted -
                  before.classic_command_buffers_submitted},
             {"dispatches_submitted",
              after_replay.dispatches_submitted - before.dispatches_submitted},
         }},
    };
    std::cout << output.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "ICB canary inspection failed: " << error.what() << '\n';
    return 1;
  }
}
