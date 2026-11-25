#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_program.h"
#include "omlx_m5/topology_profile.h"

#ifndef OMLX_M5_QWEN_AR_METALLIB_PATH
#error "OMLX_M5_QWEN_AR_METALLIB_PATH is required"
#endif

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

void test_program(const std::string &artifact_path,
                  const std::string &config_path) {
  constexpr std::uint32_t max_context = 1024;
  auto artifact = omlx::m5::ArtifactModel::open(artifact_path, true);
  const auto config = omlx::m5::load_qwen_model_config(config_path);
  auto context = omlx::m5::MetalContext::create();
  auto materialization = omlx::m5::plan_model_materialization(
      artifact, context,
      {.components = {omlx::m5::ModelComponent::text},
       .require_verified_payload = true});
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
  expect(plan.layer_count == 1 && plan.profile_count == 7 &&
             plan.weight_segment_count > 0 && plan.weight_bytes > 0 &&
             plan.workspace_bytes > 0 && plan.state_bytes > 0 &&
             plan.dispatch_table.planned_candidate_count > 0 &&
             !plan.required_object_ids.empty(),
         "Qwen Program plan is incomplete");

  auto undersized = omlx::m5::QwenProgramBuildOptions{
      .max_context = max_context, .kv_page_tokens = 64, .state_bytes = 1};
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::plan_qwen_program(
            artifact, config, materialization, layers, profiles,
            dispatch.stats(), undersized));
      },
      "undersized Program state arena was accepted");

  auto weights = omlx::m5::materialize_model(
      artifact, std::move(materialization), context);
  auto program = omlx::m5::assemble_qwen_program(
      plan, std::move(context), std::move(weights), std::move(layers),
      std::move(dispatch), std::move(profiles));
  const auto addresses = program.address_snapshot();
  expect(program.residency().committed() &&
             program.residency().allocation_count() ==
                 plan.residency_allocation_count &&
             program.stats().static_stage_plan_count == plan.stage_count &&
             program.stats().token_loop_artifact_lookups == 0 &&
             !addresses.weight_segments.empty() && addresses.workspace != 0 &&
             addresses.state != 0 &&
             addresses.parameter_tables.size() == plan.profile_count,
         "Qwen Program ownership or residency is incomplete");

  const auto decode =
      program.select_profile(omlx::m5::TopologyProfileMode::decode, 1, 1);
  const auto prefill_profile =
      program.select_profile(omlx::m5::TopologyProfileMode::prefill, 1, 5);
  const auto fallback =
      program.select_profile(omlx::m5::TopologyProfileMode::prefill, 2, 16);
  expect(decode.matched() && prefill_profile.matched() && !fallback.matched() &&
             fallback.execution_route == "execute_batch",
         "Qwen Program profile routing is incorrect");

  const auto registry_before = program.registry_stats();
  constexpr std::uint32_t iterations = 1000;
  for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
    const std::array<std::uint32_t, 1> tokens = {iteration % 63U};
    const std::array<std::uint32_t, 1> positions = {iteration};
    const std::array<std::uint32_t, 4> pages = {0, 1, 2, 3};
    program.update_parameters(decode.profile_index, {.token_ids = tokens,
                                                     .positions = positions,
                                                     .kv_frontier = iteration,
                                                     .page_table = pages});
  }
  const auto registry_after = program.registry_stats();
  expect(program.address_snapshot() == addresses &&
             registry_after.current_buffer_count ==
                 registry_before.current_buffer_count &&
             registry_after.upload_count == registry_before.upload_count &&
             registry_after.pipelines_compiled ==
                 registry_before.pipelines_compiled &&
             program.sequence_state().parameter_generation > 0,
         "token updates changed Program addresses or static resources");

  const std::array<std::uint32_t, 2> dense_routes = {0, 1};
  const std::array<float, 2> dense_scores = {0.5F, 0.5F};
  const std::array<std::uint32_t, 1> token = {1};
  const std::array<std::uint32_t, 1> position = {1};
  expect_throws(
      [&] {
        program.update_parameters(decode.profile_index,
                                  {.token_ids = token,
                                   .positions = position,
                                   .route_indices = dense_routes,
                                   .route_scores = dense_scores});
      },
      "dense Program accepted MoE route data");

  program.install_autoregressive_executor(OMLX_M5_QWEN_AR_METALLIB_PATH);
  expect(program.autoregressive_installed() &&
             program.autoregressive_stats().icb_installed &&
             program.autoregressive_stats().profile_plan_count == 7,
         "Qwen AR executor/profile installation failed");
  const auto resources_before_ar = program.registry_stats();
  const std::array<std::uint32_t, 5> prompt = {1, 7, 3, 9, 2};
  program.reset_autoregressive();
  const auto prefill = program.prefill(prompt);
  expect(prefill.route == omlx::m5::QwenArExecutionRoute::execute_batch &&
             prefill.profile_capacity == 16 &&
             prefill.oracle_max_abs_error <= 2.0e-5F,
         "Qwen AR prefill baseline failed");

  const std::array<std::uint32_t, 1> decode_input = {11};
  program.reset_autoregressive();
  const auto direct =
      program.execute_ar(decode_input, omlx::m5::TopologyProfileMode::decode,
                         omlx::m5::QwenArExecutionRoute::execute_batch);
  program.reset_autoregressive();
  const auto icb =
      program.execute_ar(decode_input, omlx::m5::TopologyProfileMode::decode,
                         omlx::m5::QwenArExecutionRoute::icb_profile);
  expect(direct.sampled_token == icb.sampled_token &&
             direct.kv_checksum == icb.kv_checksum &&
             direct.gdn_checksum == icb.gdn_checksum &&
             icb.route == omlx::m5::QwenArExecutionRoute::icb_profile &&
             !icb.fallback,
         "Qwen AR ICB replay disagrees with execute_batch");

  const auto generated = program.generate(prompt, 256);
  expect(!generated.cancelled && !generated.failed &&
             generated.events.size() == 256 &&
             generated.events.front().route == "execute_batch" &&
             generated.events.back().route == "icb_profile",
         "Qwen AR 256-token runner failed");
  std::uint32_t cancellation_checks = 0;
  const auto cancelled = program.generate(
      prompt, 32, {}, [&] { return cancellation_checks++ >= 3; });
  expect(cancelled.cancelled && !cancelled.failed &&
             cancelled.events.size() == 3,
         "Qwen AR cancellation failed");
  program.inject_failure_after(0);
  const auto failed = program.generate(prompt, 4);
  expect(failed.failed && !failed.error.empty(),
         "Qwen AR error injection failed");
  program.clear_failure_injection();
  const auto second = program.generate(
      prompt, 8, {.temperature = 0.8F, .top_k = 16, .top_p = 0.9F, .seed = 17});
  expect(!second.failed && second.events.size() == 8,
         "Qwen AR sampled second request failed");
  const auto resources_after_ar = program.registry_stats();
  const auto &ar_stats = program.autoregressive_stats();
  expect(resources_after_ar.current_buffer_count ==
                 resources_before_ar.current_buffer_count &&
             resources_after_ar.upload_count ==
                 resources_before_ar.upload_count &&
             resources_after_ar.pipelines_compiled ==
                 resources_before_ar.pipelines_compiled &&
             ar_stats.icb_recordings == 7 && ar_stats.icb_replays >= 256 &&
             ar_stats.token_loop_plan_builds == 0 &&
             ar_stats.token_loop_pipeline_compiles == 0 &&
             ar_stats.token_loop_weight_uploads == 0,
         "Qwen AR token loop changed static resources");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: omlx_m5_b4_qwen_program_tests ARTIFACT CONFIG");
    }
    test_program(argv[1], argv[2]);
    std::cout << "B4.3 Qwen Program tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B4.3 Qwen Program failure: " << error.what() << '\n';
    return 1;
  }
}
