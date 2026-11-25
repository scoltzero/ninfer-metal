#include <array>
#include <cstdint>
#include <filesystem>
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

omlx::m5::QwenProgram make_program(const std::filesystem::path &artifact_path,
                                   const std::filesystem::path &config_path) {
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
  auto weights = omlx::m5::materialize_model(
      artifact, std::move(materialization), context);
  auto program = omlx::m5::assemble_qwen_program(
      plan, std::move(context), std::move(weights), std::move(layers),
      std::move(dispatch), std::move(profiles));
  program.install_autoregressive_executor(OMLX_M5_QWEN_AR_METALLIB_PATH);
  return program;
}

void check_profile(const std::filesystem::path &root, std::string key,
                   omlx::m5::QwenArchitecture architecture,
                   omlx::m5::QwenQuantProfile quant_profile) {
  auto program =
      make_program(root / (key + ".omlxm5"), root / key / "config.json");
  expect(program.build_plan().architecture == architecture &&
             program.build_plan().quant_profile == quant_profile &&
             program.build_plan().full_attention_layers == 1 &&
             program.build_plan().gdn_layers == 1,
         key + ": Program profile identity mismatch");
  const auto addresses = program.autoregressive_address_snapshot();
  const auto registry_before = program.registry_stats();
  const std::array<std::uint32_t, 5> prompt = {1, 7, 3, 9, 2};
  program.reset_autoregressive();
  const auto baseline = program.prefill(prompt);
  expect(baseline.route == omlx::m5::QwenArExecutionRoute::execute_batch &&
             baseline.profile_capacity == 16 &&
             baseline.oracle_max_abs_error <= 2.0e-5F,
         key + ": execute_batch prefill oracle mismatch");

  std::vector<std::uint32_t> medium(17);
  for (std::uint32_t index = 0; index < medium.size(); ++index) {
    medium[index] = index % 63U;
  }
  program.reset_autoregressive();
  const auto medium_result =
      program.execute_ar(medium, omlx::m5::TopologyProfileMode::prefill,
                         omlx::m5::QwenArExecutionRoute::execute_batch);
  expect(medium_result.profile_capacity == 32 &&
             medium_result.kv_checksum != medium_result.gdn_checksum,
         key + ": state/profile-capacity update failed");

  const std::array<std::uint32_t, 1> decode = {11};
  program.reset_autoregressive();
  const auto direct =
      program.execute_ar(decode, omlx::m5::TopologyProfileMode::decode,
                         omlx::m5::QwenArExecutionRoute::execute_batch);
  program.reset_autoregressive();
  const auto icb =
      program.execute_ar(decode, omlx::m5::TopologyProfileMode::decode,
                         omlx::m5::QwenArExecutionRoute::icb_profile);
  expect(direct.sampled_token == icb.sampled_token &&
             direct.kv_checksum == icb.kv_checksum &&
             direct.gdn_checksum == icb.gdn_checksum && !icb.fallback &&
             icb.submission.dispatch_count == 4,
         key + ": ICB/execute_batch parity failed");

  const auto first = program.generate(prompt, 256);
  expect(!first.failed && !first.cancelled && first.events.size() == 256 &&
             first.events.front().route == "execute_batch" &&
             first.events.back().route == "icb_profile",
         key + ": 256-token runner failed");
  const auto first_tokens = first.events;
  const auto second = program.generate(prompt, 256);
  expect(!second.failed && second.events.size() == first_tokens.size(),
         key + ": warm second request failed");
  for (std::size_t index = 0; index < first_tokens.size(); ++index) {
    expect(first_tokens[index].token == second.events[index].token,
           key + ": reset left sequence-state pollution");
  }

  const auto sampled = program.generate(
      prompt, 16,
      {.temperature = 0.8F, .top_k = 16, .top_p = 0.9F, .seed = 17});
  expect(!sampled.failed && sampled.events.size() == 16,
         key + ": top-k/top-p sampling failed");
  std::uint32_t cancellation_checks = 0;
  const auto cancelled = program.generate(
      prompt, 32, {}, [&] { return cancellation_checks++ >= 3; });
  expect(cancelled.cancelled && !cancelled.failed &&
             cancelled.events.size() == 3,
         key + ": cancellation failed");
  program.inject_failure_after(0);
  const auto failed = program.generate(prompt, 4);
  expect(failed.failed && !failed.error.empty(),
         key + ": error injection failed");
  program.clear_failure_injection();

  const auto registry_after = program.registry_stats();
  const auto &stats = program.autoregressive_stats();
  expect(program.autoregressive_address_snapshot() == addresses &&
             registry_after.current_buffer_count ==
                 registry_before.current_buffer_count &&
             registry_after.upload_count == registry_before.upload_count &&
             registry_after.pipelines_compiled ==
                 registry_before.pipelines_compiled &&
             stats.icb_recordings == 7 && stats.icb_replays >= 512 &&
             stats.token_loop_plan_builds == 0 &&
             stats.token_loop_pipeline_compiles == 0 &&
             stats.token_loop_weight_uploads == 0,
         key + ": steady-state resource stability failed");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: omlx_m5_b4_ar_tests FIXTURE_DIRECTORY");
    }
    const std::filesystem::path root = argv[1];
    check_profile(root, "ar-dense-oq", omlx::m5::QwenArchitecture::dense,
                  omlx::m5::QwenQuantProfile::oq_affine);
    check_profile(root, "ar-dense-mxfp4", omlx::m5::QwenArchitecture::dense,
                  omlx::m5::QwenQuantProfile::mxfp4);
    check_profile(root, "ar-moe-oq", omlx::m5::QwenArchitecture::moe,
                  omlx::m5::QwenQuantProfile::oq_affine);
    check_profile(root, "ar-moe-mxfp4", omlx::m5::QwenArchitecture::moe,
                  omlx::m5::QwenQuantProfile::mxfp4);
    std::cout << "B4.4-B4.6 AR/ICB/runner tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B4.4-B4.6 failure: " << error.what() << '\n';
    return 1;
  }
}
