#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dflash2.h"
#include "omlx_m5/dispatch.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/qwen_program.h"
#include "omlx_m5/topology_profile.h"

#ifndef OMLX_M5_QWEN_AR_METALLIB_PATH
#error "OMLX_M5_QWEN_AR_METALLIB_PATH is required"
#endif
#ifndef OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH
#error "OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH is required"
#endif

namespace {

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

nlohmann::json inspect_profile(const std::filesystem::path &root,
                               const std::filesystem::path &draft_artifact,
                               const std::filesystem::path &draft_config,
                               const std::string &key) {
  const auto artifact_path = root / (key + ".omlxm5");
  const auto config_path = root / key / "config.json";
  auto artifact = omlx::m5::ArtifactModel::open(artifact_path, false);
  const std::array<std::uint32_t, 5> prompt = {1, 7, 3, 9, 2};
  auto program = omlx::m5::QwenDFlashProgram::create(
      make_program(artifact_path, config_path),
      omlx::m5::DFlash2DraftProgram::create(
          draft_artifact, draft_config, OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH,
          {.max_context = 64, .max_draft_tokens = 7}),
      prompt, {.draft_tokens = 7});

  nlohmann::json acceptance = nlohmann::json::array();
  for (std::uint32_t accepted = 0; accepted <= 7; ++accepted) {
    program.reset(prompt);
    const auto round =
        program.run_round({.forced_accepted_proposals = accepted});
    acceptance.push_back(
        {{"forced", accepted},
         {"accepted", round.accepted_proposals},
         {"verify_width", round.verify_width},
         {"licensed", round.licensed_outputs},
         {"published", round.published_outputs},
         {"correction", round.used_correction},
         {"bonus", round.used_bonus},
         {"state_parity", round.teacher_forced_state_parity},
         {"target_weight_publications", round.target_weight_publications},
         {"target_verify_calls", round.target_verify_calls},
         {"draft_dispatches", round.draft_dispatches},
         {"replay_dispatches", round.replay_dispatches}});
  }
  nlohmann::json tails = nlohmann::json::array();
  for (std::uint32_t proposals = 0; proposals <= 7; ++proposals) {
    program.reset(prompt);
    const auto round = program.run_round({.remaining_budget = proposals + 1});
    tails.push_back(
        {{"proposal_count", proposals},
         {"verify_width", round.verify_width},
         {"published", round.published_outputs.size()},
         {"state_parity", round.teacher_forced_state_parity},
         {"draft_dispatches", round.draft_dispatches},
         {"target_weight_publications", round.target_weight_publications}});
  }
  program.reset(prompt);
  const auto aborted = program.run_round({.abort_after_verify = true});
  program.reset(prompt);
  const auto cancelled = program.run_round({.cancelled = true});
  program.reset(prompt);
  const auto termination_probe =
      program.run_round({.forced_accepted_proposals = 7});
  if (termination_probe.licensed_outputs.empty()) {
    throw std::runtime_error("DFlash termination probe has no output");
  }
  program.reset(prompt);
  const auto eos = program.run_round(
      {.forced_accepted_proposals = 7,
       .eos_token_ids = {termination_probe.licensed_outputs.front()}});
  program.reset(prompt);
  const auto stopped = program.run_round(
      {.forced_accepted_proposals = 7,
       .stop_sequences = {{termination_probe.licensed_outputs.front()}}});
  program.reset(prompt);
  const auto budget = program.run_round({.remaining_budget = 1});
  program.reset(prompt);
  const auto sampled_a = program.run_round({.temperature = 1.0F, .seed = 123});
  program.reset(prompt);
  const auto sampled_b = program.run_round({.temperature = 1.0F, .seed = 123});
  program.reset(prompt);
  const auto chained_a = program.run_round();
  const auto anchor_after_a = program.pending_anchor();
  const auto chained_b = program.run_round();

  const auto &stats = program.stats();
  return {
      {"profile", key},
      {"source_repo", artifact.summary().source_repo},
      {"configured_draft_tokens", program.configured_draft_tokens()},
      {"acceptance", std::move(acceptance)},
      {"tails", std::move(tails)},
      {"abort",
       {{"stage", static_cast<int>(aborted.stage)},
        {"target_unchanged",
         aborted.target_state_before == aborted.target_state_after},
        {"draft_unchanged",
         aborted.draft_state_before == aborted.draft_state_after}}},
      {"cancel",
       {{"stage", static_cast<int>(cancelled.stage)},
        {"finish_reason",
         omlx::m5::finish_reason_name(cancelled.finish_reason)},
        {"target_unchanged",
         cancelled.target_state_before == cancelled.target_state_after},
        {"draft_unchanged",
         cancelled.draft_state_before == cancelled.draft_state_after}}},
      {"termination",
       {{"eos", eos.finish_reason == omlx::m5::PreviewFinishReason::eos &&
                    eos.published_outputs.empty()},
        {"stop", stopped.finish_reason == omlx::m5::PreviewFinishReason::stop &&
                     stopped.published_outputs.empty()},
        {"budget",
         budget.finish_reason == omlx::m5::PreviewFinishReason::length &&
             budget.published_outputs.size() == 1}}},
      {"sampling",
       {{"seed_deterministic",
         sampled_a.published_outputs == sampled_b.published_outputs},
        {"published", sampled_a.published_outputs.size()},
        {"draft_dispatches", sampled_a.draft_dispatches}}},
      {"chained_rounds",
       {{"first_published", chained_a.published_outputs.size()},
        {"second_published", chained_b.published_outputs.size()},
        {"anchor_changed", anchor_after_a != prompt.back()},
        {"frontier", program.sequence_state().target_frontier}}},
      {"stats",
       {{"rounds", stats.rounds},
        {"committed_rounds", stats.committed_rounds},
        {"aborted_rounds", stats.aborted_rounds},
        {"cancelled_rounds", stats.cancelled_rounds},
        {"proposal_tokens", stats.proposal_tokens},
        {"accepted_proposals", stats.accepted_proposals},
        {"correction_rounds", stats.correction_rounds},
        {"bonus_rounds", stats.bonus_rounds},
        {"target_weight_publications", stats.target_weight_publications},
        {"target_verify_calls", stats.target_verify_calls},
        {"draft_dispatches", stats.draft_dispatches},
        {"replay_dispatches", stats.replay_dispatches},
        {"verify_width_histogram", stats.verify_width_histogram},
        {"accepted_length_histogram", stats.accepted_length_histogram}}},
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_dflash_program TARGET_FIXTURE_DIRECTORY "
          "DRAFT_ARTIFACT DRAFT_CONFIG");
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path draft_artifact = argv[2];
    const std::filesystem::path draft_config = argv[3];
    nlohmann::json profiles = nlohmann::json::array();
    for (const std::string key :
         {"ar-dense-oq", "ar-dense-mxfp4", "ar-moe-oq", "ar-moe-mxfp4"}) {
      profiles.push_back(
          inspect_profile(root, draft_artifact, draft_config, key));
    }
    const nlohmann::json result = {
        {"schema", "omlx-m5-b5-dflash2-program-v3"},
        {"evidence_scope",
         "small real Metal DFlash2 masked-block Program with dflash-mlx oracle "
         "plus full real companion static validation; not full 27B target "
         "runtime or M5 performance"},
        {"spec_backend_contract", "none|mtp|dflash|dflash2"},
        {"r1_implemented_backends", {"none", "dflash2"}},
        {"configured_draft_tokens", 7},
        {"target_verify_width", 8},
        {"draft_threadgroup_width", 32},
        {"proposal_engine", "single_pass_masked_block"},
        {"parallel_draft_backbone_in_program", true},
        {"dynamic_convolution_in_program", true},
        {"candidate_selector_in_program", true},
        {"b5_complete", true},
        {"profiles", std::move(profiles)},
    };
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "DFlash Program inspection failed: " << error.what() << '\n';
    return 1;
  }
}
