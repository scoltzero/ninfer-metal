#include <array>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

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

omlx::m5::QwenDFlashProgram
make_dflash(const std::filesystem::path &root,
            const std::filesystem::path &draft_artifact,
            const std::filesystem::path &draft_config, const std::string &key,
            std::span<const std::uint32_t> prompt,
            std::uint32_t draft_tokens = 7) {
  const auto artifact = root / (key + ".omlxm5");
  const auto config = root / key / "config.json";
  return omlx::m5::QwenDFlashProgram::create(
      make_program(artifact, config),
      omlx::m5::DFlash2DraftProgram::create(
          draft_artifact, draft_config, OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH,
          {.max_context = 64, .max_draft_tokens = 7}),
      prompt, {.draft_tokens = draft_tokens});
}

void check_profile(const std::filesystem::path &root,
                   const std::filesystem::path &draft_artifact,
                   const std::filesystem::path &draft_config,
                   const std::string &key) {
  const std::array<std::uint32_t, 5> prompt = {1, 7, 3, 9, 2};
  auto dflash = make_dflash(root, draft_artifact, draft_config, key, prompt);

  for (std::uint32_t draft_tokens = 1; draft_tokens <= 7; ++draft_tokens) {
    auto configured = make_dflash(root, draft_artifact, draft_config, key,
                                  prompt, draft_tokens);
    const auto result = configured.run_round();
    expect(configured.configured_draft_tokens() == draft_tokens &&
               result.proposal_count == draft_tokens &&
               result.verify_width == draft_tokens + 1,
           key + ": configured draft window was not honored");
  }

  for (std::uint32_t accepted = 0; accepted <= 7; ++accepted) {
    dflash.reset(prompt);
    const auto result =
        dflash.run_round({.forced_accepted_proposals = accepted});
    expect(result.verify_width == 8 && result.proposals.size() == 7 &&
               result.accepted_proposals == accepted &&
               result.published_outputs.size() == accepted + 1 &&
               result.target_weight_publications == 1 &&
               result.target_verify_calls == 1 &&
               result.draft_dispatches == 1 &&
               result.teacher_forced_state_parity &&
               result.stage == omlx::m5::RoundStage::committed,
           key + ": accepted-length transaction coverage failed at N=" +
               std::to_string(accepted) +
               " actual=" + std::to_string(result.accepted_proposals) +
               " published=" + std::to_string(result.published_outputs.size()) +
               " stage=" + std::to_string(static_cast<int>(result.stage)) +
               " parity=" + std::to_string(result.teacher_forced_state_parity));
  }

  for (std::uint32_t proposals = 0; proposals <= 7; ++proposals) {
    dflash.reset(prompt);
    const auto result = dflash.run_round({.remaining_budget = proposals + 1});
    expect(result.verify_width == proposals + 1 &&
               result.proposals.size() == proposals &&
               result.target_weight_publications == 1 &&
               result.target_verify_calls == 1 &&
               result.draft_dispatches == (proposals == 0 ? 0 : 1) &&
               result.teacher_forced_state_parity &&
               result.stage == omlx::m5::RoundStage::committed,
           key + ": T=1..8 tail coverage failed");
  }

  dflash.reset(prompt);
  const auto aborted = dflash.run_round({.abort_after_verify = true});
  expect(aborted.stage == omlx::m5::RoundStage::aborted &&
             aborted.target_state_before == aborted.target_state_after &&
             aborted.draft_state_before == aborted.draft_state_after,
         key + ": abort mutated state");
  dflash.reset(prompt);
  const auto cancelled = dflash.run_round({.cancelled = true});
  expect(cancelled.stage == omlx::m5::RoundStage::cancelled &&
             cancelled.finish_reason ==
                 omlx::m5::PreviewFinishReason::cancelled &&
             cancelled.target_state_before == cancelled.target_state_after &&
             cancelled.draft_state_before == cancelled.draft_state_after,
         key + ": cancel mutated state");

  dflash.reset(prompt);
  const auto probe = dflash.run_round({.forced_accepted_proposals = 7});
  expect(!probe.licensed_outputs.empty(), key + ": no licensed output");
  dflash.reset(prompt);
  const auto eos =
      dflash.run_round({.forced_accepted_proposals = 7,
                        .eos_token_ids = {probe.licensed_outputs.front()}});
  expect(eos.finish_reason == omlx::m5::PreviewFinishReason::eos &&
             eos.published_outputs.empty() &&
             eos.stage == omlx::m5::RoundStage::aborted,
         key + ": EOS preview semantics failed");
  dflash.reset(prompt);
  const auto stopped =
      dflash.run_round({.forced_accepted_proposals = 7,
                        .stop_sequences = {{probe.licensed_outputs.front()}}});
  expect(stopped.finish_reason == omlx::m5::PreviewFinishReason::stop &&
             stopped.published_outputs.empty(),
         key + ": stop preview semantics failed");
  dflash.reset(prompt);
  const auto budget = dflash.run_round({.remaining_budget = 1});
  expect(budget.finish_reason == omlx::m5::PreviewFinishReason::length &&
             budget.published_outputs.size() == 1,
         key + ": budget preview semantics failed");

  // Regression tests for MIVE-18: multi-token stop_sequences across round boundaries.
  {
    dflash.reset(prompt);
    const auto base_r1 = dflash.run_round({.forced_accepted_proposals = 0});
    const auto base_r2 = dflash.run_round({.forced_accepted_proposals = 0});
    expect(base_r1.published_outputs.size() == 1 &&
               base_r2.published_outputs.size() == 1,
           key + ": MIVE-18 baseline rounds did not emit 1 token each");
    const std::int32_t tok_a = base_r1.published_outputs.front();
    const std::int32_t tok_b = base_r2.published_outputs.front();
    const std::int32_t non_b = (tok_b + 1) % 64;

    // 1) Exact match across Round 1 -> Round 2 boundary: hold back tok_a in
    // Round 1, detect {tok_a, tok_b} in Round 2, leak neither token.
    dflash.reset(prompt);
    const auto cross_r1 = dflash.run_round(
        {.forced_accepted_proposals = 0, .stop_sequences = {{tok_a, tok_b}}});
    expect(cross_r1.finish_reason == omlx::m5::PreviewFinishReason::none &&
               cross_r1.published_outputs.empty() &&
               cross_r1.stage == omlx::m5::RoundStage::committed,
           key + ": MIVE-18 Round 1 did not hold back stop prefix");
    const auto cross_r2 = dflash.run_round(
        {.forced_accepted_proposals = 0, .stop_sequences = {{tok_a, tok_b}}});
    expect(cross_r2.finish_reason == omlx::m5::PreviewFinishReason::stop &&
               cross_r2.published_outputs.empty() &&
               cross_r2.stage == omlx::m5::RoundStage::aborted,
           key + ": MIVE-18 Round 2 missed cross-boundary stop sequence");

    // 2) False-alarm holdback flush on terminal budget exhaustion.
    dflash.reset(prompt);
    static_cast<void>(dflash.run_round({.forced_accepted_proposals = 0}));
    const auto nat_r2 = dflash.run_round({.remaining_budget = 1});
    expect(nat_r2.published_outputs.size() == 1,
           key + ": MIVE-18 natural budget=1 round did not emit 1 token");
    const std::int32_t tok_b_nat = nat_r2.published_outputs.front();
    const std::int32_t non_b_nat = (tok_b_nat + 1) % 64;
    dflash.reset(prompt);
    const auto flush_r1 = dflash.run_round(
        {.forced_accepted_proposals = 0,
         .stop_sequences = {{tok_a, non_b_nat}}});
    expect(flush_r1.finish_reason == omlx::m5::PreviewFinishReason::none &&
               flush_r1.published_outputs.empty() &&
               flush_r1.stage == omlx::m5::RoundStage::committed,
           key + ": MIVE-18 false-alarm Round 1 did not hold back prefix");
    const auto flush_r2 = dflash.run_round(
        {.remaining_budget = 1, .stop_sequences = {{tok_a, non_b_nat}}});
    expect(flush_r2.finish_reason == omlx::m5::PreviewFinishReason::length &&
               flush_r2.published_outputs ==
                   std::vector<std::int32_t>({tok_a, tok_b_nat}),
           key + ": MIVE-18 false-alarm Round 2 did not flush held prefix");

    // 3) False-alarm holdback flush when Round 2 terminates on EOS.
    dflash.reset(prompt);
    const auto eos_r1 = dflash.run_round(
        {.forced_accepted_proposals = 0, .stop_sequences = {{tok_a, non_b}}});
    expect(eos_r1.finish_reason == omlx::m5::PreviewFinishReason::none &&
               eos_r1.published_outputs.empty(),
           key + ": MIVE-18 EOS Round 1 did not hold back prefix");
    const auto eos_r2 = dflash.run_round({.forced_accepted_proposals = 0,
                                          .eos_token_ids = {tok_b},
                                          .stop_sequences = {{tok_a, non_b}}});
    expect(eos_r2.finish_reason == omlx::m5::PreviewFinishReason::eos &&
               eos_r2.published_outputs == std::vector<std::int32_t>({tok_a}),
           key + ": MIVE-18 EOS Round 2 dropped held stop prefix");
  }

  dflash.reset(prompt);
  const auto sampled_a = dflash.run_round({.temperature = 1.0F, .seed = 123});
  dflash.reset(prompt);
  const auto sampled_b = dflash.run_round({.temperature = 1.0F, .seed = 123});
  expect(sampled_a.published_outputs == sampled_b.published_outputs &&
             !sampled_a.published_outputs.empty() &&
             sampled_a.draft_dispatches == 1 &&
             sampled_a.stage == omlx::m5::RoundStage::committed,
         key + ": sampled DFlash2 verification is not deterministic");

  const auto &stats = dflash.stats();
  expect(stats.rounds >= 20 && stats.target_weight_publications > 0 &&
             stats.target_weight_publications == stats.target_verify_calls &&
             stats.verify_width_histogram.size() == 9 &&
             stats.accepted_length_histogram.size() == 8,
         key + ": DFlash statistics are incomplete");
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_b5_dflash_program_tests TARGET_FIXTURE_DIRECTORY "
          "DRAFT_ARTIFACT DRAFT_CONFIG");
    }
    const std::filesystem::path root = argv[1];
    const std::filesystem::path draft_artifact = argv[2];
    const std::filesystem::path draft_config = argv[3];
    for (const std::string key :
         {"ar-dense-oq", "ar-dense-mxfp4", "ar-moe-oq", "ar-moe-mxfp4"}) {
      check_profile(root, draft_artifact, draft_config, key);
    }
    std::cout << "B5 true DFlash2 transaction tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B5 true DFlash2 transaction failure: " << error.what()
              << '\n';
    return 1;
  }
}
