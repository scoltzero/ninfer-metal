#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/dflash2.h"

#ifndef OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH
#error "OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH is required"
#endif

namespace {

using Json = nlohmann::json;

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

std::vector<float> flatten(const Json &value) {
  std::vector<float> result;
  const auto visit = [&](const auto &self, const Json &node) -> void {
    if (node.is_array()) {
      for (const auto &child : node) {
        self(self, child);
      }
      return;
    }
    result.push_back(node.get<float>());
  };
  visit(visit, value);
  return result;
}

float maximum_error(std::span<const float> actual,
                    std::span<const float> expected) {
  if (actual.size() != expected.size()) {
    throw std::runtime_error("DFlash2 oracle shape mismatch");
  }
  float result = 0.0F;
  for (std::size_t index = 0; index < actual.size(); ++index) {
    if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
      throw std::runtime_error("DFlash2 non-finite value at index " +
                               std::to_string(index));
    }
    result = std::max(result, std::abs(actual[index] - expected[index]));
  }
  return result;
}

void check_real_companion(const std::filesystem::path &artifact_path,
                          const std::filesystem::path &config_path) {
  const auto config = omlx::m5::load_dflash2_model_config(config_path);
  auto artifact = omlx::m5::ArtifactModel::open(artifact_path, false);
  const auto validation =
      omlx::m5::validate_dflash2_artifact(artifact, config, false);
  expect(config.hidden_size == 5120 && config.layer_count == 5 &&
             config.block_size == 8 && config.selector_top_k == 16 &&
             config.selector_rank == 256 && validation.object_count == 81 &&
             validation.companion_complete &&
             !validation.shared_embedding_present &&
             !validation.proposal_head_present,
         "real Qwen3.8 DFlash2 companion contract mismatch");
}

void check_fixture(const std::filesystem::path &artifact_path,
                   const std::filesystem::path &config_path,
                   const std::filesystem::path &oracle_path) {
  std::ifstream input(oracle_path);
  Json oracle;
  input >> oracle;
  expect(oracle.at("schema") == "omlx-m5-b5-dflash2-oracle-v1",
         "DFlash2 oracle schema mismatch");
  const auto context_tokens = oracle.at("context_tokens").get<std::uint32_t>();
  const auto anchor = oracle.at("anchor_token").get<std::int32_t>();
  const auto target_features = flatten(oracle.at("target_features"));
  const auto expected_hidden = flatten(oracle.at("hidden"));
  const auto expected_logits = flatten(oracle.at("logits"));
  const auto expected_proposals =
      oracle.at("proposals").get<std::vector<std::int32_t>>();

  auto program = omlx::m5::DFlash2DraftProgram::create(
      artifact_path, config_path, OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH,
      {.max_context = 64, .max_draft_tokens = 7});
  const auto greedy = program.propose(anchor, target_features, context_tokens,
                                      7, 0.0F, 17, context_tokens);
  const auto hidden_error = maximum_error(greedy.hidden, expected_hidden);
  const auto logits_error = maximum_error(greedy.logits, expected_logits);
  expect(std::isfinite(hidden_error) && hidden_error < 0.08F &&
             std::isfinite(logits_error) && logits_error < 0.08F,
         "Metal DFlash2 disagrees with dflash-mlx oracle: hidden=" +
             std::to_string(hidden_error) +
             " logits=" + std::to_string(logits_error));
  expect(greedy.selection.tokens == expected_proposals,
         "DFlash2 greedy selector disagrees with dflash-mlx");
  expect(greedy.width == 8 && greedy.draft_dispatches == 1 &&
             greedy.parallel_backbone && greedy.dynamic_convolution &&
             greedy.conditional_selector &&
             greedy.position_parallel_threads == 32 &&
             greedy.submission.dispatch_count == 1,
         "DFlash2 did not execute one complete masked-block draft");

  for (std::uint32_t draft_tokens = 1; draft_tokens <= 7; ++draft_tokens) {
    const auto round = program.propose(anchor, target_features, context_tokens,
                                       draft_tokens, 0.0F, 17, context_tokens);
    expect(round.width == draft_tokens + 1 &&
               round.selection.tokens.size() == draft_tokens &&
               round.candidate_rows.size() == draft_tokens &&
               round.draft_dispatches == 1 &&
               round.submission.dispatch_count == 1,
           "DFlash2 configured draft width was not honored");
  }

  const auto sampled_a = program.propose(
      anchor, target_features, context_tokens, 7, 1.0F, 29, context_tokens);
  const auto sampled_b = program.propose(
      anchor, target_features, context_tokens, 7, 1.0F, 29, context_tokens);
  expect(sampled_a.selection.tokens == sampled_b.selection.tokens,
         "DFlash2 sampling is not seed deterministic");
  bool observed_distribution = false;
  for (const auto &step : sampled_a.selection.steps) {
    for (const float p : step.probabilities) {
      expect(std::isfinite(p), "DFlash2 proposal probability is non-finite");
    }
    const auto total = std::accumulate(step.probabilities.begin(),
                                       step.probabilities.end(), 0.0F);
    expect(std::isfinite(total) && std::abs(total - 1.0F) < 1.0e-5F,
           "DFlash2 proposal distribution is not normalized");
    observed_distribution =
        observed_distribution ||
        std::count_if(step.probabilities.begin(), step.probabilities.end(),
                      [](float value) { return value > 1.0e-4F; }) > 1;
  }
  expect(observed_distribution,
         "DFlash2 sampling path collapsed to one-hot proposals");
  expect(program.stats().metal_dispatches == program.stats().rounds &&
             program.stats().rounds == 10,
         "DFlash2 dispatch accounting mismatch");

  // Regression test for MIVE-2/MIVK-1: context_tokens > 32 must compute all
  // context positions beyond thread_index 0..31 without stale workspace reads.
  {
    auto long_program = omlx::m5::DFlash2DraftProgram::create(
        artifact_path, config_path, OMLX_M5_DFLASH2_PROGRAM_METALLIB_PATH,
        {.max_context = 64, .max_draft_tokens = 7});
    const auto &cfg = long_program.config();
    const std::size_t target_width =
        cfg.target_layer_ids.size() * cfg.hidden_size;
    const auto require_finite = [](std::span<const float> values,
                                   const char *label) {
      for (const float v : values) {
        if (!std::isfinite(v)) {
          throw std::runtime_error(std::string("nonfinite DFlash2 output in ") +
                                   label);
        }
      }
    };

    // context_tokens = 33: flip ONLY token 32 (inside sliding_window=16).
    // Fixture weights use scale=0.08 across fc/norms/v_proj/o_proj/conv, so
    // dflash-mlx FP32 delta for 1 of 24 keys is ~4.3e-6 (hidden) / ~2.5e-7
    // (logits), whereas the buggy fixed-32-thread kernel yields exactly 0.0.
    std::vector<float> base_33(33U * target_width, 0.0F);
    for (std::size_t i = 0; i < 32U * target_width; ++i) {
      base_33[i] = float(int(i % 13U) - 6) * 0.05F;
    }
    auto pert_33 = base_33;
    for (std::size_t i = 0; i < target_width; ++i) {
      const float v = float(int(i % 9U) - 4) * 0.25F;
      base_33[32U * target_width + i] = -v;
      pert_33[32U * target_width + i] = v;
    }
    const auto r33_base =
        long_program.propose(anchor, base_33, 33U, 7U, 0.0F, 17U, 33U);
    const auto r33_pert =
        long_program.propose(anchor, pert_33, 33U, 7U, 0.0F, 17U, 33U);
    require_finite(r33_base.hidden, "context=33 base hidden");
    require_finite(r33_base.logits, "context=33 base logits");
    require_finite(r33_pert.hidden, "context=33 perturbed hidden");
    require_finite(r33_pert.logits, "context=33 perturbed logits");
    const float hdiff_33 = maximum_error(r33_base.hidden, r33_pert.hidden);
    const float ldiff_33 = maximum_error(r33_base.logits, r33_pert.logits);
    expect(hdiff_33 > 1.0e-6F && ldiff_33 > 5.0e-8F,
           "DFlash2 context_tokens=33 ignored token 32: hidden=" +
               std::to_string(hdiff_33) +
               " logits=" + std::to_string(ldiff_33));

    // context_tokens = 48: flip tokens 32..47 (all 16 in-window context keys;
    // dflash-mlx FP32 delta is ~4.6e-5 hidden / ~2.4e-6 logits).
    std::vector<float> base_48(48U * target_width, 0.0F);
    for (std::size_t i = 0; i < 32U * target_width; ++i) {
      base_48[i] = float(int(i % 13U) - 6) * 0.05F;
    }
    auto pert_48 = base_48;
    for (std::size_t t = 32U; t < 48U; ++t) {
      for (std::size_t i = 0; i < target_width; ++i) {
        const float v = float(int(i % 9U) - 4) * 0.25F;
        base_48[t * target_width + i] = -v;
        pert_48[t * target_width + i] = v;
      }
    }
    const auto r48_base =
        long_program.propose(anchor, base_48, 48U, 7U, 0.0F, 17U, 48U);
    const auto r48_pert =
        long_program.propose(anchor, pert_48, 48U, 7U, 0.0F, 17U, 48U);
    require_finite(r48_base.hidden, "context=48 base hidden");
    require_finite(r48_base.logits, "context=48 base logits");
    require_finite(r48_pert.hidden, "context=48 perturbed hidden");
    require_finite(r48_pert.logits, "context=48 perturbed logits");
    const float hdiff_48 = maximum_error(r48_base.hidden, r48_pert.hidden);
    const float ldiff_48 = maximum_error(r48_base.logits, r48_pert.logits);
    expect(hdiff_48 > 1.0e-5F && ldiff_48 > 5.0e-7F,
           "DFlash2 context_tokens=48 ignored tokens 32..47: hidden=" +
               std::to_string(hdiff_48) +
               " logits=" + std::to_string(ldiff_48));

    // context_tokens = 64: flip ONLY token 63 (dflash-mlx FP32 delta ~5.1e-6
    // hidden / ~3.1e-7 logits) and interleave a context=48 call to shift
    // workspace offsets and verify zero stale-workspace leakage.
    std::vector<float> base_64(64U * target_width, 0.0F);
    for (std::size_t i = 0; i < 63U * target_width; ++i) {
      base_64[i] = float(int(i % 11U) - 5) * 0.04F;
    }
    auto pert_64 = base_64;
    for (std::size_t i = 0; i < target_width; ++i) {
      const float v = float(int(i % 7U) - 3) * 0.3F;
      base_64[63U * target_width + i] = -v;
      pert_64[63U * target_width + i] = v;
    }
    const auto r64_a =
        long_program.propose(anchor, base_64, 64U, 7U, 0.0F, 17U, 64U);
    static_cast<void>(
        long_program.propose(anchor, pert_48, 48U, 7U, 0.0F, 17U, 48U));
    const auto r64_b =
        long_program.propose(anchor, pert_64, 64U, 7U, 0.0F, 17U, 64U);
    const auto r64_a2 =
        long_program.propose(anchor, base_64, 64U, 7U, 0.0F, 17U, 64U);
    require_finite(r64_a.hidden, "context=64 base hidden");
    require_finite(r64_a.logits, "context=64 base logits");
    require_finite(r64_b.hidden, "context=64 perturbed hidden");
    require_finite(r64_b.logits, "context=64 perturbed logits");
    const float hdiff_64 = maximum_error(r64_a.hidden, r64_b.hidden);
    const float ldiff_64 = maximum_error(r64_a.logits, r64_b.logits);
    expect(hdiff_64 > 1.0e-6F && ldiff_64 > 5.0e-8F,
           "DFlash2 context_tokens=64 ignored token 63: hidden=" +
               std::to_string(hdiff_64) +
               " logits=" + std::to_string(ldiff_64));
    expect(maximum_error(r64_a.hidden, r64_a2.hidden) == 0.0F &&
               maximum_error(r64_a.logits, r64_a2.logits) == 0.0F,
           "DFlash2 context_tokens=64 leaked stale workspace state");
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument(
          "usage: omlx_m5_b5_dflash2_program_tests FIXTURE_ARTIFACT "
          "FIXTURE_CONFIG ORACLE REAL_ARTIFACT REAL_CONFIG");
    }
    check_fixture(argv[1], argv[2], argv[3]);
    check_real_companion(argv[4], argv[5]);
    std::cout << "B5 true DFlash2 Program tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B5 true DFlash2 Program failure: " << error.what() << '\n';
    return 1;
  }
}
