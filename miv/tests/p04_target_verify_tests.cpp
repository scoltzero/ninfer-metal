#include "omlx_m5/target_verify_gpu.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/qwen_forward.h"

using namespace omlx::m5;
namespace {
constexpr std::uint32_t vocabulary = 257;

void require(bool condition, const char *message) {
  if (!condition)
    throw std::runtime_error(message);
}

TargetDistribution cpu_distribution(std::span<const float> logits,
                                    const GpuSamplingOptions &sampling) {
  std::vector<std::uint32_t> ids(logits.size());
  std::iota(ids.begin(), ids.end(), 0);
  const auto count = sampling.temperature > 0 ? sampling.top_k : 1;
  std::partial_sort(
      ids.begin(), ids.begin() + count, ids.end(), [&](auto a, auto b) {
        return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
      });
  ids.resize(count);
  std::vector<float> scores;
  for (auto id : ids)
    scores.push_back(logits[id]);
  return target_distribution_from_topk(ids, scores, vocabulary,
                                       sampling.temperature, sampling.top_p);
}

void one_case(TargetGpuVerifier &verifier, MetalBuffer &buffer, std::uint32_t k,
              std::uint32_t accepted, bool ties, GpuSamplingOptions sampling,
              std::uint64_t position) {
  std::vector<std::uint16_t> data(8 * vocabulary);
  std::vector<float> represented(8 * vocabulary);
  std::vector<SelectorStep> proposals;
  for (std::uint32_t row = 0; row < 8; ++row) {
    const auto selected = std::uint32_t(30 + row);
    const auto alternative = std::uint32_t(110 + row);
    for (std::uint32_t token = 0; token < vocabulary; ++token) {
      const float score =
          ties ? 0.0F : float(int((token * 17 + row * 11) % 47) - 23) / 16.0F;
      data[std::uint64_t(row) * vocabulary + token] = float_to_bf16(score);
    }
    if (!ties) {
      data[std::uint64_t(row) * vocabulary +
           (row < accepted ? selected : alternative)] = float_to_bf16(10.0F);
      data[std::uint64_t(row) * vocabulary +
           (row < accepted ? alternative : selected)] = float_to_bf16(-10.0F);
    }
    if (row < k) {
      SelectorStep step;
      step.token = std::int32_t(selected);
      step.candidate_ids = {std::int32_t(selected), std::int32_t(alternative)};
      step.probabilities = {0.05F, 0.95F};
      proposals.push_back(std::move(step));
    }
  }
  for (std::size_t i = 0; i < data.size(); ++i)
    represented[i] = std::bit_cast<float>(std::uint32_t(data[i]) << 16);
  buffer.write(std::as_bytes(std::span(data)));
  std::vector<TargetDistribution> target;
  for (std::uint32_t row = 0; row <= k; ++row)
    target.push_back(cpu_distribution(
        std::span(represented)
            .subspan(std::uint64_t(row) * vocabulary, vocabulary),
        sampling));
  const auto expected = verify_speculative(
      proposals, target, sampling.temperature, sampling.seed, position);
  const auto direct = verifier.select(proposals, sampling, position, false);
  const auto icb = verifier.select(proposals, sampling, position, true);
  for (const auto *actual : {&direct, &icb}) {
    require(actual->verification.licensed_outputs == expected.licensed_outputs,
            "P04 licensed token differs from CPU oracle");
    require(actual->verification.accepted_proposals ==
                    expected.accepted_proposals &&
                actual->verification.used_correction ==
                    expected.used_correction,
            "P04 accepted prefix/correction differs from CPU oracle");
    require(actual->host_read_bytes < std::uint64_t(k + 1) * vocabulary * 2,
            "P04 read whole target logits");
  }
}

void state_case(const std::filesystem::path &fixtures,
                const std::filesystem::path &libraries,
                std::uint32_t forced_accept) {
  const auto root = fixtures / "affine";
  auto config = load_qwen_model_config(root / "source/config.json");
  auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
  auto context = MetalContext::create();
  auto teacher_context = MetalContext::create();
  auto weights = materialize_model(
      artifact, plan_model_materialization(artifact, context), context);
  auto teacher_weights = materialize_model(
      artifact, plan_model_materialization(artifact, teacher_context),
      teacher_context);
  auto layers = build_qwen_layer_catalog(artifact, config, 2048, 64);
  const std::array<std::uint32_t, 2> taps{0, 1};
  QwenForward target(context, weights, layers, config, 2048, libraries, taps,
                     false, true);
  QwenForward teacher(teacher_context, teacher_weights, layers, config, 2048,
                      libraries, taps, false, true);
  TargetGpuVerifier verifier(context, target.verification_logits_buffer(),
                             vocabulary, libraries);
  std::vector<std::uint32_t> prompt(33);
  for (std::uint32_t i = 0; i < prompt.size(); ++i)
    prompt[i] = (71 * i + 256) % vocabulary;
  static_cast<void>(teacher.run(prompt));
  std::array<std::uint32_t, 8> input{};
  input[0] = 7;
  for (std::uint32_t r = 0; r < 7; ++r) {
    const auto top1 =
        teacher.run(std::span(&input[r], 1), {.temperature = 0.0F}).token;
    input[r + 1] =
        r < forced_accept ? top1 : (top1 + 1 + r - forced_accept) % vocabulary;
  }
  teacher.reset();
  static_cast<void>(target.run(prompt));
  static_cast<void>(teacher.run(prompt));
  const auto state_before = target.inspect_recurrent_state();
  const GpuSamplingOptions sampling{
      .temperature = 0.7F, .top_k = 1, .top_p = 1.0F, .seed = 123};
  static_cast<void>(target.verify(input, sampling));
  require(target.position() == prompt.size() &&
              state_before == target.inspect_recurrent_state(),
          "P04 target verification modified committed state");
  const auto logits = target.read_logits(8);
  std::vector<TargetDistribution> distributions;
  std::vector<SelectorStep> proposals;
  for (std::uint32_t row = 0; row < input.size(); ++row) {
    distributions.push_back(cpu_distribution(
        std::span(logits).subspan(std::uint64_t(row) * vocabulary, vocabulary),
        sampling));
    if (row + 1 == input.size())
      continue;
    SelectorStep step;
    step.token = std::int32_t(input[row + 1]);
    step.candidate_ids = {step.token,
                          std::int32_t((input[row + 1] + 1) % vocabulary)};
    step.probabilities = row < forced_accept ? std::vector<float>{0.0F, 1.0F}
                                             : std::vector<float>{1.0F, 0.0F};
    proposals.push_back(std::move(step));
  }
  if (forced_accept < 7)
    require(
        distributions[forced_accept].probabilities[input[forced_accept + 1]] ==
            0,
        "P04 fixture reject token unexpectedly in target top-1");
  const auto expected =
      verify_speculative(proposals, distributions, sampling.temperature,
                         sampling.seed, target.position());
  const auto selected = verifier.select(proposals, sampling, target.position());
  require(selected.verification.licensed_outputs == expected.licensed_outputs &&
              selected.verification.accepted_proposals ==
                  expected.accepted_proposals &&
              selected.verification.used_correction == expected.used_correction,
          "P04 full target GPU/CPU verification differs");
  require(expected.accepted_proposals == forced_accept,
          "P04 forced acceptance count differs");
  const auto count = std::uint32_t(expected.licensed_outputs.size());
  static_cast<void>(target.commit_prefix(count));
  static_cast<void>(teacher.run(std::span(input).first(count)));
  require(target.position() == teacher.position() &&
              target.inspect_recurrent_state() ==
                  teacher.inspect_recurrent_state(),
          "P04 ReplaySSM state differs from committed prefix teacher");
  const std::array<std::uint32_t, 1> continuation{37};
  const auto actual = target.run(continuation, {}, true);
  const auto reference = teacher.run(continuation, {}, true);
  require(actual.token < vocabulary && reference.token < vocabulary &&
              actual.logits.size() == reference.logits.size(),
          "P04 continuation token or logits shape");
  float maximum = 0.0F;
  for (std::size_t i = 0; i < actual.logits.size(); ++i) {
    require(std::isfinite(actual.logits[i]) &&
                std::isfinite(reference.logits[i]),
            "P04 continuation logits contain non-finite value");
    maximum =
        std::max(maximum, std::abs(actual.logits[i] - reference.logits[i]));
  }
  require(std::isfinite(maximum) && maximum <= 0.08F,
          "P04 continuation differs from teacher");
}

void random_boundary_case(TargetGpuVerifier &verifier, MetalBuffer &buffer) {
  std::vector<std::uint16_t> tied(8 * vocabulary, float_to_bf16(0.0F));
  buffer.write(std::as_bytes(std::span(tied)));
  const std::array<std::uint32_t, 8> ids{0, 1, 2, 3, 4, 5, 6, 7};
  const std::array<float, 8> scores{};
  const auto distribution =
      target_distribution_from_topk(ids, scores, vocabulary, 0.7F, 1.0F);
  std::uint64_t bonus_seed = 0, accept_seed = 0;
  for (std::uint64_t seed = 1; seed < 200000 && (!bonus_seed || !accept_seed);
       ++seed) {
    if (!bonus_seed && std::abs(speculative_uniform(seed, 33, 0xB0A05EEDULL) -
                                0.125) < 0.00005)
      bonus_seed = seed;
    if (!accept_seed &&
        std::abs(speculative_uniform(seed, 33, 0xACCE9701ULL) - 0.25) < 0.00005)
      accept_seed = seed;
  }
  require(bonus_seed && accept_seed,
          "P04 could not construct random sampling boundaries");
  const GpuSamplingOptions bonus{
      .temperature = 0.7F, .top_k = 8, .top_p = 1.0F, .seed = bonus_seed};
  auto selected_bonus = verifier.select({}, bonus, 33);
  const auto expected_bonus =
      verify_speculative({}, std::span(&distribution, 1), 0.7F, bonus_seed, 33);
  require(selected_bonus.compact_cpu_fallback &&
              selected_bonus.verification.licensed_outputs ==
                  expected_bonus.licensed_outputs &&
              selected_bonus.host_read_bytes < 8 * vocabulary * 2,
          "P04 bonus boundary did not use compact exact fallback");
  const GpuSamplingOptions acceptance{
      .temperature = 0.7F, .top_k = 8, .top_p = 1.0F, .seed = accept_seed};
  const SelectorStep step{
      .token = 0, .candidate_ids = {0, 1}, .probabilities = {0.5F, 0.5F}};
  const std::array target{distribution, distribution};
  auto selected_accept = verifier.select(std::span(&step, 1), acceptance, 33);
  const auto expected_accept =
      verify_speculative(std::span(&step, 1), target, 0.7F, accept_seed, 33);
  require(selected_accept.compact_cpu_fallback &&
              selected_accept.verification.licensed_outputs ==
                  expected_accept.licensed_outputs &&
              selected_accept.verification.accepted_proposals ==
                  expected_accept.accepted_proposals &&
              selected_accept.host_read_bytes < 2 * 8 * vocabulary * 2,
          "P04 acceptance boundary did not use compact exact fallback");
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument(
          "p04_target_verify_tests TARGET_FIXTURES LIBRARIES");
    auto context = MetalContext::create();
    auto logits = context.make_buffer(8ULL * vocabulary * 2);
    TargetGpuVerifier verifier(context, logits, vocabulary, argv[2]);
    const auto before = context.registry_stats();
    std::uint32_t cases = 0;
    for (std::uint32_t k = 0; k <= 7; ++k)
      for (std::uint32_t accepted : {0U, k / 2, k})
        for (const float temperature : {0.0F, 0.7F})
          for (const float top_p : {1.0F, 0.35F}) {
            const GpuSamplingOptions sampling{.temperature = temperature,
                                              .top_k = 8,
                                              .top_p = top_p,
                                              .seed = 123 + k * 17 + accepted};
            one_case(verifier, logits, k, accepted, false, sampling, 33 + k);
            ++cases;
          }
    for (auto seed : {1ULL, 2ULL, 123ULL, 123456ULL}) {
      const GpuSamplingOptions sampling{
          .temperature = 0.7F, .top_k = 64, .top_p = 0.9F, .seed = seed};
      one_case(verifier, logits, 7, 7, true, sampling, 1024);
      ++cases;
    }
    random_boundary_case(verifier, logits);
    const auto after = context.registry_stats();
    require(before.current_buffer_count == after.current_buffer_count &&
                before.pipelines_compiled == after.pipelines_compiled &&
                before.command_plans_built == after.command_plans_built,
            "P04 selector allocated/compiled during hot calls");
    for (auto accepted : {0U, 3U, 7U})
      state_case(argv[1], argv[2], accepted);

    // Regression: GPU & CPU both reject (q_token == 0, p_token == 0) and handle
    // zero-mass residual without throwing.
    {
      std::vector<std::uint16_t> custom(8 * vocabulary, float_to_bf16(-20.0F));
      custom[0] = float_to_bf16(5.0F);
      custom[1] = float_to_bf16(5.0F);
      custom[vocabulary + 2] = float_to_bf16(5.0F);
      logits.write(std::as_bytes(std::span(custom)));
      const GpuSamplingOptions opts{
          .temperature = 1.0F, .top_k = 2, .top_p = 1.0F, .seed = 77};
      const SelectorStep impossible_step{
          .token = 10, .candidate_ids = {0, 1}, .probabilities = {0.5F, 0.5F}};
      std::vector<float> rep0(vocabulary), rep1(vocabulary);
      for (std::uint32_t t = 0; t < vocabulary; ++t) {
        rep0[t] = bf16_to_float(custom[t]);
        rep1[t] = bf16_to_float(custom[vocabulary + t]);
      }
      const std::array<TargetDistribution, 2> target_dist{
          cpu_distribution(rep0, opts), cpu_distribution(rep1, opts)};
      const auto cpu_res = verify_speculative(
          std::span(&impossible_step, 1), target_dist, opts.temperature,
          opts.seed, 10);
      const auto gpu_res =
          verifier.select(std::span(&impossible_step, 1), opts, 10, false);
      require(cpu_res.accepted_proposals == 0 && cpu_res.used_correction &&
                  gpu_res.verification.accepted_proposals == 0 &&
                  gpu_res.verification.used_correction &&
                  gpu_res.verification.licensed_outputs ==
                      cpu_res.licensed_outputs,
              "P04 q_token==0, p_token==0 / zero-mass residual GPU/CPU mismatch");
    }

    std::cout << "P04 GPU/CPU verification cases: " << cases << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P04 verify failure: " << error.what() << '\n';
    return 1;
  }
}
