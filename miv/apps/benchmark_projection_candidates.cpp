#include "../src/forward_plan.h"

#include <algorithm>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/target_verify_gpu.h"

namespace {
using namespace omlx::m5;
using namespace omlx::m5::detail::forward;

float represented(const std::vector<std::byte> &bytes, std::uint64_t index) {
  std::uint16_t bits;
  std::memcpy(&bits, bytes.data() + 2 * index, 2);
  return std::bit_cast<float>(std::uint32_t(bits) << 16);
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 6 || argc > 8)
      throw std::invalid_argument(
          "benchmark_projection_candidates ARTIFACT "
          "OBJECT LIBRARIES T REPEATS [matrix] [verify]");
    const U32 t = U32(std::stoul(argv[4])), repeats = U32(std::stoul(argv[5]));
    const bool matrix = argc >= 7 && std::string_view(argv[6]) == "matrix";
    const bool verify = argc == 8 && std::string_view(argv[7]) == "verify";
    if ((argc >= 7 && !matrix) || (argc == 8 && !verify) || (verify && t != 8))
      throw std::invalid_argument("unknown projection candidate");
    if ((t != 1 && t != 8) || repeats < 5)
      throw std::invalid_argument(
          "projection candidate requires T=1/8 and 5+ repeats");
    // B0 already verifies the full payload. This slice probe reads one trusted
    // object; hashing the entire 14–20 GB artifact per geometry would pollute
    // installation time without changing the isolated GPU timing.
    auto artifact = ArtifactModel::open(argv[1], false);
    const auto &object = artifact.object(argv[2]);
    if (object.shape.size() != 2 || object.format.starts_with("direct_"))
      throw std::invalid_argument(
          "projection candidate requires quantized rank-2 weight");
    auto context = MetalContext::create();
    MaterializedWeightView weight{.object_id = object.id,
                                  .component =
                                      classify_model_component(object.name),
                                  .name = object.name,
                                  .format = object.format,
                                  .layout = object.layout,
                                  .parent_shape = object.shape,
                                  .padded_k = object.padded_k,
                                  .row_count = object.shape[0],
                                  .in_features = object.shape[1]};
    std::vector<std::unique_ptr<MetalBuffer>> weight_buffers;
    for (const auto &plane : object.planes) {
      weight_buffers.push_back(std::make_unique<MetalBuffer>(
          context.upload(artifact.plane_bytes(object, plane))));
      weight.planes.push_back({.kind = plane.kind,
                               .buffer = weight_buffers.back().get(),
                               .segment_index = U32(weight_buffers.size() - 1),
                               .offset = 0,
                               .bytes = plane.bytes,
                               .row_stride = plane.bytes / object.shape[0]});
    }
    const U32 n = U32(weight.row_count), k = U32(weight.in_features);
    std::vector<std::uint16_t> input(U64(t) * k);
    for (U32 row = 0; row < t; ++row)
      for (U32 col = 0; col < k; ++col) {
        float value =
            float(std::int32_t((col * 17 + row * 31) % 257) - 128) / 512.0f;
        input[U64(row) * k + col] = float_to_bf16(value);
      }
    auto x = context.upload(std::as_bytes(std::span(input)));
    auto old_y = context.make_buffer(U64(t) * n * 2);
    auto new_y = context.make_buffer(U64(t) * n * 2);
    auto sums = context.make_buffer(U64(t) * k * 4);
    auto zero = context.make_buffer(256);
    zero.write(std::vector<std::byte>(256));
    const std::filesystem::path libraries = argv[3];
    auto build = [&](bool candidate, const MetalBuffer &y) {
      Encoder e(context, libraries, t,
                candidate ? "projection.candidate" : "projection.baseline");
      const auto route = select_linear_route(
          weight.format, n, k, t, candidate && t == 1 && !matrix,
          candidate && t == 8 && !matrix, candidate && matrix);
      linear(e, weight, {&x, 0, x.size()}, {&y, 0, y.size()}, n, k,
             {&sums, 0, sums.size()}, {&zero, 0, zero.size()}, route);
      return e.finish();
    };
    auto old_plan = build(false, old_y), new_plan = build(true, new_y);
    std::unique_ptr<TargetGpuVerifier> verifier;
    if (verify)
      verifier =
          std::make_unique<TargetGpuVerifier>(context, new_y, n, libraries);
    const auto before = context.registry_stats();
    const auto run = [&](Profile &p) {
      const std::array call = {
          MetalCommandInvocation{.plan = &p.plan,
                                 .bindings = p.bindings,
                                 .parameters = &p.parameters}};
      return context.execute_batch(call);
    };
    std::vector<double> old_gpu, new_gpu, old_wall, new_wall;
    for (U32 i = 0; i < repeats + 3; ++i) {
      const auto a = run(old_plan), b = run(new_plan);
      if (i >= 3) {
        old_gpu.push_back(a.timing.gpu_ms);
        new_gpu.push_back(b.timing.gpu_ms);
        old_wall.push_back(a.timing.wall_ms);
        new_wall.push_back(b.timing.wall_ms);
      }
    }
    const auto baseline = old_y.read(), candidate = new_y.read();
    double error2 = 0, scale2 = 0, max_abs = 0;
    for (U64 i = 0; i < U64(t) * n; ++i) {
      const auto a = represented(baseline, i), b = represented(candidate, i);
      if (!std::isfinite(a) || !std::isfinite(b))
        throw std::runtime_error("nonfinite candidate");
      double diff = double(a) - b;
      error2 += diff * diff;
      scale2 += double(a) * a;
      max_abs = std::max(max_abs, std::abs(diff));
    }
    const double relative = std::sqrt(error2 / std::max(scale2, 1e-12));
    auto best = [&](const std::vector<std::byte> &buffer, U32 row) {
      U32 selected = 0;
      float maximum = -INFINITY;
      for (U32 i = 0; i < n; ++i) {
        float v = represented(buffer, U64(row) * n + i);
        if (v > maximum) {
          maximum = v;
          selected = i;
        }
      }
      return selected;
    };
    double pattern_relative_l2 = relative, pattern_max_abs = max_abs;
    U32 matching_top1 = best(baseline, t - 1) == best(candidate, t - 1) ? 1 : 0;
    std::optional<MetalIcbProfile> icb;
    if (context.capabilities().profile.compute_icb_eligible) {
      icb.emplace(context.prepare_icb_profile(
          new_plan.plan.key(), {.plan = &new_plan.plan,
                                .bindings = new_plan.bindings,
                                .parameters = &new_plan.parameters}));
      static_cast<void>(context.replay_icb(*icb));
      if (new_y.read() != candidate)
        throw std::runtime_error("candidate ICB/direct output mismatch");
    }
    for (U32 seed = 1; seed < 4; ++seed) {
      for (U32 row = 0; row < t; ++row)
        for (U32 col = 0; col < k; ++col) {
          const float value =
              float(std::int32_t((col * 17 + row * 31 + seed * 43) % 257) -
                    128) /
              512.0f;
          input[U64(row) * k + col] = float_to_bf16(value);
        }
      x.write(std::as_bytes(std::span(input)));
      static_cast<void>(run(old_plan));
      static_cast<void>(run(new_plan));
      const auto old_bytes = old_y.read(), new_bytes = new_y.read();
      double sum = 0.0, scale = 0.0;
      for (U64 i = 0; i < U64(t) * n; ++i) {
        const double lhs = represented(old_bytes, i);
        const double rhs = represented(new_bytes, i);
        const double delta = lhs - rhs;
        sum += delta * delta;
        scale += lhs * lhs;
        pattern_max_abs = std::max(pattern_max_abs, std::abs(delta));
      }
      pattern_relative_l2 = std::max(pattern_relative_l2,
                                     std::sqrt(sum / std::max(scale, 1e-12)));
      matching_top1 += best(old_bytes, t - 1) == best(new_bytes, t - 1);
      if (icb) {
        static_cast<void>(context.replay_icb(*icb));
        if (new_y.read() != new_bytes)
          throw std::runtime_error("candidate ICB/direct output mismatch");
      }
    }
    double tail_relative_l2 = 0.0;
    U32 matching_tail_top1 = 0;
    if (matrix && t == 8) {
      for (U32 live_rows : {5U, 6U, 7U}) {
        for (auto *p : {&old_plan, &new_plan}) {
          for (const auto &patch : p->patches) {
            if (patch.position)
              continue;
            const U32 value = live_rows * patch.multiplier;
            std::memcpy(p->parameter_data.data() + patch.offset +
                            patch.word * sizeof(U32),
                        &value, sizeof(value));
          }
          p->parameters.write("step", p->parameter_data);
        }
        static_cast<void>(run(old_plan));
        static_cast<void>(run(new_plan));
        const auto old_bytes = old_y.read(), new_bytes = new_y.read();
        double squared_error = 0.0, squared_scale = 0.0;
        for (U64 i = 0; i < U64(live_rows) * n; ++i) {
          const double lhs = represented(old_bytes, i);
          const double rhs = represented(new_bytes, i);
          squared_error += (lhs - rhs) * (lhs - rhs);
          squared_scale += lhs * lhs;
        }
        tail_relative_l2 =
            std::max(tail_relative_l2,
                     std::sqrt(squared_error / std::max(squared_scale, 1e-12)));
        matching_tail_top1 +=
            best(old_bytes, live_rows - 1) == best(new_bytes, live_rows - 1);
        if (icb) {
          static_cast<void>(context.replay_icb(*icb));
          const auto replay_bytes = new_y.read();
          const auto used_bytes = U64(live_rows) * n * 2;
          if (!std::equal(replay_bytes.begin(),
                          replay_bytes.begin() + used_bytes, new_bytes.begin()))
            throw std::runtime_error("matrix tail ICB/direct mismatch");
        }
      }
    }
    nlohmann::json verify_ab = nullptr;
    if (verify) {
      // The tail probe changes dynamic row limits; restore the real W=8 head.
      for (const auto &patch : new_plan.patches) {
        if (patch.position)
          continue;
        const U32 value = t * patch.multiplier;
        std::memcpy(new_plan.parameter_data.data() + patch.offset +
                        patch.word * sizeof(U32),
                    &value, sizeof(value));
      }
      new_plan.parameters.write("step", new_plan.parameter_data);
      static_cast<void>(run(new_plan));
      const auto represented_logits = new_y.read();
      std::vector<SelectorStep> proposals;
      for (U32 row = 0; row + 1 < t; ++row) {
        const auto token = best(represented_logits, row);
        proposals.push_back({.token = std::int32_t(token),
                             .candidate_ids = {std::int32_t(token),
                                               std::int32_t((token + 1) % n)},
                             .probabilities = {0.01F, 0.99F}});
      }
      const auto cpu_verify = [&](U64 seed) {
        auto logits_bytes = new_y.read();
        std::vector<TargetDistribution> distributions;
        for (U32 row = 0; row < t; ++row) {
          std::vector<U32> ids(n);
          std::iota(ids.begin(), ids.end(), 0);
          const auto value = [&](U32 token) {
            return represented(logits_bytes, U64(row) * n + token);
          };
          std::partial_sort(
              ids.begin(), ids.begin() + 16, ids.end(), [&](U32 a, U32 b) {
                return value(a) > value(b) || (value(a) == value(b) && a < b);
              });
          ids.resize(16);
          std::vector<float> scores;
          for (auto id : ids)
            scores.push_back(value(id));
          distributions.push_back(
              target_distribution_from_topk(ids, scores, n, 0.7F, 0.9F));
        }
        return verify_speculative(proposals, distributions, 0.7F, seed, 33);
      };
      std::vector<double> cpu_verify_wall, gpu_verify_wall, gpu_verify_time;
      U64 gpu_read_bytes = 0;
      U32 fallback_rounds = 0;
      for (U32 iteration = 0; iteration < repeats + 3; ++iteration) {
        const U64 seed = 123 + iteration;
        const GpuSamplingOptions sampling{
            .temperature = 0.7F, .top_k = 16, .top_p = 0.9F, .seed = seed};
        const auto cpu = [&] {
          const auto start = std::chrono::steady_clock::now();
          auto outcome = cpu_verify(seed);
          const auto elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
          return std::pair{std::move(outcome), elapsed};
        };
        const auto gpu = [&] {
          const auto start = std::chrono::steady_clock::now();
          auto outcome = verifier->select(proposals, sampling, 33);
          const auto elapsed = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
          return std::pair{std::move(outcome), elapsed};
        };
        VerificationResult old_result;
        TargetGpuVerifyDecision new_result;
        double old_elapsed = 0, new_elapsed = 0;
        if (iteration % 2 == 0) {
          std::tie(old_result, old_elapsed) = cpu();
          std::tie(new_result, new_elapsed) = gpu();
        } else {
          std::tie(new_result, new_elapsed) = gpu();
          std::tie(old_result, old_elapsed) = cpu();
        }
        if (old_result.licensed_outputs !=
                new_result.verification.licensed_outputs ||
            old_result.accepted_proposals !=
                new_result.verification.accepted_proposals ||
            old_result.used_correction !=
                new_result.verification.used_correction)
          throw std::runtime_error("P04 real head CPU/GPU verify mismatch");
        if (iteration >= 3) {
          cpu_verify_wall.push_back(old_elapsed);
          gpu_verify_wall.push_back(new_elapsed);
          gpu_verify_time.push_back(new_result.submission.timing.gpu_ms);
        }
        gpu_read_bytes = new_result.host_read_bytes;
        fallback_rounds += new_result.compact_cpu_fallback;
      }
      verify_ab = {
          {"scope", "real full-N shared head logits, synthetic hidden; no "
                    "target forward"},
          {"sampling", {{"temperature", 0.7}, {"top_k", 16}, {"top_p", 0.9}}},
          {"cpu_wall_ms", cpu_verify_wall},
          {"gpu_wall_ms", gpu_verify_wall},
          {"gpu_kernel_ms", gpu_verify_time},
          {"cpu_read_bytes", U64(t) * n * 2},
          {"gpu_read_bytes", gpu_read_bytes},
          {"compact_fallbacks", fallback_rounds},
          {"outcomes_equal", true}};
    }
    const auto after = context.registry_stats();
    if (after.pipelines_compiled != before.pipelines_compiled ||
        after.command_plans_built != before.command_plans_built ||
        after.upload_count != before.upload_count)
      throw std::runtime_error("candidate created hot-loop resources");
    std::cout << nlohmann::json(
                     {{"schema", "omlx-m5-projection-candidate-v1"},
                      {"artifact", argv[1]},
                      {"object", object.name},
                      {"format", object.format},
                      {"n", n},
                      {"k", k},
                      {"t", t},
                      {"repeats", repeats},
                      {"candidate",
                       matrix ? "simdgroup_matrix_k32" : "lane_accum"},
                      {"old_gpu_ms", old_gpu},
                      {"new_gpu_ms", new_gpu},
                      {"old_wall_ms", old_wall},
                      {"new_wall_ms", new_wall},
                      {"max_abs", max_abs},
                      {"relative_l2", relative},
                      {"pattern_count", 4},
                      {"pattern_relative_l2_max", pattern_relative_l2},
                      {"pattern_max_abs", pattern_max_abs},
                      {"matching_top1_patterns", matching_top1},
                      {"tail_relative_l2_max", tail_relative_l2},
                      {"matching_tail_top1", matching_tail_top1},
                      {"last_row_top1_baseline", best(baseline, t - 1)},
                      {"last_row_top1_candidate", best(candidate, t - 1)},
                      {"old_dispatches", old_plan.plan.dispatch_count()},
                      {"new_dispatches", new_plan.plan.dispatch_count()},
                      {"target_verify_ab", verify_ab}})
                     .dump(2)
              << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
