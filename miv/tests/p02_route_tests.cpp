#include "../src/forward_plan.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/qwen_forward.h"

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

namespace {
void require(bool condition, const std::string &message) {
  if (!condition)
    throw std::runtime_error(message);
}

template <class Callback>
void rejected(Callback &&callback, const std::string &message) {
  try {
    callback();
  } catch (const std::invalid_argument &) {
    return;
  }
  throw std::runtime_error(message);
}

const QwenProfileWeightRoute &find_route(const QwenProfileDescription &profile,
                                         const std::string &object) {
  const auto found = std::find_if(profile.weights.begin(), profile.weights.end(),
                                  [&](const auto &route) {
                                    return route.object == object;
                                  });
  if (found == profile.weights.end())
    throw std::runtime_error("P02 profile weight absent: " + object);
  return *found;
}

Json compare(std::span<const float> actual, std::span<const float> reference) {
  require(actual.size() == reference.size(), "P02 vector shape mismatch");
  double error = 0, scale = 0, maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]), "P02 nonfinite output");
    const double delta = double(actual[i]) - reference[i];
    maximum = std::max(maximum, std::abs(delta));
    error += delta * delta;
    scale += double(reference[i]) * reference[i];
  }
  const double relative = std::sqrt(error / std::max(scale, 1e-12));
  return {{"max_abs", maximum}, {"relative_l2", relative},
          {"passed", maximum <= 0.08 && relative <= 0.02}};
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument("p02_route_tests QWEN_FIXTURE LIBRARIES");
    MaterializedWeightView real_shape;
    real_shape.name = "language_model.lm_head";
    real_shape.format = "oq_affine_q4_g64_bf16_v1";
    const auto promoted = resolve_linear_route(
        real_shape, 248320, 5120, 8, "Apple M4", true);
    const auto not_measured = resolve_linear_route(
        real_shape, 248320, 5120, 4, "Apple M4", true);
    const auto future_device = resolve_linear_route(
        real_shape, 248320, 5120, 8, "Apple M5 Ultra", true);
    require(promoted.route.matrix && promoted.source == "m4_qualified_shape" &&
                !not_measured.route.matrix &&
                not_measured.fallback_reason == "shape_or_width_unqualified" &&
                !future_device.route.matrix &&
                future_device.fallback_reason == "device_unqualified",
            "P02 device/width promotion boundary");
    const QwenLinearRouteOverride real_override{
        real_shape.name, real_shape.format, "matrix", 248320, 5120, 8};
    rejected([&] {
      (void)resolve_linear_route(real_shape, 248320, 5120, 8,
                                 "Apple M3", true, real_override);
    }, "P02 override escaped Apple9+ name gate");
    rejected([&] {
      (void)resolve_linear_route(real_shape, 248320, 5120, 8,
                                 "Apple M3", true, real_override,
                                 LinearNumericPolicy::Legacy, 8U);
    }, "P02 override escaped Apple9+ family gate");
    const auto m5_override = resolve_linear_route(
        real_shape, 248320, 5120, 8, "Apple M5 Ultra", false, real_override,
        LinearNumericPolicy::Legacy, 10U);
    require(m5_override.route.matrix &&
                m5_override.source == "diagnostic_exact_override" &&
                m5_override.override_matched,
            "P02 M5 Ultra diagnostic override rejected");

    const auto root = std::filesystem::path(argv[1]) / "affine";
    const auto config = load_qwen_model_config(root / "source/config.json");
    const auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
    auto context = MetalContext::create();
    if (context.capabilities().profile.probe.apple_gpu_family < 9) {
      std::cout << Json({{"schema", "omlx-m5-p02-route-ab-v1"},
                         {"scope", "Apple9+ end-to-end A/B skipped"},
                         {"device", context.capabilities().device_name}})
                        .dump(2) << '\n';
      return 0;
    }
    const auto weights = materialize_model(
        artifact, plan_model_materialization(artifact, context), context);
    auto candidate_context = MetalContext::create();
    const auto candidate_weights = materialize_model(
        artifact, plan_model_materialization(artifact, candidate_context),
        candidate_context);
    const auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
    const std::array<U32, 2> taps{0, 1};
    const std::string head = "language_model.lm_head";
    const QwenLinearRouteOverride override{head, "oq_affine_q4_g64_bf16_v1",
                                            "matrix", 257, 128, 8};
    QwenForward baseline(context, weights, layers, config, 1024, argv[2], taps,
                         false, false, false);
    QwenForward candidate(candidate_context, candidate_weights, layers, config,
                          1024, argv[2], taps, false, false, false, override);
    const auto old_profile = baseline.inspect_profile(8);
    const auto new_profile = candidate.inspect_profile(8);
    require(old_profile.dispatch_count == new_profile.dispatch_count + 1,
            "P02 matrix route did not remove group-sum dispatch");
    require(old_profile.weights.size() == new_profile.weights.size(),
            "P02 profile weight count differs");
    std::size_t changed = 0;
    for (std::size_t i = 0; i < old_profile.weights.size(); ++i) {
      const auto &old = old_profile.weights[i], &next = new_profile.weights[i];
      require(old.object == next.object && old.n == next.n && old.k == next.k &&
                  old.capacity == next.capacity,
              "P02 A/B profile key differs");
      changed += old.function != next.function;
      if (old.object != head)
        require(old.function == next.function && old.grid.width == next.grid.width,
                "P02 override changed another weight");
    }
    require(changed == 1, "P02 override changed more than one kernel");
    const auto &selected = find_route(new_profile, head);
    require(selected.library == "quantized_matrix" &&
                selected.function == "omlx_m5_matrix_q4_g64" &&
                selected.family == "simdgroup_matrix" &&
                selected.source == "diagnostic_exact_override" &&
                selected.grid.width == 1152 && selected.grid.height == 1 &&
                selected.threads_per_threadgroup.width == 128,
            "P02 compiled symbol/grid differs from chosen route");
    require(find_route(candidate.inspect_profile(7), head).function ==
                selected.function &&
                find_route(candidate.inspect_profile(1), head).function ==
                    find_route(baseline.inspect_profile(1), head).function,
            "P02 width envelope or unmeasured fallback changed");
    const auto metadata = resolve_linear_route(
        weights.parent_view(head), 257, 128, 8,
        context.capabilities().device_name, false, override,
        LinearNumericPolicy::Legacy,
        context.capabilities().profile.probe.apple_gpu_family);
    const auto predicted_grid = linear_grid(metadata.route, override.format,
                                            override.n, override.capacity);
    require(metadata.route.function == selected.function &&
                predicted_grid.width == selected.grid.width &&
                predicted_grid.height == selected.grid.height,
            "P02 metadata and compiled profile disagree");
    rejected([&] {
      const QwenLinearRouteOverride invalid{head, override.format, "matrix",
                                             258, 128, 8};
      auto invalid_context = MetalContext::create();
      const auto invalid_weights = materialize_model(
          artifact, plan_model_materialization(artifact, invalid_context),
          invalid_context);
      QwenForward unused(invalid_context, invalid_weights, layers, config, 1024,
                         argv[2], taps,
                         false, false, false, invalid);
    }, "P02 unmatched override accepted");

    std::ifstream input(root / "oracle.json");
    Json oracle;
    input >> oracle;
    std::vector<U32> tokens;
    for (const auto &item : oracle.at("cases")) {
      const auto proposed = item.at("tokens").get<std::vector<U32>>();
      if (proposed.size() == 8)
        tokens = proposed;
    }
    require(tokens.size() == 8, "P02 T8 fixture absent");
    const auto old_round = baseline.run(tokens, {}, true, false);
    const auto old_state = baseline.inspect_state();
    const auto new_round = candidate.run(tokens, {}, true, false);
    const auto new_state = candidate.inspect_state();
    const auto hidden = compare(new_round.hidden, old_round.hidden);
    const auto logits = compare(new_round.logits, old_round.logits);
    require(hidden["passed"].get<bool>() && logits["passed"].get<bool>() &&
                old_round.token == new_round.token && old_state == new_state &&
                old_round.submission.dispatch_count == old_profile.dispatch_count &&
                new_round.submission.dispatch_count == new_profile.dispatch_count,
            "P02 A/B output, state or executed dispatch mismatch");
    candidate.reset();
    const auto new_icb = candidate.run(tokens, {}, true, true);
    require(new_icb.hidden == new_round.hidden &&
                new_icb.logits == new_round.logits &&
                candidate.inspect_state() == new_state,
            "P02 candidate ICB/direct changed output or state");
    const std::array<U32, 1> next_token{11};
    const auto old_next = baseline.run(next_token, {}, true, false);
    const auto new_next = candidate.run(next_token, {}, true, false);
    const auto continuation = compare(new_next.logits, old_next.logits);
    require(continuation["passed"].get<bool>() &&
                baseline.inspect_state() == candidate.inspect_state(),
            "P02 continuation state or logits diverged");
    // The numeric policy is consumed by the real installer, not only a probe.
    QwenForward exact(context, weights, layers, config, 1024, argv[2], taps,
                      false, false, false, {}, false, false, {},
                      LinearNumericPolicy::RowExactPortableV1);
    const auto exact_profile = exact.inspect_profile(8);
    require(exact_profile.numeric_contract_id == "linear_row_exact_portable_v1" &&
                !exact_profile.whole_program_row_exact_qualified,
            "P02-TF confused Linear qualification with whole-Program exactness");
    for (const auto &r : exact_profile.weights) {
      const bool nax_source =
          r.source == "bf16_nax_m5_v1" || r.source == "mxfp4_nax_m5_v1" ||
          r.source == "affine_nax_m5_v1";
      require(r.row_exact_eligible && !r.row_exact_group.empty() &&
                  ((r.source == "row_exact_portable_v1" &&
                    r.grid.height == exact_profile.capacity) ||
                   (nax_source && r.grid.height >= 1U)),
              "P02-TF real installer/route metadata differs");
    }
    const auto exact_run = exact.run(tokens, {}, true, false);
    exact.reset();
    const auto exact_icb = exact.run(tokens, {}, true, true);
    require(exact_icb.logits == exact_run.logits && exact_icb.hidden == exact_run.hidden,
            "P02-TF ICB/direct changed installed row-exact Linear output");
    rejected([&] {
      (void)resolve_linear_route(
          weights.parent_view(head), 257, 128, 8,
          context.capabilities().device_name, true, override,
          LinearNumericPolicy::RowExactPortableV1,
          context.capabilities().profile.probe.apple_gpu_family);
    }, "P02-TF allowed a legacy matrix override into exact policy");
    std::cout << Json({{"schema", "omlx-m5-p02-route-ab-v1"},
                       {"scope", "two-layer control fixture; exact head T8 A/B, not target-model performance"},
                       {"device", context.capabilities().device_name},
                       {"object", head},
                       {"format", selected.format},
                       {"n", selected.n}, {"k", selected.k},
                       {"batch", 1}, {"width", selected.capacity},
                       {"baseline_kernel", find_route(old_profile, head).function},
                       {"candidate_kernel", selected.function},
                       {"candidate_grid", {selected.grid.width, selected.grid.height,
                                           selected.grid.depth}},
                       {"changed_weight_routes", changed},
                       {"baseline_dispatches", old_round.submission.dispatch_count},
                       {"candidate_dispatches", new_round.submission.dispatch_count},
                       {"hidden", hidden}, {"logits", logits},
                       {"continuation_logits", continuation},
                       {"state_bytes_equal", old_state.size()},
                       {"candidate_icb_direct_equal", true},
                       {"row_exact_installer_metadata_equal", true},
                       {"row_exact_icb_direct_equal", true},
                       {"row_exact_rejects_legacy_override", true},
                       {"m5_ultra_qualified", false}}).dump(2) << '\n';
  } catch (const std::exception &error) {
    std::cerr << "P02 route A/B failed: " << error.what() << '\n';
    return 1;
  }
}
