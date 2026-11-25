#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/qwen_forward.h"
#include "omlx_m5/quantization.h"

using namespace omlx::m5;
using Json = nlohmann::json;
namespace {
void require(bool value, const char *why) {
  if (!value) throw std::runtime_error(why);
}
template <class F> void rejected(F &&action, const char *why) {
  bool failed = false;
  try { action(); } catch (const std::invalid_argument &) { failed = true; }
  require(failed, why);
}
std::vector<std::uint32_t> tokens(std::uint32_t count) {
  std::vector<std::uint32_t> result(count);
  for (std::uint32_t i = 0; i < count; ++i)
    result[i] = (71 * i + 7) % 257;
  return result;
}
const QwenProfileWeightRoute &find_route(const QwenProfileDescription &profile,
                                         const std::string &name) {
  const auto found = std::find_if(profile.weights.begin(), profile.weights.end(),
                                  [&](const auto &row) { return row.object == name; });
  if (found == profile.weights.end())
    throw std::runtime_error("P12 profile route absent");
  return *found;
}
void exercise(const std::filesystem::path &fixture,
              const std::filesystem::path &libs,
              const std::filesystem::path &output) {
  const auto config = load_qwen_model_config(fixture / "source/config.json");
  auto artifact = ArtifactModel::open(fixture / "model.omlxm5", true);
  auto context = MetalContext::create();
  auto weights = materialize_model(
      artifact, plan_model_materialization(artifact, context), context);
  auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
  const std::string name =
      "language_model.model.layers.0.mlp.switch_mlp.gate_proj";
  const auto &object = artifact.object(name);
  const QwenMoeRouteOverride fallback{name, object.format, object.layout,
                                       "fallback", 128, 128, 8};
  const std::array<std::uint32_t, 2> taps{0, 1};
  QwenForward baseline(context, weights, layers, config, 1024, libs, taps,
                       true, true);
  QwenForward candidate(context, weights, layers, config, 1024, libs, taps,
                        true, true, false, {}, false, false, fallback);
  const auto baseline_route = find_route(baseline.inspect_profile(8), name);
  const auto candidate_route = find_route(candidate.inspect_profile(8), name);
  require(baseline_route.function.ends_with("_grouped") &&
              candidate_route.function.ends_with("_fallback") &&
              candidate_route.family == "fallback_a16" &&
              candidate_route.source == "p12_exact_ab" &&
              candidate_route.format == object.format &&
              candidate_route.n == 128 && candidate_route.k == 128 &&
              candidate_route.capacity == 8,
          "P12 exact production profile did not select fallback");
  for (std::uint32_t width : {1U, 8U}) {
    baseline.reset(); candidate.reset();
    const auto input = tokens(width);
    auto left = baseline.run(input, {}, true);
    auto right = candidate.run(input, {}, true);
    require(left.token == right.token && left.logits == right.logits &&
                left.hidden == right.hidden &&
                left.layer_routes == right.layer_routes &&
                left.router_scores == right.router_scores &&
                baseline.inspect_recurrent_state() ==
                    candidate.inspect_recurrent_state(),
            "P12 grouped/fallback model output or route changed");
  }
  const auto verify_input = tokens(8);
  const auto verified_left = baseline.verify(verify_input, {}, true);
  const auto verified_right = candidate.verify(verify_input, {}, true);
  require(verified_left.logits == verified_right.logits &&
              verified_left.layer_routes == verified_right.layer_routes,
          "P12 verify route differs between expert families");
  static_cast<void>(baseline.commit_prefix(3));
  static_cast<void>(candidate.commit_prefix(3));
  const std::array<std::uint32_t, 1> continuation{42};
  require(baseline.run(continuation, {}, true).logits ==
              candidate.run(continuation, {}, true).logits &&
              baseline.inspect_recurrent_state() ==
                  candidate.inspect_recurrent_state(),
          "P12 ReplaySSM or continuation changed");
  auto mismatch = fallback;
  mismatch.format += "-wrong";
  rejected([&] {
    QwenForward program(context, weights, layers, config, 1024, libs, taps,
                        true, true, false, {}, false, false, mismatch);
  }, "P12 mismatched format installed");
  mismatch = fallback; mismatch.capacity = 129;
  rejected([&] {
    QwenForward program(context, weights, layers, config, 1024, libs, taps,
                        true, true, false, {}, false, false, mismatch);
  }, "P12 wrong capacity installed");
  mismatch = fallback; mismatch.choice = "other";
  rejected([&] {
    QwenForward program(context, weights, layers, config, 1024, libs, taps,
                        true, true, false, {}, false, false, mismatch);
  }, "P12 unknown expert family installed");

  // The same exact key is reachable through the Native request entrance, but
  // never as a generic format-wide preference.
  LoadOptions options;
  options.model_dir = (fixture / "source").string();
  options.max_context = 1024;
  options.path_overrides["artifact"] = (fixture / "model.omlxm5").string();
  options.path_overrides["qwen_ar_metallib"] =
      (libs / "omlx_m5_qwen_ar.metallib").string();
  auto direct = Engine::create(options);
  options.path_overrides["moe_route_ab"] = name + "|" + object.format + "|" +
      object.layout + "|128|128|8|fallback";
  auto selected = Engine::create(options);
  require(selected->stats().route_table_id.find(":moe-ab-v1:" + name) !=
              std::string::npos,
          "P12 native route identity omitted exact MoE candidate");
  const auto generate = [&](Engine &engine) {
    SubmitRequest request;
    for (auto id : tokens(8)) request.prompt_tokens.push_back(int(id));
    request.max_new_tokens = 1;
    const auto request_id = engine.submit(request);
    std::vector<std::int32_t> output;
    bool done = false;
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(10);
    while (!done && std::chrono::steady_clock::now() < deadline)
      for (const auto &event : engine.poll_events(std::chrono::milliseconds(20))) {
        require(event.request_id == request_id, "P12 foreign Engine event");
        if (event.kind == EventKind::tokens)
          output.insert(output.end(), event.tokens.begin(), event.tokens.end());
        else {
          require(event.kind == EventKind::finished, "P12 Engine request failed");
          done = true;
        }
      }
    require(done && output.size() == 1, "P12 Engine candidate did not finish");
    return output;
  };
  require(generate(*direct) == generate(*selected),
          "P12 Native grouped/fallback changed output");
  selected->shutdown(); direct->shutdown();
  auto invalid = options;
  invalid.path_overrides["moe_route_ab"] += "|extra";
  rejected([&] { static_cast<void>(Engine::create(invalid)); },
           "P12 malformed Native route key accepted");
  invalid = options;
  invalid.path_overrides["moe_route_ab"] = name + "|bad_format|" +
      object.layout + "|128|128|8|fallback";
  rejected([&] { static_cast<void>(Engine::create(invalid)); },
           "P12 mismatched Native route installed");

  // This is a small-model router trace, not a claimed real 35B route sample.
  baseline.reset();
  bool stale_rejected = false;
  try { static_cast<void>(baseline.read_moe_route_trace(0)); }
  catch (const std::logic_error &) { stale_rejected = true; }
  require(stale_rejected, "P12 read stale trace after reset");
  static_cast<void>(baseline.run(tokens(128)));
  const auto route_snapshot = baseline.read_moe_route_trace(0);
  require(route_snapshot.tokens == 128 && route_snapshot.start_position == 0 &&
              route_snapshot.expert_ids.size() == 128U * 8 &&
              route_snapshot.scores_bf16.size() == route_snapshot.expert_ids.size() &&
              route_snapshot.readback_bytes == 128U * 2 * 8 * 6,
          "P12 route-only diagnostic readback shape incorrect");
  baseline.reset();
  const auto recorded = baseline.run(tokens(128), {}, true);
  for (std::uint32_t token = 0; token < 128; ++token)
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      const auto i = token * 8 + slot;
      const auto from_full = token * 2 * 8 + slot;
      require(route_snapshot.expert_ids[i] == recorded.layer_routes[from_full] &&
                  bf16_to_float(route_snapshot.scores_bf16[i]) ==
                      recorded.router_scores[from_full],
              "P12 route-only readback differs from full diagnostic");
    }
  Json ids = Json::array(), scores = Json::array(), words = Json::array();
  for (std::uint32_t token = 0; token < 128; ++token) {
    Json id_row = Json::array(), score_row = Json::array(), word_row = Json::array();
    float mass = 0;
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      const auto offset = token * 8 + slot;
      const auto id = route_snapshot.expert_ids[offset];
      const auto code = route_snapshot.scores_bf16[offset];
      const auto score = bf16_to_float(code);
      require(id < 256 && std::isfinite(score) && score >= 0 &&
                  bf16_to_float(code) == score,
              "P12 captured route/score invalid");
      id_row.push_back(id);
      score_row.push_back(score);
      word_row.push_back(code);
      mass += score;
    }
    require(std::abs(mass - 1.0f) < 0.02f,
            "P12 captured route score mass invalid");
    ids.push_back(std::move(id_row));
    scores.push_back(std::move(score_row));
    words.push_back(std::move(word_row));
  }
  Json report = {{"schema", "omlx-m5-p12-route-trace-v1"},
                 {"origin", "QwenForward small complete control fixture layer0"},
                 {"synthetic", true},
                 {"source_kind", "small_control_fixture"},
                 {"source_artifact", artifact.summary().source_repo + "@" +
                                         artifact.summary().source_revision},
                 {"source_object", name},
                 {"router_weight_object", "language_model.model.layers.0.mlp.gate"},
                 {"router_topk_kernel", "omlx_m5_forward_router_top8_logits"},
                 {"source_format", object.format},
                 {"source_layout", object.layout},
                 {"source_profile_function", baseline_route.function},
                 {"source_profile_family", baseline_route.family},
                 {"layer", 0}, {"tokens", 128},
                 {"start_position", route_snapshot.start_position},
                 {"route_readback_bytes", route_snapshot.readback_bytes},
                 {"experts", 256},
                 {"top_k", 8}, {"expert_ids", ids}, {"scores", scores},
                 {"scores_bf16_u16", words},
                 {"production_route_changed", false}};
  std::ofstream stream(output);
  stream << report.dump(2) << '\n';
  if (!stream) throw std::runtime_error("P12 route trace write failed");
}
} // namespace
int main(int argc, char **argv) {
  try {
    if (argc != 4)
      throw std::invalid_argument("p12_route_tests FIXTURE LIBRARIES OUTPUT");
    exercise(argv[1], argv[2], argv[3]);
    std::cout << "P12 exact MoE A/B and production trace passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "P12 route failure: " << error.what() << '\n';
    return 1;
  }
}
