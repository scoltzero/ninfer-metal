#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "omlx_m5/artifact.h"
#include "omlx_m5/qwen_forward.h"
#include <nlohmann/json.hpp>

using namespace omlx::m5;
using Json = nlohmann::json;

std::vector<float> flatten(const Json &rows) {
  std::vector<float> result;
  const auto visit = [&](const auto &self, const Json &value) -> void {
    if (value.is_array()) {
      for (const auto &child : value)
        self(self, child);
    } else
      result.push_back(value.get<float>());
  };
  visit(visit, rows);
  return result;
}

Json compare(const std::vector<float> &actual,
             const std::vector<float> &expected, const std::string &label) {
  if (actual.size() != expected.size())
    throw std::runtime_error(label + ": shape mismatch");
  double sum = 0, scale = 0;
  float maximum = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    if (!std::isfinite(actual[i]))
      throw std::runtime_error(label + ": nonfinite output");
    auto diff = actual[i] - expected[i];
    maximum = std::max(maximum, std::abs(diff));
    sum += double(diff) * diff;
    scale += double(expected[i]) * expected[i];
  }
  double relative = std::sqrt(sum / std::max(scale, 1e-12));
  return {{"label", label},
          {"max_abs", maximum},
          {"relative_l2", relative},
          {"passed", maximum <= 0.08 && relative <= 0.02}};
}

int main(int argc, char **argv) {
  try {
    if (argc != 3)
      throw std::invalid_argument(
          "qwen_forward_tests FIXTURES LIBRARY_DIRECTORY");
    Json results = Json::array();
    for (const auto *mode : {"affine", "mxfp4", "moe-affine", "moe-mxfp4",
                             "hf-mxfp4", "mixed-oq", "moe-mixed-oq"}) {
      const auto root = std::filesystem::path(argv[1]) / mode;
      auto config = load_qwen_model_config(root / "source/config.json");
      auto artifact = ArtifactModel::open(root / "model.omlxm5", true);
      auto context = MetalContext::create();
      auto materialization = plan_model_materialization(artifact, context);
      auto weights =
          materialize_model(artifact, std::move(materialization), context);
      auto layers = build_qwen_layer_catalog(artifact, config, 1024, 64);
      const std::array<std::uint32_t, 2> taps{0, 1};
      const auto allocation_before = context.registry_stats();
      QwenForward program(context, weights, layers, config, 1024, argv[2], taps,
                          true);
      const auto allocation_after = context.registry_stats();
      const auto planned_bytes = program.memory_plan().buffer_bytes();
      const auto parameter_bytes = allocation_after.dynamic_parameter_bytes -
                                   allocation_before.dynamic_parameter_bytes +
                                   allocation_after.stable_parameter_bytes -
                                   allocation_before.stable_parameter_bytes;
      if (allocation_after.current_buffer_bytes -
              allocation_before.current_buffer_bytes !=
          planned_bytes + parameter_bytes)
        throw std::runtime_error(
            "Qwen buffer ledger disagrees with actual allocations");
      const auto budget =
          qwen_load_budget(weights.stats().resident_bytes,
                           program.memory_plan(), 128ULL << 20, 120ULL << 20);
      if (budget.fits)
        throw std::runtime_error(
            "Qwen admission ignored existing allocations and runtime reserve");
      const auto boundary = qwen_load_budget(
          weights.stats().resident_bytes, program.memory_plan(),
          budget.required_bytes + 4096, 4096);
      if (!boundary.fits || boundary.available_bytes != boundary.required_bytes)
        throw std::runtime_error(
            "Qwen admission rejected the exact capacity boundary");
      std::ifstream input(root / "oracle.json");
      Json oracle;
      input >> oracle;
      const auto before = context.registry_stats();
      for (const auto &test : oracle.at("cases")) {
        auto tokens = test.at("tokens").get<std::vector<std::uint32_t>>();
        program.reset();
        auto direct = program.run(tokens, {}, true, false);
        auto hidden_error =
            compare(direct.hidden, flatten(test.at("reference_hidden")),
                    std::string(mode) +
                        " FP64 hidden T=" + std::to_string(tokens.size()));
        auto logit_error =
            compare(direct.logits, flatten(test.at("reference_logits")),
                    std::string(mode) + " FP64 logits");
        auto stock_difference =
            compare(direct.logits, flatten(test.at("logits")),
                    std::string(mode) +
                        " stock MLX comparison (different private rounding)");
        auto feature_error = compare(
            direct.layer_features, flatten(test.at("reference_features")),
            std::string(mode) + " requested layer features");
        program.reset();
        auto replay = program.run(tokens, {}, true, true);
        if (direct.hidden != replay.hidden || direct.logits != replay.logits ||
            direct.token != replay.token || replay.executed_layers != 2)
          throw std::runtime_error("ICB/direct full forward mismatch");
        Json diagnostic = nullptr;
        if (std::string_view(mode) == "affine" && tokens.size() == 8) {
          const auto description =
              program.inspect_profile(std::uint32_t(tokens.size()));
          program.reset();
          const auto staged = program.run(tokens, {}, true, true, true);
          std::uint64_t dispatches = 0;
          for (const auto &stage : staged.stage_timings)
            dispatches += stage.dispatch_count;
          if (staged.hidden != direct.hidden ||
              staged.logits != direct.logits || staged.token != direct.token ||
              staged.stage_timings.size() != description.stages.size() ||
              dispatches != description.dispatch_count ||
              staged.submission.dispatch_count !=
                  direct.submission.dispatch_count ||
              description.weights.empty())
            throw std::runtime_error(
                "diagnostic stage replay changed production output or plan");
          Json stages = Json::array(), routes = Json::array();
          for (const auto &stage : staged.stage_timings)
            stages.push_back({{"stage", stage.stage},
                              {"dispatches", stage.dispatch_count},
                              {"gpu_ms", stage.gpu_ms},
                              {"wall_ms", stage.wall_ms}});
          for (const auto &route : description.weights)
            routes.push_back({{"stage", route.stage},
                              {"object", route.object},
                              {"format", route.format},
                              {"n", route.n},
                              {"k", route.k},
                              {"capacity", route.capacity},
                              {"library", route.library},
                              {"kernel", route.function},
                              {"family", route.family},
                              {"selection_source", route.source},
                              {"fallback_reason", route.fallback_reason},
                              {"batch", 1},
                              {"grid", {{"width", route.grid.width},
                                        {"height", route.grid.height},
                                        {"depth", route.grid.depth}}},
                              {"threads_per_threadgroup",
                               route.threads_per_threadgroup.width}});
          diagnostic = {{"scope", "small complete model diagnostic ICB ranges; "
                                  "extra command buffer per stage"},
                        {"capacity", description.capacity},
                        {"dispatches", description.dispatch_count},
                        {"stages", stages},
                        {"weight_routes", routes},
                        {"bit_exact_to_normal_profile", true}};
          std::map<std::string, std::vector<double>> samples;
          std::vector<double> normal_samples, staged_samples;
          for (std::uint32_t iteration = 0; iteration < 13; ++iteration) {
            program.reset();
            const auto normal_round = program.run(tokens, {}, false, true);
            program.reset();
            const auto staged_round =
                program.run(tokens, {}, false, true, true);
            if (normal_round.token != staged_round.token)
              throw std::runtime_error("diagnostic stage token changed");
            if (iteration < 3)
              continue;
            normal_samples.push_back(normal_round.submission.timing.gpu_ms);
            staged_samples.push_back(staged_round.submission.timing.gpu_ms);
            for (const auto &stage : staged_round.stage_timings)
              samples[stage.stage].push_back(stage.gpu_ms);
          }
          const auto median = [](std::vector<double> values) {
            std::ranges::sort(values);
            return (values[values.size() / 2] +
                    values[(values.size() - 1) / 2]) /
                   2;
          };
          Json stage_medians = Json::array();
          for (const auto &[name, values] : samples)
            stage_medians.push_back(
                {{"stage", name}, {"gpu_median_ms", median(values)}});
          diagnostic["repeats"] = 10;
          diagnostic["normal_gpu_median_ms"] = median(normal_samples);
          diagnostic["staged_gpu_median_ms"] = median(staged_samples);
          diagnostic["stage_medians"] = stage_medians;
        }
        program.reset();
        std::vector<float> sequential, sequential_hidden, sequential_features,
            sequential_router, sequential_scores;
        std::vector<std::uint32_t> sequential_routes;
        for (const auto token : tokens) {
          const auto step = program.run(std::span(&token, 1), {}, true);
          sequential.insert(sequential.end(), step.logits.begin(),
                            step.logits.end());
          sequential_hidden.insert(sequential_hidden.end(), step.hidden.begin(),
                                   step.hidden.end());
          sequential_features.insert(sequential_features.end(),
                                     step.layer_features.begin(),
                                     step.layer_features.end());
          sequential_router.insert(sequential_router.end(),
                                   step.router_logits.begin(),
                                   step.router_logits.end());
          sequential_scores.insert(sequential_scores.end(),
                                   step.router_scores.begin(),
                                   step.router_scores.end());
          sequential_routes.insert(sequential_routes.end(),
                                   step.layer_routes.begin(),
                                   step.layer_routes.end());
        }
        auto sequential_error =
            compare(sequential, flatten(test.at("reference_logits")),
                    std::string(mode) + " FP64 cached decode");
        if (config.num_experts) {
          std::ofstream trace(
              root / ("native-t" + std::to_string(tokens.size()) + ".json"));
          trace << Json({{"direct",
                          {{"hidden", direct.hidden},
                           {"logits", direct.logits},
                           {"features", direct.layer_features},
                           {"layer_routes", direct.layer_routes},
                           {"router_logits", direct.router_logits},
                           {"router_scores", direct.router_scores}}},
                         {"sequential",
                          {{"hidden", sequential_hidden},
                           {"logits", sequential},
                           {"features", sequential_features},
                           {"layer_routes", sequential_routes},
                           {"router_logits", sequential_router},
                           {"router_scores", sequential_scores}}}})
                       .dump();
        }
        results.push_back({{"mode", mode},
                           {"tokens", tokens.size()},
                           {"hidden", hidden_error},
                           {"logits", logit_error},
                           {"sequential", sequential_error},
                           {"features", feature_error},
                           {"stock_mlx_difference", stock_difference},
                           {"dispatches", direct.submission.dispatch_count},
                           {"diagnostic", diagnostic}});
      }
      program.reset();
      const std::array<std::uint32_t, 3> serving{256, 8, 19};
      auto production = program.run(serving);
      if (production.host_read_bytes != 4 || !production.hidden.empty() ||
          !production.logits.empty())
        throw std::runtime_error("production performed tensor readback");
      const auto after = context.registry_stats();
      if (after.pipelines_compiled != before.pipelines_compiled ||
          after.command_plans_built != before.command_plans_built ||
          after.upload_count != before.upload_count ||
          after.current_buffer_count != before.current_buffer_count)
        throw std::runtime_error("forward changed installation resources");
      if (std::string_view(mode) == "affine") {
        // Verify non-power-of-two capacity ladder tail (max_context=500 -> 300 tokens),
        // non-bucket small-T (max_context=5 -> T=5), and shared-MetalContext plan-key
        // disambiguation across distinct executor configurations.
        auto layers_500 = build_qwen_layer_catalog(artifact, config, 500, 64);
        const std::array<std::uint32_t, 1> single_tap{0};
        QwenForward program_500(context, weights, layers_500, config, 500,
                                argv[2], single_tap, false);
        std::vector<std::uint32_t> mid_prefill(300);
        for (std::size_t idx = 0; idx < mid_prefill.size(); ++idx)
          mid_prefill[idx] = std::uint32_t((idx * 13 + 5) % config.vocab_size);
        const auto res_300 = program_500.run(mid_prefill);
        if (res_300.tokens != 300 || res_300.token >= std::uint32_t(config.vocab_size))
          throw std::runtime_error("500-context prefill tail failed");

        auto layers_5 = build_qwen_layer_catalog(artifact, config, 5, 64);
        QwenForward program_5(context, weights, layers_5, config, 5, argv[2],
                              single_tap, false);
        const std::array<std::uint32_t, 5> small_t5{1, 2, 3, 4, 5};
        const auto res_5 = program_5.run(small_t5);
        if (res_5.tokens != 5 || res_5.token >= std::uint32_t(config.vocab_size))
          throw std::runtime_error("small-T=5 bucket routing failed");
      }
    }
    std::cout << results.dump(2) << '\n';
    bool routing_qualification = false;
    for (const auto &record : results)
      for (const auto *metric : {"hidden", "logits", "sequential", "features"})
        if (!record.at(metric).at("passed").get<bool>()) {
          if (!record.at("mode").get<std::string>().starts_with("moe-"))
            return 1;
          routing_qualification = true;
        }
    // A nonzero status prevents this raw trace collector from qualifying MoE
    // on its own. The independent Python validator must check routing inputs,
    // exact top-k/score semantics and the full selected arithmetic/state path.
    return routing_qualification ? 3 : 0;
  } catch (const std::exception &e) {
    std::cerr << "Qwen forward failed: " << e.what() << '\n';
    return 1;
  }
}
