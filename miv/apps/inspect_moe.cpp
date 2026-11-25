#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "../src/route_trace_schema.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/moe.h"
#include "omlx_m5/mxfp4_small_t.h"
#include "omlx_m5/quantization.h"

namespace {

constexpr std::array kAnchors = {1U, 8U, 13U, 128U};

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::vector<float> make_input(std::uint32_t tokens, std::uint64_t width,
                              std::vector<std::uint16_t> &words) {
  std::vector<float> result(tokens * width);
  words.resize(tokens * width);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint64_t column = 0; column < width; ++column) {
      const auto raw =
          static_cast<float>(
              static_cast<std::int32_t>((column * 17 + token * 31) % 257) -
              128) /
          512.0F;
      const auto index = token * width + column;
      words[index] = omlx::m5::float_to_bf16(raw);
      result[index] = omlx::m5::bf16_to_float(words[index]);
    }
  }
  return result;
}

float decode_e2m1(std::uint32_t code) {
  constexpr std::array magnitude = {0.0F, 0.5F, 1.0F, 1.5F,
                                    2.0F, 3.0F, 4.0F, 6.0F};
  const auto value = magnitude[code & 7U];
  return (code & 8U) == 0 ? value : -value;
}

float decode_e8m0(std::uint8_t code) {
  return std::ldexp(1.0F, code == 0 ? -127 : static_cast<int>(code) - 127);
}

std::uint8_t encode_e8m0_round_up(float value) {
  if (!(value > 0.0F)) {
    return 0;
  }
  const auto exponent =
      std::clamp(static_cast<int>(std::ceil(std::log2(value))), -127, 127);
  return static_cast<std::uint8_t>(exponent + 127);
}

std::uint8_t encode_e2m1(float value) {
  const auto sign = std::signbit(value) ? 8U : 0U;
  const auto magnitude = std::abs(value);
  std::uint32_t code = 0;
  if (magnitude > 5.0F) {
    code = 7;
  } else if (magnitude >= 3.5F) {
    code = 6;
  } else if (magnitude > 2.5F) {
    code = 5;
  } else if (magnitude >= 1.75F) {
    code = 4;
  } else if (magnitude > 1.25F) {
    code = 3;
  } else if (magnitude >= 0.75F) {
    code = 2;
  } else if (magnitude > 0.25F) {
    code = 1;
  }
  return static_cast<std::uint8_t>(code | sign);
}

struct QuantizedInput {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint8_t> scales;
  std::vector<float> values;
};

QuantizedInput quantize_input(std::span<const float> input,
                              std::uint32_t tokens, std::uint64_t width) {
  QuantizedInput result;
  result.codes.resize(tokens * width / 2);
  result.scales.resize(tokens * width / 32);
  result.values.resize(tokens * width);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint64_t group = 0; group < width / 32; ++group) {
      float maximum = 0.0F;
      for (std::uint32_t lane = 0; lane < 32; ++lane) {
        maximum = std::max(maximum,
                           std::abs(input[token * width + group * 32 + lane]));
      }
      const auto scale_code = encode_e8m0_round_up(maximum / 6.0F);
      const auto scale = decode_e8m0(scale_code);
      result.scales[token * (width / 32) + group] = scale_code;
      for (std::uint32_t lane = 0; lane < 32; ++lane) {
        const auto column = group * 32 + lane;
        const auto code = encode_e2m1(input[token * width + column] / scale);
        result.codes[token * (width / 2) + column / 2] |=
            static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
        result.values[token * width + column] = decode_e2m1(code) * scale;
      }
    }
  }
  return result;
}

struct Routes {
  std::vector<std::uint32_t> indices;
  std::vector<std::uint16_t> score_words;
  std::vector<float> scores;
};

Routes make_routes(std::uint32_t tokens, const std::string &mode,
                   const nlohmann::json &trace) {
  Routes result;
  result.indices.resize(tokens * 8);
  result.score_words.resize(tokens * 8);
  result.scores.resize(tokens * 8);
  if (mode == "trace" && trace.at("expert_ids").size() < tokens)
    throw std::invalid_argument("MoE route trace has too few rows");
  const bool raw_scores = mode == "trace" &&
                          trace.contains("scores_bf16_u16");
  if (raw_scores && trace.at("scores_bf16_u16").size() < tokens)
    throw std::invalid_argument("MoE raw score trace has too few rows");
  for (std::uint32_t token = 0; token < tokens; ++token) {
    float total = 0;
    std::set<std::uint32_t> seen;
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      const auto offset = token * 8 + slot;
      std::uint32_t id = 0;
      if (mode == "shared8")
        id = slot * 31U;
      else if (mode == "hot64")
        id = ((token / 16) % 8) * 8 + slot;
      else if (mode == "trace") {
        if (trace.at("expert_ids").at(token).size() != 8)
          throw std::invalid_argument(
              "MoE trace must contain eight IDs per row");
        id = trace.at("expert_ids").at(token).at(slot).get<std::uint32_t>();
      } else
        id = (token * 37U + slot * 31U + (slot == 7 ? 255U : 0U)) % 256U;
      if (id >= 256 || !seen.insert(id).second)
        throw std::invalid_argument(
            "MoE trace has invalid or duplicate expert IDs");
      result.indices[offset] = id;
      float score = float(slot + 1);
      if (raw_scores) {
        const auto &row = trace.at("scores_bf16_u16").at(token);
        if (row.size() != 8)
          throw std::invalid_argument("MoE raw score row shape mismatch");
        const auto code = row.at(slot).get<std::uint32_t>();
        if (code > UINT16_MAX)
          throw std::invalid_argument("MoE BF16 score code overflow");
        result.score_words[offset] = std::uint16_t(code);
        score = omlx::m5::bf16_to_float(result.score_words[offset]);
        if (trace.contains("scores") &&
            (trace.at("scores").size() < tokens ||
             trace.at("scores").at(token).size() != 8 ||
             trace.at("scores").at(token).at(slot).get<float>() != score))
          throw std::invalid_argument("MoE score float/BF16 mismatch");
      } else if (mode == "trace" && trace.contains("scores")) {
        if (trace.at("scores").size() < tokens ||
            trace.at("scores").at(token).size() != 8)
          throw std::invalid_argument("MoE trace score shape mismatch");
        score = trace.at("scores").at(token).at(slot).get<float>();
      }
      if (!std::isfinite(score) || score < 0)
        throw std::invalid_argument("MoE route score invalid");
      result.scores[offset] = score;
      total += score;
    }
    if (!(total > 0))
      throw std::invalid_argument("MoE route score mass is zero");
    if (raw_scores) {
      if (std::abs(total - 1.0f) > 0.02f)
        throw std::invalid_argument("MoE recorded score mass invalid");
    } else {
      for (std::uint32_t slot = 0; slot < 8; ++slot) {
        const auto offset = token * 8 + slot;
        result.score_words[offset] =
            omlx::m5::float_to_bf16(result.scores[offset] / total);
        result.scores[offset] =
            omlx::m5::bf16_to_float(result.score_words[offset]);
      }
    }
  }
  return result;
}

std::uint64_t route_digest(const Routes &routes) {
  std::uint64_t hash = 14695981039346656037ULL;
  for (auto id : routes.indices) {
    hash ^= id;
    hash *= 1099511628211ULL;
  }
  for (auto score : routes.score_words) {
    hash ^= score;
    hash *= 1099511628211ULL;
  }
  return hash;
}

nlohmann::json route_statistics(const Routes &routes, std::uint32_t tokens) {
  std::array<std::uint32_t, 256> counts{};
  double overlap = 0;
  for (std::uint32_t token = 0; token < tokens; ++token)
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      const auto id = routes.indices[token * 8 + slot];
      ++counts[id];
      if (token)
        overlap += std::find(routes.indices.begin() + (token - 1) * 8,
                             routes.indices.begin() + token * 8,
                             id) != routes.indices.begin() + token * 8;
    }
  return {{"unique_experts", std::count_if(counts.begin(), counts.end(),
                                           [](auto n) { return n > 0; })},
          {"max_assignments_per_expert",
           *std::max_element(counts.begin(), counts.end())},
          {"adjacent_top8_overlap", tokens > 1 ? overlap / (tokens - 1) : 0},
          {"expert_counts", counts}};
}

omlx::m5::KernelKey make_key(const omlx::m5::MoeExpertPlan &plan,
                             omlx::m5::ActivationProfile activation,
                             omlx::m5::KernelAlignmentClass alignment) {
  return {
      .format_id = plan.info().format_id,
      .layout_id = plan.info().layout_id,
      .activation_policy = plan.info().format_id.starts_with("mxfp4")
                               ? omlx::m5::ActivationPolicy::allow_a4
                               : omlx::m5::ActivationPolicy::a16_only,
      .preferred_activation = activation,
      .input_type = omlx::m5::KernelDataType::bf16,
      .output_type = omlx::m5::KernelDataType::bf16,
      .accumulator_type = omlx::m5::KernelDataType::fp32,
      .geometry = omlx::m5::KernelGeometry::moe_expert,
      .output_features = plan.info().output_features,
      .input_features = plan.info().input_features,
      .tokens = plan.info().tokens,
      .experts = 256,
      .top_k = 8,
      .fused_op = plan.info().fused_op,
      .alignment = alignment,
  };
}

double relative_error(double actual, double expected) {
  return std::abs(actual - expected) / std::max(1.0, std::abs(expected));
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 5 || (argc - 5) % 2)
      throw std::invalid_argument(
          "usage: inspect_moe ARTIFACT OBJECT MOE_METALLIB A4_METALLIB "
          "[--iterations N] [--route-mode spread|shared8|hot64|trace] "
          "[--route-file JSON] [--a16-only 0|1]");
    std::uint32_t iterations = 2;
    std::string route_mode = "spread", route_file;
    bool a16_only = false;
    for (int arg = 5; arg < argc; arg += 2) {
      const std::string key = argv[arg];
      if (key == "--iterations") {
        const auto value = std::stoul(argv[arg + 1]);
        if (!value || value > UINT32_MAX)
          throw std::invalid_argument("--iterations must be positive");
        iterations = std::uint32_t(value);
      } else if (key == "--route-mode")
        route_mode = argv[arg + 1];
      else if (key == "--route-file")
        route_file = argv[arg + 1];
      else if (key == "--a16-only") {
        if (std::string_view(argv[arg + 1]) != "0" &&
            std::string_view(argv[arg + 1]) != "1")
          throw std::invalid_argument("--a16-only must be 0 or 1");
        a16_only = std::string_view(argv[arg + 1]) == "1";
      } else
        throw std::invalid_argument("unknown MoE probe option");
    }
    if (route_mode != "spread" && route_mode != "shared8" &&
        route_mode != "hot64" && route_mode != "trace")
      throw std::invalid_argument("unknown MoE route mode");
    nlohmann::json trace;
    if (route_mode == "trace") {
      std::ifstream input(route_file);
      if (!input)
        throw std::invalid_argument("MoE route file unavailable");
      input >> trace;
      if (!trace.contains("origin") || !trace.contains("synthetic") ||
          !trace.at("synthetic").is_boolean() ||
          trace.at("origin").get<std::string>().empty())
        throw std::invalid_argument(
            "MoE trace requires origin and synthetic flag");
      if(trace.value("schema","")=="omlx-m5-p12-route-trace-v2")omlx::m5::detail::validate_route_trace_v2(trace);
      if (trace.contains("scores_bf16_u16") &&
          ((trace.value("schema", "") != "omlx-m5-p12-route-trace-v1" && trace.value("schema", "") != "omlx-m5-p12-route-trace-v2") ||
           (trace.value("source_kind", "") != "small_control_fixture" &&
            trace.value("source_kind", "") != "provided_target_tokens") ||
           trace.value("source_artifact", "").empty() ||
           trace.value("source_profile_function", "").empty() ||
           trace.value("router_weight_object", "").empty() ||
           trace.value("router_topk_kernel", "") !=
               "omlx_m5_forward_router_top8_logits" ||
           trace.value("experts", 0U) != 256 ||
           trace.value("top_k", 0U) != 8))
        throw std::invalid_argument("P12 recorded trace provenance invalid");
    } else if (!route_file.empty())
      throw std::invalid_argument("route file requires trace mode");
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto &object = artifact.object(argv[2]);
    if (object.shape.size() != 3 || object.shape[0] != 256 ||
        object.format.starts_with("direct_")) {
      throw std::invalid_argument(
          "MoE inspector requires a quantized 256-expert parent");
    }
    const bool mxfp4 = object.format == "mxfp4_e2m1_g32_e8m0_v1";
    const bool apply_scores = object.shape[1] == 2048;
    auto context = omlx::m5::MetalContext::create();
    std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> plane_buffers;
    omlx::m5::MaterializedWeightView weight{
        .object_id = object.id,
        .component = omlx::m5::classify_model_component(object.name),
        .name = object.name,
        .format = object.format,
        .layout = object.layout,
        .parent_shape = object.shape,
        .padded_k = object.padded_k,
        .row_count = object.shape[0] * object.shape[1],
        .in_features = object.shape[2],
    };
    std::uint64_t uploaded_weight_bytes = 0;
    for (const auto &selected_plane : object.planes) {
      plane_buffers.push_back(std::make_unique<omlx::m5::MetalBuffer>(
          context.upload(artifact.plane_bytes(object, selected_plane))));
      weight.planes.push_back({
          .kind = selected_plane.kind,
          .buffer = plane_buffers.back().get(),
          .segment_index = static_cast<std::uint32_t>(plane_buffers.size() - 1),
          .bytes = selected_plane.bytes,
          .row_stride =
              selected_plane.bytes / (object.shape[0] * object.shape[1]),
      });
      uploaded_weight_bytes += selected_plane.bytes;
    }

    std::uint64_t route_keys = 0;
    const auto check_plan = [&](std::uint32_t tokens) {
      auto plan = omlx::m5::prepare_moe_expert(
          context, argv[3], weight, tokens,
          apply_scores ? omlx::m5::KernelFusedOp::moe_down
                       : omlx::m5::KernelFusedOp::moe_gate_up,
          apply_scores);
      omlx::m5::DispatchTable table;
      for (const auto &candidate : plan.dispatch_candidates()) {
        table.register_candidate(candidate);
      }
      const auto installed = plan.installed_executor_keys();
      table.seal(installed);
      static_cast<void>(
          table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                                 omlx::m5::KernelAlignmentClass::full),
                        context.capabilities().profile));
      static_cast<void>(
          table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                                 omlx::m5::KernelAlignmentClass::tail),
                        context.capabilities().profile));
      route_keys += 2;
      if (mxfp4) {
        static_cast<void>(
            table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                                   omlx::m5::KernelAlignmentClass::full),
                          context.capabilities().profile));
        static_cast<void>(
            table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                                   omlx::m5::KernelAlignmentClass::tail),
                          context.capabilities().profile));
        route_keys += 2;
      }
    };
    for (std::uint32_t tokens = 1; tokens <= 128; ++tokens) {
      check_plan(tokens);
    }
    check_plan(512);
    check_plan(1024);

    nlohmann::json records = nlohmann::json::array();
    for (const auto tokens : kAnchors) {
      const auto routes = make_routes(tokens, route_mode, trace);
      auto route_indices = context.upload(bytes(routes.indices));
      auto route_scores = context.upload(bytes(routes.score_words));
      const omlx::m5::MoeRouteView route{
          .indices = {.buffer = &route_indices, .bytes = tokens * 8 * 4},
          .scores = {.buffer = &route_scores, .bytes = tokens * 8 * 2},
          .tokens = tokens,
      };
      std::vector<std::uint16_t> input_words;
      const auto input_rows = apply_scores ? tokens * 8 : tokens;
      const auto input_values =
          make_input(input_rows, object.shape[2], input_words);
      auto input = context.upload(bytes(input_words));
      auto plan = omlx::m5::prepare_moe_expert(
          context, argv[3], weight, tokens,
          apply_scores ? omlx::m5::KernelFusedOp::moe_down
                       : omlx::m5::KernelFusedOp::moe_gate_up,
          apply_scores);
      omlx::m5::DispatchTable table;
      for (const auto &candidate : plan.dispatch_candidates()) {
        table.register_candidate(candidate);
      }
      const auto installed = plan.installed_executor_keys();
      table.seal(installed);
      const auto grouped =
          table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                                 omlx::m5::KernelAlignmentClass::full),
                        context.capabilities().profile);
      const auto fallback =
          table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                                 omlx::m5::KernelAlignmentClass::tail),
                        context.capabilities().profile);
      auto output = context.make_buffer(plan.info().output_bytes);
      const auto before = context.registry_stats();
      const auto grouped_submission = omlx::m5::execute_moe_expert_a16(
          context, plan, grouped.candidate.executor_key, input, 0, route,
          output, 0);
      const auto grouped_output = output.read();
      const auto fallback_submission = omlx::m5::execute_moe_expert_a16(
          context, plan, fallback.candidate.executor_key, input, 0, route,
          output, 0);
      const auto fallback_output = output.read();
      if (grouped_output != fallback_output) {
        throw std::runtime_error("real MoE grouped/fallback A16 differ");
      }
      std::vector<double> grouped_a16_wall_ms;
      std::vector<double> grouped_a16_gpu_ms;
      std::vector<double> fallback_a16_wall_ms;
      std::vector<double> fallback_a16_gpu_ms;
      grouped_a16_wall_ms.reserve(iterations);
      grouped_a16_gpu_ms.reserve(iterations);
      fallback_a16_wall_ms.reserve(iterations);
      fallback_a16_gpu_ms.reserve(iterations);
      for (std::uint32_t iteration = 0; iteration < iterations + 3;
           ++iteration) {
        omlx::m5::MetalSubmission measured_grouped, measured_fallback;
        const auto run_grouped = [&] {
          measured_grouped = omlx::m5::execute_moe_expert_a16(
              context, plan, grouped.candidate.executor_key, input, 0, route,
              output, 0);
        };
        const auto run_fallback = [&] {
          measured_fallback = omlx::m5::execute_moe_expert_a16(
              context, plan, fallback.candidate.executor_key, input, 0, route,
              output, 0);
        };
        if (iteration % 2) {
          run_fallback();
          run_grouped();
        } else {
          run_grouped();
          run_fallback();
        }
        if (iteration < 3)
          continue;
        grouped_a16_wall_ms.push_back(measured_grouped.timing.wall_ms);
        grouped_a16_gpu_ms.push_back(measured_grouped.timing.gpu_ms);
        fallback_a16_wall_ms.push_back(measured_fallback.timing.wall_ms);
        fallback_a16_gpu_ms.push_back(measured_fallback.timing.gpu_ms);
      }
      const auto after_a16 = context.registry_stats();
      std::uint64_t token_loop_uploads =
          after_a16.upload_count - before.upload_count;
      std::uint64_t token_loop_pipeline_compiles =
          after_a16.pipelines_compiled - before.pipelines_compiled;

      std::optional<QuantizedInput> quantized;
      std::vector<std::byte> grouped_a4_output;
      std::vector<std::byte> fallback_a4_output;
      std::vector<double> grouped_a4_wall_ms;
      std::vector<double> grouped_a4_gpu_ms;
      std::vector<double> fallback_a4_wall_ms;
      std::vector<double> fallback_a4_gpu_ms;
      if (mxfp4 && !a16_only) {
        quantized = quantize_input(input_values, input_rows, object.shape[2]);
        auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
            context, argv[4], object.shape[2], input_rows);
        auto codes = context.make_buffer(quantizer.info().code_bytes);
        auto scales = context.make_buffer(quantizer.info().scale_bytes);
        const omlx::m5::Mxfp4ActivationView activation{
            .codes = {.buffer = &codes, .bytes = quantizer.info().code_bytes},
            .scales = {.buffer = &scales,
                       .bytes = quantizer.info().scale_bytes},
            .group_size = 32,
        };
        static_cast<void>(omlx::m5::quantize_mxfp4_a4(context, quantizer, input,
                                                      0, activation));
        const auto grouped_a4 =
            table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                                   omlx::m5::KernelAlignmentClass::full),
                          context.capabilities().profile);
        const auto fallback_a4 =
            table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                                   omlx::m5::KernelAlignmentClass::tail),
                          context.capabilities().profile);
        const auto before_a4 = context.registry_stats();
        static_cast<void>(omlx::m5::execute_moe_expert_a4(
            context, plan, grouped_a4.candidate.executor_key, activation, route,
            output, 0));
        grouped_a4_output = output.read();
        static_cast<void>(omlx::m5::execute_moe_expert_a4(
            context, plan, fallback_a4.candidate.executor_key, activation,
            route, output, 0));
        fallback_a4_output = output.read();
        if (grouped_a4_output != fallback_a4_output) {
          throw std::runtime_error("real MoE grouped/fallback A4 differ");
        }
        grouped_a4_wall_ms.reserve(iterations);
        grouped_a4_gpu_ms.reserve(iterations);
        fallback_a4_wall_ms.reserve(iterations);
        fallback_a4_gpu_ms.reserve(iterations);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          const auto measured_grouped = omlx::m5::execute_moe_expert_a4(
              context, plan, grouped_a4.candidate.executor_key, activation,
              route, output, 0);
          grouped_a4_wall_ms.push_back(measured_grouped.timing.wall_ms);
          grouped_a4_gpu_ms.push_back(measured_grouped.timing.gpu_ms);
          const auto measured_fallback = omlx::m5::execute_moe_expert_a4(
              context, plan, fallback_a4.candidate.executor_key, activation,
              route, output, 0);
          fallback_a4_wall_ms.push_back(measured_fallback.timing.wall_ms);
          fallback_a4_gpu_ms.push_back(measured_fallback.timing.gpu_ms);
        }
        const auto after_a4 = context.registry_stats();
        token_loop_uploads += after_a4.upload_count - before_a4.upload_count;
        token_loop_pipeline_compiles +=
            after_a4.pipelines_compiled - before_a4.pipelines_compiled;
      }

      const std::array sample_assignments = {0U, tokens * 8 / 2,
                                             tokens * 8 - 1};
      const std::array sample_rows = {
          0U, static_cast<std::uint32_t>(object.shape[1] / 2),
          static_cast<std::uint32_t>(object.shape[1] - 1)};
      double a16_max_rel = 0.0;
      double a4_max_rel = 0.0;
      nlohmann::json samples = nlohmann::json::array();
      std::unordered_map<std::uint64_t, std::vector<float>> decoded;
      for (const auto assignment : sample_assignments) {
        const auto token = assignment / 8;
        const auto input_row = apply_scores ? assignment : token;
        const auto expert = routes.indices[assignment];
        for (const auto row : sample_rows) {
          const auto flat_row = expert * object.shape[1] + row;
          if (!decoded.contains(flat_row)) {
            decoded.emplace(flat_row,
                            artifact.decode_weight_row(object, flat_row, 0,
                                                       object.shape[2]));
          }
          double expected_a16 = 0.0;
          double expected_a4 = 0.0;
          for (std::uint64_t column = 0; column < object.shape[2]; ++column) {
            expected_a16 +=
                static_cast<double>(
                    input_values[input_row * object.shape[2] + column]) *
                decoded.at(flat_row)[column];
            if (quantized.has_value()) {
              expected_a4 +=
                  static_cast<double>(
                      quantized->values[input_row * object.shape[2] + column]) *
                  decoded.at(flat_row)[column];
            }
          }
          if (apply_scores) {
            expected_a16 *= routes.scores[assignment];
            expected_a4 *= routes.scores[assignment];
          }
          const auto output_index = assignment * object.shape[1] + row;
          std::uint16_t word = 0;
          std::memcpy(&word, grouped_output.data() + output_index * 2,
                      sizeof(word));
          const auto actual_a16 = omlx::m5::bf16_to_float(word);
          a16_max_rel =
              std::max(a16_max_rel, relative_error(actual_a16, expected_a16));
          nlohmann::json sample = {
              {"assignment", assignment}, {"token", token},
              {"expert", expert},         {"row", row},
              {"actual_a16", actual_a16}, {"cpp_expected_a16", expected_a16},
          };
          if (quantized.has_value()) {
            std::memcpy(&word, grouped_a4_output.data() + output_index * 2,
                        sizeof(word));
            const auto actual_a4 = omlx::m5::bf16_to_float(word);
            a4_max_rel =
                std::max(a4_max_rel, relative_error(actual_a4, expected_a4));
            sample["actual_a4"] = actual_a4;
            sample["cpp_expected_a4"] = expected_a4;
          }
          samples.push_back(std::move(sample));
        }
      }
      if (a16_max_rel > 0.02 || (mxfp4 && a4_max_rel > 0.02)) {
        throw std::runtime_error("real MoE output exceeds tolerance");
      }
      records.push_back({
          {"tokens", tokens},
          {"route_statistics", route_statistics(routes, tokens)},
          {"route_digest", route_digest(routes)},
          {"warmups", 3},
          {"assignments", plan.info().assignments},
          {"apply_route_scores", apply_scores},
          {"candidate_count", table.stats().candidate_count},
          {"installed_candidate_count",
           table.stats().installed_candidate_count},
          {"grouped_family",
           omlx::m5::to_string(
               plan.executor(grouped.candidate.executor_key).family)},
          {"fallback_family",
           omlx::m5::to_string(
               plan.executor(fallback.candidate.executor_key).family)},
          {"grouped_dispatches", grouped_submission.dispatch_count},
          {"fallback_dispatches", fallback_submission.dispatch_count},
          {"iterations", iterations},
          {"grouped_a16_wall_ms", std::move(grouped_a16_wall_ms)},
          {"grouped_a16_gpu_ms", std::move(grouped_a16_gpu_ms)},
          {"fallback_a16_wall_ms", std::move(fallback_a16_wall_ms)},
          {"fallback_a16_gpu_ms", std::move(fallback_a16_gpu_ms)},
          {"grouped_a4_wall_ms", std::move(grouped_a4_wall_ms)},
          {"grouped_a4_gpu_ms", std::move(grouped_a4_gpu_ms)},
          {"fallback_a4_wall_ms", std::move(fallback_a4_wall_ms)},
          {"fallback_a4_gpu_ms", std::move(fallback_a4_gpu_ms)},
          {"grouped_fallback_a16_bit_exact", true},
          {"grouped_fallback_a4_bit_exact", mxfp4 && !a16_only},
          {"a16_max_rel_error", a16_max_rel},
          {"a4_max_rel_error", a4_max_rel},
          {"token_loop_uploads", token_loop_uploads},
          {"token_loop_pipeline_compiles", token_loop_pipeline_compiles},
          {"samples", std::move(samples)},
      });
    }
    const nlohmann::json result = {
        {"schema", "omlx-m5-b2-6-moe-v1"},
        {"evidence_scope",
         "M4 real 256-expert execution; not M5 Ultra performance"},
        {"route_mode", route_mode},
        {"route_provenance",
         route_mode == "trace"
             ? trace.at("origin")
             : nlohmann::json("deterministic synthetic route stress pattern")},
        {"synthetic_routes",
         route_mode == "trace" ? trace.at("synthetic") : nlohmann::json(true)},
        {"trace_schema", route_mode == "trace" ?
             trace.value("schema", "legacy_trace") : "synthetic_pattern"},
        {"trace_source_artifact", route_mode == "trace" ?
             trace.value("source_artifact", "unspecified") : "none"},
        {"trace_scores_bf16_preserved", route_mode == "trace" &&
             trace.contains("scores_bf16_u16")},
        {"a16_only", a16_only},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"object", object.name},
        {"format", object.format},
        {"layout", object.layout},
        {"shape", object.shape},
        {"apply_route_scores", apply_scores},
        {"uploaded_weight_bytes", uploaded_weight_bytes},
        {"route_sweep",
         {
             {"tokens", 130},
             {"route_keys", route_keys},
             {"uncovered", 0},
         }},
        {"records", std::move(records)},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MoE inspection failed: " << error.what() << '\n';
    return 1;
  }
}
