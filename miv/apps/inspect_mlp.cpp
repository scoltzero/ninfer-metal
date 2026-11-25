#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/mlp.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/moe.h"
#include "omlx_m5/mxfp4_small_t.h"
#include "omlx_m5/projection.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"

namespace {

std::uint32_t benchmark_iterations() {
  const char *value = std::getenv("M5_SLICE_ITERATIONS");
  return value == nullptr ? 0U : static_cast<std::uint32_t>(std::stoul(value));
}

constexpr std::array kAnchors = {1U, 8U, 13U, 128U, 512U, 1024U};

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::string remove_suffix(std::string_view value, std::string_view suffix) {
  if (!value.ends_with(suffix)) {
    throw std::invalid_argument("MLP binding suffix mismatch");
  }
  return std::string(value.substr(0, value.size() - suffix.size()));
}

const omlx::m5::ArtifactBinding &
binding_by_use(const omlx::m5::ArtifactModel &artifact, std::string_view use) {
  const auto found = std::ranges::find_if(
      artifact.bindings(), [&](const omlx::m5::ArtifactBinding &binding) {
        return binding.use == use;
      });
  if (found == artifact.bindings().end()) {
    throw std::out_of_range("MLP binding use is absent: " + std::string(use));
  }
  return *found;
}

const omlx::m5::ArtifactBinding &
binding_by_name(const omlx::m5::ArtifactModel &artifact,
                std::string_view name) {
  const auto found = std::ranges::find_if(
      artifact.bindings(), [&](const omlx::m5::ArtifactBinding &binding) {
        return binding.name == name;
      });
  if (found == artifact.bindings().end()) {
    throw std::out_of_range("MLP binding is absent: " + std::string(name));
  }
  return *found;
}

omlx::m5::MaterializedWeightView
upload_weight(const omlx::m5::ArtifactModel &artifact,
              const omlx::m5::ArtifactObject &object,
              omlx::m5::MetalContext &context,
              std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> &buffers) {
  const auto rows = object.shape.size() == 3 ? object.shape[0] * object.shape[1]
                                             : object.shape[0];
  omlx::m5::MaterializedWeightView weight{
      .object_id = object.id,
      .component = omlx::m5::classify_model_component(object.name),
      .name = object.name,
      .format = object.format,
      .layout = object.layout,
      .parent_shape = object.shape,
      .padded_k = object.padded_k,
      .row_count = rows,
      .in_features = object.shape.back(),
  };
  for (const auto &selected_plane : object.planes) {
    buffers.push_back(std::make_unique<omlx::m5::MetalBuffer>(
        context.upload(artifact.plane_bytes(object, selected_plane))));
    weight.planes.push_back({
        .kind = selected_plane.kind,
        .buffer = buffers.back().get(),
        .segment_index = static_cast<std::uint32_t>(buffers.size() - 1),
        .bytes = selected_plane.bytes,
        .row_stride = selected_plane.bytes / rows,
    });
  }
  return weight;
}

std::vector<std::uint16_t> read_words(const omlx::m5::MetalBuffer &buffer,
                                      std::uint64_t offset,
                                      std::uint64_t count) {
  const auto raw = buffer.read(offset, count * 2);
  std::vector<std::uint16_t> result(count);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

std::vector<float> floats(std::span<const std::uint16_t> words) {
  std::vector<float> result(words.size());
  std::ranges::transform(words, result.begin(), omlx::m5::bf16_to_float);
  return result;
}

std::vector<std::uint16_t> deterministic_words(std::uint32_t rows,
                                               std::uint64_t width,
                                               std::uint32_t column_factor,
                                               std::uint32_t row_factor,
                                               float denominator) {
  std::vector<std::uint16_t> result(rows * width);
  for (std::uint32_t row = 0; row < rows; ++row) {
    for (std::uint64_t column = 0; column < width; ++column) {
      const auto raw =
          static_cast<float>(
              static_cast<std::int32_t>(
                  (column * column_factor + row * row_factor) % 257) -
              128) /
          denominator;
      result[row * width + column] = omlx::m5::float_to_bf16(raw);
    }
  }
  return result;
}

double relative_error(double actual, double expected) {
  return std::abs(actual - expected) / std::max(1.0, std::abs(expected));
}

nlohmann::json run_swiglu_anchors(omlx::m5::MetalContext &context,
                                  std::uint64_t width, bool allow_a4,
                                  const std::filesystem::path &mlp_metallib,
                                  const std::filesystem::path &a4_metallib,
                                  std::uint32_t row_multiplier = 1) {
  nlohmann::json records = nlohmann::json::array();
  for (const auto tokens : kAnchors) {
    const auto rows = tokens * row_multiplier;
    const auto gate_words = deterministic_words(rows, width, 17, 13, 128.0F);
    const auto up_words = deterministic_words(rows, width, 23, 29, 256.0F);
    auto gate_buffer = context.upload(bytes(gate_words));
    auto up_buffer = context.upload(bytes(up_words));
    const omlx::m5::ProjectionParentOutputView gate{
        .role = "mlp_gate",
        .buffer = &gate_buffer,
        .bytes = rows * width * 2,
        .tokens = rows,
        .features = width,
        .token_stride_bytes = width * 2,
    };
    const omlx::m5::ProjectionParentOutputView up{
        .role = "mlp_up",
        .buffer = &up_buffer,
        .bytes = rows * width * 2,
        .tokens = rows,
        .features = width,
        .token_stride_bytes = width * 2,
    };
    auto output_a16 = context.make_buffer(rows * width * 2);
    std::optional<omlx::m5::MetalBuffer> output_codes;
    std::optional<omlx::m5::MetalBuffer> output_scales;
    omlx::m5::SwiGluActivationView output{
        .a16 = omlx::m5::ActivationPlaneView{.buffer = &output_a16,
                                             .bytes = rows * width * 2},
        .rows = rows,
        .features = width,
    };
    if (allow_a4) {
      output_codes.emplace(context.make_buffer(rows * width / 2));
      output_scales.emplace(context.make_buffer(rows * width / 32));
      output.a4 = omlx::m5::Mxfp4ActivationView{
          .codes = {.buffer = &*output_codes, .bytes = rows * width / 2},
          .scales = {.buffer = &*output_scales, .bytes = rows * width / 32},
          .group_size = 32,
      };
    }
    auto plan = omlx::m5::prepare_swiglu(context, mlp_metallib, rows, width);
    const auto before = context.registry_stats();
    const auto submission =
        omlx::m5::produce_swiglu_activation(context, plan, gate, up, output);
    const auto after = context.registry_stats();
    const auto actual_words = read_words(output_a16, 0, rows * width);
    double max_relative = 0.0;
    for (std::size_t index = 0; index < actual_words.size(); ++index) {
      const auto gate_value = omlx::m5::bf16_to_float(gate_words[index]);
      const auto up_value = omlx::m5::bf16_to_float(up_words[index]);
      const auto expected =
          gate_value / (1.0F + std::exp(-gate_value)) * up_value;
      max_relative =
          std::max(max_relative,
                   relative_error(omlx::m5::bf16_to_float(actual_words[index]),
                                  expected));
    }
    bool a4_bit_exact = !allow_a4;
    if (allow_a4) {
      auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
          context, a4_metallib, width, rows);
      auto baseline_codes = context.make_buffer(rows * width / 2);
      auto baseline_scales = context.make_buffer(rows * width / 32);
      const omlx::m5::Mxfp4ActivationView baseline{
          .codes = {.buffer = &baseline_codes, .bytes = rows * width / 2},
          .scales = {.buffer = &baseline_scales, .bytes = rows * width / 32},
          .group_size = 32,
      };
      static_cast<void>(omlx::m5::quantize_mxfp4_a4(context, quantizer,
                                                    output_a16, 0, baseline));
      a4_bit_exact = output_codes->read() == baseline_codes.read() &&
                     output_scales->read() == baseline_scales.read();
    }
    if (submission.dispatch_count != 1 || max_relative > 0.01 ||
        !a4_bit_exact || after.upload_count != before.upload_count ||
        after.pipelines_compiled != before.pipelines_compiled) {
      throw std::runtime_error("SwiGLU anchor validation failed");
    }
    nlohmann::json gpu_ms_samples = nlohmann::json::array();
    if (const auto iterations = benchmark_iterations()) {
      static_cast<void>(
          omlx::m5::produce_swiglu_activation(context, plan, gate, up, output));
      for (std::uint32_t i = 0; i < iterations; ++i)
        gpu_ms_samples.push_back(
            omlx::m5::produce_swiglu_activation(context, plan, gate, up, output)
                .timing.gpu_ms);
    }
    records.push_back({
        {"tokens", tokens},
        {"rows", rows},
        {"mode", allow_a4 ? "a16_a4" : "a16"},
        {"dispatches", submission.dispatch_count},
        {"gpu_ms", submission.timing.gpu_ms},
        {"gpu_ms_samples", std::move(gpu_ms_samples)},
        {"a4_bit_exact", a4_bit_exact},
        {"max_relative_error", max_relative},
        {"token_loop_uploads", after.upload_count - before.upload_count},
        {"token_loop_pipeline_compiles",
         after.pipelines_compiled - before.pipelines_compiled},
        {"a16_bytes", plan.info().a16_bytes},
        {"a4_code_bytes", plan.info().a4_code_bytes},
        {"a4_scale_bytes", plan.info().a4_scale_bytes},
    });
  }
  return records;
}

std::string moe_executor(const omlx::m5::MoeExpertPlan &plan,
                         omlx::m5::MoeExpertFamily family) {
  const auto found = std::ranges::find_if(
      plan.info().executors,
      [&](const omlx::m5::MoeExpertExecutorInfo &executor) {
        return executor.family == family;
      });
  if (found == plan.info().executors.end()) {
    throw std::out_of_range("requested MoE executor is absent");
  }
  return found->executor_key;
}

struct Routes {
  std::vector<std::uint32_t> indices;
  std::vector<std::uint16_t> score_words;
  std::vector<float> scores;
};

Routes make_routes(std::uint32_t tokens) {
  Routes result;
  result.indices.resize(tokens * 8);
  result.score_words.resize(tokens * 8);
  result.scores.resize(tokens * 8);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    float total = 0.0F;
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      const auto index = token * 8 + slot;
      result.indices[index] =
          (token * 37U + slot * 31U + (slot == 7 ? 255U : 0U)) % 256U;
      result.scores[index] = float(slot + 1);
      total += result.scores[index];
    }
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      const auto index = token * 8 + slot;
      result.score_words[index] =
          omlx::m5::float_to_bf16(result.scores[index] / total);
      result.scores[index] = omlx::m5::bf16_to_float(result.score_words[index]);
    }
  }
  return result;
}

nlohmann::json inspect_dense(const omlx::m5::ArtifactModel &artifact,
                             const omlx::m5::QwenModelConfig &config,
                             const std::filesystem::path &mlp_metallib,
                             const std::filesystem::path &a4_metallib,
                             const std::filesystem::path &qmv_metallib) {
  const auto &gate_binding = binding_by_use(artifact, "mlp_gate");
  const auto prefix = remove_suffix(gate_binding.name, ".gate_proj");
  const auto &up_binding = binding_by_name(artifact, prefix + ".up_proj");
  const auto &down_binding = binding_by_name(artifact, prefix + ".down_proj");
  const bool allow_a4 = down_binding.activation_policy == "AllowA4";
  auto context = omlx::m5::MetalContext::create();
  auto anchors = run_swiglu_anchors(context, config.intermediate_size, allow_a4,
                                    mlp_metallib, a4_metallib);
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  auto gate_weight = upload_weight(
      artifact, artifact.object(gate_binding.object_id), context, buffers);
  auto up_weight = upload_weight(
      artifact, artifact.object(up_binding.object_id), context, buffers);
  auto down_weight = upload_weight(
      artifact, artifact.object(down_binding.object_id), context, buffers);
  auto bundle = omlx::m5::make_dense_mlp_projection_bundle(config, prefix);
  omlx::m5::validate_projection_parent(bundle.parents[0], gate_binding,
                                       gate_weight);
  omlx::m5::validate_projection_parent(bundle.parents[1], up_binding,
                                       up_weight);
  omlx::m5::ProjectionOutputArena arena(context, std::move(bundle), 1);
  const auto gate_output = arena.parent("mlp_gate");
  const auto up_output = arena.parent("mlp_up");
  const auto input_words =
      deterministic_words(1, config.hidden_size, 19, 23, 256.0F);
  auto input = context.upload(bytes(input_words));
  auto gate_plan =
      omlx::m5::prepare_dense_qmv_t1(context, qmv_metallib, gate_weight);
  auto up_plan =
      omlx::m5::prepare_dense_qmv_t1(context, qmv_metallib, up_weight);
  auto down_plan =
      omlx::m5::prepare_dense_qmv_t1(context, qmv_metallib, down_weight);
  const auto gate_submission = omlx::m5::execute_dense_qmv_t1(
      context, gate_plan, input, 0, arena.buffer(), gate_output.offset);
  const auto up_submission = omlx::m5::execute_dense_qmv_t1(
      context, up_plan, input, 0, arena.buffer(), up_output.offset);
  auto activated = context.make_buffer(config.intermediate_size * 2ULL);
  auto codes = context.make_buffer(config.intermediate_size / 2ULL);
  auto scales = context.make_buffer(config.intermediate_size / 32ULL);
  omlx::m5::SwiGluActivationView activation{
      .a16 =
          omlx::m5::ActivationPlaneView{
              .buffer = &activated,
              .bytes =
                  static_cast<std::uint64_t>(config.intermediate_size) * 2},
      .rows = 1,
      .features = static_cast<std::uint64_t>(config.intermediate_size),
  };
  if (allow_a4) {
    activation.a4 = omlx::m5::Mxfp4ActivationView{
        .codes = {.buffer = &codes,
                  .bytes =
                      static_cast<std::uint64_t>(config.intermediate_size) / 2},
        .scales = {.buffer = &scales,
                   .bytes =
                       static_cast<std::uint64_t>(config.intermediate_size) /
                       32},
        .group_size = 32,
    };
  }
  auto swiglu_plan = omlx::m5::prepare_swiglu(context, mlp_metallib, 1,
                                              config.intermediate_size);
  const auto swiglu_submission = omlx::m5::produce_swiglu_activation(
      context, swiglu_plan, gate_output, up_output, activation);
  auto output = context.make_buffer(down_plan.info().output_bytes);
  const auto down_submission = omlx::m5::execute_dense_qmv_t1(
      context, down_plan, activated, 0, output, 0);
  const auto gate_words =
      read_words(arena.buffer(), gate_output.offset, config.intermediate_size);
  const auto up_words =
      read_words(arena.buffer(), up_output.offset, config.intermediate_size);
  const auto activation_words =
      read_words(activated, 0, config.intermediate_size);
  double swiglu_max_relative = 0.0;
  for (std::int32_t feature = 0; feature < config.intermediate_size;
       ++feature) {
    const auto gate = omlx::m5::bf16_to_float(gate_words[feature]);
    const auto up = omlx::m5::bf16_to_float(up_words[feature]);
    const auto expected = gate / (1.0F + std::exp(-gate)) * up;
    swiglu_max_relative = std::max(
        swiglu_max_relative,
        relative_error(omlx::m5::bf16_to_float(activation_words[feature]),
                       expected));
  }
  nlohmann::json projection_samples = nlohmann::json::array();
  for (const auto feature :
       {0, config.intermediate_size / 2, config.intermediate_size - 1}) {
    projection_samples.push_back({
        {"feature", feature},
        {"gate", omlx::m5::bf16_to_float(gate_words[feature])},
        {"up", omlx::m5::bf16_to_float(up_words[feature])},
        {"activation", omlx::m5::bf16_to_float(activation_words[feature])},
    });
  }
  const auto output_words = read_words(output, 0, config.hidden_size);
  nlohmann::json output_samples = nlohmann::json::array();
  for (const auto row : {0, config.hidden_size / 2, config.hidden_size - 1}) {
    output_samples.push_back(
        {{"row", row}, {"actual", omlx::m5::bf16_to_float(output_words[row])}});
  }
  return {
      {"kind", "dense"},
      {"prefix", prefix},
      {"gate_binding", gate_binding.name},
      {"up_binding", up_binding.name},
      {"down_binding", down_binding.name},
      {"formats",
       {gate_binding.format, up_binding.format, down_binding.format}},
      {"allow_a4", allow_a4},
      {"input", floats(input_words)},
      {"activation", floats(activation_words)},
      {"projection_samples", std::move(projection_samples)},
      {"output_samples", std::move(output_samples)},
      {"swiglu_max_relative_error", swiglu_max_relative},
      {"dispatches",
       {{"gate", gate_submission.dispatch_count},
        {"up", up_submission.dispatch_count},
        {"swiglu", swiglu_submission.dispatch_count},
        {"down", down_submission.dispatch_count}}},
      {"gpu_ms",
       {{"gate", gate_submission.timing.gpu_ms},
        {"up", up_submission.timing.gpu_ms},
        {"swiglu", swiglu_submission.timing.gpu_ms},
        {"down", down_submission.timing.gpu_ms}}},
      {"anchors", std::move(anchors)},
  };
}

nlohmann::json inspect_moe(const omlx::m5::ArtifactModel &artifact,
                           const omlx::m5::QwenModelConfig &config,
                           const std::filesystem::path &mlp_metallib,
                           const std::filesystem::path &a4_metallib,
                           const std::filesystem::path &qmv_metallib,
                           const std::filesystem::path &moe_metallib) {
  const auto &gate_binding = binding_by_use(artifact, "moe_expert_gate");
  const auto prefix = remove_suffix(gate_binding.name, ".gate_proj");
  const auto &up_binding = binding_by_name(artifact, prefix + ".up_proj");
  const auto &down_binding = binding_by_name(artifact, prefix + ".down_proj");
  const auto &shared_gate_binding = binding_by_use(artifact, "moe_shared_gate");
  const auto &shared_mlp_gate_binding =
      binding_by_use(artifact, "moe_shared_expert_gate");
  const auto shared_prefix =
      remove_suffix(shared_mlp_gate_binding.name, ".gate_proj");
  const auto &shared_mlp_up_binding =
      binding_by_name(artifact, shared_prefix + ".up_proj");
  const auto &shared_mlp_down_binding =
      binding_by_name(artifact, shared_prefix + ".down_proj");
  const bool allow_a4 = down_binding.activation_policy == "AllowA4";
  auto context = omlx::m5::MetalContext::create();
  auto anchors = run_swiglu_anchors(context, config.intermediate_size, allow_a4,
                                    mlp_metallib, a4_metallib, 8);
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  auto gate_weight = upload_weight(
      artifact, artifact.object(gate_binding.object_id), context, buffers);
  auto up_weight = upload_weight(
      artifact, artifact.object(up_binding.object_id), context, buffers);
  auto down_weight = upload_weight(
      artifact, artifact.object(down_binding.object_id), context, buffers);
  auto shared_gate_weight =
      upload_weight(artifact, artifact.object(shared_gate_binding.object_id),
                    context, buffers);
  auto shared_mlp_gate_weight = upload_weight(
      artifact, artifact.object(shared_mlp_gate_binding.object_id), context,
      buffers);
  auto shared_mlp_up_weight =
      upload_weight(artifact, artifact.object(shared_mlp_up_binding.object_id),
                    context, buffers);
  auto shared_mlp_down_weight = upload_weight(
      artifact, artifact.object(shared_mlp_down_binding.object_id), context,
      buffers);
  constexpr std::uint32_t tokens = 1;
  constexpr std::uint32_t top_k = 8;
  const auto routes = make_routes(tokens);
  auto route_indices = context.upload(bytes(routes.indices));
  auto route_scores = context.upload(bytes(routes.score_words));
  const omlx::m5::MoeRouteView route{
      .indices = {.buffer = &route_indices, .bytes = top_k * 4},
      .scores = {.buffer = &route_scores, .bytes = top_k * 2},
      .tokens = tokens,
  };
  const auto input_words =
      deterministic_words(tokens, config.hidden_size, 19, 23, 256.0F);
  auto input = context.upload(bytes(input_words));
  auto gate_plan =
      omlx::m5::prepare_moe_expert(context, moe_metallib, gate_weight, tokens,
                                   omlx::m5::KernelFusedOp::moe_gate_up, false);
  auto up_plan =
      omlx::m5::prepare_moe_expert(context, moe_metallib, up_weight, tokens,
                                   omlx::m5::KernelFusedOp::moe_gate_up, false);
  auto down_plan =
      omlx::m5::prepare_moe_expert(context, moe_metallib, down_weight, tokens,
                                   omlx::m5::KernelFusedOp::moe_down, true);
  auto gate_output = context.make_buffer(gate_plan.info().output_bytes);
  auto up_output = context.make_buffer(up_plan.info().output_bytes);
  const auto gate_submission = omlx::m5::execute_moe_expert_a16(
      context, gate_plan,
      moe_executor(gate_plan, omlx::m5::MoeExpertFamily::grouped_a16), input, 0,
      route, gate_output, 0);
  const auto up_submission = omlx::m5::execute_moe_expert_a16(
      context, up_plan,
      moe_executor(up_plan, omlx::m5::MoeExpertFamily::grouped_a16), input, 0,
      route, up_output, 0);
  const omlx::m5::ProjectionParentOutputView gate_view{
      .role = "moe_gate",
      .buffer = &gate_output,
      .bytes = gate_plan.info().output_bytes,
      .tokens = top_k,
      .features = static_cast<std::uint64_t>(config.intermediate_size),
      .token_stride_bytes =
          static_cast<std::uint64_t>(config.intermediate_size) * 2,
  };
  const omlx::m5::ProjectionParentOutputView up_view{
      .role = "moe_up",
      .buffer = &up_output,
      .bytes = up_plan.info().output_bytes,
      .tokens = top_k,
      .features = static_cast<std::uint64_t>(config.intermediate_size),
      .token_stride_bytes =
          static_cast<std::uint64_t>(config.intermediate_size) * 2,
  };
  const auto activation_values =
      static_cast<std::uint64_t>(top_k) * config.intermediate_size;
  auto activated = context.make_buffer(activation_values * 2);
  auto codes = context.make_buffer(activation_values / 2);
  auto scales = context.make_buffer(activation_values / 32);
  omlx::m5::SwiGluActivationView activation{
      .a16 = omlx::m5::ActivationPlaneView{.buffer = &activated,
                                           .bytes = activation_values * 2},
      .rows = top_k,
      .features = static_cast<std::uint64_t>(config.intermediate_size),
  };
  if (allow_a4) {
    activation.a4 = omlx::m5::Mxfp4ActivationView{
        .codes = {.buffer = &codes, .bytes = activation_values / 2},
        .scales = {.buffer = &scales, .bytes = activation_values / 32},
        .group_size = 32,
    };
  }
  auto swiglu_plan = omlx::m5::prepare_swiglu(context, mlp_metallib, top_k,
                                              config.intermediate_size);
  const auto swiglu_submission = omlx::m5::produce_swiglu_activation(
      context, swiglu_plan, gate_view, up_view, activation);
  auto assignment_output = context.make_buffer(down_plan.info().output_bytes);
  omlx::m5::MetalSubmission down_submission;
  if (allow_a4) {
    down_submission = omlx::m5::execute_moe_expert_a4(
        context, down_plan,
        moe_executor(down_plan, omlx::m5::MoeExpertFamily::grouped_a4),
        *activation.a4, route, assignment_output, 0);
  } else {
    down_submission = omlx::m5::execute_moe_expert_a16(
        context, down_plan,
        moe_executor(down_plan, omlx::m5::MoeExpertFamily::grouped_a16),
        activated, 0, route, assignment_output, 0);
  }
  auto reduced = context.make_buffer(config.hidden_size * 2ULL);
  auto reduce_plan = omlx::m5::prepare_moe_topk_reduce(
      context, mlp_metallib, tokens, config.hidden_size, top_k);
  const auto reduce_submission = omlx::m5::execute_moe_topk_reduce(
      context, reduce_plan, assignment_output, 0, reduced, 0);

  auto shared_gate_plan =
      omlx::m5::prepare_dense_qmv_t1(context, qmv_metallib, shared_gate_weight);
  auto shared_mlp_gate_plan = omlx::m5::prepare_dense_qmv_t1(
      context, qmv_metallib, shared_mlp_gate_weight);
  auto shared_mlp_up_plan = omlx::m5::prepare_dense_qmv_t1(
      context, qmv_metallib, shared_mlp_up_weight);
  auto shared_mlp_down_plan = omlx::m5::prepare_dense_qmv_t1(
      context, qmv_metallib, shared_mlp_down_weight);
  auto shared_gate_output =
      context.make_buffer(shared_gate_plan.info().output_bytes);
  auto shared_mlp_gate_output =
      context.make_buffer(shared_mlp_gate_plan.info().output_bytes);
  auto shared_mlp_up_output =
      context.make_buffer(shared_mlp_up_plan.info().output_bytes);
  const auto shared_gate_submission = omlx::m5::execute_dense_qmv_t1(
      context, shared_gate_plan, input, 0, shared_gate_output, 0);
  const auto shared_mlp_gate_submission = omlx::m5::execute_dense_qmv_t1(
      context, shared_mlp_gate_plan, input, 0, shared_mlp_gate_output, 0);
  const auto shared_mlp_up_submission = omlx::m5::execute_dense_qmv_t1(
      context, shared_mlp_up_plan, input, 0, shared_mlp_up_output, 0);
  const auto shared_intermediate =
      static_cast<std::uint64_t>(config.shared_expert_intermediate_size);
  const omlx::m5::ProjectionParentOutputView shared_mlp_gate_view{
      .role = "moe_shared_gate_proj",
      .buffer = &shared_mlp_gate_output,
      .bytes = shared_mlp_gate_plan.info().output_bytes,
      .tokens = 1,
      .features = shared_intermediate,
      .token_stride_bytes = shared_intermediate * 2,
  };
  const omlx::m5::ProjectionParentOutputView shared_mlp_up_view{
      .role = "moe_shared_up_proj",
      .buffer = &shared_mlp_up_output,
      .bytes = shared_mlp_up_plan.info().output_bytes,
      .tokens = 1,
      .features = shared_intermediate,
      .token_stride_bytes = shared_intermediate * 2,
  };
  auto shared_activation = context.make_buffer(shared_intermediate * 2);
  const omlx::m5::SwiGluActivationView shared_activation_view{
      .a16 = omlx::m5::ActivationPlaneView{.buffer = &shared_activation,
                                           .bytes = shared_intermediate * 2},
      .rows = 1,
      .features = shared_intermediate,
  };
  auto shared_swiglu_plan =
      omlx::m5::prepare_swiglu(context, mlp_metallib, 1, shared_intermediate);
  const auto shared_swiglu_submission = omlx::m5::produce_swiglu_activation(
      context, shared_swiglu_plan, shared_mlp_gate_view, shared_mlp_up_view,
      shared_activation_view);
  auto shared_down =
      context.make_buffer(shared_mlp_down_plan.info().output_bytes);
  const auto shared_down_submission = omlx::m5::execute_dense_qmv_t1(
      context, shared_mlp_down_plan, shared_activation, 0, shared_down, 0);
  auto final_output = context.make_buffer(config.hidden_size * 2ULL);
  auto combine_plan = omlx::m5::prepare_moe_shared_combine(
      context, mlp_metallib, tokens, config.hidden_size);
  const auto combine_submission = omlx::m5::execute_moe_shared_combine(
      context, combine_plan, reduced, 0, shared_down, 0, shared_gate_output, 0,
      final_output, 0);
  const auto gate_words =
      read_words(gate_output, 0, top_k * config.intermediate_size);
  const auto up_words =
      read_words(up_output, 0, top_k * config.intermediate_size);
  const auto activation_words =
      read_words(activated, 0, top_k * config.intermediate_size);
  const auto shared_gate_word = read_words(shared_gate_output, 0, 1).front();
  const auto shared_mlp_gate_words =
      read_words(shared_mlp_gate_output, 0, shared_intermediate);
  const auto shared_mlp_up_words =
      read_words(shared_mlp_up_output, 0, shared_intermediate);
  const auto shared_activation_words =
      read_words(shared_activation, 0, shared_intermediate);
  double swiglu_max_relative = 0.0;
  for (std::size_t index = 0; index < activation_words.size(); ++index) {
    const auto gate = omlx::m5::bf16_to_float(gate_words[index]);
    const auto up = omlx::m5::bf16_to_float(up_words[index]);
    const auto expected = gate / (1.0F + std::exp(-gate)) * up;
    swiglu_max_relative = std::max(
        swiglu_max_relative,
        relative_error(omlx::m5::bf16_to_float(activation_words[index]),
                       expected));
  }
  double shared_swiglu_max_relative = 0.0;
  for (std::size_t index = 0; index < shared_activation_words.size(); ++index) {
    const auto gate = omlx::m5::bf16_to_float(shared_mlp_gate_words[index]);
    const auto up = omlx::m5::bf16_to_float(shared_mlp_up_words[index]);
    const auto expected = gate / (1.0F + std::exp(-gate)) * up;
    shared_swiglu_max_relative = std::max(
        shared_swiglu_max_relative,
        relative_error(omlx::m5::bf16_to_float(shared_activation_words[index]),
                       expected));
  }
  const auto reduced_words = read_words(reduced, 0, config.hidden_size);
  const auto shared_down_words = read_words(shared_down, 0, config.hidden_size);
  const auto final_words = read_words(final_output, 0, config.hidden_size);
  nlohmann::json projection_samples = nlohmann::json::array();
  for (const auto assignment : {0U, 4U, 7U}) {
    for (const auto feature :
         {0, config.intermediate_size / 2, config.intermediate_size - 1}) {
      const auto index = assignment * config.intermediate_size + feature;
      projection_samples.push_back({
          {"assignment", assignment},
          {"expert", routes.indices[assignment]},
          {"feature", feature},
          {"gate", omlx::m5::bf16_to_float(gate_words[index])},
          {"up", omlx::m5::bf16_to_float(up_words[index])},
          {"activation", omlx::m5::bf16_to_float(activation_words[index])},
      });
    }
  }
  nlohmann::json output_samples = nlohmann::json::array();
  for (const auto row : {0, config.hidden_size / 2, config.hidden_size - 1}) {
    output_samples.push_back({
        {"row", row},
        {"routed", omlx::m5::bf16_to_float(reduced_words[row])},
        {"shared", omlx::m5::bf16_to_float(shared_down_words[row])},
        {"actual", omlx::m5::bf16_to_float(final_words[row])},
    });
  }
  nlohmann::json shared_projection_samples = nlohmann::json::array();
  for (const auto feature : {0, config.shared_expert_intermediate_size / 2,
                             config.shared_expert_intermediate_size - 1}) {
    shared_projection_samples.push_back({
        {"feature", feature},
        {"gate", omlx::m5::bf16_to_float(shared_mlp_gate_words[feature])},
        {"up", omlx::m5::bf16_to_float(shared_mlp_up_words[feature])},
        {"activation",
         omlx::m5::bf16_to_float(shared_activation_words[feature])},
    });
  }
  return {
      {"kind", "moe"},
      {"prefix", prefix},
      {"gate_binding", gate_binding.name},
      {"up_binding", up_binding.name},
      {"down_binding", down_binding.name},
      {"shared_gate_binding", shared_gate_binding.name},
      {"shared_mlp_gate_binding", shared_mlp_gate_binding.name},
      {"shared_mlp_up_binding", shared_mlp_up_binding.name},
      {"shared_mlp_down_binding", shared_mlp_down_binding.name},
      {"formats",
       {gate_binding.format, up_binding.format, down_binding.format}},
      {"allow_a4", allow_a4},
      {"down_profile", allow_a4 ? "A4" : "A16"},
      {"input", floats(input_words)},
      {"route_indices", routes.indices},
      {"route_scores", routes.scores},
      {"activation", floats(activation_words)},
      {"shared_gate", omlx::m5::bf16_to_float(shared_gate_word)},
      {"shared_activation", floats(shared_activation_words)},
      {"projection_samples", std::move(projection_samples)},
      {"shared_projection_samples", std::move(shared_projection_samples)},
      {"output_samples", std::move(output_samples)},
      {"swiglu_max_relative_error", swiglu_max_relative},
      {"shared_swiglu_max_relative_error", shared_swiglu_max_relative},
      {"dispatches",
       {{"gate", gate_submission.dispatch_count},
        {"up", up_submission.dispatch_count},
        {"swiglu", swiglu_submission.dispatch_count},
        {"down", down_submission.dispatch_count},
        {"reduce", reduce_submission.dispatch_count},
        {"shared_gate", shared_gate_submission.dispatch_count},
        {"shared_mlp_gate", shared_mlp_gate_submission.dispatch_count},
        {"shared_mlp_up", shared_mlp_up_submission.dispatch_count},
        {"shared_swiglu", shared_swiglu_submission.dispatch_count},
        {"shared_down", shared_down_submission.dispatch_count},
        {"combine", combine_submission.dispatch_count}}},
      {"gpu_ms",
       {{"gate", gate_submission.timing.gpu_ms},
        {"up", up_submission.timing.gpu_ms},
        {"swiglu", swiglu_submission.timing.gpu_ms},
        {"down", down_submission.timing.gpu_ms},
        {"reduce", reduce_submission.timing.gpu_ms},
        {"shared_gate", shared_gate_submission.timing.gpu_ms},
        {"shared_mlp_gate", shared_mlp_gate_submission.timing.gpu_ms},
        {"shared_mlp_up", shared_mlp_up_submission.timing.gpu_ms},
        {"shared_swiglu", shared_swiglu_submission.timing.gpu_ms},
        {"shared_down", shared_down_submission.timing.gpu_ms},
        {"combine", combine_submission.timing.gpu_ms}}},
      {"anchors", std::move(anchors)},
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 7) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_mlp ARTIFACT CONFIG MLP_METALLIB "
          "A4_METALLIB QMV_METALLIB MOE_METALLIB");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    const bool moe = std::ranges::any_of(
        artifact.bindings(), [](const omlx::m5::ArtifactBinding &binding) {
          return binding.use == "moe_expert_gate";
        });
    auto layer =
        moe ? inspect_moe(artifact, config, argv[3], argv[4], argv[5], argv[6])
            : inspect_dense(artifact, config, argv[3], argv[4], argv[5]);
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-5-mlp-v1"},
        {"evidence_scope", "M4 real Dense/MoE MLP layer-slice correctness"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"hidden_size", config.hidden_size},
        {"intermediate_size", config.intermediate_size},
        {"layer", std::move(layer)},
    };
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MLP inspection failed: " << error.what() << '\n';
    return 1;
  }
}
