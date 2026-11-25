#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/gdn.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/model_config.h"
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
    throw std::invalid_argument("GDN binding suffix mismatch");
  }
  return std::string(value.substr(0, value.size() - suffix.size()));
}

const omlx::m5::ArtifactBinding &
binding_by_name(const omlx::m5::ArtifactModel &artifact,
                std::string_view name) {
  const auto found = std::ranges::find_if(
      artifact.bindings(), [&](const omlx::m5::ArtifactBinding &binding) {
        return binding.name == name;
      });
  if (found == artifact.bindings().end()) {
    throw std::out_of_range("GDN binding is absent: " + std::string(name));
  }
  return *found;
}

const omlx::m5::ArtifactObject &
module_object(const omlx::m5::ArtifactModel &artifact, std::string_view name) {
  try {
    return artifact.object(name);
  } catch (const std::out_of_range &) {
    return artifact.object(std::string(name) + ".weight");
  }
}

omlx::m5::MaterializedWeightView
upload_weight(const omlx::m5::ArtifactModel &artifact,
              const omlx::m5::ArtifactObject &object,
              omlx::m5::MetalContext &context,
              std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> &buffers) {
  omlx::m5::MaterializedWeightView weight{
      .object_id = object.id,
      .component = omlx::m5::classify_model_component(object.name),
      .name = object.name,
      .format = object.format,
      .layout = object.layout,
      .parent_shape = object.shape,
      .padded_k = object.padded_k,
      .row_count = object.shape[0],
      .in_features = object.shape[1],
  };
  for (const auto &selected_plane : object.planes) {
    buffers.push_back(std::make_unique<omlx::m5::MetalBuffer>(
        context.upload(artifact.plane_bytes(object, selected_plane))));
    weight.planes.push_back({
        .kind = selected_plane.kind,
        .buffer = buffers.back().get(),
        .segment_index = static_cast<std::uint32_t>(buffers.size() - 1),
        .bytes = selected_plane.bytes,
        .row_stride = selected_plane.bytes / object.shape[0],
    });
  }
  return weight;
}

omlx::m5::MetalBuffer upload_direct(const omlx::m5::ArtifactModel &artifact,
                                    const omlx::m5::ArtifactObject &object,
                                    omlx::m5::MetalContext &context) {
  if (object.format != "direct_bf16_v1" || object.planes.size() != 1) {
    throw std::invalid_argument("GDN direct object is invalid");
  }
  return context.upload(artifact.plane_bytes(object, object.planes.front()));
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

struct DirectBuffers {
  omlx::m5::MetalBuffer conv;
  omlx::m5::MetalBuffer a_log;
  omlx::m5::MetalBuffer dt_bias;
  omlx::m5::MetalBuffer norm;
  omlx::m5::GdnDirectWeights view;
};

DirectBuffers direct_buffers(const omlx::m5::ArtifactModel &artifact,
                             std::string_view prefix,
                             omlx::m5::MetalContext &context) {
  const auto &conv_object =
      artifact.object(std::string(prefix) + ".conv1d.weight");
  const auto &a_log_object = artifact.object(std::string(prefix) + ".A_log");
  const auto &dt_bias_object =
      artifact.object(std::string(prefix) + ".dt_bias");
  const auto &norm_object =
      artifact.object(std::string(prefix) + ".norm.weight");
  DirectBuffers result{
      .conv = upload_direct(artifact, conv_object, context),
      .a_log = upload_direct(artifact, a_log_object, context),
      .dt_bias = upload_direct(artifact, dt_bias_object, context),
      .norm = upload_direct(artifact, norm_object, context),
  };
  result.view = {
      .conv = &result.conv,
      .conv_bytes = result.conv.size(),
      .a_log = &result.a_log,
      .a_log_bytes = result.a_log.size(),
      .dt_bias = &result.dt_bias,
      .dt_bias_bytes = result.dt_bias.size(),
      .norm = &result.norm,
      .norm_bytes = result.norm.size(),
  };
  return result;
}

nlohmann::json execute_anchor(omlx::m5::MetalContext &context,
                              const omlx::m5::QwenModelConfig &config,
                              const omlx::m5::GdnDirectWeights &weights,
                              const std::filesystem::path &metallib,
                              std::uint32_t tokens, bool execute) {
  const auto key_width =
      config.linear_num_key_heads * config.linear_key_head_dim;
  const auto value_width =
      config.linear_num_value_heads * config.linear_value_head_dim;
  const auto conv_width = key_width * 2 + value_width;
  omlx::m5::GdnLayerState state(context, config);
  auto plan = omlx::m5::prepare_gdn_layer(context, metallib, config, tokens);
  nlohmann::json record = {
      {"tokens", tokens},
      {"planned", true},
      {"executed", false},
      {"dispatches", plan.info().dispatch_count},
      {"workspace_bytes", plan.info().workspace_bytes},
      {"state_bytes", state.info().recurrent_state_bytes},
      {"history_bytes", state.info().conv_history_bytes},
  };
  if (!execute) {
    return record;
  }
  const auto qkv_words =
      deterministic_words(tokens, conv_width, 11, 17, 256.0F);
  const auto z_words = deterministic_words(tokens, value_width, 13, 19, 256.0F);
  const auto a_words =
      deterministic_words(tokens, config.linear_num_value_heads, 3, 5, 512.0F);
  const auto b_words =
      deterministic_words(tokens, config.linear_num_value_heads, 7, 11, 512.0F);
  auto qkv = context.upload(bytes(qkv_words));
  auto z = context.upload(bytes(z_words));
  auto a = context.upload(bytes(a_words));
  auto b = context.upload(bytes(b_words));
  const omlx::m5::ProjectionParentOutputView qkv_view{
      .role = "gdn_qkv",
      .buffer = &qkv,
      .bytes = static_cast<std::uint64_t>(tokens) * conv_width * 2,
      .tokens = tokens,
      .features = static_cast<std::uint64_t>(conv_width),
      .token_stride_bytes = static_cast<std::uint64_t>(conv_width) * 2,
  };
  const omlx::m5::ProjectionParentOutputView z_view{
      .role = "gdn_z",
      .buffer = &z,
      .bytes = static_cast<std::uint64_t>(tokens) * value_width * 2,
      .tokens = tokens,
      .features = static_cast<std::uint64_t>(value_width),
      .token_stride_bytes = static_cast<std::uint64_t>(value_width) * 2,
  };
  auto output = context.make_buffer(plan.info().output_bytes);
  const auto before = context.registry_stats();
  const auto submission = omlx::m5::execute_gdn_layer(
      context, plan, qkv_view, z_view, a, 0, b, 0, weights, state, output, 0);
  const auto after = context.registry_stats();
  const std::array sample_tokens = {0U, tokens / 2, tokens - 1};
  const std::array sample_features = {
      0U, static_cast<std::uint32_t>(value_width / 2),
      static_cast<std::uint32_t>(value_width - 1)};
  nlohmann::json samples = nlohmann::json::array();
  for (const auto token : sample_tokens) {
    for (const auto feature : sample_features) {
      const auto index =
          static_cast<std::uint64_t>(token) * value_width + feature;
      const auto word = read_words(output, index * 2, 1).front();
      samples.push_back({
          {"token", token},
          {"feature", feature},
          {"actual", omlx::m5::bf16_to_float(word)},
      });
    }
  }
  record["executed"] = true;
  record["dispatches"] = submission.dispatch_count;
  record["gpu_ms"] = submission.timing.gpu_ms;
  record["wall_ms"] = submission.timing.wall_ms;
  record["token_loop_uploads"] = after.upload_count - before.upload_count;
  record["token_loop_pipeline_compiles"] =
      after.pipelines_compiled - before.pipelines_compiled;
  record["samples"] = std::move(samples);
  if (const auto iterations = benchmark_iterations()) {
    state.reset();
    static_cast<void>(omlx::m5::execute_gdn_layer(context, plan, qkv_view,
                                                  z_view, a, 0, b, 0, weights,
                                                  state, output, 0));
    nlohmann::json timings = nlohmann::json::array();
    for (std::uint32_t i = 0; i < iterations; ++i) {
      state.reset();
      timings.push_back(omlx::m5::execute_gdn_layer(context, plan, qkv_view,
                                                    z_view, a, 0, b, 0, weights,
                                                    state, output, 0)
                            .timing.gpu_ms);
    }
    record["gpu_ms_samples"] = std::move(timings);
  }
  return record;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_gdn ARTIFACT CONFIG GDN_METALLIB "
          "QMV_METALLIB");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    const auto qkv_binding = std::ranges::find_if(
        artifact.bindings(), [](const omlx::m5::ArtifactBinding &binding) {
          return binding.use == "gdn_qkv";
        });
    if (qkv_binding == artifact.bindings().end()) {
      throw std::runtime_error("artifact has no GDN layer");
    }
    const auto prefix = remove_suffix(qkv_binding->name, ".in_proj_qkv");
    const auto &z_binding = binding_by_name(artifact, prefix + ".in_proj_z");
    const auto &out_binding = binding_by_name(artifact, prefix + ".out_proj");
    const auto &a_object = module_object(artifact, prefix + ".in_proj_a");
    const auto &b_object = module_object(artifact, prefix + ".in_proj_b");
    auto context = omlx::m5::MetalContext::create();
    auto direct = direct_buffers(artifact, prefix, context);
    direct.view.conv = &direct.conv;
    direct.view.a_log = &direct.a_log;
    direct.view.dt_bias = &direct.dt_bias;
    direct.view.norm = &direct.norm;
    nlohmann::json anchors = nlohmann::json::array();
    for (const auto tokens : kAnchors) {
      anchors.push_back(
          execute_anchor(context, config, direct.view, argv[3], tokens, true));
    }

    std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> weight_buffers;
    auto qkv_weight =
        upload_weight(artifact, artifact.object(qkv_binding->object_id),
                      context, weight_buffers);
    auto z_weight =
        upload_weight(artifact, artifact.object(z_binding.object_id), context,
                      weight_buffers);
    auto a_weight = upload_weight(artifact, a_object, context, weight_buffers);
    auto b_weight = upload_weight(artifact, b_object, context, weight_buffers);
    auto out_weight =
        upload_weight(artifact, artifact.object(out_binding.object_id), context,
                      weight_buffers);
    auto qkv_plan =
        omlx::m5::prepare_dense_qmv_t1(context, argv[4], qkv_weight);
    auto z_plan = omlx::m5::prepare_dense_qmv_t1(context, argv[4], z_weight);
    auto a_plan = omlx::m5::prepare_dense_qmv_t1(context, argv[4], a_weight);
    auto b_plan = omlx::m5::prepare_dense_qmv_t1(context, argv[4], b_weight);
    auto out_plan =
        omlx::m5::prepare_dense_qmv_t1(context, argv[4], out_weight);
    auto bundle = omlx::m5::make_gdn_projection_bundle(config, prefix);
    omlx::m5::ProjectionOutputArena arena(context, std::move(bundle), 1);
    const auto qkv_output = arena.parent("gdn_qkv");
    const auto z_output = arena.parent("gdn_z");
    const auto input_words =
        deterministic_words(1, config.hidden_size, 19, 23, 256.0F);
    auto input = context.upload(bytes(input_words));
    static_cast<void>(omlx::m5::execute_dense_qmv_t1(
        context, qkv_plan, input, 0, arena.buffer(), qkv_output.offset));
    static_cast<void>(omlx::m5::execute_dense_qmv_t1(
        context, z_plan, input, 0, arena.buffer(), z_output.offset));
    auto a_output = context.make_buffer(a_plan.info().output_bytes);
    auto b_output = context.make_buffer(b_plan.info().output_bytes);
    static_cast<void>(
        omlx::m5::execute_dense_qmv_t1(context, a_plan, input, 0, a_output, 0));
    static_cast<void>(
        omlx::m5::execute_dense_qmv_t1(context, b_plan, input, 0, b_output, 0));
    omlx::m5::GdnLayerState state(context, config);
    auto gdn_plan = omlx::m5::prepare_gdn_layer(context, argv[3], config, 1);
    auto gdn_output = context.make_buffer(gdn_plan.info().output_bytes);
    const auto gdn_submission = omlx::m5::execute_gdn_layer(
        context, gdn_plan, qkv_output, z_output, a_output, 0, b_output, 0,
        direct.view, state, gdn_output, 0);
    auto projected = context.make_buffer(out_plan.info().output_bytes);
    const auto out_submission = omlx::m5::execute_dense_qmv_t1(
        context, out_plan, gdn_output, 0, projected, 0);
    const auto qkv_words =
        read_words(arena.buffer(), qkv_output.offset, qkv_output.features);
    const auto z_words =
        read_words(arena.buffer(), z_output.offset, z_output.features);
    const auto a_words = read_words(a_output, 0, config.linear_num_value_heads);
    const auto b_words = read_words(b_output, 0, config.linear_num_value_heads);
    const auto gdn_words = read_words(gdn_output, 0,
                                      config.linear_num_value_heads *
                                          config.linear_value_head_dim);
    nlohmann::json projection_samples = nlohmann::json::array();
    const auto add_projection_samples =
        [&](std::string_view name, std::span<const std::uint16_t> words) {
          for (const auto row :
               {0U, static_cast<std::uint32_t>(words.size() / 2),
                static_cast<std::uint32_t>(words.size() - 1)}) {
            projection_samples.push_back({
                {"name", name},
                {"row", row},
                {"actual", omlx::m5::bf16_to_float(words[row])},
            });
          }
        };
    add_projection_samples("qkv", qkv_words);
    add_projection_samples("z", z_words);
    add_projection_samples("a", a_words);
    add_projection_samples("b", b_words);
    nlohmann::json output_samples = nlohmann::json::array();
    for (const auto row : {0, config.hidden_size / 2, config.hidden_size - 1}) {
      const auto word = read_words(projected, row * 2ULL, 1).front();
      output_samples.push_back({
          {"row", row},
          {"actual", omlx::m5::bf16_to_float(word)},
      });
    }
    const nlohmann::json integration = {
        {"input", floats(input_words)},
        {"qkv_binding", qkv_binding->name},
        {"z_binding", z_binding.name},
        {"a_binding", a_object.name},
        {"b_binding", b_object.name},
        {"out_binding", out_binding.name},
        {"qkv", floats(qkv_words)},
        {"z", floats(z_words)},
        {"a", floats(a_words)},
        {"b", floats(b_words)},
        {"gdn", floats(gdn_words)},
        {"projection_samples", std::move(projection_samples)},
        {"output_samples", std::move(output_samples)},
        {"gdn_dispatches", gdn_submission.dispatch_count},
        {"gdn_gpu_ms", gdn_submission.timing.gpu_ms},
        {"gdn_wall_ms", gdn_submission.timing.wall_ms},
        {"out_dispatches", out_submission.dispatch_count},
        {"out_gpu_ms", out_submission.timing.gpu_ms},
        {"out_wall_ms", out_submission.timing.wall_ms},
    };
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-7-gdn-v1"},
        {"evidence_scope", "M4 GDN conv/recurrent/output correctness"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"prefix", prefix},
        {"direct_objects",
         {{"conv", prefix + ".conv1d.weight"},
          {"a_log", prefix + ".A_log"},
          {"dt_bias", prefix + ".dt_bias"},
          {"norm", prefix + ".norm.weight"}}},
        {"anchors", std::move(anchors)},
        {"integration", integration},
    };
    std::cout << result.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "GDN inspection failed: " << error.what() << '\n';
    return 1;
  }
}
