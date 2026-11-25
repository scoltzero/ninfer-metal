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
#include "omlx_m5/attention.h"
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
constexpr std::uint64_t kStartPosition = 37;

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::string remove_suffix(std::string_view value, std::string_view suffix) {
  if (!value.ends_with(suffix)) {
    throw std::invalid_argument("attention binding suffix mismatch");
  }
  return std::string(value.substr(0, value.size() - suffix.size()));
}

std::vector<std::uint16_t>
direct_words(const omlx::m5::ArtifactModel &artifact,
             const omlx::m5::ArtifactObject &object) {
  if (object.format != "direct_bf16_v1" || object.shape.size() != 1 ||
      object.planes.size() != 1) {
    throw std::invalid_argument("attention norm object is not direct BF16");
  }
  const auto raw = artifact.plane_bytes(object, object.planes.front());
  std::vector<std::uint16_t> result(raw.size() / 2);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

float read_bf16(const omlx::m5::MetalBuffer &buffer, std::uint64_t offset) {
  const auto raw = buffer.read(offset, 2);
  std::uint16_t word = 0;
  std::memcpy(&word, raw.data(), sizeof(word));
  return omlx::m5::bf16_to_float(word);
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

std::vector<float> reference_head(std::span<const float> input,
                                  std::span<const float> weight,
                                  std::uint32_t rotary_dim,
                                  std::uint64_t position, double theta,
                                  float epsilon) {
  double squares = 0.0;
  for (const auto value : input) {
    squares += static_cast<double>(value) * value;
  }
  const auto inverse = static_cast<float>(
      1.0 / std::sqrt(squares / static_cast<double>(input.size()) + epsilon));
  std::vector<float> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index] * inverse * weight[index];
  }
  const auto half = rotary_dim / 2;
  for (std::uint32_t pair = 0; pair < half; ++pair) {
    const auto exponent = static_cast<double>(2 * pair) / rotary_dim;
    const auto angle =
        static_cast<double>(position) / std::pow(theta, exponent);
    const auto cosine = static_cast<float>(std::cos(angle));
    const auto sine = static_cast<float>(std::sin(angle));
    const auto first = output[pair];
    const auto second = output[pair + half];
    output[pair] = first * cosine - second * sine;
    output[pair + half] = first * sine + second * cosine;
  }
  return output;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 5) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_attention ARTIFACT CONFIG ATTN_METALLIB "
          "QMV_METALLIB");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    const auto q_binding = std::ranges::find_if(
        artifact.bindings(), [](const omlx::m5::ArtifactBinding &binding) {
          return binding.use == "attention_q";
        });
    if (q_binding == artifact.bindings().end()) {
      throw std::runtime_error("artifact has no attention projection");
    }
    const auto prefix = remove_suffix(q_binding->name, ".q_proj");
    const auto q_norm_name = prefix + ".q_norm.weight";
    const auto k_norm_name = prefix + ".k_norm.weight";
    const auto &q_norm_object = artifact.object(q_norm_name);
    const auto &k_norm_object = artifact.object(k_norm_name);
    const auto q_norm_words = direct_words(artifact, q_norm_object);
    const auto k_norm_words = direct_words(artifact, k_norm_object);
    if (q_norm_words.size() != static_cast<std::size_t>(config.head_dim) ||
        k_norm_words.size() != static_cast<std::size_t>(config.head_dim)) {
      throw std::runtime_error("attention norm weight geometry mismatch");
    }
    std::vector<float> q_norm_values(q_norm_words.size());
    std::vector<float> k_norm_values(k_norm_words.size());
    std::ranges::transform(q_norm_words, q_norm_values.begin(),
                           omlx::m5::bf16_to_float);
    std::ranges::transform(k_norm_words, k_norm_values.begin(),
                           omlx::m5::bf16_to_float);
    auto context = omlx::m5::MetalContext::create();
    auto q_norm = context.upload(bytes(q_norm_words));
    auto k_norm = context.upload(bytes(k_norm_words));
    const auto o_binding = std::ranges::find_if(
        artifact.bindings(), [&](const omlx::m5::ArtifactBinding &binding) {
          return binding.name == prefix + ".o_proj";
        });
    if (o_binding == artifact.bindings().end()) {
      throw std::runtime_error("artifact has no attention output projection");
    }
    std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> output_weight_buffers;
    auto output_weight =
        upload_weight(artifact, artifact.object(o_binding->object_id), context,
                      output_weight_buffers);
    auto output_plan =
        omlx::m5::prepare_dense_qmv_t1(context, argv[4], output_weight);
    nlohmann::json records = nlohmann::json::array();
    nlohmann::json attention_records = nlohmann::json::array();
    std::optional<nlohmann::json> output_integration;
    for (const auto tokens : kAnchors) {
      auto bundle =
          omlx::m5::make_full_attention_projection_bundle(config, prefix);
      omlx::m5::ProjectionOutputArena arena(context, std::move(bundle), tokens);
      const auto q_parent = arena.parent("attention_q_gate");
      const auto k_parent = arena.parent("attention_k");
      const auto v_parent = arena.parent("attention_v");
      const auto query = arena.output("attention_query");
      const auto gate = arena.output("attention_gate");
      const auto key = arena.output("attention_key");
      const auto value = arena.output("attention_value");
      std::vector<std::uint16_t> query_parent_words(tokens * q_parent.features);
      std::vector<std::uint16_t> key_parent_words(tokens * k_parent.features);
      std::vector<std::uint16_t> value_parent_words(tokens * v_parent.features);
      for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint64_t feature = 0; feature < q_parent.features;
             ++feature) {
          const auto raw =
              static_cast<float>(
                  static_cast<std::int32_t>((feature * 19 + token * 23) % 257) -
                  128) /
              256.0F;
          query_parent_words[token * q_parent.features + feature] =
              omlx::m5::float_to_bf16(raw);
        }
        for (std::uint64_t feature = 0; feature < k_parent.features;
             ++feature) {
          const auto raw =
              static_cast<float>(
                  static_cast<std::int32_t>((feature * 29 + token * 31) % 257) -
                  128) /
              256.0F;
          key_parent_words[token * k_parent.features + feature] =
              omlx::m5::float_to_bf16(raw);
          const auto value_raw =
              static_cast<float>((feature * 37 + token * 41) % 251) / 256.0F;
          value_parent_words[token * v_parent.features + feature] =
              omlx::m5::float_to_bf16(value_raw);
        }
      }
      arena.buffer().write(bytes(query_parent_words), q_parent.offset);
      arena.buffer().write(bytes(key_parent_words), k_parent.offset);
      arena.buffer().write(bytes(value_parent_words), v_parent.offset);
      std::vector<std::uint16_t> gate_before(tokens * gate.features);
      for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint64_t feature = 0; feature < gate.features; ++feature) {
          const auto raw =
              arena.buffer().read(gate.feature_offset(token, feature), 2);
          std::memcpy(&gate_before[token * gate.features + feature], raw.data(),
                      2);
        }
      }
      auto plan = omlx::m5::prepare_attention_postprocess(context, argv[3],
                                                          config, tokens);
      const auto before = context.registry_stats();
      const auto submission = omlx::m5::execute_attention_postprocess(
          context, plan, query, key, q_norm, 0, k_norm, 0, kStartPosition);
      const auto after = context.registry_stats();
      std::uint64_t gate_mismatches = 0;
      for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint64_t feature = 0; feature < gate.features; ++feature) {
          std::uint16_t actual = 0;
          const auto raw =
              arena.buffer().read(gate.feature_offset(token, feature), 2);
          std::memcpy(&actual, raw.data(), 2);
          gate_mismatches +=
              actual != gate_before[token * gate.features + feature];
        }
      }
      double max_relative_error = 0.0;
      nlohmann::json samples = nlohmann::json::array();
      const std::array sample_tokens = {0U, tokens / 2, tokens - 1};
      const std::array<std::uint32_t, 2> q_heads = {
          0U, static_cast<std::uint32_t>(config.num_attention_heads - 1)};
      const std::array<std::uint32_t, 2> k_heads = {
          0U, static_cast<std::uint32_t>(config.num_key_value_heads - 1)};
      const std::array sample_features = {0U, plan.info().rotary_dim / 2,
                                          plan.info().rotary_dim - 1,
                                          plan.info().head_dim - 1};
      const auto inspect_kind =
          [&](std::string_view kind,
              const omlx::m5::ProjectionSplitOutputView &view,
              std::span<const std::uint32_t> heads,
              const std::vector<std::uint16_t> &parent_words,
              std::uint64_t parent_features, std::span<const float> norm_weight,
              bool interleaved) {
            for (const auto token : sample_tokens) {
              for (const auto head : heads) {
                std::vector<float> input(config.head_dim);
                for (std::uint32_t feature = 0;
                     feature < static_cast<std::uint32_t>(config.head_dim);
                     ++feature) {
                  const auto parent_feature =
                      interleaved ? head * config.head_dim * 2 + feature
                                  : head * config.head_dim + feature;
                  input[feature] = omlx::m5::bf16_to_float(
                      parent_words[token * parent_features + parent_feature]);
                }
                const auto expected =
                    reference_head(input, norm_weight, plan.info().rotary_dim,
                                   kStartPosition + token, plan.info().theta,
                                   plan.info().epsilon);
                for (const auto feature : sample_features) {
                  const auto logical = head * config.head_dim + feature;
                  const auto actual = read_bf16(
                      arena.buffer(), view.feature_offset(token, logical));
                  const auto relative =
                      std::abs(static_cast<double>(actual) -
                               expected[feature]) /
                      std::max(1.0, std::abs(static_cast<double>(
                                        expected[feature])));
                  max_relative_error = std::max(max_relative_error, relative);
                  samples.push_back({
                      {"kind", kind},
                      {"token", token},
                      {"head", head},
                      {"feature", feature},
                      {"actual", actual},
                      {"cpp_expected", expected[feature]},
                  });
                }
              }
            }
          };
      inspect_kind("query", query, q_heads, query_parent_words,
                   q_parent.features, q_norm_values, true);
      inspect_kind("key", key, k_heads, key_parent_words, k_parent.features,
                   k_norm_values, false);
      if (submission.dispatch_count != 1 || gate_mismatches != 0 ||
          max_relative_error > 0.02 ||
          after.upload_count != before.upload_count ||
          after.pipelines_compiled != before.pipelines_compiled) {
        throw std::runtime_error("attention postprocess validation failed");
      }
      std::vector<double> postprocess_gpu_ms;
      if (const auto iterations = benchmark_iterations()) {
        arena.buffer().write(bytes(query_parent_words), q_parent.offset);
        arena.buffer().write(bytes(key_parent_words), k_parent.offset);
        static_cast<void>(omlx::m5::execute_attention_postprocess(
            context, plan, query, key, q_norm, 0, k_norm, 0, kStartPosition));
        for (std::uint32_t i = 0; i < iterations; ++i) {
          arena.buffer().write(bytes(query_parent_words), q_parent.offset);
          arena.buffer().write(bytes(key_parent_words), k_parent.offset);
          postprocess_gpu_ms.push_back(omlx::m5::execute_attention_postprocess(
                                           context, plan, query, key, q_norm, 0,
                                           k_norm, 0, kStartPosition)
                                           .timing.gpu_ms);
        }
      }
      records.push_back({
          {"tokens", tokens},
          {"start_position", kStartPosition},
          {"dispatches", submission.dispatch_count},
          {"gpu_ms", submission.timing.gpu_ms},
          {"wall_ms", submission.timing.wall_ms},
          {"gpu_ms_samples", std::move(postprocess_gpu_ms)},
          {"workspace_bytes", plan.info().workspace_bytes},
          {"gate_mismatches", gate_mismatches},
          {"token_loop_uploads", after.upload_count - before.upload_count},
          {"token_loop_pipeline_compiles",
           after.pipelines_compiled - before.pipelines_compiled},
          {"max_relative_error", max_relative_error},
          {"samples", std::move(samples)},
      });

      omlx::m5::PagedKvCache cache(context, config.num_key_value_heads,
                                   config.head_dim,
                                   std::max<std::uint32_t>(tokens, 32), 5);
      auto paged_plan = omlx::m5::prepare_paged_attention(
          context, argv[3], config, tokens, cache.info());
      nlohmann::json attention_record = {
          {"tokens", tokens},
          {"planned", true},
          {"page_tokens", cache.info().page_tokens},
          {"page_count", cache.info().page_count},
          {"cache_bytes", cache.info().key_bytes + cache.info().value_bytes},
          {"dispatches", paged_plan.info().dispatch_count},
          {"workspace_bytes", paged_plan.info().workspace_bytes},
          {"executed", false},
      };
      if (tokens <= 13) {
        auto attended = context.make_buffer(paged_plan.info().output_bytes);
        const auto paged_before = context.registry_stats();
        const auto paged_submission = omlx::m5::execute_paged_attention(
            context, paged_plan, query, key, value, gate, cache, attended, 0);
        const auto paged_after = context.registry_stats();
        nlohmann::json attention_samples = nlohmann::json::array();
        const std::array paged_tokens = {0U, tokens / 2, tokens - 1};
        const std::array paged_heads = {
            0U, static_cast<std::uint32_t>(config.num_attention_heads - 1)};
        const std::array paged_features = {
            0U, static_cast<std::uint32_t>(config.head_dim / 2),
            static_cast<std::uint32_t>(config.head_dim - 1)};
        for (const auto token : paged_tokens) {
          for (const auto head : paged_heads) {
            for (const auto feature : paged_features) {
              const auto index = (static_cast<std::uint64_t>(token) *
                                      config.num_attention_heads +
                                  head) *
                                     config.head_dim +
                                 feature;
              attention_samples.push_back({
                  {"token", token},
                  {"head", head},
                  {"feature", feature},
                  {"actual", read_bf16(attended, index * 2)},
              });
            }
          }
        }
        attention_record["executed"] = true;
        attention_record["dispatches"] = paged_submission.dispatch_count;
        attention_record["gpu_ms"] = paged_submission.timing.gpu_ms;
        attention_record["wall_ms"] = paged_submission.timing.wall_ms;
        attention_record["cache_tokens"] = cache.tokens();
        attention_record["token_loop_uploads"] =
            paged_after.upload_count - paged_before.upload_count;
        attention_record["token_loop_pipeline_compiles"] =
            paged_after.pipelines_compiled - paged_before.pipelines_compiled;
        attention_record["samples"] = std::move(attention_samples);
        if (const auto iterations = benchmark_iterations()) {
          cache.reset();
          static_cast<void>(omlx::m5::execute_paged_attention(
              context, paged_plan, query, key, value, gate, cache, attended,
              0));
          nlohmann::json samples = nlohmann::json::array();
          for (std::uint32_t i = 0; i < iterations; ++i) {
            cache.reset();
            samples.push_back(omlx::m5::execute_paged_attention(
                                  context, paged_plan, query, key, value, gate,
                                  cache, attended, 0)
                                  .timing.gpu_ms);
          }
          attention_record["gpu_ms_samples"] = std::move(samples);
        }
        if (tokens == 1) {
          auto projected = context.make_buffer(output_plan.info().output_bytes);
          const auto output_submission = omlx::m5::execute_dense_qmv_t1(
              context, output_plan, attended, 0, projected, 0);
          const auto attended_words = read_words(
              attended, 0,
              static_cast<std::uint64_t>(config.num_attention_heads) *
                  config.head_dim);
          nlohmann::json projected_samples = nlohmann::json::array();
          for (const auto row :
               {0, config.hidden_size / 2, config.hidden_size - 1}) {
            projected_samples.push_back({
                {"row", row},
                {"actual", read_bf16(projected, row * 2ULL)},
            });
          }
          output_integration = {
              {"binding", o_binding->name},
              {"format", o_binding->format},
              {"attention", floats(attended_words)},
              {"dispatches", output_submission.dispatch_count},
              {"gpu_ms", output_submission.timing.gpu_ms},
              {"wall_ms", output_submission.timing.wall_ms},
              {"samples", std::move(projected_samples)},
          };
        }
      }
      attention_records.push_back(std::move(attention_record));
    }
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-4-attention-v1"},
        {"evidence_scope", "M4 q/k RMSNorm + half-split RoPE correctness"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"attention_prefix", prefix},
        {"q_norm_object", q_norm_name},
        {"k_norm_object", k_norm_name},
        {"query_heads", config.num_attention_heads},
        {"key_value_heads", config.num_key_value_heads},
        {"head_dim", config.head_dim},
        {"rotary_dim", static_cast<std::uint32_t>(std::llround(
                           config.head_dim * config.partial_rotary_factor))},
        {"rope_layout", "half_split_traditional_false"},
        {"records", std::move(records)},
        {"paged_attention_records", std::move(attention_records)},
        {"output_integration", std::move(*output_integration)},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "attention inspection failed: " << error.what() << '\n';
    return 1;
  }
}
