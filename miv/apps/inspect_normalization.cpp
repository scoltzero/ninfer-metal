#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
#include "omlx_m5/mxfp4_small_t.h"
#include "omlx_m5/normalization.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/reference.h"

namespace {

constexpr std::array kAnchors = {1U, 8U, 13U, 128U, 512U, 1024U};

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::vector<std::uint16_t> words(std::span<const std::byte> raw) {
  if (raw.size() % 2 != 0) {
    throw std::invalid_argument("BF16 byte span has odd length");
  }
  std::vector<std::uint16_t> result(raw.size() / 2);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

std::vector<float> floats(std::span<const std::uint16_t> raw) {
  std::vector<float> result(raw.size());
  std::ranges::transform(raw, result.begin(), omlx::m5::bf16_to_float);
  return result;
}

const omlx::m5::ArtifactBinding &
binding_by_use(const omlx::m5::ArtifactModel &artifact, std::string_view use) {
  const auto found = std::ranges::find_if(
      artifact.bindings(), [&](const omlx::m5::ArtifactBinding &binding) {
        return binding.use == use;
      });
  if (found == artifact.bindings().end()) {
    throw std::out_of_range("binding use is absent: " + std::string(use));
  }
  return *found;
}

const omlx::m5::ArtifactObject &
input_norm_object(const omlx::m5::ArtifactModel &artifact) {
  const auto found = std::ranges::find_if(
      artifact.objects(), [](const omlx::m5::ArtifactObject &object) {
        return object.name.ends_with("model.layers.0.input_layernorm.weight") &&
               object.format == "direct_bf16_v1" && object.shape.size() == 1;
      });
  if (found == artifact.objects().end()) {
    throw std::out_of_range("layer-0 input RMSNorm weight is absent");
  }
  return *found;
}

std::vector<omlx::m5::ActivationUse> projection_uses(bool allow_a4,
                                                     std::string_view prefix) {
  const auto policy = allow_a4 ? omlx::m5::ActivationPolicy::allow_a4
                               : omlx::m5::ActivationPolicy::a16_only;
  return {
      {.binding_name = std::string(prefix) + ".q_proj",
       .role = "attention_q",
       .policy = policy},
      {.binding_name = std::string(prefix) + ".k_proj",
       .role = "attention_k",
       .policy = policy},
      {.binding_name = std::string(prefix) + ".v_proj",
       .role = "attention_v",
       .policy = policy},
      {.binding_name = std::string(prefix) + ".gate_proj",
       .role = "mlp_gate",
       .policy = policy},
      {.binding_name = std::string(prefix) + ".up_proj",
       .role = "mlp_up",
       .policy = policy},
  };
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

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 6) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_normalization ARTIFACT CONFIG RMS_METALLIB "
          "A4_METALLIB QMV_METALLIB");
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto config = omlx::m5::load_qwen_model_config(argv[2]);
    const auto &norm_object = input_norm_object(artifact);
    const auto &norm_plane = norm_object.planes.front();
    const auto norm_words =
        words(artifact.plane_bytes(norm_object, norm_plane));
    const auto norm_values = floats(norm_words);
    const auto width = static_cast<std::uint32_t>(norm_object.shape[0]);
    if (width != static_cast<std::uint32_t>(config.hidden_size)) {
      throw std::runtime_error("RMSNorm weight width disagrees with config");
    }
    const bool allow_a4 = std::ranges::any_of(
        artifact.bindings(), [](const omlx::m5::ArtifactBinding &binding) {
          return binding.activation_policy == "AllowA4";
        });
    auto context = omlx::m5::MetalContext::create();
    auto norm_weight = context.upload(bytes(norm_words));
    nlohmann::json records = nlohmann::json::array();
    for (const auto tokens : kAnchors) {
      std::vector<std::uint16_t> input_words(tokens * width);
      std::vector<float> input_values(tokens * width);
      for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::uint32_t column = 0; column < width; ++column) {
          const auto raw =
              static_cast<float>(static_cast<std::int32_t>(
                                     (column * 19U + token * 23U) % 257U) -
                                 128) /
              256.0F;
          const auto index = token * width + column;
          input_words[index] = omlx::m5::float_to_bf16(raw);
          input_values[index] = omlx::m5::bf16_to_float(input_words[index]);
        }
      }
      auto input = context.upload(bytes(input_words));
      const std::array slots = {omlx::m5::ActivationSlotSpec{
          .max_input_features = width,
          .max_tokens = tokens,
          .policy = allow_a4 ? omlx::m5::ActivationPolicy::allow_a4
                             : omlx::m5::ActivationPolicy::a16_only,
      }};
      omlx::m5::ActivationWorkspace workspace(
          context, omlx::m5::plan_activation_workspace(slots));
      workspace.begin_epoch(1);
      const auto uses = projection_uses(allow_a4, "layer.projections");
      const omlx::m5::ActivationReuseKey key{
          .source_id = 101,
          .normalization_id = 201,
          .input_features = width,
          .tokens = tokens,
      };
      const auto lease =
          workspace.acquire(key, uses,
                            allow_a4 ? omlx::m5::ActivationProfile::a4
                                     : omlx::m5::ActivationProfile::a16,
                            allow_a4);
      auto plan = omlx::m5::prepare_rmsnorm_projection(
          context, argv[3], width, tokens,
          static_cast<float>(config.rms_norm_eps));
      const auto produced = omlx::m5::produce_rmsnorm_activation(
          context, plan, input, 0, norm_weight, 0, lease);
      const auto expected =
          omlx::m5::rms_norm(input_values, tokens, width, norm_values,
                             static_cast<float>(config.rms_norm_eps));
      const auto actual_words = words(workspace.buffer().read(
          lease.bundle.normalized_a16->offset, plan.info().a16_bytes));
      double max_relative_error = 0.0;
      nlohmann::json samples = nlohmann::json::array();
      const std::array sample_tokens = {0U, tokens / 2, tokens - 1};
      const std::array sample_features = {0U, width / 2, width - 1};
      for (const auto token : sample_tokens) {
        for (const auto feature : sample_features) {
          const auto index = token * width + feature;
          const auto actual = omlx::m5::bf16_to_float(actual_words[index]);
          max_relative_error =
              std::max(max_relative_error,
                       std::abs(static_cast<double>(actual) - expected[index]) /
                           std::max(1.0, std::abs(double(expected[index]))));
          samples.push_back({
              {"token", token},
              {"feature", feature},
              {"actual", actual},
              {"cpp_expected", expected[index]},
          });
        }
      }
      bool a4_bit_exact = !allow_a4;
      if (allow_a4) {
        auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(context, argv[4],
                                                              width, tokens);
        auto baseline_codes = context.make_buffer(quantizer.info().code_bytes);
        auto baseline_scales =
            context.make_buffer(quantizer.info().scale_bytes);
        const omlx::m5::Mxfp4ActivationView baseline{
            .codes = {.buffer = &baseline_codes,
                      .bytes = quantizer.info().code_bytes},
            .scales = {.buffer = &baseline_scales,
                       .bytes = quantizer.info().scale_bytes},
            .group_size = 32,
        };
        static_cast<void>(omlx::m5::quantize_mxfp4_a4(
            context, quantizer, workspace.buffer(),
            lease.bundle.normalized_a16->offset, baseline));
        a4_bit_exact = workspace.buffer().read(lease.bundle.a4->codes.offset,
                                               quantizer.info().code_bytes) ==
                           baseline_codes.read() &&
                       workspace.buffer().read(lease.bundle.a4->scales.offset,
                                               quantizer.info().scale_bytes) ==
                           baseline_scales.read();
      }
      const auto before_reuse = context.registry_stats();
      const auto reused_lease =
          workspace.acquire(key, uses,
                            allow_a4 ? omlx::m5::ActivationProfile::a4
                                     : omlx::m5::ActivationProfile::a16,
                            allow_a4);
      const auto reused = omlx::m5::produce_rmsnorm_activation(
          context, plan, input, 0, norm_weight, 0, reused_lease);
      const auto after_reuse = context.registry_stats();
      if (max_relative_error > 0.02 || !a4_bit_exact ||
          reused.mode != omlx::m5::RmsNormProducerMode::none ||
          after_reuse.dispatches_submitted !=
              before_reuse.dispatches_submitted) {
        throw std::runtime_error("real RMSNorm production validation failed");
      }
      records.push_back({
          {"tokens", tokens},
          {"mode", omlx::m5::to_string(produced.mode)},
          {"dispatches", produced.submission->dispatch_count},
          {"max_relative_error", max_relative_error},
          {"a4_bit_exact", a4_bit_exact},
          {"reuse_mode", omlx::m5::to_string(reused.mode)},
          {"reuse_dispatch_delta", after_reuse.dispatches_submitted -
                                       before_reuse.dispatches_submitted},
          {"workspace", workspace.stats().capacity_bytes},
          {"samples", std::move(samples)},
      });
    }

    const auto &projection_binding = binding_by_use(artifact, "gdn_qkv");
    const auto &projection_object =
        artifact.object(projection_binding.object_id);
    std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> weight_buffers;
    auto projection_weight =
        upload_weight(artifact, projection_object, context, weight_buffers);
    std::vector<std::uint16_t> input_words(width);
    std::vector<float> input_values(width);
    for (std::uint32_t column = 0; column < width; ++column) {
      const auto raw =
          static_cast<float>(static_cast<std::int32_t>((column * 19U) % 257U) -
                             128) /
          256.0F;
      input_words[column] = omlx::m5::float_to_bf16(raw);
      input_values[column] = omlx::m5::bf16_to_float(input_words[column]);
    }
    auto input = context.upload(bytes(input_words));
    const std::array integration_slots = {omlx::m5::ActivationSlotSpec{
        .max_input_features = width,
        .max_tokens = 1,
        .policy = omlx::m5::ActivationPolicy::a16_only,
    }};
    omlx::m5::ActivationWorkspace integration_workspace(
        context, omlx::m5::plan_activation_workspace(integration_slots));
    integration_workspace.begin_epoch(1);
    const auto integration_lease = integration_workspace.acquire(
        {.source_id = 301,
         .normalization_id = 401,
         .input_features = width,
         .tokens = 1},
        projection_uses(false, "layer.integration"),
        omlx::m5::ActivationProfile::a16);
    auto norm_plan = omlx::m5::prepare_rmsnorm_projection(
        context, argv[3], width, 1, static_cast<float>(config.rms_norm_eps));
    static_cast<void>(omlx::m5::produce_rmsnorm_activation(
        context, norm_plan, input, 0, norm_weight, 0, integration_lease));
    auto linear_plan =
        omlx::m5::prepare_dense_qmv_t1(context, argv[5], projection_weight);
    auto output = context.make_buffer(linear_plan.info().output_bytes);
    const auto submission = omlx::m5::execute_dense_qmv_t1(
        context, linear_plan, *integration_lease.bundle.normalized_a16->buffer,
        integration_lease.bundle.normalized_a16->offset, output, 0);
    const auto expected_norm =
        omlx::m5::rms_norm(input_values, 1, width, norm_values,
                           static_cast<float>(config.rms_norm_eps));
    const auto represented_norm = [&] {
      std::vector<float> result(expected_norm.size());
      std::ranges::transform(expected_norm, result.begin(), [](float value) {
        return omlx::m5::bf16_to_float(omlx::m5::float_to_bf16(value));
      });
      return result;
    }();
    const std::array sample_rows = {0ULL, projection_object.shape[0] / 2,
                                    projection_object.shape[0] - 1};
    double integration_max_relative_error = 0.0;
    nlohmann::json integration_samples = nlohmann::json::array();
    const auto output_bytes = output.read();
    for (const auto row : sample_rows) {
      const auto weights = artifact.decode_weight_row(
          projection_object, row, 0, projection_object.shape[1]);
      double expected = 0.0;
      for (std::uint64_t column = 0; column < width; ++column) {
        expected +=
            static_cast<double>(represented_norm[column]) * weights[column];
      }
      std::uint16_t word = 0;
      std::memcpy(&word, output_bytes.data() + row * 2, sizeof(word));
      const auto actual = omlx::m5::bf16_to_float(word);
      integration_max_relative_error =
          std::max(integration_max_relative_error,
                   std::abs(static_cast<double>(actual) - expected) /
                       std::max(1.0, std::abs(expected)));
      integration_samples.push_back(
          {{"row", row}, {"actual", actual}, {"expected", expected}});
    }
    if (integration_max_relative_error > 0.02) {
      throw std::runtime_error("RMSNorm to parent Linear validation failed");
    }
    const nlohmann::json result = {
        {"schema", "omlx-m5-b3-2-3-normalization-v1"},
        {"evidence_scope",
         "M4 real RMSNorm/A4 production and parent Linear integration"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"norm_object", norm_object.name},
        {"input_features", width},
        {"allow_a4", allow_a4},
        {"records", std::move(records)},
        {"integration",
         {
             {"binding", projection_binding.name},
             {"format", projection_object.format},
             {"norm_dispatches", 1},
             {"linear_dispatches", submission.dispatch_count},
             {"max_relative_error", integration_max_relative_error},
             {"samples", std::move(integration_samples)},
         }},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "normalization inspection failed: " << error.what() << '\n';
    return 1;
  }
}
