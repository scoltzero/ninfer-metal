#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/mxfp4_small_t.h"
#include "omlx_m5/quantization.h"

namespace {

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::uint32_t parse_u32(const char *text, const char *label) {
  const auto value = std::stoull(text);
  if (value == 0 || value > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(label) + " is out of range");
  }
  return static_cast<std::uint32_t>(value);
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

double cosine(std::span<const float> left, std::span<const float> right) {
  double dot = 0.0;
  double left_norm = 0.0;
  double right_norm = 0.0;
  for (std::size_t index = 0; index < left.size(); ++index) {
    dot += static_cast<double>(left[index]) * right[index];
    left_norm += static_cast<double>(left[index]) * left[index];
    right_norm += static_cast<double>(right[index]) * right[index];
  }
  if (left_norm == 0.0 || right_norm == 0.0) {
    return left_norm == right_norm ? 1.0 : 0.0;
  }
  return dot / std::sqrt(left_norm * right_norm);
}

std::uint64_t delta(std::uint64_t after, std::uint64_t before,
                    const char *label) {
  if (after < before) {
    throw std::logic_error(std::string(label) + " counter regressed");
  }
  return after - before;
}

omlx::m5::KernelKey make_key(const omlx::m5::DenseMxfp4SmallTPlan &plan,
                             omlx::m5::ActivationProfile activation) {
  return {
      .format_id = plan.info().format_id,
      .layout_id = plan.info().layout_id,
      .activation_policy = omlx::m5::ActivationPolicy::allow_a4,
      .preferred_activation = activation,
      .input_type = omlx::m5::KernelDataType::bf16,
      .output_type = omlx::m5::KernelDataType::bf16,
      .accumulator_type = omlx::m5::KernelDataType::fp32,
      .geometry = omlx::m5::KernelGeometry::dense,
      .output_features = plan.info().output_features,
      .input_features = plan.info().input_features,
      .tokens = plan.info().tokens,
      .fused_op = omlx::m5::KernelFusedOp::linear,
      .alignment = omlx::m5::KernelAlignmentClass::full,
  };
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 5) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_mxfp4_small_t ARTIFACT OBJECT "
          "CLASSIC_METALLIB TENSOROPS_METALLIB [--iterations N] "
          "[--sample-rows N]");
    }
    std::uint32_t iterations = 2;
    std::uint32_t sample_rows = 8;
    for (int index = 5; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--iterations" && index + 1 < argc) {
        iterations = parse_u32(argv[++index], "iterations");
      } else if (argument == "--sample-rows" && index + 1 < argc) {
        sample_rows = parse_u32(argv[++index], "sample rows");
      } else {
        throw std::invalid_argument("unknown MXFP4 small-T argument: " +
                                    std::string(argument));
      }
    }

    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto &object = artifact.object(argv[2]);
    if (object.shape.size() != 2 || object.format != "mxfp4_e2m1_g32_e8m0_v1") {
      throw std::invalid_argument(
          "MXFP4 small-T inspector requires a rank-2 MXFP4 object");
    }
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
        .row_count = object.shape[0],
        .in_features = object.shape[1],
    };
    for (const auto &plane : object.planes) {
      plane_buffers.push_back(std::make_unique<omlx::m5::MetalBuffer>(
          context.upload(artifact.plane_bytes(object, plane))));
      weight.planes.push_back({
          .kind = plane.kind,
          .buffer = plane_buffers.back().get(),
          .segment_index = static_cast<std::uint32_t>(plane_buffers.size() - 1),
          .offset = 0,
          .bytes = plane.bytes,
          .row_stride = plane.bytes / object.shape[0],
      });
    }
    std::uint64_t weight_plane_bytes = 0;
    for (const auto &plane : object.planes) {
      weight_plane_bytes += plane.bytes;
    }

    const auto selected_row_count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(sample_rows, object.shape[0]));
    std::vector<std::uint64_t> rows;
    for (std::uint32_t index = 0; index < selected_row_count; ++index) {
      const auto row = selected_row_count == 1 ? 0
                                               : index * (object.shape[0] - 1) /
                                                     (selected_row_count - 1);
      if (rows.empty() || rows.back() != row) {
        rows.push_back(row);
      }
    }
    std::vector<std::vector<float>> decoded_rows;
    for (const auto row : rows) {
      decoded_rows.push_back(
          artifact.decode_weight_row(object, row, 0, object.shape[1]));
    }

    nlohmann::json records = nlohmann::json::array();
    for (std::uint32_t tokens = 2; tokens <= 12; ++tokens) {
      const auto before_prepare = context.registry_stats();
      auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
          context, argv[3], object.shape[1], tokens);
      auto plan = omlx::m5::prepare_dense_mxfp4_small_t(
          context, argv[3], argv[4], weight, tokens);
      omlx::m5::DispatchTable dispatch;
      for (const auto &candidate : plan.dispatch_candidates()) {
        dispatch.register_candidate(candidate);
      }
      const auto installed = plan.installed_executor_keys();
      dispatch.seal(installed);
      const auto selected_a16 =
          dispatch.prepare(make_key(plan, omlx::m5::ActivationProfile::a16),
                           context.capabilities().profile);
      const auto selected_a4 =
          dispatch.prepare(make_key(plan, omlx::m5::ActivationProfile::a4),
                           context.capabilities().profile);
      const auto after_prepare = context.registry_stats();

      std::vector<std::uint16_t> input_words;
      const auto input = make_input(tokens, object.shape[1], input_words);
      const auto quantized = quantize_input(input, tokens, object.shape[1]);
      auto input_buffer = context.upload(bytes(input_words));
      auto activation_codes = context.make_buffer(quantizer.info().code_bytes);
      auto activation_scales =
          context.make_buffer(quantizer.info().scale_bytes);
      const omlx::m5::Mxfp4ActivationView activation{
          .codes = {.buffer = &activation_codes,
                    .bytes = quantizer.info().code_bytes},
          .scales = {.buffer = &activation_scales,
                     .bytes = quantizer.info().scale_bytes},
          .group_size = 32,
      };
      const auto quantize_submission = omlx::m5::quantize_mxfp4_a4(
          context, quantizer, input_buffer, 0, activation);
      const auto actual_codes = activation_codes.read();
      const auto actual_scales = activation_scales.read();
      if (actual_codes != bytes(quantized.codes) ||
          actual_scales != bytes(quantized.scales)) {
        throw std::runtime_error("real A4 quantization differs from oracle");
      }

      auto output = context.make_buffer(plan.info().output_bytes);
      const auto before_execute = context.registry_stats();
      std::vector<double> a16_wall_ms;
      std::vector<double> a16_gpu_ms;
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto submission = omlx::m5::execute_dense_mxfp4_small_t_a16(
            context, plan, selected_a16.candidate.executor_key, input_buffer, 0,
            output, 0);
        a16_wall_ms.push_back(submission.timing.wall_ms);
        a16_gpu_ms.push_back(submission.timing.gpu_ms);
      }
      const auto a16_output = output.read();
      std::vector<double> a4_wall_ms;
      std::vector<double> a4_gpu_ms;
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto submission = omlx::m5::execute_dense_mxfp4_small_t_a4(
            context, plan, selected_a4.candidate.executor_key, activation,
            output, 0);
        a4_wall_ms.push_back(submission.timing.wall_ms);
        a4_gpu_ms.push_back(submission.timing.gpu_ms);
      }
      const auto a4_output = output.read();
      nlohmann::json candidate_timings = nlohmann::json::array();
      candidate_timings.push_back({
          {"family",
           omlx::m5::to_string(
               plan.executor(selected_a16.candidate.executor_key).family)},
          {"activation", "A16"},
          {"executor_key", selected_a16.candidate.executor_key},
          {"selected", true},
          {"bit_exact_selected", true},
          {"wall_ms", a16_wall_ms},
          {"gpu_ms", a16_gpu_ms},
      });
      candidate_timings.push_back({
          {"family",
           omlx::m5::to_string(
               plan.executor(selected_a4.candidate.executor_key).family)},
          {"activation", "A4"},
          {"executor_key", selected_a4.candidate.executor_key},
          {"selected", true},
          {"bit_exact_selected", true},
          {"wall_ms", a4_wall_ms},
          {"gpu_ms", a4_gpu_ms},
      });
      for (const auto &alternative : plan.info().executors) {
        if (!alternative.installed ||
            alternative.executor_key == selected_a16.candidate.executor_key ||
            alternative.executor_key == selected_a4.candidate.executor_key) {
          continue;
        }
        std::vector<double> alternative_wall_ms;
        std::vector<double> alternative_gpu_ms;
        alternative_wall_ms.reserve(iterations);
        alternative_gpu_ms.reserve(iterations);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          const auto submission =
              alternative.activation == omlx::m5::ActivationProfile::a16
                  ? omlx::m5::execute_dense_mxfp4_small_t_a16(
                        context, plan, alternative.executor_key, input_buffer,
                        0, output, 0)
                  : omlx::m5::execute_dense_mxfp4_small_t_a4(
                        context, plan, alternative.executor_key, activation,
                        output, 0);
          alternative_wall_ms.push_back(submission.timing.wall_ms);
          alternative_gpu_ms.push_back(submission.timing.gpu_ms);
        }
        const auto &reference =
            alternative.activation == omlx::m5::ActivationProfile::a16
                ? a16_output
                : a4_output;
        candidate_timings.push_back({
            {"family", omlx::m5::to_string(alternative.family)},
            {"activation", omlx::m5::to_string(alternative.activation)},
            {"executor_key", alternative.executor_key},
            {"selected", false},
            {"bit_exact_selected", output.read() == reference},
            {"wall_ms", std::move(alternative_wall_ms)},
            {"gpu_ms", std::move(alternative_gpu_ms)},
        });
      }
      const auto after_execute = context.registry_stats();
      if (before_execute.upload_count != after_execute.upload_count ||
          before_execute.pipelines_compiled !=
              after_execute.pipelines_compiled) {
        throw std::runtime_error(
            "MXFP4 small-T token loop uploaded or compiled a resource");
      }

      std::vector<float> actual_a16;
      std::vector<float> actual_a4;
      std::vector<float> expected_a16;
      std::vector<float> expected_a4;
      nlohmann::json samples = nlohmann::json::array();
      double a16_max_abs = 0.0;
      double a16_max_rel = 0.0;
      double a4_max_abs = 0.0;
      double a4_max_rel = 0.0;
      for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::size_t sample = 0; sample < rows.size(); ++sample) {
          const auto row = rows[sample];
          const auto output_index = token * object.shape[0] + row;
          std::uint16_t a16_word = 0;
          std::uint16_t a4_word = 0;
          std::memcpy(&a16_word, a16_output.data() + output_index * 2,
                      sizeof(a16_word));
          std::memcpy(&a4_word, a4_output.data() + output_index * 2,
                      sizeof(a4_word));
          const auto got_a16 = omlx::m5::bf16_to_float(a16_word);
          const auto got_a4 = omlx::m5::bf16_to_float(a4_word);
          double reference_a16 = 0.0;
          double reference_a4 = 0.0;
          for (std::uint64_t column = 0; column < object.shape[1]; ++column) {
            reference_a16 +=
                static_cast<double>(input[token * object.shape[1] + column]) *
                decoded_rows[sample][column];
            reference_a4 +=
                static_cast<double>(
                    quantized.values[token * object.shape[1] + column]) *
                decoded_rows[sample][column];
          }
          const auto error_a16 =
              std::abs(static_cast<double>(got_a16) - reference_a16);
          const auto error_a4 =
              std::abs(static_cast<double>(got_a4) - reference_a4);
          a16_max_abs = std::max(a16_max_abs, error_a16);
          a16_max_rel = std::max(
              a16_max_rel, error_a16 / std::max(1.0, std::abs(reference_a16)));
          a4_max_abs = std::max(a4_max_abs, error_a4);
          a4_max_rel = std::max(
              a4_max_rel, error_a4 / std::max(1.0, std::abs(reference_a4)));
          actual_a16.push_back(got_a16);
          actual_a4.push_back(got_a4);
          expected_a16.push_back(static_cast<float>(reference_a16));
          expected_a4.push_back(static_cast<float>(reference_a4));
          samples.push_back({
              {"token", token},
              {"row", row},
              {"actual_a16", got_a16},
              {"actual_a4", got_a4},
              {"cpp_expected_a16", reference_a16},
              {"cpp_expected_a4", reference_a4},
          });
        }
      }
      if (a16_max_rel > 0.02 || a4_max_rel > 0.02) {
        throw std::runtime_error("real MXFP4 small-T output exceeds tolerance");
      }
      nlohmann::json executors = nlohmann::json::array();
      for (const auto &executor : plan.info().executors) {
        executors.push_back({
            {"family", omlx::m5::to_string(executor.family)},
            {"activation", omlx::m5::to_string(executor.activation)},
            {"route", omlx::m5::to_string(executor.required_route)},
            {"function", executor.function_name},
            {"installed", executor.installed},
        });
      }
      records.push_back({
          {"tokens", tokens},
          {"capacity", plan.info().capacity},
          {"selected_a16", selected_a16.candidate.id},
          {"selected_a4", selected_a4.candidate.id},
          {"selected_a16_family",
           omlx::m5::to_string(
               plan.executor(selected_a16.candidate.executor_key).family)},
          {"selected_a4_family",
           omlx::m5::to_string(
               plan.executor(selected_a4.candidate.executor_key).family)},
          {"native_tensorops_runtime_eligible",
           plan.info().native_tensorops_runtime_eligible},
          {"installed_candidate_count",
           dispatch.stats().installed_candidate_count},
          {"planned_candidate_count", dispatch.stats().planned_candidate_count},
          {"executors", std::move(executors)},
          {"a4_code_bytes", quantizer.info().code_bytes},
          {"a4_scale_bytes", quantizer.info().scale_bytes},
          {"a4_quantize_dispatches", quantize_submission.dispatch_count},
          {"pipelines_compiled_during_prepare",
           delta(after_prepare.pipelines_compiled,
                 before_prepare.pipelines_compiled, "pipeline")},
          {"command_plans_built_during_prepare",
           delta(after_prepare.command_plans_built,
                 before_prepare.command_plans_built, "command plan")},
          {"token_loop_uploads", delta(after_execute.upload_count,
                                       before_execute.upload_count, "upload")},
          {"token_loop_pipeline_compiles",
           delta(after_execute.pipelines_compiled,
                 before_execute.pipelines_compiled, "pipeline")},
          {"a16_max_abs_error", a16_max_abs},
          {"a16_max_rel_error", a16_max_rel},
          {"a16_cosine", cosine(actual_a16, expected_a16)},
          {"a4_max_abs_error", a4_max_abs},
          {"a4_max_rel_error", a4_max_rel},
          {"a4_cosine", cosine(actual_a4, expected_a4)},
          {"iterations", iterations},
          {"a16_wall_ms", a16_wall_ms},
          {"a16_gpu_ms", a16_gpu_ms},
          {"a4_wall_ms", a4_wall_ms},
          {"a4_gpu_ms", a4_gpu_ms},
          {"candidate_timings", std::move(candidate_timings)},
          {"sample_count", samples.size()},
          {"samples", std::move(samples)},
      });
    }

    const auto registry = context.registry_stats();
    const nlohmann::json result = {
        {"schema", "omlx-m5-b2-4-mxfp4-small-t-v1"},
        {"evidence_scope",
         "M4 SIMD correctness and latency plus SDK 26.2 TensorOps compile; not "
         "M5 Ultra performance"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"object", object.name},
        {"format", object.format},
        {"layout", object.layout},
        {"output_features", object.shape[0]},
        {"input_features", object.shape[1]},
        {"padded_input_features", object.padded_k},
        {"weight_plane_bytes", weight_plane_bytes},
        {"records", std::move(records)},
        {"registry",
         {
             {"pipelines_compiled", registry.pipelines_compiled},
             {"command_plans_built", registry.command_plans_built},
             {"dispatches_submitted", registry.dispatches_submitted},
             {"upload_count", registry.upload_count},
             {"uploaded_bytes", registry.uploaded_bytes},
         }},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "MXFP4 small-T inspection failed: " << error.what() << '\n';
    return 1;
  }
}
