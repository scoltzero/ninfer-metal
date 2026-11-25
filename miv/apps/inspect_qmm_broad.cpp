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
#include "omlx_m5/qmm_broad.h"
#include "omlx_m5/quantization.h"

namespace {

constexpr std::array kAnchors = {13U, 16U,  17U,  31U,  32U,
                                 64U, 127U, 128U, 512U, 1024U};

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

omlx::m5::KernelKey make_key(const omlx::m5::DenseQmmBroadPlan &plan,
                             omlx::m5::ActivationProfile activation) {
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
    if (argc < 6) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_qmm_broad ARTIFACT OBJECT BROAD_METALLIB "
          "A4_METALLIB TENSOROPS_METALLIB [--rows N] [--iterations N]");
    }
    std::uint32_t execute_rows = 8;
    std::uint32_t iterations = 2;
    for (int index = 6; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--rows" && index + 1 < argc) {
        execute_rows = parse_u32(argv[++index], "rows");
      } else if (argument == "--iterations" && index + 1 < argc) {
        iterations = parse_u32(argv[++index], "iterations");
      } else {
        throw std::invalid_argument("unknown broad inspector argument");
      }
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto &object = artifact.object(argv[2]);
    if (object.shape.size() != 2 || object.format.starts_with("direct_")) {
      throw std::invalid_argument("broad inspector requires quantized rank-2");
    }
    const bool mxfp4 = object.format == "mxfp4_e2m1_g32_e8m0_v1";
    const auto rows = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(execute_rows, object.shape[0]));
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
        .row_count = rows,
        .in_features = object.shape[1],
    };
    std::uint64_t uploaded_weight_bytes = 0;
    for (const auto &selected_plane : object.planes) {
      const auto source = artifact.plane_bytes(object, selected_plane);
      const auto row_stride = selected_plane.bytes / object.shape[0];
      const auto selected_bytes = row_stride * rows;
      plane_buffers.push_back(std::make_unique<omlx::m5::MetalBuffer>(
          context.upload(source.first(selected_bytes))));
      weight.planes.push_back({
          .kind = selected_plane.kind,
          .buffer = plane_buffers.back().get(),
          .segment_index = static_cast<std::uint32_t>(plane_buffers.size() - 1),
          .bytes = selected_bytes,
          .row_stride = row_stride,
      });
      uploaded_weight_bytes += selected_bytes;
    }
    std::vector<std::vector<float>> decoded_rows;
    for (std::uint32_t row = 0; row < rows; ++row) {
      decoded_rows.push_back(
          artifact.decode_weight_row(object, row, 0, object.shape[1]));
    }

    auto full_weight = weight;
    full_weight.row_count = object.shape[0];
    for (std::size_t index = 0; index < full_weight.planes.size(); ++index) {
      full_weight.planes[index].bytes = object.planes[index].bytes;
    }
    std::uint64_t full_shape_route_keys = 0;
    std::uint64_t max_full_output_bytes = 0;
    const auto check_full_shape_route = [&](std::uint32_t tokens) {
      auto full_plan = omlx::m5::prepare_dense_qmm_broad(
          context, argv[3], argv[5], full_weight, tokens);
      omlx::m5::DispatchTable full_dispatch;
      for (const auto &candidate : full_plan.dispatch_candidates()) {
        full_dispatch.register_candidate(candidate);
      }
      const auto installed = full_plan.installed_executor_keys();
      full_dispatch.seal(installed);
      static_cast<void>(full_dispatch.prepare(
          make_key(full_plan, omlx::m5::ActivationProfile::a16),
          context.capabilities().profile));
      ++full_shape_route_keys;
      if (mxfp4) {
        static_cast<void>(full_dispatch.prepare(
            make_key(full_plan, omlx::m5::ActivationProfile::a4),
            context.capabilities().profile));
        ++full_shape_route_keys;
      }
      max_full_output_bytes =
          std::max(max_full_output_bytes, full_plan.info().output_bytes);
    };
    for (std::uint32_t tokens = 13; tokens <= 128; ++tokens) {
      check_full_shape_route(tokens);
    }
    check_full_shape_route(512);
    check_full_shape_route(1024);

    nlohmann::json records = nlohmann::json::array();
    for (const auto tokens : kAnchors) {
      auto plan = omlx::m5::prepare_dense_qmm_broad(context, argv[3], argv[5],
                                                    weight, tokens);
      omlx::m5::DispatchTable dispatch;
      for (const auto &candidate : plan.dispatch_candidates()) {
        dispatch.register_candidate(candidate);
      }
      const auto installed = plan.installed_executor_keys();
      dispatch.seal(installed);
      const auto selected_a16 =
          dispatch.prepare(make_key(plan, omlx::m5::ActivationProfile::a16),
                           context.capabilities().profile);
      std::vector<std::uint16_t> input_words;
      const auto input = make_input(tokens, object.shape[1], input_words);
      auto input_buffer = context.upload(bytes(input_words));
      auto output = context.make_buffer(plan.info().output_bytes);
      const auto before_execute = context.registry_stats();
      std::vector<double> a16_gpu_ms;
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto submission = omlx::m5::execute_dense_qmm_broad_a16(
            context, plan, selected_a16.candidate.executor_key, input_buffer, 0,
            output, 0);
        a16_gpu_ms.push_back(submission.timing.gpu_ms);
      }
      const auto a16_output = output.read();
      nlohmann::json candidate_timings = nlohmann::json::array();
      candidate_timings.push_back({
          {"family",
           omlx::m5::to_string(
               plan.executor(selected_a16.candidate.executor_key).family)},
          {"activation", "A16"},
          {"executor_key", selected_a16.candidate.executor_key},
          {"selected", true},
          {"bit_exact_selected", true},
          {"gpu_ms", a16_gpu_ms},
      });
      for (const auto &alternative : plan.info().executors) {
        if (!alternative.installed ||
            alternative.activation != omlx::m5::ActivationProfile::a16 ||
            alternative.executor_key == selected_a16.candidate.executor_key) {
          continue;
        }
        std::vector<double> alternative_gpu_ms;
        alternative_gpu_ms.reserve(iterations);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          const auto submission = omlx::m5::execute_dense_qmm_broad_a16(
              context, plan, alternative.executor_key, input_buffer, 0, output,
              0);
          alternative_gpu_ms.push_back(submission.timing.gpu_ms);
        }
        candidate_timings.push_back({
            {"family", omlx::m5::to_string(alternative.family)},
            {"activation", "A16"},
            {"executor_key", alternative.executor_key},
            {"selected", false},
            {"bit_exact_selected", output.read() == a16_output},
            {"gpu_ms", std::move(alternative_gpu_ms)},
        });
      }
      const auto after_a16 = context.registry_stats();
      std::uint64_t token_loop_uploads =
          after_a16.upload_count - before_execute.upload_count;
      std::uint64_t token_loop_pipeline_compiles =
          after_a16.pipelines_compiled - before_execute.pipelines_compiled;

      std::optional<QuantizedInput> quantized;
      std::vector<std::byte> a4_output;
      std::vector<double> a4_gpu_ms;
      std::string selected_a4_family;
      if (mxfp4) {
        quantized = quantize_input(input, tokens, object.shape[1]);
        auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
            context, argv[4], object.shape[1], tokens);
        auto code_buffer = context.make_buffer(quantizer.info().code_bytes);
        auto scale_buffer = context.make_buffer(quantizer.info().scale_bytes);
        const omlx::m5::Mxfp4ActivationView activation{
            .codes = {.buffer = &code_buffer,
                      .bytes = quantizer.info().code_bytes},
            .scales = {.buffer = &scale_buffer,
                       .bytes = quantizer.info().scale_bytes},
            .group_size = 32,
        };
        static_cast<void>(omlx::m5::quantize_mxfp4_a4(
            context, quantizer, input_buffer, 0, activation));
        if (code_buffer.read() != bytes(quantized->codes) ||
            scale_buffer.read() != bytes(quantized->scales)) {
          throw std::runtime_error("broad A4 quantization differs from oracle");
        }
        const auto selected_a4 =
            dispatch.prepare(make_key(plan, omlx::m5::ActivationProfile::a4),
                             context.capabilities().profile);
        selected_a4_family = std::string(omlx::m5::to_string(
            plan.executor(selected_a4.candidate.executor_key).family));
        const auto before_a4 = context.registry_stats();
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          const auto submission = omlx::m5::execute_dense_qmm_broad_a4(
              context, plan, selected_a4.candidate.executor_key, activation,
              output, 0);
          a4_gpu_ms.push_back(submission.timing.gpu_ms);
        }
        a4_output = output.read();
        candidate_timings.push_back({
            {"family", selected_a4_family},
            {"activation", "A4"},
            {"executor_key", selected_a4.candidate.executor_key},
            {"selected", true},
            {"bit_exact_selected", true},
            {"gpu_ms", a4_gpu_ms},
        });
        for (const auto &alternative : plan.info().executors) {
          if (!alternative.installed ||
              alternative.activation != omlx::m5::ActivationProfile::a4 ||
              alternative.executor_key == selected_a4.candidate.executor_key) {
            continue;
          }
          std::vector<double> alternative_gpu_ms;
          alternative_gpu_ms.reserve(iterations);
          for (std::uint32_t iteration = 0; iteration < iterations;
               ++iteration) {
            const auto submission = omlx::m5::execute_dense_qmm_broad_a4(
                context, plan, alternative.executor_key, activation, output, 0);
            alternative_gpu_ms.push_back(submission.timing.gpu_ms);
          }
          candidate_timings.push_back({
              {"family", omlx::m5::to_string(alternative.family)},
              {"activation", "A4"},
              {"executor_key", alternative.executor_key},
              {"selected", false},
              {"bit_exact_selected", output.read() == a4_output},
              {"gpu_ms", std::move(alternative_gpu_ms)},
          });
        }
        const auto after_a4 = context.registry_stats();
        token_loop_uploads += after_a4.upload_count - before_a4.upload_count;
        token_loop_pipeline_compiles +=
            after_a4.pipelines_compiled - before_a4.pipelines_compiled;
      }

      std::vector<std::uint32_t> sample_tokens = {0, tokens / 2, tokens - 1};
      std::ranges::sort(sample_tokens);
      sample_tokens.erase(
          std::unique(sample_tokens.begin(), sample_tokens.end()),
          sample_tokens.end());
      std::vector<float> actual_a16;
      std::vector<float> expected_a16;
      std::vector<float> actual_a4;
      std::vector<float> expected_a4;
      nlohmann::json samples = nlohmann::json::array();
      double a16_max_rel = 0.0;
      double a4_max_rel = 0.0;
      for (const auto token : sample_tokens) {
        for (std::uint32_t row = 0; row < rows; ++row) {
          const auto output_index = token * rows + row;
          std::uint16_t word = 0;
          std::memcpy(&word, a16_output.data() + output_index * 2,
                      sizeof(word));
          const auto got_a16 = omlx::m5::bf16_to_float(word);
          double reference_a16 = 0.0;
          double reference_a4 = 0.0;
          for (std::uint64_t column = 0; column < object.shape[1]; ++column) {
            reference_a16 +=
                static_cast<double>(input[token * object.shape[1] + column]) *
                decoded_rows[row][column];
            if (quantized.has_value()) {
              reference_a4 +=
                  static_cast<double>(
                      quantized->values[token * object.shape[1] + column]) *
                  decoded_rows[row][column];
            }
          }
          const auto error_a16 =
              std::abs(static_cast<double>(got_a16) - reference_a16);
          a16_max_rel = std::max(
              a16_max_rel, error_a16 / std::max(1.0, std::abs(reference_a16)));
          actual_a16.push_back(got_a16);
          expected_a16.push_back(static_cast<float>(reference_a16));
          nlohmann::json sample = {
              {"token", token},
              {"row", row},
              {"actual_a16", got_a16},
              {"cpp_expected_a16", reference_a16},
          };
          if (quantized.has_value()) {
            std::memcpy(&word, a4_output.data() + output_index * 2,
                        sizeof(word));
            const auto got_a4 = omlx::m5::bf16_to_float(word);
            const auto error_a4 =
                std::abs(static_cast<double>(got_a4) - reference_a4);
            a4_max_rel = std::max(
                a4_max_rel, error_a4 / std::max(1.0, std::abs(reference_a4)));
            actual_a4.push_back(got_a4);
            expected_a4.push_back(static_cast<float>(reference_a4));
            sample["actual_a4"] = got_a4;
            sample["cpp_expected_a4"] = reference_a4;
          }
          samples.push_back(std::move(sample));
        }
      }
      if (a16_max_rel > 0.02 || (mxfp4 && a4_max_rel > 0.02)) {
        throw std::runtime_error("broad real-weight output exceeds tolerance");
      }
      records.push_back({
          {"tokens", tokens},
          {"domain", plan.info().domain},
          {"token_tiles", plan.info().token_tiles},
          {"selected_a16_family",
           omlx::m5::to_string(
               plan.executor(selected_a16.candidate.executor_key).family)},
          {"selected_a4_family", selected_a4_family},
          {"native_tensorops_runtime_eligible",
           plan.info().native_tensorops_runtime_eligible},
          {"candidate_count", dispatch.stats().candidate_count},
          {"installed_candidate_count",
           dispatch.stats().installed_candidate_count},
          {"planned_candidate_count", dispatch.stats().planned_candidate_count},
          {"workspace_bytes", plan.info().workspace_bytes},
          {"a16_max_rel_error", a16_max_rel},
          {"a16_cosine", cosine(actual_a16, expected_a16)},
          {"a4_max_rel_error", a4_max_rel},
          {"a4_cosine", mxfp4 ? cosine(actual_a4, expected_a4) : 1.0},
          {"a16_gpu_ms", a16_gpu_ms},
          {"a4_gpu_ms", a4_gpu_ms},
          {"candidate_timings", std::move(candidate_timings)},
          {"token_loop_uploads", token_loop_uploads},
          {"token_loop_pipeline_compiles", token_loop_pipeline_compiles},
          {"samples", std::move(samples)},
      });
    }
    const nlohmann::json result = {
        {"schema", "omlx-m5-b2-5-broad-v1"},
        {"evidence_scope",
         "M4 real-weight row-slice correctness; not full-N bulk performance "
         "or M5 Ultra performance"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"object", object.name},
        {"format", object.format},
        {"layout", object.layout},
        {"parent_output_features", object.shape[0]},
        {"executed_output_features", rows},
        {"input_features", object.shape[1]},
        {"uploaded_weight_bytes", uploaded_weight_bytes},
        {"full_shape_route_sweep",
         {
             {"tokens", 118},
             {"route_keys", full_shape_route_keys},
             {"uncovered", 0},
             {"max_output_bytes", max_full_output_bytes},
             {"execution_scope", "plan-only; no undersized buffer binding"},
         }},
        {"records", std::move(records)},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "broad QMM inspection failed: " << error.what() << '\n';
    return 1;
  }
}
