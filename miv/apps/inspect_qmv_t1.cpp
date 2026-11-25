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
#include <numeric>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"

namespace {

std::uint32_t parse_u32(std::string_view value, std::string_view label) {
  std::size_t parsed = 0;
  const auto result = std::stoull(std::string(value), &parsed, 0);
  if (parsed != value.size() || result == 0 ||
      result > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument(std::string(label) + " is invalid");
  }
  return static_cast<std::uint32_t>(result);
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::vector<float> make_input(std::uint64_t width,
                              std::vector<std::uint16_t> &words) {
  std::vector<float> result(width);
  words.resize(width);
  for (std::uint64_t column = 0; column < width; ++column) {
    const auto raw =
        static_cast<float>(static_cast<std::int32_t>((column * 17U) % 257U) -
                           128) /
        512.0F;
    words[column] = omlx::m5::float_to_bf16(raw);
    result[column] = omlx::m5::bf16_to_float(words[column]);
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

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_qmv_t1 ARTIFACT OBJECT METALLIB "
          "[--iterations N] [--sample-rows N]");
    }
    std::uint32_t iterations = 3;
    std::uint32_t sample_rows = 8;
    for (int index = 4; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--iterations" && index + 1 < argc) {
        iterations = parse_u32(argv[++index], "iterations");
      } else if (argument == "--sample-rows" && index + 1 < argc) {
        sample_rows = parse_u32(argv[++index], "sample rows");
      } else {
        throw std::invalid_argument("unknown QMV argument: " +
                                    std::string(argument));
      }
    }
    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto &object = artifact.object(argv[2]);
    if (object.shape.size() != 2) {
      throw std::invalid_argument("QMV inspector requires a rank-2 object");
    }
    auto context = omlx::m5::MetalContext::create();
    std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> plane_buffers;
    plane_buffers.reserve(object.planes.size());
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
    auto plan = omlx::m5::prepare_dense_qmv_t1(context, argv[3], weight);
    const auto candidates = plan.dispatch_candidates();
    omlx::m5::DispatchTable dispatch;
    for (const auto &candidate : candidates) {
      dispatch.register_candidate(candidate);
    }
    const std::array installed = {plan.info().executor_key};
    dispatch.seal(installed);
    omlx::m5::KernelKey key{
        .format_id = object.format,
        .layout_id = object.layout,
        .activation_policy = object.format == "mxfp4_e2m1_g32_e8m0_v1"
                                 ? omlx::m5::ActivationPolicy::allow_a4
                                 : omlx::m5::ActivationPolicy::a16_only,
        .preferred_activation = omlx::m5::ActivationProfile::a16,
        .input_type = omlx::m5::KernelDataType::bf16,
        .output_type = omlx::m5::KernelDataType::bf16,
        .accumulator_type = omlx::m5::KernelDataType::fp32,
        .geometry = omlx::m5::KernelGeometry::dense,
        .output_features = object.shape[0],
        .input_features = object.shape[1],
        .tokens = 1,
        .fused_op = omlx::m5::KernelFusedOp::linear,
        .alignment = omlx::m5::KernelAlignmentClass::full,
    };
    const auto selected = dispatch.prepare(key, context.capabilities().profile);
    if (selected.candidate.executor_key != plan.info().executor_key) {
      throw std::runtime_error("installed QMV executor was not selected");
    }

    std::vector<std::uint16_t> input_words;
    const auto input = make_input(object.shape[1], input_words);
    auto input_buffer = context.upload(bytes(input_words));
    auto output_buffer = context.make_buffer(plan.info().output_bytes);
    context.seal_warmup();
    std::vector<double> wall_ms;
    std::vector<double> gpu_ms;
    wall_ms.reserve(iterations);
    gpu_ms.reserve(iterations);
    for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
      const auto submission = omlx::m5::execute_dense_qmv_t1(
          context, plan, input_buffer, 0, output_buffer, 0);
      wall_ms.push_back(submission.timing.wall_ms);
      gpu_ms.push_back(submission.timing.gpu_ms);
    }
    const auto raw_output = output_buffer.read();
    std::vector<float> all_actual(object.shape[0]);
    for (std::uint64_t row = 0; row < object.shape[0]; ++row) {
      std::uint16_t word = 0;
      std::memcpy(&word, raw_output.data() + row * 2, sizeof(word));
      all_actual[row] = omlx::m5::bf16_to_float(word);
    }
    if (!std::ranges::all_of(
            all_actual, [](float value) { return std::isfinite(value); })) {
      throw std::runtime_error("real QMV output contains a non-finite value");
    }
    const auto selected_rows = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(sample_rows, object.shape[0]));
    std::vector<std::uint64_t> rows;
    rows.reserve(selected_rows);
    for (std::uint32_t index = 0; index < selected_rows; ++index) {
      const auto row = selected_rows == 1 ? 0
                                          : index * (object.shape[0] - 1) /
                                                (selected_rows - 1);
      if (rows.empty() || rows.back() != row) {
        rows.push_back(row);
      }
    }
    std::vector<float> actual;
    std::vector<float> expected;
    actual.reserve(rows.size());
    expected.reserve(rows.size());
    nlohmann::json samples = nlohmann::json::array();
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    for (const auto row : rows) {
      const auto weights =
          artifact.decode_weight_row(object, row, 0, object.shape[1]);
      double sum = 0.0;
      for (std::uint64_t column = 0; column < object.shape[1]; ++column) {
        sum += static_cast<double>(input[column]) * weights[column];
      }
      actual.push_back(all_actual[row]);
      expected.push_back(static_cast<float>(sum));
      samples.push_back({{"row", row}, {"actual", all_actual[row]}});
      const auto error = std::abs(static_cast<double>(all_actual[row]) - sum);
      max_abs_error = std::max(max_abs_error, error);
      max_rel_error =
          std::max(max_rel_error, error / std::max(1.0, std::abs(sum)));
    }
    if (max_rel_error > 0.02) {
      throw std::runtime_error("real QMV output exceeds FP64/BF16 tolerance");
    }
    const auto registry = context.registry_stats();
    const nlohmann::json result = {
        {"schema", "omlx-m5-b2-2-qmv-t1-v1"},
        {"evidence_scope",
         "M4 correctness and latency; not M5 Ultra performance"},
        {"source_repo", artifact.summary().source_repo},
        {"source_revision", artifact.summary().source_revision},
        {"object", object.name},
        {"format", object.format},
        {"layout", object.layout},
        {"output_features", object.shape[0]},
        {"input_features", object.shape[1]},
        {"padded_input_features", object.padded_k},
        {"executor_key", plan.info().executor_key},
        {"pipeline_key", plan.info().pipeline_key},
        {"function", plan.info().function_name},
        {"dispatch_count", plan.info().dispatch_count},
        {"workspace_bytes", plan.info().workspace_bytes},
        {"installed_candidate_count",
         dispatch.stats().installed_candidate_count},
        {"selected_candidate", selected.candidate.id},
        {"iterations", iterations},
        {"max_abs_error", max_abs_error},
        {"max_rel_error", max_rel_error},
        {"cosine", cosine(actual, expected)},
        {"wall_ms", wall_ms},
        {"gpu_ms", gpu_ms},
        {"weight_plane_bytes", plan.info().weight_plane_bytes},
        {"registry",
         {
             {"pipelines_compiled", registry.pipelines_compiled},
             {"command_plans_built", registry.command_plans_built},
             {"dispatches_submitted", registry.dispatches_submitted},
             {"upload_count", registry.upload_count},
             {"uploaded_bytes", registry.uploaded_bytes},
         }},
        {"sample_count", rows.size()},
        {"samples", std::move(samples)},
    };
    std::cout << result.dump(2) << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "QMV T=1 inspection failed: " << error.what() << '\n';
    return 1;
  }
}
