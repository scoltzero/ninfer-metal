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
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <nlohmann/json.hpp>

#include "omlx_m5/artifact.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/qmm_small_t.h"
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
  if (width > std::numeric_limits<std::uint32_t>::max()) {
    throw std::overflow_error("small-T input width exceeds uint32");
  }
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

double cosine(std::span<const float> left, std::span<const float> right) {
  if (left.size() != right.size()) {
    throw std::invalid_argument("cosine vector sizes differ");
  }
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
    throw std::logic_error(std::string(label) + " registry counter regressed");
  }
  return after - before;
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 4) {
      throw std::invalid_argument(
          "usage: omlx_m5_inspect_qmm_small_t ARTIFACT OBJECT METALLIB "
          "[--iterations N] [--sample-rows N]");
    }
    std::uint32_t iterations = 2;
    std::uint32_t sample_rows = 8;
    for (int index = 4; index < argc; ++index) {
      const std::string_view argument(argv[index]);
      if (argument == "--iterations" && index + 1 < argc) {
        iterations = parse_u32(argv[++index], "iterations");
      } else if (argument == "--sample-rows" && index + 1 < argc) {
        sample_rows = parse_u32(argv[++index], "sample rows");
      } else {
        throw std::invalid_argument("unknown small-T argument: " +
                                    std::string(argument));
      }
    }

    const auto artifact = omlx::m5::ArtifactModel::open(argv[1], false);
    const auto &object = artifact.object(argv[2]);
    if (object.shape.size() != 2 || !object.format.starts_with("oq_affine_q")) {
      throw std::invalid_argument(
          "small-T inspector requires a rank-2 affine oQ object");
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
    std::uint64_t weight_plane_bytes = 0;
    for (const auto &plane : object.planes) {
      weight_plane_bytes += plane.bytes;
    }

    const auto selected_row_count = static_cast<std::uint32_t>(
        std::min<std::uint64_t>(sample_rows, object.shape[0]));
    std::vector<std::uint64_t> rows;
    rows.reserve(selected_row_count);
    for (std::uint32_t index = 0; index < selected_row_count; ++index) {
      const auto row = selected_row_count == 1 ? 0
                                               : index * (object.shape[0] - 1) /
                                                     (selected_row_count - 1);
      if (rows.empty() || rows.back() != row) {
        rows.push_back(row);
      }
    }
    std::vector<std::vector<float>> decoded_rows;
    decoded_rows.reserve(rows.size());
    for (const auto row : rows) {
      decoded_rows.push_back(
          artifact.decode_weight_row(object, row, 0, object.shape[1]));
    }

    nlohmann::json records = nlohmann::json::array();
    for (std::uint32_t tokens = 2; tokens <= 12; ++tokens) {
      const auto before_prepare = context.registry_stats();
      auto plan =
          omlx::m5::prepare_dense_oq_small_t(context, argv[3], weight, tokens);
      const auto candidates = plan.dispatch_candidates();
      omlx::m5::DispatchTable dispatch;
      for (const auto &candidate : candidates) {
        dispatch.register_candidate(candidate);
      }
      const auto installed = plan.installed_executor_keys();
      dispatch.seal(installed);
      omlx::m5::KernelKey key{
          .format_id = object.format,
          .layout_id = object.layout,
          .activation_policy = omlx::m5::ActivationPolicy::a16_only,
          .preferred_activation = omlx::m5::ActivationProfile::a16,
          .input_type = omlx::m5::KernelDataType::bf16,
          .output_type = omlx::m5::KernelDataType::bf16,
          .accumulator_type = omlx::m5::KernelDataType::fp32,
          .geometry = omlx::m5::KernelGeometry::dense,
          .output_features = object.shape[0],
          .input_features = object.shape[1],
          .tokens = tokens,
          .fused_op = omlx::m5::KernelFusedOp::linear,
          .alignment = omlx::m5::KernelAlignmentClass::full,
      };
      const auto selected =
          dispatch.prepare(key, context.capabilities().profile);
      const auto &executor = plan.executor(selected.candidate.executor_key);
      const auto after_prepare = context.registry_stats();

      std::vector<std::uint16_t> input_words;
      const auto input = make_input(tokens, object.shape[1], input_words);
      auto input_buffer = context.upload(bytes(input_words));
      auto output_buffer = context.make_buffer(plan.info().output_bytes);
      const auto before_execute = context.registry_stats();
      std::vector<double> wall_ms;
      std::vector<double> gpu_ms;
      wall_ms.reserve(iterations);
      gpu_ms.reserve(iterations);
      for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
        const auto submission = omlx::m5::execute_dense_oq_small_t(
            context, plan, selected.candidate.executor_key, input_buffer, 0,
            output_buffer, 0);
        wall_ms.push_back(submission.timing.wall_ms);
        gpu_ms.push_back(submission.timing.gpu_ms);
      }
      const auto selected_raw_output = output_buffer.read();
      nlohmann::json candidate_timings = nlohmann::json::array();
      candidate_timings.push_back({
          {"family", omlx::m5::to_string(executor.family)},
          {"executor_key", executor.executor_key},
          {"selected", true},
          {"bit_exact_selected", true},
          {"wall_ms", wall_ms},
          {"gpu_ms", gpu_ms},
      });
      for (const auto &alternative : plan.info().executors) {
        if (alternative.executor_key == executor.executor_key) {
          continue;
        }
        std::vector<double> alternative_wall_ms;
        std::vector<double> alternative_gpu_ms;
        alternative_wall_ms.reserve(iterations);
        alternative_gpu_ms.reserve(iterations);
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
          const auto submission = omlx::m5::execute_dense_oq_small_t(
              context, plan, alternative.executor_key, input_buffer, 0,
              output_buffer, 0);
          alternative_wall_ms.push_back(submission.timing.wall_ms);
          alternative_gpu_ms.push_back(submission.timing.gpu_ms);
        }
        candidate_timings.push_back({
            {"family", omlx::m5::to_string(alternative.family)},
            {"executor_key", alternative.executor_key},
            {"selected", false},
            {"bit_exact_selected", output_buffer.read() == selected_raw_output},
            {"wall_ms", std::move(alternative_wall_ms)},
            {"gpu_ms", std::move(alternative_gpu_ms)},
        });
      }
      const auto after_execute = context.registry_stats();
      if (after_execute.pipelines_compiled !=
              before_execute.pipelines_compiled ||
          after_execute.upload_count != before_execute.upload_count) {
        throw std::runtime_error(
            "small-T token loop compiled a pipeline or uploaded a resource");
      }

      const auto &raw_output = selected_raw_output;
      std::vector<float> actual;
      std::vector<float> expected;
      actual.reserve(tokens * rows.size());
      expected.reserve(tokens * rows.size());
      nlohmann::json samples = nlohmann::json::array();
      double max_abs_error = 0.0;
      double max_rel_error = 0.0;
      for (std::uint32_t token = 0; token < tokens; ++token) {
        for (std::size_t sample = 0; sample < rows.size(); ++sample) {
          const auto row = rows[sample];
          std::uint16_t word = 0;
          const auto output_index = token * object.shape[0] + row;
          std::memcpy(&word, raw_output.data() + output_index * 2,
                      sizeof(word));
          const auto got = omlx::m5::bf16_to_float(word);
          double reference = 0.0;
          for (std::uint64_t column = 0; column < object.shape[1]; ++column) {
            reference +=
                static_cast<double>(input[token * object.shape[1] + column]) *
                decoded_rows[sample][column];
          }
          if (!std::isfinite(got)) {
            throw std::runtime_error("small-T output is non-finite");
          }
          const auto error = std::abs(static_cast<double>(got) - reference);
          max_abs_error = std::max(max_abs_error, error);
          max_rel_error = std::max(max_rel_error,
                                   error / std::max(1.0, std::abs(reference)));
          actual.push_back(got);
          expected.push_back(static_cast<float>(reference));
          samples.push_back({
              {"token", token},
              {"row", row},
              {"actual", got},
              {"cpp_expected", reference},
          });
        }
      }
      if (max_rel_error > 0.02) {
        throw std::runtime_error(
            "real small-T output exceeds FP64/BF16 tolerance");
      }
      nlohmann::json executor_records = nlohmann::json::array();
      for (const auto &value : plan.info().executors) {
        executor_records.push_back({
            {"family", omlx::m5::to_string(value.family)},
            {"executor_key", value.executor_key},
            {"function", value.function_name},
            {"capacity", value.capacity},
            {"dispatch_count", value.dispatch_count},
            {"partial_workspace_bytes", value.partial_workspace_bytes},
            {"static_threadgroup_memory_bytes",
             value.static_threadgroup_memory_bytes},
        });
      }
      records.push_back({
          {"tokens", tokens},
          {"capacity", plan.info().capacity},
          {"selected_candidate", selected.candidate.id},
          {"selected_executor", selected.candidate.executor_key},
          {"selected_family", omlx::m5::to_string(executor.family)},
          {"installed_candidate_count",
           dispatch.stats().installed_candidate_count},
          {"executor_count", plan.info().executors.size()},
          {"executors", std::move(executor_records)},
          {"dispatch_count", executor.dispatch_count},
          {"group_sum_workspace_bytes", plan.info().group_sum_workspace_bytes},
          {"partial_workspace_bytes", executor.partial_workspace_bytes},
          {"input_bytes", plan.info().input_bytes},
          {"output_bytes", plan.info().output_bytes},
          {"pipelines_compiled_during_prepare",
           delta(after_prepare.pipelines_compiled,
                 before_prepare.pipelines_compiled, "pipeline")},
          {"command_plans_built_during_prepare",
           delta(after_prepare.command_plans_built,
                 before_prepare.command_plans_built, "command plan")},
          {"token_loop_pipeline_compiles",
           delta(after_execute.pipelines_compiled,
                 before_execute.pipelines_compiled, "token pipeline")},
          {"token_loop_uploads",
           delta(after_execute.upload_count, before_execute.upload_count,
                 "token upload")},
          {"iterations", iterations},
          {"wall_ms", wall_ms},
          {"gpu_ms", gpu_ms},
          {"candidate_timings", std::move(candidate_timings)},
          {"max_abs_error", max_abs_error},
          {"max_rel_error", max_rel_error},
          {"cosine", cosine(actual, expected)},
          {"sample_count", samples.size()},
          {"samples", std::move(samples)},
      });
    }

    const auto registry = context.registry_stats();
    const nlohmann::json result = {
        {"schema", "omlx-m5-b2-3-qmm-small-t-v1"},
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
    std::cerr << "QMM small-T inspection failed: " << error.what() << '\n';
    return 1;
  }
}
