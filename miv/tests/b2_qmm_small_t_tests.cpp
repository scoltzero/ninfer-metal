#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "omlx_m5/qmm_small_t.h"
#include "omlx_m5/quantization.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const std::exception &) {
    return;
  }
  throw std::runtime_error(message);
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

struct Fixture {
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  omlx::m5::MaterializedWeightView view;
  std::vector<float> expected;
};

void add_plane(Fixture &fixture, omlx::m5::MetalContext &context,
               std::string kind, std::vector<std::byte> data,
               std::uint64_t row_stride) {
  fixture.buffers.push_back(
      std::make_unique<omlx::m5::MetalBuffer>(context.upload(data)));
  fixture.view.planes.push_back({
      .kind = std::move(kind),
      .buffer = fixture.buffers.back().get(),
      .segment_index = static_cast<std::uint32_t>(fixture.buffers.size() - 1),
      .offset = 0,
      .bytes = data.size(),
      .row_stride = row_stride,
  });
}

std::vector<float> make_input(std::uint32_t tokens, std::uint32_t k,
                              std::vector<std::uint16_t> &words) {
  std::vector<float> input(tokens * k);
  words.resize(tokens * k);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto raw =
          static_cast<float>(
              static_cast<std::int32_t>((column * 17U + token * 31U) % 257U) -
              128) /
          512.0F;
      const auto index = token * k + column;
      words[index] = omlx::m5::float_to_bf16(raw);
      input[index] = omlx::m5::bf16_to_float(words[index]);
    }
  }
  return input;
}

Fixture affine_fixture(omlx::m5::MetalContext &context, std::uint32_t bits,
                       std::uint32_t group_size, std::uint32_t tokens,
                       std::span<const float> input) {
  constexpr std::uint32_t n = 5;
  constexpr std::uint32_t k = 256;
  Fixture fixture;
  fixture.buffers.reserve(4);
  fixture.view = {
      .name = "fixture.affine.small_t",
      .format = "oq_affine_q" + std::to_string(bits) + "_g" +
                std::to_string(group_size) + "_bf16_v1",
      .layout = "oq_rowsplit_k128_v1",
      .parent_shape = {n, k},
      .padded_k = k,
      .row_count = n,
      .in_features = k,
  };
  std::vector<std::uint8_t> base(n * (bits == 8 ? k : k / 2));
  std::vector<std::uint8_t> high;
  if (bits == 5) {
    high.resize(n * k / 8);
  } else if (bits == 6) {
    high.resize(n * k / 4);
  }
  const auto groups = k / group_size;
  std::vector<std::uint16_t> scales(n * groups);
  std::vector<std::uint16_t> biases(n * groups);
  std::vector<std::uint32_t> codes(n * k);
  fixture.expected.resize(tokens * n);
  for (std::uint32_t row = 0; row < n; ++row) {
    for (std::uint32_t group = 0; group < groups; ++group) {
      scales[row * groups + group] =
          omlx::m5::float_to_bf16(0.015625F * float(row + group + 1));
      biases[row * groups + group] =
          omlx::m5::float_to_bf16(-0.03125F * float(row + 1));
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto mask = (1U << bits) - 1U;
      const auto code = (row * 29U + column * 11U) & mask;
      codes[row * k + column] = code;
      if (bits == 8) {
        base[row * k + column] = static_cast<std::uint8_t>(code);
      } else {
        const auto index = row * (k / 2) + column / 2;
        base[index] |=
            static_cast<std::uint8_t>((code & 0x0FU) << ((column & 1U) * 4U));
        if (bits == 5) {
          high[row * (k / 8) + column / 8] |=
              static_cast<std::uint8_t>(((code >> 4U) & 1U) << (column & 7U));
        } else if (bits == 6) {
          high[row * (k / 4) + column / 4] |= static_cast<std::uint8_t>(
              ((code >> 4U) & 3U) << ((column & 3U) * 2U));
        }
      }
    }
    for (std::uint32_t token = 0; token < tokens; ++token) {
      double sum = 0.0;
      for (std::uint32_t column = 0; column < k; ++column) {
        const auto group = column / group_size;
        const auto scale =
            omlx::m5::bf16_to_float(scales[row * groups + group]);
        const auto bias = omlx::m5::bf16_to_float(biases[row * groups + group]);
        sum +=
            static_cast<double>(input[token * k + column]) *
            std::fma(scale, static_cast<float>(codes[row * k + column]), bias);
      }
      fixture.expected[token * n + row] = static_cast<float>(sum);
    }
  }
  const auto base_stride = bits == 8 ? k : k / 2;
  add_plane(fixture, context, bits == 8 ? "code" : "base_code", bytes(base),
            base_stride);
  if (bits == 5) {
    add_plane(fixture, context, "high_bit", bytes(high), k / 8);
  } else if (bits == 6) {
    add_plane(fixture, context, "high_2bit", bytes(high), k / 4);
  }
  add_plane(fixture, context, "scale", bytes(scales), groups * 2);
  add_plane(fixture, context, "bias", bytes(biases), groups * 2);
  return fixture;
}

omlx::m5::MetalCapabilityProfile capabilities() {
  omlx::m5::MetalCapabilityProbe probe;
  probe.classic_command_queue = true;
  probe.shared_event = true;
  return omlx::m5::select_metal_capability_profile(std::move(probe));
}

void check_output(const omlx::m5::MetalBuffer &output,
                  std::span<const float> expected) {
  const auto output_bytes = output.read();
  expect(output_bytes.size() == expected.size() * 2,
         "small-T output size is wrong");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    std::uint16_t word = 0;
    std::memcpy(&word, output_bytes.data() + index * 2, sizeof(word));
    const auto actual = omlx::m5::bf16_to_float(word);
    const auto tolerance = 0.02F * std::max(1.0F, std::abs(expected[index]));
    if (!std::isfinite(actual) ||
        std::abs(actual - expected[index]) > tolerance) {
      throw std::runtime_error(
          "small-T Metal result exceeds FP64/BF16 tolerance");
    }
  }
}

void run_case(omlx::m5::MetalContext &context, std::uint32_t bits,
              std::uint32_t group_size, std::uint32_t tokens) {
  std::vector<std::uint16_t> input_words;
  const auto input = make_input(tokens, 256, input_words);
  auto fixture = affine_fixture(context, bits, group_size, tokens, input);
  auto plan = omlx::m5::prepare_dense_oq_small_t(
      context, OMLX_M5_QMM_SMALL_T_METALLIB_PATH, fixture.view, tokens);
  const auto candidates = plan.dispatch_candidates();
  const auto installed = plan.installed_executor_keys();
  expect(candidates.size() == installed.size() * 2,
         "small-T full/tail candidate count is wrong");
  expect(plan.info().group_sum_workspace_bytes ==
             tokens * (256 / group_size) * 4,
         "small-T group-sum workspace ledger is wrong");

  omlx::m5::DispatchTable table;
  for (const auto &candidate : candidates) {
    table.register_candidate(candidate);
  }
  table.seal(installed);
  expect(table.stats().planned_candidate_count == 0 &&
             table.stats().installed_candidate_count == candidates.size(),
         "small-T installed Gate is wrong");
  omlx::m5::KernelKey key{
      .format_id = plan.info().format_id,
      .layout_id = plan.info().layout_id,
      .activation_policy = omlx::m5::ActivationPolicy::a16_only,
      .preferred_activation = omlx::m5::ActivationProfile::a16,
      .input_type = omlx::m5::KernelDataType::bf16,
      .output_type = omlx::m5::KernelDataType::bf16,
      .accumulator_type = omlx::m5::KernelDataType::fp32,
      .geometry = omlx::m5::KernelGeometry::dense,
      .output_features = plan.info().output_features,
      .input_features = plan.info().input_features,
      .tokens = tokens,
      .fused_op = omlx::m5::KernelFusedOp::linear,
      .alignment = omlx::m5::KernelAlignmentClass::full,
  };
  const auto selected = table.prepare(key, capabilities());
  const auto expected_family =
      tokens >= 9 ? omlx::m5::DenseOqSmallTFamily::cooperative
                  : ((bits == 5 || bits == 6) && tokens <= 6
                         ? omlx::m5::DenseOqSmallTFamily::split_k
                         : omlx::m5::DenseOqSmallTFamily::row_split);
  expect(plan.executor(selected.candidate.executor_key).family ==
             expected_family,
         "small-T priority selected the wrong family");
  key.alignment = omlx::m5::KernelAlignmentClass::tail;
  const auto tail = table.prepare(key, capabilities());
  expect(tail.candidate.tail_safe &&
             plan.executor(tail.candidate.executor_key).family ==
                 expected_family,
         "small-T tail route selected the wrong family");

  auto input_buffer = context.upload(bytes(input_words));
  for (const auto &executor : plan.info().executors) {
    if (executor.family == omlx::m5::DenseOqSmallTFamily::split_k) {
      expect(executor.partial_workspace_bytes ==
                 plan.info().output_features * 4 * plan.info().capacity * 4,
             "split-K workspace ledger is wrong");
      expect(executor.dispatch_count == 3, "split-K dispatch ledger is wrong");
    } else {
      expect(executor.partial_workspace_bytes == 0 &&
                 executor.dispatch_count == 2,
             "row/cooperative dispatch ledger is wrong");
    }
    if (executor.family == omlx::m5::DenseOqSmallTFamily::cooperative) {
      expect(executor.static_threadgroup_memory_bytes >= 12 * 128 * 2,
             "cooperative activation tile is not static threadgroup memory");
    }
    auto output = context.make_buffer(plan.info().output_bytes);
    const auto before = context.registry_stats();
    const auto first = omlx::m5::execute_dense_oq_small_t(
        context, plan, executor.executor_key, input_buffer, 0, output, 0);
    const auto middle = context.registry_stats();
    const auto second = omlx::m5::execute_dense_oq_small_t(
        context, plan, executor.executor_key, input_buffer, 0, output, 0);
    const auto after = context.registry_stats();
    expect(first.dispatch_count == executor.dispatch_count &&
               second.dispatch_count == executor.dispatch_count,
           "small-T submitted dispatch count is wrong");
    expect(before.pipelines_compiled == middle.pipelines_compiled &&
               middle.pipelines_compiled == after.pipelines_compiled &&
               before.upload_count == middle.upload_count &&
               middle.upload_count == after.upload_count,
           "small-T token loop compiled a pipeline or uploaded a resource");
    check_output(output, fixture.expected);
  }
}

void run_fail_closed() {
  auto context = omlx::m5::MetalContext::create();
  std::vector<std::uint16_t> input_words;
  const auto input = make_input(2, 256, input_words);
  auto fixture = affine_fixture(context, 5, 64, 2, input);
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::prepare_dense_oq_small_t(
            context, OMLX_M5_QMM_SMALL_T_METALLIB_PATH, fixture.view, 1));
      },
      "small-T accepted T=1");
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::prepare_dense_oq_small_t(
            context, OMLX_M5_QMM_SMALL_T_METALLIB_PATH, fixture.view, 13));
      },
      "small-T accepted T=13");
  auto invalid = fixture.view;
  invalid.format = "mxfp4_e2m1_g32_e8m0_v1";
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::prepare_dense_oq_small_t(
            context, OMLX_M5_QMM_SMALL_T_METALLIB_PATH, invalid, 2));
      },
      "small-T accepted a non-affine format");

  auto plan = omlx::m5::prepare_dense_oq_small_t(
      context, OMLX_M5_QMM_SMALL_T_METALLIB_PATH, fixture.view, 2);
  auto input_buffer = context.upload(bytes(input_words));
  auto short_input = context.make_buffer(plan.info().input_bytes - 2);
  auto output = context.make_buffer(plan.info().output_bytes + 2);
  const auto executor = plan.info().executors.front().executor_key;
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::execute_dense_oq_small_t(
            context, plan, "not-installed", input_buffer, 0, output, 0));
      },
      "small-T accepted an unknown executor");
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::execute_dense_oq_small_t(
            context, plan, executor, short_input, 0, output, 0));
      },
      "small-T accepted an undersized input buffer");
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::execute_dense_oq_small_t(
            context, plan, executor, input_buffer, 0, output, 1));
      },
      "small-T accepted a misaligned BF16 output");
}

} // namespace

int main() {
  try {
    auto context = omlx::m5::MetalContext::create();
    for (const auto bits : {4U, 5U, 6U, 8U}) {
      for (const auto group : {64U, 128U}) {
        for (std::uint32_t tokens = 2; tokens <= 12; ++tokens) {
          run_case(context, bits, group, tokens);
        }
      }
    }
    run_fail_closed();
    std::cout << "B2.3 Dense oQ T=2..12 tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B2.3 Dense oQ T=2..12 failure: " << error.what() << '\n';
    return 1;
  }
}
