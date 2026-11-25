#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

float decode_e2m1(std::uint32_t code) {
  constexpr std::array magnitude = {0.0F, 0.5F, 1.0F, 1.5F,
                                    2.0F, 3.0F, 4.0F, 6.0F};
  const auto value = magnitude[code & 7U];
  return (code & 8U) == 0 ? value : -value;
}

float decode_e8m0(std::uint8_t code) {
  return std::ldexp(1.0F, static_cast<int>(code) - 127);
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

std::vector<float> make_input(std::uint32_t k,
                              std::vector<std::uint16_t> &words) {
  std::vector<float> input(k);
  words.resize(k);
  for (std::uint32_t column = 0; column < k; ++column) {
    const auto raw =
        static_cast<float>(static_cast<std::int32_t>((column * 17U) % 257U) -
                           128) /
        512.0F;
    words[column] = omlx::m5::float_to_bf16(raw);
    input[column] = omlx::m5::bf16_to_float(words[column]);
  }
  return input;
}

Fixture affine_fixture(omlx::m5::MetalContext &context, std::uint32_t bits,
                       std::uint32_t group_size, std::span<const float> input) {
  constexpr std::uint32_t n = 3;
  constexpr std::uint32_t k = 256;
  Fixture fixture;
  fixture.buffers.reserve(4);
  fixture.view = {
      .name = "fixture.affine",
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
  fixture.expected.resize(n);
  for (std::uint32_t row = 0; row < n; ++row) {
    double sum = 0.0;
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
      const auto group = column / group_size;
      const auto scale = omlx::m5::bf16_to_float(scales[row * groups + group]);
      const auto bias = omlx::m5::bf16_to_float(biases[row * groups + group]);
      sum += static_cast<double>(input[column]) *
             std::fma(scale, static_cast<float>(code), bias);
    }
    fixture.expected[row] = static_cast<float>(sum);
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

Fixture mxfp4_fixture(omlx::m5::MetalContext &context,
                      std::span<const float> input) {
  constexpr std::uint32_t n = 3;
  constexpr std::uint32_t k = 256;
  constexpr std::uint32_t group_size = 32;
  Fixture fixture;
  fixture.buffers.reserve(2);
  fixture.view = {
      .name = "fixture.mxfp4",
      .format = "mxfp4_e2m1_g32_e8m0_v1",
      .layout = "mx_rowsplit_k128_v1",
      .parent_shape = {n, k},
      .padded_k = k,
      .row_count = n,
      .in_features = k,
  };
  std::vector<std::uint8_t> data(n * k / 2);
  std::vector<std::uint8_t> scales(n * k / group_size);
  fixture.expected.resize(n);
  for (std::uint32_t row = 0; row < n; ++row) {
    double sum = 0.0;
    for (std::uint32_t group = 0; group < k / group_size; ++group) {
      scales[row * (k / group_size) + group] =
          static_cast<std::uint8_t>(125U + (row + group) % 4U);
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (row * 7U + column * 5U) & 0x0FU;
      data[row * (k / 2) + column / 2] |=
          static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
      const auto scale =
          decode_e8m0(scales[row * (k / group_size) + column / group_size]);
      sum += static_cast<double>(input[column]) * decode_e2m1(code) * scale;
    }
    fixture.expected[row] = static_cast<float>(sum);
  }
  add_plane(fixture, context, "data", bytes(data), k / 2);
  add_plane(fixture, context, "scale", bytes(scales), k / group_size);
  return fixture;
}

void run_case(std::uint32_t bits, std::uint32_t group_size) {
  auto context = omlx::m5::MetalContext::create();
  std::vector<std::uint16_t> input_words;
  const auto input = make_input(256, input_words);
  auto fixture = bits == 0 ? mxfp4_fixture(context, input)
                           : affine_fixture(context, bits, group_size, input);
  auto plan = omlx::m5::prepare_dense_qmv_t1(
      context, OMLX_M5_QMV_T1_METALLIB_PATH, fixture.view);
  const auto candidates = plan.dispatch_candidates();
  omlx::m5::DispatchTable table;
  for (const auto &value : candidates) {
    table.register_candidate(value);
  }
  const std::array installed = {plan.info().executor_key};
  table.seal(installed);
  omlx::m5::MetalCapabilityProbe probe;
  probe.classic_command_queue = true;
  probe.shared_event = true;
  const auto capabilities =
      omlx::m5::select_metal_capability_profile(std::move(probe));
  omlx::m5::KernelKey key{
      .format_id = plan.info().format_id,
      .layout_id = plan.info().layout_id,
      .activation_policy = bits == 0 ? omlx::m5::ActivationPolicy::allow_a4
                                     : omlx::m5::ActivationPolicy::a16_only,
      .preferred_activation = omlx::m5::ActivationProfile::a16,
      .input_type = omlx::m5::KernelDataType::bf16,
      .output_type = omlx::m5::KernelDataType::bf16,
      .accumulator_type = omlx::m5::KernelDataType::fp32,
      .geometry = omlx::m5::KernelGeometry::dense,
      .output_features = plan.info().output_features,
      .input_features = plan.info().input_features,
      .tokens = 1,
      .fused_op = omlx::m5::KernelFusedOp::linear,
      .alignment = omlx::m5::KernelAlignmentClass::full,
  };
  const auto dispatch = table.prepare(key, capabilities);
  expect(dispatch.candidate.executor_key == plan.info().executor_key,
         "installed QMV executor was not selected");

  auto input_buffer = context.upload(bytes(input_words));
  auto output_buffer = context.make_buffer(plan.info().output_bytes);
  context.seal_warmup();
  const auto submission = omlx::m5::execute_dense_qmv_t1(
      context, plan, input_buffer, 0, output_buffer, 0);
  expect(submission.dispatch_count == plan.info().dispatch_count,
         "QMV submitted an unexpected dispatch count");
  const auto output_bytes = output_buffer.read();
  expect(output_bytes.size() == fixture.expected.size() * 2,
         "QMV output size is wrong");
  for (std::size_t row = 0; row < fixture.expected.size(); ++row) {
    std::uint16_t word = 0;
    std::memcpy(&word, output_bytes.data() + row * 2, sizeof(word));
    const auto actual = omlx::m5::bf16_to_float(word);
    const auto expected = fixture.expected[row];
    const auto tolerance = 0.02F * std::max(1.0F, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      throw std::runtime_error("QMV Metal result exceeds FP64/BF16 tolerance");
    }
  }
}

void run_bf16_padded_k_case() {
  auto context = omlx::m5::MetalContext::create();
  constexpr std::uint32_t n = 4;
  constexpr std::uint32_t logical_k = 96;
  constexpr std::uint32_t padded_k = 128;
  std::vector<std::uint16_t> input_words;
  const auto input = make_input(logical_k, input_words);

  Fixture fixture;
  fixture.view = {
      .name = "fixture.bf16.padded",
      .format = "direct_bf16_v1",
      .layout = "contiguous_le_v1",
      .parent_shape = {n, logical_k},
      .padded_k = padded_k,
      .row_count = n,
      .in_features = logical_k,
  };
  std::vector<std::uint16_t> weights(n * padded_k,
                                     omlx::m5::float_to_bf16(50.0F));
  fixture.expected.resize(n);
  for (std::uint32_t row = 0; row < n; ++row) {
    double sum = 0.0;
    for (std::uint32_t col = 0; col < logical_k; ++col) {
      const float raw =
          float(static_cast<std::int32_t>((row * 29U + col * 13U) % 199U) -
                99) /
          128.0F;
      weights[row * padded_k + col] = omlx::m5::float_to_bf16(raw);
      sum += static_cast<double>(input[col]) *
             omlx::m5::bf16_to_float(weights[row * padded_k + col]);
    }
    fixture.expected[row] = static_cast<float>(sum);
  }
  add_plane(fixture, context, "data", bytes(weights), padded_k * 2);

  auto plan = omlx::m5::prepare_dense_qmv_t1(
      context, OMLX_M5_QMV_T1_METALLIB_PATH, fixture.view);
  auto input_buffer = context.upload(bytes(input_words));
  auto output_buffer = context.make_buffer(plan.info().output_bytes);
  context.seal_warmup();
  static_cast<void>(omlx::m5::execute_dense_qmv_t1(
      context, plan, input_buffer, 0, output_buffer, 0));
  const auto output_bytes = output_buffer.read();
  expect(output_bytes.size() == n * 2, "padded BF16 QMV output size is wrong");
  for (std::uint32_t row = 0; row < n; ++row) {
    std::uint16_t word = 0;
    std::memcpy(&word, output_bytes.data() + row * 2, sizeof(word));
    const auto actual = omlx::m5::bf16_to_float(word);
    const auto expected = fixture.expected[row];
    const auto tolerance = 0.02F * std::max(1.0F, std::abs(expected));
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
      throw std::runtime_error(
          "padded BF16 QMV result diverged (ignored padded_input_features)");
    }
  }
}

} // namespace

int main() {
  try {
    for (const auto bits : {4U, 5U, 6U, 8U}) {
      for (const auto group : {64U, 128U}) {
        run_case(bits, group);
      }
    }
    run_case(0, 32);
    run_bf16_padded_k_case();
    std::cout << "B2.2 Dense T=1 QMV tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B2.2 Dense T=1 QMV failure: " << error.what() << '\n';
    return 1;
  }
}
