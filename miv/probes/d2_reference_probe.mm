#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/metal.h"

#ifndef OMLX_M5_D2_METALLIB_PATH
#define OMLX_M5_D2_METALLIB_PATH ""
#endif

namespace {

template <typename T>
std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

std::vector<std::byte> constant_u32(std::uint32_t value) {
  std::vector<std::byte> result(sizeof(value));
  std::memcpy(result.data(), &value, sizeof(value));
  return result;
}

std::vector<std::uint8_t> pack_codes(const std::vector<std::uint32_t> &codes,
                                     std::uint32_t bits) {
  std::vector<std::uint8_t> packed((codes.size() * bits + 7) / 8, 0);
  for (std::size_t index = 0; index < codes.size(); ++index) {
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
      if ((codes[index] & (1U << bit)) != 0U) {
        const auto absolute = index * bits + bit;
        packed[absolute / 8] |= static_cast<std::uint8_t>(1U << (absolute % 8));
      }
    }
  }
  return packed;
}

std::vector<float> cpu_qmm(const std::vector<float> &input,
                           const std::vector<std::uint32_t> &codes,
                           const std::vector<float> &scales,
                           const std::vector<float> &biases, std::uint32_t m,
                           std::uint32_t n, std::uint32_t k) {
  const auto groups = k / 64;
  std::vector<float> output(static_cast<std::size_t>(m) * n);
  for (std::uint32_t row = 0; row < m; ++row) {
    for (std::uint32_t out = 0; out < n; ++out) {
      float sum = 0.0F;
      for (std::uint32_t column = 0; column < k; ++column) {
        const auto group = column / 64;
        const auto weight =
            std::fma(scales[out * groups + group],
                     static_cast<float>(codes[out * k + column]),
                     biases[out * groups + group]);
        sum = std::fma(input[row * k + column], weight, sum);
      }
      output[row * n + out] = sum;
    }
  }
  return output;
}

float run_case(omlx::m5::MetalContext &context, std::uint32_t bits,
               std::uint32_t m, std::uint32_t n, std::uint32_t k) {
  const auto pipeline_key = "d2.reference.q" + std::to_string(bits);
  const auto function_name =
      bits == 4 ? "omlx_m5_reference_qmm_q4" : "omlx_m5_reference_qmm_q5";
  const auto pipeline = context.warmup_pipeline({
      .key = pipeline_key,
      .library_key = "d2.reference.library",
      .function_name = function_name,
      .variant = "metal4-direct",
  });

  std::vector<float> input(static_cast<std::size_t>(m) * k);
  std::vector<std::uint32_t> codes(static_cast<std::size_t>(n) * k);
  const auto groups = k / 64;
  std::vector<float> scales(static_cast<std::size_t>(n) * groups);
  std::vector<float> biases(scales.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] =
        static_cast<float>(static_cast<int>(index % 13) - 6) * 0.03125F;
  }
  for (std::size_t index = 0; index < codes.size(); ++index) {
    codes[index] =
        static_cast<std::uint32_t>((index * 7 + index / k) % (1U << bits));
  }
  for (std::size_t index = 0; index < scales.size(); ++index) {
    scales[index] = 0.03125F * static_cast<float>((index % 5) + 1);
    biases[index] = -0.125F + 0.03125F * static_cast<float>(index % 3);
  }
  const auto packed = pack_codes(codes, bits);
  const auto expected = cpu_qmm(input, codes, scales, biases, m, n, k);
  std::vector<float> zeros(expected.size(), 0.0F);
  auto input_buffer = context.upload(bytes(input));
  auto weight_buffer = context.upload(bytes(packed));
  auto scale_buffer = context.upload(bytes(scales));
  auto bias_buffer = context.upload(bytes(biases));
  auto output_buffer = context.upload(bytes(zeros));

  const auto plan = context
                        .prepare_command_plan({
                            .key = pipeline_key + "." + std::to_string(m) +
                                   "." + std::to_string(n) + "." +
                                   std::to_string(k),
                            .dispatches = {{
                                .pipeline_key = pipeline.descriptor.key,
                                .grid = {.width = n, .height = m},
                                .threads_per_threadgroup =
                                    {
                                        .width = std::min<std::uint32_t>(n, 16),
                                        .height = std::min<std::uint32_t>(m, 8),
                                    },
                                .resources =
                                    {
                                        {.argument_index = 0,
                                         .resource_slot = 0,
                                         .minimum_bytes =
                                             input.size() * sizeof(float)},
                                        {.argument_index = 1,
                                         .resource_slot = 1,
                                         .minimum_bytes = packed.size()},
                                        {.argument_index = 2,
                                         .resource_slot = 2,
                                         .minimum_bytes = scales.size() *
                                                          sizeof(float)},
                                        {.argument_index = 3,
                                         .resource_slot = 3,
                                         .minimum_bytes = biases.size() *
                                                          sizeof(float)},
                                        {.argument_index = 4,
                                         .resource_slot = 4,
                                         .minimum_bytes =
                                             zeros.size() * sizeof(float)},
                                    },
                                .constant_arguments =
                                    {
                                        {.argument_index = 5,
                                         .bytes = constant_u32(m)},
                                        {.argument_index = 6,
                                         .bytes = constant_u32(n)},
                                        {.argument_index = 7,
                                         .bytes = constant_u32(k)},
                                    },
                            }},
                        });
  const std::array bindings = {
      omlx::m5::MetalBufferBinding{0, &input_buffer, 0},
      omlx::m5::MetalBufferBinding{1, &weight_buffer, 0},
      omlx::m5::MetalBufferBinding{2, &scale_buffer, 0},
      omlx::m5::MetalBufferBinding{3, &bias_buffer, 0},
      omlx::m5::MetalBufferBinding{4, &output_buffer, 0},
  };
  static_cast<void>(context.execute(plan, bindings));
  const auto output_bytes = output_buffer.read();
  std::vector<float> actual(expected.size());
  std::memcpy(actual.data(), output_bytes.data(), output_bytes.size());
  float maximum_error = 0.0F;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    maximum_error =
        std::max(maximum_error, std::abs(actual[index] - expected[index]));
  }
  if (maximum_error > 2.0e-5F) {
    throw std::runtime_error("D2 Metal 4 qmm exceeded FP32 tolerance");
  }
  return maximum_error;
}

} // namespace

int main() {
  try {
    auto context = omlx::m5::MetalContext::create();
    context.register_library({
        .key = "d2.reference.library",
        .path = OMLX_M5_D2_METALLIB_PATH,
    });
    float maximum_error = 0.0F;
    std::uint32_t cases = 0;
    for (const auto bits : {4U, 5U}) {
      for (const auto rows : {1U, 2U, 4U}) {
        maximum_error =
            std::max(maximum_error, run_case(context, bits, rows, 11, 128));
        ++cases;
      }
    }
    const auto stats = context.registry_stats();
    std::cout << "{\n"
              << "  \"schema_version\": 2,\n"
              << "  \"probe\": \"d2_reference_qmm_metal4\",\n"
              << "  \"scope\": \"M4 Metal 4 correctness\",\n"
              << "  \"device_name\": \"" << context.capabilities().device_name
              << "\",\n"
              << "  \"cases\": " << cases << ",\n"
              << "  \"bits\": [4, 5],\n"
              << "  \"m_values\": [1, 2, 4],\n"
              << "  \"metal4_command_buffers\": "
              << stats.metal4_command_buffers_submitted << ",\n"
              << "  \"classic_command_buffers\": "
              << stats.classic_command_buffers_submitted << ",\n"
              << "  \"maximum_absolute_error\": " << maximum_error << "\n"
              << "}\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "omlx_m5_d2_reference_probe: " << error.what() << '\n';
    return 1;
  }
}
