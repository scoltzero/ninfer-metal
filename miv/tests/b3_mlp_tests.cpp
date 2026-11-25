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
#include <vector>

#include "omlx_m5/mlp.h"
#include "omlx_m5/mxfp4_small_t.h"
#include "omlx_m5/projection.h"
#include "omlx_m5/qmv.h"
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

std::vector<std::uint16_t> read_words(const omlx::m5::MetalBuffer &buffer,
                                      std::uint64_t offset,
                                      std::uint64_t count) {
  const auto raw = buffer.read(offset, count * 2);
  std::vector<std::uint16_t> result(count);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

omlx::m5::QwenModelConfig dense_config() {
  omlx::m5::QwenModelConfig config;
  config.hidden_size = 128;
  config.intermediate_size = 128;
  return config;
}

struct DownFixture {
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  omlx::m5::MaterializedWeightView view;
  std::vector<float> weights;
};

DownFixture make_down_fixture(omlx::m5::MetalContext &context) {
  constexpr std::uint32_t n = 4;
  constexpr std::uint32_t k = 128;
  DownFixture fixture;
  fixture.view = {
      .name = "layer.mlp.down_proj",
      .format = "oq_affine_q4_g64_bf16_v1",
      .layout = "oq_rowsplit_k128_v1",
      .parent_shape = {n, k},
      .padded_k = k,
      .row_count = n,
      .in_features = k,
  };
  std::vector<std::uint8_t> codes(n * k / 2);
  std::vector<std::uint16_t> scales(n * 2);
  std::vector<std::uint16_t> biases(n * 2);
  fixture.weights.resize(n * k);
  for (std::uint32_t row = 0; row < n; ++row) {
    for (std::uint32_t group = 0; group < 2; ++group) {
      scales[row * 2 + group] =
          omlx::m5::float_to_bf16(0.015625F * float(row + group + 1));
      biases[row * 2 + group] =
          omlx::m5::float_to_bf16(-0.03125F * float(row + 1));
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (row * 7U + column * 5U) & 0x0FU;
      codes[row * (k / 2) + column / 2] |=
          static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
      fixture.weights[row * k + column] = std::fma(
          omlx::m5::bf16_to_float(scales[row * 2 + column / 64]), float(code),
          omlx::m5::bf16_to_float(biases[row * 2 + column / 64]));
    }
  }
  const auto add = [&](std::string kind, auto data, std::uint64_t stride) {
    fixture.buffers.push_back(
        std::make_unique<omlx::m5::MetalBuffer>(context.upload(bytes(data))));
    fixture.view.planes.push_back({
        .kind = std::move(kind),
        .buffer = fixture.buffers.back().get(),
        .bytes = bytes(data).size(),
        .row_stride = stride,
    });
  };
  add("base_code", codes, 64);
  add("scale", scales, 4);
  add("bias", biases, 4);
  return fixture;
}

void test_dense_swiglu_and_down() {
  constexpr std::uint32_t tokens = 3;
  constexpr std::uint32_t width = 128;
  auto context = omlx::m5::MetalContext::create();
  const auto bundle =
      omlx::m5::make_dense_mlp_projection_bundle(dense_config(), "layer.mlp");
  expect(bundle.parents.size() == 2 && bundle.parents[0].role == "mlp_gate" &&
             bundle.parents[1].role == "mlp_up",
         "dense MLP projection bundle is wrong");
  omlx::m5::ProjectionOutputArena arena(context, bundle, tokens);
  const auto gate = arena.parent("mlp_gate");
  const auto up = arena.parent("mlp_up");
  std::vector<std::uint16_t> gate_words(tokens * width);
  std::vector<std::uint16_t> up_words(tokens * width);
  std::vector<float> expected(tokens * width);
  for (std::uint32_t row = 0; row < tokens; ++row) {
    for (std::uint32_t feature = 0; feature < width; ++feature) {
      const auto gate_raw =
          static_cast<float>(
              static_cast<std::int32_t>((feature * 17U + row * 13U) % 257U) -
              128) /
          128.0F;
      const auto up_raw =
          static_cast<float>(
              static_cast<std::int32_t>((feature * 23U + row * 29U) % 257U) -
              128) /
          256.0F;
      const auto index = row * width + feature;
      gate_words[index] = omlx::m5::float_to_bf16(gate_raw);
      up_words[index] = omlx::m5::float_to_bf16(up_raw);
      const auto gate_value = omlx::m5::bf16_to_float(gate_words[index]);
      const auto up_value = omlx::m5::bf16_to_float(up_words[index]);
      expected[index] = gate_value / (1.0F + std::exp(-gate_value)) * up_value;
    }
  }
  arena.buffer().write(bytes(gate_words), gate.offset);
  arena.buffer().write(bytes(up_words), up.offset);
  auto output_a16 = context.make_buffer(tokens * width * 2);
  auto output_codes = context.make_buffer(tokens * width / 2);
  auto output_scales = context.make_buffer(tokens * width / 32);
  const omlx::m5::SwiGluActivationView output{
      .a16 = omlx::m5::ActivationPlaneView{.buffer = &output_a16,
                                           .bytes = tokens * width * 2},
      .a4 =
          omlx::m5::Mxfp4ActivationView{
              .codes = {.buffer = &output_codes, .bytes = tokens * width / 2},
              .scales = {.buffer = &output_scales,
                         .bytes = tokens * width / 32},
              .group_size = 32,
          },
      .rows = tokens,
      .features = width,
  };
  auto plan = omlx::m5::prepare_swiglu(context, OMLX_M5_MLP_METALLIB_PATH,
                                       tokens, width);
  const auto before = context.registry_stats();
  const auto submission =
      omlx::m5::produce_swiglu_activation(context, plan, gate, up, output);
  const auto after = context.registry_stats();
  expect(submission.dispatch_count == 1 &&
             after.upload_count == before.upload_count &&
             after.pipelines_compiled == before.pipelines_compiled,
         "SwiGLU changed token-loop resources");
  const auto actual_words = read_words(output_a16, 0, tokens * width);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto actual = omlx::m5::bf16_to_float(actual_words[index]);
    expect(std::isfinite(actual) &&
               std::abs(actual - expected[index]) <=
                   0.01F * std::max(1.0F, std::abs(expected[index])),
           "SwiGLU A16 differs from reference");
  }

  auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
      context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, width, tokens);
  auto baseline_codes = context.make_buffer(tokens * width / 2);
  auto baseline_scales = context.make_buffer(tokens * width / 32);
  const omlx::m5::Mxfp4ActivationView baseline{
      .codes = {.buffer = &baseline_codes, .bytes = tokens * width / 2},
      .scales = {.buffer = &baseline_scales, .bytes = tokens * width / 32},
      .group_size = 32,
  };
  static_cast<void>(
      omlx::m5::quantize_mxfp4_a4(context, quantizer, output_a16, 0, baseline));
  expect(output_codes.read() == baseline_codes.read() &&
             output_scales.read() == baseline_scales.read(),
         "fused SwiGLU A4 differs from the independent quantizer");
  auto a4_only_codes = context.make_buffer(tokens * width / 2);
  auto a4_only_scales = context.make_buffer(tokens * width / 32);
  const omlx::m5::SwiGluActivationView a4_only{
      .a4 =
          omlx::m5::Mxfp4ActivationView{
              .codes = {.buffer = &a4_only_codes, .bytes = tokens * width / 2},
              .scales = {.buffer = &a4_only_scales,
                         .bytes = tokens * width / 32},
              .group_size = 32,
          },
      .rows = tokens,
      .features = width,
  };
  static_cast<void>(
      omlx::m5::produce_swiglu_activation(context, plan, gate, up, a4_only));
  expect(a4_only_codes.read() == baseline_codes.read() &&
             a4_only_scales.read() == baseline_scales.read(),
         "A4-only SwiGLU differs from the independent quantizer");

  auto down_fixture = make_down_fixture(context);
  auto down_plan = omlx::m5::prepare_dense_qmv_t1(
      context, OMLX_M5_QMV_T1_METALLIB_PATH, down_fixture.view);
  auto down_output = context.make_buffer(down_plan.info().output_bytes);
  static_cast<void>(omlx::m5::execute_dense_qmv_t1(
      context, down_plan, output_a16, 0, down_output, 0));
  const auto down_words = read_words(down_output, 0, 4);
  for (std::uint32_t row = 0; row < 4; ++row) {
    double wanted = 0.0;
    for (std::uint32_t column = 0; column < width; ++column) {
      wanted +=
          static_cast<double>(omlx::m5::bf16_to_float(actual_words[column])) *
          down_fixture.weights[row * width + column];
    }
    const auto actual = omlx::m5::bf16_to_float(down_words[row]);
    expect(std::abs(actual - wanted) <= 0.02 * std::max(1.0, std::abs(wanted)),
           "SwiGLU output did not feed the down projection");
  }

  auto invalid = output;
  invalid.rows = 2;
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::produce_swiglu_activation(
            context, plan, gate, up, invalid));
      },
      "SwiGLU accepted a mismatched output shape");
}

void test_moe_topk_reduce() {
  constexpr std::uint32_t tokens = 2;
  constexpr std::uint32_t top_k = 8;
  constexpr std::uint32_t features = 64;
  auto context = omlx::m5::MetalContext::create();
  std::vector<std::uint16_t> input_words(tokens * top_k * features);
  std::vector<float> expected(tokens * features, 0.0F);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t slot = 0; slot < top_k; ++slot) {
      for (std::uint32_t feature = 0; feature < features; ++feature) {
        const auto raw = static_cast<float>((token + 1) * (slot + 1)) / 64.0F +
                         static_cast<float>(feature % 7) / 128.0F;
        const auto index = (token * top_k + slot) * features + feature;
        input_words[index] = omlx::m5::float_to_bf16(raw);
        expected[token * features + feature] +=
            omlx::m5::bf16_to_float(input_words[index]);
      }
    }
  }
  auto input = context.upload(bytes(input_words));
  auto output = context.make_buffer(tokens * features * 2);
  auto plan = omlx::m5::prepare_moe_topk_reduce(
      context, OMLX_M5_MLP_METALLIB_PATH, tokens, features, top_k);
  const auto submission =
      omlx::m5::execute_moe_topk_reduce(context, plan, input, 0, output, 0);
  expect(submission.dispatch_count == 1, "MoE top-k reduce dispatch count");
  const auto words = read_words(output, 0, tokens * features);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto actual = omlx::m5::bf16_to_float(words[index]);
    expect(std::abs(actual - expected[index]) <=
               0.01F * std::max(1.0F, std::abs(expected[index])),
           "MoE top-k reduce differs from reference");
  }

  std::vector<std::uint16_t> shared_words(tokens * features);
  std::vector<std::uint16_t> gate_words(tokens);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    gate_words[token] = omlx::m5::float_to_bf16(0.25F * float(token + 1));
    for (std::uint32_t feature = 0; feature < features; ++feature) {
      shared_words[token * features + feature] =
          omlx::m5::float_to_bf16(0.01F * float((token + feature) % 11));
    }
  }
  auto shared = context.upload(bytes(shared_words));
  auto shared_gate = context.upload(bytes(gate_words));
  auto combined = context.make_buffer(tokens * features * 2);
  auto combine_plan = omlx::m5::prepare_moe_shared_combine(
      context, OMLX_M5_MLP_METALLIB_PATH, tokens, features);
  static_cast<void>(omlx::m5::execute_moe_shared_combine(
      context, combine_plan, output, 0, shared, 0, shared_gate, 0, combined,
      0));
  const auto combined_words = read_words(combined, 0, tokens * features);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto gate = omlx::m5::bf16_to_float(gate_words[token]);
    const auto scale = 1.0F / (1.0F + std::exp(-gate));
    for (std::uint32_t feature = 0; feature < features; ++feature) {
      const auto index = token * features + feature;
      const auto wanted = omlx::m5::bf16_to_float(words[index]) +
                          scale * omlx::m5::bf16_to_float(shared_words[index]);
      const auto actual = omlx::m5::bf16_to_float(combined_words[index]);
      expect(std::abs(actual - wanted) <=
                 0.01F * std::max(1.0F, std::abs(wanted)),
             "MoE shared expert combine differs from reference");
    }
  }
  // A zero router logit admits half of the shared expert, not zero (SiLU).
  output.write(std::vector<std::byte>(output.size()));
  shared_gate.write(std::vector<std::byte>(shared_gate.size()));
  std::fill(shared_words.begin(), shared_words.end(),
            omlx::m5::float_to_bf16(2.0F));
  shared.write(bytes(shared_words));
  static_cast<void>(omlx::m5::execute_moe_shared_combine(
      context, combine_plan, output, 0, shared, 0, shared_gate, 0, combined,
      0));
  const auto half_shared = read_words(combined, 0, tokens * features);
  expect(std::ranges::all_of(
             half_shared,
             [](auto value) { return omlx::m5::bf16_to_float(value) == 1.0F; }),
         "zero shared-expert gate must retain half of its output");
}

} // namespace

int main() {
  try {
    test_dense_swiglu_and_down();
    test_moe_topk_reduce();
    std::cout << "B3.5 MLP/SwiGLU/down tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B3.5 MLP failure: " << error.what() << '\n';
    return 1;
  }
}
