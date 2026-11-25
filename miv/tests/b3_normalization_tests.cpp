#include <algorithm>
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

#include "omlx_m5/mxfp4_small_t.h"
#include "omlx_m5/normalization.h"
#include "omlx_m5/qmv.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/reference.h"

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

std::vector<omlx::m5::ActivationUse>
projection_uses(omlx::m5::ActivationPolicy policy) {
  return {
      {.binding_name = "layer.q_proj", .role = "attention_q", .policy = policy},
      {.binding_name = "layer.k_proj", .role = "attention_k", .policy = policy},
      {.binding_name = "layer.v_proj", .role = "attention_v", .policy = policy},
      {.binding_name = "layer.gate_proj", .role = "mlp_gate", .policy = policy},
      {.binding_name = "layer.up_proj", .role = "mlp_up", .policy = policy},
  };
}

struct Inputs {
  std::vector<std::uint16_t> input_words;
  std::vector<std::uint16_t> weight_words;
  std::vector<float> input;
  std::vector<float> weight;
};

Inputs make_inputs(std::uint32_t tokens, std::uint32_t width) {
  Inputs result;
  result.input_words.resize(tokens * width);
  result.weight_words.resize(width);
  result.input.resize(tokens * width);
  result.weight.resize(width);
  for (std::uint32_t column = 0; column < width; ++column) {
    const auto raw_weight = 0.75F + float(column % 17) / 32.0F;
    result.weight_words[column] = omlx::m5::float_to_bf16(raw_weight);
    result.weight[column] =
        omlx::m5::bf16_to_float(result.weight_words[column]);
  }
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t column = 0; column < width; ++column) {
      const auto raw =
          static_cast<float>(
              static_cast<std::int32_t>((column * 19U + token * 23U) % 257U) -
              128) /
          256.0F;
      const auto index = token * width + column;
      result.input_words[index] = omlx::m5::float_to_bf16(raw);
      result.input[index] = omlx::m5::bf16_to_float(result.input_words[index]);
    }
  }
  return result;
}

std::vector<std::uint16_t> read_words(const omlx::m5::MetalBuffer &buffer,
                                      std::uint64_t offset,
                                      std::uint64_t count) {
  const auto raw = buffer.read(offset, count * 2);
  std::vector<std::uint16_t> result(count);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

void test_a16_a4_modes_and_reuse() {
  constexpr std::uint32_t tokens = 3;
  constexpr std::uint32_t width = 128;
  constexpr float epsilon = 1.0e-6F;
  auto context = omlx::m5::MetalContext::create();
  const auto inputs = make_inputs(tokens, width);
  auto input = context.upload(bytes(inputs.input_words));
  auto weight = context.upload(bytes(inputs.weight_words));
  auto plan = omlx::m5::prepare_rmsnorm_projection(
      context, OMLX_M5_RMSNORM_PROJECTION_METALLIB_PATH, width, tokens,
      epsilon);
  const auto expected =
      omlx::m5::rms_norm(inputs.input, tokens, width, inputs.weight, epsilon);

  const std::array dual_slots = {omlx::m5::ActivationSlotSpec{
      .max_input_features = width,
      .max_tokens = tokens,
      .policy = omlx::m5::ActivationPolicy::allow_a4,
  }};
  omlx::m5::ActivationWorkspace dual_workspace(
      context, omlx::m5::plan_activation_workspace(dual_slots));
  dual_workspace.begin_epoch(1);
  const auto uses = projection_uses(omlx::m5::ActivationPolicy::allow_a4);
  const omlx::m5::ActivationReuseKey key{
      .source_id = 11,
      .normalization_id = 17,
      .input_features = width,
      .tokens = tokens,
  };
  const auto dual_lease =
      dual_workspace.acquire(key, uses, omlx::m5::ActivationProfile::a4, true);
  const auto dual = omlx::m5::produce_rmsnorm_activation(
      context, plan, input, 0, weight, 0, dual_lease);
  expect(dual.mode == omlx::m5::RmsNormProducerMode::a16_a4 &&
             dual.submission.has_value() &&
             dual.submission->dispatch_count == 1,
         "dual RMSNorm producer did not use one dispatch");
  const auto dual_a16 =
      read_words(dual_workspace.buffer(),
                 dual_lease.bundle.normalized_a16->offset, tokens * width);
  for (std::size_t index = 0; index < expected.size(); ++index) {
    const auto actual = omlx::m5::bf16_to_float(dual_a16[index]);
    expect(std::abs(actual - expected[index]) <=
               0.01F * std::max(1.0F, std::abs(expected[index])),
           "RMSNorm A16 differs from FP32 reference");
  }

  const std::array a16_slots = {omlx::m5::ActivationSlotSpec{
      .max_input_features = width,
      .max_tokens = tokens,
      .policy = omlx::m5::ActivationPolicy::a16_only,
  }};
  omlx::m5::ActivationWorkspace a16_workspace(
      context, omlx::m5::plan_activation_workspace(a16_slots));
  a16_workspace.begin_epoch(1);
  const auto a16_lease = a16_workspace.acquire(
      key, projection_uses(omlx::m5::ActivationPolicy::a16_only),
      omlx::m5::ActivationProfile::a16);
  const auto a16 = omlx::m5::produce_rmsnorm_activation(context, plan, input, 0,
                                                        weight, 0, a16_lease);
  expect(a16.mode == omlx::m5::RmsNormProducerMode::a16,
         "A16-only RMSNorm selected the wrong mode");
  const auto baseline_a16 =
      read_words(a16_workspace.buffer(),
                 a16_lease.bundle.normalized_a16->offset, tokens * width);
  expect(dual_a16 == baseline_a16,
         "dual and A16-only RMSNorm outputs differ bitwise");

  auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
      context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, width, tokens);
  auto baseline_codes = context.make_buffer(quantizer.info().code_bytes);
  auto baseline_scales = context.make_buffer(quantizer.info().scale_bytes);
  const omlx::m5::Mxfp4ActivationView baseline_a4{
      .codes = {.buffer = &baseline_codes,
                .bytes = quantizer.info().code_bytes},
      .scales = {.buffer = &baseline_scales,
                 .bytes = quantizer.info().scale_bytes},
      .group_size = 32,
  };
  static_cast<void>(omlx::m5::quantize_mxfp4_a4(
      context, quantizer, a16_workspace.buffer(),
      a16_lease.bundle.normalized_a16->offset, baseline_a4));
  expect(dual_workspace.buffer().read(dual_lease.bundle.a4->codes.offset,
                                      quantizer.info().code_bytes) ==
                 baseline_codes.read() &&
             dual_workspace.buffer().read(dual_lease.bundle.a4->scales.offset,
                                          quantizer.info().scale_bytes) ==
                 baseline_scales.read(),
         "fused RMSNorm+A4 differs from the unfused quantizer bitwise");

  omlx::m5::ActivationWorkspace a4_workspace(
      context, omlx::m5::plan_activation_workspace(dual_slots));
  a4_workspace.begin_epoch(1);
  const auto a4_lease =
      a4_workspace.acquire(key, uses, omlx::m5::ActivationProfile::a4, false);
  const auto a4 = omlx::m5::produce_rmsnorm_activation(context, plan, input, 0,
                                                       weight, 0, a4_lease);
  expect(a4.mode == omlx::m5::RmsNormProducerMode::a4 &&
             a4_workspace.buffer().read(a4_lease.bundle.a4->codes.offset,
                                        quantizer.info().code_bytes) ==
                 baseline_codes.read() &&
             a4_workspace.buffer().read(a4_lease.bundle.a4->scales.offset,
                                        quantizer.info().scale_bytes) ==
                 baseline_scales.read(),
         "A4-only RMSNorm producer differs from baseline");

  const auto before_reuse = context.registry_stats();
  const auto reused_lease =
      dual_workspace.acquire(key, uses, omlx::m5::ActivationProfile::a4, true);
  const auto reused = omlx::m5::produce_rmsnorm_activation(
      context, plan, input, 0, weight, 0, reused_lease);
  const auto after_reuse = context.registry_stats();
  expect(reused.mode == omlx::m5::RmsNormProducerMode::none &&
             !reused.submission.has_value() &&
             before_reuse.dispatches_submitted ==
                 after_reuse.dispatches_submitted &&
             reused_lease.bundle.normalized_a16->offset ==
                 dual_lease.bundle.normalized_a16->offset &&
             reused_lease.bundle.a4->codes.offset ==
                 dual_lease.bundle.a4->codes.offset,
         "activation cache reuse reran RMSNorm or changed views");
  const auto stats = dual_workspace.stats();
  expect(stats.cross_projection_reuses >= uses.size() &&
             stats.avoided_a4_quantizations > 0,
         "cross-projection A4 reuse was not accounted");
}

struct Q4Fixture {
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  omlx::m5::MaterializedWeightView view;
  std::vector<float> weights;
};

Q4Fixture make_q4_fixture(omlx::m5::MetalContext &context) {
  constexpr std::uint32_t n = 4;
  constexpr std::uint32_t k = 128;
  Q4Fixture fixture;
  fixture.view = {
      .name = "projection.parent",
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
      const auto group = column / 64;
      fixture.weights[row * k + column] = std::fma(
          omlx::m5::bf16_to_float(scales[row * 2 + group]), float(code),
          omlx::m5::bf16_to_float(biases[row * 2 + group]));
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

void test_rmsnorm_to_parent_linear() {
  constexpr std::uint32_t width = 128;
  auto context = omlx::m5::MetalContext::create();
  const auto inputs = make_inputs(1, width);
  auto input = context.upload(bytes(inputs.input_words));
  auto norm_weight = context.upload(bytes(inputs.weight_words));
  const std::array slots = {omlx::m5::ActivationSlotSpec{
      .max_input_features = width,
      .max_tokens = 1,
      .policy = omlx::m5::ActivationPolicy::a16_only,
  }};
  omlx::m5::ActivationWorkspace workspace(
      context, omlx::m5::plan_activation_workspace(slots));
  workspace.begin_epoch(1);
  const auto lease =
      workspace.acquire({.source_id = 31,
                         .normalization_id = 41,
                         .input_features = width,
                         .tokens = 1},
                        projection_uses(omlx::m5::ActivationPolicy::a16_only),
                        omlx::m5::ActivationProfile::a16);
  auto norm_plan = omlx::m5::prepare_rmsnorm_projection(
      context, OMLX_M5_RMSNORM_PROJECTION_METALLIB_PATH, width, 1, 1.0e-6F);
  static_cast<void>(omlx::m5::produce_rmsnorm_activation(
      context, norm_plan, input, 0, norm_weight, 0, lease));
  const auto expected_norm =
      omlx::m5::rms_norm(inputs.input, 1, width, inputs.weight, 1.0e-6F);
  auto fixture = make_q4_fixture(context);
  auto linear = omlx::m5::prepare_dense_qmv_t1(
      context, OMLX_M5_QMV_T1_METALLIB_PATH, fixture.view);
  auto output = context.make_buffer(linear.info().output_bytes);
  static_cast<void>(omlx::m5::execute_dense_qmv_t1(
      context, linear, *lease.bundle.normalized_a16->buffer,
      lease.bundle.normalized_a16->offset, output, 0));
  const auto raw = output.read();
  for (std::uint32_t row = 0; row < 4; ++row) {
    double expected = 0.0;
    for (std::uint32_t column = 0; column < width; ++column) {
      const auto represented = omlx::m5::bf16_to_float(
          omlx::m5::float_to_bf16(expected_norm[column]));
      expected += static_cast<double>(represented) *
                  fixture.weights[row * width + column];
    }
    std::uint16_t word = 0;
    std::memcpy(&word, raw.data() + row * 2, sizeof(word));
    const auto actual = omlx::m5::bf16_to_float(word);
    expect(std::abs(static_cast<double>(actual) - expected) <=
               0.02 * std::max(1.0, std::abs(expected)),
           "RMSNorm output did not feed parent Linear correctly");
  }
}

} // namespace

int main() {
  try {
    test_a16_a4_modes_and_reuse();
    test_rmsnorm_to_parent_linear();
    std::cout << "B3.2/B3.3 normalization tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B3.2/B3.3 normalization failure: " << error.what() << '\n';
    return 1;
  }
}
