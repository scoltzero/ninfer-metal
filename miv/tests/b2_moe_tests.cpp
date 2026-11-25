#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "omlx_m5/moe.h"
#include "omlx_m5/mxfp4_small_t.h"
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

struct Fixture {
  std::vector<std::unique_ptr<omlx::m5::MetalBuffer>> buffers;
  omlx::m5::MaterializedWeightView view;
  std::vector<float> weights;
  bool mxfp4 = false;
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
      .bytes = data.size(),
      .row_stride = row_stride,
  });
}

Fixture affine_fixture(omlx::m5::MetalContext &context, std::uint32_t bits,
                       std::uint32_t group_size) {
  constexpr std::uint32_t experts = 256;
  constexpr std::uint32_t n = 3;
  constexpr std::uint32_t k = 128;
  Fixture fixture;
  fixture.view = {
      .name = "fixture.moe.affine",
      .format = "oq_affine_q" + std::to_string(bits) + "_g" +
                std::to_string(group_size) + "_bf16_v1",
      .layout = "oq_expert_rowsplit_k128_v1",
      .parent_shape = {experts, n, k},
      .padded_k = k,
      .row_count = experts * n,
      .in_features = k,
  };
  const auto rows = experts * n;
  std::vector<std::uint8_t> base(rows * (bits == 8 ? k : k / 2));
  std::vector<std::uint8_t> high;
  if (bits == 5) {
    high.resize(rows * k / 8);
  } else if (bits == 6) {
    high.resize(rows * k / 4);
  }
  const auto groups = k / group_size;
  std::vector<std::uint16_t> scales(rows * groups);
  std::vector<std::uint16_t> biases(rows * groups);
  fixture.weights.resize(rows * k);
  for (std::uint32_t flat_row = 0; flat_row < rows; ++flat_row) {
    for (std::uint32_t group = 0; group < groups; ++group) {
      scales[flat_row * groups + group] = omlx::m5::float_to_bf16(
          0.0078125F * float((flat_row + group) % 7 + 1));
      biases[flat_row * groups + group] =
          omlx::m5::float_to_bf16(-0.015625F * float(flat_row % 5 + 1));
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (flat_row * 29U + column * 11U) & ((1U << bits) - 1U);
      if (bits == 8) {
        base[flat_row * k + column] = static_cast<std::uint8_t>(code);
      } else {
        base[flat_row * (k / 2) + column / 2] |=
            static_cast<std::uint8_t>((code & 0x0FU) << ((column & 1U) * 4U));
        if (bits == 5) {
          high[flat_row * (k / 8) + column / 8] |=
              static_cast<std::uint8_t>(((code >> 4U) & 1U) << (column & 7U));
        } else if (bits == 6) {
          high[flat_row * (k / 4) + column / 4] |= static_cast<std::uint8_t>(
              ((code >> 4U) & 3U) << ((column & 3U) * 2U));
        }
      }
      const auto group = column / group_size;
      fixture.weights[flat_row * k + column] =
          std::fma(omlx::m5::bf16_to_float(scales[flat_row * groups + group]),
                   static_cast<float>(code),
                   omlx::m5::bf16_to_float(biases[flat_row * groups + group]));
    }
  }
  add_plane(fixture, context, bits == 8 ? "code" : "base_code", bytes(base),
            bits == 8 ? k : k / 2);
  if (bits == 5) {
    add_plane(fixture, context, "high_bit", bytes(high), k / 8);
  } else if (bits == 6) {
    add_plane(fixture, context, "high_2bit", bytes(high), k / 4);
  }
  add_plane(fixture, context, "scale", bytes(scales), groups * 2);
  add_plane(fixture, context, "bias", bytes(biases), groups * 2);
  return fixture;
}

Fixture mxfp4_fixture(omlx::m5::MetalContext &context) {
  constexpr std::uint32_t experts = 256;
  constexpr std::uint32_t n = 3;
  constexpr std::uint32_t k = 128;
  Fixture fixture;
  fixture.mxfp4 = true;
  fixture.view = {
      .name = "fixture.moe.mxfp4",
      .format = "mxfp4_e2m1_g32_e8m0_v1",
      .layout = "mx_expert_rowsplit_k128_v1",
      .parent_shape = {experts, n, k},
      .padded_k = k,
      .row_count = experts * n,
      .in_features = k,
  };
  const auto rows = experts * n;
  std::vector<std::uint8_t> data(rows * k / 2);
  std::vector<std::uint8_t> scales(rows * k / 32);
  fixture.weights.resize(rows * k);
  for (std::uint32_t flat_row = 0; flat_row < rows; ++flat_row) {
    for (std::uint32_t group = 0; group < k / 32; ++group) {
      scales[flat_row * (k / 32) + group] =
          static_cast<std::uint8_t>(123U + (flat_row + group) % 8U);
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (flat_row * 7U + column * 5U) & 0x0FU;
      data[flat_row * (k / 2) + column / 2] |=
          static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
      fixture.weights[flat_row * k + column] =
          decode_e2m1(code) *
          decode_e8m0(scales[flat_row * (k / 32) + column / 32]);
    }
  }
  add_plane(fixture, context, "data", bytes(data), k / 2);
  add_plane(fixture, context, "scale", bytes(scales), k / 32);
  return fixture;
}

std::vector<float> make_input(std::uint32_t tokens,
                              std::vector<std::uint16_t> &words) {
  constexpr std::uint32_t k = 128;
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

struct QuantizedActivation {
  std::vector<std::uint8_t> codes;
  std::vector<std::uint8_t> scales;
  std::vector<float> values;
};

QuantizedActivation quantize_cpu(std::span<const float> input,
                                 std::uint32_t tokens) {
  constexpr std::uint32_t k = 128;
  QuantizedActivation result;
  result.codes.resize(tokens * k / 2);
  result.scales.resize(tokens * k / 32);
  result.values.resize(tokens * k);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t group = 0; group < k / 32; ++group) {
      float maximum = 0.0F;
      for (std::uint32_t lane = 0; lane < 32; ++lane) {
        maximum =
            std::max(maximum, std::abs(input[token * k + group * 32 + lane]));
      }
      const auto scale_code = encode_e8m0_round_up(maximum / 6.0F);
      const auto scale = decode_e8m0(scale_code);
      result.scales[token * (k / 32) + group] = scale_code;
      for (std::uint32_t lane = 0; lane < 32; ++lane) {
        const auto column = group * 32 + lane;
        const auto code = encode_e2m1(input[token * k + column] / scale);
        result.codes[token * (k / 2) + column / 2] |=
            static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
        result.values[token * k + column] = decode_e2m1(code) * scale;
      }
    }
  }
  return result;
}

struct Routes {
  std::vector<std::uint32_t> indices;
  std::vector<std::uint16_t> score_words;
  std::vector<float> scores;
};

Routes make_routes(std::uint32_t tokens) {
  Routes result;
  result.indices.resize(tokens * 8);
  result.score_words.resize(tokens * 8);
  result.scores.resize(tokens * 8);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    float total = 0.0F;
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      result.indices[token * 8 + slot] =
          (token * 37U + slot * 31U + (slot == 7 ? 255U : 0U)) % 256U;
      result.scores[token * 8 + slot] = float(slot + 1);
      total += result.scores[token * 8 + slot];
    }
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      const auto index = token * 8 + slot;
      result.score_words[index] =
          omlx::m5::float_to_bf16(result.scores[index] / total);
      result.scores[index] = omlx::m5::bf16_to_float(result.score_words[index]);
    }
  }
  return result;
}

std::vector<float> reference(const Fixture &fixture,
                             std::span<const float> input,
                             std::span<const float> scores,
                             std::span<const std::uint32_t> indices,
                             std::uint32_t tokens, bool apply_scores) {
  constexpr std::uint32_t n = 3;
  constexpr std::uint32_t k = 128;
  std::vector<float> output(tokens * 8 * n);
  for (std::uint32_t assignment = 0; assignment < tokens * 8; ++assignment) {
    const auto token = assignment / 8;
    const auto input_row = apply_scores ? assignment : token;
    const auto expert = indices[assignment];
    for (std::uint32_t row = 0; row < n; ++row) {
      double sum = 0.0;
      const auto flat_row = expert * n + row;
      for (std::uint32_t column = 0; column < k; ++column) {
        sum += static_cast<double>(input[input_row * k + column]) *
               fixture.weights[flat_row * k + column];
      }
      if (apply_scores) {
        sum *= scores[assignment];
      }
      output[assignment * n + row] = static_cast<float>(sum);
    }
  }
  return output;
}

void check_output(const omlx::m5::MetalBuffer &output,
                  std::span<const float> expected) {
  const auto raw = output.read();
  for (std::size_t index = 0; index < expected.size(); ++index) {
    std::uint16_t word = 0;
    std::memcpy(&word, raw.data() + index * 2, sizeof(word));
    const auto actual = omlx::m5::bf16_to_float(word);
    const auto tolerance = 0.02F * std::max(1.0F, std::abs(expected[index]));
    if (!std::isfinite(actual) ||
        std::abs(actual - expected[index]) > tolerance) {
      throw std::runtime_error("MoE expert output exceeds tolerance");
    }
  }
}

omlx::m5::KernelKey make_key(const omlx::m5::MoeExpertPlan &plan,
                             omlx::m5::ActivationProfile activation,
                             omlx::m5::KernelAlignmentClass alignment) {
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
      .geometry = omlx::m5::KernelGeometry::moe_expert,
      .output_features = plan.info().output_features,
      .input_features = plan.info().input_features,
      .tokens = plan.info().tokens,
      .experts = 256,
      .top_k = 8,
      .fused_op = plan.info().fused_op,
      .alignment = alignment,
  };
}

void run_router(omlx::m5::MetalContext &context, std::uint32_t tokens) {
  std::vector<std::uint16_t> words(tokens * 256);
  std::vector<float> probabilities(tokens * 256);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t expert = 0; expert < 256; ++expert) {
      const auto raw = float((expert * 13U + token * 7U) % 41U + 1U);
      words[token * 256 + expert] = omlx::m5::float_to_bf16(raw);
      probabilities[token * 256 + expert] =
          omlx::m5::bf16_to_float(words[token * 256 + expert]);
    }
  }
  auto input = context.upload(bytes(words));
  auto indices = context.make_buffer(tokens * 8 * 4);
  auto scores = context.make_buffer(tokens * 8 * 2);
  const omlx::m5::MoeRouteView route{
      .indices = {.buffer = &indices, .bytes = tokens * 8 * 4},
      .scores = {.buffer = &scores, .bytes = tokens * 8 * 2},
      .tokens = tokens,
  };
  auto plan = omlx::m5::prepare_moe_router_top8(
      context, OMLX_M5_MOE_METALLIB_PATH, tokens);
  static_cast<void>(
      omlx::m5::execute_moe_router_top8(context, plan, input, 0, route));
  const auto raw_indices = indices.read();
  const auto raw_scores = scores.read();
  for (std::uint32_t token = 0; token < tokens; ++token) {
    std::vector<std::uint32_t> order(256);
    std::iota(order.begin(), order.end(), 0);
    std::ranges::sort(order, [&](std::uint32_t left, std::uint32_t right) {
      const auto lhs = probabilities[token * 256 + left];
      const auto rhs = probabilities[token * 256 + right];
      return lhs != rhs ? lhs > rhs : left > right;
    });
    float total = 0.0F;
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      total += probabilities[token * 256 + order[slot]];
    }
    for (std::uint32_t slot = 0; slot < 8; ++slot) {
      std::uint32_t actual_index = 0;
      std::uint16_t actual_score_word = 0;
      std::memcpy(&actual_index, raw_indices.data() + (token * 8 + slot) * 4,
                  sizeof(actual_index));
      std::memcpy(&actual_score_word,
                  raw_scores.data() + (token * 8 + slot) * 2,
                  sizeof(actual_score_word));
      const auto expected_score =
          omlx::m5::bf16_to_float(omlx::m5::float_to_bf16(
              probabilities[token * 256 + order[slot]] / total));
      expect(actual_index == order[slot] &&
                 omlx::m5::bf16_to_float(actual_score_word) == expected_score,
             "MoE router top-8 differs from the composed oracle");
    }
  }
}

void run_expert(omlx::m5::MetalContext &context, Fixture &fixture,
                std::uint32_t tokens, bool apply_scores) {
  std::vector<std::uint16_t> input_words;
  const auto input_rows = apply_scores ? tokens * 8 : tokens;
  const auto input_values = make_input(input_rows, input_words);
  auto input = context.upload(bytes(input_words));
  const auto routes = make_routes(tokens);
  auto indices = context.upload(bytes(routes.indices));
  auto scores = context.upload(bytes(routes.score_words));
  const omlx::m5::MoeRouteView route{
      .indices = {.buffer = &indices, .bytes = tokens * 8 * 4},
      .scores = {.buffer = &scores, .bytes = tokens * 8 * 2},
      .tokens = tokens,
  };
  auto plan = omlx::m5::prepare_moe_expert(
      context, OMLX_M5_MOE_METALLIB_PATH, fixture.view, tokens,
      apply_scores ? omlx::m5::KernelFusedOp::moe_down
                   : omlx::m5::KernelFusedOp::moe_gate_up,
      apply_scores);
  expect(plan.info().input_rows == input_rows &&
             plan.info().assignment_input == apply_scores,
         "MoE gate/down input-row ownership is wrong");
  omlx::m5::DispatchTable table;
  for (const auto &candidate : plan.dispatch_candidates()) {
    table.register_candidate(candidate);
  }
  const auto installed = plan.installed_executor_keys();
  table.seal(installed);
  const auto grouped =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelAlignmentClass::full),
                    context.capabilities().profile);
  const auto fallback =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelAlignmentClass::tail),
                    context.capabilities().profile);
  expect(plan.executor(grouped.candidate.executor_key).kernel_family ==
                 omlx::m5::KernelFamily::grouped_moe &&
             plan.executor(fallback.candidate.executor_key).kernel_family ==
                 omlx::m5::KernelFamily::per_expert_fallback,
         "MoE grouped/fallback selection is wrong");
  auto output = context.make_buffer(plan.info().output_bytes);
  static_cast<void>(omlx::m5::execute_moe_expert_a16(
      context, plan, grouped.candidate.executor_key, input, 0, route, output,
      0));
  check_output(output, reference(fixture, input_values, routes.scores,
                                 routes.indices, tokens, apply_scores));
  static_cast<void>(omlx::m5::execute_moe_expert_a16(
      context, plan, fallback.candidate.executor_key, input, 0, route, output,
      0));
  check_output(output, reference(fixture, input_values, routes.scores,
                                 routes.indices, tokens, apply_scores));

  if (fixture.mxfp4) {
    const auto quantized = quantize_cpu(input_values, input_rows);
    auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
        context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, 128, input_rows);
    auto code_buffer = context.make_buffer(quantizer.info().code_bytes);
    auto scale_buffer = context.make_buffer(quantizer.info().scale_bytes);
    const omlx::m5::Mxfp4ActivationView activation{
        .codes = {.buffer = &code_buffer, .bytes = quantizer.info().code_bytes},
        .scales = {.buffer = &scale_buffer,
                   .bytes = quantizer.info().scale_bytes},
        .group_size = 32,
    };
    static_cast<void>(
        omlx::m5::quantize_mxfp4_a4(context, quantizer, input, 0, activation));
    const auto grouped_a4 =
        table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                               omlx::m5::KernelAlignmentClass::full),
                      context.capabilities().profile);
    const auto fallback_a4 =
        table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                               omlx::m5::KernelAlignmentClass::tail),
                      context.capabilities().profile);
    static_cast<void>(omlx::m5::execute_moe_expert_a4(
        context, plan, grouped_a4.candidate.executor_key, activation, route,
        output, 0));
    check_output(output, reference(fixture, quantized.values, routes.scores,
                                   routes.indices, tokens, apply_scores));
    static_cast<void>(omlx::m5::execute_moe_expert_a4(
        context, plan, fallback_a4.candidate.executor_key, activation, route,
        output, 0));
    check_output(output, reference(fixture, quantized.values, routes.scores,
                                   routes.indices, tokens, apply_scores));
  }
}

void check_catalog(omlx::m5::MetalContext &context, Fixture &fixture,
                   std::uint32_t tokens) {
  auto plan = omlx::m5::prepare_moe_expert(context, OMLX_M5_MOE_METALLIB_PATH,
                                           fixture.view, tokens);
  omlx::m5::DispatchTable table;
  for (const auto &candidate : plan.dispatch_candidates()) {
    table.register_candidate(candidate);
  }
  const auto installed = plan.installed_executor_keys();
  table.seal(installed);
  const auto full =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelAlignmentClass::full),
                    context.capabilities().profile);
  const auto tail =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelAlignmentClass::tail),
                    context.capabilities().profile);
  expect(plan.executor(full.candidate.executor_key).kernel_family ==
                 omlx::m5::KernelFamily::grouped_moe &&
             plan.executor(tail.candidate.executor_key).kernel_family ==
                 omlx::m5::KernelFamily::per_expert_fallback,
         "MoE all-T catalog selection is wrong");
  if (fixture.mxfp4) {
    static_cast<void>(
        table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                               omlx::m5::KernelAlignmentClass::full),
                      context.capabilities().profile));
    static_cast<void>(
        table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                               omlx::m5::KernelAlignmentClass::tail),
                      context.capabilities().profile));
  }
}

} // namespace

int main() {
  try {
    auto context = omlx::m5::MetalContext::create();
    for (const auto tokens : {1U, 8U, 13U, 128U}) {
      run_router(context, tokens);
    }
    std::vector<Fixture> fixtures;
    for (const auto bits : {4U, 5U, 6U, 8U}) {
      for (const auto group : {64U, 128U}) {
        fixtures.push_back(affine_fixture(context, bits, group));
      }
    }
    fixtures.push_back(mxfp4_fixture(context));
    for (std::uint32_t tokens = 1; tokens <= 128; ++tokens) {
      static_cast<void>(omlx::m5::prepare_moe_router_top8(
          context, OMLX_M5_MOE_METALLIB_PATH, tokens));
      for (auto &fixture : fixtures) {
        check_catalog(context, fixture, tokens);
      }
    }
    for (const auto tokens : {512U, 1024U}) {
      static_cast<void>(omlx::m5::prepare_moe_router_top8(
          context, OMLX_M5_MOE_METALLIB_PATH, tokens));
      for (auto &fixture : fixtures) {
        check_catalog(context, fixture, tokens);
      }
    }
    for (const auto tokens : {1U, 8U, 13U}) {
      for (auto &fixture : fixtures) {
        run_expert(context, fixture, tokens, tokens == 13);
      }
    }
    auto invalid = fixtures.front().view;
    invalid.parent_shape[0] = 255;
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_moe_expert(
              context, OMLX_M5_MOE_METALLIB_PATH, invalid, 1));
        },
        "MoE expert plan accepted 255 experts");
    std::cout << "B2.6 MoE router/expert tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B2.6 MoE router/expert failure: " << error.what() << '\n';
    return 1;
  }
}
