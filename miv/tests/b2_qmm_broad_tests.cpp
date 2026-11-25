#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "omlx_m5/mxfp4_small_t.h"
#include "omlx_m5/qmm_broad.h"
#include "omlx_m5/quantization.h"

#ifndef OMLX_M5_MXFP4_TENSOROPS_METALLIB_PATH
#define OMLX_M5_MXFP4_TENSOROPS_METALLIB_PATH ""
#endif

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
  constexpr std::uint32_t n = 5;
  constexpr std::uint32_t k = 256;
  Fixture fixture;
  fixture.view = {
      .name = "fixture.affine.broad",
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
  fixture.weights.resize(n * k);
  for (std::uint32_t row = 0; row < n; ++row) {
    for (std::uint32_t group = 0; group < groups; ++group) {
      scales[row * groups + group] =
          omlx::m5::float_to_bf16(0.015625F * float(row + group + 1));
      biases[row * groups + group] =
          omlx::m5::float_to_bf16(-0.03125F * float(row + 1));
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (row * 29U + column * 11U) & ((1U << bits) - 1U);
      if (bits == 8) {
        base[row * k + column] = static_cast<std::uint8_t>(code);
      } else {
        base[row * (k / 2) + column / 2] |=
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
      fixture.weights[row * k + column] =
          std::fma(omlx::m5::bf16_to_float(scales[row * groups + group]),
                   static_cast<float>(code),
                   omlx::m5::bf16_to_float(biases[row * groups + group]));
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
  constexpr std::uint32_t n = 5;
  constexpr std::uint32_t k = 256;
  Fixture fixture;
  fixture.mxfp4 = true;
  fixture.view = {
      .name = "fixture.mxfp4.broad",
      .format = "mxfp4_e2m1_g32_e8m0_v1",
      .layout = "mx_rowsplit_k128_v1",
      .parent_shape = {n, k},
      .padded_k = k,
      .row_count = n,
      .in_features = k,
  };
  std::vector<std::uint8_t> data(n * k / 2);
  std::vector<std::uint8_t> scales(n * k / 32);
  fixture.weights.resize(n * k);
  for (std::uint32_t row = 0; row < n; ++row) {
    for (std::uint32_t group = 0; group < k / 32; ++group) {
      scales[row * (k / 32) + group] =
          static_cast<std::uint8_t>(124U + (row + group) % 7U);
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (row * 7U + column * 5U) & 0x0FU;
      data[row * (k / 2) + column / 2] |=
          static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
      fixture.weights[row * k + column] =
          decode_e2m1(code) * decode_e8m0(scales[row * (k / 32) + column / 32]);
    }
  }
  add_plane(fixture, context, "data", bytes(data), k / 2);
  add_plane(fixture, context, "scale", bytes(scales), k / 32);
  return fixture;
}

std::vector<float> make_input(std::uint32_t tokens,
                              std::vector<std::uint16_t> &words) {
  constexpr std::uint32_t k = 256;
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
  constexpr std::uint32_t k = 256;
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

std::vector<float> reference(std::span<const float> input,
                             std::span<const float> weights,
                             std::uint32_t tokens) {
  constexpr std::uint32_t n = 5;
  constexpr std::uint32_t k = 256;
  std::vector<float> output(tokens * n);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t row = 0; row < n; ++row) {
      double sum = 0.0;
      for (std::uint32_t column = 0; column < k; ++column) {
        sum += static_cast<double>(input[token * k + column]) *
               weights[row * k + column];
      }
      output[token * n + row] = static_cast<float>(sum);
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
      throw std::runtime_error("broad QMM output exceeds tolerance");
    }
  }
}

omlx::m5::KernelKey make_key(const omlx::m5::DenseQmmBroadPlan &plan,
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
      .geometry = omlx::m5::KernelGeometry::dense,
      .output_features = plan.info().output_features,
      .input_features = plan.info().input_features,
      .tokens = plan.info().tokens,
      .fused_op = omlx::m5::KernelFusedOp::linear,
      .alignment = alignment,
  };
}

omlx::m5::MetalCapabilityProfile tensorops_profile() {
  omlx::m5::MetalCapabilityProbe probe;
  probe.classic_command_queue = true;
  probe.shared_event = true;
  probe.metal4_headers = true;
  probe.metal4_command_queue = true;
  probe.metal4_command_allocator = true;
  probe.tensor_api = true;
  probe.native_tensorops_runtime = true;
  return omlx::m5::select_metal_capability_profile(std::move(probe));
}

void check_routes(omlx::m5::MetalContext &context, Fixture &fixture,
                  std::uint32_t tokens,
                  const std::filesystem::path &tensorops_metallib) {
  auto plan = omlx::m5::prepare_dense_qmm_broad(
      context, OMLX_M5_QMM_BROAD_METALLIB_PATH, tensorops_metallib,
      fixture.view, tokens);
  expect(plan.info().token_tiles == (tokens + 15) / 16 &&
             plan.info().domain == (tokens <= 128 ? "interval" : "bulk"),
         "broad token tiling is wrong");
  omlx::m5::DispatchTable table;
  for (const auto &candidate : plan.dispatch_candidates()) {
    table.register_candidate(candidate);
  }
  const auto installed = plan.installed_executor_keys();
  table.seal(installed);
  if (fixture.mxfp4) {
    const bool host_tensorops =
        context.capabilities().profile.probe.native_tensorops_runtime &&
        context.capabilities().profile.native_tensorops_eligible;
    expect(table.stats().candidate_count == 8 &&
               table.stats().installed_candidate_count ==
                   (host_tensorops ? 8U : 4U),
           "MXFP4 broad candidate Gate is wrong");
    const auto a16 =
        table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                               omlx::m5::KernelAlignmentClass::full),
                      context.capabilities().profile);
    const auto a4 =
        table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                               omlx::m5::KernelAlignmentClass::full),
                      context.capabilities().profile);
    expect(plan.executor(a16.candidate.executor_key).family ==
                   (host_tensorops
                        ? omlx::m5::DenseQmmBroadFamily::mxfp4_w4a16_tensorops
                        : omlx::m5::DenseQmmBroadFamily::mxfp4_w4a16_simd) &&
               plan.executor(a4.candidate.executor_key).family ==
                   (host_tensorops
                        ? omlx::m5::DenseQmmBroadFamily::mxfp4_w4a4_tensorops
                        : omlx::m5::DenseQmmBroadFamily::mxfp4_w4a4_simd),
           "host broad MXFP4 selection disagrees with capability");
    omlx::m5::DispatchTable future;
    for (const auto &candidate : plan.dispatch_candidates()) {
      future.register_candidate(candidate);
    }
    std::vector<std::string> all;
    for (const auto &executor : plan.info().executors) {
      all.push_back(executor.executor_key);
    }
    future.seal(all);
    const auto target =
        future.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                                omlx::m5::KernelAlignmentClass::full),
                       tensorops_profile());
    expect(plan.executor(target.candidate.executor_key).family ==
               omlx::m5::DenseQmmBroadFamily::mxfp4_w4a4_tensorops,
           "target broad MXFP4 selection is wrong");
  } else {
    expect(table.stats().candidate_count == 2 &&
               table.stats().installed_candidate_count == 2,
           "affine broad candidate Gate is wrong");
  }
  const auto tail =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelAlignmentClass::tail),
                    context.capabilities().profile);
  expect(tail.candidate.tail_safe, "broad tail route is not tail-safe");
}

void execute_anchor(omlx::m5::MetalContext &context, Fixture &fixture,
                    std::uint32_t tokens,
                    const std::filesystem::path &tensorops_metallib) {
  std::vector<std::uint16_t> input_words;
  const auto input = make_input(tokens, input_words);
  auto input_buffer = context.upload(bytes(input_words));
  auto plan = omlx::m5::prepare_dense_qmm_broad(
      context, OMLX_M5_QMM_BROAD_METALLIB_PATH, tensorops_metallib,
      fixture.view, tokens);
  omlx::m5::DispatchTable table;
  for (const auto &candidate : plan.dispatch_candidates()) {
    table.register_candidate(candidate);
  }
  const auto installed = plan.installed_executor_keys();
  table.seal(installed);
  const auto a16 =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelAlignmentClass::full),
                    context.capabilities().profile);
  auto output = context.make_buffer(plan.info().output_bytes);
  const auto before = context.registry_stats();
  const auto submission = omlx::m5::execute_dense_qmm_broad_a16(
      context, plan, a16.candidate.executor_key, input_buffer, 0, output, 0);
  expect(submission.dispatch_count == (fixture.mxfp4 ? 1 : 2),
         "broad A16 dispatch count is wrong");
  check_output(output, reference(input, fixture.weights, tokens));
  const auto after_a16 = context.registry_stats();
  expect(before.upload_count == after_a16.upload_count &&
             before.pipelines_compiled == after_a16.pipelines_compiled,
         "broad A16 execution uploaded or compiled a resource");
  if (fixture.mxfp4) {
    const auto quantized = quantize_cpu(input, tokens);
    auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
        context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, 256, tokens);
    auto code_buffer = context.make_buffer(quantizer.info().code_bytes);
    auto scale_buffer = context.make_buffer(quantizer.info().scale_bytes);
    const omlx::m5::Mxfp4ActivationView activation{
        .codes = {.buffer = &code_buffer, .bytes = quantizer.info().code_bytes},
        .scales = {.buffer = &scale_buffer,
                   .bytes = quantizer.info().scale_bytes},
        .group_size = 32,
    };
    static_cast<void>(omlx::m5::quantize_mxfp4_a4(context, quantizer,
                                                  input_buffer, 0, activation));
    const auto a4 =
        table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                               omlx::m5::KernelAlignmentClass::full),
                      context.capabilities().profile);
    const auto before_a4 = context.registry_stats();
    static_cast<void>(omlx::m5::execute_dense_qmm_broad_a4(
        context, plan, a4.candidate.executor_key, activation, output, 0));
    check_output(output, reference(quantized.values, fixture.weights, tokens));
    const auto after_a4 = context.registry_stats();
    expect(before_a4.upload_count == after_a4.upload_count &&
               before_a4.pipelines_compiled == after_a4.pipelines_compiled,
           "broad A4 execution uploaded or compiled a resource");
  }
}

} // namespace

int main() {
  try {
    std::filesystem::path tensorops_metallib =
        OMLX_M5_MXFP4_TENSOROPS_METALLIB_PATH;
    if (!tensorops_metallib.empty() &&
        !std::filesystem::exists(tensorops_metallib)) {
      if (const char *reloc_dir = std::getenv("OMLX_M5_RELOCATE_METALLIB_DIR");
          reloc_dir != nullptr && *reloc_dir != '\0') {
        const auto candidate =
            std::filesystem::path(reloc_dir) / tensorops_metallib.filename();
        if (std::filesystem::exists(candidate)) {
          tensorops_metallib = candidate;
        }
      }
    }
    expect(tensorops_metallib.empty() ||
               std::filesystem::exists(tensorops_metallib),
           "MXFP4 TensorOps metallib path is invalid");
    auto context = omlx::m5::MetalContext::create();
    auto classic_override = omlx::m5::MetalContext::create(false);
    const bool mxfp4_needs_classic_context =
        tensorops_metallib.empty() &&
        context.capabilities().profile.native_tensorops_eligible;
    std::vector<Fixture> affine_fixtures;
    for (const auto bits : {4U, 5U, 6U, 8U}) {
      for (const auto group : {64U, 128U}) {
        affine_fixtures.push_back(affine_fixture(context, bits, group));
      }
    }
    auto &mxfp4_context =
        mxfp4_needs_classic_context ? classic_override : context;
    auto mxfp4 = mxfp4_fixture(mxfp4_context);
    for (std::uint32_t tokens = 13; tokens <= 128; ++tokens) {
      for (auto &fixture : affine_fixtures) {
        check_routes(context, fixture, tokens, tensorops_metallib);
      }
      check_routes(mxfp4_context, mxfp4, tokens, tensorops_metallib);
    }
    for (const auto tokens : {512U, 1024U}) {
      for (auto &fixture : affine_fixtures) {
        check_routes(context, fixture, tokens, tensorops_metallib);
      }
      check_routes(mxfp4_context, mxfp4, tokens, tensorops_metallib);
    }
    constexpr std::array anchors = {13U, 16U,  17U,  31U,  32U,
                                    64U, 127U, 128U, 512U, 1024U};
    for (const auto tokens : anchors) {
      for (auto &fixture : affine_fixtures) {
        execute_anchor(context, fixture, tokens, tensorops_metallib);
      }
      execute_anchor(mxfp4_context, mxfp4, tokens, tensorops_metallib);
    }
    if (!mxfp4_needs_classic_context) {
      auto override_fixture = mxfp4_fixture(classic_override);
      check_routes(classic_override, override_fixture, 16, tensorops_metallib);
      execute_anchor(classic_override, override_fixture, 16,
                     tensorops_metallib);
    } else {
      auto eligible_mxfp4 = mxfp4_fixture(context);
      expect_throws(
          [&] {
            static_cast<void>(omlx::m5::prepare_dense_qmm_broad(
                context, OMLX_M5_QMM_BROAD_METALLIB_PATH, tensorops_metallib,
                eligible_mxfp4.view, 16));
          },
          "broad QMM accepted empty TensorOps metallib on eligible host");
    }
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_dense_qmm_broad(
              context, OMLX_M5_QMM_BROAD_METALLIB_PATH, tensorops_metallib,
              affine_fixtures.front().view, 12));
        },
        "broad QMM accepted T=12");
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::prepare_dense_qmm_broad(
              context, OMLX_M5_QMM_BROAD_METALLIB_PATH, tensorops_metallib,
              affine_fixtures.front().view, 129));
        },
        "broad QMM accepted an unsupported non-anchor T");
    std::cout << "B2.5 Dense broad/bulk QMM tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B2.5 Dense broad/bulk QMM failure: " << error.what() << '\n';
    return 1;
  }
}
