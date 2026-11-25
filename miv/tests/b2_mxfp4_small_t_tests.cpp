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

Fixture make_fixture(omlx::m5::MetalContext &context) {
  constexpr std::uint32_t n = 5;
  constexpr std::uint32_t k = 256;
  constexpr std::uint32_t group_size = 32;
  Fixture fixture;
  fixture.view = {
      .name = "fixture.mxfp4.small_t",
      .format = "mxfp4_e2m1_g32_e8m0_v1",
      .layout = "mx_rowsplit_k128_v1",
      .parent_shape = {n, k},
      .padded_k = k,
      .row_count = n,
      .in_features = k,
  };
  std::vector<std::uint8_t> data(n * k / 2);
  std::vector<std::uint8_t> scales(n * k / group_size);
  fixture.weights.resize(n * k);
  for (std::uint32_t row = 0; row < n; ++row) {
    for (std::uint32_t group = 0; group < k / group_size; ++group) {
      scales[row * (k / group_size) + group] =
          static_cast<std::uint8_t>(124U + (row + group) % 7U);
    }
    for (std::uint32_t column = 0; column < k; ++column) {
      const auto code = (row * 7U + column * 5U) & 0x0FU;
      data[row * (k / 2) + column / 2] |=
          static_cast<std::uint8_t>(code << ((column & 1U) * 4U));
      fixture.weights[row * k + column] =
          decode_e2m1(code) *
          decode_e8m0(scales[row * (k / group_size) + column / group_size]);
    }
  }
  add_plane(fixture, context, "data", bytes(data), k / 2);
  add_plane(fixture, context, "scale", bytes(scales), k / group_size);
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
  constexpr std::uint32_t group_size = 32;
  QuantizedActivation result;
  result.codes.resize(tokens * k / 2);
  result.scales.resize(tokens * k / group_size);
  result.values.resize(tokens * k);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint32_t group = 0; group < k / group_size; ++group) {
      float maximum = 0.0F;
      for (std::uint32_t lane = 0; lane < group_size; ++lane) {
        maximum = std::max(
            maximum, std::abs(input[token * k + group * group_size + lane]));
      }
      const auto scale_code = encode_e8m0_round_up(maximum / 6.0F);
      const auto scale = decode_e8m0(scale_code);
      result.scales[token * (k / group_size) + group] = scale_code;
      for (std::uint32_t lane = 0; lane < group_size; ++lane) {
        const auto column = group * group_size + lane;
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
  expect(raw.size() == expected.size() * 2, "MXFP4 output size is wrong");
  for (std::size_t index = 0; index < expected.size(); ++index) {
    std::uint16_t word = 0;
    std::memcpy(&word, raw.data() + index * 2, sizeof(word));
    const auto actual = omlx::m5::bf16_to_float(word);
    const auto tolerance = 0.02F * std::max(1.0F, std::abs(expected[index]));
    if (!std::isfinite(actual) ||
        std::abs(actual - expected[index]) > tolerance) {
      throw std::runtime_error("MXFP4 small-T result exceeds tolerance");
    }
  }
}

omlx::m5::KernelKey make_key(const omlx::m5::DenseMxfp4SmallTPlan &plan,
                             omlx::m5::ActivationProfile activation,
                             omlx::m5::KernelAlignmentClass alignment) {
  return {
      .format_id = plan.info().format_id,
      .layout_id = plan.info().layout_id,
      .activation_policy = omlx::m5::ActivationPolicy::allow_a4,
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

void run_case(omlx::m5::MetalContext &context, std::uint32_t tokens,
              const std::filesystem::path &tensorops_metallib) {
  auto fixture = make_fixture(context);
  std::vector<std::uint16_t> input_words;
  const auto input = make_input(tokens, input_words);
  const auto quantized = quantize_cpu(input, tokens);
  auto input_buffer = context.upload(bytes(input_words));

  const std::array slot_specs = {omlx::m5::ActivationSlotSpec{
      .max_input_features = 256,
      .max_tokens = 12,
      .policy = omlx::m5::ActivationPolicy::allow_a4,
  }};
  omlx::m5::ActivationWorkspace workspace(
      context, omlx::m5::plan_activation_workspace(slot_specs));
  workspace.begin_epoch(1);
  const std::array uses = {
      omlx::m5::ActivationUse{
          .binding_name = "fixture.q_proj",
          .role = "attention_q",
          .policy = omlx::m5::ActivationPolicy::allow_a4,
      },
      omlx::m5::ActivationUse{
          .binding_name = "fixture.k_proj",
          .role = "attention_k",
          .policy = omlx::m5::ActivationPolicy::allow_a4,
      },
  };
  const auto lease = workspace.acquire({.source_id = 1,
                                        .normalization_id = 2,
                                        .input_features = 256,
                                        .tokens = tokens},
                                       uses, omlx::m5::ActivationProfile::a4);
  expect(lease.produce_a4 && lease.bundle.a4.has_value(),
         "A4 producer lease was not issued");

  auto quantizer = omlx::m5::prepare_mxfp4_a4_quantizer(
      context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, 256, tokens);
  const auto before_quantize = context.registry_stats();
  const auto quantize_submission = omlx::m5::quantize_mxfp4_a4(
      context, quantizer, input_buffer, 0, *lease.bundle.a4);
  const auto after_quantize = context.registry_stats();
  expect(quantize_submission.dispatch_count == 1 &&
             before_quantize.upload_count == after_quantize.upload_count &&
             before_quantize.pipelines_compiled ==
                 after_quantize.pipelines_compiled,
         "A4 quantization mutated warmup resources");
  const auto actual_codes = workspace.buffer().read(
      lease.bundle.a4->codes.offset, quantizer.info().code_bytes);
  const auto actual_scales = workspace.buffer().read(
      lease.bundle.a4->scales.offset, quantizer.info().scale_bytes);
  expect(actual_scales == bytes(quantized.scales),
         "A4 scale plane differs from the CPU oracle");
  const auto expected_codes = bytes(quantized.codes);
  if (actual_codes != expected_codes) {
    const auto mismatch = std::mismatch(
        actual_codes.begin(), actual_codes.end(), expected_codes.begin());
    throw std::runtime_error(
        "A4 code plane differs at byte " +
        std::to_string(mismatch.first - actual_codes.begin()) + ": got " +
        std::to_string(std::to_integer<unsigned>(*mismatch.first)) +
        ", expected " +
        std::to_string(std::to_integer<unsigned>(*mismatch.second)));
  }

  const bool host_tensorops =
      context.capabilities().profile.probe.native_tensorops_runtime &&
      context.capabilities().profile.native_tensorops_eligible;
  auto plan = omlx::m5::prepare_dense_mxfp4_small_t(
      context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, tensorops_metallib,
      fixture.view, tokens);
  expect(plan.info().executors.size() == 4 &&
             plan.installed_executor_keys().size() ==
                 (host_tensorops ? 4U : 2U) &&
             plan.info().native_tensorops_runtime_eligible == host_tensorops,
         "installed/planned MXFP4 executor boundary disagrees with host capability");
  omlx::m5::DispatchTable table;
  for (const auto &candidate : plan.dispatch_candidates()) {
    table.register_candidate(candidate);
  }
  const auto installed = plan.installed_executor_keys();
  table.seal(installed);
  expect(table.stats().candidate_count == 8 &&
             table.stats().installed_candidate_count ==
                 (host_tensorops ? 8U : 4U) &&
             table.stats().planned_candidate_count ==
                 (host_tensorops ? 0U : 4U),
         "MXFP4 candidate Gate is wrong");
  const auto selected_a16 =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelAlignmentClass::full),
                    context.capabilities().profile);
  const auto selected_a4 =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                             omlx::m5::KernelAlignmentClass::full),
                    context.capabilities().profile);
  expect(plan.executor(selected_a16.candidate.executor_key).family ==
                 (host_tensorops
                      ? omlx::m5::DenseMxfp4SmallTFamily::w4a16_tensorops
                      : omlx::m5::DenseMxfp4SmallTFamily::w4a16_simd) &&
             plan.executor(selected_a4.candidate.executor_key).family ==
                 (host_tensorops
                      ? omlx::m5::DenseMxfp4SmallTFamily::w4a4_tensorops
                      : omlx::m5::DenseMxfp4SmallTFamily::w4a4_simd),
         "host selected the wrong MXFP4 small-T family");
  const auto selected_tail =
      table.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                             omlx::m5::KernelAlignmentClass::tail),
                    context.capabilities().profile);
  expect(selected_tail.candidate.tail_safe,
         "MXFP4 tail request did not select a tail-safe candidate");

  omlx::m5::DispatchTable future_target;
  for (const auto &candidate : plan.dispatch_candidates()) {
    future_target.register_candidate(candidate);
  }
  std::vector<std::string> all_executors;
  for (const auto &executor : plan.info().executors) {
    all_executors.push_back(executor.executor_key);
  }
  future_target.seal(all_executors);
  const auto target_a16 =
      future_target.prepare(make_key(plan, omlx::m5::ActivationProfile::a16,
                                     omlx::m5::KernelAlignmentClass::full),
                            tensorops_profile());
  const auto target_a4 =
      future_target.prepare(make_key(plan, omlx::m5::ActivationProfile::a4,
                                     omlx::m5::KernelAlignmentClass::full),
                            tensorops_profile());
  expect(plan.executor(target_a16.candidate.executor_key).family ==
                 omlx::m5::DenseMxfp4SmallTFamily::w4a16_tensorops &&
             plan.executor(target_a4.candidate.executor_key).family ==
                 omlx::m5::DenseMxfp4SmallTFamily::w4a4_tensorops,
         "target capability did not prioritize TensorOps candidates");

  auto output = context.make_buffer(plan.info().output_bytes);
  const auto before_execute = context.registry_stats();
  const auto a16_submission = omlx::m5::execute_dense_mxfp4_small_t_a16(
      context, plan, selected_a16.candidate.executor_key, input_buffer, 0,
      output, 0);
  expect(a16_submission.dispatch_count == 1, "W4A16 dispatch count is wrong");
  check_output(output, reference(input, fixture.weights, tokens));
  const auto a4_submission = omlx::m5::execute_dense_mxfp4_small_t_a4(
      context, plan, selected_a4.candidate.executor_key, *lease.bundle.a4,
      output, 0);
  expect(a4_submission.dispatch_count == 1, "W4A4 dispatch count is wrong");
  check_output(output, reference(quantized.values, fixture.weights, tokens));
  const auto after_execute = context.registry_stats();
  expect(before_execute.upload_count == after_execute.upload_count &&
             before_execute.pipelines_compiled ==
                 after_execute.pipelines_compiled,
         "MXFP4 token loop uploaded or compiled a resource");

  const auto planned_tensorops =
      std::ranges::find_if(plan.info().executors, [](const auto &executor) {
        return executor.family ==
               omlx::m5::DenseMxfp4SmallTFamily::w4a4_tensorops;
      });
  expect(planned_tensorops != plan.info().executors.end() &&
             planned_tensorops->installed == host_tensorops,
         "TensorOps installation state disagrees with host capability");
  if (!host_tensorops) {
    expect_throws(
        [&] {
          static_cast<void>(omlx::m5::execute_dense_mxfp4_small_t_a4(
              context, plan, planned_tensorops->executor_key, *lease.bundle.a4,
              output, 0));
        },
        "uninstalled TensorOps executor was runnable");
  }
}

void run_fail_closed(const std::filesystem::path &tensorops_metallib) {
  auto context = omlx::m5::MetalContext::create();
  auto fixture = make_fixture(context);
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::prepare_dense_mxfp4_small_t(
            context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, tensorops_metallib,
            fixture.view, 1));
      },
      "MXFP4 small-T accepted T=1");
  auto invalid = fixture.view;
  invalid.format = "oq_affine_q4_g64_bf16_v1";
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::prepare_dense_mxfp4_small_t(
            context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH, tensorops_metallib,
            invalid, 2));
      },
      "MXFP4 small-T accepted an affine format");
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
    if (!tensorops_metallib.empty() ||
        !context.capabilities().profile.native_tensorops_eligible) {
      for (std::uint32_t tokens = 2; tokens <= 12; ++tokens) {
        run_case(context, tokens, tensorops_metallib);
      }
      run_case(classic_override, 2, tensorops_metallib);
    } else {
      // TensorOps metallib is absent on a TensorOps-eligible host (e.g.
      // MLX-free build): still cover the full T=2..12 matrix via the classic
      // SIMD context and verify the TensorOps-eligible context fails closed.
      for (std::uint32_t tokens = 2; tokens <= 12; ++tokens) {
        run_case(classic_override, tokens, tensorops_metallib);
      }
      auto eligible_fixture = make_fixture(context);
      expect_throws(
          [&] {
            static_cast<void>(omlx::m5::prepare_dense_mxfp4_small_t(
                context, OMLX_M5_MXFP4_SMALL_T_METALLIB_PATH,
                tensorops_metallib, eligible_fixture.view, 2));
          },
          "MXFP4 small-T accepted empty TensorOps metallib on eligible host");
    }
    run_fail_closed(tensorops_metallib);
    std::cout << "B2.4 Dense MXFP4 T=2..12 tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B2.4 Dense MXFP4 T=2..12 failure: " << error.what() << '\n';
    return 1;
  }
}
