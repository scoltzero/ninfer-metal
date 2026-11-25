#include <array>
#include <cstdint>
#include <iostream>
#include <ranges>
#include <stdexcept>
#include <string>
#include <utility>

#include "omlx_m5/dispatch.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Exception, class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

omlx::m5::MetalCapabilityProfile classic_profile() {
  omlx::m5::MetalCapabilityProbe probe;
  probe.classic_command_queue = true;
  probe.shared_event = true;
  return omlx::m5::select_metal_capability_profile(std::move(probe));
}

omlx::m5::MetalCapabilityProfile tensorops_profile() {
  omlx::m5::MetalCapabilityProbe probe;
  probe.runtime_os_major = 27;
  probe.sdk_max_allowed = 270000;
  probe.apple_gpu_family = 10;
  probe.classic_command_queue = true;
  probe.shared_event = true;
  probe.metal4_headers = true;
  probe.metal4_command_queue = true;
  probe.metal4_command_allocator = true;
  probe.tensor_api = true;
  probe.native_tensorops_runtime = true;
  probe.native_tensorops_source = "test";
  return omlx::m5::select_metal_capability_profile(std::move(probe));
}

omlx::m5::KernelCandidate
candidate(std::string id, std::string executor, const omlx::m5::KernelKey &key,
          omlx::m5::ActivationProfile activation, omlx::m5::KernelFamily family,
          std::uint32_t priority, std::uint32_t min_tokens,
          std::uint32_t max_tokens, omlx::m5::MetalExecutionRoute route,
          bool tail_safe = false) {
  return {
      .id = std::move(id),
      .executor_key = std::move(executor),
      .priority = priority,
      .family = family,
      .format_id = key.format_id,
      .layout_id = key.layout_id,
      .activation = activation,
      .input_type = key.input_type,
      .output_type = key.output_type,
      .accumulator_type = key.accumulator_type,
      .geometry = key.geometry,
      .output_features = key.output_features,
      .input_features = key.input_features,
      .min_tokens = min_tokens,
      .max_tokens = max_tokens,
      .experts = key.experts,
      .top_k = key.top_k,
      .fused_op = key.fused_op,
      .alignment = tail_safe ? omlx::m5::KernelAlignmentClass::tail
                             : omlx::m5::KernelAlignmentClass::full,
      .required_route = route,
      .tail_safe = tail_safe,
  };
}

omlx::m5::KernelKey mxfp4_key(std::uint32_t tokens = 8) {
  omlx::m5::KernelKey key{
      .format_id = "mxfp4_e2m1_g32_e8m0_v1",
      .layout_id = "mx_rowsplit_k128_v1",
      .activation_policy = omlx::m5::ActivationPolicy::allow_a4,
      .preferred_activation = omlx::m5::ActivationProfile::a4,
      .input_type = omlx::m5::KernelDataType::bf16,
      .output_type = omlx::m5::KernelDataType::bf16,
      .accumulator_type = omlx::m5::KernelDataType::fp32,
      .geometry = omlx::m5::KernelGeometry::dense,
      .output_features = 14336,
      .input_features = 5120,
      .tokens = tokens,
      .fused_op = omlx::m5::KernelFusedOp::linear,
      .alignment = omlx::m5::KernelAlignmentClass::full,
  };
  omlx::m5::validate_kernel_key(key);
  return key;
}

void test_artifact_key_binding() {
  const omlx::m5::ArtifactObject expert{
      .id = 7,
      .name = "layer.experts",
      .format = "oq_affine_q4_g64_bf16_v1",
      .layout = "oq_expert_rowsplit_k128_v1",
      .shape = {256, 512, 2048},
      .padded_k = 2048,
  };
  const omlx::m5::ArtifactBinding binding{
      .name = expert.name,
      .object_id = expert.id,
      .format = expert.format,
      .use = "moe_expert_gate",
      .activation_policy = "A16Only",
      .leading_shape = {256},
      .out_features = 512,
      .in_features = 2048,
  };
  const auto key = omlx::m5::make_kernel_key(
      binding, expert, 8, omlx::m5::ActivationProfile::a16,
      omlx::m5::KernelFusedOp::moe_gate_up,
      omlx::m5::KernelAlignmentClass::tail);
  expect(key.geometry == omlx::m5::KernelGeometry::moe_expert &&
             key.experts == 256 && key.top_k == 8 &&
             key.alignment == omlx::m5::KernelAlignmentClass::tail,
         "expert artifact did not produce the expected kernel key");

  const omlx::m5::ArtifactObject direct{
      .name = "layer.proj.weight",
      .format = "direct_bf16_v1",
      .layout = "contiguous_le_v1",
      .shape = {5120, 6144},
  };
  const auto direct_key = omlx::m5::make_direct_kernel_key(direct, 1);
  expect(direct_key.input_type == omlx::m5::KernelDataType::bf16 &&
             direct_key.preferred_activation ==
                 omlx::m5::ActivationProfile::a16,
         "direct BF16 object did not produce an A16 key");

  auto invalid = mxfp4_key();
  invalid.format_id = "oq_affine_q4_g64_bf16_v1";
  invalid.layout_id = "oq_rowsplit_k128_v1";
  expect_throws<std::invalid_argument>(
      [&] { omlx::m5::validate_kernel_key(invalid); },
      "oQ key accepted an A4 activation");

  // Regression tests for MIVE-8: Q3 affine formats and NAX pre-tiled layouts.
  for (const auto *q3_fmt :
       {"oq_affine_q3_g64_bf16_v1", "oq_affine_q3_g128_bf16_v1"}) {
    for (const auto *layout : {"oq_rowsplit_k128_v1", "oq_nax_tiled_g64_v1",
                               "oq_nax_tiled_g128_v1"}) {
      const omlx::m5::ArtifactObject q3_obj{
          .id = 11,
          .name = "layer.q3_proj",
          .format = q3_fmt,
          .layout = layout,
          .shape = {1024, 512},
          .padded_k = 512,
      };
      const omlx::m5::ArtifactBinding q3_bind{
          .name = q3_obj.name,
          .object_id = q3_obj.id,
          .format = q3_obj.format,
          .use = "linear",
          .activation_policy = "A16Only",
          .out_features = 1024,
          .in_features = 512,
      };
      const auto q3_key = omlx::m5::make_kernel_key(
          q3_bind, q3_obj, 1, omlx::m5::ActivationProfile::a16);
      expect(q3_key.format_id == q3_fmt && q3_key.layout_id == layout,
             "Q3/NAX-tiled kernel key mismatch");
    }
  }

  auto mx_nax = mxfp4_key();
  mx_nax.layout_id = "mx_nax_tiled_bk128_v1";
  omlx::m5::validate_kernel_key(mx_nax);

  auto bad_mx_with_oq = mx_nax;
  bad_mx_with_oq.layout_id = "oq_nax_tiled_g64_v1";
  expect_throws<std::invalid_argument>(
      [&] { omlx::m5::validate_kernel_key(bad_mx_with_oq); },
      "MXFP4 key accepted oQ NAX tiled layout");

  auto bad_oq_with_mx = mx_nax;
  bad_oq_with_mx.format_id = "oq_affine_q3_g64_bf16_v1";
  bad_oq_with_mx.activation_policy = omlx::m5::ActivationPolicy::a16_only;
  bad_oq_with_mx.preferred_activation = omlx::m5::ActivationProfile::a16;
  expect_throws<std::invalid_argument>(
      [&] { omlx::m5::validate_kernel_key(bad_oq_with_mx); },
      "oQ Q3 key accepted MXFP4 NAX tiled layout");

  auto bad_expert_nax = key;
  bad_expert_nax.layout_id = "oq_nax_tiled_g64_v1";
  expect_throws<std::invalid_argument>(
      [&] { omlx::m5::validate_kernel_key(bad_expert_nax); },
      "MoE expert key accepted non-expert NAX tiled layout");
}

void test_dispatch_fallbacks() {
  const auto key = mxfp4_key();
  const auto a4 = candidate("mx_a4_tensorops", "exec_a4", key,
                            omlx::m5::ActivationProfile::a4,
                            omlx::m5::KernelFamily::tensorops, 10, 2, 128,
                            omlx::m5::MetalExecutionRoute::native_tensorops);
  const auto a16 = candidate("mx_a16_classic", "exec_a16", key,
                             omlx::m5::ActivationProfile::a16,
                             omlx::m5::KernelFamily::row_split, 20, 1, 128,
                             omlx::m5::MetalExecutionRoute::classic_msl);
  const auto tail = candidate("mx_a16_tail", "exec_tail", key,
                              omlx::m5::ActivationProfile::a16,
                              omlx::m5::KernelFamily::tail_safe, 100, 1, 1024,
                              omlx::m5::MetalExecutionRoute::classic_msl, true);

  omlx::m5::DispatchTable classic;
  classic.register_candidate(a4);
  classic.register_candidate(a16);
  classic.register_candidate(tail);
  const std::array installed_classic = {std::string("exec_a16"),
                                        std::string("exec_tail")};
  classic.seal(installed_classic);
  const auto fallback = classic.prepare(key, classic_profile());
  expect(fallback.candidate.id == "mx_a16_classic" &&
             fallback.resolved_activation == omlx::m5::ActivationProfile::a16 &&
             fallback.fallback() &&
             fallback.fallback_reasons.front() == "a4_executor_unavailable",
         "A4 request did not fall back to installed A16");
  const auto stats = classic.stats();
  expect(stats.candidate_count == 3 && stats.installed_candidate_count == 2 &&
             stats.planned_candidate_count == 1 && stats.sealed,
         "dispatch availability ledger is wrong");

  omlx::m5::DispatchTable native;
  native.register_candidate(a4);
  native.register_candidate(a16);
  native.register_candidate(tail);
  const std::array installed_all = {std::string("exec_a4"),
                                    std::string("exec_a16"),
                                    std::string("exec_tail")};
  native.seal(installed_all);
  const auto selected = native.prepare(key, tensorops_profile());
  expect(selected.candidate.id == "mx_a4_tensorops" &&
             selected.resolved_activation == omlx::m5::ActivationProfile::a4 &&
             !selected.fallback(),
         "native TensorOps request did not select installed A4");

  auto bulk = key;
  bulk.tokens = 512;
  const auto broad = native.prepare(bulk, classic_profile());
  expect(broad.candidate.id == "mx_a16_tail" && broad.fallback() &&
             std::ranges::find(broad.fallback_reasons, "tail_safe_route") !=
                 broad.fallback_reasons.end(),
         "bulk request did not reach the broad tail-safe fallback");

  auto tail_key = key;
  tail_key.preferred_activation = omlx::m5::ActivationProfile::a16;
  tail_key.alignment = omlx::m5::KernelAlignmentClass::tail;
  const auto tail_plan = native.prepare(tail_key, classic_profile());
  expect(tail_plan.candidate.id == "mx_a16_tail" &&
             std::ranges::find(tail_plan.fallback_reasons, "tail_safe_route") ==
                 tail_plan.fallback_reasons.end(),
         "tail request did not select its mandatory safe route");
}

void test_fail_closed_registry() {
  const auto key = mxfp4_key();
  const auto a16 =
      candidate("candidate", "executor", key, omlx::m5::ActivationProfile::a16,
                omlx::m5::KernelFamily::row_split, 10, 1, 128,
                omlx::m5::MetalExecutionRoute::classic_msl);
  omlx::m5::DispatchTable table;
  table.register_candidate(a16);
  expect_throws<std::logic_error>(
      [&] { static_cast<void>(table.prepare(key, classic_profile())); },
      "unsealed dispatch table prepared a route");
  expect_throws<std::invalid_argument>([&] { table.register_candidate(a16); },
                                       "duplicate candidate id was accepted");
  expect_throws<std::invalid_argument>(
      [&] {
        const std::array unknown = {std::string("missing")};
        table.seal(unknown);
      },
      "unknown executor was accepted");
  const std::array installed = {std::string("executor")};
  table.seal(installed);
  expect_throws<std::logic_error>([&] { table.register_candidate(a16); },
                                  "sealed table accepted another candidate");

  omlx::m5::DispatchTable planned_only;
  planned_only.register_candidate(a16);
  planned_only.seal({});
  expect_throws<std::out_of_range>(
      [&] {
        auto a16_key = key;
        a16_key.preferred_activation = omlx::m5::ActivationProfile::a16;
        static_cast<void>(planned_only.prepare(a16_key, classic_profile()));
      },
      "planned but uninstalled executor was dispatched");

  omlx::m5::DispatchTable ambiguous;
  ambiguous.register_candidate(a16);
  auto duplicate_domain = a16;
  duplicate_domain.id = "candidate_2";
  duplicate_domain.executor_key = "executor_2";
  ambiguous.register_candidate(duplicate_domain);
  const std::array both = {std::string("executor"), std::string("executor_2")};
  ambiguous.seal(both);
  expect_throws<std::logic_error>(
      [&] {
        auto a16_key = key;
        a16_key.preferred_activation = omlx::m5::ActivationProfile::a16;
        static_cast<void>(ambiguous.prepare(a16_key, classic_profile()));
      },
      "equal-priority overlapping routes were not rejected");
}

} // namespace

int main() {
  try {
    test_artifact_key_binding();
    test_dispatch_fallbacks();
    test_fail_closed_registry();
    std::cout << "B2.1 dispatch contract tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B2.1 dispatch contract failure: " << error.what() << '\n';
    return 1;
  }
}
