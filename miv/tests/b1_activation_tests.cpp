#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "omlx_m5/activation.h"

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

omlx::m5::ActivationUse
use(std::string name, std::string role, omlx::m5::ActivationPolicy policy,
    std::optional<std::uint32_t> expert = std::nullopt) {
  return {
      .binding_name = std::move(name),
      .role = std::move(role),
      .policy = policy,
      .expert = expert,
  };
}

void test_policy_contract() {
  using omlx::m5::ActivationPolicy;
  using omlx::m5::ActivationProfile;
  expect(omlx::m5::parse_activation_policy("A16Only") ==
             ActivationPolicy::a16_only,
         "A16Only parsing failed");
  expect(omlx::m5::parse_activation_policy("AllowA4") ==
             ActivationPolicy::allow_a4,
         "AllowA4 parsing failed");
  expect_throws<std::invalid_argument>(
      [] { static_cast<void>(omlx::m5::parse_activation_policy("AllowA8")); },
      "unsupported activation policy was accepted");

  const std::array all_a4 = {
      use("q", "attention_q", ActivationPolicy::allow_a4),
      use("k", "attention_k", ActivationPolicy::allow_a4),
      use("v", "attention_v", ActivationPolicy::allow_a4),
  };
  expect(omlx::m5::common_activation_policy(all_a4) ==
             ActivationPolicy::allow_a4,
         "all-A4 policy intersection failed");
  auto mixed = all_a4;
  mixed[1].policy = ActivationPolicy::a16_only;
  expect(omlx::m5::common_activation_policy(mixed) ==
             ActivationPolicy::a16_only,
         "mixed policy did not reduce to A16Only");
  expect(omlx::m5::resolve_activation_profile(ActivationPolicy::a16_only,
                                              ActivationProfile::a4) ==
             ActivationProfile::a16,
         "A16Only did not reject the preferred A4 profile");

  const omlx::m5::ArtifactBinding binding{
      .name = "layer.q_proj",
      .use = "attention_q",
      .activation_policy = "AllowA4",
  };
  const auto converted = omlx::m5::make_activation_use(binding, 7);
  expect(converted.binding_name == binding.name &&
             converted.policy == ActivationPolicy::allow_a4 &&
             converted.expert == 7,
         "artifact activation policy did not reach the runtime use");
}

void test_workspace_reuse() {
  using omlx::m5::ActivationPolicy;
  using omlx::m5::ActivationProfile;
  const std::array slots = {
      omlx::m5::ActivationSlotSpec{64, 8, ActivationPolicy::allow_a4},
      omlx::m5::ActivationSlotSpec{64, 8, ActivationPolicy::allow_a4},
      omlx::m5::ActivationSlotSpec{64, 8, ActivationPolicy::a16_only},
  };
  const auto plan = omlx::m5::plan_activation_workspace(slots);
  expect(plan.a16_bytes == 3 * 64 * 8 * 2, "planned A16 bytes are wrong");
  expect(plan.a4_code_bytes == 2 * 64 * 8 / 2,
         "planned A4 code bytes are wrong");
  expect(plan.a4_scale_bytes == 2 * 64 * 8 / 32,
         "planned A4 scale bytes are wrong");
  expect(plan.capacity_bytes == plan.a16_bytes + plan.a4_code_bytes +
                                    plan.a4_scale_bytes + plan.padding_bytes,
         "activation plan accounting is wrong");
  expect_throws<std::invalid_argument>(
      [] {
        const std::array invalid = {
            omlx::m5::ActivationSlotSpec{65, 8, ActivationPolicy::allow_a4}};
        static_cast<void>(omlx::m5::plan_activation_workspace(invalid));
      },
      "activation plan accepted K that is not divisible by 32");

  auto context = omlx::m5::MetalContext::create();
  auto modified_plan = plan;
  ++modified_plan.capacity_bytes;
  expect_throws<std::invalid_argument>(
      [&] {
        omlx::m5::ActivationWorkspace invalid(context, modified_plan);
        static_cast<void>(invalid);
      },
      "activation workspace accepted a modified plan");
  const auto registry_before = context.registry_stats();
  {
    omlx::m5::ActivationWorkspace workspace(context, plan);
    const auto registry_created = context.registry_stats();
    expect(registry_created.current_buffer_count ==
               registry_before.current_buffer_count + 1,
           "activation workspace did not create one resident buffer");
    expect(registry_created.current_buffer_bytes ==
               registry_before.current_buffer_bytes + plan.capacity_bytes,
           "activation workspace resident ledger is wrong");
    expect(registry_created.upload_count == registry_before.upload_count,
           "activation workspace allocation counted as an upload");

    workspace.begin_epoch(1);
    const std::array qkv = {
        use("q", "attention_q", ActivationPolicy::allow_a4),
        use("k", "attention_k", ActivationPolicy::allow_a4),
        use("v", "attention_v", ActivationPolicy::allow_a4),
    };
    const omlx::m5::ActivationReuseKey attention_key{1, 101, 64, 8};
    const auto attention =
        workspace.acquire(attention_key, qkv, ActivationProfile::a4);
    expect(!attention.produce_a16 && attention.produce_a4 &&
               !attention.bundle.normalized_a16.has_value() &&
               attention.bundle.a4.has_value(),
           "A4-only lease unnecessarily materialized an A16 copy");
    expect(attention.bundle.a4->codes.bytes == 64 * 8 / 2 &&
               attention.bundle.a4->scales.bytes == 64 * 8 / 32,
           "A4 activation plane geometry is wrong");

    const std::array<std::byte, 4> marker = {std::byte{0x01}, std::byte{0x23},
                                             std::byte{0x45}, std::byte{0x67}};
    attention.bundle.a4->codes.buffer->write(marker,
                                             attention.bundle.a4->codes.offset);
    expect(attention.bundle.a4->codes.buffer->read(
               attention.bundle.a4->codes.offset, marker.size()) ==
               std::vector<std::byte>(marker.begin(), marker.end()),
           "A4 code plane does not reference the workspace buffer");

    const std::array gate_up = {
        use("gate", "mlp_gate", ActivationPolicy::allow_a4),
        use("up", "mlp_up", ActivationPolicy::allow_a4),
    };
    const auto shared =
        workspace.acquire(attention_key, gate_up, ActivationProfile::a4);
    expect(!shared.produce_a16 && !shared.produce_a4 &&
               shared.bundle.a4->codes.offset ==
                   attention.bundle.a4->codes.offset,
           "cross-projection lease did not reuse the activation");
    const auto dual =
        workspace.acquire(attention_key, qkv, ActivationProfile::a4, true);
    expect(dual.produce_a16 && !dual.produce_a4 &&
               dual.bundle.normalized_a16.has_value() &&
               dual.bundle.a4.has_value() &&
               dual.bundle.a4->codes.offset ==
                   attention.bundle.a4->codes.offset,
           "A4 lease could not add a shared A16 fallback representation");

    std::vector<omlx::m5::ActivationUse> experts;
    for (std::uint32_t expert = 0; expert < 8; ++expert) {
      experts.push_back(use("expert_gate", "moe_expert_gate",
                            ActivationPolicy::allow_a4, expert));
    }
    const auto moe =
        workspace.acquire({2, 202, 64, 8}, experts, ActivationProfile::a4);
    expect(!moe.produce_a16 && moe.produce_a4,
           "MoE activation was not materialized once");

    const std::array mixed = {
        use("mx", "attention_q", ActivationPolicy::allow_a4),
        use("oq", "attention_k", ActivationPolicy::a16_only),
    };
    const auto fallback =
        workspace.acquire({3, 303, 64, 8}, mixed, ActivationProfile::a4);
    expect(fallback.bundle.profile == ActivationProfile::a16 &&
               fallback.produce_a16 && !fallback.produce_a4 &&
               fallback.bundle.normalized_a16.has_value() &&
               !fallback.bundle.a4.has_value(),
           "mixed policy did not fall back to one shared A16 activation");

    const auto before_failure = workspace.stats();
    expect_throws<std::bad_alloc>(
        [&] {
          const std::array extra = {
              use("extra", "extra", ActivationPolicy::a16_only)};
          static_cast<void>(workspace.acquire({4, 404, 64, 8}, extra,
                                              ActivationProfile::a16));
        },
        "activation workspace accepted more unique entries than planned");
    const auto stats = workspace.stats();
    expect(stats.acquire_calls == before_failure.acquire_calls &&
               stats.consumer_uses == before_failure.consumer_uses,
           "failed acquisition changed the activation ledger");
    expect(stats.acquire_calls == 5 && stats.consumer_uses == 18 &&
               stats.cache_hits == 2 && stats.policy_fallbacks == 1,
           "activation request ledger is wrong");
    expect(stats.a16_materializations == 2 && stats.a4_materializations == 2 &&
               stats.avoided_a16_materializations == 3 &&
               stats.avoided_a4_quantizations == 14,
           "activation materialization savings are wrong");
    expect(stats.cross_projection_reuses == 6 && stats.cross_expert_reuses == 7,
           "projection/expert reuse ledger is wrong");

    workspace.begin_epoch(2);
    const auto next =
        workspace.acquire(attention_key, qkv, ActivationProfile::a4);
    expect(!next.produce_a16 && next.produce_a4 && next.bundle.epoch == 2,
           "new epoch reused stale activation contents");
    expect_throws<std::invalid_argument>(
        [&] { workspace.begin_epoch(2); },
        "activation workspace accepted a repeated epoch");
  }
  const auto registry_after = context.registry_stats();
  expect(registry_after.current_buffer_count ==
                 registry_before.current_buffer_count &&
             registry_after.current_buffer_bytes ==
                 registry_before.current_buffer_bytes,
         "activation workspace leaked its Metal buffer");
}

} // namespace

int main() {
  try {
    test_policy_contract();
    test_workspace_reuse();
    std::cout << "B1.3 activation workspace tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B1.3 activation test failure: " << error.what() << '\n';
    return 1;
  }
}
