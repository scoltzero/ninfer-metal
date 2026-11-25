#include "omlx_m5/capability.h"

#include <utility>

namespace omlx::m5 {

std::string_view to_string(MetalExecutionRoute route) {
  switch (route) {
  case MetalExecutionRoute::unavailable:
    return "unavailable";
  case MetalExecutionRoute::classic_msl:
    return "classic_msl";
  case MetalExecutionRoute::mlx_bridge:
    return "mlx_bridge";
  case MetalExecutionRoute::native_tensorops:
    return "native_tensorops";
  }
  return "unavailable";
}

MetalCapabilityProfile
select_metal_capability_profile(MetalCapabilityProbe probe) {
  MetalCapabilityProfile profile;
  profile.probe = std::move(probe);
  const auto &selected = profile.probe;
  profile.classic_msl_eligible =
      selected.classic_command_queue && selected.shared_event;
  profile.mlx_bridge_eligible =
      selected.mlx_bridge_available && selected.shared_event;
  profile.compute_icb_eligible =
      selected.classic_command_queue && selected.compute_icb_api &&
      selected.compute_icb_barrier && selected.residency_sets;
  profile.metal4_direct_eligible =
      selected.metal4_headers && selected.metal4_command_queue &&
      selected.metal4_command_allocator && selected.metal4_command_buffer &&
      selected.metal4_compute_encoder && selected.metal4_compiler &&
      selected.metal4_argument_table;
  profile.metal4_icb_eligible =
      profile.compute_icb_eligible && profile.metal4_direct_eligible;
  profile.native_tensorops_eligible =
      selected.metal4_headers && selected.metal4_command_queue &&
      selected.metal4_command_allocator && selected.tensor_api &&
      selected.native_tensorops_runtime && selected.shared_event;

  if (!selected.metal4_headers) {
    profile.fallback_reasons.emplace_back("metal4_headers_unavailable");
  }
  if (!selected.metal4_command_queue || !selected.metal4_command_allocator) {
    profile.fallback_reasons.emplace_back("metal4_runtime_unavailable");
  }
  if (!selected.tensor_api) {
    profile.fallback_reasons.emplace_back("tensor_api_unavailable");
  }
  if (!selected.native_tensorops_runtime) {
    profile.fallback_reasons.emplace_back("tensorops_not_runtime_validated");
  }
  if (!selected.shared_event) {
    profile.fallback_reasons.emplace_back("shared_event_unavailable");
  }
  if (!selected.compute_icb_api) {
    profile.icb_fallback_reasons.emplace_back("compute_icb_api_unavailable");
  }
  if (!selected.compute_icb_barrier) {
    profile.icb_fallback_reasons.emplace_back(
        "compute_icb_barrier_unavailable");
  }
  if (!selected.residency_sets) {
    profile.icb_fallback_reasons.emplace_back("residency_sets_unavailable");
  }
  if (!selected.classic_command_queue) {
    profile.icb_fallback_reasons.emplace_back(
        "classic_command_queue_unavailable");
  }
  if (!profile.metal4_icb_eligible) {
    profile.icb_fallback_reasons.emplace_back("metal4_icb_outer_unavailable");
  }
  if (!selected.metal4_headers) {
    profile.metal4_fallback_reasons.emplace_back("metal4_headers_unavailable");
  }
  if (!selected.metal4_command_queue || !selected.metal4_command_allocator ||
      !selected.metal4_command_buffer || !selected.metal4_compute_encoder) {
    profile.metal4_fallback_reasons.emplace_back(
        "metal4_submission_unavailable");
  }
  if (!selected.metal4_compiler) {
    profile.metal4_fallback_reasons.emplace_back("metal4_compiler_unavailable");
  }
  if (!selected.metal4_argument_table) {
    profile.metal4_fallback_reasons.emplace_back(
        "metal4_argument_table_unavailable");
  }

  if (profile.native_tensorops_eligible) {
    profile.selected_route = MetalExecutionRoute::native_tensorops;
  } else if (profile.classic_msl_eligible) {
    profile.selected_route = MetalExecutionRoute::classic_msl;
  } else if (profile.mlx_bridge_eligible) {
    profile.selected_route = MetalExecutionRoute::mlx_bridge;
  } else {
    profile.selected_route = MetalExecutionRoute::unavailable;
  }
  return profile;
}

} // namespace omlx::m5
