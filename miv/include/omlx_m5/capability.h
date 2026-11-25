#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5 {

enum class MetalExecutionRoute : std::uint8_t {
  unavailable,
  classic_msl,
  mlx_bridge,
  native_tensorops,
};

[[nodiscard]] std::string_view to_string(MetalExecutionRoute route);

struct MetalCapabilityProbe {
  std::uint32_t runtime_os_major = 0;
  std::uint32_t runtime_os_minor = 0;
  std::uint32_t runtime_os_patch = 0;
  std::uint32_t sdk_max_allowed = 0;
  std::uint32_t apple_gpu_family = 0;
  bool classic_command_queue = false;
  bool shared_event = false;
  bool metal4_headers = false;
  bool metal4_command_queue = false;
  bool metal4_command_allocator = false;
  bool metal4_command_buffer = false;
  bool metal4_compute_encoder = false;
  bool metal4_compiler = false;
  bool metal4_argument_table = false;
  bool compute_icb_api = false;
  bool compute_icb_barrier = false;
  bool residency_sets = false;
  bool tensor_api = false;
  bool native_tensorops_runtime = false;
  bool mlx_bridge_available = false;
  std::string native_tensorops_source = "unverified";
};

struct MetalCapabilityProfile {
  MetalCapabilityProbe probe;
  MetalExecutionRoute selected_route = MetalExecutionRoute::unavailable;
  bool native_tensorops_eligible = false;
  bool classic_msl_eligible = false;
  bool mlx_bridge_eligible = false;
  bool compute_icb_eligible = false;
  bool metal4_direct_eligible = false;
  bool metal4_icb_eligible = false;
  std::vector<std::string> fallback_reasons;
  std::vector<std::string> metal4_fallback_reasons;
  std::vector<std::string> icb_fallback_reasons;
};

[[nodiscard]] MetalCapabilityProfile
select_metal_capability_profile(MetalCapabilityProbe probe);

} // namespace omlx::m5
