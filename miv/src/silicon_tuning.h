#pragma once

#include "omlx_m5/metal.h"

#if defined(__APPLE__)
#include <CoreFoundation/CoreFoundation.h>
#include <IOKit/IOKitLib.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5::detail {

struct SiliconTuningConfig {
  std::string profile_id = "apple-m5-ultra-64gpu";
  std::uint32_t gpu_cores = 64;
  std::uint32_t apple_gpu_family = 10;
  std::uint32_t split_kv_min_context = 2048;
  std::uint32_t split_kv_partitions = 12;
  std::uint32_t split_kv_chunk_tokens = 512;
  std::uint32_t gdn_tb8_min_tokens = 8;
  std::uint32_t linear_nax_min_tokens = 1;
  std::uint32_t linear_split_k_target_slices = 8;
  std::uint32_t moe_nax_min_tokens = 1;
  std::uint32_t dflash_draft_tokens = 7;
  std::uint64_t max_weight_segment_bytes = 2ULL << 30;
  bool loaded_from_json = false;
};

inline std::uint32_t probe_host_agx_gpu_cores() noexcept {
  if (const char *env = std::getenv("OMLX_M5_GPU_CORES"); env && *env) {
    const auto parsed = std::strtoul(env, nullptr, 10);
    if (parsed > 0 && parsed <= 512)
      return static_cast<std::uint32_t>(parsed);
  }
#if defined(__APPLE__)
  static const std::uint32_t cached = []() noexcept -> std::uint32_t {
    CFMutableDictionaryRef matching = IOServiceMatching("AGXAccelerator");
    if (!matching)
      return 0;
    io_iterator_t iter = 0;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, matching, &iter) !=
        KERN_SUCCESS)
      return 0;
    std::uint32_t cores = 0;
    while (io_object_t service = IOIteratorNext(iter)) {
      CFTypeRef prop = IORegistryEntryCreateCFProperty(
          service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (prop) {
        if (CFGetTypeID(prop) == CFNumberGetTypeID()) {
          int val = 0;
          if (CFNumberGetValue(static_cast<CFNumberRef>(prop), kCFNumberIntType,
                               &val) &&
              val > 0) {
            cores = static_cast<std::uint32_t>(val);
          }
        }
        CFRelease(prop);
      }
      IOObjectRelease(service);
      if (cores > 0)
        break;
    }
    IOObjectRelease(iter);
    return cores;
  }();
  return cached;
#else
  return 0;
#endif
}

inline SiliconTuningConfig
resolve_silicon_tuning(const MetalContext &context,
                       const std::filesystem::path &json_override = {}) {
  const auto &caps = context.capabilities();
  const std::uint32_t family = caps.profile.probe.apple_gpu_family;
  std::uint32_t cores = probe_host_agx_gpu_cores();
  const std::string_view dev = caps.device_name;

  if (cores == 0) {
    if (dev.find("Ultra") != std::string_view::npos)
      cores = 64;
    else if (dev.find("Max") != std::string_view::npos)
      cores = 40;
    else if (dev.find("Pro") != std::string_view::npos)
      cores = 20;
    else
      cores = 10;
  }

  SiliconTuningConfig cfg;
  cfg.gpu_cores = cores;
  cfg.apple_gpu_family = family;

  const bool forced_cores =
      (std::getenv("OMLX_M5_GPU_CORES") != nullptr);
  if (!forced_cores && family < 10 && !dev.starts_with("Apple M5")) {
    cfg.profile_id = "apple-m4-dev";
    cfg.split_kv_min_context = 8192;
    cfg.split_kv_partitions = 4;
    cfg.gdn_tb8_min_tokens = 256;
    cfg.linear_nax_min_tokens = 0;
    cfg.linear_split_k_target_slices = 1;
    cfg.moe_nax_min_tokens = 0;
  } else if (cores >= 72) {
    cfg.profile_id = "apple-m5-ultra-80gpu";
    cfg.split_kv_min_context = 2048;
    cfg.split_kv_partitions = 16;
    cfg.gdn_tb8_min_tokens = 8;
    cfg.linear_nax_min_tokens = 1;
    cfg.linear_split_k_target_slices = 8;
    cfg.moe_nax_min_tokens = 1;
    cfg.dflash_draft_tokens = 7;
  } else if (cores >= 48) {
    cfg.profile_id = "apple-m5-ultra-64gpu";
    cfg.split_kv_min_context = 2048;
    cfg.split_kv_partitions = 12;
    cfg.gdn_tb8_min_tokens = 8;
    cfg.linear_nax_min_tokens = 1;
    cfg.linear_split_k_target_slices = 8;
    cfg.moe_nax_min_tokens = 1;
    cfg.dflash_draft_tokens = 7;
  } else if (cores >= 36) {
    cfg.profile_id = "apple-m5-max-40gpu";
    cfg.split_kv_min_context = 4096;
    cfg.split_kv_partitions = 8;
    cfg.gdn_tb8_min_tokens = 16;
    cfg.linear_nax_min_tokens = 1;
    cfg.linear_split_k_target_slices = 4;
    cfg.moe_nax_min_tokens = 1;
    cfg.dflash_draft_tokens = 5;
  } else if (cores >= 28) {
    cfg.profile_id = "apple-m5-max-32gpu";
    cfg.split_kv_min_context = 4096;
    cfg.split_kv_partitions = 8;
    cfg.gdn_tb8_min_tokens = 16;
    cfg.linear_nax_min_tokens = 1;
    cfg.linear_split_k_target_slices = 4;
    cfg.moe_nax_min_tokens = 1;
    cfg.dflash_draft_tokens = 5;
  } else if (cores >= 14) {
    cfg.profile_id = cores >= 18 ? "apple-m5-pro-20gpu" : "apple-m5-pro-16gpu";
    cfg.split_kv_min_context = 8192;
    cfg.split_kv_partitions = 4;
    cfg.gdn_tb8_min_tokens = 32;
    cfg.linear_nax_min_tokens = 1;
    cfg.linear_split_k_target_slices = 2;
    cfg.moe_nax_min_tokens = 4;
    cfg.dflash_draft_tokens = 4;
  } else {
    cfg.profile_id = "apple-m5-10gpu";
    cfg.split_kv_min_context = 16384;
    cfg.split_kv_partitions = 2;
    cfg.gdn_tb8_min_tokens = 64;
    cfg.linear_nax_min_tokens = 1;
    cfg.linear_split_k_target_slices = 1;
    cfg.moe_nax_min_tokens = 8;
    cfg.dflash_draft_tokens = 3;
  }

  const auto apply_json_file = [&](const std::filesystem::path &path) -> bool {
    if (path.empty())
      return false;
    std::error_code ec;
    if (!std::filesystem::is_regular_file(path, ec) || ec)
      return false;
    std::ifstream in(path);
    if (!in)
      return false;
    const auto root = nlohmann::json::parse(in, nullptr, false);
    if (!root.is_object())
      return false;
    const std::string tbl_profile = root.value("profile_id", std::string{});
    if (!tbl_profile.empty() && tbl_profile != cfg.profile_id)
      return false;

    const auto read_u32 = [](const nlohmann::json &obj, const char *key,
                             std::uint32_t fallback) -> std::uint32_t {
      if (obj.contains(key) && !obj.at(key).is_null() &&
          obj.at(key).is_number_integer()) {
        const auto v = obj.at(key).get<std::int64_t>();
        if (v >= 0 && v <= 0xFFFFFFFFLL)
          return static_cast<std::uint32_t>(v);
      }
      return fallback;
    };
    const auto read_u64 = [](const nlohmann::json &obj, const char *key,
                             std::uint64_t fallback) -> std::uint64_t {
      if (obj.contains(key) && !obj.at(key).is_null() &&
          obj.at(key).is_number_integer()) {
        const auto v = obj.at(key).get<std::int64_t>();
        if (v >= 0)
          return static_cast<std::uint64_t>(v);
      }
      return fallback;
    };

    const std::string schema = root.value("schema", std::string{});
    const bool is_calibration_v1 =
        (schema == "omlx-m5-silicon-calibration-v1") ||
        root.contains("measured_dispatch_overrides") ||
        root.contains("measured_hardware_constants");

    if (is_calibration_v1) {
      if (root.contains("measured_hardware_constants") &&
          root["measured_hardware_constants"].is_object()) {
        const auto &hw = root["measured_hardware_constants"];
        cfg.max_weight_segment_bytes = read_u64(
            hw, "icb_safe_weight_segment_bytes", cfg.max_weight_segment_bytes);
      }
      if (root.contains("measured_dispatch_overrides") &&
          root["measured_dispatch_overrides"].is_object()) {
        const auto &ov = root["measured_dispatch_overrides"];
        cfg.split_kv_min_context = read_u32(
            ov, "attention_split_kv_min_context", cfg.split_kv_min_context);
        cfg.split_kv_partitions = read_u32(
            ov, "attention_split_kv_partitions", cfg.split_kv_partitions);
        cfg.gdn_tb8_min_tokens = read_u32(ov, "gdn_prefill_tb8_min_tokens",
                                          cfg.gdn_tb8_min_tokens);
        cfg.linear_nax_min_tokens =
            read_u32(ov, "linear_nax_min_tokens", cfg.linear_nax_min_tokens);
        cfg.linear_split_k_target_slices = read_u32(
            ov, "linear_split_k_target_slices", cfg.linear_split_k_target_slices);
        cfg.moe_nax_min_tokens = read_u32(ov, "moe_prefill_nax_min_tokens",
                                          cfg.moe_nax_min_tokens);
        cfg.dflash_draft_tokens =
            read_u32(ov, "dflash2_draft_tokens", cfg.dflash_draft_tokens);
      }
      cfg.loaded_from_json = true;
      return true;
    }

    if (root.contains("memory_and_icb_rules") &&
        root["memory_and_icb_rules"].is_object()) {
      cfg.max_weight_segment_bytes =
          read_u64(root["memory_and_icb_rules"], "max_weight_segment_bytes",
                   cfg.max_weight_segment_bytes);
    }
    if (root.contains("dispatch_policies") &&
        root["dispatch_policies"].is_object()) {
      const auto &dp = root["dispatch_policies"];
      if (dp.contains("attention") && dp["attention"].is_object()) {
        cfg.split_kv_min_context = read_u32(
            dp["attention"], "split_kv_min_context", cfg.split_kv_min_context);
        cfg.split_kv_partitions = read_u32(
            dp["attention"], "split_kv_partitions", cfg.split_kv_partitions);
      }
      if (dp.contains("gdn_prefill") && dp["gdn_prefill"].is_object()) {
        cfg.gdn_tb8_min_tokens = read_u32(dp["gdn_prefill"], "tb8_min_tokens",
                                          cfg.gdn_tb8_min_tokens);
      }
      if (dp.contains("moe_routing") && dp["moe_routing"].is_object()) {
        cfg.moe_nax_min_tokens = read_u32(
            dp["moe_routing"], "prefill_nax_min_tokens", cfg.moe_nax_min_tokens);
      }
      if (dp.contains("linear") && dp["linear"].is_object()) {
        cfg.linear_nax_min_tokens = read_u32(
            dp["linear"], "linear_nax_min_tokens", cfg.linear_nax_min_tokens);
        cfg.linear_split_k_target_slices =
            read_u32(dp["linear"], "linear_split_k_target_slices",
                     cfg.linear_split_k_target_slices);
      }
      if (dp.contains("speculative_dflash2") &&
          dp["speculative_dflash2"].is_object()) {
        cfg.dflash_draft_tokens = read_u32(dp["speculative_dflash2"],
                                           "draft_tokens",
                                           cfg.dflash_draft_tokens);
      }
    }
    cfg.loaded_from_json = true;
    return true;
  };

  if (!json_override.empty()) {
    (void)apply_json_file(json_override);
  } else {
    bool applied_explicit = false;
    if (const char *cal_env = std::getenv("OMLX_M5_SILICON_CALIBRATION_FILE");
        cal_env && *cal_env) {
      applied_explicit |= apply_json_file(std::filesystem::path(cal_env));
    }
    if (const char *tbl_env = std::getenv("OMLX_M5_STATIC_DISPATCH_TABLE");
        tbl_env && *tbl_env) {
      applied_explicit |= apply_json_file(std::filesystem::path(tbl_env));
    }
    if (!applied_explicit) {
      const std::string filename = cfg.profile_id + ".json";
      if (const char *dir_env = std::getenv("OMLX_M5_SILICON_CALIBRATION_DIR");
          dir_env && *dir_env) {
        (void)apply_json_file(std::filesystem::path(dir_env) / filename);
      } else if (!forced_cores) {
        std::vector<std::filesystem::path> candidates{
            std::filesystem::path("docs/backend/silicon-calibrations") /
                filename,
            std::filesystem::path("../docs/backend/silicon-calibrations") /
                filename,
        };
        if (const char *home = std::getenv("HOME"); home && *home) {
          const std::filesystem::path home_p(home);
          candidates.push_back(
              home_p /
              "Library/Application Support/oMLX/Providers/io.ninfer.metal/current/calibrations" /
              filename);
          candidates.push_back(
              home_p /
              "Library/Application Support/oMLX/Providers/com.omlx.mseries/calibrations" /
              filename);
        }
        for (const auto &candidate : candidates) {
          if (apply_json_file(candidate)) {
            break;
          }
        }
      }
    }
  }
  return cfg;
}

} // namespace omlx::m5::detail
