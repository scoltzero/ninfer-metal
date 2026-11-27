#pragma once

#include "forward_plan.h"

#include <algorithm>
#include <atomic>
#include <concepts>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace omlx::m5::detail::forward {

inline bool nax_apple10_eligible(const MetalContext &ctx,
                                 const std::filesystem::path &libs,
                                 std::string_view metallib_filename,
                                 const char *disable_env) {
  if (disable_env != nullptr) {
    if (const char *disable = std::getenv(disable_env);
        disable != nullptr && std::string_view(disable) == "1") {
      return false;
    }
  }
  const auto &c = ctx.capabilities();
  const auto &p = c.profile.probe;
  const bool force_apple10 = [&] {
    const char *env = std::getenv("OMLX_M5_FORCE_NAX_APPLE10");
    return env != nullptr && std::string_view(env) == "1";
  }();
  return (force_apple10 || c.device_name.starts_with("Apple M5") ||
          p.apple_gpu_family >= 10) &&
         p.runtime_os_major >= 27 && p.metal4_compiler && p.tensor_api &&
         std::filesystem::is_regular_file(libs / metallib_filename);
}

inline U32 nax_estimate_gpu_cores(std::string_view device_name,
                                  U32 explicit_gpu_cores = 0u) {
  if (explicit_gpu_cores > 0u) {
    return explicit_gpu_cores;
  }
  if (const char *env = std::getenv("OMLX_M5_GPU_CORES")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v > 0u) {
      return v;
    }
  }
  if (device_name.find("Ultra") != std::string_view::npos) {
    return 64u;
  }
  if (device_name.find("Max") != std::string_view::npos) {
    return 40u;
  }
  if (device_name.find("Pro") != std::string_view::npos) {
    return 20u;
  }
  return 64u;
}

// #5309 Skinny NAX (M=9..48 -> RF in {2,3,4,6}) and Wide NAX (M>=49 -> RF=8,
// BM_logical=64). Every slice executes matmul2d_descriptor(8, BN_stored, BK)
// with sequential split-K accumulation for strict 0 ULP RowExactPortableV1 parity.
// Calibrated on 64-core Apple10 M5 Ultra: when split-K > 1, RF=1 (independent
// BM=8 threadgroups across all 64 cores with 1 barrier) beats sequential RF=8
// by 1.51x on 1024-tok prefill; OMLX_M5_NAX_MAX_RF=2..8 enables multi-row slices.
inline U32 nax_select_row_factor(U32 t, U32 max_rf_cap = 1u) {
  U32 max_rf = max_rf_cap;
  if (const char *env = std::getenv("OMLX_M5_NAX_MAX_RF")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v == 1u || v == 2u || v == 3u || v == 4u || v == 6u || v == 8u) {
      max_rf = v;
    }
  }
  auto clamp_rf = [&](U32 rf) -> U32 {
    while (rf > max_rf) {
      if (rf == 8u) rf = 6u;
      else if (rf == 6u) rf = 4u;
      else if (rf == 4u) rf = 3u;
      else if (rf == 3u) rf = 2u;
      else rf = 1u;
    }
    return rf;
  };
  if (t <= 8u) {
    return 1u;
  }
  if (t <= 16u) {
    return clamp_rf(2u);
  }
  if (t <= 24u) {
    return clamp_rf(3u);
  }
  if (t <= 32u) {
    return clamp_rf(4u);
  }
  if (t <= 48u) {
    return clamp_rf(6u);
  }
  return clamp_rf(8u);
}

// Multi-SIMDgroup 2xBN16 selector: pairs two adjacent BN_stored=16 tiles in one
// threadgroup (sharing activation rows in L1 cache) whenever total threadgroups
// still saturate the GPU cores and threadgroup size stays within hardware limits.
inline U32 nax_select_nsg_bn16(U32 padded_n, U32 y_tiles, U32 sk,
                               U32 gpu_cores = 64u,
                               bool allow_sk8_nsg2 = false) {
  if (const char *env = std::getenv("OMLX_M5_NAX_NSG")) {
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v == 1u) return 1u;
    if (v == 2u && padded_n >= 32u && (sk < 8u || allow_sk8_nsg2)) return 2u;
  }
  // Calibrated on 64-core Apple10 M5 Ultra: NSG=2 wins when 32*sk*2 <= 128
  // threads/TG (sk <= 2, e.g. moe_nax down_scatter); at sk >= 4, NSG=1 (128..256
  // threads/TG) achieves +3.1%..+3.7% higher decode/verify throughput via higher
  // core occupancy.
  if (padded_n < 32u || sk > 2u) {
    return 1u;
  }
  const U32 nsg2_x_tiles = (padded_n + 31u) / 32u;
  if (U64(nsg2_x_tiles) * U64(std::max(1u, y_tiles)) >= U64(gpu_cores)) {
    return 2u;
  }
  return 1u;
}

inline U32 nax_y_tiles(U32 mp, U32 rf) {
  const U32 block_rows = 8u * rf;
  return (mp + block_rows - 1u) / block_rows;
}

enum class NaxWidePrefillTile : U32 {
  none = 0u,
  bm32_bn32 = 1u,
  bm64_bn64 = 2u,
};

inline U32 nax_wide_prefill_min_tokens() {
  if (const char *env = std::getenv("OMLX_M5_WIDE_PREFILL_MIN_T")) {
    if (std::string_view(env) == "0") {
      return 0xFFFFFFFFu;
    }
    const U32 v = U32(std::strtoul(env, nullptr, 10));
    if (v >= 16u) {
      return v;
    }
  }
  return 64u;
}

inline NaxWidePrefillTile
nax_select_wide_prefill_tile(U32 padded_n, U32 t,
                             std::optional<U32> explicit_t = {},
                             U32 gpu_cores = 64u) {
  if (explicit_t.has_value()) {
    return NaxWidePrefillTile::none;
  }
  if (const char *tile_env = std::getenv("OMLX_M5_WIDE_PREFILL_TILE")) {
    const std::string_view sv(tile_env);
    if (sv == "0") {
      return NaxWidePrefillTile::none;
    }
    if (sv == "64" && padded_n >= 16u && t >= 16u) {
      return NaxWidePrefillTile::bm64_bn64;
    }
    if (sv == "32" && padded_n >= 16u && t >= 16u) {
      return NaxWidePrefillTile::bm32_bn32;
    }
  }
  if (t < nax_wide_prefill_min_tokens() ||
      std::getenv("OMLX_M5_NAX_MAX_RF") != nullptr) {
    return NaxWidePrefillTile::none;
  }
  const U32 cores = std::max(1u, gpu_cores);
  const U32 mp16 = ((t + 15u) / 16u) * 16u;
  const U64 tgs_64 =
      U64((padded_n + 63u) / 64u) * U64(std::max(1u, (mp16 + 63u) / 64u));
  if (padded_n >= 32u &&
      ((mp16 >= 256u && tgs_64 >= U64(cores)) ||
       tgs_64 >= 4ULL * U64(cores))) {
    return NaxWidePrefillTile::bm64_bn64;
  }
  const U64 tgs_32 =
      U64((padded_n + 31u) / 32u) * U64(std::max(1u, (mp16 + 31u) / 32u));
  if ((padded_n >= 32u && tgs_32 >= U64(std::max(1u, cores / 2u))) ||
      (padded_n >= 16u && tgs_32 >= U64(cores))) {
    return NaxWidePrefillTile::bm32_bn32;
  }
  return NaxWidePrefillTile::none;
}

inline U32 nax_effective_select_t(const Encoder &e,
                                  std::optional<U32> explicit_t) {
  return e.delegated_prefill ? e.profile.capacity
                             : explicit_t.value_or(e.profile.capacity);
}

inline std::optional<U32>
nax_effective_select_explicit_t(const Encoder &e,
                                std::optional<U32> explicit_t) {
  return e.delegated_prefill ? std::nullopt : explicit_t;
}

class NaxSegmentArena {
public:
  struct Placement {
    U32 seg_index = 0;
    U64 offset = 0;
    U64 bytes = 0;
  };

  static constexpr U64 kAlign = 256ULL;
  // shortcut: 1 GiB weight segment cap works around Apple Silicon ICB 32-bit
  // (<4 GiB) buffer offset truncation; revisit only if Metal 4 ICB fixes >4 GiB offsets.
  static constexpr U64 kMaxSegBytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;

  static U64 align_up(U64 v) { return (v + kAlign - 1ULL) & ~(kAlign - 1ULL); }

  Placement plan_alloc(U64 bytes, std::string_view label) {
    const U64 aligned = align_up(bytes);
    if (aligned > kMaxSegBytes) {
      throw std::invalid_argument(std::string(label) +
                                  " exceeds 1 GiB segment cap");
    }
    if (seg_sizes_.empty() || seg_sizes_.back() + aligned > kMaxSegBytes) {
      seg_sizes_.push_back(0);
    }
    const U32 idx = U32(seg_sizes_.size() - 1);
    const U64 off = seg_sizes_.back();
    seg_sizes_.back() += aligned;
    return {idx, off, bytes};
  }

  void materialize(const MetalContext &ctx,
                   std::vector<std::unique_ptr<MetalBuffer>> &segments) const {
    const std::size_t base = segments.size();
    segments.reserve(base + seg_sizes_.size());
    for (const U64 sz : seg_sizes_) {
      segments.push_back(
          std::make_unique<MetalBuffer>(ctx.make_private_buffer(sz)));
    }
  }

  static Ref resolve(const std::vector<std::unique_ptr<MetalBuffer>> &segments,
                     const Placement &p, std::size_t base_seg = 0) {
    return Ref{segments[base_seg + p.seg_index].get(), p.offset, p.bytes};
  }

private:
  std::vector<U64> seg_sizes_;
};

struct NaxOwnerCacheKey {
  const void *owner_key = nullptr;
  U64 variant_key = 0;
  bool operator==(const NaxOwnerCacheKey &) const = default;
};

struct NaxOwnerCacheKeyHash {
  std::size_t operator()(const NaxOwnerCacheKey &k) const noexcept {
    const std::size_t h1 = std::hash<const void *>{}(k.owner_key);
    const std::size_t h2 = std::hash<U64>{}(k.variant_key);
    return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) + (h1 >> 2));
  }
};

template <typename CacheT>
concept NaxIncrementalOwnerCache = requires(CacheT &c) {
  { CacheT::kIncrementalOwnerCache } -> std::convertible_to<bool>;
  { c.build_mutex() } -> std::same_as<std::recursive_mutex &>;
};

template <typename CacheT, typename BuildFn>
inline std::shared_ptr<CacheT>
nax_acquire_owner_cache_variant(const void *owner_key, U64 variant_key,
                                BuildFn &&build_fn) {
  if constexpr (NaxIncrementalOwnerCache<CacheT>) {
    std::shared_ptr<CacheT> cache;
    if (owner_key == nullptr) {
      cache = std::make_shared<CacheT>();
    } else {
      static std::mutex mu;
      static std::unordered_map<NaxOwnerCacheKey, std::weak_ptr<CacheT>,
                                NaxOwnerCacheKeyHash>
          pool;
      std::lock_guard<std::mutex> lock(mu);
      const NaxOwnerCacheKey key{owner_key, variant_key};
      if (auto it = pool.find(key); it != pool.end()) {
        cache = it->second.lock();
        if (!cache) {
          pool.erase(it);
        }
      }
      if (!cache) {
        cache = std::make_shared<CacheT>();
        pool[key] = cache;
      }
    }
    std::lock_guard<std::recursive_mutex> build_lock(cache->build_mutex());
    build_fn(*cache);
    return cache;
  } else {
    if (owner_key == nullptr) {
      auto cache = std::make_shared<CacheT>();
      build_fn(*cache);
      return cache;
    }
    static std::mutex mu;
    static std::unordered_map<NaxOwnerCacheKey, std::weak_ptr<CacheT>,
                              NaxOwnerCacheKeyHash>
        pool;
    std::lock_guard<std::mutex> lock(mu);
    const NaxOwnerCacheKey key{owner_key, variant_key};
    if (auto it = pool.find(key); it != pool.end()) {
      if (auto existing = it->second.lock()) {
        return existing;
      }
      pool.erase(it);
    }
    auto cache = std::make_shared<CacheT>();
    build_fn(*cache);
    pool[key] = cache;
    return cache;
  }
}

template <typename CacheT, typename BuildFn>
inline std::shared_ptr<CacheT> nax_acquire_owner_cache(const void *owner_key,
                                                       BuildFn &&build_fn) {
  return nax_acquire_owner_cache_variant<CacheT>(
      owner_key, 0ULL, std::forward<BuildFn>(build_fn));
}

// Pre-shifted 2 KiB MXFP4 E2M1 x2 byte LUT (4 shifts x 256 packed byte pairs = 2,048 B).
// Passed via Encoder::dynamic(...) (deduplicated into one 256B-aligned MTLBuffer slice
// per compiled plan) so Apple10 kernels avoid program-scope constant arrays (which trigger
// Metal 4 Shader Validation false-positives under icb=1) while hitting L1/constant-cache peak.
inline const std::vector<std::byte> &nax_mxfp4_byte_lut_sh_bytes() {
  static const std::vector<std::byte> table = [] {
    static constexpr std::int8_t kMag[8] = {0, 1, 2, 3, 4, 6, 8, 12};
    std::vector<std::byte> out(4u * 256u * sizeof(std::uint16_t));
    auto *dst = reinterpret_cast<std::uint16_t *>(out.data());
    for (unsigned sh = 0u; sh < 4u; ++sh) {
      for (unsigned b = 0u; b < 256u; ++b) {
        const unsigned n0 = b & 15u;
        const unsigned n1 = (b >> 4u) & 15u;
        const int v0 = (n0 & 8u) ? -(int(kMag[n0 & 7u]) << sh)
                                 : (int(kMag[n0 & 7u]) << sh);
        const int v1 = (n1 & 8u) ? -(int(kMag[n1 & 7u]) << sh)
                                 : (int(kMag[n1 & 7u]) << sh);
        dst[sh * 256u + b] = static_cast<std::uint16_t>(
            static_cast<std::uint8_t>(static_cast<std::int8_t>(v0)) |
            (static_cast<std::uint16_t>(
                 static_cast<std::uint8_t>(static_cast<std::int8_t>(v1)))
             << 8u));
      }
    }
    return out;
  }();
  return table;
}

} // namespace omlx::m5::detail::forward
