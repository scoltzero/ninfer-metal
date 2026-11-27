#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>

#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"

namespace omlx::m5 {

struct RowExactPlaneSlice {
  const MetalBuffer *buffer = nullptr;
  std::uint64_t offset = 0;
  std::uint64_t bytes = 0;
  [[nodiscard]] explicit operator bool() const noexcept {
    return buffer != nullptr && bytes != 0;
  }
};

struct ResolvedKl32Mxfp4Planes {
  RowExactPlaneSlice codes{};
  RowExactPlaneSlice scales{};
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  std::uint32_t bn = 16;
  bool has_fwht64 = false;
};

struct ResolvedKl32AffinePlanes {
  RowExactPlaneSlice codes{};
  RowExactPlaneSlice high{};
  RowExactPlaneSlice sb_tiled{};
  std::uint32_t n = 0;
  std::uint32_t k = 0;
  std::uint32_t bn = 16;
  std::uint32_t bits = 4;
  bool is_lane_v2 = false;
  bool has_fwht64 = false;
};

class Q5High1Cache {
public:
  Q5High1Cache();
  ~Q5High1Cache();
  Q5High1Cache(const Q5High1Cache &) = delete;
  Q5High1Cache &operator=(const Q5High1Cache &) = delete;

  [[nodiscard]] std::optional<RowExactPlaneSlice>
  find_q5_high1(std::uint32_t object_id) const;

  [[nodiscard]] std::optional<RowExactPlaneSlice>
  find_sb_tiled(std::uint32_t object_id) const;

  [[nodiscard]] static std::shared_ptr<Q5High1Cache>
  acquire(const void *owner_key, const MetalContext &ctx,
          const std::filesystem::path &libs,
          std::span<const MaterializedWeightView> candidates);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::optional<ResolvedKl32Mxfp4Planes>
resolve_pretiled_kl32_mxfp4_planes(const MaterializedWeightView &w);

[[nodiscard]] std::optional<ResolvedKl32AffinePlanes>
resolve_pretiled_kl32_affine_planes(const MaterializedWeightView &w,
                                    const Q5High1Cache *cache,
                                    RowExactPlaneSlice zero_plane);

} // namespace omlx::m5
