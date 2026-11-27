#include "affine_nax_cache.h"
#include "bf16_nax.h"
#include "mxfp4_nax_cache.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/quantization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

namespace {

void require(bool ok, std::string_view msg) {
  if (!ok) {
    throw std::runtime_error("STEP2-WIDE-NAX FAIL: " + std::string(msg));
  }
}

struct ScopedEnv {
  std::string name;
  std::optional<std::string> prev;
  ScopedEnv(std::string key, const char *val) : name(std::move(key)) {
    if (const char *old = std::getenv(name.c_str())) {
      prev = std::string(old);
    }
    if (val != nullptr) {
      setenv(name.c_str(), val, 1);
    } else {
      unsetenv(name.c_str());
    }
  }
  ~ScopedEnv() {
    if (prev.has_value()) {
      setenv(name.c_str(), prev->c_str(), 1);
    } else {
      unsetenv(name.c_str());
    }
  }
};

float bf16_val(std::uint16_t b) { return bf16_to_float(b); }

float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }

float decode_e2m1(std::uint32_t code) {
  constexpr std::array<float, 8> mag{0.0f, 0.5f, 1.0f, 1.5f,
                                     2.0f, 3.0f, 4.0f, 6.0f};
  const float v = mag[code & 7u];
  return (code & 8u) ? -v : v;
}

float decode_e8m0(std::uint8_t sc) {
  return std::ldexp(1.0f, sc == 0 ? -127 : int(sc) - 127);
}

std::uint8_t encode_e8m0_round_up_cpu(float value) {
  if (!(value > 0.0f)) {
    return 0u;
  }
  const std::uint32_t bits = std::bit_cast<std::uint32_t>(value);
  std::uint32_t exp = (bits >> 23u) & 0xFFu;
  const std::uint32_t mant = bits & 0x7FFFFFu;
  if (exp == 0u) {
    return 1u;
  }
  if (mant != 0u && exp < 254u) {
    ++exp;
  }
  return std::uint8_t(std::min<std::uint32_t>(exp, 254u));
}

std::uint32_t encode_e2m1_cpu(float value) {
  const std::uint32_t sign = std::signbit(value) ? 8u : 0u;
  const float a = std::abs(value);
  std::uint32_t code = 0u;
  if (a > 5.0f) code = 7u;
  else if (a >= 3.5f) code = 6u;
  else if (a > 2.5f) code = 5u;
  else if (a >= 1.75f) code = 4u;
  else if (a > 1.25f) code = 3u;
  else if (a >= 0.75f) code = 2u;
  else if (a > 0.25f) code = 1u;
  return sign | code;
}

std::vector<float> quantize_a4_cpu(std::span<const float> x, U32 rows, U32 k) {
  std::vector<float> out(U64(rows) * k, 0.0f);
  for (U32 r = 0; r < rows; ++r) {
    for (U32 g = 0; g < k / 32u; ++g) {
      float mx = 0.0f;
      for (U32 l = 0; l < 32u; ++l) {
        mx = std::max(mx, std::abs(x[U64(r) * k + g * 32u + l]));
      }
      const std::uint8_t sc = encode_e8m0_round_up_cpu(mx / 6.0f);
      const float scale = decode_e8m0(sc);
      const float inv = scale > 0.0f ? (1.0f / scale) : 0.0f;
      for (U32 l = 0; l < 32u; ++l) {
        const U32 c = g * 32u + l;
        const std::uint32_t code = encode_e2m1_cpu(x[U64(r) * k + c] * inv);
        out[U64(r) * k + c] = decode_e2m1(code) * scale;
      }
    }
  }
  return out;
}

std::vector<std::uint16_t> read_u16(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 2ULL);
  std::vector<std::uint16_t> out(count);
  std::memcpy(out.data(), raw.data(), count * 2ULL);
  return out;
}

struct OwnedWeight {
  MaterializedWeightView view;
  std::vector<std::unique_ptr<MetalBuffer>> bufs;
  std::vector<float> ref_w;

  void add_plane(const MetalContext &ctx, std::string kind,
                 const void *data, std::size_t byte_len, U64 row_stride = 0) {
    std::vector<std::byte> raw(byte_len);
    std::memcpy(raw.data(), data, byte_len);
    bufs.push_back(std::make_unique<MetalBuffer>(ctx.upload(raw)));
    view.planes.push_back({.kind = std::move(kind),
                           .buffer = bufs.back().get(),
                           .segment_index = U32(bufs.size() - 1),
                           .offset = 0,
                           .bytes = byte_len,
                           .row_stride = row_stride});
  }
};

OwnedWeight make_bf16_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                             U32 seed) {
  OwnedWeight w;
  w.view = {.object_id = obj_id,
            .name = "test.bf16." + std::to_string(obj_id),
            .format = "direct_bf16_v1",
            .layout = "direct_rowmajor_v1",
            .parent_shape = {n, k},
            .padded_k = k,
            .row_count = n,
            .in_features = k};
  std::vector<std::uint16_t> words_bf16(U64(n) * k);
  w.ref_w.resize(U64(n) * k);
  for (U32 r = 0; r < n; ++r) {
    for (U32 c = 0; c < k; ++c) {
      const int v = int(((r * 13u + c * 29u + seed * 17u) % 251u)) - 125;
      const std::uint16_t b = float_to_bf16(float(v) / 128.0f);
      words_bf16[U64(r) * k + c] = b;
      w.ref_w[U64(r) * k + c] = bf16_val(b);
    }
  }
  w.add_plane(ctx, "data", words_bf16.data(), words_bf16.size() * 2ULL,
              U64(k) * 2ULL);
  return w;
}

// Builds both mx_rowsplit_k128_v1 OR pre-tiled mx_nax_tiled_bk128_v1 (for BN_stored=16 or 32).
OwnedWeight make_mxfp4_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                              U32 seed, bool pretiled, U32 bn_override = 0u) {
  OwnedWeight w;
  const U32 bn = bn_override ? bn_override : mxfp4_nax_bn(n);
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  w.view = {.object_id = obj_id,
            .name = "test.mxfp4." + std::to_string(obj_id),
            .format = "mxfp4_e2m1_g32_e8m0_v1",
            .layout = pretiled ? "mx_nax_tiled_bk128_v1" : "mx_rowsplit_k128_v1",
            .parent_shape = {n, k},
            .padded_k = k,
            .row_count = n,
            .in_features = k};
  std::vector<std::uint8_t> raw_codes(U64(n) * (k / 2u), 0);
  std::vector<std::uint8_t> raw_scales(U64(n) * (k / 32u), 127);
  w.ref_w.resize(U64(n) * k, 0.0f);
  for (U32 r = 0; r < n; ++r) {
    for (U32 g = 0; g < k / 32u; ++g) {
      raw_scales[U64(r) * (k / 32u) + g] =
          std::uint8_t(122u + ((r * 3u + g * 5u + seed) % 7u));
    }
    for (U32 c = 0; c < k; ++c) {
      const std::uint32_t code = (r * 7u + c * 11u + seed * 3u) & 0x0Fu;
      raw_codes[U64(r) * (k / 2u) + (c >> 1u)] |=
          std::uint8_t(code << ((c & 1u) * 4u));
      const float sc = decode_e8m0(raw_scales[U64(r) * (k / 32u) + c / 32u]);
      w.ref_w[U64(r) * k + c] = decode_e2m1(code) * sc;
    }
  }
  if (!pretiled) {
    w.add_plane(ctx, "data", raw_codes.data(), raw_codes.size(), k / 2u);
    w.add_plane(ctx, "scale", raw_scales.data(), raw_scales.size(), k / 32u);
    return w;
  }
  const U32 blocks = k / 128u;
  std::vector<std::uint8_t> nax_codes(U64(padded_n / bn) * blocks * bn * 64u, 0);
  std::vector<std::uint8_t> nax_scales(U64(padded_n / bn) * blocks * bn * 4u, 0x7Fu);
  for (U32 r = 0; r < padded_n; ++r) {
    const U32 tile_n = r / bn;
    const U32 rn = r % bn;
    for (U32 blk = 0; blk < blocks; ++blk) {
      const U64 idx = U64(tile_n) * blocks + blk;
      if (r < n) {
        std::memcpy(nax_codes.data() + idx * bn * 64u + rn * 64u,
                    raw_codes.data() + U64(r) * (k / 2u) + blk * 64u, 64u);
        std::memcpy(nax_scales.data() + idx * bn * 4u + rn * 4u,
                    raw_scales.data() + U64(r) * (k / 32u) + blk * 4u, 4u);
      }
    }
  }
  w.add_plane(ctx, "nax_data", nax_codes.data(), nax_codes.size(), 0);
  w.add_plane(ctx, "nax_scale", nax_scales.data(), nax_scales.size(), 0);
  return w;
}

static std::uint8_t expand_1_to_2bit_cpu(std::uint32_t nibble) {
  std::uint32_t out = 0u;
  out |= (nibble & 1u);
  out |= ((nibble >> 1u) & 1u) << 2u;
  out |= ((nibble >> 2u) & 1u) << 4u;
  out |= ((nibble >> 3u) & 1u) << 6u;
  return std::uint8_t(out);
}

static std::uint32_t pi16_cpu(std::uint32_t c4) {
  return ((c4 & 3u) << 2u) | ((c4 >> 2u) & 3u);
}

OwnedWeight make_affine_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                               U32 bits, U32 gs, U32 seed, bool pretiled,
                               U32 bn_override = 0u, bool lane_v2 = false) {
  OwnedWeight w;
  const U32 bn = bn_override ? bn_override : affine_nax_bn(n);
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  w.view = {.object_id = obj_id,
            .name = "test.affine.q" + std::to_string(bits) + "_g" +
                    std::to_string(gs) + "." + std::to_string(obj_id),
            .format = "oq_affine_q" + std::to_string(bits) + "_g" +
                      std::to_string(gs) + "_bf16_v1",
            .layout = pretiled ? (lane_v2 ? (gs == 128u ? "oq_nax_lane_g128_v2"
                                                        : "oq_nax_lane_g64_v2")
                                          : (gs == 128u ? "oq_nax_tiled_g128_v1"
                                                        : "oq_nax_tiled_g64_v1"))
                               : "oq_rowsplit_k128_v1",
            .parent_shape = {n, k},
            .padded_k = k,
            .row_count = n,
            .in_features = k};

  const U32 code_stride = (bits == 3u) ? (k / 4u) : (bits == 8u) ? k : (k / 2u);
  const U32 high_stride =
      (bits == 3u || bits == 5u) ? (k / 8u) : (bits == 6u) ? (k / 4u) : 0u;
  std::vector<std::uint8_t> raw_code(U64(n) * code_stride, 0);
  std::vector<std::uint8_t> raw_high(U64(n) * high_stride, 0);
  std::vector<std::uint16_t> raw_scale(U64(n) * (k / gs));
  std::vector<std::uint16_t> raw_bias(U64(n) * (k / gs));
  w.ref_w.resize(U64(n) * k, 0.0f);

  const U32 qmax = (1u << bits) - 1u;
  for (U32 r = 0; r < n; ++r) {
    for (U32 g = 0; g < k / gs; ++g) {
      const float sc = 0.008f * float(1u + ((r * 3u + g * 7u + seed) % 9u));
      const float bi = -sc * float(qmax) * 0.5f +
                       0.01f * float(int((r + g + seed) % 5u) - 2);
      const std::uint16_t sc_b = float_to_bf16(sc);
      const std::uint16_t bi_b = float_to_bf16(bi);
      raw_scale[U64(r) * (k / gs) + g] = sc_b;
      raw_bias[U64(r) * (k / gs) + g] = bi_b;
    }
    for (U32 c = 0; c < k; ++c) {
      const U32 q = (r * 5u + c * 9u + (c >> 4u) * 3u + seed * 7u) & qmax;
      if (bits == 3u) {
        raw_code[U64(r) * code_stride + (c >> 2u)] |=
            std::uint8_t((q & 3u) << ((c & 3u) * 2u));
        raw_high[U64(r) * high_stride + (c >> 3u)] |=
            std::uint8_t(((q >> 2u) & 1u) << (c & 7u));
      } else if (bits == 8u) {
        raw_code[U64(r) * code_stride + c] = std::uint8_t(q);
      } else {
        raw_code[U64(r) * code_stride + (c >> 1u)] |=
            std::uint8_t((q & 15u) << ((c & 1u) * 4u));
        if (bits == 5u) {
          raw_high[U64(r) * high_stride + (c >> 3u)] |=
              std::uint8_t(((q >> 4u) & 1u) << (c & 7u));
        } else if (bits == 6u) {
          raw_high[U64(r) * high_stride + (c >> 2u)] |=
              std::uint8_t(((q >> 4u) & 3u) << ((c & 3u) * 2u));
        }
      }
      const U64 gi = U64(r) * (k / gs) + (c / gs);
      w.ref_w[U64(r) * k + c] = std::fma(bf16_val(raw_scale[gi]), float(q),
                                         bf16_val(raw_bias[gi]));
    }
  }

  if (!pretiled) {
    const char *ckind =
        (bits == 3u) ? "low_2bit" : (bits == 8u) ? "code" : "base_code";
    w.add_plane(ctx, ckind, raw_code.data(), raw_code.size(), code_stride);
    if (high_stride > 0u) {
      w.add_plane(ctx, (bits == 6u) ? "high_2bit" : "high_bit",
                  raw_high.data(), raw_high.size(), high_stride);
    }
    w.add_plane(ctx, "scale", raw_scale.data(), raw_scale.size() * 2ULL,
                U64(k / gs) * 2ULL);
    w.add_plane(ctx, "bias", raw_bias.data(), raw_bias.size() * 2ULL,
                U64(k / gs) * 2ULL);
    return w;
  }

  const U32 groups = k / 64u;
  const U32 c64 = (bits == 3u) ? 16u : (bits == 8u) ? 64u : 32u;
  const U32 h64 = (bits == 3u) ? 8u : (bits == 5u || bits == 6u) ? 16u : 0u;
  std::vector<std::uint8_t> nax_codes(U64(padded_n / bn) * groups * bn * c64, 0);
  std::vector<std::uint8_t> nax_high(U64(padded_n / bn) * groups * bn * h64, 0);
  std::vector<std::uint32_t> nax_sb(U64(groups) * padded_n, 0);
  for (U32 r = 0; r < n; ++r) {
    const U32 tn = r / bn;
    const U32 rn = r % bn;
    for (U32 g = 0; g < groups; ++g) {
      const U64 idx = U64(tn) * groups + g;
      auto *dc = nax_codes.data() + idx * bn * c64 + rn * c64;
      const auto *sc = raw_code.data() + U64(r) * code_stride + g * c64;
      if (!lane_v2) {
        std::memcpy(dc, sc, c64);
        if (bits == 3u || bits == 6u) {
          std::memcpy(nax_high.data() + idx * bn * h64 + rn * h64,
                      raw_high.data() + U64(r) * high_stride + g * h64, h64);
        } else if (bits == 5u) {
          const auto *sh = raw_high.data() + U64(r) * high_stride + g * 8u;
          auto *dh = nax_high.data() + idx * bn * 16u + rn * 16u;
          for (U32 bi = 0; bi < 8u; ++bi) {
            dh[bi * 2u] = expand_1_to_2bit_cpu(sh[bi] & 0x0Fu);
            dh[bi * 2u + 1u] = expand_1_to_2bit_cpu(sh[bi] >> 4u);
          }
        }
      } else {
        const U32 chunk_bytes = c64 / 16u;
        for (U32 cs = 0; cs < 16u; ++cs) {
          std::memcpy(dc + cs * chunk_bytes, sc + pi16_cpu(cs) * chunk_bytes,
                      chunk_bytes);
        }
        if (bits == 3u) {
          const auto *sh = raw_high.data() + U64(r) * high_stride + g * 8u;
          auto *dh = nax_high.data() + idx * bn * 8u + rn * 8u;
          for (U32 b = 0; b < 8u; ++b) {
            const U32 s0 = pi16_cpu(b * 2u + 0u);
            const U32 s1 = pi16_cpu(b * 2u + 1u);
            const U32 n0 = (U32(sh[s0 >> 1u]) >> ((s0 & 1u) * 4u)) & 0x0Fu;
            const U32 n1 = (U32(sh[s1 >> 1u]) >> ((s1 & 1u) * 4u)) & 0x0Fu;
            dh[b] = std::uint8_t(n0 | (n1 << 4u));
          }
        } else if (bits == 5u) {
          const auto *sh = raw_high.data() + U64(r) * high_stride + g * 8u;
          auto *dh = nax_high.data() + idx * bn * 16u + rn * 16u;
          for (U32 cs = 0; cs < 16u; ++cs) {
            const U32 s = pi16_cpu(cs);
            const U32 nib = (U32(sh[s >> 1u]) >> ((s & 1u) * 4u)) & 0x0Fu;
            dh[cs] = expand_1_to_2bit_cpu(nib);
          }
        } else if (bits == 6u) {
          const auto *sh = raw_high.data() + U64(r) * high_stride + g * 16u;
          auto *dh = nax_high.data() + idx * bn * 16u + rn * 16u;
          for (U32 cs = 0; cs < 16u; ++cs) {
            dh[cs] = sh[pi16_cpu(cs)];
          }
        }
      }
      const U32 sg_idx = (gs == 128u) ? (g >> 1u) : g;
      const std::uint32_t sc_b = raw_scale[U64(r) * (k / gs) + sg_idx];
      const std::uint32_t bi_b = raw_bias[U64(r) * (k / gs) + sg_idx];
      nax_sb[U64(g) * padded_n + r] = sc_b | (bi_b << 16u);
    }
  }
  w.add_plane(ctx, "nax_codes", nax_codes.data(), nax_codes.size(), 0);
  if (bits == 3u) {
    w.add_plane(ctx, "nax_high1", nax_high.data(), nax_high.size(), 0);
  } else if (bits == 5u || bits == 6u) {
    w.add_plane(ctx, "nax_high2", nax_high.data(), nax_high.size(), 0);
  }
  w.add_plane(ctx, "nax_sb", nax_sb.data(), nax_sb.size() * 4ULL, 0);
  return w;
}

std::vector<float> make_activations(U32 rows, U32 k, U32 seed,
                                    std::vector<std::uint16_t> &bf16_out) {
  bf16_out.resize(U64(rows) * k);
  std::vector<float> f32(U64(rows) * k);
  for (U32 r = 0; r < rows; ++r) {
    const float amp = (r % 5u == 0u) ? 0.125f : (r % 5u == 1u) ? 2.5f : 0.75f;
    for (U32 c = 0; c < k; ++c) {
      const int v = int(((r * 31u + c * 17u + seed * 13u) % 257u)) - 128;
      const std::uint16_t b = float_to_bf16(amp * float(v) / 128.0f);
      bf16_out[U64(r) * k + c] = b;
      f32[U64(r) * k + c] = bf16_val(b);
    }
  }
  return f32;
}

std::vector<float> cpu_linear_ref(std::span<const float> x,
                                  std::span<const float> w, U32 m, U32 n,
                                  U32 k) {
  std::vector<float> y(U64(m) * n, 0.0f);
  for (U32 r = 0; r < m; ++r) {
    for (U32 c = 0; c < n; ++c) {
      double acc = 0.0;
      for (U32 d = 0; d < k; ++d) {
        acc += double(x[U64(r) * k + d]) * double(w[U64(c) * k + d]);
      }
      y[U64(r) * n + c] = float(acc);
    }
  }
  return y;
}

std::vector<float> cpu_swiglu_ref(std::span<const float> x,
                                  std::span<const float> wg,
                                  std::span<const float> wu, U32 m, U32 n,
                                  U32 k) {
  const auto g = cpu_linear_ref(x, wg, m, n, k);
  const auto u = cpu_linear_ref(x, wu, m, n, k);
  std::vector<float> y(U64(m) * n);
  for (std::size_t i = 0; i < y.size(); ++i) {
    const float gv = bf16_val(float_to_bf16(g[i]));
    const float uv = bf16_val(float_to_bf16(u[i]));
    y[i] = bf16_val(float_to_bf16(silu_ref(gv) * uv));
  }
  return y;
}

void check_close(std::span<const std::uint16_t> actual_bf16,
                 std::span<const float> ref, double max_rel_tol,
                 double per_elem_rel_tol, std::string_view label) {
  require(actual_bf16.size() >= ref.size(),
          std::string(label) + ": actual buffer smaller than ref");
  double ref2 = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    ref2 += double(ref[i]) * double(ref[i]);
  }
  const double rms_ref =
      ref.empty() ? 1.0 : std::sqrt(ref2 / double(ref.size()));
  double err2 = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const float a = bf16_val(actual_bf16[i]);
    require(std::isfinite(a), std::string(label) + ": non-finite output");
    const double diff = std::abs(double(a) - double(ref[i]));
    const double elem_tol =
        per_elem_rel_tol *
        std::max(std::max(1.0, rms_ref), std::abs(double(ref[i])));
    if (diff > elem_tol) {
      throw std::runtime_error(
          std::string(label) + " element mismatch at i=" + std::to_string(i) +
          " actual=" + std::to_string(a) + " ref=" + std::to_string(ref[i]) +
          " diff=" + std::to_string(diff) + " tol=" + std::to_string(elem_tol));
    }
    err2 += diff * diff;
  }
  const double rel = std::sqrt(err2 / std::max(ref2, 1e-12));
  if (rel > max_rel_tol) {
    throw std::runtime_error(std::string(label) + " rel_l2 exceeded: rel=" +
                             std::to_string(rel) + " tol=" +
                             std::to_string(max_rel_tol));
  }
}

void patch_and_submit(const MetalContext &ctx, Profile &p, U32 live_rows,
                      MetalBuffer &y_buf, U32 max_rows, U32 n, bool use_icb) {
  for (const auto &patch : p.patches) {
    require(!patch.position, "unexpected position patch in wide NAX test");
    const U32 v = live_rows * patch.multiplier;
    std::memcpy(p.parameter_data.data() + patch.offset + patch.word * 4, &v, 4);
  }
  p.parameters.write("step", p.parameter_data);
  std::vector<std::uint16_t> sentinel(U64(max_rows) * n, 0xA5A5u);
  y_buf.write(std::as_bytes(std::span(sentinel)));
  if (use_icb && p.icb.has_value()) {
    (void)ctx.replay_icb(*p.icb);
  } else {
    const std::array calls{MetalCommandInvocation{
        .plan = &p.plan, .bindings = p.bindings, .parameters = &p.parameters}};
    (void)ctx.execute_batch(calls);
  }
  const auto all = read_u16(y_buf, U64(max_rows) * n);
  for (U64 idx = U64(live_rows) * n; idx < all.size(); ++idx) {
    if (all[idx] != 0xA5A5u) {
      throw std::runtime_error("tail row sentinel clobbered at idx=" +
                               std::to_string(idx) + " live_rows=" +
                               std::to_string(live_rows));
    }
  }
}

Json test_tile_selector() {
  {
    ScopedEnv clear_min("OMLX_M5_WIDE_PREFILL_MIN_T", nullptr);
    ScopedEnv clear_tile("OMLX_M5_WIDE_PREFILL_TILE", nullptr);
    ScopedEnv clear_rf("OMLX_M5_NAX_MAX_RF", nullptr);
    for (U32 m : {1u, 2u, 4u, 8u, 16u, 32u, 63u}) {
      require(nax_select_wide_prefill_tile(4096u, m, std::nullopt, 64u) ==
                  NaxWidePrefillTile::none,
              "M < 64 entered wide prefill tile");
    }
    require(nax_select_wide_prefill_tile(4096u, 128u, std::optional<U32>(128u),
                                         64u) == NaxWidePrefillTile::none,
            "explicit_t (packed_batch) entered wide prefill tile");
    require(nax_select_wide_prefill_tile(4096u, 64u, std::nullopt, 64u) ==
                NaxWidePrefillTile::bm32_bn32,
            "N=4096 M=64 did not select BM=32x32 wave-tail anti-stall tile");
    require(nax_select_wide_prefill_tile(17408u, 128u, std::nullopt, 64u) ==
                NaxWidePrefillTile::bm64_bn64,
            "N=17408 M=128 (544 TGs >= 4*64) did not select BM=64x64");
    require(nax_select_wide_prefill_tile(5120u, 128u, std::nullopt, 64u) ==
                NaxWidePrefillTile::bm32_bn32,
            "N=5120 M=128 (160 TGs < 4*64) did not select BM=32x32");
    require(nax_select_wide_prefill_tile(1024u, 64u, std::nullopt, 64u) ==
                NaxWidePrefillTile::bm32_bn32,
            "N=1024 M=64 did not select BM=32x32 occupancy fallback");
    require(nax_select_wide_prefill_tile(1024u, 256u, std::nullopt, 64u) ==
                NaxWidePrefillTile::bm64_bn64,
            "N=1024 M=256 did not promote to BM=64x64");
    require(nax_select_wide_prefill_tile(64u, 64u, std::nullopt, 64u) ==
                NaxWidePrefillTile::none,
            "N=64 M=64 did not preserve Split-K saturation");
    require(nax_select_wide_prefill_tile(64u, 512u, std::nullopt, 64u) ==
                NaxWidePrefillTile::bm32_bn32,
            "N=64 M=512 (35B-A3B gdn_a+b) did not select BM=32x32");
    require(nax_select_wide_prefill_tile(64u, 1024u, std::nullopt, 64u) ==
                NaxWidePrefillTile::bm32_bn32,
            "N=64 M=1024 did not select BM=32x32");
  }
  return {{"passed", true}};
}

Json test_pso_and_tg_sram_budget(const MetalContext &ctx,
                                 const std::filesystem::path &libs) {
  const auto mxfp4_path = libs / "omlx_m5_mxfp4_nax.metallib";
  const auto affine_path = libs / "omlx_m5_affine_nax.metallib";
  const auto bf16_path = libs / "omlx_m5_bf16_nax.metallib";
  if (!std::filesystem::is_regular_file(mxfp4_path) ||
      !std::filesystem::is_regular_file(affine_path) ||
      !std::filesystem::is_regular_file(bf16_path) ||
      !ctx.capabilities().profile.probe.metal4_compiler) {
    return {{"skipped", true}, {"reason", "NAX metallibs or Metal4 compiler absent"}};
  }

  ctx.register_library({.key = "step2.mxfp4", .path = mxfp4_path});
  ctx.register_library({.key = "step2.affine", .path = affine_path});
  ctx.register_library({.key = "step2.bf16", .path = bf16_path});

  U32 compiled = 0;
  U64 max_mxfp4_tg = 0, max_bf16_tg = 0, max_affine_no_q3_tg = 0, max_q3_tg = 0;

  const bool shader_val =
      std::getenv("MTL_SHADER_VALIDATION") != nullptr &&
      std::string_view(std::getenv("MTL_SHADER_VALIDATION")) == "1";
  const U64 zero_cap = shader_val ? 4096ULL : 0ULL;
  const U64 lin_sram_cap = shader_val ? 20480ULL : 9216ULL;
  const U64 swiglu_sram_cap = shader_val ? 20480ULL : 9216ULL;

  const std::vector<std::string> mxfp4_fns = {
      "omlx_m5_mxfp4_nax_bm64_bn64_bns16_sk1",
      "omlx_m5_mxfp4_nax_bm64_bn64_bns32_sk1",
      "omlx_m5_mxfp4_nax_bm32_bn32_bns16_sk1",
      "omlx_m5_mxfp4_nax_bm32_bn32_bns32_sk1",
      "omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns16_sk1",
      "omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns32_sk1",
      "omlx_m5_mxfp4_nax_bf16_bm32_bn32_bns16_sk1",
      "omlx_m5_mxfp4_nax_bf16_bm32_bn32_bns32_sk1",
      "omlx_m5_mxfp4_nax_swiglu_bm64_bn64_sk1",
      "omlx_m5_mxfp4_nax_swiglu_f16_bm64_bn64_sk1",
      "omlx_m5_mxfp4_nax_swiglu_bm32_bn32_sk1",
      "omlx_m5_mxfp4_nax_swiglu_f16_bm32_bn32_sk1",
      "omlx_m5_mxfp4_nax_swiglu_bf16_bm64_bn64_sk1",
      "omlx_m5_mxfp4_nax_swiglu_bf16_bm32_bn32_sk1",
      "omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns16_sk1",
      "omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns32_sk1",
      "omlx_m5_mxfp4_w4a4_nax_bm32_bn32_bns16_sk1",
      "omlx_m5_mxfp4_w4a4_nax_bm32_bn32_bns32_sk1",
      "omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_sk1",
      "omlx_m5_mxfp4_w4a4_nax_swiglu_bm32_bn32_sk1",
  };
  for (const auto &fn : mxfp4_fns) {
    const auto info = ctx.warmup_pipeline({.key = "step2.pso." + fn,
                                           .library_key = "step2.mxfp4",
                                           .function_name = fn,
                                           .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= 32768ULL,
            fn + " exceeded 32 KiB Apple10 TG SRAM ceiling");
    if (fn.find("_w4a4_") != std::string::npos) {
      require(info.static_threadgroup_memory_bytes <= zero_cap,
              fn + " w4a4 unexpectedly used static TG SRAM");
    } else if (fn.find("_swiglu_") != std::string::npos) {
      require(info.static_threadgroup_memory_bytes <= swiglu_sram_cap,
              fn + " swiglu TG SRAM exceeded budget");
    } else {
      require(info.static_threadgroup_memory_bytes <= lin_sram_cap,
              fn + " linear TG SRAM exceeded budget");
    }
    max_mxfp4_tg = std::max(max_mxfp4_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }

  for (U32 bk : {64u, 128u}) {
    for (const char *sfx : {"_bm64_bn64_sk1", "_bm32_bn32_sk1"}) {
      for (const char *pfx : {"omlx_m5_bf16_nax_bk", "omlx_m5_bf16_nax_swiglu_bk"}) {
        const std::string fn = std::string(pfx) + std::to_string(bk) + sfx;
        const auto info = ctx.warmup_pipeline({.key = "step2.pso." + fn,
                                               .library_key = "step2.bf16",
                                               .function_name = fn,
                                               .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= 32768ULL,
                fn + " exceeded 32 KiB Apple10 TG SRAM ceiling");
        const U64 cap = (fn.find("_swiglu_") != std::string::npos)
                            ? swiglu_sram_cap
                            : lin_sram_cap;
        require(info.static_threadgroup_memory_bytes <= cap,
                fn + " BF16 TG SRAM exceeded budget");
        max_bf16_tg = std::max(max_bf16_tg, info.static_threadgroup_memory_bytes);
        ++compiled;
      }
    }
  }

  for (const char *tag : {"q4", "q56", "q3", "q8"}) {
    for (const char *gs : {"g64", "g128"}) {
      for (const char *tile : {"bm64_bn64_bns16_sk1", "bm64_bn64_bns32_sk1",
                               "bm32_bn32_bns16_sk1", "bm32_bn32_bns32_sk1",
                               "swiglu_bm64_bn64_sk1", "swiglu_bm32_bn32_sk1"}) {
        const std::string fn = std::string("omlx_m5_affine_nax_") + tag + "_" +
                               gs + "_" + tile;
        const auto info = ctx.warmup_pipeline({.key = "step2.pso." + fn,
                                               .library_key = "step2.affine",
                                               .function_name = fn,
                                               .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= 32768ULL,
                fn + " exceeded 32 KiB Apple10 TG SRAM ceiling");
        const U64 cap = (std::string_view(tile).find("swiglu_") != std::string_view::npos)
                            ? swiglu_sram_cap
                            : lin_sram_cap;
        require(info.static_threadgroup_memory_bytes <= cap,
                fn + " affine TG SRAM exceeded budget");
        if (std::string_view(tag) == "q3") {
          max_q3_tg = std::max(max_q3_tg, info.static_threadgroup_memory_bytes);
        } else {
          max_affine_no_q3_tg =
              std::max(max_affine_no_q3_tg, info.static_threadgroup_memory_bytes);
        }
        ++compiled;
      }
    }
  }

  for (const char *fn : {"omlx_m5_affine_nax_q5_h1_bm16_bns16_sk2",
                         "omlx_m5_affine_nax_q5_h1_bm16_bns16_sk4",
                         "omlx_m5_affine_nax_q5_h1_bm16_bns32_sk2",
                         "omlx_m5_affine_nax_q5_h1_bm16_bns32_sk4",
                         "omlx_m5_affine_nax_q5_h1_swiglu_bm16_sk2",
                         "omlx_m5_affine_nax_q5_h1_swiglu_bm16_sk4"}) {
    const auto info = ctx.warmup_pipeline({.key = std::string("step2.pso.") + fn,
                                           .library_key = "step2.affine",
                                           .function_name = fn,
                                           .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <=
                (shader_val ? 32768ULL : 16384ULL),
            std::string(fn) + " exceeded 16 KiB Apple10 TG SRAM ceiling");
    ++compiled;
  }

  return {{"compiled_psos", compiled},
          {"max_mxfp4_tg_bytes", max_mxfp4_tg},
          {"max_bf16_tg_bytes", max_bf16_tg},
          {"max_affine_q4568_tg_bytes", max_affine_no_q3_tg},
          {"max_affine_q3_tg_bytes", max_q3_tg},
          {"passed", true}};
}

Json test_live_gpu_accuracy_and_parity(const MetalContext &ctx,
                                       const std::filesystem::path &libs) {
  const auto mxfp4_path = libs / "omlx_m5_mxfp4_nax.metallib";
  const auto affine_path = libs / "omlx_m5_affine_nax.metallib";
  const auto bf16_path = libs / "omlx_m5_bf16_nax.metallib";
  const auto &probe = ctx.capabilities().profile.probe;
  if (!std::filesystem::is_regular_file(mxfp4_path) ||
      !std::filesystem::is_regular_file(affine_path) ||
      !std::filesystem::is_regular_file(bf16_path) ||
      probe.runtime_os_major < 27 || !probe.metal4_compiler || !probe.tensor_api) {
    return {{"skipped", true}, {"reason", "Metal 4 runtime unavailable"}};
  }

  constexpr U32 kMaxM = 128u;
  constexpr U32 kK = 512u;
  constexpr U32 kN = 77u;      // non-multiple-of-16 & odd-16-tile (padded_n=80 -> 5x16: exercises col1_active=false on tile 4!)
  constexpr U32 kN2 = 96u;     // even-32 multiple for SwiGLU & bns32

  std::vector<std::uint16_t> x_words;
  const auto x_f32 = make_activations(kMaxM, kK, 1u, x_words);
  auto x_buf = ctx.upload(std::as_bytes(std::span(x_words)));
  Ref x_ref{&x_buf, 0, x_buf.size()};

  auto zero_buf = ctx.make_buffer(256);
  zero_buf.write(std::vector<std::byte>(256));
  Ref zero_ref{&zero_buf, 0, zero_buf.size()};

  auto f16_scratch_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * kK * 2ULL);
  auto f16_rs_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * 4ULL);
  auto f16_mid_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * 512u * 2ULL);
  std::vector<float> ones_vec(kMaxM + 16u, 1.0f);
  auto f16_ones_buf = ctx.upload(std::as_bytes(std::span(ones_vec)));
  auto a4_codes_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * (kK / 2u));
  auto a4_scales_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * (kK / 32u));
  auto xs_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * (kK / 64u) * 4ULL);
  auto y_buf = ctx.make_buffer(U64(kMaxM) * 512u * 2ULL);

  Ref f16_scratch{&f16_scratch_buf, 0, f16_scratch_buf.size()};
  Ref f16_rs{&f16_rs_buf, 0, f16_rs_buf.size()};
  Ref f16_mid{&f16_mid_buf, 0, f16_mid_buf.size()};
  Ref f16_ones{&f16_ones_buf, 0, f16_ones_buf.size()};
  Ref a4_codes{&a4_codes_buf, 0, a4_codes_buf.size()};
  Ref a4_scales{&a4_scales_buf, 0, a4_scales_buf.size()};
  Ref xs_ref{&xs_buf, 0, xs_buf.size()};
  Ref y_ref{&y_buf, 0, y_buf.size()};

  U32 checks = 0;

  // 1. BF16 NAX (bk128 K=512 + bk64 K=320, BM=64x64 & BM=32x32, interior M=128 & tail M=73, M<64 0-ULP)
  for (U32 k_bf : {512u, 320u}) {
    std::vector<std::uint16_t> xk_words;
    const auto xk_f32 = make_activations(kMaxM, k_bf, 3u + k_bf, xk_words);
    auto xk_buf = ctx.upload(std::as_bytes(std::span(xk_words)));
    Ref xk_ref{&xk_buf, 0, xk_buf.size()};

    auto w_bf = make_bf16_weight(ctx, 100u + k_bf, kN, k_bf, 11u);
    auto wg_bf = make_bf16_weight(ctx, 200u + k_bf, kN2, k_bf, 21u);
    auto wu_bf = make_bf16_weight(ctx, 300u + k_bf, kN2, k_bf, 31u);
    const auto ref_lin_128 = cpu_linear_ref(xk_f32, w_bf.ref_w, kMaxM, kN, k_bf);
    const auto ref_swi_128 = cpu_swiglu_ref(xk_f32, wg_bf.ref_w, wu_bf.ref_w, kMaxM, kN2, k_bf);

    for (const char *tile_mode : {"64", "32"}) {
      ScopedEnv env_tile("OMLX_M5_WIDE_PREFILL_TILE", tile_mode);
      {
        Encoder e(ctx, libs, kMaxM, "step2.bf16.lin." + std::to_string(k_bf) + "." + tile_mode);
        e.begin_stage("lin");
        linear_bf16_nax(e, w_bf.view, xk_ref, y_ref, kN, k_bf, std::nullopt, 1u);
        auto prof = e.finish(true);
        const std::string expected_sub = std::string(tile_mode) == "64" ? "bm64_bn64_sk1" : "bm32_bn32_sk1";
        require(prof.weight_routes.at(0).function.find(expected_sub) != std::string::npos,
                "BF16 linear did not select " + expected_sub);
        for (U32 live : {73u, 128u}) {
          patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN, live == 73u);
          const auto out = read_u16(y_buf, U64(live) * kN);
          check_close(out, std::span(ref_lin_128).first(U64(live) * kN), 0.01, 0.05,
                      "BF16 linear k=" + std::to_string(k_bf) + " tile=" + tile_mode);
          ++checks;
        }
      }
      {
        Encoder e(ctx, libs, kMaxM, "step2.bf16.swi." + std::to_string(k_bf) + "." + tile_mode);
        e.begin_stage("swi");
        linear_bf16_nax_swiglu(e, wg_bf.view, wu_bf.view, xk_ref, y_ref, kN2, k_bf, 1u);
        auto prof = e.finish(true);
        for (U32 live : {73u, 128u}) {
          patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN2, live == 128u);
          const auto out = read_u16(y_buf, U64(live) * kN2);
          check_close(out, std::span(ref_swi_128).first(U64(live) * kN2), 0.015, 0.08,
                      "BF16 swiglu k=" + std::to_string(k_bf) + " tile=" + tile_mode);
          ++checks;
        }
      }
    }

    // M < 64 0-ULP bit-identity check (M = 31)
    {
      std::vector<std::uint16_t> out_default, out_disabled;
      std::string fn_default, fn_disabled;
      {
        ScopedEnv clear_tile("OMLX_M5_WIDE_PREFILL_TILE", nullptr);
        Encoder e(ctx, libs, 31u, "step2.bf16.m31.def." + std::to_string(k_bf));
        e.begin_stage("lin");
        linear_bf16_nax(e, w_bf.view, xk_ref, y_ref, kN, k_bf);
        auto prof = e.finish();
        fn_default = prof.weight_routes.at(0).function;
        patch_and_submit(ctx, prof, 31u, y_buf, kMaxM, kN, false);
        out_default = read_u16(y_buf, 31ULL * kN);
      }
      {
        ScopedEnv dis("OMLX_M5_WIDE_PREFILL_MIN_T", "0");
        Encoder e(ctx, libs, 31u, "step2.bf16.m31.dis." + std::to_string(k_bf));
        e.begin_stage("lin");
        linear_bf16_nax(e, w_bf.view, xk_ref, y_ref, kN, k_bf);
        auto prof = e.finish();
        fn_disabled = prof.weight_routes.at(0).function;
        patch_and_submit(ctx, prof, 31u, y_buf, kMaxM, kN, false);
        out_disabled = read_u16(y_buf, 31ULL * kN);
      }
      require(fn_default == fn_disabled && out_default == out_disabled,
              "BF16 M=31 (<64) 0-ULP bit-identity broken");
      ++checks;
    }
  }

  // 2. MXFP4 W4A16 & W4A4 (zero-copy pre-tiled bns16 & bns32 + GPU-repacked, BM=64x64 & BM=32x32, SwiGLU BF16 & F16->down_proj)
  auto mx_raw = make_mxfp4_weight(ctx, 401u, kN, kK, 41u, false);
  auto mx_pre16 = make_mxfp4_weight(ctx, 402u, kN, kK, 41u, true, 16u);
  auto mx_pre32 = make_mxfp4_weight(ctx, 403u, kN, kK, 41u, true, 32u);
  auto mx_g = make_mxfp4_weight(ctx, 404u, kN2, kK, 42u, true, 16u);
  auto mx_u = make_mxfp4_weight(ctx, 405u, kN2, kK, 43u, true, 16u);

  Mxfp4NaxCache mx_cache;
  mx_cache.register_weight(mx_raw.view);
  mx_cache.register_weight(mx_pre16.view);
  mx_cache.register_weight(mx_g.view);
  mx_cache.register_weight(mx_u.view);
  mx_cache.build(ctx, libs);

  const TiledMxfp4Weight tw_mx_bns32{
      plane(mx_pre32.view, "nax_data"), plane(mx_pre32.view, "nax_scale"),
      kN, kK, 32u, ((kN + 31u) / 32u) * 32u, 2u};

  const auto x_a4_f32 = quantize_a4_cpu(x_f32, kMaxM, kK);
  const auto ref_mx_lin = cpu_linear_ref(x_f32, mx_raw.ref_w, kMaxM, kN, kK);
  const auto ref_mx_swi = cpu_swiglu_ref(x_f32, mx_g.ref_w, mx_u.ref_w, kMaxM, kN2, kK);
  const auto ref_mx_w4a4_lin = cpu_linear_ref(x_a4_f32, mx_raw.ref_w, kMaxM, kN, kK);
  const auto ref_mx_w4a4_swi = cpu_swiglu_ref(x_a4_f32, mx_g.ref_w, mx_u.ref_w, kMaxM, kN2, kK);

  for (const char *tile_mode : {"64", "32"}) {
    ScopedEnv env_tile("OMLX_M5_WIDE_PREFILL_TILE", tile_mode);
    for (const TiledMxfp4Weight *tw :
         {mx_cache.find(401u), mx_cache.find(402u), &tw_mx_bns32}) {
      require(tw != nullptr, "TiledMxfp4Weight missing");
      // W4A16 linear (zero-cast BF16 input on Wide SK=1: 1 dispatch, 0 ensure_f16_cast!)
      {
        Encoder e(ctx, libs, kMaxM,
                  "step2.mx.w4a16." + std::to_string(tw->bn) + "." + tile_mode);
        e.begin_stage("lin");
        linear_mxfp4_nax(e, *tw, x_ref, y_ref, f16_scratch, f16_rs, std::nullopt, 1u);
        auto prof = e.finish(true);
        require(prof.plan.dispatch_count() == 1u,
                "Wide SK=1 linear_mxfp4_nax unexpectedly emitted ensure_f16_cast dispatch");
        for (U32 live : {73u, 128u}) {
          patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN, live == 73u);
          const auto out = read_u16(y_buf, U64(live) * kN);
          check_close(out, std::span(ref_mx_lin).first(U64(live) * kN), 0.01, 0.05,
                      "MXFP4 W4A16 linear bns=" + std::to_string(tw->bn));
          ++checks;
        }
      }
      // W4A4 linear
      {
        Encoder e(ctx, libs, kMaxM,
                  "step2.mx.w4a4." + std::to_string(tw->bn) + "." + tile_mode);
        e.begin_stage("lin_a4");
        linear_mxfp4_w4a4_nax(e, *tw, x_ref, y_ref, a4_codes, a4_scales, std::nullopt, 1u);
        auto prof = e.finish(true);
        for (U32 live : {73u, 128u}) {
          patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN, live == 128u);
          const auto out = read_u16(y_buf, U64(live) * kN);
          check_close(out, std::span(ref_mx_w4a4_lin).first(U64(live) * kN),
                      0.01, 0.02,
                      "MXFP4 W4A4 linear bns=" + std::to_string(tw->bn));
          ++checks;
        }
      }
    }

    // W4A16 SwiGLU (BF16 in + F16 in + F16 out) & W4A4 SwiGLU
    const auto *gw_mx = mx_cache.find(404u);
    const auto *uw_mx = mx_cache.find(405u);
    {
      Encoder e(ctx, libs, kMaxM, std::string("step2.mx.swi.bf16_in.") + tile_mode);
      e.begin_stage("swi_bf16");
      linear_mxfp4_nax_swiglu(e, *gw_mx, *uw_mx, x_ref, f16_scratch, f16_rs, y_ref, 1u);
      auto prof = e.finish(true);
      require(prof.plan.dispatch_count() == 1u,
              "Wide SK=1 linear_mxfp4_nax_swiglu unexpectedly emitted ensure_f16_cast dispatch");
      for (U32 live : {73u, 128u}) {
        patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN2, true);
        const auto out = read_u16(y_buf, U64(live) * kN2);
        check_close(out, std::span(ref_mx_swi).first(U64(live) * kN2), 0.015, 0.08,
                    "MXFP4 W4A16 SwiGLU BF16-in");
        ++checks;
      }
    }
    {
      Encoder e(ctx, libs, kMaxM, std::string("step2.mx.swi.bf16.") + tile_mode);
      e.begin_stage("swi");
      ensure_f16_cast(e, x_ref, f16_scratch, f16_rs, kK);
      linear_mxfp4_nax_swiglu_direct_f16(e, *gw_mx, *uw_mx, f16_scratch, f16_rs,
                                         y_ref, false, 1u);
      auto prof = e.finish(true);
      for (U32 live : {73u, 128u}) {
        patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN2, true);
        const auto out = read_u16(y_buf, U64(live) * kN2);
        check_close(out, std::span(ref_mx_swi).first(U64(live) * kN2), 0.015, 0.08,
                    "MXFP4 W4A16 SwiGLU BF16");
        ++checks;
      }
    }
    {
      // Exercise OutF16 SwiGLU (N=512) -> down_proj (K=512, N=77) direct-F16 chain at tail M=73!
      auto mx_g512 = make_mxfp4_weight(ctx, 406u, 512u, kK, 44u, true, 16u);
      auto mx_u512 = make_mxfp4_weight(ctx, 407u, 512u, kK, 45u, true, 16u);
      auto mx_down = make_mxfp4_weight(ctx, 408u, kN, 512u, 46u, true, 16u);
      Mxfp4NaxCache mlp_cache;
      mlp_cache.register_weight(mx_g512.view);
      mlp_cache.register_weight(mx_u512.view);
      mlp_cache.register_weight(mx_down.view);
      mlp_cache.build(ctx, libs);
      require(mlp_cache.find(406u) != nullptr && mlp_cache.find(407u) != nullptr &&
                  mlp_cache.find(408u) != nullptr,
              "MLP F16 chain weights missing");
      const auto ref_mid = cpu_swiglu_ref(x_f32, mx_g512.ref_w, mx_u512.ref_w, 73u, 512u, kK);
      const auto ref_mlp = cpu_linear_ref(ref_mid, mx_down.ref_w, 73u, kN, 512u);

      Encoder e(ctx, libs, kMaxM, std::string("step2.mx.swi.f16.") + tile_mode);
      e.begin_stage("mlp_f16");
      ensure_f16_cast(e, x_ref, f16_scratch, f16_rs, kK);
      linear_mxfp4_nax_swiglu_direct_f16(e, *mlp_cache.find(406u), *mlp_cache.find(407u),
                                         f16_scratch, f16_rs, f16_mid, true, 1u);
      linear_mxfp4_nax_direct_f16(e, *mlp_cache.find(408u), f16_mid, f16_ones,
                                  y_ref, std::nullopt, 1u);
      auto prof = e.finish(true);
      patch_and_submit(ctx, prof, 73u, y_buf, kMaxM, kN, true);
      const auto out = read_u16(y_buf, 73ULL * kN);
      check_close(out, ref_mlp, 0.02, 0.05, "MXFP4 W4A16 F16 SwiGLU->down chain");
      ++checks;
    }
    {
      Encoder e(ctx, libs, kMaxM, std::string("step2.mx.w4a4.swi.") + tile_mode);
      e.begin_stage("swi_a4");
      linear_mxfp4_w4a4_nax_swiglu(e, *gw_mx, *uw_mx, x_ref, y_ref, a4_codes,
                                   a4_scales, 1u);
      auto prof = e.finish(true);
      for (U32 live : {73u, 128u}) {
        patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN2, false);
        const auto out = read_u16(y_buf, U64(live) * kN2);
        check_close(out, std::span(ref_mx_w4a4_swi).first(U64(live) * kN2),
                    0.015, 0.03, "MXFP4 W4A4 SwiGLU");
        ++checks;
      }
    }
  }

  // MXFP4 raw vs zero-copy pre-tiled 0-ULP equivalence at M=73 & M < 64 0-ULP preservation
  {
    ScopedEnv env_tile("OMLX_M5_WIDE_PREFILL_TILE", "64");
    Encoder e_raw(ctx, libs, kMaxM, "step2.mx.raw_eq");
    e_raw.begin_stage("lin");
    linear_mxfp4_nax(e_raw, *mx_cache.find(401u), x_ref, y_ref, f16_scratch, f16_rs, std::nullopt, 1u);
    auto p_raw = e_raw.finish();
    patch_and_submit(ctx, p_raw, 73u, y_buf, kMaxM, kN, false);
    const auto out_raw = read_u16(y_buf, 73ULL * kN);

    Encoder e_pre(ctx, libs, kMaxM, "step2.mx.pre_eq");
    e_pre.begin_stage("lin");
    linear_mxfp4_nax(e_pre, *mx_cache.find(402u), x_ref, y_ref, f16_scratch, f16_rs, std::nullopt, 1u);
    auto p_pre = e_pre.finish();
    patch_and_submit(ctx, p_pre, 73u, y_buf, kMaxM, kN, false);
    const auto out_pre = read_u16(y_buf, 73ULL * kN);
    require(out_raw == out_pre, "MXFP4 GPU-repacked vs zero-copy pre-tiled 0-ULP mismatch");
    ++checks;
  }
  {
    ScopedEnv clear_tile("OMLX_M5_WIDE_PREFILL_TILE", nullptr);
    Encoder e1(ctx, libs, 17u, "step2.mx.m17.def");
    e1.begin_stage("lin");
    linear_mxfp4_nax(e1, *mx_cache.find(402u), x_ref, y_ref, f16_scratch, f16_rs);
    auto p1 = e1.finish();
    patch_and_submit(ctx, p1, 17u, y_buf, kMaxM, kN, false);
    const auto out1 = read_u16(y_buf, 17ULL * kN);

    ScopedEnv dis("OMLX_M5_WIDE_PREFILL_MIN_T", "0");
    Encoder e2(ctx, libs, 17u, "step2.mx.m17.dis");
    e2.begin_stage("lin");
    linear_mxfp4_nax(e2, *mx_cache.find(402u), x_ref, y_ref, f16_scratch, f16_rs);
    auto p2 = e2.finish();
    patch_and_submit(ctx, p2, 17u, y_buf, kMaxM, kN, false);
    const auto out2 = read_u16(y_buf, 17ULL * kN);
    require(out1 == out2, "MXFP4 M=17 (<64) 0-ULP bit-identity broken");
    ++checks;
  }

  // 3. Affine NAX: all 10 (q3, q4, q5, q6, q8) x (g64, g128) combinations,
  // testing zero-copy pre-tiled bns16 & bns32, GPU-repack, BM=64x64 & BM=32x32,
  // interior M=128 + tail M=73, SwiGLU, mixed g64+g128 SwiGLU/concat, and M<64 0-ULP!
  U32 aff_id = 1000u;
  for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
    for (U32 gs : {64u, 128u}) {
      const U32 id_raw = aff_id++;
      const U32 id_pre16 = aff_id++;
      const U32 id_pre32 = aff_id++;
      const U32 id_g = aff_id++;
      const U32 id_u = aff_id++;

      auto w_raw = make_affine_weight(ctx, id_raw, kN, kK, bits, gs, bits * 10u + gs, false);
      auto w_pre16 = make_affine_weight(ctx, id_pre16, kN, kK, bits, gs, bits * 10u + gs, true, 16u);
      auto w_pre32 = make_affine_weight(ctx, id_pre32, kN, kK, bits, gs, bits * 10u + gs, true, 32u, true);
      auto w_g = make_affine_weight(ctx, id_g, kN2, kK, bits, gs, bits * 20u + gs, true, 16u);
      auto w_u = make_affine_weight(ctx, id_u, kN2, kK, bits, gs, bits * 30u + gs, true, 16u);

      AffineNaxCache aff_cache;
      aff_cache.register_weight(w_raw.view, zero_ref);
      aff_cache.register_weight(w_pre16.view, zero_ref);
      aff_cache.register_weight(w_g.view, zero_ref);
      aff_cache.register_weight(w_u.view, zero_ref);
      aff_cache.build(ctx, libs, zero_ref);

      Ref high32 = (bits == 3u)                   ? plane(w_pre32.view, "nax_high1")
                   : (bits == 5u || bits == 6u)   ? plane(w_pre32.view, "nax_high2")
                                                  : zero_ref;
      const TiledAffineWeight tw_bns32{
          plane(w_pre32.view, "nax_codes"), high32, plane(w_pre32.view, "nax_sb"),
          kN, kK, 32u, ((kN + 31u) / 32u) * 32u, 2u, bits, gs};

      const auto ref_lin = cpu_linear_ref(x_f32, w_raw.ref_w, kMaxM, kN, kK);
      const auto ref_swi = cpu_swiglu_ref(x_f32, w_g.ref_w, w_u.ref_w, kMaxM, kN2, kK);

      for (const char *tile_mode : {"64", "32"}) {
        ScopedEnv env_tile("OMLX_M5_WIDE_PREFILL_TILE", tile_mode);
        for (const TiledAffineWeight *tw :
             {aff_cache.find(id_raw), aff_cache.find(id_pre16), &tw_bns32}) {
          require(tw != nullptr, "TiledAffineWeight missing");
          Encoder e(ctx, libs, kMaxM,
                    "step2.aff.q" + std::to_string(bits) + ".g" +
                        std::to_string(gs) + ".bns" + std::to_string(tw->bn) +
                        "." + tile_mode);
          e.begin_stage("lin");
          linear_affine_nax(e, *tw, x_ref, xs_ref, y_ref, std::nullopt, 1u);
          auto prof = e.finish(true);
          for (U32 live : {73u, 128u}) {
            patch_and_submit(ctx, prof, live, y_buf, kMaxM, kN, live == 73u);
            const auto out = read_u16(y_buf, U64(live) * kN);
            check_close(out, std::span(ref_lin).first(U64(live) * kN), 0.012, 0.06,
                        "Affine q" + std::to_string(bits) + "_g" + std::to_string(gs) +
                            " bns=" + std::to_string(tw->bn) + " tile=" + tile_mode);
            ++checks;
          }
        }

        // SwiGLU
        const auto *gw_a = aff_cache.find(id_g);
        const auto *uw_a = aff_cache.find(id_u);
        Encoder es(ctx, libs, kMaxM,
                   "step2.aff.swi.q" + std::to_string(bits) + ".g" +
                       std::to_string(gs) + "." + tile_mode);
        es.begin_stage("swi");
        linear_affine_nax_swiglu(es, *gw_a, *uw_a, x_ref, xs_ref, y_ref, 1u);
        auto ps = es.finish(true);
        patch_and_submit(ctx, ps, 73u, y_buf, kMaxM, kN2, true);
        const auto out_s = read_u16(y_buf, 73ULL * kN2);
        check_close(out_s, std::span(ref_swi).first(73ULL * kN2), 0.018, 0.09,
                    "Affine SwiGLU q" + std::to_string(bits) + "_g" + std::to_string(gs));
        ++checks;
      }

      // M < 64 (M = 23) 0-ULP bit-identity preservation
      if (gs == 128u) {
        ScopedEnv clear_tile("OMLX_M5_WIDE_PREFILL_TILE", nullptr);
        Encoder e1(ctx, libs, 23u, "step2.aff.m23.def." + std::to_string(bits));
        e1.begin_stage("lin");
        linear_affine_nax(e1, *aff_cache.find(id_pre16), x_ref, xs_ref, y_ref);
        auto p1 = e1.finish();
        patch_and_submit(ctx, p1, 23u, y_buf, kMaxM, kN, false);
        const auto out1 = read_u16(y_buf, 23ULL * kN);

        ScopedEnv dis("OMLX_M5_WIDE_PREFILL_MIN_T", "0");
        Encoder e2(ctx, libs, 23u, "step2.aff.m23.dis." + std::to_string(bits));
        e2.begin_stage("lin");
        linear_affine_nax(e2, *aff_cache.find(id_pre16), x_ref, xs_ref, y_ref);
        auto p2 = e2.finish();
        patch_and_submit(ctx, p2, 23u, y_buf, kMaxM, kN, false);
        const auto out2 = read_u16(y_buf, 23ULL * kN);
        require(out1 == out2, "Affine M=23 (<64) 0-ULP bit-identity broken");
        check_close(out1, std::span(ref_lin).first(23ULL * kN), 0.012, 0.06,
                    "Affine M=23 (<64) accuracy q" + std::to_string(bits));
        ++checks;
        if (bits == 5u && aff_cache.find(id_pre16)->high1.buffer != nullptr) {
          require(p1.plan.dispatch_count() == 1u,
                  "Q5 h1 BM=16 linear unexpectedly emitted xsum dispatch");
          require(affine_nax_group(e1, *aff_cache.find(id_pre16), std::nullopt,
                                   64u)
                          .find("q5_h1") != std::string::npos,
                  "Q5 narrow linear did not route to q5_h1_bm16");
          Encoder em1(ctx, libs, 1u, "step2.aff.q5.m1");
          em1.begin_stage("lin");
          linear_affine_nax(em1, *aff_cache.find(id_pre16), x_ref, xs_ref, y_ref);
          auto pm1 = em1.finish();
          patch_and_submit(ctx, pm1, 1u, y_buf, kMaxM, kN, false);
          const auto out_m1 = read_u16(y_buf, kN);

          Encoder em8(ctx, libs, 8u, "step2.aff.q5.m8");
          em8.begin_stage("lin");
          linear_affine_nax(em8, *aff_cache.find(id_pre16), x_ref, xs_ref, y_ref);
          auto pm8 = em8.finish();
          patch_and_submit(ctx, pm8, 8u, y_buf, kMaxM, kN, false);
          const auto out_m8 = read_u16(y_buf, 8ULL * kN);
          require(std::equal(out_m1.begin(), out_m1.end(), out_m8.begin()) &&
                      std::equal(out_m8.begin(), out_m8.end(), out1.begin()),
                  "Q5 h1 BM=16 M=1 vs M=8 vs M=23 0-ULP row-independence broken");

          Encoder es8(ctx, libs, 8u, "step2.aff.q5.swi.m8");
          es8.begin_stage("swi");
          linear_affine_nax_swiglu(es8, *aff_cache.find(id_g),
                                   *aff_cache.find(id_u), x_ref, xs_ref, y_ref);
          auto ps8 = es8.finish();
          require(ps8.plan.dispatch_count() == 1u &&
                      affine_nax_group(es8, *aff_cache.find(id_g), std::nullopt,
                                       64u, aff_cache.find(id_u)->group_size)
                              .find("q5_h1") != std::string::npos,
                  "Q5 narrow SwiGLU did not route to 1-dispatch q5_h1_swiglu_bm16");
          patch_and_submit(ctx, ps8, 8u, y_buf, kMaxM, kN2, false);
          const auto out_sw8 = read_u16(y_buf, 8ULL * kN2);
          check_close(out_sw8, std::span(ref_swi).first(8ULL * kN2), 0.018, 0.09,
                      "Q5 h1 BM=16 SwiGLU M=8 accuracy");

          Encoder es1(ctx, libs, 1u, "step2.aff.q5.swi.m1");
          es1.begin_stage("swi");
          linear_affine_nax_swiglu(es1, *aff_cache.find(id_g),
                                   *aff_cache.find(id_u), x_ref, xs_ref, y_ref);
          auto ps1 = es1.finish();
          patch_and_submit(ctx, ps1, 1u, y_buf, kMaxM, kN2, false);
          const auto out_sw1 = read_u16(y_buf, kN2);
          require(std::equal(out_sw1.begin(), out_sw1.end(), out_sw8.begin()),
                  "Q5 h1 BM=16 SwiGLU M=1 vs M=8 0-ULP row-independence broken");
          checks += 4;
        }
      }
    }
  }

  // 4. Mixed QDA g128 + g64 concat & SwiGLU safety check (g128 gate + g64 up must fall back to g64 dequant!)
  {
    ScopedEnv env_tile("OMLX_M5_WIDE_PREFILL_TILE", "64");
    auto wg_128 = make_affine_weight(ctx, 2001u, 64u, kK, 4u, 128u, 77u, false);
    auto wu_64 = make_affine_weight(ctx, 2002u, 64u, kK, 4u, 64u, 88u, false);
    AffineNaxCache mix_cache;
    mix_cache.register_weight(wg_128.view, zero_ref);
    mix_cache.register_weight(wu_64.view, zero_ref);
    const std::array<MaterializedWeightView, 2> concat_parts{wg_128.view, wu_64.view};
    require(mix_cache.register_concat(2003u, concat_parts),
            "mixed g128+g64 register_concat rejected");
    mix_cache.build(ctx, libs, zero_ref);

    const auto *tw_cat = mix_cache.find(2003u);
    require(tw_cat != nullptr && tw_cat->group_size == 64u,
            "mixed g128+g64 concat did not clamp group_size to 64");
    std::vector<float> cat_ref_w;
    cat_ref_w.insert(cat_ref_w.end(), wg_128.ref_w.begin(), wg_128.ref_w.end());
    cat_ref_w.insert(cat_ref_w.end(), wu_64.ref_w.begin(), wu_64.ref_w.end());
    const auto ref_cat = cpu_linear_ref(x_f32, cat_ref_w, 73u, 128u, kK);
    auto y_cat_buf = ctx.make_buffer(U64(kMaxM) * 128u * 2ULL);
    Ref y_cat_ref{&y_cat_buf, 0, y_cat_buf.size()};

    Encoder ec(ctx, libs, kMaxM, "step2.aff.mix.concat");
    ec.begin_stage("cat");
    linear_affine_nax(ec, *tw_cat, x_ref, xs_ref, y_cat_ref, std::nullopt, 1u);
    auto pc = ec.finish(true);
    patch_and_submit(ctx, pc, 73u, y_cat_buf, kMaxM, 128u, true);
    check_close(read_u16(y_cat_buf, 73ULL * 128u), ref_cat, 0.012, 0.06,
                "Mixed g128+g64 concat");

    const auto ref_mix_swi = cpu_swiglu_ref(x_f32, wg_128.ref_w, wu_64.ref_w, 73u, 64u, kK);
    Encoder em(ctx, libs, kMaxM, "step2.aff.mix.swi");
    em.begin_stage("swi");
    linear_affine_nax_swiglu(em, *mix_cache.find(2001u), *mix_cache.find(2002u),
                             x_ref, xs_ref, y_ref, 1u);
    auto pm = em.finish(true);
    patch_and_submit(ctx, pm, 73u, y_buf, kMaxM, 64u, true);
    check_close(read_u16(y_buf, 73ULL * 64u), ref_mix_swi, 0.018, 0.09,
                "Mixed g128+g64 SwiGLU");
    checks += 2;
  }

  // 5. Secondary-slot delegated prefill (e.delegated_prefill) 0-ULP solo parity & OMLX_M5_FORCE_NAX_APPLE10
  {
    {
      ScopedEnv force_on("OMLX_M5_FORCE_NAX_APPLE10", "1");
      ScopedEnv dis_off("OMLX_M5_DISABLE_AFFINE_NAX", nullptr);
      require(affine_nax_runtime_eligible(ctx, libs),
              "OMLX_M5_FORCE_NAX_APPLE10=1 did not enable affine_nax_runtime_eligible");
      ScopedEnv dis_on("OMLX_M5_DISABLE_AFFINE_NAX", "1");
      require(!affine_nax_runtime_eligible(ctx, libs),
              "OMLX_M5_DISABLE_AFFINE_NAX=1 did not override OMLX_M5_FORCE_NAX_APPLE10=1");
    }

    ScopedEnv clear_tile("OMLX_M5_WIDE_PREFILL_TILE", nullptr);
    ScopedEnv clear_min("OMLX_M5_WIDE_PREFILL_MIN_T", nullptr);
    auto w_del = make_affine_weight(ctx, 2101u, 128u, kK, 4u, 64u, 91u, true);
    AffineNaxCache del_cache;
    del_cache.register_weight(w_del.view, zero_ref);
    del_cache.build(ctx, libs, zero_ref);
    const auto *tw_del = del_cache.find(2101u);
    require(tw_del != nullptr, "delegated test weight missing");

    for (const auto &[cap_m, m_rows] :
         std::array<std::pair<U32, U32>, 2>{{{64u, 54u}, {128u, 73u}}}) {
      Encoder e_solo(ctx, libs, cap_m,
                     "step2.del.solo." + std::to_string(m_rows));
      e_solo.begin_stage("lin");
      linear_affine_nax(e_solo, *tw_del, x_ref, xs_ref, y_ref, std::nullopt, 2u);
      auto p_solo = e_solo.finish(true);
      require(p_solo.plan.dispatch_count() == 1u,
              "Wide SK=1 affine_nax unexpectedly emitted xsum dispatch");
      patch_and_submit(ctx, p_solo, m_rows, y_buf, kMaxM, 128u, true);
      const auto out_solo = read_u16(y_buf, U64(m_rows) * 128u);

      Encoder e_del(ctx, libs, cap_m,
                    "step2.del.slot." + std::to_string(m_rows));
      e_del.delegated_prefill = true;
      e_del.begin_stage("lin");
      linear_affine_nax(e_del, *tw_del, x_ref, xs_ref, y_ref, m_rows, 2u);
      require(affine_nax_group(e_del, *tw_del, m_rows, 2u) ==
                  affine_nax_group(e_solo, *tw_del, std::nullopt, 2u),
              "delegated_prefill affine_nax_group mismatch vs solo");
      auto p_del = e_del.finish(true);
      require(p_del.plan.dispatch_count() == 1u,
              "delegated Wide SK=1 affine_nax unexpectedly emitted xsum dispatch");
      patch_and_submit(ctx, p_del, m_rows, y_buf, kMaxM, 128u, true);
      const auto out_del = read_u16(y_buf, U64(m_rows) * 128u);
      require(out_del == out_solo,
              "delegated_prefill 0-ULP mismatch vs solo at M=" +
                  std::to_string(m_rows));

      Encoder e_pack(ctx, libs, cap_m,
                     "step2.del.pack." + std::to_string(m_rows));
      e_pack.delegated_prefill = false;
      e_pack.begin_stage("lin");
      linear_affine_nax(e_pack, *tw_del, x_ref, xs_ref, y_ref, m_rows, 2u);
      require(affine_nax_group(e_pack, *tw_del, m_rows, 2u).find(".sk1.") ==
                  std::string::npos,
              "non-delegated explicit_t unexpectedly selected wide SK=1");
      checks += 2;
    }
  }

  // 6. Tier-1 (T1-A) Multi-Supergroup L2 GROUP_M=8 Swizzling (cap_m=640 -> 10 M-tiles =
  // 8-tile supergroup + 2-tile tail supergroup, live_rows=549 -> 9 active M-tiles,
  // N=141 -> 3 N-tiles with partial tail) + slot == solo 0-ULP across MXFP4, Affine, BF16!
  {
    ScopedEnv clear_tile("OMLX_M5_WIDE_PREFILL_TILE", nullptr);
    ScopedEnv clear_min("OMLX_M5_WIDE_PREFILL_MIN_T", nullptr);
    constexpr U32 kSwizCapM = 640u;
    constexpr U32 kSwizLiveM = 549u;
    constexpr U32 kSwizN = 141u;
    constexpr U32 kSwizN2 = 160u;

    std::vector<std::uint16_t> xs_words;
    const auto xs_f32 = make_activations(kSwizCapM, kK, 71u, xs_words);
    auto xs_x_buf = ctx.upload(std::as_bytes(std::span(xs_words)));
    Ref xs_x_ref{&xs_x_buf, 0, xs_x_buf.size()};
    auto xs_f16_buf = ctx.make_private_buffer(U64(kSwizCapM + 16u) * kK * 2ULL);
    auto xs_rs_buf = ctx.make_private_buffer(U64(kSwizCapM + 16u) * 4ULL);
    auto xs_sum_buf = ctx.make_private_buffer(U64(kSwizCapM + 16u) * (kK / 64u) * 4ULL);
    auto xs_y_buf = ctx.make_buffer(U64(kSwizCapM) * kSwizN2 * 2ULL);
    Ref xs_f16{&xs_f16_buf, 0, xs_f16_buf.size()};
    Ref xs_rs{&xs_rs_buf, 0, xs_rs_buf.size()};
    Ref xs_sum{&xs_sum_buf, 0, xs_sum_buf.size()};
    Ref xs_y{&xs_y_buf, 0, xs_y_buf.size()};

    // 6a. MXFP4 W4A16 Linear + SwiGLU (solo vs delegated_prefill 0-ULP + CPU ref)
    auto mx_sw_lin = make_mxfp4_weight(ctx, 3001u, kSwizN, kK, 61u, true, 16u);
    auto mx_sw_g = make_mxfp4_weight(ctx, 3002u, kSwizN2, kK, 62u, true, 16u);
    auto mx_sw_u = make_mxfp4_weight(ctx, 3003u, kSwizN2, kK, 63u, true, 16u);
    Mxfp4NaxCache mx_sw_cache;
    mx_sw_cache.register_weight(mx_sw_lin.view);
    mx_sw_cache.register_weight(mx_sw_g.view);
    mx_sw_cache.register_weight(mx_sw_u.view);
    mx_sw_cache.build(ctx, libs);
    const auto ref_mx549 = cpu_linear_ref(xs_f32, mx_sw_lin.ref_w, kSwizLiveM, kSwizN, kK);
    const auto ref_mx_swi549 = cpu_swiglu_ref(xs_f32, mx_sw_g.ref_w, mx_sw_u.ref_w, kSwizLiveM, kSwizN2, kK);
    {
      Encoder e_solo(ctx, libs, kSwizCapM, "step2.t1a.mx.solo");
      e_solo.begin_stage("lin");
      linear_mxfp4_nax(e_solo, *mx_sw_cache.find(3001u), xs_x_ref, xs_y, xs_f16, xs_rs, std::nullopt, 2u);
      auto p_solo = e_solo.finish(true);
      patch_and_submit(ctx, p_solo, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN, true);
      const auto out_solo = read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN);
      check_close(out_solo, ref_mx549, 0.01, 0.05, "T1-A MXFP4 swizzle M=549");

      Encoder e_del(ctx, libs, kSwizCapM, "step2.t1a.mx.del");
      e_del.delegated_prefill = true;
      e_del.begin_stage("lin");
      linear_mxfp4_nax(e_del, *mx_sw_cache.find(3001u), xs_x_ref, xs_y, xs_f16, xs_rs, kSwizLiveM, 2u);
      auto p_del = e_del.finish(true);
      patch_and_submit(ctx, p_del, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN, true);
      require(read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN) == out_solo,
              "T1-A MXFP4 M=549 slot vs solo 0-ULP mismatch");

      Encoder e_swi(ctx, libs, kSwizCapM, "step2.t1a.mx.swi");
      e_swi.begin_stage("swi");
      ensure_f16_cast(e_swi, xs_x_ref, xs_f16, xs_rs, kK);
      linear_mxfp4_nax_swiglu_direct_f16(e_swi, *mx_sw_cache.find(3002u),
                                         *mx_sw_cache.find(3003u), xs_f16, xs_rs,
                                         xs_y, false, 2u);
      auto p_swi = e_swi.finish(true);
      patch_and_submit(ctx, p_swi, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN2, true);
      check_close(read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN2), ref_mx_swi549,
                  0.015, 0.08, "T1-A MXFP4 SwiGLU swizzle M=549");
      checks += 3;
    }

    // 6b. Affine q6_g64 + q3_g64 SwiGLU & Linear at M=549 (zero xsum + slot==solo 0-ULP)
    for (U32 bits : {3u, 6u}) {
      auto af_lin = make_affine_weight(ctx, 3100u + bits, kSwizN, kK, bits, 64u, 70u + bits, true, 16u);
      auto af_g = make_affine_weight(ctx, 3110u + bits, kSwizN2, kK, bits, 64u, 80u + bits, true, 16u);
      auto af_u = make_affine_weight(ctx, 3120u + bits, kSwizN2, kK, bits, 64u, 90u + bits, true, 16u);
      AffineNaxCache af_sw_cache;
      af_sw_cache.register_weight(af_lin.view, zero_ref);
      af_sw_cache.register_weight(af_g.view, zero_ref);
      af_sw_cache.register_weight(af_u.view, zero_ref);
      af_sw_cache.build(ctx, libs, zero_ref);
      const auto ref_af549 = cpu_linear_ref(xs_f32, af_lin.ref_w, kSwizLiveM, kSwizN, kK);
      const auto ref_af_swi549 = cpu_swiglu_ref(xs_f32, af_g.ref_w, af_u.ref_w, kSwizLiveM, kSwizN2, kK);

      Encoder e_solo(ctx, libs, kSwizCapM, "step2.t1a.af.solo." + std::to_string(bits));
      e_solo.begin_stage("lin");
      linear_affine_nax(e_solo, *af_sw_cache.find(3100u + bits), xs_x_ref, xs_sum, xs_y, std::nullopt, 2u);
      auto p_solo = e_solo.finish(true);
      require(p_solo.plan.dispatch_count() == 1u,
              "T1-A Affine linear emitted unexpected xsum");
      patch_and_submit(ctx, p_solo, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN, true);
      const auto out_solo = read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN);
      check_close(out_solo, ref_af549, 0.012, 0.06, "T1-A Affine swizzle M=549");

      Encoder e_del(ctx, libs, kSwizCapM, "step2.t1a.af.del." + std::to_string(bits));
      e_del.delegated_prefill = true;
      e_del.begin_stage("lin");
      linear_affine_nax(e_del, *af_sw_cache.find(3100u + bits), xs_x_ref, xs_sum, xs_y, kSwizLiveM, 2u);
      auto p_del = e_del.finish(true);
      patch_and_submit(ctx, p_del, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN, true);
      require(read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN) == out_solo,
              "T1-A Affine M=549 slot vs solo 0-ULP mismatch");

      Encoder e_swi(ctx, libs, kSwizCapM, "step2.t1a.af.swi." + std::to_string(bits));
      e_swi.begin_stage("swi");
      linear_affine_nax_swiglu(e_swi, *af_sw_cache.find(3110u + bits),
                               *af_sw_cache.find(3120u + bits), xs_x_ref, xs_sum, xs_y, 2u);
      auto p_swi = e_swi.finish(true);
      require(p_swi.plan.dispatch_count() == 1u,
              "T1-A Affine SwiGLU emitted unexpected xsum");
      patch_and_submit(ctx, p_swi, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN2, true);
      check_close(read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN2), ref_af_swi549,
                  0.018, 0.09, "T1-A Affine SwiGLU swizzle M=549");
      checks += 3;
    }

    // 6c. BF16 Linear at M=549 (slot == solo 0-ULP)
    auto bf_sw = make_bf16_weight(ctx, 3201u, kSwizN, kK, 99u);
    const auto ref_bf549 = cpu_linear_ref(xs_f32, bf_sw.ref_w, kSwizLiveM, kSwizN, kK);
    {
      Encoder e_solo(ctx, libs, kSwizCapM, "step2.t1a.bf.solo");
      e_solo.begin_stage("lin");
      linear_bf16_nax(e_solo, bf_sw.view, xs_x_ref, xs_y, kSwizN, kK, std::nullopt, 2u);
      auto p_solo = e_solo.finish(true);
      patch_and_submit(ctx, p_solo, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN, true);
      const auto out_solo = read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN);
      check_close(out_solo, ref_bf549, 0.01, 0.05, "T1-A BF16 swizzle M=549");

      Encoder e_del(ctx, libs, kSwizCapM, "step2.t1a.bf.del");
      e_del.delegated_prefill = true;
      e_del.begin_stage("lin");
      linear_bf16_nax(e_del, bf_sw.view, xs_x_ref, xs_y, kSwizN, kK, kSwizLiveM, 2u);
      auto p_del = e_del.finish(true);
      patch_and_submit(ctx, p_del, kSwizLiveM, xs_y_buf, kSwizCapM, kSwizN, true);
      require(read_u16(xs_y_buf, U64(kSwizLiveM) * kSwizN) == out_solo,
              "T1-A BF16 M=549 slot vs solo 0-ULP mismatch");
      checks += 2;
    }
  }

  return {{"live_gpu_checks", checks}, {"passed", true}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: omlx_m5_step2_wide_nax_tests <metallib_dir>");
    }
    ScopedEnv clear_tile("OMLX_M5_WIDE_PREFILL_TILE", nullptr);
    ScopedEnv clear_min_t("OMLX_M5_WIDE_PREFILL_MIN_T", nullptr);
    ScopedEnv clear_max_rf("OMLX_M5_NAX_MAX_RF", nullptr);
    const std::filesystem::path libs = argv[1];
    auto ctx = MetalContext::create();
    const auto r_sel = test_tile_selector();
    const auto r_pso = test_pso_and_tg_sram_budget(ctx, libs);
    const auto r_gpu = test_live_gpu_accuracy_and_parity(ctx, libs);

    std::cout << Json({{"schema", "omlx-m5-step2-wide-nax-v1"},
                       {"device", ctx.capabilities().device_name},
                       {"tile_selector", r_sel},
                       {"pso_and_tg_sram", r_pso},
                       {"live_gpu_accuracy", r_gpu},
                       {"passed", true}})
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
