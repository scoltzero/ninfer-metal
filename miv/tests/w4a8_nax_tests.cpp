#include "affine_nax_cache.h"
#include "gdn_prefill.h"
#include "moe_nax.h"
#include "mxfp4_nax_cache.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/row_exact.h"
#include "w4a8_nax.h"

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
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

namespace {

void require(bool ok, std::string_view msg) {
  if (!ok) {
    throw std::runtime_error("W4A8-NAX FAIL: " + std::string(msg));
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

std::vector<std::uint16_t> read_u16(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 2ULL);
  std::vector<std::uint16_t> out(count);
  std::memcpy(out.data(), raw.data(), count * 2ULL);
  return out;
}

std::vector<std::int8_t> read_i8(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count);
  std::vector<std::int8_t> out(count);
  std::memcpy(out.data(), raw.data(), count);
  return out;
}

std::vector<float> read_f32(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 4ULL);
  std::vector<float> out(count);
  std::memcpy(out.data(), raw.data(), count * 4ULL);
  return out;
}

double cosine_similarity_bf16(std::span<const std::uint16_t> actual_bf16,
                              std::span<const float> ref,
                              std::string_view label) {
  require(actual_bf16.size() >= ref.size(),
          std::string(label) + ": actual buffer smaller than ref");
  double dot = 0.0, na = 0.0, nr = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double a = double(bf16_val(actual_bf16[i]));
    const double r = double(ref[i]);
    require(std::isfinite(a), std::string(label) + ": non-finite GPU output");
    dot += a * r;
    na += a * a;
    nr += r * r;
  }
  if (na == 0.0 && nr == 0.0) {
    return 1.0;
  }
  require(na > 0.0 && nr > 0.0,
          std::string(label) + ": degenerate zero vector");
  return dot / (std::sqrt(na) * std::sqrt(nr));
}

struct OwnedWeight {
  MaterializedWeightView view;
  std::vector<std::unique_ptr<MetalBuffer>> bufs;
  std::vector<float> ref_w;

  void add_plane(const MetalContext &ctx, std::string kind, const void *data,
                 std::size_t byte_len, U64 row_stride = 0) {
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

std::uint8_t expand_1_to_2bit_cpu(std::uint32_t nibble) {
  std::uint32_t out = 0u;
  out |= (nibble & 1u);
  out |= ((nibble >> 1u) & 1u) << 2u;
  out |= ((nibble >> 2u) & 1u) << 4u;
  out |= ((nibble >> 3u) & 1u) << 6u;
  return std::uint8_t(out);
}

std::uint32_t pi16_cpu(std::uint32_t c4) {
  return ((c4 & 3u) << 2u) | ((c4 >> 2u) & 3u);
}

void fwht64_cpu_inplace(float *v) {
  for (U32 len = 1u; len < 64u; len <<= 1u) {
    for (U32 i = 0u; i < 64u; i += len * 2u) {
      for (U32 j = 0u; j < len; ++j) {
        const float a = v[i + j];
        const float b = v[i + j + len];
        v[i + j] = a + b;
        v[i + j + len] = a - b;
      }
    }
  }
  for (U32 i = 0u; i < 64u; ++i) {
    v[i] *= 0.125f;
  }
}

void fwht64_rows_cpu(std::vector<float> &mat, U32 rows, U32 cols) {
  for (U32 r = 0u; r < rows; ++r) {
    for (U32 c = 0u; c < cols; c += 64u) {
      fwht64_cpu_inplace(mat.data() + U64(r) * cols + c);
    }
  }
}

OwnedWeight make_mxfp4_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                              U32 seed, bool pretiled, U32 bn_override = 0u,
                              bool moe_expert = false,
                              bool fwht64_weight = false) {
  OwnedWeight w;
  const U32 bn = bn_override ? bn_override : mxfp4_nax_bn(n);
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  w.view = {.object_id = obj_id,
            .name = "w4a8.mxfp4." + std::to_string(obj_id),
            .format = "mxfp4_e2m1_g32_e8m0_v1",
            .layout = pretiled ? "mx_nax_tiled_bk128_v1"
                               : (moe_expert ? "mx_expert_rowsplit_k128_v1"
                                             : "mx_rowsplit_k128_v1"),
            .parent_shape = {n, k},
            .padded_k = k,
            .row_count = n,
            .in_features = k,
            .input_transform = fwht64_weight ? "fwht64" : "none"};
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
  if (fwht64_weight) {
    fwht64_rows_cpu(w.ref_w, n, k);
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

OwnedWeight make_affine_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                               U32 bits, U32 gs, U32 seed, bool pretiled,
                               U32 bn_override = 0u, bool moe_expert = false,
                               bool lane_v2 = false,
                               bool fwht64_weight = false) {
  OwnedWeight w;
  const U32 bn = bn_override ? bn_override : affine_nax_bn(n);
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  w.view = {.object_id = obj_id,
            .name = "w4a8.affine.q" + std::to_string(bits) + "_g" +
                    std::to_string(gs) + "." + std::to_string(obj_id),
            .format = "oq_affine_q" + std::to_string(bits) + "_g" +
                      std::to_string(gs) + "_bf16_v1",
            .layout = pretiled ? (lane_v2 ? (gs == 128u ? "oq_nax_lane_g128_v2"
                                                        : "oq_nax_lane_g64_v2")
                                          : (gs == 128u ? "oq_nax_tiled_g128_v1"
                                                        : "oq_nax_tiled_g64_v1"))
                               : (moe_expert ? "oq_expert_rowsplit_k128_v1"
                                             : "oq_rowsplit_k128_v1"),
            .parent_shape = {n, k},
            .padded_k = k,
            .row_count = n,
            .in_features = k,
            .input_transform = fwht64_weight ? "fwht64" : "none"};

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
      w.ref_w[U64(r) * k + c] =
          std::fma(bf16_val(raw_scale[gi]), float(q), bf16_val(raw_bias[gi]));
    }
  }
  if (fwht64_weight) {
    fwht64_rows_cpu(w.ref_w, n, k);
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
    const float amp = (r % 5u == 0u) ? 0.2f : (r % 5u == 1u) ? 2.25f : 0.85f;
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

void submit_profile(const MetalContext &ctx, Profile &p, U32 live_rows,
                    bool use_icb = false) {
  for (const auto &patch : p.patches) {
    require(!patch.position, "unexpected position patch in w4a8 test");
    const U32 v = live_rows * patch.multiplier;
    std::memcpy(p.parameter_data.data() + patch.offset + patch.word * 4, &v, 4);
  }
  p.parameters.write("step", p.parameter_data);
  if (use_icb && p.icb.has_value()) {
    (void)ctx.replay_icb(*p.icb);
  } else {
    const std::array calls{MetalCommandInvocation{
        .plan = &p.plan, .bindings = p.bindings, .parameters = &p.parameters}};
    (void)ctx.execute_batch(calls);
  }
}

void fill_sentinel_bf16(MetalBuffer &buf, U64 elems,
                        std::uint16_t sentinel = 0xA5A5u) {
  std::vector<std::uint16_t> s(elems, sentinel);
  buf.write(std::as_bytes(std::span(s)));
}

void check_tail_sentinel(const MetalBuffer &buf, U64 live_elems, U64 total_elems,
                         std::string_view label,
                         std::uint16_t sentinel = 0xA5A5u) {
  const auto all = read_u16(buf, total_elems);
  for (U64 i = live_elems; i < total_elems; ++i) {
    if (all[i] != sentinel) {
      throw std::runtime_error(std::string(label) + " clobbered sentinel at i=" +
                               std::to_string(i));
    }
  }
}

// ============================================================================
// Test 1: PSO Compilation & TG SRAM Budget Audit
// ============================================================================
Json test_pso_and_tg_sram_audit(const MetalContext &ctx,
                                const std::filesystem::path &libs) {
  const auto affine_path = libs / "omlx_m5_affine_nax.metallib";
  const auto mxfp4_path = libs / "omlx_m5_mxfp4_nax.metallib";
  const auto moe_path = libs / "omlx_m5_moe_nax.metallib";
  const auto bf16_path = libs / "omlx_m5_bf16_nax.metallib";
  const auto w4a8_path = libs / "omlx_m5_w4a8_nax.metallib";
  const auto w4a8_moe_path = libs / "omlx_m5_w4a8_moe_nax.metallib";
  require(std::filesystem::is_regular_file(affine_path), "missing affine_nax metallib");
  require(std::filesystem::is_regular_file(mxfp4_path), "missing mxfp4_nax metallib");
  require(std::filesystem::is_regular_file(moe_path), "missing moe_nax metallib");
  require(std::filesystem::is_regular_file(bf16_path), "missing bf16_nax metallib");
  require(std::filesystem::is_regular_file(w4a8_path), "missing w4a8_nax metallib");
  require(std::filesystem::is_regular_file(w4a8_moe_path), "missing w4a8_moe_nax metallib");

  ctx.register_library({.key = "w4a8.audit.affine", .path = affine_path});
  ctx.register_library({.key = "w4a8.audit.mxfp4", .path = mxfp4_path});
  ctx.register_library({.key = "w4a8.audit.moe", .path = moe_path});
  ctx.register_library({.key = "w4a8.audit.bf16", .path = bf16_path});
  ctx.register_library({.key = "w4a8.audit.w4a8", .path = w4a8_path});
  ctx.register_library({.key = "w4a8.audit.w4a8_moe", .path = w4a8_moe_path});

  const bool shader_val =
      std::getenv("MTL_SHADER_VALIDATION") != nullptr &&
      std::string_view(std::getenv("MTL_SHADER_VALIDATION")) == "1";
  const U64 high_occ_cap = shader_val ? 65536ULL : 10240ULL; // <= 10.0 KiB (3-4 TGs/core)
  const U64 max_16k_cap = shader_val ? 65536ULL : 16384ULL;  // <= 16.0 KiB global ceiling

  U32 compiled = 0;
  U64 max_out64_bf16_tg = 0, max_w4a8_lin_tg = 0, max_w4a8_swi_tg = 0,
      max_w4a8_moe_tg = 0, max_16k_cleanup_tg = 0;

  // WP3.1 Zero-Ws Dense Affine out64 SwiGLU (Mode A + Mode B) & slice linear
  for (const char *gs : {"g64", "g128"}) {
    for (const char *tag : {"q4", "q56", "q3", "q8", "q4_q56", "q56_q4"}) {
      for (const char *sfx : {"_swiglu_bm64_bn64_out64_sk1",
                              "_swiglu_bm64_bn64_out64_sk1_fwht64_a8"}) {
        const std::string fn =
            std::string("omlx_m5_affine_nax_") + tag + "_" + gs + sfx;
        const auto info = ctx.warmup_pipeline(
            {.key = "w4a8.pso." + fn,
             .library_key = "w4a8.audit.affine",
             .function_name = fn,
             .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= high_occ_cap,
                fn + " exceeded 10.0 KiB WP3.1 out64 SRAM budget (got " +
                    std::to_string(info.static_threadgroup_memory_bytes) + ")");
        max_out64_bf16_tg = std::max(max_out64_bf16_tg, info.static_threadgroup_memory_bytes);
        ++compiled;
      }
    }
    for (const char *tag : {"q4", "q56", "q3", "q8"}) {
      for (const char *bns : {"bns16", "bns32"}) {
        const std::string fn = std::string("omlx_m5_affine_nax_") + tag + "_" +
                               gs + "_bm64_bn64_" + bns + "_slice_sk1";
        const auto info = ctx.warmup_pipeline(
            {.key = "w4a8.pso." + fn,
             .library_key = "w4a8.audit.affine",
             .function_name = fn,
             .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= max_16k_cap,
                fn + " exceeded 16.0 KiB SRAM");
        max_16k_cleanup_tg = std::max(max_16k_cleanup_tg, info.static_threadgroup_memory_bytes);
        ++compiled;
      }
    }
  }

  // WP3.1 Zero-Ws Dense MXFP4 out64 SwiGLU (Mode A + Mode B) & slice linear
  for (const char *fn : {"omlx_m5_mxfp4_nax_swiglu_bf16_bm64_bn64_out64_sk1",
                         "omlx_m5_mxfp4_nax_swiglu_bf16_bm64_bn64_out64_sk1_fwht64_a8",
                         "omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns16_slice_sk1",
                         "omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns32_slice_sk1"}) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a8.pso.") + fn,
         .library_key = "w4a8.audit.mxfp4",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    const bool is_out64 = std::string_view(fn).find("_out64_") != std::string_view::npos;
    require(info.static_threadgroup_memory_bytes <= (is_out64 ? high_occ_cap : max_16k_cap),
            std::string(fn) + " exceeded WP3.1 SRAM budget");
    if (is_out64) {
      max_out64_bf16_tg = std::max(max_out64_bf16_tg, info.static_threadgroup_memory_bytes);
    } else {
      max_16k_cleanup_tg = std::max(max_16k_cleanup_tg, info.static_threadgroup_memory_bytes);
    }
    ++compiled;
  }

  // WP3.1 Zero-Ws MoE out64 SwiGLU (<= 10.0 KiB)
  {
    const std::string fn = "omlx_m5_moe_mxfp4_nax_swiglu_bm64_out64_sk1";
    const auto info = ctx.warmup_pipeline(
        {.key = "w4a8.pso." + fn,
         .library_key = "w4a8.audit.moe",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= high_occ_cap,
            fn + " exceeded 10.0 KiB MoE out64 SRAM");
    max_out64_bf16_tg = std::max(max_out64_bf16_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }
  for (const char *bits : {"3", "4", "5", "6", "8"}) {
    for (const char *gs : {"64", "128"}) {
      const std::string fn = std::string("omlx_m5_moe_affine_nax_swiglu_q") +
                             bits + "_g" + gs + "_bm64_out64_sk1";
      const auto info = ctx.warmup_pipeline(
          {.key = "w4a8.pso." + fn,
           .library_key = "w4a8.audit.moe",
           .function_name = fn,
           .support_indirect_command_buffers = true});
      require(info.static_threadgroup_memory_bytes <= high_occ_cap,
              fn + " exceeded 10.0 KiB MoE out64 SRAM");
      max_out64_bf16_tg = std::max(max_out64_bf16_tg, info.static_threadgroup_memory_bytes);
      ++compiled;
    }
  }

  // WP3.1 Global <= 16.0 KiB Cleanup Audit (q3 sk8 union, bm32_n16 sk4 cap, q5_h1 sk4, bf16 sk4)
  for (const auto &[lib_key, fn] : std::array<std::pair<const char *, const char *>, 11>{{
           {"w4a8.audit.affine", "omlx_m5_affine_nax_q3_n16_sk8"},
           {"w4a8.audit.affine", "omlx_m5_affine_nax_q3_swiglu_n16_sk8"},
           {"w4a8.audit.affine", "omlx_m5_affine_nax_q4_swiglu_bm32_n16_sk4"},
           {"w4a8.audit.affine", "omlx_m5_affine_nax_q3_swiglu_bm32_n16_sk4"},
           {"w4a8.audit.affine", "omlx_m5_affine_nax_q5_h1_bm16_bns16_sk4"},
           {"w4a8.audit.affine", "omlx_m5_affine_nax_q5_h1_swiglu_bm32_sk4"},
           {"w4a8.audit.affine", "omlx_m5_repack_oq_v1_to_lane_v2"},
           {"w4a8.audit.affine", "omlx_m5_affine_fwht64_xsum_g64"},
           {"w4a8.audit.mxfp4", "omlx_m5_mxfp4_nax_swiglu_bm32_n16_sk4"},
           {"w4a8.audit.mxfp4", "omlx_m5_bf16_fwht64_to_f16_rowscale"},
           {"w4a8.audit.bf16", "omlx_m5_bf16_nax_swiglu_bk128_bm32_n16_sk4"},
       }}) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a8.pso.") + fn,
         .library_key = lib_key,
         .function_name = fn,
         .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= max_16k_cap,
            std::string(fn) + " exceeded 16.0 KiB WP3.1 ceiling (got " +
                std::to_string(info.static_threadgroup_memory_bytes) + ")");
    max_16k_cleanup_tg = std::max(max_16k_cleanup_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }

  // WP1 Part B Zero-Ws Dense W4A8 norm/quant/fwht64/GDN Stage 3 + Linear + SwiGLU Mode A & Mode B (<= 10.0 KiB)
  for (const char *fn : {"omlx_m5_rmsnorm_a8_g64", "omlx_m5_add_rmsnorm_a8_g64",
                         "omlx_m5_quantize_a8_g64",
                         "omlx_m5_quantize_a8_g64_fwht64",
                         "omlx_m5_gdn_chunk_norm_gate_inplace_a8_g64_c64"}) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a8.pso.") + fn,
         .library_key = "w4a8.audit.w4a8",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= (shader_val ? 4096ULL : 256ULL),
            std::string(fn) + " unexpected SRAM usage");
    ++compiled;
  }
  for (const char *gs : {"g64", "g128"}) {
    for (const char *tag : {"q4", "q4_int4b", "q56", "q3", "q8"}) {
      for (const char *bns : {"bns16", "bns32"}) {
        const std::string fn = std::string("omlx_m5_affine_w4a8_nax_") + tag +
                               "_" + gs + "_bm64_bn64_" + bns + "_sk1";
        const auto info = ctx.warmup_pipeline(
            {.key = "w4a8.pso." + fn,
             .library_key = "w4a8.audit.w4a8",
             .function_name = fn,
             .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= high_occ_cap,
                fn + " exceeded 10.0 KiB linear W4A8 SRAM budget");
        max_w4a8_lin_tg = std::max(max_w4a8_lin_tg, info.static_threadgroup_memory_bytes);
        ++compiled;
      }
    }
    for (const char *tag : {"q4", "q4_int4b", "q56", "q3", "q8", "q4_q56", "q56_q4"}) {
      for (const char *sfx : {"_swiglu_bm64_bn64_out64_sk1",
                              "_swiglu_bm64_bn64_out64_sk1_fwht64_a8"}) {
        const std::string fn =
            std::string("omlx_m5_affine_w4a8_nax_") + tag + "_" + gs + sfx;
        const auto info = ctx.warmup_pipeline(
            {.key = "w4a8.pso." + fn,
             .library_key = "w4a8.audit.w4a8",
             .function_name = fn,
             .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= high_occ_cap,
                fn + " exceeded 10.0 KiB SwiGLU W4A8 SRAM budget");
        max_w4a8_swi_tg = std::max(max_w4a8_swi_tg, info.static_threadgroup_memory_bytes);
        ++compiled;
      }
    }
  }
  for (const char *fn : {"omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns16_sk1",
                         "omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns32_sk1",
                         "omlx_m5_mxfp4_w4a8_nax_swiglu_bm64_bn64_out64_sk1",
                         "omlx_m5_mxfp4_w4a8_nax_swiglu_bm64_bn64_out64_sk1_fwht64_a8"}) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a8.pso.") + fn,
         .library_key = "w4a8.audit.w4a8",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    const bool is_swi = std::string_view(fn).find("_swiglu_") != std::string_view::npos;
    require(info.static_threadgroup_memory_bytes <= high_occ_cap,
            std::string(fn) + " exceeded 10.0 KiB MXFP4 W4A8 SRAM budget");
    if (is_swi) {
      max_w4a8_swi_tg = std::max(max_w4a8_swi_tg, info.static_threadgroup_memory_bytes);
    } else {
      max_w4a8_lin_tg = std::max(max_w4a8_lin_tg, info.static_threadgroup_memory_bytes);
    }
    ++compiled;
  }

  // WP1 Part B Zero-Ws MoE W4A8 (SwiGLU Mode A + Mode B + NEW down_scatter_bm64_bn64_sk1, <= 10.0 KiB)
  for (const char *fn : {"omlx_m5_moe_mxfp4_w4a8_nax_swiglu_bm64_out64_sk1",
                         "omlx_m5_moe_mxfp4_w4a8_nax_swiglu_bm64_out64_sk1_fwht64_a8",
                         "omlx_m5_moe_mxfp4_w4a8_nax_down_scatter_bm64_bn64_sk1"}) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a8.pso.") + fn,
         .library_key = "w4a8.audit.w4a8_moe",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= high_occ_cap,
            std::string(fn) + " exceeded 10.0 KiB MoE W4A8 SRAM budget");
    max_w4a8_moe_tg = std::max(max_w4a8_moe_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }
  for (const char *bits : {"3", "4", "5", "6", "8"}) {
    for (const char *gs : {"64", "128"}) {
      for (const std::string &fn : {
               std::string("omlx_m5_moe_affine_w4a8_nax_swiglu_q") + bits +
                   "_g" + gs + "_bm64_out64_sk1",
               std::string("omlx_m5_moe_affine_w4a8_nax_swiglu_q") + bits +
                   "_g" + gs + "_bm64_out64_sk1_fwht64_a8",
               std::string("omlx_m5_moe_affine_w4a8_nax_down_scatter_q") +
                   bits + "_g" + gs + "_bm64_bn64_sk1",
           }) {
        const auto info = ctx.warmup_pipeline(
            {.key = "w4a8.pso." + fn,
             .library_key = "w4a8.audit.w4a8_moe",
             .function_name = fn,
             .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= high_occ_cap,
                fn + " exceeded 10.0 KiB MoE W4A8 SRAM budget");
        max_w4a8_moe_tg = std::max(max_w4a8_moe_tg, info.static_threadgroup_memory_bytes);
        ++compiled;
      }
    }
  }

  return {{"compiled_psos", compiled},
          {"max_out64_bf16_tg_bytes", max_out64_bf16_tg},
          {"max_w4a8_lin_tg_bytes", max_w4a8_lin_tg},
          {"max_w4a8_swi_tg_bytes", max_w4a8_swi_tg},
          {"max_w4a8_moe_tg_bytes", max_w4a8_moe_tg},
          {"max_16k_cleanup_tg_bytes", max_16k_cleanup_tg},
          {"passed", true}};
}

// ============================================================================
// Test 2: Activation Norm / Quantize / GDN Stage 3 Parity
// ============================================================================
Json test_activation_quant_and_gdn_stage3(const MetalContext &ctx,
                                          const std::filesystem::path &libs) {
  U32 checks = 0;
  constexpr U32 kMaxM = 128u;
  constexpr float kEps = 1e-6f;
  constexpr float kWOff = 1.0f;

  // 2a. rmsnorm_a8_g64 + add_rmsnorm_a8_g64 across K = {2048, 5120}, M = {33, 64, 127}
  for (U32 k : {2048u, 5120u}) {
    std::vector<std::uint16_t> x_u16, d_u16, w_u16;
    const auto x_f32 = make_activations(kMaxM, k, 101u + k, x_u16);
    const auto d_f32 = make_activations(kMaxM, k, 202u + k, d_u16);
    w_u16.resize(k);
    std::vector<float> w_f32(k);
    for (U32 c = 0; c < k; ++c) {
      const float v = -0.25f + 0.5f * float((c * 13u + 7u) % 64u) / 63.0f;
      w_u16[c] = float_to_bf16(v);
      w_f32[c] = bf16_val(w_u16[c]);
    }
    const auto layout = w4a8_nax_workspace_layout(kMaxM, k);
    auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
    auto d_buf = ctx.upload(std::as_bytes(std::span(d_u16)));
    auto w_buf = ctx.upload(std::as_bytes(std::span(w_u16)));
    auto opt_bf16_buf = ctx.make_buffer(U64(kMaxM) * k * 2ULL);
    auto qa_buf = ctx.make_buffer(layout.qa_bytes);
    auto sa_buf = ctx.make_buffer(layout.sa_bytes);
    auto ra_buf = ctx.make_buffer(layout.ra_bytes);
    Ref x_ref{&x_buf, 0, x_buf.size()};
    Ref d_ref{&d_buf, 0, d_buf.size()};
    Ref w_ref{&w_buf, 0, w_buf.size()};
    Ref opt_ref{&opt_bf16_buf, 0, opt_bf16_buf.size()};
    Ref qa_ref{&qa_buf, 0, qa_buf.size()};
    Ref sa_ref{&sa_buf, 0, sa_buf.size()};
    Ref ra_ref{&ra_buf, 0, ra_buf.size()};

    // rmsnorm_a8_g64
    {
      Encoder e(ctx, libs, kMaxM, "w4a8.rmsnorm." + std::to_string(k));
      e.begin_stage("rmsnorm_a8");
      emit_rmsnorm_a8_g64(e, x_ref, w_ref, opt_ref, qa_ref, sa_ref, ra_ref, k,
                          kEps, kWOff, true);
      auto prof = e.finish(true);
      for (U32 live : {33u, 64u, 127u}) {
        submit_profile(ctx, prof, live, live == 33u);
        const auto got_bf = read_u16(opt_bf16_buf, U64(live) * k);
        const auto got_qa = read_i8(qa_buf, U64(layout.mp) * k);
        const auto got_sa = read_f32(sa_buf, U64(k / 64u) * layout.mp);
        const auto got_ra = read_f32(ra_buf, U64(k / 64u) * layout.mp);
        for (U32 r = 0; r < live; ++r) {
          double sq = 0.0;
          for (U32 c = 0; c < k; ++c) {
            const double v = double(x_f32[U64(r) * k + c]);
            sq += v * v;
          }
          const float inv = 1.0f / std::sqrt(float(sq / double(k)) + kEps);
          for (U32 g = 0; g < k / 64u; ++g) {
            float amax = 0.0f;
            std::array<float, 64> vals{};
            for (U32 j = 0; j < 64u; ++j) {
              const U32 c = g * 64u + j;
              const std::uint16_t ref_b =
                  float_to_bf16(x_f32[U64(r) * k + c] * inv * (w_f32[c] + kWOff));
              const std::uint16_t gpu_b = got_bf[U64(r) * k + c];
              // Allow at most 1 ULP on rsqrt(sq) across reduction order
              require(std::abs(int(gpu_b) - int(ref_b)) <= 1,
                      "rmsnorm_a8 opt_out_bf16 mismatch > 1 ULP");
              vals[j] = bf16_val(gpu_b);
              amax = std::max(amax, std::abs(vals[j]));
            }
            const float ref_sa = amax / 127.0f;
            const float inv_sa = amax > 0.0f ? (127.0f / amax) : 0.0f;
            int r_sum = 0;
            for (U32 j = 0; j < 64u; ++j) {
              const int q = int(std::clamp(std::rint(vals[j] * inv_sa), -127.0f, 127.0f));
              const int gq = int(got_qa[U64(r) * k + g * 64u + j]);
              require(std::abs(gq - q) <= 1, "rmsnorm_a8 qa mismatch");
              r_sum += gq;
            }
            const float gsa = got_sa[U64(g) * layout.mp + r];
            const float gra = got_ra[U64(g) * layout.mp + r];
            require(std::abs(gsa - ref_sa) <= 1e-6f * std::max(1.0f, ref_sa),
                    "rmsnorm_a8 sa mismatch");
            require(gra == float(r_sum), "rmsnorm_a8 ra exact sum mismatch");
          }
        }
        // Verify padded rows [live, mp) are strictly zero
        for (U32 r = live; r < layout.mp; ++r) {
          for (U32 g = 0; g < k / 64u; ++g) {
            require(got_sa[U64(g) * layout.mp + r] == 0.0f &&
                        got_ra[U64(g) * layout.mp + r] == 0.0f,
                    "rmsnorm_a8 padded sa/ra non-zero");
          }
          for (U32 c = 0; c < k; ++c) {
            require(got_qa[U64(r) * k + c] == 0, "rmsnorm_a8 padded qa non-zero");
          }
        }
        ++checks;
      }
    }

    // add_rmsnorm_a8_g64
    {
      auto h_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
      Ref h_ref{&h_buf, 0, h_buf.size()};
      Encoder e(ctx, libs, kMaxM, "w4a8.add_rmsnorm." + std::to_string(k));
      e.begin_stage("add_rmsnorm_a8");
      emit_add_rmsnorm_a8_g64(e, h_ref, d_ref, w_ref, opt_ref, qa_ref, sa_ref,
                              ra_ref, k, kEps, kWOff, true);
      auto prof = e.finish();
      submit_profile(ctx, prof, 65u, false);
      const auto got_h = read_u16(h_buf, 65ULL * k);
      const auto got_bf = read_u16(opt_bf16_buf, 65ULL * k);
      const auto got_qa = read_i8(qa_buf, 65ULL * k);
      const auto got_ra = read_f32(ra_buf, U64(k / 64u) * layout.mp);
      for (U32 r = 0; r < 65u; ++r) {
        double sq = 0.0;
        for (U32 c = 0; c < k; ++c) {
          const std::uint16_t ref_h =
              float_to_bf16(x_f32[U64(r) * k + c] + d_f32[U64(r) * k + c]);
          require(got_h[U64(r) * k + c] == ref_h,
                  "add_rmsnorm_a8 hidden += delta 0-ULP mismatch");
          const double hv = double(bf16_val(ref_h));
          sq += hv * hv;
        }
        const float inv = 1.0f / std::sqrt(float(sq / double(k)) + kEps);
        for (U32 g = 0; g < k / 64u; ++g) {
          int sum_q = 0;
          for (U32 j = 0; j < 64u; ++j) {
            const U32 c = g * 64u + j;
            const std::uint16_t ref_o = float_to_bf16(
                bf16_val(got_h[U64(r) * k + c]) * inv * (w_f32[c] + kWOff));
            require(std::abs(int(got_bf[U64(r) * k + c]) - int(ref_o)) <= 1,
                    "add_rmsnorm_a8 opt_bf16 mismatch");
            sum_q += int(got_qa[U64(r) * k + c]);
          }
          require(got_ra[U64(g) * layout.mp + r] == float(sum_q),
                  "add_rmsnorm_a8 ra exact sum mismatch");
        }
      }
      ++checks;
    }
  }

  // 2b. Single-pass GDN Stage 3 norm_gate_inplace_a8_g64_c64 vs BF16 Stage 3 + quantize_a8_g64 (0-ULP!)
  if (std::filesystem::is_regular_file(libs / "omlx_m5_gdn_chunk_nax.metallib")) {
    for (U32 hv : {32u, 48u}) {
      const U32 k_gdn = hv * 128u;
      const U32 live_t = 65u;
      const U32 cap_t = 128u;
      const auto layout = w4a8_nax_workspace_layout(cap_t, k_gdn);
      std::vector<std::uint16_t> in_u16, z_u16, nw_u16;
      (void)make_activations(cap_t, k_gdn, 301u + hv, in_u16);
      (void)make_activations(cap_t, k_gdn, 401u + hv, z_u16);
      nw_u16.resize(128u);
      for (U32 d = 0; d < 128u; ++d) {
        nw_u16[d] = float_to_bf16(0.5f + float(d % 32u) / 32.0f);
      }

      auto in_fused = ctx.upload(std::as_bytes(std::span(in_u16)));
      auto in_ref = ctx.upload(std::as_bytes(std::span(in_u16)));
      auto z_buf = ctx.upload(std::as_bytes(std::span(z_u16)));
      auto nw_buf = ctx.upload(std::as_bytes(std::span(nw_u16)));
      auto qa_f = ctx.make_buffer(layout.qa_bytes);
      auto sa_f = ctx.make_buffer(layout.sa_bytes);
      auto ra_f = ctx.make_buffer(layout.ra_bytes);
      auto qa_r = ctx.make_buffer(layout.qa_bytes);
      auto sa_r = ctx.make_buffer(layout.sa_bytes);
      auto ra_r = ctx.make_buffer(layout.ra_bytes);

      // Fused single-pass GDN Stage 3 -> (inout_bf16, qa_f, sa_f, ra_f)
      {
        Encoder e(ctx, libs, cap_t, "w4a8.gdn_s3.fused." + std::to_string(hv));
        e.begin_stage("gdn_s3_a8");
        emit_gdn_norm_gate_inplace_a8_g64(
            e, cap_t, 16u, hv, 2048u + hv * 128u,
            Ref{&in_fused, 0, in_fused.size()}, Ref{&z_buf, 0, z_buf.size()},
            Ref{&nw_buf, 0, nw_buf.size()}, Ref{&qa_f, 0, qa_f.size()},
            Ref{&sa_f, 0, sa_f.size()}, Ref{&ra_f, 0, ra_f.size()}, kEps, true,
            true);
        auto prof = e.finish();
        submit_profile(ctx, prof, live_t, false);
      }
      // Reference 2-pass: BF16 GDN Stage 3 + standalone quantize_a8_g64
      {
        Encoder e(ctx, libs, cap_t, "w4a8.gdn_s3.ref." + std::to_string(hv));
        e.begin_stage("gdn_s3_bf16");
        const U64 total_heads = U64(cap_t) * hv;
        auto &ng = e.emit(
            "gdn_chunk_nax", "omlx_m5_gdn_chunk_norm_gate_inplace_bf16_c64",
            {.width = 128u, .height = (total_heads + 3u) / 4u}, {.width = 128u},
            {Ref{&in_ref, 0, in_ref.size()}, Ref{&z_buf, 0, z_buf.size()},
             Ref{&nw_buf, 0, nw_buf.size()}});
        e.dynamic(ng, 3,
                  words({16u, hv, 128u, 128u, 2048u + hv * 128u, 4u, cap_t,
                         word(kEps), cap_t}),
                  6u);
        emit_quantize_a8_g64(e, Ref{&in_ref, 0, in_ref.size()},
                             Ref{&qa_r, 0, qa_r.size()},
                             Ref{&sa_r, 0, sa_r.size()},
                             Ref{&ra_r, 0, ra_r.size()}, k_gdn);
        auto prof = e.finish();
        submit_profile(ctx, prof, live_t, false);
      }

      require(read_u16(in_fused, U64(live_t) * k_gdn) ==
                  read_u16(in_ref, U64(live_t) * k_gdn),
              "GDN Stage 3 fused BF16 writeback != reference 0-ULP");
      require(read_i8(qa_f, layout.qa_bytes) == read_i8(qa_r, layout.qa_bytes),
              "GDN Stage 3 fused Qa != reference 0-ULP");
      require(read_f32(sa_f, layout.sa_bytes / 4ULL) ==
                  read_f32(sa_r, layout.sa_bytes / 4ULL),
              "GDN Stage 3 fused Sa != reference 0-ULP");
      require(read_f32(ra_f, layout.ra_bytes / 4ULL) ==
                  read_f32(ra_r, layout.ra_bytes / 4ULL),
              "GDN Stage 3 fused Ra != reference 0-ULP");
      ++checks;
    }
  }

  return {{"checks", checks}, {"passed", true}};
}

// ============================================================================
// Test 3: Base 1.2 Lossless out64 SwiGLU 0-ULP Parity & Dual-ANE row_offset
// ============================================================================
Json test_base12_out64_and_slicing_zero_ulp(const MetalContext &ctx,
                                            const std::filesystem::path &libs) {
  constexpr U32 kMaxM = 128u;
  constexpr U32 kK = 512u;
  constexpr U32 kNFull = 512u;
  constexpr U32 kNOdd = 96u; // non-multiple of 64!

  std::vector<std::uint16_t> x_u16;
  (void)make_activations(kMaxM, kK, 501u, x_u16);
  auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
  auto zero_buf = ctx.make_buffer(256);
  zero_buf.write(std::vector<std::byte>(256));
  auto xs_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * (kK / 64u) * 4ULL);
  auto f16_sc_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * kK * 2ULL);
  auto f16_rs_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * 4ULL);
  auto y_out64 = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
  auto y_out32 = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
  auto y_slice = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);

  Ref x_ref{&x_buf, 0, x_buf.size()};
  Ref zero_ref{&zero_buf, 0, zero_buf.size()};
  Ref xs_ref{&xs_buf, 0, xs_buf.size()};
  Ref f16_sc{&f16_sc_buf, 0, f16_sc_buf.size()};
  Ref f16_rs{&f16_rs_buf, 0, f16_rs_buf.size()};
  Ref y64_ref{&y_out64, 0, y_out64.size()};
  Ref y32_ref{&y_out32, 0, y_out32.size()};
  Ref ysl_ref{&y_slice, 0, y_slice.size()};

  U32 checks = 0;
  ScopedEnv force_tile("OMLX_M5_WIDE_PREFILL_TILE", "64");

  // 3a. Dense Affine out64 vs out32 (0-ULP bit-for-bit across q3/q4/q5/q6/q8 x g64/g128, N=96 & 512)
  U32 id = 2000u;
  for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
    for (U32 gs : {64u, 128u}) {
      for (U32 n_cur : {kNOdd, kNFull}) {
        auto wg = make_affine_weight(ctx, id++, n_cur, kK, bits, gs, bits * 11u + gs, true, 16u);
        auto wu = make_affine_weight(ctx, id++, n_cur, kK, bits, gs, bits * 19u + gs, true, 16u);
        AffineNaxCache cache;
        cache.register_weight(wg.view, zero_ref);
        cache.register_weight(wu.view, zero_ref);
        cache.build(ctx, libs, zero_ref);
        const auto *tw_g = cache.find(wg.view.object_id);
        const auto *tw_u = cache.find(wu.view.object_id);
        require(tw_g != nullptr && tw_u != nullptr, "affine out64 cache missing");

        for (U32 live : {33u, 64u, 127u}) {
          {
            ScopedEnv en("OMLX_M5_DISABLE_SWIGLU_OUT64", nullptr);
            ScopedEnv opt("OMLX_M5_ENABLE_SWIGLU_OUT64", "1");
            Encoder e(ctx, libs, kMaxM, "base12.aff.out64");
            e.begin_stage("swi64");
            linear_affine_nax_swiglu(e, *tw_g, *tw_u, x_ref, xs_ref, y64_ref, 1u);
            auto prof = e.finish();
            fill_sentinel_bf16(y_out64, U64(kMaxM) * n_cur);
            submit_profile(ctx, prof, live);
            check_tail_sentinel(y_out64, U64(live) * n_cur, U64(kMaxM) * n_cur, "aff out64");
          }
          {
            ScopedEnv dis("OMLX_M5_DISABLE_SWIGLU_OUT64", "1");
            Encoder e(ctx, libs, kMaxM, "base12.aff.out32");
            e.begin_stage("swi32");
            linear_affine_nax_swiglu(e, *tw_g, *tw_u, x_ref, xs_ref, y32_ref, 1u);
            auto prof = e.finish();
            fill_sentinel_bf16(y_out32, U64(kMaxM) * n_cur);
            submit_profile(ctx, prof, live);
          }
          require(read_u16(y_out64, U64(live) * n_cur) ==
                      read_u16(y_out32, U64(live) * n_cur),
                  "Base 1.2 Affine out64 != out32 0-ULP for q" +
                      std::to_string(bits) + "_g" + std::to_string(gs) +
                      " N=" + std::to_string(n_cur) + " M=" + std::to_string(live));
          ++checks;
        }

        // Dual-ANE row_offset = 128, 256 slicing check on N=512 (linear + SwiGLU)
        if (n_cur == kNFull && (bits == 4u || bits == 5u || bits == 8u) && gs == 64u) {
          // Full-width reference linear & SwiGLU at live=73
          constexpr U32 kLive = 73u;
          {
            Encoder e(ctx, libs, kMaxM, "base12.aff.full.lin");
            e.begin_stage("lin");
            linear_affine_nax(e, *tw_g, x_ref, xs_ref, y32_ref, std::nullopt, 1u);
            auto prof = e.finish();
            submit_profile(ctx, prof, kLive);
          }
          const auto ref_lin = read_u16(y_out32, U64(kLive) * kNFull);
          {
            Encoder e(ctx, libs, kMaxM, "base12.aff.full.swi");
            e.begin_stage("swi");
            linear_affine_nax_swiglu(e, *tw_g, *tw_u, x_ref, xs_ref, y64_ref, 1u);
            auto prof = e.finish();
            submit_profile(ctx, prof, kLive);
          }
          const auto ref_swi = read_u16(y_out64, U64(kLive) * kNFull);

          for (U32 roff : {128u, 256u}) {
            // Linear slice [roff, 512) into out_stride = 512
            {
              Encoder e(ctx, libs, kMaxM, "base12.aff.slice.lin");
              e.begin_stage("lin_s");
              linear_affine_nax(e, *tw_g, x_ref, xs_ref, ysl_ref, std::nullopt,
                                1u, roff, kNFull);
              auto prof = e.finish();
              fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
              submit_profile(ctx, prof, kLive);
              const auto got = read_u16(y_slice, U64(kLive) * kNFull);
              for (U32 r = 0; r < kLive; ++r) {
                for (U32 c = 0; c < roff; ++c) {
                  require(got[U64(r) * kNFull + c] == 0xA5A5u,
                          "Base 1.2 Affine linear slice clobbered [0, row_offset)");
                }
                for (U32 c = roff; c < kNFull; ++c) {
                  require(got[U64(r) * kNFull + c] == ref_lin[U64(r) * kNFull + c],
                          "Base 1.2 Affine linear slice != full 0-ULP");
                }
              }
              check_tail_sentinel(y_slice, U64(kLive) * kNFull, U64(kMaxM) * kNFull, "aff slice lin");
              ++checks;
            }
            // SwiGLU slice [roff, 512) into out_stride = 512
            {
              Encoder e(ctx, libs, kMaxM, "base12.aff.slice.swi");
              e.begin_stage("swi_s");
              linear_affine_nax_swiglu(e, *tw_g, *tw_u, x_ref, xs_ref, ysl_ref, 1u,
                                       std::nullopt, roff, kNFull);
              auto prof = e.finish();
              fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
              submit_profile(ctx, prof, kLive);
              const auto got = read_u16(y_slice, U64(kLive) * kNFull);
              for (U32 r = 0; r < kLive; ++r) {
                for (U32 c = 0; c < roff; ++c) {
                  require(got[U64(r) * kNFull + c] == 0xA5A5u,
                          "Base 1.2 Affine SwiGLU slice clobbered [0, row_offset)");
                }
                for (U32 c = roff; c < kNFull; ++c) {
                  require(got[U64(r) * kNFull + c] == ref_swi[U64(r) * kNFull + c],
                          "Base 1.2 Affine SwiGLU slice != full 0-ULP");
                }
              }
              check_tail_sentinel(y_slice, U64(kLive) * kNFull, U64(kMaxM) * kNFull, "aff slice swi");
              ++checks;
            }
          }
        }
      }
    }
  }

  // 3b. Dense Affine mixed q4_q56 & q56_q4 out64 SwiGLU 0-ULP parity vs separate linears
  {
    auto w_q4 = make_affine_weight(ctx, id++, kNFull, kK, 4u, 64u, 77u, true, 16u);
    auto w_q5 = make_affine_weight(ctx, id++, kNFull, kK, 5u, 64u, 88u, true, 16u);
    AffineNaxCache cache;
    cache.register_weight(w_q4.view, zero_ref);
    cache.register_weight(w_q5.view, zero_ref);
    cache.build(ctx, libs, zero_ref);
    const auto *tw4 = cache.find(w_q4.view.object_id);
    const auto *tw5 = cache.find(w_q5.view.object_id);
    constexpr U32 kLive = 73u;
    {
      Encoder e(ctx, libs, kMaxM, "base12.mix.lin4");
      e.begin_stage("l4");
      linear_affine_nax(e, *tw4, x_ref, xs_ref, y32_ref, std::nullopt, 1u);
      auto p = e.finish();
      submit_profile(ctx, p, kLive);
    }
    const auto g4_u16 = read_u16(y_out32, U64(kLive) * kNFull);
    {
      Encoder e(ctx, libs, kMaxM, "base12.mix.lin5");
      e.begin_stage("l5");
      linear_affine_nax(e, *tw5, x_ref, xs_ref, y32_ref, std::nullopt, 1u);
      auto p = e.finish();
      submit_profile(ctx, p, kLive);
    }
    const auto u5_u16 = read_u16(y_out32, U64(kLive) * kNFull);

    // q4 gate, q56 up
    {
      Encoder e(ctx, libs, kMaxM, "base12.mix.q4_q56");
      e.begin_stage("mix");
      linear_affine_nax_swiglu(e, *tw4, *tw5, x_ref, xs_ref, y64_ref, 1u);
      auto p = e.finish();
      submit_profile(ctx, p, kLive);
      const auto got = read_u16(y_out64, U64(kLive) * kNFull);
      for (std::size_t i = 0; i < got.size(); ++i) {
        const std::uint16_t ref_b =
            float_to_bf16(silu_ref(bf16_val(g4_u16[i])) * bf16_val(u5_u16[i]));
        require(std::abs(int(got[i]) - int(ref_b)) <= 1,
                "Base 1.2 mixed q4_q56 SwiGLU mismatch");
      }
      ++checks;
    }
    // q56 gate, q4 up
    {
      Encoder e(ctx, libs, kMaxM, "base12.mix.q56_q4");
      e.begin_stage("mix");
      linear_affine_nax_swiglu(e, *tw5, *tw4, x_ref, xs_ref, y64_ref, 1u);
      auto p = e.finish();
      submit_profile(ctx, p, kLive);
      const auto got = read_u16(y_out64, U64(kLive) * kNFull);
      for (std::size_t i = 0; i < got.size(); ++i) {
        const std::uint16_t ref_b =
            float_to_bf16(silu_ref(bf16_val(u5_u16[i])) * bf16_val(g4_u16[i]));
        require(std::abs(int(got[i]) - int(ref_b)) <= 1,
                "Base 1.2 mixed q56_q4 SwiGLU mismatch");
      }
      ++checks;
    }
  }

  // 3c. Dense MXFP4 out64 vs out32 (0-ULP!) + Dual-ANE row_offset = 128, 256 slicing
  for (U32 n_cur : {kNOdd, kNFull}) {
    auto mg = make_mxfp4_weight(ctx, id++, n_cur, kK, 61u, true, 16u);
    auto mu = make_mxfp4_weight(ctx, id++, n_cur, kK, 62u, true, 16u);
    Mxfp4NaxCache cache;
    cache.register_weight(mg.view);
    cache.register_weight(mu.view);
    cache.build(ctx, libs);
    const auto *tw_g = cache.find(mg.view.object_id);
    const auto *tw_u = cache.find(mu.view.object_id);
    for (U32 live : {33u, 64u, 127u}) {
      {
        ScopedEnv en("OMLX_M5_DISABLE_SWIGLU_OUT64", nullptr);
        ScopedEnv opt("OMLX_M5_ENABLE_SWIGLU_OUT64", "1");
        Encoder e(ctx, libs, kMaxM, "base12.mx.out64");
        e.begin_stage("swi64");
        linear_mxfp4_nax_swiglu(e, *tw_g, *tw_u, x_ref, f16_sc, f16_rs, y64_ref, 1u);
        auto prof = e.finish();
        submit_profile(ctx, prof, live);
      }
      {
        ScopedEnv dis("OMLX_M5_DISABLE_SWIGLU_OUT64", "1");
        Encoder e(ctx, libs, kMaxM, "base12.mx.out32");
        e.begin_stage("swi32");
        linear_mxfp4_nax_swiglu(e, *tw_g, *tw_u, x_ref, f16_sc, f16_rs, y32_ref, 1u);
        auto prof = e.finish();
        submit_profile(ctx, prof, live);
      }
      require(read_u16(y_out64, U64(live) * n_cur) ==
                  read_u16(y_out32, U64(live) * n_cur),
              "Base 1.2 MXFP4 out64 != out32 0-ULP");
      ++checks;
    }

    if (n_cur == kNFull) {
      constexpr U32 kLive = 73u;
      {
        Encoder e(ctx, libs, kMaxM, "base12.mx.full.lin");
        e.begin_stage("lin");
        linear_mxfp4_nax(e, *tw_g, x_ref, y32_ref, f16_sc, f16_rs, std::nullopt, 1u);
        auto p = e.finish();
        submit_profile(ctx, p, kLive);
      }
      const auto ref_lin = read_u16(y_out32, U64(kLive) * kNFull);
      {
        Encoder e(ctx, libs, kMaxM, "base12.mx.full.swi");
        e.begin_stage("swi");
        linear_mxfp4_nax_swiglu(e, *tw_g, *tw_u, x_ref, f16_sc, f16_rs, y64_ref, 1u);
        auto p = e.finish();
        submit_profile(ctx, p, kLive);
      }
      const auto ref_swi = read_u16(y_out64, U64(kLive) * kNFull);

      for (U32 roff : {128u, 256u}) {
        {
          Encoder e(ctx, libs, kMaxM, "base12.mx.slice.lin");
          e.begin_stage("lin_s");
          linear_mxfp4_nax(e, *tw_g, x_ref, ysl_ref, f16_sc, f16_rs,
                           std::nullopt, 1u, roff, kNFull);
          auto p = e.finish();
          fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
          submit_profile(ctx, p, kLive);
          const auto got = read_u16(y_slice, U64(kLive) * kNFull);
          for (U32 r = 0; r < kLive; ++r) {
            for (U32 c = 0; c < roff; ++c) {
              require(got[U64(r) * kNFull + c] == 0xA5A5u,
                      "Base 1.2 MXFP4 linear slice clobbered [0, row_offset)");
            }
            for (U32 c = roff; c < kNFull; ++c) {
              require(got[U64(r) * kNFull + c] == ref_lin[U64(r) * kNFull + c],
                      "Base 1.2 MXFP4 linear slice != full 0-ULP");
            }
          }
          ++checks;
        }
        {
          Encoder e(ctx, libs, kMaxM, "base12.mx.slice.swi");
          e.begin_stage("swi_s");
          linear_mxfp4_nax_swiglu(e, *tw_g, *tw_u, x_ref, f16_sc, f16_rs,
                                  ysl_ref, 1u, std::nullopt, roff, kNFull);
          auto p = e.finish();
          fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
          submit_profile(ctx, p, kLive);
          const auto got = read_u16(y_slice, U64(kLive) * kNFull);
          for (U32 r = 0; r < kLive; ++r) {
            for (U32 c = 0; c < roff; ++c) {
              require(got[U64(r) * kNFull + c] == 0xA5A5u,
                      "Base 1.2 MXFP4 SwiGLU slice clobbered [0, row_offset)");
            }
            for (U32 c = roff; c < kNFull; ++c) {
              require(got[U64(r) * kNFull + c] == ref_swi[U64(r) * kNFull + c],
                      "Base 1.2 MXFP4 SwiGLU slice != full 0-ULP");
            }
          }
          ++checks;
        }
      }
    }
  }

  return {{"checks", checks}, {"passed", true}};
}

// ============================================================================
// Test 4: Route A Dense W4A8 Accuracy (cos_sim >= 0.9995), M=33/64/127/2048, row_offset=128/256
// ============================================================================
Json test_dense_w4a8_accuracy_and_slicing(const MetalContext &ctx,
                                          const std::filesystem::path &libs) {
  constexpr U32 kK = 512u;
  constexpr U32 kNFull = 512u;
  constexpr U32 kNCheck = 128u;
  constexpr double kMinCos = 0.9995;

  double min_cos_seen = 1.0;
  U32 checks = 0;

  auto zero_buf = ctx.make_buffer(256);
  zero_buf.write(std::vector<std::byte>(256));
  Ref zero_ref{&zero_buf, 0, zero_buf.size()};

  // 4a. All formats across non-64 M = {33, 64, 127} (bns16 & bns32, linear + SwiGLU)
  {
    constexpr U32 kMaxM = 128u;
    std::vector<std::uint16_t> x_u16;
    const auto x_f32 = make_activations(kMaxM, kK, 701u, x_u16);
    auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
    const auto layout = w4a8_nax_workspace_layout(kMaxM, kK);
    auto qa_buf = ctx.make_buffer(layout.qa_bytes);
    auto sa_buf = ctx.make_buffer(layout.sa_bytes);
    auto ra_buf = ctx.make_buffer(layout.ra_bytes);
    auto y_buf = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
    auto y_slice = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
    Ref x_ref{&x_buf, 0, x_buf.size()};
    Ref qa_ref{&qa_buf, 0, qa_buf.size()};
    Ref sa_ref{&sa_buf, 0, sa_buf.size()};
    Ref ra_ref{&ra_buf, 0, ra_buf.size()};
    Ref y_ref{&y_buf, 0, y_buf.size()};
    Ref ysl_ref{&y_slice, 0, y_slice.size()};

    // Quantize activations helper
    auto quantize_for_m = [&](U32 live_m) {
      Encoder eq(ctx, libs, kMaxM, "w4a8.q");
      eq.begin_stage("quant");
      emit_quantize_a8_g64(eq, x_ref, qa_ref, sa_ref, ra_ref, kK);
      auto pq = eq.finish();
      submit_profile(ctx, pq, live_m);
    };

    // Affine: all 10 (q3, q4, q5, q6, q8) x (g64, g128), bns16 & bns32 linear + SwiGLU
    U32 wid = 3000u;
    for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
      for (U32 gs : {64u, 128u}) {
        auto wg = make_affine_weight(ctx, wid++, kNCheck, kK, bits, gs, bits * 13u + gs, true, 16u);
        auto wu = make_affine_weight(ctx, wid++, kNCheck, kK, bits, gs, bits * 29u + gs, true, 16u);
        auto w32 = make_affine_weight(ctx, wid++, kNCheck, kK, bits, gs, bits * 13u + gs, true, 32u, false, true);
        AffineNaxCache cache;
        cache.register_weight(wg.view, zero_ref);
        cache.register_weight(wu.view, zero_ref);
        cache.build(ctx, libs, zero_ref);
        const auto *tw_g = cache.find(wg.view.object_id);
        const auto *tw_u = cache.find(wu.view.object_id);
        Ref h32 = (bits == 3u)                 ? plane(w32.view, "nax_high1")
                  : (bits == 5u || bits == 6u) ? plane(w32.view, "nax_high2")
                                               : zero_ref;
        const TiledAffineWeight tw32{plane(w32.view, "nax_codes"), h32,
                                     plane(w32.view, "nax_sb"), kNCheck, kK,
                                     32u, ((kNCheck + 31u) / 32u) * 32u, 1u,
                                     bits, gs};

        const auto ref_lin = cpu_linear_ref(x_f32, wg.ref_w, kMaxM, kNCheck, kK);
        const auto ref_swi = cpu_swiglu_ref(x_f32, wg.ref_w, wu.ref_w, kMaxM, kNCheck, kK);

        // Linear bns16 & bns32
        for (const TiledAffineWeight *tw : {tw_g, &tw32}) {
          Encoder e(ctx, libs, kMaxM, "w4a8.aff.lin");
          e.begin_stage("lin");
          linear_affine_w4a8_nax(e, *tw, qa_ref, sa_ref, ra_ref, y_ref);
          auto prof = e.finish(true);
          for (U32 live : {33u, 64u, 127u}) {
            quantize_for_m(live);
            fill_sentinel_bf16(y_buf, U64(kMaxM) * kNCheck);
            submit_profile(ctx, prof, live, live == 33u);
            check_tail_sentinel(y_buf, U64(live) * kNCheck, U64(kMaxM) * kNCheck, "w4a8 aff lin");
            const auto got = read_u16(y_buf, U64(live) * kNCheck);
            const double cs = cosine_similarity_bf16(
                got, std::span(ref_lin).first(U64(live) * kNCheck),
                "w4a8 aff lin q" + std::to_string(bits) + "_g" +
                    std::to_string(gs) + "_bns" + std::to_string(tw->bn));
            min_cos_seen = std::min(min_cos_seen, cs);
            require(cs >= kMinCos,
                    "W4A8 Affine linear cos_sim=" + std::to_string(cs) +
                        " < 0.9995 (q" + std::to_string(bits) + "_g" +
                        std::to_string(gs) + "_bns" + std::to_string(tw->bn) +
                        ")");
            ++checks;
          }
        }

        // SwiGLU out64
        {
          Encoder e(ctx, libs, kMaxM, "w4a8.aff.swi");
          e.begin_stage("swi");
          linear_affine_w4a8_nax_swiglu(e, *tw_g, *tw_u, qa_ref, sa_ref, ra_ref, y_ref);
          auto prof = e.finish(true);
          for (U32 live : {33u, 64u, 127u}) {
            quantize_for_m(live);
            fill_sentinel_bf16(y_buf, U64(kMaxM) * kNCheck);
            submit_profile(ctx, prof, live, live == 127u);
            check_tail_sentinel(y_buf, U64(live) * kNCheck, U64(kMaxM) * kNCheck, "w4a8 aff swi");
            const auto got = read_u16(y_buf, U64(live) * kNCheck);
            const double cs = cosine_similarity_bf16(
                got, std::span(ref_swi).first(U64(live) * kNCheck),
                "w4a8 aff swi q" + std::to_string(bits) + "_g" + std::to_string(gs));
            min_cos_seen = std::min(min_cos_seen, cs);
            require(cs >= kMinCos,
                    "W4A8 Affine SwiGLU cos_sim=" + std::to_string(cs) +
                        " < 0.9995 (q" + std::to_string(bits) + "_g" +
                        std::to_string(gs) + ")");
            ++checks;
          }
        }
      }
    }

    // Mixed q4_q56 & q56_q4 W4A8 SwiGLU
    {
      auto w4 = make_affine_weight(ctx, wid++, kNCheck, kK, 4u, 64u, 55u, true, 16u);
      auto w5 = make_affine_weight(ctx, wid++, kNCheck, kK, 5u, 64u, 66u, true, 16u);
      AffineNaxCache cache;
      cache.register_weight(w4.view, zero_ref);
      cache.register_weight(w5.view, zero_ref);
      cache.build(ctx, libs, zero_ref);
      const auto *tw4 = cache.find(w4.view.object_id);
      const auto *tw5 = cache.find(w5.view.object_id);
      const auto ref45 = cpu_swiglu_ref(x_f32, w4.ref_w, w5.ref_w, 65u, kNCheck, kK);
      const auto ref54 = cpu_swiglu_ref(x_f32, w5.ref_w, w4.ref_w, 65u, kNCheck, kK);
      quantize_for_m(65u);
      {
        Encoder e(ctx, libs, kMaxM, "w4a8.mix.45");
        e.begin_stage("swi");
        linear_affine_w4a8_nax_swiglu(e, *tw4, *tw5, qa_ref, sa_ref, ra_ref, y_ref);
        auto p = e.finish();
        submit_profile(ctx, p, 65u);
        const double cs = cosine_similarity_bf16(read_u16(y_buf, 65ULL * kNCheck), ref45, "w4a8 q4_q56");
        min_cos_seen = std::min(min_cos_seen, cs);
        require(cs >= kMinCos, "W4A8 mixed q4_q56 cos_sim < 0.9995");
        ++checks;
      }
      {
        Encoder e(ctx, libs, kMaxM, "w4a8.mix.54");
        e.begin_stage("swi");
        linear_affine_w4a8_nax_swiglu(e, *tw5, *tw4, qa_ref, sa_ref, ra_ref, y_ref);
        auto p = e.finish();
        submit_profile(ctx, p, 65u);
        const double cs = cosine_similarity_bf16(read_u16(y_buf, 65ULL * kNCheck), ref54, "w4a8 q56_q4");
        min_cos_seen = std::min(min_cos_seen, cs);
        require(cs >= kMinCos, "W4A8 mixed q56_q4 cos_sim < 0.9995");
        ++checks;
      }
    }

    // MXFP4 W4A8 bns16 & bns32 linear + out64 SwiGLU
    {
      auto mg = make_mxfp4_weight(ctx, wid++, kNCheck, kK, 91u, true, 16u);
      auto mu = make_mxfp4_weight(ctx, wid++, kNCheck, kK, 92u, true, 16u);
      auto m32 = make_mxfp4_weight(ctx, wid++, kNCheck, kK, 91u, true, 32u);
      Mxfp4NaxCache cache;
      cache.register_weight(mg.view);
      cache.register_weight(mu.view);
      cache.build(ctx, libs);
      const auto *tw_g = cache.find(mg.view.object_id);
      const auto *tw_u = cache.find(mu.view.object_id);
      const TiledMxfp4Weight tw32{plane(m32.view, "nax_data"),
                                  plane(m32.view, "nax_scale"), kNCheck, kK,
                                  32u, ((kNCheck + 31u) / 32u) * 32u, 1u};
      const auto ref_lin = cpu_linear_ref(x_f32, mg.ref_w, kMaxM, kNCheck, kK);
      const auto ref_swi = cpu_swiglu_ref(x_f32, mg.ref_w, mu.ref_w, kMaxM, kNCheck, kK);

      for (const TiledMxfp4Weight *tw : {tw_g, &tw32}) {
        Encoder e(ctx, libs, kMaxM, "w4a8.mx.lin");
        e.begin_stage("lin");
        linear_mxfp4_w4a8_nax(e, *tw, qa_ref, sa_ref, y_ref);
        auto prof = e.finish(true);
        for (U32 live : {33u, 64u, 127u}) {
          quantize_for_m(live);
          fill_sentinel_bf16(y_buf, U64(kMaxM) * kNCheck);
          submit_profile(ctx, prof, live, live == 64u);
          check_tail_sentinel(y_buf, U64(live) * kNCheck, U64(kMaxM) * kNCheck, "w4a8 mx lin");
          const double cs = cosine_similarity_bf16(
              read_u16(y_buf, U64(live) * kNCheck),
              std::span(ref_lin).first(U64(live) * kNCheck), "w4a8 mx lin");
          min_cos_seen = std::min(min_cos_seen, cs);
          require(cs >= kMinCos, "W4A8 MXFP4 linear cos_sim < 0.9995");
          ++checks;
        }
      }
      {
        Encoder e(ctx, libs, kMaxM, "w4a8.mx.swi");
        e.begin_stage("swi");
        linear_mxfp4_w4a8_nax_swiglu(e, *tw_g, *tw_u, qa_ref, sa_ref, y_ref);
        auto prof = e.finish(true);
        for (U32 live : {33u, 64u, 127u}) {
          quantize_for_m(live);
          fill_sentinel_bf16(y_buf, U64(kMaxM) * kNCheck);
          submit_profile(ctx, prof, live, live == 33u);
          check_tail_sentinel(y_buf, U64(live) * kNCheck, U64(kMaxM) * kNCheck, "w4a8 mx swi");
          const double cs = cosine_similarity_bf16(
              read_u16(y_buf, U64(live) * kNCheck),
              std::span(ref_swi).first(U64(live) * kNCheck), "w4a8 mx swi");
          min_cos_seen = std::min(min_cos_seen, cs);
          require(cs >= kMinCos, "W4A8 MXFP4 SwiGLU cos_sim < 0.9995");
          ++checks;
        }
      }
    }

    // 4b. Dual-ANE row_offset = 128, 256 slicing on Dense W4A8 Affine (q4, q5, q8) & MXFP4
    quantize_for_m(73u);
    for (U32 bits : {4u, 5u, 8u}) {
      const U32 gs = (bits == 8u) ? 128u : 64u;
      auto wg = make_affine_weight(ctx, wid++, kNFull, kK, bits, gs, bits * 37u, true, 16u);
      auto wu = make_affine_weight(ctx, wid++, kNFull, kK, bits, gs, bits * 41u, true, 16u);
      AffineNaxCache cache;
      cache.register_weight(wg.view, zero_ref);
      cache.register_weight(wu.view, zero_ref);
      cache.build(ctx, libs, zero_ref);
      const auto *tw_g = cache.find(wg.view.object_id);
      const auto *tw_u = cache.find(wu.view.object_id);

      {
        Encoder e(ctx, libs, kMaxM, "w4a8.aff.full.lin");
        e.begin_stage("lin");
        linear_affine_w4a8_nax(e, *tw_g, qa_ref, sa_ref, ra_ref, y_ref);
        auto p = e.finish();
        submit_profile(ctx, p, 73u);
      }
      const auto full_lin = read_u16(y_buf, 73ULL * kNFull);
      {
        Encoder e(ctx, libs, kMaxM, "w4a8.aff.full.swi");
        e.begin_stage("swi");
        linear_affine_w4a8_nax_swiglu(e, *tw_g, *tw_u, qa_ref, sa_ref, ra_ref, y_ref);
        auto p = e.finish();
        submit_profile(ctx, p, 73u);
      }
      const auto full_swi = read_u16(y_buf, 73ULL * kNFull);

      for (U32 roff : {128u, 256u}) {
        {
          Encoder e(ctx, libs, kMaxM, "w4a8.aff.sl.lin");
          e.begin_stage("lin");
          linear_affine_w4a8_nax(e, *tw_g, qa_ref, sa_ref, ra_ref, ysl_ref,
                                 std::nullopt, roff, kNFull);
          auto p = e.finish();
          fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
          submit_profile(ctx, p, 73u);
          const auto got = read_u16(y_slice, 73ULL * kNFull);
          for (U32 r = 0; r < 73u; ++r) {
            for (U32 c = 0; c < roff; ++c) {
              require(got[U64(r) * kNFull + c] == 0xA5A5u,
                      "W4A8 Affine linear slice clobbered [0, row_offset)");
            }
            for (U32 c = roff; c < kNFull; ++c) {
              require(got[U64(r) * kNFull + c] == full_lin[U64(r) * kNFull + c],
                      "W4A8 Affine linear slice != full 0-ULP");
            }
          }
          ++checks;
        }
        {
          Encoder e(ctx, libs, kMaxM, "w4a8.aff.sl.swi");
          e.begin_stage("swi");
          linear_affine_w4a8_nax_swiglu(e, *tw_g, *tw_u, qa_ref, sa_ref, ra_ref,
                                        ysl_ref, std::nullopt, roff, kNFull);
          auto p = e.finish();
          fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
          submit_profile(ctx, p, 73u);
          const auto got = read_u16(y_slice, 73ULL * kNFull);
          for (U32 r = 0; r < 73u; ++r) {
            for (U32 c = 0; c < roff; ++c) {
              require(got[U64(r) * kNFull + c] == 0xA5A5u,
                      "W4A8 Affine SwiGLU slice clobbered [0, row_offset)");
            }
            for (U32 c = roff; c < kNFull; ++c) {
              require(got[U64(r) * kNFull + c] == full_swi[U64(r) * kNFull + c],
                      "W4A8 Affine SwiGLU slice != full 0-ULP");
            }
          }
          ++checks;
        }
      }
    }

    // MXFP4 W4A8 Dual-ANE row_offset = 128, 256
    {
      auto mg = make_mxfp4_weight(ctx, wid++, kNFull, kK, 97u, true, 16u);
      auto mu = make_mxfp4_weight(ctx, wid++, kNFull, kK, 98u, true, 16u);
      Mxfp4NaxCache cache;
      cache.register_weight(mg.view);
      cache.register_weight(mu.view);
      cache.build(ctx, libs);
      const auto *tw_g = cache.find(mg.view.object_id);
      const auto *tw_u = cache.find(mu.view.object_id);
      {
        Encoder e(ctx, libs, kMaxM, "w4a8.mx.full.lin");
        e.begin_stage("lin");
        linear_mxfp4_w4a8_nax(e, *tw_g, qa_ref, sa_ref, y_ref);
        auto p = e.finish();
        submit_profile(ctx, p, 73u);
      }
      const auto full_lin = read_u16(y_buf, 73ULL * kNFull);
      {
        Encoder e(ctx, libs, kMaxM, "w4a8.mx.full.swi");
        e.begin_stage("swi");
        linear_mxfp4_w4a8_nax_swiglu(e, *tw_g, *tw_u, qa_ref, sa_ref, y_ref);
        auto p = e.finish();
        submit_profile(ctx, p, 73u);
      }
      const auto full_swi = read_u16(y_buf, 73ULL * kNFull);
      for (U32 roff : {128u, 256u}) {
        {
          Encoder e(ctx, libs, kMaxM, "w4a8.mx.sl.lin");
          e.begin_stage("lin");
          linear_mxfp4_w4a8_nax(e, *tw_g, qa_ref, sa_ref, ysl_ref, std::nullopt,
                                roff, kNFull);
          auto p = e.finish();
          fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
          submit_profile(ctx, p, 73u);
          const auto got = read_u16(y_slice, 73ULL * kNFull);
          for (U32 r = 0; r < 73u; ++r) {
            for (U32 c = 0; c < roff; ++c) {
              require(got[U64(r) * kNFull + c] == 0xA5A5u,
                      "W4A8 MXFP4 linear slice clobbered [0, row_offset)");
            }
            for (U32 c = roff; c < kNFull; ++c) {
              require(got[U64(r) * kNFull + c] == full_lin[U64(r) * kNFull + c],
                      "W4A8 MXFP4 linear slice != full 0-ULP");
            }
          }
          ++checks;
        }
        {
          Encoder e(ctx, libs, kMaxM, "w4a8.mx.sl.swi");
          e.begin_stage("swi");
          linear_mxfp4_w4a8_nax_swiglu(e, *tw_g, *tw_u, qa_ref, sa_ref, ysl_ref,
                                       std::nullopt, roff, kNFull);
          auto p = e.finish();
          fill_sentinel_bf16(y_slice, U64(kMaxM) * kNFull);
          submit_profile(ctx, p, 73u);
          const auto got = read_u16(y_slice, 73ULL * kNFull);
          for (U32 r = 0; r < 73u; ++r) {
            for (U32 c = 0; c < roff; ++c) {
              require(got[U64(r) * kNFull + c] == 0xA5A5u,
                      "W4A8 MXFP4 SwiGLU slice clobbered [0, row_offset)");
            }
            for (U32 c = roff; c < kNFull; ++c) {
              require(got[U64(r) * kNFull + c] == full_swi[U64(r) * kNFull + c],
                      "W4A8 MXFP4 SwiGLU slice != full 0-ULP");
            }
          }
          ++checks;
        }
      }
    }
  }

  // 4c. Large Prefill M = 2048 (exercises GROUP_M=8 swizzle + ICB replay)
  {
    constexpr U32 kM2048 = 2048u;
    constexpr U32 kK2048 = 512u;
    constexpr U32 kN2048 = 128u;
    std::vector<std::uint16_t> x_u16;
    const auto x_f32 = make_activations(kM2048, kK2048, 888u, x_u16);
    auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
    const auto layout = w4a8_nax_workspace_layout(kM2048, kK2048);
    auto qa_buf = ctx.make_buffer(layout.qa_bytes);
    auto sa_buf = ctx.make_buffer(layout.sa_bytes);
    auto ra_buf = ctx.make_buffer(layout.ra_bytes);
    auto y_buf = ctx.make_buffer(U64(kM2048) * kN2048 * 2ULL);
    Ref x_ref{&x_buf, 0, x_buf.size()};
    Ref qa_ref{&qa_buf, 0, qa_buf.size()};
    Ref sa_ref{&sa_buf, 0, sa_buf.size()};
    Ref ra_ref{&ra_buf, 0, ra_buf.size()};
    Ref y_ref{&y_buf, 0, y_buf.size()};

    {
      Encoder eq(ctx, libs, kM2048, "w4a8.m2048.q");
      eq.begin_stage("q");
      emit_quantize_a8_g64(eq, x_ref, qa_ref, sa_ref, ra_ref, kK2048);
      auto pq = eq.finish();
      submit_profile(ctx, pq, kM2048);
    }

    auto wg4 = make_affine_weight(ctx, 4101u, kN2048, kK2048, 4u, 64u, 11u, true, 16u);
    auto wu4 = make_affine_weight(ctx, 4102u, kN2048, kK2048, 4u, 64u, 12u, true, 16u);
    auto wg8 = make_affine_weight(ctx, 4103u, kN2048, kK2048, 8u, 128u, 13u, true, 16u);
    auto wu8 = make_affine_weight(ctx, 4104u, kN2048, kK2048, 8u, 128u, 14u, true, 16u);
    AffineNaxCache ac;
    ac.register_weight(wg4.view, zero_ref);
    ac.register_weight(wu4.view, zero_ref);
    ac.register_weight(wg8.view, zero_ref);
    ac.register_weight(wu8.view, zero_ref);
    ac.build(ctx, libs, zero_ref);

    auto mg = make_mxfp4_weight(ctx, 4105u, kN2048, kK2048, 15u, true, 16u);
    auto mu = make_mxfp4_weight(ctx, 4106u, kN2048, kK2048, 16u, true, 16u);
    Mxfp4NaxCache mc;
    mc.register_weight(mg.view);
    mc.register_weight(mu.view);
    mc.build(ctx, libs);

    // Spot-check first 64 & last 64 rows plus full cosine across all 2048 rows
    auto check_m2048_lin = [&](const TiledAffineWeight &tw,
                               std::span<const float> rw, const char *lbl) {
      Encoder e(ctx, libs, kM2048, std::string("w4a8.m2048.") + lbl);
      e.begin_stage("lin");
      linear_affine_w4a8_nax(e, tw, qa_ref, sa_ref, ra_ref, y_ref);
      auto p = e.finish(true);
      submit_profile(ctx, p, kM2048, true);
      const auto got = read_u16(y_buf, U64(kM2048) * kN2048);
      const auto ref = cpu_linear_ref(x_f32, rw, kM2048, kN2048, kK2048);
      const double cs = cosine_similarity_bf16(got, ref, lbl);
      min_cos_seen = std::min(min_cos_seen, cs);
      require(cs >= kMinCos, std::string(lbl) + " M=2048 cos_sim < 0.9995");
      ++checks;
    };
    auto check_m2048_swi = [&](const TiledAffineWeight &twg,
                               const TiledAffineWeight &twu,
                               std::span<const float> rwg,
                               std::span<const float> rwu, const char *lbl) {
      Encoder e(ctx, libs, kM2048, std::string("w4a8.m2048.") + lbl);
      e.begin_stage("swi");
      linear_affine_w4a8_nax_swiglu(e, twg, twu, qa_ref, sa_ref, ra_ref, y_ref);
      auto p = e.finish(true);
      submit_profile(ctx, p, kM2048, true);
      const auto got = read_u16(y_buf, U64(kM2048) * kN2048);
      const auto ref = cpu_swiglu_ref(x_f32, rwg, rwu, kM2048, kN2048, kK2048);
      const double cs = cosine_similarity_bf16(got, ref, lbl);
      min_cos_seen = std::min(min_cos_seen, cs);
      require(cs >= kMinCos, std::string(lbl) + " M=2048 cos_sim < 0.9995");
      ++checks;
    };

    check_m2048_lin(*ac.find(4101u), wg4.ref_w, "q4_g64_lin");
    check_m2048_swi(*ac.find(4101u), *ac.find(4102u), wg4.ref_w, wu4.ref_w, "q4_g64_swi");
    check_m2048_lin(*ac.find(4103u), wg8.ref_w, "q8_g128_lin");
    check_m2048_swi(*ac.find(4103u), *ac.find(4104u), wg8.ref_w, wu8.ref_w, "q8_g128_swi");

    {
      Encoder e(ctx, libs, kM2048, "w4a8.m2048.mx_lin");
      e.begin_stage("lin");
      linear_mxfp4_w4a8_nax(e, *mc.find(4105u), qa_ref, sa_ref, y_ref);
      auto p = e.finish(true);
      submit_profile(ctx, p, kM2048, true);
      const auto got = read_u16(y_buf, U64(kM2048) * kN2048);
      const auto ref = cpu_linear_ref(x_f32, mg.ref_w, kM2048, kN2048, kK2048);
      const double cs = cosine_similarity_bf16(got, ref, "mxfp4_m2048_lin");
      min_cos_seen = std::min(min_cos_seen, cs);
      require(cs >= kMinCos, "MXFP4 M=2048 linear cos_sim < 0.9995");
      ++checks;
    }
    {
      Encoder e(ctx, libs, kM2048, "w4a8.m2048.mx_swi");
      e.begin_stage("swi");
      linear_mxfp4_w4a8_nax_swiglu(e, *mc.find(4105u), *mc.find(4106u), qa_ref, sa_ref, y_ref);
      auto p = e.finish(true);
      submit_profile(ctx, p, kM2048, true);
      const auto got = read_u16(y_buf, U64(kM2048) * kN2048);
      const auto ref = cpu_swiglu_ref(x_f32, mg.ref_w, mu.ref_w, kM2048, kN2048, kK2048);
      const double cs = cosine_similarity_bf16(got, ref, "mxfp4_m2048_swi");
      min_cos_seen = std::min(min_cos_seen, cs);
      require(cs >= kMinCos, "MXFP4 M=2048 SwiGLU cos_sim < 0.9995");
      ++checks;
    }
  }

  // 4d. Regression: explicit_t = 128 < mp_override = 2048 with N = 512
  //     (8 N-tiles x 2 M-tiles; exercises GROUP_M=8 swizzle bijection via grid_tgs)
  {
    constexpr U32 kExplicitT = 128u;
    constexpr U32 kMpBig = 2048u;
    constexpr U32 kKReg = 512u;
    constexpr U32 kNReg = 512u;
    const U64 elems = U64(kExplicitT) * kNReg;

    std::vector<std::uint16_t> x_u16;
    const auto x_f32 = make_activations(kExplicitT, kKReg, 991u, x_u16);
    auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
    Ref x_ref{&x_buf, 0, x_buf.size()};

    const auto lay_big = w4a8_nax_workspace_layout(kMpBig, kKReg);
    auto qa_big = ctx.make_buffer(lay_big.qa_bytes);
    auto sa_big = ctx.make_buffer(lay_big.sa_bytes);
    auto ra_big = ctx.make_buffer(lay_big.ra_bytes);
    Ref qab_ref{&qa_big, 0, qa_big.size()};
    Ref sab_ref{&sa_big, 0, sa_big.size()};
    Ref rab_ref{&ra_big, 0, ra_big.size()};

    const auto lay_fit = w4a8_nax_workspace_layout(kExplicitT, kKReg);
    auto qa_fit = ctx.make_buffer(lay_fit.qa_bytes);
    auto sa_fit = ctx.make_buffer(lay_fit.sa_bytes);
    auto ra_fit = ctx.make_buffer(lay_fit.ra_bytes);
    Ref qaf_ref{&qa_fit, 0, qa_fit.size()};
    Ref saf_ref{&sa_fit, 0, sa_fit.size()};
    Ref raf_ref{&ra_fit, 0, ra_fit.size()};

    auto y_big = ctx.make_buffer(elems * 2ULL);
    auto y_fit = ctx.make_buffer(elems * 2ULL);
    Ref yb_ref{&y_big, 0, y_big.size()};
    Ref yf_ref{&y_fit, 0, y_fit.size()};

    {
      Encoder eq(ctx, libs, kMpBig, "w4a8.mp_ovr.q_big");
      eq.begin_stage("q");
      emit_quantize_a8_g64(eq, x_ref, qab_ref, sab_ref, rab_ref, kKReg,
                           kExplicitT, kMpBig);
      auto pq = eq.finish();
      submit_profile(ctx, pq, kExplicitT);
    }
    {
      Encoder eq(ctx, libs, kExplicitT, "w4a8.mp_ovr.q_fit");
      eq.begin_stage("q");
      emit_quantize_a8_g64(eq, x_ref, qaf_ref, saf_ref, raf_ref, kKReg,
                           kExplicitT, kExplicitT);
      auto pq = eq.finish();
      submit_profile(ctx, pq, kExplicitT);
    }

    auto wg_a = make_affine_weight(ctx, 4201u, kNReg, kKReg, 4u, 64u, 21u, true, 16u);
    auto wu_a = make_affine_weight(ctx, 4202u, kNReg, kKReg, 4u, 64u, 22u, true, 16u);
    AffineNaxCache ac;
    ac.register_weight(wg_a.view, zero_ref);
    ac.register_weight(wu_a.view, zero_ref);
    ac.build(ctx, libs, zero_ref);
    const auto *tw_ag = ac.find(4201u);
    const auto *tw_au = ac.find(4202u);

    auto mg_m = make_mxfp4_weight(ctx, 4203u, kNReg, kKReg, 23u, true, 16u);
    auto mu_m = make_mxfp4_weight(ctx, 4204u, kNReg, kKReg, 24u, true, 16u);
    Mxfp4NaxCache mc;
    mc.register_weight(mg_m.view);
    mc.register_weight(mu_m.view);
    mc.build(ctx, libs);
    const auto *tw_mg = mc.find(4203u);
    const auto *tw_mu = mc.find(4204u);

    auto fill_alt_sentinel = [](MetalBuffer &buf, U64 count, std::uint16_t pat) {
      std::vector<std::uint16_t> s(count, pat);
      buf.write(std::as_bytes(std::span(s)));
    };

    auto verify_mp_override = [&](const char *lbl,
                                  std::span<const float> ref_f32,
                                  auto emit_fn) {
      // Pre-fill with distinct sentinels so 0-ULP equality guarantees 100% of
      // [0, explicit_t * N) elements were actively written in both dispatches.
      fill_alt_sentinel(y_big, elems, 0xA5A5u);
      fill_alt_sentinel(y_fit, elems, 0x5A5Au);
      {
        Encoder e(ctx, libs, kMpBig, std::string("w4a8.mp_ovr.big.") + lbl);
        e.begin_stage("run");
        emit_fn(e, qab_ref, sab_ref, rab_ref, yb_ref, kMpBig);
        auto p = e.finish(true);
        submit_profile(ctx, p, kExplicitT, true);
      }
      {
        Encoder e(ctx, libs, kExplicitT, std::string("w4a8.mp_ovr.fit.") + lbl);
        e.begin_stage("run");
        emit_fn(e, qaf_ref, saf_ref, raf_ref, yf_ref, kExplicitT);
        auto p = e.finish();
        submit_profile(ctx, p, kExplicitT, false);
      }
      const auto got_big = read_u16(y_big, elems);
      const auto got_fit = read_u16(y_fit, elems);
      require(got_big == got_fit,
              std::string("mp_override=2048 vs 128 mismatch (swizzle bijection) for ") +
                  lbl);
      const double cs = cosine_similarity_bf16(got_big, ref_f32, lbl);
      min_cos_seen = std::min(min_cos_seen, cs);
      require(cs >= kMinCos,
              std::string("mp_override=2048 cos_sim < 0.9995 for ") + lbl);
      ++checks;
    };

    const auto ref_a_lin = cpu_linear_ref(x_f32, wg_a.ref_w, kExplicitT, kNReg, kKReg);
    const auto ref_a_swi = cpu_swiglu_ref(x_f32, wg_a.ref_w, wu_a.ref_w, kExplicitT, kNReg, kKReg);
    const auto ref_m_lin = cpu_linear_ref(x_f32, mg_m.ref_w, kExplicitT, kNReg, kKReg);
    const auto ref_m_swi = cpu_swiglu_ref(x_f32, mg_m.ref_w, mu_m.ref_w, kExplicitT, kNReg, kKReg);

    verify_mp_override(
        "aff_q4_lin_mp2048", ref_a_lin,
        [&](Encoder &e, Ref qa, Ref sa, Ref ra, Ref y, U32 mp_ovr) {
          linear_affine_w4a8_nax(e, *tw_ag, qa, sa, ra, y, kExplicitT, 0u, 0u,
                                 0u, mp_ovr);
        });
    verify_mp_override(
        "aff_q4_swi_mp2048", ref_a_swi,
        [&](Encoder &e, Ref qa, Ref sa, Ref ra, Ref y, U32 mp_ovr) {
          linear_affine_w4a8_nax_swiglu(e, *tw_ag, *tw_au, qa, sa, ra, y,
                                        kExplicitT, 0u, 0u, 0u, mp_ovr);
        });
    verify_mp_override(
        "mx_lin_mp2048", ref_m_lin,
        [&](Encoder &e, Ref qa, Ref sa, Ref /*ra*/, Ref y, U32 mp_ovr) {
          linear_mxfp4_w4a8_nax(e, *tw_mg, qa, sa, y, kExplicitT, 0u, 0u, 0u,
                                mp_ovr);
        });
    verify_mp_override(
        "mx_swi_mp2048", ref_m_swi,
        [&](Encoder &e, Ref qa, Ref sa, Ref /*ra*/, Ref y, U32 mp_ovr) {
          linear_mxfp4_w4a8_nax_swiglu(e, *tw_mg, *tw_mu, qa, sa, y, kExplicitT,
                                       0u, 0u, 0u, mp_ovr);
        });
  }

  return {{"checks", checks}, {"min_cos_sim", min_cos_seen}, {"passed", true}};
}

// ============================================================================
// Test 5: MoE Base 1.2 out64 (0-ULP) + Route A MoE W4A8 (cos_sim >= 0.9995)
// ============================================================================
std::vector<float> cpu_moe_ffn_ref(std::span<const float> norm,
                                   std::span<const float> gw,
                                   std::span<const float> uw,
                                   std::span<const float> dw,
                                   std::span<const std::uint32_t> route_ids,
                                   std::span<const std::uint16_t> route_scores,
                                   U32 t, U32 h, U32 f, U32 top_k) {
  std::vector<float> routed(U64(t) * h, 0.0f);
  std::vector<float> mid(f, 0.0f);
  for (U32 tok = 0; tok < t; ++tok) {
    for (U32 slot = 0; slot < top_k; ++slot) {
      const U32 assign = tok * top_k + slot;
      const U32 e = route_ids[assign];
      const float sc = bf16_val(route_scores[assign]);
      for (U32 j = 0; j < f; ++j) {
        double ag = 0.0, au = 0.0;
        const U64 row_off = (U64(e) * f + j) * h;
        for (U32 d = 0; d < h; ++d) {
          const double xv = double(norm[U64(tok) * h + d]);
          ag += xv * double(gw[row_off + d]);
          au += xv * double(uw[row_off + d]);
        }
        const float gv = bf16_val(float_to_bf16(float(ag)));
        const float uv = bf16_val(float_to_bf16(float(au)));
        mid[j] = bf16_val(float_to_bf16(silu_ref(gv) * uv));
      }
      for (U32 c = 0; c < h; ++c) {
        double ad = 0.0;
        const U64 row_off = (U64(e) * h + c) * f;
        for (U32 j = 0; j < f; ++j) {
          ad += double(mid[j]) * double(dw[row_off + j]);
        }
        routed[U64(tok) * h + c] += bf16_val(float_to_bf16(float(ad) * sc));
      }
    }
  }
  return routed;
}

Json test_moe_out64_and_w4a8(const MetalContext &ctx,
                             const std::filesystem::path &libs) {
  constexpr U32 kMaxT = 256u;
  constexpr U32 kH = 256u;
  constexpr U32 kF = 256u;
  constexpr U32 kExperts = 256u;
  constexpr U32 kTopK = 8u;
  constexpr double kMinCos = 0.9995;

  std::vector<std::uint16_t> norm_u16;
  const auto norm_f32 = make_activations(kMaxT, kH, 901u, norm_u16);
  auto norm_buf = ctx.upload(std::as_bytes(std::span(norm_u16)));

  // Route assignments: mix hot experts (valid_rows > 32 -> full BM=64 tiles!)
  // and cold experts (valid_rows <= 32 -> compact32 tiles + 8-row zero-pad!).
  std::vector<std::uint32_t> route_ids(U64(kMaxT) * kTopK);
  std::vector<std::uint16_t> route_scores(U64(kMaxT) * kTopK);
  for (U32 tok = 0; tok < kMaxT; ++tok) {
    for (U32 s = 0; s < kTopK; ++s) {
      // Slots 0..2 hit experts 0..5 heavily (40..130 tokens/expert -> > 32 rows);
      // Slots 3..7 spread across experts 6..255 (1..15 tokens/expert -> compact32!).
      const U32 e = (s < 3u)
                        ? ((tok + s * 2u) % 6u)
                        : (6u + ((tok * 13u + s * 37u) % (kExperts - 6u)));
      route_ids[U64(tok) * kTopK + s] = e;
      route_scores[U64(tok) * kTopK + s] =
          float_to_bf16(0.05f + 0.03f * float((tok + s) % 7u));
    }
  }
  auto rids_buf = ctx.upload(std::as_bytes(std::span(route_ids)));
  auto rsc_buf = ctx.upload(std::as_bytes(std::span(route_scores)));
  auto exp_out_buf = ctx.make_buffer(U64(kMaxT) * kTopK * kH * 2ULL);
  auto routed_a = ctx.make_buffer(U64(kMaxT) * kH * 2ULL);
  auto routed_b = ctx.make_buffer(U64(kMaxT) * kH * 2ULL);

  std::vector<std::unique_ptr<MetalBuffer>> ws_bufs;
  MoeNaxWorkspace ws;
  ws.allocate(ctx, kMaxT, kH, kF, ws_bufs, kExperts, kTopK);

  const auto a8_layout = w4a8_nax_workspace_layout(kMaxT, kH);
  auto qa_buf = ctx.make_buffer(a8_layout.qa_bytes);
  auto sa_buf = ctx.make_buffer(a8_layout.sa_bytes);
  auto ra_buf = ctx.make_buffer(a8_layout.ra_bytes);

  Ref norm_ref{&norm_buf, 0, norm_buf.size()};
  Ref rids_ref{&rids_buf, 0, rids_buf.size()};
  Ref rsc_ref{&rsc_buf, 0, rsc_buf.size()};
  Ref exp_ref{&exp_out_buf, 0, exp_out_buf.size()};
  Ref routed_a_ref{&routed_a, 0, routed_a.size()};
  Ref routed_b_ref{&routed_b, 0, routed_b.size()};
  Ref qa_ref{&qa_buf, 0, qa_buf.size()};
  Ref sa_ref{&sa_buf, 0, sa_buf.size()};
  Ref ra_ref{&ra_buf, 0, ra_buf.size()};

  auto quantize_norm = [&](U32 live_t) {
    Encoder eq(ctx, libs, kMaxT, "moe.w4a8.q");
    eq.begin_stage("q");
    emit_quantize_a8_g64(eq, norm_ref, qa_ref, sa_ref, ra_ref, kH);
    auto pq = eq.finish();
    submit_profile(ctx, pq, live_t);
  };

  double min_cos_seen = 1.0;
  U32 checks = 0;

  // 5a. MoE MXFP4: Base 1.2 out64 0-ULP parity vs bm64_bn64 + Route A W4A8 accuracy
  {
    auto mg = make_mxfp4_weight(ctx, 5001u, kExperts * kF, kH, 11u, false, 0u, true);
    auto mu = make_mxfp4_weight(ctx, 5002u, kExperts * kF, kH, 12u, false, 0u, true);
    auto md = make_mxfp4_weight(ctx, 5003u, kExperts * kH, kF, 13u, false, 0u, true);

    // Base 1.2 out64 vs out32 at T=256 (0-ULP!)
    {
      Encoder e32(ctx, libs, kMaxT, "moe.mx.base32");
      e32.begin_stage("moe");
      require(emit_moe_nax_ffn(e32, ws, mg.view, mu.view, md.view, norm_ref,
                               rids_ref, rsc_ref, exp_ref, routed_a_ref, kMaxT,
                               kH, kF, kTopK, kExperts, 64u, std::nullopt, {},
                               true, false),
              "emit_moe_nax_ffn out32 failed");
      auto p32 = e32.finish();
      submit_profile(ctx, p32, kMaxT);

      Encoder e64(ctx, libs, kMaxT, "moe.mx.base64");
      e64.begin_stage("moe");
      require(emit_moe_nax_ffn(e64, ws, mg.view, mu.view, md.view, norm_ref,
                               rids_ref, rsc_ref, exp_ref, routed_b_ref, kMaxT,
                               kH, kF, kTopK, kExperts, 64u, std::nullopt, {},
                               true, true),
              "emit_moe_nax_ffn out64 failed");
      auto p64 = e64.finish();
      submit_profile(ctx, p64, kMaxT);

      require(read_u16(routed_a, U64(kMaxT) * kH) ==
                  read_u16(routed_b, U64(kMaxT) * kH),
              "Base 1.2 MoE MXFP4 out64 != bm64_bn64 0-ULP");
      ++checks;
    }

    // Route A MoE MXFP4 W4A8 at T = {65, 256}
    for (U32 live : {65u, 256u}) {
      quantize_norm(live);
      Encoder ew(ctx, libs, kMaxT, "moe.mx.w4a8");
      ew.begin_stage("moe_w4a8");
      require(emit_moe_w4a8_nax_ffn(ew, ws, mg.view, mu.view, md.view, qa_ref,
                                    sa_ref, ra_ref, a8_layout.mp, rids_ref,
                                    rsc_ref, exp_ref, routed_a_ref, kMaxT, kH,
                                    kF, kTopK, kExperts),
              "emit_moe_w4a8_nax_ffn MXFP4 failed");
      auto pw = ew.finish();
      submit_profile(ctx, pw, live);
      const auto ref = cpu_moe_ffn_ref(norm_f32, mg.ref_w, mu.ref_w, md.ref_w,
                                       route_ids, route_scores, live, kH, kF,
                                       kTopK);
      const double cs = cosine_similarity_bf16(
          read_u16(routed_a, U64(live) * kH), ref, "moe_mxfp4_w4a8");
      min_cos_seen = std::min(min_cos_seen, cs);
      require(cs >= kMinCos,
              "MoE MXFP4 W4A8 cos_sim=" + std::to_string(cs) + " < 0.9995");
      ++checks;
    }
  }

  // 5b. MoE Affine: all 10 (q3, q4, q5, q6, q8) x (g64, g128) — Base 1.2 out64 0-ULP + Route A W4A8
  U32 wid = 5100u;
  for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
    for (U32 gs : {64u, 128u}) {
      auto ag = make_affine_weight(ctx, wid++, kExperts * kF, kH, bits, gs,
                                   bits * 7u + gs, false, 0u, true);
      auto au = make_affine_weight(ctx, wid++, kExperts * kF, kH, bits, gs,
                                   bits * 11u + gs, false, 0u, true);
      auto ad = make_affine_weight(ctx, wid++, kExperts * kH, kF, bits, gs,
                                   bits * 13u + gs, false, 0u, true);

      // Base 1.2 out64 vs out32 at T=256 (0-ULP!)
      {
        Encoder e32(ctx, libs, kMaxT, "moe.aff.base32");
        e32.begin_stage("moe");
        require(emit_moe_nax_ffn(e32, ws, ag.view, au.view, ad.view, norm_ref,
                                 rids_ref, rsc_ref, exp_ref, routed_a_ref,
                                 kMaxT, kH, kF, kTopK, kExperts, 64u,
                                 std::nullopt, {}, true, false),
                "emit_moe_nax_ffn affine out32 failed");
        auto p32 = e32.finish();
        submit_profile(ctx, p32, kMaxT);

        Encoder e64(ctx, libs, kMaxT, "moe.aff.base64");
        e64.begin_stage("moe");
        require(emit_moe_nax_ffn(e64, ws, ag.view, au.view, ad.view, norm_ref,
                                 rids_ref, rsc_ref, exp_ref, routed_b_ref,
                                 kMaxT, kH, kF, kTopK, kExperts, 64u,
                                 std::nullopt, {}, true, true),
                "emit_moe_nax_ffn affine out64 failed");
        auto p64 = e64.finish();
        submit_profile(ctx, p64, kMaxT);

        require(read_u16(routed_a, U64(kMaxT) * kH) ==
                    read_u16(routed_b, U64(kMaxT) * kH),
                "Base 1.2 MoE Affine out64 != bm64_bn64 0-ULP for q" +
                    std::to_string(bits) + "_g" + std::to_string(gs));
        ++checks;
      }

      // Route A MoE Affine W4A8 at T = {65, 256}
      for (U32 live : {65u, 256u}) {
        quantize_norm(live);
        Encoder ew(ctx, libs, kMaxT, "moe.aff.w4a8");
        ew.begin_stage("moe_w4a8");
        require(emit_moe_w4a8_nax_ffn(ew, ws, ag.view, au.view, ad.view, qa_ref,
                                      sa_ref, ra_ref, a8_layout.mp, rids_ref,
                                      rsc_ref, exp_ref, routed_a_ref, kMaxT, kH,
                                      kF, kTopK, kExperts),
                "emit_moe_w4a8_nax_ffn affine failed");
        auto pw = ew.finish();
        submit_profile(ctx, pw, live);
        const auto ref = cpu_moe_ffn_ref(norm_f32, ag.ref_w, au.ref_w, ad.ref_w,
                                         route_ids, route_scores, live, kH, kF,
                                         kTopK);
        const double cs = cosine_similarity_bf16(
            read_u16(routed_a, U64(live) * kH), ref,
            "moe_affine_w4a8_q" + std::to_string(bits) + "_g" +
                std::to_string(gs));
        min_cos_seen = std::min(min_cos_seen, cs);
        require(cs >= kMinCos,
                "MoE Affine W4A8 q" + std::to_string(bits) + "_g" +
                    std::to_string(gs) + " cos_sim=" + std::to_string(cs) +
                    " < 0.9995");
        ++checks;
      }
    }
  }

  return {{"checks", checks}, {"min_cos_sim", min_cos_seen}, {"passed", true}};
}

// ============================================================================
// Test 6: .miv v2 lane_v2 + All FWHT-64 Runtime Consumers Parity
//         (T=1, B=2, T=4, T=33, T=65, T=128, T=256)
// ============================================================================
Json test_miv2_fwht64_and_all_runtime_consumers(const MetalContext &ctx,
                                                const std::filesystem::path &libs) {
  constexpr U32 kK = 512u;
  constexpr U32 kN = 128u;
  constexpr U32 kMaxM = 128u;
  constexpr double kMinCosW4A8 = 0.9995;
  constexpr double kMinCosW4A4 = 0.9850;
  U32 checks = 0;
  double min_w4a8_cos = 1.0;

  auto zero_buf = ctx.make_buffer(256);
  zero_buf.write(std::vector<std::byte>(256));
  Ref zero_ref{&zero_buf, 0, zero_buf.size()};

  std::vector<std::uint16_t> x_u16;
  const auto x_f32 = make_activations(kMaxM, kK, 1301u, x_u16);
  auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
  Ref x_ref{&x_buf, 0, x_buf.size()};

  auto x_rot_buf = ctx.make_buffer(U64(kMaxM) * kK * 2ULL);
  Ref x_rot_ref{&x_rot_buf, 0, x_rot_buf.size()};
  auto xs_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * (kK / 64u) * 4ULL);
  Ref xs_ref{&xs_buf, 0, xs_buf.size()};
  auto f16_sc_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * kK * 2ULL);
  auto f16_rs_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * 4ULL);
  Ref f16_sc{&f16_sc_buf, 0, f16_sc_buf.size()};
  Ref f16_rs{&f16_rs_buf, 0, f16_rs_buf.size()};
  auto a4_c_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * (kK / 2u));
  auto a4_s_buf = ctx.make_private_buffer(U64(kMaxM + 16u) * (kK / 32u));
  Ref a4_c{&a4_c_buf, 0, a4_c_buf.size()};
  Ref a4_s{&a4_s_buf, 0, a4_s_buf.size()};

  const auto a8_in_lay = w4a8_nax_workspace_layout(kMaxM, kK);
  auto qa_in = ctx.make_buffer(a8_in_lay.qa_bytes);
  auto sa_in = ctx.make_buffer(a8_in_lay.sa_bytes);
  auto ra_in = ctx.make_buffer(a8_in_lay.ra_bytes);
  Ref qa_in_ref{&qa_in, 0, qa_in.size()};
  Ref sa_in_ref{&sa_in, 0, sa_in.size()};
  Ref ra_in_ref{&ra_in, 0, ra_in.size()};

  const auto a8_mid_lay = w4a8_nax_workspace_layout(kMaxM, kK);
  auto qa_mid = ctx.make_buffer(a8_mid_lay.qa_bytes);
  auto sa_mid = ctx.make_buffer(a8_mid_lay.sa_bytes);
  auto ra_mid = ctx.make_buffer(a8_mid_lay.ra_bytes);
  Ref qa_mid_ref{&qa_mid, 0, qa_mid.size()};
  Ref sa_mid_ref{&sa_mid, 0, sa_mid.size()};
  Ref ra_mid_ref{&ra_mid, 0, ra_mid.size()};

  auto y_a_buf = ctx.make_buffer(U64(kMaxM) * kK * 2ULL);
  auto y_b_buf = ctx.make_buffer(U64(kMaxM) * kK * 2ULL);
  Ref y_a_ref{&y_a_buf, 0, y_a_buf.size()};
  Ref y_b_ref{&y_b_buf, 0, y_b_buf.size()};

  U32 wid = 6000u;

  // 6a. Load-time repack_oq_v1_to_lane_v2 vs native oq_nax_lane_v2 0-ULP parity across all 10 formats
  for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
    for (U32 gs : {64u, 128u}) {
      auto w_v1 = make_affine_weight(ctx, wid++, kN, kK, bits, gs, bits * 17u + gs, true, 16u, false, false);
      auto w_v2 = make_affine_weight(ctx, wid++, kN, kK, bits, gs, bits * 17u + gs, true, 16u, false, true);
      AffineNaxCache cache;
      cache.register_weight(w_v1.view, zero_ref);
      cache.register_weight(w_v2.view, zero_ref);
      cache.build(ctx, libs, zero_ref);
      const auto *tw1 = cache.find(w_v1.view.object_id);
      const auto *tw2 = cache.find(w_v2.view.object_id);
      require(tw1 != nullptr && tw2 != nullptr, "6a cache missing");
      {
        Encoder e1(ctx, libs, kMaxM, "miv2.6a.v1");
        e1.begin_stage("lin");
        linear_affine_nax(e1, *tw1, x_ref, xs_ref, y_a_ref, std::nullopt, 1u);
        auto p1 = e1.finish();
        submit_profile(ctx, p1, 65u);
      }
      {
        Encoder e2(ctx, libs, kMaxM, "miv2.6a.v2");
        e2.begin_stage("lin");
        linear_affine_nax(e2, *tw2, x_ref, xs_ref, y_b_ref, std::nullopt, 1u);
        auto p2 = e2.finish();
        submit_profile(ctx, p2, 65u);
      }
      require(read_u16(y_a_buf, 65ULL * kN) == read_u16(y_b_buf, 65ULL * kN),
              "6a repack_oq_v1_to_lane_v2 != native lane_v2 0-ULP for q" +
                  std::to_string(bits) + "_g" + std::to_string(gs));
      ++checks;
    }
  }

  // 6b. omlx_m5_fwht64_bf16 (row_mul=1 & 2) and omlx_m5_quantize_a8_g64_fwht64 vs CPU H_64
  {
    std::vector<float> x_rot_cpu = x_f32;
    fwht64_rows_cpu(x_rot_cpu, kMaxM, kK);
    for (U32 live : {1u, 4u, 65u}) {
      Encoder ef(ctx, libs, kMaxM, "miv2.6b.fwht");
      ef.begin_stage("fwht");
      emit_fwht64_bf16(ef, x_ref, x_rot_ref, kK);
      auto pf = ef.finish();
      submit_profile(ctx, pf, live);
      const double cs = cosine_similarity_bf16(
          read_u16(x_rot_buf, U64(live) * kK),
          std::span(x_rot_cpu).first(U64(live) * kK), "6b fwht64_bf16");
      require(cs >= 0.99995, "6b omlx_m5_fwht64_bf16 cos_sim < 0.99995");
      ++checks;

      Encoder eq(ctx, libs, kMaxM, "miv2.6b.qa8_fwht");
      eq.begin_stage("qa8_fwht");
      emit_quantize_a8_g64(eq, x_ref, qa_in_ref, sa_in_ref, ra_in_ref, kK,
                           std::nullopt, std::nullopt, true);
      auto pq = eq.finish();
      submit_profile(ctx, pq, live);
      const auto got_qa = read_i8(qa_in, U64(live) * kK);
      const auto got_sa = read_f32(sa_in, U64(kK / 64u) * a8_in_lay.mp);
      const auto got_ra = read_f32(ra_in, U64(kK / 64u) * a8_in_lay.mp);
      std::vector<float> deq(U64(live) * kK);
      for (U32 r = 0; r < live; ++r) {
        for (U32 g = 0; g < kK / 64u; ++g) {
          const float sa = got_sa[U64(g) * a8_in_lay.mp + r];
          int sum_q = 0;
          for (U32 j = 0; j < 64u; ++j) {
            const int q = int(got_qa[U64(r) * kK + g * 64u + j]);
            sum_q += q;
            deq[U64(r) * kK + g * 64u + j] = float(q) * sa;
          }
          require(got_ra[U64(g) * a8_in_lay.mp + r] == float(sum_q),
                  "6b quantize_a8_g64_fwht64 ra exact sum mismatch");
        }
      }
      double dot = 0.0, na = 0.0, nb = 0.0;
      for (std::size_t i = 0; i < deq.size(); ++i) {
        const double a = deq[i], b = x_rot_cpu[i];
        dot += a * b; na += a * a; nb += b * b;
      }
      const double qcs = dot / (std::sqrt(na) * std::sqrt(nb));
      require(qcs >= 0.9997, "6b quantize_a8_g64_fwht64 dequant cos_sim < 0.9997");
      ++checks;
    }
    // Verify row_mul = 2 mode
    {
      Encoder ef2(ctx, libs, kMaxM, "miv2.6b.fwht_rm2");
      ef2.begin_stage("fwht2");
      emit_fwht64_bf16(ef2, x_ref, x_rot_ref, kK, std::nullopt, 2u);
      auto pf2 = ef2.finish();
      submit_profile(ctx, pf2, 16u); // 16 * 2 = 32 rows
      const double cs = cosine_similarity_bf16(
          read_u16(x_rot_buf, 32ULL * kK),
          std::span(x_rot_cpu).first(32ULL * kK), "6b fwht64_bf16 row_mul=2");
      require(cs >= 0.99995, "6b omlx_m5_fwht64_bf16 row_mul=2 failed");
      ++checks;
    }
  }

  // 6c & 6d. Dense Mode B (SwiGLU + FWHT-64 + GS64 A8 -> W4A8 fwht64 down_proj)
  //          for BOTH W4A8 Mode B AND WP3.1 W4A16 out64 Mode B (Affine + MXFP4, also_write_bf16=false/true)
  {
    auto quantize_in = [&](U32 live) {
      Encoder eq(ctx, libs, kMaxM, "miv2.6cd.qin");
      eq.begin_stage("q");
      emit_quantize_a8_g64(eq, x_ref, qa_in_ref, sa_in_ref, ra_in_ref, kK);
      auto pq = eq.finish();
      submit_profile(ctx, pq, live);
    };

    // Affine: q4_g64, q5_g64, q3_g64, q8_g128, mixed q4_q56
    for (const auto &[gb, ub, gs] : std::array<std::tuple<U32, U32, U32>, 5>{{
             {4u, 4u, 64u}, {5u, 5u, 64u}, {3u, 3u, 64u}, {8u, 8u, 128u}, {4u, 5u, 64u}}}) {
      auto wg = make_affine_weight(ctx, wid++, kK, kK, gb, gs, gb * 31u + gs, true, 16u, false, true, false);
      auto wu = make_affine_weight(ctx, wid++, kK, kK, ub, gs, ub * 37u + gs, true, 16u, false, true, false);
      auto wd = make_affine_weight(ctx, wid++, kN, kK, gb, gs, gb * 43u + gs, true, 16u, false, true, true);
      AffineNaxCache ac;
      ac.register_weight(wg.view, zero_ref);
      ac.register_weight(wu.view, zero_ref);
      ac.register_weight(wd.view, zero_ref);
      ac.build(ctx, libs, zero_ref);
      const auto *twg = ac.find(wg.view.object_id);
      const auto *twu = ac.find(wu.view.object_id);
      const auto *twd = ac.find(wd.view.object_id);
      const auto ref_mid = cpu_swiglu_ref(x_f32, wg.ref_w, wu.ref_w, kMaxM, kK, kK);
      const auto ref_dn = cpu_linear_ref(ref_mid, wd.ref_w, kMaxM, kN, kK);

      for (U32 live : {33u, 128u}) {
        quantize_in(live);
        // W4A8 Mode A reference BF16 mid
        {
          Encoder ea(ctx, libs, kMaxM, "miv2.6c.aff.modeA");
          ea.begin_stage("a");
          linear_affine_w4a8_nax_swiglu(ea, *twg, *twu, qa_in_ref, sa_in_ref, ra_in_ref, y_a_ref);
          auto pa = ea.finish();
          submit_profile(ctx, pa, live);
        }
        const auto mode_a_mid = read_u16(y_a_buf, U64(live) * kK);

        // W4A8 Mode B (also_write_bf16=true) -> W4A8 down_proj
        {
          Encoder eb(ctx, libs, kMaxM, "miv2.6c.aff.modeB");
          eb.begin_stage("b");
          linear_affine_w4a8_nax_swiglu_fwht64_a8(
              eb, *twg, *twu, qa_in_ref, sa_in_ref, ra_in_ref, qa_mid_ref,
              sa_mid_ref, ra_mid_ref, y_b_ref, true);
          linear_affine_w4a8_nax(eb, *twd, qa_mid_ref, sa_mid_ref, ra_mid_ref, y_a_ref);
          auto pb = eb.finish();
          submit_profile(ctx, pb, live);
        }
        require(read_u16(y_b_buf, U64(live) * kK) == mode_a_mid,
                "6c W4A8 Affine Mode B also_write_bf16 != Mode A 0-ULP");
        const double cs_w4a8 = cosine_similarity_bf16(
            read_u16(y_a_buf, U64(live) * kN),
            std::span(ref_dn).first(U64(live) * kN), "6c w4a8 modeB->down");
        min_w4a8_cos = std::min(min_w4a8_cos, cs_w4a8);
        require(cs_w4a8 >= kMinCosW4A8,
                "6c W4A8 Affine Mode B -> down_proj cos_sim=" + std::to_string(cs_w4a8));
        ++checks;

        // WP3.1 W4A16 out64 Mode B (also_write_bf16=true) -> W4A8 down_proj
        {
          ScopedEnv en_out64("OMLX_M5_ENABLE_SWIGLU_OUT64", "1");
          ScopedEnv force64("OMLX_M5_WIDE_PREFILL_TILE", "64");
          Encoder e16a(ctx, libs, kMaxM, "miv2.6d.aff.modeA");
          e16a.begin_stage("a16");
          linear_affine_nax_swiglu(e16a, *twg, *twu, x_ref, xs_ref, y_a_ref, 1u);
          auto p16a = e16a.finish();
          submit_profile(ctx, p16a, live);
        }
        const auto w4a16_mode_a = read_u16(y_a_buf, U64(live) * kK);
        {
          Encoder e16b(ctx, libs, kMaxM, "miv2.6d.aff.modeB");
          e16b.begin_stage("b16");
          linear_affine_nax_swiglu_fwht64_a8(
              e16b, *twg, *twu, x_ref, qa_mid_ref, sa_mid_ref, ra_mid_ref,
              y_b_ref, true, a8_mid_lay.mp);
          linear_affine_w4a8_nax(e16b, *twd, qa_mid_ref, sa_mid_ref, ra_mid_ref, y_a_ref);
          auto p16b = e16b.finish();
          submit_profile(ctx, p16b, live);
        }
        {
          const auto w4a16_mode_b = read_u16(y_b_buf, U64(live) * kK);
          if (w4a16_mode_b != w4a16_mode_a) {
            U64 n_diff = 0;
            U64 first_i = 0;
            std::string cols_pat;
            for (U32 c = 0; c < 64u; ++c) {
              cols_pat.push_back(w4a16_mode_b[c] == w4a16_mode_a[c] ? '.' : 'X');
            }
            for (U64 idx = 0; idx < w4a16_mode_a.size(); ++idx) {
              if (w4a16_mode_b[idx] != w4a16_mode_a[idx]) {
                if (n_diff == 0) first_i = idx;
                ++n_diff;
              }
            }
            const double cos_a = cosine_similarity_bf16(
                w4a16_mode_a, std::span(ref_mid).first(U64(live) * kK), "modeA");
            const double cos_b = cosine_similarity_bf16(
                w4a16_mode_b, std::span(ref_mid).first(U64(live) * kK), "modeB");
            require(false,
                    "6d WP3.1 Affine out64 Mode B also_write_bf16 != Mode A 0-ULP (gb=" +
                    std::to_string(gb) + " ub=" + std::to_string(ub) + " gs=" +
                    std::to_string(gs) + " live=" + std::to_string(live) +
                    " n_diff=" + std::to_string(n_diff) + "/" +
                    std::to_string(w4a16_mode_a.size()) + " first=(" +
                    std::to_string(first_i / kK) + "," + std::to_string(first_i % kK) +
                    ") row0[0..63]=" + cols_pat +
                    " cosA=" + std::to_string(cos_a) + " cosB=" + std::to_string(cos_b) + ")");
          }
        }
        const double cs_16 = cosine_similarity_bf16(
            read_u16(y_a_buf, U64(live) * kN),
            std::span(ref_dn).first(U64(live) * kN), "6d wp31 aff modeB->down");
        min_w4a8_cos = std::min(min_w4a8_cos, cs_16);
        require(cs_16 >= kMinCosW4A8,
                "6d WP3.1 Affine out64 Mode B -> down_proj cos_sim=" + std::to_string(cs_16));
        ++checks;
      }
    }

    // MXFP4 W4A8 Mode B + WP3.1 W4A16 out64 Mode B -> W4A8 MXFP4 down_proj
    {
      auto mg = make_mxfp4_weight(ctx, wid++, kK, kK, 141u, true, 16u, false, false);
      auto mu = make_mxfp4_weight(ctx, wid++, kK, kK, 142u, true, 16u, false, false);
      auto md = make_mxfp4_weight(ctx, wid++, kN, kK, 143u, true, 16u, false, true);
      Mxfp4NaxCache mc;
      mc.register_weight(mg.view);
      mc.register_weight(mu.view);
      mc.register_weight(md.view);
      mc.build(ctx, libs);
      const auto *twg = mc.find(mg.view.object_id);
      const auto *twu = mc.find(mu.view.object_id);
      const auto *twd = mc.find(md.view.object_id);
      const auto ref_mid = cpu_swiglu_ref(x_f32, mg.ref_w, mu.ref_w, kMaxM, kK, kK);
      const auto ref_dn = cpu_linear_ref(ref_mid, md.ref_w, kMaxM, kN, kK);

      for (U32 live : {33u, 128u}) {
        quantize_in(live);
        {
          Encoder ea(ctx, libs, kMaxM, "miv2.6c.mx.modeA");
          ea.begin_stage("a");
          linear_mxfp4_w4a8_nax_swiglu(ea, *twg, *twu, qa_in_ref, sa_in_ref, y_a_ref);
          auto pa = ea.finish();
          submit_profile(ctx, pa, live);
        }
        const auto mode_a_mid = read_u16(y_a_buf, U64(live) * kK);
        {
          Encoder eb(ctx, libs, kMaxM, "miv2.6c.mx.modeB");
          eb.begin_stage("b");
          linear_mxfp4_w4a8_nax_swiglu_fwht64_a8(
              eb, *twg, *twu, qa_in_ref, sa_in_ref, qa_mid_ref, sa_mid_ref,
              ra_mid_ref, y_b_ref, true);
          linear_mxfp4_w4a8_nax(eb, *twd, qa_mid_ref, sa_mid_ref, y_a_ref);
          auto pb = eb.finish();
          submit_profile(ctx, pb, live);
        }
        require(read_u16(y_b_buf, U64(live) * kK) == mode_a_mid,
                "6c W4A8 MXFP4 Mode B also_write_bf16 != Mode A 0-ULP");
        const double cs = cosine_similarity_bf16(
            read_u16(y_a_buf, U64(live) * kN),
            std::span(ref_dn).first(U64(live) * kN), "6c w4a8 mx modeB->down");
        min_w4a8_cos = std::min(min_w4a8_cos, cs);
        require(cs >= kMinCosW4A8, "6c W4A8 MXFP4 Mode B -> down_proj failed");
        ++checks;

        // WP3.1 W4A16 MXFP4 out64 Mode B
        {
          ScopedEnv en_out64("OMLX_M5_ENABLE_SWIGLU_OUT64", "1");
          ScopedEnv force64("OMLX_M5_WIDE_PREFILL_TILE", "64");
          Encoder e16a(ctx, libs, kMaxM, "miv2.6d.mx.modeA");
          e16a.begin_stage("a16");
          linear_mxfp4_nax_swiglu(e16a, *twg, *twu, x_ref, f16_sc, f16_rs, y_a_ref, 1u);
          auto p16a = e16a.finish();
          submit_profile(ctx, p16a, live);
        }
        const auto w4a16_mode_a = read_u16(y_a_buf, U64(live) * kK);
        {
          Encoder e16b(ctx, libs, kMaxM, "miv2.6d.mx.modeB");
          e16b.begin_stage("b16");
          linear_mxfp4_nax_swiglu_fwht64_a8(
              e16b, *twg, *twu, x_ref, qa_mid_ref, sa_mid_ref, ra_mid_ref,
              y_b_ref, true, a8_mid_lay.mp);
          linear_mxfp4_w4a8_nax(e16b, *twd, qa_mid_ref, sa_mid_ref, y_a_ref);
          auto p16b = e16b.finish();
          submit_profile(ctx, p16b, live);
        }
        require(read_u16(y_b_buf, U64(live) * kK) == w4a16_mode_a,
                "6d WP3.1 MXFP4 out64 Mode B also_write_bf16 != Mode A 0-ULP");
        const double cs16 = cosine_similarity_bf16(
            read_u16(y_a_buf, U64(live) * kN),
            std::span(ref_dn).first(U64(live) * kN), "6d wp31 mx modeB->down");
        min_w4a8_cos = std::min(min_w4a8_cos, cs16);
        require(cs16 >= kMinCosW4A8, "6d WP3.1 MXFP4 out64 Mode B -> down_proj failed");
        ++checks;
      }
    }
  }

  // 6e. W4A16 & W4A4 fwht64 linear fallbacks at T=4 (skinny) & T=128 (wide)
  {
    for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
      auto w = make_affine_weight(ctx, wid++, kN, kK, bits, 64u, bits * 53u, true, 16u, false, true, true);
      AffineNaxCache ac;
      ac.register_weight(w.view, zero_ref);
      ac.build(ctx, libs, zero_ref);
      const auto *tw = ac.find(w.view.object_id);
      const auto ref = cpu_linear_ref(x_f32, w.ref_w, kMaxM, kN, kK);
      for (U32 t_cur : {4u, 128u}) {
        Encoder e(ctx, libs, t_cur, "miv2.6e.aff.t" + std::to_string(t_cur));
        e.begin_stage("lin");
        linear_affine_nax_fwht64(e, *tw, x_ref, x_rot_ref, xs_ref, y_a_ref, std::nullopt, 1u);
        auto p = e.finish();
        submit_profile(ctx, p, t_cur);
        const double cs = cosine_similarity_bf16(
            read_u16(y_a_buf, U64(t_cur) * kN),
            std::span(ref).first(U64(t_cur) * kN),
            "6e affine_nax_fwht64 q" + std::to_string(bits) + "_t" + std::to_string(t_cur));
        require(cs >= 0.9999, "6e linear_affine_nax_fwht64 cos_sim < 0.9999");
        ++checks;
      }
    }
    {
      auto mw = make_mxfp4_weight(ctx, wid++, kN, kK, 151u, true, 16u, false, true);
      Mxfp4NaxCache mc;
      mc.register_weight(mw.view);
      mc.build(ctx, libs);
      const auto *tw = mc.find(mw.view.object_id);
      const auto ref = cpu_linear_ref(x_f32, mw.ref_w, kMaxM, kN, kK);
      for (U32 t_cur : {4u, 128u}) {
        // W4A16 MXFP4 fwht64 (T=4 exercises bf16_fwht64_to_f16_rowscale; T=128 exercises wide fwht64_bf16)
        Encoder e16(ctx, libs, t_cur, "miv2.6e.mx16.t" + std::to_string(t_cur));
        e16.begin_stage("lin16");
        linear_mxfp4_nax(e16, *tw, x_ref, y_a_ref, f16_sc, f16_rs, std::nullopt,
                         1u, 0u, 0u, 0u, true);
        auto p16 = e16.finish();
        submit_profile(ctx, p16, t_cur);
        const double cs16 = cosine_similarity_bf16(
            read_u16(y_a_buf, U64(t_cur) * kN),
            std::span(ref).first(U64(t_cur) * kN),
            "6e mxfp4_w4a16_fwht64_t" + std::to_string(t_cur));
        require(cs16 >= 0.9998, "6e linear_mxfp4_nax fwht64 cos_sim < 0.9998");
        ++checks;

        // W4A4 MXFP4 fwht64 (exercises bf16_fwht64_to_mxfp4_a4_bm16_bk128)
        Encoder e4(ctx, libs, t_cur, "miv2.6e.mx4.t" + std::to_string(t_cur));
        e4.begin_stage("lin4");
        linear_mxfp4_w4a4_nax(e4, *tw, x_ref, y_a_ref, a4_c, a4_s, std::nullopt, 1u, true);
        auto p4 = e4.finish();
        submit_profile(ctx, p4, t_cur);
        const double cs4 = cosine_similarity_bf16(
            read_u16(y_a_buf, U64(t_cur) * kN),
            std::span(ref).first(U64(t_cur) * kN),
            "6e mxfp4_w4a4_fwht64_t" + std::to_string(t_cur));
        require(cs4 >= kMinCosW4A4, "6e linear_mxfp4_w4a4_nax fwht64 cos_sim < 0.985");
        ++checks;
      }
    }
  }

  // 6f. Decode T=1 & M=2 pretiled_qmv (row_exact.metal) with bit-30 lane_v2 & bit-31 fwht64
  //     PLUS qwen_forward.metal portable fallback pretiled_{mxfp4,affine}_linear with fwht64!
  {
    const RowExactPlaneSlice zero_slice{zero_ref.buffer, zero_ref.offset, zero_ref.bytes};
    // MXFP4 pretiled_qmv T=1 & M=2 with do_fwht=true
    {
      auto mw = make_mxfp4_weight(ctx, wid++, kN, kK, 161u, true, 16u, false, true);
      const auto pl = resolve_pretiled_kl32_mxfp4_planes(mw.view);
      require(pl.has_value() && pl->has_fwht64, "6f resolve_pretiled_kl32_mxfp4_planes failed");
      const auto ref2 = cpu_linear_ref(x_f32, mw.ref_w, 2u, kN, kK);
      const U32 k_flagged = kK | 0x80000000u;
      // T=1
      {
        Encoder e(ctx, libs, 1u, "miv2.6f.mx.t1");
        e.begin_stage("qmv_t1");
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_t1",
                         {.width = U64((kN + 7u) / 8u) * 64u}, {.width = 64u},
                         {x_ref, Ref{pl->codes.buffer, pl->codes.offset, pl->codes.bytes},
                          Ref{pl->scales.buffer, pl->scales.offset, pl->scales.bytes}, y_a_ref});
        e.dynamic(d, 4, words({kN, k_flagged, kK, 1u}));
        auto p = e.finish();
        submit_profile(ctx, p, 1u);
        const double cs = cosine_similarity_bf16(
            read_u16(y_a_buf, kN), std::span(ref2).first(kN), "6f pretiled_qmv_mx_t1");
        require(cs >= 0.9999, "6f pretiled_qmv_mxfp4 T=1 fwht64 failed");
        ++checks;
      }
      // M=2
      {
        Encoder e(ctx, libs, 2u, "miv2.6f.mx.m2");
        e.begin_stage("qmv_m2");
        auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_mxfp4_bn16_r4_m2",
                         {.width = U64((kN + 7u) / 8u) * 64u}, {.width = 64u},
                         {x_ref, Ref{pl->codes.buffer, pl->codes.offset, pl->codes.bytes},
                          Ref{pl->scales.buffer, pl->scales.offset, pl->scales.bytes}, y_a_ref});
        e.dynamic(d, 4, words({kN, k_flagged, kK, 2u}));
        auto p = e.finish();
        submit_profile(ctx, p, 2u);
        const double cs = cosine_similarity_bf16(
            read_u16(y_a_buf, 2ULL * kN), ref2, "6f pretiled_qmv_mx_m2");
        require(cs >= 0.9999, "6f pretiled_qmv_mxfp4 M=2 fwht64 failed");
        ++checks;
      }
      // Portable qwen_forward pretiled fallback (T=4)
      {
        Encoder e(ctx, libs, 4u, "miv2.6f.mx.port");
        e.begin_stage("port");
        linear(e, mw.view, x_ref, y_a_ref, kN, kK, xs_ref, zero_ref);
        auto p = e.finish();
        submit_profile(ctx, p, 4u);
        const auto ref4 = cpu_linear_ref(x_f32, mw.ref_w, 4u, kN, kK);
        const double cs = cosine_similarity_bf16(
            read_u16(y_a_buf, 4ULL * kN), ref4, "6f portable pretiled_mx");
        require(cs >= 0.9999, "6f portable pretiled_mxfp4_linear fwht64 failed");
        ++checks;
      }
    }

    // Affine pretiled_qmv T=1 & M=2 across q3/q4/q5/q6/q8 for BOTH lane_v2=true and lane_v2=false with do_fwht=true
    for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
      for (bool is_v2 : {false, true}) {
        auto aw = make_affine_weight(ctx, wid++, kN, kK, bits, 64u, bits * 71u + (is_v2 ? 1u : 0u),
                                     true, 16u, false, is_v2, true);
        const std::array<MaterializedWeightView, 1> cand{aw.view};
        const void *owner_tok =
            reinterpret_cast<const void *>(std::uintptr_t(0x900000u + aw.view.object_id));
        auto q5_cache = Q5High1Cache::acquire(owner_tok, ctx, libs, cand);
        const auto pl = resolve_pretiled_kl32_affine_planes(aw.view, q5_cache.get(), zero_slice);
        require(pl.has_value() && pl->has_fwht64 && pl->is_lane_v2 == is_v2,
                "6f resolve_pretiled_kl32_affine_planes failed");
        const auto ref2 = cpu_linear_ref(x_f32, aw.ref_w, 2u, kN, kK);
        const U32 k_flagged = kK | (is_v2 ? 0x40000000u : 0u) | 0x80000000u;
        const Ref c_ref{pl->codes.buffer, pl->codes.offset, pl->codes.bytes};
        const Ref h_ref{pl->high.buffer, pl->high.offset, pl->high.bytes};
        const Ref sb_ref{pl->sb_tiled.buffer, pl->sb_tiled.offset, pl->sb_tiled.bytes};
        // T=1
        {
          Encoder e(ctx, libs, 1u, "miv2.6f.aff.t1");
          e.begin_stage("qmv_t1");
          auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_affine_t1",
                           {.width = U64((kN + 3u) / 4u) * 64u}, {.width = 64u},
                           {x_ref, c_ref, h_ref, sb_ref, y_a_ref});
          e.dynamic(d, 5, words({kN, 1u, 0u, k_flagged, bits}));
          auto p = e.finish();
          submit_profile(ctx, p, 1u);
          const double cs = cosine_similarity_bf16(
              read_u16(y_a_buf, kN), std::span(ref2).first(kN),
              "6f pretiled_qmv_aff_t1_q" + std::to_string(bits));
          require(cs >= 0.9999, "6f pretiled_qmv_affine T=1 fwht64 failed");
          ++checks;
        }
        // M=2
        {
          Encoder e(ctx, libs, 2u, "miv2.6f.aff.m2");
          e.begin_stage("qmv_m2");
          auto &d = e.emit("row_exact", "omlx_m5_pretiled_qmv_affine_r2_m2",
                           {.width = U64((kN + 3u) / 4u) * 64u}, {.width = 64u},
                           {x_ref, c_ref, h_ref, sb_ref, y_a_ref});
          e.dynamic(d, 5, words({kN, 2u, 0u, k_flagged, bits}));
          auto p = e.finish();
          submit_profile(ctx, p, 2u);
          const double cs = cosine_similarity_bf16(
              read_u16(y_a_buf, 2ULL * kN), ref2,
              "6f pretiled_qmv_aff_m2_q" + std::to_string(bits));
          require(cs >= 0.9999, "6f pretiled_qmv_affine M=2 fwht64 failed");
          ++checks;
        }
        // Portable qwen_forward pretiled fallback (T=4)
        {
          Encoder e(ctx, libs, 4u, "miv2.6f.aff.port");
          e.begin_stage("port");
          linear(e, aw.view, x_ref, y_a_ref, kN, kK, xs_ref, zero_ref);
          auto p = e.finish();
          submit_profile(ctx, p, 4u);
          const auto ref4 = cpu_linear_ref(x_f32, aw.ref_w, 4u, kN, kK);
          const double cs = cosine_similarity_bf16(
              read_u16(y_a_buf, 4ULL * kN), ref4,
              "6f portable pretiled_aff_q" + std::to_string(bits));
          require(cs >= 0.9999, "6f portable pretiled_affine_linear fwht64 failed");
          ++checks;
        }
      }
    }
  }

  // 6g, 6h, 6i: MoE 100% W4A8 (Mode B -> NEW down_scatter_bm64_bn64_sk1),
  //             MoE W4A16 fwht64 (T=4 narrow + T=256 wide), and MoE9 T=1 & B=2 fwht64!
  {
    constexpr U32 kMaxT = 256u;
    constexpr U32 kH = 256u;
    constexpr U32 kF = 256u;
    constexpr U32 kExperts = 256u;
    constexpr U32 kTopK = 8u;

    std::vector<std::uint16_t> norm_u16;
    const auto norm_f32 = make_activations(kMaxT, kH, 1701u, norm_u16);
    auto norm_buf = ctx.upload(std::as_bytes(std::span(norm_u16)));
    std::vector<std::uint32_t> route_ids(U64(kMaxT) * kTopK);
    std::vector<std::uint16_t> route_scores(U64(kMaxT) * kTopK);
    for (U32 tok = 0; tok < kMaxT; ++tok) {
      for (U32 s = 0; s < kTopK; ++s) {
        route_ids[U64(tok) * kTopK + s] =
            (s < 3u) ? ((tok + s * 2u) % 6u)
                     : (6u + ((tok * 13u + s * 37u) % (kExperts - 6u)));
        route_scores[U64(tok) * kTopK + s] =
            float_to_bf16(0.05f + 0.03f * float((tok + s) % 7u));
      }
    }
    auto rids_buf = ctx.upload(std::as_bytes(std::span(route_ids)));
    auto rsc_buf = ctx.upload(std::as_bytes(std::span(route_scores)));
    auto exp_out_buf = ctx.make_buffer(U64(kMaxT) * kTopK * kH * 2ULL);
    auto routed_buf = ctx.make_buffer(U64(kMaxT) * kH * 2ULL);

    std::vector<std::unique_ptr<MetalBuffer>> ws_bufs;
    MoeNaxWorkspace ws;
    ws.allocate(ctx, kMaxT, kH, kF, ws_bufs, kExperts, kTopK);

    const auto a8_lay = w4a8_nax_workspace_layout(kMaxT, kH);
    auto m_qa = ctx.make_buffer(a8_lay.qa_bytes);
    auto m_sa = ctx.make_buffer(a8_lay.sa_bytes);
    auto m_ra = ctx.make_buffer(a8_lay.ra_bytes);
    Ref norm_ref{&norm_buf, 0, norm_buf.size()};
    Ref rids_ref{&rids_buf, 0, rids_buf.size()};
    Ref rsc_ref{&rsc_buf, 0, rsc_buf.size()};
    Ref exp_ref{&exp_out_buf, 0, exp_out_buf.size()};
    Ref routed_ref{&routed_buf, 0, routed_buf.size()};
    Ref m_qa_ref{&m_qa, 0, m_qa.size()};
    Ref m_sa_ref{&m_sa, 0, m_sa.size()};
    Ref m_ra_ref{&m_ra, 0, m_ra.size()};

    auto q_norm = [&](U32 live_t) {
      Encoder eq(ctx, libs, kMaxT, "miv2.6g.q");
      eq.begin_stage("q");
      emit_quantize_a8_g64(eq, norm_ref, m_qa_ref, m_sa_ref, m_ra_ref, kH);
      auto pq = eq.finish();
      submit_profile(ctx, pq, live_t);
    };

    // 6g + 6h MXFP4 with dw.has_fwht64() == true
    {
      auto mg = make_mxfp4_weight(ctx, wid++, kExperts * kF, kH, 211u, false, 0u, true, false);
      auto mu = make_mxfp4_weight(ctx, wid++, kExperts * kF, kH, 212u, false, 0u, true, false);
      auto md = make_mxfp4_weight(ctx, wid++, kExperts * kH, kF, 213u, false, 0u, true, true);

      // 6g: 100% W4A8 MoE MXFP4 (Mode B -> NEW down_scatter_bm64_bn64_sk1) at T=65, 256
      for (U32 live : {65u, 256u}) {
        q_norm(live);
        Encoder ew(ctx, libs, kMaxT, "miv2.6g.mx");
        ew.begin_stage("moe_w4a8_v2");
        require(emit_moe_w4a8_nax_ffn(ew, ws, mg.view, mu.view, md.view,
                                      m_qa_ref, m_sa_ref, m_ra_ref, a8_lay.mp,
                                      rids_ref, rsc_ref, exp_ref, routed_ref,
                                      kMaxT, kH, kF, kTopK, kExperts),
                "6g emit_moe_w4a8_nax_ffn MXFP4 fwht64 failed");
        auto pw = ew.finish();
        require(pw.weight_routes.back().function ==
                    "omlx_m5_moe_mxfp4_w4a8_nax_down_scatter_bm64_bn64_sk1",
                "6g MoE MXFP4 fwht64 did not select W4A8 down_scatter");
        submit_profile(ctx, pw, live);
        const auto ref = cpu_moe_ffn_ref(norm_f32, mg.ref_w, mu.ref_w, md.ref_w,
                                         route_ids, route_scores, live, kH, kF, kTopK);
        const double cs = cosine_similarity_bf16(
            read_u16(routed_buf, U64(live) * kH), ref, "6g moe_mx_w4a8_v2");
        min_w4a8_cos = std::min(min_w4a8_cos, cs);
        require(cs >= kMinCosW4A8,
                "6g MoE MXFP4 100% W4A8 cos_sim=" + std::to_string(cs));
        ++checks;
      }

      // 6h: MoE W4A16 MXFP4 with dw.has_fwht64() at T=4 (narrow) and T=256 (wide)
      for (U32 live : {4u, 256u}) {
        Encoder e16(ctx, libs, live, "miv2.6h.mx.t" + std::to_string(live));
        e16.begin_stage("moe_16");
        require(emit_moe_nax_ffn(e16, ws, mg.view, mu.view, md.view, norm_ref,
                                 rids_ref, rsc_ref, exp_ref, routed_ref, live,
                                 kH, kF, kTopK, kExperts, 64u, std::nullopt, {},
                                 true, true),
                "6h emit_moe_nax_ffn MXFP4 fwht64 failed");
        auto p16 = e16.finish();
        submit_profile(ctx, p16, live);
        const auto ref = cpu_moe_ffn_ref(norm_f32, mg.ref_w, mu.ref_w, md.ref_w,
                                         route_ids, route_scores, live, kH, kF, kTopK);
        const double cs = cosine_similarity_bf16(
            read_u16(routed_buf, U64(live) * kH), ref, "6h moe_mx_w4a16_fwht64");
        require(cs >= 0.9998, "6h MoE W4A16 MXFP4 fwht64 cos_sim < 0.9998");
        ++checks;
      }

      // 6i: MoE9 T=1 & B=2 with dw.has_fwht64() == true & shared_dw.has_fwht64() == true
      {
        auto sgw = make_mxfp4_weight(ctx, wid++, kF, kH, 221u, true, 16u, false, false);
        auto suw = make_mxfp4_weight(ctx, wid++, kF, kH, 222u, true, 16u, false, false);
        auto sdw = make_mxfp4_weight(ctx, wid++, kH, kF, 223u, true, 16u, false, true);
        Mxfp4NaxCache smc;
        smc.register_weight(sgw.view);
        smc.register_weight(suw.view);
        smc.register_weight(sdw.view);
        smc.build(ctx, libs);

        std::vector<std::uint16_t> sgate_u16 = {float_to_bf16(0.65f), float_to_bf16(-0.45f)};
        auto sgate_buf = ctx.upload(std::as_bytes(std::span(sgate_u16)));
        Ref sgate_ref{&sgate_buf, 0, sgate_buf.size()};

        // Reference MoE9 (routed + sigmoid(sgate) * shared_ffn) for T=2
        auto ref2 = cpu_moe_ffn_ref(norm_f32, mg.ref_w, mu.ref_w, md.ref_w,
                                    route_ids, route_scores, 2u, kH, kF, kTopK);
        const auto sh_mid = cpu_swiglu_ref(norm_f32, sgw.ref_w, suw.ref_w, 2u, kF, kH);
        const auto sh_dn = cpu_linear_ref(sh_mid, sdw.ref_w, 2u, kH, kF);
        for (U32 tok = 0; tok < 2u; ++tok) {
          const float g = 1.0f / (1.0f + std::exp(-bf16_val(sgate_u16[tok])));
          for (U32 c = 0; c < kH; ++c) {
            ref2[U64(tok) * kH + c] += g * sh_dn[U64(tok) * kH + c];
          }
        }
        // T=1
        {
          Encoder e1(ctx, libs, 1u, "miv2.6i.moe9.t1");
          e1.begin_stage("moe9_t1");
          require(emit_moe9_mxfp4_t1(e1, ws, mg.view, mu.view, md.view,
                                     sgw.view, suw.view, sdw.view, &smc,
                                     norm_ref, rids_ref, rsc_ref, sgate_ref,
                                     routed_ref, kH, kF),
                  "6i emit_moe9_mxfp4_t1 fwht64 failed");
          auto p1 = e1.finish();
          submit_profile(ctx, p1, 1u);
          const double cs = cosine_similarity_bf16(
              read_u16(routed_buf, kH), std::span(ref2).first(kH), "6i moe9_t1_fwht64");
          require(cs >= 0.9998, "6i MoE9 T=1 fwht64 cos_sim < 0.9998");
          ++checks;
        }
        // B=2
        {
          Encoder e2(ctx, libs, 2u, "miv2.6i.moe9.b2");
          e2.begin_stage("moe9_b2");
          require(emit_moe9_mxfp4_bt(e2, ws, mg.view, mu.view, md.view,
                                     sgw.view, suw.view, sdw.view, &smc,
                                     norm_ref, rids_ref, rsc_ref, sgate_ref,
                                     routed_ref, 2u, kH, kF),
                  "6i emit_moe9_mxfp4_bt fwht64 failed");
          auto p2 = e2.finish();
          submit_profile(ctx, p2, 2u);
          const double cs = cosine_similarity_bf16(
              read_u16(routed_buf, 2ULL * kH), ref2, "6i moe9_b2_fwht64");
          require(cs >= 0.9998, "6i MoE9 B=2 fwht64 cos_sim < 0.9998");
          ++checks;
        }
      }
    }

    // 6g + 6h Affine: ALL 10 (q3/q4/q5/q6/q8) x (g64/g128) with dw.has_fwht64() == true -> 100% W4A8!
    for (U32 bits : {3u, 4u, 5u, 6u, 8u}) {
      for (U32 gs : {64u, 128u}) {
        auto ag = make_affine_weight(ctx, wid++, kExperts * kF, kH, bits, gs, bits * 19u + gs, false, 0u, true, false, false);
        auto au = make_affine_weight(ctx, wid++, kExperts * kF, kH, bits, gs, bits * 23u + gs, false, 0u, true, false, false);
        auto ad = make_affine_weight(ctx, wid++, kExperts * kH, kF, bits, gs, bits * 29u + gs, false, 0u, true, false, true);

        for (U32 live : {65u, 256u}) {
          q_norm(live);
          Encoder ew(ctx, libs, kMaxT, "miv2.6g.aff");
          ew.begin_stage("moe_w4a8_v2");
          require(emit_moe_w4a8_nax_ffn(ew, ws, ag.view, au.view, ad.view,
                                        m_qa_ref, m_sa_ref, m_ra_ref, a8_lay.mp,
                                        rids_ref, rsc_ref, exp_ref, routed_ref,
                                        kMaxT, kH, kF, kTopK, kExperts),
                  "6g emit_moe_w4a8_nax_ffn Affine fwht64 failed");
          auto pw = ew.finish();
          require(pw.weight_routes.back().function.find("_w4a8_nax_down_scatter_") !=
                      std::string::npos,
                  "6g MoE Affine fwht64 did not select W4A8 down_scatter");
          submit_profile(ctx, pw, live);
          const auto ref = cpu_moe_ffn_ref(norm_f32, ag.ref_w, au.ref_w, ad.ref_w,
                                           route_ids, route_scores, live, kH, kF, kTopK);
          const double cs = cosine_similarity_bf16(
              read_u16(routed_buf, U64(live) * kH), ref,
              "6g moe_aff_w4a8_v2_q" + std::to_string(bits) + "_g" + std::to_string(gs));
          min_w4a8_cos = std::min(min_w4a8_cos, cs);
          require(cs >= kMinCosW4A8,
                  "6g MoE Affine 100% W4A8 q" + std::to_string(bits) + "_g" +
                      std::to_string(gs) + " cos_sim=" + std::to_string(cs));
          ++checks;
        }

        // Also test MoE W4A16 fwht64 narrow (T=4) & wide (T=256) on q4_g64
        if (bits == 4u && gs == 64u) {
          for (U32 live : {4u, 256u}) {
            Encoder e16(ctx, libs, live, "miv2.6h.aff.t" + std::to_string(live));
            e16.begin_stage("moe_16");
            require(emit_moe_nax_ffn(e16, ws, ag.view, au.view, ad.view, norm_ref,
                                     rids_ref, rsc_ref, exp_ref, routed_ref, live,
                                     kH, kF, kTopK, kExperts, 64u, std::nullopt, {},
                                     true, true),
                    "6h emit_moe_nax_ffn Affine fwht64 failed");
            auto p16 = e16.finish();
            submit_profile(ctx, p16, live);
            const auto ref = cpu_moe_ffn_ref(norm_f32, ag.ref_w, au.ref_w, ad.ref_w,
                                             route_ids, route_scores, live, kH, kF, kTopK);
            const double cs = cosine_similarity_bf16(
                read_u16(routed_buf, U64(live) * kH), ref, "6h moe_aff_w4a16_fwht64");
            require(cs >= 0.9998, "6h MoE W4A16 Affine fwht64 cos_sim < 0.9998");
            ++checks;
          }
        }
      }
    }
  }

  // 6j: LoRA Mode A ("merge") on .miv v2 (oq_nax_lane_g64_v2 + fwht64):
  //     W'_eff = W' + scale * B @ (A @ H_64) written as q8 oq_nax_lane_g64_v2 + fwht64.
  {
    constexpr U32 kLN = 128u;
    constexpr U32 kLK = 512u;
    constexpr U32 kLR = 8u;
    constexpr U32 kLT = 64u;
    constexpr float kLScale = 0.25f;
    const U32 bn = 16u;
    const U32 padded_n = ((kLN + bn - 1u) / bn) * bn;
    const U32 groups = kLK / 64u;

    auto w_base = make_affine_weight(ctx, wid++, kLN, kLK, 4u, 64u, 777u, true,
                                     bn, false, true, true);

    std::vector<float> lora_a(U64(kLR) * kLK);
    std::vector<float> lora_b(U64(kLN) * kLR);
    for (U64 i = 0; i < lora_a.size(); ++i) {
      lora_a[i] = std::sin(float(i + 1u) * 0.13f) * 0.35f;
    }
    for (U64 i = 0; i < lora_b.size(); ++i) {
      lora_b[i] = std::cos(float(i + 3u) * 0.17f) * 0.35f;
    }
    // Rotate A -> A_rot = A @ H_64 along each 64-element K block (as lora_runtime.h does)
    std::vector<float> lora_a_rot = lora_a;
    fwht64_rows_cpu(lora_a_rot, kLR, kLK);

    auto a_buf = ctx.upload(std::as_bytes(std::span(lora_a_rot)));
    auto b_buf = ctx.upload(std::as_bytes(std::span(lora_b)));
    auto dst_codes = ctx.make_private_buffer(U64(padded_n) * kLK);
    auto dst_sb = ctx.make_private_buffer(U64(groups) * padded_n * 4ULL);
    auto dst_sb_tiled = ctx.make_private_buffer(U64(padded_n) * groups * 4ULL);

    Ref p0 = plane(w_base.view, "nax_codes");
    Ref p2 = plane(w_base.view, "nax_sb");
    const U32 scale_bits = std::bit_cast<U32>(kLScale);
    {
      Encoder em(ctx, libs, 1u, "miv2.6j.lora_merge");
      em.begin_stage("lora_merge");
      auto &d = em.emit(
          "lora", "omlx_m5_lora_merge_to_q8_g64",
          {.width = U64(groups) * 32u, .height = padded_n}, {.width = 32u},
          {p0, zero_ref, p2, zero_ref, Ref{&a_buf, 0, a_buf.size()},
           Ref{&b_buf, 0, b_buf.size()}, Ref{&dst_codes, 0, dst_codes.size()},
           Ref{&dst_sb, 0, dst_sb.size()},
           Ref{&dst_sb_tiled, 0, dst_sb_tiled.size()}});
      em.dynamic(d, 9,
                 words({kLN, kLK, kLK, padded_n, bn, 5u, 4u, 64u, kLR,
                        scale_bits}));
      auto pm = em.finish();
      submit_profile(ctx, pm, 1u);
    }

    MaterializedWeightView w_merged{
        .object_id = wid++,
        .component = w_base.view.component,
        .name = "miv2.6j.merged_q8_lane_v2",
        .format = "oq_affine_q8_g64_bf16_v1",
        .layout = "oq_nax_lane_g64_v2",
        .parent_shape = {kLN, kLK},
        .padded_k = kLK,
        .row_start = 0,
        .row_count = kLN,
        .in_features = kLK,
        .planes =
            {
                MaterializedPlaneView{"nax_codes", &dst_codes, 0, 0, dst_codes.size(), 0},
                MaterializedPlaneView{"nax_sb", &dst_sb, 0, 0, dst_sb.size(), 0},
                MaterializedPlaneView{"nax_sb_tiled", &dst_sb_tiled, 0, 0, dst_sb_tiled.size(), 0},
            },
        .input_transform = "fwht64",
    };

    // Exact FP32 logical reference W_eff = W_base + scale * (B @ A)
    std::vector<float> w_eff = w_base.ref_w;
    for (U32 n = 0; n < kLN; ++n) {
      for (U32 k = 0; k < kLK; ++k) {
        float delta = 0.0f;
        for (U32 r = 0; r < kLR; ++r) {
          delta += lora_b[U64(n) * kLR + r] * lora_a[U64(r) * kLK + k];
        }
        w_eff[U64(n) * kLK + k] += kLScale * delta;
      }
    }

    std::vector<std::uint16_t> lx_u16;
    const auto lx_f32 = make_activations(kLT, kLK, 888u, lx_u16);
    auto lx_buf = ctx.upload(std::as_bytes(std::span(lx_u16)));
    Ref lx_ref{&lx_buf, 0, lx_buf.size()};
    auto ly_buf = ctx.make_buffer(U64(kLT) * kLN * 2ULL);
    Ref ly_ref{&ly_buf, 0, ly_buf.size()};

    AffineNaxCache lac;
    lac.register_weight(w_merged, zero_ref);
    lac.build(ctx, libs, zero_ref);
    const auto *tw_m = lac.find(w_merged.object_id);
    require(tw_m != nullptr && tw_m->bits == 8u && tw_m->codes.buffer == &dst_codes,
            "6j merged LoRA weight missing zero-copy lane_v2 registration");

    // Verify via W4A8 + FWHT-64 activation quantization at T=64
    const auto llay = w4a8_nax_workspace_layout(kLT, kLK);
    auto lqa = ctx.make_buffer(llay.qa_bytes);
    auto lsa = ctx.make_buffer(llay.sa_bytes);
    auto lra = ctx.make_buffer(llay.ra_bytes);
    Ref lqa_ref{&lqa, 0, lqa.size()};
    Ref lsa_ref{&lsa, 0, lsa.size()};
    Ref lra_ref{&lra, 0, lra.size()};
    {
      Encoder ew(ctx, libs, kLT, "miv2.6j.w4a8");
      ew.begin_stage("lora_w4a8");
      emit_quantize_a8_g64(ew, lx_ref, lqa_ref, lsa_ref, lra_ref, kLK,
                           kLT, kLT, true);
      linear_affine_w4a8_nax(ew, *tw_m, lqa_ref, lsa_ref, lra_ref, ly_ref, kLT);
      auto pw = ew.finish();
      submit_profile(ctx, pw, kLT);
      const auto ref = cpu_linear_ref(lx_f32, w_eff, kLT, kLN, kLK);
      const double cs = cosine_similarity_bf16(read_u16(ly_buf, U64(kLT) * kLN),
                                               ref, "6j lora_merge_v2_w4a8");
      min_w4a8_cos = std::min(min_w4a8_cos, cs);
      require(cs >= kMinCosW4A8,
              "6j LoRA merge v2 W4A8 cos_sim=" + std::to_string(cs));
      ++checks;
    }
    // Verify via T=1 pretiled_qmv in-register FWHT-64 + lane_v2
    {
      Encoder eq(ctx, libs, 1u, "miv2.6j.qmv_t1");
      eq.begin_stage("lora_qmv");
      const U32 k_flags = kLK | (1u << 30u) | (1u << 31u);
      auto &d = eq.emit("row_exact", "omlx_m5_pretiled_qmv_affine_t1",
                        {.width = U64((kLN + 3u) / 4u) * 64u}, {.width = 64u},
                        {lx_ref, Ref{&dst_codes, 0, dst_codes.size()}, zero_ref,
                         Ref{&dst_sb_tiled, 0, dst_sb_tiled.size()}, ly_ref});
      eq.dynamic(d, 5, words({kLN, 1u, 0u, k_flags, 8u}));
      auto pq = eq.finish();
      submit_profile(ctx, pq, 1u);
      const auto ref = cpu_linear_ref(lx_f32, w_eff, 1u, kLN, kLK);
      const double cs = cosine_similarity_bf16(read_u16(ly_buf, kLN), ref,
                                               "6j lora_merge_v2_qmv_t1");
      require(cs >= 0.9999,
              "6j LoRA merge v2 pretiled_qmv T=1 cos_sim=" + std::to_string(cs));
      ++checks;
    }
  }

  return {{"checks", checks}, {"min_w4a8_cos_sim", min_w4a8_cos}, {"passed", true}};
}

// ============================================================================
// Test 7 (WP 3B): oQ4e Native int8 x int4b Bit-Exact Parity & Dual-Channel
// Verifies 0-ULP / bit-exact equivalence between default 2x 32x32x32 int8 x int8
// and OMLX_M5_OQ4_INT4B=1 native int8_t x int4b_format across:
//   - Linear (bns16 & bns32, g64 & g128, M=65 non-mult64, N=96, row_offset slicing, icb=0/1)
//   - SwiGLU Mode A & Mode B (+ FWHT-64 + GS64 A8, also_write_bf16=true/false, row_offset)
//   - Mixed-layer q4 + q5 sequence sharing identical (qa, sa, ra) buffers.
// ============================================================================
Json test_wp3b_oq4e_dual_channel_and_int4b_exact(
    const MetalContext &ctx, const std::filesystem::path &libs) {
  constexpr U32 kMaxM = 128u;
  constexpr U32 kLiveM = 65u;
  constexpr U32 kK = 512u;
  constexpr U32 kNFull = 256u;
  constexpr U32 kNOdd = 96u;
  constexpr double kMinCos = 0.9985;
  U32 checks = 0;
  U32 wid = 9000u;

  std::vector<std::uint16_t> x_u16;
  const auto x_f32 = make_activations(kMaxM, kK, 911u, x_u16);
  auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
  auto zero_buf = ctx.make_buffer(256);
  zero_buf.write(std::vector<std::byte>(256));
  Ref x_ref{&x_buf, 0, x_buf.size()};
  Ref zero_ref{&zero_buf, 0, zero_buf.size()};

  const auto a8_lay = w4a8_nax_workspace_layout(kMaxM, kK);
  auto qa_buf = ctx.make_buffer(a8_lay.qa_bytes);
  auto sa_buf = ctx.make_buffer(a8_lay.sa_bytes);
  auto ra_buf = ctx.make_buffer(a8_lay.ra_bytes);
  Ref qa_ref{&qa_buf, 0, qa_buf.size()};
  Ref sa_ref{&sa_buf, 0, sa_buf.size()};
  Ref ra_ref{&ra_buf, 0, ra_buf.size()};

  {
    Encoder eq(ctx, libs, kMaxM, "wp3b.quant_x");
    eq.begin_stage("quant");
    emit_quantize_a8_g64(eq, x_ref, qa_ref, sa_ref, ra_ref, kK);
    auto pq = eq.finish();
    submit_profile(ctx, pq, kLiveM, false);
  }

  auto y_def = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
  auto y_i4b = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);

  // 7a. Linear q4: bns16 & bns32, g64 & g128, full + odd N=96 + Dual-ANE row_offset slicing, icb=0/1
  for (U32 gs : {64u, 128u}) {
    for (U32 bn : {16u, 32u}) {
      for (U32 n_dim : {kNFull, kNOdd}) {
        auto wq4 = make_affine_weight(ctx, wid++, n_dim, kK, 4u, gs,
                                      9100u + gs + bn + n_dim, true, bn, false, true);
        AffineNaxCache ac;
        TiledAffineWeight tw_direct{};
        const TiledAffineWeight *tw = nullptr;
        if (bn == 16u) {
          ac.register_weight(wq4.view, zero_ref);
          ac.build(ctx, libs, zero_ref);
          tw = ac.find(wq4.view.object_id);
        } else {
          // Construct bns32 TiledAffineWeight directly from pre-tiled bn=32 planes.
          const U32 padded_n = ((n_dim + bn - 1u) / bn) * bn;
          tw_direct.codes = plane(wq4.view, "nax_codes");
          tw_direct.high2 = zero_ref;
          tw_direct.sb = plane(wq4.view, "nax_sb");
          tw_direct.sb_tiled = zero_ref;
          tw_direct.n = n_dim;
          tw_direct.k = kK;
          tw_direct.padded_n = padded_n;
          tw_direct.bn = bn;
          tw_direct.bits = 4u;
          tw_direct.group_size = gs;
          tw = &tw_direct;
        }
        require(tw != nullptr && tw->bits == 4u && tw->bn == bn,
                "wp3b 7a missing tiled q4 weight");

        auto run_lin = [&](bool enable_i4b, MetalBuffer &dst, bool use_icb,
                           U32 roff, U32 stride, U32 sn) {
          ScopedEnv env("OMLX_M5_OQ4_INT4B", enable_i4b ? "1" : "0");
          fill_sentinel_bf16(dst, U64(kMaxM) * stride);
          Encoder e(ctx, libs, kMaxM, "wp3b.lin");
          e.begin_stage("lin");
          linear_affine_w4a8_nax(e, *tw, qa_ref, sa_ref, ra_ref,
                                 Ref{&dst, 0, U64(kMaxM) * stride * 2ULL},
                                 {}, roff, stride, sn);
          e.record_weight(wq4.view, n_dim, kK, "w4a8", "test", "");
          auto p = e.finish(use_icb);
          const std::string &fn = p.weight_routes.back().function;
          const bool has_i4b = fn.find("q4_int4b") != std::string::npos;
          require(has_i4b == enable_i4b,
                  "wp3b 7a unexpected dispatch routing: " + fn);
          submit_profile(ctx, p, kLiveM, use_icb);
        };

        // Full matrix (icb=false and icb=true)
        for (bool use_icb : {false, true}) {
          run_lin(false, y_def, use_icb, 0u, n_dim, 0u);
          run_lin(true, y_i4b, use_icb, 0u, n_dim, 0u);
          const auto got_def = read_u16(y_def, U64(kLiveM) * n_dim);
          const auto got_i4b = read_u16(y_i4b, U64(kLiveM) * n_dim);
          require(got_def == got_i4b,
                  "wp3b 7a q4 linear int4b != default i8xi8 (must be 0-diff bit-exact!)");
          check_tail_sentinel(y_def, U64(kLiveM) * n_dim, U64(kMaxM) * n_dim, "wp3b 7a def");
          check_tail_sentinel(y_i4b, U64(kLiveM) * n_dim, U64(kMaxM) * n_dim, "wp3b 7a i4b");
          const auto ref = cpu_linear_ref(std::span(x_f32).first(U64(kLiveM) * kK),
                                          wq4.ref_w, kLiveM, n_dim, kK);
          const double cs = cosine_similarity_bf16(got_i4b, ref, "wp3b 7a q4_int4b");
          require(cs >= kMinCos, "wp3b 7a q4_int4b cos_sim=" + std::to_string(cs));
          ++checks;
        }

        // Dual-ANE row_offset + out_stride + slice_n slicing check on N=256
        if (n_dim == kNFull) {
          run_lin(false, y_def, true, 64u, kNFull, 128u);
          run_lin(true, y_i4b, true, 64u, kNFull, 128u);
          const auto got_def = read_u16(y_def, U64(kMaxM) * kNFull);
          const auto got_i4b = read_u16(y_i4b, U64(kMaxM) * kNFull);
          require(got_def == got_i4b,
                  "wp3b 7a q4 sliced linear int4b != default i8xi8");
          for (U32 r = 0u; r < kLiveM; ++r) {
            for (U32 c = 0u; c < 64u; ++c) {
              require(got_i4b[U64(r) * kNFull + c] == 0xA5A5u,
                      "wp3b 7a slice clobbered prefix");
            }
            for (U32 c = 192u; c < kNFull; ++c) {
              require(got_i4b[U64(r) * kNFull + c] == 0xA5A5u,
                      "wp3b 7a slice clobbered suffix");
            }
          }
          ++checks;
        }
      }
    }
  }

  // 7b. SwiGLU Mode A & Mode B (+ FWHT-64 + GS64 A8): g64 & g128, full + odd N=96 + row_offset
  const auto mid_lay = w4a8_nax_workspace_layout(kMaxM, kNFull);
  auto qao_def = ctx.make_buffer(mid_lay.qa_bytes);
  auto sao_def = ctx.make_buffer(mid_lay.sa_bytes);
  auto rao_def = ctx.make_buffer(mid_lay.ra_bytes);
  auto qao_i4b = ctx.make_buffer(mid_lay.qa_bytes);
  auto sao_i4b = ctx.make_buffer(mid_lay.sa_bytes);
  auto rao_i4b = ctx.make_buffer(mid_lay.ra_bytes);

  for (U32 gs : {64u, 128u}) {
    // Mode A on N=96 (non-multiple of 64) & N=256 (with row_offset slice)
    for (U32 n_dim : {kNOdd, kNFull}) {
      auto wg = make_affine_weight(ctx, wid++, n_dim, kK, 4u, gs, 9200u + gs + n_dim, true, 16u, false, true);
      auto wu = make_affine_weight(ctx, wid++, n_dim, kK, 4u, gs, 9300u + gs + n_dim, true, 16u, false, true);
      AffineNaxCache ac;
      ac.register_weight(wg.view, zero_ref);
      ac.register_weight(wu.view, zero_ref);
      ac.build(ctx, libs, zero_ref);
      const auto *twg = ac.find(wg.view.object_id);
      const auto *twu = ac.find(wu.view.object_id);

      auto run_swi_a = [&](bool enable_i4b, MetalBuffer &dst, U32 roff, U32 stride, U32 sn, bool use_icb) {
        ScopedEnv env("OMLX_M5_OQ4_INT4B", enable_i4b ? "1" : "0");
        fill_sentinel_bf16(dst, U64(kMaxM) * stride);
        Encoder e(ctx, libs, kMaxM, "wp3b.swiA");
        e.begin_stage("swiA");
        linear_affine_w4a8_nax_swiglu(e, *twg, *twu, qa_ref, sa_ref, ra_ref,
                                      Ref{&dst, 0, U64(kMaxM) * stride * 2ULL},
                                      {}, roff, stride, sn);
        e.record_weight(wg.view, n_dim, kK, "w4a8", "test", "");
        auto p = e.finish(use_icb);
        const std::string &fn = p.weight_routes.back().function;
        require((fn.find("q4_int4b") != std::string::npos) == enable_i4b,
                "wp3b 7b ModeA unexpected routing: " + fn);
        submit_profile(ctx, p, kLiveM, use_icb);
      };

      run_swi_a(false, y_def, 0u, n_dim, 0u, true);
      run_swi_a(true, y_i4b, 0u, n_dim, 0u, true);
      const auto got_def = read_u16(y_def, U64(kLiveM) * n_dim);
      const auto got_i4b = read_u16(y_i4b, U64(kLiveM) * n_dim);
      require(got_def == got_i4b, "wp3b 7b SwiGLU ModeA int4b != default i8xi8");
      const auto ref = cpu_swiglu_ref(std::span(x_f32).first(U64(kLiveM) * kK),
                                      wg.ref_w, wu.ref_w, kLiveM, n_dim, kK);
      require(cosine_similarity_bf16(got_i4b, ref, "wp3b 7b swiA") >= kMinCos,
              "wp3b 7b SwiGLU ModeA cos_sim too low");
      ++checks;

      if (n_dim == kNFull) {
        run_swi_a(false, y_def, 64u, kNFull, 128u, false);
        run_swi_a(true, y_i4b, 64u, kNFull, 128u, false);
        require(read_u16(y_def, U64(kMaxM) * kNFull) ==
                    read_u16(y_i4b, U64(kMaxM) * kNFull),
                "wp3b 7b SwiGLU ModeA slice int4b != default i8xi8");
        ++checks;

        // Mode B (+ FWHT-64 + GS64 A8): full + row_offset slice, also_write_bf16=true/false
        for (bool also_bf16 : {false, true}) {
          for (bool sliced : {false, true}) {
            const U32 roff = sliced ? 64u : 0u;
            const U32 sn = sliced ? 128u : 0u;
            auto run_swi_b = [&](bool enable_i4b, MetalBuffer &qao,
                                 MetalBuffer &sao, MetalBuffer &rao,
                                 MetalBuffer &yb) {
              ScopedEnv env("OMLX_M5_OQ4_INT4B", enable_i4b ? "1" : "0");
              qao.write(std::vector<std::byte>(qao.size(), std::byte{0x5A}));
              sao.write(std::vector<std::byte>(sao.size(), std::byte{0x5A}));
              rao.write(std::vector<std::byte>(rao.size(), std::byte{0x5A}));
              fill_sentinel_bf16(yb, U64(kMaxM) * kNFull);
              Encoder e(ctx, libs, kMaxM, "wp3b.swiB");
              e.begin_stage("swiB");
              linear_affine_w4a8_nax_swiglu_fwht64_a8(
                  e, *twg, *twu, qa_ref, sa_ref, ra_ref,
                  Ref{&qao, 0, qao.size()}, Ref{&sao, 0, sao.size()},
                  Ref{&rao, 0, rao.size()}, Ref{&yb, 0, yb.size()},
                  also_bf16, {}, roff, kNFull, sn);
              e.record_weight(wg.view, kNFull, kK, "w4a8", "test", "");
              auto p = e.finish(true);
              const std::string &fn = p.weight_routes.back().function;
              require((fn.find("q4_int4b") != std::string::npos) == enable_i4b,
                      "wp3b 7b ModeB unexpected routing: " + fn);
              submit_profile(ctx, p, kLiveM, true);
            };
            run_swi_b(false, qao_def, sao_def, rao_def, y_def);
            run_swi_b(true, qao_i4b, sao_i4b, rao_i4b, y_i4b);
            require(read_i8(qao_def, mid_lay.qa_bytes) ==
                        read_i8(qao_i4b, mid_lay.qa_bytes),
                    "wp3b 7b ModeB qa_out int4b != default");
            require(read_f32(sao_def, mid_lay.sa_bytes / 4ULL) ==
                        read_f32(sao_i4b, mid_lay.sa_bytes / 4ULL),
                    "wp3b 7b ModeB sa_out int4b != default");
            require(read_f32(rao_def, mid_lay.ra_bytes / 4ULL) ==
                        read_f32(rao_i4b, mid_lay.ra_bytes / 4ULL),
                    "wp3b 7b ModeB ra_out int4b != default");
            if (also_bf16) {
              require(read_u16(y_def, U64(kMaxM) * kNFull) ==
                          read_u16(y_i4b, U64(kMaxM) * kNFull),
                      "wp3b 7b ModeB opt_y_bf16 int4b != default");
            }
            // Verify padded rows [kLiveM, mp) in written groups are strictly zero!
            const auto qa_got = read_i8(qao_i4b, mid_lay.qa_bytes);
            const auto sa_got = read_f32(sao_i4b, mid_lay.sa_bytes / 4ULL);
            const auto ra_got = read_f32(rao_i4b, mid_lay.ra_bytes / 4ULL);
            const U32 g_start = roff / 64u;
            const U32 g_count = (sn > 0u ? sn : kNFull) / 64u;
            for (U32 r = kLiveM; r < mid_lay.mp; ++r) {
              for (U32 g = g_start; g < g_start + g_count; ++g) {
                require(sa_got[U64(g) * mid_lay.mp + r] == 0.0f &&
                            ra_got[U64(g) * mid_lay.mp + r] == 0.0f,
                        "wp3b 7b ModeB padded sa/ra non-zero");
                for (U32 j = 0u; j < 64u; ++j) {
                  require(qa_got[U64(r) * kNFull + g * 64u + j] == 0,
                          "wp3b 7b ModeB padded qa non-zero");
                }
              }
            }
            ++checks;
          }
        }
      }
    }
  }

  // 7c. Seamless Dual-Channel Mixed-Layer Sequence (q4 + q5 sharing identical a8 buffers):
  //   (1) q4 QKVZ Linear -> q4_int4b
  //   (2) q5 O-proj Linear -> q56 (i8xi8)
  //   (3) (q4, q5) mixed SwiGLU Mode A -> q4_q56 (i8xi8)
  //   (4) (q4, q4) Gate+Up SwiGLU Mode B (q4_int4b) -> q5 FWHT-64 Down Linear (q56 i8xi8)
  {
    constexpr U32 kNMid = 512u;
    auto w_q4_lin = make_affine_weight(ctx, wid++, kNFull, kK, 4u, 64u, 9401u, true, 16u, false, true);
    auto w_q5_lin = make_affine_weight(ctx, wid++, kNFull, kK, 5u, 64u, 9402u, true, 16u, false, true);
    auto w_q4_g   = make_affine_weight(ctx, wid++, kNMid, kK, 4u, 64u, 9403u, true, 16u, false, true);
    auto w_q4_u   = make_affine_weight(ctx, wid++, kNMid, kK, 4u, 64u, 9404u, true, 16u, false, true);
    auto w_q5_u   = make_affine_weight(ctx, wid++, kNMid, kK, 5u, 64u, 9405u, true, 16u, false, true);
    auto w_q5_dn  = make_affine_weight(ctx, wid++, kK, kNMid, 5u, 64u, 9406u, true, 16u, false, true, true);
    AffineNaxCache ac;
    for (const auto *w : {&w_q4_lin, &w_q5_lin, &w_q4_g, &w_q4_u, &w_q5_u, &w_q5_dn}) {
      ac.register_weight(w->view, zero_ref);
    }
    ac.build(ctx, libs, zero_ref);
    const auto *tw_q4_lin = ac.find(w_q4_lin.view.object_id);
    const auto *tw_q5_lin = ac.find(w_q5_lin.view.object_id);
    const auto *tw_q4_g   = ac.find(w_q4_g.view.object_id);
    const auto *tw_q4_u   = ac.find(w_q4_u.view.object_id);
    const auto *tw_q5_u   = ac.find(w_q5_u.view.object_id);
    const auto *tw_q5_dn  = ac.find(w_q5_dn.view.object_id);
    require(tw_q4_lin && tw_q5_lin && tw_q4_g && tw_q4_u && tw_q5_u && tw_q5_dn,
            "wp3b 7c missing tiled weight");

    const auto mid512_lay = w4a8_nax_workspace_layout(kMaxM, kNMid);
    auto qao_512 = ctx.make_buffer(mid512_lay.qa_bytes);
    auto sao_512 = ctx.make_buffer(mid512_lay.sa_bytes);
    auto rao_512 = ctx.make_buffer(mid512_lay.ra_bytes);
    auto y512_tmp = ctx.make_buffer(U64(kMaxM) * kNMid * 2ULL);

    auto o1_def = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
    auto o2_def = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
    auto o3_def = ctx.make_buffer(U64(kMaxM) * kNMid * 2ULL);
    auto o4_def = ctx.make_buffer(U64(kMaxM) * kK * 2ULL);
    auto o1_i4b = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
    auto o2_i4b = ctx.make_buffer(U64(kMaxM) * kNFull * 2ULL);
    auto o3_i4b = ctx.make_buffer(U64(kMaxM) * kNMid * 2ULL);
    auto o4_i4b = ctx.make_buffer(U64(kMaxM) * kK * 2ULL);

    auto run_seq = [&](bool enable_i4b, MetalBuffer &o1, MetalBuffer &o2,
                       MetalBuffer &o3, MetalBuffer &o4) {
      ScopedEnv env("OMLX_M5_OQ4_INT4B", enable_i4b ? "1" : "0");
      Encoder e(ctx, libs, kMaxM, "wp3b.dual_seq");
      e.begin_stage("dual_seq");
      linear_affine_w4a8_nax(e, *tw_q4_lin, qa_ref, sa_ref, ra_ref, Ref{&o1, 0, o1.size()});
      e.record_weight(w_q4_lin.view, kNFull, kK, "w4a8", "test", "");
      linear_affine_w4a8_nax(e, *tw_q5_lin, qa_ref, sa_ref, ra_ref, Ref{&o2, 0, o2.size()});
      e.record_weight(w_q5_lin.view, kNFull, kK, "w4a8", "test", "");
      linear_affine_w4a8_nax_swiglu(e, *tw_q4_g, *tw_q5_u, qa_ref, sa_ref, ra_ref, Ref{&o3, 0, o3.size()});
      e.record_weight(w_q4_g.view, kNMid, kK, "w4a8", "test", "");
      linear_affine_w4a8_nax_swiglu_fwht64_a8(
          e, *tw_q4_g, *tw_q4_u, qa_ref, sa_ref, ra_ref,
          Ref{&qao_512, 0, qao_512.size()}, Ref{&sao_512, 0, sao_512.size()},
          Ref{&rao_512, 0, rao_512.size()}, Ref{&y512_tmp, 0, y512_tmp.size()}, false);
      e.record_weight(w_q4_g.view, kNMid, kK, "w4a8", "test", "");
      linear_affine_w4a8_nax(e, *tw_q5_dn,
                             Ref{&qao_512, 0, qao_512.size()},
                             Ref{&sao_512, 0, sao_512.size()},
                             Ref{&rao_512, 0, rao_512.size()},
                             Ref{&o4, 0, o4.size()});
      e.record_weight(w_q5_dn.view, kK, kNMid, "w4a8", "test", "");
      auto p = e.finish(true);
      require(p.weight_routes.size() == 5u, "wp3b 7c expected 5 routes");
      require((p.weight_routes[0].function.find("q4_int4b") != std::string::npos) == enable_i4b,
              "wp3b 7c dispatch[0] q4 routing mismatch: " + p.weight_routes[0].function);
      require(p.weight_routes[1].function.find("_q56_") != std::string::npos,
              "wp3b 7c dispatch[1] q56 routing mismatch: " + p.weight_routes[1].function);
      require(p.weight_routes[2].function.find("_q4_q56_") != std::string::npos,
              "wp3b 7c dispatch[2] mixed q4_q56 routing mismatch: " + p.weight_routes[2].function);
      require((p.weight_routes[3].function.find("q4_int4b") != std::string::npos) == enable_i4b,
              "wp3b 7c dispatch[3] q4 SwiGLU ModeB routing mismatch: " + p.weight_routes[3].function);
      require(p.weight_routes[4].function.find("_q56_") != std::string::npos,
              "wp3b 7c dispatch[4] q56 Down routing mismatch: " + p.weight_routes[4].function);
      submit_profile(ctx, p, kLiveM, true);
    };

    run_seq(false, o1_def, o2_def, o3_def, o4_def);
    run_seq(true, o1_i4b, o2_i4b, o3_i4b, o4_i4b);
    require(read_u16(o1_def, U64(kLiveM) * kNFull) == read_u16(o1_i4b, U64(kLiveM) * kNFull),
            "wp3b 7c q4 QKVZ dual-channel diff");
    require(read_u16(o2_def, U64(kLiveM) * kNFull) == read_u16(o2_i4b, U64(kLiveM) * kNFull),
            "wp3b 7c q5 O-proj dual-channel diff");
    require(read_u16(o3_def, U64(kLiveM) * kNMid) == read_u16(o3_i4b, U64(kLiveM) * kNMid),
            "wp3b 7c mixed (q4,q5) SwiGLU dual-channel diff");
    require(read_u16(o4_def, U64(kLiveM) * kK) == read_u16(o4_i4b, U64(kLiveM) * kK),
            "wp3b 7c q4 SwiGLU ModeB -> q5 Down dual-channel diff");
    const auto swi_ref = cpu_swiglu_ref(std::span(x_f32).first(U64(kLiveM) * kK),
                                        w_q4_g.ref_w, w_q4_u.ref_w, kLiveM, kNMid, kK);
    const auto dn_ref = cpu_linear_ref(swi_ref, w_q5_dn.ref_w, kLiveM, kK, kNMid);
    const double cs_dn = cosine_similarity_bf16(read_u16(o4_i4b, U64(kLiveM) * kK),
                                                dn_ref, "wp3b 7c q4_swiB_to_q5_dn");
    require(cs_dn >= kMinCos, "wp3b 7c q4->q5 pipeline cos_sim=" + std::to_string(cs_dn));
    ++checks;
  }

  return {{"checks", checks}, {"bit_exact_diffs", 0}, {"passed", true}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "usage: omlx_m5_w4a8_nax_tests <metallib-dir>\n";
      return 2;
    }
    const std::filesystem::path libs(argv[1]);
    auto ctx = MetalContext::create();
    const auto &probe = ctx.capabilities().profile.probe;
    if (probe.runtime_os_major < 27 || !probe.metal4_compiler ||
        !probe.tensor_api) {
      std::cout << Json{{"skipped", true},
                        {"reason", "Metal 4 TensorOps unavailable"}}
                       .dump(2)
                << "\n";
      return 0;
    }

    Json report;
    report["pso_and_tg_sram_audit"] = test_pso_and_tg_sram_audit(ctx, libs);
    report["activation_quant_and_gdn_stage3"] =
        test_activation_quant_and_gdn_stage3(ctx, libs);
    report["base12_out64_and_slicing_zero_ulp"] =
        test_base12_out64_and_slicing_zero_ulp(ctx, libs);
    report["dense_w4a8_accuracy_and_slicing"] =
        test_dense_w4a8_accuracy_and_slicing(ctx, libs);
    report["moe_out64_and_w4a8"] = test_moe_out64_and_w4a8(ctx, libs);
    report["miv2_fwht64_all_consumers"] =
        test_miv2_fwht64_and_all_runtime_consumers(ctx, libs);
    report["wp3b_oq4e_dual_channel_and_int4b_exact"] =
        test_wp3b_oq4e_dual_channel_and_int4b_exact(ctx, libs);
    report["all_passed"] = true;
    std::cout << report.dump(2) << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "FATAL: " << ex.what() << "\n";
    return 1;
  }
}
