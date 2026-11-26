#include "affine_nax_cache.h"
#include "gdn_prefill.h"
#include "moe_nax.h"
#include "mxfp4_nax_cache.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/quantization.h"
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

OwnedWeight make_mxfp4_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                              U32 seed, bool pretiled, U32 bn_override = 0u,
                              bool moe_expert = false) {
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

OwnedWeight make_affine_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                               U32 bits, U32 gs, U32 seed, bool pretiled,
                               U32 bn_override = 0u, bool moe_expert = false) {
  OwnedWeight w;
  const U32 bn = bn_override ? bn_override : affine_nax_bn(n);
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  w.view = {.object_id = obj_id,
            .name = "w4a8.affine.q" + std::to_string(bits) + "_g" +
                    std::to_string(gs) + "." + std::to_string(obj_id),
            .format = "oq_affine_q" + std::to_string(bits) + "_g" +
                      std::to_string(gs) + "_bf16_v1",
            .layout = pretiled ? (gs == 128u ? "oq_nax_tiled_g128_v1"
                                             : "oq_nax_tiled_g64_v1")
                               : (moe_expert ? "oq_expert_rowsplit_k128_v1"
                                             : "oq_rowsplit_k128_v1"),
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
      w.ref_w[U64(r) * k + c] =
          std::fma(bf16_val(raw_scale[gi]), float(q), bf16_val(raw_bias[gi]));
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
      std::memcpy(nax_codes.data() + idx * bn * c64 + rn * c64,
                  raw_code.data() + U64(r) * code_stride + g * c64, c64);
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
  const auto w4a8_path = libs / "omlx_m5_w4a8_nax.metallib";
  const auto w4a8_moe_path = libs / "omlx_m5_w4a8_moe_nax.metallib";
  require(std::filesystem::is_regular_file(affine_path), "missing affine_nax metallib");
  require(std::filesystem::is_regular_file(mxfp4_path), "missing mxfp4_nax metallib");
  require(std::filesystem::is_regular_file(moe_path), "missing moe_nax metallib");
  require(std::filesystem::is_regular_file(w4a8_path), "missing w4a8_nax metallib");
  require(std::filesystem::is_regular_file(w4a8_moe_path), "missing w4a8_moe_nax metallib");

  ctx.register_library({.key = "w4a8.audit.affine", .path = affine_path});
  ctx.register_library({.key = "w4a8.audit.mxfp4", .path = mxfp4_path});
  ctx.register_library({.key = "w4a8.audit.moe", .path = moe_path});
  ctx.register_library({.key = "w4a8.audit.w4a8", .path = w4a8_path});
  ctx.register_library({.key = "w4a8.audit.w4a8_moe", .path = w4a8_moe_path});

  const bool shader_val =
      std::getenv("MTL_SHADER_VALIDATION") != nullptr &&
      std::string_view(std::getenv("MTL_SHADER_VALIDATION")) == "1";
  const U64 max_tg_ceil = shader_val ? 65536ULL : 32768ULL;

  U32 compiled = 0;
  U64 max_out64_bf16_tg = 0, max_w4a8_lin_tg = 0, max_w4a8_swi_tg = 0,
      max_w4a8_moe_tg = 0;

  // Base 1.2 Dense Affine out64 SwiGLU & slice linear
  for (const char *gs : {"g64", "g128"}) {
    for (const char *tag : {"q4", "q56", "q3", "q8", "q4_q56", "q56_q4"}) {
      const std::string fn = std::string("omlx_m5_affine_nax_") + tag + "_" +
                             gs + "_swiglu_bm64_bn64_out64_sk1";
      const auto info = ctx.warmup_pipeline(
          {.key = "w4a8.pso." + fn,
           .library_key = "w4a8.audit.affine",
           .function_name = fn,
           .support_indirect_command_buffers = true});
      require(info.static_threadgroup_memory_bytes <= (shader_val ? 65536ULL : 28672ULL),
              fn + " exceeded 28 KiB SRAM budget");
      max_out64_bf16_tg = std::max(max_out64_bf16_tg, info.static_threadgroup_memory_bytes);
      ++compiled;
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
        require(info.static_threadgroup_memory_bytes <= max_tg_ceil,
                fn + " exceeded 32 KiB SRAM");
        ++compiled;
      }
    }
  }

  // Base 1.2 Dense MXFP4 out64 SwiGLU & slice linear
  for (const char *fn : {"omlx_m5_mxfp4_nax_swiglu_bf16_bm64_bn64_out64_sk1",
                         "omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns16_slice_sk1",
                         "omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns32_slice_sk1"}) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a8.pso.") + fn,
         .library_key = "w4a8.audit.mxfp4",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= (shader_val ? 65536ULL : 28672ULL),
            std::string(fn) + " exceeded 28 KiB SRAM budget");
    max_out64_bf16_tg = std::max(max_out64_bf16_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }

  // Base 1.2 MoE out64 SwiGLU
  {
    const std::string fn = "omlx_m5_moe_mxfp4_nax_swiglu_bm64_out64_sk1";
    const auto info = ctx.warmup_pipeline(
        {.key = "w4a8.pso." + fn,
         .library_key = "w4a8.audit.moe",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= max_tg_ceil,
            fn + " exceeded 32 KiB SRAM");
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
      require(info.static_threadgroup_memory_bytes <= max_tg_ceil,
              fn + " exceeded 32 KiB SRAM");
      ++compiled;
    }
  }

  // Route A Dense W4A8 norm/quant/GDN Stage 3 + Linear + SwiGLU
  for (const char *fn : {"omlx_m5_rmsnorm_a8_g64", "omlx_m5_add_rmsnorm_a8_g64",
                         "omlx_m5_quantize_a8_g64",
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
    for (const char *tag : {"q4", "q56", "q3", "q8"}) {
      for (const char *bns : {"bns16", "bns32"}) {
        const std::string fn = std::string("omlx_m5_affine_w4a8_nax_") + tag +
                               "_" + gs + "_bm64_bn64_" + bns + "_sk1";
        const auto info = ctx.warmup_pipeline(
            {.key = "w4a8.pso." + fn,
             .library_key = "w4a8.audit.w4a8",
             .function_name = fn,
             .support_indirect_command_buffers = true});
        require(info.static_threadgroup_memory_bytes <= (shader_val ? 65536ULL : 16384ULL),
                fn + " exceeded 16 KiB linear W4A8 SRAM budget");
        max_w4a8_lin_tg = std::max(max_w4a8_lin_tg, info.static_threadgroup_memory_bytes);
        ++compiled;
      }
    }
    for (const char *tag : {"q4", "q56", "q3", "q8", "q4_q56", "q56_q4"}) {
      const std::string fn = std::string("omlx_m5_affine_w4a8_nax_") + tag +
                             "_" + gs + "_swiglu_bm64_bn64_out64_sk1";
      const auto info = ctx.warmup_pipeline(
          {.key = "w4a8.pso." + fn,
           .library_key = "w4a8.audit.w4a8",
           .function_name = fn,
           .support_indirect_command_buffers = true});
      require(info.static_threadgroup_memory_bytes <= (shader_val ? 65536ULL : 20480ULL),
              fn + " exceeded 20 KiB SwiGLU W4A8 SRAM budget");
      max_w4a8_swi_tg = std::max(max_w4a8_swi_tg, info.static_threadgroup_memory_bytes);
      ++compiled;
    }
  }
  for (const char *fn : {"omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns16_sk1",
                         "omlx_m5_mxfp4_w4a8_nax_bm64_bn64_bns32_sk1",
                         "omlx_m5_mxfp4_w4a8_nax_swiglu_bm64_bn64_out64_sk1"}) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a8.pso.") + fn,
         .library_key = "w4a8.audit.w4a8",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    const bool is_swi = std::string_view(fn).find("_swiglu_") != std::string_view::npos;
    const U64 cap = shader_val ? 65536ULL : (is_swi ? 20480ULL : 16384ULL);
    require(info.static_threadgroup_memory_bytes <= cap,
            std::string(fn) + " exceeded MXFP4 W4A8 SRAM budget");
    if (is_swi) {
      max_w4a8_swi_tg = std::max(max_w4a8_swi_tg, info.static_threadgroup_memory_bytes);
    } else {
      max_w4a8_lin_tg = std::max(max_w4a8_lin_tg, info.static_threadgroup_memory_bytes);
    }
    ++compiled;
  }

  // Route A MoE W4A8
  {
    const std::string fn = "omlx_m5_moe_mxfp4_w4a8_nax_swiglu_bm64_out64_sk1";
    const auto info = ctx.warmup_pipeline(
        {.key = "w4a8.pso." + fn,
         .library_key = "w4a8.audit.w4a8_moe",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    require(info.static_threadgroup_memory_bytes <= (shader_val ? 65536ULL : 20480ULL),
            fn + " exceeded 20 KiB MoE W4A8 SRAM budget");
    max_w4a8_moe_tg = std::max(max_w4a8_moe_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }
  for (const char *bits : {"3", "4", "5", "6", "8"}) {
    for (const char *gs : {"64", "128"}) {
      const std::string fn =
          std::string("omlx_m5_moe_affine_w4a8_nax_swiglu_q") + bits + "_g" +
          gs + "_bm64_out64_sk1";
      const auto info = ctx.warmup_pipeline(
          {.key = "w4a8.pso." + fn,
           .library_key = "w4a8.audit.w4a8_moe",
           .function_name = fn,
           .support_indirect_command_buffers = true});
      require(info.static_threadgroup_memory_bytes <= (shader_val ? 65536ULL : 20480ULL),
              fn + " exceeded 20 KiB MoE W4A8 SRAM budget");
      max_w4a8_moe_tg = std::max(max_w4a8_moe_tg, info.static_threadgroup_memory_bytes);
      ++compiled;
    }
  }

  return {{"compiled_psos", compiled},
          {"max_out64_bf16_tg_bytes", max_out64_bf16_tg},
          {"max_w4a8_lin_tg_bytes", max_w4a8_lin_tg},
          {"max_w4a8_swi_tg_bytes", max_w4a8_swi_tg},
          {"max_w4a8_moe_tg_bytes", max_w4a8_moe_tg},
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
        auto w32 = make_affine_weight(ctx, wid++, kNCheck, kK, bits, gs, bits * 13u + gs, true, 32u);
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
                        std::to_string(gs) + ")");
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
    report["all_passed"] = true;
    std::cout << report.dump(2) << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "FATAL: " << ex.what() << "\n";
    return 1;
  }
}
