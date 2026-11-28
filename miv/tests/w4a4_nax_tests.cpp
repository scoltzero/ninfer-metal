#include "chunk_blend_cache.h"
#include "gdn_prefill.h"
#include "moe_nax.h"
#include "mxfp4_nax_cache.h"
#include "omlx_m5/artifact.h"
#include "omlx_m5/engine.h"
#include "omlx_m5/layer_schedule.h"
#include "omlx_m5/materialization.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/qwen_forward.h"
#include "w4a4_nax.h"
#include "w4a8_nax.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unistd.h>
#include <vector>

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

namespace {

void require(bool ok, std::string_view msg) {
  if (!ok) {
    throw std::runtime_error("W4A4-NAX FAIL: " + std::string(msg));
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

struct TempDirGuard {
  std::filesystem::path path;
  ~TempDirGuard() {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

float bf16_val(std::uint16_t b) { return bf16_to_float(b); }
float silu_ref(float x) { return x / (1.0f + std::exp(-x)); }

std::uint16_t f32_to_bf16(float v) {
  const U32 bits = std::bit_cast<U32>(v);
  const U32 lsb = (bits >> 16U) & 1U;
  return static_cast<std::uint16_t>((bits + 0x7FFFU + lsb) >> 16U);
}

float decode_e2m1(std::uint32_t code) {
  constexpr std::array<float, 8> mag{0.0f, 0.5f, 1.0f, 1.5f,
                                     2.0f, 3.0f, 4.0f, 6.0f};
  const float v = mag[code & 7u];
  return (code & 8u) ? -v : v;
}

float decode_e8m0(std::uint8_t sc) {
  if (sc == 255u) return 0.0f;
  return std::ldexp(1.0f, sc == 0 ? -127 : int(sc) - 127);
}

std::vector<std::uint16_t> read_u16(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 2ULL);
  std::vector<std::uint16_t> out(count);
  std::memcpy(out.data(), raw.data(), count * 2ULL);
  return out;
}

std::vector<std::uint8_t> read_u8(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count);
  std::vector<std::uint8_t> out(count);
  std::memcpy(out.data(), raw.data(), count);
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
  if (na == 0.0 && nr == 0.0) return 1.0;
  require(na > 0.0 && nr > 0.0, std::string(label) + ": degenerate zero vector");
  return dot / (std::sqrt(na) * std::sqrt(nr));
}

double cosine_similarity_f32(std::span<const float> a,
                             std::span<const float> b,
                             std::string_view label) {
  require(a.size() == b.size() && !a.empty(),
          std::string(label) + ": size mismatch");
  double dot = 0.0, na = 0.0, nb = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    require(std::isfinite(a[i]) && std::isfinite(b[i]),
            std::string(label) + ": non-finite value");
    dot += double(a[i]) * double(b[i]);
    na += double(a[i]) * double(a[i]);
    nb += double(b[i]) * double(b[i]);
  }
  if (na == 0.0 && nb == 0.0) return 1.0;
  return (na > 0.0 && nb > 0.0) ? (dot / (std::sqrt(na) * std::sqrt(nb))) : 0.0;
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

std::vector<float> make_activations(U32 rows, U32 cols, U32 seed,
                                    std::vector<std::uint16_t> &bf16_out) {
  bf16_out.resize(U64(rows) * cols);
  std::vector<float> f32(U64(rows) * cols);
  for (U32 r = 0; r < rows; ++r) {
    for (U32 c = 0; c < cols; ++c) {
      const U64 idx = U64(r) * cols + c;
      const float v = std::sin(float(idx * 17u + seed * 31u) * 0.013f) * 1.15f +
                      std::cos(float(c * 7u + r * 5u + seed) * 0.037f) * 0.65f;
      bf16_out[idx] = float_to_bf16(v);
      f32[idx] = bf16_val(bf16_out[idx]);
    }
  }
  return f32;
}

void submit_profile(const MetalContext &ctx, Profile &prof, U32 live_t,
                    bool use_icb = false) {
  for (const auto &patch : prof.patches) {
    if (!patch.position) {
      const U32 val = live_t * patch.multiplier;
      std::memcpy(prof.parameter_data.data() + patch.offset + patch.word * 4u,
                  &val, 4u);
    }
  }
  prof.parameters.write("step", prof.parameter_data);
  if (use_icb && prof.icb.has_value()) {
    (void)ctx.replay_icb(*prof.icb);
  } else {
    const std::array<MetalCommandInvocation, 1> inv{
        MetalCommandInvocation{.plan = &prof.plan,
                               .bindings = prof.bindings,
                               .parameters = &prof.parameters}};
    (void)ctx.execute_batch(inv);
  }
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

OwnedWeight make_mxfp4_weight(const MetalContext &ctx, U32 obj_id, U32 n, U32 k,
                              U32 seed, U32 bn, bool fwht64_weight = false) {
  OwnedWeight w;
  const U32 padded_n = ((n + bn - 1u) / bn) * bn;
  w.view = {.object_id = obj_id,
            .name = "w4a4.mxfp4." + std::to_string(obj_id),
            .format = "mxfp4_e2m1_g32_e8m0_v1",
            .layout = "mx_nax_tiled_bk128_v1",
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

std::vector<float> decode_tiled_a4_cpu(std::span<const std::uint8_t> codes,
                                       std::span<const std::uint8_t> scales,
                                       U32 rows, U32 k) {
  const U32 k_tiles = k / 128u;
  std::vector<float> out(U64(rows) * k, 0.0f);
  for (U32 r = 0; r < rows; ++r) {
    const U32 tm = r / 16u;
    const U32 rm = r % 16u;
    for (U32 kt = 0; kt < k_tiles; ++kt) {
      const U64 ti = U64(tm) * k_tiles + kt;
      for (U32 s = 0; s < 4u; ++s) {
        const float sc = decode_e8m0(scales[ti * 64u + rm * 4u + s]);
        for (U32 j = 0; j < 16u; ++j) {
          const std::uint8_t b = codes[ti * 1024u + rm * 64u + s * 16u + j];
          const U32 c = kt * 128u + s * 32u + j * 2u;
          out[U64(r) * k + c + 0u] = decode_e2m1(b & 0xFu) * sc;
          out[U64(r) * k + c + 1u] = decode_e2m1(b >> 4u) * sc;
        }
      }
    }
  }
  return out;
}

// ============================================================================
// Test 1: W4A4 Dense + MoE PSO SRAM Budget Audit
// ============================================================================
Json test_sram_and_pso_audit(const MetalContext &ctx,
                             const std::filesystem::path &libs) {
  ctx.register_library(
      {.key = "w4a4.audit.dense", .path = libs / "omlx_m5_w4a4_nax.metallib"});
  ctx.register_library(
      {.key = "w4a4.audit.moe", .path = libs / "omlx_m5_w4a4_moe_nax.metallib"});

  constexpr U64 kHighOccCap = 10240ULL;
  const char *sv_env = std::getenv("MTL_SHADER_VALIDATION");
  const bool shader_val = (sv_env != nullptr && std::string_view(sv_env) == "1");
  U32 compiled = 0;
  U64 max_dense_tg = 0;
  U64 max_moe_tg = 0;

  for (const char *fn : {
           "omlx_m5_rmsnorm_a4_bm16_bk128",
           "omlx_m5_add_rmsnorm_a4_bm16_bk128",
           "omlx_m5_quantize_a4_bm16_bk128",
           "omlx_m5_quantize_a4_bm16_bk128_fwht64",
           "omlx_m5_gdn_chunk_norm_gate_inplace_a4_bm16_bk128_c64",
           "omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns16_sk1",
           "omlx_m5_mxfp4_w4a4_nax_bm64_bn64_bns32_sk1",
           "omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_out64_sk1",
           "omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_out64_sk1_fwht64_a4",
       }) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a4.pso.") + fn,
         .library_key = "w4a4.audit.dense",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    if (!shader_val) {
      require(info.static_threadgroup_memory_bytes <= kHighOccCap,
              std::string(fn) + " exceeded 10,240 B SRAM budget (" +
                  std::to_string(info.static_threadgroup_memory_bytes) + ")");
    }
    max_dense_tg = std::max(max_dense_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }

  for (const char *fn : {
           "omlx_m5_moe_mxfp4_w4a4_nax_swiglu_bm64_out64_sk1",
           "omlx_m5_moe_mxfp4_w4a4_nax_swiglu_bm64_out64_sk1_fwht64_a4",
           "omlx_m5_moe_mxfp4_w4a4_nax_down_scatter_bm64_bn64_sk1",
       }) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("w4a4.pso.") + fn,
         .library_key = "w4a4.audit.moe",
         .function_name = fn,
         .support_indirect_command_buffers = true});
    if (!shader_val) {
      require(info.static_threadgroup_memory_bytes <= kHighOccCap,
              std::string(fn) + " exceeded 10,240 B MoE SRAM budget");
    }
    max_moe_tg = std::max(max_moe_tg, info.static_threadgroup_memory_bytes);
    ++compiled;
  }

  return {{"compiled_psos", compiled},
          {"max_dense_tg_bytes", max_dense_tg},
          {"max_moe_tg_bytes", max_moe_tg},
          {"passed", true}};
}

// ============================================================================
// Test 2: All 5 Prologues + Dynamic-T Tail Zeroing [t, mp)
// ============================================================================
Json test_prologues_and_tail_zeroing(const MetalContext &ctx,
                                     const std::filesystem::path &libs) {
  constexpr U32 kCapM = 64u;
  constexpr U32 kLiveT = 37u;
  constexpr U32 kK = 256u;
  constexpr float kEps = 1e-6f;
  constexpr float kWOff = 1.0f;

  std::vector<std::uint16_t> x_u16, d_u16, w_u16;
  const auto x_f32 = make_activations(kCapM, kK, 11u, x_u16);
  const auto d_f32 = make_activations(kCapM, kK, 22u, d_u16);
  w_u16.resize(kK);
  std::vector<float> w_f32(kK);
  for (U32 c = 0; c < kK; ++c) {
    w_u16[c] = float_to_bf16(-0.15f + 0.3f * float(c % 32u) / 31.0f);
    w_f32[c] = bf16_val(w_u16[c]);
  }

  const auto lo = w4a4_nax_workspace_layout(kCapM, kK);
  auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
  auto d_buf = ctx.upload(std::as_bytes(std::span(d_u16)));
  auto w_buf = ctx.upload(std::as_bytes(std::span(w_u16)));
  auto out_bf = ctx.make_buffer(U64(kCapM) * kK * 2ULL);
  auto c_buf = ctx.make_buffer(lo.codes_bytes);
  auto s_buf = ctx.make_buffer(lo.scales_bytes);
  Ref x_ref{&x_buf, 0, x_buf.size()};
  Ref d_ref{&d_buf, 0, d_buf.size()};
  Ref w_ref{&w_buf, 0, w_buf.size()};
  Ref bf_ref{&out_bf, 0, out_bf.size()};
  Ref c_ref{&c_buf, 0, c_buf.size()};
  Ref s_ref{&s_buf, 0, s_buf.size()};

  auto check_tail_zero = [&](std::string_view tag, U32 live_t) {
    const auto got_c = read_u8(c_buf, lo.codes_bytes);
    const auto got_s = read_u8(s_buf, lo.scales_bytes);
    const U32 k_tiles = kK / 128u;
    for (U32 r = live_t; r < lo.mp; ++r) {
      const U32 tm = r / 16u;
      const U32 rm = r % 16u;
      for (U32 kt = 0; kt < k_tiles; ++kt) {
        const U64 ti = U64(tm) * k_tiles + kt;
        for (U32 b = 0; b < 64u; ++b) {
          require(got_c[ti * 1024u + rm * 64u + b] == 0u,
                  std::string(tag) + ": non-zero tail code at row " +
                      std::to_string(r));
        }
        for (U32 s = 0; s < 4u; ++s) {
          require(got_s[ti * 64u + rm * 4u + s] == 0x7Fu,
                  std::string(tag) + ": non-0x7F tail scale at row " +
                      std::to_string(r));
        }
      }
    }
  };

  // 2a. rmsnorm_a4_bm16_bk128
  double rms_cos = 0.0;
  {
    Encoder e(ctx, libs, kCapM, "w4a4.test.rmsnorm");
    e.begin_stage("rmsnorm_a4");
    emit_rmsnorm_a4_bm16_bk128(e, x_ref, w_ref, bf_ref, c_ref, s_ref, kK, kEps,
                               kWOff, /*write_bf16=*/true);
    auto prof = e.finish(true);
    submit_profile(ctx, prof, kLiveT, true);
    check_tail_zero("rmsnorm_a4", kLiveT);
    const auto got_c = read_u8(c_buf, lo.codes_bytes);
    const auto got_s = read_u8(s_buf, lo.scales_bytes);
    const auto deq = decode_tiled_a4_cpu(got_c, got_s, kLiveT, kK);
    std::vector<float> ref(U64(kLiveT) * kK);
    for (U32 r = 0; r < kLiveT; ++r) {
      double sq = 0.0;
      for (U32 c = 0; c < kK; ++c) sq += double(x_f32[r * kK + c]) * x_f32[r * kK + c];
      const float inv = 1.0f / std::sqrt(float(sq / kK) + kEps);
      for (U32 c = 0; c < kK; ++c) {
        ref[r * kK + c] = x_f32[r * kK + c] * inv * (w_f32[c] + kWOff);
      }
    }
    rms_cos = cosine_similarity_f32(deq, ref, "rmsnorm_a4");
    require(rms_cos >= 0.985, "rmsnorm_a4 cos_sim too low");
  }

  // 2b. add_rmsnorm_a4_bm16_bk128
  double add_rms_cos = 0.0;
  {
    auto h_mut = ctx.upload(std::as_bytes(std::span(x_u16)));
    Ref hm_ref{&h_mut, 0, h_mut.size()};
    Encoder e(ctx, libs, kCapM, "w4a4.test.add_rmsnorm");
    e.begin_stage("add_rmsnorm_a4");
    emit_add_rmsnorm_a4_bm16_bk128(e, hm_ref, d_ref, w_ref, bf_ref, c_ref,
                                   s_ref, kK, kEps, kWOff, true);
    auto prof = e.finish(true);
    submit_profile(ctx, prof, kLiveT, true);
    check_tail_zero("add_rmsnorm_a4", kLiveT);
    const auto got_c = read_u8(c_buf, lo.codes_bytes);
    const auto got_s = read_u8(s_buf, lo.scales_bytes);
    const auto deq = decode_tiled_a4_cpu(got_c, got_s, kLiveT, kK);
    std::vector<float> ref(U64(kLiveT) * kK);
    for (U32 r = 0; r < kLiveT; ++r) {
      double sq = 0.0;
      for (U32 c = 0; c < kK; ++c) {
        const float sum = bf16_val(float_to_bf16(x_f32[r * kK + c] + d_f32[r * kK + c]));
        sq += double(sum) * sum;
      }
      const float inv = 1.0f / std::sqrt(float(sq / kK) + kEps);
      for (U32 c = 0; c < kK; ++c) {
        const float sum = bf16_val(float_to_bf16(x_f32[r * kK + c] + d_f32[r * kK + c]));
        ref[r * kK + c] = sum * inv * (w_f32[c] + kWOff);
      }
    }
    add_rms_cos = cosine_similarity_f32(deq, ref, "add_rmsnorm_a4");
    require(add_rms_cos >= 0.985, "add_rmsnorm_a4 cos_sim too low");
  }

  // 2c. quantize_a4_bm16_bk128 & quantize_a4_bm16_bk128_fwht64
  double q_cos = 0.0, q_fwht_cos = 0.0;
  for (bool do_fwht : {false, true}) {
    Encoder e(ctx, libs, kCapM, do_fwht ? "w4a4.test.qfwht" : "w4a4.test.q");
    e.begin_stage("quant_a4");
    emit_quantize_a4_bm16_bk128(e, x_ref, c_ref, s_ref, kK, std::nullopt,
                                std::nullopt, do_fwht);
    auto prof = e.finish(true);
    submit_profile(ctx, prof, kLiveT, true);
    check_tail_zero(do_fwht ? "quantize_a4_fwht64" : "quantize_a4", kLiveT);
    const auto got_c = read_u8(c_buf, lo.codes_bytes);
    const auto got_s = read_u8(s_buf, lo.scales_bytes);
    const auto deq = decode_tiled_a4_cpu(got_c, got_s, kLiveT, kK);
    std::vector<float> ref(x_f32.begin(), x_f32.begin() + U64(kLiveT) * kK);
    if (do_fwht) fwht64_rows_cpu(ref, kLiveT, kK);
    const double cs = cosine_similarity_f32(deq, ref, "quantize_a4");
    require(cs >= 0.985, "quantize_a4 cos_sim too low");
    if (do_fwht) q_fwht_cos = cs;
    else q_cos = cs;
  }

  // 2d. gdn_chunk_norm_gate_inplace_a4_bm16_bk128_c64 (Hv=2, D=128 -> K=256)
  double gdn_cos = 0.0;
  {
    auto io_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
    std::vector<std::uint16_t> nw128_u16(128u);
    std::vector<float> nw128_f32(128u);
    for (U32 c = 0; c < 128u; ++c) {
      nw128_u16[c] = float_to_bf16(0.8f + 0.4f * float(c % 16u) / 15.0f);
      nw128_f32[c] = bf16_val(nw128_u16[c]);
    }
    auto nw_buf = ctx.upload(std::as_bytes(std::span(nw128_u16)));
    Ref io_ref{&io_buf, 0, io_buf.size()};
    Ref nw_ref{&nw_buf, 0, nw_buf.size()};
    Encoder e(ctx, libs, kCapM, "w4a4.test.gdn_a4");
    e.begin_stage("gdn_a4");
    emit_gdn_norm_gate_inplace_a4_bm16_bk128_c64(
        e, kCapM, 1u, 2u, 512u, io_ref, d_ref, nw_ref, c_ref, s_ref, kEps,
        true, true);
    auto prof = e.finish(true);
    submit_profile(ctx, prof, kLiveT, true);
    check_tail_zero("gdn_a4", kLiveT);
    const auto got_c = read_u8(c_buf, lo.codes_bytes);
    const auto got_s = read_u8(s_buf, lo.scales_bytes);
    const auto deq = decode_tiled_a4_cpu(got_c, got_s, kLiveT, kK);
    std::vector<float> ref(U64(kLiveT) * kK);
    for (U32 r = 0; r < kLiveT; ++r) {
      for (U32 vh = 0; vh < 2u; ++vh) {
        double sq = 0.0;
        for (U32 j = 0; j < 128u; ++j) {
          const float v = x_f32[r * kK + vh * 128u + j];
          sq += double(v) * v;
        }
        const float inv = 1.0f / std::sqrt(float(sq / 128.0) + kEps);
        for (U32 j = 0; j < 128u; ++j) {
          const U32 c = vh * 128u + j;
          ref[r * kK + c] =
              x_f32[r * kK + c] * inv * nw128_f32[j] * silu_ref(d_f32[r * kK + c]);
        }
      }
    }
    gdn_cos = cosine_similarity_f32(deq, ref, "gdn_a4");
    require(gdn_cos >= 0.985, "gdn_a4 cos_sim too low");
  }

  return {{"rmsnorm_cos", rms_cos},
          {"add_rmsnorm_cos", add_rms_cos},
          {"quantize_cos", q_cos},
          {"quantize_fwht64_cos", q_fwht_cos},
          {"gdn_stage3_cos", gdn_cos},
          {"tail_zero_verified", true},
          {"passed", true}};
}

// ============================================================================
// Test 3: Dense W4A4 Linear (bns16/bns32), Mode A/B SwiGLU, WP 3C ANE Splice
// ============================================================================
Json test_dense_linear_and_swiglu_modes(const MetalContext &ctx,
                                        const std::filesystem::path &libs) {
  constexpr U32 kM = 77u;
  constexpr U32 kK = 256u;
  constexpr U32 kN = 256u;
  const U32 mp = w4a4_nax_mp(kM);

  std::vector<std::uint16_t> x_u16;
  const auto x_f32 = make_activations(kM, kK, 42u, x_u16);
  auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
  const auto lo = w4a4_nax_workspace_layout(kM, kK, kN);
  auto a4_c = ctx.make_buffer(lo.codes_bytes);
  auto a4_s = ctx.make_buffer(lo.scales_bytes);
  auto mid_c_full = ctx.make_buffer(lo.mid_codes_bytes);
  auto mid_s_full = ctx.make_buffer(lo.mid_scales_bytes);
  auto mid_c_splice = ctx.make_buffer(lo.mid_codes_bytes);
  auto mid_s_splice = ctx.make_buffer(lo.mid_scales_bytes);
  auto y_buf = ctx.make_buffer(U64(kM) * kN * 2ULL);
  auto y_down = ctx.make_buffer(U64(kM) * kK * 2ULL);

  Ref x_ref{&x_buf, 0, x_buf.size()};
  Ref ac_ref{&a4_c, 0, a4_c.size()};
  Ref as_ref{&a4_s, 0, a4_s.size()};
  Ref mc_full{&mid_c_full, 0, mid_c_full.size()};
  Ref ms_full{&mid_s_full, 0, mid_s_full.size()};
  Ref mc_spl{&mid_c_splice, 0, mid_c_splice.size()};
  Ref ms_spl{&mid_s_splice, 0, mid_s_splice.size()};
  Ref y_ref{&y_buf, 0, y_buf.size()};
  Ref yd_ref{&y_down, 0, y_down.size()};

  // Quantize X once
  {
    Encoder e(ctx, libs, kM, "w4a4.dense.quant");
    e.begin_stage("q");
    emit_quantize_a4_bm16_bk128(e, x_ref, ac_ref, as_ref, kK, kM);
    auto prof = e.finish(false);
    submit_profile(ctx, prof, kM, false);
  }
  const auto a4_deq = decode_tiled_a4_cpu(read_u8(a4_c, lo.codes_bytes),
                                          read_u8(a4_s, lo.scales_bytes), kM, kK);

  // 3a. Linear bns16 & bns32 (including row_offset=64, slice_n=128)
  double lin16_cos = 0.0, lin32_cos = 0.0;
  for (U32 bn : {16u, 32u}) {
    auto ow = make_mxfp4_weight(ctx, 100u + bn, kN, kK, 7u + bn, bn, false);
    TiledMxfp4Weight tw{.codes = plane(ow.view, "nax_data"),
                        .scales = plane(ow.view, "nax_scale"),
                        .n = kN,
                        .k = kK,
                        .bn = bn,
                        .padded_n = kN};
    y_buf.zero();
    Encoder e(ctx, libs, kM, "w4a4.dense.lin.bns" + std::to_string(bn));
    e.begin_stage("lin");
    emit_mxfp4_w4a4_nax_linear(e, tw, ac_ref, as_ref, y_ref, kM, 0u, kN, 64u);
    emit_mxfp4_w4a4_nax_linear(e, tw, ac_ref, as_ref, y_ref, kM, 64u, kN, kN - 64u);
    auto prof = e.finish(false);
    submit_profile(ctx, prof, kM, false);

    std::vector<float> ref(U64(kM) * kN, 0.0f);
    for (U32 m = 0; m < kM; ++m) {
      for (U32 n = 0; n < kN; ++n) {
        float acc = 0.0f;
        for (U32 k = 0; k < kK; ++k) {
          acc += a4_deq[m * kK + k] * ow.ref_w[n * kK + k];
        }
        ref[m * kN + n] = acc;
      }
    }
    const double cs = cosine_similarity_bf16(read_u16(y_buf, U64(kM) * kN), ref,
                                             "w4a4_linear");
    require(cs >= 0.995, "w4a4_linear cos_sim vs quantized-A4 FP32 ref < 0.995");
    if (bn == 16u) lin16_cos = cs;
    else lin32_cos = cs;
  }

  // 3b. Mode A SwiGLU + Mode B SwiGLU + WP 3C ANE Splice bit-exactness
  auto gw_ow = make_mxfp4_weight(ctx, 201u, kN, kK, 31u, 16u, false);
  auto uw_ow = make_mxfp4_weight(ctx, 202u, kN, kK, 47u, 16u, false);
  auto dw_ow = make_mxfp4_weight(ctx, 203u, kK, kN, 61u, 16u, true);
  TiledMxfp4Weight gw{.codes = plane(gw_ow.view, "nax_data"),
                      .scales = plane(gw_ow.view, "nax_scale"),
                      .n = kN, .k = kK, .bn = 16u, .padded_n = kN};
  TiledMxfp4Weight uw{.codes = plane(uw_ow.view, "nax_data"),
                      .scales = plane(uw_ow.view, "nax_scale"),
                      .n = kN, .k = kK, .bn = 16u, .padded_n = kN};
  TiledMxfp4Weight dw{.codes = plane(dw_ow.view, "nax_data"),
                      .scales = plane(dw_ow.view, "nax_scale"),
                      .n = kK, .k = kN, .bn = 16u, .padded_n = kK};

  // Mode A SwiGLU
  y_buf.zero();
  {
    Encoder e(ctx, libs, kM, "w4a4.dense.swia");
    e.begin_stage("swia");
    emit_mxfp4_w4a4_nax_swiglu(e, gw, uw, ac_ref, as_ref, y_ref, kM);
    auto prof = e.finish(false);
    submit_profile(ctx, prof, kM, false);
  }
  std::vector<float> swi_ref(U64(kM) * kN, 0.0f);
  for (U32 m = 0; m < kM; ++m) {
    for (U32 n = 0; n < kN; ++n) {
      float g_acc = 0.0f, u_acc = 0.0f;
      for (U32 k = 0; k < kK; ++k) {
        g_acc += a4_deq[m * kK + k] * gw_ow.ref_w[n * kK + k];
        u_acc += a4_deq[m * kK + k] * uw_ow.ref_w[n * kK + k];
      }
      swi_ref[m * kN + n] = silu_ref(g_acc) * u_acc;
    }
  }
  const double swi_a_cos = cosine_similarity_bf16(
      read_u16(y_buf, U64(kM) * kN), swi_ref, "w4a4_swiglu_mode_a");
  require(swi_a_cos >= 0.99, "w4a4 Mode A SwiGLU cos_sim < 0.99");

  // Mode B SwiGLU full-N vs 2-half WP 3C ANE splice (row_offset % 128 == 0)
  mid_c_full.zero();
  mid_s_full.zero();
  mid_c_splice.zero();
  mid_s_splice.zero();
  {
    Encoder e(ctx, libs, kM, "w4a4.dense.swib");
    e.begin_stage("swib_full");
    emit_mxfp4_w4a4_nax_swiglu_fwht64_a4(e, gw, uw, ac_ref, as_ref, mc_full,
                                         ms_full, y_ref, false, kM, 0u, kN, kN);
    e.begin_stage("swib_splice");
    emit_mxfp4_w4a4_nax_swiglu_fwht64_a4(e, gw, uw, ac_ref, as_ref, mc_spl,
                                         ms_spl, y_ref, false, kM, 0u, kN, 128u);
    emit_mxfp4_w4a4_nax_swiglu_fwht64_a4(e, gw, uw, ac_ref, as_ref, mc_spl,
                                         ms_spl, y_ref, false, kM, 128u, kN, 128u);
    e.begin_stage("down");
    emit_mxfp4_w4a4_nax_linear(e, dw, mc_spl, ms_spl, yd_ref, kM);
    auto prof = e.finish(false);
    submit_profile(ctx, prof, kM, false);
  }
  const auto full_c = read_u8(mid_c_full, lo.mid_codes_bytes);
  const auto spl_c = read_u8(mid_c_splice, lo.mid_codes_bytes);
  const auto full_s = read_u8(mid_s_full, lo.mid_scales_bytes);
  const auto spl_s = read_u8(mid_s_splice, lo.mid_scales_bytes);
  require(full_c == spl_c && full_s == spl_s,
          "WP 3C ANE splice Mode B SwiGLU codes/scales are NOT bit-exact!");
  {
    auto swi_rot_ref = swi_ref;
    fwht64_rows_cpu(swi_rot_ref, kM, kN);
    const auto mid_deq = decode_tiled_a4_cpu(full_c, full_s, kM, kN);
    const double mid_cs = cosine_similarity_f32(mid_deq, swi_rot_ref, "w4a4_mode_b_mid");
    if (mid_cs < 0.98) {
      std::string diag = "w4a4 Mode B mid_deq cos_sim=" + std::to_string(mid_cs) + " rows:";
      for (U32 r0 = 0; r0 < kM; r0 += 16u) {
        const U32 r1 = std::min(kM, r0 + 16u);
        const std::size_t off = std::size_t(r0) * kN;
        const std::size_t cnt = std::size_t(r1 - r0) * kN;
        const double rc = cosine_similarity_f32(
            std::span<const float>(mid_deq).subspan(off, cnt),
            std::span<const float>(swi_rot_ref).subspan(off, cnt), "row");
        diag += " [" + std::to_string(r0) + ".." + std::to_string(r1) + "]=" + std::to_string(rc);
      }
      require(false, diag);
    }
  }

  // Verify Mode B -> W4A4 down_proj matches FP32 reference!
  // Note dw_ow.ref_w already has FWHT-64 baked in, so dot(swi_ref, dw_ow.ref_w) == dot(H*swi_ref, W_rot).
  std::vector<float> down_ref(U64(kM) * kK, 0.0f);
  for (U32 m = 0; m < kM; ++m) {
    for (U32 n = 0; n < kK; ++n) {
      float acc = 0.0f;
      for (U32 c = 0; c < kN; ++c) {
        acc += swi_ref[m * kN + c] * dw_ow.ref_w[n * kN + c];
      }
      down_ref[m * kK + n] = acc;
    }
  }
  const double mode_b_down_cos = cosine_similarity_bf16(
      read_u16(y_down, U64(kM) * kK), down_ref, "w4a4_mode_b_down");
  require(mode_b_down_cos >= 0.985,
          "w4a4 Mode B SwiGLU + FWHT64 + down_proj cos_sim=" +
              std::to_string(mode_b_down_cos) + " < 0.985");
  (void)mp;

  // 3c. STRESS |de| > 3 dual shift-merge fallback (extreme UE8M0 jumps on both A and W!)
  double stress_cos = 0.0;
  {
    auto ow_s = make_mxfp4_weight(ctx, 299u, kN, kK, 99u, 16u, false);
    // Patch scales of ow_s and a4_s to trigger |de| > 3 across both fast and slow SGs
    std::vector<std::uint8_t> sc_a = read_u8(a4_s, lo.scales_bytes);
    for (std::size_t i = 0; i + 1 < sc_a.size(); i += 2) {
      if ((i & 6u) == 2u) {
        sc_a[i] = 118u;
        sc_a[i + 1] = 128u; // |de_a| = 10 > 3!
      }
    }
    auto a4_s_stress = ctx.upload(std::as_bytes(std::span(sc_a)));
    Ref as_stress{&a4_s_stress, 0, a4_s_stress.size()};

    // Rebuild weight with |de_w| = 9 > 3 on half the rows (both sg_n=0 and sg_n=1!)
    std::vector<std::uint8_t> raw_codes(U64(kN) * (kK / 2u), 0);
    std::vector<std::uint8_t> raw_scales(U64(kN) * (kK / 32u), 127);
    std::vector<float> ref_w_stress(U64(kN) * kK, 0.0f);
    for (U32 r = 0; r < kN; ++r) {
      for (U32 g = 0; g < kK / 32u; ++g) {
        raw_scales[U64(r) * (kK / 32u) + g] =
            ((r % 32u) >= 16u && (g & 1u) == 1u) ? std::uint8_t(131u)
            : ((r % 32u) >= 16u)                 ? std::uint8_t(121u)
                                                 : std::uint8_t(124u + ((r + g) % 3u));
      }
      for (U32 c = 0; c < kK; ++c) {
        const U32 code = (r * 5u + c * 13u + 7u) & 0x0Fu;
        raw_codes[U64(r) * (kK / 2u) + (c >> 1u)] |= std::uint8_t(code << ((c & 1u) * 4u));
        ref_w_stress[U64(r) * kK + c] =
            decode_e2m1(code) * decode_e8m0(raw_scales[U64(r) * (kK / 32u) + c / 32u]);
      }
    }
    const U32 blocks = kK / 128u;
    std::vector<std::uint8_t> nc(U64(kN / 16u) * blocks * 16u * 64u, 0);
    std::vector<std::uint8_t> ns(U64(kN / 16u) * blocks * 16u * 4u, 0x7Fu);
    for (U32 r = 0; r < kN; ++r) {
      const U32 tn = r / 16u, rn = r % 16u;
      for (U32 blk = 0; blk < blocks; ++blk) {
        const U64 idx = U64(tn) * blocks + blk;
        std::memcpy(nc.data() + idx * 1024u + rn * 64u,
                    raw_codes.data() + U64(r) * (kK / 2u) + blk * 64u, 64u);
        std::memcpy(ns.data() + idx * 64u + rn * 4u,
                    raw_scales.data() + U64(r) * (kK / 32u) + blk * 4u, 4u);
      }
    }
    auto nc_buf = ctx.upload(std::as_bytes(std::span(nc)));
    auto ns_buf = ctx.upload(std::as_bytes(std::span(ns)));
    TiledMxfp4Weight tw_s{.codes = Ref{&nc_buf, 0, nc_buf.size()},
                          .scales = Ref{&ns_buf, 0, ns_buf.size()},
                          .n = kN, .k = kK, .bn = 16u, .padded_n = kN};
    y_buf.zero();
    {
      Encoder e(ctx, libs, kM, "w4a4.dense.stress");
      e.begin_stage("stress");
      emit_mxfp4_w4a4_nax_linear(e, tw_s, ac_ref, as_stress, y_ref, kM);
      auto prof = e.finish(false);
      submit_profile(ctx, prof, kM, false);
    }
    const auto a4_deq_s = decode_tiled_a4_cpu(read_u8(a4_c, lo.codes_bytes), sc_a, kM, kK);
    std::vector<float> ref_s(U64(kM) * kN, 0.0f);
    for (U32 m = 0; m < kM; ++m) {
      for (U32 n = 0; n < kN; ++n) {
        float acc = 0.0f;
        for (U32 k = 0; k < kK; ++k) acc += a4_deq_s[m * kK + k] * ref_w_stress[n * kK + k];
        ref_s[m * kN + n] = acc;
      }
    }
    stress_cos = cosine_similarity_bf16(read_u16(y_buf, U64(kM) * kN), ref_s, "w4a4_stress_de");
    require(stress_cos >= 0.9999,
            "w4a4 STRESS |de|>3 fallback cos_sim=" + std::to_string(stress_cos) + " < 0.9999");
  }

  return {{"lin_bns16_cos", lin16_cos},
          {"lin_bns32_cos", lin32_cos},
          {"stress_de_cos", stress_cos},
          {"swiglu_mode_a_cos", swi_a_cos},
          {"mode_b_down_cos", mode_b_down_cos},
          {"ane_splice_bit_exact", true},
          {"passed", true}};
}

// ============================================================================
// Test 4: MoE MXFP4 W4A4 (Mode A + Mode B + Down Scatter, compact32 & full64)
// ============================================================================
Json test_moe_w4a4_modes(const MetalContext &ctx,
                         const std::filesystem::path &libs) {
  constexpr U32 kT = 96u;
  constexpr U32 kH = 256u;
  constexpr U32 kF = 256u;
  constexpr U32 kTopK = 8u;
  constexpr U32 kExperts = 256u;
  const U32 mp = w4a4_nax_mp(kT);

  std::vector<std::uint16_t> x_u16;
  const auto x_f32 = make_activations(kT, kH, 91u, x_u16);
  auto x_buf = ctx.upload(std::as_bytes(std::span(x_u16)));
  const auto lo = w4a4_nax_workspace_layout(kT, kH);
  auto a4_c = ctx.make_buffer(lo.codes_bytes);
  auto a4_s = ctx.make_buffer(lo.scales_bytes);

  // Route tokens so expert 0 gets 53 rows (full64 tile) and expert 1 gets 19 rows (compact32 tile),
  // with distinct top-8 experts per token across experts 0..15!
  std::vector<U32> route_ids_h(U64(kT) * kTopK, 0u);
  std::vector<std::uint16_t> route_scores_h(U64(kT) * kTopK, 0u);
  for (U32 t = 0; t < kT; ++t) {
    for (U32 k = 0; k < kTopK; ++k) {
      U32 exp = (k == 0u && t < 53u) ? 0u
                : (k == 1u && t < 19u) ? 1u
                                       : (2u + ((t + k * 2u) % 14u));
      route_ids_h[t * kTopK + k] = exp;
      route_scores_h[t * kTopK + k] = float_to_bf16(0.125f);
    }
  }
  auto rids_buf = ctx.upload(std::as_bytes(std::span(route_ids_h)));
  auto rsc_buf = ctx.upload(std::as_bytes(std::span(route_scores_h)));
  auto exp_out_buf = ctx.make_buffer(U64(kT) * kTopK * kH * 2ULL);
  auto routed_buf = ctx.make_buffer(U64(kT) * kH * 2ULL);

  // Build 3D untiled MXFP4 expert weights (256 experts, non-zero for experts 0..15)
  auto make_3d_moe_w = [&](U32 id, U32 n, U32 k, U32 seed, bool fwht64_w) {
    OwnedWeight w;
    w.view = {.object_id = id,
              .name = "moe.w." + std::to_string(id),
              .format = "mxfp4_e2m1_g32_e8m0_v1",
              .layout = "mx_expert_rowsplit_k128_v1",
              .parent_shape = {kExperts, n, k},
              .padded_k = k,
              .row_count = kExperts * n,
              .in_features = k,
              .input_transform = fwht64_w ? "fwht64" : "none"};
    std::vector<std::uint8_t> codes(U64(kExperts) * n * (k / 2u), 0);
    std::vector<std::uint8_t> scales(U64(kExperts) * n * (k / 32u), 0x7Fu);
    w.ref_w.resize(16ULL * n * k, 0.0f);
    for (U32 ex = 0; ex < 16u; ++ex) {
      for (U32 r = 0; r < n; ++r) {
        const U64 flat_r = U64(ex) * n + r;
        for (U32 g = 0; g < k / 32u; ++g) {
          const U64 sh = (U64(ex) * 0x9E3779B9ULL) ^ (U64(r) * 0xBF58476DULL) ^
                         (U64(g) * 0x94D049BBULL) ^ U64(seed);
          scales[flat_r * (k / 32u) + g] =
              std::uint8_t(121u + ((sh ^ (sh >> 17u)) % 6u));
        }
        for (U32 c = 0; c < k; ++c) {
          U64 hq = (U64(ex) * 0x9E3779B97F4A7C15ULL) ^
                   ((U64(r) * k + c) * 0xBF58476D1CE4E5B9ULL) ^
                   (U64(seed) * 0x94D049BB133111EBULL);
          hq ^= (hq >> 30u);
          hq *= 0xBF58476D1CE4E5B9ULL;
          const U32 q = U32((hq >> 27u) & 0xFu);
          codes[flat_r * (k / 2u) + (c >> 1u)] |= std::uint8_t(q << ((c & 1u) * 4u));
          const float sc = decode_e8m0(scales[flat_r * (k / 32u) + c / 32u]);
          w.ref_w[flat_r * k + c] = decode_e2m1(q) * sc;
        }
      }
      if (fwht64_w) {
        std::vector<float> tmp(w.ref_w.begin() + U64(ex) * n * k,
                               w.ref_w.begin() + U64(ex + 1u) * n * k);
        fwht64_rows_cpu(tmp, n, k);
        std::copy(tmp.begin(), tmp.end(), w.ref_w.begin() + U64(ex) * n * k);
      }
    }
    w.add_plane(ctx, "data", codes.data(), codes.size(), k / 2u);
    w.add_plane(ctx, "scale", scales.data(), scales.size(), k / 32u);
    return w;
  };

  auto gw = make_3d_moe_w(301u, kF, kH, 11u, false);
  auto uw = make_3d_moe_w(302u, kF, kH, 23u, false);
  auto dw_a = make_3d_moe_w(303u, kH, kF, 37u, false);
  auto dw_b = make_3d_moe_w(304u, kH, kF, 37u, true);

  std::vector<std::unique_ptr<MetalBuffer>> ws_bufs;
  MoeNaxWorkspace ws;
  ws.allocate(ctx, kT, kH, kF, ws_bufs, kExperts, kTopK);

  Ref x_ref{&x_buf, 0, x_buf.size()};
  Ref ac_ref{&a4_c, 0, a4_c.size()};
  Ref as_ref{&a4_s, 0, a4_s.size()};
  Ref rids_ref{&rids_buf, 0, rids_buf.size()};
  Ref rsc_ref{&rsc_buf, 0, rsc_buf.size()};
  Ref eout_ref{&exp_out_buf, 0, exp_out_buf.size()};
  Ref rout_ref{&routed_buf, 0, routed_buf.size()};

  auto compute_ref = [&](const std::vector<float> &act_f32,
                         const OwnedWeight &dw_w) {
    std::vector<float> ref(U64(kT) * kH, 0.0f);
    for (U32 t = 0; t < kT; ++t) {
      for (U32 k = 0; k < kTopK; ++k) {
        const U32 ex = route_ids_h[t * kTopK + k];
        const float sc = bf16_val(route_scores_h[t * kTopK + k]);
        std::vector<float> mid(kF, 0.0f);
        for (U32 f = 0; f < kF; ++f) {
          float ga = 0.0f, ua = 0.0f;
          for (U32 h = 0; h < kH; ++h) {
            ga += act_f32[t * kH + h] * gw.ref_w[(U64(ex) * kF + f) * kH + h];
            ua += act_f32[t * kH + h] * uw.ref_w[(U64(ex) * kF + f) * kH + h];
          }
          mid[f] = silu_ref(ga) * ua;
        }
        for (U32 h = 0; h < kH; ++h) {
          float da = 0.0f;
          for (U32 f = 0; f < kF; ++f) {
            da += mid[f] * dw_w.ref_w[(U64(ex) * kH + h) * kF + f];
          }
          ref[t * kH + h] += da * sc;
        }
      }
    }
    return ref;
  };

  double moe_a_cos_exact = 0.0, moe_b_cos_exact = 0.0;
  double moe_a_cos_e2e = 0.0, moe_b_cos_e2e = 0.0;
  for (bool mode_b : {false, true}) {
    routed_buf.zero();
    Encoder e(ctx, libs, kT, mode_b ? "w4a4.moe.b" : "w4a4.moe.a");
    e.begin_stage("quant");
    emit_quantize_a4_bm16_bk128(e, x_ref, ac_ref, as_ref, kH, kT);
    e.begin_stage("moe_w4a4");
    const bool ok = emit_moe_w4a4_nax_ffn(
        e, ws, gw.view, uw.view, mode_b ? dw_b.view : dw_a.view, ac_ref, as_ref,
        mp, rids_ref, rsc_ref, eout_ref, rout_ref, kT, kH, kF, kTopK, kExperts,
        kT, Ref{}, /*emit_reduce=*/true, /*w4a4_down_proj=*/mode_b);
    require(ok, "emit_moe_w4a4_nax_ffn returned false");
    auto prof = e.finish(false);
    submit_profile(ctx, prof, kT, false);
    const auto a4_deq = decode_tiled_a4_cpu(read_u8(a4_c, lo.codes_bytes),
                                            read_u8(a4_s, lo.scales_bytes), kT, kH);
    const auto got = read_u16(routed_buf, U64(kT) * kH);
    const auto ref_exact = compute_ref(a4_deq, mode_b ? dw_b : dw_a);
    const auto ref_e2e = compute_ref(x_f32, mode_b ? dw_b : dw_a);
    const double cs_exact = cosine_similarity_bf16(
        got, ref_exact, mode_b ? "moe_w4a4_mode_b_exact" : "moe_w4a4_mode_a_exact");
    const double cs_e2e = cosine_similarity_bf16(
        got, ref_e2e, mode_b ? "moe_w4a4_mode_b_e2e" : "moe_w4a4_mode_a_e2e");
    const double exact_thresh = mode_b ? 0.985 : 0.995;
    require(cs_exact >= exact_thresh,
            std::string("MoE W4A4 ") + (mode_b ? "ModeB" : "ModeA") +
                " exact cos_sim=" + std::to_string(cs_exact) + " < " +
                std::to_string(exact_thresh));
    require(cs_e2e >= 0.93,
            std::string("MoE W4A4 ") + (mode_b ? "ModeB" : "ModeA") +
                " e2e cos_sim=" + std::to_string(cs_e2e) + " < 0.93");
    if (mode_b) {
      moe_b_cos_exact = cs_exact;
      moe_b_cos_e2e = cs_e2e;
    } else {
      moe_a_cos_exact = cs_exact;
      moe_a_cos_e2e = cs_e2e;
    }
  }

  return {{"moe_mode_a_cos_exact", moe_a_cos_exact},
          {"moe_mode_b_cos_exact", moe_b_cos_exact},
          {"moe_mode_a_cos_e2e", moe_a_cos_e2e},
          {"moe_mode_b_cos_e2e", moe_b_cos_e2e},
          {"passed", true}};
}

// ============================================================================
// Synthetic .miv v2 Model Builder (MXFP4 + FWHT64 or oQ4e) for E2E Tests
// ============================================================================
U64 det_hash(U32 seed, U64 idx) {
  U64 x = (U64(seed) * 0x9E3779B97F4A7C15ULL) ^ (idx * 0xBF58476D1CE4E5B9ULL);
  x = (x ^ (x >> 30U)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31U);
}

float det_val(U32 seed, U64 idx, float amp = 0.04F) {
  const U64 x = det_hash(seed, idx);
  const float u =
      static_cast<float>(static_cast<std::int32_t>(x & 0xFFFFU) - 32768) /
      32768.0F;
  return u * amp;
}

std::string hex_crc32(U32 crc) {
  std::ostringstream out;
  out << std::hex << std::nouppercase << std::setfill('0') << std::setw(8) << crc;
  return out.str();
}

void append_u32(std::vector<std::byte> &out, U32 v) {
  for (int i = 0; i < 4; ++i)
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
}

void append_u64(std::vector<std::byte> &out, U64 v) {
  for (int i = 0; i < 8; ++i)
    out.push_back(static_cast<std::byte>((v >> (i * 8)) & 0xFFU));
}

void write_binary_file(const std::filesystem::path &path,
                       std::span<const std::byte> bytes) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream out(path, std::ios::binary);
  require(out.good(), "failed to open file for write: " + path.string());
  out.write(reinterpret_cast<const char *>(bytes.data()),
            static_cast<std::streamsize>(bytes.size()));
  require(out.good(), "failed writing file: " + path.string());
}

void write_text_file(const std::filesystem::path &path, std::string_view text) {
  write_binary_file(
      path, std::as_bytes(std::span<const char>(text.data(), text.size())));
}

struct SynthObj {
  U32 id = 0;
  std::string name, format, layout, binding_use, activation_policy;
  std::string input_transform = "none";
  std::vector<U64> shape;
  std::optional<U64> padded_k;
  std::vector<std::pair<std::string, std::vector<std::byte>>> planes;
};

struct BuiltSynthModel {
  std::filesystem::path dir;
  std::filesystem::path artifact_path;
  QwenModelConfig config;
};

BuiltSynthModel build_synthetic_hybrid_model(const std::filesystem::path &dir,
                                             bool use_mxfp4,
                                             bool use_v2_fwht64) {
  std::filesystem::create_directories(dir);
  constexpr U32 kLayers = 4U; // 3 GDN + 1 Full-Attn (layer 3)
  constexpr U32 h = 1024U, f = 1024U, v = 256U;
  constexpr U32 q_heads = 4U, kv_heads = 2U, hd = 256U;
  constexpr U32 q = q_heads * hd, kv = kv_heads * hd;
  constexpr U32 gk_heads = 4U, gv_heads = 8U, gdn_kd = 128U, gdn_vd = 128U;
  constexpr U32 gk = gk_heads * gdn_kd, gv = gv_heads * gdn_vd;
  constexpr U32 conv = 2U * gk + gv;

  Json layer_types = {"linear_attention", "linear_attention",
                      "linear_attention", "full_attention"};
  Json cfg = {
      {"model_type", "qwen3_5"},
      {"text_config",
       {{"model_type", "qwen3_5_text"},
        {"dtype", "bfloat16"},
        {"hidden_size", h},
        {"intermediate_size", f},
        {"vocab_size", v},
        {"num_hidden_layers", kLayers},
        {"num_attention_heads", q_heads},
        {"num_key_value_heads", kv_heads},
        {"head_dim", hd},
        {"max_position_embeddings", 32768},
        {"rms_norm_eps", 1e-6},
        {"rope_parameters",
         {{"rope_theta", 10000000.0},
          {"partial_rotary_factor", 0.25},
          {"mrope_interleaved", true}}},
        {"linear_num_key_heads", gk_heads},
        {"linear_num_value_heads", gv_heads},
        {"linear_key_head_dim", gdn_kd},
        {"linear_value_head_dim", gdn_vd},
        {"linear_conv_kernel_dim", 4},
        {"layer_types", layer_types},
        {"tie_word_embeddings", false}}},
  };
  write_text_file(dir / "config.json", cfg.dump(2));

  std::vector<SynthObj> objs;
  auto add_bf16 = [&](std::string name, std::vector<U64> shape, U32 seed,
                      float bias = 0.0f, float amp = 0.04f,
                      std::string use = "") {
    U64 cnt = 1;
    for (U64 d : shape) cnt *= d;
    std::vector<std::uint16_t> bf(cnt);
    for (U64 i = 0; i < cnt; ++i) bf[i] = f32_to_bf16(bias + det_val(seed, i, amp));
    auto raw = std::as_bytes(std::span<const std::uint16_t>(bf));
    objs.push_back({.id = U32(objs.size()),
                    .name = std::move(name),
                    .format = "direct_bf16_v1",
                    .layout = "contiguous_le_v1",
                    .binding_use = std::move(use),
                    .activation_policy = "A16Only",
                    .shape = std::move(shape),
                    .planes = {{"data", {raw.begin(), raw.end()}}}});
  };
  auto add_mxfp4 = [&](std::string name, U32 n, U32 k, U32 seed,
                       std::string use = "linear", bool fwht64 = false) {
    const U32 bn = (n >= 65536U) ? 32U : 16U;
    const U32 padded_n = ((n + bn - 1U) / bn) * bn;
    const U32 k_blocks = k / 128U;
    std::vector<std::byte> nax_data(std::size_t(padded_n) * (k / 2U), std::byte{0});
    std::vector<std::byte> nax_scale(std::size_t(padded_n) * (k / 32U), std::byte{119});
    for (U32 r = 0; r < n; ++r) {
      const U32 tn = r / bn, rn = r % bn;
      for (U32 blk = 0; blk < k_blocks; ++blk) {
        const std::size_t ti = std::size_t(tn) * k_blocks + blk;
        for (U32 sub = 0; sub < 4U; ++sub) {
          const auto sc = static_cast<std::uint8_t>(
              119U + (det_hash(seed + 1U, U64(r) * 4096ULL + blk * 4ULL + sub) % 3U));
          nax_scale[ti * (bn * 4U) + rn * 4U + sub] = static_cast<std::byte>(sc);
          for (U32 j = 0; j < 16U; ++j) {
            const U32 c0 = blk * 128U + sub * 32U + 2U * j;
            const U64 h0 = det_hash(seed, U64(r) * k + c0);
            nax_data[ti * (bn * 64U) + rn * 64U + sub * 16U + j] =
                static_cast<std::byte>((h0 & 0xFU) | (((h0 >> 8U) & 0xFU) << 4U));
          }
        }
      }
    }
    objs.push_back({.id = U32(objs.size()),
                    .name = std::move(name),
                    .format = "mxfp4_e2m1_g32_e8m0_v1",
                    .layout = "mx_nax_tiled_bk128_v1",
                    .binding_use = std::move(use),
                    .activation_policy = "AllowA4",
                    .input_transform = fwht64 ? "fwht64" : "none",
                    .shape = {n, k},
                    .padded_k = k,
                    .planes = {{"nax_data", std::move(nax_data)},
                               {"nax_scale", std::move(nax_scale)}}});
  };
  auto add_affine4 = [&](std::string name, U32 n, U32 k, U32 seed,
                         std::string use = "linear") {
    const U32 bn = 16U;
    const U32 padded_n = ((n + bn - 1U) / bn) * bn;
    const U32 groups64 = k / 64U;
    std::vector<std::byte> nax_codes(std::size_t(padded_n) * (k / 2U), std::byte{0});
    std::vector<std::byte> nax_sb(std::size_t(groups64) * padded_n * 4U, std::byte{0});
    for (U32 r = 0; r < n; ++r) {
      for (U32 g64 = 0; g64 < groups64; ++g64) {
        const float span = 0.08f;
        const U32 packed = U32(f32_to_bf16(span / 15.0f)) |
                           (U32(f32_to_bf16(-0.5f * span)) << 16U);
        std::memcpy(nax_sb.data() + (std::size_t(g64) * padded_n + r) * 4U,
                    &packed, 4U);
      }
      const U32 tn = r / bn, rn = r % bn;
      for (U32 col = 0; col < k; ++col) {
        const U32 qv = static_cast<U32>(det_hash(seed, U64(r) * k + col)) & 0xFU;
        const std::size_t ti = std::size_t(tn) * groups64 + (col / 64U);
        const std::size_t cb = ti * (bn * 32U) + rn * 32U + ((col % 64U) >> 1U);
        const auto cur = std::to_integer<std::uint8_t>(nax_codes[cb]);
        nax_codes[cb] = static_cast<std::byte>(cur | (qv << ((col & 1U) * 4U)));
      }
    }
    objs.push_back({.id = U32(objs.size()),
                    .name = std::move(name),
                    .format = "oq_affine_q4_g64_bf16_v1",
                    .layout = "oq_nax_tiled_g64_v1",
                    .binding_use = std::move(use),
                    .activation_policy = "A16Only",
                    .shape = {n, k},
                    .padded_k = k,
                    .planes = {{"nax_codes", std::move(nax_codes)},
                               {"nax_sb", std::move(nax_sb)}}});
  };
  auto add_lin = [&](std::string name, U32 n, U32 k, U32 seed,
                     std::string use = "linear", bool fwht64 = false) {
    if (use_mxfp4) add_mxfp4(std::move(name), n, k, seed, std::move(use), fwht64);
    else add_affine4(std::move(name), n, k, seed, std::move(use));
  };

  add_bf16("model.embed_tokens.weight", {v, h}, 101U, 0.0f, 0.10f, "embedding");
  add_bf16("model.norm.weight", {h}, 102U, 0.0f, 0.03f);
  add_lin("lm_head.weight", v, h, 103U, "lm_head", false);
  for (U32 li = 0; li < kLayers; ++li) {
    const std::string p = "model.layers." + std::to_string(li);
    const U32 s = 1000U + li * 100U;
    add_bf16(p + ".input_layernorm.weight", {h}, s + 1U, 0.0f, 0.03f);
    if (li == 3U) {
      add_lin(p + ".self_attn.q_proj.weight", 2U * q, h, s + 2U);
      add_lin(p + ".self_attn.k_proj.weight", kv, h, s + 3U);
      add_lin(p + ".self_attn.v_proj.weight", kv, h, s + 4U);
      add_bf16(p + ".self_attn.q_norm.weight", {hd}, s + 5U, 0.0f, 0.03f);
      add_bf16(p + ".self_attn.k_norm.weight", {hd}, s + 6U, 0.0f, 0.03f);
      add_lin(p + ".self_attn.o_proj.weight", h, q, s + 7U);
    } else {
      add_lin(p + ".linear_attn.in_proj_qkv.weight", conv, h, s + 12U);
      add_lin(p + ".linear_attn.in_proj_z.weight", gv, h, s + 13U);
      add_lin(p + ".linear_attn.in_proj_a.weight", gv_heads, h, s + 14U);
      add_lin(p + ".linear_attn.in_proj_b.weight", gv_heads, h, s + 15U);
      add_bf16(p + ".linear_attn.conv1d.weight", {conv, 1U, 4U}, s + 16U, 0.0f, 0.05f);
      add_bf16(p + ".linear_attn.A_log", {gv_heads}, s + 17U, 0.4f, 0.05f);
      add_bf16(p + ".linear_attn.dt_bias", {gv_heads}, s + 18U, -1.2f, 0.05f);
      add_bf16(p + ".linear_attn.norm.weight", {gdn_vd}, s + 19U, 1.0f, 0.03f);
      add_lin(p + ".linear_attn.out_proj.weight", h, gv, s + 20U);
    }
    add_bf16(p + ".post_attention_layernorm.weight", {h}, s + 30U, 0.0f, 0.03f);
    add_lin(p + ".mlp.gate_proj.weight", f, h, s + 31U);
    add_lin(p + ".mlp.up_proj.weight", f, h, s + 32U);
    add_lin(p + ".mlp.down_proj.weight", h, f, s + 33U, "linear",
            use_mxfp4 && use_v2_fwht64);
  }

  std::vector<std::byte> payload;
  auto align_p = [&](std::size_t a) {
    while ((payload.size() % a) != 0U) payload.push_back(std::byte{0});
  };
  Json j_objs = Json::array(), j_binds = Json::array();
  for (const auto &o : objs) {
    align_p(256U);
    const U64 obj_off = payload.size();
    Json j_pls = Json::array();
    for (std::size_t pi = 0; pi < o.planes.size(); ++pi) {
      if (pi > 0U) align_p(256U);
      const U64 p_off = payload.size();
      const auto &pl = o.planes[pi];
      payload.insert(payload.end(), pl.second.begin(), pl.second.end());
      j_pls.push_back({{"alignment", 256},
                       {"bytes", pl.second.size()},
                       {"crc32", hex_crc32(artifact_crc32(pl.second))},
                       {"kind", pl.first},
                       {"offset", p_off},
                       {"sha256", artifact_sha256_hex(pl.second)}});
    }
    const U64 obj_bytes = payload.size() - obj_off;
    Json jo = {{"format", o.format},
               {"id", o.id},
               {"layout", o.layout},
               {"name", o.name},
               {"object_bytes", obj_bytes},
               {"object_offset", obj_off},
               {"object_sha256",
                artifact_sha256_hex(
                    std::span<const std::byte>(payload).subspan(obj_off, obj_bytes))},
               {"planes", std::move(j_pls)},
               {"shape", o.shape},
               {"source_tensors", {o.name}}};
    if (o.padded_k.has_value()) jo["padded_k"] = *o.padded_k;
    if (o.input_transform != "none") jo["input_transform"] = o.input_transform;
    j_objs.push_back(std::move(jo));
    if (!o.format.starts_with("direct_") && !o.binding_use.empty() &&
        o.shape.size() >= 2U) {
      j_binds.push_back({{"activation_policy", o.activation_policy},
                         {"format", o.format},
                         {"in_features", o.shape.back()},
                         {"leading_shape", Json::array()},
                         {"name", o.name},
                         {"object_id", o.id},
                         {"out_features", o.shape[o.shape.size() - 2U]},
                         {"use", o.binding_use}});
    }
  }
  const U32 ver = (use_mxfp4 && use_v2_fwht64) ? 2U : 1U;
  const U32 flags = (ver == 2U) ? kArtifactFlagFwht64 : 0U;
  Json manifest = {
      {"bindings", std::move(j_binds)},
      {"container_version", ver},
      {"model",
       {{"source_profile", use_mxfp4 ? "synth-mxfp4-v2" : "synth-oq4e"},
        {"source_repo", "test/synth"},
        {"source_revision", "r1"}}},
      {"objects", std::move(j_objs)},
      {"payload_bytes", payload.size()},
      {"schema", std::string(ver == 2U ? kArtifactSchemaV2 : kArtifactSchemaV1)},
      {"views", Json::array()},
  };
  const std::string mstr = manifest.dump();
  const auto mbytes = std::as_bytes(std::span<const char>(mstr.data(), mstr.size()));
  const U64 pay_off = (64ULL + mbytes.size() + 16383ULL) & ~16383ULL;
  std::vector<std::byte> file;
  for (char c : kArtifactMagic) file.push_back(static_cast<std::byte>(c));
  append_u32(file, ver);
  append_u32(file, 64U);
  append_u64(file, 64ULL);
  append_u64(file, mbytes.size());
  append_u64(file, pay_off);
  append_u64(file, payload.size());
  append_u32(file, U32(objs.size()));
  append_u32(file, artifact_crc32(mbytes));
  append_u32(file, artifact_crc32(std::span<const std::byte>(file.data(), 56U)));
  append_u32(file, flags);
  file.insert(file.end(), mbytes.begin(), mbytes.end());
  file.resize(static_cast<std::size_t>(pay_off), std::byte{0});
  file.insert(file.end(), payload.begin(), payload.end());
  const auto art_path = dir / "model.miv";
  write_binary_file(art_path, file);
  return {dir, art_path, load_qwen_model_config(dir / "config.json")};
}

// ============================================================================
// Test 5: QwenForward + NativeBackend E2E (W4A4, A4QuantKey, KEEP_QK_A8,
//         Sage2+MInference, CacheBlend, and Non-MXFP4 Safe Auto-Fallback)
// ============================================================================
Json test_qwen_forward_and_native_e2e(const MetalContext &ctx,
                                      const std::filesystem::path &libs) {
  TempDirGuard tmp{std::filesystem::temp_directory_path() /
                   ("omlx_m5_w4a4_test_" + std::to_string(::getpid()))};
  const auto mx_model =
      build_synthetic_hybrid_model(tmp.path / "mxfp4_v2", true, true);
  const auto oq_model =
      build_synthetic_hybrid_model(tmp.path / "oq4e_v1", false, false);

  const bool is_apple10 = w4a4_nax_runtime_eligible(ctx, libs);

  // 5a. QwenForward pure W4A4 vs BF16 + A4QuantKey deduplication + Hybrid KEEP_QK_A8
  bool saw_w4a4_route = false;
  bool saw_mode_b_swiglu = false;
  bool saw_gdn_mega_a4 = false;
  bool saw_hybrid_qk_a8 = false;
  bool saw_sage2_w4a4 = false;
  bool saw_blend_w4a4 = false;
  double logits_cos = 0.0;
  {
    auto art = ArtifactModel::open(mx_model.artifact_path, true);
    auto layers = build_qwen_layer_catalog(art, mx_model.config, 1024U, 64U);
    auto mat = plan_model_materialization(art, ctx, {.require_verified_payload = true});
    auto weights = materialize_model(art, std::move(mat), ctx);

    std::vector<U32> prompt(64U);
    for (U32 i = 0; i < 64U; ++i) prompt[i] = (i * 13U + 5U) % 256U;
    GpuSamplingOptions samp{.temperature = 0.0f, .top_k = 1U, .top_p = 1.0f};

    QwenForward fwd_bf16(
        ctx, weights, layers, mx_model.config, 1024U, libs, {}, false, false,
        false, {}, false, false, {}, LinearNumericPolicy::RowExactPortableV1,
        SamplingContract::LegacyRejectionV1, AttentionNumericPolicy::Legacy,
        true, "bf16", nullptr, false, 0U, nullptr, nullptr,
        QwenPrefillOptions{.linear_mode = PrefillLinearMode::BFloat16});
    const auto res_bf16 = fwd_bf16.run(prompt, samp, /*inspect_tensors=*/true);

    {
      ScopedEnv env_qk("OMLX_M5_W4A4_KEEP_QK_A8", "0");
      QwenForward fwd_a4(
          ctx, weights, layers, mx_model.config, 1024U, libs, {}, false, false,
          false, {}, false, false, {}, LinearNumericPolicy::RowExactPortableV1,
          SamplingContract::LegacyRejectionV1, AttentionNumericPolicy::Legacy,
          true, "bf16", nullptr, false, 0U, nullptr, nullptr,
          QwenPrefillOptions{.linear_mode = PrefillLinearMode::W4A4});
      const auto desc = fwd_a4.inspect_profile(64U, false);
      if (is_apple10) {
        U32 w4a4_count = 0, lm_head_w4a4 = 0;
        for (const auto &wr : desc.weights) {
          if (wr.family == "mxfp4_w4a4_nax_m5_v2") ++w4a4_count;
          if (wr.function ==
              "omlx_m5_mxfp4_w4a4_nax_swiglu_bm64_bn64_out64_sk1_fwht64_a4") {
            saw_mode_b_swiglu = true;
          }
          if (wr.object == "lm_head.weight" &&
              wr.family == "mxfp4_w4a4_nax_m5_v2") {
            ++lm_head_w4a4;
          }
        }
        saw_w4a4_route = (w4a4_count > 0U);
        saw_gdn_mega_a4 =
            (desc.gdn_numeric_contract_id ==
             "gdn.chunk64_bf16_nax.f32_state.mega_a4_v1");
        require(saw_w4a4_route, "Expected W4A4 weight routes on Apple10");
        require(saw_mode_b_swiglu,
                "Expected Mode B SwiGLU + FWHT64 + A4 on .miv v2");
        require(saw_gdn_mega_a4,
                "Expected GDN Stage 3 mega_a4_v1 contract in W4A4 mode");
        require(lm_head_w4a4 == 0U, "lm_head must NOT run in W4A4");
      }
      const auto res_a4 = fwd_a4.run(prompt, samp, /*inspect_tensors=*/true);
      if (is_apple10) {
        require(res_a4.prefill_delta.prefill_w4a4_tokens == 64ULL,
                "Expected prefill_w4a4_tokens == 64");
      }
      const std::size_t v = mx_model.config.vocab_size;
      logits_cos = cosine_similarity_f32(
          std::span<const float>(res_a4.logits).last(v),
          std::span<const float>(res_bf16.logits).last(v), "w4a4_e2e_logits");
      require(logits_cos >= 0.80,
              "W4A4 E2E logits cos_sim unexpectedly low: " +
                  std::to_string(logits_cos));
    }

    // 5b. Hybrid OMLX_M5_W4A4_KEEP_QK_A8=1
    if (is_apple10) {
      ScopedEnv env_qk("OMLX_M5_W4A4_KEEP_QK_A8", "1");
      QwenForward fwd_hyb(
          ctx, weights, layers, mx_model.config, 1024U, libs, {}, false, false,
          false, {}, false, false, {}, LinearNumericPolicy::RowExactPortableV1,
          SamplingContract::LegacyRejectionV1, AttentionNumericPolicy::Legacy,
          false, "bf16", nullptr, false, 0U, nullptr, nullptr,
          QwenPrefillOptions{.linear_mode = PrefillLinearMode::W4A4});
      const auto desc = fwd_hyb.inspect_profile(64U, false);
      bool q_a8 = false, k_a8 = false, v_a4 = false, o_a4 = false;
      for (const auto &wr : desc.weights) {
        if (wr.object.find("q_proj") != std::string::npos &&
            wr.family == "mxfp4_w4a8_nax_m5_v1")
          q_a8 = true;
        if (wr.object.find("k_proj") != std::string::npos &&
            wr.family == "mxfp4_w4a8_nax_m5_v1")
          k_a8 = true;
        if (wr.object.find("v_proj") != std::string::npos &&
            wr.family == "mxfp4_w4a4_nax_m5_v2")
          v_a4 = true;
        if (wr.object.find("o_proj") != std::string::npos &&
            wr.family == "mxfp4_w4a4_nax_m5_v2")
          o_a4 = true;
      }
      saw_hybrid_qk_a8 = q_a8 && k_a8 && v_a4 && o_a4;
      require(saw_hybrid_qk_a8,
              "OMLX_M5_W4A4_KEEP_QK_A8=1 failed to route Q/K->W4A8 and V/O->W4A4");
      const auto res_hyb = fwd_hyb.run(prompt, samp, false);
      require(res_hyb.prefill_delta.prefill_w4a4_tokens == 64ULL,
              "Hybrid KEEP_QK_A8 must report prefill_w4a4_tokens == 64");
    }

    // 5c. Orthogonality: W4A4 + Sage2 + MInference & W4A4 + CacheBlend
    if (is_apple10) {
      ScopedEnv env_qk("OMLX_M5_W4A4_KEEP_QK_A8", "0");
      QwenForward fwd_s2(
          ctx, weights, layers, mx_model.config, 1024U, libs, {}, false, false,
          false, {}, false, false, {}, LinearNumericPolicy::RowExactPortableV1,
          SamplingContract::LegacyRejectionV1, AttentionNumericPolicy::Legacy,
          false, "affine4_g128", nullptr, true, 0U, nullptr, nullptr,
          QwenPrefillOptions{
              .linear_mode = PrefillLinearMode::W4A4,
              .attention_mode = PrefillAttentionMode::Sage2,
              .sparse_minference = true,
              .cache_blend = true,
          });
      const auto r_s2 = fwd_s2.run(prompt, samp, false);
      saw_sage2_w4a4 = (r_s2.prefill_delta.prefill_w4a4_tokens == 64ULL &&
                        r_s2.prefill_delta.prefill_sage2_tokens == 64ULL);
      require(saw_sage2_w4a4,
              "W4A4 + Sage2 + MInference orthogonality failed");

      fwd_s2.reset();
      detail::ChunkBlendCache bcache(64ULL << 20);
      std::vector<U32> blend_prompt(160U);
      for (U32 i = 0; i < 160U; ++i) blend_prompt[i] = (i * 7U + 3U) % 256U;
      detail::BlendSegmentConfig scfg;
      scfg.l_min = 64U;
      scfg.l_max = 128U;
      const auto r_cb = fwd_s2.run_blend_prefill(
          blend_prompt, 0U, bcache, 0x12345678ULL, samp, false, &scfg);
      saw_blend_w4a4 = (r_cb.prefill_delta.prefill_w4a4_tokens >= 64ULL);
      require(saw_blend_w4a4, "W4A4 + CacheBlend orthogonality failed");
    }
  }

  // 5d. NativeBackend safe auto-fallback on non-MXFP4 (oQ4e) & active W4A4 on MXFP4 v2
  bool fallback_verified = false;
  bool native_mx_verified = false;
  {
    auto run_native = [&](const BuiltSynthModel &m) -> StatsReport {
      LoadOptions opts;
      opts.model_dir = m.dir.string();
      opts.max_context = 1024;
      opts.max_concurrency = 1;
      opts.kv_dtype = "bf16";
      opts.prefill_linear_mode = PrefillLinearMode::W4A4;
      opts.path_overrides["kernel_directory"] = libs.string();
      auto eng = Engine::create(opts);
      SubmitRequest req;
      req.prompt_tokens.resize(64U);
      for (U32 i = 0; i < 64U; ++i) req.prompt_tokens[i] = (i * 11U + 1U) % 256U;
      req.max_new_tokens = 1;
      req.sampling.temperature = 0.0f;
      req.sampling.top_k = 1;
      const auto id = eng->submit(req);
      bool done = false;
      while (!done) {
        const auto evs = eng->poll_events(std::chrono::milliseconds(5));
        for (const auto &ev : evs) {
          if (ev.request_id != id) continue;
          if (ev.kind == EventKind::finished) done = true;
          if (ev.kind == EventKind::failed) {
            throw std::runtime_error("NativeBackend failed: " + ev.error);
          }
        }
      }
      auto st = eng->stats();
      eng->shutdown();
      return st;
    };

    const auto st_oq = run_native(oq_model);
    require(st_oq.model_quant_format == "oq_affine",
            "oQ4e model_quant_format expected oq_affine, got " +
                st_oq.model_quant_format);
    require(st_oq.prefill_linear_mode == "w4a4",
            "requested prefill_linear_mode must stay w4a4");
    require(st_oq.prefill_linear_fallback,
            "oQ4e requesting w4a4 must set prefill_linear_fallback=true");
    require(st_oq.prefill_linear_fallback_reason == "requires_mxfp4_model",
            "oQ4e fallback reason mismatch: " +
                st_oq.prefill_linear_fallback_reason);
    require(st_oq.prefill_linear_mode_effective == (is_apple10 ? "w4a8" : "bf16"),
            "oQ4e effective linear mode mismatch: " +
                st_oq.prefill_linear_mode_effective);
    require(st_oq.prefill_w4a4_tokens == 0ULL,
            "oQ4e must have 0 w4a4 tokens");
    fallback_verified = true;

    const auto st_mx = run_native(mx_model);
    require(st_mx.model_quant_format == "mxfp4",
            "MXFP4 model_quant_format mismatch: " + st_mx.model_quant_format);
    if (is_apple10) {
      require(!st_mx.prefill_linear_fallback &&
                  st_mx.prefill_linear_mode_effective == "w4a4" &&
                  st_mx.prefill_w4a4_tokens >= 64ULL,
              "MXFP4 v2 NativeBackend W4A4 failed to execute W4A4");
    }
    native_mx_verified = true;
  }

  return {{"apple10_host", is_apple10},
          {"saw_w4a4_route", saw_w4a4_route},
          {"saw_mode_b_swiglu", saw_mode_b_swiglu},
          {"saw_gdn_mega_a4", saw_gdn_mega_a4},
          {"saw_hybrid_qk_a8", saw_hybrid_qk_a8},
          {"saw_sage2_w4a4", saw_sage2_w4a4},
          {"saw_blend_w4a4", saw_blend_w4a4},
          {"logits_cos_sim", logits_cos},
          {"non_mxfp4_fallback_verified", fallback_verified},
          {"native_mxfp4_v2_verified", native_mx_verified},
          {"passed", true}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    require(argc >= 2, "usage: omlx_m5_w4a4_nax_tests <metallib_dir>");
    const std::filesystem::path libs(argv[1]);
    auto ctx = MetalContext::create();

    Json report;
    report["device"] = ctx.capabilities().device_name;
    report["t1_sram_audit"] = test_sram_and_pso_audit(ctx, libs);
    report["t2_prologues"] = test_prologues_and_tail_zeroing(ctx, libs);
    report["t3_dense_linear_swiglu"] = test_dense_linear_and_swiglu_modes(ctx, libs);
    report["t4_moe_w4a4"] = test_moe_w4a4_modes(ctx, libs);
    report["t5_qwen_forward_e2e"] = test_qwen_forward_and_native_e2e(ctx, libs);
    report["passed"] = true;

    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << '\n';
    return 1;
  }
}
