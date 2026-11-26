// Module 2 Route B.1 (SageAttention2) & Route B.2 (MInference) Verification Gate:
//   1. Hardware Metal 4.1 cooperative_tensor identity & lane-coordinate probes
//      for INT8 16x32x128 and MXFP8 16x128x32 + <= 28.5 KiB SRAM assertions.
//   2. Inter-tile mu_K (+8/-8 bias) + 15x attention-sink outlier + valid_k=5
//      NaN/Inf page-tail trap regression test (0 NaNs, cos_sim >= 0.9992 on BOTH
//      MXFP8 and FP16 P x V, across both BF16 and Affine4Group128 KV!).
//   3. MInference Vertical-Slash retrieval gate at 16K and 64K with planted
//      vertical needle (pos=7777) & slash diagonals (lag=512 & non-local lag=2048)
//      verifying GQA max-pooling, active_indices retention, short-context bypass
//      (<4096), CacheBlend non-contiguous positions, and output cos_sim >= 0.995.

#include "attention_sage2.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

using namespace omlx::m5;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

namespace {

void require(bool ok, const std::string &msg) {
  if (!ok) {
    throw std::runtime_error("M2-SAGE2-MINF FAIL: " + msg);
  }
}

struct ScopedEnv {
  std::string key;
  std::optional<std::string> prev;
  ScopedEnv(std::string k, const char *v) : key(std::move(k)) {
    if (const char *old = std::getenv(key.c_str()))
      prev = std::string(old);
    if (v)
      setenv(key.c_str(), v, 1);
    else
      unsetenv(key.c_str());
  }
  ~ScopedEnv() {
    if (prev)
      setenv(key.c_str(), prev->c_str(), 1);
    else
      unsetenv(key.c_str());
  }
};

std::vector<std::uint16_t> read_u16(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 2ULL);
  std::vector<std::uint16_t> out(count);
  std::memcpy(out.data(), raw.data(), count * 2ULL);
  return out;
}

std::vector<U32> read_u32(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 4ULL);
  std::vector<U32> out(count);
  std::memcpy(out.data(), raw.data(), count * 4ULL);
  return out;
}

std::vector<float> read_f32(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 4ULL);
  std::vector<float> out(count);
  std::memcpy(out.data(), raw.data(), count * 4ULL);
  return out;
}

U64 affine4_page_bytes(U32 page_tokens, U32 kvh, U32 hd) {
  const U64 vec_bytes = U64(hd / 2u) + 16ull;
  const U64 head_bytes = U64(page_tokens) * vec_bytes + 32ull;
  return U64(kvh) * head_bytes;
}

QwenModelConfig make_cfg(U32 qh, U32 kvh, U32 hd = 256u) {
  QwenModelConfig c{};
  c.hidden_size = 5120;
  c.num_attention_heads = qh;
  c.num_key_value_heads = kvh;
  c.head_dim = hd;
  c.partial_rotary_factor = 0.25;
  c.rope_theta = 1000000.0;
  c.rms_norm_eps = 1e-6;
  return c;
}

void exec_profile(const MetalContext &ctx, Profile &p) {
  const std::array calls{
      MetalCommandInvocation{.plan = &p.plan,
                             .bindings = p.bindings,
                             .parameters = &p.parameters}};
  static_cast<void>(ctx.execute_batch(calls));
}

struct MetricSummary {
  double cos_sim = 0.0;
  double max_abs = 0.0;
  double rel_l2 = 0.0;
  U64 non_finite = 0;
};

MetricSummary compare_bf16_vs_f32(std::span<const std::uint16_t> actual,
                                  std::span<const float> ref) {
  require(actual.size() == ref.size(), "compare size mismatch");
  double dot = 0.0, na = 0.0, nr = 0.0, err2 = 0.0, max_abs = 0.0;
  U64 nf = 0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    const double a = bf16_to_float(actual[i]);
    const double r = ref[i];
    if (!std::isfinite(a) || !std::isfinite(r)) {
      ++nf;
      continue;
    }
    dot += a * r;
    na += a * a;
    nr += r * r;
    const double d = std::abs(a - r);
    max_abs = std::max(max_abs, d);
    err2 += d * d;
  }
  MetricSummary m{};
  m.non_finite = nf;
  m.max_abs = max_abs;
  m.cos_sim = (na > 0.0 && nr > 0.0) ? (dot / std::sqrt(na * nr)) : 1.0;
  m.rel_l2 = std::sqrt(err2 / std::max(nr, 1e-12));
  return m;
}

MetricSummary compare_bf16_vs_bf16(std::span<const std::uint16_t> a,
                                   std::span<const std::uint16_t> b) {
  std::vector<float> bf(b.size());
  for (std::size_t i = 0; i < b.size(); ++i) {
    bf[i] = bf16_to_float(b[i]);
  }
  return compare_bf16_vs_f32(a, bf);
}

// ============================================================================
// Test 1: Hardware Identity & Lane-Coordinate Probes + SRAM <= 28.5 KiB Gate
// ============================================================================
Json test_hardware_probes_and_sram(const MetalContext &ctx,
                                   const std::filesystem::path &libs) {
  ctx.register_library(
      {.key = "m2.sage2", .path = libs / "omlx_m5_attention_sage2.metallib"});
  ctx.register_library(
      {.key = "m2.minf",
       .path = libs / "omlx_m5_attention_minference.metallib"});

  // 1A. Verify SRAM <= 28.5 KiB (29,184 B) across all Sage2 & MInference kernels.
  // Note: MTL_SHADER_VALIDATION=1 instruments TG memory (~2x static TG bytes).
  const char *sv_env = std::getenv("MTL_SHADER_VALIDATION");
  const bool sv_on = sv_env != nullptr && std::string_view(sv_env) != "0" &&
                     !std::string_view(sv_env).empty();
  const U64 sram_cap = sv_on ? (29184ULL * 2ULL + 1024ULL) : 29184ULL;
  Json sram = Json::object();
  for (const U32 wm : {4u, 6u, 8u}) {
    for (const bool is_aff4 : {true, false}) {
      for (const Sage2PvMode pv : {Sage2PvMode::Mxfp8, Sage2PvMode::Fp16}) {
        for (const bool sp : {false, true}) {
          const char *fn = sage2_kernel_name(wm, is_aff4, pv, sp);
          const auto pi = ctx.warmup_pipeline(
              {.key = std::string("m2.s2.") + fn,
               .library_key = "m2.sage2",
               .function_name = fn,
               .variant = "v1"});
          require(pi.static_threadgroup_memory_bytes <= sram_cap,
                  std::string(fn) + " exceeded 28.5 KiB SRAM cap: " +
                      std::to_string(pi.static_threadgroup_memory_bytes));
          sram[fn] = pi.static_threadgroup_memory_bytes;
        }
      }
    }
  }
  for (const char *fn : {
           "omlx_m5_minference_probe_stats_affine4_g128_hd256",
           "omlx_m5_minference_probe_stats_bf16_hd256",
           "omlx_m5_minference_probe_score_affine4_g128_hd256",
           "omlx_m5_minference_probe_score_bf16_hd256",
           "omlx_m5_minference_select_topk",
           "omlx_m5_minference_build_indices",
       }) {
    const auto pi = ctx.warmup_pipeline(
        {.key = std::string("m2.mf.") + fn,
         .library_key = "m2.minf",
         .function_name = fn,
         .variant = "v1"});
    require(pi.static_threadgroup_memory_bytes <= sram_cap,
            std::string(fn) + " exceeded 28.5 KiB SRAM cap: " +
                std::to_string(pi.static_threadgroup_memory_bytes));
    if (std::string_view(fn) == "omlx_m5_minference_build_indices" && !sv_on) {
      require(pi.static_threadgroup_memory_bytes <= 2048ULL,
              "minference_build_indices exceeded 2 KiB SRAM cap: " +
                  std::to_string(pi.static_threadgroup_memory_bytes));
    }
    sram[fn] = pi.static_threadgroup_memory_bytes;
  }

  // 1B. INT8 16x32x128 Identity & Exact Matrix Multiply Probe
  std::vector<std::int8_t> q_i8(16 * 128);
  std::vector<std::int8_t> k_i8(32 * 128);
  for (int r = 0; r < 16; ++r)
    for (int d = 0; d < 128; ++d)
      q_i8[r * 128 + d] = std::int8_t(((r * 13 + d * 7 + 5) % 251) - 125);
  for (int c = 0; c < 32; ++c)
    for (int d = 0; d < 128; ++d)
      k_i8[c * 128 + d] = std::int8_t(((c * 19 + d * 11 + 3) % 251) - 125);

  const auto qb = ctx.upload(std::as_bytes(std::span(q_i8)));
  const auto kb = ctx.upload(std::as_bytes(std::span(k_i8)));
  auto sb = ctx.make_buffer(16 * 32 * sizeof(std::int32_t));
  auto cb = ctx.make_buffer(32 * 16 * 2 * sizeof(U32));
  Encoder e_i8(ctx, libs, 16u, "m2.probe.int8");
  auto &d_i8 = e_i8.emit("attention_sage2", "omlx_m5_sage2_probe_int8_16x32x128",
                         {.width = 32u}, {.width = 32u},
                         {{&qb, 0, qb.size()},
                          {&kb, 0, kb.size()},
                          {&sb, 0, sb.size()},
                          {&cb, 0, cb.size()}});
  e_i8.dynamic(d_i8, 4, words({0u}));
  auto p_i8 = e_i8.finish();
  exec_profile(ctx, p_i8);

  const auto s_raw = sb.read(0, sb.size());
  std::vector<std::int32_t> s_gpu(16 * 32);
  std::memcpy(s_gpu.data(), s_raw.data(), s_raw.size());
  for (int r = 0; r < 16; ++r) {
    for (int c = 0; c < 32; ++c) {
      std::int32_t ref = 0;
      for (int d = 0; d < 128; ++d) {
        ref += std::int32_t(q_i8[r * 128 + d]) * std::int32_t(k_i8[c * 128 + d]);
      }
      require(s_gpu[r * 32 + c] == ref,
              "INT8 16x32x128 probe mismatch at (" + std::to_string(r) + "," +
                  std::to_string(c) + "): got " +
                  std::to_string(s_gpu[r * 32 + c]) + " vs " +
                  std::to_string(ref));
    }
  }
  const auto coords = read_u32(cb, 32 * 16 * 2);
  std::vector<int> visit(16 * 32, 0);
  for (int idx = 0; idx < 32 * 16; ++idx) {
    const U32 r = coords[idx * 2];
    const U32 c = coords[idx * 2 + 1];
    require(r < 16u && c < 32u, "INT8 probe coordinate out of bounds");
    ++visit[r * 32 + c];
  }
  for (int x : visit) {
    require(x == 1, "INT8 cooperative_tensor coordinate coverage not bijective");
  }

  // 1C. MXFP8 16x128x32 Probe (multi-scale channels varying 100x!)
  std::vector<float> p_mat(16 * 32);
  std::vector<float> v_mat(32 * 128);
  for (int r = 0; r < 16; ++r) {
    float row_sum = 0.0f;
    for (int k = 0; k < 32; ++k) {
      const float x = std::exp(float(((r * 5 + k * 7) % 13) - 6) * 0.25f);
      p_mat[r * 32 + k] = x;
      row_sum += x;
    }
    for (int k = 0; k < 32; ++k) p_mat[r * 32 + k] /= row_sum;
  }
  for (int k = 0; k < 32; ++k) {
    for (int d = 0; d < 128; ++d) {
      const float ch_scale = (d % 16 == 0) ? 10.0f : ((d % 7 == 0) ? 0.1f : 1.0f);
      const float val = float(((k * 23 + d * 17 + 9) % 101) - 50) / 50.0f;
      v_mat[k * 128 + d] = val * ch_scale;
    }
  }
  const auto pb_mx = ctx.upload(std::as_bytes(std::span(p_mat)));
  const auto vb_mx = ctx.upload(std::as_bytes(std::span(v_mat)));
  auto ob_mx = ctx.make_buffer(16 * 128 * sizeof(float));
  Encoder e_mx(ctx, libs, 16u, "m2.probe.mxfp8");
  auto &d_mx = e_mx.emit("attention_sage2", "omlx_m5_sage2_probe_mxfp8_16x128x32",
                         {.width = 32u}, {.width = 32u},
                         {{&pb_mx, 0, pb_mx.size()},
                          {&vb_mx, 0, vb_mx.size()},
                          {&ob_mx, 0, ob_mx.size()}});
  e_mx.dynamic(d_mx, 3, words({0u}));
  auto p_mx = e_mx.finish();
  exec_profile(ctx, p_mx);

  const auto o_gpu = read_f32(ob_mx, 16 * 128);
  // Compute both exact FP32 reference and exact IEEE FP8-E4M3 x MXFP8-UE8M0 reference.
  auto quant_fp8_e4m3 = [](float x) -> float {
    const float s = (x < 0.0f) ? -1.0f : 1.0f;
    const float ax = std::abs(x);
    if (ax == 0.0f) return 0.0f;
    const int e = std::clamp(int(std::floor(std::log2(ax))), -6, 8);
    const float step = std::ldexp(1.0f, e - 3);
    const float q = std::min(448.0f, std::rint(ax / step) * step);
    return s * q;
  };
  std::vector<float> p_q(16 * 32), v_q(32 * 128);
  for (int i = 0; i < 16 * 32; ++i) {
    p_q[i] = quant_fp8_e4m3(p_mat[i] * 448.0f) / 448.0f;
  }
  for (int d = 0; d < 128; ++d) {
    float vmax = 0.0f;
    for (int k = 0; k < 32; ++k) vmax = std::max(vmax, std::abs(v_mat[k * 128 + d]));
    const int exp = int(std::ceil(std::log2(double(vmax) / 448.0)));
    const float sc = float(std::ldexp(1.0, exp));
    for (int k = 0; k < 32; ++k) {
      v_q[k * 128 + d] = quant_fp8_e4m3(v_mat[k * 128 + d] / sc) * sc;
    }
  }
  double dot = 0.0, ng = 0.0, nr = 0.0;
  double q_max_abs = 0.0;
  for (int r = 0; r < 16; ++r) {
    for (int d = 0; d < 128; ++d) {
      double ref = 0.0;
      double ref_q = 0.0;
      for (int k = 0; k < 32; ++k) {
        ref += double(p_mat[r * 32 + k]) * double(v_mat[k * 128 + d]);
        ref_q += double(p_q[r * 32 + k]) * double(v_q[k * 128 + d]);
      }
      const double g = o_gpu[r * 128 + d];
      require(std::isfinite(g), "MXFP8 probe produced non-finite value");
      dot += g * ref;
      ng += g * g;
      nr += ref * ref;
      q_max_abs = std::max(q_max_abs, std::abs(g - ref_q));
    }
  }
  const double mxfp8_cos = dot / std::sqrt(ng * nr);
  require(mxfp8_cos >= 0.998 && q_max_abs <= 1e-4,
          "MXFP8 16x128x32 probe mismatch: cos_sim=" + std::to_string(mxfp8_cos) +
              " q_max_abs=" + std::to_string(q_max_abs));

  return {{"int8_16x32x128_exact", true},
          {"int8_coords_bijective", true},
          {"mxfp8_16x128x32_cos_sim", mxfp8_cos},
          {"mxfp8_16x128x32_exact_q_max_abs", q_max_abs},
          {"all_sram_le_28_5kib", true},
          {"sram_bytes", sram}};
}

// ============================================================================
// Test 2: Inter-Tile mu_K (+8/-8 bias) + 15x Sink Outlier + valid_k=5 NaN Tail
// ============================================================================
Json test_mu_k_sink_nan_trap(const MetalContext &ctx,
                             const std::filesystem::path &libs, U32 qh, U32 kvh,
                             bool is_affine4) {
  constexpr U32 hd = 256u;
  constexpr U32 page_tokens = 64u;
  // History = 101, Tokens = 64 -> total_kv = 165 = 5*32 + 5 (final block valid_k = 5!)
  constexpr U32 history = 101u;
  constexpr U32 tokens = 64u;
  constexpr U32 total_kv = history + tokens; // 165 -> 3 pages (192 slots, 27 NaN tail!)
  constexpr U32 pages = (total_kv + page_tokens - 1u) / page_tokens;

  const auto cfg = make_cfg(qh, kvh, hd);
  const U32 q_stride = qh * hd * 2u;
  const U32 kv_width = kvh * hd;

  std::vector<U32> positions(tokens);
  for (U32 t = 0; t < tokens; ++t) positions[t] = history + t;
  std::vector<U32> page_table(pages);
  std::iota(page_table.begin(), page_table.end(), 0u);
  std::reverse(page_table.begin(), page_table.end());

  std::vector<std::uint16_t> query(U64(tokens) * q_stride);
  for (U32 t = 0; t < tokens; ++t) {
    for (U32 h = 0; h < qh; ++h) {
      const U64 base = U64(t) * q_stride + U64(h) * 2u * hd;
      for (U32 d = 0; d < hd; ++d) {
        // Post-RMSNorm scale (~O(1)) so softmax has realistic attention sharpness
        // while still stressing +8/-8 inter-tile mu_K shifts and 15x sink outlier.
        const float qv =
            float(((t * 17u + h * 29u + d * 13u + 7u) % 251u) - 125.0f) / 96.0f;
        const float gv =
            float(((t * 11u + h * 19u + d * 5u + 3u) % 211u) - 105.0f) / 128.0f;
        query[base + d] = float_to_bf16(qv);
        query[base + hd + d] = float_to_bf16(gv);
      }
    }
  }

  // Construct adversarial KV:
  //   - Tile 0 (pos 0..31):  +8.0f channel bias on d in [64..256), PLUS 15x outlier at sink pos=0!
  //   - Tile 1 (pos 32..63): -8.0f channel bias on d in [64..256)!
  //   - Tile 2 (pos 64..95): +6.0f bias; Tile 3: -6.0f bias; Tile 4: +4.0f; Tile 5 (valid_k=5): -4.0f
  std::vector<std::uint16_t> raw_k(U64(total_kv) * kv_width);
  std::vector<std::uint16_t> raw_v(U64(total_kv) * kv_width);
  for (U32 pos = 0; pos < total_kv; ++pos) {
    const U32 tile = pos / 32u;
    float tile_bias = 0.0f;
    if (tile == 0u) tile_bias = +8.0f;
    else if (tile == 1u) tile_bias = -8.0f;
    else if (tile == 2u) tile_bias = +6.0f;
    else if (tile == 3u) tile_bias = -6.0f;
    else if (tile == 4u) tile_bias = +4.0f;
    else tile_bias = -4.0f;

    for (U32 h = 0; h < kvh; ++h) {
      const U64 base = (U64(pos) * kvh + h) * hd;
      for (U32 d = 0; d < hd; ++d) {
        float kv =
            float(((pos * 23u + h * 37u + d * 11u + 19u) % 251u) - 125.0f) /
            128.0f;
        if (pos == 0u) {
          // 15x attention-sink outlier norm on token j=0!
          kv *= 15.0f;
        }
        if (d >= 64u) {
          // Alternating +8 / -8 persistent channel bias per tile!
          const float sign_d = ((d & 1u) == 0u) ? 1.0f : -1.0f;
          kv += tile_bias * sign_d * 0.25f;
          if (d >= 64u && d < 80u) {
            kv += tile_bias; // Exact +8.0 / -8.0 strong bias on 16 channels!
          }
        }
        // Structured + pseudo-random V activations (realistic LLM value distribution
        // without artificial 21x white-noise cancellation).
        const float v_struct =
            0.45f * std::sin(0.07f * float(d) + 0.3f * float(h) +
                             0.25f * float(tile));
        const float vv =
            v_struct +
            float(((pos * 13u + h * 41u + d * 7u + 29u) % 251u) - 125.0f) /
                160.0f;
        raw_k[base + d] = float_to_bf16(kv);
        raw_v[base + d] = float_to_bf16(vv);
      }
    }
  }

  const auto qb = ctx.upload(std::as_bytes(std::span(query)));
  const auto ptb = ctx.upload(std::as_bytes(std::span(page_table)));
  const auto posb = ctx.upload(std::as_bytes(std::span(positions)));
  const auto ref_of = [](const MetalBuffer &b) { return Ref{&b, 0u, b.size()}; };

  std::optional<MetalBuffer> kb_storage, vb_storage;
  if (!is_affine4) {
    // Poison unwritten tail slots (pos 165..191 -> slots 5..31 of kb=5!) with BF16 NaN!
    std::vector<std::uint16_t> paged_k(U64(pages) * page_tokens * kv_width,
                                       0x7fc0u);
    std::vector<std::uint16_t> paged_v(paged_k.size(), 0x7fc0u);
    for (U32 pos = 0; pos < total_kv; ++pos) {
      const U64 phys = U64(page_table[pos / page_tokens]) * page_tokens +
                       (pos % page_tokens);
      std::copy_n(raw_k.begin() + U64(pos) * kv_width, kv_width,
                  paged_k.begin() + phys * kv_width);
      std::copy_n(raw_v.begin() + U64(pos) * kv_width, kv_width,
                  paged_v.begin() + phys * kv_width);
    }
    // Also inject +Inf in a few unwritten tail slots to test Inf guarding too.
    const U64 tail_phys = U64(page_table[total_kv / page_tokens]) * page_tokens +
                          (total_kv % page_tokens);
    paged_k[tail_phys * kv_width] = 0x7f80u;
    paged_v[tail_phys * kv_width] = 0x7f80u;
    kb_storage.emplace(ctx.upload(std::as_bytes(std::span(paged_k))));
    vb_storage.emplace(ctx.upload(std::as_bytes(std::span(paged_v))));
  } else {
    const U64 pbytes = affine4_page_bytes(page_tokens, kvh, hd);
    // Poison entire slab with 0x7e (half NaN = 0x7e7e) -> unwritten meta scale/zero = NaN!
    std::vector<std::uint8_t> poison(U64(pages) * pbytes, 0x7eu);
    kb_storage.emplace(ctx.upload(std::as_bytes(std::span(poison))));
    vb_storage.emplace(ctx.upload(std::as_bytes(std::span(poison))));

    const auto k_in = ctx.upload(std::as_bytes(std::span(raw_k)));
    const auto v_in = ctx.upload(std::as_bytes(std::span(raw_v)));
    const auto app_params =
        words({qh, kvh, hd, total_kv, 0u, pages * page_tokens, page_tokens,
               pages, q_stride, hd * 2u, kv_width, kv_width, q_stride, hd * 2u});
    Encoder enc_app(ctx, libs, total_kv, "m2.trap.aff4.append");
    emit_paged_kv_append_affine4_g128(enc_app, cfg, total_kv, ref_of(k_in),
                                      ref_of(v_in), ref_of(*kb_storage),
                                      ref_of(*vb_storage), ref_of(ptb),
                                      app_params);
    auto prof_app = enc_app.finish();
    exec_profile(ctx, prof_app);
  }

  const auto params = words({qh, kvh, hd, tokens, history, pages * page_tokens,
                             page_tokens, pages, q_stride, hd * 2u, kv_width,
                             kv_width, q_stride, hd * 2u});

  // Reference exact attention output on the exact same paged KV slab:
  // Use Apple10 NAX Flash-Prefill on M5 (Apple10+), or portable exact SIMD32 /
  // FOSCAR NAX Chunk16 on M4 (Apple9, where Frag16x16 lane layout differs).
  const bool use_apple10_flash_ref =
      ctx.capabilities().device_name.starts_with("Apple M5") ||
      ctx.capabilities().profile.probe.apple_gpu_family >= 10;
  auto out_exact = ctx.make_buffer(U64(tokens) * qh * hd * 2ULL);
  Encoder enc_ex(ctx, libs, tokens, "m2.trap.exact");
  if (is_affine4) {
    if (use_apple10_flash_ref) {
      emit_paged_flash_prefill_affine4_g128(
          enc_ex, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_exact),
          ref_of(ptb), ref_of(posb), params);
    } else {
      emit_paged_attention_affine4_g128(
          enc_ex, cfg, tokens, /*split=*/false, total_kv, pages * page_tokens,
          pages, ref_of(qb), sub(ref_of(qb), hd * 2u), ref_of(*kb_storage),
          ref_of(*vb_storage), ref_of(out_exact), ref_of(ptb), ref_of(posb),
          Ref{}, Ref{}, Ref{}, params);
    }
  } else {
    if (use_apple10_flash_ref) {
      emit_paged_flash_prefill_nax(
          enc_ex, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_exact),
          ref_of(ptb), ref_of(posb), params);
    } else {
      emit_paged_causal_attention_exact_gqa(
          enc_ex, ctx, libs, cfg, tokens, /*strict_two_pass=*/false, ref_of(qb),
          sub(ref_of(qb), hd * 2u), ref_of(*kb_storage), ref_of(*vb_storage),
          ref_of(out_exact), ref_of(ptb), ref_of(posb), params);
    }
  }
  auto prof_ex = enc_ex.finish();
  exec_profile(ctx, prof_ex);
  const U64 out_elems = U64(tokens) * qh * hd;
  const auto exact_bits = read_u16(out_exact, out_elems);

  // Also compute independent FP32 reference for BF16 KV so we don't rely solely on GPU ref.
  std::vector<float> fp32_ref;
  if (!is_affine4) {
    fp32_ref.resize(out_elems, 0.0f);
    for (U32 tok = 0; tok < tokens; ++tok) {
      const U32 qpos = positions[tok];
      for (U32 h = 0; h < qh; ++h) {
        const U32 kvh_i = h / (qh / kvh);
        std::vector<double> sc(qpos + 1u);
        double mx = -INFINITY;
        for (U32 p = 0; p <= qpos; ++p) {
          double dot = 0.0;
          const U64 qb_off = U64(tok) * q_stride + U64(h) * 2u * hd;
          const U64 kb_off = (U64(p) * kvh + kvh_i) * hd;
          for (U32 d = 0; d < hd; ++d) {
            dot += double(bf16_to_float(query[qb_off + d])) *
                   double(bf16_to_float(raw_k[kb_off + d]));
          }
          sc[p] = dot / std::sqrt(double(hd));
          mx = std::max(mx, sc[p]);
        }
        double denom = 0.0;
        std::vector<double> acc(hd, 0.0);
        for (U32 p = 0; p <= qpos; ++p) {
          const double w = std::exp(sc[p] - mx);
          denom += w;
          const U64 vb_off = (U64(p) * kvh + kvh_i) * hd;
          for (U32 d = 0; d < hd; ++d) {
            acc[d] += w * double(bf16_to_float(raw_v[vb_off + d]));
          }
        }
        const U64 gb_off = U64(tok) * q_stride + U64(h) * 2u * hd + hd;
        const U64 ob_off = (U64(tok) * qh + h) * hd;
        for (U32 d = 0; d < hd; ++d) {
          const double g = bf16_to_float(query[gb_off + d]);
          fp32_ref[ob_off + d] = float((acc[d] / denom) / (1.0 + std::exp(-g)));
        }
      }
    }
  }

  Json modes = Json::object();
  for (const char *mode_str : {"fp16", "mxfp8"}) {
    ScopedEnv env_pv("OMLX_M5_SAGE2_PV_MODE", mode_str);
    auto out_s2 = ctx.make_buffer(out_elems * 2ULL);
    Encoder enc_s2(ctx, libs, tokens, std::string("m2.trap.sage2.") + mode_str);
    // Also verify short-context MInference bypass (total_kv = 165 < 4096)!
    emit_paged_flash_prefill_sage2(
        enc_s2, cfg, tokens, total_kv, is_affine4, /*sparse_minference=*/true,
        ref_of(qb), sub(ref_of(qb), hd * 2u), ref_of(*kb_storage),
        ref_of(*vb_storage), ref_of(out_s2), ref_of(ptb), ref_of(posb), params);
    auto prof_s2 = enc_s2.finish();
    require(prof_s2.plan.dispatch_count() == 1u,
            "Short-context (<4096) MInference bypass failed!");
    require(prof_s2.attention_kernel.find(mode_str) != std::string::npos,
            "OMLX_M5_SAGE2_PV_MODE=" + std::string(mode_str) +
                " did not select matching kernel: " + prof_s2.attention_kernel);
    exec_profile(ctx, prof_s2);

    const auto s2_bits = read_u16(out_s2, out_elems);
    const auto m_exact = compare_bf16_vs_bf16(s2_bits, exact_bits);
    require(m_exact.non_finite == 0u,
            "Sage2 produced NaN/Inf on valid_k=5 NaN-tail trap (" +
                std::string(mode_str) + ")");
    require(m_exact.cos_sim >= 0.9992,
            "Sage2 mu_K/sink trap cos_sim < 0.9992 (" + std::string(mode_str) +
                ", kv=" + (is_affine4 ? "affine4" : "bf16") +
                "): " + std::to_string(m_exact.cos_sim));
    double fp32_cos = m_exact.cos_sim;
    if (!is_affine4) {
      const auto m_fp32 = compare_bf16_vs_f32(s2_bits, fp32_ref);
      require(m_fp32.non_finite == 0u && m_fp32.cos_sim >= 0.9992,
              "Sage2 vs FP32 exact cos_sim < 0.9992 (" + std::string(mode_str) +
                  "): " + std::to_string(m_fp32.cos_sim));
      fp32_cos = m_fp32.cos_sim;
    }

    modes[mode_str] = {{"kernel", prof_s2.attention_kernel},
                       {"non_finite", m_exact.non_finite},
                       {"cos_sim_vs_exact", m_exact.cos_sim},
                       {"cos_sim_vs_fp32", fp32_cos},
                       {"max_abs", m_exact.max_abs},
                       {"short_ctx_minf_bypassed", true}};
  }

  return {{"qh", qh},
          {"kvh", kvh},
          {"kv", is_affine4 ? "affine4_g128" : "bf16"},
          {"valid_k_tail", 5u},
          {"modes", modes}};
}

// ============================================================================
// Test 3: MInference Synthetic Vertical Needle (pos=7777) & Slash (lag=512 &
//         non-local lag=2048) Retrieval Test at 16K & 64K
// ============================================================================
Json test_minference_needle_slash(const MetalContext &ctx,
                                  const std::filesystem::path &libs,
                                  U32 total_kv, U32 tokens, bool is_affine4,
                                  bool non_contiguous_pos = false) {
  constexpr U32 qh = 24u;
  constexpr U32 kvh = 4u;
  constexpr U32 repeat = 6u;
  constexpr U32 hd = 256u;
  constexpr U32 page_tokens = 64u;
  constexpr U32 kNeedlePos = 7777u;
  constexpr U32 kNeedleBlk = kNeedlePos / 32u; // 243
  constexpr U32 kSlashLagShort = 512u;         // delta = 16 (covered by N_local=32)
  constexpr U32 kSlashLagFar = 2048u;          // delta = 64 (> N_local=32 -> must be selected by Top-K_S!)

  const auto cfg = make_cfg(qh, kvh, hd);
  const U32 span = non_contiguous_pos ? (tokens * 2u) : tokens;
  const U32 history = total_kv - span;
  const U32 pages = (total_kv + page_tokens - 1u) / page_tokens;
  const U32 q_stride = qh * hd * 2u;
  const U32 kv_width = kvh * hd;

  std::vector<U32> positions(tokens);
  for (U32 t = 0; t < tokens; ++t) {
    positions[t] = non_contiguous_pos ? (history + t * 2u) : (history + t);
  }
  std::vector<U32> page_table(pages);
  std::iota(page_table.begin(), page_table.end(), 0u);
  std::reverse(page_table.begin(), page_table.end());

  // Construct orthonormal template directions for:
  //   - Vertical needle (aligned with query head g=2 ONLY -> tests GQA max_g pooling!)
  //   - Far slash lag=2048 (aligned with query head g=4 ONLY -> tests GQA max_g pooling + Top-K_S!)
  //   - Short slash lag=512 (aligned with query head g=1)
  //   - Attention sink pos < 4 (aligned with heads g=0,3,5 -> tests mandatory N_sink=4!)
  std::array<float, hd> dir_needle{}, dir_slash_far{}, dir_slash_short{},
      dir_sink{};
  for (U32 d = 0; d < hd; ++d) {
    dir_needle[d] = (d < 64u) ? ((d & 1u) ? 0.5f : -0.5f) : 0.0f;
    dir_slash_far[d] = (d >= 64u && d < 128u) ? ((d & 2u) ? 0.5f : -0.5f) : 0.0f;
    dir_slash_short[d] = (d >= 128u && d < 192u) ? ((d & 1u) ? 0.5f : -0.5f) : 0.0f;
    dir_sink[d] = (d >= 192u) ? ((d & 2u) ? 0.5f : -0.5f) : 0.0f;
  }

  std::vector<std::uint16_t> query(U64(tokens) * q_stride);
  for (U32 t = 0; t < tokens; ++t) {
    for (U32 h = 0; h < qh; ++h) {
      const U32 g = h % repeat;
      const U64 base = U64(t) * q_stride + U64(h) * 2u * hd;
      for (U32 d = 0; d < hd; ++d) {
        float qv =
            float(int((t * 13u + h * 7u + d * 11u) % 65u) - 32) / 256.0f;
        if (g == 2u) qv += 3.5f * dir_needle[d];
        else if (g == 4u) qv += 3.5f * dir_slash_far[d];
        else if (g == 1u) qv += 3.0f * dir_slash_short[d];
        else qv += 3.5f * dir_sink[d];
        query[base + d] = float_to_bf16(qv);
        query[base + hd + d] = float_to_bf16(0.5f);
      }
    }
  }

  std::vector<std::uint16_t> raw_k(U64(total_kv) * kv_width);
  std::vector<std::uint16_t> raw_v(U64(total_kv) * kv_width);
  for (U32 pos = 0; pos < total_kv; ++pos) {
    const bool is_needle = (pos == kNeedlePos);
    const bool is_slash_far =
        (pos + kSlashLagFar >= history && pos + kSlashLagFar < total_kv);
    const bool is_slash_short =
        (pos + kSlashLagShort >= history && pos + kSlashLagShort < total_kv);
    for (U32 h = 0; h < kvh; ++h) {
      const U64 base = (U64(pos) * kvh + h) * hd;
      for (U32 d = 0; d < hd; ++d) {
        float kv =
            float(int((pos * 19u + h * 29u + d * 5u) % 65u) - 32) / 256.0f;
        float vv =
            float(int((pos * 7u + h * 17u + d * 13u) % 65u) - 32) / 256.0f;
        if (pos < 4u) {
          // Strong attention sink (realistic LLM streaming/sink pattern)
          kv += 4.5f * dir_sink[d];
          vv += (d >= 192u) ? 1.5f : -0.5f;
        }
        if (is_needle) {
          kv += 4.5f * dir_needle[d];
          vv = (d < 64u) ? 2.5f : -1.5f;
        }
        if (is_slash_far) {
          kv += 4.5f * dir_slash_far[d];
          vv += (d >= 64u && d < 128u) ? 2.0f : 0.0f;
        }
        if (is_slash_short) {
          kv += 4.0f * dir_slash_short[d];
          vv += (d >= 128u && d < 192u) ? 1.8f : 0.0f;
        }
        raw_k[base + d] = float_to_bf16(kv);
        raw_v[base + d] = float_to_bf16(vv);
      }
    }
  }

  const auto qb = ctx.upload(std::as_bytes(std::span(query)));
  const auto ptb = ctx.upload(std::as_bytes(std::span(page_table)));
  const auto posb = ctx.upload(std::as_bytes(std::span(positions)));
  const auto ref_of = [](const MetalBuffer &b) { return Ref{&b, 0u, b.size()}; };

  std::optional<MetalBuffer> kb_storage, vb_storage;
  if (!is_affine4) {
    std::vector<std::uint16_t> paged_k(U64(pages) * page_tokens * kv_width,
                                       0x7fc0u);
    std::vector<std::uint16_t> paged_v(paged_k.size(), 0x7fc0u);
    for (U32 pos = 0; pos < total_kv; ++pos) {
      const U64 phys = U64(page_table[pos / page_tokens]) * page_tokens +
                       (pos % page_tokens);
      std::copy_n(raw_k.begin() + U64(pos) * kv_width, kv_width,
                  paged_k.begin() + phys * kv_width);
      std::copy_n(raw_v.begin() + U64(pos) * kv_width, kv_width,
                  paged_v.begin() + phys * kv_width);
    }
    kb_storage.emplace(ctx.upload(std::as_bytes(std::span(paged_k))));
    vb_storage.emplace(ctx.upload(std::as_bytes(std::span(paged_v))));
  } else {
    const U64 pbytes = affine4_page_bytes(page_tokens, kvh, hd);
    kb_storage.emplace(ctx.make_buffer(U64(pages) * pbytes));
    vb_storage.emplace(ctx.make_buffer(U64(pages) * pbytes));
    const auto k_in = ctx.upload(std::as_bytes(std::span(raw_k)));
    const auto v_in = ctx.upload(std::as_bytes(std::span(raw_v)));
    const auto app_params =
        words({qh, kvh, hd, total_kv, 0u, pages * page_tokens, page_tokens,
               pages, q_stride, hd * 2u, kv_width, kv_width, q_stride, hd * 2u});
    Encoder enc_app(ctx, libs, total_kv, "m2.minf.aff4.append");
    emit_paged_kv_append_affine4_g128(enc_app, cfg, total_kv, ref_of(k_in),
                                      ref_of(v_in), ref_of(*kb_storage),
                                      ref_of(*vb_storage), ref_of(ptb),
                                      app_params);
    auto prof_app = enc_app.finish();
    exec_profile(ctx, prof_app);
  }

  const auto params = words({qh, kvh, hd, tokens, history, pages * page_tokens,
                             page_tokens, pages, q_stride, hd * 2u, kv_width,
                             kv_width, q_stride, hd * 2u});

  // 1. Exact dense reference (Apple10 Flash-NAX on M5; portable exact on Apple9 M4)
  const bool use_apple10_flash_ref =
      ctx.capabilities().device_name.starts_with("Apple M5") ||
      ctx.capabilities().profile.probe.apple_gpu_family >= 10;
  const U64 out_elems = U64(tokens) * qh * hd;
  auto out_dense = ctx.make_buffer(out_elems * 2ULL);
  Encoder enc_d(ctx, libs, tokens, "m2.minf.dense");
  if (is_affine4) {
    if (use_apple10_flash_ref) {
      emit_paged_flash_prefill_affine4_g128(
          enc_d, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_dense),
          ref_of(ptb), ref_of(posb), params);
    } else {
      emit_paged_attention_affine4_g128(
          enc_d, cfg, tokens, /*split=*/false, total_kv, pages * page_tokens,
          pages, ref_of(qb), sub(ref_of(qb), hd * 2u), ref_of(*kb_storage),
          ref_of(*vb_storage), ref_of(out_dense), ref_of(ptb), ref_of(posb),
          Ref{}, Ref{}, Ref{}, params);
    }
  } else {
    if (use_apple10_flash_ref) {
      emit_paged_flash_prefill_nax(
          enc_d, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_dense),
          ref_of(ptb), ref_of(posb), params);
    } else {
      emit_paged_causal_attention_exact_gqa(
          enc_d, ctx, libs, cfg, tokens, /*strict_two_pass=*/false, ref_of(qb),
          sub(ref_of(qb), hd * 2u), ref_of(*kb_storage), ref_of(*vb_storage),
          ref_of(out_dense), ref_of(ptb), ref_of(posb), params);
    }
  }
  auto prof_d = enc_d.finish();
  exec_profile(ctx, prof_d);
  const auto dense_bits = read_u16(out_dense, out_elems);

  // 2. Sage2 + MInference sparse path (allocate <= 2.26 MiB scratch!)
  const auto lo = plan_minference_scratch(tokens, qh, kvh, total_kv);
  require(lo.total_bytes <= 2370000ull,
          "MInference scratch exceeded 2.26 MiB budget: " +
              std::to_string(lo.total_bytes));
  auto scratch = ctx.make_buffer(lo.total_bytes);
  auto out_sp = ctx.make_buffer(out_elems * 2ULL);

  Encoder enc_sp(ctx, libs, tokens, "m2.minf.sparse");
  emit_paged_flash_prefill_sage2(
      enc_sp, cfg, tokens, total_kv, is_affine4, /*sparse_minference=*/true,
      ref_of(qb), sub(ref_of(qb), hd * 2u), ref_of(*kb_storage),
      ref_of(*vb_storage), ref_of(out_sp), ref_of(ptb), ref_of(posb), params,
      ref_of(scratch), Sage2PvMode::Fp16);
  auto prof_sp = enc_sp.finish();
  exec_profile(ctx, prof_sp);

  // Inspect active_counts & active_indices for every (kvh, q_blk)!
  const auto tier = select_sage2_tier(tokens, qh, kvh);
  const U32 n_q_blocks = (tokens + tier.bm_tok - 1u) / tier.bm_tok;
  const auto cnt_raw =
      scratch.read(lo.active_counts_off, lo.active_counts_bytes);
  const auto idx_raw =
      scratch.read(lo.active_indices_off, lo.active_indices_bytes);
  std::vector<U32> active_counts(U64(kvh) * lo.max_q_blocks);
  std::vector<std::uint16_t> active_indices(U64(kvh) * lo.max_q_blocks *
                                            lo.active_stride);
  std::memcpy(active_counts.data(), cnt_raw.data(),
              active_counts.size() * sizeof(U32));
  std::memcpy(active_indices.data(), idx_raw.data(),
              active_indices.size() * sizeof(std::uint16_t));

  U32 max_active_seen = 0;
  for (U32 h = 0; h < kvh; ++h) {
    for (U32 qb_i = 0; qb_i < n_q_blocks; ++qb_i) {
      const U64 slot = U64(h) * lo.max_q_blocks + qb_i;
      const U32 cnt = active_counts[slot];
      max_active_seen = std::max(max_active_seen, cnt);
      const U32 total_blks = total_kv / 32u;
      require(cnt > 0u && cnt < total_blks,
              "MInference did not sparsify blocks: cnt=" + std::to_string(cnt));

      const std::uint16_t *arr = active_indices.data() + slot * lo.active_stride;
      bool has_needle = false;
      bool has_slash_short = false;
      bool has_slash_far = false;
      const U32 q_pos0 = positions[qb_i * tier.bm_tok];
      const U32 expected_short_blk = (q_pos0 - kSlashLagShort) / 32u;
      const U32 expected_far_blk = (q_pos0 - kSlashLagFar) / 32u;

      for (U32 i = 0; i < cnt; ++i) {
        if (i > 0u) {
          require(arr[i] > arr[i - 1u],
                  "active_indices not strictly ascending!");
        }
        if (arr[i] == kNeedleBlk) has_needle = true;
        if (arr[i] == expected_short_blk) has_slash_short = true;
        if (arr[i] == expected_far_blk) has_slash_far = true;
      }
      // Verify mandatory sink blocks 0..3 retained!
      require(cnt >= 4u && arr[0] == 0u && arr[1] == 1u && arr[2] == 2u &&
                  arr[3] == 3u,
              "Mandatory N_sink=4 blocks missing!");
      if (!has_needle || !has_slash_far || !has_slash_short) {
        const auto vs_raw = scratch.read(lo.v_score_off, lo.v_score_bytes);
        const auto ds_raw = scratch.read(lo.d_score_off, lo.d_score_bytes);
        const auto tc_raw = scratch.read(lo.top_counts_off, lo.top_counts_bytes);
        const auto tv_raw = scratch.read(lo.top_v_idx_off, lo.top_v_idx_bytes);
        const auto ts_raw = scratch.read(lo.top_s_lag_off, lo.top_s_lag_bytes);
        const auto mg_raw = scratch.read(lo.probe_m_global_off, lo.probe_m_global_bytes);
        const auto iz_raw = scratch.read(lo.probe_inv_z_off, lo.probe_inv_z_bytes);
        std::vector<float> vs(kvh * lo.max_kv_blocks), ds(kvh * lo.max_kv_blocks);
        std::vector<U32> tc(kvh * 2);
        std::vector<std::uint16_t> tv(kvh * lo.top_stride), ts(kvh * lo.top_stride);
        std::vector<float> mg(64 * qh), iz(64 * qh);
        std::memcpy(vs.data(), vs_raw.data(), vs.size() * 4);
        std::memcpy(ds.data(), ds_raw.data(), ds.size() * 4);
        std::memcpy(tc.data(), tc_raw.data(), tc.size() * 4);
        std::memcpy(tv.data(), tv_raw.data(), tv.size() * 2);
        std::memcpy(ts.data(), ts_raw.data(), ts.size() * 2);
        std::memcpy(mg.data(), mg_raw.data(), mg.size() * 4);
        std::memcpy(iz.data(), iz_raw.data(), iz.size() * 4);
        std::cerr << "[minf-diag] h=" << h << " qb=" << qb_i << " cnt=" << cnt
                  << " top_v_cnt=" << tc[h*2] << " top_s_cnt=" << tc[h*2+1]
                  << " vs[243]=" << vs[h*lo.max_kv_blocks + 243]
                  << " vs[0]=" << vs[h*lo.max_kv_blocks + 0]
                  << " vs[447]=" << vs[h*lo.max_kv_blocks + 447]
                  << " ds[64]=" << ds[h*lo.max_kv_blocks + 64]
                  << " mg[0..5]=" << mg[0] << "," << mg[1] << "," << mg[2] << "," << mg[3] << "," << mg[4] << "," << mg[5]
                  << " iz[2]=" << iz[2]
                  << " top_v[0..7]=" << tv[h*lo.top_stride+0] << "," << tv[h*lo.top_stride+1] << "," << tv[h*lo.top_stride+2] << "," << tv[h*lo.top_stride+3]
                  << "," << tv[h*lo.top_stride+4] << "," << tv[h*lo.top_stride+5] << "," << tv[h*lo.top_stride+6] << "," << tv[h*lo.top_stride+7]
                  << " top_s[0..3]=" << ts[h*lo.top_stride+0] << "," << ts[h*lo.top_stride+1] << "," << ts[h*lo.top_stride+2] << "," << ts[h*lo.top_stride+3]
                  << '\n';
      }
      require(has_needle,
              "Vertical needle block 243 (pos=7777) missed at kvh=" +
                  std::to_string(h) + " q_blk=" + std::to_string(qb_i));
      require(has_slash_short,
              "Slash lag=512 block missed at kvh=" + std::to_string(h) +
                  " q_blk=" + std::to_string(qb_i));
      require(has_slash_far,
              "Non-local slash lag=2048 block missed at kvh=" +
                  std::to_string(h) + " q_blk=" + std::to_string(qb_i));
    }
  }

  const auto sp_bits = read_u16(out_sp, out_elems);
  const auto m_fp16 = compare_bf16_vs_bf16(sp_bits, dense_bits);
  require(m_fp16.non_finite == 0u && m_fp16.cos_sim >= 0.995,
          "MInference FP16 PV cos_sim < 0.995 at L=" +
              std::to_string(total_kv) + ": " + std::to_string(m_fp16.cos_sim));

  // Also test MXFP8 PV under MInference at this depth!
  auto out_sp_mx = ctx.make_buffer(out_elems * 2ULL);
  Encoder enc_sp_mx(ctx, libs, tokens, "m2.minf.sparse.mxfp8");
  emit_paged_flash_prefill_sage2(
      enc_sp_mx, cfg, tokens, total_kv, is_affine4, /*sparse_minference=*/true,
      ref_of(qb), sub(ref_of(qb), hd * 2u), ref_of(*kb_storage),
      ref_of(*vb_storage), ref_of(out_sp_mx), ref_of(ptb), ref_of(posb), params,
      ref_of(scratch), Sage2PvMode::Mxfp8);
  auto prof_sp_mx = enc_sp_mx.finish();
  exec_profile(ctx, prof_sp_mx);
  const auto sp_mx_bits = read_u16(out_sp_mx, out_elems);
  const auto m_mx = compare_bf16_vs_bf16(sp_mx_bits, dense_bits);
  require(m_mx.non_finite == 0u && m_mx.cos_sim >= 0.995,
          "MInference MXFP8 PV cos_sim < 0.995 at L=" +
              std::to_string(total_kv) + ": " + std::to_string(m_mx.cos_sim));

  return {{"total_kv", total_kv},
          {"tokens", tokens},
          {"kv", is_affine4 ? "affine4_g128" : "bf16"},
          {"scratch_bytes", lo.total_bytes},
          {"total_blocks", total_kv / 32u},
          {"max_active_blocks", max_active_seen},
          {"active_ratio", double(max_active_seen) / double(total_kv / 32u)},
          {"needle_pos_7777_retained", true},
          {"slash_lag_512_retained", true},
          {"slash_lag_2048_retained", true},
          {"fp16pv_cos_sim", m_fp16.cos_sim},
          {"mxfp8pv_cos_sim", m_mx.cos_sim}};
}

// ============================================================================
// Test 4: Reusable-Profile Dynamic-Token Patching (capacity=256 -> live=65, 19)
//         for MInference + SageAttention2 (Wave 2 Profile/ICB contract!)
// ============================================================================
Json test_minference_reusable_profile_patch(const MetalContext &ctx,
                                            const std::filesystem::path &libs) {
  constexpr U32 qh = 24u;
  constexpr U32 kvh = 4u;
  constexpr U32 repeat = 6u;
  constexpr U32 hd = 256u;
  constexpr U32 page_tokens = 64u;
  constexpr U32 capacity = 256u;
  constexpr U32 total_kv = 16384u;
  constexpr U32 kNeedlePos = 7777u;
  constexpr U32 kNeedleBlk = kNeedlePos / 32u; // 243
  constexpr U32 kSlashLagFar = 2048u;

  const auto cfg = make_cfg(qh, kvh, hd);
  const U32 pages = (total_kv + page_tokens - 1u) / page_tokens;
  const U32 q_stride = qh * hd * 2u;
  const U32 kv_width = kvh * hd;
  const U32 history = total_kv - capacity;

  std::vector<U32> page_table(pages);
  std::iota(page_table.begin(), page_table.end(), 0u);
  std::reverse(page_table.begin(), page_table.end());

  std::array<float, hd> dir_needle{}, dir_slash_far{}, dir_sink{};
  for (U32 d = 0; d < hd; ++d) {
    dir_needle[d] = (d < 64u) ? ((d & 1u) ? 0.5f : -0.5f) : 0.0f;
    dir_slash_far[d] =
        (d >= 64u && d < 128u) ? ((d & 2u) ? 0.5f : -0.5f) : 0.0f;
    dir_sink[d] = (d >= 192u) ? ((d & 2u) ? 0.5f : -0.5f) : 0.0f;
  }

  // Populate 16K Affine4 KV cache once.
  std::vector<std::uint16_t> raw_k(U64(total_kv) * kv_width);
  std::vector<std::uint16_t> raw_v(U64(total_kv) * kv_width);
  for (U32 pos = 0; pos < total_kv; ++pos) {
    const bool is_needle = (pos == kNeedlePos);
    const bool is_slash_far =
        (pos + kSlashLagFar >= history && pos + kSlashLagFar < total_kv);
    for (U32 h = 0; h < kvh; ++h) {
      const U64 base = (U64(pos) * kvh + h) * hd;
      for (U32 d = 0; d < hd; ++d) {
        float kv =
            float(int((pos * 19u + h * 29u + d * 5u) % 65u) - 32) / 256.0f;
        float vv =
            float(int((pos * 7u + h * 17u + d * 13u) % 65u) - 32) / 256.0f;
        if (pos < 4u) {
          kv += 4.5f * dir_sink[d];
          vv += (d >= 192u) ? 1.5f : -0.5f;
        }
        if (is_needle) {
          kv += 4.5f * dir_needle[d];
          vv = (d < 64u) ? 2.5f : -1.5f;
        }
        if (is_slash_far) {
          kv += 4.5f * dir_slash_far[d];
          vv += (d >= 64u && d < 128u) ? 2.0f : 0.0f;
        }
        raw_k[base + d] = float_to_bf16(kv);
        raw_v[base + d] = float_to_bf16(vv);
      }
    }
  }

  const auto ref_of = [](const MetalBuffer &b) {
    return Ref{&b, 0u, b.size()};
  };
  const auto ptb = ctx.upload(std::as_bytes(std::span(page_table)));
  const U64 pbytes = affine4_page_bytes(page_tokens, kvh, hd);
  auto kb = ctx.make_buffer(U64(pages) * pbytes);
  auto vb = ctx.make_buffer(U64(pages) * pbytes);
  {
    const auto k_in = ctx.upload(std::as_bytes(std::span(raw_k)));
    const auto v_in = ctx.upload(std::as_bytes(std::span(raw_v)));
    const auto app_params =
        words({qh, kvh, hd, total_kv, 0u, pages * page_tokens, page_tokens,
               pages, q_stride, hd * 2u, kv_width, kv_width, q_stride, hd * 2u});
    Encoder enc_app(ctx, libs, total_kv, "m2.minf.patch.append");
    emit_paged_kv_append_affine4_g128(enc_app, cfg, total_kv, ref_of(k_in),
                                      ref_of(v_in), ref_of(kb), ref_of(vb),
                                      ref_of(ptb), app_params);
    auto prof_app = enc_app.finish();
    exec_profile(ctx, prof_app);
  }

  auto qb = ctx.make_buffer(U64(capacity) * q_stride * 2ULL);
  auto posb = ctx.make_buffer(U64(capacity) * sizeof(U32));
  auto out_sp = ctx.make_buffer(U64(capacity) * qh * hd * 2ULL);

  const auto lo = plan_minference_scratch(capacity, qh, kvh, total_kv);
  auto scratch = ctx.make_buffer(lo.total_bytes);

  // Compile ONE reusable Profile at capacity=256!
  const auto cap_params = words({qh, kvh, hd, capacity, history,
                                 pages * page_tokens, page_tokens, pages,
                                 q_stride, hd * 2u, kv_width, kv_width,
                                 q_stride, hd * 2u});
  Encoder enc_sp(ctx, libs, capacity, "m2.minf.reusable.cap256");
  emit_paged_flash_prefill_sage2(
      enc_sp, cfg, capacity, total_kv, /*is_affine4=*/true,
      /*sparse_minference=*/true, ref_of(qb), sub(ref_of(qb), hd * 2u),
      ref_of(kb), ref_of(vb), ref_of(out_sp), ref_of(ptb), ref_of(posb),
      cap_params, ref_of(scratch), Sage2PvMode::Fp16);
  auto prof_sp = enc_sp.finish();

  // Verify BOTH MInference (mp_bytes) AND Sage2 (sp_bytes) registered patches!
  require(prof_sp.patches.size() >= 4u,
          "MInference + Sage2 profile missing dynamic (tok_word, pos_word) patches: " +
              std::to_string(prof_sp.patches.size()));

  const auto apply_patches = [](Profile &prof, U32 live_t, U32 live_pos) {
    for (const auto &p : prof.patches) {
      const U32 val = (p.position ? live_pos : live_t) * p.multiplier;
      std::memcpy(prof.parameter_data.data() + p.offset +
                      U64(p.word) * sizeof(U32),
                  &val, sizeof(U32));
    }
    prof.parameters.write("step", prof.parameter_data);
  };

  const bool use_apple10_flash_ref =
      ctx.capabilities().device_name.starts_with("Apple M5") ||
      ctx.capabilities().profile.probe.apple_gpu_family >= 10;
  const auto tier = select_sage2_tier(capacity, qh, kvh);

  Json runs = Json::array();
  for (const U32 live_tokens : {65u, 19u}) {
    // Poison tail [live_tokens..capacity) with BF16 NaN in query, 0xdeadbeef in
    // positions (OOB trap!), and 0xa5a5 sentinel in output!
    std::vector<std::uint16_t> query_buf(U64(capacity) * q_stride, 0x7fc0u);
    std::vector<U32> pos_buf(capacity, 0xdeadbeefu);
    for (U32 t = 0; t < live_tokens; ++t) {
      pos_buf[t] = history + t;
      for (U32 h = 0; h < qh; ++h) {
        const U32 g = h % repeat;
        const U64 base = U64(t) * q_stride + U64(h) * 2u * hd;
        for (U32 d = 0; d < hd; ++d) {
          float qv =
              float(int((t * 13u + h * 7u + d * 11u) % 65u) - 32) / 256.0f;
          if (g == 2u) qv += 3.5f * dir_needle[d];
          else if (g == 4u) qv += 3.5f * dir_slash_far[d];
          else qv += 3.5f * dir_sink[d];
          query_buf[base + d] = float_to_bf16(qv);
          query_buf[base + hd + d] = float_to_bf16(0.5f);
        }
      }
    }
    qb.write(std::as_bytes(std::span(query_buf)));
    posb.write(std::as_bytes(std::span(pos_buf)));
    std::vector<std::uint16_t> sentinel(U64(capacity) * qh * hd, 0xa5a5u);
    out_sp.write(std::as_bytes(std::span(sentinel)));

    // Patch & replay the SAME capacity=256 Profile for live_tokens!
    apply_patches(prof_sp, live_tokens, history);
    exec_profile(ctx, prof_sp);

    // Exact dense reference for live_tokens only.
    const U64 live_elems = U64(live_tokens) * qh * hd;
    auto out_dense = ctx.make_buffer(live_elems * 2ULL);
    const auto live_params = words({qh, kvh, hd, live_tokens, history,
                                    pages * page_tokens, page_tokens, pages,
                                    q_stride, hd * 2u, kv_width, kv_width,
                                    q_stride, hd * 2u});
    Encoder enc_d(ctx, libs, live_tokens, "m2.minf.patch.dense");
    if (use_apple10_flash_ref) {
      emit_paged_flash_prefill_affine4_g128(
          enc_d, cfg, live_tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(kb), ref_of(vb), ref_of(out_dense), ref_of(ptb), ref_of(posb),
          live_params);
    } else {
      emit_paged_attention_affine4_g128(
          enc_d, cfg, live_tokens, /*split=*/false, total_kv,
          pages * page_tokens, pages, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(kb), ref_of(vb), ref_of(out_dense), ref_of(ptb), ref_of(posb),
          Ref{}, Ref{}, Ref{}, live_params);
    }
    auto prof_d = enc_d.finish();
    exec_profile(ctx, prof_d);

    const auto full_sp_bits = read_u16(out_sp, U64(capacity) * qh * hd);
    const auto dense_bits = read_u16(out_dense, live_elems);

    // Verify tail [live_tokens..capacity) was NOT overwritten!
    for (U64 i = live_elems; i < full_sp_bits.size(); ++i) {
      require(full_sp_bits[i] == 0xa5a5u,
              "Patched-profile MInference+Sage2 wrote past live_tokens=" +
                  std::to_string(live_tokens));
    }

    std::vector<std::uint16_t> live_sp_bits(full_sp_bits.begin(),
                                            full_sp_bits.begin() + live_elems);
    const auto m = compare_bf16_vs_bf16(live_sp_bits, dense_bits);
    require(m.non_finite == 0u && m.cos_sim >= 0.995,
            "Patched-profile MInference+Sage2 cos_sim < 0.995 for live_tokens=" +
                std::to_string(live_tokens) + ": " + std::to_string(m.cos_sim));

    // Verify needle & far slash retained in active_indices for live q_blks.
    const U32 live_n_qblks = (live_tokens + tier.bm_tok - 1u) / tier.bm_tok;
    const auto cnt_raw =
        scratch.read(lo.active_counts_off, lo.active_counts_bytes);
    const auto idx_raw =
        scratch.read(lo.active_indices_off, lo.active_indices_bytes);
    std::vector<U32> active_counts(U64(kvh) * lo.max_q_blocks);
    std::vector<std::uint16_t> active_indices(U64(kvh) * lo.max_q_blocks *
                                              lo.active_stride);
    std::memcpy(active_counts.data(), cnt_raw.data(),
                active_counts.size() * sizeof(U32));
    std::memcpy(active_indices.data(), idx_raw.data(),
                active_indices.size() * sizeof(std::uint16_t));
    for (U32 h = 0; h < kvh; ++h) {
      for (U32 qb_i = 0; qb_i < live_n_qblks; ++qb_i) {
        const U64 slot = U64(h) * lo.max_q_blocks + qb_i;
        const U32 cnt = active_counts[slot];
        const std::uint16_t *arr =
            active_indices.data() + slot * lo.active_stride;
        bool has_needle = false, has_far = false;
        const U32 exp_far_blk = (pos_buf[qb_i * tier.bm_tok] - kSlashLagFar) / 32u;
        for (U32 i = 0; i < cnt; ++i) {
          if (i > 0u) require(arr[i] > arr[i - 1u], "Non-ascending active_indices");
          if (arr[i] == kNeedleBlk) has_needle = true;
          if (arr[i] == exp_far_blk) has_far = true;
        }
        require(cnt >= 4u && arr[0] == 0u && arr[3] == 3u,
                "Patched-profile sinks missing");
        require(has_needle && has_far,
                "Patched-profile missed needle or far slash at live_tokens=" +
                    std::to_string(live_tokens));
      }
    }

    runs.push_back({{"capacity", capacity},
                    {"live_tokens", live_tokens},
                    {"tail_untouched", true},
                    {"cos_sim", m.cos_sim},
                    {"non_finite", m.non_finite}});
  }

  return {{"patches_registered", prof_sp.patches.size()}, {"runs", runs}};
}

// ============================================================================
// Test 5: Wide-Span Scattered CacheBlend MInference (32K KV, single q_blk
//         spanning pos=100..32700 -> zero gap blowup, self-attn + far slash!)
// ============================================================================
Json test_minference_wide_span_cacheblend(const MetalContext &ctx,
                                          const std::filesystem::path &libs) {
  constexpr U32 qh = 24u;
  constexpr U32 kvh = 4u;
  constexpr U32 repeat = 6u;
  constexpr U32 hd = 256u;
  constexpr U32 page_tokens = 64u;
  constexpr U32 total_kv = 32768u;
  constexpr U32 tokens = 32u; // WM=6 -> bm_tok=16 -> q_blk 0 has 16 tokens!
  constexpr U32 kNeedlePos = 7777u;
  constexpr U32 kNeedleBlk = kNeedlePos / 32u; // 243
  constexpr U32 kSlashLagFar = 2048u;
  constexpr U32 kSelfFarPos = 32700u;
  constexpr U32 kSelfFarBlk = kSelfFarPos / 32u;                    // 1021
  constexpr U32 kFarSlashBlk = (kSelfFarPos - kSlashLagFar) / 32u;  // 957

  const auto cfg = make_cfg(qh, kvh, hd);
  const U32 pages = (total_kv + page_tokens - 1u) / page_tokens;
  const U32 q_stride = qh * hd * 2u;
  const U32 kv_width = kvh * hd;

  // q_blk 0 spans pos=100..107 (tok 0..7) AND pos=32693..32700 (tok 8..15)!
  // q_blk 1 spans pos=32701..32716 (tok 16..31).
  std::vector<U32> positions(tokens);
  for (U32 t = 0; t < 8u; ++t) positions[t] = 100u + t;
  for (U32 t = 8u; t < 16u; ++t) positions[t] = 32693u + (t - 8u);
  for (U32 t = 16u; t < 32u; ++t) positions[t] = 32701u + (t - 16u);
  require(positions[0] == 100u && positions[15] == kSelfFarPos,
          "Wide-span CacheBlend positions setup error");

  std::vector<U32> page_table(pages);
  std::iota(page_table.begin(), page_table.end(), 0u);
  std::reverse(page_table.begin(), page_table.end());

  std::array<float, hd> dir_needle{}, dir_slash_far{}, dir_self{}, dir_sink{};
  for (U32 d = 0; d < hd; ++d) {
    dir_needle[d] = (d < 64u) ? ((d & 1u) ? 0.5f : -0.5f) : 0.0f;
    dir_slash_far[d] =
        (d >= 64u && d < 128u) ? ((d & 2u) ? 0.5f : -0.5f) : 0.0f;
    dir_self[d] = (d >= 128u && d < 192u) ? ((d & 1u) ? 0.5f : -0.5f) : 0.0f;
    dir_sink[d] = (d >= 192u) ? ((d & 2u) ? 0.5f : -0.5f) : 0.0f;
  }

  std::vector<std::uint16_t> query(U64(tokens) * q_stride);
  for (U32 t = 0; t < tokens; ++t) {
    for (U32 h = 0; h < qh; ++h) {
      const U32 g = h % repeat;
      const U64 base = U64(t) * q_stride + U64(h) * 2u * hd;
      for (U32 d = 0; d < hd; ++d) {
        float qv =
            float(int((t * 13u + h * 7u + d * 11u) % 65u) - 32) / 256.0f;
        if (g == 2u) qv += 3.5f * dir_needle[d];
        else if (g == 4u) qv += 3.5f * dir_slash_far[d];
        else if (g == 1u) qv += 4.0f * dir_self[d];
        else qv += 3.5f * dir_sink[d];
        query[base + d] = float_to_bf16(qv);
        query[base + hd + d] = float_to_bf16(0.5f);
      }
    }
  }

  std::vector<std::uint16_t> raw_k(U64(total_kv) * kv_width);
  std::vector<std::uint16_t> raw_v(U64(total_kv) * kv_width);
  for (U32 pos = 0; pos < total_kv; ++pos) {
    const bool is_needle = (pos == kNeedlePos);
    const bool is_slash_far =
        (pos + kSlashLagFar >= 32693u && pos + kSlashLagFar <= 32716u);
    const bool is_self_far = (pos >= 32693u && pos <= 32716u);
    const bool is_self_near = (pos >= 100u && pos <= 107u);
    for (U32 h = 0; h < kvh; ++h) {
      const U64 base = (U64(pos) * kvh + h) * hd;
      for (U32 d = 0; d < hd; ++d) {
        float kv =
            float(int((pos * 19u + h * 29u + d * 5u) % 65u) - 32) / 256.0f;
        float vv =
            float(int((pos * 7u + h * 17u + d * 13u) % 65u) - 32) / 256.0f;
        if (pos < 4u) {
          kv += 4.5f * dir_sink[d];
          vv += (d >= 192u) ? 1.5f : -0.5f;
        }
        if (is_needle) {
          kv += 4.5f * dir_needle[d];
          vv = (d < 64u) ? 2.5f : -1.5f;
        }
        if (is_slash_far) {
          kv += 4.5f * dir_slash_far[d];
          vv += (d >= 64u && d < 128u) ? 2.2f : 0.0f;
        }
        if (is_self_far) {
          kv += 4.8f * dir_self[d];
          vv += (d >= 128u && d < 192u) ? 3.0f : -0.8f;
        } else if (is_self_near) {
          kv += 4.0f * dir_self[d];
          vv += (d >= 128u && d < 192u) ? -2.0f : 0.6f;
        }
        raw_k[base + d] = float_to_bf16(kv);
        raw_v[base + d] = float_to_bf16(vv);
      }
    }
  }

  const auto ref_of = [](const MetalBuffer &b) {
    return Ref{&b, 0u, b.size()};
  };
  const auto qb = ctx.upload(std::as_bytes(std::span(query)));
  const auto ptb = ctx.upload(std::as_bytes(std::span(page_table)));
  const auto posb = ctx.upload(std::as_bytes(std::span(positions)));
  const U64 pbytes = affine4_page_bytes(page_tokens, kvh, hd);
  auto kb = ctx.make_buffer(U64(pages) * pbytes);
  auto vb = ctx.make_buffer(U64(pages) * pbytes);
  {
    const auto k_in = ctx.upload(std::as_bytes(std::span(raw_k)));
    const auto v_in = ctx.upload(std::as_bytes(std::span(raw_v)));
    const auto app_params =
        words({qh, kvh, hd, total_kv, 0u, pages * page_tokens, page_tokens,
               pages, q_stride, hd * 2u, kv_width, kv_width, q_stride, hd * 2u});
    Encoder enc_app(ctx, libs, total_kv, "m2.minf.wide.append");
    emit_paged_kv_append_affine4_g128(enc_app, cfg, total_kv, ref_of(k_in),
                                      ref_of(v_in), ref_of(kb), ref_of(vb),
                                      ref_of(ptb), app_params);
    auto prof_app = enc_app.finish();
    exec_profile(ctx, prof_app);
  }

  const auto params = words({qh, kvh, hd, tokens, 0u, pages * page_tokens,
                             page_tokens, pages, q_stride, hd * 2u, kv_width,
                             kv_width, q_stride, hd * 2u});
  const U64 out_elems = U64(tokens) * qh * hd;
  auto out_dense = ctx.make_buffer(out_elems * 2ULL);
  const bool use_apple10_flash_ref =
      ctx.capabilities().device_name.starts_with("Apple M5") ||
      ctx.capabilities().profile.probe.apple_gpu_family >= 10;
  Encoder enc_d(ctx, libs, tokens, "m2.minf.wide.dense");
  if (use_apple10_flash_ref) {
    emit_paged_flash_prefill_affine4_g128(
        enc_d, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u), ref_of(kb),
        ref_of(vb), ref_of(out_dense), ref_of(ptb), ref_of(posb), params);
  } else {
    emit_paged_attention_affine4_g128(
        enc_d, cfg, tokens, /*split=*/false, total_kv, pages * page_tokens,
        pages, ref_of(qb), sub(ref_of(qb), hd * 2u), ref_of(kb), ref_of(vb),
        ref_of(out_dense), ref_of(ptb), ref_of(posb), Ref{}, Ref{}, Ref{},
        params);
  }
  auto prof_d = enc_d.finish();
  exec_profile(ctx, prof_d);
  const auto dense_bits = read_u16(out_dense, out_elems);

  const auto lo = plan_minference_scratch(tokens, qh, kvh, total_kv);
  auto scratch = ctx.make_buffer(lo.total_bytes);
  auto out_fp16 = ctx.make_buffer(out_elems * 2ULL);
  Encoder enc_fp16(ctx, libs, tokens, "m2.minf.wide.fp16");
  emit_paged_flash_prefill_sage2(
      enc_fp16, cfg, tokens, total_kv, /*is_affine4=*/true,
      /*sparse_minference=*/true, ref_of(qb), sub(ref_of(qb), hd * 2u),
      ref_of(kb), ref_of(vb), ref_of(out_fp16), ref_of(ptb), ref_of(posb),
      params, ref_of(scratch), Sage2PvMode::Fp16);
  auto prof_fp16 = enc_fp16.finish();
  exec_profile(ctx, prof_fp16);

  // Inspect q_blk 0 (the wide-span block spanning pos=100..32700)!
  const auto cnt_raw =
      scratch.read(lo.active_counts_off, lo.active_counts_bytes);
  const auto idx_raw =
      scratch.read(lo.active_indices_off, lo.active_indices_bytes);
  std::vector<U32> active_counts(U64(kvh) * lo.max_q_blocks);
  std::vector<std::uint16_t> active_indices(U64(kvh) * lo.max_q_blocks *
                                            lo.active_stride);
  std::memcpy(active_counts.data(), cnt_raw.data(),
              active_counts.size() * sizeof(U32));
  std::memcpy(active_indices.data(), idx_raw.data(),
              active_indices.size() * sizeof(std::uint16_t));

  U32 max_qblk0_active = 0u;
  for (U32 h = 0; h < kvh; ++h) {
    const U64 slot0 = U64(h) * lo.max_q_blocks + 0u;
    const U32 cnt = active_counts[slot0];
    max_qblk0_active = std::max(max_qblk0_active, cnt);
    require(cnt > 0u && cnt < lo.active_stride,
            "Wide-span CacheBlend q_blk 0 overflowed active_stride (gap blowup!): cnt=" +
                std::to_string(cnt));
    const std::uint16_t *arr = active_indices.data() + slot0 * lo.active_stride;
    bool has_self_far = false, has_needle = false, has_far_slash = false;
    for (U32 i = 0; i < cnt; ++i) {
      if (i > 0u) require(arr[i] > arr[i - 1u], "Non-ascending active_indices");
      if (arr[i] == kSelfFarBlk) has_self_far = true;
      if (arr[i] == kNeedleBlk) has_needle = true;
      if (arr[i] == kFarSlashBlk) has_far_slash = true;
    }
    require(cnt >= 4u && arr[0] == 0u && arr[3] == 3u,
            "Wide-span CacheBlend sinks missing");
    require(has_self_far,
            "Wide-span CacheBlend dropped self-attention block 1021 (pos=32700)!");
    require(has_needle, "Wide-span CacheBlend dropped needle block 243!");
    require(has_far_slash,
            "Wide-span CacheBlend dropped far slash block 957 for pos=32700!");
  }

  const auto m_fp16 = compare_bf16_vs_bf16(read_u16(out_fp16, out_elems), dense_bits);
  require(m_fp16.non_finite == 0u && m_fp16.cos_sim >= 0.995,
          "Wide-span CacheBlend FP16 PV cos_sim < 0.995: " +
              std::to_string(m_fp16.cos_sim));

  auto out_mx = ctx.make_buffer(out_elems * 2ULL);
  Encoder enc_mx(ctx, libs, tokens, "m2.minf.wide.mxfp8");
  emit_paged_flash_prefill_sage2(
      enc_mx, cfg, tokens, total_kv, /*is_affine4=*/true,
      /*sparse_minference=*/true, ref_of(qb), sub(ref_of(qb), hd * 2u),
      ref_of(kb), ref_of(vb), ref_of(out_mx), ref_of(ptb), ref_of(posb),
      params, ref_of(scratch), Sage2PvMode::Mxfp8);
  auto prof_mx = enc_mx.finish();
  exec_profile(ctx, prof_mx);
  const auto m_mx = compare_bf16_vs_bf16(read_u16(out_mx, out_elems), dense_bits);
  require(m_mx.non_finite == 0u && m_mx.cos_sim >= 0.995,
          "Wide-span CacheBlend MXFP8 PV cos_sim < 0.995: " +
              std::to_string(m_mx.cos_sim));

  return {{"total_kv", total_kv},
          {"q_blk0_span", "100..32700"},
          {"q_blk0_max_active_blocks", max_qblk0_active},
          {"active_stride", lo.active_stride},
          {"self_attn_blk_1021_retained", true},
          {"needle_blk_243_retained", true},
          {"far_slash_blk_957_retained", true},
          {"fp16pv_cos_sim", m_fp16.cos_sim},
          {"mxfp8pv_cos_sim", m_mx.cos_sim}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument(
          "usage: m2_sage2_minference_tests METALLIB_DIR");
    }
    const std::filesystem::path libs(argv[1]);
    const auto ctx = MetalContext::create();

    Json report;
    report["device"] = ctx.capabilities().device_name;
    report["hardware_probes"] = test_hardware_probes_and_sram(ctx, libs);

    Json trap_cases = Json::array();
    trap_cases.push_back(test_mu_k_sink_nan_trap(ctx, libs, 24u, 4u, false));
    trap_cases.push_back(test_mu_k_sink_nan_trap(ctx, libs, 24u, 4u, true));
    trap_cases.push_back(test_mu_k_sink_nan_trap(ctx, libs, 16u, 2u, false));
    trap_cases.push_back(test_mu_k_sink_nan_trap(ctx, libs, 16u, 2u, true));
    report["mu_k_sink_nan_trap"] = trap_cases;

    Json minf_cases = Json::array();
    minf_cases.push_back(
        test_minference_needle_slash(ctx, libs, 16384u, 128u, false));
    minf_cases.push_back(
        test_minference_needle_slash(ctx, libs, 16384u, 128u, true));
    minf_cases.push_back(
        test_minference_needle_slash(ctx, libs, 16384u, 128u, true, true));
    minf_cases.push_back(
        test_minference_needle_slash(ctx, libs, 65536u, 128u, true));
    report["minference_retrieval"] = minf_cases;

    report["reusable_profile_patch"] =
        test_minference_reusable_profile_patch(ctx, libs);
    report["wide_span_cacheblend"] =
        test_minference_wide_span_cacheblend(ctx, libs);

    report["passed"] = true;
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
