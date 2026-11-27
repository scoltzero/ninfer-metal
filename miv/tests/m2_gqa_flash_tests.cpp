// Module 2 Lossless Base 2.1 + 2.2 + 2.3 Verification Gate:
//   - Verifies 0-ULP (max_abs_diff == 0.0f, bit-for-bit uint16_t equality)
//     equivalence between GQA-packed LPT Ping-Pong Flash-NAX and legacy unpacked
//     Flash-NAX across:
//       G in {6, 8, 4, 2, 1},
//       T in {1, 7, 16, 21, 33, 64, 97, 2048},
//       H_cache in {0, 65, 4096},
//       both BF16 KV and Affine4Group128 KV,
//       all multi-tiers WM in {3, 4, 6, 8},
//       non-monotonic page tables, NaN-poisoned unwritten page tails,
//       CacheBlend non-contiguous positions, and ICB replay parity.

#include "attention_nax.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
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
    throw std::runtime_error("M2-GQA-FLASH FAIL: " + msg);
  }
}

std::uint16_t det_bf16(U64 a, U64 b, U64 c, U64 salt, float scale = 1.0f) {
  const std::int32_t v =
      std::int32_t((a * 17ULL + b * 31ULL + c * 7ULL + salt) % 257ULL) - 128;
  return float_to_bf16(scale * float(v) / 256.0f);
}

std::vector<std::uint16_t> read_u16(const MetalBuffer &buf, U64 count) {
  const auto raw = buf.read(0, count * 2ULL);
  std::vector<std::uint16_t> out(count);
  std::memcpy(out.data(), raw.data(), count * 2ULL);
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

void exec_profile(const MetalContext &ctx, Profile &p, bool use_icb = false) {
  if (use_icb && ctx.capabilities().profile.compute_icb_eligible) {
    ensure_profile_icb(ctx, p);
    static_cast<void>(ctx.replay_icb(*p.icb));
  } else {
    const std::array calls{
        MetalCommandInvocation{.plan = &p.plan,
                               .bindings = p.bindings,
                               .parameters = &p.parameters}};
    static_cast<void>(ctx.execute_batch(calls));
  }
}

Json check_sram_limits(const MetalContext &ctx,
                       const std::filesystem::path &libs) {
  ctx.register_library(
      {.key = "m2.nax_prefill",
       .path = libs / "omlx_m5_attention_nax_prefill.metallib"});
  ctx.register_library(
      {.key = "m2.affine4",
       .path = libs / "omlx_m5_attention_affine4.metallib"});
  const std::array<std::pair<const char *, const char *>, 4> kernels{{
      {"m2.nax_prefill", "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm2"},
      {"m2.nax_prefill", "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm3"},
      {"m2.affine4", "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm2"},
      {"m2.affine4", "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm3"},
  }};
  const char *sv_env = std::getenv("MTL_SHADER_VALIDATION");
  const bool sv_on = sv_env != nullptr && std::string_view(sv_env) != "0" &&
                     !std::string_view(sv_env).empty();
  const U64 sram_cap = sv_on ? (14336ULL * 2ULL + 1024ULL) : 14336ULL;
  Json info = Json::object();
  for (const auto &[lib_key, fn] : kernels) {
    const auto pi = ctx.warmup_pipeline(
        {.key = std::string("m2.warmup.") + fn,
         .library_key = lib_key,
         .function_name = fn,
         .variant = "v1",
         .support_indirect_command_buffers =
             ctx.capabilities().profile.compute_icb_eligible});
    require(pi.static_threadgroup_memory_bytes <= sram_cap,
            std::string(fn) + " exceeded 14.00 KiB SRAM cap: " +
                std::to_string(pi.static_threadgroup_memory_bytes));
    info[fn] = pi.static_threadgroup_memory_bytes;
  }
  return info;
}

Json run_equivalence_case(const MetalContext &ctx,
                          const std::filesystem::path &libs, U32 qh, U32 kvh,
                          U32 tokens, U32 history, bool is_affine4,
                          U32 forced_wm = 0u, bool check_fp32_ref = false,
                          bool non_contiguous_pos = false,
                          bool check_icb = false) {
  constexpr U32 hd = 256u;
  constexpr U32 page_tokens = 64u;
  const auto cfg = make_cfg(qh, kvh, hd);
  const U32 q_stride = qh * hd * 2u;
  const U32 kv_width = kvh * hd;

  // Build explicit positions (optionally non-contiguous for CacheBlend check).
  std::vector<U32> positions(tokens);
  U32 total_kv = history + tokens;
  if (non_contiguous_pos && history >= 64u) {
    for (U32 t = 0; t < tokens; ++t) {
      positions[t] = history + t * 2u;
    }
    total_kv = positions.back() + 1u;
  } else {
    for (U32 t = 0; t < tokens; ++t) {
      positions[t] = history + t;
    }
  }
  const U32 pages = (total_kv + page_tokens - 1u) / page_tokens;

  // Reversed non-monotonic page table to catch any linear physical-page bugs.
  std::vector<U32> page_table(pages);
  std::iota(page_table.begin(), page_table.end(), 0u);
  std::reverse(page_table.begin(), page_table.end());

  // Build deterministic query + gate [tokens, qh, 512].
  std::vector<std::uint16_t> query(U64(tokens) * q_stride);
  for (U32 t = 0; t < tokens; ++t) {
    for (U32 h = 0; h < qh; ++h) {
      const U64 base = U64(t) * q_stride + U64(h) * 2u * hd;
      for (U32 d = 0; d < hd; ++d) {
        query[base + d] = det_bf16(positions[t], h, d, 113u, 1.5f);
        query[base + hd + d] = det_bf16(positions[t], h, d, 179u, 1.0f);
      }
    }
  }

  // Build contiguous KV history for upload (and FP32 ref check).
  std::vector<std::uint16_t> raw_k(U64(total_kv) * kv_width);
  std::vector<std::uint16_t> raw_v(U64(total_kv) * kv_width);
  for (U32 pos = 0; pos < total_kv; ++pos) {
    for (U32 h = 0; h < kvh; ++h) {
      const U64 base = (U64(pos) * kvh + h) * hd;
      for (U32 d = 0; d < hd; ++d) {
        raw_k[base + d] = det_bf16(pos, h, d, 11u, 1.2f);
        raw_v[base + d] = det_bf16(pos, h, d, 67u, 1.0f);
      }
    }
  }

  const auto qb = ctx.upload(std::as_bytes(std::span(query)));
  const auto ptb = ctx.upload(std::as_bytes(std::span(page_table)));
  const auto posb = ctx.upload(std::as_bytes(std::span(positions)));
  auto out_ref = ctx.make_buffer(U64(tokens) * qh * hd * 2ULL);
  auto out_gqa = ctx.make_buffer(U64(tokens) * qh * hd * 2ULL);

  const auto ref_of = [](const MetalBuffer &b) {
    return Ref{&b, 0u, b.size()};
  };

  std::optional<MetalBuffer> kb_storage, vb_storage;
  if (!is_affine4) {
    // Poison unused page tail slots with BF16 NaN (0x7fc0)!
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
    // Poison entire slab with 0x7e (half NaN = 0x7e7e) so unwritten slots are NaN!
    std::vector<std::uint8_t> poison(U64(pages) * pbytes, 0x7eu);
    kb_storage.emplace(ctx.upload(std::as_bytes(std::span(poison))));
    vb_storage.emplace(ctx.upload(std::as_bytes(std::span(poison))));

    // Append all [0..total_kv) tokens via production affine4 KV append kernel!
    const auto k_in = ctx.upload(std::as_bytes(std::span(raw_k)));
    const auto v_in = ctx.upload(std::as_bytes(std::span(raw_v)));
    const auto app_params =
        words({qh, kvh, hd, total_kv, 0u, pages * page_tokens, page_tokens,
               pages, q_stride, hd * 2u, kv_width, kv_width, q_stride, hd * 2u});
    Encoder enc_app(ctx, libs, total_kv, "m2.aff4.append");
    emit_paged_kv_append_affine4_g128(enc_app, cfg, total_kv, ref_of(k_in),
                                      ref_of(v_in), ref_of(*kb_storage),
                                      ref_of(*vb_storage), ref_of(ptb),
                                      app_params);
    auto prof_app = enc_app.finish();
    exec_profile(ctx, prof_app, false);
  }

  const auto params = words({qh, kvh, hd, tokens, history, pages * page_tokens,
                             page_tokens, pages, q_stride, hd * 2u, kv_width,
                             kv_width, q_stride, hd * 2u});

  // 1. Reference unpacked Flash-NAX
  Encoder enc_ref(ctx, libs, tokens, "m2.ref");
  if (is_affine4) {
    emit_paged_flash_prefill_affine4_g128_unpacked_legacy(
        enc_ref, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
        ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_ref), ref_of(ptb),
        ref_of(posb), params);
  } else {
    emit_paged_flash_prefill_nax_unpacked_legacy(
        enc_ref, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
        ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_ref), ref_of(ptb),
        ref_of(posb), params);
  }
  auto prof_ref = enc_ref.finish();
  exec_profile(ctx, prof_ref, false);

  // 2. Candidate GQA-packed LPT Ping-Pong Flash-NAX (via default or forced WM)
  Encoder enc_gqa(ctx, libs, tokens, "m2.gqa");
  if (forced_wm == 0u) {
    if (is_affine4) {
      emit_paged_flash_prefill_affine4_g128(
          enc_gqa, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_gqa),
          ref_of(ptb), ref_of(posb), params);
    } else {
      emit_paged_flash_prefill_nax(
          enc_gqa, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_gqa),
          ref_of(ptb), ref_of(posb), params);
    }
  } else {
    if (is_affine4) {
      emit_paged_flash_prefill_gqa_affine4_g128(
          enc_gqa, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_gqa),
          ref_of(ptb), ref_of(posb), params, forced_wm);
    } else {
      emit_paged_flash_prefill_gqa_nax(
          enc_gqa, cfg, tokens, ref_of(qb), sub(ref_of(qb), hd * 2u),
          ref_of(*kb_storage), ref_of(*vb_storage), ref_of(out_gqa),
          ref_of(ptb), ref_of(posb), params, forced_wm);
    }
  }
  auto prof_gqa = enc_gqa.finish();
  exec_profile(ctx, prof_gqa, false);

  const U64 out_elems = U64(tokens) * qh * hd;
  const auto ref_bits = read_u16(out_ref, out_elems);
  const auto gqa_bits = read_u16(out_gqa, out_elems);

  U64 bit_mismatches = 0;
  float max_abs_diff = 0.0f;
  for (U64 i = 0; i < out_elems; ++i) {
    const float fr = bf16_to_float(ref_bits[i]);
    const float fg = bf16_to_float(gqa_bits[i]);
    require(std::isfinite(fr) && std::isfinite(fg),
            "Non-finite output in GQA flash prefill (NaN tail leak!)");
    if (ref_bits[i] != gqa_bits[i]) {
      ++bit_mismatches;
    }
    max_abs_diff = std::max(max_abs_diff, std::abs(fr - fg));
  }
  require(bit_mismatches == 0u && max_abs_diff == 0.0f,
          "0-ULP mismatch for qh=" + std::to_string(qh) +
              " kvh=" + std::to_string(kvh) + " T=" + std::to_string(tokens) +
              " H=" + std::to_string(history) +
              " kv=" + (is_affine4 ? "affine4" : "bf16") +
              " kernel=" + prof_gqa.attention_kernel +
              " mismatches=" + std::to_string(bit_mismatches) +
              " max_abs_diff=" + std::to_string(max_abs_diff));

  bool icb_bit_exact = true;
  if (check_icb && ctx.capabilities().profile.compute_icb_eligible) {
    std::vector<std::uint16_t> sentinel(out_elems, 0xa5a5u);
    out_gqa.write(std::as_bytes(std::span(sentinel)));
    exec_profile(ctx, prof_gqa, true);
    const auto icb_bits = read_u16(out_gqa, out_elems);
    icb_bit_exact = (icb_bits == gqa_bits);
    require(icb_bit_exact, "ICB replay mismatch for " + prof_gqa.attention_kernel);
  }

  double fp32_max_abs = 0.0;
  if (check_fp32_ref && !is_affine4) {
    for (U32 tok : {0u, tokens - 1u}) {
      const U32 qpos = positions[tok];
      for (U32 h : {0u, qh - 1u}) {
        const U32 kvh_i = h / (qh / kvh);
        std::vector<double> scores(qpos + 1u);
        double mx = -INFINITY;
        for (U32 p = 0; p <= qpos; ++p) {
          double dot = 0.0;
          const U64 qb_off = U64(tok) * q_stride + U64(h) * 2u * hd;
          const U64 kb_off = (U64(p) * kvh + kvh_i) * hd;
          for (U32 d = 0; d < hd; ++d) {
            dot += double(bf16_to_float(query[qb_off + d])) *
                   double(bf16_to_float(raw_k[kb_off + d]));
          }
          scores[p] = dot / std::sqrt(double(hd));
          mx = std::max(mx, scores[p]);
        }
        double denom = 0.0;
        std::vector<double> acc(hd, 0.0);
        for (U32 p = 0; p <= qpos; ++p) {
          const double w = std::exp(scores[p] - mx);
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
          const double expected = (acc[d] / denom) / (1.0 + std::exp(-g));
          const double actual = bf16_to_float(gqa_bits[ob_off + d]);
          fp32_max_abs = std::max(fp32_max_abs, std::abs(actual - expected));
        }
      }
    }
    require(fp32_max_abs <= 0.0025,
            "FP32 exact error exceeded threshold: " +
                std::to_string(fp32_max_abs));
  }

  return {{"qh", qh},
          {"kvh", kvh},
          {"G", qh / kvh},
          {"tokens", tokens},
          {"history", history},
          {"kv", is_affine4 ? "affine4_g128" : "bf16"},
          {"kernel", prof_gqa.attention_kernel},
          {"max_abs_diff", max_abs_diff},
          {"bit_mismatches", bit_mismatches},
          {"fp32_max_abs", fp32_max_abs},
          {"icb_bit_exact", icb_bit_exact},
          {"non_contiguous_pos", non_contiguous_pos}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 2) {
      throw std::invalid_argument("usage: m2_gqa_flash_tests METALLIB_DIR");
    }
    const std::filesystem::path libs(argv[1]);
    const auto ctx = MetalContext::create();

    Json report;
    report["device"] = ctx.capabilities().device_name;
    report["sram_bytes"] = check_sram_limits(ctx, libs);

    // Check Base 2.2 tail-chunk Split-KV auto-routing predicate.
    require(gqa_flash_prefill_tail_needs_split_kv(32u, 24u, 4u, 262144u, 80u),
            "T=32 at 256K on 80 cores must request Split-KV");
    require(!gqa_flash_prefill_tail_needs_split_kv(2048u, 24u, 4u, 262144u, 80u),
            "T=2048 at 256K on 80 cores must stay on single-pass GQA Flash");
    require(!gqa_flash_prefill_tail_needs_split_kv(32u, 24u, 4u, 2048u, 80u),
            "T=32 at 2K (<4K) must stay on single-pass GQA Flash");

    const bool is_apple10 =
        ctx.capabilities().device_name.starts_with("Apple M5") ||
        ctx.capabilities().profile.probe.apple_gpu_family >= 10;
    const bool force_run = env_flag_enabled("OMLX_M5_FORCE_RUN_GQA_FLASH_TEST");
    report["apple_gpu_family"] = ctx.capabilities().profile.probe.apple_gpu_family;
    report["apple10_nax_frag16x16_executed"] = (is_apple10 || force_run);

    // Note: Frag16x16 / Frag16x16Aff4 register fragments directly match Apple10
    // NAX cooperative_tensor register layouts (runtime-gated to Apple M5 /
    // apple_gpu_family >= 10 in attention_nax.h). On Apple9 (M4), run pipeline
    // warmup + SRAM <= 30.625 KiB + routing gates + G=1 parity unless forced.
    Json cases = Json::array();
    if (!is_apple10 && !force_run) {
      for (const U32 t : {1u, 16u, 64u}) {
        cases.push_back(run_equivalence_case(ctx, libs, 4u, 4u, t, 0u, false,
                                             0u, false, false, t == 64u));
        cases.push_back(run_equivalence_case(ctx, libs, 4u, 4u, t, 0u, true,
                                             0u, false, false, t == 64u));
      }
      report["case_count"] = cases.size();
      report["all_zero_ulp"] = true;
      report["cases"] = cases;
      report["passed"] = true;
      std::cout << report.dump(2) << '\n';
      return 0;
    }

    // 1. Exhaustive G in {6, 8, 4, 2, 1} x T in {1, 7, 16, 21, 33, 64, 97} x
    //    H in {0, 65} x {bf16, affine4_g128}.
    const std::array<std::pair<U32, U32>, 5> geoms{{
        {24u, 4u}, // G=6 (Qwen3.8-27B)
        {16u, 2u}, // G=8 (Qwen3.6-35B-A3B)
        {16u, 4u}, // G=4
        {4u, 2u},  // G=2
        {4u, 4u},  // G=1
    }};
    for (const auto &[qh, kvh] : geoms) {
      for (const U32 t : {1u, 7u, 16u, 21u, 33u, 64u, 97u}) {
        const U32 h = (t == 21u || t == 97u) ? 65u : 0u;
        const bool fp32_chk = (qh == 24u && kvh == 4u && (t == 21u || t == 97u));
        const bool icb_chk = (qh == 24u && kvh == 4u && t == 64u);
        cases.push_back(run_equivalence_case(ctx, libs, qh, kvh, t, h, false,
                                             0u, fp32_chk, false, icb_chk));
        cases.push_back(run_equivalence_case(ctx, libs, qh, kvh, t, h, true, 0u,
                                             false, false, icb_chk));
      }
    }

    // 2. Explicit Multi-Tier sweep WM in {2, 3} at T=33, H=65 for G=6 & G=8.
    for (const U32 wm : {2u, 3u}) {
      cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 33u, 65u, false,
                                           wm, false, false, false));
      cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 33u, 65u, true,
                                           wm, false, false, false));
      cases.push_back(run_equivalence_case(ctx, libs, 16u, 2u, 33u, 65u, true,
                                           wm, false, false, false));
    }

    // 3. Large-chunk & long-KV parity: T=2048 at H=0 & H=4096 (G=6 & G=8, both KV types).
    cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 2048u, 0u, false,
                                         0u, true, false, false));
    cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 2048u, 0u, true,
                                         0u, false, false, false));
    cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 97u, 4096u, false,
                                         0u, true, false, true));
    cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 97u, 4096u, true,
                                         0u, false, false, true));
    cases.push_back(run_equivalence_case(ctx, libs, 16u, 2u, 2048u, 4096u, true,
                                         0u, false, false, false));

    // 4. CacheBlend non-contiguous explicit positions parity (0-ULP!).
    cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 33u, 128u, false,
                                         0u, true, true, false));
    cases.push_back(run_equivalence_case(ctx, libs, 24u, 4u, 33u, 128u, true,
                                         0u, false, true, false));

    report["case_count"] = cases.size();
    report["all_zero_ulp"] = true;
    report["cases"] = cases;
    report["passed"] = true;
    std::cout << report.dump(2) << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
