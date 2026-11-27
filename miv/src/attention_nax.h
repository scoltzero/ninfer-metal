#pragma once

// W2-B inline attention dispatch helpers for qwen_forward.cpp:
// 1. P0-5: GQA + multi-row (BQ=1..8) threadgroup KV-shared exact attention
//    (0 ULP bit-identical to 1-SIMDgroup exact attention, 6x-32x less KV DRAM).
// 2. Blade #1 (#4808): Apple10 NAX FlashAttention-2 prefill (BQ=64, BK=32,
//    WN=2 head-dim split for head_dim=256) with fused Qwen sigmoid gate.
// 3. Blade #2 (#4659 + FOSCAR): KvStorageFormat::Affine4Group128 KV append +
//    ChunkRows=16 NAX attention (0 ULP AR == DFlash2 parity across single-pass
//    and 512-token split-KV).

#include "forward_plan.h"
#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/qwen_forward.h"

#include <array>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5::detail::forward {

enum class AttentionNumericPolicy {
  ExactSimd32V1,
  FlashNaxV1,
  Affine4FoscarNaxV1,
};

inline constexpr U32 kFlashNaxPrefillMinTokens = 32u;
inline constexpr U32 kMaxExactSplitKvPartitions = 256u;

inline bool env_flag_enabled(const char *name) {
  const char *val = std::getenv(name);
  if (!val)
    return false;
  const std::string_view sv(val);
  return sv == "1" || sv == "true" || sv == "TRUE" || sv == "on" || sv == "ON";
}

inline bool strict_two_pass_attention_enabled() {
  return env_flag_enabled("OMLX_M5_STRICT_TWO_PASS_ATTN");
}

inline U32 exact_attention_split_chunk_tokens(U32 max_context,
                                              U32 target_partitions = 0u) {
  const U32 max_parts =
      (target_partitions > 0u && target_partitions < 12u)
          ? std::clamp<U32>(target_partitions * 4u, 2u,
                            kMaxExactSplitKvPartitions)
          : kMaxExactSplitKvPartitions;
  const U32 total_max_sbs = (max_context + 511u) / 512u;
  return (total_max_sbs > max_parts)
             ? (((max_context + max_parts * 512u - 1u) / (max_parts * 512u)) *
                512u)
             : 512u;
}

inline U32 exact_attention_split_count(U32 context_envelope, U32 max_context,
                                       U32 target_partitions = 0u) {
  const U32 chunk_tokens =
      exact_attention_split_chunk_tokens(max_context, target_partitions);
  return (context_envelope + chunk_tokens - 1u) / chunk_tokens;
}

inline bool flash_nax_prefill_enabled_by_env() {
  if (env_flag_enabled("OMLX_M5_DISABLE_FLASH_NAX_PREFILL"))
    return false;
  return env_flag_enabled("OMLX_M5_FLASH_NAX_PREFILL");
}

inline bool flash_nax_prefill_runtime_eligible(
    const MetalContext &ctx, const std::filesystem::path &libs, U32 head_dim,
    U32 page_tokens = 64u) {
  if (env_flag_enabled("OMLX_M5_DISABLE_FLASH_NAX_PREFILL"))
    return false;
  if (head_dim != 256u && head_dim != 128u && head_dim != 64u)
    return false;
  if (page_tokens < 32u || (page_tokens % 32u) != 0u)
    return false;
  const auto &c = ctx.capabilities();
  const auto &p = c.profile.probe;
  return (c.device_name.starts_with("Apple M5") || p.apple_gpu_family >= 10) &&
         p.runtime_os_major >= 27 && p.metal4_compiler && p.tensor_api &&
         std::filesystem::is_regular_file(
             libs / "omlx_m5_attention_nax_prefill.metallib");
}

inline bool affine4_kv_runtime_eligible(
    const MetalContext &ctx, const std::filesystem::path &libs, U32 query_heads,
    U32 kv_heads, U32 head_dim, U32 page_tokens = 64u) {
  if (env_flag_enabled("OMLX_M5_DISABLE_AFFINE4_KV_NAX"))
    return false;
  if ((head_dim != 256u && head_dim != 128u) || page_tokens != 64u ||
      kv_heads == 0u || (query_heads % kv_heads) != 0u ||
      (query_heads / kv_heads) > 16u)
    return false;
  const auto &c = ctx.capabilities();
  const auto &p = c.profile.probe;
  // Planar attention_affine4.metal append/FOSCAR/causal kernels use portable
  // Metal 4 matmul2d + get_multidimensional_index (Apple9 M4 & Apple10 M5),
  // matching Sage2/MInference/CacheBlend planar KV layout. Apple10-only
  // Frag16x16 Flash-Prefill kernels remain gated by
  // flash_nax_prefill_runtime_eligible().
  (void)c;
  return p.runtime_os_major >= 27 && p.metal4_compiler && p.tensor_api &&
         std::filesystem::is_regular_file(
             libs / "omlx_m5_attention_affine4.metallib");
}

inline U32 exact_attention_gqa_threads_y(U32 capacity, U32 query_heads,
                                         U32 kv_heads, U32 head_dim,
                                         U32 x_groups = 1u,
                                         U32 min_threadgroups = 64u) {
  if (env_flag_enabled("OMLX_M5_DISABLE_GQA_EXACT_ATTENTION"))
    return 1u;
  if (head_dim != 256u || kv_heads == 0u || (query_heads % kv_heads) != 0u)
    return 1u;
  const U32 repeat = query_heads / kv_heads;
  if (repeat == 0u || repeat > 32u)
    return 1u;
  for (U32 bq : {8u, 4u, 2u, 1u}) {
    if ((capacity % bq) == 0u && repeat * bq <= 32u) {
      const U64 total_tgs =
          U64(capacity / bq) * U64(kv_heads) * U64(std::max(1u, x_groups));
      if (total_tgs >= U64(min_threadgroups)) {
        return repeat * bq;
      }
    }
  }
  return 1u;
}

inline U32 exact_attention_1pass_gqa_threads_y(
    const MetalContext &ctx, const std::filesystem::path &libs,
    const char *fn_name, U32 capacity, U32 query_heads, U32 kv_heads,
    U32 head_dim, U32 x_groups = 1u) {
  if (env_flag_enabled("OMLX_M5_DISABLE_GQA_EXACT_ATTENTION"))
    return 1u;
  if (head_dim != 256u || kv_heads == 0u || (query_heads % kv_heads) != 0u)
    return 1u;
  const U32 repeat = query_heads / kv_heads;
  if (repeat == 0u || repeat > 32u)
    return 1u;

  const std::string lib_key = "qwen.forward.library.attention_exact";
  ctx.register_library(
      {.key = lib_key, .path = libs / "omlx_m5_attention_exact.metallib"});
  const auto pinfo = ctx.warmup_pipeline(
      {.key = std::string("qwen.forward.") + fn_name,
       .library_key = lib_key,
       .function_name = fn_name,
       .variant = "fixed-qwen",
       .support_indirect_command_buffers =
           ctx.capabilities().profile.compute_icb_eligible});
  const U32 max_pipeline_sgs =
      U32(std::min<U64>(32ULL, pinfo.max_threads_per_threadgroup / 32ULL));
  const U32 max_sgs =
      (repeat <= 16u) ? std::min<U32>(16u, max_pipeline_sgs) : max_pipeline_sgs;
  const U32 min_threadgroups = (x_groups > 1u) ? 16u : 64u;

  for (U32 bq : {8u, 4u, 2u, 1u}) {
    if ((capacity % bq) == 0u && repeat * bq <= max_sgs) {
      const U64 total_tgs =
          U64(capacity / bq) * U64(kv_heads) * U64(std::max(1u, x_groups));
      if (total_tgs >= U64(min_threadgroups)) {
        return repeat * bq;
      }
    }
  }
  return 1u;
}

inline void emit_paged_causal_attention_exact_gqa(
    Encoder &e, const MetalContext &ctx, const std::filesystem::path &libs,
    const QwenModelConfig &c, U32 t, bool strict_two_pass, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, std::vector<std::byte> params) {
  const char *fn = strict_two_pass ? "omlx_m5_paged_causal_attention_exact"
                                   : "omlx_m5_paged_causal_attention_exact_1pass";
  const U32 tptg_y =
      strict_two_pass
          ? exact_attention_gqa_threads_y(t, U32(c.num_attention_heads),
                                          U32(c.num_key_value_heads),
                                          U32(c.head_dim), 1u)
          : exact_attention_1pass_gqa_threads_y(
                ctx, libs, fn, t, U32(c.num_attention_heads),
                U32(c.num_key_value_heads), U32(c.head_dim), 1u);
  auto &attn = e.emit(
      "attention_exact", fn,
      {.width = 32u, .height = U64(t) * c.num_attention_heads},
      {.width = 32u, .height = tptg_y},
      {query, gate, state_k, state_v, mixer, page_table, attention_positions});
  e.profile.attention_kernel = fn;
  e.profile.attention_grid = attn.grid;
  e.profile.attention_numeric_contract_id =
      strict_two_pass
          ? "paged.simd32.globalmax.ordered_softmax.f32_gate.v1"
          : "paged.simd32.online_sb512.ordered_softmax.f32_gate.v1";
  e.profile.attention_row_exact_eligible = true;
  e.dynamic(attn, 7, std::move(params), 3, 4);
}

inline void emit_paged_split_attention_exact_1pass_gqa(
    Encoder &e, const MetalContext &ctx, const std::filesystem::path &libs,
    const QwenModelConfig &c, U32 t, U32 splits, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table, Ref split_max,
    Ref split_sum, Ref split_output, Ref attention_positions,
    std::vector<std::byte> split_params,
    std::optional<U32> tok_word = 3u,
    std::optional<U32> pos_word = 4u) {
  const char *partial_fn = "omlx_m5_paged_attention_exact_1pass_partial";
  const U32 partial_tptg_y = exact_attention_1pass_gqa_threads_y(
      ctx, libs, partial_fn, t, U32(c.num_attention_heads),
      U32(c.num_key_value_heads), U32(c.head_dim), splits);
  auto &partial = e.emit(
      "attention_exact", partial_fn,
      {.width = U64(splits) * 32u, .height = U64(t) * c.num_attention_heads},
      {.width = 32u, .height = partial_tptg_y},
      {query, state_k, state_v, split_max, split_sum, split_output, page_table,
       attention_positions});
  e.profile.attention_kernel = partial_fn;
  e.profile.attention_grid = partial.grid;
  e.dynamic(partial, 8, split_params, tok_word, pos_word);

  auto &merge = e.emit(
      "attention_exact", "omlx_m5_paged_attention_exact_1pass_merge",
      {.width = 32u, .height = U64(t) * c.num_attention_heads}, {.width = 32u},
      {gate, split_max, split_sum, split_output, mixer});
  e.profile.attention_merge_kernel =
      "omlx_m5_paged_attention_exact_1pass_merge";
  e.profile.attention_merge_grid = merge.grid;
  e.profile.attention_numeric_contract_id =
      "paged.simd32.online_sb512.ordered_softmax.f32_gate.v1";
  e.profile.attention_row_exact_eligible = true;
  e.dynamic(merge, 5, std::move(split_params), tok_word, pos_word);
}

inline void emit_paged_split_attention_exact_gqa(
    Encoder &e, const QwenModelConfig &c, U32 t, U32 splits, Ref query,
    Ref gate, Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref split_max, Ref attention_positions,
    std::vector<std::byte> split_params,
    std::optional<U32> tok_word = 3u,
    std::optional<U32> pos_word = 4u) {
  const U32 max_tptg_y = exact_attention_gqa_threads_y(
      t, U32(c.num_attention_heads), U32(c.num_key_value_heads),
      U32(c.head_dim), splits);
  const U32 merge_tptg_y = exact_attention_gqa_threads_y(
      t, U32(c.num_attention_heads), U32(c.num_key_value_heads),
      U32(c.head_dim), 1u);
  auto &maximum = e.emit(
      "attention_exact", "omlx_m5_paged_attention_exact_max",
      {.width = U64(splits) * 32u, .height = U64(t) * c.num_attention_heads},
      {.width = 32u, .height = max_tptg_y},
      {query, state_k, split_max, page_table, attention_positions});
  e.profile.attention_kernel = "omlx_m5_paged_attention_exact_max";
  e.profile.attention_grid = maximum.grid;
  e.dynamic(maximum, 5, split_params, tok_word, pos_word);

  auto &ordered = e.emit(
      "attention_exact", "omlx_m5_paged_attention_exact_merge",
      {.width = 32u, .height = U64(t) * c.num_attention_heads},
      {.width = 32u, .height = merge_tptg_y},
      {query, gate, state_k, state_v, mixer, page_table, split_max,
       attention_positions});
  e.profile.attention_merge_kernel = "omlx_m5_paged_attention_exact_merge";
  e.profile.attention_merge_grid = ordered.grid;
  e.profile.attention_numeric_contract_id =
      "paged.simd32.globalmax.ordered_softmax.f32_gate.v1";
  e.profile.attention_row_exact_eligible = true;
  e.dynamic(ordered, 8, std::move(split_params), tok_word, pos_word);
}

struct GqaFlashPrefillTier {
  U32 wm = 2u;
  U32 tg_threads = 256u;
  U32 m_tile = 32u;
  const char *bf16_kernel =
      "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm2";
  const char *affine4_kernel =
      "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm2";
  const char *tier_tag = "WM2";
};

inline bool gqa_flash_prefill_eligible(U32 query_heads, U32 kv_heads,
                                       U32 head_dim) {
  if (env_flag_enabled("OMLX_M5_DISABLE_GQA_FLASH_PREFILL"))
    return false;
  if (head_dim != 256u || kv_heads == 0u || (query_heads % kv_heads) != 0u)
    return false;
  const U32 repeat = query_heads / kv_heads;
  return repeat >= 1u && repeat <= 16u;
}

inline GqaFlashPrefillTier select_gqa_flash_prefill_tier(
    U32 tokens, U32 query_heads, U32 kv_heads, U32 forced_wm = 0u) {
  const U32 repeat = (kv_heads > 0u && query_heads >= kv_heads)
                         ? (query_heads / kv_heads)
                         : 1u;
  U32 wm = forced_wm;
  if (wm == 0u) {
    if (const char *env = std::getenv("OMLX_M5_GQA_FLASH_TIER")) {
      const std::string_view sv(env);
      if (sv == "2" || sv == "S" || sv == "wm2" || sv == "WM2")
        wm = 2u;
      else if (sv == "3" || sv == "M" || sv == "L" || sv == "wm3" || sv == "WM3")
        wm = 3u;
    }
  }
  if (wm == 0u) {
    // WP3.2 BK=16 WN=4 (<= 14.00 KiB, 0 B spills, 2 TGs/core):
    // WM=2 uses 4 barriers/step (256 threads); WM=3 uses 48 rows (8x6 on 27B,
    // 6x8 on 35B). Default to WM=2 unless calibrated higher on M5 Ultra.
    (void)tokens;
    (void)repeat;
    wm = 2u;
  }
  if (wm != 2u && wm != 3u) {
    wm = 2u;
  }
  if (wm == 3u) {
    return {3u, 384u, 48u,
            "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm3",
            "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm3",
            "WM3"};
  }
  return {2u, 256u, 32u,
          "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_nax_hd256_wm2",
          "omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm2",
          "WM2"};
}

// Base 2.2 Wave 2 routing helper: detect short tail / HKVD chunks at long KV
// where single-pass GQA-packed Flash would launch < 2 * gpu_cores TGs.
inline bool gqa_flash_prefill_tail_needs_split_kv(
    U32 tokens, U32 query_heads, U32 kv_heads, U32 context_envelope,
    U32 gpu_cores) {
  if (context_envelope < 4096u || kv_heads == 0u || tokens == 0u ||
      query_heads < kv_heads)
    return false;
  const U32 repeat = query_heads / kv_heads;
  const auto tier = select_gqa_flash_prefill_tier(tokens, query_heads, kv_heads);
  const U64 m_total = U64(tokens) * U64(repeat);
  const U64 m_blocks = (m_total + U64(tier.m_tile) - 1ull) / U64(tier.m_tile);
  const U64 gqa_flash_tgs = m_blocks * U64(kv_heads);
  return gqa_flash_tgs < 2ull * U64(std::max<U32>(1u, gpu_cores));
}

inline void emit_paged_flash_prefill_nax_unpacked_legacy(
    Encoder &e, const QwenModelConfig &c, U32 t, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, std::vector<std::byte> params) {
  const U32 hd = U32(c.head_dim);
  const char *fn = nullptr;
  U32 tg_threads = 128u;
  if (hd == 256u) {
    fn = "omlx_m5_paged_flash_prefill_nax_hd256";
    tg_threads = 256u;
  } else if (hd == 128u) {
    fn = "omlx_m5_paged_flash_prefill_nax_hd128";
    tg_threads = 128u;
  } else if (hd == 64u) {
    fn = "omlx_m5_paged_flash_prefill_nax_hd64";
    tg_threads = 128u;
  } else {
    throw std::invalid_argument("FlashNaxV1 unsupported head_dim");
  }
  const U32 q_blocks = (t + 63u) / 64u;
  auto &attn = e.emit(
      "attention_nax_prefill", fn,
      {.width = tg_threads, .height = U64(q_blocks) * c.num_attention_heads},
      {.width = tg_threads},
      {query, gate, state_k, state_v, mixer, page_table, attention_positions});
  e.profile.attention_kernel = fn;
  e.profile.attention_grid = attn.grid;
  e.profile.attention_numeric_contract_id =
      "paged.nax_flash_v1.bq64_bk32_dsplit.f32_softmax_gate.v1";
  e.profile.attention_row_exact_eligible = false;
  e.dynamic(attn, 7, std::move(params), 3, 4);
}

inline void emit_paged_flash_prefill_gqa_nax(
    Encoder &e, const QwenModelConfig &c, U32 t, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, std::vector<std::byte> params,
    U32 forced_wm = 0u) {
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  if (c.head_dim != 256 || kvh == 0u || (qh % kvh) != 0u) {
    throw std::invalid_argument("GQA FlashNax requires hd256 and valid GQA");
  }
  const auto tier = select_gqa_flash_prefill_tier(t, qh, kvh, forced_wm);
  const U64 m_total = U64(t) * U64(qh / kvh);
  const U64 m_blocks = (m_total + U64(tier.m_tile) - 1ull) / U64(tier.m_tile);
  auto &attn = e.emit(
      "attention_nax_prefill", tier.bf16_kernel,
      {.width = tier.tg_threads, .height = m_blocks * U64(kvh)},
      {.width = tier.tg_threads},
      {query, gate, state_k, state_v, mixer, page_table, attention_positions});
  e.profile.attention_kernel = tier.bf16_kernel;
  e.profile.attention_grid = attn.grid;
  e.profile.attention_numeric_contract_id =
      "paged.nax_flash_v3.gqa_bk16_wn4.f32_softmax_gate.v1";
  e.profile.attention_row_exact_eligible = false;
  e.dynamic(attn, 7, std::move(params), 3, 4);
}

inline void emit_paged_flash_prefill_nax(
    Encoder &e, const QwenModelConfig &c, U32 t, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, std::vector<std::byte> params) {
  if (gqa_flash_prefill_eligible(U32(c.num_attention_heads),
                                 U32(c.num_key_value_heads),
                                 U32(c.head_dim))) {
    emit_paged_flash_prefill_gqa_nax(e, c, t, query, gate, state_k, state_v,
                                     mixer, page_table, attention_positions,
                                     std::move(params));
    return;
  }
  emit_paged_flash_prefill_nax_unpacked_legacy(
      e, c, t, query, gate, state_k, state_v, mixer, page_table,
      attention_positions, std::move(params));
}

inline void emit_paged_split_attention_bf16_frag48(
    Encoder &e, const QwenModelConfig &c, U32 t, U32 context_envelope,
    U32 max_context, U32 physical_pages, Ref query, Ref gate, Ref state_k,
    Ref state_v, Ref mixer, Ref page_table, Ref split_max, Ref split_sum,
    Ref split_output, Ref attention_positions) {
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  const U32 hd = U32(c.head_dim);
  const U32 q_width = qh * hd;
  const U32 kv_width = kvh * hd;
  const U32 chunk_tokens = (context_envelope <= 32768u) ? 1024u : 2048u;
  const U32 splits = (context_envelope + chunk_tokens - 1u) / chunk_tokens;
  const U64 partial_half_bytes = U64(t) * qh * splits * hd * 2u;
  if (!splits || U64(t) * qh * splits * 4u > split_max.bytes ||
      U64(t) * qh * splits * 4u > split_sum.bytes ||
      partial_half_bytes > split_output.bytes) {
    throw std::invalid_argument("BF16 Frag48 split workspace exceeded");
  }
  const Ref partial_half_ref{split_output.buffer, split_output.offset,
                             partial_half_bytes};
  e.profile.split_attention = true;
  e.profile.split_count = splits;
  e.profile.split_workspace_bytes =
      U64(t) * qh * splits * (8u + U64(hd) * 2u);
  auto split_params = words(
      {qh, kvh, hd, t, 0u, max_context, 64u, physical_pages, q_width * 2u,
       hd * 2u, kv_width, kv_width, q_width * 2u, hd * 2u, splits,
       chunk_tokens});
  auto &partial = e.emit(
      "attention_nax_prefill",
      "omlx_m5_paged_attention_bf16_frag48_split_partial_hd256",
      {.width = U64(splits) * 192u, .height = U64(kvh)}, {.width = 192u},
      {query, state_k, state_v, split_max, split_sum, partial_half_ref,
       page_table, attention_positions});
  e.profile.attention_kernel =
      "omlx_m5_paged_attention_bf16_frag48_split_partial_hd256";
  e.profile.attention_grid = partial.grid;
  e.dynamic(partial, 8, split_params, 3, 4);

  auto &merge = e.emit(
      "attention_nax_prefill",
      "omlx_m5_paged_attention_bf16_split_merge_half_4sg_hd256",
      {.width = 128u, .height = U64(t) * qh}, {.width = 128u},
      {gate, split_max, split_sum, partial_half_ref, mixer});
  e.profile.attention_merge_kernel =
      "omlx_m5_paged_attention_bf16_split_merge_half_4sg_hd256";
  e.profile.attention_merge_grid = merge.grid;
  e.profile.attention_numeric_contract_id =
      "paged.nax_bf16_frag48_split.half_par_merge.v1";
  e.profile.attention_row_exact_eligible = false;
  e.dynamic(merge, 5, std::move(split_params), 3, 4);
}

struct Affine4SplitScratchLayout {
  U64 partial_half_bytes = 0;
  U64 q_half_offset = 0;
  U64 q_half_bytes = 0;
  U64 q_sum_offset = 0;
  U64 q_sum_bytes = 0;
  U32 n_rblks = 0;
  U32 r_stride = 0;
  bool fits = false;
};

inline Affine4SplitScratchLayout affine4_split_scratch_layout(
    U32 t, U32 qh, U32 kvh, U32 splits, U64 split_output_bytes) {
  Affine4SplitScratchLayout lo{};
  if (t == 0u || kvh == 0u || (qh % kvh) != 0u || splits == 0u) {
    return lo;
  }
  const U32 repeat = qh / kvh;
  const U32 n_rblks = (t * repeat + 15u) / 16u;
  const U32 r_stride = n_rblks * 16u;
  const auto align256 = [](U64 x) { return (x + 255ull) & ~255ull; };
  lo.n_rblks = n_rblks;
  lo.r_stride = r_stride;
  lo.partial_half_bytes = U64(t) * qh * splits * 256ull * 2ull;
  lo.q_half_offset = align256(lo.partial_half_bytes);
  lo.q_half_bytes = U64(kvh) * 2ull * r_stride * 128ull * 2ull;
  lo.q_sum_offset = align256(lo.q_half_offset + lo.q_half_bytes);
  lo.q_sum_bytes = U64(kvh) * 2ull * r_stride * 4ull;
  lo.fits = (lo.q_sum_offset + lo.q_sum_bytes <= split_output_bytes);
  return lo;
}

struct BatchedAffine4SplitPlan {
  bool valid = false;
  U32 splits = 0;
  U32 split_tokens = 512u;
  U32 n_rblks = 0;
  U32 r_stride = 0;
  U64 partial_half_bytes = 0;
  U64 q_half_offset = 0;
  U64 q_half_bytes = 0;
  U64 q_sum_offset = 0;
  U64 q_sum_bytes = 0;
};

inline BatchedAffine4SplitPlan plan_batched_affine4_split(
    U32 total_m, U32 batch_size, U32 max_w, U32 qh, U32 kvh, U32 env,
    U64 split_max_bytes, U64 split_sum_bytes, U64 split_output_bytes) {
  BatchedAffine4SplitPlan plan{};
  if (total_m == 0u || batch_size < 2u || batch_size > 8u || max_w == 0u ||
      kvh == 0u || (qh % kvh) != 0u || env == 0u) {
    return plan;
  }
  const U32 repeat = qh / kvh;
  if (repeat == 0u || repeat > 16u) {
    return plan;
  }
  const U32 n_rblks = (max_w * repeat + 15u) / 16u;
  const U32 r_stride = n_rblks * 16u;
  const auto align256 = [](U64 x) { return (x + 255ull) & ~255ull; };
  const U64 q_half_bytes = U64(batch_size) * kvh * 2ull * r_stride * 128ull * 2ull;
  const U64 q_sum_bytes = U64(batch_size) * kvh * 2ull * r_stride * 4ull;
  for (U32 split_tokens :
       {512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u, 65536u}) {
    const U32 splits = std::max(1u, (env + split_tokens - 1u) / split_tokens);
    const U64 meta_bytes = U64(total_m) * qh * splits * 4ull;
    const U64 partial_half_bytes = U64(total_m) * qh * splits * 256ull * 2ull;
    const U64 q_half_offset = align256(partial_half_bytes);
    const U64 q_sum_offset = align256(q_half_offset + q_half_bytes);
    if (meta_bytes <= split_max_bytes && meta_bytes <= split_sum_bytes &&
        q_sum_offset + q_sum_bytes <= split_output_bytes) {
      plan.valid = true;
      plan.splits = splits;
      plan.split_tokens = split_tokens;
      plan.n_rblks = n_rblks;
      plan.r_stride = r_stride;
      plan.partial_half_bytes = partial_half_bytes;
      plan.q_half_offset = q_half_offset;
      plan.q_half_bytes = q_half_bytes;
      plan.q_sum_offset = q_sum_offset;
      plan.q_sum_bytes = q_sum_bytes;
      return plan;
    }
  }
  return plan;
}

inline void emit_batched_attention_affine4_g128(
    Encoder &e, const QwenModelConfig &c, U32 batch_size, U32 total_m,
    U32 max_w, bool all_w_eq_1, bool use_flash_prefill,
    const BatchedAffine4SplitPlan &split_plan, U32 max_context,
    U32 physical_pages, U32 q_tok_stride, U32 k_tok_stride, U32 v_tok_stride,
    float norm_weight_offset, Ref query, Ref gate, Ref key, Ref value,
    Ref q_norm, Ref k_norm, Ref state_k, Ref state_v, Ref mixer,
    Ref attention_positions, const std::array<Ref, 8> &pt,
    const std::array<U32, 8> &row_offs_8, const std::array<U32, 8> &widths_8,
    Ref split_max, Ref split_sum, Ref split_output) {
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  const U32 hd = U32(c.head_dim);
  const U32 repeat = qh / kvh;
  const U32 rot_dim = U32(std::llround(c.head_dim * c.partial_rotary_factor));
  const auto to_bytes = [](const std::vector<U32> &v) {
    const auto raw = std::as_bytes(std::span<const U32>(v));
    return std::vector<std::byte>(raw.begin(), raw.end());
  };

  std::vector<U32> post_w{qh,
                          kvh,
                          hd,
                          rot_dim,
                          total_m,
                          q_tok_stride,
                          hd * 2u,
                          k_tok_stride,
                          hd,
                          v_tok_stride,
                          batch_size,
                          word(float(c.rms_norm_eps)),
                          word(float(c.rope_theta)),
                          word(norm_weight_offset),
                          use_flash_prefill ? 0u : split_plan.r_stride};
  post_w.insert(post_w.end(), row_offs_8.begin(), row_offs_8.end());
  post_w.insert(post_w.end(), widths_8.begin(), widths_8.end());

  if (use_flash_prefill) {
    auto &post = e.emit(
        "attention_affine4",
        "omlx_m5_attention_qk_norm_rope_kv_append_batched_affine4_g128",
        {.width = 32u, .height = U64(total_m) * (qh + kvh)}, {.width = 32u},
        {query, key, value, q_norm, k_norm, state_k, state_v,
         attention_positions, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6],
         pt[7]});
    e.dynamic(post, 16, to_bytes(post_w));

    std::vector<U32> attn_w{qh,           kvh,          hd,
                            batch_size,   64u,          physical_pages,
                            q_tok_stride, hd * 2u,      q_tok_stride,
                            hd * 2u,      0u,           0u,
                            0u,           0u};
    attn_w.insert(attn_w.end(), row_offs_8.begin(), row_offs_8.end());
    attn_w.insert(attn_w.end(), widths_8.begin(), widths_8.end());
    const U32 max_q_blocks = (max_w + 63u) / 64u;
    auto &attn = e.emit(
        "attention_affine4",
        "omlx_m5_paged_flash_prefill_batched_affine4_g128_hd256",
        {.width = 256u, .height = U64(max_q_blocks) * qh, .depth = U64(batch_size)},
        {.width = 256u},
        {query, gate, state_k, state_v, mixer, attention_positions, pt[0],
         pt[1], pt[2], pt[3], pt[4], pt[5], pt[6], pt[7]});
    e.dynamic(attn, 14, to_bytes(attn_w));
    e.profile.attention_kernel =
        "omlx_m5_paged_flash_prefill_batched_affine4_g128_hd256";
    e.profile.attention_grid = attn.grid;
    e.profile.attention_numeric_contract_id =
        "paged.nax_affine4_g128_flash_v1.bq64_bk32_dsplit.f32_softmax_gate.v1";
    e.profile.attention_row_exact_eligible = false;
    return;
  }

  const Ref partial_half_ref{split_output.buffer, split_output.offset,
                             split_plan.partial_half_bytes};
  const Ref q_half_ref{split_output.buffer,
                       split_output.offset + split_plan.q_half_offset,
                       split_plan.q_half_bytes};
  const Ref q_sum_ref{split_output.buffer,
                      split_output.offset + split_plan.q_sum_offset,
                      split_plan.q_sum_bytes};

  auto &post = e.emit(
      "attention_affine4",
      "omlx_m5_attention_qk_norm_rope_kv_append_pack_q_batched_affine4_g128",
      {.width = 32u, .height = U64(total_m) * (qh + kvh)}, {.width = 32u},
      {query, key, value, q_norm, k_norm, state_k, state_v,
       attention_positions, pt[0], pt[1], pt[2], pt[3], pt[4], pt[5], pt[6],
       pt[7], q_half_ref, q_sum_ref});
  e.dynamic(post, 18, to_bytes(post_w));

  std::vector<U32> attn_w{qh,
                          kvh,
                          hd,
                          batch_size,
                          64u,
                          physical_pages,
                          q_tok_stride,
                          hd * 2u,
                          q_tok_stride,
                          hd * 2u,
                          split_plan.splits,
                          split_plan.split_tokens,
                          split_plan.r_stride,
                          split_plan.n_rblks};
  attn_w.insert(attn_w.end(), row_offs_8.begin(), row_offs_8.end());
  attn_w.insert(attn_w.end(), widths_8.begin(), widths_8.end());

  const char *partial_fn =
      (all_w_eq_1 && repeat <= 6u)
          ? "omlx_m5_paged_attention_affine4_g128_nax2sg_batched_split_partial_r6_hd256"
          : (all_w_eq_1 && repeat <= 8u)
                ? "omlx_m5_paged_attention_affine4_g128_nax2sg_batched_split_partial_r8_hd256"
                : "omlx_m5_paged_attention_affine4_g128_nax2sg_batched_split_partial_r16_hd256";
  auto &partial = e.emit(
      "attention_affine4", partial_fn,
      {.width = U64(split_plan.splits) * 64u,
       .height = U64(kvh) * split_plan.n_rblks,
       .depth = U64(batch_size)},
      {.width = 64u},
      {q_half_ref, q_sum_ref, state_k, state_v, split_max, split_sum,
       partial_half_ref, attention_positions, pt[0], pt[1], pt[2], pt[3], pt[4],
       pt[5], pt[6], pt[7]});
  e.dynamic(partial, 16, to_bytes(attn_w));
  e.profile.split_attention = true;
  e.profile.split_count = split_plan.splits;
  e.profile.split_workspace_bytes =
      U64(total_m) * qh * split_plan.splits * 8u + split_plan.q_sum_offset +
      split_plan.q_sum_bytes;
  e.profile.attention_kernel = partial_fn;
  e.profile.attention_grid = partial.grid;

  const U32 kv_width = kvh * hd;
  auto merge_params = words(
      {qh, kvh, hd, total_m, 0u, max_context, 64u, physical_pages, q_tok_stride,
       hd * 2u, kv_width, kv_width, q_tok_stride, hd * 2u, split_plan.splits,
       split_plan.split_tokens});
  auto &merge = e.emit(
      "attention_affine4",
      "omlx_m5_paged_attention_affine4_g128_split_merge_half_4sg_hd256",
      {.width = 128u, .height = U64(total_m) * qh}, {.width = 128u},
      {gate, split_max, split_sum, partial_half_ref, mixer});
  e.dynamic(merge, 5, std::move(merge_params));
  e.profile.attention_merge_kernel =
      "omlx_m5_paged_attention_affine4_g128_split_merge_half_4sg_hd256";
  e.profile.attention_merge_grid = merge.grid;
  e.profile.attention_numeric_contract_id =
      "paged.nax_affine4_g128_foscar.nax2sg_half_s512.v2";
  e.profile.attention_row_exact_eligible = true;
}

inline void emit_paged_flash_prefill_affine4_g128_unpacked_legacy(
    Encoder &e, const QwenModelConfig &c, U32 t, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, std::vector<std::byte> params) {
  if (c.head_dim != 256) {
    throw std::invalid_argument("Affine4Group128 flash prefill requires hd256");
  }
  const U32 q_blocks = (t + 63u) / 64u;
  auto &attn = e.emit(
      "attention_affine4", "omlx_m5_paged_flash_prefill_affine4_g128_hd256",
      {.width = 256u, .height = U64(q_blocks) * c.num_attention_heads},
      {.width = 256u},
      {query, gate, state_k, state_v, mixer, page_table, attention_positions});
  e.profile.attention_kernel = "omlx_m5_paged_flash_prefill_affine4_g128_hd256";
  e.profile.attention_grid = attn.grid;
  e.profile.attention_numeric_contract_id =
      "paged.nax_affine4_g128_flash_v1.bq64_bk32_dsplit.f32_softmax_gate.v1";
  e.profile.attention_row_exact_eligible = false;
  e.dynamic(attn, 7, std::move(params), 3, 4);
}

inline void emit_paged_flash_prefill_gqa_affine4_g128(
    Encoder &e, const QwenModelConfig &c, U32 t, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, std::vector<std::byte> params,
    U32 forced_wm = 0u) {
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  if (c.head_dim != 256 || kvh == 0u || (qh % kvh) != 0u) {
    throw std::invalid_argument(
        "GQA Affine4Group128 flash prefill requires hd256 and valid GQA");
  }
  const auto tier = select_gqa_flash_prefill_tier(t, qh, kvh, forced_wm);
  const U64 m_total = U64(t) * U64(qh / kvh);
  const U64 m_blocks = (m_total + U64(tier.m_tile) - 1ull) / U64(tier.m_tile);
  auto &attn = e.emit(
      "attention_affine4", tier.affine4_kernel,
      {.width = tier.tg_threads, .height = m_blocks * U64(kvh)},
      {.width = tier.tg_threads},
      {query, gate, state_k, state_v, mixer, page_table, attention_positions});
  e.profile.attention_kernel = tier.affine4_kernel;
  e.profile.attention_grid = attn.grid;
  e.profile.attention_numeric_contract_id =
      "paged.nax_affine4_g128_flash_v3.gqa_bk16_wn4.f32_softmax_gate.v1";
  e.profile.attention_row_exact_eligible = false;
  e.dynamic(attn, 7, std::move(params), 3, 4);
}

inline void emit_paged_flash_prefill_affine4_g128(
    Encoder &e, const QwenModelConfig &c, U32 t, Ref query, Ref gate,
    Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, std::vector<std::byte> params) {
  // WP3.2 BK=16, WN=4 GQA Flash-Prefill (<= 14.0 KiB SRAM, 2 TGs/core, 0 B spill)
  // enabled by default for all chunk sizes (both t >= 64 and t < 64; short tails
  // at long KV are routed to Split-KV upstream via gqa_flash_prefill_tail_needs_split_kv).
  if (gqa_flash_prefill_eligible(U32(c.num_attention_heads),
                                 U32(c.num_key_value_heads),
                                 U32(c.head_dim))) {
    emit_paged_flash_prefill_gqa_affine4_g128(
        e, c, t, query, gate, state_k, state_v, mixer, page_table,
        attention_positions, std::move(params));
    return;
  }
  emit_paged_flash_prefill_affine4_g128_unpacked_legacy(
      e, c, t, query, gate, state_k, state_v, mixer, page_table,
      attention_positions, std::move(params));
}

inline void emit_paged_kv_append_affine4_g128(
    Encoder &e, const QwenModelConfig &c, U32 t, Ref key, Ref value,
    Ref state_k, Ref state_v, Ref page_table, std::vector<std::byte> params,
    std::optional<U32> tok_word = 3u, std::optional<U32> pos_word = 4u,
    std::vector<std::pair<U64, U32>> *pos_patches = nullptr) {
  auto &append = e.emit(
      "attention_affine4", "omlx_m5_paged_kv_append_affine4_g128",
      {.width = 32u, .height = U64(t) * c.num_key_value_heads}, {.width = 32u},
      {key, value, state_k, state_v, page_table});
  const U64 off_app =
      e.dynamic(append, 5, std::move(params), tok_word, pos_word);
  if (pos_patches) {
    pos_patches->emplace_back(off_app, 4u);
  }
}

inline U32 affine4_short_tail_split_tokens(
    U32 t, U32 qh, U32 kvh, U32 context_envelope, U64 split_max_bytes,
    U64 split_sum_bytes, U64 split_output_bytes) {
  if (t == 0u || qh == 0u || kvh == 0u || context_envelope == 0u) {
    return 0u;
  }
  for (U32 split_tokens : {512u, 1024u, 2048u, 4096u, 8192u}) {
    if (t <= 8u && split_tokens != 512u) {
      break;
    }
    const U32 splits = (context_envelope + split_tokens - 1u) / split_tokens;
    if (splits < 2u && t > 8u) {
      continue;
    }
    const auto lo =
        affine4_split_scratch_layout(t, qh, kvh, splits, split_output_bytes);
    if (splits && lo.fits && U64(t) * qh * splits * 4u <= split_max_bytes &&
        U64(t) * qh * splits * 4u <= split_sum_bytes) {
      return split_tokens;
    }
  }
  return 0u;
}

inline void emit_paged_attention_affine4_g128(
    Encoder &e, const QwenModelConfig &c, U32 t, bool split,
    U32 context_envelope, U32 max_context, U32 physical_pages, Ref query,
    Ref gate, Ref state_k, Ref state_v, Ref mixer, Ref page_table,
    Ref attention_positions, Ref split_max, Ref split_sum, Ref split_output,
    std::vector<std::byte> params, std::optional<U32> tok_word = 3u,
    std::optional<U32> pos_word = 4u,
    std::vector<std::pair<U64, U32>> *pos_patches = nullptr,
    bool q_prepacked = false) {
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  const U32 hd = U32(c.head_dim);
  if ((hd != 256u && hd != 128u) || kvh == 0u || (qh % kvh) != 0u ||
      (qh / kvh) > 16u) {
    throw std::invalid_argument("Affine4Group128 attention geometry unsupported");
  }
  const U32 repeat = qh / kvh;
  const U32 bq = 16u / repeat;
  const U32 t_blocks = (t + bq - 1u) / bq;
  const U32 groups = hd / 128u;
  const U32 tg_width = groups * 32u;
  const U32 q_width = qh * hd;
  const U32 kv_width = kvh * hd;

  const auto make_split_params = [&](U32 splits, U32 split_tokens) {
    std::array<U32, 16> w{qh,
                          kvh,
                          hd,
                          t,
                          0u,
                          max_context,
                          64u,
                          physical_pages,
                          q_width * 2u,
                          hd * 2u,
                          kv_width,
                          kv_width,
                          q_width * 2u,
                          hd * 2u,
                          splits,
                          split_tokens};
    if (params.size() >= 14u * sizeof(U32)) {
      std::memcpy(w.data(), params.data(), 14u * sizeof(U32));
    }
    w[14] = splits;
    w[15] = split_tokens;
    const auto raw = std::as_bytes(std::span<const U32>(w));
    return std::vector<std::byte>(raw.begin(), raw.end());
  };

  if (split && hd == 256u) {
    // Unified 512-token splits for T=1..8 (0-ULP AR == DFlash2); Base 2.2 short
    // tails (T>8) pick the smallest split_tokens in {512..8192} fitting scratch.
    const U32 split_tokens =
        (t <= 8u)
            ? 512u
            : affine4_short_tail_split_tokens(t, qh, kvh, context_envelope,
                                              split_max.bytes, split_sum.bytes,
                                              split_output.bytes);
    const U32 eff_split_tokens = split_tokens ? split_tokens : 512u;
    const U32 splits =
        (context_envelope + eff_split_tokens - 1u) / eff_split_tokens;
    const auto lo =
        affine4_split_scratch_layout(t, qh, kvh, splits, split_output.bytes);
    if (!splits || !lo.fits || U64(t) * qh * splits * 4u > split_max.bytes ||
        U64(t) * qh * splits * 4u > split_sum.bytes) {
      throw std::invalid_argument("Affine4Group128 split workspace exceeded");
    }
    const Ref partial_half_ref{split_output.buffer, split_output.offset,
                               lo.partial_half_bytes};
    const Ref q_half_ref{split_output.buffer,
                         split_output.offset + lo.q_half_offset,
                         lo.q_half_bytes};
    const Ref q_sum_ref{split_output.buffer,
                        split_output.offset + lo.q_sum_offset, lo.q_sum_bytes};
    e.profile.split_attention = true;
    e.profile.split_count = splits;
    e.profile.split_workspace_bytes =
        U64(t) * qh * splits * 8u + lo.q_sum_offset + lo.q_sum_bytes;
    auto split_params = make_split_params(splits, eff_split_tokens);

    if (!q_prepacked) {
      auto &pack_q = e.emit(
          "attention_affine4", "omlx_m5_pack_q_half_foscar_hd256",
          {.width = 32u, .height = U64(kvh) * lo.r_stride}, {.width = 32u},
          {query, q_half_ref, q_sum_ref});
      const U64 off_pack =
          e.dynamic(pack_q, 3, split_params, tok_word, pos_word);
      if (pos_patches) {
        pos_patches->emplace_back(off_pack, 4u);
      }
    }

    const char *partial_fn =
        (t == 1u && repeat <= 6u)
            ? "omlx_m5_paged_attention_affine4_g128_nax2sg_split_partial_r6_hd256"
            : (t == 1u && repeat <= 8u)
                  ? "omlx_m5_paged_attention_affine4_g128_nax2sg_split_partial_r8_hd256"
                  : "omlx_m5_paged_attention_affine4_g128_nax2sg_split_partial_r16_hd256";
    auto &partial = e.emit(
        "attention_affine4", partial_fn,
        {.width = U64(splits) * 64u,
         .height = U64(kvh),
         .depth = U64(lo.n_rblks)},
        {.width = 64u},
        {q_half_ref, q_sum_ref, state_k, state_v, split_max, split_sum,
         partial_half_ref, page_table, attention_positions});
    e.profile.attention_kernel = partial_fn;
    e.profile.attention_grid = partial.grid;
    const U64 off_part =
        e.dynamic(partial, 9, split_params, tok_word, pos_word);
    if (pos_patches) {
      pos_patches->emplace_back(off_part, 4u);
    }

    auto &merge = e.emit(
        "attention_affine4",
        "omlx_m5_paged_attention_affine4_g128_split_merge_half_4sg_hd256",
        {.width = 128u, .height = U64(t) * qh}, {.width = 128u},
        {gate, split_max, split_sum, partial_half_ref, mixer});
    e.profile.attention_merge_kernel =
        "omlx_m5_paged_attention_affine4_g128_split_merge_half_4sg_hd256";
    e.profile.attention_merge_grid = merge.grid;
    const U64 off_mrg =
        e.dynamic(merge, 5, std::move(split_params), tok_word, pos_word);
    if (pos_patches) {
      pos_patches->emplace_back(off_mrg, 4u);
    }
    e.profile.attention_numeric_contract_id =
        "paged.nax_affine4_g128_foscar.nax2sg_half_s512.v2";
    e.profile.attention_row_exact_eligible = true;
    return;
  }

  if (split) {
    // HD == 128 split compatibility path.
    const U32 splits = (context_envelope + 511u) / 512u;
    if (!splits ||
        U64(t) * qh * splits * 4u > split_max.bytes ||
        U64(t) * qh * splits * 4u > split_sum.bytes ||
        U64(t) * qh * splits * hd * 4u > split_output.bytes) {
      throw std::invalid_argument("Affine4Group128 split workspace exceeded");
    }
    e.profile.split_attention = true;
    e.profile.split_count = splits;
    e.profile.split_workspace_bytes = U64(t) * qh * splits * (8u + hd * 4u);
    auto split_params = make_split_params(splits, 512u);
    const char *partial_fn =
        "omlx_m5_paged_attention_affine4_g128_split_partial_hd128";
    auto &partial = e.emit(
        "attention_affine4", partial_fn,
        {.width = U64(splits) * tg_width, .height = U64(t_blocks) * kvh},
        {.width = tg_width},
        {query, state_k, state_v, split_max, split_sum, split_output,
         page_table, attention_positions});
    e.profile.attention_kernel = partial_fn;
    e.profile.attention_grid = partial.grid;
    const U64 off_part =
        e.dynamic(partial, 8, split_params, tok_word, pos_word);
    if (pos_patches) {
      pos_patches->emplace_back(off_part, 4u);
    }

    auto &merge = e.emit(
        "attention_affine4", "omlx_m5_paged_attention_affine4_g128_split_merge",
        {.width = 32u, .height = U64(t) * qh}, {.width = 32u},
        {gate, split_max, split_sum, split_output, mixer});
    e.profile.attention_merge_kernel =
        "omlx_m5_paged_attention_affine4_g128_split_merge";
    e.profile.attention_merge_grid = merge.grid;
    const U64 off_mrg =
        e.dynamic(merge, 5, std::move(split_params), tok_word, pos_word);
    if (pos_patches) {
      pos_patches->emplace_back(off_mrg, 4u);
    }
  } else {
    const char *fn =
        (hd == 256u) ? "omlx_m5_paged_causal_attention_affine4_g128_hd256"
                     : "omlx_m5_paged_causal_attention_affine4_g128_hd128";
    auto &attn = e.emit(
        "attention_affine4", fn,
        {.width = tg_width, .height = U64(t_blocks) * kvh},
        {.width = tg_width},
        {query, gate, state_k, state_v, mixer, page_table, attention_positions});
    e.profile.attention_kernel = fn;
    e.profile.attention_grid = attn.grid;
    const U64 off_att =
        e.dynamic(attn, 7, std::move(params), tok_word, pos_word);
    if (pos_patches) {
      pos_patches->emplace_back(off_att, 4u);
    }
  }
  e.profile.attention_numeric_contract_id =
      "paged.nax_affine4_g128_foscar.chunk16.ordered_sb512.v1";
  e.profile.attention_row_exact_eligible = true;
}

} // namespace omlx::m5::detail::forward
