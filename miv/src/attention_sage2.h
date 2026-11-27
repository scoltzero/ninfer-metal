#pragma once

// Module 2 Route B.1 (SageAttention2) & Route B.2 (MInference Vertical-Slash)
// host dispatch helpers & scratch workspace planner.
// Self-contained header ready for Wave 2 qwen_forward.cpp / qwen_memory.cpp
// wiring and standalone Wave 1 tests & benchmarks.

#include "attention_nax.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5::detail::forward {

enum class Sage2PvMode {
  Mxfp8,
  Fp16,
};

inline Sage2PvMode resolve_sage2_pv_mode(
    std::optional<Sage2PvMode> override_mode = std::nullopt) {
  if (override_mode.has_value()) {
    return *override_mode;
  }
  if (const char *env = std::getenv("OMLX_M5_SAGE2_PV_MODE")) {
    const std::string_view sv(env);
    if (sv == "mxfp8" || sv == "MXFP8" || sv == "fp8" || sv == "FP8") {
      return Sage2PvMode::Mxfp8;
    }
    if (sv == "fp16" || sv == "FP16" || sv == "bf16" || sv == "BF16") {
      return Sage2PvMode::Fp16;
    }
  }
  // Default to FP16/BF16 P x V (zero P/V quantization loss, higher cosine sim);
  // override via OMLX_M5_SAGE2_PV_MODE=mxfp8 or explicit Sage2PvMode::Mxfp8.
  return Sage2PvMode::Fp16;
}

inline const char *sage2_pv_mode_name(Sage2PvMode mode) {
  return (mode == Sage2PvMode::Mxfp8) ? "mxfp8" : "fp16";
}

inline bool sage2_runtime_eligible(
    const MetalContext &ctx, const std::filesystem::path &libs, U32 query_heads,
    U32 kv_heads, U32 head_dim, U32 page_tokens = 64u) {
  if (head_dim != 256u || page_tokens != 64u || kv_heads == 0u ||
      (query_heads % kv_heads) != 0u || (query_heads / kv_heads) > 16u) {
    return false;
  }
  const auto &p = ctx.capabilities().profile.probe;
  return p.runtime_os_major >= 27 && p.metal4_compiler && p.tensor_api &&
         std::filesystem::is_regular_file(
             libs / "omlx_m5_attention_sage2.metallib");
}

inline bool minference_runtime_eligible(
    const MetalContext &ctx, const std::filesystem::path &libs, U32 query_heads,
    U32 kv_heads, U32 head_dim, U32 page_tokens = 64u) {
  return sage2_runtime_eligible(ctx, libs, query_heads, kv_heads, head_dim,
                                page_tokens) &&
         std::filesystem::is_regular_file(
             libs / "omlx_m5_attention_minference.metallib");
}

struct Sage2Tier {
  U32 wm = 6u;
  U32 tg_threads = 384u;
  U32 bm_tok = 16u;
};

inline Sage2Tier select_sage2_tier(U32 tokens, U32 query_heads, U32 kv_heads,
                                   U32 forced_wm = 0u) {
  (void)tokens;
  const U32 repeat = (kv_heads > 0u && query_heads >= kv_heads)
                         ? (query_heads / kv_heads)
                         : 1u;
  U32 wm = forced_wm;
  if (wm == 0u) {
    if (const char *env = std::getenv("OMLX_M5_SAGE2_WM")) {
      const std::string_view sv(env);
      if (sv == "4") wm = 4u;
      else if (sv == "6") wm = 6u;
      else if (sv == "8") wm = 8u;
    }
  }
  if (wm == 0u) {
    // Real Apple10 M5 Ultra calibration: WM=4 (256 threads, M_TILE=64) keeps
    // Q_i8 100% resident in 16 KiB threadgroup SRAM with NUM_WAVES=1 (zero wave
    // serialization, 100% SG active, 6 barriers/block vs 24 for WM=8).
    // Override via OMLX_M5_SAGE2_WM=4|6|8 for A/B diagnostics.
    wm = 4u;
  }
  if (wm != 4u && wm != 6u && wm != 8u) {
    wm = 6u;
  }
  const U32 m_tile = wm * 16u;
  const U32 bm_tok = std::max<U32>(1u, m_tile / std::max<U32>(1u, repeat));
  return {wm, wm * 64u, bm_tok};
}

inline const char *sage2_kernel_name(U32 wm, bool is_affine4,
                                     Sage2PvMode pv_mode,
                                     bool sparse_minference) {
  const bool mxfp8 = (pv_mode == Sage2PvMode::Mxfp8);
  if (is_affine4) {
    if (!sparse_minference) {
      if (mxfp8) {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm8";
      } else {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm8";
      }
    } else {
      if (mxfp8) {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm8";
      } else {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm8";
      }
    }
  } else {
    if (!sparse_minference) {
      if (mxfp8) {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm8";
      } else {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm8";
      }
    } else {
      if (mxfp8) {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm8";
      } else {
        if (wm == 4u) return "omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm4";
        if (wm == 6u) return "omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm6";
        return "omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm8";
      }
    }
  }
}

// Route B.2 MInference dynamic budget & scratch workspace layout (<= 2.26 MiB).
inline constexpr U32 kMInferenceMinContextTokens = 4096u;
inline constexpr U32 kMInferenceProbeWindow = 64u;
inline constexpr U32 kMInferenceSinkBlocks = 4u;
inline constexpr U32 kMInferenceLocalBlocks = 32u;
inline constexpr U32 kMInferenceActiveStride = 512u;
inline constexpr U32 kMInferenceTopStride = 256u;
inline constexpr U32 kMInferenceMaxSplits = 64u;

struct MInferenceBudget {
  U32 k_vertical = 48u;
  U32 k_slash = 22u;
};

inline MInferenceBudget minference_budget_for_kv_tokens(U32 context_envelope) {
  const U32 nk_blocks = (context_envelope + 31u) / 32u;
  if (nk_blocks <= 512u) return {48u, 22u};    // <= 16K (<= 128 active blocks)
  if (nk_blocks <= 1024u) return {64u, 30u};   // <= 32K (<= 160 active blocks)
  if (nk_blocks <= 2048u) return {96u, 46u};   // <= 64K (<= 224 active blocks)
  if (nk_blocks <= 4096u) return {128u, 62u};  // <= 128K (<= 288 active blocks)
  return {160u, 94u};                          // <= 256K (<= 384 active blocks)
}

struct MInferenceScratchLayout {
  U32 max_kv_blocks = 0u;
  U32 max_q_blocks = 0u;
  U32 active_stride = kMInferenceActiveStride;
  U32 top_stride = kMInferenceTopStride;
  U32 split_count = 1u;

  U64 probe_m_split_off = 0u;
  U64 probe_m_split_bytes = 0u;
  U64 probe_l_split_off = 0u;
  U64 probe_l_split_bytes = 0u;
  U64 probe_m_global_off = 0u;
  U64 probe_m_global_bytes = 0u;
  U64 probe_inv_z_off = 0u;
  U64 probe_inv_z_bytes = 0u;
  U64 v_score_off = 0u;
  U64 v_score_bytes = 0u;
  U64 d_score_off = 0u;
  U64 d_score_bytes = 0u;
  U64 top_v_idx_off = 0u;
  U64 top_v_idx_bytes = 0u;
  U64 top_s_lag_off = 0u;
  U64 top_s_lag_bytes = 0u;
  U64 top_counts_off = 0u;
  U64 top_counts_bytes = 0u;
  U64 active_indices_off = 0u;
  U64 active_indices_bytes = 0u;
  U64 active_counts_off = 0u;
  U64 active_counts_bytes = 0u;
  U64 total_bytes = 0u;
};

inline MInferenceScratchLayout plan_minference_scratch(
    U32 max_tokens, U32 query_heads, U32 kv_heads, U32 max_context,
    U32 forced_wm = 0u) {
  const auto align256 = [](U64 x) { return (x + 255ull) & ~255ull; };
  MInferenceScratchLayout lo{};
  const auto tier = select_sage2_tier(std::max<U32>(1u, max_tokens),
                                      query_heads, kv_heads, forced_wm);
  lo.max_kv_blocks =
      std::min<U32>(8192u, std::max<U32>(128u, (max_context + 31u) / 32u));
  lo.max_q_blocks = std::max<U32>(1u, (max_tokens + tier.bm_tok - 1u) / tier.bm_tok);
  lo.split_count = std::clamp<U32>((lo.max_kv_blocks + 127u) / 128u, 1u,
                                   kMInferenceMaxSplits);

  U64 cursor = 0u;
  auto alloc = [&](U64 &off, U64 &bytes, U64 nbytes) {
    off = cursor;
    bytes = std::max<U64>(256ull, nbytes);
    cursor = align256(cursor + bytes);
  };

  const U64 probe_rows = U64(kMInferenceProbeWindow) * query_heads;
  alloc(lo.probe_m_split_off, lo.probe_m_split_bytes,
        probe_rows * lo.split_count * sizeof(float));
  alloc(lo.probe_l_split_off, lo.probe_l_split_bytes,
        probe_rows * lo.split_count * sizeof(float));
  alloc(lo.probe_m_global_off, lo.probe_m_global_bytes,
        probe_rows * sizeof(float));
  alloc(lo.probe_inv_z_off, lo.probe_inv_z_bytes,
        probe_rows * sizeof(float));
  alloc(lo.v_score_off, lo.v_score_bytes,
        U64(kv_heads) * lo.max_kv_blocks * sizeof(float));
  alloc(lo.d_score_off, lo.d_score_bytes,
        U64(kv_heads) * lo.max_kv_blocks * sizeof(float));
  alloc(lo.top_v_idx_off, lo.top_v_idx_bytes,
        U64(kv_heads) * lo.top_stride * sizeof(std::uint16_t));
  alloc(lo.top_s_lag_off, lo.top_s_lag_bytes,
        U64(kv_heads) * lo.top_stride * sizeof(std::uint16_t));
  alloc(lo.top_counts_off, lo.top_counts_bytes,
        U64(kv_heads) * 2ull * sizeof(U32));
  alloc(lo.active_indices_off, lo.active_indices_bytes,
        U64(kv_heads) * lo.max_q_blocks * lo.active_stride *
            sizeof(std::uint16_t));
  alloc(lo.active_counts_off, lo.active_counts_bytes,
        U64(kv_heads) * lo.max_q_blocks * sizeof(U32));
  lo.total_bytes = cursor;
  return lo;
}

inline void emit_minference_build_sparse_indices(
    Encoder &e, const QwenModelConfig &c, U32 t, U32 context_envelope,
    bool is_affine4, Ref query, Ref state_k, Ref page_table,
    Ref attention_positions, Ref scratch,
    const MInferenceScratchLayout &lo,
    const std::vector<std::byte> &base_params, U32 forced_wm = 0u) {
  if (scratch.buffer == nullptr || scratch.bytes < lo.total_bytes) {
    throw std::invalid_argument("MInference scratch buffer too small");
  }
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  const U32 repeat = qh / kvh;
  const auto tier = select_sage2_tier(t, qh, kvh, forced_wm);
  const U32 n_q_blocks = (t + tier.bm_tok - 1u) / tier.bm_tok;
  if (n_q_blocks > lo.max_q_blocks) {
    throw std::invalid_argument("MInference n_q_blocks exceeds scratch plan");
  }
  const U32 w_probe = std::min<U32>(kMInferenceProbeWindow, t);
  const U32 q_probe_start = t - w_probe;
  const U32 nk_blocks_est =
      std::min<U32>(lo.max_kv_blocks, std::max<U32>(1u, (context_envelope + 31u) / 32u));
  const U32 split_count = std::clamp<U32>(
      lo.split_count, 1u, std::min<U32>(kMInferenceMaxSplits, nk_blocks_est));
  const U32 split_blocks = (nk_blocks_est + split_count - 1u) / split_count;
  const auto budget = minference_budget_for_kv_tokens(context_envelope);

  std::array<U32, 28> mp{};
  if (base_params.size() < 14u * sizeof(U32)) {
    throw std::invalid_argument("MInference base_params too short");
  }
  std::memcpy(mp.data(), base_params.data(), 14u * sizeof(U32));
  mp[14] = w_probe;
  mp[15] = q_probe_start;
  mp[16] = split_count;
  mp[17] = split_blocks;
  mp[18] = lo.max_kv_blocks;
  mp[19] = tier.bm_tok;
  mp[20] = n_q_blocks;
  mp[21] = lo.max_q_blocks;
  mp[22] = lo.active_stride;
  mp[23] = kMInferenceSinkBlocks;
  mp[24] = kMInferenceLocalBlocks;
  mp[25] = budget.k_vertical;
  mp[26] = budget.k_slash;
  mp[27] = lo.top_stride;
  const auto mp_span = std::as_bytes(std::span<const U32>(mp));
  const std::vector<std::byte> mp_bytes(mp_span.begin(), mp_span.end());

  const auto sub_ref = [&](U64 off, U64 bytes) -> Ref {
    return {scratch.buffer, scratch.offset + off, bytes};
  };
  const Ref m_split = sub_ref(lo.probe_m_split_off, lo.probe_m_split_bytes);
  const Ref l_split = sub_ref(lo.probe_l_split_off, lo.probe_l_split_bytes);
  const Ref m_glob = sub_ref(lo.probe_m_global_off, lo.probe_m_global_bytes);
  const Ref inv_z = sub_ref(lo.probe_inv_z_off, lo.probe_inv_z_bytes);
  const Ref v_score = sub_ref(lo.v_score_off, lo.v_score_bytes);
  const Ref d_score = sub_ref(lo.d_score_off, lo.d_score_bytes);
  const Ref top_v = sub_ref(lo.top_v_idx_off, lo.top_v_idx_bytes);
  const Ref top_s = sub_ref(lo.top_s_lag_off, lo.top_s_lag_bytes);
  const Ref top_cnt = sub_ref(lo.top_counts_off, lo.top_counts_bytes);
  const Ref act_idx = sub_ref(lo.active_indices_off, lo.active_indices_bytes);
  const Ref act_cnt = sub_ref(lo.active_counts_off, lo.active_counts_bytes);

  const U32 bm_tok_p = std::max<U32>(1u, 64u / std::max<U32>(1u, repeat));
  const U32 n_probe_qblks = (w_probe + bm_tok_p - 1u) / bm_tok_p;
  const char *stats_fn =
      is_affine4 ? "omlx_m5_minference_probe_stats_affine4_g128_hd256"
                 : "omlx_m5_minference_probe_stats_bf16_hd256";
  const char *score_fn =
      is_affine4 ? "omlx_m5_minference_probe_score_affine4_g128_hd256"
                 : "omlx_m5_minference_probe_score_bf16_hd256";

  auto &d_stats = e.emit(
      "attention_minference", stats_fn,
      {.width = U64(split_count) * 256u, .height = U64(n_probe_qblks) * kvh},
      {.width = 256u},
      {query, state_k, page_table, attention_positions, m_split, l_split});
  e.dynamic(d_stats, 6, mp_bytes, 3, 4);

  const U64 clear_or_rows = std::max<U64>(
      U64(w_probe) * qh, U64(kvh) * lo.max_kv_blocks);
  auto &d_reduce = e.emit(
      "attention_minference", "omlx_m5_minference_probe_reduce_and_clear",
      {.width = ((clear_or_rows + 255ull) / 256ull) * 256ull}, {.width = 256u},
      {m_split, l_split, m_glob, inv_z, v_score, d_score});
  e.dynamic(d_reduce, 6, mp_bytes, 3, 4);

  auto &d_score_op = e.emit(
      "attention_minference", score_fn,
      {.width = U64(split_count) * 256u, .height = U64(n_probe_qblks) * kvh},
      {.width = 256u},
      {query, state_k, page_table, attention_positions, m_glob, inv_z, v_score,
       d_score});
  e.dynamic(d_score_op, 8, mp_bytes, 3, 4);

  auto &d_topk = e.emit(
      "attention_minference", "omlx_m5_minference_select_topk",
      {.width = 256u, .height = U64(kvh)}, {.width = 256u},
      {v_score, d_score, attention_positions, top_v, top_s, top_cnt});
  e.dynamic(d_topk, 6, mp_bytes, 3, 4);

  auto &d_build = e.emit(
      "attention_minference", "omlx_m5_minference_build_indices",
      {.width = 256u, .height = U64(n_q_blocks) * kvh}, {.width = 256u},
      {attention_positions, top_v, top_s, top_cnt, act_idx, act_cnt});
  e.dynamic(d_build, 6, mp_bytes, 3, 4);
}

inline void emit_paged_flash_prefill_sage2(
    Encoder &e, const QwenModelConfig &c, U32 t, U32 context_envelope,
    bool is_affine4, bool sparse_minference, Ref query, Ref gate, Ref state_k,
    Ref state_v, Ref mixer, Ref page_table, Ref attention_positions,
    const std::vector<std::byte> &base_params,
    std::optional<Ref> minf_scratch = std::nullopt,
    std::optional<Sage2PvMode> pv_mode_override = std::nullopt,
    U32 forced_wm = 0u) {
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  if (c.head_dim != 256 || kvh == 0u || (qh % kvh) != 0u) {
    throw std::invalid_argument("SageAttention2 requires hd256 and valid GQA");
  }
  const auto tier = select_sage2_tier(t, qh, kvh, forced_wm);
  const U32 q_blocks = (t + tier.bm_tok - 1u) / tier.bm_tok;
  const Sage2PvMode pv_mode = resolve_sage2_pv_mode(pv_mode_override);

  // Automatic short-context bypass for MInference (< 4096 KV tokens).
  const bool use_minf = sparse_minference &&
                        (context_envelope >= kMInferenceMinContextTokens) &&
                        minf_scratch.has_value() &&
                        minf_scratch->buffer != nullptr;

  Ref act_idx = page_table; // Unused dummy binding when !use_minf
  Ref act_cnt = page_table;
  U32 max_q_blocks = q_blocks;
  U32 active_stride = kMInferenceActiveStride;

  if (use_minf) {
    const auto lo = plan_minference_scratch(t, qh, kvh, context_envelope,
                                            forced_wm);
    emit_minference_build_sparse_indices(
        e, c, t, context_envelope, is_affine4, query, state_k, page_table,
        attention_positions, *minf_scratch, lo, base_params, forced_wm);
    act_idx = {minf_scratch->buffer,
               minf_scratch->offset + lo.active_indices_off,
               lo.active_indices_bytes};
    act_cnt = {minf_scratch->buffer,
               minf_scratch->offset + lo.active_counts_off,
               lo.active_counts_bytes};
    max_q_blocks = lo.max_q_blocks;
    active_stride = lo.active_stride;
  }

  std::array<U32, 18> sp{};
  if (base_params.size() < 14u * sizeof(U32)) {
    throw std::invalid_argument("SageAttention2 base_params too short");
  }
  std::memcpy(sp.data(), base_params.data(), 14u * sizeof(U32));
  sp[14] = max_q_blocks;
  sp[15] = active_stride;
  sp[16] = use_minf ? 1u : 0u;
  sp[17] = 0u;
  const auto sp_span = std::as_bytes(std::span<const U32>(sp));
  std::vector<std::byte> sp_bytes(sp_span.begin(), sp_span.end());

  const char *fn = sage2_kernel_name(tier.wm, is_affine4, pv_mode, use_minf);
  // Note buffer(7) is unused so buffer(8)=active_indices, buffer(9)=active_counts,
  // buffer(10)=Sage2AttentionParams match Section 6.3 exact binding contract!
  auto &attn = e.emit(
      "attention_sage2", fn,
      {.width = tier.tg_threads, .height = U64(q_blocks) * kvh},
      {.width = tier.tg_threads},
      {query, gate, state_k, state_v, mixer, page_table, attention_positions,
       page_table, act_idx, act_cnt});
  e.profile.attention_kernel = fn;
  e.profile.attention_grid = attn.grid;
  e.profile.attention_numeric_contract_id =
      std::string("paged.sage2.") + (is_affine4 ? "affine4_g128." : "bf16.") +
      sage2_pv_mode_name(pv_mode) + (use_minf ? ".minference_vs.v1" : ".dense.v1");
  e.profile.attention_row_exact_eligible = false;
  e.dynamic(attn, 10, std::move(sp_bytes), 3, 4);
}

} // namespace omlx::m5::detail::forward
