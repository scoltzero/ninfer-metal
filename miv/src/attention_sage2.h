#pragma once

// Phase 2 Work Package 2 (WP2.1 + WP2.2): SageAttention2 & MInference
// host dispatch helpers, O(N_KV) K-smoothing workspace planner, & scratch layout.

#include "attention_nax.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
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

// ============================================================================
// WP2.1 Reusable Single-Layer Transient K-Smoothing Workspace Layout (WN=4, D=64)
//   * K_int8:   [kvh, max_kv_blocks, 4, 32, 64] int8_t  (256 MiB @ 256K, kvh=4)
//   * mu_K:     [kvh, 4, pad_kv_blocks, 64]     float   ( 32 MiB @ 256K, kvh=4)
//   * s_K:      [kvh, max_kv_blocks, 4, 32]     float   ( 16 MiB @ 256K, kvh=4)
//   * valid_kv: [4]                             uint32  (256 B aligned)
// ============================================================================
struct Sage2KvWorkspaceLayout {
  U64 max_kv_blocks = 0;
  U64 pad_kv_blocks = 0;
  U64 k_int8_off = 0, k_int8_bytes = 0;
  U64 mu_k_off = 0, mu_k_bytes = 0;
  U64 s_k_off = 0, s_k_bytes = 0;
  U64 valid_kv_off = 0, valid_kv_bytes = 256ULL;
  U64 total_bytes = 0;
};

[[nodiscard]] inline Sage2KvWorkspaceLayout
plan_sage2_kv_workspace(U32 kvh, U32 max_kv_tokens) noexcept {
  const auto align256 = [](U64 x) noexcept { return (x + 255ULL) & ~255ULL; };
  Sage2KvWorkspaceLayout lo{};
  lo.max_kv_blocks = std::max<U64>(1ULL, (U64(max_kv_tokens) + 31ULL) / 32ULL);
  lo.pad_kv_blocks = (lo.max_kv_blocks + 15ULL) & ~15ULL;
  lo.k_int8_bytes = align256(U64(kvh) * lo.max_kv_blocks * 4ULL * 32ULL * 64ULL);
  lo.mu_k_bytes =
      align256(U64(kvh) * 4ULL * lo.pad_kv_blocks * 64ULL * sizeof(float));
  lo.s_k_bytes =
      align256(U64(kvh) * lo.max_kv_blocks * 4ULL * 32ULL * sizeof(float));
  lo.k_int8_off = 0ULL;
  lo.mu_k_off = lo.k_int8_off + lo.k_int8_bytes;
  lo.s_k_off = lo.mu_k_off + lo.mu_k_bytes;
  lo.valid_kv_off = lo.s_k_off + lo.s_k_bytes;
  lo.total_bytes = lo.valid_kv_off + lo.valid_kv_bytes;
  return lo;
}

struct Sage2Tier {
  U32 wm = 2u;
  U32 tg_threads = 256u;
  U32 bm_tok = 5u;
};

inline Sage2Tier select_sage2_tier(U32 tokens, U32 query_heads, U32 kv_heads,
                                   U32 forced_wm = 0u) {
  (void)tokens;
  (void)forced_wm;
  const U32 repeat = (kv_heads > 0u && query_heads >= kv_heads)
                         ? (query_heads / kv_heads)
                         : 1u;
  // Unified Apple10 zero-spill WN=4 tier: WM=2 (256 threads, M_TILE=32,
  // 13.03 KiB FP16 / 15.16 KiB MXFP8, 0 B spills, 2 TG/core = 512 thr/core).
  constexpr U32 wm = 2u;
  constexpr U32 m_tile = wm * 16u;
  const U32 bm_tok = std::max<U32>(1u, m_tile / std::max<U32>(1u, repeat));
  return {wm, wm * 128u, bm_tok};
}

inline const char *sage2_kernel_name(U32 wm, bool is_affine4,
                                     Sage2PvMode pv_mode,
                                     bool sparse_minference) {
  (void)wm;
  const bool mxfp8 = (pv_mode == Sage2PvMode::Mxfp8);
  if (is_affine4) {
    if (!sparse_minference) {
      return mxfp8
                 ? "omlx_m5_paged_flash_prefill_sage2_mxfp8_affine4_g128_hd256_wm2"
                 : "omlx_m5_paged_flash_prefill_sage2_fp16_affine4_g128_hd256_wm2";
    }
    return mxfp8
               ? "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_affine4_g128_hd256_wm2"
               : "omlx_m5_paged_flash_prefill_sage2_minf_fp16_affine4_g128_hd256_wm2";
  }
  if (!sparse_minference) {
    return mxfp8 ? "omlx_m5_paged_flash_prefill_sage2_mxfp8_bf16_hd256_wm2"
                 : "omlx_m5_paged_flash_prefill_sage2_fp16_bf16_hd256_wm2";
  }
  return mxfp8 ? "omlx_m5_paged_flash_prefill_sage2_minf_mxfp8_bf16_hd256_wm2"
               : "omlx_m5_paged_flash_prefill_sage2_minf_fp16_bf16_hd256_wm2";
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

inline void emit_sage2_prepare_k(
    Encoder &e, const QwenModelConfig &c, U32 context_envelope,
    bool is_affine4, Ref state_k, Ref page_table, Ref attention_positions,
    Ref sage2_kv_ws, const Sage2KvWorkspaceLayout &kv_lo,
    const std::vector<std::byte> &base_params) {
  if (sage2_kv_ws.buffer == nullptr || sage2_kv_ws.bytes < kv_lo.total_bytes) {
    throw std::invalid_argument("Sage2 KV workspace buffer missing or too small");
  }
  if (base_params.size() < 14u * sizeof(U32)) {
    throw std::invalid_argument("Sage2 prepare_k base_params too short");
  }
  const U32 kvh = U32(c.num_key_value_heads);
  U32 cache_cap = 0u;
  std::memcpy(&cache_cap, base_params.data() + 5u * sizeof(U32), sizeof(U32));
  const U32 ws_tok_cap =
      U32(std::min<U64>(kv_lo.max_kv_blocks * 32ULL, 0xFFFFFFFFULL));
  const U32 max_kv_cap =
      (cache_cap > 0u) ? std::min<U32>(cache_cap, ws_tok_cap) : ws_tok_cap;
  const U32 eff_env = (context_envelope > 0u)
                          ? std::min<U32>(context_envelope, max_kv_cap)
                          : max_kv_cap;
  const U32 env_blocks = std::max<U32>(1u, (eff_env + 31u) / 32u);
  const U32 env_blocks_pad16 =
      std::min<U32>(U32(kv_lo.pad_kv_blocks), (env_blocks + 15u) & ~15u);

  std::array<U32, 20> prep{};
  std::memcpy(prep.data(), base_params.data(), 14u * sizeof(U32));
  prep[14] = U32(kv_lo.max_kv_blocks);
  prep[15] = U32(kv_lo.pad_kv_blocks);
  prep[16] = max_kv_cap;
  prep[17] = env_blocks_pad16;
  prep[18] = 0u;
  prep[19] = 0u;
  const auto prep_span = std::as_bytes(std::span<const U32>(prep));
  const std::vector<std::byte> prep_bytes(prep_span.begin(), prep_span.end());

  const auto sub_ws = [&](U64 off, U64 bytes) -> Ref {
    return {sage2_kv_ws.buffer, sage2_kv_ws.offset + off, bytes};
  };
  const Ref k_int8_ref = sub_ws(kv_lo.k_int8_off, kv_lo.k_int8_bytes);
  const Ref mu_k_ref = sub_ws(kv_lo.mu_k_off, kv_lo.mu_k_bytes);
  const Ref s_k_ref = sub_ws(kv_lo.s_k_off, kv_lo.s_k_bytes);
  const Ref valid_kv_ref = sub_ws(kv_lo.valid_kv_off, kv_lo.valid_kv_bytes);

  auto &d_vkv = e.emit("attention_sage2", "omlx_m5_sage2_compute_valid_kv",
                       {.width = 256u}, {.width = 256u},
                       {attention_positions, valid_kv_ref});
  e.dynamic(d_vkv, 2, prep_bytes, 3, 4);

  const char *prep_fn = is_affine4
                            ? "omlx_m5_sage2_prepare_k_affine4_g128_hd256"
                            : "omlx_m5_sage2_prepare_k_bf16_hd256";
  auto &d_pk = e.emit(
      "attention_sage2", prep_fn,
      {.width = U64(env_blocks_pad16) * 256u, .height = U64(kvh), .depth = 4u},
      {.width = 256u},
      {state_k, page_table, valid_kv_ref, k_int8_ref, mu_k_ref, s_k_ref});
  e.dynamic(d_pk, 6, prep_bytes, 3, 4);
}

inline void emit_minference_build_sparse_indices(
    Encoder &e, const QwenModelConfig &c, U32 t, U32 context_envelope,
    Ref query, Ref sage2_kv_ws, const Sage2KvWorkspaceLayout &kv_lo,
    Ref attention_positions, Ref scratch,
    const MInferenceScratchLayout &lo,
    const std::vector<std::byte> &base_params, U32 forced_wm = 0u) {
  if (scratch.buffer == nullptr || scratch.bytes < lo.total_bytes) {
    throw std::invalid_argument("MInference scratch buffer too small");
  }
  if (sage2_kv_ws.buffer == nullptr || sage2_kv_ws.bytes < kv_lo.total_bytes) {
    throw std::invalid_argument("MInference Sage2 KV workspace buffer too small");
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
  // Keep split_blocks a multiple of 16 so every split's kb_start aligns to
  // 16-block NAX mu_K tiles inside sage2_kv_ws without boundary checks.
  const U32 max_splits = std::max<U32>(
      1u, std::min<U32>(lo.split_count,
                        std::min<U32>(kMInferenceMaxSplits,
                                      (nk_blocks_est + 15u) / 16u)));
  const U32 raw_sb = (nk_blocks_est + max_splits - 1u) / max_splits;
  const U32 split_blocks = std::max<U32>(16u, (raw_sb + 15u) & ~15u);
  const U32 split_count = std::max<U32>(
      1u, std::min<U32>(lo.split_count,
                        (nk_blocks_est + split_blocks - 1u) / split_blocks));
  const auto budget = minference_budget_for_kv_tokens(context_envelope);

  std::array<U32, 32> mp{};
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
  mp[28] = U32(kv_lo.max_kv_blocks);
  mp[29] = U32(kv_lo.pad_kv_blocks);
  mp[30] = 0u;
  mp[31] = 0u;
  const auto mp_span = std::as_bytes(std::span<const U32>(mp));
  const std::vector<std::byte> mp_bytes(mp_span.begin(), mp_span.end());

  const auto sub_ref = [&](U64 off, U64 bytes) -> Ref {
    return {scratch.buffer, scratch.offset + off, bytes};
  };
  const auto sub_ws = [&](U64 off, U64 bytes) -> Ref {
    return {sage2_kv_ws.buffer, sage2_kv_ws.offset + off, bytes};
  };
  const Ref k_int8_ref = sub_ws(kv_lo.k_int8_off, kv_lo.k_int8_bytes);
  const Ref mu_k_ref = sub_ws(kv_lo.mu_k_off, kv_lo.mu_k_bytes);
  const Ref s_k_ref = sub_ws(kv_lo.s_k_off, kv_lo.s_k_bytes);
  const Ref valid_kv_ref = sub_ws(kv_lo.valid_kv_off, kv_lo.valid_kv_bytes);

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

  const U32 bm_tok_p = std::max<U32>(1u, 32u / std::max<U32>(1u, repeat));
  const U32 n_probe_qblks = (w_probe + bm_tok_p - 1u) / bm_tok_p;

  auto &d_stats = e.emit(
      "attention_minference", "omlx_m5_minference_probe_stats_i8",
      {.width = U64(split_count) * 256u, .height = U64(n_probe_qblks) * kvh},
      {.width = 256u},
      {query, k_int8_ref, mu_k_ref, s_k_ref, valid_kv_ref, attention_positions,
       m_split, l_split});
  e.dynamic(d_stats, 8, mp_bytes, 3, 4);

  const U64 clear_or_rows = std::max<U64>(
      U64(w_probe) * qh, U64(kvh) * lo.max_kv_blocks);
  auto &d_reduce = e.emit(
      "attention_minference", "omlx_m5_minference_probe_reduce_and_clear",
      {.width = ((clear_or_rows + 255ull) / 256ull) * 256ull}, {.width = 256u},
      {m_split, l_split, m_glob, inv_z, v_score, d_score});
  e.dynamic(d_reduce, 6, mp_bytes, 3, 4);

  auto &d_score_op = e.emit(
      "attention_minference", "omlx_m5_minference_probe_accum_i8",
      {.width = U64(split_count) * 256u, .height = U64(n_probe_qblks) * kvh},
      {.width = 256u},
      {query, k_int8_ref, mu_k_ref, s_k_ref, valid_kv_ref, attention_positions,
       m_glob, inv_z, v_score, d_score});
  e.dynamic(d_score_op, 10, mp_bytes, 3, 4);

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
    const std::vector<std::byte> &base_params, Ref sage2_kv_ws,
    const Sage2KvWorkspaceLayout &kv_lo,
    std::optional<Ref> minf_scratch = std::nullopt,
    std::optional<Sage2PvMode> pv_mode_override = std::nullopt,
    U32 forced_wm = 0u) {
  const U32 qh = U32(c.num_attention_heads);
  const U32 kvh = U32(c.num_key_value_heads);
  if (c.head_dim != 256 || kvh == 0u || (qh % kvh) != 0u) {
    throw std::invalid_argument("SageAttention2 requires hd256 and valid GQA");
  }
  if (sage2_kv_ws.buffer == nullptr || sage2_kv_ws.bytes < kv_lo.total_bytes) {
    throw std::invalid_argument("SageAttention2 requires valid sage2_kv_ws buffer");
  }
  const auto tier = select_sage2_tier(t, qh, kvh, forced_wm);
  const U32 repeat = qh / kvh;
  const U32 m_tile = tier.wm * 16u;
  const Sage2PvMode pv_mode = resolve_sage2_pv_mode(pv_mode_override);

  // WP2.1: Run O(N_KV) K-smoothing prepass ONCE per layer before optional MInference & Sage2!
  emit_sage2_prepare_k(e, c, context_envelope, is_affine4, state_k, page_table,
                       attention_positions, sage2_kv_ws, kv_lo, base_params);

  // Automatic short-context bypass for MInference (< 4096 KV tokens).
  const bool use_minf = sparse_minference &&
                        (context_envelope >= kMInferenceMinContextTokens) &&
                        minf_scratch.has_value() &&
                        minf_scratch->buffer != nullptr;
  // Dense mode uses flattened (q_tok, h_sub) packing (100% tile utilization on GQA 6 & 8);
  // Sparse MInference uses integer bm_tok blocks aligned with active_indices.
  const U32 q_blocks = use_minf
                           ? ((t + tier.bm_tok - 1u) / tier.bm_tok)
                           : ((t * repeat + m_tile - 1u) / m_tile);

  Ref act_idx = page_table; // Unused dummy binding when !use_minf
  Ref act_cnt = page_table;
  U32 max_q_blocks = q_blocks;
  U32 active_stride = kMInferenceActiveStride;

  if (use_minf) {
    const auto lo = plan_minference_scratch(t, qh, kvh, context_envelope,
                                            forced_wm);
    emit_minference_build_sparse_indices(
        e, c, t, context_envelope, query, sage2_kv_ws, kv_lo,
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

  std::array<U32, 20> sp{};
  if (base_params.size() < 14u * sizeof(U32)) {
    throw std::invalid_argument("SageAttention2 base_params too short");
  }
  std::memcpy(sp.data(), base_params.data(), 14u * sizeof(U32));
  sp[14] = max_q_blocks;
  sp[15] = active_stride;
  sp[16] = use_minf ? 1u : 0u;
  sp[17] = U32(kv_lo.max_kv_blocks);
  sp[18] = U32(kv_lo.pad_kv_blocks);
  sp[19] = 0u;
  const auto sp_span = std::as_bytes(std::span<const U32>(sp));
  std::vector<std::byte> sp_bytes(sp_span.begin(), sp_span.end());

  const auto sub_ws = [&](U64 off, U64 bytes) -> Ref {
    return {sage2_kv_ws.buffer, sage2_kv_ws.offset + off, bytes};
  };
  const Ref k_int8_ref = sub_ws(kv_lo.k_int8_off, kv_lo.k_int8_bytes);
  const Ref mu_k_ref = sub_ws(kv_lo.mu_k_off, kv_lo.mu_k_bytes);
  const Ref s_k_ref = sub_ws(kv_lo.s_k_off, kv_lo.s_k_bytes);
  const Ref valid_kv_ref = sub_ws(kv_lo.valid_kv_off, kv_lo.valid_kv_bytes);

  const char *fn = sage2_kernel_name(tier.wm, is_affine4, pv_mode, use_minf);
  auto &attn = e.emit(
      "attention_sage2", fn,
      {.width = tier.tg_threads, .height = U64(q_blocks) * kvh},
      {.width = tier.tg_threads},
      {query, gate, k_int8_ref, state_v, mixer, page_table, attention_positions,
       mu_k_ref, s_k_ref, valid_kv_ref, act_idx, act_cnt});
  e.profile.attention_kernel = fn;
  e.profile.attention_grid = attn.grid;
  e.profile.attention_numeric_contract_id =
      std::string("paged.sage2.") + (is_affine4 ? "affine4_g128." : "bf16.") +
      sage2_pv_mode_name(pv_mode) + (use_minf ? ".minference_vs.v1" : ".dense.v1");
  e.profile.attention_row_exact_eligible = false;
  e.dynamic(attn, 12, std::move(sp_bytes), 3, 4);
}

} // namespace omlx::m5::detail::forward
