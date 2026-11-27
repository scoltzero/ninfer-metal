#pragma once

#include "forward_plan.h"
#include "nax_common.h"

#include <cstdlib>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace omlx::m5::detail::forward {

inline constexpr U32 kGdnChunkNaxSize = 64U;
inline constexpr U32 kGdnChunkNaxCtrlStrideBytes = 17152U;
inline constexpr U32 kGdnChunkNaxCtrlStrideFloats = 4288U;
inline constexpr U32 kGdnChunkNaxDefaultMinTokens = 64U;

[[nodiscard]] inline bool gdn_tb8_enabled() noexcept {
  if (const char *disable = std::getenv("OMLX_M5_DISABLE_GDN_TB8");
      disable && std::string_view(disable) == "1") {
    return false;
  }
  return true;
}

[[nodiscard]] inline bool gdn_chunk_nax_enabled() noexcept {
  if (const char *disable = std::getenv("OMLX_M5_DISABLE_GDN_CHUNK_NAX");
      disable && std::string_view(disable) == "1") {
    return false;
  }
  return true;
}

[[nodiscard]] inline bool gdn_chunk_nax_forced_by_env() noexcept {
  if (const char *force = std::getenv("OMLX_M5_FORCE_GDN_CHUNK_NAX");
      force && std::string_view(force) == "1") {
    return true;
  }
  return false;
}

[[nodiscard]] inline U32 gdn_chunk_nax_min_tokens() noexcept {
  if (const char *env = std::getenv("OMLX_M5_GDN_CHUNK_MIN_T")) {
    const unsigned long parsed = std::strtoul(env, nullptr, 10);
    if (parsed >= 1UL && parsed <= 4096UL) {
      return static_cast<U32>(parsed);
    }
  }
  return kGdnChunkNaxDefaultMinTokens;
}

// Selects Stage 2 slice width (`dv64` vs `dv32`):
//   - `OMLX_M5_GDN_STAGE2_SLICE=64|32` overrides the heuristic.
//   - Real 80-core Apple10 M5 Ultra calibration: on 27B (Hv=48), dv64 launches
//     96 heavy TGs (2 waves with 64 idle cores on wave 2 + 2x register pressure),
//     regressing 64K prefill by +0.46s vs dv32 (192 lightweight <=16 KiB TGs).
//     Default to dv32 across Apple10 unless explicitly requested via `=64`.
[[nodiscard]] inline bool gdn_stage2_use_dv64(U32 value_heads,
                                              U32 gpu_cores = 80U) noexcept {
  if (const char *env = std::getenv("OMLX_M5_GDN_STAGE2_SLICE")) {
    const std::string_view sv(env);
    if (sv == "64") {
      return true;
    }
    if (sv == "32") {
      return false;
    }
  }
  (void)value_heads;
  (void)gpu_cores;
  return false;
}

struct GdnChunkWorkspaceLayout {
  U64 qk_bytes = 0;
  U64 ctrl_bytes = 0;
  U64 total_bytes = 0;
  U32 max_chunks = 0;
};

[[nodiscard]] inline GdnChunkWorkspaceLayout
gdn_chunk_nax_workspace_layout(U32 max_rows, U32 key_heads, U32 value_heads) {
  const U32 max_chunks = (max_rows + kGdnChunkNaxSize - 1U) / kGdnChunkNaxSize;
  const U64 qk_bytes = mul(mul(U64(max_chunks), U64(key_heads)),
                           2ULL * U64(kGdnChunkNaxSize) * 128ULL * 2ULL);
  const U64 ctrl_bytes = mul(mul(U64(max_chunks), U64(value_heads)),
                             U64(kGdnChunkNaxCtrlStrideBytes));
  if (ctrl_bytes > std::numeric_limits<U64>::max() - qk_bytes) {
    throw std::overflow_error("GDN chunk NAX workspace size overflow");
  }
  return {.qk_bytes = qk_bytes,
          .ctrl_bytes = ctrl_bytes,
          .total_bytes = qk_bytes + ctrl_bytes,
          .max_chunks = max_chunks};
}

[[nodiscard]] inline bool gdn_chunk_nax_runtime_eligible(
    const MetalContext &ctx, const std::filesystem::path &libs, U32 key_heads,
    U32 value_heads, U32 key_dim, U32 value_dim) {
  if (!gdn_chunk_nax_enabled()) {
    return false;
  }
  if (key_dim != 128U || value_dim != 128U || key_heads == 0U ||
      value_heads < key_heads || (value_heads % key_heads) != 0U) {
    return false;
  }
  if (nax_apple10_eligible(ctx, libs, "omlx_m5_gdn_chunk_nax.metallib",
                           "OMLX_M5_DISABLE_GDN_CHUNK_NAX")) {
    return true;
  }
  if (gdn_chunk_nax_forced_by_env()) {
    const auto &p = ctx.capabilities().profile.probe;
    return p.runtime_os_major >= 27 && p.metal4_compiler && p.tensor_api &&
           std::filesystem::is_regular_file(libs /
                                            "omlx_m5_gdn_chunk_nax.metallib");
  }
  return false;
}

// Emits GDN causal depthwise conv1d + SiLU:
// - T >= 4 and (conv_width % 4) == 0: coalesced bfloat4 x 8-token sliding-window
//   `omlx_m5_gdn_conv_silu_vec4_t8` (0 B TG SRAM, 0 ULP bit-identical) followed
//   by `omlx_m5_gdn_commit_history`.
// - T >= 4 (unaligned conv_width): `omlx_m5_gdn_conv_silu_2d` + `commit_history`.
// - T < 4: single-dispatch `omlx_m5_gdn_conv_silu`.
inline void emit_gdn_conv_silu(Encoder &e, U32 capacity, U32 conv_width,
                               Ref qkv, Ref conv_weight, Ref history,
                               Ref convolved,
                               const std::vector<std::byte> &params,
                               bool dynamic_tokens = true) {
  const std::optional<U32> tok_word =
      dynamic_tokens ? std::optional<U32>{6U} : std::optional<U32>{};
  if (capacity >= 4U) {
    if ((conv_width % 4U) == 0U) {
      auto &conv_vec = e.emit(
          "gdn_layer", "omlx_m5_gdn_conv_silu_vec4_t8",
          {.width = conv_width / 4U, .height = (capacity + 7U) / 8U},
          {.width = 64U, .height = 2U}, {qkv, conv_weight, history, convolved});
      e.dynamic(conv_vec, 4, params, tok_word);
    } else {
      auto &conv2d = e.emit("gdn_layer", "omlx_m5_gdn_conv_silu_2d",
                            {.width = conv_width, .height = capacity},
                            {.width = 64, .height = 4},
                            {qkv, conv_weight, history, convolved});
      e.dynamic(conv2d, 4, params, tok_word);
    }
    auto &commit_hist =
        e.emit("gdn_layer", "omlx_m5_gdn_commit_history", {.width = conv_width},
               {.width = 256}, {history, qkv});
    e.dynamic(commit_hist, 2, params, tok_word);
  } else if (capacity == 1U && (conv_width % 4U) == 0U) {
    auto &conv_t1 = e.emit("gdn_layer", "omlx_m5_gdn_conv_silu_vec4_t1",
                           {.width = conv_width / 4U}, {.width = 64U},
                           {qkv, conv_weight, history, convolved});
    e.dynamic(conv_t1, 4, params, tok_word);
  } else {
    auto &conv1d =
        e.emit("gdn_layer", "omlx_m5_gdn_conv_silu", {.width = conv_width},
               {.width = 256}, {qkv, conv_weight, history, convolved});
    e.dynamic(conv1d, 4, params, tok_word);
  }
}

// Emits speculative verify GDN recurrence without mutating `state0` (reads
// `state0` as `device const float*`, eliminating 151 MiB/slot `.scratch` and
// 453 MiB/round copy+writeback traffic). Uses 4-value-row-per-SG fast path
// when `key_dim == 128 && (value_dim % 4) == 0` (0 ULP identical).
inline void emit_gdn_recurrent_verify(
    Encoder &e, U32 gv, U32 key_dim, U32 value_dim, Ref convolved, Ref a, Ref b,
    Ref a_log, Ref dt_bias, Ref state0, Ref recurrent, Ref replay_keys,
    Ref replay_decays, Ref replay_deltas,
    const std::vector<std::byte> &params) {
  if (key_dim == 128U && (value_dim % 4U) == 0U && (gv % 4U) == 0U) {
    auto &rec = e.emit("gdn_layer", "omlx_m5_gdn_recurrent_record_4row",
                       {.width = 32, .height = gv / 4U}, {.width = 32},
                       {convolved, a, b, a_log, dt_bias, state0, recurrent,
                        replay_keys, replay_decays, replay_deltas});
    e.dynamic(rec, 10, params, 6);
  } else {
    auto &rec = e.emit("gdn_layer", "omlx_m5_gdn_recurrent_record",
                       {.width = 32, .height = gv}, {.width = 32},
                       {convolved, a, b, a_log, dt_bias, state0, recurrent,
                        replay_keys, replay_decays, replay_deltas});
    e.dynamic(rec, 10, params, 6);
  }
}

// Monolithic Tier-1 GDN Prefill Subgraph (`C = 64`, `BF16-NAX + FP32-State`):
// Fuses causal Conv1D+SiLU(qkv, history) + shared-key-head Q/K norm & NAX solve +
// register-resident [128 x 128] FP32 recurrence + in-SRAM 0-ULP RMSNorm_128 * SiLU(z)
// -> BF16 (`mixer`) or FP16 (`f16_scratch` + tail pad), eliminating BOTH `convolved`
// and `recurrent` DRAM round-trips and `gdn_norm_gate` launch across all GDN layers.
[[nodiscard]] inline bool try_emit_gdn_mega_prefill(
    Encoder &e, U32 capacity, U32 gv, U32 key_heads, U32 value_heads,
    U32 key_dim, U32 value_dim, U32 conv_width, Ref qkv, Ref conv_weight,
    Ref history, Ref a, Ref b, Ref a_log, Ref dt_bias, Ref state0, Ref z,
    Ref norm_weight, Ref dst_out, bool out_f16, Ref gdn_chunk_ws,
    bool allow_chunk_nax, const std::vector<std::byte> &params_8w,
    const std::vector<std::byte> &mega_params_9w, bool dynamic_tokens,
    U32 gpu_cores = 80U, Ref a8_qa = {}, Ref a8_sa = {}, Ref a8_ra = {},
    U32 a8_mp = 0U, bool a8_write_bf16 = false, float norm_eps = 1.0e-6f) {
  if (!allow_chunk_nax || gdn_chunk_ws.buffer == nullptr ||
      !gdn_chunk_nax_enabled() || capacity < gdn_chunk_nax_min_tokens() ||
      key_dim != 128U || value_dim != 128U || key_heads == 0U ||
      value_heads < key_heads || (value_heads % key_heads) != 0U ||
      gv != value_heads * value_dim) {
    return false;
  }
  const auto ws =
      gdn_chunk_nax_workspace_layout(capacity, key_heads, value_heads);
  if (ws.total_bytes > gdn_chunk_ws.bytes) {
    throw std::invalid_argument("GDN chunk NAX workspace buffer too small");
  }
  const std::optional<U32> tok_word =
      dynamic_tokens ? std::optional<U32>{6U} : std::optional<U32>{};
  const Ref qk_ref{gdn_chunk_ws.buffer, gdn_chunk_ws.offset, ws.qk_bytes};
  const Ref ctrl_ref{gdn_chunk_ws.buffer, gdn_chunk_ws.offset + ws.qk_bytes,
                     ws.ctrl_bytes};

  auto &prep = e.emit(
      "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_conv_c64",
      {.width = U64(key_heads) * 128U, .height = ws.max_chunks},
      {.width = 128U},
      {qkv, conv_weight, history, a, b, a_log, dt_bias, qk_ref, ctrl_ref});
  e.dynamic(prep, 9, mega_params_9w, tok_word);

  const bool use_dv64 = gdn_stage2_use_dv64(value_heads, gpu_cores);
  auto &rec = e.emit(
      "gdn_chunk_nax",
      use_dv64 ? "omlx_m5_gdn_chunk_recurrence_conv_dv64_c64"
               : "omlx_m5_gdn_chunk_recurrence_conv_dv32_c64",
      {.width = U64(value_heads) * 128U, .height = use_dv64 ? 2U : 4U},
      {.width = 128U},
      {qkv, conv_weight, history, qk_ref, ctrl_ref, state0, dst_out});
  e.dynamic(rec, 7, mega_params_9w, tok_word);

  const bool use_w4a8_stage3 =
      (!out_f16 && a8_qa.buffer != nullptr && a8_sa.buffer != nullptr &&
       a8_ra.buffer != nullptr);
  if (use_w4a8_stage3) {
    const U32 mp = a8_mp ? a8_mp : (((capacity + 63U) / 64U) * 64U);
    const U64 ng_heads = U64(mp) * U64(value_heads);
    auto &ng = e.emit(
        "w4a8_nax", "omlx_m5_gdn_chunk_norm_gate_inplace_a8_g64_c64",
        {.width = 128U, .height = (ng_heads + 3U) / 4U}, {.width = 128U},
        {dst_out, z, norm_weight, a8_qa, a8_sa, a8_ra});
    e.dynamic(ng, 6,
              words({key_heads, value_heads, 128U, 128U, conv_width, 4U,
                     capacity, word(norm_eps), mp, a8_write_bf16 ? 1U : 0U}),
              tok_word);
  } else {
    const U64 ng_rows =
        out_f16 ? U64(((capacity + 63U) / 64U) * 64U) : U64(capacity);
    const U64 ng_heads = ng_rows * U64(value_heads);
    auto &ng = e.emit(
        "gdn_chunk_nax",
        out_f16 ? "omlx_m5_gdn_chunk_norm_gate_inplace_f16_c64"
                : "omlx_m5_gdn_chunk_norm_gate_inplace_bf16_c64",
        {.width = 128U, .height = (ng_heads + 3U) / 4U}, {.width = 128U},
        {dst_out, z, norm_weight});
    e.dynamic(ng, 3, mega_params_9w, tok_word);
  }

  auto &commit_hist =
      e.emit("gdn_layer", "omlx_m5_gdn_commit_history", {.width = conv_width},
             {.width = 256U}, {history, qkv});
  e.dynamic(commit_hist, 2, params_8w, tok_word);

  e.profile.gdn_numeric_contract_id =
      use_w4a8_stage3 ? "gdn.chunk64_bf16_nax.f32_state.mega_a8_v1"
                      : "gdn.chunk64_bf16_nax.f32_state.mega_v1";
  return true;
}

// Emits standalone GDN forward/prefill recurrence (given pre-convolved QKV):
// - When `allow_chunk_nax` is true, `capacity >= gdn_chunk_nax_min_tokens()`
//   (default 64), and `key_dim == 128 && value_dim == 128`: emits two-stage
//   C=64 BF16-NAX + FP32-state GDN prefill (`omlx_m5_gdn_chunk_prepare_c64` +
//   `omlx_m5_gdn_chunk_recurrence_c64`, 16.0 KiB TG SRAM, Hv*4 TGs).
// - Else when `capacity >= tb8_min_tokens` (default 8) and `key_dim == 128` and
//   `(value_dim % 16) == 0`: emits exact-recurrence threadgroup-staged `TB=8`
//   `omlx_m5_gdn_recurrent_tb8` (0 ULP bit-identical to `omlx_m5_gdn_recurrent`).
// - Otherwise: emits `omlx_m5_gdn_recurrent`.
inline void emit_gdn_recurrent_forward(
    Encoder &e, U32 capacity, U32 gv, U32 key_dim, U32 value_dim, Ref convolved,
    Ref a, Ref b, Ref a_log, Ref dt_bias, Ref state0, Ref recurrent,
    const std::vector<std::byte> &params, U32 tb8_min_tokens = 8U,
    Ref gdn_chunk_ws = {}, bool allow_chunk_nax = false, U32 key_heads = 0U,
    U32 value_heads = 0U, bool dynamic_tokens = true) {
  const std::optional<U32> tok_word =
      dynamic_tokens ? std::optional<U32>{6U} : std::optional<U32>{};
  if (allow_chunk_nax && gdn_chunk_ws.buffer != nullptr &&
      gdn_chunk_nax_enabled() && capacity >= gdn_chunk_nax_min_tokens() &&
      key_dim == 128U && value_dim == 128U && key_heads > 0U &&
      value_heads >= key_heads && (value_heads % key_heads) == 0U &&
      gv == value_heads * value_dim) {
    const auto ws =
        gdn_chunk_nax_workspace_layout(capacity, key_heads, value_heads);
    if (ws.total_bytes > gdn_chunk_ws.bytes) {
      throw std::invalid_argument("GDN chunk NAX workspace buffer too small");
    }
    const Ref qk_ref{gdn_chunk_ws.buffer, gdn_chunk_ws.offset, ws.qk_bytes};
    const Ref ctrl_ref{gdn_chunk_ws.buffer, gdn_chunk_ws.offset + ws.qk_bytes,
                       ws.ctrl_bytes};
    auto &prep = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_c64",
        {.width = U64(key_heads) * 128U, .height = ws.max_chunks},
        {.width = 128U},
        {convolved, a, b, a_log, dt_bias, qk_ref, ctrl_ref});
    e.dynamic(prep, 7, params, tok_word);
    auto &rec = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_recurrence_c64",
        {.width = U64(value_heads) * 128U, .height = 4U},
        {.width = 128U},
        {convolved, qk_ref, ctrl_ref, state0, recurrent});
    e.dynamic(rec, 5, params, tok_word);
    e.profile.gdn_numeric_contract_id =
        "gdn.chunk64_bf16_nax.f32_state.conv_bf16.v1";
    return;
  }
  if (capacity >= tb8_min_tokens && key_dim == 128U &&
      (value_dim % 16U) == 0U && (gv % 16U) == 0U && gdn_tb8_enabled()) {
    auto &rec = e.emit("gdn_layer", "omlx_m5_gdn_recurrent_tb8",
                       {.width = 128, .height = gv / 16U}, {.width = 128},
                       {convolved, a, b, a_log, dt_bias, state0, recurrent});
    e.dynamic(rec, 7, params, tok_word);
  } else {
    auto &rec = e.emit("gdn_layer", "omlx_m5_gdn_recurrent",
                       {.width = 32, .height = gv}, {.width = 32},
                       {convolved, a, b, a_log, dt_bias, state0, recurrent});
    e.dynamic(rec, 7, params, tok_word);
  }
}

} // namespace omlx::m5::detail::forward
