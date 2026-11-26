#include "../src/chunk_blend_cache.h"
#include "../src/forward_plan.h"
#include "../src/gdn_prefill.h"
#include "omlx_m5/kv_page_pool.h"
#include "omlx_m5/metal.h"
#include "omlx_m5/quantization.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>

using namespace omlx::m5;
using namespace omlx::m5::detail;
using namespace omlx::m5::detail::forward;
using Json = nlohmann::json;

namespace {

void require(bool ok, std::string_view msg) {
  if (!ok) {
    throw std::runtime_error("CACHE-BLEND HYBRID TEST FAIL: " +
                             std::string(msg));
  }
}

template <typename T>
std::vector<std::byte> to_bytes(const T &val) {
  std::vector<std::byte> out(sizeof(T));
  std::memcpy(out.data(), &val, sizeof(T));
  return out;
}

float f16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = std::uint32_t(h & 0x8000U) << 16U;
  const std::uint32_t exp = (h >> 10U) & 0x1FU;
  const std::uint32_t mant = h & 0x03FFU;
  if (exp == 0U) {
    if (mant == 0U) {
      return std::bit_cast<float>(sign);
    }
    const float mag = std::ldexp(float(mant), -24);
    return (sign != 0U) ? -mag : mag;
  }
  if (exp == 31U) {
    return std::bit_cast<float>(sign | 0x7F800000U | (mant << 13U));
  }
  return std::bit_cast<float>(sign | ((exp + 112U) << 23U) | (mant << 13U));
}

struct VecMetrics {
  double cos_sim = 0.0;
  double max_abs = 0.0;
  double rel_l2 = 0.0;
};

VecMetrics compare_vectors(std::span<const float> actual,
                           std::span<const float> expected,
                           std::string_view label) {
  require(actual.size() == expected.size() && !actual.empty(),
          std::string(label) + ": size mismatch");
  double dot = 0.0;
  double na = 0.0;
  double nb = 0.0;
  double diff_sq = 0.0;
  double max_abs = 0.0;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    require(std::isfinite(actual[i]) && std::isfinite(expected[i]),
            std::string(label) + ": non-finite value at " + std::to_string(i));
    const double a = double(actual[i]);
    const double b = double(expected[i]);
    dot += a * b;
    na += a * a;
    nb += b * b;
    const double d = std::abs(a - b);
    max_abs = std::max(max_abs, d);
    diff_sq += d * d;
  }
  const double denom = std::sqrt(std::max(na, 1e-24) * std::max(nb, 1e-24));
  return VecMetrics{
      .cos_sim = dot / denom,
      .max_abs = max_abs,
      .rel_l2 = std::sqrt(diff_sq / std::max(nb, 1e-24)),
  };
}

std::vector<float> read_f32(const MetalBuffer &buf, std::size_t n) {
  const auto raw = buf.read(0, n * sizeof(float));
  std::vector<float> out(n);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

std::vector<std::uint32_t> read_u32(const MetalBuffer &buf, std::size_t n) {
  const auto raw = buf.read(0, n * sizeof(std::uint32_t));
  std::vector<std::uint32_t> out(n);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

std::vector<std::uint16_t> read_u16(const MetalBuffer &buf, std::size_t n) {
  const auto raw = buf.read(0, n * sizeof(std::uint16_t));
  std::vector<std::uint16_t> out(n);
  std::memcpy(out.data(), raw.data(), raw.size());
  return out;
}

std::vector<float> bf16_buf_to_f32(const MetalBuffer &buf, std::size_t n) {
  const auto u16 = read_u16(buf, n);
  std::vector<float> out(n);
  for (std::size_t i = 0; i < n; ++i) {
    out[i] = bf16_to_float(u16[i]);
  }
  return out;
}

void exec_profile(const MetalContext &ctx, const Profile &prof) {
  const std::array inv = {MetalCommandInvocation{
      .plan = &prof.plan,
      .bindings = prof.bindings,
      .parameters = &prof.parameters}};
  (void)ctx.execute_batch(inv);
}

// Dequantize a single [256] vector from a KvPagePool Affine4Group128 slab.
std::array<float, 256> dequant_kv_slot_256(std::span<const std::byte> slab,
                                           U32 phys_page, U32 kvh, U32 within,
                                           U32 page_tokens, U32 kv_heads) {
  const std::size_t head_code_bytes = std::size_t(page_tokens) * 128U;
  const std::size_t all_codes_bytes = std::size_t(kv_heads) * head_code_bytes;
  const std::size_t page_bytes =
      std::size_t(kv_page_bytes(KvStorageFormat::Affine4Group128, page_tokens,
                                kv_heads, 256U));
  const auto *page_ptr =
      reinterpret_cast<const std::uint8_t *>(slab.data()) +
      std::size_t(phys_page) * page_bytes;
  const auto *codes_head = page_ptr + std::size_t(kvh) * head_code_bytes;
  const auto *meta_head =
      reinterpret_cast<const KvAffine4VecMeta *>(page_ptr + all_codes_bytes) +
      std::size_t(kvh) * page_tokens;
  const KvAffine4VecMeta &m = meta_head[within];
  std::array<float, 256> out{};
  for (U32 g = 0; g < 2U; ++g) {
    const float sc = f16_to_f32(m.scale_f16[g]);
    const float zp = f16_to_f32(m.zero_f16[g]);
    const auto *g_codes =
        codes_head + (std::size_t(g) * page_tokens + within) * 64U;
    for (U32 d = 0; d < 128U; ++d) {
      const std::uint8_t byte = g_codes[d >> 1U];
      const std::uint8_t nib = (d & 1U) ? (byte >> 4U) : (byte & 0x0FU);
      out[g * 128U + d] = float(nib) * sc + zp;
    }
  }
  return out;
}

// ============================================================================
// Test 0: Host C++ Segmentation, Key/Collision Guard & Thread-Safe LRU Cache
// ============================================================================
Json test_host_segmentation_and_lru() {
  // 1. Delimiter-anchored + mandatory dense tail + W_bnd=64 boundary window.
  BlendSegmentConfig cfg;
  cfg.l_min = 256U;
  cfg.l_max = 4096U;
  cfg.boundary_window = 64U;
  cfg.dense_tail_tokens = 128U;
  cfg.min_suffix_for_blend = 512U;
  cfg.prefix_delimiters = {{151644U}, {999001U, 999002U}}; // <|im_start|> or <doc

  std::vector<U32> prompt;
  // Prefix: 128 tokens
  for (U32 i = 0; i < 128U; ++i) prompt.push_back(1000U + i);
  // Doc 1: 311 tokens (non-multiple of 64!) starting with <doc
  prompt.push_back(999001U);
  prompt.push_back(999002U);
  for (U32 i = 2; i < 311U; ++i) prompt.push_back(20000U + i);
  // Doc 2: 419 tokens (non-multiple of 64!) starting with <|im_start|>
  prompt.push_back(151644U);
  for (U32 i = 1; i < 419U; ++i) prompt.push_back(40000U + i);
  // Query tail: 128 tokens
  for (U32 i = 0; i < 128U; ++i) prompt.push_back(80000U + i);

  const auto segs = segment_blend_chunks(prompt, 128U, cfg);
  require(segs.size() == 3U, "expected [Doc1, Doc2, DenseTail]");
  require(segs[0].start == 128U && segs[0].length == 311U &&
              segs[0].kind == BlendSegmentKind::CacheableChunk &&
              segs[0].boundary_window == 64U,
          "Doc1 segment mismatch");
  require(segs[1].start == 439U && segs[1].length == 419U &&
              segs[1].kind == BlendSegmentKind::CacheableChunk &&
              segs[1].boundary_window == 64U,
          "Doc2 segment mismatch");
  require(segs[2].start == 858U && segs[2].length == 128U &&
              segs[2].kind == BlendSegmentKind::UncachedDense &&
              segs[2].boundary_window == 0U,
          "Trailing 128-token dense tail mismatch");

  // Mutating trailing query must NOT alter Doc1/Doc2 cut boundaries or hashes.
  auto prompt_edited = prompt;
  for (U32 i = 0; i < 128U; ++i) prompt_edited[858U + i] = 90000U + i * 3U;
  const auto segs_edited = segment_blend_chunks(prompt_edited, 128U, cfg);
  require(segs_edited.size() == 3U &&
              segs_edited[0].content_hash == segs[0].content_hash &&
              segs_edited[1].content_hash == segs[1].content_hash &&
              segs_edited[2].content_hash != segs[2].content_hash,
          "Query edit unexpectedly shifted preceding document chunk hashes");

  // 2. Deterministic Gear rolling hash split for unbroken span > l_max (9,000 tokens).
  std::vector<U32> long_span(9128U);
  for (U32 i = 0; i < long_span.size(); ++i) {
    long_span[i] = 100U + ((i * 2654435761U) >> 18U);
  }
  const auto gear_segs = segment_blend_chunks(long_span, 0U, cfg);
  require(gear_segs.size() >= 3U, "Gear CDC failed to split 9K unbroken span");
  require(gear_segs.back().kind == BlendSegmentKind::UncachedDense &&
              gear_segs.back().length >= 128U,
          "Gear CDC missing mandatory dense tail");
  U32 covered = 0U;
  for (std::size_t i = 0; i < gear_segs.size(); ++i) {
    require(gear_segs[i].start == covered, "Gear CDC segments non-contiguous");
    covered += gear_segs[i].length;
    if (gear_segs[i].kind == BlendSegmentKind::CacheableChunk) {
      require(gear_segs[i].length >= cfg.l_min &&
                  gear_segs[i].length <= cfg.l_max,
              "Gear CDC chunk violated [L_min, L_max]");
      require(gear_segs[i].boundary_window == 64U,
              "Gear CDC chunk missing W_bnd=64");
    }
  }
  require(covered == long_span.size(), "Gear CDC coverage mismatch");
  // Determinism check: identical call produces identical cuts.
  const auto gear_segs_2 = segment_blend_chunks(long_span, 0U, cfg);
  require(gear_segs.size() == gear_segs_2.size(), "Gear CDC non-deterministic");
  for (std::size_t i = 0; i < gear_segs.size(); ++i) {
    require(gear_segs[i].start == gear_segs_2[i].start &&
                gear_segs[i].length == gear_segs_2[i].length &&
                gear_segs[i].content_hash == gear_segs_2[i].content_hash,
            "Gear CDC cut mismatch on repeat call");
  }

  // 3. Qwen3.8-27B 2048-token payload byte-budget check (matching Section 2.3).
  BlendChunkGeometry g27b_fp8;
  g27b_fp8.token_length = 2048U;
  g27b_fp8.gdn_op_format = BlendGdnOperatorFormat::Fp8E4M3RowScale;
  const double mib_2k_fp8 = double(g27b_fp8.total_payload_bytes()) / (1024.0 * 1024.0);
  require(mib_2k_fp8 > 140.0 && mib_2k_fp8 < 185.0,
          "27B 2048-tok FP8 payload size outside expected ~167 MiB range");

  // 4. Byte-capped LRU eviction, exact token equality guard, exec-hash isolation,
  //    and multi-threaded safety.
  const U64 h_bf16 = compute_blend_model_exec_hash(
      "sha256:abc", "affine4_g128", "bf16", "exact_nax", false, false);
  const U64 h_w4a8 = compute_blend_model_exec_hash(
      "sha256:abc", "affine4_g128", "w4a8", "exact_nax", false, false);
  require(h_bf16 != h_w4a8, "model_exec_hash failed to isolate linear_mode");

  ChunkBlendCache cache(2500ULL); // Fits at most 2 entries of 1000 B each
  auto make_entry = [&](U64 exec_h, std::vector<U32> toks, U64 bytes) {
    auto e = std::make_shared<BlendChunkEntry>();
    e->key = make_blend_chunk_key(exec_h, toks);
    e->tokens = std::move(toks);
    e->accounted_bytes = bytes;
    return e;
  };
  const std::vector<U32> tA(256U, 11U);
  const std::vector<U32> tB(256U, 22U);
  const std::vector<U32> tC(256U, 33U);
  std::vector<U32> tA_alt = tA;
  tA_alt[128U] = 999U; // Same first/last token & length, different middle!

  require(cache.insert(make_entry(h_bf16, tA, 1000ULL)), "insert A failed");
  require(cache.insert(make_entry(h_bf16, tB, 1000ULL)), "insert B failed");
  require(cache.lookup(h_bf16, tA_alt) == nullptr,
          "exact token equality guard failed");
  require(cache.lookup(h_w4a8, tA) == nullptr,
          "exec_hash isolation failed");
  // Touch A so B becomes LRU victim.
  auto hitA = cache.lookup(h_bf16, tA);
  require(hitA != nullptr && hitA->hit_count == 1U, "lookup A failed");
  // Insert C -> must evict B, keep A and C.
  require(cache.insert(make_entry(h_bf16, tC, 1000ULL)), "insert C failed");
  require(cache.lookup(h_bf16, tB) == nullptr, "LRU did not evict B");
  require(cache.lookup(h_bf16, tA) != nullptr, "LRU evicted MRU A");
  require(cache.lookup(h_bf16, tC) != nullptr, "C missing after insert");
  require(cache.cached_bytes() == 2000ULL, "cached_bytes accounting mismatch");

  // Multi-threaded concurrent stress test.
  ChunkBlendCache mt_cache(16384ULL);
  std::vector<std::thread> threads;
  for (int tid = 0; tid < 4; ++tid) {
    threads.emplace_back([&, tid]() {
      for (U32 k = 0; k < 64U; ++k) {
        std::vector<U32> tok(256U, U32(tid * 1000 + int(k)));
        mt_cache.insert(make_entry(h_bf16, tok, 1024ULL));
        (void)mt_cache.lookup(h_bf16, tok);
      }
    });
  }
  for (auto &th : threads) th.join();
  require(mt_cache.cached_bytes() <= mt_cache.budget_bytes(),
          "MT cache exceeded byte cap");

  return {{"delimiter_segments", segs.size()},
          {"gear_segments_9k", gear_segs.size()},
          {"qwen27b_2k_fp8_mib", mib_2k_fp8},
          {"lru_evictions", cache.stats().evictions},
          {"mt_cached_bytes", mt_cache.cached_bytes()}};
}

// ============================================================================
// Test 1: GDN Exact Affine State Composition Parity (`Hv = 48`)
//   Proves `S_out = S_in * M_C + B_C` matches dense `gdn_chunk_nax` recurrence
//   (`cos_sim >= 0.9998`, `max_abs_err < 1e-3`) across all `Hv = 48` heads,
//   plus verifies FP8 E4M3 storage mode and decayed rank-1 HKVD perturbation.
// ============================================================================
Json test_gdn_state_compose_parity(const MetalContext &ctx,
                                   const std::filesystem::path &libs) {
  constexpr U32 hk = 16U;
  constexpr U32 hv = 48U; // Exact Qwen3.8-27B GDN geometry!
  constexpr U32 kd = 128U;
  constexpr U32 vd = 128U;
  constexpr U32 gv = hv * vd;
  constexpr U32 conv_w = 2U * hk * kd + gv; // 10,240
  constexpr U32 L = 256U;                   // 4 C=64 interior chunks

  std::vector<std::uint16_t> conv_words(std::size_t(L) * conv_w);
  std::vector<std::uint16_t> a_words(std::size_t(L) * hv);
  std::vector<std::uint16_t> b_words(std::size_t(L) * hv);
  std::vector<std::uint16_t> a_log_words(hv);
  std::vector<std::uint16_t> dt_bias_words(hv);
  std::vector<float> s_in(std::size_t(hv) * vd * kd);

  for (std::size_t i = 0; i < conv_words.size(); ++i) {
    const float v = float(int((i * 37ULL + 13ULL) % 251ULL) - 125) / 128.0f;
    conv_words[i] = float_to_bf16(v);
  }
  for (std::size_t i = 0; i < a_words.size(); ++i) {
    const float av = float(int((i * 19ULL + 7ULL) % 97ULL) - 52) / 16.0f;
    const float bv = float(int((i * 29ULL + 5ULL) % 89ULL) - 44) / 16.0f;
    a_words[i] = float_to_bf16(av);
    b_words[i] = float_to_bf16(bv);
  }
  for (U32 h = 0; h < hv; ++h) {
    a_log_words[h] = float_to_bf16(-2.4f + 0.04f * float(h));
    dt_bias_words[h] = float_to_bf16(-0.8f + 0.03f * float(h));
  }
  for (std::size_t i = 0; i < s_in.size(); ++i) {
    s_in[i] = float(int((i * 43ULL + 17ULL) % 199ULL) - 99) / 384.0f;
  }

  auto conv_buf = ctx.upload(std::as_bytes(std::span(conv_words)));
  auto a_buf = ctx.upload(std::as_bytes(std::span(a_words)));
  auto b_buf = ctx.upload(std::as_bytes(std::span(b_words)));
  auto a_log_buf = ctx.upload(std::as_bytes(std::span(a_log_words)));
  auto dt_bias_buf = ctx.upload(std::as_bytes(std::span(dt_bias_words)));
  auto s_in_buf = ctx.upload(std::as_bytes(std::span(s_in)));
  auto s_dense_buf = ctx.upload(std::as_bytes(std::span(s_in)));
  auto s_zero_buf = ctx.make_buffer(s_in.size() * sizeof(float));
  s_zero_buf.zero();
  auto m_f32_buf = ctx.make_buffer(s_in.size() * sizeof(float));
  auto rec_dummy = ctx.make_buffer(std::size_t(L) * gv * sizeof(std::uint16_t));
  const auto ws_lo = gdn_chunk_nax_workspace_layout(L, hk, hv);
  auto ws_buf = ctx.make_buffer(ws_lo.total_bytes);
  const Ref qk_ref{&ws_buf, 0, ws_lo.qk_bytes};
  const Ref ctrl_ref{&ws_buf, ws_lo.qk_bytes, ws_lo.ctrl_bytes};

  // 1. Run dense gdn_chunk_nax from S_in -> S_dense_ref AND from S=0 -> B_C,
  //    plus extract M_C via `omlx_m5_gdn_extract_transition_m_c64`!
  const auto gdn_params =
      words({hk, hv, kd, vd, conv_w, 4U, L, word(1.0e-6f)});
  {
    Encoder e(ctx, libs, L, "test.blend.gdn.ref_and_extract");
    auto &prep = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_c64",
        {.width = U64(hk) * 128U, .height = ws_lo.max_chunks}, {.width = 128U},
        {{&conv_buf, 0, conv_buf.size()},
         {&a_buf, 0, a_buf.size()},
         {&b_buf, 0, b_buf.size()},
         {&a_log_buf, 0, a_log_buf.size()},
         {&dt_bias_buf, 0, dt_bias_buf.size()},
         qk_ref,
         ctrl_ref});
    e.dynamic(prep, 7, gdn_params);

    // Dense reference from non-zero S_in
    auto &rec_dense = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_recurrence_c64",
        {.width = U64(hv) * 128U, .height = 4U}, {.width = 128U},
        {{&conv_buf, 0, conv_buf.size()},
         qk_ref,
         ctrl_ref,
         {&s_dense_buf, 0, s_dense_buf.size()},
         {&rec_dummy, 0, rec_dummy.size()}});
    e.dynamic(rec_dense, 5, gdn_params);

    // Zero-initial-state B_C = S_T(0, V)
    auto &rec_b = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_recurrence_c64",
        {.width = U64(hv) * 128U, .height = 4U}, {.width = 128U},
        {{&conv_buf, 0, conv_buf.size()},
         qk_ref,
         ctrl_ref,
         {&s_zero_buf, 0, s_zero_buf.size()},
         {&rec_dummy, 0, rec_dummy.size()}});
    e.dynamic(rec_b, 5, gdn_params);

    // Homogeneous transition operator M_C = S_T(I_128, V=0)
    BlendGdnExtractMParams mp{.key_heads = hk,
                              .value_heads = hv,
                              .key_dim = kd,
                              .value_dim = vd,
                              .tokens = L};
    auto &ext_m = e.emit("cache_blend", "omlx_m5_gdn_extract_transition_m_c64",
                         {.width = 128U, .height = hv, .depth = 2U},
                         {.width = 128U},
                         {qk_ref, ctrl_ref, {&m_f32_buf, 0, m_f32_buf.size()}});
    e.dynamic(ext_m, 3, to_bytes(mp));
    exec_profile(ctx, e.finish(false));
  }

  const auto s_dense_ref = read_f32(s_dense_buf, s_in.size());

  // 2. Pack (M_C, B_C) and compose via `omlx_m5_gdn_state_compose_nax` in BOTH
  //    BF16 (format=0) and FP8 E4M3 + per-row FP16 scale (format=1).
  auto dummy_u32 = ctx.make_buffer(64);
  auto dummy_bf = ctx.make_buffer(256);
  auto dummy_f32 = ctx.make_buffer(64);
  auto cum_decay_buf = ctx.make_buffer(std::size_t(L) * hv * sizeof(std::uint16_t));
  {
    BlendGdnCumDecayParams cp{.value_heads = hv, .tokens = L};
    Encoder e(ctx, libs, 1, "test.blend.gdn.cumdecay");
    auto &d = e.emit("cache_blend", "omlx_m5_gdn_compute_cum_decay_f16",
                     {.width = hv}, {.width = 32U},
                     {{&a_buf, 0, a_buf.size()},
                      {&a_log_buf, 0, a_log_buf.size()},
                      {&dt_bias_buf, 0, dt_bias_buf.size()},
                      {&cum_decay_buf, 0, cum_decay_buf.size()}});
    e.dynamic(d, 4, to_bytes(cp));
    exec_profile(ctx, e.finish(false));
  }

  auto run_compose_format = [&](BlendGdnOperatorFormat fmt,
                                bool in_place) -> VecMetrics {
    const U64 elem_b =
        (fmt == BlendGdnOperatorFormat::Fp8E4M3RowScale) ? 1ULL : 2ULL;
    auto m_data = ctx.make_buffer(U64(hv) * kd * vd * elem_b);
    auto m_scales = ctx.make_buffer(U64(hv) * vd * sizeof(std::uint16_t));
    auto b_data = ctx.make_buffer(U64(hv) * kd * vd * elem_b);
    auto b_scales = ctx.make_buffer(U64(hv) * vd * sizeof(std::uint16_t));
    auto s_out_buf = ctx.upload(std::as_bytes(std::span(s_in)));

    BlendGdnPackOpParams pp{
        .value_heads = hv, .dim = 128U, .op_format = U32(fmt)};
    BlendGdnStateComposeParams cp{.key_heads = hk,
                                  .value_heads = hv,
                                  .dim = 128U,
                                  .op_format = U32(fmt),
                                  .chunk_tokens = L,
                                  .num_hkvd_perturb = 0U,
                                  .apply_bias = 1U};
    Encoder e(ctx, libs, 1,
              "test.blend.gdn.compose.fmt" + std::to_string(U32(fmt)) +
                  (in_place ? ".inplace" : ".oop"));
    auto &pm = e.emit("cache_blend", "omlx_m5_gdn_pack_operator",
                      {.width = 32U, .height = U64(hv) * 128U}, {.width = 32U},
                      {{&m_f32_buf, 0, m_f32_buf.size()},
                       {&m_data, 0, m_data.size()},
                       {&m_scales, 0, m_scales.size()}});
    e.dynamic(pm, 3, to_bytes(pp));
    auto &pb = e.emit("cache_blend", "omlx_m5_gdn_pack_operator",
                      {.width = 32U, .height = U64(hv) * 128U}, {.width = 32U},
                      {{&s_zero_buf, 0, s_zero_buf.size()},
                       {&b_data, 0, b_data.size()},
                       {&b_scales, 0, b_scales.size()}});
    e.dynamic(pb, 3, to_bytes(pp));

    const Ref in_ref = in_place ? Ref{&s_out_buf, 0, s_out_buf.size()}
                                : Ref{&s_in_buf, 0, s_in_buf.size()};
    auto &comp = e.emit(
        "cache_blend", "omlx_m5_gdn_state_compose_nax",
        {.width = 128U, .height = hv, .depth = 4U}, {.width = 128U},
        {in_ref,
         {&m_data, 0, m_data.size()},
         {&m_scales, 0, m_scales.size()},
         {&b_data, 0, b_data.size()},
         {&b_scales, 0, b_scales.size()},
         {&cum_decay_buf, 0, cum_decay_buf.size()},
         {&dummy_u32, 0, dummy_u32.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_f32, 0, dummy_f32.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_f32, 0, dummy_f32.size()},
         {&s_out_buf, 0, s_out_buf.size()}});
    e.dynamic(comp, 14, to_bytes(cp));
    exec_profile(ctx, e.finish(false));

    const auto s_got = read_f32(s_out_buf, s_in.size());
    return compare_vectors(s_got, s_dense_ref, "gdn_state_compose");
  };

  const auto m_bf16_oop =
      run_compose_format(BlendGdnOperatorFormat::BFloat16, false);
  const auto m_bf16_ip =
      run_compose_format(BlendGdnOperatorFormat::BFloat16, true);
  const auto m_f16_oop =
      run_compose_format(BlendGdnOperatorFormat::Float16RowScale, false);
  const auto m_fp8_oop =
      run_compose_format(BlendGdnOperatorFormat::Fp8E4M3RowScale, false);

  require(m_bf16_oop.cos_sim >= 0.9998 && m_bf16_oop.max_abs < 1.0e-3,
          "BF16 GDN compose (out-of-place) failed parity gate");
  require(m_bf16_ip.cos_sim >= 0.9998 && m_bf16_ip.max_abs < 1.0e-3,
          "BF16 GDN compose (in-place) failed parity gate");
  require(m_f16_oop.cos_sim >= 0.99995 && m_f16_oop.max_abs < 7.5e-4,
          "FP16-rowscale GDN compose failed tight parity gate (cos_sim=" +
              std::to_string(m_f16_oop.cos_sim) +
              ", max_abs=" + std::to_string(m_f16_oop.max_abs) + ")");
  require(m_fp8_oop.cos_sim >= 0.997,
          "FP8 E4M3 GDN compose failed fidelity gate");

  // 3. Verify decayed rank-1 HKVD perturbation DeltaS matches exact formula.
  constexpr U32 P = 4U;
  std::vector<U32> hkvd_t = {72U, 119U, 180U, 240U};
  std::vector<std::uint16_t> k_new(P * hk * kd), k_old(P * hk * kd);
  std::vector<std::uint16_t> v_new(P * hv * vd), v_old(P * hv * vd);
  std::vector<float> beta_new(P * hv), beta_old(P * hv);
  for (std::size_t i = 0; i < k_new.size(); ++i) {
    k_new[i] = float_to_bf16(float(int((i * 17ULL) % 41ULL) - 20) / 128.0f);
    k_old[i] = float_to_bf16(float(int((i * 23ULL) % 41ULL) - 20) / 128.0f);
  }
  for (std::size_t i = 0; i < v_new.size(); ++i) {
    v_new[i] = float_to_bf16(float(int((i * 31ULL) % 41ULL) - 20) / 64.0f);
    v_old[i] = float_to_bf16(float(int((i * 13ULL) % 41ULL) - 20) / 64.0f);
  }
  for (std::size_t i = 0; i < beta_new.size(); ++i) {
    beta_new[i] = 0.25f + 0.01f * float(i % 20U);
    beta_old[i] = 0.20f + 0.01f * float(i % 17U);
  }
  auto t_buf = ctx.upload(std::as_bytes(std::span(hkvd_t)));
  auto kn_buf = ctx.upload(std::as_bytes(std::span(k_new)));
  auto vn_buf = ctx.upload(std::as_bytes(std::span(v_new)));
  auto bn_buf = ctx.upload(std::as_bytes(std::span(beta_new)));
  auto ko_buf = ctx.upload(std::as_bytes(std::span(k_old)));
  auto vo_buf = ctx.upload(std::as_bytes(std::span(v_old)));
  auto bo_buf = ctx.upload(std::as_bytes(std::span(beta_old)));
  auto m_zero = ctx.make_buffer(U64(hv) * kd * vd * 2ULL);
  auto m_sc = ctx.make_buffer(U64(hv) * vd * 2ULL);
  m_zero.zero();
  m_sc.zero();
  auto delta_s_out = ctx.make_buffer(s_in.size() * sizeof(float));
  {
    BlendGdnStateComposeParams cp{.key_heads = hk,
                                  .value_heads = hv,
                                  .dim = 128U,
                                  .op_format = 0U,
                                  .chunk_tokens = L,
                                  .num_hkvd_perturb = P,
                                  .apply_bias = 0U};
    Encoder e(ctx, libs, 1, "test.blend.gdn.hkvd_perturb");
    auto &comp = e.emit(
        "cache_blend", "omlx_m5_gdn_state_compose_nax",
        {.width = 128U, .height = hv, .depth = 4U}, {.width = 128U},
        {{&s_in_buf, 0, s_in_buf.size()},
         {&m_zero, 0, m_zero.size()},
         {&m_sc, 0, m_sc.size()},
         {&m_zero, 0, m_zero.size()},
         {&m_sc, 0, m_sc.size()},
         {&cum_decay_buf, 0, cum_decay_buf.size()},
         {&t_buf, 0, t_buf.size()},
         {&kn_buf, 0, kn_buf.size()},
         {&vn_buf, 0, vn_buf.size()},
         {&bn_buf, 0, bn_buf.size()},
         {&ko_buf, 0, ko_buf.size()},
         {&vo_buf, 0, vo_buf.size()},
         {&bo_buf, 0, bo_buf.size()},
         {&delta_s_out, 0, delta_s_out.size()}});
    e.dynamic(comp, 14, to_bytes(cp));
    exec_profile(ctx, e.finish(false));
  }
  const auto cum_u16 = read_u16(cum_decay_buf, std::size_t(L) * hv);
  const auto delta_gpu = read_f32(delta_s_out, s_in.size());
  std::vector<float> delta_cpu(s_in.size(), 0.0f);
  const U32 repeat = hv / hk;
  for (U32 h = 0; h < hv; ++h) {
    const U32 k_h = h / repeat;
    const float g_end = f16_to_f32(cum_u16[std::size_t(L - 1U) * hv + h]);
    for (U32 p = 0; p < P; ++p) {
      const float g_t = f16_to_f32(cum_u16[std::size_t(hkvd_t[p]) * hv + h]);
      const float w_dec = std::exp(std::min(0.0f, g_end - g_t));
      const float bn = beta_new[p * hv + h];
      const float bo = beta_old[p * hv + h];
      for (U32 dv = 0; dv < vd; ++dv) {
        const float vn = bf16_to_float(v_new[(p * hv + h) * vd + dv]);
        const float vo = bf16_to_float(v_old[(p * hv + h) * vd + dv]);
        for (U32 dk = 0; dk < kd; ++dk) {
          const float kn = bf16_to_float(k_new[(p * hk + k_h) * kd + dk]);
          const float ko = bf16_to_float(k_old[(p * hk + k_h) * kd + dk]);
          delta_cpu[(std::size_t(h) * vd + dv) * kd + dk] +=
              w_dec * (bn * vn * kn - bo * vo * ko);
        }
      }
    }
  }
  const auto m_perturb = compare_vectors(delta_gpu, delta_cpu, "hkvd_delta_s");
  require(m_perturb.max_abs < 1.0e-5,
          "HKVD rank-1 perturbation DeltaS mismatch vs CPU formula");

  return {{"hv", hv},
          {"tokens", L},
          {"bf16_oop_cos_sim", m_bf16_oop.cos_sim},
          {"bf16_oop_max_abs", m_bf16_oop.max_abs},
          {"bf16_inplace_cos_sim", m_bf16_ip.cos_sim},
          {"bf16_inplace_max_abs", m_bf16_ip.max_abs},
          {"fp16_rowscale_cos_sim", m_f16_oop.cos_sim},
          {"fp16_rowscale_max_abs", m_f16_oop.max_abs},
          {"fp8_e4m3_cos_sim", m_fp8_oop.cos_sim},
          {"fp8_e4m3_max_abs", m_fp8_oop.max_abs},
          {"hkvd_perturb_max_abs", m_perturb.max_abs}};
}

// ============================================================================
// Test 2: Misaligned `p_new = 137` KV Instantiation (`137 % 64 = 9`)
//   Proves bit-exact V (100%) and K[128..256) (50%) raw copy with ZERO
//   re-quantization, and cos_sim >= 0.9995 on rotated K[0..64) vs fresh
//   `omlx_m5_attention_qk_norm_rope_kv_append_affine4_g128` at pos=137..137+L.
// ============================================================================
Json test_misaligned_kv_instantiation(const MetalContext &ctx,
                                      const std::filesystem::path &libs) {
  constexpr U32 qh = 24U;
  constexpr U32 kvh = 4U;
  constexpr U32 hd = 256U;
  constexpr U32 rotary_dim = 64U;
  constexpr U32 page_tokens = 64U;
  constexpr U32 p_new = 137U; // 137 % 64 == 9 (misaligned across 4 pages!)
  constexpr U32 L = 192U;
  constexpr U32 cap = 512U;
  constexpr float eps = 1.0e-6f;
  constexpr float theta = 10000000.0f;
  constexpr float wt_off = 1.0f;

  const U32 kv_width = kvh * hd;
  const U32 q_width = qh * hd;

  std::vector<std::uint16_t> q_raw(std::size_t(L) * q_width * 2U);
  std::vector<std::uint16_t> k_raw(std::size_t(L) * kv_width);
  std::vector<std::uint16_t> v_raw(std::size_t(L) * kv_width);
  std::vector<std::uint16_t> q_norm(hd);
  std::vector<std::uint16_t> k_norm(hd);
  for (std::size_t i = 0; i < q_raw.size(); ++i) {
    q_raw[i] = float_to_bf16(float(int((i * 19ULL + 5ULL) % 211ULL) - 105) / 64.0f);
  }
  for (std::size_t i = 0; i < k_raw.size(); ++i) {
    k_raw[i] = float_to_bf16(float(int((i * 37ULL + 11ULL) % 233ULL) - 116) / 48.0f);
    v_raw[i] = float_to_bf16(float(int((i * 53ULL + 7ULL) % 241ULL) - 120) / 48.0f);
  }
  for (U32 d = 0; d < hd; ++d) {
    q_norm[d] = float_to_bf16(0.25f + 0.5f * float(d % 19U) / 18.0f);
    k_norm[d] = float_to_bf16(0.15f + 0.6f * float(d % 23U) / 22.0f);
  }

  std::vector<U32> positions(L);
  for (U32 t = 0; t < L; ++t) positions[t] = p_new + t;

  auto q_ref_buf = ctx.upload(std::as_bytes(std::span(q_raw)));
  auto k_ref_buf = ctx.upload(std::as_bytes(std::span(k_raw)));
  auto k_src_buf = ctx.upload(std::as_bytes(std::span(k_raw)));
  auto v_src_buf = ctx.upload(std::as_bytes(std::span(v_raw)));
  auto qn_buf = ctx.upload(std::as_bytes(std::span(q_norm)));
  auto kn_buf = ctx.upload(std::as_bytes(std::span(k_norm)));
  auto pos_buf = ctx.upload(std::as_bytes(std::span(positions)));

  const U64 slab_bytes = kv_slab_bytes(KvStorageFormat::Affine4Group128,
                                       cap / page_tokens + 1U, page_tokens, kvh,
                                       hd);
  // Reference fresh KvPagePool + slabs
  KvPagePool pool_ref(ctx, cap, page_tokens, cap / page_tokens + 1U,
                      KvStorageFormat::Affine4Group128);
  (void)pool_ref.begin_append(p_new + L);
  pool_ref.commit(p_new + L);
  auto ref_k_slab = ctx.make_buffer(slab_bytes);
  auto ref_v_slab = ctx.make_buffer(slab_bytes);
  ref_k_slab.zero();
  ref_v_slab.zero();

  // Blend KvPagePool + slabs
  KvPagePool pool_blend(ctx, cap, page_tokens, cap / page_tokens + 1U,
                        KvStorageFormat::Affine4Group128);
  (void)pool_blend.begin_append(p_new + L);
  pool_blend.commit(p_new + L);
  auto blend_k_slab = ctx.make_buffer(slab_bytes);
  auto blend_v_slab = ctx.make_buffer(slab_bytes);
  blend_k_slab.zero();
  blend_v_slab.zero();

  // Cached chunk K_pre / V buffers (144 B/vec) + single-quantize 64-dim rotary BF16 sidecar
  auto cached_k_pre = ctx.make_buffer(std::size_t(L) * kvh * sizeof(BlendAffine4Vec256));
  auto cached_k_rot = ctx.make_buffer(std::size_t(L) * kvh * rotary_dim * sizeof(std::uint16_t));
  auto cached_v = ctx.make_buffer(std::size_t(L) * kvh * sizeof(BlendAffine4Vec256));

  // 1. Fresh reference path (`attention_affine4.metal`)
  {
    const auto post_p = words({qh, kvh, hd, rotary_dim, L, q_width * 2U, hd * 2U,
                               kv_width, hd, p_new, word(eps), word(theta),
                               word(wt_off)});
    Encoder e(ctx, libs, L, "test.blend.kv.fresh_ref");
    auto &d = e.emit(
        "attention_affine4",
        "omlx_m5_attention_qk_norm_rope_kv_append_affine4_g128",
        {.width = 32U, .height = U64(L) * (qh + kvh)}, {.width = 32U},
        {{&q_ref_buf, 0, q_ref_buf.size()},
         {&k_ref_buf, 0, k_ref_buf.size()},
         {&v_src_buf, 0, v_src_buf.size()},
         {&qn_buf, 0, qn_buf.size()},
         {&kn_buf, 0, kn_buf.size()},
         {&ref_k_slab, 0, ref_k_slab.size()},
         {&ref_v_slab, 0, ref_v_slab.size()},
         {&pool_ref.table_buffer(), 0, pool_ref.table_buffer().size()},
         {&pos_buf, 0, pos_buf.size()}});
    e.dynamic(d, 9, post_p);
    exec_profile(ctx, e.finish(false));
  }

  // 2. CacheBlend extract pre-RoPE -> instantiate at p_new = 137 -> refresh page summary
  {
    BlendExtractPreRopeKvParams ep{.key_value_heads = kvh,
                                   .head_dim = hd,
                                   .rotary_dim = rotary_dim,
                                   .tokens = L,
                                   .key_token_stride = kv_width,
                                   .value_token_stride = kv_width,
                                   .apply_k_norm = 1U,
                                   .write_rotary_bf16 = 1U,
                                   .epsilon = eps,
                                   .weight_offset = wt_off};
    BlendInstantiateKvParams ip{.key_value_heads = kvh,
                                .head_dim = hd,
                                .rotary_dim = rotary_dim,
                                .chunk_tokens = L,
                                .dst_start_pos = p_new,
                                .page_tokens = page_tokens,
                                .theta = theta,
                                .src_token_offset = 0U,
                                .use_rotary_bf16 = 1U};
    const U32 first_page = p_new / page_tokens;
    const U32 last_page = (p_new + L - 1U) / page_tokens;
    BlendRefreshPageSummaryParams sp{
        .key_value_heads = kvh,
        .head_dim = hd,
        .page_tokens = page_tokens,
        .first_logical_page = first_page,
        .num_logical_pages = last_page - first_page + 1U,
        .total_valid_tokens = p_new + L};

    Encoder e(ctx, libs, L, "test.blend.kv.instantiate");
    auto &ext = e.emit("cache_blend",
                       "omlx_m5_blend_extract_kv_prerope_affine4_g128",
                       {.width = 32U, .height = U64(L) * kvh}, {.width = 32U},
                       {{&k_src_buf, 0, k_src_buf.size()},
                        {&v_src_buf, 0, v_src_buf.size()},
                        {&kn_buf, 0, kn_buf.size()},
                        {&cached_k_pre, 0, cached_k_pre.size()},
                        {&cached_k_rot, 0, cached_k_rot.size()},
                        {&cached_v, 0, cached_v.size()}});
    e.dynamic(ext, 6, to_bytes(ep));

    auto &inst = e.emit(
        "cache_blend", "omlx_m5_blend_instantiate_kv_affine4_g128",
        {.width = 32U, .height = U64(L) * kvh}, {.width = 32U},
        {{&cached_k_pre, 0, cached_k_pre.size()},
         {&cached_k_rot, 0, cached_k_rot.size()},
         {&cached_v, 0, cached_v.size()},
         {&blend_k_slab, 0, blend_k_slab.size()},
         {&blend_v_slab, 0, blend_v_slab.size()},
         {&pool_blend.table_buffer(), 0, pool_blend.table_buffer().size()}});
    e.dynamic(inst, 6, to_bytes(ip));

    auto &ref_sum = e.emit(
        "cache_blend", "omlx_m5_blend_refresh_kv_page_summary_affine4_g128",
        {.width = 32U, .height = U64(sp.num_logical_pages) * kvh},
        {.width = 32U},
        {{&blend_k_slab, 0, blend_k_slab.size()},
         {&blend_v_slab, 0, blend_v_slab.size()},
         {&pool_blend.table_buffer(), 0, pool_blend.table_buffer().size()}});
    e.dynamic(ref_sum, 3, to_bytes(sp));

    exec_profile(ctx, e.finish(false));
  }

  // 3. Bit-exact byte comparisons on V (100%) and K Group 1 [128..256) (50%),
  //    plus cosine similarity on rotated K[0..64) and page summary enclosure!
  const auto ref_k_bytes = ref_k_slab.read(0, slab_bytes);
  const auto ref_v_bytes = ref_v_slab.read(0, slab_bytes);
  const auto bld_k_bytes = blend_k_slab.read(0, slab_bytes);
  const auto bld_v_bytes = blend_v_slab.read(0, slab_bytes);

  const std::size_t page_b = std::size_t(
      kv_page_bytes(KvStorageFormat::Affine4Group128, page_tokens, kvh, hd));
  const std::size_t head_code_b = std::size_t(page_tokens) * 128U;
  const std::size_t all_codes_b = std::size_t(kvh) * head_code_b;
  const std::size_t all_meta_b = std::size_t(kvh) * page_tokens * 16U;

  U64 v_byte_diffs = 0;
  U64 k_g1_byte_diffs = 0;
  std::vector<float> k_rot_got, k_rot_ref;
  k_rot_got.reserve(std::size_t(L) * kvh * rotary_dim);
  k_rot_ref.reserve(std::size_t(L) * kvh * rotary_dim);

  for (U32 t = 0; t < L; ++t) {
    const U32 pos = p_new + t;
    const U32 lp = pos / page_tokens;
    const U32 within = pos % page_tokens;
    const U32 rp = pool_ref.physical_slot(lp);
    const U32 bp = pool_blend.physical_slot(lp);

    const auto *rk_page = reinterpret_cast<const std::uint8_t *>(ref_k_bytes.data()) + rp * page_b;
    const auto *rv_page = reinterpret_cast<const std::uint8_t *>(ref_v_bytes.data()) + rp * page_b;
    const auto *bk_page = reinterpret_cast<const std::uint8_t *>(bld_k_bytes.data()) + bp * page_b;
    const auto *bv_page = reinterpret_cast<const std::uint8_t *>(bld_v_bytes.data()) + bp * page_b;

    for (U32 h = 0; h < kvh; ++h) {
      // V: g0 codes (64B) + g1 codes (64B) + meta (16B) must be BIT-EXACT!
      const auto *rv_g0 = rv_page + h * head_code_b + within * 64U;
      const auto *bv_g0 = bv_page + h * head_code_b + within * 64U;
      const auto *rv_g1 = rv_page + h * head_code_b + (page_tokens + within) * 64U;
      const auto *bv_g1 = bv_page + h * head_code_b + (page_tokens + within) * 64U;
      for (U32 i = 0; i < 64U; ++i) {
        if (rv_g0[i] != bv_g0[i]) ++v_byte_diffs;
        if (rv_g1[i] != bv_g1[i]) ++v_byte_diffs;
      }
      const auto &rv_m =
          reinterpret_cast<const KvAffine4VecMeta *>(rv_page + all_codes_b)[h * page_tokens + within];
      const auto &bv_m =
          reinterpret_cast<const KvAffine4VecMeta *>(bv_page + all_codes_b)[h * page_tokens + within];
      if (std::memcmp(&rv_m, &bv_m, sizeof(KvAffine4VecMeta)) != 0) {
        v_byte_diffs += sizeof(KvAffine4VecMeta);
      }

      // K Group 1 [128..256): g1 codes (64B) + scale[1] + zero[1] must be BIT-EXACT!
      const auto *rk_g1 = rk_page + h * head_code_b + (page_tokens + within) * 64U;
      const auto *bk_g1 = bk_page + h * head_code_b + (page_tokens + within) * 64U;
      for (U32 i = 0; i < 64U; ++i) {
        if (rk_g1[i] != bk_g1[i]) ++k_g1_byte_diffs;
      }
      const auto &rk_m =
          reinterpret_cast<const KvAffine4VecMeta *>(rk_page + all_codes_b)[h * page_tokens + within];
      const auto &bk_m =
          reinterpret_cast<const KvAffine4VecMeta *>(bk_page + all_codes_b)[h * page_tokens + within];
      if (rk_m.scale_f16[1] != bk_m.scale_f16[1] ||
          rk_m.zero_f16[1] != bk_m.zero_f16[1]) {
        k_g1_byte_diffs += 4U;
      }

      // Rotated K[0..64) comparison
      const auto dk_ref = dequant_kv_slot_256(ref_k_bytes, rp, h, within, page_tokens, kvh);
      const auto dk_bld = dequant_kv_slot_256(bld_k_bytes, bp, h, within, page_tokens, kvh);
      for (U32 d = 0; d < rotary_dim; ++d) {
        k_rot_ref.push_back(dk_ref[d]);
        k_rot_got.push_back(dk_bld[d]);
      }

      // Verify KvAffine4PageSummary bounds enclose dequantized K values.
      const auto &ksum = reinterpret_cast<const KvAffine4PageSummary *>(
          bk_page + all_codes_b + all_meta_b)[h];
      for (U32 sb = 0; sb < 8U; ++sb) {
        const float lo = f16_to_f32(ksum.subband_min_f16[sb]);
        const float hi = f16_to_f32(ksum.subband_max_f16[sb]);
        for (U32 d = sb * 32U; d < (sb + 1U) * 32U; ++d) {
          require(dk_bld[d] >= lo - 1e-2f && dk_bld[d] <= hi + 1e-2f,
                  "KvAffine4PageSummary failed to enclose instantiated K value");
        }
      }
    }
  }

  require(v_byte_diffs == 0ULL, "V instantiation was NOT bit-exact!");
  require(k_g1_byte_diffs == 0ULL,
          "K[128..256) Group-1 instantiation was NOT bit-exact!");
  const auto m_rot = compare_vectors(k_rot_got, k_rot_ref, "rotated_k_0_64");
  require(m_rot.cos_sim >= 0.9995,
          "Rotated K[0..64) cosine similarity fell below 0.9995 (got cos_sim=" +
              std::to_string(m_rot.cos_sim) +
              ", max_abs=" + std::to_string(m_rot.max_abs) +
              ", rel_l2=" + std::to_string(m_rot.rel_l2) + ")");

  return {{"p_new", p_new},
          {"p_new_mod_64", p_new % 64U},
          {"tokens", L},
          {"v_byte_diffs", v_byte_diffs},
          {"k_group1_byte_diffs", k_g1_byte_diffs},
          {"k_rot64_cos_sim", m_rot.cos_sim},
          {"k_rot64_max_abs", m_rot.max_abs}};
}

// ============================================================================
// Test 3: End-to-End Synthetic Hybrid Block (GDN + Full-Attn)
//   Reordered `[Doc B][Doc A][Query]` stitch vs Cold Dense Run:
//   - Non-64-aligned doc lengths (`L_B = 307`, `L_A = 283`, `L_Q = 128`, total `718`)
//   - GDN stitched via W_bnd=64 boundary + cached `tail_qkv` + `(M_C, B_C)` compose
//   - Full-Attn stitched via `instantiate_kv` + Layer-3 HKVD ~15% top-k selection
//     + `gather_rows` + `qk_norm_rope_kv_scatter` + production
//     `omlx_m5_paged_flash_prefill_affine4_g128_hd256` on `[M_active, H]`
//     with `active_pos` + `scatter_rows`!
// ============================================================================
Json test_hybrid_e2e_reordered_stitch(const MetalContext &ctx,
                                      const std::filesystem::path &libs) {
  // Hybrid block geometry (scaled GDN heads for fast M4 test, exact Full-Attn 27B HD=256!)
  constexpr U32 gdn_hk = 4U;
  constexpr U32 gdn_hv = 12U;
  constexpr U32 gdn_d = 128U;
  constexpr U32 gdn_gv = gdn_hv * gdn_d;
  constexpr U32 conv_w = 2U * gdn_hk * gdn_d + gdn_gv; // 2560
  constexpr U32 conv_k = 4U;

  constexpr U32 qh = 24U;
  constexpr U32 kvh = 4U;
  constexpr U32 hd = 256U;
  constexpr U32 rotary_dim = 64U;
  constexpr U32 q_width = qh * hd; // 6144
  constexpr U32 kv_width = kvh * hd; // 1024
  constexpr U32 page_tokens = 64U;
  constexpr U32 cap = 1024U;
  constexpr U32 W_bnd = 64U;
  constexpr float eps = 1.0e-6f;
  constexpr float theta = 10000000.0f;
  constexpr float wt_off = 1.0f;

  // Delimiter-anchored misaligned document lengths:
  // Doc B: 307 tok, Doc A: 283 tok (starts at 307, 307 % 64 = 51!), Query: 128 tok.
  constexpr U32 L_B = 307U;
  constexpr U32 L_A = 283U;
  constexpr U32 L_Q = 128U;
  constexpr U32 L_TOT = L_B + L_A + L_Q; // 718

  std::vector<U32> tok_B(L_B), tok_A(L_A), tok_Q(L_Q);
  tok_B[0] = 151644U;
  for (U32 i = 1; i < L_B; ++i) tok_B[i] = 10000U + i;
  tok_A[0] = 151644U;
  for (U32 i = 1; i < L_A; ++i) tok_A[i] = 20000U + i;
  for (U32 i = 0; i < L_Q; ++i) tok_Q[i] = 30000U + i;

  std::vector<U32> prompt;
  prompt.insert(prompt.end(), tok_B.begin(), tok_B.end());
  prompt.insert(prompt.end(), tok_A.begin(), tok_A.end());
  prompt.insert(prompt.end(), tok_Q.begin(), tok_Q.end());

  // Deterministic per-token synthetic projections derived purely from token ID
  // (modeling Layer 0..2 token-local projections + Layer 3 Q/K/V/Gate).
  auto fill_gdn_for_tokens = [&](std::span<const U32> toks,
                                 std::vector<std::uint16_t> &qkv,
                                 std::vector<std::uint16_t> &a,
                                 std::vector<std::uint16_t> &b) {
    qkv.resize(toks.size() * conv_w);
    a.resize(toks.size() * gdn_hv);
    b.resize(toks.size() * gdn_hv);
    for (std::size_t t = 0; t < toks.size(); ++t) {
      const U64 seed = U64(toks[t]) * 2654435761ULL;
      for (U32 c = 0; c < conv_w; ++c) {
        const U64 h = (seed ^ (U64(c) * 11400714819323198485ULL)) >> 48U;
        qkv[t * conv_w + c] = float_to_bf16((float(int(h % 251ULL) - 125)) / 128.0f);
      }
      for (U32 h = 0; h < gdn_hv; ++h) {
        const U64 x = (seed + U64(h) * 1315423911ULL) >> 48U;
        a[t * gdn_hv + h] = float_to_bf16((float(int(x % 97ULL) - 52)) / 16.0f);
        b[t * gdn_hv + h] = float_to_bf16((float(int((x >> 4U) % 89ULL) - 44)) / 16.0f);
      }
    }
  };

  auto fill_attn_for_tokens = [&](std::span<const U32> toks,
                                  std::vector<std::uint16_t> &qg,
                                  std::vector<std::uint16_t> &k,
                                  std::vector<std::uint16_t> &v) {
    qg.resize(toks.size() * q_width * 2U);
    k.resize(toks.size() * kv_width);
    v.resize(toks.size() * kv_width);
    for (std::size_t t = 0; t < toks.size(); ++t) {
      const U64 seed = U64(toks[t]) * 2246822519ULL;
      for (U32 c = 0; c < q_width * 2U; ++c) {
        const U64 x = (seed ^ (U64(c) * 3266489917ULL)) >> 48U;
        qg[t * q_width * 2U + c] =
            float_to_bf16((float(int(x % 227ULL) - 113)) / 64.0f);
      }
      for (U32 c = 0; c < kv_width; ++c) {
        const U64 x = (seed ^ (U64(c) * 668265263ULL)) >> 48U;
        k[t * kv_width + c] =
            float_to_bf16((float(int(x % 239ULL) - 119)) / 48.0f);
        v[t * kv_width + c] =
            float_to_bf16((float(int((x >> 3U) % 233ULL) - 116)) / 48.0f);
      }
    }
  };

  std::vector<std::uint16_t> conv_wt(conv_w * conv_k);
  std::vector<std::uint16_t> a_log(gdn_hv), dt_bias(gdn_hv);
  std::vector<std::uint16_t> q_norm(hd), k_norm(hd);
  for (std::size_t i = 0; i < conv_wt.size(); ++i) {
    conv_wt[i] = float_to_bf16(float(int((i * 17ULL + 3ULL) % 89ULL) - 44) / 64.0f);
  }
  for (U32 h = 0; h < gdn_hv; ++h) {
    a_log[h] = float_to_bf16(-2.2f + 0.08f * float(h));
    dt_bias[h] = float_to_bf16(-0.6f + 0.06f * float(h));
  }
  for (U32 d = 0; d < hd; ++d) {
    q_norm[d] = float_to_bf16(0.20f + 0.5f * float(d % 17U) / 16.0f);
    k_norm[d] = float_to_bf16(0.15f + 0.5f * float(d % 19U) / 18.0f);
  }

  auto conv_wt_buf = ctx.upload(std::as_bytes(std::span(conv_wt)));
  auto a_log_buf = ctx.upload(std::as_bytes(std::span(a_log)));
  auto dt_bias_buf = ctx.upload(std::as_bytes(std::span(dt_bias)));
  auto qn_buf = ctx.upload(std::as_bytes(std::span(q_norm)));
  auto kn_buf = ctx.upload(std::as_bytes(std::span(k_norm)));

  const auto ws_lo = gdn_chunk_nax_workspace_layout(L_TOT, gdn_hk, gdn_hv);
  auto gdn_ws_buf = ctx.make_buffer(ws_lo.total_bytes);
  const Ref qk_ws_ref{&gdn_ws_buf, 0, ws_lo.qk_bytes};
  const Ref ctrl_ws_ref{&gdn_ws_buf, ws_lo.qk_bytes, ws_lo.ctrl_bytes};
  const U64 state_bytes = U64(gdn_hv) * gdn_d * gdn_d * sizeof(float);
  const U64 hist_bytes = U64(conv_k - 1U) * conv_w * sizeof(std::uint16_t);
  const U64 slab_bytes = kv_slab_bytes(KvStorageFormat::Affine4Group128,
                                       cap / page_tokens + 1U, page_tokens, kvh,
                                       hd);
  auto dummy_u32 = ctx.make_buffer(64);
  auto dummy_bf = ctx.make_buffer(256);
  auto dummy_f32 = ctx.make_buffer(64);

  const bool is_apple10 =
      ctx.capabilities().device_name.starts_with("Apple M5") ||
      ctx.capabilities().profile.probe.apple_gpu_family >= 10;

  auto emit_affine4_attn = [&](Encoder &e, U32 toks, const MetalBuffer &qg_b,
                               const MetalBuffer &k_sl, const MetalBuffer &v_sl,
                               const MetalBuffer &out_b, const MetalBuffer &pt_b,
                               const MetalBuffer &pos_b) {
    const auto attn_p =
        words({qh, kvh, hd, toks, 0U, cap, page_tokens, cap / page_tokens + 1U,
               q_width * 2U, hd * 2U, kv_width, kv_width, q_width * 2U,
               hd * 2U});
    if (is_apple10) {
      const U32 q_blocks = (toks + 63U) / 64U;
      auto &d = e.emit(
          "attention_affine4",
          "omlx_m5_paged_flash_prefill_affine4_g128_hd256",
          {.width = 256U, .height = U64(q_blocks) * qh}, {.width = 256U},
          {{&qg_b, 0, qg_b.size()},
           {&qg_b, U64(hd) * 2ULL, qg_b.size() - U64(hd) * 2ULL},
           {&k_sl, 0, k_sl.size()},
           {&v_sl, 0, v_sl.size()},
           {&out_b, 0, out_b.size()},
           {&pt_b, 0, pt_b.size()},
           {&pos_b, 0, pos_b.size()}});
      e.dynamic(d, 7, attn_p);
    } else {
      const U32 repeat = qh / kvh;
      const U32 bq = 16U / repeat;
      const U32 t_blocks = (toks + bq - 1U) / bq;
      auto &d = e.emit(
          "attention_affine4",
          "omlx_m5_paged_causal_attention_affine4_g128_hd256",
          {.width = 64U, .height = U64(t_blocks) * kvh}, {.width = 64U},
          {{&qg_b, 0, qg_b.size()},
           {&qg_b, U64(hd) * 2ULL, qg_b.size() - U64(hd) * 2ULL},
           {&k_sl, 0, k_sl.size()},
           {&v_sl, 0, v_sl.size()},
           {&out_b, 0, out_b.size()},
           {&pt_b, 0, pt_b.size()},
           {&pos_b, 0, pos_b.size()}});
      e.dynamic(d, 7, attn_p);
    }
  };

  // --------------------------------------------------------------------------
  // Phase A: Pre-warm ChunkBlendCache with Doc A and Doc B in isolation (pos=0)
  // --------------------------------------------------------------------------
  const U64 exec_hash = compute_blend_model_exec_hash(
      "sha256:hybrid_e2e", "affine4_g128", "bf16", "exact_nax", false, false);
  ChunkBlendCache blend_cache(256ULL * 1024ULL * 1024ULL);

  auto populate_isolated_chunk = [&](std::span<const U32> toks,
                                     const std::string &tag) {
    const U32 L = U32(toks.size());
    std::vector<std::uint16_t> qkv, a, b, qg, k, v;
    fill_gdn_for_tokens(toks, qkv, a, b);
    fill_attn_for_tokens(toks, qg, k, v);

    auto qkv_buf = ctx.upload(std::as_bytes(std::span(qkv)));
    auto a_buf = ctx.upload(std::as_bytes(std::span(a)));
    auto b_buf = ctx.upload(std::as_bytes(std::span(b)));
    auto qg_buf = ctx.upload(std::as_bytes(std::span(qg)));
    auto k_buf = ctx.upload(std::as_bytes(std::span(k)));
    auto v_buf = ctx.upload(std::as_bytes(std::span(v)));

    BlendChunkGeometry geom;
    geom.num_full_attn_layers = 1U;
    geom.num_gdn_layers = 1U;
    geom.q_heads = qh;
    geom.kv_heads = kvh;
    geom.head_dim = hd;
    geom.rotary_dim = rotary_dim;
    geom.hidden_size = 512U;
    geom.gdn_key_heads = gdn_hk;
    geom.gdn_value_heads = gdn_hv;
    geom.gdn_key_dim = gdn_d;
    geom.gdn_value_dim = gdn_d;
    geom.gdn_conv_width = conv_w;
    geom.gdn_conv_kernel = conv_k;
    geom.token_length = L;
    geom.interior_start = W_bnd;
    geom.gdn_op_format = BlendGdnOperatorFormat::BFloat16;

    auto entry = std::make_shared<BlendChunkEntry>();
    entry->key = make_blend_chunk_key(exec_hash, toks);
    entry->tokens.assign(toks.begin(), toks.end());
    entry->geometry = geom;
    entry->payload = allocate_blend_chunk_payload(ctx, geom);

    // 1. Copy tail_qkv[3, conv_w] for exact Conv1D(K=4) handoff across chunks.
    const std::size_t tail_start = std::size_t(L - (conv_k - 1U)) * conv_w;
    entry->payload.gdn_tail_qkv.write(std::as_bytes(
        std::span(qkv).subspan(tail_start, std::size_t(conv_k - 1U) * conv_w)));

    // 2. Run isolated Conv1D (hist=0) on all L tokens, then extract interior
    //    [W_bnd, L) GDN operators (M_C, B_C) + cumulative decay G_t!
    auto conv_out = ctx.make_buffer(std::size_t(L) * conv_w * 2ULL);
    auto hist_zero = ctx.make_buffer(hist_bytes);
    hist_zero.zero();
    auto b_f32 = ctx.make_buffer(state_bytes);
    b_f32.zero();
    auto m_f32 = ctx.make_buffer(state_bytes);
    auto rec_tmp = ctx.make_buffer(std::size_t(L) * gdn_gv * 2ULL);

    const U32 L_int = L - W_bnd;
    const auto conv_p =
        words({gdn_hk, gdn_hv, gdn_d, gdn_d, conv_w, conv_k, L, word(eps)});
    const auto gdn_int_p =
        words({gdn_hk, gdn_hv, gdn_d, gdn_d, conv_w, conv_k, L_int, word(eps)});
    const U64 conv_int_off = U64(W_bnd) * conv_w * 2ULL;
    const U64 ab_int_off = U64(W_bnd) * gdn_hv * 2ULL;
    const auto ws_int = gdn_chunk_nax_workspace_layout(L_int, gdn_hk, gdn_hv);
    const Ref qk_int_ref{&gdn_ws_buf, 0, ws_int.qk_bytes};
    const Ref ctrl_int_ref{&gdn_ws_buf, ws_int.qk_bytes, ws_int.ctrl_bytes};

    // 3. Extract Full-Attn K_pre (post k_norm, pre-RoPE) + V, plus isolated
    //    Layer-3 Mix3_cached -> int8_g128!
    KvPagePool iso_pool(ctx, cap, page_tokens, cap / page_tokens + 1U,
                        KvStorageFormat::Affine4Group128);
    (void)iso_pool.begin_append(L);
    iso_pool.commit(L);
    auto iso_k_slab = ctx.make_buffer(slab_bytes);
    auto iso_v_slab = ctx.make_buffer(slab_bytes);
    iso_k_slab.zero();
    iso_v_slab.zero();
    std::vector<U32> iso_pos(L);
    std::iota(iso_pos.begin(), iso_pos.end(), 0U);
    auto iso_pos_buf = ctx.upload(std::as_bytes(std::span(iso_pos)));
    auto iso_mix_bf = ctx.make_buffer(std::size_t(L) * q_width * 2ULL);
    auto k_iso_mut = ctx.upload(std::as_bytes(std::span(k)));

    {
      Encoder e(ctx, libs, L, "test.blend.populate." + tag);
      // GDN conv on [0, L)
      emit_gdn_conv_silu(e, L, conv_w, {&qkv_buf, 0, qkv_buf.size()},
                         {&conv_wt_buf, 0, conv_wt_buf.size()},
                         {&hist_zero, 0, hist_zero.size()},
                         {&conv_out, 0, conv_out.size()}, conv_p, false);
      // GDN prepare on interior [W_bnd, L)
      auto &prep = e.emit(
          "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_c64",
          {.width = U64(gdn_hk) * 128U, .height = ws_int.max_chunks},
          {.width = 128U},
          {{&conv_out, conv_int_off, conv_out.size() - conv_int_off},
           {&a_buf, ab_int_off, a_buf.size() - ab_int_off},
           {&b_buf, ab_int_off, b_buf.size() - ab_int_off},
           {&a_log_buf, 0, a_log_buf.size()},
           {&dt_bias_buf, 0, dt_bias_buf.size()},
           qk_int_ref,
           ctrl_int_ref});
      e.dynamic(prep, 7, gdn_int_p);
      // B_C (S_init = 0)
      auto &rec_b = e.emit(
          "gdn_chunk_nax", "omlx_m5_gdn_chunk_recurrence_c64",
          {.width = U64(gdn_hv) * 128U, .height = 4U}, {.width = 128U},
          {{&conv_out, conv_int_off, conv_out.size() - conv_int_off},
           qk_int_ref,
           ctrl_int_ref,
           {&b_f32, 0, b_f32.size()},
           {&rec_tmp, 0, rec_tmp.size()}});
      e.dynamic(rec_b, 5, gdn_int_p);
      // M_C (S_init = I_128, V = 0)
      BlendGdnExtractMParams mp{.key_heads = gdn_hk,
                                .value_heads = gdn_hv,
                                .key_dim = gdn_d,
                                .value_dim = gdn_d,
                                .tokens = L_int};
      auto &ext_m = e.emit("cache_blend", "omlx_m5_gdn_extract_transition_m_c64",
                           {.width = 128U, .height = gdn_hv, .depth = 2U},
                           {.width = 128U},
                           {qk_int_ref, ctrl_int_ref, {&m_f32, 0, m_f32.size()}});
      e.dynamic(ext_m, 3, to_bytes(mp));
      // Pack M_C, B_C + cumulative decay
      BlendGdnPackOpParams pp{.value_heads = gdn_hv,
                              .dim = 128U,
                              .op_format = U32(geom.gdn_op_format)};
      auto &pm = e.emit("cache_blend", "omlx_m5_gdn_pack_operator",
                        {.width = 32U, .height = U64(gdn_hv) * 128U},
                        {.width = 32U},
                        {{&m_f32, 0, m_f32.size()},
                         {&entry->payload.gdn_m_data, 0,
                          entry->payload.gdn_m_data.size()},
                         {&entry->payload.gdn_m_row_scales, 0,
                          entry->payload.gdn_m_row_scales.size()}});
      e.dynamic(pm, 3, to_bytes(pp));
      auto &pb = e.emit("cache_blend", "omlx_m5_gdn_pack_operator",
                        {.width = 32U, .height = U64(gdn_hv) * 128U},
                        {.width = 32U},
                        {{&b_f32, 0, b_f32.size()},
                         {&entry->payload.gdn_b_data, 0,
                          entry->payload.gdn_b_data.size()},
                         {&entry->payload.gdn_b_row_scales, 0,
                          entry->payload.gdn_b_row_scales.size()}});
      e.dynamic(pb, 3, to_bytes(pp));
      BlendGdnCumDecayParams cdp{.value_heads = gdn_hv, .tokens = L};
      auto &cd = e.emit("cache_blend", "omlx_m5_gdn_compute_cum_decay_f16",
                        {.width = gdn_hv}, {.width = 32U},
                        {{&a_buf, 0, a_buf.size()},
                         {&a_log_buf, 0, a_log_buf.size()},
                         {&dt_bias_buf, 0, dt_bias_buf.size()},
                         {&entry->payload.gdn_cum_decay_f16, 0,
                          entry->payload.gdn_cum_decay_f16.size()}});
      e.dynamic(cd, 4, to_bytes(cdp));

      // Extract Full-Attn K_pre & V (+ rotary BF16 sidecar)
      BlendExtractPreRopeKvParams ekp{.key_value_heads = kvh,
                                      .head_dim = hd,
                                      .rotary_dim = rotary_dim,
                                      .tokens = L,
                                      .key_token_stride = kv_width,
                                      .value_token_stride = kv_width,
                                      .apply_k_norm = 1U,
                                      .write_rotary_bf16 = 1U,
                                      .epsilon = eps,
                                      .weight_offset = wt_off};
      auto &ek = e.emit(
          "cache_blend", "omlx_m5_blend_extract_kv_prerope_affine4_g128",
          {.width = 32U, .height = U64(L) * kvh}, {.width = 32U},
          {{&k_buf, 0, k_buf.size()},
           {&v_buf, 0, v_buf.size()},
           {&kn_buf, 0, kn_buf.size()},
           {&entry->payload.attn_k_pre, 0, entry->payload.attn_k_pre.size()},
           {&entry->payload.attn_k_rotary_bf16, 0,
            entry->payload.attn_k_rotary_bf16.size()},
           {&entry->payload.attn_v, 0, entry->payload.attn_v.size()}});
      e.dynamic(ek, 6, to_bytes(ekp));

      // Isolated Layer-3 attention -> Mix3_cached (int8_g128)
      const auto post_p = words({qh, kvh, hd, rotary_dim, L, q_width * 2U,
                                 hd * 2U, kv_width, hd, 0U, word(eps),
                                 word(theta), word(wt_off)});
      auto &ipost = e.emit(
          "attention_affine4",
          "omlx_m5_attention_qk_norm_rope_kv_append_affine4_g128",
          {.width = 32U, .height = U64(L) * (qh + kvh)}, {.width = 32U},
          {{&qg_buf, 0, qg_buf.size()},
           {&k_iso_mut, 0, k_iso_mut.size()},
           {&v_buf, 0, v_buf.size()},
           {&qn_buf, 0, qn_buf.size()},
           {&kn_buf, 0, kn_buf.size()},
           {&iso_k_slab, 0, iso_k_slab.size()},
           {&iso_v_slab, 0, iso_v_slab.size()},
           {&iso_pool.table_buffer(), 0, iso_pool.table_buffer().size()},
           {&iso_pos_buf, 0, iso_pos_buf.size()}});
      e.dynamic(ipost, 9, post_p);

      emit_affine4_attn(e, L, qg_buf, iso_k_slab, iso_v_slab, iso_mix_bf,
                        iso_pool.table_buffer(), iso_pos_buf);

      BlendInt8QuantParams qp{.tokens = L, .width = q_width};
      auto &qmix = e.emit(
          "cache_blend", "omlx_m5_blend_quantize_int8_g128_bf16",
          {.width = 32U, .height = U64(L) * (q_width / 128U)}, {.width = 32U},
          {{&iso_mix_bf, 0, iso_mix_bf.size()},
           {&entry->payload.mix3_cached_i8, 0,
            entry->payload.mix3_cached_i8.size()},
           {&entry->payload.mix3_cached_scales, 0,
            entry->payload.mix3_cached_scales.size()}});
      e.dynamic(qmix, 3, to_bytes(qp));

      exec_profile(ctx, e.finish(false));
    }

    require(blend_cache.insert(std::move(entry)), "cache insert failed: " + tag);
  };

  // Pre-warm Doc A first, then Doc B (opposite order from the query prompt!).
  populate_isolated_chunk(tok_A, "DocA");
  populate_isolated_chunk(tok_B, "DocB");

  // --------------------------------------------------------------------------
  // Phase B: Plan reordered prompt `[Doc B][Doc A][Query]` via ChunkBlendCache
  // --------------------------------------------------------------------------
  BlendSegmentConfig seg_cfg;
  seg_cfg.l_min = 256U;
  seg_cfg.l_max = 4096U;
  seg_cfg.boundary_window = W_bnd;
  seg_cfg.dense_tail_tokens = L_Q;
  seg_cfg.min_suffix_for_blend = 512U;

  const auto plan = blend_cache.plan_prompt(exec_hash, prompt, 0U, seg_cfg);
  require(plan.segments.size() == 3U, "expected 3 planned segments");
  require(plan.segments[0].is_cache_hit() && plan.segments[0].segment.length == L_B &&
              plan.segments[0].segment.boundary_window == W_bnd,
          "Doc B should hit with W_bnd=64");
  require(plan.segments[1].is_cache_hit() && plan.segments[1].segment.length == L_A &&
              plan.segments[1].segment.start == L_B &&
              plan.segments[1].segment.boundary_window == W_bnd,
          "Doc A should hit at misaligned pos=307 with W_bnd=64");
  require(!plan.segments[2].is_cache_hit() &&
              plan.segments[2].segment.kind == BlendSegmentKind::UncachedDense &&
              plan.segments[2].segment.length == L_Q,
          "Query tail should be mandatory dense");

  // Prepare full-prompt synthetic inputs for BOTH Cold Dense and Hybrid Stitch.
  std::vector<std::uint16_t> full_qkv, full_a, full_b, full_qg, full_k, full_v;
  fill_gdn_for_tokens(prompt, full_qkv, full_a, full_b);
  fill_attn_for_tokens(prompt, full_qg, full_k, full_v);
  auto full_qkv_buf = ctx.upload(std::as_bytes(std::span(full_qkv)));
  auto full_a_buf = ctx.upload(std::as_bytes(std::span(full_a)));
  auto full_b_buf = ctx.upload(std::as_bytes(std::span(full_b)));
  auto full_qg_cold = ctx.upload(std::as_bytes(std::span(full_qg)));
  auto full_k_cold = ctx.upload(std::as_bytes(std::span(full_k)));
  auto full_v_buf = ctx.upload(std::as_bytes(std::span(full_v)));

  std::vector<U32> full_pos(L_TOT);
  std::iota(full_pos.begin(), full_pos.end(), 0U);
  auto full_pos_buf = ctx.upload(std::as_bytes(std::span(full_pos)));

  // --------------------------------------------------------------------------
  // Phase C: Cold Dense Reference Run over all 718 tokens `[Doc B][Doc A][Query]`
  // --------------------------------------------------------------------------
  auto cold_hist = ctx.make_buffer(hist_bytes);
  cold_hist.zero();
  auto cold_conv = ctx.make_buffer(std::size_t(L_TOT) * conv_w * 2ULL);
  auto cold_state = ctx.make_buffer(state_bytes);
  cold_state.zero();
  auto cold_gdn_out = ctx.make_buffer(std::size_t(L_TOT) * gdn_gv * 2ULL);

  KvPagePool cold_pool(ctx, cap, page_tokens, cap / page_tokens + 1U,
                       KvStorageFormat::Affine4Group128);
  (void)cold_pool.begin_append(L_TOT);
  cold_pool.commit(L_TOT);
  auto cold_k_slab = ctx.make_buffer(slab_bytes);
  auto cold_v_slab = ctx.make_buffer(slab_bytes);
  cold_k_slab.zero();
  cold_v_slab.zero();
  auto cold_attn_out = ctx.make_buffer(std::size_t(L_TOT) * q_width * 2ULL);

  {
    const auto gdn_p =
        words({gdn_hk, gdn_hv, gdn_d, gdn_d, conv_w, conv_k, L_TOT, word(eps)});
    const auto post_p = words({qh, kvh, hd, rotary_dim, L_TOT, q_width * 2U,
                               hd * 2U, kv_width, hd, 0U, word(eps),
                               word(theta), word(wt_off)});

    Encoder e(ctx, libs, L_TOT, "test.blend.e2e.cold_dense");
    emit_gdn_conv_silu(e, L_TOT, conv_w, {&full_qkv_buf, 0, full_qkv_buf.size()},
                       {&conv_wt_buf, 0, conv_wt_buf.size()},
                       {&cold_hist, 0, cold_hist.size()},
                       {&cold_conv, 0, cold_conv.size()}, gdn_p, false);
    auto &prep = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_c64",
        {.width = U64(gdn_hk) * 128U, .height = ws_lo.max_chunks},
        {.width = 128U},
        {{&cold_conv, 0, cold_conv.size()},
         {&full_a_buf, 0, full_a_buf.size()},
         {&full_b_buf, 0, full_b_buf.size()},
         {&a_log_buf, 0, a_log_buf.size()},
         {&dt_bias_buf, 0, dt_bias_buf.size()},
         qk_ws_ref,
         ctrl_ws_ref});
    e.dynamic(prep, 7, gdn_p);
    auto &rec = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_recurrence_c64",
        {.width = U64(gdn_hv) * 128U, .height = 4U}, {.width = 128U},
        {{&cold_conv, 0, cold_conv.size()},
         qk_ws_ref,
         ctrl_ws_ref,
         {&cold_state, 0, cold_state.size()},
         {&cold_gdn_out, 0, cold_gdn_out.size()}});
    e.dynamic(rec, 5, gdn_p);

    auto &post = e.emit(
        "attention_affine4",
        "omlx_m5_attention_qk_norm_rope_kv_append_affine4_g128",
        {.width = 32U, .height = U64(L_TOT) * (qh + kvh)}, {.width = 32U},
        {{&full_qg_cold, 0, full_qg_cold.size()},
         {&full_k_cold, 0, full_k_cold.size()},
         {&full_v_buf, 0, full_v_buf.size()},
         {&qn_buf, 0, qn_buf.size()},
         {&kn_buf, 0, kn_buf.size()},
         {&cold_k_slab, 0, cold_k_slab.size()},
         {&cold_v_slab, 0, cold_v_slab.size()},
         {&cold_pool.table_buffer(), 0, cold_pool.table_buffer().size()},
         {&full_pos_buf, 0, full_pos_buf.size()}});
    e.dynamic(post, 9, post_p);

    emit_affine4_attn(e, L_TOT, full_qg_cold, cold_k_slab, cold_v_slab,
                      cold_attn_out, cold_pool.table_buffer(), full_pos_buf);
    exec_profile(ctx, e.finish(false));
  }

  // --------------------------------------------------------------------------
  // Phase D: Hybrid GDN-CacheBlend Stitched Run
  // --------------------------------------------------------------------------
  const auto &hit_B = *plan.segments[0].hit_entry;
  const auto &hit_A = *plan.segments[1].hit_entry;

  auto bld_state = ctx.make_buffer(state_bytes);
  bld_state.zero();
  auto bld_hist = ctx.make_buffer(hist_bytes);
  bld_hist.zero();
  auto bld_conv_tmp = ctx.make_buffer(std::size_t(L_Q) * conv_w * 2ULL);
  auto bld_gdn_q_out = ctx.make_buffer(std::size_t(L_Q) * gdn_gv * 2ULL);

  // D.1 GDN Stitched Execution:
  //   - Doc B [0..64) dense -> jump [64..307) via (M_B, B_B) -> restore tail_qkv_B
  //   - Doc A [307..371) dense (with Doc B's tail_qkv!) -> jump [371..590) via (M_A, B_A) -> restore tail_qkv_A
  //   - Query [590..718) dense exact!
  auto run_gdn_dense_slice = [&](U32 start_tok, U32 count,
                                 const MetalBuffer &hist_in,
                                 MetalBuffer &out_buf, const std::string &tag) {
    auto hist_mut = ctx.upload(hist_in.read(0, hist_bytes));
    const auto p =
        words({gdn_hk, gdn_hv, gdn_d, gdn_d, conv_w, conv_k, count, word(eps)});
    const auto ws_s = gdn_chunk_nax_workspace_layout(count, gdn_hk, gdn_hv);
    const Ref qk_r{&gdn_ws_buf, 0, ws_s.qk_bytes};
    const Ref ctrl_r{&gdn_ws_buf, ws_s.qk_bytes, ws_s.ctrl_bytes};
    const U64 qkv_off = U64(start_tok) * conv_w * 2ULL;
    const U64 ab_off = U64(start_tok) * gdn_hv * 2ULL;

    Encoder e(ctx, libs, count, "test.blend.e2e.gdn_slice." + tag);
    emit_gdn_conv_silu(
        e, count, conv_w, {&full_qkv_buf, qkv_off, U64(count) * conv_w * 2ULL},
        {&conv_wt_buf, 0, conv_wt_buf.size()}, {&hist_mut, 0, hist_mut.size()},
        {&bld_conv_tmp, 0, U64(count) * conv_w * 2ULL}, p, false);
    auto &prep = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_prepare_c64",
        {.width = U64(gdn_hk) * 128U, .height = ws_s.max_chunks},
        {.width = 128U},
        {{&bld_conv_tmp, 0, U64(count) * conv_w * 2ULL},
         {&full_a_buf, ab_off, U64(count) * gdn_hv * 2ULL},
         {&full_b_buf, ab_off, U64(count) * gdn_hv * 2ULL},
         {&a_log_buf, 0, a_log_buf.size()},
         {&dt_bias_buf, 0, dt_bias_buf.size()},
         qk_r,
         ctrl_r});
    e.dynamic(prep, 7, p);
    auto &rec = e.emit(
        "gdn_chunk_nax", "omlx_m5_gdn_chunk_recurrence_c64",
        {.width = U64(gdn_hv) * 128U, .height = 4U}, {.width = 128U},
        {{&bld_conv_tmp, 0, U64(count) * conv_w * 2ULL},
         qk_r,
         ctrl_r,
         {&bld_state, 0, bld_state.size()},
         {&out_buf, 0, out_buf.size()}});
    e.dynamic(rec, 5, p);
    exec_profile(ctx, e.finish(false));
  };

  auto run_gdn_jump = [&](const BlendChunkEntry &hit, const std::string &tag) {
    BlendGdnStateComposeParams cp{.key_heads = gdn_hk,
                                  .value_heads = gdn_hv,
                                  .dim = 128U,
                                  .op_format = U32(hit.geometry.gdn_op_format),
                                  .chunk_tokens = hit.geometry.token_length,
                                  .num_hkvd_perturb = 0U,
                                  .apply_bias = 1U};
    Encoder e(ctx, libs, 1, "test.blend.e2e.gdn_jump." + tag);
    auto &comp = e.emit(
        "cache_blend", "omlx_m5_gdn_state_compose_nax",
        {.width = 128U, .height = gdn_hv, .depth = 4U}, {.width = 128U},
        {{&bld_state, 0, bld_state.size()},
         {&hit.payload.gdn_m_data, 0, hit.payload.gdn_m_data.size()},
         {&hit.payload.gdn_m_row_scales, 0, hit.payload.gdn_m_row_scales.size()},
         {&hit.payload.gdn_b_data, 0, hit.payload.gdn_b_data.size()},
         {&hit.payload.gdn_b_row_scales, 0, hit.payload.gdn_b_row_scales.size()},
         {&hit.payload.gdn_cum_decay_f16, 0, hit.payload.gdn_cum_decay_f16.size()},
         {&dummy_u32, 0, dummy_u32.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_f32, 0, dummy_f32.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_bf, 0, dummy_bf.size()},
         {&dummy_f32, 0, dummy_f32.size()},
         {&bld_state, 0, bld_state.size()}});
    e.dynamic(comp, 14, to_bytes(cp));
    exec_profile(ctx, e.finish(false));
  };

  run_gdn_dense_slice(0U, W_bnd, bld_hist, bld_gdn_q_out, "docB_bnd");
  run_gdn_jump(hit_B, "docB_int");
  run_gdn_dense_slice(L_B, W_bnd, hit_B.payload.gdn_tail_qkv, bld_gdn_q_out,
                      "docA_bnd");
  run_gdn_jump(hit_A, "docA_int");
  run_gdn_dense_slice(L_B + L_A, L_Q, hit_A.payload.gdn_tail_qkv, bld_gdn_q_out,
                      "query");

  // D.2 Full-Attn Stitched Execution:
  //   1. Instantiate Doc B @ pos=0 and Doc A @ pos=307 (307 % 64 == 51!), append Query @ pos=590.
  //   2. Compute stitched Layer-3 Mix -> run `hkvd_l2_score` + `hkvd_topk_select` (~15% HKVD).
  //   3. Gather active `[M_active]` rows -> run `qk_norm_rope_kv_scatter_affine4_g128` ->
  //      run production `paged_flash_prefill_affine4_g128_hd256` on `[M_active]` with `active_pos`
  //      -> scatter outputs back!
  KvPagePool bld_pool(ctx, cap, page_tokens, cap / page_tokens + 1U,
                      KvStorageFormat::Affine4Group128);
  (void)bld_pool.begin_append(L_TOT);
  bld_pool.commit(L_TOT);
  auto bld_k_slab = ctx.make_buffer(slab_bytes);
  auto bld_v_slab = ctx.make_buffer(slab_bytes);
  bld_k_slab.zero();
  bld_v_slab.zero();

  auto qg_probe_buf = ctx.upload(std::as_bytes(std::span(full_qg)));
  auto k_query_mut = ctx.upload(std::as_bytes(
      std::span(full_k).subspan(std::size_t(L_B + L_A) * kv_width)));
  auto v_query_buf = ctx.upload(std::as_bytes(
      std::span(full_v).subspan(std::size_t(L_B + L_A) * kv_width)));
  auto q_query_mut = ctx.upload(std::as_bytes(
      std::span(full_qg).subspan(std::size_t(L_B + L_A) * q_width * 2U)));
  std::vector<U32> query_pos(L_Q);
  std::iota(query_pos.begin(), query_pos.end(), L_B + L_A);
  auto query_pos_buf = ctx.upload(std::as_bytes(std::span(query_pos)));
  auto probe_mix_buf = ctx.make_buffer(std::size_t(L_TOT) * q_width * 2ULL);

  // Assemble concatenated Mix3_cached (int8_g128) across [Doc B, Doc A, zeros(Query)].
  auto mix_cached_i8 = ctx.make_buffer(std::size_t(L_TOT) * q_width);
  auto mix_cached_sc =
      ctx.make_buffer(std::size_t(L_TOT) * (q_width / 128U) * 2ULL);
  mix_cached_i8.zero();
  mix_cached_sc.zero();
  mix_cached_i8.write(hit_B.payload.mix3_cached_i8.read(), 0ULL);
  mix_cached_sc.write(hit_B.payload.mix3_cached_scales.read(), 0ULL);
  mix_cached_i8.write(hit_A.payload.mix3_cached_i8.read(),
                      U64(L_B) * q_width);
  mix_cached_sc.write(hit_A.payload.mix3_cached_scales.read(),
                      U64(L_B) * (q_width / 128U) * 2ULL);

  auto flags_buf =
      ctx.upload(std::as_bytes(std::span(plan.suffix_token_flags)));
  auto prompt_pos_buf =
      ctx.upload(std::as_bytes(std::span(plan.suffix_prompt_positions)));
  auto l2_scores_buf = ctx.make_buffer(std::size_t(L_TOT) * sizeof(float));
  auto active_idx_buf = ctx.make_buffer(std::size_t(L_TOT) * sizeof(U32));
  auto active_pos_buf = ctx.make_buffer(std::size_t(L_TOT) * sizeof(U32));
  auto active_cnt_buf = ctx.make_buffer(sizeof(U32));

  {
    Encoder e(ctx, libs, L_TOT, "test.blend.e2e.attn_probe_and_hkvd");
    // Instantiate Doc B @ pos=0
    BlendInstantiateKvParams ip_B{.key_value_heads = kvh,
                                  .head_dim = hd,
                                  .rotary_dim = rotary_dim,
                                  .chunk_tokens = L_B,
                                  .dst_start_pos = 0U,
                                  .page_tokens = page_tokens,
                                  .theta = theta,
                                  .src_token_offset = 0U,
                                  .use_rotary_bf16 = 1U};
    auto &inst_B = e.emit(
        "cache_blend", "omlx_m5_blend_instantiate_kv_affine4_g128",
        {.width = 32U, .height = U64(L_B) * kvh}, {.width = 32U},
        {{&hit_B.payload.attn_k_pre, 0, hit_B.payload.attn_k_pre.size()},
         {&hit_B.payload.attn_k_rotary_bf16, 0,
          hit_B.payload.attn_k_rotary_bf16.size()},
         {&hit_B.payload.attn_v, 0, hit_B.payload.attn_v.size()},
         {&bld_k_slab, 0, bld_k_slab.size()},
         {&bld_v_slab, 0, bld_v_slab.size()},
         {&bld_pool.table_buffer(), 0, bld_pool.table_buffer().size()}});
    e.dynamic(inst_B, 6, to_bytes(ip_B));

    // Instantiate Doc A @ pos=307 (307 % 64 == 51!)
    BlendInstantiateKvParams ip_A{.key_value_heads = kvh,
                                  .head_dim = hd,
                                  .rotary_dim = rotary_dim,
                                  .chunk_tokens = L_A,
                                  .dst_start_pos = L_B,
                                  .page_tokens = page_tokens,
                                  .theta = theta,
                                  .src_token_offset = 0U,
                                  .use_rotary_bf16 = 1U};
    auto &inst_A = e.emit(
        "cache_blend", "omlx_m5_blend_instantiate_kv_affine4_g128",
        {.width = 32U, .height = U64(L_A) * kvh}, {.width = 32U},
        {{&hit_A.payload.attn_k_pre, 0, hit_A.payload.attn_k_pre.size()},
         {&hit_A.payload.attn_k_rotary_bf16, 0,
          hit_A.payload.attn_k_rotary_bf16.size()},
         {&hit_A.payload.attn_v, 0, hit_A.payload.attn_v.size()},
         {&bld_k_slab, 0, bld_k_slab.size()},
         {&bld_v_slab, 0, bld_v_slab.size()},
         {&bld_pool.table_buffer(), 0, bld_pool.table_buffer().size()}});
    e.dynamic(inst_A, 6, to_bytes(ip_A));

    // Dense Query KV append @ pos=590..717
    const auto q_post_p = words({qh, kvh, hd, rotary_dim, L_Q, q_width * 2U,
                                 hd * 2U, kv_width, hd, L_B + L_A, word(eps),
                                 word(theta), word(wt_off)});
    auto &qpost = e.emit(
        "attention_affine4",
        "omlx_m5_attention_qk_norm_rope_kv_append_affine4_g128",
        {.width = 32U, .height = U64(L_Q) * (qh + kvh)}, {.width = 32U},
        {{&q_query_mut, 0, q_query_mut.size()},
         {&k_query_mut, 0, k_query_mut.size()},
         {&v_query_buf, 0, v_query_buf.size()},
         {&qn_buf, 0, qn_buf.size()},
         {&kn_buf, 0, kn_buf.size()},
         {&bld_k_slab, 0, bld_k_slab.size()},
         {&bld_v_slab, 0, bld_v_slab.size()},
         {&bld_pool.table_buffer(), 0, bld_pool.table_buffer().size()},
         {&query_pos_buf, 0, query_pos_buf.size()}});
    e.dynamic(qpost, 9, q_post_p);

    // Apply Q-norm + RoPE to full probe queries, then run Layer-3 probe attention.
    const auto all_post_p = words({qh, kvh, hd, rotary_dim, L_TOT, q_width * 2U,
                                   hd * 2U, kv_width, hd, 0U, word(eps),
                                   word(theta), word(wt_off)});
    auto &q_rope = e.emit(
        "attention_postprocess", "omlx_m5_attention_qk_norm_rope",
        {.width = 32U, .height = U64(L_TOT) * (qh + kvh)}, {.width = 32U},
        {{&qg_probe_buf, 0, qg_probe_buf.size()},
         {&full_k_cold, 0, full_k_cold.size()}, // unused after cold run
         {&qn_buf, 0, qn_buf.size()},
         {&kn_buf, 0, kn_buf.size()}});
    e.dynamic(q_rope, 4, all_post_p);

    emit_affine4_attn(e, L_TOT, qg_probe_buf, bld_k_slab, bld_v_slab,
                      probe_mix_buf, bld_pool.table_buffer(), full_pos_buf);

    // HKVD L2 scoring + exact sorted top-15% selection!
    BlendHkvdSelectParams hp{.total_tokens = L_TOT,
                             .mix_width = q_width,
                             .cached_is_int8 = 1U,
                             .compute_l2_inline = 0U,
                             .explicit_hkvd_k = 0U,
                             .hkvd_ratio = 0.15f};
    auto &l2 = e.emit("cache_blend", "omlx_m5_blend_hkvd_l2_score",
                      {.width = 128U, .height = L_TOT}, {.width = 128U},
                      {{&probe_mix_buf, 0, probe_mix_buf.size()},
                       {&mix_cached_i8, 0, mix_cached_i8.size()},
                       {&mix_cached_sc, 0, mix_cached_sc.size()},
                       {&flags_buf, 0, flags_buf.size()},
                       {&l2_scores_buf, 0, l2_scores_buf.size()}});
    e.dynamic(l2, 5, to_bytes(hp));

    auto &topk = e.emit("cache_blend", "omlx_m5_blend_hkvd_topk_select",
                        {.width = 256U}, {.width = 256U},
                        {{&probe_mix_buf, 0, probe_mix_buf.size()},
                         {&mix_cached_i8, 0, mix_cached_i8.size()},
                         {&mix_cached_sc, 0, mix_cached_sc.size()},
                         {&flags_buf, 0, flags_buf.size()},
                         {&prompt_pos_buf, 0, prompt_pos_buf.size()},
                         {&l2_scores_buf, 0, l2_scores_buf.size()},
                         {&active_idx_buf, 0, active_idx_buf.size()},
                         {&active_pos_buf, 0, active_pos_buf.size()},
                         {&active_cnt_buf, 0, active_cnt_buf.size()}});
    e.dynamic(topk, 9, to_bytes(hp));
    exec_profile(ctx, e.finish(false));
  }

  // Verify HKVD selection invariants (count, strictly ascending order, exact CPU top-k match).
  const U32 m_active = read_u32(active_cnt_buf, 1U)[0];
  const U32 n_cand = (L_B - W_bnd) + (L_A - W_bnd);
  const U32 expected_k = U32(std::ceil(0.15f * float(n_cand)));
  const U32 expected_m = W_bnd + W_bnd + L_Q + expected_k;
  require(m_active == expected_m,
          "HKVD active_count mismatch: got " + std::to_string(m_active) +
              " expected " + std::to_string(expected_m));

  const auto active_idx = read_u32(active_idx_buf, m_active);
  const auto active_pos = read_u32(active_pos_buf, m_active);
  const auto l2_scores = read_f32(l2_scores_buf, L_TOT);

  for (U32 i = 0; i < m_active; ++i) {
    require(active_idx[i] < L_TOT && active_pos[i] == active_idx[i],
            "active_idx/active_pos out of range or misaligned");
    if (i > 0U) {
      require(active_idx[i] > active_idx[i - 1U] &&
                  active_pos[i] > active_pos[i - 1U],
              "active_idx/active_pos not strictly ascending");
    }
  }
  // Exact CPU reference top-k check on interior candidates:
  std::vector<U32> cpu_cand;
  for (U32 t = 0; t < L_TOT; ++t) {
    if (plan.suffix_token_flags[t] == kBlendTokenInteriorCandidate) {
      cpu_cand.push_back(t);
    }
  }
  std::stable_sort(cpu_cand.begin(), cpu_cand.end(), [&](U32 x, U32 y) {
    if (l2_scores[x] != l2_scores[y]) return l2_scores[x] > l2_scores[y];
    return x < y;
  });
  std::set<U32> expected_active;
  for (U32 t = 0; t < L_TOT; ++t) {
    if (plan.suffix_token_flags[t] == kBlendTokenMandatory) {
      expected_active.insert(t);
    }
  }
  for (U32 i = 0; i < expected_k; ++i) {
    expected_active.insert(cpu_cand[i]);
  }
  std::set<U32> gpu_active(active_idx.begin(), active_idx.end());
  require(gpu_active == expected_active,
          "GPU radix HKVD top-k set differs from exact CPU reference!");

  // D.3 Packed Gather -> Scattered QK-Norm+RoPE+KV Scatter -> Paged Flash Prefill
  //     on contiguous `[M_active]` rows -> Scatter Output Back!
  auto full_qg_raw = ctx.upload(std::as_bytes(std::span(full_qg)));
  auto full_k_raw = ctx.upload(std::as_bytes(std::span(full_k)));
  auto packed_qg = ctx.make_buffer(std::size_t(m_active) * q_width * 2U * 2ULL);
  auto packed_k = ctx.make_buffer(std::size_t(m_active) * kv_width * 2ULL);
  auto packed_v = ctx.make_buffer(std::size_t(m_active) * kv_width * 2ULL);
  auto packed_attn_out = ctx.make_buffer(std::size_t(m_active) * q_width * 2ULL);
  auto scattered_attn_out = ctx.make_buffer(std::size_t(L_TOT) * q_width * 2ULL);
  scattered_attn_out.zero();

  {
    Encoder e(ctx, libs, m_active, "test.blend.e2e.gather_scatter_attn");
    BlendGatherScatterParams gp_qg{.max_active_tokens = m_active,
                                   .width = q_width * 2U,
                                   .use_device_count = 1U,
                                   .residual_add = 0U};
    auto &g_qg = e.emit(
        "cache_blend", "omlx_m5_blend_gather_rows_bf16",
        {.width = (q_width * 2U) / 4U, .height = m_active}, {.width = 128U},
        {{&full_qg_raw, 0, full_qg_raw.size()},
         {&active_idx_buf, 0, active_idx_buf.size()},
         {&active_cnt_buf, 0, active_cnt_buf.size()},
         {&packed_qg, 0, packed_qg.size()}});
    e.dynamic(g_qg, 4, to_bytes(gp_qg));

    BlendGatherScatterParams gp_kv{.max_active_tokens = m_active,
                                   .width = kv_width,
                                   .use_device_count = 1U,
                                   .residual_add = 0U};
    auto &g_k = e.emit(
        "cache_blend", "omlx_m5_blend_gather_rows_bf16",
        {.width = kv_width / 4U, .height = m_active}, {.width = 128U},
        {{&full_k_raw, 0, full_k_raw.size()},
         {&active_idx_buf, 0, active_idx_buf.size()},
         {&active_cnt_buf, 0, active_cnt_buf.size()},
         {&packed_k, 0, packed_k.size()}});
    e.dynamic(g_k, 4, to_bytes(gp_kv));

    auto &g_v = e.emit(
        "cache_blend", "omlx_m5_blend_gather_rows_bf16",
        {.width = kv_width / 4U, .height = m_active}, {.width = 128U},
        {{&full_v_buf, 0, full_v_buf.size()},
         {&active_idx_buf, 0, active_idx_buf.size()},
         {&active_cnt_buf, 0, active_cnt_buf.size()},
         {&packed_v, 0, packed_v.size()}});
    e.dynamic(g_v, 4, to_bytes(gp_kv));

    // Scattered QK-Norm + RoPE(active_pos[i]) + Affine4Group128 KV Scatter!
    BlendQkNormRopeScatterParams sp{.query_heads = qh,
                                    .key_value_heads = kvh,
                                    .head_dim = hd,
                                    .rotary_dim = rotary_dim,
                                    .max_active_tokens = m_active,
                                    .query_token_stride = q_width * 2U,
                                    .query_head_stride = hd * 2U,
                                    .key_token_stride = kv_width,
                                    .key_head_stride = hd,
                                    .value_token_stride = kv_width,
                                    .page_tokens = page_tokens,
                                    .use_device_count = 1U,
                                    .epsilon = eps,
                                    .theta = theta,
                                    .weight_offset = wt_off};
    auto &skv = e.emit(
        "cache_blend",
        "omlx_m5_attention_qk_norm_rope_kv_scatter_affine4_g128",
        {.width = 32U, .height = U64(m_active) * (qh + kvh)}, {.width = 32U},
        {{&packed_qg, 0, packed_qg.size()},
         {&packed_k, 0, packed_k.size()},
         {&packed_v, 0, packed_v.size()},
         {&qn_buf, 0, qn_buf.size()},
         {&kn_buf, 0, kn_buf.size()},
         {&bld_k_slab, 0, bld_k_slab.size()},
         {&bld_v_slab, 0, bld_v_slab.size()},
         {&bld_pool.table_buffer(), 0, bld_pool.table_buffer().size()},
         {&active_pos_buf, 0, active_pos_buf.size()},
         {&active_cnt_buf, 0, active_cnt_buf.size()}});
    e.dynamic(skv, 10, to_bytes(sp));

    // Refresh page summaries for all logical pages covering [0..L_TOT).
    const U32 total_pages = (L_TOT + page_tokens - 1U) / page_tokens;
    BlendRefreshPageSummaryParams rsp{.key_value_heads = kvh,
                                      .head_dim = hd,
                                      .page_tokens = page_tokens,
                                      .first_logical_page = 0U,
                                      .num_logical_pages = total_pages,
                                      .total_valid_tokens = L_TOT};
    auto &rsum = e.emit(
        "cache_blend", "omlx_m5_blend_refresh_kv_page_summary_affine4_g128",
        {.width = 32U, .height = U64(total_pages) * kvh}, {.width = 32U},
        {{&bld_k_slab, 0, bld_k_slab.size()},
         {&bld_v_slab, 0, bld_v_slab.size()},
         {&bld_pool.table_buffer(), 0, bld_pool.table_buffer().size()}});
    e.dynamic(rsum, 3, to_bytes(rsp));

    // Unchanged production Route-B paged affine4 attention on contiguous [M_active]
    // using `active_pos` as causal horizon!
    emit_affine4_attn(e, m_active, packed_qg, bld_k_slab, bld_v_slab,
                      packed_attn_out, bld_pool.table_buffer(), active_pos_buf);

    // Scatter packed attention outputs back to full `[L_TOT, q_width]`.
    BlendGatherScatterParams sp_out{.max_active_tokens = m_active,
                                    .width = q_width,
                                    .use_device_count = 1U,
                                    .residual_add = 0U};
    auto &scat = e.emit(
        "cache_blend", "omlx_m5_blend_scatter_rows_bf16",
        {.width = q_width / 4U, .height = m_active}, {.width = 128U},
        {{&packed_attn_out, 0, packed_attn_out.size()},
         {&active_idx_buf, 0, active_idx_buf.size()},
         {&active_cnt_buf, 0, active_cnt_buf.size()},
         {&scattered_attn_out, 0, scattered_attn_out.size()}});
    e.dynamic(scat, 4, to_bytes(sp_out));

    exec_profile(ctx, e.finish(false));
  }

  // --------------------------------------------------------------------------
  // Phase E: Compare Stitched Hybrid vs Cold Dense Ground Truth
  // --------------------------------------------------------------------------
  const auto s_cold_v = read_f32(cold_state, std::size_t(gdn_hv) * gdn_d * gdn_d);
  const auto s_bld_v = read_f32(bld_state, std::size_t(gdn_hv) * gdn_d * gdn_d);
  const auto m_gdn_state = compare_vectors(s_bld_v, s_cold_v, "e2e_gdn_state");

  const auto cold_gdn_all = bf16_buf_to_f32(cold_gdn_out, std::size_t(L_TOT) * gdn_gv);
  const auto bld_gdn_q = bf16_buf_to_f32(bld_gdn_q_out, std::size_t(L_Q) * gdn_gv);
  const std::span<const float> cold_gdn_q(
      cold_gdn_all.data() + std::size_t(L_B + L_A) * gdn_gv,
      std::size_t(L_Q) * gdn_gv);
  const auto m_gdn_query = compare_vectors(bld_gdn_q, cold_gdn_q, "e2e_gdn_query");

  const auto cold_attn_all = bf16_buf_to_f32(cold_attn_out, std::size_t(L_TOT) * q_width);
  const auto bld_attn_all = bf16_buf_to_f32(scattered_attn_out, std::size_t(L_TOT) * q_width);
  const std::span<const float> cold_attn_q(
      cold_attn_all.data() + std::size_t(L_B + L_A) * q_width,
      std::size_t(L_Q) * q_width);
  const std::span<const float> bld_attn_q(
      bld_attn_all.data() + std::size_t(L_B + L_A) * q_width,
      std::size_t(L_Q) * q_width);
  const auto m_attn_query = compare_vectors(bld_attn_q, cold_attn_q, "e2e_attn_query");

  require(m_gdn_state.cos_sim >= 0.9998 && m_gdn_state.max_abs < 1.0e-3,
          "E2E hybrid stitched terminal GDN state diverged from cold dense");
  require(m_gdn_query.cos_sim >= 0.9998,
          "E2E hybrid stitched query GDN output diverged from cold dense");
  require(m_attn_query.cos_sim >= 0.999,
          "E2E hybrid stitched query Full-Attn output diverged from cold dense (cos_sim=" +
              std::to_string(m_attn_query.cos_sim) +
              ", max_abs=" + std::to_string(m_attn_query.max_abs) + ")");

  // Also verify single-dispatch inline L2 path (`compute_l2_inline = 1`) matches 2-stage!
  {
    auto inline_scores = ctx.make_buffer(std::size_t(L_TOT) * sizeof(float));
    auto inline_idx = ctx.make_buffer(std::size_t(L_TOT) * sizeof(U32));
    auto inline_pos = ctx.make_buffer(std::size_t(L_TOT) * sizeof(U32));
    auto inline_cnt = ctx.make_buffer(sizeof(U32));
    BlendHkvdSelectParams hp_in{.total_tokens = L_TOT,
                                .mix_width = q_width,
                                .cached_is_int8 = 1U,
                                .compute_l2_inline = 1U,
                                .explicit_hkvd_k = 0U,
                                .hkvd_ratio = 0.15f};
    Encoder e(ctx, libs, 1, "test.blend.e2e.hkvd_inline");
    auto &tk = e.emit("cache_blend", "omlx_m5_blend_hkvd_topk_select",
                      {.width = 256U}, {.width = 256U},
                      {{&probe_mix_buf, 0, probe_mix_buf.size()},
                       {&mix_cached_i8, 0, mix_cached_i8.size()},
                       {&mix_cached_sc, 0, mix_cached_sc.size()},
                       {&flags_buf, 0, flags_buf.size()},
                       {&prompt_pos_buf, 0, prompt_pos_buf.size()},
                       {&inline_scores, 0, inline_scores.size()},
                       {&inline_idx, 0, inline_idx.size()},
                       {&inline_pos, 0, inline_pos.size()},
                       {&inline_cnt, 0, inline_cnt.size()}});
    e.dynamic(tk, 9, to_bytes(hp_in));
    exec_profile(ctx, e.finish(false));
    require(read_u32(inline_cnt, 1U)[0] == m_active,
            "inline HKVD active_count mismatch");
    require(read_u32(inline_idx, m_active) == active_idx,
            "inline HKVD active_idx differs from 2-stage path");
  }

  // Also check static TG SRAM caps for ALL 14 kernels in omlx_m5_cache_blend.metallib.
  ctx.register_library(
      {.key = "test.cache_blend", .path = libs / "omlx_m5_cache_blend.metallib"});
  Json sram_report = Json::object();
  const std::array<const char *, 14> kernels = {
      "omlx_m5_blend_extract_kv_prerope_affine4_g128",
      "omlx_m5_blend_instantiate_kv_affine4_g128",
      "omlx_m5_blend_refresh_kv_page_summary_affine4_g128",
      "omlx_m5_gdn_extract_transition_m_c64",
      "omlx_m5_gdn_pack_operator",
      "omlx_m5_gdn_compute_cum_decay_f16",
      "omlx_m5_gdn_state_compose_nax",
      "omlx_m5_blend_quantize_int8_g128_bf16",
      "omlx_m5_blend_dequantize_int8_g128_bf16",
      "omlx_m5_blend_hkvd_l2_score",
      "omlx_m5_blend_hkvd_topk_select",
      "omlx_m5_blend_gather_rows_bf16",
      "omlx_m5_blend_scatter_rows_bf16",
      "omlx_m5_attention_qk_norm_rope_kv_scatter_affine4_g128",
  };
  const char *sv_env = std::getenv("MTL_SHADER_VALIDATION");
  const bool sv_on = sv_env != nullptr && std::string_view(sv_env) != "0" &&
                     !std::string_view(sv_env).empty();
  const U64 sram_cap = sv_on ? 32768ULL : 16384ULL;
  for (const char *fn : kernels) {
    const auto info = ctx.warmup_pipeline(
        {.key = std::string("test.blend.sram.") + fn,
         .library_key = "test.cache_blend",
         .function_name = fn,
         .variant = "sram-audit"});
    require(info.static_threadgroup_memory_bytes <= sram_cap,
            std::string("Static TG SRAM exceeded cap for ") + fn);
    sram_report[fn] = info.static_threadgroup_memory_bytes;
  }

  return {{"prompt_tokens", L_TOT},
          {"doc_b_tokens", L_B},
          {"doc_a_tokens", L_A},
          {"doc_a_pos_mod_64", L_B % 64U},
          {"query_tokens", L_Q},
          {"m_active", m_active},
          {"hkvd_selected", expected_k},
          {"gdn_state_cos_sim", m_gdn_state.cos_sim},
          {"gdn_state_max_abs", m_gdn_state.max_abs},
          {"gdn_query_cos_sim", m_gdn_query.cos_sim},
          {"attn_query_cos_sim", m_attn_query.cos_sim},
          {"attn_query_max_abs", m_attn_query.max_abs},
          {"kernel_sram_bytes", sram_report}};
}

} // namespace

int main(int argc, char **argv) {
  try {
    if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <metallib_dir>\n";
      return 2;
    }
    const std::filesystem::path libs(argv[1]);
    const MetalContext ctx = MetalContext::create();

    Json report;
    report["device"] = ctx.capabilities().device_name;
    report["test0_host_segmentation_lru"] = test_host_segmentation_and_lru();
    report["test1_gdn_state_compose_hv48"] =
        test_gdn_state_compose_parity(ctx, libs);
    report["test2_misaligned_kv_instantiation_p137"] =
        test_misaligned_kv_instantiation(ctx, libs);
    report["test3_hybrid_e2e_reordered_stitch"] =
        test_hybrid_e2e_reordered_stitch(ctx, libs);
    report["status"] = "PASS";
    std::cout << report.dump(2) << "\n";
    return 0;
  } catch (const std::exception &ex) {
    std::cerr << "ERROR: " << ex.what() << "\n";
    return 1;
  }
}
