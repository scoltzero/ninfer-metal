// W2-B Blade #2 + Dim E Slice E2: 4-Bit Affine Group-128 Paged KV Cache
// (KvStorageFormat::Affine4Group128: 144 B/vec = 128 B uint4 + 16 B half scale/zero
// + 32 B/head/page reserved summary slot) with:
//   1. Lean KV append (dead O(2048) subband summary scan removed) + fused T=1
//      QK-norm + RoPE + affine4 KV-append (+ optional Q->half FOSCAR pack).
//   2. Unified 2-SG HW uint4b_format 16x64x128 NAX Split-KV (R_BLK=6/8/16,
//      ZERO tg_q, ZERO c_acc spill, 2 barriers/page + Law #4 post-loop barrier,
//      normalized half partials + 4-SG log2 half parallel merge) preserving
//      0-ULP AR(T=1) == DFlash2(T=2..8) parity via unified 512-token splits.
//   3. Register-resident Frag16x16 (WM=4, WN=2) Flash-Prefill for HD=256 using
//      exact Law #1 half4(ushort4) nibble unpack + full-rate bfloat*bfloat NAX P@V.
#include <metal_simdgroup>
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

#define AFFINE4_UNROLL _Pragma("clang loop unroll(full)")
#define AFFINE4_ALWAYS_INLINE __attribute__((always_inline))

struct AttentionPostprocessParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint rotary_dim;
  uint tokens;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint key_head_stride;
  uint start_position;
  float epsilon;
  float theta;
  float weight_offset;
};

struct PagedAttentionParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint tokens;
  uint cache_start;
  uint cache_capacity;
  uint page_tokens;
  uint page_count;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint value_token_stride;
  uint gate_token_stride;
  uint gate_head_stride;
};

struct PagedSplitAttentionParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint tokens;
  uint cache_start;
  uint cache_capacity;
  uint page_tokens;
  uint page_count;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint value_token_stride;
  uint gate_token_stride;
  uint gate_head_stride;
  uint split_count;
  uint split_tokens;
};

struct BatchedAttentionPostprocessParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint rotary_dim;
  uint total_tokens;
  uint query_token_stride;
  uint query_head_stride;
  uint key_token_stride;
  uint key_head_stride;
  uint value_token_stride;
  uint batch_size;
  float epsilon;
  float theta;
  float weight_offset;
  uint r_stride;
  uint row_offsets[8];
  uint widths[8];
};

struct BatchedPagedAttentionParams {
  uint query_heads;
  uint key_value_heads;
  uint head_dim;
  uint batch_size;
  uint page_tokens;
  uint page_count;
  uint query_token_stride;
  uint query_head_stride;
  uint gate_token_stride;
  uint gate_head_stride;
  uint split_count;
  uint split_tokens;
  uint r_stride;
  uint n_rblks;
  uint row_offsets[8];
  uint widths[8];
};

struct KvAffine4VecMeta {
  half scale[4];
  half zero[4];
};

namespace affine4_kv {

constant constexpr float kNegInf = -3.402823466e+38f;
constant constexpr float kNegInfHalf = -1.701411733e+38f;
constant constexpr float kLog2e = 1.4426950408889634f;

AFFINE4_ALWAYS_INLINE
inline device const uint *select_pt8(
    uint slot,
    device const uint *pt0, device const uint *pt1,
    device const uint *pt2, device const uint *pt3,
    device const uint *pt4, device const uint *pt5,
    device const uint *pt6, device const uint *pt7) {
  switch (slot) {
  case 0u: return pt0;
  case 1u: return pt1;
  case 2u: return pt2;
  case 3u: return pt3;
  case 4u: return pt4;
  case 5u: return pt5;
  case 6u: return pt6;
  default: return pt7;
  }
}

AFFINE4_ALWAYS_INLINE
inline ulong page_byte_stride(uint page_tokens, uint kv_heads, uint head_dim) {
  // Keep +32ul per-head reserved summary stride identical to KvPagePool layout.
  const ulong vec_bytes = ulong(head_dim / 2u) + 16ul;
  const ulong head_bytes = ulong(page_tokens) * vec_bytes + 32ul;
  return ulong(kv_heads) * head_bytes;
}

AFFINE4_ALWAYS_INLINE
inline void quantize_vec4_g128_core(float4 v, device ushort *dst16, uint lane,
                                    thread half &out_s, thread half &out_z) {
  constexpr float kHalfMax = 65504.0f;
  v = clamp(v, float4(-kHalfMax), float4(kHalfMax));
  const float lane_lo = min(min(v.x, v.y), min(v.z, v.w));
  const float lane_hi = max(max(v.x, v.y), max(v.z, v.w));
  const float gmin = clamp(simd_min(lane_lo), -kHalfMax, kHalfMax);
  const float gmax = clamp(simd_max(lane_hi), -kHalfMax, kHalfMax);
  const half z_h = half(gmin);
  const float z_f = float(z_h);
  const float span = max(0.0f, gmax - z_f);
  const half s_h = half(min(span * (1.0f / 15.0f), kHalfMax));
  const float s_f = float(s_h);
  const float inv_s = (s_f > 0.0f) ? precise::divide(1.0f, s_f) : 0.0f;

  const uint q0 = uint(clamp(rint((v.x - z_f) * inv_s), 0.0f, 15.0f));
  const uint q1 = uint(clamp(rint((v.y - z_f) * inv_s), 0.0f, 15.0f));
  const uint q2 = uint(clamp(rint((v.z - z_f) * inv_s), 0.0f, 15.0f));
  const uint q3 = uint(clamp(rint((v.w - z_f) * inv_s), 0.0f, 15.0f));
  dst16[lane] = ushort(q0 | (q1 << 4u) | (q2 << 8u) | (q3 << 12u));
  out_s = s_h;
  out_z = z_h;
}

AFFINE4_ALWAYS_INLINE
inline void quantize_vec_g128_device(device const bfloat *src_vec,
                                     device uchar *codes_head,
                                     device KvAffine4VecMeta *meta_head,
                                     uint page_tokens, uint within,
                                     uint head_dim, uint lane) {
  half s0 = half(0.0h), z0 = half(0.0h);
  half s1 = half(0.0h), z1 = half(0.0h);
  {
    device const bfloat4 *src4 =
        reinterpret_cast<device const bfloat4 *>(src_vec);
    device ushort *dst16 =
        reinterpret_cast<device ushort *>(codes_head + ulong(within) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s0, z0);
  }
  if (head_dim == 256u) {
    device const bfloat4 *src4 =
        reinterpret_cast<device const bfloat4 *>(src_vec + 128u);
    device ushort *dst16 = reinterpret_cast<device ushort *>(
        codes_head + (ulong(page_tokens) + ulong(within)) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s1, z1);
  }
  if (lane == 0u) {
    device KvAffine4VecMeta &m = meta_head[within];
    m.scale[0] = s0;
    m.scale[1] = s1;
    m.scale[2] = half(0.0h);
    m.scale[3] = half(0.0h);
    m.zero[0] = z0;
    m.zero[1] = z1;
    m.zero[2] = half(0.0h);
    m.zero[3] = half(0.0h);
  }
}

AFFINE4_ALWAYS_INLINE
inline void quantize_vec_g128_tg(threadgroup const bfloat *src_vec,
                                 device uchar *codes_head,
                                 device KvAffine4VecMeta *meta_head,
                                 uint page_tokens, uint within, uint head_dim,
                                 uint lane) {
  half s0 = half(0.0h), z0 = half(0.0h);
  half s1 = half(0.0h), z1 = half(0.0h);
  {
    threadgroup const bfloat4 *src4 =
        reinterpret_cast<threadgroup const bfloat4 *>(src_vec);
    device ushort *dst16 =
        reinterpret_cast<device ushort *>(codes_head + ulong(within) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s0, z0);
  }
  if (head_dim == 256u) {
    threadgroup const bfloat4 *src4 =
        reinterpret_cast<threadgroup const bfloat4 *>(src_vec + 128u);
    device ushort *dst16 = reinterpret_cast<device ushort *>(
        codes_head + (ulong(page_tokens) + ulong(within)) * 64ul);
    quantize_vec4_g128_core(float4(src4[lane]), dst16, lane, s1, z1);
  }
  if (lane == 0u) {
    device KvAffine4VecMeta &m = meta_head[within];
    m.scale[0] = s0;
    m.scale[1] = s1;
    m.scale[2] = half(0.0h);
    m.scale[3] = half(0.0h);
    m.zero[0] = z0;
    m.zero[1] = z1;
    m.zero[2] = half(0.0h);
    m.zero[3] = half(0.0h);
  }
}

// ============================================================================
// Unified 2-SG HW uint4b_format 16x64x128 NAX Split-KV (HD == 256)
// - ZERO tg_q (Q read directly from L2-cached pre-packed [KVH, 2, R_stride, 128])
// - ZERO c_acc spill (single register-resident cooperative_tensor with
//   mode::multiply_accumulate, 0 user arrays / 0 alloca)
// - 2 barriers per page + mandatory Law #4 post-loop barrier before reusing tg_rs
// - Identical 4-lane-per-row softmax reduction across R_BLK=6/8/16 -> 0-ULP
//   AR(T=1) == DFlash2(T=2..8) parity under unified 512-token splits.
// ============================================================================
template <uint R_BLK, bool HasCausalPos>
AFFINE4_ALWAYS_INLINE
inline void a5_nax_2sg_core(
    device const half *q_half,
    device const float *q_sum,
    device const uchar *cache_key,
    device const uchar *cache_value,
    device float *partial_max,
    device float *partial_sum,
    device half *partial_output_half,
    device const uint *page_table,
    device const uint *positions,
    uint query_heads,
    uint key_value_heads,
    uint tokens,
    uint page_tokens,
    uint split_count,
    uint split_tokens,
    uint r_stride,
    uint split,
    uint kvh,
    uint rblk,
    threadgroup float *tg_scores,
    threadgroup half *tg_p,
    threadgroup float *tg_q_sum,
    threadgroup uint *tg_row_ctx,
    threadgroup float *tg_m,
    threadgroup float *tg_l,
    threadgroup float *tg_rs,
    threadgroup float *tg_vb,
    uint lane,
    uint sg) {
  constexpr uint kHeadDim = 256u;
  const uint repeat = query_heads / key_value_heads;
  if (split >= split_count || kvh >= key_value_heads || repeat == 0u ||
      repeat > 16u) {
    return;
  }

  const uint total_active_rows = tokens * repeat;
  const uint row_base = rblk * 16u;
  if (row_base >= total_active_rows) {
    return;
  }
  const uint active_rows = min(R_BLK, total_active_rows - row_base);
  const uint flat_tid = sg * 32u + lane;

  uint max_ctx = 0u;
  if constexpr (HasCausalPos) {
    if (flat_tid < R_BLK) {
      if (flat_tid < active_rows) {
        const uint tok = (row_base + flat_tid) / repeat;
        tg_row_ctx[flat_tid] = positions[tok] + 1u;
      } else {
        tg_row_ctx[flat_tid] = 0u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    AFFINE4_UNROLL
    for (uint r = 0u; r < R_BLK; ++r) {
      max_ctx = max(max_ctx, tg_row_ctx[r]);
    }
  } else {
    max_ctx = positions[0] + 1u;
  }

  const uint pages_per_split = split_tokens / page_tokens;
  const uint total_pages = (max_ctx + page_tokens - 1u) / page_tokens;
  const uint start_page = split * pages_per_split;
  const uint end_page = min(total_pages, start_page + pages_per_split);

  // Fast empty-split exit without touching KV DRAM or partial_output_half.
  if (start_page >= end_page || max_ctx == 0u) {
    if (flat_tid < active_rows) {
      const uint r_global = row_base + flat_tid;
      const uint tok = r_global / repeat;
      const uint qh = kvh * repeat + (r_global % repeat);
      const ulong record =
          (ulong(tok) * query_heads + qh) * split_count + split;
      partial_max[record] = kNegInf;
      partial_sum[record] = 0.0f;
    }
    return;
  }

  // Load FOSCAR q_sum for this (kvh, rblk) and initialize softmax accumulators.
  if (flat_tid < R_BLK) {
    tg_m[flat_tid] = kNegInf;
    tg_l[flat_tid] = 0.0f;
    tg_rs[flat_tid] = 1.0f;
    tg_vb[flat_tid] = 0.0f;
    tg_vb[R_BLK + flat_tid] = 0.0f;
    const ulong qs_base0 =
        (ulong(kvh) * 2ul + 0ul) * ulong(r_stride) + ulong(row_base);
    const ulong qs_base1 =
        (ulong(kvh) * 2ul + 1ul) * ulong(r_stride) + ulong(row_base);
    tg_q_sum[flat_tid] = q_sum[qs_base0 + flat_tid];
    tg_q_sum[R_BLK + flat_tid] = q_sum[qs_base1 + flat_tid];
  }
  // Zero inactive rows [active_rows..15] of tg_p ONCE before the page loop.
  if (lane < 16u) {
    for (uint r = active_rows; r < 16u; ++r) {
      reinterpret_cast<threadgroup half4 *>(
          tg_p + sg * (16u * 64u) + r * 64u)[lane] = half4(0.0h);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ulong page_bytes =
      page_byte_stride(page_tokens, key_value_heads, kHeadDim);
  const ulong head_code_bytes = ulong(page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(key_value_heads) * head_code_bytes;
  const float scale2 = 0.0625f * kLog2e; // rsqrt(256.0f) * log2(e)

  constexpr auto desc_qk = matmul2d_descriptor(
      16, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc_qk, execution_simdgroup> op_qk;

  constexpr auto desc_pv = matmul2d_descriptor(
      16, 128, 64, false, false, false,
      matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc_pv, execution_simdgroup> op_pv;

  device const half *q_half_g =
      q_half +
      ((ulong(kvh) * 2ul + ulong(sg)) * ulong(r_stride) + ulong(row_base)) *
          128ul;
  tensor<device half, dextents<int32_t, 2>, tensor_inline> tq(
      (device half *)q_half_g, dextents<int32_t, 2>(128, 16));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(
      tg_p + sg * (16u * 64u), dextents<int32_t, 2>(64, 16));

  // Single register-resident cooperative_tensor accumulator (ZERO c_acc spill!).
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tv_init(
      (device uchar *)cache_value, dextents<int32_t, 2>(128, 64));
  auto c_acc = op_pv.template get_destination_cooperative_tensor<
      decltype(tp), decltype(tv_init), float>();
  AFFINE4_UNROLL
  for (uint i = 0u; i < c_acc.get_capacity(); ++i) {
    c_acc[i] = 0.0f;
  }

  const uint sub_row = flat_tid >> 2u;  // 0..15
  const uint sub_lane = flat_tid & 3u;  // 0..3
  const uint c0 = sub_lane * 16u;

  for (uint page = start_page; page < end_page; ++page) {
    const uint p_start = page * page_tokens;
    const uint phys_page = page_table[page];
    device const uchar *k_page_ptr =
        cache_key + ulong(phys_page) * page_bytes;
    device const uchar *v_page_ptr =
        cache_value + ulong(phys_page) * page_bytes;

    device const uchar *k_codes_g =
        k_page_ptr + ulong(kvh) * head_code_bytes + ulong(sg) * (64ul * 64ul);
    device const KvAffine4VecMeta *k_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(k_page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;
    device const uchar *v_codes_g =
        v_page_ptr + ulong(kvh) * head_code_bytes + ulong(sg) * (64ul * 64ul);
    device const KvAffine4VecMeta *v_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(v_page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;

    // Phase 1: HW uint4b_format 16x64x128 NAX Q@K^T + FOSCAR bias fold.
    // Touches ONLY tg_scores (disjoint from previous iteration's Phase 3).
    {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tk(
          (device uchar *)k_codes_g, dextents<int32_t, 2>(128, 64));
      auto c_qk = op_qk.template get_destination_cooperative_tensor<
          decltype(tq), decltype(tk), float>();
      op_qk.run(tq, tk, c_qk);

      threadgroup float *score_slice = tg_scores + sg * (R_BLK * 64u);
      const threadgroup float *q_sum_g = tg_q_sum + sg * R_BLK;
      AFFINE4_UNROLL
      for (uint i = 0u; i < c_qk.get_capacity(); ++i) {
        if (c_qk.is_valid_element(i)) {
          const auto coord = c_qk.get_multidimensional_index(i);
          const uint col = uint(coord[0]);
          const uint row = uint(coord[1]);
          if (row < R_BLK) {
            const float sk = float(k_meta[col].scale[sg]);
            const float zk = float(k_meta[col].zero[sg]);
            score_slice[row * 64u + col] =
                fma(sk, c_qk[i], zk * q_sum_g[row]);
          }
        }
      }
    }

    // Barrier 1/2 per page: both SGs finished writing tg_scores AND finished
    // reading previous page's tg_p / tg_rs / tg_vb.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Parallel 4-lanes-per-row log2 online softmax + V scale/bias prep.
    // Identical reduction tree across R_BLK=6/8/16 -> 0-ULP T=1 == T=2..8!
    if (sub_row < active_rows) {
      const uint row_ctx = HasCausalPos ? tg_row_ctx[sub_row] : max_ctx;
      const uint row_valid =
          (row_ctx > p_start) ? min(64u, row_ctx - p_start) : 0u;

      threadgroup half4 *p0_dst = reinterpret_cast<threadgroup half4 *>(
          tg_p + sub_row * 64u + c0);
      threadgroup half4 *p1_dst = reinterpret_cast<threadgroup half4 *>(
          tg_p + (16u * 64u) + sub_row * 64u + c0);

      if (row_valid == 0u) {
        p0_dst[0] = half4(0.0h);
        p0_dst[1] = half4(0.0h);
        p0_dst[2] = half4(0.0h);
        p0_dst[3] = half4(0.0h);
        p1_dst[0] = half4(0.0h);
        p1_dst[1] = half4(0.0h);
        p1_dst[2] = half4(0.0h);
        p1_dst[3] = half4(0.0h);
        if (sub_lane == 0u) {
          tg_rs[sub_row] = 1.0f;
          tg_vb[sub_row] = 0.0f;
          tg_vb[R_BLK + sub_row] = 0.0f;
        }
      } else {
        const threadgroup float4 *sc0 =
            reinterpret_cast<const threadgroup float4 *>(
                tg_scores + sub_row * 64u + c0);
        const threadgroup float4 *sc1 =
            reinterpret_cast<const threadgroup float4 *>(
                tg_scores + (R_BLK * 64u) + sub_row * 64u + c0);

        float4 s0 = (sc0[0] + sc1[0]) * scale2;
        float4 s1 = (sc0[1] + sc1[1]) * scale2;
        float4 s2 = (sc0[2] + sc1[2]) * scale2;
        float4 s3 = (sc0[3] + sc1[3]) * scale2;

        if (row_valid < 64u) {
          const uint4 idx0 = uint4(c0 + 0u, c0 + 1u, c0 + 2u, c0 + 3u);
          const uint4 idx1 = uint4(c0 + 4u, c0 + 5u, c0 + 6u, c0 + 7u);
          const uint4 idx2 = uint4(c0 + 8u, c0 + 9u, c0 + 10u, c0 + 11u);
          const uint4 idx3 = uint4(c0 + 12u, c0 + 13u, c0 + 14u, c0 + 15u);
          s0 = select(float4(kNegInf), s0, idx0 < row_valid);
          s1 = select(float4(kNegInf), s1, idx1 < row_valid);
          s2 = select(float4(kNegInf), s2, idx2 < row_valid);
          s3 = select(float4(kNegInf), s3, idx3 < row_valid);
        }

        const float4 m01 = max(s0, s1);
        const float4 m23 = max(s2, s3);
        const float4 m0123 = max(m01, m23);
        float lmax = max(max(m0123.x, m0123.y), max(m0123.z, m0123.w));
        lmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
        lmax = max(lmax, simd_shuffle_xor(lmax, ushort(2)));

        const float prev_m = tg_m[sub_row];
        const float new_m = max(prev_m, lmax);
        const float rescale =
            (prev_m <= kNegInfHalf) ? 0.0f : fast::exp2(prev_m - new_m);

        float4 p0 = fast::exp2(s0 - new_m);
        float4 p1 = fast::exp2(s1 - new_m);
        float4 p2 = fast::exp2(s2 - new_m);
        float4 p3 = fast::exp2(s3 - new_m);
        if (row_valid < 64u) {
          const uint4 idx0 = uint4(c0 + 0u, c0 + 1u, c0 + 2u, c0 + 3u);
          const uint4 idx1 = uint4(c0 + 4u, c0 + 5u, c0 + 6u, c0 + 7u);
          const uint4 idx2 = uint4(c0 + 8u, c0 + 9u, c0 + 10u, c0 + 11u);
          const uint4 idx3 = uint4(c0 + 12u, c0 + 13u, c0 + 14u, c0 + 15u);
          p0 = select(float4(0.0f), p0, idx0 < row_valid);
          p1 = select(float4(0.0f), p1, idx1 < row_valid);
          p2 = select(float4(0.0f), p2, idx2 < row_valid);
          p3 = select(float4(0.0f), p3, idx3 < row_valid);
        }

        const float4 psum4 = (p0 + p1) + (p2 + p3);
        float lsum = (psum4.x + psum4.y) + (psum4.z + psum4.w);

        float lvb0 = 0.0f;
        float lvb1 = 0.0f;

        // Chunk 0..3: fold v_meta scale into tg_p (half4) and v_meta zero into lvb.
        {
          const KvAffine4VecMeta m0 = v_meta[c0 + 0u];
          const KvAffine4VecMeta m1 = v_meta[c0 + 1u];
          const KvAffine4VecMeta m2 = v_meta[c0 + 2u];
          const KvAffine4VecMeta m3 = v_meta[c0 + 3u];
          const float4 sv0(float(m0.scale[0]), float(m1.scale[0]),
                           float(m2.scale[0]), float(m3.scale[0]));
          const float4 zv0(float(m0.zero[0]), float(m1.zero[0]),
                           float(m2.zero[0]), float(m3.zero[0]));
          const float4 sv1(float(m0.scale[1]), float(m1.scale[1]),
                           float(m2.scale[1]), float(m3.scale[1]));
          const float4 zv1(float(m0.zero[1]), float(m1.zero[1]),
                           float(m2.zero[1]), float(m3.zero[1]));
          p0_dst[0] = half4(p0 * sv0);
          p1_dst[0] = half4(p0 * sv1);
          lvb0 += dot(p0, zv0);
          lvb1 += dot(p0, zv1);
        }
        {
          const KvAffine4VecMeta m0 = v_meta[c0 + 4u];
          const KvAffine4VecMeta m1 = v_meta[c0 + 5u];
          const KvAffine4VecMeta m2 = v_meta[c0 + 6u];
          const KvAffine4VecMeta m3 = v_meta[c0 + 7u];
          const float4 sv0(float(m0.scale[0]), float(m1.scale[0]),
                           float(m2.scale[0]), float(m3.scale[0]));
          const float4 zv0(float(m0.zero[0]), float(m1.zero[0]),
                           float(m2.zero[0]), float(m3.zero[0]));
          const float4 sv1(float(m0.scale[1]), float(m1.scale[1]),
                           float(m2.scale[1]), float(m3.scale[1]));
          const float4 zv1(float(m0.zero[1]), float(m1.zero[1]),
                           float(m2.zero[1]), float(m3.zero[1]));
          p0_dst[1] = half4(p1 * sv0);
          p1_dst[1] = half4(p1 * sv1);
          lvb0 += dot(p1, zv0);
          lvb1 += dot(p1, zv1);
        }
        {
          const KvAffine4VecMeta m0 = v_meta[c0 + 8u];
          const KvAffine4VecMeta m1 = v_meta[c0 + 9u];
          const KvAffine4VecMeta m2 = v_meta[c0 + 10u];
          const KvAffine4VecMeta m3 = v_meta[c0 + 11u];
          const float4 sv0(float(m0.scale[0]), float(m1.scale[0]),
                           float(m2.scale[0]), float(m3.scale[0]));
          const float4 zv0(float(m0.zero[0]), float(m1.zero[0]),
                           float(m2.zero[0]), float(m3.zero[0]));
          const float4 sv1(float(m0.scale[1]), float(m1.scale[1]),
                           float(m2.scale[1]), float(m3.scale[1]));
          const float4 zv1(float(m0.zero[1]), float(m1.zero[1]),
                           float(m2.zero[1]), float(m3.zero[1]));
          p0_dst[2] = half4(p2 * sv0);
          p1_dst[2] = half4(p2 * sv1);
          lvb0 += dot(p2, zv0);
          lvb1 += dot(p2, zv1);
        }
        {
          const KvAffine4VecMeta m0 = v_meta[c0 + 12u];
          const KvAffine4VecMeta m1 = v_meta[c0 + 13u];
          const KvAffine4VecMeta m2 = v_meta[c0 + 14u];
          const KvAffine4VecMeta m3 = v_meta[c0 + 15u];
          const float4 sv0(float(m0.scale[0]), float(m1.scale[0]),
                           float(m2.scale[0]), float(m3.scale[0]));
          const float4 zv0(float(m0.zero[0]), float(m1.zero[0]),
                           float(m2.zero[0]), float(m3.zero[0]));
          const float4 sv1(float(m0.scale[1]), float(m1.scale[1]),
                           float(m2.scale[1]), float(m3.scale[1]));
          const float4 zv1(float(m0.zero[1]), float(m1.zero[1]),
                           float(m2.zero[1]), float(m3.zero[1]));
          p0_dst[3] = half4(p3 * sv0);
          p1_dst[3] = half4(p3 * sv1);
          lvb0 += dot(p3, zv0);
          lvb1 += dot(p3, zv1);
        }

        lsum += simd_shuffle_xor(lsum, ushort(1));
        lsum += simd_shuffle_xor(lsum, ushort(2));
        lvb0 += simd_shuffle_xor(lvb0, ushort(1));
        lvb0 += simd_shuffle_xor(lvb0, ushort(2));
        lvb1 += simd_shuffle_xor(lvb1, ushort(1));
        lvb1 += simd_shuffle_xor(lvb1, ushort(2));

        if (sub_lane == 0u) {
          tg_rs[sub_row] = rescale;
          tg_m[sub_row] = new_m;
          tg_l[sub_row] = fma(tg_l[sub_row], rescale, lsum);
          tg_vb[sub_row] = lvb0;
          tg_vb[R_BLK + sub_row] = lvb1;
        }
      }
    }

    // Barrier 2/2 per page: tg_p, tg_rs, tg_vb ready for Phase 3, and
    // tg_scores read is complete before next iteration's Phase 1.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: Rescale c_acc in-place + HW uint4b_format 16x128x64 NAX P@V
    // directly into c_acc (mode::multiply_accumulate).
    AFFINE4_UNROLL
    for (uint i = 0u; i < c_acc.get_capacity(); ++i) {
      if (c_acc.is_valid_element(i)) {
        const auto coord = c_acc.get_multidimensional_index(i);
        const uint row = uint(coord[1]);
        if (row < R_BLK) {
          c_acc[i] = fma(c_acc[i], tg_rs[row], tg_vb[sg * R_BLK + row]);
        }
      }
    }
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tv(
        (device uchar *)v_codes_g, dextents<int32_t, 2>(128, 64));
    op_pv.run(tp, tv, c_acc);
  }

  // MANDATORY Law #4 barrier: both SGs must finish final Phase 3 before
  // sg=0 reuses tg_rs for 1.0f / tg_l!
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (flat_tid < active_rows) {
    const uint r_global = row_base + flat_tid;
    const uint tok = r_global / repeat;
    const uint qh = kvh * repeat + (r_global % repeat);
    const ulong record =
        (ulong(tok) * query_heads + qh) * split_count + split;
    const float l_val = tg_l[flat_tid];
    partial_max[record] = tg_m[flat_tid];
    partial_sum[record] = l_val;
    tg_rs[flat_tid] = (l_val > 0.0f) ? (1.0f / l_val) : 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Store normalized half partials [tok, qh, split, 256].
  AFFINE4_UNROLL
  for (uint i = 0u; i < c_acc.get_capacity(); ++i) {
    if (c_acc.is_valid_element(i)) {
      const auto coord = c_acc.get_multidimensional_index(i);
      const uint col = sg * 128u + uint(coord[0]);
      const uint row = uint(coord[1]);
      if (row < active_rows) {
        const uint r_global = row_base + row;
        const uint tok = r_global / repeat;
        const uint qh = kvh * repeat + (r_global % repeat);
        const ulong record =
            (ulong(tok) * query_heads + qh) * split_count + split;
        partial_output_half[record * kHeadDim + col] =
            half(c_acc[i] * tg_rs[row]);
      }
    }
  }
}

template <uint R_BLK, bool HasCausalPos>
AFFINE4_ALWAYS_INLINE
inline void a5_nax_2sg_impl(
    device const half *q_half,
    device const float *q_sum,
    device const uchar *cache_key,
    device const uchar *cache_value,
    device float *partial_max,
    device float *partial_sum,
    device half *partial_output_half,
    device const uint *page_table,
    device const uint *positions,
    constant PagedSplitAttentionParams &p,
    threadgroup float *tg_scores,
    threadgroup half *tg_p,
    threadgroup float *tg_q_sum,
    threadgroup uint *tg_row_ctx,
    threadgroup float *tg_m,
    threadgroup float *tg_l,
    threadgroup float *tg_rs,
    threadgroup float *tg_vb,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 tgs_per_grid) {
  a5_nax_2sg_core<R_BLK, HasCausalPos>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, p.query_heads,
      p.key_value_heads, p.tokens, p.page_tokens, p.split_count, p.split_tokens,
      tgs_per_grid.z * 16u, tg.x, tg.y, tg.z, tg_scores, tg_p, tg_q_sum,
      tg_row_ctx, tg_m, tg_l, tg_rs, tg_vb, lane, sg);
}

template <uint R_BLK, bool HasCausalPos>
AFFINE4_ALWAYS_INLINE
inline void a5_nax_2sg_batched_impl(
    device const half *q_half,
    device const float *q_sum,
    device const uchar *cache_key,
    device const uchar *cache_value,
    device float *partial_max,
    device float *partial_sum,
    device half *partial_output_half,
    device const uint *positions,
    device const uint *pt0, device const uint *pt1,
    device const uint *pt2, device const uint *pt3,
    device const uint *pt4, device const uint *pt5,
    device const uint *pt6, device const uint *pt7,
    constant BatchedPagedAttentionParams &p,
    threadgroup float *tg_scores,
    threadgroup half *tg_p,
    threadgroup float *tg_q_sum,
    threadgroup uint *tg_row_ctx,
    threadgroup float *tg_m,
    threadgroup float *tg_l,
    threadgroup float *tg_rs,
    threadgroup float *tg_vb,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint slot = tg.z;
  if (slot >= p.batch_size || p.key_value_heads == 0u) {
    return;
  }
  const uint slot_tokens = p.widths[slot];
  if (slot_tokens == 0u) {
    return;
  }
  const uint row_off = p.row_offsets[slot];
  const uint split = tg.x;
  const uint kvh = tg.y % p.key_value_heads;
  const uint rblk = tg.y / p.key_value_heads;
  if (rblk >= p.n_rblks) {
    return;
  }
  device const uint *slot_pt =
      select_pt8(slot, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7);
  const ulong slot_q_stride =
      ulong(p.key_value_heads) * 2ul * ulong(p.r_stride);
  const ulong slot_rec_base =
      ulong(row_off) * ulong(p.query_heads) * ulong(p.split_count);
  a5_nax_2sg_core<R_BLK, HasCausalPos>(
      q_half + ulong(slot) * slot_q_stride * 128ul,
      q_sum + ulong(slot) * slot_q_stride,
      cache_key, cache_value,
      partial_max + slot_rec_base,
      partial_sum + slot_rec_base,
      partial_output_half + slot_rec_base * 256ul,
      slot_pt, positions + row_off,
      p.query_heads, p.key_value_heads, slot_tokens, p.page_tokens,
      p.split_count, p.split_tokens, p.r_stride, split, kvh, rblk,
      tg_scores, tg_p, tg_q_sum, tg_row_ctx, tg_m, tg_l, tg_rs, tg_vb,
      lane, sg);
}

// ============================================================================
// Register-resident Frag16x16 helpers for c5_aff4_frag64_bfp Flash-Prefill
// (41.1 TFLOP/s, max_err = 0.001953, 0 alloca, exact Law #1 nibble unpack)
// ============================================================================
struct Frag16x16Aff4 {
  static constexpr constant short kElemsPerFrag = 8;
  static constexpr constant short kElemRows = 2;
  static constexpr constant short kElemCols = 4;
  static constexpr constant short kElemRowsJump = 8;

  template <typename U>
  using frag_t = vec<U, kElemsPerFrag>;

  AFFINE4_ALWAYS_INLINE
  static inline short2 get_coord(ushort simd_lane_id) {
    const short qid = short(simd_lane_id >> 2u);
    const short fm = (qid & 4) | short((simd_lane_id >> 1u) & 3u);
    const short fn = ((qid & 2) | short(simd_lane_id & 1u)) * 4;
    return short2(fn, fm);
  }

  AFFINE4_ALWAYS_INLINE
  static inline bfloat4 clean_bf4(bfloat4 v) {
    // Fast-math-safe BF16 NaN sanitize via integer bit test (never folded away).
    const ushort4 bits = as_type<ushort4>(v);
    return select(v, bfloat4(bfloat(0.0f)),
                  (bits & ushort4(0x7fffu)) > ushort4(0x7f80u));
  }

  AFFINE4_ALWAYS_INLINE
  static inline void load_bf4(thread frag_t<bfloat> &dst,
                              device const bfloat *src, ulong row_stride,
                              short off_c, short2 sc) {
    const device bfloat *p0 =
        src + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    const device bfloat *p1 = p0 + ulong(kElemRowsJump) * row_stride;
    const bfloat4 v0 = *reinterpret_cast<device const bfloat4 *>(p0);
    const bfloat4 v1 = *reinterpret_cast<device const bfloat4 *>(p1);
    dst = frag_t<bfloat>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  AFFINE4_ALWAYS_INLINE
  static inline void load_rows_clean_bf4(thread frag_t<bfloat> &dst,
                                         device const bfloat *src,
                                         ulong row_stride, short lim_rows,
                                         short off_c, short2 sc) {
    const device bfloat *p0 =
        src + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    const bfloat4 v0 =
        (sc.y < lim_rows)
            ? clean_bf4(*reinterpret_cast<device const bfloat4 *>(p0))
            : bfloat4(bfloat(0.0f));
    const bfloat4 v1 =
        (short(sc.y + kElemRowsJump) < lim_rows)
            ? clean_bf4(*reinterpret_cast<device const bfloat4 *>(
                  p0 + ulong(kElemRowsJump) * row_stride))
            : bfloat4(bfloat(0.0f));
    dst = frag_t<bfloat>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  AFFINE4_ALWAYS_INLINE
  static inline void load_h4(thread frag_t<half> &dst,
                             device const bfloat *src, ulong row_stride,
                             short off_c, short2 sc) {
    const device bfloat *p0 =
        src + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    const device bfloat *p1 = p0 + ulong(kElemRowsJump) * row_stride;
    const half4 v0 = half4(float4(*reinterpret_cast<device const bfloat4 *>(p0)));
    const half4 v1 = half4(float4(*reinterpret_cast<device const bfloat4 *>(p1)));
    dst = frag_t<half>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  AFFINE4_ALWAYS_INLINE
  static inline void load_rows_clean_h4(thread frag_t<half> &dst,
                                        device const bfloat *src,
                                        ulong row_stride, short lim_rows,
                                        short off_c, short2 sc) {
    const device bfloat *p0 =
        src + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    const half4 v0 =
        (sc.y < lim_rows)
            ? half4(float4(clean_bf4(*reinterpret_cast<device const bfloat4 *>(p0))))
            : half4(0.0h);
    const half4 v1 =
        (short(sc.y + kElemRowsJump) < lim_rows)
            ? half4(float4(clean_bf4(*reinterpret_cast<device const bfloat4 *>(
                  p0 + ulong(kElemRowsJump) * row_stride))))
            : half4(0.0h);
    dst = frag_t<half>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  // Exact Law #1 affine4 unpack: half4(ushort4(...)) + fma(q_h4, s, z) -> half4
  // Pre-indexed row pointers avoid repeated r*32 + sn_u16 ALU across d=0..7!
  AFFINE4_ALWAYS_INLINE
  static inline frag_t<half> dequant_ptr_h(device const ushort *p0,
                                           device const ushort *p1, half s0,
                                           half z0, half s1, half z1,
                                           uint col_off) {
    const ushort pk0 = p0[col_off];
    const ushort pk1 = p1[col_off];
    const half4 q0 = half4(ushort4(pk0 & 0xFu, (pk0 >> 4u) & 0xFu,
                                   (pk0 >> 8u) & 0xFu, pk0 >> 12u));
    const half4 q1 = half4(ushort4(pk1 & 0xFu, (pk1 >> 4u) & 0xFu,
                                   (pk1 >> 8u) & 0xFu, pk1 >> 12u));
    const half4 v0 = fma(q0, half4(s0), half4(z0));
    const half4 v1 = fma(q1, half4(s1), half4(z1));
    return frag_t<half>(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);
  }

  template <bool TransposeB>
  AFFINE4_ALWAYS_INLINE
  static inline void mma_1x2_f16(thread frag_t<float> &c0,
                                 thread frag_t<float> &c1,
                                 const thread frag_t<half> &a,
                                 const thread frag_t<half> &b0,
                                 const thread frag_t<half> &b1) {
    constexpr auto desc = matmul2d_descriptor(
        16, 32, 16, false, TransposeB, true,
        matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    auto ct_a =
        op.template get_left_input_cooperative_tensor<half, half, float>();
    auto ct_b =
        op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ct_c = op.template get_destination_cooperative_tensor<
        remove_addrspace_t<decltype(ct_a)>,
        remove_addrspace_t<decltype(ct_b)>, float>();
    AFFINE4_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      ct_a[i] = a[i];
      ct_b[i] = b0[i];
      ct_b[kElemsPerFrag + i] = b1[i];
      ct_c[i] = c0[i];
      ct_c[kElemsPerFrag + i] = c1[i];
    }
    op.run(ct_a, ct_b, ct_c);
    AFFINE4_UNROLL
    for (short i = 0; i < kElemsPerFrag; ++i) {
      c0[i] = ct_c[i];
      c1[i] = ct_c[kElemsPerFrag + i];
    }
  }

  AFFINE4_ALWAYS_INLINE
  static inline void store_bf4(const thread frag_t<float> &src,
                               device bfloat *dst, ulong row_stride,
                               short off_c, short2 sc) {
    device bfloat *p0 = dst + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    device bfloat *p1 = p0 + ulong(kElemRowsJump) * row_stride;
    *reinterpret_cast<device bfloat4 *>(p0) =
        bfloat4(float4(src[0], src[1], src[2], src[3]));
    *reinterpret_cast<device bfloat4 *>(p1) =
        bfloat4(float4(src[4], src[5], src[6], src[7]));
  }

  AFFINE4_ALWAYS_INLINE
  static inline void store_rows_bf4(const thread frag_t<float> &src,
                                    device bfloat *dst, ulong row_stride,
                                    short lim_rows, short off_c, short2 sc) {
    device bfloat *p0 = dst + ulong(sc.y) * row_stride + ulong(off_c + sc.x);
    if (sc.y < lim_rows) {
      *reinterpret_cast<device bfloat4 *>(p0) =
          bfloat4(float4(src[0], src[1], src[2], src[3]));
    }
    if (short(sc.y + kElemRowsJump) < lim_rows) {
      *reinterpret_cast<device bfloat4 *>(p0 + ulong(kElemRowsJump) * row_stride) =
          bfloat4(float4(src[4], src[5], src[6], src[7]));
    }
  }
};

// ============================================================================
// Preserved compatibility helpers for !split (small T) and HD == 128 split
// (fast-math-safe: uses kNegInf instead of -INFINITY)
// ============================================================================
template <uint HeadDim, uint Groups>
inline void run_superblock_nax_g128(
    device const uchar *cache_key,
    device const uchar *cache_value,
    device const uint *page_table,
    uint page_tokens,
    uint kv_heads,
    uint kvh,
    uint start_page,
    uint end_page,
    uint max_context_tokens,
    uint active_rows,
    const threadgroup uint *tg_row_ctx,
    const threadgroup float *tg_q_sum,
    threadgroup half *tg_q,
    threadgroup float *tg_scores,
    threadgroup half *tg_p,
    threadgroup float *tg_rescale,
    threadgroup float *tg_v_scale,
    threadgroup float *tg_v_bias,
    threadgroup float *tg_m_blk,
    threadgroup float *tg_l_blk,
    thread float (&acc_blk)[64],
    uint lane,
    uint sg) {
  constexpr uint kChunkRows = 16u;
  const uint flat_tid = sg * 32u + lane;

  if (flat_tid < kChunkRows) {
    tg_m_blk[flat_tid] = kNegInf;
    tg_l_blk[flat_tid] = 0.0f;
  }
  for (uint i = 0u; i < 64u; ++i) {
    acc_blk[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (start_page >= end_page || max_context_tokens == 0u) {
    return;
  }

  const ulong page_bytes = page_byte_stride(page_tokens, kv_heads, HeadDim);
  const ulong head_code_bytes = ulong(page_tokens) * ulong(HeadDim / 2u);
  const ulong all_codes_bytes = ulong(kv_heads) * head_code_bytes;
  const float attn_scale = rsqrt(float(HeadDim));

  constexpr auto desc_qk = matmul2d_descriptor(
      16, 64, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc_qk, execution_simdgroup> op_qk;

  constexpr auto desc_pv = matmul2d_descriptor(
      16, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc_pv, execution_simdgroup> op_pv;

  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tq(
      tg_q + sg * (16u * 128u), dextents<int32_t, 2>(128, 16));
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp(
      tg_p + sg * (16u * 64u), dextents<int32_t, 2>(64, 16));

  for (uint page = start_page; page < end_page; ++page) {
    const uint p_start = page * page_tokens;
    if (p_start >= max_context_tokens) {
      break;
    }
    const uint phys_page = page_table[page];
    device const uchar *k_page_ptr =
        cache_key + ulong(phys_page) * page_bytes;
    device const uchar *v_page_ptr =
        cache_value + ulong(phys_page) * page_bytes;

    device const uchar *k_codes_g =
        k_page_ptr + ulong(kvh) * head_code_bytes + ulong(sg) * (64ul * 64ul);
    device const KvAffine4VecMeta *k_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(k_page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;
    device const uchar *v_codes_g =
        v_page_ptr + ulong(kvh) * head_code_bytes + ulong(sg) * (64ul * 64ul);
    device const KvAffine4VecMeta *v_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(v_page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;

    if (sg < Groups) {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tk(
          (device uchar *)k_codes_g, dextents<int32_t, 2>(128, 64));
      auto c_qk = op_qk.template get_destination_cooperative_tensor<
          decltype(tq), decltype(tk), float>();
      op_qk.run(tq, tk, c_qk);

      const uint elems = c_qk.get_capacity();
      threadgroup float *score_slice = tg_scores + sg * (16u * 64u);
      const threadgroup float *q_sum_g = tg_q_sum + sg * 16u;
      for (uint i = 0u; i < elems; ++i) {
        if (c_qk.is_valid_element(i)) {
          const auto coord = c_qk.get_multidimensional_index(i);
          const uint col = uint(coord[0]);
          const uint row = uint(coord[1]);
          const float sk = float(k_meta[col].scale[sg]);
          const float zk = float(k_meta[col].zero[sg]);
          score_slice[row * 64u + col] =
              fma(sk, c_qk[i], zk * q_sum_g[row]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (flat_tid < kChunkRows) {
      const uint r = flat_tid;
      const uint row_ctx = (r < active_rows) ? tg_row_ctx[r] : 0u;
      const uint row_valid =
          (row_ctx > p_start) ? min(page_tokens, row_ctx - p_start) : 0u;

      if (row_valid == 0u) {
        tg_rescale[r] = 1.0f;
        for (uint g = 0u; g < Groups; ++g) {
          tg_v_scale[g * 16u + r] = 0.0f;
          tg_v_bias[g * 16u + r] = 0.0f;
          threadgroup half *p_row = tg_p + g * (16u * 64u) + r * 64u;
          for (uint c = 0u; c < 64u; ++c) {
            p_row[c] = half(0.0h);
          }
        }
      } else {
        float page_max = kNegInf;
        for (uint c = 0u; c < row_valid; ++c) {
          float raw = tg_scores[r * 64u + c];
          if constexpr (Groups == 2u) {
            raw += tg_scores[16u * 64u + r * 64u + c];
          }
          const float s = raw * attn_scale;
          tg_scores[r * 64u + c] = s;
          page_max = max(page_max, s);
        }
        const float prev_m = tg_m_blk[r];
        const float new_m = max(prev_m, page_max);
        const float rescale =
            (prev_m <= kNegInfHalf) ? 0.0f : exp(prev_m - new_m);

        float v_max_s[Groups];
        float v_bias[Groups];
        for (uint g = 0u; g < Groups; ++g) {
          v_max_s[g] = 0.0f;
          v_bias[g] = 0.0f;
        }
        for (uint c = 0u; c < row_valid; ++c) {
          for (uint g = 0u; g < Groups; ++g) {
            v_max_s[g] = max(v_max_s[g], fabs(float(v_meta[c].scale[g])));
          }
        }

        float inv_v_max[Groups];
        for (uint g = 0u; g < Groups; ++g) {
          inv_v_max[g] =
              (v_max_s[g] > 0.0f) ? precise::divide(1.0f, v_max_s[g]) : 0.0f;
        }

        float page_sum = 0.0f;
        for (uint c = 0u; c < row_valid; ++c) {
          const float p_val = exp(tg_scores[r * 64u + c] - new_m);
          page_sum += p_val;
          for (uint g = 0u; g < Groups; ++g) {
            const float sv = float(v_meta[c].scale[g]);
            const float zv = float(v_meta[c].zero[g]);
            v_bias[g] = fma(p_val, zv, v_bias[g]);
            tg_p[g * (16u * 64u) + r * 64u + c] =
                half((p_val * sv) * inv_v_max[g]);
          }
        }
        for (uint c = row_valid; c < 64u; ++c) {
          for (uint g = 0u; g < Groups; ++g) {
            tg_p[g * (16u * 64u) + r * 64u + c] = half(0.0h);
          }
        }

        tg_rescale[r] = rescale;
        tg_m_blk[r] = new_m;
        tg_l_blk[r] = tg_l_blk[r] * rescale + page_sum;
        for (uint g = 0u; g < Groups; ++g) {
          tg_v_scale[g * 16u + r] = v_max_s[g];
          tg_v_bias[g * 16u + r] = v_bias[g];
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < Groups) {
      tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tv(
          (device uchar *)v_codes_g, dextents<int32_t, 2>(128, 64));
      auto c_pv = op_pv.template get_destination_cooperative_tensor<
          decltype(tp), decltype(tv), float>();
      op_pv.run(tp, tv, c_pv);

      const uint elems = c_pv.get_capacity();
      for (uint i = 0u; i < elems; ++i) {
        if (c_pv.is_valid_element(i)) {
          const auto coord = c_pv.get_multidimensional_index(i);
          const uint row = uint(coord[1]);
          acc_blk[i] = fma(acc_blk[i], tg_rescale[row],
                           fma(c_pv[i], tg_v_scale[sg * 16u + row],
                               tg_v_bias[sg * 16u + row]));
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

template <uint HeadDim, uint Groups>
inline void load_q_and_foscar_sums(
    device const bfloat *query,
    device const uint *positions,
    uint tokens,
    uint query_heads,
    uint key_value_heads,
    uint query_token_stride,
    uint query_head_stride,
    uint t_base,
    uint bq,
    uint kvh,
    threadgroup half *tg_q,
    threadgroup float *tg_q_sum,
    threadgroup uint *tg_row_ctx,
    thread uint &out_active_rows,
    thread uint &out_max_ctx,
    uint lane,
    uint sg) {
  const uint repeat = query_heads / key_value_heads;
  const uint active_q_tokens =
      (t_base < tokens) ? min(bq, tokens - t_base) : 0u;
  const uint active_rows = active_q_tokens * repeat;
  out_active_rows = active_rows;

  const uint flat_tid = sg * 32u + lane;
  const uint total_threads = Groups * 32u;

  for (uint r = flat_tid; r < 16u; r += total_threads) {
    if (r < active_rows) {
      const uint tok = t_base + (r / repeat);
      tg_row_ctx[r] = positions[tok] + 1u;
    } else {
      tg_row_ctx[r] = 0u;
    }
  }

  if (sg < Groups) {
    threadgroup half *q_slice = tg_q + sg * (16u * 128u);
    threadgroup float *q_sum_slice = tg_q_sum + sg * 16u;
    for (uint r = 0u; r < 16u; ++r) {
      if (r < active_rows) {
        const uint tok = t_base + (r / repeat);
        const uint qh = kvh * repeat + (r % repeat);
        device const bfloat4 *q4 = reinterpret_cast<device const bfloat4 *>(
            query + ulong(tok) * query_token_stride +
            ulong(qh) * query_head_stride + sg * 128u);
        const float4 qv = float4(q4[lane]);
        const half4 qh4 = half4(qv);
        reinterpret_cast<threadgroup half4 *>(q_slice + r * 128u)[lane] = qh4;
        const float4 qf = float4(qh4);
        const float s = simd_sum((qf.x + qf.y) + (qf.z + qf.w));
        if (lane == 0u) {
          q_sum_slice[r] = s;
        }
      } else {
        reinterpret_cast<threadgroup half4 *>(q_slice + r * 128u)[lane] =
            half4(0.0h);
        if (lane == 0u) {
          q_sum_slice[r] = 0.0f;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint max_ctx = 0u;
  for (uint r = 0u; r < active_rows; ++r) {
    max_ctx = max(max_ctx, tg_row_ctx[r]);
  }
  out_max_ctx = max_ctx;
}

template <bool PackQHalf>
AFFINE4_ALWAYS_INLINE
inline void qk_norm_rope_kv_append_affine4_impl(
    device bfloat *query,
    device bfloat *key,
    device const bfloat *value,
    device const bfloat *query_norm,
    device const bfloat *key_norm,
    device uchar *cache_key,
    device uchar *cache_value,
    device const uint *page_table,
    device const uint *positions,
    device half *q_half,
    device float *q_sum,
    constant AttentionPostprocessParams &params,
    threadgroup bfloat *tg_vec,
    uint lane,
    uint3 tg_position) {
  const uint heads_per_token = params.query_heads + params.key_value_heads;
  const uint logical_head = tg_position.y;
  const uint token = logical_head / heads_per_token;
  if (token >= params.tokens ||
      (params.head_dim != 256u && params.head_dim != 128u)) {
    return;
  }
  const uint local_head = logical_head - token * heads_per_token;
  const bool is_query = local_head < params.query_heads;
  const uint head = is_query ? local_head : (local_head - params.query_heads);
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  device const bfloat *norm = is_query ? query_norm : key_norm;

  float square_sum = 0.0f;
  for (uint column = lane; column < params.head_dim; column += 32u) {
    const float val = float(row[column]);
    square_sum = fma(val, val, square_sum);
  }
  // Use precise:: math so fused output matches attention_exact to 0 ULP!
  const float inverse_rms = precise::rsqrt(
      precise::divide(simd_sum(square_sum), float(params.head_dim)) +
      params.epsilon);
  const uint half_rotary = params.rotary_dim / 2u;
  const uint logical_pos = positions[token];
  const float position = float(logical_pos);

  for (uint pair = lane; pair < half_rotary; pair += 32u) {
    const uint first_index = pair;
    const uint second_index = pair + half_rotary;
    const float first = float(row[first_index]) * inverse_rms *
                        (float(norm[first_index]) + params.weight_offset);
    const float second = float(row[second_index]) * inverse_rms *
                         (float(norm[second_index]) + params.weight_offset);
    const float exponent =
        precise::divide(float(2u * pair), float(params.rotary_dim));
    const float angle =
        precise::divide(position, precise::pow(params.theta, exponent));
    const float cosine = precise::cos(angle);
    const float sine = precise::sin(angle);
    const bfloat o0 = bfloat(fma(-second, sine, first * cosine));
    const bfloat o1 = bfloat(fma(first, sine, second * cosine));
    row[first_index] = o0;
    row[second_index] = o1;
    if constexpr (PackQHalf) {
      tg_vec[first_index] = o0;
      tg_vec[second_index] = o1;
    } else if (!is_query) {
      tg_vec[first_index] = o0;
      tg_vec[second_index] = o1;
    }
  }
  for (uint column = params.rotary_dim + lane; column < params.head_dim;
       column += 32u) {
    const bfloat normed = bfloat(float(row[column]) * inverse_rms *
                                 (float(norm[column]) + params.weight_offset));
    row[column] = normed;
    if constexpr (PackQHalf) {
      tg_vec[column] = normed;
    } else if (!is_query) {
      tg_vec[column] = normed;
    }
  }

  if (is_query) {
    if constexpr (PackQHalf) {
      // T == 1, HD == 256 split-KV: pack [KVH, 2, 16, 128] half + q_sum inline!
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint repeat = params.query_heads / params.key_value_heads;
      const uint kvh = head / repeat;
      const uint r = head % repeat;
      constexpr uint r_stride = 16u;
      AFFINE4_UNROLL
      for (uint g = 0u; g < 2u; ++g) {
        threadgroup const bfloat4 *q4 =
            reinterpret_cast<threadgroup const bfloat4 *>(tg_vec + g * 128u);
        const half4 qh4 = half4(float4(q4[lane]));
        const ulong base_row = (ulong(kvh) * 2ul + g) * r_stride + r;
        reinterpret_cast<device half4 *>(q_half + base_row * 128ul)[lane] = qh4;
        const float4 qf = float4(qh4);
        const float s = simd_sum((qf.x + qf.y) + (qf.z + qf.w));
        if (lane == 0u) {
          q_sum[base_row] = s;
        }
        // Head 0 of each KV group zeroes padding rows [repeat..15].
        if (r == 0u) {
          for (uint rr = repeat; rr < r_stride; ++rr) {
            const ulong pad_row = (ulong(kvh) * 2ul + g) * r_stride + rr;
            reinterpret_cast<device half4 *>(q_half + pad_row * 128ul)[lane] =
                half4(0.0h);
            if (lane == 0u) {
              q_sum[pad_row] = 0.0f;
            }
          }
        }
      }
    }
  } else {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint kPageTokens = 64u;
    const uint logical_page = logical_pos >> 6u;
    const uint within = logical_pos & 63u;
    const uint phys_page = page_table[logical_page];
    const ulong page_bytes =
        page_byte_stride(kPageTokens, params.key_value_heads, params.head_dim);
    const ulong head_code_bytes =
        ulong(kPageTokens) * ulong(params.head_dim / 2u);
    const ulong all_codes_bytes =
        ulong(params.key_value_heads) * head_code_bytes;

    device uchar *k_page = cache_key + ulong(phys_page) * page_bytes;
    device uchar *v_page = cache_value + ulong(phys_page) * page_bytes;
    const ulong kv_width = ulong(params.key_value_heads) * params.head_dim;
    device const bfloat *v_src =
        value + ulong(token) * kv_width + ulong(head) * params.head_dim;

    quantize_vec_g128_tg(
        tg_vec, k_page + ulong(head) * head_code_bytes,
        reinterpret_cast<device KvAffine4VecMeta *>(k_page + all_codes_bytes) +
            ulong(head) * kPageTokens,
        kPageTokens, within, params.head_dim, lane);
    quantize_vec_g128_device(
        v_src, v_page + ulong(head) * head_code_bytes,
        reinterpret_cast<device KvAffine4VecMeta *>(v_page + all_codes_bytes) +
            ulong(head) * kPageTokens,
        kPageTokens, within, params.head_dim, lane);
  }
}

template <bool PackQHalf>
AFFINE4_ALWAYS_INLINE
inline void qk_norm_rope_kv_append_batched_affine4_impl(
    device bfloat *query,
    device bfloat *key,
    device const bfloat *value,
    device const bfloat *query_norm,
    device const bfloat *key_norm,
    device uchar *cache_key,
    device uchar *cache_value,
    device const uint *positions,
    device const uint *pt0, device const uint *pt1,
    device const uint *pt2, device const uint *pt3,
    device const uint *pt4, device const uint *pt5,
    device const uint *pt6, device const uint *pt7,
    device half *q_half,
    device float *q_sum,
    constant BatchedAttentionPostprocessParams &params,
    threadgroup bfloat *tg_vec,
    uint lane,
    uint3 tg_position) {
  const uint heads_per_token = params.query_heads + params.key_value_heads;
  const uint logical_head = tg_position.y;
  const uint token = logical_head / heads_per_token;
  if (token >= params.total_tokens ||
      (params.head_dim != 256u && params.head_dim != 128u)) {
    return;
  }
  const uint local_head = logical_head - token * heads_per_token;
  const bool is_query = local_head < params.query_heads;
  const uint head = is_query ? local_head : (local_head - params.query_heads);
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  device const bfloat *norm = is_query ? query_norm : key_norm;

  float square_sum = 0.0f;
  for (uint column = lane; column < params.head_dim; column += 32u) {
    const float val = float(row[column]);
    square_sum = fma(val, val, square_sum);
  }
  // Use precise:: math so fused output matches attention_exact to 0 ULP!
  const float inverse_rms = precise::rsqrt(
      precise::divide(simd_sum(square_sum), float(params.head_dim)) +
      params.epsilon);
  const uint half_rotary = params.rotary_dim / 2u;
  const uint logical_pos = positions[token];
  const float position = float(logical_pos);

  for (uint pair = lane; pair < half_rotary; pair += 32u) {
    const uint first_index = pair;
    const uint second_index = pair + half_rotary;
    const float first = float(row[first_index]) * inverse_rms *
                        (float(norm[first_index]) + params.weight_offset);
    const float second = float(row[second_index]) * inverse_rms *
                         (float(norm[second_index]) + params.weight_offset);
    const float exponent =
        precise::divide(float(2u * pair), float(params.rotary_dim));
    const float angle =
        precise::divide(position, precise::pow(params.theta, exponent));
    const float cosine = precise::cos(angle);
    const float sine = precise::sin(angle);
    const bfloat o0 = bfloat(fma(-second, sine, first * cosine));
    const bfloat o1 = bfloat(fma(first, sine, second * cosine));
    row[first_index] = o0;
    row[second_index] = o1;
    if constexpr (PackQHalf) {
      tg_vec[first_index] = o0;
      tg_vec[second_index] = o1;
    } else if (!is_query) {
      tg_vec[first_index] = o0;
      tg_vec[second_index] = o1;
    }
  }
  for (uint column = params.rotary_dim + lane; column < params.head_dim;
       column += 32u) {
    const bfloat normed = bfloat(float(row[column]) * inverse_rms *
                                 (float(norm[column]) + params.weight_offset));
    row[column] = normed;
    if constexpr (PackQHalf) {
      tg_vec[column] = normed;
    } else if (!is_query) {
      tg_vec[column] = normed;
    }
  }

  uint slot = 0u;
  if (PackQHalf || !is_query) {
    for (uint s = 0u; s < params.batch_size; ++s) {
      if (token >= params.row_offsets[s] &&
          token < params.row_offsets[s] + params.widths[s]) {
        slot = s;
        break;
      }
    }
  }

  if (is_query) {
    if constexpr (PackQHalf) {
      simdgroup_barrier(mem_flags::mem_threadgroup);
      const uint repeat = params.query_heads / params.key_value_heads;
      const uint kvh = head / repeat;
      const uint q_in_group = head - kvh * repeat;
      const uint local_tok = token - params.row_offsets[slot];
      const uint r = local_tok * repeat + q_in_group;
      const uint r_stride = params.r_stride;
      const ulong slot_base =
          ulong(slot) * ulong(params.key_value_heads) * 2ul * ulong(r_stride);
      AFFINE4_UNROLL
      for (uint g = 0u; g < 2u; ++g) {
        threadgroup const bfloat4 *q4 =
            reinterpret_cast<threadgroup const bfloat4 *>(tg_vec + g * 128u);
        const half4 qh4 = half4(float4(q4[lane]));
        const ulong base_row = slot_base + (ulong(kvh) * 2ul + g) * r_stride + r;
        reinterpret_cast<device half4 *>(q_half + base_row * 128ul)[lane] = qh4;
        const float4 qf = float4(qh4);
        const float s = simd_sum((qf.x + qf.y) + (qf.z + qf.w));
        if (lane == 0u) {
          q_sum[base_row] = s;
        }
        if (r == 0u) {
          const uint active_rows = params.widths[slot] * repeat;
          const uint pad_end = min(r_stride, (active_rows + 15u) & ~15u);
          for (uint rr = active_rows; rr < pad_end; ++rr) {
            const ulong pad_row =
                slot_base + (ulong(kvh) * 2ul + g) * r_stride + rr;
            reinterpret_cast<device half4 *>(q_half + pad_row * 128ul)[lane] =
                half4(0.0h);
            if (lane == 0u) {
              q_sum[pad_row] = 0.0f;
            }
          }
        }
      }
    }
  } else {
    simdgroup_barrier(mem_flags::mem_threadgroup);
    constexpr uint kPageTokens = 64u;
    device const uint *page_table =
        select_pt8(slot, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7);
    const uint logical_page = logical_pos >> 6u;
    const uint within = logical_pos & 63u;
    const uint phys_page = page_table[logical_page];
    const ulong page_bytes =
        page_byte_stride(kPageTokens, params.key_value_heads, params.head_dim);
    const ulong head_code_bytes =
        ulong(kPageTokens) * ulong(params.head_dim / 2u);
    const ulong all_codes_bytes =
        ulong(params.key_value_heads) * head_code_bytes;

    device uchar *k_page = cache_key + ulong(phys_page) * page_bytes;
    device uchar *v_page = cache_value + ulong(phys_page) * page_bytes;
    device const bfloat *v_src = value +
                                 ulong(token) * params.value_token_stride +
                                 ulong(head) * params.head_dim;

    quantize_vec_g128_tg(
        tg_vec, k_page + ulong(head) * head_code_bytes,
        reinterpret_cast<device KvAffine4VecMeta *>(k_page + all_codes_bytes) +
            ulong(head) * kPageTokens,
        kPageTokens, within, params.head_dim, lane);
    quantize_vec_g128_device(
        v_src, v_page + ulong(head) * head_code_bytes,
        reinterpret_cast<device KvAffine4VecMeta *>(v_page + all_codes_bytes) +
            ulong(head) * kPageTokens,
        kPageTokens, within, params.head_dim, lane);
  }
}

} // namespace affine4_kv

// ============================================================================
// 1. Lean Affine4Group128 Paged KV Append (0 alloca, dead subband scan deleted)
// Launch: grid = {32, tokens * key_value_heads}, tptg = {32}
// ============================================================================
kernel void omlx_m5_paged_kv_append_affine4_g128(
    device const bfloat *key [[buffer(0)]],
    device const bfloat *value [[buffer(1)]],
    device uchar *cache_key [[buffer(2)]],
    device uchar *cache_value [[buffer(3)]],
    device const uint *page_table [[buffer(4)]],
    constant PagedAttentionParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint kvh = tg.y % params.key_value_heads;
  const uint token = tg.y / params.key_value_heads;
  if (token >= params.tokens ||
      (params.head_dim != 256u && params.head_dim != 128u)) {
    return;
  }

  const uint logical = params.cache_start + token;
  const uint logical_page = logical / params.page_tokens;
  const uint within = logical - logical_page * params.page_tokens;
  const uint phys_page = page_table[logical_page];

  const ulong page_bytes = affine4_kv::page_byte_stride(
      params.page_tokens, params.key_value_heads, params.head_dim);
  const ulong head_code_bytes =
      ulong(params.page_tokens) * ulong(params.head_dim / 2u);
  const ulong all_codes_bytes = ulong(params.key_value_heads) * head_code_bytes;

  device uchar *k_page = cache_key + ulong(phys_page) * page_bytes;
  device uchar *v_page = cache_value + ulong(phys_page) * page_bytes;

  device const bfloat *k_src = key + ulong(token) * params.key_token_stride +
                               ulong(kvh) * params.head_dim;
  device const bfloat *v_src = value + ulong(token) * params.value_token_stride +
                               ulong(kvh) * params.head_dim;

  affine4_kv::quantize_vec_g128_device(
      k_src, k_page + ulong(kvh) * head_code_bytes,
      reinterpret_cast<device KvAffine4VecMeta *>(k_page + all_codes_bytes) +
          ulong(kvh) * params.page_tokens,
      params.page_tokens, within, params.head_dim, lane);

  affine4_kv::quantize_vec_g128_device(
      v_src, v_page + ulong(kvh) * head_code_bytes,
      reinterpret_cast<device KvAffine4VecMeta *>(v_page + all_codes_bytes) +
          ulong(kvh) * params.page_tokens,
      params.page_tokens, within, params.head_dim, lane);
}

// Fused 0-ULP QK-Norm + RoPE + Lean Affine4Group128 KV-Append (page_tokens=64).
// Launch: grid = {32, tokens * (query_heads + key_value_heads)}, tptg = {32}
kernel void omlx_m5_attention_qk_norm_rope_kv_append_affine4_g128(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device uchar *cache_key [[buffer(5)]],
    device uchar *cache_value [[buffer(6)]],
    device const uint *page_table [[buffer(7)]],
    device const uint *positions [[buffer(8)]],
    constant AttentionPostprocessParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tg_vec[256];
  affine4_kv::qk_norm_rope_kv_append_affine4_impl<false>(
      query, key, value, query_norm, key_norm, cache_key, cache_value,
      page_table, positions, nullptr, nullptr, params, tg_vec, lane,
      tg_position);
}

// Fused T=1 HD=256 QK-Norm + RoPE + Affine4Group128 KV-Append + Q->half FOSCAR pack.
// Eliminates BOTH paged_kv_append AND pack_q dispatches + barriers on T=1 split decode!
kernel void omlx_m5_attention_qk_norm_rope_kv_append_pack_q_affine4_g128(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device uchar *cache_key [[buffer(5)]],
    device uchar *cache_value [[buffer(6)]],
    device const uint *page_table [[buffer(7)]],
    device const uint *positions [[buffer(8)]],
    device half *q_half [[buffer(9)]],
    device float *q_sum [[buffer(10)]],
    constant AttentionPostprocessParams &params [[buffer(11)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tg_vec[256];
  affine4_kv::qk_norm_rope_kv_append_affine4_impl<true>(
      query, key, value, query_norm, key_norm, cache_key, cache_value,
      page_table, positions, q_half, q_sum, params, tg_vec, lane, tg_position);
}

// Batched B=2..8 Fused 0-ULP QK-Norm + RoPE + Lean Affine4Group128 KV-Append.
// Launch: grid = {32, total_tokens * (query_heads + key_value_heads)}, tptg = {32}
kernel void omlx_m5_attention_qk_norm_rope_kv_append_batched_affine4_g128(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device uchar *cache_key [[buffer(5)]],
    device uchar *cache_value [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    device const uint *pt0 [[buffer(8)]],
    device const uint *pt1 [[buffer(9)]],
    device const uint *pt2 [[buffer(10)]],
    device const uint *pt3 [[buffer(11)]],
    device const uint *pt4 [[buffer(12)]],
    device const uint *pt5 [[buffer(13)]],
    device const uint *pt6 [[buffer(14)]],
    device const uint *pt7 [[buffer(15)]],
    constant BatchedAttentionPostprocessParams &params [[buffer(16)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tg_vec[256];
  affine4_kv::qk_norm_rope_kv_append_batched_affine4_impl<false>(
      query, key, value, query_norm, key_norm, cache_key, cache_value,
      positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, nullptr, nullptr,
      params, tg_vec, lane, tg_position);
}

// Batched B=2..8 Fused QK-Norm + RoPE + Affine4 KV-Append + Q->half FOSCAR pack.
// Eliminates BOTH paged_kv_append AND pack_q dispatches on B=2..8 AR & Verify!
kernel void omlx_m5_attention_qk_norm_rope_kv_append_pack_q_batched_affine4_g128(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *value [[buffer(2)]],
    device const bfloat *query_norm [[buffer(3)]],
    device const bfloat *key_norm [[buffer(4)]],
    device uchar *cache_key [[buffer(5)]],
    device uchar *cache_value [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    device const uint *pt0 [[buffer(8)]],
    device const uint *pt1 [[buffer(9)]],
    device const uint *pt2 [[buffer(10)]],
    device const uint *pt3 [[buffer(11)]],
    device const uint *pt4 [[buffer(12)]],
    device const uint *pt5 [[buffer(13)]],
    device const uint *pt6 [[buffer(14)]],
    device const uint *pt7 [[buffer(15)]],
    device half *q_half [[buffer(16)]],
    device float *q_sum [[buffer(17)]],
    constant BatchedAttentionPostprocessParams &params [[buffer(18)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup bfloat tg_vec[256];
  affine4_kv::qk_norm_rope_kv_append_batched_affine4_impl<true>(
      query, key, value, query_norm, key_norm, cache_key, cache_value,
      positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, q_half, q_sum, params,
      tg_vec, lane, tg_position);
}

// ============================================================================
// 2. Tiny Q -> half [KVH, 2, R_stride, 128] + FOSCAR q_sum pack kernel (< 5 us)
// Launch: grid = {32, KVH * R_stride}, tptg = {32}, R_stride = ceil(cap*rep/16)*16
// ============================================================================
kernel void omlx_m5_pack_q_half_foscar_hd256(
    device const bfloat *query [[buffer(0)]],
    device half *q_half [[buffer(1)]],
    device float *q_sum [[buffer(2)]],
    constant PagedSplitAttentionParams &p [[buffer(3)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  const uint r_stride = tgs_per_grid.y / p.key_value_heads;
  const uint kvh = tg.y / r_stride;
  const uint r = tg.y - kvh * r_stride;
  if (kvh >= p.key_value_heads) {
    return;
  }
  const uint repeat = p.query_heads / p.key_value_heads;
  const uint active_rows = p.tokens * repeat;

  if (r < active_rows) {
    const uint tok = r / repeat;
    const uint qh = kvh * repeat + (r - tok * repeat);
    device const bfloat *q_base = query + ulong(tok) * p.query_token_stride +
                                  ulong(qh) * p.query_head_stride;
    AFFINE4_UNROLL
    for (uint g = 0u; g < 2u; ++g) {
      device const bfloat4 *q4 =
          reinterpret_cast<device const bfloat4 *>(q_base + g * 128u);
      const half4 qh4 = half4(float4(q4[lane]));
      const ulong dst_row = (ulong(kvh) * 2ul + g) * ulong(r_stride) + r;
      reinterpret_cast<device half4 *>(q_half + dst_row * 128ul)[lane] = qh4;
      const float4 qf = float4(qh4);
      const float s = simd_sum((qf.x + qf.y) + (qf.z + qf.w));
      if (lane == 0u) {
        q_sum[dst_row] = s;
      }
    }
  } else {
    AFFINE4_UNROLL
    for (uint g = 0u; g < 2u; ++g) {
      const ulong dst_row = (ulong(kvh) * 2ul + g) * ulong(r_stride) + r;
      reinterpret_cast<device half4 *>(q_half + dst_row * 128ul)[lane] =
          half4(0.0h);
      if (lane == 0u) {
        q_sum[dst_row] = 0.0f;
      }
    }
  }
}

// ============================================================================
// 3. Unified 2-SG HW uint4b_format NAX Split-KV Entrypoints (HD == 256)
// Launch: grid = {splits * 64, KVH, n_rblks}, tptg = {64}
// ============================================================================
kernel void omlx_m5_paged_attention_affine4_g128_nax2sg_split_partial_r6_hd256(
    device const half *q_half [[buffer(0)]],
    device const float *q_sum [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device float *partial_max [[buffer(4)]],
    device float *partial_sum [[buffer(5)]],
    device half *partial_output_half [[buffer(6)]],
    device const uint *page_table [[buffer(7)]],
    device const uint *positions [[buffer(8)]],
    constant PagedSplitAttentionParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  constexpr uint kR = 6u;
  threadgroup float tg_scores[2 * kR * 64]; // 3.0 KiB
  threadgroup half tg_p[2 * 16 * 64];       // 4.0 KiB
  threadgroup float tg_q_sum[2 * kR];
  threadgroup uint tg_row_ctx_unused[1];
  threadgroup float tg_m[kR];
  threadgroup float tg_l[kR];
  threadgroup float tg_rs[kR];
  threadgroup float tg_vb[2 * kR];

  affine4_kv::a5_nax_2sg_impl<kR, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, params, tg_scores, tg_p,
      tg_q_sum, tg_row_ctx_unused, tg_m, tg_l, tg_rs, tg_vb, lane, sg, tg,
      tgs_per_grid);
}

kernel void omlx_m5_paged_attention_affine4_g128_nax2sg_split_partial_r8_hd256(
    device const half *q_half [[buffer(0)]],
    device const float *q_sum [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device float *partial_max [[buffer(4)]],
    device float *partial_sum [[buffer(5)]],
    device half *partial_output_half [[buffer(6)]],
    device const uint *page_table [[buffer(7)]],
    device const uint *positions [[buffer(8)]],
    constant PagedSplitAttentionParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  constexpr uint kR = 8u;
  threadgroup float tg_scores[2 * kR * 64]; // 4.0 KiB
  threadgroup half tg_p[2 * 16 * 64];       // 4.0 KiB
  threadgroup float tg_q_sum[2 * kR];
  threadgroup uint tg_row_ctx_unused[1];
  threadgroup float tg_m[kR];
  threadgroup float tg_l[kR];
  threadgroup float tg_rs[kR];
  threadgroup float tg_vb[2 * kR];

  affine4_kv::a5_nax_2sg_impl<kR, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, params, tg_scores, tg_p,
      tg_q_sum, tg_row_ctx_unused, tg_m, tg_l, tg_rs, tg_vb, lane, sg, tg,
      tgs_per_grid);
}

kernel void omlx_m5_paged_attention_affine4_g128_nax2sg_split_partial_r16_hd256(
    device const half *q_half [[buffer(0)]],
    device const float *q_sum [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device float *partial_max [[buffer(4)]],
    device float *partial_sum [[buffer(5)]],
    device half *partial_output_half [[buffer(6)]],
    device const uint *page_table [[buffer(7)]],
    device const uint *positions [[buffer(8)]],
    constant PagedSplitAttentionParams &params [[buffer(9)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]],
    uint3 tgs_per_grid [[threadgroups_per_grid]]) {
  constexpr uint kR = 16u;
  threadgroup float tg_scores[2 * kR * 64]; // 8.0 KiB
  threadgroup half tg_p[2 * 16 * 64];       // 4.0 KiB
  threadgroup float tg_q_sum[2 * kR];
  threadgroup uint tg_row_ctx[kR];
  threadgroup float tg_m[kR];
  threadgroup float tg_l[kR];
  threadgroup float tg_rs[kR];
  threadgroup float tg_vb[2 * kR];

  affine4_kv::a5_nax_2sg_impl<kR, true>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, params, tg_scores, tg_p,
      tg_q_sum, tg_row_ctx, tg_m, tg_l, tg_rs, tg_vb, lane, sg, tg,
      tgs_per_grid);
}

// ============================================================================
// 3b. Batched B=2..8 Unified 2-SG HW uint4b_format NAX Split-KV Entrypoints
// Launch: grid = {splits * 64, KVH * n_rblks, B}, tptg = {64}
// ============================================================================
kernel void
omlx_m5_paged_attention_affine4_g128_nax2sg_batched_split_partial_r6_hd256(
    device const half *q_half [[buffer(0)]],
    device const float *q_sum [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device float *partial_max [[buffer(4)]],
    device float *partial_sum [[buffer(5)]],
    device half *partial_output_half [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    device const uint *pt0 [[buffer(8)]],
    device const uint *pt1 [[buffer(9)]],
    device const uint *pt2 [[buffer(10)]],
    device const uint *pt3 [[buffer(11)]],
    device const uint *pt4 [[buffer(12)]],
    device const uint *pt5 [[buffer(13)]],
    device const uint *pt6 [[buffer(14)]],
    device const uint *pt7 [[buffer(15)]],
    constant BatchedPagedAttentionParams &params [[buffer(16)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kR = 6u;
  threadgroup float tg_scores[2 * kR * 64]; // 3.0 KiB
  threadgroup half tg_p[2 * 16 * 64];       // 4.0 KiB
  threadgroup float tg_q_sum[2 * kR];
  threadgroup uint tg_row_ctx_unused[1];
  threadgroup float tg_m[kR];
  threadgroup float tg_l[kR];
  threadgroup float tg_rs[kR];
  threadgroup float tg_vb[2 * kR];

  affine4_kv::a5_nax_2sg_batched_impl<kR, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7,
      params, tg_scores, tg_p, tg_q_sum, tg_row_ctx_unused, tg_m, tg_l, tg_rs,
      tg_vb, lane, sg, tg);
}

kernel void
omlx_m5_paged_attention_affine4_g128_nax2sg_batched_split_partial_r8_hd256(
    device const half *q_half [[buffer(0)]],
    device const float *q_sum [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device float *partial_max [[buffer(4)]],
    device float *partial_sum [[buffer(5)]],
    device half *partial_output_half [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    device const uint *pt0 [[buffer(8)]],
    device const uint *pt1 [[buffer(9)]],
    device const uint *pt2 [[buffer(10)]],
    device const uint *pt3 [[buffer(11)]],
    device const uint *pt4 [[buffer(12)]],
    device const uint *pt5 [[buffer(13)]],
    device const uint *pt6 [[buffer(14)]],
    device const uint *pt7 [[buffer(15)]],
    constant BatchedPagedAttentionParams &params [[buffer(16)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kR = 8u;
  threadgroup float tg_scores[2 * kR * 64]; // 4.0 KiB
  threadgroup half tg_p[2 * 16 * 64];       // 4.0 KiB
  threadgroup float tg_q_sum[2 * kR];
  threadgroup uint tg_row_ctx_unused[1];
  threadgroup float tg_m[kR];
  threadgroup float tg_l[kR];
  threadgroup float tg_rs[kR];
  threadgroup float tg_vb[2 * kR];

  affine4_kv::a5_nax_2sg_batched_impl<kR, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7,
      params, tg_scores, tg_p, tg_q_sum, tg_row_ctx_unused, tg_m, tg_l, tg_rs,
      tg_vb, lane, sg, tg);
}

kernel void
omlx_m5_paged_attention_affine4_g128_nax2sg_batched_split_partial_r16_hd256(
    device const half *q_half [[buffer(0)]],
    device const float *q_sum [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device float *partial_max [[buffer(4)]],
    device float *partial_sum [[buffer(5)]],
    device half *partial_output_half [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    device const uint *pt0 [[buffer(8)]],
    device const uint *pt1 [[buffer(9)]],
    device const uint *pt2 [[buffer(10)]],
    device const uint *pt3 [[buffer(11)]],
    device const uint *pt4 [[buffer(12)]],
    device const uint *pt5 [[buffer(13)]],
    device const uint *pt6 [[buffer(14)]],
    device const uint *pt7 [[buffer(15)]],
    constant BatchedPagedAttentionParams &params [[buffer(16)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kR = 16u;
  threadgroup float tg_scores[2 * kR * 64]; // 8.0 KiB
  threadgroup half tg_p[2 * 16 * 64];       // 4.0 KiB
  threadgroup float tg_q_sum[2 * kR];
  threadgroup uint tg_row_ctx[kR];
  threadgroup float tg_m[kR];
  threadgroup float tg_l[kR];
  threadgroup float tg_rs[kR];
  threadgroup float tg_vb[2 * kR];

  affine4_kv::a5_nax_2sg_batched_impl<kR, true>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7,
      params, tg_scores, tg_p, tg_q_sum, tg_row_ctx, tg_m, tg_l, tg_rs, tg_vb,
      lane, sg, tg);
}

// ============================================================================
// 4. 4-SG (128-thread) Parallel Log2 Half Split-KV Merge + Qwen Sigmoid Gate
// Launch: grid = {128, tokens * query_heads}, tptg = {128}
// Trailing-empty-split invariant -> 0-ULP across split_count = 1..256!
// ============================================================================
kernel void omlx_m5_paged_attention_affine4_g128_split_merge_half_4sg_hd256(
    device const bfloat *gate [[buffer(0)]],
    device const float *partial_max [[buffer(1)]],
    device const float *partial_sum [[buffer(2)]],
    device const half *partial_output_half [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant PagedSplitAttentionParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens) {
    return;
  }
  const uint query_head = logical_head - token * params.query_heads;
  const uint tid = sg * 32u + lane;
  const uint splits = params.split_count;
  const ulong base_record = ulong(logical_head) * ulong(splits);

  threadgroup float sg_m[4];
  threadgroup float sg_l[4];
  threadgroup float tg_w[256];
  threadgroup float tg_o[4 * 256]; // 4 KiB

  // Phase 1: Parallel global_m reduction across 128 threads.
  float local_m = affine4_kv::kNegInf;
  for (uint s = tid; s < splits; s += 128u) {
    const float m_s = partial_max[base_record + s];
    if (m_s > affine4_kv::kNegInfHalf) {
      local_m = max(local_m, m_s);
    }
  }
  const float warp_m = simd_max(local_m);
  if (lane == 0u) {
    sg_m[sg] = warp_m;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float global_m =
      max(max(sg_m[0], sg_m[1]), max(sg_m[2], sg_m[3]));

  // Phase 2: Parallel log2 weight & global_l reduction.
  float local_l = 0.0f;
  for (uint s = tid; s < splits; s += 128u) {
    const float m_s = partial_max[base_record + s];
    float w_s = 0.0f;
    if (m_s > affine4_kv::kNegInfHalf) {
      w_s = partial_sum[base_record + s] * fast::exp2(m_s - global_m);
      local_l += w_s;
    }
    if (s < 256u) {
      tg_w[s] = w_s;
    }
  }
  const float warp_l = simd_sum(local_l);
  if (lane == 0u) {
    sg_l[sg] = warp_l;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const float global_l = (sg_l[0] + sg_l[1]) + (sg_l[2] + sg_l[3]);
  const float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;

  // Phase 3: 4 SGs stripe across splits (s = sg, sg+4, ...), each lane holds
  // 8 contiguous halfs (2 x half4 = 16 B) -> full 512 B coalesced bursts!
  float4 acc0(0.0f);
  float4 acc1(0.0f);
  const uint d0 = lane * 8u;
  for (uint s = sg; s < splits; s += 4u) {
    float w_s;
    if (s < 256u) {
      w_s = tg_w[s];
    } else {
      const float m_s = partial_max[base_record + s];
      w_s = (m_s > affine4_kv::kNegInfHalf)
                ? (partial_sum[base_record + s] * fast::exp2(m_s - global_m))
                : 0.0f;
    }
    if (w_s > 0.0f) {
      const float nw = w_s * inv_l;
      device const half4 *part4 = reinterpret_cast<device const half4 *>(
          partial_output_half + (base_record + s) * 256ul + d0);
      acc0 = fma(float4(part4[0]), float4(nw), acc0);
      acc1 = fma(float4(part4[1]), float4(nw), acc1);
    }
  }

  reinterpret_cast<threadgroup float4 *>(tg_o + sg * 256u + d0)[0] = acc0;
  reinterpret_cast<threadgroup float4 *>(tg_o + sg * 256u + d0)[1] = acc1;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Phase 4: All 128 threads sum 4 SGs across 256 dims (2 dims/thread) + gate.
  device const bfloat *gate_row =
      gate + ulong(token) * params.gate_token_stride +
      ulong(query_head) * params.gate_head_stride;
  device bfloat *out_row = output + ulong(logical_head) * 256ul;

  AFFINE4_UNROLL
  for (uint k = 0u; k < 2u; ++k) {
    const uint d = tid + k * 128u;
    const float val = (tg_o[0u * 256u + d] + tg_o[1u * 256u + d]) +
                      (tg_o[2u * 256u + d] + tg_o[3u * 256u + d]);
    const float g_val = float(gate_row[d]);
    out_row[d] = bfloat(val / (1.0f + exp(-g_val)));
  }
}

// ============================================================================
// 5. Register-Resident Frag16x16 Affine4Group128 Flash-Prefill (HD == 256)
//    (c5_aff4_frag64_bfp: 61.0 TF, max_err = 0.001953, ZERO tg_kv/tg_p/alloca)
// Grid (threads): width = 256 (WM=4 * WN=2 * 32), height = ceil(T/64) * QH
// Threads per threadgroup: {256, 1, 1}
// ============================================================================
AFFINE4_ALWAYS_INLINE
static inline void paged_flash_prefill_affine4_hd256_core(
    device const bfloat *query,
    device const bfloat *gate,
    device const uchar *cache_key,
    device const uchar *cache_value,
    device bfloat *output,
    device const uint *page_table,
    device const uint *positions,
    uint query_heads,
    uint key_value_heads,
    uint tokens,
    uint page_tokens,
    uint query_token_stride,
    uint query_head_stride,
    uint gate_token_stride,
    uint gate_head_stride,
    threadgroup float *xchg,
    threadgroup uint *tg_q_pos,
    ushort lane,
    ushort sg,
    uint tg_y) {
  constexpr uint BD = 256u;
  constexpr uint BQ = 64u;
  constexpr uint BK = 32u;

  if (query_heads == 0u) {
    return;
  }
  const uint q_blk = tg_y / query_heads;
  const uint qh = tg_y - q_blk * query_heads;
  const uint q_base = q_blk * BQ;
  if (q_base >= tokens) {
    return;
  }

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  if (flat_tid < BQ) {
    const uint q_tok = q_base + flat_tid;
    tg_q_pos[flat_tid] = (q_tok < tokens) ? positions[q_tok] : 0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint active_q_rows = min(BQ, tokens - q_base);
  uint max_valid_kv = 0u;
  uint min_q_pos = 0xffffffffu;
  for (uint r = 0u; r < active_q_rows; ++r) {
    const uint pos = tg_q_pos[r];
    max_valid_kv = max(max_valid_kv, pos + 1u);
    min_q_pos = min(min_q_pos, pos);
  }
  const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
  const uint kb_min_causal = min_q_pos / BK;

  const uint repeat = query_heads / key_value_heads;
  const uint kvh = qh / repeat;
  const uint half_blocks_per_page = page_tokens / BK;
  const ulong page_bytes =
      affine4_kv::page_byte_stride(page_tokens, key_value_heads, BD);
  const ulong head_code_bytes = ulong(page_tokens) * 128ul;
  const ulong all_codes_bytes = ulong(key_value_heads) * head_code_bytes;

  const short row_group = short(sg >> 1u); // 0..3
  const short d_part = short(sg & 1u);     // 0..1 (= exact affine4 group g!)
  const short tm = 16 * row_group;
  const short lim_rows_q = short(int(active_q_rows) - int(tm));
  const uint d_offset = uint(d_part) * 128u;
  const ulong group_code_offset =
      ulong(kvh) * head_code_bytes + ulong(d_part) * (64ul * 64ul);

  const short2 sc = affine4_kv::Frag16x16Aff4::get_coord(lane);
  const short sm = sc.y;
  const short sn = sc.x;
  const uint sn_u16 = uint(sn >> 2u); // 0..3

  const uint r_q0 = uint(tm + sm);
  const uint r_q1 = uint(tm + 8 + sm);
  const bool qv0 = (r_q0 < active_q_rows);
  const bool qv1 = (r_q1 < active_q_rows);
  const uint qpos0 = qv0 ? tg_q_pos[r_q0] : 0u;
  const uint qpos1 = qv1 ? tg_q_pos[r_q1] : 0u;

  using H8 = affine4_kv::Frag16x16Aff4::frag_t<half>;
  using BF8 = affine4_kv::Frag16x16Aff4::frag_t<bfloat>;
  using F8 = affine4_kv::Frag16x16Aff4::frag_t<float>;

  // Explicit SSA half register fragments (0 alloca + 0 half->bfloat dequant convert!).
  H8 Q0(0.0h), Q1(0.0h), Q2(0.0h), Q3(0.0h);
  H8 Q4(0.0h), Q5(0.0h), Q6(0.0h), Q7(0.0h);
  device const bfloat *Q_ptr =
      query + ulong(q_base + uint(tm)) * query_token_stride +
      ulong(qh) * query_head_stride + d_offset;
  if (lim_rows_q >= 16) {
    affine4_kv::Frag16x16Aff4::load_h4(Q0, Q_ptr, query_token_stride, 0, sc);
    affine4_kv::Frag16x16Aff4::load_h4(Q1, Q_ptr, query_token_stride, 16, sc);
    affine4_kv::Frag16x16Aff4::load_h4(Q2, Q_ptr, query_token_stride, 32, sc);
    affine4_kv::Frag16x16Aff4::load_h4(Q3, Q_ptr, query_token_stride, 48, sc);
    affine4_kv::Frag16x16Aff4::load_h4(Q4, Q_ptr, query_token_stride, 64, sc);
    affine4_kv::Frag16x16Aff4::load_h4(Q5, Q_ptr, query_token_stride, 80, sc);
    affine4_kv::Frag16x16Aff4::load_h4(Q6, Q_ptr, query_token_stride, 96, sc);
    affine4_kv::Frag16x16Aff4::load_h4(Q7, Q_ptr, query_token_stride, 112, sc);
  } else if (lim_rows_q > 0) {
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q0, Q_ptr, query_token_stride, lim_rows_q, 0, sc);
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q1, Q_ptr, query_token_stride, lim_rows_q, 16, sc);
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q2, Q_ptr, query_token_stride, lim_rows_q, 32, sc);
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q3, Q_ptr, query_token_stride, lim_rows_q, 48, sc);
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q4, Q_ptr, query_token_stride, lim_rows_q, 64, sc);
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q5, Q_ptr, query_token_stride, lim_rows_q, 80, sc);
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q6, Q_ptr, query_token_stride, lim_rows_q, 96, sc);
    affine4_kv::Frag16x16Aff4::load_rows_clean_h4(Q7, Q_ptr, query_token_stride, lim_rows_q, 112, sc);
  }

  F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
  F8 O4(0.0f), O5(0.0f), O6(0.0f), O7(0.0f);

  float max_s0 = affine4_kv::kNegInf;
  float max_s1 = affine4_kv::kNegInf;
  float sum_s0 = 0.0f;
  float sum_s1 = 0.0f;
  const float scale2 = 0.0625f * affine4_kv::kLog2e;

  threadgroup float4 *xchg_slot = reinterpret_cast<threadgroup float4 *>(
      xchg + row_group * (2 * 8 * 32) + short(lane) * 16);

  for (uint kb = 0u; kb < nk_blocks; ++kb) {
    const uint logical_page = kb / half_blocks_per_page;
    const uint sub_offset = (kb % half_blocks_per_page) * BK;
    const uint phys_page = page_table[logical_page];
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));

    device const uchar *k_page_ptr =
        cache_key + ulong(phys_page) * page_bytes;
    device const ushort *k_codes_u16 =
        reinterpret_cast<device const ushort *>(k_page_ptr + group_code_offset);
    device const KvAffine4VecMeta *k_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(k_page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;

    F8 S0(0.0f);
    F8 S1(0.0f);
    {
      const uint rk00 = sub_offset + uint(sm);
      const uint rk01 = rk00 + 8u;
      const uint rk10 = rk00 + 16u;
      const uint rk11 = rk00 + 24u;
      half sk00 = k_meta[rk00].scale[d_part];
      half zk00 = k_meta[rk00].zero[d_part];
      half sk01 = k_meta[rk01].scale[d_part];
      half zk01 = k_meta[rk01].zero[d_part];
      half sk10 = k_meta[rk10].scale[d_part];
      half zk10 = k_meta[rk10].zero[d_part];
      half sk11 = k_meta[rk11].scale[d_part];
      half zk11 = k_meta[rk11].zero[d_part];
      if (valid_k < 32) {
        if (!(sm < valid_k)) { sk00 = half(0.0h); zk00 = half(0.0h); }
        if (!(short(sm + 8) < valid_k)) { sk01 = half(0.0h); zk01 = half(0.0h); }
        if (!(short(sm + 16) < valid_k)) { sk10 = half(0.0h); zk10 = half(0.0h); }
        if (!(short(sm + 24) < valid_k)) { sk11 = half(0.0h); zk11 = half(0.0h); }
      }
      device const ushort *kp00 = k_codes_u16 + rk00 * 32u + sn_u16;
      device const ushort *kp01 = k_codes_u16 + rk01 * 32u + sn_u16;
      device const ushort *kp10 = k_codes_u16 + rk10 * 32u + sn_u16;
      device const ushort *kp11 = k_codes_u16 + rk11 * 32u + sn_u16;

#define AFF4_QK_STEP(QREG, D_IDX)                                              \
  do {                                                                         \
    const uint c_off = uint(D_IDX) * 4u;                                       \
    const H8 K0 = affine4_kv::Frag16x16Aff4::dequant_ptr_h(                    \
        kp00, kp01, sk00, zk00, sk01, zk01, c_off);                            \
    const H8 K1 = affine4_kv::Frag16x16Aff4::dequant_ptr_h(                    \
        kp10, kp11, sk10, zk10, sk11, zk11, c_off);                            \
    affine4_kv::Frag16x16Aff4::mma_1x2_f16<true>(S0, S1, QREG, K0, K1);        \
  } while (false)

      AFF4_QK_STEP(Q0, 0);
      AFF4_QK_STEP(Q1, 1);
      AFF4_QK_STEP(Q2, 2);
      AFF4_QK_STEP(Q3, 3);
      AFF4_QK_STEP(Q4, 4);
      AFF4_QK_STEP(Q5, 5);
      AFF4_QK_STEP(Q6, 6);
      AFF4_QK_STEP(Q7, 7);
#undef AFF4_QK_STEP
    }

    // WN=2 head-dim partial score exchange via 8 KiB xchg (8.4 KiB -> 2 TGs/core!).
    if (d_part == 1) {
      xchg_slot[0] = float4(S0[0], S0[1], S0[2], S0[3]);
      xchg_slot[1] = float4(S0[4], S0[5], S0[6], S0[7]);
      xchg_slot[2] = float4(S1[0], S1[1], S1[2], S1[3]);
      xchg_slot[3] = float4(S1[4], S1[5], S1[6], S1[7]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d_part == 0) {
      const float4 p0 = xchg_slot[0];
      const float4 p1 = xchg_slot[1];
      const float4 p2 = xchg_slot[2];
      const float4 p3 = xchg_slot[3];
      S0 += F8(p0.x, p0.y, p0.z, p0.w, p1.x, p1.y, p1.z, p1.w);
      S1 += F8(p2.x, p2.y, p2.z, p2.w, p3.x, p3.y, p3.z, p3.w);
      xchg_slot[0] = float4(S0[0], S0[1], S0[2], S0[3]);
      xchg_slot[1] = float4(S0[4], S0[5], S0[6], S0[7]);
      xchg_slot[2] = float4(S1[0], S1[1], S1[2], S1[3]);
      xchg_slot[3] = float4(S1[4], S1[5], S1[6], S1[7]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (d_part == 1) {
      const float4 s00 = xchg_slot[0];
      const float4 s01 = xchg_slot[1];
      const float4 s10 = xchg_slot[2];
      const float4 s11 = xchg_slot[3];
      S0 = F8(s00.x, s00.y, s00.z, s00.w, s01.x, s01.y, s01.z, s01.w);
      S1 = F8(s10.x, s10.y, s10.z, s10.w, s11.x, s11.y, s11.z, s11.w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Hoisted causal-free fast path (99% of tiles in long-context prefill).
    const bool full_causal_free =
        (kb < kb_min_causal) && (valid_k == 32) && (lim_rows_q >= 16);
    if (full_causal_free) {
      S0 *= scale2;
      S1 *= scale2;
    } else {
      const uint4 c0_abs =
          uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
      const uint4 c1_abs = c0_abs + uint4(16u);
      const bool4 ok00 = qv0 && (c0_abs < max_valid_kv) && (c0_abs <= qpos0);
      const bool4 ok01 = qv1 && (c0_abs < max_valid_kv) && (c0_abs <= qpos1);
      const bool4 ok10 = qv0 && (c1_abs < max_valid_kv) && (c1_abs <= qpos0);
      const bool4 ok11 = qv1 && (c1_abs < max_valid_kv) && (c1_abs <= qpos1);
      const float4 s00 =
          select(float4(affine4_kv::kNegInf), float4(S0[0], S0[1], S0[2], S0[3]) * scale2, ok00);
      const float4 s01 =
          select(float4(affine4_kv::kNegInf), float4(S0[4], S0[5], S0[6], S0[7]) * scale2, ok01);
      const float4 s10 =
          select(float4(affine4_kv::kNegInf), float4(S1[0], S1[1], S1[2], S1[3]) * scale2, ok10);
      const float4 s11 =
          select(float4(affine4_kv::kNegInf), float4(S1[4], S1[5], S1[6], S1[7]) * scale2, ok11);
      S0 = F8(s00.x, s00.y, s00.z, s00.w, s01.x, s01.y, s01.z, s01.w);
      S1 = F8(s10.x, s10.y, s10.z, s10.w, s11.x, s11.y, s11.z, s11.w);
    }

    // Online softmax row-max across the 32 columns of S0, S1.
    float m0 = max(max(max(S0[0], S0[1]), max(S0[2], S0[3])),
                   max(max(S1[0], S1[1]), max(S1[2], S1[3])));
    m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));
    m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));
    const float new_m0 = max(max_s0, m0);

    float m1 = max(max(max(S0[4], S0[5]), max(S0[6], S0[7])),
                   max(max(S1[4], S1[5]), max(S1[6], S1[7])));
    m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));
    m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));
    const float new_m1 = max(max_s1, m1);

    const float fac0 = (max_s0 <= affine4_kv::kNegInfHalf)
                           ? 0.0f
                           : fast::exp2(max_s0 - new_m0);
    const float fac1 = (max_s1 <= affine4_kv::kNegInfHalf)
                           ? 0.0f
                           : fast::exp2(max_s1 - new_m1);
    max_s0 = new_m0;
    max_s1 = new_m1;
    sum_s0 *= fac0;
    sum_s1 *= fac1;

    // Compute P = exp2(S - new_max), update sum, and convert P -> half for
    // full-rate half*half NAX P@V (0 half->bfloat dequant overhead!).
    float4 p00, p01, p10, p11;
    if (full_causal_free) {
      p00 = fast::exp2(float4(S0[0], S0[1], S0[2], S0[3]) - new_m0);
      p01 = fast::exp2(float4(S0[4], S0[5], S0[6], S0[7]) - new_m1);
      p10 = fast::exp2(float4(S1[0], S1[1], S1[2], S1[3]) - new_m0);
      p11 = fast::exp2(float4(S1[4], S1[5], S1[6], S1[7]) - new_m1);
    } else {
      const bool emp0 = (new_m0 <= affine4_kv::kNegInfHalf);
      const bool emp1 = (new_m1 <= affine4_kv::kNegInfHalf);
      const float4 s00(S0[0], S0[1], S0[2], S0[3]);
      const float4 s01(S0[4], S0[5], S0[6], S0[7]);
      const float4 s10(S1[0], S1[1], S1[2], S1[3]);
      const float4 s11(S1[4], S1[5], S1[6], S1[7]);
      p00 = select(fast::exp2(s00 - new_m0), float4(0.0f),
                   emp0 || (s00 <= affine4_kv::kNegInfHalf));
      p01 = select(fast::exp2(s01 - new_m1), float4(0.0f),
                   emp1 || (s01 <= affine4_kv::kNegInfHalf));
      p10 = select(fast::exp2(s10 - new_m0), float4(0.0f),
                   emp0 || (s10 <= affine4_kv::kNegInfHalf));
      p11 = select(fast::exp2(s11 - new_m1), float4(0.0f),
                   emp1 || (s11 <= affine4_kv::kNegInfHalf));
    }

    const float4 ps0 = p00 + p10;
    float tsum0 = (ps0.x + ps0.y) + (ps0.z + ps0.w);
    tsum0 += simd_shuffle_xor(tsum0, ushort(1));
    tsum0 += simd_shuffle_xor(tsum0, ushort(8));
    sum_s0 += tsum0;

    const float4 ps1 = p01 + p11;
    float tsum1 = (ps1.x + ps1.y) + (ps1.z + ps1.w);
    tsum1 += simd_shuffle_xor(tsum1, ushort(1));
    tsum1 += simd_shuffle_xor(tsum1, ushort(8));
    sum_s1 += tsum1;

    const half4 hp00 = half4(p00);
    const half4 hp01 = half4(p01);
    const half4 hp10 = half4(p10);
    const half4 hp11 = half4(p11);
    const H8 P0(hp00.x, hp00.y, hp00.z, hp00.w, hp01.x, hp01.y, hp01.z, hp01.w);
    const H8 P1(hp10.x, hp10.y, hp10.z, hp10.w, hp11.x, hp11.y, hp11.z, hp11.w);

    // Rescale O0..O7 in registers!
    const F8 fac_vec(fac0, fac0, fac0, fac0, fac1, fac1, fac1, fac1);
    O0 *= fac_vec;
    O1 *= fac_vec;
    O2 *= fac_vec;
    O3 *= fac_vec;
    O4 *= fac_vec;
    O5 *= fac_vec;
    O6 *= fac_vec;
    O7 *= fac_vec;

    device const uchar *v_page_ptr =
        cache_value + ulong(phys_page) * page_bytes;
    device const ushort *v_codes_u16 =
        reinterpret_cast<device const ushort *>(v_page_ptr + group_code_offset);
    device const KvAffine4VecMeta *v_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(v_page_ptr +
                                                          all_codes_bytes) +
        ulong(kvh) * page_tokens;

    const uint rv00 = sub_offset + uint(sm);
    const uint rv01 = rv00 + 8u;
    const uint rv10 = rv00 + 16u;
    const uint rv11 = rv00 + 24u;
    half sv00 = v_meta[rv00].scale[d_part];
    half zv00 = v_meta[rv00].zero[d_part];
    half sv01 = v_meta[rv01].scale[d_part];
    half zv01 = v_meta[rv01].zero[d_part];
    half sv10 = v_meta[rv10].scale[d_part];
    half zv10 = v_meta[rv10].zero[d_part];
    half sv11 = v_meta[rv11].scale[d_part];
    half zv11 = v_meta[rv11].zero[d_part];
    if (valid_k < 32) {
      if (!(sm < valid_k)) { sv00 = half(0.0h); zv00 = half(0.0h); }
      if (!(short(sm + 8) < valid_k)) { sv01 = half(0.0h); zv01 = half(0.0h); }
      if (!(short(sm + 16) < valid_k)) { sv10 = half(0.0h); zv10 = half(0.0h); }
      if (!(short(sm + 24) < valid_k)) { sv11 = half(0.0h); zv11 = half(0.0h); }
    }
    device const ushort *vp00 = v_codes_u16 + rv00 * 32u + sn_u16;
    device const ushort *vp01 = v_codes_u16 + rv01 * 32u + sn_u16;
    device const ushort *vp10 = v_codes_u16 + rv10 * 32u + sn_u16;
    device const ushort *vp11 = v_codes_u16 + rv11 * 32u + sn_u16;

    simdgroup_barrier(mem_flags::mem_none);

#define AFF4_PV_PAIR(OA, OB, D_IDX)                                            \
  do {                                                                         \
    const uint c0 = uint(D_IDX) * 4u;                                          \
    const uint c1 = uint((D_IDX) + 1) * 4u;                                    \
    {                                                                          \
      const H8 V0 = affine4_kv::Frag16x16Aff4::dequant_ptr_h(                  \
          vp00, vp01, sv00, zv00, sv01, zv01, c0);                             \
      const H8 V1 = affine4_kv::Frag16x16Aff4::dequant_ptr_h(                  \
          vp00, vp01, sv00, zv00, sv01, zv01, c1);                             \
      affine4_kv::Frag16x16Aff4::mma_1x2_f16<false>(OA, OB, P0, V0, V1);       \
    }                                                                          \
    {                                                                          \
      const H8 V0 = affine4_kv::Frag16x16Aff4::dequant_ptr_h(                  \
          vp10, vp11, sv10, zv10, sv11, zv11, c0);                             \
      const H8 V1 = affine4_kv::Frag16x16Aff4::dequant_ptr_h(                  \
          vp10, vp11, sv10, zv10, sv11, zv11, c1);                             \
      affine4_kv::Frag16x16Aff4::mma_1x2_f16<false>(OA, OB, P1, V0, V1);       \
    }                                                                          \
  } while (false)

    AFF4_PV_PAIR(O0, O1, 0);
    AFF4_PV_PAIR(O2, O3, 2);
    AFF4_PV_PAIR(O4, O5, 4);
    AFF4_PV_PAIR(O6, O7, 6);
#undef AFF4_PV_PAIR
  }

  if (lim_rows_q <= 0) {
    return;
  }

  const float rcp0 = (sum_s0 > 0.0f) ? (1.0f / sum_s0) : 0.0f;
  const float rcp1 = (sum_s1 > 0.0f) ? (1.0f / sum_s1) : 0.0f;

  device const bfloat *G_ptr =
      gate + ulong(q_base + uint(tm)) * gate_token_stride +
      ulong(qh) * gate_head_stride + d_offset;
  const ulong out_stride = ulong(query_heads) * BD;
  device bfloat *O_ptr =
      output + (ulong(q_base + uint(tm)) * query_heads + qh) * BD + d_offset;

#define AFF4_STORE_O(OREG, OFF_C)                                              \
  do {                                                                         \
    BF8 Gfrag;                                                                 \
    if (lim_rows_q < 16) {                                                     \
      affine4_kv::Frag16x16Aff4::load_rows_clean_bf4(                          \
          Gfrag, G_ptr, gate_token_stride, lim_rows_q, OFF_C, sc);             \
    } else {                                                                   \
      affine4_kv::Frag16x16Aff4::load_bf4(Gfrag, G_ptr, gate_token_stride,     \
                                          OFF_C, sc);                          \
    }                                                                          \
    const float4 g0 = float4(float(Gfrag[0]), float(Gfrag[1]), float(Gfrag[2]), \
                             float(Gfrag[3]));                                  \
    const float4 g1 = float4(float(Gfrag[4]), float(Gfrag[5]), float(Gfrag[6]), \
                             float(Gfrag[7]));                                  \
    const float4 o0 = (float4(OREG[0], OREG[1], OREG[2], OREG[3]) * rcp0) /    \
                      (float4(1.0f) + exp(-g0));                               \
    const float4 o1 = (float4(OREG[4], OREG[5], OREG[6], OREG[7]) * rcp1) /    \
                      (float4(1.0f) + exp(-g1));                               \
    OREG = F8(o0.x, o0.y, o0.z, o0.w, o1.x, o1.y, o1.z, o1.w);                 \
    if (lim_rows_q < 16) {                                                     \
      affine4_kv::Frag16x16Aff4::store_rows_bf4(OREG, O_ptr, out_stride,       \
                                                lim_rows_q, OFF_C, sc);        \
    } else {                                                                   \
      affine4_kv::Frag16x16Aff4::store_bf4(OREG, O_ptr, out_stride, OFF_C,     \
                                           sc);                                \
    }                                                                          \
  } while (false)

  AFF4_STORE_O(O0, 0);
  AFF4_STORE_O(O1, 16);
  AFF4_STORE_O(O2, 32);
  AFF4_STORE_O(O3, 48);
  AFF4_STORE_O(O4, 64);
  AFF4_STORE_O(O5, 80);
  AFF4_STORE_O(O6, 96);
  AFF4_STORE_O(O7, 112);
#undef AFF4_STORE_O
}

kernel void omlx_m5_paged_flash_prefill_affine4_g128_hd256(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &p [[buffer(7)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  // 8 KiB xchg (static_tg = 8.4 KiB -> guaranteed 2 TGs/core on Apple10!).
  threadgroup float xchg[4 * 16 * 32];
  threadgroup uint tg_q_pos[64];
  paged_flash_prefill_affine4_hd256_core(
      query, gate, cache_key, cache_value, output, page_table, positions,
      p.query_heads, p.key_value_heads, p.tokens, p.page_tokens,
      p.query_token_stride, p.query_head_stride, p.gate_token_stride,
      p.gate_head_stride, xchg, tg_q_pos, lane, sg, tg.y);
}

// Batched B=2..8 Register-Resident Frag16x16 Affine4Group128 Flash-Prefill (HD == 256)
// Grid (threads): width = 256, height = ceil(max_w/64) * QH, depth = B
kernel void omlx_m5_paged_flash_prefill_batched_affine4_g128_hd256(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *positions [[buffer(5)]],
    device const uint *pt0 [[buffer(6)]],
    device const uint *pt1 [[buffer(7)]],
    device const uint *pt2 [[buffer(8)]],
    device const uint *pt3 [[buffer(9)]],
    device const uint *pt4 [[buffer(10)]],
    device const uint *pt5 [[buffer(11)]],
    device const uint *pt6 [[buffer(12)]],
    device const uint *pt7 [[buffer(13)]],
    constant BatchedPagedAttentionParams &p [[buffer(14)]],
    ushort lane [[thread_index_in_simdgroup]],
    ushort sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint slot = tg.z;
  if (slot >= p.batch_size) {
    return;
  }
  const uint slot_tokens = p.widths[slot];
  if (slot_tokens == 0u) {
    return;
  }
  const uint row_off = p.row_offsets[slot];
  device const uint *slot_pt =
      affine4_kv::select_pt8(slot, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7);
  threadgroup float xchg[4 * 16 * 32];
  threadgroup uint tg_q_pos[64];
  paged_flash_prefill_affine4_hd256_core(
      query + ulong(row_off) * p.query_token_stride,
      gate + ulong(row_off) * p.gate_token_stride,
      cache_key, cache_value,
      output + ulong(row_off) * p.query_heads * 256ul,
      slot_pt, positions + row_off,
      p.query_heads, p.key_value_heads, slot_tokens, p.page_tokens,
      p.query_token_stride, p.query_head_stride, p.gate_token_stride,
      p.gate_head_stride, xchg, tg_q_pos, lane, sg, tg.y);
}

// ============================================================================
// 6. Preserved Non-Split (!split) and HD == 128 Compatibility Paths
// ============================================================================
template <uint HeadDim, uint Groups>
inline void causal_attention_affine4_single_impl(
    device const bfloat *query,
    device const bfloat *gate,
    device const uchar *cache_key,
    device const uchar *cache_value,
    device bfloat *output,
    device const uint *page_table,
    device const uint *positions,
    constant PagedAttentionParams &p,
    threadgroup half *tg_q,
    threadgroup float *tg_q_sum,
    threadgroup uint *tg_row_ctx,
    threadgroup float *tg_scores,
    threadgroup half *tg_p,
    threadgroup float *tg_rescale,
    threadgroup float *tg_v_scale,
    threadgroup float *tg_v_bias,
    threadgroup float *tg_m_blk,
    threadgroup float *tg_l_blk,
    threadgroup float *tg_m_run,
    threadgroup float *tg_l_run,
    threadgroup float *tg_w_old,
    threadgroup float *tg_w_new,
    uint lane,
    uint sg,
    uint3 tg) {
  const uint repeat = p.query_heads / p.key_value_heads;
  if (repeat == 0u || repeat > 16u) {
    return;
  }
  const uint bq = 16u / repeat;
  const uint kvh = tg.y % p.key_value_heads;
  const uint t_base = (tg.y / p.key_value_heads) * bq;
  if (t_base >= p.tokens) {
    return;
  }

  uint active_rows = 0u;
  uint max_ctx = 0u;
  affine4_kv::load_q_and_foscar_sums<HeadDim, Groups>(
      query, positions, p.tokens, p.query_heads, p.key_value_heads,
      p.query_token_stride, p.query_head_stride, t_base, bq, kvh, tg_q,
      tg_q_sum, tg_row_ctx, active_rows, max_ctx, lane, sg);

  const uint flat_tid = sg * 32u + lane;
  if (flat_tid < 16u) {
    tg_m_run[flat_tid] = affine4_kv::kNegInf;
    tg_l_run[flat_tid] = 0.0f;
  }
  float acc_run[64];
  for (uint i = 0u; i < 64u; ++i) {
    acc_run[i] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  constexpr auto desc_pv = matmul2d_descriptor(
      16, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc_pv, execution_simdgroup> op_pv;
  tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp_dummy(
      tg_p, dextents<int32_t, 2>(64, 16));
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tv_dummy(
      (device uchar *)cache_value, dextents<int32_t, 2>(128, 64));
  auto pv_layout = op_pv.template get_destination_cooperative_tensor<
      decltype(tp_dummy), decltype(tv_dummy), float>();
  const uint pv_elems = pv_layout.get_capacity();

  const uint total_pages = (max_ctx + p.page_tokens - 1u) / p.page_tokens;
  constexpr uint kPagesPerSuperblock = 8u;
  for (uint sb_page = 0u; sb_page < total_pages; sb_page += kPagesPerSuperblock) {
    float acc_blk[64];
    affine4_kv::run_superblock_nax_g128<HeadDim, Groups>(
        cache_key, cache_value, page_table, p.page_tokens, p.key_value_heads,
        kvh, sb_page, min(total_pages, sb_page + kPagesPerSuperblock), max_ctx,
        active_rows, tg_row_ctx, tg_q_sum, tg_q, tg_scores, tg_p, tg_rescale,
        tg_v_scale, tg_v_bias, tg_m_blk, tg_l_blk, acc_blk, lane, sg);

    if (flat_tid < 16u) {
      const float m_s = tg_m_blk[flat_tid];
      if (m_s > affine4_kv::kNegInfHalf) {
        const float m_prev = tg_m_run[flat_tid];
        const float new_m = max(m_prev, m_s);
        const float w_old =
            (m_prev <= affine4_kv::kNegInfHalf) ? 0.0f : exp(m_prev - new_m);
        const float w_new = exp(m_s - new_m);
        tg_m_run[flat_tid] = new_m;
        tg_l_run[flat_tid] =
            fma(tg_l_run[flat_tid], w_old, tg_l_blk[flat_tid] * w_new);
        tg_w_old[flat_tid] = w_old;
        tg_w_new[flat_tid] = w_new;
      } else {
        tg_w_old[flat_tid] = 1.0f;
        tg_w_new[flat_tid] = 0.0f;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg < Groups) {
      for (uint i = 0u; i < pv_elems; ++i) {
        if (pv_layout.is_valid_element(i)) {
          const auto coord = pv_layout.get_multidimensional_index(i);
          const uint row = uint(coord[1]);
          acc_run[i] = fma(acc_run[i], tg_w_old[row], acc_blk[i] * tg_w_new[row]);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (sg < Groups) {
    for (uint i = 0u; i < pv_elems; ++i) {
      if (pv_layout.is_valid_element(i)) {
        const auto coord = pv_layout.get_multidimensional_index(i);
        const uint col = sg * 128u + uint(coord[0]);
        const uint row = uint(coord[1]);
        if (row < active_rows) {
          const uint tok = t_base + (row / repeat);
          const uint qh = kvh * repeat + (row % repeat);
          const float denom = tg_l_run[row];
          const float norm_o = (denom > 0.0f) ? (acc_run[i] / denom) : 0.0f;
          const float g_val = float(
              gate[ulong(tok) * p.gate_token_stride +
                   ulong(qh) * p.gate_head_stride + col]);
          output[(ulong(tok) * p.query_heads + qh) * HeadDim + col] =
              bfloat(norm_o / (1.0f + exp(-g_val)));
        }
      }
    }
  }
}

kernel void omlx_m5_paged_causal_attention_affine4_g128_hd256(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tg_q[2 * 16 * 128];
  threadgroup float tg_scores[2 * 16 * 64];
  threadgroup half tg_p[2 * 16 * 64];
  threadgroup float tg_q_sum[2 * 16];
  threadgroup uint tg_row_ctx[16];
  threadgroup float tg_rescale[16];
  threadgroup float tg_v_scale[2 * 16];
  threadgroup float tg_v_bias[2 * 16];
  threadgroup float tg_m_blk[16];
  threadgroup float tg_l_blk[16];
  threadgroup float tg_m_run[16];
  threadgroup float tg_l_run[16];
  threadgroup float tg_w_old[16];
  threadgroup float tg_w_new[16];

  causal_attention_affine4_single_impl<256u, 2u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, tg_q, tg_q_sum, tg_row_ctx, tg_scores, tg_p, tg_rescale,
      tg_v_scale, tg_v_bias, tg_m_blk, tg_l_blk, tg_m_run, tg_l_run, tg_w_old,
      tg_w_new, lane, sg, tg);
}

kernel void omlx_m5_paged_causal_attention_affine4_g128_hd128(
    device const bfloat *query [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const uchar *cache_key [[buffer(2)]],
    device const uchar *cache_value [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    device const uint *page_table [[buffer(5)]],
    device const uint *positions [[buffer(6)]],
    constant PagedAttentionParams &params [[buffer(7)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tg_q[16 * 128];
  threadgroup float tg_scores[16 * 64];
  threadgroup half tg_p[16 * 64];
  threadgroup float tg_q_sum[16];
  threadgroup uint tg_row_ctx[16];
  threadgroup float tg_rescale[16];
  threadgroup float tg_v_scale[16];
  threadgroup float tg_v_bias[16];
  threadgroup float tg_m_blk[16];
  threadgroup float tg_l_blk[16];
  threadgroup float tg_m_run[16];
  threadgroup float tg_l_run[16];
  threadgroup float tg_w_old[16];
  threadgroup float tg_w_new[16];

  causal_attention_affine4_single_impl<128u, 1u>(
      query, gate, cache_key, cache_value, output, page_table, positions,
      params, tg_q, tg_q_sum, tg_row_ctx, tg_scores, tg_p, tg_rescale,
      tg_v_scale, tg_v_bias, tg_m_blk, tg_l_blk, tg_m_run, tg_l_run, tg_w_old,
      tg_w_new, lane, sg, tg);
}

kernel void omlx_m5_paged_attention_affine4_g128_split_partial_hd128(
    device const bfloat *query [[buffer(0)]],
    device const uchar *cache_key [[buffer(1)]],
    device const uchar *cache_value [[buffer(2)]],
    device float *partial_max [[buffer(3)]],
    device float *partial_sum [[buffer(4)]],
    device float *partial_output [[buffer(5)]],
    device const uint *page_table [[buffer(6)]],
    device const uint *positions [[buffer(7)]],
    constant PagedSplitAttentionParams &p [[buffer(8)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup half tg_q[16 * 128];
  threadgroup float tg_scores[16 * 64];
  threadgroup half tg_p[16 * 64];
  threadgroup float tg_q_sum[16];
  threadgroup uint tg_row_ctx[16];
  threadgroup float tg_rescale[16];
  threadgroup float tg_v_scale[16];
  threadgroup float tg_v_bias[16];
  threadgroup float tg_m_blk[16];
  threadgroup float tg_l_blk[16];

  const uint split = tg.x;
  const uint repeat = p.query_heads / p.key_value_heads;
  if (split >= p.split_count || repeat == 0u || repeat > 16u) {
    return;
  }
  const uint bq = 16u / repeat;
  const uint kvh = tg.y % p.key_value_heads;
  const uint t_base = (tg.y / p.key_value_heads) * bq;
  if (t_base >= p.tokens) {
    return;
  }

  uint active_rows = 0u;
  uint max_ctx = 0u;
  affine4_kv::load_q_and_foscar_sums<128u, 1u>(
      query, positions, p.tokens, p.query_heads, p.key_value_heads,
      p.query_token_stride, p.query_head_stride, t_base, bq, kvh, tg_q,
      tg_q_sum, tg_row_ctx, active_rows, max_ctx, lane, sg);

  const uint pages_per_split = p.split_tokens / p.page_tokens;
  const uint total_pages = (max_ctx + p.page_tokens - 1u) / p.page_tokens;
  const uint start_page = split * pages_per_split;
  const uint end_page = min(total_pages, start_page + pages_per_split);

  float acc_blk[64];
  affine4_kv::run_superblock_nax_g128<128u, 1u>(
      cache_key, cache_value, page_table, p.page_tokens, p.key_value_heads, kvh,
      start_page, end_page, max_ctx, active_rows, tg_row_ctx, tg_q_sum, tg_q,
      tg_scores, tg_p, tg_rescale, tg_v_scale, tg_v_bias, tg_m_blk, tg_l_blk,
      acc_blk, lane, sg);

  const uint flat_tid = sg * 32u + lane;
  if (flat_tid < active_rows) {
    const uint tok = t_base + (flat_tid / repeat);
    const uint qh = kvh * repeat + (flat_tid % repeat);
    const ulong record =
        (ulong(tok) * p.query_heads + qh) * p.split_count + split;
    partial_max[record] = tg_m_blk[flat_tid];
    partial_sum[record] = tg_l_blk[flat_tid];
  }

  if (sg == 0u) {
    constexpr auto desc_pv = matmul2d_descriptor(
        16, 128, 64, false, false, false, matmul2d_descriptor::mode::multiply);
    matmul2d<desc_pv, execution_simdgroup> op_pv;
    tensor<threadgroup half, dextents<int32_t, 2>, tensor_inline> tp_dummy(
        tg_p, dextents<int32_t, 2>(64, 16));
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tv_dummy(
        (device uchar *)cache_value, dextents<int32_t, 2>(128, 64));
    auto pv_layout = op_pv.template get_destination_cooperative_tensor<
        decltype(tp_dummy), decltype(tv_dummy), float>();
    const uint pv_elems = pv_layout.get_capacity();
    for (uint i = 0u; i < pv_elems; ++i) {
      if (pv_layout.is_valid_element(i)) {
        const auto coord = pv_layout.get_multidimensional_index(i);
        const uint col = uint(coord[0]);
        const uint row = uint(coord[1]);
        if (row < active_rows) {
          const uint tok = t_base + (row / repeat);
          const uint qh = kvh * repeat + (row % repeat);
          const ulong record =
              (ulong(tok) * p.query_heads + qh) * p.split_count + split;
          partial_output[record * 128ul + col] = acc_blk[i];
        }
      }
    }
  }
}

// Preserved HD == 128 split merge (fast-math-safe: uses kNegInfHalf).
kernel void omlx_m5_paged_attention_affine4_g128_split_merge(
    device const bfloat *gate [[buffer(0)]],
    device const float *partial_max [[buffer(1)]],
    device const float *partial_sum [[buffer(2)]],
    device const float *partial_output [[buffer(3)]],
    device bfloat *output [[buffer(4)]],
    constant PagedSplitAttentionParams &params [[buffer(5)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint logical_head = tg.y;
  const uint token = logical_head / params.query_heads;
  if (token >= params.tokens) {
    return;
  }
  const uint query_head = logical_head - token * params.query_heads;
  const uint values_per_lane = (params.head_dim + 31u) / 32u;
  float acc[8];
  for (uint i = 0u; i < values_per_lane; ++i) {
    acc[i] = 0.0f;
  }

  float m_run = affine4_kv::kNegInf;
  float l_run = 0.0f;
  const ulong base_record = ulong(logical_head) * params.split_count;
  for (uint s = 0u; s < params.split_count; ++s) {
    const ulong rec = base_record + s;
    const float m_s = partial_max[rec];
    if (m_s > affine4_kv::kNegInfHalf) {
      const float new_m = max(m_run, m_s);
      const float w_old =
          (m_run <= affine4_kv::kNegInfHalf) ? 0.0f : exp(m_run - new_m);
      const float w_new = exp(m_s - new_m);
      m_run = new_m;
      l_run = fma(l_run, w_old, partial_sum[rec] * w_new);
      device const float *part_o = partial_output + rec * params.head_dim;
      for (uint i = 0u; i < values_per_lane; ++i) {
        const uint d = lane + i * 32u;
        if (d < params.head_dim) {
          acc[i] = fma(acc[i], w_old, part_o[d] * w_new);
        }
      }
    }
  }

  device const bfloat *gate_row =
      gate + ulong(token) * params.gate_token_stride +
      ulong(query_head) * params.gate_head_stride;
  device bfloat *out_row = output + ulong(logical_head) * params.head_dim;
  for (uint i = 0u; i < values_per_lane; ++i) {
    const uint d = lane + i * 32u;
    if (d < params.head_dim) {
      const float norm_o = (l_run > 0.0f) ? (acc[i] / l_run) : 0.0f;
      const float g_val = float(gate_row[d]);
      out_row[d] = bfloat(norm_o / (1.0f + exp(-g_val)));
    }
  }
}
