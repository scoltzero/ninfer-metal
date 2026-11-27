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
#include "common_math.h"
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
// Compact <= 15.5 KiB threadgroup storage for 2-SG HW uint4b_format Split-KV:
// Aliases tg_scores (<= 4 KiB) with tg_p (4 KiB) and reserves 11 KiB for
// per-lane c_acc[1..3] (44 floats) so peak live c_acc drops 64 -> 20 floats (0 B spill!).
template <uint R_BLK>
struct A5Nax2SgTgStorage {
  union {
    float scores[R_BLK * 64u]; // <= 4,096 B (R_BLK <= 16)
    half  p[2u * 16u * 64u];   //    4,096 B
  } sp;
  float acc_hi[2u][32u][44u];  // 11,264 B: c_acc1 (0..15) + c_acc2 (16..31) + c_acc3[0..11] (32..43)
  float q_sum[2u * R_BLK];
  uint  row_ctx[R_BLK];
  float m[R_BLK];
  float l[R_BLK];
  float rs[R_BLK];
  float vb[2u * R_BLK];
};
static_assert(sizeof(A5Nax2SgTgStorage<6u>) <= kApple10MaxPrefillTgSramBytes,
              "A5Nax2SgTgStorage<6> exceeds 16,384 B!");
static_assert(sizeof(A5Nax2SgTgStorage<8u>) <= kApple10MaxPrefillTgSramBytes,
              "A5Nax2SgTgStorage<8> exceeds 16,384 B!");
static_assert(sizeof(A5Nax2SgTgStorage<16u>) <= kApple10MaxPrefillTgSramBytes,
              "A5Nax2SgTgStorage<16> exceeds 16,384 B!");

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
    uint slot_row_off,
    threadgroup A5Nax2SgTgStorage<R_BLK> &tg_mem,
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

  threadgroup float *tg_scores = tg_mem.sp.scores;
  threadgroup half *tg_p = tg_mem.sp.p;
  threadgroup float *tg_q_sum = tg_mem.q_sum;
  threadgroup uint *tg_row_ctx = tg_mem.row_ctx;
  threadgroup float *tg_m = tg_mem.m;
  threadgroup float *tg_l = tg_mem.l;
  threadgroup float *tg_rs = tg_mem.rs;
  threadgroup float *tg_vb = tg_mem.vb;

  uint max_ctx = 0u;
  if constexpr (HasCausalPos) {
    if (flat_tid < R_BLK) {
      if (flat_tid < active_rows) {
        const uint tok = (row_base + flat_tid) / repeat;
        tg_row_ctx[flat_tid] = positions[slot_row_off + tok] + 1u;
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
    max_ctx = positions[slot_row_off] + 1u;
  }

  const uint pages_per_split = split_tokens / page_tokens;
  const uint total_pages = (max_ctx + page_tokens - 1u) / page_tokens;
  const uint start_page = split * pages_per_split;
  const uint end_page = min(total_pages, start_page + pages_per_split);

  // Fast empty-split exit without touching KV DRAM or partial_output_half.
  if (start_page >= end_page || max_ctx == 0u) {
    if (flat_tid < active_rows) {
      const uint r_global = row_base + flat_tid;
      const uint tok = slot_row_off + (r_global / repeat);
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
  for (uint k = 0u; k < 44u; ++k) {
    tg_mem.acc_hi[sg][lane][k] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint page_bytes_u32 =
      uint(page_byte_stride(page_tokens, key_value_heads, kHeadDim));
  const uint head_code_bytes_u32 = page_tokens * 128u;
  const uint all_codes_bytes_u32 = key_value_heads * head_code_bytes_u32;
  const float scale2 = 0.0625f * kLog2e; // rsqrt(256.0f) * log2(e)

  constexpr auto desc_qk = matmul2d_descriptor(
      16, 32, 128, false, true, false, matmul2d_descriptor::mode::multiply);
  matmul2d<desc_qk, execution_simdgroup> op_qk;

  constexpr auto desc_pv = matmul2d_descriptor(
      16, 32, 64, false, false, false,
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

  // Keep only c_acc0 (16 floats) + c3_tail (4 floats) in regs; 44 floats live in acc_hi!
  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tv_init(
      (device uchar *)cache_value, dextents<int32_t, 2>(128, 64));
  auto tv_init_sl = tv_init.slice(0, 0);
  auto c_acc0 = op_pv.template get_destination_cooperative_tensor<
      decltype(tp), decltype(tv_init_sl), float>();
  AFFINE4_UNROLL
  for (uint i = 0u; i < c_acc0.get_capacity(); ++i) {
    c_acc0[i] = 0.0f;
  }
  float4 c3_tail(0.0f);

  const uint sub_row = flat_tid >> 2u;  // 0..15
  const uint sub_lane = flat_tid & 3u;  // 0..3
  const uint c0 = sub_lane * 16u;

  for (uint page = start_page; page < end_page; ++page) {
    const uint p_start = page * page_tokens;
    const uint phys_page = page_table[page];
    device const uchar *k_page_ptr =
        cache_key + ulong(phys_page) * ulong(page_bytes_u32);
    device const uchar *k_codes_g =
        k_page_ptr + ulong(kvh * head_code_bytes_u32) + ulong(sg) * (64ul * 64ul);
    device const KvAffine4VecMeta *k_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(
            k_page_ptr + ulong(all_codes_bytes_u32)) +
        ulong(kvh) * page_tokens;

    // Phase 1: HW uint4b_format 2 x (16x32x128) NAX Q@K^T into single tg_scores.
    {
      const threadgroup float *q_sum_g = tg_q_sum + sg * R_BLK;
      #pragma clang loop unroll(disable)
      for (uint kh = 0u; kh < 2u; ++kh) {
        const uint col_off = kh * 32u;
        tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tk(
            (device uchar *)(k_codes_g + ulong(col_off) * 64ul),
            dextents<int32_t, 2>(128, 32));
        auto c_qk = op_qk.template get_destination_cooperative_tensor<
            decltype(tq), decltype(tk), float>();
        op_qk.run(tq, tk, c_qk);
        if (sg == 0u) {
          AFFINE4_UNROLL
          for (uint i = 0u; i < c_qk.get_capacity(); ++i) {
            if (c_qk.is_valid_element(i)) {
              const auto coord = c_qk.get_multidimensional_index(i);
              const uint col = col_off + uint(coord[0]);
              const uint row = uint(coord[1]);
              if (row < R_BLK) {
                const float sk = float(k_meta[col].scale[0]);
                const float zk = float(k_meta[col].zero[0]);
                tg_scores[row * 64u + col] =
                    fma(sk, c_qk[i], zk * q_sum_g[row]);
              }
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 1u) {
          AFFINE4_UNROLL
          for (uint i = 0u; i < c_qk.get_capacity(); ++i) {
            if (c_qk.is_valid_element(i)) {
              const auto coord = c_qk.get_multidimensional_index(i);
              const uint col = col_off + uint(coord[0]);
              const uint row = uint(coord[1]);
              if (row < R_BLK) {
                const float sk = float(k_meta[col].scale[1]);
                const float zk = float(k_meta[col].zero[1]);
                tg_scores[row * 64u + col] +=
                    fma(sk, c_qk[i], zk * q_sum_g[row]);
              }
            }
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
    }

    device const uchar *v_page_ptr =
        cache_value + ulong(phys_page) * ulong(page_bytes_u32);
    device const uchar *v_codes_g =
        v_page_ptr + ulong(kvh * head_code_bytes_u32) + ulong(sg) * (64ul * 64ul);
    device const KvAffine4VecMeta *v_meta =
        reinterpret_cast<device const KvAffine4VecMeta *>(
            v_page_ptr + ulong(all_codes_bytes_u32)) +
        ulong(kvh) * page_tokens;

    // Phase 2: Parallel 4-lanes-per-row log2 online softmax + V scale/bias prep.
    // Stage 16 halfs for g=0 and 16 halfs for g=1 in 8 compact half4 regs so
    // tg_p safely aliases tg_scores across a single barrier!
    half4 p0_loc0(0.0h), p0_loc1(0.0h), p0_loc2(0.0h), p0_loc3(0.0h);
    half4 p1_loc0(0.0h), p1_loc1(0.0h), p1_loc2(0.0h), p1_loc3(0.0h);
    if (sub_row < active_rows) {
      const uint row_ctx = HasCausalPos ? tg_row_ctx[sub_row] : max_ctx;
      const uint row_valid =
          (row_ctx > p_start) ? min(64u, row_ctx - p_start) : 0u;
      if (row_valid == 0u) {
        if (sub_lane == 0u) {
          tg_rs[sub_row] = 1.0f;
          tg_vb[sub_row] = 0.0f;
          tg_vb[R_BLK + sub_row] = 0.0f;
        }
      } else {
        const threadgroup float4 *sc_row =
            reinterpret_cast<const threadgroup float4 *>(
                tg_scores + sub_row * 64u + c0);
        float lmax = kNegInf;
        #pragma clang loop unroll(disable)
        for (uint ch = 0u; ch < 4u; ++ch) {
          float4 sv = sc_row[ch] * scale2;
          if (row_valid < 64u) {
            const uint base_c = c0 + ch * 4u;
            const uint4 idx =
                uint4(base_c + 0u, base_c + 1u, base_c + 2u, base_c + 3u);
            sv = select(float4(kNegInf), sv, idx < row_valid);
          }
          lmax = max(lmax, max(max(sv.x, sv.y), max(sv.z, sv.w)));
        }
        lmax = max(lmax, simd_shuffle_xor(lmax, ushort(1)));
        lmax = max(lmax, simd_shuffle_xor(lmax, ushort(2)));

        const float prev_m = tg_m[sub_row];
        const float new_m = max(prev_m, lmax);
        const float rescale =
            (prev_m <= kNegInfHalf) ? 0.0f : fast::exp2(prev_m - new_m);

        float4 psum4(0.0f);
        float lvb0 = 0.0f;
        float lvb1 = 0.0f;
        #define A5_STAGE_P_CHUNK(CH, DST0, DST1)                               \
          do {                                                                 \
            const uint base_c = c0 + uint(CH) * 4u;                            \
            float4 pv = fast::exp2(sc_row[CH] * scale2 - new_m);               \
            if (row_valid < 64u) {                                             \
              const uint4 idx =                                                \
                  uint4(base_c + 0u, base_c + 1u, base_c + 2u, base_c + 3u);   \
              pv = select(float4(0.0f), pv, idx < row_valid);                  \
            }                                                                  \
            psum4 += pv;                                                       \
            const KvAffine4VecMeta m0 = v_meta[base_c + 0u];                   \
            const KvAffine4VecMeta m1 = v_meta[base_c + 1u];                   \
            const KvAffine4VecMeta m2 = v_meta[base_c + 2u];                   \
            const KvAffine4VecMeta m3 = v_meta[base_c + 3u];                   \
            const float4 sv0(float(m0.scale[0]), float(m1.scale[0]),           \
                             float(m2.scale[0]), float(m3.scale[0]));          \
            const float4 zv0(float(m0.zero[0]), float(m1.zero[0]),             \
                             float(m2.zero[0]), float(m3.zero[0]));            \
            const float4 sv1(float(m0.scale[1]), float(m1.scale[1]),           \
                             float(m2.scale[1]), float(m3.scale[1]));          \
            const float4 zv1(float(m0.zero[1]), float(m1.zero[1]),             \
                             float(m2.zero[1]), float(m3.zero[1]));            \
            (DST0) = half4(pv * sv0);                                          \
            (DST1) = half4(pv * sv1);                                          \
            lvb0 += dot(pv, zv0);                                              \
            lvb1 += dot(pv, zv1);                                              \
          } while (false)
        A5_STAGE_P_CHUNK(0u, p0_loc0, p1_loc0);
        A5_STAGE_P_CHUNK(1u, p0_loc1, p1_loc1);
        A5_STAGE_P_CHUNK(2u, p0_loc2, p1_loc2);
        A5_STAGE_P_CHUNK(3u, p0_loc3, p1_loc3);
        #undef A5_STAGE_P_CHUNK

        float lsum = (psum4.x + psum4.y) + (psum4.z + psum4.w);
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
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write staged half4 weights into aliased tg_p (including zeroing inactive rows!).
    {
      threadgroup half4 *p0_dst =
          reinterpret_cast<threadgroup half4 *>(tg_p + sub_row * 64u + c0);
      threadgroup half4 *p1_dst = reinterpret_cast<threadgroup half4 *>(
          tg_p + (16u * 64u) + sub_row * 64u + c0);
      p0_dst[0] = p0_loc0;
      p0_dst[1] = p0_loc1;
      p0_dst[2] = p0_loc2;
      p0_dst[3] = p0_loc3;
      p1_dst[0] = p1_loc0;
      p1_dst[1] = p1_loc1;
      p1_dst[2] = p1_loc2;
      p1_dst[3] = p1_loc3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: 4 x HW uint4b_format 16x32x64 NAX P@V (c_acc0/1 in regs, c_acc2/3 stashed!).
    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tv(
        (device uchar *)v_codes_g, dextents<int32_t, 2>(128, 64));
    #define A5_PV_QUARTER(CACC, COL_OFF)                                       \
      do {                                                                     \
        AFFINE4_UNROLL                                                         \
        for (uint i = 0u; i < (CACC).get_capacity(); ++i) {                    \
          if ((CACC).is_valid_element(i)) {                                    \
            const auto coord = (CACC).get_multidimensional_index(i);           \
            const uint row = uint(coord[1]);                                   \
            if (row < R_BLK) {                                                 \
              (CACC)[i] =                                                      \
                  fma((CACC)[i], tg_rs[row], tg_vb[sg * R_BLK + row]);         \
            }                                                                  \
          }                                                                    \
        }                                                                      \
        auto tv_sl = tv.slice(int(COL_OFF), 0);                                \
        op_pv.run(tp, tv_sl, (CACC));                                          \
      } while (false)
    A5_PV_QUARTER(c_acc0, 0);
    {
      auto c_tmp = op_pv.template get_destination_cooperative_tensor<
          decltype(tp), decltype(tv_init_sl), float>();
      AFFINE4_UNROLL
      for (uint i = 0u; i < c_tmp.get_capacity(); ++i) {
        c_tmp[i] = tg_mem.acc_hi[sg][lane][i];
      }
      A5_PV_QUARTER(c_tmp, 32);
      AFFINE4_UNROLL
      for (uint i = 0u; i < c_tmp.get_capacity(); ++i) {
        tg_mem.acc_hi[sg][lane][i] = c_tmp[i];
      }
    }
    {
      auto c_tmp = op_pv.template get_destination_cooperative_tensor<
          decltype(tp), decltype(tv_init_sl), float>();
      AFFINE4_UNROLL
      for (uint i = 0u; i < c_tmp.get_capacity(); ++i) {
        c_tmp[i] = tg_mem.acc_hi[sg][lane][16u + i];
      }
      A5_PV_QUARTER(c_tmp, 64);
      AFFINE4_UNROLL
      for (uint i = 0u; i < c_tmp.get_capacity(); ++i) {
        tg_mem.acc_hi[sg][lane][16u + i] = c_tmp[i];
      }
    }
    {
      auto c_tmp = op_pv.template get_destination_cooperative_tensor<
          decltype(tp), decltype(tv_init_sl), float>();
      AFFINE4_UNROLL
      for (uint i = 0u; i < 12u; ++i) {
        c_tmp[i] = tg_mem.acc_hi[sg][lane][32u + i];
      }
      c_tmp[12] = c3_tail.x;
      c_tmp[13] = c3_tail.y;
      c_tmp[14] = c3_tail.z;
      c_tmp[15] = c3_tail.w;
      A5_PV_QUARTER(c_tmp, 96);
      AFFINE4_UNROLL
      for (uint i = 0u; i < 12u; ++i) {
        tg_mem.acc_hi[sg][lane][32u + i] = c_tmp[i];
      }
      c3_tail = float4(c_tmp[12], c_tmp[13], c_tmp[14], c_tmp[15]);
    }
    #undef A5_PV_QUARTER
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (flat_tid < active_rows) {
    const uint r_global = row_base + flat_tid;
    const uint tok = slot_row_off + (r_global / repeat);
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
  #define A5_STORE_QUARTER(CACC, COL_OFF)                                      \
    do {                                                                       \
      AFFINE4_UNROLL                                                           \
      for (uint i = 0u; i < (CACC).get_capacity(); ++i) {                      \
        if ((CACC).is_valid_element(i)) {                                      \
          const auto coord = (CACC).get_multidimensional_index(i);             \
          const uint col = sg * 128u + uint(COL_OFF) + uint(coord[0]);         \
          const uint row = uint(coord[1]);                                     \
          if (row < active_rows) {                                             \
            const uint r_global = row_base + row;                              \
            const uint tok = slot_row_off + (r_global / repeat);               \
            const uint qh = kvh * repeat + (r_global % repeat);                \
            const ulong record =                                               \
                (ulong(tok) * query_heads + qh) * split_count + split;         \
            partial_output_half[record * kHeadDim + col] =                     \
                half((CACC)[i] * tg_rs[row]);                                  \
          }                                                                    \
        }                                                                      \
      }                                                                        \
    } while (false)
  A5_STORE_QUARTER(c_acc0, 0u);
  {
    auto c_tmp = op_pv.template get_destination_cooperative_tensor<
        decltype(tp), decltype(tv_init_sl), float>();
    AFFINE4_UNROLL
    for (uint i = 0u; i < c_tmp.get_capacity(); ++i) {
      c_tmp[i] = tg_mem.acc_hi[sg][lane][i];
    }
    A5_STORE_QUARTER(c_tmp, 32u);
  }
  {
    auto c_tmp = op_pv.template get_destination_cooperative_tensor<
        decltype(tp), decltype(tv_init_sl), float>();
    AFFINE4_UNROLL
    for (uint i = 0u; i < c_tmp.get_capacity(); ++i) {
      c_tmp[i] = tg_mem.acc_hi[sg][lane][16u + i];
    }
    A5_STORE_QUARTER(c_tmp, 64u);
  }
  {
    auto c_tmp = op_pv.template get_destination_cooperative_tensor<
        decltype(tp), decltype(tv_init_sl), float>();
    AFFINE4_UNROLL
    for (uint i = 0u; i < 12u; ++i) {
      c_tmp[i] = tg_mem.acc_hi[sg][lane][32u + i];
    }
    c_tmp[12] = c3_tail.x;
    c_tmp[13] = c3_tail.y;
    c_tmp[14] = c3_tail.z;
    c_tmp[15] = c3_tail.w;
    A5_STORE_QUARTER(c_tmp, 96u);
  }
  #undef A5_STORE_QUARTER
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
    threadgroup A5Nax2SgTgStorage<R_BLK> &tg_mem,
    uint lane,
    uint sg,
    uint3 tg,
    uint3 tgs_per_grid) {
  a5_nax_2sg_core<R_BLK, HasCausalPos>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, p.query_heads,
      p.key_value_heads, p.tokens, p.page_tokens, p.split_count, p.split_tokens,
      tgs_per_grid.z * 16u, tg.x, tg.y, tg.z, 0u, tg_mem, lane, sg);
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
    threadgroup A5Nax2SgTgStorage<R_BLK> &tg_mem,
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
  a5_nax_2sg_core<R_BLK, HasCausalPos>(
      q_half + ulong(slot) * slot_q_stride * 128ul,
      q_sum + ulong(slot) * slot_q_stride,
      cache_key, cache_value,
      partial_max, partial_sum, partial_output_half,
      slot_pt, positions,
      p.query_heads, p.key_value_heads, slot_tokens, p.page_tokens,
      p.split_count, p.split_tokens, p.r_stride, split, kvh, rblk, row_off,
      tg_mem, lane, sg);
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
AFFINE4_ALWAYS_INLINE
inline void load_q_and_foscar_sums(
    device const bfloat *query, device const uint *positions, uint tokens,
    uint query_heads, uint key_value_heads, uint query_token_stride,
    uint query_head_stride, uint t_base, uint bq, uint kvh,
    threadgroup half *tg_q, threadgroup float *tg_q_sum,
    threadgroup uint *tg_row_ctx, thread uint &out_active_rows,
    thread uint &out_max_ctx, uint lane, uint sg);

// Zero-spill pure-FP32 SIMD compatibility engine for !split (small T) and HD == 128 split:
// Each SIMDgroup owns 128 dims (32 lanes x 4 dims = 1 ushort of 4 affine4 nibbles).
// 0 cooperative_tensor allocas, <= 64 B TG SRAM, ~35 GPRs, 0 B spills on Apple9 & Apple10!
template <uint HeadDim, uint Groups, bool EmitSplit>
AFFINE4_ALWAYS_INLINE
inline void run_affine4_compat_core(
    device const bfloat *query,
    device const bfloat *gate,
    device const uchar *cache_key,
    device const uchar *cache_value,
    device bfloat *output,
    device float *partial_max,
    device float *partial_sum,
    device float *partial_output,
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
    uint split_count,
    uint split_tokens,
    uint split,
    uint t_base,
    uint bq,
    uint kvh,
    threadgroup float *tg_dot,
    uint lane,
    uint sg) {
  const uint repeat = query_heads / key_value_heads;
  const uint active_q_tokens =
      (t_base < tokens) ? min(bq, tokens - t_base) : 0u;
  const uint active_rows = active_q_tokens * repeat;
  if (active_rows == 0u) {
    return;
  }

  const uint page_bytes_u32 =
      uint(page_byte_stride(page_tokens, key_value_heads, HeadDim));
  const uint head_code_bytes_u32 = page_tokens * (HeadDim / 2u);
  const uint all_codes_bytes_u32 = key_value_heads * head_code_bytes_u32;
  const uint pages_per_split =
      EmitSplit ? (split_tokens / page_tokens) : 0x7fffffffu;
  const float attn_scale = rsqrt(float(HeadDim));
  const uint d_base = sg * 128u + lane * 4u;

  for (uint r = 0u; r < active_rows; ++r) {
    const uint tok = t_base + (r / repeat);
    const uint qh = kvh * repeat + (r % repeat);
    const uint row_ctx = positions[tok] + 1u;
    const uint total_pages = (row_ctx + page_tokens - 1u) / page_tokens;
    const uint start_page = EmitSplit ? (split * pages_per_split) : 0u;
    const uint end_page =
        EmitSplit ? min(total_pages, start_page + pages_per_split) : total_pages;

    if (start_page >= end_page || row_ctx == 0u) {
      if constexpr (EmitSplit) {
        if (sg == 0u && lane == 0u) {
          const ulong record =
              (ulong(tok) * query_heads + qh) * split_count + split;
          partial_max[record] = kNegInf;
          partial_sum[record] = 0.0f;
        }
      } else {
        device bfloat4 *out4 = reinterpret_cast<device bfloat4 *>(
            output + (ulong(tok) * query_heads + qh) * HeadDim + d_base);
        *out4 = bfloat4(bfloat(0.0f));
      }
      continue;
    }

    const float4 qv = float4(*reinterpret_cast<device const bfloat4 *>(
        query + ulong(tok) * query_token_stride +
        ulong(qh) * query_head_stride + d_base));

    float m_run = kNegInf;
    float l_run = 0.0f;
    float4 o_acc(0.0f);

    for (uint page = start_page; page < end_page; ++page) {
      const uint p_start = page * page_tokens;
      const uint row_valid =
          (row_ctx > p_start) ? min(page_tokens, row_ctx - p_start) : 0u;
      if (row_valid == 0u) {
        break;
      }
      const uint phys_page = page_table[page];
      device const uchar *k_page_ptr =
          cache_key + ulong(phys_page) * ulong(page_bytes_u32);
      device const ushort *k_codes_g = reinterpret_cast<device const ushort *>(
          k_page_ptr + ulong(kvh * head_code_bytes_u32) +
          ulong(sg) * (ulong(page_tokens) * 64ul));
      device const KvAffine4VecMeta *k_meta =
          reinterpret_cast<device const KvAffine4VecMeta *>(
              k_page_ptr + ulong(all_codes_bytes_u32)) +
          ulong(kvh) * page_tokens;

      device const uchar *v_page_ptr =
          cache_value + ulong(phys_page) * ulong(page_bytes_u32);
      device const ushort *v_codes_g = reinterpret_cast<device const ushort *>(
          v_page_ptr + ulong(kvh * head_code_bytes_u32) +
          ulong(sg) * (ulong(page_tokens) * 64ul));
      device const KvAffine4VecMeta *v_meta =
          reinterpret_cast<device const KvAffine4VecMeta *>(
              v_page_ptr + ulong(all_codes_bytes_u32)) +
          ulong(kvh) * page_tokens;

      for (uint c = 0u; c < row_valid; ++c) {
        const ushort kpk = k_codes_g[c * 32u + lane];
        const float4 kq = float4(
            float(kpk & 0xFu), float((kpk >> 4u) & 0xFu),
            float((kpk >> 8u) & 0xFu), float(kpk >> 12u));
        const float ks = float(k_meta[c].scale[sg]);
        const float kz = float(k_meta[c].zero[sg]);
        const float4 kv = fma(kq, float4(ks), float4(kz));
        float dot_val = simd_sum(dot(qv, kv));
        if constexpr (Groups == 2u) {
          if (lane == 0u) {
            tg_dot[sg] = dot_val;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          dot_val = tg_dot[0] + tg_dot[1];
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float s = dot_val * attn_scale;
        const float new_m = max(m_run, s);
        const float rescale =
            (m_run <= kNegInfHalf) ? 0.0f : exp(m_run - new_m);
        const float p_val = exp(s - new_m);
        m_run = new_m;
        l_run = fma(l_run, rescale, p_val);

        const ushort vpk = v_codes_g[c * 32u + lane];
        const float4 vq = float4(
            float(vpk & 0xFu), float((vpk >> 4u) & 0xFu),
            float((vpk >> 8u) & 0xFu), float(vpk >> 12u));
        const float vs = float(v_meta[c].scale[sg]);
        const float vz = float(v_meta[c].zero[sg]);
        const float4 vv = fma(vq, float4(vs), float4(vz));
        o_acc = fma(o_acc, float4(rescale), vv * p_val);
      }
    }

    if constexpr (EmitSplit) {
      const ulong record =
          (ulong(tok) * query_heads + qh) * split_count + split;
      if (sg == 0u && lane == 0u) {
        partial_max[record] = m_run;
        partial_sum[record] = l_run;
      }
      *reinterpret_cast<device float4 *>(
          partial_output + record * HeadDim + d_base) = o_acc;
    } else {
      const float inv_l = (l_run > 0.0f) ? (1.0f / l_run) : 0.0f;
      const float4 norm_o = o_acc * inv_l;
      const float4 g_val = float4(*reinterpret_cast<device const bfloat4 *>(
          gate + ulong(tok) * gate_token_stride +
          ulong(qh) * gate_head_stride + d_base));
      *reinterpret_cast<device bfloat4 *>(
          output + (ulong(tok) * query_heads + qh) * HeadDim + d_base) =
          bfloat4(norm_o / (float4(1.0f) + exp(-g_val)));
    }
  }
}

template <uint HeadDim, uint Groups>
AFFINE4_ALWAYS_INLINE
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
[[max_total_threads_per_threadgroup(64)]]
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
  alignas(16) threadgroup affine4_kv::A5Nax2SgTgStorage<6u> tg_mem;
  affine4_kv::a5_nax_2sg_impl<6u, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, params, tg_mem, lane, sg, tg,
      tgs_per_grid);
}

[[max_total_threads_per_threadgroup(64)]]
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
  alignas(16) threadgroup affine4_kv::A5Nax2SgTgStorage<8u> tg_mem;
  affine4_kv::a5_nax_2sg_impl<8u, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, params, tg_mem, lane, sg, tg,
      tgs_per_grid);
}

[[max_total_threads_per_threadgroup(64)]]
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
  alignas(16) threadgroup affine4_kv::A5Nax2SgTgStorage<16u> tg_mem;
  affine4_kv::a5_nax_2sg_impl<16u, true>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, page_table, positions, params, tg_mem, lane, sg, tg,
      tgs_per_grid);
}

// ============================================================================
// 3b. Batched B=2..8 Unified 2-SG HW uint4b_format NAX Split-KV Entrypoints
// Launch: grid = {splits * 64, KVH * n_rblks, B}, tptg = {64}
// ============================================================================
[[max_total_threads_per_threadgroup(64)]]
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
  alignas(16) threadgroup affine4_kv::A5Nax2SgTgStorage<6u> tg_mem;
  affine4_kv::a5_nax_2sg_batched_impl<6u, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7,
      params, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
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
  alignas(16) threadgroup affine4_kv::A5Nax2SgTgStorage<8u> tg_mem;
  affine4_kv::a5_nax_2sg_batched_impl<8u, false>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7,
      params, tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(64)]]
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
  alignas(16) threadgroup affine4_kv::A5Nax2SgTgStorage<16u> tg_mem;
  affine4_kv::a5_nax_2sg_batched_impl<16u, true>(
      q_half, q_sum, cache_key, cache_value, partial_max, partial_sum,
      partial_output_half, positions, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7,
      params, tg_mem, lane, sg, tg);
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
// 5. WP3.2 BK=32, WN=4 Direct-L1 Zero-Spill Affine4Group128 Flash-Prefill
//    (WM=2: 14.00 KiB exact, WM=3: 12.00 KiB <= 14.00 KiB -> 2 TGs/core, 0 B spills!)
//    Direct-L1 dequant + row-split 2-half d_part==0 softmax avoids kv_tile SRAM
//    bottleneck and strictly beats single-head BQ=64 across 4K..128K!
//    Shared core powers BOTH single-head (BQ=64 via 2x32 passes, batched & unbatched)
//    and flattened GQA (WM=2: 256 thr, WM=3: 384 thr) with 0-ULP bit-exact parity!
// ============================================================================
namespace affine4_kv {

struct FlashGqaBk32XchgLaneHalf {
  union {
    float4 f4[4]; // 64 B: S0[0..3], S0[4..7], S1[0..3], S1[4..7]
    struct {
      half4  p00, p01, p10, p11; // 32 B
      float2 fac;                //  8 B: (fac0, fac1) or (rcp0, rcp1)
      float2 _pad[3];            // 24 B
    } bcast;
  };
};
static_assert(sizeof(FlashGqaBk32XchgLaneHalf) == 64u,
              "FlashGqaBk32XchgLaneHalf must be 64 B");

struct FlashGqaBk32Xchg32Half {
  union {
    float4 f4[2]; // 32 B: half-block S[0..3], S[4..7]
    struct {
      half4  p0;   //  8 B
      half4  p1;   //  8 B
      float2 fac;  //  8 B (slot 0)
      float2 _pad; //  8 B
    } bcast;
  };
};
static_assert(sizeof(FlashGqaBk32Xchg32Half) == 32u,
              "FlashGqaBk32Xchg32Half must be 32 B");

template <uint WM>
struct FlashGqaBk32TgStorageHalf;

template <>
struct FlashGqaBk32TgStorageHalf<2u> {
  // WM=2 (256 thr): 12,288 B xchg + 2,048 B d0_s0 = 14,336 B (14.00 KiB EXACT!)
  // Aliased as tg_q_pos[] during prologue before KV loop.
  FlashGqaBk32XchgLaneHalf xchg[2u][3u][32u];
  float4                   d0_s0[2u][32u][2u];
};

template <>
struct FlashGqaBk32TgStorageHalf<3u> {
  // WM=3 (384 thr): 3 row groups x 4 x 32 B/lane = 12,288 B (12.00 KiB <= 14.00 KiB!)
  // Slot 0..2: partial S exchange for h2=0 then h2=1;
  // Slot 3: d_part==0 scaled/masked S0 stash;
  // Broadcast: slot 0 holds (p00, p01, fac), slot 1 holds (p10, p11).
  FlashGqaBk32Xchg32Half xchg[3u][4u][32u];
};

static_assert(sizeof(FlashGqaBk32TgStorageHalf<2u>) == 14336u &&
                  sizeof(FlashGqaBk32TgStorageHalf<2u>) <=
                      kApple10MaxGqaPrefillTgSramBytes,
              "WP3.2 Affine4 WM=2 TgStorage must be exact 14.00 KiB!");
static_assert(sizeof(FlashGqaBk32TgStorageHalf<3u>) == 12288u &&
                  sizeof(FlashGqaBk32TgStorageHalf<3u>) <=
                      kApple10MaxGqaPrefillTgSramBytes,
              "WP3.2 Affine4 WM=3 TgStorage must be exact 12.00 KiB (<= 14.00 KiB)!");

AFFINE4_ALWAYS_INLINE
static inline ushort4 gqa_clean_bf4_u16(ushort4 bits) {
  return select(bits, ushort4(0u),
                (bits & ushort4(0x7fffu)) > ushort4(0x7f80u));
}

AFFINE4_ALWAYS_INLINE
static inline void fast_divmod_g(uint r, uint g, thread uint &q,
                                 thread uint &rem) {
  if (g == 6u) {
    q = (r * 43u) >> 8u;
    rem = r - q * 6u;
  } else if (g == 8u) {
    q = r >> 3u;
    rem = r & 7u;
  } else if (g == 4u) {
    q = r >> 2u;
    rem = r & 3u;
  } else if (g == 2u) {
    q = r >> 1u;
    rem = r & 1u;
  } else if (g == 1u) {
    q = r;
    rem = 0u;
  } else {
    q = r / g;
    rem = r - q * g;
  }
}

AFFINE4_ALWAYS_INLINE
static inline Frag16x16Aff4::frag_t<half> dequant_ptr_aff4_h8(
    device const ushort *p0, device const ushort *p1,
    half s0, half z0, half s1, half z1, uint col_off) {
  const ushort pk0 = p0[col_off];
  const ushort pk1 = p1[col_off];
  const half4 q0 = half4(ushort4(pk0 & 0xFu, (pk0 >> 4u) & 0xFu,
                                 (pk0 >> 8u) & 0xFu, pk0 >> 12u));
  const half4 q1 = half4(ushort4(pk1 & 0xFu, (pk1 >> 4u) & 0xFu,
                                 (pk1 >> 8u) & 0xFu, pk1 >> 12u));
  const half4 v0 = fma(q0, half4(s0), half4(z0));
  const half4 v1 = fma(q1, half4(s1), half4(z1));
  return Frag16x16Aff4::frag_t<half>(v0.x, v0.y, v0.z, v0.w,
                                     v1.x, v1.y, v1.z, v1.w);
}

template <uint WM, bool KeepQInRegs = true, typename ParamsT = PagedAttentionParams>
AFFINE4_ALWAYS_INLINE
static inline void flash_bk32_wn4_affine4_hd256_core(
    device const bfloat *query, device const bfloat *gate,
    device const uchar *cache_key, device const uchar *cache_value,
    device bfloat *output, device const uint *page_table,
    constant ParamsT &p,
    threadgroup FlashGqaBk32TgStorageHalf<WM> &tg_mem, uint kvh,
    uint nk_blocks, uint max_valid_kv, uint kb_min_causal, uint tok0, uint qh0,
    uint qpos0, bool qv0, uint tok1, uint qh1, uint qpos1, bool qv1,
    bool warp_full_rows, ushort lane, ushort sg) {
  constexpr uint BD = 256u;
  constexpr uint BK = 32u;

  const uint row_group = uint(sg) >> 2u;
  const uint d_part = uint(sg) & 3u;
  const uint aff_grp = d_part >> 1u;
  const uint d_offset = d_part * 64u;
  const short2 sc = Frag16x16Aff4::get_coord(lane);
  const short sm = sc.y;
  const short sn = sc.x;
  const uint sn_u16 = (d_part & 1u) * 16u + uint(sn >> 2u);

  using H8 = Frag16x16Aff4::frag_t<half>;
  using F8 = Frag16x16Aff4::frag_t<float>;

  const uint row0_p = (tok0 << 8u) | ((qh0 & 0x7Fu) << 1u) | (qv0 ? 1u : 0u);
  const uint row1_p = (tok1 << 8u) | ((qh1 & 0x7Fu) << 1u) | (qv1 ? 1u : 0u);

  #define AFF4_BK32_LOAD_Q(DST, OFF_C)                                         \
    do {                                                                       \
      half4 v0 = half4(0.0h), v1 = half4(0.0h);                                \
      if (row0_p & 1u) {                                                       \
        ushort4 u0 = *reinterpret_cast<device const ushort4 *>(                \
            query + ulong(row0_p >> 8u) * p.query_token_stride +               \
            ulong((row0_p >> 1u) & 0x7Fu) * p.query_head_stride + d_offset +   \
            uint(sn) + (OFF_C));                                               \
        if (!warp_full_rows) u0 = gqa_clean_bf4_u16(u0);                       \
        v0 = half4(float4(as_type<bfloat4>(u0)));                              \
      }                                                                        \
      if (row1_p & 1u) {                                                       \
        ushort4 u1 = *reinterpret_cast<device const ushort4 *>(                \
            query + ulong(row1_p >> 8u) * p.query_token_stride +               \
            ulong((row1_p >> 1u) & 0x7Fu) * p.query_head_stride + d_offset +   \
            uint(sn) + (OFF_C));                                               \
        if (!warp_full_rows) u1 = gqa_clean_bf4_u16(u1);                       \
        v1 = half4(float4(as_type<bfloat4>(u1)));                              \
      }                                                                        \
      (DST) = H8(v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w);              \
    } while (false)

  H8 Q0(0.0h), Q1(0.0h), Q2(0.0h), Q3(0.0h);
  if constexpr (KeepQInRegs) {
    AFF4_BK32_LOAD_Q(Q0, 0u);
    AFF4_BK32_LOAD_Q(Q1, 16u);
    AFF4_BK32_LOAD_Q(Q2, 32u);
    AFF4_BK32_LOAD_Q(Q3, 48u);
  }

  const uint grp_code_off_u32 =
      kvh * (p.page_tokens * 128u) + aff_grp * (p.page_tokens * 64u);
  const uint page_bytes_u32 =
      uint(page_byte_stride(p.page_tokens, p.key_value_heads, BD));
  const uint pt_shift = ctz(p.page_tokens >> 5u);
  const uint pt_mask = (p.page_tokens >> 5u) - 1u;

  F8 O0(0.0f), O1(0.0f), O2(0.0f), O3(0.0f);
  float max_s0 = kNegInf, max_s1 = kNegInf, sum_s0 = 0.0f, sum_s1 = 0.0f;
  constexpr float kScale2 = 0.0625f * kLog2e;

  for (uint kb = 0u; kb < nk_blocks; ++kb) {
    const uint lpage = kb >> pt_shift;
    const uint sub_off = (kb & pt_mask) * BK;
    const uint phys_page = page_table[lpage];
    const uint kb_col_base = kb * BK;
    const short valid_k = short(min(BK, max_valid_kv - kb_col_base));
    const bool full_cf =
        (kb < kb_min_causal) && (valid_k == 32) && warp_full_rows;

    if constexpr (WM == 2u) {
      // Compact non-unrolled h2=0..1 QK loop using a SINGLE F8 S (gpr=118, 0 B spills!).
      F8 S;
      #pragma clang loop unroll(disable)
      for (uint h2 = 0u; h2 < 2u; ++h2) {
        device const uchar *kp =
            cache_key + ulong(phys_page) * ulong(page_bytes_u32);
        device const ushort *k_cg =
            reinterpret_cast<device const ushort *>(kp + grp_code_off_u32);
        device const KvAffine4VecMeta *k_meta =
            reinterpret_cast<device const KvAffine4VecMeta *>(
                kp + p.key_value_heads * (p.page_tokens * 128u)) +
            ulong(kvh) * p.page_tokens;
        const uint row_off = h2 * 16u;
        const uint r0 = sub_off + row_off + uint(sm), r1 = r0 + 8u;
        const bool ok0 = (short(int(row_off) + int(sm)) < valid_k);
        const bool ok1 = (short(int(row_off) + int(sm) + 8) < valid_k);
        const half s0 = ok0 ? k_meta[r0].scale[aff_grp] : 0.0h;
        const half z0 = ok0 ? k_meta[r0].zero[aff_grp] : 0.0h;
        const half s1 = ok1 ? k_meta[r1].scale[aff_grp] : 0.0h;
        const half z1 = ok1 ? k_meta[r1].zero[aff_grp] : 0.0h;
        device const ushort *p0 = k_cg + (ok0 ? r0 : sub_off) * 32u + sn_u16;
        device const ushort *p1 = k_cg + (ok1 ? r1 : sub_off) * 32u + sn_u16;
        S = F8(0.0f);
        if constexpr (KeepQInRegs) {
          {
            const H8 k0 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 0u);
            const H8 k1 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 4u);
            nax_mma_16x16x32<half, true>(S, Q0, Q1, k0, k1);
          }
          {
            const H8 k2 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 8u);
            const H8 k3 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 12u);
            nax_mma_16x16x32<half, true>(S, Q2, Q3, k2, k3);
          }
        } else {
          {
            H8 q0_l, q1_l;
            AFF4_BK32_LOAD_Q(q0_l, 0u);
            AFF4_BK32_LOAD_Q(q1_l, 16u);
            const H8 k0 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 0u);
            const H8 k1 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 4u);
            nax_mma_16x16x32<half, true>(S, q0_l, q1_l, k0, k1);
          }
          simdgroup_barrier(mem_flags::mem_none);
          {
            H8 q2_l, q3_l;
            AFF4_BK32_LOAD_Q(q2_l, 32u);
            AFF4_BK32_LOAD_Q(q3_l, 48u);
            const H8 k2 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 8u);
            const H8 k3 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 12u);
            nax_mma_16x16x32<half, true>(S, q2_l, q3_l, k2, k3);
          }
        }
        if (d_part == 0u) {
          if (h2 == 0u) {
            tg_mem.d0_s0[row_group][lane][0] = float4(S[0], S[1], S[2], S[3]);
            tg_mem.d0_s0[row_group][lane][1] = float4(S[4], S[5], S[6], S[7]);
          }
        } else {
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[h2 * 2u + 0u] =
              float4(S[0], S[1], S[2], S[3]);
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[h2 * 2u + 1u] =
              float4(S[4], S[5], S[6], S[7]);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Row-split 2-half softmax on d_part == 0 (ONLY 4 floats live at a time -> 0 B spills!).
      if (d_part == 0u) {
        float m0_b, m1_b;
        {
          float4 s10 =
              (float4(S[0], S[1], S[2], S[3]) +
               tg_mem.xchg[row_group][0][lane].f4[2]) +
              (tg_mem.xchg[row_group][1][lane].f4[2] +
               tg_mem.xchg[row_group][2][lane].f4[2]);
          if (full_cf) {
            s10 *= kScale2;
          } else {
            const bool v0 = (row0_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s10 = select(float4(kNegInf), s10 * kScale2,
                         v0 && (c1 < max_valid_kv) && (c1 <= qpos0));
          }
          tg_mem.xchg[row_group][1][lane].f4[2] = s10;
          m0_b = max(max(s10.x, s10.y), max(s10.z, s10.w));
        }
        {
          float4 s11 =
              (float4(S[4], S[5], S[6], S[7]) +
               tg_mem.xchg[row_group][0][lane].f4[3]) +
              (tg_mem.xchg[row_group][1][lane].f4[3] +
               tg_mem.xchg[row_group][2][lane].f4[3]);
          if (full_cf) {
            s11 *= kScale2;
          } else {
            const bool v1 = (row1_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s11 = select(float4(kNegInf), s11 * kScale2,
                         v1 && (c1 < max_valid_kv) && (c1 <= qpos1));
          }
          tg_mem.xchg[row_group][1][lane].f4[3] = s11;
          m1_b = max(max(s11.x, s11.y), max(s11.z, s11.w));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float fac0, fac1;
        {
          float4 s00 =
              (tg_mem.d0_s0[row_group][lane][0] +
               tg_mem.xchg[row_group][0][lane].f4[0]) +
              (tg_mem.xchg[row_group][1][lane].f4[0] +
               tg_mem.xchg[row_group][2][lane].f4[0]);
          if (full_cf) {
            s00 *= kScale2;
          } else {
            const bool v0 = (row0_p & 1u) != 0u;
            const uint4 c0 =
                uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s00 = select(float4(kNegInf), s00 * kScale2,
                         v0 && (c0 < max_valid_kv) && (c0 <= qpos0));
          }
          float m0 = max(m0_b, max(max(s00.x, s00.y), max(s00.z, s00.w)));
          m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));
          m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));
          const float nm0 = max(max_s0, m0);
          fac0 = (max_s0 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s0 - nm0);
          max_s0 = nm0;
          sum_s0 *= fac0;
          float4 p00 = full_cf
                           ? fast::exp2(s00 - nm0)
                           : select(fast::exp2(s00 - nm0), float4(0.0f),
                                    (nm0 <= kNegInfHalf) || (s00 <= kNegInfHalf));
          float ts0 = (p00.x + p00.y) + (p00.z + p00.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][0][lane].bcast.p00 = half4(p00);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          float4 s01 =
              (tg_mem.d0_s0[row_group][lane][1] +
               tg_mem.xchg[row_group][0][lane].f4[1]) +
              (tg_mem.xchg[row_group][1][lane].f4[1] +
               tg_mem.xchg[row_group][2][lane].f4[1]);
          if (full_cf) {
            s01 *= kScale2;
          } else {
            const bool v1 = (row1_p & 1u) != 0u;
            const uint4 c0 =
                uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s01 = select(float4(kNegInf), s01 * kScale2,
                         v1 && (c0 < max_valid_kv) && (c0 <= qpos1));
          }
          float m1 = max(m1_b, max(max(s01.x, s01.y), max(s01.z, s01.w)));
          m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));
          m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));
          const float nm1 = max(max_s1, m1);
          fac1 = (max_s1 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s1 - nm1);
          max_s1 = nm1;
          sum_s1 *= fac1;
          float4 p01 = full_cf
                           ? fast::exp2(s01 - nm1)
                           : select(fast::exp2(s01 - nm1), float4(0.0f),
                                    (nm1 <= kNegInfHalf) || (s01 <= kNegInfHalf));
          float ts1 = (p01.x + p01.y) + (p01.z + p01.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][0][lane].bcast.p01 = half4(p01);
          tg_mem.xchg[row_group][0][lane].bcast.fac = float2(fac0, fac1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s10 = tg_mem.xchg[row_group][1][lane].f4[2];
          float4 p10 = full_cf
                           ? fast::exp2(s10 - max_s0)
                           : select(fast::exp2(s10 - max_s0), float4(0.0f),
                                    (max_s0 <= kNegInfHalf) || (s10 <= kNegInfHalf));
          float ts0 = (p10.x + p10.y) + (p10.z + p10.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][0][lane].bcast.p10 = half4(p10);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s11 = tg_mem.xchg[row_group][1][lane].f4[3];
          float4 p11 = full_cf
                           ? fast::exp2(s11 - max_s1)
                           : select(fast::exp2(s11 - max_s1), float4(0.0f),
                                    (max_s1 <= kNegInfHalf) || (s11 <= kNegInfHalf));
          float ts1 = (p11.x + p11.y) + (p11.z + p11.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][0][lane].bcast.p11 = half4(p11);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    } else {
      // WM=3 (12.00 KiB <= 14.00 KiB): 0-ULP bit-exact with WM=2 via 4 x 32 B slots!
      float m0_a = kNegInf, m1_a = kNegInf;
      F8 S;
      #pragma clang loop unroll(disable)
      for (uint h2 = 0u; h2 < 2u; ++h2) {
        device const uchar *kp =
            cache_key + ulong(phys_page) * ulong(page_bytes_u32);
        device const ushort *k_cg =
            reinterpret_cast<device const ushort *>(kp + grp_code_off_u32);
        device const KvAffine4VecMeta *k_meta =
            reinterpret_cast<device const KvAffine4VecMeta *>(
                kp + p.key_value_heads * (p.page_tokens * 128u)) +
            ulong(kvh) * p.page_tokens;
        const uint row_off = h2 * 16u;
        const uint r0 = sub_off + row_off + uint(sm), r1 = r0 + 8u;
        const bool ok0 = (short(int(row_off) + int(sm)) < valid_k);
        const bool ok1 = (short(int(row_off) + int(sm) + 8) < valid_k);
        const half s0 = ok0 ? k_meta[r0].scale[aff_grp] : 0.0h;
        const half z0 = ok0 ? k_meta[r0].zero[aff_grp] : 0.0h;
        const half s1 = ok1 ? k_meta[r1].scale[aff_grp] : 0.0h;
        const half z1 = ok1 ? k_meta[r1].zero[aff_grp] : 0.0h;
        device const ushort *p0 = k_cg + (ok0 ? r0 : sub_off) * 32u + sn_u16;
        device const ushort *p1 = k_cg + (ok1 ? r1 : sub_off) * 32u + sn_u16;
        S = F8(0.0f);
        {
          const H8 k0 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 0u);
          const H8 k1 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 4u);
          nax_mma_16x16x32<half, true>(S, Q0, Q1, k0, k1);
        }
        {
          const H8 k2 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 8u);
          const H8 k3 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 12u);
          nax_mma_16x16x32<half, true>(S, Q2, Q3, k2, k3);
        }
        if (d_part > 0u) {
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[0] =
              float4(S[0], S[1], S[2], S[3]);
          tg_mem.xchg[row_group][d_part - 1u][lane].f4[1] =
              float4(S[4], S[5], S[6], S[7]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (h2 == 0u) {
          if (d_part == 0u) {
            {
              float4 s00 =
                  (float4(S[0], S[1], S[2], S[3]) +
                   tg_mem.xchg[row_group][0][lane].f4[0]) +
                  (tg_mem.xchg[row_group][1][lane].f4[0] +
                   tg_mem.xchg[row_group][2][lane].f4[0]);
              if (full_cf) {
                s00 *= kScale2;
              } else {
                const bool v0 = (row0_p & 1u) != 0u;
                const uint4 c0 =
                    uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
                s00 = select(float4(kNegInf), s00 * kScale2,
                             v0 && (c0 < max_valid_kv) && (c0 <= qpos0));
              }
              tg_mem.xchg[row_group][3][lane].f4[0] = s00;
              m0_a = max(max(s00.x, s00.y), max(s00.z, s00.w));
            }
            {
              float4 s01 =
                  (float4(S[4], S[5], S[6], S[7]) +
                   tg_mem.xchg[row_group][0][lane].f4[1]) +
                  (tg_mem.xchg[row_group][1][lane].f4[1] +
                   tg_mem.xchg[row_group][2][lane].f4[1]);
              if (full_cf) {
                s01 *= kScale2;
              } else {
                const bool v1 = (row1_p & 1u) != 0u;
                const uint4 c0 =
                    uint4(kb_col_base + uint(sn)) + uint4(0u, 1u, 2u, 3u);
                s01 = select(float4(kNegInf), s01 * kScale2,
                             v1 && (c0 < max_valid_kv) && (c0 <= qpos1));
              }
              tg_mem.xchg[row_group][3][lane].f4[1] = s01;
              m1_a = max(max(s01.x, s01.y), max(s01.z, s01.w));
            }
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
        }
      }

      if (d_part == 0u) {
        float m0_b, m1_b;
        {
          float4 s10 =
              (float4(S[0], S[1], S[2], S[3]) +
               tg_mem.xchg[row_group][0][lane].f4[0]) +
              (tg_mem.xchg[row_group][1][lane].f4[0] +
               tg_mem.xchg[row_group][2][lane].f4[0]);
          if (full_cf) {
            s10 *= kScale2;
          } else {
            const bool v0 = (row0_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s10 = select(float4(kNegInf), s10 * kScale2,
                         v0 && (c1 < max_valid_kv) && (c1 <= qpos0));
          }
          tg_mem.xchg[row_group][2][lane].f4[0] = s10;
          m0_b = max(max(s10.x, s10.y), max(s10.z, s10.w));
        }
        {
          float4 s11 =
              (float4(S[4], S[5], S[6], S[7]) +
               tg_mem.xchg[row_group][0][lane].f4[1]) +
              (tg_mem.xchg[row_group][1][lane].f4[1] +
               tg_mem.xchg[row_group][2][lane].f4[1]);
          if (full_cf) {
            s11 *= kScale2;
          } else {
            const bool v1 = (row1_p & 1u) != 0u;
            const uint4 c1 =
                uint4(kb_col_base + 16u + uint(sn)) + uint4(0u, 1u, 2u, 3u);
            s11 = select(float4(kNegInf), s11 * kScale2,
                         v1 && (c1 < max_valid_kv) && (c1 <= qpos1));
          }
          tg_mem.xchg[row_group][2][lane].f4[1] = s11;
          m1_b = max(max(s11.x, s11.y), max(s11.z, s11.w));
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        float fac0, fac1;
        {
          const float4 s00 = tg_mem.xchg[row_group][3][lane].f4[0];
          float m0 = max(m0_b, m0_a);
          m0 = max(m0, simd_shuffle_xor(m0, ushort(1)));
          m0 = max(m0, simd_shuffle_xor(m0, ushort(8)));
          const float nm0 = max(max_s0, m0);
          fac0 = (max_s0 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s0 - nm0);
          max_s0 = nm0;
          sum_s0 *= fac0;
          float4 p00 = full_cf
                           ? fast::exp2(s00 - nm0)
                           : select(fast::exp2(s00 - nm0), float4(0.0f),
                                    (nm0 <= kNegInfHalf) || (s00 <= kNegInfHalf));
          float ts0 = (p00.x + p00.y) + (p00.z + p00.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][0][lane].bcast.p0 = half4(p00);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s01 = tg_mem.xchg[row_group][3][lane].f4[1];
          float m1 = max(m1_b, m1_a);
          m1 = max(m1, simd_shuffle_xor(m1, ushort(1)));
          m1 = max(m1, simd_shuffle_xor(m1, ushort(8)));
          const float nm1 = max(max_s1, m1);
          fac1 = (max_s1 <= kNegInfHalf) ? 0.0f : fast::exp2(max_s1 - nm1);
          max_s1 = nm1;
          sum_s1 *= fac1;
          float4 p01 = full_cf
                           ? fast::exp2(s01 - nm1)
                           : select(fast::exp2(s01 - nm1), float4(0.0f),
                                    (nm1 <= kNegInfHalf) || (s01 <= kNegInfHalf));
          float ts1 = (p01.x + p01.y) + (p01.z + p01.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][0][lane].bcast.p1 = half4(p01);
          tg_mem.xchg[row_group][0][lane].bcast.fac = float2(fac0, fac1);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s10 = tg_mem.xchg[row_group][2][lane].f4[0];
          float4 p10 = full_cf
                           ? fast::exp2(s10 - max_s0)
                           : select(fast::exp2(s10 - max_s0), float4(0.0f),
                                    (max_s0 <= kNegInfHalf) || (s10 <= kNegInfHalf));
          float ts0 = (p10.x + p10.y) + (p10.z + p10.w);
          ts0 += simd_shuffle_xor(ts0, ushort(1));
          ts0 += simd_shuffle_xor(ts0, ushort(8));
          sum_s0 += ts0;
          tg_mem.xchg[row_group][1][lane].bcast.p0 = half4(p10);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        {
          const float4 s11 = tg_mem.xchg[row_group][2][lane].f4[1];
          float4 p11 = full_cf
                           ? fast::exp2(s11 - max_s1)
                           : select(fast::exp2(s11 - max_s1), float4(0.0f),
                                    (max_s1 <= kNegInfHalf) || (s11 <= kNegInfHalf));
          float ts1 = (p11.x + p11.y) + (p11.z + p11.w);
          ts1 += simd_shuffle_xor(ts1, ushort(1));
          ts1 += simd_shuffle_xor(ts1, ushort(8));
          sum_s1 += ts1;
          tg_mem.xchg[row_group][1][lane].bcast.p1 = half4(p11);
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // PV Half 0 & Half 1 via compact non-unrolled loop (0 live-range overlap!)
    {
      const float2 fc = tg_mem.xchg[row_group][0][lane].bcast.fac;
      const F8 fv(fc.x, fc.x, fc.x, fc.x, fc.y, fc.y, fc.y, fc.y);
      O0 *= fv;
      O1 *= fv;
      O2 *= fv;
      O3 *= fv;

      #pragma clang loop unroll(disable)
      for (uint h2 = 0u; h2 < 2u; ++h2) {
        half4 hp0, hp1;
        if constexpr (WM == 2u) {
          hp0 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p00
                           : tg_mem.xchg[row_group][0][lane].bcast.p10;
          hp1 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p01
                           : tg_mem.xchg[row_group][0][lane].bcast.p11;
        } else {
          hp0 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p0
                           : tg_mem.xchg[row_group][1][lane].bcast.p0;
          hp1 = (h2 == 0u) ? tg_mem.xchg[row_group][0][lane].bcast.p1
                           : tg_mem.xchg[row_group][1][lane].bcast.p1;
        }
        const H8 P_cur(hp0.x, hp0.y, hp0.z, hp0.w, hp1.x, hp1.y, hp1.z, hp1.w);
        device const uchar *vp =
            cache_value + ulong(phys_page) * ulong(page_bytes_u32);
        device const ushort *v_cg =
            reinterpret_cast<device const ushort *>(vp + grp_code_off_u32);
        device const KvAffine4VecMeta *v_meta =
            reinterpret_cast<device const KvAffine4VecMeta *>(
                vp + p.key_value_heads * (p.page_tokens * 128u)) +
            ulong(kvh) * p.page_tokens;
        const uint row_off = h2 * 16u;
        const uint r0 = sub_off + row_off + uint(sm), r1 = r0 + 8u;
        const bool ok0 = (short(int(row_off) + int(sm)) < valid_k);
        const bool ok1 = (short(int(row_off) + int(sm) + 8) < valid_k);
        const half s0 = ok0 ? v_meta[r0].scale[aff_grp] : 0.0h;
        const half z0 = ok0 ? v_meta[r0].zero[aff_grp] : 0.0h;
        const half s1 = ok1 ? v_meta[r1].scale[aff_grp] : 0.0h;
        const half z1 = ok1 ? v_meta[r1].zero[aff_grp] : 0.0h;
        device const ushort *p0 = v_cg + (ok0 ? r0 : sub_off) * 32u + sn_u16;
        device const ushort *p1 = v_cg + (ok1 ? r1 : sub_off) * 32u + sn_u16;
        {
          const H8 v0 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 0u);
          const H8 v1 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 4u);
          nax_mma_16x32x16_nn<half>(O0, O1, P_cur, v0, v1);
        }
        {
          const H8 v2 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 8u);
          const H8 v3 = dequant_ptr_aff4_h8(p0, p1, s0, z0, s1, z1, 12u);
          nax_mma_16x32x16_nn<half>(O2, O3, P_cur, v2, v3);
        }
      }
    }
    if (kb + 1u < nk_blocks) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  #undef AFF4_BK32_LOAD_Q

  // Broadcast final (rcp0, rcp1) from d_part == 0 to d_part != 0.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rcp0 = 0.0f;
  float rcp1 = 0.0f;
  if (d_part == 0u) {
    rcp0 = (sum_s0 > 0.0f) ? (1.0f / sum_s0) : 0.0f;
    rcp1 = (sum_s1 > 0.0f) ? (1.0f / sum_s1) : 0.0f;
    tg_mem.xchg[row_group][0][lane].bcast.fac = float2(rcp0, rcp1);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (d_part != 0u) {
    const float2 r = tg_mem.xchg[row_group][0][lane].bcast.fac;
    rcp0 = r.x;
    rcp1 = r.y;
  }

  const bool ep_qv0 = (row0_p & 1u) != 0u;
  const bool ep_qv1 = (row1_p & 1u) != 0u;
  if (!ep_qv0 && !ep_qv1) {
    return;
  }
  const uint ep_tok0 = row0_p >> 8u;
  const uint ep_qh0 = (row0_p >> 1u) & 0x7Fu;
  const uint ep_tok1 = row1_p >> 8u;
  const uint ep_qh1 = (row1_p >> 1u) & 0x7Fu;

  device const bfloat *g_row0 =
      gate + ulong(ep_tok0) * p.gate_token_stride +
      ulong(ep_qh0) * p.gate_head_stride + d_offset + uint(sn);
  device const bfloat *g_row1 =
      gate + ulong(ep_tok1) * p.gate_token_stride +
      ulong(ep_qh1) * p.gate_head_stride + d_offset + uint(sn);
  device bfloat *o_row0 =
      output + (ulong(ep_tok0) * p.query_heads + ep_qh0) * BD + d_offset +
      uint(sn);
  device bfloat *o_row1 =
      output + (ulong(ep_tok1) * p.query_heads + ep_qh1) * BD + d_offset +
      uint(sn);

  #define AFF4_BK32_STORE_O(OREG, OFF_C)                                       \
    do {                                                                       \
      if (ep_qv0) {                                                            \
        ushort4 gu0 =                                                          \
            *reinterpret_cast<device const ushort4 *>(g_row0 + (OFF_C));       \
        if (!warp_full_rows) {                                                 \
          gu0 = gqa_clean_bf4_u16(gu0);                                        \
        }                                                                      \
        const float4 g0_f = float4(as_type<bfloat4>(gu0));                     \
        const float4 o0_f =                                                    \
            (float4((OREG)[0], (OREG)[1], (OREG)[2], (OREG)[3]) * rcp0) /      \
            (float4(1.0f) + exp(-g0_f));                                       \
        *reinterpret_cast<device bfloat4 *>(o_row0 + (OFF_C)) = bfloat4(o0_f); \
      }                                                                        \
      if (ep_qv1) {                                                            \
        ushort4 gu1 =                                                          \
            *reinterpret_cast<device const ushort4 *>(g_row1 + (OFF_C));       \
        if (!warp_full_rows) {                                                 \
          gu1 = gqa_clean_bf4_u16(gu1);                                        \
        }                                                                      \
        const float4 g1_f = float4(as_type<bfloat4>(gu1));                     \
        const float4 o1_f =                                                    \
            (float4((OREG)[4], (OREG)[5], (OREG)[6], (OREG)[7]) * rcp1) /      \
            (float4(1.0f) + exp(-g1_f));                                       \
        *reinterpret_cast<device bfloat4 *>(o_row1 + (OFF_C)) = bfloat4(o1_f); \
      }                                                                        \
    } while (false)
  AFF4_BK32_STORE_O(O0, 0u);
  AFF4_BK32_STORE_O(O1, 16u);
  AFF4_BK32_STORE_O(O2, 32u);
  AFF4_BK32_STORE_O(O3, 48u);
  #undef AFF4_BK32_STORE_O
}

} // namespace affine4_kv

template <bool IsBatched, typename ParamsT>
AFFINE4_ALWAYS_INLINE
static inline void paged_flash_prefill_affine4_hd256_core(
    device const bfloat *query,
    device const bfloat *gate,
    device const uchar *cache_key,
    device const uchar *cache_value,
    device bfloat *output,
    device const uint *page_table,
    device const uint *positions,
    constant ParamsT &p,
    threadgroup affine4_kv::FlashGqaBk32TgStorageHalf<2u> &tg_mem,
    ushort lane,
    ushort sg,
    uint3 tg) {
  constexpr uint BQ = 64u;
  constexpr uint BK = 32u;

  uint tokens = 0u;
  if constexpr (IsBatched) {
    tokens = p.widths[tg.z];
  } else {
    tokens = p.tokens;
  }
  if (p.query_heads == 0u || p.key_value_heads == 0u || tokens == 0u) {
    return;
  }
  const uint n_q_blocks = (tokens + BQ - 1u) / BQ;
  if (tg.y / p.query_heads >= n_q_blocks) {
    return;
  }

  #pragma clang loop unroll(disable)
  for (uint sub = 0u; sub < 2u; ++sub) {
    uint row_off = 0u;
    uint cur_tokens = 0u;
    if constexpr (IsBatched) {
      row_off = p.row_offsets[tg.z];
      cur_tokens = p.widths[tg.z];
    } else {
      cur_tokens = p.tokens;
    }
    const uint nqb = (cur_tokens + BQ - 1u) / BQ;
    const uint q_blk_rev = tg.y / p.query_heads;
    const uint qh = tg.y - q_blk_rev * p.query_heads;
    const uint q_blk = (nqb - 1u) - q_blk_rev;
    const uint sub_base = q_blk * BQ + sub * 32u;
    if (sub_base >= cur_tokens) {
      break;
    }
    if (sub > 0u) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const uint repeat = p.query_heads / p.key_value_heads;
    const uint kvh = qh / repeat;
    const uint flat_tid = uint(sg) * 32u + uint(lane);
    const uint row_group = uint(sg) >> 2u;
    const short2 sc = affine4_kv::Frag16x16Aff4::get_coord(lane);

    const uint active_rows = min(32u, cur_tokens - sub_base);
    threadgroup uint *tg_q_pos =
        reinterpret_cast<threadgroup uint *>(tg_mem.xchg);
    if (flat_tid < active_rows) {
      tg_q_pos[flat_tid] = positions[row_off + sub_base + flat_tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint max_valid_kv = 0u;
    uint min_q_pos = 0xffffffffu;
    for (uint r = 0u; r < active_rows; ++r) {
      const uint pos = tg_q_pos[r];
      max_valid_kv = max(max_valid_kv, pos + 1u);
      min_q_pos = min(min_q_pos, pos);
    }
    const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
    const uint kb_min_causal = min_q_pos / BK;

    const uint tm = row_group * 16u;
    const bool warp_full_rows = (int(active_rows) - int(tm) >= 16);
    const uint r0 = tm + uint(sc.y);
    const uint r1 = tm + 8u + uint(sc.y);
    const bool qv0 = (r0 < active_rows);
    const bool qv1 = (r1 < active_rows);
    const uint tok0 = row_off + (qv0 ? (sub_base + r0) : sub_base);
    const uint tok1 = row_off + (qv1 ? (sub_base + r1) : sub_base);
    const uint qpos0 = qv0 ? tg_q_pos[r0] : 0u;
    const uint qpos1 = qv1 ? tg_q_pos[r1] : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    affine4_kv::flash_bk32_wn4_affine4_hd256_core<2u, false, ParamsT>(
        query, gate, cache_key, cache_value, output, page_table, p, tg_mem,
        kvh, nk_blocks, max_valid_kv, kb_min_causal, tok0, qh,
        qpos0, qv0, tok1, qh, qpos1, qv1, warp_full_rows, lane, sg);
  }
}

[[max_total_threads_per_threadgroup(256)]]
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
  alignas(16) threadgroup affine4_kv::FlashGqaBk32TgStorageHalf<2u> tg_mem;
  paged_flash_prefill_affine4_hd256_core<false>(
      query, gate, cache_key, cache_value, output, page_table, positions, p,
      tg_mem, lane, sg, tg);
}

// Batched B=2..8 Zero-Spill WN=4 Affine4Group128 Flash-Prefill (HD == 256)
// Grid (threads): width = 256, height = ceil(max_w/64) * QH, depth = B
[[max_total_threads_per_threadgroup(256)]]
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
  device const uint *slot_pt =
      affine4_kv::select_pt8(slot, pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7);
  alignas(16) threadgroup affine4_kv::FlashGqaBk32TgStorageHalf<2u> tg_mem;
  paged_flash_prefill_affine4_hd256_core<true>(
      query, gate, cache_key, cache_value, output, slot_pt, positions, p,
      tg_mem, lane, sg, tg);
}

// ============================================================================
// 6. Preserved Non-Split (!split) and HD == 128 Compatibility Paths
// ============================================================================
[[max_total_threads_per_threadgroup(64)]]
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
  threadgroup float tg_dot[2u];
  const uint repeat = params.query_heads / params.key_value_heads;
  if (repeat == 0u || repeat > 16u) {
    return;
  }
  const uint bq = 16u / repeat;
  const uint kvh = tg.y % params.key_value_heads;
  const uint t_base = (tg.y / params.key_value_heads) * bq;
  if (t_base >= params.tokens) {
    return;
  }
  affine4_kv::run_affine4_compat_core<256u, 2u, false>(
      query, gate, cache_key, cache_value, output, nullptr, nullptr, nullptr,
      page_table, positions, params.query_heads, params.key_value_heads,
      params.tokens, params.page_tokens, params.query_token_stride,
      params.query_head_stride, params.gate_token_stride,
      params.gate_head_stride, 1u, 0u, 0u, t_base, bq, kvh, tg_dot, lane, sg);
}

[[max_total_threads_per_threadgroup(32)]]
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
  threadgroup float tg_dot[1u];
  const uint repeat = params.query_heads / params.key_value_heads;
  if (repeat == 0u || repeat > 16u) {
    return;
  }
  const uint bq = 16u / repeat;
  const uint kvh = tg.y % params.key_value_heads;
  const uint t_base = (tg.y / params.key_value_heads) * bq;
  if (t_base >= params.tokens) {
    return;
  }
  affine4_kv::run_affine4_compat_core<128u, 1u, false>(
      query, gate, cache_key, cache_value, output, nullptr, nullptr, nullptr,
      page_table, positions, params.query_heads, params.key_value_heads,
      params.tokens, params.page_tokens, params.query_token_stride,
      params.query_head_stride, params.gate_token_stride,
      params.gate_head_stride, 1u, 0u, 0u, t_base, bq, kvh, tg_dot, lane, sg);
}

[[max_total_threads_per_threadgroup(32)]]
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
  threadgroup float tg_dot[1u];
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
  affine4_kv::run_affine4_compat_core<128u, 1u, true>(
      query, nullptr, cache_key, cache_value, nullptr, partial_max, partial_sum,
      partial_output, page_table, positions, p.query_heads, p.key_value_heads,
      p.tokens, p.page_tokens, p.query_token_stride, p.query_head_stride, 0u,
      0u, p.split_count, p.split_tokens, split, t_base, bq, kvh, tg_dot, lane,
      sg);
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

// ============================================================================
// 7. WP3.2 BK=32, WN=4 Direct-L1 Zero-Spill GQA Flash-Prefill
//    (Affine4Group128 KV, HD == 256):
//    - Flattened (q_tok, h_sub) packing: 100% row fill on G=6 (27B) & G=8 (35B)!
//    - LPT Reverse m_blk + kvh-minor scheduling (0 atomics, ICB-safe)
//    - WM=2: 14.00 KiB exact (256 thr, 2 TGs/core, 0 B spills, 63.9–65.5 TF!)
//    - WM=3: 12.00 KiB exact (384 thr, <= 14.00 KiB, 0 B spills!)
//    - 0-ULP bit-exact parity across WM=2, WM=3, and single-head hd256!
// ============================================================================
namespace affine4_kv {

template <uint WM>
AFFINE4_ALWAYS_INLINE
static inline void paged_flash_prefill_gqa_wn4_affine4_hd256_impl(
    device const bfloat *query, device const bfloat *gate,
    device const uchar *cache_key, device const uchar *cache_value,
    device bfloat *output, device const uint *page_table,
    device const uint *positions, constant PagedAttentionParams &p,
    threadgroup FlashGqaBk32TgStorageHalf<WM> &tg_mem, ushort lane, ushort sg,
    uint3 tg) {
  constexpr uint BK = 32u;
  constexpr uint WN = 4u;
  constexpr uint M_TILE = WM * 16u;

  if (p.tokens == 0u || p.query_heads == 0u || p.key_value_heads == 0u) {
    return;
  }
  const uint repeat = p.query_heads / p.key_value_heads;
  if (repeat == 0u || repeat > 16u) {
    return;
  }
  const uint m_total = p.tokens * repeat;
  const uint n_m_blocks = (m_total + M_TILE - 1u) / M_TILE;

  // LPT Reverse m_blk + kvh-minor mapping (longest causal blocks launch first).
  const uint m_blk_rev = tg.y / p.key_value_heads;
  const uint kvh = tg.y - m_blk_rev * p.key_value_heads;
  if (m_blk_rev >= n_m_blocks) {
    return;
  }
  const uint m_blk = (n_m_blocks - 1u) - m_blk_rev;
  const uint m_base = m_blk * M_TILE;
  const uint active_m_rows = min(M_TILE, m_total - m_base);

  const uint first_tok = m_base / repeat;
  const uint last_tok = (m_base + active_m_rows - 1u) / repeat;
  const uint num_toks = last_tok - first_tok + 1u;

  const uint flat_tid = uint(sg) * 32u + uint(lane);
  threadgroup uint *tg_q_pos =
      reinterpret_cast<threadgroup uint *>(tg_mem.xchg);
  if (flat_tid < num_toks) {
    tg_q_pos[flat_tid] = positions[first_tok + flat_tid];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  uint max_valid_kv = 0u;
  uint min_q_pos = 0xffffffffu;
  for (uint r = 0u; r < num_toks; ++r) {
    const uint pos = tg_q_pos[r];
    max_valid_kv = max(max_valid_kv, pos + 1u);
    min_q_pos = min(min_q_pos, pos);
  }
  const uint nk_blocks = (max_valid_kv + BK - 1u) / BK;
  const uint kb_min_causal = min_q_pos / BK;

  const uint row_group = uint(sg) / WN;
  const uint tm = row_group * 16u;
  const bool warp_full_rows = (int(active_m_rows) - int(tm) >= 16);
  const short2 sc = Frag16x16Aff4::get_coord(lane);

  const uint m_off_base = m_base - first_tok * repeat; // < repeat <= 16
  const uint local_r0 = tm + uint(sc.y);
  const uint local_r1 = tm + 8u + uint(sc.y);
  const bool qv0 = (local_r0 < active_m_rows);
  const bool qv1 = (local_r1 < active_m_rows);
  uint tok_rel0 = 0u, g0 = 0u, tok_rel1 = 0u, g1 = 0u;
  if (qv0) {
    fast_divmod_g(m_off_base + local_r0, repeat, tok_rel0, g0);
  }
  if (qv1) {
    fast_divmod_g(m_off_base + local_r1, repeat, tok_rel1, g1);
  }
  const uint tok0 = first_tok + tok_rel0;
  const uint tok1 = first_tok + tok_rel1;
  const uint qh0 = kvh * repeat + g0;
  const uint qh1 = kvh * repeat + g1;
  const uint qpos0 = qv0 ? tg_q_pos[tok_rel0] : 0u;
  const uint qpos1 = qv1 ? tg_q_pos[tok_rel1] : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  flash_bk32_wn4_affine4_hd256_core<WM>(
      query, gate, cache_key, cache_value, output, page_table, p, tg_mem,
      kvh, nk_blocks, max_valid_kv, kb_min_causal, tok0, qh0, qpos0,
      qv0, tok1, qh1, qpos1, qv1, warp_full_rows, lane, sg);
}

} // namespace affine4_kv

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm2(
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
  alignas(16) threadgroup affine4_kv::FlashGqaBk32TgStorageHalf<2u> tg_mem;
  affine4_kv::paged_flash_prefill_gqa_wn4_affine4_hd256_impl<2u>(
      query, gate, cache_key, cache_value, output, page_table, positions, p,
      tg_mem, lane, sg, tg);
}

[[max_total_threads_per_threadgroup(384)]]
kernel void omlx_m5_paged_flash_prefill_gqa_bk16_wn4_affine4_g128_hd256_wm3(
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
  alignas(16) threadgroup affine4_kv::FlashGqaBk32TgStorageHalf<3u> tg_mem;
  affine4_kv::paged_flash_prefill_gqa_wn4_affine4_hd256_impl<3u>(
      query, gate, cache_key, cache_value, output, page_table, positions, p,
      tg_mem, lane, sg, tg);
}
