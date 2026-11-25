#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct MtpSaveTrunkParams {
  uint hidden;
  uint src_row;
  float epsilon;
  float weight_offset;
};

struct MtpStemNormsParams {
  uint hidden;
  uint rows;
  uint cond_mode;
  uint write_concat;
  uint trunk_row_base;
  float epsilon;
  float target_norm_offset;
  float emb_norm_offset;
  float hid_norm_offset;
};

struct MtpFcDualParams {
  uint n;
  uint h;
  uint rows;
  uint reserved;
};

struct MtpQkNormRopeParams {
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
  float q_weight_offset;
  float k_weight_offset;
};

struct MtpLatchParams {
  uint hidden;
  uint src_row;
  uint step_index;
  uint sampled_mode;
  uint write_chain;
  uint norm_from_sq;
  uint sq_slot;
  float inv_k;
  float eps;
  float w_off;
};

inline float mtp_simd_reduce1(float v) {
  v += simd_shuffle_xor(v, 1u);
  v += simd_shuffle_xor(v, 2u);
  v += simd_shuffle_xor(v, 4u);
  v += simd_shuffle_xor(v, 8u);
  v += simd_shuffle_xor(v, 16u);
  return v;
}

kernel void omlx_m5_mtp_save_trunk_row(
    device const bfloat *trunk_hidden [[buffer(0)]],
    device const bfloat *target_final_norm [[buffer(1)]],
    device bfloat *pending_trunk_normed [[buffer(2)]],
    constant MtpSaveTrunkParams &params [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  constexpr uint kThreads = 128u;
  constexpr uint kSimdgroups = 4u;
  threadgroup float simd_sums[kSimdgroups];

  device const bfloat *src =
      trunk_hidden + ulong(params.src_row) * params.hidden;

  if (params.hidden == 5120u) {
    device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
    device const bfloat4 *wv =
        reinterpret_cast<device const bfloat4 *>(target_final_norm);
    device bfloat4 *ov =
        reinterpret_cast<device bfloat4 *>(pending_trunk_normed);
    bfloat4 reg_x[10];
    float sq = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 x4 = sv[v];
      reg_x[i] = x4;
      const float4 xf = float4(x4);
      sq = fma(xf.x, xf.x, sq);
      sq = fma(xf.y, xf.y, sq);
      sq = fma(xf.z, xf.z, sq);
      sq = fma(xf.w, xf.w, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
      simd_sums[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total * (1.0f / 5120.0f) + params.epsilon);
    const float4 off4 = float4(params.weight_offset);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = tid + i * kThreads;
      ov[v] = bfloat4(float4(reg_x[i]) * inv * (float4(wv[v]) + off4));
    }
    return;
  }

  if (params.hidden == 2048u) {
    device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
    device const bfloat4 *wv =
        reinterpret_cast<device const bfloat4 *>(target_final_norm);
    device bfloat4 *ov =
        reinterpret_cast<device bfloat4 *>(pending_trunk_normed);
    bfloat4 reg_x[4];
    float sq = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 x4 = sv[v];
      reg_x[i] = x4;
      const float4 xf = float4(x4);
      sq = fma(xf.x, xf.x, sq);
      sq = fma(xf.y, xf.y, sq);
      sq = fma(xf.z, xf.z, sq);
      sq = fma(xf.w, xf.w, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
      simd_sums[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total * (1.0f / 2048.0f) + params.epsilon);
    const float4 off4 = float4(params.weight_offset);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = tid + i * kThreads;
      ov[v] = bfloat4(float4(reg_x[i]) * inv * (float4(wv[v]) + off4));
    }
    return;
  }

  float sq = 0.0f;
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const float v = float(src[col]);
    sq = fma(v, v, sq);
  }
  sq = simd_sum(sq);
  if (lane == 0u) {
    simd_sums[sg] = sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv = rsqrt(total / float(params.hidden) + params.epsilon);
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const float norm_v = float(src[col]) * inv *
                         (float(target_final_norm[col]) + params.weight_offset);
    pending_trunk_normed[col] = bfloat(norm_v);
  }
}

// Fused MTP stem RMSNorms across tg.z in {0 (embedding), 1 (conditioning hidden)}.
// When cond_mode == 0, normalizes trunk_hidden[row] with target_final_norm first
// (matching HEAD_HIDDEN_POST_NORM = True in BF16), then normalizes with
// pre_fc_norm_hidden without writing intermediate trunk_normed to global memory.
// When cond_mode == 1, row 0 reads pending_trunk_normed (already post-final-norm)
// while rows 1..rows-1 normalize trunk_hidden[row - 1] via target_final_norm first.
// When cond_mode == 2, row 0 reads chain_hidden[0] (already post-mtp.norm).
kernel void omlx_m5_mtp_stem_norms(
    device const bfloat *embed_out [[buffer(0)]],
    device const bfloat *trunk_hidden [[buffer(1)]],
    device const bfloat *pending_trunk_normed [[buffer(2)]],
    device const bfloat *chain_hidden [[buffer(3)]],
    device const bfloat *target_final_norm [[buffer(4)]],
    device const bfloat *pre_fc_norm_embedding [[buffer(5)]],
    device const bfloat *pre_fc_norm_hidden [[buffer(6)]],
    device bfloat *stem_e [[buffer(7)]],
    device bfloat *stem_h [[buffer(8)]],
    device bfloat *stem_concat [[buffer(9)]],
    constant MtpStemNormsParams &params [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kThreads = 128u;
  constexpr uint kSimdgroups = 4u;
  const uint row = tg.y;
  if (row >= params.rows) {
    return;
  }

  threadgroup float simd_sums[kSimdgroups];

  if (tg.z == 0u) {
    device const bfloat *in_row = embed_out + ulong(row) * params.hidden;
    device bfloat *out_row = stem_e + ulong(row) * params.hidden;
    device bfloat *cat_row =
        stem_concat + ulong(row) * (2u * params.hidden);

    if (params.hidden == 5120u) {
      device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
      device const bfloat4 *wv =
          reinterpret_cast<device const bfloat4 *>(pre_fc_norm_embedding);
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
      device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
      bfloat4 reg_x[10];
      float sq = 0.0f;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 10u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 x4 = iv[v];
        reg_x[i] = x4;
        const float4 xf = float4(x4);
        sq = fma(xf.x, xf.x, sq);
        sq = fma(xf.y, xf.y, sq);
        sq = fma(xf.z, xf.z, sq);
        sq = fma(xf.w, xf.w, sq);
      }
      sq = simd_sum(sq);
      if (lane == 0u) {
        simd_sums[sg] = sq;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float total =
          ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
      const float inv = rsqrt(total * (1.0f / 5120.0f) + params.epsilon);
      const float4 off4 = float4(params.emb_norm_offset);
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 10u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 val =
            bfloat4(float4(reg_x[i]) * inv * (float4(wv[v]) + off4));
        ov[v] = val;
        if (params.write_concat != 0u) {
          cv[v] = val;
        }
      }
      return;
    }

    if (params.hidden == 2048u) {
      device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
      device const bfloat4 *wv =
          reinterpret_cast<device const bfloat4 *>(pre_fc_norm_embedding);
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
      device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
      bfloat4 reg_x[4];
      float sq = 0.0f;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 4u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 x4 = iv[v];
        reg_x[i] = x4;
        const float4 xf = float4(x4);
        sq = fma(xf.x, xf.x, sq);
        sq = fma(xf.y, xf.y, sq);
        sq = fma(xf.z, xf.z, sq);
        sq = fma(xf.w, xf.w, sq);
      }
      sq = simd_sum(sq);
      if (lane == 0u) {
        simd_sums[sg] = sq;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float total =
          ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
      const float inv = rsqrt(total * (1.0f / 2048.0f) + params.epsilon);
      const float4 off4 = float4(params.emb_norm_offset);
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 4u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 val =
            bfloat4(float4(reg_x[i]) * inv * (float4(wv[v]) + off4));
        ov[v] = val;
        if (params.write_concat != 0u) {
          cv[v] = val;
        }
      }
      return;
    }

    float sq = 0.0f;
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const float v = float(in_row[col]);
      sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
      simd_sums[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total / float(params.hidden) + params.epsilon);
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const bfloat val =
          bfloat(float(in_row[col]) * inv *
                 (float(pre_fc_norm_embedding[col]) + params.emb_norm_offset));
      out_row[col] = val;
      if (params.write_concat != 0u) {
        cat_row[col] = val;
      }
    }
    return;
  }

  device bfloat *out_row = stem_h + ulong(row) * params.hidden;
  device bfloat *cat_row =
      stem_concat + ulong(row) * (2u * params.hidden) + params.hidden;

  const bool already_post_norm =
      (params.cond_mode == 2u) || (params.cond_mode == 1u && row == 0u);

  if (already_post_norm) {
    device const bfloat *src =
        (params.cond_mode == 2u) ? (chain_hidden + ulong(row) * params.hidden)
                                 : pending_trunk_normed;
    if (params.hidden == 5120u) {
      device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
      device const bfloat4 *wv =
          reinterpret_cast<device const bfloat4 *>(pre_fc_norm_hidden);
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
      device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
      bfloat4 reg_x[10];
      float sq = 0.0f;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 10u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 x4 = sv[v];
        reg_x[i] = x4;
        const float4 xf = float4(x4);
        sq = fma(xf.x, xf.x, sq);
        sq = fma(xf.y, xf.y, sq);
        sq = fma(xf.z, xf.z, sq);
        sq = fma(xf.w, xf.w, sq);
      }
      sq = simd_sum(sq);
      if (lane == 0u) {
        simd_sums[sg] = sq;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float total =
          ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
      const float inv = rsqrt(total * (1.0f / 5120.0f) + params.epsilon);
      const float4 off4 = float4(params.hid_norm_offset);
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 10u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 val =
            bfloat4(float4(reg_x[i]) * inv * (float4(wv[v]) + off4));
        ov[v] = val;
        if (params.write_concat != 0u) {
          cv[v] = val;
        }
      }
      return;
    }
    if (params.hidden == 2048u) {
      device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
      device const bfloat4 *wv =
          reinterpret_cast<device const bfloat4 *>(pre_fc_norm_hidden);
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
      device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
      bfloat4 reg_x[4];
      float sq = 0.0f;
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 4u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 x4 = sv[v];
        reg_x[i] = x4;
        const float4 xf = float4(x4);
        sq = fma(xf.x, xf.x, sq);
        sq = fma(xf.y, xf.y, sq);
        sq = fma(xf.z, xf.z, sq);
        sq = fma(xf.w, xf.w, sq);
      }
      sq = simd_sum(sq);
      if (lane == 0u) {
        simd_sums[sg] = sq;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float total =
          ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
      const float inv = rsqrt(total * (1.0f / 2048.0f) + params.epsilon);
      const float4 off4 = float4(params.hid_norm_offset);
#pragma clang loop unroll(full)
      for (uint i = 0u; i < 4u; ++i) {
        const uint v = tid + i * kThreads;
        const bfloat4 val =
            bfloat4(float4(reg_x[i]) * inv * (float4(wv[v]) + off4));
        ov[v] = val;
        if (params.write_concat != 0u) {
          cv[v] = val;
        }
      }
      return;
    }
    float sq = 0.0f;
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const float v = float(src[col]);
      sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
      simd_sums[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total / float(params.hidden) + params.epsilon);
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const bfloat val =
          bfloat(float(src[col]) * inv *
                 (float(pre_fc_norm_hidden[col]) + params.hid_norm_offset));
      out_row[col] = val;
      if (params.write_concat != 0u) {
        cat_row[col] = val;
      }
    }
    return;
  }

  const uint trunk_row =
      params.trunk_row_base + ((params.cond_mode == 1u) ? (row - 1u) : row);
  device const bfloat *src =
      trunk_hidden + ulong(trunk_row) * params.hidden;

  if (params.hidden == 5120u) {
    device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
    device const bfloat4 *twv =
        reinterpret_cast<device const bfloat4 *>(target_final_norm);
    device const bfloat4 *hwv =
        reinterpret_cast<device const bfloat4 *>(pre_fc_norm_hidden);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
    device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
    bfloat4 reg_x[10];
    float sq1 = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 x4 = sv[v];
      reg_x[i] = x4;
      const float4 xf = float4(x4);
      sq1 = fma(xf.x, xf.x, sq1);
      sq1 = fma(xf.y, xf.y, sq1);
      sq1 = fma(xf.z, xf.z, sq1);
      sq1 = fma(xf.w, xf.w, sq1);
    }
    sq1 = simd_sum(sq1);
    if (lane == 0u) {
      simd_sums[sg] = sq1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total1 =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv1 = rsqrt(total1 * (1.0f / 5120.0f) + params.epsilon);
    const float4 toff4 = float4(params.target_norm_offset);

    float sq2 = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 post_bf16 =
          bfloat4(float4(reg_x[i]) * inv1 * (float4(twv[v]) + toff4));
      reg_x[i] = post_bf16;
      const float4 pf = float4(post_bf16);
      sq2 = fma(pf.x, pf.x, sq2);
      sq2 = fma(pf.y, pf.y, sq2);
      sq2 = fma(pf.z, pf.z, sq2);
      sq2 = fma(pf.w, pf.w, sq2);
    }
    sq2 = simd_sum(sq2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
      simd_sums[sg] = sq2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total2 =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv2 = rsqrt(total2 * (1.0f / 5120.0f) + params.epsilon);
    const float4 hoff4 = float4(params.hid_norm_offset);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 val =
          bfloat4(float4(reg_x[i]) * inv2 * (float4(hwv[v]) + hoff4));
      ov[v] = val;
      if (params.write_concat != 0u) {
        cv[v] = val;
      }
    }
    return;
  }

  if (params.hidden == 2048u) {
    device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
    device const bfloat4 *twv =
        reinterpret_cast<device const bfloat4 *>(target_final_norm);
    device const bfloat4 *hwv =
        reinterpret_cast<device const bfloat4 *>(pre_fc_norm_hidden);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
    device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
    bfloat4 reg_x[4];
    float sq1 = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 x4 = sv[v];
      reg_x[i] = x4;
      const float4 xf = float4(x4);
      sq1 = fma(xf.x, xf.x, sq1);
      sq1 = fma(xf.y, xf.y, sq1);
      sq1 = fma(xf.z, xf.z, sq1);
      sq1 = fma(xf.w, xf.w, sq1);
    }
    sq1 = simd_sum(sq1);
    if (lane == 0u) {
      simd_sums[sg] = sq1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total1 =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv1 = rsqrt(total1 * (1.0f / 2048.0f) + params.epsilon);
    const float4 toff4 = float4(params.target_norm_offset);

    float sq2 = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 post_bf16 =
          bfloat4(float4(reg_x[i]) * inv1 * (float4(twv[v]) + toff4));
      reg_x[i] = post_bf16;
      const float4 pf = float4(post_bf16);
      sq2 = fma(pf.x, pf.x, sq2);
      sq2 = fma(pf.y, pf.y, sq2);
      sq2 = fma(pf.z, pf.z, sq2);
      sq2 = fma(pf.w, pf.w, sq2);
    }
    sq2 = simd_sum(sq2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) {
      simd_sums[sg] = sq2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total2 =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv2 = rsqrt(total2 * (1.0f / 2048.0f) + params.epsilon);
    const float4 hoff4 = float4(params.hid_norm_offset);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = tid + i * kThreads;
      const bfloat4 val =
          bfloat4(float4(reg_x[i]) * inv2 * (float4(hwv[v]) + hoff4));
      ov[v] = val;
      if (params.write_concat != 0u) {
        cv[v] = val;
      }
    }
    return;
  }

  // Stage 1: target final_norm RMS denominator.
  float sq1 = 0.0f;
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const float v = float(src[col]);
    sq1 = fma(v, v, sq1);
  }
  sq1 = simd_sum(sq1);
  if (lane == 0u) {
    simd_sums[sg] = sq1;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total1 =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv1 = rsqrt(total1 / float(params.hidden) + params.epsilon);

  // Stage 2: BF16-rounded post-final-norm values -> pre_fc_norm_hidden RMS denominator.
  float sq2 = 0.0f;
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const bfloat post_bf16 =
        bfloat(float(src[col]) * inv1 *
               (float(target_final_norm[col]) + params.target_norm_offset));
    const float pv = float(post_bf16);
    sq2 = fma(pv, pv, sq2);
  }
  sq2 = simd_sum(sq2);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u) {
    simd_sums[sg] = sq2;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total2 =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv2 = rsqrt(total2 / float(params.hidden) + params.epsilon);

  // Stage 3: write BF16 stem_h (and optional stem_concat second half).
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const bfloat post_bf16 =
        bfloat(float(src[col]) * inv1 *
               (float(target_final_norm[col]) + params.target_norm_offset));
    const bfloat val =
        bfloat(float(post_bf16) * inv2 *
               (float(pre_fc_norm_hidden[col]) + params.hid_norm_offset));
    out_row[col] = val;
    if (params.write_concat != 0u) {
      cat_row[col] = val;
    }
  }
}

// Dual-pointer BM=8 row-exact BF16 GEMV/GEMM for mtp.fc ([n, 2*h]):
// Computes dot(fc[out_col, 0..h-1], stem_e[r]) + dot(fc[out_col, h..2*h-1], stem_h[r])
// directly without materializing a concatenated [rows, 2*h] activation buffer,
// reusing weight loads across up to 8 rows when rows > 1.
kernel void omlx_m5_mtp_fc_dual_bf16(
    device const bfloat *stem_e [[buffer(0)]],
    device const bfloat *stem_h [[buffer(1)]],
    device const bfloat *fc_weight [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant MtpFcDualParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  // T=1 2-cols/SG (8 cols/TG) fast path when params.reserved == 1:
  // Halves stem_e/stem_h L1 reads while preserving exact 0-ULP FMA + reduce order.
  if (params.reserved == 1u && params.rows == 1u && (params.h & 127u) == 0u) {
    const uint out_col0 = (tg.x * 4u + sg) * 2u;
    if (out_col0 >= params.n) {
      return;
    }
    const uint h = params.h;
    const uint vecs = h >> 2u;
    device const bfloat4 *xe4 = reinterpret_cast<device const bfloat4 *>(stem_e);
    device const bfloat4 *xh4 = reinterpret_cast<device const bfloat4 *>(stem_h);
    device const bfloat *w0_e = fc_weight + ulong(out_col0) * (2u * h);
    device const bfloat4 *we0 = reinterpret_cast<device const bfloat4 *>(w0_e);
    device const bfloat4 *wh0 = reinterpret_cast<device const bfloat4 *>(w0_e + h);
    const bool has_c1 = (out_col0 + 1u) < params.n;
    device const bfloat *w1_e = has_c1 ? (w0_e + 2u * h) : w0_e;
    device const bfloat4 *we1 = reinterpret_cast<device const bfloat4 *>(w1_e);
    device const bfloat4 *wh1 = reinterpret_cast<device const bfloat4 *>(w1_e + h);

    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint v = lane; v < vecs; v += 32u) {
      const float4 fx = float4(xe4[v]);
      const float4 fw0 = float4(we0[v]);
      const float4 fw1 = float4(we1[v]);
      acc0 = fma(fw0.x, fx.x, acc0);
      acc0 = fma(fw0.y, fx.y, acc0);
      acc0 = fma(fw0.z, fx.z, acc0);
      acc0 = fma(fw0.w, fx.w, acc0);
      acc1 = fma(fw1.x, fx.x, acc1);
      acc1 = fma(fw1.y, fx.y, acc1);
      acc1 = fma(fw1.z, fx.z, acc1);
      acc1 = fma(fw1.w, fx.w, acc1);
    }
    for (uint v = lane; v < vecs; v += 32u) {
      const float4 fx = float4(xh4[v]);
      const float4 fw0 = float4(wh0[v]);
      const float4 fw1 = float4(wh1[v]);
      acc0 = fma(fw0.x, fx.x, acc0);
      acc0 = fma(fw0.y, fx.y, acc0);
      acc0 = fma(fw0.z, fx.z, acc0);
      acc0 = fma(fw0.w, fx.w, acc0);
      acc1 = fma(fw1.x, fx.x, acc1);
      acc1 = fma(fw1.y, fx.y, acc1);
      acc1 = fma(fw1.z, fx.z, acc1);
      acc1 = fma(fw1.w, fx.w, acc1);
    }
    const float sum0 = mtp_simd_reduce1(acc0);
    const float sum1 = mtp_simd_reduce1(acc1);
    if (lane == 0u) {
      output[out_col0] = bfloat(sum0);
      if (has_c1) {
        output[out_col0 + 1u] = bfloat(sum1);
      }
    }
    return;
  }

  const uint out_col = tg.x * 4u + sg;
  const uint r_base = tg.y * 8u;
  if (out_col >= params.n || r_base >= params.rows) {
    return;
  }
  const uint tile_m = min(8u, params.rows - r_base);
  const uint h = params.h;
  device const bfloat *w_e = fc_weight + ulong(out_col) * (2u * h);
  device const bfloat *w_h = w_e + h;

  if (tile_m == 1u) {
    device const bfloat *xe = stem_e + ulong(r_base) * h;
    device const bfloat *xh = stem_h + ulong(r_base) * h;
    float acc = 0.0f;
    if ((h & 127u) == 0u) {
      const uint vecs = h >> 2u;
      device const bfloat4 *we4 = reinterpret_cast<device const bfloat4 *>(w_e);
      device const bfloat4 *wh4 = reinterpret_cast<device const bfloat4 *>(w_h);
      device const bfloat4 *xe4 = reinterpret_cast<device const bfloat4 *>(xe);
      device const bfloat4 *xh4 = reinterpret_cast<device const bfloat4 *>(xh);
      for (uint v = lane; v < vecs; v += 32u) {
        const float4 fw = float4(we4[v]);
        const float4 fx = float4(xe4[v]);
        acc = fma(fw.x, fx.x, acc);
        acc = fma(fw.y, fx.y, acc);
        acc = fma(fw.z, fx.z, acc);
        acc = fma(fw.w, fx.w, acc);
      }
      for (uint v = lane; v < vecs; v += 32u) {
        const float4 fw = float4(wh4[v]);
        const float4 fx = float4(xh4[v]);
        acc = fma(fw.x, fx.x, acc);
        acc = fma(fw.y, fx.y, acc);
        acc = fma(fw.z, fx.z, acc);
        acc = fma(fw.w, fx.w, acc);
      }
    } else {
      for (uint c = lane; c < h; c += 32u) {
        acc = fma(float(w_e[c]), float(xe[c]), acc);
      }
      for (uint c = lane; c < h; c += 32u) {
        acc = fma(float(w_h[c]), float(xh[c]), acc);
      }
    }
    const float sum = mtp_simd_reduce1(acc);
    if (lane == 0u) {
      output[ulong(r_base) * params.n + out_col] = bfloat(sum);
    }
    return;
  }

  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  float a4 = 0.0f, a5 = 0.0f, a6 = 0.0f, a7 = 0.0f;
  if ((h & 127u) == 0u) {
    const uint vecs = h >> 2u;
    device const bfloat4 *we4 = reinterpret_cast<device const bfloat4 *>(w_e);
    device const bfloat4 *wh4 = reinterpret_cast<device const bfloat4 *>(w_h);
    const auto fma_row = [&](device const bfloat *base, uint m, float4 fw,
                             uint v, float acc) -> float {
      if (m >= tile_m)
        return acc;
      device const bfloat4 *x4 = reinterpret_cast<device const bfloat4 *>(
          base + ulong(r_base + m) * h);
      const float4 fx = float4(x4[v]);
      acc = fma(fw.x, fx.x, acc);
      acc = fma(fw.y, fx.y, acc);
      acc = fma(fw.z, fx.z, acc);
      return fma(fw.w, fx.w, acc);
    };
    for (uint v = lane; v < vecs; v += 32u) {
      const float4 fw = float4(we4[v]);
      a0 = fma_row(stem_e, 0u, fw, v, a0);
      a1 = fma_row(stem_e, 1u, fw, v, a1);
      a2 = fma_row(stem_e, 2u, fw, v, a2);
      a3 = fma_row(stem_e, 3u, fw, v, a3);
      if (tile_m > 4u) {
        a4 = fma_row(stem_e, 4u, fw, v, a4);
        a5 = fma_row(stem_e, 5u, fw, v, a5);
        a6 = fma_row(stem_e, 6u, fw, v, a6);
        a7 = fma_row(stem_e, 7u, fw, v, a7);
      }
    }
    for (uint v = lane; v < vecs; v += 32u) {
      const float4 fw = float4(wh4[v]);
      a0 = fma_row(stem_h, 0u, fw, v, a0);
      a1 = fma_row(stem_h, 1u, fw, v, a1);
      a2 = fma_row(stem_h, 2u, fw, v, a2);
      a3 = fma_row(stem_h, 3u, fw, v, a3);
      if (tile_m > 4u) {
        a4 = fma_row(stem_h, 4u, fw, v, a4);
        a5 = fma_row(stem_h, 5u, fw, v, a5);
        a6 = fma_row(stem_h, 6u, fw, v, a6);
        a7 = fma_row(stem_h, 7u, fw, v, a7);
      }
    }
  } else {
    for (uint c = lane; c < h; c += 32u) {
      const float fw = float(w_e[c]);
      if (0u < tile_m) a0 = fma(fw, float(stem_e[ulong(r_base + 0u) * h + c]), a0);
      if (1u < tile_m) a1 = fma(fw, float(stem_e[ulong(r_base + 1u) * h + c]), a1);
      if (2u < tile_m) a2 = fma(fw, float(stem_e[ulong(r_base + 2u) * h + c]), a2);
      if (3u < tile_m) a3 = fma(fw, float(stem_e[ulong(r_base + 3u) * h + c]), a3);
      if (4u < tile_m) a4 = fma(fw, float(stem_e[ulong(r_base + 4u) * h + c]), a4);
      if (5u < tile_m) a5 = fma(fw, float(stem_e[ulong(r_base + 5u) * h + c]), a5);
      if (6u < tile_m) a6 = fma(fw, float(stem_e[ulong(r_base + 6u) * h + c]), a6);
      if (7u < tile_m) a7 = fma(fw, float(stem_e[ulong(r_base + 7u) * h + c]), a7);
    }
    for (uint c = lane; c < h; c += 32u) {
      const float fw = float(w_h[c]);
      if (0u < tile_m) a0 = fma(fw, float(stem_h[ulong(r_base + 0u) * h + c]), a0);
      if (1u < tile_m) a1 = fma(fw, float(stem_h[ulong(r_base + 1u) * h + c]), a1);
      if (2u < tile_m) a2 = fma(fw, float(stem_h[ulong(r_base + 2u) * h + c]), a2);
      if (3u < tile_m) a3 = fma(fw, float(stem_h[ulong(r_base + 3u) * h + c]), a3);
      if (4u < tile_m) a4 = fma(fw, float(stem_h[ulong(r_base + 4u) * h + c]), a4);
      if (5u < tile_m) a5 = fma(fw, float(stem_h[ulong(r_base + 5u) * h + c]), a5);
      if (6u < tile_m) a6 = fma(fw, float(stem_h[ulong(r_base + 6u) * h + c]), a6);
      if (7u < tile_m) a7 = fma(fw, float(stem_h[ulong(r_base + 7u) * h + c]), a7);
    }
  }
  const auto store_row = [&](uint m, float acc) {
    if (m < tile_m) {
      const float sum = mtp_simd_reduce1(acc);
      if (lane == 0u) {
        output[ulong(r_base + m) * params.n + out_col] = bfloat(sum);
      }
    }
  };
  store_row(0u, a0);
  store_row(1u, a1);
  store_row(2u, a2);
  store_row(3u, a3);
  if (tile_m > 4u) {
    store_row(4u, a4);
    store_row(5u, a5);
    store_row(6u, a6);
    store_row(7u, a7);
  }
}

// QK RMSNorm + RoPE with independent q_norm and k_norm weight_offsets for
// legacy mixed-convention MTP checkpoints (repair_legacy_head_norms).
kernel void omlx_m5_mtp_qk_norm_rope(
    device bfloat *query [[buffer(0)]],
    device bfloat *key [[buffer(1)]],
    device const bfloat *query_norm [[buffer(2)]],
    device const bfloat *key_norm [[buffer(3)]],
    constant MtpQkNormRopeParams &params [[buffer(4)]],
    uint lane [[thread_index_in_simdgroup]],
    uint3 tg_position [[threadgroup_position_in_grid]]) {
  const uint heads_per_token = params.query_heads + params.key_value_heads;
  const uint logical_head = tg_position.y;
  const uint token = logical_head / heads_per_token;
  if (token >= params.tokens) {
    return;
  }
  const uint local_head = logical_head - token * heads_per_token;
  const bool is_query = local_head < params.query_heads;
  const uint head = is_query ? local_head : local_head - params.query_heads;
  device bfloat *row =
      is_query ? query + ulong(token) * params.query_token_stride +
                     ulong(head) * params.query_head_stride
               : key + ulong(token) * params.key_token_stride +
                     ulong(head) * params.key_head_stride;
  device const bfloat *norm = is_query ? query_norm : key_norm;
  const float w_off = is_query ? params.q_weight_offset : params.k_weight_offset;

  float square_sum = 0.0f;
  for (uint column = lane; column < params.head_dim; column += 32u) {
    const float value = float(row[column]);
    square_sum = fma(value, value, square_sum);
  }
  const float inverse_rms =
      rsqrt(simd_sum(square_sum) / float(params.head_dim) + params.epsilon);
  const uint half_rotary = params.rotary_dim / 2u;
  const float position = float(params.start_position + token);
  for (uint pair = lane; pair < half_rotary; pair += 32u) {
    const uint first_index = pair;
    const uint second_index = pair + half_rotary;
    const float first =
        float(row[first_index]) * inverse_rms * (float(norm[first_index]) + w_off);
    const float second =
        float(row[second_index]) * inverse_rms * (float(norm[second_index]) + w_off);
    const float exponent = float(2u * pair) / float(params.rotary_dim);
    const float angle = position / pow(params.theta, exponent);
    const float cosine = cos(angle);
    const float sine = sin(angle);
    row[first_index] = bfloat(fma(-second, sine, first * cosine));
    row[second_index] = bfloat(fma(first, sine, second * cosine));
  }
  for (uint column = params.rotary_dim + lane; column < params.head_dim;
       column += 32u) {
    row[column] = bfloat(float(row[column]) * inverse_rms *
                         (float(norm[column]) + w_off));
  }
}

// Device-side chain latch: records step d's sampled draft token into
// proposal_out[step_index] (+ flags at proposal_out[4 + step_index]), and when
// write_chain != 0 seeds chain_token[0] + chain_hidden[0..H-1] for step d + 1
// inside the same Compute ICB with zero CPU synchronization.
kernel void omlx_m5_mtp_latch_chain_step(
    device const uint *sampled_raw [[buffer(0)]],
    device const bfloat *norm_src [[buffer(1)]],
    device const bfloat *final_norm_w [[buffer(2)]],
    device const float *rms_sq_buf [[buffer(3)]],
    device uint *proposal_out [[buffer(4)]],
    device uint *chain_token [[buffer(5)]],
    device bfloat *chain_hidden [[buffer(6)]],
    constant MtpLatchParams &params [[buffer(7)]],
    uint tid [[thread_position_in_grid]]) {
  if (tid == 0u) {
    const uint tok = (params.sampled_mode == 0u)
                         ? sampled_raw[2u * params.step_index]
                         : sampled_raw[params.step_index];
    const uint flg = (params.sampled_mode == 0u)
                         ? sampled_raw[2u * params.step_index + 1u]
                         : 0u;
    proposal_out[params.step_index] = tok;
    proposal_out[4u + params.step_index] = flg;
    if (params.write_chain != 0u) {
      chain_token[0] = tok;
    }
  }
  if (params.write_chain != 0u && tid < params.hidden) {
    if (params.norm_from_sq != 0u) {
      const float inv =
          rsqrt(rms_sq_buf[params.sq_slot] * params.inv_k + params.eps);
      const float v =
          float(norm_src[ulong(params.src_row) * params.hidden + tid]);
      const float w = float(final_norm_w[tid]) + params.w_off;
      chain_hidden[tid] = bfloat(v * inv * w);
    } else {
      chain_hidden[tid] = norm_src[ulong(params.src_row) * params.hidden + tid];
    }
  }
}

// ============================================================================
// Batched multi-slot MTP packed drafter kernels (B = 2..8).
// ============================================================================

struct MtpStemNormsOffsetParams {
  uint hidden;
  uint rows;
  uint cond_mode;
  uint write_concat;
  uint trunk_row_base;
  uint io_row_base;
  float epsilon;
  float target_norm_offset;
  float emb_norm_offset;
  float hid_norm_offset;
};

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mtp_stem_norms_offset(
    device const bfloat *embed_out [[buffer(0)]],
    device const bfloat *trunk_hidden [[buffer(1)]],
    device const bfloat *pending_trunk_normed [[buffer(2)]],
    device const bfloat *chain_hidden [[buffer(3)]],
    device const bfloat *target_final_norm [[buffer(4)]],
    device const bfloat *pre_fc_norm_embedding [[buffer(5)]],
    device const bfloat *pre_fc_norm_hidden [[buffer(6)]],
    device bfloat *stem_e [[buffer(7)]],
    device bfloat *stem_h [[buffer(8)]],
    device bfloat *stem_concat [[buffer(9)]],
    constant MtpStemNormsOffsetParams &params [[buffer(10)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr uint kThreads = 128u;
  constexpr uint kSimdgroups = 4u;
  const uint row = tg.y;
  if (row >= params.rows) {
    return;
  }
  const uint io_row = params.io_row_base + row;
  threadgroup float simd_sums[kSimdgroups];

  if (tg.z == 0u) {
    device const bfloat *in_row = embed_out + ulong(io_row) * params.hidden;
    device bfloat *out_row = stem_e + ulong(io_row) * params.hidden;
    device bfloat *cat_row =
        stem_concat + ulong(io_row) * (2u * params.hidden);
    if ((params.hidden & 3u) == 0u) {
      const uint vecs = params.hidden >> 2u;
      device const bfloat4 *iv = reinterpret_cast<device const bfloat4 *>(in_row);
      device const bfloat4 *wv =
          reinterpret_cast<device const bfloat4 *>(pre_fc_norm_embedding);
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
      device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
      float sq = 0.0f;
      for (uint v = tid; v < vecs; v += kThreads) {
        const float4 xf = float4(iv[v]);
        sq = fma(xf.x, xf.x, sq);
        sq = fma(xf.y, xf.y, sq);
        sq = fma(xf.z, xf.z, sq);
        sq = fma(xf.w, xf.w, sq);
      }
      sq = simd_sum(sq);
      if (lane == 0u)
        simd_sums[sg] = sq;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float total =
          ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
      const float mean = (params.hidden == 5120u)
                             ? (total * (1.0f / 5120.0f))
                             : ((params.hidden == 2048u)
                                    ? (total * (1.0f / 2048.0f))
                                    : (total / float(params.hidden)));
      const float inv = rsqrt(mean + params.epsilon);
      const float4 off4 = float4(params.emb_norm_offset);
      for (uint v = tid; v < vecs; v += kThreads) {
        const bfloat4 val =
            bfloat4(float4(iv[v]) * inv * (float4(wv[v]) + off4));
        ov[v] = val;
        if (params.write_concat != 0u)
          cv[v] = val;
      }
      return;
    }
    float sq = 0.0f;
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const float v = float(in_row[col]);
      sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u)
      simd_sums[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total / float(params.hidden) + params.epsilon);
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const bfloat val =
          bfloat(float(in_row[col]) * inv *
                 (float(pre_fc_norm_embedding[col]) + params.emb_norm_offset));
      out_row[col] = val;
      if (params.write_concat != 0u)
        cat_row[col] = val;
    }
    return;
  }

  device bfloat *out_row = stem_h + ulong(io_row) * params.hidden;
  device bfloat *cat_row =
      stem_concat + ulong(io_row) * (2u * params.hidden) + params.hidden;
  const bool already_post_norm =
      (params.cond_mode == 2u) || (params.cond_mode == 1u && row == 0u);
  if (already_post_norm) {
    device const bfloat *src =
        (params.cond_mode == 2u) ? (chain_hidden + ulong(row) * params.hidden)
                                 : pending_trunk_normed;
    if ((params.hidden & 3u) == 0u) {
      const uint vecs = params.hidden >> 2u;
      device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
      device const bfloat4 *wv =
          reinterpret_cast<device const bfloat4 *>(pre_fc_norm_hidden);
      device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
      device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
      float sq = 0.0f;
      for (uint v = tid; v < vecs; v += kThreads) {
        const float4 xf = float4(sv[v]);
        sq = fma(xf.x, xf.x, sq);
        sq = fma(xf.y, xf.y, sq);
        sq = fma(xf.z, xf.z, sq);
        sq = fma(xf.w, xf.w, sq);
      }
      sq = simd_sum(sq);
      if (lane == 0u)
        simd_sums[sg] = sq;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const float total =
          ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
      const float mean = (params.hidden == 5120u)
                             ? (total * (1.0f / 5120.0f))
                             : ((params.hidden == 2048u)
                                    ? (total * (1.0f / 2048.0f))
                                    : (total / float(params.hidden)));
      const float inv = rsqrt(mean + params.epsilon);
      const float4 off4 = float4(params.hid_norm_offset);
      for (uint v = tid; v < vecs; v += kThreads) {
        const bfloat4 val =
            bfloat4(float4(sv[v]) * inv * (float4(wv[v]) + off4));
        ov[v] = val;
        if (params.write_concat != 0u)
          cv[v] = val;
      }
      return;
    }
    float sq = 0.0f;
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const float v = float(src[col]);
      sq = fma(v, v, sq);
    }
    sq = simd_sum(sq);
    if (lane == 0u)
      simd_sums[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total / float(params.hidden) + params.epsilon);
    for (uint col = tid; col < params.hidden; col += kThreads) {
      const bfloat val =
          bfloat(float(src[col]) * inv *
                 (float(pre_fc_norm_hidden[col]) + params.hid_norm_offset));
      out_row[col] = val;
      if (params.write_concat != 0u)
        cat_row[col] = val;
    }
    return;
  }

  const uint trunk_row =
      params.trunk_row_base + ((params.cond_mode == 1u) ? (row - 1u) : row);
  device const bfloat *src = trunk_hidden + ulong(trunk_row) * params.hidden;
  if ((params.hidden & 3u) == 0u) {
    const uint vecs = params.hidden >> 2u;
    device const bfloat4 *sv = reinterpret_cast<device const bfloat4 *>(src);
    device const bfloat4 *twv =
        reinterpret_cast<device const bfloat4 *>(target_final_norm);
    device const bfloat4 *hwv =
        reinterpret_cast<device const bfloat4 *>(pre_fc_norm_hidden);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(out_row);
    device bfloat4 *cv = reinterpret_cast<device bfloat4 *>(cat_row);
    float sq1 = 0.0f;
    for (uint v = tid; v < vecs; v += kThreads) {
      const float4 xf = float4(sv[v]);
      sq1 = fma(xf.x, xf.x, sq1);
      sq1 = fma(xf.y, xf.y, sq1);
      sq1 = fma(xf.z, xf.z, sq1);
      sq1 = fma(xf.w, xf.w, sq1);
    }
    sq1 = simd_sum(sq1);
    if (lane == 0u)
      simd_sums[sg] = sq1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total1 =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float mean1 = (params.hidden == 5120u)
                            ? (total1 * (1.0f / 5120.0f))
                            : ((params.hidden == 2048u)
                                   ? (total1 * (1.0f / 2048.0f))
                                   : (total1 / float(params.hidden)));
    const float inv1 = rsqrt(mean1 + params.epsilon);
    const float4 toff4 = float4(params.target_norm_offset);
    float sq2 = 0.0f;
    for (uint v = tid; v < vecs; v += kThreads) {
      const bfloat4 post_bf16 =
          bfloat4(float4(sv[v]) * inv1 * (float4(twv[v]) + toff4));
      const float4 pf = float4(post_bf16);
      sq2 = fma(pf.x, pf.x, sq2);
      sq2 = fma(pf.y, pf.y, sq2);
      sq2 = fma(pf.z, pf.z, sq2);
      sq2 = fma(pf.w, pf.w, sq2);
    }
    sq2 = simd_sum(sq2);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u)
      simd_sums[sg] = sq2;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total2 =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float mean2 = (params.hidden == 5120u)
                            ? (total2 * (1.0f / 5120.0f))
                            : ((params.hidden == 2048u)
                                   ? (total2 * (1.0f / 2048.0f))
                                   : (total2 / float(params.hidden)));
    const float inv2 = rsqrt(mean2 + params.epsilon);
    const float4 hoff4 = float4(params.hid_norm_offset);
    for (uint v = tid; v < vecs; v += kThreads) {
      const bfloat4 post_bf16 =
          bfloat4(float4(sv[v]) * inv1 * (float4(twv[v]) + toff4));
      const bfloat4 val =
          bfloat4(float4(post_bf16) * inv2 * (float4(hwv[v]) + hoff4));
      ov[v] = val;
      if (params.write_concat != 0u)
        cv[v] = val;
    }
    return;
  }

  float sq1 = 0.0f;
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const float v = float(src[col]);
    sq1 = fma(v, v, sq1);
  }
  sq1 = simd_sum(sq1);
  if (lane == 0u)
    simd_sums[sg] = sq1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total1 =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv1 = rsqrt(total1 / float(params.hidden) + params.epsilon);
  float sq2 = 0.0f;
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const bfloat post_bf16 =
        bfloat(float(src[col]) * inv1 *
               (float(target_final_norm[col]) + params.target_norm_offset));
    const float pv = float(post_bf16);
    sq2 = fma(pv, pv, sq2);
  }
  sq2 = simd_sum(sq2);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0u)
    simd_sums[sg] = sq2;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total2 =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv2 = rsqrt(total2 / float(params.hidden) + params.epsilon);
  for (uint col = tid; col < params.hidden; col += kThreads) {
    const bfloat post_bf16 =
        bfloat(float(src[col]) * inv1 *
               (float(target_final_norm[col]) + params.target_norm_offset));
    const bfloat val =
        bfloat(float(post_bf16) * inv2 *
               (float(pre_fc_norm_hidden[col]) + params.hid_norm_offset));
    out_row[col] = val;
    if (params.write_concat != 0u)
      cat_row[col] = val;
  }
}

struct MtpSlotLayoutParams {
  uint batch_size;
  uint q_cols;
  uint kv_cols;
  uint mixer_cols;
  uint hidden;
  uint pad0;
  uint pad1;
  uint pad2;
  uint fold_counts[8];
  uint fold_offsets[8];
};

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mtp_scatter_qkv_slots(
    device const bfloat *packed_q [[buffer(0)]],
    device const bfloat *packed_k [[buffer(1)]],
    device const bfloat *packed_v [[buffer(2)]],
    device bfloat *slot_q [[buffer(3)]],
    device bfloat *slot_k [[buffer(4)]],
    device bfloat *slot_v [[buffer(5)]],
    constant MtpSlotLayoutParams &p [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint row = gid.y;
  const uint slot = row / 5u;
  const uint local_r = row - slot * 5u;
  if (slot >= p.batch_size || local_r >= p.fold_counts[slot])
    return;
  const uint src_row = p.fold_offsets[slot] + local_r;
  const uint dst_row = slot * 5u + local_r;
  const uint col = gid.x;
  if (col < p.q_cols) {
    slot_q[ulong(dst_row) * p.q_cols + col] =
        packed_q[ulong(src_row) * p.q_cols + col];
  }
  if (col < p.kv_cols) {
    slot_k[ulong(dst_row) * p.kv_cols + col] =
        packed_k[ulong(src_row) * p.kv_cols + col];
    slot_v[ulong(dst_row) * p.kv_cols + col] =
        packed_v[ulong(src_row) * p.kv_cols + col];
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mtp_gather_mixer_slots(
    device const bfloat *slot_mixer [[buffer(0)]],
    device bfloat *packed_mixer [[buffer(1)]],
    constant MtpSlotLayoutParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint row = gid.y;
  const uint slot = row / 5u;
  const uint local_r = row - slot * 5u;
  if (slot >= p.batch_size || local_r >= p.fold_counts[slot])
    return;
  const uint dst_row = p.fold_offsets[slot] + local_r;
  const uint src_row = slot * 5u + local_r;
  const uint col = gid.x;
  if (col < p.mixer_cols) {
    packed_mixer[ulong(dst_row) * p.mixer_cols + col] =
        slot_mixer[ulong(src_row) * p.mixer_cols + col];
  }
}

[[max_total_threads_per_threadgroup(128)]]
kernel void omlx_m5_mtp_extract_last_rows(
    device const bfloat *packed_norm [[buffer(0)]],
    device bfloat *last_norm [[buffer(1)]],
    constant MtpSlotLayoutParams &p [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint slot = gid.y;
  const uint col = gid.x;
  if (slot >= p.batch_size || col >= p.hidden || p.fold_counts[slot] == 0u)
    return;
  const uint src_row = p.fold_offsets[slot] + p.fold_counts[slot] - 1u;
  last_norm[ulong(slot) * p.hidden + col] =
      packed_norm[ulong(src_row) * p.hidden + col];
}

struct MtpLatchBatchParams {
  uint batch_size;
  uint hidden;
  uint step_index;
  uint write_chain;
};

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_mtp_latch_chain_batch(
    device const uint *sampled_raw [[buffer(0)]],
    device const bfloat *norm_src [[buffer(1)]],
    device uint *proposal_out [[buffer(2)]],
    device uint *chain_tokens [[buffer(3)]],
    device bfloat *chain_hidden [[buffer(4)]],
    constant MtpLatchBatchParams &p [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]) {
  const uint slot = gid.y;
  const uint col = gid.x;
  if (slot >= p.batch_size)
    return;
  if (col == 0u) {
    const uint tok = sampled_raw[2u * slot];
    const uint flg = sampled_raw[2u * slot + 1u];
    proposal_out[slot * 8u + p.step_index] = tok;
    proposal_out[slot * 8u + 4u + p.step_index] = flg;
    if (p.write_chain != 0u) {
      chain_tokens[slot] = tok;
    }
  }
  if (p.write_chain != 0u && col < p.hidden) {
    chain_hidden[ulong(slot) * p.hidden + col] =
        norm_src[ulong(slot) * p.hidden + col];
  }
}
