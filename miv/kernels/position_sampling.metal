#include <metal_simdgroup>
#include <metal_stdlib>
using namespace metal;
#include "position_sampling_shared.h"
struct PositionRow {ulong seed,position,purpose;uint top_k;float temperature,top_p;uint valid;};
struct PositionGeometry {uint vocab,rows,max_rows,reserved;};
kernel void omlx_m5_position_verify_equality(
    device const uint2 *sampled [[buffer(0)]],device const uint *proposed [[buffer(1)]],
    device uint *licensed [[buffer(2)]],device uint4 *meta [[buffer(3)]],
    constant uint &width [[buffer(4)]],uint tid [[thread_position_in_grid]]) {
  if(tid)return;
  if(width==0){meta[0]=uint4(0,0,0,0);return;}
  uint flags=0;for(uint r=0;r<width;++r)flags|=sampled[r].y;
  if(flags){meta[0]=uint4(0,0,0,flags);return;}
  uint accepted=0;
  for(uint r=0;r+1<width;++r) {
    licensed[r]=sampled[r].x;
    if(proposed[r]!=sampled[r].x){meta[0]=uint4(r+1,accepted,1,0);return;}
    ++accepted;
  }
  licensed[width-1]=sampled[width-1].x;meta[0]=uint4(width,accepted,0,0);
}
kernel void omlx_m5_position_uniform_vectors(
    constant PositionRow *rows [[buffer(0)]],device const uint *ids [[buffer(1)]],
    device float *values [[buffer(2)]],constant uint &count [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if(i<count)values[i]=position_uniform(rows[i].seed,rows[i].position,ids[i],rows[i].purpose);
}

inline bool position_sample_better(float v, uint id, float other_v,
                                   uint other_id) {
  return v > other_v || (v == other_v && id < other_id);
}

inline void position_insert_top4(float v, uint id, thread float &v0,
                                 thread uint &i0, thread float &v1,
                                 thread uint &i1, thread float &v2,
                                 thread uint &i2, thread float &v3,
                                 thread uint &i3) {
  if (position_sample_better(v, id, v3, i3)) {
    if (position_sample_better(v, id, v2, i2)) {
      v3 = v2;
      i3 = i2;
      if (position_sample_better(v, id, v1, i1)) {
        v2 = v1;
        i2 = i1;
        if (position_sample_better(v, id, v0, i0)) {
          v1 = v0;
          i1 = i0;
          v0 = v;
          i0 = id;
        } else {
          v1 = v;
          i1 = id;
        }
      } else {
        v2 = v;
        i2 = id;
      }
    } else {
      v3 = v;
      i3 = id;
    }
  }
}

inline void position_reduce_argmax_256(float cur_v, uint cur_i, uint vocab,
                                       uint lane, uint sg,
                                       threadgroup float *sg_max,
                                       threadgroup uint *sg_idx,
                                       threadgroup float *out_val,
                                       threadgroup uint *out_idx) {
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(cur_v, offset);
    const uint other_i = simd_shuffle_down(cur_i, offset);
    if (position_sample_better(other_v, other_i, cur_v, cur_i)) {
      cur_v = other_v;
      cur_i = other_i;
    }
  }
  if (lane == 0) {
    sg_max[sg] = cur_v;
    sg_idx[sg] = cur_i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float red_v = lane < 8 ? sg_max[lane] : -INFINITY;
    uint red_i = lane < 8 ? sg_idx[lane] : vocab;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      if (position_sample_better(other_v, other_i, red_v, red_i)) {
        red_v = other_v;
        red_i = other_i;
      }
    }
    if (lane == 0) {
      *out_val = red_v;
      *out_idx = red_i;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// One independent TG per row. Compact IDs/scores are written only for the
// canonical boundary fallback; full logits remain on the GPU.
kernel void omlx_m5_position_sample_logits(
    device const bfloat *logits [[buffer(0)]],device uint2 *result [[buffer(1)]],
    device uint *compact_ids [[buffer(2)]],device float *compact_scores [[buffer(3)]],
    constant PositionRow *options [[buffer(4)]],constant PositionGeometry &g [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row=tg.y;
  if(row>=g.rows || row>=g.max_rows)return;
  const PositionRow p=options[row];
  if(!p.valid){if(tid==0)result[row]=uint2(0,0);return;}
  if(g.vocab==0){if(tid==0)result[row]=uint2(0,2);return;}
  device const bfloat *row_logits = logits + ulong(row) * g.vocab;
  const uint top_k=min(p.top_k,min(g.vocab,64u));
  const bool greedy = (p.temperature <= 0.0f || top_k <= 1u);

  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup uint shared_bad[1];
  threadgroup float values[64];
  threadgroup uint ids[64];

  float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
  uint i0 = g.vocab, i1 = g.vocab, i2 = g.vocab, i3 = g.vocab;
  uint invalid = 0;

  if (greedy) {
    for (uint t = tid; t < g.vocab; t += 256u) {
      const float v = float(row_logits[t]);
      invalid |= uint(!isfinite(v));
      if (position_sample_better(v, t, v0, i0)) {
        v0 = v;
        i0 = t;
      }
    }
  } else {
    for (uint t = tid; t < g.vocab; t += 256u) {
      const float v = float(row_logits[t]);
      invalid |= uint(!isfinite(v));
      position_insert_top4(v, t, v0, i0, v1, i1, v2, i2, v3, i3);
    }
  }

  float cur_v = v0;
  uint cur_i = i0;
  uint cur_bad = invalid;
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(cur_v, offset);
    const uint other_i = simd_shuffle_down(cur_i, offset);
    cur_bad |= simd_shuffle_down(cur_bad, offset);
    if (position_sample_better(other_v, other_i, cur_v, cur_i)) {
      cur_v = other_v;
      cur_i = other_i;
    }
  }
  if (lane == 0) {
    sg_max[sg] = cur_v;
    sg_idx[sg] = cur_i;
    sg_bad[sg] = cur_bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float red_v = lane < 8 ? sg_max[lane] : -INFINITY;
    uint red_i = lane < 8 ? sg_idx[lane] : g.vocab;
    uint red_bad = lane < 8 ? sg_bad[lane] : 0u;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      red_bad |= simd_shuffle_down(red_bad, offset);
      if (position_sample_better(other_v, other_i, red_v, red_i)) {
        red_v = other_v;
        red_i = other_i;
      }
    }
    if (lane == 0) {
      values[0] = red_v;
      ids[0] = red_i;
      shared_bad[0] = red_bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (shared_bad[0]) {
    if (tid == 0) result[row] = uint2(0, 2);
    return;
  }
  if (greedy) {
    if (tid == 0) result[row] = uint2(ids[0], 0);
    return;
  }

  uint popped = 0;
  {
    const uint win_i = ids[0];
    if (win_i < g.vocab && (win_i & 255u) == tid) {
      popped = 1u;
      v0 = v1;
      i0 = i1;
      v1 = v2;
      i1 = i2;
      v2 = v3;
      i2 = i3;
      v3 = -INFINITY;
      i3 = g.vocab;
    }
  }

  for (uint step = 1; step < top_k; ++step) {
    position_reduce_argmax_256(v0, i0, g.vocab, lane, sg, sg_max, sg_idx,
                               &values[step], &ids[step]);
    const float win_v = values[step];
    const uint win_i = ids[step];
    if (win_i < g.vocab && (win_i & 255u) == tid) {
      ++popped;
      if (popped < 4u) {
        v0 = v1;
        i0 = i1;
        v1 = v2;
        i1 = i2;
        v2 = v3;
        i2 = i3;
        v3 = -INFINITY;
        i3 = g.vocab;
      } else if (step + 1u < top_k) {
        v0 = -INFINITY;
        v1 = -INFINITY;
        v2 = -INFINITY;
        v3 = -INFINITY;
        i0 = g.vocab;
        i1 = g.vocab;
        i2 = g.vocab;
        i3 = g.vocab;
        for (uint t = tid; t < g.vocab; t += 256u) {
          const float v = float(row_logits[t]);
          if (!position_sample_better(win_v, win_i, v, t)) {
            continue;
          }
          position_insert_top4(v, t, v0, i0, v1, i1, v2, i2, v3, i3);
        }
        popped = 0;
      }
    }
  }

  if (tid) return;
  float prob[64];
  float total=0;
  const float max_val = values[0];
  for(uint i=0;i<top_k;++i) {
    const uint id_i = ids[i];
    const float val_i = values[i];
    compact_ids[ulong(row)*64+i]=id_i;compact_scores[ulong(row)*64+i]=val_i;
    prob[i]=exp((val_i-max_val)/p.temperature);total+=prob[i];
  }
  uint retained=top_k,flag=0;float cumulative=0;
  for(uint i=0;i<top_k;++i) {
    cumulative+=prob[i]/total;
    if(p.top_p<1 && fabs(cumulative-p.top_p)<2e-4f)flag=1;
    if(cumulative>=p.top_p){retained=i+1;break;}
  }
  float score=-INFINITY,runner=-INFINITY;uint selected=ids[0];
  for(uint i=0;i<retained;++i) {
    const uint id_i = ids[i];
    const float v=(values[i]-max_val)/p.temperature+position_gumbel(p.seed,p.position,id_i,p.purpose);
    if(v>score || (v==score && id_i<selected)){runner=score;score=v;selected=id_i;}
    else runner=max(runner,v);
  }
  if(score-runner<2e-4f)flag=1;
  result[row]=uint2(selected,flag);
}

// Device-feeding T=1 pure-AR variant: writes sampled token directly to
// next_token_out[0] so step t+1 can be enqueued before step t completes on GPU.
kernel void omlx_m5_position_sample_logits_feed(
    device const bfloat *logits [[buffer(0)]],device uint2 *result [[buffer(1)]],
    device uint *compact_ids [[buffer(2)]],device float *compact_scores [[buffer(3)]],
    device uint *next_token_out [[buffer(4)]],
    constant PositionRow *options [[buffer(5)]],constant PositionGeometry &g [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= g.rows || row >= g.max_rows) return;
  const PositionRow p = options[row];
  if (!p.valid) {
    if (tid == 0) {
      result[row] = uint2(0, 0);
      if (row == 0) next_token_out[0] = 0u;
    }
    return;
  }
  if (g.vocab == 0) {
    if (tid == 0) {
      result[row] = uint2(0, 2);
      if (row == 0) next_token_out[0] = 0u;
    }
    return;
  }
  device const bfloat *row_logits = logits + ulong(row) * g.vocab;
  const uint top_k = min(p.top_k, min(g.vocab, 64u));
  const bool greedy = (p.temperature <= 0.0f || top_k <= 1u);

  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup uint shared_bad[1];
  threadgroup float values[64];
  threadgroup uint ids[64];

  float v0 = -INFINITY;
  uint i0 = g.vocab;
  uint invalid = 0;
  for (uint t = tid; t < g.vocab; t += 256u) {
    const float v = float(row_logits[t]);
    invalid |= uint(!isfinite(v));
    if (position_sample_better(v, t, v0, i0)) {
      v0 = v;
      i0 = t;
    }
  }
  float cur_v = v0;
  uint cur_i = i0;
  uint cur_bad = invalid;
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(cur_v, offset);
    const uint other_i = simd_shuffle_down(cur_i, offset);
    cur_bad |= simd_shuffle_down(cur_bad, offset);
    if (position_sample_better(other_v, other_i, cur_v, cur_i)) {
      cur_v = other_v;
      cur_i = other_i;
    }
  }
  if (lane == 0) {
    sg_max[sg] = cur_v;
    sg_idx[sg] = cur_i;
    sg_bad[sg] = cur_bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float red_v = lane < 8 ? sg_max[lane] : -INFINITY;
    uint red_i = lane < 8 ? sg_idx[lane] : g.vocab;
    uint red_bad = lane < 8 ? sg_bad[lane] : 0u;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      red_bad |= simd_shuffle_down(red_bad, offset);
      if (position_sample_better(other_v, other_i, red_v, red_i)) {
        red_v = other_v;
        red_i = other_i;
      }
    }
    if (lane == 0) {
      values[0] = red_v;
      ids[0] = red_i;
      shared_bad[0] = red_bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (shared_bad[0]) {
    if (tid == 0) {
      result[row] = uint2(0, 2);
      if (row == 0) next_token_out[0] = 0u;
    }
    return;
  }
  if (greedy) {
    if (tid == 0) {
      result[row] = uint2(ids[0], 0);
      if (row == 0) next_token_out[0] = ids[0];
    }
    return;
  }
  // Stochastic fallback safety: write compact_ids[0] = ids[0] and flag = 1 if needed.
  if (tid == 0) {
    compact_ids[ulong(row) * 64u] = ids[0];
    compact_scores[ulong(row) * 64u] = values[0];
    result[row] = uint2(ids[0], 1u);
    if (row == 0) next_token_out[0] = ids[0];
  }
}


struct PositionArgmaxPart {
  float value;
  uint index;
  uint invalid;
  uint reserved;
};

// Stage 1 of 2-stage 64-TG parallel greedy argmax:
// Grid: {.width = 64 * 256, .height = rows}, Threads: {.width = 256}.
// Partitions [0, g.vocab) across 64 threadgroups (16,384 threads across all 64
// M5 Ultra GPU cores), using coalesced bfloat4 reads from hot SLC and 100%
// associative lowest-index tie-breaking + NaN/Inf detection.
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_position_argmax_stage1_64tg(
    device const bfloat *logits [[buffer(0)]],
    device PositionArgmaxPart *parts [[buffer(1)]],
    constant PositionGeometry &g [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint chunk = tg.x;
  const uint row = tg.y;
  if (chunk >= 64u || row >= g.rows || row >= g.max_rows) {
    return;
  }
  if (g.vocab == 0u) {
    if (tid == 0u) {
      parts[ulong(row) * 64ul + chunk] = PositionArgmaxPart{-INFINITY, 0u, 0u, 0u};
    }
    return;
  }

  device const bfloat *row_logits = logits + ulong(row) * g.vocab;
  const bool aligned4 = ((ulong(row) * g.vocab) & 3ul) == 0ul &&
                        (reinterpret_cast<ulong>(logits) & 7ul) == 0ul;
  const uint vocab4 = g.vocab >> 2u;

  float v0 = -INFINITY;
  uint i0 = g.vocab;
  uint invalid = 0u;

  if (aligned4) {
    device const bfloat4 *row_logits4 =
        reinterpret_cast<device const bfloat4 *>(row_logits);
    for (uint vec_idx = chunk * 256u + tid; vec_idx < vocab4;
         vec_idx += 64u * 256u) {
      const float4 fv = float4(row_logits4[vec_idx]);
      const uint base = vec_idx << 2u;
      invalid |= uint(!isfinite(fv.x)) | uint(!isfinite(fv.y)) |
                 uint(!isfinite(fv.z)) | uint(!isfinite(fv.w));
      if (position_sample_better(fv.x, base + 0u, v0, i0)) {
        v0 = fv.x;
        i0 = base + 0u;
      }
      if (position_sample_better(fv.y, base + 1u, v0, i0)) {
        v0 = fv.y;
        i0 = base + 1u;
      }
      if (position_sample_better(fv.z, base + 2u, v0, i0)) {
        v0 = fv.z;
        i0 = base + 2u;
      }
      if (position_sample_better(fv.w, base + 3u, v0, i0)) {
        v0 = fv.w;
        i0 = base + 3u;
      }
    }

    if (chunk == 0u) {
      const uint tail_start = vocab4 << 2u;
      const uint t = tail_start + tid;
      if (t < g.vocab) {
        const float v = float(row_logits[t]);
        invalid |= uint(!isfinite(v));
        if (position_sample_better(v, t, v0, i0)) {
          v0 = v;
          i0 = t;
        }
      }
    }
  } else {
    for (uint t = chunk * 256u + tid; t < g.vocab; t += 64u * 256u) {
      const float v = float(row_logits[t]);
      invalid |= uint(!isfinite(v));
      if (position_sample_better(v, t, v0, i0)) {
        v0 = v;
        i0 = t;
      }
    }
  }

  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(v0, offset);
    const uint other_i = simd_shuffle_down(i0, offset);
    invalid |= simd_shuffle_down(invalid, offset);
    if (position_sample_better(other_v, other_i, v0, i0)) {
      v0 = other_v;
      i0 = other_i;
    }
  }

  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  if (lane == 0u) {
    sg_max[sg] = v0;
    sg_idx[sg] = i0;
    sg_bad[sg] = invalid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float red_v = lane < 8u ? sg_max[lane] : -INFINITY;
    uint red_i = lane < 8u ? sg_idx[lane] : g.vocab;
    uint red_bad = lane < 8u ? sg_bad[lane] : 0u;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      red_bad |= simd_shuffle_down(red_bad, offset);
      if (position_sample_better(other_v, other_i, red_v, red_i)) {
        red_v = other_v;
        red_i = other_i;
      }
    }
    if (lane == 0u) {
      parts[ulong(row) * 64ul + chunk] =
          PositionArgmaxPart{red_v, red_i, red_bad, 0u};
    }
  }
}

// Option-aware Stage 1 of 2-stage 64xT parallel greedy/keyed sampler:
// Grid: {.width = 64 * 256, .height = rows}, Threads: {.width = 256}.
// Launches 64 * rows threadgroups across all 64 M5 Ultra GPU cores for greedy
// / top_k <= 1 rows (0-ULP lowest-index tie-breaking), and early-exits in 0
// DRAM reads when !p.valid or stochastic (temperature > 0 && top_k > 1).
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_position_sample_logits_stage1(
    device const bfloat *logits [[buffer(0)]],
    device PositionArgmaxPart *parts [[buffer(1)]],
    constant PositionRow *options [[buffer(2)]],
    constant PositionGeometry &g [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint chunk = tg.x;
  const uint row = tg.y;
  if (chunk >= 64u || row >= g.rows || row >= g.max_rows) {
    return;
  }
  const PositionRow p = options[row];
  if (!p.valid) {
    return;
  }
  if (g.vocab == 0u) {
    if (tid == 0u) {
      parts[ulong(row) * 64ul + chunk] =
          PositionArgmaxPart{-INFINITY, 0u, 0u, 0u};
    }
    return;
  }
  const uint top_k = min(p.top_k, min(g.vocab, 64u));
  if (p.temperature > 0.0f && top_k > 1u) {
    return;
  }

  device const bfloat *row_logits = logits + ulong(row) * g.vocab;
  const bool aligned4 = ((ulong(row) * g.vocab) & 3ul) == 0ul &&
                        (reinterpret_cast<ulong>(logits) & 7ul) == 0ul;
  const uint vocab4 = g.vocab >> 2u;

  float v0 = -INFINITY;
  uint i0 = g.vocab;
  uint invalid = 0u;

  if (aligned4) {
    device const bfloat4 *row_logits4 =
        reinterpret_cast<device const bfloat4 *>(row_logits);
    for (uint vec_idx = chunk * 256u + tid; vec_idx < vocab4;
         vec_idx += 64u * 256u) {
      const float4 fv = float4(row_logits4[vec_idx]);
      const uint base = vec_idx << 2u;
      invalid |= uint(!isfinite(fv.x)) | uint(!isfinite(fv.y)) |
                 uint(!isfinite(fv.z)) | uint(!isfinite(fv.w));
      if (position_sample_better(fv.x, base + 0u, v0, i0)) {
        v0 = fv.x;
        i0 = base + 0u;
      }
      if (position_sample_better(fv.y, base + 1u, v0, i0)) {
        v0 = fv.y;
        i0 = base + 1u;
      }
      if (position_sample_better(fv.z, base + 2u, v0, i0)) {
        v0 = fv.z;
        i0 = base + 2u;
      }
      if (position_sample_better(fv.w, base + 3u, v0, i0)) {
        v0 = fv.w;
        i0 = base + 3u;
      }
    }

    if (chunk == 0u) {
      const uint tail_start = vocab4 << 2u;
      const uint t = tail_start + tid;
      if (t < g.vocab) {
        const float v = float(row_logits[t]);
        invalid |= uint(!isfinite(v));
        if (position_sample_better(v, t, v0, i0)) {
          v0 = v;
          i0 = t;
        }
      }
    }
  } else {
    for (uint t = chunk * 256u + tid; t < g.vocab; t += 64u * 256u) {
      const float v = float(row_logits[t]);
      invalid |= uint(!isfinite(v));
      if (position_sample_better(v, t, v0, i0)) {
        v0 = v;
        i0 = t;
      }
    }
  }

  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(v0, offset);
    const uint other_i = simd_shuffle_down(i0, offset);
    invalid |= simd_shuffle_down(invalid, offset);
    if (position_sample_better(other_v, other_i, v0, i0)) {
      v0 = other_v;
      i0 = other_i;
    }
  }

  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  if (lane == 0u) {
    sg_max[sg] = v0;
    sg_idx[sg] = i0;
    sg_bad[sg] = invalid;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (sg == 0u) {
    float red_v = lane < 8u ? sg_max[lane] : -INFINITY;
    uint red_i = lane < 8u ? sg_idx[lane] : g.vocab;
    uint red_bad = lane < 8u ? sg_bad[lane] : 0u;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      red_bad |= simd_shuffle_down(red_bad, offset);
      if (position_sample_better(other_v, other_i, red_v, red_i)) {
        red_v = other_v;
        red_i = other_i;
      }
    }
    if (lane == 0u) {
      parts[ulong(row) * 64ul + chunk] =
          PositionArgmaxPart{red_v, red_i, red_bad, 0u};
    }
  }
}

inline void position_reduce_64_parts(
    device const PositionArgmaxPart *row_parts,
    uint vocab,
    uint tid,
    uint lane,
    uint sg,
    threadgroup float *sg_max,
    threadgroup uint *sg_idx,
    threadgroup uint *sg_bad,
    thread float &win_v,
    thread uint &win_i,
    thread uint &win_bad) {
  float cur_v = -INFINITY;
  uint cur_i = vocab;
  uint cur_bad = 0u;
  if (tid < 64u) {
    const PositionArgmaxPart pt = row_parts[tid];
    cur_v = pt.value;
    cur_i = pt.index;
    cur_bad = pt.invalid;
  }
  if (sg < 2u) {
    for (ushort offset = 16; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(cur_v, offset);
      const uint other_i = simd_shuffle_down(cur_i, offset);
      cur_bad |= simd_shuffle_down(cur_bad, offset);
      if (position_sample_better(other_v, other_i, cur_v, cur_i)) {
        cur_v = other_v;
        cur_i = other_i;
      }
    }
    if (lane == 0u) {
      sg_max[sg] = cur_v;
      sg_idx[sg] = cur_i;
      sg_bad[sg] = cur_bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float v0 = sg_max[0];
  uint i0 = sg_idx[0];
  if (position_sample_better(sg_max[1], sg_idx[1], v0, i0)) {
    v0 = sg_max[1];
    i0 = sg_idx[1];
  }
  win_v = v0;
  win_i = i0;
  win_bad = sg_bad[0] | sg_bad[1];
}

// Stage 2 of 2-stage 64-TG sampler (non-feeding):
// Uses 64-winner Stage 1 summary for 0-ULP greedy / top_k <= 1 (~3 us), and
// seamlessly falls back to full top-k Gumbel scan when temperature > 0 && top_k > 1.
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_position_sample_logits_stage2(
    device const bfloat *logits [[buffer(0)]],
    device const PositionArgmaxPart *parts [[buffer(1)]],
    device uint2 *result [[buffer(2)]],
    device uint *compact_ids [[buffer(3)]],
    device float *compact_scores [[buffer(4)]],
    constant PositionRow *options [[buffer(5)]],
    constant PositionGeometry &g [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg_id [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= g.rows || row >= g.max_rows) return;
  const PositionRow p = options[row];
  if (!p.valid) {
    if (tid == 0u) result[row] = uint2(0, 0);
    return;
  }
  if (g.vocab == 0u) {
    if (tid == 0u) result[row] = uint2(0, 2);
    return;
  }

  const uint top_k = min(p.top_k, min(g.vocab, 64u));
  const bool greedy = (p.temperature <= 0.0f || top_k <= 1u);

  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup float values[64];
  threadgroup uint ids[64];

  if (greedy) {
    float win_v = -INFINITY;
    uint win_i = g.vocab;
    uint win_bad = 0u;
    position_reduce_64_parts(parts + ulong(row) * 64ul, g.vocab, tid, lane,
                             sg_id, sg_max, sg_idx, sg_bad, win_v, win_i,
                             win_bad);
    if (tid == 0u) {
      if (!win_bad && p.temperature > 0.0f) {
        compact_ids[ulong(row) * 64ul] = win_i;
        compact_scores[ulong(row) * 64ul] = win_v;
      }
      result[row] = win_bad ? uint2(0u, 2u) : uint2(win_i, 0u);
    }
    return;
  }

  // Exact stochastic fallback (temperature > 0 && top_k > 1): identical to omlx_m5_position_sample_logits.
  device const bfloat *row_logits = logits + ulong(row) * g.vocab;
  threadgroup uint shared_bad[1];
  float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
  uint i0 = g.vocab, i1 = g.vocab, i2 = g.vocab, i3 = g.vocab;
  uint invalid = 0u;
  for (uint t = tid; t < g.vocab; t += 256u) {
    const float v = float(row_logits[t]);
    invalid |= uint(!isfinite(v));
    position_insert_top4(v, t, v0, i0, v1, i1, v2, i2, v3, i3);
  }

  float cur_v = v0;
  uint cur_i = i0;
  uint cur_bad = invalid;
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(cur_v, offset);
    const uint other_i = simd_shuffle_down(cur_i, offset);
    cur_bad |= simd_shuffle_down(cur_bad, offset);
    if (position_sample_better(other_v, other_i, cur_v, cur_i)) {
      cur_v = other_v;
      cur_i = other_i;
    }
  }
  if (lane == 0u) {
    sg_max[sg_id] = cur_v;
    sg_idx[sg_id] = cur_i;
    sg_bad[sg_id] = cur_bad;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg_id == 0u) {
    float red_v = lane < 8u ? sg_max[lane] : -INFINITY;
    uint red_i = lane < 8u ? sg_idx[lane] : g.vocab;
    uint red_bad = lane < 8u ? sg_bad[lane] : 0u;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      red_bad |= simd_shuffle_down(red_bad, offset);
      if (position_sample_better(other_v, other_i, red_v, red_i)) {
        red_v = other_v;
        red_i = other_i;
      }
    }
    if (lane == 0u) {
      values[0] = red_v;
      ids[0] = red_i;
      shared_bad[0] = red_bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (shared_bad[0]) {
    if (tid == 0u) result[row] = uint2(0u, 2u);
    return;
  }

  uint popped = 0u;
  {
    const uint win_i = ids[0];
    if (win_i < g.vocab && (win_i & 255u) == tid) {
      popped = 1u;
      v0 = v1; i0 = i1; v1 = v2; i1 = i2; v2 = v3; i2 = i3;
      v3 = -INFINITY; i3 = g.vocab;
    }
  }
  for (uint step = 1u; step < top_k; ++step) {
    position_reduce_argmax_256(v0, i0, g.vocab, lane, sg_id, sg_max, sg_idx,
                               &values[step], &ids[step]);
    const float win_v = values[step];
    const uint win_i = ids[step];
    if (win_i < g.vocab && (win_i & 255u) == tid) {
      ++popped;
      if (popped < 4u) {
        v0 = v1; i0 = i1; v1 = v2; i1 = i2; v2 = v3; i2 = i3;
        v3 = -INFINITY; i3 = g.vocab;
      } else if (step + 1u < top_k) {
        v0 = -INFINITY; v1 = -INFINITY; v2 = -INFINITY; v3 = -INFINITY;
        i0 = g.vocab; i1 = g.vocab; i2 = g.vocab; i3 = g.vocab;
        for (uint t = tid; t < g.vocab; t += 256u) {
          const float v = float(row_logits[t]);
          if (!position_sample_better(win_v, win_i, v, t)) continue;
          position_insert_top4(v, t, v0, i0, v1, i1, v2, i2, v3, i3);
        }
        popped = 0u;
      }
    }
  }
  if (tid) return;
  float prob[64];
  float total = 0.0f;
  const float max_val = values[0];
  for (uint i = 0u; i < top_k; ++i) {
    const uint id_i = ids[i];
    const float val_i = values[i];
    compact_ids[ulong(row) * 64ul + i] = id_i;
    compact_scores[ulong(row) * 64ul + i] = val_i;
    prob[i] = exp((val_i - max_val) / p.temperature);
    total += prob[i];
  }
  uint retained = top_k, flag = 0u;
  float cumulative = 0.0f;
  for (uint i = 0u; i < top_k; ++i) {
    cumulative += prob[i] / total;
    if (p.top_p < 1.0f && fabs(cumulative - p.top_p) < 2e-4f) flag = 1u;
    if (cumulative >= p.top_p) { retained = i + 1u; break; }
  }
  float score = -INFINITY, runner = -INFINITY;
  uint selected = ids[0];
  for (uint i = 0u; i < retained; ++i) {
    const uint id_i = ids[i];
    const float v = (values[i] - max_val) / p.temperature +
                    position_gumbel(p.seed, p.position, id_i, p.purpose);
    if (v > score || (v == score && id_i < selected)) {
      runner = score; score = v; selected = id_i;
    } else {
      runner = max(runner, v);
    }
  }
  if (score - runner < 2e-4f) flag = 1u;
  result[row] = uint2(selected, flag);
}

// Stage 2 of 2-stage 64-TG device-feeding sampler:
// Both greedy and boundary-fallback paths in _feed only require the global
// top-1 winner (win_v, win_i, win_bad), so Stage 2 binds parts at buffer(0)
// and keeps options/geometry at buffer(5)/buffer(6) matching _feed!
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_position_sample_logits_feed_stage2(
    device const PositionArgmaxPart *parts [[buffer(0)]],
    device uint2 *result [[buffer(1)]],
    device uint *compact_ids [[buffer(2)]],
    device float *compact_scores [[buffer(3)]],
    device uint *next_token_out [[buffer(4)]],
    constant PositionRow *options [[buffer(5)]],
    constant PositionGeometry &g [[buffer(6)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y;
  if (row >= g.rows || row >= g.max_rows) return;
  const PositionRow p = options[row];
  if (!p.valid) {
    if (tid == 0u) {
      result[row] = uint2(0u, 0u);
      if (row == 0u) next_token_out[0] = 0u;
    }
    return;
  }
  if (g.vocab == 0u) {
    if (tid == 0u) {
      result[row] = uint2(0u, 2u);
      if (row == 0u) next_token_out[0] = 0u;
    }
    return;
  }

  const uint top_k = min(p.top_k, min(g.vocab, 64u));
  const bool greedy = (p.temperature <= 0.0f || top_k <= 1u);

  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  float win_v = -INFINITY;
  uint win_i = g.vocab;
  uint win_bad = 0u;
  position_reduce_64_parts(parts + ulong(row) * 64ul, g.vocab, tid, lane, sg,
                           sg_max, sg_idx, sg_bad, win_v, win_i, win_bad);

  if (tid != 0u) return;
  if (win_bad) {
    result[row] = uint2(0u, 2u);
    if (row == 0u) next_token_out[0] = 0u;
    return;
  }
  if (greedy) {
    result[row] = uint2(win_i, 0u);
    if (row == 0u) next_token_out[0] = win_i;
    return;
  }
  compact_ids[ulong(row) * 64ul] = win_i;
  compact_scores[ulong(row) * 64ul] = win_v;
  result[row] = uint2(win_i, 1u);
  if (row == 0u) next_token_out[0] = win_i;
}

struct RequestPenaltyToken {
  uint token;
  float repetition, presence, frequency;
};

kernel void omlx_m5_position_copy_raw_logits(
    device const bfloat *source [[buffer(0)]],
    device bfloat *destination [[buffer(1)]],
    constant uint2 &shape [[buffer(2)]],
    constant uint &source_row [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  if (tid.x < shape.x && tid.y < shape.y) {
    destination[ulong(tid.y) * shape.x + tid.x] =
        source[ulong(source_row + tid.y) * shape.x + tid.x];
  }
}

kernel void omlx_m5_position_apply_sparse_penalties(
    device bfloat *logits [[buffer(0)]],
    device const RequestPenaltyToken *tokens [[buffer(1)]],
    constant uint2 *ranges [[buffer(2)]],
    constant uint2 &shape [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]) {
  if (tid.y >= shape.y || tid.x >= ranges[tid.y].y) return;
  const RequestPenaltyToken params = tokens[ranges[tid.y].x + tid.x];
  const ulong offset = ulong(tid.y) * shape.x + params.token;
  float score = float(logits[offset]);
  if (params.repetition != 1.0f) {
    score = float(bfloat(score < 0.0f ? score * params.repetition
                                      : score / params.repetition));
  }
  if (params.presence != 0.0f) {
    score = float(bfloat(score - params.presence));
  }
  if (params.frequency != 0.0f) {
    score -= params.frequency;
  }
  logits[offset] = bfloat(score);
}
