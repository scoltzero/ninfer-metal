#include "fp4_codec.h"
#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct EmbeddingParams {
  uint vocabulary;
  uint row_start;
  uint row_count;
  uint hidden_size;
  uint padded_hidden_size;
  uint tokens;
  uint reserved0;
  uint reserved1;
};

struct SamplingParams {
  uint vocabulary;
  uint top_k;
  float temperature;
  float top_p;
  ulong seed;
  ulong stream;
  ulong absolute_position;
};

inline uint vocab_affine_code(device const uchar *base,
                              device const uchar *high, uint bits,
                              uint column) {
  if (bits == 8) {
    return uint(base[column]);
  }
  const uint low = (uint(base[column / 2]) >> ((column & 1u) * 4u)) & 0x0fu;
  if (bits == 5) {
    return low | (((uint(high[column / 8]) >> (column & 7u)) & 1u) << 4u);
  }
  if (bits == 6) {
    return low |
           (((uint(high[column / 4]) >> ((column & 3u) * 2u)) & 3u) << 4u);
  }
  return low;
}

inline float vocab_decode_e2m1(uint code) { return omlx_m5_decode_fp4(code); }

inline float vocab_decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

kernel void omlx_m5_embedding_bf16(device const uint *token_ids [[buffer(0)]],
                                   device const bfloat *weights [[buffer(1)]],
                                   device bfloat *output [[buffer(2)]],
                                   constant EmbeddingParams &params
                                   [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  const ulong values = ulong(params.tokens) * ulong(params.hidden_size);
  const ulong ugid = ulong(gid);
  if (ugid >= values) {
    return;
  }
  const ulong token = ugid / ulong(params.hidden_size);
  const ulong feature = ugid - token * ulong(params.hidden_size);
  const uint id = token_ids[token];
  const ulong row_stride = ulong(
      params.padded_hidden_size ? params.padded_hidden_size : params.hidden_size);
  output[ugid] =
      id >= params.row_start && (id - params.row_start) < params.row_count
          ? weights[ulong(id - params.row_start) * row_stride + feature]
          : bfloat(0.0f);
}

template <uint Bits, uint GroupSize>
inline void
embedding_affine(device const uint *token_ids, device const uchar *base_codes,
                 device const uchar *high_codes, device const bfloat *scales,
                 device const bfloat *biases, device bfloat *output,
                 constant EmbeddingParams &params, uint gid) {
  const ulong values = ulong(params.tokens) * ulong(params.hidden_size);
  const ulong ugid = ulong(gid);
  if (ugid >= values) {
    return;
  }
  const ulong token = ugid / ulong(params.hidden_size);
  const uint feature = uint(ugid - token * ulong(params.hidden_size));
  const uint id = token_ids[token];
  if (id < params.row_start || (id - params.row_start) >= params.row_count) {
    output[ugid] = bfloat(0.0f);
    return;
  }
  const ulong row = ulong(id - params.row_start);
  const ulong padded = ulong(params.padded_hidden_size);
  const ulong base_row_bytes = Bits == 8 ? padded : padded / 2ul;
  const ulong high_row_bytes =
      Bits == 5 ? padded / 8ul : (Bits == 6 ? padded / 4ul : 0ul);
  const ulong groups_per_row = padded / ulong(GroupSize);
  device const uchar *base = base_codes + row * base_row_bytes;
  device const uchar *high = high_codes + row * high_row_bytes;
  const uint code = vocab_affine_code(base, high, Bits, feature);
  const ulong group = ulong(feature / GroupSize);
  output[ugid] =
      bfloat(fma(float(scales[row * groups_per_row + group]), float(code),
                 float(biases[row * groups_per_row + group])));
}

#define DEFINE_EMBEDDING_AFFINE(bits, group_size)                              \
  kernel void omlx_m5_embedding_q##bits##_g##group_size(                       \
      device const uint *token_ids [[buffer(0)]],                              \
      device const uchar *base_codes [[buffer(1)]],                            \
      device const uchar *high_codes [[buffer(2)]],                            \
      device const bfloat *scales [[buffer(3)]],                               \
      device const bfloat *biases [[buffer(4)]],                               \
      device bfloat *output [[buffer(5)]],                                     \
      constant EmbeddingParams &params [[buffer(6)]],                          \
      uint gid [[thread_position_in_grid]]) {                                  \
    embedding_affine<bits, group_size>(token_ids, base_codes, high_codes,      \
                                       scales, biases, output, params, gid);   \
  }

DEFINE_EMBEDDING_AFFINE(4, 64)
DEFINE_EMBEDDING_AFFINE(4, 128)
DEFINE_EMBEDDING_AFFINE(5, 64)
DEFINE_EMBEDDING_AFFINE(5, 128)
DEFINE_EMBEDDING_AFFINE(6, 64)
DEFINE_EMBEDDING_AFFINE(6, 128)
DEFINE_EMBEDDING_AFFINE(8, 64)
DEFINE_EMBEDDING_AFFINE(8, 128)

kernel void omlx_m5_embedding_mxfp4(device const uint *token_ids [[buffer(0)]],
                                    device const uchar *codes [[buffer(1)]],
                                    device const uchar *scales [[buffer(2)]],
                                    device bfloat *output [[buffer(3)]],
                                    constant EmbeddingParams &params
                                    [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
  const ulong values = ulong(params.tokens) * ulong(params.hidden_size);
  const ulong ugid = ulong(gid);
  if (ugid >= values) {
    return;
  }
  const ulong token = ugid / ulong(params.hidden_size);
  const uint feature = uint(ugid - token * ulong(params.hidden_size));
  const uint id = token_ids[token];
  if (id < params.row_start || (id - params.row_start) >= params.row_count) {
    output[ugid] = bfloat(0.0f);
    return;
  }
  const ulong row = ulong(id - params.row_start);
  const ulong padded = ulong(params.padded_hidden_size);
  const ulong code_row_bytes = padded / 2ul;
  const ulong groups_per_row = padded / 32ul;
  const uchar packed = codes[row * code_row_bytes + ulong(feature / 2u)];
  const uint code = (uint(packed) >> ((feature & 1u) * 4u)) & 0x0fu;
  output[ugid] =
      bfloat(vocab_decode_e2m1(code) *
             vocab_decode_e8m0(
                 scales[row * groups_per_row + ulong(feature / 32u)]));
}

inline ulong sampling_bits(ulong seed, ulong stream, ulong position) {
  ulong value = seed ^ (stream + 0x9E3779B97F4A7C15ul) ^
                (position * 0xD2B74407B1CE6E93ul);
  value += 0x9E3779B97F4A7C15ul;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ul;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBul;
  return value ^ (value >> 31);
}

inline bool vocab_sample_better(float v, uint id, float other_v,
                                uint other_id) {
  return v > other_v || (v == other_v && id < other_id);
}

inline void vocab_insert_top4(float v, uint id, thread float &v0,
                              thread uint &i0, thread float &v1,
                              thread uint &i1, thread float &v2,
                              thread uint &i2, thread float &v3,
                              thread uint &i3) {
  if (vocab_sample_better(v, id, v3, i3)) {
    if (vocab_sample_better(v, id, v2, i2)) {
      v3 = v2;
      i3 = i2;
      if (vocab_sample_better(v, id, v1, i1)) {
        v2 = v1;
        i2 = i1;
        if (vocab_sample_better(v, id, v0, i0)) {
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

inline void vocab_reduce_argmax_256(float cur_v, uint cur_i, uint vocabulary,
                                    uint lane, uint sg,
                                    threadgroup float *sg_max,
                                    threadgroup uint *sg_idx,
                                    threadgroup float *out_val,
                                    threadgroup uint *out_idx) {
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(cur_v, offset);
    const uint other_i = simd_shuffle_down(cur_i, offset);
    if (vocab_sample_better(other_v, other_i, cur_v, cur_i)) {
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
    uint red_i = lane < 8 ? sg_idx[lane] : vocabulary;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      if (vocab_sample_better(other_v, other_i, red_v, red_i)) {
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

inline uint vocab_is_bad_logit(float v) {
  return uint(!isfinite(v) && v != -INFINITY);
}

inline void vocab_reduce_argmax_bad_256(float cur_v, uint cur_i, uint cur_bad,
                                        uint vocabulary, uint lane, uint sg,
                                        threadgroup float *sg_max,
                                        threadgroup uint *sg_idx,
                                        threadgroup uint *sg_bad,
                                        threadgroup float *out_val,
                                        threadgroup uint *out_idx,
                                        threadgroup uint *out_bad) {
  for (ushort offset = 16; offset > 0; offset >>= 1) {
    const float other_v = simd_shuffle_down(cur_v, offset);
    const uint other_i = simd_shuffle_down(cur_i, offset);
    cur_bad |= simd_shuffle_down(cur_bad, offset);
    if (vocab_sample_better(other_v, other_i, cur_v, cur_i)) {
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
    uint red_i = lane < 8 ? sg_idx[lane] : vocabulary;
    uint red_bad = lane < 8 ? sg_bad[lane] : 0u;
    for (ushort offset = 4; offset > 0; offset >>= 1) {
      const float other_v = simd_shuffle_down(red_v, offset);
      const uint other_i = simd_shuffle_down(red_i, offset);
      red_bad |= simd_shuffle_down(red_bad, offset);
      if (vocab_sample_better(other_v, other_i, red_v, red_i)) {
        red_v = other_v;
        red_i = other_i;
      }
    }
    if (lane == 0) {
      *out_val = red_v;
      *out_idx = red_i;
      *out_bad = red_bad;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline void sample_logits_parallel(device const bfloat *logits,
                                   device uint *out_token,
                                   const SamplingParams sp, uint thread_index,
                                   uint lane, uint sg,
                                   threadgroup float *sg_max,
                                   threadgroup uint *sg_idx,
                                   threadgroup uint *sg_bad,
                                   threadgroup uint *any_bad,
                                   threadgroup float *top_values,
                                   threadgroup uint *top_indices) {
  if (sp.vocabulary == 0u) {
    if (thread_index == 0) {
      *out_token = 0xffffffffu;
    }
    return;
  }
  if (sp.temperature <= 0.0f) {
    float local_maximum = -INFINITY;
    uint local_index = sp.vocabulary;
    uint local_bad = 0u;
    for (uint index = thread_index; index < sp.vocabulary; index += 256) {
      const float value = float(logits[index]);
      local_bad |= vocab_is_bad_logit(value);
      if (vocab_sample_better(value, index, local_maximum, local_index)) {
        local_maximum = value;
        local_index = index;
      }
    }
    vocab_reduce_argmax_bad_256(local_maximum, local_index, local_bad,
                                sp.vocabulary, lane, sg, sg_max, sg_idx, sg_bad,
                                &top_values[0], &top_indices[0], any_bad);
    if (thread_index == 0) {
      if (*any_bad != 0u) {
        *out_token = 0xffffffffu;
      } else if (top_indices[0] >= sp.vocabulary || !isfinite(top_values[0])) {
        *out_token = sp.vocabulary;
      } else {
        *out_token = top_indices[0];
      }
    }
    return;
  }

  const uint top_k = min(sp.top_k, 64u);
  if (top_k == 0) {
    if (thread_index == 0) {
      *out_token = 0xffffffffu;
    }
    return;
  }
  if (top_k == 1) {
    float v0 = -INFINITY;
    uint i0 = sp.vocabulary;
    uint local_bad = 0u;
    for (uint index = thread_index; index < sp.vocabulary; index += 256) {
      const float value = float(logits[index]);
      local_bad |= vocab_is_bad_logit(value);
      if (vocab_sample_better(value, index, v0, i0)) {
        v0 = value;
        i0 = index;
      }
    }
    vocab_reduce_argmax_bad_256(v0, i0, local_bad, sp.vocabulary, lane, sg,
                                sg_max, sg_idx, sg_bad, &top_values[0],
                                &top_indices[0], any_bad);
  } else {
    float v0 = -INFINITY, v1 = -INFINITY, v2 = -INFINITY, v3 = -INFINITY;
    uint i0 = sp.vocabulary, i1 = sp.vocabulary, i2 = sp.vocabulary,
         i3 = sp.vocabulary;
    uint local_bad = 0u;
    for (uint index = thread_index; index < sp.vocabulary; index += 256) {
      const float value = float(logits[index]);
      local_bad |= vocab_is_bad_logit(value);
      vocab_insert_top4(value, index, v0, i0, v1, i1, v2, i2, v3, i3);
    }
    uint popped = 0;
    for (uint step = 0; step < top_k; ++step) {
      if (step == 0) {
        vocab_reduce_argmax_bad_256(v0, i0, local_bad, sp.vocabulary, lane, sg,
                                    sg_max, sg_idx, sg_bad, &top_values[0],
                                    &top_indices[0], any_bad);
        if (*any_bad != 0u) {
          if (thread_index == 0) {
            *out_token = 0xffffffffu;
          }
          return;
        }
        if (top_indices[0] >= sp.vocabulary || !isfinite(top_values[0])) {
          if (thread_index == 0) {
            *out_token = sp.vocabulary;
          }
          return;
        }
      } else {
        vocab_reduce_argmax_256(v0, i0, sp.vocabulary, lane, sg, sg_max, sg_idx,
                                &top_values[step], &top_indices[step]);
      }
      const float win_v = top_values[step];
      const uint win_i = top_indices[step];
      if (win_i < sp.vocabulary && (win_i & 255u) == thread_index) {
        ++popped;
        if (popped < 4u) {
          v0 = v1;
          i0 = i1;
          v1 = v2;
          i1 = i2;
          v2 = v3;
          i2 = i3;
          v3 = -INFINITY;
          i3 = sp.vocabulary;
        } else if (step + 1u < top_k) {
          v0 = -INFINITY;
          v1 = -INFINITY;
          v2 = -INFINITY;
          v3 = -INFINITY;
          i0 = sp.vocabulary;
          i1 = sp.vocabulary;
          i2 = sp.vocabulary;
          i3 = sp.vocabulary;
          for (uint index = thread_index; index < sp.vocabulary; index += 256) {
            const float value = float(logits[index]);
            if (!vocab_sample_better(win_v, win_i, value, index)) {
              continue;
            }
            vocab_insert_top4(value, index, v0, i0, v1, i1, v2, i2, v3, i3);
          }
          popped = 0;
        }
      }
    }
  }

  if (thread_index != 0) {
    return;
  }
  if (*any_bad != 0u) {
    *out_token = 0xffffffffu;
    return;
  }
  if (top_indices[0] >= sp.vocabulary || !isfinite(top_values[0])) {
    *out_token = sp.vocabulary;
    return;
  }
  float probabilities[64];
  float total = 0.0f;
  const float max_logit = top_values[0];
  uint valid_count = 0;
  for (uint index = 0; index < top_k; ++index) {
    if (top_indices[index] >= sp.vocabulary || !isfinite(top_values[index])) {
      break;
    }
    probabilities[index] =
        exp((top_values[index] - max_logit) / sp.temperature);
    total += probabilities[index];
    ++valid_count;
  }
  if (valid_count == 0 || !(total > 0.0f) || !isfinite(total)) {
    *out_token = 0xffffffffu;
    return;
  }
  float cumulative = 0.0f;
  uint retained = valid_count;
  for (uint index = 0; index < valid_count; ++index) {
    cumulative += probabilities[index] / total;
    if (cumulative >= sp.top_p) {
      retained = index + 1;
      break;
    }
  }
  float retained_total = 0.0f;
  for (uint index = 0; index < retained; ++index) {
    retained_total += probabilities[index];
  }
  const ulong bits =
      sampling_bits(sp.seed, sp.stream, sp.absolute_position);
  const float uniform = float((bits >> 40) & 0xFFFFFFul) / 16777216.0f;
  const float target = uniform * retained_total;
  float prefix = 0.0f;
  uint selected = top_indices[retained - 1];
  for (uint index = 0; index < retained; ++index) {
    prefix += probabilities[index];
    if (target < prefix) {
      selected = top_indices[index];
      break;
    }
  }
  *out_token = selected;
}

kernel void omlx_m5_sample_logits(device const bfloat *logits [[buffer(0)]],
                                  device uint *selected_token [[buffer(1)]],
                                  constant SamplingParams &params [[buffer(2)]],
                                  uint thread_index
                                  [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup uint any_bad;
  threadgroup float top_values[64];
  threadgroup uint top_indices[64];
  sample_logits_parallel(logits, selected_token, params, thread_index, lane, sg,
                         sg_max, sg_idx, sg_bad, &any_bad, top_values,
                         top_indices);
}

kernel void omlx_m5_sample_logits_feed(
    device const bfloat *logits [[buffer(0)]],
    device uint *selected_token [[buffer(1)]],
    device uint *next_token_out [[buffer(2)]],
    constant SamplingParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup uint any_bad;
  threadgroup float top_values[64];
  threadgroup uint top_indices[64];
  sample_logits_parallel(logits, selected_token, params, thread_index, lane, sg,
                         sg_max, sg_idx, sg_bad, &any_bad, top_values,
                         top_indices);
  if (thread_index == 0) {
    next_token_out[0] = *selected_token;
  }
}

// P10: one sampler threadgroup per ready row. Every resource binding stays
// attached to its installed physical slot; compact rows carry explicit slot,
// width and validity so a cancelled row cannot consume another row's RNG.
struct BatchSamplingParams {
  uint count;
  uint context_envelope;
  uint valid[8];
  uint slot[8];
  uint width[8];
  SamplingParams sampling[8];
};

kernel void omlx_m5_sample_logits_batched(
    device const bfloat *slot0 [[buffer(0)]],
    device const bfloat *slot1 [[buffer(1)]],
    device const bfloat *slot2 [[buffer(2)]],
    device const bfloat *slot3 [[buffer(3)]],
    device const bfloat *slot4 [[buffer(4)]],
    device const bfloat *slot5 [[buffer(5)]],
    device const bfloat *slot6 [[buffer(6)]],
    device const bfloat *slot7 [[buffer(7)]],
    device uint *selected_token [[buffer(8)]],
    constant BatchSamplingParams &batch [[buffer(9)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
  const uint row = group.y;
  if (row >= batch.count || batch.valid[row] == 0) return;
  const uint slot = batch.slot[row];
  if (slot >= 8 || batch.width[row] == 0 || batch.width[row] > 8 ||
      batch.sampling[row].vocabulary == 0 ||
      batch.sampling[row].absolute_position >= batch.context_envelope) return;
  device const bfloat *logits =
      slot == 0 ? slot0 : slot == 1 ? slot1 : slot == 2 ? slot2 :
      slot == 3 ? slot3 : slot == 4 ? slot4 : slot == 5 ? slot5 :
      slot == 6 ? slot6 : slot7;
  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup uint any_bad;
  threadgroup float top_values[64];
  threadgroup uint top_indices[64];
  sample_logits_parallel(logits, selected_token + row, batch.sampling[row],
                         thread_index, lane, sg, sg_max, sg_idx, sg_bad,
                         &any_bad, top_values, top_indices);
}


struct VocabArgmaxPart {
  float value;
  uint index;
};

// Stage 1 of 2-stage 64-TG parallel greedy argmax for LegacyRejectionV1:
// Grid: {.width = 64 * 256}, Threads: {.width = 256}.
[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_vocab_argmax_stage1_64tg(
    device const bfloat *logits [[buffer(0)]],
    device VocabArgmaxPart *parts [[buffer(1)]],
    constant uint &vocabulary [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint chunk = tg.x;
  if (chunk >= 64u) return;
  if (vocabulary == 0u) {
    if (tid == 0u) parts[chunk] = VocabArgmaxPart{-INFINITY, 0xffffffffu};
    return;
  }

  const bool aligned4 = (reinterpret_cast<ulong>(logits) & 7ul) == 0ul;
  const uint vocab4 = vocabulary >> 2u;
  float v0 = -INFINITY;
  uint i0 = vocabulary;
  uint local_bad = 0u;

  if (aligned4) {
    device const bfloat4 *logits4 =
        reinterpret_cast<device const bfloat4 *>(logits);
    for (uint vec_idx = chunk * 256u + tid; vec_idx < vocab4;
         vec_idx += 64u * 256u) {
      const float4 fv = float4(logits4[vec_idx]);
      local_bad |= vocab_is_bad_logit(fv.x) | vocab_is_bad_logit(fv.y) |
                   vocab_is_bad_logit(fv.z) | vocab_is_bad_logit(fv.w);
      const uint base = vec_idx << 2u;
      if (vocab_sample_better(fv.x, base + 0u, v0, i0)) {
        v0 = fv.x;
        i0 = base + 0u;
      }
      if (vocab_sample_better(fv.y, base + 1u, v0, i0)) {
        v0 = fv.y;
        i0 = base + 1u;
      }
      if (vocab_sample_better(fv.z, base + 2u, v0, i0)) {
        v0 = fv.z;
        i0 = base + 2u;
      }
      if (vocab_sample_better(fv.w, base + 3u, v0, i0)) {
        v0 = fv.w;
        i0 = base + 3u;
      }
    }

    if (chunk == 0u) {
      const uint t = (vocab4 << 2u) + tid;
      if (t < vocabulary) {
        const float v = float(logits[t]);
        local_bad |= vocab_is_bad_logit(v);
        if (vocab_sample_better(v, t, v0, i0)) {
          v0 = v;
          i0 = t;
        }
      }
    }
  } else {
    for (uint t = chunk * 256u + tid; t < vocabulary; t += 64u * 256u) {
      const float v = float(logits[t]);
      local_bad |= vocab_is_bad_logit(v);
      if (vocab_sample_better(v, t, v0, i0)) {
        v0 = v;
        i0 = t;
      }
    }
  }

  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup float out_val;
  threadgroup uint out_idx;
  threadgroup uint out_bad;
  vocab_reduce_argmax_bad_256(v0, i0, local_bad, vocabulary, lane, sg, sg_max,
                              sg_idx, sg_bad, &out_val, &out_idx, &out_bad);
  if (tid == 0u) {
    parts[chunk] = (out_bad != 0u) ? VocabArgmaxPart{NAN, 0xffffffffu}
                                   : VocabArgmaxPart{out_val, out_idx};
  }
}

inline void sample_logits_stage2_impl(
    device const bfloat *logits,
    device const VocabArgmaxPart *parts,
    device uint *out_token,
    const SamplingParams sp,
    uint tid,
    uint lane,
    uint sg,
    threadgroup float *sg_max,
    threadgroup uint *sg_idx,
    threadgroup uint *sg_bad,
    threadgroup uint *any_bad,
    threadgroup float *top_values,
    threadgroup uint *top_indices) {
  if (sp.vocabulary == 0u) {
    if (tid == 0u) {
      *out_token = 0xffffffffu;
    }
    return;
  }
  const uint top_k = min(sp.top_k, 64u);
  if (sp.temperature <= 0.0f || top_k == 1u) {
    if (sp.temperature > 0.0f && top_k == 0u) {
      if (tid == 0u) {
        *out_token = 0xffffffffu;
      }
      return;
    }
    float cur_v = -INFINITY;
    uint cur_i = sp.vocabulary;
    uint cur_bad = 0u;
    if (tid < 64u) {
      const VocabArgmaxPart pt = parts[tid];
      cur_v = pt.value;
      cur_i = pt.index;
      cur_bad = vocab_is_bad_logit(cur_v) | uint(cur_i == 0xffffffffu);
    }
    if (sg < 2u) {
      for (ushort offset = 16; offset > 0; offset >>= 1) {
        const float other_v = simd_shuffle_down(cur_v, offset);
        const uint other_i = simd_shuffle_down(cur_i, offset);
        cur_bad |= simd_shuffle_down(cur_bad, offset);
        if (vocab_sample_better(other_v, other_i, cur_v, cur_i)) {
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
    if (tid == 0u) {
      if ((sg_bad[0] | sg_bad[1]) != 0u) {
        *out_token = 0xffffffffu;
      } else {
        float win_v = sg_max[0];
        uint win_i = sg_idx[0];
        if (vocab_sample_better(sg_max[1], sg_idx[1], win_v, win_i)) {
          win_v = sg_max[1];
          win_i = sg_idx[1];
        }
        *out_token = (win_i < sp.vocabulary && isfinite(win_v))
                         ? win_i
                         : sp.vocabulary;
      }
    }
    return;
  }

  sample_logits_parallel(logits, out_token, sp, tid, lane, sg, sg_max, sg_idx,
                         sg_bad, any_bad, top_values, top_indices);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_sample_logits_stage2(
    device const bfloat *logits [[buffer(0)]],
    device const VocabArgmaxPart *parts [[buffer(1)]],
    device uint *selected_token [[buffer(2)]],
    constant SamplingParams &params [[buffer(3)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup uint any_bad;
  threadgroup float top_values[64];
  threadgroup uint top_indices[64];
  sample_logits_stage2_impl(logits, parts, selected_token, params, thread_index,
                            lane, sg, sg_max, sg_idx, sg_bad, &any_bad,
                            top_values, top_indices);
}

[[max_total_threads_per_threadgroup(256)]]
kernel void omlx_m5_sample_logits_feed_stage2(
    device const bfloat *logits [[buffer(0)]],
    device const VocabArgmaxPart *parts [[buffer(1)]],
    device uint *selected_token [[buffer(2)]],
    device uint *next_token_out [[buffer(3)]],
    constant SamplingParams &params [[buffer(4)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sg_max[8];
  threadgroup uint sg_idx[8];
  threadgroup uint sg_bad[8];
  threadgroup uint any_bad;
  threadgroup float top_values[64];
  threadgroup uint top_indices[64];
  sample_logits_stage2_impl(logits, parts, selected_token, params, thread_index,
                            lane, sg, sg_max, sg_idx, sg_bad, &any_bad,
                            top_values, top_indices);
  if (thread_index == 0u) {
    next_token_out[0] = *selected_token;
  }
}
