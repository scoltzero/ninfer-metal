#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct RmsNormProjectionParams {
  uint input_features;
  uint tokens;
  float epsilon;
  float weight_offset;
};

inline float rms_decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

inline uchar rms_encode_e8m0_round_up(float value) {
  if (!(value > 0.0f)) {
    return 0;
  }
  const int exponent = clamp(int(ceil(log2(value))), -127, 127);
  return uchar(exponent + 127);
}

inline uchar rms_encode_e2m1(float value) {
  const uchar sign = signbit(value) ? 8u : 0u;
  const float magnitude = abs(value);
  uchar code = 0;
  if (magnitude > 5.0f) {
    code = 7;
  } else if (magnitude >= 3.5f) {
    code = 6;
  } else if (magnitude > 2.5f) {
    code = 5;
  } else if (magnitude >= 1.75f) {
    code = 4;
  } else if (magnitude > 1.25f) {
    code = 3;
  } else if (magnitude >= 0.75f) {
    code = 2;
  } else if (magnitude > 0.25f) {
    code = 1;
  }
  return code | sign;
}

template <bool ProduceA16, bool ProduceA4>
inline void
rmsnorm_projection(device const bfloat *input, device const bfloat *weight,
                   device bfloat *output_a16, device uchar *output_codes,
                   device uchar *output_scales,
                   constant RmsNormProjectionParams &params,
                   threadgroup float *simd_sums, threadgroup float &inverse_rms,
                   uint thread_index, uint lane, uint simdgroup, uint token) {
  constexpr uint threads = 128;
  constexpr uint simdgroups = 4;
  if (token >= params.tokens)
    return;
  float square_sum = 0.0f;
  for (uint column = thread_index; column < params.input_features;
       column += threads) {
    const float value = float(input[token * params.input_features + column]);
    square_sum = fma(value, value, square_sum);
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0) {
    simd_sums[simdgroup] = square_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (thread_index == 0) {
    float total = 0.0f;
    for (uint index = 0; index < simdgroups; ++index) {
      total += simd_sums[index];
    }
    inverse_rms = rsqrt(total / float(params.input_features) + params.epsilon);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if constexpr (ProduceA16) {
    for (uint column = thread_index; column < params.input_features;
         column += threads) {
      const float normalized =
          float(input[token * params.input_features + column]) * inverse_rms *
          (float(weight[column]) + params.weight_offset);
      output_a16[token * params.input_features + column] = bfloat(normalized);
    }
  }

  if constexpr (ProduceA4) {
    const uint groups = params.input_features / 32;
    for (uint group = simdgroup; group < groups; group += simdgroups) {
      const uint column = group * 32 + lane;
      // Quantize the BF16 representation that the unfused A16 path publishes.
      const bfloat represented =
          bfloat(float(input[token * params.input_features + column]) *
                 inverse_rms * (float(weight[column]) + params.weight_offset));
      const float value = float(represented);
      const float maximum = simd_max(abs(value));
      const uchar scale_code = rms_encode_e8m0_round_up(maximum / 6.0f);
      const float scale = rms_decode_e8m0(scale_code);
      const uchar code = rms_encode_e2m1(value / scale);
      if (lane == 0) {
        output_scales[token * groups + group] = scale_code;
      }
      const uint high = simd_shuffle_down(uint(code), 1);
      if ((lane & 1u) == 0) {
        output_codes[token * (params.input_features / 2) + group * 16 +
                     lane / 2] = code | uchar(high << 4);
      }
    }
  }
}

kernel void
omlx_m5_rmsnorm_a16(device const bfloat *input [[buffer(0)]],
                    device const bfloat *weight [[buffer(1)]],
                    device bfloat *output_a16 [[buffer(2)]],
                    constant RmsNormProjectionParams &params [[buffer(3)]],
                    uint thread_index [[thread_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]],
                    uint simdgroup [[simdgroup_index_in_threadgroup]],
                    uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup float simd_sums[4];
  threadgroup float inverse_rms;
  if (thread_index == 0) {
    inverse_rms = 0.0f;
  }
  rmsnorm_projection<true, false>(input, weight, output_a16, nullptr, nullptr,
                                  params, simd_sums, inverse_rms, thread_index,
                                  lane, simdgroup, tg_position.y);
}

kernel void
omlx_m5_rmsnorm_a4(device const bfloat *input [[buffer(0)]],
                   device const bfloat *weight [[buffer(1)]],
                   device uchar *output_codes [[buffer(2)]],
                   device uchar *output_scales [[buffer(3)]],
                   constant RmsNormProjectionParams &params [[buffer(4)]],
                   uint thread_index [[thread_index_in_threadgroup]],
                   uint lane [[thread_index_in_simdgroup]],
                   uint simdgroup [[simdgroup_index_in_threadgroup]],
                   uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup float simd_sums[4];
  threadgroup float inverse_rms;
  if (thread_index == 0) {
    inverse_rms = 0.0f;
  }
  rmsnorm_projection<false, true>(input, weight, nullptr, output_codes,
                                  output_scales, params, simd_sums, inverse_rms,
                                  thread_index, lane, simdgroup, tg_position.y);
}

kernel void
omlx_m5_rmsnorm_a16_a4(device const bfloat *input [[buffer(0)]],
                       device const bfloat *weight [[buffer(1)]],
                       device bfloat *output_a16 [[buffer(2)]],
                       device uchar *output_codes [[buffer(3)]],
                       device uchar *output_scales [[buffer(4)]],
                       constant RmsNormProjectionParams &params [[buffer(5)]],
                       uint thread_index [[thread_index_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint simdgroup [[simdgroup_index_in_threadgroup]],
                       uint3 tg_position [[threadgroup_position_in_grid]]) {
  threadgroup float simd_sums[4];
  threadgroup float inverse_rms;
  if (thread_index == 0) {
    inverse_rms = 0.0f;
  }
  rmsnorm_projection<true, true>(input, weight, output_a16, output_codes,
                                 output_scales, params, simd_sums, inverse_rms,
                                 thread_index, lane, simdgroup, tg_position.y);
}

// NInfer-style fused residual add + RMSNorm: updates hidden in-place and emits
// normalized output in a single dispatch, eliminating 128 dispatches + barriers
// on 64-layer models while remaining bit-exact with forward_add + rmsnorm_a16.
kernel void
omlx_m5_add_rmsnorm_a16(device bfloat *hidden [[buffer(0)]],
                        device const bfloat *delta [[buffer(1)]],
                        device const bfloat *weight [[buffer(2)]],
                        device bfloat *output_a16 [[buffer(3)]],
                        constant RmsNormProjectionParams &params [[buffer(4)]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                        uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 128;
  const uint token = tg_position.y;
  if (token >= params.tokens)
    return;
  threadgroup float simd_sums[4];
  const ulong row_base = ulong(token) * params.input_features;
  device bfloat *h_row = hidden + row_base;
  device const bfloat *d_row = delta + row_base;
  device bfloat *o_row = output_a16 + row_base;

  if (params.input_features == 5120u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(o_row);
    bfloat4 reg_s[10];
    float square_sum = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = thread_index + i * threads;
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      reg_s[i] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) {
      simd_sums[simdgroup] = square_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total * (1.0f / 5120.0f) + params.epsilon);
    const float4 off4 = float4(params.weight_offset);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 10u; ++i) {
      const uint v = thread_index + i * threads;
      ov[v] = bfloat4(float4(reg_s[i]) * inv * (float4(wv[v]) + off4));
    }
    return;
  }
  if (params.input_features == 2048u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(o_row);
    bfloat4 reg_s[4];
    float square_sum = 0.0f;
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = thread_index + i * threads;
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      reg_s[i] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
    square_sum = simd_sum(square_sum);
    if (lane == 0u) {
      simd_sums[simdgroup] = square_sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total =
        ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
    const float inv = rsqrt(total * (1.0f / 2048.0f) + params.epsilon);
    const float4 off4 = float4(params.weight_offset);
#pragma clang loop unroll(full)
    for (uint i = 0u; i < 4u; ++i) {
      const uint v = thread_index + i * threads;
      ov[v] = bfloat4(float4(reg_s[i]) * inv * (float4(wv[v]) + off4));
    }
    return;
  }

  float square_sum = 0.0f;
  if ((params.input_features & 511u) == 0u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    const uint vecs = params.input_features >> 2u;
    for (uint v = thread_index; v < vecs; v += threads) {
      const bfloat4 h4 = hv[v];
      const bfloat4 d4 = dv[v];
      const bfloat4 s4 = bfloat4(float4(h4) + float4(d4));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
  } else {
    for (uint column = thread_index; column < params.input_features;
         column += threads) {
      const bfloat sum = bfloat(float(h_row[column]) + float(d_row[column]));
      h_row[column] = sum;
      const float value = float(sum);
      square_sum = fma(value, value, square_sum);
    }
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0) {
    simd_sums[simdgroup] = square_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv =
      rsqrt(total / float(params.input_features) + params.epsilon);
  if ((params.input_features & 511u) == 0u) {
    device const bfloat4 *hv = reinterpret_cast<device const bfloat4 *>(h_row);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(o_row);
    const uint vecs = params.input_features >> 2u;
    const float4 off4 = float4(params.weight_offset);
    for (uint v = thread_index; v < vecs; v += threads) {
      const float4 sf = float4(hv[v]);
      const float4 wf = float4(wv[v]) + off4;
      ov[v] = bfloat4(sf * inv * wf);
    }
  } else {
    for (uint column = thread_index; column < params.input_features;
         column += threads) {
      const float normalized = float(h_row[column]) * inv *
                               (float(weight[column]) + params.weight_offset);
      o_row[column] = bfloat(normalized);
    }
  }
}

// Fused RMSNorm -> FP16 output (for M5 MXFP4 NAX) with automatic tail-row zero padding.
// Grid height = mp (multiple of 8 >= params.tokens). Rows >= params.tokens are zeroed.
kernel void
omlx_m5_rmsnorm_a16_f16(device const bfloat *input_a16 [[buffer(0)]],
                        device const bfloat *weight [[buffer(1)]],
                        device half *output_f16 [[buffer(2)]],
                        constant RmsNormProjectionParams &params [[buffer(3)]],
                        uint thread_index [[thread_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]],
                        uint simdgroup [[simdgroup_index_in_threadgroup]],
                        uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 128;
  const uint token = tg_position.y;
  const ulong row_base = ulong(token) * params.input_features;
  device half *o_row = output_f16 + row_base;

  if (token >= params.tokens) {
    if ((params.input_features & 3u) == 0u) {
      device half4 *ov = reinterpret_cast<device half4 *>(o_row);
      const uint vecs = params.input_features >> 2u;
      for (uint v = thread_index; v < vecs; v += threads) {
        ov[v] = half4(0.0h);
      }
    } else {
      for (uint c = thread_index; c < params.input_features; c += threads) {
        o_row[c] = half(0.0h);
      }
    }
    return;
  }

  threadgroup float simd_sums[4];
  device const bfloat *x_row = input_a16 + row_base;

  float square_sum = 0.0f;
  if ((params.input_features & 511u) == 0u) {
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(x_row);
    const uint vecs = params.input_features >> 2u;
    for (uint v = thread_index; v < vecs; v += threads) {
      const float4 sf = float4(xv[v]);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
  } else {
    for (uint c = thread_index; c < params.input_features; c += threads) {
      const float value = float(x_row[c]);
      square_sum = fma(value, value, square_sum);
    }
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0) {
    simd_sums[simdgroup] = square_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv =
      rsqrt(total / float(params.input_features) + params.epsilon);
  if ((params.input_features & 511u) == 0u) {
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(x_row);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device half4 *ov = reinterpret_cast<device half4 *>(o_row);
    const uint vecs = params.input_features >> 2u;
    const float4 off4 = float4(params.weight_offset);
    for (uint v = thread_index; v < vecs; v += threads) {
      const float4 sf = float4(xv[v]);
      const float4 wf = float4(wv[v]) + off4;
      const float4 bf = float4(bfloat4(sf * inv * wf));
      ov[v] = half4(clamp(bf, float4(-65504.0f), float4(65504.0f)));
    }
  } else {
    for (uint c = thread_index; c < params.input_features; c += threads) {
      const float normalized = float(bfloat(
          float(x_row[c]) * inv * (float(weight[c]) + params.weight_offset)));
      o_row[c] = half(clamp(normalized, -65504.0f, 65504.0f));
    }
  }
}

// Fused residual-add + RMSNorm -> FP16 output (for M5 MXFP4 NAX) with tail-row zero padding.
kernel void
omlx_m5_add_rmsnorm_a16_f16(device bfloat *hidden [[buffer(0)]],
                            device const bfloat *delta [[buffer(1)]],
                            device const bfloat *weight [[buffer(2)]],
                            device half *output_f16 [[buffer(3)]],
                            constant RmsNormProjectionParams &params [[buffer(4)]],
                            uint thread_index [[thread_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint simdgroup [[simdgroup_index_in_threadgroup]],
                            uint3 tg_position [[threadgroup_position_in_grid]]) {
  constexpr uint threads = 128;
  const uint token = tg_position.y;
  const ulong row_base = ulong(token) * params.input_features;
  device half *o_row = output_f16 + row_base;

  if (token >= params.tokens) {
    if ((params.input_features & 3u) == 0u) {
      device half4 *ov = reinterpret_cast<device half4 *>(o_row);
      const uint vecs = params.input_features >> 2u;
      for (uint v = thread_index; v < vecs; v += threads) {
        ov[v] = half4(0.0h);
      }
    } else {
      for (uint c = thread_index; c < params.input_features; c += threads) {
        o_row[c] = half(0.0h);
      }
    }
    return;
  }

  threadgroup float simd_sums[4];
  device bfloat *h_row = hidden + row_base;
  device const bfloat *d_row = delta + row_base;

  float square_sum = 0.0f;
  if ((params.input_features & 511u) == 0u) {
    device bfloat4 *hv = reinterpret_cast<device bfloat4 *>(h_row);
    device const bfloat4 *dv = reinterpret_cast<device const bfloat4 *>(d_row);
    const uint vecs = params.input_features >> 2u;
    for (uint v = thread_index; v < vecs; v += threads) {
      const bfloat4 s4 = bfloat4(float4(hv[v]) + float4(dv[v]));
      hv[v] = s4;
      const float4 sf = float4(s4);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
  } else {
    for (uint c = thread_index; c < params.input_features; c += threads) {
      const bfloat sum = bfloat(float(h_row[c]) + float(d_row[c]));
      h_row[c] = sum;
      const float value = float(sum);
      square_sum = fma(value, value, square_sum);
    }
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0) {
    simd_sums[simdgroup] = square_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv =
      rsqrt(total / float(params.input_features) + params.epsilon);
  if ((params.input_features & 511u) == 0u) {
    device const bfloat4 *hv = reinterpret_cast<device const bfloat4 *>(h_row);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device half4 *ov = reinterpret_cast<device half4 *>(o_row);
    const uint vecs = params.input_features >> 2u;
    const float4 off4 = float4(params.weight_offset);
    for (uint v = thread_index; v < vecs; v += threads) {
      const float4 sf = float4(hv[v]);
      const float4 wf = float4(wv[v]) + off4;
      const float4 bf = float4(bfloat4(sf * inv * wf));
      ov[v] = half4(clamp(bf, float4(-65504.0f), float4(65504.0f)));
    }
  } else {
    for (uint c = thread_index; c < params.input_features; c += threads) {
      const float normalized = float(bfloat(
          float(h_row[c]) * inv * (float(weight[c]) + params.weight_offset)));
      o_row[c] = half(clamp(normalized, -65504.0f, 65504.0f));
    }
  }
}

struct RmsNormInitZeroParams {
  uint input_features;
  uint num_sq_words;
  float epsilon;
  float weight_offset;
  uint num_moe_words;
};

// Layer-0 T=1 RMSNorm + one-shot zeroing of rms_sq_buf[0..num_sq_words-1] & moe_tg_done_buf[0..num_moe_words-1]
// so zero-dispatch epilogue/prologue norms & 1-dispatch MoE router+top8 work across infinite ICB replays!
kernel void
omlx_m5_rmsnorm_a16_init_zero_t1(device const bfloat *input [[buffer(0)]],
                                 device const bfloat *weight [[buffer(1)]],
                                 device bfloat *output_a16 [[buffer(2)]],
                                 device float *rms_sq_buf [[buffer(3)]],
                                 device uint *moe_tg_done_buf [[buffer(4)]],
                                 constant RmsNormInitZeroParams &params [[buffer(5)]],
                                 uint thread_index [[thread_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simdgroup [[simdgroup_index_in_threadgroup]]) {
  constexpr uint threads = 128;
  for (uint i = thread_index; i < params.num_sq_words; i += threads) {
    rms_sq_buf[i] = 0.0f;
  }
  for (uint i = thread_index; i < params.num_moe_words; i += threads) {
    moe_tg_done_buf[i] = 0u;
  }
  threadgroup float simd_sums[4];
  float square_sum = 0.0f;
  if ((params.input_features & 3u) == 0u) {
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(input);
    const uint vecs = params.input_features >> 2u;
    for (uint v = thread_index; v < vecs; v += threads) {
      const float4 sf = float4(xv[v]);
      square_sum = fma(sf.x, sf.x, square_sum);
      square_sum = fma(sf.y, sf.y, square_sum);
      square_sum = fma(sf.z, sf.z, square_sum);
      square_sum = fma(sf.w, sf.w, square_sum);
    }
  } else {
    for (uint column = thread_index; column < params.input_features;
         column += threads) {
      const float value = float(input[column]);
      square_sum = fma(value, value, square_sum);
    }
  }
  square_sum = simd_sum(square_sum);
  if (lane == 0u) {
    simd_sums[simdgroup] = square_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float total =
      ((simd_sums[0] + simd_sums[1]) + simd_sums[2]) + simd_sums[3];
  const float inv =
      rsqrt(total / float(params.input_features) + params.epsilon);
  if ((params.input_features & 3u) == 0u) {
    device const bfloat4 *xv = reinterpret_cast<device const bfloat4 *>(input);
    device const bfloat4 *wv = reinterpret_cast<device const bfloat4 *>(weight);
    device bfloat4 *ov = reinterpret_cast<device bfloat4 *>(output_a16);
    const uint vecs = params.input_features >> 2u;
    const float4 off4 = float4(params.weight_offset);
    for (uint v = thread_index; v < vecs; v += threads) {
      ov[v] = bfloat4(float4(xv[v]) * inv * (float4(wv[v]) + off4));
    }
  } else {
    for (uint column = thread_index; column < params.input_features;
         column += threads) {
      output_a16[column] = bfloat(float(input[column]) * inv *
                                  (float(weight[column]) + params.weight_offset));
    }
  }
}
