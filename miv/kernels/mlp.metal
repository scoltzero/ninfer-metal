#include <metal_simdgroup>
#include <metal_stdlib>

using namespace metal;

struct SwiGluParams {
  uint rows;
  uint features;
  uint groups;
  uint reserved;
};

struct MoeReduceParams {
  uint tokens;
  uint top_k;
  uint features;
  uint reserved;
};

inline float mlp_decode_e8m0(uchar code) {
  const uint bits = code == 0 ? 0x00400000u : uint(code) << 23;
  return as_type<float>(bits);
}

inline uchar mlp_encode_e8m0_round_up(float value) {
  if (!(value > 0.0f)) {
    return 0;
  }
  const int exponent = clamp(int(ceil(log2(value))), -127, 127);
  return uchar(exponent + 127);
}

inline uchar mlp_encode_e2m1(float value) {
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
inline void swiglu(device const bfloat *gate, device const bfloat *up,
                   device bfloat *output_a16, device uchar *output_codes,
                   device uchar *output_scales, constant SwiGluParams &params,
                   uint lane, uint row_group) {
  if (params.groups == 0)
    return;
  const uint row = row_group / params.groups;
  if (row >= params.rows)
    return;
  const uint group = row_group - row * params.groups;
  const uint column = group * 32 + lane;
  const bool valid_col = column < params.features;
  const ulong index = ulong(row) * params.features + column;
  bfloat represented = bfloat(0.0f);
  if (valid_col) {
    const float gate_value = float(gate[index]);
    represented =
        bfloat((gate_value / (1.0f + exp(-gate_value))) * float(up[index]));
    if constexpr (ProduceA16) {
      output_a16[index] = represented;
    }
  }
  if constexpr (ProduceA4) {
    const float value = valid_col ? float(represented) : 0.0f;
    const float maximum = simd_max(abs(value));
    const uchar scale_code = mlp_encode_e8m0_round_up(maximum / 6.0f);
    const float scale = mlp_decode_e8m0(scale_code);
    const uchar code = mlp_encode_e2m1(value / scale);
    if (lane == 0 && group * 32 < params.features) {
      output_scales[ulong(row) * params.groups + group] = scale_code;
    }
    const uint high = simd_shuffle_down(uint(code), 1);
    if ((lane & 1u) == 0 && valid_col) {
      output_codes[ulong(row) * (params.features / 2) + ulong(group) * 16 +
                   lane / 2] = code | uchar(high << 4);
    }
  }
}

kernel void omlx_m5_swiglu_a16(device const bfloat *gate [[buffer(0)]],
                               device const bfloat *up [[buffer(1)]],
                               device bfloat *output_a16 [[buffer(2)]],
                               constant SwiGluParams &params [[buffer(3)]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint3 tg_position
                               [[threadgroup_position_in_grid]]) {
  swiglu<true, false>(gate, up, output_a16, nullptr, nullptr, params, lane,
                      tg_position.y);
}

kernel void omlx_m5_swiglu_a4(device const bfloat *gate [[buffer(0)]],
                              device const bfloat *up [[buffer(1)]],
                              device uchar *output_codes [[buffer(2)]],
                              device uchar *output_scales [[buffer(3)]],
                              constant SwiGluParams &params [[buffer(4)]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint3 tg_position
                              [[threadgroup_position_in_grid]]) {
  swiglu<false, true>(gate, up, nullptr, output_codes, output_scales, params,
                      lane, tg_position.y);
}

kernel void omlx_m5_swiglu_a16_a4(device const bfloat *gate [[buffer(0)]],
                                  device const bfloat *up [[buffer(1)]],
                                  device bfloat *output_a16 [[buffer(2)]],
                                  device uchar *output_codes [[buffer(3)]],
                                  device uchar *output_scales [[buffer(4)]],
                                  constant SwiGluParams &params [[buffer(5)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg_position
                                  [[threadgroup_position_in_grid]]) {
  swiglu<true, true>(gate, up, output_a16, output_codes, output_scales, params,
                     lane, tg_position.y);
}

kernel void omlx_m5_moe_topk_reduce(device const bfloat *assignment_output
                                    [[buffer(0)]],
                                    device bfloat *token_output [[buffer(1)]],
                                    constant MoeReduceParams &params
                                    [[buffer(2)]],
                                    uint gid [[thread_position_in_grid]]) {
  const ulong output_values = ulong(params.tokens) * params.features;
  if (ulong(gid) >= output_values) {
    return;
  }
  const uint token = gid / params.features;
  const uint feature = gid - token * params.features;
  float sum = 0.0f;
  for (uint slot = 0; slot < params.top_k; ++slot) {
    sum += float(
        assignment_output[(ulong(token) * params.top_k + slot) *
                              params.features +
                          feature]);
  }
  token_output[gid] = bfloat(sum);
}

kernel void
omlx_m5_moe_shared_combine(device const bfloat *routed [[buffer(0)]],
                           device const bfloat *shared [[buffer(1)]],
                           device const bfloat *shared_gate [[buffer(2)]],
                           device bfloat *output [[buffer(3)]],
                           constant MoeReduceParams &params [[buffer(4)]],
                           uint gid [[thread_position_in_grid]]) {
  const ulong output_values = ulong(params.tokens) * params.features;
  if (ulong(gid) >= output_values) {
    return;
  }
  const uint token = gid / params.features;
  const float gate = float(shared_gate[token]);
  output[gid] = bfloat(float(routed[gid]) +
                       (1.0f / (1.0f + exp(-gate))) * float(shared[gid]));
}

// Fused top-k assignment reduction + sigmoid-gated shared-expert combine:
// Preserves exact intermediate BF16 rounding (0-ULP parity with separate
// omlx_m5_moe_topk_reduce + omlx_m5_moe_shared_combine) while eliminating
// routed scratch write/read and one dispatch + barrier per MoE layer.
kernel void omlx_m5_moe_topk_shared_combine(
    device const bfloat *assignment_output [[buffer(0)]],
    device const bfloat *shared [[buffer(1)]],
    device const bfloat *shared_gate [[buffer(2)]],
    device bfloat *output [[buffer(3)]],
    constant MoeReduceParams &params [[buffer(4)]],
    uint gid [[thread_position_in_grid]]) {
  const ulong output_values = ulong(params.tokens) * params.features;
  if (ulong(gid) >= output_values) {
    return;
  }
  const uint token = gid / params.features;
  const uint feature = gid - token * params.features;
  float sum = 0.0f;
  for (uint slot = 0; slot < params.top_k; ++slot) {
    sum += float(
        assignment_output[(ulong(token) * params.top_k + slot) *
                              params.features +
                          feature]);
  }
  const bfloat routed_bf16 = bfloat(sum);
  const float gate = float(shared_gate[token]);
  output[gid] = bfloat(float(routed_bf16) +
                       (1.0f / (1.0f + exp(-gate))) * float(shared[gid]));
}
