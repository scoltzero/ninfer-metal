#include <metal_stdlib>

using namespace metal;

kernel void omlx_m5_gdn_replay(device float *state [[buffer(0)]],
                               device const float *keys [[buffer(1)]],
                               device const float *values [[buffer(2)]],
                               device const float *log_decay [[buffer(3)]],
                               device const float *beta [[buffer(4)]],
                               constant uint &key_dim [[buffer(5)]],
                               constant uint &value_dim [[buffer(6)]],
                               constant uint &record_start [[buffer(7)]],
                               constant uint &record_count [[buffer(8)]],
                               constant float &epsilon [[buffer(9)]],
                               uint value_index [[thread_position_in_grid]]) {
  if (value_index >= value_dim) {
    return;
  }
  device float *row = state + value_index * key_dim;
  for (uint local_step = 0; local_step < record_count; ++local_step) {
    const uint step = record_start + local_step;
    device const float *key = keys + step * key_dim;
    float norm_squared = 0.0f;
    for (uint index = 0; index < key_dim; ++index) {
      norm_squared = fma(key[index], key[index], norm_squared);
    }
    const float inverse_norm = rsqrt(norm_squared + epsilon);
    const float alpha = exp(log_decay[step]);
    float correction = 0.0f;
    for (uint index = 0; index < key_dim; ++index) {
      row[index] *= alpha;
      correction = fma(row[index], key[index] * inverse_norm, correction);
    }
    const float update =
        beta[step] * (values[step * value_dim + value_index] - correction);
    for (uint index = 0; index < key_dim; ++index) {
      row[index] = fma(update, key[index] * inverse_norm, row[index]);
    }
  }
}
