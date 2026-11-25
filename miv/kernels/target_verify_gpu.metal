#include <metal_stdlib>
using namespace metal;

struct VerifyParams {
  uint vocab, topk, blocks, groups, width, proposal_k, position;
  float temperature, top_p;
  float accept_uniform[7], correction_uniform[7], bonus_uniform;
};
inline bool better(float score, uint id, float other, uint other_id) {
  return score > other || (score == other && id < other_id);
}
kernel void
omlx_m5_target_topk_chunks(device const bfloat *logits [[buffer(0)]],
                           device uint *partial_ids [[buffer(1)]],
                           device float *partial_scores [[buffer(2)]],
                           constant VerifyParams &p [[buffer(3)]],
                           uint2 index [[thread_position_in_grid]]) {
  uint block = index.x, row = index.y;
  if (block >= p.blocks || row >= p.width)
    return;
  if (p.topk == 1u) {
    float best_val = -INFINITY;
    uint best_id = 0xffffffffu;
    bool has_nan = false;
    const ulong row_base = ulong(row) * p.vocab;
    const uint end_tok = min((block + 1u) * 256u, p.vocab);
    for (uint token = block * 256u; token < end_tok; ++token) {
      const float value = float(logits[row_base + token]);
      if (!isfinite(value)) {
        if (isnan(value) || value > 0.0f) {
          has_nan = true;
          break;
        }
        continue;
      }
      if (better(value, token, best_val, best_id)) {
        best_val = value;
        best_id = token;
      }
    }
    const uint base = (row * p.blocks + block) * 64u;
    partial_ids[base] = has_nan ? 0xffffffffu : best_id;
    partial_scores[base] = has_nan ? NAN : best_val;
    return;
  }
  float values[64];
  uint ids[64];
  for (uint i = 0; i < p.topk; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  bool has_nan = false;
  for (uint token = block * 256; token < min((block + 1) * 256, p.vocab);
       ++token) {
    float value = float(logits[ulong(row) * p.vocab + token]);
    if (!isfinite(value)) {
      if (isnan(value) || value > 0.0f) {
        has_nan = true;
        break;
      }
      continue;
    }
    if (!better(value, token, values[p.topk - 1], ids[p.topk - 1]))
      continue;
    uint place = p.topk - 1;
    while (place && better(value, token, values[place - 1], ids[place - 1])) {
      values[place] = values[place - 1];
      ids[place] = ids[place - 1];
      --place;
    }
    values[place] = value;
    ids[place] = token;
  }
  uint base = (row * p.blocks + block) * 64;
  if (has_nan) {
    for (uint i = 0; i < p.topk; ++i) {
      partial_ids[base + i] = 0xffffffffu;
      partial_scores[base + i] = NAN;
    }
  } else {
    for (uint i = 0; i < p.topk; ++i) {
      partial_ids[base + i] = ids[i];
      partial_scores[base + i] = values[i];
    }
  }
}
kernel void
omlx_m5_target_topk_reduce32(device const uint *partial_ids [[buffer(0)]],
                             device const float *partial_scores [[buffer(1)]],
                             device uint *group_ids [[buffer(2)]],
                             device float *group_scores [[buffer(3)]],
                             constant VerifyParams &p [[buffer(4)]],
                             uint2 index [[thread_position_in_grid]]) {
  uint group = index.x, row = index.y;
  if (group >= p.groups || row >= p.width)
    return;
  if (p.topk == 1u) {
    float best_val = -INFINITY;
    uint best_id = 0xffffffffu;
    bool has_nan = false;
    const uint end_blk = min((group + 1u) * 32u, p.blocks);
    for (uint block = group * 32u; block < end_blk; ++block) {
      const uint base = (row * p.blocks + block) * 64u;
      const uint token = partial_ids[base];
      const float value = partial_scores[base];
      if (isnan(value)) {
        has_nan = true;
        break;
      }
      if (token != 0xffffffffu && better(value, token, best_val, best_id)) {
        best_val = value;
        best_id = token;
      }
    }
    const uint base = (row * p.groups + group) * 64u;
    group_ids[base] = has_nan ? 0xffffffffu : best_id;
    group_scores[base] = has_nan ? NAN : best_val;
    return;
  }
  float values[64];
  uint ids[64];
  for (uint i = 0; i < p.topk; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  bool has_nan = false;
  for (uint block = group * 32; block < min((group + 1) * 32, p.blocks);
       ++block) {
    uint base = (row * p.blocks + block) * 64;
    for (uint candidate = 0; candidate < p.topk; ++candidate) {
      uint token = partial_ids[base + candidate];
      float value = partial_scores[base + candidate];
      if (isnan(value)) {
        has_nan = true;
        break;
      }
      if (token == 0xffffffffu ||
          !better(value, token, values[p.topk - 1], ids[p.topk - 1]))
        continue;
      uint place = p.topk - 1;
      while (place && better(value, token, values[place - 1], ids[place - 1])) {
        values[place] = values[place - 1];
        ids[place] = ids[place - 1];
        --place;
      }
      values[place] = value;
      ids[place] = token;
    }
    if (has_nan)
      break;
  }
  uint base = (row * p.groups + group) * 64;
  if (has_nan) {
    for (uint i = 0; i < p.topk; ++i) {
      group_ids[base + i] = 0xffffffffu;
      group_scores[base + i] = NAN;
    }
  } else {
    for (uint i = 0; i < p.topk; ++i) {
      group_ids[base + i] = ids[i];
      group_scores[base + i] = values[i];
    }
  }
}
kernel void omlx_m5_target_topk_merge(device const uint *group_ids
                                      [[buffer(0)]],
                                      device const float *group_scores
                                      [[buffer(1)]],
                                      device uint *ids_out [[buffer(2)]],
                                      device float *scores_out [[buffer(3)]],
                                      constant VerifyParams &p [[buffer(4)]],
                                      uint row [[thread_position_in_grid]]) {
  if (row >= p.width)
    return;
  if (p.topk == 1u) {
    float best_val = -INFINITY;
    uint best_id = 0xffffffffu;
    bool has_nan = false;
    for (uint group = 0u; group < p.groups; ++group) {
      const uint base = (row * p.groups + group) * 64u;
      const uint token = group_ids[base];
      const float value = group_scores[base];
      if (isnan(value)) {
        has_nan = true;
        break;
      }
      if (token != 0xffffffffu && better(value, token, best_val, best_id)) {
        best_val = value;
        best_id = token;
      }
    }
    const uint base = row * 64u;
    ids_out[base] = has_nan ? 0xffffffffu : best_id;
    scores_out[base] = has_nan ? NAN : best_val;
    return;
  }
  float values[64];
  uint ids[64];
  for (uint i = 0; i < p.topk; ++i) {
    values[i] = -INFINITY;
    ids[i] = 0xffffffffu;
  }
  bool has_nan = false;
  for (uint group = 0; group < p.groups; ++group) {
    uint base = (row * p.groups + group) * 64;
    for (uint candidate = 0; candidate < p.topk; ++candidate) {
      uint token = group_ids[base + candidate];
      float value = group_scores[base + candidate];
      if (isnan(value)) {
        has_nan = true;
        break;
      }
      if (token == 0xffffffffu ||
          !better(value, token, values[p.topk - 1], ids[p.topk - 1]))
        continue;
      uint place = p.topk - 1;
      while (place && better(value, token, values[place - 1], ids[place - 1])) {
        values[place] = values[place - 1];
        ids[place] = ids[place - 1];
        --place;
      }
      values[place] = value;
      ids[place] = token;
    }
    if (has_nan)
      break;
  }
  uint base = row * 64;
  if (has_nan) {
    for (uint i = 0; i < p.topk; ++i) {
      ids_out[base + i] = 0xffffffffu;
      scores_out[base + i] = NAN;
    }
  } else {
    for (uint i = 0; i < p.topk; ++i) {
      ids_out[base + i] = ids[i];
      scores_out[base + i] = values[i];
    }
  }
}

kernel void
omlx_m5_target_verify_select(device const uint *ids [[buffer(0)]],
                             device const float *scores [[buffer(1)]],
                             device const uint *proposal_tokens [[buffer(2)]],
                             device const uint *proposal_ids [[buffer(3)]],
                             device const float *proposal_q [[buffer(4)]],
                             device uint *licensed [[buffer(5)]],
                             device uint *meta [[buffer(6)]],
                             constant VerifyParams &p [[buffer(7)]],
                             uint tid [[thread_position_in_grid]]) {
  if (tid)
    return;
  uint accepted = 0;
  meta[3] = 0; // numerical boundary: reconstruct from compact top-k on host
  for (uint row = 0; row < p.width; ++row) {
    float distribution[64];
    uint retained = p.topk;
    if (p.temperature <= 0.0f) {
      if (ids[row * 64] >= p.vocab || !isfinite(scores[row * 64]))
        meta[3] = 1;
      distribution[0] = 1.0f;
      retained = 1;
    } else {
      float raw[64], total = 0.0f;
      float maximum = scores[row * 64];
      if (ids[row * 64] >= p.vocab || !isfinite(maximum))
        meta[3] = 1;
      for (uint i = 0; i < p.topk; ++i) {
        if (ids[row * 64 + i] >= p.vocab || !isfinite(scores[row * 64 + i]))
          meta[3] = 1;
        raw[i] = exp((scores[row * 64 + i] - maximum) / p.temperature);
        total += raw[i];
      }
      if (!(total > 0.0f) || !isfinite(total))
        meta[3] = 1;
      float cumulative = 0.0f, retained_mass = 0.0f;
      for (uint i = 0; i < p.topk; ++i) {
        distribution[i] = raw[i] / total;
        cumulative += distribution[i];
        retained_mass += distribution[i];
        if (!isfinite(distribution[i]) ||
            (p.top_p < 1.0f && fabs(cumulative - p.top_p) < 1e-4f))
          meta[3] = 1;
        if (p.top_p < 1.0f && cumulative >= p.top_p) {
          retained = i + 1;
          break;
        }
      }
      for (uint i = 0; i < retained; ++i)
        distribution[i] /= retained_mass;
    }
    if (row + 1 < p.width) {
      uint proposed = proposal_tokens[row];
      float p_token = 0.0f, q_token = 0.0f;
      for (uint i = 0; i < retained; ++i)
        if (ids[row * 64 + i] == proposed)
          p_token = distribution[i];
      for (uint i = 0; i < p.proposal_k; ++i)
        if (proposal_ids[row * 32 + i] == proposed) {
          q_token = proposal_q[row * 32 + i];
          break;
        }
      bool accept;
      if (p.temperature <= 0.0f)
        accept = proposed == ids[row * 64];
      else {
        float chance = q_token > 0.0f ? min(1.0f, p_token / q_token)
                                      : (p_token > 0.0f ? 1.0f : 0.0f);
        if (fabs(p.accept_uniform[row] - chance) < 1e-4f || !isfinite(chance))
          meta[3] = 1;
        accept = p.accept_uniform[row] < chance;
      }
      if (accept) {
        licensed[row] = proposed;
        ++accepted;
        continue;
      }
      if (p.temperature <= 0.0f) {
        licensed[row] = ids[row * 64];
      } else {
        float correction[64], sum = 0.0f;
        for (uint i = 0; i < retained; ++i) {
          float value = distribution[i];
          for (uint j = 0; j < p.proposal_k; ++j)
            if (proposal_ids[row * 32 + j] == ids[row * 64 + i])
              value = max(0.0f, value - proposal_q[row * 32 + j]);
          correction[i] = value;
          sum += value;
        }
        if (!(sum > 0.0f))
          meta[3] = 1;
        uint order[64];
        for (uint i = 0; i < retained; ++i) {
          uint place = i;
          while (place &&
                 ids[row * 64 + order[place - 1]] > ids[row * 64 + i]) {
            order[place] = order[place - 1];
            --place;
          }
          order[place] = i;
        }
        float cumulative = 0.0f;
        uint choice = order[retained - 1];
        for (uint i = 0; i < retained; ++i) {
          uint index = order[i];
          cumulative += sum > 0.0f ? (correction[index] / sum)
                                   : distribution[index];
          if (fabs(p.correction_uniform[row] - cumulative) < 1e-4f)
            meta[3] = 1;
          if (p.correction_uniform[row] < cumulative) {
            choice = index;
            break;
          }
        }
        licensed[row] = ids[row * 64 + choice];
      }
      meta[0] = row + 1;
      meta[1] = accepted;
      meta[2] = 1;
      return;
    }
    if (p.temperature <= 0.0f) {
      licensed[row] = ids[row * 64];
    } else {
      uint order[64];
      for (uint i = 0; i < retained; ++i) {
        uint place = i;
        while (place && ids[row * 64 + order[place - 1]] > ids[row * 64 + i]) {
          order[place] = order[place - 1];
          --place;
        }
        order[place] = i;
      }
      float cumulative = 0.0f;
      uint choice = order[retained - 1];
      for (uint i = 0; i < retained; ++i) {
        uint index = order[i];
        cumulative += distribution[index];
        if (fabs(p.bonus_uniform - cumulative) < 1e-4f)
          meta[3] = 1;
        if (p.bonus_uniform < cumulative) {
          choice = index;
          break;
        }
      }
      licensed[row] = ids[row * 64 + choice];
    }
    meta[0] = p.width;
    meta[1] = accepted;
    meta[2] = 0;
  }
}
