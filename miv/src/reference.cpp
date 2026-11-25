#include "omlx_m5/reference.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace omlx::m5 {

namespace {

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

float sigmoid(float value) {
  if (value >= 0.0F) {
    const auto term = std::exp(-value);
    return 1.0F / (1.0F + term);
  }
  const auto term = std::exp(value);
  return term / (1.0F + term);
}

std::size_t checked_elements(std::size_t left, std::size_t right,
                             const char *message) {
  if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
    throw std::overflow_error(message);
  }
  return left * right;
}

std::vector<float> attention_current(std::span<const float> query,
                                     std::size_t query_heads,
                                     const KvCache &cache) {
  require(query_heads > 0 && query.size() == query_heads * cache.head_dim(),
          "attention query geometry mismatch");
  require(cache.tokens() > 0, "attention cache is empty");
  require(query_heads % cache.kv_heads() == 0,
          "query heads must be divisible by KV heads");

  const auto head_dim = cache.head_dim();
  const auto group_size = query_heads / cache.kv_heads();
  const auto scale = 1.0F / std::sqrt(static_cast<float>(head_dim));
  std::vector<float> output(query.size(), 0.0F);
  std::vector<float> scores(cache.tokens());
  for (std::size_t query_head = 0; query_head < query_heads; ++query_head) {
    const auto kv_head = query_head / group_size;
    const auto query_row = query.subspan(query_head * head_dim, head_dim);
    auto max_score = -std::numeric_limits<float>::infinity();
    for (std::size_t token = 0; token < cache.tokens(); ++token) {
      const auto key_row = cache.key(token, kv_head);
      float score = 0.0F;
      for (std::size_t index = 0; index < head_dim; ++index) {
        score = std::fma(query_row[index], key_row[index], score);
      }
      scores[token] = score * scale;
      max_score = std::max(max_score, scores[token]);
    }
    float denominator = 0.0F;
    for (auto &score : scores) {
      score = std::exp(score - max_score);
      denominator += score;
    }
    const auto output_offset = query_head * head_dim;
    for (std::size_t token = 0; token < cache.tokens(); ++token) {
      const auto probability = scores[token] / denominator;
      const auto value_row = cache.value(token, kv_head);
      for (std::size_t index = 0; index < head_dim; ++index) {
        output[output_offset + index] =
            std::fma(probability, value_row[index],
                     output[output_offset + index]);
      }
    }
  }
  return output;
}

void validate_gdn_geometry(const GdnReferenceGeometry &geometry) {
  require(geometry.key_heads > 0 && geometry.value_heads > 0 &&
              geometry.key_dim > 0 && geometry.value_dim > 0,
          "GDN geometry dimensions must be positive");
  require(geometry.value_heads % geometry.key_heads == 0,
          "GDN value heads must be divisible by key heads");
}

std::size_t gdn_state_size(const GdnReferenceGeometry &geometry) {
  return checked_elements(
      checked_elements(geometry.value_heads, geometry.value_dim,
                       "GDN state size overflow"),
      geometry.key_dim, "GDN state size overflow");
}

} // namespace

std::vector<float> quantized_qmv(const QuantizedModule &module,
                                 std::span<const float> input) {
  require(input.size() == module.in_features,
          "qmv input width does not match quantized module");
  std::vector<float> output(module.out_features);
  for (std::uint64_t row = 0; row < module.out_features; ++row) {
    const auto weights =
        dequantize_row(module, row, 0, module.in_features);
    float sum = 0.0F;
    for (std::size_t column = 0; column < input.size(); ++column) {
      sum = std::fma(input[column], weights[column], sum);
    }
    output[row] = sum;
  }
  return output;
}

std::vector<float> quantized_qmm(const QuantizedModule &module,
                                 std::span<const float> input,
                                 std::size_t rows) {
  require(rows > 0, "qmm row count must be positive");
  const auto elements = checked_elements(
      rows, static_cast<std::size_t>(module.in_features),
      "qmm input size overflow");
  require(input.size() == elements,
          "qmm input shape does not match quantized module");
  std::vector<float> output(checked_elements(
      rows, static_cast<std::size_t>(module.out_features),
      "qmm output size overflow"));
  for (std::uint64_t out = 0; out < module.out_features; ++out) {
    const auto weights =
        dequantize_row(module, out, 0, module.in_features);
    for (std::size_t row = 0; row < rows; ++row) {
      float sum = 0.0F;
      const auto input_offset = row * module.in_features;
      for (std::size_t column = 0; column < input.size() / rows; ++column) {
        sum = std::fma(input[input_offset + column], weights[column], sum);
      }
      output[row * module.out_features + out] = sum;
    }
  }
  return output;
}

std::vector<float> rms_norm(std::span<const float> input, std::size_t rows,
                            std::size_t width,
                            std::span<const float> weight, float epsilon) {
  require(rows > 0 && width > 0 && input.size() == rows * width,
          "RMSNorm input geometry mismatch");
  require(weight.empty() || weight.size() == width,
          "RMSNorm weight width mismatch");
  require(std::isfinite(epsilon) && epsilon > 0.0F,
          "RMSNorm epsilon must be positive and finite");
  std::vector<float> output(input.size());
  for (std::size_t row = 0; row < rows; ++row) {
    float square_sum = 0.0F;
    for (std::size_t column = 0; column < width; ++column) {
      const auto value = input[row * width + column];
      square_sum = std::fma(value, value, square_sum);
    }
    const auto inverse =
        1.0F / std::sqrt(square_sum / static_cast<float>(width) + epsilon);
    for (std::size_t column = 0; column < width; ++column) {
      const auto scale = weight.empty() ? 1.0F : weight[column];
      output[row * width + column] =
          input[row * width + column] * inverse * scale;
    }
  }
  return output;
}

void apply_rope(std::span<float> values, std::size_t positions,
                std::size_t heads, std::size_t head_dim,
                std::size_t rotary_dim, std::uint64_t start_position,
                double theta) {
  require(positions > 0 && heads > 0 && head_dim > 0 &&
              values.size() == positions * heads * head_dim,
          "RoPE input geometry mismatch");
  require(rotary_dim > 0 && rotary_dim <= head_dim && rotary_dim % 2 == 0,
          "RoPE dimension must be positive, even and within head_dim");
  require(std::isfinite(theta) && theta > 0.0,
          "RoPE theta must be positive and finite");
  for (std::size_t position = 0; position < positions; ++position) {
    const auto absolute = static_cast<double>(start_position + position);
    for (std::size_t head = 0; head < heads; ++head) {
      auto *row = values.data() + (position * heads + head) * head_dim;
      for (std::size_t pair = 0; pair < rotary_dim / 2; ++pair) {
        const auto exponent =
            static_cast<double>(2 * pair) / static_cast<double>(rotary_dim);
        const auto angle = absolute / std::pow(theta, exponent);
        const auto cosine = static_cast<float>(std::cos(angle));
        const auto sine = static_cast<float>(std::sin(angle));
        const auto first = row[2 * pair];
        const auto second = row[2 * pair + 1];
        row[2 * pair] = first * cosine - second * sine;
        row[2 * pair + 1] = first * sine + second * cosine;
      }
    }
  }
}

std::vector<float> silu_multiply(std::span<const float> gate,
                                 std::span<const float> values) {
  require(gate.size() == values.size(),
          "SiLU multiply operands must have equal length");
  std::vector<float> output(gate.size());
  for (std::size_t index = 0; index < gate.size(); ++index) {
    output[index] = gate[index] * sigmoid(gate[index]) * values[index];
  }
  return output;
}

CounterRng::CounterRng(std::uint64_t seed) : seed_(seed) {}

std::uint64_t CounterRng::bits(std::uint64_t stream,
                               std::uint64_t counter) const {
  auto value = seed_ ^ (stream + 0x9E3779B97F4A7C15ULL) ^
               (counter * 0xD2B74407B1CE6E93ULL);
  value += 0x9E3779B97F4A7C15ULL;
  value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9ULL;
  value = (value ^ (value >> 27)) * 0x94D049BB133111EBULL;
  return value ^ (value >> 31);
}

double CounterRng::uniform(std::uint64_t stream, std::uint64_t counter) const {
  return static_cast<double>(bits(stream, counter) >> 11) *
         0x1.0p-53;
}

std::int32_t sample_logits(std::span<const float> logits,
                           const SamplingOptions &options,
                           const CounterRng &rng, std::uint64_t stream,
                           std::uint64_t absolute_position) {
  require(!logits.empty(), "sampling logits must not be empty");
  require(std::isfinite(options.temperature) && options.temperature >= 0.0F,
          "sampling temperature must be finite and non-negative");
  require(std::isfinite(options.top_p) && options.top_p > 0.0F &&
              options.top_p <= 1.0F,
          "top_p must be in (0, 1]");
  if (options.temperature == 0.0F) {
    return static_cast<std::int32_t>(std::distance(
        logits.begin(), std::max_element(logits.begin(), logits.end())));
  }

  std::vector<std::size_t> indices(logits.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::ranges::sort(indices, [&](std::size_t left, std::size_t right) {
    if (logits[left] != logits[right]) {
      return logits[left] > logits[right];
    }
    return left < right;
  });
  if (options.top_k != 0 && indices.size() > options.top_k) {
    indices.resize(options.top_k);
  }
  const double maximum = static_cast<double>(logits[indices.front()]);
  if (!std::isfinite(maximum)) {
    return static_cast<std::int32_t>(indices.front());
  }
  const double temperature = static_cast<double>(options.temperature);
  std::vector<double> probabilities(indices.size());
  double total = 0.0;
  for (std::size_t index = 0; index < indices.size(); ++index) {
    probabilities[index] = std::exp(
        (static_cast<double>(logits[indices[index]]) - maximum) / temperature);
    total += probabilities[index];
  }
  if (!std::isfinite(total) || total <= 0.0) {
    return static_cast<std::int32_t>(indices.front());
  }
  double cumulative = 0.0;
  std::size_t retained = probabilities.size();
  for (std::size_t index = 0; index < probabilities.size(); ++index) {
    cumulative += probabilities[index] / total;
    if (cumulative >= options.top_p) {
      retained = index + 1;
      break;
    }
  }
  const auto retained_total =
      std::accumulate(probabilities.begin(), probabilities.begin() +
                                                   static_cast<std::ptrdiff_t>(retained),
                      0.0);
  const auto draw = rng.uniform(stream, absolute_position) * retained_total;
  cumulative = 0.0;
  for (std::size_t index = 0; index < retained; ++index) {
    cumulative += probabilities[index];
    if (draw < cumulative || index + 1 == retained) {
      return static_cast<std::int32_t>(indices[index]);
    }
  }
  throw std::logic_error("sampling failed to select a token");
}

KvCache::KvCache(std::size_t kv_heads, std::size_t head_dim,
                 std::size_t capacity_tokens)
    : kv_heads_(kv_heads), head_dim_(head_dim),
      capacity_tokens_(capacity_tokens),
      keys_(checked_elements(checked_elements(capacity_tokens, kv_heads,
                                              "KV cache size overflow"),
                             head_dim, "KV cache size overflow")),
      values_(keys_.size()) {
  require(kv_heads_ > 0 && head_dim_ > 0 && capacity_tokens_ > 0,
          "KV cache geometry must be positive");
}

void KvCache::append(std::span<const float> keys,
                     std::span<const float> values) {
  require(keys.size() == kv_heads_ * head_dim_ && values.size() == keys.size(),
          "KV append geometry mismatch");
  if (tokens_ == capacity_tokens_) {
    throw std::length_error("KV cache capacity exceeded");
  }
  const auto offset = tokens_ * kv_heads_ * head_dim_;
  std::ranges::copy(keys, keys_.begin() + static_cast<std::ptrdiff_t>(offset));
  std::ranges::copy(values,
                    values_.begin() + static_cast<std::ptrdiff_t>(offset));
  ++tokens_;
}

void KvCache::truncate(std::size_t tokens) {
  if (tokens > tokens_) {
    throw std::out_of_range("KV truncate length exceeds current tokens");
  }
  tokens_ = tokens;
}

void KvCache::clear() { tokens_ = 0; }
std::size_t KvCache::kv_heads() const { return kv_heads_; }
std::size_t KvCache::head_dim() const { return head_dim_; }
std::size_t KvCache::capacity_tokens() const { return capacity_tokens_; }
std::size_t KvCache::tokens() const { return tokens_; }

std::span<const float> KvCache::key(std::size_t token,
                                    std::size_t head) const {
  if (token >= tokens_ || head >= kv_heads_) {
    throw std::out_of_range("KV key index is out of bounds");
  }
  const auto offset = (token * kv_heads_ + head) * head_dim_;
  return std::span<const float>(keys_).subspan(offset, head_dim_);
}

std::span<const float> KvCache::value(std::size_t token,
                                      std::size_t head) const {
  if (token >= tokens_ || head >= kv_heads_) {
    throw std::out_of_range("KV value index is out of bounds");
  }
  const auto offset = (token * kv_heads_ + head) * head_dim_;
  return std::span<const float>(values_).subspan(offset, head_dim_);
}

std::vector<float> attention_decode(std::span<const float> query,
                                    std::size_t query_heads,
                                    std::span<const float> key,
                                    std::span<const float> value,
                                    KvCache &cache) {
  cache.append(key, value);
  return attention_current(query, query_heads, cache);
}

std::vector<float> attention_prefill(std::span<const float> queries,
                                     std::span<const float> keys,
                                     std::span<const float> values,
                                     std::size_t positions,
                                     std::size_t query_heads, KvCache &cache,
                                     std::size_t chunk_size) {
  require(positions > 0 && chunk_size > 0,
          "attention prefill positions and chunk size must be positive");
  const auto query_width = query_heads * cache.head_dim();
  const auto kv_width = cache.kv_heads() * cache.head_dim();
  require(queries.size() == positions * query_width &&
              keys.size() == positions * kv_width &&
              values.size() == positions * kv_width,
          "attention prefill geometry mismatch");
  std::vector<float> output(queries.size());
  for (std::size_t chunk = 0; chunk < positions; chunk += chunk_size) {
    const auto end = std::min(positions, chunk + chunk_size);
    for (std::size_t position = chunk; position < end; ++position) {
      const auto current = attention_decode(
          queries.subspan(position * query_width, query_width), query_heads,
          keys.subspan(position * kv_width, kv_width),
          values.subspan(position * kv_width, kv_width), cache);
      std::ranges::copy(
          current,
          output.begin() + static_cast<std::ptrdiff_t>(position * query_width));
    }
  }
  return output;
}

GdnReferenceState::GdnReferenceState(GdnReferenceGeometry geometry)
    : geometry_(geometry), values_(gdn_state_size(geometry), 0.0F) {
  validate_gdn_geometry(geometry_);
}

GdnReferenceState::GdnReferenceState(GdnReferenceGeometry geometry,
                                     std::vector<float> values)
    : geometry_(geometry), values_(std::move(values)) {
  validate_gdn_geometry(geometry_);
  require(values_.size() == gdn_state_size(geometry_),
          "GDN state storage geometry mismatch");
}

const GdnReferenceGeometry &GdnReferenceState::geometry() const {
  return geometry_;
}
const std::vector<float> &GdnReferenceState::values() const { return values_; }
std::vector<float> &GdnReferenceState::mutable_values() { return values_; }
void GdnReferenceState::clear() { std::ranges::fill(values_, 0.0F); }

std::vector<float> gdn_step(std::span<const float> query,
                            std::span<const float> key,
                            std::span<const float> value,
                            std::span<const float> decay,
                            std::span<const float> beta,
                            GdnReferenceState &state) {
  const auto &geometry = state.geometry();
  require(query.size() == geometry.key_heads * geometry.key_dim &&
              key.size() == query.size() &&
              value.size() == geometry.value_heads * geometry.value_dim &&
              decay.size() == geometry.value_heads &&
              beta.size() == geometry.value_heads,
          "GDN step geometry mismatch");
  const auto repeat = geometry.value_heads / geometry.key_heads;
  std::vector<float> output(value.size(), 0.0F);
  auto &storage = state.mutable_values();
  for (std::size_t value_head = 0; value_head < geometry.value_heads;
       ++value_head) {
    const auto key_head = value_head / repeat;
    const auto key_row = key.subspan(key_head * geometry.key_dim,
                                     geometry.key_dim);
    const auto query_row = query.subspan(key_head * geometry.key_dim,
                                         geometry.key_dim);
    for (std::size_t value_index = 0; value_index < geometry.value_dim;
         ++value_index) {
      auto *state_row =
          storage.data() +
          (value_head * geometry.value_dim + value_index) * geometry.key_dim;
      float memory = 0.0F;
      for (std::size_t key_index = 0; key_index < geometry.key_dim;
           ++key_index) {
        state_row[key_index] *= decay[value_head];
        memory = std::fma(state_row[key_index], key_row[key_index], memory);
      }
      const auto delta =
          (value[value_head * geometry.value_dim + value_index] - memory) *
          beta[value_head];
      float projected = 0.0F;
      for (std::size_t key_index = 0; key_index < geometry.key_dim;
           ++key_index) {
        state_row[key_index] =
            std::fma(key_row[key_index], delta, state_row[key_index]);
        projected =
            std::fma(state_row[key_index], query_row[key_index], projected);
      }
      output[value_head * geometry.value_dim + value_index] = projected;
    }
  }
  return output;
}

std::vector<float> gdn_prefill(std::span<const float> queries,
                               std::span<const float> keys,
                               std::span<const float> values,
                               std::span<const float> decays,
                               std::span<const float> betas,
                               std::size_t positions,
                               GdnReferenceState &state,
                               std::size_t chunk_size) {
  require(positions > 0 && chunk_size > 0,
          "GDN prefill positions and chunk size must be positive");
  const auto &geometry = state.geometry();
  const auto key_width = geometry.key_heads * geometry.key_dim;
  const auto value_width = geometry.value_heads * geometry.value_dim;
  require(queries.size() == positions * key_width &&
              keys.size() == positions * key_width &&
              values.size() == positions * value_width &&
              decays.size() == positions * geometry.value_heads &&
              betas.size() == decays.size(),
          "GDN prefill geometry mismatch");
  std::vector<float> output(values.size());
  for (std::size_t chunk = 0; chunk < positions; chunk += chunk_size) {
    const auto end = std::min(positions, chunk + chunk_size);
    for (std::size_t position = chunk; position < end; ++position) {
      const auto current = gdn_step(
          queries.subspan(position * key_width, key_width),
          keys.subspan(position * key_width, key_width),
          values.subspan(position * value_width, value_width),
          decays.subspan(position * geometry.value_heads,
                         geometry.value_heads),
          betas.subspan(position * geometry.value_heads,
                        geometry.value_heads),
          state);
      std::ranges::copy(current,
                        output.begin() + static_cast<std::ptrdiff_t>(
                                             position * value_width));
    }
  }
  return output;
}

std::vector<float> causal_depthwise_conv_silu(
    std::span<const float> input, std::size_t positions, std::size_t width,
    std::span<const float> kernel, std::size_t kernel_size,
    std::vector<float> &history) {
  require(positions > 0 && width > 0 && input.size() == positions * width,
          "depthwise convolution input geometry mismatch");
  require(kernel_size > 0 && kernel.size() == kernel_size * width,
          "depthwise convolution kernel geometry mismatch");
  const auto history_positions = kernel_size - 1;
  require(history.size() == history_positions * width,
          "depthwise convolution history geometry mismatch");

  std::vector<float> window;
  window.reserve((history_positions + positions) * width);
  window.insert(window.end(), history.begin(), history.end());
  window.insert(window.end(), input.begin(), input.end());
  std::vector<float> output(input.size());
  for (std::size_t position = 0; position < positions; ++position) {
    for (std::size_t column = 0; column < width; ++column) {
      float sum = 0.0F;
      for (std::size_t tap = 0; tap < kernel_size; ++tap) {
        sum = std::fma(window[(position + tap) * width + column],
                       kernel[tap * width + column], sum);
      }
      output[position * width + column] = sum * sigmoid(sum);
    }
  }
  if (history_positions > 0) {
    const auto begin = window.end() -
                       static_cast<std::ptrdiff_t>(history_positions * width);
    history.assign(begin, window.end());
  }
  return output;
}

} // namespace omlx::m5
