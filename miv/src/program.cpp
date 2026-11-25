#include "omlx_m5/program.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
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

float softplus(float value) {
  if (value > 20.0F) {
    return value;
  }
  if (value < -20.0F) {
    return std::exp(value);
  }
  return std::log1p(std::exp(value));
}

struct DenseMatrix {
  std::size_t output = 0;
  std::size_t input = 0;
  std::vector<float> values;

  [[nodiscard]] std::vector<float>
  apply(std::span<const float> vector) const {
    require(vector.size() == input, "dense input width mismatch");
    std::vector<float> result(output);
    for (std::size_t row = 0; row < output; ++row) {
      float sum = 0.0F;
      const auto offset = row * input;
      for (std::size_t column = 0; column < input; ++column) {
        sum = std::fma(values[offset + column], vector[column], sum);
      }
      result[row] = sum;
    }
    return result;
  }

  [[nodiscard]] std::uint64_t parameter_count() const {
    return static_cast<std::uint64_t>(values.size());
  }
};

DenseMatrix make_matrix(std::size_t output, std::size_t input,
                        const CounterRng &rng, std::uint64_t stream) {
  require(output > 0 && input > 0, "dense matrix dimensions must be positive");
  if (output > std::numeric_limits<std::size_t>::max() / input) {
    throw std::overflow_error("dense matrix size overflow");
  }
  const auto elements = output * input;
  constexpr std::size_t max_reference_parameters = 64ULL * 1024ULL * 1024ULL;
  if (elements > max_reference_parameters) {
    throw std::length_error(
        "deterministic reference weights are for small-model correctness only");
  }
  DenseMatrix matrix{.output = output,
                     .input = input,
                     .values = std::vector<float>(elements)};
  const auto scale = 0.35F / std::sqrt(static_cast<float>(input));
  for (std::size_t index = 0; index < elements; ++index) {
    const auto unit = static_cast<float>(rng.uniform(stream, index));
    matrix.values[index] = (unit * 2.0F - 1.0F) * scale;
  }
  return matrix;
}

std::vector<float> make_norm(std::size_t width, const CounterRng &rng,
                             std::uint64_t stream) {
  std::vector<float> result(width);
  for (std::size_t index = 0; index < width; ++index) {
    result[index] =
        0.95F + 0.1F * static_cast<float>(rng.uniform(stream, index));
  }
  return result;
}

void add_in_place(std::vector<float> &target,
                  std::span<const float> residual) {
  require(target.size() == residual.size(), "residual width mismatch");
  for (std::size_t index = 0; index < target.size(); ++index) {
    target[index] += residual[index];
  }
}

void validate_config(const ReferenceProgramConfig &config) {
  require(config.vocab_size > 1 && config.hidden_size > 0 &&
              config.intermediate_size > 0 && config.attention_heads > 0 &&
              config.kv_heads > 0 && config.head_dim > 0 &&
              config.max_context > 0 && !config.layer_types.empty(),
          "reference Program dimensions must be positive");
  require(config.attention_heads % config.kv_heads == 0,
          "attention heads must be divisible by KV heads");
  require(config.linear_key_heads > 0 && config.linear_value_heads > 0 &&
              config.linear_key_dim > 0 && config.linear_value_dim > 0 &&
              config.linear_value_heads % config.linear_key_heads == 0,
          "reference Program GDN geometry is invalid");
  require(config.linear_conv_kernel > 0,
          "reference Program convolution width must be positive");
  require(config.rotary_dim > 0 && config.rotary_dim <= config.head_dim &&
              config.rotary_dim % 2 == 0,
          "reference Program rotary dimension is invalid");
  require(config.rms_norm_eps > 0.0F && config.rope_theta > 0.0,
          "reference Program norm/RoPE constants are invalid");
}

struct AttentionWeights {
  DenseMatrix q_gate;
  DenseMatrix key;
  DenseMatrix value;
  DenseMatrix output;
  std::vector<float> query_norm;
  std::vector<float> key_norm;
};

struct GdnWeights {
  DenseMatrix qkv;
  DenseMatrix z;
  DenseMatrix a;
  DenseMatrix b;
  DenseMatrix output;
  std::vector<float> conv;
  std::vector<float> state_norm;
  std::vector<float> a_log;
  std::vector<float> dt_bias;
};

struct LayerWeights {
  LayerType type = LayerType::linear_attention;
  std::vector<float> input_norm;
  std::vector<float> post_norm;
  std::optional<AttentionWeights> attention;
  std::optional<GdnWeights> gdn;
  DenseMatrix gate;
  DenseMatrix up;
  DenseMatrix down;
};

struct LayerState {
  std::optional<KvCache> kv;
  std::optional<GdnReferenceState> gdn;
  std::vector<float> conv_history;
};

} // namespace

ReferenceProgramConfig
ReferenceProgramConfig::from_qwen(const QwenModelConfig &config,
                                  std::size_t max_context) {
  require(config.hidden_size > 0 && config.intermediate_size > 0 &&
              config.vocab_size > 0 && config.num_attention_heads > 0 &&
              config.num_key_value_heads > 0 && config.head_dim > 0,
          "Qwen config is incomplete");
  auto rotary_dim = static_cast<std::size_t>(std::llround(
      static_cast<double>(config.head_dim) * config.partial_rotary_factor));
  rotary_dim -= rotary_dim % 2;
  return {
      .vocab_size = static_cast<std::size_t>(config.vocab_size),
      .hidden_size = static_cast<std::size_t>(config.hidden_size),
      .intermediate_size = static_cast<std::size_t>(config.intermediate_size),
      .attention_heads =
          static_cast<std::size_t>(config.num_attention_heads),
      .kv_heads = static_cast<std::size_t>(config.num_key_value_heads),
      .head_dim = static_cast<std::size_t>(config.head_dim),
      .linear_key_heads =
          static_cast<std::size_t>(config.linear_num_key_heads),
      .linear_value_heads =
          static_cast<std::size_t>(config.linear_num_value_heads),
      .linear_key_dim =
          static_cast<std::size_t>(config.linear_key_head_dim),
      .linear_value_dim =
          static_cast<std::size_t>(config.linear_value_head_dim),
      .linear_conv_kernel =
          static_cast<std::size_t>(config.linear_conv_kernel_dim),
      .max_context = max_context,
      .rotary_dim = rotary_dim,
      .rms_norm_eps = static_cast<float>(config.rms_norm_eps),
      .rope_theta = config.rope_theta,
      .eos_token_id = config.eos_token_id,
      .layer_types = config.layer_types,
  };
}

class ReferenceProgram::Impl {
public:
  Impl(ReferenceProgramConfig selected_config, std::uint64_t weight_seed)
      : config(std::move(selected_config)), rng(weight_seed) {
    validate_config(config);
    std::uint64_t stream = 1;
    embedding = make_matrix(config.vocab_size, config.hidden_size, rng,
                            stream++);
    final_norm = make_norm(config.hidden_size, rng, stream++);
    lm_head =
        make_matrix(config.vocab_size, config.hidden_size, rng, stream++);
    layers.reserve(config.layer_types.size());
    for (const auto type : config.layer_types) {
      LayerWeights layer{
          .type = type,
          .input_norm = make_norm(config.hidden_size, rng, stream++),
          .post_norm = make_norm(config.hidden_size, rng, stream++),
          .gate = make_matrix(config.intermediate_size, config.hidden_size, rng,
                              stream++),
          .up = make_matrix(config.intermediate_size, config.hidden_size, rng,
                            stream++),
          .down = make_matrix(config.hidden_size, config.intermediate_size, rng,
                              stream++),
      };
      if (type == LayerType::full_attention) {
        const auto query_width = config.attention_heads * config.head_dim;
        const auto kv_width = config.kv_heads * config.head_dim;
        layer.attention = AttentionWeights{
            .q_gate = make_matrix(2 * query_width, config.hidden_size, rng,
                                  stream++),
            .key = make_matrix(kv_width, config.hidden_size, rng, stream++),
            .value = make_matrix(kv_width, config.hidden_size, rng, stream++),
            .output = make_matrix(config.hidden_size, query_width, rng,
                                  stream++),
            .query_norm = make_norm(config.head_dim, rng, stream++),
            .key_norm = make_norm(config.head_dim, rng, stream++),
        };
      } else {
        const auto key_width = config.linear_key_heads * config.linear_key_dim;
        const auto value_width =
            config.linear_value_heads * config.linear_value_dim;
        const auto conv_width = 2 * key_width + value_width;
        std::vector<float> conv(config.linear_conv_kernel * conv_width);
        for (std::size_t index = 0; index < conv.size(); ++index) {
          conv[index] =
              (static_cast<float>(rng.uniform(stream, index)) * 2.0F - 1.0F) /
              static_cast<float>(config.linear_conv_kernel);
        }
        ++stream;
        std::vector<float> a_log(config.linear_value_heads);
        std::vector<float> dt_bias(config.linear_value_heads);
        for (std::size_t head = 0; head < config.linear_value_heads; ++head) {
          a_log[head] = std::log(
              0.1F + static_cast<float>(rng.uniform(stream, head)) * 0.9F);
          dt_bias[head] =
              -1.0F + static_cast<float>(rng.uniform(stream + 1, head));
        }
        stream += 2;
        layer.gdn = GdnWeights{
            .qkv = make_matrix(conv_width, config.hidden_size, rng, stream++),
            .z = make_matrix(value_width, config.hidden_size, rng, stream++),
            .a = make_matrix(config.linear_value_heads, config.hidden_size, rng,
                             stream++),
            .b = make_matrix(config.linear_value_heads, config.hidden_size, rng,
                             stream++),
            .output = make_matrix(config.hidden_size, value_width, rng,
                                  stream++),
            .conv = std::move(conv),
            .state_norm =
                make_norm(config.linear_value_dim, rng, stream++),
            .a_log = std::move(a_log),
            .dt_bias = std::move(dt_bias),
        };
      }
      layers.push_back(std::move(layer));
    }
    reset();
  }

  void reset() {
    position = 0;
    states.clear();
    states.reserve(config.layer_types.size());
    const auto key_width = config.linear_key_heads * config.linear_key_dim;
    const auto value_width =
        config.linear_value_heads * config.linear_value_dim;
    const auto conv_width = 2 * key_width + value_width;
    for (const auto type : config.layer_types) {
      LayerState state;
      if (type == LayerType::full_attention) {
        state.kv.emplace(config.kv_heads, config.head_dim,
                         config.max_context);
      } else {
        state.gdn.emplace(GdnReferenceGeometry{
            .key_heads = config.linear_key_heads,
            .value_heads = config.linear_value_heads,
            .key_dim = config.linear_key_dim,
            .value_dim = config.linear_value_dim,
        });
        state.conv_history.assign(
            (config.linear_conv_kernel - 1) * conv_width, 0.0F);
      }
      states.push_back(std::move(state));
    }
  }

  [[nodiscard]] std::vector<float> run_attention(
      std::span<const float> input, const AttentionWeights &weights,
      LayerState &state) const {
    const auto query_width = config.attention_heads * config.head_dim;
    const auto combined = weights.q_gate.apply(input);
    auto query = std::vector<float>(combined.begin(),
                                    combined.begin() +
                                        static_cast<std::ptrdiff_t>(query_width));
    const auto gate = std::span<const float>(combined).subspan(query_width);
    auto key = weights.key.apply(input);
    const auto value = weights.value.apply(input);
    query = rms_norm(query, config.attention_heads, config.head_dim,
                     weights.query_norm, config.rms_norm_eps);
    key = rms_norm(key, config.kv_heads, config.head_dim, weights.key_norm,
                   config.rms_norm_eps);
    apply_rope(query, 1, config.attention_heads, config.head_dim,
               config.rotary_dim, position, config.rope_theta);
    apply_rope(key, 1, config.kv_heads, config.head_dim, config.rotary_dim,
               position, config.rope_theta);
    auto attended = attention_decode(query, config.attention_heads, key, value,
                                     *state.kv);
    for (std::size_t index = 0; index < attended.size(); ++index) {
      attended[index] *= sigmoid(gate[index]);
    }
    return weights.output.apply(attended);
  }

  [[nodiscard]] std::vector<float>
  run_gdn(std::span<const float> input, const GdnWeights &weights,
          LayerState &state) const {
    const auto key_width = config.linear_key_heads * config.linear_key_dim;
    const auto projected = weights.qkv.apply(input);
    const auto convolved = causal_depthwise_conv_silu(
        projected, 1, projected.size(), weights.conv,
        config.linear_conv_kernel, state.conv_history);
    auto query = std::vector<float>(
        convolved.begin(),
        convolved.begin() + static_cast<std::ptrdiff_t>(key_width));
    auto key = std::vector<float>(
        convolved.begin() + static_cast<std::ptrdiff_t>(key_width),
        convolved.begin() + static_cast<std::ptrdiff_t>(2 * key_width));
    const auto value = std::span<const float>(convolved).subspan(2 * key_width);
    const std::span<const float> no_weight;
    query = rms_norm(query, config.linear_key_heads, config.linear_key_dim,
                     no_weight, 1.0e-6F);
    key = rms_norm(key, config.linear_key_heads, config.linear_key_dim,
                   no_weight, 1.0e-6F);
    const auto inv_scale =
        1.0F / std::sqrt(static_cast<float>(config.linear_key_dim));
    for (auto &item : query) {
      item *= inv_scale * inv_scale;
    }
    for (auto &item : key) {
      item *= inv_scale;
    }

    const auto projected_a = weights.a.apply(input);
    const auto projected_b = weights.b.apply(input);
    std::vector<float> decay(config.linear_value_heads);
    std::vector<float> beta(config.linear_value_heads);
    for (std::size_t head = 0; head < config.linear_value_heads; ++head) {
      decay[head] = std::exp(-std::exp(weights.a_log[head]) *
                             softplus(projected_a[head] +
                                      weights.dt_bias[head]));
      beta[head] = sigmoid(projected_b[head]);
    }
    auto recurrent =
        gdn_step(query, key, value, decay, beta, *state.gdn);
    recurrent = rms_norm(recurrent, config.linear_value_heads,
                         config.linear_value_dim, weights.state_norm,
                         config.rms_norm_eps);
    const auto z = weights.z.apply(input);
    for (std::size_t index = 0; index < recurrent.size(); ++index) {
      recurrent[index] *= z[index] * sigmoid(z[index]);
    }
    return weights.output.apply(recurrent);
  }

  [[nodiscard]] std::vector<float> step(std::int32_t token) {
    if (position == config.max_context) {
      throw std::length_error("reference Program context capacity exceeded");
    }
    if (token < 0 || static_cast<std::size_t>(token) >= config.vocab_size) {
      throw std::out_of_range("reference Program token is outside vocabulary");
    }
    const auto begin = embedding.values.begin() +
                       static_cast<std::ptrdiff_t>(token * config.hidden_size);
    std::vector<float> hidden(begin, begin +
                                        static_cast<std::ptrdiff_t>(
                                            config.hidden_size));
    for (std::size_t index = 0; index < layers.size(); ++index) {
      const auto &layer = layers[index];
      auto normalized = rms_norm(hidden, 1, config.hidden_size,
                                 layer.input_norm, config.rms_norm_eps);
      const auto mixed = layer.type == LayerType::full_attention
                             ? run_attention(normalized, *layer.attention,
                                             states[index])
                             : run_gdn(normalized, *layer.gdn, states[index]);
      add_in_place(hidden, mixed);
      normalized = rms_norm(hidden, 1, config.hidden_size, layer.post_norm,
                            config.rms_norm_eps);
      const auto activated =
          silu_multiply(layer.gate.apply(normalized),
                        layer.up.apply(normalized));
      add_in_place(hidden, layer.down.apply(activated));
    }
    hidden = rms_norm(hidden, 1, config.hidden_size, final_norm,
                      config.rms_norm_eps);
    ++position;
    return lm_head.apply(hidden);
  }

  [[nodiscard]] std::uint64_t parameter_count() const {
    std::uint64_t count = embedding.parameter_count() +
                          final_norm.size() + lm_head.parameter_count();
    for (const auto &layer : layers) {
      count += layer.input_norm.size() + layer.post_norm.size() +
               layer.gate.parameter_count() + layer.up.parameter_count() +
               layer.down.parameter_count();
      if (layer.attention.has_value()) {
        count += layer.attention->q_gate.parameter_count() +
                 layer.attention->key.parameter_count() +
                 layer.attention->value.parameter_count() +
                 layer.attention->output.parameter_count() +
                 layer.attention->query_norm.size() +
                 layer.attention->key_norm.size();
      } else {
        count += layer.gdn->qkv.parameter_count() +
                 layer.gdn->z.parameter_count() +
                 layer.gdn->a.parameter_count() +
                 layer.gdn->b.parameter_count() +
                 layer.gdn->output.parameter_count() +
                 layer.gdn->conv.size() + layer.gdn->state_norm.size() +
                 layer.gdn->a_log.size() + layer.gdn->dt_bias.size();
      }
    }
    return count;
  }

  ReferenceProgramConfig config;
  CounterRng rng;
  DenseMatrix embedding;
  std::vector<float> final_norm;
  DenseMatrix lm_head;
  std::vector<LayerWeights> layers;
  std::vector<LayerState> states;
  std::size_t position = 0;
};

ReferenceProgram ReferenceProgram::create_deterministic(
    ReferenceProgramConfig config, std::uint64_t weight_seed) {
  return ReferenceProgram(
      std::make_unique<Impl>(std::move(config), weight_seed));
}

ReferenceProgram::ReferenceProgram(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
ReferenceProgram::ReferenceProgram(ReferenceProgram &&) noexcept = default;
ReferenceProgram &
ReferenceProgram::operator=(ReferenceProgram &&) noexcept = default;
ReferenceProgram::~ReferenceProgram() = default;

const ReferenceProgramConfig &ReferenceProgram::config() const {
  return impl_->config;
}
std::size_t ReferenceProgram::position() const { return impl_->position; }
std::uint64_t ReferenceProgram::parameter_count() const {
  return impl_->parameter_count();
}
void ReferenceProgram::reset() { impl_->reset(); }
std::vector<float> ReferenceProgram::step(std::int32_t token) {
  return impl_->step(token);
}

std::vector<float>
ReferenceProgram::prefill(std::span<const std::int32_t> tokens,
                          std::size_t chunk_size) {
  require(!tokens.empty() && chunk_size > 0,
          "reference prefill needs tokens and a positive chunk size");
  std::vector<float> logits;
  for (std::size_t chunk = 0; chunk < tokens.size(); chunk += chunk_size) {
    const auto end = std::min(tokens.size(), chunk + chunk_size);
    for (std::size_t index = chunk; index < end; ++index) {
      logits = impl_->step(tokens[index]);
    }
  }
  return logits;
}

ReferenceRunResult ReferenceProgram::generate(
    std::span<const std::int32_t> prompt, std::size_t prefill_chunk,
    std::size_t max_new_tokens, const SamplingOptions &sampling,
    std::uint64_t sampling_seed, const std::function<bool()> &cancelled) {
  require(!prompt.empty(), "reference generation prompt must not be empty");
  auto logits = prefill(prompt, prefill_chunk);
  ReferenceRunResult result;
  result.tokens.reserve(max_new_tokens);
  const CounterRng sampling_rng(sampling_seed);
  for (std::size_t index = 0; index < max_new_tokens; ++index) {
    if (cancelled && cancelled()) {
      result.cancelled = true;
      break;
    }
    const auto token = sample_logits(logits, sampling, sampling_rng, 0,
                                     impl_->position);
    result.tokens.push_back(token);
    if (token == impl_->config.eos_token_id) {
      result.reached_eos = true;
      break;
    }
    if (index + 1 < max_new_tokens) {
      logits = impl_->step(token);
    }
  }
  return result;
}

} // namespace omlx::m5
