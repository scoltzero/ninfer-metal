#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/gdn.h"
#include "omlx_m5/quantization.h"
#include "omlx_m5/reference.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

omlx::m5::QwenModelConfig config() {
  omlx::m5::QwenModelConfig value;
  value.linear_num_key_heads = 2;
  value.linear_num_value_heads = 4;
  value.linear_key_head_dim = 64;
  value.linear_value_head_dim = 32;
  value.linear_conv_kernel_dim = 4;
  value.rms_norm_eps = 1.0e-6;
  return value;
}

struct DirectWeights {
  std::vector<std::uint16_t> conv_words;
  std::vector<std::uint16_t> a_log_words;
  std::vector<std::uint16_t> dt_bias_words;
  std::vector<std::uint16_t> norm_words;
  std::vector<float> conv_tap_major;
  std::vector<float> a_log;
  std::vector<float> dt_bias;
  std::vector<float> norm;
  omlx::m5::MetalBuffer conv;
  omlx::m5::MetalBuffer a_log_buffer;
  omlx::m5::MetalBuffer dt_bias_buffer;
  omlx::m5::MetalBuffer norm_buffer;
  omlx::m5::GdnDirectWeights view;
};

DirectWeights make_weights(omlx::m5::MetalContext &context,
                           const omlx::m5::QwenModelConfig &cfg) {
  const auto key_width = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
  const auto value_width =
      cfg.linear_num_value_heads * cfg.linear_value_head_dim;
  const auto conv_width = key_width * 2 + value_width;
  DirectWeights result;
  result.conv_words.resize(conv_width * cfg.linear_conv_kernel_dim);
  result.conv_tap_major.resize(result.conv_words.size());
  for (std::int32_t channel = 0; channel < conv_width; ++channel) {
    for (std::int32_t tap = 0; tap < cfg.linear_conv_kernel_dim; ++tap) {
      const auto raw = 0.02F * float((channel + tap * 3) % 9 - 4);
      const auto word = omlx::m5::float_to_bf16(raw);
      result.conv_words[channel * cfg.linear_conv_kernel_dim + tap] = word;
      result.conv_tap_major[tap * conv_width + channel] =
          omlx::m5::bf16_to_float(word);
    }
  }
  result.a_log_words.resize(cfg.linear_num_value_heads);
  result.dt_bias_words.resize(cfg.linear_num_value_heads);
  result.a_log.resize(cfg.linear_num_value_heads);
  result.dt_bias.resize(cfg.linear_num_value_heads);
  for (std::int32_t head = 0; head < cfg.linear_num_value_heads; ++head) {
    result.a_log_words[head] =
        omlx::m5::float_to_bf16(-1.5F + 0.1F * float(head));
    result.dt_bias_words[head] =
        omlx::m5::float_to_bf16(-0.25F + 0.05F * float(head));
    result.a_log[head] = omlx::m5::bf16_to_float(result.a_log_words[head]);
    result.dt_bias[head] = omlx::m5::bf16_to_float(result.dt_bias_words[head]);
  }
  result.norm_words.resize(cfg.linear_value_head_dim);
  result.norm.resize(cfg.linear_value_head_dim);
  for (std::int32_t feature = 0; feature < cfg.linear_value_head_dim;
       ++feature) {
    result.norm_words[feature] =
        omlx::m5::float_to_bf16(0.75F + float(feature % 7) / 32.0F);
    result.norm[feature] = omlx::m5::bf16_to_float(result.norm_words[feature]);
  }
  result.conv = context.upload(bytes(result.conv_words));
  result.a_log_buffer = context.upload(bytes(result.a_log_words));
  result.dt_bias_buffer = context.upload(bytes(result.dt_bias_words));
  result.norm_buffer = context.upload(bytes(result.norm_words));
  result.view = {
      .conv = &result.conv,
      .conv_bytes = bytes(result.conv_words).size(),
      .a_log = &result.a_log_buffer,
      .a_log_bytes = bytes(result.a_log_words).size(),
      .dt_bias = &result.dt_bias_buffer,
      .dt_bias_bytes = bytes(result.dt_bias_words).size(),
      .norm = &result.norm_buffer,
      .norm_bytes = bytes(result.norm_words).size(),
  };
  return result;
}

struct ChunkInputs {
  std::vector<std::uint16_t> qkv_words;
  std::vector<std::uint16_t> z_words;
  std::vector<std::uint16_t> a_words;
  std::vector<std::uint16_t> b_words;
  std::vector<float> qkv;
  std::vector<float> z;
  std::vector<float> a;
  std::vector<float> b;
};

ChunkInputs make_inputs(const omlx::m5::QwenModelConfig &cfg,
                        std::uint32_t tokens, std::uint32_t start) {
  const auto key_width = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
  const auto value_width =
      cfg.linear_num_value_heads * cfg.linear_value_head_dim;
  const auto conv_width = key_width * 2 + value_width;
  ChunkInputs result;
  result.qkv_words.resize(tokens * conv_width);
  result.z_words.resize(tokens * value_width);
  result.a_words.resize(tokens * cfg.linear_num_value_heads);
  result.b_words.resize(tokens * cfg.linear_num_value_heads);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto absolute = start + token;
    for (std::int32_t feature = 0; feature < conv_width; ++feature) {
      result.qkv_words[token * conv_width + feature] = omlx::m5::float_to_bf16(
          float(
              static_cast<std::int32_t>((feature * 11 + absolute * 17) % 113) -
              56) /
          256.0F);
    }
    for (std::int32_t feature = 0; feature < value_width; ++feature) {
      result.z_words[token * value_width + feature] = omlx::m5::float_to_bf16(
          float(
              static_cast<std::int32_t>((feature * 13 + absolute * 19) % 107) -
              53) /
          256.0F);
    }
    for (std::int32_t head = 0; head < cfg.linear_num_value_heads; ++head) {
      result.a_words[token * cfg.linear_num_value_heads + head] =
          omlx::m5::float_to_bf16(-0.2F + 0.01F * float(absolute + head));
      result.b_words[token * cfg.linear_num_value_heads + head] =
          omlx::m5::float_to_bf16(0.1F + 0.02F * float(absolute + head));
    }
  }
  const auto convert = [](std::span<const std::uint16_t> words) {
    std::vector<float> values(words.size());
    std::ranges::transform(words, values.begin(), omlx::m5::bf16_to_float);
    return values;
  };
  result.qkv = convert(result.qkv_words);
  result.z = convert(result.z_words);
  result.a = convert(result.a_words);
  result.b = convert(result.b_words);
  return result;
}

std::vector<std::uint16_t> execute_chunk(omlx::m5::MetalContext &context,
                                         const omlx::m5::QwenModelConfig &cfg,
                                         const DirectWeights &weights,
                                         omlx::m5::GdnLayerState &state,
                                         const ChunkInputs &input,
                                         std::uint32_t tokens) {
  const auto key_width = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
  const auto value_width =
      cfg.linear_num_value_heads * cfg.linear_value_head_dim;
  const auto conv_width = key_width * 2 + value_width;
  auto qkv = context.upload(bytes(input.qkv_words));
  auto z = context.upload(bytes(input.z_words));
  auto a = context.upload(bytes(input.a_words));
  auto b = context.upload(bytes(input.b_words));
  const omlx::m5::ProjectionParentOutputView qkv_view{
      .role = "gdn_qkv",
      .buffer = &qkv,
      .bytes = static_cast<std::uint64_t>(tokens) * conv_width * 2,
      .tokens = tokens,
      .features = static_cast<std::uint64_t>(conv_width),
      .token_stride_bytes = static_cast<std::uint64_t>(conv_width) * 2,
  };
  const omlx::m5::ProjectionParentOutputView z_view{
      .role = "gdn_z",
      .buffer = &z,
      .bytes = static_cast<std::uint64_t>(tokens) * value_width * 2,
      .tokens = tokens,
      .features = static_cast<std::uint64_t>(value_width),
      .token_stride_bytes = static_cast<std::uint64_t>(value_width) * 2,
  };
  auto plan = omlx::m5::prepare_gdn_layer(
      context, OMLX_M5_GDN_LAYER_METALLIB_PATH, cfg, tokens);
  auto output = context.make_buffer(plan.info().output_bytes);
  const auto submission =
      omlx::m5::execute_gdn_layer(context, plan, qkv_view, z_view, a, 0, b, 0,
                                  weights.view, state, output, 0);
  expect(submission.dispatch_count == 3, "GDN dispatch count is wrong");
  const auto raw = output.read();
  std::vector<std::uint16_t> words(raw.size() / 2);
  std::memcpy(words.data(), raw.data(), raw.size());
  return words;
}

void run_gdn() {
  constexpr std::uint32_t tokens = 5;
  auto cfg = config();
  auto context = omlx::m5::MetalContext::create();
  auto weights = make_weights(context, cfg);
  const auto all_input = make_inputs(cfg, tokens, 0);
  omlx::m5::GdnLayerState whole_state(context, cfg);
  const auto whole =
      execute_chunk(context, cfg, weights, whole_state, all_input, tokens);

  omlx::m5::GdnLayerState chunked_state(context, cfg);
  const auto first_input = make_inputs(cfg, 3, 0);
  const auto second_input = make_inputs(cfg, 2, 3);
  auto chunked =
      execute_chunk(context, cfg, weights, chunked_state, first_input, 3);
  const auto second =
      execute_chunk(context, cfg, weights, chunked_state, second_input, 2);
  chunked.insert(chunked.end(), second.begin(), second.end());
  expect(chunked == whole, "GDN output changed across chunk boundaries");
  expect(chunked_state.recurrent().read() == whole_state.recurrent().read() &&
             chunked_state.history().read() == whole_state.history().read(),
         "GDN state changed across chunk boundaries");

  const auto key_width = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
  const auto value_width =
      cfg.linear_num_value_heads * cfg.linear_value_head_dim;
  const auto conv_width = key_width * 2 + value_width;
  std::vector<float> history((cfg.linear_conv_kernel_dim - 1) * conv_width,
                             0.0F);
  const auto convolved = omlx::m5::causal_depthwise_conv_silu(
      all_input.qkv, tokens, conv_width, weights.conv_tap_major,
      cfg.linear_conv_kernel_dim, history);
  std::vector<float> query(tokens * key_width);
  std::vector<float> key(tokens * key_width);
  std::vector<float> value(tokens * value_width);
  std::vector<float> decay(tokens * cfg.linear_num_value_heads);
  std::vector<float> beta(decay.size());
  const auto inv_scale = 1.0F / std::sqrt(float(cfg.linear_key_head_dim));
  for (std::uint32_t token = 0; token < tokens; ++token) {
    std::copy_n(convolved.begin() + token * conv_width, key_width,
                query.begin() + token * key_width);
    std::copy_n(convolved.begin() + token * conv_width + key_width, key_width,
                key.begin() + token * key_width);
    std::copy_n(convolved.begin() + token * conv_width + key_width * 2,
                value_width, value.begin() + token * value_width);
    for (std::int32_t head = 0; head < cfg.linear_num_key_heads; ++head) {
      auto q = omlx::m5::rms_norm(
          std::span<const float>(query).subspan(
              token * key_width + head * cfg.linear_key_head_dim,
              cfg.linear_key_head_dim),
          1, cfg.linear_key_head_dim, {}, 1.0e-6F);
      auto k = omlx::m5::rms_norm(
          std::span<const float>(key).subspan(
              token * key_width + head * cfg.linear_key_head_dim,
              cfg.linear_key_head_dim),
          1, cfg.linear_key_head_dim, {}, 1.0e-6F);
      for (std::int32_t feature = 0; feature < cfg.linear_key_head_dim;
           ++feature) {
        query[token * key_width + head * cfg.linear_key_head_dim + feature] =
            q[feature] * inv_scale * inv_scale;
        key[token * key_width + head * cfg.linear_key_head_dim + feature] =
            k[feature] * inv_scale;
      }
    }
    for (std::int32_t head = 0; head < cfg.linear_num_value_heads; ++head) {
      const auto activation =
          all_input.a[token * cfg.linear_num_value_heads + head] +
          weights.dt_bias[head];
      const auto softplus =
          activation > 20.0F ? activation : std::log1p(std::exp(activation));
      decay[token * cfg.linear_num_value_heads + head] =
          std::exp(-std::exp(weights.a_log[head]) * softplus);
      const auto b = all_input.b[token * cfg.linear_num_value_heads + head];
      beta[token * cfg.linear_num_value_heads + head] =
          1.0F / (1.0F + std::exp(-b));
    }
  }
  omlx::m5::GdnReferenceState reference_state({
      .key_heads = static_cast<std::size_t>(cfg.linear_num_key_heads),
      .value_heads = static_cast<std::size_t>(cfg.linear_num_value_heads),
      .key_dim = static_cast<std::size_t>(cfg.linear_key_head_dim),
      .value_dim = static_cast<std::size_t>(cfg.linear_value_head_dim),
  });
  const auto recurrent = omlx::m5::gdn_prefill(query, key, value, decay, beta,
                                               tokens, reference_state, tokens);
  std::vector<float> expected(tokens * value_width);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::int32_t head = 0; head < cfg.linear_num_value_heads; ++head) {
      const auto normalized = omlx::m5::rms_norm(
          std::span<const float>(recurrent).subspan(
              token * value_width + head * cfg.linear_value_head_dim,
              cfg.linear_value_head_dim),
          1, cfg.linear_value_head_dim, weights.norm,
          static_cast<float>(cfg.rms_norm_eps));
      for (std::int32_t feature = 0; feature < cfg.linear_value_head_dim;
           ++feature) {
        const auto index =
            token * value_width + head * cfg.linear_value_head_dim + feature;
        const auto gate = all_input.z[index];
        expected[index] = normalized[feature] * gate / (1.0F + std::exp(-gate));
      }
    }
  }
  for (std::size_t index = 0; index < whole.size(); ++index) {
    const auto actual = omlx::m5::bf16_to_float(whole[index]);
    expect(std::abs(actual - expected[index]) <=
               0.02F * std::max(1.0F, std::abs(expected[index])),
           "GDN output differs from the independent reference");
  }
  whole_state.reset();
  expect(std::ranges::all_of(
             whole_state.recurrent().read(),
             [](std::byte value) { return value == std::byte{}; }) &&
             std::ranges::all_of(
                 whole_state.history().read(),
                 [](std::byte value) { return value == std::byte{}; }),
         "GDN reset did not clear persistent state");
}

} // namespace

int main() {
  try {
    run_gdn();
    std::cout << "B3.7 GDN layer tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B3.7 GDN layer failure: " << error.what() << '\n';
    return 1;
  }
}
