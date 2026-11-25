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

#include "omlx_m5/attention.h"
#include "omlx_m5/projection.h"
#include "omlx_m5/quantization.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <class Exception = std::exception, class Function>
void expect_throws(Function &&function, const std::string &message) {
  try {
    function();
  } catch (const Exception &) {
    return;
  }
  throw std::runtime_error(message);
}

template <class T> std::vector<std::byte> bytes(const std::vector<T> &values) {
  const auto view = std::as_bytes(std::span(values));
  return {view.begin(), view.end()};
}

omlx::m5::QwenModelConfig config() {
  omlx::m5::QwenModelConfig value;
  value.hidden_size = 128;
  value.num_attention_heads = 2;
  value.num_key_value_heads = 1;
  value.head_dim = 64;
  value.partial_rotary_factor = 0.25;
  value.rope_theta = 10000.0;
  value.rms_norm_eps = 1.0e-6;
  value.linear_num_key_heads = 1;
  value.linear_num_value_heads = 2;
  value.linear_key_head_dim = 64;
  value.linear_value_head_dim = 64;
  return value;
}

float read_bf16(const omlx::m5::MetalBuffer &buffer, std::uint64_t offset) {
  const auto raw = buffer.read(offset, 2);
  std::uint16_t word = 0;
  std::memcpy(&word, raw.data(), sizeof(word));
  return omlx::m5::bf16_to_float(word);
}

std::vector<std::uint16_t> read_words(const omlx::m5::MetalBuffer &buffer,
                                      std::uint64_t offset,
                                      std::uint64_t count) {
  const auto raw = buffer.read(offset, count * 2);
  std::vector<std::uint16_t> result(count);
  std::memcpy(result.data(), raw.data(), raw.size());
  return result;
}

std::vector<float> reference_head(std::span<const float> input,
                                  std::span<const float> weight,
                                  std::uint32_t rotary_dim,
                                  std::uint64_t position, double theta,
                                  float epsilon) {
  double squares = 0.0;
  for (const auto value : input) {
    squares += static_cast<double>(value) * value;
  }
  const auto inverse = static_cast<float>(
      1.0 / std::sqrt(squares / static_cast<double>(input.size()) + epsilon));
  std::vector<float> output(input.size());
  for (std::size_t index = 0; index < input.size(); ++index) {
    output[index] = input[index] * inverse * weight[index];
  }
  const auto half = rotary_dim / 2;
  for (std::uint32_t pair = 0; pair < half; ++pair) {
    const auto exponent = static_cast<double>(2 * pair) / rotary_dim;
    const auto angle =
        static_cast<double>(position) / std::pow(theta, exponent);
    const auto cosine = static_cast<float>(std::cos(angle));
    const auto sine = static_cast<float>(std::sin(angle));
    const auto first = output[pair];
    const auto second = output[pair + half];
    output[pair] = first * cosine - second * sine;
    output[pair + half] = first * sine + second * cosine;
  }
  return output;
}

void run_qk_norm_rope() {
  constexpr std::uint32_t tokens = 3;
  constexpr std::uint64_t start_position = 17;
  auto context = omlx::m5::MetalContext::create();
  const auto selected_config = config();
  auto bundle = omlx::m5::make_full_attention_projection_bundle(
      selected_config, "layer.self_attn");
  omlx::m5::ProjectionOutputArena arena(context, std::move(bundle), tokens);
  const auto q_parent = arena.parent("attention_q_gate");
  const auto k_parent = arena.parent("attention_k");
  const auto query = arena.output("attention_query");
  const auto gate = arena.output("attention_gate");
  const auto key = arena.output("attention_key");

  std::vector<std::uint16_t> query_parent_words(tokens * q_parent.features);
  std::vector<std::uint16_t> key_parent_words(tokens * k_parent.features);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint64_t feature = 0; feature < q_parent.features; ++feature) {
      const auto raw =
          static_cast<float>(
              static_cast<std::int32_t>((feature * 19 + token * 23) % 257) -
              128) /
          256.0F;
      query_parent_words[token * q_parent.features + feature] =
          omlx::m5::float_to_bf16(raw);
    }
    for (std::uint64_t feature = 0; feature < k_parent.features; ++feature) {
      const auto raw =
          static_cast<float>(
              static_cast<std::int32_t>((feature * 29 + token * 31) % 257) -
              128) /
          256.0F;
      key_parent_words[token * k_parent.features + feature] =
          omlx::m5::float_to_bf16(raw);
    }
  }
  arena.buffer().write(bytes(query_parent_words), q_parent.offset);
  arena.buffer().write(bytes(key_parent_words), k_parent.offset);

  std::vector<std::uint16_t> norm_words(selected_config.head_dim);
  std::vector<float> norm_values(selected_config.head_dim);
  for (std::uint32_t feature = 0; feature < selected_config.head_dim;
       ++feature) {
    norm_words[feature] =
        omlx::m5::float_to_bf16(0.75F + float(feature % 13) / 32.0F);
    norm_values[feature] = omlx::m5::bf16_to_float(norm_words[feature]);
  }
  auto q_norm = context.upload(bytes(norm_words));
  auto k_norm = context.upload(bytes(norm_words));
  std::vector<std::uint16_t> gate_before(tokens * gate.features);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint64_t feature = 0; feature < gate.features; ++feature) {
      const auto value =
          arena.buffer().read(gate.feature_offset(token, feature), 2);
      std::memcpy(&gate_before[token * gate.features + feature], value.data(),
                  2);
    }
  }
  auto plan = omlx::m5::prepare_attention_postprocess(
      context, OMLX_M5_ATTENTION_POSTPROCESS_METALLIB_PATH, selected_config,
      tokens);
  const auto before = context.registry_stats();
  const auto submission = omlx::m5::execute_attention_postprocess(
      context, plan, query, key, q_norm, 0, k_norm, 0, start_position);
  const auto after = context.registry_stats();
  expect(submission.dispatch_count == 1 &&
             after.upload_count == before.upload_count &&
             after.pipelines_compiled == before.pipelines_compiled,
         "attention postprocess changed token-loop resources");

  const auto check = [&](const omlx::m5::ProjectionSplitOutputView &view,
                         std::uint32_t heads,
                         const std::vector<std::uint16_t> &parent_words,
                         std::uint64_t parent_features, bool interleaved) {
    for (std::uint32_t token = 0; token < tokens; ++token) {
      for (std::uint32_t head = 0; head < heads; ++head) {
        std::vector<float> input(selected_config.head_dim);
        for (std::uint32_t feature = 0; feature < selected_config.head_dim;
             ++feature) {
          const auto parent_feature =
              interleaved ? head * selected_config.head_dim * 2 + feature
                          : head * selected_config.head_dim + feature;
          input[feature] = omlx::m5::bf16_to_float(
              parent_words[token * parent_features + parent_feature]);
        }
        const auto expected = reference_head(
            input, norm_values, plan.info().rotary_dim, start_position + token,
            plan.info().theta, plan.info().epsilon);
        for (std::uint32_t feature = 0; feature < selected_config.head_dim;
             ++feature) {
          const auto logical = head * selected_config.head_dim + feature;
          const auto actual =
              read_bf16(arena.buffer(), view.feature_offset(token, logical));
          const auto tolerance =
              0.012F * std::max(1.0F, std::abs(expected[feature]));
          expect(std::isfinite(actual) &&
                     std::abs(actual - expected[feature]) <= tolerance,
                 "attention q/k norm+RoPE differs from reference");
        }
      }
    }
  };
  check(query, selected_config.num_attention_heads, query_parent_words,
        q_parent.features, true);
  check(key, selected_config.num_key_value_heads, key_parent_words,
        k_parent.features, false);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    for (std::uint64_t feature = 0; feature < gate.features; ++feature) {
      std::uint16_t actual = 0;
      const auto raw =
          arena.buffer().read(gate.feature_offset(token, feature), 2);
      std::memcpy(&actual, raw.data(), 2);
      expect(actual == gate_before[token * gate.features + feature],
             "attention postprocess modified the query gate block");
    }
  }
  auto invalid = query;
  invalid.block_stride_bytes -= 2;
  expect_throws(
      [&] {
        static_cast<void>(omlx::m5::execute_attention_postprocess(
            context, plan, invalid, key, q_norm, 0, k_norm, 0, start_position));
      },
      "attention postprocess accepted an invalid query stride");
}

std::vector<std::uint16_t>
run_paged_chunk(omlx::m5::MetalContext &context,
                const omlx::m5::QwenModelConfig &selected_config,
                omlx::m5::PagedKvCache &cache, std::uint32_t tokens,
                std::uint32_t absolute_start) {
  auto bundle = omlx::m5::make_full_attention_projection_bundle(
      selected_config, "layer.self_attn");
  omlx::m5::ProjectionOutputArena arena(context, std::move(bundle), tokens);
  const auto q_parent = arena.parent("attention_q_gate");
  const auto k_parent = arena.parent("attention_k");
  const auto v_parent = arena.parent("attention_v");
  std::vector<std::uint16_t> q_words(tokens * q_parent.features);
  std::vector<std::uint16_t> k_words(tokens * k_parent.features);
  std::vector<std::uint16_t> v_words(tokens * v_parent.features);
  for (std::uint32_t token = 0; token < tokens; ++token) {
    const auto absolute = absolute_start + token;
    for (std::uint64_t feature = 0; feature < q_parent.features; ++feature) {
      q_words[token * q_parent.features + feature] = omlx::m5::float_to_bf16(
          static_cast<float>(
              static_cast<std::int32_t>((feature * 7 + absolute * 11) % 97) -
              48) /
          128.0F);
    }
    for (std::uint64_t feature = 0; feature < k_parent.features; ++feature) {
      k_words[token * k_parent.features + feature] = omlx::m5::float_to_bf16(
          static_cast<float>(
              static_cast<std::int32_t>((feature * 13 + absolute * 17) % 89) -
              44) /
          128.0F);
      v_words[token * v_parent.features + feature] = omlx::m5::float_to_bf16(
          static_cast<float>((feature * 19 + absolute * 23) % 83) / 128.0F);
    }
  }
  arena.buffer().write(bytes(q_words), q_parent.offset);
  arena.buffer().write(bytes(k_words), k_parent.offset);
  arena.buffer().write(bytes(v_words), v_parent.offset);
  const auto query = arena.output("attention_query");
  const auto key = arena.output("attention_key");
  const auto value = arena.output("attention_value");
  const auto gate = arena.output("attention_gate");
  auto plan = omlx::m5::prepare_paged_attention(
      context, OMLX_M5_ATTENTION_POSTPROCESS_METALLIB_PATH, selected_config,
      tokens, cache.info());
  auto output = context.make_buffer(plan.info().output_bytes);
  const auto submission = omlx::m5::execute_paged_attention(
      context, plan, query, key, value, gate, cache, output, 0);
  expect(submission.dispatch_count == 2,
         "paged attention did not use append+attention dispatches");
  return read_words(output, 0,
                    tokens * selected_config.num_attention_heads *
                        selected_config.head_dim);
}

void run_paged_attention() {
  auto selected_config = config();
  auto context = omlx::m5::MetalContext::create();
  omlx::m5::PagedKvCache chunked_cache(context,
                                       selected_config.num_key_value_heads,
                                       selected_config.head_dim, 8, 3);
  const auto first =
      run_paged_chunk(context, selected_config, chunked_cache, 3, 0);
  const auto second =
      run_paged_chunk(context, selected_config, chunked_cache, 2, 3);
  expect(chunked_cache.tokens() == 5 && chunked_cache.info().page_count == 3,
         "paged KV frontier or page count is wrong");

  omlx::m5::PagedKvCache whole_cache(context,
                                     selected_config.num_key_value_heads,
                                     selected_config.head_dim, 8, 3);
  const auto whole =
      run_paged_chunk(context, selected_config, whole_cache, 5, 0);
  omlx::m5::PagedKvCache poisoned_cache(context,
                                        selected_config.num_key_value_heads,
                                        selected_config.head_dim, 8, 3);
  auto &poisoned_keys =
      const_cast<omlx::m5::MetalBuffer &>(poisoned_cache.keys());
  auto &poisoned_values =
      const_cast<omlx::m5::MetalBuffer &>(poisoned_cache.values());
  poisoned_keys.write(
      std::vector<std::byte>(poisoned_keys.size(), std::byte{0x7f}));
  poisoned_values.write(
      std::vector<std::byte>(poisoned_values.size(), std::byte{0x7f}));
  const auto poisoned =
      run_paged_chunk(context, selected_config, poisoned_cache, 5, 0);
  expect(poisoned == whole, "paged attention read an unwritten page or tail");
  std::vector<std::uint16_t> chunked = first;
  chunked.insert(chunked.end(), second.begin(), second.end());
  expect(chunked == whole,
         "paged attention output changed across prefill chunk boundaries");
  const auto used_bytes = 5 * chunked_cache.info().bytes_per_token;
  expect(chunked_cache.keys().read(0, used_bytes) ==
                 whole_cache.keys().read(0, used_bytes) &&
             chunked_cache.values().read(0, used_bytes) ==
                 whole_cache.values().read(0, used_bytes),
         "paged KV contents changed across chunk boundaries");
  chunked_cache.truncate(3);
  expect(chunked_cache.tokens() == 3, "paged KV truncate failed");
  chunked_cache.reset();
  expect(chunked_cache.tokens() == 0, "paged KV reset failed");
  expect_throws([&] { chunked_cache.truncate(1); },
                "paged KV accepted truncate beyond frontier");
}

void run_paged_attention_validation() {
  constexpr std::uint32_t tokens = 2;
  const auto selected_config = config();
  auto context = omlx::m5::MetalContext::create();
  auto bundle = omlx::m5::make_full_attention_projection_bundle(
      selected_config, "layer.self_attn");
  omlx::m5::ProjectionOutputArena arena(context, std::move(bundle), tokens);
  const auto query = arena.output("attention_query");
  const auto key = arena.output("attention_key");
  const auto value = arena.output("attention_value");
  const auto gate = arena.output("attention_gate");

  omlx::m5::PagedKvCache cache(context, selected_config.num_key_value_heads,
                               selected_config.head_dim, 8, 3);
  auto plan = omlx::m5::prepare_paged_attention(
      context, OMLX_M5_ATTENTION_POSTPROCESS_METALLIB_PATH, selected_config,
      tokens, cache.info());
  auto output = context.make_buffer(plan.info().output_bytes);

  // Mismatched page_tokens
  omlx::m5::PagedKvCache wrong_page_tokens_cache(
      context, selected_config.num_key_value_heads, selected_config.head_dim, 8,
      4);
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, key, value, gate, wrong_page_tokens_cache,
            output, 0));
      },
      "MIVE-16 accepted mismatched cache page_tokens");

  // Mismatched capacity_tokens / physical_page_count
  omlx::m5::PagedKvCache wrong_capacity_cache(
      context, selected_config.num_key_value_heads, selected_config.head_dim,
      12, 3);
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, key, value, gate, wrong_capacity_cache,
            output, 0));
      },
      "MIVE-16 accepted mismatched cache capacity/physical_page_count");

  // Non-compact token_stride_bytes (even when contiguous_per_token() is true)
  auto strided_key = key;
  strided_key.token_stride_bytes += 2;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, strided_key, value, gate, cache, output, 0));
      },
      "MIVE-16 accepted strided key view");

  auto strided_value = value;
  strided_value.token_stride_bytes += 2;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, key, strided_value, gate, cache, output, 0));
      },
      "MIVE-16 accepted strided value view");

  auto strided_query = query;
  strided_query.token_stride_bytes += 2;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, strided_query, key, value, gate, cache, output, 0));
      },
      "MIVE-16 accepted strided query view");

  auto strided_gate = gate;
  strided_gate.token_stride_bytes += 2;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, key, value, strided_gate, cache, output, 0));
      },
      "MIVE-16 accepted strided gate view");

  // Truncated span_bytes
  auto short_key = key;
  short_key.span_bytes -= 2;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, short_key, value, gate, cache, output, 0));
      },
      "MIVE-16 accepted under-sized key span_bytes");

  // Odd byte offsets on BF16 input views
  auto odd_query = query;
  odd_query.offset += 1;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, odd_query, key, value, gate, cache, output, 0));
      },
      "MIVE-16 accepted unaligned query offset");

  auto odd_key = key;
  odd_key.offset += 1;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, odd_key, value, gate, cache, output, 0));
      },
      "MIVE-16 accepted unaligned key offset");

  auto odd_value = value;
  odd_value.offset += 1;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, key, odd_value, gate, cache, output, 0));
      },
      "MIVE-16 accepted unaligned value offset");

  auto odd_gate = gate;
  odd_gate.offset += 1;
  expect_throws<std::invalid_argument>(
      [&] {
        static_cast<void>(omlx::m5::execute_paged_attention(
            context, plan, query, key, value, odd_gate, cache, output, 0));
      },
      "MIVE-16 accepted unaligned gate offset");

  expect(cache.tokens() == 0 && wrong_page_tokens_cache.tokens() == 0 &&
             wrong_capacity_cache.tokens() == 0,
         "MIVE-16 rejected calls mutated cache token frontier");

  const auto sub = omlx::m5::execute_paged_attention(
      context, plan, query, key, value, gate, cache, output, 0);
  expect(sub.dispatch_count == 2 && cache.tokens() == tokens,
         "MIVE-16 valid paged attention invocation failed after validation checks");
}

} // namespace

int main() {
  try {
    run_qk_norm_rope();
    run_paged_attention();
    run_paged_attention_validation();
    std::cout << "B3.4 q/k norm + RoPE tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "B3.4 attention postprocess failure: " << error.what() << '\n';
    return 1;
  }
}
