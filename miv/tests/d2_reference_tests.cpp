#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/program.h"
#include "omlx_m5/reference.h"

namespace {

void expect(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void expect_close(float left, float right, float tolerance,
                  const std::string &message) {
  if (!std::isfinite(left) || !std::isfinite(right) ||
      !(std::abs(left - right) <= tolerance)) {
    throw std::runtime_error(message + ": " + std::to_string(left) +
                             " vs " + std::to_string(right));
  }
}

bool bitwise_equal(const std::vector<float> &left,
                   const std::vector<float> &right) {
  return left.size() == right.size() &&
         std::memcmp(left.data(), right.data(), left.size() * sizeof(float)) ==
             0;
}

std::vector<std::byte> pack_codes(const std::vector<std::uint32_t> &codes,
                                  std::uint32_t bits) {
  std::vector<std::byte> packed((codes.size() * bits + 7) / 8);
  for (std::size_t index = 0; index < codes.size(); ++index) {
    for (std::uint32_t bit = 0; bit < bits; ++bit) {
      if ((codes[index] & (1U << bit)) != 0U) {
        const auto absolute = index * bits + bit;
        packed[absolute / 8] |=
            static_cast<std::byte>(1U << (absolute % 8));
      }
    }
  }
  return packed;
}

struct QuantFixture {
  omlx::m5::TensorInfo weight_info;
  omlx::m5::TensorInfo scale_info;
  omlx::m5::TensorInfo bias_info;
  std::vector<std::byte> weights;
  std::vector<std::uint16_t> scales;
  std::vector<std::uint16_t> biases;
  omlx::m5::QuantizedModule module;

  explicit QuantFixture(std::int32_t bits) {
    constexpr std::uint64_t output = 3;
    constexpr std::uint64_t input = 64;
    std::vector<std::uint32_t> codes;
    codes.reserve(output * input);
    for (std::uint64_t row = 0; row < output; ++row) {
      for (std::uint64_t column = 0; column < input; ++column) {
        codes.push_back(static_cast<std::uint32_t>(
            (column * (row + 1) + row) % (1U << bits)));
      }
    }
    weights = pack_codes(codes, bits);
    for (std::size_t row = 0; row < output; ++row) {
      scales.push_back(omlx::m5::float_to_bf16(0.125F * (row + 1)));
      biases.push_back(omlx::m5::float_to_bf16(-0.25F * row));
    }
    weight_info = {.name = "weight",
                   .dtype = omlx::m5::TensorDType::u32,
                   .shape = {output,
                             input * static_cast<std::uint64_t>(bits) / 32},
                   .nbytes = weights.size()};
    scale_info = {.name = "scales",
                  .dtype = omlx::m5::TensorDType::bf16,
                  .shape = {output, 1},
                  .nbytes = scales.size() * sizeof(std::uint16_t)};
    bias_info = {.name = "biases",
                 .dtype = omlx::m5::TensorDType::bf16,
                 .shape = {output, 1},
                 .nbytes = biases.size() * sizeof(std::uint16_t)};
    module = {
        .name = "fixture",
        .bits = bits,
        .group_size = 64,
        .out_features = output,
        .in_features = input,
        .weights = {.data = weights.data(),
                    .nbytes = weights.size(),
                    .info = &weight_info},
        .scales = {.data = reinterpret_cast<const std::byte *>(scales.data()),
                   .nbytes = scales.size() * sizeof(std::uint16_t),
                   .info = &scale_info},
        .biases = {.data = reinterpret_cast<const std::byte *>(biases.data()),
                   .nbytes = biases.size() * sizeof(std::uint16_t),
                   .info = &bias_info},
    };
  }
};

void test_quantized_qmv_qmm() {
  std::vector<float> input(64);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(static_cast<int>(index % 9) - 4) * 0.1F;
  }
  for (const auto bits : {4, 5}) {
    QuantFixture fixture(bits);
    const auto qmv = omlx::m5::quantized_qmv(fixture.module, input);
    for (std::size_t row = 0; row < qmv.size(); ++row) {
      const auto weight =
          omlx::m5::dequantize_row_f64(fixture.module, row, 0, 64);
      double expected = 0.0;
      for (std::size_t column = 0; column < input.size(); ++column) {
        expected = std::fma(static_cast<double>(input[column]), weight[column],
                            expected);
      }
      expect_close(qmv[row], static_cast<float>(expected), 2.0e-5F,
                   "qmv differs from FP64 oracle");
    }
    std::vector<float> matrix(input);
    matrix.insert(matrix.end(), input.rbegin(), input.rend());
    const auto qmm = omlx::m5::quantized_qmm(fixture.module, matrix, 2);
    expect(qmm.size() == 6, "qmm output geometry mismatch");
    expect(std::equal(qmv.begin(), qmv.end(), qmm.begin()),
           "qmm first row differs from qmv");
  }
}

void test_elementwise_norm_rope_rng() {
  const std::vector<float> input = {3, 4, 0, 5};
  const std::vector<float> weight = {2, 0.5F};
  const auto normalized = omlx::m5::rms_norm(input, 2, 2, weight, 1.0e-6F);
  expect_close(normalized[0], 3.0F / std::sqrt(12.5F) * 2.0F, 1.0e-6F,
               "RMSNorm first element mismatch");
  expect_close(normalized[3], 5.0F / std::sqrt(12.5F) * 0.5F, 1.0e-6F,
               "RMSNorm last element mismatch");

  std::vector<float> rope = {1, 0, 2, 3, 1, 0, 2, 3};
  omlx::m5::apply_rope(rope, 2, 1, 4, 2, 0, 10000.0);
  expect(rope[0] == 1.0F && rope[1] == 0.0F,
         "RoPE position zero must be identity");
  expect_close(rope[4], std::cos(1.0F), 1.0e-6F,
               "RoPE cosine mismatch");
  expect_close(rope[5], std::sin(1.0F), 1.0e-6F,
               "RoPE sine mismatch");
  expect(rope[6] == 2.0F && rope[7] == 3.0F,
         "RoPE rotated dimensions beyond rotary_dim");

  const auto activated =
      omlx::m5::silu_multiply({input.data(), 2}, {weight.data(), 2});
  expect_close(activated[0], 3.0F / (1.0F + std::exp(-3.0F)) * 2.0F,
               1.0e-6F, "SiLU multiply mismatch");

  const omlx::m5::CounterRng rng(99);
  expect(rng.bits(3, 7) == rng.bits(3, 7), "counter RNG is not repeatable");
  expect(rng.bits(3, 7) != rng.bits(4, 7), "counter RNG ignored stream");
  const std::vector<float> logits = {0.0F, 1.0F, 3.0F, 2.0F};
  expect(omlx::m5::sample_logits(logits, {}, rng, 0, 0) == 2,
         "greedy sampling mismatch");
  const omlx::m5::SamplingOptions sampled{
      .temperature = 0.8F,
      .top_k = 2,
      .top_p = 0.9F,
  };
  const auto token = omlx::m5::sample_logits(logits, sampled, rng, 5, 11);
  expect(token == omlx::m5::sample_logits(logits, sampled, rng, 5, 11),
         "sampling is not counter deterministic");
  expect(token == 2 || token == 3, "top-k sampling escaped retained set");

  // Regression tests for MIVE-9: tiny positive temperature and all -inf logits
  // must not overflow to inf-inf=NaN and fall through to the worst candidate.
  const std::vector<float> extreme_logits = {10.0F, -10.0F};
  const omlx::m5::SamplingOptions tiny_temp{
      .temperature = 1.0e-38F,
      .top_k = 0,
      .top_p = 1.0F,
  };
  for (std::uint64_t pos = 0; pos < 8; ++pos) {
    expect(omlx::m5::sample_logits(extreme_logits, tiny_temp, rng, 1, pos) == 0,
           "MIVE-9: tiny positive temperature selected worst token");
  }
  const std::vector<float> all_neg_inf = {-INFINITY, -INFINITY, -INFINITY};
  const omlx::m5::SamplingOptions unit_temp{
      .temperature = 1.0F,
      .top_k = 0,
      .top_p = 1.0F,
  };
  expect(omlx::m5::sample_logits(all_neg_inf, unit_temp, rng, 2, 0) == 0,
         "MIVE-9: all -inf logits did not fall back to indices.front()");
  const std::vector<float> mixed_neg_inf = {0.0F, 5.0F, -INFINITY};
  const omlx::m5::SamplingOptions tiny_top_p{
      .temperature = 1.0e-38F,
      .top_k = 0,
      .top_p = 0.9F,
  };
  expect(omlx::m5::sample_logits(mixed_neg_inf, tiny_top_p, rng, 3, 0) == 1,
         "MIVE-9: mixed -inf with tiny temperature selected wrong token");

  // Regression check for TEST-5: expect_close must reject NaN.
  bool nan_rejected = false;
  try {
    expect_close(std::numeric_limits<float>::quiet_NaN(), 0.0F, 1.0F, "nan");
  } catch (const std::runtime_error &) {
    nan_rejected = true;
  }
  expect(nan_rejected, "TEST-5: expect_close failed to reject NaN");
}

void test_kv_attention_and_gdn_chunking() {
  constexpr std::size_t positions = 5;
  constexpr std::size_t query_heads = 4;
  constexpr std::size_t kv_heads = 2;
  constexpr std::size_t dim = 3;
  std::vector<float> queries(positions * query_heads * dim);
  std::vector<float> keys(positions * kv_heads * dim);
  std::vector<float> values(keys.size());
  for (std::size_t index = 0; index < queries.size(); ++index) {
    queries[index] = static_cast<float>(static_cast<int>(index % 7) - 3) *
                     0.125F;
  }
  for (std::size_t index = 0; index < keys.size(); ++index) {
    keys[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.2F;
    values[index] = static_cast<float>(index % 11) * 0.1F;
  }
  omlx::m5::KvCache cache_a(kv_heads, dim, 8);
  omlx::m5::KvCache cache_b(kv_heads, dim, 8);
  const auto attention_a = omlx::m5::attention_prefill(
      queries, keys, values, positions, query_heads, cache_a, 1);
  const auto attention_b = omlx::m5::attention_prefill(
      queries, keys, values, positions, query_heads, cache_b, 3);
  expect(bitwise_equal(attention_a, attention_b),
         "attention prefill changed with chunk size");
  expect(cache_a.tokens() == positions, "attention KV write count mismatch");
  cache_a.truncate(3);
  expect(cache_a.tokens() == 3, "KV truncate failed");
  cache_a.clear();
  expect(cache_a.tokens() == 0, "KV clear failed");

  const omlx::m5::GdnReferenceGeometry geometry{
      .key_heads = 2,
      .value_heads = 4,
      .key_dim = 3,
      .value_dim = 2,
  };
  std::vector<float> gq(positions * geometry.key_heads * geometry.key_dim);
  std::vector<float> gk(gq.size());
  std::vector<float> gv(positions * geometry.value_heads * geometry.value_dim);
  std::vector<float> decay(positions * geometry.value_heads);
  std::vector<float> beta(decay.size());
  for (std::size_t index = 0; index < gq.size(); ++index) {
    gq[index] = static_cast<float>(index % 7) * 0.03F;
    gk[index] = static_cast<float>(static_cast<int>(index % 5) - 2) * 0.04F;
  }
  for (std::size_t index = 0; index < gv.size(); ++index) {
    gv[index] = static_cast<float>(index % 9) * 0.02F;
  }
  for (std::size_t index = 0; index < decay.size(); ++index) {
    decay[index] = 0.90F + static_cast<float>(index % 3) * 0.01F;
    beta[index] = 0.2F + static_cast<float>(index % 4) * 0.05F;
  }
  omlx::m5::GdnReferenceState state_a(geometry);
  omlx::m5::GdnReferenceState state_b(geometry);
  const auto gdn_a = omlx::m5::gdn_prefill(
      gq, gk, gv, decay, beta, positions, state_a, 1);
  const auto gdn_b = omlx::m5::gdn_prefill(
      gq, gk, gv, decay, beta, positions, state_b, 4);
  expect(bitwise_equal(gdn_a, gdn_b),
         "GDN prefill changed with chunk size");
  expect(bitwise_equal(state_a.values(), state_b.values()),
         "GDN final state changed with chunk size");
}

void test_depthwise_conv_chunking() {
  constexpr std::size_t positions = 7;
  constexpr std::size_t width = 3;
  constexpr std::size_t kernel_size = 4;
  std::vector<float> input(positions * width);
  std::vector<float> kernel(kernel_size * width);
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = static_cast<float>(index + 1) * 0.01F;
  }
  for (std::size_t index = 0; index < kernel.size(); ++index) {
    kernel[index] = static_cast<float>(static_cast<int>(index % 5) - 2) *
                    0.1F;
  }
  std::vector<float> all_history((kernel_size - 1) * width, 0.0F);
  const auto all = omlx::m5::causal_depthwise_conv_silu(
      input, positions, width, kernel, kernel_size, all_history);
  std::vector<float> chunk_history((kernel_size - 1) * width, 0.0F);
  std::vector<float> chunked;
  for (std::size_t position = 0; position < positions; ++position) {
    const auto current = omlx::m5::causal_depthwise_conv_silu(
        std::span<const float>(input).subspan(position * width, width), 1,
        width, kernel, kernel_size, chunk_history);
    chunked.insert(chunked.end(), current.begin(), current.end());
  }
  expect(bitwise_equal(all, chunked),
         "depthwise convolution changed with chunk size");
  expect(bitwise_equal(all_history, chunk_history),
         "depthwise convolution history changed with chunk size");
}

omlx::m5::ReferenceProgramConfig tiny_config() {
  return {
      .vocab_size = 64,
      .hidden_size = 16,
      .intermediate_size = 32,
      .attention_heads = 2,
      .kv_heads = 1,
      .head_dim = 8,
      .linear_key_heads = 1,
      .linear_value_heads = 2,
      .linear_key_dim = 4,
      .linear_value_dim = 4,
      .linear_conv_kernel = 4,
      .max_context = 320,
      .rotary_dim = 4,
      .rms_norm_eps = 1.0e-6F,
      .rope_theta = 10000.0,
      .eos_token_id = -1,
      .layer_types = {omlx::m5::LayerType::linear_attention,
                      omlx::m5::LayerType::full_attention},
  };
}

void test_program_runner() {
  const std::vector<std::int32_t> prompt = {1, 2, 3, 4, 5, 6, 7};
  auto program_a = omlx::m5::ReferenceProgram::create_deterministic(
      tiny_config(), 20260919);
  auto program_b = omlx::m5::ReferenceProgram::create_deterministic(
      tiny_config(), 20260919);
  const auto logits_a = program_a.prefill(prompt, 1);
  const auto logits_b = program_b.prefill(prompt, 4);
  expect(bitwise_equal(logits_a, logits_b),
         "Program prefill changed with chunk size");
  expect(program_a.position() == prompt.size(),
         "Program prefill frontier mismatch");
  expect(program_a.parameter_count() > 0,
         "Program parameter accounting is empty");

  auto runner = omlx::m5::ReferenceProgram::create_deterministic(
      tiny_config(), 20260919);
  const auto generated = runner.generate(prompt, 3, 256, {}, 17);
  expect(generated.tokens.size() == 256,
         "small-model runner did not complete 256 decode tokens");
  expect(!generated.cancelled && !generated.reached_eos,
         "small-model runner ended unexpectedly");
  expect(runner.position() == prompt.size() + 255,
         "runner frontier does not reflect fed tokens");

  runner.reset();
  std::size_t polls = 0;
  const auto cancelled = runner.generate(
      prompt, 2, 20, {}, 17, [&] { return polls++ == 3; });
  expect(cancelled.cancelled && cancelled.tokens.size() == 3,
         "Program cancellation boundary mismatch");
  runner.reset();
  expect(runner.position() == 0,
         "Program reset did not release sequence state");
}

} // namespace

int main() {
  try {
    test_quantized_qmv_qmm();
    test_elementwise_norm_rope_rng();
    test_kv_attention_and_gdn_chunking();
    test_depthwise_conv_chunking();
    test_program_runner();
  } catch (const std::exception &error) {
    std::cerr << "D2 reference test failed: " << error.what() << '\n';
    return 1;
  }
  std::cout << "D2 reference tests passed\n";
  return 0;
}
