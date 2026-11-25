#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "../src/forward_plan.h"
#include "omlx_m5/model_config.h"
#include "omlx_m5/quantization.h"

namespace {
using namespace omlx::m5;
using namespace omlx::m5::detail::forward;

U32 pattern = 0;
float represented(std::uint16_t bits) {
  return std::bit_cast<float>(std::uint32_t(bits) << 16);
}
std::uint16_t sample(std::uint64_t position, std::uint32_t head,
                     std::uint32_t feature, std::uint32_t salt) {
  const auto code =
      std::int32_t((position * 17 + head * 31 + feature * 7 + salt) % 257) -
      128;
  if (pattern == 1)
    return float_to_bf16(float((head + feature + salt) % 2 ? 1 : -1) / 32.0f);
  if (pattern == 2)
    return float_to_bf16(float(code) / 128.0f);
  if (pattern == 3)
    return float_to_bf16(
        float(std::int32_t((position * 3 + feature * 11 + salt) % 17) - 8) /
        64.0f);
  return float_to_bf16(float(code) / 512.0f);
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  return (values[values.size() / 2] + values[(values.size() - 1) / 2]) / 2;
}

nlohmann::json benchmark(const MetalContext &context,
                         const QwenModelConfig &config,
                         const std::filesystem::path &libraries,
                         std::uint32_t history, std::uint32_t tokens,
                         std::uint32_t repeats) {
  const U32 qh = U32(config.num_attention_heads);
  const U32 kvh = U32(config.num_key_value_heads);
  const U32 hd = U32(config.head_dim);
  if (!qh || !kvh || qh % kvh || hd == 0 || hd > 256 || hd % 32)
    throw std::invalid_argument("unsupported attention geometry");
  const U32 kv_width = kvh * hd, query_stride = qh * hd * 2;
  std::vector<std::uint16_t> query(U64(tokens) * query_stride);
  std::vector<std::uint16_t> history_keys(U64(history + tokens) * kv_width);
  std::vector<std::uint16_t> history_values(history_keys.size());
  for (U32 pos = 0; pos < history + tokens; ++pos)
    for (U32 head = 0; head < kvh; ++head)
      for (U32 feature = 0; feature < hd; ++feature) {
        const U64 index = (U64(pos) * kvh + head) * hd + feature;
        history_keys[index] = sample(pos, head, feature, 11);
        history_values[index] = sample(pos, head, feature, 67);
      }
  for (U32 token = 0; token < tokens; ++token)
    for (U32 head = 0; head < qh; ++head)
      for (U32 feature = 0; feature < hd; ++feature) {
        const U64 index = (U64(token) * qh + head) * 2 * hd;
        query[index + feature] = sample(history + token, head, feature, 113);
        query[index + hd + feature] =
            sample(history + token, head, feature, 179);
      }
  const auto query_buffer = context.upload(std::as_bytes(std::span(query)));
  const U32 pages = (history + tokens + 63) / 64;
  const bool permuted = pattern == 4;
  const U64 physical_tokens = permuted ? U64(pages) * 64 : history + tokens;
  auto cache_key = context.make_buffer(physical_tokens * kv_width * 2);
  auto cache_value = context.make_buffer(physical_tokens * kv_width * 2);
  std::vector<U32> mapping(pages);
  std::iota(mapping.begin(), mapping.end(), 0);
  if (permuted)
    std::reverse(mapping.begin(), mapping.end());
  const auto page_table = context.upload(std::as_bytes(std::span(mapping)));
  std::optional<MetalBuffer> legacy_key_storage, legacy_value_storage;
  const MetalBuffer *legacy_key = &cache_key, *legacy_value = &cache_value;
  if (permuted) {
    std::vector<std::uint16_t> mapped_keys(physical_tokens * kv_width, 0x7f7f);
    std::vector<std::uint16_t> mapped_values(mapped_keys.size(), 0x7f7f);
    for (U32 logical = 0; logical < history; ++logical) {
      const U64 physical = U64(mapping[logical / 64]) * 64 + logical % 64;
      std::copy_n(history_keys.begin() + U64(logical) * kv_width, kv_width,
                  mapped_keys.begin() + physical * kv_width);
      std::copy_n(history_values.begin() + U64(logical) * kv_width, kv_width,
                  mapped_values.begin() + physical * kv_width);
    }
    cache_key.write(std::as_bytes(std::span(mapped_keys)));
    cache_value.write(std::as_bytes(std::span(mapped_values)));
    legacy_key_storage.emplace(context.make_buffer(history_keys.size() * 2));
    legacy_value_storage.emplace(
        context.make_buffer(history_values.size() * 2));
    legacy_key_storage->write(
        std::as_bytes(std::span(history_keys).first(U64(history) * kv_width)));
    legacy_value_storage->write(std::as_bytes(
        std::span(history_values).first(U64(history) * kv_width)));
    legacy_key = &*legacy_key_storage;
    legacy_value = &*legacy_value_storage;
  } else {
    cache_key.write(
        std::as_bytes(std::span(history_keys).first(U64(history) * kv_width)));
    cache_value.write(std::as_bytes(
        std::span(history_values).first(U64(history) * kv_width)));
  }
  const auto next_key = context.upload(
      std::as_bytes(std::span(history_keys).subspan(U64(history) * kv_width)));
  const auto next_value = context.upload(std::as_bytes(
      std::span(history_values).subspan(U64(history) * kv_width)));
  auto output = context.make_buffer(U64(tokens) * qh * hd * 2);
  auto legacy_output = context.make_buffer(output.size());
  const std::array<U32, 14> parameters{
      qh,           kvh,   hd,           tokens, history,  history + tokens,
      64,           pages, query_stride, hd * 2, kv_width, kv_width,
      query_stride, hd * 2};
  const auto param_span = std::as_bytes(std::span(parameters));
  const std::vector<std::byte> param_bytes(param_span.begin(),
                                           param_span.end());
  Encoder encoder(context, libraries, tokens,
                  "qwen.long_kv.h" + std::to_string(history));
  encoder.begin_stage("kv_append");
  auto &append =
      encoder.emit("attention_postprocess", "omlx_m5_paged_kv_append",
                   {.width = U64(tokens) * kv_width}, {.width = 128},
                   {{&next_key, 0, next_key.size()},
                    {&next_value, 0, next_value.size()},
                    {&cache_key, 0, cache_key.size()},
                    {&cache_value, 0, cache_value.size()},
                    {&page_table, 0, page_table.size()}});
  encoder.dynamic(append, 5, param_bytes);
  encoder.begin_stage("attention");
  auto &attention = encoder.emit(
      "attention_postprocess", "omlx_m5_paged_causal_attention",
      {.width = 32, .height = U64(tokens) * qh}, {.width = 32},
      {{&query_buffer, 0, query_buffer.size()},
       {&query_buffer, U64(hd) * 2, query_buffer.size() - U64(hd) * 2},
       {&cache_key, 0, cache_key.size()},
       {&cache_value, 0, cache_value.size()},
       {&output, 0, output.size()},
       {&page_table, 0, page_table.size()}});
  encoder.dynamic(attention, 6, param_bytes);
  auto profile = encoder.finish();
  const std::array calls = {
      MetalCommandInvocation{.plan = &profile.plan,
                             .bindings = profile.bindings,
                             .parameters = &profile.parameters}};
  Encoder legacy_encoder(context, libraries, tokens,
                         "qwen.long_kv.contiguous_control.h" +
                             std::to_string(history));
  auto &legacy_append = legacy_encoder.emit(
      "attention_contiguous_control", "omlx_m5_contiguous_kv_append_control",
      {.width = U64(tokens) * kv_width}, {.width = 128},
      {{&next_key, 0, next_key.size()},
       {&next_value, 0, next_value.size()},
       {legacy_key, 0, legacy_key->size()},
       {legacy_value, 0, legacy_value->size()}});
  legacy_encoder.dynamic(legacy_append, 4, param_bytes);
  auto &legacy_attention = legacy_encoder.emit(
      "attention_contiguous_control", "omlx_m5_contiguous_attention_control",
      {.width = 32, .height = U64(tokens) * qh}, {.width = 32},
      {{&query_buffer, 0, query_buffer.size()},
       {&query_buffer, U64(hd) * 2, query_buffer.size() - U64(hd) * 2},
       {legacy_key, 0, legacy_key->size()},
       {legacy_value, 0, legacy_value->size()},
       {&legacy_output, 0, legacy_output.size()}});
  legacy_encoder.dynamic(legacy_attention, 5, param_bytes);
  auto legacy_profile = legacy_encoder.finish();
  const std::array legacy_calls = {
      MetalCommandInvocation{.plan = &legacy_profile.plan,
                             .bindings = legacy_profile.bindings,
                             .parameters = &legacy_profile.parameters}};
  const U32 split_tokens = 512;
  const U32 splits = (history + tokens + split_tokens - 1) / split_tokens;
  const U64 partial_count = U64(tokens) * qh * splits;
  auto partial_max = context.make_buffer(partial_count * 4);
  auto partial_sum = context.make_buffer(partial_count * 4);
  auto partial_output = context.make_buffer(partial_count * hd * 4);
  auto split_output = context.make_buffer(output.size());
  std::array<U32, 16> split_parameters{};
  std::copy(parameters.begin(), parameters.end(), split_parameters.begin());
  split_parameters[14] = splits;
  split_parameters[15] = split_tokens;
  const auto split_span = std::as_bytes(std::span(split_parameters));
  const std::vector<std::byte> split_bytes(split_span.begin(),
                                           split_span.end());
  Encoder split_encoder(context, libraries, tokens,
                        "qwen.long_kv.split.h" + std::to_string(history));
  split_encoder.begin_stage("kv_append");
  auto &split_append =
      split_encoder.emit("attention_postprocess", "omlx_m5_paged_kv_append",
                         {.width = U64(tokens) * kv_width}, {.width = 128},
                         {{&next_key, 0, next_key.size()},
                          {&next_value, 0, next_value.size()},
                          {&cache_key, 0, cache_key.size()},
                          {&cache_value, 0, cache_value.size()},
                          {&page_table, 0, page_table.size()}});
  split_encoder.dynamic(split_append, 5, split_bytes);
  split_encoder.begin_stage("split_partial");
  auto &partial = split_encoder.emit(
      "attention_split_kv", "omlx_m5_paged_attention_split_partial",
      {.width = U64(splits) * 32, .height = U64(tokens) * qh}, {.width = 32},
      {{&query_buffer, 0, query_buffer.size()},
       {&cache_key, 0, cache_key.size()},
       {&cache_value, 0, cache_value.size()},
       {&partial_max, 0, partial_max.size()},
       {&partial_sum, 0, partial_sum.size()},
       {&partial_output, 0, partial_output.size()},
       {&page_table, 0, page_table.size()}});
  split_encoder.dynamic(partial, 7, split_bytes);
  split_encoder.begin_stage("split_merge");
  auto &merge = split_encoder.emit(
      "attention_split_kv", "omlx_m5_paged_attention_split_merge",
      {.width = 32, .height = U64(tokens) * qh}, {.width = 32},
      {{&query_buffer, U64(hd) * 2, query_buffer.size() - U64(hd) * 2},
       {&partial_max, 0, partial_max.size()},
       {&partial_sum, 0, partial_sum.size()},
       {&partial_output, 0, partial_output.size()},
       {&split_output, 0, split_output.size()}});
  split_encoder.dynamic(merge, 5, split_bytes);
  auto split_profile = split_encoder.finish();
  const std::array split_calls = {
      MetalCommandInvocation{.plan = &split_profile.plan,
                             .bindings = split_profile.bindings,
                             .parameters = &split_profile.parameters}};
  auto ungated = context.make_buffer(U64(tokens) * qh * hd * 4);
  auto unfused_output = context.make_buffer(output.size());
  Encoder unfused_encoder(context, libraries, tokens,
                          "qwen.long_kv.unfused_gate.h" +
                              std::to_string(history));
  auto &unfused_append =
      unfused_encoder.emit("attention_postprocess", "omlx_m5_paged_kv_append",
                           {.width = U64(tokens) * kv_width}, {.width = 128},
                           {{&next_key, 0, next_key.size()},
                            {&next_value, 0, next_value.size()},
                            {&cache_key, 0, cache_key.size()},
                            {&cache_value, 0, cache_value.size()},
                            {&page_table, 0, page_table.size()}});
  unfused_encoder.dynamic(unfused_append, 5, split_bytes);
  auto &unfused_partial = unfused_encoder.emit(
      "attention_split_kv", "omlx_m5_paged_attention_split_partial",
      {.width = U64(splits) * 32, .height = U64(tokens) * qh}, {.width = 32},
      {{&query_buffer, 0, query_buffer.size()},
       {&cache_key, 0, cache_key.size()},
       {&cache_value, 0, cache_value.size()},
       {&partial_max, 0, partial_max.size()},
       {&partial_sum, 0, partial_sum.size()},
       {&partial_output, 0, partial_output.size()},
       {&page_table, 0, page_table.size()}});
  unfused_encoder.dynamic(unfused_partial, 7, split_bytes);
  auto &unfused_merge = unfused_encoder.emit(
      "attention_split_kv", "omlx_m5_paged_attention_split_merge_ungated",
      {.width = 32, .height = U64(tokens) * qh}, {.width = 32},
      {{&partial_max, 0, partial_max.size()},
       {&partial_sum, 0, partial_sum.size()},
       {&partial_output, 0, partial_output.size()},
       {&ungated, 0, ungated.size()}});
  unfused_encoder.dynamic(unfused_merge, 4, split_bytes);
  auto &unfused_gate = unfused_encoder.emit(
      "attention_split_kv", "omlx_m5_attention_apply_gate_fp32",
      {.width = U64(tokens) * qh * hd}, {.width = 128},
      {{&ungated, 0, ungated.size()},
       {&query_buffer, U64(hd) * 2, query_buffer.size() - U64(hd) * 2},
       {&unfused_output, 0, unfused_output.size()}});
  unfused_encoder.dynamic(unfused_gate, 3, split_bytes);
  auto unfused_profile = unfused_encoder.finish();
  const std::array unfused_calls = {
      MetalCommandInvocation{.plan = &unfused_profile.plan,
                             .bindings = unfused_profile.bindings,
                             .parameters = &unfused_profile.parameters}};
  std::vector<double> full_gpu, full_wall, split_gpu, split_wall;
  std::vector<double> unfused_gpu, unfused_wall;
  std::vector<double> legacy_gpu, legacy_wall;
  std::vector<double> attention_gpu, attention_wall;
  for (U32 i = 0; i < repeats + 3; ++i) {
    MetalSubmission baseline_submission, legacy_submission;
    if (i % 2 == 0) {
      baseline_submission = context.execute_batch(calls);
      legacy_submission = context.execute_batch(legacy_calls);
    } else {
      legacy_submission = context.execute_batch(legacy_calls);
      baseline_submission = context.execute_batch(calls);
    }
    MetalSubmission split_submission, unfused_submission;
    if (i % 2 == 0) {
      split_submission = context.execute_batch(split_calls);
      unfused_submission = context.execute_batch(unfused_calls);
    } else {
      unfused_submission = context.execute_batch(unfused_calls);
      split_submission = context.execute_batch(split_calls);
    }
    if (i >= 3) {
      full_gpu.push_back(baseline_submission.timing.gpu_ms);
      full_wall.push_back(baseline_submission.timing.wall_ms);
      legacy_gpu.push_back(legacy_submission.timing.gpu_ms);
      legacy_wall.push_back(legacy_submission.timing.wall_ms);
      split_gpu.push_back(split_submission.timing.gpu_ms);
      split_wall.push_back(split_submission.timing.wall_ms);
      unfused_gpu.push_back(unfused_submission.timing.gpu_ms);
      unfused_wall.push_back(unfused_submission.timing.wall_ms);
    }
  }
  // Verify both routes appended exactly the same KV payload; readback is
  // diagnostic-only and excluded from GPU timing.
  for (const U32 position : {0U, history - 1, history, history + tokens - 1}) {
    const U64 physical = U64(mapping[position / 64]) * 64 + position % 64;
    const U64 offset = physical * kv_width * 2;
    const U64 bytes = U64(kv_width) * 2;
    const auto expected_key = std::as_bytes(
        std::span(history_keys).subspan(U64(position) * kv_width, kv_width));
    const auto expected_value = std::as_bytes(
        std::span(history_values).subspan(U64(position) * kv_width, kv_width));
    const auto actual_key = cache_key.read(offset, bytes);
    const auto actual_value = cache_value.read(offset, bytes);
    if (!std::equal(actual_key.begin(), actual_key.end(),
                    expected_key.begin()) ||
        !std::equal(actual_value.begin(), actual_value.end(),
                    expected_value.begin()))
      throw std::runtime_error("long KV cache state differs from seed");
  }
  const auto baseline = output.read(), candidate = split_output.read();
  const bool gate_fusion_bit_exact = unfused_output.read() == candidate;
  if (!gate_fusion_bit_exact)
    throw std::runtime_error(
        "P06 gated split differs from FP32 unfused control");
  const bool contiguous_bit_exact = legacy_output.read() == baseline;
  if (!contiguous_bit_exact)
    throw std::runtime_error(
        "P05 identity page table differs from contiguous control");
  if (baseline.size() != candidate.size())
    throw std::runtime_error("split KV output size changed");
  double candidate_error2 = 0, candidate_scale2 = 0, candidate_max_abs = 0;
  for (U64 i = 0; i < U64(tokens) * qh * hd; ++i) {
    std::uint16_t old_bits = 0, new_bits = 0;
    std::memcpy(&old_bits, baseline.data() + i * 2, 2);
    std::memcpy(&new_bits, candidate.data() + i * 2, 2);
    const double old = represented(old_bits), updated = represented(new_bits);
    if (!std::isfinite(old) || !std::isfinite(updated))
      throw std::runtime_error("split KV nonfinite output");
    candidate_max_abs = std::max(candidate_max_abs, std::abs(old - updated));
    candidate_error2 += (old - updated) * (old - updated);
    candidate_scale2 += old * old;
  }
  const double candidate_relative =
      std::sqrt(candidate_error2 / std::max(candidate_scale2, 1e-12));
  if (candidate_max_abs > 0.015 || candidate_relative > 0.03)
    throw std::runtime_error("split KV candidate differs from baseline");
  bool icb_exact = false, split_icb_exact = false;
  if (context.capabilities().profile.compute_icb_eligible) {
    auto icb = context.prepare_icb_profile(profile.plan.key(), calls.front());
    auto split_icb = context.prepare_icb_profile(split_profile.plan.key(),
                                                 split_calls.front());
    for (U32 i = 0; i < repeats + 3; ++i) {
      const auto submission = context.replay_icb(icb, "attention");
      if (i >= 3) {
        attention_gpu.push_back(submission.timing.gpu_ms);
        attention_wall.push_back(submission.timing.wall_ms);
      }
    }
    icb_exact = output.read() == baseline;
    static_cast<void>(context.replay_icb(split_icb));
    split_icb_exact = split_output.read() == candidate;
    if (!icb_exact || !split_icb_exact)
      throw std::runtime_error("long KV ICB/direct output changed");
  }
  double error2 = 0, scale2 = 0, max_abs = 0;
  U32 checked = 0;
  for (const U32 token : {0U, tokens - 1})
    for (const U32 head : {0U, qh - 1}) {
      const U32 kv_head = head / (qh / kvh);
      const U32 length = history + token + 1;
      std::vector<double> logits(length);
      double maximum = -INFINITY;
      for (U32 pos = 0; pos < length; ++pos) {
        double dot = 0;
        for (U32 feature = 0; feature < hd; ++feature) {
          const U64 qi = (U64(token) * qh + head) * 2 * hd + feature;
          const U64 ki = (U64(pos) * kvh + kv_head) * hd + feature;
          dot += double(represented(query[qi])) * represented(history_keys[ki]);
        }
        logits[pos] = dot / std::sqrt(double(hd));
        maximum = std::max(maximum, logits[pos]);
      }
      double denominator = 0;
      std::vector<double> sums(hd, 0);
      for (U32 pos = 0; pos < length; ++pos) {
        const double probability = std::exp(logits[pos] - maximum);
        denominator += probability;
        for (U32 feature = 0; feature < hd; ++feature)
          sums[feature] +=
              probability *
              represented(
                  history_values[(U64(pos) * kvh + kv_head) * hd + feature]);
      }
      for (U32 feature = 0; feature < hd; ++feature) {
        const U64 qi = (U64(token) * qh + head) * 2 * hd + hd + feature;
        const double gate = represented(query[qi]);
        const double expected =
            sums[feature] / denominator / (1.0 + std::exp(-gate));
        const U64 oi = (U64(token) * qh + head) * hd + feature;
        std::uint16_t bits = 0;
        std::memcpy(&bits, baseline.data() + oi * 2, 2);
        const double diff = represented(bits) - expected;
        if (!std::isfinite(diff))
          throw std::runtime_error("long KV nonfinite output");
        max_abs = std::max(max_abs, std::abs(diff));
        error2 += diff * diff;
        scale2 += expected * expected;
        ++checked;
      }
    }
  const double relative = std::sqrt(error2 / std::max(scale2, 1e-12));
  if (max_abs > 0.015 || relative > 0.03)
    throw std::runtime_error("long KV reference mismatch");
  return {
      {"history_tokens", history},
      {"tokens", tokens},
      {"query_heads", qh},
      {"kv_heads", kvh},
      {"head_dim", hd},
      {"cache_bytes", 2 * cache_key.size()},
      {"dispatches", profile.plan.dispatch_count()},
      {"full_append_attention_gpu_ms", full_gpu},
      {"full_append_attention_wall_ms", full_wall},
      {"full_gpu_median_ms", median(full_gpu)},
      {"legacy_contiguous_gpu_ms", legacy_gpu},
      {"legacy_contiguous_wall_ms", legacy_wall},
      {"identity_table_bit_exact_to_legacy", !permuted && contiguous_bit_exact},
      {"nonmonotonic_table_bit_exact_to_legacy",
       permuted && contiguous_bit_exact},
      {"nonmonotonic_page_table", permuted},
      {"split_count", splits},
      {"split_tokens", split_tokens},
      {"split_workspace_bytes",
       partial_max.size() + partial_sum.size() + partial_output.size()},
      {"split_append_attention_gpu_ms", split_gpu},
      {"split_append_attention_wall_ms", split_wall},
      {"split_gpu_median_ms", median(split_gpu)},
      {"unfused_gate_gpu_ms", unfused_gpu},
      {"unfused_gate_wall_ms", unfused_wall},
      {"unfused_gate_workspace_bytes", ungated.size()},
      {"gate_fusion_bit_exact", gate_fusion_bit_exact},
      {"split_relative_l2_to_baseline", candidate_relative},
      {"split_max_abs_to_baseline", candidate_max_abs},
      {"split_icb_direct_bit_exact", split_icb_exact},
      {"attention_only_icb_gpu_ms", attention_gpu},
      {"attention_only_icb_wall_ms", attention_wall},
      {"attention_gpu_median_ms",
       attention_gpu.empty() ? 0 : median(attention_gpu)},
      {"kv_state_checked", true},
      {"icb_direct_bit_exact", icb_exact},
      {"reference_max_abs", max_abs},
      {"reference_relative_l2", relative},
      {"reference_values_checked", checked}};
}
} // namespace

int main(int argc, char **argv) {
  try {
    if (argc != 4 && argc != 5)
      throw std::invalid_argument("usage: benchmark_long_kv_attention CONFIG "
                                  "LIBRARIES REPEATS [PATTERN=0..4]");
    const auto config = load_qwen_model_config(argv[1]);
    const auto context = MetalContext::create();
    const U32 repeats = U32(std::stoul(argv[3]));
    pattern = argc == 5 ? U32(std::stoul(argv[4])) : 0;
    if (pattern > 4)
      throw std::invalid_argument("unknown long KV input pattern");
    if (repeats < 5)
      throw std::invalid_argument("long KV benchmark requires 5+ repeats");
    nlohmann::json results = nlohmann::json::array();
    for (U32 history : {2048U, 8192U, 32768U})
      for (U32 tokens : {1U, 8U})
        results.push_back(
            benchmark(context, config, argv[2], history, tokens, repeats));
    std::cout << nlohmann::json(
                     {
                         {"schema", "omlx-m5-long-kv-attention-v1"},
                         {"scope",
                          "current production append+attention Metal kernels "
                          "with real Qwen config geometry and deterministic "
                          "synthetic BF16 history; T8 is the attention stage "
                          "of K7 verify, not full target verify"},
                         {"config", argv[1]},
                         {"device", context.capabilities().device_name},
                         {"warmups", 3},
                         {"input_pattern", pattern},
                         {"nonmonotonic_page_table", pattern == 4},
                         {"repeats", repeats},
                         {"results", results},
                     })
                     .dump(2)
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "long KV benchmark failed: " << error.what() << '\n';
    return 1;
  }
}
