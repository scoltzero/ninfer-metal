#include "omlx_m5/memory_plan.h"

#include <limits>
#include <stdexcept>

namespace omlx::m5 {

namespace {

std::uint64_t checked_mul(std::uint64_t left, std::uint64_t right) {
  if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right) {
    throw std::overflow_error("memory plan overflow");
  }
  return left * right;
}

std::uint64_t checked_add(std::uint64_t left, std::uint64_t right) {
  if (left > std::numeric_limits<std::uint64_t>::max() - right) {
    throw std::overflow_error("memory plan overflow");
  }
  return left + right;
}

} // namespace

SequenceMemoryPlan plan_sequence_memory(const QwenModelConfig &config,
                                        std::uint64_t max_context,
                                        std::uint64_t kv_page_tokens) {
  if (max_context == 0 || kv_page_tokens == 0) {
    throw std::invalid_argument("context and KV page size must be positive");
  }
  if (config.full_attention_layers() <= 0 ||
      config.linear_attention_layers() <= 0 ||
      config.num_key_value_heads <= 0 || config.head_dim <= 0 ||
      config.linear_num_key_heads <= 0 || config.linear_num_value_heads <= 0 ||
      config.linear_key_head_dim <= 0 || config.linear_value_head_dim <= 0 ||
      config.linear_conv_kernel_dim <= 0) {
    throw std::invalid_argument("model geometry is incomplete");
  }
  SequenceMemoryPlan plan;
  plan.max_context = max_context;
  plan.kv_page_tokens = kv_page_tokens;
  plan.kv_page_count = (max_context + kv_page_tokens - 1) / kv_page_tokens;

  constexpr std::uint64_t bf16_bytes = 2;
  constexpr std::uint64_t fp32_bytes = 4;
  plan.kv_bytes_per_token = checked_mul(
      static_cast<std::uint64_t>(config.full_attention_layers()),
      checked_mul(2, checked_mul(config.num_key_value_heads,
                                 checked_mul(config.head_dim, bf16_bytes))));
  plan.kv_bytes_per_page = checked_mul(plan.kv_bytes_per_token, kv_page_tokens);
  plan.kv_total_bytes = checked_mul(plan.kv_bytes_per_page, plan.kv_page_count);

  plan.gdn_state_bytes = checked_mul(
      static_cast<std::uint64_t>(config.linear_attention_layers()),
      checked_mul(
          config.linear_num_value_heads,
          checked_mul(config.linear_value_head_dim,
                      checked_mul(config.linear_key_head_dim, fp32_bytes))));

  const auto qkv_width = checked_add(
      checked_mul(2, checked_mul(config.linear_num_key_heads,
                                 config.linear_key_head_dim)),
      checked_mul(config.linear_num_value_heads, config.linear_value_head_dim));
  plan.gdn_conv_history_bytes = checked_mul(
      static_cast<std::uint64_t>(config.linear_attention_layers()),
      checked_mul(qkv_width,
                  checked_mul(config.linear_conv_kernel_dim - 1, bf16_bytes)));
  plan.total_bytes = checked_add(
      plan.kv_total_bytes,
      checked_add(plan.gdn_state_bytes, plan.gdn_conv_history_bytes));
  return plan;
}

} // namespace omlx::m5
