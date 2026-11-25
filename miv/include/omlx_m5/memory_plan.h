#pragma once

#include <cstdint>

#include "omlx_m5/model_config.h"

namespace omlx::m5 {

struct SequenceMemoryPlan {
  std::uint64_t max_context = 0;
  std::uint64_t kv_page_tokens = 64;
  std::uint64_t kv_page_count = 0;
  std::uint64_t kv_bytes_per_token = 0;
  std::uint64_t kv_bytes_per_page = 0;
  std::uint64_t kv_total_bytes = 0;
  std::uint64_t gdn_state_bytes = 0;
  std::uint64_t gdn_conv_history_bytes = 0;
  std::uint64_t total_bytes = 0;
};

[[nodiscard]] SequenceMemoryPlan
plan_sequence_memory(const QwenModelConfig &config, std::uint64_t max_context,
                     std::uint64_t kv_page_tokens = 64);

} // namespace omlx::m5
