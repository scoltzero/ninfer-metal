#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace omlx::m5 {

inline constexpr char kDefaultServingSamplingContract[] = "position_coupled_v1";
inline constexpr std::uint32_t kMaxMtpDraftTokens = 4;
inline constexpr std::uint32_t kMaxDFlash2DraftTokens = 7;
inline constexpr std::int32_t kMaxRepetitionContextSize = 4096;

enum class PrefillLinearMode : std::uint8_t {
  BFloat16 = 0, // "bf16"
  W4A8 = 1,     // "w4a8"
};

enum class PrefillAttentionMode : std::uint8_t {
  ExactNax = 0, // "exact_nax"
  Sage2 = 1,    // "sage2"
};

namespace detail {
[[nodiscard]] constexpr char ascii_lower(char c) noexcept {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

[[nodiscard]] constexpr bool ascii_iequals(std::string_view a,
                                           std::string_view b) noexcept {
  if (a.size() != b.size()) {
    return false;
  }
  for (std::size_t i = 0; i < a.size(); ++i) {
    if (ascii_lower(a[i]) != ascii_lower(b[i])) {
      return false;
    }
  }
  return true;
}
} // namespace detail

[[nodiscard]] constexpr const char *
prefill_linear_mode_to_string(PrefillLinearMode mode) noexcept {
  switch (mode) {
  case PrefillLinearMode::BFloat16:
    return "bf16";
  case PrefillLinearMode::W4A8:
    return "w4a8";
  }
  return "bf16";
}

[[nodiscard]] constexpr const char *
to_string(PrefillLinearMode mode) noexcept {
  return prefill_linear_mode_to_string(mode);
}

[[nodiscard]] constexpr const char *
prefill_attention_mode_to_string(PrefillAttentionMode mode) noexcept {
  switch (mode) {
  case PrefillAttentionMode::ExactNax:
    return "exact_nax";
  case PrefillAttentionMode::Sage2:
    return "sage2";
  }
  return "exact_nax";
}

[[nodiscard]] constexpr const char *
to_string(PrefillAttentionMode mode) noexcept {
  return prefill_attention_mode_to_string(mode);
}

[[nodiscard]] constexpr std::optional<PrefillLinearMode>
try_parse_prefill_linear_mode(std::string_view value) noexcept {
  if (detail::ascii_iequals(value, "bf16")) {
    return PrefillLinearMode::BFloat16;
  }
  if (detail::ascii_iequals(value, "w4a8")) {
    return PrefillLinearMode::W4A8;
  }
  return std::nullopt;
}

[[nodiscard]] inline PrefillLinearMode
parse_prefill_linear_mode(std::string_view value) {
  if (const auto parsed = try_parse_prefill_linear_mode(value)) {
    return *parsed;
  }
  throw std::invalid_argument(
      "prefill_linear_mode must be 'bf16' or 'w4a8', got: " +
      std::string(value));
}

[[nodiscard]] constexpr std::optional<PrefillAttentionMode>
try_parse_prefill_attention_mode(std::string_view value) noexcept {
  if (detail::ascii_iequals(value, "exact_nax")) {
    return PrefillAttentionMode::ExactNax;
  }
  if (detail::ascii_iequals(value, "sage2")) {
    return PrefillAttentionMode::Sage2;
  }
  return std::nullopt;
}

[[nodiscard]] inline PrefillAttentionMode
parse_prefill_attention_mode(std::string_view value) {
  if (const auto parsed = try_parse_prefill_attention_mode(value)) {
    return *parsed;
  }
  throw std::invalid_argument(
      "prefill_attention_mode must be 'exact_nax' or 'sage2', got: " +
      std::string(value));
}

[[nodiscard]] constexpr bool
prefill_sparse_minference_is_effective(PrefillAttentionMode attention_mode,
                                       bool sparse_minference) noexcept {
  return attention_mode == PrefillAttentionMode::Sage2 && sparse_minference;
}

enum class SpeculativeBackend {
  none,
  mtp,
  dflash,
  dflash2,
};

struct SpeculativeOptions {
  SpeculativeBackend backend = SpeculativeBackend::none;
  std::uint32_t draft_tokens = 0;
};

enum class EventKind {
  tokens,
  finished,
  aborted,
  failed,
};

struct LoadOptions {
  std::string model_dir;
  std::string draft_dir;
  std::int32_t max_concurrency = 1;
  std::int32_t max_context = 32768;
  SpeculativeOptions speculative;
  std::string kv_dtype = "bf16";
  std::string prefill_engine = "auto";
  PrefillLinearMode prefill_linear_mode = PrefillLinearMode::BFloat16;
  PrefillAttentionMode prefill_attention_mode = PrefillAttentionMode::ExactNax;
  bool prefill_sparse_minference = false;
  bool prefill_cache_blend = false;
  bool prefill_ane_offload = false;
  std::map<std::string, std::string> path_overrides;
  std::string sampling_contract = "legacy_rejection_v1";
  std::string spec_search = "auto";
  std::uint64_t memory_budget_bytes = 0;
  std::string lora_adapter;
  std::string lora_mode = "merge";
};

struct TrimCacheReport {
  std::uint64_t freed_bytes = 0;
  std::uint64_t resident_bytes = 0;
  std::uint64_t evicted_entries = 0;
};

struct SchedulerHint {
  bool gpu_contended = false;
};

struct SamplingSpec {
  float temperature = 0.0F;
  float top_p = 1.0F;
  float min_p = 0.0F;
  std::int32_t top_k = 0;
  std::uint64_t seed = 0;
  float repetition_penalty = 1.0F;
  float presence_penalty = 0.0F;
  float frequency_penalty = 0.0F;
  std::int32_t repetition_context_size = 20;

  [[nodiscard]] bool has_penalties() const noexcept {
    return repetition_penalty != 1.0F || presence_penalty != 0.0F ||
           frequency_penalty != 0.0F;
  }
};

struct ThinkingBudget {
  std::int32_t budget = 0;
  bool start_in_thinking = false;
  std::vector<std::int32_t> start_token_ids;
  std::vector<std::int32_t> end_token_ids;
  std::vector<std::int32_t> force_token_ids;
};

struct SubmitRequest {
  std::vector<std::int32_t> prompt_tokens;
  std::int32_t max_new_tokens = 1;
  SamplingSpec sampling;
  std::vector<std::vector<std::int32_t>> stop_token_seqs;
  bool pin_prefix = false;
  std::optional<ThinkingBudget> thinking_budget;
};

struct RequestStats {
  std::uint64_t prefix_restored_tokens = 0;
  std::uint64_t spec_rounds = 0;
  std::uint64_t ar_steps = 0;
  std::string decode_mode;
  std::string fallback_reason;
};

struct TokenEvent {
  std::uint64_t request_id = 0;
  std::vector<std::int32_t> tokens;
  EventKind kind = EventKind::tokens;
  std::string finish_reason;
  std::string error;
  std::optional<RequestStats> request_stats;
};

struct CapacityReport {
  std::int32_t max_concurrency = 1;
  std::int32_t active_requests = 0;
  std::int32_t queued_requests = 0;
  bool accepting = true;
};

struct StatsReport {
  std::uint64_t submitted = 0;
  std::uint64_t completed = 0;
  std::uint64_t aborted = 0;
  std::uint64_t failed = 0;
  std::uint64_t emitted_tokens = 0;
  std::string backend_kind;
  std::int32_t max_context = 0;
  std::int32_t max_concurrency = 0;
  std::string kv_dtype;
  std::string speculative_backend;
  std::uint32_t max_draft_tokens = 0;
  // Execution identity is independent of native transport and artifact size.
  // "control_fixture" never qualifies as Qwen model execution.
  std::string model_execution_kind = "unverified";
  std::string sampling_contract_id;
  std::string numeric_contract_id;
  std::string chunk_policy_id;
  std::string execution_identity_id;
  std::string compiled_execution_source_id;
  std::string kernel_bundle_id;
  std::string batch_execution_kind;

  // Cumulative real AR observations. GPU totals require complete feedback:
  // ar_gpu_timed_steps == ar_steps > 0.
  std::uint64_t batch_submissions = 0;
  std::uint64_t batch_rows = 0;
  std::uint64_t batch_fallback_steps = 0;
  std::vector<std::uint64_t> batch_size_counts =
      std::vector<std::uint64_t>(9);
  std::vector<std::uint64_t> batch_width_counts =
      std::vector<std::uint64_t>(9);
  std::uint64_t batched_ar_steps = 0;
  std::uint64_t batched_verify_rounds = 0;
  double batch_gpu_ms = 0.0;
  std::uint64_t batch_sampler_icb_replays = 0;
  std::uint64_t batch_sampling_host_read_bytes = 0;
  std::uint64_t batch_compact_fallback_rows = 0;
  std::uint64_t shared_head_projection_calls = 0;
  std::uint64_t shared_head_multirow_calls = 0;
  std::uint64_t shared_head_projection_rows = 0;
  double shared_head_projection_gpu_ms = 0.0;
  std::uint32_t actual_b = 1;
  std::uint32_t actual_w = 1;
  std::uint64_t prefix_hits = 0;
  std::uint64_t prefix_partial_hits = 0;
  std::uint64_t prefix_misses = 0;
  std::uint64_t prefix_evictions = 0;
  std::uint64_t prefix_restored_tokens = 0;
  std::uint64_t ar_steps = 0;
  std::uint64_t ar_gpu_timed_steps = 0;
  std::uint64_t ar_layer_calls = 0;
  std::uint64_t ar_host_read_bytes = 0;
  double ar_prefill_gpu_ms = 0.0;
  double ar_decode_gpu_ms = 0.0;
  std::uint64_t spec_rounds = 0;
  // Wall phases: draft, verify, sampling, Replay/commit, publish, control/queue gap.
  std::uint64_t spec_cost_completed_rounds = 0;
  std::uint64_t spec_cost_aborted_rounds = 0;
  std::uint64_t spec_cost_committed_tokens = 0;
  double spec_cost_round_wall_ms = 0.0;
  std::vector<double> spec_cost_phase_wall_ms = std::vector<double>(6);
  std::vector<std::uint64_t> spec_cost_depth_rounds = std::vector<std::uint64_t>(8);
  std::vector<std::uint64_t> spec_cost_depth_committed_tokens = std::vector<std::uint64_t>(8);
  std::vector<double> spec_cost_depth_wall_ms = std::vector<double>(8);
  std::uint64_t spec_gpu_timed_rounds = 0;
  std::uint64_t spec_target_verify_calls = 0;
  std::uint64_t spec_replay_calls = 0;
  std::uint64_t spec_proposal_tokens = 0;
  std::uint64_t spec_accepted_tokens = 0;
  std::uint64_t spec_host_read_bytes = 0;
  std::uint64_t spec_context_rows_projected = 0;
  std::uint64_t spec_context_rows_reused = 0;
  std::uint64_t spec_context_rows_launched = 0;
  std::uint64_t short_matrix_calls = 0;
  std::uint64_t short_lane_calls = 0;
  double spec_draft_gpu_ms = 0.0;
  double spec_verify_gpu_ms = 0.0;
  // GPU distribution/acceptance pass after the single target forward.
  double spec_verify_selector_gpu_ms = 0.0;
  double spec_commit_gpu_ms = 0.0;
  std::string artifact_id;
  std::string route_table_id;
  std::uint32_t actual_t = 0;
  std::uint32_t actual_m = 0;
  std::uint32_t actual_k = 0;
  std::string kernel_family;
  std::string fallback_reason;
  std::uint64_t spec_search_gated_rounds = 0;
  std::uint64_t spec_search_probe_rounds = 0;
  std::uint64_t prefix_shared_hits = 0;
  std::uint64_t prefix_pinned_entries = 0;
  std::uint64_t contended_prefill_chunks = 0;
  std::uint64_t packed_batch_calls = 0;
  std::uint64_t packed_batch_tokens = 0;
  std::string lora_adapter;
  std::string lora_mode;
  std::uint64_t lora_buffer_bytes = 0;
  std::uint32_t lora_adapted_layers = 0;

  std::string prefill_linear_mode = "bf16";
  std::string prefill_attention_mode = "exact_nax";
  bool prefill_sparse_minference = false;
  bool prefill_sparse_minference_effective = false;
  std::string prefill_sparse_minference_inactive_reason;
  bool prefill_cache_blend = false;
  bool prefill_ane_offload = false;
  bool prefill_ane_active = false;
  double prefill_ane_split_alpha = 0.0;
  std::uint64_t prefill_w4a8_tokens = 0;
  std::uint64_t prefill_sage2_tokens = 0;
  std::uint64_t prefill_minference_blocks_skipped = 0;
  std::uint64_t prefill_minference_blocks_total = 0;
  std::uint64_t prefill_cache_blend_hit_tokens = 0;
  std::uint64_t prefill_cache_blend_recomputed_tokens = 0;
  std::uint64_t prefill_ane_offload_tokens = 0;
  std::uint64_t prefill_ane_fallback_tokens = 0;
};

using RuntimeStats = StatsReport;

inline void append_prefill_execution_identity_fields(
    const LoadOptions &options,
    std::map<std::string, std::string> &execution_fields) {
  if (options.prefill_linear_mode != PrefillLinearMode::BFloat16) {
    execution_fields.emplace(
        "prefill_linear_mode",
        prefill_linear_mode_to_string(options.prefill_linear_mode));
  }
  if (options.prefill_attention_mode != PrefillAttentionMode::ExactNax) {
    execution_fields.emplace(
        "prefill_attention_mode",
        prefill_attention_mode_to_string(options.prefill_attention_mode));
    if (options.prefill_sparse_minference) {
      execution_fields.emplace("prefill_sparse_minference", "1");
    }
  }
  if (options.prefill_ane_offload) {
    execution_fields.emplace("prefill_ane_offload", "1");
  }
}

inline void populate_prefill_config_stats(const LoadOptions &options,
                                          StatsReport &stats) {
  stats.prefill_linear_mode =
      prefill_linear_mode_to_string(options.prefill_linear_mode);
  stats.prefill_attention_mode =
      prefill_attention_mode_to_string(options.prefill_attention_mode);
  stats.prefill_sparse_minference = options.prefill_sparse_minference;
  stats.prefill_sparse_minference_effective =
      prefill_sparse_minference_is_effective(options.prefill_attention_mode,
                                             options.prefill_sparse_minference);
  stats.prefill_sparse_minference_inactive_reason =
      (options.prefill_sparse_minference &&
       !stats.prefill_sparse_minference_effective)
          ? std::string("requires_sage2")
          : std::string{};
  stats.prefill_cache_blend = options.prefill_cache_blend;
  stats.prefill_ane_offload = options.prefill_ane_offload;
}

struct FakeBackendOptions {
  std::int32_t event_batch_size = 1;
  std::int32_t step_delay_us = 0;
  std::int32_t fail_after_tokens = -1;
  bool fail_before_first = false;
  std::int32_t vocab_size = 248320;
};

} // namespace omlx::m5
