#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace omlx::m5 {

inline constexpr char kDefaultServingSamplingContract[] = "position_coupled_v1";
inline constexpr std::uint32_t kMaxMtpDraftTokens = 4;
inline constexpr std::uint32_t kMaxDFlash2DraftTokens = 7;

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
};

struct SubmitRequest {
  std::vector<std::int32_t> prompt_tokens;
  std::int32_t max_new_tokens = 1;
  SamplingSpec sampling;
  std::vector<std::vector<std::int32_t>> stop_token_seqs;
  bool pin_prefix = false;
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
};

struct FakeBackendOptions {
  std::int32_t event_batch_size = 1;
  std::int32_t step_delay_us = 0;
  std::int32_t fail_after_tokens = -1;
  bool fail_before_first = false;
  std::int32_t vocab_size = 248320;
};

} // namespace omlx::m5
