#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "omlx_m5/engine.h"
#include "omlx_m5/qwen_forward.h"
#include <nlohmann/json.hpp>

#ifndef OMLX_M5_QWEN_AR_METALLIB_PATH
#error "OMLX_M5_QWEN_AR_METALLIB_PATH is required"
#endif

namespace {
using namespace omlx::m5;
using Json = nlohmann::json;
struct Options {
  std::filesystem::path artifact, config, draft_artifact, draft_config;
  std::filesystem::path libraries =
      std::filesystem::path(OMLX_M5_QWEN_AR_METALLIB_PATH).parent_path();
  std::vector<std::int32_t> prompt;
  std::int32_t max_context = 1024, max_new = 16, repeat = 1;
  std::optional<std::size_t> cancel_after;
  SamplingSpec sampling;
  bool plan_only = false;
  std::optional<std::string> linear_route_ab;
  std::optional<std::string> m4_short_routes;
  std::string sampling_contract="legacy_rejection_v1",spec_backend="none";
  std::int32_t draft_tokens=0,max_concurrency=1;
  bool batch_enabled=false,projection_prototype=false;
};
void emit(const Json &value) {
  std::cout << value.dump() << '\n' << std::flush;
}
std::uint64_t integer(const std::string &value) {
  std::size_t used = 0;
  if (value.empty() || value.front() == '-')
    throw std::invalid_argument("expected nonnegative integer");
  auto result = std::stoull(value, &used);
  if (used != value.size())
    throw std::invalid_argument("invalid integer: " + value);
  return result;
}
std::int32_t i32(const std::string &value) {
  auto n = integer(value);
  if (n > INT32_MAX)
    throw std::out_of_range("integer exceeds supported range");
  return static_cast<std::int32_t>(n);
}
Options parse(int argc, char **argv) {
  Options o;
  for (int i = 1; i < argc; ++i) {
    std::string key(argv[i]);
    auto value = [&]() -> std::string {
      if (++i == argc)
        throw std::invalid_argument("missing value for " + key);
      return argv[i];
    };
    if (key == "--artifact")
      o.artifact = value();
    else if (key == "--config")
      o.config = value();
    else if (key == "--kernel-directory")
      o.libraries = value();
    else if (key == "--linear-route-ab")
      o.linear_route_ab = value();
    else if (key == "--m4-short-routes")
      o.m4_short_routes = value();
    else if(key=="--sampling-contract")
      o.sampling_contract=value();
    else if (key == "--spec-backend")o.spec_backend=value();
    else if (key == "--draft-tokens")o.draft_tokens=i32(value());
    else if (key == "--draft-artifact")o.draft_artifact=value();
    else if (key == "--draft-config")o.draft_config=value();
    else if (key == "--max-concurrency")o.max_concurrency=i32(value());
    else if (key == "--batch-mode") {auto mode=value();if(mode!="on" && mode!="off")throw std::invalid_argument("batch-mode must be on or off");o.batch_enabled=mode=="on";}
    else if (key == "--batch-projection-prototype") {auto mode=value();if(mode!="on" && mode!="off")throw std::invalid_argument("batch projection prototype must be on or off");o.projection_prototype=mode=="on";}
    else if (key == "--max-context")
      o.max_context = i32(value());
    else if (key == "--max-new-tokens")
      o.max_new = i32(value());
    else if (key == "--repeat")
      o.repeat = i32(value());
    else if (key == "--cancel-after")
      o.cancel_after = integer(value());
    else if (key == "--temperature")
      o.sampling.temperature = std::stof(value());
    else if (key == "--top-k")
      o.sampling.top_k = i32(value());
    else if (key == "--top-p")
      o.sampling.top_p = std::stof(value());
    else if (key == "--seed")
      o.sampling.seed = integer(value());
    else if (key == "--plan-only")
      o.plan_only = true;
    else if (key == "--prompt") {
      std::stringstream input(value());
      std::string token;
      while (std::getline(input, token, ','))
        o.prompt.push_back(i32(token));
    } else
      throw std::invalid_argument("unknown runner argument: " + key);
  }
  if (o.artifact.empty() || o.config.empty() || o.max_context < 1024 ||
      o.max_new <= 0 || o.repeat <= 0 || (!o.plan_only && o.prompt.empty()))
    throw std::invalid_argument(
        "requires artifact/config, max-context>=1024, positive limits and "
        "prompt (unless plan-only)");
  if(o.spec_backend!="none" && o.spec_backend!="dflash2")throw std::invalid_argument("unsupported spec backend");
  if(o.spec_backend=="none" && o.draft_tokens)throw std::invalid_argument("none requires draft tokens 0");
  if(o.spec_backend=="dflash2" && (o.draft_tokens<1 || o.draft_tokens>7 || o.draft_artifact.empty() || o.draft_config.empty() || o.plan_only))
    throw std::invalid_argument("dflash2 requires D1-7 and artifact/config; plan-only remains AR");
  if(o.max_concurrency<1 || o.max_concurrency>8 || (o.batch_enabled && o.max_concurrency<2))throw std::invalid_argument("invalid concurrency/batch envelope");
  return o;
}
Json stats_json(const StatsReport &s) {
  return {{"model_execution_kind", s.model_execution_kind},
      {"sampling_contract_id", s.sampling_contract_id},
      {"numeric_contract_id", s.numeric_contract_id},
      {"chunk_policy_id", s.chunk_policy_id},
      {"execution_identity_id", s.execution_identity_id},
      {"compiled_execution_source_id", s.compiled_execution_source_id},
      {"kernel_bundle_id", s.kernel_bundle_id},
      {"batch_execution_kind", s.batch_execution_kind},
      {"batch_sampling_host_read_bytes", s.batch_sampling_host_read_bytes},
      {"batch_compact_fallback_rows", s.batch_compact_fallback_rows},
      {"shared_head_projection_calls", s.shared_head_projection_calls},
      {"shared_head_multirow_calls", s.shared_head_multirow_calls},
      {"shared_head_projection_rows", s.shared_head_projection_rows},
      {"shared_head_projection_gpu_ms", s.shared_head_projection_gpu_ms},

          {"artifact_id", s.artifact_id},
          {"route_table_id", s.route_table_id},
          {"kernel_family", s.kernel_family},
          {"spec_gpu_timed_rounds", s.spec_gpu_timed_rounds},
          {"spec_rounds", s.spec_rounds},
      {"spec_cost_completed_rounds", s.spec_cost_completed_rounds},
      {"spec_cost_aborted_rounds", s.spec_cost_aborted_rounds},
      {"spec_cost_committed_tokens", s.spec_cost_committed_tokens},
      {"spec_cost_round_wall_ms", s.spec_cost_round_wall_ms},
      {"spec_cost_phase_wall_ms", s.spec_cost_phase_wall_ms},
      {"spec_cost_depth_rounds", s.spec_cost_depth_rounds},
      {"spec_cost_depth_committed_tokens", s.spec_cost_depth_committed_tokens},
      {"spec_cost_depth_wall_ms", s.spec_cost_depth_wall_ms},

          {"spec_target_verify_calls", s.spec_target_verify_calls},
          {"spec_replay_calls", s.spec_replay_calls},
          {"spec_proposal_tokens", s.spec_proposal_tokens},
          {"spec_accepted_tokens", s.spec_accepted_tokens},
          {"spec_host_read_bytes", s.spec_host_read_bytes},
          {"spec_context_rows_projected", s.spec_context_rows_projected},
          {"spec_context_rows_reused", s.spec_context_rows_reused},
          {"spec_context_rows_launched", s.spec_context_rows_launched},
          {"short_matrix_calls", s.short_matrix_calls},
          {"short_lane_calls", s.short_lane_calls},
          {"spec_draft_gpu_ms", s.spec_draft_gpu_ms},
          {"spec_verify_gpu_ms", s.spec_verify_gpu_ms},
          {"spec_verify_selector_gpu_ms", s.spec_verify_selector_gpu_ms},
          {"spec_commit_gpu_ms", s.spec_commit_gpu_ms},
          {"batch_submissions", s.batch_submissions},
          {"batch_rows", s.batch_rows},
          {"batch_fallback_steps", s.batch_fallback_steps},
          {"batch_size_counts", s.batch_size_counts},
          {"batch_width_counts", s.batch_width_counts},
          {"batched_ar_steps", s.batched_ar_steps},
          {"batched_verify_rounds", s.batched_verify_rounds},
          {"batch_gpu_ms", s.batch_gpu_ms},
          {"batch_sampler_icb_replays", s.batch_sampler_icb_replays},
          {"actual_b", s.actual_b},
          {"actual_w", s.actual_w},
          {"prefix_hits", s.prefix_hits},
          {"prefix_partial_hits", s.prefix_partial_hits},
          {"prefix_misses", s.prefix_misses},
          {"prefix_evictions", s.prefix_evictions},
          {"prefix_restored_tokens", s.prefix_restored_tokens},
          {"ar_steps", s.ar_steps},
          {"ar_layer_calls", s.ar_layer_calls},
          {"ar_host_read_bytes", s.ar_host_read_bytes},
          {"ar_gpu_timed_steps", s.ar_gpu_timed_steps},
          {"ar_prefill_gpu_ms", s.ar_prefill_gpu_ms},
          {"ar_decode_gpu_ms", s.ar_decode_gpu_ms},
          {"fallback_reason", s.fallback_reason}};
}
} // namespace

int main(int argc, char **argv) {
  try {
    auto o = parse(argc, argv);
    auto config = load_qwen_model_config(o.config);
    if (o.plan_only) {
      auto context = MetalContext::create();
      auto artifact = ArtifactModel::open(o.artifact, false);
      auto weights =
          plan_model_materialization(artifact, context,
                                     {.components = {ModelComponent::text},
                                      .require_verified_payload = false});
      auto memory = plan_qwen_forward_memory(config,o.max_context,0,false,false,false,
        parse_sampling_contract(o.sampling_contract)==SamplingContract::PositionCoupledV1,
        parse_sampling_contract(o.sampling_contract)==SamplingContract::PositionCoupledV1);
      auto budget =
          qwen_load_budget(weights.resident_bytes, memory,
                           context.capabilities().recommended_working_set_bytes,
                           context.current_allocated_bytes());
      Json buffers = Json::array();
      for (const auto &b : memory.buffers)
        buffers.push_back(
            {{"name", b.name}, {"bytes", b.bytes}, {"state", b.state}});
      emit({{"event", "plan"},
            {"scope", "allocation plan, no model execution"},
            {"device", context.capabilities().device_name},
            {"weights_bytes", weights.resident_bytes},
            {"workspace_bytes", memory.workspace_bytes},
            {"state_bytes", memory.state_bytes},
            {"runtime_reserve_bytes", budget.runtime_reserve_bytes},
            {"required_bytes", budget.required_bytes},
            {"available_bytes", budget.available_bytes},
            {"fits_current_device", budget.fits},
            {"buffers", buffers}});
      return 0;
    }
    LoadOptions load;
    load.sampling_contract=o.sampling_contract;
    load.model_dir = o.config.parent_path().string();
    load.max_context = o.max_context;
    load.max_concurrency=o.max_concurrency;
    if(o.spec_backend=="dflash2") {
      load.draft_dir=o.draft_config.parent_path().string();
      load.speculative={.backend=SpeculativeBackend::dflash2,.draft_tokens=std::uint32_t(o.draft_tokens)};
    }
    load.path_overrides = {{"artifact", o.artifact.string()},
                           {"kernel_directory", o.libraries.string()}};
    if(o.spec_backend=="dflash2")load.path_overrides["draft_artifact"]=o.draft_artifact.string();
    if(o.batch_enabled)load.path_overrides["batch_mode"]="on";
    if(o.projection_prototype)load.path_overrides["batch_projection_prototype"]="on";
    if (o.linear_route_ab)
      load.path_overrides["linear_route_ab"] = *o.linear_route_ab;
    if (o.m4_short_routes)
      load.path_overrides["m4_short_routes"] = *o.m4_short_routes;
    auto engine = Engine::create(load);
    emit({{"event", "ready"},
          {"layer_count", config.num_hidden_layers},
          {"vocabulary_size", config.vocab_size},
          {"stats", stats_json(engine->stats())}});
    std::vector<std::int32_t> previous;
    bool identical = true, failed = false;
    for (int round = 0; round < o.repeat; ++round) {
      SubmitRequest request;
      request.prompt_tokens = o.prompt;
      request.max_new_tokens = o.max_new;
      request.sampling = o.sampling;
      if (config.eos_token_id >= 0)
        request.stop_token_seqs = {{config.eos_token_id}};
      auto id = engine->submit(request);
      bool cancel_requested = o.cancel_after && *o.cancel_after == 0;
      if (cancel_requested)
        static_cast<void>(engine->cancel(id));
      std::optional<TokenEvent> terminal;
      std::vector<std::int32_t> tokens;
      const auto deadline =
          std::chrono::steady_clock::now() + std::chrono::minutes(15);
      while (!terminal) {
        if (std::chrono::steady_clock::now() > deadline) {
          static_cast<void>(engine->cancel(id));
          throw std::runtime_error("runner request timed out");
        }
        for (auto &event :
             engine->poll_events(std::chrono::microseconds(20000))) {
          if (event.request_id != id)
            throw std::runtime_error("unexpected request event");
          if (event.kind != EventKind::tokens) {
            terminal = std::move(event);
            continue;
          }
          for (auto token : event.tokens) {
            if (cancel_requested)
              break;
            emit({{"event", "token"},
                  {"request", round},
                  {"index", tokens.size()},
                  {"token", token},
                  {"position", o.prompt.size() + tokens.size()}});
            tokens.push_back(token);
            if (o.cancel_after && tokens.size() >= *o.cancel_after) {
              static_cast<void>(engine->cancel(id));
              cancel_requested = true;
              break;
            }
          }
        }
      }
      failed = failed || terminal->kind == EventKind::failed;
      if (round)
        identical = identical && tokens == previous;
      previous = tokens;
      emit({{"event",
             terminal->kind == EventKind::failed ? "error" : "request_done"},
            {"request", round},
            {"tokens", tokens.size()},
            {"cancel_requested", cancel_requested},
            {"cancelled", terminal->kind == EventKind::aborted},
            {"failed", terminal->kind == EventKind::failed},
            {"finish_reason", terminal->finish_reason},
            {"error", terminal->error}});
      if (failed)
        break;
    }
    emit({{"event", "done"},
          {"repeat_identical", identical},
          {"stats", stats_json(engine->stats())}});
    engine->shutdown();
    return failed ? 2 : 0;
  } catch (const std::exception &e) {
    emit({{"event", "fatal"}, {"error", e.what()}});
    return 1;
  }
}
