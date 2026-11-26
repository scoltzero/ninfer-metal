#include <chrono>
#include <cstdint>
#include <memory>

#include <nanobind/nanobind.h>
#include <nanobind/stl/map.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/vector.h>

#include "omlx_m5/engine.h"

namespace nb = nanobind;
using namespace nb::literals;

NB_MODULE(omlx_m5_native, module) {
  using omlx::m5::CapacityReport;
  using omlx::m5::Engine;
  using omlx::m5::EventKind;
  using omlx::m5::FakeBackendOptions;
  using omlx::m5::LoadOptions;
  using omlx::m5::RequestStats;
  using omlx::m5::SamplingSpec;
  using omlx::m5::SpeculativeBackend;
  using omlx::m5::SpeculativeOptions;
  using omlx::m5::StatsReport;
  using omlx::m5::SubmitRequest;
  using omlx::m5::TokenEvent;

  module.doc() = "Native and fake backends for the oMLX M5 runtime";

  nb::enum_<EventKind>(module, "EventKind")
      .value("TOKENS", EventKind::tokens)
      .value("FINISHED", EventKind::finished)
      .value("ABORTED", EventKind::aborted)
      .value("FAILED", EventKind::failed);

  nb::enum_<SpeculativeBackend>(module, "SpeculativeBackend")
      .value("NONE", SpeculativeBackend::none)
      .value("MTP", SpeculativeBackend::mtp)
      .value("DFLASH", SpeculativeBackend::dflash)
      .value("DFLASH2", SpeculativeBackend::dflash2);

  nb::class_<SpeculativeOptions>(module, "SpeculativeOptions")
      .def(nb::init<>())
      .def_rw("backend", &SpeculativeOptions::backend)
      .def_rw("draft_tokens", &SpeculativeOptions::draft_tokens);

  nb::class_<LoadOptions>(module, "LoadOptions")
      .def(nb::init<>())
      .def_rw("model_dir", &LoadOptions::model_dir)
      .def_rw("draft_dir", &LoadOptions::draft_dir)
      .def_rw("max_concurrency", &LoadOptions::max_concurrency)
      .def_rw("max_context", &LoadOptions::max_context)
      .def_rw("speculative", &LoadOptions::speculative)
      .def_rw("kv_dtype", &LoadOptions::kv_dtype)
      .def_rw("prefill_engine", &LoadOptions::prefill_engine)
      .def_rw("spec_search", &LoadOptions::spec_search)
      .def_rw("sampling_contract", &LoadOptions::sampling_contract)
      .def_rw("path_overrides", &LoadOptions::path_overrides);

  nb::class_<SamplingSpec>(module, "SamplingSpec")
      .def(nb::init<>())
      .def_rw("temperature", &SamplingSpec::temperature)
      .def_rw("top_p", &SamplingSpec::top_p)
      .def_rw("min_p", &SamplingSpec::min_p)
      .def_rw("top_k", &SamplingSpec::top_k)
      .def_rw("seed", &SamplingSpec::seed);

  nb::class_<SubmitRequest>(module, "SubmitRequest")
      .def(nb::init<>())
      .def_rw("prompt_tokens", &SubmitRequest::prompt_tokens)
      .def_rw("max_new_tokens", &SubmitRequest::max_new_tokens)
      .def_rw("sampling", &SubmitRequest::sampling)
      .def_rw("stop_token_seqs", &SubmitRequest::stop_token_seqs)
      .def_rw("pin_prefix", &SubmitRequest::pin_prefix);

  nb::class_<RequestStats>(module, "RequestStats")
      .def_ro("prefix_restored_tokens", &RequestStats::prefix_restored_tokens)
      .def_ro("spec_rounds", &RequestStats::spec_rounds)
      .def_ro("ar_steps", &RequestStats::ar_steps)
      .def_ro("decode_mode", &RequestStats::decode_mode)
      .def_ro("fallback_reason", &RequestStats::fallback_reason);

  nb::class_<TokenEvent>(module, "TokenEvent")
      .def_ro("request_id", &TokenEvent::request_id)
      .def_ro("tokens", &TokenEvent::tokens)
      .def_ro("kind", &TokenEvent::kind)
      .def_ro("finish_reason", &TokenEvent::finish_reason)
      .def_ro("error", &TokenEvent::error)
      .def_ro("request_stats", &TokenEvent::request_stats);

  nb::class_<CapacityReport>(module, "CapacityReport")
      .def_ro("max_concurrency", &CapacityReport::max_concurrency)
      .def_ro("active_requests", &CapacityReport::active_requests)
      .def_ro("queued_requests", &CapacityReport::queued_requests)
      .def_ro("accepting", &CapacityReport::accepting);

  nb::class_<StatsReport>(module, "StatsReport")
      .def_ro("submitted", &StatsReport::submitted)
      .def_ro("completed", &StatsReport::completed)
      .def_ro("aborted", &StatsReport::aborted)
      .def_ro("failed", &StatsReport::failed)
      .def_ro("emitted_tokens", &StatsReport::emitted_tokens)
      .def_ro("backend_kind", &StatsReport::backend_kind)
      .def_ro("max_context", &StatsReport::max_context)
      .def_ro("max_concurrency", &StatsReport::max_concurrency)
      .def_ro("kv_dtype", &StatsReport::kv_dtype)
      .def_ro("speculative_backend", &StatsReport::speculative_backend)
      .def_ro("max_draft_tokens", &StatsReport::max_draft_tokens)
      .def_ro("model_execution_kind", &StatsReport::model_execution_kind)
      .def_ro("sampling_contract_id", &StatsReport::sampling_contract_id)
      .def_ro("numeric_contract_id", &StatsReport::numeric_contract_id)
      .def_ro("chunk_policy_id", &StatsReport::chunk_policy_id)
      .def_ro("execution_identity_id", &StatsReport::execution_identity_id)
      .def_ro("compiled_execution_source_id", &StatsReport::compiled_execution_source_id)
      .def_ro("kernel_bundle_id", &StatsReport::kernel_bundle_id)
      .def_ro("batch_execution_kind", &StatsReport::batch_execution_kind)
      .def_ro("batch_sampling_host_read_bytes", &StatsReport::batch_sampling_host_read_bytes)
      .def_ro("batch_compact_fallback_rows", &StatsReport::batch_compact_fallback_rows)
      .def_ro("shared_head_projection_calls", &StatsReport::shared_head_projection_calls)
      .def_ro("shared_head_multirow_calls", &StatsReport::shared_head_multirow_calls)
      .def_ro("shared_head_projection_rows", &StatsReport::shared_head_projection_rows)
      .def_ro("shared_head_projection_gpu_ms", &StatsReport::shared_head_projection_gpu_ms)

      .def_ro("spec_gpu_timed_rounds", &StatsReport::spec_gpu_timed_rounds)
      .def_ro("spec_rounds", &StatsReport::spec_rounds)
      .def_ro("spec_cost_completed_rounds", &StatsReport::spec_cost_completed_rounds)
      .def_ro("spec_cost_aborted_rounds", &StatsReport::spec_cost_aborted_rounds)
      .def_ro("spec_cost_committed_tokens", &StatsReport::spec_cost_committed_tokens)
      .def_ro("spec_cost_round_wall_ms", &StatsReport::spec_cost_round_wall_ms)
      .def_ro("spec_cost_phase_wall_ms", &StatsReport::spec_cost_phase_wall_ms)
      .def_ro("spec_cost_depth_rounds", &StatsReport::spec_cost_depth_rounds)
      .def_ro("spec_cost_depth_committed_tokens", &StatsReport::spec_cost_depth_committed_tokens)
      .def_ro("spec_cost_depth_wall_ms", &StatsReport::spec_cost_depth_wall_ms)

      .def_ro("spec_target_verify_calls",
              &StatsReport::spec_target_verify_calls)
      .def_ro("spec_replay_calls", &StatsReport::spec_replay_calls)
      .def_ro("spec_proposal_tokens", &StatsReport::spec_proposal_tokens)
      .def_ro("spec_accepted_tokens", &StatsReport::spec_accepted_tokens)
      .def_ro("spec_host_read_bytes", &StatsReport::spec_host_read_bytes)
      .def_ro("spec_context_rows_projected",
              &StatsReport::spec_context_rows_projected)
      .def_ro("spec_context_rows_reused",
              &StatsReport::spec_context_rows_reused)
      .def_ro("spec_context_rows_launched",
              &StatsReport::spec_context_rows_launched)
      .def_ro("short_matrix_calls", &StatsReport::short_matrix_calls)
      .def_ro("short_lane_calls", &StatsReport::short_lane_calls)
      .def_ro("spec_draft_gpu_ms", &StatsReport::spec_draft_gpu_ms)
      .def_ro("spec_verify_gpu_ms", &StatsReport::spec_verify_gpu_ms)
      .def_ro("spec_verify_selector_gpu_ms",
              &StatsReport::spec_verify_selector_gpu_ms)
      .def_ro("spec_commit_gpu_ms", &StatsReport::spec_commit_gpu_ms)
      .def_ro("batch_submissions", &StatsReport::batch_submissions)
      .def_ro("batch_rows", &StatsReport::batch_rows)
      .def_ro("batch_fallback_steps", &StatsReport::batch_fallback_steps)
      .def_ro("batch_size_counts", &StatsReport::batch_size_counts)
      .def_ro("batch_width_counts", &StatsReport::batch_width_counts)
      .def_ro("batched_ar_steps", &StatsReport::batched_ar_steps)
      .def_ro("batched_verify_rounds", &StatsReport::batched_verify_rounds)
      .def_ro("batch_gpu_ms", &StatsReport::batch_gpu_ms)
      .def_ro("batch_sampler_icb_replays",
              &StatsReport::batch_sampler_icb_replays)
      .def_ro("actual_b", &StatsReport::actual_b)
      .def_ro("actual_w", &StatsReport::actual_w)
      .def_ro("prefix_hits", &StatsReport::prefix_hits)
      .def_ro("prefix_partial_hits", &StatsReport::prefix_partial_hits)
      .def_ro("prefix_misses", &StatsReport::prefix_misses)
      .def_ro("prefix_evictions", &StatsReport::prefix_evictions)
      .def_ro("prefix_restored_tokens", &StatsReport::prefix_restored_tokens)
      .def_ro("ar_steps", &StatsReport::ar_steps)
      .def_ro("ar_gpu_timed_steps", &StatsReport::ar_gpu_timed_steps)
      .def_ro("ar_layer_calls", &StatsReport::ar_layer_calls)
      .def_ro("ar_host_read_bytes", &StatsReport::ar_host_read_bytes)
      .def_ro("ar_prefill_gpu_ms", &StatsReport::ar_prefill_gpu_ms)
      .def_ro("ar_decode_gpu_ms", &StatsReport::ar_decode_gpu_ms)
      .def_ro("artifact_id", &StatsReport::artifact_id)
      .def_ro("route_table_id", &StatsReport::route_table_id)
      .def_ro("actual_t", &StatsReport::actual_t)
      .def_ro("actual_m", &StatsReport::actual_m)
      .def_ro("actual_k", &StatsReport::actual_k)
      .def_ro("kernel_family", &StatsReport::kernel_family)
      .def_ro("fallback_reason", &StatsReport::fallback_reason)
      .def_ro("spec_search_gated_rounds",
              &StatsReport::spec_search_gated_rounds)
      .def_ro("spec_search_probe_rounds",
              &StatsReport::spec_search_probe_rounds)
      .def_ro("prefix_shared_hits", &StatsReport::prefix_shared_hits)
      .def_ro("prefix_pinned_entries", &StatsReport::prefix_pinned_entries)
      .def_ro("contended_prefill_chunks",
              &StatsReport::contended_prefill_chunks)
      .def_ro("packed_batch_calls", &StatsReport::packed_batch_calls)
      .def_ro("packed_batch_tokens", &StatsReport::packed_batch_tokens);

  nb::class_<FakeBackendOptions>(module, "FakeBackendOptions")
      .def(nb::init<>())
      .def_rw("event_batch_size", &FakeBackendOptions::event_batch_size)
      .def_rw("step_delay_us", &FakeBackendOptions::step_delay_us)
      .def_rw("fail_after_tokens", &FakeBackendOptions::fail_after_tokens)
      .def_rw("fail_before_first", &FakeBackendOptions::fail_before_first)
      .def_rw("vocab_size", &FakeBackendOptions::vocab_size);

  nb::class_<Engine>(module, "Engine")
      .def_static("create", &Engine::create, "load_options"_a = LoadOptions{},
                  nb::call_guard<nb::gil_scoped_release>())
      .def_static("create_fake", &Engine::create_fake,
                  "load_options"_a = LoadOptions{},
                  "fake_options"_a = FakeBackendOptions{},
                  nb::call_guard<nb::gil_scoped_release>())
      .def("submit", &Engine::submit, "request"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def("cancel", &Engine::cancel, "request_id"_a,
           nb::call_guard<nb::gil_scoped_release>())
      .def(
          "poll_events",
          [](Engine &engine, std::int64_t max_wait_us) {
            return engine.poll_events(std::chrono::microseconds(max_wait_us));
          },
          "max_wait_us"_a, nb::call_guard<nb::gil_scoped_release>())
      .def("capacity", &Engine::capacity,
           nb::call_guard<nb::gil_scoped_release>())
      .def("stats", &Engine::stats,
           nb::call_guard<nb::gil_scoped_release>())
      .def("shutdown", &Engine::shutdown,
           nb::call_guard<nb::gil_scoped_release>());
}
