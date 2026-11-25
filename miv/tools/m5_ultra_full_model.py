#!/usr/bin/env python3
"""Run or plan full-model native AR and DFlash2 measurements on M5 Ultra."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPT_PARENT = _SCRIPT_DIR.parent
ROOT = _SCRIPT_PARENT.parent if _SCRIPT_PARENT.name == "miv" else _SCRIPT_PARENT
sys.path.insert(0, str(_SCRIPT_DIR))

from m5_target_guard import hardware_facts

DEFAULT_DRAFT_DIR = "models/z-lab/Qwen3.8-27B-DFlash2"
DEFAULT_DRAFT_ARTIFACT = ".cache/b5/real/qwen38-27b-dflash2.omlxm5"

ARTIFACT_PRESETS: dict[str, dict[str, object]] = {
    "qwen38-27b-oq4e": {
        "model_dir": "models/Jundot/Qwen3.8-27B-oQ4e-mtp",
        "artifact": ".cache/b0/artifacts/qwen38-27b-oq4e.omlxm5",
        "draft_supported": True,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen38-27b-oq4e-pretiled": {
        "model_dir": "models/Jundot/Qwen3.8-27B-oQ4e-mtp",
        "artifact": ".cache/b0/artifacts/qwen38-27b-oq4e-pretiled.omlxm5",
        "draft_supported": True,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen38-27b-mlx-mxfp4": {
        "model_dir": "models/mlx-community/Qwen3.8-27B-mxfp4",
        "artifact": ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4.omlxm5",
        "draft_supported": True,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen38-27b-mlx-mxfp4-pretiled": {
        "model_dir": "models/mlx-community/Qwen3.8-27B-mxfp4",
        "artifact": ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4-pretiled.omlxm5",
        "draft_supported": True,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen38-27b-redhat-mxfp4": {
        "model_dir": "models/RedHatAI/Qwen3.8-27B-MXFP4",
        "artifact": ".cache/b0/artifacts/qwen38-27b-redhat-mxfp4.omlxm5",
        "draft_supported": True,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen36-27b-oq4e": {
        "model_dir": "models/Jundot/Qwen3.6-27B-oQ4e-mtp",
        "artifact": ".cache/b0/artifacts/qwen36-27b-oq4e.omlxm5",
        "draft_supported": False,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen36-27b-mxfp4": {
        "model_dir": "models/OsaurusAI/Qwen3.6-27B-MXFP4",
        "artifact": ".cache/b0/artifacts/qwen36-27b-mxfp4.omlxm5",
        "draft_supported": False,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen36-35b-a3b-oq4e": {
        "model_dir": "models/Jundot/Qwen3.6-35B-A3B-oQ4e-mtp",
        "artifact": ".cache/b0/artifacts/qwen36-35b-a3b-oq4e.omlxm5",
        "draft_supported": False,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
    "qwen36-35b-a3b-mxfp4": {
        "model_dir": "models/OsaurusAI/Qwen3.6-35B-A3B-mxfp4",
        "artifact": ".cache/b0/artifacts/qwen36-35b-a3b-mxfp4.omlxm5",
        "draft_supported": False,
        "draft_dir": DEFAULT_DRAFT_DIR,
        "draft_artifact": DEFAULT_DRAFT_ARTIFACT,
    },
}

PRESET_ALIASES: dict[str, str] = {
    "27b-oq4e": "qwen38-27b-oq4e",
    "27b-oq4e-pretiled": "qwen38-27b-oq4e-pretiled",
    "27b-mxfp4": "qwen38-27b-mlx-mxfp4",
    "27b-mxfp4-pretiled": "qwen38-27b-mlx-mxfp4-pretiled",
    "27b-redhat-mxfp4": "qwen38-27b-redhat-mxfp4",
    "35b-oq4e": "qwen36-35b-a3b-oq4e",
    "35b-moe-oq4e": "qwen36-35b-a3b-oq4e",
    "35b-mxfp4": "qwen36-35b-a3b-mxfp4",
    "35b-moe-mxfp4": "qwen36-35b-a3b-mxfp4",
}

STATS_FIELDS = (
    "submitted",
    "completed",
    "aborted",
    "failed",
    "emitted_tokens",
    "backend_kind",
    "model_execution_kind",
    "sampling_contract_id",
    "numeric_contract_id",
    "chunk_policy_id",
    "execution_identity_id",
    "compiled_execution_source_id",
    "kernel_bundle_id",
    "batch_execution_kind",
    "batch_sampling_host_read_bytes",
    "batch_compact_fallback_rows",
    "shared_head_projection_calls",
    "shared_head_multirow_calls",
    "shared_head_projection_rows",
    "shared_head_projection_gpu_ms",
    "spec_rounds",
    "spec_gpu_timed_rounds",
    "spec_cost_completed_rounds",
    "spec_cost_aborted_rounds",
    "spec_cost_committed_tokens",
    "spec_cost_round_wall_ms",
    "spec_cost_phase_wall_ms",
    "spec_cost_depth_rounds",
    "spec_cost_depth_committed_tokens",
    "spec_cost_depth_wall_ms",
    "spec_target_verify_calls",
    "spec_replay_calls",
    "spec_proposal_tokens",
    "spec_accepted_tokens",
    "spec_host_read_bytes",
    "spec_context_rows_projected",
    "spec_context_rows_reused",
    "spec_context_rows_launched",
    "short_matrix_calls",
    "short_lane_calls",
    "spec_draft_gpu_ms",
    "spec_verify_gpu_ms",
    "spec_verify_selector_gpu_ms",
    "spec_commit_gpu_ms",
    "batch_submissions",
    "batch_rows",
    "batch_fallback_steps",
    "batch_size_counts",
    "batch_width_counts",
    "batched_ar_steps",
    "batched_verify_rounds",
    "batch_gpu_ms",
    "batch_sampler_icb_replays",
    "actual_b",
    "actual_w",
    "prefix_hits",
    "prefix_partial_hits",
    "prefix_misses",
    "prefix_evictions",
    "prefix_restored_tokens",
    "ar_steps",
    "ar_gpu_timed_steps",
    "ar_layer_calls",
    "ar_host_read_bytes",
    "ar_prefill_gpu_ms",
    "ar_decode_gpu_ms",
    "artifact_id",
    "route_table_id",
    "actual_t",
    "actual_m",
    "actual_k",
    "kernel_family",
    "fallback_reason",
    "spec_search_gated_rounds",
    "spec_search_probe_rounds",
    "prefix_shared_hits",
    "prefix_pinned_entries",
    "contended_prefill_chunks",
    "packed_batch_calls",
    "packed_batch_tokens",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def parse_tokens(value: str) -> list[int]:
    tokens = [int(part) for part in value.split(",") if part]
    require(bool(tokens) and all(token >= 0 for token in tokens), "invalid prompt tokens")
    return tokens


def normalize_preset(name: str | None) -> str | None:
    if name is None:
        return None
    canonical = PRESET_ALIASES.get(name, name)
    require(
        canonical in ARTIFACT_PRESETS,
        f"unknown preset {name!r}; valid presets: {', '.join(sorted(ARTIFACT_PRESETS))}",
    )
    return canonical


def collect(
    native, engine, request_id: int, timeout_seconds: float, *, start: float
) -> dict:
    first_token_at: float | None = None
    last_token_at: float | None = None
    first_batch_tokens = 0
    tokens: list[int] = []
    terminal = None
    while terminal is None:
        if time.perf_counter() - start > timeout_seconds:
            engine.cancel(request_id)
            raise TimeoutError(f"request {request_id} exceeded {timeout_seconds}s")
        for event in engine.poll_events(20_000):
            if event.request_id != request_id:
                continue
            if event.kind == native.EventKind.TOKENS and event.tokens:
                last_token_at = time.perf_counter()
                if first_token_at is None:
                    first_token_at = last_token_at
                    first_batch_tokens = len(event.tokens)
                tokens.extend(int(token) for token in event.tokens)
            else:
                terminal = event
    elapsed = time.perf_counter() - start
    require(
        terminal.kind == native.EventKind.FINISHED, f"request failed: {terminal.error}"
    )
    require(bool(tokens), "full-model request emitted no tokens")
    decode_elapsed = last_token_at - first_token_at
    decode_tokens = len(tokens) - first_batch_tokens
    return {
        "tokens": tokens,
        "token_count": len(tokens),
        "elapsed_seconds": elapsed,
        "ttft_seconds": (first_token_at - start) if first_token_at else None,
        "request_output_tokens_per_second": len(tokens) / elapsed,
        "decode_tokens_per_second": (
            decode_tokens / decode_elapsed
            if decode_elapsed > 0 and decode_tokens
            else None
        ),
        "decode_token_count": decode_tokens,
        "decode_seconds": decode_elapsed,
        "timing_scope": "host event arrival; first batch excluded from decode",
        "finish_reason": terminal.finish_reason,
    }


def collect_concurrent(
    native,
    engine,
    request_ids: list[int],
    timeout_seconds: float,
    *,
    start: float,
) -> dict[str, object]:
    pending = set(request_ids)
    per_req: dict[int, dict[str, object]] = {
        rid: {
            "first_token_at": None,
            "last_token_at": None,
            "first_batch_tokens": 0,
            "tokens": [],
            "terminal": None,
            "finished_at": None,
        }
        for rid in request_ids
    }
    while pending:
        if time.perf_counter() - start > timeout_seconds:
            for rid in sorted(pending):
                engine.cancel(rid)
            raise TimeoutError(
                f"concurrent requests {sorted(pending)} exceeded {timeout_seconds}s"
            )
        for event in engine.poll_events(20_000):
            if event.request_id not in per_req:
                continue
            state = per_req[event.request_id]
            if event.kind == native.EventKind.TOKENS and event.tokens:
                now = time.perf_counter()
                state["last_token_at"] = now
                if state["first_token_at"] is None:
                    state["first_token_at"] = now
                    state["first_batch_tokens"] = len(event.tokens)
                state["tokens"].extend(int(token) for token in event.tokens)
            elif state["terminal"] is None:
                state["terminal"] = event
                state["finished_at"] = time.perf_counter()
                pending.discard(event.request_id)

    wall_elapsed = time.perf_counter() - start
    streams = []
    total_tokens = 0
    total_decode_tokens = 0
    for rid in request_ids:
        state = per_req[rid]
        terminal = state["terminal"]
        require(
            terminal is not None and terminal.kind == native.EventKind.FINISHED,
            f"request {rid} failed: {getattr(terminal, 'error', 'missing terminal')}",
        )
        toks: list[int] = list(state["tokens"])
        require(bool(toks), f"request {rid} emitted no tokens")
        first_at = float(state["first_token_at"])
        last_at = float(state["last_token_at"])
        fin_at = float(state["finished_at"])
        req_elapsed = fin_at - start
        dec_elapsed = last_at - first_at
        dec_tokens = len(toks) - int(state["first_batch_tokens"])
        total_tokens += len(toks)
        total_decode_tokens += dec_tokens
        streams.append(
            {
                "request_id": rid,
                "tokens": toks,
                "token_count": len(toks),
                "elapsed_seconds": req_elapsed,
                "ttft_seconds": first_at - start,
                "request_output_tokens_per_second": len(toks) / req_elapsed,
                "decode_tokens_per_second": (
                    dec_tokens / dec_elapsed
                    if dec_elapsed > 0 and dec_tokens > 0
                    else None
                ),
                "decode_token_count": dec_tokens,
                "decode_seconds": dec_elapsed,
                "finish_reason": terminal.finish_reason,
            }
        )

    stream_decode_rates = [
        float(s["decode_tokens_per_second"])
        for s in streams
        if s["decode_tokens_per_second"] is not None
    ]
    return {
        "tokens": streams[0]["tokens"],
        "token_count": total_tokens,
        "concurrent_streams": len(streams),
        "streams": streams,
        "elapsed_seconds": wall_elapsed,
        "ttft_seconds": statistics.median(float(s["ttft_seconds"]) for s in streams),
        "request_output_tokens_per_second": total_tokens / wall_elapsed,
        "per_stream_median_output_tokens_per_second": statistics.median(
            float(s["request_output_tokens_per_second"]) for s in streams
        ),
        "decode_tokens_per_second": (
            sum(stream_decode_rates) if stream_decode_rates else None
        ),
        "per_stream_median_decode_tokens_per_second": (
            statistics.median(stream_decode_rates) if stream_decode_rates else None
        ),
        "decode_token_count": total_decode_tokens,
        "timing_scope": "concurrent host event arrival; aggregate tokens/wall",
        "finish_reason": streams[0]["finish_reason"],
    }


def serialize_stat_value(value: object) -> object:
    if isinstance(value, (list, tuple)):
        return [serialize_stat_value(item) for item in value]
    return value


def stats_record(report) -> dict[str, object]:
    record: dict[str, object] = {}
    for name in STATS_FIELDS:
        if hasattr(report, name):
            record[name] = serialize_stat_value(getattr(report, name))
    return record


def list_delta(after: object, before: object) -> list[float | int] | None:
    if not isinstance(after, list) or not isinstance(before, list):
        return None
    if len(after) != len(before):
        return list(after)
    return [a - b for a, b in zip(after, before)]


def evaluate_parity_report(parity_path: Path | None) -> tuple[str, dict[str, object] | None]:
    if parity_path is None:
        return "not_run", None
    resolved = parity_path.resolve()
    require(resolved.is_file(), f"parity report file does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    require(
        isinstance(payload, dict) and bool(payload),
        f"parity report must be a non-empty JSON object: {resolved}",
    )
    explicit_fail = (
        payload.get("passed") is False
        or payload.get("parity_passed") is False
        or payload.get("status") == "failed"
        or payload.get("independent_full_model_parity") == "failed"
    )
    explicit_pass = not explicit_fail and (
        payload.get("passed") is True
        or payload.get("parity_passed") is True
        or payload.get("status") == "passed"
        or payload.get("independent_full_model_parity") == "passed"
    )
    status = "passed" if explicit_pass else "failed"
    return status, {"path": str(resolved), "status": status, "summary": payload}


def build_load_overrides(
    args: argparse.Namespace,
    *,
    mode: str,
    artifact_path: Path,
    qwen_metallib: Path,
    draft_artifact: Path | None,
    dflash_metallib: Path | None,
) -> dict[str, str]:
    overrides = {
        "artifact": str(artifact_path),
        "qwen_ar_metallib": str(qwen_metallib),
    }
    if mode == "dflash2":
        require(
            draft_artifact is not None and dflash_metallib is not None,
            "dflash2 mode requires --draft-artifact and --dflash-metallib",
        )
        overrides["draft_artifact"] = str(draft_artifact)
        overrides["dflash2_metallib"] = str(dflash_metallib)
    if args.prefix_cache_bytes is not None:
        overrides["prefix_cache_bytes"] = str(args.prefix_cache_bytes)
    if args.pin_prefix_min_tokens is not None:
        overrides["pin_prefix_min_tokens"] = str(args.pin_prefix_min_tokens)
    if args.flash_nax_prefill == "on":
        overrides["flash_nax_prefill"] = "on"
    if args.batch_mode is not None:
        overrides["batch_mode"] = args.batch_mode
    if args.batch_projection_prototype is not None:
        overrides["batch_projection_prototype"] = args.batch_projection_prototype
    if args.attention_split_kv is not None:
        overrides["attention_split_kv"] = args.attention_split_kv
    return overrides


def run_mode(
    native,
    args: argparse.Namespace,
    mode: str,
    resolved_paths: dict[str, Path | None],
) -> dict[str, object]:
    options = native.LoadOptions()
    options.model_dir = str(resolved_paths["model_dir"])
    options.max_context = args.max_context
    options.max_concurrency = args.max_concurrency
    if args.kv_dtype is not None:
        options.kv_dtype = args.kv_dtype
    if args.flash_nax_prefill == "on":
        options.prefill_engine = "flash_nax"
    if args.spec_search is not None and hasattr(options, "spec_search"):
        options.spec_search = args.spec_search
    if args.sampling_contract is not None:
        options.sampling_contract = args.sampling_contract
    options.path_overrides = build_load_overrides(
        args,
        mode=mode,
        artifact_path=resolved_paths["artifact"],
        qwen_metallib=resolved_paths["qwen_metallib"],
        draft_artifact=resolved_paths["draft_artifact"],
        dflash_metallib=resolved_paths["dflash_metallib"],
    )
    if mode == "dflash2":
        options.draft_dir = str(resolved_paths["draft_dir"])
        options.speculative.backend = native.SpeculativeBackend.DFLASH2
        options.speculative.draft_tokens = args.draft_tokens

    active_streams = args.concurrent_requests or args.max_concurrency

    load_started = time.perf_counter()
    engine = native.Engine.create(options)
    load_seconds = time.perf_counter() - load_started
    try:

        def make_request(slot_offset: int) -> object:
            request = native.SubmitRequest()
            request.prompt_tokens = args.prompt_tokens
            request.max_new_tokens = args.max_new_tokens
            request.sampling.temperature = args.temperature
            request.sampling.top_p = args.top_p
            request.sampling.top_k = args.top_k
            request.sampling.seed = (
                args.seed + slot_offset if args.temperature > 0.0 else args.seed
            )
            if args.pin_prefix == "on" and hasattr(request, "pin_prefix"):
                request.pin_prefix = True
            return request

        def measure_request() -> dict[str, object]:
            before = stats_record(engine.stats())
            # Start before submission: the worker may finish before submit returns.
            started = time.perf_counter()
            if active_streams == 1:
                request_id = engine.submit(make_request(0))
                result = collect(
                    native, engine, request_id, args.timeout_seconds, start=started
                )
            else:
                request_ids = [
                    engine.submit(make_request(slot)) for slot in range(active_streams)
                ]
                result = collect_concurrent(
                    native, engine, request_ids, args.timeout_seconds, start=started
                )
            after = stats_record(engine.stats())
            steps = int(after["ar_steps"]) - int(before["ar_steps"])
            timed = int(after["ar_gpu_timed_steps"]) - int(before["ar_gpu_timed_steps"])
            result["native_ar"] = None
            if steps:
                result["native_ar"] = {
                    "steps": steps,
                    "layer_calls": int(after["ar_layer_calls"])
                    - int(before["ar_layer_calls"]),
                    "host_read_bytes": int(after["ar_host_read_bytes"])
                    - int(before["ar_host_read_bytes"]),
                    "short_matrix_calls": int(after["short_matrix_calls"])
                    - int(before["short_matrix_calls"]),
                    "short_lane_calls": int(after["short_lane_calls"])
                    - int(before["short_lane_calls"]),
                    "gpu_timing_complete": timed == steps,
                    "prefill_gpu_ms": float(after["ar_prefill_gpu_ms"])
                    - float(before["ar_prefill_gpu_ms"])
                    if timed == steps
                    else None,
                    "decode_gpu_ms": float(after["ar_decode_gpu_ms"])
                    - float(before["ar_decode_gpu_ms"])
                    if timed == steps
                    else None,
                }
            rounds = int(after["spec_rounds"]) - int(before["spec_rounds"])
            result["native_speculative"] = None
            if rounds:
                complete = (
                    int(after["spec_gpu_timed_rounds"])
                    - int(before["spec_gpu_timed_rounds"])
                    == rounds
                )
                spec_summary: dict[str, object] = {
                    "rounds": rounds,
                    "gpu_timing_complete": complete,
                    "short_matrix_calls": int(after["short_matrix_calls"])
                    - int(before["short_matrix_calls"]),
                    "short_lane_calls": int(after["short_lane_calls"])
                    - int(before["short_lane_calls"]),
                    **{
                        name.removeprefix("spec_"): int(after[name]) - int(before[name])
                        for name in (
                            "spec_target_verify_calls",
                            "spec_replay_calls",
                            "spec_proposal_tokens",
                            "spec_accepted_tokens",
                            "spec_host_read_bytes",
                            "spec_context_rows_projected",
                            "spec_context_rows_reused",
                            "spec_context_rows_launched",
                        )
                    },
                    **{
                        name.removeprefix("spec_"): float(after[name])
                        - float(before[name])
                        if complete
                        else None
                        for name in (
                            "spec_draft_gpu_ms",
                            "spec_verify_gpu_ms",
                            "spec_verify_selector_gpu_ms",
                            "spec_commit_gpu_ms",
                        )
                    },
                }
                if "spec_cost_completed_rounds" in after:
                    spec_summary["cost_completed_rounds"] = int(
                        after["spec_cost_completed_rounds"]
                    ) - int(before["spec_cost_completed_rounds"])
                    spec_summary["cost_committed_tokens"] = int(
                        after["spec_cost_committed_tokens"]
                    ) - int(before["spec_cost_committed_tokens"])
                    spec_summary["cost_round_wall_ms"] = float(
                        after["spec_cost_round_wall_ms"]
                    ) - float(before["spec_cost_round_wall_ms"])
                    spec_summary["cost_phase_wall_ms"] = list_delta(
                        after.get("spec_cost_phase_wall_ms"),
                        before.get("spec_cost_phase_wall_ms"),
                    )
                for sname in ("spec_search_gated_rounds", "spec_search_probe_rounds"):
                    if sname in after:
                        spec_summary[sname.removeprefix("spec_")] = int(
                            after.get(sname, 0)
                        ) - int(before.get(sname, 0))
                result["native_speculative"] = spec_summary

            batch_subs = int(after.get("batch_submissions", 0)) - int(
                before.get("batch_submissions", 0)
            )
            shared_calls = int(after.get("shared_head_projection_calls", 0)) - int(
                before.get("shared_head_projection_calls", 0)
            )
            packed_calls = int(after.get("packed_batch_calls", 0)) - int(
                before.get("packed_batch_calls", 0)
            )
            result["native_batch"] = None
            if batch_subs or shared_calls or packed_calls:
                result["native_batch"] = {
                    "batch_submissions": batch_subs,
                    "batch_rows": int(after.get("batch_rows", 0))
                    - int(before.get("batch_rows", 0)),
                    "batch_gpu_ms": float(after.get("batch_gpu_ms", 0.0))
                    - float(before.get("batch_gpu_ms", 0.0)),
                    "batch_sampler_icb_replays": int(
                        after.get("batch_sampler_icb_replays", 0)
                    )
                    - int(before.get("batch_sampler_icb_replays", 0)),
                    "shared_head_projection_calls": shared_calls,
                    "shared_head_projection_rows": int(
                        after.get("shared_head_projection_rows", 0)
                    )
                    - int(before.get("shared_head_projection_rows", 0)),
                    "shared_head_projection_gpu_ms": float(
                        after.get("shared_head_projection_gpu_ms", 0.0)
                    )
                    - float(before.get("shared_head_projection_gpu_ms", 0.0)),
                    "packed_batch_calls": packed_calls,
                    "packed_batch_tokens": int(after.get("packed_batch_tokens", 0))
                    - int(before.get("packed_batch_tokens", 0)),
                    "contended_prefill_chunks": int(
                        after.get("contended_prefill_chunks", 0)
                    )
                    - int(before.get("contended_prefill_chunks", 0)),
                }

            prefix_events = sum(
                int(after.get(k, 0)) - int(before.get(k, 0))
                for k in (
                    "prefix_hits",
                    "prefix_partial_hits",
                    "prefix_misses",
                    "prefix_shared_hits",
                )
            )
            result["native_prefix_cache"] = None
            if prefix_events or (args.prefix_cache_bytes or 0) > 0:
                result["native_prefix_cache"] = {
                    key: int(after.get(key, 0)) - int(before.get(key, 0))
                    for key in (
                        "prefix_hits",
                        "prefix_partial_hits",
                        "prefix_misses",
                        "prefix_evictions",
                        "prefix_restored_tokens",
                        "prefix_shared_hits",
                    )
                }
                result["native_prefix_cache"]["prefix_pinned_entries"] = int(
                    after.get("prefix_pinned_entries", 0)
                )
            return result

        for _ in range(args.warmups):
            measure_request()
        runs = [measure_request() for _ in range(args.repeats)]
        throughputs = [float(run["request_output_tokens_per_second"]) for run in runs]
        decode_rates = [
            float(run["decode_tokens_per_second"])
            for run in runs
            if run["decode_tokens_per_second"] is not None
        ]
        ttft = [float(run["ttft_seconds"]) for run in runs]
        return {
            "mode": mode,
            "max_concurrency": args.max_concurrency,
            "concurrent_requests": active_streams,
            "load_seconds": load_seconds,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "runs": runs,
            "median_request_output_tokens_per_second": statistics.median(throughputs),
            "median_decode_tokens_per_second": (
                statistics.median(decode_rates) if decode_rates else None
            ),
            "median_ttft_seconds": statistics.median(ttft),
            "stats": stats_record(engine.stats()),
        }
    finally:
        engine.shutdown()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preset",
        default=None,
        help=(
            "Target model/artifact preset: "
            + ", ".join(sorted(ARTIFACT_PRESETS))
            + " (or aliases 27b-oq4e, 27b-mxfp4, 35b-oq4e, 35b-mxfp4)."
        ),
    )
    parser.add_argument("--extension-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--artifact", type=Path, default=None)
    parser.add_argument("--qwen-metallib", type=Path, required=True)
    parser.add_argument("--draft-dir", type=Path, default=None)
    parser.add_argument("--draft-artifact", type=Path, default=None)
    parser.add_argument("--dflash-metallib", type=Path, default=None)
    parser.add_argument(
        "--mode", choices=("ar", "dflash2", "both"), default=None
    )
    parser.add_argument("--prompt-tokens", type=parse_tokens, default=[1, 7, 3, 9, 2])
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context", type=int, default=32768)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--concurrent-requests", type=int, default=None)
    parser.add_argument(
        "--sampling-contract",
        choices=("legacy_rejection_v1", "position_coupled_v1"),
        default=None,
    )
    parser.add_argument("--prefix-cache-bytes", type=int, default=None)
    parser.add_argument("--batch-mode", choices=("off", "on"), default=None)
    parser.add_argument(
        "--batch-projection-prototype", choices=("off", "on"), default=None
    )
    parser.add_argument("--attention-split-kv", choices=("off", "on"), default=None)
    parser.add_argument(
        "--kv-dtype", choices=("bf16", "affine4_g128"), default=None
    )
    parser.add_argument(
        "--flash-nax-prefill", choices=("off", "on"), default=None
    )
    parser.add_argument(
        "--spec-search", choices=("auto", "on", "off"), default=None
    )
    parser.add_argument(
        "--pin-prefix",
        nargs="?",
        const="on",
        choices=("off", "on"),
        default=None,
    )
    parser.add_argument("--pin-prefix-min-tokens", type=int, default=None)
    parser.add_argument("--draft-tokens", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument("--parity-report", type=Path, default=None)
    parser.add_argument(
        "--min-target-artifact-bytes",
        type=int,
        default=10 * (1 << 30),
        help="Minimum .omlxm5 byte size required to qualify as a real target model.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-non-target", action="store_true")
    args = parser.parse_args()

    preset_name = normalize_preset(args.preset)
    preset = ARTIFACT_PRESETS[preset_name] if preset_name else None

    model_dir = args.model_dir or (
        ROOT / str(preset["model_dir"]) if preset else None
    )
    artifact = args.artifact or (
        ROOT / str(preset["artifact"]) if preset else None
    )
    draft_dir = args.draft_dir or (
        ROOT / str(preset["draft_dir"]) if preset else ROOT / DEFAULT_DRAFT_DIR
    )
    draft_artifact = args.draft_artifact or (
        ROOT / str(preset["draft_artifact"])
        if preset
        else ROOT / DEFAULT_DRAFT_ARTIFACT
    )
    require(
        model_dir is not None and artifact is not None,
        "provide --preset or both --model-dir and --artifact",
    )

    if args.mode is None:
        if preset is not None and not bool(preset["draft_supported"]):
            effective_mode = "ar"
        else:
            effective_mode = "both"
    else:
        effective_mode = args.mode
    if preset is not None and not bool(preset["draft_supported"]):
        require(
            effective_mode == "ar",
            f"preset {preset_name} does not have a matching DFlash2 companion; use --mode ar",
        )

    require(args.max_new_tokens > 0 and args.max_context >= 1024, "invalid limits")
    require(1 <= args.max_concurrency <= 8, "max-concurrency must be 1..8")
    if args.concurrent_requests is not None:
        require(
            1 <= args.concurrent_requests <= args.max_concurrency,
            "concurrent-requests must be in 1..max-concurrency",
        )
    if args.prefix_cache_bytes is not None:
        require(
            0 <= args.prefix_cache_bytes <= (32 << 30),
            "prefix-cache-bytes must be in 0..32GiB",
        )
    if args.pin_prefix_min_tokens is not None:
        require(
            args.pin_prefix_min_tokens >= 0,
            "pin-prefix-min-tokens must be >= 0",
        )
    if args.batch_mode == "on":
        require(
            args.max_concurrency >= 2,
            "batch-mode=on requires --max-concurrency >= 2",
        )
    if args.batch_projection_prototype == "on":
        require(
            args.batch_mode == "on"
            and args.sampling_contract == "position_coupled_v1",
            "batch-projection-prototype=on requires --batch-mode on and --sampling-contract position_coupled_v1",
        )
    require(args.draft_tokens in range(1, 8), "draft tokens must be 1..7")
    require(args.repeats > 0 and args.warmups >= 0, "invalid repeat counts")
    require(
        args.timeout_seconds > 0 and math.isfinite(args.timeout_seconds),
        "invalid timeout",
    )
    require(args.min_target_artifact_bytes > 0, "invalid min-target-artifact-bytes")

    parity_status, parity_record = evaluate_parity_report(args.parity_report)

    hardware = hardware_facts()
    target = bool(hardware["target_device"])
    if not target and not args.allow_non_target and not args.plan_only:
        print("full-model execution requires an Apple M5 Ultra host", file=sys.stderr)
        return 2
    modes = ["ar", "dflash2"] if effective_mode == "both" else [effective_mode]
    if "dflash2" in modes:
        require(
            args.dflash_metallib is not None,
            "--dflash-metallib is required when mode includes dflash2",
        )

    resolved_paths: dict[str, Path | None] = {
        "extension_dir": args.extension_dir.resolve(),
        "model_dir": model_dir.resolve(),
        "artifact": artifact.resolve(),
        "qwen_metallib": args.qwen_metallib.resolve(),
        "draft_dir": draft_dir.resolve() if draft_dir else None,
        "draft_artifact": draft_artifact.resolve() if draft_artifact else None,
        "dflash_metallib": (
            args.dflash_metallib.resolve() if args.dflash_metallib else None
        ),
    }
    paths = {
        key: (str(val) if val is not None else None)
        for key, val in resolved_paths.items()
    }
    result: dict[str, object] = {
        "schema": "omlx-m5-ultra-full-model-v2",
        "hardware": hardware,
        "preset": preset_name,
        "available_presets": sorted(ARTIFACT_PRESETS),
        "measurement_scope": (
            "M5 Ultra measurement"
            if target
            else "non-target full-model harness validation; not M5 Ultra performance"
        ),
        "plan_only": args.plan_only,
        "modes": modes,
        "paths": paths,
        "engine_options": {
            "max_context": args.max_context,
            "max_concurrency": args.max_concurrency,
            "concurrent_requests": args.concurrent_requests or args.max_concurrency,
            "sampling_contract": args.sampling_contract or "legacy_rejection_v1",
            "prefix_cache_bytes": args.prefix_cache_bytes or 0,
            "batch_mode": args.batch_mode or "off",
            "batch_projection_prototype": args.batch_projection_prototype or "off",
            "attention_split_kv": args.attention_split_kv or "off",
            "kv_dtype": args.kv_dtype or "bf16",
            "flash_nax_prefill": args.flash_nax_prefill or "off",
            "spec_search": args.spec_search or "auto",
            "pin_prefix": args.pin_prefix or "off",
            "pin_prefix_min_tokens": args.pin_prefix_min_tokens or 0,
        },
        "request": {
            "prompt_tokens": args.prompt_tokens,
            "max_new_tokens": args.max_new_tokens,
            "max_context": args.max_context,
            "max_concurrency": args.max_concurrency,
            "concurrent_requests": args.concurrent_requests or args.max_concurrency,
            "sampling_contract": args.sampling_contract or "legacy_rejection_v1",
            "prefix_cache_bytes": args.prefix_cache_bytes or 0,
            "batch_mode": args.batch_mode or "off",
            "attention_split_kv": args.attention_split_kv or "off",
            "kv_dtype": args.kv_dtype or "bf16",
            "flash_nax_prefill": args.flash_nax_prefill or "off",
            "spec_search": args.spec_search or "auto",
            "pin_prefix": args.pin_prefix or "off",
            "pin_prefix_min_tokens": args.pin_prefix_min_tokens or 0,
            "draft_tokens": args.draft_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "seed": args.seed,
            "warmups": args.warmups,
            "repeats": args.repeats,
        },
        "parity_report": parity_record,
        "gate": {
            "h2_complete": False,
            "independent_full_model_parity": parity_status,
            "reason": "plan_only" if args.plan_only else "execution_pending",
        },
    }
    if args.plan_only:
        atomic_json(args.output.resolve(), result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    required_path_keys = ["extension_dir", "model_dir", "artifact", "qwen_metallib"]
    if "dflash2" in modes:
        required_path_keys.extend(["draft_dir", "draft_artifact", "dflash_metallib"])
    for key in required_path_keys:
        path = resolved_paths[key]
        require(
            path is not None and path.exists(),
            f"required full-model path is missing: {path}",
        )
    sys.path.insert(0, str(resolved_paths["extension_dir"]))
    import omlx_m5_native as native

    mode_results = [run_mode(native, args, mode, resolved_paths) for mode in modes]
    result["results"] = mode_results

    if len(mode_results) == 2 and [m["mode"] for m in mode_results] == [
        "ar",
        "dflash2",
    ]:
        ar_runs = mode_results[0]["runs"]
        df_runs = mode_results[1]["runs"]
        compared = min(len(ar_runs), len(df_runs))
        exact_all = compared > 0
        matched_tokens = 0
        total_tokens = 0
        first_mismatch: dict[str, object] | None = None
        for r_idx in range(compared):
            ar_r = ar_runs[r_idx]
            df_r = df_runs[r_idx]
            if "streams" in ar_r and "streams" in df_r:
                ar_streams = ar_r["streams"]
                df_streams = df_r["streams"]
                if len(ar_streams) != len(df_streams):
                    exact_all = False
                for s_idx in range(min(len(ar_streams), len(df_streams))):
                    a_toks = ar_streams[s_idx]["tokens"]
                    d_toks = df_streams[s_idx]["tokens"]
                    total_tokens += max(len(a_toks), len(d_toks))
                    for t_idx in range(min(len(a_toks), len(d_toks))):
                        if a_toks[t_idx] == d_toks[t_idx]:
                            matched_tokens += 1
                        elif first_mismatch is None:
                            first_mismatch = {
                                "run": r_idx,
                                "stream": s_idx,
                                "pos": t_idx,
                                "ar": a_toks[t_idx],
                                "dflash2": d_toks[t_idx],
                            }
                    if a_toks != d_toks:
                        exact_all = False
            else:
                a_toks = ar_r.get("tokens", [])
                d_toks = df_r.get("tokens", [])
                total_tokens += max(len(a_toks), len(d_toks))
                for t_idx in range(min(len(a_toks), len(d_toks))):
                    if a_toks[t_idx] == d_toks[t_idx]:
                        matched_tokens += 1
                    elif first_mismatch is None:
                        first_mismatch = {
                            "run": r_idx,
                            "pos": t_idx,
                            "ar": a_toks[t_idx],
                            "dflash2": d_toks[t_idx],
                        }
                if a_toks != d_toks:
                    exact_all = False
        result["inline_ar_dflash2_parity"] = {
            "exact_match": exact_all,
            "compared_runs": compared,
            "matched_tokens": matched_tokens,
            "total_tokens": total_tokens,
            "first_mismatch": first_mismatch,
        }

    artifact_bytes = resolved_paths["artifact"].stat().st_size
    draft_artifact_bytes = (
        resolved_paths["draft_artifact"].stat().st_size
        if resolved_paths["draft_artifact"] is not None
        and resolved_paths["draft_artifact"].exists()
        else 0
    )
    result["artifact_bytes"] = artifact_bytes
    result["draft_artifact_bytes"] = draft_artifact_bytes

    real_target_artifact = artifact_bytes >= args.min_target_artifact_bytes
    real_draft_artifact = ("dflash2" not in modes) or (
        draft_artifact_bytes >= (1 << 30)
    )
    real_model_artifacts = bool(real_target_artifact and real_draft_artifact)
    measurements_complete = bool(mode_results) and all(
        len(item["runs"]) == args.repeats
        and int(item["stats"].get("failed", 0) or 0) == 0
        and "fake" not in str(item["stats"].get("backend_kind", ""))
        for item in mode_results
    )
    sample_gate_met = bool(args.warmups >= 3 and args.repeats >= 10)
    native_full_model_complete = bool(
        target and real_model_artifacts and measurements_complete
    )
    h2_complete = bool(
        native_full_model_complete
        and parity_status == "passed"
        and sample_gate_met
    )

    reasons: list[str] = []
    if not target:
        reasons.append("non-target host")
    if not real_model_artifacts:
        reasons.append("control/sub-threshold artifact size")
    if not measurements_complete:
        reasons.append("incomplete native mode runs")
    if not sample_gate_met:
        reasons.append(f"warmups/repeats ({args.warmups}/{args.repeats}) below 3/10")
    if parity_status != "passed":
        reasons.append(f"independent_full_model_parity={parity_status}")

    result["gate"] = {
        "h2_complete": h2_complete,
        "native_execution_complete": measurements_complete,
        "native_full_model_complete": native_full_model_complete,
        "target_device": target,
        "real_model_artifacts": real_model_artifacts,
        "sample_gate_met": sample_gate_met,
        "execution_kinds": sorted(
            {str(item["stats"]["model_execution_kind"]) for item in mode_results}
        ),
        "sampling_contracts": sorted(
            {
                str(item["stats"].get("sampling_contract_id", "legacy_rejection_v1"))
                for item in mode_results
            }
        ),
        "independent_full_model_parity": parity_status,
        "reason": "complete" if h2_complete else "; ".join(reasons),
    }
    atomic_json(args.output.resolve(), result)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
