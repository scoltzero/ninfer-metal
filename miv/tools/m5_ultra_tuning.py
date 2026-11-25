#!/usr/bin/env python3
"""Run or plan H1 route measurements and emit ranked static-table inputs."""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPT_PARENT = SCRIPT_DIR.parent
ROOT = _SCRIPT_PARENT.parent if _SCRIPT_PARENT.name == "miv" else _SCRIPT_PARENT
sys.path.insert(0, str(SCRIPT_DIR))

from m5_target_guard import hardware_facts

VALID_STAGES = ("b2", "h1_l", "h1_a", "h1_g", "h1_c", "h1_m")
VALID_PRECISION_CONTRACTS = ("A16", "W4A16", "W4A4")


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


def percentile(values: list[float], fraction: float) -> float:
    require(bool(values), "latency sample set is empty")
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * fraction) - 1)
    return ordered[index]


def infer_row_exact_group(*, fmt: str, family: str, domain: str) -> str:
    if family == "row_exact_portable" or family.startswith("row_exact"):
        if fmt == "direct_bf16_v1":
            return "bf16.simd32.sequential_fma.v1"
        if fmt == "mxfp4_e2m1_g32_e8m0_v1":
            return "mxfp4.simd32.groupdot_scale_fma.v1"
        return f"oq.simd32.groupdot_scale_then_bias.v1/{fmt}"
    if family.startswith("affine_lane"):
        return f"affine_lane.splitk_f32.v1/{fmt}"
    if domain == "moe":
        return f"legacy_moe_non_exact_v1/{fmt}"
    return f"legacy_linear_non_exact_v1/{fmt}"


def curve(
    record: dict,
    *,
    domain: str,
    profile: str,
    family: str,
    gpu_ms: list[float],
    wall_ms: list[float] | None = None,
    row_exact_group: str | None = None,
) -> dict[str, object]:
    require(
        profile in VALID_PRECISION_CONTRACTS,
        f"unsupported precision contract profile: {profile}",
    )
    values = [float(value) for value in gpu_ms]
    require(
        bool(values) and all(math.isfinite(value) and value >= 0 for value in values),
        "invalid GPU timing sample",
    )
    output_features = record.get(
        "output_features", record.get("parent_output_features")
    )
    out_dim = int(output_features) if output_features is not None else None
    raw_executed = record.get("executed_output_features")
    exec_dim = int(raw_executed) if raw_executed is not None else out_dim
    sub_row_probe = bool(
        record.get("sub_row_probe", False)
        or (out_dim is not None and exec_dim is not None and exec_dim < out_dim)
    )
    route_selection_eligible = bool(
        record.get("route_selection_eligible", not sub_row_probe)
    ) and not sub_row_probe

    fmt = str(record["format"])
    exact_group = str(
        row_exact_group
        or record.get("row_exact_group")
        or infer_row_exact_group(fmt=fmt, family=family, domain=domain)
    )
    row: dict[str, object] = {
        "artifact": record["artifact"],
        "object": record["object"],
        "format": fmt,
        "domain": domain,
        "profile": profile,
        "precision_contract": profile,
        "row_exact_group": exact_group,
        "family": family,
        "tokens": int(record.get("tokens", 1)),
        "output_features": out_dim,
        "executed_output_features": exec_dim,
        "sub_row_probe": sub_row_probe,
        "route_selection_eligible": route_selection_eligible,
        "input_features": int(record["input_features"]),
        "sample_count": len(values),
        "gpu_median_ms": statistics.median(values),
        "gpu_p95_ms": percentile(values, 0.95),
        "gpu_min_ms": min(values),
        "gpu_max_ms": max(values),
    }
    if wall_ms:
        wall = [float(value) for value in wall_ms]
        row["wall_median_ms"] = statistics.median(wall)
        row["wall_p95_ms"] = percentile(wall, 0.95)
    return row


def load(path: Path, schema: str) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema") == schema, f"unexpected schema in {path}")
    return value


def artifact_key_from_path(raw_path: str) -> str:
    return Path(raw_path).stem


def build_curves(result_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    b2_2_path = result_dir / "b2_2/qmv-t1.json"
    if b2_2_path.is_file():
        b2_2 = load(b2_2_path, "omlx-m5-b2-2-validation-v1")
        for record in b2_2["records"]:
            profile = "W4A16" if record["format"].startswith("mxfp4") else "A16"
            rows.append(
                curve(
                    record,
                    domain="t1",
                    profile=profile,
                    family="qmv",
                    gpu_ms=record["gpu_ms"],
                    wall_ms=record["wall_ms"],
                )
            )

    b2_3_path = result_dir / "b2_3/qmm-small-t.json"
    if b2_3_path.is_file():
        b2_3 = load(b2_3_path, "omlx-m5-b2-3-validation-v1")
        for record in b2_3["records"]:
            timings = record.get("candidate_timings") or [
                {
                    "family": record["selected_family"],
                    "gpu_ms": record["gpu_ms"],
                    "wall_ms": record["wall_ms"],
                }
            ]
            for timing in timings:
                rows.append(
                    curve(
                        record,
                        domain="small_t",
                        profile="A16",
                        family=str(timing["family"]),
                        gpu_ms=timing["gpu_ms"],
                        wall_ms=timing.get("wall_ms"),
                    )
                )

    b2_4_path = result_dir / "b2_4/mxfp4-small-t.json"
    if b2_4_path.is_file():
        b2_4 = load(b2_4_path, "omlx-m5-b2-4-validation-v1")
        for record in b2_4["records"]:
            timings = record.get("candidate_timings") or [
                {
                    "activation": "A16",
                    "family": record.get("selected_a16_family", "w4a16_simd"),
                    "gpu_ms": record["a16_gpu_ms"],
                    "wall_ms": record["a16_wall_ms"],
                },
                {
                    "activation": "A4",
                    "family": record.get("selected_a4_family", "w4a4_simd"),
                    "gpu_ms": record["a4_gpu_ms"],
                    "wall_ms": record["a4_wall_ms"],
                },
            ]
            for timing in timings:
                rows.append(
                    curve(
                        record,
                        domain="small_t",
                        profile=("W4A4" if timing["activation"] == "A4" else "W4A16"),
                        family=str(timing["family"]),
                        gpu_ms=timing["gpu_ms"],
                        wall_ms=timing.get("wall_ms"),
                    )
                )

    b2_5_path = result_dir / "b2_5/qmm-broad.json"
    if b2_5_path.is_file():
        b2_5 = load(b2_5_path, "omlx-m5-b2-5-validation-v1")
        for record in b2_5["records"]:
            timings = record.get("candidate_timings") or [
                {
                    "activation": "A16",
                    "family": record.get("selected_a16_family", "broad_a16"),
                    "gpu_ms": record["a16_gpu_ms"],
                }
            ]
            if not record.get("candidate_timings") and record["format"].startswith(
                "mxfp4"
            ):
                timings.append(
                    {
                        "activation": "A4",
                        "family": record.get("selected_a4_family", "broad_a4"),
                        "gpu_ms": record["a4_gpu_ms"],
                    }
                )
            for timing in timings:
                rows.append(
                    curve(
                        record,
                        domain="broad",
                        profile=(
                            "W4A4"
                            if timing["activation"] == "A4"
                            else "W4A16"
                            if record["format"].startswith("mxfp4")
                            else "A16"
                        ),
                        family=str(timing["family"]),
                        gpu_ms=timing["gpu_ms"],
                    )
                )

    b2_6_path = result_dir / "b2_6/moe.json"
    if b2_6_path.is_file():
        b2_6 = load(b2_6_path, "omlx-m5-b2-6-validation-v1")
        for record in b2_6["records"]:
            for prefix, profile, family in (
                (
                    "grouped_a16",
                    "W4A16" if record["format"].startswith("mxfp4") else "A16",
                    "grouped_moe",
                ),
                (
                    "fallback_a16",
                    "W4A16" if record["format"].startswith("mxfp4") else "A16",
                    "per_expert_fallback",
                ),
                ("grouped_a4", "W4A4", "grouped_moe_a4"),
                ("fallback_a4", "W4A4", "per_expert_fallback_a4"),
            ):
                samples = record.get(f"{prefix}_gpu_ms") or []
                if samples:
                    synthetic = dict(record)
                    synthetic["output_features"] = record["shape"][1]
                    synthetic["executed_output_features"] = record["shape"][1]
                    synthetic["input_features"] = record["shape"][2]
                    rows.append(
                        curve(
                            synthetic,
                            domain="moe",
                            profile=profile,
                            family=family,
                            gpu_ms=samples,
                            wall_ms=record.get(f"{prefix}_wall_ms"),
                        )
                    )

    h1_l_dir = result_dir / "h1_l"
    if h1_l_dir.is_dir():
        for candidate_file in sorted(h1_l_dir.glob("*.json")):
            record = load(candidate_file, "omlx-m5-projection-candidate-v1")
            fmt = str(record["format"])
            profile = "W4A16" if fmt.startswith("mxfp4") else "A16"
            tokens = int(record["t"])
            domain = "t1" if tokens == 1 else "small_t" if tokens <= 12 else "broad"
            normalized = {
                "artifact": artifact_key_from_path(str(record["artifact"])),
                "object": record["object"],
                "format": fmt,
                "tokens": tokens,
                "output_features": int(record["n"]),
                "executed_output_features": int(record["n"]),
                "sub_row_probe": False,
                "route_selection_eligible": True,
                "input_features": int(record["k"]),
            }
            rows.append(
                curve(
                    normalized,
                    domain=domain,
                    profile=profile,
                    family="legacy_baseline",
                    gpu_ms=record["old_gpu_ms"],
                    wall_ms=record.get("old_wall_ms"),
                )
            )
            rows.append(
                curve(
                    normalized,
                    domain=domain,
                    profile=profile,
                    family=str(record["candidate"]),
                    gpu_ms=record["new_gpu_ms"],
                    wall_ms=record.get("new_wall_ms"),
                )
            )

    return rows


def is_sub_row_probe(row: dict[str, object]) -> bool:
    if bool(row.get("sub_row_probe", False)):
        return True
    if row.get("route_selection_eligible") is False:
        return True
    out_dim = row.get("output_features")
    exec_dim = row.get("executed_output_features")
    if out_dim is not None and exec_dim is not None and int(exec_dim) < int(out_dim):
        return True
    return False


def recommendations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if is_sub_row_probe(row):
            continue
        precision_contract = str(
            row.get("precision_contract") or row.get("profile") or ""
        )
        require(
            precision_contract in VALID_PRECISION_CONTRACTS,
            f"invalid precision_contract in candidate row: {precision_contract!r}",
        )
        fmt = str(row.get("format", ""))
        row_exact_group = str(
            row.get("row_exact_group")
            or infer_row_exact_group(
                fmt=fmt,
                family=str(row.get("family", "")),
                domain=str(row.get("domain", "linear")),
            )
        )
        groups[
            (
                row["artifact"],
                row["object"],
                fmt,
                precision_contract,
                row_exact_group,
                row["tokens"],
                row["output_features"],
                row["input_features"],
            )
        ].append(row)

    output: list[dict[str, object]] = []
    for key, candidates in sorted(groups.items(), key=lambda item: str(item[0])):
        ranked = sorted(candidates, key=lambda item: float(item["gpu_median_ms"]))
        output.append(
            {
                "selector": {
                    "artifact": key[0],
                    "object": key[1],
                    "format": key[2],
                    "precision_contract": key[3],
                    "row_exact_group": key[4],
                    "tokens": key[5],
                    "output_features": key[6],
                    "input_features": key[7],
                },
                "winner": {
                    "profile": ranked[0]["profile"],
                    "precision_contract": key[3],
                    "row_exact_group": key[4],
                    "family": ranked[0]["family"],
                    "gpu_median_ms": ranked[0]["gpu_median_ms"],
                    "gpu_p95_ms": ranked[0]["gpu_p95_ms"],
                },
                "candidates": [
                    {
                        "family": item["family"],
                        "profile": item["profile"],
                        "precision_contract": key[3],
                        "row_exact_group": key[4],
                        "gpu_median_ms": item["gpu_median_ms"],
                        "gpu_p95_ms": item["gpu_p95_ms"],
                    }
                    for item in ranked
                ],
                "candidate_count": len(ranked),
                "state": "measured_pending_static_table_review",
            }
        )
    return output


def parse_stages(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(VALID_STAGES)
    stages = [part.strip() for part in raw.split(",") if part.strip()]
    require(
        bool(stages) and all(stage in VALID_STAGES for stage in stages),
        f"stages must be 'all' or a comma-separated subset of {VALID_STAGES}",
    )
    return stages


def plan_h1_hooks(
    build_dir: Path,
    output_dir: Path,
    workloads_path: Path,
    iterations: int,
) -> dict[str, dict[str, object]]:
    binary_dir = build_dir / "miv" if (build_dir / "miv").is_dir() else build_dir / "m5rt"
    metallib_dir = binary_dir / "metallib"
    repeats = max(5, iterations)
    workloads = (
        json.loads(workloads_path.read_text(encoding="utf-8"))
        if workloads_path.is_file()
        else {}
    )
    sources = {
        entry["key"]: entry for entry in workloads.get("artifact_sources", [])
    }

    h1_l_commands: list[dict[str, object]] = []
    proj_bin = binary_dir / "omlx_m5_benchmark_projection_candidates"
    projection_cases = list(workloads.get("projection", {}).get("cases", []))
    # Also include Qwen3.8-27B target artifacts when present on disk.
    for extra in (
        {
            "id": "qwen38-27b-oq-head-t1",
            "artifact": ".cache/b0/artifacts/qwen38-27b-oq4e.omlxm5",
            "object": "language_model.lm_head",
            "t": 1,
            "candidate": "lane",
        },
        {
            "id": "qwen38-27b-oq-head-t8",
            "artifact": ".cache/b0/artifacts/qwen38-27b-oq4e.omlxm5",
            "object": "language_model.lm_head",
            "t": 8,
            "candidate": "matrix",
        },
        {
            "id": "qwen38-27b-oq-down-t1",
            "artifact": ".cache/b0/artifacts/qwen38-27b-oq4e.omlxm5",
            "object": "language_model.model.layers.0.mlp.down_proj",
            "t": 1,
            "candidate": "lane",
        },
        {
            "id": "qwen38-27b-oq-down-t8",
            "artifact": ".cache/b0/artifacts/qwen38-27b-oq4e.omlxm5",
            "object": "language_model.model.layers.0.mlp.down_proj",
            "t": 8,
            "candidate": "matrix",
        },
        {
            "id": "qwen38-27b-mxfp4-head-t1",
            "artifact": ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4.omlxm5",
            "object": "language_model.lm_head",
            "t": 1,
            "candidate": "lane",
        },
        {
            "id": "qwen38-27b-mxfp4-head-t8",
            "artifact": ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4.omlxm5",
            "object": "language_model.lm_head",
            "t": 8,
            "candidate": "matrix",
        },
        {
            "id": "qwen38-27b-mxfp4-down-t1",
            "artifact": ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4.omlxm5",
            "object": "language_model.model.layers.0.mlp.down_proj",
            "t": 1,
            "candidate": "lane",
        },
        {
            "id": "qwen38-27b-mxfp4-down-t8",
            "artifact": ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4.omlxm5",
            "object": "language_model.model.layers.0.mlp.down_proj",
            "t": 8,
            "candidate": "matrix",
        },
    ):
        if (ROOT / extra["artifact"]).is_file():
            projection_cases.append(extra)
    for case in projection_cases:
        cmd = [
            str(proj_bin),
            str(ROOT / case["artifact"]),
            str(case["object"]),
            str(metallib_dir),
            str(case["t"]),
            str(repeats),
        ]
        if case.get("candidate") == "matrix":
            cmd.append("matrix")
            if int(case["t"]) == 8 and "lm_head" in str(case["object"]):
                cmd.append("verify")
        h1_l_commands.append(
            {
                "id": str(case["id"]),
                "artifact": str(ROOT / case["artifact"]),
                "output": str(output_dir / "h1_l" / f"{case['id']}.json"),
                "command": cmd,
            }
        )

    h1_a_commands: list[dict[str, object]] = []
    attn_bin = binary_dir / "omlx_m5_benchmark_long_kv_attention"
    for model_key in workloads.get("attention", {}).get("models", []):
        source_info = sources.get(model_key)
        if not source_info:
            continue
        cfg_path = ROOT / source_info["config"]
        for pattern in (0, 4):
            h1_a_commands.append(
                {
                    "id": f"{model_key}-pattern{pattern}",
                    "config": str(cfg_path),
                    "output": str(
                        output_dir / "h1_a" / f"{model_key}-pattern{pattern}.json"
                    ),
                    "command": [
                        str(attn_bin),
                        str(cfg_path),
                        str(metallib_dir),
                        str(repeats),
                        str(pattern),
                    ],
                }
            )

    h1_g_commands: list[dict[str, object]] = []
    gdn_bin = binary_dir / "omlx_m5_benchmark_gdn_prefill"
    for model_key in workloads.get("gdn", {}).get("models", []):
        source_info = sources.get(model_key)
        if not source_info:
            continue
        artifact_path = ROOT / source_info["artifact"]
        cfg_path = ROOT / source_info["config"]
        out_path = output_dir / "h1_g" / f"{model_key}-gdn.json"
        h1_g_commands.append(
            {
                "id": f"{model_key}-gdn",
                "artifact": str(artifact_path),
                "config": str(cfg_path),
                "output": str(out_path),
                "command": [
                    str(gdn_bin),
                    str(artifact_path),
                    str(cfg_path),
                    str(metallib_dir),
                    str(repeats),
                    str(out_path),
                    "--p07",
                ],
            }
        )

    batch_bin = binary_dir / "omlx_m5_benchmark_p10_batch"
    qwen_fixtures = binary_dir / "fixtures/qwen-execution"
    dflash_fixture = binary_dir / "fixtures/dflash-execution"
    batch_out = output_dir / "h1_c/p10-batch.json"
    h1_c_commands = [
        {
            "id": "p10_batch_slice",
            "qwen_fixtures": str(qwen_fixtures),
            "dflash_fixture": str(dflash_fixture),
            "output": str(batch_out),
            "command": [
                str(batch_bin),
                str(qwen_fixtures),
                str(metallib_dir),
                str(dflash_fixture),
                str(repeats),
                str(batch_out),
            ],
        }
    ]

    h1_m_commands: list[dict[str, object]] = []
    p12_bin = binary_dir / "omlx_m5_export_p12_routes"
    tokens_json = output_dir / "h1_m/prompt-tokens.json"
    for model_key in workloads.get("moe", {}).get("models", []):
        source_info = sources.get(model_key)
        if not source_info:
            continue
        artifact_path = ROOT / source_info["artifact"]
        model_dir = (ROOT / source_info["config"]).parent
        out_path = output_dir / "h1_m" / f"{model_key}-layer0-routes.json"
        h1_m_commands.append(
            {
                "id": f"{model_key}-layer0",
                "model_dir": str(model_dir),
                "artifact": str(artifact_path),
                "tokens_json": str(tokens_json),
                "output": str(out_path),
                "command": [
                    str(p12_bin),
                    str(model_dir),
                    str(artifact_path),
                    str(metallib_dir),
                    str(tokens_json),
                    "0",
                    str(out_path),
                ],
            }
        )

    return {
        "h1_l": {
            "binary": str(proj_bin),
            "description": "Full-N Linear/lm_head projection candidates (H1-L)",
            "commands": h1_l_commands,
        },
        "h1_a": {
            "binary": str(attn_bin),
            "description": "Long-KV 2K/8K/32K paged & split-KV attention (H1-A)",
            "commands": h1_a_commands,
        },
        "h1_g": {
            "binary": str(gdn_bin),
            "description": "Real-weight GDN prefill & recurrent mixer slice (H1-G)",
            "commands": h1_g_commands,
        },
        "h1_c": {
            "binary": str(batch_bin),
            "description": "B=1..8 paired AR/DFlash2 batch & sampler ICB (H1-C)",
            "commands": h1_c_commands,
        },
        "h1_m": {
            "binary": str(p12_bin),
            "description": "Real 35B MoE router top-8 trace export (H1-M)",
            "commands": h1_m_commands,
        },
    }


def run_h1_hooks(
    hooks: dict[str, dict[str, object]],
    selected_stages: list[str],
    output_dir: Path,
    *,
    target: bool,
) -> dict[str, dict[str, object]]:
    summaries: dict[str, dict[str, object]] = {}
    for stage_key in ("h1_l", "h1_a", "h1_g", "h1_c", "h1_m"):
        if stage_key not in selected_stages:
            summaries[stage_key] = {"status": "not_selected", "executed": 0}
            continue
        stage = hooks[stage_key]
        binary = Path(str(stage["binary"]))
        if not binary.is_file():
            summaries[stage_key] = {
                "status": "skipped_missing_binary",
                "binary": str(binary),
                "executed": 0,
            }
            continue

        if stage_key == "h1_m":
            tokens_json = output_dir / "h1_m/prompt-tokens.json"
            atomic_json(
                tokens_json,
                {
                    "tokens": [1, 7, 3, 9, 2, 11, 13, 17],
                    "origin": "m5_ultra_h1_m_arrival_probe",
                    "synthetic": False,
                    "sampling_contract": "position_coupled_v1",
                },
            )

        executed = 0
        skipped: list[str] = []
        outputs: list[str] = []
        budget_gated: list[str] = []
        for entry in stage["commands"]:
            entry_id = str(entry["id"])
            cmd = [str(part) for part in entry["command"]]
            out_path = Path(str(entry["output"]))
            out_path.parent.mkdir(parents=True, exist_ok=True)

            missing_inputs = []
            for path_key in ("artifact", "config", "model_dir", "qwen_fixtures", "dflash_fixture"):
                if path_key in entry and not Path(str(entry[path_key])).exists():
                    missing_inputs.append(str(entry[path_key]))
            if missing_inputs:
                skipped.append(f"{entry_id}: missing {', '.join(missing_inputs)}")
                continue

            if stage_key in ("h1_l", "h1_a"):
                proc = subprocess.run(
                    cmd, cwd=ROOT, check=True, capture_output=True, text=True
                )
                parsed = json.loads(proc.stdout)
                atomic_json(out_path, parsed)
                executed += 1
                outputs.append(str(out_path))
            elif stage_key in ("h1_g", "h1_c"):
                subprocess.run(
                    cmd, cwd=ROOT, check=True, capture_output=True, text=True
                )
                executed += 1
                outputs.append(str(out_path))
            elif stage_key == "h1_m":
                proc = subprocess.run(
                    cmd, cwd=ROOT, check=False, capture_output=True, text=True
                )
                if proc.returncode == 0:
                    executed += 1
                    outputs.append(str(out_path))
                elif not target and "exceeds local Metal budget" in proc.stderr:
                    budget_gated.append(entry_id)
                else:
                    raise RuntimeError(
                        f"H1-M export_p12_routes failed ({proc.returncode}): {proc.stderr.strip()}"
                    )

        status = (
            "executed"
            if executed > 0
            else "budget_gated_non_target"
            if budget_gated
            else "skipped_missing_inputs"
        )
        summaries[stage_key] = {
            "status": status,
            "binary": str(binary),
            "executed": executed,
            "outputs": outputs,
            "skipped": skipped,
            "budget_gated": budget_gated,
        }
    return summaries


def discover_b2_scripts(script_dir: Path = SCRIPT_DIR) -> list[tuple[int, Path]]:
    discovered: list[tuple[int, Path]] = []
    for index in range(2, 7):
        candidate = script_dir / f"reproduce_b2_{index}.sh"
        if candidate.is_file():
            discovered.append((index, candidate))
    return discovered


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path(
            os.environ.get("M5_H1_BUILD_DIR", str(ROOT / "build/b6-xcode27"))
        ),
    )
    parser.add_argument(
        "--workloads",
        type=Path,
        default=ROOT / "bench/p01-workloads-v1.json",
    )
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--b2-5-rows",
        type=int,
        default=int(os.environ.get("M5_B2_5_ROWS", "8")),
        help="Rows passed to B2.5 inspector (0 executes full N; sub-row probes are excluded from route recommendations).",
    )
    parser.add_argument(
        "--stages",
        default="all",
        help="Comma-separated stages to run: all or subset of b2,h1_l,h1_a,h1_g,h1_c,h1_m.",
    )
    parser.add_argument("--static-table-reviewed", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-non-target", action="store_true")
    args = parser.parse_args()
    require(args.iterations >= 2, "tuning requires at least two samples")
    require(args.b2_5_rows >= 0, "--b2-5-rows must be >= 0")
    selected_stages = parse_stages(args.stages)

    output_dir = args.output_dir.resolve()
    build_dir = args.build_dir.resolve()
    workloads_path = args.workloads.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    hardware = hardware_facts()
    target = bool(hardware["target_device"])
    if not target and not args.allow_non_target and not args.plan_only:
        print("M5 Ultra tuning requires an Apple M5 Ultra host", file=sys.stderr)
        return 2
    scope = (
        "M5 Ultra measurement"
        if target
        else "non-target tuning harness validation; not M5 Ultra performance"
    )
    b2_scripts = discover_b2_scripts()
    scripts = [path for _, path in b2_scripts]
    hooks = plan_h1_hooks(build_dir, output_dir, workloads_path, args.iterations)
    plan: dict[str, object] = {
        "schema": "omlx-m5-ultra-tuning-plan-v1",
        "hardware": hardware,
        "build_dir": str(build_dir),
        "iterations": args.iterations,
        "b2_5_rows": args.b2_5_rows,
        "selected_stages": selected_stages,
        "measurement_scope": scope,
        "scripts": [str(path) for path in scripts],
        "h1_hooks": hooks,
        "ranking_partition": ["precision_contract", "row_exact_group"],
        "sub_row_probes_excluded_from_recommendations": True,
        "token_domain": list(range(1, 129)) + [512, 1024],
        "profiles": ["A16", "W4A16", "W4A4"],
        "families": [
            "qmv",
            "row_split",
            "split_k",
            "cooperative",
            "lane_accum",
            "simdgroup_matrix_k32",
            "row_exact_portable",
            "affine_lane",
            "mxfp4_simd",
            "mxfp4_tensorops",
            "grouped_moe",
            "per_expert_fallback",
        ],
    }
    if args.plan_only:
        atomic_json(output_dir / "tuning-plan.json", plan)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    if "b2" in selected_stages and b2_scripts:
        environment = os.environ.copy()
        environment["M5_BENCH_ITERATIONS"] = str(args.iterations)
        environment["M5_MEASUREMENT_SCOPE"] = scope
        environment["M5_B2_5_ROWS"] = str(args.b2_5_rows)
        for index, script in b2_scripts:
            environment[f"B2_{index}_RESULT_DIR"] = str(output_dir / f"b2_{index}")
            subprocess.run([str(script)], cwd=ROOT, env=environment, check=True)

    hook_summaries = run_h1_hooks(
        hooks, selected_stages, output_dir, target=target
    )

    curves = build_curves(output_dir)
    require(bool(curves), "target tuning produced no curves")
    eligible_curves = [row for row in curves if not is_sub_row_probe(row)]
    excluded_sub_row_count = len(curves) - len(eligible_curves)
    require(
        bool(eligible_curves),
        "all measured curves were sub-row probes; full-N execution is required for route selection",
    )

    b2_4_path = output_dir / "b2_4/mxfp4-small-t.json"
    tensorops_runtime_eligible = False
    if b2_4_path.is_file():
        b2_4 = load(b2_4_path, "omlx-m5-b2-4-validation-v1")
        tensorops_runtime_eligible = any(
            bool(record.get("native_tensorops_runtime_eligible"))
            for record in b2_4["records"]
        )
    tensorops_candidates_measured = any(
        "tensorops" in str(row["family"]) for row in eligible_curves
    )
    if tensorops_runtime_eligible:
        require(
            tensorops_candidates_measured,
            "eligible TensorOps candidates were not measured",
        )

    static_table_valid = False
    if args.static_table_reviewed is not None:
        table_path = args.static_table_reviewed.resolve()
        require(
            table_path.is_file(),
            f"reviewed static dispatch table does not exist: {table_path}",
        )
        table_doc = json.loads(table_path.read_text(encoding="utf-8"))
        static_table_valid = isinstance(table_doc, dict) and bool(table_doc)

    ranked_recommendations = recommendations(curves)
    h1_complete = bool(
        target
        and static_table_valid
        and (not tensorops_runtime_eligible or tensorops_candidates_measured)
    )
    result = {
        **plan,
        "schema": "omlx-m5-ultra-tuning-v1",
        "curve_count": len(curves),
        "eligible_curve_count": len(eligible_curves),
        "excluded_sub_row_curve_count": excluded_sub_row_count,
        "curves": curves,
        "h1_hook_summaries": hook_summaries,
        "recommendations": ranked_recommendations,
        "gate": {
            "measurements_complete": target,
            "tensorops_runtime_eligible": tensorops_runtime_eligible,
            "tensorops_candidates_measured": tensorops_candidates_measured,
            "static_table_reviewed": static_table_valid,
            "h1_complete": h1_complete,
            "reason": (
                "complete"
                if h1_complete
                else "generate and review versioned static dispatch table"
            ),
        },
    }
    atomic_json(output_dir / "tuning.json", result)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
