#!/usr/bin/env python3
"""Run or plan the M5 Ultra H0 capability and execution-path probes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPT_PARENT = _SCRIPT_DIR.parent
ROOT = _SCRIPT_PARENT.parent if _SCRIPT_PARENT.name == "miv" else _SCRIPT_PARENT
sys.path.insert(0, str(_SCRIPT_DIR))

from m5_target_guard import hardware_facts

INVENTORY_ARTIFACTS = (
    ("27b_mxfp4", "qwen38-27b-mlx-mxfp4.omlxm5"),
    ("35b_mxfp4", "qwen36-35b-a3b-mxfp4.omlxm5"),
    ("27b_oq4e", "qwen38-27b-oq4e.omlxm5"),
    ("35b_oq4e", "qwen36-35b-a3b-oq4e.omlxm5"),
)

P11_CANARY_SHAPES = (
    ("synthetic", 32, 256),
    ("27b", 48, 5120),
    ("35b", 8192, 2048),
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


def checked_json(command: list[str], *, output: Path | None = None) -> dict:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    if output is not None:
        return json.loads(output.read_text(encoding="utf-8"))
    return json.loads(result.stdout)


def resolve_trace_evidence(
    system_trace_json: Path | None,
    attach_trace_evidence: Path | None,
) -> tuple[bool, dict[str, object]]:
    if system_trace_json is None and attach_trace_evidence is None:
        return False, {
            "attached": False,
            "valid": False,
            "reason": "no trace evidence supplied",
        }

    evidence: dict[str, object] = {
        "attached": False,
        "valid": True,
    }
    if system_trace_json is not None:
        trace_json_path = system_trace_json.resolve()
        require(
            trace_json_path.is_file(),
            f"system trace JSON does not exist: {trace_json_path}",
        )
        parsed = json.loads(trace_json_path.read_text(encoding="utf-8"))
        require(
            isinstance(parsed, dict) and bool(parsed),
            f"system trace JSON must be a non-empty object: {trace_json_path}",
        )
        if parsed.get("valid") is False or parsed.get("attached") is False:
            evidence["valid"] = False
            evidence["reason"] = str(
                parsed.get("reason", "trace JSON explicitly marked invalid/unattached")
            )
        referenced = parsed.get("trace_path")
        if referenced is not None:
            ref_path = Path(str(referenced))
            if not ref_path.is_absolute():
                ref_path = (trace_json_path.parent / ref_path).resolve()
            require(
                ref_path.exists(),
                f"referenced trace_path in system trace JSON does not exist: {ref_path}",
            )
            evidence["referenced_trace_path"] = str(ref_path)
        evidence["system_trace_json"] = str(trace_json_path)
        evidence["summary"] = parsed

    if attach_trace_evidence is not None:
        bundle_path = attach_trace_evidence.resolve()
        require(
            bundle_path.exists(),
            f"attached trace evidence path does not exist: {bundle_path}",
        )
        if bundle_path.is_dir():
            entries = list(bundle_path.iterdir())
            require(
                bool(entries),
                f"attached trace directory is empty: {bundle_path}",
            )
            evidence["trace_bundle_dir"] = str(bundle_path)
            evidence["trace_bundle_entries"] = len(entries)
        else:
            size = bundle_path.stat().st_size
            require(
                size > 0,
                f"attached trace evidence file is empty: {bundle_path}",
            )
            if bundle_path.suffix.lower() == ".json":
                parsed_attachment = json.loads(bundle_path.read_text(encoding="utf-8"))
                require(
                    isinstance(parsed_attachment, dict) and bool(parsed_attachment),
                    f"attached trace JSON must be a non-empty object: {bundle_path}",
                )
                if (
                    parsed_attachment.get("valid") is False
                    or parsed_attachment.get("attached") is False
                ):
                    evidence["valid"] = False
                    evidence["reason"] = str(
                        parsed_attachment.get(
                            "reason",
                            "attached trace JSON explicitly marked invalid/unattached",
                        )
                    )
                evidence.setdefault("summary", parsed_attachment)
            evidence["trace_evidence_file"] = str(bundle_path)
            evidence["trace_evidence_bytes"] = size

    attached = bool(evidence["valid"])
    evidence["attached"] = attached
    if attached:
        evidence["reason"] = "trace_evidence_verified"
    return attached, evidence


def command_plan(
    build_dir: Path,
    output_dir: Path,
    iterations: int,
    artifacts_dir: Path,
) -> list[dict[str, object]]:
    binary = build_dir / "miv" if (build_dir / "miv").is_dir() else build_dir / "m5rt"
    metallib = binary / "metallib"
    commands: list[dict[str, object]] = [
        {
            "id": "metal4_copy",
            "optional": False,
            "output": str(output_dir / "metal4-copy.json"),
            "command": [
                str(binary / "omlx_m5_probe"),
                "--bytes",
                str(64 * 1024 * 1024),
                "--iterations",
                str(iterations),
                "--output",
                str(output_dir / "metal4-copy.json"),
            ],
        },
        {
            "id": "metal4_copy_256m",
            "optional": False,
            "output": str(output_dir / "metal4-copy-256m.json"),
            "command": [
                str(binary / "omlx_m5_probe"),
                "--bytes",
                str(256 * 1024 * 1024),
                "--iterations",
                str(max(4, iterations // 2)),
                "--output",
                str(output_dir / "metal4-copy-256m.json"),
            ],
        },
        *[
            {
                "id": f"nax_q{bits}_m{tokens}",
                "optional": False,
                "output": str(output_dir / f"nax-q{bits}-m{tokens}.json"),
                "command": [
                    str(binary / "omlx_m5_nax_qmm_probe"),
                    "--m",
                    str(tokens),
                    "--n",
                    "5120",
                    "--k",
                    "5120",
                    "--bits",
                    str(bits),
                    "--iterations",
                    str(iterations),
                    "--output",
                    str(output_dir / f"nax-q{bits}-m{tokens}.json"),
                ],
            }
            for bits in (4, 5)
            for tokens in (1, 8)
        ],
        {
            "id": "runtime_registry",
            "optional": False,
            "command": [
                str(binary / "omlx_m5_inspect_runtime"),
                str(metallib / "omlx_m5_smoke.metallib"),
            ],
        },
        {
            "id": "icb_replay",
            "optional": False,
            "command": [
                str(binary / "omlx_m5_inspect_icb_canary"),
                str(max(10_000, iterations)),
            ],
        },
    ]

    p11_canary_bin = binary / "omlx_m5_p11_tensor_canary"
    p11_metallib = metallib / "omlx_m5_p11_fp4_native.metallib"
    for label, n_dim, k_dim in P11_CANARY_SHAPES:
        out_path = output_dir / f"p11-canary-{label}.json"
        commands.append(
            {
                "id": f"p11_canary_{label}",
                "optional": True,
                "requires_files": [str(p11_canary_bin), str(p11_metallib)],
                "output": str(out_path),
                "command": [
                    str(p11_canary_bin),
                    str(p11_metallib),
                    str(out_path),
                    str(n_dim),
                    str(k_dim),
                ],
            }
        )

    inventory_bin = binary / "omlx_m5_p11_layout_inventory"
    for label, filename in INVENTORY_ARTIFACTS:
        artifact_path = artifacts_dir / filename
        out_path = output_dir / f"p11-inventory-{label}.json"
        commands.append(
            {
                "id": f"p11_inventory_{label}",
                "optional": True,
                "requires_files": [str(inventory_bin), str(artifact_path)],
                "output": str(out_path),
                "command": [
                    str(inventory_bin),
                    str(artifact_path),
                    str(out_path),
                ],
            }
        )

    affine_bin = binary / "omlx_m5_tf_affine_lane_tests"
    affine_metallib = metallib / "omlx_m5_affine_lane.metallib"
    commands.append(
        {
            "id": "tf_affine_lane_tensor",
            "optional": True,
            "requires_files": [str(affine_bin), str(affine_metallib)],
            "output": str(output_dir / "tf-affine-lane-tensor.json"),
            "command": [
                str(affine_bin),
                str(metallib),
                "--tensor",
            ],
        }
    )
    return commands


def extract_device_capabilities(records: dict[str, dict]) -> dict[str, object]:
    probe = records["metal4_copy"]
    probe_large = records.get("metal4_copy_256m")
    runtime = records["runtime_registry"]
    probe_hw = probe.get("hardware", {})
    runtime_hw = runtime.get("hardware", {})
    runtime_cap = runtime.get("capability", {})
    runtime_pipe = runtime.get("pipeline", {})
    runtime_reg = runtime.get("registry", {})
    return {
        "device_name": probe_hw.get("device_name") or runtime_hw.get("device_name"),
        "architecture_name": runtime_hw.get("architecture_name"),
        "apple_gpu_family": runtime_hw.get("apple_gpu_family"),
        "system_memory_bytes": probe_hw.get("system_memory_bytes"),
        "recommended_working_set_bytes": probe_hw.get(
            "recommended_working_set_bytes",
            runtime_hw.get("recommended_working_set_bytes"),
        ),
        "max_buffer_length": probe_hw.get(
            "max_buffer_length",
            runtime_hw.get("max_buffer_length"),
        ),
        "current_allocated_buffer_bytes": runtime_reg.get("current_buffer_bytes"),
        "peak_allocated_buffer_bytes": runtime_reg.get("peak_buffer_bytes"),
        "max_threadgroup_memory_length_prior_bytes": 32768,
        "thread_execution_width": runtime_pipe.get("thread_execution_width"),
        "max_threads_per_threadgroup": runtime_pipe.get("max_threads_per_threadgroup"),
        "static_threadgroup_memory_bytes": runtime_pipe.get(
            "static_threadgroup_memory_bytes"
        ),
        "copy_bandwidth_64mb_gbs": probe.get("copy_probe", {}).get(
            "read_plus_write_gbs"
        ),
        "copy_bandwidth_256mb_gbs": (
            probe_large.get("copy_probe", {}).get("read_plus_write_gbs")
            if probe_large
            else None
        ),
        "metal4_direct_eligible": runtime_cap.get("metal4_direct_eligible"),
        "compute_icb_eligible": runtime_cap.get("compute_icb_eligible"),
        "metal4_icb_eligible": runtime_cap.get("metal4_icb_eligible"),
        "tensor_api": runtime_cap.get("tensor_api"),
        "native_tensorops_eligible": runtime_cap.get("native_tensorops_eligible"),
        "native_tensorops_runtime": runtime_cap.get("native_tensorops_runtime"),
        "nax_metallib_loaded_on_current_host": probe.get("nax_metallib", {}).get(
            "loaded_on_current_host"
        ),
        "nax_metallib_loaded_on_m5_ultra": probe.get("nax_metallib", {}).get(
            "loaded_on_m5_ultra"
        ),
    }


def evaluate_tensorops_execution(
    records: dict[str, dict],
    skipped_optional: dict[str, str],
) -> tuple[bool, dict[str, object]]:
    nax_records = [
        value
        for key, value in sorted(records.items())
        if key.startswith("nax_q")
    ]
    nax_m5_validated = bool(
        len(nax_records) == 4
        and all(bool(record.get("runtime_validated_on_m5")) for record in nax_records)
        and all(int(record.get("metal4_command_buffers", 0)) > 0 for record in nax_records)
    )

    p11_records = [
        value
        for key, value in sorted(records.items())
        if key.startswith("p11_canary_")
    ]
    p11_canary_passed: bool | None
    if p11_records:
        p11_canary_passed = all(
            bool(record.get("pipeline_compiled"))
            and bool(record.get("runtime_executed"))
            and bool(record.get("zero_checked"))
            and bool(record.get("scale_sign_checked"))
            for record in p11_records
        )
    elif any(k.startswith("p11_canary_") for k in skipped_optional):
        p11_canary_passed = None
    else:
        p11_canary_passed = False

    affine_record = records.get("tf_affine_lane_tensor")
    tf_affine_tensor_passed: bool | None
    if affine_record is not None:
        cases = affine_record.get("cases")
        cases_executed = (
            isinstance(cases, list)
            and bool(cases)
            and all(
                bool(case.get("passed"))
                and bool(case.get("m5_tensor_candidate_kernel_executed"))
                for case in cases
            )
        )
        tf_affine_tensor_passed = bool(
            affine_record.get("passed")
            and (
                affine_record.get("m5_tensor_candidate_kernel_executed")
                or cases_executed
            )
        )
    elif "tf_affine_lane_tensor" in skipped_optional:
        tf_affine_tensor_passed = None
    else:
        tf_affine_tensor_passed = False

    inventory_records = {
        key: value.get("categories", {})
        for key, value in sorted(records.items())
        if key.startswith("p11_inventory_")
    }

    tensorops_execution = bool(
        nax_m5_validated
        and (p11_canary_passed is not False)
        and (tf_affine_tensor_passed is not False)
    )
    summary = {
        "nax_qmm_runtime_validated_on_m5": nax_m5_validated,
        "p11_fp4_canary_passed": p11_canary_passed,
        "p11_fp4_canary_count": len(p11_records),
        "tf_affine_tensor_passed": tf_affine_tensor_passed,
        "p11_layout_inventories": inventory_records,
        "tensorops_execution": tensorops_execution,
    }
    return tensorops_execution, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ROOT / ".cache/b0/artifacts",
    )
    parser.add_argument("--system-trace-json", type=Path, default=None)
    parser.add_argument("--attach-trace-evidence", type=Path, default=None)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--allow-non-target", action="store_true")
    args = parser.parse_args()
    require(args.iterations > 0, "iterations must be positive")

    build_dir = args.build_dir.resolve()
    output_dir = args.output_dir.resolve()
    artifacts_dir = args.artifacts_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_attached, trace_evidence = resolve_trace_evidence(
        args.system_trace_json,
        args.attach_trace_evidence,
    )

    hardware = hardware_facts()
    target = bool(hardware["target_device"])
    if not target and not args.allow_non_target and not args.plan_only:
        print(
            "M5 Ultra capability execution requires an Apple M5 Ultra host",
            file=sys.stderr,
        )
        return 2

    commands = command_plan(build_dir, output_dir, args.iterations, artifacts_dir)
    result: dict[str, object] = {
        "schema": "omlx-m5-ultra-capability-v1",
        "hardware": hardware,
        "plan_only": args.plan_only,
        "commands": commands,
        "trace_evidence": trace_evidence,
        "checks": {},
        "gate": {
            "h0_complete": False,
            "reason": "plan_only" if args.plan_only else "probes_pending",
        },
    }
    if args.plan_only:
        atomic_json(output_dir / "capability-plan.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    records: dict[str, dict] = {}
    skipped_optional: dict[str, str] = {}
    for entry in commands:
        entry_id = str(entry["id"])
        command = [str(part) for part in entry["command"]]
        optional = bool(entry.get("optional", False))
        required_files = [
            Path(str(path))
            for path in entry.get("requires_files", [command[0]])
        ]
        missing = [str(path) for path in required_files if not path.is_file()]
        if missing:
            if optional:
                skipped_optional[entry_id] = f"missing: {', '.join(missing)}"
                continue
            raise ValueError(f"missing capability binary: {missing[0]}")

        output_path = Path(str(entry["output"])) if "output" in entry else None
        if entry_id == "tf_affine_lane_tensor":
            completed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            if completed.returncode == 0:
                parsed = json.loads(completed.stdout)
                if output_path is not None:
                    atomic_json(output_path, parsed)
                records[entry_id] = parsed
            else:
                failure_record: dict[str, object] = {
                    "schema": "omlx-m5-p11-tf-affine-v1",
                    "passed": False,
                    "m5_tensor_candidate_kernel_executed": False,
                    "returncode": completed.returncode,
                    "stderr": completed.stderr.strip(),
                }
                # Also capture portable reference status on non-M5 hosts when available.
                portable = subprocess.run(
                    command[:-1], check=False, capture_output=True, text=True
                )
                if portable.returncode == 0:
                    failure_record["portable_reference"] = json.loads(portable.stdout)
                if output_path is not None:
                    atomic_json(output_path, failure_record)
                records[entry_id] = failure_record
            continue

        records[entry_id] = checked_json(command, output=output_path)

    # Fallback layout inventory on build fixture when real .omlxm5 cache is absent.
    bin_sub = build_dir / "miv" if (build_dir / "miv").is_dir() else build_dir / "m5rt"
    inventory_bin = bin_sub / "omlx_m5_p11_layout_inventory"
    fixture_mxfp4 = bin_sub / "fixtures/qwen-execution/mxfp4/model.omlxm5"
    if (
        not any(key.startswith("p11_inventory_") for key in records)
        and inventory_bin.is_file()
        and fixture_mxfp4.is_file()
    ):
        fixture_out = output_dir / "p11-inventory-fixture-mxfp4.json"
        records["p11_inventory_fixture_mxfp4"] = checked_json(
            [str(inventory_bin), str(fixture_mxfp4), str(fixture_out)],
            output=fixture_out,
        )

    nax_records = [
        value for key, value in records.items() if key.startswith("nax_q")
    ]
    tensorops_execution, tensorops_summary = evaluate_tensorops_execution(
        records, skipped_optional
    )
    device_capabilities = extract_device_capabilities(records)

    checks = {
        "metal4_submission": bool(
            records["runtime_registry"]["capability"]["metal4_direct_eligible"]
            and records["runtime_registry"]["registry"][
                "metal4_command_buffers_submitted"
            ]
            > 0
        ),
        "compute_icb_replay": bool(
            records["icb_replay"]["replay"]["bit_exact"]
            and records["icb_replay"]["replay"]["icb_execute_calls"] >= 10_000
        ),
        "nax_q4_q5_m1_m8_execution": bool(
            len(nax_records) == 4
            and all(record["runtime_available"] for record in nax_records)
            and all(record["metal4_command_buffers"] > 0 for record in nax_records)
        ),
        "target_identity": target,
        "tensorops_execution": tensorops_execution,
        "system_trace_attached": trace_attached,
    }
    pending = [name for name, passed in sorted(checks.items()) if not passed]
    result["device_capabilities"] = device_capabilities
    result["tensorops_summary"] = tensorops_summary
    result["skipped_optional"] = skipped_optional
    result["checks"] = checks
    result["records"] = records
    result["gate"] = {
        "h0_complete": all(checks.values()),
        "pending_checks": pending,
        "reason": (
            "complete"
            if not pending
            else f"incomplete checks: {', '.join(pending)}"
        ),
    }
    atomic_json(output_dir / "capability.json", result)
    print(json.dumps(result["gate"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
