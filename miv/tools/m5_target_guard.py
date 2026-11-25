#!/usr/bin/env python3
"""Collect safe target facts and fail closed outside an M5 Ultra host."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m5_silicon_profile import probe_silicon_facts, resolve_silicon_profile


def run(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def hardware_facts() -> dict[str, object]:
    probed = probe_silicon_facts()
    profile = resolve_silicon_profile(probed)
    chip = str(probed.get("chip") or "unknown")
    target = "m5 ultra" in chip.casefold()
    return {
        "chip": chip,
        "cpu_total_cores": probed.get("cpu_total_cores"),
        "cpu_perf_cores": probed.get("cpu_perf_cores"),
        "cpu_eff_cores": probed.get("cpu_eff_cores"),
        "gpu_core_count": probed.get("gpu_core_count"),
        "agx_io_class": probed.get("agx_io_class"),
        "system_memory_bytes": probed.get("system_memory_bytes"),
        "macos": probed.get("macos"),
        "macos_build": probed.get("macos_build"),
        "xcode": probed.get("xcode"),
        "macos_sdk": probed.get("macos_sdk"),
        "silicon_profile_id": profile["profile_id"],
        "silicon_matched_by": profile["matched_by"],
        "result_namespace": profile["result_namespace"],
        "calibration_status": profile["calibration_status"],
        "target_device": target,
        "measurement_scope": (
            f"M5 Ultra measurement ({profile['profile_id']})"
            if target
            else "non-target harness validation; not M5 Ultra measurement"
        ),
    }


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-non-target", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    facts = hardware_facts()
    allowed = bool(facts["target_device"] or args.allow_non_target or args.plan_only)
    result = {
        "schema": "omlx-m5-target-guard-v1",
        **facts,
        "plan_only": args.plan_only,
        "allowed": allowed,
        "reason": (
            "target_device"
            if facts["target_device"]
            else "explicit_non_target_harness"
            if args.allow_non_target
            else "plan_only"
            if args.plan_only
            else "requires_apple_m5_ultra"
        ),
    }
    if args.output:
        atomic_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if allowed else 2


if __name__ == "__main__":
    raise SystemExit(main())
