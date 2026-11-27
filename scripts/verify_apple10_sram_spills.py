#!/usr/bin/env python3
"""
Apple10 (M5 Ultra) Global <= 16.0 KiB Static TG SRAM & 0-Byte Register-Spill
4-Layer CI Verification Harness.

Layers enforced:
  1. Public Metal API `staticThreadgroupMemoryLength <= 16,384 B`
     (plus per-WP caps: GQA <= 14,336 B, dv16_flex <= 10,240 B, out64 <= 10,240 B)
     for both ICB=0 and ICB=1 pipelines.
  2. `@encode`-safe AGX `RuntimeState.total_spill_buffer_bytes == 0` on Apple10.
  3. `MTLBinaryArchive` serialization + offline `xcrun metal-lipo` /
     `xcrun metal-objdump` binary slice & header integrity check.
  4. `MTL_SHADER_VALIDATION=1` live GPU dispatch smoke pass.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def run_cmd(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def verify_binary_archive(archive_path: Path) -> dict[str, object]:
    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError(f"Layer 3 failed: binary archive missing or empty at {archive_path}")

    info: dict[str, object] = {
        "archive_path": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
    }

    archs_proc = run_cmd(["xcrun", "-sdk", "macosx", "metal-lipo", "-archs", str(archive_path)])
    if archs_proc.returncode != 0:
        raise RuntimeError(
            f"Layer 3 failed: metal-lipo -archs returned {archs_proc.returncode}: {archs_proc.stderr.strip()}"
        )
    archs = [tok for tok in archs_proc.stdout.strip().split() if tok]
    if not archs:
        raise RuntimeError("Layer 3 failed: metal-lipo -archs returned empty architecture list")
    info["metal_lipo_archs"] = archs

    gpu_slices = [a for a in archs if a.startswith("applegpu")]
    chosen_arch = gpu_slices[0] if gpu_slices else archs[0]
    info["inspected_arch"] = chosen_arch

    thin_path = archive_path.parent / f"thin_{chosen_arch}.bin"
    thin_proc = run_cmd(
        ["xcrun", "-sdk", "macosx", "metal-lipo", "-thin", chosen_arch, str(archive_path), "-output", str(thin_path)]
    )
    if thin_proc.returncode != 0 or not thin_path.is_file():
        raise RuntimeError(
            f"Layer 3 failed: metal-lipo -thin {chosen_arch} failed: {thin_proc.stderr.strip()}"
        )
    info["thin_slice_bytes"] = thin_path.stat().st_size

    objdump = run_cmd(["xcrun", "-sdk", "macosx", "metal-objdump", "-h", str(thin_path)])
    if objdump.returncode != 0:
        raise RuntimeError(
            f"Layer 3 failed: metal-objdump -h returned {objdump.returncode}: {objdump.stderr.strip()}"
        )
    lines = [ln for ln in objdump.stdout.splitlines() if ln.strip()]
    info["metal_objdump_ok"] = True
    info["metal_objdump_sections"] = lines
    if gpu_slices and not any("__compute" in ln for ln in lines):
        raise RuntimeError("Layer 3 failed: Apple GPU Mach-O slice missing __compute section")
    return info


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify Apple10 <= 16.0 KiB SRAM & 0 B register spills across prefill metallibs."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="CMake build directory containing miv/omlx_m5_prefill_sram_spill_audit_tests and miv/metallib.",
    )
    parser.add_argument(
        "--audit-bin",
        type=Path,
        default=None,
        help="Explicit path to omlx_m5_prefill_sram_spill_audit_tests binary.",
    )
    parser.add_argument(
        "--metallib-dir",
        type=Path,
        default=None,
        help="Explicit path to directory containing .metallib files.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Audit all 12 prefill/NAX metallibs (default behavior; kept for CLI compatibility).",
    )
    parser.add_argument(
        "--libs",
        type=str,
        default="",
        help="Comma-separated list of specific .metallib filenames to audit.",
    )
    parser.add_argument(
        "--skip-shader-validation",
        action="store_true",
        help="Skip Layer 4 MTL_SHADER_VALIDATION=1 re-run.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write combined JSON report.",
    )
    args = parser.parse_args()

    audit_bin = args.audit_bin
    if audit_bin is None:
        candidates = [
            args.build_dir / "miv" / "omlx_m5_prefill_sram_spill_audit_tests",
            args.build_dir / "omlx_m5_prefill_sram_spill_audit_tests",
        ]
        for c in candidates:
            if c.is_file():
                audit_bin = c
                break
        if audit_bin is None:
            print(
                f"ERROR: Could not find omlx_m5_prefill_sram_spill_audit_tests under {args.build_dir}",
                file=sys.stderr,
            )
            return 2

    metallib_dir = args.metallib_dir
    if metallib_dir is None:
        candidates = [
            args.build_dir / "miv" / "metallib",
            args.build_dir / "metallib",
        ]
        for c in candidates:
            if c.is_dir():
                metallib_dir = c
                break
        if metallib_dir is None:
            print(
                f"ERROR: Could not find metallib directory under {args.build_dir}",
                file=sys.stderr,
            )
            return 2

    with tempfile.TemporaryDirectory(prefix="apple10_sram_audit_") as tmpdir:
        archive_path = Path(tmpdir) / "audited_pipelines.metallib"

        base_cmd = [str(audit_bin), str(metallib_dir), "--archive-out", str(archive_path)]
        if args.all:
            base_cmd.append("--all")
        if args.libs:
            base_cmd.extend(["--libs", args.libs])

        # Layers 1, 2, 3 (+ normal smoke)
        clean_env = os.environ.copy()
        clean_env.pop("MTL_SHADER_VALIDATION", None)
        res = run_cmd(base_cmd, env=clean_env)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr, file=sys.stderr)
            return res.returncode

        audit_json = json.loads(res.stdout)
        if not audit_json.get("passed", False):
            print(res.stdout)
            return 1

        # Layer 3 offline binary-archive verification via metal-lipo / metal-objdump
        layer3_info = verify_binary_archive(archive_path)

        # Layer 4: MTL_SHADER_VALIDATION=1 dispatch smoke
        layer4_info: dict[str, object] = {"skipped": args.skip_shader_validation}
        if not args.skip_shader_validation:
            sv_env = os.environ.copy()
            sv_env["MTL_SHADER_VALIDATION"] = "1"
            sv_cmd = [str(audit_bin), str(metallib_dir), "--smoke-only"]
            if args.all:
                sv_cmd.append("--all")
            if args.libs:
                sv_cmd.extend(["--libs", args.libs])
            sv_res = run_cmd(sv_cmd, env=sv_env)
            if sv_res.returncode != 0:
                print(sv_res.stdout)
                print(sv_res.stderr, file=sys.stderr)
                return sv_res.returncode
            sv_json = json.loads(sv_res.stdout)
            if not sv_json.get("passed", False):
                print(sv_res.stdout)
                return 1
            layer4_info = {
                "skipped": False,
                "passed": True,
                "smoke": sv_json.get("layer4_smoke", {}),
            }

        combined = {
            "schema": "omlx-m5-verify-apple10-sram-spills-v1",
            "audit": audit_json,
            "layer3_offline_binary_check": layer3_info,
            "layer4_shader_validation_smoke": layer4_info,
            "passed": True,
        }
        out_str = json.dumps(combined, indent=2)
        if args.json_out is not None:
            args.json_out.write_text(out_str + "\n", encoding="utf-8")
        print(out_str)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
