#!/usr/bin/env python3
"""Assemble a self-contained, clean-Mac relocatable ninfer-metal (miv) release archive.

Verifies that every shipped Mach-O binary links only macOS system frameworks
(no Homebrew, /usr/local, or Xcode.app runtime dylibs), packages compiled
Metal 4 / NAX .metallib libraries, Python bindings, silicon profile tools,
measured calibration overlays, an installer script, provider.json, latest.json,
and a SHA-256 RELEASE-MANIFEST.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import stat
import subprocess
import tomllib
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_TOOLS_DIR = Path(__file__).resolve().parent
_SCRIPT_PARENT = _TOOLS_DIR.parent
ROOT = _SCRIPT_PARENT.parent if _SCRIPT_PARENT.name == "miv" else _SCRIPT_PARENT

REQUIRED_BINARIES = (
    "mseries_backendd",
    "ninfer-metal-serve",
    "m5rt_runner",
    "omlx_m5_prefill_matrix_bench",
    "omlx_m5_probe",
    "omlx_m5_nax_qmm_probe",
    "omlx_m5_inspect_runtime",
    "omlx_m5_inspect_icb_canary",
    "omlx_m5_p11_tensor_canary",
    "omlx_m5_p11_layout_inventory",
    "omlx_m5_tf_affine_lane_tests",
    "omlx_m5_benchmark_projection_candidates",
    "omlx_m5_benchmark_long_kv_attention",
    "omlx_m5_benchmark_gdn_prefill",
    "omlx_m5_benchmark_p10_batch",
    "omlx_m5_export_p12_routes",
    "omlx_m5_hw_counter_probe",
)

SHIPPED_TOOLS = (
    "m5_silicon_profile.py",
    "m5_target_guard.py",
    "m5_artifact.py",
    "m5_hw_trace_stats.py",
    "download_b0_models.py",
)

REQUIRED_METALLIBS = (
    "omlx_m5_qwen_forward.metallib",
    "omlx_m5_attention_exact.metallib",
    "omlx_m5_mtp_forward.metallib",
)

REQUIRED_NAX_METALLIBS = (
    "omlx_m5_mxfp4_nax.metallib",
    "omlx_m5_affine_nax.metallib",
    "omlx_m5_bf16_nax.metallib",
    "omlx_m5_moe_nax.metallib",
    "omlx_m5_gdn_chunk_nax.metallib",
    "omlx_m5_w4a8_nax.metallib",
    "omlx_m5_w4a8_moe_nax.metallib",
    "omlx_m5_w4a4_nax.metallib",
    "omlx_m5_w4a4_moe_nax.metallib",
    "omlx_m5_attention_sage2.metallib",
    "omlx_m5_attention_minference.metallib",
    "omlx_m5_cache_blend.metallib",
    "omlx_m5_ane_pack_unpack.metallib",
)

FORBIDDEN_DYLIB_PREFIXES = (
    "/opt/homebrew/",
    "/usr/local/",
    "/Applications/Xcode",
    "/Users/",
)

PROVIDER_SUPPORTED_TOPOLOGIES = [
    "qwen35_hybrid_dense_v1",
    "qwen36_hybrid_moe256_k8_shared_v1",
]

PROVIDER_SUPPORTED_MODEL_TYPES = [
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_6",
    "qwen3_6_moe",
    "qwen3_8",
    "qwen3_8_moe",
]

PROVIDER_SUPPORTED_DRAFT_ARCHITECTURES = ["DFlash2DraftModel"]

PROVIDER_SUPPORTED_DFLASH2_MODELS = [
    "z-lab/Qwen3.8-27B-DFlash2",
    "incoai/Qwen3.8-27B-DFlash2",
    "z-lab/Qwen3.6-27B-DFlash2",
    "incoai/Qwen3.6-27B-DFlash2",
    "z-lab/Qwen3.5-27B-DFlash2",
    "incoai/Qwen3.5-27B-DFlash2",
    "z-lab/Qwen3.6-35B-A3B-DFlash2",
    "incoai/Qwen3.6-35B-A3B-DFlash2",
    "z-lab/Qwen3.5-35B-A3B-DFlash2",
    "incoai/Qwen3.5-35B-A3B-DFlash2",
    "z-lab/Qwen3.8-35B-A3B-DFlash2",
    "incoai/Qwen3.8-35B-A3B-DFlash2",
    "incoai/Ornith-1.5-122B-DFlash2",
    "z-lab/Ornith-1.5-122B-DFlash2",
    "z-lab/Qwen3.5-122B-A10B-DFlash2",
    "incoai/Qwen3.5-122B-A10B-DFlash2",
    "z-lab/Qwen3.6-122B-A10B-DFlash2",
    "incoai/Qwen3.6-122B-A10B-DFlash2",
    "z-lab/Qwen3.8-122B-A10B-DFlash2",
    "incoai/Qwen3.8-122B-A10B-DFlash2",
    "z-lab/Qwen3.5-9B-DFlash2",
    "incoai/Qwen3.5-9B-DFlash2",
    "z-lab/Qwen3.6-9B-DFlash2",
    "incoai/Qwen3.6-9B-DFlash2",
    "z-lab/Qwen3.8-9B-DFlash2",
    "incoai/Qwen3.8-9B-DFlash2",
    "z-lab/Qwen3.5-4B-DFlash2",
    "incoai/Qwen3.5-4B-DFlash2",
    "z-lab/Qwen3.6-4B-DFlash2",
    "incoai/Qwen3.6-4B-DFlash2",
    "z-lab/Qwen3.8-4B-DFlash2",
    "incoai/Qwen3.8-4B-DFlash2",
    "z-lab/Qwen3.5-2B-DFlash2",
    "incoai/Qwen3.5-2B-DFlash2",
    "z-lab/Qwen3.6-2B-DFlash2",
    "incoai/Qwen3.6-2B-DFlash2",
    "z-lab/Qwen3.8-2B-DFlash2",
    "incoai/Qwen3.8-2B-DFlash2",
    "z-lab/Qwen3.5-0.8B-DFlash2",
    "incoai/Qwen3.5-0.8B-DFlash2",
    "z-lab/Qwen3.6-0.8B-DFlash2",
    "incoai/Qwen3.6-0.8B-DFlash2",
    "z-lab/Qwen3.8-0.8B-DFlash2",
    "incoai/Qwen3.8-0.8B-DFlash2",
]

PROVIDER_ARTIFACT_EXTENSIONS = [".miv", ".omlxm5"]

PROVIDER_CONVERTER = {
    "entrypoint": "tools/m5_artifact.py",
    "subcommand": "convert",
    "output_filename": "model.miv",
    "artifact_extension": ".miv",
    "default_args": ["--nax-pretile"],
    "supported_quant_modes": ["affine", "mxfp4", "mxfp4-pack-quantized"],
}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def run_text(cmd: list[str]) -> str:
    res = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return res.stdout.strip()


def audit_macho_dependencies(binary: Path) -> list[str]:
    out = run_text(["otool", "-L", str(binary)])
    libs: list[str] = []
    for line in out.splitlines()[1:]:
        dep = line.strip().split(" (", 1)[0].strip()
        if not dep:
            continue
        libs.append(dep)
        for bad in FORBIDDEN_DYLIB_PREFIXES:
            if dep.startswith(bad):
                raise RuntimeError(
                    f"Release binary {binary.name} links non-system dylib: {dep}"
                )
    return libs


def resolve_build_subdir(build_dir: Path) -> Path:
    resolved = build_dir.resolve()
    for sub in (resolved / "miv", resolved / "m5rt", resolved):
        if (sub / "metallib").is_dir():
            return sub
    raise SystemExit(
        f"Build directory missing miv/metallib (or m5rt/metallib): {resolved}"
    )


def cmake_python_disabled(build_dir: Path) -> bool:
    cache = build_dir.resolve() / "CMakeCache.txt"
    if not cache.is_file():
        return False
    for line in cache.read_text(encoding="utf-8", errors="ignore").splitlines():
        if line.startswith("OMLX_M5_BUILD_PYTHON:"):
            val = line.split("=", 1)[-1].strip().upper()
            return val in ("OFF", "FALSE", "0", "NO")
    return False


def build_install_script(version: str) -> str:
    return f"""#!/bin/zsh
set -euo pipefail

BUNDLE_DIR="${{0:A:h}}"
PROVIDER_ROOT="${{HOME}}/Library/Application Support/oMLX/Providers/io.ninfer.metal"
DEFAULT_PREFIX="${{PROVIDER_ROOT}}/versions/{version}"
PREFIX="${{1:-${{OMLX_M5RT_PREFIX:-${{DEFAULT_PREFIX}}}}}}"

echo "Installing ninfer-metal ({version}) -> ${{PREFIX}}"
mkdir -p "${{PREFIX}}/bin" "${{PREFIX}}/metallib" "${{PREFIX}}/python" "${{PREFIX}}/tools" "${{PREFIX}}/calibrations" "${{PREFIX}}/config" "${{PREFIX}}/scripts"

cp -f "${{BUNDLE_DIR}}/bin/"* "${{PREFIX}}/bin/"
chmod +x "${{PREFIX}}/bin/"*
cp -f "${{BUNDLE_DIR}}/metallib/"*.metallib "${{PREFIX}}/metallib/"
cp -f "${{BUNDLE_DIR}}/python/"* "${{PREFIX}}/python/" 2>/dev/null || true
cp -f "${{BUNDLE_DIR}}/tools/"*.py "${{PREFIX}}/tools/"
chmod +x "${{PREFIX}}/tools/"*.py
cp -f "${{BUNDLE_DIR}}/calibrations/"*.json "${{PREFIX}}/calibrations/" 2>/dev/null || true
if [[ -d "${{BUNDLE_DIR}}/config" ]]; then
  cp -f "${{BUNDLE_DIR}}/config/"* "${{PREFIX}}/config/" 2>/dev/null || true
fi
if [[ -d "${{BUNDLE_DIR}}/scripts" ]]; then
  cp -f "${{BUNDLE_DIR}}/scripts/"* "${{PREFIX}}/scripts/" 2>/dev/null || true
  chmod +x "${{PREFIX}}/scripts/"*.sh 2>/dev/null || true
fi
cp -f "${{BUNDLE_DIR}}/RELEASE-MANIFEST.json" "${{PREFIX}}/"
cp -f "${{BUNDLE_DIR}}/provider.json" "${{PREFIX}}/"
cp -f "${{BUNDLE_DIR}}/README-RELEASE.md" "${{PREFIX}}/"

USER_CONFIG_DIR="${{HOME}}/.config/ninfer-metal"
USER_CONFIG_FILE="${{USER_CONFIG_DIR}}/ninfer-metal.toml"
if [[ -f "${{BUNDLE_DIR}}/config/ninfer-metal.toml" ]]; then
  if [[ ! -f "${{USER_CONFIG_FILE}}" ]]; then
    mkdir -p "${{USER_CONFIG_DIR}}"
    cp "${{BUNDLE_DIR}}/config/ninfer-metal.toml" "${{USER_CONFIG_FILE}}"
    echo "Installed default shared config -> ${{USER_CONFIG_FILE}}"
  else
    echo "Preserving existing user config: ${{USER_CONFIG_FILE}}"
  fi
fi

if [[ "${{PREFIX}}" == "${{DEFAULT_PREFIX}}" ]]; then
  CURRENT_PATH="${{PROVIDER_ROOT}}/current"
  if [[ -d "${{CURRENT_PATH}}" && ! -L "${{CURRENT_PATH}}" ]]; then
    echo "ERROR: ${{CURRENT_PATH}} exists as a real directory (expected symlink); refusing to overwrite." >&2
    exit 1
  fi
  TMP_LINK="${{PROVIDER_ROOT}}/.current.tmp.$$"
  rm -f "${{TMP_LINK}}"
  ln -s "${{PREFIX}}" "${{TMP_LINK}}"
  if ! python3 - "${{TMP_LINK}}" "${{CURRENT_PATH}}" <<'PY'
import os, sys
os.replace(sys.argv[1], sys.argv[2])
PY
  then
    rm -f "${{TMP_LINK}}"
    exit 1
  fi
  cp -f "${{BUNDLE_DIR}}/provider.json" "${{PROVIDER_ROOT}}/provider.json"
fi

echo "=== Verifying Sidecar Daemon ==="
"${{PREFIX}}/bin/mseries_backendd" --version

echo "=== Verifying Native HTTP Server ==="
"${{PREFIX}}/bin/ninfer-metal-serve" --version

echo "=== Verifying Metal Probe ==="
"${{PREFIX}}/bin/omlx_m5_probe" --bytes 16777216 --iterations 5 | head -n 20

echo ""
echo "Installation complete!"
echo "oMLX auto-discovers:       ${{PREFIX}}/bin/mseries_backendd"
echo "1-click standalone server: \"${{PREFIX}}/scripts/start_ninfer_serve.sh\" [MODEL_DIR]"
echo "Shared default config:     ${{USER_CONFIG_FILE}}"
echo "Optional env override:     export OMLX_M5RT_BIN=\"${{PREFIX}}/bin/mseries_backendd\""
"""


def build_release_readme(version: str, bundle_name: str, manifest: dict[str, Any]) -> str:
    cal_lines = []
    for c in manifest.get("calibrations", []):
        cal_lines.append(
            f"- `{c['profile_id']}` (`{c['result_namespace']}`): status=`{c['calibration_status']}`"
        )
    cal_block = "\n".join(cal_lines) if cal_lines else "- (uncalibrated priors only)"
    return f"""# ninfer-metal — Release `{version}`

Standalone, zero-Xcode-required Apple Silicon (macOS 27 / Apple10+ Metal 4 & NAX) inference runtime and `mseries_backendd` UDS Sidecar bundle for `oMLX-extend`.

## Requirements
- Apple Silicon Mac running **macOS 27.0+** (recommended: Apple M5 / M5 Pro / M5 Max / M5 Ultra; M4 supported for correctness/dev without NAX)
- **No Xcode, Homebrew, or MLX runtime required** (all Mach-O binaries link only Apple system frameworks)

## Quick Install
```bash
tar -xzf {bundle_name}.tar.gz
cd {bundle_name}
./install.sh
```
By default this installs into `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/versions/{version}/` and atomically updates `current/`, where `oMLX-extend` automatically discovers `provider.json`, `bin/mseries_backendd`, `metallib/*.metallib`, and `config/ninfer-metal.toml`.

## 1-Click Standalone Server
```bash
./scripts/start_ninfer_serve.sh [MODEL_DIR]
```
Both `ninfer-metal-serve` and `oMLX-extend` automatically share `config/ninfer-metal.toml` (and `~/.config/ninfer-metal/ninfer-metal.toml` when present).

## Included Silicon Calibrations
{cal_block}

## Build Identity
- Version: `{version}` (codename=`{manifest['codename']}`, semver=`{manifest['semver']}`, core_abi=`{manifest['core_abi']}`)
- Git Commit: `{manifest['git_commit']}`
- Built At (UTC): `{manifest['built_at_utc']}`
- SDK / Xcode: `{manifest['build_host'].get('macos_sdk')}` / `{manifest['build_host'].get('xcode_line')}`
- Binaries: `{len(manifest['binaries'])}` Mach-O executables + `{len(manifest['metallibs'])}` `.metallib` libraries
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        default="miv-v0.03",
        help="Release display version tag (default: miv-v0.03).",
    )
    parser.add_argument(
        "--codename",
        default="miv",
        help="Core release codename (default: miv).",
    )
    parser.add_argument(
        "--semver",
        default="0.0.3",
        help="Core semantic version (default: 0.0.3).",
    )
    parser.add_argument(
        "--core-abi",
        type=int,
        default=1,
        help="Core ABI integer version (default: 1).",
    )
    parser.add_argument(
        "--package-prefix",
        default="ninfer-metal",
        help="Release tarball/folder prefix (default: ninfer-metal).",
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=ROOT / "build/miv-v001",
        help="CMake build directory containing compiled miv targets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "dist",
        help="Output directory for staged release directory, tarball, provider.json, and latest.json.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=ROOT / "docs/backend/silicon-calibrations",
        help="Directory of measured silicon calibration overlays to bundle.",
    )
    parser.add_argument(
        "--evidence-summary",
        type=Path,
        default=None,
        help="Optional hardware qualification summary JSON to embed in RELEASE-MANIFEST.json.",
    )
    parser.add_argument(
        "--allow-no-python",
        action="store_true",
        help="Allow packaging without omlx_m5_native*.so (e.g. OMLX_M5_BUILD_PYTHON=OFF).",
    )
    args = parser.parse_args()

    build_miv = resolve_build_subdir(args.build_dir)
    metallib_src = build_miv / "metallib"

    bundle_name = f"{args.package_prefix}-{args.version}-macos27-arm64"
    out_dir = args.output_dir.resolve()
    stage_dir = out_dir / bundle_name
    if stage_dir.exists():
        shutil.rmtree(stage_dir)

    for sub in ("bin", "metallib", "python", "tools", "calibrations", "config", "scripts"):
        (stage_dir / sub).mkdir(parents=True, exist_ok=True)

    # 1) Copy & audit Mach-O binaries
    binary_records: list[dict[str, Any]] = []
    for name in REQUIRED_BINARIES:
        src = build_miv / name
        if not src.is_file():
            raise SystemExit(f"Missing required release binary: {src}")
        dst = stage_dir / "bin" / name
        shutil.copy2(src, dst)
        dst.chmod(dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        deps = audit_macho_dependencies(dst)
        binary_records.append(
            {
                "name": name,
                "path": f"bin/{name}",
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "dylibs": deps,
            }
        )

    # 2) Copy Python native binding(s)
    python_records: list[dict[str, Any]] = []
    for so in sorted(build_miv.glob("omlx_m5_native*.so")):
        dst = stage_dir / "python" / so.name
        shutil.copy2(so, dst)
        deps = audit_macho_dependencies(dst)
        python_records.append(
            {
                "name": so.name,
                "path": f"python/{so.name}",
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
                "dylibs": deps,
            }
        )
    if not python_records and not (
        args.allow_no_python or cmake_python_disabled(args.build_dir)
    ):
        raise SystemExit(
            f"No omlx_m5_native*.so found in {build_miv} "
            "(pass --allow-no-python or configure with -DOMLX_M5_BUILD_PYTHON=OFF)"
        )

    # 3) Copy compiled .metallib libraries (no .air intermediates)
    metallib_records: list[dict[str, Any]] = []
    for lib in sorted(metallib_src.glob("*.metallib")):
        dst = stage_dir / "metallib" / lib.name
        shutil.copy2(lib, dst)
        metallib_records.append(
            {
                "name": lib.name,
                "path": f"metallib/{lib.name}",
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )
    if not metallib_records:
        raise SystemExit(f"No .metallib files found in {metallib_src}")
    metallib_names = {m["name"] for m in metallib_records}
    for req_lib in REQUIRED_METALLIBS:
        if req_lib not in metallib_names:
            raise SystemExit(
                f"Missing required core metallib in {metallib_src}: {req_lib}"
            )
    if any(n.endswith("_nax.metallib") for n in metallib_names):
        for req_nax in REQUIRED_NAX_METALLIBS:
            if req_nax not in metallib_names:
                raise SystemExit(
                    f"Missing required NAX metallib in {metallib_src}: {req_nax}"
                )

    # 4) Copy helper tools (prefer miv/tools/, fallback to root tools/)
    tool_records: list[dict[str, Any]] = []
    for name in SHIPPED_TOOLS:
        src = _TOOLS_DIR / name
        if not src.is_file():
            src = ROOT / "tools" / name
        if not src.is_file():
            raise SystemExit(f"Missing shipped tool: {name}")
        dst = stage_dir / "tools" / name
        shutil.copy2(src, dst)
        tool_records.append(
            {
                "name": name,
                "path": f"tools/{name}",
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )

    # 5) Copy measured calibration overlays (if any)
    cal_records: list[dict[str, Any]] = []
    if args.calibration_dir.is_dir():
        for cal in sorted(args.calibration_dir.glob("*.json")):
            dst = stage_dir / "calibrations" / cal.name
            shutil.copy2(cal, dst)
            payload = json.loads(cal.read_text(encoding="utf-8"))
            overrides = payload.get("measured_dispatch_overrides") or {}
            has_overrides = any(v is not None for v in overrides.values())
            cal_records.append(
                {
                    "name": cal.name,
                    "path": f"calibrations/{cal.name}",
                    "profile_id": payload.get("profile_id"),
                    "result_namespace": payload.get("result_namespace"),
                    "calibration_status": (
                        "measured_calibrated" if has_overrides else "template_only"
                    ),
                    "sha256": sha256_file(dst),
                }
            )

    # 5b) Copy shared default config & 1-click standalone launcher
    config_records: list[dict[str, Any]] = []
    default_cfg_src = ROOT / "omlx-extend" / "config" / "ninfer-metal.toml"
    if not default_cfg_src.is_file():
        raise SystemExit(f"Missing required default config: {default_cfg_src}")
    for cfg_name, required in (
        ("ninfer-metal.toml", True),
        ("ninfer-metal.toml.example", False),
    ):
        src = ROOT / "omlx-extend" / "config" / cfg_name
        if not src.is_file():
            if required:
                raise SystemExit(f"Missing required config file: {src}")
            continue
        dst = stage_dir / "config" / cfg_name
        shutil.copy2(src, dst)
        config_records.append(
            {
                "name": cfg_name,
                "path": f"config/{cfg_name}",
                "bytes": dst.stat().st_size,
                "sha256": sha256_file(dst),
            }
        )

    script_records: list[dict[str, Any]] = []
    launcher_src = ROOT / "omlx-extend" / "scripts" / "start_ninfer_serve.sh"
    if not launcher_src.is_file():
        raise SystemExit(f"Missing required 1-click launcher script: {launcher_src}")
    launcher_dst = stage_dir / "scripts" / "start_ninfer_serve.sh"
    shutil.copy2(launcher_src, launcher_dst)
    launcher_dst.chmod(
        launcher_dst.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    )
    script_records.append(
        {
            "name": launcher_dst.name,
            "path": f"scripts/{launcher_dst.name}",
            "bytes": launcher_dst.stat().st_size,
            "sha256": sha256_file(launcher_dst),
        }
    )

    # 6) Build RELEASE-MANIFEST.json, provider.json, README, installer
    try:
        git_commit = run_text(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
        git_dirty = bool(run_text(["git", "-C", str(ROOT), "status", "--porcelain"]))
    except subprocess.SubprocessError:
        git_commit = "unknown"
        git_dirty = True

    try:
        xcode_line = run_text(["xcodebuild", "-version"]).replace("\n", " | ")
    except (OSError, subprocess.SubprocessError):
        xcode_line = "unknown"
    try:
        macos_sdk = run_text(["xcrun", "-sdk", "macosx", "--show-sdk-version"])
    except (OSError, subprocess.SubprocessError):
        macos_sdk = "unknown"

    built_at_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    manifest: dict[str, Any] = {
        "schema": "ninfer-metal-release-manifest-v1",
        "package_prefix": args.package_prefix,
        "version": args.version,
        "codename": args.codename,
        "semver": args.semver,
        "core_abi": args.core_abi,
        "protocol_version": 2,
        "bundle_name": bundle_name,
        "built_at_utc": built_at_utc,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "build_host": {
            "macos_sdk": macos_sdk,
            "xcode_line": xcode_line,
            "build_dir": str(args.build_dir),
        },
        "clean_mac_dylib_audit": "passed_system_frameworks_only",
        "binaries": binary_records,
        "python_extensions": python_records,
        "metallibs": metallib_records,
        "tools": tool_records,
        "calibrations": cal_records,
        "configs": config_records,
        "scripts": script_records,
    }
    if args.evidence_summary and args.evidence_summary.is_file():
        manifest["hardware_evidence"] = json.loads(
            args.evidence_summary.read_text(encoding="utf-8")
        )

    try:
        contract = json.loads(run_text([str(stage_dir / "bin" / "mseries_backendd"), "--describe"]))
    except (subprocess.SubprocessError, ValueError) as exc:
        raise SystemExit(f"Unable to read the packaged miv daemon's provider contract: {exc}")
    default_config = tomllib.loads(default_cfg_src.read_text(encoding="utf-8"))
    for key in (
        "max_concurrency",
        "max_context",
        "kv_dtype",
        "sampling_contract",
        "prefill_linear_mode",
        "prefill_attention_mode",
        "prefill_sparse_minference",
        "prefill_cache_blend",
        "prefill_ane_mode",
        "prefill_ane_offload",
    ):
        if default_config["engine"].get(key) != contract["load_defaults"][key]:
            raise SystemExit(f"Default TOML and miv daemon disagree on {key}")
    if default_config["speculative"].get("draft_tokens") != contract["load_defaults"]["draft_tokens"]:
        raise SystemExit("Default TOML and miv daemon disagree on draft_tokens")

    provider_manifest: dict[str, Any] = {
        "schema": "ninfer-metal-provider-v1",
        "provider_id": "io.ninfer.metal",
        "legacy_provider_ids": ["com.omlx.mseries"],
        "engine_id": "m5rt",
        "display_name": "miv Core Backend (ninfer-metal)",
        "display_version": args.version,
        "codename": args.codename,
        "semver": args.semver,
        "core_abi": args.core_abi,
        "protocol_version": 2,
        "supported_protocols": [1, 2],
        "engine_contract_version": 1,
        "min_omlx_extend_version": "0.8.0-extend",
        "min_macos_version": "27.0",
        "arch": "arm64",
        "binary": "bin/mseries_backendd",
        "entrypoint": "bin/mseries_backendd",
        "metallib_dir": "metallib",
        "calibrations_dir": "calibrations",
        "default_config": "config/ninfer-metal.toml",
        "launcher_script": "scripts/start_ninfer_serve.sh",
        "supported_topologies": PROVIDER_SUPPORTED_TOPOLOGIES,
        "supported_model_types": PROVIDER_SUPPORTED_MODEL_TYPES,
        "supported_draft_architectures": PROVIDER_SUPPORTED_DRAFT_ARCHITECTURES,
        "supported_dflash2_models": PROVIDER_SUPPORTED_DFLASH2_MODELS,
        "artifact_extensions": PROVIDER_ARTIFACT_EXTENSIONS,
        "converter": PROVIDER_CONVERTER,
        **{key: contract[key] for key in (
            "capabilities", "kv_formats", "supported_standard_settings", "options_schema",
            "speculative_limits", "load_defaults",
        )},
        "built_at_utc": built_at_utc,
        "git_commit": git_commit,
    }

    manifest_path = stage_dir / "RELEASE-MANIFEST.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    stage_provider_path = stage_dir / "provider.json"
    stage_provider_path.write_text(
        json.dumps(provider_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme_path = stage_dir / "README-RELEASE.md"
    readme_path.write_text(
        build_release_readme(args.version, bundle_name, manifest), encoding="utf-8"
    )

    install_path = stage_dir / "install.sh"
    install_path.write_text(build_install_script(args.version), encoding="utf-8")
    install_path.chmod(0o755)

    # 7) Create reproducible tarball + sha256 + top-level provider.json & latest.json
    tarball_path = out_dir / f"{bundle_name}.tar.gz"
    with tarfile.open(tarball_path, "w:gz") as tar:
        tar.add(stage_dir, arcname=bundle_name)

    tar_sha = sha256_file(tarball_path)
    sha_path = out_dir / f"{bundle_name}.tar.gz.sha256"
    sha_path.write_text(f"{tar_sha}  {tarball_path.name}\n", encoding="utf-8")

    out_provider = dict(provider_manifest)
    out_provider["tarball"] = tarball_path.name
    out_provider["tarball_sha256"] = tar_sha
    out_provider["tarball_bytes"] = tarball_path.stat().st_size
    (out_dir / "provider.json").write_text(
        json.dumps(out_provider, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    latest_feed = {
        "schema": "ninfer-metal-release-feed-v1",
        "channel": "stable",
        "provider_id": "io.ninfer.metal",
        "display_version": args.version,
        "version": args.version,
        "codename": args.codename,
        "semver": args.semver,
        "core_abi": args.core_abi,
        "protocol_version": 2,
        "min_omlx_extend_version": "0.8.0-extend",
        "bundle_name": bundle_name,
        "tarball": tarball_path.name,
        "sha256": tar_sha,
        "bytes": tarball_path.stat().st_size,
        "built_at_utc": built_at_utc,
        "provider_manifest": "provider.json",
    }
    (out_dir / "latest.json").write_text(
        json.dumps(latest_feed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary = {
        "tarball": str(tarball_path),
        "bytes": tarball_path.stat().st_size,
        "sha256": tar_sha,
        "provider_json": str(out_dir / "provider.json"),
        "latest_json": str(out_dir / "latest.json"),
        "binaries": len(binary_records),
        "python_extensions": len(python_records),
        "metallibs": len(metallib_records),
        "calibrations": len(cal_records),
        "configs": len(config_records),
        "scripts": len(script_records),
        "clean_mac_dylib_audit": "passed",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
