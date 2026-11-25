#!/usr/bin/env python3
"""Apple Silicon SKU recognizer, theoretical preset registry, and calibration loader.

Design contract:
- Collects only non-identifying hardware facts (chip name, CPU/GPU core counts,
  unified memory size, AGX IOClass, OS/SDK versions). Never reads serial numbers
  or hardware UUIDs.
- Distinguishes Apple M5 / M5 Pro / M5 Max / M5 Ultra (64-core binned vs
  80-core full-core) and M4 dev hosts.
- Emits explicitly labeled ``theoretical_prior`` presets scaled by silicon tier
  and detected unified memory. Priors are starting points for H0/H1/H2 bring-up
  and are never marked ``device_performance_qualified`` or ``production_enabled``.
- Supports overlaying measured calibration JSON files once real-hardware H0/H1/H2
  runs complete, recording an explicit ``prior_vs_measured_diff`` audit trail.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import plistlib
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPT_PARENT = Path(__file__).resolve().parents[1]
ROOT = _SCRIPT_PARENT.parent if _SCRIPT_PARENT.name == "miv" else _SCRIPT_PARENT
DEFAULT_CALIBRATION_DIR = ROOT / "docs/backend/silicon-calibrations"
GIB = 1 << 30


@dataclass(frozen=True)
class SiliconPreset:
    profile_id: str
    chip_family: str
    chip_variant: str
    generation: str
    support_tier_prior: str
    gpu_cores_nominal: int
    nax_units_nominal: int
    die_count_nominal: int
    fusion_topology: str
    memory_bandwidth_gbs_nominal: float
    compute_scale_vs_ultra_80: float
    bandwidth_scale_vs_ultra_80: float
    # Theoretical dispatch / tuning priors (NOT measured qualification)
    linear_nax_min_tokens_prior: int
    linear_split_k_target_slices_prior: int
    attention_split_kv_min_context_prior: int
    attention_split_kv_partitions_prior: int
    gdn_prefill_tb8_min_tokens_prior: int
    dflash2_draft_tokens_prior: int
    moe_grouped_vs_fallback_active_tokens_prior: int
    notes: str


# Baseline public / architectural priors for Apple M5 family and M4 dev host.
# All numeric dispatch parameters are uncalibrated starting priors until
# replaced by measured H0/H1/H2 calibration overlays on physical hardware.
SILICON_PRESETS: dict[str, SiliconPreset] = {
    "apple-m5-ultra-80gpu": SiliconPreset(
        profile_id="apple-m5-ultra-80gpu",
        chip_family="M5",
        chip_variant="Ultra-80GPU",
        generation="Apple10",
        support_tier_prior="Tier A (target full-core)",
        gpu_cores_nominal=80,
        nax_units_nominal=80,
        die_count_nominal=4,
        fusion_topology="ultrafusion_quad_die",
        memory_bandwidth_gbs_nominal=1200.0,
        compute_scale_vs_ultra_80=1.00,
        bandwidth_scale_vs_ultra_80=1.00,
        linear_nax_min_tokens_prior=1,
        linear_split_k_target_slices_prior=8,
        attention_split_kv_min_context_prior=8192,
        attention_split_kv_partitions_prior=16,
        gdn_prefill_tb8_min_tokens_prior=256,
        dflash2_draft_tokens_prior=7,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Full-core Apple M5 Ultra (80 GPU/NAX, 1.2 TB/s nominal). Primary release target; requires dedicated H0/H1/H2 qualification.",
    ),
    "apple-m5-ultra-64gpu": SiliconPreset(
        profile_id="apple-m5-ultra-64gpu",
        chip_family="M5",
        chip_variant="Ultra-64GPU",
        generation="Apple10",
        support_tier_prior="Tier A (binned early-bringup)",
        gpu_cores_nominal=64,
        nax_units_nominal=64,
        die_count_nominal=4,
        fusion_topology="ultrafusion_quad_die",
        memory_bandwidth_gbs_nominal=1200.0,
        compute_scale_vs_ultra_80=0.80,
        bandwidth_scale_vs_ultra_80=1.00,
        linear_nax_min_tokens_prior=1,
        linear_split_k_target_slices_prior=6,
        attention_split_kv_min_context_prior=8192,
        attention_split_kv_partitions_prior=12,
        gdn_prefill_tb8_min_tokens_prior=256,
        dflash2_draft_tokens_prior=7,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Binned Apple M5 Ultra (30 CPU / 64 GPU/NAX, 1.2 TB/s nominal). Shares quad-die UltraFusion bandwidth class with 80-core; 0.80x compute/NAX prior.",
    ),
    "apple-m5-max-40gpu": SiliconPreset(
        profile_id="apple-m5-max-40gpu",
        chip_family="M5",
        chip_variant="Max-40GPU",
        generation="Apple10",
        support_tier_prior="Tier A/B (memory-dependent)",
        gpu_cores_nominal=40,
        nax_units_nominal=40,
        die_count_nominal=2,
        fusion_topology="fusion_dual_die",
        memory_bandwidth_gbs_nominal=614.0,
        compute_scale_vs_ultra_80=0.50,
        bandwidth_scale_vs_ultra_80=0.512,
        linear_nax_min_tokens_prior=1,
        linear_split_k_target_slices_prior=4,
        attention_split_kv_min_context_prior=16384,
        attention_split_kv_partitions_prior=8,
        gdn_prefill_tb8_min_tokens_prior=256,
        dflash2_draft_tokens_prior=5,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Full Apple M5 Max (40 GPU/NAX, ~614 GB/s nominal). Half of M5 Ultra 80-core silicon.",
    ),
    "apple-m5-max-32gpu": SiliconPreset(
        profile_id="apple-m5-max-32gpu",
        chip_family="M5",
        chip_variant="Max-32GPU",
        generation="Apple10",
        support_tier_prior="Tier A/B (memory-dependent)",
        gpu_cores_nominal=32,
        nax_units_nominal=32,
        die_count_nominal=2,
        fusion_topology="fusion_dual_die",
        memory_bandwidth_gbs_nominal=614.0,
        compute_scale_vs_ultra_80=0.40,
        bandwidth_scale_vs_ultra_80=0.512,
        linear_nax_min_tokens_prior=1,
        linear_split_k_target_slices_prior=4,
        attention_split_kv_min_context_prior=16384,
        attention_split_kv_partitions_prior=8,
        gdn_prefill_tb8_min_tokens_prior=256,
        dflash2_draft_tokens_prior=5,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Binned Apple M5 Max (32 GPU/NAX). Half of M5 Ultra 64-core silicon.",
    ),
    "apple-m5-pro-20gpu": SiliconPreset(
        profile_id="apple-m5-pro-20gpu",
        chip_family="M5",
        chip_variant="Pro-20GPU",
        generation="Apple10",
        support_tier_prior="Tier B (capacity-bounded)",
        gpu_cores_nominal=20,
        nax_units_nominal=20,
        die_count_nominal=2,
        fusion_topology="fusion",
        memory_bandwidth_gbs_nominal=307.0,
        compute_scale_vs_ultra_80=0.25,
        bandwidth_scale_vs_ultra_80=0.256,
        linear_nax_min_tokens_prior=4,
        linear_split_k_target_slices_prior=2,
        attention_split_kv_min_context_prior=32768,
        attention_split_kv_partitions_prior=4,
        gdn_prefill_tb8_min_tokens_prior=512,
        dflash2_draft_tokens_prior=4,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Full Apple M5 Pro (20 GPU/NAX, ~307 GB/s nominal). Check unified memory before enabling 27B/35B.",
    ),
    "apple-m5-pro-16gpu": SiliconPreset(
        profile_id="apple-m5-pro-16gpu",
        chip_family="M5",
        chip_variant="Pro-16GPU",
        generation="Apple10",
        support_tier_prior="Tier B (capacity-bounded)",
        gpu_cores_nominal=16,
        nax_units_nominal=16,
        die_count_nominal=2,
        fusion_topology="fusion",
        memory_bandwidth_gbs_nominal=307.0,
        compute_scale_vs_ultra_80=0.20,
        bandwidth_scale_vs_ultra_80=0.256,
        linear_nax_min_tokens_prior=4,
        linear_split_k_target_slices_prior=2,
        attention_split_kv_min_context_prior=32768,
        attention_split_kv_partitions_prior=4,
        gdn_prefill_tb8_min_tokens_prior=512,
        dflash2_draft_tokens_prior=4,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Binned Apple M5 Pro (16 GPU/NAX).",
    ),
    "apple-m5-10gpu": SiliconPreset(
        profile_id="apple-m5-10gpu",
        chip_family="M5",
        chip_variant="Base-10GPU",
        generation="Apple10",
        support_tier_prior="Tier B/C (32GB max; single-stream or small models)",
        gpu_cores_nominal=10,
        nax_units_nominal=10,
        die_count_nominal=1,
        fusion_topology="single_die",
        memory_bandwidth_gbs_nominal=153.0,
        compute_scale_vs_ultra_80=0.125,
        bandwidth_scale_vs_ultra_80=0.1275,
        linear_nax_min_tokens_prior=8,
        linear_split_k_target_slices_prior=1,
        attention_split_kv_min_context_prior=32768,
        attention_split_kv_partitions_prior=2,
        gdn_prefill_tb8_min_tokens_prior=512,
        dflash2_draft_tokens_prior=3,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Base Apple M5 (10 GPU/NAX, ~153 GB/s nominal). 27B/35B full residency requires >=32 GB and tight context limits.",
    ),
    "apple-m4-dev": SiliconPreset(
        profile_id="apple-m4-dev",
        chip_family="M4",
        chip_variant="Dev-Host",
        generation="Apple9",
        support_tier_prior="Tier C (dev/correctness only; no NAX)",
        gpu_cores_nominal=10,
        nax_units_nominal=0,
        die_count_nominal=1,
        fusion_topology="single_die",
        memory_bandwidth_gbs_nominal=120.0,
        compute_scale_vs_ultra_80=0.0,
        bandwidth_scale_vs_ultra_80=0.10,
        linear_nax_min_tokens_prior=0,
        linear_split_k_target_slices_prior=1,
        attention_split_kv_min_context_prior=32768,
        attention_split_kv_partitions_prior=2,
        gdn_prefill_tb8_min_tokens_prior=512,
        dflash2_draft_tokens_prior=0,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Development and correctness host (Apple9). Not a performance qualification target.",
    ),
}

# Exact workload memory figures aligned with m5-ultra-arrival-runbook.md (Section 3.1/3.2)
# and m5rt/src/qwen_memory.cpp.
WORKLOAD_MEMORY_MODEL: dict[str, dict[str, Any]] = {
    "qwen38-27b-oq4e-ar": {
        "shared_weights_gib": 15.81,
        "per_slot_gib": {"32k": 2.85, "64k": 4.85, "128k": 8.85},
    },
    "qwen38-27b-mlx-mxfp4-ar": {
        "shared_weights_gib": 14.17,
        "per_slot_gib": {"32k": 2.85, "64k": 4.85, "128k": 8.85},
    },
    "qwen38-27b-oq4e-dflash2": {
        "shared_weights_gib": 15.81 + 3.58,
        "per_slot_gib": {"32k": 4.83, "64k": 8.39, "128k": 15.52},
    },
    "qwen38-27b-mlx-mxfp4-dflash2": {
        "shared_weights_gib": 14.17 + 3.58,
        "per_slot_gib": {"32k": 4.83, "64k": 8.39, "128k": 15.52},
    },
    "qwen36-35b-a3b-oq4e-ar": {
        "shared_weights_gib": 20.13,
        "per_slot_gib": {"32k": 1.32, "64k": 1.94, "128k": 3.19},
    },
    "qwen36-35b-a3b-mxfp4-ar": {
        "shared_weights_gib": 17.99,
        "per_slot_gib": {"32k": 1.32, "64k": 1.94, "128k": 3.19},
    },
}


def _run_text(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _run_int(command: list[str]) -> int | None:
    raw = _run_text(command)
    if raw and raw.isdigit():
        return int(raw)
    return None


def probe_agx_accelerator() -> dict[str, Any]:
    """Read GPU core count and AGX class from macOS IORegistry (safe, no UUIDs)."""
    info: dict[str, Any] = {
        "gpu_core_count": None,
        "agx_io_class": None,
        "agx_model": None,
        "metal_plugin_name": None,
        "probe_source": "unavailable",
    }
    try:
        raw = subprocess.check_output(
            ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-a"],
            stderr=subprocess.DEVNULL,
        )
        entries = plistlib.loads(raw)
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                gpu_cores = entry.get("gpu-core-count")
                if isinstance(gpu_cores, int) and gpu_cores > 0:
                    info["gpu_core_count"] = gpu_cores
                    info["agx_io_class"] = entry.get("IOClass")
                    info["agx_model"] = entry.get("model")
                    info["metal_plugin_name"] = entry.get("MetalPluginName")
                    info["probe_source"] = "ioreg_plist"
                    return info
    except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException):
        pass

    # Fallback: system_profiler JSON
    sp_text = _run_text(["system_profiler", "SPDisplaysDataType", "-json"])
    if sp_text:
        try:
            payload = json.loads(sp_text)
            for gpu in payload.get("SPDisplaysDataType", []):
                cores = gpu.get("sppci_cores")
                if cores is not None and str(cores).isdigit():
                    info["gpu_core_count"] = int(str(cores))
                    info["agx_model"] = gpu.get("sppci_model")
                    info["probe_source"] = "system_profiler"
                    return info
        except (ValueError, TypeError):
            pass
    return info


def probe_silicon_facts() -> dict[str, Any]:
    """Collect safe live hardware facts on the current macOS host."""
    chip = _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]) or "unknown"
    agx = probe_agx_accelerator()
    mem_bytes = _run_int(["sysctl", "-n", "hw.memsize"])
    return {
        "chip": chip,
        "cpu_total_cores": _run_int(["sysctl", "-n", "hw.ncpu"]),
        "cpu_perf_cores": _run_int(["sysctl", "-n", "hw.perflevel0.logicalcpu"]),
        "cpu_eff_cores": _run_int(["sysctl", "-n", "hw.perflevel1.logicalcpu"]),
        "gpu_core_count": agx["gpu_core_count"],
        "agx_io_class": agx["agx_io_class"],
        "metal_plugin_name": agx["metal_plugin_name"],
        "gpu_probe_source": agx["probe_source"],
        "system_memory_bytes": mem_bytes,
        "system_memory_gib": round(mem_bytes / GIB, 2) if mem_bytes else None,
        "macos": _run_text(["sw_vers", "-productVersion"]),
        "macos_build": _run_text(["sw_vers", "-buildVersion"]),
        "xcode": _run_text(["xcodebuild", "-version"]),
        "macos_sdk": _run_text(
            ["xcrun", "-sdk", "macosx", "--show-sdk-version"]
        ),
    }


def _synthesize_m5_ratio_prior(chip: str, gpu_cores: int) -> SiliconPreset:
    """Proportional theoretical prior for an unlisted Apple M5 GPU bin."""
    lower = chip.casefold()
    if "ultra" in lower:
        variant = f"Ultra-{gpu_cores}GPU"
        die_count = 4
        topology = "ultrafusion_quad_die"
        bw = 1200.0
        tier = "Tier A (unlisted Ultra bin; prior only)"
    elif "max" in lower:
        variant = f"Max-{gpu_cores}GPU"
        die_count = 2
        topology = "fusion_dual_die"
        bw = 614.0
        tier = "Tier A/B (unlisted Max bin; prior only)"
    elif "pro" in lower:
        variant = f"Pro-{gpu_cores}GPU"
        die_count = 2
        topology = "fusion"
        bw = 307.0
        tier = "Tier B (unlisted Pro bin; prior only)"
    else:
        variant = f"Base-{gpu_cores}GPU"
        die_count = 1
        topology = "single_die"
        bw = 153.0
        tier = "Tier B/C (unlisted M5 bin; prior only)"

    compute_scale = round(gpu_cores / 80.0, 4)
    bw_scale = round(bw / 1200.0, 4)
    split_k = max(1, int(round(8 * compute_scale)))
    split_kv_parts = max(2, int(round(16 * compute_scale)))
    split_kv_min_ctx = 8192 if gpu_cores >= 56 else (16384 if gpu_cores >= 32 else 32768)
    nax_min_t = 1 if gpu_cores >= 32 else (4 if gpu_cores >= 16 else 8)
    draft_k = 7 if bw >= 1000.0 else (5 if bw >= 500.0 else (4 if bw >= 250.0 else 3))
    slug = re.sub(r"[^a-z0-9]+", "-", variant.casefold()).strip("-")
    return SiliconPreset(
        profile_id=f"apple-m5-{slug}",
        chip_family="M5",
        chip_variant=variant,
        generation="Apple10",
        support_tier_prior=tier,
        gpu_cores_nominal=gpu_cores,
        nax_units_nominal=gpu_cores,
        die_count_nominal=die_count,
        fusion_topology=topology,
        memory_bandwidth_gbs_nominal=bw,
        compute_scale_vs_ultra_80=compute_scale,
        bandwidth_scale_vs_ultra_80=bw_scale,
        linear_nax_min_tokens_prior=nax_min_t,
        linear_split_k_target_slices_prior=split_k,
        attention_split_kv_min_context_prior=split_kv_min_ctx,
        attention_split_kv_partitions_prior=split_kv_parts,
        gdn_prefill_tb8_min_tokens_prior=256 if gpu_cores >= 32 else 512,
        dflash2_draft_tokens_prior=draft_k,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Dynamically synthesized proportional M5 theoretical prior; requires H0/H1/H2 measurement.",
    )


def match_silicon_preset(
    chip: str, gpu_core_count: int | None
) -> tuple[SiliconPreset, str]:
    """Match detected chip string and GPU core count to a SiliconPreset."""
    lower = chip.casefold()
    if "m5 ultra" in lower:
        if gpu_core_count is None:
            return SILICON_PRESETS["apple-m5-ultra-80gpu"], "chip_name_default_gpu_unknown"
        if gpu_core_count == 64:
            return SILICON_PRESETS["apple-m5-ultra-64gpu"], "exact_chip_and_gpu_cores"
        if gpu_core_count == 80:
            return SILICON_PRESETS["apple-m5-ultra-80gpu"], "exact_chip_and_gpu_cores"
        return _synthesize_m5_ratio_prior(chip, gpu_core_count), "dynamic_m5_ratio_prior"

    if "m5 max" in lower:
        if gpu_core_count is None or gpu_core_count == 40:
            return SILICON_PRESETS["apple-m5-max-40gpu"], (
                "exact_chip_and_gpu_cores"
                if gpu_core_count == 40
                else "chip_name_default_gpu_unknown"
            )
        if gpu_core_count == 32:
            return SILICON_PRESETS["apple-m5-max-32gpu"], "exact_chip_and_gpu_cores"
        return _synthesize_m5_ratio_prior(chip, gpu_core_count), "dynamic_m5_ratio_prior"

    if "m5 pro" in lower:
        if gpu_core_count is None or gpu_core_count == 20:
            return SILICON_PRESETS["apple-m5-pro-20gpu"], (
                "exact_chip_and_gpu_cores"
                if gpu_core_count == 20
                else "chip_name_default_gpu_unknown"
            )
        if gpu_core_count == 16:
            return SILICON_PRESETS["apple-m5-pro-16gpu"], "exact_chip_and_gpu_cores"
        return _synthesize_m5_ratio_prior(chip, gpu_core_count), "dynamic_m5_ratio_prior"

    if re.search(r"\bm5\b", lower):
        if gpu_core_count is None or gpu_core_count == 10:
            return SILICON_PRESETS["apple-m5-10gpu"], (
                "exact_chip_and_gpu_cores"
                if gpu_core_count == 10
                else "chip_name_default_gpu_unknown"
            )
        return _synthesize_m5_ratio_prior(chip, gpu_core_count), "dynamic_m5_ratio_prior"

    if "m4" in lower:
        return SILICON_PRESETS["apple-m4-dev"], "exact_chip_family_dev_host"

    fallback = SiliconPreset(
        profile_id="apple-unknown-host",
        chip_family="unknown",
        chip_variant="unknown",
        generation="unknown",
        support_tier_prior="Unsupported / unverified",
        gpu_cores_nominal=gpu_core_count or 0,
        nax_units_nominal=0,
        die_count_nominal=1,
        fusion_topology="unknown",
        memory_bandwidth_gbs_nominal=0.0,
        compute_scale_vs_ultra_80=0.0,
        bandwidth_scale_vs_ultra_80=0.0,
        linear_nax_min_tokens_prior=0,
        linear_split_k_target_slices_prior=1,
        attention_split_kv_min_context_prior=32768,
        attention_split_kv_partitions_prior=2,
        gdn_prefill_tb8_min_tokens_prior=512,
        dflash2_draft_tokens_prior=0,
        moe_grouped_vs_fallback_active_tokens_prior=8,
        notes="Unrecognized host silicon; fallback safe profile.",
    )
    return fallback, "unrecognized_host"


def estimate_memory_budget(
    system_memory_bytes: int | None,
    measured_working_set_bytes: int | None = None,
) -> dict[str, Any]:
    """Compute Metal working-set estimate and safe B=1..8 admission matrix."""
    if not system_memory_bytes or system_memory_bytes <= 0:
        return {
            "working_set_source": "unknown",
            "estimated_metal_working_set_gib": None,
            "recommended_os_headroom_gib": None,
            "usable_inference_budget_gib": None,
            "prefix_cache_budget_bytes_prior": 256 * (1 << 20),
            "safe_max_concurrency": {},
        }

    total_gib = system_memory_bytes / GIB
    if measured_working_set_bytes and measured_working_set_bytes > 0:
        ws_gib = measured_working_set_bytes / GIB
        ws_source = "measured_h0_metal_probe"
    else:
        # macOS Apple Silicon default recommendedMaxWorkingSetSize heuristic:
        # ~75% of RAM for >=64 GiB systems, ~66.7% for smaller unified memory.
        ratio = 0.75 if total_gib >= 64.0 else (2.0 / 3.0)
        ws_gib = total_gib * ratio
        ws_source = "theoretical_macos_ratio_prior"

    if total_gib >= 192.0:
        headroom_gib = 24.0
        prefix_bytes = 1 << 30  # 1 GiB
    elif total_gib >= 90.0:
        headroom_gib = 16.0
        prefix_bytes = 512 * (1 << 20)  # 512 MiB
    elif total_gib >= 48.0:
        headroom_gib = 10.0
        prefix_bytes = 256 * (1 << 20)
    else:
        headroom_gib = 6.0
        prefix_bytes = 128 * (1 << 20)

    usable_gib = max(0.0, ws_gib - headroom_gib)
    prefix_gib = prefix_bytes / GIB

    safe_matrix: dict[str, dict[str, dict[str, Any]]] = {}
    for workload_id, spec in WORKLOAD_MEMORY_MODEL.items():
        weights = float(spec["shared_weights_gib"])
        ctx_map: dict[str, dict[str, Any]] = {}
        for ctx_label, per_slot in spec["per_slot_gib"].items():
            remaining = usable_gib - weights - prefix_gib
            if remaining < per_slot:
                max_b = 0
                status = "exceeds_budget"
            else:
                max_b = min(8, int(math.floor(remaining / per_slot)))
                status = "full_b8_ok" if max_b == 8 else f"bounded_b{max_b}"
            ctx_map[ctx_label] = {
                "max_safe_concurrency": max_b,
                "b1_estimated_gib": round(weights + per_slot + prefix_gib, 2),
                "b4_estimated_gib": round(weights + 4 * per_slot + prefix_gib, 2),
                "b8_estimated_gib": round(weights + 8 * per_slot + prefix_gib, 2),
                "status": status,
            }
        safe_matrix[workload_id] = ctx_map

    return {
        "working_set_source": ws_source,
        "system_memory_gib": round(total_gib, 2),
        "estimated_metal_working_set_gib": round(ws_gib, 2),
        "recommended_os_headroom_gib": headroom_gib,
        "usable_inference_budget_gib": round(usable_gib, 2),
        "prefix_cache_budget_bytes_prior": prefix_bytes,
        "safe_max_concurrency": safe_matrix,
    }


def build_result_namespace(
    preset: SiliconPreset, system_memory_bytes: int | None
) -> str:
    """Create a collision-free result directory slug per silicon SKU + RAM size."""
    base = preset.profile_id.removeprefix("apple-")
    if system_memory_bytes and system_memory_bytes > 0:
        ram_gb = int(round(system_memory_bytes / GIB))
        return f"{base}-{ram_gb}g"
    return base


def _apply_calibration_overlay(
    effective_dispatch: dict[str, Any],
    calibration_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"calibration file must be a JSON object: {calibration_path}")
    overrides = payload.get("measured_dispatch_overrides", {})
    if not isinstance(overrides, dict):
        raise ValueError("measured_dispatch_overrides must be an object")

    diff: dict[str, dict[str, Any]] = {}
    merged = copy.deepcopy(effective_dispatch)
    for key, measured_val in overrides.items():
        if measured_val is None:
            continue
        prior_val = merged.get(key)
        merged[key] = measured_val
        diff[key] = {"prior": prior_val, "measured": measured_val}

    meta = {
        "calibration_file": str(calibration_path.resolve()),
        "calibration_status": (
            "measured_calibrated" if diff else "calibration_file_empty_prior_active"
        ),
        "measured_at": payload.get("measured_at"),
        "source_h0_report": payload.get("source_h0_report"),
        "source_h1_report": payload.get("source_h1_report"),
        "source_h2_report": payload.get("source_h2_report"),
        "prior_vs_measured_diff": diff,
    }
    return merged, meta


def resolve_silicon_profile(
    facts: dict[str, Any] | None = None,
    *,
    calibration_path: Path | None = None,
    auto_load_calibration: bool = True,
    measured_working_set_bytes: int | None = None,
) -> dict[str, Any]:
    """Resolve full DeviceProfile + theoretical presets + optional measured calibration."""
    hw = dict(facts) if facts is not None else probe_silicon_facts()
    chip = str(hw.get("chip") or "unknown")
    gpu_cores = hw.get("gpu_core_count")
    if isinstance(gpu_cores, str) and gpu_cores.isdigit():
        gpu_cores = int(gpu_cores)
    elif not isinstance(gpu_cores, int):
        gpu_cores = None

    preset, matched_by = match_silicon_preset(chip, gpu_cores)
    mem_bytes = hw.get("system_memory_bytes")
    if isinstance(mem_bytes, str) and mem_bytes.isdigit():
        mem_bytes = int(mem_bytes)
    elif not isinstance(mem_bytes, int):
        mem_bytes = None

    mem_budget = estimate_memory_budget(
        mem_bytes, measured_working_set_bytes=measured_working_set_bytes
    )
    result_ns = build_result_namespace(preset, mem_bytes)

    theoretical_dispatch = {
        "linear_nax_min_tokens": preset.linear_nax_min_tokens_prior,
        "linear_split_k_target_slices": preset.linear_split_k_target_slices_prior,
        "attention_split_kv_min_context": preset.attention_split_kv_min_context_prior,
        "attention_split_kv_partitions": preset.attention_split_kv_partitions_prior,
        "gdn_prefill_tb8_min_tokens": preset.gdn_prefill_tb8_min_tokens_prior,
        "dflash2_draft_tokens": preset.dflash2_draft_tokens_prior,
        "moe_grouped_vs_fallback_active_tokens": preset.moe_grouped_vs_fallback_active_tokens_prior,
        "prefix_cache_budget_bytes": mem_budget["prefix_cache_budget_bytes_prior"],
    }

    resolved_cal_path = calibration_path
    if resolved_cal_path is None and auto_load_calibration:
        candidate = DEFAULT_CALIBRATION_DIR / f"{preset.profile_id}.json"
        if candidate.is_file():
            resolved_cal_path = candidate

    if resolved_cal_path is not None and resolved_cal_path.is_file():
        effective_dispatch, calibration_meta = _apply_calibration_overlay(
            theoretical_dispatch, resolved_cal_path
        )
    else:
        effective_dispatch = copy.deepcopy(theoretical_dispatch)
        calibration_meta = {
            "calibration_file": None,
            "calibration_status": "uncalibrated_prior",
            "prior_vs_measured_diff": {},
        }

    is_m5_family = preset.chip_family == "M5"
    is_m5_ultra = "m5 ultra" in chip.casefold()
    is_full_core_ultra = preset.profile_id == "apple-m5-ultra-80gpu"

    return {
        "schema": "omlx-m5-silicon-profile-v1",
        "profile_id": preset.profile_id,
        "matched_by": matched_by,
        "result_namespace": result_ns,
        "parameter_origin": (
            "measured_calibration_overlay"
            if calibration_meta["calibration_status"] == "measured_calibrated"
            else "theoretical_prior"
        ),
        "calibration_status": calibration_meta["calibration_status"],
        "device_performance_qualified": False,
        "production_enabled": False,
        "target_flags": {
            "is_m5_family": is_m5_family,
            "is_m5_ultra": is_m5_ultra,
            "is_m5_ultra_full_core_80gpu": is_full_core_ultra,
            "is_m5_ultra_binned_64gpu": preset.profile_id == "apple-m5-ultra-64gpu",
        },
        "detected_hardware": hw,
        "silicon_spec_prior": {
            "chip_family": preset.chip_family,
            "chip_variant": preset.chip_variant,
            "generation": preset.generation,
            "support_tier_prior": preset.support_tier_prior,
            "gpu_cores_nominal": preset.gpu_cores_nominal,
            "nax_units_nominal": preset.nax_units_nominal,
            "die_count_nominal": preset.die_count_nominal,
            "fusion_topology": preset.fusion_topology,
            "memory_bandwidth_gbs_nominal": preset.memory_bandwidth_gbs_nominal,
            "compute_scale_vs_ultra_80": preset.compute_scale_vs_ultra_80,
            "bandwidth_scale_vs_ultra_80": preset.bandwidth_scale_vs_ultra_80,
            "notes": preset.notes,
        },
        "memory_budget": mem_budget,
        "theoretical_dispatch_prior": theoretical_dispatch,
        "effective_dispatch_parameters": effective_dispatch,
        "calibration": calibration_meta,
        "recommended_paths": {
            "h0_result_dir": f"bench/results/{result_ns}/h0",
            "h1_result_dir": f"bench/results/{result_ns}/h1",
            "h2_result_dir": f"bench/results/{result_ns}/h2",
            "calibration_overlay_path": f"docs/backend/silicon-calibrations/{preset.profile_id}.json",
            "static_dispatch_table_path": f"docs/backend/m5-static-dispatch-table-{preset.profile_id}-v1.json",
        },
    }


def build_calibration_template(profile: dict[str, Any]) -> dict[str, Any]:
    """Generate a blank calibration overlay template for post-H0/H1/H2 sign-off."""
    prior = profile["theoretical_dispatch_prior"]
    return {
        "schema": "omlx-m5-silicon-calibration-v1",
        "profile_id": profile["profile_id"],
        "result_namespace": profile["result_namespace"],
        "measured_at": None,
        "macos_build": profile["detected_hardware"].get("macos_build"),
        "xcode": profile["detected_hardware"].get("xcode"),
        "source_h0_report": f"{profile['recommended_paths']['h0_result_dir']}/capability.json",
        "source_h1_report": f"{profile['recommended_paths']['h1_result_dir']}/tuning-summary.json",
        "source_h2_report": f"{profile['recommended_paths']['h2_result_dir']}/full-model.json",
        "measured_hardware_constants": {
            "recommended_working_set_bytes": None,
            "max_buffer_length": None,
            "copy_bandwidth_64mb_gbs": None,
            "copy_bandwidth_256mb_gbs": None,
            "native_tensorops_runtime_verified": False,
        },
        "theoretical_prior_reference": prior,
        "measured_dispatch_overrides": {key: None for key in prior},
        "notes": "Populate measured_dispatch_overrides only from verified H0/H1/H2 evidence on this physical silicon SKU.",
    }


def atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def emit_shell_env(profile: dict[str, Any]) -> str:
    paths = profile["recommended_paths"]
    eff = profile["effective_dispatch_parameters"]
    exports = {
        "M5_SILICON_PROFILE_ID": profile["profile_id"],
        "M5_RESULT_NAMESPACE": profile["result_namespace"],
        "M5_CALIBRATION_STATUS": profile["calibration_status"],
        "M5_H0_RESULT_DIR": str(ROOT / paths["h0_result_dir"]),
        "M5_H1_RESULT_DIR": str(ROOT / paths["h1_result_dir"]),
        "M5_H2_RESULT_DIR": str(ROOT / paths["h2_result_dir"]),
        "M5_FULL_MODEL_RESULT": str(
            ROOT / paths["h2_result_dir"] / "full-model.json"
        ),
        "M5_PRIOR_DFLASH2_DRAFT_TOKENS": str(eff["dflash2_draft_tokens"]),
        "M5_PRIOR_PREFIX_CACHE_BYTES": str(eff["prefix_cache_budget_bytes"]),
        "M5_PRIOR_SPLIT_KV_MIN_CONTEXT": str(
            eff["attention_split_kv_min_context"]
        ),
    }
    return "\n".join(
        f"export {key}={shlex.quote(val)}" for key, val in exports.items()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List all built-in silicon SKU theoretical presets and exit.",
    )
    parser.add_argument(
        "--simulate-chip",
        default=None,
        help="Override detected chip string for dry-run planning (e.g. 'Apple M5 Ultra').",
    )
    parser.add_argument(
        "--simulate-gpu-cores",
        type=int,
        default=None,
        help="Override detected GPU core count for dry-run planning (e.g. 64 or 80).",
    )
    parser.add_argument(
        "--simulate-memory-gb",
        type=float,
        default=None,
        help="Override detected unified memory in GiB for dry-run planning (e.g. 96 or 256).",
    )
    parser.add_argument(
        "--simulate-cpu-cores",
        type=int,
        default=None,
        help="Override detected CPU core count for dry-run planning (e.g. 30 or 36).",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help="Explicit measured calibration overlay JSON path.",
    )
    parser.add_argument(
        "--no-auto-calibration",
        action="store_true",
        help="Ignore any auto-discovered file in docs/backend/silicon-calibrations/.",
    )
    parser.add_argument(
        "--emit-calibration-template",
        type=Path,
        default=None,
        help="Write a blank calibration overlay template for the resolved profile.",
    )
    parser.add_argument(
        "--shell-env",
        action="store_true",
        help="Print shell export lines for isolated result directories and priors.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write the resolved profile JSON to this file.",
    )
    args = parser.parse_args()

    if args.list_profiles:
        catalog = {
            "schema": "omlx-m5-silicon-preset-catalog-v1",
            "parameter_origin": "theoretical_prior",
            "profiles": {
                pid: {
                    "chip_family": p.chip_family,
                    "chip_variant": p.chip_variant,
                    "generation": p.generation,
                    "gpu_cores_nominal": p.gpu_cores_nominal,
                    "nax_units_nominal": p.nax_units_nominal,
                    "fusion_topology": p.fusion_topology,
                    "memory_bandwidth_gbs_nominal": p.memory_bandwidth_gbs_nominal,
                    "compute_scale_vs_ultra_80": p.compute_scale_vs_ultra_80,
                    "bandwidth_scale_vs_ultra_80": p.bandwidth_scale_vs_ultra_80,
                    "support_tier_prior": p.support_tier_prior,
                    "theoretical_dispatch_prior": {
                        "linear_nax_min_tokens": p.linear_nax_min_tokens_prior,
                        "linear_split_k_target_slices": p.linear_split_k_target_slices_prior,
                        "attention_split_kv_min_context": p.attention_split_kv_min_context_prior,
                        "attention_split_kv_partitions": p.attention_split_kv_partitions_prior,
                        "gdn_prefill_tb8_min_tokens": p.gdn_prefill_tb8_min_tokens_prior,
                        "dflash2_draft_tokens": p.dflash2_draft_tokens_prior,
                        "moe_grouped_vs_fallback_active_tokens": p.moe_grouped_vs_fallback_active_tokens_prior,
                    },
                }
                for pid, p in sorted(SILICON_PRESETS.items())
            },
        }
        print(json.dumps(catalog, indent=2, sort_keys=True))
        return 0

    simulated = any(
        val is not None
        for val in (
            args.simulate_chip,
            args.simulate_gpu_cores,
            args.simulate_memory_gb,
            args.simulate_cpu_cores,
        )
    )
    if simulated:
        facts: dict[str, Any] = {
            "chip": args.simulate_chip or "Apple M5 Ultra",
            "cpu_total_cores": args.simulate_cpu_cores,
            "cpu_perf_cores": None,
            "cpu_eff_cores": None,
            "gpu_core_count": args.simulate_gpu_cores,
            "agx_io_class": "simulated",
            "metal_plugin_name": "simulated",
            "gpu_probe_source": "cli_simulation",
            "system_memory_bytes": (
                int(args.simulate_memory_gb * GIB)
                if args.simulate_memory_gb is not None
                else None
            ),
            "system_memory_gib": args.simulate_memory_gb,
            "macos": "simulated",
            "macos_build": "simulated",
            "xcode": "simulated",
            "macos_sdk": "simulated",
        }
    else:
        facts = probe_silicon_facts()

    profile = resolve_silicon_profile(
        facts,
        calibration_path=args.calibration,
        auto_load_calibration=not args.no_auto_calibration,
    )
    if simulated:
        profile["simulated"] = True

    if args.emit_calibration_template is not None:
        template = build_calibration_template(profile)
        atomic_json(args.emit_calibration_template.resolve(), template)

    if args.output is not None:
        atomic_json(args.output.resolve(), profile)

    if args.shell_env:
        print(emit_shell_env(profile))
    else:
        print(json.dumps(profile, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
