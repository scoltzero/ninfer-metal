# SPDX-License-Identifier: Apache-2.0
"""miv Core Backend (`ninfer-metal`) external sidecar engine (`mseries_backendd` over Unix domain sockets)
and lightweight `entry_points(group="omlx.external_engines")` plugin discovery.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import importlib.metadata
import json
import logging
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
import zlib
from collections import OrderedDict
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..api.thinking import (
    _convert_token_to_id,
    _encode_prompt_ids,
    _safe_tokenizer_attr,
)
from ..api.tool_calling import convert_tools_for_template
from ..api.utils import (
    _consolidate_system_messages,
    detect_and_strip_partial,
)
from ..exceptions import InvalidRequestError, ModelLoadingError
from ..lora_adapters import (
    estimate_sidecar_lora_bytes,
    fingerprint as lora_fingerprint,
    resolve_adapter_path,
)
from ..reasoning_effort import apply_chat_template_with_reasoning_effort_fallback
from ..request import DecodeMarks, add_decode_mark, live_decode_rate
from ..utils.proc_memory import get_phys_footprint, list_process_group_pids
from ..utils.tokenizer import create_streaming_detokenizer, get_tokenizer_config
from .base import BaseEngine, GenerationOutput
from .splash import EXTERNAL_ENGINES

logger = logging.getLogger(__name__)

EXTERNAL_ENGINE_ENTRY_POINTS_GROUP = "omlx.external_engines"
M5RT_ENGINE_ID = "m5rt"
M5RT_ENGINE_LABEL = "miv Core Backend"
M5RT_DEFAULT_VERSION = "miv-v0.03"
M5RT_PROVIDER_ID = "io.ninfer.metal"
M5RT_LEGACY_PROVIDER_ID = "com.omlx.mseries"
M5RT_BUILTIN_PROVIDER_IDS = frozenset({M5RT_PROVIDER_ID, M5RT_LEGACY_PROVIDER_ID})
M5RT_OPTION_KEYS = (
    M5RT_LEGACY_PROVIDER_ID,
    M5RT_PROVIDER_ID,
    "ninfer-metal",
    M5RT_ENGINE_ID,
)
M5RT_SUPPORTED_TOPOLOGIES: tuple[str, ...] = (
    "qwen35_hybrid_dense_v1",
    "qwen36_hybrid_moe256_k8_shared_v1",
)
M5RT_SUPPORTED_MODEL_TYPES: tuple[str, ...] = (
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_6",
    "qwen3_6_moe",
    "qwen3_8",
    "qwen3_8_moe",
)
# Companion DFlash2 draft model whitelist for miv-supported Qwen3.5 / Qwen3.6 / Qwen3.8
# Dense (0.8B / 2B / 4B / 9B / 27B) and MoE-256 (35B-A3B / 122B-A10B) target families.
M5RT_SUPPORTED_DFLASH2_ARCHITECTURES: tuple[str, ...] = ("DFlash2DraftModel",)
M5RT_SUPPORTED_DFLASH2_MODEL_TYPES: tuple[str, ...] = ("qwen3",)
M5RT_SUPPORTED_DFLASH2_TARGET_GEOMETRIES: frozenset[tuple[int, int]] = frozenset(
    {
        (1024, 24),  # Qwen3.5 / Qwen3.6 / Qwen3.8 0.8B Dense
        (2048, 24),  # Qwen3.5 / Qwen3.6 / Qwen3.8 2B Dense
        (2560, 32),  # Qwen3.5 / Qwen3.6 / Qwen3.8 4B Dense
        (4096, 32),  # Qwen3.5 / Qwen3.6 / Qwen3.8 9B Dense
        (5120, 64),  # Qwen3.5 / Qwen3.6 / Qwen3.8 27B Dense
        (2048, 40),  # Qwen3.5 / Qwen3.6 / Qwen3.8 35B-A3B MoE (256x k=8)
        (3072, 48),  # Qwen3.5 / Qwen3.6 / Qwen3.8 122B-A10B MoE (256x k=8)
    }
)
M5RT_SUPPORTED_DFLASH2_REPOS: tuple[str, ...] = (
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
)
M5RT_ARTIFACT_EXTENSIONS: tuple[str, ...] = (".miv", ".omlxm5")
M5RT_DEFAULT_CONVERTER: dict[str, Any] = {
    "entrypoint": "tools/m5_artifact.py",
    "subcommand": "convert",
    "output_filename": "model.miv",
    "artifact_extension": ".miv",
    "default_args": ["--nax-pretile"],
    "supported_quant_modes": [
        "affine",
        "mxfp4",
        "mxfp4-pack-quantized",
        "bf16",
        "fp16",
        "mixed",
    ],
}
M5RT_DEFAULT_ARCHITECTURE_CAPABILITIES: dict[str, Any] = {
    "topologies": ["gdn_hybrid_gqa", "pure_gqa_swiglu", "dflash2_draft"],
    "model_type_whitelist_enforced": False,
    "attn_output_gate": [True, False],
    "qk_norm": [True, False],
    "head_dims": [64, 128, 256],
    "moe": {
        "min_experts": 2,
        "max_experts": 256,
        "min_top_k": 1,
        "max_top_k": 8,
        "shared_expert": "optional",
    },
    "quantization": {
        "affine_bits": [3, 4, 5, 6, 8],
        "affine_group_sizes": [64, 128],
        "mxfp4_group_sizes": [32],
        "direct_dtypes": ["bf16", "f16"],
        "per_tensor_mixed_precision": True,
        "strip_non_text_vlm": True,
    },
}
M5RT_DEFAULT_CAPABILITIES: tuple[str, ...] = (
    "ar",
    "mtp",
    "dflash2",
    "lora",
    "reload_lora",
    "kv_affine4_g128",
    "rolling_prefix_cache",
    "trim_cache",
    "memory_budget_bytes",
    "set_scheduler_hint",
    "cancel",
    "stats",
    "capacity",
    "thinking_budget",
    "sampling_penalties",
    "subscribe_events",
)
M5RT_SUPPORTED_STANDARD_SETTINGS: tuple[str, ...] = (
    "max_context_window",
    "mtp_enabled",
    "mtp_adaptive_max_depth",
    "mtp_fixed_depth",
    "dflash_enabled",
    "dflash_draft_model",
    "dflash_block_size",
    "turboquant_kv_enabled",
    "turboquant_kv_scheme",
    "lora_adapter",
)
M5RT_DEFAULT_OPTIONS_SCHEMA: tuple[dict[str, Any], ...] = (
    {
        "key": "max_concurrency",
        "label": "Concurrency (B/C)",
        "type": "int",
        "min": 1,
        "max": 8,
        "default": 4,
        "reload_required": True,
    },
    {
        "key": "max_context",
        "label": "Max Context Window",
        "type": "int",
        "min": 1024,
        "max": 1048576,
        "default": 262144,
        "description": (
            "Maximum sequence context length in tokens (supports 256K = 262144 "
            "and above when model permits)."
        ),
        "reload_required": True,
    },
    {
        "key": "rolling_prefix_cache",
        "label": "Rolling Prefix Cache",
        "type": "bool",
        "default": True,
        "reload_required": True,
    },
    {
        "key": "prefix_cache_gb",
        "label": "Prefix Cache Budget (GiB)",
        "type": "bytes_gb",
        "min": 0,
        "max": 128,
        "default": 4,
        "reload_required": True,
    },
    {
        "key": "prefill_linear_mode",
        "label": "W4A8 Linear",
        "type": "enum",
        "options": ["bf16", "w4a8"],
        "option_labels": {
            "bf16": "BF16 (Exact)",
            "w4a8": "W4A8 (INT8 Prefill)",
        },
        "description": (
            "INT8 prefill linear matmul (strongest on oQ4e; down_proj stays W4A16). "
            "Decode remains exact W4A16."
        ),
        "default": "bf16",
        "reload_required": True,
    },
    {
        "key": "prefill_attention_mode",
        "label": "Sage2 Attention",
        "type": "enum",
        "options": ["exact_nax", "sage2"],
        "option_labels": {
            "exact_nax": "Exact NAX",
            "sage2": "Sage2 (INT8 QK + FP16 PV)",
        },
        "description": (
            "Low-bit INT8 QK prefill attention; pair with MInference below "
            "(dense Sage2 alone is slower on affine4 KV)."
        ),
        "default": "exact_nax",
        "reload_required": True,
    },
    {
        "key": "prefill_sparse_minference",
        "label": "MInference Sparse",
        "type": "bool",
        "description": (
            "Skips low-weight KV blocks (2.1x-2.35x faster at 64K-256K). "
            "Slight quality loss, but turning it off under Sage2 causes severe slowdown."
        ),
        "default": False,
        "reload_required": True,
    },
    {
        "key": "prefill_cache_blend",
        "label": "CacheBlend Reuse",
        "type": "bool",
        "description": (
            "Non-contiguous RAG chunk KV+GDN reuse (4.3x-4.5x warmer TTFT). "
            "Increase Prefix Cache Budget (e.g. 16-32 GiB) for long docs."
        ),
        "default": False,
        "reload_required": True,
    },
    {
        "key": "prefill_ane_offload",
        "label": "Dual-ANE Offload",
        "type": "bool",
        "description": (
            "Orthogonal Dual-ANE co-execution (never disables W4A8/Sage2); "
            "Dense-only and skips GDN recurrence. Pure GPU is faster on 80-core M5 Ultra."
        ),
        "default": False,
        "reload_required": True,
    },
    {
        "key": "lora_mode",
        "label": "LoRA Mode",
        "type": "enum",
        "options": ["merge", "bypass"],
        "option_labels": {
            "merge": "Merge (Load-Time UMA Fusion)",
            "bypass": "Bypass (Runtime BF16 Low-Rank)",
        },
        "description": (
            "Merge fuses LoRA into 8-bit NAX tiles at load time with zero runtime overhead; "
            "Bypass injects low-rank BF16 projections at runtime with fast hot-swap."
        ),
        "default": "merge",
        "reload_required": True,
    },
    {
        "key": "speculative_backend",
        "label": "Speculative Backend",
        "type": "enum",
        "options": ["auto", "none", "mtp", "dflash2"],
        "default": "auto",
        "reload_required": True,
    },
    {
        "key": "sampling_contract",
        "label": "Sampling Contract",
        "type": "enum",
        "options": ["position_coupled_v1", "legacy_rejection_v1"],
        "option_labels": {
            "position_coupled_v1": "Position Coupled v1 (GPU Fast Path)",
            "legacy_rejection_v1": "Legacy Rejection v1 (Debug / Slow)",
        },
        "description": (
            "Position Coupled v1 keeps speculative verification on GPU with zero host logit readback. "
            "Legacy Rejection v1 is for debug parity only and substantially slower."
        ),
        "default": "position_coupled_v1",
        "reload_required": True,
    },
    {
        "key": "artifact_path",
        "label": "Artifact Path",
        "type": "string",
        "default": "",
        "reload_required": True,
    },
    {
        "key": "draft_artifact_path",
        "label": "Draft Artifact Path",
        "type": "string",
        "default": "",
        "reload_required": True,
    },
)
M5RT_INSTALL_HINT = (
    "miv Core Backend (ninfer-metal) daemon (mseries_backendd) not found; set integrations.m5rt_path, "
    "OMLX_M5RT_BIN, install provider io.ninfer.metal, or build build/release/miv/mseries_backendd"
)

_BREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
_USER_BIN_DIR = Path.home() / ".local" / "bin"
_OMLX_APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "oMLX"
_PROVIDERS_BASE_DIR = _OMLX_APP_SUPPORT_DIR / "Providers"
_PROVIDER_BIN_DIR = _PROVIDERS_BASE_DIR / M5RT_LEGACY_PROVIDER_ID / "bin"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DAEMON_RELPATHS = (
    Path("build/release/miv/mseries_backendd"),
    Path("build/miv-v001/miv/mseries_backendd"),
    Path("build/b6-xcode27/miv/mseries_backendd"),
    Path("build/d0-m4/miv/mseries_backendd"),
    Path("build/b6-xcode27/m5rt/mseries_backendd"),
    Path("build/d0-m4/m5rt/mseries_backendd"),
)
_DEFAULT_ARTIFACT_CACHE_RELPATH = Path(".cache/b0/artifacts")

UDS_MAX_LINE_BYTES = 16 * 1024 * 1024
STARTUP_TIMEOUT_SECONDS = 300.0
CONNECT_TIMEOUT_SECONDS = 15.0
CONNECT_POLL_SECONDS = 0.02
STOP_TIMEOUT_SECONDS = 10.0
STOP_POLL_SECONDS = 0.05
WATCHDOG_POLL_SECONDS = 2
STATS_POLL_SECONDS = 5.0
VERSION_TIMEOUT_SECONDS = 10.0
VERSION_RETRY_SECONDS = 30.0
LOG_TAIL_LINES = 8

SUPPORTED_PROTOCOLS = frozenset({1, 2})
VALID_KV_DTYPES = frozenset({"bf16", "affine4_g128"})
VALID_PREFILL_ENGINES = frozenset({"auto", "nax", "nax_ane", "flash_nax"})
VALID_PREFILL_LINEAR_MODES = frozenset({"bf16", "w4a8"})
VALID_PREFILL_ATTENTION_MODES = frozenset({"exact_nax", "sage2"})
M5RT_DEFAULT_SAMPLING_CONTRACT = "position_coupled_v1"
M5RT_MAX_DRAFT_TOKENS = {"mtp": 4, "dflash2": 7}
M5RT_MAX_REPETITION_CONTEXT_SIZE = 4096
M5RT_REQUEST_SETTINGS = ("max_tokens", "temperature", "top_p", "top_k")
VALID_SAMPLING_CONTRACTS = frozenset({"legacy_rejection_v1", "position_coupled_v1"})
VALID_SPEC_BACKENDS = frozenset({"auto", "none", "dflash2", "mtp"})
VALID_LORA_MODES = frozenset({"merge", "bypass"})


# -- External engine plugin discovery ------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """Fast O(1) filesystem probe result for automatic model routing."""

    adapted: bool
    artifact_path: Path | None = None
    draft_path: Path | None = None
    reason: str = ""
    explicit_provider_match: bool = False


@dataclass(frozen=True)
class ExternalEnginePlugin:
    """Descriptor for an external sidecar engine registered via entry_points."""

    id: str
    label: str
    factory: Callable[..., BaseEngine]
    validator: Callable[..., tuple[bool, str]] | None = None
    status: Callable[..., dict[str, Any]] | None = None
    estimate_memory_bytes: Callable[..., int | None] | None = None
    runtime_signature: Callable[..., Any] | None = None
    supported_standard_settings: tuple[str, ...] = ()
    detect_adapted: Callable[..., ProbeResult | bool] | None = None
    can_auto_convert_fast: Callable[..., bool] | None = None
    can_convert: Callable[..., tuple[bool, str]] | None = None
    convert_model: Callable[..., dict[str, Any]] | None = None
    options_schema: tuple[dict[str, Any], ...] = ()
    provider_id: str | None = None


def _coerce_plugin(raw: Any, fallback_name: str) -> ExternalEnginePlugin | None:
    if callable(raw) and not hasattr(raw, "factory") and not isinstance(raw, dict):
        raw = raw()
    if isinstance(raw, ExternalEnginePlugin):
        return raw
    if isinstance(raw, dict):
        engine_id = str(raw.get("id") or fallback_name).strip()
        label = str(raw.get("label") or engine_id).strip()
        factory = raw.get("factory")
        if not engine_id or not callable(factory):
            return None
        raw_supported = raw.get("supported_standard_settings") or ()
        raw_schema = raw.get("options_schema") or ()
        return ExternalEnginePlugin(
            id=engine_id,
            label=label,
            factory=factory,
            validator=raw.get("validator"),
            status=raw.get("status"),
            estimate_memory_bytes=raw.get("estimate_memory_bytes"),
            runtime_signature=raw.get("runtime_signature"),
            supported_standard_settings=tuple(str(s) for s in raw_supported),
            detect_adapted=raw.get("detect_adapted"),
            options_schema=tuple(dict(s) for s in raw_schema if isinstance(s, dict)),
        )
    engine_id = str(getattr(raw, "id", None) or fallback_name).strip()
    label = str(getattr(raw, "label", None) or engine_id).strip()
    factory = getattr(raw, "factory", None)
    if not engine_id or not callable(factory):
        return None
    raw_supported = getattr(raw, "supported_standard_settings", None) or ()
    raw_schema = getattr(raw, "options_schema", None) or ()
    return ExternalEnginePlugin(
        id=engine_id,
        label=label,
        factory=factory,
        validator=getattr(raw, "validator", None),
        status=getattr(raw, "status", None),
        estimate_memory_bytes=getattr(raw, "estimate_memory_bytes", None),
        runtime_signature=getattr(raw, "runtime_signature", None),
        supported_standard_settings=tuple(str(s) for s in raw_supported),
        detect_adapted=getattr(raw, "detect_adapted", None),
        options_schema=tuple(dict(s) for s in raw_schema if isinstance(s, dict)),
    )


def discover_external_engine_plugins() -> dict[str, ExternalEnginePlugin]:
    """Discover third-party external engines from `entry_points` and SPI `Providers/`."""
    discovered: dict[str, ExternalEnginePlugin] = {}
    try:
        eps = importlib.metadata.entry_points(group=EXTERNAL_ENGINE_ENTRY_POINTS_GROUP)
    except Exception:
        logger.debug("External engine entry_points lookup failed", exc_info=True)
        eps = ()

    for ep in eps:
        try:
            loaded = ep.load()
            plugin = _coerce_plugin(loaded, ep.name)
        except Exception:
            logger.warning(
                "Failed to load external engine plugin %r", ep.name, exc_info=True
            )
            continue
        if plugin is None:
            logger.warning(
                "External engine plugin %r did not expose valid {id, label, factory}",
                ep.name,
            )
            continue
        if plugin.id in EXTERNAL_ENGINES:
            logger.warning(
                "External engine plugin %r collides with built-in engine %r; skipping",
                ep.name,
                plugin.id,
            )
            continue
        discovered[plugin.id] = plugin

    for plugin_id, plugin in _discover_spi_provider_plugins().items():
        if plugin_id in EXTERNAL_ENGINES:
            continue
        if plugin_id in discovered:
            logger.debug(
                "SPI provider plugin %r shadowed by entry_points plugin; keeping entry_points",
                plugin_id,
            )
            continue
        discovered[plugin_id] = plugin
    return discovered


def get_external_engines() -> dict[str, str]:
    """Return `{engine_id: label}` for built-in and plugin external engines."""
    engines = dict(EXTERNAL_ENGINES)
    for plugin_id, plugin in discover_external_engine_plugins().items():
        engines[plugin_id] = plugin.label
    return engines


def canonical_external_engine_id(engine_id: str | None) -> str | None:
    """Normalize built-in or SPI `provider_id` / alias keys to canonical `engine_id`."""
    if not engine_id:
        return None
    cleaned = str(engine_id).strip()
    if not cleaned:
        return None
    if cleaned == "stock":
        return "stock"
    if cleaned in M5RT_OPTION_KEYS:
        return M5RT_ENGINE_ID
    if cleaned == "splash":
        return "splash"
    plugins = discover_external_engine_plugins()
    if cleaned in plugins:
        return cleaned
    for pid_key, plugin in plugins.items():
        if plugin.provider_id == cleaned:
            return pid_key
    manifest = _spi_provider_manifest_for_engine(cleaned)
    if isinstance(manifest, dict):
        eid = str(manifest.get("engine_id") or manifest.get("provider_id") or cleaned).strip()
        if eid:
            return eid
    return cleaned


def get_external_engine_plugin(engine_id: str) -> ExternalEnginePlugin | None:
    """Return a discovered external engine plugin by ID or `provider_id`, or None."""
    if not engine_id or engine_id in EXTERNAL_ENGINES or engine_id in M5RT_OPTION_KEYS:
        return None
    plugins = discover_external_engine_plugins()
    direct = plugins.get(engine_id)
    if direct is not None:
        return direct
    for plugin in plugins.values():
        if plugin.provider_id == engine_id:
            return plugin
    return None


def get_external_engine_supported_settings(engine_id: str | None) -> tuple[str, ...]:
    """Return standard oMLX `ModelSettings` keys honored by `engine_id`.

    Built-in `splash` returns `()` (preserving exact upstream UI lock behavior);
    `m5rt` (`ninfer-metal` / `io.ninfer.metal` / `miv`) returns its directly-mapped
    standard keys (merged with any installed `provider.json` keys).
    """
    if not engine_id:
        return ()
    cleaned = str(engine_id).strip()
    if not cleaned or cleaned in ("stock", "splash"):
        return ()
    if cleaned in M5RT_OPTION_KEYS:
        merged = list(M5RT_SUPPORTED_STANDARD_SETTINGS)
        manifest = _builtin_m5rt_manifest()
        for item in manifest.get("supported_standard_settings") or ():
            if isinstance(item, str) and item.strip() and item.strip() not in merged:
                merged.append(item.strip())
        return tuple(merged)
    plugin = get_external_engine_plugin(cleaned)
    if plugin is not None:
        return tuple(plugin.supported_standard_settings)
    manifest = _spi_provider_manifest_for_engine(cleaned)
    if isinstance(manifest, dict):
        return tuple(
            str(s).strip()
            for s in (manifest.get("supported_standard_settings") or ())
            if isinstance(s, str) and s.strip()
        )
    return ()


def get_external_engine_options_schema(
    engine_id: str | None,
) -> tuple[dict[str, Any], ...]:
    """Return `options_schema` entries exposed by `engine_id`."""
    if not engine_id:
        return ()
    cleaned = str(engine_id).strip()
    if not cleaned or cleaned in ("stock", "splash"):
        return ()
    if cleaned in M5RT_OPTION_KEYS:
        manifest = _builtin_m5rt_manifest()
        raw_schema = manifest.get("options_schema")
        if isinstance(raw_schema, list) and raw_schema:
            by_key: dict[str, dict[str, Any]] = {}
            ordered: list[dict[str, Any]] = []
            for entry in raw_schema:
                if isinstance(entry, dict) and entry.get("key"):
                    copied = dict(entry)
                    by_key[str(copied["key"])] = copied
                    ordered.append(copied)
            for fallback_entry in M5RT_DEFAULT_OPTIONS_SCHEMA:
                fkey = str(fallback_entry["key"])
                if fkey not in by_key:
                    ordered.append(dict(fallback_entry))
            return tuple(ordered)
        return tuple(dict(x) for x in M5RT_DEFAULT_OPTIONS_SCHEMA)
    plugin = get_external_engine_plugin(cleaned)
    if plugin is not None:
        return tuple(dict(x) for x in plugin.options_schema)
    manifest = _spi_provider_manifest_for_engine(cleaned)
    if isinstance(manifest, dict):
        return tuple(
            dict(s) for s in (manifest.get("options_schema") or ()) if isinstance(s, dict)
        )
    return ()


def is_external_process_engine(engine: object | None) -> bool:
    """Whether `engine` manages model weights out-of-process (Splash, M5, plugins).

    Reads `type(engine).external_engine_id` (not instance `getattr`) so `MagicMock`
    test doubles for in-process engines never manufacture a truthy attribute.
    """
    if engine is None:
        return False
    engine_id = getattr(type(engine), "external_engine_id", None)
    return isinstance(engine_id, str) and bool(engine_id)


def mark_external_engine_instance(engine: BaseEngine, engine_id: str) -> BaseEngine:
    """Ensure a plugin-created engine class carries a string `external_engine_id`."""
    cls = type(engine)
    if not isinstance(getattr(cls, "external_engine_id", None), str):
        with contextlib.suppress(TypeError):
            setattr(cls, "external_engine_id", engine_id)
    return engine


# -- M5 binary & artifact resolution ------------------------------------------


@dataclass(frozen=True)
class M5Binary:
    """Resolved Provider SPI sidecar executable (`mseries_backendd` or Path B daemon)."""

    path: Path
    version: str
    protocol_version: int
    auto_detected: bool
    provider_manifest: dict[str, Any] | None = None

    @property
    def engine_id(self) -> str:
        if self.provider_manifest:
            eid = self.provider_manifest.get("engine_id") or self.provider_manifest.get("provider_id")
            if isinstance(eid, str) and eid.strip():
                return eid.strip()
        return M5RT_ENGINE_ID

    @property
    def is_builtin_m5rt(self) -> bool:
        return self.provider_id in M5RT_BUILTIN_PROVIDER_IDS

    @property
    def label(self) -> str:
        if self.provider_manifest and not self.is_builtin_m5rt:
            lbl = self.provider_manifest.get("display_name") or self.provider_manifest.get("label")
            if isinstance(lbl, str) and lbl.strip():
                return lbl.strip()
            return self.engine_id
        return M5RT_ENGINE_LABEL

    @property
    def provider_id(self) -> str:
        if self.provider_manifest and self.provider_manifest.get("provider_id"):
            return str(self.provider_manifest["provider_id"])
        return M5RT_PROVIDER_ID

    @property
    def codename(self) -> str | None:
        if self.provider_manifest and self.provider_manifest.get("codename"):
            return str(self.provider_manifest["codename"])
        return None


@dataclass(frozen=True)
class M5Target:
    """Resolved M5 sidecar runtime target for a model."""

    binary: M5Binary
    artifact_path: Path | None
    metallib_dir: Path | None
    draft_dir: Path | None = None
    draft_artifact_path: Path | None = None
    fake: bool = False
    auto_convert_output: Path | None = None
    auto_convert_draft_output: Path | None = None


@dataclass(frozen=True)
class M5Options:
    """Resolved load plan used for admission, reloads and the sidecar RPC."""

    max_context: int = 32768
    max_concurrency: int = 1
    kv_dtype: str = "bf16"
    prefill_engine: str = "auto"
    prefill_linear_mode: str = "bf16"
    prefill_attention_mode: str = "exact_nax"
    prefill_sparse_minference: bool = False
    prefill_cache_blend: bool = False
    prefill_ane_offload: bool = False
    sampling_contract: str = M5RT_DEFAULT_SAMPLING_CONTRACT
    speculative_backend: str = "none"
    speculative_policy: str = "auto"
    spec_search: str | None = None
    draft_tokens: int = 0
    concurrency_source: str = "default"
    prefix_cache_bytes: int = 0
    rolling_prefix_cache: bool | None = None
    batch_mode: bool | None = None
    batch_projection_prototype: bool = False
    path_overrides: dict[str, str] = field(default_factory=dict)
    fake_options: dict[str, Any] = field(default_factory=dict)
    memory_budget_bytes: int = 0
    trust_remote_code: bool = False
    lora_adapter: str = ""
    lora_mode: str = "merge"

    def to_rpc_dict(self) -> dict[str, Any]:
        """Serialize resolved M5 options for the sidecar `load` RPC payload."""
        payload: dict[str, Any] = {
            "max_concurrency": self.max_concurrency,
            "max_context": self.max_context,
            "kv_dtype": self.kv_dtype,
            "prefill_engine": self.prefill_engine,
            "prefill_linear_mode": self.prefill_linear_mode,
            "prefill_attention_mode": self.prefill_attention_mode,
            "prefill_sparse_minference": bool(self.prefill_sparse_minference),
            "prefill_cache_blend": bool(self.prefill_cache_blend),
            "prefill_ane_offload": bool(self.prefill_ane_offload),
            "sampling_contract": self.sampling_contract,
            "lora_adapter": self.lora_adapter,
            "lora_mode": self.lora_mode,
            "speculative": {
                "backend": self.speculative_backend,
                "draft_tokens": self.draft_tokens,
            },
        }
        if self.spec_search is not None:
            payload["spec_search"] = self.spec_search
        if self.memory_budget_bytes > 0:
            payload["memory_budget_bytes"] = int(self.memory_budget_bytes)
        if self.rolling_prefix_cache is not None:
            payload["rolling_prefix_cache"] = "on" if self.rolling_prefix_cache else "off"
        if self.path_overrides:
            payload["path_overrides"] = dict(self.path_overrides)
        return payload


def _executable(path: Path) -> bool:
    try:
        return path.is_file() and os.access(path, os.X_OK)
    except OSError:
        return False


def _clean_sidecar_env() -> dict[str, str]:
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONHOME", "PYTHONPATH", "PYTHONDONTWRITEBYTECODE")
    }
    path = env.get("PATH", "/usr/bin:/bin").split(":")
    env["PATH"] = ":".join([d for d in _BREW_BIN_DIRS if d not in path] + path)
    return env


_binary_versions: dict[
    tuple[str, int, bool, int | None], tuple[tuple[str, int] | None, float]
] = {}


def _probe_m5rt_version(
    path: Path,
    *,
    allow_generic: bool = False,
    fallback_protocol: int | None = None,
) -> tuple[str, int] | None:
    try:
        stat_key = (str(path), path.stat().st_mtime_ns, allow_generic, fallback_protocol)
    except OSError:
        return None
    now = time.monotonic()
    cached = _binary_versions.get(stat_key)
    if cached is not None and (
        cached[0] is not None or now - cached[1] < VERSION_RETRY_SECONDS
    ):
        return cached[0]

    parsed: tuple[str, int] | None = None
    try:
        proc = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=VERSION_TIMEOUT_SECONDS,
            env=_clean_sidecar_env(),
        )
        output = (proc.stdout + proc.stderr).strip()
        if proc.returncode == 0 and output:
            ver_match = re.search(r"mseries[-_]backendd\s+(\S+)", output)
            proto_match = re.search(
                r"protocol(?:_version)?[=:\s]+(\d+)", output, re.IGNORECASE
            )
            if proto_match is not None:
                proto = int(proto_match.group(1))
                if ver_match is not None:
                    version = ver_match.group(1)
                elif allow_generic:
                    gen_ver = re.search(
                        r"(?:^|\s)([A-Za-z0-9._-]*v?\d+\.\d+(?:\.\d+)*(?:-[0-9A-Za-z._-]+)?)(?:\s|$)",
                        output,
                    )
                    version = gen_ver.group(1) if gen_ver else "0.1.0"
                else:
                    version = "0.1.0"
                parsed = (version, proto)
            elif allow_generic and fallback_protocol is not None:
                gen_ver = re.search(
                    r"(?:^|\s)([A-Za-z0-9._-]*v?\d+\.\d+(?:\.\d+)*(?:-[0-9A-Za-z._-]+)?)(?:\s|$)",
                    output,
                )
                first_line = output.splitlines()[0].strip()
                tokens = first_line.split()
                if gen_ver is not None:
                    version = gen_ver.group(1)
                elif len(tokens) >= 2:
                    version = tokens[1]
                else:
                    version = tokens[0] if tokens else "0.1.0"
                parsed = (version, int(fallback_protocol))
    except (OSError, subprocess.SubprocessError) as exc:
        logger.warning("Sidecar binary %s --version failed: %s", path, exc)

    _binary_versions[stat_key] = (parsed, now)
    return parsed


def _providers_base_dir() -> Path:
    env_dir = os.environ.get("OMLX_PROVIDERS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return Path.home() / "Library" / "Application Support" / "oMLX" / "Providers"


def _bundled_providers_dirs() -> tuple[Path, ...]:
    """Locate offline-bundle `Providers/` directories (cwd, venv parent, repo/bundle root)."""
    raw_candidates: list[Path] = [
        Path.cwd() / "Providers",
        Path(sys.prefix).resolve().parent / "Providers",
        _REPO_ROOT / "Providers",
    ]
    seen: set[Path] = set()
    out: list[Path] = []
    for cand in raw_candidates:
        if not cand.is_dir():
            continue
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            out.append(cand)
    return tuple(out)


def _provider_search_bases() -> tuple[Path, ...]:
    """Return ordered `Providers/` roots: explicit env override, else user dir + bundled dirs."""
    env_dir = os.environ.get("OMLX_PROVIDERS_DIR", "").strip()
    if env_dir:
        return (Path(env_dir).expanduser(),)
    primary = _providers_base_dir()
    bases: list[Path] = [primary]
    seen: set[Path] = set()
    with contextlib.suppress(OSError):
        seen.add(primary.resolve())
    for cand in _bundled_providers_dirs():
        try:
            resolved = cand.resolve()
        except OSError:
            continue
        if resolved not in seen:
            seen.add(resolved)
            bases.append(cand)
    return tuple(bases)


def _read_provider_manifest(provider_dir: Path) -> dict[str, Any] | None:
    manifest_path = provider_dir / "provider.json"
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    if manifest is None:
        return None
    manifest = dict(manifest)
    manifest["_provider_dir"] = str(provider_dir.resolve())
    return manifest


def _provider_root_dirs() -> tuple[Path, ...]:
    roots: list[Path] = []
    for base in _provider_search_bases():
        roots.extend(
            (
                base / M5RT_PROVIDER_ID / "current",
                base / M5RT_PROVIDER_ID,
                base / M5RT_LEGACY_PROVIDER_ID / "current",
                base / M5RT_LEGACY_PROVIDER_ID,
            )
        )
    return tuple(roots)


def _discover_provider_candidates() -> list[tuple[Path, dict[str, Any] | None]]:
    """Return `(binary_path, provider_manifest)` pairs from installed provider layouts."""
    candidates: list[tuple[Path, dict[str, Any] | None]] = []
    seen: set[Path] = set()
    for root in _provider_root_dirs():
        if not root.is_dir():
            continue
        manifest = _read_provider_manifest(root)
        rel_bins: list[str] = []
        if manifest:
            for key in ("entrypoint", "binary", "daemon_path"):
                val = manifest.get(key)
                if isinstance(val, str) and val.strip():
                    rel_bins.append(val.strip())
        rel_bins.extend(("bin/mseries_backendd", "bin/mseries-backendd", "mseries_backendd"))
        for rel in rel_bins:
            cand = Path(rel)
            if not cand.is_absolute():
                cand = root / cand
            if _executable(cand):
                resolved = cand.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    candidates.append((resolved, manifest))
                break
    return candidates


def _iter_repo_build_daemons() -> list[Path]:
    """Enumerate repo build daemon candidates, preferring `miv/` then `m5rt/`."""
    seen: set[Path] = set()
    ordered: list[Path] = []
    for rel in _DEFAULT_DAEMON_RELPATHS:
        cand = _REPO_ROOT / rel
        if _executable(cand):
            resolved = cand.resolve()
            if resolved not in seen:
                seen.add(resolved)
                ordered.append(resolved)

    build_root = _REPO_ROOT / "build"
    if build_root.is_dir():
        for subdir in ("miv", "m5rt"):
            matches: list[tuple[int, Path]] = []
            for cand in build_root.glob(f"*/{subdir}/mseries_backendd"):
                if not _executable(cand):
                    continue
                resolved = cand.resolve()
                if resolved in seen:
                    continue
                try:
                    mtime = cand.stat().st_mtime_ns
                except OSError:
                    mtime = 0
                matches.append((mtime, resolved))
            matches.sort(key=lambda item: item[0], reverse=True)
            for _, resolved in matches:
                seen.add(resolved)
                ordered.append(resolved)
    return ordered


def inspect_m5rt_binary(
    path: Path,
    *,
    auto_detected: bool,
    provider_manifest: dict[str, Any] | None = None,
    allow_generic: bool = False,
    fallback_protocol: int | None = None,
) -> tuple[M5Binary | None, str]:
    """Validate that `path` is a working Provider SPI protocol=1/2 daemon."""
    if not _executable(path):
        return None, f"{path} is not an executable file"
    if provider_manifest is None:
        for parent in (path.resolve().parent, path.resolve().parent.parent):
            manifest = _read_provider_manifest(parent)
            if manifest is not None:
                provider_manifest = manifest
                break
    manifest_proto = (
        provider_manifest.get("protocol_version")
        if isinstance(provider_manifest, dict)
        else None
    )
    eff_fallback = (
        fallback_protocol
        if fallback_protocol is not None
        else (
            manifest_proto
            if isinstance(manifest_proto, int) and not isinstance(manifest_proto, bool)
            else None
        )
    )
    is_foreign = bool(
        provider_manifest
        and str(provider_manifest.get("provider_id") or "") not in ("", *M5RT_BUILTIN_PROVIDER_IDS)
    )
    probed = _probe_m5rt_version(
        path,
        allow_generic=allow_generic or is_foreign,
        fallback_protocol=eff_fallback if (allow_generic or is_foreign) else None,
    )
    if probed is None:
        return None, f"{path} did not report a valid sidecar daemon version"
    version, proto = probed
    if proto not in SUPPORTED_PROTOCOLS:
        return (
            None,
            f"{path} speaks protocol={proto}; oMLX expects protocol in {sorted(SUPPORTED_PROTOCOLS)}",
        )
    return (
        M5Binary(
            path=path.resolve(),
            version=version,
            protocol_version=proto,
            auto_detected=auto_detected,
            provider_manifest=provider_manifest,
        ),
        "",
    )


def _find_in_path(names: tuple[str, ...]) -> Path | None:
    for name in names:
        found = shutil.which(name)
        if found:
            return Path(found)
        for directory in (*_BREW_BIN_DIRS, str(_USER_BIN_DIR), str(_PROVIDER_BIN_DIR)):
            candidate = Path(directory) / name
            if _executable(candidate):
                return candidate
    return None


def has_m5rt_binary_candidate(configured: str | None = None) -> bool:
    """Fast O(1) check whether an `mseries_backendd` binary exists (zero subprocess)."""
    if configured and str(configured).strip():
        return _executable(Path(str(configured).strip()).expanduser())
    for env_key in ("NINFER_METAL_BIN", "OMLX_M5RT_BIN"):
        env_bin = os.environ.get(env_key, "").strip()
        if env_bin:
            return _executable(Path(env_bin).expanduser())
    if _discover_provider_candidates():
        return True
    if _find_in_path(("mseries_backendd", "mseries-backendd")) is not None:
        return True
    return bool(_iter_repo_build_daemons())


def find_m5rt_binary(configured: str | None = None) -> tuple[M5Binary | None, str]:
    """Resolve `mseries_backendd` via settings, env, Provider manifests, PATH, or build dirs."""
    if configured and str(configured).strip():
        return inspect_m5rt_binary(
            Path(str(configured).strip()).expanduser(), auto_detected=False
        )

    for env_key in ("NINFER_METAL_BIN", "OMLX_M5RT_BIN"):
        env_bin = os.environ.get(env_key, "").strip()
        if env_bin:
            return inspect_m5rt_binary(Path(env_bin).expanduser(), auto_detected=True)

    reason = M5RT_INSTALL_HINT

    # 1) Installed oMLX Providers (`io.ninfer.metal/current/` then legacy `com.omlx.mseries/current/`).
    for cand_path, manifest in _discover_provider_candidates():
        binary, candidate_reason = inspect_m5rt_binary(
            cand_path, auto_detected=True, provider_manifest=manifest
        )
        if binary is not None:
            return binary, ""
        reason = candidate_reason

    # 2) PATH / Homebrew / ~/.local/bin
    path_candidate = _find_in_path(("mseries_backendd", "mseries-backendd"))
    if path_candidate is not None:
        binary, candidate_reason = inspect_m5rt_binary(path_candidate, auto_detected=True)
        if binary is not None:
            return binary, ""
        reason = candidate_reason

    # 3) Repo build dirs (`build/*/miv/` then `build/*/m5rt/`)
    for candidate in _iter_repo_build_daemons():
        binary, candidate_reason = inspect_m5rt_binary(candidate, auto_detected=True)
        if binary is not None:
            return binary, ""
        reason = candidate_reason

    return None, reason


def _normalize_extensions(raw: Any) -> tuple[str, ...]:
    if not isinstance(raw, (list, tuple, set, frozenset)) or not raw:
        return M5RT_ARTIFACT_EXTENSIONS
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            continue
        ext = item.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in out:
            out.append(ext)
    return tuple(out) if out else M5RT_ARTIFACT_EXTENSIONS


def validate_provider_artifact_file(
    path_value: str | Path,
    *,
    allowed_extensions: Iterable[str] = M5RT_ARTIFACT_EXTENSIONS,
    field_name: str = "artifact_path",
) -> Path:
    """Validate that an artifact path points to an existing regular file with an allowed extension."""
    exts = _normalize_extensions(list(allowed_extensions))
    ext_desc = "/".join(exts)
    expanded = Path(path_value).expanduser()
    if expanded.is_dir():
        raise ValueError(f"{field_name} must be a {ext_desc} file, not a directory: {expanded}")
    if not expanded.is_file():
        raise ValueError(f"{field_name} does not exist or is not a file: {expanded}")
    if expanded.suffix.lower() not in exts:
        raise ValueError(f"{field_name} must point to a {ext_desc} file: {expanded}")
    return expanded.resolve()


def validate_m5rt_artifact_file(path_value: str | Path, field_name: str = "m5rt_artifact_path") -> Path:
    """Validate that an `.omlxm5` path points to an existing regular file."""
    return validate_provider_artifact_file(
        path_value, allowed_extensions=M5RT_ARTIFACT_EXTENSIONS, field_name=field_name
    )


def _parse_optional_toggle(value: Any, field_name: str) -> bool | None:
    """Parse an optional boolean or 'on'/'off'/'1'/'0'/'true'/'false' toggle."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in ("on", "1", "true"):
            return True
        if norm in ("off", "0", "false"):
            return False
    raise ValueError(
        f"{field_name} must be a boolean or one of 'on', 'off', '1', '0', 'true', 'false'"
    )


def validate_m5rt_options(options: dict[str, Any] | None) -> dict[str, Any] | None:
    """Validate optional per-model `m5rt_options` / `external_engine_options` dictionary."""
    if options is None:
        return None
    if not isinstance(options, dict):
        raise ValueError("m5rt_options must be an object")
    options = _normalize_m5_aliases(options)

    kv_dtype = options.get("kv_dtype")
    if kv_dtype is not None and kv_dtype not in VALID_KV_DTYPES:
        raise ValueError(f"m5rt_options.kv_dtype must be one of {sorted(VALID_KV_DTYPES)}")

    prefill = options.get("prefill_engine")
    if prefill is not None and prefill not in VALID_PREFILL_ENGINES:
        raise ValueError(
            f"m5rt_options.prefill_engine must be one of {sorted(VALID_PREFILL_ENGINES)}"
        )

    contract = options.get("sampling_contract")
    if contract is not None and contract not in VALID_SAMPLING_CONTRACTS:
        raise ValueError(
            "m5rt_options.sampling_contract must be one of "
            f"{sorted(VALID_SAMPLING_CONTRACTS)}"
        )

    spec_backend = options.get("speculative_backend")
    if spec_backend is not None and spec_backend not in VALID_SPEC_BACKENDS:
        raise ValueError(
            f"m5rt_options.speculative_backend must be one of {sorted(VALID_SPEC_BACKENDS)}"
        )

    max_concurrency = options.get("max_concurrency")
    if max_concurrency is not None:
        if isinstance(max_concurrency, bool) or not isinstance(max_concurrency, int) or not 1 <= max_concurrency <= 8:
            raise ValueError("m5rt_options.max_concurrency must be an integer in [1, 8]")

    draft_tokens = options.get("draft_tokens")
    if draft_tokens is not None:
        maximum = M5RT_MAX_DRAFT_TOKENS.get(spec_backend, M5RT_MAX_DRAFT_TOKENS["dflash2"])
        if isinstance(draft_tokens, bool) or not isinstance(draft_tokens, int) or not 0 <= draft_tokens <= maximum:
            raise ValueError(f"m5rt_options.draft_tokens must be an integer in [0, {maximum}]")

    prefix_cache_bytes = options.get("prefix_cache_bytes")
    if prefix_cache_bytes is not None:
        if isinstance(prefix_cache_bytes, bool) or not isinstance(prefix_cache_bytes, int) or prefix_cache_bytes < 0:
            raise ValueError("m5rt_options.prefix_cache_bytes must be a non-negative integer")

    prefix_cache_gb = options.get("prefix_cache_gb")
    if prefix_cache_gb is not None:
        if (
            isinstance(prefix_cache_gb, bool)
            or not isinstance(prefix_cache_gb, (int, float))
            or prefix_cache_gb < 0
        ):
            raise ValueError("m5rt_options.prefix_cache_gb must be a non-negative number")

    if "rolling_prefix_cache" in options:
        _parse_optional_toggle(
            options.get("rolling_prefix_cache"), "m5rt_options.rolling_prefix_cache"
        )

    if options.get("spec_search") not in (None, "auto", "off", "on"):
        raise ValueError("m5rt_options.spec_search must be auto, off, or on")

    if "batch_mode" in options:
        _parse_optional_toggle(
            options.get("batch_mode"), "m5rt_options.batch_mode"
        )

    for file_key in ("artifact_path", "draft_artifact_path"):
        if options.get(file_key):
            validate_m5rt_artifact_file(options[file_key], f"m5rt_options.{file_key}")

    metallib_dir = options.get("metallib_dir")
    if metallib_dir is not None and str(metallib_dir).strip():
        mdir = Path(str(metallib_dir)).expanduser()
        if not mdir.is_dir():
            raise ValueError(f"m5rt_options.metallib_dir is not a directory: {mdir}")

    cleaned = dict(options)
    if "lora_mode" in options and options.get("lora_mode") is not None:
        raw_mode = options.get("lora_mode")
        if not isinstance(raw_mode, str) or raw_mode.strip().lower() not in VALID_LORA_MODES:
            raise ValueError(
                f"m5rt_options.lora_mode must be one of {sorted(VALID_LORA_MODES)}"
            )
        cleaned["lora_mode"] = raw_mode.strip().lower()

    if "lora_adapter" in options and options.get("lora_adapter") is not None:
        raw_lora = options.get("lora_adapter")
        if not isinstance(raw_lora, str):
            raise ValueError("m5rt_options.lora_adapter must be a string")
        cleaned["lora_adapter"] = raw_lora.strip()

    if "prefill_linear_mode" in options and options.get("prefill_linear_mode") is not None:
        raw_plm = options.get("prefill_linear_mode")
        if not isinstance(raw_plm, str) or raw_plm.strip().lower() not in VALID_PREFILL_LINEAR_MODES:
            raise ValueError(
                f"m5rt_options.prefill_linear_mode must be one of {sorted(VALID_PREFILL_LINEAR_MODES)}"
            )
        cleaned["prefill_linear_mode"] = raw_plm.strip().lower()

    if "prefill_attention_mode" in options and options.get("prefill_attention_mode") is not None:
        raw_pam = options.get("prefill_attention_mode")
        if not isinstance(raw_pam, str) or raw_pam.strip().lower() not in VALID_PREFILL_ATTENTION_MODES:
            raise ValueError(
                f"m5rt_options.prefill_attention_mode must be one of {sorted(VALID_PREFILL_ATTENTION_MODES)}"
            )
        cleaned["prefill_attention_mode"] = raw_pam.strip().lower()

    for bool_key in ("prefill_sparse_minference", "prefill_cache_blend", "prefill_ane_offload"):
        if bool_key in options and options.get(bool_key) is not None:
            parsed = _parse_optional_toggle(options.get(bool_key), f"m5rt_options.{bool_key}")
            if parsed is not None:
                cleaned[bool_key] = parsed

    return cleaned


def sync_m5rt_ui_kv_setting(
    current_settings: Any,
    request: Any,
    sent: set[str] | frozenset[str],
) -> None:
    """Persist an explicit UI KV-compression toggle for `m5rt` so disabling overrides shared TOML."""
    target_engine = (
        getattr(request, "external_engine", None)
        if "external_engine" in sent
        else getattr(current_settings, "external_engine", None)
    )
    switched_to_m5rt = (
        "external_engine" in sent
        and canonical_external_engine_id(target_engine) == M5RT_ENGINE_ID
    )
    # Normalize leftover Stock MLX dflash_block_size values (> 8) when switching to
    # or saving on m5rt without an explicit dflash_block_size field in the request.
    if "dflash_block_size" not in sent and (
        switched_to_m5rt or canonical_external_engine_id(target_engine) == M5RT_ENGINE_ID
    ):
        cur_block = getattr(current_settings, "dflash_block_size", None)
        max_block = M5RT_MAX_DRAFT_TOKENS["dflash2"] + 1
        if (
            isinstance(cur_block, int)
            and not isinstance(cur_block, bool)
            and cur_block > max_block
            and hasattr(current_settings, "dflash_block_size")
        ):
            current_settings.dflash_block_size = max_block
    if "turboquant_kv_enabled" not in sent and not switched_to_m5rt:
        return
    req_explicit_kv = False
    if "m5rt_options" in sent and isinstance(getattr(request, "m5rt_options", None), dict):
        if request.m5rt_options.get("kv_dtype") is not None:
            req_explicit_kv = True
    for field_name in ("external_engine_options", "external_engine_options_patch"):
        updates = getattr(request, field_name, None)
        if field_name in sent and isinstance(updates, dict):
            for eng_key in M5RT_OPTION_KEYS:
                nested = updates.get(eng_key)
                if isinstance(nested, dict) and nested.get("kv_dtype") is not None:
                    req_explicit_kv = True
                    break
    if req_explicit_kv:
        return

    kv_enabled = (
        bool(getattr(request, "turboquant_kv_enabled", False))
        if "turboquant_kv_enabled" in sent
        else bool(getattr(current_settings, "turboquant_kv_enabled", False))
    )
    if "turboquant_kv_enabled" in sent and not kv_enabled:
        opts_copy = dict(getattr(current_settings, "m5rt_options", None) or {})
        opts_copy["kv_dtype"] = "bf16"
        current_settings.m5rt_options = opts_copy
        existing_ext = getattr(current_settings, "external_engine_options", None)
        if isinstance(existing_ext, dict):
            ext_copy = dict(existing_ext)
            updated_ext = False
            for eng_key in M5RT_OPTION_KEYS:
                nested = ext_copy.get(eng_key)
                if isinstance(nested, dict):
                    nested_copy = dict(nested)
                    nested_copy["kv_dtype"] = "bf16"
                    ext_copy[eng_key] = nested_copy
                    updated_ext = True
            if updated_ext:
                current_settings.external_engine_options = ext_copy
    elif kv_enabled:
        if "turboquant_kv_enabled" in sent:
            existing_opts = getattr(current_settings, "m5rt_options", None)
            if isinstance(existing_opts, dict) and existing_opts.get("kv_dtype") == "bf16":
                opts_copy = dict(existing_opts)
                opts_copy.pop("kv_dtype", None)
                current_settings.m5rt_options = opts_copy or None
            existing_ext = getattr(current_settings, "external_engine_options", None)
            if isinstance(existing_ext, dict):
                ext_copy = dict(existing_ext)
                updated_ext = False
                for eng_key in M5RT_OPTION_KEYS:
                    nested = ext_copy.get(eng_key)
                    if isinstance(nested, dict) and nested.get("kv_dtype") == "bf16":
                        nested_copy = dict(nested)
                        nested_copy.pop("kv_dtype", None)
                        if nested_copy:
                            ext_copy[eng_key] = nested_copy
                        else:
                            ext_copy.pop(eng_key, None)
                        updated_ext = True
                if updated_ext:
                    current_settings.external_engine_options = ext_copy or None
        # Web hides scheme/bitwidth pickers for miv; normalize leftover Stock
        # MLX values (e.g. affine8 or 6/8-bit) when the user did not explicitly
        # send them so turning on 4-bit KV is never silently trapped in bf16.
        if "turboquant_kv_scheme" not in sent and "turboquant_kv_bits" not in sent:
            scheme = str(getattr(current_settings, "turboquant_kv_scheme", "") or "").strip().lower()
            raw_bits = getattr(current_settings, "turboquant_kv_bits", None)
            unsupported_bits = (
                isinstance(raw_bits, (int, float))
                and not isinstance(raw_bits, bool)
                and float(raw_bits) > 4.0
            )
            if scheme not in ("", "affine4", "turboquant") or unsupported_bits:
                if hasattr(current_settings, "turboquant_kv_scheme"):
                    current_settings.turboquant_kv_scheme = "affine4"
                if hasattr(current_settings, "turboquant_kv_bits"):
                    current_settings.turboquant_kv_bits = 4.0


def _validate_external_engine_option_fields(engine_key: str, fields: Any) -> dict[str, Any]:
    if not isinstance(fields, dict):
        raise ValueError(f"external_engine_options.{engine_key} must be an object")
    if engine_key in M5RT_OPTION_KEYS:
        return validate_m5rt_options(fields) or {}
    cleaned = dict(fields)
    for item in get_external_engine_options_schema(engine_key):
        key = item.get("key")
        value = cleaned.get(key)
        if value is None or item.get("type") != "enum" or not isinstance(item.get("options"), list):
            continue
        allowed = [str(option) for option in item["options"]]
        if not isinstance(value, str) or value.strip() not in allowed:
            raise ValueError(
                f"external_engine_options.{engine_key}.{key} must be one of {sorted(allowed)}"
            )
        cleaned[key] = value.strip()
    return cleaned


def apply_m5rt_model_settings_update(
    current_settings: Any,
    request: Any,
    sent: set[str] | frozenset[str],
    *,
    is_diffusion_model: bool,
    external_engine: str | None = None,
) -> None:
    """Validate and apply `m5rt_*` and `external_engine_options` fields from admin requests."""
    if "m5rt_artifact_path" in sent:
        art_path = (
            None
            if is_diffusion_model
            else (request.m5rt_artifact_path or "").strip() or None
        )
        if art_path and art_path != current_settings.m5rt_artifact_path:
            art_path = str(validate_m5rt_artifact_file(art_path, "m5rt_artifact_path"))
        current_settings.m5rt_artifact_path = art_path
    if "m5rt_draft_artifact_path" in sent:
        draft_path = (
            None
            if is_diffusion_model
            else (request.m5rt_draft_artifact_path or "").strip() or None
        )
        if draft_path and draft_path != current_settings.m5rt_draft_artifact_path:
            draft_path = str(
                validate_m5rt_artifact_file(draft_path, "m5rt_draft_artifact_path")
            )
        current_settings.m5rt_draft_artifact_path = draft_path
    if "m5rt_options" in sent:
        current_settings.m5rt_options = (
            None
            if is_diffusion_model or not request.m5rt_options
            else validate_m5rt_options(request.m5rt_options)
        )
    if "external_engine_options" in sent and "external_engine_options_patch" in sent:
        raise ValueError("Send either external_engine_options or external_engine_options_patch")
    if "external_engine_options" in sent or "external_engine_options_patch" in sent:
        partial = "external_engine_options_patch" in sent
        field_name = "external_engine_options_patch" if partial else "external_engine_options"
        updates = getattr(request, field_name, None) or {}
        merged = {
            key: _normalize_m5_aliases(value) if key in M5RT_OPTION_KEYS else dict(value)
            for key, value in (current_settings.external_engine_options or {}).items()
            if isinstance(value, dict)
        } if partial and not is_diffusion_model else {}
        legacy = _normalize_m5_aliases(getattr(current_settings, "m5rt_options", None) or {})
        if not isinstance(updates, dict):
            raise ValueError(f"{field_name} must be an object")
        if not is_diffusion_model:
            for engine_key, fields in updates.items():
                cleaned = _validate_external_engine_option_fields(str(engine_key), fields)
                if not partial:
                    merged[engine_key] = cleaned
                for key, value in cleaned.items():
                    if engine_key in M5RT_OPTION_KEYS and "m5rt_options" not in sent:
                        if value is None and partial:
                            legacy.pop(key, None)
                        elif key in legacy:
                            legacy[key] = value
                    if partial and value is None:
                        aliases = M5RT_OPTION_KEYS if engine_key in M5RT_OPTION_KEYS else (engine_key,)
                        for alias in aliases:
                            if alias in merged:
                                merged[alias].pop(key, None)
                    elif partial:
                        merged.setdefault(engine_key, {})[key] = value
        current_settings.external_engine_options = {k: v for k, v in merged.items() if v} or None
        if "m5rt_options" not in sent and not is_diffusion_model:
            current_settings.m5rt_options = legacy or None

    eff_ext = external_engine
    if eff_ext is None:
        if "external_engine" in sent:
            eff_ext = None if is_diffusion_model else getattr(request, "external_engine", None) or None
        else:
            eff_ext = getattr(current_settings, "external_engine", None)
    eff_ext = canonical_external_engine_id(eff_ext)
    if not is_diffusion_model and eff_ext == M5RT_ENGINE_ID:
        sync_m5rt_ui_kv_setting(current_settings, request, sent)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _sha256_file(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _normalize_slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _normalize_m5_aliases(options: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(options)
    # Match the standalone reader's precedence when both spellings are present.
    for legacy, canonical in (
        ("spec_backend", "speculative_backend"),
        ("artifact", "artifact_path"),
        ("draft_artifact", "draft_artifact_path"),
        ("kernel_dir", "metallib_dir"),
    ):
        value = normalized.pop(legacy, None)
        if value is not None and (legacy != "kernel_dir" or canonical not in normalized):
            normalized[canonical] = value
    return normalized


def _extract_toml_layered_options(
    doc: dict[str, Any], *, model_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split `ninfer-metal.toml` into `(shared_opts, per_model_opts)`."""
    shared: dict[str, Any] = {}
    model_sec = doc.get("model")
    if isinstance(model_sec, dict):
        model_sec = _normalize_m5_aliases(model_sec)
        for key in (
            "artifact_path",
            "draft_artifact_path",
            "draft_dir",
            "mtp_artifact",
            "mtp_artifact_path",
            "lora_adapter",
            "lora_mode",
        ):
            if model_sec.get(key) is not None:
                shared[key] = model_sec[key]

    engine_sec = doc.get("engine")
    if isinstance(engine_sec, dict):
        engine_sec = _normalize_m5_aliases(engine_sec)
        for key in (
            "max_concurrency",
            "max_context",
            "kv_dtype",
            "prefill_engine",
            "prefill_linear_mode",
            "prefill_attention_mode",
            "prefill_sparse_minference",
            "prefill_cache_blend",
            "prefill_ane_offload",
            "spec_search",
            "speculative_backend",
            "sampling_contract",
            "rolling_prefix_cache",
            "prefix_cache_bytes",
            "prefix_cache_gb",
            "batch_mode",
            "batch_projection_prototype",
            "lora_adapter",
            "lora_mode",
            "metallib_dir",
            "m5rt_path",
            "path_overrides",
        ):
            if engine_sec.get(key) is not None:
                shared[key] = engine_sec[key]

    spec_sec = doc.get("speculative")
    if isinstance(spec_sec, dict):
        if spec_sec.get("backend") is not None:
            shared["speculative_backend"] = spec_sec["backend"]
        elif spec_sec.get("speculative_backend") is not None:
            shared["speculative_backend"] = spec_sec["speculative_backend"]
        for key in ("draft_tokens", "draft_dir", "draft_artifact_path"):
            if spec_sec.get(key) is not None:
                shared[key] = spec_sec[key]

    per_model: dict[str, Any] = {}
    models_arr = doc.get("models")
    if model_id and isinstance(models_arr, list):
        for item in models_arr:
            if not isinstance(item, dict):
                continue
            item_id = str(item.get("id") or "").strip()
            item_dir = str(item.get("model_dir") or "").strip()
            if item_id == model_id or (item_dir and Path(item_dir).name == model_id):
                for key, val in _normalize_m5_aliases(item).items():
                    if key not in ("id", "model_dir") and val is not None:
                        per_model[key] = val

    sidecar_sec = doc.get("sidecar")
    if isinstance(sidecar_sec, dict):
        for key, val in _normalize_m5_aliases(sidecar_sec).items():
            if key == "models" or val is None:
                continue
            shared[key] = val
        models_sec = sidecar_sec.get("models")
        if model_id and isinstance(models_sec, dict):
            model_entry = models_sec.get(model_id)
            if isinstance(model_entry, dict):
                for key, val in _normalize_m5_aliases(model_entry).items():
                    if val is not None:
                        per_model[key] = val
    return shared, per_model


def _extract_toml_engine_options(
    doc: dict[str, Any], *, model_id: str | None = None
) -> dict[str, Any]:
    """Flatten `ninfer-metal.toml` sections into `m5rt_options`-compatible keys."""
    shared, per_model = _extract_toml_layered_options(doc, model_id=model_id)
    merged = dict(shared)
    merged.update(per_model)
    return merged


def _no_default_toml_requested() -> bool:
    raw = os.environ.get("NINFER_METAL_NO_DEFAULT_CONFIG", "").strip().lower()
    return raw in ("1", "true", "on", "yes")


def discover_ninfer_metal_toml_path() -> Path | None:
    """Discover the first existing shared `ninfer-metal.toml` in exact precedence order.

    Precedence (matches standalone `ninfer-metal-serve`):
      1. `$NINFER_METAL_CONFIG`
      2. `./ninfer-metal.toml`
      3. `~/.config/ninfer-metal/ninfer-metal.toml`
      4. `~/Library/Application Support/oMLX/ninfer-metal.toml` (legacy user path)
      5. `<providers_dir>/io.ninfer.metal/current/config/ninfer-metal.toml`
      6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`
    """
    env_cfg = os.environ.get("NINFER_METAL_CONFIG", "").strip()
    if env_cfg:
        cand = Path(env_cfg).expanduser()
        if cand.is_file():
            return cand
    if _no_default_toml_requested():
        return None

    home = Path.home()
    env_providers = os.environ.get("OMLX_PROVIDERS_DIR", "").strip()
    providers_base = (
        Path(env_providers).expanduser()
        if env_providers
        else (home / "Library" / "Application Support" / "oMLX" / "Providers")
    )

    candidates: list[Path] = [
        Path.cwd() / "ninfer-metal.toml",
        home / ".config" / "ninfer-metal" / "ninfer-metal.toml",
        home / "Library" / "Application Support" / "oMLX" / "ninfer-metal.toml",
        providers_base / M5RT_PROVIDER_ID / "current" / "config" / "ninfer-metal.toml",
    ]
    if not env_providers:
        candidates.append(Path.cwd() / "config" / "ninfer-metal.toml")
        for bundled_providers in _bundled_providers_dirs():
            candidates.append(bundled_providers.parent / "config" / "ninfer-metal.toml")
            candidates.append(
                bundled_providers / M5RT_PROVIDER_ID / "current" / "config" / "ninfer-metal.toml"
            )
    candidates.append(_REPO_ROOT / "omlx-extend" / "config" / "ninfer-metal.toml")
    for cfg_path in candidates:
        if cfg_path.is_file():
            return cfg_path
    return None


def _load_ninfer_metal_toml_layers(
    model_dir: Path | None = None, *, model_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load `(shared_toml_opts, per_model_toml_opts)` from shared `ninfer-metal.toml`."""
    effective_model_id = model_id or (model_dir.name if model_dir is not None else None)
    shared: dict[str, Any] = {}
    per_model: dict[str, Any] = {}

    primary_path = discover_ninfer_metal_toml_path()
    if primary_path is not None:
        try:
            with primary_path.open("rb") as fh:
                parsed = tomllib.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to parse %s: %s", primary_path, exc)
        else:
            if isinstance(parsed, dict):
                s_opts, m_opts = _extract_toml_layered_options(
                    parsed, model_id=effective_model_id
                )
                shared.update(s_opts)
                per_model.update(m_opts)

    env_cfg = os.environ.get("NINFER_METAL_CONFIG", "").strip()
    env_override_active = bool(env_cfg and Path(env_cfg).expanduser().is_file())
    if model_dir is not None and not env_override_active:
        local_cfg = model_dir / "ninfer-metal.toml"
        if local_cfg.is_file() and (
            primary_path is None or local_cfg.resolve() != primary_path.resolve()
        ):
            try:
                with local_cfg.open("rb") as fh:
                    parsed_local = tomllib.load(fh)
            except (OSError, ValueError) as exc:
                logger.warning("Failed to parse %s: %s", local_cfg, exc)
            else:
                if isinstance(parsed_local, dict):
                    per_model.update(
                        _extract_toml_engine_options(
                            parsed_local, model_id=effective_model_id
                        )
                    )
    return shared, per_model


def load_ninfer_metal_toml(
    model_dir: Path | None = None, *, model_id: str | None = None
) -> dict[str, Any]:
    """Load shared `ninfer-metal.toml` config + per-model `[sidecar.models]` overrides."""
    shared, per_model = _load_ninfer_metal_toml_layers(model_dir, model_id=model_id)
    merged = dict(shared)
    merged.update(per_model)
    return merged


def _builtin_m5rt_manifest() -> dict[str, Any]:
    """Return the built-in `io.ninfer.metal` (`m5rt`) routing manifest (O(1), no subprocess)."""
    manifest: dict[str, Any] = {
        "provider_id": M5RT_PROVIDER_ID,
        "legacy_provider_ids": [M5RT_LEGACY_PROVIDER_ID],
        "engine_id": M5RT_ENGINE_ID,
        "display_name": M5RT_ENGINE_LABEL,
        "capabilities": list(M5RT_DEFAULT_CAPABILITIES),
        "supported_topologies": list(M5RT_SUPPORTED_TOPOLOGIES),
        "supported_model_types": list(M5RT_SUPPORTED_MODEL_TYPES),
        "supported_draft_architectures": list(M5RT_SUPPORTED_DFLASH2_ARCHITECTURES),
        "supported_dflash2_models": list(M5RT_SUPPORTED_DFLASH2_REPOS),
        "supported_standard_settings": list(M5RT_SUPPORTED_STANDARD_SETTINGS),
        "options_schema": [dict(s) for s in M5RT_DEFAULT_OPTIONS_SCHEMA],
        "artifact_extensions": list(M5RT_ARTIFACT_EXTENSIONS),
        "converter": dict(M5RT_DEFAULT_CONVERTER),
        "architecture_capabilities": dict(M5RT_DEFAULT_ARCHITECTURE_CAPABILITIES),
    }
    for root in _provider_root_dirs():
        if not root.is_dir():
            continue
        installed = _read_provider_manifest(root)
        if installed is None:
            continue
        for key in ("supported_topologies", "supported_model_types"):
            vals = installed.get(key)
            if isinstance(vals, list) and vals:
                manifest[key] = vals
        if isinstance(installed.get("architecture_capabilities"), dict):
            merged_arch_caps = dict(M5RT_DEFAULT_ARCHITECTURE_CAPABILITIES)
            merged_arch_caps.update(installed["architecture_capabilities"])
            merged_arch_caps["model_type_whitelist_enforced"] = False
            manifest["architecture_capabilities"] = merged_arch_caps
        for union_key, default_vals in (
            ("capabilities", M5RT_DEFAULT_CAPABILITIES),
            ("supported_standard_settings", M5RT_SUPPORTED_STANDARD_SETTINGS),
        ):
            inst_vals = installed.get(union_key)
            if isinstance(inst_vals, list) and inst_vals:
                merged_vals = list(default_vals)
                for item in inst_vals:
                    if isinstance(item, str) and item.strip() and item.strip() not in merged_vals:
                        merged_vals.append(item.strip())
                manifest[union_key] = merged_vals
        inst_schema = installed.get("options_schema")
        if isinstance(inst_schema, list) and inst_schema:
            parsed_schema = [dict(s) for s in inst_schema if isinstance(s, dict)]
            seen_keys = {str(s.get("key")) for s in parsed_schema if s.get("key")}
            for default_item in M5RT_DEFAULT_OPTIONS_SCHEMA:
                if default_item["key"] not in seen_keys:
                    parsed_schema.append(dict(default_item))
            manifest["options_schema"] = parsed_schema
        inst_exts = installed.get("artifact_extensions")
        if isinstance(inst_exts, list) and inst_exts:
            merged_exts = list(M5RT_ARTIFACT_EXTENSIONS)
            for ext in _normalize_extensions(inst_exts):
                if ext not in merged_exts:
                    merged_exts.append(ext)
            manifest["artifact_extensions"] = merged_exts
        if isinstance(installed.get("converter"), dict):
            manifest["converter"] = dict(installed["converter"])
        if installed.get("_provider_dir"):
            manifest["_provider_dir"] = installed["_provider_dir"]
        break
    return manifest


def _provider_option_keys(
    engine_id: str,
    provider_id: str,
    manifest: dict[str, Any] | None = None,
) -> tuple[str, ...]:
    if engine_id == M5RT_ENGINE_ID or provider_id in M5RT_BUILTIN_PROVIDER_IDS:
        return M5RT_OPTION_KEYS
    keys: list[str] = []
    legacy = manifest.get("legacy_provider_ids") if isinstance(manifest, dict) else None
    raw_candidates: list[Any] = [provider_id]
    if isinstance(legacy, (list, tuple)):
        raw_candidates.extend(legacy)
    raw_candidates.append(engine_id)
    for item in raw_candidates:
        if isinstance(item, str) and item.strip() and item.strip() not in keys:
            keys.append(item.strip())
    return tuple(keys)


def _explicit_m5rt_model_options(model_settings: object | None) -> dict[str, Any]:
    """Extract per-model explicit `external_engine_options` < `m5rt_options`."""
    explicit: dict[str, Any] = {}
    ext_opts = getattr(model_settings, "external_engine_options", None)
    if isinstance(ext_opts, dict):
        for key in M5RT_OPTION_KEYS:
            nested = ext_opts.get(key)
            if isinstance(nested, dict):
                explicit.update(_normalize_m5_aliases(nested))
    legacy_opts = getattr(model_settings, "m5rt_options", None)
    if isinstance(legacy_opts, dict):
        explicit.update(_normalize_m5_aliases(legacy_opts))
    return explicit


def _layered_provider_options(
    model_dir: Path | None,
    model_settings: object | None,
    *,
    engine_id: str = M5RT_ENGINE_ID,
    provider_id: str = M5RT_PROVIDER_ID,
    manifest: dict[str, Any] | None = None,
    model_id: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Return `(shared_toml, model_toml, explicit_opts)` in ascending precedence order."""
    if engine_id == M5RT_ENGINE_ID and provider_id in M5RT_BUILTIN_PROVIDER_IDS:
        shared_toml, model_toml = _load_ninfer_metal_toml_layers(
            model_dir, model_id=model_id
        )
        explicit = _explicit_m5rt_model_options(model_settings)
        return shared_toml, model_toml, explicit
    explicit = {}
    ext_opts = getattr(model_settings, "external_engine_options", None)
    if isinstance(ext_opts, dict):
        for key in _provider_option_keys(engine_id, provider_id, manifest):
            nested = ext_opts.get(key)
            if isinstance(nested, dict):
                explicit.update(nested)
    return {}, {}, explicit


def _merged_m5rt_options(
    model_dir: Path | None,
    model_settings: object | None,
    *,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Merge `ninfer-metal.toml` < `[sidecar.models]` < `external_engine_options` < `m5rt_options`."""
    shared_toml, model_toml, explicit = _layered_provider_options(
        model_dir, model_settings, model_id=model_id
    )
    merged = dict(shared_toml)
    merged.update(model_toml)
    merged.update(explicit)
    return merged


def _merged_provider_options(
    model_dir: Path | None,
    model_settings: object | None,
    *,
    engine_id: str = M5RT_ENGINE_ID,
    provider_id: str = M5RT_PROVIDER_ID,
    manifest: dict[str, Any] | None = None,
    model_id: str | None = None,
) -> dict[str, Any]:
    """Merge per-provider options (`external_engine_options[provider_id|engine_id]`)."""
    shared_toml, model_toml, explicit = _layered_provider_options(
        model_dir,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id,
        manifest=manifest,
        model_id=model_id,
    )
    merged = dict(shared_toml)
    merged.update(model_toml)
    merged.update(explicit)
    return merged


def _extract_model_type_tokens(
    config: dict[str, Any] | None, pack_meta: dict[str, Any] | None = None
) -> set[str]:
    tokens: set[str] = set()
    sources: list[Any] = []
    if isinstance(pack_meta, dict):
        sources.append(pack_meta.get("model_type"))
    if isinstance(config, dict):
        sources.append(config.get("model_type"))
        text_cfg = _text_config(config)
        if text_cfg is not config:
            sources.append(text_cfg.get("model_type"))
    for raw in sources:
        if isinstance(raw, str) and raw.strip():
            norm = raw.strip().lower()
            tokens.add(norm)
            if norm.endswith("_text") and len(norm) > 5:
                tokens.add(norm[:-5])
    return tokens


def _has_miv_container_magic(path: Path | None) -> bool:
    """Fast 8-byte check whether `path` begins with `MIV00001` or `OMLXM5A1`."""
    if path is None:
        return False
    try:
        with path.open("rb") as fh:
            magic = fh.read(8)
        return len(magic) == 8 and magic in _MIV_VALID_MAGICS
    except OSError:
        return False


def _foreign_spi_claims_model_artifact(model_types: set[str], ext: str) -> bool:
    """Return True when an installed Path B SPI provider explicitly claims `(model_types, ext)`."""
    if not model_types:
        return False
    norm_ext = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    for base in _provider_search_bases():
        if not base.is_dir():
            continue
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in children:
            if not entry.is_dir() or entry.name in M5RT_BUILTIN_PROVIDER_IDS:
                continue
            for cand_root in (entry / "current", entry):
                if not cand_root.is_dir():
                    continue
                parsed = _read_provider_manifest(cand_root)
                if parsed is None:
                    continue
                pid = str(parsed.get("provider_id") or entry.name).strip()
                if not pid or pid in M5RT_BUILTIN_PROVIDER_IDS:
                    break
                p_exts = _normalize_extensions(parsed.get("artifact_extensions"))
                p_types = {
                    str(m).strip().lower()
                    for m in (parsed.get("supported_model_types") or ())
                    if isinstance(m, str) and m.strip()
                }
                if norm_ext in p_exts and (model_types & p_types):
                    return True
                break
    return False


def _config_structural_blocker(config: dict[str, Any] | None) -> str:
    """Fast config-only check for incompatible architectures on adapted/candidate directories."""
    if not isinstance(config, dict):
        return ""
    text_cfg = _text_config(config)
    for mla_key in (
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    ):
        if text_cfg.get(mla_key) is not None or config.get(mla_key) is not None:
            return (
                f"MLA (Multi-head Latent Attention) architecture is not supported by "
                f"{M5RT_ENGINE_LABEL} (found config key {mla_key!r})"
            )
    for ssm_key in ("mamba_d_state", "ssm_state_size", "mamba_d_conv"):
        val = text_cfg.get(ssm_key) if text_cfg.get(ssm_key) is not None else config.get(ssm_key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            return (
                f"Mamba/SSM architecture is not supported by {M5RT_ENGINE_LABEL} "
                f"(found config key {ssm_key!r}={val})"
            )
    raw_ltypes = text_cfg.get("layer_types")
    if isinstance(raw_ltypes, list):
        for idx, lt in enumerate(raw_ltypes):
            lt_s = str(lt or "").strip()
            if lt_s not in ("full_attention", "linear_attention"):
                if "mamba" in lt_s.lower() or "ssm" in lt_s.lower():
                    return (
                        f"Mamba/SSM layer_type {lt_s!r} at layer {idx} is not supported "
                        f"by {M5RT_ENGINE_LABEL}"
                    )
                return (
                    f"Unsupported layer_type {lt_s!r} at layer {idx} "
                    "(supported: 'full_attention', 'linear_attention')"
                )
    hidden_act = text_cfg.get("hidden_act") or text_cfg.get("hidden_activation")
    if isinstance(hidden_act, str) and hidden_act.strip():
        if hidden_act.strip().lower() not in ("silu", "swiglu"):
            return (
                f"Unsupported hidden_act={hidden_act!r} "
                f"({M5RT_ENGINE_LABEL} requires 'silu' / SwiGLU)"
            )
    raw_experts = (
        text_cfg.get("num_experts")
        if text_cfg.get("num_experts") is not None
        else (
            text_cfg.get("n_routed_experts")
            if text_cfg.get("n_routed_experts") is not None
            else text_cfg.get("num_local_experts")
        )
    )
    if isinstance(raw_experts, int) and not isinstance(raw_experts, bool) and raw_experts != 0:
        if raw_experts < 2 or raw_experts > 256:
            return (
                f"MoE num_experts={raw_experts} is outside {M5RT_ENGINE_LABEL} "
                "supported range [2, 256]"
            )
    return ""


def _probe_provider_adapted(
    model_dir: str | Path,
    manifest: dict[str, Any],
    model_settings: object | None = None,
) -> ProbeResult:
    """Fast O(1) topology/extension/provider-aware probe for Path A (`m5rt`) and Path B providers.

    Checks in order with zero subprocess spawn and zero full-file hashing:
      (a) explicit `provider_id` / `engine_id` in `model_pack.json` / `omlxm5-manifest.json`,
      (b) structural capability check for built-in `m5rt` (decoupled from `model_type` string
          whitelist!), or `supported_topologies` / `supported_model_types` for third-party
          Path B providers unless `architecture_capabilities.model_type_whitelist_enforced` is False,
      (c) artifact files matching `manifest.get("artifact_extensions", [".miv", ".omlxm5"])`.
    """
    mdir = Path(model_dir).expanduser()
    provider_id = str(manifest.get("provider_id") or "").strip()
    engine_id = str(manifest.get("engine_id") or provider_id or M5RT_ENGINE_ID).strip()
    legacy_ids = [
        str(x).strip()
        for x in (manifest.get("legacy_provider_ids") or ())
        if isinstance(x, str) and x.strip()
    ]
    is_builtin = (engine_id == M5RT_ENGINE_ID) or (provider_id in M5RT_BUILTIN_PROVIDER_IDS)
    arch_caps = manifest.get("architecture_capabilities")
    whitelist_enforced = (not is_builtin) and not (
        isinstance(arch_caps, dict)
        and arch_caps.get("model_type_whitelist_enforced") is False
    )
    identity_tokens = {tok.lower() for tok in (provider_id, engine_id, *legacy_ids) if tok}
    if is_builtin:
        identity_tokens.update(k.lower() for k in M5RT_OPTION_KEYS)

    extensions = _normalize_extensions(manifest.get("artifact_extensions"))
    supported_topologies = {
        str(t).strip().lower()
        for t in (manifest.get("supported_topologies") or ())
        if isinstance(t, str) and t.strip()
    }
    if is_builtin:
        supported_topologies.update({"gdn_hybrid_gqa", "pure_gqa_swiglu"})
    supported_model_types = {
        str(m).strip().lower()
        for m in (manifest.get("supported_model_types") or ())
        if isinstance(m, str) and m.strip()
    }

    opts = _merged_provider_options(
        None,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id or M5RT_PROVIDER_ID,
        manifest=manifest,
    )
    explicit = (
        getattr(model_settings, "m5rt_artifact_path", None) if is_builtin else None
    ) or opts.get("artifact_path")
    if explicit:
        cand = Path(str(explicit)).expanduser()
        if cand.is_file() and cand.suffix.lower() in extensions:
            return ProbeResult(
                adapted=True,
                artifact_path=cand.resolve(),
                reason="explicit artifact_path",
                explicit_provider_match=True,
            )

    pack_meta: dict[str, Any] | None = None
    pack_name = ""
    for manifest_name in ("model_pack.json", "omlxm5-manifest.json"):
        mpath = mdir / manifest_name
        if not mpath.is_file():
            continue
        parsed = _read_json(mpath)
        if isinstance(parsed, dict):
            pack_meta = parsed
            pack_name = manifest_name
            break

    # (a) Explicit provider_id / engine_id claim in pack manifest
    explicit_provider_match = False
    if pack_meta is not None:
        raw_pack_provider = pack_meta.get("provider_id") or pack_meta.get("provider")
        raw_pack_engine = pack_meta.get("engine_id") or pack_meta.get("engine")
        pack_provider = (
            str(raw_pack_provider).strip()
            if isinstance(raw_pack_provider, str) and raw_pack_provider.strip()
            else None
        )
        pack_engine = (
            str(raw_pack_engine).strip()
            if isinstance(raw_pack_engine, str) and raw_pack_engine.strip()
            else None
        )
        if pack_provider is not None and pack_provider.lower() not in identity_tokens:
            return ProbeResult(
                adapted=False,
                reason=f"model pack targets provider '{pack_provider}', not '{provider_id or engine_id}'",
            )
        if pack_engine is not None and pack_engine.lower() not in identity_tokens:
            return ProbeResult(
                adapted=False,
                reason=f"model pack targets engine '{pack_engine}', not '{engine_id}'",
            )
        if pack_provider is not None or pack_engine is not None:
            explicit_provider_match = True

    # (b) Topology / structural compatibility check
    config = _read_json(mdir / "config.json") if (mdir / "config.json").is_file() else None
    if is_builtin and not explicit_provider_match and _is_dflash_family_architecture(config)[0]:
        return ProbeResult(
            adapted=False,
            reason="DFlash draft helper model is not a standalone engine target",
        )

    raw_topology = (
        (pack_meta.get("topology_id") or pack_meta.get("topology"))
        if pack_meta is not None
        else None
    ) or (config.get("topology_id") if isinstance(config, dict) else None)
    pack_topology = (
        str(raw_topology).strip()
        if isinstance(raw_topology, str) and raw_topology.strip()
        else None
    )
    model_types = _extract_model_type_tokens(config, pack_meta)

    topology_matched = False
    if pack_topology is not None:
        if supported_topologies:
            if pack_topology.lower() not in supported_topologies:
                return ProbeResult(
                    adapted=False,
                    reason=f"topology '{pack_topology}' not supported by '{provider_id or engine_id}'",
                )
            topology_matched = True

    model_type_matched = bool(model_types and supported_model_types and (model_types & supported_model_types))
    if whitelist_enforced and model_types and supported_model_types:
        if not model_type_matched and not topology_matched and not explicit_provider_match:
            return ProbeResult(
                adapted=False,
                reason=f"model_type {sorted(model_types)} not supported by '{provider_id or engine_id}'",
            )

    if is_builtin and not explicit_provider_match:
        cfg_blocker = _config_structural_blocker(config)
        if cfg_blocker:
            return ProbeResult(adapted=False, reason=cfg_blocker)

    def _can_claim_artifact(cand_path: Path | None, ext: str) -> bool:
        if explicit_provider_match or topology_matched or model_type_matched:
            return True
        if is_builtin:
            if (
                ext == ".omlxm5"
                and _foreign_spi_claims_model_artifact(model_types, ext)
                and not _has_miv_container_magic(cand_path)
            ):
                return False
            return True
        if not whitelist_enforced:
            return True
        # Non-builtin Path B providers without an explicit provider/topology/model_type
        # match may only auto-claim custom non-.miv/.omlxm5 artifact extensions.
        return ext not in (".miv", ".omlxm5")

    # (c) Artifact resolution matching manifest artifact_extensions
    if pack_meta is not None:
        art_rel = pack_meta.get("artifact_path") or pack_meta.get("artifact")
        art_path: Path | None = None
        if isinstance(art_rel, str) and art_rel.strip():
            cand = Path(art_rel.strip()).expanduser()
            if not cand.is_absolute():
                cand = mdir / cand
            if cand.suffix.lower() not in extensions:
                return ProbeResult(
                    adapted=False,
                    reason=f"artifact '{cand.name}' extension not in {list(extensions)}",
                )
            if cand.is_file():
                art_path = cand.resolve()

        if art_path is None:
            for ext in extensions:
                for cand in (
                    mdir / f"model{ext}",
                    mdir.parent / f"model{ext}" if mdir.name == "source" else None,
                ):
                    if cand is not None and cand.is_file():
                        art_path = cand.resolve()
                        break
                if art_path is not None:
                    break

        draft_rel = pack_meta.get("draft_artifact_path") or pack_meta.get("draft_artifact")
        draft_path: Path | None = None
        if isinstance(draft_rel, str) and draft_rel.strip():
            dcand = Path(draft_rel.strip()).expanduser()
            if not dcand.is_absolute():
                dcand = mdir / dcand
            if dcand.is_file():
                draft_path = dcand.resolve()

        matched_ext = art_path.suffix.lower() if art_path is not None else extensions[0]
        if _can_claim_artifact(art_path, matched_ext) and (
            art_path is not None or explicit_provider_match or is_builtin
        ):
            return ProbeResult(
                adapted=True,
                artifact_path=art_path,
                draft_path=draft_path,
                reason=pack_name,
                explicit_provider_match=explicit_provider_match,
            )

    for ext in extensions:
        direct = mdir / f"model{ext}"
        if direct.is_file() and _can_claim_artifact(direct, ext):
            return ProbeResult(
                adapted=True,
                artifact_path=direct.resolve(),
                reason=f"model{ext}",
                explicit_provider_match=explicit_provider_match,
            )
        parent_art = mdir.parent / f"model{ext}" if mdir.name == "source" else None
        if parent_art is not None and parent_art.is_file() and _can_claim_artifact(parent_art, ext):
            return ProbeResult(
                adapted=True,
                artifact_path=parent_art.resolve(),
                reason=f"fixture parent model{ext}",
                explicit_provider_match=explicit_provider_match,
            )
        if ext not in (".miv", ".omlxm5") and mdir.is_dir():
            custom_matches = [
                p
                for p in sorted(mdir.glob(f"*{ext}"))
                if p.is_file() and not p.name.startswith(".")
            ]
            if len(custom_matches) == 1 and _can_claim_artifact(custom_matches[0], ext):
                return ProbeResult(
                    adapted=True,
                    artifact_path=custom_matches[0].resolve(),
                    reason=custom_matches[0].name,
                    explicit_provider_match=explicit_provider_match,
                )

    return ProbeResult(
        adapted=False,
        reason=f"no matching {'/'.join(extensions)} artifact or manifest in model directory",
    )


def detect_adapted(
    model_dir: str | Path,
    model_settings: object | None = None,
) -> ProbeResult:
    """Fast O(1) filesystem probe for whether `model_dir` has an adapted `m5rt` artifact.

    Performs pure `stat()` / small-manifest checks with zero subprocess spawn and
    zero full-file hashing, guarded by built-in `io.ninfer.metal` topology/model_type/provider
    metadata so `m5rt` never hijacks foreign Path B `.omlxm5` artifacts.
    """
    return _probe_provider_adapted(model_dir, _builtin_m5rt_manifest(), model_settings)


def _search_artifact_cache(model_dir: Path) -> Path | None:
    """Match a model folder against `.cache/b0/artifacts/*.{miv,omlxm5}`."""
    cache_dirs: list[Path] = []
    env_cache = os.environ.get("OMLX_M5RT_ARTIFACT_DIR", "").strip()
    if env_cache:
        cache_dirs.append(Path(env_cache).expanduser())
    cache_dirs.append(_REPO_ROOT / _DEFAULT_ARTIFACT_CACHE_RELPATH)

    config_path = model_dir / "config.json"
    config_sha = _sha256_file(config_path) if config_path.is_file() else None
    model_slug = _normalize_slug(model_dir.name)

    slug_match: Path | None = None
    for cache_dir in cache_dirs:
        if not cache_dir.is_dir():
            continue
        for ext in M5RT_ARTIFACT_EXTENSIONS:
            for art in sorted(cache_dir.glob(f"*{ext}")):
                if not art.is_file() or art.name.startswith("."):
                    continue
                conv = _read_json(Path(f"{art}.conversion.json")) or {}
                if config_sha and conv.get("source_config_sha256") == config_sha:
                    return art.resolve()
                source_repo = str(conv.get("source_repo") or "")
                repo_tail = source_repo.rsplit("/", 1)[-1]
                if model_slug and (
                    _normalize_slug(repo_tail) == model_slug
                    or _normalize_slug(art.stem) == model_slug
                ):
                    slug_match = slug_match or art.resolve()
    return slug_match


def _resolve_target_artifact(
    model_dir: Path,
    model_settings: object | None,
    *,
    manifest: dict[str, Any] | None = None,
    engine_id: str = M5RT_ENGINE_ID,
    provider_id: str = M5RT_PROVIDER_ID,
    model_id: str | None = None,
) -> tuple[Path | None, str]:
    is_builtin = (engine_id == M5RT_ENGINE_ID) and (provider_id in M5RT_BUILTIN_PROVIDER_IDS)
    eff_manifest = manifest if manifest is not None else _builtin_m5rt_manifest()
    extensions = _normalize_extensions(eff_manifest.get("artifact_extensions"))
    opts = _merged_provider_options(
        model_dir,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id,
        manifest=eff_manifest,
        model_id=model_id,
    )
    explicit = (
        getattr(model_settings, "m5rt_artifact_path", None) if is_builtin else None
    ) or opts.get("artifact_path")
    if explicit:
        try:
            field_name = "m5rt_artifact_path" if is_builtin else f"{engine_id}.artifact_path"
            return (
                validate_provider_artifact_file(
                    explicit, allowed_extensions=extensions, field_name=field_name
                ),
                "",
            )
        except ValueError as exc:
            return None, str(exc)

    if is_builtin:
        env_art = os.environ.get("OMLX_M5RT_ARTIFACT", "").strip()
        if env_art:
            try:
                return validate_m5rt_artifact_file(env_art, "OMLX_M5RT_ARTIFACT"), ""
            except ValueError as exc:
                return None, str(exc)

    probe = _probe_provider_adapted(model_dir, eff_manifest, model_settings)
    if probe.adapted and probe.artifact_path is not None:
        return probe.artifact_path, ""

    local_candidates = [
        p
        for ext in extensions
        for p in sorted(model_dir.glob(f"*{ext}"))
        if p.is_file()
        and not p.name.startswith(".")
        and p.name.lower() not in ("mtp.miv", "mtp.omlxm5")
    ]
    if len(local_candidates) == 1:
        return local_candidates[0].resolve(), ""

    if is_builtin:
        cached = _search_artifact_cache(model_dir)
        if cached is not None:
            return cached, ""

    ext_desc = "/".join(extensions)
    hint = (
        "(set m5rt_artifact_path or OMLX_M5RT_ARTIFACT)"
        if is_builtin
        else f"(set external_engine_options['{engine_id}'].artifact_path)"
    )
    return (
        None,
        f"No {ext_desc} artifact found for '{model_dir.name}' {hint}",
    )


def _resolve_metallib_dir(
    model_dir: Path,
    binary: M5Binary,
    model_settings: object | None,
    *,
    engine_id: str | None = None,
    provider_id: str | None = None,
    model_id: str | None = None,
) -> tuple[Path | None, str]:
    eff_provider_id = provider_id or binary.provider_id
    eff_engine_id = engine_id or binary.engine_id
    is_builtin = binary.is_builtin_m5rt and eff_provider_id in M5RT_BUILTIN_PROVIDER_IDS
    opts = _merged_provider_options(
        model_dir,
        model_settings,
        engine_id=eff_engine_id,
        provider_id=eff_provider_id,
        manifest=binary.provider_manifest,
        model_id=model_id,
    )
    explicit = opts.get("metallib_dir") or (
        os.environ.get("OMLX_M5RT_METALLIB_DIR", "").strip() if is_builtin else ""
    )
    if explicit:
        candidate = Path(str(explicit)).expanduser()
        if is_builtin:
            if (candidate / "omlx_m5_qwen_ar.metallib").is_file():
                return candidate.resolve(), ""
            return (
                None,
                f"omlx_m5_qwen_ar.metallib not found in metallib_dir: {candidate}",
            )
        if candidate.is_dir():
            return candidate.resolve(), ""
        return None, f"metallib_dir is not a directory: {candidate}"

    search: list[Path] = []
    if is_builtin:
        search.append(model_dir)
    manifest_explicit_dir: Path | None = None
    if binary.provider_manifest:
        pdir_str = binary.provider_manifest.get("_provider_dir")
        mdir_rel = binary.provider_manifest.get("metallib_dir")
        if pdir_str and isinstance(mdir_rel, str) and mdir_rel.strip():
            cand = Path(mdir_rel.strip())
            if not cand.is_absolute():
                cand = Path(pdir_str) / cand
            manifest_explicit_dir = cand
            search.append(cand)
        if pdir_str:
            search.append(Path(pdir_str) / "metallib")

    search.extend(
        (
            binary.path.parent / "metallib",
            binary.path.parent.parent / "metallib",
        )
    )
    if is_builtin:
        search.extend(
            (
                binary.path.parent,
                _REPO_ROOT / "build/release/miv/metallib",
                _REPO_ROOT / "build/miv-v001/miv/metallib",
                _REPO_ROOT / "build/b6-xcode27/miv/metallib",
                _REPO_ROOT / "build/d0-m4/miv/metallib",
                _REPO_ROOT / "build/b6-xcode27/m5rt/metallib",
                _REPO_ROOT / "build/d0-m4/m5rt/metallib",
            )
        )
        build_root = _REPO_ROOT / "build"
        if build_root.is_dir():
            for subdir in ("miv", "m5rt"):
                for cand in sorted(build_root.glob(f"*/{subdir}/metallib")):
                    if cand not in search:
                        search.append(cand)

        for directory in search:
            if (directory / "omlx_m5_qwen_ar.metallib").is_file():
                return directory.resolve(), ""
        return (
            None,
            "omlx_m5_qwen_ar.metallib not found alongside mseries_backendd or model folder",
        )

    # Non-io.ninfer.metal Path B providers: `omlx_m5_qwen_ar.metallib` is optional.
    for directory in search:
        if not directory.is_dir():
            continue
        if any(directory.glob("*.metallib")) or directory == manifest_explicit_dir:
            return directory.resolve(), ""
    return None, ""


def _find_draft_artifact_in_dir(
    resolved_dir: Path,
    extensions: tuple[str, ...],
    *,
    model_dir: Path | None = None,
    is_builtin: bool = True,
) -> Path | None:
    """Locate a companion DFlash2 artifact (`model.<ext>`, `draft.<ext>`, `*.<ext>`, or `.cache/b0/artifacts`)."""
    parent_safe = False
    if resolved_dir.name == "source":
        try:
            parent_safe = (
                model_dir is None
                or resolved_dir.parent.resolve() != model_dir.resolve()
            )
        except OSError:
            parent_safe = False

    for ext in extensions:
        for cand in (resolved_dir / f"model{ext}", resolved_dir / f"draft{ext}"):
            if cand.is_file():
                return cand.resolve()
        if parent_safe:
            for cand in (
                resolved_dir.parent / f"model{ext}",
                resolved_dir.parent / f"draft{ext}",
            ):
                if cand.is_file():
                    return cand.resolve()

    for ext in extensions:
        matches = [
            p
            for p in sorted(resolved_dir.glob(f"*{ext}"))
            if p.is_file()
            and not p.name.startswith(".")
            and p.name.lower() not in ("mtp.miv", "mtp.omlxm5")
        ]
        if matches:
            return matches[0].resolve()
        if parent_safe:
            parent_matches = [
                p
                for p in sorted(resolved_dir.parent.glob(f"*{ext}"))
                if p.is_file()
                and not p.name.startswith(".")
                and p.name.lower() not in ("mtp.miv", "mtp.omlxm5")
            ]
            if parent_matches:
                return parent_matches[0].resolve()

    if is_builtin:
        cached = _search_artifact_cache(resolved_dir)
        if cached is not None:
            return cached

    return None


def _dflash2_draft_matches_target(
    draft_cfg: dict[str, Any] | None, target_cfg: dict[str, Any] | None
) -> bool:
    """Check whether a `DFlash2DraftModel` config matches the target model's geometry."""
    if not isinstance(draft_cfg, dict) or not isinstance(target_cfg, dict):
        return False
    archs = draft_cfg.get("architectures") or ()
    if isinstance(archs, str):
        archs = (archs,)
    if not isinstance(archs, (list, tuple)) or "DFlash2DraftModel" not in archs:
        return False
    ok, _ = _is_supported_dflash2_draft_config(
        draft_cfg, is_builtin=True, target_config=target_cfg
    )
    return ok


def _requested_speculative_backend(
    model_settings: object | None,
    explicit: dict[str, Any] | None,
    per_model: dict[str, Any] | None = None,
    shared: dict[str, Any] | None = None,
) -> str:
    raw = (explicit or {}).get("speculative_backend")
    if raw is None and getattr(model_settings, "mtp_enabled", False):
        raw = "mtp"
    if raw is None and getattr(model_settings, "dflash_enabled", False):
        raw = "dflash2"
    if raw is None:
        raw = (per_model or {}).get("speculative_backend")
    if raw is None:
        raw = (shared or {}).get("speculative_backend")
    backend = str(raw or "auto").strip().lower()
    if backend not in VALID_SPEC_BACKENDS:
        raise ValueError(f"Unsupported speculative backend: {backend!r}")
    return backend


def _resolve_draft_target(
    model_dir: Path,
    model_settings: object | None,
    draft_candidates: Iterable[str],
    *,
    manifest: dict[str, Any] | None = None,
    engine_id: str = M5RT_ENGINE_ID,
    provider_id: str = M5RT_PROVIDER_ID,
    allow_missing_artifact: bool = False,
    model_id: str | None = None,
) -> tuple[Path | None, Path | None, str]:
    is_builtin = (engine_id == M5RT_ENGINE_ID) and (provider_id in M5RT_BUILTIN_PROVIDER_IDS)
    extensions = _normalize_extensions(manifest.get("artifact_extensions") if manifest else None)
    shared_toml, model_toml, explicit_opts = _layered_provider_options(
        model_dir,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id,
        manifest=manifest,
        model_id=model_id,
    )
    opts = {**shared_toml, **model_toml, **explicit_opts}

    spec_backend = _requested_speculative_backend(
        model_settings, explicit_opts, model_toml, shared_toml
    )
    if spec_backend not in ("auto", "dflash2"):
        return None, None, ""

    explicit_draft_art = (
        (getattr(model_settings, "m5rt_draft_artifact_path", None) if is_builtin else None)
        or opts.get("draft_artifact_path")
        or (os.environ.get("OMLX_M5RT_DRAFT_ARTIFACT", "").strip() if is_builtin else "")
    )
    draft_art: Path | None = None
    if explicit_draft_art:
        try:
            draft_art = validate_provider_artifact_file(
                explicit_draft_art,
                allowed_extensions=extensions,
                field_name="m5rt_draft_artifact_path" if is_builtin else f"{engine_id}.draft_artifact_path",
            )
        except ValueError as exc:
            return None, None, str(exc)

    draft_dir_raw = opts.get("draft_dir") or getattr(
        model_settings, "dflash_draft_model", None
    )
    explicit_dirs: list[Path] = []
    if draft_dir_raw:
        explicit_dirs.append(Path(str(draft_dir_raw)).expanduser())
    if draft_art is not None:
        explicit_dirs.extend([draft_art.parent / "source", draft_art.parent])

    discovered_dirs: list[Path] = []
    companion_dir = model_dir / "dflash2"
    if companion_dir.is_dir():
        discovered_dirs.extend([companion_dir / "source", companion_dir])
    # Auto looks at explicitly selected or colocated drafts; explicit DFlash2
    # can also search the model library's helpers.
    if spec_backend == "dflash2" or draft_dir_raw:
        discovered_dirs.extend(Path(c).expanduser() for c in draft_candidates)

    target_cfg = _read_json(model_dir / "config.json")
    resolved_dir: Path | None = None
    for cdir in explicit_dirs:
        if (cdir / "config.json").is_file():
            resolved_dir = cdir.resolve()
            break

    if resolved_dir is None and discovered_dirs:
        fallback_dir: Path | None = None
        for cdir in discovered_dirs:
            cfg_file = cdir / "config.json"
            if not cfg_file.is_file():
                continue
            cand_resolved = cdir.resolve()
            if fallback_dir is None:
                fallback_dir = cand_resolved
            if _dflash2_draft_matches_target(_read_json(cfg_file), target_cfg) or (
                cand_resolved.is_relative_to(companion_dir.resolve())
                and _find_draft_artifact_in_dir(cand_resolved, extensions, model_dir=model_dir, is_builtin=is_builtin)
            ):
                resolved_dir = cand_resolved
                break
        if resolved_dir is None and spec_backend == "dflash2":
            resolved_dir = fallback_dir

    if is_builtin and resolved_dir is not None and target_cfg is not None:
        draft_cfg = _read_json(resolved_dir / "config.json")
        if _is_dflash_family_architecture(draft_cfg)[0]:
            ok_pair, pair_reason = _is_supported_dflash2_draft_config(
                draft_cfg, manifest, is_builtin=True, target_config=target_cfg
            )
            if not ok_pair:
                if spec_backend == "dflash2":
                    return None, None, pair_reason
                resolved_dir = None
                draft_art = None

    if draft_art is None and resolved_dir is not None:
        draft_art = _find_draft_artifact_in_dir(
            resolved_dir,
            extensions,
            model_dir=model_dir,
            is_builtin=is_builtin,
        )

    if spec_backend == "dflash2":
        if resolved_dir is None:
            return None, None, "DFlash2 speculative backend requires a draft model directory with config.json"
        if draft_art is None and not allow_missing_artifact:
            return None, None, f"DFlash2 speculative backend requires a draft {'/'.join(extensions)} artifact"

    return resolved_dir, draft_art, ""


def resolve_spi_provider_target(
    model_path: str | Path,
    binary_path: Path,
    manifest: dict[str, Any],
    model_settings: object | None = None,
    draft_candidates: Iterable[str] = (),
    *,
    allow_auto_convert: bool = True,
) -> tuple[M5Target | None, str]:
    """Check whether a Path B SPI provider can serve `model_path`, returning `(M5Target, reason)`."""
    model_dir = Path(model_path).expanduser()
    provider_id = str(manifest.get("provider_id") or "").strip()
    engine_id = str(manifest.get("engine_id") or provider_id).strip()
    proto_raw = manifest.get("protocol_version")
    fallback_proto = (
        proto_raw if isinstance(proto_raw, int) and not isinstance(proto_raw, bool) else None
    )

    binary, reason = inspect_m5rt_binary(
        binary_path,
        auto_detected=True,
        provider_manifest=manifest,
        allow_generic=True,
        fallback_protocol=fallback_proto,
    )
    if binary is None:
        return None, reason

    opts = _merged_provider_options(
        model_dir,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id,
        manifest=manifest,
    )
    fake = bool(opts.get("fake", False))
    if not (model_dir / "config.json").is_file() and not fake:
        return None, f"config.json not found in {model_dir}"

    if fake:
        return (
            M5Target(
                binary=binary,
                artifact_path=None,
                metallib_dir=None,
                fake=True,
            ),
            "",
        )

    metallib_dir, reason = _resolve_metallib_dir(
        model_dir,
        binary,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id,
    )
    if reason:
        return None, reason

    extensions = _normalize_extensions(manifest.get("artifact_extensions"))
    draft_dir, draft_art, reason = _resolve_draft_target(
        model_dir,
        model_settings,
        draft_candidates,
        manifest=manifest,
        engine_id=engine_id,
        provider_id=provider_id,
        allow_missing_artifact=allow_auto_convert,
    )
    if reason:
        return None, reason

    auto_convert_draft_output: Path | None = None
    if draft_dir is not None and draft_art is None:
        draft_conv_reason = ""
        if allow_auto_convert:
            can_draft_conv, draft_conv_reason = can_convert_provider_model(
                draft_dir,
                binary=binary,
                manifest=manifest,
                is_builtin=False,
            )
            if can_draft_conv:
                auto_convert_draft_output = _converter_output_path(
                    draft_dir, manifest, is_builtin=False
                )
        if auto_convert_draft_output is None:
            base_msg = f"DFlash2 speculative backend requires a draft {'/'.join(extensions)} artifact"
            if draft_conv_reason and _has_valid_safetensors_header(draft_dir):
                return None, f"{base_msg} ({draft_conv_reason})"
            return None, base_msg

    artifact_path, art_reason = _resolve_target_artifact(
        model_dir,
        model_settings,
        manifest=manifest,
        engine_id=engine_id,
        provider_id=provider_id,
    )
    auto_convert_output: Path | None = None
    if artifact_path is None:
        if allow_auto_convert:
            can_conv, _ = can_convert_provider_model(
                model_dir,
                binary=binary,
                manifest=manifest,
                model_settings=model_settings,
                is_builtin=False,
            )
            if can_conv:
                auto_convert_output = _converter_output_path(model_dir, manifest, is_builtin=False)
        if auto_convert_output is None:
            return None, art_reason

    return (
        M5Target(
            binary=binary,
            artifact_path=artifact_path,
            metallib_dir=metallib_dir,
            draft_dir=draft_dir,
            draft_artifact_path=draft_art,
            fake=False,
            auto_convert_output=auto_convert_output,
            auto_convert_draft_output=auto_convert_draft_output,
        ),
        "",
    )


def resolve_m5rt_target(
    model_path: str | Path,
    configured_binary: str | None,
    model_settings: object | None = None,
    draft_candidates: Iterable[str] = (),
    *,
    allow_auto_convert: bool = True,
    model_id: str | None = None,
) -> tuple[M5Target | None, str]:
    """Check whether `m5rt` can serve `model_path`, returning `(M5Target, reason)`."""
    model_dir = Path(model_path).expanduser()
    shared, per_model, explicit = _layered_provider_options(model_dir, model_settings, model_id=model_id)
    opts = {**shared, **per_model, **explicit}
    binary_override = opts.get("m5rt_path") or configured_binary
    binary, reason = find_m5rt_binary(binary_override)
    if binary is None:
        return None, reason

    fake = bool(opts.get("fake", False))
    if not (model_dir / "config.json").is_file() and not fake:
        return None, f"config.json not found in {model_dir}"

    if fake:
        return (
            M5Target(
                binary=binary,
                artifact_path=None,
                metallib_dir=None,
                fake=True,
            ),
            "",
        )

    metallib_dir, reason = _resolve_metallib_dir(model_dir, binary, model_settings, model_id=model_id)
    if metallib_dir is None:
        return None, reason

    manifest = _builtin_m5rt_manifest()
    extensions = _normalize_extensions(manifest.get("artifact_extensions"))
    draft_dir, draft_art, reason = _resolve_draft_target(
        model_dir,
        model_settings,
        draft_candidates,
        allow_missing_artifact=allow_auto_convert,
        model_id=model_id,
    )
    if reason:
        return None, reason

    auto_convert_draft_output: Path | None = None
    if draft_dir is not None and draft_art is None:
        draft_conv_reason = ""
        if allow_auto_convert:
            can_draft_conv, draft_conv_reason = can_convert_provider_model(
                draft_dir,
                binary=binary,
                manifest=manifest,
                is_builtin=True,
                target_model_dir=model_dir,
            )
            if can_draft_conv:
                auto_convert_draft_output = _converter_output_path(
                    draft_dir, manifest, is_builtin=True
                )
        if auto_convert_draft_output is None:
            if _requested_speculative_backend(model_settings, explicit, per_model, shared) == "dflash2":
                base_msg = f"DFlash2 speculative backend requires a draft {'/'.join(extensions)} artifact"
                if draft_conv_reason and _has_valid_safetensors_header(draft_dir):
                    return None, f"{base_msg} ({draft_conv_reason})"
                return None, base_msg
            draft_dir = None

    artifact_path, art_reason = _resolve_target_artifact(model_dir, model_settings, model_id=model_id)
    auto_convert_output: Path | None = None
    if artifact_path is None:
        conv_reason = ""
        if allow_auto_convert:
            can_conv, conv_reason = can_convert_provider_model(
                model_dir,
                binary=binary,
                manifest=manifest,
                model_settings=model_settings,
                is_builtin=True,
            )
            if can_conv:
                auto_convert_output = _converter_output_path(model_dir, manifest, is_builtin=True)
        if auto_convert_output is None:
            if (
                allow_auto_convert
                and conv_reason
                and _has_valid_safetensors_header(model_dir)
                and art_reason.startswith("No ")
            ):
                return None, conv_reason
            return None, art_reason

    return (
        M5Target(
            binary=binary,
            artifact_path=artifact_path,
            metallib_dir=metallib_dir,
            draft_dir=draft_dir,
            draft_artifact_path=draft_art,
            fake=False,
            auto_convert_output=auto_convert_output,
            auto_convert_draft_output=auto_convert_draft_output,
        ),
        "",
    )


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


_MIV_MAGIC = b"MIV00001"
_LEGACY_OMLXM5_MAGIC = b"OMLXM5A1"
_MIV_VALID_MAGICS: tuple[bytes, ...] = (_MIV_MAGIC, _LEGACY_OMLXM5_MAGIC)
_OMLXM5_MAGIC = _LEGACY_OMLXM5_MAGIC
_MIV_SCHEMAS_V1: tuple[str, ...] = ("miv-artifact-v1", "omlx-m5-artifact-v1")
_MIV_SCHEMAS_V2: tuple[str, ...] = ("miv-artifact-v2", "omlx-m5-artifact-v2")
_OMLXM5_HEADER_STRUCT = struct.Struct("<8sIIQQQQIIII")
_OMLXM5_HEADER_BYTES = 64
_OMLXM5_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def inspect_omlxm5_has_mtp_weights(artifact_path: Path | None) -> bool | None:
    """Inspect a `.miv` (or legacy `.omlxm5`) header + manifest for embedded MTP objects.

    Returns:
      - ``True`` if ``artifact_path`` is a valid ``MIV00001`` or legacy ``OMLXM5A1``
        container with at least one MTP object (``mtp.*`` or ``*.mtp.*``), matching
        C++ ``has_mtp_weights``.
      - ``False`` if it is a valid ``MIV00001`` / ``OMLXM5A1`` container with no MTP objects.
      - ``None`` if the file does not exist or is not a valid container
        (e.g. a synthetic test placeholder or mismatched header/schema).
    """
    if artifact_path is None:
        return None
    try:
        stat = artifact_path.stat()
    except OSError:
        return None
    return _inspect_mtp_manifest(artifact_path, stat.st_ino, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=64)
def _inspect_mtp_manifest(artifact_path: Path, inode: int, modified_ns: int, file_size: int) -> bool | None:
    try:
        if file_size < _OMLXM5_HEADER_BYTES:
            return None
        with artifact_path.open("rb") as fh:
            header = fh.read(_OMLXM5_HEADER_BYTES)
            if len(header) != _OMLXM5_HEADER_BYTES:
                return None
            (
                magic,
                version,
                header_bytes,
                manifest_offset,
                manifest_bytes,
                _payload_offset,
                _payload_bytes,
                _object_count,
                _manifest_crc,
                header_crc,
                flags,
            ) = _OMLXM5_HEADER_STRUCT.unpack(header)
            if (
                magic not in _MIV_VALID_MAGICS
                or version not in (1, 2)
                or header_bytes != _OMLXM5_HEADER_BYTES
                or manifest_offset != _OMLXM5_HEADER_BYTES
                or (version == 1 and flags != 0)
                or (flags & ~0x3) != 0
            ):
                return None
            if (zlib.crc32(header[:56]) & 0xFFFFFFFF) != header_crc:
                return None
            if (
                manifest_bytes == 0
                or manifest_bytes > _OMLXM5_MAX_MANIFEST_BYTES
                or manifest_offset + manifest_bytes > file_size
            ):
                return None
            fh.seek(manifest_offset)
            manifest_raw = fh.read(manifest_bytes)
    except OSError:
        return None
    if len(manifest_raw) != manifest_bytes:
        return None
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None
    if not isinstance(manifest, dict):
        return None
    if "schema" in manifest:
        valid_schemas = _MIV_SCHEMAS_V2 if version == 2 else _MIV_SCHEMAS_V1
        if manifest["schema"] not in valid_schemas:
            return None
    if "container_version" in manifest and manifest["container_version"] != version:
        return None
    objects = manifest.get("objects")
    if not isinstance(objects, list):
        return None
    for obj in objects:
        if not isinstance(obj, dict):
            continue
        name = obj.get("name")
        if isinstance(name, str) and (name.startswith("mtp.") or ".mtp." in name):
            return True
    return False


def _config_json_has_mtp(config: dict[str, Any] | None) -> bool:
    """Mirror C++ `config_json_has_mtp`: check top-level and `text_config` for MTP heads."""
    if not isinstance(config, dict):
        return False
    candidates = [config]
    nested = config.get("text_config")
    if isinstance(nested, dict):
        candidates.append(nested)
    for obj in candidates:
        for key in ("mtp_num_hidden_layers", "num_nextn_predict_layers"):
            val = obj.get(key)
            if isinstance(val, int) and not isinstance(val, bool) and val > 0:
                return True
    return False


def inspect_safetensors_has_mtp_weights(model_dir: Path | None) -> bool | None:
    """Inspect `model.safetensors.index.json` or `model.safetensors` header for MTP tensors.

    Returns ``True``/``False`` when real safetensors metadata is present (catching
    MLX conversions such as `OsaurusAI/Qwen3.6-35B-A3B-mxfp4` that retain
    `mtp_num_hidden_layers: 1` in `config.json` while stripping `mtp.*` weights),
    or ``None`` when no valid safetensors metadata exists (e.g. synthetic fixtures).
    """
    if model_dir is None:
        return None
    idx_json = _read_json(model_dir / "model.safetensors.index.json")
    if isinstance(idx_json, dict):
        weight_map = idx_json.get("weight_map")
        if isinstance(weight_map, dict) and weight_map:
            return any(
                isinstance(k, str) and (k.startswith("mtp.") or ".mtp." in k)
                for k in weight_map
            )
    single_st = model_dir / "model.safetensors"
    try:
        st_stat = single_st.stat()
        if st_stat.st_size < 8:
            return None
        with single_st.open("rb") as fh:
            hdr_len = int.from_bytes(fh.read(8), "little")
            if hdr_len <= 0 or hdr_len > _OMLXM5_MAX_MANIFEST_BYTES or 8 + hdr_len > st_stat.st_size:
                return None
            hdr_raw = fh.read(hdr_len)
        if len(hdr_raw) != hdr_len:
            return None
        hdr = json.loads(hdr_raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError):
        return None
    if not isinstance(hdr, dict):
        return None
    tensor_keys = [k for k in hdr if isinstance(k, str) and k != "__metadata__"]
    if not tensor_keys:
        return None
    return any(k.startswith("mtp.") or ".mtp." in k for k in tensor_keys)


def _effective_head_dim(text_cfg: dict[str, Any] | None) -> int | None:
    """Resolve explicit or implicit `head_dim = hidden_size // num_attention_heads`."""
    if not isinstance(text_cfg, dict):
        return None
    raw_hd = text_cfg.get("head_dim")
    if isinstance(raw_hd, int) and not isinstance(raw_hd, bool) and raw_hd > 0:
        return raw_hd
    raw_h = text_cfg.get("hidden_size")
    raw_q = text_cfg.get("num_attention_heads")
    if (
        isinstance(raw_h, int)
        and not isinstance(raw_h, bool)
        and raw_h > 0
        and isinstance(raw_q, int)
        and not isinstance(raw_q, bool)
        and raw_q > 0
        and raw_h % raw_q == 0
    ):
        return raw_h // raw_q
    return None


def _resolve_effective_kv_dtype(
    opts: dict[str, Any],
    model_settings: object | None,
    *,
    allowed_kv_dtypes: frozenset[str] | None = None,
    shared_toml: dict[str, Any] | None = None,
    model_toml: dict[str, Any] | None = None,
    explicit_opts: dict[str, Any] | None = None,
    head_dim: int | None = None,
    is_builtin: bool = False,
    fake: bool = False,
) -> str:
    allowed = allowed_kv_dtypes if allowed_kv_dtypes else VALID_KV_DTYPES

    def _validate_kv(raw: Any) -> str:
        kv = str(raw).strip()
        if kv in allowed:
            return kv
        logger.warning(
            "Sidecar Runtime: unsupported kv_dtype %r (supported: %s); falling back to 'bf16'",
            kv,
            sorted(allowed),
        )
        return "bf16"

    # Tier 1a: explicit per-model m5rt_options / external_engine_options
    if explicit_opts is not None:
        if explicit_opts.get("kv_dtype") is not None:
            return _validate_kv(explicit_opts["kv_dtype"])
    elif shared_toml is None and model_toml is None and opts.get("kv_dtype") is not None:
        return _validate_kv(opts["kv_dtype"])

    # Tier 1b: explicit oMLX per-model UI KV compression toggle
    if bool(getattr(model_settings, "turboquant_kv_enabled", False)):
        scheme = str(getattr(model_settings, "turboquant_kv_scheme", "") or "").strip().lower()
        raw_bits = getattr(model_settings, "turboquant_kv_bits", None)
        if scheme == "affine4" and "affine4_g128" in allowed:
            return "affine4_g128"
        if (
            scheme in ("", "turboquant")
            and (
                raw_bits is None
                or (
                    isinstance(raw_bits, (int, float))
                    and not isinstance(raw_bits, bool)
                    and float(raw_bits) <= 4.0
                )
            )
            and "affine4_g128" in allowed
        ):
            return "affine4_g128"
        # CRITICAL: Do NOT map "affine8" to "int8_g64" — miv only implements
        # bf16 and affine4_g128 in kv_page_pool.cpp.
        logger.warning(
            "miv Core Backend: turboquant_kv_scheme=%r (bits=%r) is not implemented in miv "
            "(only 4-bit 'affine4'/'turboquant' -> 'affine4_g128' or disabled -> 'bf16'); using 'bf16'",
            scheme,
            raw_bits,
        )
        return "bf16"

    # Tier 2: [sidecar.models."<name>"] in TOML
    if model_toml is not None and model_toml.get("kv_dtype") is not None:
        return _validate_kv(model_toml["kv_dtype"])

    # Tier 3: [engine] in shared ninfer-metal.toml (affine4_g128 NAX KV requires head_dim == 256)
    if shared_toml is not None and shared_toml.get("kv_dtype") is not None:
        resolved_shared = _validate_kv(shared_toml["kv_dtype"])
        if (
            is_builtin
            and resolved_shared == "affine4_g128"
            and head_dim is not None
            and head_dim != 256
        ):
            return "bf16"
        return resolved_shared

    # Tier 4: structural fallback based on head_dim == 256 for built-in miv
    if is_builtin and not fake and head_dim == 256 and "affine4_g128" in allowed:
        return "affine4_g128"
    return "bf16"


def _resolve_effective_speculative(
    opts: dict[str, Any],
    model_settings: object | None,
    target: M5Target | None = None,
    *,
    shared_toml: dict[str, Any] | None = None,
    model_toml: dict[str, Any] | None = None,
    explicit_opts: dict[str, Any] | None = None,
    model_dir: Path | None = None,
) -> tuple[str, int]:
    layered = any(layer is not None for layer in (explicit_opts, model_toml, shared_toml))
    explicit_opts = explicit_opts if layered else opts
    spec_backend = _requested_speculative_backend(
        model_settings, explicit_opts, model_toml, shared_toml
    )
    if spec_backend == "auto":
        # "auto" (or unset): detect companion or embedded speculative weights
        # matching standalone `ninfer-metal-serve` (`resolve_auto_speculative`).
        raw_mtp_art = (
            opts.get("mtp_artifact_path")
            or opts.get("mtp_artifact")
            or (
                opts["path_overrides"].get("mtp_artifact")
                if isinstance(opts.get("path_overrides"), dict)
                else None
            )
        )
        mtp_art_path = (
            Path(str(raw_mtp_art)).expanduser()
            if isinstance(raw_mtp_art, (str, Path)) and str(raw_mtp_art).strip()
            else None
        )
        target_art = (
            target.artifact_path
            if target is not None and target.artifact_path is not None
            else (
                (
                    model_dir / "model.miv"
                    if (model_dir / "model.miv").is_file()
                    or not (model_dir / "model.omlxm5").is_file()
                    else model_dir / "model.omlxm5"
                )
                if model_dir is not None
                else None
            )
        )
        if (
            target is not None
            and (
                target.draft_artifact_path is not None
                or target.auto_convert_draft_output is not None
            )
            and target.draft_dir is not None
        ):
            spec_backend = "dflash2"
        elif mtp_art_path is not None and mtp_art_path.is_file():
            spec_backend = "mtp"
        elif model_dir is not None and (
            (model_dir / "mtp.miv").is_file() or (model_dir / "mtp.omlxm5").is_file()
        ):
            spec_backend = "mtp"
        else:
            embedded_mtp = inspect_omlxm5_has_mtp_weights(target_art)
            if embedded_mtp is True:
                spec_backend = "mtp"
            elif embedded_mtp is False:
                spec_backend = "none"
            else:
                st_mtp = inspect_safetensors_has_mtp_weights(model_dir)
                if st_mtp is True:
                    spec_backend = "mtp"
                elif st_mtp is False:
                    spec_backend = "none"
                elif (
                    model_dir is not None
                    and _config_json_has_mtp(_read_json(model_dir / "config.json"))
                ):
                    spec_backend = "mtp"
                else:
                    spec_backend = "none"

    if spec_backend == "none":
        return "none", 0

    raw_depth = (explicit_opts or {}).get("draft_tokens")
    from_stock_dflash_block = False
    if raw_depth is None and spec_backend == "mtp":
        raw_depth = getattr(model_settings, "mtp_fixed_depth", None)
        if raw_depth is None:
            raw_depth = getattr(model_settings, "mtp_adaptive_max_depth", None)
    if raw_depth is None and spec_backend == "dflash2":
        block = getattr(model_settings, "dflash_block_size", None)
        if block is not None and int(block) > 0:
            raw_depth = int(block) - 1
            from_stock_dflash_block = True
    if raw_depth is None:
        raw_depth = (model_toml or {}).get("draft_tokens")
    if raw_depth is None:
        raw_depth = (shared_toml or {}).get("draft_tokens", 0)
    depth = int(raw_depth)
    maximum = M5RT_MAX_DRAFT_TOKENS[spec_backend]
    if from_stock_dflash_block and depth > maximum:
        logger.warning(
            "miv Core Backend: Stock dflash_block_size=%d exceeds miv DFlash2 max "
            "block size %d (draft_tokens=%d); clamping draft_tokens to %d",
            depth + 1,
            maximum + 1,
            maximum,
            maximum,
        )
        depth = maximum
    if not 0 <= depth <= maximum:
        raise ValueError(f"{spec_backend} draft_tokens must be in [0, {maximum}] (0 = auto)")
    return spec_backend, depth


def _resolve_effective_concurrency(
    opts: dict[str, Any], global_settings: Any | None = None
) -> int:
    raw = opts.get("max_concurrency")
    if raw is None and global_settings is not None:
        sched = getattr(global_settings, "scheduler", None)
        raw = getattr(sched, "max_concurrent_requests", None)
    return max(1, min(8, int(raw or 1)))


def _resolve_prefix_cache_bytes(
    opts: dict[str, Any],
    *,
    shared_toml: dict[str, Any] | None = None,
    model_toml: dict[str, Any] | None = None,
    explicit_opts: dict[str, Any] | None = None,
) -> int:
    layers = (
        (explicit_opts, model_toml, shared_toml)
        if (explicit_opts is not None or model_toml is not None or shared_toml is not None)
        else (opts,)
    )
    for layer in layers:
        if not layer:
            continue
        if layer.get("prefix_cache_bytes") is not None:
            return max(0, int(layer["prefix_cache_bytes"]))
        if layer.get("prefix_cache_gb") is not None:
            return max(0, int(float(layer["prefix_cache_gb"]) * (1 << 30)))
    return 0


def _kv_bytes_per_token(text_cfg: dict[str, Any], kv_dtype: str) -> int:
    layer_types = text_cfg.get("layer_types") or []
    full_layers = sum(1 for k in layer_types if k == "full_attention")
    if not full_layers:
        num_layers = int(text_cfg.get("num_hidden_layers") or 0)
        interval = max(1, int(text_cfg.get("full_attention_interval") or 1))
        full_layers = num_layers // interval
    kv_heads = int(text_cfg.get("num_key_value_heads") or 0)
    head_dim = _effective_head_dim(text_cfg) or 128
    values = full_layers * 2 * kv_heads * head_dim
    if values <= 0:
        return 0
    if kv_dtype == "affine4_g128":
        # Exact 0.5625 B/val (9/16 = 4-bit payload + 16B scale/min metadata per 256 vals).
        return (values * 9 + 15) // 16
    return values * 2


def _manifest_allowed_kv_dtypes(manifest: dict[str, Any] | None) -> frozenset[str] | None:
    if not isinstance(manifest, dict):
        return None
    raw = manifest.get("supported_kv_formats") or manifest.get("kv_formats")
    if not isinstance(raw, (list, tuple)) or not raw:
        return None
    vals = frozenset(str(x).strip() for x in raw if isinstance(x, str) and x.strip())
    return vals or None


def resolve_sidecar_lora_settings(
    model_dir: str | Path | None,
    model_settings: object | None,
    *,
    shared_toml: dict[str, Any] | None = None,
    model_toml: dict[str, Any] | None = None,
    explicit_opts: dict[str, Any] | None = None,
    lora_adapters: dict[str, str] | None = None,
    model_dirs: Iterable[str | Path] = (),
    engine_id: str = M5RT_ENGINE_ID,
    provider_id: str = M5RT_PROVIDER_ID,
    manifest: dict[str, Any] | None = None,
) -> tuple[str, str]:
    """Resolve `(lora_adapter_abs_path, lora_mode)` for SPI v2 sidecar `load` RPC."""
    mdir = Path(model_dir).expanduser() if model_dir is not None else None
    if shared_toml is None and model_toml is None and explicit_opts is None:
        shared_toml, model_toml, explicit_opts = _layered_provider_options(
            mdir,
            model_settings,
            engine_id=engine_id,
            provider_id=provider_id,
            manifest=manifest,
        )

    is_builtin = (engine_id == M5RT_ENGINE_ID) or (provider_id in M5RT_BUILTIN_PROVIDER_IDS)
    supports_std_lora = is_builtin or (
        "lora_adapter" in get_external_engine_supported_settings(engine_id)
    )
    if not supports_std_lora and isinstance(manifest, dict):
        raw_std = manifest.get("supported_standard_settings") or ()
        supports_std_lora = any(str(s).strip() == "lora_adapter" for s in raw_std)

    raw_lora = ""
    ms_lora = getattr(model_settings, "lora_adapter", None) if supports_std_lora else None
    if isinstance(ms_lora, str) and ms_lora.strip():
        raw_lora = ms_lora.strip()
    else:
        for layer in (explicit_opts, model_toml, shared_toml):
            if layer is not None and "lora_adapter" in layer and layer["lora_adapter"] is not None:
                raw_lora = str(layer["lora_adapter"]).strip()
                break

    resolved_lora = ""
    if raw_lora:
        resolved_lora = str(
            resolve_adapter_path(
                raw_lora,
                adapters=lora_adapters,
                model_dir=mdir,
                model_dirs=model_dirs,
            )
        )

    raw_mode: str | None = None
    for layer in (explicit_opts, model_toml, shared_toml):
        if layer is not None and layer.get("lora_mode") is not None:
            cand_mode = str(layer["lora_mode"]).strip().lower()
            if cand_mode:
                raw_mode = cand_mode
                break
    if raw_mode is None:
        lora_mode = "merge"
    elif raw_mode in VALID_LORA_MODES:
        lora_mode = raw_mode
    else:
        logger.warning(
            "Sidecar Runtime: unsupported lora_mode=%r (expected %s); falling back to 'merge'",
            raw_mode,
            sorted(VALID_LORA_MODES),
        )
        lora_mode = "merge"

    return resolved_lora, lora_mode


def _estimate_shared_kv_pool_bytes(
    text_cfg: dict[str, Any],
    kv_dtype: str,
    max_context: int,
    max_concurrency: int,
    *,
    model_type: str | None = None,
) -> int:
    """Estimate shared `KvPagePool` base allocation matching `native_backend.cpp`."""
    del model_type
    inferred_hd = _effective_head_dim(text_cfg)
    head_dim = inferred_hd if inferred_hd is not None else 128
    eff_kv_dtype = str(kv_dtype or "").strip().lower()
    if eff_kv_dtype in ("", "auto"):
        # affine4_g128 NAX KV cache requires head_dim == 256; use structural head_dim
        # rather than model_type string whitelisting.
        eff_kv_dtype = "affine4_g128" if head_dim == 256 else "bf16"
    layer_types = text_cfg.get("layer_types") or []
    full_layers = sum(1 for k in layer_types if k == "full_attention")
    if not full_layers:
        num_layers = int(text_cfg.get("num_hidden_layers") or 0)
        interval = max(1, int(text_cfg.get("full_attention_interval") or 1))
        full_layers = num_layers // interval
    kv_heads = int(text_cfg.get("num_key_value_heads") or 0)
    if full_layers <= 0 or kv_heads <= 0 or head_dim <= 0 or max_context <= 0:
        return 0
    page_tokens = 64
    logical_pages = (int(max_context) + page_tokens - 1) // page_tokens
    b_slots = max(1, int(max_concurrency))
    if b_slots > 1:
        physical_pages = min(
            logical_pages * b_slots + b_slots,
            max(logical_pages + 8 * b_slots, 1088),
        )
        physical_pages = max(physical_pages, logical_pages + 1)
    else:
        physical_pages = logical_pages + 1
    if eff_kv_dtype == "affine4_g128":
        vec_bytes = (head_dim // 2) + 16
        head_bytes = page_tokens * vec_bytes + 32
        kv_page_b = kv_heads * head_bytes
    else:
        kv_page_b = page_tokens * kv_heads * head_dim * 2
    return physical_pages * full_layers * 2 * kv_page_b


def m5rt_memory_estimate(
    model_path: str | Path,
    target: M5Target,
    max_context: int | None,
    model_settings: object | None = None,
    *,
    global_settings: Any | None = None,
    engine_id: str | None = None,
    provider_id: str | None = None,
    manifest: dict[str, Any] | None = None,
    lora_adapters: dict[str, str] | None = None,
    model_dirs: Iterable[str | Path] = (),
    model_id: str | None = None,
    plan: M5Options | None = None,
) -> int:
    """Estimate unified memory footprint for `M5SidecarEngine`."""
    mdir = Path(model_path)
    if plan is None:
        plan = build_m5rt_options(
            mdir, target, model_settings, default_context=max_context,
            memory_budget_bytes=0, global_settings=global_settings,
            engine_id=engine_id, provider_id=provider_id, manifest=manifest,
            lora_adapters=lora_adapters, model_dirs=model_dirs, model_id=model_id,
        )
    lora_bytes = (
        estimate_sidecar_lora_bytes(plan.lora_adapter, lora_mode=plan.lora_mode)
        if plan.lora_adapter else 0
    )
    if target.fake:
        return 64 * 1024**2 + lora_bytes
    raw_cfg = _read_json(mdir / "config.json") or {}
    text_cfg = _text_config(raw_cfg)
    model_type = str(raw_cfg.get("model_type") or text_cfg.get("model_type") or "").strip().lower()
    vocab_size = max(256, int(text_cfg.get("vocab_size") or 248320))

    if target.artifact_path is not None:
        weights = target.artifact_path.stat().st_size
    else:
        try:
            st_bytes = sum(
                f.stat().st_size for f in mdir.glob("*.safetensors") if f.is_file()
            )
        except OSError:
            st_bytes = 0
        weights = int(st_bytes * 1.02) if st_bytes > 0 else 0
    if target.draft_artifact_path is not None:
        draft = target.draft_artifact_path.stat().st_size
    elif target.draft_dir is not None and target.auto_convert_draft_output is not None:
        try:
            draft_st = sum(
                f.stat().st_size
                for f in target.draft_dir.glob("*.safetensors")
                if f.is_file()
            )
        except OSError:
            draft_st = 0
        draft = int(draft_st * 1.01) if draft_st > 0 else 0
    else:
        draft = 0
    kv = _estimate_shared_kv_pool_bytes(
        text_cfg,
        plan.kv_dtype,
        plan.max_context,
        plan.max_concurrency,
        model_type=model_type,
    )
    workspace = max(64 * 1024**2, 1024 * vocab_size * 2 * plan.max_concurrency)
    return weights + draft + kv + workspace + plan.prefix_cache_bytes + lora_bytes


estimate_m5rt_memory_bytes = m5rt_memory_estimate


def _safe_lora_fingerprint(
    lora_ref: str | None,
    *,
    lora_adapters: dict[str, str] | None = None,
    model_path: str | Path | None = None,
    model_dirs: Iterable[str | Path] = (),
) -> str | None:
    if not lora_ref or not str(lora_ref).strip():
        return None
    with contextlib.suppress(Exception):
        resolved = resolve_adapter_path(
            lora_ref,
            adapters=lora_adapters,
            model_dir=model_path,
            model_dirs=model_dirs,
        )
        if resolved.is_dir():
            return lora_fingerprint(resolved)
    return None


def m5rt_runtime_signature(
    model_settings: object | None,
    global_m5rt_path: str | None,
    *,
    global_settings: Any | None = None,
    lora_adapters: dict[str, str] | None = None,
    model_path: str | Path | None = None,
    model_dirs: Iterable[str | Path] = (),
    model_id: str | None = None,
    target: M5Target | None = None,
    default_context: int | None = None,
) -> tuple[Any, ...]:
    """Canonical tuple of load-time M5 settings that require engine reload."""
    if model_path is not None and target is not None:
        plan = build_m5rt_options(
            model_path, target, model_settings,
            default_context=default_context, memory_budget_bytes=0,
            global_settings=global_settings, model_id=model_id,
            lora_adapters=lora_adapters, model_dirs=model_dirs,
        )
        return (
            str(target.binary.path), target.binary.version,
            str(target.artifact_path), str(target.metallib_dir),
            str(target.draft_dir), str(target.draft_artifact_path),
            asdict(replace(plan, concurrency_source="")),
            _safe_lora_fingerprint(plan.lora_adapter),
        )
    global_conc = (
        getattr(getattr(global_settings, "scheduler", None), "max_concurrent_requests", None)
        if global_settings is not None
        else None
    )
    if model_settings is None:
        return (global_m5rt_path or None, None, None, None, None, global_conc)
    opts = getattr(model_settings, "m5rt_options", None) or None
    canonical_opts = (
        tuple(sorted((k, repr(v)) for k, v in opts.items()))
        if isinstance(opts, dict) and opts
        else None
    )
    ext_opts = getattr(model_settings, "external_engine_options", None) or None
    canonical_ext_opts = (
        tuple(
            sorted(
                (eng_k, tuple(sorted((k, repr(v)) for k, v in eng_v.items())))
                for eng_k, eng_v in ext_opts.items()
                if isinstance(eng_v, dict)
            )
        )
        if isinstance(ext_opts, dict) and ext_opts
        else None
    )
    lora_ref = getattr(model_settings, "lora_adapter", None) or None
    lora_fp = _safe_lora_fingerprint(
        lora_ref,
        lora_adapters=lora_adapters,
        model_path=model_path,
        model_dirs=model_dirs,
    )
    synced_standard = (
        getattr(model_settings, "max_context_window", None),
        bool(getattr(model_settings, "mtp_enabled", False)),
        getattr(model_settings, "mtp_fixed_depth", None),
        getattr(model_settings, "mtp_adaptive_max_depth", None),
        bool(getattr(model_settings, "dflash_enabled", False)),
        getattr(model_settings, "dflash_draft_model", None),
        getattr(model_settings, "dflash_block_size", None),
        bool(getattr(model_settings, "turboquant_kv_enabled", False)),
        getattr(model_settings, "turboquant_kv_scheme", None),
        getattr(model_settings, "turboquant_kv_bits", None),
        lora_ref,
        lora_fp,
    )
    return (
        global_m5rt_path or None,
        getattr(model_settings, "m5rt_artifact_path", None) or None,
        getattr(model_settings, "m5rt_draft_artifact_path", None) or None,
        canonical_opts,
        canonical_ext_opts,
        synced_standard,
        global_conc,
    )


def m5rt_status(global_settings: Any, engine_pool: Any | None = None) -> dict[str, Any]:
    """Admin global-settings status payload for miv Core Backend (`ninfer-metal`)."""
    configured = getattr(getattr(global_settings, "integrations", None), "m5rt_path", None)
    binary, reason = find_m5rt_binary(configured)
    manifest = _builtin_m5rt_manifest()
    if binary and isinstance(binary.provider_manifest, dict):
        for k in ("supported_topologies", "supported_model_types", "display_version"):
            if binary.provider_manifest.get(k):
                manifest[k] = binary.provider_manifest[k]
    models = []
    entries = getattr(engine_pool, "_entries", None) or {}
    for model_id, entry in list(entries.items()):
        engine = getattr(entry, "engine", None)
        if (
            not isinstance(engine, M5SidecarEngine)
            or engine.external_engine_id != M5RT_ENGINE_ID
        ):
            continue
        stats = engine.get_stats()
        models.append(
            {
                "id": model_id,
                "memory_bytes": stats["process_memory_bytes"],
                "memory_budget_bytes": stats["memory_budget_bytes"],
                "lora_adapter": stats.get("lora_adapter", ""),
                "lora_mode": stats.get("lora_mode", "merge"),
                "lora_buffer_bytes": stats.get("lora_buffer_bytes", 0),
                "log_path": stats["log_path"],
            }
        )
    supported_topologies = list(
        manifest.get("supported_topologies") or M5RT_SUPPORTED_TOPOLOGIES
    )
    supported_model_types = list(
        manifest.get("supported_model_types") or M5RT_SUPPORTED_MODEL_TYPES
    )
    artifact_extensions = list(
        _normalize_extensions(manifest.get("artifact_extensions"))
        if manifest.get("artifact_extensions")
        else M5RT_ARTIFACT_EXTENSIONS
    )
    version = (
        str(manifest.get("display_version") or "").strip()
        or (binary.version if binary else "")
        or M5RT_DEFAULT_VERSION
    )
    return {
        "available": binary is not None,
        "installed": binary is not None,
        "provider_id": binary.provider_id if binary else M5RT_PROVIDER_ID,
        "engine_id": M5RT_ENGINE_ID,
        "display_name": M5RT_ENGINE_LABEL,
        "path_tier": "Path A",
        "codename": (binary.codename if binary else None) or "miv",
        "protocol_version": binary.protocol_version if binary else 2,
        "path": str(binary.path) if binary else None,
        "binary_path": str(binary.path) if binary else None,
        "auto_detected": binary.auto_detected if binary else None,
        "version": version,
        "label": binary.label if binary else M5RT_ENGINE_LABEL,
        "capabilities": list(manifest.get("capabilities") or M5RT_DEFAULT_CAPABILITIES),
        "supported_topologies": supported_topologies,
        "supported_model_types": supported_model_types,
        "artifact_extensions": artifact_extensions,
        "kv_formats": sorted(VALID_KV_DTYPES),
        "supported_standard_settings": list(
            manifest.get("supported_standard_settings") or M5RT_SUPPORTED_STANDARD_SETTINGS
        ),
        "options_schema": list(manifest.get("options_schema") or M5RT_DEFAULT_OPTIONS_SCHEMA),
        "reason": reason,
        "install_hint": M5RT_INSTALL_HINT,
        "models": models,
    }


def build_m5rt_options(
    model_path: str | Path,
    target: M5Target,
    model_settings: object | None,
    *,
    default_context: int | None,
    memory_budget_bytes: int,
    global_settings: Any | None = None,
    model_id: str | None = None,
    engine_id: str | None = None,
    provider_id: str | None = None,
    manifest: dict[str, Any] | None = None,
    lora_adapters: dict[str, str] | None = None,
    model_dirs: Iterable[str | Path] = (),
) -> M5Options:
    """Build `M5Options` from `ModelSettings`, provider options, and model config."""
    mdir = Path(model_path)
    eff_manifest = manifest if manifest is not None else target.binary.provider_manifest
    eff_provider_id = provider_id or target.binary.provider_id
    eff_engine_id = engine_id or target.binary.engine_id
    shared_toml, model_toml, explicit_opts = _layered_provider_options(
        mdir,
        model_settings,
        engine_id=eff_engine_id,
        provider_id=eff_provider_id,
        manifest=eff_manifest,
        model_id=model_id,
    )
    opts = {**shared_toml, **model_toml, **explicit_opts}
    config = _read_json(mdir / "config.json") or {}
    text_cfg = _text_config(config)
    native_ctx_raw = text_cfg.get("max_position_embeddings")

    per_model_context = (
        explicit_opts.get("max_context")
        or getattr(model_settings, "max_context_window", None)
        or model_toml.get("max_context")
    )
    configured_ctx = per_model_context or shared_toml.get("max_context") or default_context
    if not configured_ctx and not native_ctx_raw:
        configured_ctx = getattr(getattr(global_settings, "sampling", None), "max_context_window", None)
    native_ctx = int(native_ctx_raw) if native_ctx_raw else max(32768, int(configured_ctx or 32768))
    max_context = max(1024, min(int(configured_ctx or native_ctx), native_ctx))
    policy = getattr(getattr(global_settings, "sampling", None), "max_context_window_policy", None)
    has_explicit_model_ctx = bool(getattr(model_settings, "max_context_window", None)) or (
        explicit_opts.get("max_context") is not None
    )
    if not has_explicit_model_ctx and policy:
        max_context = max(1024, min(max_context, int(policy)))

    max_concurrency = _resolve_effective_concurrency(opts, global_settings)
    if explicit_opts.get("max_concurrency") is not None:
        concurrency_source = "model_options"
    elif model_toml.get("max_concurrency") is not None:
        concurrency_source = "model_toml"
    elif shared_toml.get("max_concurrency") is not None:
        concurrency_source = "shared_toml"
    elif getattr(getattr(global_settings, "scheduler", None), "max_concurrent_requests", None):
        concurrency_source = "global"
    else:
        concurrency_source = "default"
    is_miv = eff_engine_id == M5RT_ENGINE_ID and eff_provider_id in M5RT_BUILTIN_PROVIDER_IDS
    allowed_kv = (
        None
        if eff_provider_id in M5RT_BUILTIN_PROVIDER_IDS
        else _manifest_allowed_kv_dtypes(eff_manifest)
    )
    kv_dtype = _resolve_effective_kv_dtype(
        opts,
        model_settings,
        allowed_kv_dtypes=allowed_kv,
        shared_toml=shared_toml,
        model_toml=model_toml,
        explicit_opts=explicit_opts,
        head_dim=_effective_head_dim(text_cfg),
        is_builtin=is_miv,
        fake=bool(target.fake),
    )
    spec_backend, draft_tokens = _resolve_effective_speculative(
        opts,
        model_settings,
        target,
        shared_toml=shared_toml,
        model_toml=model_toml,
        explicit_opts=explicit_opts,
        model_dir=mdir,
    )
    lora_adapter, lora_mode = resolve_sidecar_lora_settings(
        mdir,
        model_settings,
        shared_toml=shared_toml,
        model_toml=model_toml,
        explicit_opts=explicit_opts,
        lora_adapters=lora_adapters,
        model_dirs=model_dirs,
        engine_id=eff_engine_id,
        provider_id=eff_provider_id,
        manifest=eff_manifest,
    )

    raw_overrides = opts.get("path_overrides")
    overrides = (
        {
            str(k): ("on" if v else "off") if isinstance(v, bool) else str(v)
            for k, v in raw_overrides.items() if v is not None
        }
        if isinstance(raw_overrides, dict)
        else {}
    )
    if is_miv:
        if spec_backend == "none":
            overrides["enable_mtp"] = "off"
        elif spec_backend == "mtp" and explicit_opts.get("draft_tokens") is None:
            if getattr(model_settings, "mtp_fixed_depth", None) is not None:
                overrides["mtp_fixed_depth"] = "on"
            elif getattr(model_settings, "mtp_adaptive_max_depth", None) is not None:
                overrides["mtp_fixed_depth"] = "off"
    prefix_layers = (explicit_opts, model_toml, shared_toml)
    prefix_bytes = _resolve_prefix_cache_bytes(
        opts, shared_toml=shared_toml, model_toml=model_toml, explicit_opts=explicit_opts
    )
    if not any(layer.get(key) is not None for layer in prefix_layers
               for key in ("prefix_cache_bytes", "prefix_cache_gb")):
        prefix_bytes = max(0, int(overrides.get("prefix_cache_bytes", "0")))
    mtp_art_raw = opts.get("mtp_artifact_path") or opts.get("mtp_artifact")
    if mtp_art_raw and "mtp_artifact" not in overrides:
        overrides["mtp_artifact"] = str(Path(str(mtp_art_raw)).expanduser())
    elif spec_backend == "mtp" and "mtp_artifact" not in overrides:
        for mtp_name in ("mtp.miv", "mtp.omlxm5"):
            companion_mtp = mdir / mtp_name
            if companion_mtp.is_file():
                overrides["mtp_artifact"] = str(companion_mtp.resolve())
                break

    rolling_prefix_cache = _parse_optional_toggle(
        opts.get("rolling_prefix_cache"), f"{eff_engine_id}.rolling_prefix_cache"
    )
    raw_batch_mode = opts.get("batch_mode")
    if isinstance(raw_batch_mode, str) and not raw_batch_mode.strip():
        raw_batch_mode = None
    batch_mode = _parse_optional_toggle(
        raw_batch_mode, f"{eff_engine_id}.batch_mode"
    )

    raw_linear_mode = str(opts.get("prefill_linear_mode") or "bf16").strip().lower()
    if raw_linear_mode not in VALID_PREFILL_LINEAR_MODES:
        raise ValueError(
            f"{eff_engine_id}.prefill_linear_mode must be one of {sorted(VALID_PREFILL_LINEAR_MODES)}"
        )
    raw_attn_mode = str(opts.get("prefill_attention_mode") or "exact_nax").strip().lower()
    if raw_attn_mode not in VALID_PREFILL_ATTENTION_MODES:
        raise ValueError(
            f"{eff_engine_id}.prefill_attention_mode must be one of {sorted(VALID_PREFILL_ATTENTION_MODES)}"
        )
    prefill_sparse_minference = bool(
        _parse_optional_toggle(
            opts.get("prefill_sparse_minference"),
            f"{eff_engine_id}.prefill_sparse_minference",
        )
        or False
    )
    prefill_cache_blend = bool(
        _parse_optional_toggle(
            opts.get("prefill_cache_blend"),
            f"{eff_engine_id}.prefill_cache_blend",
        )
        or False
    )
    prefill_ane_offload = False
    for layer in (explicit_opts, model_toml, shared_toml):
        if layer.get("prefill_ane_offload") is not None:
            prefill_ane_offload = bool(
                _parse_optional_toggle(
                    layer.get("prefill_ane_offload"),
                    f"{eff_engine_id}.prefill_ane_offload",
                )
            )
            break
        if str(layer.get("prefill_engine") or "").strip().lower() == "nax_ane":
            prefill_ane_offload = True
            break

    return M5Options(
        max_context=max_context,
        max_concurrency=max_concurrency,
        kv_dtype=kv_dtype,
        prefill_engine=str(opts.get("prefill_engine") or "auto"),
        prefill_linear_mode=raw_linear_mode,
        prefill_attention_mode=raw_attn_mode,
        prefill_sparse_minference=prefill_sparse_minference,
        prefill_cache_blend=prefill_cache_blend,
        prefill_ane_offload=prefill_ane_offload,
        sampling_contract=str(opts.get("sampling_contract") or (
            M5RT_DEFAULT_SAMPLING_CONTRACT if is_miv else "legacy_rejection_v1"
        )),
        speculative_backend=spec_backend,
        speculative_policy=_requested_speculative_backend(
            model_settings, explicit_opts, model_toml, shared_toml
        ),
        spec_search=str(opts["spec_search"]) if opts.get("spec_search") is not None else None,
        draft_tokens=draft_tokens,
        concurrency_source=concurrency_source,
        lora_adapter=lora_adapter,
        lora_mode=lora_mode,
        prefix_cache_bytes=prefix_bytes,
        rolling_prefix_cache=rolling_prefix_cache,
        batch_mode=batch_mode,
        batch_projection_prototype=bool(opts.get("batch_projection_prototype", False)),
        path_overrides=overrides,
        fake_options=dict(opts.get("fake_options") or {}),
        memory_budget_bytes=max(0, int(memory_budget_bytes)),
        trust_remote_code=bool(getattr(model_settings, "trust_remote_code", False)),
    )


def create_m5rt_pool_engine(
    entry: Any,
    m5_target: M5Target,
    model_settings: object | None,
    global_settings: Any,
    *,
    sizing_context: int | None,
    admission_limit: int,
    current_model_memory: int,
    engine_id: str | None = None,
    provider_id: str | None = None,
    manifest: dict[str, Any] | None = None,
    lora_adapters: dict[str, str] | None = None,
    model_dirs: Iterable[str | Path] = (),
) -> M5SidecarEngine | None:
    """Construct `M5SidecarEngine` for `EnginePool`, or None when memory budget <= 0."""
    eff_manifest = manifest if manifest is not None else m5_target.binary.provider_manifest
    eff_provider_id = provider_id or m5_target.binary.provider_id
    eff_engine_id = engine_id or m5_target.binary.engine_id
    m5_options = build_m5rt_options(
        entry.model_path,
        m5_target,
        model_settings,
        default_context=sizing_context,
        memory_budget_bytes=0,
        global_settings=global_settings,
        model_id=entry.model_id,
        engine_id=eff_engine_id,
        provider_id=eff_provider_id,
        manifest=eff_manifest,
        lora_adapters=lora_adapters,
        model_dirs=model_dirs,
    )
    wanted = m5rt_memory_estimate(
        entry.model_path, m5_target, sizing_context, plan=m5_options
    )
    if admission_limit > 0:
        room = admission_limit - current_model_memory
        if room <= 0:
            return None
        budget = min(room, wanted)
    else:
        budget = wanted
    m5_options = replace(m5_options, memory_budget_bytes=budget)
    logger.info(
        "Serving %s through %s %s (%s)",
        entry.model_id,
        m5_target.binary.label,
        m5_target.binary.version,
        m5_target.binary.path,
    )
    safe_eng_dir = re.sub(r"[^A-Za-z0-9._-]+", "_", eff_engine_id).strip("_") or M5RT_ENGINE_ID
    log_dir = global_settings.logging.get_log_dir(global_settings.base_path) / safe_eng_dir
    return M5SidecarEngine(
        model_name=entry.model_path,
        model_id=entry.model_id,
        target=m5_target,
        log_dir=log_dir,
        options=m5_options,
        enable_thinking=getattr(model_settings, "enable_thinking", None),
        engine_id=eff_engine_id,
        provider_id=eff_provider_id,
        manifest=eff_manifest,
    )


def _spi_provider_runtime_signature(
    engine_id: str,
    provider_id: str,
    binary_path: Path,
    manifest: dict[str, Any],
    model_settings: object | None,
    global_settings: Any | None = None,
    *,
    lora_adapters: dict[str, str] | None = None,
    model_path: str | Path | None = None,
    model_dirs: Iterable[str | Path] = (),
) -> tuple[Any, ...]:
    """Canonical tuple of load-time settings for a Path B SPI provider plugin."""
    global_conc = (
        getattr(getattr(global_settings, "scheduler", None), "max_concurrent_requests", None)
        if global_settings is not None
        else None
    )
    try:
        bin_mtime = binary_path.stat().st_mtime_ns
    except OSError:
        bin_mtime = 0
    if model_settings is None:
        return (engine_id, provider_id, str(binary_path), bin_mtime, None, None, global_conc)

    ext_opts = getattr(model_settings, "external_engine_options", None)
    opt_keys = _provider_option_keys(engine_id, provider_id, manifest)
    scoped_opts: list[tuple[str, tuple[tuple[str, str], ...]]] = []
    if isinstance(ext_opts, dict):
        for key in opt_keys:
            nested = ext_opts.get(key)
            if isinstance(nested, dict) and nested:
                scoped_opts.append(
                    (key, tuple(sorted((str(k), repr(v)) for k, v in nested.items())))
                )
    canonical_opts = tuple(scoped_opts) if scoped_opts else None

    supported_std = tuple(
        str(s)
        for s in (manifest.get("supported_standard_settings") or ())
        if isinstance(s, str) and s.strip()
    )
    std_items: list[tuple[str, Any]] = [
        (key, getattr(model_settings, key, None)) for key in supported_std
    ]
    if "lora_adapter" in supported_std:
        lora_ref = getattr(model_settings, "lora_adapter", None) or None
        if lora_ref:
            std_items.append(
                (
                    "lora_adapter_fingerprint",
                    _safe_lora_fingerprint(
                        lora_ref,
                        lora_adapters=lora_adapters,
                        model_path=model_path,
                        model_dirs=model_dirs,
                    ),
                )
            )
    synced_std = tuple(std_items)
    return (
        engine_id,
        provider_id,
        str(binary_path),
        bin_mtime,
        canonical_opts,
        synced_std or None,
        global_conc,
    )


def _spi_provider_status(
    engine_id: str,
    provider_id: str,
    label: str,
    binary_path: Path,
    manifest: dict[str, Any],
    global_settings: Any,
    engine_pool: Any | None = None,
) -> dict[str, Any]:
    """Admin global-settings status payload for a discovered Path B SPI provider."""
    del global_settings
    proto_raw = manifest.get("protocol_version")
    fallback_proto = (
        proto_raw if isinstance(proto_raw, int) and not isinstance(proto_raw, bool) else None
    )
    binary, reason = inspect_m5rt_binary(
        binary_path,
        auto_detected=True,
        provider_manifest=manifest,
        allow_generic=True,
        fallback_protocol=fallback_proto,
    )
    models = []
    entries = getattr(engine_pool, "_entries", None) or {}
    for model_id, entry in list(entries.items()):
        engine = getattr(entry, "engine", None)
        if not isinstance(engine, M5SidecarEngine) or engine.external_engine_id != engine_id:
            continue
        stats = engine.get_stats()
        models.append(
            {
                "id": model_id,
                "memory_bytes": stats["process_memory_bytes"],
                "memory_budget_bytes": stats["memory_budget_bytes"],
                "lora_adapter": stats.get("lora_adapter", ""),
                "lora_mode": stats.get("lora_mode", "merge"),
                "lora_buffer_bytes": stats.get("lora_buffer_bytes", 0),
                "log_path": stats["log_path"],
            }
        )
    supported_std = [
        str(s)
        for s in (manifest.get("supported_standard_settings") or ())
        if isinstance(s, str) and s.strip()
    ]
    kv_formats = [
        str(k)
        for k in (manifest.get("supported_kv_formats") or manifest.get("kv_formats") or ())
        if isinstance(k, str) and k.strip()
    ]
    return {
        "available": binary is not None,
        "installed": binary is not None,
        "provider_id": provider_id,
        "engine_id": engine_id,
        "display_name": label,
        "path_tier": "Path B",
        "codename": manifest.get("codename"),
        "protocol_version": binary.protocol_version if binary else fallback_proto,
        "path": str(binary.path) if binary else str(binary_path),
        "binary_path": str(binary.path) if binary else str(binary_path),
        "auto_detected": True,
        "version": (
            binary.version
            if binary
            else str(
                manifest.get("display_version")
                or manifest.get("version")
                or manifest.get("semver")
                or ""
            )
            or None
        ),
        "label": label,
        "capabilities": list(manifest.get("capabilities") or ()),
        "supported_topologies": list(manifest.get("supported_topologies") or ()),
        "supported_model_types": list(manifest.get("supported_model_types") or ()),
        "artifact_extensions": list(_normalize_extensions(manifest.get("artifact_extensions"))),
        "kv_formats": kv_formats,
        "supported_standard_settings": supported_std,
        "options_schema": [
            dict(s) for s in (manifest.get("options_schema") or ()) if isinstance(s, dict)
        ],
        "reason": reason,
        "models": models,
    }


def _build_spi_provider_plugin(
    engine_id: str,
    provider_id: str,
    binary_path: Path,
    manifest: dict[str, Any],
) -> ExternalEnginePlugin:
    """Wrap an installed `Providers/<provider_id>/current/provider.json` bundle as an `ExternalEnginePlugin`."""
    label = str(
        manifest.get("display_name") or manifest.get("label") or engine_id
    ).strip()
    supported_standard = tuple(
        str(s).strip()
        for s in (manifest.get("supported_standard_settings") or ())
        if isinstance(s, str) and s.strip()
    )
    options_schema = tuple(
        dict(s)
        for s in (manifest.get("options_schema") or ())
        if isinstance(s, dict)
    )
    manifest_copy = dict(manifest)

    def _detect(model_dir: str | Path, model_settings: object | None = None) -> ProbeResult:
        return _probe_provider_adapted(model_dir, manifest_copy, model_settings)

    def _validate(
        model_path: str | Path, model_settings: object | None, global_settings: Any
    ) -> tuple[bool, str]:
        del global_settings
        target, reason = resolve_spi_provider_target(
            model_path, binary_path, manifest_copy, model_settings
        )
        return target is not None, reason

    def _status(global_settings: Any, engine_pool: Any | None = None) -> dict[str, Any]:
        return _spi_provider_status(
            engine_id,
            provider_id,
            label,
            binary_path,
            manifest_copy,
            global_settings,
            engine_pool,
        )

    def _estimate_memory(
        entry: Any,
        model_settings: object | None,
        global_settings: Any,
        *,
        sizing_context: int | None = None,
        lora_adapters: dict[str, str] | None = None,
        model_dirs: Iterable[str | Path] = (),
    ) -> int | None:
        target, _ = resolve_spi_provider_target(
            entry.model_path, binary_path, manifest_copy, model_settings
        )
        if target is None:
            return None
        return m5rt_memory_estimate(
            entry.model_path,
            target,
            sizing_context,
            model_settings,
            global_settings=global_settings,
            engine_id=engine_id,
            provider_id=provider_id,
            manifest=manifest_copy,
            lora_adapters=lora_adapters,
            model_dirs=model_dirs,
            model_id=entry.model_id,
        )

    def _runtime_sig(
        model_settings: object | None,
        global_settings: Any,
        *,
        lora_adapters: dict[str, str] | None = None,
        model_path: str | Path | None = None,
        model_dirs: Iterable[str | Path] = (),
    ) -> tuple[Any, ...]:
        return _spi_provider_runtime_signature(
            engine_id,
            provider_id,
            binary_path,
            manifest_copy,
            model_settings,
            global_settings,
            lora_adapters=lora_adapters,
            model_path=model_path,
            model_dirs=model_dirs,
        )

    def _factory(
        entry: Any,
        model_settings: object | None,
        global_settings: Any,
        scheduler_config: Any,
        *,
        sizing_context: int | None = None,
        admission_limit: int = 0,
        current_model_memory: int = 0,
        lora_adapters: dict[str, str] | None = None,
        model_dirs: Iterable[str | Path] = (),
    ) -> M5SidecarEngine | None:
        del scheduler_config
        target, reason = resolve_spi_provider_target(
            entry.model_path, binary_path, manifest_copy, model_settings
        )
        if target is None:
            raise RuntimeError(f"{label} cannot serve '{entry.model_id}': {reason}")
        return create_m5rt_pool_engine(
            entry,
            target,
            model_settings,
            global_settings,
            sizing_context=sizing_context,
            admission_limit=admission_limit,
            current_model_memory=current_model_memory,
            engine_id=engine_id,
            provider_id=provider_id,
            manifest=manifest_copy,
            lora_adapters=lora_adapters,
            model_dirs=model_dirs,
        )

    def _can_auto_convert_fast(
        model_path: str | Path,
        model_settings: object | None = None,
    ) -> bool:
        return can_auto_convert_provider_fast(
            model_path,
            binary_path=binary_path,
            manifest=manifest_copy,
            model_settings=model_settings,
            is_builtin=False,
        )

    def _can_convert(
        model_path: str | Path,
        model_settings: object | None = None,
    ) -> tuple[bool, str]:
        return can_convert_provider_model(
            model_path,
            binary_path_hint=binary_path,
            manifest=manifest_copy,
            model_settings=model_settings,
            is_builtin=False,
        )

    def _convert_model(
        model_path: str | Path,
        *,
        source_repo: str | None = None,
        model_settings: object | None = None,
    ) -> dict[str, Any]:
        return convert_provider_model(
            model_path,
            source_repo=source_repo,
            binary_path_hint=binary_path,
            manifest=manifest_copy,
            model_settings=model_settings,
            is_builtin=False,
        )

    return ExternalEnginePlugin(
        id=engine_id,
        label=label,
        factory=_factory,
        validator=_validate,
        status=_status,
        estimate_memory_bytes=_estimate_memory,
        runtime_signature=_runtime_sig,
        supported_standard_settings=supported_standard,
        detect_adapted=_detect,
        can_auto_convert_fast=_can_auto_convert_fast,
        can_convert=_can_convert,
        convert_model=_convert_model,
        options_schema=options_schema,
        provider_id=provider_id,
    )


def _spi_provider_manifest_for_engine(engine_id: str) -> dict[str, Any] | None:
    """Locate a Path B provider manifest by `engine_id` (O(1) JSON read, no subprocess)."""
    if not engine_id or engine_id in EXTERNAL_ENGINES:
        return None
    for base in _provider_search_bases():
        if not base.is_dir():
            continue
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        for entry in children:
            if not entry.is_dir() or entry.name in M5RT_BUILTIN_PROVIDER_IDS:
                continue
            for cand_root in (entry / "current", entry):
                if not cand_root.is_dir():
                    continue
                parsed = _read_provider_manifest(cand_root)
                if parsed is None:
                    continue
                pid = str(parsed.get("provider_id") or entry.name).strip()
                eid = str(parsed.get("engine_id") or pid).strip()
                legacy_ids = [
                    str(x).strip()
                    for x in (parsed.get("legacy_provider_ids") or ())
                    if isinstance(x, str) and x.strip()
                ]
                if engine_id in (eid, pid, *legacy_ids):
                    return parsed
                break
    return None


def external_engine_mtp_compatibility(
    external_engine: str,
    model_info: dict[str, Any],
    *,
    model_settings: object | None = None,
) -> tuple[bool, str]:
    """Resolve Admin API MTP compatibility for an active external engine."""
    if "mtp_enabled" not in get_external_engine_supported_settings(external_engine):
        return False, "MTP is not used when an external engine serves this model"

    model_path = model_info.get("model_path") or ""
    if not model_path:
        return False, "model_path missing"
    mdir = Path(str(model_path)).expanduser()
    cfg_path = mdir / "config.json"
    if not cfg_path.is_file():
        return False, "config.json not found"
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, f"failed to read config: {exc}"
    if not isinstance(cfg, dict):
        return False, "config.json is not a JSON object"

    model_types = _extract_model_type_tokens(cfg)
    raw_info_type = model_info.get("config_model_type")
    if isinstance(raw_info_type, str) and raw_info_type.strip():
        norm = raw_info_type.strip().lower()
        model_types.add(norm)
        if norm.endswith("_text") and len(norm) > 5:
            model_types.add(norm[:-5])
    raw_model_type = cfg.get("model_type") or raw_info_type or ""

    if external_engine == M5RT_ENGINE_ID:
        cfg_blocker = _config_structural_blocker(cfg)
        if cfg_blocker:
            return False, cfg_blocker

        opts = _merged_m5rt_options(mdir, model_settings)
        raw_mtp_art = (
            opts.get("mtp_artifact_path")
            or opts.get("mtp_artifact")
            or (
                opts["path_overrides"].get("mtp_artifact")
                if isinstance(opts.get("path_overrides"), dict)
                else None
            )
        )
        if isinstance(raw_mtp_art, (str, Path)) and str(raw_mtp_art).strip():
            if Path(str(raw_mtp_art)).expanduser().is_file():
                return True, ""
        if (mdir / "mtp.miv").is_file() or (mdir / "mtp.omlxm5").is_file():
            return True, ""

        art_path: Path | None = None
        explicit_art = (
            getattr(model_settings, "m5rt_artifact_path", None)
            or opts.get("artifact_path")
        )
        if explicit_art:
            cand = Path(str(explicit_art)).expanduser()
            if cand.is_file():
                art_path = cand
        if art_path is None:
            probe = detect_adapted(mdir, model_settings)
            if probe.adapted and probe.artifact_path is not None:
                art_path = probe.artifact_path
            elif (mdir / "model.miv").is_file():
                art_path = mdir / "model.miv"
            elif (mdir / "model.omlxm5").is_file():
                art_path = mdir / "model.omlxm5"

        embedded_mtp = inspect_omlxm5_has_mtp_weights(art_path)
        if embedded_mtp is True:
            return True, ""
        if embedded_mtp is False and art_path is not None:
            return False, (
                f"{M5RT_ENGINE_LABEL} artifact '{art_path.name}' contains no embedded MTP "
                "weights and no companion mtp.miv / mtp.omlxm5 was found."
            )
        if _config_json_has_mtp(cfg):
            return True, ""
        return False, "model has no MTP heads in config and no companion mtp.miv / mtp.omlxm5"

    # Path B provider that explicitly declared "mtp_enabled"
    manifest = _spi_provider_manifest_for_engine(external_engine) or {}
    supported_types = tuple(
        str(m).strip().lower()
        for m in (manifest.get("supported_model_types") or ())
        if isinstance(m, str) and m.strip()
    )
    if supported_types and model_types and not (model_types & set(supported_types)):
        return False, (
            f"model_type={raw_model_type!r} is not supported by '{external_engine}' MTP "
            f"(supported: {', '.join(supported_types)})"
        )
    if (
        (mdir / "mtp.miv").is_file()
        or (mdir / "mtp.omlxm5").is_file()
        or _config_json_has_mtp(cfg)
    ):
        return True, ""
    return False, "model has no MTP heads in config and no companion mtp.miv / mtp.omlxm5"


def external_engine_dflash_compatibility(
    external_engine: str,
    model_info: dict[str, Any],
) -> tuple[bool, str]:
    """Resolve Admin API DFlash compatibility for an active external engine."""
    if "dflash_enabled" not in get_external_engine_supported_settings(external_engine):
        return False, "DFlash is not used when an external engine serves this model"

    engine_type = model_info.get("engine_type")
    if engine_type and engine_type not in ("batched", "vlm"):
        return False, f"DFlash is only supported on LLM/VLM models (engine='{engine_type}')"

    model_types: set[str] = set()
    raw_info_type = model_info.get("config_model_type")
    if isinstance(raw_info_type, str) and raw_info_type.strip():
        norm = raw_info_type.strip().lower()
        model_types.add(norm)
        if norm.endswith("_text") and len(norm) > 5:
            model_types.add(norm[:-5])
    model_path = model_info.get("model_path") or ""
    cfg: dict[str, Any] | None = None
    if model_path:
        cfg = _read_json(Path(str(model_path)).expanduser() / "config.json")
        if cfg is not None:
            model_types.update(_extract_model_type_tokens(cfg))
    raw_model_type = raw_info_type or (cfg.get("model_type") if isinstance(cfg, dict) else "") or ""

    if external_engine == M5RT_ENGINE_ID:
        cfg_blocker = _config_structural_blocker(cfg)
        if cfg_blocker:
            return False, cfg_blocker
        if cfg is not None:
            return True, ""
        manifest = _builtin_m5rt_manifest()
        supported_types = tuple(
            str(m).strip().lower()
            for m in (manifest.get("supported_model_types") or M5RT_SUPPORTED_MODEL_TYPES)
            if isinstance(m, str) and m.strip()
        )
        if model_types & set(supported_types):
            return True, ""
        return False, (
            f"DFlash on {M5RT_ENGINE_LABEL} requires a valid language model config "
            f"(model_type='{raw_model_type}', engine='{engine_type}')"
        )

    manifest = _spi_provider_manifest_for_engine(external_engine) or {}
    supported_types = tuple(
        str(m).strip().lower()
        for m in (manifest.get("supported_model_types") or ())
        if isinstance(m, str) and m.strip()
    )
    if supported_types and model_types and not (model_types & set(supported_types)):
        return False, (
            f"DFlash on '{external_engine}' does not support model_type='{raw_model_type}'"
        )
    return True, ""


def _discover_spi_provider_plugins() -> dict[str, ExternalEnginePlugin]:
    """Scan `Providers/*/current/provider.json` (fallback `*/provider.json`) for Path B SPI providers.

    Skips built-in `io.ninfer.metal` and legacy `com.omlx.mseries`, validates
    `protocol_version in SUPPORTED_PROTOCOLS` and an executable `entrypoint`/`binary`
    using fast O(1) filesystem checks (zero subprocess spawn).
    """
    discovered: dict[str, ExternalEnginePlugin] = {}
    for base in _provider_search_bases():
        if not base.is_dir():
            continue
        try:
            children = sorted(base.iterdir(), key=lambda p: p.name)
        except OSError:
            continue

        for entry in children:
            if not entry.is_dir():
                continue
            if entry.name in M5RT_BUILTIN_PROVIDER_IDS:
                continue

            manifest: dict[str, Any] | None = None
            provider_root: Path | None = None
            for cand_root in (entry / "current", entry):
                if not cand_root.is_dir():
                    continue
                parsed = _read_provider_manifest(cand_root)
                if parsed is not None:
                    manifest = parsed
                    provider_root = cand_root
                    break
            if manifest is None or provider_root is None:
                continue

            provider_id = str(manifest.get("provider_id") or entry.name).strip()
            if not provider_id or provider_id in M5RT_BUILTIN_PROVIDER_IDS:
                continue
            engine_id = str(manifest.get("engine_id") or provider_id).strip()
            if not engine_id or engine_id in EXTERNAL_ENGINES:
                logger.warning(
                    "SPI provider %r at %s collides with built-in engine %r; skipping",
                    provider_id,
                    provider_root,
                    engine_id,
                )
                continue

            proto_raw = manifest.get("protocol_version")
            if (
                isinstance(proto_raw, bool)
                or not isinstance(proto_raw, int)
                or proto_raw not in SUPPORTED_PROTOCOLS
            ):
                logger.warning(
                    "SPI provider %r at %s has unsupported protocol_version=%r; skipping",
                    provider_id,
                    provider_root,
                    proto_raw,
                )
                continue

            binary_path: Path | None = None
            for key in ("entrypoint", "binary", "daemon_path"):
                val = manifest.get(key)
                if isinstance(val, str) and val.strip():
                    cand = Path(val.strip())
                    if not cand.is_absolute():
                        cand = provider_root / cand
                    if _executable(cand):
                        binary_path = cand.resolve()
                        break
            if binary_path is None:
                logger.warning(
                    "SPI provider %r at %s has no executable entrypoint/binary; skipping",
                    provider_id,
                    provider_root,
                )
                continue

            if engine_id in discovered:
                logger.warning(
                    "Duplicate SPI provider engine_id %r at %s; keeping first registration",
                    engine_id,
                    provider_root,
                )
                continue
            discovered[engine_id] = _build_spi_provider_plugin(
                engine_id, provider_id, binary_path, manifest
            )
    return discovered


# -- Fallback tokenizer & O(1) incremental detokenizer + stop matcher ----------


class _ByteFallbackTokenizer:
    """Deterministic UTF-8 byte-level tokenizer for raw `.omlxm5` C++ fixtures
    (such as 2-layer `qwen-execution/affine/source`) that ship `config.json`
    without HuggingFace `tokenizer.json` files.
    """

    def __init__(self, vocab_size: int = 257, eos_token_id: int | None = None) -> None:
        self.vocab_size = max(16, int(vocab_size))
        self.eos_token = "<|im_end|>"
        self.eos_token_id = (
            int(eos_token_id)
            if isinstance(eos_token_id, int) and 0 <= eos_token_id < self.vocab_size
            else (self.vocab_size - 1)
        )
        self.eos_token_ids = {self.eos_token_id}

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        del add_special_tokens
        if not text:
            return [1]
        data = text.encode("utf-8", errors="replace")
        modulus = max(1, min(256, self.vocab_size - 1))
        return [b % modulus for b in data] or [1]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        raw = bytearray()
        for tok in token_ids:
            tid = int(tok)
            if skip_special_tokens and tid == self.eos_token_id:
                continue
            if tid == self.eos_token_id:
                raw.extend(b"<|im_end|>")
            elif 0 <= tid < 256:
                # Printable ASCII stays readable; non-ASCII bytes preserve UTF-8 streams.
                raw.append(tid)
            else:
                raw.extend(f"<{tid}>".encode("utf-8"))
        return raw.decode("utf-8", errors="replace")

    def convert_tokens_to_ids(self, token: str) -> int | None:
        if token in ("<|im_end|>", "<|endoftext|>"):
            return self.eos_token_id
        return None

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        continue_final_message: bool = False,
        **_kwargs: Any,
    ) -> str | list[int]:
        parts: list[str] = []
        if tools:
            parts.append(f"<|im_start|>system\nTools: {json.dumps(tools)}<|im_end|>\n")
        for idx, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(p.get("text", "")) if isinstance(p, dict) else str(p)
                    for p in content
                )
            is_last = idx == len(messages) - 1
            if is_last and continue_final_message:
                parts.append(f"<|im_start|>{role}\n{content}")
            else:
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
        if add_generation_prompt and not continue_final_message:
            parts.append("<|im_start|>assistant\n")
        rendered = "".join(parts)
        return self.encode(rendered) if tokenize else rendered


class _IncrementalDetokenizer:
    """O(1)-per-token streaming detokenizer with incomplete UTF-8 (`\\ufffd`) holdback.

    Uses oMLX's cached `mlx_lm` BPE/SPM streaming detokenizer when available,
    falling back to a bounded 6-token context window decoder so each step never
    re-decodes the full generated token history.
    """

    _CONTEXT_WINDOW = 6
    _MAX_UTF8_HOLD = 8

    def __init__(self, tokenizer: Any, model_path: Path, special_ids: set[int]) -> None:
        self._tokenizer = tokenizer
        self._special_ids = special_ids
        self._streaming = None
        if not isinstance(tokenizer, _ByteFallbackTokenizer):
            try:
                cand = create_streaming_detokenizer(tokenizer, model_path)
                # Use mlx_lm's O(1) BPE/SPM detokenizers; avoid O(N) naive full-buffer ones.
                if cand is not None and "Naive" not in type(cand).__name__:
                    self._streaming = cand
            except Exception:
                self._streaming = None

        self._hold_text = ""
        self._prefix_tokens: list[int] = []
        self._prefix_text = ""
        self._pending_tokens: list[int] = []

    def add_tokens(self, tokens: Iterable[int]) -> str:
        chunks: list[str] = []
        for raw_tok in tokens:
            tok = int(raw_tok)
            if tok in self._special_ids:
                continue
            piece = self._step(tok)
            if piece:
                chunks.append(piece)
        return "".join(chunks)

    def _decode_slice(self, token_ids: list[int]) -> str:
        if not token_ids:
            return ""
        try:
            return self._tokenizer.decode(
                token_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return self._tokenizer.decode(token_ids)

    def _step(self, tok: int) -> str:
        if self._streaming is not None:
            self._streaming.add_token(tok)
            seg = self._hold_text + self._streaming.last_segment
            if seg.endswith("\ufffd") and len(self._hold_text) < self._MAX_UTF8_HOLD:
                self._hold_text = seg
                return ""
            self._hold_text = ""
            return seg

        self._pending_tokens.append(tok)
        window = self._prefix_tokens + self._pending_tokens
        decoded = self._decode_slice(window)
        if decoded.startswith(self._prefix_text):
            delta = decoded[len(self._prefix_text) :]
        else:
            # Rare multi-token boundary normalization: recompute prefix cleanly.
            delta = self._decode_slice(self._pending_tokens)

        if delta.endswith("\ufffd") and len(self._pending_tokens) < self._MAX_UTF8_HOLD:
            return ""

        self._prefix_tokens = window[-self._CONTEXT_WINDOW :]
        self._prefix_text = self._decode_slice(self._prefix_tokens)
        self._pending_tokens.clear()
        return delta

    def finalize(self) -> str:
        if self._streaming is not None:
            try:
                self._streaming.finalize()
                seg = self._hold_text + self._streaming.last_segment
            except Exception:
                seg = self._hold_text
            self._hold_text = ""
            return seg

        if not self._pending_tokens:
            return ""
        window = self._prefix_tokens + self._pending_tokens
        decoded = self._decode_slice(window)
        delta = (
            decoded[len(self._prefix_text) :]
            if decoded.startswith(self._prefix_text)
            else self._decode_slice(self._pending_tokens)
        )
        self._pending_tokens.clear()
        return delta


class _StopMatcher:
    """O(1)-per-step matcher for both `stop_token_seqs` and string `stop` sequences."""

    def __init__(
        self,
        stop_token_seqs: list[list[int]],
        stop_strings: list[str],
        *,
        skip_reasoning: bool = False,
        prompt_opens_think: bool = False,
        think_open: str = "<think>",
        think_close: str = "</think>",
    ) -> None:
        self._think_open = think_open or "<think>"
        self._think_close = think_close or "</think>"
        self._think_state = "answer"
        if skip_reasoning:
            self._think_state = "thinking" if prompt_opens_think else "leading"
        self._think_pending = ""
        self._token_seqs = [list(s) for s in stop_token_seqs if s]
        self._max_tok_seq = max((len(s) for s in self._token_seqs), default=0)
        self._recent_tokens: list[int] = []

        self._stop_strings = [s for s in stop_strings if s]
        self._max_str_len = max((len(s) for s in self._stop_strings), default=0)
        self._pending_text = ""
        self.stopped = False

    def filter_tokens(self, incoming: Iterable[int]) -> list[int]:
        """Truncate `incoming` before any matched `stop_token_seqs`."""
        if not self._token_seqs:
            return [int(t) for t in incoming]
        accepted: list[int] = []
        for raw in incoming:
            tok = int(raw)
            self._recent_tokens.append(tok)
            accepted.append(tok)
            for seq in self._token_seqs:
                n = len(seq)
                if len(self._recent_tokens) >= n and self._recent_tokens[-n:] == seq:
                    # Strip matched stop token(s) that were part of this batch.
                    trim = min(len(accepted), n)
                    if trim:
                        del accepted[-trim:]
                    self.stopped = True
                    return accepted
            if len(self._recent_tokens) > self._max_tok_seq * 2:
                del self._recent_tokens[: -self._max_tok_seq]
        return accepted

    def push_text(self, piece: str) -> str:
        """Feed decoded text, holding back partial stop prefixes in O(1) time."""
        if not self._stop_strings:
            return piece
        if self._think_state == "answer":
            return self._push_answer(piece)

        buf = self._think_pending + piece
        self._think_pending = ""
        opened, closed = self._think_open, self._think_close
        prefix = ""
        if self._think_state == "leading":
            start = buf.lstrip()
            if not start:
                return buf
            if start.startswith(opened):
                self._think_state = "thinking"
                opening_end = len(buf) - len(start) + len(opened)
                prefix, buf = buf[:opening_end], buf[opening_end:]
            elif opened.startswith(start):
                self._think_pending = start
                return buf[:len(buf) - len(start)]
            else:
                self._think_state = "answer"
                return self._push_answer(buf)

        closing = buf.find(closed)
        if closing >= 0:
            end = closing + len(closed)
            self._think_state = "answer"
            return prefix + buf[:end] + self._push_answer(buf[end:])
        for n in range(min(len(closed) - 1, len(buf)), 0, -1):
            if closed.startswith(buf[-n:]):
                self._think_pending = buf[-n:]
                return prefix + buf[:-n]
        return prefix + buf

    def _push_answer(self, piece: str) -> str:
        buf = self._pending_text + piece
        earliest_idx: int | None = None
        for s in self._stop_strings:
            pos = buf.find(s)
            if pos != -1 and (earliest_idx is None or pos < earliest_idx):
                earliest_idx = pos
        if earliest_idx is not None:
            self.stopped = True
            self._pending_text = ""
            return buf[:earliest_idx]

        # Hold back the longest suffix of `buf` that is a proper prefix of any stop string.
        hold = 0
        max_check = min(len(buf), self._max_str_len - 1)
        for k in range(max_check, 0, -1):
            suffix = buf[-k:]
            if any(s.startswith(suffix) for s in self._stop_strings):
                hold = k
                break
        if hold:
            emit = buf[:-hold]
            self._pending_text = buf[-hold:]
            return emit
        self._pending_text = ""
        return buf

    def flush(self) -> str:
        rem = "" if self.stopped else self._think_pending + self._pending_text
        self._think_pending = self._pending_text = ""
        return rem


# -- Process management & live request state -----------------------------------


def _watchdog_command(command: list[str]) -> list[str]:
    """Wrap `command` in a process-group leader shell watching parent PID."""
    script = (
        'parent=$1; shift; "$@" & child=$!; '
        'while kill -0 "$parent" 2>/dev/null && kill -0 "$child" 2>/dev/null; '
        f"do sleep {WATCHDOG_POLL_SECONDS}; done; "
        'if kill -0 "$child" 2>/dev/null; then trap "" TERM; kill -TERM 0; fi; '
        'wait "$child"'
    )
    return ["/bin/sh", "-c", script, "m5rt-watchdog", str(os.getpid()), *command]


def _signal_group(group: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(group, sig)


def _group_alive(group: int) -> bool:
    try:
        os.killpg(group, 0)
    except (ProcessLookupError, PermissionError):
        return False
    return True


async def _wait_group_exit(
    group: int, timeout: float, resend: int | None = None
) -> bool:
    deadline = time.monotonic() + timeout
    while _group_alive(group):
        if time.monotonic() >= deadline:
            return False
        await asyncio.sleep(STOP_POLL_SECONDS)
        if resend is not None:
            _signal_group(group, resend)
    return True


def _short_socket_path(engine_id: str = M5RT_ENGINE_ID) -> Path:
    """Return a short private AF_UNIX path well within macOS's 104-byte sun_path limit."""
    if not engine_id or engine_id == M5RT_ENGINE_ID:
        slug = M5RT_ENGINE_ID
        prefix = "m5"
    else:
        slug = re.sub(r"[^a-z0-9_-]+", "-", engine_id.lower()).strip("-")[:16] or "ext"
        prefix = slug[:8] or "ext"
    base = Path("/tmp") / f"omlx-{slug}-{os.getuid()}"
    base.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        os.chmod(base, 0o700)
    return base / f"{prefix}-{uuid.uuid4().hex[:12]}.sock"


def _failure_summary(log_path: Path, offset: int) -> str:
    try:
        with open(log_path, "rb") as fh:
            fh.seek(offset)
            lines = fh.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    lines = [ln.strip() for ln in lines if ln.strip()]
    return "\n".join(lines[-LOG_TAIL_LINES:])


@dataclass(frozen=True)
class _SidecarEvent:
    request_id: int
    kind: str
    tokens: tuple[int, ...] = ()
    finish_reason: str = ""
    error: str = ""
    request_stats: dict[str, Any] | None = None


@dataclass
class _LiveRequest:
    """In-flight M5 sidecar request telemetry for admin dashboard."""

    started_at: float
    prompt_tokens: int
    max_tokens: int | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    generated_tokens: int = 0
    decode_marks: DecodeMarks = ()

    def add_tokens(self, count: int, now: float) -> None:
        self.decode_marks = add_decode_mark(
            self.decode_marks, self.generated_tokens, self.last_token_at, now
        )
        self.first_token_at = self.first_token_at or now
        self.last_token_at = now
        self.generated_tokens += count

    def prefill_row(
        self, request_id: str, now: float, *, detail: str = M5RT_ENGINE_ID
    ) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "processed": 0,
            "total": self.prompt_tokens,
            "speed": 0.0,
            "eta": None,
            "elapsed": max(0.0, now - self.started_at),
            "detail": detail,
        }

    def generating_row(self, request_id: str, now: float) -> dict[str, Any]:
        first = self.first_token_at or now
        last = self.last_token_at or now
        return {
            "request_id": request_id,
            "elapsed_seconds": max(0.0, now - first),
            "generated_tokens": self.generated_tokens,
            "tokens_per_second": live_decode_rate(
                self.decode_marks,
                self.generated_tokens,
                last,
                first,
                now,
            ),
            "last_activity_age_seconds": max(0.0, now - last),
            "prompt_tokens": self.prompt_tokens,
            "max_tokens": self.max_tokens,
        }


# -- M5SidecarEngine -----------------------------------------------------------


class M5SidecarEngine(BaseEngine):
    """Serves a model through an out-of-process `mseries_backendd` UDS sidecar."""

    _prefill_memory_guard_managed_externally = True
    consumes_stock_cache = False
    external_engine_id = M5RT_ENGINE_ID
    supports_request_scoped_abort = True
    supports_native_response_format = False

    def __init__(
        self,
        model_name: str,
        model_id: str,
        target: M5Target,
        log_dir: Path,
        options: M5Options,
        *,
        enable_thinking: bool | None = None,
        engine_id: str | None = None,
        provider_id: str | None = None,
        manifest: dict[str, Any] | None = None,
    ) -> None:
        self._path = Path(model_name)
        self._model_id = model_id
        self._target = target
        self._options = options
        self._enable_thinking = enable_thinking

        resolved_manifest = (
            dict(manifest)
            if isinstance(manifest, dict)
            else (
                dict(target.binary.provider_manifest)
                if isinstance(target.binary.provider_manifest, dict)
                else None
            )
        )
        resolved_provider_id = str(
            provider_id
            or (resolved_manifest.get("provider_id") if resolved_manifest else None)
            or target.binary.provider_id
            or M5RT_PROVIDER_ID
        ).strip()
        resolved_engine_id = str(
            engine_id
            or (resolved_manifest.get("engine_id") if resolved_manifest else None)
            or (
                M5RT_ENGINE_ID
                if resolved_provider_id in M5RT_BUILTIN_PROVIDER_IDS
                else resolved_provider_id
            )
        ).strip()
        self.external_engine_id = resolved_engine_id or M5RT_ENGINE_ID
        self._provider_id = resolved_provider_id or M5RT_PROVIDER_ID
        self._provider_manifest = resolved_manifest
        self._is_builtin_m5rt = (
            self.external_engine_id == M5RT_ENGINE_ID
            and self._provider_id in M5RT_BUILTIN_PROVIDER_IDS
        )

        safe_slug = re.sub(r"[^A-Za-z0-9._-]+", "--", model_id).strip("-") or "model"
        self._log_path = log_dir / f"{safe_slug}.log"
        self._log_offset = 0
        self._socket_path: Path | None = None

        self._process: subprocess.Popen | None = None
        self._sidecar_pid: int | None = None
        self._ctrl_reader: asyncio.StreamReader | None = None
        self._ctrl_writer: asyncio.StreamWriter | None = None
        self._ctrl_lock = asyncio.Lock()
        self._next_rpc_id = 1

        self._event_reader: asyncio.StreamReader | None = None
        self._event_writer: asyncio.StreamWriter | None = None
        self._event_pump_task: asyncio.Task | None = None
        self._stats_task: asyncio.Task | None = None
        self._stats_refresh: asyncio.Task | None = None
        self._push_subscribed = False

        self._stopping = False
        self._runtime_failed_reason_override: str | None = None

        self._tokenizer: Any = None
        self._model_type: str | None = None
        self._vocab_size: int = 248320
        self._eos_token_ids: list[int] = []

        self._hello_info: dict[str, Any] = {}
        self._last_stats: dict[str, Any] | None = None
        self._warned_request_controls: set[str] = set()
        self._last_capacity: dict[str, Any] | None = None
        self._reported_footprint_bytes: int = 0
        self._reported_lora_buffer_bytes: int | None = None
        self._reported_lora_adapted_layers: int = 0
        self._weights_size: int | None = None
        self._lora_estimated_size: int | None = None
        self._last_gpu_contended_hint: bool | None = None
        self._trim_cache_unsupported: bool = False
        self._scheduler_hint_unsupported: bool = False

        self._active = 0
        self._requests = 0
        self._live: dict[str, _LiveRequest] = {}
        self._request_queues: dict[int, asyncio.Queue[_SidecarEvent]] = {}
        self._retired_request_ids: OrderedDict[int, None] = OrderedDict()
        self._client_to_backend_id: dict[str, int] = {}
        self._aborted_client_ids: OrderedDict[str, None] = OrderedDict()

    # -- Identity & external-engine contract -----------------------------------

    @property
    def model_name(self) -> str:
        return str(self._path)

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def model_type(self) -> str | None:
        return self._model_type

    @property
    def accepts_image_input(self) -> bool:
        return False

    @property
    def log_path(self) -> Path:
        return self._log_path

    @property
    def max_context_tokens(self) -> int | None:
        return self._options.max_context if self._options.max_context > 0 else None

    def active_request_count(self) -> int:
        return self._active

    def has_active_requests(self) -> bool:
        return self._active > 0

    @property
    def runtime_failed_reason(self) -> str | None:
        if self._runtime_failed_reason_override:
            return self._runtime_failed_reason_override
        if self._stopping:
            return None
        proc = self._process
        if proc is None or proc.poll() is None:
            return None
        summary = _failure_summary(self._log_path, self._log_offset)
        detail = f": {summary}" if summary else ""
        return (
            f"{self._target.binary.path.name} exited with code {proc.returncode}{detail} "
            f"(log: {self._log_path})"
        )

    # -- Tokenizer & EOS discovery ---------------------------------------------

    def _has_hf_tokenizer_files(self) -> bool:
        return any(
            (self._path / name).is_file()
            for name in (
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
                "vocab.json",
                "tekken.json",
            )
        )

    @staticmethod
    def _collect_int_ids(value: Any, out: set[int]) -> None:
        if isinstance(value, bool):
            return
        if isinstance(value, int) and value >= 0:
            out.add(value)
        elif isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                M5SidecarEngine._collect_int_ids(item, out)

    def _discover_eos_token_ids(self, config: dict[str, Any]) -> list[int]:
        eos_ids: set[int] = set()
        tok = self._tokenizer
        self._collect_int_ids(getattr(tok, "eos_token_id", None), eos_ids)
        self._collect_int_ids(getattr(tok, "eos_token_ids", None), eos_ids)
        self._collect_int_ids(config.get("eos_token_id"), eos_ids)
        self._collect_int_ids(_text_config(config).get("eos_token_id"), eos_ids)

        gen_cfg = _read_json(self._path / "generation_config.json") or {}
        self._collect_int_ids(gen_cfg.get("eos_token_id"), eos_ids)

        unk_id = getattr(tok, "unk_token_id", None)
        convert = getattr(tok, "convert_tokens_to_ids", None)
        if callable(convert):
            for special in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
                try:
                    tid = convert(special)
                except Exception:
                    continue
                if (
                    isinstance(tid, int)
                    and tid >= 0
                    and tid != unk_id
                    and tid < self._vocab_size
                ):
                    eos_ids.add(tid)
        return sorted(tid for tid in eos_ids if 0 <= tid < self._vocab_size)

    def _load_tokenizer(self) -> None:
        config = _read_json(self._path / "config.json") or {}
        text_cfg = _text_config(config)
        self._model_type = config.get("model_type") or text_cfg.get("model_type")
        if self._target.fake and isinstance(
            self._options.fake_options.get("vocab_size"), int
        ):
            self._vocab_size = max(2, int(self._options.fake_options["vocab_size"]))
        else:
            self._vocab_size = max(
                2, int(text_cfg.get("vocab_size") or config.get("vocab_size") or 248320)
            )

        if self._has_hf_tokenizer_files():
            tok_kwargs = get_tokenizer_config(
                str(self._path), trust_remote_code=self._options.trust_remote_code
            )
            self._tokenizer = AutoTokenizer.from_pretrained(
                str(self._path), **tok_kwargs
            )
            tok_len = len(self._tokenizer) if hasattr(self._tokenizer, "__len__") else 0
            if not text_cfg.get("vocab_size") and tok_len > 0:
                self._vocab_size = max(self._vocab_size, tok_len)
        elif self._target.fake or self._vocab_size <= 1024:
            cfg_eos = text_cfg.get("eos_token_id", config.get("eos_token_id"))
            first_eos = cfg_eos[0] if isinstance(cfg_eos, list) and cfg_eos else cfg_eos
            self._tokenizer = _ByteFallbackTokenizer(
                vocab_size=self._vocab_size,
                eos_token_id=first_eos if isinstance(first_eos, int) else None,
            )
        else:
            raise ModelLoadingError(
                self._model_id,
                f"{self._target.binary.label} cannot serve '{self._model_id}': "
                f"missing HuggingFace tokenizer files (tokenizer.json / "
                f"tokenizer_config.json / tokenizer.model / vocab.json) in {self._path}",
            )

        self._eos_token_ids = self._discover_eos_token_ids(config)

    # -- UDS transport & lifecycle ---------------------------------------------

    def _build_load_params(self) -> dict[str, Any]:
        opts = self._options
        if self._target.fake:
            fake_opts = {"vocab_size": self._vocab_size, **opts.fake_options}
            fake_payload: dict[str, Any] = {
                "fake": True,
                "max_concurrency": opts.max_concurrency,
                "max_context": opts.max_context,
                "kv_dtype": opts.kv_dtype,
                "prefill_engine": opts.prefill_engine,
                "prefill_linear_mode": opts.prefill_linear_mode,
                "prefill_attention_mode": opts.prefill_attention_mode,
                "prefill_sparse_minference": bool(opts.prefill_sparse_minference),
                "prefill_cache_blend": bool(opts.prefill_cache_blend),
                "prefill_ane_offload": bool(opts.prefill_ane_offload),
                "sampling_contract": "legacy_rejection_v1",
                "lora_adapter": opts.lora_adapter,
                "lora_mode": opts.lora_mode,
                "fake_options": fake_opts,
            }
            if opts.memory_budget_bytes > 0:
                fake_payload["memory_budget_bytes"] = int(opts.memory_budget_bytes)
            return fake_payload

        overrides = dict(opts.path_overrides)
        if self._target.artifact_path is not None:
            overrides.setdefault("artifact", str(self._target.artifact_path))
        if self._target.metallib_dir is not None:
            qwen_ar_lib = self._target.metallib_dir / "omlx_m5_qwen_ar.metallib"
            if self._is_builtin_m5rt or qwen_ar_lib.is_file():
                overrides.setdefault(
                    "qwen_ar_metallib",
                    str(qwen_ar_lib),
                )
            overrides.setdefault("kernel_directory", str(self._target.metallib_dir))
        eff_batch_mode: bool | None = opts.batch_mode
        if eff_batch_mode is None and "batch_mode" in overrides:
            raw_override_bm = overrides.get("batch_mode")
            if isinstance(raw_override_bm, str) and not raw_override_bm.strip():
                overrides.pop("batch_mode", None)
            else:
                eff_batch_mode = _parse_optional_toggle(
                    raw_override_bm, "m5rt_options.path_overrides.batch_mode"
                )
        if opts.max_concurrency < 2:
            if eff_batch_mode is True:
                logger.warning(
                    "Sidecar Runtime: batch_mode=on requires max_concurrency >= 2 (got %d); "
                    "disabling batch_mode",
                    opts.max_concurrency,
                )
                overrides.pop("batch_mode", None)
            elif eff_batch_mode is False:
                overrides["batch_mode"] = "off"
            else:
                overrides.pop("batch_mode", None)
        else:
            if eff_batch_mode is False:
                overrides["batch_mode"] = "off"
            else:
                overrides["batch_mode"] = "on"
        if opts.batch_projection_prototype:
            overrides["batch_projection_prototype"] = "on"
        overrides["prefix_cache_bytes"] = str(opts.prefix_cache_bytes)
        if opts.rolling_prefix_cache is not None:
            overrides["rolling_prefix_cache"] = (
                "on" if opts.rolling_prefix_cache else "off"
            )
        elif "rolling_prefix_cache" in overrides:
            parsed_override = _parse_optional_toggle(
                overrides["rolling_prefix_cache"],
                "m5rt_options.path_overrides.rolling_prefix_cache",
            )
            if parsed_override is not None:
                overrides["rolling_prefix_cache"] = "on" if parsed_override else "off"

        spec_backend = opts.speculative_backend
        draft_tokens = opts.draft_tokens
        if spec_backend == "dflash2":
            if self._target.draft_artifact_path is not None:
                overrides.setdefault(
                    "draft_artifact", str(self._target.draft_artifact_path)
                )
            if self._target.metallib_dir is not None:
                dflash_lib = (
                    self._target.metallib_dir / "omlx_m5_dflash2_program.metallib"
                )
                if dflash_lib.is_file():
                    overrides.setdefault("dflash2_metallib", str(dflash_lib))
        elif spec_backend != "mtp":
            draft_tokens = 0

        payload: dict[str, Any] = {
            "model_dir": str(self._path),
            "draft_dir": str(self._target.draft_dir) if self._target.draft_dir else "",
            **opts.to_rpc_dict(),
            "speculative": {
                "backend": spec_backend,
                "draft_tokens": draft_tokens,
            },
            "path_overrides": overrides,
        }
        if opts.spec_search is not None:
            payload["spec_search"] = opts.spec_search
        if opts.memory_budget_bytes > 0:
            payload["memory_budget_bytes"] = int(opts.memory_budget_bytes)
        if "rolling_prefix_cache" in overrides:
            payload["rolling_prefix_cache"] = overrides["rolling_prefix_cache"]
        return payload

    def _record_footprint_fields(self, payload: dict[str, Any]) -> None:
        pid = payload.get("pid")
        if isinstance(pid, int) and pid > 0:
            self._sidecar_pid = pid
        fp = payload.get("phys_footprint_bytes")
        if isinstance(fp, int) and fp >= 0:
            self._reported_footprint_bytes = fp
        lora_buf = payload.get("lora_buffer_bytes")
        if isinstance(lora_buf, int) and not isinstance(lora_buf, bool) and lora_buf >= 0:
            self._reported_lora_buffer_bytes = lora_buf
        lora_layers = payload.get("lora_adapted_layers")
        if isinstance(lora_layers, int) and not isinstance(lora_layers, bool) and lora_layers >= 0:
            self._reported_lora_adapted_layers = lora_layers

    def _mark_runtime_failed(self, detail: str) -> str:
        if self._stopping:
            return detail
        if not self._runtime_failed_reason_override:
            proc = self._process
            exit_info = (
                f" (exit={proc.returncode})"
                if proc is not None and proc.poll() is not None
                else ""
            )
            summary = _failure_summary(self._log_path, self._log_offset)
            tail = f": {summary}" if summary else ""
            self._runtime_failed_reason_override = (
                f"M5 sidecar failed{exit_info}: {detail}{tail} (log: {self._log_path})"
            )
        reason = self._runtime_failed_reason_override
        fail_event = _SidecarEvent(
            request_id=0,
            kind="failed",
            finish_reason="error",
            error=reason,
        )
        for queue in list(self._request_queues.values()):
            queue.put_nowait(fail_event)
        return reason

    async def _open_uds(self) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
        assert self._socket_path is not None
        deadline = time.monotonic() + CONNECT_TIMEOUT_SECONDS
        last_err: Exception | None = None
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise RuntimeError(
                    self.runtime_failed_reason
                    or f"mseries_backendd exited early ({self._process.returncode})"
                )
            try:
                return await asyncio.open_unix_connection(
                    str(self._socket_path), limit=UDS_MAX_LINE_BYTES
                )
            except (FileNotFoundError, ConnectionRefusedError, OSError) as exc:
                last_err = exc
                await asyncio.sleep(CONNECT_POLL_SECONDS)
        raise RuntimeError(
            f"Timed out connecting to mseries_backendd socket {self._socket_path}: {last_err}"
        )

    async def _rpc_on(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        req_id: int,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        wire = (
            json.dumps(
                {"id": req_id, "method": method, "params": params or {}},
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        writer.write(wire)
        await writer.drain()
        read_coro = reader.readline()
        raw = (
            await asyncio.wait_for(read_coro, timeout=timeout)
            if timeout is not None
            else await read_coro
        )
        if not raw:
            reason = self._mark_runtime_failed(f"EOF on sidecar '{method}' RPC")
            raise RuntimeError(reason)
        resp = json.loads(raw.decode("utf-8"))
        if resp.get("id") != req_id:
            raise RuntimeError(
                f"M5 sidecar RPC id mismatch on '{method}': expected {req_id}, got {resp.get('id')}"
            )
        if not resp.get("ok"):
            err = resp.get("error") or {}
            msg = str(err.get("message") or f"{method} failed")
            if method == "submit" and (
                "invalid" in msg.lower()
                or "exceeds max_context" in msg.lower()
                or "must " in msg.lower()
                or "supports " in msg.lower()
            ):
                raise InvalidRequestError(f"miv Core Backend: {msg}")
            raise RuntimeError(f"M5 sidecar '{method}' failed: {msg}")
        result = resp.get("result")
        if isinstance(result, dict):
            self._record_footprint_fields(result)
            return result
        return {"value": result}

    async def _rpc(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        *,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        if self._ctrl_reader is None or self._ctrl_writer is None:
            raise RuntimeError(
                self.runtime_failed_reason or "M5 sidecar engine is not started"
            )
        async with self._ctrl_lock:
            req_id = self._next_rpc_id
            self._next_rpc_id += 1
            try:
                return await self._rpc_on(
                    self._ctrl_reader,
                    self._ctrl_writer,
                    req_id,
                    method,
                    params,
                    timeout=timeout,
                )
            except (InvalidRequestError, asyncio.CancelledError):
                raise
            except (OSError, ConnectionError, EOFError) as exc:
                reason = self._mark_runtime_failed(str(exc))
                raise RuntimeError(reason) from exc

    def _retire_backend_id(self, backend_id: int) -> None:
        self._request_queues.pop(backend_id, None)
        self._retired_request_ids[backend_id] = None
        if len(self._retired_request_ids) > 4096:
            self._retired_request_ids.popitem(last=False)

    def _dispatch_raw_event(self, raw_event: dict[str, Any]) -> bool:
        """Route one decoded event frame into its per-request queue. Returns False on EOF."""
        kind = str(raw_event.get("kind") or "")
        if kind == "eof":
            return False
        req_id = int(raw_event.get("request_id") or 0)
        if req_id <= 0 or req_id in self._retired_request_ids:
            return True
        tokens_raw = raw_event.get("tokens") or ()
        event = _SidecarEvent(
            request_id=req_id,
            kind=kind,
            tokens=tuple(int(t) for t in tokens_raw),
            finish_reason=str(raw_event.get("finish_reason") or ""),
            error=str(raw_event.get("error") or ""),
            request_stats=(
                raw_event["request_stats"]
                if isinstance(raw_event.get("request_stats"), dict) else None
            ),
        )
        queue = self._request_queues.setdefault(req_id, asyncio.Queue())
        queue.put_nowait(event)
        return True

    async def _push_event_pump(self) -> None:
        """Read `subscribe_events` push frames from Connection 2 without blocking Control RPC."""
        assert self._event_reader is not None
        try:
            while not self._stopping:
                raw = await self._event_reader.readline()
                if not raw:
                    if not self._stopping:
                        self._mark_runtime_failed(
                            "event stream connection closed unexpectedly"
                        )
                    return
                frame = json.loads(raw.decode("utf-8"))
                event_obj = frame.get("event") if isinstance(frame, dict) else None
                if isinstance(event_obj, dict):
                    if not self._dispatch_raw_event(event_obj):
                        return
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if not self._stopping:
                logger.error("M5 sidecar push event pump crashed: %s", exc)
                self._mark_runtime_failed(f"event stream error: {exc}")

    async def _poll_event_pump(self) -> None:
        """Fallback `poll` loop when talking to a legacy daemon without `subscribe_events`."""
        poll_id = 1
        try:
            while not self._stopping:
                if self._event_reader is not None and self._event_writer is not None:
                    result = await self._rpc_on(
                        self._event_reader,
                        self._event_writer,
                        poll_id,
                        "poll",
                        {"max_wait_us": 20_000},
                    )
                    poll_id += 1
                elif self._request_queues:
                    result = await self._rpc("poll", {"max_wait_us": 10_000})
                else:
                    await asyncio.sleep(0.01)
                    continue
                for ev in result.get("events") or ():
                    if isinstance(ev, dict):
                        self._dispatch_raw_event(ev)
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            if not self._stopping:
                logger.error("M5 sidecar poll event pump crashed: %s", exc)
                self._mark_runtime_failed(f"poll pump error: {exc}")

    async def _setup_event_stream(self) -> None:
        caps = set(self._hello_info.get("capabilities") or ())
        if "subscribe_events" in caps:
            reader, writer = await self._open_uds()
            self._event_reader, self._event_writer = reader, writer
            await self._rpc_on(
                reader,
                writer,
                1,
                "subscribe_events",
                {"poll_interval_us": 20_000},
                timeout=5.0,
            )
            self._push_subscribed = True
            self._event_pump_task = asyncio.create_task(
                self._push_event_pump(), name=f"{self.external_engine_id}-events:{self._model_id}"
            )
            return

        # Legacy single-client or non-push server fallback: probe second connection
        # gently with short timeout, otherwise share Control connection via short polls.
        try:
            assert self._socket_path is not None
            reader, writer = await asyncio.wait_for(
                asyncio.open_unix_connection(
                    str(self._socket_path), limit=UDS_MAX_LINE_BYTES
                ),
                timeout=0.15,
            )
            # Verify the daemon actually services a second socket synchronously.
            await self._rpc_on(reader, writer, 1, "hello", timeout=0.25)
            self._event_reader, self._event_writer = reader, writer
        except Exception:
            if "writer" in locals():
                writer.close()
                with contextlib.suppress(Exception):
                    await writer.wait_closed()
            self._event_reader = self._event_writer = None

        self._push_subscribed = False
        self._event_pump_task = asyncio.create_task(
            self._poll_event_pump(), name=f"{self.external_engine_id}-poll:{self._model_id}"
        )

    async def _refresh_stats(self) -> None:
        if self._stopping or self._ctrl_writer is None:
            return
        task = asyncio.current_task()
        cancelling = task.cancelling() if task is not None else 0
        try:
            self._last_stats = await self._rpc("stats", timeout=2.0)
            self._last_capacity = await self._rpc("capacity", timeout=2.0)
        except Exception:
            if task is not None and task.cancelling() > cancelling:
                raise asyncio.CancelledError from None
            logger.debug("M5 sidecar stats refresh failed", exc_info=True)

    async def _poll_stats_loop(self) -> None:
        while not self._stopping:
            await asyncio.sleep(STATS_POLL_SECONDS)
            await self._refresh_stats()

    async def start(self) -> None:
        if self._process is not None:
            if self._process.poll() is None and self.runtime_failed_reason is None:
                return
            raise RuntimeError(
                self.runtime_failed_reason or "M5 sidecar process is not healthy"
            )

        self._stopping = False
        self._runtime_failed_reason_override = None
        await asyncio.to_thread(self._load_tokenizer)

        if (
            not self._target.fake
            and self._target.artifact_path is None
            and self._target.auto_convert_output is not None
        ):
            logger.info(
                "%s: auto-repacking weights in %s -> %s on first load",
                self._target.binary.label,
                self._path,
                self._target.auto_convert_output.name,
            )
            conv = await asyncio.to_thread(
                convert_provider_model,
                self._path,
                source_repo=self._model_id,
                binary=self._target.binary,
                manifest=self._provider_manifest,
                is_builtin=self._is_builtin_m5rt,
            )
            built_art = Path(str(conv["artifact_path"])).resolve()
            self._target = replace(
                self._target,
                artifact_path=built_art,
                auto_convert_output=None,
            )
            self._weights_size = None
            if (
                self._options.speculative_backend == "mtp"
                and "mtp_artifact" not in self._options.path_overrides
                and not (self._path / "mtp.miv").is_file()
                and not (self._path / "mtp.omlxm5").is_file()
                and inspect_omlxm5_has_mtp_weights(built_art) is False
            ):
                if self._options.speculative_policy != "auto":
                    raise ValueError("MTP was requested but the converted model has no MTP weights")
                logger.warning("%s: converted model has no MTP weights; auto selected AR", self._model_id)
                self._options = replace(
                    self._options,
                    speculative_backend="none",
                    draft_tokens=0,
                    path_overrides={**self._options.path_overrides, "enable_mtp": "off"},
                )

        if (
            not self._target.fake
            and self._target.draft_dir is not None
            and self._target.draft_artifact_path is None
            and self._target.auto_convert_draft_output is not None
        ):
            logger.info(
                "%s: auto-repacking DFlash2 draft weights in %s -> %s on first load",
                self._target.binary.label,
                self._target.draft_dir,
                self._target.auto_convert_draft_output.name,
            )
            draft_conv = await asyncio.to_thread(
                convert_provider_model,
                self._target.draft_dir,
                source_repo=self._target.draft_dir.name,
                binary=self._target.binary,
                manifest=self._provider_manifest,
                nax_pretile=False if self._is_builtin_m5rt else None,
                is_builtin=self._is_builtin_m5rt,
            )
            built_draft_art = Path(str(draft_conv["artifact_path"])).resolve()
            self._target = replace(
                self._target,
                draft_artifact_path=built_draft_art,
                auto_convert_draft_output=None,
            )
            self._weights_size = None

        self._socket_path = _short_socket_path(self.external_engine_id)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [str(self._target.binary.path), "--socket", str(self._socket_path)]
        logger.info(
            "%s: starting %s (log: %s)",
            self._target.binary.label,
            " ".join(command),
            self._log_path,
        )
        with open(self._log_path, "ab") as log_fh:
            self._log_offset = log_fh.tell()
            self._process = subprocess.Popen(
                _watchdog_command(command),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                env=_clean_sidecar_env(),
                start_new_session=True,
            )

        try:
            self._ctrl_reader, self._ctrl_writer = await self._open_uds()
            self._hello_info = await self._rpc("hello", timeout=10.0)
            self._last_stats = await self._rpc(
                "load", self._build_load_params(), timeout=STARTUP_TIMEOUT_SECONDS
            )
            self._last_capacity = await self._rpc("capacity", timeout=5.0)
            await self._setup_event_stream()
            self._stats_task = asyncio.create_task(
                self._poll_stats_loop(), name=f"{self.external_engine_id}-stats:{self._model_id}"
            )
        except BaseException:
            await self.stop()
            raise

    @staticmethod
    async def _close_writer(writer: asyncio.StreamWriter | None) -> None:
        if writer is None:
            return
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()

    async def stop(self) -> None:
        self._stopping = True

        # Send graceful shutdown RPC first while event pump / process can exit cleanly.
        if (
            self._ctrl_writer is not None
            and self._process is not None
            and self._process.poll() is None
        ):
            with contextlib.suppress(Exception):
                await self._rpc("shutdown", timeout=1.5)

        for task in (self._event_pump_task, self._stats_task, self._stats_refresh):
            if task is not None and not task.done():
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError, Exception):
                    await task
        self._event_pump_task = self._stats_task = self._stats_refresh = None

        await self._close_writer(self._event_writer)
        await self._close_writer(self._ctrl_writer)
        self._event_reader = self._event_writer = None
        self._ctrl_reader = self._ctrl_writer = None

        # Unblock any callers still awaiting per-request queues.
        stop_ev = _SidecarEvent(
            request_id=0,
            kind="aborted",
            finish_reason="cancelled",
            error="M5 sidecar stopped",
        )
        for queue in list(self._request_queues.values()):
            queue.put_nowait(stop_ev)
        self._request_queues.clear()
        self._client_to_backend_id.clear()
        self._live.clear()
        self._active = 0

        proc, self._process = self._process, None
        if proc is not None:
            group = proc.pid
            if proc.poll() is None:
                _signal_group(group, signal.SIGTERM)
                if not await _wait_group_exit(
                    group, STOP_TIMEOUT_SECONDS, resend=signal.SIGTERM
                ):
                    logger.warning("miv Core Backend: sidecar ignored SIGTERM; sending SIGKILL")
                    _signal_group(group, signal.SIGKILL)
                    await _wait_group_exit(group, 3.0)
            await asyncio.to_thread(proc.wait)
            # Ensure no child PID lingers in the process group before admitting another model.
            if _group_alive(group):
                _signal_group(group, signal.SIGKILL)
                await _wait_group_exit(group, 1.0)

        sock_path, self._socket_path = self._socket_path, None
        if sock_path is not None:
            with contextlib.suppress(OSError):
                sock_path.unlink(missing_ok=True)
        self._sidecar_pid = None
        self._reported_footprint_bytes = 0
        self._reported_lora_buffer_bytes = None
        self._reported_lora_adapted_layers = 0
        self._last_gpu_contended_hint = None

    async def reload_lora(
        self,
        lora_adapter: str | Path | None,
        lora_mode: str = "merge",
        *,
        lora_adapters: dict[str, str] | None = None,
        model_dirs: Iterable[str | Path] = (),
    ) -> dict[str, Any]:
        """Swap or clear the active LoRA adapter via `reload_lora` RPC (fallback `load`)."""
        norm_mode = str(lora_mode or "merge").strip().lower()
        if norm_mode not in VALID_LORA_MODES:
            raise ValueError(f"lora_mode must be one of {sorted(VALID_LORA_MODES)}")
        raw_ref = str(lora_adapter or "").strip()
        resolved_lora = (
            str(
                resolve_adapter_path(
                    raw_ref,
                    adapters=lora_adapters,
                    model_dir=self._path,
                    model_dirs=model_dirs,
                )
            )
            if raw_ref
            else ""
        )
        self._options = replace(
            self._options,
            lora_adapter=resolved_lora,
            lora_mode=norm_mode,
        )
        self._lora_estimated_size = None
        self._reported_lora_buffer_bytes = None
        self._reported_lora_adapted_layers = 0
        if self._ctrl_writer is None or self._stopping:
            return {
                "lora_adapter": resolved_lora,
                "lora_mode": norm_mode,
                "lora_buffer_bytes": self.lora_memory_bytes(),
                "lora_adapted_layers": 0,
            }
        try:
            res = await self._rpc(
                "reload_lora",
                {"lora_adapter": resolved_lora, "lora_mode": norm_mode},
                timeout=STARTUP_TIMEOUT_SECONDS,
            )
        except RuntimeError as exc:
            if not self._is_unknown_rpc_error(exc):
                raise
            res = await self._rpc(
                "load",
                self._build_load_params(),
                timeout=STARTUP_TIMEOUT_SECONDS,
            )
        self._last_stats = res
        return res

    @staticmethod
    def _is_unknown_rpc_error(exc: BaseException) -> bool:
        msg = str(exc).lower()
        return any(
            token in msg
            for token in (
                "unknown",
                "unsupported",
                "not implemented",
                "unrecognized",
                "method not found",
            )
        )

    async def trim_cache(self, target_free_bytes: int) -> dict[str, Any]:
        """Request non-destructive prefix/checkpoint cache trim under soft/hard pressure.

        Gracefully no-ops if connected to a v1 daemon that does not implement `trim_cache`.
        """
        if (
            self._stopping
            or self._ctrl_writer is None
            or self._trim_cache_unsupported
            or target_free_bytes <= 0
        ):
            return {"freed_bytes": 0, "supported": not self._trim_cache_unsupported}
        try:
            res = await self._rpc(
                "trim_cache",
                {"target_free_bytes": int(target_free_bytes)},
                timeout=5.0,
            )
            freed = max(0, int(res.get("freed_bytes", 0) or 0))
            resident = res.get("resident_bytes")
            if isinstance(resident, int) and resident >= 0:
                self._reported_footprint_bytes = resident
            return {
                "freed_bytes": freed,
                "resident_bytes": resident,
                "supported": True,
            }
        except RuntimeError as exc:
            if self._is_unknown_rpc_error(exc):
                self._trim_cache_unsupported = True
                logger.debug("M5 sidecar daemon does not implement trim_cache; skipping")
                return {"freed_bytes": 0, "supported": False}
            logger.debug("M5 sidecar trim_cache failed: %s", exc)
            return {"freed_bytes": 0, "supported": False}
        except Exception:
            logger.debug("M5 sidecar trim_cache failed", exc_info=True)
            return {"freed_bytes": 0, "supported": False}

    async def set_scheduler_hint(self, *, gpu_contended: bool) -> bool:
        """Inform the sidecar scheduler whether a coexisting stock MLX model is active.

        Gracefully no-ops if connected to a v1 daemon that does not implement
        `set_scheduler_hint`.
        """
        desired = bool(gpu_contended)
        if self._last_gpu_contended_hint is desired:
            return True
        if (
            self._stopping
            or self._ctrl_writer is None
            or self._scheduler_hint_unsupported
        ):
            return False
        try:
            await self._rpc(
                "set_scheduler_hint",
                {"gpu_contended": desired},
                timeout=2.0,
            )
            self._last_gpu_contended_hint = desired
            return True
        except RuntimeError as exc:
            if self._is_unknown_rpc_error(exc):
                self._scheduler_hint_unsupported = True
                self._last_gpu_contended_hint = desired
                logger.debug(
                    "M5 sidecar daemon does not implement set_scheduler_hint; skipping"
                )
                return False
            logger.debug("M5 sidecar set_scheduler_hint failed: %s", exc)
            return False
        except Exception:
            logger.debug("M5 sidecar set_scheduler_hint failed", exc_info=True)
            return False

    # -- Chat templating & token counting --------------------------------------

    def _ensure_tokenizer_loaded(self) -> None:
        if self._tokenizer is None:
            self._load_tokenizer()

    def _apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
        add_generation_prompt: bool | None = None,
        *,
        system_messages_prepared: bool = False,
    ) -> str:
        self._ensure_tokenizer_loaded()
        messages = [dict(m) for m in messages]
        if not system_messages_prepared:
            messages = _consolidate_system_messages(messages)
        if hasattr(self._tokenizer, "apply_chat_template"):
            if is_partial is None:
                is_partial = detect_and_strip_partial(messages)
            else:
                for msg in messages:
                    msg.pop("partial", None)

            template_kwargs: dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": (
                    not is_partial
                    if add_generation_prompt is None
                    else add_generation_prompt
                ),
            }
            if is_partial:
                template_kwargs["continue_final_message"] = True
            if tools:
                template_kwargs["tools"] = tools
            if self._enable_thinking is not None:
                template_kwargs["enable_thinking"] = self._enable_thinking
            if chat_template_kwargs:
                template_kwargs.update(chat_template_kwargs)

            try:
                return apply_chat_template_with_reasoning_effort_fallback(
                    self._tokenizer,
                    messages,
                    template_kwargs,
                    is_harmony=self._model_type == "gpt_oss",
                )
            except TypeError:
                if chat_template_kwargs:
                    for key in chat_template_kwargs:
                        template_kwargs.pop(key, None)
                template_kwargs.pop("tools", None)
                template_kwargs.pop("enable_thinking", None)
                return self._tokenizer.apply_chat_template(messages, **template_kwargs)
        prompt = "\n".join(f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages)
        return prompt + "\nassistant:"

    def prepare_chat_prompt(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
        add_generation_prompt: bool | None = None,
        add_special_tokens: bool | None = None,
        *,
        system_messages_prepared: bool = False,
    ) -> tuple[str, list[int]]:
        self._ensure_tokenizer_loaded()
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
            add_generation_prompt=add_generation_prompt,
            system_messages_prepared=system_messages_prepared,
        )
        if add_special_tokens is None:
            token_ids = list(self._tokenizer.encode(prompt))
        else:
            token_ids = list(
                self._tokenizer.encode(prompt, add_special_tokens=add_special_tokens)
            )
        return prompt, token_ids

    def count_chat_tokens(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
    ) -> int:
        _, token_ids = self.prepare_chat_prompt(
            messages,
            tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
        )
        return len(token_ids)

    async def tokenize_chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict] | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
        is_partial: bool | None = None,
        add_generation_prompt: bool | None = None,
        add_special_tokens: bool | None = None,
        *,
        system_messages_prepared: bool = False,
    ) -> list[int]:
        if self._process is None:
            await self.start()
        _, token_ids = await asyncio.to_thread(
            self.prepare_chat_prompt,
            messages,
            tools,
            chat_template_kwargs,
            is_partial,
            add_generation_prompt,
            add_special_tokens,
            system_messages_prepared=system_messages_prepared,
        )
        return token_ids

    # -- Request abort & streaming generation ----------------------------------

    def _remember_aborted_client(self, request_id: str) -> None:
        self._aborted_client_ids[request_id] = None
        if len(self._aborted_client_ids) > 4096:
            self._aborted_client_ids.popitem(last=False)

    async def _cancel_backend_request(self, backend_id: int) -> bool:
        if self._ctrl_writer is None or self._stopping:
            return False
        try:
            res = await self._rpc(
                "cancel", {"request_id": int(backend_id)}, timeout=3.0
            )
            return bool(res.get("cancelled", False))
        except Exception:
            logger.debug(
                "M5 sidecar cancel for backend request %d failed",
                backend_id,
                exc_info=True,
            )
            return False

    async def abort_request(
        self,
        request_id: str,
        *,
        reason: str | None = None,
        error_code: str | None = None,
    ) -> bool:
        """Immediately cancel an in-flight request over the Control UDS connection."""
        del reason, error_code
        key = str(request_id)
        self._remember_aborted_client(key)
        backend_id = self._client_to_backend_id.get(key)
        if backend_id is None and key.isdigit():
            cand = int(key)
            if cand in self._request_queues:
                backend_id = cand
        if backend_id is None:
            return False
        return await self._cancel_backend_request(backend_id)

    async def abort_all_requests(
        self,
        *,
        reason: str | None = None,
        error_code: str | None = None,
    ) -> int:
        del reason, error_code
        count = 0
        for backend_id in list(self._request_queues):
            if await self._cancel_backend_request(backend_id):
                count += 1
        return count

    @staticmethod
    def _normalize_stop_strings(
        stop: str | list[str] | None, extra_stop: Any
    ) -> list[str]:
        result: list[str] = []
        for source in (stop, extra_stop):
            if isinstance(source, str):
                if source:
                    result.append(source)
            elif isinstance(source, (list, tuple)):
                for item in source:
                    if isinstance(item, str) and item:
                        result.append(item)
        return list(dict.fromkeys(result))

    def _build_stop_token_seqs(
        self, *, ignore_eos: bool, extra_token_seqs: Any = None
    ) -> list[list[int]]:
        seqs: list[list[int]] = []
        if not ignore_eos:
            for eos_id in self._eos_token_ids:
                seqs.append([int(eos_id)])
        if isinstance(extra_token_seqs, (list, tuple)):
            for seq in extra_token_seqs:
                if isinstance(seq, (list, tuple)) and seq:
                    seqs.append([int(t) for t in seq])
        dedup: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for seq in seqs:
            tup = tuple(seq)
            if tup not in seen:
                seen.add(tup)
                dedup.append(seq)
        return dedup

    def _thinking_token_ids(self) -> tuple[list[int], list[int]] | None:
        """Use model-declared markers or real tokenizer tokens for Qwen markers."""
        boundaries = []
        for side, default_tag in (("start", "<think>"), ("end", "</think>")):
            tokenizer = self._tokenizer
            ids = _safe_tokenizer_attr(tokenizer, f"think_{side}_tokens")
            if ids is None:
                token_id = _safe_tokenizer_attr(tokenizer, f"think_{side}_id")
                if token_id is not None:
                    ids = [token_id]
            if ids is None:
                tag = _safe_tokenizer_attr(tokenizer, f"think_{side}")
                if tag:
                    ids = _encode_prompt_ids(tokenizer, tag)
                else:
                    token_id = _convert_token_to_id(tokenizer, default_tag)
                    ids = [token_id] if token_id is not None else None
            if not ids or any(not isinstance(i, int) or not 0 <= i < self._vocab_size for i in ids):
                return None
            boundaries.append(list(ids))
        start, end = boundaries
        return (start, end) if start != end else None

    def _thinking_marker_strings(self) -> tuple[str, str]:
        """Resolve literal thinking open/close markers for reasoning-aware stop matching."""
        tokenizer = self._tokenizer
        start_tag = _safe_tokenizer_attr(tokenizer, "think_start")
        end_tag = _safe_tokenizer_attr(tokenizer, "think_end")
        if (
            isinstance(start_tag, str)
            and start_tag
            and isinstance(end_tag, str)
            and end_tag
            and start_tag != end_tag
        ):
            return start_tag, end_tag
        markers = self._thinking_token_ids()
        if markers is not None and tokenizer is not None and hasattr(tokenizer, "decode"):
            with contextlib.suppress(Exception):
                dec_start = tokenizer.decode(markers[0], skip_special_tokens=False)
                dec_end = tokenizer.decode(markers[1], skip_special_tokens=False)
                if (
                    isinstance(dec_start, str)
                    and dec_start
                    and isinstance(dec_end, str)
                    and dec_end
                    and dec_start != dec_end
                ):
                    return dec_start, dec_end
        return "<think>", "</think>"

    def _prompt_tokens_open_think(self, prompt_tokens: list[int]) -> bool:
        markers = self._thinking_token_ids()
        if markers is None:
            return False
        start, end = markers
        for tail_size in range(3):
            end_of_start = len(prompt_tokens) - tail_size
            beginning = end_of_start - len(start)
            if beginning < 0 or prompt_tokens[beginning:end_of_start] != start:
                continue
            suffix = prompt_tokens[end_of_start:]
            return not any(
                suffix[i : i + len(end)] == end
                for i in range(len(suffix) - len(end) + 1)
            )
        return False

    def _request_sampling_contract(self) -> str:
        return (
            (self._last_stats or {}).get("sampling_contract_id")
            or self._options.sampling_contract
        )

    def request_control_settings(self) -> tuple[str, ...]:
        """Model settings the loaded daemon will apply to generation."""
        if self._request_sampling_contract() != "position_coupled_v1":
            return ()
        capabilities = self._hello_info.get("capabilities") or ()
        settings = []
        if "thinking_budget" in capabilities and self._thinking_token_ids() is not None:
            settings.extend(("thinking_budget_enabled", "thinking_budget_tokens"))
        if "sampling_penalties" in capabilities:
            settings.extend(("repetition_penalty", "presence_penalty", "frequency_penalty"))
        return tuple(settings)

    def reconcile_request_controls(
        self,
        options: dict[str, Any],
        explicit_fields: dict[str, str] | None = None,
    ) -> list[str]:
        """Require support for explicit controls and drop unsupported inherited values."""
        defaults = {
            "thinking_budget": None,
            "repetition_penalty": 1.0,
            "repetition_context_size": None,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }
        active = [
            name for name, default in defaults.items()
            if options.get(name) is not None and options[name] != default
        ]
        if not active:
            return []

        explicit_fields = explicit_fields or {}
        capabilities = self._hello_info.get("capabilities") or ()
        contract = self._request_sampling_contract()
        warnings = []
        for name in active:
            capability = "thinking_budget" if name == "thinking_budget" else "sampling_penalties"
            reason = None
            if capability not in capabilities:
                reason = f"the selected sidecar did not advertise {capability}"
            elif contract != "position_coupled_v1":
                reason = "it requires sampling_contract=position_coupled_v1"
            elif name == "thinking_budget" and self._thinking_token_ids() is None:
                reason = "the selected tokenizer has no supported thinking marker pair"

            field = explicit_fields.get(name, name)
            value = options[name]
            if reason is None and name in ("thinking_budget", "repetition_context_size"):
                minimum = 0 if name == "thinking_budget" else 1
                maximum = (
                    2**31 - 1
                    if name == "thinking_budget"
                    else M5RT_MAX_REPETITION_CONTEXT_SIZE
                )
                if (
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or not minimum <= value <= maximum
                ):
                    raise InvalidRequestError(
                        f"{field} must be an integer in [{minimum}, {maximum}]", field
                    )
            elif reason is None:
                try:
                    value = struct.unpack("f", struct.pack("f", value))[0]
                except (OverflowError, TypeError, struct.error):
                    value = float("nan")
                if not math.isfinite(value) or (name == "repetition_penalty" and value <= 0):
                    requirement = "a positive finite number" if name == "repetition_penalty" else "a finite number"
                    raise InvalidRequestError(f"{field} must be {requirement}", field)

            if reason is not None:
                message = f"{field} is unsupported because {reason}"
                if name in explicit_fields:
                    raise InvalidRequestError(message, field)
                options.pop(name)
                warnings.append(f"Inherited {message}. Using the backend default.")
            elif name != "thinking_budget" and name != "repetition_context_size":
                options[name] = value

        self._warn_unsupported_controls(warnings)
        return warnings

    def _warn_unsupported_controls(self, warnings: Iterable[str]) -> None:
        for warning in warnings:
            if warning not in self._warned_request_controls:
                self._warned_request_controls.add(warning)
                logger.warning("%s: %s", self._model_id, warning)

    def _thinking_budget_payload(self, budget: int, prompt_tokens: list[int]) -> dict[str, Any]:
        markers = self._thinking_token_ids()
        if markers is None:
            raise InvalidRequestError("thinking_budget requires supported thinking marker tokens", "thinking_budget")
        start, end = markers
        # As in the stock template check, only the final opener with up to
        # two tokens after it can begin this turn's reasoning.
        opened = False
        for tail_size in range(3):
            end_of_start = len(prompt_tokens) - tail_size
            beginning = end_of_start - len(start)
            if beginning < 0 or prompt_tokens[beginning:end_of_start] != start:
                continue
            suffix = prompt_tokens[end_of_start:]
            opened = not any(
                suffix[i:i + len(end)] == end
                for i in range(len(suffix) - len(end) + 1)
            )
            break
        return {
            "budget": budget,
            "start_in_thinking": opened,
            "start_token_ids": start,
            "end_token_ids": end,
        }

    def _validate_and_normalize_sampling(
        self,
        *,
        temperature: float,
        top_p: float,
        top_k: int,
        min_p: float,
        kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        if kwargs.get("compiled_grammar") is not None:
            raise InvalidRequestError(
                "miv Core Backend sidecar does not support xgrammar logit-constrained decoding"
            )
        if kwargs.get("logprobs"):
            raise InvalidRequestError(
                "miv Core Backend sidecar does not return per-token logprobs"
            )
        if min_p != 0.0:
            raise InvalidRequestError("miv Core Backend supports min_p=0.0 only")

        temp = float(temperature)
        tp = float(top_p)
        if temp < 0.0:
            raise InvalidRequestError("temperature must be non-negative")
        if tp == 0.0:
            temp = 0.0
            tp = 1.0
        elif not 0.0 < tp <= 1.0:
            raise InvalidRequestError("top_p must be in (0, 1]")

        tk = int(top_k or 0)
        if tk < 0:
            raise InvalidRequestError("top_k must be non-negative")
        max_hw_topk = min(64, self._vocab_size)
        if temp > 0.0:
            # Native Metal `greedy_topk64` requires 1 <= top_k <= min(64, vocab_size).
            tk = max_hw_topk if tk == 0 else min(tk, max_hw_topk)
        else:
            tk = min(tk, max_hw_topk) if tk > 0 else 0

        seed_raw = kwargs.get("seed")
        seed = int(seed_raw) & ((1 << 64) - 1) if seed_raw is not None else 0
        return {
            "temperature": temp,
            "top_p": tp,
            "min_p": 0.0,
            "top_k": tk,
            "seed": seed,
        }

    @staticmethod
    def _compute_tps(
        *,
        prompt_tokens: int,
        completion_tokens: int,
        started_at: float,
        first_token_at: float | None,
        now: float,
    ) -> tuple[float, float]:
        prompt_tps = 0.0
        gen_tps = 0.0
        if first_token_at is not None:
            ttft = first_token_at - started_at
            if ttft > 0 and prompt_tokens > 0:
                prompt_tps = prompt_tokens / ttft
            gen_window = now - first_token_at
            if gen_window > 1e-6 and completion_tokens > 1:
                gen_tps = (completion_tokens - 1) / gen_window
            elif now - started_at > 1e-6 and completion_tokens > 0:
                gen_tps = completion_tokens / (now - started_at)
        return prompt_tps, gen_tps

    async def stream_generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationOutput]:
        if self._process is None:
            await self.start()
        if self.runtime_failed_reason is not None:
            raise RuntimeError(self.runtime_failed_reason)

        controls = {
            "repetition_penalty": repetition_penalty,
            "presence_penalty": presence_penalty,
            "frequency_penalty": kwargs.get("frequency_penalty"),
            "repetition_context_size": kwargs.get("repetition_context_size"),
            "thinking_budget": kwargs.get("thinking_budget"),
        }
        self.reconcile_request_controls(controls)
        self._warn_unsupported_controls(
            f"{self._target.binary.label} does not support {key}"
            for key in ("no_repeat_ngram_size", "xtc_probability")
            if kwargs.get(key)
        )

        client_req_id = str(kwargs.pop("_request_id", None) or uuid.uuid4())
        if client_req_id in self._aborted_client_ids:
            return

        sampling = self._validate_and_normalize_sampling(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            kwargs=kwargs,
        )

        thinking_budget = controls.pop("thinking_budget", None)
        for name, neutral in (
            ("repetition_penalty", 1.0),
            ("presence_penalty", 0.0),
            ("frequency_penalty", 0.0),
            ("repetition_context_size", 20),
        ):
            value = controls.get(name)
            if value is not None and value != neutral:
                sampling[name] = value

        if isinstance(prompt, str):
            prompt_tokens = await asyncio.to_thread(
                lambda: list(self._tokenizer.encode(prompt))
            )
        else:
            prompt_tokens = [int(t) for t in prompt]
        if not prompt_tokens:
            prompt_tokens = [self._eos_token_ids[0] if self._eos_token_ids else 1]

        num_prompt = len(prompt_tokens)
        if num_prompt >= self._options.max_context:
            raise InvalidRequestError(
                f"prompt is too long: {num_prompt} tokens >= {self._options.max_context} maximum context length"
            )
        budget = min(max(1, int(max_tokens)), self._options.max_context - num_prompt)

        ignore_eos = bool(kwargs.get("ignore_eos", False))
        stop_strings = self._normalize_stop_strings(
            stop, kwargs.get("stop_sequences")
        )
        stop_token_seqs = self._build_stop_token_seqs(
            ignore_eos=ignore_eos,
            extra_token_seqs=kwargs.get("stop_token_seqs"),
        )
        skip_reasoning = bool(kwargs.get("stop_skips_reasoning"))
        think_open, think_close = (
            self._thinking_marker_strings() if skip_reasoning else ("<think>", "</think>")
        )
        if "_prompt_opens_think" in kwargs:
            prompt_opens_think = bool(kwargs.get("_prompt_opens_think"))
        elif skip_reasoning:
            if isinstance(prompt, str):
                prompt_opens_think = prompt.rfind(think_open) > prompt.rfind(think_close)
            else:
                prompt_opens_think = self._prompt_tokens_open_think(prompt_tokens)
        else:
            prompt_opens_think = False
        stop_matcher = _StopMatcher(
            stop_token_seqs,
            stop_strings,
            skip_reasoning=skip_reasoning,
            prompt_opens_think=prompt_opens_think,
            think_open=think_open,
            think_close=think_close,
        )
        special_ids = set() if ignore_eos else set(self._eos_token_ids)
        detok = _IncrementalDetokenizer(self._tokenizer, self._path, special_ids)

        started_perf = time.perf_counter()
        live = _LiveRequest(
            started_at=time.monotonic(),
            prompt_tokens=num_prompt,
            max_tokens=budget,
        )
        self._live[client_req_id] = live
        self._active += 1
        self._requests += 1

        backend_id: int | None = None
        terminal_seen = False
        all_tokens: list[int] = []
        accumulated_text = ""
        first_token_at: float | None = None
        finish_reason = "stop"
        request_stats: dict[str, Any] = {}

        try:
            submit_params: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": budget,
                "sampling": sampling,
                "stop_token_seqs": stop_token_seqs,
            }
            if thinking_budget is not None:
                submit_params["thinking_budget"] = self._thinking_budget_payload(
                    thinking_budget, prompt_tokens
                )
            if kwargs.get("pin_prefix"):
                submit_params["pin_prefix"] = True
            submit_res = await self._rpc(
                "submit",
                submit_params,
            )
            backend_id = int(submit_res["request_id"])
            queue = self._request_queues.setdefault(backend_id, asyncio.Queue())
            self._client_to_backend_id[client_req_id] = backend_id

            if client_req_id in self._aborted_client_ids:
                await self._cancel_backend_request(backend_id)

            while True:
                event = await queue.get()
                if event.kind == "tokens":
                    batch = stop_matcher.filter_tokens(event.tokens)
                    if batch:
                        now_perf = time.perf_counter()
                        first_token_at = first_token_at or now_perf
                        all_tokens.extend(batch)
                        live.add_tokens(len(batch), time.monotonic())

                        raw_piece = detok.add_tokens(batch)
                        emit_piece = stop_matcher.push_text(raw_piece)
                        if emit_piece:
                            accumulated_text += emit_piece
                            p_tps, g_tps = self._compute_tps(
                                prompt_tokens=num_prompt,
                                completion_tokens=len(all_tokens),
                                started_at=started_perf,
                                first_token_at=first_token_at,
                                now=now_perf,
                            )
                            yield GenerationOutput(
                                text=accumulated_text,
                                tokens=list(batch),
                                new_text=emit_piece,
                                prompt_tokens=num_prompt,
                                completion_tokens=len(all_tokens),
                                finished=False,
                                finish_reason=None,
                                first_token_at=first_token_at,
                                generated_at=now_perf,
                                prompt_tps=p_tps,
                                generation_tps=g_tps,
                            )

                    if stop_matcher.stopped:
                        finish_reason = "stop"
                        break
                    continue

                terminal_seen = True
                if event.kind == "failed":
                    raise RuntimeError(
                        event.error
                        or self.runtime_failed_reason
                        or "M5 sidecar request failed"
                    )
                if event.kind == "aborted":
                    finish_reason = "cancelled"
                    return

                # `finished`
                request_stats = event.request_stats or {}
                if stop_matcher.stopped or event.finish_reason == "stop":
                    finish_reason = "stop"
                elif len(all_tokens) >= budget or event.finish_reason == "length":
                    finish_reason = "length"
                else:
                    finish_reason = event.finish_reason or "stop"
                break

            tail_raw = "" if stop_matcher.stopped else detok.finalize()
            tail_emit = (
                ""
                if stop_matcher.stopped
                else stop_matcher.push_text(tail_raw) + stop_matcher.flush()
            )
            if stop_matcher.stopped:
                finish_reason = "stop"
            if tail_emit:
                accumulated_text += tail_emit

            now_perf = time.perf_counter()
            if all_tokens and first_token_at is None:
                first_token_at = now_perf
            if not terminal_seen and backend_id is not None:
                await self._cancel_backend_request(backend_id)
                terminal_seen = True
                deadline = time.monotonic() + 0.25
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                    if event.kind != "tokens":
                        request_stats = event.request_stats or {}
                        break
            cached_tokens = max(0, min(num_prompt, int(request_stats.get("prefix_restored_tokens") or 0)))
            mode = request_stats.get("decode_mode")
            if mode:
                logger.info(
                    "miv request %s model=%s mode=%s cached_tokens=%d spec_rounds=%s ar_steps=%s fallback=%s",
                    backend_id, self._model_id, mode, cached_tokens, request_stats.get("spec_rounds"),
                    request_stats.get("ar_steps"), request_stats.get("fallback_reason") or "-",
                )
            p_tps, g_tps = self._compute_tps(
                prompt_tokens=num_prompt - cached_tokens,
                completion_tokens=len(all_tokens),
                started_at=started_perf,
                first_token_at=first_token_at,
                now=now_perf,
            )
            final_output = GenerationOutput(
                text=accumulated_text,
                tokens=list(all_tokens),
                new_text=tail_emit,
                prompt_tokens=num_prompt,
                completion_tokens=len(all_tokens),
                finished=True,
                finish_reason=finish_reason,
                cached_tokens=cached_tokens,
                first_token_at=first_token_at,
                generated_until=now_perf,
                prompt_tps=p_tps,
                generation_tps=g_tps,
            )
            yield final_output
        finally:
            if backend_id is not None:
                if not terminal_seen:
                    with contextlib.suppress(Exception):
                        await asyncio.shield(self._cancel_backend_request(backend_id))
                self._retire_backend_id(backend_id)
            self._client_to_backend_id.pop(client_req_id, None)
            self._aborted_client_ids.pop(client_req_id, None)
            self._live.pop(client_req_id, None)
            self._active = max(0, self._active - 1)
            if (
                not self._stopping
                and self._ctrl_writer is not None
                and (self._stats_refresh is None or self._stats_refresh.done())
            ):
                self._stats_refresh = asyncio.create_task(self._refresh_stats())

    @staticmethod
    async def _collect(outputs: AsyncIterator[GenerationOutput]) -> GenerationOutput:
        final: GenerationOutput | None = None
        async for out in outputs:
            final = out
        if final is None:
            return GenerationOutput(
                text="",
                tokens=[],
                prompt_tokens=0,
                completion_tokens=0,
                finish_reason="cancelled",
                new_text="",
                finished=True,
            )
        final.new_text = final.text
        return final

    async def generate(
        self,
        prompt: str | list[int],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        return await self._collect(
            self.stream_generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                stop=stop,
                **kwargs,
            )
        )

    async def stream_chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[GenerationOutput]:
        if self._process is None:
            await self.start()
        ct_kwargs = kwargs.pop("chat_template_kwargs", None)
        is_partial = kwargs.pop("is_partial", None)
        prepared = kwargs.pop("_prepared_prompt", None)
        if prepared is not None:
            prompt_text, token_ids = prepared[0], list(prepared[1])
        else:
            prompt_text, token_ids = await asyncio.to_thread(
                self.prepare_chat_prompt,
                messages,
                tools,
                ct_kwargs,
                is_partial,
            )
        if kwargs.get("stop_skips_reasoning") and isinstance(prompt_text, str):
            think_open, think_close = self._thinking_marker_strings()
            kwargs["_prompt_opens_think"] = (
                prompt_text.rfind(think_open) > prompt_text.rfind(think_close)
            )
        async for output in self.stream_generate(
            prompt=token_ids,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            **kwargs,
        ):
            yield output

    async def chat(
        self,
        messages: list[dict[str, Any]],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 0,
        min_p: float = 0.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> GenerationOutput:
        return await self._collect(
            self.stream_chat(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                presence_penalty=presence_penalty,
                tools=tools,
                **kwargs,
            )
        )

    # -- Memory, stats & observability -----------------------------------------

    def reserved_bytes(self) -> int:
        return int(self._options.memory_budget_bytes or 0)

    def _weights_bytes(self) -> int:
        if self._weights_size is None:
            total = 0
            if self._target.artifact_path and self._target.artifact_path.is_file():
                total += self._target.artifact_path.stat().st_size
            if (
                self._target.draft_artifact_path
                and self._target.draft_artifact_path.is_file()
            ):
                total += self._target.draft_artifact_path.stat().st_size
            self._weights_size = total
        return self._weights_size

    def lora_memory_bytes(self) -> int:
        """Return resident GPU UMA buffer bytes for the active LoRA adapter."""
        if (
            self._reported_lora_buffer_bytes is not None
            and self._reported_lora_buffer_bytes > 0
        ):
            return self._reported_lora_buffer_bytes
        if not self._options.lora_adapter:
            return 0
        if self._lora_estimated_size is None:
            self._lora_estimated_size = estimate_sidecar_lora_bytes(
                self._options.lora_adapter, lora_mode=self._options.lora_mode
            )
        if self._reported_lora_buffer_bytes == 0 and self._lora_estimated_size == 0:
            return 0
        return self._lora_estimated_size

    def _group_pids(self) -> list[int]:
        proc = self._process
        if proc is None or proc.poll() is not None:
            return []
        pids = list(list_process_group_pids(proc.pid))
        if self._sidecar_pid and self._sidecar_pid not in pids:
            pids.append(self._sidecar_pid)
        return pids

    def process_memory_bytes(self) -> int:
        pids = self._group_pids()
        if not pids:
            return 0
        group_footprint = sum(get_phys_footprint(pid) for pid in pids)
        floor = max(
            self._reported_footprint_bytes,
            self._weights_bytes() + self.lora_memory_bytes(),
        )
        return max(group_footprint, floor)

    def effective_load_settings(self) -> dict[str, Any] | None:
        """Return live daemon values only when it reports the load contract."""
        last = self._last_stats or {}
        if not last.get("max_context") or not last.get("speculative_backend"):
            return None
        keys = (
            "max_context",
            "max_concurrency",
            "kv_dtype",
            "sampling_contract_id",
            "speculative_backend",
            "max_draft_tokens",
            "prefill_linear_mode",
            "prefill_attention_mode",
            "prefill_sparse_minference",
            "prefill_sparse_minference_effective",
            "prefill_sparse_minference_inactive_reason",
            "prefill_cache_blend",
            "prefill_ane_offload",
            "prefill_ane_active",
            "prefill_ane_split_alpha",
            "prefill_w4a8_tokens",
            "prefill_sage2_tokens",
            "prefill_minference_blocks_skipped",
            "prefill_minference_blocks_total",
            "prefill_cache_blend_hit_tokens",
            "prefill_cache_blend_recomputed_tokens",
            "prefill_ane_offload_tokens",
            "prefill_ane_fallback_tokens",
        )
        return {
            **{k: last[k] for k in keys if k in last},
            "concurrency_source": self._options.concurrency_source,
        }

    def get_stats(self) -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
        last = self._last_stats or {}
        active_lora_adapter = (
            str(last["lora_adapter"])
            if isinstance(last.get("lora_adapter"), str)
            else self._options.lora_adapter
        )
        active_lora_mode = (
            str(last["lora_mode"])
            if isinstance(last.get("lora_mode"), str) and last.get("lora_mode")
            else self._options.lora_mode
        )
        active_lora_layers = (
            int(last["lora_adapted_layers"])
            if isinstance(last.get("lora_adapted_layers"), int)
            and not isinstance(last.get("lora_adapted_layers"), bool)
            else self._reported_lora_adapted_layers
        )
        stats: dict[str, Any] = {
            "engine_type": self.external_engine_id,
            "provider_id": self._provider_id,
            "m5rt_version": self._target.binary.version,
            "provider_version": self._target.binary.version,
            "model_name": self.model_name,
            "socket_path": str(self._socket_path) if self._socket_path else None,
            "log_path": str(self._log_path),
            "running": running,
            "push_subscribed": self._push_subscribed,
            "requests": self._requests,
            "active_requests": self._active,
            "lora_adapter": active_lora_adapter,
            "lora_mode": active_lora_mode,
            "lora_buffer_bytes": self.lora_memory_bytes(),
            "lora_adapted_layers": active_lora_layers,
            "memory_budget_bytes": self.reserved_bytes(),
            "process_memory_bytes": self.process_memory_bytes(),
        }
        if self._last_stats:
            stats["backend_stats"] = dict(self._last_stats)
        if (settings := self.effective_load_settings()) is not None:
            stats["effective_load_settings"] = settings
        if self._last_capacity:
            stats["capacity"] = dict(self._last_capacity)
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None

    def get_runtime_cache_stats(self) -> dict[str, Any] | None:
        if self._last_stats is None:
            return None
        s = self._last_stats
        hits = int(s.get("prefix_hits", 0))
        misses = int(s.get("prefix_misses", 0))
        evictions = int(s.get("prefix_evictions", 0))
        restored = int(s.get("prefix_restored_tokens", 0))
        return {
            "block_size": 64,
            "indexed_blocks": 0,
            "prefix_cache": {"block_size": 64},
            "cache_rates": {
                "cumulative": {
                    "prefix_hits": hits,
                    "prefix_misses": misses,
                    "evictions": evictions,
                    "ssd_disk_loads": 0,
                    "ssd_saves": 0,
                }
            },
            "m5rt_cache": {
                "prefix_hits": int(s.get("prefix_hits", 0)),
                "prefix_partial_hits": int(s.get("prefix_partial_hits", 0)),
                "prefix_misses": misses,
                "prefix_evictions": evictions,
                "prefix_restored_tokens": restored,
            },
        }

    def get_live_requests(self) -> dict[str, list[dict[str, Any]]]:
        now = time.monotonic()
        live_items = list(self._live.items())
        return {
            "prefilling": [
                req.prefill_row(rid, now, detail=self.external_engine_id)
                for rid, req in live_items
                if req.first_token_at is None
            ],
            "generating": [
                req.generating_row(rid, now)
                for rid, req in live_items
                if req.first_token_at is not None
            ],
        }


# ---------------------------------------------------------------------------
# In-App & First-Load .safetensors -> .miv / SPI Converter Helpers (oMLX-extend)
# ---------------------------------------------------------------------------

from contextlib import suppress
import threading

_CONVERT_LOCKS_GUARD = threading.Lock()
_CONVERT_LOCKS: dict[str, threading.Lock] = {}


def _has_valid_safetensors_header(model_dir: Path) -> bool:
    """Fast O(1) 9-byte check confirming at least one non-stub `.safetensors` file exists."""
    try:
        entries = [
            f
            for f in model_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() == ".safetensors"
            and not f.name.startswith(".")
        ]
    except OSError:
        return False
    for sf in entries:
        try:
            fsize = sf.stat().st_size
            if fsize < 16:
                continue
            with sf.open("rb") as fh:
                head = fh.read(9)
            if len(head) < 9:
                continue
            (hdr_len,) = struct.unpack("<Q", head[:8])
            if 2 <= hdr_len <= min(64 * 1024 * 1024, fsize - 8) and head[8] == ord("{"):
                return True
        except OSError:
            continue
    return False


def _converter_spec_for_manifest(
    manifest: dict[str, Any] | None, *, is_builtin: bool
) -> dict[str, Any] | None:
    raw = manifest.get("converter") if isinstance(manifest, dict) else None
    if isinstance(raw, dict) and raw.get("entrypoint"):
        return dict(raw)
    if is_builtin:
        return dict(M5RT_DEFAULT_CONVERTER)
    return None


def _converter_output_path(
    model_dir: Path, manifest: dict[str, Any] | None, *, is_builtin: bool
) -> Path:
    spec = _converter_spec_for_manifest(manifest, is_builtin=is_builtin) or {}
    out_name = str(spec.get("output_filename") or "").strip()
    if not out_name:
        ext = str(spec.get("artifact_extension") or "").strip()
        if not ext:
            exts = _normalize_extensions(
                manifest.get("artifact_extensions") if isinstance(manifest, dict) else None
            )
            ext = exts[0] if exts else ".miv"
        if not ext.startswith("."):
            ext = f".{ext}"
        out_name = f"model{ext}"
    return model_dir / Path(out_name).name


_NON_TEXT_TENSOR_PREFIXES: tuple[str, ...] = (
    "visual.",
    "model.visual.",
    "vision_tower.",
    "model.vision_tower.",
    "vision_model.",
    "model.vision_model.",
    "multi_modal_projector.",
    "model.multi_modal_projector.",
    "mm_projector.",
    "model.mm_projector.",
    "aligner.",
    "model.aligner.",
    "audio_tower.",
    "model.audio_tower.",
    "talker.",
    "model.talker.",
)
_NON_TEXT_TENSOR_MARKERS: tuple[str, ...] = (
    ".visual.",
    ".vision_tower.",
    ".vision_model.",
    ".multi_modal_projector.",
    ".mm_projector.",
    ".aligner.",
    ".audio_tower.",
    ".talker.",
)


def _classify_source_tensor_domain(name: str) -> str:
    """Classify tensor name into `'skip_non_text'`, `'companion_mtp'`, or `'text'`."""
    if name.startswith(_NON_TEXT_TENSOR_PREFIXES) or any(
        m in name for m in _NON_TEXT_TENSOR_MARKERS
    ):
        return "skip_non_text"
    if name.startswith("mtp.") or ".mtp." in name:
        return "companion_mtp"
    return "text"


def _read_safetensors_headers_fast(
    model_dir: Path,
) -> tuple[dict[str, tuple[str, tuple[int, ...]]], str | None, str]:
    """Read ONLY 8-byte header length + JSON header of `.safetensors` files (zero weight bytes)."""
    shard_paths: list[Path] = []
    idx_path = model_dir / "model.safetensors.index.json"
    if idx_path.is_file():
        idx_doc = _read_json(idx_path)
        if not isinstance(idx_doc, dict) or not isinstance(idx_doc.get("weight_map"), dict):
            return {}, "invalid_safetensors_index", "model.safetensors.index.json is invalid"
        seen_shards: set[str] = set()
        for shard_rel in idx_doc["weight_map"].values():
            if not isinstance(shard_rel, str) or not shard_rel.strip():
                continue
            sname = shard_rel.strip()
            if Path(sname).name != sname:
                return {}, "invalid_safetensors_index", f"unsafe shard path in index: {sname!r}"
            if sname not in seen_shards:
                seen_shards.add(sname)
                shard_paths.append(model_dir / sname)
        shard_paths.sort(key=lambda p: p.name)
    elif (model_dir / "model.safetensors").is_file():
        shard_paths.append(model_dir / "model.safetensors")
    else:
        try:
            for entry in sorted(model_dir.iterdir(), key=lambda p: p.name):
                if not entry.is_file() or entry.name.startswith("."):
                    continue
                suf = entry.suffix.lower()
                if suf == ".safetensors":
                    shard_paths.append(entry)
                elif suf == ".gguf" and not shard_paths:
                    return (
                        {},
                        "unsupported_gguf",
                        f"GGUF checkpoint '{entry.name}' is not supported by {M5RT_ENGINE_LABEL}",
                    )
        except OSError as exc:
            return {}, "io_error", f"failed scanning model directory: {exc}"

    if not shard_paths:
        return {}, "missing_safetensors", "no valid .safetensors weights found in model directory"

    tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    valid_shards = 0
    for spath in shard_paths:
        try:
            st = spath.stat()
            if st.st_size < 10:
                continue
            with spath.open("rb") as fh:
                raw_len = fh.read(8)
                if len(raw_len) != 8:
                    continue
                (hdr_len,) = struct.unpack("<Q", raw_len)
                if hdr_len < 2 or hdr_len > min(64 * 1024 * 1024, st.st_size - 8):
                    continue
                raw_hdr = fh.read(hdr_len)
            if len(raw_hdr) != hdr_len:
                continue
            parsed = json.loads(raw_hdr.decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError):
            if idx_path.is_file():
                return (
                    {},
                    "missing_safetensors_shard",
                    f"missing or unreadable safetensors shard '{spath.name}'",
                )
            continue
        if not isinstance(parsed, dict):
            continue
        valid_shards += 1
        for tname, rec in parsed.items():
            if tname == "__metadata__" or not isinstance(rec, dict):
                continue
            dtype = str(rec.get("dtype") or "")
            raw_shape = rec.get("shape")
            if not dtype or not isinstance(raw_shape, list):
                return (
                    {},
                    "invalid_safetensors_header",
                    f"malformed safetensors record for tensor {tname!r} in '{spath.name}'",
                )
            try:
                shape = tuple(int(dim) for dim in raw_shape)
            except (TypeError, ValueError):
                return (
                    {},
                    "invalid_safetensors_header",
                    f"invalid shape in safetensors header for tensor {tname!r}",
                )
            tensors[tname] = (dtype, shape)

    if valid_shards == 0:
        return {}, "missing_safetensors", "no valid .safetensors weights found in model directory"
    return tensors, None, ""


def _extract_quant_dicts(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    text_cfg = _text_config(cfg)
    raw_q = (
        cfg.get("quantization")
        or cfg.get("quantization_config")
        or text_cfg.get("quantization")
        or text_cfg.get("quantization_config")
        or {}
    )
    raw_qcfg = cfg.get("quantization_config") or text_cfg.get("quantization_config") or {}
    return (
        raw_q if isinstance(raw_q, dict) else {},
        raw_qcfg if isinstance(raw_qcfg, dict) else {},
    )


def _module_quant_spec(cfg: dict[str, Any], module_name: str) -> dict[str, Any]:
    quant, _ = _extract_quant_dicts(cfg)
    spec: dict[str, Any] = {
        "bits": quant.get("bits"),
        "group_size": quant.get("group_size"),
        "mode": str(quant.get("mode") or "").strip().lower(),
    }
    candidates = [module_name]
    if module_name.startswith("language_model."):
        candidates.append(module_name[len("language_model.") :])
    for cand in list(candidates):
        if cand.startswith("model."):
            candidates.append(cand[len("model.") :])
    if module_name.endswith(".lm_head"):
        candidates.append("lm_head")
    for key in candidates:
        override = quant.get(key)
        if isinstance(override, dict):
            if "bits" in override:
                spec["bits"] = override["bits"]
            if "group_size" in override:
                spec["group_size"] = override["group_size"]
            if "mode" in override and isinstance(override["mode"], str):
                spec["mode"] = override["mode"].strip().lower()
            break
    return spec


def _infer_expected_module_in_features(
    module_name: str,
    text_cfg: dict[str, Any],
    *,
    head_dim: int | None,
    shared_inter_override: int | None = None,
) -> int | None:
    hidden = text_cfg.get("hidden_size")
    hidden_i = hidden if isinstance(hidden, int) and not isinstance(hidden, bool) and hidden > 0 else None
    inter = text_cfg.get("intermediate_size")
    inter_i = inter if isinstance(inter, int) and not isinstance(inter, bool) and inter > 0 else None
    moe_inter = text_cfg.get("moe_intermediate_size")
    moe_inter_i = (
        moe_inter
        if isinstance(moe_inter, int) and not isinstance(moe_inter, bool) and moe_inter > 0
        else inter_i
    )
    q_heads = text_cfg.get("num_attention_heads")
    q_heads_i = (
        q_heads if isinstance(q_heads, int) and not isinstance(q_heads, bool) and q_heads > 0 else None
    )
    probe = module_name if module_name.startswith(".") else f".{module_name}"
    if probe.endswith(".embed_tokens") or probe.endswith(".lm_head"):
        return hidden_i
    if ".self_attn." in probe:
        if probe.endswith((".self_attn.q_proj", ".self_attn.k_proj", ".self_attn.v_proj")):
            return hidden_i
        if probe.endswith(".self_attn.o_proj") and q_heads_i is not None and head_dim is not None:
            return q_heads_i * head_dim
    if ".linear_attn." in probe:
        if probe.endswith(
            (
                ".linear_attn.in_proj_qkv",
                ".linear_attn.in_proj_z",
                ".linear_attn.in_proj_a",
                ".linear_attn.in_proj_b",
            )
        ):
            return hidden_i
        if probe.endswith(".linear_attn.out_proj"):
            v_heads = text_cfg.get("linear_num_value_heads")
            v_dim = text_cfg.get("linear_value_head_dim")
            if (
                isinstance(v_heads, int)
                and not isinstance(v_heads, bool)
                and v_heads > 0
                and isinstance(v_dim, int)
                and not isinstance(v_dim, bool)
                and v_dim > 0
            ):
                return v_heads * v_dim
    if ".mlp." in probe:
        if ".shared_expert." in probe or ".shared_experts." in probe:
            if probe.endswith((".gate_proj", ".up_proj")):
                return hidden_i
            if probe.endswith(".down_proj"):
                raw_sh = text_cfg.get("shared_expert_intermediate_size")
                if isinstance(raw_sh, int) and not isinstance(raw_sh, bool) and raw_sh > 0:
                    return raw_sh
                return shared_inter_override
        elif ".switch_mlp." in probe or ".experts." in probe:
            if probe.endswith((".gate_proj", ".up_proj")):
                return hidden_i
            if probe.endswith(".down_proj"):
                return moe_inter_i
        else:
            if probe.endswith((".mlp.gate", ".mlp.shared_expert_gate", ".mlp.gate_proj", ".mlp.up_proj")):
                return hidden_i
            if probe.endswith(".mlp.down_proj"):
                return inter_i
    if ".mtp." in probe and probe.endswith(".mtp.fc") and hidden_i is not None:
        return 2 * hidden_i
    return None


def _config_matches_converter_quant(
    cfg: dict[str, Any],
    converter_spec: dict[str, Any],
    *,
    is_builtin: bool,
) -> tuple[bool, str]:
    """Verify `config.json` matches the converter's supported quantization modes."""
    raw_modes = converter_spec.get("supported_quant_modes")
    if not isinstance(raw_modes, (list, tuple)) or not raw_modes:
        return True, ""
    allowed_modes = {str(m).strip().lower() for m in raw_modes if isinstance(m, str) and m.strip()}
    if not allowed_modes:
        return True, ""

    quant, qcfg = _extract_quant_dicts(cfg)
    qmethod = str(
        qcfg.get("quant_method") or quant.get("quant_method") or ""
    ).strip().lower()
    mode = str(quant.get("mode") or "").strip().lower()
    fmt = str(qcfg.get("format") or quant.get("format") or "").strip().lower()

    for bad_m, label in (
        ("awq", "AWQ"),
        ("gptq", "GPTQ"),
        ("fp8", "FP8"),
        ("fbgemm_fp8", "FP8"),
        ("bitsandbytes", "bitsandbytes"),
        ("bnb", "bitsandbytes"),
        ("gguf", "GGUF"),
    ):
        if qmethod == bad_m or mode == bad_m or fmt == bad_m:
            return False, f"{label} quantization is not supported by {M5RT_ENGINE_LABEL}"

    has_affine_fields = (
        mode == "affine"
        or (not mode and not fmt and (quant.get("bits") is not None or quant.get("group_size") is not None))
    )
    if has_affine_fields and "affine" in allowed_modes:
        if is_builtin:
            bits = quant.get("bits")
            gs = quant.get("group_size")
            if bits not in (3, 4, 5, 6, 8) or gs not in (64, 128):
                return (
                    False,
                    f"affine quantization bits={bits!r}, group_size={gs!r} is not supported "
                    "(expected bits in {3,4,5,6,8} and group_size in {64,128})",
                )
            for k, v in quant.items():
                if not isinstance(v, dict):
                    continue
                ov_mode = str(v.get("mode") or mode or "affine").strip().lower()
                if ov_mode == "mxfp4":
                    if v.get("group_size", 32) != 32:
                        return False, f"MXFP4 override on {k!r} requires group_size=32"
                    continue
                ov_bits = v.get("bits", bits)
                ov_gs = v.get("group_size", gs)
                if ov_bits not in (3, 4, 5, 6, 8) or ov_gs not in (64, 128):
                    return (
                        False,
                        f"affine quantization override on {k!r} (bits={ov_bits!r}, "
                        f"group_size={ov_gs!r}) is not supported (expected bits in "
                        "{3,4,5,6,8} and group_size in {64,128})",
                    )
        return True, ""

    if mode == "mxfp4" and "mxfp4" in allowed_modes:
        if is_builtin and quant.get("group_size") not in (None, 32):
            return (
                False,
                f"MXFP4 quantization group_size={quant.get('group_size')!r} is not supported "
                "(expected group_size=32)",
            )
        return True, ""

    if fmt == "mxfp4-pack-quantized" and "mxfp4-pack-quantized" in allowed_modes:
        return True, ""

    if not mode and not fmt and quant.get("bits") is None and quant.get("group_size") is None:
        if {"bf16", "fp16", "mixed", "unquantized"} & allowed_modes:
            return True, ""

    return (
        False,
        f"checkpoint quantization is not in {sorted(allowed_modes)} "
        f"(mode={mode or 'none'!r}, format={fmt or 'none'!r})",
    )


def inspect_m5rt_source_fast(
    model_dir: str | Path,
    *,
    target_model_dir: str | Path | None = None,
    target_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Built-in `< 5 ms` O(1) zero-weight-read Two-Layer Structural & Per-Tensor Probe.

    Reads ONLY `config.json`, optional `model.safetensors.index.json`, and the 8-byte
    header size + JSON header of `.safetensors` shards (never loading weight payload bytes).
    """
    t0 = time.perf_counter()
    mdir = Path(model_dir).expanduser()

    def _finish(
        eligible: bool,
        *,
        blocker_code: str | None = None,
        reason: str = "",
        is_dflash2_draft: bool = False,
        model_type: str = "",
        text_model_type: str = "",
        model_type_hint_matched: bool = False,
        topology: str | None = None,
        head_dim: int | None = None,
        has_attn_output_gate: bool | None = None,
        has_qk_norm: bool | None = None,
        linear_attention_layers: int = 0,
        full_attention_layers: int = 0,
        num_experts: int = 0,
        num_experts_per_tok: int = 0,
        has_shared_expert: bool = False,
        tie_word_embeddings: bool = False,
        quant_modes: list[str] | None = None,
        skipped_non_text_tensors: list[str] | None = None,
        companion_mtp_tensors: list[str] | None = None,
        repack_counts: dict[str, int] | None = None,
        first_blocker: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        modes = sorted(set(quant_modes or ()))
        return {
            "eligible": eligible,
            "blocker_code": blocker_code,
            "reason": reason,
            "is_dflash2_draft": is_dflash2_draft,
            "model_type": model_type,
            "text_model_type": text_model_type,
            "model_type_hint_matched": model_type_hint_matched,
            "topology": topology,
            "head_dim": head_dim,
            "kv_dtype_default": "affine4_g128" if head_dim == 256 else "bf16",
            "has_attn_output_gate": has_attn_output_gate,
            "has_qk_norm": has_qk_norm,
            "linear_attention_layers": linear_attention_layers,
            "full_attention_layers": full_attention_layers,
            "num_experts": num_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "has_shared_expert": has_shared_expert,
            "tie_word_embeddings": tie_word_embeddings,
            "quant_modes": modes,
            "is_mixed_precision": len(modes) > 1,
            "skipped_non_text_tensors": skipped_non_text_tensors or [],
            "companion_mtp_tensors": companion_mtp_tensors or [],
            "repack_counts": repack_counts
            or {
                "nax_pretile": 0,
                "rowsplit_or_direct": 0,
                "companion_mtp": 0,
                "skip_non_text": 0,
                "incompatible_blocker": 0 if eligible else 1,
            },
            "first_blocker": first_blocker,
            "elapsed_ms": round((time.perf_counter() - t0) * 1000.0, 4),
        }

    if not mdir.is_dir():
        return _finish(
            False,
            blocker_code="missing_model_dir",
            reason=f"model directory not found: {mdir}",
        )

    cfg_path = mdir / "config.json"
    if not cfg_path.is_file():
        return _finish(False, blocker_code="missing_config", reason="config.json not found")
    cfg = _read_json(cfg_path)
    if cfg is None:
        return _finish(
            False,
            blocker_code="invalid_config",
            reason="config.json is not a valid JSON object",
        )

    text_cfg = _text_config(cfg)
    raw_mt = str(cfg.get("model_type") or "").strip()
    raw_text_mt = str(text_cfg.get("model_type") or raw_mt).strip()
    mt_tokens = _extract_model_type_tokens(cfg)
    hint_matched = bool(mt_tokens & {m.lower() for m in M5RT_SUPPORTED_MODEL_TYPES})
    tie_embed_raw = (
        text_cfg.get("tie_word_embeddings")
        if text_cfg.get("tie_word_embeddings") is not None
        else cfg.get("tie_word_embeddings", False)
    )
    tie_word_embeddings = bool(tie_embed_raw)

    # DFlash2 Draft Mode
    is_draft_arch, _ = _is_dflash_family_architecture(cfg)
    if is_draft_arch:
        eff_target_cfg = target_config
        if eff_target_cfg is None and target_model_dir is not None:
            eff_target_cfg = _read_json(Path(target_model_dir).expanduser() / "config.json")
        ok_df, df_reason = _is_supported_dflash2_draft_config(
            cfg, is_builtin=True, target_config=eff_target_cfg
        )
        if not ok_df:
            return _finish(
                False,
                blocker_code="unsupported_dflash2_draft",
                reason=df_reason,
                is_dflash2_draft=True,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
            )
        tensors, st_code, st_err = _read_safetensors_headers_fast(mdir)
        if st_code is not None:
            return _finish(
                False,
                blocker_code=st_code,
                reason=st_err,
                is_dflash2_draft=True,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
            )
        repack_counts = {
            "nax_pretile": 0,
            "rowsplit_or_direct": 0,
            "companion_mtp": 0,
            "skip_non_text": 0,
            "incompatible_blocker": 0,
        }
        for tname in sorted(tensors):
            tdtype, tshape = tensors[tname]
            if tdtype != "BF16":
                reason = (
                    f"DFlash2 draft requires all-BF16 tensors; found {tname!r} "
                    f"with dtype={tdtype!r}, shape={list(tshape)}"
                )
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="unsupported_dflash2_dtype",
                    reason=reason,
                    is_dflash2_draft=True,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    repack_counts=repack_counts,
                    first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
                )
            repack_counts["rowsplit_or_direct"] += 1
        return _finish(
            True,
            is_dflash2_draft=True,
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            topology="dflash2_draft",
            head_dim=_effective_head_dim(cfg),
            quant_modes=["bf16"],
            repack_counts=repack_counts,
        )

    tensors, st_code, st_err = _read_safetensors_headers_fast(mdir)

    skipped_non_text: list[str] = []
    companion_mtp: list[str] = []
    text_tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    active_tensors: dict[str, tuple[str, tuple[int, ...]]] = {}
    for tname in sorted(tensors):
        domain = _classify_source_tensor_domain(tname)
        if domain == "skip_non_text":
            skipped_non_text.append(tname)
        elif domain == "companion_mtp":
            companion_mtp.append(tname)
            active_tensors[tname] = tensors[tname]
        else:
            text_tensors[tname] = tensors[tname]
            active_tensors[tname] = tensors[tname]

    base_counts = {
        "nax_pretile": 0,
        "rowsplit_or_direct": 0,
        "companion_mtp": 0,
        "skip_non_text": len(skipped_non_text),
        "incompatible_blocker": 0,
    }

    # Layer 1A: Immediate topology blockers (MLA, Mamba/SSM, unsupported layer_types/activations, linear .bias)
    for tname, (_tdtype, tshape) in text_tensors.items():
        if any(
            mla_tok in tname
            for mla_tok in ("kv_a_proj_with_mqa", "kv_b_proj", "q_a_proj", "q_b_proj")
        ):
            reason = (
                f"MLA (Multi-head Latent Attention) topology is not supported by "
                f"{M5RT_ENGINE_LABEL} (found tensor {tname!r} with shape={list(tshape)})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_mla",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
    for mla_key in (
        "kv_lora_rank",
        "q_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    ):
        if text_cfg.get(mla_key) is not None or cfg.get(mla_key) is not None:
            reason = (
                f"MLA (Multi-head Latent Attention) architecture is not supported by "
                f"{M5RT_ENGINE_LABEL} (found config key {mla_key!r})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_mla",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )

    for tname, (_tdtype, tshape) in text_tensors.items():
        if ".mamba." in tname or ".ssm." in tname:
            reason = (
                f"Mamba/SSM state-space topology is not supported by {M5RT_ENGINE_LABEL} "
                f"(found tensor {tname!r} with shape={list(tshape)})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_mamba",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )

    raw_layer_types = text_cfg.get("layer_types")
    if isinstance(raw_layer_types, list):
        for idx, lt in enumerate(raw_layer_types):
            lt_s = str(lt or "").strip()
            if lt_s not in ("full_attention", "linear_attention"):
                code = (
                    "unsupported_mamba"
                    if ("mamba" in lt_s.lower() or "ssm" in lt_s.lower())
                    else "unsupported_layer_type"
                )
                label = "Mamba/SSM" if code == "unsupported_mamba" else "Unsupported"
                reason = (
                    f"{label} layer_type {lt_s!r} at layer {idx} is not supported by "
                    f"{M5RT_ENGINE_LABEL} (supported: 'full_attention', 'linear_attention')"
                )
                base_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code=code,
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=base_counts,
                )

    for ssm_key in ("mamba_d_state", "ssm_state_size", "mamba_d_conv"):
        val = text_cfg.get(ssm_key) if text_cfg.get(ssm_key) is not None else cfg.get(ssm_key)
        if isinstance(val, int) and not isinstance(val, bool) and val > 0:
            reason = (
                f"Mamba/SSM architecture is not supported by {M5RT_ENGINE_LABEL} "
                f"(found config key {ssm_key!r}={val})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_mamba",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )

    hidden_act = text_cfg.get("hidden_act") or text_cfg.get("hidden_activation")
    if isinstance(hidden_act, str) and hidden_act.strip():
        if hidden_act.strip().lower() not in ("silu", "swiglu"):
            reason = (
                f"Unsupported hidden_act={hidden_act!r} "
                f"({M5RT_ENGINE_LABEL} requires 'silu' / SwiGLU)"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_activation",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )

    for tname, (_tdtype, tshape) in active_tensors.items():
        if tname.endswith(".bias") and not tname.endswith(".dt_bias"):
            reason = (
                f"Additive linear bias tensor {tname!r} (shape={list(tshape)}) is not "
                f"supported by {M5RT_ENGINE_LABEL}"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_linear_bias",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )

    # Layer 2A: Immediate per-tensor & config quantization method blockers (AWQ, GPTQ, FP8, BnB)
    for tname, (tdtype, tshape) in active_tensors.items():
        if tname.endswith((".qweight", ".qzeros")):
            reason = (
                f"AWQ quantization is not supported by {M5RT_ENGINE_LABEL} "
                f"(found tensor {tname!r} with shape={list(tshape)})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_quant_awq",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
        if tname.endswith(".g_idx"):
            reason = (
                f"GPTQ quantization is not supported by {M5RT_ENGINE_LABEL} "
                f"(found tensor {tname!r} with shape={list(tshape)})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_quant_gptq",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
        if tname.endswith(".weight_scale_inv") or tdtype in ("F8_E4M3", "F8_E5M2"):
            reason = (
                f"FP8 quantization is not supported by {M5RT_ENGINE_LABEL} "
                f"(found tensor {tname!r} with dtype={tdtype!r}, shape={list(tshape)})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_quant_fp8",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
        if tname.endswith((".absmax", ".quant_state", ".SCB")):
            reason = (
                f"bitsandbytes quantization is not supported by {M5RT_ENGINE_LABEL} "
                f"(found tensor {tname!r} with shape={list(tshape)})"
            )
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_quant_bnb",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )

    quant_dict, qcfg_dict = _extract_quant_dicts(cfg)
    qmethod_raw = str(
        qcfg_dict.get("quant_method")
        or quant_dict.get("quant_method")
        or quant_dict.get("mode")
        or qcfg_dict.get("format")
        or ""
    ).strip().lower()
    for bad_m, b_code, b_label in (
        ("awq", "unsupported_quant_awq", "AWQ"),
        ("gptq", "unsupported_quant_gptq", "GPTQ"),
        ("fp8", "unsupported_quant_fp8", "FP8"),
        ("fbgemm_fp8", "unsupported_quant_fp8", "FP8"),
        ("bitsandbytes", "unsupported_quant_bnb", "bitsandbytes"),
        ("bnb", "unsupported_quant_bnb", "bitsandbytes"),
        ("gguf", "unsupported_gguf", "GGUF"),
    ):
        if qmethod_raw == bad_m:
            first_t = next(iter(active_tensors.items()), None)
            fb = (
                {
                    "tensor": first_t[0],
                    "shape": list(first_t[1][1]),
                    "reason": f"{b_label} quantization is not supported",
                }
                if first_t is not None
                else None
            )
            detail = (
                f" (first tensor {first_t[0]!r} shape={list(first_t[1][1])})"
                if first_t is not None
                else ""
            )
            reason = f"{b_label} quantization is not supported by {M5RT_ENGINE_LABEL}{detail}"
            base_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code=b_code,
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
                first_blocker=fb,
            )

    if st_code is not None:
        return _finish(
            False,
            blocker_code=st_code,
            reason=st_err,
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
        )

    # Synthetic empty `{}` safetensors header support for minimal mock tests
    if not tensors:
        q_ok, q_reason = _config_matches_converter_quant(
            cfg, M5RT_DEFAULT_CONVERTER, is_builtin=True
        )
        if not q_ok:
            return _finish(
                False,
                blocker_code="unsupported_quant_config",
                reason=q_reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
            )
        has_explicit_quant = bool(
            quant_dict.get("mode")
            or quant_dict.get("bits") is not None
            or qcfg_dict.get("format")
        )
        if hint_matched and has_explicit_quant:
            # Validate any geometry fields present on mock config
            raw_h = text_cfg.get("hidden_size")
            if isinstance(raw_h, int) and not isinstance(raw_h, bool):
                if raw_h < 64 or raw_h % 64 != 0:
                    return _finish(
                        False,
                        blocker_code="invalid_hidden_size",
                        reason=f"hidden_size={raw_h} must be >= 64 and divisible by 64",
                        model_type=raw_mt,
                        text_model_type=raw_text_mt,
                        model_type_hint_matched=hint_matched,
                    )
            return _finish(
                True,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                topology="gdn_hybrid_gqa",
                head_dim=_effective_head_dim(text_cfg),
                tie_word_embeddings=tie_word_embeddings,
                quant_modes=[str(quant_dict.get("mode") or qcfg_dict.get("format") or "affine")],
            )
        return _finish(
            False,
            blocker_code="incomplete_checkpoint_metadata",
            reason="checkpoint has empty .safetensors header and incomplete structural config",
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
        )

    # Layer 1B: Full Structural Geometry Validation (matching C++ QwenForward::supports_config)
    def _pos_int(obj: dict[str, Any], key: str) -> int | None:
        v = obj.get(key)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v
        return None

    hidden_size = _pos_int(text_cfg, "hidden_size")
    if hidden_size is None or hidden_size < 64 or hidden_size % 64 != 0:
        return _finish(
            False,
            blocker_code="invalid_hidden_size",
            reason=(
                f"Unsupported hidden_size={text_cfg.get('hidden_size')!r} "
                "(must be >= 64 and divisible by 64)"
            ),
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )

    num_hidden_layers = _pos_int(text_cfg, "num_hidden_layers")
    if num_hidden_layers is None:
        return _finish(
            False,
            blocker_code="invalid_num_hidden_layers",
            reason=f"Unsupported num_hidden_layers={text_cfg.get('num_hidden_layers')!r} (must be >= 1)",
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )

    vocab_size = _pos_int(text_cfg, "vocab_size") or _pos_int(cfg, "vocab_size")
    if vocab_size is None:
        return _finish(
            False,
            blocker_code="invalid_vocab_size",
            reason=f"Unsupported vocab_size={text_cfg.get('vocab_size')!r} (must be > 0)",
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )

    raw_experts = (
        text_cfg.get("num_experts")
        if text_cfg.get("num_experts") is not None
        else (
            text_cfg.get("n_routed_experts")
            if text_cfg.get("n_routed_experts") is not None
            else text_cfg.get("num_local_experts", 0)
        )
    )
    if isinstance(raw_experts, bool) or not isinstance(raw_experts, int) or raw_experts < 0:
        return _finish(
            False,
            blocker_code="invalid_moe_geometry",
            reason=f"Invalid MoE num_experts={raw_experts!r}",
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )
    num_experts = raw_experts

    inter_size = _pos_int(text_cfg, "intermediate_size")
    moe_inter_size = _pos_int(text_cfg, "moe_intermediate_size")
    eff_inter = (moe_inter_size or inter_size) if num_experts > 0 else (inter_size or moe_inter_size)
    if eff_inter is None or eff_inter < 64 or eff_inter % 64 != 0:
        return _finish(
            False,
            blocker_code="invalid_intermediate_size",
            reason=(
                f"Unsupported intermediate_size={eff_inter!r} "
                "(must be >= 64 and divisible by 64)"
            ),
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )

    num_q_heads = _pos_int(text_cfg, "num_attention_heads")
    num_kv_heads = _pos_int(text_cfg, "num_key_value_heads")
    if num_q_heads is None or num_kv_heads is None or num_q_heads % num_kv_heads != 0:
        return _finish(
            False,
            blocker_code="invalid_attention_heads",
            reason=(
                f"Invalid GQA head geometry: num_attention_heads={text_cfg.get('num_attention_heads')!r}, "
                f"num_key_value_heads={text_cfg.get('num_key_value_heads')!r}"
            ),
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )

    if text_cfg.get("head_dim") is not None:
        raw_hd = text_cfg.get("head_dim")
        if isinstance(raw_hd, bool) or not isinstance(raw_hd, int):
            head_dim = None
        else:
            head_dim = raw_hd
    elif hidden_size % num_q_heads == 0:
        head_dim = hidden_size // num_q_heads
    else:
        head_dim = None
    if head_dim not in (64, 128, 256):
        return _finish(
            False,
            blocker_code="invalid_head_dim",
            reason=f"Unsupported head_dim={head_dim!r} (expected one of 64, 128, 256)",
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )

    # RoPE validation (supports rope_parameters object and top-level rope_theta / partial_rotary_factor)
    rope_obj = text_cfg.get("rope_parameters")
    if not isinstance(rope_obj, dict):
        rope_obj = cfg.get("rope_parameters") if isinstance(cfg.get("rope_parameters"), dict) else {}
    raw_theta = (
        rope_obj.get("rope_theta")
        if rope_obj.get("rope_theta") is not None
        else (
            text_cfg.get("rope_theta")
            if text_cfg.get("rope_theta") is not None
            else cfg.get("rope_theta", 10000.0)
        )
    )
    raw_prf = (
        rope_obj.get("partial_rotary_factor")
        if rope_obj.get("partial_rotary_factor") is not None
        else (
            text_cfg.get("partial_rotary_factor")
            if text_cfg.get("partial_rotary_factor") is not None
            else cfg.get("partial_rotary_factor", 1.0)
        )
    )
    try:
        rope_theta = float(raw_theta)
        partial_rotary = float(raw_prf)
    except (TypeError, ValueError):
        rope_theta = -1.0
        partial_rotary = -1.0
    rotary_dim = int(round(head_dim * partial_rotary)) if partial_rotary > 0 else 0
    if (
        not math.isfinite(rope_theta)
        or rope_theta <= 0.0
        or not math.isfinite(partial_rotary)
        or not (0.0 < partial_rotary <= 1.0)
        or rotary_dim < 2
        or rotary_dim % 2 != 0
    ):
        return _finish(
            False,
            blocker_code="invalid_rope",
            reason=(
                f"Invalid RoPE configuration (rope_theta={raw_theta!r}, "
                f"partial_rotary_factor={raw_prf!r})"
            ),
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            head_dim=head_dim,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=base_counts,
        )

    # Layer types resolution (supports explicit layer_types or implicit 0-GDN pure GQA / tensor-inferred GDN)
    if isinstance(raw_layer_types, list):
        if len(raw_layer_types) != num_hidden_layers:
            return _finish(
                False,
                blocker_code="invalid_layer_types_length",
                reason=(
                    f"layer_types length ({len(raw_layer_types)}) does not match "
                    f"num_hidden_layers ({num_hidden_layers})"
                ),
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )
        layer_types = [str(x) for x in raw_layer_types]
    else:
        has_any_gdn_tensor = any(".linear_attn." in ("." + n) for n in text_tensors)
        if has_any_gdn_tensor:
            layer_types = [
                (
                    "linear_attention"
                    if any(f".layers.{idx}.linear_attn." in ("." + n) for n in text_tensors)
                    else "full_attention"
                )
                for idx in range(num_hidden_layers)
            ]
        else:
            layer_types = ["full_attention"] * num_hidden_layers

    linear_layers = sum(1 for lt in layer_types if lt == "linear_attention")
    full_layers = sum(1 for lt in layer_types if lt == "full_attention")

    # GDN geometry validation (ONLY required when linear_attention_layers > 0!)
    if linear_layers > 0:
        lin_kh = _pos_int(text_cfg, "linear_num_key_heads")
        lin_vh = _pos_int(text_cfg, "linear_num_value_heads")
        lin_kd = _pos_int(text_cfg, "linear_key_head_dim")
        lin_vd = _pos_int(text_cfg, "linear_value_head_dim")
        lin_conv = _pos_int(text_cfg, "linear_conv_kernel_dim")
        if (
            lin_kh is None
            or lin_vh is None
            or lin_vh % lin_kh != 0
            or lin_kd is None
            or lin_vd is None
            or lin_conv is None
            or lin_conv < 2
        ):
            return _finish(
                False,
                blocker_code="invalid_gdn_geometry",
                reason=(
                    "Invalid GDN linear_attention head geometry in config.json "
                    f"(linear_num_key_heads={lin_kh!r}, linear_num_value_heads={lin_vh!r}, "
                    f"linear_key_head_dim={lin_kd!r}, linear_value_head_dim={lin_vd!r}, "
                    f"linear_conv_kernel_dim={lin_conv!r})"
                ),
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                linear_attention_layers=linear_layers,
                full_attention_layers=full_layers,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )

    # MoE geometry validation
    num_experts_per_tok = 0
    has_shared_expert = False
    inferred_shared_inter: int | None = None
    if num_experts != 0:
        if not (2 <= num_experts <= 256):
            return _finish(
                False,
                blocker_code="invalid_moe_geometry",
                reason=(
                    f"MoE num_experts={num_experts} is outside {M5RT_ENGINE_LABEL} "
                    "supported range [2, 256]"
                ),
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                linear_attention_layers=linear_layers,
                full_attention_layers=full_layers,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )
        raw_topk = (
            text_cfg.get("num_experts_per_tok")
            if text_cfg.get("num_experts_per_tok") is not None
            else text_cfg.get("num_experts_per_token", 0)
        )
        if (
            isinstance(raw_topk, bool)
            or not isinstance(raw_topk, int)
            or not (1 <= raw_topk <= 8)
            or raw_topk > num_experts
        ):
            return _finish(
                False,
                blocker_code="invalid_moe_geometry",
                reason=(
                    f"MoE num_experts_per_tok={raw_topk!r} is outside {M5RT_ENGINE_LABEL} "
                    f"supported range [1, min(8, num_experts={num_experts})]"
                ),
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                linear_attention_layers=linear_layers,
                full_attention_layers=full_layers,
                num_experts=num_experts,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )
        num_experts_per_tok = raw_topk

        raw_shared_inter = text_cfg.get("shared_expert_intermediate_size", 0) or 0
        if isinstance(raw_shared_inter, bool) or not isinstance(raw_shared_inter, int) or raw_shared_inter < 0:
            return _finish(
                False,
                blocker_code="invalid_moe_geometry",
                reason=f"Invalid shared_expert_intermediate_size={raw_shared_inter!r}",
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )
        if raw_shared_inter > 0 and (raw_shared_inter < 64 or raw_shared_inter % 64 != 0):
            return _finish(
                False,
                blocker_code="invalid_moe_geometry",
                reason=(
                    f"shared_expert_intermediate_size={raw_shared_inter} must be 0 "
                    "or a multiple of 64"
                ),
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=base_counts,
            )
        for tname, (_td, tsh) in text_tensors.items():
            if (
                (".shared_expert.gate_proj." in ("." + tname) or ".shared_experts.gate_proj." in ("." + tname))
                and len(tsh) >= 2
            ):
                inferred_shared_inter = tsh[0]
                break
        if inferred_shared_inter is not None:
            if inferred_shared_inter < 64 or inferred_shared_inter % 64 != 0:
                return _finish(
                    False,
                    blocker_code="invalid_moe_geometry",
                    reason=(
                        f"shared_expert gate_proj output dim={inferred_shared_inter} "
                        "must be >= 64 and divisible by 64"
                    ),
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=base_counts,
                )
            has_shared_expert = True
        else:
            has_shared_expert = raw_shared_inter > 0

    # Layer 2B: Per-Tensor Quantization & Repack Classification
    consumed_tensors: set[str] = set()
    quant_modes: list[str] = []
    repack_counts = dict(base_counts)

    def _record_module_category(module_name: str, shape: tuple[int, ...], fmt_kind: str, logical_k: int, gs: int) -> None:
        if module_name.startswith("mtp.") or ".mtp." in module_name:
            repack_counts["companion_mtp"] += 1
            return
        probe = module_name if module_name.startswith(".") else f".{module_name}"
        is_embed = probe.endswith(".embed_tokens")
        is_router = probe.endswith((".mlp.gate", ".mlp.shared_expert_gate"))
        if len(shape) == 2 and not is_embed and not is_router and logical_k >= 512:
            if fmt_kind == "mxfp4" and logical_k % 128 == 0:
                repack_counts["nax_pretile"] += 1
                return
            if fmt_kind == "affine" and logical_k % gs == 0:
                repack_counts["nax_pretile"] += 1
                return
        elif len(shape) == 3 and logical_k >= 128:
            if fmt_kind == "mxfp4" and logical_k % 128 == 0:
                repack_counts["nax_pretile"] += 1
                return
            if fmt_kind == "affine" and logical_k % gs == 0:
                repack_counts["nax_pretile"] += 1
                return
        repack_counts["rowsplit_or_direct"] += 1

    # 1) Compressed MXFP4 modules (*.weight_packed + *.weight_scale)
    for tname in list(active_tensors):
        if not tname.endswith(".weight_packed"):
            continue
        mod = tname[: -len(".weight_packed")]
        sc_name = mod + ".weight_scale"
        sh_name = mod + ".weight_shape"
        p_dtype, p_shape = active_tensors[tname]
        if sc_name not in active_tensors:
            reason = f"Compressed MXFP4 tensor {tname!r} is missing companion {sc_name!r}"
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="invalid_mxfp4_module",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
                first_blocker={"tensor": tname, "shape": list(p_shape), "reason": reason},
            )
        s_dtype, s_shape = active_tensors[sc_name]
        logical_k = p_shape[-1] * 2 if len(p_shape) >= 2 else 0
        if (
            p_dtype not in ("U8", "I8")
            or s_dtype not in ("U8", "F8_E8M0")
            or len(p_shape) < 2
            or logical_k <= 0
            or logical_k % 32 != 0
            or s_shape != (*p_shape[:-1], logical_k // 32)
        ):
            reason = (
                f"Invalid compressed MXFP4 layout on {tname!r} "
                f"(dtype={p_dtype}, shape={list(p_shape)}, scale_shape={list(s_shape)})"
            )
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="invalid_mxfp4_module",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
                first_blocker={"tensor": tname, "shape": list(p_shape), "reason": reason},
            )
        consumed_tensors.update((tname, sc_name))
        if sh_name in active_tensors:
            consumed_tensors.add(sh_name)
        quant_modes.append("mxfp4-pack-quantized")
        _record_module_category(mod, (*p_shape[:-1], logical_k), "mxfp4", logical_k, 32)

    # 2) MLX quantized modules (*.weight + *.scales [+ *.biases])
    for tname in list(active_tensors):
        if tname in consumed_tensors or not tname.endswith(".weight"):
            continue
        mod = tname[: -len(".weight")]
        sc_name = mod + ".scales"
        bi_name = mod + ".biases"
        if sc_name not in active_tensors:
            continue
        w_dtype, w_shape = active_tensors[tname]
        s_dtype, s_shape = active_tensors[sc_name]
        expected_k = _infer_expected_module_in_features(
            mod, text_cfg, head_dim=head_dim, shared_inter_override=inferred_shared_inter
        )
        if bi_name in active_tensors:
            # Affine quantized module
            b_dtype, b_shape = active_tensors[bi_name]
            if (
                w_dtype != "U32"
                or s_dtype not in ("BF16", "F16")
                or b_dtype not in ("BF16", "F16")
                or len(w_shape) < 2
                or len(s_shape) != len(w_shape)
                or s_shape != b_shape
                or w_shape[:-1] != s_shape[:-1]
                or s_shape[-1] <= 0
            ):
                reason = (
                    f"Invalid affine quantization tensors on '{mod}' "
                    f"(weight={w_dtype}{list(w_shape)}, scales={s_dtype}{list(s_shape)}, "
                    f"biases={b_dtype}{list(b_shape)})"
                )
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="invalid_affine_tensors",
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=repack_counts,
                    first_blocker={"tensor": tname, "shape": list(w_shape), "reason": reason},
                )
            groups = s_shape[-1]
            total_row_bits = w_shape[-1] * 32
            spec = _module_quant_spec(cfg, mod)
            spec_bits = spec.get("bits")
            spec_gs = spec.get("group_size")
            bits: int | None = None
            gs: int | None = None
            logical_k = 0
            if (
                expected_k is not None
                and expected_k > 0
                and expected_k % groups == 0
                and total_row_bits % expected_k == 0
            ):
                gs = expected_k // groups
                bits = total_row_bits // expected_k
                logical_k = expected_k
            elif (
                isinstance(spec_bits, int)
                and not isinstance(spec_bits, bool)
                and spec_bits > 0
                and isinstance(spec_gs, int)
                and not isinstance(spec_gs, bool)
                and spec_gs > 0
                and groups * spec_gs * spec_bits == total_row_bits
            ):
                bits = spec_bits
                gs = spec_gs
                logical_k = groups * spec_gs
            elif (
                isinstance(spec_bits, int)
                and not isinstance(spec_bits, bool)
                and spec_bits > 0
                and total_row_bits % spec_bits == 0
                and (total_row_bits // spec_bits) % groups == 0
            ):
                bits = spec_bits
                logical_k = total_row_bits // spec_bits
                gs = logical_k // groups
            elif (
                isinstance(spec_gs, int)
                and not isinstance(spec_gs, bool)
                and spec_gs > 0
                and total_row_bits % (groups * spec_gs) == 0
            ):
                gs = spec_gs
                logical_k = groups * spec_gs
                bits = total_row_bits // logical_k
            else:
                for cand_gs in (64, 128, 32, 16, 256):
                    cand_k = groups * cand_gs
                    if cand_k > 0 and total_row_bits % cand_k == 0:
                        cand_b = total_row_bits // cand_k
                        if cand_b in (2, 3, 4, 5, 6, 8):
                            gs = cand_gs
                            bits = cand_b
                            logical_k = cand_k
                            break

            if bits is None or bits not in (3, 4, 5, 6, 8):
                reason = (
                    f"Unsupported affine quantization bits={bits!r} on tensor {tname!r} "
                    f"shape={list(w_shape)} (supported bits: 3, 4, 5, 6, 8)"
                )
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="unsupported_affine_bits",
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=repack_counts,
                    first_blocker={"tensor": tname, "shape": list(w_shape), "reason": reason},
                )
            if gs is None or gs not in (64, 128):
                reason = (
                    f"Unsupported affine quantization group_size={gs!r} on tensor {tname!r} "
                    f"shape={list(w_shape)} (supported group_size: 64, 128)"
                )
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="unsupported_affine_group_size",
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=repack_counts,
                    first_blocker={"tensor": tname, "shape": list(w_shape), "reason": reason},
                )
            consumed_tensors.update((tname, sc_name, bi_name))
            quant_modes.append(f"affine_q{bits}_g{gs}")
            _record_module_category(mod, (*s_shape[:-1], logical_k), "affine", logical_k, gs)
        else:
            # MXFP4 (U32 weight + U8/F8_E8M0 scales) or unsupported scale-only format
            if w_dtype == "U32" and s_dtype in ("U8", "F8_E8M0") and len(w_shape) >= 2 and len(s_shape) == len(w_shape) and w_shape[:-1] == s_shape[:-1] and s_shape[-1] > 0:
                groups = s_shape[-1]
                gs = (expected_k // groups) if (expected_k and expected_k % groups == 0) else 32
                logical_k = groups * gs
                if gs != 32 or w_shape[-1] * 8 != logical_k:
                    reason = (
                        f"Unsupported MXFP4 geometry (group_size={gs}) on tensor {tname!r} "
                        f"shape={list(w_shape)} (expected group_size=32)"
                    )
                    repack_counts["incompatible_blocker"] = 1
                    return _finish(
                        False,
                        blocker_code="unsupported_mxfp4_geometry",
                        reason=reason,
                        model_type=raw_mt,
                        text_model_type=raw_text_mt,
                        model_type_hint_matched=hint_matched,
                        head_dim=head_dim,
                        skipped_non_text_tensors=skipped_non_text,
                        companion_mtp_tensors=companion_mtp,
                        repack_counts=repack_counts,
                        first_blocker={"tensor": tname, "shape": list(w_shape), "reason": reason},
                    )
                consumed_tensors.update((tname, sc_name))
                quant_modes.append("mxfp4")
                _record_module_category(mod, (*s_shape[:-1], logical_k), "mxfp4", logical_k, 32)
            else:
                reason = (
                    f"Unsupported scale-only quantization on tensor {tname!r} "
                    f"(weight={w_dtype}{list(w_shape)}, scales={s_dtype}{list(s_shape)} without .biases)"
                )
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="unsupported_quant_format",
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=repack_counts,
                    first_blocker={"tensor": tname, "shape": list(w_shape), "reason": reason},
                )

    # 3) Direct unquantized tensors (norms, GDN scalars, BF16/FP16 embeddings/lm_head/layers)
    for tname, (tdtype, tshape) in active_tensors.items():
        if tname in consumed_tensors:
            continue
        probe_t = "." + tname.lstrip(".")
        if tname.endswith((".scales", ".biases", ".weight_scale")):
            reason = f"Orphaned quantization metadata tensor {tname!r} shape={list(tshape)}"
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="orphaned_quant_metadata",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
        if len(tshape) == 3 and (".switch_mlp." in probe_t or ".experts." in probe_t):
            reason = (
                f"Unquantized 3D MoE expert tensor {tname!r} (dtype={tdtype}, shape={list(tshape)}) "
                f"is not supported by {M5RT_ENGINE_LABEL} (MoE experts must be affine or MXFP4 quantized)"
            )
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_unquantized_moe_experts",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
        if ".mlp.experts." in probe_t and any(
            f".mlp.experts.{d}." in probe_t for d in ("0", "1")
        ):
            reason = (
                f"Unfused per-expert 2D MoE tensor {tname!r} shape={list(tshape)} is not supported "
                f"by {M5RT_ENGINE_LABEL} (expected 3D [E, N, K] switch_mlp/experts tensors)"
            )
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_unfused_moe_experts",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
        is_scalar_or_conv = tname.endswith((".A_log", ".dt_bias", ".conv1d.weight")) or (
            len(tshape) == 1 and tname.endswith("norm.weight")
        )
        allowed_dtypes = ("BF16", "F16", "F32") if is_scalar_or_conv else ("BF16", "F16")
        if tdtype not in allowed_dtypes:
            reason = (
                f"Unsupported tensor dtype={tdtype!r} on {tname!r} shape={list(tshape)} "
                f"(expected {allowed_dtypes})"
            )
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="unsupported_tensor_dtype",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
                first_blocker={"tensor": tname, "shape": list(tshape), "reason": reason},
            )
        if not is_scalar_or_conv and len(tshape) == 2 and not probe_t.endswith(
            (".mlp.gate.weight", ".mlp.shared_expert_gate.weight")
        ):
            quant_modes.append("bf16" if tdtype == "BF16" else "fp16")
        if tname.startswith("mtp.") or ".mtp." in tname:
            repack_counts["companion_mtp"] += 1
        else:
            repack_counts["rowsplit_or_direct"] += 1

    # Layer 1C: Attention Output Gate (`has_attn_output_gate`) & QK-Norm (`has_qk_norm`) Auto-Detection
    has_attn_output_gate: bool | None = None
    has_qk_norm: bool | None = None
    if full_layers > 0:
        q_ungated = num_q_heads * head_dim
        q_gated = 2 * q_ungated
        gate_flags: list[bool] = []
        qknorm_flags: list[bool] = []
        for l_idx, lt in enumerate(layer_types):
            if lt != "full_attention":
                continue
            q_marker = f".layers.{l_idx}.self_attn.q_proj."
            q_entry: tuple[str, tuple[int, ...]] | None = None
            for cand_suffix in ("weight", "weight_packed"):
                target_s = q_marker + cand_suffix
                for tname, (_td, tsh) in text_tensors.items():
                    if ("." + tname).endswith(target_s):
                        q_entry = (tname, tsh)
                        break
                if q_entry is not None:
                    break
            if q_entry is None:
                reason = f"Missing self_attn.q_proj tensor for full_attention layer {l_idx}"
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="missing_layer_tensor",
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=repack_counts,
                )
            q_tname, q_shape = q_entry
            n_q = q_shape[0] if q_shape else 0
            if n_q == q_gated:
                gate_flags.append(True)
            elif n_q == q_ungated:
                gate_flags.append(False)
            else:
                reason = (
                    f"Unsupported self_attn.q_proj output dim {n_q} on {q_tname!r} "
                    f"shape={list(q_shape)} (expected {q_ungated} for ungated GQA or "
                    f"{q_gated} for gated GQA)"
                )
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="invalid_q_proj_shape",
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=repack_counts,
                    first_blocker={"tensor": q_tname, "shape": list(q_shape), "reason": reason},
                )

            qn_suffix = f".layers.{l_idx}.self_attn.q_norm.weight"
            kn_suffix = f".layers.{l_idx}.self_attn.k_norm.weight"
            has_qn = any(("." + n).endswith(qn_suffix) for n in text_tensors)
            has_kn = any(("." + n).endswith(kn_suffix) for n in text_tensors)
            if has_qn != has_kn:
                reason = (
                    f"Inconsistent QK-norm tensors at layer {l_idx} "
                    f"(q_norm={has_qn}, k_norm={has_kn})"
                )
                repack_counts["incompatible_blocker"] = 1
                return _finish(
                    False,
                    blocker_code="inconsistent_qk_norm",
                    reason=reason,
                    model_type=raw_mt,
                    text_model_type=raw_text_mt,
                    model_type_hint_matched=hint_matched,
                    head_dim=head_dim,
                    skipped_non_text_tensors=skipped_non_text,
                    companion_mtp_tensors=companion_mtp,
                    repack_counts=repack_counts,
                )
            qknorm_flags.append(has_qn)

        if len(set(gate_flags)) > 1:
            reason = "Inconsistent self_attn.q_proj output gate across full-attention layers"
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="inconsistent_attn_output_gate",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
            )
        if len(set(qknorm_flags)) > 1:
            reason = "Inconsistent QK-norm presence across full-attention layers"
            repack_counts["incompatible_blocker"] = 1
            return _finish(
                False,
                blocker_code="inconsistent_qk_norm",
                reason=reason,
                model_type=raw_mt,
                text_model_type=raw_text_mt,
                model_type_hint_matched=hint_matched,
                head_dim=head_dim,
                skipped_non_text_tensors=skipped_non_text,
                companion_mtp_tensors=companion_mtp,
                repack_counts=repack_counts,
            )
        has_attn_output_gate = gate_flags[0]
        has_qk_norm = qknorm_flags[0]

    # Verify required global text backbone tensors
    has_embed = any(
        n.endswith(("embed_tokens.weight", "embed_tokens.weight_packed"))
        for n in text_tensors
    )
    has_final_norm = any(
        n == "norm.weight"
        or n.endswith("model.norm.weight")
        or (n.endswith(".norm.weight") and ".layers." not in ("." + n))
        for n in text_tensors
    )
    has_lm_head = tie_word_embeddings or any(
        n.endswith(("lm_head.weight", "lm_head.weight_packed")) for n in text_tensors
    )
    if not has_embed or not has_final_norm or not has_lm_head:
        missing_parts = []
        if not has_embed:
            missing_parts.append("embed_tokens")
        if not has_final_norm:
            missing_parts.append("model.norm.weight")
        if not has_lm_head:
            missing_parts.append("lm_head.weight")
        reason = (
            f"Checkpoint is missing required text backbone tensor(s): "
            f"{', '.join(missing_parts)}"
        )
        repack_counts["incompatible_blocker"] = 1
        return _finish(
            False,
            blocker_code="missing_backbone_tensors",
            reason=reason,
            model_type=raw_mt,
            text_model_type=raw_text_mt,
            model_type_hint_matched=hint_matched,
            head_dim=head_dim,
            skipped_non_text_tensors=skipped_non_text,
            companion_mtp_tensors=companion_mtp,
            repack_counts=repack_counts,
        )

    topology = "gdn_hybrid_gqa" if linear_layers > 0 else "pure_gqa_swiglu"
    return _finish(
        True,
        model_type=raw_mt,
        text_model_type=raw_text_mt,
        model_type_hint_matched=hint_matched,
        topology=topology,
        head_dim=head_dim,
        has_attn_output_gate=has_attn_output_gate,
        has_qk_norm=has_qk_norm,
        linear_attention_layers=linear_layers,
        full_attention_layers=full_layers,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        has_shared_expert=has_shared_expert,
        tie_word_embeddings=tie_word_embeddings,
        quant_modes=quant_modes or ["bf16"],
        skipped_non_text_tensors=skipped_non_text,
        companion_mtp_tensors=companion_mtp,
        repack_counts=repack_counts,
    )


def find_provider_converter_tool_fast(
    binary_path: Path | None,
    manifest: dict[str, Any] | None = None,
    *,
    is_builtin: bool = False,
) -> Path | None:
    """Locate a provider's bundled converter script in O(1) without spawning subprocesses."""
    if is_builtin:
        for env_key in ("NINFER_METAL_ARTIFACT_TOOL", "OMLX_M5_ARTIFACT_TOOL"):
            raw_env = os.environ.get(env_key, "").strip()
            if raw_env:
                p = Path(raw_env).expanduser()
                if p.is_file():
                    return p

    spec = _converter_spec_for_manifest(manifest, is_builtin=is_builtin)
    if spec is None:
        return None
    rel_entry = str(spec.get("entrypoint") or "").strip()
    if not rel_entry:
        return None

    rel_path = Path(rel_entry)
    if rel_path.is_absolute():
        return rel_path if rel_path.is_file() else None

    if isinstance(manifest, dict) and manifest.get("_provider_dir"):
        pdir_cand = Path(str(manifest["_provider_dir"])) / rel_path
        if pdir_cand.is_file():
            return pdir_cand

    if binary_path is not None:
        for base in (binary_path.resolve().parent.parent, binary_path.parent.parent):
            cand = base / rel_path
            if cand.is_file():
                return cand

    if is_builtin:
        repo_cand = _REPO_ROOT / "miv" / "tools" / "m5_artifact.py"
        if repo_cand.is_file():
            if binary_path is not None:
                with suppress(OSError, ValueError):
                    if binary_path.resolve().is_relative_to(_REPO_ROOT.resolve()):
                        return repo_cand
            elif not os.environ.get("OMLX_PROVIDERS_DIR", "").strip():
                return repo_cand
    return None


def _find_m5rt_binary_path_fast(configured_binary: str | None = None) -> tuple[Path | None, dict[str, Any] | None]:
    """Fast O(1) resolution of `(binary_path, provider_manifest)` with zero subprocess spawn."""
    if configured_binary and str(configured_binary).strip():
        cand = Path(str(configured_binary).strip()).expanduser()
        if _executable(cand):
            manifest = None
            for parent in (cand.resolve().parent, cand.resolve().parent.parent):
                manifest = _read_provider_manifest(parent)
                if manifest is not None:
                    break
            return cand.resolve(), manifest
        return None, None
    for env_key in ("NINFER_METAL_BIN", "OMLX_M5RT_BIN"):
        env_bin = os.environ.get(env_key, "").strip()
        if env_bin:
            cand = Path(env_bin).expanduser()
            if _executable(cand):
                manifest = None
                for parent in (cand.resolve().parent, cand.resolve().parent.parent):
                    manifest = _read_provider_manifest(parent)
                    if manifest is not None:
                        break
                return cand.resolve(), manifest
            return None, None
    prov_cands = _discover_provider_candidates()
    if prov_cands:
        return prov_cands[0]
    path_cand = _find_in_path(("mseries_backendd", "mseries-backendd"))
    if path_cand is not None:
        return path_cand.resolve(), None
    repo_daemons = _iter_repo_build_daemons()
    if repo_daemons:
        return repo_daemons[0], None
    return None, None


def find_m5_artifact_tool(configured_binary: str | None = None) -> Path | None:
    """Locate `m5_artifact.py` shipped inside the installed `miv` provider or monorepo."""
    bin_path, inst_manifest = _find_m5rt_binary_path_fast(configured_binary)
    tool = find_provider_converter_tool_fast(
        bin_path, inst_manifest or _builtin_m5rt_manifest(), is_builtin=True
    )
    if tool is not None:
        return tool
    # Fallback for explicit custom `configured_binary` wrapper scripts in tests/dev
    with suppress(Exception):
        repo_cand = _REPO_ROOT / "miv" / "tools" / "m5_artifact.py"
        if repo_cand.is_file() and not os.environ.get("OMLX_PROVIDERS_DIR", "").strip():
            return repo_cand
    return None


def can_auto_convert_provider_fast(
    model_path: str | Path,
    *,
    binary_path: Path | None,
    manifest: dict[str, Any],
    model_settings: object | None = None,
    is_builtin: bool = False,
) -> bool:
    """Pure O(1) zero-subprocess check whether `model_path` can auto-repack on first Load."""
    if not model_path or binary_path is None:
        return False
    model_dir = Path(model_path).expanduser()
    if not model_dir.is_dir():
        return False
    if _probe_provider_adapted(model_dir, manifest, model_settings).adapted:
        return False
    if is_builtin and _search_artifact_cache(model_dir) is not None:
        return False

    spec = _converter_spec_for_manifest(manifest, is_builtin=is_builtin)
    if spec is None:
        return False
    if find_provider_converter_tool_fast(binary_path, manifest, is_builtin=is_builtin) is None:
        return False

    # Verify required metallibs exist before claiming auto-route
    dummy_bin = M5Binary(
        path=binary_path,
        version=M5RT_DEFAULT_VERSION,
        protocol_version=2,
        auto_detected=True,
        provider_manifest=manifest,
    )
    mdir_resolved, mdir_err = _resolve_metallib_dir(
        model_dir,
        dummy_bin,
        model_settings,
        engine_id=str(manifest.get("engine_id") or M5RT_ENGINE_ID),
        provider_id=str(manifest.get("provider_id") or M5RT_PROVIDER_ID),
    )
    if mdir_err or (is_builtin and mdir_resolved is None):
        return False

    if not os.access(model_dir, os.W_OK):
        return False

    if is_builtin:
        probe = inspect_m5rt_source_fast(model_dir)
        if probe["is_dflash2_draft"]:
            return False
        return bool(probe["eligible"])

    cfg = _read_json(model_dir / "config.json")
    if cfg is None:
        return False

    arch_caps = manifest.get("architecture_capabilities")
    whitelist_enforced = not (
        isinstance(arch_caps, dict)
        and arch_caps.get("model_type_whitelist_enforced") is False
    )
    supported_types = {
        str(m).strip().lower()
        for m in (manifest.get("supported_model_types") or ())
        if isinstance(m, str) and m.strip()
    }
    model_types = _extract_model_type_tokens(cfg)
    if whitelist_enforced and supported_types and (
        not model_types or not (model_types & supported_types)
    ):
        return False

    quant_ok, _ = _config_matches_converter_quant(cfg, spec, is_builtin=False)
    if not quant_ok:
        return False

    return _has_valid_safetensors_header(model_dir)


def can_auto_convert_m5rt_fast(
    model_path: str | Path,
    configured_binary: str | None = None,
    model_settings: object | None = None,
) -> bool:
    """Fast O(1) zero-subprocess check whether `m5rt` (`miv`) can auto-repack `model_path`."""
    bin_path, inst_manifest = _find_m5rt_binary_path_fast(configured_binary)
    if bin_path is None:
        return False
    manifest = _builtin_m5rt_manifest()
    if isinstance(inst_manifest, dict) and inst_manifest.get("_provider_dir"):
        manifest["_provider_dir"] = inst_manifest["_provider_dir"]
    return can_auto_convert_provider_fast(
        model_path,
        binary_path=bin_path,
        manifest=manifest,
        model_settings=model_settings,
        is_builtin=True,
    )


def _is_dflash_family_architecture(cfg: dict[str, Any] | None) -> tuple[bool, str]:
    """Return `(is_draft_arch, arch_name)` when `cfg` declares a DFlash draft architecture."""
    if not isinstance(cfg, dict):
        return False, ""
    raw_archs = cfg.get("architectures") or ()
    if isinstance(raw_archs, str):
        raw_archs = (raw_archs,)
    if not isinstance(raw_archs, (list, tuple)):
        return False, ""
    for item in raw_archs:
        arch = str(item).strip()
        if "dflash" in arch.lower() or arch.endswith("DraftModel"):
            return True, arch
    return False, ""


def _is_supported_dflash2_draft_config(
    cfg: dict[str, Any] | None,
    manifest: dict[str, Any] | None = None,
    *,
    is_builtin: bool = True,
    target_config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Verify structural DFlash2 draft compatibility (and optional target model pairing)."""
    if not isinstance(cfg, dict):
        return False, "config.json is not a valid JSON object"

    raw_archs = cfg.get("architectures") or ()
    if isinstance(raw_archs, str):
        archs = [raw_archs.strip()]
    elif isinstance(raw_archs, (list, tuple)):
        archs = [str(a).strip() for a in raw_archs if isinstance(a, str) and a.strip()]
    else:
        archs = []

    allowed_draft_archs = [
        str(a).strip()
        for a in (
            (manifest.get("supported_draft_architectures") if isinstance(manifest, dict) else None)
            or (M5RT_SUPPORTED_DFLASH2_ARCHITECTURES if is_builtin else ())
        )
        if isinstance(a, str) and a.strip()
    ]
    if not allowed_draft_archs:
        return False, "provider does not declare supported DFlash2 draft architectures"
    if archs != ["DFlash2DraftModel"] and not (set(archs) & set(allowed_draft_archs)):
        if any("dflash" in a.lower() for a in archs):
            return (
                False,
                f"draft architecture {archs!r} is not supported (expected {allowed_draft_archs})",
            )
        return False, f"architectures {archs!r} is not a DFlash2 draft model"

    if not is_builtin:
        return True, ""

    if cfg.get("is_causal", True) is not False:
        return False, "DFlash2 draft config must set is_causal=false"

    extra = cfg.get("dflash_config")
    if not isinstance(extra, dict):
        return False, "DFlash2 draft config is missing dflash_config object"

    layer_count = cfg.get("num_hidden_layers")
    layer_types = cfg.get("layer_types")
    block_size = extra.get("block_size")
    conv_kernel = extra.get("conv_kernel_size")
    conv_group = extra.get("conv_group_size")
    sel_rank = extra.get("selector_rank")
    sel_topk = extra.get("selector_top_k")
    target_lids = extra.get("target_layer_ids")
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count < 1
        or layer_count > 5
        or not isinstance(layer_types, list)
        or len(layer_types) != layer_count
        or any(lt != "sliding_attention" for lt in layer_types)
        or isinstance(conv_kernel, bool)
        or conv_kernel != 2
        or isinstance(conv_group, bool)
        or not isinstance(conv_group, int)
        or conv_group <= 0
        or isinstance(sel_rank, bool)
        or not isinstance(sel_rank, int)
        or sel_rank <= 0
        or isinstance(sel_topk, bool)
        or not isinstance(sel_topk, int)
        or sel_topk <= 0
        or isinstance(block_size, bool)
        or not isinstance(block_size, int)
        or block_size < 2
        or block_size > 16
        or not isinstance(target_lids, list)
        or not target_lids
        or any(isinstance(tid, bool) or not isinstance(tid, int) or tid < 0 for tid in target_lids)
    ):
        return False, "DFlash2 draft config geometry does not match miv DFlash2 contract"

    hidden_size = cfg.get("hidden_size")
    if (
        isinstance(hidden_size, bool)
        or not isinstance(hidden_size, int)
        or hidden_size < 64
        or hidden_size % 64 != 0
    ):
        return (
            False,
            f"DFlash2 draft hidden_size={hidden_size!r} must be >= 64 and divisible by 64",
        )

    vocab_size = cfg.get("vocab_size")
    if isinstance(vocab_size, bool) or not isinstance(vocab_size, int) or vocab_size <= 0:
        return False, f"DFlash2 draft vocab_size={vocab_size!r} must be a positive integer"

    quant = cfg.get("quantization") or cfg.get("quantization_config") or {}
    if isinstance(quant, dict) and (
        quant.get("mode") or quant.get("format") or quant.get("quant_method") or quant.get("bits")
    ):
        return False, "DFlash2 draft conversion requires an unquantized BF16 checkpoint"
    raw_dtype = str(cfg.get("dtype") or cfg.get("torch_dtype") or "bfloat16").strip().lower()
    if raw_dtype not in ("bfloat16", "bf16"):
        return False, f"DFlash2 draft dtype={raw_dtype!r} is not supported (expected 'bfloat16')"

    if isinstance(target_config, dict):
        t_text = _text_config(target_config)
        t_vocab = (
            t_text.get("vocab_size")
            if t_text.get("vocab_size") is not None
            else target_config.get("vocab_size")
        )
        t_hidden = (
            t_text.get("hidden_size")
            if t_text.get("hidden_size") is not None
            else target_config.get("hidden_size")
        )
        t_layers = (
            t_text.get("num_hidden_layers")
            if t_text.get("num_hidden_layers") is not None
            else target_config.get("num_hidden_layers")
        )
        if isinstance(t_vocab, int) and not isinstance(t_vocab, bool) and vocab_size != t_vocab:
            return (
                False,
                f"DFlash2 draft vocab_size={vocab_size} does not match target vocab_size={t_vocab}",
            )
        if isinstance(t_hidden, int) and not isinstance(t_hidden, bool) and hidden_size != t_hidden:
            return (
                False,
                f"DFlash2 draft hidden_size={hidden_size} does not match target hidden_size={t_hidden}",
            )
        if isinstance(t_layers, int) and not isinstance(t_layers, bool) and t_layers > 0:
            num_target_layers = cfg.get("num_target_layers")
            if (
                isinstance(num_target_layers, int)
                and not isinstance(num_target_layers, bool)
                and num_target_layers != t_layers
            ):
                return (
                    False,
                    f"DFlash2 draft num_target_layers={num_target_layers} does not match "
                    f"target num_hidden_layers={t_layers}",
                )
            max_tid = max(target_lids)
            if max_tid >= t_layers:
                return (
                    False,
                    f"DFlash2 draft max(target_layer_ids)={max_tid} must be < "
                    f"target num_hidden_layers={t_layers}",
                )

    return True, ""


def _can_convert_dflash2_draft_fast(
    model_dir: Path,
    configured_binary: str | None = None,
    *,
    target_config: dict[str, Any] | None = None,
) -> bool:
    """Fast O(1) zero-subprocess check whether `model_dir` is a convertible DFlash2 draft."""
    if not model_dir.is_dir():
        return False
    manifest = _builtin_m5rt_manifest()
    extensions = _normalize_extensions(manifest.get("artifact_extensions"))
    if _find_draft_artifact_in_dir(model_dir, extensions, is_builtin=True) is not None:
        return False
    probe = inspect_m5rt_source_fast(model_dir, target_config=target_config)
    if not probe["is_dflash2_draft"] or not probe["eligible"]:
        return False
    bin_path, inst_manifest = _find_m5rt_binary_path_fast(configured_binary)
    if bin_path is None:
        return False
    if isinstance(inst_manifest, dict) and inst_manifest.get("_provider_dir"):
        manifest["_provider_dir"] = inst_manifest["_provider_dir"]
    if find_provider_converter_tool_fast(bin_path, manifest, is_builtin=True) is None:
        return False
    return os.access(model_dir, os.W_OK)


def can_convert_provider_model(
    model_path: str | Path,
    *,
    binary: M5Binary | None = None,
    binary_path_hint: Path | None = None,
    configured_binary: str | None = None,
    manifest: dict[str, Any] | None = None,
    model_settings: object | None = None,
    is_builtin: bool = True,
    target_model_dir: str | Path | None = None,
    target_config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return `(True, '')` when `model_path` has `.safetensors` convertible for this provider."""
    if not model_path:
        return False, "model_path is empty"
    model_dir = Path(model_path).expanduser()
    if not model_dir.is_dir():
        return False, f"model directory not found: {model_dir}"

    eff_manifest = dict(manifest) if isinstance(manifest, dict) else (
        _builtin_m5rt_manifest() if is_builtin else {}
    )
    if _probe_provider_adapted(model_dir, eff_manifest, model_settings).adapted:
        return False, "model already has repacked microkernel weights"

    spec = _converter_spec_for_manifest(eff_manifest, is_builtin=is_builtin)
    if spec is None:
        return False, "provider manifest does not declare a weight converter"

    cfg_path = model_dir / "config.json"
    if not cfg_path.is_file():
        return False, "config.json not found"
    cfg = _read_json(cfg_path)
    if cfg is None:
        return False, "config.json is not a valid JSON object"

    extensions = _normalize_extensions(eff_manifest.get("artifact_extensions"))
    is_draft_arch, _ = _is_dflash_family_architecture(cfg)
    if is_draft_arch and _find_draft_artifact_in_dir(model_dir, extensions, is_builtin=False) is not None:
        return False, "model already has repacked microkernel weights"

    if is_builtin:
        probe = inspect_m5rt_source_fast(
            model_dir,
            target_model_dir=target_model_dir,
            target_config=target_config,
        )
        if not probe["eligible"]:
            return False, str(probe["reason"])
    else:
        if is_draft_arch:
            draft_ok, draft_reason = _is_supported_dflash2_draft_config(
                cfg, eff_manifest, is_builtin=False, target_config=target_config
            )
            if not draft_ok:
                return False, draft_reason
        else:
            arch_caps = eff_manifest.get("architecture_capabilities")
            whitelist_enforced = not (
                isinstance(arch_caps, dict)
                and arch_caps.get("model_type_whitelist_enforced") is False
            )
            supported_types = {
                str(m).strip().lower()
                for m in (eff_manifest.get("supported_model_types") or ())
                if isinstance(m, str) and m.strip()
            }
            model_types = _extract_model_type_tokens(cfg)
            if whitelist_enforced and supported_types and (
                not model_types or not (model_types & supported_types)
            ):
                return False, f"model_type is not in {sorted(supported_types)}"

            quant_ok, quant_reason = _config_matches_converter_quant(
                cfg, spec, is_builtin=False
            )
            if not quant_ok:
                return False, quant_reason

        if not _has_valid_safetensors_header(model_dir):
            return False, "no valid .safetensors weights found in model directory"

    if not os.access(model_dir, os.W_OK):
        return False, f"model directory is not writable: {model_dir}"

    resolved_bin = binary
    if resolved_bin is None:
        if is_builtin:
            resolved_bin, bin_reason = find_m5rt_binary(configured_binary)
            if resolved_bin is None:
                return False, bin_reason
        elif binary_path_hint is not None:
            proto_raw = eff_manifest.get("protocol_version")
            fb_proto = proto_raw if isinstance(proto_raw, int) and not isinstance(proto_raw, bool) else 2
            resolved_bin, bin_reason = inspect_m5rt_binary(
                binary_path_hint,
                auto_detected=True,
                provider_manifest=eff_manifest,
                allow_generic=True,
                fallback_protocol=fb_proto,
            )
            if resolved_bin is None:
                return False, bin_reason
        else:
            return False, "provider binary not found"

    if isinstance(resolved_bin.provider_manifest, dict) and resolved_bin.provider_manifest.get("_provider_dir"):
        eff_manifest.setdefault("_provider_dir", resolved_bin.provider_manifest["_provider_dir"])

    tool = find_provider_converter_tool_fast(
        resolved_bin.path, eff_manifest, is_builtin=is_builtin
    )
    if tool is None and is_builtin:
        tool = find_m5_artifact_tool(configured_binary)
    if tool is None:
        return False, f"converter tool '{spec.get('entrypoint')}' not found in installed provider bundle"

    return True, ""


def convert_provider_model(
    model_path: str | Path,
    *,
    source_repo: str | None = None,
    binary: M5Binary | None = None,
    binary_path_hint: Path | None = None,
    configured_binary: str | None = None,
    manifest: dict[str, Any] | None = None,
    model_settings: object | None = None,
    nax_pretile: bool | None = None,
    is_builtin: bool = True,
    target_model_dir: str | Path | None = None,
    target_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically repack `.safetensors` weights in `model_path` alongside original weights."""
    model_dir = Path(model_path).expanduser().resolve()
    eff_manifest = dict(manifest) if isinstance(manifest, dict) else (
        _builtin_m5rt_manifest() if is_builtin else {}
    )
    out_path = _converter_output_path(model_dir, eff_manifest, is_builtin=is_builtin)
    lock_key = f"{model_dir}::{out_path.name}"
    with _CONVERT_LOCKS_GUARD:
        lock = _CONVERT_LOCKS.setdefault(lock_key, threading.Lock())

    with lock:
        existing_probe = _probe_provider_adapted(model_dir, eff_manifest, model_settings)
        if existing_probe.adapted and existing_probe.artifact_path is not None and existing_probe.artifact_path.is_file():
            art = existing_probe.artifact_path
            return {
                "artifact_path": str(art),
                "artifact_bytes": art.stat().st_size,
                "artifact_format": art.suffix.lstrip(".").lower() or "miv",
                "already_existed": True,
            }

        cfg_for_conv = _read_json(model_dir / "config.json")
        is_draft_arch, _ = _is_dflash_family_architecture(cfg_for_conv)
        if is_draft_arch:
            extensions = _normalize_extensions(eff_manifest.get("artifact_extensions"))
            existing_draft_art = _find_draft_artifact_in_dir(
                model_dir, extensions, is_builtin=False
            )
            if existing_draft_art is not None and existing_draft_art.is_file():
                return {
                    "artifact_path": str(existing_draft_art),
                    "artifact_bytes": existing_draft_art.stat().st_size,
                    "artifact_format": existing_draft_art.suffix.lstrip(".").lower() or "miv",
                    "already_existed": True,
                }

        ok, reason = can_convert_provider_model(
            model_dir,
            binary=binary,
            binary_path_hint=binary_path_hint,
            configured_binary=configured_binary,
            manifest=eff_manifest,
            model_settings=model_settings,
            is_builtin=is_builtin,
            target_model_dir=target_model_dir,
            target_config=target_config,
        )
        if not ok:
            raise ValueError(f"Cannot convert model to {out_path.suffix}: {reason}")

        resolved_bin = binary
        if resolved_bin is None:
            if is_builtin:
                resolved_bin, _ = find_m5rt_binary(configured_binary)
            elif binary_path_hint is not None:
                resolved_bin, _ = inspect_m5rt_binary(
                    binary_path_hint,
                    auto_detected=True,
                    provider_manifest=eff_manifest,
                    allow_generic=True,
                    fallback_protocol=2,
                )
        if resolved_bin is not None and isinstance(resolved_bin.provider_manifest, dict):
            if resolved_bin.provider_manifest.get("_provider_dir"):
                eff_manifest.setdefault("_provider_dir", resolved_bin.provider_manifest["_provider_dir"])

        tool = (
            find_provider_converter_tool_fast(
                resolved_bin.path if resolved_bin else binary_path_hint,
                eff_manifest,
                is_builtin=is_builtin,
            )
            or (find_m5_artifact_tool(configured_binary) if is_builtin else None)
        )
        if tool is None:
            raise ValueError("Provider converter tool not found")

        spec = _converter_spec_for_manifest(eff_manifest, is_builtin=is_builtin) or {}
        subcommand = str(spec.get("subcommand") or "").strip()
        default_args = [
            str(a) for a in (spec.get("default_args") or ()) if isinstance(a, str) and a.strip()
        ]
        if is_builtin and (nax_pretile is False or is_draft_arch):
            default_args = [a for a in default_args if a != "--nax-pretile"]
        elif is_builtin and nax_pretile is True and "--nax-pretile" not in default_args:
            default_args.append("--nax-pretile")
        if is_builtin and is_draft_arch and "--profile" not in default_args:
            default_args.extend(["--profile", "dflash2_bf16"])

        repo_label = (source_repo or "").strip() or model_dir.name
        tmp_out = model_dir / f".{out_path.name}.tmp.{os.getpid()}"
        try:
            if tool.suffix.lower() == ".py":
                cmd = [sys.executable, str(tool)]
            else:
                cmd = [str(tool)]
            if subcommand:
                cmd.append(subcommand)
            cmd.extend(
                [
                    "--source",
                    str(model_dir),
                    "--output",
                    str(tmp_out),
                    "--source-repo",
                    repo_label,
                    *default_args,
                ]
            )
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                if len(detail) > 600:
                    detail = detail[-600:]
                raise ValueError(
                    f"{tool.name} convert failed (exit {proc.returncode}): {detail}"
                )
            if not tmp_out.is_file() or tmp_out.stat().st_size == 0:
                raise ValueError(f"Converter finished without producing {out_path.name}")
            if is_builtin:
                if tmp_out.stat().st_size < _OMLXM5_HEADER_BYTES:
                    raise ValueError("Converter produced a truncated .miv container")
                with tmp_out.open("rb") as fh:
                    if fh.read(len(_MIV_MAGIC)) not in _MIV_VALID_MAGICS:
                        raise ValueError(
                            "Converted .miv file does not have a valid MIV00001/OMLXM5A1 header"
                        )
            os.replace(tmp_out, out_path)
        finally:
            with suppress(OSError):
                if tmp_out.exists():
                    tmp_out.unlink()

        return {
            "artifact_path": str(out_path),
            "artifact_bytes": out_path.stat().st_size,
            "artifact_format": out_path.suffix.lstrip(".").lower() or "miv",
            "already_existed": False,
        }


def can_convert_to_miv(
    model_path: str | Path,
    configured_binary: str | None = None,
    model_settings: object | None = None,
    *,
    target_model_dir: str | Path | None = None,
    target_config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return `(True, '')` when `model_path` has `.safetensors` convertible to `.miv`."""
    return can_convert_provider_model(
        model_path,
        configured_binary=configured_binary,
        manifest=_builtin_m5rt_manifest(),
        model_settings=model_settings,
        is_builtin=True,
        target_model_dir=target_model_dir,
        target_config=target_config,
    )


def convert_model_to_miv(
    model_path: str | Path,
    *,
    source_repo: str | None = None,
    configured_binary: str | None = None,
    model_settings: object | None = None,
    nax_pretile: bool = True,
    target_model_dir: str | Path | None = None,
    target_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert `.safetensors` weights in `model_path` to `model.miv` atomically alongside `.safetensors`."""
    return convert_provider_model(
        model_path,
        source_repo=source_repo,
        configured_binary=configured_binary,
        manifest=_builtin_m5rt_manifest(),
        model_settings=model_settings,
        nax_pretile=nax_pretile,
        is_builtin=True,
        target_model_dir=target_model_dir,
        target_config=target_config,
    )


# Backward-compatible aliases for external callers / tests
can_convert_to_omlxm5 = can_convert_to_miv
convert_model_to_omlxm5 = convert_model_to_miv


def detect_model_weight_format_info(
    model_path: str | Path | None,
    *,
    model_format: str | None = None,
    active_external_engine: str | None = None,
    model_settings: object | None = None,
    configured_m5rt_binary: str | None = None,
) -> dict[str, Any]:
    """Return O(1) weight-format metadata for `/admin/api/models` UI badges."""
    if model_format == "gguf":
        return {
            "weight_format": "gguf",
            "has_miv_weights": False,
            "pending_auto_repack": False,
            "repack_target_format": None,
        }
    if not model_path:
        return {
            "weight_format": model_format or "safetensors",
            "has_miv_weights": False,
            "pending_auto_repack": False,
            "repack_target_format": None,
        }
    mdir = Path(str(model_path)).expanduser()
    if not mdir.is_dir():
        return {
            "weight_format": model_format or "safetensors",
            "has_miv_weights": False,
            "pending_auto_repack": False,
            "repack_target_format": None,
        }

    has_safetensors = False
    has_miv_file = False
    has_omlxm5_file = False
    try:
        for f in mdir.iterdir():
            if not f.is_file() or f.name.startswith("."):
                continue
            low_name = f.name.lower()
            suf = f.suffix.lower()
            if suf in (".safetensors", ".bin"):
                has_safetensors = True
            elif suf == ".miv" and low_name != "mtp.miv":
                has_miv_file = True
            elif suf == ".omlxm5" and low_name != "mtp.omlxm5":
                has_omlxm5_file = True
    except OSError:
        pass

    # Also check Path B custom artifact extensions when routed to a Path B engine
    custom_art_fmt: str | None = None
    repack_target_fmt: str | None = None
    if active_external_engine and active_external_engine not in ("splash", "stock", M5RT_ENGINE_ID):
        p_manifest = _spi_provider_manifest_for_engine(active_external_engine)
        if p_manifest is not None:
            p_probe = _probe_provider_adapted(mdir, p_manifest, model_settings)
            if p_probe.adapted and p_probe.artifact_path is not None:
                custom_art_fmt = p_probe.artifact_path.suffix.lstrip(".").lower() or "miv"
            else:
                out_p = _converter_output_path(mdir, p_manifest, is_builtin=False)
                repack_target_fmt = out_p.suffix.lstrip(".").lower() or "miv"

    has_microkernel_weights = bool(has_miv_file or has_omlxm5_file or custom_art_fmt)
    on_disk_microkernel_fmt = (
        custom_art_fmt
        or ("miv" if has_miv_file else ("omlxm5" if has_omlxm5_file else None))
    )

    # Companion DFlash2 helper checkpoints are not routed as standalone engines
    # (`active_external_engine` is None unless overridden to `"stock"`), so surface
    # `.miv` badge / pending-repack state directly when supported by miv.
    if not active_external_engine:
        cfg_local = _read_json(mdir / "config.json")
        is_df_arch, _ = _is_dflash_family_architecture(cfg_local)
        if is_df_arch:
            ok_df, _ = _is_supported_dflash2_draft_config(
                cfg_local, _builtin_m5rt_manifest(), is_builtin=True
            )
            if ok_df:
                if on_disk_microkernel_fmt is not None:
                    return {
                        "weight_format": on_disk_microkernel_fmt,
                        "has_miv_weights": True,
                        "pending_auto_repack": False,
                        "repack_target_format": None,
                    }
                cached_df = _search_artifact_cache(mdir)
                if cached_df is not None:
                    return {
                        "weight_format": cached_df.suffix.lstrip(".").lower() or "miv",
                        "has_miv_weights": True,
                        "pending_auto_repack": False,
                        "repack_target_format": None,
                    }
                if _can_convert_dflash2_draft_fast(mdir, configured_m5rt_binary):
                    return {
                        "weight_format": "safetensors",
                        "has_miv_weights": False,
                        "pending_auto_repack": True,
                        "repack_target_format": "miv",
                    }

    # When running on Stock MLX (or explicitly overridden to "stock"), the active format is safetensors
    if not active_external_engine or active_external_engine in ("stock", "splash"):
        active_fmt = (
            "safetensors"
            if has_safetensors
            else (on_disk_microkernel_fmt or model_format or "safetensors")
        )
        return {
            "weight_format": active_fmt,
            "has_miv_weights": has_microkernel_weights,
            "pending_auto_repack": False,
            "repack_target_format": None,
        }

    # Active engine is m5rt or a Path B microkernel
    if on_disk_microkernel_fmt is not None:
        return {
            "weight_format": on_disk_microkernel_fmt,
            "has_miv_weights": True,
            "pending_auto_repack": False,
            "repack_target_format": None,
        }

    # Check `.cache/b0/artifacts` hit for m5rt
    if active_external_engine == M5RT_ENGINE_ID:
        cached = _search_artifact_cache(mdir)
        if cached is not None:
            return {
                "weight_format": cached.suffix.lstrip(".").lower() or "miv",
                "has_miv_weights": True,
                "pending_auto_repack": False,
                "repack_target_format": None,
            }
        if can_auto_convert_m5rt_fast(mdir, configured_m5rt_binary, model_settings):
            return {
                "weight_format": "safetensors",
                "has_miv_weights": False,
                "pending_auto_repack": True,
                "repack_target_format": "miv",
            }
    elif repack_target_fmt is not None:
        plugin = get_external_engine_plugin(active_external_engine)
        if plugin is not None:
            ok_conv = False
            if callable(plugin.can_auto_convert_fast):
                ok_conv = bool(plugin.can_auto_convert_fast(mdir, model_settings))
            elif callable(plugin.can_convert):
                ok_conv, _ = plugin.can_convert(mdir, model_settings)
            if ok_conv:
                return {
                    "weight_format": "safetensors",
                    "has_miv_weights": False,
                    "pending_auto_repack": True,
                    "repack_target_format": repack_target_fmt,
                }

    return {
        "weight_format": "safetensors" if has_safetensors else (model_format or "safetensors"),
        "has_miv_weights": has_microkernel_weights,
        "pending_auto_repack": False,
        "repack_target_format": None,
    }
