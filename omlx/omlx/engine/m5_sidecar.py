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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ..api.tool_calling import convert_tools_for_template
from ..api.utils import (
    _consolidate_system_messages,
    detect_and_strip_partial,
)
from ..exceptions import InvalidRequestError
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
M5RT_DEFAULT_VERSION = "miv-v0.01"
M5RT_PROVIDER_ID = "io.ninfer.metal"
M5RT_LEGACY_PROVIDER_ID = "com.omlx.mseries"
M5RT_BUILTIN_PROVIDER_IDS = frozenset({M5RT_PROVIDER_ID, M5RT_LEGACY_PROVIDER_ID})
M5RT_OPTION_KEYS = (M5RT_ENGINE_ID, "ninfer-metal", M5RT_PROVIDER_ID, M5RT_LEGACY_PROVIDER_ID)
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
M5RT_ARTIFACT_EXTENSIONS: tuple[str, ...] = (".omlxm5",)
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
)
M5RT_INSTALL_HINT = (
    "miv Core Backend (ninfer-metal) daemon (mseries_backendd) not found; set integrations.m5rt_path, "
    "OMLX_M5RT_BIN, install provider io.ninfer.metal, or build build/miv-v001/miv/mseries_backendd"
)

_BREW_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin")
_USER_BIN_DIR = Path.home() / ".local" / "bin"
_OMLX_APP_SUPPORT_DIR = Path.home() / "Library" / "Application Support" / "oMLX"
_PROVIDERS_BASE_DIR = _OMLX_APP_SUPPORT_DIR / "Providers"
_PROVIDER_BIN_DIR = _PROVIDERS_BASE_DIR / M5RT_LEGACY_PROVIDER_ID / "bin"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_DAEMON_RELPATHS = (
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
VALID_PREFILL_ENGINES = frozenset({"auto", "nax", "nax_ane"})
VALID_SAMPLING_CONTRACTS = frozenset({"legacy_rejection_v1", "position_coupled_v1"})
VALID_SPEC_BACKENDS = frozenset({"auto", "none", "dflash2", "mtp"})


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


def get_external_engine_plugin(engine_id: str) -> ExternalEnginePlugin | None:
    """Return a discovered external engine plugin by ID, or None."""
    if not engine_id or engine_id in EXTERNAL_ENGINES:
        return None
    return discover_external_engine_plugins().get(engine_id)


def get_external_engine_supported_settings(engine_id: str | None) -> tuple[str, ...]:
    """Return standard oMLX `ModelSettings` keys honored by `engine_id`.

    Built-in `splash` returns `()` (preserving exact upstream UI lock behavior);
    `m5rt` (`ninfer-metal` / `miv`) returns its directly-mapped standard keys.
    """
    if not engine_id or engine_id == "stock":
        return ()
    if engine_id == M5RT_ENGINE_ID:
        return M5RT_SUPPORTED_STANDARD_SETTINGS
    if engine_id == "splash":
        return ()
    plugin = get_external_engine_plugin(engine_id)
    if plugin is not None:
        return tuple(plugin.supported_standard_settings)
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


@dataclass(frozen=True)
class M5Options:
    """Runtime options passed to `M5SidecarEngine`."""

    max_context: int = 32768
    max_concurrency: int = 1
    kv_dtype: str = "bf16"
    prefill_engine: str = "auto"
    sampling_contract: str = "legacy_rejection_v1"
    speculative_backend: str = "none"
    draft_tokens: int = 0
    prefix_cache_bytes: int = 0
    rolling_prefix_cache: bool | None = None
    batch_mode: bool | None = None
    batch_projection_prototype: bool = False
    path_overrides: dict[str, str] = field(default_factory=dict)
    fake_options: dict[str, Any] = field(default_factory=dict)
    memory_budget_bytes: int = 0
    trust_remote_code: bool = False


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
        if isinstance(draft_tokens, bool) or not isinstance(draft_tokens, int) or not 0 <= draft_tokens <= 7:
            raise ValueError("m5rt_options.draft_tokens must be an integer in [0, 7]")

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

    return dict(options)


def sync_m5rt_ui_kv_setting(
    current_settings: Any,
    request: Any,
    sent: set[str] | frozenset[str],
) -> None:
    """Persist an explicit UI KV-compression toggle for `m5rt` so disabling overrides shared TOML."""
    if "turboquant_kv_enabled" not in sent:
        return
    req_explicit_kv = False
    if "m5rt_options" in sent and isinstance(getattr(request, "m5rt_options", None), dict):
        if request.m5rt_options.get("kv_dtype") is not None:
            req_explicit_kv = True
    if "external_engine_options" in sent and isinstance(
        getattr(request, "external_engine_options", None), dict
    ):
        for eng_key in M5RT_OPTION_KEYS:
            nested = request.external_engine_options.get(eng_key)
            if isinstance(nested, dict) and nested.get("kv_dtype") is not None:
                req_explicit_kv = True
                break
    if req_explicit_kv:
        return

    if not bool(getattr(request, "turboquant_kv_enabled", False)):
        opts_copy = dict(getattr(current_settings, "m5rt_options", None) or {})
        opts_copy["kv_dtype"] = "bf16"
        current_settings.m5rt_options = opts_copy
    else:
        existing_opts = getattr(current_settings, "m5rt_options", None)
        if isinstance(existing_opts, dict) and existing_opts.get("kv_dtype") == "bf16":
            opts_copy = dict(existing_opts)
            opts_copy.pop("kv_dtype", None)
            current_settings.m5rt_options = opts_copy or None


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
    if "external_engine_options" in sent:
        raw_ext_opts = getattr(request, "external_engine_options", None)
        if is_diffusion_model or not raw_ext_opts:
            current_settings.external_engine_options = None
        else:
            if not isinstance(raw_ext_opts, dict):
                raise ValueError("external_engine_options must be an object")
            validated_ext: dict[str, dict[str, Any]] = {}
            for eng_key, eng_val in raw_ext_opts.items():
                if not isinstance(eng_val, dict):
                    raise ValueError(
                        f"external_engine_options.{eng_key} must be an object"
                    )
                if eng_key in M5RT_OPTION_KEYS:
                    validated_ext[eng_key] = validate_m5rt_options(eng_val) or {}
                else:
                    validated_ext[str(eng_key)] = dict(eng_val)
            current_settings.external_engine_options = validated_ext

    eff_ext = external_engine
    if eff_ext is None:
        if "external_engine" in sent:
            eff_ext = None if is_diffusion_model else getattr(request, "external_engine", None) or None
        else:
            eff_ext = getattr(current_settings, "external_engine", None)
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


def _extract_toml_layered_options(
    doc: dict[str, Any], *, model_id: str | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split `ninfer-metal.toml` into `(shared_opts, per_model_opts)`."""
    shared: dict[str, Any] = {}
    model_sec = doc.get("model")
    if isinstance(model_sec, dict):
        for key in (
            "artifact_path",
            "draft_artifact_path",
            "draft_dir",
            "mtp_artifact",
            "mtp_artifact_path",
        ):
            if model_sec.get(key) is not None:
                shared[key] = model_sec[key]

    engine_sec = doc.get("engine")
    if isinstance(engine_sec, dict):
        for key in (
            "max_concurrency",
            "max_context",
            "kv_dtype",
            "prefill_engine",
            "sampling_contract",
            "rolling_prefix_cache",
            "prefix_cache_bytes",
            "prefix_cache_gb",
            "batch_mode",
            "batch_projection_prototype",
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
    sidecar_sec = doc.get("sidecar")
    if isinstance(sidecar_sec, dict):
        for key, val in sidecar_sec.items():
            if key == "models" or val is None:
                continue
            shared[key] = val
        models_sec = sidecar_sec.get("models")
        if model_id and isinstance(models_sec, dict):
            model_entry = models_sec.get(model_id)
            if isinstance(model_entry, dict):
                for key, val in model_entry.items():
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
        "supported_topologies": list(M5RT_SUPPORTED_TOPOLOGIES),
        "supported_model_types": list(M5RT_SUPPORTED_MODEL_TYPES),
        "artifact_extensions": list(M5RT_ARTIFACT_EXTENSIONS),
    }
    for root in _provider_root_dirs():
        if not root.is_dir():
            continue
        installed = _read_provider_manifest(root)
        if installed is None:
            continue
        for key in ("supported_topologies", "supported_model_types", "artifact_extensions"):
            vals = installed.get(key)
            if isinstance(vals, list) and vals:
                manifest[key] = vals
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
                explicit.update(nested)
    legacy_opts = getattr(model_settings, "m5rt_options", None)
    if isinstance(legacy_opts, dict):
        explicit.update(legacy_opts)
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


def _probe_provider_adapted(
    model_dir: str | Path,
    manifest: dict[str, Any],
    model_settings: object | None = None,
) -> ProbeResult:
    """Fast O(1) topology/extension/provider-aware probe for Path A (`m5rt`) and Path B providers.

    Checks in order with zero subprocess spawn and zero full-file hashing:
      (a) explicit `provider_id` / `engine_id` in `model_pack.json` / `omlxm5-manifest.json`,
      (b) `topology_id` or `config.json` `model_type` against `manifest["supported_topologies"]`
          and `manifest["supported_model_types"]`, and
      (c) artifact files matching `manifest.get("artifact_extensions", [".omlxm5"])`.
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
    identity_tokens = {tok.lower() for tok in (provider_id, engine_id, *legacy_ids) if tok}
    if is_builtin:
        identity_tokens.update(k.lower() for k in M5RT_OPTION_KEYS)

    extensions = _normalize_extensions(manifest.get("artifact_extensions"))
    supported_topologies = {
        str(t).strip().lower()
        for t in (manifest.get("supported_topologies") or ())
        if isinstance(t, str) and t.strip()
    }
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

    # (b) Topology / config.json model_type compatibility check
    config = _read_json(mdir / "config.json") if (mdir / "config.json").is_file() else None
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

    model_type_matched = False
    if model_types and supported_model_types:
        if model_types & supported_model_types:
            model_type_matched = True
        elif not topology_matched and not explicit_provider_match:
            return ProbeResult(
                adapted=False,
                reason=f"model_type {sorted(model_types)} not supported by '{provider_id or engine_id}'",
            )

    def _can_claim_extension(ext: str) -> bool:
        if is_builtin or explicit_provider_match or topology_matched or model_type_matched:
            return True
        # Non-builtin Path B providers without an explicit provider/topology/model_type
        # match may only auto-claim custom non-.omlxm5 artifact extensions.
        return ext != ".omlxm5"

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
        if _can_claim_extension(matched_ext) and (
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
        if not _can_claim_extension(ext):
            continue
        direct = mdir / f"model{ext}"
        if direct.is_file():
            return ProbeResult(
                adapted=True,
                artifact_path=direct.resolve(),
                reason=f"model{ext}",
                explicit_provider_match=explicit_provider_match,
            )
        if mdir.name == "source" and (mdir.parent / f"model{ext}").is_file():
            return ProbeResult(
                adapted=True,
                artifact_path=(mdir.parent / f"model{ext}").resolve(),
                reason=f"fixture parent model{ext}",
                explicit_provider_match=explicit_provider_match,
            )
        if ext != ".omlxm5" and mdir.is_dir():
            custom_matches = [p for p in sorted(mdir.glob(f"*{ext}")) if p.is_file()]
            if len(custom_matches) == 1:
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
    """Match a model folder against `.cache/b0/artifacts/*.omlxm5`."""
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
        for art in sorted(cache_dir.glob("*.omlxm5")):
            if not art.is_file():
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


def _resolve_draft_target(
    model_dir: Path,
    model_settings: object | None,
    draft_candidates: Iterable[str],
    *,
    manifest: dict[str, Any] | None = None,
    engine_id: str = M5RT_ENGINE_ID,
    provider_id: str = M5RT_PROVIDER_ID,
) -> tuple[Path | None, Path | None, str]:
    is_builtin = (engine_id == M5RT_ENGINE_ID) and (provider_id in M5RT_BUILTIN_PROVIDER_IDS)
    extensions = _normalize_extensions(manifest.get("artifact_extensions") if manifest else None)
    shared_toml, model_toml, explicit_opts = _layered_provider_options(
        model_dir,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id,
        manifest=manifest,
    )
    opts = {**shared_toml, **model_toml, **explicit_opts}

    explicit_spec = explicit_opts.get("speculative_backend")
    if explicit_spec is not None and str(explicit_spec).strip():
        spec_backend = str(explicit_spec).strip()
    elif bool(getattr(model_settings, "mtp_enabled", False)):
        spec_backend = "mtp"
    elif bool(getattr(model_settings, "dflash_enabled", False)):
        spec_backend = "dflash2"
    elif model_toml.get("speculative_backend") is not None and str(model_toml["speculative_backend"]).strip():
        spec_backend = str(model_toml["speculative_backend"]).strip()
    elif shared_toml.get("speculative_backend") is not None and str(shared_toml["speculative_backend"]).strip():
        spec_backend = str(shared_toml["speculative_backend"]).strip()
    else:
        spec_backend = "none"

    explicit_draft_art = (
        (getattr(model_settings, "m5rt_draft_artifact_path", None) if is_builtin else None)
        or opts.get("draft_artifact_path")
        or (os.environ.get("OMLX_M5RT_DRAFT_ARTIFACT", "").strip() if is_builtin else "")
    )
    if spec_backend == "auto" and not explicit_draft_art:
        # Opportunistically detect companion `dflash2/` without failing when absent.
        companion_dir = model_dir / "dflash2"
        if companion_dir.is_dir():
            cfg_dir: Path | None = None
            for cdir in (companion_dir, companion_dir / "source"):
                if (cdir / "config.json").is_file():
                    cfg_dir = cdir.resolve()
                    break
            comp_art: Path | None = None
            for ext in extensions:
                for cand in (companion_dir / f"model{ext}", companion_dir / f"draft{ext}"):
                    if cand.is_file():
                        comp_art = cand.resolve()
                        break
                if comp_art is None:
                    matches = [p for p in sorted(companion_dir.glob(f"*{ext}")) if p.is_file()]
                    if matches:
                        comp_art = matches[0].resolve()
                if comp_art is not None:
                    break
            if cfg_dir is not None and comp_art is not None:
                return cfg_dir, comp_art, ""
        return None, None, ""

    if spec_backend != "dflash2" and not explicit_draft_art:
        return None, None, ""

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
    candidate_dirs: list[Path] = []
    if draft_dir_raw:
        candidate_dirs.append(Path(str(draft_dir_raw)).expanduser())
    if draft_art is not None:
        candidate_dirs.extend([draft_art.parent / "source", draft_art.parent])
    candidate_dirs.extend(Path(c).expanduser() for c in draft_candidates)

    resolved_dir: Path | None = None
    for cdir in candidate_dirs:
        if (cdir / "config.json").is_file():
            resolved_dir = cdir.resolve()
            break

    if draft_art is None and resolved_dir is not None:
        for ext in extensions:
            for cand in (resolved_dir / f"model{ext}", resolved_dir.parent / f"model{ext}"):
                if cand.is_file():
                    draft_art = cand.resolve()
                    break
            if draft_art is not None:
                break

    if spec_backend == "dflash2":
        if resolved_dir is None:
            return None, None, "DFlash2 speculative backend requires a draft model directory with config.json"
        if draft_art is None:
            return None, None, f"DFlash2 speculative backend requires a draft {'/'.join(extensions)} artifact"

    return resolved_dir, draft_art, ""


def resolve_spi_provider_target(
    model_path: str | Path,
    binary_path: Path,
    manifest: dict[str, Any],
    model_settings: object | None = None,
    draft_candidates: Iterable[str] = (),
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

    artifact_path, reason = _resolve_target_artifact(
        model_dir,
        model_settings,
        manifest=manifest,
        engine_id=engine_id,
        provider_id=provider_id,
    )
    if artifact_path is None:
        return None, reason

    metallib_dir, reason = _resolve_metallib_dir(
        model_dir,
        binary,
        model_settings,
        engine_id=engine_id,
        provider_id=provider_id,
    )
    if reason:
        return None, reason

    draft_dir, draft_art, reason = _resolve_draft_target(
        model_dir,
        model_settings,
        draft_candidates,
        manifest=manifest,
        engine_id=engine_id,
        provider_id=provider_id,
    )
    if reason:
        return None, reason

    return (
        M5Target(
            binary=binary,
            artifact_path=artifact_path,
            metallib_dir=metallib_dir,
            draft_dir=draft_dir,
            draft_artifact_path=draft_art,
            fake=False,
        ),
        "",
    )


def resolve_m5rt_target(
    model_path: str | Path,
    configured_binary: str | None,
    model_settings: object | None = None,
    draft_candidates: Iterable[str] = (),
) -> tuple[M5Target | None, str]:
    """Check whether `m5rt` can serve `model_path`, returning `(M5Target, reason)`."""
    model_dir = Path(model_path).expanduser()
    opts = _merged_m5rt_options(model_dir, model_settings)
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

    artifact_path, reason = _resolve_target_artifact(model_dir, model_settings)
    if artifact_path is None:
        return None, reason

    metallib_dir, reason = _resolve_metallib_dir(model_dir, binary, model_settings)
    if metallib_dir is None:
        return None, reason

    draft_dir, draft_art, reason = _resolve_draft_target(
        model_dir, model_settings, draft_candidates
    )
    if reason:
        return None, reason

    return (
        M5Target(
            binary=binary,
            artifact_path=artifact_path,
            metallib_dir=metallib_dir,
            draft_dir=draft_dir,
            draft_artifact_path=draft_art,
            fake=False,
        ),
        "",
    )


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) else config


_OMLXM5_MAGIC = b"OMLXM5A1"
_OMLXM5_HEADER_STRUCT = struct.Struct("<8sIIQQQQIIII")
_OMLXM5_HEADER_BYTES = 64
_OMLXM5_MAX_MANIFEST_BYTES = 64 * 1024 * 1024


def inspect_omlxm5_has_mtp_weights(artifact_path: Path | None) -> bool | None:
    """Inspect an `.omlxm5` v1 header + manifest for embedded MTP objects.

    Returns:
      - ``True`` if ``artifact_path`` is a valid ``OMLXM5A1`` container with at least
        one MTP object (``mtp.*`` or ``*.mtp.*``), matching C++ ``has_mtp_weights``.
      - ``False`` if it is a valid ``OMLXM5A1`` container with no MTP objects.
      - ``None`` if the file does not exist or is not a valid ``OMLXM5A1`` container
        (e.g. a synthetic test placeholder).
    """
    if artifact_path is None:
        return None
    try:
        if not artifact_path.is_file():
            return None
        file_size = artifact_path.stat().st_size
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
                magic != _OMLXM5_MAGIC
                or version != 1
                or header_bytes != _OMLXM5_HEADER_BYTES
                or manifest_offset != _OMLXM5_HEADER_BYTES
                or flags != 0
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


def _resolve_effective_kv_dtype(
    opts: dict[str, Any],
    model_settings: object | None,
    *,
    allowed_kv_dtypes: frozenset[str] | None = None,
    shared_toml: dict[str, Any] | None = None,
    model_toml: dict[str, Any] | None = None,
    explicit_opts: dict[str, Any] | None = None,
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
        # CRITICAL: Do NOT map "affine8" to "int8_g64" — miv-v0.01 only implements
        # bf16 and affine4_g128 in kv_page_pool.cpp.
        logger.warning(
            "miv Core Backend: turboquant_kv_scheme=%r (bits=%r) is not implemented in miv-v0.01 "
            "(only 4-bit 'affine4'/'turboquant' -> 'affine4_g128' or disabled -> 'bf16'); using 'bf16'",
            scheme,
            raw_bits,
        )
        return "bf16"

    # Tier 2: [sidecar.models."<name>"] in TOML
    if model_toml is not None and model_toml.get("kv_dtype") is not None:
        return _validate_kv(model_toml["kv_dtype"])

    # Tier 3: [engine] in shared ninfer-metal.toml
    if shared_toml is not None and shared_toml.get("kv_dtype") is not None:
        return _validate_kv(shared_toml["kv_dtype"])

    # Tier 4: hardcoded fallback
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
    def _nonempty_str(d: dict[str, Any] | None, key: str) -> str | None:
        if d is None:
            return None
        val = d.get(key)
        if val is None:
            return None
        s = str(val).strip()
        return s or None

    layered = explicit_opts is not None or model_toml is not None or shared_toml is not None
    explicit_spec = (
        _nonempty_str(explicit_opts, "speculative_backend")
        if layered
        else _nonempty_str(opts, "speculative_backend")
    )
    model_toml_spec = _nonempty_str(model_toml, "speculative_backend")
    shared_toml_spec = _nonempty_str(shared_toml, "speculative_backend")

    # Precedence: explicit per-model options > oMLX UI toggles > [sidecar.models] > shared TOML > auto/fallback
    if explicit_spec is not None and explicit_spec != "auto":
        spec_backend = explicit_spec
    elif bool(getattr(model_settings, "mtp_enabled", False)):
        spec_backend = "mtp"
    elif bool(getattr(model_settings, "dflash_enabled", False)):
        spec_backend = "dflash2"
    elif model_toml_spec is not None and model_toml_spec != "auto":
        spec_backend = model_toml_spec
    elif shared_toml_spec is not None and shared_toml_spec != "auto":
        spec_backend = shared_toml_spec
    else:
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
            else (model_dir / "model.omlxm5" if model_dir is not None else None)
        )
        if (
            target is not None
            and target.draft_artifact_path is not None
            and target.draft_dir is not None
        ):
            spec_backend = "dflash2"
        elif mtp_art_path is not None and mtp_art_path.is_file():
            spec_backend = "mtp"
        elif model_dir is not None and (model_dir / "mtp.omlxm5").is_file():
            spec_backend = "mtp"
        else:
            embedded_mtp = inspect_omlxm5_has_mtp_weights(target_art)
            if embedded_mtp is True:
                spec_backend = "mtp"
            elif (
                embedded_mtp is None
                and model_dir is not None
                and _config_json_has_mtp(_read_json(model_dir / "config.json"))
            ):
                spec_backend = "mtp"
            else:
                spec_backend = "none"

    def _positive_int(d: dict[str, Any] | None, key: str) -> int | None:
        if d is None or d.get(key) is None:
            return None
        val = int(d[key])
        return val if val > 0 else None

    explicit_dt = (
        _positive_int(explicit_opts, "draft_tokens")
        if layered
        else _positive_int(opts, "draft_tokens")
    )
    if explicit_dt is not None:
        draft_tokens = explicit_dt
    elif spec_backend == "mtp":
        fixed = getattr(model_settings, "mtp_fixed_depth", None)
        adaptive = getattr(model_settings, "mtp_adaptive_max_depth", None)
        chosen = fixed if fixed is not None else adaptive
        if chosen is not None:
            draft_tokens = int(chosen)
        else:
            toml_dt = _positive_int(model_toml, "draft_tokens") or _positive_int(
                shared_toml, "draft_tokens"
            )
            draft_tokens = toml_dt if toml_dt is not None else 3
    elif spec_backend == "dflash2":
        block_size = getattr(model_settings, "dflash_block_size", None)
        if isinstance(block_size, int) and block_size >= 2:
            draft_tokens = block_size - 1
        else:
            toml_dt = _positive_int(model_toml, "draft_tokens") or _positive_int(
                shared_toml, "draft_tokens"
            )
            draft_tokens = toml_dt if toml_dt is not None else 4
    else:
        draft_tokens = 0

    if spec_backend in ("mtp", "dflash2"):
        draft_tokens = max(1, min(7, draft_tokens))
    else:
        draft_tokens = 0
    return spec_backend, draft_tokens


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
    head_dim = int(text_cfg.get("head_dim") or 128)
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
) -> int:
    """Estimate unified memory footprint for `M5SidecarEngine`."""
    if target.fake:
        return 64 * 1024**2
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
    )
    opts = {**shared_toml, **model_toml, **explicit_opts}
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
    )
    max_concurrency = _resolve_effective_concurrency(opts, global_settings)
    config = _read_json(mdir / "config.json") or {}
    text_cfg = _text_config(config)
    native_ctx_raw = text_cfg.get("max_position_embeddings")
    configured_ctx = (
        explicit_opts.get("max_context")
        or getattr(model_settings, "max_context_window", None)
        or max_context
        or model_toml.get("max_context")
        or shared_toml.get("max_context")
    )
    if native_ctx_raw:
        native_ctx = max(1024, int(native_ctx_raw))
    else:
        native_ctx = max(32768, int(configured_ctx or 32768))
    context = max(1024, min(int(configured_ctx or native_ctx), native_ctx))
    vocab_size = max(256, int(text_cfg.get("vocab_size") or 248320))

    weights = target.artifact_path.stat().st_size if target.artifact_path else 0
    draft = (
        target.draft_artifact_path.stat().st_size
        if target.draft_artifact_path
        else 0
    )
    kv = _kv_bytes_per_token(text_cfg, kv_dtype) * context * max_concurrency
    workspace = max(64 * 1024**2, 1024 * vocab_size * 2 * max_concurrency)
    prefix_cache = _resolve_prefix_cache_bytes(
        opts,
        shared_toml=shared_toml,
        model_toml=model_toml,
        explicit_opts=explicit_opts,
    )
    return weights + draft + kv + workspace + prefix_cache


def m5rt_runtime_signature(
    model_settings: object | None,
    global_m5rt_path: str | None,
    *,
    global_settings: Any | None = None,
) -> tuple[Any, ...]:
    """Canonical tuple of load-time M5 settings that require engine reload."""
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
    manifest = (
        binary.provider_manifest
        if binary and isinstance(binary.provider_manifest, dict)
        else {}
    )
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
        "supported_topologies": supported_topologies,
        "supported_model_types": supported_model_types,
        "artifact_extensions": artifact_extensions,
        "kv_formats": sorted(VALID_KV_DTYPES),
        "supported_standard_settings": list(M5RT_SUPPORTED_STANDARD_SETTINGS),
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

    # Precedence: explicit per-model options > oMLX UI max_context_window >
    # [sidecar.models."<name>"] > shared ninfer-metal.toml > default_context
    configured_ctx = (
        explicit_opts.get("max_context")
        or getattr(model_settings, "max_context_window", None)
        or model_toml.get("max_context")
        or shared_toml.get("max_context")
        or default_context
    )
    if native_ctx_raw:
        native_ctx = max(1024, int(native_ctx_raw))
    else:
        native_ctx = max(32768, int(configured_ctx or 32768))
    max_context = max(1024, min(int(configured_ctx or native_ctx), native_ctx))
    max_concurrency = _resolve_effective_concurrency(opts, global_settings)
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

    raw_overrides = opts.get("path_overrides")
    overrides = (
        {str(k): str(v) for k, v in raw_overrides.items()}
        if isinstance(raw_overrides, dict)
        else {}
    )
    mtp_art_raw = opts.get("mtp_artifact_path") or opts.get("mtp_artifact")
    if mtp_art_raw and "mtp_artifact" not in overrides:
        overrides["mtp_artifact"] = str(Path(str(mtp_art_raw)).expanduser())
    elif spec_backend == "mtp" and "mtp_artifact" not in overrides:
        companion_mtp = mdir / "mtp.omlxm5"
        if companion_mtp.is_file():
            overrides["mtp_artifact"] = str(companion_mtp.resolve())

    rolling_prefix_cache = _parse_optional_toggle(
        opts.get("rolling_prefix_cache"), f"{eff_engine_id}.rolling_prefix_cache"
    )
    raw_batch_mode = opts.get("batch_mode")
    if isinstance(raw_batch_mode, str) and not raw_batch_mode.strip():
        raw_batch_mode = None
    batch_mode = _parse_optional_toggle(
        raw_batch_mode, f"{eff_engine_id}.batch_mode"
    )
    return M5Options(
        max_context=max_context,
        max_concurrency=max_concurrency,
        kv_dtype=kv_dtype,
        prefill_engine=str(opts.get("prefill_engine") or "auto"),
        sampling_contract=str(opts.get("sampling_contract") or "legacy_rejection_v1"),
        speculative_backend=spec_backend,
        draft_tokens=draft_tokens,
        prefix_cache_bytes=_resolve_prefix_cache_bytes(
            opts,
            shared_toml=shared_toml,
            model_toml=model_toml,
            explicit_opts=explicit_opts,
        ),
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
) -> M5SidecarEngine | None:
    """Construct `M5SidecarEngine` for `EnginePool`, or None when memory budget <= 0."""
    eff_manifest = manifest if manifest is not None else m5_target.binary.provider_manifest
    eff_provider_id = provider_id or m5_target.binary.provider_id
    eff_engine_id = engine_id or m5_target.binary.engine_id
    wanted = m5rt_memory_estimate(
        entry.model_path,
        m5_target,
        sizing_context,
        model_settings,
        global_settings=global_settings,
        engine_id=eff_engine_id,
        provider_id=eff_provider_id,
        manifest=eff_manifest,
    )
    if admission_limit > 0:
        room = admission_limit - current_model_memory
        if room <= 0:
            return None
        budget = min(room, wanted)
    else:
        budget = wanted
    m5_options = build_m5rt_options(
        entry.model_path,
        m5_target,
        model_settings,
        default_context=sizing_context,
        memory_budget_bytes=budget,
        global_settings=global_settings,
        model_id=entry.model_id,
        engine_id=eff_engine_id,
        provider_id=eff_provider_id,
        manifest=eff_manifest,
    )
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
    synced_std = tuple(
        (key, getattr(model_settings, key, None)) for key in supported_std
    )
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
        "supported_topologies": list(manifest.get("supported_topologies") or ()),
        "supported_model_types": list(manifest.get("supported_model_types") or ()),
        "artifact_extensions": list(_normalize_extensions(manifest.get("artifact_extensions"))),
        "kv_formats": kv_formats,
        "supported_standard_settings": supported_std,
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
        )

    def _runtime_sig(model_settings: object | None, global_settings: Any) -> tuple[Any, ...]:
        return _spi_provider_runtime_signature(
            engine_id, provider_id, binary_path, manifest_copy, model_settings, global_settings
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
                if eid == engine_id:
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
        manifest = _builtin_m5rt_manifest()
        supported_types = tuple(
            str(m).strip().lower()
            for m in (manifest.get("supported_model_types") or M5RT_SUPPORTED_MODEL_TYPES)
            if isinstance(m, str) and m.strip()
        )
        if not model_types or not (model_types & set(supported_types)):
            return False, (
                f"model_type={raw_model_type!r} is not supported by {M5RT_ENGINE_LABEL} MTP "
                f"(supported: {', '.join(supported_types)})"
            )

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
        if (mdir / "mtp.omlxm5").is_file():
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
            elif (mdir / "model.omlxm5").is_file():
                art_path = mdir / "model.omlxm5"

        embedded_mtp = inspect_omlxm5_has_mtp_weights(art_path)
        if embedded_mtp is True:
            return True, ""
        if embedded_mtp is False and art_path is not None:
            return False, (
                f"{M5RT_ENGINE_LABEL} artifact '{art_path.name}' contains no embedded MTP "
                "weights and no companion mtp.omlxm5 was found."
            )
        if _config_json_has_mtp(cfg):
            return True, ""
        return False, "model has no MTP heads in config and no companion mtp.omlxm5"

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
    if (mdir / "mtp.omlxm5").is_file() or _config_json_has_mtp(cfg):
        return True, ""
    return False, "model has no MTP heads in config and no companion mtp.omlxm5"


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
        manifest = _builtin_m5rt_manifest()
        supported_types = tuple(
            str(m).strip().lower()
            for m in (manifest.get("supported_model_types") or M5RT_SUPPORTED_MODEL_TYPES)
            if isinstance(m, str) and m.strip()
        )
        if model_types & set(supported_types):
            return True, ""
        return False, (
            f"DFlash on {M5RT_ENGINE_LABEL} supports {', '.join(supported_types)} "
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
    ) -> None:
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
        if not piece and not self._pending_text:
            return ""
        if not self._stop_strings:
            return piece

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
        if self.stopped or not self._pending_text:
            self._pending_text = ""
            return ""
        rem = self._pending_text
        self._pending_text = ""
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
        self._last_capacity: dict[str, Any] | None = None
        self._reported_footprint_bytes: int = 0
        self._weights_size: int | None = None
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
        else:
            cfg_eos = text_cfg.get("eos_token_id", config.get("eos_token_id"))
            first_eos = cfg_eos[0] if isinstance(cfg_eos, list) and cfg_eos else cfg_eos
            self._tokenizer = _ByteFallbackTokenizer(
                vocab_size=self._vocab_size,
                eos_token_id=first_eos if isinstance(first_eos, int) else None,
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
                "sampling_contract": opts.sampling_contract,
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
        if opts.prefix_cache_bytes > 0:
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
        elif spec_backend == "mtp":
            draft_tokens = max(1, min(7, int(draft_tokens or 3)))
        else:
            draft_tokens = 0

        payload: dict[str, Any] = {
            "model_dir": str(self._path),
            "draft_dir": str(self._target.draft_dir) if self._target.draft_dir else "",
            "max_concurrency": opts.max_concurrency,
            "max_context": opts.max_context,
            "kv_dtype": opts.kv_dtype,
            "prefill_engine": opts.prefill_engine,
            "sampling_contract": opts.sampling_contract,
            "speculative": {
                "backend": spec_backend,
                "draft_tokens": draft_tokens,
            },
            "path_overrides": overrides,
        }
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
        self._last_gpu_contended_hint = None

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
    ) -> str:
        self._ensure_tokenizer_loaded()
        messages = _consolidate_system_messages([dict(m) for m in messages])
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
    ) -> tuple[str, list[int]]:
        self._ensure_tokenizer_loaded()
        template_tools = convert_tools_for_template(tools) if tools else None
        prompt = self._apply_chat_template(
            messages,
            template_tools,
            chat_template_kwargs=chat_template_kwargs,
            is_partial=is_partial,
            add_generation_prompt=add_generation_prompt,
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
        del repetition_penalty, presence_penalty
        if self._process is None:
            await self.start()
        if self.runtime_failed_reason is not None:
            raise RuntimeError(self.runtime_failed_reason)

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
        stop_matcher = _StopMatcher(stop_token_seqs, stop_strings)
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

        try:
            submit_params: dict[str, Any] = {
                "prompt_tokens": prompt_tokens,
                "max_new_tokens": budget,
                "sampling": sampling,
                "stop_token_seqs": stop_token_seqs,
            }
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
            p_tps, g_tps = self._compute_tps(
                prompt_tokens=num_prompt,
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
                first_token_at=first_token_at,
                generated_until=now_perf,
                prompt_tps=p_tps,
                generation_tps=g_tps,
            )
            if not terminal_seen and backend_id is not None:
                # Client-side stop matched before daemon finished; cancel remaining steps.
                await self._cancel_backend_request(backend_id)
                terminal_seen = True
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
            token_ids = list(prepared[1])
        else:
            _, token_ids = await asyncio.to_thread(
                self.prepare_chat_prompt,
                messages,
                tools,
                ct_kwargs,
                is_partial,
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
        floor = max(self._reported_footprint_bytes, self._weights_bytes())
        return max(group_footprint, floor)

    def get_stats(self) -> dict[str, Any]:
        running = self._process is not None and self._process.poll() is None
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
            "memory_budget_bytes": self.reserved_bytes(),
            "process_memory_bytes": self.process_memory_bytes(),
        }
        if self._last_stats:
            stats["backend_stats"] = dict(self._last_stats)
        if self._last_capacity:
            stats["capacity"] = dict(self._last_capacity)
        return stats

    def get_cache_stats(self) -> dict[str, Any] | None:
        return None

    def get_runtime_cache_stats(self) -> dict[str, Any] | None:
        if self._last_stats is None:
            return None
        s = self._last_stats
        hits = int(s.get("prefix_hits", 0)) + int(s.get("prefix_partial_hits", 0))
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
