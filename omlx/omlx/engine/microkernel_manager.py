"""Native microkernel registry, download, SHA-256 verification, atomic hot-update, and lifecycle manager.

Powers the `oMLX-extend` Integrations -> External Engines microkernel management UI
without shelling out to external bash scripts (safe inside standalone `.dmg` app bundles).

Security invariants (matching `omlx-extend/scripts/sync_microkernels.sh`):
- Validates `provider_id`, `version`, and archive `filename` against `SAFE_SEGMENT_RE`.
- Rejects non-HTTP(S)/file URL schemes.
- Verifies exact SHA-256 digest before archive extraction.
- Hardens tar extraction against path traversal, absolute paths, symlink/hardlink escapes,
  special files, and setuid/setgid/sticky mode bits.
- Validates SPI v2 `provider.json` and executable entrypoint containment inside bundle root.
- Atomically flips `Providers/<provider_id>/current` via `os.replace()` for zero-downtime hot-updates.
"""

from __future__ import annotations

import concurrent.futures
import datetime as _dt
import hashlib
import json
import logging
import os
import re
import shutil
import tarfile
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from .m5_sidecar import (
    M5RT_ARTIFACT_EXTENSIONS,
    M5RT_BUILTIN_PROVIDER_IDS,
    M5RT_DEFAULT_VERSION,
    M5RT_ENGINE_ID,
    M5RT_ENGINE_LABEL,
    M5RT_PROVIDER_ID,
    M5RT_SUPPORTED_MODEL_TYPES,
    M5RT_SUPPORTED_TOPOLOGIES,
    M5SidecarEngine,
    SUPPORTED_PROTOCOLS,
    _providers_base_dir,
    discover_external_engine_plugins,
    m5rt_status,
)

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/scoltzero/ninfer-metal/main/"
    "omlx-extend/registry/microkernels.json"
)
SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")
_ALLOWED_SCHEMES = frozenset({"https", "http", "file"})
_HTTP_TIMEOUT_SECONDS = 12.0
_DOWNLOAD_TIMEOUT_SECONDS = 180.0
_MAX_MANIFEST_BYTES = 2 * 1024 * 1024  # 2 MiB guard for JSON manifests


class MicrokernelManagerError(RuntimeError):
    """Structured error raised by microkernel lifecycle operations."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 400,
        active_models: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.active_models = list(active_models or ())


@dataclass
class _ManagerState:
    lock: threading.RLock = field(default_factory=threading.RLock)
    provider_locks: dict[str, threading.Lock] = field(default_factory=dict)
    refreshed: bool = False
    last_refreshed_at: str | None = None
    registry_source: str = "local"
    registry_error: str | None = None
    remote_catalog: list[dict[str, Any]] | None = None
    remote_releases: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_reload: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_provider_lock(self, provider_id: str) -> threading.Lock:
        with self.lock:
            lk = self.provider_locks.get(provider_id)
            if lk is None:
                lk = threading.Lock()
                self.provider_locks[provider_id] = lk
            return lk


_STATE = _ManagerState()


def reset_manager_state_for_tests() -> None:
    """Clear in-memory refresh cache (intended for unit tests)."""
    with _STATE.lock:
        _STATE.refreshed = False
        _STATE.last_refreshed_at = None
        _STATE.registry_source = "local"
        _STATE.registry_error = None
        _STATE.remote_catalog = None
        _STATE.remote_releases.clear()
        _STATE.pending_reload.clear()


def _writable_providers_dir() -> Path:
    env_dir = os.environ.get("OMLX_PROVIDERS_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser()
    return _providers_base_dir()


def _validate_safe_segment(raw: str, label: str) -> str:
    cleaned = str(raw or "").strip()
    if (
        not cleaned
        or cleaned in (".", "..")
        or ".." in cleaned
        or "/" in cleaned
        or "\\" in cleaned
        or "\x00" in cleaned
        or not SAFE_SEGMENT_RE.fullmatch(cleaned)
    ):
        raise MicrokernelManagerError(
            f"Refusing unsafe {label}: {raw!r}", status_code=400
        )
    return cleaned


def _validate_url(url: str, label: str = "URL") -> str:
    cleaned = str(url or "").strip()
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
        raise MicrokernelManagerError(
            f"Unsupported {label} scheme {parsed.scheme!r}: {cleaned!r}",
            status_code=400,
        )
    if parsed.scheme.lower() in ("http", "https") and not parsed.netloc:
        raise MicrokernelManagerError(f"Malformed {label}: {cleaned!r}", status_code=400)
    return cleaned


def _fetch_bytes(url: str, *, timeout: float = _HTTP_TIMEOUT_SECONDS, max_bytes: int | None = None) -> bytes:
    cleaned = _validate_url(url)
    parsed = urllib.parse.urlparse(cleaned)
    if parsed.scheme.lower() == "file":
        src = Path(urllib.request.url2pathname(parsed.path))
        if not src.is_file():
            raise MicrokernelManagerError(f"Local file not found: {src}", status_code=404)
        data = src.read_bytes()
        if max_bytes is not None and len(data) > max_bytes:
            raise MicrokernelManagerError(
                f"Manifest exceeds maximum allowed size ({max_bytes} bytes)", status_code=400
            )
        return data

    req = urllib.request.Request(
        cleaned,
        headers={
            "User-Agent": "oMLX-extend-microkernel-manager/1.0",
            "Accept": "application/json, application/octet-stream, */*",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = resp.read(64 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if max_bytes is not None and total > max_bytes:
                    raise MicrokernelManagerError(
                        f"Response exceeds maximum allowed size ({max_bytes} bytes)",
                        status_code=400,
                    )
                chunks.append(chunk)
            return b"".join(chunks)
    except urllib.error.HTTPError as exc:
        raise MicrokernelManagerError(
            f"HTTP {exc.code} fetching {cleaned}: {exc.reason}", status_code=502
        ) from exc
    except urllib.error.URLError as exc:
        raise MicrokernelManagerError(
            f"Network error fetching {cleaned}: {exc.reason}", status_code=502
        ) from exc
    except OSError as exc:
        raise MicrokernelManagerError(
            f"I/O error fetching {cleaned}: {exc}", status_code=502
        ) from exc


def _download_to_file(url: str, dest: Path, *, timeout: float = _DOWNLOAD_TIMEOUT_SECONDS) -> str:
    """Download `url` to `dest` and return its lowercase SHA-256 hex digest."""
    cleaned = _validate_url(url, "tarball URL")
    parsed = urllib.parse.urlparse(cleaned)
    hasher = hashlib.sha256()
    if parsed.scheme.lower() == "file":
        src = Path(urllib.request.url2pathname(parsed.path))
        if not src.is_file():
            raise MicrokernelManagerError(f"Local archive not found: {src}", status_code=404)
        with src.open("rb") as fin, dest.open("wb") as fout:
            while chunk := fin.read(1 << 20):
                hasher.update(chunk)
                fout.write(chunk)
        return hasher.hexdigest()

    req = urllib.request.Request(
        cleaned,
        headers={"User-Agent": "oMLX-extend-microkernel-manager/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, dest.open("wb") as fout:
            while chunk := resp.read(1 << 20):
                hasher.update(chunk)
                fout.write(chunk)
        return hasher.hexdigest()
    except urllib.error.HTTPError as exc:
        raise MicrokernelManagerError(
            f"HTTP {exc.code} downloading {cleaned}: {exc.reason}", status_code=502
        ) from exc
    except urllib.error.URLError as exc:
        raise MicrokernelManagerError(
            f"Network error downloading {cleaned}: {exc.reason}", status_code=502
        ) from exc
    except OSError as exc:
        raise MicrokernelManagerError(
            f"I/O error downloading {cleaned}: {exc}", status_code=502
        ) from exc


def _local_registry_candidates() -> list[Path]:
    cands: list[Path] = []
    env_file = os.environ.get("OMLX_MICROKERNEL_REGISTRY_FILE", "").strip()
    if env_file:
        cands.append(Path(env_file).expanduser())
    here = Path(__file__).resolve()
    # Monorepo layout: <repo>/omlx/omlx/engine/microkernel_manager.py -> <repo>/omlx-extend/registry/microkernels.json
    for ancestor in here.parents[:5]:
        cands.append(ancestor / "omlx-extend" / "registry" / "microkernels.json")
        cands.append(ancestor / "registry" / "microkernels.json")
    cands.append(_writable_providers_dir() / "microkernels.json")
    return cands


def _builtin_fallback_catalog() -> list[dict[str, Any]]:
    return [
        {
            "provider_id": M5RT_PROVIDER_ID,
            "engine_id": M5RT_ENGINE_ID,
            "display_name": M5RT_ENGINE_LABEL,
            "path": "A",
            "channel": "stable",
            "status": "shipped",
            "codename": "miv",
            "current_version": M5RT_DEFAULT_VERSION,
            "repo": "https://github.com/scoltzero/ninfer-metal",
            "latest_manifest_url": (
                "https://github.com/scoltzero/ninfer-metal/releases/latest/download/latest.json"
            ),
            "supported_topologies": list(M5RT_SUPPORTED_TOPOLOGIES),
            "supported_model_types": list(M5RT_SUPPORTED_MODEL_TYPES),
            "artifact_extensions": list(M5RT_ARTIFACT_EXTENSIONS),
            "core_abi": 1,
            "protocol_version": 2,
            "min_omlx_extend_version": "0.8.0-extend",
            "min_macos_version": "27.0",
            "certification_tier": "tier1-mainline",
        }
    ]


def _normalize_catalog_entries(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, dict):
        return []
    providers = raw.get("providers")
    if not isinstance(providers, list):
        return []
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in providers:
        if not isinstance(item, dict):
            continue
        pid = str(item.get("provider_id") or "").strip()
        if not pid or not SAFE_SEGMENT_RE.fullmatch(pid) or pid in seen:
            continue
        seen.add(pid)
        eid = str(item.get("engine_id") or pid).strip() or pid
        raw_path = str(item.get("path") or item.get("path_tier") or "B").strip()
        if raw_path.upper() == "A" or "path a" in raw_path.lower():
            path_tier = "Path A"
        else:
            path_tier = "Path B"
        out.append(
            {
                "provider_id": pid,
                "engine_id": eid,
                "display_name": str(item.get("display_name") or item.get("label") or eid).strip(),
                "path_tier": path_tier,
                "channel": str(item.get("channel") or "stable").strip(),
                "status": str(item.get("status") or "shipped").strip(),
                "codename": str(item.get("codename") or "").strip() or None,
                "current_version": str(
                    item.get("current_version") or item.get("version") or ""
                ).strip()
                or None,
                "repo": str(item.get("repo") or "").strip() or None,
                "latest_manifest_url": str(item.get("latest_manifest_url") or "").strip() or None,
                "supported_topologies": [
                    str(x).strip()
                    for x in (item.get("supported_topologies") or ())
                    if str(x).strip()
                ],
                "supported_model_types": [
                    str(x).strip()
                    for x in (item.get("supported_model_types") or ())
                    if str(x).strip()
                ],
                "artifact_extensions": [
                    str(x).strip()
                    for x in (item.get("artifact_extensions") or ())
                    if str(x).strip()
                ],
                "core_abi": item.get("core_abi") if isinstance(item.get("core_abi"), int) else 1,
                "protocol_version": (
                    item.get("protocol_version")
                    if isinstance(item.get("protocol_version"), int)
                    else 2
                ),
                "min_omlx_extend_version": str(
                    item.get("min_omlx_extend_version") or ""
                ).strip()
                or None,
                "min_macos_version": str(item.get("min_macos_version") or "").strip() or None,
                "certification_tier": str(item.get("certification_tier") or "").strip() or None,
            }
        )
    return out


def _load_local_registry() -> tuple[list[dict[str, Any]], str]:
    for cand in _local_registry_candidates():
        if not cand.is_file():
            continue
        try:
            raw = json.loads(cand.read_text(encoding="utf-8"))
            entries = _normalize_catalog_entries(raw)
            if entries:
                return entries, "local"
        except Exception:
            logger.warning("Failed reading local microkernel registry %s", cand, exc_info=True)
    return _normalize_catalog_entries({"providers": _builtin_fallback_catalog()}), "builtin"


def _registry_url() -> str:
    env_url = os.environ.get("OMLX_MICROKERNEL_REGISTRY_URL", "").strip()
    return env_url or DEFAULT_REGISTRY_URL


def _parse_latest_manifest(
    raw: dict[str, Any], latest_url: str, expected_pid: str
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise MicrokernelManagerError("latest.json is not a JSON object", status_code=502)
    pid = str(raw.get("provider_id") or expected_pid).strip()
    if pid != expected_pid:
        raise MicrokernelManagerError(
            f"latest.json provider_id mismatch: expected '{expected_pid}', got '{pid}'",
            status_code=502,
        )
    proto = raw.get("protocol_version")
    if proto is not None and (isinstance(proto, bool) or proto != 2):
        raise MicrokernelManagerError(
            f"latest.json protocol_version={proto!r} is not SPI v2",
            status_code=502,
        )
    version = _validate_safe_segment(
        str(raw.get("display_version") or raw.get("version") or raw.get("semver") or "").strip(),
        "version",
    )
    tar_url = ""
    tar_sha = ""
    tar_name = ""
    assets = raw.get("assets")
    if isinstance(assets, dict) and isinstance(assets.get("macos-arm64"), dict):
        arm64 = assets["macos-arm64"]
        tar_url = str(arm64.get("url") or "").strip()
        tar_sha = str(arm64.get("sha256") or "").strip().lower()
        raw_filename = str(arm64.get("filename") or "").strip()
        if raw_filename:
            if (
                "/" in raw_filename
                or "\\" in raw_filename
                or ".." in raw_filename
                or PurePosixPath(raw_filename).name != raw_filename
            ):
                raise MicrokernelManagerError(
                    f"Refusing unsafe filename in latest.json: {raw_filename!r}",
                    status_code=502,
                )
            tar_name = _validate_safe_segment(raw_filename, "filename")
    if not tar_url:
        tarball_rel = str(raw.get("tarball") or "").strip()
        if tarball_rel:
            tar_name = Path(tarball_rel).name
            tar_url = urllib.parse.urljoin(latest_url, tarball_rel)
    if not tar_sha:
        tar_sha = str(raw.get("sha256") or raw.get("tarball_sha256") or "").strip().lower()
    if not tar_name and tar_url:
        tar_name = Path(tar_url.split("?", 1)[0]).name
    if not tar_name:
        tar_name = f"{expected_pid}-{version}.tar.gz"
    tar_name = _validate_safe_segment(Path(tar_name).name, "filename")
    if not tar_url or not tar_sha or len(tar_sha) != 64 or not re.fullmatch(r"[0-9a-f]{64}", tar_sha):
        raise MicrokernelManagerError(
            "latest.json must provide a valid macos-arm64 tarball URL and 64-char SHA-256",
            status_code=502,
        )
    _validate_url(tar_url, "tarball URL")
    return {
        "provider_id": expected_pid,
        "remote_version": version,
        "semver": str(raw.get("semver") or "").strip() or None,
        "tar_url": tar_url,
        "sha256": tar_sha,
        "tar_name": tar_name,
        "latest_manifest_url": latest_url,
    }


def _fetch_latest_manifest(latest_url: str, expected_pid: str) -> dict[str, Any]:
    raw_bytes = _fetch_bytes(
        latest_url, timeout=_HTTP_TIMEOUT_SECONDS, max_bytes=_MAX_MANIFEST_BYTES
    )
    try:
        parsed = json.loads(raw_bytes.decode("utf-8"))
    except Exception as exc:
        raise MicrokernelManagerError(
            f"Invalid JSON in {latest_url}: {exc}", status_code=502
        ) from exc
    return _parse_latest_manifest(parsed, latest_url, expected_pid)


def _version_tuple(raw: str | None) -> tuple[int, ...] | None:
    text = str(raw or "").strip()
    if not text:
        return None
    # Strip optional prefix such as "miv-v", "v", "release-"
    m = re.search(r"(\d+(?:\.\d+)+)", text)
    if not m:
        return None
    return tuple(int(part) for part in m.group(1).split("."))


def _is_update_available(installed_version: str | None, remote_version: str | None) -> bool:
    inst = str(installed_version or "").strip()
    rem = str(remote_version or "").strip()
    if not inst or not rem:
        return False
    if inst == rem:
        return False
    t_inst = _version_tuple(inst)
    t_rem = _version_tuple(rem)
    if t_inst is not None and t_rem is not None:
        max_len = max(len(t_inst), len(t_rem))
        p_inst = t_inst + (0,) * (max_len - len(t_inst))
        p_rem = t_rem + (0,) * (max_len - len(t_rem))
        if p_rem != p_inst:
            return p_rem > p_inst
        # Same numeric tuple (e.g. "0.0.1" vs "miv-v0.01"): treat as equal unless exact mismatch is meaningful
        return False
    return rem != inst


def is_microkernel_enabled(
    global_settings: Any | None,
    *,
    provider_id: str | None = None,
    engine_id: str | None = None,
) -> bool:
    """Return False when `provider_id` or `engine_id` is in `integrations.disabled_microkernels`."""
    integrations = getattr(global_settings, "integrations", None)
    raw = getattr(integrations, "disabled_microkernels", None)
    if not raw:
        return True
    disabled = {str(x).strip() for x in raw if str(x).strip()}
    if not disabled:
        return True
    if provider_id and str(provider_id).strip() in disabled:
        return False
    if engine_id and str(engine_id).strip() in disabled:
        return False
    # Map built-in legacy/canonical aliases cleanly
    if engine_id == M5RT_ENGINE_ID or provider_id in M5RT_BUILTIN_PROVIDER_IDS:
        if M5RT_ENGINE_ID in disabled or any(p in disabled for p in M5RT_BUILTIN_PROVIDER_IDS):
            return False
    return True


def _is_path_inside(candidate: Path | None, root: Path) -> bool:
    if candidate is None:
        return False
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except (OSError, ValueError):
        return False


def _collect_installed_providers(
    global_settings: Any, engine_pool: Any | None
) -> list[dict[str, Any]]:
    """Inspect currently installed Path A (`miv`) and Path B microkernels on disk."""
    providers_root = _writable_providers_dir()
    items: list[dict[str, Any]] = []

    # 1. Path A built-in miv Core Backend (when available on disk)
    m5_info = m5rt_status(global_settings, engine_pool)
    if m5_info.get("available"):
        pid = str(m5_info.get("provider_id") or M5RT_PROVIDER_ID)
        eid = str(m5_info.get("engine_id") or M5RT_ENGINE_ID)
        bin_str = m5_info.get("binary_path") or m5_info.get("path")
        bin_path = Path(bin_str) if bin_str else None
        managed_root = providers_root / pid
        managed = managed_root.is_dir() and _is_path_inside(bin_path, managed_root)
        items.append(
            {
                "provider_id": pid,
                "engine_id": eid,
                "display_name": str(m5_info.get("display_name") or M5RT_ENGINE_LABEL),
                "version": str(m5_info.get("version") or M5RT_DEFAULT_VERSION),
                "codename": m5_info.get("codename") or "miv",
                "path_tier": "Path A",
                "protocol_version": int(m5_info.get("protocol_version") or 2),
                "installed": True,
                "enabled": is_microkernel_enabled(
                    global_settings, provider_id=pid, engine_id=eid
                ),
                "managed": managed,
                "can_delete": managed,
                "binary_path": str(bin_path) if bin_path else None,
                "install_root": str(managed_root) if managed else (str(bin_path.parent) if bin_path else None),
                "supported_topologies": list(m5_info.get("supported_topologies") or ()),
                "supported_model_types": list(m5_info.get("supported_model_types") or ()),
                "artifact_extensions": list(m5_info.get("artifact_extensions") or ()),
                "models": list(m5_info.get("models") or ()),
                "latest_manifest_url": None,
            }
        )

    # 2. Auto-discovered Path B SPI v2 microkernels
    for plugin_id, plugin in sorted(discover_external_engine_plugins().items()):
        info: dict[str, Any] = {}
        if callable(plugin.status):
            try:
                res = plugin.status(global_settings, engine_pool)
                if isinstance(res, dict):
                    info = res
            except Exception:
                logger.warning("Path B plugin %r status() failed", plugin_id, exc_info=True)
        if not bool(info.get("available", True)):
            continue
        pid = str(info.get("provider_id") or plugin_id)
        eid = str(info.get("engine_id") or plugin_id)
        bin_str = info.get("binary_path") or info.get("path")
        bin_path = Path(bin_str) if bin_str else None
        managed_root = providers_root / pid
        managed = managed_root.is_dir() and _is_path_inside(bin_path, managed_root)
        local_latest_url: str | None = None
        if managed and (managed_root / "current" / "provider.json").is_file():
            try:
                mjson = json.loads(
                    (managed_root / "current" / "provider.json").read_text(encoding="utf-8")
                )
                if isinstance(mjson, dict) and mjson.get("latest_manifest_url"):
                    local_latest_url = str(mjson["latest_manifest_url"]).strip() or None
            except Exception:
                pass
        items.append(
            {
                "provider_id": pid,
                "engine_id": eid,
                "display_name": str(info.get("display_name") or info.get("label") or plugin.label),
                "version": str(info.get("version") or "unknown"),
                "codename": info.get("codename"),
                "path_tier": str(info.get("path_tier") or "Path B"),
                "protocol_version": int(info.get("protocol_version") or 2),
                "installed": True,
                "enabled": is_microkernel_enabled(
                    global_settings, provider_id=pid, engine_id=eid
                ),
                "managed": managed,
                "can_delete": managed,
                "binary_path": str(bin_path) if bin_path else None,
                "install_root": str(managed_root) if managed else (str(bin_path.parent) if bin_path else None),
                "supported_topologies": list(info.get("supported_topologies") or ()),
                "supported_model_types": list(info.get("supported_model_types") or ()),
                "artifact_extensions": list(info.get("artifact_extensions") or ()),
                "models": list(info.get("models") or ()),
                "latest_manifest_url": local_latest_url,
            }
        )

    return items


def _check_one_release(pid: str, url: str) -> tuple[str, dict[str, Any]]:
    try:
        rel = _fetch_latest_manifest(url, pid)
        rel["error"] = None
        return pid, rel
    except MicrokernelManagerError as exc:
        return pid, {"provider_id": pid, "latest_manifest_url": url, "error": str(exc)}
    except Exception as exc:
        return pid, {"provider_id": pid, "latest_manifest_url": url, "error": str(exc)}


def list_catalog(
    global_settings: Any,
    engine_pool: Any | None = None,
    *,
    refresh: bool = False,
) -> dict[str, Any]:
    """Return microkernel registry catalog + installed cores state.

    When `refresh=False`, performs zero network requests; `can_update` stays False
    unless a prior `refresh=True` call discovered a newer remote version.
    """
    installed_list = _collect_installed_providers(global_settings, engine_pool)

    if refresh:
        reg_url = _registry_url()
        remote_entries: list[dict[str, Any]] | None = None
        registry_source = "remote"
        registry_error: str | None = None
        try:
            raw_bytes = _fetch_bytes(
                reg_url, timeout=_HTTP_TIMEOUT_SECONDS, max_bytes=_MAX_MANIFEST_BYTES
            )
            parsed = json.loads(raw_bytes.decode("utf-8"))
            remote_entries = _normalize_catalog_entries(parsed)
            if not remote_entries:
                raise MicrokernelManagerError("Remote microkernels.json contains no providers")
        except Exception as exc:
            registry_error = str(exc)
            remote_entries, registry_source = _load_local_registry()

        # Gather all (provider_id, latest_manifest_url) targets to probe
        probe_targets: dict[str, str] = {}
        for entry in remote_entries:
            if entry.get("latest_manifest_url"):
                probe_targets[entry["provider_id"]] = str(entry["latest_manifest_url"])
        for inst in installed_list:
            pid = inst["provider_id"]
            if pid not in probe_targets and inst.get("latest_manifest_url"):
                probe_targets[pid] = str(inst["latest_manifest_url"])

        releases: dict[str, dict[str, Any]] = {}
        if probe_targets:
            max_workers = min(4, len(probe_targets))
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = [
                    pool.submit(_check_one_release, pid, url)
                    for pid, url in probe_targets.items()
                ]
                for fut in concurrent.futures.as_completed(futs):
                    pid, rel = fut.result()
                    releases[pid] = rel

        now_iso = _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()
        with _STATE.lock:
            _STATE.remote_catalog = remote_entries
            _STATE.remote_releases = releases
            _STATE.refreshed = True
            _STATE.last_refreshed_at = now_iso
            _STATE.registry_source = registry_source
            _STATE.registry_error = registry_error

    with _STATE.lock:
        if _STATE.remote_catalog is not None:
            catalog_base = [dict(e) for e in _STATE.remote_catalog]
            registry_source = _STATE.registry_source
        else:
            catalog_base, registry_source = _load_local_registry()
        refreshed = _STATE.refreshed
        last_refreshed_at = _STATE.last_refreshed_at
        registry_error = _STATE.registry_error
        releases_snap = {k: dict(v) for k, v in _STATE.remote_releases.items()}
        pending_snap = {k: dict(v) for k, v in _STATE.pending_reload.items()}

    installed_by_pid = {item["provider_id"]: item for item in installed_list}
    catalog_by_pid = {item["provider_id"]: item for item in catalog_base}

    # Enrich catalog rows
    enriched_catalog: list[dict[str, Any]] = []
    for entry in catalog_base:
        pid = entry["provider_id"]
        inst = installed_by_pid.get(pid)
        rel = releases_snap.get(pid) or {}
        remote_version = rel.get("remote_version") or entry.get("current_version")
        installed_version = inst["version"] if inst else None
        has_update = bool(
            refreshed
            and inst is not None
            and rel.get("remote_version")
            and _is_update_available(installed_version, rel.get("remote_version"))
        )
        enriched_catalog.append(
            {
                **entry,
                "remote_version": remote_version,
                "installed": inst is not None,
                "installed_version": installed_version,
                "managed": bool(inst["managed"]) if inst else False,
                "enabled": bool(inst["enabled"]) if inst else True,
                "has_update": has_update,
                "can_download": inst is None or not bool(inst["managed"]),
                "update_check_error": rel.get("error"),
            }
        )

    # Enrich installed rows
    enriched_installed: list[dict[str, Any]] = []
    for inst in installed_list:
        pid = inst["provider_id"]
        cat = catalog_by_pid.get(pid) or {}
        rel = releases_snap.get(pid) or {}
        latest_url = inst.get("latest_manifest_url") or cat.get("latest_manifest_url")
        remote_version = rel.get("remote_version")
        has_update = bool(
            refreshed
            and remote_version
            and _is_update_available(inst["version"], remote_version)
        )
        # Update button only lights up AFTER refresh AND when a newer remote release exists!
        can_update = bool(refreshed and has_update and latest_url)
        pend = pending_snap.get(pid)
        active_model_ids = [str(m.get("id") or "") for m in inst.get("models") or () if m.get("id")]
        # Clear pending reload banner once all running models on this engine have been unloaded
        if pend and not active_model_ids:
            with _STATE.lock:
                _STATE.pending_reload.pop(pid, None)
            pend = None
        enriched_installed.append(
            {
                **inst,
                "latest_manifest_url": latest_url,
                "remote_version": remote_version,
                "has_update": has_update,
                "can_update": can_update,
                "update_check_error": rel.get("error"),
                "pending_reload": bool(pend),
                "pending_reload_version": pend.get("version") if pend else None,
            }
        )

    return {
        "providers_dir": str(_writable_providers_dir()),
        "registry_url": _registry_url(),
        "registry_source": registry_source,
        "registry_error": registry_error,
        "refreshed": refreshed,
        "last_refreshed_at": last_refreshed_at,
        "catalog": enriched_catalog,
        "installed": enriched_installed,
    }


def _require_inside(candidate: Path, root: Path, label: str) -> None:
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise MicrokernelManagerError(
            f"Archive entry escapes staging directory ({label}): {candidate}",
            status_code=400,
        ) from exc


def _validate_tar_member(member: tarfile.TarInfo, dest: Path) -> None:
    raw = member.name
    if not raw or "\x00" in raw or "\\" in raw:
        raise MicrokernelManagerError(f"Refusing unsafe tar path: {raw!r}", status_code=400)
    pp = PurePosixPath(raw)
    p = Path(raw)
    if pp.is_absolute() or p.is_absolute() or ".." in pp.parts or ".." in p.parts:
        raise MicrokernelManagerError(f"Refusing unsafe tar path: {raw!r}", status_code=400)
    if not (member.isreg() or member.isdir() or member.issym() or member.islnk()):
        raise MicrokernelManagerError(
            f"Refusing unsupported tar special entry: {raw!r}", status_code=400
        )
    member.mode = member.mode & 0o777
    target = Path(os.path.normpath(str(dest / p)))
    _require_inside(target, dest, raw)

    if member.issym():
        link_raw = member.linkname
        if not link_raw or "\x00" in link_raw or "\\" in link_raw:
            raise MicrokernelManagerError(
                f"Refusing unsafe tar symlink target for {raw!r}: {link_raw!r}",
                status_code=400,
            )
        lpp = PurePosixPath(link_raw)
        lp = Path(link_raw)
        if lpp.is_absolute() or lp.is_absolute():
            raise MicrokernelManagerError(
                f"Refusing absolute tar symlink target for {raw!r}: {link_raw!r}",
                status_code=400,
            )
        resolved_link = Path(os.path.normpath(str(target.parent / lp)))
        _require_inside(resolved_link, dest, f"symlink {raw} -> {link_raw}")
    elif member.islnk():
        link_raw = member.linkname
        if not link_raw or "\x00" in link_raw or "\\" in link_raw:
            raise MicrokernelManagerError(
                f"Refusing unsafe tar hardlink target for {raw!r}: {link_raw!r}",
                status_code=400,
            )
        lpp = PurePosixPath(link_raw)
        lp = Path(link_raw)
        if lpp.is_absolute() or lp.is_absolute() or ".." in lpp.parts or ".." in lp.parts:
            raise MicrokernelManagerError(
                f"Refusing unsafe tar hardlink target for {raw!r}: {link_raw!r}",
                status_code=400,
            )
        resolved_link = Path(os.path.normpath(str(dest / lp)))
        _require_inside(resolved_link, dest, f"hardlink {raw} -> {link_raw}")


def _extract_tar_safely(tar_path: Path, dest: Path) -> Path:
    dest_resolved = dest.resolve()
    try:
        with tarfile.open(tar_path, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                _validate_tar_member(m, dest_resolved)
            try:
                tf.extractall(dest_resolved, members=members, filter="tar")
            except TypeError:
                for m in members:
                    p = Path(m.name)
                    rel_parts = [part for part in p.parts if part not in ("", ".")]
                    if not rel_parts:
                        continue
                    cursor = dest_resolved
                    for part in rel_parts[:-1]:
                        cursor = cursor / part
                        if cursor.is_symlink():
                            raise MicrokernelManagerError(
                                f"Refusing extraction through symlink dir: {m.name!r}",
                                status_code=400,
                            )
                    target = cursor / rel_parts[-1]
                    if target.is_symlink():
                        raise MicrokernelManagerError(
                            f"Refusing extraction over existing symlink: {m.name!r}",
                            status_code=400,
                        )
                    _require_inside(target.parent.resolve(), dest_resolved, m.name)
                    tf.extract(m, path=dest_resolved, set_attrs=True)
    except MicrokernelManagerError:
        raise
    except (tarfile.TarError, OSError) as exc:
        raise MicrokernelManagerError(
            f"Failed to extract archive {tar_path.name}: {exc}", status_code=400
        ) from exc

    if (dest_resolved / "provider.json").is_file():
        return dest_resolved
    subdirs = [
        p
        for p in sorted(dest_resolved.iterdir())
        if p.is_dir() and (p / "provider.json").is_file()
    ]
    if len(subdirs) == 1:
        return subdirs[0]
    raise MicrokernelManagerError(
        f"Archive {tar_path.name} must contain a single SPI v2 provider.json bundle root",
        status_code=400,
    )


def _validate_extracted_bundle(bundle_dir: Path, expected_pid: str) -> dict[str, Any]:
    bundle = bundle_dir.resolve()
    manifest_path = bundle / "provider.json"
    if not manifest_path.is_file():
        raise MicrokernelManagerError(
            f"Extracted bundle missing provider.json: {bundle}", status_code=400
        )
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise MicrokernelManagerError(
            f"Invalid provider.json in archive: {exc}", status_code=400
        ) from exc
    if not isinstance(data, dict):
        raise MicrokernelManagerError("provider.json must be a JSON object", status_code=400)
    pid = str(data.get("provider_id") or "").strip()
    if pid != expected_pid:
        raise MicrokernelManagerError(
            f"provider.json provider_id mismatch: expected '{expected_pid}', got '{pid}'",
            status_code=400,
        )
    proto = data.get("protocol_version")
    supported = data.get("supported_protocols") or []
    if proto not in SUPPORTED_PROTOCOLS and 2 not in supported:
        raise MicrokernelManagerError(
            f"provider.json for '{pid}' must declare SPI protocol_version=2 (got {proto!r})",
            status_code=400,
        )
    entry_rel = None
    for key in ("entrypoint", "binary", "daemon_path"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            entry_rel = val.strip()
            break
    if not entry_rel:
        raise MicrokernelManagerError(
            f"provider.json for '{pid}' missing entrypoint/binary", status_code=400
        )
    entry_path = Path(entry_rel)
    entry_path = (bundle / entry_path).resolve() if not entry_path.is_absolute() else entry_path.resolve()
    try:
        entry_path.relative_to(bundle)
    except ValueError as exc:
        raise MicrokernelManagerError(
            f"entrypoint escapes bundle directory: {entry_path}", status_code=400
        ) from exc
    if not entry_path.is_file() or not os.access(entry_path, os.X_OK):
        raise MicrokernelManagerError(
            f"entrypoint is missing or not executable: {entry_path}", status_code=400
        )
    return data


def _atomic_flip_current(provider_root: Path, target_version: str) -> None:
    tmp_link = provider_root / f".current.tmp.{os.getpid()}.{threading.get_ident()}"
    if tmp_link.is_symlink() or tmp_link.exists():
        tmp_link.unlink()
    os.symlink(f"versions/{target_version}", tmp_link)
    os.replace(tmp_link, provider_root / "current")


def _active_models_for_provider(
    engine_pool: Any | None, *, provider_id: str, engine_id: str
) -> list[str]:
    entries = getattr(engine_pool, "_entries", None) or {}
    active: list[str] = []
    for model_id, entry in list(entries.items()):
        engine = getattr(entry, "engine", None)
        if not isinstance(engine, M5SidecarEngine):
            continue
        if (
            engine.external_engine_id == engine_id
            or getattr(engine, "provider_id", None) == provider_id
        ):
            active.append(str(model_id))
    return active


def _find_provider_entry(
    query: str, global_settings: Any, engine_pool: Any | None
) -> tuple[str, str, str | None]:
    """Resolve `(provider_id, engine_id, latest_manifest_url)` from catalog or installed list."""
    cleaned = str(query or "").strip()
    if not cleaned:
        raise MicrokernelManagerError("provider_id is required", status_code=400)
    with _STATE.lock:
        catalog = list(_STATE.remote_catalog) if _STATE.remote_catalog is not None else None
    if catalog is None:
        catalog, _ = _load_local_registry()
    for item in catalog:
        if cleaned in (item.get("provider_id"), item.get("engine_id")):
            pid = _validate_safe_segment(item["provider_id"], "provider_id")
            return pid, str(item.get("engine_id") or pid), item.get("latest_manifest_url")
    for inst in _collect_installed_providers(global_settings, engine_pool):
        if cleaned in (inst.get("provider_id"), inst.get("engine_id")):
            pid = _validate_safe_segment(inst["provider_id"], "provider_id")
            return pid, str(inst.get("engine_id") or pid), inst.get("latest_manifest_url")
    raise MicrokernelManagerError(
        f"Microkernel '{cleaned}' not found in registry or installed providers",
        status_code=404,
    )


def _persist_enabled_flag(
    global_settings: Any, *, provider_id: str, engine_id: str, enabled: bool
) -> None:
    integrations = getattr(global_settings, "integrations", None)
    if integrations is None:
        return
    current = list(getattr(integrations, "disabled_microkernels", None) or ())
    aliases = {provider_id, engine_id}
    if engine_id == M5RT_ENGINE_ID or provider_id in M5RT_BUILTIN_PROVIDER_IDS:
        aliases.update(M5RT_BUILTIN_PROVIDER_IDS)
        aliases.add(M5RT_ENGINE_ID)
    filtered = [x for x in current if str(x).strip() and str(x).strip() not in aliases]
    if not enabled:
        filtered.append(provider_id)
    integrations.disabled_microkernels = filtered
    save_fn = getattr(global_settings, "save", None)
    if callable(save_fn):
        save_fn()


def install_or_update_provider(
    query: str,
    global_settings: Any,
    engine_pool: Any | None = None,
) -> dict[str, Any]:
    """Download, verify SHA-256, unpack, validate SPI v2, and atomically activate a microkernel."""
    provider_id, engine_id, latest_url = _find_provider_entry(
        query, global_settings, engine_pool
    )
    if not latest_url:
        raise MicrokernelManagerError(
            f"Microkernel '{provider_id}' does not declare a latest_manifest_url",
            status_code=400,
        )

    lk = _STATE.get_provider_lock(provider_id)
    with lk:
        release = _fetch_latest_manifest(latest_url, provider_id)
        version = release["remote_version"]
        tar_url = release["tar_url"]
        expected_sha = release["sha256"]
        tar_name = release["tar_name"]

        providers_root = _writable_providers_dir()
        provider_root = (providers_root / provider_id).resolve()
        try:
            provider_root.relative_to(providers_root.resolve())
        except ValueError as exc:
            raise MicrokernelManagerError(
                f"Refusing unsafe provider root: {provider_root}", status_code=400
            ) from exc

        with tempfile.TemporaryDirectory(prefix="omlx-mk-install-") as tmp_str:
            stage = Path(tmp_str)
            tar_path = stage / tar_name
            actual_sha = _download_to_file(tar_url, tar_path)
            if actual_sha != expected_sha:
                raise MicrokernelManagerError(
                    f"SHA-256 mismatch for {tar_name}: expected {expected_sha[:12]}…, got {actual_sha[:12]}…",
                    status_code=400,
                )
            unpack_dir = stage / "unpacked"
            unpack_dir.mkdir(parents=True, exist_ok=True)
            bundle_root = _extract_tar_safely(tar_path, unpack_dir)
            manifest = _validate_extracted_bundle(bundle_root, provider_id)
            effective_engine_id = str(manifest.get("engine_id") or engine_id).strip() or engine_id

            versions_dir = provider_root / "versions"
            versions_dir.mkdir(parents=True, exist_ok=True)
            dest_version_dir = versions_dir / version
            tmp_version_dir = versions_dir / f".{version}.staging.{os.getpid()}.{threading.get_ident()}"
            if tmp_version_dir.exists():
                shutil.rmtree(tmp_version_dir)
            shutil.copytree(bundle_root, tmp_version_dir, symlinks=True)

            # Record previous version for clean rollback
            cur_link = provider_root / "current"
            if cur_link.is_symlink() or cur_link.is_dir():
                try:
                    prev_name = cur_link.resolve().name
                    if prev_name and prev_name != version and (versions_dir / prev_name).is_dir():
                        (provider_root / ".previous_version").write_text(
                            prev_name + "\n", encoding="utf-8"
                        )
                except OSError:
                    pass

            if dest_version_dir.exists():
                backup_dir = versions_dir / f".{version}.old.{os.getpid()}.{threading.get_ident()}"
                os.replace(dest_version_dir, backup_dir)
                os.replace(tmp_version_dir, dest_version_dir)
                shutil.rmtree(backup_dir, ignore_errors=True)
            else:
                os.replace(tmp_version_dir, dest_version_dir)

            _atomic_flip_current(provider_root, version)

        # Auto-enable newly installed/updated microkernel
        _persist_enabled_flag(
            global_settings,
            provider_id=provider_id,
            engine_id=effective_engine_id,
            enabled=True,
        )

        # Update cached remote release record & pending reload marker
        active_models = _active_models_for_provider(
            engine_pool, provider_id=provider_id, engine_id=effective_engine_id
        )
        with _STATE.lock:
            _STATE.remote_releases[provider_id] = {**release, "error": None}
            if active_models:
                _STATE.pending_reload[provider_id] = {
                    "version": version,
                    "models": active_models,
                }
            else:
                _STATE.pending_reload.pop(provider_id, None)

        state = list_catalog(global_settings, engine_pool, refresh=False)
        return {
            "success": True,
            "provider_id": provider_id,
            "engine_id": effective_engine_id,
            "version": version,
            "sha256": actual_sha,
            "pending_reload_models": active_models,
            **state,
        }


def set_provider_enabled(
    query: str,
    enabled: bool,
    global_settings: Any,
    engine_pool: Any | None = None,
) -> dict[str, Any]:
    """Enable or disable a microkernel and persist to `settings.json`."""
    provider_id, engine_id, _ = _find_provider_entry(query, global_settings, engine_pool)
    _persist_enabled_flag(
        global_settings,
        provider_id=provider_id,
        engine_id=engine_id,
        enabled=bool(enabled),
    )
    state = list_catalog(global_settings, engine_pool, refresh=False)
    return {
        "success": True,
        "provider_id": provider_id,
        "engine_id": engine_id,
        "enabled": bool(enabled),
        **state,
    }


async def delete_provider(
    query: str,
    global_settings: Any,
    engine_pool: Any | None = None,
    *,
    unload_active: bool = False,
) -> dict[str, Any]:
    """Delete a managed microkernel bundle under `Providers/<provider_id>/`.

    Refuses to delete unmanaged/external paths. If models are currently loaded on
    this microkernel and `unload_active=False`, raises 409 with `active_models`.
    """
    provider_id, engine_id, _ = _find_provider_entry(query, global_settings, engine_pool)
    providers_root = _writable_providers_dir().resolve()
    provider_root = (providers_root / provider_id).resolve()
    try:
        provider_root.relative_to(providers_root)
    except ValueError as exc:
        raise MicrokernelManagerError(
            f"Refusing to delete outside Providers directory: {provider_root}",
            status_code=400,
        ) from exc
    if provider_root == providers_root or not provider_root.is_dir():
        raise MicrokernelManagerError(
            f"Microkernel '{provider_id}' is not a managed install in {providers_root}; uncheck it to disable instead.",
            status_code=400,
        )

    active_models = _active_models_for_provider(
        engine_pool, provider_id=provider_id, engine_id=engine_id
    )
    if active_models and not unload_active:
        raise MicrokernelManagerError(
            f"Microkernel '{provider_id}' is currently serving {len(active_models)} active model(s). Unload them before deleting.",
            status_code=409,
            active_models=active_models,
        )
    if active_models and unload_active and engine_pool is not None:
        unload_fn = getattr(engine_pool, "unload_model", None)
        if callable(unload_fn):
            for mid in active_models:
                await unload_fn(mid)

    lk = _STATE.get_provider_lock(provider_id)
    with lk:
        if provider_root.is_dir():
            shutil.rmtree(provider_root)
        _persist_enabled_flag(
            global_settings,
            provider_id=provider_id,
            engine_id=engine_id,
            enabled=True,
        )
        with _STATE.lock:
            _STATE.pending_reload.pop(provider_id, None)

    state = list_catalog(global_settings, engine_pool, refresh=False)
    return {
        "success": True,
        "provider_id": provider_id,
        "engine_id": engine_id,
        "deleted": True,
        "unloaded_models": active_models if unload_active else [],
        **state,
    }
