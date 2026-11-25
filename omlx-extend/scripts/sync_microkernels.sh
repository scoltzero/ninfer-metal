#!/usr/bin/env bash
set -euo pipefail

# Microkernel sync, verification, and rollback manager for oMLX-extend (Path A & Path B).
#
# Reads `omlx-extend/registry/microkernels.json`, downloads each provider's `latest.json`
# and macOS ARM64 release tarball, verifies SHA-256 and SPI v2 `provider.json` + executable
# entrypoint, unpacks into:
#   ~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<version>/
# and atomically flips the `current` symlink (zero-downtime hot-update / rollback).
#
# Usage:
#   ./omlx-extend/scripts/sync_microkernels.sh --list
#   ./omlx-extend/scripts/sync_microkernels.sh --provider <provider_id|engine_id> [--verify-only]
#   ./omlx-extend/scripts/sync_microkernels.sh --all [--verify-only]
#   ./omlx-extend/scripts/sync_microkernels.sh --verify-only
#   ./omlx-extend/scripts/sync_microkernels.sh --rollback <provider_id|engine_id>
#
# Options:
#   --registry FILE         Path to microkernels.json (default: omlx-extend/registry/microkernels.json)
#   --providers-dir DIR     Override Providers install root (default: $OMLX_PROVIDERS_DIR or
#                           ~/Library/Application Support/oMLX/Providers)
#   -h, --help              Show this help message

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REGISTRY_FILE="${EXTEND_DIR}/registry/microkernels.json"
PROVIDERS_DIR="${OMLX_PROVIDERS_DIR:-${HOME}/Library/Application Support/oMLX/Providers}"

MODE=""
TARGET_PROVIDER=""
VERIFY_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)
      MODE="list"
      shift
      ;;
    --provider)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --provider requires an argument" >&2
        exit 2
      fi
      MODE="sync"
      TARGET_PROVIDER="$2"
      shift 2
      ;;
    --all)
      MODE="all"
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --rollback)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --rollback requires a provider_id or engine_id" >&2
        exit 2
      fi
      MODE="rollback"
      TARGET_PROVIDER="$2"
      shift 2
      ;;
    --registry)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --registry requires a path" >&2
        exit 2
      fi
      REGISTRY_FILE="$2"
      shift 2
      ;;
    --providers-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --providers-dir requires a directory path" >&2
        exit 2
      fi
      PROVIDERS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '4,23p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
    MODE="verify_installed"
  else
    sed -n '4,23p' "$0"
    exit 2
  fi
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required by sync_microkernels.sh" >&2
  exit 1
fi

if [[ ! -f "${REGISTRY_FILE}" && "${MODE}" != "verify_installed" ]]; then
  echo "ERROR: Registry file not found: ${REGISTRY_FILE}" >&2
  exit 1
fi

STAGE_DIR=""
cleanup() {
  if [[ -n "${STAGE_DIR}" && -d "${STAGE_DIR}" ]]; then
    rm -rf "${STAGE_DIR}"
  fi
}
trap cleanup EXIT

fetch_url() {
  local url="$1"
  local dest="$2"
  if [[ "${url}" == file://* ]]; then
    local src="${url#file://}"
    if [[ ! -f "${src}" ]]; then
      echo "ERROR: Local file not found: ${src}" >&2
      return 1
    fi
    cp -f "${src}" "${dest}"
  else
    curl -fsSL --retry 3 --retry-delay 1 -o "${dest}" "${url}"
  fi
}

atomic_flip_current() {
  local provider_root="$1"
  local target_version="$2"
  local tmp_link="${provider_root}/.current.tmp.$$"
  rm -f "${tmp_link}"
  ln -s "versions/${target_version}" "${tmp_link}"
  python3 - "${tmp_link}" "${provider_root}/current" <<'PY'
import os, sys
os.replace(sys.argv[1], sys.argv[2])
PY
}

validate_provider_dir() {
  local bundle_dir="$1"
  local expected_provider_id="$2"
  python3 - "${bundle_dir}" "${expected_provider_id}" <<'PY'
import json
import os
import sys
from pathlib import Path

bundle = Path(sys.argv[1]).resolve()
expected_id = sys.argv[2].strip()
manifest_path = bundle / "provider.json"
if not manifest_path.is_file():
    raise SystemExit(f"Missing provider.json in {bundle}")

data = json.loads(manifest_path.read_text(encoding="utf-8"))
if not isinstance(data, dict):
    raise SystemExit(f"Invalid provider.json (not a JSON object): {manifest_path}")

pid = str(data.get("provider_id") or "").strip()
if not pid:
    raise SystemExit(f"provider.json missing provider_id: {manifest_path}")
if expected_id and pid != expected_id:
    raise SystemExit(
        f"provider.json provider_id mismatch: expected '{expected_id}', found '{pid}'"
    )

proto = data.get("protocol_version")
supported = data.get("supported_protocols") or []
if proto != 2 and 2 not in supported:
    raise SystemExit(
        f"provider.json for '{pid}' must declare SPI protocol_version=2 (got {proto!r})"
    )

entry_rel = None
for key in ("entrypoint", "binary", "daemon_path"):
    val = data.get(key)
    if isinstance(val, str) and val.strip():
        entry_rel = val.strip()
        break
if not entry_rel:
    raise SystemExit(f"provider.json for '{pid}' missing entrypoint/binary")

entry_path = Path(entry_rel)
if not entry_path.is_absolute():
    entry_path = (bundle / entry_path).resolve()
else:
    entry_path = entry_path.resolve()

try:
    entry_path.relative_to(bundle)
except ValueError as exc:
    raise SystemExit(f"entrypoint escapes bundle directory: {entry_path}") from exc

if not entry_path.is_file() or not os.access(entry_path, os.X_OK):
    raise SystemExit(f"entrypoint is missing or not executable: {entry_path}")

engine_id = str(data.get("engine_id") or pid)
version = str(data.get("display_version") or data.get("version") or data.get("semver") or "unknown")
print(f"OK provider_id={pid} engine_id={engine_id} version={version} protocol={proto} entrypoint={entry_rel}")
PY
}

list_providers() {
  python3 - "${REGISTRY_FILE}" "${PROVIDERS_DIR}" <<'PY'
import json
import sys
from pathlib import Path

registry = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
providers_dir = Path(sys.argv[2]).expanduser()
providers = registry.get("providers") or []

print(f"Microkernel Registry: {sys.argv[1]}")
print(f"Providers Directory : {providers_dir}")
print("")
header = f"{'PROVIDER_ID':<28} {'ENGINE_ID':<14} {'PATH':<5} {'PROTO':<5} {'TIER':<16} {'INSTALLED':<14} {'MODEL_TYPES'}"
print(header)
print("-" * len(header))
for item in providers:
    pid = str(item.get("provider_id") or "")
    eid = str(item.get("engine_id") or pid)
    path = str(item.get("path") or "B")
    proto = str(item.get("protocol_version") or "")
    tier = str(item.get("certification_tier") or "uncertified")
    mtypes = ",".join(item.get("supported_model_types") or ()) or "-"

    cur_link = providers_dir / pid / "current"
    installed = "-"
    if cur_link.is_symlink() or cur_link.is_dir():
        cur_manifest = cur_link / "provider.json"
        if cur_manifest.is_file():
            try:
                m = json.loads(cur_manifest.read_text(encoding="utf-8"))
                installed = str(m.get("display_version") or m.get("version") or m.get("semver") or cur_link.resolve().name)
            except Exception:
                installed = cur_link.resolve().name
        else:
            installed = cur_link.resolve().name
    print(f"{pid:<28} {eid:<14} {path:<5} {proto:<5} {tier:<16} {installed:<14} {mtypes}")
PY
}

verify_installed_providers() {
  if [[ ! -d "${PROVIDERS_DIR}" ]]; then
    echo "No providers installed at ${PROVIDERS_DIR}."
    return 0
  fi
  local found=0
  local failed=0
  for pdir in "${PROVIDERS_DIR}"/*; do
    [[ -d "${pdir}" ]] || continue
    local pid
    pid="$(basename "${pdir}")"
    local cur="${pdir}/current"
    if [[ ! -d "${cur}" ]]; then
      echo "WARN: ${pid} has no active current/ directory (skipping)" >&2
      continue
    fi
    found=$((found + 1))
    if ! validate_provider_dir "${cur}" "${pid}"; then
      failed=$((failed + 1))
    fi
  done
  if [[ "${found}" -eq 0 ]]; then
    echo "No active provider bundles found under ${PROVIDERS_DIR}."
  fi
  if [[ "${failed}" -gt 0 ]]; then
    echo "ERROR: ${failed} installed provider(s) failed SPI v2 verification." >&2
    return 1
  fi
  echo "==> Verified ${found} installed provider(s) cleanly."
}

sync_one_provider() {
  local query="$1"
  local meta_lines
  meta_lines="$(python3 - "${REGISTRY_FILE}" "${query}" <<'PY'
import json
import re
import sys
from pathlib import Path

SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")

registry = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
query = sys.argv[2].strip()
for item in registry.get("providers") or []:
    if query in (item.get("provider_id"), item.get("engine_id")):
        pid = str(item.get("provider_id") or "").strip()
        if not pid or ".." in pid or not SAFE_SEGMENT_RE.fullmatch(pid):
            raise SystemExit(f"Refusing unsafe provider_id from registry: {pid!r}")
        print(pid)
        print(item.get("engine_id") or pid)
        print(item["latest_manifest_url"])
        sys.exit(0)
raise SystemExit(f"Provider '{query}' not found in {sys.argv[1]}")
PY
)"
  local provider_id engine_id latest_url
  provider_id="$(printf '%s\n' "${meta_lines}" | sed -n '1p')"
  engine_id="$(printf '%s\n' "${meta_lines}" | sed -n '2p')"
  latest_url="$(printf '%s\n' "${meta_lines}" | sed -n '3p')"

  STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/omlx-mk-sync.XXXXXX")"
  local latest_file="${STAGE_DIR}/latest.json"
  echo "==> Fetching release manifest for ${provider_id} (${engine_id}) from ${latest_url}..."
  fetch_url "${latest_url}" "${latest_file}"

  local release_lines
  release_lines="$(python3 - "${latest_file}" "${latest_url}" "${provider_id}" <<'PY'
import json
import re
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import urljoin

SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]*$")


def _validate_safe_segment(raw: str, label: str) -> str:
    if (
        not raw
        or raw in (".", "..")
        or ".." in raw
        or "/" in raw
        or "\\" in raw
        or "\x00" in raw
        or not SAFE_SEGMENT_RE.fullmatch(raw)
    ):
        raise SystemExit(f"Refusing unsafe {label} in latest.json: {raw!r}")
    return raw


data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
latest_url = sys.argv[2]
expected_pid = sys.argv[3]

pid = str(data.get("provider_id") or expected_pid).strip()
if pid != expected_pid:
    raise SystemExit(f"latest.json provider_id mismatch: expected '{expected_pid}', got '{pid}'")

proto = data.get("protocol_version")
if proto is not None and proto != 2:
    raise SystemExit(f"latest.json protocol_version={proto!r} is not SPI v2")

version = str(data.get("display_version") or data.get("version") or data.get("semver") or "").strip()
if not version:
    raise SystemExit("latest.json missing version/display_version")
version = _validate_safe_segment(version, "version")

tar_url = ""
tar_sha = ""
tar_name = ""
assets = data.get("assets")
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
            raise SystemExit(
                f"Refusing unsafe filename in latest.json: {raw_filename!r}"
            )
        tar_name = _validate_safe_segment(raw_filename, "filename")

if not tar_url:
    tarball_rel = str(data.get("tarball") or "").strip()
    if tarball_rel:
        tar_name = Path(tarball_rel).name
        tar_url = urljoin(latest_url, tarball_rel)
if not tar_sha:
    tar_sha = str(data.get("sha256") or data.get("tarball_sha256") or "").strip().lower()
if not tar_name and tar_url:
    tar_name = Path(tar_url.split("?", 1)[0]).name
if not tar_name:
    tar_name = f"{expected_pid}-{version}.tar.gz"
tar_name = _validate_safe_segment(Path(tar_name).name, "filename")

if not tar_url or not tar_sha:
    raise SystemExit("latest.json must provide macos-arm64 tarball url and sha256")

print(version)
print(tar_url)
print(tar_sha)
print(tar_name)
PY
)"

  local version tar_url expected_sha tar_name
  version="$(printf '%s\n' "${release_lines}" | sed -n '1p')"
  tar_url="$(printf '%s\n' "${release_lines}" | sed -n '2p')"
  expected_sha="$(printf '%s\n' "${release_lines}" | sed -n '3p')"
  tar_name="$(printf '%s\n' "${release_lines}" | sed -n '4p')"

  local tarball_path="${STAGE_DIR}/${tar_name}"
  echo "==> Downloading ${tar_name} (${version})..."
  fetch_url "${tar_url}" "${tarball_path}"

  local actual_sha
  actual_sha="$(python3 - "${tarball_path}" <<'PY'
import hashlib, sys
from pathlib import Path
h = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as fh:
    while chunk := fh.read(1 << 20):
        h.update(chunk)
print(h.hexdigest())
PY
)"
  if [[ "${actual_sha}" != "${expected_sha}" ]]; then
    echo "ERROR: SHA-256 mismatch for ${tar_name}!" >&2
    echo "  expected: ${expected_sha}" >&2
    echo "  actual  : ${actual_sha}" >&2
    return 1
  fi
  echo "==> Verified SHA-256: ${actual_sha}"

  local unpack_root="${STAGE_DIR}/unpacked"
  mkdir -p "${unpack_root}"
  local bundle_root
  bundle_root="$(python3 - "${tarball_path}" "${unpack_root}" <<'PY'
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

tar_path = Path(sys.argv[1])
dest = Path(sys.argv[2]).resolve()


def _require_inside_dest(candidate: Path, label: str) -> None:
    try:
        candidate.relative_to(dest)
    except ValueError as exc:
        raise SystemExit(f"Tar entry escapes staging dir ({label}): {candidate}") from exc


def _validate_member_static(m: tarfile.TarInfo) -> Path:
    raw = m.name
    if not raw or "\x00" in raw or "\\" in raw:
        raise SystemExit(f"Refusing unsafe tar path: {raw!r}")
    pp = PurePosixPath(raw)
    p = Path(raw)
    if pp.is_absolute() or p.is_absolute() or ".." in pp.parts or ".." in p.parts:
        raise SystemExit(f"Refusing unsafe tar path: {raw!r}")
    if not (m.isreg() or m.isdir() or m.issym() or m.islnk()):
        raise SystemExit(f"Refusing unsupported tar special entry: {raw!r} (type={m.type!r})")

    # Strip setuid/setgid/sticky bits.
    m.mode = m.mode & 0o777

    # Lexical containment check before extraction.
    target = Path(os.path.normpath(str(dest / p)))
    _require_inside_dest(target, raw)

    if m.issym():
        link_raw = m.linkname
        if not link_raw or "\x00" in link_raw or "\\" in link_raw:
            raise SystemExit(f"Refusing unsafe tar symlink target for {raw!r}: {link_raw!r}")
        lpp = PurePosixPath(link_raw)
        lp = Path(link_raw)
        if lpp.is_absolute() or lp.is_absolute():
            raise SystemExit(f"Refusing absolute tar symlink target for {raw!r}: {link_raw!r}")
        resolved_link = Path(os.path.normpath(str(target.parent / lp)))
        _require_inside_dest(resolved_link, f"symlink {raw} -> {link_raw}")
    elif m.islnk():
        link_raw = m.linkname
        if not link_raw or "\x00" in link_raw or "\\" in link_raw:
            raise SystemExit(f"Refusing unsafe tar hardlink target for {raw!r}: {link_raw!r}")
        lpp = PurePosixPath(link_raw)
        lp = Path(link_raw)
        if lpp.is_absolute() or lp.is_absolute() or ".." in lpp.parts or ".." in lp.parts:
            raise SystemExit(f"Refusing unsafe tar hardlink target for {raw!r}: {link_raw!r}")
        resolved_link = Path(os.path.normpath(str(dest / lp)))
        _require_inside_dest(resolved_link, f"hardlink {raw} -> {link_raw}")

    return p


def _extract_members_safely(tf: tarfile.TarFile, members: list[tarfile.TarInfo]) -> None:
    """Safe member-by-member fallback for Python < 3.11.4 without filter='tar'."""
    for m in members:
        p = Path(m.name)
        rel_parts = [part for part in p.parts if part not in ("", ".")]
        if not rel_parts:
            continue
        # Ensure no intermediate ancestor on disk is a symlink (blocks symlink-chain traversal).
        cursor = dest
        for part in rel_parts[:-1]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise SystemExit(f"Refusing tar extraction through symlink directory: {m.name!r}")
        target = cursor / rel_parts[-1]
        if target.is_symlink():
            raise SystemExit(f"Refusing tar extraction over existing symlink: {m.name!r}")
        _require_inside_dest(target.parent.resolve(), m.name)
        if m.issym():
            live_link = (target.parent.resolve() / m.linkname).resolve()
            _require_inside_dest(live_link, f"symlink {m.name} -> {m.linkname}")
        elif m.islnk():
            live_link = (dest / m.linkname).resolve()
            _require_inside_dest(live_link, f"hardlink {m.name} -> {m.linkname}")
            if (dest / m.linkname).is_symlink():
                raise SystemExit(f"Refusing hardlink to symlink: {m.name!r} -> {m.linkname!r}")
        tf.extract(m, path=dest, set_attrs=True)


with tarfile.open(tar_path, "r:*") as tf:
    members = tf.getmembers()
    for m in members:
        _validate_member_static(m)
    try:
        tf.extractall(dest, members=members, filter="tar")
    except TypeError:
        _extract_members_safely(tf, members)

if (dest / "provider.json").is_file():
    print(str(dest))
else:
    subdirs = [p for p in sorted(dest.iterdir()) if p.is_dir() and (p / "provider.json").is_file()]
    if len(subdirs) == 1:
        print(str(subdirs[0]))
    else:
        raise SystemExit(f"Could not locate single provider.json bundle root inside {tar_path.name}")
PY
)"

  echo "==> Validating SPI v2 provider.json and entrypoint..."
  validate_provider_dir "${bundle_root}" "${provider_id}"

  if [[ "${VERIFY_ONLY}" -eq 1 ]]; then
    echo "==> --verify-only complete for ${provider_id} (${version}); Providers directory untouched."
    rm -rf "${STAGE_DIR}"
    STAGE_DIR=""
    return 0
  fi

  local provider_root="${PROVIDERS_DIR}/${provider_id}"
  local versions_dir="${provider_root}/versions"
  local dest_version_dir="${versions_dir}/${version}"
  local tmp_version_dir="${versions_dir}/.${version}.staging.$$"

  mkdir -p "${versions_dir}"
  rm -rf "${tmp_version_dir}"
  cp -R "${bundle_root}" "${tmp_version_dir}"

  # Record previous version (if different) for clean rollback.
  if [[ -L "${provider_root}/current" || -d "${provider_root}/current" ]]; then
    local prev_name
    prev_name="$(basename "$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${provider_root}/current")")"
    if [[ -n "${prev_name}" && "${prev_name}" != "${version}" && -d "${versions_dir}/${prev_name}" ]]; then
      printf '%s\n' "${prev_name}" > "${provider_root}/.previous_version"
    fi
  fi

  if [[ -d "${dest_version_dir}" ]]; then
    local backup_dir="${versions_dir}/.${version}.old.$$"
    mv "${dest_version_dir}" "${backup_dir}"
    mv "${tmp_version_dir}" "${dest_version_dir}"
    rm -rf "${backup_dir}"
  else
    mv "${tmp_version_dir}" "${dest_version_dir}"
  fi

  atomic_flip_current "${provider_root}" "${version}"
  echo "==> Activated ${provider_id} -> ${provider_root}/current -> versions/${version}"

  rm -rf "${STAGE_DIR}"
  STAGE_DIR=""
}

rollback_provider() {
  local query="$1"
  local provider_id="${query}"
  if [[ -f "${REGISTRY_FILE}" ]]; then
    provider_id="$(python3 - "${REGISTRY_FILE}" "${query}" <<'PY'
import json, sys
from pathlib import Path
reg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
q = sys.argv[2].strip()
for item in reg.get("providers") or []:
    if q in (item.get("provider_id"), item.get("engine_id")):
        print(item["provider_id"])
        sys.exit(0)
print(q)
PY
)"
  fi

  local provider_root="${PROVIDERS_DIR}/${provider_id}"
  local versions_dir="${provider_root}/versions"
  if [[ ! -d "${versions_dir}" ]]; then
    echo "ERROR: No installed versions found for '${provider_id}' in ${versions_dir}" >&2
    return 1
  fi

  local current_ver=""
  if [[ -L "${provider_root}/current" || -d "${provider_root}/current" ]]; then
    current_ver="$(basename "$(python3 -c 'import os, sys; print(os.path.realpath(sys.argv[1]))' "${provider_root}/current")")"
  fi

  local prev_ver=""
  if [[ -f "${provider_root}/.previous_version" ]]; then
    local cand
    cand="$(tr -d '[:space:]' < "${provider_root}/.previous_version")"
    if [[ -n "${cand}" && "${cand}" != "${current_ver}" && -d "${versions_dir}/${cand}" ]]; then
      prev_ver="${cand}"
    fi
  fi

  if [[ -z "${prev_ver}" ]]; then
    prev_ver="$(python3 - "${versions_dir}" "${current_ver}" <<'PY'
import sys
from pathlib import Path

vdir = Path(sys.argv[1])
cur = sys.argv[2]
cands = [
    p for p in vdir.iterdir()
    if p.is_dir() and not p.name.startswith(".") and p.name != cur and (p / "provider.json").is_file()
]
cands.sort(key=lambda p: p.stat().st_mtime_ns, reverse=True)
print(cands[0].name if cands else "")
PY
)"
  fi

  if [[ -z "${prev_ver}" ]]; then
    echo "ERROR: No previous version available to roll back '${provider_id}' (current=${current_ver:-none})" >&2
    return 1
  fi

  validate_provider_dir "${versions_dir}/${prev_ver}" "${provider_id}"
  if [[ -n "${current_ver}" && -d "${versions_dir}/${current_ver}" ]]; then
    printf '%s\n' "${current_ver}" > "${provider_root}/.previous_version"
  fi
  atomic_flip_current "${provider_root}" "${prev_ver}"
  echo "==> Rolled back ${provider_id}: ${current_ver:-none} -> ${prev_ver}"
}

case "${MODE}" in
  list)
    list_providers
    ;;
  verify_installed)
    verify_installed_providers
    ;;
  sync)
    sync_one_provider "${TARGET_PROVIDER}"
    ;;
  all)
    ALL_PROVIDERS="$(python3 - "${REGISTRY_FILE}" <<'PY'
import json, sys
from pathlib import Path
reg = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for item in reg.get("providers") or []:
    pid = str(item.get("provider_id") or "").strip()
    if pid:
        print(pid)
PY
)"
    if [[ -z "${ALL_PROVIDERS}" ]]; then
      echo "No providers registered in ${REGISTRY_FILE}."
      exit 0
    fi
    while IFS= read -r pid; do
      [[ -n "${pid}" ]] || continue
      sync_one_provider "${pid}"
    done <<< "${ALL_PROVIDERS}"
    ;;
  rollback)
    rollback_provider "${TARGET_PROVIDER}"
    ;;
  *)
    echo "ERROR: Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
