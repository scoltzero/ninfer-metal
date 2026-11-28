#!/usr/bin/env bash
set -euo pipefail

# Monorepo direct-sync tool for oMLX-extend:
#   1. Performs a true 3-way Git merge from `CURRENT_UPSTREAM_TAG` -> `TARGET_TAG`
#      using the live monorepo `omlx/` tree as "ours", so every oMLX-extend and
#      `-mix1..N` edit is automatically carried forward into the new upstream
#      release (falls back to static patch replay when bootstrapping or when
#      `--patch` is passed explicitly).
#   2. Resolves disjoint `Localizable.xcstrings` JSON key additions deterministically
#      via `merge_xcstrings.py`; real conflicts fail cleanly without modifying `omlx/`.
#   3. Syncs the merged tree into `omlx/` (preserving local ignored dirs like `.venv/`),
#      re-exports the binary-safe SPI patch `omlx-extend/patches/omlx-${TARGET_TAG}-provider-spi.patch`,
#      updates `omlx-extend/UPSTREAM_VERSION`, and runs the pytest regression suite.
#
# Also supports `--export-patch` to regenerate the binary-safe SPI patch from the
# monorepo's `omlx/` directory against the recorded upstream tag.
#
# Usage:
#   ./omlx-extend/scripts/sync_omlx.sh [--tag vX.Y.Z] [--omlx-dir PATH] [--upstream URL] [--patch FILE] [--skip-tests] [--allow-dirty] [--keep-staging]
#   ./omlx-extend/scripts/sync_omlx.sh --export-patch [--tag vX.Y.Z] [--omlx-dir PATH] [--upstream URL] [--patch FILE]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXTEND_DIR}/.." && pwd)"
UPSTREAM_VERSION_FILE="${EXTEND_DIR}/UPSTREAM_VERSION"

DEFAULT_UPSTREAM_REPO="https://github.com/jundot/omlx"
CURRENT_UPSTREAM_TAG=""
if [[ -f "${UPSTREAM_VERSION_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${UPSTREAM_VERSION_FILE}"
  DEFAULT_UPSTREAM_REPO="${UPSTREAM_REPO:-${DEFAULT_UPSTREAM_REPO}}"
  CURRENT_UPSTREAM_TAG="${UPSTREAM_TAG:-}"
fi

UPSTREAM_URL="${DEFAULT_UPSTREAM_REPO%.git}.git"
OMLX_DIR="${REPO_ROOT}/omlx"
PATCH_FILE=""
EXPLICIT_PATCH=0
TARGET_TAG=""
SKIP_TESTS=0
ALLOW_DIRTY=0
KEEP_STAGING=0
EXPORT_PATCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)
      TARGET_TAG="$2"
      shift 2
      ;;
    --base-tag)
      CURRENT_UPSTREAM_TAG="$2"
      shift 2
      ;;
    --omlx-dir)
      OMLX_DIR="$2"
      shift 2
      ;;
    --upstream)
      UPSTREAM_URL="$2"
      shift 2
      ;;
    --patch)
      PATCH_FILE="$2"
      EXPLICIT_PATCH=1
      shift 2
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --allow-dirty)
      ALLOW_DIRTY=1
      shift
      ;;
    --keep-staging)
      KEEP_STAGING=1
      shift
      ;;
    --export-patch)
      EXPORT_PATCH=1
      shift
      ;;
    -h|--help)
      sed -n '4,21p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

resolve_latest_upstream_tag() {
  git ls-remote --tags --refs "${UPSTREAM_URL}" 'v*' \
    | awk '{print $2}' \
    | sed 's|^refs/tags/||' \
    | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' \
    | awk -F'[v.]' '{printf "%08d.%08d.%08d %s\n", $2, $3, $4, $0}' \
    | sort \
    | tail -n 1 \
    | awk '{print $2}'
}

if [[ -z "${TARGET_TAG}" ]]; then
  if [[ "${EXPORT_PATCH}" -eq 1 && -n "${CURRENT_UPSTREAM_TAG}" ]]; then
    TARGET_TAG="${CURRENT_UPSTREAM_TAG}"
  else
    echo "==> Resolving latest upstream tag from ${UPSTREAM_URL}..."
    TARGET_TAG="$(resolve_latest_upstream_tag)"
    if [[ -z "${TARGET_TAG}" ]]; then
      echo "ERROR: Could not resolve latest upstream vX.Y.Z tag from ${UPSTREAM_URL}." >&2
      exit 1
    fi
  fi
fi

if [[ -z "${PATCH_FILE}" ]]; then
  PATCH_FILE="${EXTEND_DIR}/patches/omlx-${TARGET_TAG}-provider-spi.patch"
  if [[ "${EXPORT_PATCH}" -eq 0 && ! -f "${PATCH_FILE}" && -n "${CURRENT_UPSTREAM_TAG}" &&
        -f "${EXTEND_DIR}/patches/omlx-${CURRENT_UPSTREAM_TAG}-provider-spi.patch" ]]; then
    PATCH_FILE="${EXTEND_DIR}/patches/omlx-${CURRENT_UPSTREAM_TAG}-provider-spi.patch"
  fi
fi

RSYNC_EXCLUDES=(
  --exclude='/.git/'
  --exclude='/.venv/'
  --exclude='/venv/'
  --exclude='/build/'
  --exclude='/dist/'
  --exclude='/.pytest_cache/'
  --exclude='/.mypy_cache/'
  --exclude='/.ruff_cache/'
  --exclude='/.mlx_cache/'
  --exclude='/*.egg-info/'
  --exclude='/packaging/_build/'
  --exclude='/packaging/_export/'
  --exclude='/packaging/_wheels/'
  --exclude='/apps/omlx-mac/.build/'
  --exclude='/apps/omlx-mac/build/'
  --exclude='/apps/omlx-mac/.swiftpm/'
  --exclude='/apps/omlx-mac/DerivedData/'
  --exclude='/apps/omlx-mac/oMLX.xcodeproj/project.xcworkspace/xcshareddata/'
  --exclude='/node_modules/'
  --exclude='__pycache__/'
  --exclude='*.pyc'
  --exclude='*.pyo'
  --exclude='.DS_Store'
  --exclude='*.dylib'
  --exclude='*.so'
  --exclude='*.metallib'
  --exclude='*.dmg'
  --exclude='*.pkg'
  --exclude='*.db'
  --exclude='*.sqlite'
  --exclude='*.sqlite3'
  --exclude='*.log'
)

STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/omlx-sync.XXXXXX")"
cleanup() {
  if [[ "${KEEP_STAGING}" -eq 0 && -n "${STAGE_DIR}" && -d "${STAGE_DIR}" ]]; then
    rm -rf "${STAGE_DIR}"
  elif [[ -n "${STAGE_DIR}" && -d "${STAGE_DIR}" ]]; then
    echo "==> Staging checkout preserved at: ${STAGE_DIR}"
  fi
}
trap cleanup EXIT

if [[ "${EXPORT_PATCH}" -eq 1 ]]; then
  if [[ ! -d "${OMLX_DIR}" ]]; then
    echo "ERROR: Monorepo omlx directory not found: ${OMLX_DIR}" >&2
    exit 1
  fi
  echo "==> Exporting binary-safe SPI patch from ${OMLX_DIR} against upstream ${TARGET_TAG}..."
  git clone --quiet --depth 1 --branch "${TARGET_TAG}" "${UPSTREAM_URL}" "${STAGE_DIR}"
  rsync -ac --delete "${RSYNC_EXCLUDES[@]}" "${OMLX_DIR}/" "${STAGE_DIR}/"
  git -C "${STAGE_DIR}" add -A
  mkdir -p "$(dirname "${PATCH_FILE}")"
  git -C "${STAGE_DIR}" diff --cached --binary HEAD > "${PATCH_FILE}"
  echo "==> Exported patch to ${PATCH_FILE}:"
  git -C "${STAGE_DIR}" diff --cached --stat HEAD
  exit 0
fi

# Guard against silently overwriting uncommitted monorepo edits in omlx/.
if [[ "${ALLOW_DIRTY}" -eq 0 && -d "${REPO_ROOT}/.git" ]]; then
  OMLX_ABS="$(cd "$(dirname "${OMLX_DIR}")" && pwd)/$(basename "${OMLX_DIR}")"
  if [[ "${OMLX_ABS}" == "${REPO_ROOT}/"* ]]; then
    REL_OMLX="${OMLX_ABS#"${REPO_ROOT}/"}"
    if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain -- "${REL_OMLX}")" ]]; then
      echo "ERROR: Monorepo path '${REL_OMLX}' has uncommitted changes. Commit, stash, or re-run with --allow-dirty." >&2
      exit 1
    fi
  fi
fi

EXTEND_TAG="omlx-extend-${TARGET_TAG}"
echo "==> Monorepo direct-sync: upstream ${TARGET_TAG} -> ${OMLX_DIR} (target release: ${EXTEND_TAG})"

USE_THREE_WAY=0
if [[ "${EXPLICIT_PATCH}" -eq 0 && -d "${OMLX_DIR}" && -n "${CURRENT_UPSTREAM_TAG}" && "${CURRENT_UPSTREAM_TAG}" != "${TARGET_TAG}" ]]; then
  USE_THREE_WAY=1
fi

THEIRS_REF=""
if [[ "${USE_THREE_WAY}" -eq 1 ]]; then
  echo "==> 3-way merge mode: base=${CURRENT_UPSTREAM_TAG}, ours=live ${OMLX_DIR}, theirs=${TARGET_TAG}"
  git init -q "${STAGE_DIR}"
  git -C "${STAGE_DIR}" config user.name "oMLX-extend Sync"
  git -C "${STAGE_DIR}" config user.email "sync@omlx-extend.local"
  git -C "${STAGE_DIR}" remote add origin "${UPSTREAM_URL}"

  echo "==> Fetching upstream tags ${CURRENT_UPSTREAM_TAG} and ${TARGET_TAG} (depth=1)..."
  git -C "${STAGE_DIR}" fetch --quiet --depth=1 --no-tags origin \
    "refs/tags/${CURRENT_UPSTREAM_TAG}:refs/tags/${CURRENT_UPSTREAM_TAG}" \
    "refs/tags/${TARGET_TAG}:refs/tags/${TARGET_TAG}"

  UPSTREAM_COMMIT="$(git -C "${STAGE_DIR}" rev-parse "refs/tags/${TARGET_TAG}^{commit}")"
  BASE_TREE="$(git -C "${STAGE_DIR}" rev-parse "refs/tags/${CURRENT_UPSTREAM_TAG}^{tree}")"
  THEIRS_TREE="$(git -C "${STAGE_DIR}" rev-parse "refs/tags/${TARGET_TAG}^{tree}")"

  BASE_COMMIT="$(git -C "${STAGE_DIR}" commit-tree "${BASE_TREE}" -m "base: upstream ${CURRENT_UPSTREAM_TAG}")"
  THEIRS_COMMIT="$(git -C "${STAGE_DIR}" commit-tree "${THEIRS_TREE}" -p "${BASE_COMMIT}" -m "theirs: upstream ${TARGET_TAG}")"
  THEIRS_REF="${THEIRS_COMMIT}"

  git -C "${STAGE_DIR}" checkout -q -b omlx-extend-live "${BASE_COMMIT}"
  rsync -ac --delete "${RSYNC_EXCLUDES[@]}" "${OMLX_DIR}/" "${STAGE_DIR}/"
  git -C "${STAGE_DIR}" add -A
  if ! git -C "${STAGE_DIR}" diff --cached --quiet; then
    git -C "${STAGE_DIR}" commit -q -m "ours: live oMLX-extend (including mix updates) atop ${CURRENT_UPSTREAM_TAG}"
  fi

  echo "==> Running 3-way Git merge onto upstream ${TARGET_TAG}..."
  set +e
  git -C "${STAGE_DIR}" merge --no-edit "${THEIRS_COMMIT}"
  MERGE_RC=$?
  set -e

  if [[ "${MERGE_RC}" -ne 0 ]]; then
    XCSTRINGS_REL="apps/omlx-mac/Resources/Localizable.xcstrings"
    if git -C "${STAGE_DIR}" diff --name-only --diff-filter=U | grep -qx "${XCSTRINGS_REL}"; then
      echo "==> Attempting deterministic 3-way JSON merge for ${XCSTRINGS_REL}..."
      PYTHON_BIN="${OMLX_DIR}/.venv/bin/python"
      if [[ ! -x "${PYTHON_BIN}" ]]; then
        PYTHON_BIN="$(command -v python3 || command -v python)"
      fi
      XC_TMP="${STAGE_DIR}/.xcstrings-merge-tmp"
      mkdir -p "${XC_TMP}"
      if git -C "${STAGE_DIR}" show ":1:${XCSTRINGS_REL}" > "${XC_TMP}/base.json" 2>/dev/null \
        && git -C "${STAGE_DIR}" show ":2:${XCSTRINGS_REL}" > "${XC_TMP}/ours.json" 2>/dev/null \
        && git -C "${STAGE_DIR}" show ":3:${XCSTRINGS_REL}" > "${XC_TMP}/theirs.json" 2>/dev/null \
        && "${PYTHON_BIN}" "${SCRIPT_DIR}/merge_xcstrings.py" \
          "${XC_TMP}/base.json" "${XC_TMP}/ours.json" "${XC_TMP}/theirs.json" \
          "${STAGE_DIR}/${XCSTRINGS_REL}"; then
        git -C "${STAGE_DIR}" add "${XCSTRINGS_REL}"
        echo "    Resolved ${XCSTRINGS_REL} cleanly via 3-way JSON merge."
      else
        echo "    Could not auto-resolve ${XCSTRINGS_REL}; leaving conflict for manual resolution." >&2
      fi
      rm -rf "${XC_TMP}"
    fi

    REMAINING_CONFLICTS="$(git -C "${STAGE_DIR}" diff --name-only --diff-filter=U)"
    if [[ -n "${REMAINING_CONFLICTS}" ]]; then
      echo "ERROR: 3-way merge conflict against upstream ${TARGET_TAG} in:" >&2
      echo "${REMAINING_CONFLICTS}" >&2
      exit 3
    fi

    git -C "${STAGE_DIR}" commit -q --no-edit
  fi
  echo "==> 3-way merge onto ${TARGET_TAG} succeeded with all oMLX-extend / mix edits preserved."
else
  echo "==> Patch-replay mode: cloning ${UPSTREAM_URL}@${TARGET_TAG} into staging dir ${STAGE_DIR}..."
  git clone --quiet --depth 1 --branch "${TARGET_TAG}" "${UPSTREAM_URL}" "${STAGE_DIR}"
  UPSTREAM_COMMIT="$(git -C "${STAGE_DIR}" rev-parse 'HEAD^{commit}')"
  THEIRS_REF="${UPSTREAM_COMMIT}"

  echo "==> Applying Provider SPI patch (${PATCH_FILE}) onto staging checkout..."
  "${SCRIPT_DIR}/apply_spi_patch.sh" \
    --omlx-dir "${STAGE_DIR}" \
    --patch "${PATCH_FILE}"
fi

echo "==> Syncing merged upstream tree into monorepo directory ${OMLX_DIR}..."
mkdir -p "${OMLX_DIR}"
rsync -ac --delete "${RSYNC_EXCLUDES[@]}" "${STAGE_DIR}/" "${OMLX_DIR}/"

# Double-check upstream .git was never copied into monorepo omlx/.
if [[ -e "${OMLX_DIR}/.git" ]]; then
  echo "ERROR: Unexpected ${OMLX_DIR}/.git present after rsync." >&2
  exit 1
fi

# Only update monorepo UPSTREAM_VERSION and patches/ when syncing the monorepo's own omlx/ dir.
OMLX_REAL="$(cd "${OMLX_DIR}" && pwd)"
REPO_OMLX_REAL="$(cd "${REPO_ROOT}" && pwd)/omlx"
if [[ "${OMLX_REAL}" == "${REPO_OMLX_REAL}" ]]; then
  UPSTREAM_REPO_CLEAN="${UPSTREAM_URL%.git}"
  cat > "${UPSTREAM_VERSION_FILE}" <<EOF
UPSTREAM_REPO=${UPSTREAM_REPO_CLEAN}
UPSTREAM_TAG=${TARGET_TAG}
UPSTREAM_COMMIT=${UPSTREAM_COMMIT}
EXTEND_BRANCH=tags/${TARGET_TAG}
EOF
  echo "==> Updated ${UPSTREAM_VERSION_FILE} (tag=${TARGET_TAG}, commit=${UPSTREAM_COMMIT})"

  NEW_PATCH_FILE="${EXTEND_DIR}/patches/omlx-${TARGET_TAG}-provider-spi.patch"
  mkdir -p "${EXTEND_DIR}/patches"
  if [[ "${USE_THREE_WAY}" -eq 1 ]]; then
    git -C "${STAGE_DIR}" diff --binary "${THEIRS_REF}..HEAD" > "${NEW_PATCH_FILE}"
  else
    git -C "${STAGE_DIR}" add -A
    git -C "${STAGE_DIR}" diff --cached --binary "${THEIRS_REF}" > "${NEW_PATCH_FILE}"
  fi
  echo "==> Refreshed binary-safe SPI patch: ${NEW_PATCH_FILE}"

  if [[ -n "${CURRENT_UPSTREAM_TAG}" && "${CURRENT_UPSTREAM_TAG}" != "${TARGET_TAG}" ]]; then
    OLD_PATCH_FILE="${EXTEND_DIR}/patches/omlx-${CURRENT_UPSTREAM_TAG}-provider-spi.patch"
    if [[ -f "${OLD_PATCH_FILE}" && "${OLD_PATCH_FILE}" != "${NEW_PATCH_FILE}" ]]; then
      rm -f "${OLD_PATCH_FILE}"
      echo "==> Removed superseded patch: ${OLD_PATCH_FILE}"
    fi
  fi
else
  echo "==> Custom --omlx-dir (${OMLX_REAL}) used; leaving ${UPSTREAM_VERSION_FILE} and patches/ unchanged."
fi

if [[ "${SKIP_TESTS}" -eq 0 ]]; then
  PYTEST_BIN="${OMLX_DIR}/.venv/bin/pytest"
  if [[ ! -x "${PYTEST_BIN}" ]]; then
    PYTEST_BIN="$(command -v pytest || true)"
  fi
  if [[ -z "${PYTEST_BIN}" ]]; then
    echo "ERROR: pytest not found in ${OMLX_DIR}/.venv/bin/pytest or PATH" >&2
    exit 1
  fi
  TEST_TARGETS=(tests/test_m5_sidecar_engine.py tests/test_splash_engine.py)
  if [[ -f "${OMLX_DIR}/tests/test_microkernel_manager.py" ]]; then
    TEST_TARGETS+=(tests/test_microkernel_manager.py)
  fi
  if [[ -f "${OMLX_DIR}/tests/test_process_memory_enforcer.py" ]]; then
    TEST_TARGETS+=(tests/test_process_memory_enforcer.py)
  fi
  echo "==> Running oMLX-extend regression suite in ${OMLX_DIR}..."
  (cd "${OMLX_DIR}" && "${PYTEST_BIN}" "${TEST_TARGETS[@]}" -q)
fi

echo "==> Monorepo direct-sync complete (${TARGET_TAG} / ${UPSTREAM_COMMIT}; ready for ${EXTEND_TAG} release)."
