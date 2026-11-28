#!/usr/bin/env bash
set -euo pipefail

# Apply the thin oMLX-extend Provider SPI patch cleanly onto an upstream oMLX checkout.
# Usage:
#   ./omlx-extend/scripts/apply_spi_patch.sh [--omlx-dir PATH] [--patch FILE] [--check-only]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXTEND_DIR}/.." && pwd)"

OMLX_DIR="${REPO_ROOT}/omlx"
PATCH_FILE=""
CHECK_ONLY=0
RUN_TESTS=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --omlx-dir)
      OMLX_DIR="$2"
      shift 2
      ;;
    --patch)
      PATCH_FILE="$2"
      shift 2
      ;;
    --check-only)
      CHECK_ONLY=1
      shift
      ;;
    --test)
      RUN_TESTS=1
      shift
      ;;
    -h|--help)
      sed -n '3,6p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${PATCH_FILE}" ]]; then
  UPSTREAM_VERSION_FILE="${EXTEND_DIR}/UPSTREAM_VERSION"
  UPSTREAM_TAG=""
  if [[ -f "${UPSTREAM_VERSION_FILE}" ]]; then
    # shellcheck disable=SC1090
    source "${UPSTREAM_VERSION_FILE}"
  fi
  if [[ -z "${UPSTREAM_TAG}" ]]; then
    echo "ERROR: UPSTREAM_TAG missing from ${UPSTREAM_VERSION_FILE}; pass --patch FILE." >&2
    exit 1
  fi
  PATCH_FILE="${EXTEND_DIR}/patches/omlx-${UPSTREAM_TAG}-provider-spi.patch"
fi

if [[ ! -d "${OMLX_DIR}" ]]; then
  echo "ERROR: oMLX directory not found: ${OMLX_DIR}" >&2
  exit 1
fi
if [[ ! -f "${PATCH_FILE}" ]]; then
  echo "ERROR: SPI patch file not found: ${PATCH_FILE}" >&2
  exit 1
fi

echo "==> Checking patch applicability: ${PATCH_FILE} -> ${OMLX_DIR}"
if git -C "${OMLX_DIR}" apply --check --3way "${PATCH_FILE}" 2>/dev/null; then
  echo "    Patch applies cleanly (3-way check passed)."
elif git -C "${OMLX_DIR}" apply --check "${PATCH_FILE}"; then
  echo "    Patch applies cleanly."
else
  echo "ERROR: Patch conflict detected against $(git -C "${OMLX_DIR}" describe --tags --always 2>/dev/null || echo 'target tree')." >&2
  exit 3
fi

if [[ "${CHECK_ONLY}" -eq 1 ]]; then
  exit 0
fi

echo "==> Applying Provider SPI patch..."
git -C "${OMLX_DIR}" apply --3way "${PATCH_FILE}" 2>/dev/null || git -C "${OMLX_DIR}" apply "${PATCH_FILE}"
echo "==> Patch applied successfully."

if [[ "${RUN_TESTS}" -eq 1 ]]; then
  PYTEST_BIN="${OMLX_DIR}/.venv/bin/pytest"
  if [[ ! -x "${PYTEST_BIN}" ]]; then
    PYTEST_BIN="$(command -v pytest || true)"
  fi
  if [[ -z "${PYTEST_BIN}" ]]; then
    echo "ERROR: pytest not found in ${OMLX_DIR}/.venv/bin/pytest or PATH" >&2
    exit 1
  fi
  echo "==> Running oMLX-extend sidecar & Splash regression tests..."
  (cd "${OMLX_DIR}" && "${PYTEST_BIN}" tests/test_m5_sidecar_engine.py tests/test_splash_engine.py -q)
fi
