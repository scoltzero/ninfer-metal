#!/usr/bin/env bash
set -euo pipefail

# Build a complete, drag-to-install macOS `.dmg` for oMLX-extend (`oMLX.app`):
#   1. Obtains the verified Python 3.11 + MLX/NAX venvstacks layers and compiled
#      stock custom kernels from upstream `jundot/omlx`'s official `macos26-27`
#      DMG (or falls back to a full `venvstacks` + `--with-custom-kernel` source
#      build if upstream has not published a DMG for that tag).
#   2. Compiles the native SwiftUI `oMLX.app` (Release) from `omlx/apps/omlx-mac`,
#      including the Microkernel Manager UI, `IntegrationNinferMetal` assets, and
#      Tahoe `AppIcon.icns` (preserving oMLX's own brand icon).
#   3. Embeds the full `oMLX-extend` `Resources/omlx` + `Resources/omlx_web`
#      packages alongside all 5 stock MLX/NAX custom kernels, stamps
#      `CFBundleShortVersionString` (e.g. `0.8.3` or `0.8.3-mix1`), ad-hoc signs
#      the bundle coherently, runs `codesign --verify --deep --strict` + live
#      embedded-Python smoke checks, and packs an `hdiutil` UDZO `.dmg` + `.sha256`.
#
# Usage:
#   ./omlx-extend/scripts/build_omlx_extend_dmg.sh \
#     [--extend-tag omlx-extend-v0.8.3] \
#     [--upstream-tag v0.8.3] \
#     [--donor-dmg PATH] \
#     [--omlx-dir PATH] \
#     [--out-dir PATH]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXTEND_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${EXTEND_DIR}/.." && pwd)"
UPSTREAM_VERSION_FILE="${EXTEND_DIR}/UPSTREAM_VERSION"

UPSTREAM_TAG=""
if [[ -f "${UPSTREAM_VERSION_FILE}" ]]; then
  UPSTREAM_TAG="$(
    # shellcheck disable=SC1090
    source "${UPSTREAM_VERSION_FILE}"
    printf '%s' "${UPSTREAM_TAG:-}"
  )"
fi

EXTEND_TAG=""
DONOR_DMG=""
OMLX_DIR="${REPO_ROOT}/omlx"
OUT_DIR="${REPO_ROOT}/dist/omlx-extend"
UPSTREAM_SLUG="jundot/omlx"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --extend-tag)
      EXTEND_TAG="$2"
      shift 2
      ;;
    --upstream-tag)
      UPSTREAM_TAG="$2"
      shift 2
      ;;
    --donor-dmg)
      DONOR_DMG="$2"
      shift 2
      ;;
    --omlx-dir)
      OMLX_DIR="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    -h|--help)
      sed -n '4,24p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${UPSTREAM_TAG}" ]]; then
  echo "ERROR: --upstream-tag not set and UPSTREAM_VERSION missing." >&2
  exit 1
fi
if [[ -z "${EXTEND_TAG}" ]]; then
  EXTEND_TAG="omlx-extend-${UPSTREAM_TAG}"
fi

# e.g. omlx-extend-v0.8.3 -> v0.8.3 ; omlx-extend-v0.8.3-mix1 -> v0.8.3-mix1
VERSION_SUFFIX="${EXTEND_TAG#omlx-extend-}"
APP_MARKETING_VERSION="${VERSION_SUFFIX#v}"
DMG_NAME="oMLX-extend-${VERSION_SUFFIX}-macos26-27.dmg"

mkdir -p "${OUT_DIR}"
OUT_DIR="$(cd "${OUT_DIR}" && pwd)"
OMLX_DIR="$(cd "${OMLX_DIR}" && pwd)"
OUT_DMG="${OUT_DIR}/${DMG_NAME}"

WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/omlx-extend-dmg.XXXXXX")"
DONOR_MNT=""
cleanup() {
  if [[ -n "${DONOR_MNT}" && -d "${DONOR_MNT}" ]]; then
    hdiutil detach "${DONOR_MNT}" -force >/dev/null 2>&1 || true
    rmdir "${DONOR_MNT}" 2>/dev/null || true
  fi
  if [[ -n "${WORK_DIR}" && -d "${WORK_DIR}" ]]; then
    rm -rf "${WORK_DIR}"
  fi
}
trap cleanup EXIT

echo "==> Building ${DMG_NAME} (marketing version: ${APP_MARKETING_VERSION}, upstream: ${UPSTREAM_TAG})"

# 1) Resolve upstream donor DMG unless already supplied.
if [[ -z "${DONOR_DMG}" ]]; then
  CACHED_DONOR="${REPO_ROOT}/dist/donor-${UPSTREAM_TAG}/oMLX-${UPSTREAM_TAG#v}-macos26-27.dmg"
  if [[ -f "${CACHED_DONOR}" ]]; then
    DONOR_DMG="${CACHED_DONOR}"
    echo "==> Using cached upstream donor DMG: ${DONOR_DMG}"
  elif command -v gh >/dev/null 2>&1; then
    DONOR_DL_DIR="${WORK_DIR}/donor-dl"
    mkdir -p "${DONOR_DL_DIR}"
    echo "==> Downloading official upstream donor DMG from ${UPSTREAM_SLUG}@${UPSTREAM_TAG}..."
    if gh release download "${UPSTREAM_TAG}" --repo "${UPSTREAM_SLUG}" --pattern "oMLX-*-macos26*.dmg" --dir "${DONOR_DL_DIR}"; then
      DONOR_DMG="$(find "${DONOR_DL_DIR}" -maxdepth 1 -name "*.dmg" | head -n 1)"
    fi
  fi
fi

STAGE_OUT="${WORK_DIR}/Stage"
mkdir -p "${STAGE_OUT}"

if [[ -n "${DONOR_DMG}" && -f "${DONOR_DMG}" ]]; then
  DONOR_MNT="$(mktemp -d "${TMPDIR:-/tmp}/omlx-donor-mnt.XXXXXX")"
  echo "==> Mounting upstream donor DMG: ${DONOR_DMG}"
  hdiutil attach -nobrowse -readonly -mountpoint "${DONOR_MNT}" "${DONOR_DMG}" >/dev/null
  DONOR_APP="${DONOR_MNT}/oMLX.app"
  if [[ ! -d "${DONOR_APP}" ]]; then
    echo "ERROR: oMLX.app not found inside donor DMG ${DONOR_DMG}" >&2
    exit 1
  fi

  echo "==> Building native Swift Release oMLX.app + overlaying oMLX-extend packages..."
  OMLX_DONOR_APP="${DONOR_APP}" \
  OMLX_NEXT_OUT="${STAGE_OUT}" \
  "${OMLX_DIR}/apps/omlx-mac/Scripts/build.sh" release

  STAGED_APP="${STAGE_OUT}/oMLX.app"

  # Copy prebuilt stock MLX/NAX custom kernels (.so, .dylib, .metallib) from donor
  # since build.sh excludes custom_kernels binaries when --with-custom-kernel is off.
  DONOR_KERNELS="${DONOR_APP}/Contents/Resources/omlx/custom_kernels"
  STAGED_KERNELS="${STAGED_APP}/Contents/Resources/omlx/custom_kernels"
  if [[ -d "${DONOR_KERNELS}" ]]; then
    echo "==> Copying prebuilt stock MLX/NAX custom kernels from donor..."
    for kdir in "${DONOR_KERNELS}"/*/; do
      [[ -d "${kdir}" ]] || continue
      kname="$(basename "${kdir}")"
      mkdir -p "${STAGED_KERNELS}/${kname}"
      find "${kdir}" -maxdepth 1 -type f \( -name "*.so" -o -name "*.dylib" -o -name "*.metallib" \) \
        -exec cp -p {} "${STAGED_KERNELS}/${kname}/" \;
    done
  fi
else
  echo "==> No upstream donor DMG found; building full venvstacks + custom kernels from source..."
  OMLX_NEXT_OUT="${STAGE_OUT}" \
  "${OMLX_DIR}/apps/omlx-mac/Scripts/build.sh" release --rebuild-donor --with-custom-kernel
  STAGED_APP="${STAGE_OUT}/oMLX.app"
fi

if [[ ! -d "${STAGED_APP}" ]]; then
  echo "ERROR: Staged oMLX.app missing at ${STAGED_APP}" >&2
  exit 1
fi

# 2) Stamp exact oMLX-extend marketing version (e.g. 0.8.3 or 0.8.3-mix1) & re-sign coherently.
INFO_PLIST="${STAGED_APP}/Contents/Info.plist"
/usr/libexec/PlistBuddy -c "Set :CFBundleShortVersionString ${APP_MARKETING_VERSION}" "${INFO_PLIST}"

echo "==> Signing custom kernel extensions and sealing oMLX.app bundle..."
if [[ -d "${STAGED_APP}/Contents/Resources/omlx/custom_kernels" ]]; then
  while IFS= read -r -d '' bin_file; do
    codesign --force --sign - "${bin_file}" >/dev/null 2>&1
  done < <(
    find "${STAGED_APP}/Contents/Resources/omlx/custom_kernels" \
      -type f \( -name "*.so" -o -name "*.dylib" \) -print0
  )
fi
codesign --force --sign - "${STAGED_APP}"
xattr -dr com.apple.quarantine "${STAGED_APP}" 2>/dev/null || true

echo "==> Verifying strict bundle signature (identifier=app.omlx)..."
codesign --verify --deep --strict -R '=identifier "app.omlx"' "${STAGED_APP}"

# 3) Run live smoke verification using the app bundle's own embedded Python!
echo "==> Running embedded-Python & custom-kernel smoke checks inside staged oMLX.app..."
"${STAGED_APP}/Contents/MacOS/omlx-cluster-python" - <<'PY'
import omlx
import omlx.engine.m5_sidecar as m5_sidecar
import omlx.engine.microkernel_manager as mk_mgr
import omlx_web
from omlx.custom_kernels import native_kernel_status

status = native_kernel_status()
missing = {k: v.get("import_error") for k, v in status.items() if not v.get("available")}
if missing:
    raise SystemExit(f"Bundled custom_kernels failed to load: {missing}")

assert hasattr(mk_mgr, "list_catalog"), "microkernel_manager missing expected API"
assert hasattr(m5_sidecar, "M5SidecarEngine"), "m5_sidecar missing M5SidecarEngine"
print(f"Smoke OK: omlx={omlx.__version__}, custom_kernels={list(status.keys())}")
PY

# 4) Assemble drag-to-Applications DMG folder and pack compressed UDZO DMG.
DMG_ROOT="${WORK_DIR}/dmg-root"
mkdir -p "${DMG_ROOT}"
ditto "${STAGED_APP}" "${DMG_ROOT}/oMLX.app"
ln -s /Applications "${DMG_ROOT}/Applications"

echo "==> Packing compressed UDZO disk image: ${OUT_DMG}"
rm -f "${OUT_DMG}" "${OUT_DMG}.sha256"
hdiutil create \
  -volname "oMLX-extend ${APP_MARKETING_VERSION}" \
  -srcfolder "${DMG_ROOT}" \
  -ov \
  -format UDZO \
  -imagekey zlib-level=6 \
  "${OUT_DMG}"

(cd "${OUT_DIR}" && shasum -a 256 "${DMG_NAME}" > "${DMG_NAME}.sha256")
ls -lh "${OUT_DMG}" "${OUT_DMG}.sha256"
echo "==> DMG ready: ${OUT_DMG}"
