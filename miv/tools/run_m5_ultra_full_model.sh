#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
[[ "${repo_root:t}" == "miv" ]] && repo_root="${repo_root:h}"
python_bin="${PYTHON_BIN:-${repo_root}/omlx/.venv/bin/python}"
build_dir="${M5_FULL_MODEL_BUILD_DIR:-${repo_root}/build/miv-v001}"
result_file="${M5_FULL_MODEL_RESULT:-${repo_root}/bench/results/h2/full-model.json}"
developer_dir="${OMLX_M5_DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

plan_only=false
allow_non_target=false
want_help=false
has_preset=false
has_model_dir=false
has_artifact=false
has_draft_dir=false
has_draft_artifact=false
for argument in "$@"; do
  if [[ "${argument}" == "--plan-only" ]]; then
    plan_only=true
  elif [[ "${argument}" == "--allow-non-target" ]]; then
    allow_non_target=true
  elif [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    want_help=true
  elif [[ "${argument}" == "--preset" || "${argument}" == --preset=* ]]; then
    has_preset=true
  elif [[ "${argument}" == "--model-dir" || "${argument}" == --model-dir=* ]]; then
    has_model_dir=true
  elif [[ "${argument}" == "--artifact" || "${argument}" == --artifact=* ]]; then
    has_artifact=true
  elif [[ "${argument}" == "--draft-dir" || "${argument}" == --draft-dir=* ]]; then
    has_draft_dir=true
  elif [[ "${argument}" == "--draft-artifact" || "${argument}" == --draft-artifact=* ]]; then
    has_draft_artifact=true
  fi
done

if [[ "${plan_only}" == false && "${want_help}" == false && "${M5_H2_SKIP_BUILD:-0}" != "1" ]]; then
  if [[ "${allow_non_target}" == false ]]; then
    "${python_bin}" "${script_dir}/m5_target_guard.py" >/dev/null
  fi
  xcode_clang="$(DEVELOPER_DIR="${developer_dir}" xcrun -f clang++)"
  cmake -S "${repo_root}" -B "${build_dir}" -G Ninja \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCMAKE_CXX_COMPILER="${xcode_clang}" \
    -DCMAKE_OBJCXX_COMPILER="${xcode_clang}" \
    -DPython_EXECUTABLE="${python_bin}" \
    -DOMLX_M5_BUILD_NAX_METALLIB=ON \
    -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF \
    -DOMLX_M5_BUILD_PYTHON=ON \
    -DOMLX_M5_BUILD_PROBES=ON \
    -DOMLX_M5_DEVELOPER_DIR="${developer_dir}"
  cmake --build "${build_dir}" --target omlx_m5_native
fi

miv_dir="${build_dir}/miv"
[[ -d "${miv_dir}" ]] || miv_dir="${build_dir}/m5rt"
default_args=(
  --extension-dir "${miv_dir}"
  --qwen-metallib "${M5_QWEN_METALLIB:-${miv_dir}/metallib/omlx_m5_qwen_ar.metallib}"
  --dflash-metallib "${M5_DFLASH_METALLIB:-${miv_dir}/metallib/omlx_m5_dflash2_program.metallib}"
  --output "${result_file}"
)

if [[ -n "${M5_MODEL_PRESET:-}" && "${has_preset}" == false ]]; then
  default_args+=(--preset "${M5_MODEL_PRESET}")
  has_preset=true
fi

if [[ "${has_preset}" == false ]]; then
  if [[ "${has_model_dir}" == false ]]; then
    default_args+=(--model-dir "${M5_TARGET_MODEL_DIR:-${repo_root}/models/Jundot/Qwen3.8-27B-oQ4e-mtp}")
  fi
  if [[ "${has_artifact}" == false ]]; then
    default_args+=(--artifact "${M5_TARGET_ARTIFACT:-${repo_root}/.cache/b0/artifacts/qwen38-27b-oq4e.omlxm5}")
  fi
  if [[ "${has_draft_dir}" == false ]]; then
    default_args+=(--draft-dir "${M5_DRAFT_MODEL_DIR:-${repo_root}/models/z-lab/Qwen3.8-27B-DFlash2}")
  fi
  if [[ "${has_draft_artifact}" == false ]]; then
    default_args+=(--draft-artifact "${M5_DRAFT_ARTIFACT:-${repo_root}/.cache/b5/real/qwen38-27b-dflash2.omlxm5}")
  fi
else
  if [[ -n "${M5_TARGET_MODEL_DIR:-}" && "${has_model_dir}" == false ]]; then
    default_args+=(--model-dir "${M5_TARGET_MODEL_DIR}")
  fi
  if [[ -n "${M5_TARGET_ARTIFACT:-}" && "${has_artifact}" == false ]]; then
    default_args+=(--artifact "${M5_TARGET_ARTIFACT}")
  fi
  if [[ -n "${M5_DRAFT_MODEL_DIR:-}" && "${has_draft_dir}" == false ]]; then
    default_args+=(--draft-dir "${M5_DRAFT_MODEL_DIR}")
  fi
  if [[ -n "${M5_DRAFT_ARTIFACT:-}" && "${has_draft_artifact}" == false ]]; then
    default_args+=(--draft-artifact "${M5_DRAFT_ARTIFACT}")
  fi
fi

exec "${python_bin}" "${script_dir}/m5_ultra_full_model.py" \
  "${default_args[@]}" \
  "$@"
