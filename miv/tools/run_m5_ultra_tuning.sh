#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
[[ "${repo_root:t}" == "miv" ]] && repo_root="${repo_root:h}"
python_bin="${PYTHON_BIN:-${repo_root}/omlx/.venv/bin/python}"
build_dir="${M5_H1_BUILD_DIR:-${repo_root}/build/miv-v001}"
output_dir="${M5_H1_RESULT_DIR:-${repo_root}/bench/results/h1}"
developer_dir="${OMLX_M5_DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

plan_only=false
allow_non_target=false
want_help=false
has_build_dir_arg=false
for argument in "$@"; do
  if [[ "${argument}" == "--plan-only" ]]; then
    plan_only=true
  elif [[ "${argument}" == "--allow-non-target" ]]; then
    allow_non_target=true
  elif [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    want_help=true
  elif [[ "${argument}" == "--build-dir" || "${argument}" == --build-dir=* ]]; then
    has_build_dir_arg=true
  fi
done

if [[ "${plan_only}" == false && "${want_help}" == false && "${M5_H1_SKIP_BUILD:-0}" != "1" ]]; then
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
  cmake --build "${build_dir}" --target \
    omlx_m5_benchmark_projection_candidates \
    omlx_m5_benchmark_long_kv_attention \
    omlx_m5_benchmark_gdn_prefill \
    omlx_m5_benchmark_p10_batch \
    omlx_m5_export_p12_routes
fi

extra_args=()
if [[ "${has_build_dir_arg}" == false ]]; then
  extra_args+=(--build-dir "${build_dir}")
fi

exec "${python_bin}" "${script_dir}/m5_ultra_tuning.py" \
  "${extra_args[@]}" \
  --output-dir "${output_dir}" \
  "$@"
