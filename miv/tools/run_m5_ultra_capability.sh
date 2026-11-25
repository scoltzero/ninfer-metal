#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
[[ "${repo_root:t}" == "miv" ]] && repo_root="${repo_root:h}"
python_bin="${PYTHON_BIN:-${repo_root}/omlx/.venv/bin/python}"
build_dir="${M5_H0_BUILD_DIR:-${repo_root}/build/miv-v001}"
output_dir="${M5_H0_RESULT_DIR:-${repo_root}/bench/results/h0}"
developer_dir="${OMLX_M5_DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"

plan_only=false
allow_non_target=false
want_help=false
has_system_trace_json=false
has_attach_trace_evidence=false
iterations=30

args=("$@")
for (( idx = 1; idx <= ${#args[@]}; idx++ )); do
  argument="${args[$idx]}"
  if [[ "${argument}" == "--plan-only" ]]; then
    plan_only=true
  elif [[ "${argument}" == "--allow-non-target" ]]; then
    allow_non_target=true
  elif [[ "${argument}" == "-h" || "${argument}" == "--help" ]]; then
    want_help=true
  elif [[ "${argument}" == "--system-trace-json" || "${argument}" == --system-trace-json=* ]]; then
    has_system_trace_json=true
  elif [[ "${argument}" == "--attach-trace-evidence" || "${argument}" == --attach-trace-evidence=* ]]; then
    has_attach_trace_evidence=true
  elif [[ "${argument}" == "--iterations" && $((idx + 1)) -le ${#args[@]} ]]; then
    iterations="${args[$((idx + 1))]}"
  elif [[ "${argument}" == --iterations=* ]]; then
    iterations="${argument#--iterations=}"
  fi
done

if [[ "${plan_only}" == false && "${want_help}" == false ]]; then
  if [[ "${allow_non_target}" == false ]]; then
    "${python_bin}" "${script_dir}/m5_target_guard.py" >/dev/null
  fi

  if [[ "${M5_H0_SKIP_BUILD:-0}" != "1" ]]; then
    if command -v cmake >/dev/null 2>&1 && DEVELOPER_DIR="${developer_dir}" xcrun -f metal >/dev/null 2>&1; then
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
        omlx_m5_probe \
        omlx_m5_nax_qmm_probe \
        omlx_m5_hw_counter_probe \
        omlx_m5_inspect_runtime \
        omlx_m5_inspect_icb_canary \
        omlx_m5_p11_layout_inventory \
        omlx_m5_tf_affine_lane_tests
      cmake --build "${build_dir}" --target omlx_m5_p11_tensor_canary || true
    elif [[ -z "${M5_H0_BUILD_DIR:-}" && ! -x "${build_dir}/miv/omlx_m5_probe" && -x "${repo_root}/build/miv-v001/miv/omlx_m5_probe" ]]; then
      build_dir="${repo_root}/build/miv-v001"
    fi
  elif [[ -z "${M5_H0_BUILD_DIR:-}" && ! -x "${build_dir}/miv/omlx_m5_probe" && -x "${repo_root}/build/miv-v001/miv/omlx_m5_probe" ]]; then
    build_dir="${repo_root}/build/miv-v001"
  fi
fi

extra_cap_args=()
if [[ "${plan_only}" == false && "${want_help}" == false && "${M5_H0_SKIP_HW_COUNTERS:-0}" != "1" ]]; then
  hw_probe_bin="${build_dir}/miv/omlx_m5_hw_counter_probe"
  [[ -x "${hw_probe_bin}" ]] || hw_probe_bin="${build_dir}/m5rt/omlx_m5_hw_counter_probe"
  if [[ -x "${hw_probe_bin}" ]]; then
    mkdir -p "${output_dir}"
    hw_counters_json="${output_dir}/hw-counters-h0.json"
    hw_trace_summary_json="${output_dir}/hw-trace-summary.json"
    "${python_bin}" "${script_dir}/m5_hw_trace_stats.py" \
      --mode h0 \
      --iterations "${iterations}" \
      --xctrace-mode "${M5_H0_XCTRACE_MODE:-auto}" \
      --developer-dir "${developer_dir}" \
      --hw-counter-probe "${hw_probe_bin}" \
      --hw-counters-output "${hw_counters_json}" \
      --output "${hw_trace_summary_json}"
    if [[ "${has_system_trace_json}" == false && -f "${hw_trace_summary_json}" ]]; then
      extra_cap_args+=(--system-trace-json "${hw_trace_summary_json}")
    fi
    if [[ "${has_attach_trace_evidence}" == false && -f "${hw_counters_json}" ]]; then
      extra_cap_args+=(--attach-trace-evidence "${hw_counters_json}")
    fi
  fi
fi

exec "${python_bin}" "${script_dir}/m5_ultra_capability.py" \
  --build-dir "${build_dir}" \
  --output-dir "${output_dir}" \
  "${extra_cap_args[@]}" \
  "$@"
