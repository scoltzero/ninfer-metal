#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
[[ "${repo_root:t}" == "miv" ]] && repo_root="${repo_root:h}"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-${repo_root}/omlx/.venv/bin/python}"
build_dir="${M5_H1_BUILD_DIR:-${repo_root}/build/miv-v001}"
bin_dir="${build_dir}/miv"
[[ -d "${bin_dir}" ]] || bin_dir="${build_dir}/m5rt"
lib_dir="${bin_dir}/metallib"

eval "$("${python_bin}" "${script_dir}/m5_silicon_profile.py" --shell-env)"
mkdir -p "${M5_H0_RESULT_DIR}" "${M5_H1_RESULT_DIR}/h1_l" "${M5_H1_RESULT_DIR}/h1_g" "${M5_H2_RESULT_DIR}" .cache/b0/artifacts .cache/b5/real

mxfp4_art="${repo_root}/.cache/b0/artifacts/qwen38-27b-mlx-mxfp4.omlxm5"
mxfp4_cfg="${repo_root}/models/mlx-community/Qwen3.8-27B-mxfp4/config.json"
dflash_art="${repo_root}/.cache/b5/real/qwen38-27b-dflash2.omlxm5"
dflash_cfg="${repo_root}/models/z-lab/Qwen3.8-27B-DFlash2/config.json"

if [[ ! -f "${mxfp4_art}" || ! -f "${dflash_art}" ]]; then
  echo "ERROR: Required artifacts missing: ${mxfp4_art} / ${dflash_art}" >&2
  exit 1
fi

echo "[$(date -u +%FT%TZ)] === STEP 1: H0 Capability Refresh (with real 27B MXFP4 P11 inventory) ==="
M5_H0_SKIP_BUILD=1 M5_H0_BUILD_DIR="${build_dir}" \
  "${script_dir}/run_m5_ultra_capability.sh" --iterations 50

echo "[$(date -u +%FT%TZ)] === STEP 2: H1-L Full-N Projection Candidates on Qwen3.8-27B-MXFP4 ==="
"${bin_dir}/omlx_m5_benchmark_projection_candidates" \
  "${mxfp4_art}" language_model.lm_head "${lib_dir}" 1 30 \
  > "${M5_H1_RESULT_DIR}/h1_l/qwen38-27b-mxfp4-lm_head-t1.json"

"${bin_dir}/omlx_m5_benchmark_projection_candidates" \
  "${mxfp4_art}" language_model.lm_head "${lib_dir}" 8 30 matrix verify \
  > "${M5_H1_RESULT_DIR}/h1_l/qwen38-27b-mxfp4-lm_head-t8-matrix.json"

"${bin_dir}/omlx_m5_benchmark_projection_candidates" \
  "${mxfp4_art}" language_model.model.layers.0.mlp.down_proj "${lib_dir}" 1 30 \
  > "${M5_H1_RESULT_DIR}/h1_l/qwen38-27b-mxfp4-down_proj-t1.json"

"${bin_dir}/omlx_m5_benchmark_projection_candidates" \
  "${mxfp4_art}" language_model.model.layers.0.mlp.down_proj "${lib_dir}" 8 30 matrix \
  > "${M5_H1_RESULT_DIR}/h1_l/qwen38-27b-mxfp4-down_proj-t8-matrix.json"

echo "[$(date -u +%FT%TZ)] === STEP 3: H1-G Real 27B MXFP4 GDN Prefill (--p07) ==="
"${bin_dir}/omlx_m5_benchmark_gdn_prefill" \
  "${mxfp4_art}" "${mxfp4_cfg}" "${lib_dir}" 15 \
  "${M5_H1_RESULT_DIR}/h1_g/qwen38-27b-mxfp4-gdn.json" --p07

echo "[$(date -u +%FT%TZ)] === STEP 4: Real 27B DFlash2 Companion Slice Benchmark ==="
"${bin_dir}/omlx_m5_benchmark_dflash_slice" \
  "${dflash_art}" "${dflash_cfg}" "${mxfp4_art}" "${lib_dir}" 2048 3 15 7 on gpu \
  > "${M5_H1_RESULT_DIR}/qwen38-27b-mxfp4-dflash2-slice.json" || true

echo "[$(date -u +%FT%TZ)] === STEP 5: H2 FULL 64-LAYER Qwen3.8-27B-MXFP4 B=1 32K (AR + DFlash2, warmups=3, repeats=10) ==="
M5_H2_SKIP_BUILD=1 M5_FULL_MODEL_BUILD_DIR="${build_dir}" \
M5_FULL_MODEL_RESULT="${M5_H2_RESULT_DIR}/27b-mxfp4-b1-32k.json" \
"${script_dir}/run_m5_ultra_full_model.sh" \
  --preset qwen38-27b-mlx-mxfp4 \
  --mode both \
  --sampling-contract position_coupled_v1 \
  --prefix-cache-bytes 536870912 \
  --attention-split-kv on \
  --max-context 32768 \
  --max-new-tokens 256 \
  --warmups 3 --repeats 10

echo "[$(date -u +%FT%TZ)] === STEP 6: H2 FULL 64-LAYER Qwen3.8-27B-MXFP4 B=1 64K (AR + DFlash2, warmups=1, repeats=3) ==="
M5_H2_SKIP_BUILD=1 M5_FULL_MODEL_BUILD_DIR="${build_dir}" \
M5_FULL_MODEL_RESULT="${M5_H2_RESULT_DIR}/27b-mxfp4-b1-64k.json" \
"${script_dir}/run_m5_ultra_full_model.sh" \
  --preset qwen38-27b-mlx-mxfp4 \
  --mode both \
  --sampling-contract position_coupled_v1 \
  --prefix-cache-bytes 536870912 \
  --attention-split-kv on \
  --max-context 65536 \
  --max-new-tokens 256 \
  --warmups 1 --repeats 3

echo "[$(date -u +%FT%TZ)] === STEP 7: H2 FULL 64-LAYER Qwen3.8-27B-MXFP4 B=4 32K batch_mode=on (AR + DFlash2, warmups=1, repeats=3) ==="
M5_H2_SKIP_BUILD=1 M5_FULL_MODEL_BUILD_DIR="${build_dir}" \
M5_FULL_MODEL_RESULT="${M5_H2_RESULT_DIR}/27b-mxfp4-b4-32k-batch.json" \
"${script_dir}/run_m5_ultra_full_model.sh" \
  --preset qwen38-27b-mlx-mxfp4 \
  --mode both \
  --max-concurrency 4 \
  --batch-mode on \
  --sampling-contract position_coupled_v1 \
  --prefix-cache-bytes 536870912 \
  --attention-split-kv on \
  --max-context 32768 \
  --max-new-tokens 128 \
  --warmups 1 --repeats 3

"${python_bin}" - <<PY
import json, datetime, pathlib
out = pathlib.Path("${repo_root}/bench/results/${M5_RESULT_NAMESPACE}/STAGE_A_27B_DONE.json")
out.write_text(json.dumps({
    "status": "completed",
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "profile_id": "${M5_SILICON_PROFILE_ID}",
    "result_namespace": "${M5_RESULT_NAMESPACE}",
}, indent=2) + "\n")
print("Wrote", out)
PY

echo "[$(date -u +%FT%TZ)] === STAGE A BATCH 1 (Qwen3.8-27B-MXFP4 + DFlash2) COMPLETED! ==="
