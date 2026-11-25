#!/bin/zsh
set -euo pipefail

script_dir="${0:A:h}"
repo_root="${script_dir:h}"
[[ "${repo_root:t}" == "miv" ]] && repo_root="${repo_root:h}"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-${repo_root}/omlx/.venv/bin/python}"
build_dir="${M5_H1_BUILD_DIR:-${repo_root}/build/miv-v001}"
proxy_url="${M5_HF_PROXY:-http://127.0.0.1:6152}"

export http_proxy="${proxy_url}"
export https_proxy="${proxy_url}"
export all_proxy="${M5_HF_SOCKS_PROXY:-socks5://127.0.0.1:6153}"

eval "$("${python_bin}" "${script_dir}/m5_silicon_profile.py" --shell-env)"
mkdir -p "${M5_H0_RESULT_DIR}" "${M5_H1_RESULT_DIR}" "${M5_H2_RESULT_DIR}" .cache/b0/artifacts .cache/b5/real

wait_for_batch1_download() {
  echo "[$(date -u +%FT%TZ)] Waiting for Batch 1 (Qwen3.8-27B-mxfp4) network download to finish..."
  while pgrep -f "huggingface.co/mlx-community/Qwen3.8-27B-mxfp4" >/dev/null 2>&1; do
    sleep 15
  done
  echo "[$(date -u +%FT%TZ)] Batch 1 network download finished! Proceeding to Batch 2 download."
}

wait_for_gpu_idle() {
  while pgrep -f "run_stage_a_27b_pipeline.sh|m5_ultra_full_model.py|m5_ultra_tuning.py|omlx_m5_benchmark_" >/dev/null 2>&1; do
    echo "[$(date -u +%FT%TZ)] Waiting for active GPU benchmark to finish before starting next GPU suite..."
    sleep 15
  done
}

parallel_fetch() {
  local repo="$1"
  local rev="$2"
  local out_dir="$3"
  shift 3
  mkdir -p "${out_dir}"
  local pids=()
  for f in "$@"; do
    if [[ -f "${out_dir}/${f}" ]]; then
      echo "[$(date -u +%FT%TZ)] Already present: ${out_dir}/${f}"
      continue
    fi
    (
      # Do NOT use curl's internal --retry with --continue-at -: internal retries
      # truncate --output back to 0 bytes on transient proxy disconnects. Let the
      # outer loop re-invoke curl --continue-at - so every reconnect resumes cleanly.
      until curl -x "${proxy_url}" --http1.1 --fail --location \
        --connect-timeout 20 --speed-limit 10240 --speed-time 30 \
        --continue-at - --silent --show-error \
        --output "${out_dir}/${f}.part" \
        "https://huggingface.co/${repo}/resolve/${rev}/${f}"; do
        sleep 2
      done
      echo "[$(date -u +%FT%TZ)] Downloaded ${repo}/${f}"
    ) &
    pids+=($!)
  done
  for p in "${pids[@]}"; do
    wait "$p"
  done
}

# ==============================================================================
# BATCH 2: Jundot/Qwen3.8-27B-oQ4e-mtp (latest HF rev 836759ee)
# ==============================================================================
wait_for_batch1_download

B2_REPO="Jundot/Qwen3.8-27B-oQ4e-mtp"
B2_REV="836759eefbef7ef588a5ffd44cc59a640808f944"
B2_DIR="${repo_root}/models/Jundot/Qwen3.8-27B-oQ4e-mtp-hf"

echo "[$(date -u +%FT%TZ)] === BATCH 2 DOWNLOAD START: ${B2_REPO}@${B2_REV} ==="
parallel_fetch "${B2_REPO}" "${B2_REV}" "${B2_DIR}" \
  config.json \
  model.safetensors.index.json \
  model-00001-of-00004.safetensors \
  model-00002-of-00004.safetensors \
  model-00003-of-00004.safetensors \
  model-00004-of-00004.safetensors

echo "[$(date -u +%FT%TZ)] Verifying Batch 2 SHA-256 via download_b0_models.py..."
# Move downloaded files to standard models/Jundot/Qwen3.8-27B-oQ4e-mtp path (keeping tokenizer files)
B2_STD="${repo_root}/models/Jundot/Qwen3.8-27B-oQ4e-mtp"
mkdir -p "${B2_STD}"
for f in config.json model.safetensors.index.json model-00001-of-00004.safetensors model-00002-of-00004.safetensors model-00003-of-00004.safetensors model-00004-of-00004.safetensors; do
  if [[ -f "${B2_DIR}/${f}.part" ]]; then
    mv -f "${B2_DIR}/${f}.part" "${B2_STD}/${f}.part"
  fi
done
rm -f "${B2_STD}/config.json" "${B2_STD}/model.safetensors.index.json"
"${python_bin}" "${script_dir}/download_b0_models.py" --model q38_27_oq_latest

# Convert Batch 2 to .omlxm5 (CPU-only, can run while Batch 3 downloads!)
B2_ART="${repo_root}/.cache/b0/artifacts/qwen38-27b-oq4e.omlxm5"
echo "[$(date -u +%FT%TZ)] Converting Batch 2 -> ${B2_ART}"
"${python_bin}" "${script_dir}/m5_artifact.py" convert \
  --source "${B2_STD}" \
  --output "${B2_ART}" \
  --source-repo "${B2_REPO}" \
  --expected-revision "${B2_REV}" \
  --profile oq_affine

# ==============================================================================
# START BATCH 3 DOWNLOAD IN BACKGROUND WHILE BATCH 2 BENCHMARKS RUN!
# ==============================================================================
B3_REPO="OsaurusAI/Qwen3.6-35B-A3B-mxfp4"
B3_REV="5a6aaf3d0db3590f186d743b99c9b9d270ec7857"
B3_DIR="${repo_root}/models/OsaurusAI/Qwen3.6-35B-A3B-mxfp4"

echo "[$(date -u +%FT%TZ)] === BATCH 3 DOWNLOAD START (parallel): ${B3_REPO}@${B3_REV} ==="
(
  parallel_fetch "${B3_REPO}" "${B3_REV}" "${B3_DIR}" \
    model.safetensors.index.json \
    model-00001-of-00005.safetensors \
    model-00002-of-00005.safetensors \
    model-00003-of-00005.safetensors \
    model-00004-of-00005.safetensors \
    model-00005-of-00005.safetensors
  "${python_bin}" "${script_dir}/download_b0_models.py" --model q36_35_mx
  echo "[$(date -u +%FT%TZ)] === BATCH 3 DOWNLOAD & SHA-256 VERIFIED ==="
) > "${HOME}/hf-batch3-35b.log" 2>&1 &
B3_DL_PID=$!

# ==============================================================================
# RUN BATCH 2 (Qwen3.8-27B-oQ4e + DFlash2) H1-L / H1-G / H2 ON M5 ULTRA
# ==============================================================================
wait_for_gpu_idle
echo "[$(date -u +%FT%TZ)] === RUNNING BATCH 2 GPU BENCHMARKS (Qwen3.8-27B-oQ4e + DFlash2) ==="

M5_H1_SKIP_BUILD=1 M5_H1_BUILD_DIR="${build_dir}" \
  "${script_dir}/run_m5_ultra_tuning.sh" --stages h1_l,h1_g --iterations 30

M5_H2_SKIP_BUILD=1 M5_FULL_MODEL_BUILD_DIR="${build_dir}" \
M5_FULL_MODEL_RESULT="${M5_H2_RESULT_DIR}/27b-oq4e-b1-32k.json" \
"${script_dir}/run_m5_ultra_full_model.sh" \
  --preset qwen38-27b-oq4e \
  --mode both \
  --sampling-contract position_coupled_v1 \
  --prefix-cache-bytes 536870912 \
  --attention-split-kv on \
  --max-context 32768 \
  --max-new-tokens 256 \
  --warmups 3 --repeats 10

M5_H2_SKIP_BUILD=1 M5_FULL_MODEL_BUILD_DIR="${build_dir}" \
M5_FULL_MODEL_RESULT="${M5_H2_RESULT_DIR}/27b-oq4e-b4-32k-batch.json" \
"${script_dir}/run_m5_ultra_full_model.sh" \
  --preset qwen38-27b-oq4e \
  --mode both \
  --max-concurrency 4 \
  --batch-mode on \
  --sampling-contract position_coupled_v1 \
  --prefix-cache-bytes 536870912 \
  --attention-split-kv on \
  --max-context 32768 \
  --max-new-tokens 128 \
  --warmups 1 --repeats 3

echo "[$(date -u +%FT%TZ)] === BATCH 2 GPU BENCHMARKS COMPLETE! Waiting for Batch 3 download... ==="
wait "${B3_DL_PID}"

# ==============================================================================
# CONVERT & RUN BATCH 3 (Qwen3.6-35B-A3B-mxfp4 MoE: H1-M + H1-L + H1-G + H2)
# ==============================================================================
B3_ART="${repo_root}/.cache/b0/artifacts/qwen36-35b-a3b-mxfp4.omlxm5"
echo "[$(date -u +%FT%TZ)] Converting Batch 3 -> ${B3_ART}"
"${python_bin}" "${script_dir}/m5_artifact.py" convert \
  --source "${B3_DIR}" \
  --output "${B3_ART}" \
  --source-repo "${B3_REPO}" \
  --expected-revision "${B3_REV}" \
  --profile mlx_mxfp4

echo "[$(date -u +%FT%TZ)] === RUNNING BATCH 3 GPU BENCHMARKS (35B-A3B MoE: H1-M / H1-L / H1-G / H2) ==="
M5_H0_SKIP_BUILD=1 M5_H0_BUILD_DIR="${build_dir}" \
  "${script_dir}/run_m5_ultra_capability.sh" --iterations 50

M5_H1_SKIP_BUILD=1 M5_H1_BUILD_DIR="${build_dir}" \
  "${script_dir}/run_m5_ultra_tuning.sh" --stages h1_l,h1_g,h1_m --iterations 30

M5_H2_SKIP_BUILD=1 M5_FULL_MODEL_BUILD_DIR="${build_dir}" \
M5_FULL_MODEL_RESULT="${M5_H2_RESULT_DIR}/35b-mxfp4-b1-32k.json" \
"${script_dir}/run_m5_ultra_full_model.sh" \
  --preset qwen36-35b-a3b-mxfp4 \
  --mode ar \
  --sampling-contract position_coupled_v1 \
  --prefix-cache-bytes 536870912 \
  --attention-split-kv on \
  --max-context 32768 \
  --max-new-tokens 256 \
  --warmups 3 --repeats 10

M5_H2_SKIP_BUILD=1 M5_FULL_MODEL_BUILD_DIR="${build_dir}" \
M5_FULL_MODEL_RESULT="${M5_H2_RESULT_DIR}/35b-mxfp4-b8-64k-batch.json" \
"${script_dir}/run_m5_ultra_full_model.sh" \
  --preset qwen36-35b-a3b-mxfp4 \
  --mode ar \
  --max-concurrency 8 \
  --batch-mode on \
  --sampling-contract position_coupled_v1 \
  --prefix-cache-bytes 536870912 \
  --attention-split-kv on \
  --max-context 65536 \
  --max-new-tokens 128 \
  --warmups 1 --repeats 3

"${python_bin}" - <<PY
import json, datetime, pathlib
out = pathlib.Path("${repo_root}/bench/results/${M5_RESULT_NAMESPACE}/STAGE_A_ALL_BATCHES_DONE.json")
out.write_text(json.dumps({
    "status": "completed",
    "batches": ["batch1_27b_mxfp4_dflash2", "batch2_27b_oq4e_dflash2", "batch3_35b_a3b_mxfp4_moe"],
    "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "profile_id": "${M5_SILICON_PROFILE_ID}",
    "result_namespace": "${M5_RESULT_NAMESPACE}",
}, indent=2) + "\n")
print("Wrote", out)
PY

echo "[$(date -u +%FT%TZ)] === ALL 3 BATCHES (27B MXFP4 + 27B oQ4e + 35B MoE) COMPLETED! ==="
