#!/usr/bin/env bash
set -euo pipefail

# 1-Click standalone launcher for `ninfer-metal-serve`.
#
# Usage:
#   ./omlx-extend/scripts/start_ninfer_serve.sh [MODEL_DIR] [extra ninfer-metal-serve args...]
#   ./omlx-extend/scripts/start_ninfer_serve.sh --dry-run [MODEL_DIR] [extra args...]
#
# Automatically discovers:
#   - `ninfer-metal-serve` binary (release bundle -> installed provider -> repo build -> $PATH)
#   - Shared `ninfer-metal.toml` ($NINFER_METAL_CONFIG -> cwd -> ~/.config -> bundle/repo default)
#   - First local model containing `model.miv` (or legacy `model.omlxm5`) when MODEL_DIR is omitted and TOML has no model_dir/[[models]]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PARENT_DIR}/.." 2>/dev/null && pwd || echo "${PARENT_DIR}")"
BUNDLED_PROVIDERS_DIR="${PARENT_DIR}/Providers"
PROVIDERS_DEFAULT_DIR="${HOME}/Library/Application Support/oMLX/Providers"

DRY_RUN=0
SHOW_HELP=0
NO_DEFAULT_CONFIG=0
CLI_MODEL_DIR=""
CLI_HAS_MODEL_DIR=0
CLI_CONFIG=""
CLI_HOST=""
CLI_PORT=""
CLI_PROVIDERS_DIR=""
CLI_LORA_ADAPTER=""
CLI_HAS_LORA_ADAPTER=0
CLI_LORA_MODE=""
CLI_HAS_LORA_MODE=0
PASSTHROUGH_ARGS=()

print_usage() {
  cat <<EOF
Usage: $0 [MODEL_DIR] [extra ninfer-metal-serve args...]
       $0 --dry-run [MODEL_DIR] [extra ninfer-metal-serve args...]

1-Click launcher for ninfer-metal-serve using the shared ninfer-metal.toml config.
Options handled by launcher:
  MODEL_DIR                        Optional positional model directory (sets --model-dir MODEL_DIR)
  --lora, --lora-adapter DIR       LoRA adapter directory (or set \$NINFER_LORA_ADAPTER)
  --lora-mode merge|bypass         LoRA execution mode: merge (default) | bypass (or set \$NINFER_LORA_MODE)
  --dry-run                        Print resolved binary, config, model, endpoint, and command without starting
  -h, --help                       Show this launcher usage message
All other flags (e.g. --port, --host, --config, --max-concurrency, --kv-dtype) pass through to ninfer-metal-serve.
EOF
}

validate_lora_mode() {
  local mode="$1"
  local src="$2"
  case "${mode}" in
    merge|bypass) ;;
    *)
      echo "ERROR: ${src} must be 'merge' or 'bypass' (got '${mode}')" >&2
      exit 2
      ;;
  esac
}

expand_tilde() {
  local p="$1"
  if [[ "$p" == "~" ]]; then
    printf '%s\n' "${HOME}"
  elif [[ "$p" == "~/"* ]]; then
    printf '%s/%s\n' "${HOME%/}" "${p:2}"
  else
    printf '%s\n' "$p"
  fi
}

# Parse arguments while preserving extra ninfer-metal-serve flags verbatim.
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      SHOW_HELP=1
      shift
      ;;
    --no-default-config)
      NO_DEFAULT_CONFIG=1
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --config)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --config requires a path" >&2
        exit 2
      fi
      CLI_CONFIG="$(expand_tilde "$2")"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --config=*)
      CLI_CONFIG="$(expand_tilde "${1#--config=}")"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --model-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --model-dir requires a path" >&2
        exit 2
      fi
      CLI_HAS_MODEL_DIR=1
      if [[ -z "${CLI_MODEL_DIR}" ]]; then
        CLI_MODEL_DIR="$(expand_tilde "$2")"
      fi
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --model-dir=*)
      CLI_HAS_MODEL_DIR=1
      if [[ -z "${CLI_MODEL_DIR}" ]]; then
        CLI_MODEL_DIR="$(expand_tilde "${1#--model-dir=}")"
      fi
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --host)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --host requires a value" >&2
        exit 2
      fi
      CLI_HOST="$2"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --host=*)
      CLI_HOST="${1#--host=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --port)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --port requires a value" >&2
        exit 2
      fi
      CLI_PORT="$2"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --port=*)
      CLI_PORT="${1#--port=}"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --providers-dir)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: --providers-dir requires a path" >&2
        exit 2
      fi
      CLI_PROVIDERS_DIR="$(expand_tilde "$2")"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --providers-dir=*)
      CLI_PROVIDERS_DIR="$(expand_tilde "${1#--providers-dir=}")"
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --lora|--lora-adapter)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a path" >&2
        exit 2
      fi
      CLI_HAS_LORA_ADAPTER=1
      CLI_LORA_ADAPTER="$(expand_tilde "$2")"
      PASSTHROUGH_ARGS+=("$1" "${CLI_LORA_ADAPTER}")
      shift 2
      ;;
    --lora=*|--lora-adapter=*)
      CLI_HAS_LORA_ADAPTER=1
      CLI_LORA_ADAPTER="$(expand_tilde "${1#*=}")"
      PASSTHROUGH_ARGS+=("${1%%=*}=${CLI_LORA_ADAPTER}")
      shift
      ;;
    --lora-mode)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires merge or bypass" >&2
        exit 2
      fi
      validate_lora_mode "$2" "--lora-mode"
      CLI_HAS_LORA_MODE=1
      CLI_LORA_MODE="$2"
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --lora-mode=*)
      CLI_LORA_MODE="${1#--lora-mode=}"
      validate_lora_mode "${CLI_LORA_MODE}" "--lora-mode"
      CLI_HAS_LORA_MODE=1
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    --cors)
      if [[ $# -ge 2 ]]; then
        case "$2" in
          on|off|true|false|1|0)
            PASSTHROUGH_ARGS+=("$1" "$2")
            shift 2
            ;;
          *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
        esac
      else
        PASSTHROUGH_ARGS+=("$1")
        shift
      fi
      ;;
    --provider|--artifact|--draft-dir|--draft-artifact|--mtp-artifact|\
    --metallib-dir|--kernel-dir|--chat-template|--model-id|--api-key|\
    --memory-budget-gb|--max-concurrency|-c|-b|--max-context|--kv-dtype|\
    --spec-backend|--draft-tokens|--rolling-prefix-cache|--prefix-cache-gb|\
    --sampling-contract)
      if [[ $# -lt 2 ]]; then
        echo "ERROR: $1 requires a value" >&2
        exit 2
      fi
      PASSTHROUGH_ARGS+=("$1" "$2")
      shift 2
      ;;
    --)
      shift
      while [[ $# -gt 0 ]]; do
        PASSTHROUGH_ARGS+=("$1")
        shift
      done
      break
      ;;
    -*)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
    *)
      if [[ "${CLI_HAS_MODEL_DIR}" -eq 0 && -z "${CLI_MODEL_DIR}" ]]; then
        CLI_MODEL_DIR="$(expand_tilde "$1")"
        CLI_HAS_MODEL_DIR=1
      else
        PASSTHROUGH_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ "${SHOW_HELP}" -eq 1 ]]; then
  print_usage
  exit 0
fi

if [[ "${CLI_HAS_LORA_MODE}" -eq 0 && -n "${NINFER_LORA_MODE:-}" ]]; then
  validate_lora_mode "${NINFER_LORA_MODE}" "NINFER_LORA_MODE"
fi

# 1. Auto-discover `ninfer-metal-serve` binary.
discover_serve_binary() {
  local cand
  if [[ -n "${NINFER_METAL_SERVE_BIN:-}" ]]; then
    cand="$(expand_tilde "${NINFER_METAL_SERVE_BIN}")"
    if [[ -x "${cand}" && -f "${cand}" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  fi

  local candidates=(
    "${PARENT_DIR}/bin/ninfer-metal-serve"
    "${BUNDLED_PROVIDERS_DIR}/io.ninfer.metal/current/bin/ninfer-metal-serve"
  )
  if [[ -n "${CLI_PROVIDERS_DIR}" ]]; then
    candidates+=("${CLI_PROVIDERS_DIR}/io.ninfer.metal/current/bin/ninfer-metal-serve")
  fi
  if [[ -n "${OMLX_PROVIDERS_DIR:-}" ]]; then
    candidates+=("$(expand_tilde "${OMLX_PROVIDERS_DIR}")/io.ninfer.metal/current/bin/ninfer-metal-serve")
  fi
  candidates+=(
    "${PROVIDERS_DEFAULT_DIR}/io.ninfer.metal/current/bin/ninfer-metal-serve"
    "${REPO_ROOT}/build/release/miv/ninfer-metal-serve"
    "./build/release/miv/ninfer-metal-serve"
    "${REPO_ROOT}/build/miv-v001/miv/ninfer-metal-serve"
    "./build/miv-v001/miv/ninfer-metal-serve"
  )
  for cand in "${candidates[@]}"; do
    if [[ -x "${cand}" && -f "${cand}" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  done

  if cand="$(command -v ninfer-metal-serve 2>/dev/null)"; then
    if [[ -x "${cand}" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  fi
  return 1
}

SERVE_BIN=""
if ! SERVE_BIN="$(discover_serve_binary)"; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    SERVE_BIN="ninfer-metal-serve (not built/installed yet)"
  else
    echo "ERROR: ninfer-metal-serve binary not found. Build build/miv-v001/miv/ninfer-metal-serve or install io.ninfer.metal provider." >&2
    exit 1
  fi
fi

# 2. Auto-discover shared `ninfer-metal.toml`.
discover_config_file() {
  local cand
  if [[ -n "${NINFER_METAL_CONFIG:-}" ]]; then
    cand="$(expand_tilde "${NINFER_METAL_CONFIG}")"
    if [[ -f "${cand}" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  fi

  local candidates=(
    "./ninfer-metal.toml"
    "${HOME}/.config/ninfer-metal/ninfer-metal.toml"
  )
  if [[ -n "${CLI_PROVIDERS_DIR}" ]]; then
    candidates+=("${CLI_PROVIDERS_DIR}/io.ninfer.metal/current/config/ninfer-metal.toml")
  fi
  if [[ -n "${OMLX_PROVIDERS_DIR:-}" ]]; then
    candidates+=("$(expand_tilde "${OMLX_PROVIDERS_DIR}")/io.ninfer.metal/current/config/ninfer-metal.toml")
  fi
  candidates+=(
    "${PROVIDERS_DEFAULT_DIR}/io.ninfer.metal/current/config/ninfer-metal.toml"
    "${PARENT_DIR}/config/ninfer-metal.toml"
    "${BUNDLED_PROVIDERS_DIR}/io.ninfer.metal/current/config/ninfer-metal.toml"
    "${REPO_ROOT}/omlx-extend/config/ninfer-metal.toml"
  )
  for cand in "${candidates[@]}"; do
    if [[ -f "${cand}" ]]; then
      printf '%s\n' "${cand}"
      return 0
    fi
  done
  return 1
}

CONFIG_FILE=""
PASS_CONFIG_FLAG=0
if [[ -n "${CLI_CONFIG}" ]]; then
  CONFIG_FILE="${CLI_CONFIG}"
elif [[ "${NO_DEFAULT_CONFIG}" -eq 0 ]]; then
  if CONFIG_FILE="$(discover_config_file)"; then
    PASS_CONFIG_FLAG=1
  fi
fi

# 3. Extract display metadata ([server].host/port/providers_dir, [model].model_dir, [[models]]) from TOML without Python.
TOML_HOST=""
TOML_PORT=""
TOML_PROVIDERS=""
TOML_MODEL_DIR=""
TOML_HAS_MODELS=0

if [[ -n "${CONFIG_FILE}" && -f "${CONFIG_FILE}" ]]; then
  while IFS=$'\t' read -r key val; do
    case "${key}" in
      host) TOML_HOST="${val}" ;;
      port) TOML_PORT="${val}" ;;
      providers_dir) TOML_PROVIDERS="${val}" ;;
      model_dir) TOML_MODEL_DIR="${val}" ;;
      has_models) TOML_HAS_MODELS="${val}" ;;
    esac
  done < <(
    awk '
      function trim(s) {
        sub(/^[ \t\r\n]+/, "", s)
        sub(/[ \t\r\n]+$/, "", s)
        return s
      }
      function strip_comment(s,    i, c, in_s, in_d, esc) {
        in_s = 0; in_d = 0; esc = 0
        for (i = 1; i <= length(s); i++) {
          c = substr(s, i, 1)
          if (esc) { esc = 0; continue }
          if (c == "\\" && in_d) { esc = 1; continue }
          if (c == "\"" && !in_s) { in_d = !in_d; continue }
          if (c == "'"'"'" && !in_d) { in_s = !in_s; continue }
          if (c == "#" && !in_s && !in_d) { return substr(s, 1, i - 1) }
        }
        return s
      }
      function unquote(s) {
        s = trim(s)
        if ((substr(s, 1, 1) == "\"" && substr(s, length(s), 1) == "\"") ||
            (substr(s, 1, 1) == "'"'"'" && substr(s, length(s), 1) == "'"'"'")) {
          return substr(s, 2, length(s) - 2)
        }
        return s
      }
      {
        line = trim(strip_comment($0))
        if (line == "") next
        if (line == "[[models]]") {
          has_models = 1
          section = "models"
          next
        }
        if (substr(line, 1, 1) == "[" && substr(line, length(line), 1) == "]") {
          section = trim(substr(line, 2, length(line) - 2))
          next
        }
        eq = index(line, "=")
        if (eq == 0) next
        k = trim(substr(line, 1, eq - 1))
        v = trim(substr(line, eq + 1))
        if (section == "server") {
          if (k == "host") host = unquote(v)
          else if (k == "port") port = unquote(v)
          else if (k == "providers_dir" || k == "providers_dirs") {
            gsub(/^\[|\]$/, "", v)
            gsub(/"/, "", v)
            gsub(/'"'"'/, "", v)
            providers_dir = trim(v)
          }
        } else if (section == "model") {
          if (k == "model_dir" || k == "dir") model_dir = unquote(v)
        }
      }
      END {
        if (host != "") printf "host\t%s\n", host
        if (port != "") printf "port\t%s\n", port
        if (providers_dir != "") printf "providers_dir\t%s\n", providers_dir
        if (model_dir != "") printf "model_dir\t%s\n", model_dir
        printf "has_models\t%d\n", has_models + 0
      }
    ' "${CONFIG_FILE}"
  )
fi

# 4. Auto-discover first model containing `model.miv` (or legacy `model.omlxm5`) if none specified on CLI or TOML.
discover_first_model_dir() {
  local root found artifact_name
  local roots=("./models")
  if [[ -d "${REPO_ROOT}/models" && "${REPO_ROOT}/models" != "./models" ]]; then
    roots+=("${REPO_ROOT}/models")
  fi
  roots+=("${HOME}/Library/Application Support/oMLX/models")

  for artifact_name in "model.miv" "model.omlxm5"; do
    for root in "${roots[@]}"; do
      [[ -d "${root}" ]] || continue
      found="$(find "${root}" -maxdepth 3 -type f -name "${artifact_name}" 2>/dev/null | LC_ALL=C sort | head -n 1 || true)"
      if [[ -n "${found}" ]]; then
        dirname "${found}"
        return 0
      fi
    done
  done
  return 1
}

AUTO_MODEL_DIR=""
NEED_MODEL_FLAG=0
if [[ "${CLI_HAS_MODEL_DIR}" -eq 1 ]]; then
  # Only inject --model-dir if user gave a positional MODEL_DIR (not already in PASSTHROUGH_ARGS).
  HAS_MODEL_FLAG_IN_PASS=0
  for arg in "${PASSTHROUGH_ARGS[@]+"${PASSTHROUGH_ARGS[@]}"}"; do
    if [[ "${arg}" == "--model-dir" || "${arg}" == --model-dir=* ]]; then
      HAS_MODEL_FLAG_IN_PASS=1
      break
    fi
  done
  if [[ "${HAS_MODEL_FLAG_IN_PASS}" -eq 0 && -n "${CLI_MODEL_DIR}" ]]; then
    NEED_MODEL_FLAG=1
  fi
elif [[ -z "${TOML_MODEL_DIR}" && "${TOML_HAS_MODELS}" -eq 0 ]]; then
  if AUTO_MODEL_DIR="$(discover_first_model_dir)"; then
    NEED_MODEL_FLAG=1
  fi
fi

PASS_PROVIDERS_FLAG=0
EFFECTIVE_HOST="${CLI_HOST:-${TOML_HOST:-127.0.0.1}}"
EFFECTIVE_PORT="${CLI_PORT:-${TOML_PORT:-8080}}"
if [[ -n "${CLI_PROVIDERS_DIR}" ]]; then
  EFFECTIVE_PROVIDERS="${CLI_PROVIDERS_DIR}"
elif [[ -d "${BUNDLED_PROVIDERS_DIR}" ]]; then
  EFFECTIVE_PROVIDERS="${BUNDLED_PROVIDERS_DIR}"
  PASS_PROVIDERS_FLAG=1
elif [[ -n "${TOML_PROVIDERS}" ]]; then
  EFFECTIVE_PROVIDERS="${TOML_PROVIDERS}"
else
  EFFECTIVE_PROVIDERS="${PROVIDERS_DEFAULT_DIR}"
fi
EFFECTIVE_PROVIDERS="$(expand_tilde "${EFFECTIVE_PROVIDERS}")"

if [[ -n "${CLI_MODEL_DIR}" ]]; then
  EFFECTIVE_MODEL="${CLI_MODEL_DIR}"
elif [[ -n "${AUTO_MODEL_DIR}" ]]; then
  EFFECTIVE_MODEL="${AUTO_MODEL_DIR} (auto-discovered)"
elif [[ -n "${TOML_MODEL_DIR}" ]]; then
  EFFECTIVE_MODEL="$(expand_tilde "${TOML_MODEL_DIR}") (from config)"
elif [[ "${TOML_HAS_MODELS}" -eq 1 ]]; then
  EFFECTIVE_MODEL="[[models]] in config"
else
  EFFECTIVE_MODEL=""
fi

CMD=("${SERVE_BIN}")
if [[ "${PASS_CONFIG_FLAG}" -eq 1 && -n "${CONFIG_FILE}" ]]; then
  CMD+=("--config" "${CONFIG_FILE}")
fi
if [[ "${PASS_PROVIDERS_FLAG}" -eq 1 ]]; then
  CMD+=("--providers-dir" "${BUNDLED_PROVIDERS_DIR}")
fi
if [[ "${NEED_MODEL_FLAG}" -eq 1 ]]; then
  if [[ -n "${CLI_MODEL_DIR}" ]]; then
    CMD+=("--model-dir" "${CLI_MODEL_DIR}")
  elif [[ -n "${AUTO_MODEL_DIR}" ]]; then
    CMD+=("--model-dir" "${AUTO_MODEL_DIR}")
  fi
fi
if [[ "${CLI_HAS_LORA_ADAPTER}" -eq 0 && -n "${NINFER_LORA_ADAPTER:-}" ]]; then
  CMD+=("--lora" "$(expand_tilde "${NINFER_LORA_ADAPTER}")")
fi
if [[ "${CLI_HAS_LORA_MODE}" -eq 0 && -n "${NINFER_LORA_MODE:-}" ]]; then
  validate_lora_mode "${NINFER_LORA_MODE}" "NINFER_LORA_MODE"
  CMD+=("--lora-mode" "${NINFER_LORA_MODE}")
fi
if [[ ${#PASSTHROUGH_ARGS[@]} -gt 0 ]]; then
  CMD+=("${PASSTHROUGH_ARGS[@]}")
fi

if [[ -z "${EFFECTIVE_MODEL}" ]]; then
  if [[ "${DRY_RUN}" -eq 1 ]]; then
    echo "============================================================"
    echo " ninfer-metal-serve (1-Click Native M5+ Inference Server)"
    echo "------------------------------------------------------------"
    echo " Binary:    ${SERVE_BIN}"
    echo " Config:    ${CONFIG_FILE:-<compiled defaults>}"
    echo " Model:     <none found in ./models or ~/Library/Application Support/oMLX/models>"
    echo " Providers: ${EFFECTIVE_PROVIDERS}"
    echo " Endpoint:  http://${EFFECTIVE_HOST}:${EFFECTIVE_PORT}/v1"
    echo "============================================================"
    echo "Hint: $0 /path/to/model-dir [extra ninfer-metal-serve args...]"
    printf 'Dry-run command:'
    printf ' %q' "${CMD[@]}"
    printf '\n'
    exit 0
  fi
  echo "No model.miv (or model.omlxm5) found in ./models/ or ~/Library/Application Support/oMLX/models/. Usage: $0 /path/to/model-dir [extra ninfer-metal-serve args...]" >&2
  exit 2
fi

echo "============================================================"
echo " ninfer-metal-serve (1-Click Native M5+ Inference Server)"
echo "------------------------------------------------------------"
echo " Binary:    ${SERVE_BIN}"
echo " Config:    ${CONFIG_FILE:-<compiled defaults>}"
echo " Model:     ${EFFECTIVE_MODEL}"
echo " Providers: ${EFFECTIVE_PROVIDERS}"
echo " Endpoint:  http://${EFFECTIVE_HOST}:${EFFECTIVE_PORT}/v1"
echo "============================================================"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  printf 'Dry-run command:'
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"
