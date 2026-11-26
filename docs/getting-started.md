**English** | [简体中文](i18n/zh-CN/getting-started.md) | [繁體中文](i18n/zh-TW/getting-started.md) | [日本語](i18n/ja/getting-started.md) | [한국어](i18n/ko/getting-started.md) | [Español](i18n/es/getting-started.md) | [Français](i18n/fr/getting-started.md) | [Deutsch](i18n/de/getting-started.md)

# Getting Started & Configuration

This guide covers building the zero-MLX `ninfer-metal` core (`miv/`), converting weights to `.miv`, launching the standalone C++20 OpenAI server (`ninfer-metal-serve`) in one command, integrating **`miv Core Backend`** with `oMLX-extend`, customizing `ninfer-metal.toml`, and calling the low-level CLI and daemon binaries. If you use the `oMLX` macOS app, start with the **[`oMLX-extend` Visual Guide](omlx-extend-guide.md)** for a 5-step screenshot walkthrough.

---

## Prerequisites

| Mode | Hardware | OS & Toolchain |
|---|---|---|
| **Runtime (prebuilt bundle)** | Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`) | `macOS 27+` (zero Xcode, Python, MLX, or Homebrew dependencies) |
| **Build from source** | Apple Silicon (`arm64`) | `macOS 27+`, `Xcode 27+` (macOS 27 SDK), `CMake 3.27+`, `Ninja` |

---

## 1. Build `ninfer-metal` (`miv/`, Zero MLX / Zero Python)

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

# Configure and compile all 38 pure Metal 4 .metallib shaders and 16 binaries
cmake -S . -B build/miv-v001 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOMLX_M5_BUILD_NAX_METALLIB=ON \
  -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF \
  -DOMLX_M5_BUILD_PROBES=ON \
  -DOMLX_M5_BUILD_PYTHON=OFF
cmake --build build/miv-v001

# Verify version identity for both standalone server and sidecar daemon
./build/miv-v001/miv/ninfer-metal-serve --version
# ninfer-metal-serve miv-v0.01 (codename=miv semver=0.0.1 core_abi=1)

./build/miv-v001/miv/mseries_backendd --version
# mseries-backendd miv-v0.01 (codename=miv semver=0.0.1 core_abi=1) protocol=2

# Run the CTest suite
ctest --test-dir build/miv-v001 --output-on-failure
```

To also build the optional `omlx_m5_native` Python extension, pass `-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>`. Full development presets live in [`CMakePresets.json`](../CMakePresets.json).

---

## 2. Package & Convert Weights

### Clean-Mac Release Bundle

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

This outputs `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` (plus `.sha256`, `provider.json`, and `latest.json`) and runs `otool -L` across all 16 Mach-O binaries (`ninfer-metal-serve`, `mseries_backendd`, `m5rt_runner`, and 13 hardware/calibration probes) to confirm they link solely against Apple system frameworks (`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` and `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`).

Unpack the archive on any Apple M5+ Mac running macOS 27+ and run `install.sh` (copies the default `ninfer-metal.toml` without overwriting existing user edits) to install under:

```text
~/Library/Application Support/oMLX/Providers/io.ninfer.metal/
├── current -> versions/miv-v0.01/     # Atomic symlink for hot-update & rollback
└── versions/
    └── miv-v0.01/
        ├── provider.json
        ├── RELEASE-MANIFEST.json
        ├── bin/                       # ninfer-metal-serve, mseries_backendd, m5rt_runner, probes
        ├── config/ninfer-metal.toml   # Ready-to-run shared M5+ defaults
        ├── scripts/start_ninfer_serve.sh
        ├── metallib/                  # 38 precompiled Metal 4 .metallib files
        ├── calibrations/              # SKU-isolated JSON profiles (64-GPU, 80-GPU)
        └── tools/
```

### Convert a Model to `.miv`

Offline NAX pre-tiling packs `MXFP4` (`mxfp4_e2m1_g32_e8m0_v1` -> `mx_nax_tiled_bk128_v1`), `oQ` Affine `q3–q6`, and 8-bit `affine8` (`oq_affine_q8_g64_bf16_v1` / `oq_affine_q8_g128_bf16_v1` -> `oq_nax_tiled_g64_v1` with `nax_codes` + `nax_sb` and zero `nax_high*` planes, including per-layer mixed `MXFP4 + affine8` checkpoints) once so Prefill, `T=1` AR decode, and speculative verify share one zero-copy resident memory copy:

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.miv \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. Run Standalone with `ninfer-metal-serve` (Native C++)

`ninfer-metal-serve` is a single C++20 binary linked directly against `libomlx_m5_core_runtime.a`. It serves in-process `miv` models and external Path B Provider SPI v2 microkernels in parallel on one HTTP/SSE port, bundling `ninfer-metal.toml` loading, a C++ BPE tokenizer (`tokenizer.json`), Jinja chat templates (`chat_template.jinja` / `tokenizer_config.json` / ChatML fallback), `<think>` reasoning splitting, and Qwen XML/JSON `<tool_call>` parsing — with zero Python and zero MLX.

### 1-Click Launch (Shared Zero-Edit Config)

Start `ninfer-metal-serve` right away without hand-writing a config file. The launcher script auto-discovers `ninfer-metal-serve` and loads [`omlx-extend/config/ninfer-metal.toml`](../omlx-extend/config/ninfer-metal.toml) — the exact same default file reused by `oMLX-extend` (`max_concurrency = 1`, `max_context = 65536`, `kv_dtype = "bf16"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`, `providers_dir = ["~/Library/Application Support/oMLX/Providers"]`):

```bash
# 1-click launch with a specific model directory (default port 8080, separate from oMLX port 8000)
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# Or omit the model directory to auto-pick the first .miv model in ./models or oMLX's models folder
./omlx-extend/scripts/start_ninfer_serve.sh

# Pass any extra CLI flag (including --lora / --lora-mode) to override the shared default config
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --lora /path/to/my-lora --lora-mode merge
```

`start_ninfer_serve.sh` also honors `NINFER_LORA_ADAPTER` and `NINFER_LORA_MODE` environment overrides when `--lora` / `--lora-mode` are not passed explicitly.

### Direct CLI Overrides or Custom `ninfer-metal.toml`

Running `ninfer-metal-serve` directly also auto-loads the shared default `ninfer-metal.toml` unless `--config` or `--no-default-config` is passed; CLI flags always override TOML values:

```bash
# Direct CLI launch with explicit overrides
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# Launch with a HuggingFace PEFT or mlx-lm LoRA adapter (zero offline conversion)
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --lora /path/to/adapter-dir \
  --lora-mode merge

# Launch with explicit DFlash2 speculative decoding (or rely on speculative.backend = "auto")
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# Load a specific config file (CLI flags override TOML values)
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### Native Dual-Mode LoRA (`merge` vs `bypass`, Zero Conversion)

`ninfer-metal` loads standard **HuggingFace PEFT** (`adapter_config.json` + `adapter_model.safetensors`) and **`mlx-lm`** (`adapter_config.json` + `adapters.safetensors`) adapter directories directly in pure C++20/Metal 4 with **zero offline weight conversion** (supporting `r`/`rank`, `lora_alpha`/`alpha`, `scale`, `use_rslora`, `rank_pattern`, `alpha_pattern`, `fan_in_fan_out`, and `BF16`/`FP16`/`FP32` safetensors):

| Mode (`--lora-mode` / `lora_mode`) | Execution & UMA Behavior | Best For |
|---|---|---|
| **`"merge"`** *(default)* | **Mode A — Load-Time UMA GPU Fusion**: Keeps on-disk `.miv` completely read-only; unpacks only LoRA-targeted base projections (`MXFP4` or `oQ` Affine `q4`/`q6`/`q8`) on GPU at load time, fuses $W_{\text{eff}} = W_{\text{base}} + \text{scale} \cdot (B A)$ in `FP32`, and quantizes + NAX-pretiles directly into resident UMA buffers as `oq_nax_tiled_g64_v1` `q8` in `AffineNaxCache` (**`0` extra runtime kernel dispatches**). | Maximum steady-state decode & prefill throughput for a fixed adapter |
| **`"bypass"`** | **Mode B — Runtime BF16 Low-Rank Injection**: Leaves base `Mxfp4NaxCache` / `AffineNaxCache` untouched, stores `(A, B, scale)` as resident `BF16` Metal buffers, and injects `y += scale * ((x @ A^T) @ B^T)` at linear projection dispatch points with **zero re-quantization drift** and fast `reload_lora` adapter hot-swapping without reloading base weights. | Zero re-quantization drift evaluation and rapid multi-adapter switching |

Both modes stay exact under `mtp` and `dflash2` speculative target verification and report `lora_adapter`, `lora_mode`, `lora_buffer_bytes`, and `lora_adapted_layers` in `/health`, `/v1/models`, and sidecar `stats` (accounted against `memory_budget_bytes`).

### Running Multiple Microkernels in Parallel (`[[models]]` / `--providers-dir`)

Host in-process `miv` (Path A) and Path B SPI v2 sidecar microkernels together on one port with unified UMA budgeting (`--memory-budget-gb`, default 81% physical RAM) and automatic `gpu_contended` scheduling:

```bash
./build/miv-v001/miv/ninfer-metal-serve \
  --providers-dir "$HOME/Library/Application Support/oMLX/Providers" \
  --model-dir /path/to/Qwen3.8-27B \
  --model-dir /path/to/Llama-3.3-70B-Instruct \
  --memory-budget-gb 64
```

For per-model options (`provider = "auto" | "inproc" | "<provider_id>"`, `kv_dtype`, `max_concurrency`, `memory_budget_gb`), declare `[[models]]` blocks in `ninfer-metal.toml` and pass `--config`. Requests route by JSON `"model"` (`404 model_not_found` for unknown IDs).

### Connect Clients (Open WebUI, Cursor, Cline, NewAPI, OpenAI SDK)

Point any OpenAI-compatible client to `http://127.0.0.1:8080/v1`:

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": "Write a quicksort in C++20."}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### HTTP Endpoints (`miv-v0.01`)

| Endpoint | Behavior |
|---|---|
| `GET /health` | Readiness JSON (`status`, `model`, `models[]`, `gpu_contended`, `memory_budget_bytes`, `lora_adapter`, `lora_mode`, `lora_buffer_bytes`, `lora_adapted_layers`, `version`) |
| `GET /metrics` | Prometheus text exposition (aggregate + per-model `model`/`provider`/`backend_mode` series: `requests_total`, `active_requests`, `generated_tokens_total`, UMA/cache stats) |
| `GET /v1/models`, `GET /v1/models/:id` | OpenAI model listing with `max_model_len`, `provider_id`, `backend_mode`, and active LoRA metadata (`lora_adapter`, `lora_mode`, `lora_buffer_bytes`, `lora_adapted_layers`) |
| `POST /v1/chat/completions` | Routes by JSON `"model"` (`404 model_not_found` on unknown ID; single-model omission defaults to sole model); sync JSON and streaming SSE (`chat.completion.chunk`), `<think>...</think>` to `reasoning_content`, Qwen XML and JSON `<tool_call>` parsing, `stop` strings, `usage`, `stream_options.include_usage`, `[DONE]` |
| `POST /v1/completions` | Raw text completion routed by JSON `"model"` (sync JSON, streaming SSE, `usage`, `[DONE]`) |

Optional Bearer auth (`--api-key`) returns `401` on mismatch; CORS preflight (`OPTIONS`) is enabled by default.

### `ninfer-metal-serve` CLI Flags

When `--config` is omitted, `ninfer-metal-serve` auto-loads [`omlx-extend/config/ninfer-metal.toml`](../omlx-extend/config/ninfer-metal.toml) (`max_concurrency = 1`, `max_context = 65536`, `kv_dtype = "bf16"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`). Pass `--no-default-config` to start from the bare compiled defaults listed below; see [`ninfer-metal.toml.example`](../omlx-extend/config/ninfer-metal.toml.example) for the full commented reference.

| Flag | Bare CLI Default / Values | Description |
|---|---|---|
| `--config PATH` | auto-discovered | Load an explicit `ninfer-metal.toml` (`[server]`, `[model]`, `[[models]]`, `[engine]`, `[speculative]`); CLI flags override TOML |
| `--no-default-config` | off | Skip automatic `ninfer-metal.toml` discovery and start from bare compiled defaults |
| `--model-dir PATH` | required (or via TOML) | Model directory containing `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, and model artifact; **repeatable** for multi-model parallel serving |
| `--providers-dir PATH` | auto (`$OMLX_PROVIDERS_DIR`, `~/Library/Application Support/oMLX/Providers`) | **Repeatable** root directory of Path B SPI v2 microkernel sidecars; scans `<dir>/*/current/provider.json` (fallback `<dir>/*/provider.json`) |
| `--provider MODE` | `auto` (`auto` \| `inproc` \| `<provider_id>`) | Backend routing mode: O(1) auto-resolution, in-process `miv` (`inproc`), or explicit Path B SPI v2 `provider_id` |
| `--metallib-dir PATH` | auto-discovered | Metal 4 / NAX `.metallib` shader directory (`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`). Alias: `--kernel-dir` (points to Metal shaders, **not** Path B microkernels — use `--providers-dir` for microkernels) |
| `--artifact PATH` | auto-discovered | Explicit target `.miv` (or provider) artifact path |
| `--lora PATH`, `--lora-adapter PATH` | `""` (disabled) | HuggingFace PEFT (`adapter_model.safetensors`) or `mlx-lm` (`adapters.safetensors`) LoRA directory (expands `~/`, zero conversion) |
| `--lora-mode MODE` | `merge` (`merge` \| `bypass`) | LoRA execution mode: load-time GPU `q8` fusion (`merge`, 0 runtime overhead) or BF16 low-rank runtime injection (`bypass`, fast hot-swap) |
| `--draft-dir PATH`, `--draft-artifact PATH`, `--mtp-artifact PATH` | — | Draft model directory or `.miv` artifact for `dflash2` / `mtp` |
| `--chat-template PATH`, `--model-id NAME` | auto | Override Jinja chat template file or advertised model ID |
| `--host ADDR`, `--port PORT` | `127.0.0.1`, `8080` | HTTP bind address and port |
| `--api-key KEY`, `--cors` / `--no-cors` | `""`, enabled | Require `Authorization: Bearer <KEY>`; control CORS headers |
| `--memory-budget-gb FLOAT` | `81%` physical RAM | Server-wide UMA memory budget cap (GiB) across all hosted models |
| `--max-concurrency N` (`-c`, `-b`) | `1` (`1..8`) | Multi-user concurrent batch slots (`B`/`C`) |
| `--max-context N` | `32768` | Context window length (`L`) |
| `--kv-dtype MODE` | `bf16` (`bf16` \| `affine4_g128`) | KV cache storage format |
| `--spec-backend MODE` | `none` (`auto` \| `none` \| `mtp` \| `dflash2`) | Speculative decoding backend (`auto` enables `dflash2` if companion `dflash2/` exists, else `mtp`, else `none`) |
| `--draft-tokens N` | `0` (auto; `1..7` explicit) | Draft tokens per verify round |
| `--rolling-prefix-cache on\|off` | unset (engine default) | Enable/disable multi-turn rolling UMA prefix reuse with `1×` ancestor supersede |
| `--prefix-cache-gb FLOAT` | unset (`0..128`) | Prefix cache and UMA checkpoint pool budget (GiB) |
| `--sampling-contract NAME` | `position_coupled_v1` | Speculative verification sampling contract (`legacy_rejection_v1` \| `position_coupled_v1`) |
| `--version`, `--help` (`-h`) | — | Print structured version string or usage summary |

---

## 4. Run with `oMLX-extend` (Sidecar Coexistence)

This monorepo tracks the patched `omlx/` host directly alongside `omlx-extend/` (synced to upstream `jundot/omlx` via [`omlx-extend/patches/`](../omlx-extend/patches/)).

```bash
# Verify or apply the Provider SPI patch on ./omlx and run pytest
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test

# Sync ./omlx to a specific upstream jundot/omlx tag
./omlx-extend/scripts/sync_omlx.sh --tag v0.8.3 --omlx-dir ./omlx
```

Once `io.ninfer.metal/current/` is installed, `oMLX-extend` auto-discovers **`miv Core Backend`** (`mseries_backendd`, falling back to `com.omlx.mseries/current/`, `PATH`, or `build/*/miv/`) and displays installed Path A and Path B microkernels under **`Integrations → External Engines`**. Any model directory containing `model.miv` (or legacy `model.omlxm5`), `model_pack.json`, `omlxm5-manifest.json`, or convertible Qwen `.safetensors` routes lazily to **`miv Core Backend`** (auto-repacking `.safetensors → model.miv` on first Load); unadapted models stay on Stock MLX unchanged. Because `io.ninfer.metal` declares `"lora"` in `capabilities` and `"lora_adapter"` in `supported_standard_settings` (with `lora_mode` `"merge" | "bypass"`, `reload_required: true` exposed in `options_schema` under `external_engine_options`), standard `oMLX` LoRA selection works directly on `miv Core Backend` while external providers that do not declare `"lora_adapter"` preserve conflict checks. Flip **`Use External Engine`** off in **`Model Settings → Advanced Settings`** (writes `external_engine: "stock"`) to force Stock MLX for an adapted model.

For a screenshot walkthrough of the `oMLX` macOS app, see **[`oMLX-extend` Visual Guide](omlx-extend-guide.md)**. For host internals and UI setting mappings, see [`omlx-extend/README.md`](../omlx-extend/README.md).

---

## 5. Configure `ninfer-metal.toml`

Both standalone `ninfer-metal-serve` and `oMLX-extend` reuse the same ready-to-run default config ([`omlx-extend/config/ninfer-metal.toml`](../omlx-extend/config/ninfer-metal.toml)) with zero manual edits. When `--config` is not passed, both hosts check the same paths in descending priority (first existing file wins):

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml` (relocatable release bundle)
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

To keep custom defaults across updates, copy the default config (or the commented [`ninfer-metal.toml.example`](../omlx-extend/config/ninfer-metal.toml.example)) into `~/.config/ninfer-metal/`:

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

In `oMLX-extend` sidecar mode, per-model UI settings (`lora_adapter` and `external_engine_options["ninfer-metal"]`) override `[sidecar.models."<id>"]` in TOML, which override the shared `[model]` / `[engine]` / `[sidecar]` / `[speculative]` defaults. Across all hosts, LoRA settings (`lora_adapter` and `lora_mode`) follow strict **`CLI > per-model (`[[models]]` / `[sidecar.models."<id>"]` / `external_engine_options`) > top-level TOML (`[model]` / `[engine]` / `[sidecar]`)`** precedence.

### Core Knobs

| Key | Section | Values (`miv-v0.01`) | Notes |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`, `8080`, `""` | Standalone `ninfer-metal-serve` settings (ignored by `oMLX-extend` AF_UNIX sidecar). |
| `providers_dir` / `memory_budget_gb` | `[server]` | string or array, float (GiB) | Path B SPI v2 provider roots (scans `<dir>/*/current/provider.json`) and server-wide UMA cap (default `81%` RAM). |
| `model_dir` / `artifact` (`artifact_path`) | `[model]` | Paths | Standalone model directory and optional explicit `.miv` (or provider) artifact path; both `artifact` and `artifact_path` (plus `draft_artifact`/`draft_artifact_path` and `metallib_dir`/`kernel_dir`) are accepted. |
| `lora_adapter` / `lora_mode` | `[model]`, `[engine]`, `[sidecar]`, `[sidecar.models."<id>"]`, `[[models]]` | `""` (path), `"merge"` \| `"bypass"` (default `"merge"`) | Direct HuggingFace PEFT or `mlx-lm` adapter directory and mode (`"merge"` load-time UMA `q8` fusion vs `"bypass"` BF16 low-rank injection). Precedence: `CLI > per-model > TOML`. |
| `id` / `model_dir` / `provider` / overrides | `[[models]]` | Array of tables | Multi-model parallel hosting in standalone `ninfer-metal-serve`. Each entry inherits unset `[engine]`/`[speculative]` defaults (including `lora_adapter` / `lora_mode`) and sets `provider = "auto" \| "inproc" \| "<provider_id>"`. |
| `max_concurrency` | `[engine]` | `1..8` (CLI default `1`, template `4`) | Multi-user concurrent batch slots (**`B`** in `miv` docs, **`C`** in upstream `ninfer`). Inherits oMLX `max_concurrent_requests` when omitted in sidecar mode. |
| `max_context` | `[engine]` | `32768`–`262144+` (CLI default `32768`, template `65536`) | Context window length (**`L`**). Maps from oMLX `max_context_window` in sidecar mode. |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128` stores `0.5625 B/val` with verified `0-ULP` parity at `64K+`. `int8_g64` is absent in `miv-v0.01`. |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | Enables multi-turn rolling UMA prefix reuse with `1×` in-place ancestor supersede. |
| `prefix_cache_gb` | `[engine]` | `0`–`128` (template `4.0`) | Prefix cache and UMA checkpoint pool budget (GiB). |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`, `0` (auto) or `1..7` | `"auto"` enables `dflash2` if companion `dflash2/` exists, else `mtp`, else `none`. Maps from oMLX Acceleration toggles (`mtp_*`, `dflash_*`) in sidecar mode. |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"` (CLI default) \| `"position_coupled_v1"` | Speculative verification sampling contract. |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | Prefill execution backend selector. |

Default config: [`omlx-extend/config/ninfer-metal.toml`](../omlx-extend/config/ninfer-metal.toml) · Full commented template: [`omlx-extend/config/ninfer-metal.toml.example`](../omlx-extend/config/ninfer-metal.toml.example).

---

## 6. Low-Level CLI, Daemon & C++ Library

Beyond `ninfer-metal-serve` and `oMLX-extend`, `miv/` exposes four lower-level integration surfaces:

1. **`m5rt_runner` (token-ID CLI)** — Runs single-stream, batched (`--batch-mode on --max-concurrency 4`), DFlash2 speculative, or LoRA-adapted (`--lora <adapter_dir>`, alias `--lora-adapter`, with `--lora-mode <merge|bypass>`) inference from integer token IDs and emits NDJSON metrics/tokens:
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.miv \
     --config /path/to/config.json \
     --lora /path/to/adapter-dir \
     --lora-mode merge \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd` (AF_UNIX NDJSON RPC v2 daemon)** — Persistent token-ID engine server supporting `hello`, `load` (including `"lora_adapter"` and `"lora_mode": "merge" | "bypass"`, with fast `reload_lora` hot-swap in `bypass` mode), `submit`, `cancel`, `subscribe_events`, `trim_cache`, `set_scheduler_hint`, `stats`, and `shutdown`, plus CLI defaults `--lora` (`--lora-adapter`) and `--lora-mode <merge|bypass>`:
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a` (C++20 static library)** — Embed `omlx::m5::Engine` directly via [`miv/include/omlx_m5/engine.h`](../miv/include/omlx_m5/engine.h).
4. **`omlx_m5_native` (optional Python module)** — Nanobind bindings when built with `-DOMLX_M5_BUILD_PYTHON=ON`.
