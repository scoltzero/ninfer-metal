# oMLX-extend — Thin External Provider SPI & `miv Core Backend (ninfer-metal)` Adapter

`oMLX-extend` is the upstream-synced extension layer over [`jundot/omlx`](https://github.com/jundot/omlx) (tagged `omlx-extend-vX.Y.Z` / `omlx-extend-vX.Y.Z-mix1..9`, currently `omlx-extend-v0.8.3`) that integrates the pure-Metal-4 **`miv Core Backend (ninfer-metal)`** inference core (first codename **`miv`**, `miv-v0.01`, `semver: 0.0.1`, `core_abi: 1`, `protocol_version: 2`) while keeping stock oMLX **100% untouched** for unadapted models and keeping the oMLX UI minimal and clean.

## 1. Architecture: Thin Provider SPI + Zero Stock Regression

- **Unadapted models -> 100% stock oMLX**: `detect_adapted(model_dir)` runs a fast O(1) filesystem check (`model.miv`, legacy `model.omlxm5`, `model_pack.json`, `omlxm5-manifest.json`, zero subprocess spawn). Models without an adapted `.miv` artifact (and not auto-convertible Qwen `.safetensors`) return `None` and run unmodified through stock oMLX (`BatchedEngine` / `VLMBatchedEngine`) with zero sidecar overhead.
- **Adapted models -> automatic lazy-load routing**: Models carrying `.miv` (or legacy `.omlxm5`) artifacts — covering pre-tiled `MXFP4` (`mx_nax_tiled_bk128_v1`), `oQ` Affine `q3–q6`, and 8-bit `affine8` (`oq_affine_q8_g64_bf16_v1` / `oq_affine_q8_g128_bf16_v1` via `--nax-pretile` -> `oq_nax_tiled_g64_v1`, including per-layer `MXFP4 + affine8` mixed models), as well as convertible Qwen `.safetensors` models (which auto-repack to `model.miv --nax-pretile` on first Load) — auto-route to `miv Core Backend (ninfer-metal)` (`m5rt`) when the provider binary is installed, spawning the out-of-process `mseries_backendd` UDS sidecar lazily on first request (or when pinned). Setting `external_engine: "stock"` opts any adapted model back into stock in-process MLX.
- **Simultaneous coexistence (sidecar + stock MLX)**:
  - **Unified UMA budget**: `load` RPC forwards `memory_budget_bytes` derived from oMLX's `ProcessMemoryEnforcer` ceiling minus active resident models (accounting for resident LoRA buffers reported via `lora_buffer_bytes`), preventing jetsam OOM when coexisting with stock MLX models.
  - **Soft/hard pressure cache trim**: Before evicting a coexisting model, `ProcessMemoryEnforcer` invokes `trim_cache(target_free_bytes)` to drop unpinned prefix-cache entries and checkpoint slabs on the sidecar.
  - **GPU contention hints**: `EnginePool` sends `set_scheduler_hint(gpu_contended=true/false)` whenever both an in-process MLX model and an external sidecar model are active concurrently.
  - **Clean teardown**: `stop()` drains the process group and verifies `process_memory_bytes() == 0` before admitting another model.

## 2. Minimal UI + Config-File Control Split

Stock oMLX UI stays clean — **zero extra settings forms are added**:

| Control Category | Where Configured | Mapped `ninfer-metal` (`miv-v0.01`) Knob |
|---|---|---|
| Context Window | Stock oMLX Model Settings (`max_context_window`) | `max_context` (Context `L`, up to `262144`+, no 32K cap) |
| Multi-user Concurrency | Stock oMLX Global Settings (`max_concurrent_requests`) | `max_concurrency` (Concurrency `B`/`C`, clamped `1..8`) |
| Lightning MTP | Stock oMLX Acceleration (`mtp_enabled`, `mtp_fixed_depth`, `mtp_adaptive_max_depth`) | `speculative.backend = "mtp"`, `draft_tokens` (`0..4`, with `0` for automatic depth) |
| DFlash2 | Stock oMLX Acceleration (`dflash_enabled`, `dflash_draft_model`, `dflash_block_size`) | `speculative.backend = "dflash2"`, `draft_tokens = block_size - 1`, up to `7` |
| KV Cache Quantization | Stock oMLX Memory Saving (`turboquant_kv_enabled`, `turboquant_kv_scheme == "affine4"`) | `kv_dtype = "affine4_g128"` (`0.5625 B/val`); disabled -> `"bf16"` *(note: KV cache `int8_g64` is intentionally NOT mapped in `miv-v0.01`, whereas linear weight `affine8` `oq_affine_q8_g64/g128_bf16_v1` IS supported via `--nax-pretile`)* |
| LoRA Adapter | Stock oMLX Model Settings (`lora_adapter`) | `lora_adapter` (resolved local directory path forwarded in SPI v2 `load` RPC; unlocked via `"lora"` in `capabilities` and `"lora_adapter"` in `supported_standard_settings`) |
| LoRA Execution Mode | Admin UI `options_schema` (`external_engine_options["ninfer-metal"]["lora_mode"]`) or `ninfer-metal.toml` | `lora_mode = "merge"` (default) \| `"bypass"` (`reload_required: true`) |
| Advanced / Core-only Knobs | `ninfer-metal.toml` or `model_settings.json` -> `external_engine_options["ninfer-metal"]` | `lora_mode`, `rolling_prefix_cache`, `prefix_cache_gb`, per-model `max_concurrency` override, `sampling_contract`, `prefill_engine`, `artifact_path`, standalone `[server]` params |

Standard UI sections unlock **per setting** against the active provider's `supported_standard_settings`. Capability-aware validation allows `lora_adapter` alongside `external_engine` when the target provider declares `"lora_adapter"` in `supported_standard_settings` (`io.ninfer.metal` / `ninfer-metal` / `m5rt`), while **preserving the conflict check** for external providers that do not declare `"lora_adapter"` (such as `Splash`, `supported_standard_settings: []`).

### Native Dual-Mode LoRA (`merge` vs `bypass`)

Standard **HuggingFace PEFT** (`adapter_config.json` + `adapter_model.safetensors`) and **`mlx-lm`** (`adapter_config.json` + `adapters.safetensors`) directories load directly on `miv Core Backend` with **zero offline conversion**:

- **Mode A — `"merge"` (default)**: At load time, unpacks only LoRA-hit base projections on GPU, fuses $W_{\text{eff}} = W_{\text{base}} + \text{scale} \cdot (B A)$ in `FP32`, and quantizes + NAX-pretiles into resident UMA buffers as `oq_nax_tiled_g64_v1` `q8` in `AffineNaxCache` — keeping the on-disk `.miv` file completely read-only and adding **0 runtime kernel overhead**.
- **Mode B — `"bypass"`**: Keeps base `Mxfp4NaxCache` / `AffineNaxCache` untouched and injects `BF16` low-rank residuals `y += scale * ((x @ A^T) @ B^T)` at runtime with **zero re-quantization drift** and fast `reload_lora` hot-swapping without reloading base weights.
- **Observability & Memory Accounting**: `mseries_backendd` reports `lora_adapter`, `lora_mode`, `lora_buffer_bytes`, and `lora_adapted_layers` in `load` and `stats` RPCs, feeding directly into `ProcessMemoryEnforcer` / `memory_budget_bytes` UMA admission accounting.

### Settings that reach miv

The black dot marks a field that reaches miv when that model uses the miv provider. Global defaults apply when the model inherits them. The model settings panel shows the daemon's loaded values after Load, including the source of its concurrency limit.

| Setting | Effective behavior |
|---|---|
| Global concurrency and model provider concurrency | The model provider option wins, followed by model TOML and shared TOML. An omitted value inherits oMLX global concurrency. miv supports 1–8 slots. The shipped preset is B=1. |
| Model context window and global context policy | Admission, load and the loaded model use the same window, bounded by the model. Only the explicit oMLX model window can exceed the global context policy. |
| MTP, DFlash2, draft selection and depth | An explicit backend takes effect, including `none` for AR. Auto discovers a matching DFlash2 draft, then MTP, then AR. The core selects the depth when it receives 0. MTP supports depths up to 4 and DFlash2 up to 7. DFlash2 block size is depth plus one, subject to the checkpoint's limit. |
| KV compression, prefix reuse, sampling contract | Use `bf16` or `affine4_g128` for miv KV. The miv prefix cache uses its own rolling switch and budget. The shipped verification contract is `position_coupled_v1`. |
| Artifact paths and LoRA | Model and draft paths are resolved on load. The selected LoRA adapter and its merge or bypass mode are also load settings. |
| Generation defaults | Max output tokens, temperature, top-p and top-k affect the submit RPC when inherited. The request can supply these values, and the available context limits output tokens. Seed and stop also affect individual requests. |

Stock SSD and hot cache parameters, GDN storage, and the Stock prefill scheduling controls belong to the oMLX scheduler. The host still applies memory pressure and model idle policy to miv. With `position_coupled_v1`, a sidecar that advertises `thinking_budget` or `sampling_penalties` can enforce the corresponding request controls. Thinking budgets require a tokenizer with supported thinking markers. Requests with active controls keep speculative decoding in their slots and currently run outside packed batches. An explicit unsupported control returns HTTP 400 before streaming. Unsupported inherited settings are reported in the response warning and fall back to backend defaults. String stops can skip the leading think block when the API asks for that behavior.

The default is a single-client preset. For a deliberate throughput comparison on a measured model, use a per-model `max_concurrency = 4` with representative concurrent requests. Compare cold prefill, warm continuation, actual speculative rounds and memory usage as well as generated tokens per second.

## 3. Config File Discovery & Precedence (`ninfer-metal.toml`)

Copy [`config/ninfer-metal.toml.example`](config/ninfer-metal.toml.example) to one of these locations (lowest to highest precedence, overridden by per-model `external_engine_options` / `m5rt_options`):

1. `~/Library/Application Support/oMLX/ninfer-metal.toml`
2. `~/.config/ninfer-metal/ninfer-metal.toml`
3. `<model_dir>/ninfer-metal.toml`
4. Path pointed to by `$NINFER_METAL_CONFIG`

`lora_adapter = ""` and `lora_mode = "merge"` (`"merge"` | `"bypass"`) are supported across `[model]`, `[engine]`, `[sidecar]`, `[sidecar.models."<id>"]`, and `[[models]]` with strict **`CLI > per-model (`[[models]]` / `[sidecar.models."<id>"]` / `external_engine_options`) > TOML (`[model]` / `[engine]` / `[sidecar]`)`** precedence.

## 4. Provider Hot-Update Layout (`io.ninfer.metal`)

`ninfer-metal` releases unpack side-by-side without mutating oMLX's Python environment:

```text
~/Library/Application Support/oMLX/Providers/
└── io.ninfer.metal/
    ├── current -> versions/miv-v0.01/     # Atomic symlink flip for zero-restart hot-update & 1-click rollback
    └── versions/
        └── miv-v0.01/
            ├── provider.json              # ABI/protocol manifest (see provider/provider.json.template)
            ├── RELEASE-MANIFEST.json
            ├── bin/mseries_backendd       # Pure Apple-system-framework Mach-O (zero MLX/Homebrew dylibs)
            ├── bin/m5rt_runner
            ├── metallib/*.metallib        # Precompiled Metal 4 / NAX shaders (SDK 27+)
            ├── calibrations/*.json
            └── tools/*.py
```

Auto-discovery checks `io.ninfer.metal/current/`, legacy `com.omlx.mseries/current/`, `PATH`, and repo build directories (`build/*/miv/` then `build/*/m5rt/`).

### Multi-Microkernel Parallel Sidecars (Path B Registry & Sync)

`oMLX-extend` also discovers any additional Provider SPI v2 microkernel installed under `~/Library/Application Support/oMLX/Providers/<provider_id>/current/provider.json` (override with `OMLX_PROVIDERS_DIR`) and routes models deterministically by `provider_id`, `supported_topologies` / `supported_model_types`, and `artifact_extensions` — running multiple sidecars in parallel with shared UMA admission budgeting and GPU contention hints.

Manage registered Path A / Path B microkernels via [`registry/microkernels.json`](registry/microkernels.json) and [`scripts/sync_microkernels.sh`](scripts/sync_microkernels.sh):

```bash
# List registered microkernels and locally active versions:
./omlx-extend/scripts/sync_microkernels.sh --list

# Install / hot-update a specific provider or all registered providers (verifies SHA-256 + SPI v2):
./omlx-extend/scripts/sync_microkernels.sh --provider io.ninfer.metal
./omlx-extend/scripts/sync_microkernels.sh --all

# Verify without mutating installed providers, or atomically roll back to the previous version:
./omlx-extend/scripts/sync_microkernels.sh --verify-only
./omlx-extend/scripts/sync_microkernels.sh --rollback io.ninfer.metal
```

## 5. Dual-Mode Core: Standalone Server vs oMLX Sidecar

Both modes share the same pure C++20/ObjC++/Metal 4 `libomlx_m5_core_runtime.a` runtime:

- **oMLX Sidecar Mode (`mseries_backendd --socket /tmp/omlx-m5rt-<uid>/m5-<id>.sock`)**: Managed automatically by `oMLX-extend` over AF_UNIX NDJSON RPC (`protocol_version: 2`, backward-compatible with v1), forwarding `lora_adapter` + `lora_mode` (`merge` | `bypass`) in `load` RPCs. oMLX handles tokenization, chat templates, tool calling, auth, dashboard, and multi-model coexistence.
- **Standalone Mode (`ninfer-metal-serve --config ninfer-metal.toml` / `start_ninfer_serve.sh` / `m5rt_runner`)**: Direct native inference configured by `[server]`, `[model]`, `[[models]]`, `[engine]`, and `[speculative]` in `ninfer-metal.toml` or CLI flags (`--lora <adapter_dir>` / `--lora-adapter <adapter_dir>`, `--lora-mode <merge|bypass>`), using a separate default port (`8080`, disjoint from oMLX's `8000`).

## 6. Monorepo Layout & Upstream Direct-Sync Workflow

`omlx/` is tracked directly inside this monorepo (no external fork or git submodule required). Upstream provenance is recorded in [`UPSTREAM_VERSION`](UPSTREAM_VERSION) (`UPSTREAM_REPO`, `UPSTREAM_TAG`, `UPSTREAM_COMMIT`), and the thin Provider SPI diff against upstream is maintained in [`patches/`](patches/).

```bash
# Sync a new upstream jundot/omlx release tag vX.Y.Z directly into monorepo omlx/,
# apply omlx-extend/patches/, update omlx-extend/UPSTREAM_VERSION, and run pytest:
./omlx-extend/scripts/sync_omlx.sh --tag v0.8.3

# After editing omlx/ directly in the monorepo, regenerate the upstream SPI patch:
./omlx-extend/scripts/sync_omlx.sh --export-patch

# Build the complete drag-to-install macOS oMLX-extend .dmg (oMLX.app + embedded Python/MLX + custom kernels):
./omlx-extend/scripts/build_omlx_extend_dmg.sh --extend-tag omlx-extend-v0.8.3 --upstream-tag v0.8.3
```

On 6-hour scheduled CI runs ([`.github/workflows/sync-omlx-extend.yml`](../.github/workflows/sync-omlx-extend.yml)), clean upstream `jundot/omlx` releases are automatically 3-way merged into `omlx/` (carrying forward every prior `oMLX-extend` and `-mix1..9` update, plus deterministic 3-way JSON merge for `Localizable.xcstrings`), re-exported to a binary-safe SPI patch, verified with `pytest`, packaged into an installable macOS **`oMLX-extend-vX.Y.Z-macos26-27.dmg`** (+ `.sha256`), pushed to `main` (or opened as a `chore/omlx-sync-vX.Y.Z` PR if branch-protected), and published immediately as **`omlx-extend-vX.Y.Z`**. When you ship extend-only updates on the same upstream version, run the workflow with `release_mode: mix` to refresh the SPI patch and publish **`omlx-extend-vX.Y.Z-mix1`**..**`mix9`** (with its `.dmg` + `.sha256`) without overwriting prior tags; any upstream merge conflict opens a `chore/omlx-sync-conflict-vX.Y.Z` PR without touching `main`. See [`.github/workflows/release-ninfer-metal.yml`](../.github/workflows/release-ninfer-metal.yml) for the parallel `ninfer-metal` core release pipeline.
