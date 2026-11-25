# oMLX-extend — Thin External Provider SPI & `miv Core Backend (ninfer-metal)` Adapter

`oMLX-extend` is the upstream-synced extension layer over [`jundot/omlx`](https://github.com/jundot/omlx) (tagged `omlx-extend-vX.Y.Z` / `omlx-extend-vX.Y.Z-mix1..9`, currently `omlx-extend-v0.8.3`) that integrates the pure-Metal-4 **`miv Core Backend (ninfer-metal)`** inference core (first codename **`miv`**, `miv-v0.01`, `semver: 0.0.1`, `core_abi: 1`, `protocol_version: 2`) while keeping stock oMLX **100% untouched** for unadapted models and keeping the oMLX UI minimal and clean.

## 1. Architecture: Thin Provider SPI + Zero Stock Regression

- **Unadapted models -> 100% stock oMLX**: `detect_adapted(model_dir)` runs a fast O(1) filesystem check (`model.omlxm5`, `model_pack.json`, `omlxm5-manifest.json`, zero subprocess spawn). Models without an adapted `.omlxm5` artifact return `None` and run unmodified through stock oMLX (`BatchedEngine` / `VLMBatchedEngine`) with zero sidecar overhead.
- **Adapted models -> automatic lazy-load routing**: Models carrying `.omlxm5` artifacts auto-route to `miv Core Backend (ninfer-metal)` (`m5rt`) when the provider binary is installed, spawning the out-of-process `mseries_backendd` UDS sidecar lazily on first request (or when pinned). Setting `external_engine: "stock"` opts any adapted model back into stock in-process MLX.
- **Simultaneous coexistence (sidecar + stock MLX)**:
  - **Unified UMA budget**: `load` RPC forwards `memory_budget_bytes` derived from oMLX's `ProcessMemoryEnforcer` ceiling minus active resident models, preventing jetsam OOM when coexisting with stock MLX models.
  - **Soft/hard pressure cache trim**: Before evicting a coexisting model, `ProcessMemoryEnforcer` invokes `trim_cache(target_free_bytes)` to drop unpinned prefix-cache entries and checkpoint slabs on the sidecar.
  - **GPU contention hints**: `EnginePool` sends `set_scheduler_hint(gpu_contended=true/false)` whenever both an in-process MLX model and an external sidecar model are active concurrently.
  - **Clean teardown**: `stop()` drains the process group and verifies `process_memory_bytes() == 0` before admitting another model.

## 2. Minimal UI + Config-File Control Split

Stock oMLX UI stays clean — **zero extra settings forms are added**:

| Control Category | Where Configured | Mapped `ninfer-metal` (`miv-v0.01`) Knob |
|---|---|---|
| Context Window | Stock oMLX Model Settings (`max_context_window`) | `max_context` (Context `L`, up to `262144`+, no 32K cap) |
| Multi-user Concurrency | Stock oMLX Global Settings (`max_concurrent_requests`) | `max_concurrency` (Concurrency `B`/`C`, clamped `1..8`) |
| Lightning MTP | Stock oMLX Acceleration (`mtp_enabled`, `mtp_fixed_depth`, `mtp_adaptive_max_depth`) | `speculative.backend = "mtp"`, `draft_tokens` (`1..7`) |
| DFlash2 | Stock oMLX Acceleration (`dflash_enabled`, `dflash_draft_model`, `dflash_block_size`) | `speculative.backend = "dflash2"`, `draft_tokens = block_size - 1` |
| KV Cache Quantization | Stock oMLX Memory Saving (`turboquant_kv_enabled`, `turboquant_kv_scheme == "affine4"`) | `kv_dtype = "affine4_g128"` (`0.5625 B/val`); disabled -> `"bf16"` *(note: `affine8`/`int8_g64` is intentionally NOT mapped in `miv-v0.01`)* |
| Advanced / Core-only Knobs | `ninfer-metal.toml` or `model_settings.json` -> `external_engine_options["ninfer-metal"]` | `rolling_prefix_cache`, `prefix_cache_gb`, per-model `max_concurrency` override, `sampling_contract`, `prefill_engine`, `artifact_path`, standalone `[server]` params |

Standard UI sections unlock **per setting** against the active provider's `supported_standard_settings`. Providers such as `Splash` (`supported_standard_settings: []`) keep exact upstream lock behavior.

## 3. Config File Discovery (`ninfer-metal.toml`)

Copy [`config/ninfer-metal.toml.example`](config/ninfer-metal.toml.example) to one of these locations (lowest to highest precedence, overridden by per-model `external_engine_options` / `m5rt_options`):

1. `~/Library/Application Support/oMLX/ninfer-metal.toml`
2. `~/.config/ninfer-metal/ninfer-metal.toml`
3. `<model_dir>/ninfer-metal.toml`
4. Path pointed to by `$NINFER_METAL_CONFIG`

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

- **oMLX Sidecar Mode (`mseries_backendd --socket /tmp/omlx-m5rt-<uid>/m5-<id>.sock`)**: Managed automatically by `oMLX-extend` over AF_UNIX NDJSON RPC (`protocol_version: 2`, backward-compatible with v1). oMLX handles tokenization, chat templates, tool calling, auth, dashboard, and multi-model coexistence.
- **Standalone Mode (`ninfer-metal serve --config ninfer-metal.toml` / `m5rt_runner`)**: Direct native inference configured by `[server]`, `[model]`, `[engine]`, and `[speculative]` in `ninfer-metal.toml`, using a separate default port (`8080`, disjoint from oMLX's `8000`).

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
