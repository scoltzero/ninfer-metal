**English** | [简体中文](../i18n/zh-CN/miv-v0.01-changelog.md) | [繁體中文](../i18n/zh-TW/miv-v0.01-changelog.md) | [日本語](../i18n/ja/miv-v0.01-changelog.md) | [한국어](../i18n/ko/miv-v0.01-changelog.md) | [Español](../i18n/es/miv-v0.01-changelog.md) | [Français](../i18n/fr/miv-v0.01-changelog.md) | [Deutsch](../i18n/de/miv-v0.01-changelog.md)

# `ninfer-metal` `miv-v0.01` Release Notes

> **Release**: `ninfer-metal` `miv-v0.01` (`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`)
> **Companion Host**: `oMLX-extend` `omlx-extend-v0.8.3` (Provider SPI v2 extension over `jundot/omlx` `v0.8.3`)
> **Platform**: Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`), macOS 27+ (no Xcode needed at runtime; Xcode 27+ to build from source)

`miv-v0.01` is the first public release of `ninfer-metal`, a pure C++20, Objective-C++, and Metal 4 inference engine built on Apple M5+ Neural Accelerators (NAX) and Compute ICB with zero Python or MLX runtime dependencies. It runs **Qwen3.5 / Qwen3.6 / Qwen3.8 Hybrid** models (Dense 0.8B–27B and 256×8 MoE such as 35B-A3B) either standalone via the native OpenAI HTTP/SSE server (`ninfer-metal-serve`) or inside `oMLX-extend` as **`miv Core Backend`** for OpenAI, Anthropic, and macOS GUI support alongside Stock MLX. See [Getting Started & Config](../getting-started.md) for setup and CLI flags, and [`oMLX-extend` Visual Guide](../omlx-extend-guide.md) for the macOS app walkthrough.

## 1. Benchmarks on Apple M5 Ultra

All numbers below come from physical Apple M5 Ultra machines running isolated subprocesses with identical model weights, identical token sequences, `temperature=0`, and `phys_footprint` memory accounting against stock `oMLX 0.8.0` (`mlx 0.32.4`, NAX enabled). Results for the 80-GPU / 256 GiB SKU (`macOS 27.2.1` Build `26B131`, `Xcode 27.2` Build `27B52`) and the 64-GPU / 96 GiB SKU are listed separately.

### Single-Stream Decode & `0-ULP` Speculative Speedup

![Single-stream decode and 0-ULP speculative speedup](../assets/charts/decode-speedup.svg)

- Pure AR on the 80-GPU M5 Ultra leads stock oMLX across every tested weight format: `64.63 tok/s` (`+14.5%`) on `Qwen3.8-27B MXFP4`, `54.03 tok/s` (`+7.3%`) on `27B oQ4e`, and `216.44 tok/s` (`+23.7%`) on `Qwen3.6-35B-A3B MXFP4`.
- Speculative decoding lifts `27B` coding streams to `186.74 tok/s` with `DFlash2` (`2.89×` AR) and `140.48 tok/s` with `Native MTP` (`+36.0%` over stock oMLX MTP at `103.28 tok/s`), holding `128/128` `0-ULP` greedy parity.
- On the 64-GPU / 96 GiB SKU, `27B oQ4e` reaches `138.92 tok/s` (`DFlash2`, `+33.9%`) and `107.52 tok/s` (`MTP`, `+21.2%`) on Coding, plus `110.01` / `100.28 tok/s` on Reasoning (`6/6` sets, `256/256` `0-ULP`).

### Multi-User Concurrency Scaling

![Multi-user concurrency scaling aggregate throughput](../assets/charts/concurrency-scaling.svg)

- Across all 19 concurrency sweeps on the 80-GPU SKU, `miv` beats stock oMLX at every point: at `B=4`, `27B` hits `500.57 tok/s` (`oQ4e DFlash2`), `496.60 tok/s` (`MXFP4 DFlash2`), and `434.92 tok/s` (`oQ4e MTP`), running `2.93×–3.83×` faster than stock oMLX (`148.37` MTP / `129.80` AR).
- `Qwen3.6-35B-A3B` scales to `846.29 tok/s` window (`794.39 tok/s` end-to-end) at `B=8` pure AR, `+52.6%` over stock oMLX with `4.16×` faster TTFT.
- Every concurrent slot holds `128/128 × 4` `0-ULP` greedy parity against single-stream execution.

### 64K Long-Context Stability, UMA Footprint & Prefill

![64K long-context stability and affine4_g128 UMA footprint](../assets/charts/long-context-curve.svg)

- From `2K` to `64K`, `27B MXFP4` AR speed with `affine4_g128` KV drops only `-4.1%` (`61.42 → 58.92 tok/s`, `+28.6%` over oMLX), while real English+Code `DFlash2` sustains `139.70 tok/s` at `64K` (`0-ULP` across `2K–64K`).
- `affine4_g128` cuts `64K` KV cache by `71.8%` (`4.00 → 1.13 GiB`) and holds clean peak UMA to `15.69 GiB` (`-3.44 GiB` vs oMLX; `-10.06 GiB` with all four prefix depths cached). Warm `64K + 32` TTFT drops to `176.19 ms` (`bf16`) and `201.42 ms` (`affine4`), `7.05×–9.48×` faster than oMLX.
- Prefill reaches `2,058.5–2,110.6 tok/s` (`100.25–102.80 TF`) at `1K` and `41.02–41.84 s` cold at `64K` on the 80-GPU SKU, and `10,048.5 tok/s` (`35B-A3B`, `+15.9%`) at `4K` on the 64-GPU SKU (`15/15` wins).

### Multi-Turn Chat & Rolling Prefix Reuse

![Multi-turn short chat rolling prefix reuse and UMA flatness](../assets/charts/multiturn-rolling.svg)

- Rolling UMA prefix reuse pushes multi-turn hit rates above `92%` (`92.33%` at `B=1`, `92.02%` at `B=4`), halving re-prefilled tokens and speeding up warm TTFT by `2.44×–2.94×` (`101.25 ms` at `B=1`, `282.80 ms` at `B=4`).
- In-place slab reuse holds 10-turn `B=1` UMA growth to `+0.02 MB` with zero evictions (vs `+397.5 MB` on oMLX `affine4` and `+704.8 MB` on `bf16`), cutting 10-turn wall time from `16.99 s` to `9.26 s` (`DFlash2`).
- At `B=4` (4 users × 6 turns), batched `affine4` NAX decode reaches `176.95 tok/s` (`AR`) and `206.35 tok/s` (`DFlash2`, `1.72×` oMLX), reducing peak UMA by `2.12 GiB` with `0-ULP` verified on `10/10` (`B=1`) and `24/24` (`B=4`) turns.

## 2. What Ships in `miv-v0.01`

### Standalone OpenAI Server (`ninfer-metal-serve`)
- Single C++20 HTTP/SSE binary serving `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, and Prometheus `/metrics` with sync JSON, SSE streaming (`usage`, `[DONE]`), optional Bearer auth (`--api-key`), CORS, and clean shutdown.
- Built-in C++ BPE tokenizer (`tokenizer.json`), Jinja chat template rendering, `<think>` `reasoning_content` splitting, and Qwen XML/JSON `<tool_call>` parsing for sync and streaming responses.
- Hosts in-process `miv` alongside external Path B Provider SPI v2 sidecars on one port (`--providers-dir`, `[[models]]`) under a shared UMA memory cap (`81%` RAM default) and contention-aware scheduling.
- Ships a ready-to-run [`ninfer-metal.toml`](../../omlx-extend/config/ninfer-metal.toml) (auto-discovered and shared with `oMLX-extend`) paired with the one-command launcher `./omlx-extend/scripts/start_ninfer_serve.sh`.

### Metal 4 NAX Engine (`miv/`)
- Self-contained C++20 / Objective-C++ / Metal 4 tree with 38 precompiled `.metallib` shaders covering NAX prefill, fused decode, 256×8 MoE routing, speculative verify, and 4-bit Split-KV, built without MLX or Python headers.
- Dedicated hardware calibration profiles for both Apple M5 Ultra SKUs ([80-GPU / 256 GiB](../backend/silicon-calibrations/apple-m5-ultra-80gpu.json) and [64-GPU / 96 GiB](../backend/silicon-calibrations/apple-m5-ultra-64gpu.json)), tuning split-K, Split-KV partitions, and draft depth with environment variables taking precedence.
- Zero-copy GPU prefix continuation (`0.73–1.62 ms` restore at `64K`), in-place rolling UMA prefix reuse, and batched `affine4_g128` NAX attention for concurrent sessions.

### `oMLX-extend` `omlx-extend-v0.8.3` (macOS App & Sidecar)
- Registers **`miv Core Backend`** over an `AF_UNIX` sidecar (`mseries_backendd`): adapted `.omlxm5` models route automatically in O(1), while standard MLX models stay on Stock MLX without spawning extra processes.
- Native macOS UI adds `Integrations → External Engines` cards for `miv Core Backend` and Path B microkernels, per-model engine badges, an Auto / Stock MLX picker, and capability-driven controls for `64K+` context, `MTP`, `DFlash2`, and `affine4_g128` KV cache.
- Coordinates UMA budgets, idle cache trimming, and contention-aware chunked prefill between sidecar and MLX models, with atomic symlink updates under `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/`.

## 3. Clean-Mac Packaging & Verification

- Release archive `ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` packages 16 Mach-O binaries (including `ninfer-metal-serve`, `mseries_backendd`, and CLI/calibration tools), 38 Metal 4 `.metallib` shaders, 2 SKU calibration profiles, `provider.json`, and SHA-256 checksums.
- `otool -L` across all 16 binaries confirms linking only to Apple system frameworks (`CoreFoundation`, `Foundation`, `IOKit`, `Metal`) and system dylibs (`libc++.1`, `libSystem.B`, `libobjc.A`), with zero Homebrew, Xcode, MLX, OpenSSL, or Python runtime dependencies.
- End-to-end server tests (sync/SSE, `<think>`/`<tool_call>`, `B=4` batching, and parallel in-process + sidecar hosting), `638 / 638` `oMLX-extend` pytest tests, and `75 / 75` macOS XCTests pass, and the Provider SPI patch applies cleanly to `jundot/omlx` `v0.8.3`.

## 4. Supported Scope

| Area | `miv-v0.01` Support |
|---|---|
| Hardware | Apple Silicon `MTLGPUFamilyApple10+` (Apple M5 / M5 Pro / M5 Max / M5 Ultra, 64-GPU & 80-GPU) |
| OS & Toolchain | macOS 27+ (no Xcode needed at runtime; Xcode 27+ / macOS 27 SDK to build from source) |
| Models | **Qwen3.5 / Qwen3.6 / Qwen3.8 Hybrid** (3:1 GDN + GQA; Dense 0.8B–27B and 256×8 MoE, e.g. 35B-A3B) |
| Weight Formats | Offline NAX pre-tiled `.omlxm5`: `MXFP4` (`E2M1 × UE8M0`) and `oQ` Affine (`q3–q8`, `g64 / g128`) |
| KV Cache | `bf16` and `affine4_g128` (`0.5625 B/val`, `0-ULP` verified across `2K–64K`) |
| Speculative Decoding | Pure AR (`none`), embedded `Native MTP`, and `DFlash2` (single-stream and `B=2..4` batched) |
| Serving Interfaces | OpenAI HTTP/SSE endpoints in `ninfer-metal-serve`; OpenAI + Anthropic endpoints and macOS GUI in `oMLX-extend` |
