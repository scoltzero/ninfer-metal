**English** | [简体中文](../i18n/zh-CN/miv-v0.02-changelog.md)

# `ninfer-metal` `miv-v0.02` Release Notes

> **Release**: `ninfer-metal` `miv-v0.02` (`codename: miv` · `semver: 0.0.2` · `core_abi: 1` · `protocol_version: 2`)
> **Companion Host**: `oMLX-extend` `omlx-extend-v0.8.3-mix4` (Provider SPI v2 extension over `jundot/omlx` `v0.8.3`)
> **Target Platform**: Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`), macOS 27+

`miv-v0.02` consolidates all post-`miv-v0.01` microkernel upgrades (`mix1`–`mix4`) and eliminates long-context, multi-turn, tool-calling, and `<think>`-mode performance bottlenecks out of the box on Apple10 GPUs without requiring manual `path_overrides`.

---

## 1. Highlights in `miv-v0.02`

### Out-of-the-Box Apple10 Long-Context & Prefill Defaults
- **Default Split-KV Attention (`attention_split_kv`)**: Automatically activates 2-SG NAX Split-KV (`s512`) on real models (`hidden_size >= 1024`) on Apple9+/Apple10 GPUs, slashing 20K–34K `DFlash2` target verify GPU latency by **`4.0×`** (`93.45 ms/round` → **`23.3–24.7 ms/round`**) with flat step latency from `100` to `34K+` tokens.
- **Default NAX Flash-Prefill under `position_coupled_v1`**: Unlocks `omlx_m5_paged_flash_prefill_affine4_g128_hd256` (`61.0 TFLOP/s`) and `omlx_m5_paged_flash_prefill_nax_hd256` (`69.7 TFLOP/s`) by default on Apple10 real models (`hidden_size >= 1024`), cutting ~20K cold TTFT nearly in half (`18.5–19.3 s` → **`9.65 s`**) while preserving `0-ULP` parity.
- **Pure-AR `t_domain_dispatch` Default**: Enables `t_domain_dispatch` out of the box for non-speculative (`!speculative_enabled`) Apple10 AR sessions while keeping speculative `T=1 == T=8` bit-exact routes untouched.

### DFlash2 Speculative Decoding & Multi-Turn / Tool-Call Optimization
- **8-SIMDgroup Cooperative `omlx_m5_draft_attention`**: Upgrades `DFlash2` sliding-window draft attention (`head_dim` 128/256, `total_keys > 64`) from single-SIMDgroup serial scan to 8-SIMDgroup (`256`-thread) chunked online-softmax reduction with zero register spills, cutting `>= 2K` draft GPU time by **`2.1×`** (`13.52–14.20 ms/round` → **`6.54–6.63 ms/round`**).
- **Same-Slot DFlash2 KV Frontier Reuse (`retain_frontier`)**: Retains and extends the draft model's ring KV cache across same-slot rolling prefix cache hits (such as multi-step agentic `<tool_call>` loops) instead of resetting and re-projecting `2,048` context rows on every turn.
- **Deterministic Dense `K = max_k` Speculative Controller**: Keeps single-slot (`B = 1`) dense models at deterministic full draft depth in `SpecSearchMode::Auto` and isolates Round-1 cold context projection from `ema_draft_base_ms_`, preventing false `K = 0` gating backoffs (`spec_search_gated_rounds = 0`).

### `<think>`-Aware Multi-Turn Rolling UMA Prefix Reuse
- **Zero-Readback Pre-Decode Thinking Anchor**: When `rolling_prefix_cache` is enabled on `<think>`-capable models (Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8), `miv` records a lightweight `include_anchor_logits = false` UMA checkpoint at the prompt boundary alongside the post-decode checkpoint (preserving 1-CB fused GPU sampling and `0` host logit readback).
- **Automatic Stripped-`<think>` Prefix Recovery**: When chat templates strip prior `<think>...</think>` blocks on subsequent user turns, `SharedPrefixCache` seamlessly restores the pre-decode prompt checkpoint (cutting 20K `<think>` Turn 2+ TTFT from `~9.7 s` to **`~0.25–0.30 s`**) while in-place superseding both prior turn checkpoints so UMA slab usage stays bounded with `0` evictions.

### `.miv` Format, `affine8` NAX Pre-Tile, and Dual-Mode LoRA (Consolidated from `mix2`–`mix3`)
- **`.miv` Canonical Artifact & Auto-Repack**: Native `.miv` extension (retaining `.omlxm5` compatibility), on-demand companion `DFlash2` / target auto-repack, and expanded Qwen3.5 / 3.6 / 3.8 `DFlash2` draft whitelist.
- **`affine8` (`oq_nax_tiled_g64_v1`) NAX Support**: Zero-copy `AffineNaxCache` pre-tiled execution for 8-bit `oQ` affine weights.
- **Dual-Mode PEFT / `mlx-lm` LoRA (`merge` + `bypass`)**: Live `reload_lora` hot-swap, UMA memory ledger accounting, CLI / `ninfer-metal.toml` / SPI v2 integration, and capability-aware UI controls.

---

## 2. Companion `oMLX-extend` `v0.8.3-mix4` Changes

- **UI KV Compression Toggle Sync (`sync_m5rt_ui_kv_setting`)**: Toggling `turboquant_kv_enabled` in the macOS / Web Model Settings UI now updates and clears `external_engine_options.m5rt.kv_dtype` in lockstep with `m5rt_options.kv_dtype`, ensuring `affine4_g128` activates immediately upon reload.
- **Microkernel Registry & SPI Patch Sync**: Updates `M5RT_DEFAULT_VERSION` and `microkernels.json` baseline metadata to `miv-v0.02` with a 0-byte synced `omlx-v0.8.3-provider-spi.patch`.

---

## 3. Baseline Reference

For foundational M5 Ultra 80-GPU / 64-GPU Dim A–F charts and architecture baselines, see **[`miv-v0.01` Release Notes](miv-v0.01-changelog.md)**.
