**English** | [简体中文](../i18n/zh-CN/roadmap.md) | [繁體中文](../i18n/zh-TW/roadmap.md) | [日本語](../i18n/ja/roadmap.md) | [한국어](../i18n/ko/roadmap.md) | [Español](../i18n/es/roadmap.md) | [Français](../i18n/fr/roadmap.md) | [Deutsch](../i18n/de/roadmap.md)

# `ninfer-metal` Roadmap

Shipped baseline: **`ninfer-metal` `miv-v0.01`** (with native C++20 `ninfer-metal-serve`) + **`oMLX-extend` `omlx-extend-v0.8.3`** (see [Release Notes](../releases/miv-v0.01-changelog.md)).

Planned engineering work beyond `miv-v0.01` runs across four tracks:

---

## 1. `v0.02+` Agent Memory Virtualization (`KVMem-UMA + ActKV`)

Full architecture spec, Apple M5 UMA memory physics, and Verification Gates V1–V6 live in **[`v0.02-kvmem-actkv-plan.md`](v0.02-kvmem-actkv-plan.md)**.

- **Tier-1: `KVMem-UMA + ActKV-Lite` (`256K → 1M+` workspaces in UMA)**
  - Opt-in (`kv_policy = "kvmem_actkv"`, keeping `"lossless"` as the permanent `0-ULP` default).
  - Asymmetric cold archive in `MTLStorageModeShared` (FP8/BF16 Pre-RoPE K + `affine4_g128` V — zero 4-bit double quantization; `25.03 GiB` total for a `1M`-token `Qwen3.8-27B` workspace).
  - Standalone `MeanKIndexBuffer` (`< 0.6 ms` scan) + Action-LRFU + confidence-driven working set `W ∈ [16K, 64K]` + single-kernel `re-RoPE + affine4_pack` materialization (`~1.9–4.4 ms` at `64K`), leaving hot-path `attention_affine4.metal` and static ICBs unchanged.
- **Tier-2: Apple NVMe Persistence & Full Token-Level ActKV**
  - `.omlxkv` workspace bundles over `F_NOCACHE` + `dispatch_io` for 32–64 GiB M5 Pro/Max machines and `< 1 s` cross-restart resume.
  - Base-2 LSE attention recovery (`+96 B/tok`) and zero-dequant in-place `affine4_g128` page compaction (`144 B/tok` @ `hd=256`, `80 B/tok` @ `hd=128`) targeting pure-attention 32B/70B and MoE models.

---

## 2. `v0.02+` Native `MXFP8` (`E4M3 × UE8M0`) & Per-Layer Mixed Precision

Full container contract, Metal 4 NAX operator stack, mixed-precision rules, and Acceptance Gates N1–N4 & P1–P4 live in **[`v0.02-mxfp8-native-plan.md`](v0.02-mxfp8-native-plan.md)**.

- **Container Format & Converter Contract (`.miv` v1)**
  - New untiled format `mxfp8_e4m3_g32_e8m0_v1` (`1.03125 B/param`, `activation_policy = "AllowA8"`) and single-copy offline NAX pre-tiled layout `mx_nax_tiled_fp8_bk128_v1` (`nax_data` `[padded_n/bn, K/128, bn, 128]` + `nax_scale` `[padded_n/bn, K/128, bn, 4]` with `bn = 16/32`, `bk = 128`, and packed `uint32` `UE8M0` scales).
  - Lifts `miv/tools/m5_artifact.py`'s hardcoded 4-bit `logical_k // 8` assumption so MLX `U32`-packed (`logical_k // 4`) and HuggingFace `F8_E4M3`/`U8` (`logical_k`) MXFP8 checkpoints convert cleanly with `0-ULP` roundtrip parity.
- **Metal 4 NAX Operator Stack & Per-Layer Mixed Precision**
  - `mxfp8_nax.metal` (both W8A16 high-fidelity MPP/NAX and zero-TG-SRAM W8A8 dynamic-FP8-activation prefill paths), zero-copy `Mxfp8NaxCache`, and zero-`uitofp` `T=1` narrow-vector `pretiled_qmv_mxfp8_*_t1` kernels wired across `qwen_forward`, `mtp_forward`, and `dflash_forward`.
  - Projection-level coexistence of `MXFP8` + `MXFP4` + `oQ` (`affine3`–`affine8`) in one `.miv` container (preserving closed-loop `f16` across `MXFP4 + MXFP8` blocks and gracefully falling back on heterogeneous sibling projections), while permanently rejecting uncompressed `BF16` backbone linear mixing (`2.0 B/param` UMA decode wall and `> 2 GiB` Compute ICB segment overflow).

---

## 3. Shared Substrate (`libninfer_metal_substrate`) & New Model-Family Microkernels

- Extract the Metal 4 runtime, `< 2 GiB` Compute ICB encoder, `KvPagePool`, `SharedPrefixCache`, `UmaCheckpointPool`, and generic NAX primitives from `miv/` into `libninfer_metal_substrate`.
- Add Pure-Attention GQA (`Qwen3-32B`, `Llama-3.3-70B`), DeepSeek MLA, Gemma, GLM, and MiniMax microkernels through the **[Microkernel Foundry](../microkernel-foundry.md)** under the **[Core Certification Standard](../standards/core-certification-standard.md)**.

---

## 4. `ninfer-metal-serve` Phase 2 (Anthropic API & Grammar-Constrained Decoding)

Building on the native C++20 OpenAI HTTP/SSE server shipped in `miv-v0.01`:

- **Anthropic Messages API (`POST /v1/messages`)**: Native Anthropic request/response translation and SSE event streaming for Claude-compatible clients and coding agents.
- **Grammar-constrained decoding (`xgrammar`)**: Add a per-step logit-mask callback to `omlx::m5::Engine` and integrate `xgrammar` for strict JSON Schema (`response_format`) and structured tool-call grammar enforcement without external runtime dependencies.
