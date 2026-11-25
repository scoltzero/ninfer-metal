**English** | [简体中文](../i18n/zh-CN/roadmap.md) | [繁體中文](../i18n/zh-TW/roadmap.md) | [日本語](../i18n/ja/roadmap.md) | [한국어](../i18n/ko/roadmap.md) | [Español](../i18n/es/roadmap.md) | [Français](../i18n/fr/roadmap.md) | [Deutsch](../i18n/de/roadmap.md)

# `ninfer-metal` Roadmap

Shipped baseline: **`ninfer-metal` `miv-v0.01`** (with native C++20 `ninfer-metal-serve`) + **`oMLX-extend` `omlx-extend-v0.8.3`** (see [Release Notes](../releases/miv-v0.01-changelog.md)).

Planned engineering work beyond `miv-v0.01` runs across three tracks:

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

## 2. Shared Substrate (`libninfer_metal_substrate`) & New Model-Family Microkernels

- Extract the Metal 4 runtime, `< 2 GiB` Compute ICB encoder, `KvPagePool`, `SharedPrefixCache`, `UmaCheckpointPool`, and generic NAX primitives from `miv/` into `libninfer_metal_substrate`.
- Add Pure-Attention GQA (`Qwen3-32B`, `Llama-3.3-70B`), DeepSeek MLA, Gemma, GLM, and MiniMax microkernels through the **[Microkernel Foundry](../microkernel-foundry.md)** under the **[Core Certification Standard](../standards/core-certification-standard.md)**.

---

## 3. `ninfer-metal-serve` Phase 2 (Anthropic API & Grammar-Constrained Decoding)

Building on the native C++20 OpenAI HTTP/SSE server shipped in `miv-v0.01`:

- **Anthropic Messages API (`POST /v1/messages`)**: Native Anthropic request/response translation and SSE event streaming for Claude-compatible clients and coding agents.
- **Grammar-constrained decoding (`xgrammar`)**: Add a per-step logit-mask callback to `omlx::m5::Engine` and integrate `xgrammar` for strict JSON Schema (`response_format`) and structured tool-call grammar enforcement without external runtime dependencies.
