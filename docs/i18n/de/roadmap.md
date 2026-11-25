[English](../../roadmap/README.md) | [简体中文](../zh-CN/roadmap.md) | [繁體中文](../zh-TW/roadmap.md) | [日本語](../ja/roadmap.md) | [한국어](../ko/roadmap.md) | [Español](../es/roadmap.md) | [Français](../fr/roadmap.md) | **Deutsch**

# `ninfer-metal` Roadmap

Ausgelieferte Basis: **`ninfer-metal` `miv-v0.01`** (mit nativem C++20-`ninfer-metal-serve`) + **`oMLX-extend` `omlx-extend-v0.8.3`** (siehe [Release Notes](miv-v0.01-changelog.md)).

Die geplanten Entwicklungsarbeiten über `miv-v0.01` hinaus gliedern sich in drei Stränge:

---

## 1. `v0.02+` Agent-Speichervirtualisierung (`KVMem-UMA + ActKV`)

Die vollständige Architekturspezifikation, die Apple-M5-UMA-Speicherphysik und die Verifikations-Gates V1–V6 stehen in **[`v0.02-kvmem-actkv-plan.md`](../../roadmap/v0.02-kvmem-actkv-plan.md)**.

- **Tier-1: `KVMem-UMA + ActKV-Lite` (`256K → 1M+`-Workspaces in UMA)**
  - Opt-in (`kv_policy = "kvmem_actkv"`, während `"lossless"` dauerhaft als `0-ULP`-Standard erhalten bleibt).
  - Asymmetrisches Kaltarchiv in `MTLStorageModeShared` (FP8/BF16 Pre-RoPE K + `affine4_g128` V — keine doppelte 4-Bit-Quantisierung; `25.03 GiB` gesamt für einen `1M`-Token-`Qwen3.8-27B`-Workspace).
  - Eigenständiger `MeanKIndexBuffer` (`< 0.6 ms` Scan) + Action-LRFU + konfidenzgesteuertes Working Set `W ∈ [16K, 64K]` + Single-Kernel-`re-RoPE + affine4_pack`-Materialisierung (`~1.9–4.4 ms` bei `64K`), während der Hot-Path `attention_affine4.metal` und statische ICBs unverändert bleiben.
- **Tier-2: Apple-NVMe-Persistenz & vollständiges Token-Level-ActKV**
  - `.omlxkv`-Workspace-Bundles über `F_NOCACHE` + `dispatch_io` für 32–64-GiB-M5-Pro/Max-Systeme und `< 1 s` Wiederaufnahme nach Neustart.
  - Base-2-LSE-Attention-Wiederherstellung (`+96 B/tok`) und dequantisierungsfreie In-Place-Kompaktierung von `affine4_g128`-Seiten (`144 B/tok` @ `hd=256`, `80 B/tok` @ `hd=128`) für Pure-Attention-32B/70B- und MoE-Modelle.

---

## 2. Gemeinsames Substrat (`libninfer_metal_substrate`) & neue Modellfamilien-Microkernel

- Auslagern der Metal-4-Laufzeit, des `< 2 GiB`-Compute-ICB-Encoders, von `KvPagePool`, `SharedPrefixCache`, `UmaCheckpointPool` und generischen NAX-Primitiven aus `miv/` in `libninfer_metal_substrate`.
- Ergänzen von Pure-Attention-GQA- (`Qwen3-32B`, `Llama-3.3-70B`), DeepSeek-MLA-, Gemma-, GLM- und MiniMax-Microkerneln über die **[Microkernel Foundry](microkernel-foundry.md)** nach dem **[Core-Zertifizierungsstandard](../../standards/core-certification-standard.md)**.

---

## 3. `ninfer-metal-serve` Phase 2 (Anthropic-API & grammatikbeschränktes Dekodieren)

Aufbauend auf dem in `miv-v0.01` ausgelieferten nativen C++20-OpenAI-HTTP/SSE-Server:

- **Anthropic Messages API (`POST /v1/messages`)**: Native Übersetzung von Anthropic-Requests/-Responses und SSE-Event-Streaming für Claude-kompatible Clients und Coding-Agents.
- **Grammatikbeschränktes Dekodieren (`xgrammar`)**: Ergänzen eines Per-Step-Logit-Mask-Callbacks in `omlx::m5::Engine` und Integration von `xgrammar` für strikte JSON-Schema- (`response_format`) und strukturierte Tool-Call-Grammatikdurchsetzung ohne externe Laufzeitabhängigkeiten.
