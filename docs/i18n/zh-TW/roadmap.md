[English](../../roadmap/README.md) | [简体中文](../zh-CN/roadmap.md) | **繁體中文** | [日本語](../ja/roadmap.md) | [한국어](../ko/roadmap.md) | [Español](../es/roadmap.md) | [Français](../fr/roadmap.md) | [Deutsch](../de/roadmap.md)

# `ninfer-metal` 開發路線圖

已發行基線：**`ninfer-metal` `miv-v0.01`**（含原生 C++20 `ninfer-metal-serve`）+ **`oMLX-extend` `omlx-extend-v0.8.3`**（見 [發行說明](miv-v0.01-changelog.md)）。

`miv-v0.01` 之後的工程規劃分三條主軸推進：

---

## 1. `v0.02+` Agent 記憶體虛擬化（`KVMem-UMA + ActKV`）

完整架構規格、Apple M5 UMA 記憶體物理模型與驗證關卡 V1–V6 見 **[`v0.02-kvmem-actkv-plan.md`](../../roadmap/v0.02-kvmem-actkv-plan.md)**。

- **第一階段：`KVMem-UMA + ActKV-Lite`（UMA 內 `256K → 1M+` 工作區）**
  - 選擇性啟用（`kv_policy = "kvmem_actkv"`，永久保留 `"lossless"` 作為 `0-ULP` 預設策略）。
  - 於 `MTLStorageModeShared` 存放非對稱冷歸檔（FP8/BF16 Pre-RoPE K + `affine4_g128` V，零 4-bit 二次量化；`1M`-token `Qwen3.8-27B` 工作區總佔用 `25.03 GiB`）。
  - 獨立 `MeanKIndexBuffer`（掃描 `< 0.6 ms`）+ Action-LRFU + 信心驅動工作集 `W ∈ [16K, 64K]` + 單核心 `re-RoPE + affine4_pack` 實體化（`64K` 下 `~1.9–4.4 ms`），熱路徑 `attention_affine4.metal` 與靜態 ICB 維持原樣不變。
- **第二階段：Apple NVMe 持久化與完整 Token 級 ActKV**
  - 透過 `F_NOCACHE` + `dispatch_io` 存取 `.omlxkv` 工作區封裝，針對 32–64 GiB M5 Pro/Max 機型提供 `< 1 s` 跨重啟恢復。
  - Base-2 LSE 注意力還原（`+96 B/tok`）與零反量化原地 `affine4_g128` 分頁緊縮（`hd=256` 下 `144 B/tok`，`hd=128` 下 `80 B/tok`），鎖定純注意力 32B/70B 與 MoE 模型。

---

## 2. 共享基底（`libninfer_metal_substrate`）與新模型家族微核心

- 從 `miv/` 抽離 Metal 4 執行期、`< 2 GiB` Compute ICB 編碼器、`KvPagePool`、`SharedPrefixCache`、`UmaCheckpointPool` 與通用 NAX 原語，獨立為 `libninfer_metal_substrate`。
- 透過 **[微核心工坊](microkernel-foundry.md)** 並依 **[核心認證標準](../../standards/core-certification-standard.md)**，新增純注意力 GQA（`Qwen3-32B`、`Llama-3.3-70B`）、DeepSeek MLA、Gemma、GLM 與 MiniMax 微核心。

---

## 3. `ninfer-metal-serve` 第二階段（Anthropic API 與語法受限解碼）

延續 `miv-v0.01` 已發行的原生 C++20 OpenAI HTTP/SSE 伺服器：

- **Anthropic Messages API（`POST /v1/messages`）**：原生實作 Anthropic 請求／回應轉換與 SSE 事件串流，支援 Claude 相容用戶端與程式開發 Agent。
- **語法受限解碼（`xgrammar`）**：為 `omlx::m5::Engine` 新增逐拍 logit 遮罩回呼並整合 `xgrammar`，在不引入外部執行期相依的前提下，強制執行嚴格 JSON Schema（`response_format`）與結構化工具呼叫語法。
