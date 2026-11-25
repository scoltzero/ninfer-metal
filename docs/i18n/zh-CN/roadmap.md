[English](../../roadmap/README.md) | **简体中文** | [繁體中文](../zh-TW/roadmap.md) | [日本語](../ja/roadmap.md) | [한국어](../ko/roadmap.md) | [Español](../es/roadmap.md) | [Français](../fr/roadmap.md) | [Deutsch](../de/roadmap.md)

# `ninfer-metal` 路线图

已发布基线：**`ninfer-metal` `miv-v0.01`**（含原生 C++20 `ninfer-metal-serve`）+ **`oMLX-extend` `omlx-extend-v0.8.3`**（详见 [发布说明](miv-v0.01-changelog.md)）。

`miv-v0.01` 之后的工程计划分三条主线推进：

---

## 1. `v0.02+` 智能体内存虚拟化（`KVMem-UMA + ActKV`）

完整架构设计、Apple M5 UMA 内存物理模型与验证门禁 V1–V6 见 **[`v0.02-kvmem-actkv-plan.md`](../../roadmap/v0.02-kvmem-actkv-plan.md)**。

- **第一阶段：`KVMem-UMA + ActKV-Lite`（UMA 内 `256K → 1M+` 工作区）**
  - 按需开启（`kv_policy = "kvmem_actkv"`，永久保留 `"lossless"` 作为 `0-ULP` 默认策略）。
  - 在 `MTLStorageModeShared` 中存放非对称冷归档（FP8/BF16 Pre-RoPE K + `affine4_g128` V，避免 4-bit 二次量化；`1M` Token 的 `Qwen3.8-27B` 工作区总占用 `25.03 GiB`）。
  - 独立 `MeanKIndexBuffer`（扫描 `< 0.6 ms`）+ Action-LRFU + 置信度驱动工作集 `W ∈ [16K, 64K]` + 单内核 `re-RoPE + affine4_pack` 物化（`64K` 下 `~1.9–4.4 ms`），热路径 `attention_affine4.metal` 与静态 ICB 保持零改动。
- **第二阶段：Apple NVMe 持久化与完整 Token 级 ActKV**
  - 基于 `F_NOCACHE` + `dispatch_io` 的 `.omlxkv` 工作区归档，面向 32–64 GiB M5 Pro/Max 机型，支持 `< 1 s` 跨重启恢复。
  - Base-2 LSE 注意力恢复（`+96 B/tok`）与零反量化原地 `affine4_g128` 页紧缩（`hd=256` 下 `144 B/tok`，`hd=128` 下 `80 B/tok`），面向纯注意力 32B/70B 与 MoE 模型。

---

## 2. 共享底座（`libninfer_metal_substrate`）与新模型家族微内核

- 将 `miv/` 中的 Metal 4 运行时、`< 2 GiB` Compute ICB 编码器、`KvPagePool`、`SharedPrefixCache`、`UmaCheckpointPool` 与通用 NAX 原语抽离为 `libninfer_metal_substrate`。
- 通过 **[微内核工坊](microkernel-foundry.md)**，按 **[核心认证标准](../../standards/core-certification-standard.md)** 接入纯注意力 GQA（`Qwen3-32B`、`Llama-3.3-70B`）、DeepSeek MLA、Gemma、GLM 与 MiniMax 微内核。

---

## 3. `ninfer-metal-serve` 二期（Anthropic 接口与语法受限解码）

在 `miv-v0.01` 原生 C++20 OpenAI HTTP/SSE 服务基础上扩展：

- **Anthropic Messages API（`POST /v1/messages`）**：原生实现 Anthropic 请求/响应转换与 SSE 事件流，适配 Claude 兼容客户端与编程 Agent。
- **语法受限解码（`xgrammar`）**：为 `omlx::m5::Engine` 增加逐拍 logit 掩码回调并集成 `xgrammar`，在不引入外部运行时依赖的前提下提供严格 JSON Schema（`response_format`）与结构化工具调用语法约束。
