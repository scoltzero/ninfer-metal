<p align="center"><img src="../../assets/logo.png" width="140" alt="ninfer-metal" /></p>

<h1 align="center">ninfer-metal</h1>

<p align="center">
  <strong>面向 Apple M5+ 的原生 Metal 4 NAX 与 Compute ICB 大模型推理引擎（v1 核心代号 <code>miv</code>）</strong>
</p>

<!-- Replace scoltzero/ninfer-metal with final GitHub slug if changed -->
<p align="center">
  <a href="https://github.com/scoltzero/ninfer-metal/stargazers"><img src="https://img.shields.io/github/stars/scoltzero/ninfer-metal?style=flat-square&logo=github" alt="GitHub Stars" /></a>
  <a href="https://github.com/scoltzero/ninfer-metal/releases"><img src="https://img.shields.io/github/downloads/scoltzero/ninfer-metal/total?style=flat-square" alt="GitHub Downloads" /></a>
  <a href="miv-v0.02-changelog.md"><img src="https://img.shields.io/badge/release-miv--v0.02-0969da?style=flat-square" alt="Latest Release miv-v0.02" /></a>
  <img src="https://img.shields.io/badge/platform-Apple%20M5%2B%20%C2%B7%20macOS%2027%2B-111827?style=flat-square" alt="Platform Apple M5+ macOS 27+" />
  <img src="https://img.shields.io/badge/backend-Metal%204%20NAX-10b981?style=flat-square" alt="Backend Metal 4 NAX" />
  <a href="../../../LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square" alt="License GPL-3.0" /></a>
</p>

<div align="center">

[English](../../../README.md) | **简体中文** | [繁體中文](../zh-TW/README.md) | [日本語](../ja/README.md) | [한국어](../ko/README.md) | [Español](../es/README.md) | [Français](../fr/README.md) | [Deutsch](../de/README.md)

</div>

---

**`ninfer-metal`** 是纯 C++20 / Objective-C++ / Metal 4 推理引擎，面向配备硬件 **Neural Accelerators (NAX)** 与 **Compute ICB** 的 Apple Silicon 芯片（`M5 / M5 Pro / M5 Max / M5 Ultra`，`MTLGPUFamilyApple10+`，`macOS 27+`）。

- **零 Python，零 MLX**：包含 16 个 Mach-O 二进制（含原生 `ninfer-metal-serve` 与 `mseries_backendd`）和 38 个预编译 `.metallib` 着色器，仅链接 macOS 系统框架——在干净的 Mac 上无需 Xcode、Python 或 Homebrew 动态库即可直接运行。
- **独立 OpenAI 服务（`ninfer-metal-serve`）**：单二进制 C++20 服务，在同一 HTTP/SSE 端口并行托管进程内 `miv` 与路径 B SPI v2 微内核（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`、`/metrics`），内置 C++ BPE 分词器、Jinja 聊天模板、`<think>` `reasoning_content` 拆分与 Qwen XML/JSON `<tool_call>` 解析，可直接对接 Open WebUI、Cursor、Cline、NewAPI 与 OpenAI SDK。
- **单副本 NAX 权重与 `0-ULP` 一致性**：离线 NAX 预重排（`.miv`）让预填、`T=1` 自回归解码与投机校验共用同一份驻留权重（体积 `1.0×`）。基线版本 **`miv-v0.01`** 支持 Qwen3.5 / Qwen3.6 / Qwen3.8 混合架构模型（Dense 27B 与 35B-A3B MoE），覆盖 `MXFP4` 与 `oQ` Affine（`q3–q8`）量化，提供 `affine4_g128` KV 缓存、Native MTP / DFlash2、多用户并发批处理（`B/C = 1..8`）及滚动 UMA 前缀复用。
- **`oMLX-extend` 即插即用共存**：配套 `oMLX-extend`（基于 `jundot/omlx` 的 `omlx-extend-v0.8.3`）通过 AF_UNIX Provider SPI v2 Sidecar 接入 **`miv 核心后端`**——已适配的 `.miv` 模型按 O(1) 自动路由，未适配模型原样留在 Stock MLX 上运行，Sidecar 与 MLX 模型在同一内存上限下共享 UMA。

---

## 性能一览

![单流解码与 0-ULP 投机加速](../../assets/charts/decode-speedup.svg)

![多用户并发扩展聚合吞吐](../../assets/charts/concurrency-scaling.svg)

以上数据实测于物理 Apple M5 Ultra（`80-GPU / 256 GiB` 与 `64-GPU / 96 GiB`），在独立干净子进程中对标原生 `oMLX 0.8.0`（`mlx 0.32.4`，NAX 开启），全部通过 `0-ULP` 贪婪解码一致性校验。全部四张图表与 Dim A–F 完整数据见 **[`miv-v0.01` 发布说明](miv-v0.01-changelog.md)**。

---

## 支持的模型与微内核

| 模型家族 | 量化格式 | 微内核 | 接入模式 | 署名 / 来源 |
|---|---|---|---|---|
| **Qwen3.5 / 3.6 / 3.8 Dense**（0.8B – 27B，3:1 GDN+GQA） | `MXFP4`、`oQ4e` | `miv-v0.01` | 路径 A（主线） | `ninfer-metal` 核心 |
| **Qwen3.5 / 3.6 Hybrid MoE**（256×8 + 共享专家，如 35B-A3B） | `MXFP4`、`oQ4e` | `miv-v0.01` | 路径 A（主线） | `ninfer-metal` 核心 |
| *Llama、Qwen3（纯注意力）、DeepSeek MLA、Gemma、GLM、MiniMax 等* | *开放* | *征集 PR* | 路径 A / 路径 B | **[加入微内核工坊 →](microkernel-foundry.md)** |

开发者与编程 Agent 可通过 **[微内核工坊](microkernel-foundry.md)** 扩充这份清单：

- **路径 A（主线 PR）**：直接合入 `ninfer-metal` 的共享 Metal 4 NAX 底座，并在本表标明**贡献者署名**。
- **路径 B（并行 Sidecar）**：就绪后提交 PR 或新建 Issue，维护者将其登记至 [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json)（`sync_microkernels.sh`）并在本表附上**原始项目仓库链接**。`oMLX-extend` 与独立服务 `ninfer-metal-serve --providers-dir` 均原生扫描 `Providers/*/current/provider.json`，与 `miv`（及 `oMLX-extend` 中的 Stock MLX）并行运行路径 B 微内核。

---

## 快速开始

编译零 MLX 的 `miv/` 核心（`macOS 27+`、`Xcode 27+`）：

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cmake -S . -B build/miv-v001 -G Ninja -DCMAKE_BUILD_TYPE=Release -DOMLX_M5_BUILD_NAX_METALLIB=ON -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF -DOMLX_M5_BUILD_PYTHON=OFF
cmake --build build/miv-v001
```

一条命令启动独立 C++ OpenAI 服务（自动加载与 `oMLX-extend` 共用的默认配置 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)），或验证 `oMLX-extend` Sidecar 扩展层：

```bash
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test
```

完整编译参数、`.miv` 权重转换、多微内核 `ninfer-metal-serve --providers-dir` / `[[models]]` 配置及 `oMLX-extend` 热更新说明见 **[快速入门与配置](getting-started.md)**；macOS 应用界面图文指引见 **[`oMLX-extend` 可视化上手指南](omlx-extend-guide.md)**。

---

## 深入阅读

| 文档 | 内容 |
|---|---|
| **[`oMLX-extend` 可视化上手指南](omlx-extend-guide.md)** | 5 步看懂 `oMLX` macOS 界面：微内核卡片、模型引擎徽章、自动路由与切回原生 MLX、MTP / DFlash2 与 4-bit KV 量化，以及零手写共享配置 |
| **[快速入门与配置](getting-started.md)** | 环境要求、零 MLX CMake 编译、一键 `start_ninfer_serve.sh`、`ninfer-metal-serve` 独立服务、`oMLX-extend` Sidecar 配置、`ninfer-metal.toml` 以及底层 CLI/守护进程用法 |
| **[架构与多模型共存](architecture.md)** | 双模式 `miv/` 核心（`ninfer-metal-serve` + `mseries_backendd`）、Provider SPI v2、O(1) 自动路由、UMA 内存协调与原子符号链接热更新 |
| **[微内核工坊](microkernel-foundry.md)** | 面向开发者与编程 Agent 的 M5+ 新模型家族微内核征集指南（Llama、DeepSeek、Gemma、GLM、MiniMax）及 Silicon-Squeeze Skill |
| **[发布说明（`miv-v0.01`）](miv-v0.01-changelog.md)** | Dim A–F 基准图表、`ninfer-metal-serve` 与 `oMLX-extend` 变更摘要、测试与 `otool -L` 审计结果以及支持范围 |
| **[路线图](roadmap.md)** | `miv-v0.01` 之后的计划里程碑 |
| **[文档索引](../../README.md)** | 8 语种文档镜像、硬件指南、SKU 校准档案（`64-GPU` / `80-GPU`）、后端规范与认证标准 |

---

## 鸣谢

- **[`jundot/omlx`](https://github.com/jundot/omlx)** — 上游 oMLX 推理服务与 macOS 应用，`oMLX-extend` 在此基础上扩展。
- **[`Neroued/ninfer`](https://github.com/Neroued/ninfer)** — 参考原生推理引擎，为显式内存所有权与确定性调度设计提供重要启发。
- **[`z-lab/dflash`](https://github.com/z-lab/dflash)** 与 **[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx)** — DFlash / DFlash2 块扩散投机解码框架与 Apple Silicon 草稿模型生态。
- **[`SqueezeAILab/KVQuant`](https://github.com/SqueezeAILab/KVQuant)** 与 **[`FutureMLS-Lab/FOSCAR`](https://github.com/FutureMLS-Lab/FOSCAR)** — 低比特 KV Cache 量化研究与仿射折叠（Affine Folding）解码注意力思想，支撑了 `miv` 的 `affine4_g128` Metal 4 NAX 内核。
- **[`p-e-w/heretic`](https://github.com/p-e-w/heretic)** 与 **[`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)** — 上游 Heretic 免训练模型编辑框架，及其 Apple Silicon 原生 MLX 移植与社区量化衍生权重。
