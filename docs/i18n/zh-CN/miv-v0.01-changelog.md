[English](../../releases/miv-v0.01-changelog.md) | **简体中文** | [繁體中文](../zh-TW/miv-v0.01-changelog.md) | [日本語](../ja/miv-v0.01-changelog.md) | [한국어](../ko/miv-v0.01-changelog.md) | [Español](../es/miv-v0.01-changelog.md) | [Français](../fr/miv-v0.01-changelog.md) | [Deutsch](../de/miv-v0.01-changelog.md)

# `ninfer-metal` `miv-v0.01` 发布说明

> **版本**：`ninfer-metal` `miv-v0.01`（`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`）
> **配套宿主**：`oMLX-extend` `omlx-extend-v0.8.3`（面向 `jundot/omlx` `v0.8.3` 的 Provider SPI v2 扩展）
> **运行环境**：Apple M5 / M5 Pro / M5 Max / M5 Ultra（`MTLGPUFamilyApple10+`），macOS 27+（运行无需 Xcode；源码编译需 Xcode 27+）

`miv-v0.01` 是 `ninfer-metal` 的首个公开发布版本。引擎采用纯 C++20、Objective-C++ 与 Metal 4 编写，直接调用 Apple M5+ 硬件神经网络加速器（NAX）与 Compute ICB，运行时不依赖 Python 或 MLX。本版本支持 **Qwen3.5 / Qwen3.6 / Qwen3.8 混合架构**模型（Dense 0.8B–27B 与 256×8 MoE，如 35B-A3B），既可用原生 OpenAI HTTP/SSE 服务 `ninfer-metal-serve` 独立部署，也可作为 **`miv 核心后端`** 接入 `oMLX-extend`，与 Stock MLX 同机运行并提供 OpenAI、Anthropic 接口及原生 macOS 界面。安装步骤与命令行参数见 [快速入门与配置](getting-started.md)，macOS 客户端界面操作见 [`oMLX-extend` 可视化上手指南](omlx-extend-guide.md)。

## 1. Apple M5 Ultra 实机基准

以下数据均在物理 Apple M5 Ultra 上以独立干净子进程实测，对比同机原版 `oMLX 0.8.0`（`mlx 0.32.4`，开启 NAX），采用相同模型权重、相同 Token 序列、`temperature=0` 与 `phys_footprint` 内存统计。80 核 256 GiB（`macOS 27.2.1` Build `26B131`，`Xcode 27.2` Build `27B52`）与 64 核 96 GiB 两款规格分列记录。

### 单流解码与 `0-ULP` 投机加速

![单流解码与 0-ULP 投机加速](../../assets/charts/decode-speedup.svg)

- 80 核 M5 Ultra 纯自回归解码在全部受测权重格式上均快于原版 oMLX：`Qwen3.8-27B MXFP4` 达 `64.63 tok/s`（`+14.5%`），`27B oQ4e` 达 `54.03 tok/s`（`+7.3%`），`Qwen3.6-35B-A3B MXFP4` 达 `216.44 tok/s`（`+23.7%`）。
- 投机解码让 `27B` 代码生成达到 `186.74 tok/s`（`DFlash2`，AR 的 `2.89×`）与 `140.48 tok/s`（`Native MTP`，较原版 oMLX MTP 的 `103.28 tok/s` 快 `+36.0%`），全部保持 `128/128` `0-ULP` 贪婪解码一致性。
- 64 核 96 GiB 机型上，`27B oQ4e` 在 Coding 场景达到 `138.92 tok/s`（`DFlash2`，`+33.9%`）和 `107.52 tok/s`（`MTP`，`+21.2%`），Reasoning 场景为 `110.01` / `100.28 tok/s`（`6/6` 组、`256/256` `0-ULP`）。

### 多用户并发扩展

![多用户并发扩展聚合吞吐](../../assets/charts/concurrency-scaling.svg)

- 80 核机型共测 19 组并发点，`miv` 全部领先：`B=4` 下 `27B` 达到 `500.57 tok/s`（`oQ4e DFlash2`）、`496.60 tok/s`（`MXFP4 DFlash2`）和 `434.92 tok/s`（`oQ4e MTP`），为原版 oMLX 基线（`148.37` MTP / `129.80` AR）的 `2.93×–3.83×`。
- `Qwen3.6-35B-A3B` 在 `B=8` 纯 AR 下达到 `846.29 tok/s` 窗口吞吐（端到端 `794.39 tok/s`），较原版 oMLX 高 `+52.6%`，首字延迟（TTFT）快 `4.16×`。
- 所有并发槽位均与单流输出保持 `128/128 × 4` `0-ULP` 贪婪解码一致性。

### 64K 长上下文稳定性、UMA 占用与预填

![64K 长上下文稳定性与 affine4_g128 内存占用](../../assets/charts/long-context-curve.svg)

- 启用 `affine4_g128` KV 后，`27B MXFP4` 从 `2K` 增至 `64K` 时 AR 速度仅降 `-4.1%`（`61.42 → 58.92 tok/s`，较 oMLX 快 `+28.6%`）；真实英文与代码混合负载下，`64K` `DFlash2` 保持 `139.70 tok/s`（`2K–64K` 全程 `0-ULP`）。
- `affine4_g128` 将 `64K` KV 缓存缩减 `71.8%`（`4.00 → 1.13 GiB`），单次请求峰值 UMA 降至 `15.69 GiB`（比 oMLX 省 `3.44 GiB`；四档前缀同时驻留共省 `10.06 GiB`）。`64K + 32` 热前缀 TTFT 降至 `176.19 ms`（`bf16`）与 `201.42 ms`（`affine4`），较 oMLX 快 `7.05×–9.48×`。
- 80 核机型 `1K` 预填达到 `2,058.5–2,110.6 tok/s`（`100.25–102.80 TF`），`64K` 冷预填耗时 `41.02–41.84 s`；64 核机型 `4K` 预填 15 项全胜，`35B-A3B` 达 `10,048.5 tok/s`（`+15.9%`）。

### 多轮短对话与滚动前缀复用

![多轮短对话滚动前缀复用与内存平直度](../../assets/charts/multiturn-rolling.svg)

- 滚动 UMA 前缀缓存将多轮命中率提高到 `92%` 以上（`B=1` `92.33%`，`B=4` `92.02%`），重复预填的 Token 减半，热 TTFT 缩短 `2.44×–2.94×`（`B=1` `101.25 ms`，`B=4` `282.80 ms`）。
- 原地 slab 复用让 `B=1` 10 轮对话的 UMA 增量保持在 `+0.02 MB` 且零驱逐（原版 oMLX `affine4` 增长 `+397.5 MB`，`bf16` 增长 `+704.8 MB`），10 轮总耗时从 `16.99 s` 降至 `9.26 s`（`DFlash2`）。
- `B=4`（4 用户 × 6 轮）下，批处理 `affine4` NAX 解码达 `176.95 tok/s`（`AR`）和 `206.35 tok/s`（`DFlash2`，oMLX 的 `1.72×`），峰值 UMA 省 `2.12 GiB`，`B=1`（`10/10`）与 `B=4`（`24/24`）全轮次通过 `0-ULP` 校验。

## 2. 本版包含内容

### 独立 OpenAI 服务（`ninfer-metal-serve`）
- 单一 C++20 HTTP/SSE 二进制，提供 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health` 与 Prometheus `/metrics`，支持同步 JSON、SSE 流式输出（含 `usage` 与 `[DONE]`）、可选 Bearer 鉴权（`--api-key`）、CORS 及平滑退出。
- 内置 C++ BPE 分词器（直接读取 `tokenizer.json`）、Jinja 聊天模板渲染、`<think>` `reasoning_content` 拆分以及 Qwen XML/JSON `<tool_call>` 同步与流式解析。
- 支持同端口多微内核并行托管：使用 `--providers-dir` 与 `[[models]]` 在进程内 `miv` 旁加载外部路径 B Provider SPI v2 Sidecar，统一执行 UMA 内存上限（默认物理内存的 `81%`）与并发争用调度。
- 随附可直接运行的 [`ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)（与 `oMLX-extend` 共用并自动加载）及一键启动脚本 `./omlx-extend/scripts/start_ninfer_serve.sh`。

### Metal 4 NAX 推理引擎（`miv/`）
- 自包含的 C++20 / Objective-C++ / Metal 4 代码树，内置 38 个预编译 `.metallib` 着色器，覆盖 NAX 预填、融合解码、256×8 MoE 路由、投机校验与 4-bit Split-KV，编译与运行均不依赖 MLX 或 Python 头文件。
- 为 Apple M5 Ultra 两款规格内置独立硬件校准档案（[80 核 256 GiB](../../backend/silicon-calibrations/apple-m5-ultra-80gpu.json) 与 [64 核 96 GiB](../../backend/silicon-calibrations/apple-m5-ultra-64gpu.json)），按芯片自动匹配 split-K、Split-KV 分区与投机草稿深度，同时保留环境变量覆盖。
- 支持零拷贝 GPU 前缀续接（`64K` 恢复耗时 `0.73–1.62 ms`）、原地滚动 UMA 前缀复用以及面向并发请求的批处理 `affine4_g128` NAX 注意力。

### `oMLX-extend` `omlx-extend-v0.8.3`（macOS 应用与 Sidecar）
- 以 `AF_UNIX` Sidecar（`mseries_backendd`）接入 **`miv 核心后端`**：已适配的 `.omlxm5` 模型按 O(1) 自动路由，普通 MLX 模型留在 Stock MLX 上运行，不拉起额外进程。
- 原生 macOS 界面在 `Integrations → External Engines` 中新增 `miv 核心后端` 与路径 B 微内核卡片，并在模型列表与详情页提供活跃引擎徽章、自动路由 / Stock MLX 切换，以及按后端能力解锁的 `64K+` 上下文、`MTP`、`DFlash2` 和 `affine4_g128` KV 控件。
- 在 Sidecar 与 MLX 模型之间协调 UMA 预算、空闲缓存回收与并发分块预填，并支持以 `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/` 原子符号链接直接升级与回滚。

## 3. 独立打包与验证

- 发布包 `ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` 包含 16 个 Mach-O 二进制（含 `ninfer-metal-serve`、`mseries_backendd` 与命令行/硬件校准工具）、38 个 Metal 4 `.metallib` 着色器、2 套 SKU 校准档案、`provider.json` 及 SHA-256 校验文件。
- 对全部 16 个二进制执行 `otool -L` 审计，确认仅链接 Apple 系统框架（`CoreFoundation`、`Foundation`、`IOKit`、`Metal`）与系统动态库（`libc++.1`、`libSystem.B`、`libobjc.A`），运行时无 Homebrew、Xcode、MLX、OpenSSL 或 Python 依赖。
- 端到端服务测试（覆盖同步/SSE、`<think>`/`<tool_call>`、`B=4` 批处理以及进程内 + 外部 Sidecar 并行托管）、`oMLX-extend` 全部 `638 / 638` 项 pytest 以及 `75 / 75` 项 macOS XCTest 均已通过，Provider SPI 补丁可无冲突应用于 `jundot/omlx` `v0.8.3`。

## 4. 支持范围

| 项目 | `miv-v0.01` 支持范围 |
|---|---|
| 硬件 | Apple Silicon `MTLGPUFamilyApple10+`（Apple M5 / M5 Pro / M5 Max / M5 Ultra，含 64 核与 80 核） |
| 系统与编译 | macOS 27+（运行无需 Xcode；源码编译需 Xcode 27+ / macOS 27 SDK） |
| 模型架构 | **Qwen3.5 / Qwen3.6 / Qwen3.8 混合架构**（3:1 GDN + GQA；Dense 0.8B–27B 与 256×8 MoE，如 35B-A3B） |
| 权重格式 | 离线 NAX 预重排 `.omlxm5`：`MXFP4`（`E2M1 × UE8M0`）与 `oQ` Affine（`q3–q8`，`g64 / g128`） |
| KV 缓存 | `bf16` 与 `affine4_g128`（`0.5625 B/val`，`2K–64K` 通过 `0-ULP` 校验） |
| 投机解码 | 纯 AR（`none`）、内嵌 `Native MTP` 与 `DFlash2`（支持单流及 `B=2..4` 批处理） |
| 服务形态 | `ninfer-metal-serve` 提供 OpenAI HTTP/SSE 接口；`oMLX-extend` 提供 OpenAI + Anthropic 接口与原生 macOS 界面 |
