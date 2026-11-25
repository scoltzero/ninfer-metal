[English](../../microkernel-foundry.md) | **简体中文** | [繁體中文](../zh-TW/microkernel-foundry.md) | [日本語](../ja/microkernel-foundry.md) | [한국어](../ko/microkernel-foundry.md) | [Español](../es/microkernel-foundry.md) | [Français](../fr/microkernel-foundry.md) | [Deutsch](../de/microkernel-foundry.md)

# 微内核工坊（Microkernel Foundry）

`ninfer-metal`（`miv-v0.01`）已在 Apple M5+ 芯片上提供 Qwen3.5、Qwen3.6、Qwen3.8 混合架构模型的裸硅 Metal 4 NAX 推理。**微内核工坊**向社区贡献者和编程 Agent 开放同一套运行时底座、服务前端、标定工具和认证流水线，支持通过主线 Pull Request 或并行 Sidecar 接入新的模型家族。

目标家族包括 **Llama、Qwen3 纯注意力（Pure-Attention）、DeepSeek（MLA）、Gemma、GLM、MiniMax**，以及社区量化衍生权重（如 [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)）。

---

## 两种贡献路径

### 路径 A：主线 PR → 贡献者署名（推荐）

在共享运行时 **`libninfer_metal_substrate`** 之上，将新模型家族的微内核直接合入 `ninfer-metal`。

**共享底座已提供：**
- Metal 4 执行运行时（`MetalContext`、`MTL4CommandQueue`、双缓冲 `ar_pipe_[2]`、`< 2 GiB` Compute ICB 编码器、`read_write` 屏障反射）。
- 共享分页 KV 与前缀内存管理（`KvPagePool`、`SharedKvArenaPool`、CoW 引用计数、`SharedPrefixCache`、`UmaCheckpointPool`）。
- 通用 NAX 与解码原语（`mxfp4_nax`、`affine_nax`、`bf16_nax`、`moe_nax`、零 `uitofp` QMV、两阶段并行 `Argmax`、`Split-KV` / `Flash-NAX`）。
- `.miv`（兼容旧版 `.omlxm5`）容器加载器、随包权重重排转换器（`tools/m5_artifact.py`）、`ExecutionIdentityFields`、`24 MHz` 硬件计数器、原生独立 OpenAI 服务（`ninfer-metal-serve`）以及 Provider SPI v2 守护进程（`mseries_backendd`）。

**你的 PR 补充内容与运行范围：** 补充目标架构的计算图、权重映射、配套的 `.safetensors → .miv` 重排转换规则，以及在冷 DRAM 微探针上较通用原语净增益 **`>= +10%–12%`** 的专属内核（例如 MLA、GeGLU、Softcap 或自定义 MoE 路由）。复用 `libninfer_metal_substrate` 可避免多架构间的代码与常驻内存重复，使模型同时原生运行在独立 `ninfer-metal-serve` 与 `oMLX-extend` 中（支持镜像站下载通用 `.safetensors` 后首次点击 `Load` 自动同目录重排生成 `.miv` 并保留原权重），并在[支持的模型清单](README.md)中标注贡献者署名。

### 路径 B：并行 Sidecar → PR 或 Issue → 原项目仓库署名

对于实验性架构、自定义量化排布或独立发版节奏，可实现独立的 Sidecar SPI v2 发布包，安装至：

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

通过原子 `current` 符号链接暴露，并在 `provider.json` 中声明 `supported_topologies`、`supported_model_types`、`artifact_extensions` 以及必选的随包重排转换器 `converter`（标准参数契约 `--source <dir> --output <file> --source-repo <id>`，重排产物与原 `.safetensors` 同目录并存），使从镜像源下载的标准模型在 `oMLX-extend` 首次加载或点击一键转换时即可自动完成重排。就绪后提交 PR 或新建 Issue：维护者会将其登记至 [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json)，供用户通过 [`./omlx-extend/scripts/sync_microkernels.sh`](../../../omlx-extend/scripts/sync_microkernels.sh) 安装和更新，同时在[支持的模型清单](README.md)中附上原始仓库直链。

**统一多微内核宿主：** `oMLX-extend` 与独立服务 `ninfer-metal-serve --providers-dir` 均原生扫描 `Providers/*/current/provider.json`，在同一端口上并行运行多个路径 A（**`miv 核心后端`**）与路径 B 微内核，并统一执行 UMA 内存预算仲裁与 `gpu_contended` 调度（在 `oMLX-extend` 中同时兼容 Stock MLX）。在 `oMLX` macOS 应用中，已安装的路径 B 微内核会自动出现在 **`Integrations → External Engines`** 卡片列表，并在 **`Models`** 页面显示引擎徽章（见 [`oMLX-extend` 可视化上手指南](omlx-extend-guide.md)）。微内核作者只需实现 Metal 4 推理核心与 Sidecar SPI v2 守护进程；HTTP/SSE 服务、分词、聊天模板与多模型调度均由两种宿主直接承接。

**开源许可：** **路径 A** 主线 PR 链接至 `miv` 并按 **[GPL-3.0](../../../LICENSE)** 分发（贡献者可提交 GPL-3.0、Apache-2.0 或 MIT 许可代码）；**路径 B** 并行 Sidecar 作为独立进程通过 AF_UNIX Sidecar SPI v2 通信，可使用任意许可证。

---

## 模型家族清单

| 架构家族 | 代表模型 | 状态 | 推荐路径 |
|---|---|---|---|
| **Qwen 混合架构（3:1 GDN + GQA）** | Qwen3.5 / 3.6 / 3.8 Dense（0.8B – 27B）与 35B-A3B MoE | 已发布（`miv-v0.01`） | 路径 A（主线，`miv/`） |
| **纯注意力 GQA** | Llama-3.3-70B、Qwen3-32B、Gemma | 开放 PR | 路径 A |
| **多头潜在注意力（MLA）** | DeepSeek V3 / R1 系列 | 开放 PR | 路径 A / 路径 B |
| **MoE 与社区量化权重** | GLM、MiniMax、`heretic-mlx` 衍生权重 | 开放 PR | 路径 A / 路径 B |

---

## 如何编写并认证一个微内核

无论是人类工程师还是编程 Agent，都按以下四步可复现流程推进：

1. **标定目标 Mac（`Stage B: H0 + H1`）**
   运行 `H0` 硬件 DMA 扫描（`MTLBlitPassDescriptor`，`4–512+ MiB`）实测本机芯片的 $B_{\text{DRAM}}$、$C_{\text{SLC}}$、$P_{\text{NAX}}$ 与 $I_{\text{ridge}}$，随后执行 `H1` 算子扫描。所有验收阈值均以这些实测常数为基准归一化。
2. **先写冷 DRAM 微探针（`~50–100` 行）**
   在模板化整套新算子前，先用独立冷 DRAM 探针（`working set >= 3–4 × C_SLC`，`>= 30` 次串行屏障中位数）对标固定功能 NAX/硬件基线。只保留净收益 **`>= +10%–12%`** 的内核，提交 PR 前删掉未达标的实验性 PSO。
3. **`< 10 s` 子层快速迭代与寄存器溢出审计**
   用 `omlx_m5_hw_counter_probe --mode model`（`24 MHz` 硬件计数器）定位各子层耗时占比（`mlp/moe`、`attn`、`lm_head`、`sampling`），再用 `metal-objdump -d` 审计着色器汇编（`0 alloca` 溢出、宽块预填 `<= 104` VGPR、`T=1` 预重排 QMV 保持 `0 uitofp`）。
4. **提交 5 文件机读证据包**
   运行净室 Dim A–F 评测脚本（`miv/tools/bench_omlx_vs_m5rt.py`），并随 PR 附上 `certification-evidence/<core-id>-<version>-<sku>/`：
   - `h0-capability.json` — 裸硅 `H0` 实测常数与 10,000 步 ICB 金丝雀测试
   - `h1-calibration-<sku>.json` — SKU 隔离的 `H1` 调优参数与 `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — 净室 Dim A–F 基准成绩与 `phys_footprint`
   - `parity-report.json` — `0-ULP` 一致性、源权重 PPL/KL、NIAH/RULER 召回率、`0 NaN/Inf`
   - `otool-audit.txt` — 纯 Apple 系统框架 `otool -L` 审计、`0 alloca` 审计、100 轮长跑测试与 CTest 输出

---

## 贡献者与 Agent 必读文档

- **[Silicon-Squeeze 核心编写 Skill](../../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — 完整的 Apple M5+（`MTLGPUFamilyApple10+`）裸硅调优方法论，可直接加载给编程 Agent 使用。
- **[核心认证标准](../../standards/core-certification-standard.md)** — Gate 1–4 精确通过门槛、Provider SPI v2 契约、Roofline 阈值与 Tier 1 / Tier 2 认证徽章规则。
- **[Apple M5 系列优化手册](../../silicon-guides/m5-series-optimization-playbook.md)** — `32 KiB` 线程组驻留率推导、统一内存准入方程以及 Base / Pro / Max / Ultra 扩展规律。
