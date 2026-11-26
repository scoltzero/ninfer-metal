[English](../../releases/miv-v0.02-changelog.md) | **简体中文**

# `ninfer-metal` `miv-v0.02` 发布说明

> **版本**：`ninfer-metal` `miv-v0.02`（`codename: miv` · `semver: 0.0.2` · `core_abi: 1` · `protocol_version: 2`）
> **配套宿主**：`oMLX-extend` `omlx-extend-v0.8.3-mix4`（面向 `jundot/omlx` `v0.8.3` 的 Provider SPI v2 扩展）
> **运行环境**：Apple M5 / M5 Pro / M5 Max / M5 Ultra（`MTLGPUFamilyApple10+`），macOS 27+

`miv-v0.02` 汇总了 `miv-v0.01` 发布以来的全部微内核演进（`mix1`–`mix4`），并彻底消除了 Apple10 GPU 在长上下文、多轮对话、Agent 工具调用与 `<think>` 思考模式下的默认性能瓶颈，开箱即享完整加速，无需手工配置 `path_overrides`。

---

## 1. `miv-v0.02` 核心更新

### Apple10 长上下文与预填默认高速路由
- **默认启用 Split-KV 注意力（`attention_split_kv`）**：真实模型（`hidden_size >= 1024`）在 Apple9+/Apple10 GPU 上自动启用 2-SG NAX Split-KV（`s512`），20K–34K 上下文下 `DFlash2` Target Verify GPU 耗时降低 **`4.0×`**（从 `93.45 ms/round` 降至 **`23.3–24.7 ms/round`**），从 `100` token 到 `34K+` token 单轮耗时保持平直。
- **`position_coupled_v1` 下默认启用 NAX Flash-Prefill**：在 Apple10 真实模型上默认解锁 `omlx_m5_paged_flash_prefill_affine4_g128_hd256`（`61.0 TFLOP/s`）与 `omlx_m5_paged_flash_prefill_nax_hd256`（`69.7 TFLOP/s`），~20K 冷启动首字延迟（TTFT）从 `18.5–19.3 s` 降至 **`9.65 s`**（提速近 **2×**），同时保持 `0-ULP` 一致性。
- **纯 AR 默认启用 `t_domain_dispatch`**：非投机纯自回归模式下默认开启 `t_domain_dispatch`；投机模式自动保持 `T=1 == T=8` 位级一致路由。

### DFlash2 投机解码与多轮 / 工具调用优化
- **8-SIMDgroup 协作归约 `omlx_m5_draft_attention`**：将 `DFlash2` 滑窗注意力（`head_dim` 128/256，`total_keys > 64`）从单 SIMDgroup 串行扫描升级为 8-SIMDgroup（256 线程）分块在线 Softmax 归约（0 寄存器溢出），`>= 2K` 窗口下草稿 GPU 耗时降低 **`2.1×`**（`13.52–14.20 ms/round` → **`6.54–6.63 ms/round`**）。
- **同槽位跨轮复用 DFlash2 KV 缓存（`retain_frontier`）**：在同槽位前缀命中（如连续多步 `<tool_call>` 网页搜索循环）时直接保留并推进草稿环形 KV 缓存前沿，仅投影新增 token，不再每轮重置重算 `2,048` 行。
- **单流稠密模型自适应门控修复**：`SpecSearchMode::Auto` 下单并发（`B = 1`）稠密模型稳定保持满深度 `K = max_k`，并隔离首轮冷上下文投影耗时对 `ema_draft_base_ms_` 的干扰，杜绝误退回 `K = 0`（`spec_search_gated_rounds = 0`）。

### `<think>` 思考模式跨轮滚动前缀缓存复用
- **零回读预解码思考锚点**：在开启 `rolling_prefix_cache` 的 `<think>` 模型（Qwen3 / Qwen3.5 / Qwen3.6 / Qwen3.8）上，Prefill 结束时自动记录轻量的 `include_anchor_logits = false` UMA 检查点（完全保留 1-CB 融合采样、零 CPU Logits 回读）。
- **剥离 `<think>` 后的跨轮秒级首字恢复**：当下一轮聊天模板自动剔除历史轮次的 `<think>...</think>` 思考链时，`SharedPrefixCache` 自动回退命中上一轮预解码锚点（20K `<think>` 多轮追问 TTFT 从 `~9.7 s` 降至 **`~0.25–0.30 s`**），并原地回收上一轮双检查点，保证 UMA 内存恒定有界、零意外驱逐。

### `.miv` 格式、`affine8` NAX 预重排与双模 LoRA（汇总 `mix2`–`mix3`）
- **`.miv` 标准格式与自动重排**：支持 `.miv`（向下兼容 `.omlxm5`）、首次加载自动重排主模型与 `DFlash2` 草稿模型，并扩展 Qwen3.5 / 3.6 / 3.8 `DFlash2` 白名单。
- **`affine8`（`oq_nax_tiled_g64_v1`）NAX 支持**：支持 8-bit `oQ` 权重的零拷贝 `AffineNaxCache` 执行。
- **双模 PEFT / `mlx-lm` LoRA（`merge` + `bypass`）**：支持 `reload_lora` 在线热切换、UMA 内存账本对齐及客户端界面配置。

---

## 2. 配套 `oMLX-extend` `v0.8.3-mix4` 更新

- **UI KV 压缩开关双向同步（`sync_m5rt_ui_kv_setting`）**：修复在界面切换 `turboquant_kv_enabled` 时漏同步 `external_engine_options.m5rt.kv_dtype` 的问题，开启后重载模型即可立即启用 `affine4_g128` 4-bit KV。
- **微内核注册表与 SPI 补丁同步**：更新 `M5RT_DEFAULT_VERSION` 与 `microkernels.json` 默认版本至 `miv-v0.02`，并保持 `omlx-v0.8.3-provider-spi.patch` 零差异同步。

---

## 3. 基准参考

Apple M5 Ultra（80 核 / 64 核）Dim A–F 完整基准图表与架构说明见 **[`miv-v0.01` 发布说明](miv-v0.01-changelog.md)**。
