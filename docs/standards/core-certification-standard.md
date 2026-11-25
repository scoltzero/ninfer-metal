# `ninfer-metal` 核认证与准入标准（Core Certification Standard v1.0-draft）

> **状态**：`Draft — 待用户评审确认（Pending User Sign-Off）`  
> **适用对象**：`ninfer-metal`（基准核 `miv-v0.01`，独立运行或作为 `oMLX-extend` `vX.Y.Z-extend` 的 **`miv 核心后端`** / `miv Core Backend` Sidecar）  
> **配套文档**：Core-Authoring Skill [`qwen-mxfp4-oq4e-silicon-squeeze-skill.md`](../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md) · 基准套件 [`miv/tools/bench_omlx_vs_m5rt.py`](../../miv/tools/bench_omlx_vs_m5rt.py)（Dim A–E + Dim F `--suite multiturn`）  
> **参考边界**：独立服务参考 `ninfer`，Sidecar 对齐 `oMLX-extend` Provider SPI v2，模型与量化生态覆盖含 [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx) 在内的 Apple Silicon 社区权重。

---

## 0. 适用范围与三条原则

本标准规定新模型核、架构模块与第三方并行 Sidecar 核在合入 `ninfer-metal` 主干或取得 `oMLX-extend` 认证前必须通过的四道门禁（Gate 1–Gate 4）。

1. **物理常数归一化（Roofline-Normalized）**：禁止用单一 SKU 或单一参数量模型的绝对 `tok/s` / `ms` 作为跨机型门槛。所有算力与带宽指标必须归一化至目标测试机 Stage B `H0` 实测物理常数：`gpu_cores`、纯 `FP16/BF16 -> FP32` NAX 峰值 $P_{\text{NAX}}$、硬件 DMA 持续带宽 $B_{\text{DRAM}}$、算术平衡点 $I_{\text{ridge}} = P_{\text{NAX}} / B_{\text{DRAM}}$、片上缓存拐点 $C_{\text{SLC}}$ 与推荐工作集 $W_{\text{ws}} \approx 0.81 R$。
2. **失败即拒绝（Fail-Closed）**：任一强制门禁未通过、存在跨 SKU 借标、能力虚报或证据缺失，直接判定 `REJECT`。
3. **机读证据审计**：所有通过声明必须绑定机读 JSON 证据包与执行指纹（`ExecutionIdentityFields` / `kernel_bundle_id`），供 CI 自动核验。

---

## 1. 双轨接入路径与反重复约束

贡献者为新模型家族（Llama、Qwen3 Pure-Attn、DeepSeek MLA、Gemma、GLM、MiniMax）或社区量化衍生权重（如 [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)）提交 M5+ 核时，选择以下两条路径之一。

### 1.1 Path A：主干核模块 PR（In-Tree Core Module）

合入 `ninfer-metal` 主仓库并随主版本分发，必须链接公共运行时底座 `libninfer_metal_substrate`（从 `miv-v0.01` 抽离的共享库）：

| 公共子系统 | 底座组件 | 模块可扩展边界 |
|---|---|---|
| **Metal 4 运行时** | `MetalContext`、`MTL4CommandQueue`、双槽 `ar_pipe_[2]` `MTL4CommandAllocator`、`MTLResidencySet`、`< 2 GiB` / `< 1 GiB` Compute ICB 编码器、`read_write` 反射屏障 | 仅编排本架构子图；禁止自建队列/分配器或绕过 ICB 分段上限 |
| **分页 KV 池** | `KvPagePool`、`SharedKvArenaPool`、物理页 CoW 引用计数、`retained_kv_pages_` 唯一页计费 | 仅注册本架构 `(kv_heads, head_dim, kv_dtype)` 几何 |
| **前缀与 UMA 快照** | `SharedPrefixCache`、`UmaCheckpointPool`（`32 MiB` 量子化、`supersede_entry_id` `1×` 覆写、pinned 保护） | 仅实现架构专属递推状态的 GPU 快照回调（若有） |
| **通用 NAX 原语** | `mxfp4_nax`、`affine_nax`、`bf16_nax`、`moe_nax`、`row_exact` 零 `uitofp` QMV、两阶段并行 `Argmax`、统一 `Split-KV` / `Flash-NAX` | 仅在通用原语无法表达的算子（如 MLA、GeGLU、Softcap、Lightning Attention、Sigmoid+Bias 路由）上新增专属核 |
| **身份与制品契约** | `.miv`（兼容历史 `.omlxm5`）容器读取器、配套重排转换脚本、`ExecutionIdentityFields`、`kernel_bundle_id`、`execution_artifact_digest` | 仅注册 `topology_id`、权重映射与本架构重排规则 |
| **标定与观测** | Stage B `H0`/`H1` 探针、`24 MHz` `HwCounterCollector`、校准表加载器 | 仅补充本架构专属 `H1` 扫描项 |
| **双模前端** | Sidecar UDS NDJSON RPC（`ninfer-metal sidecar` / `mseries_backendd`）与独立前端（`ninfer-metal serve`） | 禁止模块自建通信协议或 HTTP 服务器 |

**Path A 反重复规则**：
1. 禁止复制 `MetalContext`、ICB `Encoder`、`KvPagePool`、`SharedPrefixCache`、`UmaCheckpointPool` 或 Sidecar RPC 代码。
2. 通用投影、`SwiGLU`、`RMSNorm`、词表头与采样必须调用底座 PSO；除非冷 DRAM 微探针证实净增益 **`>= +10%–12%`**（Gate 2.2），否则禁止新增平行 `.metallib`。
3. 同一发布包内的模型模块共享同一套 `.metallib` 与静态底座二进制。

### 1.2 Path B：独立并行 Sidecar 核（Standalone Parallel-Loadable Sidecar）

面向独立仓库、独立发版节奏或实验架构，完整实现 `oMLX-extend` Provider SPI v2，与 `miv` 及 Stock oMLX 同机并行加载：
1. **目录隔离**：安装于 `~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/`，经原子 `current` 符号链接暴露 `provider.json`、二进制与 `.metallib`；禁止写入其他 Provider 目录或改动 `/Applications/oMLX.app` Python `venv`。
2. **进程隔离**：每个加载的模型运行在独立子进程，持有独立 `MTLDevice`、`MTL4CommandQueue`、`MTLResidencySet` 与临时 `AF_UNIX` 套接字；进程异常不影响宿主或其他核。
3. **UMA 预算服从**：遵守 `ProcessMemoryEnforcer` 下发的 `memory_budget_bytes` 硬上限与 `trim_cache` 回收指令。

### 1.3 路径对照

| 维度 | Path A（主干模块 PR） | Path B（独立并行 Sidecar） |
|---|---|---|
| **仓库与分发** | 合入 `ninfer-metal` 主干，共享 `libninfer_metal_substrate` | 独立仓库与发布，通过 `provider.json` 发现 |
| **复用要求** | 仅写拓扑编排与过探针门限的专属核 | 自行打包运行时或静态链接 `libninfer_metal_substrate` |
| **宿主协议** | 复用主干 SPI v2 与独立模式 | 完整实现 `oMLX-extend` Provider SPI v2（含共存 RPC） |
| **准入门禁** | **Gate 1–4 全部强制** | **Gate 1–4 全部强制** |

---

## 2. Gate 1 — Sidecar SPI v2 与多核共存

### 2.1 `provider.json` 清单与真实能力声明

`current/provider.json` 必须包含以下结构与字段：

```json
{
  "schema": "omlx-provider-manifest-v2",
  "provider_id": "io.ninfer.metal",
  "display_name": "miv Core Backend (ninfer-metal)",
  "version": "0.0.1",
  "protocol_version": 2,
  "core_abi_version": 1,
  "min_omlx_extend_version": "0.8.0-extend",
  "entrypoint": "bin/mseries_backendd",
  "metallib_dir": "metallib",
  "artifact_extensions": [".miv", ".omlxm5"],
  "converter": {
    "entrypoint": "tools/m5_artifact.py",
    "subcommand": "convert",
    "output_filename": "model.miv",
    "artifact_extension": ".miv",
    "default_args": ["--nax-pretile"],
    "supported_quant_modes": ["affine", "mxfp4", "mxfp4-pack-quantized"]
  },
  "supported_topologies": ["qwen35_hybrid_dense_v1", "qwen36_hybrid_moe256_k8_shared_v1"],
  "supported_quant_formats": ["mxfp4_e2m1_g32_e8m0_v1", "oq_affine_q4_g64_bf16_v1", "oq_affine_q5_g64_bf16_v1", "oq_affine_q6_g64_bf16_v1", "oq_affine_q8_g64_bf16_v1"],
  "supported_kv_formats": ["bf16", "affine4_g128"],
  "supported_speculative": ["none", "mtp", "dflash2"],
  "supported_standard_settings": [
    "max_context_window",
    "mtp_enabled", "mtp_adaptive_max_depth", "mtp_fixed_depth",
    "dflash_enabled", "dflash_draft_model", "dflash_block_size",
    "turboquant_kv_enabled", "turboquant_kv_scheme"
  ],
  "options_schema": [
    {"key": "max_concurrency", "type": "int", "min": 1, "max": 8, "default": 4, "reload_required": true},
    {"key": "prefix_cache_gb", "type": "bytes_gb", "min": 0, "max": 128, "default": 4, "reload_required": true},
    {"key": "rolling_prefix_cache", "type": "bool", "default": true, "reload_required": true},
    {"key": "sampling_contract", "type": "enum", "options": ["position_coupled_v1", "legacy_rejection_v1"], "default": "position_coupled_v1", "reload_required": true}
  ]
}
```

- **零虚假能力（Zero Phantom Capabilities）**：`provider.json` 与 `hello` 声明的每一项 `supported_kv_formats`、`supported_speculative`、`supported_topologies` 必须具备可执行的底层算子与对应 CTest；仅在参数解析白名单放行但运行时报错的选项禁止声明。
- **随包权重重排转换器（Mandatory Bundled Weight Converter）**：凡使用硬件预重排容器（如 `.miv` 或自定义后缀）而非直接读取通用 `.safetensors` 的 Path A / Path B 微内核，必须在发布包内随附重排脚本并在 `provider.json` 的 `converter` 字段声明标准 CLI 契约（接收 `--source <model_dir> --output <tmp_file> --source-repo <repo_id>`，原子生成同目录目标权重文件且**严禁删除或覆盖原 `.safetensors`**），以保证用户从 Hugging Face / 镜像站下载通用模型后，在 `oMLX-extend` 中首次点击 `Load` 或点击一键转换即可自动完成重排。
- **UI 同步**：映射到 `oMLX-extend` 原生设置的键通过 `supported_standard_settings` 声明，核心专属项通过 `options_schema` 暴露。

### 2.2 `hello` 握手、`core_abi` 与回退边界

| 检查项 | 强制要求 | 失败条件 |
|---|---|---|
| **`hello` 必返字段** | 返回 `provider_id`、`provider_version`、`protocol_version`、`engine_contract_version`、`core_abi_version`、`pid`、`phys_footprint_bytes`、`capabilities[]`、`supported_topologies[]`、`supported_kv_formats[]` | 缺字段、与 `provider.json` 冲突或硬编码旧版本 |
| **`core_abi` 与拓扑校验** | `load` 校验制品 `core_abi_min/max`、`topology_id` 与 `.metallib` 完整性，不匹配立即返回结构化错误码 | 强行加载、静默截断配置或崩溃 |
| **首 Token 回退边界** | 启动或首 Token 前失败允许宿主记录 `fallback_reason` 回退至 Stock oMLX；**首 Token 发出后禁止重放或切换引擎** | 首 Token 后重复输出、状态错乱或吞错 |

### 2.3 统一内存硬上限 `memory_budget_bytes` 与 `trim_cache`

核内禁止仅凭进程局部 `device.recommendedMaxWorkingSetSize - current_allocated_bytes()` 规划 KV 页池或前缀缓存（该值对父进程 MLX 与其他 Sidecar 不可见，会诱发 Jetsam OOM）。

1. **`load` `memory_budget_bytes` 硬钳位**：宿主在全局上限 $W_{\text{cap}} = \min(\text{enforcer\_ceiling},\; 0.90 \times W_{\text{ws}})$ 内扣除其他驻留模型后下发 `memory_budget_bytes`。核内一切池（权重、`shared_kv_pages`、`prefix_cache_bytes`、`UmaCheckpointPool`、工作区）可用额度取双上界最小值：
   $$\text{avail\_ws} = \min\!\Big(W_{\text{ws}} - \text{cur\_alloc},\; \max\big(0,\; \text{memory\_budget\_bytes} - \text{cur\_alloc}\big)\Big)$$
   请求超预算时拒绝加载或缩减非必需缓存；稳态峰值 `phys_footprint_bytes` 不得超过 `memory_budget_bytes`（允许 `<= 2%` 页对齐/驱动元数据误差）。
2. **`trim_cache(target_free_bytes)`**：宿主触及软水位（如 `0.85 × W_cap`）时调用，核必须在 **`<= 100 ms`** 内驱逐未 `pinned` 前缀并释放 `UmaCheckpointPool::free_slabs`，同时将对应 `MTLBuffer` 从 `MTLResidencySet` 解绑提交以即时释放 Wired 页，返回 `freed_bytes` 与最新 `phys_footprint_bytes`。

### 2.4 共享 GPU 争用协调与零端口冲突 IPC

- **`set_scheduler_hint({"gpu_contended": bool, ...})`**：`gpu_contended == true` 时，长预填切入有界分块（如 `prefill_next_contended_chunk`，`64`-token 页对齐且不跨 `1024`-token 检查点边界）并关闭异构 ANE 路径。
- **AF_UNIX IPC**：Sidecar 模式使用每模型实例独立临时套接字（`/tmp/omlx-<provider>-<uid>/<id>-<uuid12>.sock`，目录 `0700`、套接字 `0600`），双连接（控制 RPC + `subscribe_events`），支持 `ECONNREFUSED` 过期套接字清理与 `--exit-on-disconnect` / 父 PID 看门狗；Sidecar 模式禁止监听 TCP 端口，独立 `serve` 模式须显式 `--port` 且默认端口避开 oMLX `8000`。
- **未适配模型零影响与退出归零**：`detect_adapted(model_dir)` 经文件 `stat()` / 轻量 JSON 在 **`< 1 ms`** 内返回；未适配模型不拉起子进程、不占额外内存、不改原版 Stock oMLX 调度与 UI。`shutdown` / `stop()` 后经 `proc.wait()` 确认进程退出、`process_memory_bytes() == 0` 且无残留 `.sock`。

---

## 3. Gate 2 — 硅片工程与 Roofline 门禁

| 编号 | 检查维度 | 硬门限（Fail-Closed） | 核验依据 |
|---|---|---|---|
| **G2.1** | **Stage B `H0` 裸硅片标定** | 目标 `MTLGPUFamilyApple10+` 真机核验 `apple_gpu_family >= 10`、`native_tensorops_eligible`、`metal4_icb_eligible`、`max_threadgroup_memory_length == 32,768 B`；用硬件 DMA（`MTLBlitPassDescriptor`）扫 `4–512+ MiB` 锁定 $C_{\text{SLC}}$ 与 `>= 4 × C_SLC` 稳态 $B_{\text{DRAM}}$（禁止标量 Compute 拷贝）；测 `matmul2d` `FP16/BF16 -> FP32` $P_{\text{NAX}}$ 与 $I_{\text{ridge}} = P_{\text{NAX}} / B_{\text{DRAM}}$ | `h0-capability.json`、`hw-counters-h0.json` |
| **G2.2** | **Stage B `H1` 与 `50` 行冷 DRAM 探针** | 任何新算子族模板化前须经 `~50–100` 行冷 DRAM 探针（工作集 `>= 3–4 × C_SLC`、`MTLResourceStorageModePrivate`、`>= 30` 次串行屏障中位数），相对硬件固定功能基线净增益 **`>= +10%–12%`**；未达标实验 PSO 与废弃分支合入前物理删除 | `h1-calibration-<sku>.json`（含 `prior_vs_measured_diff`）、代码 diff |
| **G2.3** | **Compute ICB 分段与子图隔离** | 绑定进 `MTLIndirectComputeCommand` 的权重段严格 **`< 2 GiB`**（`<= 2,147,483,648 B`）、辅助紧凑段（`Q5High1Cache`、`sb_tiled`）**`< 1 GiB`**、KV 页视图偏移 **`< 2 GiB`**（防 `>= 4 GiB` 32-bit 回绕）；子图边界调用 `Encoder::dynamic_parameters.clear()`；并发无屏障重叠核反射标记 `read_write`；多槽动态采样参数带唯一槽位标签 | 静态断言 + `omlx_m5_inspect_icb_canary` 10,000 步重放 |
| **G2.4** | **微架构驻留与零溢出** | 1. 宽块 Prefill 核 **`<= 104` VGPR**（维持 `3` TGs/core × `128` 线程，Zero-`As` + `Ws <= 9,216 B` + 单级预取，禁止双级 `swpipe`）；<br>2. 热路径生产核 **`0` `alloca`**；<br>3. `T=1` / `M=2` 预重排直读核 **`0` `uitofp`/`sitofp`**、**`0` `1 B` `uchar`** 加载；<br>4. 禁止 `M >= 3` 多行标量 PSO（$P_{\text{scalar\_FP32}} \approx 0.08 P_{\text{NAX}}$）、`_rf2/_rf4` 行复用循环、$M \le 64$ ANE 卸载、`-O3 -ffast-math` 下 FP16 `0x6400` 减法位铸造 | `metal-objdump -d`（`objdump-spill-audit.txt`）、CMake 标志审计 |
| **G2.5** | **单份预重排内存上限** | 加载后空闲权重物理驻留满足 $M_{\text{resident,weights}} \le \mathbf{1.05\times} M_{\text{artifact}}$（离线预重排单份跨 Prefill / AR / Verify 复用，拦阻在线双份 `~1.92×` 膨胀） | `post_load_phys_footprint_gb` vs 制品字节数 |
| **G2.6** | **`24 MHz` 子层硬件计数** | 支持 `MTL4CounterHeap` / `MTLCounterSampleBuffer`（`24 MHz` = `41.67 ns/tick`，关闭态内联早退零堆分配），`< 10 s` 输出 `mlp/moe`、`linear_attn/gdn`、`full_attn`、`lm_head`、`sampling` 耗时占比、派发/屏障数、`% of P_NAX` 与 `% of B_DRAM` | `hw_counter_probe --mode model` |
| **G2.7** | **多 SKU 校准隔离** | 校准文件严格绑定单个 `profile_id`（禁止跨 SKU 借用 `measured_qualified`）；未实测 SKU 回退至标明 `theoretical_prior` 的参数模型；跨 SKU 对比采用同二进制双锚点方程 $\tau_{\text{step}}(M_{\text{eff}})$ 与 $\eta_{\text{ckpt}}$ | `calibrations/<profile_id>.json` 指纹 |

---

## 4. Gate 3 — Dim A–E + Dim F 基准矩阵

### 4.1 净室执行协议

所有测试由 [`miv/tools/bench_omlx_vs_m5rt.py`](../../miv/tools/bench_omlx_vs_m5rt.py)（或同构契约 Harness）运行：
1. **独立子进程**：每个引擎与配置在独立进程运行，防止 Metal 堆残留与峰值内存串扰。
2. **锁定输入与采样**：相同本地权重、相同原始 Token ID 序列（记录 `prompt_sha256`）与确定性贪心（`temperature=0.0, top_p=1.0, top_k=0, seed=0`）。
3. **统计口径**：Dim A–E `warmups >= 2, repeats >= 5`（以 `median` 验收，记 `mean/min/max/stdev`）；Dim F（`--suite multiturn`）`warmups >= 1, repeats >= 3`（记 `median`、`p95` 与逐轮轨迹）。
4. **物理内存口径**：统一采集 XNU `task_vm_info.phys_footprint` 与 `ledger_phys_footprint_peak`（`read_phys_footprint_bytes()`），禁止用 `ps rss` 替代。

### 4.2 Dim A–F 门限总览

| 维度 | 套件与输出 | 硬通过门限（Fail-Closed） | 目标（Target） |
|---|---|---|---|
| **Dim A** | `--suite dim_a`<br>`dim-a-prefill.json` | Linear/MLP **`>= 85%`** $P_{\text{NAX}}$；Dense 整网 **`>= 75%`** $P_{\text{NAX}}$；`0` 瞬时非 KV 预填 RSS 爆炸；`0` `bf16->f16`/`xsum`/`gather`/reduce 税 | Linear/MLP `90%–98%`；Dense `85%–92%` |
| **Dim B** | `--suite dim_b`<br>`dim-b-ar-decode.json` | Dense `TDom=1` **`>= 90%`** $B_{\text{DRAM}}$（纯 GPU TPOT `<= 1.11× TPOT_floor`）；主机间隙 **`< 3%`**；`0 uitofp / 0 uchar / 0 alloca` | `95%–100%` $B_{\text{DRAM}}$；主机间隙 `< 2%` |
| **Dim C** | `--suite dim_c`<br>`dim-c-spec-decode.json` | $S_{\text{spec}} > \mathbf{1.05\times}$（分母严格为 `PureAR`）；**`100%`（`256/256`）`0-ULP`**；非计算开销 `< 5%` | Coding $S_{\text{spec}} \ge 2.0\times$；非计算 `< 3%` |
| **Dim D** | `--suite dim_d`<br>`dim-d-batched-decode.json` | `B=2 >= 1.50×`、`B=4 >= 2.20×`（vs `B=1`）；`B=8 > B=4`；`B=4` 单流中位 `>= 55%` `B=1`；**`100%`（`128/128`）`0-ULP`**；`1`-CB 单步提交 | `B=2` `1.7×–1.95×`、`B=4` `2.6×–3.5×` |
| **Dim E** | `--suite dim_e`<br>`dim-e-longctx-mem-prefix.json` | 混合架构 `64K` 衰减 **`<= 10%`**（纯全注意力 `>= 65%` $B_{\text{DRAM}}$ 且 `<= 1.15× TPOT_floor(64K)`）；4-bit KVQuant 体积 **`>= 3.0×`** 压缩且 `64K` 解码快于 `BF16`、预填差 `< 10%`；**`100%` `0-ULP`**；`0` Swap/`0` Jetsam；`prefix=2048` `100%` 命中 | 混合衰减 `<= 5%`；KVQuant 预填差 `< 4%` |
| **Dim F** | `--suite multiturn`<br>`multiturn-short-chat.json` | `B=1` 10 轮与 `B=4` 6 波命中率 **`>= 90%`**；重预填削减 **`>= 1.7×`**；AR 10 轮内存漂移 **`<= 32 MB`**（`0` pinned 驱逐、`0` 孤儿块）；GPU 快照/恢复 **`<= 2.0 ms`**；`10/10` & `24/24` **`0-ULP`**；`dim_e` 单轮回归 `<= 1.0%` | 命中 `>= 92%`；AR 漂移 `<= 1 MB`（`miv` `+0.02 MB`）；保存 `<= 0.5 ms`、恢复 `<= 1.6 ms` |

### 4.3 分维度负载与细则

- **Dim A（`prefill_sweep`：`[128, 512, 1024, 4096, 8192]`，`max_new_tokens=1`）**：报 `ttft_ms`、`prefill_gpu_ms`、`prefill_tok_per_s`、`peak_phys_footprint_gb` 及 `L >= 2048` 的 `24 MHz` 子层 `% of P_NAX`。稀疏 MoE 须实现零主机同步 GPU 流式分桶（如 `compact32` 4-SG）并报 `0` `gather_f16` 派发；采用流式分桶预填（大内存 Dense `2048` / MoE `4096` 行，紧凑内存 `512–1024` 行）保持 `L=128..8192` 非 KV 工作区峰值恒定；短提示词自动切小块（如 `32×32 SK=1`），`SK=1` 路径 `bf16->f16` / `xsum` / reduce 派发数均为 `0`。
- **Dim B（`single_decode`：Coding 提示词，`max_new_tokens=256`，测 `TDom=1` `t_domain_ar` 与 `TDom=0` `coupled_ar`）**：物理线速下界
  $$\text{TPOT}_{\text{floor}} = \frac{W_{\text{active}} + 2 S_{\text{recurrent}} + D_{\text{KV}}(L_{\text{kv}})}{B_{\text{DRAM}}}$$
  （纯全注意力 $S_{\text{recurrent}} = 0$；稀疏 MoE $W_{\text{active}}$ 为常驻层加当步激活专家权重）。具备双缓冲异步 ICB 乒乓流水（`ar_pipe_[2]`）；稀疏 MoE 须报全网融合派发（如 3-dispatch 专家融合）相对冷 DRAM 单算子探针的带宽比。
- **Dim C（`single_decode`：覆盖 `coding` 与 `reasoning`，`max_new_tokens=256`，声明投机时强制）**：净加速比
  $$S_{\text{spec}}(K) = \frac{\mathbb{E}[\tau(K)] \cdot \text{TPOT}_{\text{PureAR}}}{T_{\text{round}}(K)}, \qquad T_{\text{round}}(K) = T_{\text{draft}} + T_{\text{verify}} + T_{\text{select}} + T_{\text{commit}} + T_{\text{glue}}$$
  分母严格为通过 Dim B 的 `PureAR`（禁止以 `coupled_ar` 充当分母）；`T_select + T_commit + T_glue < 5%`，单轮主机读回百字节量级；低接受率支持在线裁剪或回退纯 AR，且暖会话新请求无重复 `K=0` 冷测速轮。
- **Dim D（`batch_single` / `batch_sweep`：`B ∈ {1, 2, 4}` 及声明支持的 `B=8`，`max_new_tokens=128`，覆盖 AR 与投机）**：JSON 以 `batch_size` / `max_concurrency` 记录并发槽位（对应 `miv` 的 **`B`**、上游 `ninfer` 的 **`C`**），以 `max_context` 区分上下文长度 **`L`**。要求单次读权重多槽打包推进（`BM=16/32` NAX 或 `M=2` `KL32-BM2`）、共享 `KvPagePool` / `SharedKvArenaPool` CoW 页表，多流纯 AR 每步压平为 `1` 个同步 Compute ICB（含原位采样、`0 B` logits 散射拷贝）。
- **Dim E（三项子负载）**：① `longctx_uncontaminated_footprint`（`prefix_cache_bytes=0`，`[2048, 8192, 32768, 64512]`，`max_new_tokens=64`）；② `longctx_decode_scaling`（`2K/8K/32K/64K` 前缀 + `32`-tok 尾，`bf16` 与 `affine4_g128` 的 AR + 投机）；③ `prefix_cache_bench`（`2048`-tok 共享前缀 + `32`-tok 尾，`B=1` 与 `B=4 CoW`）。全程满足 UMA 准入不等式
  $$W_{\text{shared}}(\text{pretiled}) + B \times S_{\text{slot}}(L) + P_{\text{cache}} \le W_{\text{ws}} - H_{\text{os}}$$
  关闭前缀缓存时额外缓存驻留 `0 B`；`B=4` `prefix_shared_hits = 3`、`0` 固定前缀驱逐；多流 `B >= 2` 必须使用并行分片 KVQuant 注意力核。
- **Dim F（`build_multiturn_specs` 两项负载）**：① `multiturn_b1`（`B=1` 10 轮递进短对话，`256`-tok `pinned` 系统提示词，逐轮拼真实回复 `A_{k-1}` 与 `Q_k` 跨 `1024` 至 `~1,730` tok，`4 GiB` 预算）；② `multiturn_b4`（`B=4` × 6 波，`512`-tok `pinned` 系统提示词，跨 `1024` 至 `~1,266` tok/路，`gen_len=64`）。要求任意长度（`< 1024` 与跨 `1024`）精确命中、`B=4` Wave 1 `prefix_shared_hits == B - 1`、`supersede_entry_id` 原地 `1×` 覆写、增长型草稿特征按 `32 MiB` 量子对齐且释放旧块、`0` CPU `memcpy` 与 `0` 软件 SHA-256。

---

## 5. Gate 4 — 数值质量、长稳、代码卫生与证据包（补充强制项，待确认）

Gate 4 补足吞吐与自洽 `0-ULP` 之外的四项生产盲区：系统性精度偏移、长时浸泡泄漏与崩溃残留、死配置/死 PSO 污染、证据包复核。

### 5.1 数值质量与长上下文召回

| 检查项 | 方法 | 建议硬门限 |
|---|---|---|
| **源精度 PPL / KL** | 与上游 FP16/BF16 参考（或同权重官方实现）在 WikiText-2 / C4 + 中英代码集对比 | 同权重无损重排/核：`KL <= 1e-4`、`ΔPPL <= 0.02`；新量化 / KVQuant（如 `affine4_g128` vs `BF16` KV）：`ΔPPL <= 0.05` |
| **长上下文召回（NIAH / RULER）** | 在最大声明上下文（`32K`/`64K`+）对比 `BF16` KV 与 KVQuant | KVQuant 召回折损 **`<= 2.0` pp** |
| **长序列零数值异常** | 连续生成 `>= 4,096` 步（短/长上下文 + 高温随机采样） | **`0` NaN、`0` Inf** |

### 5.2 稳定性、浸泡与洁净动态库

| 检查项 | 规程 | 建议硬门限 |
|---|---|---|
| **万步 ICB 重放金丝雀** | `omlx_m5_inspect_icb_canary` 连续重放 `>= 10,000` 步 | `100%` 逐位一致、`0` Metal API/Shader Validation 错误、`0` GPU Hang/Reset |
| **100 轮滚动浸泡压测** | 连续 `100` 轮会话（新建、滚动续写、主动 `cancel`、跨 `1024`/`2048` 边界） | 第 10→100 轮 `phys_footprint` 漂移 **`<= 1%` 且 `<= 64 MB`**；结束时 `KvPagePool` 空闲页 `100%` 回弹、`0` 孤儿 Slab |
| **混合 OOM / Jetsam 并发压测** | 同机 `1` 个 Stock MLX + `1..2` 个 Sidecar，压至 `0.85 × W_cap` 与 `0.95 × W_cap` | `trim_cache <= 100 ms` 生效；**`0` Jetsam、`0` 死锁、`0` 跨模型污染** |
| **异常退出与孤儿清理** | 对宿主父进程发 `SIGTERM` 与 `SIGKILL` | Sidecar **`<= 2.0 s`** 内自退；**`0` 孤儿进程、`0` GPU 残留、`0` `/tmp` 残留 `.sock`** |
| **Clean-Mac `otool -L` 审计** | 对包内全部 Mach-O 执行 `otool -L`（对齐 `miv/tools/build_m5_release.py`） | 仅允许 Apple 系统框架（`Metal`、`Foundation`、`CoreFoundation`、`IOKit`、`libc++.1`、`libSystem.B`、`libobjc.A`）；禁止 `libmlx`、`libpython`、`/opt/homebrew/`、`/usr/local/`、`/Applications/Xcode`、`/Users/` |

### 5.3 代码、二进制与 ABI 卫生

- **零死调优字段**：`SiliconTuningConfig` 与校准表 JSON 中每个声明字段必须在运行时读取并生效，并附至少 1 个断言其改变派发的单元测试。
- **零死实验 PSO**：未过 Gate 2.2（`>= +10%–12%`）的实验核、死模板与废弃环境变量合入前物理删除；`-O3 -ffast-math` 仅白名单用于经数值验证的 `.metal` 文件，其余保持 `-fno-fast-math`。
- **确定性执行身份**：完整暴露 `ExecutionIdentityFields`（`execution_source_id`、遍历全部 `.metallib` 有序 SHA-256 的 `kernel_bundle_id`、`payload_verified == true` 的 `artifact_digest`、`sampling_contract_id`、`kv_dtype`、`prefix_mode`）。
- **干净构建与全量 CTest**：单条 CMake 命令从零构建零编译警告、无构建期隐式 `import mlx` 耦合，`ctest --output-on-failure` `100%` 通过（含 Fake Sidecar SPI、取消与首 Token 回退测试）。

### 5.4 五文件机读证据包

每次 Path A PR 或 Path B 认证申请随提交附上 `certification-evidence/<core-id>-<version>-<sku>/`，全部 5 个文件顶层绑定一致的 `git_commit`、`core_version`、`core_abi_version`、`kernel_bundle_id` 与 `hardware_fingerprint`：

```text
certification-evidence/<core-id>-<version>-<sku>/
├── h0-capability.json             # Stage B H0 能力、DMA 带宽、P_NAX、I_ridge、24 MHz 计数器与万步 ICB 校验
├── h1-calibration-<sku>.json      # 目标 SKU H1 标定、冷 DRAM 探针净增益与 prior_vs_measured_diff
├── dim-a-f-bench.json             # 净室 Dim A–E + Dim F (--suite multiturn) 汇总、24 MHz 占比与 phys_footprint
├── parity-report.json             # 0-ULP 报告 (Dim C/D/E/F)、源模型 PPL/KL、NIAH/RULER 召回与 0 NaN/Inf
└── otool-audit.txt                # 全部 Mach-O otool -L 清单、metal-objdump 0 alloca 摘要、100 轮浸泡与 CTest 结果
```

CI 自动校验 5 文件齐全、指纹一致并重算 Gate 1–Gate 4 门限；任一项未通过即阻断合并。

---

## 6. 认证分级与待确认清单（User Sign-Off）

### 6.1 两级徽章

- **`Tier 1 — SPI & Correctness Preview`**：通过 Gate 1（SPI v2 共存）、Gate 4.2（`otool -L` 与崩溃清理）及 `0-ULP` / 数值正确性检查，仅作实验预览，不授硅片认证标识。
- **`Tier 2 — Silicon-Squeezed Certified Core`**：Gate 1–Gate 4（含 5 文件证据包）`100%` 通过，且在至少 `1` 款真实 M5+ SKU 取得 `measured_qualified` 实测标定，可进入官方支持列表与认证徽章。

### 6.2 待用户确认清单（确认后由 `v1.0-draft` 锁定为 `v1.0`）

| 序号 | 评审项 | 默认方案 | 确认选项 |
|---:|---|---|---|
| **C1** | **Gate 4.1 源精度 PPL/KL 与 NIAH/RULER 召回** | 设为 `Tier 2` **Hard Gate**（无损核 `ΔPPL <= 0.02`；KVQuant `ΔPPL <= 0.05` 且 NIAH/RULER 折损 `<= 2.0 pp`） | `[ ]` 采纳 Hard Gate / `[ ]` 调整阈值 / `[ ]` 降为 Advisory |
| **C2** | **Gate 4.2 100 轮浸泡与混合 Jetsam 并发压测** | 设为 **Hard Gate**（100 轮漂移 `<= 1%` 且 `<= 64 MB`；并发 `trim_cache <= 100 ms` 零 Jetsam；`SIGKILL <= 2 s` 零孤儿） | `[ ]` 采纳 / `[ ]` 调整轮次或阈值 |
| **C3** | **Gate 4.2 `otool -L` 零 MLX / 零 Python 依赖** | 运行期全部 Mach-O 严格 0 MLX / 0 Python；Path A 同步清理遗留构建期 `import mlx` 探针，做到构建与运行双洁净 | `[ ]` 采纳双洁净 / `[ ]` 仅限制运行期 |
| **C4** | **Gate 4.3 零死调优字段与零死 PSO** | 设为 CI **Hard Gate**（每个配置参数须有单测验证生效，未过 `+10%–12%` 冷 DRAM 探针的实验 PSO 合入前物理删除） | `[ ]` 采纳 / `[ ]` 仅人工 Code Review |
| **C5** | **Gate 3 Dim A / B / D 的 Dense vs MoE 分档** | Dense 按 `>= 85%` $P_{\text{NAX}}$（Prefill）与 `>= 90%` $B_{\text{DRAM}}$（`T=1`）硬卡；稀疏 MoE 结合 $W_{\text{active}}$ 与冷 DRAM 分桶/3-dispatch 探针按拓扑核验 | `[ ]` 采纳分档 / `[ ]` 为 MoE 设独立硬下限百分比 |
| **C6** | **Gate 4.4 五文件证据包强制提交** | 强制要求 `h0-capability.json`、`h1-calibration-<sku>.json`、`dim-a-f-bench.json`、`parity-report.json`、`otool-audit.txt` 绑定 `kernel_bundle_id` | `[ ]` 采纳 / `[ ]` 增减必选文件 |
