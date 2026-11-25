# NInfer/CUDA 与 Apple Metal 4/NAX：硬件、软件栈及实机标定边界

> **版本**：`v2.0`（原始 64 核 96GB Stage A 标定基线；80 核 256GB 实测校准值见 [`../backend/silicon-calibrations/apple-m5-ultra-80gpu.json`](../backend/silicon-calibrations/apple-m5-ultra-80gpu.json)）
> **对照对象**：上游 `ninfer` `ebdc67e5`（GeForce RTX 5090 / `sm_120a`）vs 本仓库 `ninfer-metal`（`miv/`）在物理 **Apple M5 Ultra 64 核 GPU / 96 GiB（`apple-m5-ultra-64gpu`）** 与 **Apple M5 Ultra 80 核 GPU / 256 GiB（`apple-m5-ultra-80gpu`）** 上的硬件/软件边界。
> **关联权威文档**：[`m5-ultra-porting-foundations.md`](m5-ultra-porting-foundations.md)、[`../silicon-guides/m5-series-optimization-playbook.md`](../silicon-guides/m5-series-optimization-playbook.md)、[`../backend/ninfer-m5-porting-architecture-v2.md`](../backend/ninfer-m5-porting-architecture-v2.md)。

---

## 1. 已确认的物理硬件与软件栈硬边界

1. **Apple M5 Ultra（`MTLGPUFamilyApple10` / `applegpu_g17d` / `AGXAcceleratorG17X`）**：
   - 官方 Mac Studio 规格提供 64 核与 80 核 GPU 两档，共享 Quad-die UltraFusion 互联（标称带宽 `1.2 TB/s`），统一内存可选 `96 / 256 / 512 GiB`。[A1][A2]
   - **64 核 96 GiB 真机实测核实（`apple-m5-ultra-64gpu`，macOS `27.2.1` Build `26B131`）**：30 核 CPU（10P+20E）、64 核 GPU / 64 NAX；物理内存 `96 GiB`（`103,079,215,104 B`），Metal 推荐工作集 **`77.76 GiB`**（`83,494,174,720 B` = `81.0%` RAM），单 `MTLBuffer` 上限 `58.32 GiB`；`24 MHz` Metal 4 硬件计数器实测 `256 MiB` Shared Blit DMA 峰值带宽达 **`1095.19 GB/s` p50**（标称带宽的 **`91.3%`**），独立 `[5120×5120]` Q4 NAX `M=256` 达 **`36.68 TFLOPS`**（校准文件见 [`../backend/silicon-calibrations/apple-m5-ultra-64gpu.json`](../backend/silicon-calibrations/apple-m5-ultra-64gpu.json)）。
   - **80 核 256 GiB 真机实测核实（`apple-m5-ultra-80gpu`，macOS `27.2.1` Build `26B131`）**：36 核 CPU（12P+24E）、80 核 GPU / 80 NAX、2 × 16 核 `h17g` ANE；物理内存 `256 GiB`（推荐工作集 `240 GiB`），持续冷 DRAM Compute-Copy **`992.1 GB/s`**、MXFP4 `lm_head` **`967.9 GB/s`**、NAX MLP 峰值 **`118.4 TFLOPS`**（校准文件见 [`../backend/silicon-calibrations/apple-m5-ultra-80gpu.json`](../backend/silicon-calibrations/apple-m5-ultra-80gpu.json)）。
2. **Compute ICB 4 GiB 静默偏移截断红线（Stage A 真机发现）**：
   - Apple Silicon 在通过 `MTLIndirectComputeCommand` 编码缓冲偏移 `offset >= 4 GiB`（`4,294,963,200 B`）时发生 **32-bit 无符号静默回绕截断**且 Metal API/Shader Validation 不报错。`ninfer-metal`（`miv/`）强制将不可变权重段钳定为 **`<= 2 GiB`**，在线重排临时段钳定为 **`<= 1 GiB`**，并严禁多子图融合时混用 `Encoder::dynamic_parameters` 常量去重缓存。
3. **32 KiB 线程组共享内存硬上限 vs NVIDIA RTX 5090 (`sm_120a`)**：
   - RTX 5090 具备 `32 GB` GDDR7（标称 `1,792 GB/s`）及每 SM `100 KB` / 每 Block `99 KB` Shared Memory [N1][N2][N3]；Apple9/Apple10 每线程组共享内存上限恒为 **`32 KiB`（`32,768 B`）** [A4]。超限方案（如 `TB16/WY64` 的 `40–52 KiB`）在 M5 上永久淘汰；在 Apple10 `matmul2d` 中，`SK > 1` 时必须钳定 `RF = 1`，`SK <= 2` 可启用 `NSG = 2`（`2×BN16`），`SK >= 4` 保持 `NSG = 1` 以守住驻留度。
4. **原生量化张量与零 MLX 生产 `.metallib` 库**：
   - macOS 27 SDK 原生提供 `MTLTensorDataTypeMetalFloat4E2M1`、`MetalFloat8UE8M0`、`int4b_format`、`uint4b/uint2b` 多平面视图及 `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` `matmul2d` [A3]。在 `miv-v0.01` 中，全套 **38 个纯 Metal 4 `.metallib`**（含 5 个生产 NAX 库）实现零 MLX 构建解耦，已在 64 核与 80 核 Apple10 真机通过完整 27B/35B 推理与 `0-ULP` 奇偶验证。

---

## 2. NInfer (`ebdc67e5` / RTX 5090) vs `ninfer-metal` (`miv` / Apple10) 栈对比表

| 架构维度 | NInfer (`ebdc67e5`) / RTX 5090 / CUDA | `ninfer-metal` (`miv`) / Apple10 Metal 4 & NAX | Apple 路线关键设计决策与真机证据 |
|---|---|---|---|
| **API 与矩阵硬件** | CUDA + 第五代 Tensor Cores (`mma.sync`) | Metal 4 直接提交 + GPU 内置 Neural Accelerators (`mpp::tensor_ops::matmul2d`) + SIMD ALU | 保持同一 GPU 管线内 SIMD/NAX 协同；外部 ANE（实测 Dual-ANE `27.65 TOPS` / `298 µs` relay / `-13.2%` UMA 带宽争抢）不混入低延迟 token 循环 [A5][N2] |
| **内存拓扑与预重排** | `32 GB` 独立显存，MoE 需做 PCIe 双窗口专家卸载 | `96–512 GiB` 零拷贝统一内存（UMA）；离线预重排 `mx_nax_tiled_bk128_v1` / `oq_nax_tiled_g{64,128}_v1` | 跳过 PCIe 专家卸载；离线预重排消除双份驻留（27B-MXFP4 `27→14 GB`，oQ4e `32.8→18.4 GB`） |
| **量化表示体系** | NVFP4（G16 `E2M1+E4M3` + 全局尺度）+ groupwise int + QDA | MXFP4（G32 `E2M1+UE8M0`）+ oQ Affine `q3/q4/q5/q6/q8 × g64/g128` + `qda-search` (`3.55 bpw`) | 离线转换保留原数学合同；`q3` 双平面（`24 B / 64w`）与 QDA 已在 Apple10 真机逐位验通 |
| **线性与 MoE 路由** | 按 shape/T 细分 SIMT/MMA/W4A4 | `T=1` 可选 `TDom=1` 零派发融合 QMV；耦合模式统一 `BM=8/16/32` NAX；宽块 `64×64 SK=1`；`moe9_bt` / `moe_nax` | 80 核 35B `B=8` 达 **`846.29 tok/s`**、27B 1K 预填 **`100.3–102.8 TFLOPS`**；64 核 35B `4K` 预填 **`10,048.5 tok/s`** |
| **图捕获与批提交** | CUDA Graph 捕获 + 紧凑请求批处理 | Metal 4 参数表 + Compute ICB（`<=2 GiB` 分段 + 子图清空 `dynamic_parameters`）+ `run_packed_batch` + `replay_icb_batch` | 单 ICB 融合草稿 + 批投机（80 核 27B-oQ4e `B=4 DFlash2` **`500.57 tok/s`**，`B=4 MTP` **`434.92 tok/s`**） |
| **Attention 与压缩 KV** | FlashAttention + `#4659`/`FOSCAR` 4-bit KV 页池 | Split-KV (`L>=2048`) + 2-SG 硬件 `uint4b` NAX `affine4_g128` + `Frag16x16` Flash-Prefill (`61.0–69.7 TF`) + `BF16 Frag48` | `64K` `affine4_g128` AR 稳居 **`58.92 tok/s`**（仅占 **`15.69 GiB`** 净峰值），`B=4` 批注意力提速 `+102.4%` |
| **GDN 线性注意力** | 融合门控与块并行递推 | 删死 `151 MiB/slot` `.scratch` + `kDvSlice=32` 满波次 `C=64` 分块 NAX + `QwenUmaContinuation` 单 CB 快照 | 64 核 GDN 预填达 **`78.5–79.4 TF`**；48 层 GDN 状态 UMA 保存仅 `~0.22 ms`、热恢复 `0.73–1.62 ms` |
| **前缀缓存与防卡顿** | 引用计数前缀树（`3b76b98c` 二次方成本）+ 解码保护分块 | 全局跨槽 CoW `SharedPrefixCache` + `rolling_uma_v1` 原地覆写 + `#5164` 页对齐解码保护与 `set_scheduler_hint` | `64K` 热启动 TTFT **`176–201 ms`**（`7×–10.6×` vs Stock oMLX）；多轮短对话 `92.3%` 命中率且 `T1→T10` 仅 `+0.02 MB` |
| **硬件可观测性** | Nsight Compute / CUDA Events | `omlx_m5_hw_counter_probe`（`MTLCounterSampleBuffer` + `MTL4CounterHeap` `24 MHz` 时间戳，禁用开销 `0.00042 ns/call`） | 无 GUI SSH 下完成纳秒级子层带宽/算力剖析，正式闭合 64 核与 80 核 `h0_complete == true` |

---

## 3. C01–C13 标定边界账本：64 核 96GB 基线 vs 80 核 256GB 实测闭环

| 边界 ID | 标定问题 | M5 Ultra 64 核 96GB 基线（`apple-m5-ultra-64gpu`） | M5 Ultra 80 核 256GB 实测状态（`apple-m5-ultra-80gpu`） |
|---|---|---|---|
| **C01 能力** | 原生 MXFP4 / Affine / BF16 / `int4b` NAX、Compute ICB、`24 MHz` 计数器可用性 | **已闭环**：`2 GiB` 权重分段规避 4 GiB ICB 截断；万步 ICB 逐位一致；`h0_complete=true` | **已闭环**：80 核 `AGXAcceleratorG17X` + Dual-ANE (`2×16c h17g`) 身份与硬件计数器基线核验通过 |
| **C02 内存预算** | 推荐工作集、预重排权重、循环特征环、`SharedPrimary` 与 `Affine4` KV 账本 | **已闭环**：`WS = 77.76 GiB`；MXFP4 `14.0G` / oQ4e `18.4G`；`Affine4` 下 `B=8@64K` `24.5G`、`B=4@128K` `23.8G` | **已闭环**：`WS = 240 GiB`；`64K` `affine4` 净峰值 `15.69 GiB`、四档共存缓存峰值 `20.80 GiB` |
| **C03 带宽** | 物理峰值 DMA 与各子层真实持续有效带宽 | **已闭环**：DMA 峰值 **`1095.2 GB/s`**；`lm_head` **`890 GB/s`**；27B `TDom=1` MLP `881.4 GB/s` | **已闭环**：持续冷 DRAM Compute-Copy **`992.1 GB/s`**；`lm_head` **`967.9 GB/s`**；27B `TDom=1` MLP `933 GB/s` |
| **C04 矩阵交点** | `T=1..1024` SIMD vs NAX、`W4A16` vs `W4A4`、`SK`/`RF`/`NSG` 交面 | **已闭环**：纯 AR `T=1` 选融合 QMV；耦合/投机统一 `BM=8/16/32 SK=8`；宽块 `M>=64` 锁 `SK=1` | **已闭环**：`BM=16/32 SK=8` 验证与 `Q5 q5_h1` 零 `uitofp` 在 80 核达成 `19/19` 全胜 |
| **C05 片上资源** | `32 KiB` TG 上限、寄存器压力与单核驻留度 | **已闭环**：淘汰 `TB16/WY64`；`<=104` VGPR 保 `3` TG/核；`0` 用户 `alloca` | 硬件每核 32 KiB 不变，全套新增核经 `metal-objdump -d` 审计保持 `0` 溢出 |
| **C06 提交与同步** | 编码器屏障间隙、单 ICB 草稿、并行冷加载与 `32 B` 读回 | **已闭环**：屏障间隙 `p50 = 416.7 ns`；冷加载 **`1.84 s`**；双 `MTL4CommandAllocator` 异步乒乓 | **已闭环**：`B=1..8` 单 CB `replay_icb_batch` + 在片 `PositionKeyedSampler` 保持 `0` 主机同步回退 |
| **C07 Attention** | Split-KV 阈值/分片、GQA 门控、`Flash-Prefill` 与 `FOSCAR` Affine4 | **已闭环**：`Split-KV = 2048` (`12` 路)（`32K` `44.5×`）；`total_tgs>=64` 护栏 | **已闭环**：`Split-KV = 2048` (`16` 路)；`61.0 TF` `affine4` & `69.7 TF` `bf16` Flash-Prefill；批 `affine4` NAX 提速 `+102.4%` |
| **C08 GDN** | 删死 `.scratch`、2D 卷积与 `kDvSlice=32` `C=64` 分块 NAX | **已闭环**：省 `151 MiB/slot`；`Hv×4` 满波次达 `78.5–79.4 TF` | **已闭环**：结合 `QwenUmaContinuation` 实现 `0.22 ms` 异步解码后状态快照 |
| **C09 页与前缀** | CoW 页池、全局 `SharedPrefixCache` 与任意长度滚动复用 | **已闭环**：`B=4` 前缀命中 `100%` | **已闭环**：`2K..64K` `100%` 零淘汰命中（`0.73–1.62 ms` 恢复）；多轮短对话 `92.33%` 滚动复用 |
| **C10 并发调度** | `run_packed_batch`、`AdaptiveSpecController` 与 `#5164` 防卡顿 | **已闭环**：27B-oQ4e `B=4` `168.08 tok/s`；35B `B=8` `463.46 tok/s`；`24.53 ms` p50 ITL | **已闭环**：27B-oQ4e `B=4 DF2` **`500.57 tok/s`**；`B=4 MTP` **`434.92 tok/s`**；35B `B=8 AR` **`846.29 tok/s`** |
| **C11 MoE** | 真实 35B 256 专家分桶零重排 `moe_nax` 与 `moe9_bt` | **已闭环**：`moe_prefill_nax_min_tokens = 8`；`4K` 预填 `10,048.5 tok/s` | **已闭环**：`3`-dispatch `moe9_bt`（`B=2..8`）将 MoE 派发从 `560` 压至 `120`，`B=8` 达 `846.29 tok/s` |
| **C12 精度与奇偶** | `position_coupled_v1` 下 `AR == DFlash2 == MTP == Batch`、QDA/`q3` 与 `Affine4` | **已闭环**：`6/6` `256/256` `0-ULP` 与 QDA/`q3` 逐位一致 | **已闭环**：`B=1..4` `128/128`、`2K..64K` `affine4` `128/128`、多轮 `B=1` `10/10` & `B=4` `24/24` 全量 `0-ULP` |
| **C13 长稳与运维** | 连续压测无泄漏、Sidecar 共存自愈、免 Xcode Tarball 净室安装 | **已闭环**：净室无 Xcode 运行通过 | **已闭环**：`oMLX-extend` Provider SPI v2（`426/426` pytest）+ `otool -L` 纯系统框架审计 + `miv-v0.01` 打包验证 |

---

## 4. 跨硬件与同机对照使用规则

1. **机制同构而非参数照搬**：跨 CUDA (`sm_120a`) 与 Metal 4 (`applegpu_g17d`) 仅复用状态所有权、单次权重读取、零解包折叠与缓存淘汰机制，绝不照搬 block 大小、shared memory 尺寸或显存卸载阈值。
2. **严格隔离各 SKU 校准档案**：`apple-m5-ultra-64gpu.json` 与 `apple-m5-ultra-80gpu.json` 严格按 `profile_id` 隔离，禁止跨核数/跨代 SKU 混标。
3. **同机对照公平性（[`../backend/tensorfold-benchmark-recipe-v1.json`](../backend/tensorfold-benchmark-recipe-v1.json)）**：同 checkpoint、同量化、同输入 token 序列、同采样/数值合同分组报告；显式精度门控项（`W4A4`、`FlashNaxV1`、`affine4_g128`）单独标注合同身份。

---

## 5. 参考来源

| ID | 来源与定位 | 地址 / 仓库路径 |
|---|---|---|
| A1–A6 | Apple Mac Studio M5 Ultra 官方规格、WWDC25/26 Metal 4 & MPP TensorOps、Metal Feature Set Tables (`32 KB` TG 上限) | `https://www.apple.com/mac-studio/specs/`；`https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf` |
| N1–N6 | NVIDIA GeForce RTX 5090 规格、Blackwell (`sm_120a`) 调优指南、NVFP4 格式与 CUDA Graph 文档 | `https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html` |
| L1 | 本仓量化与 NInfer 移植架构规范 | [`../backend/quantization-kernel-strategy-v1.md`](../backend/quantization-kernel-strategy-v1.md)；[`../backend/ninfer-m5-porting-architecture-v2.md`](../backend/ninfer-m5-porting-architecture-v2.md) |
| L2 | Metal 4 / NAX 运行时、硬件计数器与硅片识别器 | `miv/src/metal.mm`；`miv/src/hw_counters.mm`；`miv/tools/m5_silicon_profile.py`；`miv/tools/m5_target_guard.py` |
