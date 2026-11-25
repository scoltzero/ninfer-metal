# Apple M5 全系硅片推理优化与校准实战指南（`m5-series-optimization-playbook`）

> **状态**：基于 **Apple M5 Ultra 64 核 GPU / 96 GiB（`applegpu_g17d` / Family 10 / macOS 27.2.1）** 历经 `v0.8.0-m5rt.1` → `v0.8.0-m5rt.4` 四轮实机验证提炼的完整工程定律（6 条基础定律 + 4 条 `m5rt.4` 真机深化定律）与全系调优手册。
> **配套工具**：[`tools/m5_silicon_profile.py`](../../tools/m5_silicon_profile.py)、[`docs/backend/silicon-calibrations/apple-m5-ultra-64gpu.json`](../../docs/backend/silicon-calibrations/apple-m5-ultra-64gpu.json)、[`docs/backend/m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json`](../../docs/backend/m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json)

---

## 1. Apple10 (M5) 架构核心硬件不变量与真机硬红线

在 M5 系列 GPU（`MTLGPUFamilyApple10`）上编写 Metal 4、Compute ICB 与 `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>`（NAX）推理算子，必须遵守以下五条经真机验证的物理硬边界：

### 1.1 `MTLIndirectComputeCommand` 2 GiB 权重分段红线（防 4 GiB 静默截断）
- **实测陷阱**：即使 macOS 27 在 96 GiB M5 Ultra 上报告单 `MTLBuffer` 上限 `max_buffer_length = 58.32 GiB`（`62,620,631,040 B`），Apple Silicon 硬件在通过 **Compute ICB（`MTLIndirectComputeCommand`）** 编码缓冲地址偏移时，对 **`offset >= 4 GiB`（`4,294,963,200 B`）会发生 32-bit 无符号整数静默回绕截断**，且 **Metal API Validation 与 Shader Validation 均不报错**，直接导致 14–20 GiB 模型后半段层输出错乱。
- **强制工程准则**：
  1. 不可变模型权重池单段容量硬钳定为 **`2 GiB`**（`icb_safe_weight_segment_bytes = 2,147,483,648 B`，见 `m5rt/src/materialization.cpp`）。
  2. 在线 NAX 重排临时段硬钳定为 **`1 GiB`**（见 `m5rt/src/mxfp4_nax_cache.h`、`affine_nax_cache.h`）。
  3. 任何被纳入 `MetalIcbProfile` 录制的 KV 页池或工作区子视图，绑定偏移严禁跨越 `4 GiB` 边界；大上下文多槽缓冲优先按槽位拆分独立 `MTLBuffer` 或保证段内偏移 `< 2 GiB`。

### 1.2 32 KiB Threadgroup Memory 上限与 NAX Split-K 占用率公式
- **硬件常数**：Apple9 与 Apple10 的每线程组共享内存上限同为 **`32 KiB`（`32,768 B`）**，`thread_execution_width = 32`，`max_threads_per_threadgroup = 1024`。任何超过 `32 KiB` 的静态+动态共享内存核（如早期实验中的 GDN `TB16/WY64` 需 `40–52 KiB`）在 M5 上同样非法，绝不可寄希望于“换大芯片自动跑通”。
- **NAX `matmul2d` 共享内存精确公式**（推导自 `m5rt/kernels/mxfp4_nax.metal` 与 `bf16_nax.metal`）：
  对于 `execution_simdgroup` 模式下的 `matmul2d_descriptor(BM, BN, BK)`，每个 lane 持有的累加器寄存器容量为 `kCap = (16 * BN) / 32 = BN / 2` 个 `float`。当采用 `SK` 路 Split-K（每线程组含 `SK` 个 simdgroup，共 `32 * SK` 线程，`sg = 1..SK-1` 通过 threadgroup memory 向 `sg = 0` 归约）时：
  $$\text{TG\_Bytes}_{\text{linear}} = (SK - 1) \times \frac{BN}{2} \times 32 \times 4\text{ B} = (SK - 1) \times BN \times 64\text{ B}$$
  $$\text{TG\_Bytes}_{\text{SwiGLU}} = 2 \times \text{TG\_Bytes}_{\text{linear}} = (SK - 1) \times BN \times 128\text{ B}$$

| 算子与适用场景 | Tile 配置 (`BM, BN, BK, SK`) | 线程数/TG | Linear TG 内存 (驻留上限) | Fused SwiGLU TG 内存 (驻留上限) | 64 核实测结论与物理原因 |
|---|---|---:|---|---|---|
| **BF16 NAX（DFlash2 5 层骨干）** | **`BM=8, BN=16, BK=128, SK=4`** | `128` | **`3.0 KiB`** (`10` TGs/core) | **`6.0 KiB`** (`5` TGs/core) | **最优**：高驻留隐藏 BF16 访存延迟，草稿 7 步耗时压至 **`7.88 ms`**，全面击败 `SK=8` 与 `BN=32/64` |
| **MXFP4 / Affine NAX（骨干投影 `N≤17408`）** | **`BM=8, BN=16, BK=128, SK=8`** | `256` | **`7.0 KiB`** (`4` TGs/core) | **`14.0 KiB`** (`2` TGs/core) | **最优**：4-bit 权重算术强度低且 `N/16` 网格较小（如 `N=1024` 仅 64 个输出块），需 `SK=8` 填满 64/80 核并行度 |
| **超大词表投影（`lm_head`, `N=248,320`）** | **`BM=8, BN=32, BK=128, SK=2`** | `64` | **`2.0 KiB`** (`16` TGs/core) | — | **最优**：`N/32 = 7,760` 个输出块已远超 GPU 核数，`SK=2` 最小化归约屏障，实测有效带宽达 **`882–890 GB/s`** |
| *反例：过宽 SwiGLU Tile* | *`BM=8, BN=32, BK=128, SK=8`* | *`256`* | *`14.0 KiB` (`2` TGs/core)* | ***`28.0 KiB` (`1` TG/core)*** | **严重退化**：单核仅能驻留 1 个 TG，丧失延迟隐藏；若升至 `BN=64, SK=8` 则达 `56 KiB` 直接编译/校验失败 |

---

## 2. 六大经实测验证的 M5 核心性能优化定律

### 定律 1：单核高带宽不对称性与 GQA Split-KV 阈值前移（`8192 → 2048`）
- **物理机制**：Qwen3.8-27B 仅有 `kv_heads = 4`（Qwen3.6-35B-A3B 仅 `kv_heads = 2`）。在未切分 KV 长度时，单个 `T=1` 解码注意力最多只能按 Query Head 分组发射极少量线程组，导致 64 颗（或 80 颗）GPU 核心绝大部分处于空闲等待状态，`1.2 TB/s` 内存控制器并发队列深度严重不足。
- **64 核实测拐点**（[`h1/h1-a-27b-p0.json`](../../bench/results/m5-ultra-64gpu-96g/h1/h1-a-27b-p0.json)）：
  - `L = 2,048`：`1.3699 ms → 0.3397 ms`（**`4.03×` 提速**）
  - `L = 8,192`：`5.4129 ms → 0.3747 ms`（**`14.44×` 提速**）
  - `L = 32,768`：`26.4580 ms → 0.5941 ms`（**`44.53×` 提速**）
- **跨硅片推论**：GPU 核心数越多、模型的 `kv_heads` 越少，`attention_split_kv_min_context` 必须越早开启。对于 `gpu_cores >= 32`（M5 Max / M5 Ultra）且 `kv_heads <= 8` 的模型，`2048`（甚至 `1024`）即进入正收益区；仅在 10 核 Base M5 上才需推迟至 `8192–16384` 以避免 merge 开销反噬。

### 定律 2：T-Domain 分流准则（`T=1` 合并 SIMD QMV vs `T=2..8` `BM=8` NAX）与精确奇偶权衡
- **单步 `T=1` 物理瓶颈**：`T=1` 矩阵向量乘算术强度仅为 `~2 FLOP/Byte`（MXFP4 约 `4 FLOP/Byte`），完全受限于权重读取带宽。若强行进入 `BM=8` NAX：
  1. 需额外执行一次 `omlx_m5_bf16_to_f16_rowscale` 转换核；
  2. NAX 硬件单元执行 8 行矩阵乘却丢弃 7 行无效输出。
  在 64 核 M5 Ultra 上，这导致 27B-MXFP4 纯 AR 从合并 SIMD QMV（`row_exact v3`，128B `uint + bfloat4` 零 shuffle）的 **`48.92 tok/s`（`20.02 ms/step`）** 降至 **`40.44 tok/s`（`24.43 ms/step`）**。
- **双模式工程策略**：
  - **投机解码模式（`DFlash2` 开启 + `position_coupled_v1` 默认合同）**：关闭 `OMLX_M5_T_DOMAIN_DISPATCH`，令 `T=1` 与 `T=2..8` 统一走 `BM=8` NAX，换取 **100% `AR == DFlash2` 逐位确定性一致**，且 `T=8` Verify 仅需 **`26.53 ms`**（相对 `T=1` 仅慢 `8.6%`），轻松跑出 **`142.49 tok/s`**！
  - **纯自回归模式（`mode = ar` 且无需跨模式耦合复现）**：开启 `OMLX_M5_T_DOMAIN_DISPATCH=1`，`T=1` 直走 `row_exact v3` SIMD QMV 吃满 **`48.92 tok/s`**，`T >= 2` Prefill 自动切入 NAX。

### 定律 3：Prefill 双侧量化交点（`W4A16 BM=8` vs `W4A4 BM=16` vs 分桶 `moe_nax`）
- **Dense MXFP4 `W4A4` 交点**：Apple10 原生支持 `E2M1 × E2M1`（`block_factor_k = 32`）双侧 FP4 `BM=16` NAX `matmul2d`，但需前置运行 `omlx_m5_bf16_to_mxfp4_a4_bm16_bk128` 动态量化激活（含 `simd_max` 与 `E8M0` 向上取整编码）：
  - 当 `T < 32` 时：`BM=16` 行填充浪费与动态量化核派发开销抵消硬件吞吐优势，`W4A16` 更快且精度更高。
  - 当 `T >= 32` 时：算力成为主矛盾，`W4A4` 在 `T=1024`（27B-MXFP4，64 核）上将 TTFT 从 `2131.27 ms` 压至 **`1650.60 ms`（`1.29×` 提速）**。因激活 FP4 量化改变数值合同，生产通过显式门控 `OMLX_M5_ALLOW_W4A4_PREFILL=1` 启用。
- **MoE 256 专家分桶零重排 `moe_nax`（静态 ICB 兼容范式）**：
  - 对 `Qwen3.6-35B-A3B`（256 专家，`top_k=8`），严禁在 CPU 侧读回每个专家的动态 token 数再重录命令。
  - 正确范式：在单线程组 `omlx_m5_moe_nax_bucket_schedule` 内完成 GPU 端计数与 `BM=8` 分桶，按最坏上界 **`max_tiles = T + 256`**（`top_k=8` 且 `BM=8` 时，总有效分配行 `8T` 除以 `8` 为 `T` 个满桶，外加最多 256 个专家各 1 个尾桶）固定发射静态网格，越界桶在核首行 `if (tile_idx >= *total_tiles) return;` 零成本早退。在 `T >= 32` 启用后，64 核 `T=1024` Prefill TTFT 从 `905.20 ms` 骤降至 **`381.71 ms`（`2.37×` 提速）**。

### 定律 4：离线 NAX 预重排消除双份驻留（物理内存立减 **`-48.1%`** 公式）
- **内存翻倍根源**：若 `.omlxm5` 保存行主序 `rowsplit` 权重而运行时 `Mxfp4NaxCache` / `AffineNaxCache` 在 GPU 上重排生成 `Private` tiled buffer，则在 `QwenProgram` 生命周期内会同时常驻原始 `Shared/Immutable` 权重池（供元数据/回退使用）与重排后 `Private` 权重池，总权重内存膨胀为：
  $$M_{\text{untiled\_nax}} \approx M_{\text{artifact}} + M_{\text{2D\_dense\_projections}} \approx 1.92 \times M_{\text{artifact}}$$
  实测 27B-MXFP4（`14.17 GiB`）在 `m5rt.2` 中 `phys_footprint` 高达 **`27.0 GiB`**！
- **零拷贝离线预重排定律**：使用 `tools/m5_artifact.py convert --nax-pretile`（或 `pretile` 子命令）在磁盘离线写好 `mx_nax_tiled_bk128_v1` / `oq_nax_tiled_g64_v1` 布局（自动按 `N >= 65536 ? BN=32 : BN=16` 对齐），运行时直接将 `ModelPlaneBinding` 指向 mmap/上传后的唯一不可变分段池：
  $$M_{\text{pretiled\_nax}} = M_{\text{artifact}} \quad (\text{27B-MXFP4 实测 } \text{phys\_footprint}: \mathbf{27.0\text{ GB} \rightarrow 14.0\text{ GB}, -48.1\%})$$
  **在所有 `<= 96 GiB` 的 M5 设备（M5 Base / Pro / Max / 96GB Ultra）上，离线 NAX 预重排是必选生产配置。**

### 定律 5：单命令缓冲 `replay_icb_batch` 与零拷贝 UMA 硬件 CRC32
- **消除 `O(B)` 主机提交间隙**：Apple Metal 4 虽大幅降低了单次命令编码开销，但在 `B=4` 或 `B=8` 并发下，若按槽位分别 `commit + wait` Draft、Verify 与 Commit，驱动同步气泡会吞噬超过 `40%` 的并发吞吐。通过 `replay_icb_batch` 将 `B` 个活跃槽的同类阶段打包进**单次 Metal 4 命令缓冲提交**，配合 UMA 统一内存指针直接执行 ARMv8 `__crc32cd` 硬件指令计算特征哈希（零 `std::vector` 拷贝），使 64 核 `B=4@32K` 27B-oQ4e DFlash2 吞吐暴涨至 **`143.32 tok/s`（较初值 `9.50×`）**、35B-A3B `B=8@64K` 飙升至 **`247.71 tok/s`（`2.50×`）**。

---

## 3. 统一内存容量准入方程（Unified Memory Admission Model）

设系统总物理内存为 $R\text{ (GiB)}$，H0 实测（或启发式估算）Metal 推荐工作集为 $W_{\text{ws}}$：
- **macOS 27 实测工作集比例**：在 96 GiB M5 Ultra 上实测 $W_{\text{ws}} = 77.76\text{ GiB}$（占物理内存 **`81.0%`**，优于保守先验 `75% = 72 GiB`）；256 GiB 机型预计 $W_{\text{ws}} \approx 192–207\text{ GiB}$。
- **安全可用推理预算**：扣除 macOS 内核、桌面/SSH 守护进程、Metal 管线与 ICB 描述符安全余量 $H_{\text{os}}$（建议 96GB 留 `14–16 GiB`，256GB 留 `20–24 GiB`）：
  $$B_{\text{usable}} = W_{\text{ws}} - H_{\text{os}}$$
- **离线预重排后的并发准入判据**（并发槽位数 $B$，上下文窗口 $L$，前缀缓存预算 $P_{\text{cache}} \le 1.0\text{ GiB}$）：
  $$W_{\text{shared}}(\text{pretiled}) + B \times S_{\text{slot}}(L) + P_{\text{cache}} \le B_{\text{usable}}$$
  其中单槽内存 $S_{\text{slot}}(L)$ 精确值（源自 `m5rt/src/qwen_memory.cpp`）：
  - `Qwen3.8-27B AR`（`W_shared`: MXFP4 `14.17 GiB` / oQ4e `15.81 GiB`）：`32K = 2.85 GiB`，`64K = 4.85 GiB`，`128K = 8.85 GiB`
  - `Qwen3.8-27B + DFlash2`（`+3.58 GiB` 草稿权重，含 5 层目标特征历史 `L × 5 × 5120 × 2B`）：`32K = 4.83 GiB`，`64K = 8.39 GiB`，`128K = 15.52 GiB`
  - `Qwen3.6-35B-A3B MoE AR`（`W_shared`: MXFP4 `17.99 GiB` / oQ4e `20.13 GiB`，GQA `kv_heads=2` 极省 KV）：`32K = 1.32 GiB`，`64K = 1.94 GiB`，`128K = 3.19 GiB`

---

## 4. Apple M5 全系硅片分级调优手册（Base / Pro / Max / Ultra 64c / Ultra 80c）

### 4.1 全家族规格、关键参数与推荐工作负载总表

| 硅片档位 (`profile_id`) | GPU / NAX 核数 | 标称带宽 (相对 80c) | 常见内存档位 | 预重排 `--nax-pretile` | Dense Split-K (`N<65K` / `lm_head`) | Split-KV 起始上下文 / 分片数 | DFlash2 推荐深度 `K` | 主力 27B / 35B 安全并发与定位 |
|---|---:|---|---|:-:|---|---|---:|---|
| **M5 Base**<br>`apple-m5-10gpu` | `10 / 10`<br>(`0.125×`) | `153 GB/s`<br>(`0.13×`) | `16 / 24 / 32 GB` | **强制必开** | `SK=1..2` / `SK=1` | `16K–32K` / `2..4` | `3..4` | `16/24GB` 跑 `≤14B`；`32GB` 仅限预重排 `27B-MXFP4` 单流 `B=1 @ 8K–16K`（`~6–7 tok/s` AR，`~15 tok/s` DFlash2） |
| **M5 Pro**<br>`apple-m5-pro-16gpu`<br>`apple-m5-pro-20gpu` | `16–20`<br>(`0.20–0.25×`) | `307 GB/s`<br>(`0.26×`) | `36 / 48 / 64 GB` | **强制必开** | `SK=2..4` / `SK=1..2` | `8K–16K` / `4` | `4..5` | `36GB` 跑预重排 `27B AR B=1@32K`；`48–64GB` 稳定跑 `27B+DFlash2 B=1..2@32K` 或 `35B MoE B=1..4@32K` |
| **M5 Max**<br>`apple-m5-max-32gpu`<br>`apple-m5-max-40gpu` | `32–40`<br>(`0.40–0.50×`) | `614 GB/s`<br>(`0.51×`) | `64 / 96 / 128 GB` | **强制必开** | `SK=4` / `SK=2` | **`2K–4K`** / `8` | `5..7` | M5 Ultra 的精确半片（Dual-die Fusion）：`64GB` 跑 `27B+DFlash2 B=2@32K`；`128GB` 解锁 `27B+DFlash2 B=4@64K` 与 `35B MoE B=8@128K`（预计单流 `27B` 达 `~70–75 tok/s` DFlash2） |
| **M5 Ultra 丐版 (已实测)**<br>`apple-m5-ultra-64gpu` | **`64 / 64`**<br>(`0.80×`) | **`1.2 TB/s`**<br>(实测 `1095 GB/s`) | **`96 GB`**<br>(`WS=77.8 GiB`) | **强制推荐**<br>(MXFP4 `14.0G` / oQ4e `18.4G`) | **`SK=8` / `SK=2`**<br>(`SK>1` 钳 `RF=1`; `SK≤2` 取 `NSG=2`) | **`2048` / `12`** *(实测校准)* | **`7`** *(可选 `SpecSearch`)* | **实测主力甜点**：`27B+DFlash2 B=1@32K` **`161.2 tok/s`**、`B=4@32K` **`137.8–168.1 tok/s`**；开启 `FOSCAR` `Affine4Group128` 后 **`24.5 GB` 跑通 `B=8@64K` (`129.9 tok/s`)**、**`23.8 GB` 跑通 `B=4@128K` (`89.4 tok/s`)**；`35B MoE B=8@64K` **`463.5 tok/s`** |
| **M5 Ultra 满血 (Stage B)**<br>`apple-m5-ultra-80gpu` | **`80 / 80`**<br>(`1.00×`) | **`1.2 TB/s`**<br>(`1.00×`) | **`256 / 512 GB`**<br>(`WS≈192+ GiB`) | **推荐开启**<br>(零开销省内存) | **`SK=8` / `SK=2`**<br>(待 B-H1 复核) | **`2048` / `16`** *(待 B-H1 复核)* | **`7`** *(可选 `SpecSearch`)* | **全矩阵无死角**：在 64 核基础上获得 `+25%` Prefill/Verify/高并发 NAX 算力，无需压缩即可跑通原生 BF16 KV `27B+DFlash2 64K×B=8` 与 `128K×B=4/8`，并用 `Affine4` 扩展至 `B=16..32 @ 128K` |

---

### 4.2 各档硅片具体调优操作要诀（Per-SKU Playbook）

#### A. M5 Base（10 核）& M5 Pro（16–20 核）：内存寸土寸金与低核数防过切分
1. **必须离线预重排**：在 32–48 GB 机型上，未预重排的 `27 GiB` 占用会直接触发 macOS 内存压缩或拒绝准入；转换时务必加 `--nax-pretile` 将 `27B-MXFP4` 压在 `14.0 GiB`。
2. **收缩 Split-K 避免过度切分开销**：在 10–20 核 GPU 上，对于 `N=4096..17408`（已有 `256..1088` 个 `BN=16` 输出块），无需 `SK=8` 即可占满全部核心；可通过环境变量 `OMLX_M5_MXFP4_NAX_SK=2`（M5 Base）或 `4`（M5 Pro）减少 threadgroup 归约屏障。
3. **调低草稿深度 `K`**：在 `153–307 GB/s` 带宽下，7 步 BF16 草稿与 `W=8` 验证的绝对延迟放大了 `4×–8×`，将 `--draft-tokens` 调为 `3..4`（Base）或 `4..5`（Pro）可降低首 token 抖动与低接受率轮次的无效计算。

#### B. M5 Max（32–40 核，64–128 GB）：单人开发工作站黄金比例
1. **算力与带宽同比例缩放（`0.50×` 于 80 核 Ultra）**：M5 Max 40 核的算力/带宽比（算数平衡点 Ridge Point）与 M5 Ultra 80 核**几乎完全一致**（均为 `~15.4 GB/s` 每 GPU 核）！因此 80 核 Ultra 上验证最优的大多数算子交点（如 `W4A4` `T >= 32`、`moe_nax` `T >= 32`）可直接平移至 M5 Max。
2. **优先选 `35B-A3B MoE` 跑高并发长文本**：在 64 GB / 96 GB M5 Max 上，`Qwen3.6-35B-A3B-MXFP4` 每次仅激活 3B 参数且每槽 64K KV 仅 `1.94 GiB`（约为 Dense 27B DFlash2 `8.39 GiB` 的 **四分之一**），在 `614 GB/s` 带宽下即可跑出 `~65+ tok/s` 单流与 `~120+ tok/s` 多并发。

#### C. M5 Ultra 64 核 96GB（`apple-m5-ultra-64gpu`）：高带宽“大马拉小车”特化策略
1. **核心特征（全系最高单核带宽比）**：64 颗核心独享完整的 `1.2 TB/s` Quad-die UltraFusion 带宽（每核 **`18.75 GB/s`**，比 80 核的 `15.0 GB/s` 高出 `25%`）。因此：
   - **纯带宽型工作负载（`B=1` AR Decode、`lm_head`）几乎不输 80 核满血版**（实测 27B AR `48.92 tok/s`、35B AR `129.69 tok/s`）。
   - **必须激进开启横向并行切分**（`Split-KV = 2048`、骨干层 `SK = 8`），用足够的线程组数量把 64 核背后的四通道内存控制器队列塞满。
2. **96GB 容量红线守则**：开启预重排（省 `13 GiB`）后，`27B + DFlash2` 在 `32K` 可安全开到 `B=4`（实测 `124–143 tok/s`），在 `64K` 限 `B=2`，在 `128K` 限 `B=1..2`；切勿在 96GB 上强开 `27B+DFlash2 128K × B=4/8`。

#### D. M5 Ultra 80 核 256GB（`apple-m5-ultra-80gpu`）：Stage B 到货一键校准协议
80 核 256GB 到货后，直接复用 64 核已排雷的全部内核与离线预重排 `.omlxm5` 制品，按以下 4 步生成 `apple-m5-ultra-80gpu.json`：
1. 运行 `eval "$(./omlx/.venv/bin/python tools/m5_silicon_profile.py --shell-env)"` 锁定 `m5-ultra-80gpu-256g`。
2. 执行 `./tools/run_m5_ultra_capability.sh --iterations 50` 采集 80 核 H0 硬件计数器，验证 64→80 核在 `nax-q4-m8` / Prefill 上的真实加速比（理论预期 `1.20×–1.25×`）与 `recommended_working_set_bytes`（预期 `~192–207 GiB`）。
3. 执行 `./tools/run_m5_ultra_tuning.sh --iterations 30` 检查 `+25%` NAX 核心是否使 `Split-KV` 分片数从 `12` 升至 `16`、以及 `moe_nax_min_tokens` 是否从 `32` 下移至 `16`。
4. 解锁并实测 96GB 跳过的 **`128K × B=4/8`（最高 `143.6 GiB`）** 全并发矩阵与桌面端签名 DMG（`B-U5B`）。

---

## 5. 新硅片三步标准校准工作流（SOP）

未来接入任何新款 Apple Silicon（如 M5 Max 40 核、M5 Pro 20 核或后续世代）时，统一执行以下零侵入校准闭环：

```bash
# 1. 自动识别硬件 SKU 并导出隔离环境变量（未校准前自动加载比例合成先验 theoretical_prior）
eval "$(./omlx/.venv/bin/python tools/m5_silicon_profile.py --shell-env)"

# 2. 跑通 H0 探针 + 硬件计数器与 H1 分项调优曲线
./tools/run_m5_ultra_capability.sh --iterations 50
./tools/run_m5_ultra_tuning.sh --stages h1_l,h1_a,h1_g,h1_c,h1_m --iterations 30

# 3. 导出该 SKU 专属校准模板，填入实测最优覆盖项并打包进独立 Release
./omlx/.venv/bin/python tools/m5_silicon_profile.py \
  --emit-calibration-template "docs/backend/silicon-calibrations/${M5_SILICON_PROFILE_ID}.json"
./omlx/.venv/bin/python tools/build_m5_release.py --version v0.8.0-m5rt.X
```

---

## 6. 环境变量、CLI 标志与生产门控速查表（截至 `v0.8.0-m5rt.4`）

| 环境变量 / CLI 标志 | 默认值 | 作用与适用条件 |
|---|---|---|
| `OMLX_M5_T_DOMAIN_DISPATCH` | `0` (关) | 置 `1` 时将 `T=1` 纯 AR 路由至 `row_exact v3` SIMD QMV（27B-MXFP4 达 `48.8–48.9 tok/s`）；开启 DFlash2 时自动忽略以保障 `AR==DFlash2` 奇偶 |
| `OMLX_M5_ALLOW_W4A4_PREFILL` / `--flash-nax-prefill` | `0` / `off` | 对 `T >= 32` Dense MXFP4 Prefill 启用双侧 FP4 `BM=16` NAX，叠加强制对齐的 `FlashNaxV1` (`#4808`, `T >= 64`) 将 27B `1K/8K` TTFT 压至 `1471.8 ms / 12.02 s`（`1.45×–1.48×` 提速） |
| `--kv-dtype affine4_g128` | `bf16` | 启用 `FOSCAR` 零均值缩放偏置折叠 4-bit 分页 KV（压缩 `4.27×`，合同内 `100% AR==DFlash2`），在 96GB 上以 `23.8–24.5 GB` 跑通 `B=8@64K` 与 `B=4@128K` |
| `--spec-search on` | `off` (固定 `K=7`) | 启用 `AdaptiveSpecController`（`SpecSearch` + `DraftDepth` + MoE 唯一专家数 $U(M)$ 动态 `K∈[0..7]` 门控），防止高并发/极低接受率下的负投机收益 |
| `OMLX_M5_MOE_NAX_MIN_T` | **`8`** *(64c 校准)* | 触发 256 专家分桶零重排 `moe_nax`（含 `down_scatter_n16_sk2_nsg2`）的最小活跃 token 数；使 `B=8` 解码与 `T>=8` 验证直入 NAX（`35B B=8` 达 `463.46 tok/s`） |
| `OMLX_M5_MXFP4_NAX_SK` | 自动 (`8`/`2`) | 手动覆盖 `N < 65536` 骨干层的 MXFP4 NAX Split-K 路数（可选 `1, 2, 4, 8`，用于小核数 M5 Pro/Base 调优；`SK > 1` 时内核自动钳定 `RF = 1`） |
| `OMLX_M5_BF16_NAX_BN` / `_SK` | `16` / `4` | 手动覆盖 BF16 NAX 的 `BN`（`16, 32, 64`）与 `SK`（`1, 2, 4, 8`） |
| `OMLX_M5_DISABLE_{MXFP4,AFFINE,BF16,MOE}_NAX` | `0` (不禁用) | 故障隔离诊断开关：置 `1` 强制对应算子回退至可移植 SIMD 路径 |

---

## 7. 常见性能反模式排查清单（Troubleshooting Pitfalls）

1. **`phys_footprint` 接近模型体积的 2 倍（如 27B 占 27 GB）**：检查 `.omlxm5` 制品是否未经过 `m5_artifact.py pretile` / `convert --nax-pretile` 转换，触发了加载期在线 GPU 重排双份驻留。
2. **长上下文（`>= 4K`）下 GQA 模型解码 TPOT 随长度线性暴涨**：检查 `--attention-split-kv` 是否误设为 `off`，或未加载校准覆盖文件导致 `split_kv_min_context` 停留在过高阈值。
3. **高并发（`B=4/8`）吞吐不随槽位数扩展**：确认 `--batch-mode on` 已开启，使引擎走单命令缓冲 `replay_icb_batch` 而非串行单槽轮询。
4. **新增自定义大缓冲在 `execute_batch` 正常但 `ICB` 重放数值损坏**：立即核查缓冲内子对象 `offset` 是否可能达到或超过 `4 GiB`（`4,294,963,200 B`），按 `<= 2 GiB` 分段切分。

---

## 8. Wave 1 统计硬件计数器真机剖析与 Wave 3 `v0.8.0-m5rt.4` 极限挤压增补

### 8.1 M5 Ultra 64 核硬件计数器实测分解（Stage A Step 1 已完成，`gate.h0_complete == true`）

通过 `omlx_m5_hw_counter_probe`（融合 Metal 3 `MTLCounterSampleBuffer` 与 Metal 4 `MTL4CounterHeap` `24 MHz` = `41.67 ns/tick` 硬件时间戳，禁用时开销仅 `0.00042 ns/call` 零分配）及 `tools/m5_hw_trace_stats.py` 在 **M5 Ultra 64 核 96GB（macOS `27.2.1` Build `26B131`）** 上的实测采样（证据：[`h0/capability.json`](../../bench/results/m5-ultra-64gpu-96g/h0/capability.json)、[`h0/hw-counters-h0.json`](../../bench/results/m5-ultra-64gpu-96g/h0/hw-counters-h0.json)、[`h0/hw-trace-27b-mxfp4-summary.json`](../../bench/results/m5-ultra-64gpu-96g/h0/hw-trace-27b-mxfp4-summary.json)、[`h0/hw-trace-35b-a3b-mxfp4-summary.json`](../../bench/results/m5-ultra-64gpu-96g/h0/hw-trace-35b-a3b-mxfp4-summary.json)），得出以下关键硬件定律：

1. **物理统一内存峰值带宽实测达标称值的 `91.3%`（`1095.2 GB/s`）**：
   - `256 MiB` Blit DMA 硬件时间戳 `p50`：**Shared `1095.19 GB/s` / Private `1093.89 GB/s`**（`64 MiB` Private `1025.87 GB/s`），澄清了早期 `omlx_m5_probe`（`188.3 / 251.5 GB/s`）受限于标量 `u32` 计算核而非内存控制器的假象。
   - 硬件派发与屏障气泡极小（`N=750`）：同编码器内屏障间隙 **`p50 = 416.7 ns`**（`p90 = 666.7 ns, p99 = 708.3 ns`），跨编码器边界间隙 **`p50 = 416.0 ns`**（`p99 = 417.0 ns`）；独立 `5120×5120` NAX Q4 矩阵乘在 `M=256` 实测达 **`36.68 TFLOPS`**（`M=64` 为 `18.16 TFLOPS`）。

2. **实测子层 GPU 耗时与有效带宽分解表（揭示 `m5rt.4` 最优发力点）**：

| 模型与工作模式（64 核 96GB） | 总 GPU `p50` (壁钟 / 主机间隙) | 子层一：MLP / MoE | 子层二：GDN (48L / 30L) | 子层三：Full Attn (16L / 10L) | 子层四：`lm_head` (`BN=32, SK=2`) | 硬件计数器揭示的物理根因 |
|---|---|---|---|---|---|---|
| **27B-MXFP4 Pre-tiled** · `T=1` Decode | **`24.39 ms`**<br>(`24.68 ms` / `290 µs`) | **`62.4%`** (`15.02 ms`)<br>**`571.2 GB/s`** | **`23.7%`** (`5.71 ms`)<br>`488.2 GB/s` (`385` disp) | **`9.6%`** (`2.31 ms`)<br>`363.5 GB/s` (`144` disp) | **`3.0%`** (`0.73 ms`)<br>**`873.7 GB/s`** | `N<65K` 单 SG `BN=16, SK=8` + 每步 `240` 次细碎输入派发比 `lm_head` 少吃 `~300 GB/s` → 需 `2×BN16` 多 SG + `[q,k,v]`/`[qkv,z,a,b]` 拼接 |
| **27B-MXFP4 Pre-tiled** · `T=8` Verify | **`25.27 ms`**<br>(`316.6 tok/s` 验证率) | **`59.2%`** (`14.95 ms`)<br>`577.1 GB/s` (`17.4 TF`) | **`27.4%`** (`6.92 ms`)<br>`405.3 GB/s` (`12.2 TF`) | **`9.3%`** (`2.35 ms`)<br>`360.3 GB/s` (`10.9 TF`) | **`2.9%`** (`0.74 ms`)<br>**`866.1 GB/s`** (`26.1 TF`) | `BM=8` NAX 使 `T=8` MLP 耗时与 `T=1` 完全持平（`14.95 vs 15.02 ms`）；GDN 验证因死 `.scratch` 与未矢量化略增 `1.21 ms` |
| **27B-MXFP4 Pre-tiled** · `T=128` Prefill | **`266.23 ms`**<br>(`479.1 tok/s`) | **`63.8%`** (`169.7 ms`)<br>`25.77 TFLOPS` | **`29.3%`** (`77.9 ms`)<br>`18.24 TFLOPS` | **`6.5%`** (`17.3 ms`)<br>`24.74 TFLOPS` | **`0.3%`** (`0.79 ms`)<br>`857.8 GB/s` | `BM=8` 重读权重 `16×` + GDN 串行递推占近三成 → 需 `RF=8` Wide NAX + GDN `TB=8` & 2D 卷积 + NAX FlashAttention |
| **35B-A3B-MXFP4 MoE** · `T=1` Decode | **`7.58 ms`**<br>(`7.79 ms` / `206 µs`) | **`58.2%`** (`4.31 ms`)<br>**仅 `136.1 GB/s`!** (`520` disp) | **`21.9%`** (`1.62 ms`)<br>`310.7 GB/s` (`241` disp) | **`11.5%`** (`0.85 ms`)<br>`160.3 GB/s` (`90` disp) | **`4.1%`** (`0.31 ms`)<br>**`831.4 GB/s`** | **核心发现**：`T=1` 因 `<32` 阈值回退非 NAX SIMD MoE，带宽仅 `136.1 GB/s`，独占近六成耗时！ |
| **35B-A3B-MXFP4 MoE** · `T=8` Verify/Batch | **`13.34 ms`** | **`72.2%`** (`9.64 ms`)<br>`392.1 GB/s` (`1.84 TF`) | **`16.5%`** (`2.20 ms`)<br>`234.5 GB/s` | **`6.5%`** (`0.86 ms`)<br>`161.5 GB/s` | **`2.4%`** (`0.32 ms`)<br>**`816.5 GB/s`** (`24.6 TF`) | `T=8` 仍卡在非 NAX MoE，占比飙升至 `72.2%`！下探 `moe_nax` 至 `T=1..31` 是头号开关 |
| **35B-A3B-MXFP4 MoE** · `T=128` Prefill | **`58.30 ms`**<br>(`2163.6 tok/s`) | **`55.2%`** (`32.07 ms`)<br>**跃升至 `522.4 GB/s`!** | **`37.9%`** (`22.01 ms`)<br>`11.64 TFLOPS` | **`5.8%`** (`3.39 ms`)<br>`20.40 TFLOPS` | **`0.6%`** (`0.33 ms`)<br>`805.1 GB/s` | 一旦激活 `moe_nax`，MoE 带宽暴涨 `3.84×`；剩余 `37.9%` 瓶颈转移至 GDN 递推（由 `TB=8` + 2D 卷积攻克） |

### 8.2 `v0.8.0-m5rt.4` 实机提炼的四条新 Apple10 工程定律与极限挤压实绩

在 `v0.8.0-m5rt.4` 真机收官压测中（架构见 [`ninfer-m5-porting-architecture-v2.md`](../backend/ninfer-m5-porting-architecture-v2.md)，完整榜单见 [`m5rt.4 WALKTHROUGH.md`](../versions/v0.8.0-m5rt.4-full-ninfer-squeeze-64gpu/WALKTHROUGH.md)），进一步验证并确立了以下四条 Apple10 核心准则与生产实绩：

1. **定律 7 — 多子图融合 ICB 的 `Encoder::dynamic_parameters` 边界清空铁律**：
   - 将多个具有独立动态 token 宽度的逻辑子图（如 `DFlashForward` 中 `ctx_capacity=8` 的增量上下文更新与 `capacity=8` 的 5 层草稿骨干）融合进单条 Compute ICB 时，**必须在子图切换处清空 `Encoder::dynamic_parameters` 常量去重表**，否则短增量步（`rem=1`）补丁会静默覆写后续子图的静态 `width=8` 参数。隔离后草稿单轮压至 **`7.42 ms`**，27B-MXFP4 Coding 飙升至 **`161.23 tok/s`**（27B-oQ4e 达 **`131.98 tok/s`**）。
2. **定律 8 — `SK > 1` 下强制钳定 `RF = 1` 的寄存器与共享内存保驻留定律**：
   - 在 Apple10 `matmul2d` 中，当 `SK > 1` 时若叠加 `RF > 1` 多行循环，寄存器累加器与 `parts[]` 线程组归约区同步膨胀 `RF` 倍会导致驻留崩盘（`T=1024` 曾倒退至 `2991.8 ms`）。强制 **`SK > 1 → RF = 1`**（仅在 `SK == 1` 时启用 `RF > 1`）后，配合 `run_packed_batch` 跨槽单次读权重，使 27B `B=4` AR 跃升 **`2.45×–3.81×`**（MXFP4 `135.49 tok/s` / oQ4e `132.36 tok/s`，oQ4e DFlash2 达 **`168.08 tok/s`**）。
3. **定律 9 — 多 SG 列分块分界（`SK <= 2` 选 `NSG = 2` vs `SK >= 4` 选 `NSG = 1`）**：
   - 在 `SK <= 2`（`64–128` 线程/TG）场景下启用 `NSG = 2`（`2 × BN_stored=16` 零重排双列块）并将 `moe_nax` 阈值下探至 `T >= 8`，使 35B-A3B MoE `T=1024` Prefill 从 `381.7 ms` 降至 **`282.27 ms`**（`T=128` `moe_sublayer` 有效带宽达 **`633.7 GB/s`**），`B=8 @ 64K` 并发总吞吐从 `247.71` 暴涨 **`+87.1%` 至 `463.46 tok/s`**；而在 `SK >= 4` 时保持 `NSG = 1` 守住 `256` 线程高驻留甜点。
4. **定律 10 — 占用率门控 GQA 共享（`total_tgs >= 64`）与 96GB 双刃剑容量突破**：
   - **GQA 门控**：在 64 核 M5 Ultra 上，`attention_exact` 的同组 Query Head 共享 KV 页加载必须门控在 **`total_tgs >= 64`** 才激活，防止 `B=1, T=1` 短上下文（`kv_heads=4` 仅 4 个 TG）饿死 60 颗 GPU 核。
   - **长上下文双刃剑 + 并发服务治理**：`#4808` `FlashNaxV1` (`BQ=64, BK=32, WN=2`) 将 27B `8K` Prefill 砍掉 **`5.78 s`**（`17.80 → 12.02 s`）；`g128` 离线预重排将 27B-oQ4e 内存从 `32.8 GB` 砍至 **`18.4 GB`**，10P 核并行校验将冷加载从 `11.84 s` 砍至 **`1.84 s` (`6.43×`)**；`FOSCAR` `Affine4Group128` KV + 循环 `feature_history` 环 + `SharedPrimary` 在 96GB 真机上仅用 **`24.5 GB` 跑通 `B=8 @ 64K` (`129.86 tok/s`)**、**`23.8 GB` 跑通 `B=4 @ 128K` (`89.40 tok/s`)**；全局 COW `SharedPrefixCache` 实现 `B=4` **`100%` (`4/4`) 命中（热启动 TTFT `633.89 ms`）**，`#5164` 解码保护在 `2048`-token 预填竞争下稳守 **`24.53 ms` p50 ITL**。
