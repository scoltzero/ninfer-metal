---
name: qwen-mxfp4-oq4e-silicon-squeeze
description: >-
  面向 Apple M5 系列及后续搭载硬件 NAX 矩阵单元的 Apple Silicon（M5 / M5 Pro / M5 Max / M5 Ultra
  及后续世代，MTLGPUFamilyApple10+，10–80+ GPU 核）的无损硅片级极限推理压榨方法论。当需要在 Metal 4
  与硬件 NAX 上将任意规模 Qwen 混合架构（3:1 Gated DeltaNet + GQA + Dense SwiGLU / 稀疏 MoE）的
  MXFP4（E2M1 × UE8M0）与 oQ4e（Affine Q4/Q5/Q6/Q8，group_size=64）单/多流预填、纯自回归解码、
  投机解码、长上下文分组仿射 KV 量化压缩及 UMA 零搬移常驻与多轮滚动前缀复用从通用框架基线推至本机实测
  FP16/BF16 NAX 算力峰值与持续 DRAM 物理带宽的 95%–100%，同时严守 IEEE-754 FP32 累加无损精度与投机解码
  0-ULP 贪心奇偶契约时启用本技能。
---

# Qwen 混合架构 `MXFP4` & `oQ4e` Apple M5+（`MTLGPUFamilyApple10+`）硅片极限压榨方法论

本技能严格面向搭载**硬件 NAX 矩阵张量单元**的 **Apple M5 系列及后续世代 Apple Silicon（`M5 / M5 Pro / M5 Max / M5 Ultra` 及后续 NAX 架构，`MTLGPUFamilyApple10+`）**。严禁照搬其他芯片型号或特定参数量模型的绝对每秒令牌数与毫秒延迟数字作为优化目标；一切分块几何、并行切分与融合决策必须由**本机实测硅片物理常数**（GPU/NAX 核数 `gpu_cores`、纯 `FP16/BF16 -> FP32` 矩阵峰值 $P_{\text{NAX}}$、持续内存带宽 $B_{\text{DRAM}}$、算术平衡点 $I_{\text{ridge}}$、片上 SLC 容量 $C_{\text{SLC}}$、统一内存容量 $R$）与**模型拓扑参数**（隐藏维 `H`、中间维 `F`、线性注意力头 `Hv`、KV 头 `kv_heads`、激活专家数 `top_k`、词表维 `V`）共同驱动。

全流程遵循统一工程诊断闭环：**查什么硬件指标 -> 为什么卡住 -> 怎么改算子与调度 -> 用什么物理比例（Roofline 百分比）验收**，在 **100% 无损反量化 + FP32 累加 + 投机解码 `0-ULP` 贪心一致** 且 **常驻内存不高于单份离线预重排体积** 的硬约束下，将计算受限段推至 `92%–98%` $P_{\text{NAX}}$、将带宽受限段推至 `99%–100%` $B_{\text{DRAM}}$ 物理线速地板。

---

## 1. Phase 0（必选前置步）：Stage B（`H0` + `H1`）本机硅片自标定闭环

在修改任何模型算子或运行端到端评测前，Agent **必须先在当前目标机上执行 Stage B（`H0 -> H1`）硅片自标定**，用实测硬件常数替换静态理论先验（`theoretical_prior`），杜绝跨芯片盲猜参数。

### 1.1 `H0` 裸硅片物理能力与硬件不变量探针（Bare-Silicon Roofline Probe）

- **查什么**：通过 `m5rt/src/capability.cpp`（`select_metal_capability_profile`）、`tools/m5_silicon_profile.py`、`tools/run_m5_ultra_capability.sh`（驱动 `omlx_m5_hw_counter_probe --mode h0`、`omlx_m5_probe`、`omlx_m5_nax_qmm_probe`、`omlx_m5_inspect_icb_canary`）采集本机八项底层不变量并落盘 `h0/capability.json` 与 `h0/hw-counters-h0.json`。
- **为什么卡住（未标定常见误判）**：
  1. 用普通标量/向量 Compute 拷贝核测带宽，会受限于计算核指令发射率而仅测出 `~20%–25%` 真实内存带宽，导致严重低估 $B_{\text{DRAM}}$；
  2. 测试缓冲落入片上 SLC 容量内会把 SLC 带宽误认为 DRAM 带宽；
  3. 盲信 `MTLDevice.maxBufferLength`（几十 GiB）分配单块连续大权重池，在 Compute ICB 重放时触发静默数据损坏；
  4. 忽略 Binned 与 Full-Core 芯片（如共享同一多裸片内存控制器但核心数不同的 SKU）之间单核带宽比 $B_{\text{DRAM}} / \text{gpu\_cores}$ 的差异，套用错误的 Split-K / Split-KV 门限。
- **怎么探针标定（标准执行序列）**：
  1. **核验 `MTLGPUFamilyApple10+` 与 Metal 4 / NAX 资格门控**：确认 `apple_gpu_family >= 10`（每颗 GPU 核心配备 `1` 个硬件 NAX 单元，`nax_units == gpu_cores`）、`native_tensorops_eligible == true`（`metal4_command_queue`、`metal4_command_allocator`、`tensor_api`、`native_tensorops_runtime`、`shared_event` 全绿）、`metal4_icb_eligible == true`（含 `compute_icb_barrier`、`residency_sets`、`metal4_argument_table`）、每线程组共享内存硬上限 `max_threadgroup_memory_length == 32,768 B`（`32 KiB`）以及系统推荐工作集 $W_{\text{ws}} \approx 0.81 R$。
  2. **Compute ICB `< 2 GiB` 分段边界与万步重放金丝雀（`omlx_m5_inspect_icb_canary`）**：运行 `>= 10,000` 步动态参数补丁重放校验逐位一致性，并锁定硬件段上限 `icb_safe_weight_segment_bytes = 2 GiB`（`2,147,483,648 B`），严防 `offset >= 4 GiB` 时的 32-bit 偏移回绕。
  3. **标定 `24 MHz` 硬件时间戳与编码器屏障气泡（`profile_dispatch_gap_counters`）**：校核 `MTL4CounterHeap` / `MTLCounterSampleBuffer` 硬件计数器频率（`24 MHz` = `41.67 ns/tick`；关闭态 `HwCounterCollector` 内联早退开销仅 `~0.0004 ns/call`，零堆分配），实测同编码器连续派发屏障间隙 `intra_encoder_dispatch_gap_ns`（典型值 `~416 ns`）与跨编码器间隙，据此量化每次多余 `setBarrier` 的真实时间成本。
  4. **标定持续物理带宽 $B_{\text{DRAM}}$ 与片上 SLC 容量折点 $C_{\text{SLC}}$（`profile_blit_copy_counters` vs `profile_compute_copy_counters`）**：
     - 必须使用硬件 DMA 引擎（`MTLBlitPassDescriptor` 挂载时间戳采样）配合多 SG 合并宽向量读取测量真实内存控制器吞吐，严禁用单核受限的标量拷贝替代；
     - 将拷贝工作集从 `4 MiB` 按 2 的幂次递增扫描至 `512+ MiB`（`4, 8, 16, 32, 64, 128, 256, 512 MiB`）：带宽曲线发生断崖下跌的拐点即揭示本机有效 **SLC 容量 $C_{\text{SLC}}$**；小于 $C_{\text{SLC}}$ 时测得片上缓存带宽，大于 $4 \times C_{\text{SLC}}$（`256–512+ MiB`）后的稳态中位数 `p50` 即为真实持续 DRAM 带宽 $B_{\text{DRAM}}$。
  5. **标定各数据类型纯 NAX 算力峰值 $P_{\text{NAX}}$ 与芯片算术平衡点 $I_{\text{ridge}}$**：运行寄存器/片上常驻的 `mpp::tensor_ops::matmul2d` 探针，实测 `FP16/BF16 -> FP32` 满血峰值 $P_{\text{NAX}}$ 及 `INT8`、`FP8`、`MXFP4×MXFP4` 相对倍率，计算本机精确 **Roofline 算术平衡点（Ridge Point）**：
     $$I_{\text{ridge}} = \frac{P_{\text{NAX}}}{B_{\text{DRAM}}} \quad (\text{FLOP/Byte})$$
     凡算术强度 $I < I_{\text{ridge}}$ 的算子（如 `T=1` 解码、窄批验证）以 `% of B_DRAM` 验收；凡 $I > I_{\text{ridge}}$ 的算子（如大块 Prefill）以 `% of P_NAX` 验收。

### 1.2 `H1` 算子族微基准扫描与本机静态派发表生成（Operator-Class Calibration Sweep）

- **查什么**：在 `H0` 通过后，运行 `tools/run_m5_ultra_tuning.sh` / `tools/m5_ultra_tuning.py --stages h1_l,h1_a,h1_g,h1_c,h1_m` 生成本机实测校准覆盖文件 `docs/backend/silicon-calibrations/${M5_SILICON_PROFILE_ID}.json`（自动记录 `prior_vs_measured_diff`）。
- **为什么卡住**：算子分块交点（如 `T=1` QMV vs `BM=8` NAX、`SK` 切分路数、`Split-KV` 触发长度、稀疏专家分桶阈值）高度依赖本机的 `gpu_cores`、$I_{\text{ridge}}$ 与模型的 `kv_heads`/`Hv` 几何；未经 `H1` 扫描直接沿用未校准先验会导致核心闲置或过度切分归约反噬。
- **怎么扫描定参（五大必跑子包）**：
  1. **`h1_l`（线性投影 / SwiGLU / `lm_head` 分块与 `SK` 交点，`omlx_m5_benchmark_projection_candidates`）**：在相同 `precision_contract` 与 `row_exact_group` 约束下，分别扫描 `T=1`（`KL32`/`KL16` QMV vs 窄块 NAX）、`T=2..16`（窄块 `BM=8, BN ∈ {16, 32}, BK=128, SK ∈ {1, 2, 4, 8}`, `NSG ∈ {1, 2}`）与 `M >= 32`（宽块 `64×64 SK=1` vs `32×32 SK=1`）的 `gpu_median_ms` 交点。
  2. **`h1_a`（窄 `kv_heads` GQA `Split-KV` 拐点，`omlx_m5_benchmark_long_kv_attention`）**：针对目标模型的真实 `(Q_heads, kv_heads, head_dim)`，在 `L ∈ [512 .. 65536]` 上对比未分片因果核与在线 Softmax `Split-KV`，锁定正收益起点 `attention_split_kv_min_context` 与最优分片数 `attention_split_kv_partitions`（`gpu_cores / kv_heads` 越大，拐点越靠前）。
  3. **`h1_g`（GDN 线性注意力分块与 `kDvSlice` 波次对齐，`omlx_m5_benchmark_gdn_prefill`）**：扫描串行递推切换到分块矩阵扫描的最小序列长度，以及 `value_dim = 128` 的切片粒度 `kDvSlice ∈ {16, 32}`，确保线程组总数 `Hv × (128 / kDvSlice)` 对齐 `gpu_cores` 的整数波次且单线程状态寄存器零溢出。
  4. **`h1_m`（稀疏 MoE 真实路由导出与分桶阈值，`omlx_m5_export_p12_routes`）**：导出真实提示词下的 `top_k` 专家激活分布（该分布仅取决于模型权重，跨同架构芯片可复用），标定 GPU 端流式分桶 NAX 超越逐专家 QMV 的精确活跃 token 阈值 `moe_nax_min_tokens`。
  5. **`h1_c`（Compute ICB 整图重放与批合并提交开销，`omlx_m5_benchmark_p10_batch`）**：量化单命令缓冲 `replay_icb_batch` 相对逐槽提交的壁钟收益及双分配器乒乓流水（`ar_pipe_[2]`）的主机气泡隐藏率。
- **标准校准命令链**：
  ```bash
  # 1. 识别当前 M5+ 硅片 SKU 并加载初始先验环境
  eval "$(./omlx/.venv/bin/python tools/m5_silicon_profile.py --shell-env)"
  # 2. 执行 H0 裸硅片 Roofline + 24 MHz 硬件计数器标定
  ./tools/run_m5_ultra_capability.sh --iterations 50
  # 3. 执行 H1 五大算子族微基准扫描并生成本机校准配置
  ./tools/run_m5_ultra_tuning.sh --stages h1_l,h1_a,h1_g,h1_c,h1_m --iterations 30
  ./omlx/.venv/bin/python tools/m5_silicon_profile.py \
    --emit-calibration-template "docs/backend/silicon-calibrations/${M5_SILICON_PROFILE_ID}.json"
  ```

### 1.3 `50` 行冷 DRAM 微探针先行、合并前废代码清零与 `< 10 s` 子层计数器闭环

1. **铁律一：`50` 行真机冷 DRAM 单算子微探针先行 + 合并前清零废代码（Cold-DRAM Micro-Probe First & Dead-Code Purge）**：
   - **查什么**：在铺开全位宽/多形态算子模板族（如 `Q3/Q4/Q5/Q6/Q8 × linear/concat/SwiGLU/norm` 或多行标量特化族）之前，先用独立单算子探针实测候选核相对硬件固定功能（Fixed-Function）基线的**真实冷 DRAM 有效带宽比（`% of B_DRAM`）**、数值逐位残差（`max_abs_diff == 0.0`），并用 `metal-objdump -d` 检查是否存在栈帧分配（`alloca`）或 VGPR 越界。
   - **为什么卡住**：若不经冷 DRAM 单算子验证就直接批量铺写数百个模板 PSO：（a）小工作集微测会全部落入片上 $C_{\text{SLC}}$，掩盖手动解包 ALU 指令与屏障在真实 DRAM 延迟下的反噬（例如 4-bit 手动 `Ws` 解包在冷 DRAM 下反而比零 ALU 固定功能张量核慢 `12%–16%`）；（b）未过门限的实验 PSO 与废弃环境变量分支一旦残留，会导致 `.metallib` 编译膨胀数千行并污染运行时派发树。
   - **怎么执行与清零**：
     1. 编写独立 **`~50–100` 行单算子 Metal 微探针**，在容量**严格大于片上 SLC（`>= 3–4 × C_SLC`，如 `> 2.5–3.6 GiB`）**的 `MTLResourceStorageModePrivate` 权重池上轮转派发（预热后取 `>= 30` 次串行屏障 `GPUEndTime - GPUStartTime` 中位数），按无填充真实信息位宽字节数折算冷 DRAM 有效 `GB/s / B_DRAM`。
     2. **严格准入与合并前清零门禁**：**仅针对超越硬件基线显著高于测量噪声（有效带宽或算力净增益 `>= +10%–12%`）的特定位宽与形状**推进模板化合入；凡未达门限的实验 PSO、被证伪的多行模板族及死派发分支，**必须在合入 `main` 前立即物理删除清零**，严禁携带死代码合入主干。
2. **铁律二：`< 10 s` 整网子层 `24 MHz` 硬件计数器迭代闭环（Fast Sublayer Loop）**：
   单算子微探针准入后，严禁每改一行 Shader 就盲跑漫长的端到端全量套件：必须先运行 `omlx_m5_hw_counter_probe --mode model`（`< 10 s` 完成），读取 `24 MHz` 时间戳分解出的 `dense_mlp / moe`、`gdn`、`full_attn`、`lm_head`、`sampling` 各子层的精确 GPU 耗时占比、派发数、屏障数、`mean_effective_tflops / P_NAX` 与 `mean_effective_gbs / B_DRAM`；确认子层物理比例正向逼近 Roofline 且无寄存器溢出退化后，再运行净室端到端回归核验吞吐、`phys_footprint` 与 `0-ULP` 奇偶。

---

## 2. 脱离通用框架桎梏：硬件内存包络、拓扑特化与 `0-ULP` 无损契约

### 2.1 通用动态图框架的六大结构性停顿与 Qwen 混合拓扑映射

- **查什么**：审计现有推理图中的类型转换核数量、动态分支谓词、`SK > 1` 临时工作区字节数、单线程组（`1-TG`）派发占比、进程常驻物理内存与权重文件体积比 $M_{\text{resident}} / M_{\text{artifact}}$，以及主机端命令提交间隙。
- **为什么卡住**：通用逐算子框架（如 `MLX` / `oMLX`）在 `MTLGPUFamilyApple10+` 高带宽密度与 `32 KiB` 共享内存约束下存在六大固有停顿：
  1. **冗余激活转换遍（`x_f16` / `xsum` Passes）**：投影前独立发射 `bf16 -> f16`（如 `ensure_f16_cast` / `bf16_to_f16_rowscale`）或激活分组求和（`activation_group_sum`），凭空增加数百次内核启动与全局内存往返。
  2. **运行期位宽与边界分支**：单核动态分支兼容多种 `bits`、`group_size` 与非对齐行列谓词，阻断 `#pragma clang loop unroll(full)` 完全展开，诱发寄存器数组溢出至栈内存（stack spill）。
  3. **`SK > 1` 拆分工作区与重复激活读取**：在大 `M` 预填或输出块已充足的矩阵上盲目沿 `K` 维拆分，导致激活重读 `SK` 次，并消耗 $\text{TG\_Bytes}_{\text{linear}} = (SK - 1) \times BN \times 64\text{ B}$（融合 SwiGLU 翻倍至 $(SK - 1) \times BN \times 128\text{ B}$）共享内存压低驻留度。
  4. **单核 `1-TG` 串行孤岛与全芯片 `setBarrier` 排空**：独立发射 `1-TG` 的 `RMSNorm`、`add_rmsnorm`、`MoE Router`、`Top-K Softmax` 或大词表 `Argmax`，令其余 `gpu_cores - 1` 颗核心空转，且前后各插入一道 `MTLBarrierScopeBuffers` 全芯片屏障打断大网格尾波重叠。
  5. **行主序与在线重排双份驻留**：磁盘存行主序而加载期在 GPU 上生成 `Private` NAX Tiled 副本，导致进程同时常驻 `Shared` 原始池与 `Private` 重排池，内存膨胀至 $M_{\text{untiled}} \approx 1.92 \times M_{\text{artifact}}$。
  6. **动态图主机调度空泡**：缺少静态 Compute ICB 整图录制与双 `MTL4CommandAllocator` 乒乓流水（`ar_pipe_[2]`），步间 CPU 编码间隙暴露在关键路径上。
- **怎么按 Qwen 混合拓扑特化**：一切网格（Grid）、分块几何（Tile）与融合边界直接参数化锚定 Qwen 混合架构的拓扑结构：

| 拓扑子系统 | 通用拓扑参数表示 | 结构性瓶颈特征 | 硅片压榨特化策略 |
|---|---|---|---|
| **主干层间配比** | 总层数 $L_{\text{layers}}$，**`3:1` 混合**（`75%` GDN 线性注意力 + `25%` GQA 全注意力） | 四分之三层含跨步状态递推，逐 token 串行扫描会彻底饿死 NAX 单元 | Prefill 采用 `Hv × (128 / kDvSlice)` 满波 `C=64` 块矩阵 NAX 扫描 + 原位 `norm_gate`；`T=1` 采用 `concat4` + 单核 `conv+recurrent+norm_gate` |
| **GQA 全注意力** | 隐藏维 `H`，查询头 `Q_heads`，极窄 **`kv_heads`（典型 `2..8`）**，头维 `D`（典型 `128..256`） | 未切分 KV 时并行度受限于窄 `kv_heads`，远不足以填满高核数芯片的内存队列 | `L_kv <= 512` 走单遍无切分因果核（省 merge）；中长上下文按 `gpu_cores / kv_heads` 前移在线 Softmax `Split-KV` 门限，`>= 16K` 启用 `512`-tok 超块 |
| **GDN 线性注意力** | 键头 `Hk`，值头 **`Hv`**，`kd = vd = 128`，因果卷积核 `conv = 4`，FP32 状态 `[Hv, 128, 128]` | 状态体积达 $64\text{ KiB} \times H_v$，四路异构输入投影 `[qkv, z, a, b]` 单独发射屏障密集 | `T=1` 将四路异构位宽/行数投影合并为单次 `concat4` 发射，并令 `gdn_commit_history` 与 `gdn_out` 在同一无屏障波次并发重叠 |
| **FFN（Dense vs MoE）** | Dense `SwiGLU(H, F)`；稀疏 MoE：`E` 专家选 `top_k` + `shared_expert` + `shared_gate`，中间维 `F_moe` | Dense 双投影重复读激活与写中间态；MoE CPU 同步分桶或逐专家串行派发开销巨大 | Dense 始终双指针融合 `gate+up+SiLU`；MoE Prefill 走零同步 GPU 流式分桶 + `compact32` 4-SG 切分，`T=1` 压至极致 **3-dispatch** |
| **超大词表输出头** | `lm_head`：`[V, H]`（如 `V = 248,320`） | 输出维极大（$V \ge 65536$），且单核扫描词表采样耗时达数百微秒 | $N \ge 65536$ 离线固定采用 `BN_stored = 32`；`T=1` 融合 `final_norm` 前奏直读预重排块 + 多 TG 两阶段并行 `Argmax` |

### 2.2 Compute ICB `< 2 GiB` 分段铁律、子图参数隔离与单份 `.miv` 零拷贝复用

- **查什么**：核查所有绑定进 `MTLIndirectComputeCommand` 的 `MTLBuffer` 段内最大字节偏移、多子图合并录制时的常量去重表状态，以及磁盘模型制品的物理块布局。
- **为什么卡住**：
  1. **32-bit ICB 偏移静默回绕陷阱**：Apple Silicon 硬件在通过 `MTLIndirectComputeCommand` 编码缓冲绑定地址时，**凡段内字节偏移 `offset >= 4 GiB`（`4,294,963,200 B`）均会发生 32-bit 无符号整数静默截断回绕**，且 Metal API Validation 与 Shader Validation 均不报错，直接导致后半段网络读错权重或 KV 页。
  2. **多子图动态常量串扰**：单条 Compute ICB 串联不同 token 宽度的逻辑子图（如增量上下文更新与多步草稿骨干）时，若复用同一常量去重缓存，前序子图的短步动态补丁会静默覆写后续子图的宽度参数。
- **怎么改运行时与离线制品**：
  1. **硬性分段上限**：不可变模型权重池单段容量钳定为 **`2 GiB`（`2,147,483,648 B`）**；加载期生成的辅助紧凑段（`Q5High1Cache`、`sb_tiled`）钳定为 **`1 GiB`**；多槽并发 KV 页池采用跨槽共享 `SharedKvArenaPool`，确保任意绑定视图段内偏移严格 `< 2 GiB`。
  2. **子图边界清空常量缓存与双槽乒乓流水**：在子图切换处强制调用 `Encoder::dynamic_parameters.clear()` 隔离动态参数槽；异步解码管线维护双槽 `ar_pipe_[2]`（各自持有独立 `MTL4CommandAllocator` 与预录 ICB），步间无锁交替提交以彻底隐藏主机编码延迟。
  3. **单份 `.miv` 离线预重排跨三态零拷贝复用**：转换期通过 `tools/m5_artifact.py convert --nax-pretile` 生成唯一不可变 `mx_nax_tiled_bk128_v1` / `oq_nax_tiled_g64_v1` 布局（`N < 65536` 取 `BN_stored = 16`，`N >= 65536` 取 `BN_stored = 32`）。由于内存中连续 `4` 个 `BN_stored = 16` 物理块天然构成连续的 `BN_logical = 64` 宽块，同一份 `mmap`/上传权重可在 **`T=1` 纯 AR 直读 QMV**、**`BM=8` 耦合验证 NAX** 与 **`BM=64, BN=64, SK=1` 宽块预填 NAX** 之间实现 **0 字节重排拷贝切换**。
- **用什么物理比例验收**：进程权重常驻内存从双份驻留的 $1.92 \times M_{\text{artifact}}$ 压降至 **$1.00 \times M_{\text{artifact}}$（物理内存占用立减 `~48%`）**，且严格满足统一内存安全准入不等式（$W_{\text{ws}} \approx 0.81 R$，$H_{\text{os}}$ 为系统与 ICB 预留余量）：
  $$W_{\text{shared}}(\text{pretiled}) + B \times S_{\text{slot}}(L) + P_{\text{cache}} \le W_{\text{ws}} - H_{\text{os}}$$

### 2.3 无损反量化证明、NAX 硬件指令速率比、Dual-ANE 交点方程与双轨 `0-ULP` 奇偶锁

1. **`MXFP4` 与 `oQ4e` 片上反量化至 `FP16`/`BF16` 的 100% 数学无损证明**：
   - **`MXFP4`（`E2M1 × UE8M0`，`group_size = 32`）**：4-bit `E2M1`（1 位符号、2 位指数、1 位尾数）的全部 16 个编码精确落在有理数集 $\{0, \pm 0.5, \pm 1.0, \pm 1.5, \pm 2.0, \pm 3.0, \pm 4.0, \pm 6.0\}$（至多 2 位有效尾数），8-bit `UE8M0` 为纯 2 的整数幂缩放因子 $2^{e - 127}$。相乘仅平移二进制指数而不增加尾数位宽，在 `bfloat`（8 位指数、7 位尾数）全动态范围内 **100% 零舍入无损表示**；在 `half`（5 位指数、10 位尾数）路径下配合行级 2 的幂次预缩放 `row_scale` 同样 **100% 精确无损**。
   - **`oQ4e`（`Affine Q4/Q5/Q6/Q8`，`group_size = 64`）**：量化整数码 $q \in [0, 2^b - 1] \subseteq [0, 255]$（至多 8 位整数），每 64 元素组的 `scale` 与 `bias` 原生存储为 16-bit 浮点。无论在片上还原 $w = \text{scale} \times q + \text{bias}$ 还是按分配律展开 $\text{scale} \sum_i x_i q_i + \text{bias} \sum_i x_i$，整数码 $q$ 在 IEEE-754 `half`（11 位精确整数精度，精确覆盖 $[0, 2048]$）中均 **100% 零精度损失**。
2. **`MTLGPUFamilyApple10+` 硬件 NAX 矩阵管线相对吞吐层级**：

| 硬件矩阵指令路径 (`mpp::tensor_ops::matmul2d`) | 累加器精度 | 相对 $P_{\text{NAX}}$ 吞吐倍率 | 数值契约与架构选型裁定 |
|---|---|---:|---|
| **`FP16 × FP16 -> FP32` / `BF16 × BF16 -> FP32`** (`relaxed_precision=true`) | **FP32** | **`1.00×`** | **唯一满速无损主力路径**：硬件满速率发射，权重与激活 100% 无损 |
| `INT8 × INT8 -> INT32` | INT32 | `~0.83×` | 算力折损 `~17%`，且需插入有损动态激活量化 |
| `FP8 × FP8 -> FP32` | FP32 | `~0.82×` | 算力折损 `~18%`，截断 BF16/FP16 激活尾数 |
| **`MXFP4 × MXFP4 -> FP32`** (`W4A4` `E2M1 × E2M1`) | FP32 | **`~0.48×`** | **硬件吞吐减半**，且在线 `A4` 激活量化破坏无损契约并打断上下游算子融合 |

   > **核心推论**：在宽块 `BM=64, BN=64, SK=1` 架构下，将 `MXFP4` / `oQ4e` 权重在片上无损解包入 `9,216 B` `Ws` 共享内存并喂入 `1.00×` 速率的 `FP16/BF16 -> FP32` NAX 管线，单算子有效算力可达 **`~97%–98%` $P_{\text{NAX}}$**——比硬件原生双侧 `MXFP4 × MXFP4`（`~0.48×` $P_{\text{NAX}}$）理论上限还快 **`2.0×` 以上**，且零激活量化误差。

3. **Dual-ANE `INT8` 异构分流交点公式与严禁事项**：
   - 设芯片 Neural Engine 持续 INT8 算力为 $P_{\text{ANE}}$。将单层线性投影按输出维 $N$ 切分给 ANE 与 GPU NAX 并行执行时，不计同步开销的理论最优切分比例为：
     $$\alpha_{\text{opt}} = \frac{P_{\text{ANE}}}{P_{\text{NAX}} + P_{\text{ANE}}}$$
   - **小于 `10%` 净收益跳过准则**：当纯 GPU NAX 已压榨至 `>95%` $P_{\text{NAX}}$ 时（典型高核 SKU 上 $P_{\text{ANE}} \approx 0.30 P_{\text{NAX}} \implies \alpha_{\text{opt}} \approx 22\%$），计入输入激活打包、跨引擎共享事件同步 fence、输出拼接及片上激活工作集折点后，MLP 异构混合的整网端到端净提速在同步模式下仅 `~4%–5%`、理想异步流水上限亦 `< 9%`（不足 `10%`），却需额外常驻数 GiB 的 INT8 权重副本并牺牲无损契约。**凡评估整网净收益 `< 10%`、受限于统一内存容量或要求 `0-ULP` 无损确定性的场景，一律跳过 ANE，坚持 100% 纯 GPU `W4A16` NAX**。
   - **GDN 递推层严禁启用 INT8 ANE**：GDN 层的 `in_proj_qkv` 直接喂入因果卷积、块内 Neumann/WY 三角矩阵求逆与跨块指数状态递推 $S_{c+1} = \gamma S_c + \Delta^\top K$；将该投影降为 INT8 会导致量化噪声沿时间步指数累积放大，严禁触碰。

4. **双轨执行契约：`TDom=1` 纯 AR 极速轨 vs `Coupled AR == Verify` 投机奇偶锁**：
   - 浮点加法不满足结合律：`M=1` SIMD32 横向树状归约（`simd_sum`）与 `M=8` NAX `matmul2d` 顺序块累加之间存在 `1–2 ULP` 尾数差异，经数十层混合 GDN + GQA 传递后会引发贪心 token 分叉。必须实施编译/调度双轨隔离：
     - **耦合验证轨（`OMLX_M5_T_DOMAIN_DISPATCH=0`，投机解码强制默认）**：启用 `DFlash2` 或 `Native MTP`（及对齐基准 `coupled_ar`）时，令 `T=1` 与 `T=2..16` 验证步共享**完全相同的窄块 NAX 算子（4-bit `BM=8` 固定功能描述符或 `Q5` `BM=16` 专核）、相同的 `s = 1..SK-1` 顺序归约以及相同的绝对位置 KV 分区边界**（且窄块遵循 **`SK > 1` 强制钳定 `RF = 1`** 防共享内存归约区膨胀、`SK <= 2` 选 `NSG = 2` vs `SK >= 4` 选 `NSG = 1`），从数学构造上锁定 **`100%`（`256/256`）`0-ULP` 贪心逐位一致**。
     - **纯自回归极速轨（`OMLX_M5_T_DOMAIN_DISPATCH=1`，`!speculative_enabled && T == 1` 默认）**：无跨 `M` 验证耦合约束时，彻底抛弃 `BM=8` NAX（消除 `7/8` 无效填充行计算与 `bf16_to_f16_rowscale` 转换遍），切入预重排直读零 `uitofp` QMV、全网 `0`-dispatch `add_rmsnorm` 原子链与 `3`-dispatch 稀疏 MoE 专核。

---

## 3. Dim A（计算受限 Prefill 压榨方法论：逼近 `92%–98%` $P_{\text{NAX}}$）

### 3.1 第一步：清除转换/工作区税负与波次感知宽块路由

- **查什么**：用 `hw_counter_probe` 检查预填阶段是否存在独立的 `bf16 -> f16` 转换核、`xsum` 激活分组求和核、`SK > 1` 归约屏障、MoE `gather_f16` 拷贝核，以及短提示词下线程组总数 `total_tgs` 相对 `gpu_cores` 的波次比。
- **为什么卡住**：独立转换与求和核在每个流式 chunk 内凭空插入上百次派发与全局读写；固定大块在短序列或窄投影上产生的 `total_tgs < 2 × gpu_cores` 会引发严重的半波尾块空转；而 MoE 若在主机端读回专家计数或显式写出 `gather` 中间张量，会同时制造 CPU 同步泡沫与显存带宽浪费。
- **怎么改算子与调度**：
  1. **原生 `bfloat` 宽块直通 + `unit_scale` 高位标志 + `SK=1` 零 `xsum`**：
     - 在 `mxfp4_nax.metal` 中直接提供原生 `bfloat` 输入的 `SK=1` 宽块线性与融合 SwiGLU PSO（`omlx_m5_mxfp4_nax_bf16_bm64_bn64_bns{16,32}_sk1`、`bm32_bn32`）；在 `half` 通路利用参数 `p.mp` 最高位 `(p.mp >> 31u) != 0u` 编码 `unit_scale` 快路径，直接跳过尾声 `row_scale[row]` 全局读取与乘法。
     - 在 `affine_nax.metal` 宽块 `SK=1` 中直接在片上将 `[scale, bias]` 反量化写入 `Ws`，彻底省去窄块所需的 `omlx_m5_affine_nax_xsum_g64` 派发（单稀疏 MoE chunk 即可削减 `~12%` 总派发数）。
  2. **`BM=64, BN=64, SK=1` 宽块 + `GROUP_M = 8` L2/SLC 二维交织 + 波次感知 `32×32` 回退**：
     - 主力预填块采用 `4` 个 simdgroup（`WM=2, WN=2`，`128` 线程/TG）协作计算 `64×64` 输出块，`SK=1` 令 Split-K 工作区与跨 SG 归约屏障降为 **`0`**；线程组二维坐标经 `GROUP_M = 8` 重排（`mxfp4_swizzle_tg_group_m8` / `affine_swizzle_tg_group_m8`），使连续调度的 8 行 TG 复用同一列权重块，L2/SLC 权重命中率提升 `8×`。
     - 通过 `nax_select_wide_prefill_tile(padded_n, t, explicit_t, gpu_cores)` 动态防止尾波饥饿：当 `64×64` 输出块总数不足以充分铺满核心（例如输出块 `< 2 × gpu_cores`，或 `mp16 < 256` 且 `tgs_64 < 4 × gpu_cores`）时，自动回切至单 SG 的 **`BM=32, BN=32, SK=1`**（`32` 线程/TG），将并行块数量放大 `4×` 喂饱全部核心。
  3. **GPU 端零同步稀疏 MoE 流式分桶**：
     - 根据统一内存工作集缩放无争用流式预填桶上限（大内存机型取 Dense `2048` 行、Sparse MoE `4096` 行；紧凑内存机型取 `512–1024` 行）。
     - 在 `moe_nax.metal` 中废除显式 `gather_f16` 与 `xsum` 中间张量（每层省 `4` 次派发）：由单核 `omlx_m5_moe_nax_bucket_schedule` 在 GPU 端一次性统计并写出 `tile_expert` / `tile_assign` / `tile_valid_rows`，按静态最坏上界发射网格并在越界桶首行 `if (tile_idx >= *total_tiles) return;` 早退；专家宽块核在寄存器中预计算源 token 偏移直读 `norm`，全程 **0 主机同步**。
- **用什么物理比例验收**：`SK=1` 宽块路径下 `bf16->f16` cast、`xsum`、`gather_f16` 及 Split-K 归约派发计数全部降为 **`0`**；短提示词（如 `L=128`）相比固定宽块获得 `+35% ~ +50%` 吞吐跃升。

### 3.2 第二步：突破共享内存与寄存器驻留墙（Zero-`As` + 成对 `Ws[64×72]` + 屏障前预取 + `<= 104` VGPR 红线）

- **查什么**：检查 Metal 编译期输出的每线程共享内存字节数（Threadgroup Memory）与通用向量寄存器数（VGPRs），计算每颗 GPU 核心实际可并发驻留的线程组数 $\text{TGs/core} = \min(\lfloor 32768 / \text{SRAM}_{\text{TG}} \rfloor, \text{VGPR\_limit})$，并观测纯 MLP 宽块核的有效算力占 $P_{\text{NAX}}$ 百分比。
- **为什么卡住**：若照搬传统双缓冲套路将激活 `As[64×72]` 与反量化权重 `Ws[64×72]` 同时搬入共享内存（耗用 `18–24 KiB` SRAM），`32 KiB` 硬上限会将驻留度死锁在 `1–2` TGs/core，导致片上解包屏障无法被并发波次掩盖，MLP 有效算力卡死在 **`~67%–68%` $P_{\text{NAX}}$**。
- **怎么改算子微架构（四项核心技术）**：
  1. **Zero-`As` 设备内存直装 `16×32×16` 硬件寄存器片段**：
     `MTLGPUFamilyApple10+` 的 `16×32×16` `relaxed_precision=true` NAX 指令每 lane 仅消费 `8` 个激活元素（两次 `bfloat4`/`half4` `16 B` 合并读取），同一 SG 内相邻行访问天然命中 L1/SLC。彻底删除 `As` 共享内存，按硬件矩阵片段坐标 `(fm, fn)` 直接从 `device` 装填激活寄存器：
     ```metal
     const uint qid = lane >> 2u;
     const uint fm = ((qid & 4u) | ((lane >> 1u) & 3u));
     const uint fn = ((qid & 2u) | (lane & 1u)) * 4u;
     device const bfloat *x_sg = x + ulong(rb_sg + fm) * p.k + fn;
     ```
  2. **成对 `16g + 16u` 紧凑共享内存 `Ws[64 × 72]`（锁定 `9,216 B` = `3` TGs/core）**：
     取 `kBK = 64`，行步幅填充 **`kBKP = 72`**（`+8` 个 `bfloat`/`half` = `16 B` 偏移，彻底消除 32-bank 共享内存写后读冲突）。在融合 SwiGLU 核中，每个 `WN=2` 列区的 `32` 行 `Ws` 交错排布 **`16` 行 `gate` + `16` 行 `up`**，使单条 `16×32×16` NAX 指令同时算出 `D[mm][0]`（`gate`）与 `D[mm][1]`（`up`），尾声直接在寄存器内完成 `SiLU(gate) * up`：
     $$\text{SRAM}_{\text{Wide\_SK1}} = 64 \times 72 \times 2\text{ B} = \mathbf{9,216\text{ B}} \implies \left\lfloor \frac{32,768\text{ B}}{9,216\text{ B}} \right\rfloor = \mathbf{3\text{ TGs/core}} \quad (12\text{ SGs/core})$$
  3. **屏障前全局预取（Prefetch-Before-Barrier）与无谓词满块主循环外提（`full_tile`）**：
     在核入口外提 `full_tile = (rb_tg + 64 <= p.rows) && (n0_tg + OUT_N_TG <= p.n)` 剥离内层越界谓词；每轮 `k0` 迭代中，**赶在第一道 `threadgroup_barrier` 之前发起全局压缩权重 `uint4` 与 `scale/bias` 寄存器预取（`raw_f`）**，使其访存延迟与并发波次的尾部 MMA 重叠，越过屏障后仅做纯算术解包写 `Ws`：
     ```metal
     for (uint g = 0u; g < groups; ++g) {
       // 1. 第一道屏障前：提前向全局内存发起压缩权重与 scale/bias 寄存器预取
       AffineRaw32<Mode> raw_f[kFetches];
       #pragma clang loop unroll(full)
       for (uint f = 0u; f < kFetches; ++f) {
         raw_f[f] = affine_fetch_raw_32_fast<Mode, BN_STORED>(
             wc_row[f], wh_row[f], sb_col[f], p.padded_n, g, half_k_f[f]);
       }
       // 2. 两道屏障间仅夹纯算术片上无损解包 -> 写入 9,216 B Ws[64 x 72]
       threadgroup_barrier(mem_flags::mem_threadgroup);
       #pragma clang loop unroll(full)
       for (uint f = 0u; f < kFetches; ++f) {
         affine_unpack_raw_32<Mode>(raw_f[f], w_dst8[f]);
       }
       threadgroup_barrier(mem_flags::mem_threadgroup);
       // 3. 零 As、零谓词：device 直读 A 片段 + Ws 读 B 片段驱动 4 条 16x32x16 NAX MMA
       #pragma clang loop unroll(full)
       for (uint kk = 0u; kk < 64u; kk += 32u) {
         affine_mma_16x32x16_bf16(D[0][0], D[0][1], a00, b00, b10);
         affine_mma_16x32x16_bf16(D[0][0], D[0][1], a01, b01, b11);
         affine_mma_16x32x16_bf16(D[1][0], D[1][1], a10, b00, b10);
         affine_mma_16x32x16_bf16(D[1][0], D[1][1], a11, b01, b11);
       }
     }
     ```
  4. **严守 `<= 104` VGPR 寄存器驻留悬崖（严禁双级软件流水 `swpipe` 反模式）**：
     - **硬件红线**：在 `MTLGPUFamilyApple10+` 上维持 **`3` TGs/core × `128` 线程 = `384` 线程/核** 的硬性寄存器上限是 **每线程 `<= 104` 个 VGPR**。
     - **致命陷阱**：若盲目套用传统 GPU 的双级跨循环软件流水（`2`-stage `swpipe`，在执行 `4` 条 NAX MMA 期间同时跨迭代存活 `cur` 与 `next` 两组压缩权重寄存器），或在寄存器内常驻大型局部缩放数组（如 `s_reg[16]`），VGPR 一旦越过 `104`，硬件调度器会静默将驻留度从 **`3` TGs/core 砍至 `2` TGs/core（损失 `-33%` 并发波次）**，直接导致 MLP 子层有效算力回落 **`~6%–8%` $P_{\text{NAX}}$**。必须保持单级屏障前预取，依靠 `3` TGs/core 硬件波次切换隐藏延迟。
- **用什么物理比例验收**：单算子宽块线性 / SwiGLU 核实测有效算力从 `~68%` 跃升至 **`~95%–98%` $P_{\text{NAX}}$**，Dense 整网矩阵有效算力达到 **`>= 90%` $P_{\text{NAX}}$**。

### 3.3 第三步：Qwen `3:1` 混合 GDN 满波量化与稀疏 MoE `compact32` 动态切分

- **查什么**：核查 GDN 块递推核的网格线程组数 `(Hv, slices)` 是否为 `gpu_cores` 的整数倍、每 lane 持久状态寄存器数是否引发栈溢出，以及稀疏 MoE 在 `256` 专家细粒度路由下 `valid_rows <= 32` 的尾桶占比。
- **为什么卡住**：Qwen 的值头数 `Hv`（常见 `32` 或 `48`）往往小于高配芯片的 `gpu_cores`（如 `64` 或 `80` 核）；若每头仅发 `1` 个 `256`-线程 TG，不仅连 `1` 个满波都填不满（核心大量闲置），还会迫使每 lane 背负 `64` 个 FP32 状态寄存器逼近溢出边缘。同时，稀疏 MoE 分桶后大量活跃桶的有效行数落在 `1..32` 之间，标准 `WBM=64`（`WM=2, WN=2`）中负责下半区的两个 SG 会完全空转浪费 `50%` 算力。
- **怎么改算子**：
  1. **波次量化 `Hv × 4`（`kDvSlice = 32`）`C=64` GDN 块递推 + `0 B`-SRAM 原位 `norm_gate`（`gdn_chunk_nax.metal`）**：
     - 在 `omlx_m5_gdn_chunk_recurrence_conv_dv32_c64` 中将 `value_dim = 128` 沿列切分为 `4` 个 `kDvSlice = 32` 切片（`128` 线程/TG，静态共享内存仅 `4096 × 4 B = 16 KiB`），网格扩展为 **`(Hv, 4)`**（在超多核芯片上若 `4 × Hv` 仍不足整数满波，可进一步切至 `kDvSlice = 16` 即 `(Hv, 8)`）：
       - 线程组数放大 `4×`（如 `Hv ∈ {32, 48}` 变为 `128 / 192` TGs，在主流多核芯片上恰好构成精确整数满波）；
       - 每 lane 持久状态压半至 **`32` 个 FP32 寄存器**（`vec<float, 8> S[2][2]`，零栈溢出）；
       - 片上直接融合 `conv1d(k=4) + SiLU`，并对 `64×64` 块内因果注意矩阵实施**下三角剪枝**（`for (uint kt = 0u; kt <= sg; ++kt)`，直接砍掉近半数上三角全零 `16×32×16` MMA）。
     - 紧接发射 **`0 B` 共享内存**的 `omlx_m5_gdn_chunk_norm_gate_inplace_{bf16,f16}_c64`（`128` 线程/TG），趁递推输出热驻片上 SLC 时单遍原位完成 `RMSNorm_128 * SiLU(z)` 与尾行清零。
  2. **稀疏 MoE `WBM=64` 片内动态 `compact32` 4-SG 并行切分（`moe_nax.metal`）**：
     在 `WBM=64` 专家宽块核内零成本动态判断分桶有效行数：
     ```metal
     const uint sg_m = sg >> 1u;
     const uint sg_n = sg & 1u;
     const bool compact32 = (WBM == 64u) && (valid_rows <= 32u);
     const uint m_off = sg_m * (compact32 ? 16u : 32u);
     ```
     当 `valid_rows <= 32` 时，`sg_m = 0` 处理行 `0..15`、`sg_m = 1` 处理行 `16..31`，同一份 `Ws` 片上反量化专家权重同时喂饱全部 `4` 个 SG，**0 额外 DRAM 读取、`0-ULP` 严格等价**。
- **用什么物理比例验收**：GDN 子层预填耗时削减 `~11%–26%`，子层矩阵有效算力跃升至 **`~78%–82%` $P_{\text{NAX}}$**；稀疏 MoE 整网预填在含路由与小桶尾耗损后仍稳定跑出高比例有效算力。

### 3.4 跨 `M5+` 核心规模与统一内存档位的 Prefill 参数化缩放表

| 调优参数维度 | `gpu_cores ∈ [10..20]` (`M5 / M5 Pro` 级) | `gpu_cores ∈ [32..40]` (`M5 Max` 级) | `gpu_cores >= 64` (`M5 Ultra` 及更高核级) |
|---|---|---|---|
| **NAX 硬件路径** | 原生 `16×32×16` Zero-`As` 寄存器直装 | 原生 `16×32×16` Zero-`As` 寄存器直装 | 原生 `16×32×16` Zero-`As` 寄存器直装 |
| **宽块 `64×64` vs `32×32` 门限** | `M >= 32` 即可满载 `64×64 SK=1`（总块数极易 `>= 2 × gpu_cores`） | `M >= 64` 选 `64×64 SK=1`；`M = 32` 按 `N` 由 `nax_select_wide_prefill_tile` 回切 `32×32` | 当 `64×64` 输出块不足铺满核心（如 `M <= 128` 且窄 `N`）时自动切 `32×32 SK=1` 防半波空转；大 `M` 全跑 `64×64` |
| **GDN `dv` 切片粒度** | `Hv × 4`（`kDvSlice = 32`），`32` FP32 状态寄存器零溢出，多波次隐藏访存延迟 | `Hv × 4`（`kDvSlice = 32`），维持 `>= 3` 波并发 | 优先使 `Hv × (128 / kDvSlice)` 对齐 `gpu_cores` 整数波次（`kDvSlice = 32` 或在 `80+` 核按 `H1-G` 实测选 `16`） |
| **流式 Prefill 桶行数上限** | 视 $W_{\text{ws}}$ 取 `512–1024`（小内存）或 Dense `1024–2048` / MoE `2048` | 取 Dense `2048` / Sparse MoE `4096` | 锁死 Dense `2048` / Sparse MoE `4096` |

---

## 4. Dim B（带宽与屏障受限 `T=1` 纯 AR 解码方法论：逼近 `99%–100%` $B_{\text{DRAM}}$）

### 4.1 第一步：根除 `1/4` 速率 `uitofp` ALU 瓶颈，以 IEEE-754 位铸造直读预重排块

- **查什么**：在 `T=1` 纯自回归（算术强度仅 `~2–4 FLOP/Byte` $\ll I_{\text{ridge}}$）下，检查 QMV 内核是否加载了行主序副本（看 RSS 是否翻倍），并反汇编或用计数器检查预重排直读核是否因整数转浮点指令（`uitofp` / `sitofp`）导致有效带宽远低于 $B_{\text{DRAM}}$。
- **为什么卡住**：若为迁就传统行主序 QMV 而额外驻留一份未重排权重，物理内存直接暴涨近一倍；而直接在 `.miv` 预重排块（`BN_stored = 16 / 32`）上解包 4/5/6/8-bit 码字时，若手写 `float(q)` 或 `half(q)`，会撞上 **Apple GPU 整数转浮点指令（`uitofp`）仅 `1/4` FMA 发射速率的微架构瓶颈**——每 lane 解包 16 个权重即阻塞数十个 ALU 周期，把内存受限算子硬生生卡在 **`~20%–65%` $B_{\text{DRAM}}$** 的 ALU 天花板上。
- **怎么改算子（Metal `27.0` 零 `uitofp` IEEE-754 位模式重解释，直接读取 `.miv` 预重排平面）**：
  1. **`MXFP4`（`E2M1`）零 `uitofp` 次正规数安全无分支解码（`row_exact.metal` / `moe_nax.metal`）**：
     将 3-bit 幅值 `(nib & 7u)` 左移 `9` 位直接拼成 IEEE-754 `half` 位模式（其中 `nib = 1` 即 `0.5` 映射为精确 FP16 次正规数 `0x0200 = 2^{-15}`，其余映射为正规数 $2^{E-15}(1 + M/2)$），经全速 `fcvt` 升 `float` 后乘以 $16384.0 = 2^{14}$ 精确还原全部 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，零分支、零查表、零 FTZ 截断：
     ```metal
     inline half pretiled_fp4_to_h16(uint nib) {
       const float f = float(as_type<half>(ushort((nib & 7u) << 9u))) * 16384.0f;
       return half((nib & 8u) != 0u ? -f : f);
     }
     ```
  2. **`oQ4e`（`Affine Q3/Q4/Q5/Q6/Q8`）零 `uitofp` 双位铸造范式与动态范围前置条件**：
     - **范式 A（生产首选零溢出通用范式：`0x6400u | q` + FP32 FMA）**：
       IEEE-754 `half` 常量 `0x6400u` 对应指数 $25 - 15 = 10$（基值 $2^{10} = 1024.0$），最低 1 ULP 恰好等于 $1.0$。对任意无符号量化码 $q \in [0, 255]$，`as_type<half>(ushort(0x6400u | q))` 精确等于 $1024.0 + q$；升为 `float` 做全速 FP32 FMA 后，在 16 元素内积循环外将 $1024.0$ 一次性代数折入修正偏置 `bi_adj = bi - 1024.0f * sc`，对任意幅度未缩放 `BF16` 隐藏态 **100% 免疫上溢**：
       ```metal
       const float sc = as_type<float>((sb_p & 0xFFFFu) << 16u);
       const float bi_adj = as_type<float>(sb_p & 0xFFFF0000u) - 1024.0f * sc;
       // 16 元素内积循环内（零 uitofp、零 shuffle）：
       d = fma(xf[idx], float(as_type<half>(ushort(0x6400u | q_code))), d);
       // 循环外一次性合成：
       acc[r] = fma(sc, d, fma(bi_adj, local_sum, acc[r]));
       ```
     - **范式 B（纯 `half4` 打包点积范式：`0x3800u` `half4` dot16 + `bi_adj`）**：
       IEEE-754 `half` 常量 `0x3800u` 基值为 $0.5$（指数 $14 - 15 = -1$）。将整数码移入尾数构造 `as_type<half>(ushort(0x3800u | (q << shift)))` $= 0.5 + q / \text{sc\_mul}$（其中 $\text{sc\_mul} = 2^{11 - \text{shift}}$），即可用双发射 `half4` FMA 计算 16 元素点积，并在循环外以修正缩放偏置抵消 $0.5$ 基值：
       $$\text{sc\_eff} = \text{sc} \times \text{sc\_mul}, \qquad \text{bi\_adj} = \text{bi} - 0.5 \times \text{sc\_mul} \times \text{sc}$$
       **严格启用前提**：仅当输入激活 $|x|$ 与 16 元素部分和严格受界于 FP16 有限上限 `±65504` 时方可启用；凡直接消费深层未裁剪 `BF16` 激活的骨干投影，一律采用范式 A（`0x6400u` + FP32 FMA）。
- **用什么物理比例验收**：替换 `uitofp` 后，预重排直读 QMV 单核带宽利用率从 ALU 受限的 `~20%–65%` 直接拉升至 **`> 97%` $B_{\text{DRAM}}$**，且省去整份行主序副本内存。

### 4.2 第二步：`oQ4e` 格式专属三项内存控制器无损榨取技术

- **查什么**：审计 `oQ4e` 混合精度模型在 `T=1` 下每步实际读取的物理字节数（是否含对齐填充废位）、`scale/bias` 元数据的跨 lane 访存步幅（是否跨缓存行/跨页），以及压缩码字的加载指令位宽。
- **为什么卡住**：
  1. `.miv` 为兼容硬件 NAX 的 `uint2b_format` 张量描述符，将 `Q5` 的第 5 个 bit 填充存储为 2-bit `nax_high2`（`256 B/tile`），导致 `T=1` 每步白读一倍全零填充位；
  2. 原始 `nax_sb` 按 `[groups, padded_n]` 存储，`KL32` 内核中单条 SIMD32 指令的 32 个 lane 横跨 `8` 个连续 group（`g_sub = lane >> 2 ∈ [0..7]`），相邻组 lane 访问地址跨度高达 `padded_n × 4 B`（大投影与词表头跨度达数十 KiB 至近 MiB），每次仅需 `16 B` 有效元数据却拉取整条 `128 B` 缓存行并打穿 TLB，造成 **`8×–16×` 跨页缓存行放大**；
  3. 使用 `1 B` `uchar` 逐字节加载压缩码会阻塞 LSU 合并队列。
- **怎么改内存平面与加载指令**：
  1. **`Q5High1Cache` 加载期 SWAR `2-bit -> 1-bit` 高位紧缩（`omlx_m5_compact_q5_high2_to_high1`）**：
     模型加载期在 GPU 上运行一次位并行紧缩核（利用 `0x55555555u -> 0x33333333u -> 0x0F0F0F0Fu -> 0x00FF00FFu` SWAR 归并），将 `256 B/tile` 的 `nax_high2` 无损压回 `128 B/tile` 的紧凑 1-bit `q5_high1`（按 `<= 1 GiB` ICB 安全段分配），直接将所有 `Q5` 层的高位平面 DRAM 流量削减 **`-50%`**（混合位宽整网总读取流量净减 `~4%–6%`）。
  2. **加载期转置 `sb_tiled [tile_n, groups, BN]`（`omlx_m5_transpose_affine_sb_to_tiled`）**：
     加载期将 `[groups, padded_n]` 一次性转置为块局部连续布局 `sb_tiled [tile_n, groups, BN]`（每 tile 每 group 紧邻 `64 B` 或 `128 B`），核内直接用单条对齐 `uint4`（`ROWS_SG=4`）或 `uint2`（`ROWS_SG=2`）向量指令合并读取，彻底消灭跨页步幅抖动。
  3. **死锁 `k_l = lane & 3` 对齐宽向量读取（严禁 `1 B` `uchar` 子字加载）**：
     固定每 `64` 元素 group 由 `4` 个 lane 协作（`g_sub = lane >> 2`，`k_l = lane & 3`，每轮迭代推进 `8` 个 group = `512` 个 K 元素），使所有位宽均编译为对齐宽字加载：`Q4/Q5/Q6` 低位主码读 `uint2`（`8 B`），`q5_high1` 读 `ushort`（`2 B`），`q6_high2` 读 `uint`（`4 B`），`Q8` 读 `uint4`（`16 B`），`Q3` 读 `uint + ushort`，全程 **0 `uchar` 加载、0 跨 lane `simd_shuffle_xor`**。
- **用什么物理比例验收**：消除跨页 `sb` 抖动与高位冗余后，`oQ4e` MLP 子层有效权重带宽从 `~85%–90%` 直接推满至 **`~100%+` $B_{\text{DRAM}}$**（含激活 SLC 命中增益）。

### 4.3 第三步：清剿全网 `1-TG` 串行孤岛与全芯片屏障（`-55% ~ -67%` 派发与屏障削减）

- **查什么**：统计单步 `T=1` Compute ICB 的 `total_dispatches`、`total_barriers`、所有网格小于 `gpu_cores` 的细碎派发，以及是否存在并发重叠核漏标读写属性引发的数据竞争或栈溢出。
- **为什么卡住**：每层两次独立的 `1-TG` `add_rmsnorm`、四次串行的稀疏 MoE 路由启动、分立的 `[Q,K,V]` / `[qkv,z,a,b]` 投影与单核 `Argmax`，在整网累积出数百次 `~0.4 µs` 屏障气泡并反复排空 GPU 流水线。
- **怎么改全网融合拓扑（四大手术）**：
  1. **全网 `0`-Dispatch `add_rmsnorm` 生产者尾声 + 消费者前奏原子归约链**：
     - **Layer-0 ICB 安全自清零（`omlx_m5_rmsnorm_a16_init_zero_t1`）**：第 0 层输入归一化在单 TG 内顺带将 GPU 私有缓冲 `rms_sq_buf[0..num_sq_words-1]` 与 `moe_tg_done_buf[0..num_moe_words-1]` 原位清零，确保 Compute ICB 无限次重放与双槽 `ar_pipe_[2]` 乒乓无需任何主机 `blit/fill` 介入。
     - **生产者尾声（`*_add_sq_t1`：`attention_o`、`gdn_out`、`mlp_down`、`moe9_down`）**：各 SG 算出输出行后原地更新残差 `h_new = bfloat(float(hidden[row]) + float(bfloat(v)))`，累计局部平方和 `sg_sq` 并由 `lane == 0` 原子累加：
       ```metal
       atomic_fetch_add_explicit(&rms_sq_buf[sq_slot], sg_sq, memory_order_relaxed);
       ```
     - **消费者前奏（`*_norm_t1`：`concat3`、`concat4`、`swiglu`、`moe9_router`、`lm_head`）**：直接读取 `inv_rms = rsqrt(rms_sq_buf[p.sq_slot] * p.inv_k + p.eps)`，在寄存器内或协作载入 `threadgroup tg_x[H]` 即时还原归一化激活（含 `final_norm` 折入 `lm_head`），每层立省 `2` 次派发与 `2` 道屏障。
     - **`T > 1` 独立 `add_rmsnorm` 编译期定长展开防栈溢出**：在验证/预填仍需调用的 `omlx_m5_add_rmsnorm_a16`（`rmsnorm_projection.metal`）中，必须按模型隐藏维 `H`（如 `H == 5120 -> bfloat4 reg_s[10]`、`H == 2048 -> bfloat4 reg_s[4]`）拆出编译期定长分支并标注 `#pragma clang loop unroll(full)`，严防动态循环索引将 `reg_s[]` 挤入慢速栈帧内存。
     - **`Encoder` `read_write` 反射屏障追踪铁律**：当无屏障并发重叠 `omlx_m5_gdn_commit_history_vec4_t1`（读写 `hist_ref, qkv_ref`）与 `gdn_out`（`*_add_sq_t1`，读写 `hidden_inout` 与 `atomic_float *rms_sq_buf`）时，`Encoder` 必须通过 Metal 参数反射将 `hidden_inout` 与 `rms_sq_buf` 严格识别为 `"read_write"` 并登记入 `written_since_barrier`，确保紧随其后的下一层 `*_norm_t1` 前必定插入一道精确 RAW 屏障，严防漏同步读取脏平方和。
  2. **同源输入投影拼接与单核状态更新合并**：
     - 全注意力层：以 `concat3_*_norm_t1` 单次发射产出 `[Q, K, V]`，紧接单核 `omlx_m5_attention_qk_norm_rope_kv_append_exact` 一次性完成 Q/K Head RMSNorm、部分 RoPE 与分页 KV 写入。
     - GDN 层：以 `concat4_*_norm_t1` 单次发射产出四路异构位宽/行数的 `[qkv, z, a, b]`，紧接单核 `omlx_m5_gdn_t1_conv_recurrent_norm_gate` 完成因果卷积、SiLU、FP32 状态递推与门控 RMSNorm，并令 `gdn_commit_history_vec4_t1` 与 `gdn_out` 并行重叠。
  3. **极致 3-Dispatch 稀疏 MoE `T=1` 子层（每层仅 `3` 次派发）**：
     - **Dispatch 1（`(N_router_tgs + 1)` TGs，如 `256` 专家取 `64 + 1 = 65` TGs，`omlx_m5_moe9_router_q8_pretiled_norm_fused_top8_t1`）**：将 `post_attn_norm + router + shared_gate + topK` 四步压为 **1 次并发启动**！其中 `tg.x = 0 .. N_router_tgs - 1` 写出 `norm_out` 并计算预重排路由 logits 写入 `router_logits_u32`，随后用设备级顺序一致栅栏与原子计数选举**最后一个完成的 TG** 在单 SG 寄存器内执行无分支 IEEE 有序键 `simd_max` `topK + Softmax`；与此同时 **`tg.x == N_router_tgs` 在额外第 `+1` 个 TG 上零等待并发算出 `shared_gate_out[0]`**：
       ```metal
       atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
       if (atomic_fetch_add_explicit(&moe_tg_done_buf[p.moe_slot], 1u,
                                     memory_order_acq_rel, mem_flags::mem_device) == (N_router_tgs - 1u)) {
         atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
         moe9_top8_from_atomic_u32(router_logits_u32, route_ids_out, route_scores_out, lane);
       }
       ```
     - **Dispatch 2（三维网格 `(F_moe / 4, top_k + 1, 1)`，`0 B` SRAM，`omlx_m5_moe9_mxfp4_kl16_swiglu_t1`）**：`tg.y = 0 .. top_k - 1` 按 `route_ids` 合并直读行主序专家平面（`128 B` 满缓存行），`tg.y == top_k` 并发计算共享专家，上千个 TG 铺满全核以隐藏随机专家寻址延迟。
     - **Dispatch 3（`H / 8` TGs，`96` 线程 = `3` SGs，`140 B` SRAM，`omlx_m5_moe9_mxfp4_kl16_down_sk3_r8_add_sq_t1`）**：多 SG 并行完成 `top_k` 路由专家加权求和、`Sigmoid(shared_gate) * shared_down`、残差原位累加与下一层 `rms_sq_buf` 原子平方和归约。
  4. **两阶段多 TG 并行 `Argmax` 与 `gpu_cores / kv_heads` 自适应 `Split-KV`**：
     - **两阶段并行采样（`position_sampling.metal`）**：`stage1` 发射多 TG（如 `64` TGs）以 `bfloat4` 并行扫描超大词表 `V` 写出局部冠军，`stage2` 单 TG 归约出严格 `0-ULP` 最小下标优先贪心 token（`temperature > 0` 无缝回退 Gumbel Top-K），较单 TG 词表扫描提速 **`~20×`**。
     - **自适应注意力分片**：当 `L_kv <= 512` 时直接绑定单遍无切分因果核（省去 `split + merge` 双派发）；当 `L_kv >= split_kv_min_context` 时切入单遍在线 Softmax `Split-KV`——门限随比值 `gpu_cores / kv_heads` 增大而前移（如 `gpu_cores / kv_heads >= 8` 时取 `2048`，`4..8` 时取 `4096–8192`，`<= 2.5` 时取 `16384`，中长上下文注意力可获 `4×–45×` 提速），并在 `L_kv >= 16384` 时启用 `512`-token 超块（较两遍归约再快 `> 5×`）。

### 4.4 物理线速时间下界（Wire-Speed Floor）与输出分块自适应验收方程

1. **单步解码物理线速下界方程**：
   设当步激活模型权重字节数为 $W_{\text{active}}$（Dense 为全网压缩权重字节数，稀疏 MoE 为常驻层 + `top_k + 1` 个活跃专家字节数），全网 GDN 状态读写字节数为 $2 S_{\text{GDN}}$，分页 KV 读取字节数为 $D_{\text{KV}}(L_{\text{kv}})$，本机 `H0` 实测持续带宽为 $B_{\text{DRAM}}$：
   $$\text{TPOT}_{\text{floor}} = \frac{W_{\text{active}} + 2 S_{\text{GDN}} + D_{\text{KV}}(L_{\text{kv}})}{B_{\text{DRAM}}}$$
2. **跨核心数输出行粒度与窄块 `SK` 自适应规则**：
   - `KL32` 预重排 QMV（`64` 线程 = `2` SGs/TG）：当输出维 `N >= 2048` 时取 `ROWS_SG = 4`（`8` 行/TG）；当 `N < 2048` 时自动降为 `ROWS_SG = 2`（`4` 行/TG），保证在高核数芯片上依然维持 `>= 3` 满波并发。
   - 耦合验证窄块 `BM=8` NAX 的 Split-K 路数随核心数缩放：`N < 65536` 骨干层按 `gpu_cores` 从 `10..20`、`32..40` 到 `64–80+` 依次取 `SK = 1..2`、`4`、`8`；`N >= 65536` 超大词表头固定取 `BN_stored = 32, SK = 1..2` 以最小化归约屏障。
3. **物理比例验收红线**：
   - Dense 模型整网单步纯 GPU 耗时进入 **$1.00\times \sim 1.05\times \text{TPOT}_{\text{floor}}$**（整网等效权重带宽达 **`95%–101%` $B_{\text{DRAM}}$**；融合 `final_norm` 的 `lm_head` 因输入向量驻留 SLC，等效带宽可达 `120%–135%` 裸 $B_{\text{DRAM}}$）；
   - 整网派发与屏障数较逐算子基线下降 **`55%–67%`**，主机提交间隙占比 `< 2%`。

---

## 5. Dim C：单流投机解码 `DFlash2` 与 `Native MTP` 压榨方法论

在 `MTLGPUFamilyApple10+` 硬件 NAX 架构上同时压榨**块扩散/外部草稿投机（`DFlash2`）**与**权重内嵌多令牌预测头（`Native MTP`）**，必须同时解决四大微架构矛盾：窄批验证步（`M = K + 1 ∈ [2..16]`）跨越标量 ALU 算力墙与位宽异构解码效率分野、草稿链与目标验证的数值契约解耦、超大词表候选筛选的栈溢出与非合并访存，以及轮间高频采样/比对/GDN 状态提交的主机同步气泡。全流程严守 **`100%`（`256/256`）`0-ULP` 贪心逐位一致（`speculative == coupled_ar`）** 硬门控，严禁以放宽验证精度换取虚假吞吐。

### 5.1 投机回合五阶段拆解、收益平衡方程与自适应深度控制（`MtpDepthController`）

- **查什么**：通过回合级计时器（`RoundCostRecorder`）与 `24 MHz` 硬件时间戳，采集每轮投机解码的五阶段正交耗时分解、各深度步条件接受率 $p_j = P(\text{accept}_j \mid \text{accept}_{1..j-1})$、期望单轮提交长度 $\mathbb{E}[\tau(K)] = 1 + \sum_{j=1}^{K} \prod_{m=1}^{j} p_m \in [1, K + 1]$，以及新请求首轮是否存在冗余 `K = 0` 测速轮：
  $$T_{\text{round}}(K) = T_{\text{draft}}(K) + T_{\text{verify}}(K + 1) + T_{\text{select}} + T_{\text{commit}} + T_{\text{glue}}$$
  对比已压至物理线速地板的 **`TDom=1` 纯自回归**单步耗时 $\text{TPOT}_{\text{PureAR}}$（严禁用慢速 `coupled_ar` 充当加速比分母），核验真实净加速比：
  $$S_{\text{spec}}(K) = \frac{\mathbb{E}[\tau(K)] \cdot \text{TPOT}_{\text{PureAR}}}{T_{\text{round}}(K)} > 1$$
- **为什么卡住**：
  1. **静态过深草稿陷阱（Over-Drafting Trap）**：链式草稿每加深一步 $K - 1 \to K$ 均需付出近似固定的单步草稿耗时 $t_{\text{draft}}$，而验证步在单个 NAX 块内对深度的耗时斜率极小；一旦尾部步的边际联合接受概率跌破时间成本比（即 $\prod_{m=1}^{K} p_m \ll t_{\text{draft}} / T_{\text{round}}$），继续锁死最大深度会白白消耗整步草稿 GPU 时间却几乎不增加 $\mathbb{E}[\tau(K)]$。
  2. **控制器开关耦合错误与跨请求重复冷预热**：若将内嵌 `MTP` 深度控制器的启用条件与 `DFlash2` 的搜索模式混为一谈，会导致默认退化为静态死深度；反之若在新请求到来时将控制器状态彻底清零（`reset()`），又会丢弃已标定的硬件周期耗时，迫使每个新请求在开头白跑多轮 `K = 0` 纯 AR 测速轮。
- **怎么改调度与自适应控制器（`MtpDepthController`，`m5rt/src/mtp_forward.cpp`）**：
  1. **独立默认启用与在线吞吐率寻优**：只要未显式指定固定草稿深度覆盖值，默认启用 `MtpDepthController`，对 $d \in [0, K_{\max}]$ 在线最大化单位时间期望令牌率 $\text{Score}(d) = \mathbb{E}[\tau(d)] / t_{\text{est}}(d)$（配迟滞切换阈值 `kHysteresis` 与尖峰阻尼 EMA），自动剪枝满足边际接受率 $p_K \ll t_{\text{draft}} / T_{\text{round}}$ 的低收益尾步；当所有 $d \ge 1$ 连续多轮落后于 $d = 0$ 基线乘以退出裕度（`exit_margin_`）时，自动平滑回切标准纯 AR 流。
  2. **硬件耗时跨请求持久化 + 序列条件接受率按请求重置（`on_new_request()`）**：严格正交分离**硅片物理常数**与**提示词语义统计量**——各深度 GPU 周期耗时 $t_d$（`t_[0..K_max]`）仅取决于本机硬件与模型结构，跨请求**持久保留**；各步条件接受率 EMA $p_j$（`p_[0..K_max-1]`）随任务领域剧烈波动，在新请求进入 `on_new_request()` 时**仅重置 $p_j$ 与探索计数器**并直接切入已校准的历史最优深度，实现温暖请求 **`0` 轮 `K = 0` 预热损耗**。
- **用什么物理比例验收**：温暖会话新请求的 `K = 0` 探测轮数严格为 **`0`**；在高低接受率分布下均自动锁定最大化 $\mathbb{E}[\tau(K)] / T_{\text{round}}(K)$ 的深度 $K^*$，避免尾部过深草稿造成的 `10%–15%` 吞吐回吐。

### 5.2 验证步（`M = 2..16`）硬件分流定律：标量 SIMD 算力墙 vs 固定功能 `cooperative_tensor` vs `Q5` 1-bit `BM=16` 专核

- **查什么**：在目标验证宽度 `M = 2..16` 下，用 `> 3× C_SLC` 冷 DRAM 单算子微探针与整网计数器对比多行标量 SIMD `KL32`、硬件固定功能 `cooperative_tensor`（`BM=8`）与片上 `Ws` 手动解包 `16×32×16` NAX（`BM=16`）在 `MXFP4`、`Q4`、`Q5` 各格式上的有效权重带宽（`% of B_DRAM`）与有效算力（`% of P_NAX`）。
- **为什么卡住（两条核心硅片分流边界）**：
  1. **标量 SIMD vs NAX 张量核交点定律（Scalar vs NAX Crossover）**：
     在 `MTLGPUFamilyApple10+` 上，全芯片标量 FP32 FMA 持续算力天花板仅约为硬件 NAX 张量核峰值算力的 **`~0.08×`（即 $P_{\text{scalar\_FP32}} \approx 0.08 \times P_{\text{NAX}}$）**。在 `T = 1` 时，算术强度低至 `~2–4 FLOP/Byte`，零溢出标量 `KL32` 纯受限于 DRAM 带宽，可跑满 **`~99%–100%` $B_{\text{DRAM}}$**；但当验证宽度增至 **`M >= 3..8`** 时，单次权重读取对应的乘加计算量放大 `M` 倍，标量 `KL32` 立刻撞上 `~0.08×` $P_{\text{NAX}}$ 的标量 FP32 FMA 算力墙，实测耗时退化至硬件 NAX 张量核的 **`> 2.0×`（慢一倍以上）**！因此 **`M >= 2` 目标验证与 `coupled_ar` 严禁启用多行标量核（实验死模板必须彻底清零），必须全量走硬件 NAX 张量核**。
  2. **4-bit 固定功能零开销解码 vs 5-bit（`Q5`）高位填充双遍惩罚分野**：
     冷 DRAM 单算子微探针揭示了 4-bit 与 5-bit 在硬件 NAX 上的根本非对称性：
     - **4-bit（`oQ4e Q4` 与 `MXFP4`）**：硬件固定功能描述符 `tensor<device uint4b_format, ...>`（`Q4`）与 `tensor<device metal_fp4_e2m1_format, ..., Mxfp4ScaleTag>`（`MXFP4`）在张量拾取管线内以 **`0` ALU 指令成本**原生解码 4-bit 权重；一旦将 `[groups, padded_n]` 元数据转置为块连续布局 `sb_tiled [tile_n, groups, BN]`（消除跨页缓存行抖动）并开启 `-O3` 编译，4-bit 固定功能 `BM=8` 核在 `Q4` 上直达 **`~98%`（`~98.6%`）$B_{\text{DRAM}}$**，而改用手动 `BM=16` `Ws` 解包反而因解包指令与屏障税负倒退 **`-12% ~ -16%`**。故 **4-bit 权重必须坚守硬件固定功能 `cooperative_tensor` + `sb_tiled`**。
     - **5-bit（`oQ4e Q5`，`bits == 5`）**：硬件 NAX 缺乏原生 1-bit 张量格式，迫使传统固定功能路径将第 5 位填充为 2-bit `nax_high2`（`uint2b_format`，**高位平面 DRAM 流量翻倍 `2×`**），且必须额外派发 `omlx_m5_affine_nax_xsum_g64` 并在核内执行 **`b4 + b2` 两遍 MMA 累加**，将 `Q5` 有效带宽死锁在 `~76%` $B_{\text{DRAM}}$。
- **怎么改算子（仅对 `Q5` 外科手术切入 `BM=16, BN=32, BK=64` `16×32×16` 专核，`m5rt/kernels/affine_nax.metal`）**：
  1. **零拷贝复用 1-bit `q5_high1` + 块连续 `sb_tiled` + `0` `xsum` 派发**：在 `AffineNaxCache::build` 中直接生成/共享紧凑 1-bit `q5_high1`（与 `Q5High1Cache` 及 `MtpQ5High1Cache` 零拷贝共用物理段，高位平面 DRAM 流量削减 **`-50%`**）与全模型线性/SwiGLU/拼接层 `sb_tiled`，并在片上解包时直接折入 `[scale, bias]`，彻底消除 `Q5` 层的全部 `xsum` 派发（**`0` `xsum` dispatches**）。
  2. **`Zero-As` `bfloat` `16×32×16` + 零 `uitofp`（`0x6400u | q5`）+ `4 KiB` XOR 无冲突 `Ws`**：
     每 simdgroup 仅分配 **`512 × sizeof(bfloat4) = 4 KiB` 共享内存 `Ws`**（`[32 × 64]` `bfloat`），合并读取 `uint4` 低 4 位码、`uint` 1-bit 高位码与 `sb_tiled`，通过 `0x6400u | q5` 零 `uitofp` IEEE-754 位铸造一步还原 `bfloat` 权重，并以 **`col4 ^ ((row & 3u) << 2u)` XOR 地址交错**写入 `Ws`（彻底根除跨行读取的共享内存 Bank 冲突）；激活 `A` 维持 **`Zero-As`** 由 `device const bfloat4` 直装寄存器，单遍调用 `affine_mma_16x32x16_bf16` 完成 `16×32×16` `bfloat -> float` NAX 累加：
     ```metal
     // 零 uitofp 还原 5-bit 权重并按 col4 ^ ((row & 3) << 2) 写入 4 KiB XOR-swizzled Ws
     const float sc = as_type<float>((sb_u & 0xFFFFu) << 16u);
     const float bi_adj = as_type<float>(sb_u & 0xFFFF0000u) - 1024.0f * sc;
     const uint xor_mask = (w_row & 3u) << 2u;
     const uint q0 = ((v >> (j * 4u)) & 0x0fu) | (((hb >> j) & 1u) << 4u);
     lo4[j] = bfloat(fma(sc, float(as_type<half>(ushort(0x6400u | q0))), bi_adj));
     ws_sg4[w_row * 16u + ((base_c4 + w * 2u) ^ xor_mask)] = lo4;
     // Zero-As 寄存器直读激活 A + XOR 无冲突读取 Ws -> 单遍 16x32x16 bfloat NAX MMA
     const bfloat4 vb0 = ws_sg4[r0 * 16u + ((kk4 + c4_lo) ^ ((fm & 3u) << 2u))];
     affine_mma_16x32x16_bf16(D0, D1, a0, b00, b10);
     ```
  3. **严格行独立构造锁定 `T=1 == T=2..16` `0-ULP` 恒等**：核内每输出行 `r` 的 FP32 累加链路（固定 `g` 分组与 `s = 1..SK-1` 顺序归约）与其他行完全解耦，越界行填 `0.0bf` 零串扰；`coupled_ar`（`T = 1`）与 `verify`（`T = 2..16`）统一绑定该精简 7-PSO 核族（`omlx_m5_affine_nax_q5_h1_bm16_bns{16,32}_sk{2,4,8}` 与 `swiglu_bm16_sk{2,4}`），从构造上保证 **`100%` `0-ULP` 行间恒等**。
- **用什么物理比例验收**：`Q4` 固定功能路径稳居 **`~98%` $B_{\text{DRAM}}$**；`Q5` 冷 DRAM 有效带宽跃升 **`+16% ~ +25%`**（SwiGLU 有效带宽推进至 **`~95.5%` $B_{\text{DRAM}}$**），整网 `M = 8` 验证纯 GPU 耗时仅比 `T = 1` 耦合基线高出 **`~4%–6%`**。

### 5.3 `Native MTP` 零中心归一化修复陷阱与解耦 `T=1` 链式草稿压榨

- **查什么**：核查 `repair_legacy_head_norms`（`m5rt/src/mtp_forward.cpp`）中施加 `+1.0f` 权重偏移修复的层名范围，以及 `MtpForward::compile_chain` 在链式步 `d >= 1`（`T = 1`）和共享超大词表头 `lm_head [1, V, H]` 上绑定的算子族与采样网格。
- **为什么卡住**：
  1. **`pre_fc_norm_embedding` 零中心修复越界陷阱（静默吞噬 `-13+` 个百分点接受率）**：
     为兼容早期未做 `+1.0f` 零中心平移的遗留检查点，加载器会对比目标主干与 MTP 头部归一化层的均值差。若不加区分地对全部 MTP 归一化层套用均值差阈值（如 `anchor_mean - mean > 0.40`），就会误触发致命静默退化——在已正确转换的模型中，**`pre_fc_norm_embedding` 的自然收敛均值完全可能落在 `~0.54` 附近**（低于主干 `input_layernorm` 均值 `~0.97` 逾 `0.40`）！对其错误叠加第二次 `+1.0f` 偏移不会触发任何运行期报错或非有限数，却会严重破坏输入嵌入与隐藏态在 `fc` 投影前的特征配比，**静默砍掉 `-13+` 个百分点的草稿接受率**！
  2. **草稿链误套验证耦合锁与单核词表扫描**：`MtpForward` 链式步 `d >= 1` 每次仅前向 `T = 1` 个草稿位置，却共享庞大的主干 `lm_head [1, V, H]`；若沿用 `BM=8` NAX 与 `1-TG` 串行采样，不仅浪费 `7/8` 矩阵行算力，还会让大词表采样单核阻塞数百微秒。
- **怎么改加载修复与草稿子图（`m5rt/src/mtp_forward.cpp`）**：
  1. **严格白名单限定 `+1.0f` 修复作用域**：在 `repair_legacy_head_norms` 中严格锁定白名单 **`{q_norm, k_norm, post_attention_layernorm, mtp.norm}`**（仅这四类头部层的原始 HF 均值可能跨过 `0.5` 启发边界而漏转）；对原始均值恒负、转换期必然已平移过的 **`pre_fc_norm_embedding`、`pre_fc_norm_hidden` 与 `input_layernorm` 硬性固定附加偏移为 `0.0f`，绝对禁止触碰**。
  2. **彻底解耦 `T = 1` 草稿链直走 Dim B `KL32` 极速轨**：
     因为所有草稿提案在下一步都会由目标验证图执行 `0-ULP` 精确裁决，**草稿头内部零跨 `M` 奇偶约束**！为 MTP 层与共享 `lm_head` 建立/复用 `MtpQ5High1Cache`（1-bit `q5_high1`）与 `sb_tiled`，将全部 `d >= 1`（`T = 1`）链式步投影（`concat3` `QKV`、`swiglu_*_norm_t1`、`o_proj`、`down_proj`、`*_add_sq_t1` 零派发原子归一化）及共享超大词表头 **`lm_head [1, V, H]`** 彻底脱离 `BM=8` NAX，全量路由至 **Dim B `KL32` 预重排直读专核**（跑满 `~100%` $B_{\text{DRAM}}$），并在跨子图录制边界强制调用 `Encoder::dynamic_parameters.clear()` 隔离动态宽度常量。
  3. **两阶段多 TG 并行草稿贪心采样**：在 `temperature == 0` 时，将链式步尾声的单核 `omlx_m5_position_sample_logits` 替换为多 TG 两阶段并行贪心 `Argmax`（`stage1` + `stage2`），采样耗时立降 **`~20×`**。
- **用什么物理比例验收**：白名单修复后草稿接受率无损恢复 **`+13+` 个百分点**；`d >= 1` 链式步与共享词表头等效权重带宽贴近 **`~100%` $B_{\text{DRAM}}$**，单步链式草稿耗时较 `BM=8` 耦合路径压缩 **`~45%`**。

### 5.4 `DFlash2` 块扩散草稿核融合与零栈溢出合并访存 `Top-K` Selector

- **查什么**：用 `metal-objdump -d` 审计块扩散草稿筛选核 `selector_topk` 是否存在栈内存溢出指令（`alloca`）及跨步幅非合并读取，并检查多层 `BF16` 块扩散骨干中 `[Q, K, V]` / 上下文 `[K, V]` 投影、动态分组卷积与 `selector_projection` 的派发和屏障拓扑。
- **为什么卡住**：
  1. **`selector_topk` 跨缓存行离散扫描与 `256 B/thread` 栈帧 `alloca` 溢出**：旧版分块筛选以 `stride-256` 跳跃读取大词表 logits（丧失 SIMD 合并访存），且每线程声明 `float scores[32]; uint ids[32];` 并用运行期动态下标执行插入排序，迫使编译器生成 **`256 B/thread` 的慢速栈内存 `alloca` 溢出**，严重拖慢草稿收尾。
  2. **重复分立投影、卷积中间读写与串行投影等待**：多层 `BF16` 草稿骨干若逐矩阵分立发射噪声 `Q/K/V` 与上下文 `K/V`、分立执行 `draft_conv -> add -> rmsnorm`，并将仅依赖 `norm` 的 `selector_projection` 串行堵在 `lm_head` 之后，会凭空累积数十道屏障与冗余全局流量。
- **怎么改算子与融合调度（`m5rt/kernels/dflash_forward.metal` & `m5rt/src/dflash_forward.cpp`）**：
  1. **SIMD 合并 `bfloat4` + 编译期 `K=16` 零 `alloca` 寄存器/SRAM 归并 Selector（`omlx_m5_selector_topk16_stage1` / `stage2`）**：
     - **Stage 1（`(64, K, 1)` 网格，`256` 线程/TG 铺满全核）**：每线程以 **`100%` SIMD 合并的 `bfloat4` 向量加载**扫描词表分块，仅在 **`4` 对纯标量寄存器 `(v0..v3, i0..i3)`** 中维护线程局部 Top-4，随后借助 **`128 B` 双缓冲共享内存**（`sg_v[2][8], sg_i[2][8]`）与全展开 `simd_shuffle_down` 在 `K = 16` 步内归并产出每块有序候选（仅当单线程 4 个局部缓存全部弹出时才带精确上界重扫一次）。
     - **Stage 2（`(1, K, 1)` 网格，`64` 线程双 warp）**：64 个线程各持 1 路预排序分块游标，在寄存器与极小 TG SRAM 中经 `16` 步双 warp 归并输出最终 `Top-16`。
     - **零栈溢出与 `0-ULP` 破平校验**：全程严守精确破平谓词 `(score > best) || (score == best && id < best_id)`，并用 `metal-objdump -d` 反汇编确认 **`0` 条 `alloca` 栈溢出指令**。
  2. **`selector_projection` 零屏障隐藏于 `lm_head` 背后**：紧接 `lm_head` 立即发射只读 `norm`（不依赖 `logits`）的 `selector_projection`，无需插入任何缓冲屏障即可与大词表 `lm_head` 尾波并发重叠执行。
  3. **`DFlashBf16QkvCache` 零冗余子切片复用 + `omlx_m5_draft_conv_add_rmsnorm` 融合**：
     - 加载期以 `omlx_m5_draft_pack_qkv_bf16` 将每层 `BF16` 权重一次性紧凑打包为 `[Q; K; V]` 单块连续缓冲；噪声块单次投影产出 `[Q, K, V]`，上下文块直接绑定偏移子视图 `kv_data = qkv_data + q_bytes`（`[K; V]`）单次投影——**`0` 字节权重重复驻留**，每层立省 `3` 次矩阵派发。
     - 以 `256` 线程/TG `bfloat4` 展开的 `omlx_m5_draft_conv_add_rmsnorm` 将后半段动态分组卷积、残差原位更新与下一子层 `RMSNorm` 三合一融合，彻底消灭中间 `delta` 全局往返并每层削减 `2` 次派发与 `2` 道屏障。
- **用什么物理比例验收**：`metal-objdump -d` 实测 `selector_topk16` 栈帧溢出严格为 **`0 B`**，且无任何双份权重膨胀；整轮多层块扩散草稿 + 词表头 + 筛选总耗时压缩至目标验证耗时的 **`~21%–27%`**。

### 5.5 单 ICB `Verify + 多行并行采样 + Equality` 与跨轮异步 `commit_prefix` 流水

- **查什么**：审计每轮投机尾声的命令缓冲提交与 `waitUntilCompleted` 同步次数、主机读回字节数、多行目标采样耗时 `spec_verify_selector_gpu_ms`、GDN 验证/提交拷贝派发数，以及非终结轮 `commit_prefix` 阻塞主机发起下一轮草稿的等待时间。
- **为什么卡住**：
  1. **分立提交同步往返与低并行度采样**：若将目标验证、多位置采样与贪心相等性判定拆成多次同步提交，每轮平白多承受 1–2 次主机同步延迟；且若目标采样仅按行数发射少量 TG，无法利用全芯片核心并行横扫 `[T, V]` 词表矩阵。
  2. **GDN 投机状态保护拷贝冗余与同步提交阻塞**：验证步在结果未决前不得覆写已提交 GDN 卷积历史 `state1`，朴素做法在全网数十个 GDN 层分别发射 `state1 -> history` 拷贝、`qkv -> replay_qkv` 保存以及提交期的多遍拷贝回放，且同步等待 `commit_prefix` 跑完才开始下一轮草稿。
- **怎么改算子与跨轮流水（`m5rt/src/target_verify_gpu.cpp`、`m5rt/kernels/gdn_layer.metal`、`m5rt/src/native_backend.cpp`）**：
  1. **单命令缓冲 `Verify + (64 × T)-TG 并行采样 + VerifyEquality` 融合（`~40 B` 极小读回，`15×` 采样提速）**：
     - 将两阶段位置耦合采样扩展为二维满核网格：`omlx_m5_position_sample_logits_stage1` 发射 **`(64, T, 1)` TGs**（全核合并 `bfloat4` 并行扫描 `T` 行超大词表），`stage2` 发射 `(1, T, 1)` 归约出严格最小下标优先的 `0-ULP` 贪心令牌（随机采样近边界自动标记紧凑回退），紧接 `omlx_m5_position_verify_equality` 在 GPU 上原地比对提案并写出授权输出 `licensed` 与 `16 B` `meta` 状态字。
     - 将目标 `verify` 与 `TargetGpuVerifier`（采样 + 相等性判定）合并压入单次命令流提交，消除额外 `waitUntilCompleted` 同步往返，整轮主机仅读回 **`~40 B`**（`16 B` `meta` + 已接受 `licensed` 令牌 ID），目标选择 GPU 耗时锐减 **`15×`**。
  2. **GDN 验证原位保存与双状态向量化合并提交（`2.0×` 提速）**：
     - **验证期融合核（`omlx_m5_gdn_conv_silu_vec4_t8_verify_save`，即 `conv_silu_spec`）**：直接只读绑定 `state1` 作为历史前缀，并在单次 `bfloat4` 因果卷积 + SiLU 遍历中同步写出 `replay_qkv`，一举省去全网 GDN 层全部 `state1 -> history` 与独立保存拷贝（每轮验证立减近百次派发）。
     - **提交期融合核（`omlx_m5_gdn_replay_prefix_commit_history`，即向量化 `commit_both_vec4`）**：在单次无层间屏障发射内用 `float4` 向量化回放已接受前缀的 FP32 状态矩阵更新（`state0`）并用 `bfloat4` 同步滑动卷积历史（`state1`），提交派发数减半、`commit_prefix` GPU 耗时直接提速 **`2.0×`**。
  3. **非终结轮 `commit_prefix` 跨轮异步前置隐藏（`0 ns` 主机等待）**：
     在所有未触发停止词且未满生成上限的非终结轮中，完成 `prepare_commit_prefix(count)` 参数补丁后立即调用 `replay_icb_async` 将 `commit_prefix` **异步挂入串行 Metal 4 队列最前端并不阻塞等待**，主机立即发布令牌并直接编码发射下一轮 `propose` 草稿！依靠队列硬件顺序保证状态一致性，待下一轮草稿返回后再以 **`0 ns` 主机等待**回收 `in_flight_commit_` 时间戳。
- **用什么物理比例验收**：多位置目标采样与比对提速 **`15×`**（占整轮耗时 **`< 0.1%`**），`commit_prefix` GPU 耗时提速 **`2.0×`** 且主机端等待降至 **`0 ns`**，每轮主机读回严格收敛至 **`~40 B`**，整轮 `select + commit + glue` 非计算/非权重总开销压至整轮壁钟的 **`< 3%`**。

---

## 6. Dim D：多流并发纯 AR 与多流投机解码压榨方法论

在多请求并发（槽位数 $B \ge 2$）场景下，在 `MTLGPUFamilyApple10+` 上压榨混合架构推理的本质矛盾，是将原本退化为 $B$ 次串行单流重读权重的低效执行，重构为**跨槽共享物理内存池、全主干与草稿头单次权重读取、零栈溢出多槽融合核与单命令缓冲闭环**；同时随跨槽打包总行数 $M_{\text{eff}}$ 攀升跨越标量/窄块分界线，精确匹配 `0-ULP` 行独立硬件 NAX 分块几何。全流程严守**逐流 `100%` `0-ULP` 贪心逐位一致（多流投机解码 `== coupled_ar`）**硬契约。

### 6.1 多流打包算术强度跃迁与跨核数物理折算方程（Cross-Core Roofline Scaling Law）

- **查什么**：审计并发槽物理 KV 页池分配方式（`SharedKvArenaPool`）、打包批次总有效行数 $M_{\text{eff}} = \sum_{b=0}^{B-1} w_b$ 对应的矩阵算术强度 $I(M_{\text{eff}})$ 相对芯片算术平衡点 $I_{\text{ridge}} = P_{\text{NAX}} / B_{\text{DRAM}}$ 的迁移区间，以及跨两款不同规格 `M5+` SKU（不同 `gpu_cores` 与 $B_{\text{DRAM}}$）对标时的硬件步耗时分量与检查点投机接受率统计量。
- **为什么卡住**：
  1. **按槽静态预留与逐槽串行提交的双重浪费**：若为每个并发槽静态划拨独立连续 KV 缓冲并循环发射独立命令缓冲，不仅产生数 GiB 物理内存碎片压低统一内存 $R$ 下的最大可准入槽数 $B$，更迫使全网权重每轮被重复读取 $B$ 次，算术强度死锁在单流 `~2–4 FLOP/Byte` 带宽地板且叠加 $O(B)$ 驱动同步泡沫。
  2. **跨 `M5+` 核数档位线性折算失真与检查点接受率混杂**：当在两款不同核数与带宽配比的 `M5+` 芯片（设高配核数 $\text{cores}_{\text{hi}}$、持续带宽 $B_{\text{DRAM,hi}}$，低配核数 $\text{cores}_{\text{lo}}$、持续带宽 $B_{\text{DRAM,lo}}$）之间进行物理性能投射与回归验收时，粗暴套用单一核数比 $\text{cores}_{\text{hi}} / \text{cores}_{\text{lo}}$ 会严重失真——多流工作负载随 $M_{\text{eff}}$ 增长在纯带宽受限与 NAX 波次计算受限之间连续过渡；同时若跨机挂载的模型制品存在修订差异，投机草稿期望接受长度 $\mathbb{E}[\tau]$ 的语义变化会与硅片步耗时混杂，造成错误的回归判定。
- **怎么改内存池与建立跨 SKU 物理折算方程**：
  1. **`SharedKvArenaPool` 零碎片共享页池与单次权重流过算术强度跃迁**：
     - 所有并发槽共享单一物理分页池 `SharedKvArenaPool`（单 slab 严格遵守 `< 2 GiB` Compute ICB 32-bit 偏移红线）与只读预重排权重池；
     - 将 $B$ 个活跃流紧致拼装为 $M_{\text{eff}}$ 行单次流过模型权重（纯自回归 $w_b = 1 \implies M = B$；多流投机验证 $w_b = K_b + 1 \implies M = \sum_{b=0}^{B-1}(K_b + 1) \approx B(K + 1)$）。设每权重元素有效位宽为 $b_{\text{eff}}$ bits，当 $M_{\text{eff}} \ll N, K$ 时，矩阵投影算术强度按 $M_{\text{eff}}$ 倍线性放大：
       $$I(M_{\text{eff}}) \approx \frac{2 M_{\text{eff}}}{b_{\text{eff}} / 8 + 2 M_{\text{eff}} \cdot (1/N + 1/K)} \approx M_{\text{eff}} \cdot \frac{16}{b_{\text{eff}}} \quad (\text{FLOP/Byte})$$
       在多流纯 AR（$M = B \in [2..8]$）下，$I(B) < I_{\text{ridge}}$ 仍由 DRAM 带宽主导，压榨核心是清零多槽转换/切分/同步税负以贴紧 $B_{\text{DRAM}}$；在多流投机验证与打包草稿（$M = B(K+1) \in [9..32]$）下，$I(M)$ 逼近或跨越 $I_{\text{ridge}}$，压榨重心转向消除跨行块重复读权并拉满硬件 NAX 利用率。
  2. **跨 `M5+` SKU 步耗时物理投射方法（Cross-SKU Step-Time Projection，$\tau_{c_1 / c_2}$）**：
     - **同二进制双物理锚点实测标定**：在两款 `M5+` 目标机上运行**完全相同的编译二进制**，实测标定两个正交极限的步耗时倍率（记 $\tau_{\text{lo/hi}} = t_{\text{lo}} / t_{\text{hi}}$）：
       $$\tau_{\text{comp}} = \frac{\text{cores}_{\text{hi}}}{\text{cores}_{\text{lo}}} \quad (\text{纯 NAX 计算/满波受限锚点，由大块 Prefill 实测核验}), \qquad \tau_{\text{bw}} = \frac{B_{\text{DRAM,hi}}}{B_{\text{DRAM,lo}}} \quad (\text{H0 冷 DRAM 与同构 TDom=1 解码锚点})$$
     - **子层带宽/算力分权合成方程**：通过 `omlx_m5_hw_counter_probe --mode model` 读取当前 $M_{\text{eff}}$ 下各子层 $i \in \{\text{MLP/MoE}, \text{GDN}, \text{Attn}, \text{Head}\}$ 的耗时权重 $w_i$ 及其带宽受限份额 $\alpha_{\text{bw}, i}(M_{\text{eff}}) \in [0, 1]$、计算/波次受限份额 $1 - \alpha_{\text{bw}, i}(M_{\text{eff}})$（随 $M_{\text{eff}}$ 从 `2` 升至 `16 -> 32`，$1 - \alpha_{\text{bw}, i}$ 单调上升），加权合成整步纯硬件折算倍率：
       $$\tau_{\text{step}}(M_{\text{eff}}) = \sum_{i} w_i \Big[ \alpha_{\text{bw}, i}(M_{\text{eff}}) \cdot \tau_{\text{bw}} + \big(1 - \alpha_{\text{bw}, i}(M_{\text{eff}})\big) \cdot \tau_{\text{comp}} \Big]$$
     - **检查点投机接受率 $\mathbb{E}[\tau]$ 独立解耦归一化**：跨机对标投机解码（`Native MTP` / `DFlash2`）时，必须将仅取决于模型权重的单轮期望接受长度 $\mathbb{E}[\tau]$ 与硬件周期耗时 $T_{\text{round}}$ 严格正交分离。若参考机与被测机检查点版本的实测单轮提交令牌数存在比值 $\eta_{\text{ckpt}} = \mathbb{E}[\tau]_{\text{ref}} / \mathbb{E}[\tau]_{\text{meas}}$，则目标 SKU 对齐投射吞吐必须同时乘以硬件步耗时倒数与接受率归一化系数：
       $$\widehat{\text{Throughput}}_{\text{lo}} = \text{Throughput}_{\text{hi}} \times \tau_{\text{round}}(M_{\text{eff}})^{-1} \times \eta_{\text{ckpt}}$$
       验收报告中须并列给出同检查点纯硬件折算值（$\eta_{\text{ckpt}} = 1$）与跨检查点归一化值。
- **用什么物理比例验收**：大块 Prefill 实测耗时比与核心数比 $\tau_{\text{comp}}$ 偏差 **`< 1%`**；跨 SKU 分权折算预测与独立冷 DRAM 微探针合成预测全面自洽。

### 6.2 稠密多流矩阵分流定律：`M=2` 专属 `KL32-BM2` vs `BM=16 / BM=32 SK=8` `0-ULP` 硬件 NAX（及三大负优化禁区）

- **查什么**：在 `> 3× C_SLC` 冷 DRAM 私有权重池探针下，逐档实测稠密线性、拼接与融合 SwiGLU 核在 `M = 2`、`M = 4`、`M = 9..16`、`M = 17..32` 下的有效权重带宽（`% of B_DRAM`）、有效算力（`% of P_NAX`）、每线程组共享内存占用（`<= 32 KiB`）、`metal-objdump -d` 栈帧分配（`0` `alloca`）以及相对单流 `BM=8 SK=8` 的逐元素 `ULP` 误差。
- **为什么卡住（两端分块失配与三大真机实测证伪的负优化禁区）**：
  1. **`M = 2`（双流纯 AR）误走 `BM=8` NAX 的填充与转换税负**：`M = 2` 退回 `BM=8` 会浪费 `75%` 矩阵行算力，并被迫插入类型转换或独立 `xsum` 派发、打断 `0`-dispatch `add_rmsnorm` 原子归约链。
  2. **`M = 9..16` 与 `M = 17..32`（多流投机验证与 `BF16` 块扩散草稿）死守 `BM=8` 触发 `2× / 4×` 权重重读**：沿用单流 `BM=8` 描述符时，网格沿行维分裂为 `2` 个或 `4` 个线程组块，导致全网压缩权重与草稿权重在每层从冷 DRAM 重复读取 **`2×` 或 `4×`**，令多流验证耗时成倍膨胀。
  3. **三大硅片实测证伪的负优化禁区（严禁实例化与合入）**：
     - **禁区一：严禁生成 `M >= 3` 多行标量 `KL32` PSO（如 `KL32 (BM = 4)`）**：受限于全芯片标量 FP32 FMA 算力墙（$\approx 0.08 \times P_{\text{NAX}}$），在 `M = 4` 冷 DRAM 实测下，`MXFP4` 四行标量 `KL32 (BM = 4)` 已落后于 `BM=8 SK=8` 硬件 NAX（仅 **`0.93×`**），`oQ4e Q4` 亦仅高出噪声边缘（`+3.7%`，远低于 `+12%` 准入红线）。**标量多行核必须硬性封顶于 `M = 2`，绝不为 `M >= 3` 生成任何标量 PSO**。
     - **禁区二：严禁启用顺序行复用循环 `_rf2 / _rf4`（`RF = 2 / 4`）**：在 `BM=8` 核内循环多个行块复用权重，会使多组目的 `cooperative_tensor` 同时驻留撑爆寄存器触发溢出，并在尾声串联多轮跨 SG `SK=8` 共享内存归约屏障（`need_pre_barrier = true`），真机多流验证反而**倒退 `+20% ~ +35%`**。
     - **禁区三：严禁将 `1-SG SK=1` 宽块门限下调至 `M = 16..32`（如 `WIDE_PREFILL_MIN_T = 16`）**：在窄 `M` 与常规骨干列宽 `N` 下切入 `BM=32, BN=32, SK=1` 或 `BM=64, BN=64, SK=1` 会令每核并发 simdgroup 跌破延迟隐藏阈值，真机耗时**暴跌退化 `+45% ~ +57%`**。
- **怎么改算子与分流路由（`row_exact.metal`、`mxfp4_nax.metal`、`affine_nax.metal`、`bf16_nax`）**：
  1. **`M = 2` 专属零溢出标量 `KL32-BM2`（`_m2`）+ `m2_sq_mul` 原子归一化桥**：
     - 严格门控于 `t_domain_dispatch_enabled() && !speculative_enabled && !verify && total_m == 2`，提供零 `alloca` 的 `_m2` 预重排直读核族（`MXFP4` `r4/r2` 与 `Affine Q4/Q5/Q6/Q8` `r2`，直连 `Q5High1Cache` 与 `sb_tiled`），单次加载压缩码字在寄存器内并行累加两行输出（`acc0, acc1`），完整覆盖 `linear`、`swiglu`、`concat3`、`concat4` 与 `lm_head`。
     - 以 `omlx_m5_rmsnorm_init_zero_m2` 配合双行尾声 `pretiled_m2_epilogue_add_sq_row` 与前奏 `*_norm_m2` 延续全网 `0`-dispatch `add_rmsnorm` 原子链，并通过步幅乘子 `m2_sq_mul = config.num_experts ? 4U : 1U` 统一对齐稠密双槽（步幅 `2`）与稀疏 MoE 多槽（步幅 `8`）在 `rms_sq_buf` 中的平方和槽位索引；同时在 `batch_enabled == true` 且活跃流 `B = 1` 时，用 `ScopedTdomOverride` 为异步流水 `ar_pipe_` 与 `run_packed_batch(B=1)` 解锁 `TDom=1` 与 `q5_high1_cache`（不改动静态 `inspect_profile(1)` 契约）。
  2. **`M = 9..16 -> BM=16`、`M = 17..32 -> BM=32` 硬件 `matmul2d_descriptor(BM, 16, 128)` `SK=8` `0-ULP` 张量核**：
     - 锁死 `RF = 1`，直接采用硬件原生宽行描述符 `matmul2d_descriptor(BM, 16, 128)`（`BM ∈ {16, 32}`, `BN = 16`, `SK = 8`；超大词表 `BN = 32` 取 `SK ∈ {1, 2}`），将 `M = 9..16` 与 `M = 17..32` 的全网冷 DRAM 权重读取次数严格砍至 **`1/2` 与 `1/4`**！
     - **`<= 28 KiB` SwiGLU 共享内存精确边界**：设每 lane 目的合作张量元素数为 `kCap = (BM × BN) / 32`。在最严苛的融合 SwiGLU `BM = 32, BN = 16, SK = 8`（双路 `gate + up` FP32 归约，`kCap = 16`）下，片上归约区字节数为：
       $$\text{SRAM}_{\text{SwiGLU, BM32, BN16, SK8}} = (SK - 1) \times 2 \times \text{kCap} \times 32 \times 4\text{ B} = 7 \times 2 \times 16 \times 128\text{ B} = \mathbf{28,672\text{ B}} = \mathbf{28\text{ KiB}} \le 32\text{ KiB}$$
       恰好卡入 `32 KiB` 线程组共享内存上限内，单次屏障完成全部 `32 × 16` 块归约，零栈溢出。
     - **`Q5` 1-bit `q5_h1_bm32` 与 `BF16` `BM=16/32` 协同补齐**：在 `affine_nax.metal` 中扩展 `omlx_m5_affine_nax_q5_h1_bm32_bns{16,32}_sk{2,4,8}` 与 `swiglu_bm32_sk{2,4}`，令每 SG `4 KiB` XOR 无冲突 `Ws` 同时服务上半区（行 `0..15`）与下半区（行 `16..31`）两组 `Zero-As` `16×32×16` NAX 指令；在 `bf16_nax` 中同步补齐 `BM ∈ {16, 32}` 核族加速打包块扩散草稿。
     - **`100%` 行独立 `ulp = 0` 逐位等价**：各输出行 $r \in [0, \text{BM})$ 在硬件描述符内沿完全相同的 `BK = 128` 分块与 `s = 1..SK-1` 顺序独立累加，与单流 `BM=8 SK=8` 及 `coupled_ar` 严格满足 **`ulp = 0` 数学逐位恒等**。
- **用什么物理比例验收**：`M = 2` 下 `KL32-BM2` 冷 DRAM 有效带宽达 **`~98%` $B_{\text{DRAM}}$**（较 `BM=8` NAX 提速 **`~1.2×–1.5×`**）；`M = 16 / 32` 下硬件 `BM=16 / BM=32 SK=8` 单算子冷 DRAM 提速 **`1.6×–2.2×`**，整网 `M = 32` 目标验证与 `BF16` 草稿骨干双双提速 **`~2.1×`**，且全流 `0-ULP` 贪心一致率 **`100%`**。

### 6.3 稀疏 MoE 多流零栈溢出 3-Dispatch `moe9_bt`（`B = 2..8`）

- **查什么**：在细粒度稀疏 MoE（如 `E = 256` 专家选 `top_k = 8` + 常驻共享专家）的 `B = 2..8` 多流纯自回归下，核查单步 MoE 总派发数、`BM=8` 分桶块内有效行占比、共享专家是否拆分为独立核序列，并用 `metal-objdump -d` 检查多流专家核是否存在栈帧分配（`alloca`）。
- **为什么卡住**：
  1. **小批量 `BM=8` 专家分桶 `> 80%` 行空转与共享专家 `7`-Dispatch 膨胀**：在 $B \in [2..8]$ 时，全批仅产生 $16..64$ 条专家路由并高度分散于 `256` 个专家中，绝大多数命中的专家桶仅含 `1–2` 行有效激活；套用 `BM=8` 分桶 NAX 会造成 **`> 80%`（`83%–85%`）的无效填充行 MMA 浪费**，且共享专家若分立执行 `f16_rowscale`、`swiglu`、`down`、`shared_gate` 与 `combine` 会额外增加 **`7` 次派发/层**，令全网 MoE 派发膨胀至 **`560` 次/步**。
  2. **动态下标累加器引发 `2×` 栈溢出反噬**：若在核内尝试按唯一专家去重并声明二维寄存器数组（如 `acc[8][8]` 或 `ag[8][2]`）做运行期动态索引，编译器会将其整块降级为慢速栈内存 `alloca`，导致单层耗时直接恶化 **`2.0×`**。
- **怎么改算子（零栈溢出 3-Dispatch `moe9_bt` 架构，`m5rt/kernels/moe_nax.metal` & `m5rt/src/moe_nax.h`）**：
  严格门控于 `t_domain_dispatch_enabled() && !speculative_enabled && !verify && rows ∈ [2..8]`，将每层稀疏 MoE 压缩为 **固定 `3` 次零 `alloca` 派发**（全网 MoE 派发从 `560` 直降至 **`120`**）：
  1. **Dispatch 1 — `grid = (65, B, 1)` 融合前奏归一化 + `Q8` 路由 + 并发共享门控 + 自清零原子尾块 `Top-8`（`omlx_m5_moe9_router_q8_pretiled[_norm]_fused_top8_bt`）**：
     - `tg.x = 0..63`（每 TG `64` 线程，`b = tg.y`）：读取 `rms_sq_buf[p.sq_slot * 8 + b]` 即时还原并写出 `norm_out[b]`，以 `0x6400u | q8` 零 `uitofp` 计算 `256` 维路由 logits 写入 `router_logits_u32[b * 128 ..]`；随后经设备级 `seq_cst` 栅栏对 **`moe_tg_done_buf[p.moe_slot * 8 + b]`** 原子加 `1`，**选举第 `64` 个完成的 TG（旧值 `== 63u`）立即将该计数器原位写 `0u` 自清零**，并在单 SG 寄存器内执行零溢出 `moe9_top8_from_atomic_u32_nospill` 产出 `route_ids_out[b * 8 .. +8]` 与 `route_scores_out[b * 8 .. +8]`——**无需任何主机重置或 Layer-0 干预即可无限次重放 Compute ICB**！
     - `tg.x == 64`：同波次并发算出第 `b` 行共享门控 `shared_gate_out[b]`。
     ```metal
     const uint done_idx = p.moe_slot * 8u + b;
     if (sg == 0u && lane == 0u) {
       atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
       if (atomic_fetch_add_explicit(&moe_tg_done_buf[done_idx], 1u,
                                     memory_order_acq_rel, mem_flags::mem_device) == 63u) {
         is_last = 1u;
       }
     }
     if (sg == 0u && simd_broadcast_first(is_last) != 0u) {
       atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
       if (lane == 0u) {
         atomic_store_explicit(&moe_tg_done_buf[done_idx], 0u, memory_order_relaxed); // ICB 自清零
       }
       moe9_top8_from_atomic_u32_nospill(row_logits_u32,
                                         route_ids_out + ulong(b) * 8ul,
                                         route_scores_out + ulong(b) * 8ul, lane);
     }
     ```
  2. **Dispatch 2 — `grid = (⌈F_moe / 4⌉, (top_k + 1) * B, 1)` 零溢出协作缓存融合 SwiGLU（`omlx_m5_moe9_mxfp4_kl16_swiglu_bt`）**：
     - 按 `b = tg.y / (top_k + 1)`、`slot = tg.y % (top_k + 1)` 解码，线程组以 `bfloat4` 协作将 `norm[b]` 装入 **`4 KiB` 共享内存 `tg_x4[512]`**；`slot < top_k` 直读 3D 行主序路由专家平面，`slot == top_k` 并发直读共享专家平面，**全程仅用显式标量寄存器 `ag0, ag1, au0, au1`（`0` 动态数组索引、`0` `alloca`）** 算出 `SiLU(gate) * up`。
  3. **Dispatch 3 — `grid = (⌈H / 8⌉, B, 1)` `sk3_r8` 加权归并 + 残差与 `rms_sq` 原子闭环（`omlx_m5_moe9_mxfp4_kl16_down_sk3_r8[_add_sq]_bt`）**：
     - 每 TG `96` 线程（`3` SGs × `3` 槽位 = `9` 路专家/行 `b`，显式标量 `pr0..pr7`），单遍完成 `top_k` 路由专家加权和与 `Sigmoid(shared_gate[b]) * shared_down` 合并、原位更新 `hidden_inout[b]`，并将平方和原子累加至 `rms_sq_buf[sq_slot * 8 + b]` 闭环衔接下一层。
- **用什么物理比例验收**：`metal-objdump -d` 核验全部 `_bt` 内核 **`0` `alloca`**；全网 MoE 派发数从 `560` 压降至 **`120`（削减 `-78.6%`）**，冷 DRAM 专家子层在 `B = 8` 下跑出 **`~90.5%` $B_{\text{DRAM}}$**（较分桶基线提速 **`1.36×–1.52×`**），逐行输出与单流 `moe9_t1` 严格 `0-ULP` 相等。

### 6.4 全打包主干消除逐槽串行循环、单 ICB 原位采样与多槽打包 TTFT 预填

- **查什么**：审计 `QwenForward::compile_packed_batch_profile`（`m5rt/src/qwen_forward.cpp`）与 `NativeBackend::advance_batch`（`m5rt/src/native_backend.cpp`）中是否存在冗余 `bf16_to_f16_rowscale` / `xsum_g64` 转换核、`emit_split2/3` 拷贝核、`for (b = 0..B-1)` 逐槽 GDN/Attention 发射循环、多段同步 CB 词表拷贝、`Encoder::dynamic` 常量去重碰撞，以及多并发新请求的串行预填排队。
- **为什么卡住**：
  1. **打包主干丢失融合引发上千次派发膨胀**：未启用 `in_f16`/`mlp_f16` 会在 `MXFP4` 上每步凭空注入 `4 × L_layers` 次 `bf16_to_f16_rowscale`；`oQ4e` 在 `total_m > 1` 下每步多出 `4 × L_layers` 次独立 `xsum_g64`；拼接投影后插入上百次 `split2/3` 拷贝，外加主机端 `for (b = 0..B-1)` 逐槽串行编码 GDN 与窄头 Attention，导致打包步派发膨胀数倍。
  2. **三段同步 CB 采样往返与动态参数去重碰撞陷阱**：将“前向散射 `sp.last_logits` -> `gather_slot_logits` -> 采样”拆成 `3` 次同步 `waitUntilCompleted` 提交，每步白白往返拷贝数 MiB 词表 logits；而在单 ICB 内为各槽编码逐槽采样核（如 `LegacyRejectionV1`）时，若初始占位 `SampleParams` 字节完全相同，`Encoder::dynamic` 会将不同槽位的采样参数静默去重折叠到**同一个常量偏移地址**，导致运行期跨槽参数覆写。
  3. **多流到达串行预填放大尾槽 TTFT**：$B$ 个并发新提示词若拆成 `2B - 1` 次串行预填提交，首字延迟随并发度线性劣化并拉低端到端吞吐。
- **怎么改算子与整图调度（四项核心收敛）**：
  1. **全通打包 `in_f16` / `mlp_f16` 与跨阶段 `carried_affine_xs` 融合归一化**：
     - `MXFP4` 窄块路径全面启用 `[add_]normalize_f16` + `linear_from_f16`（配 `f16_ones_scale`，**清零全网 `4 × L_layers` 次 `bf16_to_f16_rowscale`**）；
     - `oQ4e` 窄块路径采用 `0-ULP` 融合核 `omlx_m5_affine_[add_]rmsnorm_xsum_g64` 同步产出归一化激活与分组和，并通过 `carried_affine_xs` 跨越 `Encoder::begin_stage` 边界延续 `GroupSumKey`（**清零全网独立 `xsum_g64` 派发**）。
  2. **步幅视图零拷贝直读 + `grid.z = B` 多槽并行 GDN 与 Attention（`gdn_layer.metal` & `attention_exact.metal`）**：
     - 拼接投影输出 `concat_out` 后直接传入列偏移与行步幅参数（彻底消灭 `emit_split2` / `emit_split3`）；
     - **多槽 GDN**：纯 AR 以三维网格 `grid = (256, Hv, B)` 单次发射 `omlx_m5_gdn_t1_batched_conv_recurrent_norm_gate[_f16]` 与 `omlx_m5_gdn_commit_history_batched_vec4_t1`（子层提速 **`1.40×–1.94×`**）；打包验证以 `depth = B` 单次发射 `omlx_m5_gdn_conv_silu_vec4_t8_verify_save_batched` 接无屏障 4 槽组 `omlx_m5_gdn_recurrent_record_4row_batched4` 与 `omlx_m5_gdn_norm_gate[_f16]_strided`；
     - **多槽分页 Attention**：基于共享 `SharedKvArenaPool` 与 `pt[0..7]` 页表数组，单次发射 `omlx_m5_attention_qk_norm_rope_kv_append_batched_exact` 与 `depth = B` 的 `omlx_m5_paged_causal_attention_batched_exact[_1pass]`（按 `repeat * bq` 合并 GQA 头块复用 KV 页面），彻底铲除主机侧 `for (b = 0..B-1)` 循环。
  3. **单 ICB 原位连续采样（`run_packed_batch_and_sample`）与槽位差异化参数防去重碰撞**：
     - 省去 `sp.last_logits` 散射与二次收集，直接在主干同一条 Compute ICB 末尾对连续 `primary.logits[0..B-1]` 发射 `(64 × 256, B)` 两阶段并行采样，将每步 `3` 个同步 CB 压平为 **`1` 个单 ICB**；
     - 编译期为每个槽位的初始动态采样结构注入唯一槽位标签（如 `init_sp.seed = 0x51B00000ULL | U64(b_idx); init_sp.absolute_position = b_idx;`），确保 `Encoder::dynamic` 为每个槽分配严格隔离的参数偏移。
  4. **多槽紧致打包 TTFT 预填（Packed Multi-Slot Prefill）**：
     - 当无活跃解码流且 `>= 2` 个新到达单块提示词满足总长度 $\sum_{b=0}^{B-1} L_b \le \text{workspace\_rows}$（单槽短提示词上限保护分块公平让步）时，将其打包进单次宽块 NAX 预填提交（同时完好保留 `primary.hidden[0..total_m-1]` 供多槽 `MTP` 原位批量预热），以 `1` 次提交替代 `2B - 1` 次串行提交。
- **用什么物理比例验收**：多流纯 AR 每步同步提交数严格降至 **`1` CB**、中间 logits 散射/收集拷贝降至 **`0 B`**；并发新提示词首字延迟 `med_TTFT` 提速 **`2.6×–4.2×`**。

### 6.5 多流 `Native MTP` 解锁与打包 `[8B, H]` `DFlash2` 跨槽权重共享

- **查什么**：核查 `NativeTask::batchable()` 是否拦截活跃 `MTP` 任务、打包验证后次级槽 `sp.hidden` 是否同步获得归一化前主干隐藏态、多流草稿阶段是否对 `MTP` / `DFlash2` 骨干与超大词表头重复读取 $B$ 次、多槽 `TargetGpuVerifier` 是否串行等待，以及自适应投机控制器在批分摊计时下是否发生误熔断。
- **为什么卡住**：
  1. **`MTP` 单槽串行门锁与次级槽主干隐藏态缺失**：若 `batchable()` 拒绝 `mtp_active()` 任务，多流 `MTP` 被迫退回逐槽串行执行，每轮重读整份主干权重 $B$ 次，并发扩展倍率死锁在 **`1.00×`**；且打包验证中 `final_norm` 前的主干隐藏态仅写入主槽 `slot 0` 的 `hidden` 缓冲，若不向次级槽散射，次级槽下一轮 `MtpForward`（`cond_mode = 0`）将丢失真实主干条件输入。
  2. **逐槽回放草稿 ICB 导致草稿骨干与共享 `lm_head` 重读 $B$ 次**：仅用 `replay_icb_batch` 串联 $B$ 个单槽草稿 ICB 仍会令 GPU 重复流过 $B$ 遍多层 `BF16` 块扩散草稿权重及超大共享 `lm_head`，草稿耗时随 $B$ 线性膨胀 $B$ 倍。
  3. **逐槽验证采样同步链与分摊计时触发的虚假投机熔断**：逐槽调用 `verifier()->select()` 产生 $B$ 次同步阻塞；同时批处理将整轮壁钟按 $1/B$ 分摊计入单槽 `RoundCostRecorder` 后，若直接喂入按单流时间尺度标定的 `AdaptiveSpecController::observe_round`，会扭曲草稿与验证成本比并诱发错误门控关闭（false gating）。
- **怎么改多流投机全链路（`native_backend.cpp`、`qwen_forward.cpp`、`mtp_forward.cpp`、`dflash_forward.cpp`、`target_verify_gpu.cpp`）**：
  1. **解除 `MTP` 批处理门锁 + 打包验证 `[w_b, H]` 主干隐藏态散射**：
     - 移除 `NativeTask::batchable()` 中的 `mtp_active()` 串行限制；在 `compile_packed_batch_profile(verify=true)` 完成末层残差相加后、覆写归一化前，将每个次级槽的 `[w_b, H]` 归一化前主干隐藏行 `hidden[r_off .. r_off + w_b]` 精确拷贝至 `sp.hidden[0 .. w_b]`（主槽若处于 `r_off != 0` 则经 `delta` 安全搬移），确保全槽 `cond_mode = 0` 条件输入逐位精确。
  2. **`MtpForward::propose_packed_batch` 每深度步跨槽单次共享 `[B, V]` `lm_head`**：
     - 深度步 `d = 0` 将全批已提交折叠令牌紧致拼为 $M_0 = \sum_{b=0}^{B-1} c_b \le 5B$ 行单次流过 `MTP` 层（仅注意力按槽隔离分页 KV）并抽取各槽末行构成 `[B, H]`；后续链式步 `d = 1 .. max_k - 1` 固定以 `[B, H]` 矩阵推进；
     - 每个深度步在全部 $B$ 个槽之间**仅执行 `1` 次共享 `lm_head [B, V]` 投影**与 `(64 × 256, B)` 并行采样锁存，将词表头 DRAM 读取从 $B \times K$ 次锐减至 **$K$ 次**（配合 `MtpDepthController` 自适应收敛深度使 $B(K+1)$ 贴合 `BM=16` 单遍验证块）。
  3. **`DFlashForward::propose_packed_batch` 打包 `[8B, H]` 块扩散骨干与 `[7B, V]` 合并筛选**：
     - 将 $B \in [2..8]$ 个槽打包为 `total_rows = 8 * B` 行（增量上下文特征同样打包投影后按槽偏移追加），以 `BM=16 / BM=32` 原生 `bf16_nax` 张量核**单次读取多层 `BF16` 块扩散骨干权重**（保持每槽滑动窗口注意力与 `omlx_m5_draft_conv_add_rmsnorm_batched` 边界隔离），尾声单次执行 **`[8B, V]` 共享 `lm_head`**（命中 `BM=16 / BM=32 SK=8` NAX）与 **`(64, 7B, 1)` 零溢出合并 `Top-16` 筛选**，将 `B = 4` 草稿总耗时压缩 **`> 2.8×`**（`BF16` 骨干提速 `2.15×`）。
  4. **单 CB 批量 `TargetGpuVerifier` 与 `slot_scale` 自适应计时还原**：
     - 以 `TargetGpuVerifier::prepare_select` + 单次 `replay_icb_batch(select_icbs)` + `finish_prepared_select` 一次性完成全部 $B$ 个槽的多位置目标采样与相等性裁决，再单次批量提交 `commit_prefix`；
     - 在调用 `AdaptiveSpecController::observe_round` 前，按真实并发槽数 `slot_scale = B` 还原分摊计时（验证耗时乘以 `slot_scale`，草稿与提交耗时按跨槽共享度乘以阻尼系数 `slot_scale / (1 + c * (slot_scale - 1))` 并传入 `batch_total_m`），将多流自适应误熔断率彻底清零。
- **用什么物理比例验收**：多流 `Native MTP` 从 `B = 1 -> 4` 实现 **`~3.1×` 并发吞吐扩展**；多流 `DFlash2` 在 `B = 2`（`M = 16`）下仅付出 **`~8%–12%`** 额外轮壁钟即可验证双倍流，在 `B = 4`（`M = 32`）下整轮耗时较未打包 `BM=8` 基线压缩 **`> 2.3×`**，全流保持 **`100%` `0-ULP` 贪心奇偶一致**且自适应门控误触发率为 **`0`**。
- **异构 Dual-ANE 中转禁区（窄批 $M \le 64$ 下严禁将草稿骨干或分批乒乓卸载至 ANE）**：
  真机冷 DRAM（`> 3× C_SLC`）与零拷贝 `MTLStorageModeShared` `IOSurface` 探针证实，窄批多流投机绝对禁止向双擎 ANE 分流或乒乓中转，受制于两道硬性物理墙：
  1. **分批乒乓双倍主干读权墙**：若将 $B$ 流批次拆分为 $B/2 + B/2$ 两组子批次做 GPU 验证与 ANE 草稿乒乓流水，GPU 每轮被迫将全网目标主干权重从冷 DRAM **重复流式读取 `2` 遍**——哪怕假设 ANE 草稿耗时为理想的 `0 ms`，两次 `B/2` 验证之和也已慢于单次全打包 `B` 流验证（端到端反退 `~9%–34%`）；
  2. **片上 SRAM 溢出后的 ANE 权重 DMA 天花板与共享 UMA 带宽争用墙**：在 $M \le 64$（如多流 `MTP` $M = B$ 或打包 `DFlash2` $M = 8B$）下，数百 MiB 至数 GiB 的草稿 MLP 权重溢出单颗 ANE 引擎片上 SRAM（`~4–5 MiB/engine`）达 **`> 10×–100×`**，将双擎 ANE 锁死在其权重 DMA 天花板（仅约为全芯片 GPU 持续 $B_{\text{DRAM}}$ 的 **`~1/4` 至 `~1/3`**，单层较 GPU NAX 慢 `1.7×–4.7×`）并叠加 `IOSurface` 打包与 XPC/共享事件同步固定税负（**`~300 µs`**），同时并发 ANE 权重 DMA 会从共享统一内存控制器抢走 **`~10%–15%` 的持续 DRAM 带宽**拖慢带宽受限的 GPU 主干与词表核。**故窄批 $M \le 64$ 投机草稿必须 `100%` 留驻纯 GPU 硬件 NAX 执行**。

---

## 7. Dim E：长上下文、分组仿射 KV 量化压缩与 UMA 零搬移常驻及多轮滚动前缀复用方法论

在中长及超长上下文（`2K .. 64K+`）与多轮递进对话场景下，在 `MTLGPUFamilyApple10+` 统一内存架构上压榨 `3:1` 混合模型（不可任意回退切片的 GDN 线性递推状态 + 分页 GQA 全注意力 KV）的核心，是彻底铲除通用框架中的 **CPU↔GPU 序列化搬移与 $O(K^2)$ 虚假计费雪崩**、**多轮对话块边界截断引发的回复重预填翻倍**、**4-bit 压缩 KV 在单线程 `exp()` 或多槽串行回退下的带宽塌陷**，以及**超长预填与多令牌验证的重复 KV 读取与 `-ffast-math` 灾难性抵消**，同时在全上下文深度与单/多流多轮会话下严守 **`100%` `0-ULP` 贪心逐位一致（`AR == 投机验证`）**。

### 7.1 UMA 零搬移物理页常驻、唯一物理页计费与单里程碑锚点（`QwenUmaContinuation` & `SharedKvArenaPool`）

- **查什么**：审计前缀缓存（`SharedPrefixCache`）命中与保存时的 CPU↔GPU 字节拷贝量、单核软件哈希耗时、跨稀疏锚点的 KV 预算计费公式（是否 $O(K^2)$ 重复累加）、单流 `B = 1`（`max_concurrency == 1`）下物理页池上限是否随 `prefix_cache_bytes` 扩容、冷种子预填是否被高频同步锚点截断大块异步流水，以及暖命中尾预填是否被冗余存盘与 `last_logits` 回读阻断 `B >= 2` CoW 单命令缓冲融合采样。
- **为什么卡住（六大前缀退化陷阱）**：
  1. **百 MiB 级 CPU 往返拷贝、软件 SHA-256 与冗余 `reset()` `memset`**：每次存/取将全网数十层上百 MiB 的 GDN FP32 递推状态与草稿特征拷回主机 `std::vector` 并跑单核软件 SHA-256（白耗百毫秒级主机阻塞），且暖命中恢复前调用 `dst_prog.reset()` 对上百 MiB 状态缓冲执行冗余单核 CPU `memset`。
  2. **稀疏锚点 $O(K^2)$ 虚假预算爆炸与全量 `pinned` 驱逐雪崩**：为每 `1024` token 稀疏锚点叠加完整累积 KV 体积造成 $\sum_{m=1}^{K} m = O(K^2)$ 虚假预算爆炸，并全部标记为 `pinned`，诱发长序列（`32K .. 64K+`）共存时的缓存驱逐雪崩。
  3. **`B = 1` 物理页池饥饿**：仅在 `max_concurrency > 1` 时扩容共享物理页池，导致单流 `B = 1` 物理页总数仅够容纳单条最大序列，无法在统一内存中同时常驻多条长前缀。
  4. **重复逐页加引用与过激 `1/4` 预算腰斩**：在 `retain_prefix()` 后重复逐页递增引用计数，并因缓存持有物理页而将有效预算粗暴砍至 `1/4`，引发热前缀自我驱逐。
  5. **高频同步锚点切碎冷预填流水**：冷种子每 `1024` 步同步停顿抓快照，直接打断 `2048/4096` 大块双缓冲异步预填流水。
  6. **暖尾冗余存盘打断多槽 CoW 融合采样**：未标记 `pin_prefix` 的暖尾查询在命中已固定共享前缀后仍执行全量 `save_prefix()` + `read_last_logits_bf16()`，强行拆散 `B >= 2` CoW 的 `1`-CB 融合采样。
- **怎么改内存池与前缀调度（`qwen_forward.cpp`、`shared_prefix_cache.h`、`native_backend.cpp`）**：
  1. **回收式 `UmaCheckpointPool` 与单 CB 并行 GPU 原地快照（`QwenUmaContinuation`）**：
     - 在主槽与次级并发槽间共享 `MTLStorageModeShared` 连续块池 `UmaCheckpointPool`；
     - `checkpoint_uma()`、`restore_uma()` 与跨槽 `restore_shared_uma()` 通过**单条无内部屏障（`barrier_before = false`）的并行 `execute_batch` GPU 命令缓冲**（`uint4` `16 B` 向量拷贝 + 预编译缓存 `MetalCommandPlan` + 单调递增 `uma_epoch_id` 存活校验）原地搬移全部 GDN 状态与有效 `feature_history`，并在 `restore_into_slot` 中直接覆写目标槽状态而**跳过冗余 `dst_prog.reset()`**——实现 **`0` CPU `memcpy`、`0` 软件 SHA-256**（同时保留旧版 CPU 向量契约供防篡改测试校验），将百 MiB 级混合状态恢复压至 **`~1 ms` 量级**。
  2. **唯一物理页精确边际计费（`retained_kv_pages_`）与按预算扩容 `KvPagePool`**：
     - 以 `retained_kv_pages_` 集合（键 `{kv_allocator_id, phys_page}`）精确追踪被缓存持有的唯一物理 KV 页，新条目边际空间严格按未共享新增页 $\Delta B_{\text{entry}} = B_{\text{non\_kv}} + |P_{\text{entry}} \setminus P_{\text{retained}}| \times B_{\text{page, all\_layers}}$ 计费（彻底消除重叠前缀重复计费，仅当空闲页 `< 16` 且 `< 6.25%` 时结合在线 Mattson 栈局部性温和节流）；
     - 只要 `prefix_cache_bytes > 0`（含单流 `B = 1`），即按缓存预算与推荐工作集安全上限 $W_{\text{ws}}$ 扩容 `shared_kv_pages`；而当 `prefix_cache_bytes == 0` 时严格维持 **`0` 额外缓存页开销**。
  3. **单预尾里程碑锚点（`pre_tail`）与暖尾跳过冗余存盘**：
     - 冷种子预填全程跑满大块异步流水，仅在最接近末尾的**唯一 `1024` 对齐边界** `pre_tail = (L / 1024) * 1024`（`L >= 2048` 且 `pre_tail < L`）保存并固定共享前缀锚点（若 `pre_tail` 已固定，则完整提示词尾态以非固定存入）；
     - 在 `should_save_prefix()` 中对命中已固定前缀的未 pin 暖尾请求跳过预填末尾 `save_prefix()` 与 `read_last_logits_bf16()`，100% 保住多槽 CoW `1`-CB 融合采样。
- **用什么物理比例验收**：多档长前缀（`2K .. 64K+`）共存下缓存命中率达 **`100%`（`0` 固定条目驱逐）**，无缓存模式额外页开销为 **`0 B`**；主机 + UMA 恢复开销 **`< 2 ms`**（暖命中首字延迟较 CPU 序列化框架提速 **`~3×–10×`**，较长序列驱逐态提速**两个数量级以上**）。

### 7.2 多轮短对话 UMA 滚动前缀接力与同会话 `1×` Slab 原地覆盖（`rolling_prefix_cache` & `supersede_entry_id`）

- **查什么**：在混合架构（不可任意回退切片的 GDN 递推状态 + 分页 KV）的多轮递进短对话（Turn $1 \to 2 \to \dots \to N$）下，核查上一轮解码生成的回复令牌 $A_{k-1}$ 是否入库复用、`< 1024` 非对齐前缀是否被位置耦合分块契约拦截、跨 `1024` 边界的多槽并发是否失去打包预填，以及连续多轮是否造成 $N \times$ GDN 状态空间膨胀或孤儿块残留。
- **为什么卡住**：
  1. **块边界截断与仅存预填尾直接丢弃整段已生成回复 $A_{k-1}$**：传统混合引擎受制于 GDN 状态不可任意切片回退，仅在大块边界（如 `2048` 整数倍）或预填结束点抓快照；当 `(P + A) % block_size != 0` 时直接丢弃上一轮整段助手回复 $A_{k-1}$，迫使下一轮重预填 $A_{k-1} + Q_k$（每轮重预填令牌量直接翻倍 `2.0×`）。
  2. **严格 `1024` 块不变性契约锁死短对话 `0%` 命中率**：要求 `position % 1024 == 0` 的静态分块契约会将 `< 1024` 的系统提示词与前序短对话命中率锁死在 **`0.0%`**，并拆碎跨越 `1024` 边界的多槽打包预填。
  3. **逐轮留存旧游标引发 $N \times$ GDN 快照泄漏**：单份全网 GDN 快照定长上百 MiB；若逐轮留存同会话旧游标或任由投机草稿 `feature_history` 缓冲阶梯式申请新块而不销毁旧块，单会话连续 $N$ 轮即产生 $N$ 倍内存膨胀。
- **怎么改滚动状态机与缓存替换（`native_backend.cpp`、`shared_prefix_cache.h`、`qwen_forward.cpp`）**：
  1. **全路径静止点精确令牌账本（`committed_tokens_`）与解码尾端 `save_post_decode_prefix()`**：
     - 在预填、同步/异步双缓冲自回归（含停机前 `drain_in_flight_ar()` 排空追平 `in_flight_ar_token_`）、`DFlash2` / `Native MTP` 的 `commit_prefix(count)` 及多槽 `advance_batch` 的每个状态提交点精确追加已写入 GPU 状态的令牌，严格维持静止点不变量 **`committed_tokens_.size() == program().position()`**；
     - 在 `rolling_prefix_cache` 模式下跳过非固定多步生成轮次预填末尾的 `save_prefix()`（保住 `1`-CB 融合采样与 `0` `last_logits` 回读），改在 `EventKind::finished` 尾端以单条并行 GPU UMA 拷贝 CB（**`~0.2 ms`**，`0` CPU `memcpy`）写入精确前沿 $P_k + \text{committed}(A_k)$（即 $|P_k| + |A_k| - 1$），令第 $k+1$ 轮仅需预填上一轮尾末 `1` 个生成令牌与新提问 $Q_{k+1}$。
  2. **同会话 `1×` UMA Slab 原地覆盖（`supersede_entry_id`）与草稿 `32 MiB` 量子化无孤儿扩容**：
     - `SharedPrefixCache::insert_checkpoint(..., supersede_entry_id)` 在边际预算计费与 `checkpoint_uma()` **之前**，若匹配到非 `pinned` 的严格前缀祖先条目则先行调用 `release_entry` 释放——其 `uma_slab` 立即归还至 `UmaCheckpointPool::free_slabs` 被本轮快照**零分配 `1×` 原地复用**（CoW 分叉旧尾页同步回收，而 `pinned` 共享系统提示词受绝对保护**永不被覆盖或驱逐**），配合 `max_unpinned_entries = max(8, 2 * max_concurrency)` 上限治理多会话轮转；
     - 对携带增长型 `feature_history` 的投机草稿快照按 **`32 MiB` 量子对齐**容量，并在扩容分配前从空闲链表弹出销毁过小旧块，实现全程 **`0` 孤儿缓冲**。
  3. **`effective_prefill_contract()` 解锁任意边界命中与跨 `1024` 打包预填**：
     - 在 `rolling_prefix_cache` 模式下保持采样与验证端耦合契约不变，而向 `find_best()` 与单/多槽预填分块门控传入任意边界兼容策略，全面打通 `< 1024` 短对话精确命中与跨 `1024` 边界的 `B >= 2` 单次打包预填。
- **用什么物理比例验收**：多轮短对话后续轮次重预填令牌数较仅存预填尾基线**直接减半（`~2.0×` 削减）**，温暖首字延迟提速 **`~2.4×–3.0×`**；纯 AR 连续 `10+` 轮物理内存增长 **`~0 MB`（严格维持 `1×` GDN 快照体积，`0` 固定前缀驱逐，`0` 孤儿块）**，单/多槽多轮投机解码保持 **`100%` `0-ULP` 逐位一致**，且单轮 `B = 1` / `B >= 2` CoW 延迟 **`0.0%` 回归**。

### 7.3 分组仿射压缩 KV（`FOSCAR` `Affine4Group128`）单流与多流（`grid.z = B`）统一 2-SG 硬件 `uint4b_format` NAX Split-KV（`TDom=1` 全通 & `0-ULP` `T=1 == T=8`）

- **查什么**：在 `> 3× C_SLC` 冷 DRAM 私有分页 KV 池下，先用 `~50` 行独立单算子微探针实测 4-bit 分组仿射压缩 KV（`KvStorageFormat::Affine4Group128`，每 `head_dim = 256` 向量占 `144 B` = `128 B` `uint4` 码字 + `16 B` `half` `[scale, zero]`，较原生 `BF16` 削减 **`~3.5×–3.8×` KV DRAM 流量**）在单流 `T = 1` 解码、`T = K + 1`（`2 <= T <= 8`）多令牌验证及 `B >= 2` 多槽打包下的冷 DRAM 有效带宽（`% of B_DRAM`）、活跃线程占比、每页屏障数、`metal-objdump -d` 栈帧分配（`0` `alloca`）及 `T = 1` vs `T = 8` 逐行误差，并核查纯 AR 长序列运行期与评测管线是否完整开启 `TDom=1` 及多档上下文包络（`ar_envelopes` 补齐 `16384, 32768`）。
- **为什么卡住（微探针实测证伪的两大算子陷阱与多槽退化）**：
  1. **单线程串行 `exp()` 循环与纯标量 SIMD 反量化双双撞墙**：若在 `16×64×128` NAX 分数块后仅用单线程串行跑 `64` 次 `exp()`（`> 90%` 线程空转 + 每页 `3` 次屏障），有效带宽跌至 **`< 15%` $B_{\text{DRAM}}$**，令 4-bit KV 反而慢于原生 `BF16` KV；而纯标量 SIMD 位铸造点积核在 `MTLGPUFamilyApple10+` 上同样受限于标量 `simd_sum` / ALU 吞吐墙（实测仅达硬件张量核的 **`~1/3`** 速度）。
  2. **多流 `packed_batch` 退回逐槽串行非分片窄核**：若 `compile_packed_batch_profile` 将批处理注意力硬门控于 `BFloat16`，压缩 KV 在 $B \ge 2$ 时被迫退化为 $B$ 次带屏障的串行 `kv_heads` 窄网格调用，令 **`> 90%` GPU 核心闲置**、多流吞吐腰斩。
- **怎么改算子与多槽派发（`attention_affine4.metal`、`attention_nax.h`、`qwen_forward.cpp`）**：
  1. **融合前奏 `qk_norm_rope_kv_append_pack_q[_batched]_affine4_g128` 与零死代码清理**：
     - 删除 `kv_append` 中无下游读者消费的冗余整页子带重扫描；
     - 在单流 `T = 1` 与多流 `B = 2..8`（`pt0..pt7` 页表数组选择 + 步幅直读 `concat_out`）中单次派发同步完成精确 `precise::` QK 归一化、RoPE、4-bit 分组仿射量化入页，并将预缩放 `half` 查询 `[KVH, 2, R_stride, 128]` 与 FOSCAR 行和 `q_sum` 直接打包进既有 `split_output` 上半区（**`0` 额外 `QwenMemoryPlan::state_bytes` 内存开销**，每步省去全网 `full_attn` 层独立 `kv_append` 与 `pack_q` 派发及屏障）。
  2. **单流与 `grid.z = B` 多流统一的零溢出 2-SG 硬件 `uint4b_format` `16×64×128` NAX Split-KV（`a5_nax_2sg[_batched]_impl`）**：
     - 采用 `2` 个 SG（`64` 线程，`8.5–13 KiB` 共享内存，零 `tg_q`、单一寄存器常驻 `c_acc` `mode::multiply_accumulate` 实现 **`0` `alloca` 栈溢出**），直接调用硬件固定功能 `tensor<device uint4b_format, ...>` `16×64×128` `matmul2d` 解码 4-bit KV 页；每页以 `4` 线程/行并行 `fast::exp2` 在线 Softmax 将 `v_meta.scale` 折入 `tg_p`、`v_meta.zero` 折入 `tg_vb`（每页仅 `2` 道屏障）；
     - 单流 `T = 1` 取 `R_BLK = 6`（`r6`，`grid.z = 1`），多令牌验证 `2 <= T <= 8` 取 `R_BLK = 16`（`r16`，`grid = (splits × 64, kv_heads, ⌈T × repeat / 16⌉)`，跨验证令牌与 GQA 头单遍共享 KV 页读取），多流 `packed_batch` 沿 **`grid.z = B`** 单次并发覆盖全部活跃槽；
     - **铁律屏障与 `0-ULP` 行独立一致**：**必须在分页主循环结束后插入一道 `threadgroup_barrier(mem_flags::mem_threadgroup)`，再由 `sg = 0` 复用片上 `tg_rs` 存放 `1.0f / tg_l` 倒数**（否则 `sg = 0` 提前覆写 `tg_rs` 会与 `sg = 1` 末页 Phase 3 `fma(c_acc[i], tg_rs[row], ...)` 发生跨 SG 数据竞争）：
       ```metal
       for (uint page = start_page; page < end_page; ++page) {
         op_qk.run(tq, tk, c_qk);                         // Phase 1: HW uint4b_format 16x64x128 Q@K^T
         threadgroup_barrier(mem_flags::mem_threadgroup); // 页内屏障 1/2
         // Phase 2: 4 lanes/row 并行 fast::exp2 在线 Softmax + V scale/zero 折入 tg_p / tg_vb
         threadgroup_barrier(mem_flags::mem_threadgroup); // 页内屏障 2/2
         op_pv.run(tp, tv, c_acc);                        // Phase 3: 原地重缩放 c_acc + 16x128x64 P@V
       }
       // 铁律屏障：双 SG 必须全部完成末页 Phase 3 后，方可复用 tg_rs 写入归一化倒数！
       threadgroup_barrier(mem_flags::mem_threadgroup);
       if (flat_tid < active_rows) {
         tg_rs[flat_tid] = (tg_l[flat_tid] > 0.0f) ? (1.0f / tg_l[flat_tid]) : 0.0f;
       }
       ```
     - 写出归一化 `half` 分片后统一接入 **`4`-SG 对数并行 `half` 归并核 `omlx_m5_paged_attention_affine4_g128_split_merge_half_4sg_hd256`**。由于 `T = 1`（`r6`）、`T = 8`（`r16`）与多流 `batched` 统一采用 **`s512`（`512`-token）分片**与完全相同的硬件 `16×64×128` NAX 归约树，**天然锁定全上下文深度（`2K .. 64K+`）与单/多流下的 `100%` `0-ULP` 逐位恒等（`AR == 投机验证`）**！
- **用什么物理比例验收**：`T = 1` 长序列压缩 KV 分片核冷 DRAM 有效带宽达 **`~70%–82%` $B_{\text{DRAM}}$**（算子提速 **`> 5×–6×`**），使 4-bit 压缩 KV 在 `64K` 长序列下的整网纯 AR 吞吐反超原生 `BF16` KV **`> 20%`** 且从短序列扩展至 `64K` 整步衰减 **`< 5%`**，无缓存污染峰值内存紧贴单份权重 + 压缩 KV 理论下界；$B \ge 2$ 多流压缩 KV 解码吞吐翻倍且跨轮次保持**零衰减水平线**。

### 7.4 超长上下文寄存器常驻 `Frag16x16` 分块因果 `Flash-NAX` 预填与 `BF16 Frag48` 多令牌验证（及三大微探针证伪禁区）

- **查什么**：核查超长序列（`32K .. 64K+`）预填在压缩 KV 下是否误退回窄 `bq = 2` 解码核、原生 `BF16` 多令牌投机验证（`T = K + 1`）是否因线程组 SG 上限而重复读取全量长 KV 缓冲多遍，以及编译选项与 4-bit 反量化表达式在 `-ffast-math` 下的逐元素精度。
- **为什么卡住（三大微探针真机证伪禁区与长验证重读墙）**：
  1. **压缩 KV 漏接 `Flash-NAX` 引发十亿次串行 `exp()` 灾难**：若压缩 KV 在大块预填入口提前返回退走窄块解码因果核，超长序列冷预填耗时将恶化 **`> 2×–4×`**。
  2. **禁区一：严禁在 `-O3 -ffast-math` 下对 4-bit 半字节使用 `0x6400` 魔数减法位铸造（`fma(x - 1024.0h, s, z)`）**：编译器会将该式代数重结合为 `fma(x, s, z - 1024.0h * s)`，在 FP16 下诱发灾难性大数吃小数抵消误差（单元素最大绝对误差高达 **`0.31`**）；而若在注意力预填核关闭 fast-math（`-fno-fast-math`），`Frag16x16` 预填有效算力又会暴跌**近四成**。
  3. **禁区二：严禁在 `Frag16x16` 预填或 `T = 8` 多查询核中引入 `16 KiB` `tg_kv` 搬移或将 `S / P` 矩阵往返落盘共享内存**：片上并发驻留度腰斩与高频屏障会使有效算力反退 **`1.7×–3.0×`**。
  4. **`BF16` `T = 8` 验证 `max_sgs = 16` 门锁导致 `4×` 全量 KV 重读**：当 `repeat = 6` 时，`bq` 被卡死在 `⌊16 / 6⌋ = 2`（裂变为 `4` 个查询块），每轮投机验证从冷 DRAM **重复搬运 `4` 遍**数 GiB 的长上下文 `BF16` KV 缓冲。
- **怎么改预填与长验证算子（`attention_affine4.metal`、`attention_nax_prefill.metal`、`m5rt/CMakeLists.txt`）**：
  1. **作用域隔离 `-O3 -ffast-math` + 寄存器常驻 `Frag16x16` 压缩/原生 `Flash-NAX` 预填（`omlx_m5_paged_flash_prefill[_batched]_affine4_g128_hd256` & `omlx_m5_paged_flash_prefill_nax_hd256`）**：
     - 在 `m5rt/CMakeLists.txt` 中通过 `FAST_MATH` 选项**仅对 `attention_affine4.metal` 与 `attention_nax_prefill.metal` 局部启用 `-O3 -ffast-math`**（其余算子库严格保留 `-fno-fast-math`），并用有限常量 `kNegInf` / `kNegInfHalf` 与整数位掩码 `(bits & 0x7fffu) > 0x7f80u` 替代 `-INFINITY` / `isnan`；
     - 在 `BQ = 64, BK = 32, WM = 4, WN = 2`（`256` 线程/TG，支持单流与 `grid.z = B` 多槽打包尾预填）压缩 `Flash-NAX` 中，4-bit 半字节改用**无抵消精确无符号转换 `half4(ushort4(...))`** 配合预计算行指针、零 `tg_kv` / 零 `tg_p` 寄存器驻留 `Frag16x16`（仅 **`8.4 KiB`** 3-屏障 `xchg` 跨半头归约区 = **每核 `2+` TGs 驻留**，`0` `alloca`）与满速率 `half × half -> float` NAX `P@V`；原生 `BF16` 同步升级 `bfloat4` 向量化读写、因果满块快路径外提 `full_causal_free` 与满速 `bfloat × bfloat -> float` NAX `P@V`，**同时兼得满血张量算力与 `<= 1 BF16 ULP` 精度**：
       ```metal
       // 局部 -O3 -ffast-math 下零抵消误差的精确 4-bit 半字节寄存器反量化 -> 满速 half*half NAX
       const ushort pk0 = p0[col_off];
       const half4 q0 = half4(ushort4(pk0 & 0xFu, (pk0 >> 4u) & 0xFu,
                                      (pk0 >> 8u) & 0xFu, pk0 >> 12u));
       const half4 v0 = fma(q0, half4(s0), half4(z0));
       ```
  2. **`BF16 Frag48` 单遍 KV 共享长序列验证核（`omlx_m5_paged_attention_bf16_frag48_split_partial_hd256`）**：
     - 以 `WM = 3, WN = 2`（`192` 线程 = `6` SGs/TG，仅 **`12 KiB` `xchg`** = **每核 `2` TGs 驻留**，`0` `alloca`）将 **`8` 个验证令牌 × `6` 个 GQA 头 = `48` 条查询行**全部装入单一线程组寄存器，在每个分片内**单次（`1.0×`）流过冷 DRAM 中的 `BF16` KV 页**（`L <= 32K` 取 `s1024`，`L > 32K` 取 `s2048`），后接 `4`-SG 对数并行 `half` 归并；
     - 安全门控于长上下文 `flash_nax_prefill && !strict_two_pass && env >= 4096 && hd == 256 && repeat == 6 && 2 <= t <= 8`，以零风险保护短序列严格两遍契约。
- **用什么物理比例验收**：超长序列（`32K .. 64K+`）下原生 `BF16` 与 4-bit 压缩 `Flash-NAX` 预填单算子有效注意力算力分别达 **`~55%–60%` 与 `~50%+` $P_{\text{NAX}}$**（压缩 KV 超长冷预填提速 **`> 3.2×`**，整网预填与原生 `BF16` 差距收敛至 **`< 4%`**）；`T = 8` 长序列验证注意力子层冷 DRAM 有效带宽达 **`> 70%` $B_{\text{DRAM}}$**（子层提速 **`~4.0×–4.6×`**），从短上下文扩展至 `64K` 整轮投机验证步耗时增量控制在 **`< 10%`** 以内。
