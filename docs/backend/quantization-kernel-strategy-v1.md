# Qwen3.6/3.8 Dense/MoE M5 backend 量化格式与算子优化策略

> 版本：v1.5
> 日期：2026-09-21
> 状态：R1 设计基线
> 上位计划：`../../omlx-m5-backend-nlnfer-项目实施计划书.md` v4.13
> 目标：明确 NInfer groupwise-int、NInfer NVFP4、oQ4e 与 Apple MXFP4 的数值合同、优化方式、可复用设计和不可复用实现。

> 2026-09-23 v5.1状态补充（及 `2026-11-21` `v0.8.0-m5rt.4` 更新）：下文SDK26.2的“无公开FP4 operand”属于历史实现说明。macOS 27 SDK已提供FP4 E2M1、E8M0辅助plane、`int4b`与`uint4b/uint2b`，且已在64核M5 Ultra真机全量激活（含 `q3` 与 QDA `3.55 bpw`，详见 [`ninfer-m5-porting-architecture-v2.md`](ninfer-m5-porting-architecture-v2.md)）。参见[硬件边界](../../hardware-software-boundaries-v1.md)与[历史实施表v5.1](../versions/v0.1-v5.1-pre-silicon/plans/pre-post-m5ultra-implementation-plan-v5.1.md)。

---

## 1. 最终产品决策

R1 只实现两个权重 Profile：

| Profile | 权重 | 执行配置 | 定位 |
|---|---|---|---|
| A | oQ affine Q4/Q5/Q6/Q8 G64/G128 | W4/5/6/8A16 | 速度优先且原样保留checkpoint混合精度的decode主线 |
| B | MXFP4 E2M1 + E8M0 G32 | W4A16、W4A4 | 速度优先、TensorOps/NAX 主线 |

MXFP8 延期，不进入 R1 artifact、kernel catalog 和目标机调优矩阵。NInfer 的
groupwise-int 与 NVFP4 是设计参考，不是 R1 运行时格式。

Profile B使用四份现有权重作为输入源，不自行量化：

| 输入源 | 固定 revision | 用途 |
|---|---|---|
| `OsaurusAI/Qwen3.6-27B-MXFP4` | `3258bea991decf16bf84fcd21cd73ae089bf3bbc` | Dense uniform MXFP4速度候选 |
| `OsaurusAI/Qwen3.6-35B-A3B-mxfp4` | `5a6aaf3d0db3590f186d743b99c9b9d270ec7857` | MoE expert-major MXFP4 + affine Q8 gate |
| `mlx-community/Qwen3.8-27B-mxfp4` | `97ab0819817ab1c61d7d39f9169fc71999915641` | Qwen3.8 Dense uniform MXFP4速度基线 |
| `RedHatAI/Qwen3.8-27B-MXFP4` | `651eed99386dae5645cd4949024e8ebd163a533b` | AWQ smoothing + GPTQ 校准权重，质量对照与第二性能候选 |

四份MXFP4权重必须离线规范化到同一个backend artifact合同；它们共享kernel，
但不共享量化 code、受保护层清单或性能结论。

---

## 2. 必须分开的四层合同

量化名称不能代替完整执行合同。backend 必须分别记录：

1. **数值格式**：code、scale、bias、group 和重建公式；
2. **量化 recipe**：RTN、absmax、AWQ、GPTQ、imatrix clipping 以及受保护层；
3. **物理布局**：plane、packing、padding、alignment、row/tile ordering；
4. **compute profile**：A16/A4、累加类型、SIMD/TensorOps、融合和 dispatch。

同为 MXFP4 只表示 E2M1 code + E8M0/G32 scale 合同一致，不表示两个 checkpoint
的 code/scale 值或 BF16 保护层一致。相反，同一份 MXFP4 权重可以在运行时同时支持
W4A16 与 W4A4，因为 activation policy 不属于持久权重格式。

---

## 3. 四种量化体系的准确差异

| 项目 | NInfer groupwise-int | NInfer NVFP4 profile | oQ4e | Apple MXFP4 |
|---|---|---|---|---|
| code | signed INT4/5/8 | E2M1 FP4 | unsigned INT4/5/6/8 | E2M1 FP4 |
| group | Q4/Q5 G64 | G16 | G64/G128 | G32 |
| scale | FP16/group | E4M3/group | BF16/group | E8M0/group |
| bias | 无 | 无 | BF16/group | 无 |
| global field | 无 | FP32 divisor/matrix | 无 | 无 |
| 重建 | `q_signed * scale` | FP4 × block/global scale | `q_unsigned * scale + bias` | E2M1 × E8M0 scale |
| activation | 主要 A16 | A4 或 A8 | A16 | A16 或动态 A4 |
| 优先执行 | CUDA SIMT/MMA | Blackwell FP4 Tensor Core | Apple SIMD + custom TensorOps | Metal quantized TensorOps/NAX |

逻辑 payload 成本（不含对象 padding）为：

```text
NInfer Q4 G64  = 4 + 16/64 = 4.25 bit/weight
NInfer Q5 G64  = 5 + 16/64 = 5.25 bit/weight
oQ4e Q4 G64    = 4 + 32/64 = 4.50 bit/weight
oQ4e Q5 G64    = 5 + 32/64 = 5.50 bit/weight
oQ4e Q5 G128   = 5 + 32/128 = 5.25 bit/weight
oQ4e Q6 G64    = 6 + 32/64 = 6.50 bit/weight
oQ4e Q8 G128   = 8 + 32/128 = 8.25 bit/weight
Apple MXFP4    = 4 + 8/32  = 4.25 bit/weight
```

NInfer Qwen3.8 groupwise recipe 按角色固定：attention query/key、GDN query/key、
MLP gate/up 为 Q4，其余主投影为 Q5，词表权重为 Q8。NInfer NVFP4 profile 则主要
在前 56 层 MLP 使用 NVFP4，其余大量投影使用 row-scaled FP8。因此“NVFP4 profile”
不是全模型纯 NVFP4。

oQ的Q4/Q5/Q6/Q8分配和每组scale/bias来自oQ混合精度与oQe imatrix加权clipping；
它不能无损降为 NInfer scale-only groupwise-int。MXFP4 同样不能与 NVFP4 互换：二者
group、scale 类型和 global divisor 合同不同。

---

## 4. NInfer groupwise-int 的优化方式

NInfer groupwise-int 是 oQ4e 最接近的架构参考，因为两者都是小 group 的 Q4/Q5
权重，并且都需要按 T 选择 QMV、SIMT 或矩阵路径。

### 4.1 持久布局

NInfer `row_split_k128_v1` 将权重组织为：

```text
base-code plane
256-byte alignment
optional high-bit plane       # Q5/Q6
256-byte alignment
FP16 scale plane
```

K 物理维扩展到 128 的倍数，row view 由独立 code/high/scale spans 组成。converter
完成 padding 与 plane packing，loader 只上传，不运行时 repack。

### 4.2 `T=1` QMV

- 一个 warp/线程组负责一个或若干输出行；
- packed code 与 scale 直接顺序读取；
- 每个 code group 只解码一次；
- activation 向量与解码权重做 FP32 累加；
- 使用专用 epilogue 直接发布 split output、norm、conv 或其它后续输入；
- 不为了使用矩阵单元而制造低占用的伪 GEMM。

### 4.3 small-T row-split

- 一个 warp 拥有一个输出行和多个 token 列；
- code/scale 分 stage 载入 shared memory 或寄存器；
- 一次解码出的 FP32 权重跨多个 T 列复用；
- activation 按列读取，权重不按列重复读取；
- full/tail contract 分开，完整 tile 移除边界判断。

### 4.4 Q5 split-K 与 row-block

Q5 多一个 high-bit plane，解码和寄存器使用更重。NInfer 在 small-T 下通过：

- `T=2..6` split-K×4 增加并行度；
- `T=7/8` row-block/activation staging 降低激活重复读取；
- `T=9..12` 容量型 row-split；
- 更大 T 转向 MMA/broad routes。

### 4.5 大 T

K-split/MMA 或 broad tile 负责 `T=13..128` 与 bulk。host 只做静态 route lookup；
kernel instance、capacity 和 seam 由离线实测决定。

---

## 5. NInfer NVFP4 的优化方式

NVFP4 路径的核心不是“另一种 Q4 解码”，而是硬件原生低精度矩阵计算：

1. E2M1 weight + E4M3/G16 scale 作为 Tensor Core operand；
2. 可选 activation A4/A8；
3. activation divisor/scale 在 use 级别绑定，不与共享权重错误共用；
4. W4A4/W8A8 路径面向 compute-bound small/bulk GEMM；
5. QMV、small-T、W4A4 和 broad route 分开；
6. fused Linear+residual、Linear+SwiGLU、attention/GDN input projection 减少中间流量；
7. artifact 保留 kernel-native block-scale 布局，运行时不重排 scale tiles。

NInfer Qwen3.8 NVFP4 recipe 对 NVFP4 use 标记 `AllowA4`，对 row-scaled FP8 use 标记
`AllowA8`。Program 可以为同一输入只生成一次量化 activation，再供多个投影共享。

这套架构可供 Apple MXFP4 参考，但 NVFP4 的 E4M3/G16 + divisor decode 和
Blackwell TMA/Tensor Core tile 不能复制到 Metal。

---

## 6. oQ4e 的详细优化方式

### 6.1 数值与 artifact 布局

R1定义affine格式族：

```text
oq_affine_q4_g64_bf16_v1
oq_affine_q{4,5,6,8}_g{64,128}_bf16_v1
```

推荐物理 plane：

```text
Q4: base-code | alignment | BF16 scale | alignment | BF16 bias
Q5: base-nibble | alignment | high-bit | alignment | BF16 scale | alignment | BF16 bias
Q6: base-nibble | alignment | high-2bit | alignment | BF16 scale | alignment | BF16 bias
Q8: byte-code | alignment | BF16 scale | alignment | BF16 bias
```

Q5/Q6从MLX连续bitstream离线无损拆成base/high planes，Q8保留byte code；converter保留原code、
scale 和 bias 数值，生成 golden vectors。可将 unsigned code 重新居中为 signed code，
但必须相应调整 bias，且不能删除 bias plane。

### 6.2 `T=1` affine QMV

直接逐权重执行 `x * (q*scale+bias)`会重复做 64 次 bias 工作。按 group 改写：

```text
sum(x_k * (q_k * scale_g + bias_g))
= scale_g * sum(x_k * q_k) + bias_g * sum(x_k)
```

对每个 token 列预计算一次 G64/G128 activation group sum；所有输出行复用。候选实现：

1. Q4/Q5/Q6/Q8 code dot；
2. shared activation-group sums；
3. scale dot + bias correction；
4. FP32 accumulation；
5. fused output epilogue。

该优化只改变求值组织，不改变 oQ4e 重建语义。

### 6.3 `T=2..8`

候选族：

- affine SIMD row-split：一份 code/scale/bias 跨 T 复用；
- Q5/Q6 split-K：补足高位plane解码路径的并行度；
- activation-cached/row-block：减少重复 activation load；
- custom affine decode → cooperative tensor → TensorOps。

权重 tile 只能按 K-stage 读取和解码一次，禁止每个 T 列重新读取完整 code/metadata。

### 6.4 `T=9..128` 与 bulk

oQ Q4/Q5/Q6/Q8没有完全匹配的Metal原生affine tensor，尤其Q5/Q6无完全同构的5/6-bit合同，且都有BF16
bias。因此大 T 的主要候选为：

```text
custom affine decode into cooperative tensor
→ matmul2d
→ fused epilogue
```

若直接 SIMD broad GEMM 在某些 small shape 更快则保留 interval route；不强制所有大 T
都进入 TensorOps。

### 6.5 oQ4e 可复用 NInfer groupwise-int 的部分

- offline kernel-native planes；
- K-padding、row views 与 fused parent；
- T=1 QMV / small-T / broad route 分层；
- Q5 high-bit、Q6 high-2bit与Q8 byte-code分plane；
- row-split、split-K、row-block 候选结构；
- 权重解码跨 T 重用；
- split-output 与 fused epilogue；
- per-shape static dispatch table；
- numerical oracle 与 production kernel 分离。

### 6.6 oQ4e 不能照搬的部分

- signed symmetric decode；
- scale-only weight contract；
- FP16 scale 假设；
- CUDA `cp.async`/warp/MMA 指令；
- CUDA tile、occupancy 和 cache policy；
- NInfer 按角色固定的 q4/q5 分配（我们必须跟随已固定checkpoint的Q4/Q5/Q6/Q8分配）；
- NVFP4 W4A4 activation policy。

---

## 7. Apple MXFP4 的详细优化方式

### 7.1 数值与 artifact 布局

R1 定义：

```text
mxfp4_e2m1_g32_e8m0_v1
```

MLX 原生逻辑布局为：

```text
weight: uint32 [N, K/8]   # 8 × E2M1 per word
scales: uint8  [N, K/32]  # E8M0 per 32 weights
```

converter 为 mlx-community 与 Red Hat 各实现 source adapter，最终都输出同一 data/scale
plane contract。若源 packing/stride 已满足 Metal，则零拷贝绑定；否则只离线 repack 一次。

### 7.2 W4A16

同一份 MXFP4 weight 接收 BF16/FP16 activation。主要用途：

- `T=1` 专用 QMV；
- small-T 低开销 fallback；
- W4A4 数值与性能基线；
- 动态 A4 量化成本高于收益的 shape。

`T=1` 自定义 kernel 顺序读取 E2M1 code 与 E8M0 scale，执行向量 decode/dot；无 bias
correction。small-T 必须像 groupwise-int 一样跨列复用 code/scale。

对支持混合 operand 的 macOS 27 TensorOps，同时保留 direct quantized-weight + A16 candidate；
是否晋级由目标机实测决定。

### 7.3 W4A4

W4A4 是 Profile B 的速度主线：

```text
BF16 normalized activation
→ dynamic MXFP4 quantization
→ MXFP4 activation + MXFP4 weight
→ direct multi-plane TensorOps/NAX
```

关键要求：

1. RMSNorm 与 A4 quantize 尽量融合；
2. 同一输入只量化一次；
3. attention q/k/v/gate 或 MLP gate/up 共享 A4 activation；
4. data/scale planes 驻留 workspace，不按 Linear 重建；
5. 当前SDK 26.2直接消费code/scale plane并在threadgroup内展开当前tile，不写全局BF16镜像；
6. 当前TensorOps负责展开后tile的NAX映射；若目标Xcode 27公开FP4/MX operand type，
   H0再把tile内部替换为原生量化operand绑定；
7. custom cooperative decode 只作为 small-T、特殊融合或 operand 不兼容时的后备。

B1.3已冻结这一资源合同：`AllowA4`只表示资格，不强制路由；A4-only
lease只物化E2M1 code/E8M0 scale，不额外写A16镜像；同一epoch/key的
q/k/v、gate/up和MoE expert复用同一份plane。实际route由B2选择。

### 7.4 按 T 路由

| T | 首选候选 | 对照/后备 |
|---|---|---|
| 1 | W4A16 SIMD QMV | W4A4 实验候选 |
| 2..8 | W4A16 small-T、W4A4 TensorOps 竞选 | cooperative decode |
| 9..128 | W4A4 TensorOps | W4A16 TensorOps/SIMD |
| 512/1024 | W4A4 TensorOps/NAX | W4A16 fallback |
| 其它 | broad/tail-safe W4A4 | W4A16 fallback |

速度优先不代表强制 W4A4。每个真实 shape/T 选择实测更快者；W4A16 仍是 T=1、
small-T 和故障 fallback 的必要路线。

### 7.5 四份 MXFP4 输入源的边界

OsaurusAI的Qwen3.6 Dense/MoE是对应产品架构的MXFP4速度候选，其中MoE的
router/shared gate保留affine Q8。`mlx-community` Qwen3.8更紧凑，适合测量
memory-bound decode与纯MXFP4上限。Red Hat使用AWQ smoothing + GPTQ，
保留更多BF16层并带W4A4/KV FP8源配置，适合质量对照。

backend 导入 Red Hat 时只采纳持久权重及受保护层；activation 与 KV policy 由 Program
独立决定。为公平比较，可用同一权重分别运行 A16/A4，并把 KV 固定为 BF16。不得把
Red Hat 原配置中的 KV FP8 收益或损失记到 MXFP4 weight 上。

---

## 8. 不受量化格式影响、必须保持不变的部分

以下内容不因 Profile A/B、groupwise-int/NVFP4 参考关系而改变：

### 8.1 产品与仓库边界

- oMLX 只负责 tokenizer、协议、路由、fallback 和控制面；
- `m5rt` 负责 artifact、weights、kernel、Program、workspace 和 state；
- NInfer 只作设计参考，不成为构建或运行依赖；
- backend固定Qwen3.5架构族的Qwen3.6/3.8‑27B Dense与Qwen3.6‑35B‑A3B
  MoE产品矩阵，不扩展成通用模型编译器；Qwen3.6增补详见
  `qwen36-format-extension-v1.md`。

### 8.2 资源所有权

- converter 离线决定物理布局；
- artifact 带 format/layout/version/checksum；
- loader 每个 parent 只 materialize 一次；
- token loop 不编译 pipeline、不查找 artifact id、不 repack 权重；
- Program 持有 immutable weights、workspace、KV/GDN state 和 dispatch plan。

### 8.3 T 域与 dispatch 原则

- 完整支持 `T=1..128`；
- 单独覆盖 bulk anchors `T=512/1024`；
- 核心点 `1/4/8` 分别验收；
- exact-T 只保留核心点或显著 seam，其余用 capacity interval；
- M4 只产生 correctness/候选数据；M5 Ultra 才生成生产 route table。

### 8.4 DFlash2 语义

- R1启动期配置`K=1..7`；`K=7` proposal + anchor等于核心target verify点`M/T=8`；
- 每个配置K覆盖尾部`T=1..K+1`；
- 每个配置K覆盖accepted length `N=0..K`；
- 只有一次masked-block draft、动态卷积和条件selector进入真实Program后，G3才能关闭；
- ReplaySSM、commit/abort/cancel、bonus/correction/EOS/stop/budget 语义不变；
- activation/weight 格式不得改变事务边界。

### 8.5 正确性与证据

- 每种持久格式有独立 reference decoder 与 golden vectors；
- kernel 对 FP32/reference、MLX 和真实 layer slice 分层对齐；
- fused op 必须保留 unfused oracle；
- Profile A/B 的质量、性能和 DFlash2 acceptance 分开报告；
- prefill、decode、verify、proposal、cold/warm 和 cache state 分开；
- 估算、M4 测量和 M5 Ultra 测量严格分级。

### 8.6 Gate

- G0：artifact/loader contract；
- G1：kernel 全 T 正确性；
- G2：layer 与 AR Program；
- G3：DFlash2 Program；
- G4：M5Engine binding；
- H0：目标机能力确认；
- H1：route/tile 晋级；
- H2：整机性能、长稳与发布。

目标机只决定候选晋级，不决定是否开始写代码。

---

## 9. Kernel catalog 合同

`KernelKey` 至少包含：

```text
format_id
activation_dtype / activation_policy
output_dtype / accumulator_dtype
N, K
T interval or exact T
fused operation id
device capability class
full/tail alignment class
```

R1 kernel family：

```text
oQ4e
├── oq_q{4,5,6,8}_qmv_a16
├── oq_q{4,5,6,8}_small_t_a16
├── oq_q{5,6}_small_t_a16_splitk
├── oq_q{4,5,6,8}_cooperative_tensorops
└── oq_q{4,5,6,8}_broad_tail_safe

MXFP4
├── mxfp4_qmv_a16
├── mxfp4_small_t_a16
├── mxfp4_tensorops_a16
├── rmsnorm_quantize_mxfp4
├── mxfp4_tensorops_a4
└── mxfp4_fused_projection_a4
```

MXFP8 family 不在 R1 编译清单中。

B2.1已将上述维度实现为可封存的`KernelKey/KernelCandidate/DispatchTable`，
并通过executor安装白名单区分planned与installed。B2.1结项时850个真实shape候选合同
全部是planned，没有一个被冒充为可执行kernel；详见`dispatch-contract-v1.md`。

B2.2已将Dense `T=1` oQ Q4/Q5/Q6/Q8 A16与MXFP4 W4A16 QMV晋级为installed；
affine路径使用一次activation group-sum prepass消除输出行间重复`sum(x)`，
MXFP4使用E2M1查表与E8M0位解码。它们尚未经M5 Ultra H1晋级为qualified。

B2.3已将Dense affine oQ `T=2..12`容量型row-split、Q5/Q6 split-K×4与
cooperative row-block晋级为installed。每个code/scale/bias在一次Linear内只解码/
读取一次并跨T列复用；38个真实N/K问题的418个problem-token组合通过artifact与
stock MLX双oracle。split-K静态优先级和全部tile仍只是M4正确性候选，尚未经M5 Ultra
H1晋级为qualified。MXFP4 small-T继续由B2.4独立实现，不复用affine解码公式。

B2.4已将Dense MXFP4 `T=2..12` W4A16/W4A4 SIMD executor和共享A4 quantizer
晋级为installed，并生成直接消费A4/W4多plane的TensorOps metallib。当前SDK 26.2的
`matmul2d`没有公开FP4 operand type，因此TensorOps kernel只在threadgroup内展开当前tile
后计算，不创建全局BF16镜像；这两个目标路径保持planned，等macOS 27目标runtime通过
capability Gate后才installed。16个真实N/K问题的352个profile组合已通过双oracle。

B2.5已用`Ttile=16`二维token-block grid补齐Dense `T=13..128,512,1024`。9种合成
格式的1,062个plan覆盖每个整数T；55个真实N/K问题生成8,378个精确full-shape key，
十个接缝/anchor的710个真实权重profile通过双oracle。classic broad保持installed
correctness fallback，TensorOps保持目标runtime候选，M4结果不用于H1静态表。

B2.6已完成256-expert/top-8 router与expert-major catalog。grouped kernel用一个dispatch
覆盖全部`T×8` assignment，fallback用单simdgroup assignment直接寻址expert而不循环
launch 256次；affine Q4/Q5/Q6/Q8 G64/G128与MXFP4 A16/A4覆盖全部T。243个真实
expert parent生成94,380个key且uncovered=0，B2至此通过G1开发正确性Gate。

---

## 10. 实施顺序

### B0：格式与 artifact

1. container 与 object directory；
2. oQ Q4/Q5/Q6/Q8、G64/G128与Dense/expert-major numeric/layout contracts；
3. MXFP4 numeric/layout contract；
4. oQ4e、mlx-community MXFP4、Red Hat MXFP4 source adapters；
5. inspector、round-trip decoder、golden vectors；
6. 七份真实Dense/MoE manifest、hash与speed-first profile lock。

### B1：runtime

1. multi-plane weight views；
2. MTLTensor/tensor-inline binding；
3. immutable weight/workspace/state arenas；
4. pipeline cache 与 capability adapter；
5. activation A4 workspace 与 per-use policy；
6. resource/dispatch accounting。

### B2：Linear catalog

1. oQ4e T=1；
2. MXFP4 W4A16 T=1；
3. oQ4e T=2..12；
4. MXFP4 W4A16/W4A4 T=2..12；
5. 两格式 T=13..128；
6. MXFP4 W4A4 512/1024；
7. broad/tail routes 与完整 sweep。

B3 之后沿用上位计划，不因格式扩展改变 Program、DFlash2、binding 和发布顺序。

---

## 11. 验收矩阵

### 11.1 格式

- oQ q4/q5/q6/q8 code/scale/bias bit-exact round trip；
- q5 base/high plane 重组 bit-exact；
- q6 base/high-2bit plane重组bit-exact，q8 byte-code不得展开为常驿BF16镜像；
- Qwen3.6-35B-A3B oQ六种实际format和123个expert-major parent固定回归；
- MXFP4 E2M1/E8M0 bit-exact round trip；
- 四份MXFP4 source经两类adapter形态输出相同layout contract；
- padding、alignment、overflow、corruption fail closed。

### 11.2 kernel

- 每个真实 N/K 家族；
- `T=1..128,512,1024`；
- A16/A4 分开；
- full/tail、BF16/FP16 input、FP32 accumulation；
- fused/unfused parity；
- 无 token-column 级权重重复读取设计。

### 11.3 端到端

- oQ4e 与 stock MLX；
- mlx-community MXFP4 与 stock MLX；
- Red Hat MXFP4 与其 source/runtime reference；
- AR 与 DFlash2 token/state parity；
- acceptance、质量、内存和吞吐分开；
- 同一 MXFP4 weight 的 W4A16/W4A4 A/B。

---

## 12. 一句话结论

oQ mixed在架构上参考 NInfer groupwise-int：复用 plane、QMV、row-split、split-K、
权重跨 T 重用、融合和静态 dispatch，但重写 affine code/scale/bias 与 Metal 执行。

MXFP4 在架构上参考 NInfer NVFP4：复用 activation policy、W4A4、activation 复用和
Tensor route，但采用 Apple E2M1+E8M0/G32 多平面 TensorOps，不复制 NVFP4 数值和
Blackwell kernel。

artifact ownership、Program/state、T 域、DFlash2 事务、测试分层和 M5 Ultra 性能 Gate
不受格式差异影响，继续按 backend-first 主线执行。

---

## 13. 证据与源码依据

- 目标 oQ4e 张量分布：`../../docs/quantization.md`
- oQe affine/imatrix 量化：`../../omlx/omlx/oq.py`
- NInfer 数值格式：`../../ninfer/docs/maintainer/tensor-formats.md`
- NInfer 物理布局：`../../ninfer/docs/maintainer/storage-layouts.md`
- NInfer Qwen3.8 recipe：`../../ninfer/tools/convert/official_recipes.py`
- NInfer Linear T 域：`../../ninfer/docs/maintainer/linear-tuning.md`
- NInfer Q4 row-split：`../../ninfer/src/ops/linear/q4/q4_rowsplit_gemm_simt.cuh`
- NInfer Q4/Q5 small-T dispatch：`../../ninfer/src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu`
- Apple WWDC26 TensorOps：<https://developer.apple.com/videos/play/wwdc2026/330/>
- Apple MPP Programming Guide：<https://developer.apple.com/download/files/Metal-Performance-Primitives-Programming-Guide.pdf>
- mlx-community MXFP4：<https://huggingface.co/mlx-community/Qwen3.8-27B-mxfp4/tree/97ab0819817ab1c61d7d39f9169fc71999915641>
- Red Hat MXFP4：<https://huggingface.co/RedHatAI/Qwen3.8-27B-MXFP4/tree/651eed99386dae5645cd4949024e8ebd163a533b>
