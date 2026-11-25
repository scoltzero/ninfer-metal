# Qwen3.6 Dense/MoE format extension v1

> 状态：完成并冻结
> 原则：复用`.omlxm5` v1容器；按固定checkpoint保留速度优先的原始压缩表示
> 范围：Qwen3.6‑27B Dense、Qwen3.6‑35B‑A3B MoE；oQ affine与MXFP4

## 1. 固定输入

| 模型 | Profile | Revision |
|---|---|---|
| `Jundot/Qwen3.6-27B-oQ4e-mtp` | oQ affine mixed | `2523e7a5702d38a1a319c50d193cbac20f9ecb78` |
| `OsaurusAI/Qwen3.6-27B-MXFP4` | uniform MXFP4 | `3258bea991decf16bf84fcd21cd73ae089bf3bbc` |
| `Jundot/Qwen3.6-35B-A3B-oQ4e-mtp` | oQ affine mixed MoE | `14c285372cbdb1777adea5bb49087ced0bffc0b5` |
| `OsaurusAI/Qwen3.6-35B-A3B-mxfp4` | MXFP4 MoE + affine Q8 gates | `5a6aaf3d0db3590f186d743b99c9b9d270ec7857` |

checkpoint名称不决定每个tensor的格式。converter逐tensor读取quantization表，并以实际
weight/scales/biases dtype与shape交叉验证。

## 2. 真实格式矩阵

### 2.1 Qwen3.6‑27B oQe

- 默认Q4/G64；
- 136个显式Q5/G64模块；
- 3个显式Q6/G64模块；
- 无G128或Q8覆盖。

Q6模块原样保存，不展开BF16：

```text
layers.27.mlp.down_proj
layers.31.mlp.down_proj
layers.27.self_attn.v_proj
```

### 2.2 Qwen3.6‑27B MXFP4

uniform MXFP4 E2M1/G32/E8M0，没有显式高bit覆盖。

### 2.3 Qwen3.6‑35B‑A3B oQe

默认Q4/G64，显式覆盖为：

| 格式 | 模块数 |
|---|---:|
| Q5/G128 | 106 |
| Q6/G64 | 19 |
| Q6/G128 | 1 |
| Q8/G64 | 66 |
| Q8/G128 | 123 |

这些格式全部保持压缩形式。B0不把Q6/Q8展开BF16，也不重新量化到Q4/Q5；后续B2分别
提供对应decode atom和kernel candidate，避免增加常驻字节与decode带宽。

### 2.4 Qwen3.6‑35B‑A3B MXFP4

- expert与普通主权重：MXFP4/G32；
- 40层router gate和shared-expert gate共80个模块：affine Q8/G64；
- Q8同样保留压缩表示，可复用oQ Q8 decode路径。

### 2.5 Qwen3.6‑35B‑A3B oQ速度优先合同

“支持oQ混合格式”不是把所有量化权重展开成统一BF16矩阵。R1把下列要求作为
不可回退的验收条件：

1. Q4/Q5/Q6/Q8与G64/G128按固定checkpoint逐tensor原样保留，不重新量化；
2. rank-3 expert保持expert-major压缩plane，B1.2不得建立解码后的常驻权重镜像；
3. 权重只在启动时一次materialize，token loop不做整权重copy、transpose或repack；
4. B2为四种bit width提供直接消费plane的decode atom，并按`use/shape/T`选择
   QMV、row-split、split-K、row-block或cooperative/TensorOps候选；
5. 低4位load/decode骨架允许共享，但Q5 high-bit、Q6 high-2bit、Q8 byte-code及
   G64/G128 metadata必须保留各自专用路径；
6. M5 Ultra只决定候选route/tile谁更快，不允许为了调参方便删除混合格式支持。

`validate_b0_artifact.py`对固定35B-A3B oQ revision锁定六种量化format的对象数、
123个Q4/G64 expert-major parent以及“无展开镜像/artifact plane无需repack”标志，
防止后续converter修改静默退回慢路径。B1.2已用35B-A3B真实MTP子集验证
Metal多段一次上传，重复解析view不再产生权重级copy/repack。

## 3. Expert-major layout

真实expert tensor不是rank-2：

```text
gate/up:   [256,  512, Kpacked]
down:      [256, 2048, Kpacked]
scale/bias [256, N, groups]
```

artifact保留逻辑shape `[experts,N,K]`，物理plane以expert-major顺序将前两维展平为
`experts*N`个row：

```text
expert 0 rows 0..N-1
expert 1 rows 0..N-1
...
expert 255 rows 0..N-1
```

使用独立layout id：

- `oq_expert_rowsplit_k128_v1`；
- `mx_expert_rowsplit_k128_v1`。

format id继续表达数值语义；layout id表达Dense与expert-major物理排列。运行时不转置、
不按expert拆成256份副本。

## 4. 新增affine plane

| bits | code plane | 重组 |
|---:|---|---|
| 4 | `base_code` | low/high nibble |
| 5 | `base_code + high_bit` | `low4 | high1<<4` |
| 6 | `base_code + high_2bit` | `low4 | high2<<4` |
| 8 | `code` | 1 byte/weight |

scale/bias保持little-endian BF16；group size由format id明确为G64或G128。Q6沿用Q4
base plane，使后续kernel可共享低4位load/decode骨架。

## 5. 与B1的关系

B1.1不需要修改设计：library/pipeline/command-plan registry不读取模型shape或量化格式，
新增格式只会新增pipeline与plan key。

B1.2已实现并验收：

- Dense `[N,K]`与MoE `[E,N,K]` parent；
- 一个parent的多plane view；
- expert、expert-row与fused-parent non-owning view；
- 超过单个Metal buffer上限时的segmented immutable weight arena；
- 只materialize启动时选中的Text/Vision/MTP/spec组件。

用人话说：B1.1是“调度台”，B1.2是“GPU仓库和货架”。B0把权重装成标准箱子；B1.2
在启动时把箱子一次搬进GPU可见内存，给每个parent、plane和expert贴好固定地址标签，另设
workspace工作台和每请求独立储物柜。token loop只拿地址，不再搬整箱权重。
完成证据见[`b1-2-completion-report.md`](../versions/v0.1-v5.1-pre-silicon/b0-b7-u0-u4-reports/b1-2-completion-report.md)。

## 6. Exit条件

- 四份固定Qwen3.6源模型各执行两次完整转换且SHA稳定；
- Q4/Q5/Q6/Q8、G64/G128、MXFP4 source/MLX/Python/C++ oracle一致；
- rank-3 expert parent与任意expert row bit-exact；
- 原有三份Qwen3.8 artifact继续被新reader接受；
- corruption、alignment、padding、overflow与checksum继续fail closed；
- 35B-A3B oQ speed-first profile锁定六种format及123个expert-major parent，且无
  expanded quantized mirror。

以上条件均已通过。四份真实artifact的对象数、format counts、materialization账本和
重复SHA-256见[`b0-completion-report.md`](b0-completion-report.md)。
