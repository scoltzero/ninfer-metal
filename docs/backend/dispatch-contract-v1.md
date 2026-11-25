# B2.1 Kernel dispatch公共合同v1

> 状态：完成并冻结
> 范围：R1 Dense/MoE Linear候选登记、静态路由与fallback；不包含production kernel
> 实现：`m5rt/include/omlx_m5/dispatch.h`、`m5rt/src/dispatch.cpp`

## 1. KernelKey

`KernelKey`描述一次已经由Program解析完成的物理调用，不在token loop查artifact名称：

```text
format_id + layout_id
activation_policy + preferred_activation
input/output/accumulator dtype
geometry(dense/moe_expert/moe_router)
N + K + T + experts + top_k
fused_op
alignment(full/tail)
```

R1接受：

- oQ affine Q4/Q5/Q6/Q8、G64/G128；
- MXFP4 E2M1/G32/E8M0；
- 受保护层的direct BF16/F16 A16 fallback；
- Dense、256-expert/top-8 MoE expert与router几何；
- BF16/FP16输入输出与FP32累加。

A4 key必须同时满足`AllowA4 + MXFP4`。oQ或direct格式携带A4、expert geometry使用
Dense layout、Dense key携带expert元数据、K/N/T为0等情况均fail closed。

## 2. KernelCandidate

Candidate在固定selector上声明：

- executor key和静态priority；
- QMV、row-split、split-K、cooperative、TensorOps、grouped-MoE、
  per-expert fallback或tail-safe family；
- format/layout/profile/dtype/geometry/N/K；
- `[min_T,max_T]`或`min_T=max_T` exact-T；
- fused op、alignment class与required Metal route；
- 是否tail-safe。

Candidate不带运行时benchmark分数。priority来自版本化静态表；H1实测后可以生成M5 Ultra
qualified顺序，但token loop不调优、不枚举、不编译。

## 3. planned、installed与qualified

三种状态必须分开：

| 状态 | 含义 | 能否执行 |
|---|---|---|
| planned | B2.1只有合同，executor尚不存在 | 否 |
| installed | executor/pipeline已构建并通过正确性Gate | 是，但不代表M5默认快路径 |
| qualified | 目标机H1性能与稳定性晋级 | 是，可进入生产静态表 |

`DispatchTable::seal(installed_executor_keys)`是唯一安装边界。没有出现在installed集合中的
candidate即使selector完全匹配也不可执行。planned-only table的`prepare()`必须失败。

## 4. fallback顺序

一次`prepare()`只在load/warmup期发生，返回可缓存的immutable `KernelDispatchPlan`：

1. 选择同activation profile、shape/T/alignment/fused-op匹配且capability允许的最低priority
   installed candidate；
2. `AllowA4`请求如果没有可执行A4，尝试同key的A16 candidate；
3. tail key只能使用tail-safe candidate；full key可在无专用route时退到tail-safe；
4. native TensorOps不可用时只允许显式classic MSL或MLX bridge fallback；
5. 没有installed候选时fail closed，不把planned条目或不兼容route当成fallback。

相同最低priority的重叠installed candidate属于歧义配置，直接拒绝，不依赖注册顺序。

## 5. T域

合同覆盖：

| T | 主候选合同 | 必须保留的fallback |
|---|---|---|
| 1 | A16 QMV | tail-safe A16 |
| 2–8 | row-split/split-K；MXFP4 A4 TensorOps可选 | A4 SIMD、A16 + tail-safe |
| 9–12 | capacity small-T；MXFP4 A4 TensorOps可选 | A4 SIMD、A16 + tail-safe |
| 13–128 | cooperative/broad；MXFP4 A4 TensorOps可选 | A16 + tail-safe |
| 512/1024 | exact bulk anchor | A16 + tail-safe |
| 其它正整数 | 无预设快路径 | 通用tail-safe A16 |

MoE主候选使用grouped-MoE，tail fallback使用per-expert路径。B2.1只冻结这一关系；
grouped与per-expert kernel分别在B2.6实现和验收。

## 6. 外部静态表schema

JSON Schema与合同示例位于：

- `bench/schema/kernel-route-table-v1.schema.json`；
- `bench/schema/kernel-route-table-v1.example.json`。

每个entry保存selector、candidate和`planned/installed/qualified`状态。示例固定为
`contract_only + planned`，不能被误读成M5 Ultra测量或已安装kernel。

## 7. B2.1与后续边界

B2.1完成的是类型、校验、封存、预解析、capability filtering与fallback；它本身
不声称production kernel存在。后续B2.2已实现Dense `T=1` oQ/MXFP4 executor，
B2.3已实现Dense affine oQ `T=2..12` row-split/split-K/cooperative executor，
B2.4已实现Dense MXFP4 `T=2..12` W4A16/W4A4 SIMD executor，并把当前可执行key
加入installed集合；B2.5已实现Dense `T=13..128,512,1024` oQ/MXFP4 broad/bulk
executor并完成真实full-shape key sweep；B2.6已实现top-8 router、grouped-MoE和
per-expert fallback并把对应key加入installed集合。B2.4/B2.5 TensorOps仍按capability
保持planned。B2已通过G1；M4候选顺序不等于M5 Ultra最优顺序。
