# `.miv`（兼容历史 `.omlxm5`）artifact format v1

> 状态：B0冻结；little-endian；schema `omlx-m5-artifact-v1`（magic `OMLXM5A1`）  
> 生产实现：`miv/tools/m5_artifact.py`、`miv/src/artifact.cpp`  
> R1范围：Qwen3.6/3.8 Dense与Qwen3.6 MoE的oQ affine Q4/Q5/Q6/Q8、MXFP4；MXFP8不在v1中

## 1. 设计目标

`.miv`（向后兼容历史后缀 `.omlxm5`）是backend自有的只读权重容器。safetensors和compressed-tensors只在
离线导入时使用；运行时mmap容器后直接得到各plane的非拥有视图，不解包、不转置、
不按请求重排。格式把数值语义、物理布局、来源版本和activation policy一起冻结，
防止converter和kernel各自猜测布局。

v1有以下量化对象族：

| format id | 数值合同 | plane顺序 | layout id |
|---|---|---|---|
| `oq_affine_q3_g64_bf16_v1` / `oq_affine_q3_g128_bf16_v1` | unsigned Q3 (`24 B / 64w`)，G64/G128，BF16 scale+bias | `low_2bit, high_bit, scale, bias` | `oq_rowsplit_k128_v1` |
| `oq_affine_q4_g64_bf16_v1` / `oq_affine_q4_g128_bf16_v1` | unsigned Q4，G64/G128，BF16 scale+bias | `base_code, scale, bias` | `oq_rowsplit_k128_v1` |
| `oq_affine_q5_g64_bf16_v1` | unsigned Q5，G64，BF16 scale+bias | `base_code, high_bit, scale, bias` | `oq_rowsplit_k128_v1` |
| `oq_affine_q5_g128_bf16_v1` | unsigned Q5，G128，BF16 scale+bias | `base_code, high_bit, scale, bias` | `oq_rowsplit_k128_v1` |
| `oq_affine_q6_g64_bf16_v1` / `oq_affine_q6_g128_bf16_v1` | unsigned Q6，G64/G128 | `base_code, high_2bit, scale, bias` | `oq_rowsplit_k128_v1` |
| `oq_affine_q8_g64_bf16_v1` / `oq_affine_q8_g128_bf16_v1` | unsigned Q8，G64/G128 | `code, scale, bias` | `oq_rowsplit_k128_v1` |
| `mxfp4_e2m1_g32_e8m0_v1` | E2M1，G32，E8M0 scale，无bias | `data, scale` | `mx_rowsplit_k128_v1` |

rank-3 MoE expert parent使用相同format id，但layout改为
`oq_expert_rowsplit_k128_v1`或`mx_expert_rowsplit_k128_v1`，shape保留为
`[experts,N,K]`；物理plane按expert-major顺序把前两维展平成连续row。

未量化tensor使用`direct_<dtype>_v1`和`contiguous_le_v1`，原始字节保持不变。

## 2. 固定header

header固定64字节，所有整数为little-endian：

| offset | 类型 | 字段 | v1约束 |
|---:|---|---|---|
| 0 | `char[8]` | magic | `OMLXM5A1` |
| 8 | `u32` | version | `1` |
| 12 | `u32` | header bytes | `64` |
| 16 | `u64` | manifest offset | `64` |
| 24 | `u64` | manifest bytes | 大于0 |
| 32 | `u64` | payload offset | 4096字节对齐 |
| 40 | `u64` | payload bytes | 文件必须恰好在payload末尾结束 |
| 48 | `u32` | object count | 等于manifest `objects`长度 |
| 52 | `u32` | manifest CRC32 | zlib/IEEE CRC32 |
| 56 | `u32` | header CRC32 | 对header `[0,56)`计算 |
| 60 | `u32` | flags | v1固定为0 |

manifest是UTF-8 canonical JSON：key排序、无多余空白、`,`/`:`紧凑分隔。manifest到
payload之间必须用零填充。reader拒绝未知版本、非canonical manifest、非零padding、
越界/溢出、尾随字节和CRC错误。

## 3. object directory与校验

对象按name字典序排列，`id`必须为从0开始的连续整数。每个object起点按256字节
对齐；每个plane起点也按256字节对齐。offset均相对payload起点。object记录至少包含：

```text
id, name, format, layout, shape, padded_k,
object_offset, object_bytes, object_sha256,
planes[], source_tensors[]
```

plane记录包含`kind, offset, bytes, alignment, crc32, sha256`。所有间隙必须为零；
`object_sha256`覆盖从object起点到最后一个plane末尾，因而也覆盖plane间padding。
conversion report另存整个artifact的SHA-256。Python和C++ reader在
`verify_payload=true`时同时验证逐plane CRC32/SHA-256和逐object SHA-256。

`bindings[]`为每个量化对象保存`object_id/name/format/use/activation_policy/
out_features/in_features`。oQ4e固定`A16Only`，MXFP4固定`AllowA4`。`use`区分
embedding、lm-head、MLP、attention和GDN投影；kernel选择不需要解析tensor名。

## 4. oQ affine物理布局

设逻辑shape为`[...,N,K]`，`rows=product(shape[:-1])`，
`Kp=align_up(K,128)`。Dense的`...`为空；MoE为expert轴。

### 4.1 Q4

- `base_code`：每行`Kp/2`字节；偶数k在低nibble，奇数k在高nibble；padding code为0。
- `scale`、`bias`：每行各`Kp/64`个little-endian BF16。
- 解码：`w[n,k] = bf16(scale[n,k/64]) * q[n,k] + bf16(bias[n,k/64])`。

### 4.2 Q5

源MLX Q5是连续little-endian 5-bit bitstream。converter无损拆成：

- `base_code`：与Q4相同，保存code低4位；
- `high_bit`：每行`Kp/8`字节，bit `k%8`保存code第5位；
- `scale`、`bias`：与Q4相同。

重组公式为：

```text
q = nibble(base_code, k) | (bit(high_bit, k) << 4)
```

这使Q4/Q5共享低4位load/decode骨架，同时保持Q5 bit-exact，不把affine bias误写成
groupwise-int的零点或scale-only格式。

### 4.3 Q6

Q6继续复用Q4低位plane：

- `base_code`保存低4位；
- `high_2bit`每个byte保存连续4个code的高2位；
- scale/bias按format id指定的G64或G128解释。

重组公式：

```text
q = nibble(base_code, k) | (two_bits(high_2bit, k) << 4)
```

### 4.4 Q8

Q8的`code` plane为1 byte/weight，继续使用BF16 scale+bias及G64/G128。它不拆成
低/高plane，避免为已按byte排列的权重增加无意义重排。

## 5. MXFP4物理布局

设逻辑shape为`[...,N,K]`，`rows=product(shape[:-1])`，
`Kp=align_up(K,128)`：

- `data`：每行`Kp/2`字节；两个E2M1 code按低/高nibble排列；
- `scale`：每行`Kp/32`个E8M0 code；
- 解码：`w[n,k] = E2M1(data[n,k]) * E8M0(scale[n,k/32])`；
- E2M1正幅值表为`[0, .5, 1, 1.5, 2, 3, 4, 6]`，bit3为符号；
- E8M0 code `0`为`2^-127`，`1..254`为`2^(code-127)`，`255`是NaN并被reader拒绝解码。

`mlx-community`的`U32 weight + U8 scales`和Red Hat compressed-tensors的
`U8 weight_packed + U8 weight_scale`都规范化到完全相同的`data/scale`合同。二者
checkpoint recipe、保护层和质量结论仍分别保留，不因共用kernel而合并。

### 5.1 离线 NAX 预重排布局扩展（`v0.8.0-m5rt.3`）

为消除 Apple10 M5 NAX 加载期 GPU repack 产生的 ~14–17 GiB 冗余权重副本，`tools/m5_artifact.py`（`convert --nax-pretile` 与 `pretile` 子命令）及 `m5rt/src/artifact.cpp` 支持可选的 2D Dense 投影离线预重排布局（`embedding` 与 rank-3 MoE 权重保持原 `rowsplit` 布局）：

| format id | 预重排 layout id | plane 顺序 | 物理分块合同（`BN = 32 if N >= 65536 else 16`） |
|---|---|---|---|
| `mxfp4_e2m1_g32_e8m0_v1` | `mx_nax_tiled_bk128_v1` | `nax_data, nax_scale` | `N_pad = align_up(N, BN)`，`K % 128 == 0`；按 `[N_pad/BN, K/128]` tile 存储，`nax_data` 每 tile `BN*64` 字节，`nax_scale` 每 tile `BN*4` 字节 |
| `oq_affine_q3_g64_bf16_v1` / `oq_affine_q3_g128_bf16_v1` | `oq_nax_tiled_g64_v1` | `nax_codes, nax_high1, nax_sb` | `N_pad = align_up(N, BN)`，`K % group_size == 0`；`nax_codes` 按 `[N_pad/BN, K/64, BN, 16]`（`uint2b` 低 2 bit）存储，`nax_high1` 按 `[N_pad/BN, K/64, BN, 8]`（高 1 bit，共 `24 B/64w`）存储，`nax_sb` 按 `[K/64, N_pad, 2]` 交错 BF16 `(scale, bias)` 存储（`G128` 沿相邻两 `K/64` 块无损重复 `(scale, bias)`） |
| `oq_affine_q4_g64_bf16_v1` / `oq_affine_q4_g128_bf16_v1` | `oq_nax_tiled_g64_v1` | `nax_codes, nax_sb` | `N_pad = align_up(N, BN)`，`K % group_size == 0`；`nax_codes` 按 `[N_pad/BN, K/64, BN, 32]` 存储，`nax_sb` 按 `[K/64, N_pad, 2]` 交错 BF16 `(scale, bias)` 存储 |
| `oq_affine_q5_g{64,128}_bf16_v1` / `oq_affine_q6_g{64,128}_bf16_v1` | `oq_nax_tiled_g64_v1` | `nax_codes, nax_high2, nax_sb` | 同上，附加 `nax_high2` 按 `[N_pad/BN, K/64, BN, 16]` 2-bit 高位平面存储（Q5 低位保存第 5 bit；`G128` 沿相邻两 `K/64` 块重复 `nax_sb`） |

运行时 `Mxfp4NaxCache` 与 `AffineNaxCache` 检测到上述预重排 layout 后直接零拷贝绑定 `ModelPlaneBinding`（不分配额外 Metal 权重 buffer，亦不派发 repack kernel），`decode_artifact_row` 与 `read_artifact(verify_payload=True)` 保持逐元素 bit-exact 校验。

## 6. fused parent与view规则

v1只持久化拥有字节的parent object，manifest中的`views`固定为空数组。fused QKV、
gate-up等源tensor作为一个parent object保存；B1/B3可基于`object id + row_start +
row_count`建立非拥有row view。view必须满足：

1. `row_start + row_count <= N`；
2. 继承parent的format、layout、`padded_k`和全部plane row stride；
3. 不产生新payload，不改变object/plane hash；
4. scale/bias/high-bit的row起点分别按各自plane row stride计算，不能只偏移data plane；
5. token loop只能使用load/warmup期已经解析好的view，不能按tensor名查目录。

v1禁止converter自动猜测Q/K/V切分，因为不同模型的投影行数由模型配置决定。显式view
目录如果将来持久化，必须提升container/schema版本。

## 7. materialization合同

C++ reader提供`materialization_plan(alignment)`：每个plane对应一个upload span，记录
源文件绝对offset、arena目标offset和bytes，并分别统计payload bytes、实际upload bytes、
arena bytes与alignment padding。该计划只做plane级拷贝，不做解码、转置或repack。
Metal resource创建、一次上传和生命周期已由B1.2实现：超过单buffer上限时按
plane边界分段，parent/fused/expert view只保留segment+offset+bytes，Reader不进入
Model生命周期。证据见`b1-2-completion-report.md`。

## 8. 可复现与fail-closed

同一source repo、revision和converter版本必须生成逐字节相同的artifact。manifest保存
source revision、config/index SHA-256、tensor count/bytes；conversion report保存manifest、
object和完整artifact hash。下列情况必须拒绝：

- magic/version/flags不支持；
- manifest或payload checksum错误；
- object id/name重复、binding覆盖不完整；
- offset加法溢出、越界、重叠、错误alignment；
- 非零padding、尾随数据；
- plane类型、顺序或大小不符合format公式；
- Q5 high-bit或Q6 high-2bit plane缺失；
- MXFP4 E8M0 NaN scale在解码路径出现。

静态向量位于`m5rt/tests/data/b0_golden_vectors.json`，覆盖Q4/Q5/Q6/Q8与MXFP4；
合成round-trip与corruption矩阵
由`m5rt/tests/b0_artifact_roundtrip.py`执行，七份固定Dense/MoE源再由`validate_b0_artifact.py`
完成D1/stock MLX oracle和artifact Python/C++交叉解码。
