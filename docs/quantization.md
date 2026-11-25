# Qwen3.8-27B oQ4e 格式剖析（P-4）

> 日期：2026-09-19 · 证据状态：已完成并保留
> 当前对应：实施计划 v3.2 §2.1、D1、M0、M1；本文只负责 checkpoint 结构、量化统计与估算，不承担实测性能结论
> 数据：`docs/p4/*.headers.json`（safetensors 头）、`docs/p4/analysis.json`（统计结果）
> 脚本：`tools/p4_read_headers.py`（读头）、`tools/p4_analyze.py`（统计 + roofline）、`tools/p4_verify_sha256.py`（完整性）
> 结论只来自文件元数据和形状推算；**质量评测与实测性能要到 M5 Ultra 到货后做**。

## 1. 使用的 checkpoint（已下载到 `models/`，版本固定）

| 用途 | 仓库 | 固定版本 | 大小 | 完整性 |
|---|---|---|---:|---|
| **基线（目标模型）** | `Jundot/Qwen3.8-27B-oQ4e-mtp` | `04dc5509edd8670fc78cc8c2f74bf9b77b1f2acc` | 16.97 GB | sha256 全部通过 |
| DFlash2 草稿模型 | `z-lab/Qwen3.8-27B-DFlash2` | `50307d4c4cde6860d4eee73e2547cd786fe8e8a4` | 3.85 GB | sha256 全部通过 |
| 对照（统一 4-bit，可选实验用） | `mlx-community/Qwen3.8-27B-4bit` | `10c35caafbb80f7dc6a7a432cdd11af10a6d4818` | 16.05 GB | sha256 全部通过 |

- 基线来源：Jundot 是 oMLX 作者（`github.com/jundot/omlx`），用 oMLX v0.6.1 量化，模型没有经过微调。Hugging Face 上没有不带 MTP 的官方 oQ4e；`-mtp` 版本只是多了 MTP 头（0.31 GB），我们的 runtime 忽略它，到货后跑 stock MTP 对照组时正好可以用上。
- 草稿模型：模型卡写明是 `incoai/Qwen3.8-27B-DFlash2` 的镜像，Apache-2.0，对应 `Qwen/Qwen3.8-27B`。决策者已确定先用这份现成的草稿模型，不自己训练（2026-09-19）。
- 重新下载：`python3 models/download.py`（只用 Python 标准库 + curl，可断点续传，远端版本对不上就拒绝下载）。

## 2. 目标模型结构（从 config 读出，与计划书 §2.1 一致）

64 层，每 4 层一个 full attention（第 3, 7, …, 63 层，共 16 层），其余 48 层是 GDN；hidden 5120；attention 24 / 4 heads，head_dim 256；GDN 48 个 value head；MLP intermediate 17408；vocab 248320；embedding 与 lm_head 不共享。`model_type = qwen3_5`，带视觉模块（0.92 GB，BF16），带 1 层 MTP。

## 3. 量化方式

- **全部是 affine、group size 64**（默认 `bits=4, group_size=64, mode=affine`，与 `omlx/omlx/oq.py:4545-4552` 一致）。
- 每个张量的位数用两种方法独立判断，结果**全部一致**：(a) 从形状推算：packed uint32 宽度 × 32 ÷（scales 宽度 × 64）；(b) 读 `config.json` 里的逐张量设置。
- **只有 4-bit 和 5-bit 两种**。166 个张量被提到 5-bit，没有 6-bit 或 8-bit。
- **lm_head 是 4-bit**（`[248320, 640]` uint32 = 5120 × 4 / 32），embedding 也是 4-bit。这与 oQ 文档中"lm_head 8-bit"的描述不同，以本 checkpoint 的实际数据为准。
- 未量化的张量（norm、`A_log`、`dt_bias`、conv1d、q/k_norm 等）共 353 个，只有 5 MB，全部是 BF16。

## 4. 解码时每个 token 要读的数据

计入：64 层全部权重 + lm_head + 最后的 norm。不计：embedding（每个 token 只读 1 行）、视觉模块、MTP 头。

| 位数 | 张量数 | 字节 | 占比 |
|---|---:|---:|---:|
| 4-bit | 331 | 11.700 GB | **77.9%** |
| 5-bit | 166 | 3.316 GB | **22.1%** |
| BF16（未量化） | 353 | 0.005 GB | 0.0% |
| **合计** | | **15.020 GB** | 等效 4.69 bit / 权重（25.62 B 个量化参数） |

对照：统一 4-bit 版本合计 14.418 GB（4.50 bit）。**oQ4e 多读 0.60 GB，约 +4.2%。**

## 5. 5-bit 张量的分布

| 张量（形状 out × in） | 个数 | 字节 | 所在层 |
|---|---:|---:|---|
| GDN 层 `mlp.down_proj`（5120 × 17408） | 21 | 1.654 GB（与下一行合计 27 个） | 第 0–26 层中的 GDN 层 |
| attention 层 `mlp.down_proj`（5120 × 17408） | 6 | 同上 | 3, 7, 11, 15, 19, 23 |
| GDN `linear_attn.out_proj`（5120 × 6144） | 48 | 1.038 GB | **全部 48 个 GDN 层** |
| GDN `linear_attn.in_proj_z`（6144 × 5120） | 21 | 0.454 GB | 第 0–26 层中的 GDN 层 |
| attention `self_attn.o_proj`（5120 × 6144） | 7 | 0.151 GB | 3, 7, 11, 15, 19, 23, 27 |
| attention `self_attn.k_proj`（1024 × 5120） | 2 | 0.007 GB | 27, 35 |
| GDN `in_proj_a` / `in_proj_b`（48 × 5120） | 31 / 30 | 0.010 GB | 第 0–40 层中的 GDN 层 |

规律：**所有 GDN 输出投影都是 5-bit，网络前半部分（约第 0–27 层）的 down / z / o 投影也是 5-bit**。字节量最大的 MLP `gate_proj` / `up_proj`（17408 × 5120）和 GDN `in_proj_qkv`（10240 × 5120）全部是 4-bit。

## 6. 对 kernel 工作的影响

1. **位数集合 = {4, 5}**。qmm kernel 仍以 bits 为模板参数，但调优和测试矩阵只覆盖 4 和 5；6 / 8 保留实现路径，不投入优化。
2. **4-bit 决定大头（约 78% 字节）**，重点形状：17408 × 5120（gate / up）、10240 × 5120（GDN qkv）、12288 × 5120（attention q）、5120 × 17408（down 的 4-bit 部分）、248320 × 5120（lm_head）。
3. **5-bit 只落在 3 个主要形状**：5120 × 17408（down）、5120 × 6144（GDN out_proj / attention o_proj）、6144 × 5120（GDN z），可以针对性特化。
4. 48 × 5120 的 `in_proj_a/b` 字节极小，但每层都要单独调度一次，适合作为后续 profiling 触发的融合候选；是否实施由 v3.2 ADR-07 决定。

## 7. M=1 decode 性能上限（按实际字节重算）

假设：带宽 1.2 TB/s（M0 实测后替换）；GDN 状态每 token 读写各一次，2 × 144 MiB；KV 每 token 64 KiB。

| 上下文 | 总读取量 | KV 占比 | 理论上限 | 75%（达标） | 85%（优秀） |
|---:|---:|---:|---:|---:|---:|
| 512 | 15.36 GB | 0.2% | 78.1 | 58.6 | 66.4 |
| **2K** | **15.46 GB** | 0.9% | **77.6** | **58.2** | **66.0** |
| 8K | 15.86 GB | 3.4% | 75.7 | 56.7 | 64.3 |
| 32K | 17.47 GB | 12.3% | 68.7 | 51.5 | 58.4 |
| 64K | 19.62 GB | 21.9% | 61.2 | 45.9 | 52.0 |
| 128K | 23.91 GB | 35.9% | 50.2 | 37.6 | 42.7 |
| 256K | 32.50 GB | 52.9% | 36.9 | 27.7 | 31.4 |

（单位 tok/s）统一 4-bit 版本在 2K 时上限为 80.8 tok/s。**选择 oQ4e 的速度代价约 4%。**

## 8. DFlash2 草稿模型

- 结构：5 层，全部 sliding attention（窗口 2048），hidden 5120，32 / 8 heads，head_dim 128，MLP 17408；`block_size = 8`（K = 7 + 1 个 anchor）；从目标模型第 `[5, 19, 33, 47, 61]` 层取特征（`fc`：25600 → 5120）；每层 attention 和 MLP 两侧都有两抽头动态卷积；候选选择器为 top-16 / rank 256（两个 248320 × 256 的 codebook）。
- **全部是 BF16，没有量化。**
- **没有自己的输出层**，候选词由目标模型的 lm_head（proposal head）给出（NInfer `dflash.md:124-126, 175-176`），codebook 只按 top-16 候选取行。
- 每轮要读：5 层 3.33 GB + `fc` 0.26 GB + 借用的目标 lm_head 0.715 GB ≈ **4.3 GB，在 1.2 TB/s 下约 3.6 ms**。一轮约 24 ms，其中草稿约占 15%。
- 如果把草稿模型的 5 层和 `fc` 量化成 4-bit（oMLX 已支持 `dflash_draft_quant_*`），每轮约 1.7 GB、约 1.4 ms，但接受率可能下降。**取舍在 M5 实测后决定**（实施计划 v3.2 ADR-07 / M4）。

## 9. 仍待目标机完成

- 目标机的 stock 质量记录与最终代表性质量回归（实施计划 v3.2 M7）。
- 4-bit / 5-bit 各主要形状的 stock MLX qmm 微基准（M=1..8，由 v3.2 M0/M1 完成）。
- 可选：oQ4e 与统一 4-bit 的一天对照实验（速度 + 质量）。
