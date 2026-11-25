# 本项目的三个参考源

本项目的生产实现由 `m5rt` 负责。三个独立克隆用于比较源码、执行合同和实验方法；参考实现通过的测试不自动成为本项目的验收证据。

| 序号 | 本地目录 | 当前核查 HEAD（2026-11-21） | 主要学习与同构移植对象 |
|---|---|---|---|
| 1 | [`ninfer/`](../../ninfer/) | `ebdc67e5`（原快照 `9e163eee`） | runtime、scheduler、量化数据流、`run_packed_batch`、`FlashNax`、`FOSCAR` Affine4 KV、COW 前缀树与 `qda-search`（完整映射见 [`ninfer-m5-porting-architecture-v2.md`](ninfer-m5-porting-architecture-v2.md)） |
| 2 | [`omlx/`](../../omlx/) | `fbea0364`（`v0.8.0`） | Apple UMA 下的 scheduler、cache、`M5SidecarEngine` UDS Sidecar、`#5164` 解码保护与 MLX 对照 |
| 3 | [`TensorFold/`](../../TensorFold/) | `cb2ebf0540f42604e2759b2ddef497861e928248` | M5 tensor-unit lane kernel、row-exact forward、位置确定的采样、DFlash verify/状态提交、基于实测成本的 draft 分配 |

TensorFold 来源：[ashhart/TensorFold](https://github.com/ashhart/TensorFold)，源码版本为 0.6.6。本地完整克隆保持独立 Git 历史，根仓库 `.gitignore` 与另两个参考源一样排除其目录。更新参考源时，应重新记录 commit 并重建受影响图谱；本报告的结论固定在上述快照。

MLX/MLX-LM 继续承担 Apple 执行语义、模型数学与对照的角色，目前通过 `omlx` 及既有依赖/测试使用。此处“第三个参考源”按本地三个克隆计数。

## TensorFold 阅读入口

- [对 v5.1 的影响分析](../versions/v0.1-v5.1-pre-silicon/p01-p12-and-tensorfold/tensorfold-plan-impact-2026-10-06.md)：核对先前建议、源码证据、候选差异及拟议验收增补。
- [全仓分批图谱](../../graphify-out/tensorfold-reference/README.md)：完整 JSON 保留跨域边，HTML 分域浏览。
- [`lane_qmm.py`](../../TensorFold/src/tensorfold/kernels/qwen/dense/v1/lane_qmm.py)：固定 K 分片、BF16×unsigned codes、逐组 affine FMA、权重预处理。
- [`lane_fuse.py`](../../TensorFold/src/tensorfold/kernels/qwen/dense/v1/lane_fuse.py)：相同 K 分片才融合投影，保留每列原有舍入。
- [`exact_sampling.py`](../../TensorFold/src/tensorfold/engine/exact_sampling.py) 与 [`lane_family.py`](../../TensorFold/src/tensorfold/engine/lane_family.py)：同引擎串行/投机采样合同。
- [`Qwen35Family`](../../TensorFold/src/tensorfold/families/qwen3_5/family.py)：加载时检查 exact width 和逐宽度 forward 成本。
- [`family_depth.py`](../../TensorFold/src/tensorfold/engine/family_depth.py) 与 [`allocate.py`](../../TensorFold/src/tensorfold/engine/allocate.py)：按期望提交 token/时间分配深度与共享行。

## 复用与证据

TensorFold 0.6.0 起为 Apache-2.0，早期代码保留 MIT notice；直接移植源码时按具体文件来源保留 LICENSE/NOTICE 和修改说明。权重与 vendored drafter 的许可在其独立 notices 中。本次仅克隆与研究。

图谱是源码导航和文档主张的索引。Python 内嵌 Metal、独立 CUDA/Metal kernel 与 `INFERRED` 关系须直接核读；上游发布说明中的速度与本项目目标机器测量分开记录。
