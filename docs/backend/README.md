# `docs/backend/` 现行技术规范与硅片校准锁索引

> **当前基线**：`ninfer-metal` **`miv-v0.01`** + **`oMLX-extend` `v0.8.0-extend`**
> **目录职责**：本目录仅保留现行活跃技术合同，以及由 `miv/src/silicon_tuning.h`、`miv/tools/m5_silicon_profile.py`、`miv/tools/build_m5_release.py` 和 CTest 直接加载的机器可读校准 JSON。

---

## 1. 现行架构与技术合同（Active Contracts）

| 规范文件 | 适用范围 |
|---|---|
| [`ninfer-m5-porting-architecture-v2.md`](ninfer-m5-porting-architecture-v2.md) | NInfer (`ebdc67e5`) → Apple M5 系列（`miv/`）七大子系统移植架构规范与确定性奇偶矩阵 |
| [`artifact-format-v1.md`](artifact-format-v1.md) | `.miv` 容器规范（16 KiB 页对齐、ARMv8 HW CRC32、离线 NAX 预重排 `mx_nax_tiled_bk128_v1` / `oq_nax_tiled_g{64,128}_v1`） |
| [`dispatch-contract-v1.md`](dispatch-contract-v1.md) | `KernelKey`、数值等价组（`row_exact_group`）与静态路由表 Schema |
| [`quantization-kernel-strategy-v1.md`](quantization-kernel-strategy-v1.md) | MXFP4、oQ Affine（`q3/q4/q5/q6/q8 × g64/g128`）与 `FOSCAR` `Affine4Group128` KV 内核策略 |
| [`qwen36-format-extension-v1.md`](qwen36-format-extension-v1.md) | Qwen3.6 / Qwen3.8 Dense & MoE 源矩阵与混合精度格式扩展 |
| [`omlx-provider-dmg-roadmap-v1.md`](omlx-provider-dmg-roadmap-v1.md) | `oMLX-extend` UDS Sidecar / Provider 分发与桌面打包路线 |
| [`reference-sources.md`](reference-sources.md) | 上游 `ninfer`、`jundot/omlx`、`TensorFold` 参考源与工程边界说明 |

---

## 2. 机器可读 SKU 校准与路由锁文件（Tier-0 Locked Assets）

以下路径由运行时校准加载器与发布打包脚本直接读取，按 `profile_id` 严格隔离，禁止跨 SKU 混用或随意移动：

- [`silicon-calibrations/apple-m5-ultra-64gpu.json`](silicon-calibrations/apple-m5-ultra-64gpu.json)：Apple M5 Ultra 64 核 96 GiB 实机校准覆盖文件。
- [`silicon-calibrations/apple-m5-ultra-80gpu.json`](silicon-calibrations/apple-m5-ultra-80gpu.json)：Apple M5 Ultra 80 核 256 GiB 实机校准覆盖文件（覆盖 Dim D / E / F 实测常数）。
- [`m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json`](m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json)：64 核 96 GiB 静态派发表与证据溯源清单。
- [`route-candidate-catalog-v1.json`](route-candidate-catalog-v1.json)：23 族候选路由目录。
- [`tensorfold-benchmark-recipe-v1.json`](tensorfold-benchmark-recipe-v1.json)：同机公平对照基准协议配置。

---

## 3. 上级文档与发布导航

- **全仓精选文档总索引**：[`../README.md`](../README.md)
- **`miv-v0.01` 发布说明**：[`../releases/miv-v0.01-changelog.md`](../releases/miv-v0.01-changelog.md)
- **核认证标准与 `v0.02+` 路线图**：[`../standards/core-certification-standard.md`](../standards/core-certification-standard.md) · [`../roadmap/v0.02-kvmem-actkv-plan.md`](../roadmap/v0.02-kvmem-actkv-plan.md)
