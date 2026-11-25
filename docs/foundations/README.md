# 架构核心理念与跨引擎技术移植永久基石区（`docs/foundations/`）

> **专区性质**：**永久保留基石区（Permanent Architectural Foundations）**
> **维护守则**：本目录下的文档沉淀跨世代长期有效的硬件第一性原理、核心架构哲学与外部先进技术本土化移植准则。**在任何版本演进与仓库整理中均永久保留于本目录；每当吸收新一代突破性技术并经真机验证后，将提炼出的新公理增补入内。**

---

## 核心基石文档目录

| 基石文档 | 核心定位与用途 |
|---|---|
| **[`m5-ultra-porting-foundations.md`](m5-ultra-porting-foundations.md)** | **《Apple M5 Ultra (`Apple10`) 硬件禀赋、`ninfer` 核心哲学与跨源先进技术吸收移植基石宪章》**：系统沉淀 M5 Ultra 五大物理优势与五大硬红线、`ninfer` (RTX 5090) vs `ninfer-metal` (Apple10) 硬件驱动的架构分野、从 `ninfer` / `oMLX` (MLX) / `TensorFold` 萃取的设计精华、`ninfer-metal` 六大不可妥协护城河，以及**未来新算子/新架构移植的“六步第一性原理准入清单”**。 |
| **[`hardware-software-boundaries-v1.md`](hardware-software-boundaries-v1.md)** | **《NInfer/CUDA 与 Apple Metal 4/NAX：硬件、软件栈及实机标定边界》**：汇总 64 核 96GB / 80 核 256GB 物理硬边界、NInfer (`ebdc67e5` / RTX 5090) vs `ninfer-metal` (`miv` / Apple10) 九大子系统映射表、**C01–C13 实机标定边界账本**与跨硬件公平对标规则。 |

---

## 与仓库其他文档层级的分工边界

- **`docs/foundations/`（本目录 — 为什么这么设计 & 硬件物理边界）**：跨版本恒定的底层设计哲学、硅片基因映射、C01–C13 边界账本与跨框架技术吸收方法论。
- **[`docs/dim-guides/`](../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md) & [`docs/standards/`](../standards/core-certification-standard.md)（怎么写新核 & 怎么过认证）**：Apple M5+ 无损硅片压榨 Core-Authoring Skill 与四门禁核认证准入标准。
- **[`docs/silicon-guides/`](../silicon-guides/m5-series-optimization-playbook.md)（怎么跨 SKU 调优）**：Apple M5 Base/Pro/Max/Ultra 分档调优定律与统一内存准入方程。
- **[`docs/backend/`](../backend/README.md)（具体二进制与接口合同）**：`.omlxm5` 字节级容器规范、静态派发表 JSON 与按 SKU 严格隔离的硬件校准锁文件。
- **[`docs/releases/`](../releases/miv-v0.01-changelog.md) & [`docs/roadmap/`](../roadmap/v0.02-kvmem-actkv-plan.md)（版本交付与演进路线）**：当前 `miv-v0.01` 正式发布说明与 `v0.02+` `KVMem-UMA + ActKV` 技术路线图。
