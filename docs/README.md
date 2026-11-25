# Documentation Index

Current baseline: **`ninfer-metal` `miv-v0.01`** (`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`, with native C++20 `ninfer-metal-serve`) + **`oMLX-extend` `omlx-extend-v0.8.3`**.

All seven primary user-facing guides (`README`, `oMLX-extend` Visual Guide, Getting Started, Architecture, Microkernel Foundry, `miv-v0.01` Release Notes, and Roadmap) ship in **8 languages**: English at the canonical paths below, plus full mirrors under `docs/i18n/<lang>/` for `zh-CN`, `zh-TW`, `ja`, `ko`, `es`, `fr`, and `de`:

[English](../README.md) | [简体中文](i18n/zh-CN/README.md) | [繁體中文](i18n/zh-TW/README.md) | [日本語](i18n/ja/README.md) | [한국어](i18n/ko/README.md) | [Español](i18n/es/README.md) | [Français](i18n/fr/README.md) | [Deutsch](i18n/de/README.md)

---

## User Guides

| Document | Summary |
|---|---|
| **[`oMLX-extend` Visual Guide](omlx-extend-guide.md)** | 5-step beginner screenshot walkthrough of `oMLX` on macOS: microkernel cards, active engine badges, `miv Core Backend (Auto)` & Stock MLX toggle, MTP / DFlash2 & 4-bit KVQuant, and 1-click shared config |
| **[Getting Started & Config](getting-started.md)** | Prerequisites, zero-MLX CMake build, clean-Mac packaging, 1-click `start_ninfer_serve.sh`, `ninfer-metal-serve` standalone OpenAI server, `oMLX-extend` setup, `ninfer-metal.toml`, and low-level CLI/daemon usage |
| **[Architecture & Coexistence](architecture.md)** | Dual-mode `miv/` core (`ninfer-metal-serve` + `mseries_backendd`), Provider SPI v2, O(1) auto-routing, UMA memory coordination, monorepo dual-track CI & hot-update |
| **[`oMLX-extend` README](../omlx-extend/README.md)** | Thin Provider SPI patch over `jundot/omlx`, `miv Core Backend` UI setting unlock table, hot-update directory layout, upstream sync scripts |
| **[`ninfer-metal.toml`](../omlx-extend/config/ninfer-metal.toml)** & **[`.example`](../omlx-extend/config/ninfer-metal.toml.example)** | Ready-to-run shared default config (plus full commented reference) auto-loaded by standalone `ninfer-metal-serve` and `oMLX-extend` |

---

## Microkernel Foundry & Certification

| Document | Summary |
|---|---|
| **[Microkernel Foundry](microkernel-foundry.md)** | Open contributor hub for adding new M5+ model-family microkernels (Path A shared substrate PRs & Path B parallel sidecars) |
| **[Silicon-Squeeze Core-Authoring Skill](dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** | *(ZH)* Bare-silicon `MTLGPUFamilyApple10+` methodology: `H0/H1` calibration, `50`-line cold-DRAM probes, zero-spill NAX/ICB, `< 10 s` `24 MHz` profiling |
| **[Core Certification Standard](standards/core-certification-standard.md)** | *(ZH)* Mandatory Gate 1–4 acceptance rules (SPI v2, Roofline discipline, Dim A–F clean-room matrix, 5-file evidence pack, C1–C6 sign-off) |

---

## Releases & Roadmap

| Document | Summary |
|---|---|
| **[`miv-v0.01` Release Notes](releases/miv-v0.01-changelog.md)** | Visual Dim A–F benchmark charts, `ninfer-metal-serve` / `miv/` / `oMLX-extend` changes, `otool -L` clean-Mac audit, supported scope, and known limitations |
| **[Roadmap](roadmap/README.md)** | Planned milestones beyond `miv-v0.01` (`KVMem-UMA + ActKV`, `libninfer_metal_substrate`, new model families, `ninfer-metal-serve` Phase 2) |
| **[`v0.02+` KVMem-UMA & ActKV Plan](roadmap/v0.02-kvmem-actkv-plan.md)** | *(ZH)* Detailed engineering spec and Verification Gates V1–V6 for `256K → 1M+` agent workspaces and NVMe / token-level ActKV |

---

## Hardware Guides & Foundations

| Document | Summary |
|---|---|
| **[Foundations Index](foundations/README.md)** | *(ZH)* Reading guide to permanent Apple M5 porting foundations |
| **[M5 Ultra Porting Foundations](foundations/m5-ultra-porting-foundations.md)** | *(ZH)* Apple M5 Ultra hardware invariants, RTX 5090 vs Apple10 architectural differences, and six-step porting checklist |
| **[Hardware & Software Boundaries v1](foundations/hardware-software-boundaries-v1.md)** | *(ZH)* 64-GPU 96 GiB & 80-GPU 256 GiB C01–C13 boundaries, NInfer subsystem mapping, and cross-hardware benchmarking rules |
| **[Apple M5 Series Optimization Playbook](silicon-guides/m5-series-optimization-playbook.md)** | *(ZH)* `32 KiB` threadgroup occupancy formulas, UMA admission math, and Base / Pro / Max / Ultra tuning matrix |
| **[Qwen3.8-27B oQ4e Quantization Analysis](quantization.md)** | *(ZH)* Mixed 4-bit / 5-bit `g64` affine tensor inventory and checkpoint geometry |
| **[NAX–ANE Crossover Model v0](nax-ane-crossover-model-v0.md)** | *(ZH)* Analytical model for GPU NAX vs standalone ANE dispatch crossover |

---

## Backend Specifications & SKU Calibrations (`docs/backend/`)

Full overview in **[`backend/README.md`](backend/README.md)**:

- **Architecture & Contracts**: [`ninfer-m5-porting-architecture-v2.md`](backend/ninfer-m5-porting-architecture-v2.md) · [`artifact-format-v1.md`](backend/artifact-format-v1.md) · [`dispatch-contract-v1.md`](backend/dispatch-contract-v1.md) · [`quantization-kernel-strategy-v1.md`](backend/quantization-kernel-strategy-v1.md) · [`qwen36-format-extension-v1.md`](backend/qwen36-format-extension-v1.md)
- **Distribution & Sources**: [`omlx-provider-dmg-roadmap-v1.md`](backend/omlx-provider-dmg-roadmap-v1.md) · [`reference-sources.md`](backend/reference-sources.md)
- **Measured SKU Calibrations**: [`apple-m5-ultra-64gpu.json`](backend/silicon-calibrations/apple-m5-ultra-64gpu.json) · [`apple-m5-ultra-80gpu.json`](backend/silicon-calibrations/apple-m5-ultra-80gpu.json)
- **Static Dispatch & Benchmark Schemas**: [`m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json`](backend/m5-static-dispatch-table-apple-m5-ultra-64gpu-v1.json) · [`route-candidate-catalog-v1.json`](backend/route-candidate-catalog-v1.json) · [`tensorfold-benchmark-recipe-v1.json`](backend/tensorfold-benchmark-recipe-v1.json)

---

[Back to Repository Root](../README.md)
