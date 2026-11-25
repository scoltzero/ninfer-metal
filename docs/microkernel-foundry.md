**English** | [简体中文](i18n/zh-CN/microkernel-foundry.md) | [繁體中文](i18n/zh-TW/microkernel-foundry.md) | [日本語](i18n/ja/microkernel-foundry.md) | [한국어](i18n/ko/microkernel-foundry.md) | [Español](i18n/es/microkernel-foundry.md) | [Français](i18n/fr/microkernel-foundry.md) | [Deutsch](i18n/de/microkernel-foundry.md)

# Microkernel Foundry

`ninfer-metal` (`miv-v0.01`) ships bare-silicon Metal 4 NAX inference on Apple M5+ for Qwen3.5, Qwen3.6, and Qwen3.8 hybrid models. **Microkernel Foundry** opens the same runtime substrate, serving frontends, calibration tooling, and certification pipeline to contributors and coding agents adding new model families via mainline pull requests or parallel sidecars.

Target families include **Llama, Qwen3 Pure-Attention, DeepSeek (MLA), Gemma, GLM, MiniMax**, and community quantized derivatives such as [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx).

---

## Two Ways to Contribute

### Path A: Mainline PR → Contributor Credit (Preferred)

Merge your model-family microkernel directly into `ninfer-metal` on top of the shared **`libninfer_metal_substrate`** runtime.

**Provided by the shared substrate:**
- Metal 4 execution runtime (`MetalContext`, `MTL4CommandQueue`, double-buffered `ar_pipe_[2]`, `< 2 GiB` Compute ICB encoder, `read_write` barrier reflection).
- Shared paged KV and prefix memory (`KvPagePool`, `SharedKvArenaPool`, CoW refcounting, `SharedPrefixCache`, `UmaCheckpointPool`).
- Generic NAX and decode primitives (`mxfp4_nax`, `affine_nax`, `bf16_nax`, `moe_nax`, zero-`uitofp` QMV, two-stage parallel `Argmax`, `Split-KV` / `Flash-NAX`).
- `.miv` (and legacy `.omlxm5`) container loader, bundled weight-repack converter (`tools/m5_artifact.py`), `ExecutionIdentityFields`, `24 MHz` hardware counters, native standalone OpenAI server (`ninfer-metal-serve`), and Provider SPI v2 daemon (`mseries_backendd`).

**What your PR adds & runtime scope:** Add your architecture's layer graph, weight mapping, `.safetensors → .miv` weight-repack rule in the converter, and architecture-specific kernels (for example MLA, GeGLU, Softcap, or custom MoE routers) that beat generic primitives by **`>= +10%–12%`** on cold-DRAM micro-probes. Sharing `libninfer_metal_substrate` avoids code and resident memory duplication across architectures, runs your model natively in both standalone `ninfer-metal-serve` and `oMLX-extend` (with automatic first-`Load` `.safetensors → .miv` repacking alongside original weights), and lists your work in the [Supported Models](../README.md) roster with contributor attribution.

### Path B: Parallel Sidecar → PR or Issue → Original Repo Attribution

For experimental architectures, custom quantization layouts, or an independent release schedule, package your core as a standalone Sidecar SPI v2 bundle under:

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

Expose an atomic `current` symlink with a `provider.json` declaring `supported_topologies`, `supported_model_types`, `artifact_extensions`, and a bundled `converter` entrypoint (`--source <dir> --output <file> --source-repo <id>`) so standard `.safetensors` models downloaded from Hugging Face or mirrors automatically repack on first `Load` in `oMLX-extend` without deleting original weights. Once ready, submit a PR or open an Issue: maintainers register it in [`omlx-extend/registry/microkernels.json`](../omlx-extend/registry/microkernels.json) so users install and update it via [`./omlx-extend/scripts/sync_microkernels.sh`](../omlx-extend/scripts/sync_microkernels.sh), and list it in the [Supported Models](../README.md) roster with a direct link to your original repository.

**Unified multi-microkernel hosting:** Both `oMLX-extend` and standalone `ninfer-metal-serve --providers-dir` natively discover `Providers/*/current/provider.json` and run multiple Path A (**`miv Core Backend`**) and Path B microkernels in parallel on one port with unified UMA budget arbitration and `gpu_contended` scheduling (alongside Stock MLX in `oMLX-extend`). In the `oMLX` macOS app, installed Path B microkernels appear automatically as cards under **`Integrations → External Engines`** and display active engine badges in **`Models`** (see [`oMLX-extend` Visual Guide](omlx-extend-guide.md)). Microkernel authors only write the Metal 4 core and Sidecar SPI v2 daemon; HTTP/SSE serving, tokenization, chat templates, and multi-model scheduling are handled by both hosts.

**Licensing:** **Path A** mainline PRs link into `miv` and are distributed under **[GPL-3.0](../LICENSE)** (contributors may submit GPL-3.0, Apache-2.0, or MIT code); **Path B** parallel sidecars communicate over AF_UNIX Sidecar SPI v2 as separate processes and may use any license.

---

## Model-Family Roster

| Architecture Family | Target Examples | Status | Recommended Path |
|---|---|---|---|
| **Qwen Hybrid (3:1 GDN + GQA)** | Qwen3.5 / 3.6 / 3.8 Dense (0.8B – 27B) & 35B-A3B MoE | Shipped (`miv-v0.01`) | Path A (Mainline, `miv/`) |
| **Pure-Attention GQA** | Llama-3.3-70B, Qwen3-32B, Gemma | Open for PRs | Path A |
| **Multi-Head Latent Attention (MLA)** | DeepSeek V3 / R1 family | Open for PRs | Path A / Path B |
| **MoE & Community Quantizations** | GLM, MiniMax, `heretic-mlx` derivatives | Open for PRs | Path A / Path B |

---

## How to Author and Certify a Microkernel

Both human engineers and coding agents follow one reproducible four-step loop:

1. **Calibrate the target Mac (`Stage B: H0 + H1`)**
   Run `H0` hardware DMA sweeps (`MTLBlitPassDescriptor`, `4–512+ MiB`) to measure your chip's $B_{\text{DRAM}}$, $C_{\text{SLC}}$, $P_{\text{NAX}}$, and $I_{\text{ridge}}$, followed by `H1` operator sweeps. Every acceptance threshold normalizes to these measured constants.
2. **Cold-DRAM micro-probe first (`~50–100` lines)**
   Before templating a new kernel family, benchmark a standalone cold-DRAM probe (`working set >= 3–4 × C_SLC`, `>= 30` serial-barrier medians) against fixed-function NAX/hardware baselines. Keep kernels with **`>= +10%–12%`** net gain; delete failed experimental PSOs before opening a PR.
3. **Fast `< 10 s` sublayer loop & spill audit**
   Profile GPU sublayer shares (`mlp/moe`, `attn`, `lm_head`, `sampling`) with `omlx_m5_hw_counter_probe --mode model` (`24 MHz` counters) and audit shaders with `metal-objdump -d` (`0 alloca` spills, `<= 104` VGPR on wide prefill, `0 uitofp` on `T=1` pre-tiled QMV).
4. **Submit the 5-file machine-readable evidence pack**
   Run the clean-room Dim A–F harness (`miv/tools/bench_omlx_vs_m5rt.py`) and attach `certification-evidence/<core-id>-<version>-<sku>/`:
   - `h0-capability.json` — Bare-silicon `H0` constants & 10,000-step ICB canary
   - `h1-calibration-<sku>.json` — SKU-isolated `H1` tuning & `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — Clean-room Dim A–F results & `phys_footprint`
   - `parity-report.json` — `0-ULP` parity, source PPL/KL, NIAH/RULER recall, `0 NaN/Inf`
   - `otool-audit.txt` — Pure Apple-framework `otool -L`, `0 alloca` audit, 100-turn soak & CTest output

---

## Required Reading for Contributors & Agents

- **[Silicon-Squeeze Core-Authoring Skill](dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — Complete M5+ (`MTLGPUFamilyApple10+`) bare-silicon methodology ready to load into coding agents.
- **[Core Certification Standard](standards/core-certification-standard.md)** — Exact Gate 1–4 pass criteria, Provider SPI v2 contract, Roofline thresholds, and Tier 1 / Tier 2 certification badges.
- **[Apple M5 Series Optimization Playbook](silicon-guides/m5-series-optimization-playbook.md)** — `32 KiB` threadgroup occupancy math, unified-memory admission equations, and Base / Pro / Max / Ultra scaling.
