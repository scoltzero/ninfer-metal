# m5rt — Apple10 (M5) Native Metal 4 & NAX Inference Runtime

> **Current Release (`v0.8.0-m5rt.4`, 2026-11-21)**: Qualified on physical **Apple M5 Ultra 64-core GPU / 96 GiB (`apple-m5-ultra-64gpu`, macOS `27.2.1`)** with `15` Mach-O executables (incl. `omlx_m5_hw_counter_probe`), `43` precompiled `.metallib` libraries, `84/84` local CTests + `79/79` live Apple10 CTests + `13/13` Sidecar pytests, and 100% `AR == DFlash2 == Batch` exact parity (`161.23 tok/s` 27B-MXFP4 Coding, `168.08 tok/s` 27B-oQ4e `B=4`, `463.46 tok/s` 35B-A3B MoE `B=8@64K`, `1.84 s` cold load, `24.5 GB` `B=8@64K` & `23.8 GB` `B=4@128K` via `FOSCAR` `Affine4Group128` KV). See [`../docs/versions/README.md`](../docs/versions/README.md) and [`../docs/versions/v0.8.0-m5rt.4-full-ninfer-squeeze-64gpu/WALKTHROUGH.md`](../docs/versions/v0.8.0-m5rt.4-full-ninfer-squeeze-64gpu/WALKTHROUGH.md).

## Historical Pre-Silicon Software Qualification (M4 Baseline)

P02-TF, P11-TF, P03-TF and P06-TF passed their separate M4 software gates on 2026-10-07. See [the per-package ledger](../docs/versions/v0.1-v5.1-pre-silicon/p01-p12-and-tensorfold/tensorfold-first-half-progress.md). The additions provide opt-in row-exact Linear/Attention contracts, a separately prepared affine Q4/G64 lane candidate, and position-keyed sampling with actual native AR/draft callers. Legacy remains the default in historical slice harnesses, while production `m5rt` runs `position_coupled_v1` by default.

This directory contains the native runtime developed by the M5 backend project.

The master implementation plan is in
[`../omlx-m5-backend-nlnfer-项目实施计划书.md`](../omlx-m5-backend-nlnfer-项目实施计划书.md) (historical pre-arrival execution table archived at
[`../docs/versions/v0.1-v5.1-pre-silicon/plans/pre-post-m5ultra-implementation-plan-v5.1.md`](../docs/versions/v0.1-v5.1-pre-silicon/plans/pre-post-m5ultra-implementation-plan-v5.1.md)), with a separate
hardware/software calibration reference in
[`../hardware-software-boundaries-v1.md`](../hardware-software-boundaries-v1.md).

Production AR and DFlash2 now execute the real `QwenForward` layer chain.
`DFlashForward` binds shared target weights and GPU layer features, uses
parallel multi-dispatch plans, and commits accepted prefixes with GPU ReplaySSM
after one target verify. Historical control programs remain test-only. The
qualification covers 57 Qwen cases, 55 draft cases through context 4096 and 54
ReplaySSM state cases. Full 27B/35B residency, NAX qualification and M5 Ultra
performance remain open. Run `../tools/reproduce_execution_repairs.sh` for the
current-machine checks.

The M4 slice qualification added two bounded improvements: T=8 SIMD-group
quantized matrix and T=1 lane candidates for measured 27B/35B weight shapes,
and incremental DFlash2 context projection with sliding K/V storage. The
production short-T route defaults on only when the device name is exactly
`Apple M4` and the format/shape gate matches; `m4_short_routes=on/off` is a
diagnostic override. On the real 27B Q4 head slice, paired T=8 GPU time fell
from 353 to 74 ms, while its candidate wall time remains about 74 ms versus
about 19 ms for the separately measured same-weight MLX operator. With the real
DFlash2 companion and synthetic target features at context 2048, hot-round GPU
time fell from about 4026 ms full-context rework to about 511 ms with incremental
context alone and about 229 ms combined with the M4 short-T candidate. Cold
fill remains about 3.83 seconds; the real companion output is numerically close
but not bit-exact. Neither full-model acceptance/tok/s nor M5 Ultra/NAX routes
are qualified by these M4 slices. See
[`../docs/versions/v0.1-v5.1-pre-silicon/m4-experiments/sliced-performance-assessment-2026-09-23.md`](../docs/versions/v0.1-v5.1-pre-silicon/m4-experiments/sliced-performance-assessment-2026-09-23.md) and
`../bench/results/sliced-baseline-2026-09-23/priority-1-2-qualification.json`.

Production route and long-KV diagnosis can be repeated with
`../tools/reproduce_route_attention.sh`. It audits the B2 slice candidates
against the exact selector used when QwenForward installs profiles, verifies
per-stage diagnostic ICB replay on a small complete model, and compares the
current Attention kernel with an isolated 512-token split-KV candidate at
2K/8K/32K synthetic history using the real 27B/35B head geometries. The
split-KV kernel is diagnostic-only; no production Attention or M5 Ultra route
is promoted by these M4 results. See
[`../docs/versions/v0.1-v5.1-pre-silicon/m4-experiments/route-attention-experiment-2026-09-23.md`](../docs/versions/v0.1-v5.1-pre-silicon/m4-experiments/route-attention-experiment-2026-09-23.md).

E08/E11 diagnostic slices can be repeated with
`../tools/reproduce_e08_e11.sh`. A TB=8 GDN mixer candidate passes Metal
Validation and numerical state comparison, but four complete quantized
projection-to-output slices show only about 0.5–1.1% T=1024 GPU improvement
on this M4. The 35B MoE grouped/fallback A16 comparison uses real expert
weights and explicitly synthetic route conditions, including a small control
model trace. Neither candidate is installed as a production route. See
[`../docs/versions/v0.1-v5.1-pre-silicon/m4-experiments/e08-e11-experiment-2026-09-23.md`](../docs/versions/v0.1-v5.1-pre-silicon/m4-experiments/e08-e11-experiment-2026-09-23.md).

The historical backend-first v4.30 plan is recorded in
`../omlx-m5-backend-nlnfer-项目实施计划书.md`. D0-D4 are completed
foundation and correctness work. B0 is complete: the backend-owned artifact,
three source-adapter families, Python/C++ readers and seven-checkpoint
Dense/MoE correctness evidence are
frozen. B1.1 has frozen the Metal library/pipeline registry, immutable command
plans, shared-event synchronization, capability adapter and resource ledger.
B1.2 has frozen segmented weight materialization, non-owning views and
workspace/state arenas. B1.3 has frozen activation-policy intersection, A4
workspace and cross-projection/expert reuse. B2.1 has frozen kernel keys,
candidate domains, installed-executor gates and fallback tables. B2.2 installed
Dense `T=1` oQ/MXFP4 QMV and B2.3 installed Dense affine oQ `T=2..12`
row-split/split-K/cooperative executors. B2.4 installed Dense MXFP4 `T=2..12`
W4A16/W4A4 SIMD executors plus shared A4 quantization and compiled the direct
multi-plane TensorOps candidates. B2.5 completed Dense oQ/MXFP4 interval and
bulk routes for every `T=13..128` plus 512/1024. B2.6 has completed the top-8
router and 256-expert grouped/fallback catalog.
The B3.1 parent/split-output contract and B3.2/B3.3 RMSNorm A16/A4 activation
production/reuse are complete. B3.4/B3.5 have completed attention q/k
postprocessing and Dense/MoE MLP layer execution. B3.6-B3.8 have completed
paged attention, persistent GDN and the vocabulary boundary. B3.9/B3.10 have
completed ordered command-plan batching and the immutable layer catalog;
G2-A has passed. B4-pre has migrated B1-B3 production submission and probes to
Metal 4 direct execution, with stable plan-owned constant buffers and classic
APIs confined to the compatibility adapter. B4.0 has proved record-once
compute-ICB replay over 10,000 changing parameter sets, consumer-barrier
correctness and whole-step
`execute_batch` fallback on both the classic compatibility route and the real
macOS 27 Metal 4 route. B4.1 has frozen dynamic per-step parameter tables,
persistent residency and the ICB-ready range ABI. B4.2 has frozen seven bounded
decode/prefill profiles per model topology, and B4.3 has assembled the native
`QwenProgram` resource owner and planned dispatch catalog. B5/G3 now pass with
a true one-dispatch, position-parallel DFlash2 masked-block Program, dynamic convolution,
conditional selector, dflash-mlx oracle parity and the full public companion
artifact contract. B6/G4 connect the production native Engine and nanobind
module to oMLX `M5Engine`. B7 has frozen the artifact set, full candidate
catalog, Profile A/B layer evidence, M4 candidate curves and fail-closed H0/H1/H2
entrypoints. The production path now separates the historical control executor
from real Qwen execution; full target-checkpoint and device qualification still
precede final route/tile promotion and product validation.

The completed D0-D4 foundation provides:

- a single-owner asynchronous Engine API;
- a deterministic fake backend with cancellation and failure injection;
- classic Metal and compile-only Metal 4/NAX smoke libraries;
- a Metal capability/bandwidth probe;
- C++ and nanobind contract tests.
- read-only sharded safetensors mapping and manifest validation;
- fixed Qwen3.8 model/config binding;
- affine g64 q4/q5 independent dequantization;
- Metal buffer/heap resources and per-sequence KV/GDN capacity planning.
- portable q4/q5 qmv/qmm, RMSNorm, RoPE, SiLU, sampling and counter RNG;
- KV attention plus single-step/chunked GDN reference execution;
- a config-driven, cancellable single-request Program and 256-token runner;
- ordinary Metal q4/q5 reference qmm and real 27B layer-slice parity tools.
- segmented immutable weight materialization with selective Text/Vision/MTP/spec
  loading and reader-independent parent/fused/expert views;
- resettable Program-owned workspace and per-sequence state arena primitives.
- `A16Only/AllowA4` policy intersection, epoch-scoped activation cache,
  producer leases and shared MXFP4 activation planes for projections/experts.
- sealed KernelKey/Candidate/DispatchTable contracts with planned versus
  installed executor gating and Dense/MoE/direct tail-safe fallbacks.
- installed Dense `T=1` oQ Q4/Q5/Q6/Q8 and MXFP4 W4A16 QMV executors with
  shared affine activation-group sums and real-shape oracle evidence.
- installed Dense affine oQ `T=2..12` row-split, split-K and cooperative
  executors that decode each weight value once per Linear and reuse it across T.
- installed Dense MXFP4 `T=2..12` W4A16/W4A4 SIMD executors, shared A4
  quantization and capability-gated TensorOps candidates.
- installed Dense oQ/MXFP4 broad/bulk executors covering every `T=13..128`
  plus exact 512/1024 anchors with two-dimensional token-block routing.
- installed 256-expert/top-8 MoE router, grouped A16/A4 expert executors and
  assignment-wise per-expert fallback routes for every supported T.
- fixed full-attention and GDN projection bundles with zero-copy strided
  query/gate, key/value and q/k/v/z output views.
- one-dispatch RMSNorm A16/A4/dual production with direct parent-Linear input,
  epoch-scoped activation reuse and bit-exact fused MXFP4 A4 planes.
- in-place per-head q/k RMSNorm plus MLX half-split RoPE over zero-copy
  interleaved query/gate views.
- Dense and MoE gate/up → SwiGLU A16/A4 → down execution, including
  assignment-major expert down, top-8 reduction and shared-expert combine.
- page-major causal GQA, persistent GDN convolution/recurrent state, direct
  BF16/quantized embedding and lm-head, and deterministic GPU sampling.
- one-command-buffer ordered plan batches and artifact-resolved static
  Dense/MoE layer schedules with zero token-loop artifact lookup.
- Metal 4 direct pipeline/argument-table execution for every B1-B3 command
  plan, stable constant buffers, explicit dispatch barriers and centralized
  classic compatibility fallback.
- record-once compute ICB replay with a stable parameter buffer, explicit RAW
  dependency barrier, residency set, 10,000-round oracle parity and fail-closed
  whole-step `execute_batch` fallback.
- fixed-address dynamic parameter tables, ICB-qualified plan validation,
  persistent residency ownership and named full/diagnostic command ranges.
- finite exact-topology decode/prefill profiles whose dynamic request data stays
  in stable parameter buffers and whose unsupported shapes fall back as a whole
  step.
- a native `QwenProgram` owner for materialized weights, static layer/stage
  plans, planned dispatch candidates, workspace/state, parameter tables and a
  single persistent residency set.
- whole-step `execute_batch` prefill/decode over real oQ/MXFP4 planes with
  paged KV/GDN state and Dense/MoE control-path oracle coverage.
- reusable `MetalIcbProfile` recording, persistent profile residency and one
  Metal 4 outer commit per steady-state decode step.
- `m5rt_runner` JSONL token events, greedy/top-k/top-p sampling, reset,
  second-request, cancellation and structured failure semantics.
- a startup-fixed `spec_backend + draft_tokens` contract (`none` and
  `dflash2 K=1..7` in R1; MTP deferred), a true one-dispatch masked-block
  companion Program with dynamic convolution and conditional selector, and
  `T=1..8` target/draft checkpoints, pending-anchor handling, ReplaySSM and
  atomic correction/bonus/commit paths.
- production `Engine::create`, native AR/DFlash2 event execution, nanobind
  load/submit/cancel/poll/stats/shutdown and the oMLX `M5Engine` adapter.

Build and verify:

```bash
./tools/reproduce_d0.sh
./tools/reproduce_d1.sh
./tools/reproduce_d2.sh
./tools/reproduce_b0.sh
./tools/reproduce_b1_1.sh
./tools/reproduce_b1_2.sh
./tools/reproduce_b1_3.sh
./tools/reproduce_b2_1.sh
./tools/reproduce_b2_2.sh
./tools/reproduce_b2_3.sh
./tools/reproduce_b2_4.sh
./tools/reproduce_b2_5.sh
./tools/reproduce_b2_6.sh
./tools/reproduce_b3_1.sh
./tools/reproduce_b3_2_3.sh
./tools/reproduce_b3_4.sh
./tools/reproduce_b3_5.sh
./tools/reproduce_b3_6.sh
./tools/reproduce_b3_7.sh
./tools/reproduce_b3_8.sh
./tools/reproduce_b3_9.sh
./tools/reproduce_b3_10.sh
./tools/reproduce_b4_pre.sh
./tools/reproduce_b4_0.sh
./tools/reproduce_b4_1.sh
./tools/reproduce_b4_2.sh
./tools/reproduce_b4_3.sh
./tools/reproduce_b4_4.sh
./tools/reproduce_b4_5.sh
./tools/reproduce_b4_6.sh
./tools/reproduce_b5.sh
./tools/reproduce_b6.sh
./tools/reproduce_b7.sh
```

On the target machine, run `run_m5_ultra_capability.sh`, then
`run_m5_ultra_tuning.sh`, then `run_m5_ultra_full_model.sh`. These entrypoints
fail closed outside an Apple M5 Ultra unless explicitly invoked for plan-only
or non-target harness development.

The reproduction script checks and, when needed, installs the exact D0 build
dependencies: MLX 0.32.2 and nanobind 2.15.0.

Results produced on the current M4 are correctness/harness evidence only. They
must never be presented as M5 Ultra performance.

## TensorFold execution contracts

Explicit `position_coupled_v1` provides coupled AR/DFlash target samples, per-slot execution identity and fixed1024 checkpoint boundaries. In `v0.8.0-m5rt.4`, `run_packed_batch` executes cross-slot single-weight-read `B×W` NAX forward across `B=1..8` slots with global COW `SharedPrefixCache` and `#5164` decode-protect chunking. See [`../omlx-m5-backend-nlnfer-项目实施计划书.md`](../omlx-m5-backend-nlnfer-项目实施计划书.md), [`../docs/backend/ninfer-m5-porting-architecture-v2.md`](../docs/backend/ninfer-m5-porting-architecture-v2.md), and [`../docs/versions/v0.1-v5.1-pre-silicon/p01-p12-and-tensorfold/tensorfold-second-half-progress.md`](../docs/versions/v0.1-v5.1-pre-silicon/p01-p12-and-tensorfold/tensorfold-second-half-progress.md) for qualification history and target-machine calibration.

## oMLX v0.8.0 Sidecar (`mseries_backendd`)

`m5rt/apps/mseries_backendd.cpp` builds the native `mseries_backendd` daemon, which serves `omlx` `v0.8.0` (`omlx/omlx/engine/m5_sidecar.py`, `M5SidecarEngine`, `external_engine="m5rt"`) over a user-private Unix domain socket using newline-delimited JSON-RPC v1 (`hello`, `load`, `submit`, `poll`, `cancel`, `capacity`, `stats`, `shutdown`), multi-connection push-streaming, pipelined JSONL, stale-socket recovery, parent-PID watchdog, and GIL-released Python bindings (`m5rt/bindings/python_module.cpp`).

- Sidecar RPC smoke (fake, native AR, native DFlash2, pipelining, dual-connection push, reconnect): `m5rt/tests/provider_sidecar_smoke.py`
- oMLX v0.8.0 unit, live daemon, and FastAPI E2E tests: `omlx/tests/test_m5_sidecar_engine.py`
- Full documentation index: [`../README.md`](../README.md), [`../docs/README.md`](../docs/README.md), and [`../m5-ultra-arrival-runbook.md`](../m5-ultra-arrival-runbook.md)
