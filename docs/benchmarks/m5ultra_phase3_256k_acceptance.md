# M5 Ultra Phase 3 `1K–256K` Prefill Acceleration Full Hardware Acceptance Matrix (`miv-v0.05-p3-rc1` vs Phase 2 `miv-v0.04-p2-rc1`)

> **Superseded by Phase 3.1 (`miv-v0.05-p3.1-rc1`)**: See [m5ultra_phase3_1_256k_acceptance.md](m5ultra_phase3_1_256k_acceptance.md) (and `.json`) for the subsequent Zero-LUT `B1_swar` MXFP4 + Zero-DRAM `128t BM=64 SWI8` Mode B acceptance report, sublayer GPU breakdown, and `ane_cal_v3_p31` canary results.

- **Date (UTC)**: `2026-11-28T00:08:48Z`
- **Host**: `scoltcdeMac-Studio.local` (`Apple M5 Ultra`, `36` CPU cores, **`80`-core Apple10 NAX GPU**, **Dual-ANE** `ane0`+`ane1`, **`256.0 GiB` UMA**, `macOS 27.2.1` Build `26B131`)
- **Deployed Provider**: `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/versions/miv-v0.05-p3-rc1` (**`current` symlink atomically switched**; release tarball `dist/ninfer-metal-miv-v0.05-macos27-arm64.tar.gz`, SHA-256 `b5eaa95ef8d763b8e58a421bf1aab61ea27c40e30b770be7be6b91c7a96b9397`, `17` clean-Mac binaries + `48` `.metallib`s)
- **Git Lineage**: Phase 2 Baseline `4db5b44` (`miv-v0.04-p2-rc1`) → Phase 3 Stage 2 Merge `c5f20cf` → Stage 3 Acceptance Branch `feat/p3-stage3-acceptance` (Hybrid QK-A8 `W4A4` default promotion + `ane_cal_v2_p3` calibration key upgrade)
- **Cold Prefill Envelope**: `--max-context 262144 --max-concurrency 1 --kv-dtype affine4_g128 --spec-backend none --prefix-cache-gb 0 --gen-tokens 128 --multi-needle --depths 10,50,90` (explicit `.miv v2` artifact binding, zero weight reconversion, unique nonce per trial, `cached_tokens == 0`, isolated port `:18090` under `/tmp/p3_m5u_gpu.lock`)
- **Production Safety Redlines**: Standalone `:8080` (`PID 75288`, `model.miv` v1) verified healthy & untouched (`0.0%–0.1%` CPU, `active_requests == 0`, continuous uptime `01-18:16:23`), `:8000` `oMLX` idle (`loaded_count == 0`), `versions/miv-v0.04-p2-rc1/` untouched, temporary `/tmp/p3_stage*` directories removed.
- **Global Multi-Needle NIAH & RAG QA Pass Rate**: **`540/540` (`100.0%`)** simultaneous `10%/50%/90%` multi-needle NIAH across all `21` full `1K–256K` cold matrix cells (`504/504`), Step 2 `W4A4` precision gate (`12/12`), and stale-cache ANE diagnostic (`24/24`), plus **`24/24` (`100.0%`)** multi-hop `CacheBlend` RAG QA!

## 1. Executive Summary & Headline Phase 3 Real-Hardware Wins

1. **`Qwen3.8-27B-mxfp4` (`model.v2.miv`) — WP 3A Shift-Merge `W4A8` Delivers `+15.2%` to `+17.5%` End-to-End Speedup Over Phase 2**:
   - **`M3b_A8` (`W4A8 + Sage2 + MInf`)**: **`2,576.2 tok/s` @ 16K** (`6.31s` vs Phase 2 `7.37s`, **`+16.8%`**), **`2,600.4 tok/s` @ 32K** (`12.55s` vs Phase 2 `14.75s`, **`+17.5%`**), and **`114.06s` (`2,297.2 tok/s`, `230.9 eff. TFLOPS`) @ 256K** vs Phase 2 `131.34s` (`1,995.0 tok/s`) — **`-17.28s` (`+15.2%`) faster than Phase 2** and **`3.10x` faster than `miv-v0.03 M0` (`353.84s`)** (`24/24` NIAH PASS, `19.33 GiB` Peak UMA).
   - **`M1_A8` (`W4A8 + Exact`)**: **`2408.9 tok/s` @ 16K** (`6.75s` vs Phase 2 `7.85s`, **`+16.3%`**) and **`312.65s` @ 256K** vs Phase 2 `330.68s` (**`-18.02s`**), while **improving `16K` teacher-forced PPL** from `1.4805` to **`1.4285`** (`91.16%` top-1, beating BF16 `M0` `1.4532`).
   - **`M3a_A8` (`W4A8 + Dense Sage2`, `MInf=OFF`)**: **`2,452.7 tok/s` @ 16K** and **`266.06s` (`984.8 tok/s`) @ 256K** — **`1.29x` faster than lossless `M0`** (`343.65s`) with zero sparsity (`24/24` NIAH PASS).
2. **`Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`) — WP 3B `2x 32x32x32` `W4A8` Pushes `256K` Down to `107.59s` (`2,435.3 tok/s`)**:
   - **`M3b_A8` (`32x32x32 W4A8 + Sage2 + MInf`)**: **`2,792.0 tok/s` @ 4K** (`+12.2%`), **`2,803.3 tok/s` @ 8K** (`+10.7%`), **`2,754.6 tok/s` @ 16K** (`5.90s` vs Phase 2 `6.17s`, `+4.6%`), **`2,779.3 tok/s` @ 32K**, and **`107.59s` (`2,435.3 tok/s`, `244.8 eff. TFLOPS`) @ 256K** vs Phase 2 `111.47s` (**`-3.88s`, `+3.6%`**) and **`3.23x` faster than `miv-v0.03 M0` (`347.23s`)** (`24/24` NIAH PASS, `23.26 GiB` Peak UMA).
   - **`M3a_A8` (`32x32x32 W4A8 + Dense Sage2`, `MInf=OFF`)**: breaks **`1,008.6 tok/s`** (`259.77s`) at `256K` dense attention (`1.31x` faster than `M0` `338.93s`, `24/24` NIAH PASS).
   - **Native `INT8xINT4 + q5` Dual-Channel (`OMLX_M5_OQ4_INT4B=1`)**: `24/24` NIAH PASS across `1K–256K` (`117.09s` / `2,237.8 tok/s` `M3b`, `317.45s` `M1`) with **bit-for-bit identical `4K` & `16K` PPL** (`2.5512 / 3.1696` `M1`, `2.4957 / 2.8898` `M3b`) — confirming WP 3B's Option B decision to ship `2x 32x32x32 i8xi8` as default (`+8.8%` faster at `256K`) while retaining `INT8xINT4` as an exact opt-in path.
3. **`MXFP4` Native `W4A4` Precision Gate & Default Promotion (`Hybrid QK-A8 W4A4`)**:
   - Pure `W4A4` (`OMLX_M5_W4A4_KEEP_QK_A8=0`) passed `3/3` `256K` NIAH (`133.98s`), but suffered severe `16K` teacher-forced PPL drift (`37.7321` Exact / `15.2800` `Sage2+MInf`).
   - Keeping only full-attention `q_proj`/`k_proj` on `W4A8` (`Hybrid QK-A8 W4A4`, `OMLX_M5_W4A4_KEEP_QK_A8=1`) slashed `16K` Exact PPL by **`6.96x`** (`37.7321 -> 5.4215`, top-1 `61.09% -> 76.10%`) and `Sage2+MInf` PPL by **`1.97x`** (`15.2800 -> 7.7565`) while running **`+0.4%` to `+0.8%` FASTER** (`7.473s` vs `7.535s` @ `16K`, `133.48s` vs `133.98s` @ `256K`). Promoted to **default-ON** in `w4a4_nax.h` / `native_backend.cpp` / `qwen_forward.cpp`, rebuilt & re-audited (`0 B` spills). Across full `1K–256K`, `M3b_A4` hits `2,224.1 tok/s` @ 32K and `132.44s` (`1,978.4 tok/s`) @ 256K (`24/24` NIAH PASS).
4. **WP 3C Dual-ANE `ultra_fp16_swiglu` Canary Auto-Guard & `ane_cal_v2_p3` Cache Fix**:
   - Live M5 Ultra `ane_cal_v1` inspection caught a real upgrade hazard: stale Stage 1 calibration records (taken before WP 3A's `+17%` `W4A8` speedup when Layer-0 `gpu_solo` was `9.30 ms`) kept `mxfp4 W4A8` active at `alpha=0.10`, slowing `256K` `M3b_A8_ANE` from `114.06s` to `121.80s` (`-6.4%`).
   - After bumping the calibration key to `ane_cal_v2_p3`, fresh Phase 3 canary sweeps showed WP 3A/3B accelerated Layer-0 `gpu_solo` to `7.70 ms` (`mxfp4 W4A8`) and `7.25 ms` (`oQ4e W4A8`), causing the `<= 0.96` auto-guard to cleanly bypass (`calibration_bypassed`) every accelerated route (`0.0%` regression across all `1K–256K` tiers), while activating `alpha=0.10` (`0.9586x` L0 speedup) on `mxfp4 BF16` (`+0.5%` net win at `8K–16K`: `8.638s` vs `8.680s`).
5. **Orthogonal `W4A4 + Sage2 + MInf + CacheBlend` & `oQ4e-mtp` Safe Fallback**:
   - Permuted multi-document RAG (`16K / 32K / 64K`) on `mxfp4` `M5b_A4` achieved **`4.26x / 4.24x / 4.09x`** warm prefill speedup (`ANE=off`, up to **`8,111.1 tok/s`**) and **`4.25x / 4.26x / 4.08x`** (`ANE=ultra_fp16_swiglu`) with **`24/24` (`100%`)** exact multi-hop QA PASS.
   - Live `oQ4e-mtp` `prefill_linear_mode="w4a4"` verification passed with zero error on both `mseries_backendd` (`prefill_linear_mode_effective="w4a8"`, `prefill_linear_fallback=true`, `prefill_linear_fallback_reason="requires_mxfp4_model"`, `w4a8_tokens=64`, `w4a4_tokens=0`) and `ninfer-metal-serve` (HTTP 200).

## 2. Apple10 4-Layer `<= 16.0 KiB` TG SRAM & `0 B` Register-Spill Audit (`15/15` Metallibs)

Verified via `python3 scripts/verify_apple10_sram_spills.py --all` directly against staged `versions/miv-v0.05-p3-rc1/` on 80-core Apple M5 Ultra (`applegpu_g17d`): **`passed == true`**, **`15/15` prefill/NAX/ANE metallibs**, **`1,327` kernels**, **`2,654/2,654` AGX-inspected PSOs (`icb=0` & `icb=1`)**, **`712` strict zero-spill prefill PSOs**, **`max_tg_sram_bytes = 16,384 B <= 16,384 B`**, **`max_spill_bytes = 0 B`**, Layer 3 Mach-O `applegpu_g17d` `__compute` PASS, Layer 4 `MTL_SHADER_VALIDATION=1` **`34` live smoke kernels PASS**.

| # | Audited `.metallib` | Total Kernels | Strict Prefill PSOs (`icb=0/1`) | Max TG SRAM (B) | Cap (B) | Max Prefill AGX Spill (B) |
|---:|---|---:|---:|---:|---:|---:|
| 1 | `omlx_m5_affine_nax.metallib` | 569 | 142 | `16,384` | `16,384` | **`0`** |
| 2 | `omlx_m5_mxfp4_nax.metallib` | 211 | 36 | `14,336` | `16,384` | **`0`** |
| 3 | `omlx_m5_moe_nax.metallib` | 127 | 152 | `16,384` | `16,384` | **`0`** |
| 4 | `omlx_m5_bf16_nax.metallib` | 233 | 8 | `14,336` | `16,384` | **`0`** |
| 5 | `omlx_m5_w4a8_nax.metallib` | 57 | 114 | `8,992` | `10,240` | **`0`** |
| 6 | `omlx_m5_w4a8_moe_nax.metallib` | 33 | 66 | `9,248` | `10,240` | **`0`** |
| 7 | `omlx_m5_w4a4_nax.metallib` | 9 | 18 | `9,264` | `10,240` | **`0`** |
| 8 | `omlx_m5_w4a4_moe_nax.metallib` | 3 | 6 | `9,520` | `10,240` | **`0`** |
| 9 | `omlx_m5_attention_nax_prefill.metallib` | 9 | 18 | `14,336` | `16,384` | **`0`** |
| 10 | `omlx_m5_attention_affine4.metallib` | 21 | 42 | `15,872` | `16,384` | **`0`** |
| 11 | `omlx_m5_attention_sage2.metallib` | 14 | 28 | `15,520` | `16,384` | **`0`** |
| 12 | `omlx_m5_attention_minference.metallib` | 5 | 10 | `11,264` | `16,384` | **`0`** |
| 13 | `omlx_m5_gdn_chunk_nax.metallib` | 14 | 28 | `16,384` | `16,384` | **`0`** |
| 14 | `omlx_m5_cache_blend.metallib` | 14 | 28 | `9,216` | `16,384` | **`0`** |
| 15 | `omlx_m5_ane_pack_unpack.metallib` | 8 | 16 | `4,224` | `10,240` | **`0`** |

## 3. Apple10 Silicon Microbenchmark Decisions (WP 3A Candidate α vs β & WP 3B `32x32x32` vs `INT8xINT4`)

### 3.1 WP 3A `MXFP4` Kernel Pipe Discovery (`[2048, 5120] x [17408, 5120]`, Realistic Gaussian Weights, `icb=1`, `0 B` Spills)

| Kernel / Candidate | NAX TensorOp Pipe | TG SRAM (B) | 8xCB TFLOPS | Q+8CB E2E TFLOPS | vs Phase 2 W4A8 (`130.7 TF`) | Decision |
|---|---|---:|---:|---:|---:|---|
| Phase 2 `mxfp4_w4a8` Baseline | `INT8 x INT8` (`16x32x32`) | `8,992` | `130.67` | `121.88` | `1.00x` | Replaced |
| **WP 3A Candidate α (`W4A8` Shift-Merge + CB-LUT)** | **`INT8 x INT8` (`32x32x32`)** | **`8,992`** | **`153.23`** *(AB: `150.40`)* | **`138.88`** | **`+17.3%` (`1.15x` E2E)** | **Winner (Default `W4A8`)** |
| WP 3A Candidate β (`TgMxfp8` Native FP8) | `FP8 x FP8` (`32x32`) | `<= 10,240` | `81.07` | `80.49` | `0.62x` *(HW pipe ceiling)* | Rejected |
| Native `TgMxfp4` (`FP4 x FP4` TensorOp) | `FP4 x FP4` (`32x32`) | `<= 10,240` | `87.65` | `86.18` | `0.67x` *(HW pipe ceiling)* | Superseded |
| **WP 3A Shift-Merge `W4A4` (Dual-Row `|de|<=3`)** | **`INT8 x INT8` (`32x32x32`)** | **`9,264`** | **`125.94`** *(AB: `123.45`)* | **`120.55`** | **`+43.7%` vs `TgMxfp4` / `+6.4%` vs BF16** | **Winner (Default `W4A4`)** |

### 3.2 WP 3B `oQ4e` Kernel Microbenchmark (`M=2048`, `0 B` Spills)

| Projection / Op | Phase 2 (`16x32x32`) | **Phase 3 Default (`2x 32x32x32`)** | Delta | `OMLX_M5_OQ4_INT4B=1` (`INT8xINT4`) | TG SRAM (B) |
|---|---:|---:|---:|---:|---:|
| QKVZ `q4` `[2048,5120]x[10240,5120]` | `163.5 TF` | **`166.0 TF`** | **`+1.5%`** | `150.9 TF` (`6,912 B`) | `8,960` |
| Down `q4` `[2048,17408]x[5120,17408]` | `161.3 TF` | **`163.8 TF`** | **`+1.5%`** | `149.2 TF` (`6,912 B`) | `8,960` |
| SwiGLU Mode A `q4` `[2048,5120]x[17408,5120]` | `151.3 TF` | **`170.8 TF`** | **`+12.9%`** | `151.7 TF` (`8,720 B`) | `8,976` |
| SwiGLU Mode A `q56` | `134.9 TF` | **`156.8 TF`** | **`+16.2%`** | — (`i8xi8` dual-ch) | `8,976` |
| SwiGLU Mode B `q4` (`+FWHT64+A8`) | `144.1 TF` | **`159.4 TF`** | **`+10.6%`** | `133.0 TF` (`8,720 B`) | `8,976` |
| SwiGLU Mode B `q56` (`+FWHT64+A8`) | `127.1 TF` | **`144.5 TF`** | **`+13.7%`** | — (`i8xi8` dual-ch) | `8,976` |

## 4. Step 2 `MXFP4 W4A4` Precision Gate (`Pure W4A4` vs `Hybrid QK-A8 W4A4`)

Evaluated on `Qwen3.8-27B-mxfp4/model.v2.miv` (`affine4_g128`, C++ teacher-forced PPL `chunk=512`, and live `ninfer-metal-serve` `16K/256K` simultaneous `10%/50%/90%` multi-needle NIAH):

| Variant (`OMLX_M5_W4A4_KEEP_QK_A8`) | `M1_A4` 4K PPL (Top-1) | `M1_A4` 16K PPL (Top-1) | `M3b_A4` 4K PPL (Top-1) | `M3b_A4` 16K PPL (Top-1) | `M3b_A4` 16K TTFT (tok/s) | `M3b_A4` 256K TTFT (tok/s) | 256K NIAH | Outcome |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `Pure W4A4` (`=0`) | `4.1492` (75.24%) | `37.7321` (61.09%) | `3.1050` (78.73%) | `15.2800` (67.58%) | `7.535s` (2157.5) | `133.98s` (1955.6) | **3/3** | Opt-in (`=0`) |
| **`Hybrid QK-A8 W4A4` (`=1`, Promoted Default)** | **`2.6961`** (79.83%) | **`5.4215`** (76.10%) | **`2.7929`** (79.37%) | **`7.7565`** (72.95%) | **`7.473s`** (2175.2) | **`133.48s`** (1963.0) | **3/3** | **Promoted Default** |

> **Decision**: Keeping full-attention `q_proj`/`k_proj` on `W4A8` cuts `16K` Exact PPL by **`6.96x`** (`37.73 -> 5.42`) and `Sage2+MInf` PPL by **`1.97x`** (`15.28 -> 7.76`) while running **`+0.82%` faster at `16K`** and **`+0.38%` faster at `256K`** (since `W4A8` at `153 TF` is faster than `W4A4` at `126 TF`). Promoted to default-ON in `w4a4_nax.h`, `native_backend.cpp`, and `qwen_forward.cpp`; rebuilt, re-staged, and re-audited (`0 B` spills).

## 5. Full 8-Tier Cold Prefill Matrix (`1K–256K`) — `Qwen3.8-27B-mxfp4` (`model.v2.miv`)

Artifact: `/Users/scoltc/repo/models/mlx-community/Qwen3.8-27B-mxfp4/model.v2.miv` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Config | 1K TTFT (tok/s) | 4K TTFT (tok/s) | 8K TTFT (tok/s) | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs P2 `M3b` / `v0.03 M0` | 16K / 256K Eff. TF (HW TF) | Peak UMA | NIAH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.03 M0` (Ref BF16+Exact) | `0.62s` (1451) | `2.05s` (1933) | `4.04s` (1998) | `8.30s` (1944) | `18.24s` (1782) | `43.56s` (1499) | `116.59s` (1122) | `353.84s` (740) | `0.37x` / `1.00x` | `101.4` / `74.4` (`74.4`) | `19.34 GiB` | **24/24** |
| P2 `M0_v2` (BF16+Exact) | `0.62s` (1452) | `2.12s` (1867) | `4.20s` (1921) | `8.65s` (1879) | `18.77s` (1739) | `43.90s` (1490) | `114.10s` (1148) | `343.94s` (762) | `0.38x` / `1.03x` | `98.1` / `76.6` (`76.6`) | `19.13 GiB` | **24/24** |
| P2 `M1_v2` (W4A8+Exact) | `0.58s` (1549) | `1.93s` (2056) | `3.80s` (2120) | `7.85s` (2071) | `17.16s` (1902) | `40.67s` (1608) | `107.60s` (1217) | `330.68s` (792) | `0.40x` / `1.07x` | `108.1` / `79.6` (`79.6`) | `19.16 GiB` | **24/24** |
| P2 `M2a_v2` (BF16+Dense Sage2) | `0.63s` (1412) | `2.13s` (1861) | `4.17s` (1932) | `8.35s` (1947) | `18.06s` (1808) | `41.12s` (1591) | `103.50s` (1265) | `296.95s` (882) | `0.44x` / `1.19x` | `101.6` / `88.7` (`88.7`) | `19.45 GiB` | **24/24** |
| P2 `M3b_v2` (W4A8+Sage2+MInf) | `0.59s` (1530) | `1.93s` (2055) | `3.76s` (2146) | `7.37s` (2206) | `14.75s` (2213) | `29.99s` (2181) | `61.93s` (2114) | `131.34s` (1995) | `1.00x` / `2.69x` | `115.1` / `200.5` (`104.2`) | `19.33 GiB` | **24/24** |
| `M0` (Lossless Base (BF16 + Exact, ANE=off)) | `0.64s` (1401) | `2.15s` (1846) | `4.22s` (1909) | `8.68s` (1873) | `18.80s` (1736) | `43.93s` (1489) | `114.15s` (1147) | `343.65s` (762) | **`0.38x`** / **`1.03x`** | `97.8` / `76.6` (`76.6`) | `19.13 GiB` | **24/24** |
| `M0_ANE` (Lossless Base + Dual-ANE (`ultra_fp16_swiglu`, active `a=0.10`)) | `0.65s` (1378) | `2.31s` (1720) | `4.20s` (1919) | `8.64s` (1882) | `18.92s` (1725) | `45.21s` (1447) | `121.12s` (1081) | `362.53s` (723) | **`0.36x`** / **`0.98x`** | `98.2` / `72.7` (`72.7`) | `21.63 GiB` | **24/24** |
| **`M1_A8`** (Route A (`W4A8` Shift-Merge + Exact, ANE=off)) | `0.54s` (1651) | `1.68s` (2357) | `3.27s` (2465) | `6.75s` (2409) | `14.93s` (2186) | `36.19s` (1807) | `98.64s` (1328) | `312.65s` (838) | **`0.42x`** / **`1.13x`** | `125.7` / `84.2` (`84.2`) | `19.14 GiB` | **24/24** |
| `M1_A8_ANE` (Route A (`W4A8` + Exact, `ultra_fp16_swiglu` auto-bypass)) | `0.54s` (1671) | `1.68s` (2360) | `3.27s` (2467) | `6.75s` (2410) | `14.93s` (2186) | `36.17s` (1808) | `98.66s` (1327) | `312.61s` (838) | **`0.42x`** / **`1.13x`** | `125.8` / `84.2` (`84.2`) | `19.13 GiB` | **24/24** |
| `M1_A4` (Route A4 (`W4A4` Hybrid QK-A8 + Exact, ANE=off)) | `0.61s` (1470) | `1.95s` (2036) | `3.80s` (2120) | `7.81s` (2081) | `17.06s` (1913) | `40.42s` (1618) | `107.06s` (1223) | `330.08s` (794) | **`0.40x`** / **`1.07x`** | `108.6` / `79.8` (`79.8`) | `19.13 GiB` | **24/24** |
| `M1_A4_ANE` (Route A4 (`W4A4` + Exact, `ultra_fp16_swiglu` auto-bypass)) | `0.61s` (1474) | `1.95s` (2035) | `3.80s` (2120) | `7.81s` (2081) | `17.06s` (1914) | `40.44s` (1617) | `107.13s` (1222) | `329.81s` (794) | **`0.40x`** / **`1.07x`** | `108.6` / `79.8` (`79.8`) | `19.14 GiB` | **24/24** |
| **`M3a_A8`** (Route A+B1 (`W4A8` + Dense `Sage2`, `MInf=OFF`)) | `0.55s` (1639) | `1.69s` (2341) | `3.19s` (2531) | `6.63s` (2453) | `14.32s` (2280) | `33.57s` (1948) | `88.45s` (1480) | `266.06s` (985) | **`0.49x`** / **`1.33x`** | `128.0` / `99.0` (`99.0`) | `19.46 GiB` | **24/24** |
| `M3a_A4` (Route A4+B1 (`W4A4` + Dense `Sage2`, `MInf=OFF`)) | `0.61s` (1477) | `2.03s` (1958) | `3.81s` (2117) | `7.81s` (2082) | `16.53s` (1974) | `38.33s` (1706) | `97.45s` (1344) | `285.78s` (917) | **`0.46x`** / **`1.24x`** | `108.7` / `92.2` (`92.2`) | `19.45 GiB` | **24/24** |
| **`M3b_A8`** (Route A+B2 (`W4A8` Shift-Merge + `Sage2` + `MInf`, ANE=off)) | `0.54s` (1653) | `1.69s` (2344) | `3.25s` (2477) | `6.31s` (2576) | `12.55s` (2600) | `25.65s` (2550) | `53.29s` (2457) | `114.06s` (2297) | **`1.15x`** / **`3.10x`** | `134.5` / `230.9` (`120.0`) | `19.33 GiB` | **24/24** |
| `M3b_A8_ANE` (Route A+B2 (`W4A8` + `Sage2` + `MInf`, `ultra_fp16_swiglu` auto-bypass)) | `0.54s` (1655) | `1.70s` (2341) | `3.26s` (2473) | `6.32s` (2572) | `12.56s` (2598) | `25.65s` (2550) | `53.34s` (2455) | `114.09s` (2297) | **`1.15x`** / **`3.10x`** | `134.3` / `230.8` (`119.9`) | `19.35 GiB` | **24/24** |
| **`M3b_A4`** (Route A4+B2 (`W4A4` Hybrid QK-A8 + `Sage2` + `MInf`, ANE=off)) | `0.61s` (1480) | `2.00s` (1984) | `3.82s` (2113) | `7.39s` (2200) | `14.68s` (2224) | `30.07s` (2175) | `62.45s` (2097) | `132.44s` (1978) | **`0.99x`** / **`2.67x`** | `114.8` / `198.8` (`103.3`) | `19.33 GiB` | **24/24** |
| `M3b_A4_ANE` (Route A4+B2 (`W4A4` + `Sage2` + `MInf`, `ultra_fp16_swiglu` auto-bypass)) | `0.61s` (1472) | `2.00s` (1983) | `3.81s` (2114) | `7.39s` (2200) | `14.65s` (2227) | `30.07s` (2175) | `62.33s` (2101) | `132.19s` (1982) | **`0.99x`** / **`2.68x`** | `114.8` / `199.2` (`103.5`) | `19.34 GiB` | **24/24** |

> Decode throughput across `Qwen3.8-27B-mxfp4` remained invariant across all Phase 3 prefill modes: **`53.7–53.8 tok/s` (`1K–8K`) → `53.3–53.4 tok/s` (`16K`) → `45.8 tok/s` (`64K`) → `42.6 tok/s` (`128K`) → `39.3 tok/s` (`256K`)**.

### 5.1 `4K` & `16K` C++ Teacher-Forced Logprob Perplexity — `Qwen3.8-27B-mxfp4` (`model.v2.miv`)

| Mode | 4K Mean NLL | 4K PPL | 4K Top-1 | 16K Mean NLL | 16K PPL | 16K `ΔPPL` vs P2 `M0_v2` (`1.4532`) | 16K Top-1 | Peak UMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Phase 2 `M1_v2` (Ref `W4A8` + Exact) | `0.72508` | `2.0649` | `83.17%` | `0.39240` | `1.4805` | `+0.0273` | `90.55%` | `15.49 GiB` |
| Phase 2 `M3b_v2` (Ref `W4A8` + Sage2 + MInf) | `0.74199` | `2.1001` | `83.08%` | `0.40629` | `1.5012` | `+0.0480` | `90.36%` | `15.56 GiB` |
| **`M0`** (BF16 + Exact) | `0.73995` | **`2.0958`** | `83.03%` | `0.37377` | **`1.4532`** | `+0.0000` | `90.90%` | `15.42 GiB` |
| **`M2a`** (BF16 + Dense Sage2) | `0.73077` | **`2.0767`** | `83.17%` | `0.35892` | **`1.4318`** | `-0.0214` | `91.21%` | `15.47 GiB` |
| **`M2b`** (BF16 + Sage2 + MInf) | `0.73077` | **`2.0767`** | `83.17%` | `0.36895` | **`1.4462`** | `-0.0070` | `90.87%` | `15.50 GiB` |
| **`M1_A8`** (W4A8 Shift-Merge + Exact) | `0.73036` | **`2.0758`** | `83.17%` | `0.35660` | **`1.4285`** | `-0.0247` | `91.16%` | `15.49 GiB` |
| **`M3a_A8`** (W4A8 Shift-Merge + Dense Sage2) | `0.73475` | **`2.0850`** | `83.05%` | `0.36318` | **`1.4379`** | `-0.0153` | `91.12%` | `15.54 GiB` |
| **`M3b_A8`** (W4A8 Shift-Merge + Sage2 + MInf) | `0.73475` | **`2.0850`** | `83.05%` | `0.38263` | **`1.4661`** | `+0.0129` | `90.53%` | `15.57 GiB` |
| **`M1_A4_hybrid`** (W4A4 Hybrid QK-A8 (Default) + Exact) | `0.99179` | **`2.6961`** | `79.83%` | `1.69037` | **`5.4215`** | `+3.9683` | `76.10%` | `15.49 GiB` |
| **`M3a_A4_hybrid`** (W4A4 Hybrid QK-A8 (Default) + Dense Sage2) | `1.02709` | **`2.7929`** | `79.37%` | `1.87649` | **`6.5305`** | `+5.0773` | `75.13%` | `15.54 GiB` |
| **`M3b_A4_hybrid`** (W4A4 Hybrid QK-A8 (Default) + Sage2 + MInf) | `1.02709` | **`2.7929`** | `79.37%` | `2.04853` | **`7.7565`** | `+6.3033` | `72.95%` | `15.56 GiB` |
| **`M1_A4_pure`** (W4A4 Pure (KEEP_QK_A8=0) + Exact) | `1.42292` | **`4.1492`** | `75.24%` | `3.63051` | **`37.7321`** | `+36.2789` | `61.09%` | `15.45 GiB` |
| **`M3a_A4_pure`** (W4A4 Pure (KEEP_QK_A8=0) + Dense Sage2) | `1.13301` | **`3.1050`** | `78.73%` | `2.99148` | **`19.9151`** | `+18.4619` | `67.14%` | `15.50 GiB` |
| **`M3b_A4_pure`** (W4A4 Pure (KEEP_QK_A8=0) + Sage2 + MInf) | `1.13301` | **`3.1050`** | `78.73%` | `2.72655` | **`15.2800`** | `+13.8268` | `67.58%` | `15.53 GiB` |

## 6. Full 8-Tier Cold Prefill Matrix (`1K–256K`) — `Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`)

Artifact: `/Users/scoltc/repo/models/Jundot/Qwen3.8-27B-oQ4e-mtp/model.v2.miv` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Config | 1K TTFT (tok/s) | 4K TTFT (tok/s) | 8K TTFT (tok/s) | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs P2 `M3b` / `v0.03 M0` | 16K / 256K Eff. TF (HW TF) | Peak UMA | NIAH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.03 M0` (Ref BF16+Exact) | `0.51s` (1746) | `2.02s` (1966) | `3.94s` (2047) | `8.21s` (1965) | `17.82s` (1824) | `42.76s` (1526) | `114.31s` (1144) | `347.23s` (754) | `0.32x` / `1.00x` | `102.5` / `75.8` (`75.8`) | `23.21 GiB` | **24/24** |
| P2 `M0_v2` (BF16+Exact) | `0.47s` (1902) | `2.14s` (1857) | `3.92s` (2057) | `8.36s` (1944) | `18.03s` (1810) | `42.72s` (1531) | `111.70s` (1172) | `339.21s` (772) | `0.33x` / `1.02x` | `101.5` / `77.6` (`77.6`) | `23.09 GiB` | **24/24** |
| P2 `M1_v2` (W4A8+Exact) | `0.51s` (1766) | `1.67s` (2380) | `3.22s` (2503) | `6.63s` (2451) | `14.69s` (2221) | `35.65s` (1835) | `97.89s` (1338) | `311.45s` (841) | `0.36x` / `1.11x` | `127.9` / `84.6` (`84.6`) | `23.04 GiB` | **24/24** |
| P2 `M2a_v2` (BF16+Dense Sage2) | `0.60s` (1482) | `2.12s` (1872) | `3.89s` (2072) | `8.05s` (2018) | `17.46s` (1870) | `39.99s` (1636) | `101.08s` (1295) | `291.81s` (898) | `0.38x` / `1.19x` | `105.3` / `90.3` (`90.3`) | `23.40 GiB` | **24/24** |
| P2 `M3b_v2` (W4A8+Sage2+MInf) | `0.50s` (1801) | `1.59s` (2489) | `3.18s` (2533) | `6.17s` (2634) | `12.32s` (2649) | `25.09s` (2607) | `52.09s` (2514) | `111.47s` (2350) | `1.00x` / `3.11x` | `137.5` / `236.3` (`122.8`) | `23.26 GiB` | **24/24** |
| `M0` (Lossless Base (BF16 + Exact, ANE=off)) | `0.65s` (1383) | `2.10s` (1886) | `3.92s` (2056) | `8.44s` (1925) | `18.04s` (1809) | `42.74s` (1530) | `111.78s` (1171) | `338.93s` (773) | **`0.33x`** / **`1.02x`** | `100.5` / `77.7` (`77.7`) | `23.07 GiB` | **24/24** |
| `M0_ANE` (Lossless Base (BF16 + Exact, `ultra_fp16_swiglu` auto-bypass)) | `0.60s` (1504) | `2.14s` (1856) | `3.92s` (2055) | `8.44s` (1926) | `18.26s` (1787) | `42.71s` (1531) | `111.69s` (1172) | `338.83s` (773) | **`0.33x`** / **`1.02x`** | `100.5` / `77.7` (`77.7`) | `23.09 GiB` | **24/24** |
| **`M1_A8`** (Route A (New `32x32x32` `W4A8` + Exact, ANE=off)) | `0.39s` (2277) | `1.66s` (2389) | `2.91s` (2771) | `6.20s` (2623) | `14.17s` (2303) | `34.67s` (1887) | `95.63s` (1369) | `306.76s` (854) | **`0.36x`** / **`1.13x`** | `136.9` / `85.8` (`85.8`) | `23.07 GiB` | **24/24** |
| `M1_A8_ANE` (Route A (`32x32x32` `W4A8` + Exact, `ultra_fp16_swiglu` auto-bypass)) | `0.43s` (2100) | `1.54s` (2581) | `2.91s` (2769) | `6.32s` (2572) | `14.17s` (2304) | `34.67s` (1887) | `95.63s` (1369) | `306.91s` (854) | **`0.36x`** / **`1.13x`** | `134.3` / `85.8` (`85.8`) | `23.07 GiB` | **24/24** |
| `M1_A8_INT4B` (Route A (`OMLX_M5_OQ4_INT4B=1` `INT8xINT4+q5` + Exact)) | `0.52s` (1719) | `1.76s` (2251) | `3.36s` (2399) | `7.01s` (2320) | `15.44s` (2114) | `37.22s` (1757) | `100.78s` (1299) | `317.45s` (825) | **`0.35x`** / **`1.09x`** | `121.1` / `83.0` (`83.0`) | `23.07 GiB` | **24/24** |
| **`M3a_A8`** (Route A+B1 (`32x32x32` `W4A8` + Dense `Sage2`, `MInf=OFF`)) | `0.51s` (1746) | `1.63s` (2431) | `2.88s` (2802) | `6.20s` (2621) | `13.40s` (2435) | `31.92s` (2049) | `84.87s` (1543) | `259.77s` (1009) | **`0.43x`** / **`1.34x`** | `136.8` / `101.4` (`101.4`) | `23.39 GiB` | **24/24** |
| **`M3b_A8`** (Route A+B2 (New `32x32x32` `W4A8` + `Sage2` + `MInf`, ANE=off)) | `0.53s` (1704) | `1.42s` (2792) | `2.88s` (2803) | `5.90s` (2755) | `11.74s` (2779) | `23.93s` (2734) | `49.87s` (2626) | `107.59s` (2435) | **`1.04x`** / **`3.23x`** | `143.8` / `244.8` (`127.3`) | `23.26 GiB` | **24/24** |
| `M3b_A8_ANE` (Route A+B2 (`32x32x32` `W4A8` + `Sage2` + `MInf`, `ultra_fp16_swiglu` auto-bypass)) | `0.51s` (1768) | `1.62s` (2446) | `3.04s` (2650) | `5.91s` (2753) | `11.76s` (2776) | `23.96s` (2730) | `49.91s` (2624) | `107.05s` (2448) | **`1.04x`** / **`3.24x`** | `143.7` / `246.0` (`127.9`) | `23.29 GiB` | **24/24** |
| `M3b_A8_INT4B` (Route A+B2 (`OMLX_M5_OQ4_INT4B=1` `INT8xINT4+q5` + `Sage2` + `MInf`)) | `0.55s` (1634) | `1.65s` (2410) | `3.35s` (2409) | `6.53s` (2490) | `13.01s` (2508) | `26.49s` (2469) | `54.90s` (2385) | `117.09s` (2238) | **`0.95x`** / **`2.97x`** | `129.9` / `224.9` (`116.9`) | `23.27 GiB` | **24/24** |

> Decode throughput across `Qwen3.8-27B-oQ4e-mtp` remained invariant across all Phase 3 prefill modes: **`49.4–49.5 tok/s` (`1K–8K`) → `49.2 tok/s` (`16K`) → `42.5 tok/s` (`64K`) → `39.6 tok/s` (`128K`) → `36.8 tok/s` (`256K`)**.

### 6.1 `4K` & `16K` C++ Teacher-Forced Logprob Perplexity — `Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`)

| Mode | 4K Mean NLL | 4K PPL | 4K Top-1 | 16K Mean NLL | 16K PPL | 16K `ΔPPL` vs `M0` (`4.9330`) | 16K Top-1 | Peak UMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`M0`** (BF16 + Exact) | `1.08587` | **`2.9620`** | `77.73%` | `1.59595` | **`4.9330`** | `+0.0000` | `76.35%` | `19.35 GiB` |
| **`M2a`** (BF16 + Dense Sage2) | `1.00625` | **`2.7353`** | `78.83%` | `1.37446` | **`3.9529`** | `-0.9801` | `78.78%` | `19.40 GiB` |
| **`M2b`** (BF16 + Sage2 + MInf) | `1.00625` | **`2.7353`** | `78.83%` | `1.43524` | **`4.2006`** | `-0.7324` | `78.01%` | `19.43 GiB` |
| **`M1_A8`** (W4A8 32x32x32 + Exact) | `0.93657` | **`2.5512`** | `79.58%` | `1.15362` | **`3.1696`** | `-1.7634` | `81.05%` | `19.36 GiB` |
| **`M3a_A8`** (W4A8 32x32x32 + Dense Sage2) | `0.91456` | **`2.4957`** | `79.71%` | `1.12575` | **`3.0825`** | `-1.8505` | `81.22%` | `19.41 GiB` |
| **`M3b_A8`** (W4A8 32x32x32 + Sage2 + MInf) | `0.91456` | **`2.4957`** | `79.71%` | `1.06117` | **`2.8898`** | `-2.0432` | `81.63%` | `19.43 GiB` |
| **`M1_A8_INT4B`** (W4A8 INT8xINT4+q5 (OQ4_INT4B=1) + Exact) | `0.93657` | **`2.5512`** | `79.58%` | `1.15362` | **`3.1696`** | `-1.7634` | `81.05%` | `19.36 GiB` |
| **`M3a_A8_INT4B`** (W4A8 INT8xINT4+q5 (OQ4_INT4B=1) + Dense Sage2) | `0.91456` | **`2.4957`** | `79.71%` | `1.12575` | **`3.0825`** | `-1.8505` | `81.22%` | `19.41 GiB` |
| **`M3b_A8_INT4B`** (W4A8 INT8xINT4+q5 (OQ4_INT4B=1) + Sage2 + MInf) | `0.91456` | **`2.4957`** | `79.71%` | `1.06117` | **`2.8898`** | `-2.0432` | `81.63%` | `19.44 GiB` |

> Both `32x32x32` `W4A8` and `OMLX_M5_OQ4_INT4B=1` (`INT8xINT4 + q5` dual-channel) match Phase 2 `W4A8` PPL to **5 decimal places of Mean NLL** (`0.93657 / 1.15362` `M1`, `0.91456 / 1.06117` `M3b`), proving bit-exact numerical equivalence of both WP 3B pipelines.

## 7. WP 3C Dual-ANE `ultra_fp16_swiglu` Canary Auto-Guard & `ane_cal_v2_p3` Findings

| Model & Route | L0 `gpu_solo` (ms) | L0 Best Hybrid (ms) | Ratio (`<=0.96` gate) | `best_alpha` | Phase 3 Decision | 16K GPU vs ANE | 256K GPU vs ANE |
|---|---:|---:|---:|---:|---|---:|---:|
| `mxfp4` `BF16` (`M0`) | `10.358` | `9.930` | **`0.9586x`** | `0.10` | **ACTIVE** (`ane=260,096` @ 256K) | `8.68s` → **`8.64s` (`+0.5%`)** | `343.65s` → `362.53s` (`-5.2%` quad-attn DRAM) |
| `mxfp4` `W4A8` (`M3b_A8`) | `7.702` | `7.516` | `0.9759x` | `0.09` | **Bypassed** (`calibration_bypassed`) | `6.31s` → **`6.32s` (`0.0%`)** | `114.06s` → **`114.09s` (`0.0%`)** |
| `mxfp4` `W4A4` (`M3b_A4`) | `9.031` | `8.744` | `0.9682x` | `0.10` | **Bypassed** (`calibration_bypassed`) | `7.39s` → **`7.39s` (`0.0%`)** | `132.44s` → **`132.19s` (`+0.2%`)** |
| `oQ4e` `BF16` (`M0`) | `10.009` | `9.639` | `0.9631x` | `0.09` | **Bypassed** (`calibration_bypassed`) | `8.44s` → **`8.44s` (`0.0%`)** | `338.93s` → **`338.83s` (`0.0%`)** |
| `oQ4e` `W4A8` (`M3b_A8`) | `7.251` | `7.143` | `0.9851x` | `0.08` | **Bypassed** (`calibration_bypassed`) | `5.90s` → **`5.91s` (`0.0%`)** | `107.59s` → **`107.05s` (`+0.5%`)** |

- **Real-Hardware Upgrade Catch (`ane_cal_v1 -> ane_cal_v2_p3`)**: Before bumping the calibration key, `mxfp4 M3b_A8_ANE` reused a cached Stage 1 `ane_cal_v1` record measured against slower Phase 2 `W4A8` (`gpu_solo=9.299 ms -> active=True, alpha=0.10`), causing `256K` TTFT to regress from `114.06s` to **`121.80s`** (`2151.2 tok/s`, `-6.4%`, `ane=260,096`, `24/24` NIAH PASS). Bumping the cache namespace to `ane_cal_v2_p3` in `miv/src/ane/ane_dual_offload.mm` forced fresh recalibration against Phase 3's `7.70 ms` `W4A8` kernels, restoring `114.09s` (`0.0%` regression).

## 8. Orthogonal `CacheBlend` (`W4A4 + Sage2 + MInf`) & Live `oQ4e` Safe-Fallback Verification

### 8.1 `Qwen3.8-27B-mxfp4` Permuted Multi-Doc RAG (`M5b_A4`: `W4A4 + Sage2 + MInf + CacheBlend`, `prefix_cache_gb=24.0`)

| ANE Mode | Context (Docs) | Cold Prime TTFT (tok/s) | Permuted Warm TTFT (tok/s) | Warm Speedup | Blend Hit / Recomp Tokens | Peak UMA | Multi-Hop QA |
|---|---|---:|---:|---:|---:|---:|---|
| `off` (GPU-only) | **16K** (`4x4K`) | `8.972s` (1877.4) | **`2.105s` (7996.9)** | **`4.26x`** | `16,657` / `2,992` | `28.18 GiB` | **4/4 PASS** |
| `off` (GPU-only) | **32K** (`8x4K`) | `17.380s` (1911.9) | **`4.096s` (8111.1)** | **`4.24x`** | `33,041` / `5,888` | `30.00 GiB` | **4/4 PASS** |
| `off` (GPU-only) | **64K** (`16x4K`) | `34.737s` (1899.9) | **`8.492s` (7770.6)** | **`4.09x`** | `65,809` / `11,680` | `33.63 GiB` | **4/4 PASS** |
| `ultra_fp16_swiglu` | **16K** (`4x4K`) | `8.936s` (1884.9) | **`2.104s` (8000.6)** | **`4.25x`** | `16,657` / `2,992` | `28.53 GiB` | **4/4 PASS** |
| `ultra_fp16_swiglu` | **32K** (`8x4K`) | `17.371s` (1912.8) | **`4.073s` (8155.8)** | **`4.26x`** | `33,041` / `5,888` | `32.22 GiB` | **4/4 PASS** |
| `ultra_fp16_swiglu` | **64K** (`16x4K`) | `34.715s` (1901.1) | **`8.510s` (7754.6)** | **`4.08x`** | `65,809` / `11,680` | `35.86 GiB` | **4/4 PASS** |

### 8.2 Live `Qwen3.8-27B-oQ4e-mtp` `w4a4 → w4a8` Safe-Fallback Verification

- **`mseries_backendd` (AF_UNIX JSON-RPC `load` + `submit` + `stats`)**: zero error, `model_quant_format="oq_affine"`, `prefill_linear_mode="w4a4"`, **`prefill_linear_mode_effective="w4a8"`**, **`prefill_linear_fallback=true`**, **`prefill_linear_fallback_reason="requires_mxfp4_model"`**, `prefill_w4a8_tokens=64`, `prefill_w4a4_tokens=0`.
- **`ninfer-metal-serve` (`:18090` `--prefill-linear-mode w4a4`)**: HTTP `200` completion succeeded cleanly with `ninfer_metal_prefill_w4a8_tokens_total = 64`, `ninfer_metal_prefill_w4a4_tokens_total = 0`.

## 9. Release Packaging, Atomic `current` Switch & Safety Audit

- **Release Tarball**: `dist/ninfer-metal-miv-v0.05-macos27-arm64.tar.gz` (`25,016,918 B`, SHA-256 `b5eaa95ef8d763b8e58a421bf1aab61ea27c40e30b770be7be6b91c7a96b9397`), clean-Mac dylib audit PASSED (macOS system frameworks only), `17` binaries + `48` `.metallib`s (including `omlx_m5_w4a4_nax.metallib`, `omlx_m5_w4a4_moe_nax.metallib`, `omlx_m5_ane_pack_unpack.metallib`) + `omlx_m5_prefill_sram_spill_audit_tests` staged in `versions/miv-v0.05-p3-rc1/`.
- **Atomic Deployment**: `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current` atomically replaced via POSIX `rename(2)` (`os.replace`) to `versions/miv-v0.05-p3-rc1` alongside `provider.json`. Post-switch health & `/v1/chat/completions` smoke test via `current/bin/ninfer-metal-serve` on `:18090` PASSED (resolved `realpath` to `versions/miv-v0.05-p3-rc1/bin/ninfer-metal-serve`).
- **Cleanup & Production Safety**: Removed all temporary `/tmp/p3_stage*` directories on `192.168.10.245`. Verified `versions/miv-v0.04-p2-rc1/` untouched, `:8000` `oMLX` idle (`loaded_count == 0`), and production `:8080` `PID 75288` continuously healthy (`ELAPSED 01-18:16:23`, `0.1%` CPU, `active_requests == 0`, `status == "ok"`).

