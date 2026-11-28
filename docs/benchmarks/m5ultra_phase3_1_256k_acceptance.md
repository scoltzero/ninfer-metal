# M5 Ultra Phase 3.1 `1K–256K` Prefill Acceleration Full Hardware Acceptance Matrix (`miv-v0.05-p3.1-rc1` vs Phase 3 `miv-v0.05-p3-rc1` & Phase 2 `miv-v0.04-p2-rc1`)

- **Date (UTC)**: `2026-11-28T06:18:01Z`
- **Host**: `scoltcdeMac-Studio.local` (`Apple M5 Ultra`, `36` CPU cores, **`80`-core Apple10 NAX GPU**, **Dual-ANE** `ane0`+`ane1`, **`256.0 GiB` UMA**, `macOS 27.2.1` Build `26B131`)
- **Deployed Provider**: `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/versions/miv-v0.05-p3.1-rc1` (**`current` symlink & `provider.json` atomically switched via POSIX `os.replace`**; release tarball `dist/ninfer-metal-miv-v0.05-p3.1-rc1-macos27-arm64.tar.gz`, `25,590,931 B`, SHA-256 `4712afd5cacf36a264c55c31ac9c24da58fa7b173c2d1b00705ecc293f832957`, `17` clean-Mac binaries + `48` `.metallib`s; `oMLX-extend` UI package `dist/omlx-extend/oMLX-extend-v0.8.3-mix7-macos26-27.dmg`, SHA-256 `3e11b788ec68a6f43a33796977f5c55e609d2a91bd5cf9ce53508327b2418572`)
- **Git Lineage**: Phase 2 `4db5b44` (`miv-v0.04-p2-rc1`) → Phase 3 `ceb0586` (`miv-v0.05-p3-rc1`) → Stage 1A UI & Universal Baseline `d6dd5c5` (`feat/p31-ui-merged`) + Stage 1B NAX Zero-LUT & Zero-DRAM Mode B `4d884a0` (`feat/p31-nax-kernels`) → Release Branch `feat/p31-release` (`ane_cal_v3_p31`)
- **Cold Prefill Envelope**: `--max-context 262144 --max-concurrency 1 --kv-dtype affine4_g128 --spec-backend none --prefix-cache-gb 0 --gen-tokens 128 --multi-needle --depths 10,50,90` (explicit `.miv v2` artifact binding, zero weight reconversion, unique nonce per trial, `cached_tokens == 0`, isolated port `:18090` under `flock /tmp/p3_m5u_gpu.lock`)
- **Production Safety Redlines**: Standalone `:8080` (`PID 75288`, `model.miv` v1) continuously healthy & untouched (`0.0%` CPU, `active_requests == 0`, uptime `02-00:26:20`), `:8000` `oMLX` healthy & idle (`loaded_count == 0`), `versions/miv-v0.04-p2-rc1/` & `versions/miv-v0.05-p3-rc1/` preserved intact, `/tmp/p31*` scratch cleaned.
- **Global Multi-Needle NIAH Pass Rate**: **`192/192` (`100.0%`)** simultaneous `10%/50%/90%` multi-needle NIAH across all `8` full `1K–256K` `A8` acceptance matrix cells (`64` context tiers × `3` depths) with **bit-for-bit identical `4K` & `16K` teacher-forced PPL** against Phase 3!

## 1. Executive Summary & Phase 3.1 Headline Wins

1. **Zero-LUT `B1_swar` MXFP4 Unpack + Zero-DRAM `128t BM=64 SWI8` Mode B Break the MLP Ceiling (`+16.0%` MXFP4 / `+8.5%` Affine Mode B)**:
   - Eliminated all W4A8 host `mxfp4_lut` buffer bindings via a `512 B` on-chip TG SRAM LUT + SWAR bit-shift (`B1_swar + OptMode=1`: `153.15 TF` GateUp / `151.70 TF` Down / `152.04 TF` QKVZ, `0 B` spills) with zero-block `slow_flag` exemption (`99.999666%` fast-path on `Qwen3.8-27B-mxfp4/model.v2.miv`).
   - Replaced `256t` Mode B (trapped at `1` TG/core above Apple10's `96`-GPR occupancy cliff) with zero-DRAM `128t BM=64 OUT_N=32x2 SWI8` (`3` TGs/core = `12` SGs/core up to `128` GPRs, `-4.56 GiB/chunk` DRAM eliminated), lifting live 64-layer `mlp_swiglu_mode_b` throughput to **`150.11 TF`** on `MXFP4` (`2,490.4 ms` @ `16K` vs `2,898.4 ms` pre-P3.1, **`-407.9 ms` / `+16.4%`**) and **`161.53 TF`** on `oQ4e` (`2,314.4 ms` @ `16K`, outperforming `linear_mlp_down` at `155.20 TF`).
2. **`Qwen3.8-27B-mxfp4` (`model.v2.miv`) — `M3b_A8` Reaches `2,715.4 tok/s` @ 32K and `110.52s`–`111.99s` (`2,340–2,371 tok/s`) @ 256K**:
   - **`M3b_A8` (`W4A8 + Sage2 + MInf`)**: **`2,645.4 tok/s` @ 16K** (`6.14s` vs P3 `6.31s`, `+2.7%`), **`2,715.4 tok/s` @ 32K** (`12.02s` vs P3 `12.55s`, **`+4.4%`**), **`2,534.0 tok/s` @ 128K** (`51.68s` vs P3 `53.29s`, `+3.1%`), and **`111.99s` (`2,339.7 tok/s`) @ 256K** (`M3b_A8_ANE` auto-bypass: **`110.52s` / `2,370.9 tok/s`**, **`1.19x` faster than Phase 2 `131.34s`** and **`3.20x` faster than `v0.03 M0` `353.84s`**).
   - **`M3a_A8` (`W4A8 + Dense Sage2`)**: **`2,520.8 tok/s` @ 16K** (`6.45s` vs P3 `6.63s`), **`2,345.1 tok/s` @ 32K** (`13.92s` vs P3 `14.32s`), **`263.85s` (`993.0 tok/s`) @ 256K** (`-2.21s` vs P3).
   - **`M1_A8` (`W4A8 + Exact`)**: **`2,457.8 tok/s` @ 16K** (`6.61s` vs P3 `6.75s`), **`310.44s` (`844.0 tok/s`) @ 256K** (`-2.21s` vs P3).
   - **Teacher-Forced PPL (`4K / 16K`)**: **Bit-exact** vs Phase 3 across all three modes (`M1_A8`: **`2.0758 / 1.4285`**, `M3a_A8`: **`2.0850 / 1.4379`**, `M3b_A8`: **`2.0850 / 1.4661`**).
3. **`Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`) — `M3b_A8` Breaks `2,803–2,832 tok/s` @ 32K and `106.65s` (`2,456.8 tok/s`) @ 256K**:
   - **`M3b_A8` / `M3b_A8_ANE` (`W4A8 + Sage2 + MInf`)**: **`2,803.8–2,831.7 tok/s` @ 32K** (`11.53s–11.64s` vs P3 `11.74s`), **`2,735.8–2,741.0 tok/s` @ 64K** (`23.86s`), **`2,633.2–2,643.8 tok/s` @ 128K** (`49.53s` vs P3 `49.87s`), and **`106.65s` (`2,456.8 tok/s`, `246.2 eff. TF`) @ 256K** on `M3b_A8_ANE` (`109.47s` / `2,393.4 tok/s` on `M3b_A8` right after 25m sustained `mxfp4` thermal load) — up to **`3.26x` faster than `v0.03 M0` (`347.23s`)**.
   - **Teacher-Forced PPL (`4K / 16K`)**: **Bit-exact** vs Phase 3 (`M1_A8`: **`2.5512 / 3.1696`**, `M3a_A8`: **`2.4957 / 3.0825`**, `M3b_A8`: **`2.4957 / 2.8898`**).
4. **Full-Stack Productization & `ane_cal_v3_p31` Auto-Guard**:
   - Integrated `p31-ui-merged` (`d6dd5c5`, universal capability + 6 semantic fixes + public `W4A4 / A4` UI hide narrowing `provider.json` `prefill_linear_mode` to `["bf16", "w4a8"]`) with `p31-nax-kernels` (`4d884a0`).
   - Bumped ANE canary key `ane_cal_v2_p3 -> ane_cal_v3_p31`: fresh M5 Ultra canary measured faster Layer-0 `gpu_solo` (`7.702 -> 7.372 ms` on `mxfp4`, `7.251 -> 7.171 ms` on `oQ4e`), cleanly auto-bypassing (`calibration_bypassed`, `ane=0`, zero regression across `1K–256K`).
   - All local gates passed **100%**: `ctest` **`101/101`**, `pytest` **`128 passed / 4 skipped`**, `apply_spi_patch.sh --check-only` **3-way PASS**, `apps/miv-mac` `swift test` **`7/7`**, Xcode `oMLXTests` **`54/54`** (`ModelSettingsScreenVMTests` `49/49` + `LocalizationSmokeTests` `5/5`).

## 2. Apple10 4-Layer `<= 16.0 KiB` TG SRAM & `0 B` Register-Spill Audit (`15/15` Metallibs)

Verified via `python3 scripts/verify_apple10_sram_spills.py --all` against staged `versions/miv-v0.05-p3.1-rc1/` on 80-core Apple M5 Ultra (`applegpu_g17d`): **`passed == true`**, **`15/15` metallibs**, **`1,328` kernels**, **`2,656/2,656` AGX-inspected PSOs (`icb=0` & `icb=1`)**, **`714` strict zero-spill prefill PSOs**, **`w4a8_nax` max `10,000 B <= 10,240 B`**, **`max_spill_bytes = 0 B`**, Layer 3 Mach-O `applegpu_g17d` `__compute` PASS, Layer 4 `MTL_SHADER_VALIDATION=1` **`34/34` live smoke kernels PASS**.

| # | Audited `.metallib` | Total Kernels | Strict Prefill PSOs (`icb=0/1`) | Max TG SRAM (B) | Cap (B) | Max GPR | Max Prefill AGX Spill (B) |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | `omlx_m5_affine_nax.metallib` | 569 | 142 | `16,384` | `16,384` | `126` | **`0`** |
| 2 | `omlx_m5_mxfp4_nax.metallib` | 211 | 36 | `14,336` | `16,384` | `123` | **`0`** |
| 3 | `omlx_m5_moe_nax.metallib` | 127 | 152 | `16,384` | `16,384` | `126` | **`0`** |
| 4 | `omlx_m5_bf16_nax.metallib` | 233 | 8 | `14,336` | `16,384` | `126` | **`0`** |
| 5 | `omlx_m5_w4a8_nax.metallib` | 58 | 116 | `10,000` | `10,240` | `126` | **`0`** |
| 6 | `omlx_m5_w4a8_moe_nax.metallib` | 33 | 66 | `9,760` | `10,240` | `126` | **`0`** |
| 7 | `omlx_m5_w4a4_nax.metallib` | 9 | 18 | `9,264` | `10,240` | `120` | **`0`** |
| 8 | `omlx_m5_w4a4_moe_nax.metallib` | 3 | 6 | `9,520` | `10,240` | `126` | **`0`** |
| 9 | `omlx_m5_attention_nax_prefill.metallib` | 9 | 18 | `14,336` | `16,384` | `126` | **`0`** |
| 10 | `omlx_m5_attention_affine4.metallib` | 21 | 42 | `15,872` | `16,384` | `125` | **`0`** |
| 11 | `omlx_m5_attention_sage2.metallib` | 14 | 28 | `15,520` | `16,384` | `125` | **`0`** |
| 12 | `omlx_m5_attention_minference.metallib` | 5 | 10 | `11,264` | `16,384` | `114` | **`0`** |
| 13 | `omlx_m5_gdn_chunk_nax.metallib` | 14 | 28 | `16,384` | `16,384` | `126` | **`0`** |
| 14 | `omlx_m5_cache_blend.metallib` | 14 | 28 | `9,216` | `16,384` | `111` | **`0`** |
| 15 | `omlx_m5_ane_pack_unpack.metallib` | 8 | 16 | `4,224` | `10,240` | `33` | **`0`** |

## 3. Phase 3.1 Microbenchmark Decisions & `.miv v2` `slow_flag` Hardware Scan

### 3.1 Zero-LUT MXFP4 W4A8 Unpack Candidates (`M=2048`, Qwen3.8-27B Gaussian Weights, `icb=1`, `0/73,728` Bit-Exact)

| Candidate | TG SRAM | GPR | Spill | GateUp `17408x5120` | Down `5120x17408` | QKVZ `12288x5120` | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| const_lut (Phase 3 baseline) | `8,992 B` | `124` | `0 B` | `143.67 TF` | `136.60 TF` | `142.83 TF` | Replaced |
| A (simd_shuffle) | `8,992 B` | — | `0 B` | `89.19 TF` | `87.62 TF` | `88.74 TF` | Rejected |
| B1 naive (char2+char4<<shv) | `9,504 B` | — | `0 B` | `141.66 TF` | `139.86 TF` | `140.92 TF` | Superseded |
| B1_swar + OptMode=1 (512 B TG LUT + SWAR) | `9,504 B` | `126` | `0 B` | **`153.15 TF`** | **`151.70 TF`** | **`152.04 TF`** | WINNER (Landed) |
| B1_sh01_swar (1024 B TG LUT) | `10,016 B` | — | `0 B` | `150.93 TF` | `149.10 TF` | `149.71 TF` | Rejected |
| B2_bm32 (2048 B, BM=32) | `8,864 B` | `86` | `0 B` | `101.41 TF` | `101.16 TF` | `101.09 TF` | Rejected |
| B2_as32 (2048 B, BM=64) | `8,992 B` | `126` | `0 B` | `144.73 TF` | `142.76 TF` | `143.96 TF` | Rejected |
| B2_ws32 (Ws[64*32]) | `8,992 B` | — | `32 B` | `140.89 TF` | — | — | Rejected (spill) |
| C1 (0xC8643210) | `8,992 B` | — | `0 B` | `137.50 TF` | `135.20 TF` | `136.70 TF` | Rejected |
| C2 (uchar4) | `8,992 B` | — | `0 B` | `118.00 TF` | `116.00 TF` | `117.30 TF` | Rejected |
| C3 (unpack<fp4>) | `8,992 B` | — | `0 B` | `67.50 TF` | `70.90 TF` | `68.20 TF` | Rejected |
| C4_swar (pure ALU, 0 B LUT) | `8,992 B` | `126` | `0 B` | `145.17 TF` | `143.18 TF` | `143.87 TF` | Runner-up |

### 3.2 Zero-DRAM `128t BM=64 SWI8` SwiGLU Mode B Microbench (`M=2048, N=17408, K=5120`, `icb=1`, `cos=1.000000`)

| Variant | Pre-P3.1 (`256t` / `bm32`) | **Phase 3.1 (`128t BM=64 SWI8`)** | Gain | TG SRAM | GPR | Spill |
|---|---:|---:|---:|---:|---:|---:|
| MXFP4 Mode B (`-O3`) | `5.595 ms` (`130.51 TF`) | **`4.814 ms` (`151.67 TF`)** | **`+16.2%` (`+21.2 TF`)** | `9,504 B` | `126` | **`0 B`** |
| MXFP4 Mode B (`-fno-fast-math`) | `5.595 ms` (`130.51 TF`) | **`4.894 ms` (`149.19 TF`)** | **`+14.3%` (`+18.7 TF`)** | `9,504 B` | `126` | **`0 B`** |
| Affine `q4` Mode B | `4.820 ms` (`151.47 TF`) | **`4.437 ms` (`164.54 TF`)** | **`+8.6%` (`+13.1 TF`)** | `10,000 B` | `124` | **`0 B`** |
| Affine `q4_int4b` Mode B | `4.820 ms` (`151.47 TF`) | **`4.608 ms` (`158.45 TF`)** | **`+4.6%` (`+7.0 TF`)** | `10,000 B` | `126` | **`0 B`** |

### 3.3 `Qwen3.8-27B-mxfp4/model.v2.miv` Real-Weight `slow_flag` Distribution Scan

- **Total Trunk `(row, K=64)` Pairs Scanned**: `380,477,440`
- **Fast-Path (`|Δe| <= 3`)**: **`99.999666%`** (`|Δe|=0`: `67.81%`, `1`: `31.95%`, `2`: `0.24%`, `3`: `0.0056%`)
- **Slow-Path Outliers (`|Δe| > 3`)**: only `1,272` rows (`0.000334%`) across `1,252 / 11,893,760` (`0.0105%`) 32-column tiles (`0` in `down_proj` / `o_proj`; `0` all-zero half-blocks in dense 27B; `0x77777777u` zero-block exemption guards pruned/padded/adapter weights cleanly).

## 4. Live M5 Ultra Sublayer GPU Breakdown (`--profile-sublayers`, `W4A8 + Sage2 + MInf`, `ANE=off`)

Both models audited **`497/497` (`100.0%`) fast-path matrices** (`496` W4A8 NAX trunk + `1` quantized `lm_head`, **`0` BF16 fallbacks**).

### 4.1 `Qwen3.8-27B-mxfp4` (`model.v2.miv`) — `T=2048` (`2,330.0 tok/s`, `14.92 GiB`) & `T=16384` (**`2,711.9 tok/s`**, `14.93 GiB`)

| Sublayer Category | `T=2048` GPU ms (%) | `T=2048` Eff. TF | `T=2048` Disp | `T=16384` GPU ms (%) | `T=16384` Eff. TF | `T=16384` Disp |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_swiglu_mode_b` | `311.28` (43.8%) | **`150.12`** | 64 | `2490.44` (42.3%) | **`150.11`** | 512 |
| `linear_mlp_down` | `153.17` (21.5%) | **`152.54`** | 64 | `1227.50` (20.8%) | **`152.27`** | 512 |
| `linear_gdn_qkvz_ab` | `108.12` (15.2%) | **`153.43`** | 192 | `864.58` (14.7%) | **`153.50`** | 1536 |
| `linear_gdn_out` | `41.47` (5.8%) | **`149.14`** | 48 | `331.48` (5.6%) | **`149.26`** | 384 |
| `sage2_minf_attn` | `11.92` (1.7%) | — | 80 | `296.48` (5.0%) | — | 1120 |
| `linear_attn_qkv` | `31.12` (4.4%) | **`154.59`** | 48 | `249.03` (4.2%) | **`154.53`** | 384 |
| `gdn_stage1_3` | `27.65` (3.9%) | — | 192 | `222.20` (3.8%) | — | 1536 |
| `linear_attn_o` | `16.50` (2.3%) | **`124.94`** | 32 | `132.22` (2.2%) | **`124.74`** | 256 |
| `rmsnorm_a8` | `9.82` (1.4%) | — | 129 | `78.68` (1.3%) | — | 1032 |
| `embed_lookup` | `0.14` (0.0%) | — | 1 | `0.55` (0.0%) | — | 8 |
| **Sum Trunk Sublayers** (`whole_trunk` / `ovhd`) | **`711.18`** (`716.13` / `-4.95`) | — | — | **`5893.17`** (`5873.18` / `+19.98`) | — | — |

### 4.2 `Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`) — `T=2048` (**`2,800.0 tok/s`**, `18.89 GiB`) & `T=16384` (**`2,879.0 tok/s`**, `18.89 GiB`)

| Sublayer Category | `T=2048` GPU ms (%) | `T=2048` Eff. TF | `T=2048` Disp | `T=16384` GPU ms (%) | `T=16384` Eff. TF | `T=16384` Disp |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_swiglu_mode_b` | `289.37` (42.6%) | **`161.49`** | 64 | `2314.35` (41.1%) | **`161.53`** | 512 |
| `linear_mlp_down` | `150.54` (22.2%) | **`155.21`** | 64 | `1204.34` (21.4%) | **`155.20`** | 512 |
| `linear_gdn_qkvz_ab` | `101.40` (14.9%) | **`163.61`** | 192 | `811.32` (14.4%) | **`163.58`** | 1536 |
| `linear_gdn_out` | `42.92` (6.3%) | **`144.11`** | 48 | `342.82` (6.1%) | **`144.33`** | 384 |
| `sage2_minf_attn` | `11.90` (1.8%) | — | 80 | `296.58` (5.3%) | — | 1120 |
| `linear_attn_qkv` | `28.82` (4.2%) | **`166.91`** | 48 | `230.64` (4.1%) | **`166.85`** | 384 |
| `gdn_stage1_3` | `27.88` (4.1%) | — | 192 | `221.69` (3.9%) | — | 1536 |
| `linear_attn_o` | `16.15` (2.4%) | **`127.66`** | 32 | `129.12` (2.3%) | **`127.73`** | 256 |
| `rmsnorm_a8` | `9.80` (1.4%) | — | 129 | `78.94` (1.4%) | — | 1032 |
| `embed_lookup` | `0.06` (0.0%) | — | 1 | `0.48` (0.0%) | — | 8 |
| **Sum Trunk Sublayers** (`whole_trunk` / `ovhd`) | **`678.83`** (`683.27` / `-4.44`) | — | — | **`5630.28`** (`5614.69` / `+15.60`) | — | — |

## 5. Full 8-Tier Cold Prefill Matrix (`1K–256K`) — `Qwen3.8-27B-mxfp4` (`model.v2.miv`)

Artifact: `/Users/scoltc/repo/models/mlx-community/Qwen3.8-27B-mxfp4/model.v2.miv` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Config | 1K TTFT (tok/s) | 4K TTFT (tok/s) | 8K TTFT (tok/s) | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs P3 / P2 `M3b` / `v0.03 M0` | 16K / 256K Eff. TF (HW TF) | Peak UMA | NIAH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.03 M0` (Ref BF16+Exact) | `0.62s` (1451) | `2.05s` (1933) | `4.04s` (1998) | `8.30s` (1944) | `18.24s` (1782) | `43.56s` (1499) | `116.59s` (1122) | `353.84s` (740) | `0.32x` / `0.37x` / `1.00x` | `100.8` / `74.2` (`74.2`) | `19.34 GiB` | **24/24** |
| P2 `M1_v2` (W4A8+Exact) | `0.58s` (1549) | `1.93s` (2056) | `3.80s` (2120) | `7.85s` (2071) | `17.16s` (1902) | `40.67s` (1608) | `107.60s` (1217) | `330.68s` (792) | `0.34x` / `0.40x` / `1.07x` | `107.5` / `79.4` (`79.4`) | `19.16 GiB` | **24/24** |
| P2 `M3b_v2` (W4A8+Sage2+MInf) | `0.59s` (1530) | `1.93s` (2055) | `3.76s` (2146) | `7.37s` (2206) | `14.75s` (2213) | `29.99s` (2181) | `61.93s` (2114) | `131.34s` (1995) | `0.87x` / `1.00x` / `2.69x` | `114.5` / `199.9` (`103.6`) | `19.33 GiB` | **24/24** |
| P3 `M0` (BF16+Exact) | `0.64s` (1401) | `2.15s` (1846) | `4.22s` (1909) | `8.68s` (1873) | `18.80s` (1736) | `43.93s` (1489) | `114.15s` (1147) | `343.65s` (762) | `0.33x` / `0.38x` / `1.03x` | `97.8` / `76.6` (`76.6`) | `19.13 GiB` | **24/24** |
| P3 `M1_A8` (W4A8+Exact) | `0.54s` (1651) | `1.68s` (2357) | `3.27s` (2465) | `6.75s` (2409) | `14.93s` (2186) | `36.19s` (1807) | `98.64s` (1328) | `312.65s` (838) | `0.36x` / `0.42x` / `1.13x` | `125.7` / `84.2` (`84.2`) | `19.14 GiB` | **24/24** |
| P3 `M3a_A8` (W4A8+Dense Sage2) | `0.55s` (1639) | `1.69s` (2341) | `3.19s` (2531) | `6.63s` (2453) | `14.32s` (2280) | `33.57s` (1948) | `88.45s` (1480) | `266.06s` (985) | `0.43x` / `0.49x` / `1.33x` | `128.0` / `99.0` (`99.0`) | `19.46 GiB` | **24/24** |
| P3 `M3b_A8` (W4A8+Sage2+MInf) | `0.54s` (1653) | `1.69s` (2344) | `3.25s` (2477) | `6.31s` (2576) | `12.55s` (2600) | `25.65s` (2550) | `53.29s` (2457) | `114.06s` (2297) | `1.00x` / `1.15x` / `3.10x` | `134.5` / `230.9` (`120.0`) | `19.33 GiB` | **24/24** |
| **P3.1 `M1_A8` (W4A8 + Exact)** | `0.55s` (1642) | `1.65s` (2410) | `3.21s` (2513) | **`6.61s`** (2458) | **`14.63s`** (2231) | `35.58s` (1838) | **`97.49s`** (1343) | **`310.44s`** (844) | `0.37x` / `0.42x` / `1.14x` | `127.5` / `84.6` (`84.6`) | `19.14 GiB` | **24/24** |
| **P3.1 `M3a_A8` (W4A8 + Dense Sage2)** | `0.56s` (1591) | `1.67s` (2376) | `3.19s` (2529) | **`6.45s`** (2521) | **`13.92s`** (2345) | `32.80s` (1994) | **`86.87s`** (1507) | **`263.85s`** (993) | `0.43x` / `0.50x` / `1.34x` | `130.8` / `99.5` (`99.5`) | `19.47 GiB` | **24/24** |
| **P3.1 `M3b_A8` (W4A8 + Sage2 + MInf)** | `0.57s` (1558) | `1.66s` (2390) | `3.18s` (2533) | **`6.14s`** (2645) | **`12.02s`** (2715) | `24.89s` (2628) | **`51.68s`** (2534) | **`111.99s`** (2340) | `1.02x` / `1.17x` / `3.16x` | `137.3` / `234.5` (`121.4`) | `19.33 GiB` | **24/24** |
| **P3.1 `M3b_A8_ANE` (`ultra_fp16_swiglu` auto-bypass)** | `0.54s` (1648) | `1.65s` (2398) | `3.14s` (2569) | **`6.13s`** (2654) | **`12.19s`** (2677) | `24.85s` (2632) | **`51.66s`** (2535) | **`110.52s`** (2371) | `1.03x` / `1.19x` / `3.20x` | `137.7` / `237.6` (`123.1`) | `19.35 GiB` | **24/24** |

> Decode throughput remained invariant across all Phase 3.1 prefill modes: **`53.5–53.6 tok/s` (`1K–8K`) → `53.2–53.3 tok/s` (`16K`) → `45.7–45.8 tok/s` (`64K`) → `42.4–42.5 tok/s` (`128K`) → `39.1–39.2 tok/s` (`256K`)**.

### 5.1 `4K` & `16K` C++ Teacher-Forced Logprob Perplexity — `Qwen3.8-27B-mxfp4` (`model.v2.miv`)

| Mode | 4K Mean NLL | 4K PPL | 4K Top-1 | 16K Mean NLL | 16K PPL | `ΔPPL` vs Phase 3 | 16K Top-1 | Peak UMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 `M0` (Ref BF16 + Exact) | `0.73995` | `2.0958` | `83.03%` | `0.37377` | `1.4532` | — | `90.90%` | `15.42 GiB` |
| **P3.1 `M1_A8`** (`W4A8` + Exact) | `0.73036` | **`2.0758`** | `83.17%` | `0.35660` | **`1.4285`** | **`0.0000`** *(bit-exact)* | `91.16%` | `15.49 GiB` |
| **P3.1 `M3a_A8`** (`W4A8` + Dense Sage2) | `0.73475` | **`2.0850`** | `83.05%` | `0.36318` | **`1.4379`** | **`0.0000`** *(bit-exact)* | `91.12%` | `15.55 GiB` |
| **P3.1 `M3b_A8`** (`W4A8` + Sage2 + MInf) | `0.73475` | **`2.0850`** | `83.05%` | `0.38263` | **`1.4661`** | **`0.0000`** *(bit-exact)* | `90.53%` | `15.56 GiB` |

## 6. Full 8-Tier Cold Prefill Matrix (`1K–256K`) — `Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`)

Artifact: `/Users/scoltc/repo/models/Jundot/Qwen3.8-27B-oQ4e-mtp/model.v2.miv` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Config | 1K TTFT (tok/s) | 4K TTFT (tok/s) | 8K TTFT (tok/s) | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs P3 / P2 `M3b` / `v0.03 M0` | 16K / 256K Eff. TF (HW TF) | Peak UMA | NIAH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `v0.03 M0` (Ref BF16+Exact) | `0.51s` (1746) | `2.02s` (1966) | `3.94s` (2047) | `8.21s` (1965) | `17.82s` (1824) | `42.76s` (1526) | `114.31s` (1144) | `347.23s` (754) | `0.31x` / `0.32x` / `1.00x` | `101.9` / `75.6` (`75.6`) | `23.21 GiB` | **24/24** |
| P2 `M1_v2` (W4A8+Exact) | `0.51s` (1766) | `1.67s` (2380) | `3.22s` (2503) | `6.63s` (2451) | `14.69s` (2221) | `35.65s` (1835) | `97.89s` (1338) | `311.45s` (841) | `0.35x` / `0.36x` / `1.11x` | `127.2` / `84.3` (`84.3`) | `23.04 GiB` | **24/24** |
| P2 `M3b_v2` (W4A8+Sage2+MInf) | `0.50s` (1801) | `1.59s` (2489) | `3.18s` (2533) | `6.17s` (2634) | `12.32s` (2649) | `25.09s` (2607) | `52.09s` (2514) | `111.47s` (2350) | `0.97x` / `1.00x` / `3.11x` | `136.7` / `235.6` (`122.1`) | `23.26 GiB` | **24/24** |
| P3 `M0` (BF16+Exact) | `0.65s` (1383) | `2.10s` (1886) | `3.92s` (2056) | `8.44s` (1925) | `18.04s` (1809) | `42.74s` (1530) | `111.78s` (1171) | `338.93s` (773) | `0.32x` / `0.33x` / `1.02x` | `100.5` / `77.7` (`77.7`) | `23.07 GiB` | **24/24** |
| P3 `M1_A8` (W4A8+Exact) | `0.39s` (2277) | `1.66s` (2389) | `2.91s` (2771) | `6.20s` (2623) | `14.17s` (2303) | `34.67s` (1887) | `95.63s` (1369) | `306.76s` (854) | `0.35x` / `0.36x` / `1.13x` | `136.9` / `85.8` (`85.8`) | `23.07 GiB` | **24/24** |
| P3 `M3a_A8` (W4A8+Dense Sage2) | `0.51s` (1746) | `1.63s` (2431) | `2.88s` (2802) | `6.20s` (2621) | `13.40s` (2435) | `31.92s` (2049) | `84.87s` (1543) | `259.77s` (1009) | `0.41x` / `0.43x` / `1.34x` | `136.8` / `101.4` (`101.4`) | `23.39 GiB` | **24/24** |
| P3 `M3b_A8` (W4A8+Sage2+MInf) | `0.53s` (1704) | `1.42s` (2792) | `2.88s` (2803) | `5.90s` (2755) | `11.74s` (2779) | `23.93s` (2734) | `49.87s` (2626) | `107.59s` (2435) | `1.00x` / `1.04x` / `3.23x` | `143.8` / `244.8` (`127.3`) | `23.26 GiB` | **24/24** |
| **P3.1 `M1_A8` (W4A8 + Exact)** | `0.38s` (2342) | `1.64s` (2423) | `2.89s` (2787) | **`6.29s`** (2585) | **`14.13s`** (2310) | `34.57s` (1892) | **`95.46s`** (1372) | **`306.01s`** (856) | `0.35x` / `0.36x` / `1.13x` | `134.2` / `85.8` (`85.8`) | `23.04 GiB` | **24/24** |
| **P3.1 `M3a_A8` (W4A8 + Dense Sage2)** | `0.55s` (1623) | `1.51s` (2630) | `3.07s` (2629) | **`6.22s`** (2611) | **`13.39s`** (2438) | `31.71s` (2063) | **`84.69s`** (1546) | **`260.02s`** (1008) | `0.41x` / `0.43x` / `1.34x` | `135.5` / `101.0` (`101.0`) | `23.39 GiB` | **24/24** |
| **P3.1 `M3b_A8` (W4A8 + Sage2 + MInf)** | `0.47s` (1894) | `1.60s` (2481) | `3.08s` (2622) | **`5.93s`** (2740) | **`11.64s`** (2804) | `23.86s` (2741) | **`49.53s`** (2644) | **`109.47s`** (2393) | `0.98x` / `1.02x` / `3.17x` | `142.2` / `239.9` (`124.5`) | `23.27 GiB` | **24/24** |
| **P3.1 `M3b_A8_ANE` (`ultra_fp16_swiglu` auto-bypass)** | `0.44s` (2039) | `1.55s` (2563) | `3.05s` (2646) | **`5.93s`** (2741) | **`11.53s`** (2832) | `23.91s` (2736) | **`49.73s`** (2633) | **`106.65s`** (2457) | `1.01x` / `1.05x` / `3.26x` | `142.3` / `246.2` (`127.6`) | `23.29 GiB` | **24/24** |

> Decode throughput remained invariant across all Phase 3.1 prefill modes: **`49.1–49.3 tok/s` (`1K–8K`) → `48.9–49.1 tok/s` (`16K`) → `42.2–42.4 tok/s` (`64K`) → `39.4–39.6 tok/s` (`128K`) → `36.5–36.8 tok/s` (`256K`)**. Note: `M3b_A8` `256K` (`109.47s`) ran immediately following the 25-minute sustained `mxfp4` `1K–256K` sweep; `M3b_A8_ANE` (identical GPU-only execution path with `ane=0` auto-bypass) achieved **`106.65s` (`2,456.8 tok/s`)** at `256K` and **`11.53s` (`2,831.7 tok/s`)** at `32K`.

### 6.1 `4K` & `16K` C++ Teacher-Forced Logprob Perplexity — `Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`)

| Mode | 4K Mean NLL | 4K PPL | 4K Top-1 | 16K Mean NLL | 16K PPL | `ΔPPL` vs Phase 3 | 16K Top-1 | Peak UMA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 `M0` (Ref BF16 + Exact) | `1.08587` | `2.9620` | `77.73%` | `1.59595` | `4.9330` | — | `76.35%` | `19.35 GiB` |
| **P3.1 `M1_A8`** (`W4A8` + Exact) | `0.93657` | **`2.5512`** | `79.58%` | `1.15362` | **`3.1696`** | **`0.0000`** *(bit-exact)* | `81.05%` | `19.36 GiB` |
| **P3.1 `M3a_A8`** (`W4A8` + Dense Sage2) | `0.91456` | **`2.4957`** | `79.71%` | `1.12575` | **`3.0825`** | **`0.0000`** *(bit-exact)* | `81.22%` | `19.42 GiB` |
| **P3.1 `M3b_A8`** (`W4A8` + Sage2 + MInf) | `0.91456` | **`2.4957`** | `79.71%` | `1.06117` | **`2.8898`** | **`0.0000`** *(bit-exact)* | `81.63%` | `19.44 GiB` |

## 7. `ane_cal_v3_p31` Fresh Dual-ANE Canary Auto-Guard Verification

Bumping the calibration cache namespace from `ane_cal_v2_p3` to `ane_cal_v3_p31` in `miv/src/ane/ane_dual_offload.mm` forced fresh Layer-0 canary sweeps against Phase 3.1's `150–165 TF` zero-DRAM Mode B kernels:

| Model & Route | Phase 3 L0 `gpu_solo` (ms) | **P3.1 L0 `gpu_solo` (ms)** | P3.1 Best Hybrid (ms) | Ratio (`<=0.96` gate) | `best_alpha` | Decision | `ane_offload_tokens` | 256K GPU vs ANE |
|---|---:|---:|---:|---:|---:|---|---:|---:|
| `mxfp4` `W4A8` (`M3b_A8`) | `7.702` | **`7.372`** (`-4.3%`) | `7.186` | `0.9748x` | `0.09` | **Bypassed** (`calibration_bypassed`) | `0` | `111.99s` → **`110.52s`** *(zero regression)* |
| `oQ4e` `W4A8` (`M3b_A8`) | `7.251` | **`7.171`** (`-1.1%`) | `7.075` | `0.9867x` | `0.08` | **Bypassed** (`calibration_bypassed`) | `0` | `109.47s` → **`106.65s`** *(zero regression)* |

## 8. Release Packaging, Local Full-Stack Gates & Atomic M5 Ultra Deployment

- **Local Verification Gates (Apple M4)**:
  - `ctest --output-on-failure` (`build/p31-release`, Ninja Release, NAX + Python nanobind ON): **`101/101` PASS (`100%`)**
  - `pytest` (`test_m5_sidecar_engine.py`, `test_m5_capability_gate.py`, `omlx-extend/tests/`, `test_admin_model_settings_template.py`): **`128 passed, 4 skipped, 0 failed`**
  - `./omlx-extend/scripts/apply_spi_patch.sh --check-only`: **3-way check passed**
  - `apps/miv-mac` (`swift test`): **`7/7` PASS**
  - `omlx-mac` Xcode `oMLXTests`: **`54/54` PASS** (`ModelSettingsScreenVMTests` `49/49` + `LocalizationSmokeTests` `5/5`)
- **Packaged Artifacts (`dist/`)**:
  - `dist/ninfer-metal-miv-v0.05-p3.1-rc1-macos27-arm64.tar.gz` (`25,590,931 B`, SHA-256 `4712afd5cacf36a264c55c31ac9c24da58fa7b173c2d1b00705ecc293f832957`, clean-Mac system-frameworks-only dylib audit PASS, `17` binaries + `48` `.metallib`s, `provider.json` public `prefill_linear_mode = ["bf16", "w4a8"]`)
  - `dist/omlx-extend/oMLX-extend-v0.8.3-mix7-macos26-27.dmg` (`901,062,825 B`, SHA-256 `3e11b788ec68a6f43a33796977f5c55e609d2a91bd5cf9ce53508327b2418572`, signed `identifier "app.omlx"`, embedded Python + custom kernels smoke OK)
- **Atomic M5 Ultra Deployment (`scoltc@192.168.10.245`)**:
  - Atomically switched `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current -> versions/miv-v0.05-p3.1-rc1` + `provider.json` via POSIX `os.replace`.
  - Post-switch `/health` + `/v1/chat/completions` smoke via `current/bin/ninfer-metal-serve` on `:18090` returned HTTP `200` (`'PONG'`, `realpath` verified to `versions/miv-v0.05-p3.1-rc1/bin/ninfer-metal-serve`).
  - Removed all `/tmp/p31*` scratch directories; verified `versions/miv-v0.04-p2-rc1/` and `versions/miv-v0.05-p3-rc1/` intact, `:8000` `oMLX` healthy (`loaded_count == 0`), and production `:8080` `PID 75288` continuously healthy & untouched (`ELAPSED 02-00:26:20`, `0.0%` CPU, `active_requests == 0`).
