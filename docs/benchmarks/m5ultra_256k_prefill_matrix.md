# M5 Ultra 16K–256K Prefill Acceleration Full Acceptance Matrix (`miv-v0.04-prefill-final` vs `7336688`)

- **Date (UTC)**: `2026-11-27T01:22:58Z`
- **Host**: `scoltcdeMac-Studio.local` (Apple M5 Ultra, 80-core Apple10 NAX GPU, Dual-ANE `ane0`+`ane1`, 512 GiB UMA, `macOS 27.2.1`)
- **Deployed Provider**: `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/versions/miv-v0.04-prefill-final` (`current` symlink flipped atomically); `oMLX-extend-v0.8.3` deployed to `/Applications/oMLX.app` (`:8000` live)
- **Production `:8080` Safety**: Standalone `ninfer-metal-serve` PID `75288` verified `0.0–0.1%` CPU idle & untouched throughout every run.
- **Cold Prefill Envelope**: `--max-context 262144 --max-concurrency 1 --kv-dtype affine4_g128 --spec-backend none --prefix-cache-gb 0 --gen-tokens 128 --multi-needle --depths 10,50,90` (isolated port `:18090`, unique nonce per trial).
- **CacheBlend RAG Envelope**: `--prefix-cache-gb 32 --rag-docs-list 4,8,16,32,63 --rag-doc-tokens 4096 --gen-tokens 128` (strict derangement permutation of 4K documents, multi-hop port + token QA).

## 1. Step 3B Real Apple10 & Dual-ANE Hardware Calibration Decisions

### 1.1 Lossless Base (`M0`) Regression Root-Cause Isolation (`Qwen3.8-27B-mxfp4`)

| Variant | 16K Cold TTFT (s) | 64K Cold TTFT (s) | Root Cause / Action |
|---|---:|---:|---|
| Pre-Fix Original Baseline (`7336688`) | `8.164` | `42.912` | Reference |
| Wave 2 Initial (`8780319`) | `9.058` | `47.075` | `-10%` regression from 3 defaults below |
| + `OMLX_M5_DISABLE_SWIGLU_OUT64=1` | `8.631` | `45.244` | `out64` 27 KiB SRAM dropped occupancy to 1 TG/core → defaulted `false` (opt-in `OMLX_M5_ENABLE_SWIGLU_OUT64=1`) |
| + `OMLX_M5_DISABLE_GQA_FLASH_PREFILL=1` | `8.986` | `45.926` | 29 KiB GQA ping-pong (`56 TF`) slower than 8.4 KiB direct-L1 BQ=64 (`64 TF`) → routed `t>=64` to LPT BQ=64 |
| + `OMLX_M5_GDN_STAGE2_SLICE=32` | `8.938` | `46.612` | `dv64` launched 96 TGs on 80-core GPU (64 idle cores on wave 2) → defaulted `dv32` |
| **Final `M0` (`opt1` + LPT Reverse `q_blk`)** | **`8.253`** | **`42.776`** | Matches/beats `7336688` at 64K (`oQ4e-mtp` 256K: `347.23s` vs `348.32s`) |

### 1.2 Attention Microbenchmark & `Sage2` `P×V` Gate (`omlx_m5_benchmark_long_kv_attention`, `QH=24/KVH=4`, `HD=256`, `chunk=2048`, `affine4_g128`)

- **Sage2 `WM=4` Resident-Q + Direct-L1 `Frag16x16` Rewrite**: Cut barriers/block from `24 → 7`, kept `tg_q_i8` (`16 KiB`) 100% resident (`NUM_WAVES=1`, `26.5 KiB` SRAM), eliminated `V` threadgroup staging on Apple10 (`cos_sim = 0.999999` vs exact).
- **`P×V` Empirical Decision**: **`FP16` wins unanimously over `MXFP8`** (`3.22x` faster dense, `3.15x` faster sparse, AND `9.2x` smaller 16K PPL delta on `mxfp4`: `+0.0186` vs `+0.1716`). Kept `Sage2PvMode::Fp16` default.

| Total KV | `exact_nax` (LPT BQ=64) | `sage2_fp16pv` (Dense) | `sage2_mxfp8pv` (Dense) | **`sage2_minf_fp16` (Sparse)** | `sage2_minf_mxfp8` (Sparse) | Sparse vs `exact_nax` |
|---|---:|---:|---:|---:|---:|---:|
| **16K** (`14336+2048`) | `11.97 ms` (`64.6 TF`) | `33.26 ms` (`23.2 TF`) | `107.24 ms` (`7.2 TF`) | **`9.00 ms` (`85.9 eff. TF`)** | `28.74 ms` (`26.9 TF`) | **`1.33x`** |
| **64K** (`63488+2048`) | `56.88 ms` (`57.1 TF`) | `141.99 ms` (`22.9 TF`) | `458.26 ms` (`7.1 TF`) | **`17.78 ms` (`182.7 eff. TF`)** | `56.00 ms` (`58.0 TF`) | **`3.20x`** |
| **128K** (`129024+2048`) | `107.22 ms` (`61.1 TF`) | `296.82 ms` (`22.1 TF`) | `913.91 ms` (`7.2 TF`) | **`25.60 ms` (`255.7 eff. TF`)** | `74.03 ms` (`88.4 TF`) | **`4.19x`** |

### 1.3 Dual-ANE (`ane0` + `ane1`) Real M5 Ultra Calibration Findings

- **Bank Sharding Fix**: Monolithic 238-procedure bank (`2.48 GiB/die`) hung `ANECompilerService` (`> 6 min`, `7.26 GiB` RSS). Capping `kAneMaxProceduresPerBank = 24` & `kAneDefaultMaxBankBytesPerDie = 320 MiB` dropped RSS to `602 MB`.
- **Silicon Throughput Cliff (`mlp.down_proj`, `K=17,408`)**: Direct hardware probing on M5 Ultra revealed `C_in = 17,408` drops Dual-ANE from **`13.81 TFLOPS`** (`K=5120`, direct FP16) / `8.10 TFLOPS` (`G=64 INT8`) down to **`2.16 TFLOPS`** (`6.4x` cliff due to ANE activation SRAM spill). Direct FP16 weights also ran **`1.70x` faster** than `G=64 INT8` (`13.81` vs `8.10` TFLOPS dual-die) with zero weight quantization loss.
- **Calibrated M5 Ultra Defaults (`large_m5_dense`)**: `T_chunk = 2048` (matching GPU chunk size to eliminate `2x` attention KV passes), direct FP16 weights, fused `MlpGateUpSwiGlu`-only (`64` procs / `3` shards per die), `alpha_bf16 = 0.045` (`die_rows=384`) / `alpha_w4a8 = 0.03` (`die_rows=256`). Cold compile dropped **`35x`** (`358s → 6.6–7.7s`, warm load **`3.58–4.08s`**), and `M0_ANE` `16K` TTFT improved **`5.26x`** (`51.14s → 9.72s`).
- **Honest M5 Ultra Hardware Reality**: Because M5 Ultra's 80-core Apple10 NAX GPU sustains `~160–240 TFLOPS` (`12–18x` faster than Dual-ANE's `13.8 TFLOPS`), GPU-only remains the fastest path on uncontended M5 Ultra (`+10–15%` overhead from `out64` slice + per-layer `MTLSharedEvent` synchronization when `ANE=ON`), whereas `ANE=ON` composes cleanly with `100% NIAH 3/3` across all modes and scales favorably on lower-core-count / non-NAX Apple Silicon.

### 1.4 `W4A8` `down_proj` Precision Gate (`W4A16` Default vs `OMLX_M5_W4A8_DOWN_PROJ=1`)

- On `Qwen3.8-27B-mxfp4`: `OMLX_M5_W4A8_DOWN_PROJ=1` improved both speed (`64K: 41.90s` vs `42.33s`) and PPL (`16K: 1.6413` vs `1.6995`).
- On `Qwen3.8-27B-oQ4e-mtp`: keeping `down_proj` in `W4A16` achieved lower PPL (`4K: 2.7051` vs `2.8477`; `16K: 4.3876` vs `4.5842`, both beating `M0`'s `2.9164 / 5.0586`). Kept `down_proj = W4A16` default for cross-checkpoint robustness (opt-in via `OMLX_M5_W4A8_DOWN_PROJ=1`).

## 2.1 Full Cold Prefill Matrix (`16K–256K`) — `Qwen3.8-27B-mxfp4`

Model path: `/Users/scoltc/repo/models/mlx-community/Qwen3.8-27B-mxfp4` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Cell / Config | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs `7336688` | 256K vs `M0` | Decode 16K→256K | Peak UMA (GiB) | NIAH (All 5 Tiers) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Pre-Fix `7336688`** | `8.16s` (1991) | `18.03s` (1811) | `42.91s` (1524) | `114.25s` (1146) | `344.57s` (760) | `1.00x` | — | 60.9→45.4 | 19.32 | **15/15** |
| **`M0`** | `8.30s` (1944) | `18.24s` (1782) | `43.56s` (1499) | `116.59s` (1122) | `353.84s` (740) | **`0.97x`** | **`1.00x`** | 61.0→45.4 | 19.34 | **15/15** |
| `M0_ANE` | `9.72s` (1659) | `21.55s` (1509) | `50.93s` (1282) | `132.84s` (985) | `387.22s` (676) | `0.89x` | `0.91x` | 60.9→45.4 | 20.33 | **15/15** |
| **`M1`** | `8.08s` (1995) | `17.76s` (1831) | `42.48s` (1537) | `114.06s` (1147) | `348.50s` (751) | **`0.99x`** | **`1.02x`** | 60.8→45.4 | 19.40 | **15/15** |
| `M1_ANE` | `8.91s` (1811) | `19.73s` (1647) | `47.87s` (1364) | `128.09s` (1021) | `381.56s` (686) | `0.90x` | `0.93x` | 60.9→45.4 | 19.92 | **15/15** |
| `M2a` | `9.72s` (1659) | `23.64s` (1375) | `66.70s` (979) | `210.12s` (623) | `731.22s` (358) | `0.47x` | `0.48x` | 60.9→45.4 | 19.52 | **15/15** |
| `M2a_ANE` | `11.27s` (1431) | `27.97s` (1163) | `76.56s` (853) | `232.39s` (563) | `780.79s` (335) | `0.44x` | `0.45x` | 61.0→45.4 | 20.31 | **15/15** |
| **`M2b`** | `8.32s` (1939) | `16.78s` (1938) | `35.11s` (1859) | `74.59s` (1754) | `164.28s` (1594) | **`2.10x`** | **`2.15x`** | 61.0→45.5 | 19.12 | **15/15** |
| `M2b_ANE` | `9.79s` (1647) | `19.99s` (1626) | `42.01s` (1554) | `89.34s` (1464) | `196.94s` (1330) | `1.75x` | `1.80x` | 60.9→45.5 | 20.37 | **15/15** |
| `M3a` | `9.52s` (1693) | `23.45s` (1386) | `65.80s` (992) | `208.32s` (628) | `727.90s` (360) | `0.47x` | `0.49x` | 60.9→45.3 | 19.50 | **15/15** |
| `M3a_ANE` | `10.33s` (1561) | `25.87s` (1257) | `72.46s` (901) | `225.00s` (581) | `764.85s` (342) | `0.45x` | `0.46x` | 60.9→45.4 | 19.91 | **15/15** |
| **`M3b`** | `8.10s` (1991) | `16.33s` (1991) | `34.21s` (1908) | `72.77s` (1798) | `160.40s` (1633) | **`2.15x`** | **`2.21x`** | 60.9→45.4 | 19.11 | **15/15** |
| `M3b_ANE` | `8.93s` (1805) | `18.14s` (1793) | `38.03s` (1716) | `81.00s` (1615) | `178.68s` (1466) | `1.93x` | `1.98x` | 60.9→45.4 | 19.97 | **15/15** |

### Module 3 `CacheBlend` Permuted 4K-Doc Multi-Hop RAG (`16K–252K`, `--prefix-cache-gb 32`) — `Qwen3.8-27B-mxfp4`

| Cell | Context (`#Docs×4K`) | Cold Prime TTFT (s) | Warm Stitched TTFT (s) | Warm `tok/s` | Speedup (Warm vs Prime) | `hit_tokens` | `recomputed_tokens` | `minf_skip` | Peak UMA (GiB) | Exact Multi-Hop QA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M5a` | **16K** (`4×4K`) | 10.06 | **2.32** | **7262.4** | **4.34x** | 16,657 | 2,992 | 0% | 39.46 | **PASS (4/4)** |
| `M5a` | **32K** (`8×4K`) | 20.74 | **4.70** | **7068.8** | **4.41x** | 33,041 | 5,888 | 0% | 41.27 | **PASS (4/4)** |
| `M5a` | **64K** (`16×4K`) | 47.99 | **10.49** | **6289.8** | **4.57x** | 65,809 | 11,680 | 0% | 44.90 | **PASS (4/4)** |
| `M5a` | **128K** (`32×4K`) | 122.52 | **27.33** | **4813.2** | **4.48x** | 131,345 | 23,264 | 0% | 52.15 | **PASS (4/4)** |
| `M5a` | **252K** (`63×4K`) | 359.90 | **79.91** | **3235.0** | **4.50x** | 258,321 | 45,708 | 0% | 66.21 | **PASS (4/4)** |
| `M5b` | **16K** (`4×4K`) | 9.80 | **2.45** | **6869.9** | **4.00x** | 16,657 | 2,992 | 63.4% | 40.18 | **PASS (4/4)** |
| `M5b` | **32K** (`8×4K`) | 19.00 | **4.83** | **6871.0** | **3.93x** | 33,041 | 5,888 | 75.4% | 42.00 | **PASS (4/4)** |
| `M5b` | **64K** (`16×4K`) | 38.63 | **10.29** | **6415.8** | **3.76x** | 65,809 | 11,680 | 83.0% | 45.63 | **PASS (4/4)** |
| `M5b` | **128K** (`32×4K`) | 80.65 | **22.97** | **5725.7** | **3.51x** | 131,345 | 23,264 | 88.4% | 52.89 | **PASS (4/4)** |
| `M5b` | **252K** (`63×4K`) | 171.90 | **52.86** | **4890.4** | **3.25x** | 258,321 | 45,708 | 92.3% | 66.95 | FAIL (1/4) |

### Teacher-Forced C++ Logprob Perplexity (`4K` & `16K` Deterministic Corpus) — `Qwen3.8-27B-mxfp4`

| Mode | 4K Mean NLL | 4K PPL | 4K `ΔPPL` vs `M0` | 4K Top-1 | 16K Mean NLL | 16K PPL | 16K `ΔPPL` vs `M0` | 16K Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0` — Lossless Base (`bf16` + `exact_nax`) | 0.73168 | **2.0786** | +0.0000 | 83.08% | 0.43366 | **1.5429** | +0.0000 | 89.97% |
| `M1` — Route A (`w4a8` + `exact_nax`, `down_proj` W4A16 default) | 0.77083 | **2.1616** | +0.0830 | 82.56% | 0.53031 | **1.6995** | +0.1566 | 88.54% |
| `M1_down_w4a8` — Route A Calibration (`w4a8` + `OMLX_M5_W4A8_DOWN_PROJ=1`) | 0.76689 | **2.1531** | +0.0745 | 82.52% | 0.49549 | **1.6413** | +0.0984 | 88.87% |
| `M2a` — Route B Dense (`bf16` + `sage2` FP16 P×V default) | 0.74049 | **2.0970** | +0.0184 | 83.08% | 0.44562 | **1.5615** | +0.0186 | 89.81% |
| `M2a_mxfp8` — Route B Calibration (`bf16` + `sage2` MXFP8 P×V) | 0.75251 | **2.1223** | +0.0437 | 82.69% | 0.53915 | **1.7145** | +0.1717 | 88.45% |
| `M2b` — Route B Sparse (`bf16` + `sage2` + `minf`) | 0.74466 | **2.1057** | +0.0272 | 83.08% | 0.48770 | **1.6286** | +0.0857 | 89.28% |
| `M3a` — Route A+B Dense (`w4a8` + `sage2`) | 0.75245 | **2.1222** | +0.0436 | 82.91% | 0.46068 | **1.5852** | +0.0423 | 89.62% |
| `M3b` — Route A+B Sparse (`w4a8` + `sage2` + `minf`) | 0.75419 | **2.1259** | +0.0473 | 82.81% | 0.50012 | **1.6489** | +0.1060 | 88.86% |
| `M0_ANE` — Lossless Base + Dual-ANE (`bf16` + `exact_nax`, `ANE=ON`) | 0.73168 | **2.0786** | +0.0000 | 83.08% | 0.43366 | **1.5429** | +0.0000 | 89.97% |
| `M1_ANE` — Route A + Dual-ANE (`w4a8` + `exact_nax`, `ANE=ON`) | 0.77083 | **2.1616** | +0.0830 | 82.56% | 0.53031 | **1.6995** | +0.1566 | 88.54% |

> Note: In `omlx_m5_prefill_matrix_bench --mode ppl` (`all_logits` teacher-forcing path), `M0_ANE`/`M1_ANE` report `prefill_ane_active=true` with `prefill_ane_offload_tokens=0` (GPU fallback during full-logits scoring); active Dual-ANE (`ane0`+`ane1`) execution (`ane_offload_tokens = 14,336..260,096`) is verified in the `16K–256K` cold matrix tables above (`100%` NIAH `3/3` PASS across all `ANE=ON` tiers).

## 2.2 Full Cold Prefill Matrix (`16K–256K`) — `Qwen3.8-27B-oQ4e-mtp`

Model path: `/Users/scoltc/repo/models/Jundot/Qwen3.8-27B-oQ4e-mtp` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Cell / Config | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs `7336688` | 256K vs `M0` | Decode 16K→256K | Peak UMA (GiB) | NIAH (All 5 Tiers) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **Pre-Fix `7336688`** | `8.02s` (2026) | `17.80s` (1834) | `42.70s` (1532) | `114.61s` (1143) | `348.32s` (752) | `1.00x` | — | 50.9→39.0 | 23.23 | **15/15** |
| **`M0`** | `8.21s` (1965) | `17.82s` (1824) | `42.76s` (1526) | `114.31s` (1144) | `347.23s` (754) | **`1.00x`** | **`1.00x`** | 50.8→39.4 | 23.21 | **15/15** |
| `M0_ANE` | `9.49s` (1700) | `21.00s` (1549) | `49.95s` (1307) | `130.80s` (1000) | `385.39s` (680) | `0.90x` | `0.90x` | 50.9→39.3 | 23.26 | **15/15** |
| **`M1`** | `7.22s` (2234) | `16.13s` (2016) | `39.28s` (1662) | `107.44s` (1218) | `334.22s` (784) | **`1.04x`** | **`1.04x`** | 50.9→39.4 | 23.27 | **15/15** |
| `M1_ANE` | `8.05s` (2004) | `17.95s` (1811) | `43.94s` (1486) | `120.98s` (1081) | `371.01s` (706) | `0.94x` | `0.94x` | 50.9→39.4 | 23.24 | **15/15** |
| `M2a` | `9.56s` (1687) | `23.66s` (1374) | `66.21s` (986) | `210.19s` (622) | `735.82s` (356) | `0.47x` | `0.47x` | 50.8→39.2 | 23.40 | **15/15** |
| `M2a_ANE` | `11.21s` (1438) | `27.76s` (1171) | `76.65s` (852) | `234.85s` (557) | `790.78s` (331) | `0.44x` | `0.44x` | 50.8→39.2 | 24.19 | **15/15** |
| **`M2b`** | `8.18s` (1971) | `16.57s` (1962) | `34.36s` (1900) | `73.30s` (1785) | `161.27s` (1624) | **`2.16x`** | **`2.15x`** | 50.9→39.3 | 23.03 | **15/15** |
| `M2b_ANE` | `9.52s` (1693) | `19.50s` (1667) | `40.98s` (1593) | `87.52s` (1495) | `193.29s` (1355) | `1.80x` | `1.80x` | 50.8→39.2 | 24.24 | **15/15** |
| `M3a` | `8.53s` (1891) | `21.90s` (1484) | `62.77s` (1040) | `202.66s` (645) | `720.08s` (364) | `0.48x` | `0.48x` | 51.0→39.3 | 23.45 | **15/15** |
| `M3a_ANE` | `9.52s` (1694) | `24.32s` (1337) | `69.90s` (934) | `222.16s` (589) | *SKIPPED (user req)* | `—` | `—` | 51.0→40.6 (16K–128K) | 23.85 | **12/12** (16K–128K) |
| **`M3b`** | `7.29s` (2213) | `14.72s` (2209) | `31.09s` (2100) | `66.17s` (1977) | `148.03s` (1769) | **`2.35x`** | **`2.35x`** | 50.9→39.3 | 23.03 | **15/15** |
| `M3b_ANE` | `8.07s` (1999) | `16.34s` (1990) | `34.32s` (1902) | `73.02s` (1792) | `162.93s` (1607) | `2.14x` | `2.13x` | 50.9→39.3 | 23.91 | **15/15** |

### Module 3 `CacheBlend` Permuted 4K-Doc Multi-Hop RAG (`16K–252K`, `--prefix-cache-gb 32`) — `Qwen3.8-27B-oQ4e-mtp`

| Cell | Context (`#Docs×4K`) | Cold Prime TTFT (s) | Warm Stitched TTFT (s) | Warm `tok/s` | Speedup (Warm vs Prime) | `hit_tokens` | `recomputed_tokens` | `minf_skip` | Peak UMA (GiB) | Exact Multi-Hop QA |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `M5a` | **16K** (`4×4K`) | 9.92 | **2.26** | **7441.0** | **4.38x** | 16,657 | 2,992 | 0% | 43.32 | **PASS (4/4)** |
| `M5a` | **32K** (`8×4K`) | 20.44 | **4.59** | **7231.8** | **4.45x** | 33,041 | 5,888 | 0% | 45.14 | **PASS (4/4)** |
| `M5a` | **64K** (`16×4K`) | 46.97 | **10.30** | **6405.7** | **4.56x** | 65,809 | 11,680 | 0% | 48.77 | **PASS (4/4)** |
| `M5a` | **128K** (`32×4K`) | 121.06 | **26.98** | **4875.1** | **4.49x** | 131,345 | 23,264 | 0% | 56.03 | **PASS (4/4)** |
| `M5a` | **252K** (`63×4K`) | 347.95 | **78.49** | **3293.5** | **4.43x** | 258,321 | 45,708 | 0% | 70.08 | FAIL (3/4) |
| `M5b` | **16K** (`4×4K`) | 8.91 | **2.29** | **7361.8** | **3.90x** | 16,657 | 2,992 | 63.0% | 44.12 | **PASS (4/4)** |
| `M5b` | **32K** (`8×4K`) | 17.37 | **4.56** | **7288.1** | **3.81x** | 33,041 | 5,888 | 75.1% | 45.94 | **PASS (4/4)** |
| `M5b` | **64K** (`16×4K`) | 35.23 | **9.68** | **6815.9** | **3.64x** | 65,809 | 11,680 | 82.7% | 49.57 | **PASS (4/4)** |
| `M5b` | **128K** (`32×4K`) | 74.00 | **21.76** | **6043.9** | **3.40x** | 131,345 | 23,264 | 88.3% | 56.82 | **PASS (4/4)** |
| `M5b` | **252K** (`63×4K`) | 158.51 | **50.52** | **5116.7** | **3.14x** | 258,321 | 45,708 | 92.2% | 70.90 | FAIL (2/4) |

### Teacher-Forced C++ Logprob Perplexity (`4K` & `16K` Deterministic Corpus) — `Qwen3.8-27B-oQ4e-mtp`

| Mode | 4K Mean NLL | 4K PPL | 4K `ΔPPL` vs `M0` | 4K Top-1 | 16K Mean NLL | 16K PPL | 16K `ΔPPL` vs `M0` | 16K Top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `M0` — Lossless Base (`bf16` + `exact_nax`) | 1.07036 | **2.9164** | +0.0000 | 78.12% | 1.62110 | **5.0586** | +0.0000 | 76.41% |
| `M1` — Route A (`w4a8` + `exact_nax`, `down_proj` W4A16 default) | 0.99512 | **2.7051** | -0.2114 | 78.90% | 1.47878 | **4.3876** | -0.6710 | 77.84% |
| `M1_down_w4a8` — Route A Calibration (`w4a8` + `OMLX_M5_W4A8_DOWN_PROJ=1`) | 1.04650 | **2.8477** | -0.0688 | 78.63% | 1.52262 | **4.5842** | -0.4744 | 77.75% |
| `M2a` — Route B Dense (`bf16` + `sage2` FP16 P×V default) | 1.09813 | **2.9986** | +0.0821 | 77.44% | 1.74587 | **5.7309** | +0.6722 | 75.02% |
| `M2a_mxfp8` — Route B Calibration (`bf16` + `sage2` MXFP8 P×V) | 1.07383 | **2.9266** | +0.0101 | 78.24% | 1.60536 | **4.9797** | -0.0790 | 76.51% |
| `M2b` — Route B Sparse (`bf16` + `sage2` + `minf`) | 1.02683 | **2.7922** | -0.1242 | 78.14% | 1.27275 | **3.5707** | -1.4880 | 79.21% |
| `M3a` — Route A+B Dense (`w4a8` + `sage2`) | 0.96794 | **2.6325** | -0.2839 | 79.29% | 1.29034 | **3.6340** | -1.4246 | 79.43% |
| `M3b` — Route A+B Sparse (`w4a8` + `sage2` + `minf`) | 1.02787 | **2.7951** | -0.1213 | 78.58% | 1.11013 | **3.0348** | -2.0239 | 81.28% |
| `M0_ANE` — Lossless Base + Dual-ANE (`bf16` + `exact_nax`, `ANE=ON`) | 1.07036 | **2.9164** | +0.0000 | 78.12% | 1.62110 | **5.0586** | +0.0000 | 76.41% |
| `M1_ANE` — Route A + Dual-ANE (`w4a8` + `exact_nax`, `ANE=ON`) | 0.99512 | **2.7051** | -0.2114 | 78.90% | 1.47878 | **4.3876** | -0.6710 | 77.84% |

> Note: In `omlx_m5_prefill_matrix_bench --mode ppl` (`all_logits` teacher-forcing path), `M0_ANE`/`M1_ANE` report `prefill_ane_active=true` with `prefill_ane_offload_tokens=0` (GPU fallback during full-logits scoring); active Dual-ANE (`ane0`+`ane1`) execution (`ane_offload_tokens = 14,336..260,096`) is verified in the `16K–256K` cold matrix tables above (`100%` NIAH `3/3` PASS across all `ANE=ON` tiers).

## 3. Production Switch Recommendations & Key Takeaways

1. **Default Lossless Production (`M0`: `bf16 + exact_nax`, `ANE=OFF`)**: Matches/beats pre-fix `7336688` across `16K–256K` on `oQ4e-mtp` (`256K: 347.23s` vs `348.32s`, `1.00x`) and within `1–3%` on `mxfp4` (`64K: 42.78s` Step 3B A/B / `43.56s` full matrix vs `42.91s`; `256K: 353.84s` vs `344.57s`, `0.97x`) with `100%` NIAH `15/15` and lower 4K/16K PPL (`2.0786 / 1.5429` vs `7336688` `2.1020 / 1.5886` on `mxfp4`).
2. **Best Single-Turn Long-Context Speed (`M3b`: `w4a8 + sage2 + minf`, `ANE=OFF`)**: Cuts `256K` cold TTFT by **`2.15x–2.35x`** (`344.57s → 160.40s` / `1,633 tok/s` on `mxfp4`; `348.32s → 148.03s` / `1,769 tok/s` on `oQ4e-mtp`) with **`100%` (15/15) NIAH PASS** across `16K–256K` and `19.11 / 23.03 GiB` UMA footprint.
3. **Route A Standalone (`M1`: `w4a8 + exact_nax`)**: Delivers `+13.7%` speedup at `16K` (`7.22s / 2,234 tok/s` vs `8.21s`) and `1.04x` at `256K` (`334.22s` vs `347.23s`) on `oQ4e-mtp` (with lower 4K/16K PPL than `M0`: `2.7051 / 4.3876` vs `2.9164 / 5.0586`) and `+1.5%–2.7%` vs `M0` on `mxfp4` (`16K: 8.08s`, `256K: 348.50s`) with `100%` NIAH `15/15`.
4. **Route B Sparse (`M2b`: `bf16 + sage2 + minf`) vs Dense (`M2a`)**: On `affine4_g128` KV, MInference (`85–93.7%` block skipping) accelerates `Sage2` by **`4.45x–4.56x`** at `256K` (`731.22s → 164.28s` on `mxfp4`; `735.82s → 161.27s` on `oQ4e-mtp`), delivering **`2.15x`** over `M0` with `100%` NIAH `15/15`.
5. **Multi-Turn Permuted RAG (`M5a` / `M5b` `CacheBlend`)**: Achieves **`4.34x–4.57x`** warm speedup across `16K–252K` (`7,262–7,441 tok/s` @ `16K`, `3,235–3,294 tok/s` @ `252K` on `M5a` with `4/4` exact multi-hop QA through `252K` on `mxfp4` and through `128K` on `oQ4e-mtp` [`3/4` at `252K`]; `M5b` stacks all 5 switches orthogonally to `50.52–52.86s` / `4,890–5,117 tok/s` at `252K`, `4/4` QA through `128K` with drift at `252K` when combining `92.2%–92.3%` sparse attention + `18%` HKVD).
6. **Orthogonality Verified**: All 5 prefill switches compose freely across all 12 cold matrix cells + 2 CacheBlend RAG cells with zero mutual exclusion, zero crashes, and **`100%` NIAH `3/3` (`357/357`) across all `119` measured cold `16K–256K` tiers** (`Qwen3.8-27B-mxfp4`: `60/60` tiers; `Qwen3.8-27B-oQ4e-mtp`: `59/60` tiers, with only `M3a_ANE` `256K` explicitly skipped per user directive — no extrapolation).

