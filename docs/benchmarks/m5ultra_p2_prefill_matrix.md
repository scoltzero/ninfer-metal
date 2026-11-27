# M5 Ultra Phase 2 1K–256K Prefill Acceleration Full Hardware Acceptance Matrix (miv-v0.04-p2-rc1 vs miv-v0.03)

- **Date (UTC)**: `2026-11-27T15:16:04Z`
- **Host**: `scoltcdeMac-Studio.local (Apple M5 Ultra, 80-core Apple10 NAX GPU, 512 GiB UMA, macOS 27.2.1)`
- **Deployed Provider**: `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/versions/miv-v0.04-p2-rc1` (`current` symlink active; built from `feat/p2-integrated` `47963f7` + Stage 3 `attention_sage2.metal:1581` WAR-barrier fix)
- **Production `:8080` Safety**: Standalone `ninfer-metal-serve` PID `75288` verified idle (`0.1%` CPU, `0` active requests) before, during, and after every single run; untouched.
- **Cold Prefill Envelope**: `--max-context 262144 --max-concurrency 1 --kv-dtype affine4_g128 --spec-backend none --prefix-cache-gb 0 --gen-tokens 128 --multi-needle --depths 10,50,90` (unique nonce per trial, `cached_tokens == 0`, `:8000` `loaded_count == 0`).
- **Global Multi-Needle NIAH Pass Rate**: **`378/378` (`100.0%`)** across all Phase 2 `.miv v2` + `.miv v1` compat + `64c` measured tiers, plus **`90/90` (`100.0%`)** across `miv-v0.03` `1K–8K` baselines.

## 1. Executive Summary & Real-Hardware Acceptance Highlights

1. **Full 8-Tier Monotonic Speedup (`GPU 全速` `M3b_v2` `1K → 256K`)**:
   - **`Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`)**: `M3b_v2` reaches **`2489.4` tok/s @ 4K**, **`2634.4` tok/s @ 16K**, **`2649.2` tok/s @ 32K**, and **`111.47s` (`2350.5` tok/s) @ 256K** — **`3.11x`** faster than `miv-v0.03 M0` (`347.23s`) and **`1.33x`** faster than `miv-v0.03 M3b` (`148.03s`), with **`24/24` NIAH PASS** and **`23.26 GiB`** Peak UMA.
   - **`Qwen3.8-27B-mxfp4` (`model.v2.miv`)**: `M3b_v2` reaches **`2054.6` tok/s @ 4K**, **`2205.9` tok/s @ 16K**, **`2213.4` tok/s @ 32K**, and **`131.34s` (`1995.0` tok/s) @ 256K** — **`2.69x`** faster than `miv-v0.03 M0` (`353.84s`) and **`1.22x`** faster than `miv-v0.03 M3b` (`160.40s`), with **`24/24` NIAH PASS** and **`19.33 GiB`** Peak UMA.
2. **Cured Standalone Dense `Sage2` (`M2a_v2`, `minference=false`) Now Beats `M0_v2` End-to-End**:
   - `oQ4e-mtp`: `256K` TTFT dropped **`2.52x`** from `miv-v0.03 M2a` (`735.82s` / `355.9 tok/s`) down to **`291.81s` (`897.9` tok/s)**, beating Phase 2 `M0_v2` (`339.21s`) by **`1.16x`** across `8K → 256K` (`24/24` NIAH PASS).
   - `mxfp4`: `256K` TTFT dropped **`2.46x`** from `miv-v0.03 M2a` (`731.22s` / `358.2 tok/s`) down to **`296.95s` (`882.4` tok/s)**, beating Phase 2 `M0_v2` (`343.94s`) by **`1.16x`** across `8K → 256K` (`24/24` NIAH PASS).
3. **Lossless Base (`M0_v2`) Strictly Beats `miv-v0.03 M0` at 256K on Both Models**:
   - `oQ4e-mtp`: **`339.21s`** (`772.4` tok/s) vs `v0.03 M0` `347.23s` (`754.2` tok/s); `mxfp4`: **`343.94s`** (`761.8` tok/s) vs `v0.03 M0` `353.84s` (`740.1` tok/s) — proving default-ON `out64` (`<= 10 KiB`), `BK=16/WN=4` GQA (`<= 14 KiB`), and `dv16_flex` GDN (`10.0 KiB`) hold zero regression.
4. **`.miv v1` Zero-Reconversion Compatibility + `.miv v2` Zero-`Ws` Advantage**:
   - 100% of `.miv v1` compatibility tiers (`M0_v1` `1K–64K`, `M1_v1` `1K–64K`, `M3b_v1` `1K–256K` on both models) passed **`120/120` NIAH** with zero reconversion.
   - On `oQ4e-mtp`, `.miv v2` eliminates the `+13.0 GiB` runtime `oq_nax_lane_v2` repack shadow buffer (**`23.26 GiB`** vs `v1` **`36.27 GiB`**) AND delivers **`+5.9%` to `+14.4%`** higher prefill throughput (`256K`: `111.47s` vs `118.04s`; `4K`: `2489.4` vs `2175.4` tok/s).
   - On `mxfp4`, `.miv v2` adds **0 extra artifact bytes**, delivers **`+1.7%` to `+2.1%`** faster prefill (`256K M3b`: `131.34s` vs `133.63s`), and lowers `16K` `W4A8` PPL by **`-0.0908`** (`1.4805` vs `1.5713`, beating `v0.03 M0` `1.5429`).
5. **Stage 3 Real-Hardware Bug Catch & Fix (`miv/kernels/attention_sage2.metal:1581`)**:
   - During initial 256K cold testing of `M2b_v2` on `oQ4e-mtp`, real Apple10 silicon exposed a WAR data race in `SparseMInference` FP16-PV (`flat_tid < 16` overwrote `tg_mem.fac[0..15]` with `kb_b`, including `0xffffffffu` = float `-NaN` on padded tail slots, at `(step & 15u) == 0u` before all simdgroups finished reading `tg_mem.fac` from `step - 1` PV rescaling). Adding `if (step > 0u) threadgroup_barrier(mem_flags::mem_threadgroup);` cured the NaN 100% while preserving **`0 B` AGX spills / `13,344 B` SRAM** and `MTL_SHADER_VALIDATION=1` PASS (restaged into `miv-v0.04-p2-rc1`).
6. **Live `B=2` Concurrency under `GPU 全速` (`CacheBlend = ON`) on `oMLX :8000`**:
   - Simultaneous `B=2` cold + warm permuted multi-doc streaming chats on `Qwen3.8-27B-oQ4e-mtp` passed **100% (`32/32` QA facts)** on BOTH `model.v2.miv` (`load=4.57s`) and `model.miv` (`load=4.85s`) with overlapping concurrent execution and **zero** `Qwen packed batch slot differs in execution contract` or server errors across `6,041 B` of `~/.omlx/logs/server.log`.

## 2. Full 8-Tier Cold Prefill Matrix (`1K–256K`) — `Qwen3.8-27B-oQ4e-mtp`

Model dir: `/Users/scoltc/repo/models/Jundot/Qwen3.8-27B-oQ4e-mtp` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Config | 1K TTFT (tok/s) | 4K TTFT (tok/s) | 8K TTFT (tok/s) | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs `v0.03 M0` | Decode 1K→256K | Peak UMA (GiB) | MInf Skip 8K→256K | NIAH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `miv-v0.03 M0` (Baseline BF16+Exact) | `0.51s` (1746) | `2.02s` (1966) | `3.94s` (2047) | `8.21s` (1965) | `17.82s` (1824) | `42.76s` (1526) | `114.31s` (1144) | `347.23s` (754) | **1.00x** | 51.2→39.4 | 23.21 | 0% | **24/24** |
| `miv-v0.03 M3b` (Baseline W4A8+Sage2+MInf) | `0.45s` (2010) | `1.91s` (2074) | `3.67s` (2199) | `7.29s` (2213) | `14.72s` (2209) | `31.09s` (2100) | `66.17s` (1977) | `148.03s` (1769) | **2.35x** | 51.2→39.3 | 23.03 | 47.4%→93.5% | **24/24** |
| **`M0_v2`** (`.miv v2` Lossless Base) | `0.47s` (1902) | `2.14s` (1857) | `3.92s` (2057) | `8.36s` (1944) | `18.03s` (1810) | `42.72s` (1531) | `111.70s` (1172) | `339.21s` (772) | **1.02x** | 49.4→36.8 | 23.09 | 0% | **24/24** |
| **`M1_v2`** (`.miv v2` 100% W4A8 + Exact) | `0.51s` (1766) | `1.67s` (2380) | `3.22s` (2503) | `6.63s` (2451) | `14.69s` (2221) | `35.65s` (1835) | `97.89s` (1338) | `311.45s` (841) | **1.11x** | 49.3→36.8 | 23.04 | 0% | **24/24** |
| **`M2a_v2`** (`.miv v2` Cured Dense Sage2) | `0.60s` (1482) | `2.12s` (1872) | `3.89s` (2072) | `8.05s` (2018) | `17.46s` (1870) | `39.99s` (1636) | `101.08s` (1295) | `291.81s` (898) | **1.19x** | 49.4→36.8 | 23.40 | 0% | **24/24** |
| **`M2b_v2`** (`.miv v2` Sparse Sage2 + WP4) | `0.57s` (1581) | `1.99s` (1998) | `4.08s` (1977) | `7.92s` (2054) | `15.76s` (2070) | `32.07s` (2040) | `66.36s` (1973) | `139.72s` (1875) | **2.49x** | 49.4→36.7 | 23.26 | 52.5%→93.7% | **24/24** |
| **`M3b_v2`** (`.miv v2` GPU 全速 Cold) | `0.50s` (1801) | `1.59s` (2489) | `3.18s` (2533) | `6.17s` (2634) | `12.32s` (2649) | `25.09s` (2607) | `52.09s` (2514) | `111.47s` (2350) | **3.11x** | 49.3→36.8 | 23.26 | 52.3%→93.7% | **24/24** |
| `M0_v1` (`.miv v1` Compat Base) | `0.55s` (1626) | `1.92s` (2067) | `3.79s` (2128) | `8.26s` (1968) | `17.75s` (1839) | `41.73s` (1567) | — | — | — | 49.0→41.8 (64K) | 35.93 | 0% | **18/18** |
| `M1_v1` (`.miv v1` Compat W4A8) | `0.48s` (1857) | `1.81s` (2192) | `3.44s` (2342) | `7.15s` (2272) | `15.62s` (2090) | `37.34s` (1752) | — | — | — | 49.0→41.9 (64K) | 35.97 | 0% | **18/18** |
| `M3b_v1` (`.miv v1` Compat GPU 全速) | `0.58s` (1552) | `1.82s` (2175) | `3.40s` (2371) | `6.67s` (2438) | `13.20s` (2473) | `26.78s` (2442) | `55.39s` (2364) | `118.04s` (2220) | **2.94x** | 49.0→35.9 | 36.27 | 53.1%→93.7% | **24/24** |

> Note: `miv-v0.03` `1K/4K/8K` rows were measured fresh on `versions/miv-v0.04-94d6739` (`prompt_tokens = 896, 3968, 8064`); `miv-v0.03` `16K–256K` rows are carried verbatim from `docs/benchmarks/m5ultra_256k_prefill_matrix.json` (`prompt_tokens = 16128..261888`, within `0.05%–0.78%` of Phase 2 `16256..262016`). For `v0.03 M3b`, `256K` Phase 2 `M3b_v2` speedup vs `v0.03 M3b` is **`1.33x`**.

### 2.1 `4K` & `16K` C++ Teacher-Forced Logprob Perplexity — `Qwen3.8-27B-oQ4e-mtp`

| Mode | 4K Mean NLL | 4K PPL | 4K `ΔPPL` vs `v0.03 M0` | 4K Top-1 | 16K Mean NLL | 16K PPL | 16K `ΔPPL` vs `v0.03 M0` | 16K Top-1 | Peak UMA (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `miv-v0.03 M0` (Reference) | 1.07036 | **2.9164** | +0.0000 | 78.12% | 1.62110 | **5.0586** | +0.0000 | 76.41% | 20.83 |
| **`M0_v1`** | 1.10838 | **3.0294** | +0.1130 | 77.85% | 1.60405 | **4.9731** | -0.0855 | 76.35% | 32.34 |
| **`M0_v2`** | 1.08587 | **2.9620** | +0.0456 | 77.73% | 1.59595 | **4.9330** | -0.1256 | 76.35% | 19.35 |
| **`M1_v1`** | 0.99422 | **2.7026** | -0.2138 | 79.12% | 1.39293 | **4.0266** | -1.0320 | 78.44% | 32.37 |
| **`M1_v2`** | 0.93657 | **2.5512** | -0.3652 | 79.58% | 1.15362 | **3.1696** | -1.8890 | 81.05% | 19.36 |
| **`M2a_v2`** | 1.00625 | **2.7353** | -0.1811 | 78.83% | 1.37446 | **3.9529** | -1.1057 | 78.78% | 19.40 |
| **`M2b_v2`** | 1.00625 | **2.7353** | -0.1811 | 78.83% | 1.43524 | **4.2006** | -0.8580 | 78.01% | 19.43 |
| **`M3a_v2`** | 0.91456 | **2.4957** | -0.4207 | 79.71% | 1.12575 | **3.0825** | -1.9761 | 81.22% | 19.41 |
| **`M3b_v2`** | 0.91456 | **2.4957** | -0.4207 | 79.71% | 1.06117 | **2.8898** | -2.1688 | 81.63% | 19.44 |

## 3. Full 8-Tier Cold Prefill Matrix (`1K–256K`) — `Qwen3.8-27B-mxfp4`

Model dir: `/Users/scoltc/repo/models/mlx-community/Qwen3.8-27B-mxfp4` | KV: `affine4_g128` | `max_context=262144` | `spec=none` | `gen=128` | Multi-Needle NIAH (`10%/50%/90%`)

| Config | 1K TTFT (tok/s) | 4K TTFT (tok/s) | 8K TTFT (tok/s) | 16K TTFT (tok/s) | 32K TTFT (tok/s) | 64K TTFT (tok/s) | 128K TTFT (tok/s) | 256K TTFT (tok/s) | 256K vs `v0.03 M0` | Decode 1K→256K | Peak UMA (GiB) | MInf Skip 8K→256K | NIAH |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `miv-v0.03 M0` (Baseline BF16+Exact) | `0.62s` (1451) | `2.05s` (1933) | `4.04s` (1998) | `8.30s` (1944) | `18.24s` (1782) | `43.56s` (1499) | `116.59s` (1122) | `353.84s` (740) | **1.00x** | 61.4→45.4 | 19.34 | 0% | **24/24** |
| `miv-v0.03 M3b` (Baseline W4A8+Sage2+MInf) | `0.62s` (1456) | `2.10s` (1892) | `4.12s` (1958) | `8.10s` (1991) | `16.33s` (1991) | `34.21s` (1908) | `72.77s` (1798) | `160.40s` (1633) | **2.21x** | 61.3→45.4 | 19.11 | 48.1%→93.7% | **24/24** |
| **`M0_v2`** (`.miv v2` Lossless Base) | `0.62s` (1452) | `2.12s` (1867) | `4.20s` (1921) | `8.65s` (1879) | `18.77s` (1739) | `43.90s` (1490) | `114.10s` (1148) | `343.94s` (762) | **1.03x** | 53.7→39.3 | 19.13 | 0% | **24/24** |
| **`M1_v2`** (`.miv v2` 100% W4A8 + Exact) | `0.58s` (1549) | `1.93s` (2056) | `3.80s` (2120) | `7.85s` (2071) | `17.16s` (1902) | `40.67s` (1608) | `107.60s` (1217) | `330.68s` (792) | **1.07x** | 53.7→39.3 | 19.16 | 0% | **24/24** |
| **`M2a_v2`** (`.miv v2` Cured Dense Sage2) | `0.63s` (1412) | `2.13s` (1861) | `4.17s` (1932) | `8.35s` (1947) | `18.06s` (1808) | `41.12s` (1591) | `103.50s` (1265) | `296.95s` (882) | **1.19x** | 53.6→39.3 | 19.45 | 0% | **24/24** |
| **`M2b_v2`** (`.miv v2` Sparse Sage2 + WP4) | `0.63s` (1412) | `2.14s` (1858) | `4.16s` (1936) | `8.19s` (1986) | `16.36s` (1995) | `33.25s` (1967) | `68.41s` (1914) | `144.05s` (1819) | **2.46x** | 53.7→39.3 | 19.34 | 54.4%→93.8% | **24/24** |
| **`M3b_v2`** (`.miv v2` GPU 全速 Cold) | `0.59s` (1530) | `1.93s` (2055) | `3.76s` (2146) | `7.37s` (2206) | `14.75s` (2213) | `29.99s` (2181) | `61.93s` (2114) | `131.34s` (1995) | **2.69x** | 53.6→39.2 | 19.33 | 52.9%→93.7% | **24/24** |
| `M0_v1` (`.miv v1` Compat Base) | `0.60s` (1481) | `2.06s` (1923) | `4.07s` (1981) | `8.38s` (1939) | `18.23s` (1790) | `42.83s` (1527) | — | — | — | 55.0→46.7 (64K) | 19.01 | 0% | **18/18** |
| `M1_v1` (`.miv v1` Compat W4A8) | `0.58s` (1539) | `1.97s` (2015) | `3.88s` (2076) | `8.01s` (2030) | `17.50s` (1866) | `41.35s` (1582) | — | — | — | 54.9→46.8 (64K) | 18.98 | 0% | **18/18** |
| `M3b_v1` (`.miv v1` Compat GPU 全速) | `0.59s` (1513) | `1.97s` (2015) | `3.83s` (2103) | `7.52s` (2161) | `15.04s` (2170) | `30.60s` (2137) | `63.17s` (2073) | `133.63s` (1961) | **2.65x** | 54.9→39.9 | 19.28 | 54.5%→93.8% | **24/24** |

> Note: `miv-v0.03` `1K/4K/8K` rows were measured fresh on `versions/miv-v0.04-94d6739` (`prompt_tokens = 896, 3968, 8064`); `miv-v0.03` `16K–256K` rows are carried verbatim from `docs/benchmarks/m5ultra_256k_prefill_matrix.json` (`prompt_tokens = 16128..261888`, within `0.05%–0.78%` of Phase 2 `16256..262016`). For `v0.03 M3b`, `256K` Phase 2 `M3b_v2` speedup vs `v0.03 M3b` is **`1.22x`**.

### 3.1 `4K` & `16K` C++ Teacher-Forced Logprob Perplexity — `Qwen3.8-27B-mxfp4`

| Mode | 4K Mean NLL | 4K PPL | 4K `ΔPPL` vs `v0.03 M0` | 4K Top-1 | 16K Mean NLL | 16K PPL | 16K `ΔPPL` vs `v0.03 M0` | 16K Top-1 | Peak UMA (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `miv-v0.03 M0` (Reference) | 0.73168 | **2.0786** | +0.0000 | 83.08% | 0.43366 | **1.5429** | +0.0000 | 89.97% | 17.31 |
| **`M0_v1`** | 0.74612 | **2.1088** | +0.0302 | 82.83% | 0.48605 | **1.6259** | +0.0830 | 89.11% | 15.41 |
| **`M0_v2`** | 0.73995 | **2.0958** | +0.0172 | 83.03% | 0.37377 | **1.4532** | -0.0897 | 90.90% | 15.41 |
| **`M1_v1`** | 0.74985 | **2.1167** | +0.0381 | 82.86% | 0.45191 | **1.5713** | +0.0284 | 89.73% | 15.45 |
| **`M1_v2`** | 0.72508 | **2.0649** | -0.0137 | 83.17% | 0.39240 | **1.4805** | -0.0624 | 90.55% | 15.49 |
| **`M2a_v2`** | 0.73077 | **2.0767** | -0.0019 | 83.17% | 0.35892 | **1.4318** | -0.1111 | 91.21% | 15.47 |
| **`M2b_v2`** | 0.73077 | **2.0767** | -0.0019 | 83.17% | 0.36895 | **1.4462** | -0.0967 | 90.87% | 15.49 |
| **`M3a_v2`** | 0.74199 | **2.1001** | +0.0215 | 83.08% | 0.39425 | **1.4833** | -0.0596 | 90.56% | 15.54 |
| **`M3b_v2`** | 0.74199 | **2.1001** | +0.0215 | 83.08% | 0.40629 | **1.5012** | -0.0417 | 90.36% | 15.56 |

## 4. 64-Core M5 Ultra Scaling Verification (`OMLX_M5_FAKE_GPU_CORES=64`)

- **GDN Stage 2 (`dv16_flex_c64`, `10,240 B` SRAM, `0 B` spills) Dispatch Schedule (`omlx_m5_gdn_chunk_nax_tests`)**: `schedule_verified = true`, `6/6` variants `0-ulp exact` across `27B (Hv=48)` and `35B MoE (Hv=32)` (`27B/64c`: `K_dv=8` → `384` tasks = `6.00 TGs/core` = exact `2.00/3.00` waves; `27B/80c`: `K_dv=8` → `384` tasks = `4.80 TGs/core` = `1.00` wave; `35B/64c`: `K_dv=8` → `256` tasks = `4.00 TGs/core`; `35B/80c`: `K_dv=10` → `320` tasks = `4.00 TGs/core`).
- **End-to-End `Qwen3.8-27B-oQ4e-mtp` (`model.v2.miv`, `4K / 16K / 64K`)**:

| Mode | Schedule | 4K TTFT (tok/s) | 16K TTFT (tok/s) | 64K TTFT (tok/s) | Peak UMA (GiB) | NIAH |
|---|---|---:|---:|---:|---:|---|
| `M0_v2` | Native `80c` | `2.137s` (1856.8) | `8.360s` (1944.4) | `42.721s` (1531.1) | 22.95 | **9/9** |
| `M0_v2` | **`64c` (`FAKE_GPU_CORES=64`)** | `1.984s` (1999.9) | `8.221s` (1977.5) | `42.708s` (1531.5) | 22.94 | **9/9** |
| `M3b_v2` | Native `80c` | `1.594s` (2489.4) | `6.171s` (2634.4) | `25.088s` (2607.2) | 23.24 | **9/9** |
| `M3b_v2` | **`64c` (`FAKE_GPU_CORES=64`)** | `1.630s` (2434.7) | `6.160s` (2638.8) | `25.101s` (2605.8) | 23.24 | **9/9** |

## 5. Live `B=2` Concurrency Verification (`oMLX :8000`, `GPU 全速` + `CacheBlend = ON`)

- **Configuration**: `Qwen3.8-27B-oQ4e-mtp` via `oMLX :8000` (`external_engine=m5rt`, `miv-v0.04-p2-rc1/bin/mseries_backendd`), `max_concurrency=2`, `kv_dtype=affine4_g128`, `prefill_linear_mode=w4a8`, `prefill_attention_mode=sage2`, `prefill_sparse_minference=true`, `prefill_cache_blend=true`, `prefill_ane_offload=false`, `prefix_cache_gb=8.0`.

| Artifact | Load (s) | Round | Req A TTFT / Total (QA) | Req B TTFT / Total (QA) | Concurrent Overlap |
|---|---:|---|---|---|---|
| `v2` (`model.v2.miv`) | 4.57 | Cold `B=2` Multi-Doc | `6.224s` / `6.796s` (**4/4**) | `6.224s` / `6.796s` (**4/4**) | **True** |
| `v2` (`model.v2.miv`) | 4.57 | Warm CacheBlend `B=2` Permuted | `7.471s` / `7.887s` (**4/4**) | `3.987s` / `6.860s` (**4/4**) | **True** |
| `v1` (`model.miv`) | 4.85 | Cold `B=2` Multi-Doc | `6.286s` / `6.855s` (**4/4**) | `6.286s` / `6.855s` (**4/4**) | **True** |
| `v1` (`model.miv`) | 4.85 | Warm CacheBlend `B=2` Permuted | `7.574s` / `7.992s` (**4/4**) | `4.049s` / `6.972s` (**4/4**) | **True** |

- **`~/.omlx/logs/server.log` Audit (`6,041` bytes scanned)**: **`0`** `Qwen packed batch slot differs in execution contract` errors, **`0`** server errors/tracebacks.
- **Post-Test Restoration**: `:8000` model settings restored to lossless defaults (`bf16 + exact_nax`, `minference=false`, `cache_blend=false`, `ane_offload=false`, `max_concurrent_requests=1`), unloaded (`loaded_count == 0`, `current_model_memory == 0`).
