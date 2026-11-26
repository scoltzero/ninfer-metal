# Pre-Fix Original Baseline (7336688) — M5 Ultra 16K-256K Full Matrix

- **Date (UTC)**: `2026-11-26T13:46:42Z`
- **Host**: `scoltcdeMac-Studio.local` (`arm64`, `macOS 27.2.1`)
- **Binary / Baseline Commit**: `7336688` (`/Users/scoltc/Library/Application Support/oMLX/Providers/io.ninfer.metal/versions/miv-v0.03-7336688/bin/ninfer-metal-serve`)
- **Cell**: `M0` — Baseline (BF16 + Exact NAX)
- **Envelope**: `--max-context 262144 --max-concurrency 1 --kv-dtype affine4_g128 --spec-backend none --prefix-cache-gb 0.0`
- **Production `:8080` Safety**: Verified idle (`0.0%` CPU / `0` active requests) before & after every model run; untouched.

## Qwen3.8-27B-mxfp4 (`/Users/scoltc/repo/models/mlx-community/Qwen3.8-27B-mxfp4`)

| Context | Prompt Tok | Gen Tok | Median Cold TTFT (s) | Median Cold TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak `phys_footprint` (GiB) | NIAH (10/50/90%) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **16K** | 16,256 | 128 | 8.164 | 8163.7 | **1991.2** | 60.9 | 19.13 | 3/3 (100%) |
| **32K** | 32,640 | 128 | 18.028 | 18027.9 | **1810.5** | 60.4 | 19.14 | 3/3 (100%) |
| **64K** | 65,408 | 128 | 42.912 | 42911.8 | **1524.2** | 54.6 | 19.14 | 3/3 (100%) |
| **128K** | 130,944 | 128 | 114.249 | 114249.0 | **1146.1** | 48.0 | 19.28 | 3/3 (100%) |
| **256K** | 262,016 | 128 | 344.572 | 344572.4 | **760.4** | 45.4 | 19.32 | 3/3 (100%) |

## Qwen3.8-27B-oQ4e-mtp (`/Users/scoltc/repo/models/Jundot/Qwen3.8-27B-oQ4e-mtp`)

| Context | Prompt Tok | Gen Tok | Median Cold TTFT (s) | Median Cold TTFT (ms) | Prefill (tok/s) | Decode (tok/s) | Peak `phys_footprint` (GiB) | NIAH (10/50/90%) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **16K** | 16,256 | 128 | 8.025 | 8025.0 | **2025.7** | 50.9 | 23.00 | 3/3 (100%) |
| **32K** | 32,640 | 128 | 17.801 | 17800.8 | **1833.6** | 50.4 | 23.01 | 3/3 (100%) |
| **64K** | 65,408 | 128 | 42.700 | 42700.1 | **1531.8** | 46.0 | 23.01 | 3/3 (100%) |
| **128K** | 130,944 | 128 | 114.609 | 114609.1 | **1142.5** | 40.5 | 23.09 | 3/3 (100%) |
| **256K** | 262,016 | 128 | 348.321 | 348320.9 | **752.2** | 39.0 | 23.23 | 3/3 (100%) |

## Teacher-Forced Logprob Perplexity (`4K` & `16K` Deterministic Corpus)

| Model | Tokens | Scored | Mean NLL | PPL | Top-1 Acc | Last-Token Argmax | Peak `phys_footprint` (GiB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3.8-27B-mxfp4 | 4,096 | 4,095 | 0.74290 | **2.1020** | 82.81% | `1302` | 15.53 |
| Qwen3.8-27B-mxfp4 | 16,384 | 16,383 | 0.46288 | **1.5886** | 89.36% | `4895` | 15.53 |
| Qwen3.8-27B-oQ4e-mtp | 4,096 | 4,095 | 1.07688 | **2.9355** | 78.12% | `1302` | 19.40 |
| Qwen3.8-27B-oQ4e-mtp | 16,384 | 16,383 | 1.57776 | **4.8441** | 76.85% | `4895` | 19.40 |

