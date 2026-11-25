[English](../../releases/miv-v0.01-changelog.md) | [简体中文](../zh-CN/miv-v0.01-changelog.md) | **繁體中文** | [日本語](../ja/miv-v0.01-changelog.md) | [한국어](../ko/miv-v0.01-changelog.md) | [Español](../es/miv-v0.01-changelog.md) | [Français](../fr/miv-v0.01-changelog.md) | [Deutsch](../de/miv-v0.01-changelog.md)

# `ninfer-metal` `miv-v0.01` 發行說明

> **版本**：`ninfer-metal` `miv-v0.01`（`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`）
> **搭配宿主**：`oMLX-extend` `omlx-extend-v0.8.3`（面向 `jundot/omlx` `v0.8.3` 的 Provider SPI v2 擴充層）
> **平台需求**：Apple M5 / M5 Pro / M5 Max / M5 Ultra（`MTLGPUFamilyApple10+`），macOS 27+（執行免 Xcode；從原始碼編譯需 Xcode 27+）

`miv-v0.01` 是 `ninfer-metal` 的首個公開發行版本。引擎以純 C++20、Objective-C++ 與 Metal 4 實作，直接驅動 Apple M5+ 硬體神經網路加速器（NAX）與 Compute ICB，執行期零 Python、零 MLX 相依。本版本支援 **Qwen3.5 / Qwen3.6 / Qwen3.8 混合架構**模型（Dense 0.8B–27B 與 256×8 MoE，如 35B-A3B），可透過原生 OpenAI HTTP/SSE 伺服器 `ninfer-metal-serve` 獨立執行，也能作為 **`miv 核心後端`** 整合進 `oMLX-extend`，與原版 MLX 同機運作並提供 OpenAI、Anthropic 端點與原生 macOS 介面。安裝與命令列旗標見 [快速上手與設定](getting-started.md)，macOS 應用程式操作畫面見 [`oMLX-extend` 視覺化上手指南](omlx-extend-guide.md)。

## 1. Apple M5 Ultra 實機基準

以下數據均於實體 Apple M5 Ultra 上以獨立子程序實測，對標同機原版 `oMLX 0.8.0`（`mlx 0.32.4`，開啟 NAX），採用相同模型權重、相同 Token 序列、`temperature=0` 與 `phys_footprint` 記憶體統計。80-GPU 256 GiB（`macOS 27.2.1` Build `26B131`，`Xcode 27.2` Build `27B52`）與 64-GPU 96 GiB 兩種規格分別記錄。

### 單流解碼與 `0-ULP` 推測加速

![單流解碼與 0-ULP 推測加速](../../assets/charts/decode-speedup.svg)

- 80-GPU M5 Ultra 上，純自回歸（AR）解碼在所有受測權重格式皆領先原版 oMLX：`Qwen3.8-27B MXFP4` 達 `64.63 tok/s`（`+14.5%`），`27B oQ4e` 達 `54.03 tok/s`（`+7.3%`），`Qwen3.6-35B-A3B MXFP4` 達 `216.44 tok/s`（`+23.7%`）。
- 推測解碼讓 `27B` 程式碼生成達到 `186.74 tok/s`（`DFlash2`，AR 的 `2.89×`）與 `140.48 tok/s`（`Native MTP`，較原版 oMLX MTP 的 `103.28 tok/s` 快 `+36.0%`），全數維持 `128/128` `0-ULP` 貪婪解碼一致性。
- 64-GPU 96 GiB 機型上，`27B oQ4e` 於 Coding 情境達 `138.92 tok/s`（`DFlash2`，`+33.9%`）與 `107.52 tok/s`（`MTP`，`+21.2%`），Reasoning 情境為 `110.01` / `100.28 tok/s`（`6/6` 組、`256/256` `0-ULP`）。

### 多使用者並發擴展

![多使用者並發擴展總吞吐量](../../assets/charts/concurrency-scaling.svg)

- 80-GPU 機型共測 19 組並發點，`miv` 全數勝出：`B=4` 時 `27B` 達 `500.57 tok/s`（`oQ4e DFlash2`）、`496.60 tok/s`（`MXFP4 DFlash2`）與 `434.92 tok/s`（`oQ4e MTP`），為原版 oMLX 基線（`148.37` MTP / `129.80` AR）的 `2.93×–3.83×`。
- `Qwen3.6-35B-A3B` 於 `B=8` 純 AR 擴展至 `846.29 tok/s` 視窗吞吐（端對端 `794.39 tok/s`），較原版 oMLX 高 `+52.6%`，首字延遲（TTFT）快 `4.16×`。
- 所有並發槽位皆與單流輸出保持 `128/128 × 4` `0-ULP` 貪婪解碼一致性。

### 64K 長上下文穩定性、UMA 佔用與 Prefill

![64K 長上下文穩定性與 affine4_g128 UMA 記憶體佔用](../../assets/charts/long-context-curve.svg)

- 啟用 `affine4_g128` KV 後，`27B MXFP4` 自 `2K` 延伸至 `64K` 時 AR 速度僅降 `-4.1%`（`61.42 → 58.92 tok/s`，較 oMLX 快 `+28.6%`）；真實英文與程式碼混合負載下，`64K` `DFlash2` 維持 `139.70 tok/s`（`2K–64K` 全程 `0-ULP`）。
- `affine4_g128` 將 `64K` KV 快取縮減 `71.8%`（`4.00 → 1.13 GiB`），單次請求峰值 UMA 降至 `15.69 GiB`（較 oMLX 省 `3.44 GiB`；四種前綴深度同時駐留共省 `10.06 GiB`）。`64K + 32` 熱前綴 TTFT 降至 `176.19 ms`（`bf16`）與 `201.42 ms`（`affine4`），較 oMLX 快 `7.05×–9.48×`。
- 80-GPU 機型 `1K` Prefill 達 `2,058.5–2,110.6 tok/s`（`100.25–102.80 TF`），`64K` 冷 Prefill 耗時 `41.02–41.84 s`；64-GPU 機型 `4K` Prefill 15 項全勝，`35B-A3B` 達 `10,048.5 tok/s`（`+15.9%`）。

### 多輪短對話與滾動前綴重用

![多輪短對話滾動前綴重用與 UMA 平坦度](../../assets/charts/multiturn-rolling.svg)

- 滾動 UMA 前綴快取將多輪命中率提高至 `92%` 以上（`B=1` `92.33%`，`B=4` `92.02%`），重複 Prefill 的 Token 減半，熱 TTFT 縮短 `2.44×–2.94×`（`B=1` `101.25 ms`，`B=4` `282.80 ms`）。
- 原地 slab 重用讓 `B=1` 10 輪對話的 UMA 增幅維持在 `+0.02 MB` 且零逐出（原版 oMLX `affine4` 增加 `+397.5 MB`，`bf16` 增加 `+704.8 MB`），10 輪總耗時由 `16.99 s` 降至 `9.26 s`（`DFlash2`）。
- `B=4`（4 位使用者 × 6 輪）下，批次 `affine4` NAX 解碼達 `176.95 tok/s`（`AR`）與 `206.35 tok/s`（`DFlash2`，oMLX 的 `1.72×`），峰值 UMA 省下 `2.12 GiB`，`B=1`（`10/10`）與 `B=4`（`24/24`）全輪次通過 `0-ULP` 驗證。

## 2. 本版包含內容

### 獨立 OpenAI 伺服器（`ninfer-metal-serve`）
- 單一 C++20 HTTP/SSE 二進位檔，提供 `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health` 與 Prometheus `/metrics`，支援同步 JSON、SSE 串流（含 `usage` 與 `[DONE]`）、選用 Bearer 驗證（`--api-key`）、CORS 及平滑關閉。
- 內建 C++ BPE 分詞器（直接讀取 `tokenizer.json`）、Jinja 聊天範本渲染、`<think>` `reasoning_content` 拆分，以及 Qwen XML/JSON `<tool_call>` 同步與串流解析。
- 支援同連接埠多微核心平行託管：以 `--providers-dir` 與 `[[models]]` 在程序內 `miv` 旁載入外部路徑 B Provider SPI v2 Sidecar，統一控管 UMA 記憶體上限（預設實體記憶體的 `81%`）與並發爭用排程。
- 隨附可直接執行的 [`ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)（與 `oMLX-extend` 共用並自動載入）以及一鍵啟動腳本 `./omlx-extend/scripts/start_ninfer_serve.sh`。

### Metal 4 NAX 推論引擎（`miv/`）
- 自包含的 C++20 / Objective-C++ / Metal 4 原始碼樹，隨附 38 個預先編譯的 `.metallib` 著色器，涵蓋 NAX Prefill、融合解碼、256×8 MoE 路由、推測驗證與 4-bit Split-KV，編譯與執行皆不需 MLX 或 Python 標頭檔。
- 為 Apple M5 Ultra 兩款規格內建獨立硬體校準檔（[80-GPU 256 GiB](../../backend/silicon-calibrations/apple-m5-ultra-80gpu.json) 與 [64-GPU 96 GiB](../../backend/silicon-calibrations/apple-m5-ultra-64gpu.json)），依晶片自動對應 split-K、Split-KV 分割數與推測草稿深度，並保留環境變數覆寫優先權。
- 支援零拷貝 GPU 前綴續接（`64K` 還原耗時 `0.73–1.62 ms`）、原地滾動 UMA 前綴重用，以及面向並發連線的批次 `affine4_g128` NAX 注意力。

### `oMLX-extend` `omlx-extend-v0.8.3`（macOS 應用程式與 Sidecar）
- 以 `AF_UNIX` Sidecar（`mseries_backendd`）接入 **`miv 核心後端`**：已適配的 `.miv` 模型以 O(1) 自動路由，一般 MLX 模型維持在原版 MLX 上運作，不啟動額外程序。
- 原生 macOS 介面於 `Integrations → External Engines` 新增 `miv 核心後端` 與路徑 B 微核心卡片，並在模型清單與設定頁提供執行引擎徽章、自動路由／原版 MLX 切換，以及依後端能力解鎖的 `64K+` 上下文、`MTP`、`DFlash2` 與 `affine4_g128` KV 控制項。
- 在 Sidecar 與 MLX 模型之間協調 UMA 預算、閒置快取回收與並發分塊 Prefill，並支援以 `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/` 原子符號連結直接升級與回滾。

## 3. 獨立封裝與驗證

- 發行封裝檔 `ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` 收錄 16 個 Mach-O 二進位檔（含 `ninfer-metal-serve`、`mseries_backendd` 及命令列／硬體校準工具）、38 個 Metal 4 `.metallib` 著色器、2 份 SKU 校準檔、`provider.json` 與 SHA-256 檢查碼。
- 對全部 16 個二進位檔執行 `otool -L` 稽核，確認僅連結 Apple 系統框架（`CoreFoundation`、`Foundation`、`IOKit`、`Metal`）與系統動態函式庫（`libc++.1`、`libSystem.B`、`libobjc.A`），執行期無 Homebrew、Xcode、MLX、OpenSSL 或 Python 相依。
- 端對端伺服器測試（涵蓋同步／SSE、`<think>`／`<tool_call>`、`B=4` 批次以及程序內＋外部 Sidecar 平行託管）、`oMLX-extend` 全部 `638 / 638` 項 pytest 以及 `75 / 75` 項 macOS XCTest 均已通過，Provider SPI 修補檔可無衝突套用於 `jundot/omlx` `v0.8.3`。

## 4. 支援範圍

| 項目 | `miv-v0.01` 支援範圍 |
|---|---|
| 硬體 | Apple Silicon `MTLGPUFamilyApple10+`（Apple M5 / M5 Pro / M5 Max / M5 Ultra，含 64-GPU 與 80-GPU） |
| 系統與編譯 | macOS 27+（執行免 Xcode；從原始碼編譯需 Xcode 27+ / macOS 27 SDK） |
| 模型架構 | **Qwen3.5 / Qwen3.6 / Qwen3.8 混合架構**（3:1 GDN + GQA；Dense 0.8B–27B 與 256×8 MoE，如 35B-A3B） |
| 權重格式 | 離線 NAX 預先重排 `.miv`：`MXFP4`（`E2M1 × UE8M0`）與 `oQ` Affine（`q3–q8`，`g64 / g128`） |
| KV 快取 | `bf16` 與 `affine4_g128`（`0.5625 B/val`，`2K–64K` 通過 `0-ULP` 驗證） |
| 推測解碼 | 純 AR（`none`）、內嵌 `Native MTP` 與 `DFlash2`（支援單流與 `B=2..4` 批次） |
| 服務介面 | `ninfer-metal-serve` 提供 OpenAI HTTP/SSE 端點；`oMLX-extend` 提供 OpenAI + Anthropic 端點與原生 macOS 介面 |
