[English](../../microkernel-foundry.md) | [简体中文](../zh-CN/microkernel-foundry.md) | **繁體中文** | [日本語](../ja/microkernel-foundry.md) | [한국어](../ko/microkernel-foundry.md) | [Español](../es/microkernel-foundry.md) | [Français](../fr/microkernel-foundry.md) | [Deutsch](../de/microkernel-foundry.md)

# 微核心工坊（Microkernel Foundry）

`ninfer-metal`（`miv-v0.01`）已在 Apple M5+ 晶片上提供 Qwen3.5、Qwen3.6 與 Qwen3.8 混合架構模型的裸晶片 Metal 4 NAX 推論。**微核心工坊**將同一套執行期基底、服務前端、校準工具與認證流程開放給社群貢獻者與開發 Agent，支援透過主線 Pull Request 或平行 Sidecar 接入新的模型家族。

目標模型家族包含 **Llama、Qwen3 純注意力（Pure-Attention）、DeepSeek（MLA）、Gemma、GLM、MiniMax**，以及社群量化衍生權重（例如 [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)）。

---

## 兩種貢獻路徑

### 路徑 A：主線 PR → 貢獻者署名（推薦）

在共享執行期 **`libninfer_metal_substrate`** 之上，將新模型家族的微核心直接併入 `ninfer-metal`。

**共享基底已提供：**
- Metal 4 執行期（`MetalContext`、`MTL4CommandQueue`、雙緩衝 `ar_pipe_[2]`、`< 2 GiB` Compute ICB 編碼器、`read_write` 屏障反射）。
- 共享分頁 KV 與前綴記憶體（`KvPagePool`、`SharedKvArenaPool`、CoW 參考計數、`SharedPrefixCache`、`UmaCheckpointPool`）。
- 通用 NAX 與解碼原語（`mxfp4_nax`、`affine_nax`、`bf16_nax`、`moe_nax`、零 `uitofp` QMV、兩階段平行 `Argmax`、`Split-KV` / `Flash-NAX`）。
- `.miv`（相容舊版 `.omlxm5`）容器載入器、隨附權重重排轉換器（`tools/m5_artifact.py`）、`ExecutionIdentityFields`、`24 MHz` 硬體計數器、原生獨立 OpenAI 伺服器（`ninfer-metal-serve`）與 Provider SPI v2 常駐程式（`mseries_backendd`）。

**你的 PR 補充內容與執行範圍：** 補充目標架構的計算層圖、權重對應、配套的 `.safetensors → .miv` 重排轉換規則，以及在冷 DRAM 微探針上比通用原語快 **`>= +10%–12%`** 的架構專屬核心（例如 MLA、GeGLU、Softcap 或自訂 MoE 路由器）。重用 `libninfer_metal_substrate` 可避免多架構間的程式碼與常駐記憶體重複，讓模型同時原生執行於獨立 `ninfer-metal-serve` 與 `oMLX-extend`（支援首次點擊 `Load` 自動同目錄重排生成 `.miv` 並保留原權重），並於[支援的模型清單](README.md)標註貢獻者署名。

### 路徑 B：平行 Sidecar → PR 或 Issue → 原始專案儲存庫署名

若為實驗性架構、自訂量化排布或獨立發版節奏，可實作獨立的 Sidecar SPI v2 套件並安裝至：

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

透過原子 `current` 符號連結暴露，並在 `provider.json` 宣告 `supported_topologies`、`supported_model_types`、`artifact_extensions` 與必選的隨附重排轉換器 `converter`（`--source <dir> --output <file> --source-repo <id>`，與原 `.safetensors` 同目錄並存）。就緒後發送 PR 或開啟 Issue：維護者會將其登錄至 [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json)，供使用者透過 [`./omlx-extend/scripts/sync_microkernels.sh`](../../../omlx-extend/scripts/sync_microkernels.sh) 安裝與更新，同時在[支援的模型清單](README.md)附上原始儲存庫直連。

**統一多微核心宿主：** `oMLX-extend` 與獨立伺服器 `ninfer-metal-serve --providers-dir` 皆原生探索 `Providers/*/current/provider.json`，在同一連接埠上平行執行多個路徑 A（**`miv 核心後端`**）與路徑 B 微核心，並統一執行 UMA 記憶體預算仲裁與 `gpu_contended` 排程（於 `oMLX-extend` 中同時相容原版 MLX）。在 `oMLX` macOS 應用程式中，已安裝的路徑 B 微核心會自動顯示於 **`Integrations → External Engines`** 卡片清單，並在 **`Models`** 畫面標示執行引擎徽章（見 [`oMLX-extend` 視覺化上手指南](omlx-extend-guide.md)）。微核心作者只需實作 Metal 4 推論核心與 Sidecar SPI v2 常駐程式；HTTP/SSE 服務、分詞、聊天範本與多模型排程皆由兩種宿主直接承接。

**開源授權：** **路徑 A** 主線 PR 連結至 `miv` 並以 **[GPL-3.0](../../../LICENSE)** 散佈（貢獻者可提交 GPL-3.0、Apache-2.0 或 MIT 授權程式碼）；**路徑 B** 平行 Sidecar 作為獨立行程透過 AF_UNIX Sidecar SPI v2 通訊，可使用任意授權條款。

---

## 模型家族清單

| 架構家族 | 目標範例 | 狀態 | 建議路徑 |
|---|---|---|---|
| **Qwen 混合架構（3:1 GDN + GQA）** | Qwen3.5 / 3.6 / 3.8 Dense（0.8B – 27B）與 35B-A3B MoE | 已發行（`miv-v0.01`） | 路徑 A（主線，`miv/`） |
| **純注意力 GQA** | Llama-3.3-70B、Qwen3-32B、Gemma | 開放 PR | 路徑 A |
| **多頭潛在注意力（MLA）** | DeepSeek V3 / R1 系列 | 開放 PR | 路徑 A / 路徑 B |
| **MoE 與社群量化權重** | GLM、MiniMax、`heretic-mlx` 衍生權重 | 開放 PR | 路徑 A / 路徑 B |

---

## 如何開發並認證微核心

不論人類工程師或開發 Agent，皆遵循同一套可重現的四步驟流程：

1. **校準目標 Mac（`Stage B: H0 + H1`）**
   執行 `H0` 硬體 DMA 掃描（`MTLBlitPassDescriptor`，`4–512+ MiB`）實測晶片本身的 $B_{\text{DRAM}}$、$C_{\text{SLC}}$、$P_{\text{NAX}}$ 與 $I_{\text{ridge}}$，接著執行 `H1` 算子掃描。所有驗收門檻均以這些實測常數正規化。
2. **先做冷 DRAM 微探針（`~50–100` 行）**
   在範本化整套新核心家族之前，先用獨立冷 DRAM 探針（`working set >= 3–4 × C_SLC`，`>= 30` 次序列屏障中位數）對標固定功能 NAX／硬體基線。僅保留淨增益達 **`>= +10%–12%`** 的核心；發送 PR 前刪除未達標的實驗性 PSO。
3. **`< 10 s` 子層快速迴圈與暫存器溢位稽核**
   使用 `omlx_m5_hw_counter_probe --mode model`（`24 MHz` 計數器）量測 GPU 各子層佔比（`mlp/moe`、`attn`、`lm_head`、`sampling`），並以 `metal-objdump -d` 稽核著色器組合語言（`0 alloca` 溢位、寬磚塊 Prefill `<= 104` VGPR、`T=1` 預先重排 QMV 保持 `0 uitofp`）。
4. **附上 5 檔案機器可讀證據包**
   執行潔淨室 Dim A–F 評測套件（`miv/tools/bench_omlx_vs_m5rt.py`），隨 PR 附上 `certification-evidence/<core-id>-<version>-<sku>/`：
   - `h0-capability.json` — 裸晶片 `H0` 實測常數與 10,000 步 ICB 金絲雀測試
   - `h1-calibration-<sku>.json` — SKU 隔離的 `H1` 調校參數與 `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — 潔淨室 Dim A–F 基準數據與 `phys_footprint`
   - `parity-report.json` — `0-ULP` 一致性、來源權重 PPL/KL、NIAH/RULER 召回率、`0 NaN/Inf`
   - `otool-audit.txt` — 純 Apple 系統框架 `otool -L` 稽核、`0 alloca` 稽核、100 輪穩定性長跑測試與 CTest 輸出

---

## 貢獻者與 Agent 必讀文件

- **[Silicon-Squeeze 核心開發 Skill](../../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — 完整的 Apple M5+（`MTLGPUFamilyApple10+`）裸晶片調校方法論，可直接載入開發 Agent 使用。
- **[核心認證標準](../../standards/core-certification-standard.md)** — Gate 1–4 精確通過門檻、Provider SPI v2 契約、Roofline 閾值與 Tier 1 / Tier 2 認證標章規範。
- **[Apple M5 系列最佳化手冊](../../silicon-guides/m5-series-optimization-playbook.md)** — `32 KiB` 執行緒群組佔用率計算、統一記憶體准入方程式與 Base / Pro / Max / Ultra 擴展特性。
