[English](../../getting-started.md) | [简体中文](../zh-CN/getting-started.md) | **繁體中文** | [日本語](../ja/getting-started.md) | [한국어](../ko/getting-started.md) | [Español](../es/getting-started.md) | [Français](../fr/getting-started.md) | [Deutsch](../de/getting-started.md)

# 快速上手與設定

本指南說明如何編譯零 MLX 的 `ninfer-metal` 核心（`miv/`）、將模型權重轉換為 `.miv`、一行指令啟動原生 C++20 獨立 OpenAI 伺服器（`ninfer-metal-serve`）、將 **`miv 核心後端`** 整合至 `oMLX-extend`、調整 `ninfer-metal.toml`，以及呼叫底層 CLI 與常駐程式。若使用 `oMLX` macOS 應用程式，可先看 **[`oMLX-extend` 視覺化上手指南](omlx-extend-guide.md)** 的 5 步驟圖解教學。

---

## 環境需求

| 模式 | 硬體 | 作業系統與工具鏈 |
|---|---|---|
| **執行環境（預先編譯套件）** | Apple M5 / M5 Pro / M5 Max / M5 Ultra（`MTLGPUFamilyApple10+`） | `macOS 27+`（零 Xcode、Python、MLX 或 Homebrew 相依） |
| **從原始碼編譯** | Apple Silicon（`arm64`） | `macOS 27+`、`Xcode 27+`（macOS 27 SDK）、`CMake 3.27+`、`Ninja` |

---

## 1. 編譯 `ninfer-metal`（`miv/`，零 MLX / 零 Python）

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer

# Configure and compile all 38 pure Metal 4 .metallib shaders and 16 binaries
cmake -S . -B build/miv-v001 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DOMLX_M5_BUILD_NAX_METALLIB=ON \
  -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF \
  -DOMLX_M5_BUILD_PROBES=ON \
  -DOMLX_M5_BUILD_PYTHON=OFF
cmake --build build/miv-v001

# Verify version identity for both standalone server and sidecar daemon
./build/miv-v001/miv/ninfer-metal-serve --version
# ninfer-metal-serve miv-v0.01 (codename=miv semver=0.0.1 core_abi=1)

./build/miv-v001/miv/mseries_backendd --version
# mseries-backendd miv-v0.01 (codename=miv semver=0.0.1 core_abi=1) protocol=2

# Run the CTest suite
ctest --test-dir build/miv-v001 --output-on-failure
```

若需要一併編譯選用的 `omlx_m5_native` Python 擴充模組，加上 `-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>`。完整開發預設組態見 [`CMakePresets.json`](../../../CMakePresets.json)。

---

## 2. 封裝發行套件與轉換權重

### 潔淨 Mac 發行套件

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

此指令會產出 `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz`（含 `.sha256`、`provider.json` 與 `latest.json`），並對全部 16 個 Mach-O 二進位檔（`ninfer-metal-serve`、`mseries_backendd`、`m5rt_runner` 及 13 個硬體／校準探針）執行 `otool -L` 檢查，確認僅連結 Apple 系統框架（`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` 與 `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`）。

在任何執行 macOS 27+ 的 Apple M5+ Mac 上解壓縮後，執行 `install.sh`（會自動複製預設 `ninfer-metal.toml`，且不覆寫已存在的自訂修改）即可安裝至：

```text
~/Library/Application Support/oMLX/Providers/io.ninfer.metal/
├── current -> versions/miv-v0.01/     # Atomic symlink for hot-update & rollback
└── versions/
    └── miv-v0.01/
        ├── provider.json
        ├── RELEASE-MANIFEST.json
        ├── bin/                       # ninfer-metal-serve, mseries_backendd, m5rt_runner, probes
        ├── config/ninfer-metal.toml   # Ready-to-run shared M5+ defaults
        ├── scripts/start_ninfer_serve.sh
        ├── metallib/                  # 38 precompiled Metal 4 .metallib files
        ├── calibrations/              # SKU-isolated JSON profiles (64-GPU, 80-GPU)
        └── tools/
```

### 將模型轉換為 `.miv`

離線 NAX 預先磚塊化重排會一次性封裝 `MXFP4`（`E2M1 × UE8M0`）或 `oQ` Affine（`q3–q8`、`g64/g128`）權重，讓 Prefill、`T=1` 自回歸解碼與推測驗證共用同一份常駐記憶體副本：

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.miv \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. 使用 `ninfer-metal-serve` 獨立執行（原生 C++）

`ninfer-metal-serve` 是直接連結 `libomlx_m5_core_runtime.a` 的單一 C++20 二進位檔。它在同一 HTTP/SSE 連接埠平行託管程序內 `miv` 模型與外部路徑 B Provider SPI v2 微核心，內建 `ninfer-metal.toml` 設定讀取、C++ BPE 分詞器（`tokenizer.json`）、Jinja 聊天範本（`chat_template.jinja` / `tokenizer_config.json` / ChatML 備援）、`<think>` 推理內容拆分及 Qwen XML/JSON `<tool_call>` 解析，零 Python 且零 MLX 相依。

### 一鍵啟動（免手寫共用預設設定檔）

不需手動撰寫設定檔即可直接啟動 `ninfer-metal-serve`。啟動腳本會自動尋找 `ninfer-metal-serve` 並載入 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)——`oMLX-extend` 共用的是同一份預設檔（`max_concurrency = 4`、`max_context = 65536`、`kv_dtype = "affine4_g128"`、`rolling_prefix_cache = true`、`prefix_cache_gb = 4.0`、`speculative.backend = "auto"`、`providers_dir = ["~/Library/Application Support/oMLX/Providers"]`）：

```bash
# 指定模型目錄一鍵啟動（預設連接埠 8080，與 oMLX 的 8000 錯開）
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# 省略路徑時，自動選取 ./models 或 oMLX 模型目錄中的第一個 .miv 模型
./omlx-extend/scripts/start_ninfer_serve.sh

# 加上任意命令列旗標即可覆寫共用預設組態
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
```

### 命令列直接覆寫或載入自訂 `ninfer-metal.toml`

直接執行 `ninfer-metal-serve` 時，只要未指定 `--config` 或 `--no-default-config`，同樣會自動載入預設 `ninfer-metal.toml`；CLI 旗標一律優先於 TOML：

```bash
# 直接以命令列顯式覆寫參數
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# 明確啟用 DFlash2 推測解碼（或直接使用預設的 speculative.backend = "auto"）
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# 載入指定的 ninfer-metal.toml（CLI 旗標優先於 TOML）
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### 平行執行多個微核心（`[[models]]` / `--providers-dir`）

在單一連接埠同時託管程序內 `miv`（路徑 A）與路徑 B SPI v2 Sidecar 微核心，統一執行 UMA 記憶體預算上限（`--memory-budget-gb`，預設實體記憶體的 81%）與自動 `gpu_contended` 排程：

```bash
./build/miv-v001/miv/ninfer-metal-serve \
  --providers-dir "$HOME/Library/Application Support/oMLX/Providers" \
  --model-dir /path/to/Qwen3.8-27B \
  --model-dir /path/to/Llama-3.3-70B-Instruct \
  --memory-budget-gb 64
```

如需個別模型設定（`provider = "auto" | "inproc" | "<provider_id>"`、`kv_dtype`、`max_concurrency`、`memory_budget_gb`），於 `ninfer-metal.toml` 宣告 `[[models]]` 區塊並以 `--config` 啟動。請求依 JSON `"model"` 欄位路由（未知 ID 回傳 `404 model_not_found`）。

### 連線至用戶端（Open WebUI、Cursor、Cline、NewAPI、OpenAI SDK）

將任一 OpenAI 相容用戶端的 API 位址設為 `http://127.0.0.1:8080/v1`：

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "model": "Qwen3.8-27B",
    "messages": [{"role": "user", "content": "Write a quicksort in C++20."}],
    "stream": true,
    "stream_options": {"include_usage": true}
  }'
```

### HTTP 端點（`miv-v0.01`）

| 端點 | 行為 |
|---|---|
| `GET /health` | 就緒狀態 JSON（`status`、`model`、`models[]`、`gpu_contended`、`memory_budget_bytes`、`version`） |
| `GET /metrics` | Prometheus 純文字指標（彙總 + 依 `model`/`provider`/`backend_mode` 標記之分模型序列：`requests_total`、`active_requests`、`generated_tokens_total`、UMA／快取統計） |
| `GET /v1/models`、`GET /v1/models/:id` | OpenAI 模型清單，含 `max_model_len`、`provider_id` 與 `backend_mode` |
| `POST /v1/chat/completions` | 依 JSON `"model"` 路由（未知 ID 回傳 `404 model_not_found`；單模型時省略則預設路由至唯一模型）；同步 JSON 與 SSE 串流（`chat.completion.chunk`）、`<think>...</think>` 拆至 `reasoning_content`、Qwen XML 與 JSON `<tool_call>` 解析、`stop` 字串、`usage`、`stream_options.include_usage` 及 `[DONE]` |
| `POST /v1/completions` | 依 JSON `"model"` 路由的原始文字補全（同步 JSON、SSE 串流、`usage`、`[DONE]`） |

選用的 Bearer 驗證（`--api-key`）於金鑰不符時回傳 `401`；CORS 預檢（`OPTIONS`）預設啟用。

### `ninfer-metal-serve` 命令列旗標

未指定 `--config` 時，`ninfer-metal-serve` 會自動載入 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)（`max_concurrency = 4`、`max_context = 65536`、`kv_dtype = "affine4_g128"`、`rolling_prefix_cache = true`、`prefix_cache_gb = 4.0`、`speculative.backend = "auto"`）。傳入 `--no-default-config` 可退回下表的純編譯預設值；完整註解參考見 [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)。

| 旗標 | 純 CLI 預設值 / 選項 | 說明 |
|---|---|---|
| `--config PATH` | 自動偵測 | 明確載入 `ninfer-metal.toml`（`[server]`、`[model]`、`[[models]]`、`[engine]`、`[speculative]`）；CLI 旗標優先於 TOML |
| `--no-default-config` | 關閉 | 略過預設 `ninfer-metal.toml` 自動載入，僅使用純編譯預設值 |
| `--model-dir PATH` | 必填（或於 TOML 指定） | 模型目錄，包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 與模型權重檔；**可重複指定**以平行託管多個模型 |
| `--providers-dir PATH` | 自動（`$OMLX_PROVIDERS_DIR`、`~/Library/Application Support/oMLX/Providers`） | **可重複指定**的路徑 B SPI v2 微核心根目錄；掃描 `<dir>/*/current/provider.json`（備援 `<dir>/*/provider.json`） |
| `--provider MODE` | `auto`（`auto` \| `inproc` \| `<provider_id>`） | 後端路由模式：O(1) 自動對應、程序內 `miv`（`inproc`）或明確指定路徑 B SPI v2 `provider_id` |
| `--metallib-dir PATH` | 自動偵測 | Metal 4 / NAX `.metallib` 著色器目錄（搜尋順序：`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`）。別名 `--kernel-dir`（指向 Metal 著色器，**不是**路徑 B 微核心；微核心請使用 `--providers-dir`） |
| `--artifact PATH` | 自動偵測 | 明確指定目標 `.miv`（或對應 Provider）權重檔路徑 |
| `--draft-dir PATH`、`--draft-artifact PATH`、`--mtp-artifact PATH` | — | 供 `dflash2` / `mtp` 使用的草稿模型目錄或 `.miv` 檔案 |
| `--chat-template PATH`、`--model-id NAME` | 自動 | 覆寫 Jinja 聊天範本檔或對外宣告的模型 ID |
| `--host ADDR`、`--port PORT` | `127.0.0.1`、`8080` | HTTP 繫結位址與連接埠 |
| `--api-key KEY`、`--cors` / `--no-cors` | `""`、啟用 | 要求 `Authorization: Bearer <KEY>` 驗證；控制 CORS 標頭 |
| `--memory-budget-gb FLOAT` | 實體記憶體的 `81%` | 跨所有託管模型的伺服器總 UMA 記憶體預算上限（GiB） |
| `--max-concurrency N`（`-c`、`-b`） | `1`（`1..8`） | 多使用者並發批次槽位數（`B`/`C`） |
| `--max-context N` | `32768` | 上下文視窗長度（`L`） |
| `--kv-dtype MODE` | `bf16`（`bf16` \| `affine4_g128`） | KV 快取儲存格式 |
| `--spec-backend MODE` | `none`（`auto` \| `none` \| `mtp` \| `dflash2`） | 推測解碼後端（`auto` 偵測到配套 `dflash2/` 時選 `dflash2`，其次 `mtp`，否則 `none`） |
| `--draft-tokens N` | `0`（自動；可明確設 `1..7`） | 每輪驗證產生的草稿 Token 數 |
| `--rolling-prefix-cache on\|off` | 未設（沿用引擎預設） | 啟用或停用多輪滾動 UMA 前綴重用與 `1×` 祖先原地覆寫 |
| `--prefix-cache-gb FLOAT` | 未設（`0..128`） | 前綴快取與 UMA 檢查點池記憶體預算（GiB） |
| `--sampling-contract NAME` | `legacy_rejection_v1` | 推測驗證取樣契約（`legacy_rejection_v1` \| `position_coupled_v1`） |
| `--version`、`--help`（`-h`） | — | 輸出結構化版本字串或使用說明 |

---

## 4. 搭配 `oMLX-extend` 執行（Sidecar 共存）

本單一儲存庫在 `omlx-extend/` 旁直接追蹤已套用修補檔的 `omlx/` 宿主程式碼，並透過 [`omlx-extend/patches/`](../../../omlx-extend/patches/) 與上游 `jundot/omlx` 保持同步。

```bash
# Check the Provider SPI patch against a separate clean checkout of the pinned upstream tag
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir /path/to/upstream-omlx --check-only

# Sync the vendored ./omlx tree to the latest upstream jundot/omlx release
./omlx-extend/scripts/sync_omlx.sh
```

安裝完成 `io.ninfer.metal/current/` 後，`oMLX-extend` 會自動偵測 **`miv 核心後端`**（`mseries_backendd`，依序退回搜尋 `com.omlx.mseries/current/`、`PATH` 或 `build/*/miv/`），並於 **整合 → 外部引擎**（`Integrations → External Engines`）顯示已安裝的路徑 A 與路徑 B 微核心卡片。只要模型目錄含 `model.miv` / `model.omlxm5`、`model_pack.json` 或 `omlxm5-manifest.json`，系統即會在首次請求時延遲路由至 **`miv 核心後端`**，未適配模型維持原版 MLX 執行。若要讓已適配模型強制改回原版 MLX，在 **模型設定 → 進階設定** 關閉 **`Use External Engine`** 開關（寫入 `external_engine: "stock"`）即可。

macOS 應用程式畫面圖文教學見 **[`oMLX-extend` 視覺化上手指南](omlx-extend-guide.md)**；宿主內部機制與 UI 設定對應詳見 [`omlx-extend/README.md`](../../../omlx-extend/README.md)。

---

## 5. 設定 `ninfer-metal.toml`

獨立伺服器 `ninfer-metal-serve` 與 `oMLX-extend` 開箱共用同一份預設設定檔 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)，不需手動撰寫。未明確傳入 `--config` 時，兩者皆依相同優先順序由高至低尋找第一個存在的檔案：

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml`（可重定位發行套件）
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

如需跨版本保留自訂預設值，將預設設定檔（或註解範本 [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)）複製至 `~/.config/ninfer-metal/` 後修改：

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

在 `oMLX-extend` Sidecar 模式下，介面上的個別模型設定（`external_engine_options["ninfer-metal"]`）優先於 TOML 的 `[sidecar.models."<name>"]`，再優先於共用 `[engine]` / `[speculative]` 預設值。在獨立 `ninfer-metal-serve` 模式下，CLI 旗標一律覆寫 TOML。

### 核心參數

| 鍵名 | 區段 | 選項值（`miv-v0.01`） | 說明 |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`、`8080`、`""` | 獨立 `ninfer-metal-serve` 設定（`oMLX-extend` AF_UNIX Sidecar 模式忽略）。 |
| `providers_dir` / `memory_budget_gb` | `[server]` | 字串或陣列，浮點數（GiB） | 路徑 B SPI v2 微核心根目錄（掃描 `<dir>/*/current/provider.json`）與伺服器總 UMA 上限（預設實體記憶體的 `81%`）。 |
| `model_dir` / `artifact`（`artifact_path`） | `[model]` | 路徑 | 獨立模式模型目錄與選用的明確 `.miv`（或對應 Provider）權重路徑；`artifact` 與 `artifact_path`（以及 `draft_artifact`/`draft_artifact_path`、`metallib_dir`/`kernel_dir`）皆受支援。 |
| `id` / `model_dir` / `provider` / 覆寫項 | `[[models]]` | 表格陣列 | 獨立 `ninfer-metal-serve` 多模型平行託管項目。未指定項目沿用 `[engine]`/`[speculative]` 預設值，並以 `provider = "auto" \| "inproc" \| "<provider_id>"` 選擇後端。 |
| `max_concurrency` | `[engine]` | `1..8`（純 CLI 預設 `1`，範本 `4`） | 多使用者並發批次槽位（`miv` 文件記為 **`B`**，上游 `ninfer` 記為 **`C`**）。Sidecar 模式下省略時沿用 oMLX 的 `max_concurrent_requests`。 |
| `max_context` | `[engine]` | `32768`–`262144+`（純 CLI 預設 `32768`，範本 `65536`） | 上下文視窗長度（**`L`**）。Sidecar 模式下對應 oMLX `max_context_window`。 |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128` 僅佔 `0.5625 B/val`，並於 `64K+` 通過 `0-ULP` 驗證。`miv-v0.01` 不包含 `int8_g64`。 |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | 啟用多輪滾動 UMA 前綴重用與 `1×` 原地祖先覆寫。 |
| `prefix_cache_gb` | `[engine]` | `0`–`128`（範本 `4.0`） | 前綴快取與 UMA 檢查點池預算（GiB）。 |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`，`0`（自動）或 `1..7` | `"auto"` 偵測到配套 `dflash2/` 時選 `dflash2`，其次 `mtp`，否則 `none`。Sidecar 模式下對應 oMLX 加速開關（`mtp_*`、`dflash_*`）。 |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"`（純 CLI 預設） \| `"position_coupled_v1"` | 推測驗證取樣契約。 |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | Prefill 執行後端選擇。 |

預設設定檔：[`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) · 完整註解範本：[`omlx-extend/config/ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)。

---

## 6. 底層 CLI、常駐程式與 C++ 函式庫

除 `ninfer-metal-serve` 與 `oMLX-extend` 之外，`miv/` 另提供四個底層整合介面：

1. **`m5rt_runner`（Token ID 命令列工具）** — 直接接受整數 Token ID，執行單流、批次（`--batch-mode on --max-concurrency 4`）或 DFlash2 推測推論，並以 NDJSON 輸出指標與生成 Token：
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.miv \
     --config /path/to/config.json \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd`（AF_UNIX NDJSON RPC v2 常駐程式）** — 常駐 Token ID 引擎服務，支援 `hello`、`load`、`submit`、`cancel`、`subscribe_events`、`trim_cache`、`set_scheduler_hint` 與 `shutdown`：
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a`（C++20 靜態函式庫）** — 透過 [`miv/include/omlx_m5/engine.h`](../../../miv/include/omlx_m5/engine.h) 直接嵌入 `omlx::m5::Engine`。
4. **`omlx_m5_native`（選用 Python 模組）** — 編譯時開啟 `-DOMLX_M5_BUILD_PYTHON=ON` 產生的 Nanobind 繫結。
