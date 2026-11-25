[English](../../getting-started.md) | [简体中文](../zh-CN/getting-started.md) | [繁體中文](../zh-TW/getting-started.md) | **日本語** | [한국어](../ko/getting-started.md) | [Español](../es/getting-started.md) | [Français](../fr/getting-started.md) | [Deutsch](../de/getting-started.md)

# クイックスタートと設定

ゼロ MLX の `ninfer-metal` コア（`miv/`）のビルド、`.omlxm5` へのウェイト変換、単体 C++20 OpenAI サーバー（`ninfer-metal-serve`）のワンクリック起動、**`miv Core Backend`** の `oMLX-extend` 連携、`ninfer-metal.toml` のカスタマイズ、および低レベル CLI・デーモンの呼び出し手順をまとめる。`oMLX` macOS アプリを利用する場合は、まず 5 ステップの画面ガイドである **[`oMLX-extend` ビジュアルガイド](omlx-extend-guide.md)** から参照されたい。

---

## 動作要件

| モード | ハードウェア | OS・ツールチェーン |
|---|---|---|
| **ランタイム（ビルド済みバンドル）** | Apple M5 / M5 Pro / M5 Max / M5 Ultra（`MTLGPUFamilyApple10+`） | `macOS 27+`（Xcode・Python・MLX・Homebrew 非依存） |
| **ソースからビルド** | Apple Silicon（`arm64`） | `macOS 27+`、`Xcode 27+`（macOS 27 SDK）、`CMake 3.27+`、`Ninja` |

---

## 1. `ninfer-metal` のビルド（`miv/`、ゼロ MLX / ゼロ Python）

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

オプションの Python 拡張 `omlx_m5_native` も同時にビルドする場合は、`-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>` を指定する。開発用プリセット一式は [`CMakePresets.json`](../../../CMakePresets.json) にある。

---

## 2. パッケージ化とウェイト変換

### クリーン Mac 向けリリースバンドル

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

このコマンドは `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz`（および `.sha256`、`provider.json`、`latest.json`）を生成し、全 16 本の Mach-O バイナリ（`ninfer-metal-serve`、`mseries_backendd`、`m5rt_runner`、13 本のハードウェア／キャリブレーションプローブ）に対して `otool -L` を実行し、Apple システムフレームワーク（`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` および `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`）のみにリンクしていることを検証する。

macOS 27+ を搭載した Apple M5+ Mac でアーカイブを展開し、`install.sh`（既存のユーザー編集を上書きせずにデフォルトの `ninfer-metal.toml` をコピー）を実行すると以下に配置される：

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

### モデルの `.omlxm5` 変換

オフライン NAX プリタイリングにより、`MXFP4`（`E2M1 × UE8M0`）または `oQ` Affine（`q3–q8`、`g64/g128`）ウェイトを一度だけパックする。Prefill・`T=1` AR デコード・投機検証が単一の常駐メモリコピーを共有する。

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.omlxm5 \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. `ninfer-metal-serve` による単体起動（ネイティブ C++）

`ninfer-metal-serve` は `libomlx_m5_core_runtime.a` に直接リンクされた単一 C++20 バイナリである。POSIX HTTP/SSE、`ninfer-metal.toml` 読み込み、C++ BPE トークナイザ（`tokenizer.json`）、Jinja チャットテンプレート（`chat_template.jinja` / `tokenizer_config.json` / ChatML フォールバック）、`<think>` 推論出力分離、Qwen XML/JSON `<tool_call>` パーサを Python 非依存・MLX 非依存で内蔵する。

### ワンクリック起動（編集不要の共有設定）

設定ファイルを手書きせずに `ninfer-metal-serve` をすぐ起動できる。ランチャースクリプトは `ninfer-metal-serve` を自動検出し、`oMLX-extend` と共有するデフォルト設定 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)（`max_concurrency = 4`、`max_context = 65536`、`kv_dtype = "affine4_g128"`、`rolling_prefix_cache = true`、`prefix_cache_gb = 4.0`、`speculative.backend = "auto"`、`providers_dir = ["~/Library/Application Support/oMLX/Providers"]`）を読み込む：

```bash
# 特定のモデルディレクトリを指定してワンクリック起動（デフォルトポート 8080、oMLX のポート 8000 と分離）
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# モデルディレクトリを省略すると ./models または oMLX のモデルフォルダ内の最初の .omlxm5 モデルを自動選択
./omlx-extend/scripts/start_ninfer_serve.sh

# 任意の CLI フラグを追加して共有デフォルト設定を上書き
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
```

### 直接 CLI 上書きまたはカスタム `ninfer-metal.toml`

`ninfer-metal-serve` を直接起動した場合も、`--config` または `--no-default-config` を指定しない限り共有デフォルトの `ninfer-metal.toml` が自動読み込みされる。CLI フラグは常に TOML 値を上書きする：

```bash
# 明示的な上書き付きの直接 CLI 起動
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# DFlash2 投機デコードを明示指定して起動（または speculative.backend = "auto" を利用）
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# 指定した設定ファイルを読み込む（CLI フラグが TOML 値を上書き）
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### 複数マイクロカーネルの並列ホスト（`--providers-dir` / `[[models]]`）

単一の `ninfer-metal-serve` ポート上で、インプロセス `miv`（ルート A）と外部 SPI v2 マイクロカーネルサイドカー（ルート B）を並列実行できる。`--model-dir` を複数回指定するか、TOML の `[[models]]` でモデルを列挙する：

```bash
# CLI: host Path A (inproc miv) + Path B SPI v2 sidecar models on one port
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --model-dir /path/to/Llama-3.3-70B-Instruct \
  --providers-dir "~/Library/Application Support/oMLX/Providers" \
  --memory-budget-gb 48.0 \
  --port 8080
```

```toml
# ninfer-metal.toml: per-model settings with unified UMA budget arbitration
[server]
providers_dir = "~/Library/Application Support/oMLX/Providers"
memory_budget_gb = 48.0

[[models]]
id = "Qwen3.8-27B"
model_dir = "/path/to/Qwen3.8-27B"
provider = "auto"          # in-process miv (io.ninfer.metal)
kv_dtype = "affine4_g128"

[[models]]
id = "Llama-3.3-70B-Instruct"
model_dir = "/path/to/Llama-3.3-70B-Instruct"
provider = "auto"          # auto-routed via Providers/*/current/provider.json
```

### クライアント接続（Open WebUI、Cursor、Cline、NewAPI、OpenAI SDK）

OpenAI 互換クライアントのエンドポイントを `http://127.0.0.1:8080/v1` に設定する：

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

### HTTP エンドポイント（`miv-v0.01`）

| エンドポイント | 動作 |
|---|---|
| `GET /health` | 準備完了状態 JSON（`status`、`model`、`models[]`、`gpu_contended`、`memory_budget_bytes`、`version`） |
| `GET /metrics` | Prometheus テキスト出力（全体集計および `model`、`provider`、`backend_mode` ラベル別のメトリクス） |
| `GET /v1/models`、`GET /v1/models/:id` | `max_model_len`、`provider_id`、`backend_mode` 付き OpenAI モデル一覧 |
| `POST /v1/chat/completions` | JSON `"model"` によるルーティング（不明モデルは `404 model_not_found`、単一モデル時は省略可）、同期 JSON・ストリーミング SSE（`chat.completion.chunk`）、`<think>...</think>` の `reasoning_content` 分離、Qwen XML / JSON `<tool_call>` 解析、`stop`、`usage`、`stream_options.include_usage`、`[DONE]` |
| `POST /v1/completions` | 生テキスト補完（JSON `"model"` ルーティング、同期 JSON、ストリーミング SSE、`usage`、`[DONE]`） |

任意の Bearer 認証（`--api-key`）は不一致時に `401` を返す。CORS プリフライト（`OPTIONS`）はデフォルトで有効である。

### `ninfer-metal-serve` CLI フラグ

`--config` を省略した場合、`ninfer-metal-serve` は [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)（`max_concurrency = 4`、`max_context = 65536`、`kv_dtype = "affine4_g128"`、`rolling_prefix_cache = true`、`prefix_cache_gb = 4.0`、`speculative.backend = "auto"`）を自動読み込みする。以下の素のコンパイル時デフォルトから開始する場合は `--no-default-config` を渡す。コメント付き完全リファレンスは [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example) を参照。

| フラグ | 素の CLI デフォルト / 値 | 説明 |
|---|---|---|
| `--config PATH` | 自動検出 | 明示的な `ninfer-metal.toml`（`[server]`、`[model]`、`[[models]]`、`[engine]`、`[speculative]`）を読み込む。CLI フラグが TOML 値を上書きする |
| `--no-default-config` | オフ | `ninfer-metal.toml` の自動検出をスキップし、素のコンパイル時デフォルトから起動する |
| `--model-dir PATH` | 必須（または TOML 指定） | `config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja`、`model.omlxm5` を含むモデルディレクトリ。複数回指定で単一ポート並列ホスト |
| `--providers-dir PATH` | 自動（`$OMLX_PROVIDERS_DIR`、`~/Library/Application Support/oMLX/Providers`） | ルート B SPI v2 マイクロカーネルサイドカーの検索ルート（複数回指定可）。`<dir>/*/current/provider.json`（フォールバック `<dir>/*/provider.json`）を走査する |
| `--provider MODE` | `auto`（`auto` \| `inproc` \| `<provider_id>`） | バックエンド選択：O(1) 自動ルーティング（`auto`）、インプロセス `miv`（`inproc`）、または明示 SPI v2 `<provider_id>` |
| `--metallib-dir PATH`（エイリアス：`--kernel-dir`） | 自動検出 | Metal 4 / NAX コンパイル済み `.metallib` シェーダディレクトリ（`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`）。Metal シェーダ用であり、SPI v2 マイクロカーネルプロバイダではない |
| `--artifact PATH` | 自動検出 | 対象 `.omlxm5`（またはプロバイダ成果物）の明示パス |
| `--draft-dir PATH`、`--draft-artifact PATH`、`--mtp-artifact PATH` | — | `dflash2` / `mtp` 用ドラフトモデルディレクトリまたは `.omlxm5` アーティファクト |
| `--chat-template PATH`、`--model-id NAME` | 自動 | Jinja チャットテンプレートファイルまたは公開モデル ID の上書き |
| `--host ADDR`、`--port PORT` | `127.0.0.1`、`8080` | HTTP バインドアドレスおよびポート |
| `--api-key KEY`、`--cors` / `--no-cors` | `""`、有効 | `Authorization: Bearer <KEY>` の要求、CORS ヘッダ制御 |
| `--memory-budget-gb FLOAT` | 自動（物理 RAM の `81%`） | `MultiModelHost` が全モデル間で調停するサーバー全体の UMA メモリ予算上限（GiB） |
| `--max-concurrency N`（`-c`、`-b`） | `1`（`1..8`） | マルチユーザー同時バッチスロット数（`B`/`C`） |
| `--max-context N` | `32768` | コンテキストウィンドウ長（`L`） |
| `--kv-dtype MODE` | `bf16`（`bf16` \| `affine4_g128`） | KV キャッシュ格納形式 |
| `--spec-backend MODE` | `none`（`auto` \| `none` \| `mtp` \| `dflash2`） | 投機デコードバックエンド（`auto` は付属 `dflash2/` があれば `dflash2`、なければ `mtp`、それもなければ `none` を有効化） |
| `--draft-tokens N` | `0`（自動、明示指定時は `1..7`） | 検証ラウンドあたりのドラフトトークン数（`0` = 自動） |
| `--rolling-prefix-cache on\|off` | ランタイム既定 | `1×` 祖先置換を伴うマルチターン・ローリング UMA プレフィックス再利用 |
| `--prefix-cache-gb FLOAT` | ランタイム既定（`0..128`） | プレフィックスキャッシュおよび UMA チェックポイントプール予算（GiB） |
| `--sampling-contract NAME` | `legacy_rejection_v1` | 投機検証サンプリング契約（`legacy_rejection_v1` \| `position_coupled_v1`） |
| `--version`、`--help` | — | 構造化バージョン文字列または使用方法の出力 |

---

## 4. `oMLX-extend` での実行（サイドカー共存）

本モノレポは、パッチ適用済みの `omlx/` ホストを `omlx-extend/` と並置して追跡する（[`omlx-extend/patches/`](../../../omlx-extend/patches/) 経由でアップストリーム `jundot/omlx` と同期）。

```bash
# Verify or apply the Provider SPI patch on ./omlx and run pytest
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test

# Sync ./omlx to a specific upstream jundot/omlx tag
./omlx-extend/scripts/sync_omlx.sh --tag v0.8.3 --omlx-dir ./omlx
```

`io.ninfer.metal/current/` がインストール済みであれば、`oMLX-extend` は **`miv Core Backend`**（`mseries_backendd`、フォールバック先：`com.omlx.mseries/current/`、`PATH`、`build/*/miv/`）を自動検出し、導入済みのルート A およびルート B マイクロカーネルを **`Integrations → External Engines`** に表示する。`model.omlxm5`、`model_pack.json`、または `omlxm5-manifest.json` を含むモデルディレクトリは初回リクエスト時に遅延で **`miv Core Backend`** へルーティングされ、未変換モデルは Stock MLX のまま動作する。変換済みモデルに対して Stock MLX を強制する場合は、**`Model Settings → Advanced Settings`** で **`Use External Engine`** をオフにする（`external_engine: "stock"` を書き込む）。

`oMLX` macOS アプリの画面付きガイドは **[`oMLX-extend` ビジュアルガイド](omlx-extend-guide.md)** を参照。ホスト内部構造および UI 設定マッピングの詳細は [`omlx-extend/README.md`](../../../omlx-extend/README.md) を参照。

---

## 5. `ninfer-metal.toml` の設定

単体 `ninfer-metal-serve` と `oMLX-extend` は、編集不要の同一デフォルト設定（[`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)）を共有する。`--config` を指定しない場合、双方のホストが同じ優先順（最初に見つかったファイルを採用）で探索する：

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml`（リロケータブル・リリースバンドル）
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

アップデート後も独自のデフォルト設定を維持するには、デフォルト設定（またはコメント付きの [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)）を `~/.config/ninfer-metal/` へコピーする：

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

`oMLX-extend` サイドカーモードでは、モデル別の UI 設定（`external_engine_options["ninfer-metal"]`）が TOML の `[sidecar.models."<name>"]` を上書きし、それがさらに共有の `[engine]` / `[speculative]` デフォルトを上書きする。単体 `ninfer-metal-serve` では CLI フラグが常に TOML 値を上書きする。

### 主要設定項目

| キー | セクション | 値（`miv-v0.01`） | 備考 |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`、`8080`、`""` | 単体 `ninfer-metal-serve` 専用設定（`oMLX-extend` AF_UNIX サイドカーでは無視） |
| `providers_dir` / `memory_budget_gb` | `[server]` | パス（文字列または配列）、GiB 浮動小数 | ルート B SPI v2 プロバイダ検索ディレクトリおよびサーバー全体の UMA 予算上限（デフォルト：物理 RAM の `81%`） |
| `model_dir` / `artifact`（または `artifact_path`） | `[model]` / `[[models]]` | パス | 単体起動時のモデルディレクトリおよび任意の明示 `.omlxm5` パス（`artifact` / `artifact_path`、`draft_artifact` / `draft_artifact_path`、`metallib_dir` の双方を受理）。`[[models]]` 配列で複数モデルを並列定義 |
| `max_concurrency` | `[engine]` | `1..8`（素の CLI 既定 `1`、TOML テンプレート `4`） | マルチユーザー同時バッチスロット数（`miv` 文書では **`B`**、アップストリーム `ninfer` では **`C`**）。サイドカーモードで省略時は oMLX の `max_concurrent_requests` を継承 |
| `max_context` | `[engine]` | `32768`–`262144+`（素の CLI 既定 `32768`） | コンテキストウィンドウ長（**`L`**）。サイドカーモードでは oMLX の `max_context_window` からマッピング |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128` は `0.5625 B/val` で格納し `64K+` で `0-ULP` 一致を検証済み。`miv-v0.01` に `int8_g64` は未搭載 |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | `1×` インプレース祖先置換を伴うマルチターン・ローリング UMA プレフィックス再利用を有効化 |
| `prefix_cache_gb` | `[engine]` | `0`–`128`（TOML テンプレート `4.0`） | プレフィックスキャッシュおよび UMA チェックポイントプール予算（GiB） |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`、`0`（自動）または `1..7` | `"auto"` は付属 `dflash2/` があれば `dflash2`、なければ `mtp`、それもなければ `none` を有効化。サイドカーモードでは oMLX Acceleration トグル（`mtp_*`、`dflash_*`）からマッピング |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"` \| `"position_coupled_v1"` | 投機検証サンプリング契約（素の CLI 既定 `"legacy_rejection_v1"`、TOML テンプレート `"position_coupled_v1"`） |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | Prefill 実行バックエンド選択 |

デフォルト設定：[`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) · コメント付き完全テンプレート：[`omlx-extend/config/ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)。

---

## 6. 低レベル CLI・デーモン・C++ ライブラリ

`ninfer-metal-serve` と `oMLX-extend` に加え、`miv/` は 4 つの低レベル統合インターフェースを提供する：

1. **`m5rt_runner`（トークン ID CLI）** — 整数トークン ID 入力からシングルストリーム、バッチ（`--batch-mode on --max-concurrency 4`）、または DFlash2 投機推論を実行し、NDJSON 形式でメトリクスとトークンを出力する：
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.omlxm5 \
     --config /path/to/config.json \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd`（AF_UNIX NDJSON RPC v2 デーモン）** — `hello`、`load`、`submit`、`cancel`、`subscribe_events`、`trim_cache`、`set_scheduler_hint`、`shutdown` に対応する常駐トークン ID エンジンサーバー：
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a`（C++20 静的ライブラリ）** — [`miv/include/omlx_m5/engine.h`](../../../miv/include/omlx_m5/engine.h) 経由で `omlx::m5::Engine` を直接組み込む。
4. **`omlx_m5_native`（オプション Python モジュール）** — `-DOMLX_M5_BUILD_PYTHON=ON` 指定時にビルドされる Nanobind バインディング。
