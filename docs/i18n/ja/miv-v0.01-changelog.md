[English](../../releases/miv-v0.01-changelog.md) | [简体中文](../zh-CN/miv-v0.01-changelog.md) | [繁體中文](../zh-TW/miv-v0.01-changelog.md) | **日本語** | [한국어](../ko/miv-v0.01-changelog.md) | [Español](../es/miv-v0.01-changelog.md) | [Français](../fr/miv-v0.01-changelog.md) | [Deutsch](../de/miv-v0.01-changelog.md)

# `ninfer-metal` `miv-v0.01` リリースノート

> **リリース**：`ninfer-metal` `miv-v0.01`（`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`）
> **対応ホスト**：`oMLX-extend` `omlx-extend-v0.8.3`（`jundot/omlx` `v0.8.3` 向け Provider SPI v2 拡張）
> **動作環境**：Apple M5 / M5 Pro / M5 Max / M5 Ultra（`MTLGPUFamilyApple10+`）、macOS 27+（実行時は Xcode 不要、ソースビルドには Xcode 27+ が必要）

`miv-v0.01` は、Apple M5+ の Neural Accelerators（NAX）と Compute ICB 上に純粋な C++20・Objective-C++・Metal 4 で構築した `ninfer-metal` の初回公開リリースであり、Python や MLX のランタイム依存を持たない。**Qwen3.5 / Qwen3.6 / Qwen3.8 ハイブリッド**モデル（Dense 0.8B–27B および 35B-A3B などの 256×8 MoE）に対応し、ネイティブ OpenAI HTTP/SSE サーバー（`ninfer-metal-serve`）による単体運用のほか、`oMLX-extend` 内の **`miv Core Backend`** として Stock MLX と共存しながら OpenAI・Anthropic エンドポイントおよびネイティブ macOS GUI を利用できる。導入手順と CLI フラグは [クイックスタートと設定](getting-started.md)、macOS アプリの操作手順は [`oMLX-extend` ビジュアルガイド](omlx-extend-guide.md) を参照。

## 1. Apple M5 Ultra 実機ベンチマーク

以下はすべて物理 Apple M5 Ultra 上で独立サブプロセスを用いて実測した値であり、同一モデルウェイト・同一トークン列・`temperature=0`・`phys_footprint` メモリ計測の条件で同機上の標準 `oMLX 0.8.0`（`mlx 0.32.4`、NAX 有効）と比較している。80-GPU / 256 GiB SKU（`macOS 27.2.1` Build `26B131`、`Xcode 27.2` Build `27B52`）と 64-GPU / 96 GiB SKU の結果は分けて記載する。

### シングルストリームデコードと `0-ULP` 投機高速化

![シングルストリームデコードと 0-ULP 投機高速化](../../assets/charts/decode-speedup.svg)

- 80-GPU M5 Ultra の純 AR デコードは検証した全ウェイト形式で標準 oMLX を上回る：`Qwen3.8-27B MXFP4` で `64.63 tok/s`（`+14.5%`）、`27B oQ4e` で `54.03 tok/s`（`+7.3%`）、`Qwen3.6-35B-A3B MXFP4` で `216.44 tok/s`（`+23.7%`）。
- 投機デコードにより、`27B` コーディング生成は `DFlash2` で `186.74 tok/s`（AR 比 `2.89×`）、`Native MTP` で `140.48 tok/s`（標準 oMLX MTP の `103.28 tok/s` 比 `+36.0%`）に達し、`128/128` の `0-ULP` Greedy 一致を維持する。
- 64-GPU / 96 GiB SKU では、`27B oQ4e` が Coding で `138.92 tok/s`（`DFlash2`、`+33.9%`）と `107.52 tok/s`（`MTP`、`+21.2%`）、Reasoning で `110.01` / `100.28 tok/s` を記録した（`6/6` セット、`256/256` `0-ULP`）。

### マルチユーザー並列スケーリング

![マルチユーザー並列スケーリング合計スループット](../../assets/charts/concurrency-scaling.svg)

- 80-GPU SKU で計測した全 19 件の並列条件すべてで `miv` が標準 oMLX を上回る。`B=4` の `27B` は `500.57 tok/s`（`oQ4e DFlash2`）、`496.60 tok/s`（`MXFP4 DFlash2`）、`434.92 tok/s`（`oQ4e MTP`）に到達し、標準 oMLX（`148.37` MTP / `129.80` AR）の `2.93×–3.83×` で動作する。
- `Qwen3.6-35B-A3B` は `B=8` 純 AR でウィンドウ `846.29 tok/s`（エンドツーエンド `794.39 tok/s`）までスケールし、標準 oMLX 比 `+52.6%` のスループットと `4.16×` 高速な TTFT を達成する。
- すべての並列スロットで、シングルストリーム実行に対する `128/128 × 4` の `0-ULP` Greedy 一致を保持する。

### 64K 長文コンテキスト安定性・UMA 占有量・プリフィル

![64K 長文コンテキスト安定性と affine4_g128 UMA 占有量](../../assets/charts/long-context-curve.svg)

- `affine4_g128` KV を有効にした `27B MXFP4` は、`2K` から `64K` までの AR 速度低下が `-4.1%` にとどまる（`61.42 → 58.92 tok/s`、oMLX 比 `+28.6%`）。実英文＋コード混在ワークロードの `DFlash2` も `64K` で `139.70 tok/s` を維持し、`2K–64K` 全域で `0-ULP` を確認した。
- `affine4_g128` は `64K` KV キャッシュを `71.8%` 削減し（`4.00 → 1.13 GiB`）、単独リクエストのピーク UMA を `15.69 GiB` に抑える（oMLX 比 `-3.44 GiB`、4 段階のプレフィックス同時常駐時は `-10.06 GiB`）。ウォーム `64K + 32` TTFT は `176.19 ms`（`bf16`）および `201.42 ms`（`affine4`）へ短縮し、oMLX 比 `7.05×–9.48×` 高速となる。
- プリフィルは 80-GPU SKU で `1K` 時 `2,058.5–2,110.6 tok/s`（`100.25–102.80 TF`）、`64K` コールド時 `41.02–41.84 s` を記録。64-GPU SKU の `4K` プリフィルは 15 条件すべてで優位に立ち、`35B-A3B` で `10,048.5 tok/s`（`+15.9%`）に達する。

### マルチターン会話とローリング・プレフィックス再利用

![マルチターン短会話のローリング・プレフィックス再利用と UMA フラット性](../../assets/charts/multiturn-rolling.svg)

- ローリング UMA プレフィックス再利用により、マルチターンヒット率は `92%` 超（`B=1` で `92.33%`、`B=4` で `92.02%`）へ向上し、再プリフィル対象トークンを半減させ、ウォーム TTFT を `2.44×–2.94×` 高速化する（`B=1` で `101.25 ms`、`B=4` で `282.80 ms`）。
- インプレース Slab 再利用によって `B=1` 10 ターンの UMA 増加量を `+0.02 MB`・エビクション 0 件に抑え（oMLX は `affine4` で `+397.5 MB`、`bf16` で `+704.8 MB` 増加）、10 ターン合計所要時間を `16.99 s` から `9.26 s` へ短縮する（`DFlash2`）。
- `B=4`（4 ユーザー × 6 ターン）では、バッチ化 `affine4` NAX デコードが `176.95 tok/s`（`AR`）と `206.35 tok/s`（`DFlash2`、oMLX の `1.72×`）を達成し、ピーク UMA を `2.12 GiB` 削減。`B=1`（`10/10`）および `B=4`（`24/24`）の全ターンで `0-ULP` を検証済み。

## 2. `miv-v0.01` の主な同梱機能

### 単体 OpenAI サーバー（`ninfer-metal-serve`）
- `/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`、Prometheus `/metrics` を提供する単一 C++20 HTTP/SSE バイナリ。同期 JSON、SSE ストリーミング（`usage`、`[DONE]`）、任意の Bearer 認証（`--api-key`）、CORS、グレースフルシャットダウンに対応する。
- C++ 製 BPE トークナイザー（`tokenizer.json`）、Jinja チャットテンプレート描画、`<think>` `reasoning_content` 分離、同期・ストリーミング両対応の Qwen XML/JSON `<tool_call>` パーサーを内蔵する。
- インプロセス `miv` と外部ルート B Provider SPI v2 サイドカーを単一ポートで同居ホストし（`--providers-dir`、`[[models]]`）、共有 UMA メモリ上限（デフォルトは物理 RAM の `81%`）と競合認識スケジューリングで統制する。
- そのまま起動できる [`ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)（`oMLX-extend` と自動共有）およびワンコマンド起動スクリプト `./omlx-extend/scripts/start_ninfer_serve.sh` を同梱する。

### Metal 4 NAX 推論エンジン（`miv/`）
- MLX や Python ヘッダーを一切含まない自己完結型 C++20 / Objective-C++ / Metal 4 ソースツリー。NAX プリフィル、融合デコード、256×8 MoE ルーティング、投機検証、4-bit Split-KV を網羅する 38 個のプリコンパイル済み `.metallib` シェーダーを収録する。
- Apple M5 Ultra の両 SKU 向け専用ハードウェアキャリブレーションプロファイル（[80-GPU / 256 GiB](../../backend/silicon-calibrations/apple-m5-ultra-80gpu.json) および [64-GPU / 96 GiB](../../backend/silicon-calibrations/apple-m5-ultra-64gpu.json)）を内蔵し、split-K・Split-KV 分割数・ドラフト深度を自動最適化しつつ環境変数による上書きを優先する。
- ゼロコピー GPU プレフィックス継続（`64K` 復元 `0.73–1.62 ms`）、インプレースのローリング UMA プレフィックス再利用、並列セッション向けバッチ化 `affine4_g128` NAX アテンションを実装する。

### `oMLX-extend` `omlx-extend-v0.8.3`（macOS アプリ＆サイドカー）
- `AF_UNIX` サイドカー（`mseries_backendd`）経由で **`miv Core Backend`** を登録。変換済み `.miv` モデルを O(1) で自動ルーティングし、標準 MLX モデルは余分なプロセスを起動せず Stock MLX のまま実行する。
- ネイティブ macOS UI の `Integrations → External Engines` に `miv Core Backend` とルート B マイクロカーネルのカードを追加し、モデルごとのエンジンバッジ、Auto / Stock MLX 切替、バックエンド能力に応じた `64K+` コンテキスト・`MTP`・`DFlash2`・`affine4_g128` KV キャッシュ設定を提供する。
- サイドカーと MLX モデル間で UMA 予算・アイドルキャッシュ回収・競合認識チャンクプリフィルを協調制御し、`~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/` のアトミックなシンボリックリンク更新に対応する。

## 3. クリーン Mac パッケージングと検証

- リリースアーカイブ `ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` には、16 個の Mach-O バイナリ（`ninfer-metal-serve`、`mseries_backendd`、各種 CLI・キャリブレーションツール）、38 個の Metal 4 `.metallib` シェーダー、2 種の SKU キャリブレーションプロファイル、`provider.json`、SHA-256 チェックサムを収録する。
- 全 16 バイナリに対する `otool -L` 検査で、リンク先が Apple システムフレームワーク（`CoreFoundation`、`Foundation`、`IOKit`、`Metal`）とシステム dylib（`libc++.1`、`libSystem.B`、`libobjc.A`）のみであることを確認済み。Homebrew・Xcode・MLX・OpenSSL・Python のランタイム依存はゼロである。
- エンドツーエンドのサーバーテスト（同期/SSE、`<think>`/`<tool_call>`、`B=4` バッチ、インプロセス＋サイドカー並列ホスティング）、`638 / 638` の `oMLX-extend` pytest、および `75 / 75` の macOS XCTest がすべて合格し、Provider SPI パッチも `jundot/omlx` `v0.8.3` へ競合なく適用できる。

## 4. サポート範囲

| 項目 | `miv-v0.01` サポート範囲 |
|---|---|
| ハードウェア | Apple Silicon `MTLGPUFamilyApple10+`（Apple M5 / M5 Pro / M5 Max / M5 Ultra、64-GPU & 80-GPU） |
| OS・ツールチェーン | macOS 27+（実行時は Xcode 不要、ソースビルドは Xcode 27+ / macOS 27 SDK） |
| モデル | **Qwen3.5 / Qwen3.6 / Qwen3.8 ハイブリッド**（3:1 GDN + GQA、Dense 0.8B–27B および 35B-A3B などの 256×8 MoE） |
| ウェイト形式 | オフライン NAX 事前タイル化 `.miv`：`MXFP4`（`E2M1 × UE8M0`）および `oQ` Affine（`q3–q8`、`g64 / g128`） |
| KV キャッシュ | `bf16` および `affine4_g128`（`0.5625 B/val`、`2K–64K` 全域で `0-ULP` 検証済み） |
| 投機デコード | 純 AR（`none`）、内蔵 `Native MTP`、`DFlash2`（シングルストリームおよび `B=2..4` バッチ） |
| サービングインターフェース | `ninfer-metal-serve` の OpenAI HTTP/SSE エンドポイント、`oMLX-extend` の OpenAI + Anthropic エンドポイントおよび macOS GUI |
