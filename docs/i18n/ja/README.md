<p align="center"><img src="../../assets/logo.png" width="140" alt="ninfer-metal" /></p>

<h1 align="center">ninfer-metal</h1>

<p align="center">
  <strong>Apple M5+ 向けネイティブ Metal 4 NAX &amp; Compute ICB LLM 推論エンジン（v1 コアコードネーム <code>miv</code>）</strong>
</p>

<!-- Replace scoltzero/ninfer-metal with final GitHub slug if changed -->
<p align="center">
  <a href="https://github.com/scoltzero/ninfer-metal/stargazers"><img src="https://img.shields.io/github/stars/scoltzero/ninfer-metal?style=flat-square&logo=github" alt="GitHub Stars" /></a>
  <a href="https://github.com/scoltzero/ninfer-metal/releases"><img src="https://img.shields.io/github/downloads/scoltzero/ninfer-metal/total?style=flat-square" alt="GitHub Downloads" /></a>
  <a href="miv-v0.01-changelog.md"><img src="https://img.shields.io/badge/release-miv--v0.01-0969da?style=flat-square" alt="Latest Release miv-v0.01" /></a>
  <img src="https://img.shields.io/badge/platform-Apple%20M5%2B%20%C2%B7%20macOS%2027%2B-111827?style=flat-square" alt="Platform Apple M5+ macOS 27+" />
  <img src="https://img.shields.io/badge/backend-Metal%204%20NAX-10b981?style=flat-square" alt="Backend Metal 4 NAX" />
  <a href="../../../LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square" alt="License GPL-3.0" /></a>
</p>

<div align="center">

[English](../../../README.md) | [简体中文](../zh-CN/README.md) | [繁體中文](../zh-TW/README.md) | **日本語** | [한국어](../ko/README.md) | [Español](../es/README.md) | [Français](../fr/README.md) | [Deutsch](../de/README.md)

</div>

---

**`ninfer-metal`** は、ハードウェア **Neural Accelerators (NAX)** と **Compute ICB** を備えた Apple Silicon（`M5 / M5 Pro / M5 Max / M5 Ultra`、`MTLGPUFamilyApple10+`、`macOS 27+`）向けの純 C++20 / Objective-C++ / Metal 4 推論エンジンである。

- **Python 非依存・MLX 非依存**: ネイティブの `ninfer-metal-serve` および `mseries_backendd` を含む 16 本の Mach-O バイナリと 38 本の事前コンパイル済み `.metallib` シェーダを収録する。リンク先は macOS システムフレームワークのみであり、Xcode・Python・Homebrew dylib のないクリーンな Mac でそのまま動作する。
- **単体 OpenAI サーバー（`ninfer-metal-serve`）**: 単一の C++20 バイナリでインプロセス `miv` と並列ルート B SPI v2 マイクロカーネルを 1 つの HTTP/SSE ポート（`/v1/chat/completions`、`/v1/completions`、`/v1/models`、`/health`、`/metrics`）にホストし、C++ BPE トークナイザ、Jinja チャットテンプレート、`<think>` の `reasoning_content` 分離、Qwen XML/JSON `<tool_call>` パーサを内蔵する。Open WebUI、Cursor、Cline、NewAPI、各種 OpenAI SDK からそのまま接続できる。
- **単一常駐 NAX ウェイト & `0-ULP` 完全一致**: オフライン NAX プリタイリング（`.miv`）により、Prefill・`T=1` AR デコード・投機検証のすべてを単一の常駐ウェイトコピー（成果物サイズ `1.0×`）から供給する。ベースライン **`miv-v0.01`** は Qwen3.5 / Qwen3.6 / Qwen3.8 ハイブリッドモデル（Dense 27B および 35B-A3B MoE）を `MXFP4` および `oQ` Affine（`q3–q8`）で実行し、`affine4_g128` KV キャッシュ、Native MTP / DFlash2、マルチユーザーバッチ処理（`B/C = 1..8`）、ローリング UMA プレフィックス再利用に対応する。
- **ドロップイン `oMLX-extend` 共存**: AF_UNIX Provider SPI v2 サイドカー経由で **`miv Core Backend`** を公開する `oMLX-extend`（`jundot/omlx` `v0.8.4` ベースのソースコード）を同梱する。変換済み `.miv` モデルは O(1) で自動ルーティングされ、未変換モデルは Stock MLX 上でそのまま動作し、サイドカーモデルと MLX モデルは単一のメモリ上限のもとで UMA を共有する。

---

## パフォーマンス概要

![シングルストリームデコードおよび 0-ULP 投機デコードの高速化](../../assets/charts/decode-speedup.svg)

![マルチユーザー並列スケーリングにおける合計スループット](../../assets/charts/concurrency-scaling.svg)

物理 Apple M5 Ultra（`80-GPU / 256 GiB` および `64-GPU / 96 GiB`）上の独立クリーンサブプロセスにおいて、標準 `oMLX 0.8.0`（`mlx 0.32.4`、NAX 有効）と `0-ULP` Greedy 一致条件で計測した。全 4 枚のチャートと Dim A–F の実測値は **[`miv-v0.01` リリースノート](miv-v0.01-changelog.md)** に掲載している。

---

## 対応モデルとマイクロカーネル

| モデルファミリー | 量子化 | マイクロカーネル | 形態 | クレジット / 出典 |
|---|---|---|---|---|
| **Qwen3.5 / 3.6 / 3.8 Dense**（0.8B – 27B、3:1 GDN+GQA） | `MXFP4`, `oQ4e` | `miv-v0.01` | ルート A（メインライン） | `ninfer-metal` コア |
| **Qwen3.5 / 3.6 Hybrid MoE**（256×8 + 共有、例：35B-A3B） | `MXFP4`, `oQ4e` | `miv-v0.01` | ルート A（メインライン） | `ninfer-metal` コア |
| *Llama, Qwen3 (Pure-Attn), DeepSeek MLA, Gemma, GLM, MiniMax, ...* | *募集対象* | *PR 募集中* | ルート A / ルート B | **[Microkernel Foundry へ参加 →](microkernel-foundry.md)** |

コントリビュータやコーディングエージェントは **[Microkernel Foundry](microkernel-foundry.md)** を通じて対応モデルを拡張できる：

- **ルート A（メインライン PR）**：`ninfer-metal` の共有 Metal 4 NAX 基盤へ直接マージし、本一覧に**コントリビュータクレジット**を記載する。
- **ルート B（並列サイドカー）**：準備完了後に PR または Issue を提出すれば、[`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json)（`sync_microkernels.sh`）へ登録し、本一覧に**オリジナルリポジトリへのリンク**を掲載する。`oMLX-extend` と単体 `ninfer-metal-serve --providers-dir` の双方が `Providers/*/current/provider.json` を自動検出し、ルート B マイクロカーネルを `miv`（`oMLX-extend` では標準 MLX とも）と並列実行する。

---

## クイックスタート

ゼロ MLX の `miv/` コアをビルドする（`macOS 27+`、`Xcode 27+`）：

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cmake -S . -B build/miv-v001 -G Ninja -DCMAKE_BUILD_TYPE=Release -DOMLX_M5_BUILD_NAX_METALLIB=ON -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF -DOMLX_M5_BUILD_PYTHON=OFF
cmake --build build/miv-v001
```

単体 C++ OpenAI サーバーを 1 コマンドで起動する（`oMLX-extend` と共有する [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) のデフォルト設定を自動読み込み）、または `oMLX-extend` サイドカー層を検証する：

```bash
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test
```

ビルドフラグ一式、`.miv` ウェイト変換、`ninfer-metal-serve --providers-dir` / `[[models]]` 並列設定、および `oMLX-extend` ホットアップデート手順は **[クイックスタートと設定](getting-started.md)** を参照。macOS アプリの画面付きガイドは **[`oMLX-extend` ビジュアルガイド](omlx-extend-guide.md)** を参照。

---

## ドキュメント一覧

| ガイド | 内容 |
|---|---|
| **[`oMLX-extend` ビジュアルガイド](omlx-extend-guide.md)** | macOS 版 `oMLX` の 5 ステップ画面ガイド：マイクロカーネルカード、アクティブエンジンバッジ、自動ルーティング / Stock MLX 切替、MTP / DFlash2 & 4-bit KVQuant、ワンクリック共有設定 |
| **[クイックスタートと設定](getting-started.md)** | 動作要件、ゼロ MLX CMake ビルド、ワンクリック `start_ninfer_serve.sh`、`ninfer-metal-serve` 単体サーバー、`oMLX-extend` サイドカー構築、`ninfer-metal.toml`、低レベル CLI / デーモンの使い方 |
| **[アーキテクチャと共存設計](architecture.md)** | デュアルモード `miv/` コア（`ninfer-metal-serve` + `mseries_backendd`）、Provider SPI v2、O(1) 自動ルーティング、UMA メモリ協調、アトミック symlink ホットアップデート |
| **[Microkernel Foundry](microkernel-foundry.md)** | Silicon-Squeeze Skill を用いた M5+ 新モデルファミリー向けマイクロカーネル（Llama、DeepSeek、Gemma、GLM、MiniMax）の貢献ガイド |
| **[リリースノート（`miv-v0.01`）](miv-v0.01-changelog.md)** | Dim A–F ベンチマークチャート、`ninfer-metal-serve` & `oMLX-extend` 変更点、テスト & `otool -L` 監査結果、対応範囲 |
| **[ロードマップ](roadmap.md)** | `miv-v0.01` 以降の計画マイルストーン |
| **[ドキュメント索引](../../README.md)** | 全 8 言語ドキュメントミラー、ハードウェアガイド、SKU キャリブレーション（`64-GPU` / `80-GPU`）、バックエンド仕様、認証基準 |

---

## 謝辞

- **[`jundot/omlx`](https://github.com/jundot/omlx)** — `oMLX-extend` の拡張基盤となるアップストリーム oMLX 推論サーバーおよび macOS アプリ。
- **[`Neroued/ninfer`](https://github.com/Neroued/ninfer)** — 明示的メモリ所有権と決定論的スケジューリング設計のリファレンスとなるネイティブ推論エンジン。
- **[`z-lab/dflash`](https://github.com/z-lab/dflash)** および **[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx)** — DFlash / DFlash2 ブロック拡散投機デコードと Apple Silicon 向けドラフトモデルエコシステム。
- **[`SqueezeAILab/KVQuant`](https://github.com/SqueezeAILab/KVQuant)** および **[`FutureMLS-Lab/FOSCAR`](https://github.com/FutureMLS-Lab/FOSCAR)** — 低ビット KV キャッシュ量子化研究およびアフィン畳み込みデコードアテンション（`miv` の `affine4_g128` Metal 4 NAX カーネルの基盤）。
- **[`p-e-w/heretic`](https://github.com/p-e-w/heretic)** および **[`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)** — アップストリームの学習不要モデル改変フレームワーク Heretic と、その Apple Silicon MLX ネイティブ移植およびコミュニティ量子化ウェイト。
