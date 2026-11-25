[English](../../roadmap/README.md) | [简体中文](../zh-CN/roadmap.md) | [繁體中文](../zh-TW/roadmap.md) | **日本語** | [한국어](../ko/roadmap.md) | [Español](../es/roadmap.md) | [Français](../fr/roadmap.md) | [Deutsch](../de/roadmap.md)

# `ninfer-metal` ロードマップ

出荷済みベースライン：**`ninfer-metal` `miv-v0.01`**（ネイティブ C++20 `ninfer-metal-serve` 同梱）+ **`oMLX-extend` `omlx-extend-v0.8.3`**（[リリースノート](miv-v0.01-changelog.md) を参照）。

`miv-v0.01` 以降の計画開発は 3 つのトラックで進める：

---

## 1. `v0.02+` エージェントメモリ仮想化（`KVMem-UMA + ActKV`）

完全なアーキテクチャ仕様、Apple M5 UMA メモリ物理モデル、および検証ゲート V1–V6 は **[`v0.02-kvmem-actkv-plan.md`](../../roadmap/v0.02-kvmem-actkv-plan.md)** に記載している。

- **Tier-1：`KVMem-UMA + ActKV-Lite`（UMA 内の `256K → 1M+` ワークスペース）**
  - オプトイン方式（`kv_policy = "kvmem_actkv"`、`"lossless"` を恒久的な `0-ULP` デフォルトとして維持）。
  - `MTLStorageModeShared` 上の非対称コールドアーカイブ（FP8/BF16 Pre-RoPE K + `affine4_g128` V — 4 ビット二重量子化ゼロ、`1M` トークン `Qwen3.8-27B` ワークスペース全体で `25.03 GiB`）。
  - 独立 `MeanKIndexBuffer`（`< 0.6 ms` スキャン）+ Action-LRFU + 信頼度駆動ワーキングセット `W ∈ [16K, 64K]` + 単一カーネル `re-RoPE + affine4_pack` 実体化（`64K` で `~1.9–4.4 ms`）。ホットパスの `attention_affine4.metal` と静的 ICB には手を加えない。
- **Tier-2：Apple NVMe 永続化 & フル・トークンレベル ActKV**
  - 32–64 GiB の M5 Pro/Max マシンおよび `< 1 s` の再起動跨ぎ再開に向けた、`F_NOCACHE` + `dispatch_io` 上の `.omlxkv` ワークスペースバンドル。
  - Pure-Attention 32B/70B および MoE モデル向けの Base-2 LSE アテンション復元（`+96 B/tok`）と、ゼロ逆量子化インプレース `affine4_g128` ページ圧縮（`hd=256` で `144 B/tok`、`hd=128` で `80 B/tok`）。

---

## 2. 共有基盤（`libninfer_metal_substrate`）& 新モデルファミリー・マイクロカーネル

- `miv/` から Metal 4 ランタイム、`< 2 GiB` Compute ICB エンコーダ、`KvPagePool`、`SharedPrefixCache`、`UmaCheckpointPool`、汎用 NAX プリミティブを `libninfer_metal_substrate` へ抽出する。
- **[Microkernel Foundry](microkernel-foundry.md)** および **[コア認証基準](../../standards/core-certification-standard.md)** のもとで、Pure-Attention GQA（`Qwen3-32B`、`Llama-3.3-70B`）、DeepSeek MLA、Gemma、GLM、MiniMax マイクロカーネルを追加する。

---

## 3. `ninfer-metal-serve` Phase 2（Anthropic API & 文法制約デコード）

`miv-v0.01` で出荷したネイティブ C++20 OpenAI HTTP/SSE サーバーを基盤として以下を拡張する：

- **Anthropic Messages API（`POST /v1/messages`）**：Claude 互換クライアントおよびコーディングエージェント向けのネイティブ Anthropic リクエスト／レスポンス変換と SSE イベントストリーミング。
- **文法制約デコード（`xgrammar`）**：`omlx::m5::Engine` にステップ単位のロジットマスクコールバックを追加し、`xgrammar` を統合することで、外部ランタイム依存なしに厳密な JSON Schema（`response_format`）および構造化ツール呼び出し文法制約を適用する。
