[English](../../microkernel-foundry.md) | [简体中文](../zh-CN/microkernel-foundry.md) | [繁體中文](../zh-TW/microkernel-foundry.md) | **日本語** | [한국어](../ko/microkernel-foundry.md) | [Español](../es/microkernel-foundry.md) | [Français](../fr/microkernel-foundry.md) | [Deutsch](../de/microkernel-foundry.md)

# Microkernel Foundry

`ninfer-metal`（`miv-v0.01`）は、Apple M5+ 上の Qwen3.5 / Qwen3.6 / Qwen3.8 ハイブリッドモデル向けにベアシリコン Metal 4 NAX 推論を提供する。**Microkernel Foundry** は、同一のランタイム基盤・サービングフロントエンド・キャリブレーションツール・認証パイプラインをコントリビュータとコーディングエージェントに開放し、メインライン PR または並列サイドカー経由で新モデルファミリーを追加できる仕組みである。

対象ファミリーには **Llama、Qwen3 Pure-Attention、DeepSeek (MLA)、Gemma、GLM、MiniMax**、および [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx) などのコミュニティ量子化派生モデルが含まれる。

---

## 2 つの貢献ルート

### ルート A：メインライン PR → コントリビュータクレジット（推奨）

共有ランタイム **`libninfer_metal_substrate`** の上にモデルファミリー向けマイクロカーネルを実装し、`ninfer-metal` 本体へ直接マージする。

**共有基盤が提供するコンポーネント：**
- Metal 4 実行ランタイム（`MetalContext`、`MTL4CommandQueue`、ダブルバッファ `ar_pipe_[2]`、`< 2 GiB` Compute ICB エンコーダ、`read_write` バリアリフレクション）。
- 共有ページ化 KV・プレフィックスメモリ（`KvPagePool`、`SharedKvArenaPool`、CoW 参照カウント、`SharedPrefixCache`、`UmaCheckpointPool`）。
- 汎用 NAX・デコードプリミティブ（`mxfp4_nax`、`affine_nax`、`bf16_nax`、`moe_nax`、ゼロ `uitofp` QMV、2 段階並列 `Argmax`、`Split-KV` / `Flash-NAX`）。
- `.omlxm5` コンテナローダ、`ExecutionIdentityFields`、`24 MHz` ハードウェアカウンタ、ネイティブ単体 OpenAI サーバー（`ninfer-metal-serve`）、Provider SPI v2 デーモン（`mseries_backendd`）。

**PR で追加する要素と実行範囲：** 対象アーキテクチャのレイヤーグラフ、ウェイトマッピング、およびコールド DRAM マイクロプローブで汎用プリミティブを **`>= +10%–12%`** 上回るアーキテクチャ固有カーネル（MLA、GeGLU、Softcap、カスタム MoE ルータなど）を追加する。`libninfer_metal_substrate` を共有することでアーキテクチャ間のコードと常駐メモリの重複を防ぎ、単体 `ninfer-metal-serve` と `oMLX-extend` の双方でネイティブ動作させ、[対応モデル一覧](README.md)にコントリビュータクレジットを記載する。

### ルート B：並列サイドカー → PR または Issue → オリジナルプロジェクトへのリンク記載

実験的アーキテクチャ、独自量子化レイアウト、または独立したリリースサイクルが必要な場合は、独立した Sidecar SPI v2 バンドルとして以下にパッケージ化する：

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

`supported_topologies`、`supported_model_types`、`artifact_extensions` を宣言した `provider.json` とともにアトミックな `current` シンボリックリンクを公開する。準備完了後に PR または Issue を提出すれば、メンテナが [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json) へ登録し、ユーザーが [`./omlx-extend/scripts/sync_microkernels.sh`](../../../omlx-extend/scripts/sync_microkernels.sh) で導入・更新できるようにしたうえで、[対応モデル一覧](README.md)にオリジナルリポジトリへの直接リンクを掲載する。

**並列ホスト対応：** **`oMLX-extend` と単体 `ninfer-metal-serve --providers-dir` の双方**が `Providers/*/current/provider.json` をネイティブに自動検出し、統一 UMA 予算調停と `gpu_contended` スケジューリングのもとで複数のルート A（**`miv Core Backend`**）およびルート B マイクロカーネルを単一ポート上で並列実行する（`oMLX-extend` では Stock MLX とも共存）。`oMLX` macOS アプリでは、導入済みのルート B マイクロカーネルが **`Integrations → External Engines`** にカードとして自動表示され、**`Models`** にアクティブエンジンバッジが表示される（[`oMLX-extend` ビジュアルガイド](omlx-extend-guide.md) 参照）。マイクロカーネル作者が実装するのは Metal 4 コアと Sidecar SPI v2 デーモンのみであり、HTTP/SSE 配信、トークン化、チャットテンプレート、マルチモデルスケジューリングは双方のホストが担う。

**ライセンス：** **ルート A** のメインライン PR は `miv` にリンクされ **[GPL-3.0](../../../LICENSE)** のもとで配布される（コントリビュータは GPL-3.0、Apache-2.0、または MIT のコードを提出可能）。**ルート B** の並列サイドカーは独立プロセスとして AF_UNIX Sidecar SPI v2 経由で通信するため、任意のライセンスを使用できる。

---

## モデルファミリー一覧

| アーキテクチャファミリー | 対象モデル例 | ステータス | 推奨ルート |
|---|---|---|---|
| **Qwen Hybrid（3:1 GDN + GQA）** | Qwen3.5 / 3.6 / 3.8 Dense（0.8B – 27B）& 35B-A3B MoE | 出荷済み（`miv-v0.01`） | ルート A（メインライン、`miv/`） |
| **Pure-Attention GQA** | Llama-3.3-70B、Qwen3-32B、Gemma | PR 受付中 | ルート A |
| **Multi-Head Latent Attention (MLA)** | DeepSeek V3 / R1 ファミリー | PR 受付中 | ルート A / ルート B |
| **MoE & コミュニティ量子化** | GLM、MiniMax、`heretic-mlx` 派生モデル | PR 受付中 | ルート A / ルート B |

---

## マイクロカーネルの作成・認証フロー

人間のエンジニアもコーディングエージェントも、再現可能な 4 ステップのループに従う：

1. **対象 Mac のキャリブレーション（`Stage B: H0 + H1`）**
   `H0` ハードウェア DMA スイープ（`MTLBlitPassDescriptor`、`4–512+ MiB`）を実行してチップ固有の $B_{\text{DRAM}}$、$C_{\text{SLC}}$、$P_{\text{NAX}}$、$I_{\text{ridge}}$ を実測し、続けて `H1` オペレータスイープを実行する。すべての合否基準はこれらの実測定数で正規化する。
2. **コールド DRAM マイクロプローブを先行実装（`~50–100` 行）**
   新カーネルファミリーをテンプレート化する前に、独立したコールド DRAM プローブ（`working set >= 3–4 × C_SLC`、`>= 30` 回の直列バリア中央値）で固定機能 NAX / ハードウェアベースラインと比較する。正味 **`>= +10%–12%`** の改善があるカーネルのみを採用し、不採用の実験的 PSO は PR 作成前に削除する。
3. **`< 10 s` 高速サブレイヤ計測 & スピル監査**
   `omlx_m5_hw_counter_probe --mode model`（`24 MHz` カウンタ）で GPU サブレイヤ内訳（`mlp/moe`、`attn`、`lm_head`、`sampling`）を計測し、`metal-objdump -d` でシェーダを監査する（`0 alloca` スピル、ワイド Prefill で `<= 104` VGPR、`T=1` プリタイル QMV で `0 uitofp`）。
4. **5 ファイル構成の機械可読エビデンスパックを提出**
   クリーンルーム Dim A–F ハーネス（`miv/tools/bench_omlx_vs_m5rt.py`）を実行し、`certification-evidence/<core-id>-<version>-<sku>/` を添付する：
   - `h0-capability.json` — ベアシリコン `H0` 定数 & 10,000 ステップ ICB カナリア
   - `h1-calibration-<sku>.json` — SKU 分離済み `H1` チューニング & `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — クリーンルーム Dim A–F 実測値 & `phys_footprint`
   - `parity-report.json` — `0-ULP` 一致、ソース PPL/KL、NIAH/RULER 再現率、`0 NaN/Inf`
   - `otool-audit.txt` — 純 Apple フレームワーク `otool -L`、`0 alloca` 監査、100 ターン連続稼働テスト & CTest 出力

---

## コントリビュータ・エージェント必読資料

- **[Silicon-Squeeze コア作成スキル](../../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — コーディングエージェントに直接読み込める M5+（`MTLGPUFamilyApple10+`）ベアシリコン最適化メソドロジー。
- **[コア認証基準](../../standards/core-certification-standard.md)** — Gate 1–4 合格基準、Provider SPI v2 契約、Roofline しきい値、Tier 1 / Tier 2 認証バッジ仕様。
- **[Apple M5 シリーズ最適化プレイブック](../../silicon-guides/m5-series-optimization-playbook.md)** — `32 KiB` スレッドグループ占有率計算、Unified Memory アドミッション式、Base / Pro / Max / Ultra スケーリング指針。
