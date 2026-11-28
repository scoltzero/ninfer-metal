<p align="center"><img src="../../assets/logo.png" width="140" alt="ninfer-metal" /></p>

<h1 align="center">ninfer-metal</h1>

<p align="center">
  <strong>Moteur d'inférence LLM natif Metal 4 NAX et Compute ICB pour Apple M5+ (nom de code du cœur v1 : <code>miv</code>)</strong>
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

[English](../../../README.md) | [简体中文](../zh-CN/README.md) | [繁體中文](../zh-TW/README.md) | [日本語](../ja/README.md) | [한국어](../ko/README.md) | [Español](../es/README.md) | **Français** | [Deutsch](../de/README.md)

</div>

---

**`ninfer-metal`** est un moteur d'inférence en pur C++20 / Objective-C++ / Metal 4 conçu pour Apple Silicon avec **Neural Accelerators (NAX)** et **Compute ICB** matériels (`M5 / M5 Pro / M5 Max / M5 Ultra`, `MTLGPUFamilyApple10+`, `macOS 27+`).

- **Zéro Python, zéro MLX** : Livre 16 binaires Mach-O (dont le serveur natif `ninfer-metal-serve` et le démon `mseries_backendd`) ainsi que 38 shaders `.metallib` précompilés, liés exclusivement aux frameworks système macOS — fonctionne sur un Mac vierge sans Xcode, sans Python et sans dylib Homebrew.
- **Serveur OpenAI autonome (`ninfer-metal-serve`)** : Un unique binaire C++20 hébergeant `miv` in-process et des microkernels SPI v2 Voie B en parallèle sur un même port HTTP/SSE (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`), avec tokenizer BPE en C++, modèles de chat Jinja, séparation `<think>` vers `reasoning_content` et parsing XML/JSON `<tool_call>` de Qwen — prêt pour Open WebUI, Cursor, Cline, NewAPI et les SDK OpenAI.
- **Copie unique des poids NAX et parité `0-ULP`** : Le pré-tuilage NAX hors ligne (`.miv`) alimente le Prefill, le décodage AR `T=1` et la vérification spéculative depuis une seule copie résidente en mémoire (`1.0×` la taille de l'artefact). La version de référence **`miv-v0.01`** exécute les modèles hybrides Qwen3.5 / Qwen3.6 / Qwen3.8 (Dense 27B et 35B-A3B MoE) en `MXFP4` et `oQ` Affine (`q3–q8`) avec cache KV `affine4_g128`, Native MTP / DFlash2, batching multi-utilisateurs (`B/C = 1..8`) et réutilisation glissante des préfixes en UMA.
- **Coexistence directe avec `oMLX-extend`** : Intègre `oMLX-extend` (code source basé sur `jundot/omlx` `v0.8.4`) exposant **`miv Core Backend`** via un sidecar AF_UNIX Provider SPI v2 — les modèles `.miv` adaptés sont routés automatiquement en O(1), les modèles non adaptés restent intacts sur Stock MLX, et le sidecar partage l'UMA avec MLX sous un plafond mémoire unique.

---

## Aperçu des performances

![Accélération du décodage mono-flux et spéculatif 0-ULP](../../assets/charts/decode-speedup.svg)

![Montée en charge du débit agrégé en concurrence multi-utilisateurs](../../assets/charts/concurrency-scaling.svg)

Mesuré sur matériel physique Apple M5 Ultra (`80-GPU / 256 GiB` et `64-GPU / 96 GiB`) face à `oMLX 0.8.0` standard (`mlx 0.32.4`, NAX activé) dans des sous-processus isolés avec parité gloutonne `0-ULP`. Les quatre graphiques et l'ensemble des chiffres Dim A–F figurent dans les **[Notes de version `miv-v0.01`](miv-v0.01-changelog.md)**.

---

## Modèles et microkernels pris en charge

| Famille de modèles | Quantification | Microkernel | Mode | Crédit / Source |
|---|---|---|---|---|
| **Qwen3.5 / 3.6 / 3.8 Dense** (0.8B – 27B, 3:1 GDN+GQA) | `MXFP4`, `oQ4e` | `miv-v0.01` | Voie A (Mainline) | cœur `ninfer-metal` |
| **Qwen3.5 / 3.6 Hybrid MoE** (256×8 + partagé, ex. 35B-A3B) | `MXFP4`, `oQ4e` | `miv-v0.01` | Voie A (Mainline) | cœur `ninfer-metal` |
| *Llama, Qwen3 (Pure-Attn), DeepSeek MLA, Gemma, GLM, MiniMax, ...* | *Ouvert* | *Appel aux PR* | Voie A / Voie B | **[Rejoindre Microkernel Foundry →](microkernel-foundry.md)** |

Contributeurs et agents de code peuvent enrichir ce catalogue via **[Microkernel Foundry](microkernel-foundry.md)** :

- **Voie A (PR Mainline)** : Fusionner directement sur le substrat partagé Metal 4 NAX de `ninfer-metal` — listé ici avec **crédit contributeur**.
- **Voie B (Sidecar parallèle)** : Ouvrir une PR ou une Issue dès que le cœur est prêt — nous l'inscrivons dans [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json) (`sync_microkernels.sh`), le listons ici avec un **lien vers le dépôt d'origine**, et aussi bien `oMLX-extend` que le serveur autonome `ninfer-metal-serve --providers-dir` découvrent automatiquement `Providers/*/current/provider.json` et exécutent les microkernels Voie B en parallèle avec `miv` (et avec MLX standard sous `oMLX-extend`).

---

## Démarrage rapide

Compiler le cœur `miv/` sans MLX (`macOS 27+`, `Xcode 27+`) :

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cmake -S . -B build/miv-v001 -G Ninja -DCMAKE_BUILD_TYPE=Release -DOMLX_M5_BUILD_NAX_METALLIB=ON -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF -DOMLX_M5_BUILD_PYTHON=OFF
cmake --build build/miv-v001
```

Lancer le serveur OpenAI C++ autonome en une commande (charge automatiquement les valeurs partagées de [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml), réutilisées par `oMLX-extend`) ou vérifier la couche sidecar `oMLX-extend` :

```bash
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test
```

Les options de compilation complètes, la conversion de poids `.miv`, les configurations multi-microkernel `ninfer-metal-serve --providers-dir` / `[[models]]` et la mise à jour à chaud d'`oMLX-extend` sont détaillés dans **[Démarrage et configuration](getting-started.md)** ; pour un guide illustré de l'application macOS, voir le **[Guide visuel d'`oMLX-extend`](omlx-extend-guide.md)**.

---

## Aller plus loin

| Guide | Contenu |
|---|---|
| **[Guide visuel d'`oMLX-extend`](omlx-extend-guide.md)** | Parcours en 5 étapes illustré d'`oMLX` sur macOS : cartes de microkernels, badges de moteur actif, routage auto / bascule Stock MLX, MTP / DFlash2 et KVQuant 4 bits, configuration partagée en 1 clic |
| **[Démarrage et configuration](getting-started.md)** | Prérequis, compilation CMake sans MLX, lancement 1 clic `start_ninfer_serve.sh`, serveur autonome `ninfer-metal-serve`, installation du sidecar `oMLX-extend`, `ninfer-metal.toml` et utilisation CLI/démon bas niveau |
| **[Architecture et coexistence](architecture.md)** | Cœur bimode `miv/` (`ninfer-metal-serve` + `mseries_backendd`), Provider SPI v2, routage automatique O(1), coordination mémoire UMA et mise à jour atomique par lien symbolique |
| **[Microkernel Foundry](microkernel-foundry.md)** | Appel ouvert aux microkernels pour de nouvelles familles de modèles sur M5+ (Llama, DeepSeek, Gemma, GLM, MiniMax) avec le Silicon-Squeeze Skill |
| **[Notes de version (`miv-v0.01`)](miv-v0.01-changelog.md)** | Graphiques de benchmarks Dim A–F, évolutions `ninfer-metal-serve` et `oMLX-extend`, résultats des tests et de l'audit `otool -L`, périmètre pris en charge |
| **[Feuille de route](roadmap.md)** | Jalons planifiés au-delà de `miv-v0.01` |
| **[Index de la documentation](../../README.md)** | Miroir complet en 8 langues, guides matériel, calibrations par SKU (`64-GPU` / `80-GPU`), spécifications backend et standards de certification |

---

## Remerciements

- **[`jundot/omlx`](https://github.com/jundot/omlx)** — Serveur d'inférence oMLX upstream et application macOS étendus via `oMLX-extend`.
- **[`Neroued/ninfer`](https://github.com/Neroued/ninfer)** — Moteur d'inférence natif de référence pour la gestion explicite de la mémoire et l'ordonnancement déterministe.
- **[`z-lab/dflash`](https://github.com/z-lab/dflash)** et **[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx)** — Décodage spéculatif par diffusion par blocs DFlash / DFlash2 et écosystème de modèles brouillons pour Apple Silicon.
- **[`SqueezeAILab/KVQuant`](https://github.com/SqueezeAILab/KVQuant)** et **[`FutureMLS-Lab/FOSCAR`](https://github.com/FutureMLS-Lab/FOSCAR)** — Travaux sur la quantification bas-bits du cache KV et l'attention de décodage par repliement affine qui sous-tendent les noyaux `affine4_g128` Metal 4 NAX de `miv`.
- **[`p-e-w/heretic`](https://github.com/p-e-w/heretic)** et **[`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)** — Framework Heretic de modification de modèles sans réentraînement ainsi que son portage MLX natif pour Apple Silicon et ses poids communautaires quantifiés.
