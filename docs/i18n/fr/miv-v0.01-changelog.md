[English](../../releases/miv-v0.01-changelog.md) | [简体中文](../zh-CN/miv-v0.01-changelog.md) | [繁體中文](../zh-TW/miv-v0.01-changelog.md) | [日本語](../ja/miv-v0.01-changelog.md) | [한국어](../ko/miv-v0.01-changelog.md) | [Español](../es/miv-v0.01-changelog.md) | **Français** | [Deutsch](../de/miv-v0.01-changelog.md)

# Notes de version `ninfer-metal` `miv-v0.01`

> **Version** : `ninfer-metal` `miv-v0.01` (`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`)
> **Hôte compagnon** : `oMLX-extend` `omlx-extend-v0.8.3` (extension Provider SPI v2 au-dessus de `jundot/omlx` `v0.8.3`)
> **Plateforme** : Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`), macOS 27+ (Xcode inutile à l'exécution ; Xcode 27+ requis pour compiler depuis les sources)

`miv-v0.01` est la première version publique de `ninfer-metal`, un moteur d'inférence pur C++20, Objective-C++ et Metal 4 bâti sur les accélérateurs neuronaux Apple M5+ (NAX) et Compute ICB, sans aucune dépendance d'exécution Python ou MLX. Il exécute les modèles **Qwen3.5 / Qwen3.6 / Qwen3.8 Hybrid** (Dense 0.8B–27B et MoE 256×8 comme 35B-A3B) en autonome via le serveur natif OpenAI HTTP/SSE (`ninfer-metal-serve`) ou au sein d'`oMLX-extend` en tant que **`miv Core Backend`** pour bénéficier des endpoints OpenAI, Anthropic et de l'interface macOS aux côtés de Stock MLX. Consulte [Démarrage et configuration](getting-started.md) pour l'installation et les options CLI, et le [Guide visuel d'`oMLX-extend`](omlx-extend-guide.md) pour la prise en main de l'application macOS.

## 1. Benchmarks sur Apple M5 Ultra

Toutes les mesures ci-dessous proviennent de machines Apple M5 Ultra physiques exécutant des sous-processus isolés avec poids de modèle identiques, séquences de tokens identiques, `temperature=0` et suivi mémoire `phys_footprint` face à `oMLX 0.8.0` standard (`mlx 0.32.4`, NAX activé). Les résultats des modèles 80-GPU / 256 GiB (`macOS 27.2.1` Build `26B131`, `Xcode 27.2` Build `27B52`) et 64-GPU / 96 GiB sont présentés séparément.

### Décodage mono-flux et accélération spéculative `0-ULP`

![Décodage mono-flux et accélération spéculative 0-ULP](../../assets/charts/decode-speedup.svg)

- En AR pur sur le M5 Ultra 80-GPU, `miv` devance oMLX standard sur chaque format de poids testé : `64.63 tok/s` (`+14.5%`) sur `Qwen3.8-27B MXFP4`, `54.03 tok/s` (`+7.3%`) sur `27B oQ4e` et `216.44 tok/s` (`+23.7%`) sur `Qwen3.6-35B-A3B MXFP4`.
- Le décodage spéculatif porte les flux de code `27B` à `186.74 tok/s` avec `DFlash2` (`2.89×` l'AR) et `140.48 tok/s` avec `Native MTP` (`+36.0%` par rapport à oMLX standard MTP à `103.28 tok/s`), tout en conservant une parité greedy `128/128` `0-ULP`.
- Sur la configuration 64-GPU / 96 GiB, `27B oQ4e` atteint `138.92 tok/s` (`DFlash2`, `+33.9%`) et `107.52 tok/s` (`MTP`, `+21.2%`) en Coding, ainsi que `110.01` / `100.28 tok/s` en Reasoning (`6/6` séries, `256/256` `0-ULP`).

### Montée en charge multi-utilisateurs

![Montée en charge du débit agrégé multi-utilisateurs](../../assets/charts/concurrency-scaling.svg)

- Sur les 19 points de concurrence mesurés sur la version 80-GPU, `miv` surpasse oMLX standard à chaque palier : à `B=4`, `27B` atteint `500.57 tok/s` (`oQ4e DFlash2`), `496.60 tok/s` (`MXFP4 DFlash2`) et `434.92 tok/s` (`oQ4e MTP`), soit `2.93×–3.83×` plus vite qu'oMLX standard (`148.37` MTP / `129.80` AR).
- `Qwen3.6-35B-A3B` monte à `846.29 tok/s` sur fenêtre (`794.39 tok/s` de bout en bout) à `B=8` en AR pur, soit `+52.6%` au-dessus d'oMLX standard avec un TTFT `4.16×` plus rapide.
- Chaque slot concurrent maintient la parité greedy `128/128 × 4` `0-ULP` par rapport à l'exécution mono-flux.

### Stabilité en contexte long 64K, empreinte UMA et préremplissage

![Stabilité en contexte long 64K et empreinte UMA affine4_g128](../../assets/charts/long-context-curve.svg)

- De `2K` à `64K`, la vitesse AR de `27B MXFP4` avec KV `affine4_g128` ne fléchit que de `-4.1%` (`61.42 → 58.92 tok/s`, `+28.6%` devant oMLX), tandis que `DFlash2` sur charge réelle anglais+code maintient `139.70 tok/s` à `64K` (`0-ULP` de `2K–64K`).
- `affine4_g128` réduit le cache KV `64K` de `71.8%` (`4.00 → 1.13 GiB`) et contient le pic UMA propre à `15.69 GiB` (`-3.44 GiB` face à oMLX ; `-10.06 GiB` avec les quatre profondeurs de préfixe en cache). Le TTFT à chaud `64K + 32` descend à `176.19 ms` (`bf16`) et `201.42 ms` (`affine4`), soit `7.05×–9.48×` plus rapide qu'oMLX.
- Le préremplissage atteint `2,058.5–2,110.6 tok/s` (`100.25–102.80 TF`) à `1K` et `41.02–41.84 s` à froid à `64K` sur la version 80-GPU, et `10,048.5 tok/s` (`35B-A3B`, `+15.9%`) à `4K` sur la version 64-GPU (`15/15` victoires).

### Chat multi-tours et réutilisation glissante de préfixes

![Réutilisation glissante de préfixes et stabilité UMA en chat multi-tours](../../assets/charts/multiturn-rolling.svg)

- La réutilisation glissante des préfixes UMA porte les taux de succès multi-tours au-delà de `92%` (`92.33%` à `B=1`, `92.02%` à `B=4`), divise par deux les tokens préremplis à nouveau et accélère le TTFT à chaud de `2.44×–2.94×` (`101.25 ms` à `B=1`, `282.80 ms` à `B=4`).
- La réutilisation de slabs sur place limite la hausse UMA sur 10 tours `B=1` à `+0.02 MB` sans aucune éviction (contre `+397.5 MB` sur oMLX `affine4` et `+704.8 MB` sur `bf16`), ramenant le temps total des 10 tours de `16.99 s` à `9.26 s` (`DFlash2`).
- À `B=4` (4 utilisateurs × 6 tours), le décodage NAX par lots `affine4` atteint `176.95 tok/s` (`AR`) et `206.35 tok/s` (`DFlash2`, `1.72×` oMLX), abaisse le pic UMA de `2.12 GiB` et valide le `0-ULP` sur `10/10` tours (`B=1`) et `24/24` tours (`B=4`).

## 2. Contenu livré dans `miv-v0.01`

### Serveur OpenAI autonome (`ninfer-metal-serve`)
- Binaire unique C++20 HTTP/SSE exposant `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health` et `/metrics` Prometheus, avec JSON synchrone, streaming SSE (`usage`, `[DONE]`), authentification Bearer optionnelle (`--api-key`), CORS et arrêt propre.
- Intègre un tokenizer BPE C++ (`tokenizer.json`), le rendu de gabarits de chat Jinja, l'extraction de `reasoning_content` pour `<think>` et le parsing Qwen XML/JSON `<tool_call>` en synchrone comme en streaming.
- Héberge `miv` in-process aux côtés de sidecars externes Provider SPI v2 Voie B sur un port unique (`--providers-dir`, `[[models]]`), sous un plafond UMA partagé (`81%` de la RAM par défaut) et un ordonnancement sensible à la contention.
- Livré avec un fichier [`ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) prêt à l'emploi (détecté automatiquement et partagé avec `oMLX-extend`) accompagné du lanceur en une commande `./omlx-extend/scripts/start_ninfer_serve.sh`.

### Moteur Metal 4 NAX (`miv/`)
- Arborescence autonome C++20 / Objective-C++ / Metal 4 réunissant 38 shaders `.metallib` précompilés (préremplissage NAX, décodage fusionné, routage MoE 256×8, vérification spéculative et Split-KV 4 bits), compilée sans en-têtes MLX ni Python.
- Profils dédiés de calibration matérielle pour les deux déclinaisons Apple M5 Ultra ([80-GPU / 256 GiB](../../backend/silicon-calibrations/apple-m5-ultra-80gpu.json) et [64-GPU / 96 GiB](../../backend/silicon-calibrations/apple-m5-ultra-64gpu.json)), réglant split-K, partitions Split-KV et profondeur de brouillon avec priorité aux variables d'environnement.
- Continuation de préfixes GPU zéro copie (restauration en `0.73–1.62 ms` à `64K`), réutilisation glissante sur place des préfixes UMA et attention NAX `affine4_g128` par lots pour les sessions concurrentes.

### `oMLX-extend` `omlx-extend-v0.8.3` (Application macOS et Sidecar)
- Enregistre **`miv Core Backend`** via un sidecar `AF_UNIX` (`mseries_backendd`) : les modèles `.omlxm5` adaptés sont routés automatiquement en O(1), tandis que les modèles MLX standard restent sur Stock MLX sans lancer de processus supplémentaire.
- L'interface macOS native ajoute dans `Integrations → External Engines` les cartes `miv Core Backend` et microkernels Voie B, des badges de moteur par modèle, un sélecteur Auto / Stock MLX et des réglages guidés par capacités pour le contexte `64K+`, `MTP`, `DFlash2` et le cache KV `affine4_g128`.
- Coordonne budgets UMA, purge des caches inactifs et préremplissage fragmenté sensible à la contention entre sidecar et modèles MLX, avec mises à jour atomiques par lien symbolique sous `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/`.

## 3. Empaquetage Mac autonome et vérification

- L'archive de version `ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` regroupe 16 binaires Mach-O (dont `ninfer-metal-serve`, `mseries_backendd` et les outils CLI/calibration), 38 shaders Metal 4 `.metallib`, 2 profils de calibration SKU, `provider.json` et les empreintes SHA-256.
- `otool -L` sur l'ensemble des 16 binaires confirme une liaison exclusive aux frameworks système Apple (`CoreFoundation`, `Foundation`, `IOKit`, `Metal`) et aux bibliothèques système (`libc++.1`, `libSystem.B`, `libobjc.A`), sans aucune dépendance d'exécution Homebrew, Xcode, MLX, OpenSSL ou Python.
- Les tests serveur de bout en bout (synchrone/SSE, `<think>`/`<tool_call>`, lots `B=4` et hébergement parallèle in-process + sidecar), `638 / 638` tests pytest `oMLX-extend` et `75 / 75` tests XCTest macOS passent intégralement, et le patch Provider SPI s'applique proprement sur `jundot/omlx` `v0.8.3`.

## 4. Périmètre pris en charge

| Domaine | Prise en charge `miv-v0.01` |
|---|---|
| Matériel | Apple Silicon `MTLGPUFamilyApple10+` (Apple M5 / M5 Pro / M5 Max / M5 Ultra, 64-GPU et 80-GPU) |
| OS et outils | macOS 27+ (sans Xcode à l'exécution ; Xcode 27+ / macOS 27 SDK pour compiler depuis les sources) |
| Modèles | **Qwen3.5 / Qwen3.6 / Qwen3.8 Hybrid** (3:1 GDN + GQA ; Dense 0.8B–27B et MoE 256×8, ex. 35B-A3B) |
| Formats de poids | `.omlxm5` pré-tuilé NAX hors ligne : `MXFP4` (`E2M1 × UE8M0`) et `oQ` Affine (`q3–q8`, `g64 / g128`) |
| Cache KV | `bf16` et `affine4_g128` (`0.5625 B/val`, vérifié `0-ULP` sur `2K–64K`) |
| Décodage spéculatif | AR pur (`none`), `Native MTP` intégré et `DFlash2` (mono-flux et lots `B=2..4`) |
| Interfaces de service | Endpoints OpenAI HTTP/SSE dans `ninfer-metal-serve` ; endpoints OpenAI + Anthropic et GUI macOS dans `oMLX-extend` |
