[English](../../getting-started.md) | [简体中文](../zh-CN/getting-started.md) | [繁體中文](../zh-TW/getting-started.md) | [日本語](../ja/getting-started.md) | [한국어](../ko/getting-started.md) | [Español](../es/getting-started.md) | **Français** | [Deutsch](../de/getting-started.md)

# Démarrage et configuration

Ce guide couvre la compilation du cœur `ninfer-metal` sans MLX (`miv/`), la conversion des poids au format `.omlxm5`, le lancement en une commande du serveur OpenAI autonome en C++20 (`ninfer-metal-serve`), l'intégration de **`miv Core Backend`** avec `oMLX-extend`, la personnalisation de `ninfer-metal.toml` et l'appel des binaires CLI et démon bas niveau. Si tu utilises l'application macOS `oMLX`, commence par le **[Guide visuel d'`oMLX-extend`](omlx-extend-guide.md)** pour un parcours illustré en 5 étapes.

---

## Prérequis

| Mode | Matériel | OS et chaîne d'outils |
|---|---|---|
| **Exécution (archive précompilée)** | Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`) | `macOS 27+` (aucune dépendance Xcode, Python, MLX ou Homebrew) |
| **Compilation depuis les sources** | Apple Silicon (`arm64`) | `macOS 27+`, `Xcode 27+` (SDK macOS 27), `CMake 3.27+`, `Ninja` |

---

## 1. Compiler `ninfer-metal` (`miv/`, sans MLX / sans Python)

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

Pour compiler également l'extension Python optionnelle `omlx_m5_native`, ajoute `-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>`. L'ensemble des presets de développement se trouve dans [`CMakePresets.json`](../../../CMakePresets.json).

---

## 2. Empaqueter et convertir les poids

### Archive de distribution pour Mac vierge

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

Cette commande génère `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` (ainsi que `.sha256`, `provider.json` et `latest.json`) et exécute `otool -L` sur les 16 binaires Mach-O (`ninfer-metal-serve`, `mseries_backendd`, `m5rt_runner` et 13 sondes matériel/calibration) afin de confirmer qu'ils ne sont liés qu'aux frameworks système Apple (`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` et `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`).

Extrais l'archive sur n'importe quel Mac Apple M5+ sous macOS 27+ et lance `install.sh` (copie le fichier `ninfer-metal.toml` par défaut sans écraser tes modifications existantes) pour l'installer sous :

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

### Convertir un modèle vers `.omlxm5`

Le pré-tuilage NAX hors ligne prépare les poids `MXFP4` (`E2M1 × UE8M0`) ou `oQ` Affine (`q3–q8`, `g64/g128`) une seule fois afin que le Prefill, le décodage AR `T=1` et la vérification spéculative partagent une unique copie résidente en mémoire :

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.omlxm5 \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. Exécution autonome avec `ninfer-metal-serve` (C++ natif)

`ninfer-metal-serve` est un binaire C++20 unique lié directement à `libomlx_m5_core_runtime.a`. Il embarque le serveur POSIX HTTP/SSE, le chargement de `ninfer-metal.toml`, un tokenizer BPE en C++ (`tokenizer.json`), les templates de chat Jinja (`chat_template.jinja` / `tokenizer_config.json` / repli ChatML), la séparation du raisonnement `<think>` et le parsing XML/JSON `<tool_call>` de Qwen — sans Python ni MLX.

### Lancement en 1 clic (configuration partagée sans édition)

Démarre `ninfer-metal-serve` immédiatement sans rédiger de fichier de configuration à la main. Le script lanceur détecte automatiquement `ninfer-metal-serve` et charge [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) — exactement le même fichier par défaut réutilisé par `oMLX-extend` (`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`, `providers_dir = ["~/Library/Application Support/oMLX/Providers"]`) :

```bash
# Lancement en 1 clic avec un répertoire de modèle précis (port 8080 par défaut, distinct du port 8000 d'oMLX)
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# Ou omettre le répertoire pour sélectionner automatiquement le premier modèle .omlxm5 dans ./models ou le dossier des modèles d'oMLX
./omlx-extend/scripts/start_ninfer_serve.sh

# Passer n'importe quelle option CLI supplémentaire pour surcharger la configuration partagée par défaut
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
```

### Surcharges CLI directes ou `ninfer-metal.toml` personnalisé

Lancer `ninfer-metal-serve` directement charge aussi automatiquement le fichier `ninfer-metal.toml` partagé par défaut, sauf si `--config` ou `--no-default-config` est passé ; les options CLI priment toujours sur les valeurs TOML :

```bash
# Lancement CLI direct avec surcharges explicites
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# Lancer avec le décodage spéculatif DFlash2 explicite (ou s'appuyer sur speculative.backend = "auto")
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# Charger un fichier de configuration précis (les options CLI priment sur les valeurs TOML)
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### Exécuter plusieurs microkernels en parallèle (`--providers-dir` / `[[models]]`)

Un seul port `ninfer-metal-serve` héberge en parallèle des modèles `miv` in-process (Voie A) et des sidecars externes de microkernels SPI v2 (Voie B). Répète `--model-dir` en ligne de commande ou déclare des blocs `[[models]]` dans `ninfer-metal.toml` :

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

### Connecter des clients (Open WebUI, Cursor, Cline, NewAPI, OpenAI SDK)

Pointe tout client compatible OpenAI vers `http://127.0.0.1:8080/v1` :

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

### Endpoints HTTP (`miv-v0.01`)

| Endpoint | Comportement |
|---|---|
| `GET /health` | JSON d'état (`status`, `model`, `models[]`, `gpu_contended`, `memory_budget_bytes`, `version`) |
| `GET /metrics` | Exposition texte Prometheus (séries agrégées et par modèle étiquetées `model`, `provider`, `backend_mode`) |
| `GET /v1/models`, `GET /v1/models/:id` | Liste des modèles au format OpenAI avec `max_model_len`, `provider_id` et `backend_mode` |
| `POST /v1/chat/completions` | Routage via le champ JSON `"model"` (`404 model_not_found` si inconnu ; optionnel en mode mono-modèle), JSON synchrone et streaming SSE (`chat.completion.chunk`), `<think>...</think>` vers `reasoning_content`, parsing XML et JSON `<tool_call>` Qwen, chaînes `stop`, `usage`, `stream_options.include_usage`, `[DONE]` |
| `POST /v1/completions` | Complétion de texte brut (routage par `"model"`, JSON synchrone, streaming SSE, `usage`, `[DONE]`) |

L'authentification Bearer optionnelle (`--api-key`) renvoie `401` en cas d'écart ; les requêtes CORS preflight (`OPTIONS`) sont activées par défaut.

### Options CLI de `ninfer-metal-serve`

Lorsque `--config` est omis, `ninfer-metal-serve` charge automatiquement [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) (`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`). Passe `--no-default-config` pour démarrer depuis les valeurs compilées minimales listées ci-dessous ; voir [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example) pour la référence complète commentée.

| Option | Défaut CLI nu / Valeurs | Description |
|---|---|---|
| `--config PATH` | auto-découvert | Charge un `ninfer-metal.toml` explicite (`[server]`, `[model]`, `[[models]]`, `[engine]`, `[speculative]`) ; les options CLI priment sur le TOML |
| `--no-default-config` | désactivé | Ignore la découverte automatique de `ninfer-metal.toml` et démarre avec les valeurs compilées minimales |
| `--model-dir PATH` | requis (ou via TOML) | Répertoire du modèle contenant `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` et `model.omlxm5`. Répétable pour héberger plusieurs modèles en parallèle sur un même port |
| `--providers-dir PATH` | auto (`$OMLX_PROVIDERS_DIR`, `~/Library/Application Support/oMLX/Providers`) | Répertoire racine de sidecars de microkernels SPI v2 Voie B (répétable) ; parcourt `<dir>/*/current/provider.json` (repli `<dir>/*/provider.json`) |
| `--provider MODE` | `auto` (`auto` \| `inproc` \| `<provider_id>`) | Sélection du backend : routage automatique O(1) (`auto`), `miv` in-process (`inproc`) ou `<provider_id>` SPI v2 explicite |
| `--metallib-dir PATH` (alias `--kernel-dir`) | auto-détecté | Répertoire des shaders Metal 4 / NAX `.metallib` précompilés (`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`). Désigne des shaders Metal et non des fournisseurs de microkernels SPI v2 |
| `--artifact PATH` | auto-détecté | Chemin explicite de l'artefact `.omlxm5` cible (ou du fournisseur) |
| `--draft-dir PATH`, `--draft-artifact PATH`, `--mtp-artifact PATH` | — | Répertoire du modèle brouillon ou artefact `.omlxm5` pour `dflash2` / `mtp` |
| `--chat-template PATH`, `--model-id NAME` | auto | Remplace le fichier de template Jinja ou l'identifiant de modèle exposé |
| `--host ADDR`, `--port PORT` | `127.0.0.1`, `8080` | Adresse et port d'écoute HTTP |
| `--api-key KEY`, `--cors` / `--no-cors` | `""`, activé | Exige `Authorization: Bearer <KEY>` ; contrôle les en-têtes CORS |
| `--memory-budget-gb FLOAT` | auto (`81%` de la RAM physique) | Plafond global du budget UMA du serveur en GiB arbitré entre modèles par `MultiModelHost` |
| `--max-concurrency N` (`-c`, `-b`) | `1` (`1..8`) | Emplacements de batch concurrents multi-utilisateurs (`B`/`C`) |
| `--max-context N` | `32768` | Longueur de fenêtre de contexte (`L`) |
| `--kv-dtype MODE` | `bf16` (`bf16` \| `affine4_g128`) | Format de stockage du cache KV |
| `--spec-backend MODE` | `none` (`auto` \| `none` \| `mtp` \| `dflash2`) | Backend de décodage spéculatif (`auto` active `dflash2` si un répertoire `dflash2/` compagnon existe, sinon `mtp`, sinon `none`) |
| `--draft-tokens N` | `0` (auto ; `1..7` si explicite) | Nombre de tokens brouillons par tour de vérification (`0` = auto) |
| `--rolling-prefix-cache on\|off` | défaut runtime | Réutilisation glissante des préfixes en UMA multi-tours avec remplacement `1×` de l'ancêtre |
| `--prefix-cache-gb FLOAT` | défaut runtime (`0..128`) | Budget du cache de préfixes et du pool de points de contrôle UMA (GiB) |
| `--sampling-contract NAME` | `legacy_rejection_v1` | Contrat d'échantillonnage de la vérification spéculative (`legacy_rejection_v1` \| `position_coupled_v1`) |
| `--version`, `--help` | — | Affiche la chaîne de version structurée ou le résumé des options |

---

## 4. Exécuter avec `oMLX-extend` (coexistence sidecar)

Ce monorepo suit directement l'hôte `omlx/` patché aux côtés d'`omlx-extend/` (synchronisé avec l'upstream `jundot/omlx` via [`omlx-extend/patches/`](../../../omlx-extend/patches/)).

```bash
# Verify or apply the Provider SPI patch on ./omlx and run pytest
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test

# Sync ./omlx to a specific upstream jundot/omlx tag
./omlx-extend/scripts/sync_omlx.sh --tag v0.8.3 --omlx-dir ./omlx
```

Une fois `io.ninfer.metal/current/` installé, `oMLX-extend` détecte automatiquement **`miv Core Backend`** (`mseries_backendd`, avec repli sur `com.omlx.mseries/current/`, `PATH` ou `build/*/miv/`) et affiche les microkernels Voie A et Voie B installés sous **`Integrations → External Engines`**. Tout répertoire de modèle contenant `model.omlxm5`, `model_pack.json` ou `omlxm5-manifest.json` est routé à la demande vers **`miv Core Backend`** ; les modèles non adaptés restent sur Stock MLX sans modification. Désactive **`Use External Engine`** dans **`Model Settings → Advanced Settings`** (écrit `external_engine: "stock"`) pour forcer Stock MLX sur un modèle adapté.

Pour un parcours illustré de l'application macOS `oMLX`, voir le **[Guide visuel d'`oMLX-extend`](omlx-extend-guide.md)**. Voir [`omlx-extend/README.md`](../../../omlx-extend/README.md) pour le fonctionnement interne de l'hôte et la correspondance des réglages d'interface.

---

## 5. Configurer `ninfer-metal.toml`

Le serveur autonome `ninfer-metal-serve` et `oMLX-extend` réutilisent la même configuration par défaut prête à l'emploi ([`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)) sans édition manuelle. Lorsque `--config` n'est pas fourni, les deux hôtes vérifient les mêmes chemins par priorité décroissante (le premier fichier existant l'emporte) :

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml` (archive de version déplaçable)
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

Pour conserver des valeurs par défaut personnalisées entre les mises à jour, copie la configuration par défaut (ou le modèle commenté [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)) dans `~/.config/ninfer-metal/` :

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

En mode sidecar `oMLX-extend`, les réglages UI par modèle (`external_engine_options["ninfer-metal"]`) surchargent `[sidecar.models."<name>"]` dans le TOML, qui surchargent eux-mêmes les valeurs partagées `[engine]` / `[speculative]`. En mode autonome `ninfer-metal-serve`, les options CLI priment toujours sur les valeurs TOML.

### Paramètres clés

| Clé | Section | Valeurs (`miv-v0.01`) | Notes |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`, `8080`, `""` | Réglages autonomes de `ninfer-metal-serve` (ignorés par le sidecar AF_UNIX `oMLX-extend`). |
| `providers_dir` / `memory_budget_gb` | `[server]` | Chemin (chaîne ou tableau), flottant GiB | Répertoires de fournisseurs SPI v2 Voie B et plafond global du budget UMA serveur (par défaut `81%` de la RAM physique). |
| `model_dir` / `artifact` (ou `artifact_path`) | `[model]` / `[[models]]` | Chemins | Répertoire du modèle autonome et chemin explicite optionnel vers `.omlxm5` (`artifact` et `artifact_path` sont tous deux acceptés, comme `draft_artifact`/`draft_artifact_path` et `metallib_dir`). Utilise `[[models]]` pour héberger plusieurs modèles en parallèle. |
| `max_concurrency` | `[engine]` | `1..8` (CLI nue `1`, modèle TOML `4`) | Emplacements de batch concurrents multi-utilisateurs (**`B`** dans la doc `miv`, **`C`** dans `ninfer` upstream). Hérite de `max_concurrent_requests` d'oMLX s'il est omis en mode sidecar. |
| `max_context` | `[engine]` | `32768`–`262144+` (CLI nue `32768`) | Longueur de fenêtre de contexte (**`L`**). Mappé depuis `max_context_window` d'oMLX en mode sidecar. |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128` occupe `0.5625 B/val` avec parité `0-ULP` vérifiée à `64K+`. `int8_g64` est absent dans `miv-v0.01`. |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | Active la réutilisation glissante des préfixes en UMA multi-tours avec remplacement sur place `1×` de l'ancêtre. |
| `prefix_cache_gb` | `[engine]` | `0`–`128` (modèle TOML `4.0`) | Budget du cache de préfixes et du pool de points de contrôle UMA (GiB). |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`, `0` (auto) ou `1..7` | `"auto"` active `dflash2` si un répertoire `dflash2/` compagnon existe, sinon `mtp`, sinon `none`. Mappé depuis les réglages Acceleration d'oMLX (`mtp_*`, `dflash_*`) en mode sidecar. |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"` \| `"position_coupled_v1"` | Contrat d'échantillonnage pour la vérification spéculative (CLI nue `"legacy_rejection_v1"`, modèle TOML `"position_coupled_v1"`). |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | Sélecteur du backend d'exécution Prefill. |

Configuration par défaut : [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) · Modèle complet commenté : [`omlx-extend/config/ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example).

---

## 6. CLI bas niveau, démon et bibliothèque C++

Au-delà de `ninfer-metal-serve` et d'`oMLX-extend`, `miv/` expose quatre surfaces d'intégration bas niveau :

1. **`m5rt_runner` (CLI par ID de tokens)** — Exécute l'inférence mono-flux, par lots (`--batch-mode on --max-concurrency 4`) ou spéculative DFlash2 à partir d'ID entiers de tokens et émet des métriques/tokens au format NDJSON :
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.omlxm5 \
     --config /path/to/config.json \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd` (démon AF_UNIX NDJSON RPC v2)** — Serveur moteur persistant par ID de tokens prenant en charge `hello`, `load`, `submit`, `cancel`, `subscribe_events`, `trim_cache`, `set_scheduler_hint` et `shutdown` :
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a` (bibliothèque statique C++20)** — Embarque directement `omlx::m5::Engine` via [`miv/include/omlx_m5/engine.h`](../../../miv/include/omlx_m5/engine.h).
4. **`omlx_m5_native` (module Python optionnel)** — Liaisons Nanobind compilées avec `-DOMLX_M5_BUILD_PYTHON=ON`.
