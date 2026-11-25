[English](../../omlx-extend-guide.md) | [简体中文](../zh-CN/omlx-extend-guide.md) | [繁體中文](../zh-TW/omlx-extend-guide.md) | [日本語](../ja/omlx-extend-guide.md) | [한국어](../ko/omlx-extend-guide.md) | [Español](../es/omlx-extend-guide.md) | **Français** | [Deutsch](../de/omlx-extend-guide.md)

# Guide visuel d'`oMLX-extend`

`oMLX-extend` ajoute **`miv Core Backend (ninfer-metal)`** et les microkernels communautaires Metal 4 Voie B à l'application macOS `oMLX` sur Mac Apple M5+ (`macOS 27+`). Les modèles `.miv` adaptés basculent automatiquement sur l'exécution native Metal 4 NAX, tandis que tes modèles MLX habituels continuent de tourner sur Stock MLX.

Parcours les cinq écrans ci-dessous pour vérifier l'installation, contrôler le routage des moteurs, activer les fonctions de décodage rapide et lancer le serveur autonome sans écrire de fichier de configuration à la main.

---

## Étape 1 : Découvrir, télécharger, mettre à jour à chaud et gérer les microkernels

Ouvre **`Integrations → External Engines`** dans `oMLX` pour gérer **`miv Core Backend (ninfer-metal)`** et tout microkernel communautaire SPI v2 Voie B directement depuis l'application macOS :

![Gestionnaire de registre et cycle de vie des microkernels dans Integrations External Engines](../../assets/omlx-mac/integrations-microkernels.png)

- **Microkernel Registry (barre supérieure)** :
  - Clique sur **`Refresh`** pour récupérer le dernier registre `microkernels.json` du dépôt principal et vérifier les versions distantes `latest.json`.
  - Choisis un cœur dans le menu déroulant pour consulter juste en dessous ses **Modèles pris en charge (Supported Models)**, topologies, version SPI et version minimale de macOS avant de le télécharger.
  - Clique sur **`Download`** (à droite de **`Refresh`**) pour télécharger, vérifier le SHA-256, extraire en toute sécurité, valider `provider.json` SPI v2 et installer dans `~/Library/Application Support/oMLX/Providers/` (le bouton passe à **`Installed`** dès que la dernière version gérée est installée).
- **Liste des microkernels installés (lignes inférieures)** :
  - **Case à cocher carrée à gauche (`☑️`)** : Cochée = activé (par défaut après téléchargement) et persisté dans `settings.json`. Décocher exclut immédiatement le cœur du routage automatique O(1) et des menus par modèle.
  - **Bouton `Update` (à droite)** : Grisé par défaut ; ne s'allume qu'après qu'un clic sur **`Refresh`** a détecté une version distante plus récente. Il télécharge et bascule atomiquement le lien symbolique `current` pour une mise à jour à chaud sans interruption (les modèles actifs basculent au rechargement).
  - **Bouton `Delete` (à droite)** : Supprime un cœur géré sous `Providers/` après confirmation (en proposant de décharger d'abord les modèles actifs). Les chemins externes ou de build local gardent **`Delete`** désactivé pour éviter toute suppression accidentelle ; décoche simplement la case pour le désactiver.
- **Alternative CLI** : En ligne de commande ou script, `./omlx-extend/scripts/sync_microkernels.sh --provider io.ninfer.metal` (ainsi que `--all` / `--rollback`) reste disponible.
- **Splash** : Reste juste en dessous avec l'ensemble de ses contrôles existants inchangés.

---

## Étape 2 : Voir quel moteur exécute chaque modèle

Passe à l'écran **`Models`** pour visualiser le routage de chaque modèle sur ton Mac :

![Écran Models affichant les badges de moteur actif et les pastilles de la bibliothèque](../../assets/omlx-mac/models-overview.png)

- **Active Models** : Chaque modèle chargé affiche un badge de moteur (`miv Core`, le nom de ton microkernel Voie B ou `MLX`) juste à côté de **`Loaded`**.
- **Model Library** : Les modèles équipés de poids `.miv` affichent une pastille mise en évidence (`miv Core · Auto`, un microkernel Voie B ou `Stock MLX` si tu as désactivé le routage externe).
- **Modèles non adaptés inchangés** : Les checkpoints MLX standard n'affichent aucune pastille de moteur externe et s'exécutent via Stock MLX sans aucun processus sidecar. Tu peux charger côte à côte des modèles `miv Core Backend`, Voie B et Stock MLX sous un plafond de mémoire unifiée commun.

---

## Étape 3 : Routage automatique et retour vers Stock MLX

Clique sur n'importe quel modèle adapté dans **Model Library**, puis ouvre **`Advanced Settings`** :

![Model Settings Advanced affichant le routage automatique vers miv Core Backend](../../assets/omlx-mac/engine-routing-auto.png)

- **`Use External Engine`** est activé (**ON**) par défaut pour les modèles `.miv` adaptés, avec le menu déroulant réglé sur **`miv Core Backend (Auto)`** (l'en-tête principal de Model Settings affiche également la pastille du moteur actif).
- **Revenir à Stock MLX à tout moment** : Bascule **`Use External Engine`** sur **OFF**. `oMLX` enregistre `external_engine: "stock"` et affiche *"Using Stock MLX (auto-route overridden)"* sans supprimer tes poids `.miv`. Repasse le bouton sur **ON** pour rétablir le routage automatique.
- **Choisir un autre moteur** : Si un microkernel Voie B compatible ou Splash est installé pour le même modèle, sélectionne-le directement dans le menu déroulant.

---

## Étape 4 : Activer MTP / DFlash2 et KVQuant 4 bits

`oMLX-extend` réutilise les écrans de réglages standard d'`oMLX`. Les contrôles pris en charge par le microkernel actif se déverrouillent en pleine luminosité ; les réglages propres à MLX restent grisés pour identifier d'un coup d'œil ce que le moteur exécute.

### Acceleration (`Lightning MTP` et `DFlash`)

Ouvre **`Model Settings → Acceleration`** :

![Model Settings Acceleration déverrouillé pour miv Core Backend](../../assets/omlx-mac/acceleration-unlocked.png)

- **`Lightning MTP`** (ainsi que **`Adaptive max depth`**) se déverrouille lorsque le modèle intègre une tête MTP embarquée.
- **`DFlash`** (avec **`DFlash Draft Model`** et **`Runtime Block Size`**) se déverrouille pour le décodage spéculatif par diffusion par blocs.
- Les lignes propres à MLX (`Draft Quantization`, `Verify Mode`, `Draft Window Size`, `VLM MTP`) restent grisées.

### Memory Saving (`4-bit KVQuant`)

Ouvre **`Model Settings → Memory Saving`** :

![Model Settings Memory Saving déverrouillé pour TurboQuant KV Cache Affine4](../../assets/omlx-mac/kvquant-unlocked.png)

- Active **`TurboQuant KV Cache`** et sélectionne **`Affine4 · 4-bit`** pour utiliser le cache KV `affine4_g128` de `miv Core Backend` (`0.5625 B/val`, parité `0-ULP` vérifiée jusqu'à `64K+` de contexte).
- Désactiver l'interrupteur utilise un cache KV `bf16` complet. Sélectionner `affine8` ou `turboquant` bascule sur `bf16` dans `miv-v0.01`.
- Les options mémoire propres à MLX telles que **`MoE Expert Offload`** restent grisées.

---

## Étape 5 : Configuration partagée sans édition et lancement autonome en 1 clic

Inutile d'écrire un fichier de configuration à la main. **`oMLX-extend`** et le serveur C++ autonome (**`ninfer-metal-serve`**) chargent automatiquement [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) avec des valeurs par défaut prêtes pour Apple M5+ :

| Réglage | Défaut partagé | Rôle |
|---|---|---|
| `max_concurrency` | `4` | Emplacements de batch concurrents multi-utilisateurs (`1..8`) |
| `max_context` | `65536` | Longueur de la fenêtre de contexte (tokens) |
| `kv_dtype` | `"affine4_g128"` | Cache KV 4 bits groupe 128 (`0-ULP` à `64K+`) |
| `rolling_prefix_cache` / `prefix_cache_gb` | `true` / `4.0` | Réutilisation glissante des préfixes UMA multi-tours (pool de `4 GiB`) |
| `speculative.backend` | `"auto"` | Choisit `dflash2` si un brouillon compagnon existe, sinon `mtp`, sinon AR pur |
| `providers_dir` | `["~/Library/Application Support/oMLX/Providers"]` | Détecte automatiquement les microkernels SPI v2 Voie B installés |

Pour démarrer le serveur autonome compatible OpenAI sur le port `8080` (distinct du port `8000` d'`oMLX`), exécute :

```bash
# Pointer directement vers un répertoire de modèle adapté
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# Ou omettre le chemin pour sélectionner automatiquement le premier modèle .miv dans ./models ou le dossier des modèles d'oMLX
./omlx-extend/scripts/start_ninfer_serve.sh
```

**Priorité des surcharges :**
- Dans **`oMLX-extend`** : les réglages par modèle dans l'interface priment, suivis de `[sidecar.models."<name>"]`, puis des valeurs partagées `[engine]` / `[speculative]` de `ninfer-metal.toml`.
- Dans **`ninfer-metal-serve`** : les options CLI priment toujours sur `ninfer-metal.toml`. Copie `ninfer-metal.toml` vers `~/.config/ninfer-metal/ninfer-metal.toml` lorsque tu souhaites conserver tes propres réglages par défaut après les mises à jour.

---

## Prochaines étapes

- **[Démarrage et configuration](getting-started.md)** — Compiler depuis les sources, convertir des checkpoints en `.miv` et passer en revue toutes les options CLI de `ninfer-metal-serve`.
- **[Architecture et coexistence](architecture.md)** — Fonctionnement interne du routage O(1), de Provider SPI v2 et des budgets mémoire UMA partagés.
- **[Microkernel Foundry](microkernel-foundry.md)** — Ajouter de nouvelles familles de modèles Voie A ou Voie B pour Apple M5+.
