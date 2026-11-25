[English](../../microkernel-foundry.md) | [简体中文](../zh-CN/microkernel-foundry.md) | [繁體中文](../zh-TW/microkernel-foundry.md) | [日本語](../ja/microkernel-foundry.md) | [한국어](../ko/microkernel-foundry.md) | [Español](../es/microkernel-foundry.md) | **Français** | [Deutsch](../de/microkernel-foundry.md)

# Microkernel Foundry

`ninfer-metal` (`miv-v0.01`) délivre une inférence Metal 4 NAX directe sur silicium sur Apple M5+ pour les modèles hybrides Qwen3.5, Qwen3.6 et Qwen3.8. **Microkernel Foundry** ouvre ce même substrat d'exécution, ses frontends de service, ses outils de calibration et sa chaîne de certification aux contributeurs et agents de code qui ajoutent de nouvelles familles de modèles via des pull requests mainline ou des sidecars parallèles.

Les familles cibles incluent **Llama, Qwen3 Pure-Attention, DeepSeek (MLA), Gemma, GLM, MiniMax** ainsi que les dérivés communautaires quantifiés tels que [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx).

---

## Deux façons de contribuer

### Voie A : PR Mainline → Crédit contributeur (recommandée)

Fusionne le microkernel de ta famille de modèles directement dans `ninfer-metal` au-dessus du runtime partagé **`libninfer_metal_substrate`**.

**Fourni par le substrat partagé :**
- Runtime d'exécution Metal 4 (`MetalContext`, `MTL4CommandQueue`, double tampon `ar_pipe_[2]`, encodeur Compute ICB `< 2 GiB`, réflexion des barrières `read_write`).
- Mémoire partagée KV paginée et de préfixes (`KvPagePool`, `SharedKvArenaPool`, comptage de références CoW, `SharedPrefixCache`, `UmaCheckpointPool`).
- Primitives génériques NAX et de décodage (`mxfp4_nax`, `affine_nax`, `bf16_nax`, `moe_nax`, QMV sans `uitofp`, `Argmax` parallèle à deux étages, `Split-KV` / `Flash-NAX`).
- Chargeur de conteneurs `.omlxm5`, `ExecutionIdentityFields`, compteurs matériels `24 MHz`, serveur OpenAI natif autonome (`ninfer-metal-serve`) et démon Provider SPI v2 (`mseries_backendd`).

**Ce qu'ajoute ta PR et périmètre d'exécution :** Ajoute le graphe de couches de ton architecture, le mapping des poids et les kernels spécifiques (par exemple MLA, GeGLU, Softcap ou routeurs MoE dédiés) qui dépassent les primitives génériques d'au moins **`>= +10%–12%`** sur micro-sondes en DRAM froide. Partager `libninfer_metal_substrate` évite toute duplication de code et de mémoire résidente entre architectures, exécute le modèle nativement dans le serveur autonome `ninfer-metal-serve` comme dans `oMLX-extend`, et inscrit ta contribution dans le tableau [Modèles pris en charge](README.md) avec crédit contributeur.

### Voie B : Sidecar parallèle → PR ou Issue → Lien vers le dépôt d'origine

Pour les architectures expérimentales, les schémas de quantification personnalisés ou un calendrier de publication indépendant, empaquette ton cœur comme bundle Sidecar SPI v2 autonome sous :

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

Expose un lien symbolique atomique `current` accompagné d'un `provider.json` déclarant `supported_topologies`, `supported_model_types` et `artifact_extensions`. Dès qu'il est prêt, soumets une PR ou ouvre une Issue : les mainteneurs l'inscrivent dans [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json) afin que les utilisateurs l'installent et le mettent à jour via [`./omlx-extend/scripts/sync_microkernels.sh`](../../../omlx-extend/scripts/sync_microkernels.sh), et le listent dans [Modèles pris en charge](README.md) avec un lien direct vers ton dépôt d'origine.

**Hébergement multi-microkernel unifié :** **`oMLX-extend` comme le serveur autonome `ninfer-metal-serve --providers-dir`** découvrent nativement `Providers/*/current/provider.json` et exécutent plusieurs microkernels Voie A (**`miv Core Backend`**) et Voie B en parallèle sur un même port sous arbitrage unifié du budget UMA et ordonnancement `gpu_contended` (aux côtés de Stock MLX dans `oMLX-extend`). Dans l'application macOS `oMLX`, les microkernels Voie B installés apparaissent automatiquement sous forme de cartes dans **`Integrations → External Engines`** et affichent des badges de moteur actif dans **`Models`** (voir le [Guide visuel d'`oMLX-extend`](omlx-extend-guide.md)). Les auteurs de microkernels n'écrivent que le cœur Metal 4 et le démon Sidecar SPI v2 ; le service HTTP/SSE, la tokenisation, les templates de chat et l'ordonnancement multi-modèles sont pris en charge par les deux hôtes.

**Licence :** Les PR de la **Voie A** sont liées à `miv` et distribuées sous **[GPL-3.0](../../../LICENSE)** (les contributeurs peuvent soumettre du code GPL-3.0, Apache-2.0 ou MIT) ; les sidecars parallèles de la **Voie B** communiquent via AF_UNIX Sidecar SPI v2 en tant que processus distincts et peuvent utiliser n'importe quelle licence.

---

## Registre des familles de modèles

| Famille d'architecture | Exemples cibles | Statut | Voie recommandée |
|---|---|---|---|
| **Qwen Hybrid (3:1 GDN + GQA)** | Qwen3.5 / 3.6 / 3.8 Dense (0.8B – 27B) et 35B-A3B MoE | Livré (`miv-v0.01`) | Voie A (Mainline, `miv/`) |
| **Pure-Attention GQA** | Llama-3.3-70B, Qwen3-32B, Gemma | Ouvert aux PR | Voie A |
| **Multi-Head Latent Attention (MLA)** | Famille DeepSeek V3 / R1 | Ouvert aux PR | Voie A / Voie B |
| **MoE et quantifications communautaires** | GLM, MiniMax, dérivés `heretic-mlx` | Ouvert aux PR | Voie A / Voie B |

---

## Concevoir et certifier un microkernel

Ingénieurs humains et agents de code suivent la même boucle reproductible en quatre étapes :

1. **Calibrer le Mac cible (`Stage B: H0 + H1`)**
   Lance les balayages DMA matériels `H0` (`MTLBlitPassDescriptor`, `4–512+ MiB`) pour mesurer $B_{\text{DRAM}}$, $C_{\text{SLC}}$, $P_{\text{NAX}}$ et $I_{\text{ridge}}$ de ta puce, suivis des balayages d'opérateurs `H1`. Chaque seuil d'acceptation est normalisé par rapport à ces constantes mesurées.
2. **Micro-sonde DRAM froide d'abord (`~50–100` lignes)**
   Avant de générer une nouvelle famille de kernels, benchmarke une sonde autonome en DRAM froide (`working set >= 3–4 × C_SLC`, `>= 30` médianes avec barrière série) face aux références NAX/matérielles à fonction fixe. Conserve les kernels offrant un gain net **`>= +10%–12%`** ; supprime les PSO expérimentaux non retenus avant d'ouvrir la PR.
3. **Boucle rapide de sous-couches `< 10 s` et audit de spills**
   Profile la répartition GPU par sous-couche (`mlp/moe`, `attn`, `lm_head`, `sampling`) avec `omlx_m5_hw_counter_probe --mode model` (compteurs `24 MHz`) et audite les shaders avec `metal-objdump -d` (`0 alloca` spills, `<= 104` VGPR sur prefill large, `0 uitofp` sur QMV pré-tuilé `T=1`).
4. **Soumettre le dossier de preuves machine-readable en 5 fichiers**
   Exécute le banc isolé Dim A–F (`miv/tools/bench_omlx_vs_m5rt.py`) et joins `certification-evidence/<core-id>-<version>-<sku>/` :
   - `h0-capability.json` — Constantes `H0` sur silicium direct et test canari ICB à 10,000 étapes
   - `h1-calibration-<sku>.json` — Calibration `H1` isolée par SKU et `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — Résultats propres Dim A–F et `phys_footprint`
   - `parity-report.json` — Parité `0-ULP`, PPL/KL par rapport à la source, rappel NIAH/RULER, `0 NaN/Inf`
   - `otool-audit.txt` — Audit `otool -L` limité aux frameworks Apple, audit `0 alloca`, test d'endurance de 100 tours et sortie CTest

---

## Lectures requises pour contributeurs et agents

- **[Silicon-Squeeze Core-Authoring Skill](../../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — Méthodologie complète sur silicium direct M5+ (`MTLGPUFamilyApple10+`), prête à être chargée dans les agents de code.
- **[Standard de certification du cœur](../../standards/core-certification-standard.md)** — Critères exacts des Gates 1–4, contrat Provider SPI v2, seuils Roofline et badges de certification Tier 1 / Tier 2.
- **[Manuel d'optimisation Apple M5 Series](../../silicon-guides/m5-series-optimization-playbook.md)** — Calcul d'occupation des threadgroups `32 KiB`, équations d'admission en mémoire unifiée et passage à l'échelle Base / Pro / Max / Ultra.
