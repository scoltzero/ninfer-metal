[English](../../roadmap/README.md) | [简体中文](../zh-CN/roadmap.md) | [繁體中文](../zh-TW/roadmap.md) | [日本語](../ja/roadmap.md) | [한국어](../ko/roadmap.md) | [Español](../es/roadmap.md) | **Français** | [Deutsch](../de/roadmap.md)

# Feuille de route `ninfer-metal`

Socle publié : **`ninfer-metal` `miv-v0.01`** (avec `ninfer-metal-serve` natif C++20) + **`oMLX-extend` `omlx-extend-v0.8.3`** (voir les [Notes de version](miv-v0.01-changelog.md)).

Les travaux d'ingénierie planifiés au-delà de `miv-v0.01` suivent trois axes :

---

## 1. Virtualisation mémoire agent `v0.02+` (`KVMem-UMA + ActKV`)

La spécification d'architecture complète, la physique mémoire UMA sur Apple M5 et les portes de vérification V1–V6 figurent dans **[`v0.02-kvmem-actkv-plan.md`](../../roadmap/v0.02-kvmem-actkv-plan.md)**.

- **Tier-1 : `KVMem-UMA + ActKV-Lite` (espaces de travail `256K → 1M+` en UMA)**
  - Activé sur option (`kv_policy = "kvmem_actkv"`, `"lossless"` restant le mode `0-ULP` permanent par défaut).
  - Archive froide asymétrique en `MTLStorageModeShared` (Pre-RoPE K en FP8/BF16 + V en `affine4_g128` — zéro double quantification 4 bits ; `25.03 GiB` au total pour un espace de travail `Qwen3.8-27B` de `1M` de tokens).
  - `MeanKIndexBuffer` dédié (balayage `< 0.6 ms`) + Action-LRFU + ensemble de travail piloté par confiance `W ∈ [16K, 64K]` + matérialisation `re-RoPE + affine4_pack` en un seul kernel (`~1.9–4.4 ms` à `64K`), sans toucher au chemin chaud `attention_affine4.metal` ni aux ICB statiques.
- **Tier-2 : Persistance NVMe Apple et ActKV complet au niveau token**
  - Paquets d'espace de travail `.omlxkv` sur `F_NOCACHE` + `dispatch_io` pour les machines M5 Pro/Max 32–64 GiB et reprise après redémarrage en `< 1 s`.
  - Restauration d'attention LSE en base 2 (`+96 B/tok`) et compaction sur place des pages `affine4_g128` sans déquantification (`144 B/tok` @ `hd=256`, `80 B/tok` @ `hd=128`) ciblant les modèles pure attention 32B/70B et MoE.

---

## 2. Substrat partagé (`libninfer_metal_substrate`) et nouveaux microkernels par famille de modèles

- Extraire de `miv/` vers `libninfer_metal_substrate` le runtime Metal 4, l'encodeur Compute ICB `< 2 GiB`, `KvPagePool`, `SharedPrefixCache`, `UmaCheckpointPool` et les primitives NAX génériques.
- Ajouter les microkernels Pure-Attention GQA (`Qwen3-32B`, `Llama-3.3-70B`), DeepSeek MLA, Gemma, GLM et MiniMax via **[Microkernel Foundry](microkernel-foundry.md)** conformément au **[Standard de certification du cœur](../../standards/core-certification-standard.md)**.

---

## 3. `ninfer-metal-serve` Phase 2 (API Anthropic et décodage contraint par grammaire)

En s'appuyant sur le serveur natif C++20 OpenAI HTTP/SSE livré dans `miv-v0.01` :

- **API Anthropic Messages (`POST /v1/messages`)** : Traduction native des requêtes/réponses Anthropic et streaming d'événements SSE pour les clients compatibles Claude et les agents de code.
- **Décodage contraint par grammaire (`xgrammar`)** : Ajouter un callback de masque de logits par étape dans `omlx::m5::Engine` et intégrer `xgrammar` pour appliquer strictement les schémas JSON (`response_format`) et les grammaires structurées d'appels d'outils sans dépendance runtime externe.
