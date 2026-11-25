[English](../../microkernel-foundry.md) | [简体中文](../zh-CN/microkernel-foundry.md) | [繁體中文](../zh-TW/microkernel-foundry.md) | [日本語](../ja/microkernel-foundry.md) | [한국어](../ko/microkernel-foundry.md) | [Español](../es/microkernel-foundry.md) | [Français](../fr/microkernel-foundry.md) | **Deutsch**

# Microkernel Foundry

`ninfer-metal` (`miv-v0.01`) liefert Bare-Silicon-Metal-4-NAX-Inferenz auf Apple M5+ für hybride Qwen3.5-, Qwen3.6- und Qwen3.8-Modelle. **Microkernel Foundry** öffnet dasselbe Laufzeit-Substrat, dieselben Serving-Frontends, Kalibrierungswerkzeuge und die Zertifizierungs-Pipeline für Mitwirkende und Coding-Agents, die neue Modellfamilien über Mainline-Pull-Requests oder parallele Sidecars hinzufügen.

Zu den Zielfamilien gehören **Llama, Qwen3 Pure-Attention, DeepSeek (MLA), Gemma, GLM, MiniMax** sowie quantisierte Community-Derivate wie [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx).

---

## Zwei Wege zur Mitarbeit

### Pfad A: Mainline-PR → Mitwirkenden-Nennung (empfohlen)

Integriere den Microkernel deiner Modellfamilie direkt in `ninfer-metal` auf Basis der gemeinsamen Laufzeit **`libninfer_metal_substrate`**.

**Vom gemeinsamen Substrat bereitgestellt:**
- Metal-4-Ausführungslaufzeit (`MetalContext`, `MTL4CommandQueue`, doppelt gepuffertes `ar_pipe_[2]`, `< 2 GiB`-Compute-ICB-Encoder, `read_write`-Barrier-Reflexion).
- Gemeinsamer Paged-KV- und Präfix-Speicher (`KvPagePool`, `SharedKvArenaPool`, CoW-Refcounting, `SharedPrefixCache`, `UmaCheckpointPool`).
- Generische NAX- und Decode-Primitive (`mxfp4_nax`, `affine_nax`, `bf16_nax`, `moe_nax`, `uitofp`-freies QMV, zweistufiges paralleles `Argmax`, `Split-KV` / `Flash-NAX`).
- `.omlxm5`-Container-Loader, `ExecutionIdentityFields`, `24 MHz`-Hardware-Counter, eigenständiger nativer OpenAI-Server (`ninfer-metal-serve`) und Provider-SPI-v2-Daemon (`mseries_backendd`).

**Was dein PR ergänzt & Laufzeitumfang:** Ergänze den Layer-Graphen deiner Architektur, das Gewichtsmapping sowie architekturspezifische Kernel (etwa MLA, GeGLU, Softcap oder eigene MoE-Router), die generische Primitive in Cold-DRAM-Mikrosonden um mindestens **`>= +10%–12%`** schlagen. Die gemeinsame Nutzung von `libninfer_metal_substrate` vermeidet Code- und Speicherduplikate zwischen Architekturen, führt das Modell nativ sowohl im eigenständigen `ninfer-metal-serve` als auch in `oMLX-extend` aus und listet deinen Beitrag mit Nennung in [Unterstützte Modelle](README.md).

### Pfad B: Paralleler Sidecar → PR oder Issue → Verweis auf das Original-Repository

Für experimentelle Architekturen, eigene Quantisierungs-Layouts oder einen unabhängigen Release-Zyklus paketierst du deinen Kern als eigenständiges Sidecar-SPI-v2-Bundle unter:

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

Stelle einen atomaren `current`-Symlink samt `provider.json` mit `supported_topologies`, `supported_model_types` und `artifact_extensions` bereit. Sobald dein Bundle fertig ist, reiche einen PR ein oder öffne ein Issue: Maintainer tragen es in [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json) ein, sodass Nutzer es per [`./omlx-extend/scripts/sync_microkernels.sh`](../../../omlx-extend/scripts/sync_microkernels.sh) installieren und aktualisieren können, und listen es mit direktem Link zu deinem Original-Repository in [Unterstützte Modelle](README.md).

**Einheitliches Multi-Microkernel-Hosting:** Sowohl **`oMLX-extend` als auch das eigenständige `ninfer-metal-serve --providers-dir`** erkennen `Providers/*/current/provider.json` nativ und betreiben mehrere Pfad-A- (**`miv Core Backend`**) und Pfad-B-Microkernel parallel auf einem Port unter einheitlicher UMA-Budget-Arbitrierung und `gpu_contended`-Scheduling (in `oMLX-extend` neben Stock MLX). In der `oMLX`-macOS-App erscheinen installierte Pfad-B-Microkernel automatisch als Karten unter **`Integrations → External Engines`** und zeigen aktive Engine-Badges in **`Models`** (siehe [Visueller Leitfaden zu `oMLX-extend`](omlx-extend-guide.md)). Microkernel-Autoren schreiben ausschließlich den Metal-4-Kern und den Sidecar-SPI-v2-Daemon; HTTP/SSE-Serving, Tokenisierung, Chat-Templates und Multi-Modell-Scheduling übernehmen beide Hosts.

**Lizenzierung:** Mainline-PRs über **Pfad A** linken in `miv` und werden unter **[GPL-3.0](../../../LICENSE)** vertrieben (Mitwirkende können GPL-3.0-, Apache-2.0- oder MIT-Code einreichen); parallele Sidecars über **Pfad B** kommunizieren als getrennte Prozesse über AF_UNIX Sidecar SPI v2 und dürfen eine beliebige Lizenz verwenden.

---

## Übersicht der Modellfamilien

| Architekturfamilie | Zielbeispiele | Status | Empfohlener Pfad |
|---|---|---|---|
| **Qwen Hybrid (3:1 GDN + GQA)** | Qwen3.5 / 3.6 / 3.8 Dense (0.8B – 27B) & 35B-A3B MoE | Ausgeliefert (`miv-v0.01`) | Pfad A (Mainline, `miv/`) |
| **Pure-Attention GQA** | Llama-3.3-70B, Qwen3-32B, Gemma | Offen für PRs | Pfad A |
| **Multi-Head Latent Attention (MLA)** | DeepSeek V3 / R1-Familie | Offen für PRs | Pfad A / Pfad B |
| **MoE & Community-Quantisierungen** | GLM, MiniMax, `heretic-mlx`-Derivate | Offen für PRs | Pfad A / Pfad B |

---

## Microkernel entwickeln und zertifizieren

Menschliche Entwickler und Coding-Agents durchlaufen denselben reproduzierbaren Vier-Schritte-Zyklus:

1. **Ziel-Mac kalibrieren (`Stage B: H0 + H1`)**
   Führe `H0`-Hardware-DMA-Sweeps (`MTLBlitPassDescriptor`, `4–512+ MiB`) aus, um $B_{\text{DRAM}}$, $C_{\text{SLC}}$, $P_{\text{NAX}}$ und $I_{\text{ridge}}$ deines Chips zu messen, gefolgt von `H1`-Operator-Sweeps. Jede Akzeptanzschwelle normiert auf diese gemessenen Konstanten.
2. **Zuerst Cold-DRAM-Mikrosonde (`~50–100` Zeilen)**
   Bevor eine neue Kernel-Familie als Template ausgebaut wird, vergleiche eine eigenständige Cold-DRAM-Sonde (`working set >= 3–4 × C_SLC`, `>= 30` Mediane mit serieller Barriere) gegen Fixed-Function-NAX-/Hardware-Baselines. Behalte nur Kernel mit **`>= +10%–12%`** Nettogewinn; lösche gescheiterte experimentelle PSOs vor dem Öffnen eines PRs.
3. **Schneller `< 10 s`-Sublayer-Zyklus & Spill-Audit**
   Ermittle GPU-Sublayer-Anteile (`mlp/moe`, `attn`, `lm_head`, `sampling`) mit `omlx_m5_hw_counter_probe --mode model` (`24 MHz`-Counter) und prüfe Shader mit `metal-objdump -d` (`0 alloca`-Spills, `<= 104` VGPR bei breitem Prefill, `0 uitofp` bei vor-gekacheltem `T=1`-QMV).
4. **Maschinenlesbares 5-Dateien-Nachweispaket einreichen**
   Führe den isolierten Dim-A–F-Harness (`miv/tools/bench_omlx_vs_m5rt.py`) aus und hänge `certification-evidence/<core-id>-<version>-<sku>/` an:
   - `h0-capability.json` — Bare-Silicon-`H0`-Konstanten & 10,000-Schritt-ICB-Canary
   - `h1-calibration-<sku>.json` — SKU-isoliertes `H1`-Tuning & `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — Clean-Room-Ergebnisse Dim A–F & `phys_footprint`
   - `parity-report.json` — `0-ULP`-Parität, Quell-PPL/KL, NIAH/RULER-Recall, `0 NaN/Inf`
   - `otool-audit.txt` — Reines Apple-Framework-`otool -L`, `0 alloca`-Audit, 100-Turn-Soak-Test & CTest-Ausgabe

---

## Pflichtlektüre für Mitwirkende & Agents

- **[Silicon-Squeeze Core-Authoring Skill](../../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — Vollständige M5+-Bare-Silicon-Methodik (`MTLGPUFamilyApple10+`), direkt ladbar in Coding-Agents.
- **[Core-Zertifizierungsstandard](../../standards/core-certification-standard.md)** — Exakte Kriterien für Gate 1–4, Provider-SPI-v2-Vertrag, Roofline-Schwellenwerte sowie Tier-1-/Tier-2-Zertifizierungsabzeichen.
- **[Apple M5 Series Optimization Playbook](../../silicon-guides/m5-series-optimization-playbook.md)** — `32 KiB`-Threadgroup-Belegungsrechnung, Unified-Memory-Zulassungsgleichungen und Skalierung über Base / Pro / Max / Ultra.
