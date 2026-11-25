<p align="center"><img src="../../assets/logo.png" width="140" alt="ninfer-metal" /></p>

<h1 align="center">ninfer-metal</h1>

<p align="center">
  <strong>Native Metal 4 NAX &amp; Compute ICB LLM-Inferenz-Engine für Apple M5+ (v1-Kern-Codename <code>miv</code>)</strong>
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

[English](../../../README.md) | [简体中文](../zh-CN/README.md) | [繁體中文](../zh-TW/README.md) | [日本語](../ja/README.md) | [한국어](../ko/README.md) | [Español](../es/README.md) | [Français](../fr/README.md) | **Deutsch**

</div>

---

**`ninfer-metal`** ist eine reine C++20- / Objective-C++- / Metal-4-Inferenz-Engine für Apple Silicon mit Hardware-**Neural Accelerators (NAX)** und **Compute ICB** (`M5 / M5 Pro / M5 Max / M5 Ultra`, `MTLGPUFamilyApple10+`, `macOS 27+`).

- **Kein Python, kein MLX**: Liefert 16 Mach-O-Binaries (darunter der native Server `ninfer-metal-serve` und der Daemon `mseries_backendd`) sowie 38 vorkompilierte `.metallib`-Shader, die ausschließlich gegen macOS-System-Frameworks gelinkt sind — läuft auf sauberen Macs ohne Xcode, Python oder Homebrew-Dylibs.
- **Eigenständiger OpenAI-Server (`ninfer-metal-serve`)**: Ein einzelnes C++20-Binary, das In-Process-`miv` und parallele Pfad-B-SPI-v2-Microkernel auf einem HTTP/SSE-Port (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`) hostet — inklusive C++-BPE-Tokenizer, Jinja-Chat-Templates, `<think>`-Trennung in `reasoning_content` sowie Qwen-XML/JSON-`<tool_call>`-Parsing, direkt nutzbar mit Open WebUI, Cursor, Cline, NewAPI und den OpenAI-SDKs.
- **Einfache NAX-Gewichtskopie & `0-ULP`-Parität**: Offline-NAX-Pre-Tiling (`.miv`) bedient Prefill, `T=1`-AR-Decode und spekulative Verifikation aus einer einzigen residenten Gewichtskopie (`1.0×` Artefaktgröße). Die Basisversion **`miv-v0.01`** führt hybride Qwen3.5- / Qwen3.6- / Qwen3.8-Modelle (Dense 27B & 35B-A3B MoE) in `MXFP4` und `oQ` Affine (`q3–q8`) mit `affine4_g128`-KV-Cache, Native MTP / DFlash2, Multi-User-Batching (`B/C = 1..8`) und rollierender UMA-Präfix-Wiederverwendung aus.
- **Direkte Koexistenz über `oMLX-extend`**: Bündelt `oMLX-extend` (`omlx-extend-v0.8.3` auf Basis von `jundot/omlx`), das **`miv Core Backend`** über einen AF_UNIX-Provider-SPI-v2-Sidecar bereitstellt — angepasste `.miv`-Modelle werden automatisch in O(1) geroutet, unangepasste Modelle bleiben unverändert auf Stock MLX, und Sidecar sowie MLX teilen sich UMA unter einem gemeinsamen Speicherlimit.

---

## Leistung auf einen Blick

![Single-Stream-Decode und 0-ULP spekulative Beschleunigung](../../assets/charts/decode-speedup.svg)

![Skalierung des Gesamtdurchsatzes bei Multi-User-Parallelität](../../assets/charts/concurrency-scaling.svg)

Gemessen auf physischer Apple-M5-Ultra-Hardware (`80-GPU / 256 GiB` und `64-GPU / 96 GiB`) gegen Stock-`oMLX 0.8.0` (`mlx 0.32.4`, NAX aktiviert) in sauberen Subprozessen mit `0-ULP`-Greedy-Parität. Alle vier Diagramme und sämtliche Dim-A–F-Messwerte stehen in den **[Release Notes zu `miv-v0.01`](miv-v0.01-changelog.md)**.

---

## Unterstützte Modelle & Microkernel

| Modellfamilie | Quantisierung | Microkernel | Modus | Urheber / Quelle |
|---|---|---|---|---|
| **Qwen3.5 / 3.6 / 3.8 Dense** (0.8B – 27B, 3:1 GDN+GQA) | `MXFP4`, `oQ4e` | `miv-v0.01` | Pfad A (Mainline) | `ninfer-metal`-Kern |
| **Qwen3.5 / 3.6 Hybrid MoE** (256×8 + Shared, z. B. 35B-A3B) | `MXFP4`, `oQ4e` | `miv-v0.01` | Pfad A (Mainline) | `ninfer-metal`-Kern |
| *Llama, Qwen3 (Pure-Attn), DeepSeek MLA, Gemma, GLM, MiniMax, ...* | *Offen* | *PRs gesucht* | Pfad A / Pfad B | **[Microkernel Foundry beitreten →](microkernel-foundry.md)** |

Mitwirkende und Coding-Agents können diese Liste über **[Microkernel Foundry](microkernel-foundry.md)** erweitern:

- **Pfad A (Mainline-PR)**: Direkt auf dem gemeinsamen Metal-4-NAX-Substrat von `ninfer-metal` mergen — hier mit **Mitwirkenden-Nennung** gelistet.
- **Pfad B (Paralleler Sidecar)**: Sobald bereit, einen PR einreichen oder ein Issue öffnen — wir tragen ihn in [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json) (`sync_microkernels.sh`) ein, listen ihn hier mit einem **Link zu deinem Original-Repository**, und sowohl `oMLX-extend` als auch das eigenständige `ninfer-metal-serve --providers-dir` erkennen `Providers/*/current/provider.json` automatisch und führen Pfad-B-Microkernel parallel neben `miv` (und in `oMLX-extend` neben Stock-MLX) aus.

---

## Schnellstart

Kompiliere den MLX-freien `miv/`-Kern (`macOS 27+`, `Xcode 27+`):

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cmake -S . -B build/miv-v001 -G Ninja -DCMAKE_BUILD_TYPE=Release -DOMLX_M5_BUILD_NAX_METALLIB=ON -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF -DOMLX_M5_BUILD_PYTHON=OFF
cmake --build build/miv-v001
```

Starte den eigenständigen C++-OpenAI-Server mit einem Befehl (lädt automatisch die gemeinsamen Standardwerte aus [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml), die auch `oMLX-extend` nutzt) oder prüfe die `oMLX-extend`-Sidecar-Schicht:

```bash
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test
```

Alle Build-Flags, die `.miv`-Gewichtskonvertierung, Multi-Microkernel-Konfigurationen für `ninfer-metal-serve --providers-dir` / `[[models]]` und Anleitungen für Hot-Updates in `oMLX-extend` findest du unter **[Einstieg & Konfiguration](getting-started.md)**; für eine bebilderte Einführung in die macOS-App siehe den **[Visuellen Leitfaden zu `oMLX-extend`](omlx-extend-guide.md)**.

---

## Weiterführende Dokumentation

| Leitfaden | Inhalt |
|---|---|
| **[Visueller Leitfaden zu `oMLX-extend`](omlx-extend-guide.md)** | 5-Schritte-Screenshot-Tour durch `oMLX` unter macOS: Microkernel-Karten, aktive Engine-Badges, Auto-Routing / Stock-MLX-Schalter, MTP / DFlash2 & 4-Bit-KVQuant sowie 1-Klick-Standardkonfiguration |
| **[Einstieg & Konfiguration](getting-started.md)** | Voraussetzungen, MLX-freier CMake-Build, 1-Klick-`start_ninfer_serve.sh`, eigenständiger Server `ninfer-metal-serve`, `oMLX-extend`-Sidecar-Setup, `ninfer-metal.toml` und Low-Level-CLI/Daemon-Nutzung |
| **[Architektur & Koexistenz](architecture.md)** | Dual-Mode-`miv/`-Kern (`ninfer-metal-serve` + `mseries_backendd`), Provider SPI v2, O(1)-Auto-Routing, UMA-Speicherkoordination und atomares Symlink-Hot-Update |
| **[Microkernel Foundry](microkernel-foundry.md)** | Offener Aufruf für neue M5+-Modellfamilien-Microkernel (Llama, DeepSeek, Gemma, GLM, MiniMax) mit dem Silicon-Squeeze Skill |
| **[Release Notes (`miv-v0.01`)](miv-v0.01-changelog.md)** | Dim-A–F-Benchmark-Diagramme, Änderungen in `ninfer-metal-serve` & `oMLX-extend`, Test- & `otool -L`-Auditergebnisse sowie unterstützter Umfang |
| **[Roadmap](roadmap.md)** | Geplante Meilensteine über `miv-v0.01` hinaus |
| **[Dokumentationsindex](../../README.md)** | Vollständiger 8-Sprachen-Spiegel, Hardware-Guides, SKU-Kalibrierungen (`64-GPU` / `80-GPU`), Backend-Spezifikationen und Zertifizierungsstandards |

---

## Danksagungen

- **[`jundot/omlx`](https://github.com/jundot/omlx)** — Upstream-oMLX-Inferenzserver und macOS-App, erweitert über `oMLX-extend`.
- **[`Neroued/ninfer`](https://github.com/Neroued/ninfer)** — Native Referenz-Inferenz-Engine für explizite Speicherverwaltung und deterministisches Scheduling.
- **[`z-lab/dflash`](https://github.com/z-lab/dflash)** & **[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx)** — Block-Diffusions-Spekulationsdekodierung (DFlash / DFlash2) und Apple-Silicon-Draft-Modell-Ökosystem.
- **[`SqueezeAILab/KVQuant`](https://github.com/SqueezeAILab/KVQuant)** & **[`FutureMLS-Lab/FOSCAR`](https://github.com/FutureMLS-Lab/FOSCAR)** — Forschung zur Low-Bit-KV-Cache-Quantisierung und Affine-Folding-Decode-Attention hinter den `affine4_g128`-Metal-4-NAX-Kernels von `miv`.
- **[`p-e-w/heretic`](https://github.com/p-e-w/heretic)** & **[`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)** — Upstream-Framework Heretic für trainingsfreie Modellanpassung sowie dessen nativer Apple-Silicon-MLX-Port und quantisierte Community-Gewichte.
