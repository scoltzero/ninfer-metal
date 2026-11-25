[English](../../getting-started.md) | [简体中文](../zh-CN/getting-started.md) | [繁體中文](../zh-TW/getting-started.md) | [日本語](../ja/getting-started.md) | [한국어](../ko/getting-started.md) | [Español](../es/getting-started.md) | [Français](../fr/getting-started.md) | **Deutsch**

# Einstieg & Konfiguration

Dieser Leitfaden beschreibt den Build des MLX-freien `ninfer-metal`-Kerns (`miv/`), die Konvertierung von Gewichten nach `.omlxm5`, den Ein-Befehl-Start des eigenständigen C++20-OpenAI-Servers (`ninfer-metal-serve`), die Integration von **`miv Core Backend`** in `oMLX-extend`, die Anpassung von `ninfer-metal.toml` sowie den direkten Aufruf der Low-Level-CLI- und Daemon-Binaries. Wenn du die `oMLX`-macOS-App nutzt, beginne mit dem **[Visuellen Leitfaden zu `oMLX-extend`](omlx-extend-guide.md)** für eine 5-Schritte-Tour mit Screenshots.

---

## Voraussetzungen

| Modus | Hardware | Betriebssystem & Toolchain |
|---|---|---|
| **Laufzeit (vorkompiliertes Bundle)** | Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`) | `macOS 27+` (keine Abhängigkeiten zu Xcode, Python, MLX oder Homebrew) |
| **Build aus dem Quellcode** | Apple Silicon (`arm64`) | `macOS 27+`, `Xcode 27+` (macOS 27 SDK), `CMake 3.27+`, `Ninja` |

---

## 1. `ninfer-metal` bauen (`miv/`, ohne MLX / ohne Python)

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

Um zusätzlich die optionale Python-Erweiterung `omlx_m5_native` zu bauen, ergänze `-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>`. Sämtliche Entwicklungs-Presets liegen in [`CMakePresets.json`](../../../CMakePresets.json).

---

## 2. Paketieren & Gewichte konvertieren

### Release-Bundle für saubere Macs

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

Der Befehl erzeugt `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` (samt `.sha256`, `provider.json` und `latest.json`) und prüft alle 16 Mach-O-Binaries (`ninfer-metal-serve`, `mseries_backendd`, `m5rt_runner` sowie 13 Hardware-/Kalibrierungs-Probes) mit `otool -L`, um sicherzustellen, dass sie ausschließlich gegen Apple-System-Frameworks gelinkt sind (`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` und `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`).

Entpacke das Archiv auf einem beliebigen Apple-M5+-Mac mit macOS 27+ und führe `install.sh` aus (kopiert die Standard-`ninfer-metal.toml`, ohne bestehende Nutzeränderungen zu überschreiben), um unter folgendem Pfad zu installieren:

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

### Modell nach `.omlxm5` konvertieren

Offline-NAX-Pre-Tiling packt `MXFP4`- (`E2M1 × UE8M0`) oder `oQ`-Affine-Gewichte (`q3–q8`, `g64/g128`) einmalig so vor, dass Prefill, `T=1`-AR-Decode und spekulative Verifikation dieselbe residente Speicherkopie nutzen:

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.omlxm5 \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. Eigenständiger Betrieb mit `ninfer-metal-serve` (Natives C++)

`ninfer-metal-serve` ist ein einzelnes C++20-Binary, das direkt gegen `libomlx_m5_core_runtime.a` gelinkt ist. Es bündelt POSIX-HTTP/SSE, Laden von `ninfer-metal.toml`, einen C++-BPE-Tokenizer (`tokenizer.json`), Jinja-Chat-Templates (`chat_template.jinja` / `tokenizer_config.json` / ChatML-Fallback), `<think>`-Reasoning-Splitting und Qwen-XML/JSON-`<tool_call>`-Parsing — komplett ohne Python und MLX.

### 1-Klick-Start (gemeinsame Standardkonfiguration ohne Bearbeitung)

Starte `ninfer-metal-serve` direkt, ohne eine Konfigurationsdatei von Hand zu schreiben. Das Startskript findet `ninfer-metal-serve` automatisch und lädt [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) — exakt dieselbe Standarddatei, die auch `oMLX-extend` verwendet (`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`, `providers_dir = ["~/Library/Application Support/oMLX/Providers"]`):

```bash
# 1-Klick-Start mit einem bestimmten Modellverzeichnis (Standardport 8080, getrennt von oMLX-Port 8000)
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# Oder das Modellverzeichnis weglassen, um automatisch das erste .omlxm5-Modell in ./models oder im oMLX-Modellordner zu wählen
./omlx-extend/scripts/start_ninfer_serve.sh

# Zusätzliche CLI-Flags übergeben, um die gemeinsame Standardkonfiguration zu überschreiben
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
```

### Direkte CLI-Überschreibungen oder eigene `ninfer-metal.toml`

Auch beim direkten Aufruf von `ninfer-metal-serve` wird automatisch die gemeinsame Standard-`ninfer-metal.toml` geladen, sofern weder `--config` noch `--no-default-config` übergeben wird; CLI-Flags überschreiben TOML-Werte immer:

```bash
# Direkter CLI-Start mit expliziten Überschreibungen
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# Start mit expliziter DFlash2-Spekulationsdekodierung (oder auf speculative.backend = "auto" verlassen)
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# Eine bestimmte Konfigurationsdatei laden (CLI-Flags überschreiben TOML-Werte)
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### Mehrere Microkernel parallel hosten (`--providers-dir` / `[[models]]`)

Ein einzelner `ninfer-metal-serve`-Port hostet parallel In-Process-`miv`-Modelle (Pfad A) und externe SPI-v2-Microkernel-Sidecars (Pfad B). Wiederhole `--model-dir` auf der Kommandozeile oder definiere `[[models]]`-Einträge in `ninfer-metal.toml`:

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

### Clients verbinden (Open WebUI, Cursor, Cline, NewAPI, OpenAI SDK)

Verbinde beliebige OpenAI-kompatible Clients mit `http://127.0.0.1:8080/v1`:

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

### HTTP-Endpunkte (`miv-v0.01`)

| Endpunkt | Verhalten |
|---|---|
| `GET /health` | Readiness-JSON (`status`, `model`, `models[]`, `gpu_contended`, `memory_budget_bytes`, `version`) |
| `GET /metrics` | Prometheus-Textausgabe (Gesamtmetriken sowie Serien pro Modell mit Labels `model`, `provider`, `backend_mode`) |
| `GET /v1/models`, `GET /v1/models/:id` | OpenAI-Modellliste mit `max_model_len`, `provider_id` und `backend_mode` |
| `POST /v1/chat/completions` | Routing über JSON-Feld `"model"` (`404 model_not_found` bei unbekanntem Modell; im Einzelmodellbetrieb optional), synchrones JSON und SSE-Streaming (`chat.completion.chunk`), `<think>...</think>` nach `reasoning_content`, Qwen-XML- und JSON-`<tool_call>`-Parsing, `stop`-Strings, `usage`, `stream_options.include_usage`, `[DONE]` |
| `POST /v1/completions` | Rohtext-Completion (Routing per `"model"`, synchrones JSON, SSE-Streaming, `usage`, `[DONE]`) |

Optionale Bearer-Authentifizierung (`--api-key`) liefert bei Abweichung `401`; CORS-Preflight (`OPTIONS`) ist standardmäßig aktiv.

### CLI-Flags von `ninfer-metal-serve`

Wird `--config` weggelassen, lädt `ninfer-metal-serve` automatisch [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) (`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`). Übergib `--no-default-config`, um mit den unten aufgeführten reinen Kompilier-Standardwerten zu starten; die vollständig kommentierte Referenz steht in [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example).

| Flag | Reiner CLI-Standard / Werte | Beschreibung |
|---|---|---|
| `--config PATH` | automatisch erkannt | Lädt eine explizite `ninfer-metal.toml` (`[server]`, `[model]`, `[[models]]`, `[engine]`, `[speculative]`); CLI-Flags überschreiben TOML-Werte |
| `--no-default-config` | aus | Überspringt die automatische Suche nach `ninfer-metal.toml` und startet mit den reinen Kompilier-Standardwerten |
| `--model-dir PATH` | erforderlich (oder via TOML) | Modellverzeichnis mit `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` und `model.omlxm5`. Wiederholbar für paralleles Multi-Modell-Hosting auf einem Port |
| `--providers-dir PATH` | automatisch (`$OMLX_PROVIDERS_DIR`, `~/Library/Application Support/oMLX/Providers`) | Wurzelverzeichnis für Pfad-B-SPI-v2-Microkernel-Sidecars (wiederholbar); durchsucht `<dir>/*/current/provider.json` (Fallback `<dir>/*/provider.json`) |
| `--provider MODE` | `auto` (`auto` \| `inproc` \| `<provider_id>`) | Backend-Auswahl: O(1)-Auto-Routing (`auto`), In-Process-`miv` (`inproc`) oder explizite SPI-v2-`<provider_id>` |
| `--metallib-dir PATH` (Alias `--kernel-dir`) | automatisch erkannt | Verzeichnis kompilierter Metal-4-/NAX-`.metallib`-Shader (`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`). Verweist auf Metal-Shader, nicht auf SPI-v2-Microkernel-Provider |
| `--artifact PATH` | automatisch erkannt | Expliziter Pfad zum Ziel-Artefakt `.omlxm5` (oder Provider-Artefakt) |
| `--draft-dir PATH`, `--draft-artifact PATH`, `--mtp-artifact PATH` | — | Draft-Modellverzeichnis oder `.omlxm5`-Artefakt für `dflash2` / `mtp` |
| `--chat-template PATH`, `--model-id NAME` | auto | Überschreibt die Jinja-Chat-Template-Datei oder die angekündigte Modell-ID |
| `--host ADDR`, `--port PORT` | `127.0.0.1`, `8080` | HTTP-Bind-Adresse und Port |
| `--api-key KEY`, `--cors` / `--no-cors` | `""`, aktiviert | Erzwingt `Authorization: Bearer <KEY>`; steuert CORS-Header |
| `--memory-budget-gb FLOAT` | automatisch (`81%` des physischen RAM) | Serverweites UMA-Speicherbudget in GiB, modellübergreifend koordiniert durch `MultiModelHost` |
| `--max-concurrency N` (`-c`, `-b`) | `1` (`1..8`) | Parallele Multi-User-Batch-Slots (`B`/`C`) |
| `--max-context N` | `32768` | Kontextfensterlänge (`L`) |
| `--kv-dtype MODE` | `bf16` (`bf16` \| `affine4_g128`) | Speicherformat des KV-Caches |
| `--spec-backend MODE` | `none` (`auto` \| `none` \| `mtp` \| `dflash2`) | Backend für spekulatives Dekodieren (`auto` aktiviert `dflash2` bei vorhandenem Begleit-`dflash2/`, sonst `mtp`, sonst `none`) |
| `--draft-tokens N` | `0` (auto; `1..7` bei Angabe) | Draft-Tokens pro Verifikationsrunde (`0` = automatisch) |
| `--rolling-prefix-cache on\|off` | Laufzeit-Standard | Rollierende UMA-Präfix-Wiederverwendung über mehrere Turns mit `1×`-Vorgänger-Ersetzung |
| `--prefix-cache-gb FLOAT` | Laufzeit-Standard (`0..128`) | Budget für Präfix-Cache und UMA-Checkpoint-Pool (GiB) |
| `--sampling-contract NAME` | `legacy_rejection_v1` | Sampling-Vertrag für spekulative Verifikation (`legacy_rejection_v1` \| `position_coupled_v1`) |
| `--version`, `--help` | — | Gibt strukturierte Versionskennung oder Kurzübersicht der Optionen aus |

---

## 4. Betrieb mit `oMLX-extend` (Sidecar-Koexistenz)

Dieses Monorepo führt den gepatchten `omlx/`-Host direkt neben `omlx-extend/` (synchronisiert mit Upstream `jundot/omlx` über [`omlx-extend/patches/`](../../../omlx-extend/patches/)).

```bash
# Verify or apply the Provider SPI patch on ./omlx and run pytest
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test

# Sync ./omlx to a specific upstream jundot/omlx tag
./omlx-extend/scripts/sync_omlx.sh --tag v0.8.3 --omlx-dir ./omlx
```

Sobald `io.ninfer.metal/current/` installiert ist, findet `oMLX-extend` **`miv Core Backend`** (`mseries_backendd`, Fallback auf `com.omlx.mseries/current/`, `PATH` oder `build/*/miv/`) automatisch und zeigt installierte Pfad-A- und Pfad-B-Microkernel unter **`Integrations → External Engines`** an. Jedes Modellverzeichnis mit `model.omlxm5`, `model_pack.json` oder `omlxm5-manifest.json` wird bei Bedarf an **`miv Core Backend`** geroutet; unangepasste Modelle bleiben unverändert auf Stock MLX. Schalte **`Use External Engine`** in **`Model Settings → Advanced Settings`** aus (schreibt `external_engine: "stock"`), um für ein angepasstes Modell Stock MLX zu erzwingen.

Für eine bebilderte Tour durch die `oMLX`-macOS-App siehe den **[Visuellen Leitfaden zu `oMLX-extend`](omlx-extend-guide.md)**. Details zu den Host-Interna und UI-Einstellungszuordnungen stehen in [`omlx-extend/README.md`](../../../omlx-extend/README.md).

---

## 5. `ninfer-metal.toml` konfigurieren

Sowohl das eigenständige `ninfer-metal-serve` als auch `oMLX-extend` nutzen dieselbe sofort lauffähige Standardkonfiguration ([`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)) ohne manuelle Bearbeitung. Ohne `--config` prüfen beide Hosts dieselben Pfade in absteigender Priorität (die erste vorhandene Datei gewinnt):

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml` (verschiebbares Release-Bundle)
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

Um eigene Standardwerte über Updates hinweg zu behalten, kopiere die Standardkonfiguration (oder die kommentierte [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)) nach `~/.config/ninfer-metal/`:

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

Im `oMLX-extend`-Sidecar-Modus überschreiben modellbezogene UI-Einstellungen (`external_engine_options["ninfer-metal"]`) die Einträge `[sidecar.models."<name>"]` in der TOML-Datei, die wiederum die gemeinsamen `[engine]`- / `[speculative]`-Standardwerte überschreiben. Im eigenständigen `ninfer-metal-serve` überschreiben CLI-Flags TOML-Werte immer.

### Zentrale Parameter

| Schlüssel | Abschnitt | Werte (`miv-v0.01`) | Hinweise |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`, `8080`, `""` | Einstellungen für eigenständiges `ninfer-metal-serve` (im AF_UNIX-Sidecar von `oMLX-extend` ignoriert). |
| `providers_dir` / `memory_budget_gb` | `[server]` | Pfad (String oder Array), GiB-Fließkomma | Suchverzeichnisse für Pfad-B-SPI-v2-Provider sowie serverweites UMA-Budgetlimit (Standard: `81%` des physischen RAM). |
| `model_dir` / `artifact` (oder `artifact_path`) | `[model]` / `[[models]]` | Pfade | Modellverzeichnis im Standalone-Modus und optionaler expliziter `.omlxm5`-Pfad (`artifact` und `artifact_path` werden beide akzeptiert, ebenso `draft_artifact`/`draft_artifact_path` und `metallib_dir`). Nutze `[[models]]` für paralleles Multi-Modell-Hosting. |
| `max_concurrency` | `[engine]` | `1..8` (CLI-Standard `1`, TOML-Vorlage `4`) | Parallele Multi-User-Batch-Slots (**`B`** in `miv`-Docs, **`C`** im Upstream-`ninfer`). Erbt `max_concurrent_requests` von oMLX, wenn im Sidecar-Modus nicht gesetzt. |
| `max_context` | `[engine]` | `32768`–`262144+` (CLI-Standard `32768`) | Kontextfensterlänge (**`L`**). Übernimmt `max_context_window` von oMLX im Sidecar-Modus. |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128` benötigt `0.5625 B/val` bei verifizierter `0-ULP`-Parität ab `64K+`. `int8_g64` existiert in `miv-v0.01` nicht. |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | Aktiviert rollierende Multi-Turn-UMA-Präfix-Wiederverwendung mit In-Place-`1×`-Vorgänger-Ersetzung. |
| `prefix_cache_gb` | `[engine]` | `0`–`128` (TOML-Vorlage `4.0`) | Budget für Präfix-Cache und UMA-Checkpoint-Pool (GiB). |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`, `0` (auto) oder `1..7` | `"auto"` aktiviert `dflash2` bei vorhandenem Begleit-`dflash2/`, sonst `mtp`, sonst `none`. Übernimmt die oMLX-Acceleration-Schalter (`mtp_*`, `dflash_*`) im Sidecar-Modus. |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"` \| `"position_coupled_v1"` | Sampling-Vertrag für die spekulative Verifikation (CLI-Standard `"legacy_rejection_v1"`, TOML-Vorlage `"position_coupled_v1"`). |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | Auswahl des Prefill-Ausführungs-Backends. |

Standardkonfiguration: [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) · Vollständig kommentierte Vorlage: [`omlx-extend/config/ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example).

---

## 6. Low-Level-CLI, Daemon & C++-Bibliothek

Über `ninfer-metal-serve` und `oMLX-extend` hinaus bietet `miv/` vier Low-Level-Schnittstellen:

1. **`m5rt_runner` (Token-ID-CLI)** — Führt Single-Stream-, Batch- (`--batch-mode on --max-concurrency 4`) oder spekulative DFlash2-Inferenz auf Basis ganzzahliger Token-IDs aus und gibt Metriken/Tokens als NDJSON aus:
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.omlxm5 \
     --config /path/to/config.json \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd` (AF_UNIX-NDJSON-RPC-v2-Daemon)** — Persistenter Token-ID-Engine-Server mit Unterstützung für `hello`, `load`, `submit`, `cancel`, `subscribe_events`, `trim_cache`, `set_scheduler_hint` und `shutdown`:
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a` (statische C++20-Bibliothek)** — Bindet `omlx::m5::Engine` direkt über [`miv/include/omlx_m5/engine.h`](../../../miv/include/omlx_m5/engine.h) ein.
4. **`omlx_m5_native` (optionales Python-Modul)** — Nanobind-Bindings bei Build mit `-DOMLX_M5_BUILD_PYTHON=ON`.
