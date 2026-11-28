[English](../../getting-started.md) | [简体中文](../zh-CN/getting-started.md) | [繁體中文](../zh-TW/getting-started.md) | [日本語](../ja/getting-started.md) | [한국어](../ko/getting-started.md) | **Español** | [Français](../fr/getting-started.md) | [Deutsch](../de/getting-started.md)

# Inicio y configuración

Esta guía cubre la compilación del núcleo `ninfer-metal` sin MLX (`miv/`), la conversión de pesos a `.miv`, el arranque en un comando del servidor OpenAI autónomo en C++20 (`ninfer-metal-serve`), la integración de **`miv Core Backend`** con `oMLX-extend`, la personalización de `ninfer-metal.toml` y el uso directo de los binarios CLI y demonio de bajo nivel. Si utilizas la aplicación macOS `oMLX`, empieza por la **[Guía visual de `oMLX-extend`](omlx-extend-guide.md)** para un recorrido con capturas en 5 pasos.

---

## Requisitos previos

| Modo | Hardware | Sistema operativo y cadena de herramientas |
|---|---|---|
| **Ejecución (paquete precompilado)** | Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`) | `macOS 27+` (sin dependencias de Xcode, Python, MLX ni Homebrew) |
| **Compilación desde código fuente** | Apple Silicon (`arm64`) | `macOS 27+`, `Xcode 27+` (macOS 27 SDK), `CMake 3.27+`, `Ninja` |

---

## 1. Compilar `ninfer-metal` (`miv/`, sin MLX / sin Python)

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

Para compilar también la extensión opcional de Python `omlx_m5_native`, pasa `-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>`. Los presets completos de desarrollo se encuentran en [`CMakePresets.json`](../../../CMakePresets.json).

---

## 2. Empaquetar y convertir pesos

### Paquete de distribución para Mac limpio

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

Este comando genera `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` (junto con `.sha256`, `provider.json` y `latest.json`) y ejecuta `otool -L` sobre los 16 binarios Mach-O (`ninfer-metal-serve`, `mseries_backendd`, `m5rt_runner` y 13 sondas de hardware/calibración) para confirmar que enlazan únicamente contra frameworks del sistema Apple (`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` y `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`).

Descomprime el archivo en cualquier Mac Apple M5+ con macOS 27+ y ejecuta `install.sh` (copia el `ninfer-metal.toml` predeterminado sin sobrescribir ediciones previas del usuario) para instalarlo en:

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

### Convertir un modelo a `.miv`

El premosaico NAX offline empaqueta los pesos `MXFP4` (`E2M1 × UE8M0`) u `oQ` Affine (`q3–q8`, `g64/g128`) una sola vez para que Prefill, descodificación AR `T=1` y verificación especulativa compartan una única copia residente en memoria:

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.miv \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. Ejecución autónoma con `ninfer-metal-serve` (C++ nativo)

`ninfer-metal-serve` es un único binario C++20 enlazado directamente con `libomlx_m5_core_runtime.a`. Integra servidor POSIX HTTP/SSE, carga de `ninfer-metal.toml`, tokenizador BPE en C++ (`tokenizer.json`), plantillas de chat Jinja (`chat_template.jinja` / `tokenizer_config.json` / respaldo ChatML), separación de razonamiento `<think>` y análisis de llamadas `<tool_call>` en formato XML/JSON de Qwen — sin Python ni MLX.

### Arranque en 1 clic (configuración compartida sin edición)

Arranca `ninfer-metal-serve` al instante sin escribir un archivo de configuración a mano. El script de inicio descubre automáticamente `ninfer-metal-serve` y carga [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) — exactamente el mismo archivo predeterminado que reutiliza `oMLX-extend` (`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`, `providers_dir = ["~/Library/Application Support/oMLX/Providers"]`):

```bash
# Arranque en 1 clic con un directorio de modelo concreto (puerto 8080 por defecto, separado del puerto 8000 de oMLX)
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# U omite el directorio para seleccionar automáticamente el primer modelo .miv en ./models o en la carpeta de modelos de oMLX
./omlx-extend/scripts/start_ninfer_serve.sh

# Pasa cualquier opción CLI adicional para sobrescribir la configuración predeterminada compartida
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
```

### Sobrescrituras directas por CLI o `ninfer-metal.toml` personalizado

Ejecutar `ninfer-metal-serve` directamente también carga automáticamente el `ninfer-metal.toml` predeterminado compartido salvo que pases `--config` o `--no-default-config`; las opciones CLI siempre sobrescriben los valores TOML:

```bash
# Arranque directo por CLI con sobrescrituras explícitas
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# Arrancar con descodificación especulativa DFlash2 explícita (o confiar en speculative.backend = "auto")
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# Cargar un archivo de configuración específico (las opciones CLI sobrescriben los valores TOML)
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### Ejecutar múltiples microkernels en paralelo (`--providers-dir` / `[[models]]`)

Un único puerto de `ninfer-metal-serve` aloja en paralelo modelos `miv` en proceso (Vía A) y sidecars externos de microkernels SPI v2 (Vía B). Repite `--model-dir` por línea de comandos o declara entradas `[[models]]` en `ninfer-metal.toml`:

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

### Conectar clientes (Open WebUI, Cursor, Cline, NewAPI, OpenAI SDK)

Apunta cualquier cliente compatible con OpenAI a `http://127.0.0.1:8080/v1`:

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

| Endpoint | Comportamiento |
|---|---|
| `GET /health` | JSON de estado (`status`, `model`, `models[]`, `gpu_contended`, `memory_budget_bytes`, `version`) |
| `GET /metrics` | Exposición en formato texto Prometheus (series agregadas y por modelo etiquetadas con `model`, `provider`, `backend_mode`) |
| `GET /v1/models`, `GET /v1/models/:id` | Listado de modelos estilo OpenAI con `max_model_len`, `provider_id` y `backend_mode` |
| `POST /v1/chat/completions` | Enrutamiento por `"model"` en JSON (`404 model_not_found` si no existe; opcional en modo de un solo modelo), JSON síncrono y streaming SSE (`chat.completion.chunk`), `<think>...</think>` hacia `reasoning_content`, análisis de `<tool_call>` en XML y JSON de Qwen, cadenas `stop`, `usage`, `stream_options.include_usage`, `[DONE]` |
| `POST /v1/completions` | Completado de texto sin formato (enrutamiento por `"model"`, JSON síncrono, streaming SSE, `usage`, `[DONE]`) |

La autenticación Bearer opcional (`--api-key`) devuelve `401` si no coincide; las solicitudes preflight CORS (`OPTIONS`) vienen habilitadas por defecto.

### Opciones CLI de `ninfer-metal-serve`

Cuando se omite `--config`, `ninfer-metal-serve` carga automáticamente [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) (`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`). Pasa `--no-default-config` para arrancar desde los valores compilados mínimos que figuran abajo; consulta [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example) para la referencia completa comentada.

| Opción | Valor CLI mínimo / Valores | Descripción |
|---|---|---|
| `--config PATH` | autodescubierto | Carga un `ninfer-metal.toml` explícito (`[server]`, `[model]`, `[[models]]`, `[engine]`, `[speculative]`); los argumentos CLI sobrescriben el TOML |
| `--no-default-config` | desactivado | Omite el descubrimiento automático de `ninfer-metal.toml` y parte de los valores compilados mínimos |
| `--model-dir PATH` | obligatorio (o vía TOML) | Directorio del modelo con `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja` y `model.miv`. Repetible para alojar múltiples modelos en paralelo en un solo puerto |
| `--providers-dir PATH` | auto (`$OMLX_PROVIDERS_DIR`, `~/Library/Application Support/oMLX/Providers`) | Directorio raíz de sidecars de microkernels SPI v2 de la Vía B (repetible); escanea `<dir>/*/current/provider.json` (respaldo `<dir>/*/provider.json`) |
| `--provider MODE` | `auto` (`auto` \| `inproc` \| `<provider_id>`) | Selección de backend: enrutamiento automático en O(1) (`auto`), `miv` en proceso (`inproc`) o `<provider_id>` SPI v2 explícito |
| `--metallib-dir PATH` (alias `--kernel-dir`) | autodetección | Directorio de shaders Metal 4 / NAX `.metallib` precompilados (`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`). Corresponde a shaders Metal, no a proveedores de microkernels SPI v2 |
| `--artifact PATH` | autodetección | Ruta explícita al artefacto `.miv` objetivo (o del proveedor) |
| `--draft-dir PATH`, `--draft-artifact PATH`, `--mtp-artifact PATH` | — | Directorio o artefacto `.miv` del modelo borrador para `dflash2` / `mtp` |
| `--chat-template PATH`, `--model-id NAME` | auto | Sobrescribe el archivo de plantilla Jinja o el ID de modelo anunciado |
| `--host ADDR`, `--port PORT` | `127.0.0.1`, `8080` | Dirección y puerto de escucha HTTP |
| `--api-key KEY`, `--cors` / `--no-cors` | `""`, activado | Exige `Authorization: Bearer <KEY>`; controla las cabeceras CORS |
| `--memory-budget-gb FLOAT` | auto (`81%` de RAM física) | Límite global de presupuesto UMA del servidor en GiB coordinado entre modelos por `MultiModelHost` |
| `--max-concurrency N` (`-c`, `-b`) | `1` (`1..8`) | Ranuras de lote concurrentes multiusuario (`B`/`C`) |
| `--max-context N` | `32768` | Longitud de la ventana de contexto (`L`) |
| `--kv-dtype MODE` | `bf16` (`bf16` \| `affine4_g128`) | Formato de almacenamiento de la caché KV |
| `--spec-backend MODE` | `none` (`auto` \| `none` \| `mtp` \| `dflash2`) | Backend de descodificación especulativa (`auto` activa `dflash2` si existe `dflash2/` complementario; si no `mtp`; en su defecto `none`) |
| `--draft-tokens N` | `0` (auto; `1..7` si se fija) | Tokens borrador por ronda de verificación (`0` = automático) |
| `--rolling-prefix-cache on\|off` | valor del runtime | Reutilización continua de prefijos en UMA para múltiples turnos con reemplazo `1×` del ancestro |
| `--prefix-cache-gb FLOAT` | valor del runtime (`0..128`) | Presupuesto para caché de prefijos y pool de puntos de control UMA (GiB) |
| `--sampling-contract NAME` | `legacy_rejection_v1` | Contrato de muestreo para verificación especulativa (`legacy_rejection_v1` \| `position_coupled_v1`) |
| `--version`, `--help` | — | Imprime la cadena de versión estructurada o el resumen de uso |

---

## 4. Ejecutar con `oMLX-extend` (coexistencia sidecar)

Este monorrepositorio mantiene el host `omlx/` parcheado junto a `omlx-extend/` (sincronizado con el upstream `jundot/omlx` mediante [`omlx-extend/patches/`](../../../omlx-extend/patches/)).

```bash
# Check the Provider SPI patch against a separate clean checkout of the pinned upstream tag
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir /path/to/upstream-omlx --check-only

# Sync the vendored ./omlx tree to the latest upstream jundot/omlx release
./omlx-extend/scripts/sync_omlx.sh
```

Una vez instalado `io.ninfer.metal/current/`, `oMLX-extend` descubre automáticamente **`miv Core Backend`** (`mseries_backendd`, con respaldo en `com.omlx.mseries/current/`, `PATH` o `build/*/miv/`) y muestra los microkernels instalados de la Vía A y la Vía B en **`Integrations → External Engines`**. Cualquier directorio de modelo que contenga `model.miv` / `model.omlxm5`, `model_pack.json` u `omlxm5-manifest.json` se enruta de forma diferida a **`miv Core Backend`**; los modelos sin adaptar permanecen intactos en Stock MLX. Desactiva **`Use External Engine`** en **`Model Settings → Advanced Settings`** (escribe `external_engine: "stock"`) para forzar Stock MLX en un modelo adaptado.

Para un recorrido con capturas de la aplicación macOS `oMLX`, consulta la **[Guía visual de `oMLX-extend`](omlx-extend-guide.md)**. Consulta [`omlx-extend/README.md`](../../../omlx-extend/README.md) para los detalles internos del host y el mapeo de ajustes de la interfaz.

---

## 5. Configurar `ninfer-metal.toml`

Tanto el servidor autónomo `ninfer-metal-serve` como `oMLX-extend` reutilizan la misma configuración predeterminada lista para usar ([`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)) sin ediciones manuales. Cuando no se pasa `--config`, ambos hosts revisan las mismas rutas en prioridad descendente (prevalece el primer archivo existente):

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml` (paquete de distribución reubicable)
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

Para conservar valores predeterminados personalizados entre actualizaciones, copia la configuración predeterminada (o la plantilla comentada [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)) en `~/.config/ninfer-metal/`:

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

En modo sidecar `oMLX-extend`, los ajustes por modelo de la interfaz (`external_engine_options["ninfer-metal"]`) sobrescriben `[sidecar.models."<name>"]` en TOML, que a su vez sobrescriben los valores predeterminados compartidos `[engine]` / `[speculative]`. En el servidor autónomo `ninfer-metal-serve`, las opciones CLI siempre sobrescriben los valores TOML.

### Parámetros principales

| Clave | Sección | Valores (`miv-v0.01`) | Notas |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`, `8080`, `""` | Ajustes del modo autónomo `ninfer-metal-serve` (ignorados por el sidecar AF_UNIX de `oMLX-extend`). |
| `providers_dir` / `memory_budget_gb` | `[server]` | Ruta (cadena o arreglo), GiB decimal | Directorios de proveedores SPI v2 de la Vía B y límite global de presupuesto UMA del servidor (por defecto `81%` de RAM física). |
| `model_dir` / `artifact` (o `artifact_path`) | `[model]` / `[[models]]` | Rutas | Directorio del modelo en modo autónomo y ruta explícita opcional al `.miv` (se aceptan tanto `artifact` como `artifact_path`, igual que `draft_artifact`/`draft_artifact_path` y `metallib_dir`). Usa `[[models]]` para alojar múltiples modelos en paralelo. |
| `max_concurrency` | `[engine]` | `1..8` (CLI sin TOML `1`, plantilla TOML `4`) | Ranuras de lote concurrentes multiusuario (**`B`** en la documentación de `miv`, **`C`** en el `ninfer` upstream). Hereda `max_concurrent_requests` de oMLX cuando se omite en modo sidecar. |
| `max_context` | `[engine]` | `32768`–`262144+` (CLI sin TOML `32768`) | Longitud de ventana de contexto (**`L`**). Se mapea desde `max_context_window` de oMLX en modo sidecar. |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128` ocupa `0.5625 B/val` con paridad `0-ULP` verificada a `64K+`. `int8_g64` no existe en `miv-v0.01`. |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | Activa la reutilización continua de prefijos en UMA entre turnos con reemplazo in-situ `1×` del ancestro. |
| `prefix_cache_gb` | `[engine]` | `0`–`128` (plantilla TOML `4.0`) | Presupuesto para caché de prefijos y pool de puntos de control UMA (GiB). |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`, `0` (auto) o `1..7` | `"auto"` activa `dflash2` si existe `dflash2/` complementario; si no `mtp`; en su defecto `none`. Se mapea desde los conmutadores de Acceleration de oMLX (`mtp_*`, `dflash_*`) en modo sidecar. |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"` \| `"position_coupled_v1"` | Contrato de muestreo de verificación especulativa (CLI sin TOML `"legacy_rejection_v1"`, plantilla TOML `"position_coupled_v1"`). |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | Selector del backend de ejecución de Prefill. |

Configuración predeterminada: [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) · Plantilla completa comentada: [`omlx-extend/config/ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example).

---

## 6. CLI de bajo nivel, demonio y biblioteca C++

Además de `ninfer-metal-serve` y `oMLX-extend`, `miv/` expone cuatro interfaces de integración de bajo nivel:

1. **`m5rt_runner` (CLI por IDs de token)** — Ejecuta inferencia de flujo único, por lotes (`--batch-mode on --max-concurrency 4`) o especulativa DFlash2 a partir de IDs enteros de token y emite métricas/tokens en NDJSON:
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.miv \
     --config /path/to/config.json \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd` (demonio AF_UNIX NDJSON RPC v2)** — Servidor persistente de motor por IDs de token con soporte para `hello`, `load`, `submit`, `cancel`, `subscribe_events`, `trim_cache`, `set_scheduler_hint` y `shutdown`:
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a` (biblioteca estática C++20)** — Incrusta `omlx::m5::Engine` directamente mediante [`miv/include/omlx_m5/engine.h`](../../../miv/include/omlx_m5/engine.h).
4. **`omlx_m5_native` (módulo Python opcional)** — Enlaces Nanobind al compilar con `-DOMLX_M5_BUILD_PYTHON=ON`.
