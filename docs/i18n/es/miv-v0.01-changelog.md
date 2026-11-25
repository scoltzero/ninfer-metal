[English](../../releases/miv-v0.01-changelog.md) | [简体中文](../zh-CN/miv-v0.01-changelog.md) | [繁體中文](../zh-TW/miv-v0.01-changelog.md) | [日本語](../ja/miv-v0.01-changelog.md) | [한국어](../ko/miv-v0.01-changelog.md) | **Español** | [Français](../fr/miv-v0.01-changelog.md) | [Deutsch](../de/miv-v0.01-changelog.md)

# Notas de lanzamiento de `ninfer-metal` `miv-v0.01`

> **Lanzamiento**: `ninfer-metal` `miv-v0.01` (`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`)
> **Host complementario**: `oMLX-extend` `omlx-extend-v0.8.3` (extensión Provider SPI v2 sobre `jundot/omlx` `v0.8.3`)
> **Plataforma**: Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`), macOS 27+ (sin necesidad de Xcode en tiempo de ejecución; Xcode 27+ para compilar desde el código fuente)

`miv-v0.01` es el primer lanzamiento público de `ninfer-metal`, un motor de inferencia puro en C++20, Objective-C++ y Metal 4 diseñado sobre los aceleradores neuronales (NAX) y Compute ICB de Apple M5+, sin dependencias de Python ni MLX en tiempo de ejecución. Ejecuta modelos **Qwen3.5 / Qwen3.6 / Qwen3.8 Hybrid** (Dense 0.8B–27B y MoE 256×8 como 35B-A3B), tanto de forma autónoma mediante el servidor nativo OpenAI HTTP/SSE (`ninfer-metal-serve`) como integrado en `oMLX-extend` bajo **`miv Core Backend`** para ofrecer API OpenAI, Anthropic e interfaz macOS junto a Stock MLX. Consulta [Inicio y configuración](getting-started.md) para instalación y opciones CLI, y la [Guía visual de `oMLX-extend`](omlx-extend-guide.md) para el recorrido por la aplicación macOS.

## 1. Benchmarks en Apple M5 Ultra

Todas las cifras provienen de equipos Apple M5 Ultra físicos ejecutando subprocesos aislados con los mismos pesos de modelo, las mismas secuencias de tokens, `temperature=0` y contabilidad de memoria `phys_footprint` frente a `oMLX 0.8.0` estándar (`mlx 0.32.4`, NAX activado). Los resultados del SKU 80-GPU / 256 GiB (`macOS 27.2.1` Build `26B131`, `Xcode 27.2` Build `27B52`) y del SKU 64-GPU / 96 GiB figuran por separado.

### Descodificación de flujo único y aceleración especulativa `0-ULP`

![Descodificación de flujo único y aceleración especulativa 0-ULP](../../assets/charts/decode-speedup.svg)

- La descodificación AR pura en el M5 Ultra de 80 GPU supera a oMLX estándar en todos los formatos de pesos evaluados: `64.63 tok/s` (`+14.5%`) en `Qwen3.8-27B MXFP4`, `54.03 tok/s` (`+7.3%`) en `27B oQ4e` y `216.44 tok/s` (`+23.7%`) en `Qwen3.6-35B-A3B MXFP4`.
- La descodificación especulativa eleva los flujos de código en `27B` a `186.74 tok/s` con `DFlash2` (`2.89×` AR) y `140.48 tok/s` con `Native MTP` (`+36.0%` frente a oMLX estándar MTP con `103.28 tok/s`), manteniendo paridad greedy `128/128` `0-ULP`.
- En el SKU 64-GPU / 96 GiB, `27B oQ4e` alcanza `138.92 tok/s` (`DFlash2`, `+33.9%`) y `107.52 tok/s` (`MTP`, `+21.2%`) en Coding, además de `110.01` / `100.28 tok/s` en Reasoning (`6/6` conjuntos, `256/256` `0-ULP`).

### Escalado de concurrencia multiusuario

![Escalado de rendimiento agregado con concurrencia multiusuario](../../assets/charts/concurrency-scaling.svg)

- En los 19 barridos de concurrencia del SKU de 80 GPU, `miv` supera a oMLX estándar en cada punto: con `B=4`, `27B` alcanza `500.57 tok/s` (`oQ4e DFlash2`), `496.60 tok/s` (`MXFP4 DFlash2`) y `434.92 tok/s` (`oQ4e MTP`), entre `2.93×–3.83×` más rápido que oMLX estándar (`148.37` MTP / `129.80` AR).
- `Qwen3.6-35B-A3B` escala hasta `846.29 tok/s` en ventana (`794.39 tok/s` de extremo a extremo) con `B=8` en AR puro, un `+52.6%` sobre oMLX estándar y un TTFT `4.16×` más rápido.
- Cada ranura concurrente mantiene paridad greedy `128/128 × 4` `0-ULP` respecto a la ejecución de flujo único.

### Estabilidad en contexto largo 64K, huella UMA y prefill

![Estabilidad en contexto largo 64K y huella UMA de affine4_g128](../../assets/charts/long-context-curve.svg)

- De `2K` a `64K`, la velocidad AR de `27B MXFP4` con KV `affine4_g128` cae apenas `-4.1%` (`61.42 → 58.92 tok/s`, `+28.6%` sobre oMLX), mientras que `DFlash2` sobre carga real de inglés y código sostiene `139.70 tok/s` en `64K` (`0-ULP` de `2K–64K`).
- `affine4_g128` recorta un `71.8%` la caché KV de `64K` (`4.00 → 1.13 GiB`) y limita el pico UMA limpio a `15.69 GiB` (`-3.44 GiB` frente a oMLX; `-10.06 GiB` con las cuatro profundidades de prefijo en caché). El TTFT en caliente `64K + 32` baja a `176.19 ms` (`bf16`) y `201.42 ms` (`affine4`), `7.05×–9.48×` más rápido que oMLX.
- El prefill llega a `2,058.5–2,110.6 tok/s` (`100.25–102.80 TF`) en `1K` y `41.02–41.84 s` en frío a `64K` en el SKU de 80 GPU, y a `10,048.5 tok/s` (`35B-A3B`, `+15.9%`) en `4K` en el SKU de 64 GPU (`15/15` victorias).

### Chat multiturno y reutilización continua de prefijos

![Reutilización continua de prefijos y estabilidad UMA en chat multiturno](../../assets/charts/multiturn-rolling.svg)

- La reutilización continua de prefijos en UMA eleva la tasa de aciertos por encima del `92%` (`92.33%` en `B=1`, `92.02%` en `B=4`), reduce a la mitad los tokens reprocesados en prefill y acelera el TTFT en caliente `2.44×–2.94×` (`101.25 ms` en `B=1`, `282.80 ms` en `B=4`).
- La reutilización de slabs en sitio mantiene el crecimiento UMA en 10 turnos `B=1` en `+0.02 MB` sin desalojos (frente a `+397.5 MB` en oMLX `affine4` y `+704.8 MB` en `bf16`), reduciendo el tiempo total de 10 turnos de `16.99 s` a `9.26 s` (`DFlash2`).
- Con `B=4` (4 usuarios × 6 turnos), la descodificación NAX por lotes con `affine4` alcanza `176.95 tok/s` (`AR`) y `206.35 tok/s` (`DFlash2`, `1.72×` oMLX), ahorra `2.12 GiB` de pico UMA y verifica `0-ULP` en `10/10` turnos (`B=1`) y `24/24` turnos (`B=4`).

## 2. Qué incluye `miv-v0.01`

### Servidor OpenAI autónomo (`ninfer-metal-serve`)
- Un único binario HTTP/SSE en C++20 que expone `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health` y `/metrics` de Prometheus, con JSON síncrono, streaming SSE (`usage`, `[DONE]`), autenticación Bearer opcional (`--api-key`), CORS y apagado limpio.
- Incorpora tokenizador BPE en C++ (`tokenizer.json`), renderizado de plantillas de chat Jinja, separación de `reasoning_content` para `<think>` y análisis síncrono y en streaming de `<tool_call>` XML/JSON de Qwen.
- Aloja `miv` en proceso junto con sidecars externos Provider SPI v2 de la Vía B en un mismo puerto (`--providers-dir`, `[[models]]`), bajo un límite UMA compartido (`81%` de RAM por defecto) y planificación sensible a la contención.
- Incluye un archivo [`ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) listo para usar (autodescubierto y compartido con `oMLX-extend`) junto con el lanzador de un solo comando `./omlx-extend/scripts/start_ninfer_serve.sh`.

### Motor Metal 4 NAX (`miv/`)
- Árbol autónomo en C++20 / Objective-C++ / Metal 4 con 38 shaders `.metallib` precompilados que cubren prefill NAX, descodificación fusionada, enrutamiento MoE 256×8, verificación especulativa y Split-KV de 4 bits, compilado sin cabeceras de MLX ni Python.
- Perfiles dedicados de calibración de hardware para ambos SKU de Apple M5 Ultra ([80-GPU / 256 GiB](../../backend/silicon-calibrations/apple-m5-ultra-80gpu.json) y [64-GPU / 96 GiB](../../backend/silicon-calibrations/apple-m5-ultra-64gpu.json)), que ajustan split-K, particiones Split-KV y profundidad de borrador dando prioridad a las variables de entorno.
- Continuación de prefijos en GPU sin copias (restauración de `0.73–1.62 ms` en `64K`), reutilización continua de prefijos UMA en sitio y atención NAX `affine4_g128` por lotes para sesiones concurrentes.

### `oMLX-extend` `omlx-extend-v0.8.3` (App macOS y Sidecar)
- Registra **`miv Core Backend`** mediante un sidecar `AF_UNIX` (`mseries_backendd`): los modelos `.miv` adaptados se enrutan automáticamente en O(1), mientras que los modelos MLX estándar permanecen en Stock MLX sin lanzar procesos adicionales.
- La interfaz nativa de macOS añade tarjetas en `Integrations → External Engines` para `miv Core Backend` y los microkernels de la Vía B, distintivos de motor por modelo, selector Auto / Stock MLX y controles guiados por capacidades para contexto `64K+`, `MTP`, `DFlash2` y caché KV `affine4_g128`.
- Coordina presupuestos UMA, limpieza de cachés inactivas y prefill fragmentado sensible a la contención entre los modelos sidecar y MLX, con actualizaciones atómicas por enlace simbólico en `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/`.

## 3. Empaquetado limpio para Mac y verificación

- El archivo de lanzamiento `ninfer-metal-miv-v0.01-macos27-arm64.tar.gz` empaqueta 16 binarios Mach-O (incluidos `ninfer-metal-serve`, `mseries_backendd` y utilidades CLI y de calibración), 38 shaders Metal 4 `.metallib`, 2 perfiles de calibración por SKU, `provider.json` y sumas de verificación SHA-256.
- `otool -L` en los 16 binarios confirma que solo enlazan con frameworks del sistema de Apple (`CoreFoundation`, `Foundation`, `IOKit`, `Metal`) y bibliotecas dinámicas del sistema (`libc++.1`, `libSystem.B`, `libobjc.A`), sin dependencias en tiempo de ejecución de Homebrew, Xcode, MLX, OpenSSL ni Python.
- Superan todas las comprobaciones las pruebas de servidor de extremo a extremo (síncrono/SSE, `<think>`/`<tool_call>`, lotes `B=4` y alojamiento paralelo en proceso + sidecar), `638 / 638` pruebas pytest de `oMLX-extend` y `75 / 75` pruebas XCTest de macOS; el parche Provider SPI se aplica limpiamente sobre `jundot/omlx` `v0.8.3`.

## 4. Alcance soportado

| Área | Soporte en `miv-v0.01` |
|---|---|
| Hardware | Apple Silicon `MTLGPUFamilyApple10+` (Apple M5 / M5 Pro / M5 Max / M5 Ultra, 64-GPU y 80-GPU) |
| SO y herramientas | macOS 27+ (sin Xcode en tiempo de ejecución; Xcode 27+ / macOS 27 SDK para compilar desde el código fuente) |
| Modelos | **Qwen3.5 / Qwen3.6 / Qwen3.8 Hybrid** (3:1 GDN + GQA; Dense 0.8B–27B y MoE 256×8, p. ej., 35B-A3B) |
| Formatos de pesos | `.miv` preteselado NAX offline: `MXFP4` (`E2M1 × UE8M0`) y `oQ` Affine (`q3–q8`, `g64 / g128`) |
| Caché KV | `bf16` y `affine4_g128` (`0.5625 B/val`, verificado `0-ULP` en `2K–64K`) |
| Descodificación especulativa | AR puro (`none`), `Native MTP` integrado y `DFlash2` (flujo único y lotes `B=2..4`) |
| Interfaces de servicio | Endpoints OpenAI HTTP/SSE en `ninfer-metal-serve`; endpoints OpenAI + Anthropic y GUI de macOS en `oMLX-extend` |
