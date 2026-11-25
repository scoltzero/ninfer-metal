[English](../../roadmap/README.md) | [简体中文](../zh-CN/roadmap.md) | [繁體中文](../zh-TW/roadmap.md) | [日本語](../ja/roadmap.md) | [한국어](../ko/roadmap.md) | **Español** | [Français](../fr/roadmap.md) | [Deutsch](../de/roadmap.md)

# Hoja de ruta de `ninfer-metal`

Línea base publicada: **`ninfer-metal` `miv-v0.01`** (con el servidor nativo C++20 `ninfer-metal-serve`) + **`oMLX-extend` `omlx-extend-v0.8.3`** (ver [Notas de lanzamiento](miv-v0.01-changelog.md)).

El trabajo de ingeniería planificado más allá de `miv-v0.01` avanza en tres líneas:

---

## 1. Virtualización de memoria de agentes `v0.02+` (`KVMem-UMA + ActKV`)

La especificación completa de arquitectura, la física de memoria UMA en Apple M5 y las puertas de verificación V1–V6 están documentadas en **[`v0.02-kvmem-actkv-plan.md`](../../roadmap/v0.02-kvmem-actkv-plan.md)**.

- **Tier-1: `KVMem-UMA + ActKV-Lite` (espacios de trabajo `256K → 1M+` en UMA)**
  - Opcional por configuración (`kv_policy = "kvmem_actkv"`, manteniendo `"lossless"` como valor por defecto permanente `0-ULP`).
  - Archivo frío asimétrico en `MTLStorageModeShared` (Pre-RoPE K en FP8/BF16 + V en `affine4_g128` — cero doble cuantización de 4 bits; `25.03 GiB` en total para un espacio de trabajo `Qwen3.8-27B` de `1M` de tokens).
  - `MeanKIndexBuffer` independiente (escaneo `< 0.6 ms`) + Action-LRFU + conjunto de trabajo guiado por confianza `W ∈ [16K, 64K]` + materialización `re-RoPE + affine4_pack` en un solo kernel (`~1.9–4.4 ms` a `64K`), manteniendo intactos `attention_affine4.metal` en la ruta caliente y los ICB estáticos.
- **Tier-2: Persistencia NVMe de Apple y ActKV completo a nivel de token**
  - Paquetes de espacio de trabajo `.omlxkv` sobre `F_NOCACHE` + `dispatch_io` para equipos M5 Pro/Max de 32–64 GiB y reanudación tras reinicio en `< 1 s`.
  - Recuperación de atención LSE en base 2 (`+96 B/tok`) y compactación in-situ de páginas `affine4_g128` sin descuantización (`144 B/tok` @ `hd=256`, `80 B/tok` @ `hd=128`) orientada a modelos de atención pura 32B/70B y MoE.

---

## 2. Sustrato compartido (`libninfer_metal_substrate`) y nuevos microkernels de familias de modelos

- Extraer de `miv/` a `libninfer_metal_substrate` el runtime Metal 4, el codificador Compute ICB `< 2 GiB`, `KvPagePool`, `SharedPrefixCache`, `UmaCheckpointPool` y las primitivas genéricas NAX.
- Incorporar microkernels de atención pura GQA (`Qwen3-32B`, `Llama-3.3-70B`), DeepSeek MLA, Gemma, GLM y MiniMax a través de **[Microkernel Foundry](microkernel-foundry.md)** bajo el **[Estándar de certificación del núcleo](../../standards/core-certification-standard.md)**.

---

## 3. `ninfer-metal-serve` Fase 2 (API de Anthropic y descodificación restringida por gramática)

Sobre el servidor nativo C++20 OpenAI HTTP/SSE publicado en `miv-v0.01`:

- **API Anthropic Messages (`POST /v1/messages`)**: Traducción nativa de peticiones/respuestas de Anthropic y streaming de eventos SSE para clientes compatibles con Claude y agentes de código.
- **Descodificación restringida por gramática (`xgrammar`)**: Añadir un callback por paso de máscara de logits a `omlx::m5::Engine` e integrar `xgrammar` para hacer cumplir de forma estricta esquemas JSON (`response_format`) y gramáticas estructuradas de llamadas a herramientas sin dependencias externas en tiempo de ejecución.
