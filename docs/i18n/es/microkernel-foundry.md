[English](../../microkernel-foundry.md) | [简体中文](../zh-CN/microkernel-foundry.md) | [繁體中文](../zh-TW/microkernel-foundry.md) | [日本語](../ja/microkernel-foundry.md) | [한국어](../ko/microkernel-foundry.md) | **Español** | [Français](../fr/microkernel-foundry.md) | [Deutsch](../de/microkernel-foundry.md)

# Microkernel Foundry

`ninfer-metal` (`miv-v0.01`) ejecuta inferencia Metal 4 NAX sobre silicio directo en Apple M5+ para modelos híbridos Qwen3.5, Qwen3.6 y Qwen3.8. **Microkernel Foundry** abre ese mismo sustrato de ejecución, sus frontends de servicio, las herramientas de calibración y el flujo de certificación a colaboradores y agentes de código que incorporen nuevas familias de modelos mediante pull requests en la rama principal o sidecars paralelos.

Las familias objetivo incluyen **Llama, Qwen3 Pure-Attention, DeepSeek (MLA), Gemma, GLM, MiniMax** y derivados cuantizados de la comunidad como [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx).

---

## Dos vías para contribuir

### Vía A: PR Mainline → Crédito al colaborador (recomendada)

Integra el microkernel de tu familia de modelos directamente en `ninfer-metal` sobre el runtime compartido **`libninfer_metal_substrate`**.

**Lo que aporta el sustrato compartido:**
- Runtime de ejecución Metal 4 (`MetalContext`, `MTL4CommandQueue`, doble búfer `ar_pipe_[2]`, codificador Compute ICB `< 2 GiB`, reflexión de barreras `read_write`).
- Memoria compartida KV paginada y de prefijos (`KvPagePool`, `SharedKvArenaPool`, conteo de referencias CoW, `SharedPrefixCache`, `UmaCheckpointPool`).
- Primitivas genéricas NAX y de descodificación (`mxfp4_nax`, `affine_nax`, `bf16_nax`, `moe_nax`, QMV sin `uitofp`, `Argmax` paralelo en dos etapas, `Split-KV` / `Flash-NAX`).
- Cargador de contenedores `.miv` (compatible con `.omlxm5`), conversor incluido de reempaquetado de pesos (`tools/m5_artifact.py`), `ExecutionIdentityFields`, contadores hardware de `24 MHz`, servidor OpenAI nativo autónomo (`ninfer-metal-serve`) y demonio Provider SPI v2 (`mseries_backendd`).

**Lo que añade tu PR y alcance de ejecución:** Añade el grafo de capas de tu arquitectura, el mapeo de pesos, la regla de conversión `.safetensors → .miv` y los kernels específicos de arquitectura (por ejemplo, MLA, GeGLU, Softcap o enrutadores MoE propios) que superen a las primitivas genéricas en **`>= +10%–12%`** en microsondas sobre DRAM fría. Compartir `libninfer_metal_substrate` evita duplicar código y memoria residente entre arquitecturas, ejecuta el modelo de forma nativa tanto en el servidor autónomo `ninfer-metal-serve` como en `oMLX-extend` (con conversión automática en la primera carga junto a los pesos originales), y lista tu aportación en el catálogo de [Modelos soportados](README.md) con atribución de colaborador.

### Vía B: Sidecar paralelo → PR o Issue → Atribución al repositorio original

Para arquitecturas experimentales, disposiciones de cuantización personalizadas o un calendario de lanzamientos independiente, empaqueta tu núcleo como un bundle autónomo Sidecar SPI v2 en:

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

Expón un enlace simbólico atómico `current` con un `provider.json` que declare `supported_topologies`, `supported_model_types`, `artifact_extensions` y un `converter` incluido (`--source <dir> --output <file> --source-repo <id>`). Cuando esté listo, envía un PR o abre un Issue: los mantenedores lo registran en [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json) para que los usuarios lo instalen y actualicen con [`./omlx-extend/scripts/sync_microkernels.sh`](../../../omlx-extend/scripts/sync_microkernels.sh), y lo listan en [Modelos soportados](README.md) con un enlace directo a tu repositorio original.

**Alojamiento multimicrokernel unificado:** Tanto **`oMLX-extend` como el servidor autónomo `ninfer-metal-serve --providers-dir`** descubren de forma nativa `Providers/*/current/provider.json` y ejecutan múltiples microkernels de la Vía A (**`miv Core Backend`**) y la Vía B en paralelo sobre un mismo puerto con arbitraje unificado de memoria UMA y planificación `gpu_contended` (junto a Stock MLX en `oMLX-extend`). En la aplicación macOS `oMLX`, los microkernels de la Vía B instalados aparecen automáticamente como tarjetas en **`Integrations → External Engines`** y muestran insignias de motor activo en **`Models`** (consulta la [Guía visual de `oMLX-extend`](omlx-extend-guide.md)). Quienes desarrollan un microkernel solo escriben el núcleo Metal 4 y el demonio Sidecar SPI v2; el servicio HTTP/SSE, la tokenización, las plantillas de chat y la planificación multimodelo corren a cargo de ambos hosts.

**Licencias:** Los PR de la **Vía A** se enlazan con `miv` y se distribuyen bajo **[GPL-3.0](../../../LICENSE)** (los colaboradores pueden aportar código GPL-3.0, Apache-2.0 o MIT); los sidecars paralelos de la **Vía B** se comunican por AF_UNIX Sidecar SPI v2 como procesos independientes y pueden usar cualquier licencia.

---

## Catálogo de familias de modelos

| Familia de arquitectura | Ejemplos objetivo | Estado | Vía recomendada |
|---|---|---|---|
| **Qwen Hybrid (3:1 GDN + GQA)** | Qwen3.5 / 3.6 / 3.8 Dense (0.8B – 27B) y 35B-A3B MoE | Publicado (`miv-v0.01`) | Vía A (Mainline, `miv/`) |
| **Pure-Attention GQA** | Llama-3.3-70B, Qwen3-32B, Gemma | Abierto a PRs | Vía A |
| **Multi-Head Latent Attention (MLA)** | Familia DeepSeek V3 / R1 | Abierto a PRs | Vía A / Vía B |
| **MoE y cuantizaciones comunitarias** | GLM, MiniMax, derivados `heretic-mlx` | Abierto a PRs | Vía A / Vía B |

---

## Cómo crear y certificar un microkernel

Tanto ingenieros humanos como agentes de código siguen un mismo ciclo reproducible de cuatro pasos:

1. **Calibrar el Mac objetivo (`Stage B: H0 + H1`)**
   Ejecuta barridos DMA de hardware `H0` (`MTLBlitPassDescriptor`, `4–512+ MiB`) para medir $B_{\text{DRAM}}$, $C_{\text{SLC}}$, $P_{\text{NAX}}$ e $I_{\text{ridge}}$ de tu chip, seguidos de los barridos de operadores `H1`. Cada umbral de aceptación se normaliza respecto a estas constantes medidas.
2. **Microsonda en DRAM fría primero (`~50–100` líneas)**
   Antes de generar plantillas para una nueva familia de kernels, evalúa una microsonda autónoma en DRAM fría (`working set >= 3–4 × C_SLC`, `>= 30` medianas con barrera serie) frente a las líneas base NAX/hardware de función fija. Conserva solo los kernels con una ganancia neta **`>= +10%–12%`** y elimina los PSO experimentales descartados antes de abrir el PR.
3. **Ciclo rápido de subcapas `< 10 s` y auditoría de derrames**
   Perfila la distribución de subcapas en GPU (`mlp/moe`, `attn`, `lm_head`, `sampling`) con `omlx_m5_hw_counter_probe --mode model` (contadores de `24 MHz`) y audita los shaders con `metal-objdump -d` (`0 alloca` spills, `<= 104` VGPR en prefill ancho, `0 uitofp` en QMV premosaico `T=1`).
4. **Entregar el paquete de evidencia legible por máquina de 5 archivos**
   Ejecuta el banco de pruebas aislado Dim A–F (`miv/tools/bench_omlx_vs_m5rt.py`) y adjunta `certification-evidence/<core-id>-<version>-<sku>/`:
   - `h0-capability.json` — Constantes `H0` en silicio directo y prueba canario ICB de 10,000 pasos
   - `h1-calibration-<sku>.json` — Ajuste `H1` aislado por SKU y `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — Resultados limpios de Dim A–F y `phys_footprint`
   - `parity-report.json` — Paridad `0-ULP`, PPL/KL frente al origen, recuperación NIAH/RULER y `0 NaN/Inf`
   - `otool-audit.txt` — Auditoría `otool -L` exclusivamente contra frameworks Apple, auditoría `0 alloca`, prueba de resistencia de 100 turnos y salida de CTest

---

## Lectura obligatoria para colaboradores y agentes

- **[Silicon-Squeeze Core-Authoring Skill](../../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — Metodología completa en silicio directo para M5+ (`MTLGPUFamilyApple10+`), lista para cargar en agentes de código.
- **[Estándar de certificación del núcleo](../../standards/core-certification-standard.md)** — Criterios exactos de aprobación de Gates 1–4, contrato Provider SPI v2, umbrales Roofline y distintivos de certificación Tier 1 / Tier 2.
- **[Manual de optimización para la serie Apple M5](../../silicon-guides/m5-series-optimization-playbook.md)** — Cálculo de ocupación de threadgroups de `32 KiB`, ecuaciones de admisión en memoria unificada y escalado Base / Pro / Max / Ultra.
