[English](../../omlx-extend-guide.md) | [简体中文](../zh-CN/omlx-extend-guide.md) | [繁體中文](../zh-TW/omlx-extend-guide.md) | [日本語](../ja/omlx-extend-guide.md) | [한국어](../ko/omlx-extend-guide.md) | **Español** | [Français](../fr/omlx-extend-guide.md) | [Deutsch](../de/omlx-extend-guide.md)

# Guía visual de `oMLX-extend`

`oMLX-extend` incorpora **`miv Core Backend (ninfer-metal)`** y microkernels comunitarios Metal 4 de la Vía B a la aplicación macOS `oMLX` en equipos Apple M5+ (`macOS 27+`). Los modelos `.omlxm5` adaptados pasan automáticamente a ejecución nativa en Metal 4 NAX, mientras que tus modelos MLX habituales siguen ejecutándose en Stock MLX.

Recorre las cinco pantallas siguientes para comprobar la instalación, revisar el enrutamiento de motores, activar funciones de descodificación rápida y arrancar el servidor autónomo sin escribir archivos de configuración a mano.

---

## Paso 1: Descubrir, descargar, actualizar en caliente y gestionar microkernels

Abre **`Integrations → External Engines`** en `oMLX` para gestionar **`miv Core Backend (ninfer-metal)`** y cualquier microkernel comunitario SPI v2 de la Vía B directamente desde la aplicación de macOS:

![Gestor de registro y ciclo de vida de microkernels en Integrations External Engines](../../assets/omlx-mac/integrations-microkernels.png)

- **Microkernel Registry (barra superior)**:
  - Haz clic en **`Refresh`** para obtener el catálogo `microkernels.json` más reciente del repositorio principal y comprobar las versiones remotas en `latest.json`.
  - Selecciona cualquier núcleo en el menú desplegable para inspeccionar justo debajo sus **Modelos soportados (Supported Models)**, topologías, versión SPI y requisito mínimo de macOS antes de descargarlo.
  - Haz clic en **`Download`** (a la derecha de **`Refresh`**) para descargar, verificar SHA-256, descomprimir de forma segura, validar `provider.json` SPI v2 e instalar en `~/Library/Application Support/oMLX/Providers/` (el botón cambia a **`Installed`** cuando ya tienes la última versión gestionada).
- **Lista de microkernels instalados (filas inferiores)**:
  - **Casilla cuadrada izquierda (`☑️`)**: Marcada significa habilitado (por defecto al descargar) y se guarda en `settings.json`. Al desmarcarla, el enrutamiento automático O(1) omite ese núcleo y desaparece del selector por modelo.
  - **Botón `Update` (derecha)**: Atenuado por defecto; solo se ilumina después de que **`Refresh`** detecte una versión remota más reciente. Al pulsarlo, actualiza en caliente volteando de forma atómica el enlace simbólico `current` (los modelos activos siguen respondiendo hasta recargarse).
  - **Botón `Delete` (derecha)**: Elimina un núcleo gestionado bajo `Providers/` tras confirmación (ofreciendo descargar primero los modelos activos). Las rutas locales/externas de compilación mantienen **`Delete`** deshabilitado para evitar borrados accidentales; desmarca la casilla para desactivarlo.
- **Alternativa CLI**: Para servidores sin pantalla o scripts, puedes seguir usando `./omlx-extend/scripts/sync_microkernels.sh --provider io.ninfer.metal` (o `--all` / `--rollback`).
- **Splash**: Permanece justo debajo con todos sus controles existentes intactos.

---

## Paso 2: Ver qué motor ejecuta cada modelo

Cambia a la pantalla **`Models`** para ver cómo `oMLX` enruta cada modelo en tu Mac:

![Pantalla Models con insignias de motor activo y etiquetas en la biblioteca](../../assets/omlx-mac/models-overview.png)

- **Active Models**: Cada modelo cargado muestra una insignia de motor (`miv Core`, el nombre de tu microkernel de la Vía B o `MLX`) justo al lado de **`Loaded`**.
- **Model Library**: Los modelos con pesos `.omlxm5` muestran una etiqueta destacada (`miv Core · Auto`, un microkernel de la Vía B o `Stock MLX` si desactivaste el enrutamiento externo).
- **Modelos sin adaptar intactos**: Los checkpoints MLX estándar no llevan insignia de motor externo y corren sobre Stock MLX con cero procesos sidecar. Puedes mantener cargados simultáneamente modelos de `miv Core Backend`, de la Vía B y de Stock MLX bajo un único límite de memoria unificada.

---

## Paso 3: Enrutamiento automático y vuelta a Stock MLX

Haz clic en cualquier modelo adaptado en **Model Library** y abre **`Advanced Settings`**:

![Model Settings Advanced mostrando el enrutamiento automático a miv Core Backend](../../assets/omlx-mac/engine-routing-auto.png)

- **`Use External Engine`** arranca en **ON** para los modelos `.omlxm5` adaptados, con el selector fijado en **`miv Core Backend (Auto)`** (la cabecera superior de Model Settings también muestra la insignia del motor activo).
- **Volver a Stock MLX en cualquier momento**: Cambia **`Use External Engine`** a **OFF**. `oMLX` guarda `external_engine: "stock"` y muestra *"Using Stock MLX (auto-route overridden)"* sin borrar tus pesos `.omlxm5`. Vuelve a ponerlo en **ON** para recuperar el enrutamiento automático.
- **Elegir otro motor**: Si tienes instalado un microkernel compatible de la Vía B o Splash para el mismo modelo, selecciónalo directamente en el desplegable.

---

## Paso 4: Activar MTP / DFlash2 y KVQuant de 4 bits

`oMLX-extend` reutiliza las pantallas estándar de ajustes de `oMLX`. Los controles compatibles con el microkernel activo se desbloquean a brillo completo; los controles exclusivos de MLX permanecen atenuados para que veas de inmediato qué ejecuta el motor.

### Acceleration (`Lightning MTP` y `DFlash`)

Abre **`Model Settings → Acceleration`**:

![Model Settings Acceleration desbloqueado para miv Core Backend](../../assets/omlx-mac/acceleration-unlocked.png)

- **`Lightning MTP`** (y **`Adaptive max depth`**) se desbloquea cuando el modelo incluye una cabeza MTP integrada.
- **`DFlash`** (con **`DFlash Draft Model`** y **`Runtime Block Size`**) se desbloquea para descodificación especulativa por difusión en bloques.
- Las filas exclusivas de MLX (`Draft Quantization`, `Verify Mode`, `Draft Window Size`, `VLM MTP`) permanecen atenuadas.

### Memory Saving (`4-bit KVQuant`)

Abre **`Model Settings → Memory Saving`**:

![Model Settings Memory Saving desbloqueado para TurboQuant KV Cache Affine4](../../assets/omlx-mac/kvquant-unlocked.png)

- Activa **`TurboQuant KV Cache`** y selecciona **`Affine4 · 4-bit`** para usar la caché KV `affine4_g128` de `miv Core Backend` (`0.5625 B/val`, paridad `0-ULP` verificada hasta contextos de `64K+`).
- Al desactivar el interruptor se utiliza caché KV `bf16` completa. Seleccionar `affine8` o `turboquant` revierte a `bf16` en `miv-v0.01`.
- Las opciones de memoria exclusivas de MLX, como **`MoE Expert Offload`**, permanecen atenuadas.

---

## Paso 5: Configuración compartida sin edición y arranque autónomo en 1 clic

No hace falta escribir un archivo de configuración a mano. Tanto **`oMLX-extend`** como el servidor C++ autónomo (**`ninfer-metal-serve`**) cargan automáticamente [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) con valores predeterminados listos para Apple M5+:

| Ajuste | Valor compartido | Función |
|---|---|---|
| `max_concurrency` | `4` | Ranuras de lote concurrentes multiusuario (`1..8`) |
| `max_context` | `65536` | Longitud de la ventana de contexto (tokens) |
| `kv_dtype` | `"affine4_g128"` | Caché KV de 4 bits grupo 128 (`0-ULP` en `64K+`) |
| `rolling_prefix_cache` / `prefix_cache_gb` | `true` / `4.0` | Reutilización continua de prefijos UMA multiturno (pool de `4 GiB`) |
| `speculative.backend` | `"auto"` | Elige `dflash2` si existe borrador complementario; si no `mtp`; en su defecto AR puro |
| `providers_dir` | `["~/Library/Application Support/oMLX/Providers"]` | Descubre automáticamente microkernels SPI v2 de la Vía B instalados |

Para iniciar el servidor autónomo compatible con OpenAI en el puerto `8080` (separado del puerto `8000` de `oMLX`), ejecuta:

```bash
# Apuntar directamente a un directorio de modelo adaptado
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# U omitir la ruta para elegir automáticamente el primer modelo .omlxm5 en ./models o en la carpeta de modelos de oMLX
./omlx-extend/scripts/start_ninfer_serve.sh
```

**Cómo funcionan las sobrescrituras:**
- En **`oMLX-extend`**: los ajustes por modelo en la interfaz tienen prioridad máxima, seguidos de `[sidecar.models."<name>"]` y después los valores compartidos `[engine]` / `[speculative]` de `ninfer-metal.toml`.
- En **`ninfer-metal-serve`**: las opciones CLI siempre prevalecen sobre `ninfer-metal.toml`. Copia `ninfer-metal.toml` a `~/.config/ninfer-metal/ninfer-metal.toml` cuando quieras valores predeterminados propios que sobrevivan a las actualizaciones.

---

## Siguientes pasos

- **[Inicio y configuración](getting-started.md)** — Compilar desde el código fuente, convertir checkpoints a `.omlxm5` y revisar todas las opciones CLI de `ninfer-metal-serve`.
- **[Arquitectura y coexistencia](architecture.md)** — Cómo funcionan internamente el enrutamiento O(1), Provider SPI v2 y el presupuesto compartido de memoria UMA.
- **[Microkernel Foundry](microkernel-foundry.md)** — Añadir nuevas familias de modelos de la Vía A o la Vía B para Apple M5+.
