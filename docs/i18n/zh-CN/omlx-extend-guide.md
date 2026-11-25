[English](../../omlx-extend-guide.md) | **简体中文** | [繁體中文](../zh-TW/omlx-extend-guide.md) | [日本語](../ja/omlx-extend-guide.md) | [한국어](../ko/omlx-extend-guide.md) | [Español](../es/omlx-extend-guide.md) | [Français](../fr/omlx-extend-guide.md) | [Deutsch](../de/omlx-extend-guide.md)

# `oMLX-extend` 可视化上手指南

`oMLX-extend` 为 Apple M5+（`macOS 27+`）上的 `oMLX` macOS 应用接入 **`miv 核心后端（ninfer-metal）`** 与社区路径 B Metal 4 微内核。已转换 `.omlxm5` 权重的模型会自动走原生 Metal 4 NAX 推理，未适配的模型照常留在原生 MLX 上运行。

跟着下面 5 个界面走一遍，就能确认微内核状态、看清每个模型在用哪个引擎、打开加速与 4-bit KV 量化，并用现成的共享配置一键拉起独立服务。

---

## 第 1 步：发现、下载、热更新与管理微内核

在 `oMLX` 中打开 **集成 → 外部引擎**（`Integrations → External Engines`），即可在 macOS 界面内直接管理 **`miv 核心后端（ninfer-metal）`** 及后续接入的社区路径 B SPI v2 微内核：

![Integrations External Engines 微内核主库与管理面板](../../assets/omlx-mac/integrations-microkernels.png)

- **微内核主库（顶部操作区）**：
  - 点击 **`刷新`**，实时拉取主库最新核心清单（`microkernels.json`）并核对各核心的远端 `latest.json` 版本。
  - 在下拉清单中切换任意核心，下拉框下方会立即展示该核心的 **支持模型**、拓扑与最低系统要求，方便判断是否需要下载。
  - 点击刷新右侧的 **`下载`**，自动完成拉包、SHA-256 校验、防穿越安全解压与 SPI v2 校验，并安装到 `~/Library/Application Support/oMLX/Providers/`（已安装最新版本时自动变为 **`已安装`** 置灰态）。
- **已安装核心列表（下方列表区）**：
  - **左侧方形勾选框（`☑️`）**：勾选即启用（新下载的核心默认勾选）并持久化到 `settings.json`；取消勾选后，O(1) 自动路由和单模型引擎下拉菜单会立即跳过该核心。
  - **右侧 `更新` 按钮**：平时保持置灰不可点，**只有在点击 `刷新` 且检测到远端新版本时才会亮起**；点击即可后台下载并原子切换 `current` 软链完成热更新（运行中的模型不受打断，重载模型后切换至新核）。
  - **右侧 `删除` 按钮**：确认后清理 `Providers/` 下的托管安装目录（若当前有模型正在运行，会提示先卸载再删除）；外部或本地编译路径挂载的核心会置灰禁用删除以防误删，可取消勾选停用。
- **命令行方式（可选）**：终端或自动化部署仍可直接运行 `./omlx-extend/scripts/sync_microkernels.sh --provider io.ninfer.metal`（支持 `--all` 与 `--rollback`）。
- **Splash**：保留在下方，原有选项完全不变。

---

## 第 2 步：查看每个模型正在运行哪个引擎

切到 **模型**（`Models`）页面，可以一眼分辨每个模型实际使用的后端：

![Models 页面引擎状态徽章与模型库标签](../../assets/omlx-mac/models-overview.png)

- **活跃模型**（`Active Models`）：每个正在运行的模型在绿色 **`Loaded`** 旁边都会标出当前引擎徽章（`miv Core`、路径 B 微内核名或 `MLX`）。
- **模型库**（`Model Library`）：带 `.omlxm5` 权重的模型会高亮显示匹配标签（`miv Core · Auto`、路径 B 微内核，或手动关闭外部引擎后的 `Stock MLX`）。
- **未适配模型零干扰**：普通 MLX 模型不带外部引擎徽章，直接走原生 MLX，不会启动任何 Sidecar 进程。`miv 核心后端`、路径 B 微内核与原生 MLX 模型可以在同一个统一内存上限下同时加载。

---

## 第 3 步：自动路由与切回原生 MLX

在模型库点击任意已适配模型，进入 **高级设置**（`Model Settings → Advanced Settings`）：

![Model Settings Advanced 自动路由到 miv 核心后端](../../assets/omlx-mac/engine-routing-auto.png)

- 对已适配 `.omlxm5` 的模型，**使用外部引擎**（`Use External Engine`）开关默认处于**开启**状态，下拉菜单自动选中 **`miv Core Backend (Auto)`**（上一级模型设置页头也会同步显示当前引擎徽章）。
- **随时切回原生 MLX**：把 **`Use External Engine`** 开关**关闭**即可。`oMLX` 会写入 `external_engine: "stock"` 并提示 *"Using Stock MLX (auto-route overridden)"*，不会删除 `.omlxm5` 文件；重新打开开关即可恢复自动路由。
- **切换其他引擎**：如果同一模型还安装了兼容的路径 B 微内核或 Splash，直接在同一个下拉菜单里切换。

---

## 第 4 步：开启 MTP / DFlash2 与 4-bit KV 量化

`oMLX-extend` 直接沿用 `oMLX` 原生设置页：当前微内核支持的开关会正常点亮，仅原生 MLX 使用的选项保持置灰，避免误调无效项。

### 推理加速（`Lightning MTP` 与 `DFlash`）

打开 **模型设置 → 加速**（`Model Settings → Acceleration`）：

![Model Settings Acceleration 解锁 MTP 与 DFlash 选项](../../assets/omlx-mac/acceleration-unlocked.png)

- 模型自带 MTP 头时，**`Lightning MTP`**（及 **`Adaptive max depth`**）自动解锁。
- 需要块扩散投机解码时，**`DFlash`**（含 **`DFlash Draft Model`** 与 **`Runtime Block Size`**）处于可用状态。
- 仅限 MLX 的选项（`Draft Quantization`、`Verify Mode`、`Draft Window Size`、`VLM MTP`）保持置灰。

### 内存节省（4-bit KV 量化）

打开 **模型设置 → 内存节省**（`Model Settings → Memory Saving`）：

![Model Settings Memory Saving 解锁 TurboQuant KV Cache Affine4](../../assets/omlx-mac/kvquant-unlocked.png)

- 打开 **`TurboQuant KV Cache`** 并选择 **`Affine4 · 4-bit`**，即可启用 `miv 核心后端` 的 `affine4_g128` KV 缓存（`0.5625 B/val`，`64K+` 上下文通过 `0-ULP` 校验）。
- 关闭开关即使用完整 `bf16` KV 缓存；在 `miv-v0.01` 中选择 `affine8` 或 `turboquant` 会回退到 `bf16`。
- 仅限 MLX 的选项（例如 **`MoE Expert Offload`**）保持置灰。

---

## 第 5 步：免手写共享配置与一键独立启动

不需要自己从头写配置文件。**`oMLX-extend`** 与独立 C++ 服务（**`ninfer-metal-serve`**）开箱自动加载同一份 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)，已为 Apple M5+ 预设好默认值：

| 配置项 | 共享默认值 | 作用 |
|---|---|---|
| `max_concurrency` | `4` | 多用户并发批槽位数（`1..8`） |
| `max_context` | `65536` | 上下文窗口长度（Token 数） |
| `kv_dtype` | `"affine4_g128"` | 4-bit 分组 128 KV 缓存（`64K+` `0-ULP`） |
| `rolling_prefix_cache` / `prefix_cache_gb` | `true` / `4.0` | 多轮对话 UMA 前缀复用（`4 GiB` 池） |
| `speculative.backend` | `"auto"` | 存在配套 `dflash2/` 时自动启用 DFlash2，其次 `mtp`，否则纯自回归 |
| `providers_dir` | `["~/Library/Application Support/oMLX/Providers"]` | 自动扫描已安装的路径 B SPI v2 微内核 |

想在 `8080` 端口直接启动独立 OpenAI 兼容服务（与 `oMLX` 的 `8000` 端口互不冲突），运行一行命令：

```bash
# 指定已转换的 .omlxm5 模型目录
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# 不传路径时，自动扫描 ./models 或 oMLX 模型目录中的首个 .omlxm5 模型
./omlx-extend/scripts/start_ninfer_serve.sh
```

**参数优先级：**
- **`oMLX-extend` 模式**：界面上的单模型设置 > TOML 中的 `[sidecar.models."<name>"]` > 共享 `ninfer-metal.toml` 中的 `[engine]` / `[speculative]` 默认值。
- **`ninfer-metal-serve` 独立模式**：命令行参数始终覆盖 `ninfer-metal.toml`。如果要保留长期自定义配置，复制到 `~/.config/ninfer-metal/ninfer-metal.toml` 修改即可，后续更新不会覆盖它。

---

## 下一步

- **[快速入门与配置](getting-started.md)** — 源码编译、`.omlxm5` 权重转换以及 `ninfer-metal-serve` 全部命令行参数。
- **[架构与多模型共存](architecture.md)** — 了解 O(1) 自动路由、Provider SPI v2 与共享 UMA 内存预算的工作原理。
- **[微内核工坊](microkernel-foundry.md)** — 为 Apple M5+ 贡献新的路径 A 或路径 B 模型家族微内核。
