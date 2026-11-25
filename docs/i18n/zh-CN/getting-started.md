[English](../../getting-started.md) | **简体中文** | [繁體中文](../zh-TW/getting-started.md) | [日本語](../ja/getting-started.md) | [한국어](../ko/getting-started.md) | [Español](../es/getting-started.md) | [Français](../fr/getting-started.md) | [Deutsch](../de/getting-started.md)

# 快速入门与配置

本文说明如何编译零 MLX 的 `ninfer-metal` 核心（`miv/`）、将权重转换为 `.miv`、一行命令启动原生 C++20 独立 OpenAI 服务（`ninfer-metal-serve`）、将 **`miv 核心后端`** 接入 `oMLX-extend`、调整 `ninfer-metal.toml`，以及调用底层 CLI 和守护进程。如果使用 `oMLX` macOS 应用，先看 **[`oMLX-extend` 可视化上手指南](omlx-extend-guide.md)** 的 5 步截图说明。

---

## 环境要求

| 模式 | 硬件 | 系统与工具链 |
|---|---|---|
| **运行时（预编译包）** | Apple M5 / M5 Pro / M5 Max / M5 Ultra（`MTLGPUFamilyApple10+`） | `macOS 27+`（零 Xcode、Python、MLX 或 Homebrew 依赖） |
| **源码编译** | Apple Silicon（`arm64`） | `macOS 27+`、`Xcode 27+`（macOS 27 SDK）、`CMake 3.27+`、`Ninja` |

---

## 1. 编译 `ninfer-metal`（`miv/`，零 MLX / 零 Python）

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

如需同时编译可选的 `omlx_m5_native` Python 扩展，附加 `-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>`。完整开发预设见 [`CMakePresets.json`](../../../CMakePresets.json)。

---

## 2. 打包发布与权重转换

### 净室 Mac 发布包

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

脚本会生成 `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz`（附带 `.sha256`、`provider.json` 与 `latest.json`），并对全部 16 个 Mach-O 二进制（`ninfer-metal-serve`、`mseries_backendd`、`m5rt_runner` 及 13 个硬件/标定探针）执行 `otool -L` 检查，确认只链接 Apple 系统框架（`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` 与 `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`）。

在任意运行 macOS 27+ 的 Apple M5+ Mac 上解压后，执行 `install.sh`（自动复制默认 `ninfer-metal.toml`，且不覆盖已有自定义修改）即可安装至：

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

### 将模型转换为 `.miv`

离线 NAX 预重排会把 `MXFP4`（`E2M1 × UE8M0`）或 `oQ` Affine（`q3–q8`、`g64/g128`）权重一次性打包，使预填、`T=1` 自回归解码与投机校验共用同一份驻留内存副本：

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.miv \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. 使用 `ninfer-metal-serve` 独立运行（原生 C++）

`ninfer-metal-serve` 是直接链接 `libomlx_m5_core_runtime.a` 的单二进制 C++20 服务。它在同一个 HTTP/SSE 端口并行托管进程内 `miv` 模型与外部路径 B Provider SPI v2 微内核，内置 `ninfer-metal.toml` 配置加载、C++ BPE 分词器（`tokenizer.json`）、Jinja 聊天模板（`chat_template.jinja` / `tokenizer_config.json` / ChatML 回退）、`<think>` 思考段拆分和 Qwen XML/JSON `<tool_call>` 解析，零 Python、零 MLX 依赖。

### 一键启动（免手写共享默认配置）

无需自己编写配置文件即可直接启动 `ninfer-metal-serve`。启动脚本会自动定位 `ninfer-metal-serve` 并加载 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)——`oMLX-extend` 复用的是同一份默认文件（`max_concurrency = 4`、`max_context = 65536`、`kv_dtype = "affine4_g128"`、`rolling_prefix_cache = true`、`prefix_cache_gb = 4.0`、`speculative.backend = "auto"`、`providers_dir = ["~/Library/Application Support/oMLX/Providers"]`）：

```bash
# 指定模型目录一键启动（默认端口 8080，避开 oMLX 的 8000 端口）
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# 省略路径时，自动选取 ./models 或 oMLX 模型目录中的首个 .miv 模型
./omlx-extend/scripts/start_ninfer_serve.sh

# 追加任意命令行参数即可覆盖共享默认配置
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
```

### 命令行直接覆盖或指定自定义 `ninfer-metal.toml`

直接运行 `ninfer-metal-serve` 时，只要未传 `--config` 或 `--no-default-config`，同样会自动加载默认 `ninfer-metal.toml`；命令行参数优先级始终高于 TOML：

```bash
# 直接命令行显式覆盖参数
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# 显式指定 DFlash2 投机解码（或直接依赖默认的 speculative.backend = "auto"）
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# 加载指定的 ninfer-metal.toml（命令行参数优先于 TOML）
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### 并行运行多个微内核（`[[models]]` / `--providers-dir`）

在同一个端口上同时托管进程内 `miv`（路径 A）与路径 B SPI v2 Sidecar 微内核，统一执行 UMA 内存预算上限（`--memory-budget-gb`，默认物理内存的 81%）和自动 `gpu_contended` 调度：

```bash
./build/miv-v001/miv/ninfer-metal-serve \
  --providers-dir "$HOME/Library/Application Support/oMLX/Providers" \
  --model-dir /path/to/Qwen3.8-27B \
  --model-dir /path/to/Llama-3.3-70B-Instruct \
  --memory-budget-gb 64
```

如需逐模型配置（`provider = "auto" | "inproc" | "<provider_id>"`、`kv_dtype`、`max_concurrency`、`memory_budget_gb`），在 `ninfer-metal.toml` 中编写 `[[models]]` 条目并通过 `--config` 启动。请求按 JSON `"model"` 字段路由（未知 ID 返回 `404 model_not_found`）。

### 接入客户端（Open WebUI、Cursor、Cline、NewAPI、OpenAI SDK）

将任意 OpenAI 兼容客户端的 API 地址指向 `http://127.0.0.1:8080/v1`：

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

### HTTP 接口（`miv-v0.01`）

| 接口 | 行为 |
|---|---|
| `GET /health` | 就绪检查 JSON（`status`、`model`、`models[]`、`gpu_contended`、`memory_budget_bytes`、`version`） |
| `GET /metrics` | Prometheus 文本指标（汇总 + 按 `model`/`provider`/`backend_mode` 区分的明细序列：`requests_total`、`active_requests`、`generated_tokens_total`、UMA/缓存统计） |
| `GET /v1/models`、`GET /v1/models/:id` | OpenAI 模型列表，包含 `max_model_len`、`provider_id` 与 `backend_mode` |
| `POST /v1/chat/completions` | 按 JSON `"model"` 路由（未知 ID 返回 `404 model_not_found`；单模型时省略则默认路由到唯一模型）；支持同步 JSON 与流式 SSE（`chat.completion.chunk`）、`<think>...</think>` 拆入 `reasoning_content`、Qwen XML 与 JSON `<tool_call>` 解析、`stop` 停用词、`usage`、`stream_options.include_usage` 及 `[DONE]` |
| `POST /v1/completions` | 按 JSON `"model"` 路由的原始文本补全（同步 JSON、流式 SSE、`usage`、`[DONE]`） |

支持可选 Bearer 鉴权（`--api-key`），校验失败返回 `401`；CORS 预检（`OPTIONS`）默认开启。

### `ninfer-metal-serve` 命令行参数

未传 `--config` 时，`ninfer-metal-serve` 自动加载 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)（`max_concurrency = 4`、`max_context = 65536`、`kv_dtype = "affine4_g128"`、`rolling_prefix_cache = true`、`prefix_cache_gb = 4.0`、`speculative.backend = "auto"`）。传入 `--no-default-config` 可退回下表中的裸编译默认值；完整注释参考见 [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)。

| 参数 | 裸命令行默认值 / 取值 | 说明 |
|---|---|---|
| `--config PATH` | 自动发现 | 显式加载 `ninfer-metal.toml`（`[server]`、`[model]`、`[[models]]`、`[engine]`、`[speculative]`）；命令行参数优先级高于 TOML |
| `--no-default-config` | 关闭 | 跳过默认 `ninfer-metal.toml` 自动加载，仅使用裸编译默认值 |
| `--model-dir PATH` | 必填（或通过 TOML 指定） | 模型目录，包含 `config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 与模型权重；**可重复传入**以并行托管多个模型 |
| `--providers-dir PATH` | 自动（`$OMLX_PROVIDERS_DIR`、`~/Library/Application Support/oMLX/Providers`） | **可重复传入**的路径 B SPI v2 微内核根目录；扫描 `<dir>/*/current/provider.json`（回退 `<dir>/*/provider.json`） |
| `--provider MODE` | `auto`（`auto` \| `inproc` \| `<provider_id>`） | 后端路由模式：O(1) 自动匹配、进程内 `miv`（`inproc`）或指定路径 B SPI v2 `provider_id` |
| `--metallib-dir PATH` | 自动发现 | Metal 4 / NAX `.metallib` 着色器目录（查找顺序：`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`）。别名 `--kernel-dir`（指向 Metal 着色器，**不是**路径 B 微内核；微内核使用 `--providers-dir`） |
| `--artifact PATH` | 自动发现 | 显式指定目标 `.miv`（或对应 Provider）权重文件路径 |
| `--draft-dir PATH`、`--draft-artifact PATH`、`--mtp-artifact PATH` | — | 用于 `dflash2` / `mtp` 的草稿模型目录或 `.miv` 文件 |
| `--chat-template PATH`、`--model-id NAME` | 自动 | 覆盖 Jinja 聊天模板文件或对外暴露的模型 ID |
| `--host ADDR`、`--port PORT` | `127.0.0.1`、`8080` | HTTP 监听地址与端口 |
| `--api-key KEY`、`--cors` / `--no-cors` | `""`、开启 | 开启 `Authorization: Bearer <KEY>` 鉴权；控制 CORS 响应头 |
| `--memory-budget-gb FLOAT` | 物理内存的 `81%` | 跨全部托管模型的服务端总 UMA 内存预算上限（GiB） |
| `--max-concurrency N`（`-c`、`-b`） | `1`（`1..8`） | 多用户并发批槽位数（`B`/`C`） |
| `--max-context N` | `32768` | 上下文窗口长度（`L`） |
| `--kv-dtype MODE` | `bf16`（`bf16` \| `affine4_g128`） | KV 缓存存储格式 |
| `--spec-backend MODE` | `none`（`auto` \| `none` \| `mtp` \| `dflash2`） | 投机解码后端（`auto` 在存在配套 `dflash2/` 时选 `dflash2`，其次 `mtp`，否则 `none`） |
| `--draft-tokens N` | `0`（自动；可显式设 `1..7`） | 每轮校验生成的草稿 Token 数 |
| `--rolling-prefix-cache on\|off` | 未设（沿用引擎默认） | 开启或关闭多轮滚动 UMA 前缀复用与 `1×` 祖先原地覆写 |
| `--prefix-cache-gb FLOAT` | 未设（`0..128`） | 前缀缓存与 UMA 检查点池内存预算（GiB） |
| `--sampling-contract NAME` | `legacy_rejection_v1` | 投机校验采样契约（`legacy_rejection_v1` \| `position_coupled_v1`） |
| `--version`、`--help`（`-h`） | — | 打印结构化版本信息或命令行帮助 |

---

## 4. 配合 `oMLX-extend` 运行（Sidecar 共存）

本仓在 `omlx-extend/` 旁直接维护已打补丁的 `omlx/` 宿主代码，并通过 [`omlx-extend/patches/`](../../../omlx-extend/patches/) 与上游 `jundot/omlx` 同步。

```bash
# Verify or apply the Provider SPI patch on ./omlx and run pytest
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test

# Sync ./omlx to a specific upstream jundot/omlx tag
./omlx-extend/scripts/sync_omlx.sh --tag v0.8.3 --omlx-dir ./omlx
```

安装好 `io.ninfer.metal/current/` 后，`oMLX-extend` 会自动发现 **`miv 核心后端`**（`mseries_backendd`，按顺序回退查找 `com.omlx.mseries/current/`、`PATH` 或 `build/*/miv/`），并在 **集成 → 外部引擎**（`Integrations → External Engines`）中展示已安装的路径 A 与路径 B 微内核卡片。目录内含 `model.miv` / `model.omlxm5`、`model_pack.json` 或 `omlxm5-manifest.json` 的模型会懒加载路由到 **`miv 核心后端`**，未适配模型原样留在 Stock MLX。若要让已适配模型改跑原生 MLX，在 **模型设置 → 高级设置** 关闭 **`Use External Engine`** 开关（写入 `external_engine: "stock"`）即可。

macOS 应用界面图文指引见 **[`oMLX-extend` 可视化上手指南](omlx-extend-guide.md)**；宿主内部机制与 UI 设置映射见 [`omlx-extend/README.md`](../../../omlx-extend/README.md)。

---

## 5. 配置 `ninfer-metal.toml`

独立服务 `ninfer-metal-serve` 与 `oMLX-extend` 开箱共用同一份默认配置 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)，无需手动编写。未显式传 `--config` 时，两种模式按同一优先级从高到低查找首个存在的文件：

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml`（可重定位发布包）
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

如需跨版本保留自定义默认值，将默认配置（或带注释模板 [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)）复制到 `~/.config/ninfer-metal/` 后修改：

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

在 `oMLX-extend` Sidecar 模式下，界面上的单模型设置（`external_engine_options["ninfer-metal"]`）优先于 TOML 中的 `[sidecar.models."<name>"]`，再优先于共享 `[engine]` / `[speculative]` 默认值。在独立 `ninfer-metal-serve` 下，命令行参数始终覆盖 TOML。

### 核心参数

| 键名 | 配置节 | 取值（`miv-v0.01`） | 说明 |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`、`8080`、`""` | 独立 `ninfer-metal-serve` 服务参数（`oMLX-extend` AF_UNIX Sidecar 模式忽略）。 |
| `providers_dir` / `memory_budget_gb` | `[server]` | 字符串或数组，浮点数（GiB） | 路径 B SPI v2 微内核根目录（扫描 `<dir>/*/current/provider.json`）与服务端总 UMA 上限（默认物理内存的 `81%`）。 |
| `model_dir` / `artifact`（`artifact_path`） | `[model]` | 路径 | 独立服务模型目录及可选的显式 `.miv`（或对应 Provider）权重路径；`artifact` 与 `artifact_path`（以及 `draft_artifact`/`draft_artifact_path`、`metallib_dir`/`kernel_dir`）均支持。 |
| `id` / `model_dir` / `provider` / 覆盖项 | `[[models]]` | 表数组 | 独立 `ninfer-metal-serve` 多模型并行托管条目。未填写项继承 `[engine]`/`[speculative]` 默认值，通过 `provider = "auto" \| "inproc" \| "<provider_id>"` 选择后端。 |
| `max_concurrency` | `[engine]` | `1..8`（裸 CLI 默认 `1`，模板 `4`） | 多用户并发批槽位数（`miv` 文档记作 **`B`**，上游 `ninfer` 记作 **`C`**）。Sidecar 模式下留空则继承 oMLX 的 `max_concurrent_requests`。 |
| `max_context` | `[engine]` | `32768`–`262144+`（裸 CLI 默认 `32768`，模板 `65536`） | 上下文窗口长度（**`L`**）。Sidecar 模式下映射自 oMLX `max_context_window`。 |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128` 占用 `0.5625 B/val`，在 `64K+` 下通过 `0-ULP` 校验。`miv-v0.01` 不含 `int8_g64`。 |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | 开启多轮滚动 UMA 前缀复用与 `1×` 祖先原地覆写。 |
| `prefix_cache_gb` | `[engine]` | `0`–`128`（模板 `4.0`） | 前缀缓存与 UMA 检查点池内存预算（GiB）。 |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`，`0`（自动）或 `1..7` | `"auto"` 存在配套 `dflash2/` 时选 `dflash2`，其次 `mtp`，否则 `none`。Sidecar 模式下映射自 oMLX 加速选项（`mtp_*`、`dflash_*`）。 |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"`（裸 CLI 默认） \| `"position_coupled_v1"` | 投机校验采样契约。 |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | 预填执行后端选择。 |

默认配置：[`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) · 完整带注释模板：[`omlx-extend/config/ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)。

---

## 6. 底层 CLI、守护进程与 C++ 静态库

除 `ninfer-metal-serve` 和 `oMLX-extend` 外，`miv/` 还暴露四个更底层的集成接口：

1. **`m5rt_runner`（Token ID 命令行工具）** — 直接输入整型 Token ID，执行单流、批处理（`--batch-mode on --max-concurrency 4`）或 DFlash2 投机推理，以 NDJSON 输出指标与生成 Token：
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.miv \
     --config /path/to/config.json \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd`（AF_UNIX NDJSON RPC v2 守护进程）** — 常驻 Token ID 推理服务端，支持 `hello`、`load`、`submit`、`cancel`、`subscribe_events`、`trim_cache`、`set_scheduler_hint` 与 `shutdown`：
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a`（C++20 静态库）** — 通过 [`miv/include/omlx_m5/engine.h`](../../../miv/include/omlx_m5/engine.h) 直接嵌入 `omlx::m5::Engine`。
4. **`omlx_m5_native`（可选 Python 模块）** — 编译时开启 `-DOMLX_M5_BUILD_PYTHON=ON` 生成的 Nanobind 绑定。
