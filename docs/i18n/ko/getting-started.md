[English](../../getting-started.md) | [简体中文](../zh-CN/getting-started.md) | [繁體中文](../zh-TW/getting-started.md) | [日本語](../ja/getting-started.md) | **한국어** | [Español](../es/getting-started.md) | [Français](../fr/getting-started.md) | [Deutsch](../de/getting-started.md)

# 시작 가이드 및 설정

제로 MLX `ninfer-metal` 코어(`miv/`) 빌드, `.miv` 가중치 변환, 독립 실행형 C++20 OpenAI 서버(`ninfer-metal-serve`) 원클릭 실행, **`miv Core Backend`**의 `oMLX-extend` 연동, `ninfer-metal.toml` 사용자 지정, 저수준 CLI 및 데몬 호출 방법을 다룹니다. `oMLX` macOS 앱을 사용한다면 5단계 스크린샷 안내인 **[`oMLX-extend` 비주얼 가이드](omlx-extend-guide.md)**부터 시작하세요.

---

## 사전 요구 사항

| 모드 | 하드웨어 | OS 및 툴체인 |
|---|---|---|
| **런타임 (사전 빌드 번들)** | Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`) | `macOS 27+` (Xcode, Python, MLX, Homebrew 무의존) |
| **소스 빌드** | Apple Silicon (`arm64`) | `macOS 27+`, `Xcode 27+` (macOS 27 SDK), `CMake 3.27+`, `Ninja` |

---

## 1. `ninfer-metal` 빌드 (`miv/`, 제로 MLX / 제로 Python)

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

선택적 Python 확장 모듈 `omlx_m5_native`를 함께 빌드하려면 `-DOMLX_M5_BUILD_PYTHON=ON -DPython_EXECUTABLE=<path-to-python-with-nanobind>` 옵션을 추가합니다. 전체 개발 프리셋은 [`CMakePresets.json`](../../../CMakePresets.json)에 정의되어 있습니다.

---

## 2. 패키징 및 가중치 변환

### 클린 Mac용 릴리스 번들

```bash
python3 miv/tools/build_m5_release.py \
  --build-dir build/miv-v001 \
  --version miv-v0.01 \
  --output-dir dist \
  --allow-no-python
```

이 명령은 `dist/ninfer-metal-miv-v0.01-macos27-arm64.tar.gz`(및 `.sha256`, `provider.json`, `latest.json`)를 생성하고, Mach-O 바이너리 16개 전체(`ninfer-metal-serve`, `mseries_backendd`, `m5rt_runner`, 하드웨어·캘리브레이션 프로브 13개)에 `otool -L`을 실행해 Apple 시스템 프레임워크(`/System/Library/Frameworks/{CoreFoundation,Foundation,IOKit,Metal}.framework` 및 `/usr/lib/{libc++.1,libSystem.B,libobjc.A}.dylib`)만 링크되었는지 검증합니다.

macOS 27+가 설치된 Apple M5+ Mac에서 아카이브를 풀고 `install.sh`(기존 사용자 수정본을 덮어쓰지 않고 기본 `ninfer-metal.toml` 복사)를 실행하면 다음 경로에 설치됩니다:

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

### 모델을 `.miv`로 변환

오프라인 NAX 프리타일링은 `MXFP4`(`E2M1 × UE8M0`) 또는 `oQ` Affine(`q3–q8`, `g64/g128`) 가중치를 한 번만 패킹해 Prefill, `T=1` AR 디코딩, 투기적 검증이 단일 상주 메모리 사본을 공유하도록 구성합니다:

```bash
python3 miv/tools/m5_artifact.py convert \
  --source <model_dir> \
  --output <model_dir>/model.miv \
  --source-repo <repo_id> \
  --nax-pretile
```

---

## 3. `ninfer-metal-serve` 단독 실행 (네이티브 C++)

`ninfer-metal-serve`는 `libomlx_m5_core_runtime.a`에 직접 링크된 단일 C++20 바이너리입니다. Python과 MLX 없이 POSIX HTTP/SSE, `ninfer-metal.toml` 로딩, C++ BPE 토크나이저(`tokenizer.json`), Jinja 채팅 템플릿(`chat_template.jinja` / `tokenizer_config.json` / ChatML 폴백), `<think>` 추론 내용 분리, Qwen XML/JSON `<tool_call>` 파서를 내장합니다.

### 원클릭 실행 (편집 없는 공유 설정)

설정 파일을 손으로 작성하지 않고 바로 `ninfer-metal-serve`를 시작할 수 있습니다. 실행 스크립트는 `ninfer-metal-serve`를 자동 탐색하고 `oMLX-extend`가 그대로 재사용하는 기본 설정 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)(`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`, `providers_dir = ["~/Library/Application Support/oMLX/Providers"]`)을 불러옵니다:

```bash
# 특정 모델 디렉터리로 원클릭 실행 (기본 포트 8080, oMLX 포트 8000과 분리)
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B

# 모델 디렉터리를 생략하면 ./models 또는 oMLX 모델 폴더의 첫 번째 .miv 모델을 자동 선택
./omlx-extend/scripts/start_ninfer_serve.sh

# 추가 CLI 플래그를 전달해 공유 기본 설정 덮어쓰기
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B --port 8081 --kv-dtype bf16
```

### 직접 CLI 재정의 또는 사용자 지정 `ninfer-metal.toml`

`ninfer-metal-serve`를 직접 실행해도 `--config`나 `--no-default-config`를 지정하지 않으면 공유 기본 `ninfer-metal.toml`이 자동 로드되며, CLI 플래그는 항상 TOML 값을 덮어씁니다:

```bash
# 명시적 재정의를 포함한 직접 CLI 실행
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --host 127.0.0.1 \
  --port 8080 \
  --max-concurrency 4 \
  --max-context 65536 \
  --kv-dtype affine4_g128 \
  --rolling-prefix-cache on \
  --prefix-cache-gb 4.0

# DFlash2 투기적 디코딩 명시 실행 (또는 speculative.backend = "auto" 사용)
./build/miv-v001/miv/ninfer-metal-serve \
  --model-dir /path/to/Qwen3.8-27B \
  --spec-backend dflash2 \
  --draft-dir /path/to/Qwen3.8-27B-DFlash2 \
  --draft-tokens 7 \
  --kv-dtype affine4_g128

# 특정 설정 파일 로드 (CLI 플래그가 TOML 값을 재정의)
./build/miv-v001/miv/ninfer-metal-serve --config ~/.config/ninfer-metal/ninfer-metal.toml
```

### 여러 마이크로커널 병렬 호스팅 (`--providers-dir` / `[[models]]`)

단일 `ninfer-metal-serve` 포트에서 인프로세스 `miv`(경로 A)와 외부 SPI v2 마이크로커널 사이드카(경로 B)를 병렬 실행할 수 있습니다. `--model-dir`을 여러 번 전달하거나 TOML에서 `[[models]]` 항목을 나열합니다:

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

### 클라이언트 연결 (Open WebUI, Cursor, Cline, NewAPI, OpenAI SDK)

OpenAI 호환 클라이언트의 베이스 URL을 `http://127.0.0.1:8080/v1`로 지정합니다:

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

### HTTP 엔드포인트 (`miv-v0.01`)

| 엔드포인트 | 동작 |
|---|---|
| `GET /health` | 준비 상태 JSON (`status`, `model`, `models[]`, `gpu_contended`, `memory_budget_bytes`, `version`) |
| `GET /metrics` | Prometheus 텍스트 지표 (전체 집계 및 `model`, `provider`, `backend_mode` 라벨별 시계열) |
| `GET /v1/models`, `GET /v1/models/:id` | `max_model_len`, `provider_id`, `backend_mode`가 포함된 OpenAI 모델 목록 |
| `POST /v1/chat/completions` | JSON `"model"` 기반 라우팅(미등록 모델 시 `404 model_not_found`, 단일 모델 시 생략 가능), 동기 JSON 및 스트리밍 SSE(`chat.completion.chunk`), `<think>...</think>`의 `reasoning_content` 분리, Qwen XML/JSON `<tool_call>` 파싱, `stop`, `usage`, `stream_options.include_usage`, `[DONE]` |
| `POST /v1/completions` | 원시 텍스트 완성(JSON `"model"` 라우팅, 동기 JSON, 스트리밍 SSE, `usage`, `[DONE]`) |

선택적 Bearer 인증(`--api-key`)은 불일치 시 `401`을 반환하며, CORS 프리플라이트(`OPTIONS`)는 기본적으로 활성화되어 있습니다.

### `ninfer-metal-serve` CLI 플래그

`--config`를 생략하면 `ninfer-metal-serve`는 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)(`max_concurrency = 4`, `max_context = 65536`, `kv_dtype = "affine4_g128"`, `rolling_prefix_cache = true`, `prefix_cache_gb = 4.0`, `speculative.backend = "auto"`)을 자동 로드합니다. 아래에 나열된 순수 컴파일 기본값으로 시작하려면 `--no-default-config`를 전달하세요. 주석이 달린 전체 레퍼런스는 [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example)을 참고하세요.

| 플래그 | 순수 CLI 기본값 / 허용값 | 설명 |
|---|---|---|
| `--config PATH` | 자동 탐색 | 명시적 `ninfer-metal.toml`(`[server]`, `[model]`, `[[models]]`, `[engine]`, `[speculative]`) 로드. CLI 플래그가 TOML 값을 재정의 |
| `--no-default-config` | 꺼짐 | 자동 `ninfer-metal.toml` 탐색을 건너뛰고 순수 컴파일 기본값으로 시작 |
| `--model-dir PATH` | 필수 (또는 TOML 지정) | `config.json`, `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `model.miv`가 있는 모델 디렉터리. 반복 지정으로 단일 포트 병렬 호스팅 |
| `--providers-dir PATH` | 자동 (`$OMLX_PROVIDERS_DIR`, `~/Library/Application Support/oMLX/Providers`) | 경로 B SPI v2 마이크로커널 사이드카 검색 루트(반복 지정 가능). `<dir>/*/current/provider.json`(폴백 `<dir>/*/provider.json`)을 스캔 |
| `--provider MODE` | `auto` (`auto` \| `inproc` \| `<provider_id>`) | 백엔드 선택: O(1) 자동 라우팅(`auto`), 인프로세스 `miv`(`inproc`), 또는 명시적 SPI v2 `<provider_id>` |
| `--metallib-dir PATH` (별칭: `--kernel-dir`) | 자동 탐지 | 컴파일된 Metal 4 / NAX `.metallib` 셰이더 디렉터리(`$OMLX_M5_METALLIB_DIR` -> `<exe>/../metallib` -> `<exe>/metallib`). Metal 셰이더용이며 SPI v2 마이크로커널 프로바이더가 아님 |
| `--artifact PATH` | 자동 탐지 | 대상 `.miv`(또는 프로바이더 아티팩트) 명시 경로 |
| `--draft-dir PATH`, `--draft-artifact PATH`, `--mtp-artifact PATH` | — | `dflash2` / `mtp`용 드래프트 모델 디렉터리 또는 `.miv` 아티팩트 |
| `--chat-template PATH`, `--model-id NAME` | 자동 | Jinja 채팅 템플릿 파일 또는 노출 모델 ID 재정의 |
| `--host ADDR`, `--port PORT` | `127.0.0.1`, `8080` | HTTP 바인드 주소 및 포트 |
| `--api-key KEY`, `--cors` / `--no-cors` | `""`, 활성화 | `Authorization: Bearer <KEY>` 인증 요구 및 CORS 헤더 제어 |
| `--memory-budget-gb FLOAT` | 자동 (물리 RAM의 `81%`) | `MultiModelHost`가 전체 모델 간 조정하는 서버 전체 UMA 메모리 예산 상한 (GiB) |
| `--max-concurrency N` (`-c`, `-b`) | `1` (`1..8`) | 다중 사용자 동시 배치 슬롯 수 (`B`/`C`) |
| `--max-context N` | `32768` | 컨텍스트 윈도우 길이 (`L`) |
| `--kv-dtype MODE` | `bf16` (`bf16` \| `affine4_g128`) | KV 캐시 저장 포맷 |
| `--spec-backend MODE` | `none` (`auto` \| `none` \| `mtp` \| `dflash2`) | 투기적 디코딩 백엔드 (`auto`는 동반 `dflash2/`가 있으면 `dflash2`, 없으면 `mtp`, 둘 다 없으면 `none` 선택) |
| `--draft-tokens N` | `0` (자동, 명시 시 `1..7`) | 검증 라운드당 드래프트 토큰 수 (`0` = 자동) |
| `--rolling-prefix-cache on\|off` | 런타임 기본값 | `1×` 상위 엔트리 교체를 포함한 멀티턴 롤링 UMA 프리픽스 재사용 |
| `--prefix-cache-gb FLOAT` | 런타임 기본값 (`0..128`) | 프리픽스 캐시 및 UMA 체크포인트 풀 예산 (GiB) |
| `--sampling-contract NAME` | `legacy_rejection_v1` | 투기적 검증 샘플링 계약 (`legacy_rejection_v1` \| `position_coupled_v1`) |
| `--version`, `--help` | — | 구조화 버전 문자열 또는 사용법 출력 |

---

## 4. `oMLX-extend`와 함께 실행 (사이드카 공존)

이 모노레포는 패치된 `omlx/` 호스트를 `omlx-extend/`와 함께 추적하며, [`omlx-extend/patches/`](../../../omlx-extend/patches/)를 통해 업스트림 `jundot/omlx`와 동기화합니다.

```bash
# Check the Provider SPI patch against a separate clean checkout of the pinned upstream tag
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir /path/to/upstream-omlx --check-only

# Sync the vendored ./omlx tree to the latest upstream jundot/omlx release
./omlx-extend/scripts/sync_omlx.sh
```

`io.ninfer.metal/current/`가 설치되어 있으면 `oMLX-extend`가 **`miv Core Backend`**(`mseries_backendd`, 폴백 경로: `com.omlx.mseries/current/`, `PATH`, `build/*/miv/`)를 자동 탐지하고 설치된 경로 A 및 경로 B 마이크로커널을 **`Integrations → External Engines`**에 표시합니다. `model.miv` / `model.omlxm5`, `model_pack.json`, `omlxm5-manifest.json` 중 하나가 있는 모델 디렉터리는 첫 요청 시 지연 방식으로 **`miv Core Backend`**에 라우팅되며, 미변환 모델은 Stock MLX에서 그대로 실행됩니다. 변환된 모델에 Stock MLX를 강제하려면 **`Model Settings → Advanced Settings`**에서 **`Use External Engine`** 스위치를 끄세요(`external_engine: "stock"` 기록).

`oMLX` macOS 앱의 스크린샷 가이드는 **[`oMLX-extend` 비주얼 가이드](omlx-extend-guide.md)**를, 호스트 내부 구조와 UI 설정 매핑은 [`omlx-extend/README.md`](../../../omlx-extend/README.md)를 참고하세요.

---

## 5. `ninfer-metal.toml` 설정

독립 실행형 `ninfer-metal-serve`와 `oMLX-extend`는 편집 없이 바로 동작하는 동일한 기본 설정([`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml))을 재사용합니다. `--config`를 전달하지 않으면 두 호스트 모두 다음 우선순위대로 탐색해 먼저 찾은 파일을 사용합니다:

1. `$NINFER_METAL_CONFIG`
2. `./ninfer-metal.toml`
3. `~/.config/ninfer-metal/ninfer-metal.toml`
4. `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/config/ninfer-metal.toml`
5. `<exe_dir>/../config/ninfer-metal.toml` (재배치 가능한 릴리스 번들)
6. `<repo_root>/omlx-extend/config/ninfer-metal.toml`

업데이트 후에도 사용자 지정 기본값을 유지하려면 기본 설정(또는 주석이 포함된 [`ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example))을 `~/.config/ninfer-metal/`에 복사합니다:

```bash
mkdir -p ~/.config/ninfer-metal
cp omlx-extend/config/ninfer-metal.toml ~/.config/ninfer-metal/ninfer-metal.toml
```

`oMLX-extend` 사이드카 모드에서는 모델별 UI 설정(`external_engine_options["ninfer-metal"]`)이 TOML의 `[sidecar.models."<name>"]`을 덮어쓰고, 이는 다시 공유 `[engine]` / `[speculative]` 기본값을 덮어씁니다. 독립 실행형 `ninfer-metal-serve`에서는 CLI 플래그가 항상 TOML 값을 덮어씁니다.

### 주요 설정 항목

| 키 | 섹션 | 허용값 (`miv-v0.01`) | 비고 |
|---|---|---|---|
| `host` / `port` / `api_key` | `[server]` | `"127.0.0.1"`, `8080`, `""` | 독립 `ninfer-metal-serve` 전용 설정 (`oMLX-extend` AF_UNIX 사이드카에서는 무시) |
| `providers_dir` / `memory_budget_gb` | `[server]` | 경로 (문자열 또는 배열), GiB 실수 | 경로 B SPI v2 프로바이더 검색 경로 및 서버 전체 UMA 예산 상한 (기본값: 물리 RAM의 `81%`) |
| `model_dir` / `artifact` (또는 `artifact_path`) | `[model]` / `[[models]]` | 경로 | 독립 서버용 모델 디렉터리 및 선택적 `.miv` 명시 경로(`artifact`/`artifact_path`, `draft_artifact`/`draft_artifact_path`, `metallib_dir` 모두 허용). `[[models]]` 배열로 다중 모델 병렬 정의 |
| `max_concurrency` | `[engine]` | `1..8` (기본 CLI `1`, TOML 템플릿 `4`) | 다중 사용자 동시 배치 슬롯 수 (`miv` 문서에서는 **`B`**, 업스트림 `ninfer`에서는 **`C`**). 사이드카 모드에서 생략 시 oMLX `max_concurrent_requests` 상속 |
| `max_context` | `[engine]` | `32768`–`262144+` (기본 CLI `32768`) | 컨텍스트 윈도우 길이 (**`L`**). 사이드카 모드에서 oMLX `max_context_window`와 매핑 |
| `kv_dtype` | `[engine]` | `"bf16"` \| `"affine4_g128"` | `affine4_g128`은 `0.5625 B/val`로 저장하며 `64K+`에서 `0-ULP` 일치 검증 완료. `miv-v0.01`에는 `int8_g64` 미포함 |
| `rolling_prefix_cache` | `[engine]` | `true` \| `false` | `1×` 인플레이스 상위 엔트리 교체를 포함한 멀티턴 롤링 UMA 프리픽스 재사용 활성화 |
| `prefix_cache_gb` | `[engine]` | `0`–`128` (TOML 템플릿 `4.0`) | 프리픽스 캐시 및 UMA 체크포인트 풀 예산 (GiB) |
| `backend` / `draft_tokens` | `[speculative]` | `"auto"` \| `"none"` \| `"mtp"` \| `"dflash2"`, `0` (자동) 또는 `1..7` | `"auto"`는 동반 `dflash2/`가 있으면 `dflash2`, 없으면 `mtp`, 둘 다 없으면 `none` 선택. 사이드카 모드에서 oMLX Acceleration 토글(`mtp_*`, `dflash_*`)과 매핑 |
| `sampling_contract` | `[engine]` | `"legacy_rejection_v1"` \| `"position_coupled_v1"` | 투기적 검증 샘플링 계약 (기본 CLI `"legacy_rejection_v1"`, TOML 템플릿 `"position_coupled_v1"`) |
| `prefill_engine` | `[engine]` | `"auto"` \| `"nax"` \| `"nax_ane"` | Prefill 실행 백엔드 선택 |

기본 설정: [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml) · 주석이 포함된 전체 템플릿: [`omlx-extend/config/ninfer-metal.toml.example`](../../../omlx-extend/config/ninfer-metal.toml.example).

---

## 6. 저수준 CLI, 데몬 및 C++ 라이브러리

`ninfer-metal-serve`와 `oMLX-extend` 외에도 `miv/`는 네 가지 저수준 연동 인터페이스를 제공합니다:

1. **`m5rt_runner` (토큰 ID CLI)** — 정수 토큰 ID를 입력받아 단일 스트림, 배치(`--batch-mode on --max-concurrency 4`), 또는 DFlash2 투기적 추론을 실행하고 NDJSON 지표와 토큰을 출력합니다:
   ```bash
   ./build/miv-v001/miv/m5rt_runner \
     --artifact /path/to/model.miv \
     --config /path/to/config.json \
     --prompt 151644,8948,198,2610,525,264 \
     --max-context 8192 \
     --max-new-tokens 128
   ```
2. **`mseries_backendd` (AF_UNIX NDJSON RPC v2 데몬)** — `hello`, `load`, `submit`, `cancel`, `subscribe_events`, `trim_cache`, `set_scheduler_hint`, `shutdown`을 지원하는 상주 토큰 ID 엔진 서버입니다:
   ```bash
   ./build/miv-v001/miv/mseries_backendd --socket /tmp/ninfer-metal.sock --exit-on-disconnect
   ```
3. **`libomlx_m5_core_runtime.a` (C++20 정적 라이브러리)** — [`miv/include/omlx_m5/engine.h`](../../../miv/include/omlx_m5/engine.h)를 통해 `omlx::m5::Engine`을 직접 임베드합니다.
4. **`omlx_m5_native` (선택적 Python 모듈)** — `-DOMLX_M5_BUILD_PYTHON=ON`으로 빌드할 때 생성되는 Nanobind 바인딩입니다.
