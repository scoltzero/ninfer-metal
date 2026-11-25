[English](../../releases/miv-v0.01-changelog.md) | [简体中文](../zh-CN/miv-v0.01-changelog.md) | [繁體中文](../zh-TW/miv-v0.01-changelog.md) | [日本語](../ja/miv-v0.01-changelog.md) | **한국어** | [Español](../es/miv-v0.01-changelog.md) | [Français](../fr/miv-v0.01-changelog.md) | [Deutsch](../de/miv-v0.01-changelog.md)

# `ninfer-metal` `miv-v0.01` 릴리스 노트

> **릴리스**: `ninfer-metal` `miv-v0.01` (`codename: miv` · `semver: 0.0.1` · `core_abi: 1` · `protocol_version: 2`)
> **연동 호스트**: `oMLX-extend` `omlx-extend-v0.8.3` (`jundot/omlx` `v0.8.3`용 Provider SPI v2 확장)
> **플랫폼**: Apple M5 / M5 Pro / M5 Max / M5 Ultra (`MTLGPUFamilyApple10+`), macOS 27+ (실행 시 Xcode 불필요, 소스 빌드 시 Xcode 27+ 필요)

`miv-v0.01`은 Python이나 MLX 런타임 의존성 없이 순수 C++20, Objective-C++, Metal 4로 구현해 Apple M5+ Neural Accelerators(NAX)와 Compute ICB 위에서 동작하는 `ninfer-metal`의 첫 공개 릴리스입니다. **Qwen3.5 / Qwen3.6 / Qwen3.8 하이브리드** 모델(Dense 0.8B–27B 및 35B-A3B 같은 256×8 MoE)을 네이티브 OpenAI HTTP/SSE 서버(`ninfer-metal-serve`)로 단독 실행하거나, `oMLX-extend` 내부의 **`miv Core Backend`**로 연동해 Stock MLX와 나란히 OpenAI, Anthropic API 및 macOS GUI를 함께 사용할 수 있습니다. 설치와 CLI 플래그는 [시작 가이드 및 설정](getting-started.md)을, macOS 앱 화면 안내는 [`oMLX-extend` 비주얼 가이드](omlx-extend-guide.md)를 참고하세요.

## 1. Apple M5 Ultra 실기 벤치마크

아래 수치는 모두 물리 Apple M5 Ultra 장비에서 격리된 서브프로세스로 동일한 모델 가중치, 동일한 토큰 시퀀스, `temperature=0`, `phys_footprint` 메모리 계측 조건하에 기본 `oMLX 0.8.0`(`mlx 0.32.4`, NAX 활성화)과 비교한 결과입니다. 80-GPU / 256 GiB SKU(`macOS 27.2.1` Build `26B131`, `Xcode 27.2` Build `27B52`)와 64-GPU / 96 GiB SKU 결과는 구분해 기재합니다.

### 단일 스트림 디코드 및 `0-ULP` 투기적 디코딩 가속

![단일 스트림 디코드 및 0-ULP 투기적 디코딩 가속](../../assets/charts/decode-speedup.svg)

- 80-GPU M5 Ultra의 순수 AR 디코드는 테스트한 모든 가중치 형식에서 기본 oMLX를 앞섭니다: `Qwen3.8-27B MXFP4`에서 `64.63 tok/s`(`+14.5%`), `27B oQ4e`에서 `54.03 tok/s`(`+7.3%`), `Qwen3.6-35B-A3B MXFP4`에서 `216.44 tok/s`(`+23.7%`).
- 투기적 디코딩은 `27B` 코딩 스트림 속도를 `DFlash2`로 `186.74 tok/s`(AR 대비 `2.89×`), `Native MTP`로 `140.48 tok/s`(기본 oMLX MTP `103.28 tok/s` 대비 `+36.0%`)까지 끌어올리며 `128/128` `0-ULP` 그리디 일치를 유지합니다.
- 64-GPU / 96 GiB SKU에서는 `27B oQ4e`가 Coding에서 `138.92 tok/s`(`DFlash2`, `+33.9%`)와 `107.52 tok/s`(`MTP`, `+21.2%`), Reasoning에서 `110.01` / `100.28 tok/s`를 기록했습니다(`6/6` 세트, `256/256` `0-ULP`).

### 다중 사용자 동시성 확장

![다중 사용자 동시성 확장 총 처리량](../../assets/charts/concurrency-scaling.svg)

- 80-GPU SKU에서 측정한 19개 동시성 구간 전체에서 `miv`가 기본 oMLX보다 빠릅니다. `B=4`에서 `27B`는 `500.57 tok/s`(`oQ4e DFlash2`), `496.60 tok/s`(`MXFP4 DFlash2`), `434.92 tok/s`(`oQ4e MTP`)에 도달해 기본 oMLX(`148.37` MTP / `129.80` AR)보다 `2.93×–3.83×` 높은 처리량을 냅니다.
- `Qwen3.6-35B-A3B`는 `B=8` 순수 AR에서 윈도우 `846.29 tok/s`(종단 간 `794.39 tok/s`)까지 확장되며 기본 oMLX 대비 `+52.6%` 처리량과 `4.16×` 빠른 TTFT를 제공합니다.
- 모든 동시 슬롯이 단일 스트림 실행과 `128/128 × 4` `0-ULP` 그리디 일치를 유지합니다.

### 64K 장문 컨텍스트 안정성, UMA 점유율 및 프리필

![64K 장문 컨텍스트 안정성 및 affine4_g128 UMA 점유율](../../assets/charts/long-context-curve.svg)

- `affine4_g128` KV를 적용한 `27B MXFP4`는 컨텍스트가 `2K`에서 `64K`로 늘어나도 AR 속도 하락이 `-4.1%`에 그치며(`61.42 → 58.92 tok/s`, oMLX 대비 `+28.6%`), 실제 영문+코드 워크로드에서 `DFlash2`가 `64K`에서도 `139.70 tok/s`를 유지합니다(`2K–64K` 전 구간 `0-ULP`).
- `affine4_g128`은 `64K` KV 캐시를 `71.8%` 절감(`4.00 → 1.13 GiB`)해 단일 요청 최대 UMA를 `15.69 GiB`로 낮춥니다(oMLX 대비 `-3.44 GiB`, 4개 프리픽스 깊이 동시 캐시 시 `-10.06 GiB`). 웜 `64K + 32` TTFT는 `176.19 ms`(`bf16`)와 `201.42 ms`(`affine4`)로 단축되어 oMLX보다 `7.05×–9.48×` 빠릅니다.
- 프리필은 80-GPU SKU에서 `1K` 기준 `2,058.5–2,110.6 tok/s`(`100.25–102.80 TF`), `64K` 콜드 기준 `41.02–41.84 s`를 달성했고, 64-GPU SKU `4K` 프리필에서는 `15/15` 전 항목 우세와 함께 `35B-A3B` `10,048.5 tok/s`(`+15.9%`)를 기록했습니다.

### 멀티턴 대화 및 롤링 프리픽스 재사용

![멀티턴 짧은 대화 롤링 프리픽스 재사용 및 UMA 평탄도](../../assets/charts/multiturn-rolling.svg)

- 롤링 UMA 프리픽스 재사용으로 멀티턴 적중률이 `92%`를 넘어서며(`B=1` `92.33%`, `B=4` `92.02%`), 재프리필 토큰 수를 절반으로 줄이고 웜 TTFT를 `2.44×–2.94×` 단축합니다(`B=1` `101.25 ms`, `B=4` `282.80 ms`).
- 인플레이스 슬랩(slab) 재사용 덕분에 `B=1` 10턴 대화의 UMA 증가폭이 축출 없이 `+0.02 MB`에 머물며(oMLX는 `affine4` `+397.5 MB`, `bf16` `+704.8 MB` 증가), 10턴 전체 소요 시간이 `16.99 s`에서 `9.26 s`로 줄어듭니다(`DFlash2`).
- `B=4`(사용자 4명 × 6턴)에서 배치 `affine4` NAX 디코드는 `176.95 tok/s`(`AR`)와 `206.35 tok/s`(`DFlash2`, oMLX 대비 `1.72×`)에 도달하고 최대 UMA를 `2.12 GiB` 절감합니다. `B=1`(`10/10`)과 `B=4`(`24/24`) 전 턴에서 `0-ULP` 검증을 마쳤습니다.

## 2. `miv-v0.01` 주요 구성

### 독립 실행형 OpenAI 서버 (`ninfer-metal-serve`)
- `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, Prometheus `/metrics`를 제공하는 단일 C++20 HTTP/SSE 바이너리로 동기 JSON, SSE 스트리밍(`usage`, `[DONE]`), 선택적 Bearer 인증(`--api-key`), CORS, 정상 종료를 지원합니다.
- 내장 C++ BPE 토크나이저(`tokenizer.json`), Jinja 채팅 템플릿 렌더링, `<think>` `reasoning_content` 분리, 동기 및 스트리밍 Qwen XML/JSON `<tool_call>` 파싱을 기본 포함합니다.
- 인프로세스 `miv`와 외부 경로 B Provider SPI v2 사이드카를 단일 포트에서 함께 호스팅하며(`--providers-dir`, `[[models]]`), 공유 UMA 메모리 상한(기본 물리 RAM의 `81%`)과 경합 인식 스케줄링을 적용합니다.
- 바로 실행할 수 있는 [`ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)(`oMLX-extend`와 자동 탐색·공유)과 원커맨드 실행 스크립트 `./omlx-extend/scripts/start_ninfer_serve.sh`를 함께 제공합니다.

### Metal 4 NAX 추론 엔진 (`miv/`)
- MLX나 Python 헤더 없이 빌드되는 자립형 C++20 / Objective-C++ / Metal 4 코드트리로, NAX 프리필, 퓨즈드 디코드, 256×8 MoE 라우팅, 투기 검증, 4-bit Split-KV를 담당하는 사전 컴파일 `.metallib` 셰이더 38종을 포함합니다.
- Apple M5 Ultra 두 SKU별 전용 하드웨어 보정 프로필([80-GPU / 256 GiB](../../backend/silicon-calibrations/apple-m5-ultra-80gpu.json) 및 [64-GPU / 96 GiB](../../backend/silicon-calibrations/apple-m5-ultra-64gpu.json))을 탑재해 split-K, Split-KV 분할, 드래프트 깊이를 자동 조정하며 환경 변수가 우선 적용됩니다.
- 제로카피 GPU 프리픽스 이어쓰기(`64K` 복원 `0.73–1.62 ms`), 인플레이스 롤링 UMA 프리픽스 재사용, 동시 세션을 위한 배치 `affine4_g128` NAX 어텐션을 제공합니다.

### `oMLX-extend` `omlx-extend-v0.8.3` (macOS 앱 및 사이드카)
- `AF_UNIX` 사이드카(`mseries_backendd`)를 통해 **`miv Core Backend`**를 등록해 변환된 `.miv` 모델은 O(1)로 자동 라우팅하고, 일반 MLX 모델은 추가 프로세스 없이 Stock MLX에서 그대로 실행합니다.
- 네이티브 macOS UI의 `Integrations → External Engines`에 `miv Core Backend` 및 경로 B 마이크로커널 카드, 모델별 엔진 배지, Auto / Stock MLX 선택기, 백엔드 기능에 따라 활성화되는 `64K+` 컨텍스트, `MTP`, `DFlash2`, `affine4_g128` KV 캐시 제어를 추가했습니다.
- 사이드카와 MLX 모델 간 UMA 예산, 유휴 캐시 회수, 경합 인식 청크 프리필을 조율하며 `~/Library/Application Support/oMLX/Providers/io.ninfer.metal/current/`의 원자적 심볼릭 링크 업데이트를 지원합니다.

## 3. 클린 Mac 패키징 및 검증

- 릴리스 아카이브 `ninfer-metal-miv-v0.01-macos27-arm64.tar.gz`에는 Mach-O 바이너리 16개(`ninfer-metal-serve`, `mseries_backendd`, CLI 및 보정 도구 포함), Metal 4 `.metallib` 셰이더 38개, SKU 보정 프로필 2종, `provider.json`, SHA-256 체크섬이 담겨 있습니다.
- 16개 바이너리 전체에 대한 `otool -L` 검사에서 Apple 시스템 프레임워크(`CoreFoundation`, `Foundation`, `IOKit`, `Metal`)와 시스템 dylib(`libc++.1`, `libSystem.B`, `libobjc.A`)만 링크함을 확인했으며 Homebrew, Xcode, MLX, OpenSSL, Python 런타임 의존성이 전혀 없습니다.
- 종단 간 서버 테스트(동기/SSE, `<think>`/`<tool_call>`, `B=4` 배치, 인프로세스 + 사이드카 병렬 호스팅), `638 / 638` `oMLX-extend` pytest, `75 / 75` macOS XCTest를 모두 통과했으며 Provider SPI 패치도 `jundot/omlx` `v0.8.3`에 충돌 없이 적용됩니다.

## 4. 지원 범위

| 영역 | `miv-v0.01` 지원 범위 |
|---|---|
| 하드웨어 | Apple Silicon `MTLGPUFamilyApple10+` (Apple M5 / M5 Pro / M5 Max / M5 Ultra, 64-GPU & 80-GPU) |
| OS 및 툴체인 | macOS 27+ (실행 시 Xcode 불필요, 소스 빌드 시 Xcode 27+ / macOS 27 SDK) |
| 모델 | **Qwen3.5 / Qwen3.6 / Qwen3.8 하이브리드** (3:1 GDN + GQA; Dense 0.8B–27B 및 35B-A3B 등 256×8 MoE) |
| 가중치 형식 | 오프라인 NAX 사전 타일링 `.miv`: `MXFP4` (`E2M1 × UE8M0`) 및 `oQ` Affine (`q3–q8`, `g64 / g128`) |
| KV 캐시 | `bf16` 및 `affine4_g128` (`0.5625 B/val`, `2K–64K` 전 구간 `0-ULP` 검증) |
| 투기적 디코딩 | 순수 AR (`none`), 내장 `Native MTP`, `DFlash2` (단일 스트림 및 `B=2..4` 배치) |
| 서빙 인터페이스 | `ninfer-metal-serve`의 OpenAI HTTP/SSE 엔드포인트, `oMLX-extend`의 OpenAI + Anthropic 엔드포인트 및 macOS GUI |
