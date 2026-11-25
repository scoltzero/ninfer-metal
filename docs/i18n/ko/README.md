<p align="center"><img src="../../assets/logo.png" width="140" alt="ninfer-metal" /></p>

<h1 align="center">ninfer-metal</h1>

<p align="center">
  <strong>Apple M5+ 전용 네이티브 Metal 4 NAX &amp; Compute ICB LLM 추론 엔진 (v1 코어 코드네임 <code>miv</code>)</strong>
</p>

<!-- Replace scoltzero/ninfer-metal with final GitHub slug if changed -->
<p align="center">
  <a href="https://github.com/scoltzero/ninfer-metal/stargazers"><img src="https://img.shields.io/github/stars/scoltzero/ninfer-metal?style=flat-square&logo=github" alt="GitHub Stars" /></a>
  <a href="https://github.com/scoltzero/ninfer-metal/releases"><img src="https://img.shields.io/github/downloads/scoltzero/ninfer-metal/total?style=flat-square" alt="GitHub Downloads" /></a>
  <a href="miv-v0.01-changelog.md"><img src="https://img.shields.io/badge/release-miv--v0.01-0969da?style=flat-square" alt="Latest Release miv-v0.01" /></a>
  <img src="https://img.shields.io/badge/platform-Apple%20M5%2B%20%C2%B7%20macOS%2027%2B-111827?style=flat-square" alt="Platform Apple M5+ macOS 27+" />
  <img src="https://img.shields.io/badge/backend-Metal%204%20NAX-10b981?style=flat-square" alt="Backend Metal 4 NAX" />
  <a href="../../../LICENSE"><img src="https://img.shields.io/badge/License-GPLv3-blue.svg?style=flat-square" alt="License GPL-3.0" /></a>
</p>

<div align="center">

[English](../../../README.md) | [简体中文](../zh-CN/README.md) | [繁體中文](../zh-TW/README.md) | [日本語](../ja/README.md) | **한국어** | [Español](../es/README.md) | [Français](../fr/README.md) | [Deutsch](../de/README.md)

</div>

---

**`ninfer-metal`**은 하드웨어 **Neural Accelerators (NAX)**와 **Compute ICB**를 탑재한 Apple Silicon(`M5 / M5 Pro / M5 Max / M5 Ultra`, `MTLGPUFamilyApple10+`, `macOS 27+`) 전용 순수 C++20 / Objective-C++ / Metal 4 추론 엔진입니다.

- **Python 무의존·MLX 무의존**: 네이티브 `ninfer-metal-serve`와 `mseries_backendd`를 포함한 Mach-O 바이너리 16개와 사전 컴파일된 `.metallib` 셰이더 38개를 제공합니다. macOS 시스템 프레임워크에만 링크되어 Xcode·Python·Homebrew dylib가 없는 클린 Mac에서도 바로 실행됩니다.
- **독립 실행형 OpenAI 서버 (`ninfer-metal-serve`)**: 단일 C++20 바이너리로 인프로세스 `miv`와 병렬 경로 B SPI v2 마이크로커널을 하나의 HTTP/SSE 포트(`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/metrics`)에서 호스팅하며, C++ BPE 토크나이저, Jinja 채팅 템플릿, `<think>` `reasoning_content` 분리, Qwen XML/JSON `<tool_call>` 파서를 내장합니다. Open WebUI, Cursor, Cline, NewAPI, OpenAI SDK와 바로 연결됩니다.
- **단일 상주 NAX 가중치 & `0-ULP` 일치**: 오프라인 NAX 프리타일링(`.omlxm5`)으로 Prefill, `T=1` AR 디코딩, 투기적 검증을 단일 상주 가중치 사본(`1.0×` 아티팩트 크기)에서 처리합니다. 기준선 **`miv-v0.01`**은 Qwen3.5 / Qwen3.6 / Qwen3.8 하이브리드 모델(Dense 27B 및 35B-A3B MoE)을 `MXFP4` 및 `oQ` Affine(`q3–q8`)에서 실행하며, `affine4_g128` KV 캐시, Native MTP / DFlash2, 다중 사용자 배칭(`B/C = 1..8`), 롤링 UMA 프리픽스 재사용을 지원합니다.
- **드롭인 `oMLX-extend` 공존**: AF_UNIX Provider SPI v2 사이드카로 **`miv Core Backend`**를 노출하는 `oMLX-extend`(`jundot/omlx` 기반 `omlx-extend-v0.8.3`)를 번들합니다. 변환된 `.omlxm5` 모델은 O(1)으로 자동 라우팅되며, 미변환 모델은 Stock MLX에서 그대로 실행되고 사이드카와 MLX 모델은 단일 메모리 상한 아래 UMA를 공유합니다.

---

## 성능 요약

![단일 스트림 디코딩 및 0-ULP 투기적 디코딩 가속](../../assets/charts/decode-speedup.svg)

![다중 사용자 동시성 확장 합산 처리량](../../assets/charts/concurrency-scaling.svg)

실제 Apple M5 Ultra(`80-GPU / 256 GiB` 및 `64-GPU / 96 GiB`)의 격리된 클린 서브프로세스에서 기본 `oMLX 0.8.0`(`mlx 0.32.4`, NAX 활성화) 대비 `0-ULP` 그리디 일치 조건으로 측정했습니다. 차트 4종 전체와 Dim A–F 수치는 **[`miv-v0.01` 릴리스 노트](miv-v0.01-changelog.md)**에 수록되어 있습니다.

---

## 지원 모델 및 마이크로커널

| 모델 패밀리 | 양자화 | 마이크로커널 | 모드 | 크레딧 / 출처 |
|---|---|---|---|---|
| **Qwen3.5 / 3.6 / 3.8 Dense** (0.8B – 27B, 3:1 GDN+GQA) | `MXFP4`, `oQ4e` | `miv-v0.01` | 경로 A (메인라인) | `ninfer-metal` 코어 |
| **Qwen3.5 / 3.6 Hybrid MoE** (256×8 + 공유, 예: 35B-A3B) | `MXFP4`, `oQ4e` | `miv-v0.01` | 경로 A (메인라인) | `ninfer-metal` 코어 |
| *Llama, Qwen3 (Pure-Attn), DeepSeek MLA, Gemma, GLM, MiniMax, ...* | *개방* | *PR 모집* | 경로 A / 경로 B | **[Microkernel Foundry 참여 →](microkernel-foundry.md)** |

기여자와 코딩 에이전트는 **[Microkernel Foundry](microkernel-foundry.md)**를 통해 이 목록에 새 모델 패밀리를 추가할 수 있습니다:

- **경로 A (메인라인 PR)**: `ninfer-metal`의 공유 Metal 4 NAX 기반에 직접 병합하며, 이 목록에 **기여자 크레딧**을 표기합니다.
- **경로 B (병렬 사이드카)**: 준비되면 PR이나 Issue를 제출해 주세요. [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json)(`sync_microkernels.sh`)에 등록하고 이 목록에 **원본 저장소 링크**를 게시하며, `oMLX-extend`와 독립 실행형 `ninfer-metal-serve --providers-dir` 모두 `Providers/*/current/provider.json`을 자동 검색해 경로 B 마이크로커널을 `miv`(`oMLX-extend`에서는 기본 MLX 포함)와 병렬로 실행합니다.

---

## 빠른 시작

제로 MLX `miv/` 코어를 빌드합니다(`macOS 27+`, `Xcode 27+`):

```bash
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
cmake -S . -B build/miv-v001 -G Ninja -DCMAKE_BUILD_TYPE=Release -DOMLX_M5_BUILD_NAX_METALLIB=ON -DOMLX_M5_BUILD_MLX_LEGACY_SMOKE=OFF -DOMLX_M5_BUILD_PYTHON=OFF
cmake --build build/miv-v001
```

한 줄 명령으로 독립 실행형 C++ OpenAI 서버를 띄우거나(`oMLX-extend`와 함께 쓰는 [`omlx-extend/config/ninfer-metal.toml`](../../../omlx-extend/config/ninfer-metal.toml)의 공유 기본값을 자동 로드) `oMLX-extend` 사이드카 계층을 검증합니다:

```bash
./omlx-extend/scripts/start_ninfer_serve.sh /path/to/Qwen3.8-27B
./omlx-extend/scripts/apply_spi_patch.sh --omlx-dir ./omlx --test
```

전체 빌드 플래그, `.omlxm5` 가중치 변환, 멀티 마이크로커널 `ninfer-metal-serve --providers-dir` / `[[models]]` 설정, `oMLX-extend` 핫 업데이트 절차는 **[시작 가이드 및 설정](getting-started.md)**을 참고하세요. macOS 앱 화면별 안내는 **[`oMLX-extend` 비주얼 가이드](omlx-extend-guide.md)**에서 확인할 수 있습니다.

---

## 문서 안내

| 가이드 | 내용 |
|---|---|
| **[`oMLX-extend` 비주얼 가이드](omlx-extend-guide.md)** | macOS용 `oMLX` 5단계 스크린샷 가이드: 마이크로커널 카드, 활성 엔진 배지, 자동 라우팅 / Stock MLX 전환, MTP / DFlash2 & 4-bit KVQuant, 원클릭 공유 설정 |
| **[시작 가이드 및 설정](getting-started.md)** | 사전 요구 사항, 제로 MLX CMake 빌드, 원클릭 `start_ninfer_serve.sh`, `ninfer-metal-serve` 독립 서버, `oMLX-extend` 사이드카 구성, `ninfer-metal.toml`, 저수준 CLI/데몬 사용법 |
| **[아키텍처 및 공존 설계](architecture.md)** | 듀얼 모드 `miv/` 코어(`ninfer-metal-serve` + `mseries_backendd`), Provider SPI v2, O(1) 자동 라우팅, UMA 메모리 조정, 원자적 심볼릭 링크 핫 업데이트 |
| **[Microkernel Foundry](microkernel-foundry.md)** | Silicon-Squeeze Skill을 활용한 M5+ 신규 모델 패밀리 마이크로커널(Llama, DeepSeek, Gemma, GLM, MiniMax) 기여 안내 |
| **[릴리스 노트 (`miv-v0.01`)](miv-v0.01-changelog.md)** | Dim A–F 벤치마크 차트, `ninfer-metal-serve` & `oMLX-extend` 변경 사항, 테스트 & `otool -L` 감사 결과, 지원 범위 |
| **[로드맵](roadmap.md)** | `miv-v0.01` 이후 개발 마일스톤 |
| **[문서 색인](../../README.md)** | 8개 언어 문서 미러, 하드웨어 가이드, SKU 캘리브레이션(`64-GPU` / `80-GPU`), 백엔드 사양, 인증 기준 |

---

## 감사의 말

- **[`jundot/omlx`](https://github.com/jundot/omlx)** — `oMLX-extend`가 확장하는 업스트림 oMLX 추론 서버 및 macOS 앱.
- **[`Neroued/ninfer`](https://github.com/Neroued/ninfer)** — 명시적 메모리 소유권과 결정론적 스케줄링 설계의 기준이 되는 네이티브 추론 엔진.
- **[`z-lab/dflash`](https://github.com/z-lab/dflash)** & **[`bstnxbt/dflash-mlx`](https://github.com/bstnxbt/dflash-mlx)** — DFlash / DFlash2 블록 확산 투기적 디코딩 아키텍처와 Apple Silicon 드래프트 모델 생태계.
- **[`SqueezeAILab/KVQuant`](https://github.com/SqueezeAILab/KVQuant)** & **[`FutureMLS-Lab/FOSCAR`](https://github.com/FutureMLS-Lab/FOSCAR)** — 저비트 KV 캐시 양자화 연구 및 `miv`의 `affine4_g128` Metal 4 NAX 커널에 반영된 아핀 폴딩 디코드 어텐션.
- **[`p-e-w/heretic`](https://github.com/p-e-w/heretic)** & **[`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx)** — 업스트림 무학습 언어 모델 수정 프레임워크 Heretic과 Apple Silicon MLX 네이티브 포트 및 커뮤니티 양자화 가중치.
