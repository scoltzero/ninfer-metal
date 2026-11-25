[English](../../roadmap/README.md) | [简体中文](../zh-CN/roadmap.md) | [繁體中文](../zh-TW/roadmap.md) | [日本語](../ja/roadmap.md) | **한국어** | [Español](../es/roadmap.md) | [Français](../fr/roadmap.md) | [Deutsch](../de/roadmap.md)

# `ninfer-metal` 로드맵

출시 기준선: **`ninfer-metal` `miv-v0.01`** (네이티브 C++20 `ninfer-metal-serve` 포함) + **`oMLX-extend` `omlx-extend-v0.8.3`** ([릴리스 노트](miv-v0.01-changelog.md) 참고).

`miv-v0.01` 이후 개발 계획은 세 트랙으로 진행됩니다:

---

## 1. `v0.02+` 에이전트 메모리 가상화 (`KVMem-UMA + ActKV`)

전체 아키텍처 사양, Apple M5 UMA 메모리 물리 모델, 검증 게이트 V1–V6는 **[`v0.02-kvmem-actkv-plan.md`](../../roadmap/v0.02-kvmem-actkv-plan.md)**에 정의되어 있습니다.

- **Tier-1: `KVMem-UMA + ActKV-Lite` (UMA 내 `256K → 1M+` 워크스페이스)**
  - 옵트인 방식 (`kv_policy = "kvmem_actkv"`, `"lossless"`를 영구 `0-ULP` 기본값으로 유지).
  - `MTLStorageModeShared` 기반 비대칭 콜드 아카이브(FP8/BF16 Pre-RoPE K + `affine4_g128` V — 4비트 이중 양자화 없음, `1M` 토큰 `Qwen3.8-27B` 워크스페이스 기준 총 `25.03 GiB`).
  - 독립 `MeanKIndexBuffer`(`< 0.6 ms` 스캔) + Action-LRFU + 신뢰도 기반 워킹 셋 `W ∈ [16K, 64K]` + 단일 커널 `re-RoPE + affine4_pack` 구체화(`64K` 기준 `~1.9–4.4 ms`). 핫패스 `attention_affine4.metal`과 정적 ICB는 변경하지 않습니다.
- **Tier-2: Apple NVMe 영속화 & 전체 토큰 수준 ActKV**
  - 32–64 GiB M5 Pro/Max 머신과 `< 1 s` 재시작 복원을 위한 `F_NOCACHE` + `dispatch_io` 기반 `.omlxkv` 워크스페이스 번들.
  - Pure-Attention 32B/70B 및 MoE 모델을 겨냥한 Base-2 LSE 어텐션 복원(`+96 B/tok`)과 제로 역양자화 인플레이스 `affine4_g128` 페이지 압축(`hd=256` 기준 `144 B/tok`, `hd=128` 기준 `80 B/tok`).

---

## 2. 공유 기반 (`libninfer_metal_substrate`) & 신규 모델 패밀리 마이크로커널

- `miv/`에서 Metal 4 런타임, `< 2 GiB` Compute ICB 인코더, `KvPagePool`, `SharedPrefixCache`, `UmaCheckpointPool`, 범용 NAX 프리미티브를 `libninfer_metal_substrate`로 추출합니다.
- **[Microkernel Foundry](microkernel-foundry.md)**와 **[코어 인증 기준](../../standards/core-certification-standard.md)**에 따라 Pure-Attention GQA(`Qwen3-32B`, `Llama-3.3-70B`), DeepSeek MLA, Gemma, GLM, MiniMax 마이크로커널을 추가합니다.

---

## 3. `ninfer-metal-serve` Phase 2 (Anthropic API & 문법 제약 디코딩)

`miv-v0.01`에 탑재된 네이티브 C++20 OpenAI HTTP/SSE 서버 위에 다음 기능을 확장합니다:

- **Anthropic Messages API (`POST /v1/messages`)**: Claude 호환 클라이언트와 코딩 에이전트를 위한 네이티브 Anthropic 요청/응답 변환 및 SSE 이벤트 스트리밍.
- **문법 제약 디코딩 (`xgrammar`)**: `omlx::m5::Engine`에 스텝별 로짓 마스크 콜백을 추가하고 `xgrammar`를 통합해 외부 런타임 의존성 없이 엄격한 JSON Schema(`response_format`) 및 구조화 도구 호출 문법을 강제합니다.
