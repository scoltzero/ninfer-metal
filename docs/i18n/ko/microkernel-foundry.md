[English](../../microkernel-foundry.md) | [简体中文](../zh-CN/microkernel-foundry.md) | [繁體中文](../zh-TW/microkernel-foundry.md) | [日本語](../ja/microkernel-foundry.md) | **한국어** | [Español](../es/microkernel-foundry.md) | [Français](../fr/microkernel-foundry.md) | [Deutsch](../de/microkernel-foundry.md)

# Microkernel Foundry

`ninfer-metal`(`miv-v0.01`)은 Apple M5+ 환경에서 Qwen3.5, Qwen3.6, Qwen3.8 하이브리드 모델을 위한 베어실리콘 Metal 4 NAX 추론을 제공합니다. **Microkernel Foundry**는 기여자와 코딩 에이전트가 메인라인 풀 리퀘스트 또는 병렬 사이드카로 새 모델 패밀리를 추가하도록 동일한 런타임 기반, 서빙 프론트엔드, 캘리브레이션 도구, 인증 파이프라인을 개방합니다.

대상 모델 패밀리는 **Llama, Qwen3 Pure-Attention, DeepSeek (MLA), Gemma, GLM, MiniMax** 및 [`scoltzero/heretic-mlx`](https://github.com/scoltzero/heretic-mlx) 같은 커뮤니티 양자화 파생 모델입니다.

---

## 두 가지 기여 경로

### 경로 A: 메인라인 PR → 기여자 크레딧 (권장)

공유 런타임 **`libninfer_metal_substrate`** 위에 모델 패밀리 마이크로커널을 구현해 `ninfer-metal` 저장소에 직접 병합합니다.

**공유 기반이 제공하는 구성 요소:**
- Metal 4 실행 런타임(`MetalContext`, `MTL4CommandQueue`, 더블 버퍼 `ar_pipe_[2]`, `< 2 GiB` Compute ICB 인코더, `read_write` 배리어 리플렉션).
- 공유 페이지 기반 KV 및 프리픽스 메모리(`KvPagePool`, `SharedKvArenaPool`, CoW 참조 카운팅, `SharedPrefixCache`, `UmaCheckpointPool`).
- 범용 NAX 및 디코딩 프리미티브(`mxfp4_nax`, `affine_nax`, `bf16_nax`, `moe_nax`, 제로 `uitofp` QMV, 2단계 병렬 `Argmax`, `Split-KV` / `Flash-NAX`).
- `.miv`(기존 `.omlxm5` 호환) 컨테이너 로더, 번들 가중치 재배치 변환기(`tools/m5_artifact.py`), `ExecutionIdentityFields`, `24 MHz` 하드웨어 카운터, 네이티브 독립 OpenAI 서버(`ninfer-metal-serve`), Provider SPI v2 데몬(`mseries_backendd`).

**PR에 추가할 내용과 실행 범위:** 대상 아키텍처의 레이어 그래프, 가중치 매핑, `.safetensors → .miv` 가중치 재배치 규칙, 그리고 콜드 DRAM 마이크로프로브에서 범용 프리미티브보다 **`>= +10%–12%`** 이상 빠른 아키텍처 전용 커널(예: MLA, GeGLU, Softcap, 커스텀 MoE 라우터)을 추가합니다. `libninfer_metal_substrate`를 공유해 아키텍처 간 코드와 상주 메모리 중복을 없애며, 독립형 `ninfer-metal-serve`와 `oMLX-extend`(최초 `Load` 시 원본 가중치와 같은 폴더에 `.miv` 자동 생성 지원) 모두에서 네이티브로 실행되고 [지원 모델 목록](README.md)에 기여자 크레딧이 표기됩니다.

### 경로 B: 병렬 사이드카 → PR 또는 Issue → 원본 프로젝트 링크 표기

실험적 아키텍처, 커스텀 양자화 레이아웃, 또는 독립 릴리스 주기가 필요하다면 코어를 단독 Sidecar SPI v2 번들로 아래 경로에 패키징합니다:

```text
~/Library/Application Support/oMLX/Providers/<provider_id>/versions/<vA.B.C>/
```

`supported_topologies`, `supported_model_types`, `artifact_extensions` 및 번들 `converter` 엔트리포인트(`--source <dir> --output <file> --source-repo <id>`)를 선언한 `provider.json`과 함께 원자적 `current` 심볼릭 링크를 노출합니다. 준비되면 PR을 보내거나 Issue를 열어 주세요. 메인테이너가 [`omlx-extend/registry/microkernels.json`](../../../omlx-extend/registry/microkernels.json)에 등록해 사용자가 [`./omlx-extend/scripts/sync_microkernels.sh`](../../../omlx-extend/scripts/sync_microkernels.sh)로 설치·업데이트할 수 있게 하며, [지원 모델 목록](README.md)에 원본 저장소 직접 링크를 게시합니다.

**통합 멀티 마이크로커널 호스팅:** **`oMLX-extend`와 독립 실행형 `ninfer-metal-serve --providers-dir` 모두** `Providers/*/current/provider.json`을 네이티브로 탐색하고 통합 UMA 예산 조정 및 `gpu_contended` 스케줄링 아래 여러 경로 A(**`miv Core Backend`**)와 경로 B 마이크로커널을 단일 포트에서 병렬 실행합니다(`oMLX-extend`에서는 Stock MLX와 함께 실행). `oMLX` macOS 앱에서는 설치된 경로 B 마이크로커널이 **`Integrations → External Engines`**에 카드로 자동 표시되고 **`Models`**에 활성 엔진 배지가 나타납니다([`oMLX-extend` 비주얼 가이드](omlx-extend-guide.md) 참고). 마이크로커널 작성자는 Metal 4 코어와 Sidecar SPI v2 데몬만 구현하면 되며, HTTP/SSE 서빙, 토크나이징, 채팅 템플릿, 멀티 모델 스케줄링은 두 호스트가 처리합니다.

**라이선스:** **경로 A** 메인라인 PR은 `miv`에 링크되어 **[GPL-3.0](../../../LICENSE)**으로 배포됩니다(기여자는 GPL-3.0, Apache-2.0 또는 MIT 코드를 제출할 수 있음). **경로 B** 병렬 사이드카는 별도 프로세스로 AF_UNIX Sidecar SPI v2를 통해 통신하므로 원하는 라이선스를 자유롭게 사용할 수 있습니다.

---

## 모델 패밀리 현황

| 아키텍처 패밀리 | 대상 예시 | 상태 | 권장 경로 |
|---|---|---|---|
| **Qwen Hybrid (3:1 GDN + GQA)** | Qwen3.5 / 3.6 / 3.8 Dense (0.8B – 27B) & 35B-A3B MoE | 출시 완료 (`miv-v0.01`) | 경로 A (메인라인, `miv/`) |
| **Pure-Attention GQA** | Llama-3.3-70B, Qwen3-32B, Gemma | PR 모집 중 | 경로 A |
| **Multi-Head Latent Attention (MLA)** | DeepSeek V3 / R1 패밀리 | PR 모집 중 | 경로 A / 경로 B |
| **MoE 및 커뮤니티 양자화** | GLM, MiniMax, `heretic-mlx` 파생 모델 | PR 모집 중 | 경로 A / 경로 B |

---

## 마이크로커널 작성 및 인증 절차

엔지니어와 코딩 에이전트 모두 재현 가능한 4단계 루프를 따릅니다:

1. **대상 Mac 캘리브레이션 (`Stage B: H0 + H1`)**
   `H0` 하드웨어 DMA 스윕(`MTLBlitPassDescriptor`, `4–512+ MiB`)으로 칩의 $B_{\text{DRAM}}$, $C_{\text{SLC}}$, $P_{\text{NAX}}$, $I_{\text{ridge}}$를 측정한 뒤 `H1` 연산자 스윕을 실행합니다. 모든 통과 기준은 이 실측 상수로 정규화됩니다.
2. **콜드 DRAM 마이크로프로브 우선 검증 (`~50–100`줄)**
   새 커널 패밀리를 템플릿화하기 전에 독립 콜드 DRAM 프로브(`working set >= 3–4 × C_SLC`, 직렬 배리어 중앙값 `>= 30`회)로 고정 기능 NAX 및 하드웨어 기준선과 비교합니다. 순 이득이 **`>= +10%–12%`** 이상인 커널만 채택하고, 실패한 실험용 PSO는 PR 전에 삭제합니다.
3. **`< 10 s` 고속 서브레이어 프로파일링 및 스필 감사**
   `omlx_m5_hw_counter_probe --mode model`(`24 MHz` 카운터)로 GPU 서브레이어 비중(`mlp/moe`, `attn`, `lm_head`, `sampling`)을 측정하고, `metal-objdump -d`로 셰이더를 감사합니다(`0 alloca` 스필, 와이드 Prefill `<= 104` VGPR, `T=1` 프리타일 QMV `0 uitofp`).
4. **5개 파일 기계 판독형 증빙 패키지 제출**
   클린룸 Dim A–F 하네스(`miv/tools/bench_omlx_vs_m5rt.py`)를 실행하고 `certification-evidence/<core-id>-<version>-<sku>/`를 첨부합니다:
   - `h0-capability.json` — 베어실리콘 `H0` 상수 및 10,000스텝 ICB 카나리
   - `h1-calibration-<sku>.json` — SKU 격리 `H1` 튜닝 및 `prior_vs_measured_diff`
   - `dim-a-f-bench.json` — 클린룸 Dim A–F 측정 결과 및 `phys_footprint`
   - `parity-report.json` — `0-ULP` 일치, 소스 PPL/KL, NIAH/RULER 리콜, `0 NaN/Inf`
   - `otool-audit.txt` — 순수 Apple 프레임워크 `otool -L`, `0 alloca` 감사, 100턴 연속 안정성 테스트 및 CTest 출력

---

## 기여자 및 에이전트 필독 문서

- **[Silicon-Squeeze 코어 작성 스킬](../../dim-guides/qwen-mxfp4-oq4e-silicon-squeeze-skill.md)** — 코딩 에이전트에 바로 로드해 쓰는 M5+(`MTLGPUFamilyApple10+`) 베어실리콘 최적화 방법론.
- **[코어 인증 기준](../../standards/core-certification-standard.md)** — Gate 1–4 통과 기준, Provider SPI v2 계약, Roofline 임계값, Tier 1 / Tier 2 인증 배지 사양.
- **[Apple M5 시리즈 최적화 플레이북](../../silicon-guides/m5-series-optimization-playbook.md)** — `32 KiB` 스레드그룹 점유율 계산, 통합 메모리 수용 방정식, Base / Pro / Max / Ultra 확장 지침.
