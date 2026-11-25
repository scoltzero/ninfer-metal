#!/usr/bin/env python3
"""Multi-dimensional head-to-head benchmark: Stock oMLX (native MLX Metal) vs m5rt v0.8.0-m5rt.11.

Executes reproducible, GPU-synchronized comparisons on identical local checkpoints,
identical raw prompt token ID sequences, and identical decode lengths across 5 dimensions:
  - Dimension A: Multi-length prefill scaling (128, 512, 1024, 4096, 8192 tokens)
  - Dimension B: Single-stream B=1 pure AR decode (256 tokens, Coding prompt)
  - Dimension C: Single-stream B=1 speculative decode (DFlash2 & Native MTP; Coding & Reasoning, 256 tokens)
  - Dimension D: Multi-concurrency batched decode scaling (B=1, 2, 4, 8; 128 tokens/stream)
  - Dimension E: Long-context KV scaling (2K, 8K, 32K, 64K), cold load time, phys_footprint,
                 and warm prefix-cache hit (B=1 & B=4)

Every engine configuration runs inside a fresh isolated subprocess so environment flags
(e.g., OMLX_M5_ALLOW_W4A4_PREFILL, OMLX_M5_T_DOMAIN_DISPATCH, OMLX_M5_DISABLE_MTP),
Metal heaps, monkey-patches, and macOS kernel phys_footprint ledgers never cross-contaminate.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import hashlib
import json
import os
import platform
import random
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

_SCRIPT_DIR = Path(__file__).resolve().parent
_SCRIPT_PARENT = _SCRIPT_DIR.parent
ROOT = _SCRIPT_PARENT.parent if _SCRIPT_PARENT.name == "miv" else _SCRIPT_PARENT

# Exact deterministic real prompts from Stage A H2 evaluation (Qwen3.5/3.8 248,320 vocab).
CODING_PROMPT_TOKENS: list[int] = [
    248045, 846, 198, 7734, 264, 4101, 11, 1575, 1676, 35468, 12654, 7873,
    314, 3841, 6646, 440, 303, 40284, 436, 315, 1481, 16168, 286, 321,
    4821, 6813, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
]

REASONING_PROMPT_TOKENS: list[int] = [
    248045, 846, 198, 814, 20139, 303, 7472, 1204, 7905, 35919, 41275, 4779,
    17120, 42103, 494, 41755, 21966, 18588, 1354, 364, 3349, 3992, 1558, 42903,
    13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
]

MODELS: dict[str, dict[str, str]] = {
    "27b-mxfp4": {
        "label": "Qwen3.8-27B-MXFP4",
        "model_dir": "models/mlx-community/Qwen3.8-27B-mxfp4",
        "artifact": ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4-pretiled.omlxm5",
        "draft_dir": "models/z-lab/Qwen3.8-27B-DFlash2",
        "draft_artifact": ".cache/b5/real/qwen38-27b-dflash2.omlxm5",
    },
    "27b-oq4e": {
        "label": "Qwen3.8-27B-oQ4e-mtp",
        "model_dir": "models/Jundot/Qwen3.8-27B-oQ4e-mtp",
        "artifact": ".cache/b0/artifacts/qwen38-27b-oq4e-pretiled.omlxm5",
        "mtp_artifact": ".cache/b5/real/qwen38-27b-oq4e-mtp-pretiled.omlxm5",
        "draft_dir": "models/z-lab/Qwen3.8-27B-DFlash2",
        "draft_artifact": ".cache/b5/real/qwen38-27b-dflash2.omlxm5",
    },
    "35b-a3b-mxfp4": {
        "label": "Qwen3.6-35B-A3B-MXFP4",
        "model_dir": "models/OsaurusAI/Qwen3.6-35B-A3B-mxfp4",
        "artifact": ".cache/b0/artifacts/qwen36-35b-a3b-mxfp4.omlxm5",
        "draft_dir": "",
        "draft_artifact": "",
    },
}


# Exact Qwen3.8 (248,320 vocab) ChatML token framing (enable_thinking=False short-chat).
QWEN38_IM_START = 248045
QWEN38_IM_END = 248046
QWEN38_THINK_OPEN = 248068
QWEN38_THINK_CLOSE = 248069
QWEN38_ROLE_SYSTEM = 8678
QWEN38_ROLE_USER = 846
QWEN38_ROLE_ASSISTANT = 74455
QWEN38_NL = 198
QWEN38_NL2 = 271

QWEN38_SYSTEM_OPEN: list[int] = [QWEN38_IM_START, QWEN38_ROLE_SYSTEM, QWEN38_NL]
QWEN38_TURN_CLOSE: list[int] = [QWEN38_IM_END, QWEN38_NL]
QWEN38_USER_OPEN: list[int] = [QWEN38_IM_START, QWEN38_ROLE_USER, QWEN38_NL]
QWEN38_ASSISTANT_HEADER: list[int] = [
    QWEN38_IM_START,
    QWEN38_ROLE_ASSISTANT,
    QWEN38_NL,
    QWEN38_THINK_OPEN,
    QWEN38_NL2,
    QWEN38_THINK_CLOSE,
    QWEN38_NL2,
]
QWEN38_NEXT_USER_BRIDGE: list[int] = [
    QWEN38_IM_END,
    QWEN38_NL,
    QWEN38_IM_START,
    QWEN38_ROLE_USER,
    QWEN38_NL,
]
QWEN38_USER_TO_ASSISTANT: list[int] = [
    QWEN38_IM_END,
    QWEN38_NL,
    *QWEN38_ASSISTANT_HEADER,
]

MULTITURN_B1_SYSTEM_LEN = 256
MULTITURN_B1_USER_LENS: list[int] = [48, 64, 52, 76, 48, 68, 56, 80, 60, 72]
MULTITURN_B1_GEN_LENS: list[int] = [64, 80, 72, 96, 64, 88, 72, 96, 80, 88]

MULTITURN_B4_SYSTEM_LEN = 512
MULTITURN_B4_USER_LENS: list[int] = [48, 56, 64, 52, 72, 60]
MULTITURN_B4_GEN_LEN = 64

_MULTITURN_SYSTEM_BODY_TEXT = (
    "You are an expert systems engineer and bilingual technical assistant helping developers "
    "design, debug, and optimize low-latency Apple Silicon inference engines, Python async web "
    "services, Metal compute shaders, and distributed caching layers. "
    "Follow these rules strictly on every turn: "
    "1. Give direct, concrete, technically accurate answers without filler or hedging. "
    "2. When code is requested, write clean, idiomatic Python 3.12, C++20, TypeScript, or Metal "
    "Shading Language snippets with explicit edge-case handling and exact complexity bounds. "
    "3. When the user writes in Chinese (中文), respond concisely in clear technical Chinese while "
    "keeping API identifiers, compiler flags, and kernel names in English. "
    "4. Track every constraint established in earlier turns of the conversation: preserve exact "
    "struct field names, buffer alignments, memory budgets, latency targets, and API schemas "
    "unless the user explicitly revises them. "
    "5. When analyzing performance bottlenecks, separate prefill compute throughput, KV/GDN state "
    "bandwidth, host-device synchronization overhead, and allocator fragmentation into explicit "
    "measurable quantities. "
    "6. Prefer minimal diffs, deterministic tests, and zero-copy unified memory layouts over "
    "speculative abstractions. "
    "7. For concurrent workloads, explicitly verify lock ordering, reference counting, page-table "
    "copy-on-write safety, cancellation cleanup, and bounded memory footprints under sustained "
    "multi-turn traffic. "
    "8. Summarize multi-step checklists in compact numbered lists with concrete pass/fail criteria "
    "and rollback triggers suitable for automated CI gates and production deployment runbooks. "
    "9. For storage and serialization paths, state byte alignment, endianness, checksum coverage, "
    "and partial-write recovery semantics explicitly so offline tooling and live servers stay "
    "wire-compatible across point releases. "
    "10. When comparing benchmark runs, require identical prompts, greedy temperature zero, matched "
    "memory budgets, and per-turn cache-hit telemetry so every latency claim is independently "
    "auditable from raw artifacts."
)

_MULTITURN_B1_TURN_TEMPLATES: list[str] = [
    (
        "We are designing a rolling prefix cache for a hybrid linear-attention and full-attention "
        "transformer serving short multi-turn chat sessions. Explain how to keep the per-session "
        "recurrent state footprint flat at 1x instead of growing linearly across ten turns, and "
        "outline the exact eviction invariant for pinned system prompts."
    ),
    (
        "请用简洁的中文解释一下，为什么混合 GDN（linear attention）层不能像标准 KV cache 那样直接按 token 切片回退，"
        "以及在每次 decode 结束后做一个轻量级 GPU UMA checkpoint 相比把状态序列化到 CPU/SSD 有哪些延迟和内存优势。"
    ),
    (
        "Write a concise Python dataclass `TurnMetrics` and a helper function `compute_p95(values)` "
        "using linear interpolation with no third-party dependencies. Make sure empty lists return "
        "None, single-element lists return that element, and values are rounded to three decimals."
    ),
    (
        "Suppose a Metal compute pipeline runs single-request autoregressive decode with one "
        "asynchronous in-flight command buffer ahead of token publishing. If a stop sequence matches "
        "the newly sampled token while the next step is already queued on the GPU, walk through "
        "why draining that command buffer advances the KV/GDN sequence position by one extra token "
        "and how to keep the committed token history exact."
    ),
    (
        "接着上面的异步管线问题，如果我们使用 speculative decoding（例如 DFlash2 每次验证 1+k 个 token），"
        "当 `commit_prefix(count)` 提交前 `count` 个输入 token 时，请给出此时 KV cache 位置与 `committed_tokens` "
        "数组长度之间必须满足的不变量。"
    ),
    (
        "Show a compact C++20 helper method `supersede_unpinned_entry` for a prefix cache table "
        "that checks `entry.id == supersede_id`, verifies `!entry.pinned`, checks strict prefix "
        "equality with `std::equal`, and releases the old UMA slab back into the free pool before "
        "allocating the new checkpoint buffer."
    ),
    (
        "In our DFlash2 mode, the auxiliary feature history stored inside each UMA checkpoint slab "
        "grows by 50 KiB per token as the conversation lengthens from 300 to 1,600 tokens. Explain "
        "how rounding slab allocations to 32 MiB quanta and evicting one undersized free slab on "
        "growth prevents 10x orphan buffer accumulation in the checkpoint pool."
    ),
    (
        "现在我们需要把同一个 256-token 的系统提示词固定在前缀缓存里，同时串行跑多个独立的 10 轮用户会话做压测。"
        "请说明为什么不同会话的第 1 轮必须使用互不相同的用户首句，否则基准测试会把历史会话当成缓存命中而失真。"
    ),
    (
        "Give a short checklist of four automated CTest assertions to verify our 10-turn rolling "
        "prefix implementation: exact partial-hit count on turns 2 through 10, bounded unpinned "
        "entry count, zero orphan slabs in the UMA pool, and bit-exact greedy parity against a "
        "cold full-history reference run."
    ),
    (
        "Finally, summarize the exact per-turn and aggregate metrics we should log in our JSON "
        "benchmark report so reviewers can verify Turn 1 vs Turns 2-10 TTFT, total re-prefilled "
        "tokens, overall cache hit percentage, decode tok/s, total 10-turn wall time, and Turn 1 "
        "to Turn 10 physical memory footprint delta in MB."
    ),
]

_MULTITURN_B4_PERSONA_TURNS: list[list[str]] = [
    # Slot 0: Python async & FastAPI streaming engineer
    [
        "We have four concurrent SSE clients hitting our async inference server. Sketch how to propagate client disconnects into immediate backend request cancellation without leaking active request counters.",
        "Follow up: how should we guard against a race where the sidecar push socket delivers the first token event before the submit RPC response returns the backend request ID?",
        "Add a bounded LRU retired-request-ID set of size 4096 so late events arriving after cancellation are dropped in O(1) time without growing memory over days of uptime.",
        "Now design a compact Python helper that tracks per-request TTFT, first-batch token count, and steady-state decode tok/s cleanly even when speculative decoding emits 4 tokens per event.",
        "What timeout and backpressure policy should we apply to the control UDS lock so periodic stats polling never stalls request submission or cancellation under B=4 bursts?",
        "Summarize the three unit tests we should add to verify disconnect cancellation, early push delivery, and clean shutdown on EOF.",
    ],
    # Slot 1: Chinese distributed systems & caching architect
    [
        "我们在多用户并发对话场景下共享一个 512 token 的系统提示词，请分析 4 个并发会话同时从该系统提示词分叉时，KV page table 的 Copy-on-Write 引用计数应该如何维护。",
        "继续上一个问题：当第 2 轮到第 6 轮每个会话各自推进自己的上下文并替换上一轮的 unpinned rolling checkpoint 时，如何确保共享的 pinned 系统提示词绝不会被误淘汰？",
        "请比较在 4 GiB 前缀缓存预算下，4 个活跃会话各自保留 1 份 rolling frontier（总计 5 条缓存项）与不替换历史版本（24 条缓存项）在统一内存占用上的差异。",
        "当某个会话在第 4 轮和第 5 轮之间跨越 1024 token 边界时，如果 prefill 调度器错误地把第 5 轮拆成两段串行 command buffer，会对 B=4 packed prefill 吞吐造成什么影响？",
        "针对这种多轮短对话负载，给出监控面板最关键的 5 个缓存可观测指标，并说明正常稳态下 `prefix_evictions` 与 `prefix_shared_hits` 的预期范围。",
        "最后用 4 条要点总结上线 rolling prefix cache 时的灰度开关与回滚策略（含环境变量与 JSON 配置项优先级）。",
    ],
    # Slot 2: Metal 4 / Apple Silicon kernel & UMA engineer
    [
        "Explain how a single-command-buffer fused packed prefill plus GPU argmax kernel eliminates host readback stalls for B=4 concurrent chat requests sharing a cached system prefix.",
        "In our Qwen3.8 3:1 hybrid architecture, 48 layers are GDN linear attention and 16 layers are full attention. Break down why the recurrent UMA checkpoint slab is 146.8 MiB and stays constant in AR mode.",
        "Write a concise Metal Shading Language kernel signature and threadgroup layout for copying contiguous BF16 recurrent state buffers into a pooled UMA checkpoint slab with zero CPU synchronization.",
        "When kv_dtype is set to affine4_g128 instead of bf16, how does KV page memory scale across 4 concurrent 1,280-token sessions compared to the fixed GDN state budget?",
        "If all 4 slots finish decode in a synchronized wave and each runs a 0.25 ms post-decode GPU blit after emitting its final token, explain why both TTFT and TPOT stay completely unaffected.",
        "List the exact Instruments / Metal debug layer checks to confirm zero buffer hazards and zero per-turn MetalBuffer reallocations across 6 waves.",
    ],
    # Slot 3: SQL / TypeScript / observability developer
    [
        "Design a minimal JSON schema for recording B=4 6-wave multi-turn benchmark results so both wave-level medians and per-stream TTFT arrays can be audited without storing megabytes of raw text.",
        "Write a TypeScript function `summarizeWaves(waves)` that computes Wave 1 median TTFT, Waves 2-6 median TTFT, all-turns p95 TTFT, and overall cache hit percentage.",
        "How should we compare Stock oMLX 0.8.0 (block_size=2048 prefill-tail only) against m5rt.16 (post-decode UMA rolling prefix) so both engines receive identical tokenized prompts and matched 4 GiB cache budgets?",
        "Explain why appending each engine's own actual generated token IDs to construct Turn k+1 is essential for measuring true post-decode prefix hit rates under greedy decoding.",
        "Write a compact SQL query over a `multiturn_runs` table to pivot Stock oMLX vs m5rt.15 vs m5rt.16 on median Turn 2-10 TTFT, total re-prefilled tokens, and peak footprint GB.",
        "Provide a short release-note table template highlighting the exact speedup factors and re-prefill token reductions for both B=1 10-turn and B=4 6-wave short chat.",
    ],
]

_MULTITURN_CONTINUATION_FILLER: list[str] = [
    " Include concrete numbers, exact edge cases, and clear invariants so an engineer can verify every detail directly in code and tests.",
    " Keep the explanation focused on measurable latency, unified memory residency, cache hit ratio, and deterministic greedy reproducibility.",
    " 请补充具体的边界条件、内存预算估算以及可验证的验收标准，便于自动化测试与回归对比。",
    " Specify exact struct fields, error handling paths, and zero-overhead fast paths for steady-state multi-turn serving.",
]

_QWEN38_TOKENIZER_CACHE: dict[str, tuple[Any, str]] = {}


def _load_qwen38_tokenizer(model_dir: Path | None = None) -> tuple[Any | None, str]:
    """Load Qwen3.8 HuggingFace `tokenizer.json` if present locally; deterministic fallback otherwise."""
    cache_key = str(model_dir) if model_dir is not None else ""
    if cache_key in _QWEN38_TOKENIZER_CACHE:
        return _QWEN38_TOKENIZER_CACHE[cache_key]

    try:
        from tokenizers import Tokenizer  # type: ignore[import-not-found]
    except Exception:
        _QWEN38_TOKENIZER_CACHE[cache_key] = (None, "synthetic_fallback")
        return None, "synthetic_fallback"

    candidates: list[Path] = []
    if model_dir is not None:
        candidates.append(model_dir / "tokenizer.json")
    for mk in ("27b-oq4e", "27b-mxfp4"):
        info = MODELS.get(mk, {})
        if info.get("model_dir"):
            candidates.append(ROOT / info["model_dir"] / "tokenizer.json")
        if info.get("draft_dir"):
            candidates.append(ROOT / info["draft_dir"] / "tokenizer.json")

    for cand in candidates:
        if cand.is_file():
            try:
                tok = Tokenizer.from_file(str(cand))
                res = (tok, str(cand))
                _QWEN38_TOKENIZER_CACHE[cache_key] = res
                return res
            except Exception:
                continue

    _QWEN38_TOKENIZER_CACHE[cache_key] = (None, "synthetic_fallback")
    return None, "synthetic_fallback"


def _encode_exact_length(
    tok: Any | None, text: str, target_len: int, *, seed: int
) -> list[int]:
    """Encode natural-language text to an exact non-special Qwen3.8 token length."""
    if target_len <= 0:
        return []
    if tok is None:
        return build_synthetic_tokens(target_len, seed=seed)

    def _clean_encode(s: str) -> list[int]:
        raw = tok.encode(s, add_special_tokens=False).ids
        return [int(t) for t in raw if 0 <= int(t) < 248000]

    ids = _clean_encode(text)
    step = 0
    while len(ids) < target_len:
        filler = _MULTITURN_CONTINUATION_FILLER[
            (seed + step) % len(_MULTITURN_CONTINUATION_FILLER)
        ]
        more = _clean_encode(f" (Detail #{(seed + step) % 997}):{filler}")
        if not more:
            ids.extend(build_synthetic_tokens(target_len - len(ids), seed=seed + step))
            break
        ids.extend(more)
        step += 1
    return ids[:target_len]


def build_multiturn_system_prompt(
    target_len: int, *, tok: Any | None, seed: int
) -> list[int]:
    """Build exact-length ChatML system prompt: `<|im_start|>system\\n...<|im_end|>\\n`."""
    body_len = target_len - len(QWEN38_SYSTEM_OPEN) - len(QWEN38_TURN_CLOSE)
    if body_len <= 0:
        raise ValueError(f"system prompt target_len={target_len} too small for ChatML framing")
    body = _encode_exact_length(tok, _MULTITURN_SYSTEM_BODY_TEXT, body_len, seed=seed)
    prompt = [*QWEN38_SYSTEM_OPEN, *body, *QWEN38_TURN_CLOSE]
    assert len(prompt) == target_len
    return prompt


def build_multiturn_b1_session_queries(
    session_idx: int, *, tok: Any | None, user_lens: list[int] | None = None
) -> list[list[int]]:
    """Build 10 distinct realistic user queries for one independent B=1 session."""
    lens = list(user_lens or MULTITURN_B1_USER_LENS)
    modules = (
        "prefix-cache-core",
        "uma-slab-pool",
        "gdn-recurrent-state",
        "dflash2-verifier",
        "packed-prefill-sched",
        "kv-page-allocator",
        "sidecar-uds-bridge",
        "metal4-icb-runner",
    )
    module_name = modules[session_idx % len(modules)]
    queries: list[list[int]] = []
    for turn_idx, target_len in enumerate(lens):
        base = _MULTITURN_B1_TURN_TEMPLATES[turn_idx % len(_MULTITURN_B1_TURN_TEMPLATES)]
        if turn_idx == 0:
            # Distinct opener ensures every session diverges from prior sessions right after System(256).
            text = (
                f"[Session {session_idx + 1} | Ticket MT-{1000 + session_idx} | Component {module_name}] "
                f"{base}"
            )
        else:
            text = f"[Turn {turn_idx + 1} for {module_name} / MT-{1000 + session_idx}] {base}"
        queries.append(
            _encode_exact_length(
                tok, text, target_len, seed=41000 + session_idx * 97 + turn_idx * 13
            )
        )
    return queries


def build_multiturn_b4_wave_queries(
    repeat_idx: int,
    *,
    tok: Any | None,
    batch_size: int = 4,
    user_lens: list[int] | None = None,
) -> list[list[list[int]]]:
    """Return `[wave_idx][slot_idx]` token lists for a B=4 x 6-wave concurrent multi-turn repeat."""
    lens = list(user_lens or MULTITURN_B4_USER_LENS)
    waves: list[list[list[int]]] = []
    for wave_idx, target_len in enumerate(lens):
        wave_queries: list[list[int]] = []
        for slot_idx in range(batch_size):
            persona = _MULTITURN_B4_PERSONA_TURNS[
                slot_idx % len(_MULTITURN_B4_PERSONA_TURNS)
            ]
            base = persona[wave_idx % len(persona)]
            if wave_idx == 0:
                text = (
                    f"[Run {repeat_idx + 1} | User {slot_idx + 1} | PR #{2000 + repeat_idx * 17 + slot_idx}] "
                    f"{base}"
                )
            else:
                text = f"[Wave {wave_idx + 1} | User {slot_idx + 1} | PR #{2000 + repeat_idx * 17 + slot_idx}] {base}"
            wave_queries.append(
                _encode_exact_length(
                    tok,
                    text,
                    target_len,
                    seed=61000 + repeat_idx * 313 + wave_idx * 31 + slot_idx * 7,
                )
            )
        waves.append(wave_queries)
    return waves


def build_multiturn_turn1_prompt(
    system_prompt: list[int], user_tokens: list[int]
) -> list[int]:
    return [*system_prompt, *QWEN38_USER_OPEN, *user_tokens, *QWEN38_USER_TO_ASSISTANT]


def extend_multiturn_prompt(
    prev_prompt: list[int],
    actual_generated_tokens: list[int],
    next_user_tokens: list[int],
) -> list[int]:
    return [
        *prev_prompt,
        *actual_generated_tokens,
        *QWEN38_NEXT_USER_BRIDGE,
        *next_user_tokens,
        *QWEN38_USER_TO_ASSISTANT,
    ]


def compute_percentile(values: list[float], pct: float) -> float | None:
    clean = sorted(float(v) for v in values if v is not None)
    if not clean:
        return None
    if len(clean) == 1:
        return round(clean[0], 4)
    pos = (len(clean) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(len(clean) - 1, lo + 1)
    frac = pos - lo
    return round(clean[lo] + (clean[hi] - clean[lo]) * frac, 4)


def _summarize_multiturn_b1(
    *,
    sessions: list[dict[str, Any]],
    system_prompt_tokens: int,
    user_lens: list[int],
    gen_lens: list[int],
    warmups: int,
    repeats: int,
    seed_ttft_ms: float,
    tokenizer_source: str,
) -> dict[str, Any]:
    num_turns = len(gen_lens)
    per_turn_agg: list[dict[str, Any]] = []
    for t_idx in range(num_turns):
        t_rows = [s["per_turn"][t_idx] for s in sessions]
        dec_vals = [r["decode_tok_per_s"] for r in t_rows if r["decode_tok_per_s"] is not None]
        per_turn_agg.append(
            {
                "turn": t_idx + 1,
                "prompt_tokens": int(round(statistics.median(r["prompt_tokens"] for r in t_rows))),
                "cached_tokens": int(round(statistics.median(r["cached_tokens"] for r in t_rows))),
                "re_prefilled_tokens": int(
                    round(statistics.median(r["re_prefilled_tokens"] for r in t_rows))
                ),
                "hit_rate_pct": round(statistics.median(r["hit_rate_pct"] for r in t_rows), 2),
                "ttft_ms": round(statistics.median(r["ttft_ms"] for r in t_rows), 3),
                "decode_tok_per_s": round(statistics.median(dec_vals), 3) if dec_vals else None,
                "phys_footprint_gb": round(
                    statistics.median(r["phys_footprint_gb"] for r in t_rows), 3
                ),
            }
        )

    turn1_ttfts = [s["per_turn"][0]["ttft_ms"] for s in sessions]
    turn2_10_ttfts = [
        r["ttft_ms"] for s in sessions for r in s["per_turn"][1:]
    ]
    all_decode_tps = [
        r["decode_tok_per_s"]
        for s in sessions
        for r in s["per_turn"]
        if r["decode_tok_per_s"] is not None
    ]
    spec_tpr = [
        s["spec_tok_per_round"]
        for s in sessions
        if s.get("spec_tok_per_round") is not None
    ]

    summary: dict[str, Any] = {
        "turn1_ttft_ms_median": round(statistics.median(turn1_ttfts), 3),
        "turn2_10_ttft_ms_median": round(statistics.median(turn2_10_ttfts), 3)
        if turn2_10_ttfts
        else None,
        "turn2_10_ttft_ms_p95": round(compute_percentile(turn2_10_ttfts, 95.0), 3)
        if turn2_10_ttfts
        else None,
        "total_prompt_tokens": int(
            round(statistics.median(s["total_prompt_tokens"] for s in sessions))
        ),
        "total_cached_tokens": int(
            round(statistics.median(s["total_cached_tokens"] for s in sessions))
        ),
        "total_re_prefilled_tokens": int(
            round(statistics.median(s["total_re_prefilled_tokens"] for s in sessions))
        ),
        "overall_cache_hit_pct": round(
            statistics.median(s["overall_cache_hit_pct"] for s in sessions), 2
        ),
        "decode_tok_per_s_median": round(statistics.median(all_decode_tps), 3)
        if all_decode_tps
        else None,
        "total_10turn_wall_s_median": round(
            statistics.median(s["session_wall_s"] for s in sessions), 4
        ),
        "peak_phys_footprint_gb": round(
            max(s["peak_phys_footprint_gb"] for s in sessions), 3
        ),
        "turn1_to_turn10_footprint_delta_mb": round(
            statistics.median(s["turn1_to_turn10_footprint_delta_mb"] for s in sessions),
            2,
        ),
        "prefix_evictions": sum(int(s.get("prefix_evictions", 0)) for s in sessions),
    }
    if spec_tpr:
        summary["spec_tok_per_round_median"] = round(statistics.median(spec_tpr), 3)

    return {
        "workload": "multiturn_b1",
        "system_prompt_tokens": system_prompt_tokens,
        "turns_per_session": num_turns,
        "user_lens": user_lens,
        "gen_lens": gen_lens,
        "tokenizer_source": tokenizer_source,
        "warmups": warmups,
        "repeats": repeats,
        "seed_ttft_ms": round(seed_ttft_ms, 3),
        "per_turn": per_turn_agg,
        "summary": summary,
        "first_session_turn_output_sha256s": sessions[0].get(
            "turn_output_sha256s", []
        ),
        "sessions": sessions,
    }


def _summarize_multiturn_b4(
    *,
    runs: list[dict[str, Any]],
    system_prompt_tokens: int,
    batch_size: int,
    user_lens: list[int],
    gen_len: int,
    warmups: int,
    repeats: int,
    seed_ttft_ms: float,
    tokenizer_source: str,
) -> dict[str, Any]:
    num_waves = len(user_lens)
    per_wave_agg: list[dict[str, Any]] = []
    for w_idx in range(num_waves):
        w_rows = [r["waves"][w_idx] for r in runs]
        per_wave_agg.append(
            {
                "wave": w_idx + 1,
                "batch_size": batch_size,
                "total_prompt_tokens": int(
                    round(statistics.median(w["total_prompt_tokens"] for w in w_rows))
                ),
                "total_cached_tokens": int(
                    round(statistics.median(w["total_cached_tokens"] for w in w_rows))
                ),
                "total_re_prefilled_tokens": int(
                    round(statistics.median(w["total_re_prefilled_tokens"] for w in w_rows))
                ),
                "hit_rate_pct": round(
                    statistics.median(w["hit_rate_pct"] for w in w_rows), 2
                ),
                "median_ttft_ms": round(
                    statistics.median(w["median_ttft_ms"] for w in w_rows), 3
                ),
                "max_ttft_ms": round(
                    statistics.median(w["max_ttft_ms"] for w in w_rows), 3
                ),
                "window_decode_tok_per_s": round(
                    statistics.median(w["window_decode_tok_per_s"] for w in w_rows), 3
                ),
                "wave_wall_s": round(
                    statistics.median(w["wave_wall_s"] for w in w_rows), 4
                ),
                "prefix_shared_hits": int(
                    round(statistics.median(w.get("prefix_shared_hits", 0) for w in w_rows))
                ),
                "packed_batch_calls": int(
                    round(statistics.median(w.get("packed_batch_calls", 0) for w in w_rows))
                ),
                "phys_footprint_gb": round(
                    statistics.median(w["phys_footprint_gb"] for w in w_rows), 3
                ),
            }
        )

    wave1_stream_ttfts = [
        t for r in runs for t in r["waves"][0]["stream_ttfts_ms"]
    ]
    wave2_6_stream_ttfts = [
        t for r in runs for w in r["waves"][1:] for t in w["stream_ttfts_ms"]
    ]
    all_stream_ttfts = [
        t for r in runs for w in r["waves"] for t in w["stream_ttfts_ms"]
    ]
    all_wave_agg_decode = [
        w["window_decode_tok_per_s"] for r in runs for w in r["waves"]
    ]
    spec_tpr = [
        r["spec_tok_per_round"]
        for r in runs
        if r.get("spec_tok_per_round") is not None
    ]

    summary: dict[str, Any] = {
        "wave1_ttft_ms_median": round(statistics.median(wave1_stream_ttfts), 3),
        "wave2_6_ttft_ms_median": round(statistics.median(wave2_6_stream_ttfts), 3)
        if wave2_6_stream_ttfts
        else None,
        "all_turns_ttft_ms_p95": round(compute_percentile(all_stream_ttfts, 95.0), 3),
        "aggregate_decode_tok_per_s_median": round(
            statistics.median(all_wave_agg_decode), 3
        ),
        "total_6wave_wall_s_median": round(
            statistics.median(r["run_wall_s"] for r in runs), 4
        ),
        "total_prompt_tokens": int(
            round(statistics.median(r["total_prompt_tokens"] for r in runs))
        ),
        "total_cached_tokens": int(
            round(statistics.median(r["total_cached_tokens"] for r in runs))
        ),
        "total_re_prefilled_tokens": int(
            round(statistics.median(r["total_re_prefilled_tokens"] for r in runs))
        ),
        "overall_cache_hit_pct": round(
            statistics.median(r["overall_cache_hit_pct"] for r in runs), 2
        ),
        "prefix_shared_hits": int(
            round(statistics.median(r["prefix_shared_hits"] for r in runs))
        ),
        "prefix_shared_hits_total": sum(int(r["prefix_shared_hits"]) for r in runs),
        "prefix_evictions": sum(int(r["prefix_evictions"]) for r in runs),
        "packed_batch_calls": int(
            round(statistics.median(r["packed_batch_calls"] for r in runs))
        ),
        "peak_phys_footprint_gb": round(
            max(r["peak_phys_footprint_gb"] for r in runs), 3
        ),
    }
    if spec_tpr:
        summary["spec_tok_per_round_median"] = round(statistics.median(spec_tpr), 3)

    return {
        "workload": "multiturn_b4",
        "system_prompt_tokens": system_prompt_tokens,
        "batch_size": batch_size,
        "waves_per_repeat": num_waves,
        "requests_per_repeat": batch_size * num_waves,
        "user_lens": user_lens,
        "gen_len": gen_len,
        "tokenizer_source": tokenizer_source,
        "warmups": warmups,
        "repeats": repeats,
        "seed_ttft_ms": round(seed_ttft_ms, 3),
        "per_wave": per_wave_agg,
        "summary": summary,
        "runs": runs,
    }


def build_synthetic_tokens(length: int, *, seed: int = 20261121) -> list[int]:
    """Deterministic, non-EOS token sequence shared identically across both engines."""
    rng = random.Random(seed + length * 131)
    return [100 + rng.randrange(32000) for _ in range(length)]


def tokens_sha256(tokens: list[int]) -> str:
    payload = ",".join(str(t) for t in tokens).encode("ascii")
    return hashlib.sha256(payload).hexdigest()[:16]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_phys_footprint_bytes() -> tuple[int, int]:
    """Return (current_phys_footprint_bytes, lifetime_max_phys_footprint_bytes) via XNU ledger."""
    sys.path.insert(0, str(ROOT / "omlx"))
    from omlx.utils.proc_memory import (
        get_lifetime_max_phys_footprint,
        get_phys_footprint,
    )

    return int(get_phys_footprint()), int(get_lifetime_max_phys_footprint())


def summarize_series(values: list[float]) -> dict[str, float] | None:
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return None
    s = sorted(clean)
    return {
        "median": round(statistics.median(s), 4),
        "mean": round(statistics.fmean(s), 4),
        "min": round(s[0], 4),
        "max": round(s[-1], 4),
        "stdev": round(statistics.stdev(s), 4) if len(s) >= 2 else 0.0,
    }


def _read_omlx_cache_evictions(engine: Any) -> int:
    """Safely read cumulative prefix + hot-cache evictions from Stock oMLX scheduler."""
    try:
        sched = engine._engine.engine.scheduler
    except AttributeError:
        return 0
    total = 0
    try:
        if getattr(sched, "block_aware_cache", None) is not None:
            total += int(getattr(sched.block_aware_cache.get_stats(), "evictions", 0) or 0)
    except Exception:
        pass
    try:
        if getattr(sched, "paged_ssd_cache_manager", None) is not None:
            total += int(
                getattr(
                    sched.paged_ssd_cache_manager.get_stats(), "hot_cache_evictions", 0
                )
                or 0
            )
    except Exception:
        pass
    return total


async def run_omlx_multiturn_b1(
    *,
    run_single_raw: Any,
    read_evictions: Any,
    spec: dict[str, Any],
    model_dir: Path,
    post_seed_sleep_s: float = 0.3,
) -> dict[str, Any]:
    """Workload 1 (Stock oMLX): B=1 10-turn sequential short chat w/ shared 256-tok System Prompt."""
    warmups = int(spec["warmups"])
    repeats = int(spec["repeats"])
    system_len = int(spec.get("system_prompt_tokens", MULTITURN_B1_SYSTEM_LEN))
    user_lens = [int(x) for x in spec.get("user_lens", MULTITURN_B1_USER_LENS)]
    gen_lens = [int(x) for x in spec.get("gen_lens", MULTITURN_B1_GEN_LENS)]
    if len(user_lens) != len(gen_lens):
        raise ValueError("multiturn_b1 requires len(user_lens) == len(gen_lens)")

    tok, tok_src = _load_qwen38_tokenizer(model_dir)
    system_prompt = build_multiturn_system_prompt(system_len, tok=tok, seed=51001)

    # Warm Metal/MLX JIT with a tiny unrelated prompt (do not pollute prefix cache).
    await run_single_raw([1, 2, 3, 4, 5, 6, 7, 8], 4, skip_cache_store=True)

    # Seed shared 256-tok System Prompt once with system_prefix_ids so oMLX stores
    # its shared system-prefix boundary snapshot cleanly before Session 0 starts.
    seed_user = _encode_exact_length(tok, "Warmup system prompt seed check.", 8, seed=51002)
    seed_prompt = build_multiturn_turn1_prompt(system_prompt, seed_user)
    seed_run = await run_single_raw(
        seed_prompt,
        1,
        skip_cache_store=False,
        system_prefix_ids=system_prompt,
    )
    if post_seed_sleep_s > 0:
        await asyncio.sleep(post_seed_sleep_s)

    measured_sessions: list[dict[str, Any]] = []
    for s_idx in range(warmups + repeats):
        queries = build_multiturn_b1_session_queries(s_idx, tok=tok, user_lens=user_lens)
        ev_before = int(read_evictions())
        prompt = build_multiturn_turn1_prompt(system_prompt, queries[0])

        t_sess0 = time.perf_counter()
        per_turn: list[dict[str, Any]] = []
        turn_hashes: list[str] = []
        turn1_phys_bytes = 0
        turn_last_phys_bytes = 0
        peak_phys_gb = 0.0

        for t_idx, gen_len in enumerate(gen_lens):
            res = await run_single_raw(
                prompt,
                gen_len,
                skip_cache_store=False,
                system_prefix_ids=system_prompt,
            )
            gen_toks = [int(t) for t in res["output_token_ids"]]
            if len(gen_toks) != gen_len:
                raise RuntimeError(
                    f"oMLX multiturn_b1 session {s_idx} turn {t_idx + 1} generated "
                    f"{len(gen_toks)} tokens, expected {gen_len}"
                )
            p_toks = int(res["prompt_tokens"])
            c_toks = min(p_toks, max(0, int(res["cached_tokens"])))
            re_prefill = max(0, p_toks - c_toks)
            hit_pct = round(100.0 * c_toks / p_toks, 2) if p_toks > 0 else 0.0
            phys_bytes = int(res["phys_footprint_bytes"])
            if t_idx == 0:
                turn1_phys_bytes = phys_bytes
            turn_last_phys_bytes = phys_bytes
            peak_phys_gb = max(peak_phys_gb, float(res["peak_phys_footprint_gb"]))
            turn_hashes.append(tokens_sha256(gen_toks))

            per_turn.append(
                {
                    "turn": t_idx + 1,
                    "prompt_tokens": p_toks,
                    "cached_tokens": c_toks,
                    "re_prefilled_tokens": re_prefill,
                    "hit_rate_pct": hit_pct,
                    "completion_tokens": len(gen_toks),
                    "ttft_ms": res["ttft_ms"],
                    "decode_tok_per_s": res["decode_tok_per_s"],
                    "tpot_ms": res["tpot_ms"],
                    "e2e_seconds": res["e2e_seconds"],
                    "phys_footprint_gb": res["phys_footprint_gb"],
                    "peak_phys_footprint_gb": res["peak_phys_footprint_gb"],
                }
            )
            if t_idx + 1 < len(gen_lens):
                prompt = extend_multiturn_prompt(prompt, gen_toks, queries[t_idx + 1])

        sess_wall_s = time.perf_counter() - t_sess0
        ev_delta = max(0, int(read_evictions()) - ev_before)

        if s_idx >= warmups:
            tot_prompt = sum(r["prompt_tokens"] for r in per_turn)
            tot_cached = sum(r["cached_tokens"] for r in per_turn)
            tot_reprefill = sum(r["re_prefilled_tokens"] for r in per_turn)
            t2_10 = [r["ttft_ms"] for r in per_turn[1:]]
            dec_vals = [
                r["decode_tok_per_s"]
                for r in per_turn
                if r["decode_tok_per_s"] is not None
            ]
            measured_sessions.append(
                {
                    "session_index": s_idx - warmups,
                    "session_wall_s": round(sess_wall_s, 4),
                    "turn1_ttft_ms": per_turn[0]["ttft_ms"],
                    "turn2_10_ttft_ms_median": round(statistics.median(t2_10), 3)
                    if t2_10
                    else None,
                    "turn2_10_ttft_ms_p95": round(compute_percentile(t2_10, 95.0), 3)
                    if t2_10
                    else None,
                    "total_prompt_tokens": tot_prompt,
                    "total_cached_tokens": tot_cached,
                    "total_re_prefilled_tokens": tot_reprefill,
                    "overall_cache_hit_pct": round(100.0 * tot_cached / tot_prompt, 2)
                    if tot_prompt > 0
                    else 0.0,
                    "decode_tok_per_s_median": round(statistics.median(dec_vals), 3)
                    if dec_vals
                    else None,
                    "peak_phys_footprint_gb": round(peak_phys_gb, 3),
                    "turn1_to_turn10_footprint_delta_mb": round(
                        (turn_last_phys_bytes - turn1_phys_bytes) / float(1 << 20), 2
                    ),
                    "prefix_evictions": ev_delta,
                    "turn_output_sha256s": turn_hashes,
                    "per_turn": per_turn,
                }
            )

    return _summarize_multiturn_b1(
        sessions=measured_sessions,
        system_prompt_tokens=system_len,
        user_lens=user_lens,
        gen_lens=gen_lens,
        warmups=warmups,
        repeats=repeats,
        seed_ttft_ms=float(seed_run["ttft_ms"]),
        tokenizer_source=tok_src,
    )


async def run_omlx_multiturn_b4(
    *,
    run_single_raw: Any,
    run_batch_raw: Any,
    read_evictions: Any,
    spec: dict[str, Any],
    model_dir: Path,
    post_seed_sleep_s: float = 0.3,
) -> dict[str, Any]:
    """Workload 2 (Stock oMLX): B=4 concurrent 4-session x 6-wave short chat w/ shared 512-tok System Prompt."""
    warmups = int(spec["warmups"])
    repeats = int(spec["repeats"])
    batch_size = int(spec.get("batch_size", 4))
    system_len = int(spec.get("system_prompt_tokens", MULTITURN_B4_SYSTEM_LEN))
    user_lens = [int(x) for x in spec.get("user_lens", MULTITURN_B4_USER_LENS)]
    gen_len = int(spec.get("gen_len", MULTITURN_B4_GEN_LEN))

    tok, tok_src = _load_qwen38_tokenizer(model_dir)
    system_prompt = build_multiturn_system_prompt(system_len, tok=tok, seed=52001)

    await run_single_raw([1, 2, 3, 4, 5, 6, 7, 8], 4, skip_cache_store=True)

    seed_user = _encode_exact_length(tok, "Warmup B=4 shared system seed.", 8, seed=52002)
    seed_prompt = build_multiturn_turn1_prompt(system_prompt, seed_user)
    seed_run = await run_single_raw(
        seed_prompt,
        1,
        skip_cache_store=False,
        system_prefix_ids=system_prompt,
    )
    if post_seed_sleep_s > 0:
        await asyncio.sleep(post_seed_sleep_s)

    measured_runs: list[dict[str, Any]] = []
    for r_idx in range(warmups + repeats):
        wave_queries = build_multiturn_b4_wave_queries(
            r_idx, tok=tok, batch_size=batch_size, user_lens=user_lens
        )
        prompts = [
            build_multiturn_turn1_prompt(system_prompt, wave_queries[0][s])
            for s in range(batch_size)
        ]

        t_run0 = time.perf_counter()
        waves: list[dict[str, Any]] = []
        peak_phys_gb = 0.0

        for w_idx in range(len(user_lens)):
            ev_before = int(read_evictions())
            wres = await run_batch_raw(
                prompts,
                gen_len,
                skip_cache_store=False,
                system_prefix_ids=system_prompt,
            )
            ev_delta = max(0, int(read_evictions()) - ev_before)
            stream_outputs = wres["stream_output_token_ids"]
            for s_idx, out_ids in enumerate(stream_outputs):
                if len(out_ids) != gen_len:
                    raise RuntimeError(
                        f"oMLX multiturn_b4 run {r_idx} wave {w_idx + 1} slot {s_idx} "
                        f"generated {len(out_ids)} tokens, expected {gen_len}"
                    )
            prompt_lens = [len(p) for p in prompts]
            tot_prompt = sum(prompt_lens)
            tot_cached = min(tot_prompt, max(0, int(wres["total_cached_tokens"])))
            tot_reprefill = max(0, tot_prompt - tot_cached)
            hit_pct = round(100.0 * tot_cached / tot_prompt, 2) if tot_prompt > 0 else 0.0
            peak_phys_gb = max(peak_phys_gb, float(wres["peak_phys_footprint_gb"]))

            waves.append(
                {
                    "wave": w_idx + 1,
                    "batch_size": batch_size,
                    "prompt_tokens_per_session": prompt_lens,
                    "total_prompt_tokens": tot_prompt,
                    "total_cached_tokens": tot_cached,
                    "total_re_prefilled_tokens": tot_reprefill,
                    "hit_rate_pct": hit_pct,
                    "median_ttft_ms": wres["median_ttft_ms"],
                    "max_ttft_ms": wres["max_ttft_ms"],
                    "stream_ttfts_ms": wres["stream_ttfts_ms"],
                    "window_decode_tok_per_s": wres["window_decode_tok_per_s"],
                    "sum_stream_decode_tok_per_s": wres["sum_stream_decode_tok_per_s"],
                    "wave_wall_s": wres["e2e_seconds"],
                    "prefix_shared_hits": 0,
                    "prefix_evictions": ev_delta,
                    "packed_batch_calls": 0,
                    "phys_footprint_gb": wres["phys_footprint_gb"],
                    "peak_phys_footprint_gb": wres["peak_phys_footprint_gb"],
                }
            )
            if w_idx + 1 < len(user_lens):
                prompts = [
                    extend_multiturn_prompt(
                        prompts[s],
                        [int(t) for t in stream_outputs[s]],
                        wave_queries[w_idx + 1][s],
                    )
                    for s in range(batch_size)
                ]

        run_wall_s = time.perf_counter() - t_run0
        if r_idx >= warmups:
            tot_prompt = sum(w["total_prompt_tokens"] for w in waves)
            tot_cached = sum(w["total_cached_tokens"] for w in waves)
            tot_reprefill = sum(w["total_re_prefilled_tokens"] for w in waves)
            measured_runs.append(
                {
                    "repeat_index": r_idx - warmups,
                    "run_wall_s": round(run_wall_s, 4),
                    "total_prompt_tokens": tot_prompt,
                    "total_cached_tokens": tot_cached,
                    "total_re_prefilled_tokens": tot_reprefill,
                    "overall_cache_hit_pct": round(100.0 * tot_cached / tot_prompt, 2)
                    if tot_prompt > 0
                    else 0.0,
                    "prefix_shared_hits": 0,
                    "prefix_evictions": sum(int(w["prefix_evictions"]) for w in waves),
                    "packed_batch_calls": 0,
                    "peak_phys_footprint_gb": round(peak_phys_gb, 3),
                    "waves": waves,
                }
            )

    return _summarize_multiturn_b4(
        runs=measured_runs,
        system_prompt_tokens=system_len,
        batch_size=batch_size,
        user_lens=user_lens,
        gen_len=gen_len,
        warmups=warmups,
        repeats=repeats,
        seed_ttft_ms=float(seed_run["ttft_ms"]),
        tokenizer_source=tok_src,
    )


def run_m5rt_multiturn_b1(
    *,
    run_single: Any,
    spec: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    """Workload 1 (m5rt): B=1 10-turn sequential short chat w/ pinned 256-tok System Prompt."""
    warmups = int(spec["warmups"])
    repeats = int(spec["repeats"])
    system_len = int(spec.get("system_prompt_tokens", MULTITURN_B1_SYSTEM_LEN))
    user_lens = [int(x) for x in spec.get("user_lens", MULTITURN_B1_USER_LENS)]
    gen_lens = [int(x) for x in spec.get("gen_lens", MULTITURN_B1_GEN_LENS)]
    if len(user_lens) != len(gen_lens):
        raise ValueError("multiturn_b1 requires len(user_lens) == len(gen_lens)")

    tok, tok_src = _load_qwen38_tokenizer(model_dir)
    system_prompt = build_multiturn_system_prompt(system_len, tok=tok, seed=51001)

    # Warm Metal pipelines with a tiny unrelated prompt, then pin exact 256-tok System Prompt once.
    run_single([1, 2, 3, 4, 5, 6, 7, 8], 4, pin_prefix=False)
    seed_run = run_single(system_prompt, 1, pin_prefix=True)

    measured_sessions: list[dict[str, Any]] = []
    for s_idx in range(warmups + repeats):
        queries = build_multiturn_b1_session_queries(s_idx, tok=tok, user_lens=user_lens)
        prompt = build_multiturn_turn1_prompt(system_prompt, queries[0])

        t_sess0 = time.perf_counter()
        per_turn: list[dict[str, Any]] = []
        turn_hashes: list[str] = []
        turn1_phys_bytes = 0
        turn_last_phys_bytes = 0
        peak_phys_gb = 0.0
        tot_evictions = 0
        tot_spec_rounds = 0
        tot_spec_accepted = 0

        for t_idx, gen_len in enumerate(gen_lens):
            res = run_single(prompt, gen_len, pin_prefix=False, return_tokens=True)
            gen_toks = [int(t) for t in res["output_token_ids"]]
            if len(gen_toks) != gen_len:
                raise RuntimeError(
                    f"m5rt multiturn_b1 session {s_idx} turn {t_idx + 1} generated "
                    f"{len(gen_toks)} tokens, expected {gen_len}"
                )
            p_toks = int(res["prompt_tokens"])
            c_toks = min(p_toks, max(0, int(res["cached_tokens"])))
            re_prefill = max(0, p_toks - c_toks)
            hit_pct = round(100.0 * c_toks / p_toks, 2) if p_toks > 0 else 0.0
            phys_bytes = int(res["phys_footprint_bytes"])
            if t_idx == 0:
                turn1_phys_bytes = phys_bytes
            turn_last_phys_bytes = phys_bytes
            peak_phys_gb = max(peak_phys_gb, float(res["peak_phys_footprint_gb"]))
            tot_evictions += int(res.get("prefix_evictions", 0))
            tot_spec_rounds += int(res.get("spec_rounds", 0))
            tot_spec_accepted += int(res.get("spec_accepted_tokens", 0))
            turn_hashes.append(tokens_sha256(gen_toks))

            per_turn.append(
                {
                    "turn": t_idx + 1,
                    "prompt_tokens": p_toks,
                    "cached_tokens": c_toks,
                    "re_prefilled_tokens": re_prefill,
                    "hit_rate_pct": hit_pct,
                    "completion_tokens": len(gen_toks),
                    "ttft_ms": res["ttft_ms"],
                    "decode_tok_per_s": res["decode_tok_per_s"],
                    "tpot_ms": res["tpot_ms"],
                    "e2e_seconds": res["e2e_seconds"],
                    "spec_rounds": int(res.get("spec_rounds", 0)),
                    "spec_accepted_tokens": int(res.get("spec_accepted_tokens", 0)),
                    "spec_proposed_tokens": int(res.get("spec_proposed_tokens", 0)),
                    "phys_footprint_gb": res["phys_footprint_gb"],
                    "peak_phys_footprint_gb": res["peak_phys_footprint_gb"],
                }
            )
            if t_idx + 1 < len(gen_lens):
                prompt = extend_multiturn_prompt(prompt, gen_toks, queries[t_idx + 1])

        sess_wall_s = time.perf_counter() - t_sess0
        if s_idx >= warmups:
            tot_prompt = sum(r["prompt_tokens"] for r in per_turn)
            tot_cached = sum(r["cached_tokens"] for r in per_turn)
            tot_reprefill = sum(r["re_prefilled_tokens"] for r in per_turn)
            t2_10 = [r["ttft_ms"] for r in per_turn[1:]]
            dec_vals = [
                r["decode_tok_per_s"]
                for r in per_turn
                if r["decode_tok_per_s"] is not None
            ]
            sess_entry: dict[str, Any] = {
                "session_index": s_idx - warmups,
                "session_wall_s": round(sess_wall_s, 4),
                "turn1_ttft_ms": per_turn[0]["ttft_ms"],
                "turn2_10_ttft_ms_median": round(statistics.median(t2_10), 3)
                if t2_10
                else None,
                "turn2_10_ttft_ms_p95": round(compute_percentile(t2_10, 95.0), 3)
                if t2_10
                else None,
                "total_prompt_tokens": tot_prompt,
                "total_cached_tokens": tot_cached,
                "total_re_prefilled_tokens": tot_reprefill,
                "overall_cache_hit_pct": round(100.0 * tot_cached / tot_prompt, 2)
                if tot_prompt > 0
                else 0.0,
                "decode_tok_per_s_median": round(statistics.median(dec_vals), 3)
                if dec_vals
                else None,
                "peak_phys_footprint_gb": round(peak_phys_gb, 3),
                "turn1_to_turn10_footprint_delta_mb": round(
                    (turn_last_phys_bytes - turn1_phys_bytes) / float(1 << 20), 2
                ),
                "prefix_evictions": tot_evictions,
                "turn_output_sha256s": turn_hashes,
                "per_turn": per_turn,
            }
            if tot_spec_rounds > 0:
                sess_entry["spec_tok_per_round"] = round(
                    (tot_spec_accepted + tot_spec_rounds) / float(tot_spec_rounds), 3
                )
            measured_sessions.append(sess_entry)

    return _summarize_multiturn_b1(
        sessions=measured_sessions,
        system_prompt_tokens=system_len,
        user_lens=user_lens,
        gen_lens=gen_lens,
        warmups=warmups,
        repeats=repeats,
        seed_ttft_ms=float(seed_run["ttft_ms"]),
        tokenizer_source=tok_src,
    )


def run_m5rt_multiturn_b4(
    *,
    run_single: Any,
    run_batch: Any,
    spec: dict[str, Any],
    model_dir: Path,
) -> dict[str, Any]:
    """Workload 2 (m5rt): B=4 concurrent 4-session x 6-wave short chat w/ pinned 512-tok System Prompt."""
    warmups = int(spec["warmups"])
    repeats = int(spec["repeats"])
    batch_size = int(spec.get("batch_size", 4))
    system_len = int(spec.get("system_prompt_tokens", MULTITURN_B4_SYSTEM_LEN))
    user_lens = [int(x) for x in spec.get("user_lens", MULTITURN_B4_USER_LENS)]
    gen_len = int(spec.get("gen_len", MULTITURN_B4_GEN_LEN))

    tok, tok_src = _load_qwen38_tokenizer(model_dir)
    system_prompt = build_multiturn_system_prompt(system_len, tok=tok, seed=52001)

    run_single([1, 2, 3, 4, 5, 6, 7, 8], 4, pin_prefix=False)
    seed_run = run_single(system_prompt, 1, pin_prefix=True)

    measured_runs: list[dict[str, Any]] = []
    for r_idx in range(warmups + repeats):
        wave_queries = build_multiturn_b4_wave_queries(
            r_idx, tok=tok, batch_size=batch_size, user_lens=user_lens
        )
        prompts = [
            build_multiturn_turn1_prompt(system_prompt, wave_queries[0][s])
            for s in range(batch_size)
        ]

        t_run0 = time.perf_counter()
        waves: list[dict[str, Any]] = []
        peak_phys_gb = 0.0
        tot_spec_rounds = 0
        tot_spec_accepted = 0

        for w_idx in range(len(user_lens)):
            wres = run_batch(prompts, gen_len, pin_prefix=False, return_tokens=True)
            stream_outputs = wres["stream_output_token_ids"]
            for s_idx, out_ids in enumerate(stream_outputs):
                if len(out_ids) != gen_len:
                    raise RuntimeError(
                        f"m5rt multiturn_b4 run {r_idx} wave {w_idx + 1} slot {s_idx} "
                        f"generated {len(out_ids)} tokens, expected {gen_len}"
                    )
            prompt_lens = [len(p) for p in prompts]
            tot_prompt = sum(prompt_lens)
            tot_cached = min(tot_prompt, max(0, int(wres["total_cached_tokens"])))
            tot_reprefill = max(0, tot_prompt - tot_cached)
            hit_pct = round(100.0 * tot_cached / tot_prompt, 2) if tot_prompt > 0 else 0.0
            peak_phys_gb = max(peak_phys_gb, float(wres["peak_phys_footprint_gb"]))
            tot_spec_rounds += int(wres.get("spec_rounds", 0))
            tot_spec_accepted += int(wres.get("spec_accepted_tokens", 0))

            waves.append(
                {
                    "wave": w_idx + 1,
                    "batch_size": batch_size,
                    "prompt_tokens_per_session": prompt_lens,
                    "total_prompt_tokens": tot_prompt,
                    "total_cached_tokens": tot_cached,
                    "total_re_prefilled_tokens": tot_reprefill,
                    "hit_rate_pct": hit_pct,
                    "median_ttft_ms": wres["median_ttft_ms"],
                    "max_ttft_ms": wres["max_ttft_ms"],
                    "stream_ttfts_ms": wres["stream_ttfts_ms"],
                    "window_decode_tok_per_s": wres["window_decode_tok_per_s"],
                    "sum_stream_decode_tok_per_s": wres["sum_stream_decode_tok_per_s"],
                    "wave_wall_s": wres["e2e_seconds"],
                    "prefix_shared_hits": int(wres.get("prefix_shared_hits", 0)),
                    "prefix_evictions": int(wres.get("prefix_evictions", 0)),
                    "packed_batch_calls": int(wres.get("packed_batch_calls", 0)),
                    "phys_footprint_gb": wres["phys_footprint_gb"],
                    "peak_phys_footprint_gb": wres["peak_phys_footprint_gb"],
                }
            )
            if w_idx + 1 < len(user_lens):
                prompts = [
                    extend_multiturn_prompt(
                        prompts[s],
                        [int(t) for t in stream_outputs[s]],
                        wave_queries[w_idx + 1][s],
                    )
                    for s in range(batch_size)
                ]

        run_wall_s = time.perf_counter() - t_run0
        if r_idx >= warmups:
            tot_prompt = sum(w["total_prompt_tokens"] for w in waves)
            tot_cached = sum(w["total_cached_tokens"] for w in waves)
            tot_reprefill = sum(w["total_re_prefilled_tokens"] for w in waves)
            run_entry: dict[str, Any] = {
                "repeat_index": r_idx - warmups,
                "run_wall_s": round(run_wall_s, 4),
                "total_prompt_tokens": tot_prompt,
                "total_cached_tokens": tot_cached,
                "total_re_prefilled_tokens": tot_reprefill,
                "overall_cache_hit_pct": round(100.0 * tot_cached / tot_prompt, 2)
                if tot_prompt > 0
                else 0.0,
                "prefix_shared_hits": sum(int(w["prefix_shared_hits"]) for w in waves),
                "prefix_evictions": sum(int(w["prefix_evictions"]) for w in waves),
                "packed_batch_calls": sum(int(w["packed_batch_calls"]) for w in waves),
                "peak_phys_footprint_gb": round(peak_phys_gb, 3),
                "waves": waves,
            }
            if tot_spec_rounds > 0:
                run_entry["spec_tok_per_round"] = round(
                    (tot_spec_accepted + tot_spec_rounds) / float(tot_spec_rounds), 3
                )
            measured_runs.append(run_entry)

    return _summarize_multiturn_b4(
        runs=measured_runs,
        system_prompt_tokens=system_len,
        batch_size=batch_size,
        user_lens=user_lens,
        gen_len=gen_len,
        warmups=warmups,
        repeats=repeats,
        seed_ttft_ms=float(seed_run["ttft_ms"]),
        tokenizer_source=tok_src,
    )


# ---------------------------------------------------------------------------
# Stock oMLX Native MLX Metal Worker
# ---------------------------------------------------------------------------


async def run_omlx_worker_async(spec: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "omlx"))
    import mlx.core as mx
    import mlx_lm
    import omlx
    from omlx.custom_kernels import native_kernel_status
    from omlx.custom_kernels.nax import is_nax_available
    from omlx.engine.batched import BatchedEngine
    from omlx.engine_core import get_mlx_executor
    from omlx.model_settings import ModelSettings
    from omlx.scheduler import SchedulerConfig

    model_info = MODELS[spec["model_key"]]
    model_dir = str((ROOT / model_info["model_dir"]).resolve())
    mode = spec.get("mode", "ar")  # ar | mtp | dflash2
    kv_mode = spec.get("kv_mode", "bf16")  # bf16 | affine4
    use_prefix_cache = bool(spec.get("use_prefix_cache", False))
    max_num_seqs = int(spec.get("max_num_seqs", 16))
    prefill_step_size = int(spec.get("prefill_step_size", 2048))

    settings_kwargs: dict[str, Any] = {
        "external_engine": None,  # Strictly native in-process MLX Metal backend
    }
    if mode == "mtp":
        settings_kwargs["mtp_enabled"] = True
    elif mode == "dflash2":
        draft_dir = str((ROOT / model_info["draft_dir"]).resolve())
        settings_kwargs["dflash_enabled"] = True
        settings_kwargs["dflash_draft_model"] = draft_dir

    if kv_mode == "affine4":
        settings_kwargs["turboquant_kv_enabled"] = True
        settings_kwargs["turboquant_kv_scheme"] = "affine4"
        settings_kwargs["turboquant_kv_bits"] = 4

    model_settings = ModelSettings(**settings_kwargs)

    cache_tmp_ctx = tempfile.TemporaryDirectory() if use_prefix_cache else None
    cache_dir = cache_tmp_ctx.name if cache_tmp_ctx is not None else None

    try:
        hot_cache_max_size = (
            int(spec.get("prefix_cache_bytes", 12 * (1 << 30)))
            if use_prefix_cache
            else 0
        )
        sc = SchedulerConfig(
            max_num_seqs=max(16, max_num_seqs),
            prefill_step_size=prefill_step_size,
            prefill_speed_priority=True,
            decode_fairness=False,
            paged_ssd_cache_dir=cache_dir,
            hot_cache_only=bool(use_prefix_cache),
            hot_cache_max_size=hot_cache_max_size,
            model_name=model_info["label"],
            model_path=model_dir,
        )

        loop = asyncio.get_running_loop()
        mlx_exec = get_mlx_executor()

        async def sync_gpu() -> None:
            await loop.run_in_executor(mlx_exec, mx.synchronize)

        t_load0 = time.perf_counter()
        engine = BatchedEngine(
            model_name=model_dir,
            scheduler_config=sc,
            model_settings=model_settings,
        )
        await engine.start()
        if mode == "dflash2":
            from omlx.speculative.dflash import load_dflash_drafter

            draft_dir = str((ROOT / model_info["draft_dir"]).resolve())

            def _load_df():
                return load_dflash_drafter(
                    draft_dir,
                    engine.prepare_dflash_target(),
                    tokenizer=engine.tokenizer,
                )

            drafter = await loop.run_in_executor(mlx_exec, _load_df)
            engine.set_dflash_drafter(drafter)

        await sync_gpu()
        load_seconds = time.perf_counter() - t_load0
        post_load_phys, post_load_peak_phys = read_phys_footprint_bytes()

        lm_inner = getattr(engine._model, "language_model", engine._model)
        backend_identity = {
            "backend": "stock_omlx_mlx_metal",
            "omlx_version": getattr(omlx, "__version__", "unknown"),
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "mlx_lm_version": getattr(mlx_lm, "__version__", "unknown"),
            "engine_class": f"{type(engine).__module__}.{type(engine).__qualname__}",
            "model_class": f"{type(engine._model).__module__}.{type(engine._model).__qualname__}",
            "external_engine": getattr(model_settings, "external_engine", None),
            "is_nax_available": bool(is_nax_available()),
            "native_kernel_status": native_kernel_status(),
            "device_info": dict(mx.device_info()),
            "mtp_active": bool(getattr(lm_inner, "_omlx_mtp_decode_enabled", False)),
            "mtp_depth": getattr(lm_inner, "_omlx_mtp_depth", None),
            "dflash_attached": engine.dflash_drafter is not None,
            "kv_mode": kv_mode,
            "prefill_step_size": prefill_step_size,
            "use_prefix_cache": use_prefix_cache,
            "hot_cache_max_size": hot_cache_max_size,
        }

        async def run_single(
            prompt_tokens: list[int],
            max_new_tokens: int,
            *,
            skip_cache_store: bool = True,
        ) -> dict[str, Any]:
            await sync_gpu()
            mx.reset_peak_memory()

            start = time.perf_counter()
            first_token_at: float | None = None
            last_token_at: float | None = None
            first_batch_tokens = 0
            prev_tokens = 0
            last_out = None

            async for out in engine.stream_generate(
                prompt=prompt_tokens,
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
                skip_cache_store=skip_cache_store,
            ):
                delta = out.completion_tokens - prev_tokens
                if delta > 0:
                    ga = (
                        float(out.generated_at)
                        if out.generated_at is not None
                        else time.perf_counter()
                    )
                    gu = (
                        float(out.generated_until)
                        if out.generated_until is not None
                        else ga
                    )
                    if first_token_at is None:
                        first_token_at = ga
                        first_batch_tokens = out.completion_tokens
                    last_token_at = gu
                prev_tokens = out.completion_tokens
                last_out = out

            await sync_gpu()
            end = time.perf_counter()
            cur_phys, peak_phys = read_phys_footprint_bytes()
            mlx_peak = int(mx.get_peak_memory())

            if last_out is None or first_token_at is None:
                raise RuntimeError("Stock oMLX produced no output tokens")

            comp_tokens = int(last_out.completion_tokens)
            rep_prompt_tokens = int(last_out.prompt_tokens)
            cached_tokens = int(last_out.cached_tokens)
            ttft_s = max(1e-9, first_token_at - start)
            e2e_s = max(1e-9, end - start)
            decode_tokens = comp_tokens - first_batch_tokens
            decode_s = (
                (last_token_at - first_token_at)
                if (last_token_at is not None and last_token_at > first_token_at)
                else max(0.0, end - first_token_at)
            )
            decode_tps = (
                decode_tokens / decode_s
                if decode_tokens > 0 and decode_s > 0
                else None
            )
            tpot_ms = (
                (decode_s / decode_tokens) * 1000.0
                if decode_tokens > 0 and decode_s > 0
                else None
            )
            return {
                "prompt_tokens": rep_prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": comp_tokens,
                "first_batch_tokens": first_batch_tokens,
                "decode_token_count": decode_tokens,
                "ttft_ms": round(ttft_s * 1000.0, 3),
                "prefill_tok_per_s": round(rep_prompt_tokens / ttft_s, 3),
                "decode_seconds": round(decode_s, 5),
                "decode_tok_per_s": round(decode_tps, 3) if decode_tps is not None else None,
                "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
                "e2e_seconds": round(e2e_s, 5),
                "e2e_output_tok_per_s": round(comp_tokens / e2e_s, 3),
                "phys_footprint_gb": round(cur_phys / (1 << 30), 3),
                "peak_phys_footprint_gb": round(peak_phys / (1 << 30), 3),
                "mlx_peak_memory_gb": round(mlx_peak / (1 << 30), 3),
                "text_preview": last_out.text[:160] if last_out.text else "",
                "output_text": last_out.text if spec.get("capture_tokens", False) else None,
            }

        async def run_batch(
            prompts: list[list[int]],
            max_new_tokens: int,
            *,
            skip_cache_store: bool = True,
        ) -> dict[str, Any]:
            from omlx.request import SamplingParams

            await sync_gpu()
            mx.reset_peak_memory()
            engine_core = engine._engine
            sp = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
            )

            async def _one_stream(p_toks: list[int]) -> dict[str, Any]:
                s_start = time.perf_counter()
                ft: float | None = None
                lt: float | None = None
                fb = 0
                prev = 0
                comp = 0
                cached = 0
                rid = await engine_core.add_request(
                    prompt=p_toks,
                    sampling_params=sp,
                    skip_cache_store=skip_cache_store,
                )
                async for out in engine_core.stream_outputs(rid):
                    if out.completion_tokens > prev:
                        now = (
                            float(out.generated_at)
                            if getattr(out, "generated_at", None) is not None
                            else time.perf_counter()
                        )
                        until = (
                            float(out.generated_until)
                            if getattr(out, "generated_until", None) is not None
                            else now
                        )
                        if ft is None:
                            ft = now
                            fb = out.completion_tokens
                        lt = until
                    prev = out.completion_tokens
                    if out.finished:
                        comp = int(out.completion_tokens)
                        cached = int(getattr(out, "cached_tokens", 0) or 0)
                s_end = time.perf_counter()
                if ft is None:
                    ft = s_end
                if lt is None or lt <= ft:
                    lt = s_end
                dec_toks = comp - fb
                dec_s = max(1e-9, lt - ft)
                return {
                    "ttft_s": ft - s_start,
                    "first_at": ft,
                    "last_at": lt,
                    "end_at": s_end,
                    "completion_tokens": comp,
                    "first_batch_tokens": fb,
                    "decode_tokens": dec_toks,
                    "decode_tps": dec_toks / dec_s if dec_toks > 0 else 0.0,
                    "cached_tokens": cached,
                }

            wall_start = time.perf_counter()
            streams = await asyncio.gather(*[_one_stream(p) for p in prompts])
            await sync_gpu()
            wall_end = time.perf_counter()
            cur_phys, peak_phys = read_phys_footprint_bytes()
            mlx_peak = int(mx.get_peak_memory())

            wall_s = max(1e-9, wall_end - wall_start)
            total_comp = sum(s["completion_tokens"] for s in streams)
            total_dec = sum(s["decode_tokens"] for s in streams)
            sum_stream_dec_tps = sum(s["decode_tps"] for s in streams)
            min_first = min(s["first_at"] for s in streams)
            max_first = max(s["first_at"] for s in streams)
            max_last = max(s["last_at"] for s in streams)
            win_dec_s = max(1e-9, max_last - min_first)
            return {
                "batch_size": len(prompts),
                "total_completion_tokens": total_comp,
                "total_decode_tokens": total_dec,
                "total_cached_tokens": sum(s["cached_tokens"] for s in streams),
                "median_ttft_ms": round(
                    statistics.median(s["ttft_s"] for s in streams) * 1000.0, 3
                ),
                "max_ttft_ms": round((max_first - wall_start) * 1000.0, 3),
                "sum_stream_decode_tok_per_s": round(sum_stream_dec_tps, 3),
                "window_decode_tok_per_s": round(total_dec / win_dec_s, 3),
                "per_stream_median_decode_tok_per_s": round(
                    statistics.median(s["decode_tps"] for s in streams), 3
                ),
                "e2e_seconds": round(wall_s, 5),
                "e2e_output_tok_per_s": round(total_comp / wall_s, 3),
                "phys_footprint_gb": round(cur_phys / (1 << 30), 3),
                "peak_phys_footprint_gb": round(peak_phys / (1 << 30), 3),
                "mlx_peak_memory_gb": round(mlx_peak / (1 << 30), 3),
            }

        async def run_single_raw(
            prompt_tokens: list[int],
            max_new_tokens: int,
            *,
            skip_cache_store: bool = False,
            system_prefix_ids: list[int] | None = None,
        ) -> dict[str, Any]:
            from omlx.request import SamplingParams

            await sync_gpu()
            mx.reset_peak_memory()
            engine_core = engine._engine
            sp = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
            )
            add_kwargs: dict[str, Any] = {
                "prompt": list(prompt_tokens),
                "sampling_params": sp,
                "skip_cache_store": skip_cache_store,
            }
            if system_prefix_ids:
                add_kwargs["system_prefix_ids"] = list(system_prefix_ids)

            start = time.perf_counter()
            rid = await engine_core.add_request(**add_kwargs)
            first_token_at: float | None = None
            last_token_at: float | None = None
            first_batch_tokens = 0
            prev_tokens = 0
            accum_tokens: list[int] = []
            final_tokens: list[int] = []
            last_out = None

            async for out in engine_core.stream_outputs(rid):
                new_ids = getattr(out, "new_token_ids", None)
                if new_ids:
                    accum_tokens.extend(int(t) for t in new_ids)
                delta = out.completion_tokens - prev_tokens
                if delta > 0:
                    ga = (
                        float(out.generated_at)
                        if getattr(out, "generated_at", None) is not None
                        else time.perf_counter()
                    )
                    gu = (
                        float(out.generated_until)
                        if getattr(out, "generated_until", None) is not None
                        else ga
                    )
                    if first_token_at is None:
                        first_token_at = ga
                        first_batch_tokens = out.completion_tokens
                    last_token_at = gu
                prev_tokens = out.completion_tokens
                if out.finished and getattr(out, "output_token_ids", None):
                    final_tokens = [int(t) for t in out.output_token_ids]
                last_out = out

            await sync_gpu()
            end = time.perf_counter()
            cur_phys, peak_phys = read_phys_footprint_bytes()
            mlx_peak = int(mx.get_peak_memory())

            if last_out is None or first_token_at is None:
                raise RuntimeError("Stock oMLX raw token runner produced no output tokens")

            output_token_ids = final_tokens if final_tokens else accum_tokens
            comp_tokens = len(output_token_ids) or int(last_out.completion_tokens)
            rep_prompt_tokens = int(
                getattr(last_out, "prompt_tokens", 0) or len(prompt_tokens)
            )
            cached_tokens = int(getattr(last_out, "cached_tokens", 0) or 0)
            ttft_s = max(1e-9, first_token_at - start)
            e2e_s = max(1e-9, end - start)
            decode_tokens = comp_tokens - first_batch_tokens
            decode_s = (
                (last_token_at - first_token_at)
                if (last_token_at is not None and last_token_at > first_token_at)
                else max(0.0, end - first_token_at)
            )
            decode_tps = (
                decode_tokens / decode_s
                if decode_tokens > 0 and decode_s > 0
                else None
            )
            tpot_ms = (
                (decode_s / decode_tokens) * 1000.0
                if decode_tokens > 0 and decode_s > 0
                else None
            )
            return {
                "prompt_tokens": rep_prompt_tokens,
                "cached_tokens": cached_tokens,
                "completion_tokens": comp_tokens,
                "output_token_ids": output_token_ids,
                "first_batch_tokens": first_batch_tokens,
                "decode_token_count": decode_tokens,
                "ttft_ms": round(ttft_s * 1000.0, 3),
                "prefill_tok_per_s": round(rep_prompt_tokens / ttft_s, 3),
                "decode_seconds": round(decode_s, 5),
                "decode_tok_per_s": round(decode_tps, 3) if decode_tps is not None else None,
                "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
                "e2e_seconds": round(e2e_s, 5),
                "e2e_output_tok_per_s": round(comp_tokens / e2e_s, 3),
                "phys_footprint_bytes": cur_phys,
                "phys_footprint_gb": round(cur_phys / (1 << 30), 3),
                "peak_phys_footprint_gb": round(peak_phys / (1 << 30), 3),
                "mlx_peak_memory_gb": round(mlx_peak / (1 << 30), 3),
            }

        async def run_batch_raw(
            prompts: list[list[int]],
            max_new_tokens: int,
            *,
            skip_cache_store: bool = False,
            system_prefix_ids: list[int] | None = None,
        ) -> dict[str, Any]:
            from omlx.request import SamplingParams

            await sync_gpu()
            mx.reset_peak_memory()
            engine_core = engine._engine
            sp = SamplingParams(
                max_tokens=max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                ignore_eos=True,
            )

            async def _one_stream(p_toks: list[int]) -> dict[str, Any]:
                add_kwargs: dict[str, Any] = {
                    "prompt": list(p_toks),
                    "sampling_params": sp,
                    "skip_cache_store": skip_cache_store,
                }
                if system_prefix_ids:
                    add_kwargs["system_prefix_ids"] = list(system_prefix_ids)
                s_start = time.perf_counter()
                ft: float | None = None
                lt: float | None = None
                fb = 0
                prev = 0
                comp = 0
                cached = 0
                accum_ids: list[int] = []
                final_ids: list[int] = []
                rid = await engine_core.add_request(**add_kwargs)
                async for out in engine_core.stream_outputs(rid):
                    new_ids = getattr(out, "new_token_ids", None)
                    if new_ids:
                        accum_ids.extend(int(t) for t in new_ids)
                    if out.completion_tokens > prev:
                        now = (
                            float(out.generated_at)
                            if getattr(out, "generated_at", None) is not None
                            else time.perf_counter()
                        )
                        until = (
                            float(out.generated_until)
                            if getattr(out, "generated_until", None) is not None
                            else now
                        )
                        if ft is None:
                            ft = now
                            fb = out.completion_tokens
                        lt = until
                    prev = out.completion_tokens
                    if out.finished:
                        comp = int(out.completion_tokens)
                        cached = int(getattr(out, "cached_tokens", 0) or 0)
                        if getattr(out, "output_token_ids", None):
                            final_ids = [int(t) for t in out.output_token_ids]
                s_end = time.perf_counter()
                if ft is None:
                    ft = s_end
                if lt is None or lt <= ft:
                    lt = s_end
                out_ids = final_ids if final_ids else accum_ids
                comp = len(out_ids) or comp
                dec_toks = comp - fb
                dec_s = max(1e-9, lt - ft)
                return {
                    "ttft_s": ft - s_start,
                    "first_at": ft,
                    "last_at": lt,
                    "end_at": s_end,
                    "completion_tokens": comp,
                    "output_token_ids": out_ids,
                    "first_batch_tokens": fb,
                    "decode_tokens": dec_toks,
                    "decode_tps": dec_toks / dec_s if dec_toks > 0 else 0.0,
                    "cached_tokens": cached,
                }

            wall_start = time.perf_counter()
            streams = await asyncio.gather(*[_one_stream(p) for p in prompts])
            await sync_gpu()
            wall_end = time.perf_counter()
            cur_phys, peak_phys = read_phys_footprint_bytes()
            mlx_peak = int(mx.get_peak_memory())

            wall_s = max(1e-9, wall_end - wall_start)
            total_comp = sum(s["completion_tokens"] for s in streams)
            total_dec = sum(s["decode_tokens"] for s in streams)
            sum_stream_dec_tps = sum(s["decode_tps"] for s in streams)
            min_first = min(s["first_at"] for s in streams)
            max_first = max(s["first_at"] for s in streams)
            max_last = max(s["last_at"] for s in streams)
            win_dec_s = max(1e-9, max_last - min_first)
            return {
                "batch_size": len(prompts),
                "total_completion_tokens": total_comp,
                "total_decode_tokens": total_dec,
                "total_cached_tokens": sum(s["cached_tokens"] for s in streams),
                "stream_ttfts_ms": [round(s["ttft_s"] * 1000.0, 3) for s in streams],
                "stream_output_token_ids": [s["output_token_ids"] for s in streams],
                "median_ttft_ms": round(
                    statistics.median(s["ttft_s"] for s in streams) * 1000.0, 3
                ),
                "max_ttft_ms": round((max_first - wall_start) * 1000.0, 3),
                "sum_stream_decode_tok_per_s": round(sum_stream_dec_tps, 3),
                "window_decode_tok_per_s": round(total_dec / win_dec_s, 3),
                "per_stream_median_decode_tok_per_s": round(
                    statistics.median(s["decode_tps"] for s in streams), 3
                ),
                "e2e_seconds": round(wall_s, 5),
                "e2e_output_tok_per_s": round(total_comp / wall_s, 3),
                "phys_footprint_bytes": cur_phys,
                "phys_footprint_gb": round(cur_phys / (1 << 30), 3),
                "peak_phys_footprint_gb": round(peak_phys / (1 << 30), 3),
                "mlx_peak_memory_gb": round(mlx_peak / (1 << 30), 3),
            }

        task_kind = spec["task_kind"]
        payload_results: list[dict[str, Any]] = []

        if task_kind == "prefill_sweep":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec.get("max_new_tokens", 1))
            for p_len in spec["prompt_lengths"]:
                ptoks = build_synthetic_tokens(int(p_len))
                for _ in range(warmups):
                    await run_single(ptoks, max_new, skip_cache_store=True)
                runs = [
                    await run_single(ptoks, max_new, skip_cache_store=True)
                    for _ in range(repeats)
                ]
                payload_results.append(
                    {
                        "prompt_length": int(p_len),
                        "prompt_sha256": tokens_sha256(ptoks),
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "ttft_ms": summarize_series([r["ttft_ms"] for r in runs]),
                        "prefill_tok_per_s": summarize_series(
                            [r["prefill_tok_per_s"] for r in runs]
                        ),
                        "peak_phys_footprint_gb": max(
                            r["peak_phys_footprint_gb"] for r in runs
                        ),
                        "runs": runs,
                    }
                )

        elif task_kind == "single_decode":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec["max_new_tokens"])
            prompt_name = spec["prompt_name"]
            ptoks = (
                CODING_PROMPT_TOKENS
                if prompt_name == "coding"
                else REASONING_PROMPT_TOKENS
                if prompt_name == "reasoning"
                else build_synthetic_tokens(int(prompt_name))
            )
            for _ in range(warmups):
                await run_single(ptoks, max_new, skip_cache_store=True)
            runs = [
                await run_single(ptoks, max_new, skip_cache_store=True)
                for _ in range(repeats)
            ]
            payload_results.append(
                {
                    "prompt_name": prompt_name,
                    "prompt_length": len(ptoks),
                    "prompt_sha256": tokens_sha256(ptoks),
                    "max_new_tokens": max_new,
                    "warmups": warmups,
                    "repeats": repeats,
                    "ttft_ms": summarize_series([r["ttft_ms"] for r in runs]),
                    "decode_tok_per_s": summarize_series(
                        [r["decode_tok_per_s"] for r in runs]
                    ),
                    "tpot_ms": summarize_series([r["tpot_ms"] for r in runs]),
                    "e2e_output_tok_per_s": summarize_series(
                        [r["e2e_output_tok_per_s"] for r in runs]
                    ),
                    "peak_phys_footprint_gb": max(
                        r["peak_phys_footprint_gb"] for r in runs
                    ),
                    "runs": runs,
                }
            )

        elif task_kind == "batch_sweep":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec["max_new_tokens"])
            for b in spec["batch_sizes"]:
                b_int = int(b)
                prompts = [list(CODING_PROMPT_TOKENS) for _ in range(b_int)]
                for _ in range(warmups):
                    await run_batch(prompts, max_new, skip_cache_store=True)
                runs = [
                    await run_batch(prompts, max_new, skip_cache_store=True)
                    for _ in range(repeats)
                ]
                payload_results.append(
                    {
                        "batch_size": b_int,
                        "prompt_length": len(CODING_PROMPT_TOKENS),
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "median_ttft_ms": summarize_series(
                            [r["median_ttft_ms"] for r in runs]
                        ),
                        "max_ttft_ms": summarize_series(
                            [r["max_ttft_ms"] for r in runs]
                        ),
                        "sum_stream_decode_tok_per_s": summarize_series(
                            [r["sum_stream_decode_tok_per_s"] for r in runs]
                        ),
                        "window_decode_tok_per_s": summarize_series(
                            [r["window_decode_tok_per_s"] for r in runs]
                        ),
                        "e2e_output_tok_per_s": summarize_series(
                            [r["e2e_output_tok_per_s"] for r in runs]
                        ),
                        "peak_phys_footprint_gb": max(
                            r["peak_phys_footprint_gb"] for r in runs
                        ),
                        "runs": runs,
                    }
                )

        elif task_kind == "longctx_uncontaminated_footprint":
            max_new = int(spec.get("max_new_tokens", 64))
            for ctx_len in spec["context_lengths"]:
                ptoks = build_synthetic_tokens(int(ctx_len))
                run = await run_single(ptoks, max_new, skip_cache_store=True)
                payload_results.append(
                    {
                        "context_length": int(ctx_len),
                        "prompt_sha256": tokens_sha256(ptoks),
                        "max_new_tokens": max_new,
                        "cold_ttft_ms": run["ttft_ms"],
                        "prefill_tok_per_s": run["prefill_tok_per_s"],
                        "decode_tok_per_s": run["decode_tok_per_s"],
                        "phys_footprint_gb": run["phys_footprint_gb"],
                        "peak_phys_footprint_gb": run["peak_phys_footprint_gb"],
                        "mlx_peak_memory_gb": run["mlx_peak_memory_gb"],
                    }
                )

        elif task_kind == "longctx_decode_scaling":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec.get("max_new_tokens", 64))
            for item in spec["depths"]:
                prefix_len = int(item["prefix_len"])
                tail_len = int(item["tail_len"])
                shared_prefix = build_synthetic_tokens(prefix_len, seed=77000 + prefix_len)
                # Prime prefix into hot cache on first warmup, then warmups+repeats hit full KV depth
                seed_prompt = shared_prefix + build_synthetic_tokens(tail_len, seed=1)
                cold_seed = await run_single(seed_prompt, max_new, skip_cache_store=False)
                await asyncio.sleep(0.25)
                for w_i in range(max(0, warmups - 1)):
                    wp = shared_prefix + build_synthetic_tokens(tail_len, seed=100 + w_i)
                    await run_single(wp, max_new, skip_cache_store=True)
                runs = []
                for r_i in range(repeats):
                    rp = shared_prefix + build_synthetic_tokens(tail_len, seed=500 + r_i)
                    runs.append(await run_single(rp, max_new, skip_cache_store=True))
                payload_results.append(
                    {
                        "total_prompt_tokens": prefix_len + tail_len,
                        "prefix_len": prefix_len,
                        "tail_len": tail_len,
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "cold_seed_ttft_ms": cold_seed["ttft_ms"],
                        "warm_cached_tokens": runs[0]["cached_tokens"],
                        "decode_tok_per_s": summarize_series(
                            [r["decode_tok_per_s"] for r in runs]
                        ),
                        "tpot_ms": summarize_series([r["tpot_ms"] for r in runs]),
                        "runs": runs,
                    }
                )

        elif task_kind == "prefix_cache_bench":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            prefix_len = int(spec.get("prefix_len", 2048))
            tail_len = int(spec.get("tail_len", 32))
            max_new = int(spec.get("max_new_tokens", 16))
            shared_prefix = build_synthetic_tokens(prefix_len, seed=9001)
            # Warm JIT with tiny unrelated prompt
            await run_single([1, 2, 3, 4, 5, 6, 7, 8], 4, skip_cache_store=True)
            # Cold seed (populates prefix cache)
            cold_prompt = shared_prefix + build_synthetic_tokens(tail_len, seed=10)
            cold_run = await run_single(cold_prompt, max_new, skip_cache_store=False)
            await asyncio.sleep(0.3)
            # B=1 warm hits
            for w_i in range(warmups):
                wp = shared_prefix + build_synthetic_tokens(tail_len, seed=200 + w_i)
                await run_single(wp, max_new, skip_cache_store=True)
            b1_runs = [
                await run_single(
                    shared_prefix + build_synthetic_tokens(tail_len, seed=300 + r_i),
                    max_new,
                    skip_cache_store=True,
                )
                for r_i in range(repeats)
            ]
            # B=4 warm concurrent hits
            for w_i in range(warmups):
                wps = [
                    shared_prefix
                    + build_synthetic_tokens(tail_len, seed=1000 + w_i * 10 + s)
                    for s in range(4)
                ]
                await run_batch(wps, max_new, skip_cache_store=True)
            b4_runs = [
                await run_batch(
                    [
                        shared_prefix
                        + build_synthetic_tokens(tail_len, seed=2000 + r_i * 10 + s)
                        for s in range(4)
                    ],
                    max_new,
                    skip_cache_store=True,
                )
                for r_i in range(repeats)
            ]
            payload_results.append(
                {
                    "prefix_len": prefix_len,
                    "tail_len": tail_len,
                    "max_new_tokens": max_new,
                    "warmups": warmups,
                    "repeats": repeats,
                    "cold_b1_ttft_ms": cold_run["ttft_ms"],
                    "warm_b1_ttft_ms": summarize_series([r["ttft_ms"] for r in b1_runs]),
                    "warm_b1_cached_tokens": b1_runs[0]["cached_tokens"],
                    "warm_b4_median_ttft_ms": summarize_series(
                        [r["median_ttft_ms"] for r in b4_runs]
                    ),
                    "warm_b4_max_ttft_ms": summarize_series(
                        [r["max_ttft_ms"] for r in b4_runs]
                    ),
                    "warm_b4_total_cached_tokens": b4_runs[0]["total_cached_tokens"],
                    "b1_runs": b1_runs,
                    "b4_runs": b4_runs,
                }
            )

        elif task_kind == "multiturn_b1":
            payload_results.append(
                await run_omlx_multiturn_b1(
                    run_single_raw=run_single_raw,
                    read_evictions=lambda: _read_omlx_cache_evictions(engine),
                    spec=spec,
                    model_dir=Path(model_dir),
                )
            )

        elif task_kind == "multiturn_b4":
            payload_results.append(
                await run_omlx_multiturn_b4(
                    run_single_raw=run_single_raw,
                    run_batch_raw=run_batch_raw,
                    read_evictions=lambda: _read_omlx_cache_evictions(engine),
                    spec=spec,
                    model_dir=Path(model_dir),
                )
            )

        else:
            raise ValueError(f"Unknown omlx task_kind: {task_kind}")

        await engine.stop()
        gc.collect()
        mx.synchronize()
        mx.clear_cache()

        return {
            "status": "ok",
            "spec_name": spec["name"],
            "model_key": spec["model_key"],
            "backend_identity": backend_identity,
            "load_seconds": round(load_seconds, 4),
            "post_load_phys_footprint_gb": round(post_load_phys / (1 << 30), 3),
            "post_load_peak_phys_footprint_gb": round(post_load_peak_phys / (1 << 30), 3),
            "results": payload_results,
        }
    finally:
        if cache_tmp_ctx is not None:
            cache_tmp_ctx.cleanup()


# ---------------------------------------------------------------------------
# m5rt v0.8.0-m5rt.11 Native Metal 4 / Apple10 Worker
# ---------------------------------------------------------------------------


def _resolve_m5rt_ext_dir(spec: dict[str, Any]) -> Path:
    raw = (
        spec.get("m5rt_build_dir")
        or os.environ.get("OMLX_M5RT_BUILD_DIR")
        or "build/miv-v001"
    )
    cand = Path(raw)
    if not cand.is_absolute():
        cand = ROOT / cand
    cand = cand.resolve()
    if (cand / "miv").is_dir():
        return (cand / "miv").resolve()
    if (cand / "m5rt").is_dir():
        return (cand / "m5rt").resolve()
    return cand


def _ensure_mtp_ready_for_artifact(
    artifact: Path, model_dir: Path, model_info: dict[str, str], spec: dict[str, Any]
) -> tuple[str, Path | None]:
    """Check whether base .omlxm5 already embeds ModelComponent::mtp; if not, ensure sidecar MTP .omlxm5 exists."""
    sys.path.insert(0, str(_SCRIPT_DIR))
    from m5_artifact import _is_mtp_object_name, extract_mtp_artifact, read_artifact

    _, manifest = read_artifact(artifact, verify_payload=False)
    has_embedded = any(
        _is_mtp_object_name(str(obj.get("name", "")))
        for obj in manifest.get("objects", [])
    )
    if has_embedded and not spec.get("mtp_artifact_override"):
        return "embedded_in_base_artifact", None

    rel_mtp = (
        spec.get("mtp_artifact_override")
        or model_info.get("mtp_artifact")
        or f".cache/b5/real/{spec['model_key']}-mtp-pretiled.omlxm5"
    )
    mtp_path = Path(str(rel_mtp))
    if not mtp_path.is_absolute():
        mtp_path = (ROOT / mtp_path).resolve()
    if not mtp_path.is_file():
        want_pretile = any(
            str(obj.get("layout", ""))
            in ("mx_nax_tiled_bk128_v1", "oq_nax_tiled_g64_v1", "oq_nax_tiled_g128_v1")
            for obj in manifest.get("objects", [])
        )
        extract_mtp_artifact(
            model_dir,
            mtp_path,
            source_repo=str(manifest.get("model", {}).get("source_repo", "")),
            nax_pretile=want_pretile,
        )
    return "sidecar_mtp_artifact", mtp_path


def run_m5rt_worker(spec: dict[str, Any]) -> dict[str, Any]:
    ext_dir = _resolve_m5rt_ext_dir(spec)
    sys.path.insert(0, str(ext_dir))
    import omlx_m5_native as native

    model_info = MODELS[spec["model_key"]]
    model_dir = (ROOT / model_info["model_dir"]).resolve()
    artifact = (
        ROOT / str(spec.get("artifact_override") or model_info["artifact"])
    ).resolve()
    qwen_metallib = ext_dir / "metallib/omlx_m5_qwen_ar.metallib"
    dflash_metallib = ext_dir / "metallib/omlx_m5_dflash2_program.metallib"

    m5rt_tag = str(spec.get("m5rt_tag", "m5rt11"))
    mode = spec.get("mode", "ar")  # ar | dflash2 | mtp
    max_context = int(spec.get("max_context", 32768))
    max_concurrency = int(spec.get("max_concurrency", 1))
    kv_dtype = spec.get("kv_dtype", "bf16")
    flash_nax_prefill = spec.get("flash_nax_prefill", "off")
    attention_split_kv = spec.get("attention_split_kv", "on")
    sampling_contract = spec.get("sampling_contract", "position_coupled_v1")
    prefix_cache_bytes = int(spec.get("prefix_cache_bytes", 0))
    spec_search = spec.get("spec_search", "off")
    batch_mode = "on" if max_concurrency >= 2 else "off"

    options = native.LoadOptions()
    options.model_dir = str(model_dir)
    options.max_context = max_context
    options.max_concurrency = max_concurrency
    options.kv_dtype = kv_dtype
    options.sampling_contract = sampling_contract
    if flash_nax_prefill == "on":
        options.prefill_engine = "flash_nax"
    if hasattr(options, "spec_search"):
        options.spec_search = spec_search

    overrides: dict[str, str] = {
        "artifact": str(artifact),
        "qwen_ar_metallib": str(qwen_metallib),
        "attention_split_kv": attention_split_kv,
        "batch_mode": batch_mode,
        "batch_projection_prototype": "off",
    }
    if flash_nax_prefill == "on":
        overrides["flash_nax_prefill"] = "on"
    if prefix_cache_bytes > 0:
        overrides["prefix_cache_bytes"] = str(prefix_cache_bytes)
    rolling_prefix_cache_raw = spec.get("rolling_prefix_cache")
    if rolling_prefix_cache_raw is not None:
        if isinstance(rolling_prefix_cache_raw, bool):
            overrides["rolling_prefix_cache"] = "on" if rolling_prefix_cache_raw else "off"
        else:
            norm_rpc = str(rolling_prefix_cache_raw).strip().lower()
            if norm_rpc in ("on", "1", "true"):
                overrides["rolling_prefix_cache"] = "on"
            elif norm_rpc in ("off", "0", "false"):
                overrides["rolling_prefix_cache"] = "off"
            else:
                raise ValueError(
                    f"Invalid rolling_prefix_cache value in spec {spec.get('name')!r}: "
                    f"{rolling_prefix_cache_raw!r}"
                )

    mtp_source = "disabled"
    if mode == "dflash2":
        overrides["enable_mtp"] = "off"
        draft_dir = (ROOT / model_info["draft_dir"]).resolve()
        draft_artifact = (ROOT / model_info["draft_artifact"]).resolve()
        overrides["draft_artifact"] = str(draft_artifact)
        overrides["dflash2_metallib"] = str(dflash_metallib)
        options.draft_dir = str(draft_dir)
        options.speculative.backend = native.SpeculativeBackend.DFLASH2
        options.speculative.draft_tokens = int(spec.get("draft_tokens", 7))
    elif mode == "mtp":
        mtp_source, ext_mtp = _ensure_mtp_ready_for_artifact(
            artifact, model_dir, model_info, spec
        )
        overrides["enable_mtp"] = "on"
        if ext_mtp is not None:
            overrides["mtp_artifact"] = str(ext_mtp)
        if spec.get("mtp_draft_tokens") is not None:
            overrides["mtp_draft_tokens"] = str(int(spec["mtp_draft_tokens"]))
    else:
        overrides["enable_mtp"] = "off"

    options.path_overrides = overrides

    t_load0 = time.perf_counter()
    engine = native.Engine.create(options)
    load_seconds = time.perf_counter() - t_load0
    post_load_phys, post_load_peak_phys = read_phys_footprint_bytes()

    try:
        st0 = engine.stats()
        backend_identity = {
            "backend": f"m5rt_v0.8.0_{m5rt_tag}_apple10_metal4",
            "m5rt_build_dir": str(ext_dir),
            "mode": mode,
            "mtp_source": mtp_source,
            "backend_kind": str(getattr(st0, "backend_kind", "")),
            "model_execution_kind": str(getattr(st0, "model_execution_kind", "")),
            "compiled_execution_source_id": str(
                getattr(st0, "compiled_execution_source_id", "")
            ),
            "sampling_contract_id": str(getattr(st0, "sampling_contract_id", "")),
            "kv_dtype": kv_dtype,
            "flash_nax_prefill": flash_nax_prefill,
            "attention_split_kv": attention_split_kv,
            "spec_search": spec_search,
            "prefix_cache_bytes": prefix_cache_bytes,
            "rolling_prefix_cache": overrides.get("rolling_prefix_cache", "default"),
            "env_overrides": spec.get("env", {}),
        }

        def run_single(
            prompt_tokens: list[int],
            max_new_tokens: int,
            *,
            pin_prefix: bool = False,
            return_tokens: bool = False,
            timeout_s: float = 900.0,
        ) -> dict[str, Any]:
            before = engine.stats()
            req = native.SubmitRequest()
            req.prompt_tokens = prompt_tokens
            req.max_new_tokens = max_new_tokens
            req.sampling.temperature = 0.0
            req.sampling.top_p = 1.0
            req.sampling.top_k = 0
            req.sampling.seed = 0
            if hasattr(req, "stop_token_seqs"):
                req.stop_token_seqs = []
            if pin_prefix and hasattr(req, "pin_prefix"):
                req.pin_prefix = True

            start = time.perf_counter()
            rid = engine.submit(req)
            first_at: float | None = None
            last_at: float | None = None
            first_batch = 0
            tokens: list[int] = []
            terminal = None
            while terminal is None:
                if time.perf_counter() - start > timeout_s:
                    engine.cancel(rid)
                    raise TimeoutError(f"m5rt request {rid} timed out")
                for ev in engine.poll_events(20_000):
                    if ev.request_id != rid:
                        continue
                    if ev.kind == native.EventKind.TOKENS and ev.tokens:
                        now = time.perf_counter()
                        last_at = now
                        if first_at is None:
                            first_at = now
                            first_batch = len(ev.tokens)
                        tokens.extend(int(t) for t in ev.tokens)
                    else:
                        terminal = ev
            end = time.perf_counter()
            after = engine.stats()
            cur_phys, peak_phys = read_phys_footprint_bytes()

            if terminal.kind != native.EventKind.FINISHED or first_at is None:
                raise RuntimeError(f"m5rt request failed: {getattr(terminal, 'error', 'unknown')}")

            ttft_s = max(1e-9, first_at - start)
            e2e_s = max(1e-9, end - start)
            dec_toks = len(tokens) - first_batch
            dec_s = (
                (last_at - first_at)
                if (last_at is not None and last_at > first_at)
                else max(0.0, end - first_at)
            )
            dec_tps = dec_toks / dec_s if dec_toks > 0 and dec_s > 0 else None
            tpot_ms = (dec_s / dec_toks) * 1000.0 if dec_toks > 0 and dec_s > 0 else None
            prefill_gpu_ms = float(after.ar_prefill_gpu_ms) - float(before.ar_prefill_gpu_ms)
            decode_gpu_ms = float(after.ar_decode_gpu_ms) - float(before.ar_decode_gpu_ms)
            restored_toks = int(after.prefix_restored_tokens) - int(before.prefix_restored_tokens)
            evictions = int(after.prefix_evictions) - int(before.prefix_evictions)
            spec_rounds = int(after.spec_rounds) - int(before.spec_rounds)
            spec_accepted = int(after.spec_accepted_tokens) - int(before.spec_accepted_tokens)
            spec_proposed = int(after.spec_proposal_tokens) - int(before.spec_proposal_tokens)

            out_dict: dict[str, Any] = {
                "prompt_tokens": len(prompt_tokens),
                "cached_tokens": restored_toks,
                "prefix_evictions": evictions,
                "completion_tokens": len(tokens),
                "first_batch_tokens": first_batch,
                "decode_token_count": dec_toks,
                "ttft_ms": round(ttft_s * 1000.0, 3),
                "prefill_gpu_ms": round(prefill_gpu_ms, 3) if prefill_gpu_ms > 0 else None,
                "decode_gpu_ms": round(decode_gpu_ms, 3) if decode_gpu_ms > 0 else None,
                "prefill_tok_per_s": round(len(prompt_tokens) / ttft_s, 3),
                "decode_seconds": round(dec_s, 5),
                "decode_tok_per_s": round(dec_tps, 3) if dec_tps is not None else None,
                "tpot_ms": round(tpot_ms, 3) if tpot_ms is not None else None,
                "e2e_seconds": round(e2e_s, 5),
                "e2e_output_tok_per_s": round(len(tokens) / e2e_s, 3),
                "spec_rounds": spec_rounds,
                "spec_accepted_tokens": spec_accepted,
                "spec_proposed_tokens": spec_proposed,
                "phys_footprint_bytes": cur_phys,
                "phys_footprint_gb": round(cur_phys / (1 << 30), 3),
                "peak_phys_footprint_gb": round(peak_phys / (1 << 30), 3),
                "tokens": tokens if spec.get("capture_tokens", False) else None,
            }
            if return_tokens:
                out_dict["output_token_ids"] = tokens
            return out_dict

        def run_batch(
            prompts: list[list[int]],
            max_new_tokens: int,
            *,
            pin_prefix: bool = False,
            return_tokens: bool = False,
            timeout_s: float = 900.0,
        ) -> dict[str, Any]:
            before = engine.stats()
            reqs = []
            for p_toks in prompts:
                r = native.SubmitRequest()
                r.prompt_tokens = p_toks
                r.max_new_tokens = max_new_tokens
                r.sampling.temperature = 0.0
                r.sampling.top_p = 1.0
                r.sampling.top_k = 0
                r.sampling.seed = 0
                if hasattr(r, "stop_token_seqs"):
                    r.stop_token_seqs = []
                if pin_prefix and hasattr(r, "pin_prefix"):
                    r.pin_prefix = True
                reqs.append(r)

            wall_start = time.perf_counter()
            rids = [engine.submit(r) for r in reqs]
            pending = set(rids)
            per_req: dict[int, dict[str, Any]] = {
                rid: {
                    "first_at": None,
                    "last_at": None,
                    "fb": 0,
                    "toks": [],
                    "terminal": None,
                }
                for rid in rids
            }
            while pending:
                if time.perf_counter() - wall_start > timeout_s:
                    for rid in sorted(pending):
                        engine.cancel(rid)
                    raise TimeoutError("m5rt batch timed out")
                for ev in engine.poll_events(20_000):
                    if ev.request_id not in per_req:
                        continue
                    st = per_req[ev.request_id]
                    if ev.kind == native.EventKind.TOKENS and ev.tokens:
                        now = time.perf_counter()
                        st["last_at"] = now
                        if st["first_at"] is None:
                            st["first_at"] = now
                            st["fb"] = len(ev.tokens)
                        st["toks"].extend(int(t) for t in ev.tokens)
                    elif st["terminal"] is None:
                        st["terminal"] = ev
                        pending.discard(ev.request_id)
            wall_end = time.perf_counter()
            after = engine.stats()
            cur_phys, peak_phys = read_phys_footprint_bytes()

            streams = []
            for rid in rids:
                st = per_req[rid]
                if st["terminal"].kind != native.EventKind.FINISHED:
                    raise RuntimeError(f"m5rt batch stream {rid} failed")
                ft = float(st["first_at"])
                lt = float(st["last_at"])
                comp = len(st["toks"])
                dec_toks = comp - int(st["fb"])
                dec_s = max(1e-9, lt - ft)
                streams.append(
                    {
                        "ttft_s": ft - wall_start,
                        "first_at": ft,
                        "last_at": lt,
                        "completion_tokens": comp,
                        "decode_tokens": dec_toks,
                        "decode_tps": dec_toks / dec_s if dec_toks > 0 else 0.0,
                        "toks": st["toks"],
                    }
                )

            wall_s = max(1e-9, wall_end - wall_start)
            total_comp = sum(s["completion_tokens"] for s in streams)
            total_dec = sum(s["decode_tokens"] for s in streams)
            sum_stream_dec_tps = sum(s["decode_tps"] for s in streams)
            min_first = min(s["first_at"] for s in streams)
            max_first = max(s["first_at"] for s in streams)
            max_last = max(s["last_at"] for s in streams)
            win_dec_s = max(1e-9, max_last - min_first)
            restored_toks = int(after.prefix_restored_tokens) - int(
                before.prefix_restored_tokens
            )
            packed_calls = int(after.packed_batch_calls) - int(before.packed_batch_calls)
            shared_hits = int(after.prefix_shared_hits) - int(before.prefix_shared_hits)
            evictions = int(after.prefix_evictions) - int(before.prefix_evictions)
            spec_rounds = int(after.spec_rounds) - int(before.spec_rounds)
            spec_accepted = int(after.spec_accepted_tokens) - int(before.spec_accepted_tokens)
            spec_proposed = int(after.spec_proposal_tokens) - int(before.spec_proposal_tokens)

            out_dict = {
                "batch_size": len(prompts),
                "total_completion_tokens": total_comp,
                "total_decode_tokens": total_dec,
                "total_cached_tokens": restored_toks,
                "prefix_shared_hits": shared_hits,
                "prefix_evictions": evictions,
                "packed_batch_calls": packed_calls,
                "spec_rounds": spec_rounds,
                "spec_accepted_tokens": spec_accepted,
                "spec_proposed_tokens": spec_proposed,
                "stream_ttfts_ms": [round(s["ttft_s"] * 1000.0, 3) for s in streams],
                "median_ttft_ms": round(
                    statistics.median(s["ttft_s"] for s in streams) * 1000.0, 3
                ),
                "max_ttft_ms": round((max_first - wall_start) * 1000.0, 3),
                "sum_stream_decode_tok_per_s": round(sum_stream_dec_tps, 3),
                "window_decode_tok_per_s": round(total_dec / win_dec_s, 3),
                "per_stream_median_decode_tok_per_s": round(
                    statistics.median(s["decode_tps"] for s in streams), 3
                ),
                "e2e_seconds": round(wall_s, 5),
                "e2e_output_tok_per_s": round(total_comp / wall_s, 3),
                "phys_footprint_bytes": cur_phys,
                "phys_footprint_gb": round(cur_phys / (1 << 30), 3),
                "peak_phys_footprint_gb": round(peak_phys / (1 << 30), 3),
            }
            if return_tokens:
                out_dict["stream_output_token_ids"] = [s["toks"] for s in streams]
            return out_dict

        task_kind = spec["task_kind"]
        payload_results: list[dict[str, Any]] = []

        if task_kind == "prefill_sweep":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec.get("max_new_tokens", 1))
            for p_len in spec["prompt_lengths"]:
                ptoks = build_synthetic_tokens(int(p_len))
                for _ in range(warmups):
                    run_single(ptoks, max_new)
                runs = [run_single(ptoks, max_new) for _ in range(repeats)]
                payload_results.append(
                    {
                        "prompt_length": int(p_len),
                        "prompt_sha256": tokens_sha256(ptoks),
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "ttft_ms": summarize_series([r["ttft_ms"] for r in runs]),
                        "prefill_gpu_ms": summarize_series(
                            [r["prefill_gpu_ms"] for r in runs if r["prefill_gpu_ms"] is not None]
                        ),
                        "prefill_tok_per_s": summarize_series(
                            [r["prefill_tok_per_s"] for r in runs]
                        ),
                        "peak_phys_footprint_gb": max(
                            r["peak_phys_footprint_gb"] for r in runs
                        ),
                        "runs": runs,
                    }
                )

        elif task_kind == "single_decode":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec["max_new_tokens"])
            prompt_name = spec["prompt_name"]
            ptoks = (
                CODING_PROMPT_TOKENS
                if prompt_name == "coding"
                else REASONING_PROMPT_TOKENS
                if prompt_name == "reasoning"
                else build_synthetic_tokens(int(prompt_name))
            )
            for _ in range(warmups):
                run_single(ptoks, max_new)
            runs = [run_single(ptoks, max_new) for _ in range(repeats)]
            payload_results.append(
                {
                    "prompt_name": prompt_name,
                    "prompt_length": len(ptoks),
                    "prompt_sha256": tokens_sha256(ptoks),
                    "max_new_tokens": max_new,
                    "warmups": warmups,
                    "repeats": repeats,
                    "ttft_ms": summarize_series([r["ttft_ms"] for r in runs]),
                    "decode_tok_per_s": summarize_series(
                        [r["decode_tok_per_s"] for r in runs]
                    ),
                    "tpot_ms": summarize_series([r["tpot_ms"] for r in runs]),
                    "e2e_output_tok_per_s": summarize_series(
                        [r["e2e_output_tok_per_s"] for r in runs]
                    ),
                    "peak_phys_footprint_gb": max(
                        r["peak_phys_footprint_gb"] for r in runs
                    ),
                    "runs": runs,
                }
            )

        elif task_kind == "batch_single":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec["max_new_tokens"])
            b_int = int(spec["batch_size"])
            prompts = [list(CODING_PROMPT_TOKENS) for _ in range(b_int)]
            if b_int == 1:
                for _ in range(warmups):
                    run_single(CODING_PROMPT_TOKENS, max_new)
                s_runs = [run_single(CODING_PROMPT_TOKENS, max_new) for _ in range(repeats)]
                payload_results.append(
                    {
                        "batch_size": 1,
                        "prompt_length": len(CODING_PROMPT_TOKENS),
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "median_ttft_ms": summarize_series([r["ttft_ms"] for r in s_runs]),
                        "max_ttft_ms": summarize_series([r["ttft_ms"] for r in s_runs]),
                        "sum_stream_decode_tok_per_s": summarize_series(
                            [r["decode_tok_per_s"] for r in s_runs]
                        ),
                        "window_decode_tok_per_s": summarize_series(
                            [r["decode_tok_per_s"] for r in s_runs]
                        ),
                        "e2e_output_tok_per_s": summarize_series(
                            [r["e2e_output_tok_per_s"] for r in s_runs]
                        ),
                        "peak_phys_footprint_gb": max(
                            r["peak_phys_footprint_gb"] for r in s_runs
                        ),
                        "runs": s_runs,
                    }
                )
            else:
                for _ in range(warmups):
                    run_batch(prompts, max_new)
                runs = [run_batch(prompts, max_new) for _ in range(repeats)]
                payload_results.append(
                    {
                        "batch_size": b_int,
                        "prompt_length": len(CODING_PROMPT_TOKENS),
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "median_ttft_ms": summarize_series(
                            [r["median_ttft_ms"] for r in runs]
                        ),
                        "max_ttft_ms": summarize_series([r["max_ttft_ms"] for r in runs]),
                        "sum_stream_decode_tok_per_s": summarize_series(
                            [r["sum_stream_decode_tok_per_s"] for r in runs]
                        ),
                        "window_decode_tok_per_s": summarize_series(
                            [r["window_decode_tok_per_s"] for r in runs]
                        ),
                        "e2e_output_tok_per_s": summarize_series(
                            [r["e2e_output_tok_per_s"] for r in runs]
                        ),
                        "peak_phys_footprint_gb": max(
                            r["peak_phys_footprint_gb"] for r in runs
                        ),
                        "runs": runs,
                    }
                )

        elif task_kind == "longctx_uncontaminated_footprint":
            max_new = int(spec.get("max_new_tokens", 64))
            for ctx_len in spec["context_lengths"]:
                ptoks = build_synthetic_tokens(int(ctx_len))
                run = run_single(ptoks, max_new)
                payload_results.append(
                    {
                        "context_length": int(ctx_len),
                        "prompt_sha256": tokens_sha256(ptoks),
                        "max_new_tokens": max_new,
                        "cold_ttft_ms": run["ttft_ms"],
                        "prefill_tok_per_s": run["prefill_tok_per_s"],
                        "decode_tok_per_s": run["decode_tok_per_s"],
                        "phys_footprint_gb": run["phys_footprint_gb"],
                        "peak_phys_footprint_gb": run["peak_phys_footprint_gb"],
                    }
                )

        elif task_kind == "longctx_decode_scaling":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            max_new = int(spec.get("max_new_tokens", 64))
            for item in spec["depths"]:
                prefix_len = int(item["prefix_len"])
                tail_len = int(item["tail_len"])
                shared_prefix = build_synthetic_tokens(prefix_len, seed=77000 + prefix_len)
                seed_prompt = shared_prefix + build_synthetic_tokens(tail_len, seed=1)
                cold_seed = run_single(seed_prompt, max_new, pin_prefix=True)
                for w_i in range(max(0, warmups - 1)):
                    wp = shared_prefix + build_synthetic_tokens(tail_len, seed=100 + w_i)
                    run_single(wp, max_new, pin_prefix=False)
                runs = []
                for r_i in range(repeats):
                    rp = shared_prefix + build_synthetic_tokens(tail_len, seed=500 + r_i)
                    runs.append(run_single(rp, max_new, pin_prefix=False))
                payload_results.append(
                    {
                        "total_prompt_tokens": prefix_len + tail_len,
                        "prefix_len": prefix_len,
                        "tail_len": tail_len,
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "cold_seed_ttft_ms": cold_seed["ttft_ms"],
                        "warm_cached_tokens": runs[0]["cached_tokens"],
                        "decode_tok_per_s": summarize_series(
                            [r["decode_tok_per_s"] for r in runs]
                        ),
                        "tpot_ms": summarize_series([r["tpot_ms"] for r in runs]),
                        "runs": runs,
                    }
                )

        elif task_kind == "prefix_cache_bench":
            warmups = int(spec["warmups"])
            repeats = int(spec["repeats"])
            prefix_len = int(spec.get("prefix_len", 2048))
            tail_len = int(spec.get("tail_len", 32))
            max_new = int(spec.get("max_new_tokens", 16))
            shared_prefix = build_synthetic_tokens(prefix_len, seed=9001)
            # Warm pipelines with short unrelated prompt
            run_single([1, 2, 3, 4, 5, 6, 7, 8], 4, pin_prefix=False)
            # Cold seed (pins shared prefix)
            cold_prompt = shared_prefix + build_synthetic_tokens(tail_len, seed=10)
            cold_run = run_single(cold_prompt, max_new, pin_prefix=True)
            if int(spec.get("batch_size", 1)) == 1:
                for w_i in range(warmups):
                    wp = shared_prefix + build_synthetic_tokens(tail_len, seed=200 + w_i)
                    run_single(wp, max_new, pin_prefix=False)
                b1_runs = [
                    run_single(
                        shared_prefix + build_synthetic_tokens(tail_len, seed=300 + r_i),
                        max_new,
                        pin_prefix=False,
                    )
                    for r_i in range(repeats)
                ]
                payload_results.append(
                    {
                        "prefix_len": prefix_len,
                        "tail_len": tail_len,
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "cold_b1_ttft_ms": cold_run["ttft_ms"],
                        "warm_b1_ttft_ms": summarize_series([r["ttft_ms"] for r in b1_runs]),
                        "warm_b1_cached_tokens": b1_runs[0]["cached_tokens"],
                        "b1_runs": b1_runs,
                    }
                )
            else:
                for w_i in range(warmups):
                    wps = [
                        shared_prefix
                        + build_synthetic_tokens(tail_len, seed=1000 + w_i * 10 + s)
                        for s in range(4)
                    ]
                    run_batch(wps, max_new, pin_prefix=False)
                b4_runs = [
                    run_batch(
                        [
                            shared_prefix
                            + build_synthetic_tokens(tail_len, seed=2000 + r_i * 10 + s)
                            for s in range(4)
                        ],
                        max_new,
                        pin_prefix=False,
                    )
                    for r_i in range(repeats)
                ]
                payload_results.append(
                    {
                        "prefix_len": prefix_len,
                        "tail_len": tail_len,
                        "max_new_tokens": max_new,
                        "warmups": warmups,
                        "repeats": repeats,
                        "cold_seed_ttft_ms": cold_run["ttft_ms"],
                        "warm_b4_median_ttft_ms": summarize_series(
                            [r["median_ttft_ms"] for r in b4_runs]
                        ),
                        "warm_b4_max_ttft_ms": summarize_series(
                            [r["max_ttft_ms"] for r in b4_runs]
                        ),
                        "warm_b4_total_cached_tokens": b4_runs[0]["total_cached_tokens"],
                        "warm_b4_shared_hits": b4_runs[0]["prefix_shared_hits"],
                        "b4_runs": b4_runs,
                    }
                )
        elif task_kind == "multiturn_b1":
            payload_results.append(
                run_m5rt_multiturn_b1(
                    run_single=run_single,
                    spec=spec,
                    model_dir=model_dir,
                )
            )
        elif task_kind == "multiturn_b4":
            payload_results.append(
                run_m5rt_multiturn_b4(
                    run_single=run_single,
                    run_batch=run_batch,
                    spec=spec,
                    model_dir=model_dir,
                )
            )
        else:
            raise ValueError(f"Unknown m5rt task_kind: {task_kind}")

        return {
            "status": "ok",
            "spec_name": spec["name"],
            "model_key": spec["model_key"],
            "backend_identity": backend_identity,
            "load_seconds": round(load_seconds, 4),
            "post_load_phys_footprint_gb": round(post_load_phys / (1 << 30), 3),
            "post_load_peak_phys_footprint_gb": round(post_load_peak_phys / (1 << 30), 3),
            "results": payload_results,
        }
    finally:
        engine.shutdown()


# ---------------------------------------------------------------------------
# Subprocess Orchestrator & Suite Definitions
# ---------------------------------------------------------------------------


def execute_worker_subprocess(spec: dict[str, Any]) -> dict[str, Any]:
    print(f"\n>>> Running [{spec['name']}] ({spec['engine']} / {spec['model_key']}) ...", flush=True)
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    # Strip any prior OMLX_M5_* flags unless explicitly specified in spec["env"]
    for k in list(env.keys()):
        if k.startswith("OMLX_M5_"):
            del env[k]
    for k, v in spec.get("env", {}).items():
        env[str(k)] = str(v)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".spec.json", delete=False, encoding="utf-8"
    ) as sf, tempfile.NamedTemporaryFile(
        mode="w", suffix=".out.json", delete=False, encoding="utf-8"
    ) as of:
        json.dump(spec, sf)
        sf_path = Path(sf.name)
        of_path = Path(of.name)

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker-spec",
                str(sf_path),
                "--worker-out",
                str(of_path),
            ],
            env=env,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=int(spec.get("subprocess_timeout_s", 1800)),
        )
        wall = time.perf_counter() - t0
        if of_path.exists() and of_path.stat().st_size > 0:
            payload = json.loads(of_path.read_text(encoding="utf-8"))
            print(
                f"<<< Finished [{spec['name']}] in {wall:.1f}s (status={payload.get('status')})",
                flush=True,
            )
            return payload
        return {
            "status": "error",
            "spec_name": spec["name"],
            "model_key": spec["model_key"],
            "returncode": proc.returncode,
            "stderr_tail": proc.stderr[-2000:],
            "stdout_tail": proc.stdout[-1000:],
        }
    finally:
        for p in (sf_path, of_path):
            if p.exists():
                p.unlink()


def build_dim_a_specs(
    warmups: int, repeats: int, m5rt_tag: str = "m5rt11"
) -> list[dict[str, Any]]:
    lengths = [128, 512, 1024, 4096, 8192]
    return [
        # 1) Qwen3.8-27B-MXFP4
        {
            "name": "27b_mxfp4_omlx_native",
            "engine": "omlx",
            "model_key": "27b-mxfp4",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_default_a16",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "flash_nax_prefill": "off",
            "attention_split_kv": "off",
            "env": {"OMLX_M5_DISABLE_MTP": "1"},
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_fast_a16_flashnax",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "flash_nax_prefill": "on",
            "attention_split_kv": "off",
            "env": {
                "OMLX_M5_DISABLE_MTP": "1",
            },
            "warmups": warmups,
            "repeats": repeats,
        },
        # 2) Qwen3.8-27B-oQ4e
        {
            "name": "27b_oq4e_omlx_native",
            "engine": "omlx",
            "model_key": "27b-oq4e",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_oq4e_{m5rt_tag}_default_rowexact",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-oq4e",
            "mode": "ar",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "flash_nax_prefill": "off",
            "attention_split_kv": "off",
            "env": {"OMLX_M5_DISABLE_MTP": "1"},
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_oq4e_{m5rt_tag}_flashnax",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-oq4e",
            "mode": "ar",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "flash_nax_prefill": "on",
            "attention_split_kv": "off",
            "env": {"OMLX_M5_DISABLE_MTP": "1"},
            "warmups": warmups,
            "repeats": repeats,
        },
        # 3) Qwen3.6-35B-A3B-MXFP4 (MoE)
        {
            "name": "35b_a3b_mxfp4_omlx_native",
            "engine": "omlx",
            "model_key": "35b-a3b-mxfp4",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"35b_a3b_mxfp4_{m5rt_tag}_moe_nax_flashnax",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "35b-a3b-mxfp4",
            "mode": "ar",
            "task_kind": "prefill_sweep",
            "prompt_lengths": lengths,
            "max_new_tokens": 1,
            "flash_nax_prefill": "on",
            "attention_split_kv": "off",
            "env": {"OMLX_M5_DISABLE_MTP": "1"},
            "warmups": warmups,
            "repeats": repeats,
        },
    ]


def build_dim_b_specs(
    warmups: int, repeats: int, m5rt_tag: str = "m5rt11"
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for mk in ("27b-mxfp4", "27b-oq4e", "35b-a3b-mxfp4"):
        prefix = mk.replace("-", "_")
        specs.extend(
            [
                {
                    "name": f"{prefix}_omlx_ar",
                    "engine": "omlx",
                    "model_key": mk,
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": "coding",
                    "max_new_tokens": 256,
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"{prefix}_{m5rt_tag}_coupled_ar",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": mk,
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": "coding",
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "env": {"OMLX_M5_DISABLE_MTP": "1"},
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"{prefix}_{m5rt_tag}_t_domain_ar",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": mk,
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": "coding",
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "env": {
                        "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                        "OMLX_M5_DISABLE_MTP": "1",
                    },
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
            ]
        )
        if mk in ("27b-mxfp4", "27b-oq4e"):
            rowmajor_art = (
                ".cache/b0/artifacts/qwen38-27b-mlx-mxfp4.omlxm5"
                if mk == "27b-mxfp4"
                else ".cache/b0/artifacts/qwen38-27b-oq4e.omlxm5"
            )
            specs.append(
                {
                    "name": f"{prefix}_{m5rt_tag}_t_domain_ar_rowmajor",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": mk,
                    "artifact_override": rowmajor_art,
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": "coding",
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "env": {
                        "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                        "OMLX_M5_DISABLE_MTP": "1",
                    },
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                }
            )
    return specs


def build_dim_c_specs(
    warmups: int, repeats: int, m5rt_tag: str = "m5rt11"
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for prompt_name in ("coding", "reasoning"):
        # 27B-MXFP4: oMLX AR, oMLX DFlash2 (records upstream UE8M0 bug), m5rt AR, m5rt DFlash2
        specs.extend(
            [
                {
                    "name": f"27b_mxfp4_{prompt_name}_omlx_ar",
                    "engine": "omlx",
                    "model_key": "27b-mxfp4",
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_mxfp4_{prompt_name}_omlx_dflash2",
                    "engine": "omlx",
                    "model_key": "27b-mxfp4",
                    "mode": "dflash2",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_mxfp4_{prompt_name}_{m5rt_tag}_ar",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": "27b-mxfp4",
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "env": {"OMLX_M5_DISABLE_MTP": "1"},
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_mxfp4_{prompt_name}_{m5rt_tag}_dflash2",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": "27b-mxfp4",
                    "mode": "dflash2",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "spec_search": "off",
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
            ]
        )
        # 27B-oQ4e-mtp: oMLX AR, oMLX Native MTP, oMLX DFlash2, m5rt AR, m5rt Native MTP, m5rt DFlash2
        specs.extend(
            [
                {
                    "name": f"27b_oq4e_{prompt_name}_omlx_ar",
                    "engine": "omlx",
                    "model_key": "27b-oq4e",
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_oq4e_{prompt_name}_omlx_native_mtp",
                    "engine": "omlx",
                    "model_key": "27b-oq4e",
                    "mode": "mtp",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_oq4e_{prompt_name}_omlx_dflash2",
                    "engine": "omlx",
                    "model_key": "27b-oq4e",
                    "mode": "dflash2",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_oq4e_{prompt_name}_{m5rt_tag}_ar",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": "27b-oq4e",
                    "mode": "ar",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "env": {"OMLX_M5_DISABLE_MTP": "1"},
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_oq4e_{prompt_name}_{m5rt_tag}_native_mtp",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": "27b-oq4e",
                    "mode": "mtp",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
                {
                    "name": f"27b_oq4e_{prompt_name}_{m5rt_tag}_dflash2",
                    "engine": "m5rt",
                    "m5rt_tag": m5rt_tag,
                    "model_key": "27b-oq4e",
                    "mode": "dflash2",
                    "task_kind": "single_decode",
                    "prompt_name": prompt_name,
                    "max_new_tokens": 256,
                    "sampling_contract": "position_coupled_v1",
                    "attention_split_kv": "on",
                    "spec_search": "off",
                    "capture_tokens": True,
                    "warmups": warmups,
                    "repeats": repeats,
                },
            ]
        )
    return specs


def build_dim_d_specs(
    warmups: int, repeats: int, m5rt_tag: str = "m5rt11"
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    # 27B-MXFP4 (B = 1, 2, 4)
    specs.append(
        {
            "name": "27b_mxfp4_omlx_batched_ar",
            "engine": "omlx",
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "task_kind": "batch_sweep",
            "batch_sizes": [1, 2, 4],
            "max_num_seqs": 16,
            "max_new_tokens": 128,
            "warmups": warmups,
            "repeats": repeats,
        }
    )
    for b in (1, 2, 4):
        for mode in ("ar", "dflash2"):
            s_entry: dict[str, Any] = {
                "name": f"27b_mxfp4_{m5rt_tag}_b{b}_{mode}",
                "engine": "m5rt",
                "m5rt_tag": m5rt_tag,
                "model_key": "27b-mxfp4",
                "mode": mode,
                "task_kind": "batch_single",
                "batch_size": b,
                "max_concurrency": b,
                "max_context": 32768,
                "max_new_tokens": 128,
                "warmups": warmups,
                "repeats": repeats,
            }
            if mode == "ar":
                s_entry["env"] = {
                    "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                    "OMLX_M5_DISABLE_MTP": "1",
                }
            specs.append(s_entry)

    # 27B-oQ4e (B = 1, 2, 4): oMLX AR, oMLX MTP, m5rt AR, m5rt Native MTP, m5rt DFlash2
    for omlx_mode in ("ar", "mtp"):
        specs.append(
            {
                "name": f"27b_oq4e_omlx_batched_{omlx_mode}",
                "engine": "omlx",
                "model_key": "27b-oq4e",
                "mode": omlx_mode,
                "task_kind": "batch_sweep",
                "batch_sizes": [1, 2, 4],
                "max_num_seqs": 16,
                "max_new_tokens": 128,
                "warmups": warmups,
                "repeats": repeats,
            }
        )
    for b in (1, 2, 4):
        for mode in ("ar", "mtp", "dflash2"):
            s_entry = {
                "name": f"27b_oq4e_{m5rt_tag}_b{b}_{mode}",
                "engine": "m5rt",
                "m5rt_tag": m5rt_tag,
                "model_key": "27b-oq4e",
                "mode": mode,
                "task_kind": "batch_single",
                "batch_size": b,
                "max_concurrency": b,
                "max_context": 32768,
                "max_new_tokens": 128,
                "warmups": warmups,
                "repeats": repeats,
            }
            if mode == "ar":
                s_entry["env"] = {
                    "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                    "OMLX_M5_DISABLE_MTP": "1",
                }
            specs.append(s_entry)

    # 35B-A3B-MXFP4 MoE (B = 1, 2, 4, 8)
    specs.append(
        {
            "name": "35b_a3b_mxfp4_omlx_batched_ar",
            "engine": "omlx",
            "model_key": "35b-a3b-mxfp4",
            "mode": "ar",
            "task_kind": "batch_sweep",
            "batch_sizes": [1, 2, 4, 8],
            "max_num_seqs": 16,
            "max_new_tokens": 128,
            "warmups": warmups,
            "repeats": repeats,
        }
    )
    for b in (1, 2, 4, 8):
        specs.append(
            {
                "name": f"35b_a3b_mxfp4_{m5rt_tag}_b{b}_ar",
                "engine": "m5rt",
                "m5rt_tag": m5rt_tag,
                "model_key": "35b-a3b-mxfp4",
                "mode": "ar",
                "task_kind": "batch_single",
                "batch_size": b,
                "max_concurrency": b,
                "max_context": 65536 if b == 8 else 32768,
                "max_new_tokens": 128,
                "env": {
                    "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                    "OMLX_M5_DISABLE_MTP": "1",
                },
                "warmups": warmups,
                "repeats": repeats,
            }
        )
    return specs


def build_dim_e_specs(
    warmups: int, repeats: int, m5rt_tag: str = "m5rt11"
) -> list[dict[str, Any]]:
    depths = [
        {"prefix_len": 2048, "tail_len": 32},    # ~2K
        {"prefix_len": 8192, "tail_len": 32},    # ~8K
        {"prefix_len": 31744, "tail_len": 32},   # ~32K
        {"prefix_len": 64512, "tail_len": 32},   # ~64K
    ]
    footprint_lengths = [2048, 8192, 32768, 64512]
    return [
        # 1) Uncontaminated physical footprint & cold TTFT across 2K -> 8K -> 32K -> 64K (prefix cache OFF)
        {
            "name": "27b_mxfp4_omlx_bf16_kv_footprint",
            "engine": "omlx",
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_mode": "bf16",
            "use_prefix_cache": False,
            "task_kind": "longctx_uncontaminated_footprint",
            "context_lengths": footprint_lengths,
            "max_new_tokens": 64,
        },
        {
            "name": "27b_mxfp4_omlx_affine4_kv_footprint",
            "engine": "omlx",
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_mode": "affine4",
            "use_prefix_cache": False,
            "task_kind": "longctx_uncontaminated_footprint",
            "context_lengths": footprint_lengths,
            "max_new_tokens": 64,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_bf16_kv_footprint",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_dtype": "bf16",
            "max_context": 65536,
            "max_concurrency": 1,
            "flash_nax_prefill": "on",
            "env": {
                "OMLX_M5_DISABLE_MTP": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
            },
            "prefix_cache_bytes": 0,
            "task_kind": "longctx_uncontaminated_footprint",
            "context_lengths": footprint_lengths,
            "max_new_tokens": 64,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_foscar_affine4_kv_footprint",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_dtype": "affine4_g128",
            "max_context": 65536,
            "max_concurrency": 1,
            "flash_nax_prefill": "on",
            "env": {
                "OMLX_M5_DISABLE_MTP": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
            },
            "prefix_cache_bytes": 0,
            "task_kind": "longctx_uncontaminated_footprint",
            "context_lengths": footprint_lengths,
            "max_new_tokens": 64,
        },
        # 2) Multi-repeat (warmups>=2, repeats>=5) KV-depth decode scaling (2K -> 8K -> 32K -> 64K)
        {
            "name": "27b_mxfp4_omlx_bf16_kv_decode_scaling",
            "engine": "omlx",
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_mode": "bf16",
            "use_prefix_cache": True,
            "task_kind": "longctx_decode_scaling",
            "depths": depths,
            "max_new_tokens": 64,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": "27b_mxfp4_omlx_affine4_kv_decode_scaling",
            "engine": "omlx",
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_mode": "affine4",
            "use_prefix_cache": True,
            "task_kind": "longctx_decode_scaling",
            "depths": depths,
            "max_new_tokens": 64,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_bf16_kv_ar_decode_scaling",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_dtype": "bf16",
            "max_context": 65536,
            "max_concurrency": 1,
            "flash_nax_prefill": "on",
            "env": {
                "OMLX_M5_DISABLE_MTP": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
            },
            "prefix_cache_bytes": 12 * (1 << 30),
            "task_kind": "longctx_decode_scaling",
            "depths": depths,
            "max_new_tokens": 64,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_bf16_kv_dflash2_decode_scaling",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "dflash2",
            "kv_dtype": "bf16",
            "max_context": 65536,
            "max_concurrency": 1,
            "flash_nax_prefill": "on",
            "env": {},
            "prefix_cache_bytes": 12 * (1 << 30),
            "task_kind": "longctx_decode_scaling",
            "depths": depths,
            "max_new_tokens": 64,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_foscar_affine4_ar_decode_scaling",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "kv_dtype": "affine4_g128",
            "max_context": 65536,
            "max_concurrency": 1,
            "flash_nax_prefill": "on",
            "env": {
                "OMLX_M5_DISABLE_MTP": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
            },
            "prefix_cache_bytes": 4 * (1 << 30),
            "task_kind": "longctx_decode_scaling",
            "depths": depths,
            "max_new_tokens": 64,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_foscar_affine4_dflash2_decode_scaling",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "dflash2",
            "kv_dtype": "affine4_g128",
            "max_context": 65536,
            "max_concurrency": 1,
            "flash_nax_prefill": "on",
            "env": {},
            "prefix_cache_bytes": 4 * (1 << 30),
            "task_kind": "longctx_decode_scaling",
            "depths": depths,
            "max_new_tokens": 64,
            "warmups": warmups,
            "repeats": repeats,
        },
        # 3) Prefix Cache Warm Hit (B=1 & B=4 shared 2048-token prefix + 32-token tail)
        {
            "name": "27b_mxfp4_omlx_prefix_cache_b1_and_b4",
            "engine": "omlx",
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "use_prefix_cache": True,
            "max_num_seqs": 16,
            "task_kind": "prefix_cache_bench",
            "prefix_len": 2048,
            "tail_len": 32,
            "max_new_tokens": 16,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_prefix_cache_b1",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "max_context": 32768,
            "max_concurrency": 1,
            "flash_nax_prefill": "on",
            "env": {
                "OMLX_M5_DISABLE_MTP": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
            },
            "prefix_cache_bytes": 4 * (1 << 30),
            "task_kind": "prefix_cache_bench",
            "batch_size": 1,
            "prefix_len": 2048,
            "tail_len": 32,
            "max_new_tokens": 16,
            "warmups": warmups,
            "repeats": repeats,
        },
        {
            "name": f"27b_mxfp4_{m5rt_tag}_prefix_cache_b4_cow",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "model_key": "27b-mxfp4",
            "mode": "ar",
            "max_context": 32768,
            "max_concurrency": 4,
            "flash_nax_prefill": "on",
            "env": {
                "OMLX_M5_DISABLE_MTP": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
            },
            "prefix_cache_bytes": 4 * (1 << 30),
            "task_kind": "prefix_cache_bench",
            "batch_size": 4,
            "prefix_len": 2048,
            "tail_len": 32,
            "max_new_tokens": 16,
            "warmups": warmups,
            "repeats": repeats,
        },
    ]


def build_multiturn_specs(
    m5rt_tag: str = "m5rt16",
    *,
    warmups: int = 1,
    repeats: int = 3,
) -> list[dict[str, Any]]:
    """Build all 10 clean-room specs for Multi-Turn Short-Chat (3-way: Stock oMLX vs m5rt.15 vs m5rt.16)."""
    cache_budget_bytes = 4 * (1 << 30)
    b1_common = {
        "model_key": "27b-mxfp4",
        "task_kind": "multiturn_b1",
        "system_prompt_tokens": MULTITURN_B1_SYSTEM_LEN,
        "user_lens": list(MULTITURN_B1_USER_LENS),
        "gen_lens": list(MULTITURN_B1_GEN_LENS),
        "prefix_cache_bytes": cache_budget_bytes,
        "warmups": int(warmups),
        "repeats": int(repeats),
    }
    b4_common = {
        "model_key": "27b-mxfp4",
        "task_kind": "multiturn_b4",
        "batch_size": 4,
        "system_prompt_tokens": MULTITURN_B4_SYSTEM_LEN,
        "user_lens": list(MULTITURN_B4_USER_LENS),
        "gen_len": MULTITURN_B4_GEN_LEN,
        "prefix_cache_bytes": cache_budget_bytes,
        "warmups": int(warmups),
        "repeats": int(repeats),
    }
    return [
        # Workload 1: B=1 10-Turn Sequential Short Chat w/ Pinned 256-Token System Prompt
        {
            **b1_common,
            "name": "27b_mxfp4_omlx_bf16_multiturn_b1",
            "engine": "omlx",
            "mode": "ar",
            "kv_mode": "bf16",
            "use_prefix_cache": True,
            "max_num_seqs": 16,
        },
        {
            **b1_common,
            "name": "27b_mxfp4_omlx_affine4_multiturn_b1",
            "engine": "omlx",
            "mode": "ar",
            "kv_mode": "affine4",
            "use_prefix_cache": True,
            "max_num_seqs": 16,
        },
        {
            **b1_common,
            "name": "27b_mxfp4_m5rt15_affine4_ar_multiturn_b1",
            "engine": "m5rt",
            "m5rt_tag": "m5rt15",
            "mode": "ar",
            "kv_dtype": "affine4_g128",
            "max_context": 32768,
            "max_concurrency": 1,
            "sampling_contract": "position_coupled_v1",
            "flash_nax_prefill": "on",
            "attention_split_kv": "on",
            "rolling_prefix_cache": "off",
            "env": {
                "OMLX_M5_ROLLING_PREFIX_CACHE": "0",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                "OMLX_M5_DISABLE_MTP": "1",
            },
        },
        {
            **b1_common,
            "name": f"27b_mxfp4_{m5rt_tag}_bf16_ar_multiturn_b1",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "mode": "ar",
            "kv_dtype": "bf16",
            "max_context": 32768,
            "max_concurrency": 1,
            "sampling_contract": "position_coupled_v1",
            "flash_nax_prefill": "on",
            "attention_split_kv": "on",
            "rolling_prefix_cache": "on",
            "env": {
                "OMLX_M5_ROLLING_PREFIX_CACHE": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                "OMLX_M5_DISABLE_MTP": "1",
            },
        },
        {
            **b1_common,
            "name": f"27b_mxfp4_{m5rt_tag}_foscar_affine4_ar_multiturn_b1",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "mode": "ar",
            "kv_dtype": "affine4_g128",
            "max_context": 32768,
            "max_concurrency": 1,
            "sampling_contract": "position_coupled_v1",
            "flash_nax_prefill": "on",
            "attention_split_kv": "on",
            "rolling_prefix_cache": "on",
            "env": {
                "OMLX_M5_ROLLING_PREFIX_CACHE": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                "OMLX_M5_DISABLE_MTP": "1",
            },
        },
        {
            **b1_common,
            "name": f"27b_mxfp4_{m5rt_tag}_foscar_affine4_dflash2_multiturn_b1",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "mode": "dflash2",
            "kv_dtype": "affine4_g128",
            "max_context": 32768,
            "max_concurrency": 1,
            "sampling_contract": "position_coupled_v1",
            "flash_nax_prefill": "on",
            "attention_split_kv": "on",
            "spec_search": "off",
            "rolling_prefix_cache": "on",
            "env": {
                "OMLX_M5_ROLLING_PREFIX_CACHE": "1",
            },
        },
        # Workload 2: B=4 Concurrent 4-Session x 6-Turn Short Chat w/ Shared 512-Token System Prompt
        {
            **b4_common,
            "name": "27b_mxfp4_omlx_bf16_multiturn_b4",
            "engine": "omlx",
            "mode": "ar",
            "kv_mode": "bf16",
            "use_prefix_cache": True,
            "max_num_seqs": 16,
        },
        {
            **b4_common,
            "name": "27b_mxfp4_omlx_affine4_multiturn_b4",
            "engine": "omlx",
            "mode": "ar",
            "kv_mode": "affine4",
            "use_prefix_cache": True,
            "max_num_seqs": 16,
        },
        {
            **b4_common,
            "name": f"27b_mxfp4_{m5rt_tag}_foscar_affine4_ar_multiturn_b4",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "mode": "ar",
            "kv_dtype": "affine4_g128",
            "max_context": 32768,
            "max_concurrency": 4,
            "sampling_contract": "position_coupled_v1",
            "flash_nax_prefill": "on",
            "attention_split_kv": "on",
            "rolling_prefix_cache": "on",
            "env": {
                "OMLX_M5_ROLLING_PREFIX_CACHE": "1",
                "OMLX_M5_T_DOMAIN_DISPATCH": "1",
                "OMLX_M5_DISABLE_MTP": "1",
            },
        },
        {
            **b4_common,
            "name": f"27b_mxfp4_{m5rt_tag}_foscar_affine4_dflash2_multiturn_b4",
            "engine": "m5rt",
            "m5rt_tag": m5rt_tag,
            "mode": "dflash2",
            "kv_dtype": "affine4_g128",
            "max_context": 32768,
            "max_concurrency": 4,
            "sampling_contract": "position_coupled_v1",
            "flash_nax_prefill": "on",
            "attention_split_kv": "on",
            "spec_search": "off",
            "rolling_prefix_cache": "on",
            "env": {
                "OMLX_M5_ROLLING_PREFIX_CACHE": "1",
            },
        },
    ]


SUITE_SUFFIXES = {
    "dim_b": ("dim-b-ar-decode.json", build_dim_b_specs),
    "dim_c": ("dim-c-spec-decode.json", build_dim_c_specs),
    "dim_d": ("dim-d-batched-decode.json", build_dim_d_specs),
    "dim_a": ("dim-a-prefill.json", build_dim_a_specs),
    "dim_e": ("dim-e-longctx-mem-prefix.json", build_dim_e_specs),
    "multiturn": (
        "multiturn-short-chat.json",
        lambda w, r, tag: build_multiturn_specs(tag, warmups=w, repeats=r),
    ),
}


def _extract_first_run_tokens(entry: dict[str, Any] | None) -> list[int] | None:
    if not entry or entry.get("status") != "ok":
        return None
    results = entry.get("results") or []
    if not results:
        return None
    runs = results[0].get("runs") or []
    if not runs:
        return None
    toks = runs[0].get("tokens")
    return [int(x) for x in toks] if isinstance(toks, list) else None


def compute_greedy_parity_checks(
    entries: list[dict[str, Any]], m5rt_tag: str
) -> list[dict[str, Any]]:
    by_name = {e.get("spec_name"): e for e in entries if e.get("spec_name")}
    checks: list[dict[str, Any]] = []
    for prompt_name in ("coding", "reasoning"):
        for mk_prefix, cand_modes in (
            ("27b_mxfp4", ("dflash2",)),
            ("27b_oq4e", ("native_mtp", "dflash2")),
        ):
            ar_name = f"{mk_prefix}_{prompt_name}_{m5rt_tag}_ar"
            ar_toks = _extract_first_run_tokens(by_name.get(ar_name))
            if ar_toks is None:
                continue
            for cm in cand_modes:
                cand_name = f"{mk_prefix}_{prompt_name}_{m5rt_tag}_{cm}"
                cand_toks = _extract_first_run_tokens(by_name.get(cand_name))
                if cand_toks is None:
                    continue
                n = min(len(ar_toks), len(cand_toks))
                prefix_match = 0
                for i in range(n):
                    if ar_toks[i] == cand_toks[i]:
                        prefix_match += 1
                    else:
                        break
                exact = ar_toks == cand_toks
                checks.append(
                    {
                        "reference_spec": ar_name,
                        "candidate_spec": cand_name,
                        "reference_tokens": len(ar_toks),
                        "candidate_tokens": len(cand_toks),
                        "matching_prefix_tokens": prefix_match,
                        "exact_0ulp_greedy_match": exact,
                    }
                )

    # Multi-turn greedy trajectory parity (m5rt15 baseline vs m5rt16 rolling AR & DFlash2)
    def _extract_mt_hashes(entry: dict[str, Any] | None) -> list[str] | None:
        if not entry or entry.get("status") != "ok":
            return None
        results = entry.get("results") or []
        if not results:
            return None
        hashes = results[0].get("first_session_turn_output_sha256s")
        return [str(h) for h in hashes] if isinstance(hashes, list) and hashes else None

    mt_ref_name = "27b_mxfp4_m5rt15_affine4_ar_multiturn_b1"
    mt_ref_hashes = _extract_mt_hashes(by_name.get(mt_ref_name))
    if mt_ref_hashes is not None:
        for cand_name in (
            f"27b_mxfp4_{m5rt_tag}_foscar_affine4_ar_multiturn_b1",
            f"27b_mxfp4_{m5rt_tag}_foscar_affine4_dflash2_multiturn_b1",
        ):
            cand_hashes = _extract_mt_hashes(by_name.get(cand_name))
            if cand_hashes is None:
                continue
            matching_turns = 0
            for rh, ch in zip(mt_ref_hashes, cand_hashes):
                if rh == ch:
                    matching_turns += 1
                else:
                    break
            checks.append(
                {
                    "reference_spec": mt_ref_name,
                    "candidate_spec": cand_name,
                    "reference_turns": len(mt_ref_hashes),
                    "candidate_turns": len(cand_hashes),
                    "matching_prefix_turns": matching_turns,
                    "exact_multiturn_greedy_match": mt_ref_hashes == cand_hashes,
                }
            )
    return checks


def run_worker_entrypoint(spec_path: Path, out_path: Path) -> int:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    try:
        if spec["engine"] == "omlx":
            result = asyncio.run(run_omlx_worker_async(spec))
        elif spec["engine"] == "m5rt":
            result = run_m5rt_worker(spec)
        else:
            raise ValueError(f"Unknown engine: {spec['engine']}")
    except Exception as exc:
        result = {
            "status": "error",
            "spec_name": spec.get("name"),
            "model_key": spec.get("model_key"),
            "engine": spec.get("engine"),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    atomic_json(out_path, result)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-spec", type=Path, default=None)
    parser.add_argument("--worker-out", type=Path, default=None)
    parser.add_argument(
        "--suite",
        choices=("dim_a", "dim_b", "dim_c", "dim_d", "dim_e", "multiturn", "all"),
        default="all",
    )
    parser.add_argument(
        "--engines",
        choices=("all", "m5rt", "omlx"),
        default="all",
        help="Which engine family to run ('m5rt' carries forward Stock oMLX entries from --baseline-prefix when present).",
    )
    parser.add_argument(
        "--m5rt-tag",
        type=str,
        default="m5rt11",
        help="Version tag for m5rt spec names & schema (e.g. m5rt11).",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help="Output filename prefix (defaults to 'omlx-vs-<m5rt-tag>', e.g. 'omlx-vs-m5rt11').",
    )
    parser.add_argument(
        "--baseline-prefix",
        type=str,
        default="omlx-vs-m5rt10",
        help="Baseline JSON prefix to carry forward clean-room Stock oMLX & prior m5rt reference entries from when --engines m5rt is used.",
    )
    parser.add_argument(
        "--m5rt-build-dir",
        type=Path,
        default=None,
        help="CMake build directory containing compiled m5rt targets (defaults to OMLX_M5RT_BUILD_DIR or build/b6-xcode27).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "bench/results/m5-ultra-64gpu-96g/h2",
    )
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--filter", type=str, default=None)
    args = parser.parse_args()

    if args.worker_spec is not None:
        if args.worker_out is None:
            raise ValueError("--worker-out is required with --worker-spec")
        return run_worker_entrypoint(args.worker_spec, args.worker_out)

    sys.path.insert(0, str(_SCRIPT_DIR))
    from m5_target_guard import hardware_facts

    hw = hardware_facts()
    suite_keys = list(SUITE_SUFFIXES.keys()) if args.suite == "all" else [args.suite]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = args.out_prefix or f"omlx-vs-{args.m5rt_tag}"
    m5rt_build_dir_str = (
        str(args.m5rt_build_dir.resolve()) if args.m5rt_build_dir is not None else None
    )

    for sk in suite_keys:
        suffix, builder = SUITE_SUFFIXES[sk]
        out_name = f"{out_prefix}-{suffix}"
        all_specs = builder(args.warmups, args.repeats, args.m5rt_tag)
        for s in all_specs:
            if s["engine"] == "m5rt" and m5rt_build_dir_str is not None:
                s["m5rt_build_dir"] = m5rt_build_dir_str

        specs = [
            s
            for s in all_specs
            if args.engines == "all" or s["engine"] == args.engines
        ]
        if args.filter:
            specs = [s for s in specs if args.filter in s["name"]]

        out_path = out_dir / out_name
        existing_entries: dict[str, dict[str, Any]] = {}
        baseline_m5rt_entries: list[dict[str, Any]] = []

        # Carry forward clean-room Stock oMLX entries (and prior m5rt4 reference) when running m5rt-only
        if args.engines == "m5rt" and args.baseline_prefix:
            base_path = out_dir / f"{args.baseline_prefix}-{suffix}"
            if not base_path.exists() and args.baseline_prefix == "omlx-vs-m5rt10":
                base_path = out_dir / f"omlx-vs-m5rt6-{suffix}"
            if base_path.exists() and base_path.resolve() != out_path.resolve():
                base_doc = json.loads(base_path.read_text(encoding="utf-8"))
                for item in base_doc.get("benchmarks", []):
                    sname = str(item.get("spec_name", ""))
                    b_id = str((item.get("backend_identity") or {}).get("backend", ""))
                    if "_omlx_" in sname or b_id == "stock_omlx_mlx_metal":
                        existing_entries[sname] = item
                    elif sname:
                        baseline_m5rt_entries.append(item)

        if (args.filter or args.engines != "all") and out_path.exists():
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            for item in prev.get("benchmarks", []):
                existing_entries[item["spec_name"]] = item

        started_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        t_suite0 = time.perf_counter()
        entries: list[dict[str, Any]] = []
        for s in specs:
            res = execute_worker_subprocess(s)
            existing_entries[s["name"]] = res
            entries.append(res)

        if args.filter or args.engines != "all":
            ordered_names = [s["name"] for s in all_specs if s["name"] in existing_entries]
            for extra_k in existing_entries:
                if extra_k not in ordered_names and f"_{args.m5rt_tag}_" not in extra_k:
                    ordered_names.append(extra_k)
            merged = [existing_entries[k] for k in ordered_names]
        else:
            merged = entries

        parity_checks = compute_greedy_parity_checks(merged, args.m5rt_tag)
        suite_doc: dict[str, Any] = {
            "schema": f"omlx-vs-{args.m5rt_tag}-comparative-v1",
            "suite": sk,
            "m5rt_tag": args.m5rt_tag,
            "engines_executed": args.engines,
            "started_at_utc": started_iso,
            "elapsed_seconds": round(time.perf_counter() - t_suite0, 2),
            "host_os": platform.platform(),
            "hardware": hw,
            "warmups": args.warmups,
            "repeats": args.repeats,
            "benchmarks": merged,
        }
        if parity_checks:
            suite_doc["greedy_parity_checks"] = parity_checks
        if baseline_m5rt_entries:
            suite_doc["baseline_m5rt_reference"] = baseline_m5rt_entries
        atomic_json(out_path, suite_doc)
        print(f"=== Saved {out_path} ({len(merged)} entries) ===", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
