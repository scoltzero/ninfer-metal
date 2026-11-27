#!/usr/bin/env python3
"""Reusable M5 Ultra 16K-256K Prefill Matrix, NIAH, PPL & CacheBlend RAG Harness.

Drives `ninfer-metal-serve` on an isolated ephemeral port (default 18090) — or
an optional `mseries_backendd` AF_UNIX socket / C++ `omlx_m5_prefill_matrix_bench`
helper — using ONLY Python 3.9+ stdlib so it runs identically on macOS host
Python 3.9.6 (`scoltc@192.168.10.245`) and local `.venv` Python.

Safety invariants:
  - Never touches, stops, or disturbs production `ninfer-metal-serve` on :8080.
  - Verifies :8080 (if present) is idle (<5% CPU) before starting timed trials.
  - Cleans up its own ephemeral server on exit (SIGTERM -> wait -> SIGKILL).
  - Injects a unique nonce prefix on every cold-prefill trial and asserts
    `cached_tokens == 0` so prefix/OS caches never fake a cold hit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import http.client
import json
import math
import os
import platform
import re
import signal
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Qwen3.8 ChatML special token IDs (248,320 vocab; enable_thinking=False framing).
QWEN38_IM_START = 248045
QWEN38_IM_END = 248046
QWEN38_THINK_OPEN = 248068
QWEN38_THINK_CLOSE = 248069
QWEN38_ROLE_USER = 846
QWEN38_ROLE_ASSISTANT = 74455
QWEN38_NL = 198
QWEN38_NL2 = 271

QWEN38_USER_OPEN: List[int] = [QWEN38_IM_START, QWEN38_ROLE_USER, QWEN38_NL]
QWEN38_USER_TO_ASSISTANT_NOTHINK: List[int] = [
    QWEN38_IM_END,
    QWEN38_NL,
    QWEN38_IM_START,
    QWEN38_ROLE_ASSISTANT,
    QWEN38_NL,
    QWEN38_THINK_OPEN,
    QWEN38_NL2,
    QWEN38_THINK_CLOSE,
    QWEN38_NL2,
]

# Full Module 5 matrix preset definitions (Spec Section 5.2 + Wave 3 ANE pairs).
MATRIX_CELLS: Dict[str, Dict[str, Any]] = {
    "M0": {
        "label": "Lossless Base Default (BF16 + Exact NAX, ANE=OFF)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 0.0,
    },
    "M0_ANE": {
        "label": "Lossless Base + Dual-ANE (BF16 + Exact NAX, ANE=ON)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M1": {
        "label": "Route A (W4A8 + Exact NAX, ANE=OFF)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 0.0,
    },
    "M1_ANE": {
        "label": "Route A + Dual-ANE (W4A8 + Exact NAX, ANE=ON)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M2a": {
        "label": "Route B1 (BF16 + Sage2, MInf=OFF, ANE=OFF)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 0.0,
    },
    "M2a_ANE": {
        "label": "Route B1 + Dual-ANE (BF16 + Sage2, MInf=OFF, ANE=ON)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M2b": {
        "label": "Route B2 (BF16 + Sage2 + MInference, ANE=OFF)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 0.0,
    },
    "M2b_ANE": {
        "label": "Route B2 + Dual-ANE (BF16 + Sage2 + MInference, ANE=ON)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M3a": {
        "label": "Route A+B1 (W4A8 + Sage2, MInf=OFF, ANE=OFF)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 0.0,
    },
    "M3a_ANE": {
        "label": "Route A+B1 + Dual-ANE (W4A8 + Sage2, MInf=OFF, ANE=ON)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M3": {
        "label": "Route A+B2 (W4A8 + Sage2 + MInference, ANE=OFF)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 0.0,
    },
    "M3b": {
        "label": "Route A+B2 (W4A8 + Sage2 + MInference, ANE=OFF)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 0.0,
    },
    "M3b_ANE": {
        "label": "Route A+B2 + Dual-ANE (W4A8 + Sage2 + MInference, ANE=ON)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M4a": {
        "label": "Dual-ANE + BF16",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M4b": {
        "label": "Dual-ANE + W4A8",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M4c": {
        "label": "Dual-ANE + W4A8 + Sage2 + MInference",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefix_cache_gb": 0.0,
    },
    "M5a": {
        "label": "CacheBlend + BF16 + Exact NAX (RAG)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": True,
        "prefill_ane_offload": False,
        "prefix_cache_gb": 24.0,
    },
    "M5b": {
        "label": "All-On Stack (W4A8 + Sage2 + MInf + CacheBlend + ANE)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": True,
        "prefill_ane_offload": True,
        "prefill_ane_mode": "auto",
        "prefix_cache_gb": 24.0,
    },
    # Phase 3 cells: W4A4 (MXFP4) & Ultra FP16 SwiGLU Dual-ANE
    "M0_ULTRA_ANE": {
        "label": "Lossless Base + Ultra FP16 SwiGLU ANE (BF16 + Exact NAX)",
        "prefill_linear_mode": "bf16",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefill_ane_mode": "ultra_fp16_swiglu",
        "prefix_cache_gb": 0.0,
    },
    "M1_ULTRA_ANE": {
        "label": "Route A (W4A8) + Ultra FP16 SwiGLU ANE",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefill_ane_mode": "ultra_fp16_swiglu",
        "prefix_cache_gb": 0.0,
    },
    "M1_A4": {
        "label": "Route A4 (W4A4 + Exact NAX, ANE=OFF)",
        "prefill_linear_mode": "w4a4",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefill_ane_mode": "off",
        "prefix_cache_gb": 0.0,
    },
    "M1_A4_ANE": {
        "label": "Route A4 + Ultra FP16 SwiGLU ANE (W4A4 + Exact NAX)",
        "prefill_linear_mode": "w4a4",
        "prefill_attention_mode": "exact_nax",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefill_ane_mode": "ultra_fp16_swiglu",
        "prefix_cache_gb": 0.0,
    },
    "M3a_A4": {
        "label": "Route A4+B1 (W4A4 + Dense Sage2, MInf=OFF, ANE=OFF)",
        "prefill_linear_mode": "w4a4",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": False,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefill_ane_mode": "off",
        "prefix_cache_gb": 0.0,
    },
    "M3b_A4": {
        "label": "Route A4+B2 (W4A4 + Sage2 + MInference, ANE=OFF)",
        "prefill_linear_mode": "w4a4",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": False,
        "prefill_ane_mode": "off",
        "prefix_cache_gb": 0.0,
    },
    "M3b_ULTRA_ANE": {
        "label": "Route A+B2 + Ultra FP16 SwiGLU ANE (W4A8 + Sage2 + MInf)",
        "prefill_linear_mode": "w4a8",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefill_ane_mode": "ultra_fp16_swiglu",
        "prefix_cache_gb": 0.0,
    },
    "M3b_A4_ANE": {
        "label": "Route A4+B2 + Ultra FP16 SwiGLU ANE (W4A4 + Sage2 + MInf)",
        "prefill_linear_mode": "w4a4",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": False,
        "prefill_ane_offload": True,
        "prefill_ane_mode": "ultra_fp16_swiglu",
        "prefix_cache_gb": 0.0,
    },
    "M5b_A4": {
        "label": "All-On Stack A4 (W4A4 + Sage2 + MInf + CacheBlend + Ultra ANE)",
        "prefill_linear_mode": "w4a4",
        "prefill_attention_mode": "sage2",
        "prefill_sparse_minference": True,
        "prefill_cache_blend": True,
        "prefill_ane_offload": True,
        "prefill_ane_mode": "ultra_fp16_swiglu",
        "prefix_cache_gb": 24.0,
    },
}

DEFAULT_CONTEXT_TIERS: List[int] = [16384, 32768, 65536, 131072, 262144]

# High-entropy technical prose & code corpus blocks (deterministic, zero network).
_CORPUS_SECTIONS: List[str] = [
    (
        "### Architecture Note: Unified Memory Paged KV Pools\n"
        "On Apple Silicon unified memory systems, long-context autoregressive transformers "
        "require strict separation between read-only weight buffers and dynamically leased "
        "key-value pages. Using 64-token physical pages with affine 4-bit group-128 quantization "
        "reduces per-token full-attention storage for a 24-query-head, 4-KV-head, 256-dim model "
        "from 65,536 bytes down to 18,496 bytes, keeping a full 262,144-token context under "
        "4.52 GiB across 16 attention layers while preserving copy-on-write prefix sharing.\n"
    ),
    (
        "### Kernel Analysis: Gated DeltaNet Recurrent State Transitions\n"
        "Hybrid sequence models interleave three linear-recurrent Gated DeltaNet layers before "
        "every full-attention block. Each GDN layer maintains a causal depthwise convolution "
        "ring of width four over projected QKV channels together with a float32 associative "
        "matrix state of shape [48, 128, 128]. Because state size is strictly O(1) in context "
        "length, all 48 GDN layers occupy exactly 146.81 MiB regardless of whether the prompt "
        "spans 4,096 or 262,144 tokens.\n"
    ),
    (
        "### Code Sample: Deterministic Checksum & Ring Buffer Indexer\n"
        "```cpp\n"
        "inline std::uint64_t fnv1a_64_span(std::span<const std::uint32_t> tokens) noexcept {\n"
        "  std::uint64_t h = 14695981039346656037ULL;\n"
        "  for (const std::uint32_t tok : tokens) {\n"
        "    h ^= static_cast<std::uint64_t>(tok);\n"
        "    h *= 1099511628211ULL;\n"
        "  }\n"
        "  return h;\n"
        "}\n"
        "```\n"
        "Wait-free ring indices mask logical positions modulo power-of-two capacity so host "
        "parameter patches avoid modulo division inside indirect command buffer replays.\n"
    ),
    (
        "### Numerical Stability: Partial Rotary Position Embeddings\n"
        "When partial_rotary_factor equals 0.25 with head dimension 256, rotary embeddings "
        "modify only dimensions 0 through 63 while dimensions 64 through 255 remain invariant "
        "to sequence offset shifts. Under group-128 KV quantization, this splits cleanly into "
        "a rotated leading group [0..128) and a position-independent trailing group [128..256) "
        "that can be relocated across arbitrary document offsets without dequantization loss.\n"
    ),
    (
        "### Systems Telemetry: Mach Physical Footprint Accounting\n"
        "Measuring resident set size alone under-reports compressed pages and wired Metal "
        "allocations on macOS. Querying TASK_VM_INFO via Mach task_info returns the authoritative "
        "phys_footprint ledger tracked by jetsam and memory pressure monitors. Sampling at "
        "200 ms intervals during chunked prefill captures peak workspace plus KV slab usage.\n"
    ),
    (
        "### Distributed Scheduling: Longest-Processing-Time Causal Tile Dispatch\n"
        "Causal self-attention over long prompts forms a lower-triangular workload where later "
        "query blocks traverse exponentially more key-value pages than leading blocks. Issuing "
        "threadgroups in reverse query-block order schedules the heaviest tiles first across all "
        "80 GPU cores, minimizing tail-wave stragglers without atomic work-stealing overhead.\n"
    ),
    (
        "### Quantization Math: Block-Scaled E2M1 and Asymmetric Affine Group Kernels\n"
        "Microscaling FP4 stores 32-element blocks as 4-bit E2M1 nibbles paired with an 8-bit "
        "UE8M0 power-of-two exponent. Scaling every E2M1 code by two maps all sixteen values "
        "exactly onto signed integers in [-12, +12], enabling lossless unpacking into int8 "
        "registers for hardware int8-by-int8 matrix-multiply-accumulate instructions.\n"
    ),
    (
        "### Networking & Streaming Protocol: Server-Sent Events Latency Measurement\n"
        "To isolate true time-to-first-token from steady-state autoregressive decode throughput, "
        "HTTP benchmark clients record a monotonic timestamp immediately before writing the "
        "JSON completion request, mark t_first upon receiving the first non-empty token delta, "
        "and compute decode tok/s over the remaining N - 1 inter-token intervals.\n"
    ),
]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check_port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0


def inspect_host_processes() -> List[Dict[str, Any]]:
    """Return running ninfer/omlx processes via read-only `ps aux`."""
    out = subprocess.check_output(["ps", "aux"], text=True)
    procs: List[Dict[str, Any]] = []
    for line in out.splitlines()[1:]:
        if not any(k in line for k in ("ninfer-metal-serve", "mseries_backendd", "oMLX")):
            continue
        parts = line.split(None, 10)
        if len(parts) < 11:
            continue
        procs.append(
            {
                "user": parts[0],
                "pid": int(parts[1]),
                "cpu_pct": float(parts[2]),
                "mem_pct": float(parts[3]),
                "rss_kb": int(parts[5]),
                "rss_gib": round(int(parts[5]) / (1024.0 * 1024.0), 3),
                "cmd": parts[10],
            }
        )
    return procs


def verify_port_8080_idle(max_cpu_pct: float = 5.0) -> Dict[str, Any]:
    """Verify production :8080 ninfer-metal-serve (if running) is idle and untouched."""
    procs = inspect_host_processes()
    prod = [p for p in procs if "ninfer-metal-serve" in p["cmd"] and "--port 8080" in p["cmd"]]
    status: Dict[str, Any] = {"present": bool(prod), "processes": prod, "healthy": True}
    for p in prod:
        if p["cpu_pct"] > max_cpu_pct:
            status["healthy"] = False
            raise RuntimeError(
                f"Production :8080 ninfer-metal-serve PID {p['pid']} is busy "
                f"(CPU={p['cpu_pct']}% > {max_cpu_pct}%); aborting to avoid GPU contention"
            )
    if prod:
        try:
            with urllib.request.urlopen("http://127.0.0.1:8080/health", timeout=3.0) as resp:
                h = json.loads(resp.read().decode("utf-8"))
                status["health_status"] = h.get("status")
                status["active_requests"] = h.get("active_requests", 0)
                if h.get("active_requests", 0) != 0:
                    raise RuntimeError(
                        f"Production :8080 has {h.get('active_requests')} active requests"
                    )
        except urllib.error.URLError:
            pass
    return status


def get_process_rss_bytes(pid: int) -> int:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        return int(out) * 1024 if out else 0
    except Exception:
        return 0


def parse_prometheus_metric(metrics_text: str, metric_name: str) -> Optional[float]:
    prefix = metric_name + " "
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            try:
                return float(line[len(prefix) :].strip().split()[0])
            except ValueError:
                return None
    return None


def query_server_phys_footprint(host: str, port: int) -> Optional[int]:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=1.5) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        val = parse_prometheus_metric(body, "ninfer_metal_phys_footprint_bytes")
        return int(val) if val is not None else None
    except Exception:
        return None


PREFILL_TELEMETRY_METRICS: Tuple[Tuple[str, str], ...] = (
    ("w4a8_tokens", "ninfer_metal_prefill_w4a8_tokens_total"),
    ("w4a4_tokens", "ninfer_metal_prefill_w4a4_tokens_total"),
    ("sage2_tokens", "ninfer_metal_prefill_sage2_tokens_total"),
    ("minference_blocks_skipped", "ninfer_metal_prefill_minference_blocks_skipped_total"),
    ("minference_blocks_total", "ninfer_metal_prefill_minference_blocks_total"),
    ("cache_blend_hit_tokens", "ninfer_metal_prefill_cache_blend_hit_tokens_total"),
    ("cache_blend_recomputed_tokens", "ninfer_metal_prefill_cache_blend_recomputed_tokens_total"),
    ("ane_offload_tokens", "ninfer_metal_prefill_ane_offload_tokens_total"),
    ("ane_fallback_tokens", "ninfer_metal_prefill_ane_fallback_tokens_total"),
)


def query_prefill_telemetry(host: str, port: int) -> Dict[str, int]:
    out: Dict[str, int] = {k: 0 for k, _ in PREFILL_TELEMETRY_METRICS}
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=2.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        for key, prom_name in PREFILL_TELEMETRY_METRICS:
            v = parse_prometheus_metric(body, prom_name)
            if v is not None:
                out[key] = int(v)
    except Exception:
        pass
    return out


def diff_prefill_telemetry(before: Dict[str, int], after: Dict[str, int]) -> Dict[str, int]:
    return {k: max(0, after.get(k, 0) - before.get(k, 0)) for k, _ in PREFILL_TELEMETRY_METRICS}


def query_footprint_cli_bytes(pid: int) -> Optional[int]:
    """Best-effort macOS `/usr/bin/footprint --pid` fallback."""
    try:
        out = subprocess.check_output(
            ["/usr/bin/footprint", "--pid", str(pid), "--bytes"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5.0,
        )
        m = re.search(r"phys_footprint:\s*(\d+)\s*B", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


class FootprintSampler:
    """Polls server `/metrics` phys_footprint and `ps` RSS during a request."""

    def __init__(self, host: str, port: int, pid: Optional[int], interval_s: float = 0.2):
        self.host = host
        self.port = port
        self.pid = pid
        self.interval_s = interval_s
        self.peak_phys_bytes = 0
        self.peak_rss_bytes = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _poll_once(self) -> None:
        phys = query_server_phys_footprint(self.host, self.port)
        if phys is not None and phys > self.peak_phys_bytes:
            self.peak_phys_bytes = phys
        if self.pid is not None:
            rss = get_process_rss_bytes(self.pid)
            if rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.is_set():
            self._poll_once()
            self._stop.wait(self.interval_s)

    def start(self) -> None:
        self._poll_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> Dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._poll_once()
        return {
            "peak_phys_footprint_bytes": self.peak_phys_bytes,
            "peak_phys_footprint_gib": round(self.peak_phys_bytes / float(1 << 30), 4),
            "peak_rss_bytes": self.peak_rss_bytes,
            "peak_rss_gib": round(self.peak_rss_bytes / float(1 << 30), 4),
            "samples": self.samples,
        }


class TokenizerBridge:
    """Exact Qwen3.8 tokenizer via C++ `omlx_m5_prefill_matrix_bench` or `tokenizers`."""

    def __init__(self, model_dir: str, cpp_bench_bin: Optional[str] = None):
        self.model_dir = model_dir
        self.cpp_bench_bin = cpp_bench_bin
        self._hf_tok: Any = None
        if not self.cpp_bench_bin:
            try:
                from tokenizers import Tokenizer  # type: ignore

                tok_path = Path(model_dir) / "tokenizer.json"
                if tok_path.is_file():
                    self._hf_tok = Tokenizer.from_file(str(tok_path))
            except Exception:
                self._hf_tok = None
        if not self.cpp_bench_bin and self._hf_tok is None:
            raise RuntimeError(
                "No tokenizer available: pass --cpp-bench-bin PATH or run in a Python env with `tokenizers`"
            )

    def encode(self, text: str) -> List[int]:
        if self._hf_tok is not None:
            return list(self._hf_tok.encode(text, add_special_tokens=False).ids)
        assert self.cpp_bench_bin is not None
        proc = subprocess.run(
            [self.cpp_bench_bin, "--mode", "tokenize", "--model-dir", self.model_dir],
            input=text,
            text=True,
            capture_output=True,
            check=True,
        )
        data = json.loads(proc.stdout)
        return [int(x) for x in data["tokens"]]


def generate_deterministic_corpus_text(min_chars: int, seed_tag: str = "wikitext_code_v1") -> str:
    """Generate deterministic, non-trivial high-entropy technical prose/code corpus."""
    parts: List[str] = []
    total = 0
    idx = 0
    while total < min_chars:
        sec = _CORPUS_SECTIONS[idx % len(_CORPUS_SECTIONS)]
        digest = hashlib.sha256(f"{seed_tag}:{idx}".encode("utf-8")).hexdigest()[:12]
        block = (
            f"\n[Section {idx:04d} | Ref {digest} | Epoch {2026 + (idx % 7)} | "
            f"Tile {(idx * 37) % 512:03d}]\n"
            f"{sec}"
            f"Telemetry snapshot #{idx}: throughput_ratio={1.12 + (idx % 19) * 0.07:.2f}x, "
            f"sram_bytes={16384 + (idx % 13) * 1024}, active_pages={64 + (idx * 29) % 4096}, "
            f"checksum_tag=0x{digest}.\n"
        )
        parts.append(block)
        total += len(block)
        idx += 1
    return "".join(parts)


def build_haystack_pool_tokens(
    tokenizer: TokenizerBridge, target_tokens: int, seed_tag: str = "niah_haystack_v1"
) -> List[int]:
    """Pre-tokenize a deterministic high-entropy haystack pool >= target_tokens."""
    # ~3.5 chars/token conservative estimate; grow if needed.
    est_chars = max(int(target_tokens * 4.8), 64000)
    text = generate_deterministic_corpus_text(est_chars, seed_tag=seed_tag)
    tokens = tokenizer.encode(text)
    extra_round = 0
    while len(tokens) < target_tokens:
        extra_round += 1
        more = generate_deterministic_corpus_text(
            max(int((target_tokens - len(tokens)) * 5.2), 32000),
            seed_tag=f"{seed_tag}_ext{extra_round}",
        )
        tokens.extend(tokenizer.encode(more))
    return tokens[:target_tokens]


def deterministic_passkey(tier: int, depth_pct: int, trial_idx: int) -> str:
    h = hashlib.sha256(f"m5ultra-niah-passkey:{tier}:{depth_pct}:{trial_idx}".encode("utf-8")).hexdigest()
    num = int(h[:8], 16) % 900000 + 100000
    return f"{num:06d}"


def build_exact_niah_prompt_tokens(
    tokenizer: TokenizerBridge,
    haystack_pool: Sequence[int],
    prompt_tokens: int,
    tier: int,
    depth_pct: int,
    trial_idx: int,
    nonce_tag: str,
) -> Tuple[List[int], str]:
    """Build an EXACT `prompt_tokens`-length unique-nonce NIAH + 128-tok continuation prompt."""
    passkey = deterministic_passkey(tier, depth_pct, trial_idx)
    nonce_hex = hashlib.sha256(
        f"{nonce_tag}:tier={tier}:depth={depth_pct}:trial={trial_idx}".encode("utf-8")
    ).hexdigest()[:16]

    header_text = (
        f"[BENCH_NONCE={nonce_hex} CONTEXT_TIER={tier} DEPTH={depth_pct}% TRIAL={trial_idx}]\n"
        "You are reading a long technical archive. Exactly one line in the archive contains "
        "a secret 6-digit verification passkey formatted as `VERIFICATION_PASSKEY: XXXXXX`.\n\n"
    )
    needle_text = (
        f"\n>>> CRITICAL AUDIT RECORD (NONCE {nonce_hex}): "
        f"VERIFICATION_PASSKEY: {passkey} <<<\n"
    )
    footer_text = (
        "\n\n=== END OF ARCHIVE ===\n"
        "Task:\n"
        "1. On the very first line, output ONLY `PASSKEY: <the exact 6-digit number>`.\n"
        "2. Immediately on the next line, write a detailed 200-word technical engineering "
        "summary covering paged affine4 KV pools, Gated DeltaNet recurrent state sizing, "
        "partial RoPE group alignment, and causal LPT attention dispatch.\n"
    )

    header_ids = QWEN38_USER_OPEN + tokenizer.encode(header_text)
    needle_ids = tokenizer.encode(needle_text)
    footer_ids = tokenizer.encode(footer_text) + QWEN38_USER_TO_ASSISTANT_NOTHINK

    overhead = len(header_ids) + len(needle_ids) + len(footer_ids)
    if overhead >= prompt_tokens:
        raise ValueError(f"NIAH prompt overhead ({overhead}) >= prompt_tokens ({prompt_tokens})")

    filler_total = prompt_tokens - overhead
    if filler_total > len(haystack_pool):
        raise ValueError(
            f"Haystack pool ({len(haystack_pool)}) smaller than needed filler ({filler_total})"
        )

    before_len = int(round(filler_total * (depth_pct / 100.0)))
    before_len = max(0, min(filler_total, before_len))
    after_len = filler_total - before_len

    # Shift haystack slice deterministically per trial so prefix/body never accidentally collide.
    shift = (trial_idx * 997 + depth_pct * 31) % max(1, len(haystack_pool) - filler_total + 1)
    before_slice = list(haystack_pool[shift : shift + before_len])
    after_slice = list(haystack_pool[shift + before_len : shift + before_len + after_len])

    full_ids = header_ids + before_slice + needle_ids + after_slice + footer_ids
    assert len(full_ids) == prompt_tokens, f"Expected {prompt_tokens}, got {len(full_ids)}"
    return full_ids, passkey


def build_exact_multi_needle_prompt_tokens(
    tokenizer: TokenizerBridge,
    haystack_pool: Sequence[int],
    prompt_tokens: int,
    tier: int,
    depths: Sequence[int],
    trial_idx: int,
    nonce_tag: str,
) -> Tuple[List[int], Dict[int, str]]:
    """Build an EXACT `prompt_tokens`-length prompt embedding passkeys at all `depths` simultaneously."""
    sorted_depths = sorted(set(int(d) for d in depths))
    passkeys: Dict[int, str] = {
        d: deterministic_passkey(tier, d, trial_idx) for d in sorted_depths
    }
    nonce_hex = hashlib.sha256(
        f"{nonce_tag}:multi:tier={tier}:depths={sorted_depths}:trial={trial_idx}".encode("utf-8")
    ).hexdigest()[:16]

    header_text = (
        f"[BENCH_NONCE={nonce_hex} CONTEXT_TIER={tier} MULTI_DEPTHS={sorted_depths} TRIAL={trial_idx}]\n"
        "You are reading a long technical archive containing three verification passkey records "
        "(labeled DEPTH_10, DEPTH_50, DEPTH_90).\n\n"
    )
    needle_tokens: Dict[int, List[int]] = {}
    for d in sorted_depths:
        nt = (
            f"\n>>> CRITICAL AUDIT RECORD (NONCE {nonce_hex}): "
            f"VERIFICATION_PASSKEY_DEPTH_{d}: {passkeys[d]} <<<\n"
        )
        needle_tokens[d] = tokenizer.encode(nt)

    footer_lines = [
        "\n\n=== END OF ARCHIVE ===\n",
        "Task:\n",
        "1. On the first three lines, output the exact 6-digit passkeys in order:\n",
    ]
    for d in sorted_depths:
        footer_lines.append(f"PASSKEY_{d}: <6-digit code>\n")
    footer_lines.append(
        "2. Immediately afterwards, write a detailed 200-word technical engineering summary "
        "covering paged affine4 KV pools, Gated DeltaNet recurrent state sizing, partial RoPE "
        "group alignment, and causal LPT attention dispatch.\n"
    )
    footer_text = "".join(footer_lines)

    header_ids = QWEN38_USER_OPEN + tokenizer.encode(header_text)
    footer_ids = tokenizer.encode(footer_text) + QWEN38_USER_TO_ASSISTANT_NOTHINK
    total_needle_len = sum(len(v) for v in needle_tokens.values())
    overhead = len(header_ids) + total_needle_len + len(footer_ids)
    if overhead >= prompt_tokens:
        raise ValueError(f"Multi-needle overhead ({overhead}) >= prompt_tokens ({prompt_tokens})")

    filler_total = prompt_tokens - overhead
    if filler_total > len(haystack_pool):
        raise ValueError(
            f"Haystack pool ({len(haystack_pool)}) smaller than needed filler ({filler_total})"
        )
    shift = (trial_idx * 1301 + tier * 7) % max(1, len(haystack_pool) - filler_total + 1)
    filler = list(haystack_pool[shift : shift + filler_total])

    full_ids = list(header_ids)
    prev_cut = 0
    for d in sorted_depths:
        cut = max(prev_cut, min(filler_total, int(round(filler_total * (d / 100.0)))))
        full_ids.extend(filler[prev_cut:cut])
        full_ids.extend(needle_tokens[d])
        prev_cut = cut
    full_ids.extend(filler[prev_cut:])
    full_ids.extend(footer_ids)
    assert len(full_ids) == prompt_tokens, f"Expected {prompt_tokens}, got {len(full_ids)}"
    return full_ids, passkeys


def generate_permuted_rag_dataset(
    tokenizer: TokenizerBridge,
    num_docs: int = 8,
    doc_tokens: int = 4096,
    seed: int = 20261126,
    sub_chunk_tokens: int = 2048,
) -> Dict[str, Any]:
    """Generate deterministic multi-document 4K-chunk RAG QA dataset for CacheBlend testing.

    Each 4K document is cleanly structured into `<= sub_chunk_tokens` (2048) segments
    bounded by Qwen3.8 `<|im_start|>`/`<|im_end|>` structural delimiters so `ChunkBlendCache`
    achieves 100% deterministic chunk boundary hits under arbitrary document permutations.
    """
    docs: List[Dict[str, Any]] = []
    pool = build_haystack_pool_tokens(tokenizer, doc_tokens * 2, seed_tag=f"rag_pool_{seed}")
    for i in range(num_docs):
        code = deterministic_passkey(doc_tokens, i + 1, seed)
        port = 19000 + i * 17
        subsystem = f"SUBSYS-{chr(65 + (i % 26))}{i:02d}"
        head = (
            f'<doc id="D{i:02d}" subsystem="{subsystem}">\n'
            f"Document D{i:02d} Specification: Service {subsystem} listens on TCP port {port} "
            f"and requires release token RAG_TOKEN_{code}.\n"
        )
        tail = f"\nSummary for {subsystem}: port={port}, token=RAG_TOKEN_{code}.\n</doc>\n\n"
        head_ids = tokenizer.encode(head)
        tail_ids = tokenizer.encode(tail)
        # Split into 1 or 2 `<= sub_chunk_tokens` delimiter-bounded chunks.
        n_parts = max(1, (doc_tokens + sub_chunk_tokens - 1) // sub_chunk_tokens)
        part_target = doc_tokens // n_parts
        toks: List[int] = []
        for p_idx in range(n_parts):
            this_target = part_target if p_idx < n_parts - 1 else (doc_tokens - len(toks))
            p_head = [QWEN38_IM_START] + (
                head_ids if p_idx == 0 else tokenizer.encode(f'<doc id="D{i:02d}" part="{p_idx}">\n')
            )
            p_tail = (
                tail_ids if p_idx == n_parts - 1 else tokenizer.encode(f"\n</part>\n")
            ) + [QWEN38_IM_END]
            fill_n = max(64, this_target - len(p_head) - len(p_tail))
            shift = (i * 409 + p_idx * 977) % max(1, len(pool) - fill_n)
            toks.extend(p_head + list(pool[shift : shift + fill_n]) + p_tail)
        docs.append(
            {
                "doc_id": f"D{i:02d}",
                "subsystem": subsystem,
                "port": port,
                "token_code": f"RAG_TOKEN_{code}",
                "num_tokens": len(toks),
                "tokens": toks,
            }
        )

    # Deterministic derangement permutation (every document moves to a new position).
    perm = [(i * 5 + 3) % num_docs for i in range(num_docs)]
    if len(set(perm)) != num_docs or any(perm[i] == i for i in range(num_docs)):
        perm = list(reversed(range(num_docs)))
        if num_docs % 2 == 1 and num_docs >= 3:
            mid = num_docs // 2
            perm[mid], perm[mid + 1] = perm[mid + 1], perm[mid]

    # Deterministic >= 260-token shared system preamble (so segment[0] cleanly closes
    # BEFORE doc 0's <|im_start|> delimiter under kBlendDefaultLMin=256!) and >= 140-token
    # trailing instruction block (so the last document never overlaps kBlendDefaultDenseTail=128).
    sys_preamble_text = (
        "System Retrieval Policy (RAG-BLEND-v1): You are a deterministic technical auditor "
        "analyzing a multi-document engineering corpus delimited by structured document segments. "
        "Each document specifies a unique subsystem identifier, a dedicated TCP service port, "
        "and a cryptographic release token prefixed with RAG_TOKEN_. "
        "When documents are reordered across retrieval turns, preserve exact fidelity to every "
        "subsystem specification regardless of document order or surrounding context length. "
        "Always report exact numeric TCP ports and exact release tokens verbatim without "
        "paraphrasing, rounding, or omitting digits. "
        "Standard audit protocol covers unified memory page tables, affine4 group-128 KV storage, "
        "Gated DeltaNet recurrent state transitions, partial RoPE frequency alignment, "
        "block-sparse attention verification, and cross-chunk KV cache stitching fidelity across "
        "16K, 32K, 64K, 128K, and 256K context windows. "
        "Verify all cross-document references carefully before emitting your final structured response. "
        "Do not invent ports or tokens that are not explicitly recorded in the provided corpus. "
        "Maintain concise, deterministic formatting for automated downstream verification pipelines. "
        "Begin document archive below.\n"
    )
    sys_preamble_ids = list(QWEN38_USER_OPEN) + tokenizer.encode(sys_preamble_text)
    if len(sys_preamble_ids) < 272:
        pad_needed = 272 - len(sys_preamble_ids)
        sys_preamble_ids.extend(list(pool[:pad_needed]))
    sys_preamble_ids.append(QWEN38_IM_END)

    target_a = docs[perm[1]]
    target_b = docs[perm[-2 if num_docs >= 3 else 0]]
    query_text = (
        "\n=== MULTI-HOP VERIFICATION QUERY ===\n"
        "Based strictly on the reordered engineering documents provided above, perform an exact "
        "cross-document extraction and answer concisely following the mandatory output schema.\n"
        f"1. Locate the exact TCP port and release token (RAG_TOKEN_...) for subsystem {target_a['subsystem']}.\n"
        f"2. Locate the exact TCP port and release token (RAG_TOKEN_...) for subsystem {target_b['subsystem']}.\n"
        "Format your first two lines strictly as:\n"
        f"{target_a['subsystem']}: <port> <token>\n"
        f"{target_b['subsystem']}: <port> <token>\n"
        "3. Immediately afterwards, provide a concise technical summary describing whether the "
        "retrieved documents specify any direct architectural dependency, shared transport port, "
        "or synchronization protocol between both subsystems, citing only verified facts present "
        "in the retrieved archive without speculation.\n"
    )
    query_ids = [QWEN38_IM_START] + tokenizer.encode(query_text)
    if len(query_ids) < 144:
        query_ids.extend(tokenizer.encode(" Verify exact numeric fidelity before answering." * 12))
    query_ids = query_ids[: max(len(query_ids), 144)]

    cold_ack_ids = [QWEN38_IM_START] + tokenizer.encode(
        "\n=== ARCHIVE INDEXING ACKNOWLEDGEMENT ===\n"
        "All engineering specification documents in this archive have been provided for indexing. "
        "Confirm receipt of the complete document set and state that the specifications are ready "
        "for subsequent multi-hop retrieval queries across arbitrary document permutations. "
        "Do not repeat the full documents; emit a brief confirmation acknowledging that all "
        "subsystems, TCP ports, and release tokens have been ingested for exact verification.\n"
        "Confirm indexing readiness concisely in one short paragraph covering archive integrity, "
        "delimiter boundaries, and deterministic retrieval readiness.\n"
    )
    if len(cold_ack_ids) < 144:
        cold_ack_ids.extend(tokenizer.encode(" Confirm deterministic archive indexing readiness." * 12))
    cold_ack_ids = cold_ack_ids[: max(len(cold_ack_ids), 144)]

    cold_order_prompt: List[int] = list(sys_preamble_ids)
    for d in docs:
        cold_order_prompt.extend(d["tokens"])
    cold_order_prompt.extend(cold_ack_ids)
    cold_order_prompt.extend(QWEN38_USER_TO_ASSISTANT_NOTHINK)

    permuted_prompt: List[int] = list(sys_preamble_ids)
    for idx in perm:
        permuted_prompt.extend(docs[idx]["tokens"])
    permuted_prompt.extend(query_ids)
    permuted_prompt.extend(QWEN38_USER_TO_ASSISTANT_NOTHINK)

    return {
        "schema": "m5ultra-cacheblend-rag-v1",
        "num_docs": num_docs,
        "doc_tokens": doc_tokens,
        "permutation": perm,
        "docs": docs,
        "expected_answers": [
            str(target_a["port"]),
            target_a["token_code"],
            str(target_b["port"]),
            target_b["token_code"],
        ],
        "cold_order_prompt_tokens": cold_order_prompt,
        "permuted_prompt_tokens": permuted_prompt,
        "total_prompt_tokens": len(permuted_prompt),
    }


def detect_supported_prefill_flags(serve_bin: str) -> set:
    try:
        out = subprocess.check_output([serve_bin, "--help"], text=True, stderr=subprocess.STDOUT)
    except Exception:
        return set()
    flags = set()
    for name in (
        "--prefill-linear-mode",
        "--prefill-attention-mode",
        "--prefill-sparse-minference",
        "--no-prefill-sparse-minference",
        "--prefill-cache-blend",
        "--no-prefill-cache-blend",
        "--prefill-ane-mode",
        "--prefill-ane-offload",
        "--no-prefill-ane-offload",
    ):
        alt = name.replace("--no-", "--[no-]", 1) if name.startswith("--no-") else name.replace("--", "--[no-]", 1)
        if name in out or alt in out:
            flags.add(name)
    return flags


class IsolatedServeProcess:
    """Manages an isolated `ninfer-metal-serve` child process with guaranteed cleanup."""

    def __init__(
        self,
        serve_bin: str,
        model_dir: str,
        host: str,
        port: int,
        max_context: int,
        kv_dtype: str,
        spec_backend: str,
        prefix_cache_gb: float,
        metallib_dir: Optional[str],
        cell_cfg: Dict[str, Any],
        log_path: Path,
        extra_env: Optional[Dict[str, str]] = None,
        artifact_path: Optional[str] = None,
    ):
        self.serve_bin = serve_bin
        self.model_dir = model_dir
        self.artifact_path = artifact_path
        self.host = host
        self.port = port
        self.max_context = max_context
        self.kv_dtype = kv_dtype
        self.spec_backend = spec_backend
        self.prefix_cache_gb = prefix_cache_gb
        self.metallib_dir = metallib_dir
        self.cell_cfg = cell_cfg
        self.log_path = log_path
        self.extra_env = dict(extra_env or {})
        self.proc: Optional[subprocess.Popen] = None
        self.cmd: List[str] = []
        self._log_fh: Optional[Any] = None

    def start(self, startup_timeout_s: float = 900.0) -> Dict[str, Any]:
        if check_port_open(self.host, self.port):
            raise RuntimeError(f"Isolated port {self.host}:{self.port} is already in use")

        cmd = [
            self.serve_bin,
            "--no-default-config",
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--provider",
            "inproc",
            "--model-dir",
            self.model_dir,
            "--max-context",
            str(self.max_context),
            "--max-concurrency",
            "1",
            "--kv-dtype",
            self.kv_dtype,
            "--spec-backend",
            self.spec_backend,
            "--draft-tokens",
            "0",
            "--sampling-contract",
            "position_coupled_v1",
            "--prefix-cache-gb",
            f"{self.prefix_cache_gb:.2f}",
            "--rolling-prefix-cache",
            "off",
        ]
        if self.artifact_path:
            cmd.extend(["--artifact", self.artifact_path])
        if self.metallib_dir:
            cmd.extend(["--metallib-dir", self.metallib_dir])

        supported = detect_supported_prefill_flags(self.serve_bin)
        if "--prefill-linear-mode" in supported:
            cmd.extend(["--prefill-linear-mode", str(self.cell_cfg["prefill_linear_mode"])])
        if "--prefill-attention-mode" in supported:
            cmd.extend(["--prefill-attention-mode", str(self.cell_cfg["prefill_attention_mode"])])
        if self.cell_cfg.get("prefill_sparse_minference") and "--prefill-sparse-minference" in supported:
            cmd.append("--prefill-sparse-minference")
        if self.cell_cfg.get("prefill_cache_blend") and "--prefill-cache-blend" in supported:
            cmd.append("--prefill-cache-blend")
        ane_mode = self.cell_cfg.get("prefill_ane_mode")
        if ane_mode and "--prefill-ane-mode" in supported:
            cmd.extend(["--prefill-ane-mode", str(ane_mode)])
        elif self.cell_cfg.get("prefill_ane_offload"):
            if "--prefill-ane-mode" in supported:
                cmd.extend(["--prefill-ane-mode", "auto"])
            elif "--prefill-ane-offload" in supported:
                cmd.append("--prefill-ane-offload")

        self.cmd = cmd
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "w", encoding="utf-8")
        child_env = os.environ.copy()
        child_env.update(self.extra_env)
        t0 = time.monotonic()
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            env=child_env,
            start_new_session=True,
        )

        deadline = t0 + startup_timeout_s
        last_err = ""
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                self._log_fh.flush()
                tail = self.log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
                raise RuntimeError(
                    f"ninfer-metal-serve exited prematurely (rc={self.proc.returncode}):\n{tail}"
                )
            try:
                with urllib.request.urlopen(
                    f"http://{self.host}:{self.port}/health", timeout=2.0
                ) as resp:
                    if resp.status == 200:
                        health = json.loads(resp.read().decode("utf-8"))
                        if health.get("status") == "ok":
                            return {
                                "pid": self.proc.pid,
                                "load_wall_s": round(time.monotonic() - t0, 3),
                                "health": health,
                                "cmd": cmd,
                                "idle_phys_footprint_bytes": query_server_phys_footprint(
                                    self.host, self.port
                                ),
                            }
            except Exception as exc:
                last_err = str(exc)
            time.sleep(0.5)

        self.stop()
        raise RuntimeError(
            f"Timed out waiting for ninfer-metal-serve on {self.host}:{self.port}: {last_err}"
        )

    def stop(self) -> None:
        if self.proc is not None:
            if self.proc.poll() is None:
                try:
                    os.killpg(self.proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    self.proc.wait(timeout=15.0)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(self.proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    self.proc.wait(timeout=5.0)
            self.proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None


def stream_completion_request(
    host: str,
    port: int,
    model_id: str,
    prompt_token_ids: Sequence[int],
    max_tokens: int,
    temperature: float = 0.0,
    timeout_s: float = 1800.0,
) -> Dict[str, Any]:
    """Send streaming `/v1/completions` request with exact token IDs and measure TTFT & decode."""
    payload = json.dumps(
        {
            "model": model_id,
            "prompt": list(prompt_token_ids),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        },
        separators=(",", ":"),
    ).encode("utf-8")

    conn = http.client.HTTPConnection(host, port, timeout=timeout_s)
    t_start = time.monotonic()
    t_first_tok: Optional[float] = None
    t_last_tok: Optional[float] = None
    chunks_text: List[str] = []
    token_events = 0
    finish_reason: Optional[str] = None
    usage: Dict[str, Any] = {}

    try:
        conn.request(
            "POST",
            "/v1/completions",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            },
        )
        resp = conn.getresponse()
        if resp.status != 200:
            err_body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {resp.status} from /v1/completions: {err_body[:1000]}")

        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                break
            obj = json.loads(data_str)
            if "error" in obj:
                raise RuntimeError(f"Server SSE error: {obj['error']}")
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                ch0 = choices[0]
                text = ch0.get("text")
                fr = ch0.get("finish_reason")
                if fr is not None:
                    finish_reason = fr
                if text:
                    now = time.monotonic()
                    if t_first_tok is None:
                        t_first_tok = now
                    t_last_tok = now
                    token_events += 1
                    chunks_text.append(text)
    finally:
        conn.close()

    t_end = time.monotonic()
    if t_first_tok is None:
        t_first_tok = t_end
    if t_last_tok is None:
        t_last_tok = t_first_tok

    ttft_s = t_first_tok - t_start
    total_wall_s = t_end - t_start
    prompt_n = int(usage.get("prompt_tokens", len(prompt_token_ids)))
    completion_n = int(usage.get("completion_tokens", token_events))
    cached_tokens = int(
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    )
    decode_span_s = max(0.0, t_last_tok - t_first_tok)
    decode_tok_per_s = (
        (completion_n - 1) / decode_span_s
        if completion_n > 1 and decode_span_s > 0.0
        else 0.0
    )
    prefill_tok_per_s = prompt_n / ttft_s if ttft_s > 0.0 else 0.0
    full_text = "".join(chunks_text)

    return {
        "prompt_tokens": prompt_n,
        "completion_tokens": completion_n,
        "cached_tokens": cached_tokens,
        "finish_reason": finish_reason,
        "ttft_s": ttft_s,
        "ttft_ms": ttft_s * 1000.0,
        "prefill_tok_per_s": prefill_tok_per_s,
        "decode_span_s": decode_span_s,
        "decode_tok_per_s": decode_tok_per_s,
        "total_wall_s": total_wall_s,
        "generated_text": full_text,
        "generated_sha256": sha256_text(full_text),
    }


def run_model_cold_matrix(
    *,
    model_label: str,
    model_dir: str,
    artifact_path: Optional[str] = None,
    serve_bin: str,
    metallib_dir: Optional[str],
    cpp_bench_bin: Optional[str],
    host: str,
    port: int,
    max_context: int,
    kv_dtype: str,
    spec_backend: str,
    tiers: Sequence[int],
    gen_tokens: int,
    depths: Sequence[int],
    warmup_tokens: int,
    cell_id: str,
    cell_cfg: Dict[str, Any],
    jsonl_fh: Any,
    run_tag: str,
    multi_needle: bool = False,
    trials_per_tier: int = 1,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run isolated cold-prefill + decode + NIAH sweep across all tiers for one model."""
    pre_8080 = verify_port_8080_idle()
    tokenizer = TokenizerBridge(model_dir, cpp_bench_bin=cpp_bench_bin)
    max_prompt = max(tiers) - gen_tokens
    print(
        f"[{model_label}] Building deterministic haystack pool ({max_prompt + 4096} tokens)...",
        flush=True,
    )
    haystack_pool = build_haystack_pool_tokens(
        tokenizer, max_prompt + 4096, seed_tag=f"niah_pool_{run_tag}"
    )

    log_path = Path(tempfile.gettempdir()) / f"ninfer_bench_{model_label}_{cell_id}_{port}.log"
    server = IsolatedServeProcess(
        serve_bin=serve_bin,
        model_dir=model_dir,
        host=host,
        port=port,
        max_context=max_context,
        kv_dtype=kv_dtype,
        spec_backend=spec_backend,
        prefix_cache_gb=float(cell_cfg.get("prefix_cache_gb", 0.0)),
        metallib_dir=metallib_dir,
        cell_cfg=cell_cfg,
        log_path=log_path,
        extra_env=extra_env,
        artifact_path=artifact_path,
    )

    tier_summaries: List[Dict[str, Any]] = []
    startup_info: Dict[str, Any] = {}
    try:
        print(
            f"[{model_label}] Starting isolated ninfer-metal-serve on {host}:{port} "
            f"(cell={cell_id}, max_context={max_context}, kv_dtype={kv_dtype}, spec={spec_backend})...",
            flush=True,
        )
        startup_info = server.start()
        model_id = startup_info["health"].get("model", Path(model_dir).name)
        srv_pid = startup_info["pid"]
        print(
            f"[{model_label}] Ready PID={srv_pid} in {startup_info['load_wall_s']:.2f}s "
            f"(idle phys_footprint={(startup_info['idle_phys_footprint_bytes'] or 0)/(1<<30):.2f} GiB)",
            flush=True,
        )

        # Warmup pass (triggers Metal pipeline compilation & UMA page faults before timed sweep).
        warm_ids, _ = build_exact_niah_prompt_tokens(
            tokenizer,
            haystack_pool,
            prompt_tokens=warmup_tokens,
            tier=warmup_tokens + 16,
            depth_pct=50,
            trial_idx=0,
            nonce_tag=f"{run_tag}_{model_label}_warmup",
        )
        print(f"[{model_label}] Running warmup ({warmup_tokens} prompt + 16 gen)...", flush=True)
        warm_res = stream_completion_request(
            host, port, model_id, warm_ids, max_tokens=16, temperature=0.0
        )
        print(
            f"[{model_label}] Warmup done: TTFT={warm_res['ttft_s']:.3f}s "
            f"({warm_res['prefill_tok_per_s']:.1f} tok/s), decode={warm_res['decode_tok_per_s']:.1f} tok/s",
            flush=True,
        )

        for tier in tiers:
            prompt_len = tier - gen_tokens
            assert prompt_len < max_context, f"prompt_len={prompt_len} >= max_context={max_context}"
            verify_port_8080_idle()
            trials: List[Dict[str, Any]] = []
            trial_specs: List[Tuple[int, Optional[int]]] = (
                [(i, None) for i in range(1, max(1, trials_per_tier) + 1)]
                if multi_needle
                else [(i, d) for i, d in enumerate(depths, start=1)]
            )

            for trial_idx, depth_pct in trial_specs:
                if multi_needle:
                    prompt_ids, passkeys_map = build_exact_multi_needle_prompt_tokens(
                        tokenizer,
                        haystack_pool,
                        prompt_tokens=prompt_len,
                        tier=tier,
                        depths=depths,
                        trial_idx=trial_idx,
                        nonce_tag=f"{run_tag}_{model_label}_{cell_id}",
                    )
                    passkey_summary = ",".join(f"{d}%:{passkeys_map[d]}" for d in sorted(passkeys_map))
                else:
                    assert depth_pct is not None
                    prompt_ids, single_pk = build_exact_niah_prompt_tokens(
                        tokenizer,
                        haystack_pool,
                        prompt_tokens=prompt_len,
                        tier=tier,
                        depth_pct=depth_pct,
                        trial_idx=trial_idx,
                        nonce_tag=f"{run_tag}_{model_label}_{cell_id}",
                    )
                    passkeys_map = {depth_pct: single_pk}
                    passkey_summary = single_pk

                tel_before = query_prefill_telemetry(host, port)
                sampler = FootprintSampler(host, port, srv_pid, interval_s=0.2)
                sampler.start()
                try:
                    res = stream_completion_request(
                        host,
                        port,
                        model_id,
                        prompt_ids,
                        max_tokens=gen_tokens,
                        temperature=0.0,
                    )
                finally:
                    mem_stats = sampler.stop()
                tel_after = query_prefill_telemetry(host, port)
                tel_delta = diff_prefill_telemetry(tel_before, tel_after)

                if cell_cfg.get("prefix_cache_gb", 0.0) == 0.0 and res["cached_tokens"] != 0:
                    raise RuntimeError(
                        f"Expected 0 cached_tokens on cold prefill, got {res['cached_tokens']}"
                    )

                depth_hits = {str(d): (pk in res["generated_text"]) for d, pk in passkeys_map.items()}
                n_hits = sum(1 for v in depth_hits.values() if v)
                n_total = len(depth_hits)
                niah_hit = (n_hits == n_total)
                trial_rec = {
                    "timestamp": utc_now_iso(),
                    "model_label": model_label,
                    "model_dir": model_dir,
                    "cell_id": cell_id,
                    "context_tier": tier,
                    "prompt_tokens": res["prompt_tokens"],
                    "requested_gen_tokens": gen_tokens,
                    "completion_tokens": res["completion_tokens"],
                    "cached_tokens": res["cached_tokens"],
                    "multi_needle": multi_needle,
                    "depth_pct": depth_pct,
                    "trial_idx": trial_idx,
                    "ttft_s": round(res["ttft_s"], 5),
                    "ttft_ms": round(res["ttft_ms"], 2),
                    "prefill_tok_per_s": round(res["prefill_tok_per_s"], 2),
                    "decode_tok_per_s": round(res["decode_tok_per_s"], 2),
                    "decode_span_s": round(res["decode_span_s"], 5),
                    "total_wall_s": round(res["total_wall_s"], 4),
                    "finish_reason": res["finish_reason"],
                    "peak_phys_footprint_bytes": mem_stats["peak_phys_footprint_bytes"],
                    "peak_phys_footprint_gib": mem_stats["peak_phys_footprint_gib"],
                    "peak_rss_gib": mem_stats["peak_rss_gib"],
                    "prefill_telemetry": tel_delta,
                    "niah_expected_passkey": passkey_summary,
                    "niah_depth_hits": depth_hits,
                    "niah_hits": n_hits,
                    "niah_total": n_total,
                    "niah_hit": niah_hit,
                    "generated_sha256": res["generated_sha256"],
                    "generated_text": res["generated_text"],
                }
                trials.append(trial_rec)
                jsonl_fh.write(json.dumps(trial_rec, ensure_ascii=False) + "\n")
                jsonl_fh.flush()
                depth_lbl = f"multi({n_hits}/{n_total})" if multi_needle else f"depth={depth_pct:2d}%"
                print(
                    f"[{model_label}][{cell_id}] tier={tier//1024}K {depth_lbl} "
                    f"TTFT={trial_rec['ttft_s']:7.3f}s ({trial_rec['prefill_tok_per_s']:7.1f} tok/s) "
                    f"decode={trial_rec['decode_tok_per_s']:5.1f} tok/s (n={trial_rec['completion_tokens']}) "
                    f"phys={trial_rec['peak_phys_footprint_gib']:5.2f} GiB "
                    f"w4a8={tel_delta['w4a8_tokens']} sage2={tel_delta['sage2_tokens']} "
                    f"ane={tel_delta['ane_offload_tokens']} minf_skip={tel_delta['minference_blocks_skipped']}/{tel_delta['minference_blocks_total']} "
                    f"NIAH={'PASS' if niah_hit else 'FAIL'} ({n_hits}/{n_total})",
                    flush=True,
                )

            total_hits = sum(t["niah_hits"] for t in trials)
            total_needles = sum(t["niah_total"] for t in trials)
            tier_summary = {
                "context_tier": tier,
                "context_label": f"{tier // 1024}K",
                "prompt_tokens": prompt_len,
                "gen_tokens": gen_tokens,
                "num_trials": len(trials),
                "median_ttft_s": round(statistics.median(t["ttft_s"] for t in trials), 5),
                "median_ttft_ms": round(statistics.median(t["ttft_ms"] for t in trials), 2),
                "min_ttft_s": round(min(t["ttft_s"] for t in trials), 5),
                "max_ttft_s": round(max(t["ttft_s"] for t in trials), 5),
                "median_prefill_tok_per_s": round(
                    statistics.median(t["prefill_tok_per_s"] for t in trials), 2
                ),
                "median_decode_tok_per_s": round(
                    statistics.median(t["decode_tok_per_s"] for t in trials), 2
                ),
                "peak_phys_footprint_gib": round(
                    max(t["peak_phys_footprint_gib"] for t in trials), 4
                ),
                "peak_rss_gib": round(max(t["peak_rss_gib"] for t in trials), 4),
                "niah_hits": total_hits,
                "niah_total": total_needles,
                "niah_pass_rate": round(total_hits / float(max(1, total_needles)), 4),
                "prefill_telemetry_last": trials[-1]["prefill_telemetry"] if trials else {},
                "trials": trials,
            }
            tier_summaries.append(tier_summary)
    finally:
        server.stop()

    post_8080 = verify_port_8080_idle()
    return {
        "model_label": model_label,
        "model_dir": model_dir,
        "artifact_path": artifact_path,
        "cell_id": cell_id,
        "cell_config": cell_cfg,
        "max_context": max_context,
        "kv_dtype": kv_dtype,
        "spec_backend": spec_backend,
        "startup": startup_info,
        "port_8080_pre": pre_8080,
        "port_8080_post": post_8080,
        "tiers": tier_summaries,
    }


def run_cpp_ppl_eval(
    *,
    cpp_bench_bin: str,
    model_label: str,
    model_dir: str,
    artifact_path: Optional[str] = None,
    metallib_dir: Optional[str],
    kv_dtype: str,
    ppl_lengths: Sequence[int],
    chunk_size: int,
    logits_dir: Optional[Path],
    cell_cfg: Dict[str, Any],
    extra_env: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Run deterministic 4K / 16K teacher-forced PPL via C++ `omlx_m5_prefill_matrix_bench`."""
    verify_port_8080_idle()
    results: List[Dict[str, Any]] = []
    max_len = max(ppl_lengths)
    corpus_text = generate_deterministic_corpus_text(
        max(int(max_len * 5.2), 64000), seed_tag="wikitext_code_ppl_v1"
    )
    child_env = os.environ.copy()
    if extra_env:
        child_env.update(extra_env)
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tf:
        tf.write(corpus_text)
        corpus_path = Path(tf.name)
    try:
        for tok_len in ppl_lengths:
            cmd = [
                cpp_bench_bin,
                "--mode",
                "ppl",
                "--model-dir",
                model_dir,
                "--kv-dtype",
                kv_dtype,
                "--max-context",
                str(max(tok_len + 128, 32768)),
                "--chunk-size",
                str(chunk_size),
                "--text-file",
                str(corpus_path),
                "--max-tokens",
                str(tok_len),
                "--prefill-linear-mode",
                str(cell_cfg["prefill_linear_mode"]),
                "--prefill-attention-mode",
                str(cell_cfg["prefill_attention_mode"]),
            ]
            if artifact_path:
                cmd.extend(["--artifact", artifact_path])
            if metallib_dir:
                cmd.extend(["--metallib-dir", metallib_dir])
            if cell_cfg.get("prefill_sparse_minference"):
                cmd.append("--prefill-sparse-minference")
            if cell_cfg.get("prefill_cache_blend"):
                cmd.append("--prefill-cache-blend")
            if cell_cfg.get("prefill_ane_mode"):
                cmd.extend(["--prefill-ane-mode", str(cell_cfg["prefill_ane_mode"])])
            elif cell_cfg.get("prefill_ane_offload"):
                cmd.append("--prefill-ane-offload")
            if logits_dir is not None:
                logits_dir.mkdir(parents=True, exist_ok=True)
                lpath = logits_dir / f"{model_label}_ppl_{tok_len}_logits.bin"
                cmd.extend(["--logits-out", str(lpath)])

            print(f"[{model_label}] Running C++ teacher-forced PPL ({tok_len} tokens)...", flush=True)
            proc = subprocess.run(cmd, text=True, capture_output=True, env=child_env)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"omlx_m5_prefill_matrix_bench failed (rc={proc.returncode}):\n"
                    f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout[:1000]}"
                )
            parsed = json.loads(proc.stdout)
            parsed["model_label"] = model_label
            results.append(parsed)
            print(
                f"[{model_label}] PPL@{tok_len}: ppl={parsed['ppl']:.4f} "
                f"mean_nll={parsed['mean_nll']:.5f} top1={parsed['top1_accuracy']*100:.2f}% "
                f"phys={parsed['peak_phys_footprint_gib']:.2f} GiB",
                flush=True,
            )
    finally:
        corpus_path.unlink(missing_ok=True)
    return results


def render_markdown_report(report: Dict[str, Any]) -> str:
    """Render structured Markdown baseline/matrix report from JSON summary."""
    lines: List[str] = []
    meta = report["metadata"]
    lines.append(f"# {meta['title']}")
    lines.append("")
    lines.append(f"- **Date (UTC)**: `{meta['date_utc']}`")
    lines.append(f"- **Host**: `{meta['hostname']}` (`{meta['machine']}`, `{meta['os_version']}`)")
    lines.append(f"- **Binary / Baseline Commit**: `{meta['baseline_commit']}` (`{meta['serve_bin']}`)")
    lines.append(f"- **Cell**: `{meta['cell_id']}` — {meta['cell_label']}")
    lines.append(
        f"- **Envelope**: `--max-context {meta['max_context']} --max-concurrency 1 "
        f"--kv-dtype {meta['kv_dtype']} --spec-backend {meta['spec_backend']} "
        f"--prefix-cache-gb {meta['prefix_cache_gb']}`"
    )
    lines.append(f"- **Production `:8080` Safety**: Verified idle (`0.0%` CPU / `0` active requests) before & after every model run; untouched.")
    lines.append("")

    for model_entry in report["models"]:
        lines.append(f"## {model_entry['model_label']} (`{model_entry['model_dir']}`)")
        lines.append("")
        lines.append(
            "| Context | Prompt Tok | Gen Tok | Median Cold TTFT (s) | Median Cold TTFT (ms) | "
            "Prefill (tok/s) | Decode (tok/s) | Peak `phys_footprint` (GiB) | NIAH (10/50/90%) |"
        )
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for t in model_entry["tiers"]:
            niah_str = f"{t['niah_hits']}/{t['niah_total']} ({t['niah_pass_rate']*100:.0f}%)"
            lines.append(
                f"| **{t['context_label']}** | {t['prompt_tokens']:,} | {t['gen_tokens']} | "
                f"{t['median_ttft_s']:.3f} | {t['median_ttft_ms']:.1f} | "
                f"**{t['median_prefill_tok_per_s']:.1f}** | {t['median_decode_tok_per_s']:.1f} | "
                f"{t['peak_phys_footprint_gib']:.2f} | {niah_str} |"
            )
        lines.append("")

    if report.get("ppl_evals"):
        lines.append("## Teacher-Forced Logprob Perplexity (`4K` & `16K` Deterministic Corpus)")
        lines.append("")
        lines.append("| Model | Tokens | Scored | Mean NLL | PPL | Top-1 Acc | Last-Token Argmax | Peak `phys_footprint` (GiB) |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for p in report["ppl_evals"]:
            lines.append(
                f"| {p['model_label']} | {p['num_tokens']:,} | {p['scored_tokens']:,} | "
                f"{p['mean_nll']:.5f} | **{p['ppl']:.4f}** | {p['top1_accuracy']*100:.2f}% | "
                f"`{p['last_token_argmax']}` | {p['peak_phys_footprint_gib']:.2f} |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


class BackenddJsonRpcClient:
    """Minimal AF_UNIX newline-delimited JSON-RPC client for `mseries_backendd`."""

    def __init__(self, socket_path: str, timeout_s: float = 600.0):
        self.socket_path = socket_path
        self.timeout_s = timeout_s
        self._seq = 0
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.settimeout(timeout_s)
        self._sock.connect(socket_path)
        self._rfile = self._sock.makefile("r", encoding="utf-8")

    def call(self, method: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self._seq += 1
        req = {"id": self._seq, "method": method, "params": params or {}}
        self._sock.sendall((json.dumps(req, separators=(",", ":")) + "\n").encode("utf-8"))
        line = self._rfile.readline()
        if not line:
            raise RuntimeError(f"EOF from mseries_backendd during {method}")
        resp = json.loads(line)
        if not resp.get("ok"):
            raise RuntimeError(f"mseries_backendd {method} failed: {resp.get('error')}")
        return resp.get("result", {})

    def close(self) -> None:
        try:
            self._rfile.close()
        finally:
            self._sock.close()


def run_cacheblend_rag_eval(
    *,
    model_label: str,
    model_dir: str,
    artifact_path: Optional[str] = None,
    serve_bin: str,
    metallib_dir: Optional[str],
    cpp_bench_bin: Optional[str],
    host: str,
    port: int,
    max_context: int,
    kv_dtype: str,
    spec_backend: str,
    num_docs: int,
    doc_tokens: int,
    gen_tokens: int,
    cell_id: str,
    cell_cfg: Dict[str, Any],
    prefix_cache_gb_override: Optional[float] = None,
    extra_env: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Run multi-document permuted 4K-chunk RAG QA benchmark (Wave 3 CacheBlend gate)."""
    verify_port_8080_idle()
    tokenizer = TokenizerBridge(model_dir, cpp_bench_bin=cpp_bench_bin)
    dataset = generate_permuted_rag_dataset(tokenizer, num_docs=num_docs, doc_tokens=doc_tokens)
    log_path = Path(tempfile.gettempdir()) / f"ninfer_rag_{model_label}_{cell_id}_{num_docs}_{port}.log"
    prefix_gb = (
        prefix_cache_gb_override
        if prefix_cache_gb_override is not None
        else max(24.0, float(cell_cfg.get("prefix_cache_gb", 24.0)))
    )
    server = IsolatedServeProcess(
        serve_bin=serve_bin,
        model_dir=model_dir,
        host=host,
        port=port,
        max_context=max_context,
        kv_dtype=kv_dtype,
        spec_backend=spec_backend,
        prefix_cache_gb=prefix_gb,
        metallib_dir=metallib_dir,
        cell_cfg=cell_cfg,
        log_path=log_path,
        extra_env=extra_env,
        artifact_path=artifact_path,
    )
    try:
        startup = server.start()
        model_id = startup["health"].get("model", Path(model_dir).name)
        srv_pid = startup["pid"]

        # Phase 1: Cold prime of all documents in original order [D0 .. D_{K-1}].
        prime_tel_before = query_prefill_telemetry(host, port)
        prime_sampler = FootprintSampler(host, port, srv_pid, interval_s=0.2)
        prime_sampler.start()
        try:
            prime_res = stream_completion_request(
                host,
                port,
                model_id,
                dataset["cold_order_prompt_tokens"],
                max_tokens=8,
                temperature=0.0,
            )
        finally:
            prime_mem = prime_sampler.stop()
        prime_tel_after = query_prefill_telemetry(host, port)

        # Phase 2: Warm stitched evaluation of permuted multi-document RAG prompt.
        warm_tel_before = query_prefill_telemetry(host, port)
        sampler = FootprintSampler(host, port, srv_pid, interval_s=0.2)
        sampler.start()
        try:
            rag_res = stream_completion_request(
                host,
                port,
                model_id,
                dataset["permuted_prompt_tokens"],
                max_tokens=gen_tokens,
                temperature=0.0,
            )
        finally:
            mem_stats = sampler.stop()
        warm_tel_after = query_prefill_telemetry(host, port)
        warm_tel = diff_prefill_telemetry(warm_tel_before, warm_tel_after)
        prime_tel = diff_prefill_telemetry(prime_tel_before, prime_tel_after)

        matched = [ans for ans in dataset["expected_answers"] if ans in rag_res["generated_text"]]
        speedup = (
            prime_res["ttft_s"] / rag_res["ttft_s"]
            if rag_res["ttft_s"] > 0.0
            else 0.0
        )
        rec = {
            "model_label": model_label,
            "model_dir": model_dir,
            "cell_id": cell_id,
            "num_docs": num_docs,
            "doc_tokens": doc_tokens,
            "context_label": f"{(num_docs * doc_tokens) // 1024}K",
            "prefix_cache_gb": prefix_gb,
            "cold_prime_tokens": prime_res["prompt_tokens"],
            "cold_prime_ttft_s": round(prime_res["ttft_s"], 5),
            "cold_prime_tok_per_s": round(prime_res["prefill_tok_per_s"], 2),
            "cold_prime_peak_phys_gib": prime_mem["peak_phys_footprint_gib"],
            "cold_prime_telemetry": prime_tel,
            "total_prompt_tokens": dataset["total_prompt_tokens"],
            "permutation": dataset["permutation"],
            "permuted_ttft_s": round(rag_res["ttft_s"], 5),
            "permuted_prefill_tok_per_s": round(rag_res["prefill_tok_per_s"], 2),
            "permuted_decode_tok_per_s": round(rag_res["decode_tok_per_s"], 2),
            "warm_vs_prime_speedup": round(speedup, 3),
            "cached_tokens": rag_res["cached_tokens"],
            "warm_telemetry": warm_tel,
            "peak_phys_footprint_gib": max(
                prime_mem["peak_phys_footprint_gib"], mem_stats["peak_phys_footprint_gib"]
            ),
            "expected_answers": dataset["expected_answers"],
            "matched_answers": matched,
            "exact_qa_pass": len(matched) == len(dataset["expected_answers"]),
            "generated_sha256": rag_res["generated_sha256"],
            "generated_text": rag_res["generated_text"],
        }
        print(
            f"[RAG][{model_label}][{cell_id}] {rec['context_label']} ({num_docs}x{doc_tokens}): "
            f"prime={rec['cold_prime_ttft_s']:.3f}s ({rec['cold_prime_tok_per_s']:.1f} tok/s) -> "
            f"warm={rec['permuted_ttft_s']:.3f}s ({rec['permuted_prefill_tok_per_s']:.1f} tok/s, {speedup:.2f}x) "
            f"hit={warm_tel['cache_blend_hit_tokens']} recomp={warm_tel['cache_blend_recomputed_tokens']} "
            f"ane={warm_tel['ane_offload_tokens']} phys={rec['peak_phys_footprint_gib']:.2f} GiB "
            f"QA={'PASS' if rec['exact_qa_pass'] else 'FAIL'} ({len(matched)}/{len(dataset['expected_answers'])})",
            flush=True,
        )
        return rec
    finally:
        server.stop()


def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="M5 Ultra 16K-256K Prefill Matrix, NIAH, PPL & CacheBlend RAG Harness"
    )
    ap.add_argument(
        "--action",
        choices=["bench", "ppl", "bench-rag", "generate-rag", "check-host"],
        default="bench",
        help="Harness action (default: bench)",
    )
    ap.add_argument(
        "--serve-bin",
        default=os.path.expanduser(
            "~/Library/Application Support/oMLX/Providers/io.ninfer.metal/"
            "versions/miv-v0.03-7336688/bin/ninfer-metal-serve"
        ),
        help="Path to ninfer-metal-serve binary",
    )
    ap.add_argument("--artifact", default=None, help="Optional explicit .miv artifact path override")
    ap.add_argument("--metallib-dir", default=None, help="Optional metallib directory override")
    ap.add_argument(
        "--cpp-bench-bin",
        default=None,
        help="Path to omlx_m5_prefill_matrix_bench (for tokenization & C++ PPL)",
    )
    ap.add_argument(
        "--model",
        action="append",
        default=[],
        help="Model spec as label=path (repeatable)",
    )
    ap.add_argument("--cell", default="M0", choices=sorted(MATRIX_CELLS.keys()))
    ap.add_argument(
        "--prefill-linear-mode",
        default=None,
        choices=["bf16", "w4a8", "w4a4"],
        help="Optional override for cell prefill_linear_mode (bf16|w4a8|w4a4)",
    )
    ap.add_argument(
        "--prefill-ane-mode",
        default=None,
        choices=["off", "auto", "compact_int8", "ultra_fp16_swiglu"],
        help="Optional override for cell prefill_ane_mode (off|auto|compact_int8|ultra_fp16_swiglu)",
    )
    ap.add_argument("--baseline-commit", default="7336688")
    ap.add_argument("--title", default="M5 Ultra 16K-256K Prefill Baseline (7336688)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18090)
    ap.add_argument("--max-context", type=int, default=262144)
    ap.add_argument("--kv-dtype", default="affine4_g128", choices=["affine4_g128", "bf16"])
    ap.add_argument("--spec-backend", default="none", choices=["none", "mtp", "dflash2", "auto"])
    ap.add_argument(
        "--tiers",
        default="16384,32768,65536,131072,262144",
        help="Comma-separated context tiers",
    )
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--depths", default="10,50,90", help="Comma-separated NIAH depths (%%)")
    ap.add_argument(
        "--multi-needle",
        action="store_true",
        help="Embed all --depths passkeys in a single prompt per trial",
    )
    ap.add_argument(
        "--trials-per-tier",
        type=int,
        default=1,
        help="Trials per tier when --multi-needle is enabled (default: 1)",
    )
    ap.add_argument(
        "--env",
        action="append",
        default=[],
        help="Extra KEY=VALUE env override for server & C++ PPL child processes",
    )
    ap.add_argument(
        "--prefix-cache-gb",
        type=float,
        default=None,
        help="Optional override for --prefix-cache-gb (e.g. 24.0 for 256K CacheBlend RAG)",
    )
    ap.add_argument("--warmup-tokens", type=int, default=2048)
    ap.add_argument("--run-ppl", action="store_true", help="Run C++ teacher-forced PPL")
    ap.add_argument("--ppl-lengths", default="4096,16384")
    ap.add_argument("--ppl-chunk-size", type=int, default=512)
    ap.add_argument("--logits-dir", default=None)
    ap.add_argument("--out-json", required=False, default=None)
    ap.add_argument("--out-jsonl", required=False, default=None)
    ap.add_argument("--out-md", required=False, default=None)
    ap.add_argument("--rag-docs", type=int, default=8)
    ap.add_argument("--rag-docs-list", default=None, help="Comma-separated doc counts for multi-tier bench-rag (e.g. 4,8,16,32,63)")
    ap.add_argument("--rag-doc-tokens", type=int, default=4096)
    ap.add_argument("--run-tag", default="baseline_7336688")
    return ap.parse_args()


def parse_env_list(entries: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in entries:
        if "=" not in item:
            raise ValueError(f"--env entry must be KEY=VALUE: {item}")
        k, v = item.split("=", 1)
        out[k] = v
    return out


def parse_model_spec(spec: str, default_artifact: Optional[str] = None) -> Tuple[str, str, Optional[str]]:
    label, rest = spec.split("=", 1)
    if "::" in rest:
        model_dir, art = rest.split("::", 1)
        return label, model_dir, art or default_artifact
    return label, rest, default_artifact


def _resolve_cell_cfg(args: argparse.Namespace) -> Dict[str, Any]:
    cell_cfg = _resolve_cell_cfg(args)
    if args.prefill_linear_mode is not None:
        cell_cfg["prefill_linear_mode"] = args.prefill_linear_mode
    if args.prefill_ane_mode is not None:
        cell_cfg["prefill_ane_mode"] = args.prefill_ane_mode
        cell_cfg["prefill_ane_offload"] = args.prefill_ane_mode != "off"
    elif "prefill_ane_mode" not in cell_cfg:
        cell_cfg["prefill_ane_mode"] = "auto" if cell_cfg.get("prefill_ane_offload") else "off"
    return cell_cfg


def main() -> int:
    args = parse_cli()
    extra_env = parse_env_list(args.env)

    if args.action == "check-host":
        info = verify_port_8080_idle()
        print(json.dumps(info, indent=2))
        return 0

    if args.action == "generate-rag":
        if not args.model:
            raise SystemExit("--model label=path is required for generate-rag")
        _, model_dir, _ = parse_model_spec(args.model[0], args.artifact)
        tok = TokenizerBridge(model_dir, cpp_bench_bin=args.cpp_bench_bin)
        ds = generate_permuted_rag_dataset(tok, num_docs=args.rag_docs, doc_tokens=args.rag_doc_tokens)
        if args.out_json:
            Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out_json).write_text(json.dumps(ds, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "schema": ds["schema"],
                    "num_docs": ds["num_docs"],
                    "doc_tokens": ds["doc_tokens"],
                    "total_prompt_tokens": ds["total_prompt_tokens"],
                    "permutation": ds["permutation"],
                    "expected_answers": ds["expected_answers"],
                },
                indent=2,
            )
        )
        return 0

    if args.action == "ppl":
        if not args.cpp_bench_bin:
            raise SystemExit("--action ppl requires --cpp-bench-bin PATH")
        if not args.model:
            raise SystemExit("--model label=path is required for ppl")
        cell_cfg = _resolve_cell_cfg(args)
        ppl_lengths = [int(x.strip()) for x in args.ppl_lengths.split(",") if x.strip()]
        logits_dir = Path(args.logits_dir) if args.logits_dir else None
        ppl_results: List[Dict[str, Any]] = []
        for spec in args.model:
            label, model_dir, artifact_path = parse_model_spec(spec, args.artifact)
            res = run_cpp_ppl_eval(
                cpp_bench_bin=args.cpp_bench_bin,
                model_label=label,
                model_dir=model_dir,
                artifact_path=artifact_path,
                metallib_dir=args.metallib_dir,
                kv_dtype=args.kv_dtype,
                ppl_lengths=ppl_lengths,
                chunk_size=args.ppl_chunk_size,
                logits_dir=logits_dir,
                cell_cfg=cell_cfg,
                extra_env=extra_env,
            )
            ppl_results.extend(res)
            if args.out_json:
                Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.out_json).write_text(json.dumps(ppl_results, indent=2), encoding="utf-8")
        return 0

    if args.action == "bench-rag":
        if not args.model:
            raise SystemExit("--model label=path is required for bench-rag")
        cell_cfg = _resolve_cell_cfg(args)
        doc_counts = (
            [int(x.strip()) for x in args.rag_docs_list.split(",") if x.strip()]
            if args.rag_docs_list
            else [args.rag_docs]
        )
        rag_results: List[Dict[str, Any]] = []
        for spec in args.model:
            label, model_dir, artifact_path = parse_model_spec(spec, args.artifact)
            for ndocs in doc_counts:
                rec = run_cacheblend_rag_eval(
                    model_label=label,
                    model_dir=model_dir,
                    artifact_path=artifact_path,
                    serve_bin=args.serve_bin,
                    metallib_dir=args.metallib_dir,
                    cpp_bench_bin=args.cpp_bench_bin,
                    host=args.host,
                    port=args.port,
                    max_context=args.max_context,
                    kv_dtype=args.kv_dtype,
                    spec_backend=args.spec_backend,
                    num_docs=ndocs,
                    doc_tokens=args.rag_doc_tokens,
                    gen_tokens=args.gen_tokens,
                    cell_id=args.cell,
                    cell_cfg=cell_cfg,
                    prefix_cache_gb_override=args.prefix_cache_gb,
                    extra_env=extra_env,
                )
                rag_results.append(rec)
                if args.out_json:
                    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out_json).write_text(json.dumps(rag_results, indent=2), encoding="utf-8")
        print(json.dumps(rag_results, indent=2))
        return 0

    if not args.model:
        args.model = [
            "Qwen3.8-27B-mxfp4=/Users/scoltc/repo/models/mlx-community/Qwen3.8-27B-mxfp4",
            "Qwen3.8-27B-oQ4e-mtp=/Users/scoltc/repo/models/Jundot/Qwen3.8-27B-oQ4e-mtp",
        ]

    tiers = [int(x.strip()) for x in args.tiers.split(",") if x.strip()]
    depths = [int(x.strip()) for x in args.depths.split(",") if x.strip()]
    ppl_lengths = [int(x.strip()) for x in args.ppl_lengths.split(",") if x.strip()]
    cell_cfg = _resolve_cell_cfg(args)
    if args.prefix_cache_gb is not None:
        cell_cfg["prefix_cache_gb"] = args.prefix_cache_gb

    out_json_path = Path(args.out_json) if args.out_json else Path("bench_result.json")
    out_jsonl_path = (
        Path(args.out_jsonl)
        if args.out_jsonl
        else out_json_path.with_suffix(".jsonl")
    )
    out_md_path = Path(args.out_md) if args.out_md else out_json_path.with_suffix(".md")
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)

    sw_vers = ""
    try:
        sw_vers = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
    except Exception:
        sw_vers = platform.mac_ver()[0]

    report: Dict[str, Any] = {
        "metadata": {
            "schema_version": 1,
            "title": args.title,
            "date_utc": utc_now_iso(),
            "hostname": socket.gethostname(),
            "machine": platform.machine(),
            "os_version": f"macOS {sw_vers}",
            "baseline_commit": args.baseline_commit,
            "serve_bin": args.serve_bin,
            "artifact": args.artifact,
            "cell_id": args.cell,
            "cell_label": cell_cfg["label"],
            "cell_config": cell_cfg,
            "extra_env": extra_env,
            "max_context": args.max_context,
            "kv_dtype": args.kv_dtype,
            "spec_backend": args.spec_backend,
            "prefix_cache_gb": float(cell_cfg.get("prefix_cache_gb", 0.0)),
            "tiers": tiers,
            "gen_tokens": args.gen_tokens,
            "depths": depths,
            "multi_needle": args.multi_needle,
            "trials_per_tier": args.trials_per_tier,
        },
        "models": [],
        "ppl_evals": [],
    }

    with open(out_jsonl_path, "w", encoding="utf-8") as jsonl_fh:
        for spec in args.model:
            label, model_dir, artifact_path = parse_model_spec(spec, args.artifact)
            entry = run_model_cold_matrix(
                model_label=label,
                model_dir=model_dir,
                artifact_path=artifact_path,
                serve_bin=args.serve_bin,
                metallib_dir=args.metallib_dir,
                cpp_bench_bin=args.cpp_bench_bin,
                host=args.host,
                port=args.port,
                max_context=args.max_context,
                kv_dtype=args.kv_dtype,
                spec_backend=args.spec_backend,
                tiers=tiers,
                gen_tokens=args.gen_tokens,
                depths=depths,
                warmup_tokens=args.warmup_tokens,
                cell_id=args.cell,
                cell_cfg=cell_cfg,
                jsonl_fh=jsonl_fh,
                run_tag=args.run_tag,
                multi_needle=args.multi_needle,
                trials_per_tier=args.trials_per_tier,
                extra_env=extra_env,
            )
            report["models"].append(entry)
            # Persist incrementally after each model so partial progress is never lost.
            out_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            out_md_path.write_text(render_markdown_report(report), encoding="utf-8")

    if args.run_ppl:
        if not args.cpp_bench_bin:
            raise SystemExit("--run-ppl requires --cpp-bench-bin PATH")
        logits_dir = Path(args.logits_dir) if args.logits_dir else None
        for spec in args.model:
            label, model_dir, artifact_path = parse_model_spec(spec, args.artifact)
            ppl_res = run_cpp_ppl_eval(
                cpp_bench_bin=args.cpp_bench_bin,
                model_label=label,
                model_dir=model_dir,
                artifact_path=artifact_path,
                metallib_dir=args.metallib_dir,
                kv_dtype=args.kv_dtype,
                ppl_lengths=ppl_lengths,
                chunk_size=args.ppl_chunk_size,
                logits_dir=logits_dir,
                cell_cfg=cell_cfg,
                extra_env=extra_env,
            )
            report["ppl_evals"].extend(ppl_res)
            out_json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
            out_md_path.write_text(render_markdown_report(report), encoding="utf-8")

    print(f"Saved JSON:  {out_json_path}", flush=True)
    print(f"Saved JSONL: {out_jsonl_path}", flush=True)
    print(f"Saved MD:    {out_md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
