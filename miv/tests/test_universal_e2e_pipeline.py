#!/usr/bin/env python3
"""True full-stack end-to-end tests across all 6 required scenarios:
synthetic .safetensors -> omlx.engine.m5_sidecar.inspect_m5rt_source_fast (< 5 ms)
-> m5_artifact.py inspect-source --json (< 5 ms)
-> m5_artifact.py convert --nax-pretile (.miv + .conversion.json)
-> C++ mseries_backendd (ArtifactModel -> QwenForward -> Engine -> Metal GPU) load & forward decode.
"""

from __future__ import annotations

import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MIV_TOOLS = REPO_ROOT / "miv" / "tools"
OMLX_SRC = REPO_ROOT / "omlx"
for _p in (str(MIV_TOOLS), str(OMLX_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import m5_artifact
from omlx.engine.m5_sidecar import inspect_m5rt_source_fast


def _find_daemon_and_metallib() -> tuple[Path, Path]:
    env_bin = os.environ.get("OMLX_M5_DAEMON_BIN")
    if env_bin:
        bin_path = Path(env_bin).resolve()
        mlib = bin_path.parent / "metallib" / "omlx_m5_qwen_ar.metallib"
        if not mlib.is_file():
            mlib = bin_path.parent.parent / "miv" / "metallib" / "omlx_m5_qwen_ar.metallib"
        if bin_path.is_file() and mlib.is_file():
            return bin_path, mlib.resolve()

    candidates = [
        REPO_ROOT / "build" / "release" / "miv",
        REPO_ROOT / "build" / "d0-m4" / "miv",
    ]
    for base in candidates:
        bin_p = base / "mseries_backendd"
        mlib_p = base / "metallib" / "omlx_m5_qwen_ar.metallib"
        if bin_p.is_file() and os.access(bin_p, os.X_OK) and mlib_p.is_file():
            return bin_p.resolve(), mlib_p.resolve()
    pytest.skip("mseries_backendd and omlx_m5_qwen_ar.metallib not built")


def _bf16_bytes(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype("<u2").tobytes()


def _pack_codes(codes: np.ndarray, bits: int) -> bytes:
    leading = codes.shape[:-1]
    k = int(codes.shape[-1])
    rows = int(np.prod(leading))
    flat = codes.reshape(rows, k)
    bit_mat = ((flat[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1).reshape(
        rows, k * bits
    )
    packed = np.packbits(bit_mat, axis=1, bitorder="little")
    return np.ascontiguousarray(packed).tobytes()


def _affine_triplet(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    bits: int = 4,
    group_size: int = 64,
) -> dict[str, tuple[str, list[int], bytes]]:
    leading = shape[:-1]
    k = int(shape[-1])
    groups = k // group_size
    codes = rng.integers(0, 1 << bits, size=shape, dtype=np.uint8)
    scales = rng.uniform(0.01, 0.06, size=(*leading, groups)).astype(np.float32)
    biases = -scales * float(1 << (bits - 1))
    return {
        ".weight": ("U32", [*leading, k * bits // 32], _pack_codes(codes, bits)),
        ".scales": ("BF16", [*leading, groups], _bf16_bytes(scales)),
        ".biases": ("BF16", [*leading, groups], _bf16_bytes(biases)),
    }


def _add_module(
    tensors: dict[str, tuple[str, list[int], bytes]],
    module: str,
    spec: dict[str, tuple[str, list[int], bytes]],
) -> None:
    for suffix, val in spec.items():
        tensors[module + suffix] = val


def _write_checkpoint(
    model_dir: Path,
    config: dict[str, Any],
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    header: dict[str, dict[str, Any]] = {}
    offset = 0
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded += b" " * (-len(encoded) % 8)
    shard_path = model_dir / "model.safetensors"
    with shard_path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for name in sorted(tensors):
            fh.write(tensors[name][2])
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (model_dir / "REVISION").write_text("e2e-universal-test-v1\n", encoding="utf-8")
    return model_dir


def _build_checkpoint_tensors(
    rng: np.random.Generator,
    *,
    prefix: str = "model",
    hidden_size: int = 512,
    intermediate_size: int = 1024,
    vocab_size: int = 256,
    q_heads: int = 4,
    kv_heads: int = 2,
    head_dim: int = 128,
    layer_types: list[str] | None = None,
    gated_attn: bool = False,
    qk_norm: bool = True,
    tie_word_embeddings: bool = False,
    num_experts: int = 0,
    moe_intermediate_size: int = 256,
    shared_expert_intermediate_size: int = 0,
    expert_container_name: str = "switch_mlp",
) -> dict[str, tuple[str, list[int], bytes]]:
    ltypes = layer_types or ["full_attention", "full_attention"]
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    head_prefix = (
        prefix[: -len("model")] if prefix.endswith("model") else (prefix + ".")
    )
    tensors[f"{prefix}.embed_tokens.weight"] = (
        "BF16",
        [vocab_size, hidden_size],
        _bf16_bytes(rng.normal(0, 0.04, size=(vocab_size, hidden_size))),
    )
    tensors[f"{prefix}.norm.weight"] = (
        "BF16",
        [hidden_size],
        _bf16_bytes(np.ones(hidden_size, dtype=np.float32)),
    )
    if not tie_word_embeddings:
        _add_module(
            tensors,
            f"{head_prefix}lm_head",
            _affine_triplet(rng, (vocab_size, hidden_size), bits=4, group_size=64),
        )

    q_out = (2 if gated_attn else 1) * q_heads * head_dim
    kv_out = kv_heads * head_dim
    o_in = q_heads * head_dim

    for l_idx, lt in enumerate(ltypes):
        lp = f"{prefix}.layers.{l_idx}"
        tensors[f"{lp}.input_layernorm.weight"] = (
            "BF16",
            [hidden_size],
            _bf16_bytes(np.ones(hidden_size, dtype=np.float32)),
        )
        tensors[f"{lp}.post_attention_layernorm.weight"] = (
            "BF16",
            [hidden_size],
            _bf16_bytes(np.ones(hidden_size, dtype=np.float32)),
        )
        if lt == "full_attention":
            _add_module(
                tensors,
                f"{lp}.self_attn.q_proj",
                _affine_triplet(rng, (q_out, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.self_attn.k_proj",
                _affine_triplet(rng, (kv_out, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.self_attn.v_proj",
                _affine_triplet(rng, (kv_out, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.self_attn.o_proj",
                _affine_triplet(rng, (hidden_size, o_in), bits=4, group_size=64),
            )
            if qk_norm:
                tensors[f"{lp}.self_attn.q_norm.weight"] = (
                    "BF16",
                    [head_dim],
                    _bf16_bytes(np.ones(head_dim, dtype=np.float32)),
                )
                tensors[f"{lp}.self_attn.k_norm.weight"] = (
                    "BF16",
                    [head_dim],
                    _bf16_bytes(np.ones(head_dim, dtype=np.float32)),
                )
        else:
            qkv_w = 2 * (2 * 128) + (4 * 128)
            val_w = 4 * 128
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_qkv",
                _affine_triplet(rng, (qkv_w, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_z",
                _affine_triplet(rng, (val_w, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_a",
                _affine_triplet(rng, (4, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_b",
                _affine_triplet(rng, (4, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.out_proj",
                _affine_triplet(rng, (hidden_size, val_w), bits=4, group_size=64),
            )
            tensors[f"{lp}.linear_attn.conv1d.weight"] = (
                "BF16",
                [qkv_w, 4, 1],
                _bf16_bytes(rng.normal(0, 0.03, size=(qkv_w, 4, 1))),
            )
            tensors[f"{lp}.linear_attn.A_log"] = (
                "F32",
                [4],
                np.linspace(-0.5, -0.1, 4, dtype="<f4").tobytes(),
            )
            tensors[f"{lp}.linear_attn.dt_bias"] = (
                "BF16",
                [4],
                _bf16_bytes(np.full(4, -1.0, dtype=np.float32)),
            )
            tensors[f"{lp}.linear_attn.norm.weight"] = (
                "BF16",
                [128],
                _bf16_bytes(np.ones(128, dtype=np.float32)),
            )

        if num_experts == 0:
            _add_module(
                tensors,
                f"{lp}.mlp.gate_proj",
                _affine_triplet(rng, (intermediate_size, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.mlp.up_proj",
                _affine_triplet(rng, (intermediate_size, hidden_size), bits=4, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.mlp.down_proj",
                _affine_triplet(rng, (hidden_size, intermediate_size), bits=4, group_size=64),
            )
        else:
            _add_module(
                tensors,
                f"{lp}.mlp.gate",
                _affine_triplet(rng, (num_experts, hidden_size), bits=8, group_size=64),
            )
            exp_base = f"{lp}.mlp.{expert_container_name}"
            _add_module(
                tensors,
                f"{exp_base}.gate_proj",
                _affine_triplet(
                    rng, (num_experts, moe_intermediate_size, hidden_size), bits=4, group_size=64
                ),
            )
            _add_module(
                tensors,
                f"{exp_base}.up_proj",
                _affine_triplet(
                    rng, (num_experts, moe_intermediate_size, hidden_size), bits=4, group_size=64
                ),
            )
            _add_module(
                tensors,
                f"{exp_base}.down_proj",
                _affine_triplet(
                    rng, (num_experts, hidden_size, moe_intermediate_size), bits=4, group_size=64
                ),
            )
            if shared_expert_intermediate_size > 0:
                _add_module(
                    tensors,
                    f"{lp}.mlp.shared_expert_gate",
                    _affine_triplet(rng, (1, hidden_size), bits=8, group_size=64),
                )
                _add_module(
                    tensors,
                    f"{lp}.mlp.shared_expert.gate_proj",
                    _affine_triplet(
                        rng, (shared_expert_intermediate_size, hidden_size), bits=4, group_size=64
                    ),
                )
                _add_module(
                    tensors,
                    f"{lp}.mlp.shared_expert.up_proj",
                    _affine_triplet(
                        rng, (shared_expert_intermediate_size, hidden_size), bits=4, group_size=64
                    ),
                )
                _add_module(
                    tensors,
                    f"{lp}.mlp.shared_expert.down_proj",
                    _affine_triplet(
                        rng, (hidden_size, shared_expert_intermediate_size), bits=4, group_size=64
                    ),
                )
    return tensors


class _RpcClient:
    def __init__(self, sock_path: Path) -> None:
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        deadline = time.monotonic() + 10.0
        while True:
            try:
                self._sock.connect(str(sock_path))
                break
            except (FileNotFoundError, ConnectionRefusedError):
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.01)
        self._r = self._sock.makefile("r", encoding="utf-8")
        self._w = self._sock.makefile("w", encoding="utf-8")
        self._id = 1

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        rid = self._id
        self._id += 1
        self._w.write(json.dumps({"id": rid, "method": method, "params": params or {}}) + "\n")
        self._w.flush()
        line = self._r.readline()
        assert line, f"EOF waiting for {method}"
        resp = json.loads(line)
        assert resp["id"] == rid
        if not resp["ok"]:
            raise RuntimeError(resp["error"]["message"])
        return resp["result"]

    def close(self) -> None:
        self._r.close()
        self._w.close()
        self._sock.close()


def _run_full_pipeline_and_daemon_forward(
    ckpt_dir: Path,
    out_miv: Path,
    daemon_bin: Path,
    qwen_metallib: Path,
    *,
    vocab_size: int = 256,
    kv_dtype: str = "bf16",
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[int]]:
    # 1. Fast in-process oMLX sidecar probe (< 5 ms)
    sidecar_probe = inspect_m5rt_source_fast(ckpt_dir)
    assert sidecar_probe["eligible"] is True, sidecar_probe
    assert sidecar_probe["elapsed_ms"] < 5.0, sidecar_probe["elapsed_ms"]

    # 2. CLI inspect-source --json (< 5 ms internal probe)
    insp_proc = subprocess.run(
        [
            sys.executable,
            str(MIV_TOOLS / "m5_artifact.py"),
            "inspect-source",
            str(ckpt_dir),
            "--json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_inspect = json.loads(insp_proc.stdout)
    assert cli_inspect["eligible"] is True, cli_inspect
    assert cli_inspect["elapsed_ms"] < 5.0, cli_inspect["elapsed_ms"]

    # 3. CLI convert --nax-pretile -> .miv + .conversion.json
    subprocess.run(
        [
            sys.executable,
            str(MIV_TOOLS / "m5_artifact.py"),
            "convert",
            "--source",
            str(ckpt_dir),
            "--output",
            str(out_miv),
            "--source-repo",
            "e2e/universal-test",
            "--nax-pretile",
            "--jobs",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert out_miv.is_file()
    conv_report = json.loads(
        out_miv.with_name(out_miv.name + ".conversion.json").read_text(encoding="utf-8")
    )

    # 4. C++ mseries_backendd (ArtifactModel -> QwenForward -> Engine -> Metal GPU) load & forward!
    with tempfile.TemporaryDirectory(prefix="miv-univ-e2e-") as sock_tmp:
        sock_path = Path(sock_tmp) / "provider.sock"
        proc = subprocess.Popen(
            [str(daemon_bin), "--socket", str(sock_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        client = _RpcClient(sock_path)
        shutdown_sent = False
        try:
            hello = client.call("hello")
            assert "architecture_capabilities" in hello
            assert hello["architecture_capabilities"]["model_type_whitelist_enforced"] is False

            load_res = client.call(
                "load",
                {
                    "model_dir": str(ckpt_dir),
                    "max_context": 1024,
                    "max_concurrency": 1,
                    "kv_dtype": kv_dtype,
                    "path_overrides": {
                        "artifact": str(out_miv),
                        "qwen_ar_metallib": str(qwen_metallib),
                    },
                },
            )
            assert load_res["max_context"] == 1024

            req_id = client.call(
                "submit",
                {
                    "prompt_tokens": [1, 7, 3, 9, 2],
                    "max_new_tokens": 6,
                    "sampling": {"temperature": 0, "top_p": 1, "seed": 11},
                },
            )["request_id"]

            generated: list[int] = []
            terminal = None
            deadline = time.monotonic() + 30.0
            while terminal is None and time.monotonic() < deadline:
                for ev in client.call("poll", {"max_wait_us": 20_000})["events"]:
                    if ev["request_id"] != req_id:
                        continue
                    if ev["kind"] == "tokens":
                        generated.extend(ev["tokens"])
                    else:
                        terminal = ev
            assert terminal is not None and terminal["kind"] == "finished", terminal
            assert len(generated) == 6
            assert all(0 <= tok < vocab_size for tok in generated)
            client.call("shutdown")
            shutdown_sent = True
        finally:
            client.close()
            if not shutdown_sent and proc.poll() is None:
                proc.terminate()
            try:
                out, err = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            if shutdown_sent:
                assert proc.returncode == 0, (out, err)

    return sidecar_probe, cli_inspect, conv_report, generated


def test_e2e_1_custom_model_type_finetune(tmp_path: Path) -> None:
    """Scenario 1: Custom model_type isomorphic fine-tune (hybrid GDN + gated GQA)."""
    daemon_bin, qwen_metallib = _find_daemon_and_metallib()
    rng = np.random.default_rng(9101)
    tensors = _build_checkpoint_tensors(
        rng,
        prefix="language_model.model",
        layer_types=["linear_attention", "full_attention"],
        gated_attn=True,
        qk_norm=True,
        head_dim=256,
        q_heads=2,
        kv_heads=1,
    )
    cfg = {
        "model_type": "community_acme_hybrid_v2",
        "text_config": {
            "model_type": "community_acme_text_v2",
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "vocab_size": 256,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            # Intentionally omit layer_types here to ALSO verify C++/Python implicit GDN layer inference!
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.25},
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt = _write_checkpoint(tmp_path / "s1_custom_ft", cfg, tensors)
    sp, ci, rep, toks = _run_full_pipeline_and_daemon_forward(
        ckpt, tmp_path / "s1.miv", daemon_bin, qwen_metallib, kv_dtype="affine4_g128"
    )
    assert sp["model_type_hint_matched"] is False
    assert ci["graph_gate"]["has_attn_output_gate"] is True
    assert rep["repack_plan_summary"]["nax_pretile"] > 0
    assert len(toks) == 6


@pytest.mark.parametrize(
    ("head_dim", "q_heads", "qk_norm", "tied"),
    [
        (128, 4, False, True),   # Llama-style: flat config, 0-GDN, ungated, no QK-norm, tied embeddings
        (256, 2, True, False),   # Qwen3-style: flat config, 0-GDN, ungated, QK-norm, affine4_g128 KV
    ],
)
def test_e2e_2_zero_gdn_ungated_pure_gqa(
    tmp_path: Path, head_dim: int, q_heads: int, qk_norm: bool, tied: bool
) -> None:
    """Scenario 2: 0-layer GDN + ungated q_proj pure GQA (Qwen3 / Llama morphology)."""
    daemon_bin, qwen_metallib = _find_daemon_and_metallib()
    rng = np.random.default_rng(9200 + head_dim)
    tensors = _build_checkpoint_tensors(
        rng,
        prefix="model",
        layer_types=["full_attention", "full_attention"],
        gated_attn=False,
        qk_norm=qk_norm,
        tie_word_embeddings=tied,
        head_dim=head_dim,
        q_heads=q_heads,
        kv_heads=1,
    )
    cfg = {
        "model_type": "llama" if not qk_norm else "qwen3",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "vocab_size": 256,
        "num_attention_heads": q_heads,
        "num_key_value_heads": 1,
        "rope_theta": 500000.0,
        "tie_word_embeddings": tied,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt = _write_checkpoint(tmp_path / f"s2_gqa_hd{head_dim}", cfg, tensors)
    sp, ci, _rep, toks = _run_full_pipeline_and_daemon_forward(
        ckpt,
        tmp_path / f"s2_hd{head_dim}.miv",
        daemon_bin,
        qwen_metallib,
        kv_dtype="affine4_g128" if head_dim == 256 else "bf16",
    )
    assert sp["topology"] == "pure_gqa_swiglu"
    assert sp["has_attn_output_gate"] is False
    assert ci["graph_gate"]["has_attn_output_gate"] is False
    assert ci["graph_gate"]["linear_attention_layers"] == 0
    assert len(toks) == 6


@pytest.mark.parametrize(
    ("num_experts", "top_k", "shared_inter", "exp_container"),
    [
        (64, 4, 0, "experts"),         # 64 experts, top_k=4, NO shared_expert, .mlp.experts.* container!
        (128, 8, 256, "switch_mlp"),   # 128 experts, top_k=8, WITH shared_expert, .mlp.switch_mlp.*
    ],
)
def test_e2e_3_non_256_moe_experts_and_optional_shared_expert(
    tmp_path: Path,
    num_experts: int,
    top_k: int,
    shared_inter: int,
    exp_container: str,
) -> None:
    """Scenario 3: Non-256 expert MoE (64 top-4 no-shared & 128 top-8 with-shared)."""
    daemon_bin, qwen_metallib = _find_daemon_and_metallib()
    rng = np.random.default_rng(9300 + num_experts)
    tensors = _build_checkpoint_tensors(
        rng,
        layer_types=["full_attention"],
        gated_attn=False,
        qk_norm=True,
        num_experts=num_experts,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=shared_inter,
        expert_container_name=exp_container,
    )
    cfg: dict[str, Any] = {
        "model_type": "qwen3_moe",
        "hidden_size": 512,
        "moe_intermediate_size": 256,
        "num_hidden_layers": 1,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "num_experts": num_experts,
        "num_experts_per_tok": top_k,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    if shared_inter > 0:
        cfg["shared_expert_intermediate_size"] = shared_inter
    ckpt = _write_checkpoint(tmp_path / f"s3_moe_{num_experts}", cfg, tensors)
    sp, ci, _rep, toks = _run_full_pipeline_and_daemon_forward(
        ckpt, tmp_path / f"s3_moe_{num_experts}.miv", daemon_bin, qwen_metallib
    )
    assert sp["num_experts"] == num_experts
    assert sp["num_experts_per_tok"] == top_k
    assert sp["has_shared_expert"] is (shared_inter > 0)
    assert ci["graph_gate"]["has_shared_expert"] is (shared_inter > 0)
    assert len(toks) == 6


@pytest.mark.parametrize("head_mode", ["bf16", "q8_g64"])
def test_e2e_4_mixed_precision_affine_trunk_plus_bf16_or_8bit(
    tmp_path: Path, head_mode: str
) -> None:
    """Scenario 4: Mixed-precision (4-bit affine g64 trunk + BF16 o_proj + BF16/8-bit lm_head)."""
    daemon_bin, qwen_metallib = _find_daemon_and_metallib()
    rng = np.random.default_rng(9404)
    tensors = _build_checkpoint_tensors(
        rng, layer_types=["full_attention"], gated_attn=False, qk_norm=True
    )
    # Replace self_attn.o_proj with unquantized BF16 in the same .miv
    for suf in (".weight", ".scales", ".biases"):
        tensors.pop(f"model.layers.0.self_attn.o_proj{suf}", None)
    tensors["model.layers.0.self_attn.o_proj.weight"] = (
        "BF16",
        [512, 512],
        _bf16_bytes(rng.normal(0, 0.04, size=(512, 512))),
    )
    # Replace lm_head with BF16 or 8-bit affine g64 without per-layer config override
    for suf in (".weight", ".scales", ".biases"):
        tensors.pop(f"lm_head{suf}", None)
    if head_mode == "bf16":
        tensors["lm_head.weight"] = (
            "BF16",
            [256, 512],
            _bf16_bytes(rng.normal(0, 0.05, size=(256, 512))),
        )
    else:
        _add_module(
            tensors, "lm_head", _affine_triplet(rng, (256, 512), bits=8, group_size=64)
        )

    cfg = {
        "model_type": "community_mixed_quant",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 1,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt = _write_checkpoint(tmp_path / f"s4_mixed_{head_mode}", cfg, tensors)
    sp, _ci, rep, toks = _run_full_pipeline_and_daemon_forward(
        ckpt, tmp_path / f"s4_mixed_{head_mode}.miv", daemon_bin, qwen_metallib
    )
    assert sp["is_mixed_precision"] is True
    assert "direct_bf16_v1" in rep["repack_plan_summary"]["format_histogram"]
    assert "oq_affine_q4_g64_bf16_v1" in rep["repack_plan_summary"]["format_histogram"]
    assert len(toks) == 6


def test_e2e_5_vlm_checkpoint_strips_visual_weights_and_executes_text_miv(
    tmp_path: Path,
) -> None:
    """Scenario 5: VLM checkpoint with visual.* / projector weights stripped into runnable text .miv."""
    daemon_bin, qwen_metallib = _find_daemon_and_metallib()
    rng = np.random.default_rng(9505)
    tensors = _build_checkpoint_tensors(
        rng,
        prefix="language_model.model",
        layer_types=["full_attention"],
        gated_attn=True,
        qk_norm=True,
    )
    visual_tensors = {
        "visual.patch_embed.proj.weight": (
            "F32",
            [16, 3, 2, 4, 4],
            np.zeros((16, 3, 2, 4, 4), dtype="<f4").tobytes(),
        ),
        "model.visual.blocks.0.attn.qkv.bias": (
            "BF16",
            [384],
            _bf16_bytes(np.zeros(384, dtype=np.float32)),
        ),
        "vision_tower.encoder.layers.0.fc1.weight": (
            "BF16",
            [256, 128],
            _bf16_bytes(np.zeros((256, 128), dtype=np.float32)),
        ),
        "multi_modal_projector.linear_1.weight": (
            "BF16",
            [512, 256],
            _bf16_bytes(np.zeros((512, 256), dtype=np.float32)),
        ),
    }
    tensors.update(visual_tensors)
    cfg = {
        "model_type": "community_vlm_arch",
        "vision_config": {"hidden_size": 1152, "depth": 24},
        "text_config": {
            "model_type": "community_vlm_text",
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 1,
            "vocab_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt = _write_checkpoint(tmp_path / "s5_vlm", cfg, tensors)
    sp, ci, rep, toks = _run_full_pipeline_and_daemon_forward(
        ckpt, tmp_path / "s5_vlm.miv", daemon_bin, qwen_metallib
    )
    expected_skipped = sorted(visual_tensors.keys())
    assert sorted(sp["skipped_non_text_tensors"]) == expected_skipped
    assert ci["skipped_tensors"] == expected_skipped
    assert rep["skipped_tensors"] == expected_skipped
    assert rep["repack_plan_summary"]["skip_non_text"] == len(expected_skipped)
    assert len(toks) == 6


def test_e2e_6_negative_blockers_sub_5ms_and_zero_partial_files(tmp_path: Path) -> None:
    """Scenario 6: MLA & affine gs=32 intercepted in < 5 ms with exact layer & zero partial files."""
    rng = np.random.default_rng(9606)
    base_tensors = _build_checkpoint_tensors(
        rng, layer_types=["full_attention"], gated_attn=False, qk_norm=False
    )
    base_cfg = {
        "model_type": "custom_arch",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 1,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }

    # Case A: DeepSeek MLA topology
    mla_tensors = dict(base_tensors)
    mla_tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"] = (
        "BF16",
        [576, 512],
        _bf16_bytes(np.zeros((576, 512), dtype=np.float32)),
    )
    mla_ckpt = _write_checkpoint(
        tmp_path / "s6_mla", {**base_cfg, "kv_lora_rank": 512}, mla_tensors
    )
    sp_mla = inspect_m5rt_source_fast(mla_ckpt)
    ci_mla = m5_artifact.inspect_source_checkpoint(mla_ckpt)
    assert sp_mla["eligible"] is False and sp_mla["elapsed_ms"] < 5.0
    assert ci_mla["eligible"] is False and ci_mla["elapsed_ms"] < 5.0
    assert "MLA" in sp_mla["reason"] and "MLA" in ci_mla["reason"]
    assert "kv_a_proj_with_mqa" in sp_mla["reason"]

    out_mla_dir = tmp_path / "out_mla"
    conv_mla = subprocess.run(
        [
            sys.executable,
            str(MIV_TOOLS / "m5_artifact.py"),
            "convert",
            "--source",
            str(mla_ckpt),
            "--output",
            str(out_mla_dir / "model.miv"),
            "--source-repo",
            "e2e/blocked-mla",
            "--nax-pretile",
        ],
        capture_output=True,
        text=True,
    )
    assert conv_mla.returncode != 0
    assert "MLA" in (conv_mla.stderr + conv_mla.stdout)
    assert not out_mla_dir.exists() or list(out_mla_dir.iterdir()) == []

    # Case B: Per-tensor affine group_size=32 even when global config claims gs=64
    gs32_tensors = dict(base_tensors)
    _add_module(
        gs32_tensors,
        "model.layers.0.self_attn.k_proj",
        _affine_triplet(rng, (256, 512), bits=4, group_size=32),
    )
    gs32_ckpt = _write_checkpoint(tmp_path / "s6_gs32", base_cfg, gs32_tensors)
    sp_gs32 = inspect_m5rt_source_fast(gs32_ckpt)
    ci_gs32 = m5_artifact.inspect_source_checkpoint(gs32_ckpt)
    assert sp_gs32["eligible"] is False and sp_gs32["elapsed_ms"] < 5.0
    assert ci_gs32["eligible"] is False and ci_gs32["elapsed_ms"] < 5.0
    assert sp_gs32["first_blocker"]["tensor"] == "model.layers.0.self_attn.k_proj.weight"
    assert ci_gs32["first_blocker"]["tensor"] == "model.layers.0.self_attn.k_proj.weight"
    assert "32" in sp_gs32["reason"] and "32" in ci_gs32["reason"]

    out_gs32_dir = tmp_path / "out_gs32"
    conv_gs32 = subprocess.run(
        [
            sys.executable,
            str(MIV_TOOLS / "m5_artifact.py"),
            "convert",
            "--source",
            str(gs32_ckpt),
            "--output",
            str(out_gs32_dir / "model.miv"),
            "--source-repo",
            "e2e/blocked-gs32",
            "--nax-pretile",
        ],
        capture_output=True,
        text=True,
    )
    assert conv_gs32.returncode != 0
    assert "model.layers.0.self_attn.k_proj.weight" in (conv_gs32.stderr + conv_gs32.stdout)
    assert not out_gs32_dir.exists() or list(out_gs32_dir.iterdir()) == []
