"""Comprehensive tests for Task 1: Two-Layer Capability Gate + Per-Tensor Repack Planner."""

from __future__ import annotations

import json
import mmap
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MIV_TOOLS = REPO_ROOT / "miv" / "tools"
if str(MIV_TOOLS) not in sys.path:
    sys.path.insert(0, str(MIV_TOOLS))

import m5_artifact


def _bf16_bytes(values: np.ndarray) -> bytes:
    arr = np.asarray(values, dtype=np.float32)
    bits = arr.view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype("<u2").tobytes()


def _f16_bytes(values: np.ndarray) -> bytes:
    return np.asarray(values, dtype="<f2").tobytes()


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
    *,
    f16_sb: bool = False,
) -> dict[str, tuple[str, list[int], bytes]]:
    leading = shape[:-1]
    k = int(shape[-1])
    groups = k // group_size
    codes = rng.integers(0, 1 << bits, size=shape, dtype=np.uint8)
    scales = rng.uniform(0.02, 0.15, size=(*leading, groups)).astype(np.float32)
    biases = rng.uniform(-1.0, 1.0, size=(*leading, groups)).astype(np.float32)
    sb_dtype = "F16" if f16_sb else "BF16"
    sb_enc = _f16_bytes if f16_sb else _bf16_bytes
    return {
        ".weight": ("U32", [*leading, k * bits // 32], _pack_codes(codes, bits)),
        ".scales": (sb_dtype, [*leading, groups], sb_enc(scales)),
        ".biases": (sb_dtype, [*leading, groups], sb_enc(biases)),
    }


def _add_module(
    tensors: dict[str, tuple[str, list[int], bytes]],
    module: str,
    spec: dict[str, tuple[str, list[int], bytes]],
) -> None:
    for suffix, val in spec.items():
        tensors[module + suffix] = val


def _write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> int:
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
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(encoded)))
        fh.write(encoded)
        for name in sorted(tensors):
            fh.write(tensors[name][2])
    return path.stat().st_size


def _write_checkpoint(
    model_dir: Path,
    config: dict[str, Any],
    tensors: dict[str, tuple[str, list[int], bytes]],
    *,
    with_index: bool = True,
    revision: str = "rev-cap-test-001",
) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    shard = "model.safetensors"
    size = _write_safetensors(model_dir / shard, tensors)
    (model_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if with_index:
        index = {
            "metadata": {"total_size": size},
            "weight_map": {name: shard for name in sorted(tensors)},
        }
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    (model_dir / "REVISION").write_text(revision + "\n", encoding="utf-8")
    return model_dir


def _build_full_text_tensors(
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
    gated_attn: bool = True,
    qk_norm: bool = True,
    tie_word_embeddings: bool = False,
    quantized_embed: bool = False,
    num_experts: int = 0,
    moe_intermediate_size: int = 512,
    shared_expert_intermediate_size: int = 0,
    bits: int = 4,
    group_size: int = 64,
) -> dict[str, tuple[str, list[int], bytes]]:
    ltypes = layer_types or ["full_attention", "full_attention"]
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    head_prefix = (
        prefix[: -len("model")] if prefix.endswith("model") else (prefix + ".")
    )
    if quantized_embed:
        _add_module(
            tensors,
            f"{prefix}.embed_tokens",
            _affine_triplet(rng, (vocab_size, hidden_size), bits=bits, group_size=group_size),
        )
    else:
        tensors[f"{prefix}.embed_tokens.weight"] = (
            "BF16",
            [vocab_size, hidden_size],
            _bf16_bytes(rng.normal(0, 0.05, size=(vocab_size, hidden_size))),
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
            _affine_triplet(rng, (vocab_size, hidden_size), bits=bits, group_size=group_size),
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
                _affine_triplet(rng, (q_out, hidden_size), bits=bits, group_size=group_size),
            )
            _add_module(
                tensors,
                f"{lp}.self_attn.k_proj",
                _affine_triplet(rng, (kv_out, hidden_size), bits=bits, group_size=group_size),
            )
            _add_module(
                tensors,
                f"{lp}.self_attn.v_proj",
                _affine_triplet(rng, (kv_out, hidden_size), bits=bits, group_size=group_size),
            )
            _add_module(
                tensors,
                f"{lp}.self_attn.o_proj",
                _affine_triplet(rng, (hidden_size, o_in), bits=bits, group_size=group_size),
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
            # GDN layer: key_heads=2, val_heads=4, key_dim=128, val_dim=128, conv=4
            qkv_w = 2 * (2 * 128) + (4 * 128)
            val_w = 4 * 128
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_qkv",
                _affine_triplet(rng, (qkv_w, hidden_size), bits=bits, group_size=group_size),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_z",
                _affine_triplet(rng, (val_w, hidden_size), bits=bits, group_size=group_size),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_a",
                _affine_triplet(rng, (4, hidden_size), bits=bits, group_size=group_size),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.in_proj_b",
                _affine_triplet(rng, (4, hidden_size), bits=bits, group_size=group_size),
            )
            _add_module(
                tensors,
                f"{lp}.linear_attn.out_proj",
                _affine_triplet(rng, (hidden_size, val_w), bits=bits, group_size=group_size),
            )
            tensors[f"{lp}.linear_attn.conv1d.weight"] = (
                "BF16",
                [qkv_w, 1, 4],
                _bf16_bytes(rng.normal(0, 0.05, size=(qkv_w, 1, 4))),
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
                _affine_triplet(
                    rng, (intermediate_size, hidden_size), bits=bits, group_size=group_size
                ),
            )
            _add_module(
                tensors,
                f"{lp}.mlp.up_proj",
                _affine_triplet(
                    rng, (intermediate_size, hidden_size), bits=bits, group_size=group_size
                ),
            )
            _add_module(
                tensors,
                f"{lp}.mlp.down_proj",
                _affine_triplet(
                    rng, (hidden_size, intermediate_size), bits=bits, group_size=group_size
                ),
            )
        else:
            _add_module(
                tensors,
                f"{lp}.mlp.gate",
                _affine_triplet(rng, (num_experts, hidden_size), bits=8, group_size=64),
            )
            _add_module(
                tensors,
                f"{lp}.mlp.switch_mlp.gate_proj",
                _affine_triplet(
                    rng,
                    (num_experts, moe_intermediate_size, hidden_size),
                    bits=bits,
                    group_size=group_size,
                ),
            )
            _add_module(
                tensors,
                f"{lp}.mlp.switch_mlp.up_proj",
                _affine_triplet(
                    rng,
                    (num_experts, moe_intermediate_size, hidden_size),
                    bits=bits,
                    group_size=group_size,
                ),
            )
            _add_module(
                tensors,
                f"{lp}.mlp.switch_mlp.down_proj",
                _affine_triplet(
                    rng,
                    (num_experts, hidden_size, moe_intermediate_size),
                    bits=bits,
                    group_size=group_size,
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
                        rng,
                        (shared_expert_intermediate_size, hidden_size),
                        bits=bits,
                        group_size=group_size,
                    ),
                )
                _add_module(
                    tensors,
                    f"{lp}.mlp.shared_expert.up_proj",
                    _affine_triplet(
                        rng,
                        (shared_expert_intermediate_size, hidden_size),
                        bits=bits,
                        group_size=group_size,
                    ),
                )
                _add_module(
                    tensors,
                    f"{lp}.mlp.shared_expert.down_proj",
                    _affine_triplet(
                        rng,
                        (hidden_size, shared_expert_intermediate_size),
                        bits=bits,
                        group_size=group_size,
                    ),
                )
    return tensors


# ---------------------------------------------------------------------------
# 1. Custom model_type + < 5 ms O(1) zero-weight-read & CLI inspect-source
# ---------------------------------------------------------------------------


def test_scenario1_custom_model_type_and_zero_weight_read_fast_inspect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rng = np.random.default_rng(1001)
    tensors = _build_full_text_tensors(
        rng,
        prefix="language_model.model",
        layer_types=["linear_attention", "full_attention"],
        gated_attn=True,
        qk_norm=True,
    )
    cfg = {
        "model_type": "custom_finetune_arch",
        "text_config": {
            "model_type": "custom_finetune_text_v1",
            "hidden_size": 512,
            "intermediate_size": 1024,
            "num_hidden_layers": 2,
            "vocab_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 128,
            "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "layer_types": ["linear_attention", "full_attention"],
            "rope_parameters": {"rope_theta": 10000.0, "partial_rotary_factor": 0.5},
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt = _write_checkpoint(tmp_path / "custom_gdn_hybrid", cfg, tensors)

    # Warm once, then forbid mmap.mmap completely during inspect_source_checkpoint!
    def _forbid_mmap(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("inspect_source_checkpoint must NEVER call mmap.mmap!")

    with monkeypatch.context() as mp:
        mp.setattr(mmap, "mmap", _forbid_mmap)
        res = m5_artifact.inspect_source_checkpoint(ckpt, nax_pretile=True)

    assert res["eligible"] is True
    assert res["blocker_code"] is None
    assert res["reason"] == "ok"
    assert res["model_type"] == "custom_finetune_arch"
    assert res["model_type_hint_matched"] is False
    assert res["graph_gate"]["model_type_hint_matched"] is False
    assert res["topology"] == "gdn_hybrid_gqa"
    assert res["graph_gate"]["has_attn_output_gate"] is True
    assert res["graph_gate"]["has_qk_norm"] is True
    assert res["elapsed_ms"] < 5.0, f"inspect_source_checkpoint took {res['elapsed_ms']} ms"

    # CLI inspect-source --json and --no-nax-pretile
    cli_proc = subprocess.run(
        [
            sys.executable,
            str(MIV_TOOLS / "m5_artifact.py"),
            "inspect-source",
            str(ckpt),
            "--json",
            "--no-nax-pretile",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    cli_doc = json.loads(cli_proc.stdout)
    assert cli_doc["eligible"] is True
    assert cli_doc["repack_plan"]["category_counts"]["nax_pretile"] == 0
    assert cli_doc["repack_plan"]["category_counts"]["rowsplit_or_direct"] > 0

    # Full conversion succeeds, F32 A_log converted to direct_bf16_v1, manifest model metadata recorded
    out_miv = tmp_path / "out" / "custom.miv"
    src = m5_artifact.SafeTensorSource(ckpt)
    try:
        rep = m5_artifact.write_artifact(
            src, out_miv, "community/custom-finetune", "auto", nax_pretile=True, jobs=2
        )
    finally:
        src.close()
    _, manifest = m5_artifact.read_artifact(out_miv, verify_payload=True)
    assert manifest["model"]["has_attn_output_gate"] is True
    assert manifest["model"]["has_qk_norm"] is True
    assert manifest["model"]["has_shared_expert"] is False
    assert manifest["model"]["skipped_non_text_count"] == 0
    objs = {o["name"]: o for o in manifest["objects"]}
    assert objs["language_model.model.layers.0.linear_attn.A_log"]["format"] == "direct_bf16_v1"
    assert rep["skipped_tensors"] == []
    assert rep["repack_plan_summary"]["nax_pretile"] > 0


# ---------------------------------------------------------------------------
# 2. 0-GDN pure GQA + ungated q_proj + flat config.json + tied embeddings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("quantized_embed", [False, True])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_scenario2_zero_gdn_ungated_flat_config_and_tied_embeddings(
    tmp_path: Path, quantized_embed: bool, qk_norm: bool
) -> None:
    rng = np.random.default_rng(2002)
    tensors = _build_full_text_tensors(
        rng,
        prefix="model",
        layer_types=["full_attention", "full_attention"],
        gated_attn=False,
        qk_norm=qk_norm,
        tie_word_embeddings=True,
        quantized_embed=quantized_embed,
    )
    # Flat config.json: no text_config, no layer_types, no linear_* fields, implicit head_dim=512//4=128
    flat_cfg = {
        "model_type": "llama_custom_gqa",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "rope_theta": 500000.0,
        "tie_word_embeddings": True,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt = _write_checkpoint(
        tmp_path / f"pure_gqa_qe{int(quantized_embed)}_qkn{int(qk_norm)}",
        flat_cfg,
        tensors,
    )
    insp = m5_artifact.inspect_source_checkpoint(ckpt)
    assert insp["eligible"] is True
    assert insp["topology"] == "pure_gqa_swiglu"
    assert insp["graph_gate"]["config_layout"] == "flat_root"
    assert insp["graph_gate"]["linear_attention_layers"] == 0
    assert insp["graph_gate"]["full_attention_layers"] == 2
    assert insp["graph_gate"]["head_dim"] == 128
    assert insp["graph_gate"]["has_attn_output_gate"] is False
    assert insp["graph_gate"]["has_qk_norm"] is qk_norm
    assert insp["graph_gate"]["tie_word_embeddings"] is True
    assert insp["elapsed_ms"] < 5.0

    out_miv = tmp_path / f"pure_gqa_{int(quantized_embed)}_{int(qk_norm)}.miv"
    src = m5_artifact.SafeTensorSource(ckpt)
    try:
        m5_artifact.write_artifact(
            src, out_miv, "test/pure-gqa-tied", "auto", nax_pretile=True, jobs=2
        )
    finally:
        src.close()

    _, manifest = m5_artifact.read_artifact(out_miv, verify_payload=True)
    assert manifest["model"]["has_attn_output_gate"] is False
    assert manifest["model"]["has_qk_norm"] is qk_norm
    embed_obj = next(
        o for o in manifest["objects"] if "embed_tokens" in o["name"]
    )
    lm_head_bindings = [b for b in manifest["bindings"] if b["use"] == "lm_head"]
    assert len(lm_head_bindings) == 1
    assert lm_head_bindings[0]["object_id"] == embed_obj["id"]
    assert lm_head_bindings[0]["out_features"] == 256
    assert lm_head_bindings[0]["in_features"] == 512

    # Pretile round-trip also preserves tied lm_head alias binding cleanly
    pt_out = tmp_path / f"pure_gqa_pt_{int(quantized_embed)}_{int(qk_norm)}.miv"
    m5_artifact.pretile_artifact(out_miv, pt_out, jobs=2)
    _, pt_man = m5_artifact.read_artifact(pt_out, verify_payload=True)
    assert [b for b in pt_man["bindings"] if b["use"] == "lm_head"] == lm_head_bindings


def test_scenario2_rejects_bad_q_proj_rows_and_partial_qk_norm(tmp_path: Path) -> None:
    rng = np.random.default_rng(2099)
    tensors = _build_full_text_tensors(
        rng, gated_attn=False, qk_norm=True, layer_types=["full_attention", "full_attention"]
    )
    # Remove only k_norm on layer 1 -> partial QK-norm mismatch
    tensors.pop("model.layers.1.self_attn.k_norm.weight")
    cfg = {
        "model_type": "qwen3",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 2,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt_qkn = _write_checkpoint(tmp_path / "bad_qkn", cfg, tensors)
    r1 = m5_artifact.inspect_source_checkpoint(ckpt_qkn)
    assert r1["eligible"] is False
    assert r1["blocker_code"] == "partial_qk_norm"

    # Bad N_q (768 instead of 512 or 1024)
    bad_q_tensors = _build_full_text_tensors(
        rng, gated_attn=False, qk_norm=False, layer_types=["full_attention"]
    )
    _add_module(
        bad_q_tensors,
        "model.layers.0.self_attn.q_proj",
        _affine_triplet(rng, (768, 512), bits=4, group_size=64),
    )
    cfg_1l = {**cfg, "num_hidden_layers": 1}
    ckpt_bad_q = _write_checkpoint(tmp_path / "bad_nq", cfg_1l, bad_q_tensors)
    r2 = m5_artifact.inspect_source_checkpoint(ckpt_bad_q)
    assert r2["eligible"] is False
    assert r2["blocker_code"] == "invalid_q_proj_shape"


# ---------------------------------------------------------------------------
# 3. 64-expert & 128-expert MoE (with and without shared_expert)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("num_experts", "top_k", "shared_inter"),
    [
        (64, 4, 0),
        (128, 8, 256),
    ],
)
def test_scenario3_generalized_moe_64_and_128_experts(
    tmp_path: Path, num_experts: int, top_k: int, shared_inter: int
) -> None:
    rng = np.random.default_rng(3000 + num_experts)
    tensors = _build_full_text_tensors(
        rng,
        layer_types=["full_attention"],
        gated_attn=False,
        qk_norm=True,
        num_experts=num_experts,
        moe_intermediate_size=256,
        shared_expert_intermediate_size=shared_inter,
    )
    cfg: dict[str, Any] = {
        "model_type": "qwen3_moe_custom",
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

    ckpt = _write_checkpoint(tmp_path / f"moe_{num_experts}_sh{shared_inter}", cfg, tensors)
    insp = m5_artifact.inspect_source_checkpoint(ckpt, nax_pretile=True)
    assert insp["eligible"] is True
    assert insp["elapsed_ms"] < 5.0
    assert insp["graph_gate"]["num_experts"] == num_experts
    assert insp["graph_gate"]["num_experts_per_tok"] == top_k
    assert insp["graph_gate"]["has_shared_expert"] is (shared_inter > 0)

    entries_by_name = {e["name"]: e for e in insp["repack_plan"]["entries"]}
    # 3D MoE experts have K >= 128 -> category "nax_pretile", layout preserved as oq_expert_rowsplit_k128_v1
    for exp_p in ("gate_proj", "up_proj", "down_proj"):
        exp_e = entries_by_name[f"model.layers.0.mlp.switch_mlp.{exp_p}"]
        assert exp_e["category"] == "nax_pretile"
        assert exp_e["layout"] == "oq_expert_rowsplit_k128_v1"
    # Router mlp.gate stays rowsplit_or_direct even with K=512
    router_e = entries_by_name["model.layers.0.mlp.gate"]
    assert router_e["category"] == "rowsplit_or_direct"
    assert router_e["layout"] == "oq_rowsplit_k128_v1"

    out_miv = tmp_path / f"moe_{num_experts}.miv"
    src = m5_artifact.SafeTensorSource(ckpt)
    try:
        m5_artifact.write_artifact(
            src, out_miv, f"test/moe-{num_experts}", "auto", nax_pretile=True, jobs=2
        )
    finally:
        src.close()
    _, man = m5_artifact.read_artifact(out_miv, verify_payload=True)
    assert man["model"]["has_shared_expert"] is (shared_inter > 0)
    objs = {o["name"]: o for o in man["objects"]}
    assert objs["model.layers.0.mlp.switch_mlp.gate_proj"]["layout"] == "oq_expert_rowsplit_k128_v1"
    assert objs["model.layers.0.mlp.gate"]["layout"] == "oq_rowsplit_k128_v1"


# ---------------------------------------------------------------------------
# 4. Mixed-precision 4-bit g64 + BF16/FP16/8-bit/6-bitg128 layers & head
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head_kind", ["bf16", "fp16", "q8_g64_unlisted"])
def test_scenario4_mixed_precision_per_tensor_inference_and_f16_cast(
    tmp_path: Path, head_kind: str
) -> None:
    rng = np.random.default_rng(4004)
    tensors = _build_full_text_tensors(
        rng,
        layer_types=["full_attention"],
        gated_attn=False,
        qk_norm=True,
    )
    # Use F16 scales/biases on q_proj to test F16->BF16 affine plane conversion
    _add_module(
        tensors,
        "model.layers.0.self_attn.q_proj",
        _affine_triplet(rng, (512, 512), bits=4, group_size=64, f16_sb=True),
    )
    # Put o_proj in unquantized BF16
    for suf in (".weight", ".scales", ".biases"):
        tensors.pop(f"model.layers.0.self_attn.o_proj{suf}", None)
    w_o = rng.normal(0, 0.05, size=(512, 512)).astype(np.float32)
    tensors["model.layers.0.self_attn.o_proj.weight"] = ("BF16", [512, 512], _bf16_bytes(w_o))

    # Put down_proj in 6-bit g128 WITHOUT listing it in config["quantization"]!
    for suf in (".weight", ".scales", ".biases"):
        tensors.pop(f"model.layers.0.mlp.down_proj{suf}", None)
    _add_module(
        tensors,
        "model.layers.0.mlp.down_proj",
        _affine_triplet(rng, (512, 1024), bits=6, group_size=128),
    )

    # Configure lm_head per head_kind WITHOUT per-layer override in config["quantization"]!
    for suf in (".weight", ".scales", ".biases"):
        tensors.pop(f"lm_head{suf}", None)
    head_ref_f32 = rng.normal(0, 0.1, size=(256, 512)).astype(np.float32)
    if head_kind == "bf16":
        tensors["lm_head.weight"] = ("BF16", [256, 512], _bf16_bytes(head_ref_f32))
    elif head_kind == "fp16":
        tensors["lm_head.weight"] = ("F16", [256, 512], _f16_bytes(head_ref_f32))
    else:
        _add_module(
            tensors,
            "lm_head",
            _affine_triplet(rng, (256, 512), bits=8, group_size=64),
        )

    # Global config["quantization"] ONLY mentions 4-bit g64!
    cfg = {
        "model_type": "qwen3",
        "hidden_size": 512,
        "intermediate_size": 1024,
        "num_hidden_layers": 1,
        "vocab_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 128,
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    ckpt = _write_checkpoint(tmp_path / f"mixed_{head_kind}", cfg, tensors)
    insp = m5_artifact.inspect_source_checkpoint(ckpt, nax_pretile=True)
    assert insp["eligible"] is True
    assert insp["elapsed_ms"] < 5.0
    plan_map = {e["name"]: e for e in insp["repack_plan"]["entries"]}
    assert plan_map["model.layers.0.mlp.down_proj"]["format"] == "oq_affine_q6_g128_bf16_v1"
    assert plan_map["model.layers.0.self_attn.o_proj.weight"]["format"] == "direct_bf16_v1"
    assert plan_map["model.layers.0.self_attn.o_proj.weight"]["category"] == "rowsplit_or_direct"

    out_miv = tmp_path / f"mixed_{head_kind}.miv"
    src = m5_artifact.SafeTensorSource(ckpt)
    try:
        m5_artifact.write_artifact(
            src,
            out_miv,
            f"test/mixed-{head_kind}",
            "auto",
            artifact_version=m5_artifact.VERSION_V2,
            nax_pretile=True,
            jobs=2,
        )
    finally:
        src.close()

    _, man = m5_artifact.read_artifact(out_miv, verify_payload=True)
    objs = {o["name"]: o for o in man["objects"]}
    assert objs["model.layers.0.mlp.down_proj"]["format"] == "oq_affine_q6_g128_bf16_v1"
    assert objs["model.layers.0.mlp.down_proj"]["layout"] == "oq_nax_lane_g128_v2"
    assert objs["model.layers.0.self_attn.o_proj.weight"]["format"] == "direct_bf16_v1"
    if head_kind in ("bf16", "fp16"):
        assert objs["lm_head.weight"]["format"] == "direct_bf16_v1"
        assert objs["lm_head.weight"]["planes"][0]["bytes"] == 256 * 512 * 2
    else:
        assert objs["lm_head"]["format"] == "oq_affine_q8_g64_bf16_v1"
        assert objs["lm_head"]["layout"] == "oq_nax_lane_g64_v2"
    # Verify decoded values of F16-scaled q_proj are finite
    dec_q = m5_artifact.decode_artifact_row(
        out_miv, man, "model.layers.0.self_attn.q_proj", 0, 0, 64
    )
    assert len(dec_q) == 64 and all(np.isfinite(dec_q))


def test_scenario4_pure_bf16_and_fp16_without_quantization_config(tmp_path: Path) -> None:
    rng = np.random.default_rng(4099)
    h, inter, vocab = 512, 1024, 256
    tensors: dict[str, tuple[str, list[int], bytes]] = {
        "model.embed_tokens.weight": ("F16", [vocab, h], _f16_bytes(rng.normal(0, 0.05, size=(vocab, h)))),
        "model.norm.weight": ("BF16", [h], _bf16_bytes(np.ones(h, dtype=np.float32))),
        "lm_head.weight": ("F16", [vocab, h], _f16_bytes(rng.normal(0, 0.05, size=(vocab, h)))),
        "model.layers.0.input_layernorm.weight": ("BF16", [h], _bf16_bytes(np.ones(h, dtype=np.float32))),
        "model.layers.0.post_attention_layernorm.weight": ("BF16", [h], _bf16_bytes(np.ones(h, dtype=np.float32))),
        "model.layers.0.self_attn.q_proj.weight": ("BF16", [h, h], _bf16_bytes(rng.normal(0, 0.05, size=(h, h)))),
        "model.layers.0.self_attn.k_proj.weight": ("BF16", [256, h], _bf16_bytes(rng.normal(0, 0.05, size=(256, h)))),
        "model.layers.0.self_attn.v_proj.weight": ("BF16", [256, h], _bf16_bytes(rng.normal(0, 0.05, size=(256, h)))),
        "model.layers.0.self_attn.o_proj.weight": ("F16", [h, h], _f16_bytes(rng.normal(0, 0.05, size=(h, h)))),
        "model.layers.0.mlp.gate_proj.weight": ("BF16", [inter, h], _bf16_bytes(rng.normal(0, 0.05, size=(inter, h)))),
        "model.layers.0.mlp.up_proj.weight": ("BF16", [inter, h], _bf16_bytes(rng.normal(0, 0.05, size=(inter, h)))),
        "model.layers.0.mlp.down_proj.weight": ("F16", [h, inter], _f16_bytes(rng.normal(0, 0.05, size=(h, inter)))),
    }
    cfg = {
        "model_type": "community_unquantized_llm",
        "hidden_size": h,
        "intermediate_size": inter,
        "num_hidden_layers": 1,
        "vocab_size": vocab,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
    }
    ckpt = _write_checkpoint(tmp_path / "pure_bf16_f16", cfg, tensors)
    insp = m5_artifact.inspect_source_checkpoint(ckpt)
    assert insp["eligible"] is True
    assert insp["detected_profile"] == "direct_bf16"
    assert insp["elapsed_ms"] < 5.0

    out_miv = tmp_path / "pure_bf16.miv"
    src = m5_artifact.SafeTensorSource(ckpt)
    try:
        rep = m5_artifact.write_artifact(src, out_miv, "test/pure-bf16", "auto", nax_pretile=True)
    finally:
        src.close()
    assert rep["source_profile"] == "direct_bf16"
    _, man = m5_artifact.read_artifact(out_miv, verify_payload=True)
    assert all(o["format"] == "direct_bf16_v1" for o in man["objects"])


# ---------------------------------------------------------------------------
# 5. VLM checkpoint with visual.* / projector / audio stripped + companion MTP
# ---------------------------------------------------------------------------


def test_scenario5_vlm_non_text_stripping_and_companion_mtp(tmp_path: Path) -> None:
    rng = np.random.default_rng(5005)
    tensors = _build_full_text_tensors(
        rng,
        prefix="language_model.model",
        layer_types=["full_attention"],
        gated_attn=True,
        qk_norm=True,
    )
    # Add VLM non-text tensors (including 5D Conv3D F32 & additive vision .bias!)
    non_text = {
        "aligner.mlp.0.weight": ("F16", [512, 1024], _f16_bytes(np.zeros((512, 1024)))),
        "audio_tower.conv1.weight": ("BF16", [64, 80, 3], _bf16_bytes(np.zeros((64, 80, 3)))),
        "model.visual.blocks.0.attn.qkv.bias": ("F32", [384], np.zeros(384, dtype="<f4").tobytes()),
        "multi_modal_projector.linear_1.weight": ("F16", [512, 512], _f16_bytes(np.zeros((512, 512)))),
        "visual.patch_embed.proj.weight": (
            "F32",
            [16, 3, 2, 4, 4],
            np.zeros((16, 3, 2, 4, 4), dtype="<f4").tobytes(),
        ),
    }
    tensors.update(non_text)
    # Add companion MTP projection
    _add_module(
        tensors,
        "mtp.fc",
        _affine_triplet(rng, (512, 1024), bits=4, group_size=64),
    )

    cfg = {
        "model_type": "qwen3_vl_custom",
        "vision_config": {"hidden_size": 1280},
        "text_config": {
            "model_type": "qwen3_text_custom",
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
    ckpt = _write_checkpoint(tmp_path / "vlm_src", cfg, tensors)
    insp = m5_artifact.inspect_source_checkpoint(ckpt, nax_pretile=True)
    assert insp["eligible"] is True
    assert insp["elapsed_ms"] < 5.0
    expected_skipped = sorted(non_text.keys())
    assert insp["skipped_tensors"] == expected_skipped
    assert insp["repack_plan"]["category_counts"]["skip_non_text"] == len(expected_skipped)
    assert insp["repack_plan"]["category_counts"]["companion_mtp"] == 1

    out_miv = tmp_path / "out_vlm" / "model.miv"
    src = m5_artifact.SafeTensorSource(ckpt)
    try:
        rep = m5_artifact.write_artifact(
            src, out_miv, "test/vlm-strip", "auto", nax_pretile=True, jobs=2
        )
    finally:
        src.close()

    _, man = m5_artifact.read_artifact(out_miv, verify_payload=True)
    conv_json = json.loads(
        (out_miv.parent / "model.miv.conversion.json").read_text(encoding="utf-8")
    )
    obj_names = {o["name"] for o in man["objects"]}
    assert "mtp.fc" in obj_names
    assert not any(
        n.startswith(("visual.", "model.visual.", "multi_modal_projector.", "aligner.", "audio_tower."))
        for n in obj_names
    )
    assert man["model"]["skipped_non_text_count"] == len(expected_skipped)
    for doc in (rep, conv_json):
        assert doc["skipped_tensors"] == expected_skipped
        assert doc["repack_plan_summary"]["skip_non_text"] == len(expected_skipped)
        assert doc["repack_plan_summary"]["companion_mtp"] == 1
        assert "oq_affine_q4_g64_bf16_v1" in doc["repack_plan_summary"]["format_histogram"]


# ---------------------------------------------------------------------------
# 6. DFlash2 structural contract & target compatibility (no 248320/repo lock)
# ---------------------------------------------------------------------------


def test_scenario6_dflash2_structural_and_target_compatibility(tmp_path: Path) -> None:
    rng = np.random.default_rng(6006)
    draft_cfg = {
        "architectures": ["DFlash2DraftModel"],
        "model_type": "community_draft_arch",
        "is_causal": False,
        "hidden_size": 512,
        "intermediate_size": 1024,
        "vocab_size": 32000,
        "num_hidden_layers": 2,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "num_target_layers": 4,
        "dflash_config": {
            "block_size": 8,
            "conv_kernel_size": 2,
            "conv_group_size": 16,
            "selector_rank": 16,
            "selector_top_k": 32,
            "mask_token_id": 31999,
            "target_layer_ids": [1, 3],
        },
    }
    draft_tensors = {
        "norm.weight": ("BF16", [512], _bf16_bytes(np.ones(512, dtype=np.float32))),
        "layers.0.self_attn.q_norm.weight": ("BF16", [64], _bf16_bytes(np.ones(64, dtype=np.float32))),
        "layers.0.self_attn.k_norm.weight": ("BF16", [64], _bf16_bytes(np.ones(64, dtype=np.float32))),
        "layers.1.self_attn.q_norm.weight": ("BF16", [64], _bf16_bytes(np.ones(64, dtype=np.float32))),
        "layers.1.self_attn.k_norm.weight": ("BF16", [64], _bf16_bytes(np.ones(64, dtype=np.float32))),
    }
    draft_dir = _write_checkpoint(tmp_path / "my_custom_dflash2", draft_cfg, draft_tensors)

    target_good = tmp_path / "target_good"
    target_good.mkdir()
    (target_good / "config.json").write_text(
        json.dumps(
            {
                "model_type": "custom_llm",
                "hidden_size": 512,
                "vocab_size": 32000,
                "num_hidden_layers": 4,
            }
        ),
        encoding="utf-8",
    )

    ok_res = m5_artifact.inspect_source_checkpoint(draft_dir, target_model_dir=target_good)
    assert ok_res["eligible"] is True
    assert ok_res["topology"] == "dflash2_draft"
    assert ok_res["graph_gate"]["target_compatible"] is True
    assert ok_res["elapsed_ms"] < 5.0

    # Target vocab mismatch -> rejected
    target_bad_vocab = tmp_path / "target_bad_vocab"
    target_bad_vocab.mkdir()
    (target_bad_vocab / "config.json").write_text(
        json.dumps({"hidden_size": 512, "vocab_size": 248320, "num_hidden_layers": 4}),
        encoding="utf-8",
    )
    bad_v = m5_artifact.inspect_source_checkpoint(draft_dir, target_model_dir=target_bad_vocab)
    assert bad_v["eligible"] is False
    assert bad_v["blocker_code"] == "dflash2_target_vocab_mismatch"

    # Target layers too shallow for target_layer_ids=[1, 3] -> rejected
    target_shallow = tmp_path / "target_shallow"
    target_shallow.mkdir()
    (target_shallow / "config.json").write_text(
        json.dumps({"hidden_size": 512, "vocab_size": 32000, "num_hidden_layers": 3}),
        encoding="utf-8",
    )
    bad_l = m5_artifact.inspect_source_checkpoint(draft_dir, target_model_dir=target_shallow)
    assert bad_l["eligible"] is False
    assert bad_l["blocker_code"] == "dflash2_target_layer_out_of_range"


# ---------------------------------------------------------------------------
# 7. < 5 ms Negative Blockers (MLA, Mamba2, AWQ, GPTQ, FP8, affine gs=32)
#    with zero partial .miv / .conversion.json files!
# ---------------------------------------------------------------------------


def _assert_blocked_with_zero_partial_files(
    ckpt: Path,
    out_dir: Path,
    expected_code_substr: str,
    *,
    expected_tensor: str | None = None,
) -> dict[str, Any]:
    insp = m5_artifact.inspect_source_checkpoint(ckpt)
    assert insp["eligible"] is False
    assert insp["elapsed_ms"] < 5.0, f"blocker check took {insp['elapsed_ms']} ms"
    assert expected_code_substr in str(insp["blocker_code"])
    if expected_tensor is not None:
        assert insp["first_blocker"] is not None
        assert insp["first_blocker"]["tensor"] == expected_tensor
        assert expected_tensor in insp["reason"]

    out_miv = out_dir / f"{ckpt.name}.miv"
    src = m5_artifact.SafeTensorSource(ckpt)
    try:
        with pytest.raises(ValueError) as exc_info:
            m5_artifact.write_artifact(src, out_miv, "test/blocked", "auto")
        if expected_tensor is not None:
            assert expected_tensor in str(exc_info.value)
    finally:
        src.close()

    assert not out_miv.exists()
    if out_dir.exists():
        assert list(out_dir.iterdir()) == []
    return insp


def test_scenario7_fast_negative_blockers_and_zero_partial_files(tmp_path: Path) -> None:
    rng = np.random.default_rng(7007)
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
    base_tensors = _build_full_text_tensors(
        rng, layer_types=["full_attention"], gated_attn=False, qk_norm=False
    )

    # 1) DeepSeek MLA
    mla_cfg = {**base_cfg, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
    mla_tensors = dict(base_tensors)
    mla_tensors["model.layers.0.self_attn.kv_a_proj_with_mqa.weight"] = (
        "BF16",
        [576, 512],
        _bf16_bytes(np.zeros((576, 512), dtype=np.float32)),
    )
    mla_ckpt = _write_checkpoint(tmp_path / "blk_mla", mla_cfg, mla_tensors)
    _assert_blocked_with_zero_partial_files(
        mla_ckpt, tmp_path / "out_mla", "mla"
    )

    # 2) Mamba2 / SSM
    mamba_cfg = {**base_cfg, "layer_types": ["mamba"], "ssm_state_size": 128}
    mamba_tensors = dict(base_tensors)
    mamba_tensors["model.layers.0.mamba.in_proj.weight"] = (
        "BF16",
        [1024, 512],
        _bf16_bytes(np.zeros((1024, 512), dtype=np.float32)),
    )
    mamba_ckpt = _write_checkpoint(tmp_path / "blk_mamba", mamba_cfg, mamba_tensors)
    _assert_blocked_with_zero_partial_files(
        mamba_ckpt, tmp_path / "out_mamba", "mamba"
    )

    # 3) AWQ (.qweight / .qzeros)
    awq_tensors = dict(base_tensors)
    for suf in (".weight", ".scales", ".biases"):
        awq_tensors.pop(f"model.layers.0.self_attn.q_proj{suf}", None)
    awq_tensors["model.layers.0.self_attn.q_proj.qweight"] = (
        "I32",
        [512, 64],
        np.zeros((512, 64), dtype="<i4").tobytes(),
    )
    awq_tensors["model.layers.0.self_attn.q_proj.qzeros"] = (
        "I32",
        [4, 64],
        np.zeros((4, 64), dtype="<i4").tobytes(),
    )
    awq_tensors["model.layers.0.self_attn.q_proj.scales"] = (
        "F16",
        [4, 512],
        _f16_bytes(np.ones((4, 512))),
    )
    awq_cfg = {
        **{k: v for k, v in base_cfg.items() if k != "quantization"},
        "quantization_config": {"quant_method": "awq", "bits": 4, "group_size": 128},
    }
    awq_ckpt = _write_checkpoint(tmp_path / "blk_awq", awq_cfg, awq_tensors)
    _assert_blocked_with_zero_partial_files(
        awq_ckpt,
        tmp_path / "out_awq",
        "awq",
        expected_tensor="model.layers.0.self_attn.q_proj.qweight",
    )

    # 4) GPTQ (.g_idx)
    gptq_tensors = dict(awq_tensors)
    gptq_tensors["model.layers.0.self_attn.q_proj.g_idx"] = (
        "I32",
        [512],
        np.arange(512, dtype="<i4").tobytes(),
    )
    gptq_cfg = {
        **{k: v for k, v in base_cfg.items() if k != "quantization"},
        "quantization_config": {"quant_method": "gptq", "bits": 4, "group_size": 128},
    }
    gptq_ckpt = _write_checkpoint(tmp_path / "blk_gptq", gptq_cfg, gptq_tensors)
    _assert_blocked_with_zero_partial_files(
        gptq_ckpt,
        tmp_path / "out_gptq",
        "gptq",
        expected_tensor="model.layers.0.self_attn.q_proj.qweight",
    )

    # 5) FP8 (F8_E4M3FN + .weight_scale_inv)
    fp8_tensors = dict(base_tensors)
    for suf in (".weight", ".scales", ".biases"):
        fp8_tensors.pop(f"model.layers.0.self_attn.q_proj{suf}", None)
    fp8_tensors["model.layers.0.self_attn.q_proj.weight"] = (
        "F8_E4M3FN",
        [512, 512],
        bytes(512 * 512),
    )
    fp8_tensors["model.layers.0.self_attn.q_proj.weight_scale_inv"] = (
        "F32",
        [4, 4],
        np.ones((4, 4), dtype="<f4").tobytes(),
    )
    fp8_cfg = {
        **{k: v for k, v in base_cfg.items() if k != "quantization"},
        "quantization_config": {"quant_method": "fp8"},
    }
    fp8_ckpt = _write_checkpoint(tmp_path / "blk_fp8", fp8_cfg, fp8_tensors)
    _assert_blocked_with_zero_partial_files(
        fp8_ckpt,
        tmp_path / "out_fp8",
        "fp8",
        expected_tensor="model.layers.0.self_attn.q_proj.weight",
    )

    # 6) Affine group_size=32 on a single layer even when global config says gs=64!
    gs32_tensors = dict(base_tensors)
    _add_module(
        gs32_tensors,
        "model.layers.0.self_attn.o_proj",
        _affine_triplet(rng, (512, 512), bits=4, group_size=32),
    )
    gs32_ckpt = _write_checkpoint(tmp_path / "blk_gs32", base_cfg, gs32_tensors)
    _assert_blocked_with_zero_partial_files(
        gs32_ckpt,
        tmp_path / "out_gs32",
        "affine_group_size",
        expected_tensor="model.layers.0.self_attn.o_proj.weight",
    )
