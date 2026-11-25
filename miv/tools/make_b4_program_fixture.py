#!/usr/bin/env python3
"""Create a tiny deterministic Qwen oQ artifact for B4 Program ownership tests."""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import m5_artifact


def zeros(dtype: str, shape: tuple[int, ...]) -> bytes:
    widths = {"U32": 4, "BF16": 2}
    elements = 1
    for dimension in shape:
        elements *= dimension
    return bytes(elements * widths[dtype])


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, tuple[int, ...], bytes]]
) -> None:
    header: dict[str, object] = {"__metadata__": {"fixture": "b4-qwen-program-v1"}}
    payload = bytearray()
    for name in sorted(tensors):
        dtype, shape, data = tensors[name]
        begin = len(payload)
        payload.extend(data)
        header[name] = {
            "dtype": dtype,
            "shape": list(shape),
            "data_offsets": [begin, len(payload)],
        }
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)


def add_quantized(
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]],
    name: str,
    rows: int,
    *,
    quant: str = "oq",
    experts: int = 0,
) -> None:
    leading = (experts, rows) if experts else (rows,)
    packed_rows = (experts if experts else 1) * rows
    packed_word = 0x88888888 if quant == "oq" else 0x22222222
    packed = struct.pack("<I", packed_word) * (packed_rows * 8)
    tensors[f"{name}.weight"] = (
        "U32",
        (*leading, 8),
        packed,
    )
    if quant == "oq":
        one_bf16 = struct.pack("<H", 0x3F80) * packed_rows
        tensors[f"{name}.scales"] = (
            "BF16",
            (*leading, 1),
            one_bf16,
        )
        tensors[f"{name}.biases"] = (
            "BF16",
            (*leading, 1),
            zeros("BF16", (*leading, 1)),
        )
    else:
        tensors[f"{name}.scales"] = (
            "U8",
            (*leading, 2),
            bytes([127]) * (packed_rows * 2),
        )


def write_artifact_fixture(
    source: Path, artifact_path: Path, source_repo: str, profile: str
) -> None:
    model = m5_artifact.SafeTensorSource(source, "b4-program-fixture-v1")
    try:
        m5_artifact.write_artifact(model, artifact_path, source_repo, profile)
    finally:
        model.close()


def build_ar_fixture(output_dir: Path, architecture: str, quant: str) -> None:
    key = f"ar-{architecture}-{quant}"
    source = output_dir / key
    source.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for layer in range(2):
        prefix = f"language_model.model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            name = f"{prefix}.{norm}.weight"
            tensors[name] = ("BF16", (64,), zeros("BF16", (64,)))
        if layer == 0:
            for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
                add_quantized(
                    tensors,
                    f"{prefix}.self_attn.{projection}",
                    64,
                    quant=quant,
                )
            for norm in ("q_norm", "k_norm"):
                name = f"{prefix}.self_attn.{norm}.weight"
                tensors[name] = ("BF16", (64,), zeros("BF16", (64,)))
        else:
            for projection in (
                "in_proj_qkv",
                "in_proj_z",
                "in_proj_a",
                "in_proj_b",
                "out_proj",
            ):
                add_quantized(
                    tensors,
                    f"{prefix}.linear_attn.{projection}",
                    64,
                    quant=quant,
                )
            for name, shape in {
                f"{prefix}.linear_attn.conv1d.weight": (64, 1, 4),
                f"{prefix}.linear_attn.A_log": (64,),
                f"{prefix}.linear_attn.dt_bias": (64,),
                f"{prefix}.linear_attn.norm.weight": (64,),
            }.items():
                tensors[name] = ("BF16", shape, zeros("BF16", shape))

        mlp = f"{prefix}.mlp"
        if architecture == "dense":
            for projection in ("gate_proj", "up_proj", "down_proj"):
                add_quantized(tensors, f"{mlp}.{projection}", 64, quant=quant)
        else:
            add_quantized(tensors, f"{mlp}.gate", 8, quant=quant)
            for projection in ("gate_proj", "up_proj", "down_proj"):
                add_quantized(
                    tensors,
                    f"{mlp}.switch_mlp.{projection}",
                    64,
                    quant=quant,
                    experts=8,
                )
            add_quantized(tensors, f"{mlp}.shared_expert_gate", 1, quant=quant)
            for projection in ("gate_proj", "up_proj", "down_proj"):
                add_quantized(
                    tensors,
                    f"{mlp}.shared_expert.{projection}",
                    64,
                    quant=quant,
                )
    add_quantized(tensors, "language_model.model.embed_tokens", 64, quant=quant)
    add_quantized(tensors, "language_model.lm_head", 64, quant=quant)
    tensors["language_model.model.norm.weight"] = (
        "BF16",
        (64,),
        zeros("BF16", (64,)),
    )
    text_config = {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "intermediate_size": 64,
        "num_experts": 8 if architecture == "moe" else 0,
        "num_experts_per_tok": 2 if architecture == "moe" else 0,
        "shared_expert_intermediate_size": 64 if architecture == "moe" else 0,
        "vocab_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "linear_num_key_heads": 4,
        "linear_num_value_heads": 4,
        "linear_key_head_dim": 16,
        "linear_value_head_dim": 16,
        "linear_conv_kernel_dim": 4,
        "max_position_embeddings": 4096,
        "layer_types": ["full_attention", "linear_attention"],
        "rms_norm_eps": 1e-6,
    }
    quantization = (
        {"bits": 4, "group_size": 64, "mode": "affine"}
        if quant == "oq"
        else {"bits": 4, "group_size": 32, "mode": "mxfp4"}
    )
    config = {
        "model_type": "qwen3_5",
        "text_config": text_config,
        "quantization": quantization,
    }
    (source / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (source / "REVISION").write_text("b4-program-fixture-v1\n")
    write_safetensors(source / "model.safetensors", tensors)
    write_artifact_fixture(
        source,
        output_dir / f"{key}.omlxm5",
        f"local/{key}",
        "oq_affine" if quant == "oq" else "mlx_mxfp4",
    )


def build(output_dir: Path) -> None:
    source = output_dir / "source"
    source.mkdir(parents=True, exist_ok=True)
    prefix = "language_model.model.layers.0"
    tensors: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
    for name in (
        f"{prefix}.linear_attn.in_proj_qkv",
        f"{prefix}.linear_attn.in_proj_z",
        f"{prefix}.linear_attn.in_proj_a",
        f"{prefix}.linear_attn.in_proj_b",
        f"{prefix}.linear_attn.out_proj",
        f"{prefix}.mlp.gate_proj",
        f"{prefix}.mlp.up_proj",
        f"{prefix}.mlp.down_proj",
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    ):
        add_quantized(tensors, name, 64)
    direct = {
        f"{prefix}.input_layernorm.weight": (64,),
        f"{prefix}.linear_attn.conv1d.weight": (64, 1, 4),
        f"{prefix}.linear_attn.A_log": (64,),
        f"{prefix}.linear_attn.dt_bias": (64,),
        f"{prefix}.linear_attn.norm.weight": (64,),
        f"{prefix}.post_attention_layernorm.weight": (64,),
        "language_model.model.norm.weight": (64,),
    }
    for name, shape in direct.items():
        tensors[name] = ("BF16", shape, zeros("BF16", shape))

    config = {
        "model_type": "qwen3_5",
        "text_config": {
            "model_type": "qwen3_5_text",
            "dtype": "bfloat16",
            "hidden_size": 64,
            "num_hidden_layers": 1,
            "intermediate_size": 64,
            "num_experts": 0,
            "vocab_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "linear_num_key_heads": 4,
            "linear_num_value_heads": 4,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_conv_kernel_dim": 4,
            "max_position_embeddings": 4096,
            "layer_types": ["linear_attention"],
            "rms_norm_eps": 1e-6,
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
    }
    (source / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (source / "REVISION").write_text("b4-program-fixture-v1\n")
    write_safetensors(source / "model.safetensors", tensors)
    artifact_path = output_dir / "fixture.omlxm5"
    write_artifact_fixture(
        source,
        artifact_path,
        "local/b4-qwen-program-fixture",
        "oq_affine",
    )
    for architecture in ("dense", "moe"):
        for quant in ("oq", "mxfp4"):
            build_ar_fixture(output_dir, architecture, quant)
    (output_dir / "fixture.stamp").write_text("b4-qwen-program-fixture-v1\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    build(args.output_dir.resolve())


if __name__ == "__main__":
    main()
