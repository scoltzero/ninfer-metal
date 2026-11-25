#!/usr/bin/env python3
"""Create shape-valid Qwen weights and an independent stock MLX model oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import m5_artifact
import mlx.core as mx
import numpy as np
from mlx import nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs
from qwen_reference import QwenReference


def build(
    output: Path,
    mode: str,
    moe: bool = False,
    hf_layout: bool = False,
    mixed: bool = False,
) -> None:
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3_5_text",
        "dtype": "bfloat16",
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": 2,
        "vocab_size": 257,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 256,
        "linear_num_key_heads": 1,
        "linear_num_value_heads": 2,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "full_attention_interval": 2,
        "max_position_embeddings": 2048,
        "rms_norm_eps": 1e-6,
        "layer_types": ["linear_attention", "full_attention"],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.25,
        },
    }
    if moe:
        config.update(
            num_experts=256,
            num_experts_per_tok=8,
            intermediate_size=128,
            moe_intermediate_size=128,
            shared_expert_intermediate_size=256,
            norm_topk_prob=True,
        )
    model = TextModel(TextModelArgs.from_dict(config))
    rng = np.random.default_rng(230922)
    weights = []
    for name, value in tree_flatten(model.parameters()):
        shape = tuple(value.shape)
        if "norm.weight" in name:
            array = 1 + rng.normal(0, 0.1, shape)
        elif name.endswith("A_log"):
            array = rng.uniform(-0.5, 0.5, shape)
        elif name.endswith("dt_bias"):
            array = rng.uniform(-1.5, -0.5, shape)
        else:
            scale = 0.12 if "conv1d" in name else 0.6 / np.sqrt(shape[-1])
            array = rng.normal(0, scale, shape)
        weights.append((name, mx.array(array.astype(np.float32)).astype(mx.bfloat16)))
    model.load_weights(weights)
    protected = {}
    if mixed:
        formats = [(4, 64), (5, 64), (5, 128), (6, 64), (6, 128), (8, 64), (8, 128)]
        modules = sorted(
            name.removesuffix(".weight")
            for name, value in weights
            if value.ndim >= 2 and "conv1d" not in name
        )
        for i, name in enumerate(modules):
            bits, group = formats[i % len(formats)]
            protected[name] = {"bits": bits, "group_size": group, "mode": "affine"}
    if moe:
        for layer in range(config["num_hidden_layers"]):
            for role in ("gate", "shared_expert_gate"):
                protected[f"model.layers.{layer}.mlp.{role}"] = {
                    "bits": 8,
                    "group_size": 64,
                    "mode": "affine",
                }

    def quantize_module(path, module):
        return protected.get(path, hasattr(module, "to_quantized"))

    nn.quantize(
        model,
        group_size=64 if mode == "affine" else 32,
        bits=4,
        mode=mode,
        class_predicate=quantize_module,
    )
    model.eval()
    stored = {
        "language_model." + name: value
        for name, value in tree_flatten(model.parameters())
    }
    if hf_layout:
        for name, value in list(stored.items()):
            if name.endswith("conv1d.weight"):
                stored[name] = mx.swapaxes(value, -1, -2)
            elif name.endswith(
                (
                    ".input_layernorm.weight",
                    ".post_attention_layernorm.weight",
                    "model.norm.weight",
                    ".q_norm.weight",
                    ".k_norm.weight",
                )
            ):
                stored[name] = value - mx.array(1, dtype=mx.bfloat16)
    mx.save_safetensors(str(source / "model.safetensors"), stored)
    quantization = {
        "bits": 4,
        "group_size": 64 if mode == "affine" else 32,
        "mode": mode,
    }
    quantization.update(
        {"language_model." + name: spec for name, spec in protected.items()}
    )
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "text_config": config,
                "quantization": quantization,
            },
            indent=2,
        )
        + "\n"
    )
    (source / "REVISION").write_text("qwen-execution-fixture-v1\n")
    reader = m5_artifact.SafeTensorSource(source, "qwen-execution-fixture-v1")
    try:
        m5_artifact.write_artifact(
            reader,
            output / "model.omlxm5",
            "fixture/qwen-execution-" + mode,
            "oq_affine" if mode == "affine" else "mlx_mxfp4",
        )
    finally:
        reader.close()

    cases = []
    reference = QwenReference(
        config,
        {
            name.removeprefix("language_model."): np.asarray(
                value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
            )
            for name, value in stored.items()
        },
        mode,
        protected,
        norm_offset=1 if hf_layout else 0,
    )
    extents = (1, 3, 4, 8, 17, 33, 129)
    if mixed or (not moe and not hf_layout):
        extents += (512, 1024)
    for count in extents:
        tokens = [(i * 71 + 256) % config["vocab_size"] for i in range(count)]
        hidden = model.model(mx.array([tokens], dtype=mx.int32))
        logits = model.lm_head(hidden)
        mx.eval(hidden, logits)
        # Sequential execution uses the actual MLX KV and recurrent cache types.
        cache = model.make_cache()
        sequential = [
            model(mx.array([[t]], dtype=mx.int32), cache=cache) for t in tokens
        ]
        sequential = mx.concatenate(sequential, axis=1)
        mx.eval(sequential)
        as_float = lambda x: np.asarray(x.astype(mx.float32)).tolist()
        reference_hidden, reference_logits, reference_features = reference(tokens)
        cases.append(
            {
                "tokens": tokens,
                "hidden": as_float(hidden[0]),
                "logits": as_float(logits[0]),
                "sequential_logits": as_float(sequential[0]),
                "reference_hidden": reference_hidden.tolist(),
                "reference_logits": reference_logits.tolist(),
                "reference_features": reference_features.tolist(),
                "reference_routes": np.stack(reference.route_history, axis=1).tolist()
                if moe
                else [],
            }
        )
    (output / "oracle.json").write_text(
        json.dumps(
            {
                "schema": "omlx-qwen-execution-oracle-v1",
                "oracle": "independent NumPy FP64 model; stock MLX measured separately",
                "arithmetic": "stored quantized values; BF16 public Op outputs; FP32 recurrent state",
                "config": config,
                "cases": cases,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    for mode in ("affine", "mxfp4"):
        build(args.output_dir / mode, mode)
        build(args.output_dir / ("moe-" + mode), mode, moe=True)
    build(args.output_dir / "hf-mxfp4", "mxfp4", hf_layout=True)
    build(args.output_dir / "mixed-oq", "affine", mixed=True)
    build(args.output_dir / "moe-mixed-oq", "affine", moe=True, mixed=True)
    (args.output_dir / "fixture.stamp").write_text("qwen-execution-fixture-v1\n")


if __name__ == "__main__":
    main()
