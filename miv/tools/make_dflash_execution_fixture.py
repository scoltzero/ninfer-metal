#!/usr/bin/env python3
"""Companion-only BF16 fixture bound to real target embedding/head views."""

import argparse
import json
from pathlib import Path

import m5_artifact
import mlx.core as mx
import numpy as np
from dflash_mlx.model import DFlash2DraftModel, DFlashDraftModelArgs
from make_b5_dflash2_fixture import config as base_config
from mlx.utils import tree_flatten
from qwen_reference import QwenReference, bf16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir
    source = out / "source"
    source.mkdir(parents=True, exist_ok=True)
    cfg = base_config()
    cfg.update(
        hidden_size=128,
        intermediate_size=256,
        vocab_size=257,
        head_dim=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_target_layers=2,
        max_position_embeddings=4096,
        sliding_window=64,
    )
    cfg["dflash_config"].update(target_layer_ids=[0, 1], mask_token_id=256)
    (source / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    (source / "REVISION").write_text("real-dflash-fixture-v1\n")
    rng = np.random.default_rng(220924)
    model = DFlash2DraftModel(DFlashDraftModelArgs.from_dict(cfg))
    weights = {}
    for name, value in tree_flatten(model.parameters()):
        shape = tuple(value.shape)
        if "norm.weight" in name:
            array = 1 + rng.normal(0, 0.05, shape)
        elif name.endswith("base_kernel"):
            array = rng.normal(0, 0.01, shape)
            array[:, 0, :] += 1
            array[:, 1, :] += 0.1
        else:
            scale = (0.03 if "kernel_projection" in name else 0.4) / np.sqrt(shape[-1])
            array = rng.normal(0, scale, shape)
        weights[name] = mx.array(array.astype(np.float32)).astype(mx.bfloat16)
    model.load_weights(list(weights.items()))
    stored = dict(weights)
    for name in ("predecessor_codebook", "successor_codebook"):
        stored[f"candidate_selector.{name}"] = stored.pop(
            f"candidate_selector.{name}.weight"
        )
    mx.save_safetensors(str(source / "model.safetensors"), stored)
    reader = m5_artifact.SafeTensorSource(source, "real-dflash-fixture-v1")
    try:
        m5_artifact.write_artifact(
            reader, out / "model.omlxm5", "fixture/real-dflash", "dflash2_bf16"
        )
    finally:
        reader.close()
    target_cfg = json.loads((args.target / "source/config.json").read_text())
    target_weights = {
        n.removeprefix("language_model."): np.asarray(
            v.astype(mx.float32) if v.dtype == mx.bfloat16 else v
        )
        for n, v in mx.load(str(args.target / "source/model.safetensors")).items()
    }
    ref = QwenReference(target_cfg["text_config"], target_weights, "affine")
    embed = ref.weight("model.embed_tokens")
    head = ref.weight("lm_head")
    cases = []
    for context in (1, 3, 32, 33, 65, 129, 1024, 2048, 2049, 4096):
        features = bf16(rng.normal(0, 0.2, (context, 256)))
        for k in range(1, 8) if context <= 1024 else (1, 7):
            noise = bf16(embed[[7] + [256] * k])
            hidden = model(
                noise_embedding=mx.array(noise[None].astype(np.float32)).astype(
                    mx.bfloat16
                ),
                target_hidden=mx.array(features[None].astype(np.float32)).astype(
                    mx.bfloat16
                ),
            )
            mx.eval(hidden)
            hidden_np = np.asarray(hidden[0].astype(mx.float32))
            logits = bf16(hidden_np.astype(np.float64) @ head.T)
            proposals, _, _ = model.select_candidates(
                hidden[:, 1:],
                mx.array(logits[None, 1:].astype(np.float32)).astype(mx.bfloat16),
                mx.array([7]),
                0.0,
            )
            mx.eval(proposals)
            cases.append(
                {
                    "context": context,
                    "k": k,
                    "features": features.tolist(),
                    "hidden": hidden_np.tolist(),
                    "logits": logits.tolist(),
                    "proposals": np.asarray(proposals[0]).tolist(),
                }
            )
    (out / "oracle.json").write_text(
        json.dumps(
            {
                "oracle": "dflash-mlx BF16; independently decoded target weights",
                "cases": cases,
            }
        )
    )
    (out / "fixture.stamp").write_text("real-dflash-fixture-v1\n")


if __name__ == "__main__":
    main()
