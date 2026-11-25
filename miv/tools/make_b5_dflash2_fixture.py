#!/usr/bin/env python3
"""Generate a small true-DFlash2 BF16 artifact and dflash-mlx oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import m5_artifact
import mlx.core as mx
import numpy as np
from dflash_mlx.model import DFlash2DraftModel, DFlashDraftModelArgs
from mlx.utils import tree_flatten


def deterministic(shape: tuple[int, ...], seed: int, scale: float = 0.08) -> mx.array:
    count = int(np.prod(shape))
    values = np.arange(count, dtype=np.float32)
    values = np.sin(values * 0.173 + seed * 0.619) * scale
    return mx.array(values.reshape(shape)).astype(mx.bfloat16)


def config() -> dict:
    return {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "dflash_config": {
            "block_size": 8,
            "conv_group_size": 16,
            "conv_kernel_size": 2,
            "mask_token_id": 63,
            "selector_rank": 8,
            "selector_top_k": 8,
            "target_layer_ids": [0, 2],
        },
        "dtype": "bfloat16",
        "head_dim": 8,
        "hidden_size": 64,
        "intermediate_size": 128,
        "is_causal": False,
        "layer_types": ["sliding_attention", "sliding_attention"],
        "max_position_embeddings": 1024,
        "model_type": "qwen3",
        "num_attention_heads": 8,
        "num_hidden_layers": 2,
        "num_key_value_heads": 2,
        "num_target_layers": 4,
        "rms_norm_eps": 1.0e-6,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "sliding_window": 16,
        "tie_word_embeddings": False,
        "use_cache": True,
        "vocab_size": 64,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)

    cfg = config()
    (source / "config.json").write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")
    (source / "REVISION").write_text("b5-dflash2-fixture-v1\n")

    model_args = DFlashDraftModelArgs.from_dict(cfg)
    model = DFlash2DraftModel(model_args)
    parameters: dict[str, mx.array] = {}
    for index, (name, value) in enumerate(tree_flatten(model.parameters())):
        parameters[name] = deterministic(tuple(value.shape), index + 1)
    model.load_weights(list(parameters.items()))

    stored = dict(parameters)
    for name in ("predecessor_codebook", "successor_codebook"):
        model_name = f"candidate_selector.{name}.weight"
        stored[f"candidate_selector.{name}"] = stored.pop(model_name)

    embedding = deterministic((cfg["vocab_size"], cfg["hidden_size"]), 101, 0.12)
    proposal_head = deterministic((cfg["vocab_size"], cfg["hidden_size"]), 103, 0.15)
    stored["shared_embedding.weight"] = embedding
    stored["proposal_head.weight"] = proposal_head
    mx.save_safetensors(str(source / "model.safetensors"), stored)

    context_tokens = 3
    target_width = len(cfg["dflash_config"]["target_layer_ids"]) * cfg["hidden_size"]
    target_features = deterministic((1, context_tokens, target_width), 107, 0.11)
    anchor = 7
    mask = cfg["dflash_config"]["mask_token_id"]
    token_ids = mx.array([[anchor] + [mask] * 7], dtype=mx.int32)
    noise_embedding = embedding[token_ids]
    hidden = model(noise_embedding=noise_embedding, target_hidden=target_features)
    logits = hidden @ proposal_head.T
    proposals, candidate_ids, _ = model.select_candidates(
        hidden[:, 1:], logits[:, 1:], mx.array([anchor], dtype=mx.int32), 0.0
    )
    mx.eval(hidden, logits, proposals, candidate_ids)

    def floats(value: mx.array) -> list:
        return np.asarray(value.astype(mx.float32)).tolist()

    oracle = {
        "schema": "omlx-m5-b5-dflash2-oracle-v1",
        "config": cfg,
        "context_tokens": context_tokens,
        "anchor_token": anchor,
        "target_features": floats(target_features[0]),
        "noise_embedding": floats(noise_embedding[0]),
        "hidden": floats(hidden[0]),
        "logits": floats(logits[0]),
        "proposal_hidden": floats(hidden[0, 1:]),
        "proposal_logits": floats(logits[0, 1:]),
        "candidate_ids": np.asarray(candidate_ids[0]).tolist(),
        "proposals": np.asarray(proposals[0]).tolist(),
    }
    (output / "oracle.json").write_text(json.dumps(oracle, indent=2, sort_keys=True) + "\n")

    artifact_path = output / "dflash2-fixture.omlxm5"
    source_model = m5_artifact.SafeTensorSource(source, "b5-dflash2-fixture-v1")
    try:
        report = m5_artifact.write_artifact(
            source_model,
            artifact_path,
            "fixture/Qwen3.8-27B-DFlash2",
            "dflash2_bf16",
        )
    finally:
        source_model.close()
    (output / "fixture.stamp").write_text(report["artifact_sha256"] + "\n")
    print(json.dumps({"artifact": str(artifact_path), "oracle": str(output / "oracle.json")}))


if __name__ == "__main__":
    main()
