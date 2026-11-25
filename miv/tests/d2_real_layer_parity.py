#!/usr/bin/env python3
"""Validate complete real 27B GDN/attention layer slices against MLX."""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys


MIN_COSINE = 0.999
MAX_ABS_ERROR = 1e-2


def cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    denom = left_norm * right_norm
    return dot / denom if denom > 0.0 else float("nan")


def verify_parity(
    label: str,
    actual: list[float],
    expected: list[float],
    *,
    min_cosine: float = MIN_COSINE,
    max_abs_tol: float = MAX_ABS_ERROR,
) -> tuple[float, float]:
    if not actual or len(actual) != len(expected):
        raise AssertionError(
            f"{label}: length mismatch actual={len(actual)}, expected={len(expected)}"
        )
    maximum_error = 0.0
    for idx, (left, right) in enumerate(zip(actual, expected, strict=True)):
        if not (math.isfinite(left) and math.isfinite(right)):
            raise AssertionError(
                f"{label}: non-finite value at index {idx}: actual={left}, expected={right}"
            )
        diff = abs(left - right)
        if diff > maximum_error:
            maximum_error = diff
    similarity = cosine(actual, expected)
    if not (
        math.isfinite(similarity)
        and similarity >= min_cosine
        and math.isfinite(maximum_error)
        and maximum_error <= max_abs_tol
    ):
        raise AssertionError(
            f"{label}: cosine={similarity} (min {min_cosine}), "
            f"max_abs_error={maximum_error} (max {max_abs_tol})"
        )
    return similarity, maximum_error


class Oracle:
    def __init__(self, model_dir: pathlib.Path):
        import mlx.core as mx

        self.mx = mx
        self.model_dir = model_dir
        self.index = json.loads(
            (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
        )
        self.config = json.loads(
            (model_dir / "config.json").read_text(encoding="utf-8")
        )
        self.text = self.config["text_config"]
        self.quantization = self.config["quantization"]
        self.shards: dict[str, dict] = {}

    def tensor(self, name: str):
        shard_name = self.index["weight_map"][name]
        values = self.shards.setdefault(
            shard_name, self.mx.load(str(self.model_dir / shard_name))
        )
        return values[name]

    def qlinear(self, name: str, value):
        spec = self.quantization.get(name)
        bits = (
            spec.get("bits", self.quantization["bits"])
            if isinstance(spec, dict)
            else self.quantization["bits"]
        )
        return self.mx.quantized_matmul(
            value,
            self.tensor(f"{name}.weight"),
            self.tensor(f"{name}.scales"),
            self.tensor(f"{name}.biases"),
            group_size=64,
            bits=bits,
            mode="affine",
        )

    def rms(self, value, weight=None, epsilon: float | None = None):
        mx = self.mx
        epsilon = self.text["rms_norm_eps"] if epsilon is None else epsilon
        normalized = value * mx.rsqrt(
            mx.mean(value * value, axis=-1, keepdims=True) + epsilon
        )
        return normalized if weight is None else normalized * weight

    def mlp(self, prefix: str, value):
        mx = self.mx
        gate = self.qlinear(f"{prefix}.gate_proj", value)
        up = self.qlinear(f"{prefix}.up_proj", value)
        return self.qlinear(f"{prefix}.down_proj", mx.sigmoid(gate) * gate * up)

    def input_hidden(self):
        mx = self.mx
        width = self.text["hidden_size"]
        return mx.array(
            [(((index * 13) % 257) - 128) / 1024.0 for index in range(width)],
            dtype=mx.float32,
        )

    def attention(self, layer: int):
        mx = self.mx
        cfg = self.text
        prefix = f"language_model.model.layers.{layer}"
        attn = f"{prefix}.self_attn"
        hidden = self.input_hidden()
        normalized = self.rms(
            hidden, self.tensor(f"{prefix}.input_layernorm.weight").astype(mx.float32)
        )
        projected = self.qlinear(f"{attn}.q_proj", normalized).reshape(
            cfg["num_attention_heads"], 2 * cfg["head_dim"]
        )
        query, gate = mx.split(projected, 2, axis=-1)
        key = self.qlinear(f"{attn}.k_proj", normalized).reshape(
            cfg["num_key_value_heads"], cfg["head_dim"]
        )
        value = self.qlinear(f"{attn}.v_proj", normalized).reshape(
            cfg["num_key_value_heads"], cfg["head_dim"]
        )
        query = self.rms(query, self.tensor(f"{attn}.q_norm.weight").astype(mx.float32))
        key = self.rms(key, self.tensor(f"{attn}.k_norm.weight").astype(mx.float32))
        # Position zero is the identity RoPE. With one cached token, causal
        # attention has probability one and therefore returns repeated V.
        repeat = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        del query, key
        attended = mx.repeat(value, repeat, axis=0) * mx.sigmoid(gate)
        hidden = hidden + self.qlinear(f"{attn}.o_proj", attended.reshape(-1))
        normalized = self.rms(
            hidden,
            self.tensor(f"{prefix}.post_attention_layernorm.weight").astype(mx.float32),
        )
        return hidden + self.mlp(f"{prefix}.mlp", normalized)

    def gdn(self, layer: int):
        mx = self.mx
        cfg = self.text
        prefix = f"language_model.model.layers.{layer}"
        gdn = f"{prefix}.linear_attn"
        hidden = self.input_hidden()
        normalized = self.rms(
            hidden, self.tensor(f"{prefix}.input_layernorm.weight").astype(mx.float32)
        )
        projected = self.qlinear(f"{gdn}.in_proj_qkv", normalized)
        conv_weight = self.tensor(f"{gdn}.conv1d.weight").astype(mx.float32)
        convolved = mx.sigmoid(projected * conv_weight[:, -1, 0]) * (
            projected * conv_weight[:, -1, 0]
        )
        key_width = cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"]
        query = convolved[:key_width].reshape(
            cfg["linear_num_key_heads"], cfg["linear_key_head_dim"]
        )
        key = convolved[key_width : 2 * key_width].reshape(query.shape)
        value = convolved[2 * key_width :].reshape(
            cfg["linear_num_value_heads"], cfg["linear_value_head_dim"]
        )
        inverse = cfg["linear_key_head_dim"] ** -0.5
        query = self.rms(query, epsilon=1.0e-6) * (inverse**2)
        key = self.rms(key, epsilon=1.0e-6) * inverse
        beta = mx.sigmoid(self.qlinear(f"{gdn}.in_proj_b", normalized))
        repeat = cfg["linear_num_value_heads"] // cfg["linear_num_key_heads"]
        score = mx.sum(
            mx.repeat(query, repeat, axis=0) * mx.repeat(key, repeat, axis=0), axis=-1
        )
        recurrent = value * beta[:, None] * score[:, None]
        recurrent = self.rms(
            recurrent, self.tensor(f"{gdn}.norm.weight").astype(mx.float32)
        )
        z = self.qlinear(f"{gdn}.in_proj_z", normalized).reshape(recurrent.shape)
        recurrent = recurrent * (z * mx.sigmoid(z))
        hidden = hidden + self.qlinear(f"{gdn}.out_proj", recurrent.reshape(-1))
        normalized = self.rms(
            hidden,
            self.tensor(f"{prefix}.post_attention_layernorm.weight").astype(mx.float32),
        )
        return hidden + self.mlp(f"{prefix}.mlp", normalized)


def independent_output(executable: pathlib.Path, model_dir: pathlib.Path, kind: str):
    completed = subprocess.run(
        [str(executable), str(model_dir), kind],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: d2_real_layer_parity.py LAYER_EXE MODEL_DIR")
    executable = pathlib.Path(sys.argv[1]).resolve()
    model_dir = pathlib.Path(sys.argv[2]).resolve()
    oracle = Oracle(model_dir)
    reports = []
    for kind, layer in (("gdn", 0), ("attention", 3)):
        independent = independent_output(executable, model_dir, kind)
        expected_array = oracle.gdn(layer) if kind == "gdn" else oracle.attention(layer)
        oracle.mx.eval(expected_array)
        expected = [float(value) for value in expected_array.tolist()]
        actual = independent["output"]
        similarity, maximum_error = verify_parity(
            f"{kind} layer {layer}", actual, expected
        )
        reports.append(
            {
                "kind": kind,
                "layer": layer,
                "width": len(actual),
                "cosine": similarity,
                "maximum_absolute_error": maximum_error,
            }
        )
    oracle.shards.clear()
    oracle.mx.clear_cache()
    print(
        json.dumps(
            {
                "scope": "M4 correctness; real Qwen3.8-27B layer slices",
                "cases": reports,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
