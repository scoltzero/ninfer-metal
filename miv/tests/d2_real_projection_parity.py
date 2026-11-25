#!/usr/bin/env python3
"""Compare D2 q4/q5 real-layer projections with MLX on M4.

The selected modules are slices of a real GDN layer and a real full-attention
layer. This test intentionally validates correctness only; timings are not an
M5 performance result.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

CASES = (
    ("language_model.model.layers.0.linear_attn.in_proj_qkv", 4),
    ("language_model.model.layers.3.self_attn.o_proj", 5),
)
ROWS = (1, 2, 8)


def make_input(rows: int, width: int):
    import mlx.core as mx

    values = [
        (((row * 17 + column * 13) % 257) - 128) / 1024.0
        for row in range(rows)
        for column in range(width)
    ]
    return mx.array(values, dtype=mx.float32).reshape(rows, width)


MIN_COSINE = 0.9999
MAX_ABS_ERROR = 1e-3


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


def cpp_result(
    executable: pathlib.Path, model_dir: pathlib.Path, module: str, rows: int
) -> dict:
    completed = subprocess.run(
        [str(executable), str(model_dir), module, str(rows)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: d2_real_projection_parity.py PROJECTION_EXE MODEL_DIR")
    executable = pathlib.Path(sys.argv[1]).resolve()
    model_dir = pathlib.Path(sys.argv[2]).resolve()
    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    import mlx.core as mx

    shard_cache: dict[str, dict] = {}
    reports = []
    for module, expected_bits in CASES:
        shard_name = index["weight_map"][f"{module}.weight"]
        tensors = shard_cache.setdefault(
            shard_name, mx.load(str(model_dir / shard_name))
        )
        for rows in ROWS:
            independent = cpp_result(executable, model_dir, module, rows)
            if independent["bits"] != expected_bits:
                raise AssertionError(
                    f"{module}: expected q{expected_bits}, got q{independent['bits']}"
                )
            source = make_input(rows, independent["input_width"])
            oracle = mx.quantized_matmul(
                source,
                tensors[f"{module}.weight"],
                tensors[f"{module}.scales"],
                tensors[f"{module}.biases"],
                group_size=64,
                bits=expected_bits,
                mode="affine",
            ).astype(mx.float32)
            mx.eval(oracle)
            expected = [float(value) for row in oracle.tolist() for value in row]
            actual = independent["output"]
            similarity, maximum_error = verify_parity(
                f"{module} M={rows}", actual, expected
            )
            reports.append(
                {
                    "module": module,
                    "bits": expected_bits,
                    "rows": rows,
                    "cosine": similarity,
                    "maximum_absolute_error": maximum_error,
                }
            )
    shard_cache.clear()
    mx.clear_cache()
    print(json.dumps({"scope": "M4 correctness", "cases": reports}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
