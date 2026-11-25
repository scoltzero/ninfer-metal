#!/usr/bin/env python3
"""Compare the independent C++ packed decoder with MLX dequantize."""

from __future__ import annotations

import json
import math
import pathlib
import random
import subprocess
import sys

EXPECTED_SAMPLE_COUNT = 64
MAX_ABS_ERROR = 1e-4


def verify_sample_parity(
    module: str,
    bits: int,
    independent: list[float],
    mlx: list[float],
    *,
    expected_count: int = EXPECTED_SAMPLE_COUNT,
    max_abs_tol: float = MAX_ABS_ERROR,
) -> float:
    if len(independent) != expected_count or len(mlx) != expected_count:
        raise AssertionError(
            f"{module} q{bits} sample length mismatch: "
            f"independent={len(independent)}, mlx={len(mlx)}, expected={expected_count}"
        )
    max_error = 0.0
    for idx, (left, right) in enumerate(zip(independent, mlx, strict=True)):
        if not (math.isfinite(left) and math.isfinite(right)):
            raise AssertionError(
                f"{module} q{bits} non-finite value at index {idx}: "
                f"independent={left}, mlx={right}"
            )
        diff = abs(left - right)
        if not (math.isfinite(diff) and diff <= max_abs_tol):
            raise AssertionError(
                f"{module} q{bits} decoder mismatch at index {idx}: "
                f"abs_error={diff} > {max_abs_tol}"
            )
        if diff > max_error:
            max_error = diff
    return max_error


def cpp_sample(
    inspector: pathlib.Path, model_dir: pathlib.Path, module: str
) -> list[float]:
    result = subprocess.run(
        [
            str(inspector),
            str(model_dir),
            "--expected-revision",
            "04dc5509edd8670fc78cc8c2f74bf9b77b1f2acc",
            "--sample-module",
            module,
            "--sample-row",
            "0",
            "--sample-start",
            "0",
            "--sample-count",
            "64",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)["sample"]["values"]


def mlx_sample(
    model_dir: pathlib.Path,
    module: str,
    bits: int,
    shard_cache: dict[str, dict],
) -> list[float]:
    import mlx.core as mx

    index = json.loads(
        (model_dir / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    shard_name = index["weight_map"][f"{module}.weight"]
    tensors = shard_cache.setdefault(
        shard_name, mx.load(str(model_dir / shard_name))
    )
    value = mx.dequantize(
        tensors[f"{module}.weight"][0:1],
        tensors[f"{module}.scales"][0:1],
        tensors[f"{module}.biases"][0:1],
        group_size=64,
        bits=bits,
        mode="affine",
        dtype=mx.float32,
    )[0, :64]
    mx.eval(value)
    result = [float(item) for item in value.tolist()]
    del value
    return result


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit(
            "usage: d1_quantization_parity.py INSPECTOR MODEL_DIR P4_HEADERS"
        )
    inspector = pathlib.Path(sys.argv[1]).resolve()
    model_dir = pathlib.Path(sys.argv[2]).resolve()
    headers = json.loads(pathlib.Path(sys.argv[3]).read_text(encoding="utf-8"))
    quantization = headers["config"]["quantization"]
    default_bits = quantization["bits"]
    by_bits: dict[int, list[str]] = {4: [], 5: []}
    for tensor_name, tensor in headers["tensors"].items():
        if not tensor_name.endswith(".weight") or tensor["dtype"] != "U32":
            continue
        module = tensor_name.removesuffix(".weight")
        spec = quantization.get(module)
        bits = spec.get("bits", default_bits) if isinstance(spec, dict) else default_bits
        by_bits[bits].append(module)
    rng = random.Random(20260919)
    cases = [
        (module, bits)
        for bits in (4, 5)
        for module in rng.sample(sorted(by_bits[bits]), 3)
    ]

    shard_cache: dict[str, dict] = {}
    for module, bits in cases:
        independent = cpp_sample(inspector, model_dir, module)
        mlx = mlx_sample(model_dir, module, bits, shard_cache)
        # The independent decoder keeps scale*q+bias in FP32. MLX performs
        # the affine arithmetic at its BF16 representation boundary before
        # returning the requested FP32 array, so one BF16 tail ULP is valid.
        error = verify_sample_parity(module, bits, independent, mlx)
        print(
            f"{module} q{bits}: 64 values match MLX within one BF16 tail ULP "
            f"(max_abs_error={error})"
        )
    import mlx.core as mx

    shard_cache.clear()
    mx.clear_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
