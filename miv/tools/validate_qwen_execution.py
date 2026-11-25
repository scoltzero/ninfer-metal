#!/usr/bin/env python3
"""Qualify native arithmetic, including discontinuous MoE routing boundaries.

Unconditional FP64 and stock MLX differences remain in the report. Conditional
checks are explicitly labelled, never promoted to bit-identical model parity.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from qwen_reference import QwenReference


def compare(actual, expected):
    actual = np.asarray(actual).reshape(expected.shape)
    error = actual - expected
    maximum = float(np.max(np.abs(error)))
    relative = float(np.linalg.norm(error) / max(np.linalg.norm(expected), 1e-12))
    return {
        "max_abs": maximum,
        "relative_l2": relative,
        "passed": bool(
            np.isfinite(actual).all() and maximum <= 0.08 and relative <= 0.02
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("fixtures", type=Path)
    parser.add_argument("libraries", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run = subprocess.run(
        [str(args.executable), str(args.fixtures), str(args.libraries)],
        capture_output=True,
        check=False,
        text=True,
    )
    if run.stderr:
        print(run.stderr, file=sys.stderr)
    if run.returncode not in (0, 3):
        print(run.stdout)
        raise SystemExit(run.returncode or 1)
    records = json.loads(run.stdout)
    passed = True
    for record in records:
        if not record["mode"].startswith("moe-"):
            passed &= all(
                record[key]["passed"]
                for key in ("hidden", "logits", "features", "sequential")
            )
            continue
        root = args.fixtures / record["mode"]
        stored_config = json.loads((root / "source/config.json").read_text())
        config = stored_config["text_config"]
        quant = stored_config["quantization"]
        source = {
            name.removeprefix("language_model."): np.asarray(
                value.astype(mx.float32) if value.dtype == mx.bfloat16 else value
            )
            for name, value in mx.load(str(root / "source/model.safetensors")).items()
        }
        reference = QwenReference(
            config,
            source,
            quant["mode"],
            {
                name.removeprefix("language_model."): value
                for name, value in quant.items()
                if isinstance(value, dict)
            },
        )
        count = record["tokens"]
        tokens = [(i * 71 + 256) % config["vocab_size"] for i in range(count)]
        trace = json.loads((root / f"native-t{count}.json").read_text())
        qualified = {}
        for kind, observed in trace.items():
            observation = {
                key: np.asarray(observed[key]).reshape(
                    count, config["num_hidden_layers"], -1
                )
                for key in ("router_logits", "router_scores", "layer_routes")
            }
            try:
                hidden, logits, features = reference(tokens, observation)
                result = {
                    "routing": reference.route_qualification,
                    "hidden": compare(observed["hidden"], hidden),
                    "logits": compare(observed["logits"], logits),
                    "features": compare(observed["features"], features),
                }
                result["passed"] = all(
                    result[key]["passed"] for key in ("hidden", "logits", "features")
                )
            except AssertionError as error:
                result = {"passed": False, "error": str(error)}
            qualified[kind] = result
            passed &= result["passed"]
        record["qualified_represented_routing"] = qualified
    report = {
        "schema": "qwen-execution-qualification-v1",
        "passed": bool(passed),
        "scope": "small full-layer models; FP64 arithmetic with individually qualified BF16 routing boundaries; no full-checkpoint or NAX performance claim",
        "cases": records,
    }
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
