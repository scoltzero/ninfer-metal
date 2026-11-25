#!/usr/bin/env python3
"""Independent selector semantics over numerically qualified draft boundaries."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from qwen_reference import QwenReference, bf16

MASK = (1 << 64) - 1


def splitmix(x):
    x = (int(x) + 0x9E3779B97F4A7C15) & MASK
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & MASK
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & MASK
    return x ^ (x >> 31)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("executable", type=Path)
    parser.add_argument("targets", type=Path)
    parser.add_argument("draft", type=Path)
    parser.add_argument("libraries", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = subprocess.run(
        [str(args.executable), str(args.targets), str(args.draft), str(args.libraries)],
        capture_output=True,
        check=False,
        text=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode:
        print(result.stdout)
        raise SystemExit(result.returncode)
    report = json.loads(result.stdout)
    target = args.targets / "affine/source"
    cfg = json.loads((target / "config.json").read_text())
    source = {
        n.removeprefix("language_model."): np.asarray(
            v.astype(mx.float32) if v.dtype == mx.bfloat16 else v
        )
        for n, v in mx.load(str(target / "model.safetensors")).items()
    }
    head = QwenReference(cfg["text_config"], source, "affine").weight("lm_head")
    weights = {
        n: np.asarray(v.astype(mx.float32)).astype(np.float64)
        for n, v in mx.load(str(args.draft / "source/model.safetensors")).items()
    }
    projection = weights["candidate_selector.hidden_projection.weight"]
    predecessor = weights["candidate_selector.predecessor_codebook"]
    successor = weights["candidate_selector.successor_codebook"]
    traces = json.loads((args.draft / "native-traces.json").read_text())
    for trace in traces:
        h = np.asarray(trace["hidden"]).reshape(-1, 128)
        logits = np.asarray(trace["logits"]).reshape(-1, 257)
        # Shared quantized head consumes represented native hidden, decoded
        # independently from the target checkpoint (no private companion head).
        expected = bf16(h @ head.T)
        assert np.max(abs(expected - logits)) <= 0.03125
        previous = 7
        for t, step in enumerate(trace["steps"]):
            row = logits[t + 1]
            ids = np.lexsort((np.arange(257), -row))[:8]
            assert np.array_equal(ids, step["ids"]), "candidate top-k mismatch"
            projected = projection @ h[t + 1]
            energy = row[ids] + np.sum(
                predecessor[previous] * projected * successor[ids], axis=1
            )
            if trace["temperature"] == 0:
                probability = np.zeros(8)
                probability[np.argmax(energy)] = 1
                token = int(ids[np.argmax(energy)])
            else:
                probability = np.exp((energy - energy.max()) / trace["temperature"])
                probability /= probability.sum()
                bits = splitmix(
                    123 ^ splitmix(trace["context"] + t) ^ splitmix(0xDFA52002)
                )
                uniform = (bits >> 11) / 9007199254740992
                token = int(
                    ids[
                        min(
                            np.searchsorted(
                                np.cumsum(probability), uniform, side="right"
                            ),
                            7,
                        )
                    ]
                )
            assert (
                np.max(abs(probability - np.asarray(step["probabilities"]))) < 2e-6
            ), "conditional proposal probabilities"
            assert token == step["token"], "proposal sampling or argmax mismatch"
            previous = token
    report["selector_cases"] = len(traces)
    report["selector_qualification"] = (
        "FP64 independent codebook/hidden projection, exact top-k, normalized conditional q, seeded sampling over represented boundaries"
    )
    report["scope"] = (
        "small complete models on current Metal device; unconditional dflash-mlx proposal differences retained"
    )
    output = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(output)
    print(output)


if __name__ == "__main__":
    main()
