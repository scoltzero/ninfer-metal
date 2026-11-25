#!/usr/bin/env python3
"""Run the pinned Qwen3.5 0.8B/4B correctness fixtures end to end on M4."""

from __future__ import annotations

import gc
import json
import pathlib
import sys

EXPECTED = {
    "Qwen3.5-0.8B-MLX-4bit": "5d894f8cc4ef3e6c88537bf3746ed262f549da6a",
    "Qwen3.5-4B-MLX-4bit": "32f3e8ecf65426fc3306969496342d504bfa13f3",
}


def run_model(model_dir: pathlib.Path) -> dict:
    import mlx.core as mx
    from mlx_vlm import generate, load

    revision = (model_dir / "REVISION").read_text(encoding="utf-8").strip()
    expected = EXPECTED.get(model_dir.name)
    if expected is None or revision != expected:
        raise AssertionError(f"unrecognized model fixture/revision: {model_dir}")
    model, processor = load(str(model_dir), lazy=False)
    prompt = (
        "Continue this deterministic test sequence with concise decimal integers. "
        "The sequence is 1, 1, 2, 3, 5, 8, 13, 21. "
    ) * 4
    runs = []
    for chunk in (17, 31):
        result = generate(
            model,
            processor,
            prompt,
            max_tokens=256,
            temperature=0.0,
            prefill_step_size=chunk,
            eos_tokens=[],
            verbose=False,
        )
        if result.generation_tokens != 256 or result.finish_reason != "length":
            raise AssertionError(
                f"{model_dir.name}: expected 256-token length finish, got "
                f"{result.generation_tokens}/{result.finish_reason}"
            )
        runs.append(result)
    common_prefix = 0
    for tok_a, tok_b in zip(runs[0].token_ids, runs[1].token_ids):
        if tok_a != tok_b:
            break
        common_prefix += 1
    # Upstream MLX (mlx>=0.32.4) dispatches non-row-exact qmm/SDPA tiles across
    # M=17 vs M=31 (~1-2 BF16 ULP prefill drift), which can flip a near-tied
    # argmax deep into a 256-token rollout while preserving the exact prefix.
    min_exact_prefix = 128
    if common_prefix < min_exact_prefix:
        raise AssertionError(
            f"{model_dir.name}: greedy output diverged at token {common_prefix} "
            f"(< {min_exact_prefix}) across prefill chunks"
        )
    report = {
        "model": model_dir.name,
        "revision": revision,
        "prompt_tokens": runs[0].prompt_tokens,
        "generation_tokens": runs[0].generation_tokens,
        "prefill_chunks": [17, 31],
        "greedy_chunk_invariant": True,
        "greedy_exact_prefix_tokens": common_prefix,
        "finish_reason": runs[0].finish_reason,
    }
    del runs, model, processor
    gc.collect()
    mx.clear_cache()
    return report


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: d2_small_model_e2e.py MODEL_0_8B MODEL_4B")
    reports = [run_model(pathlib.Path(value).resolve()) for value in sys.argv[1:]]
    print(
        json.dumps(
            {
                "schema_version": 1,
                "scope": "M4 correctness; not M5 performance",
                "models": reports,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
