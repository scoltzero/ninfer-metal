#!/usr/bin/env python3
"""Compare one numeric metric from two benchmark result files."""

from __future__ import annotations

import argparse
import json
import pathlib


def load(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=pathlib.Path)
    parser.add_argument("candidate", type=pathlib.Path)
    parser.add_argument("--metric", required=True)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    for field in ("model_revision", "mode", "prompt_id", "context_tokens", "m", "k"):
        if baseline.get(field) != candidate.get(field):
            raise SystemExit(
                f"incomparable results: {field} differs "
                f"({baseline.get(field)!r} != {candidate.get(field)!r})"
            )
    baseline_value = baseline["metrics"][args.metric]
    candidate_value = candidate["metrics"][args.metric]
    if not isinstance(baseline_value, (int, float)) or not isinstance(
        candidate_value, (int, float)
    ):
        raise SystemExit("selected metric must be numeric")
    if baseline_value == 0:
        raise SystemExit("baseline metric must not be zero")
    print(
        json.dumps(
            {
                "metric": args.metric,
                "baseline": baseline_value,
                "candidate": candidate_value,
                "candidate_over_baseline": candidate_value / baseline_value,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

