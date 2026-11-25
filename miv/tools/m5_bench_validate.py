#!/usr/bin/env python3
"""Validate the stable D0 benchmark result envelope with no third-party deps."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys


REQUIRED = {
    "schema_version": int,
    "source_id": str,
    "date": str,
    "machine_id": str,
    "macos": str,
    "xcode": str,
    "model_revision": str,
    "mode": str,
    "engine": str,
    "measurement_scope": str,
    "metrics": dict,
}
MODES = {"probe", "kernel", "prefill", "decode", "verify", "dflash2"}
ENGINES = {"probe", "stock", "m5-reference", "m5-specialized"}
SCOPES = {"M4 correctness and harness only", "M5 Ultra measurement"}
OPTIONAL = {
    "omlx_commit",
    "commit",
    "manifest_hash",
    "prompt_id",
    "prompt_tokens",
    "gen_tokens",
    "context_tokens",
    "m",
    "k",
    "seed",
    "environment",
    "error",
    "fallback_reason",
}


def validate(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return ["root must be a JSON object"]
    errors: list[str] = []
    for field, expected_type in REQUIRED.items():
        if field not in payload:
            errors.append(f"missing required field: {field}")
        elif not isinstance(payload[field], expected_type):
            errors.append(f"{field} must be {expected_type.__name__}")
    if payload.get("schema_version") != 1:
        errors.append("schema_version must be 1")
    unknown = set(payload) - set(REQUIRED) - OPTIONAL
    if unknown:
        errors.append(f"unknown fields: {sorted(unknown)}")
    date = payload.get("date")
    if isinstance(date, str):
        try:
            dt.datetime.fromisoformat(date.replace("Z", "+00:00"))
        except ValueError:
            errors.append("date must be an ISO-8601 timestamp")
    if payload.get("mode") not in MODES:
        errors.append(f"mode must be one of {sorted(MODES)}")
    if payload.get("engine") not in ENGINES:
        errors.append(f"engine must be one of {sorted(ENGINES)}")
    if payload.get("measurement_scope") not in SCOPES:
        errors.append(f"measurement_scope must be one of {sorted(SCOPES)}")
    metrics = payload.get("metrics")
    if isinstance(metrics, dict):
        if not metrics:
            errors.append("metrics must not be empty")
        for name, value in metrics.items():
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, (int, float))
            ):
                errors.append(f"metrics.{name} must be numeric or null")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=pathlib.Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.result.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        print(f"{args.result}: {error}", file=sys.stderr)
        return 2
    errors = validate(payload)
    if errors:
        for error in errors:
            print(f"{args.result}: {error}", file=sys.stderr)
        return 1
    print(f"{args.result}: valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
