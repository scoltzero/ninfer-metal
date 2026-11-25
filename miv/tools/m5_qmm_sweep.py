#!/usr/bin/env python3
"""Build or execute the fixed Qwen3.8 oQ4e NAX qmm sweep."""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=pathlib.Path,
        default=pathlib.Path("bench/probes/qwen38_27b_oq4e_shapes.json"),
    )
    parser.add_argument("--probe", type=pathlib.Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=pathlib.Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if args.execute and not args.probe:
        parser.error("--probe is required with --execute")
    rows: list[dict] = []
    for shape in manifest["shapes"]:
        for m in manifest["m_values"]:
            row = {
                "shape_id": shape["id"],
                "m": m,
                "n": shape["out_features"],
                "k": shape["in_features"],
                "bits": shape["bits"],
                "tensor_count": shape["tensor_count"],
            }
            if shape["out_features"] % 64:
                row.update(
                    {
                        "route": "reference",
                        "executed": False,
                        "skip_reason": "out_features is not divisible by NAX BN=64",
                    }
                )
            elif not args.execute:
                row.update({"route": "nax-candidate", "executed": False})
            else:
                command = [
                    str(args.probe),
                    "--m",
                    str(m),
                    "--n",
                    str(shape["out_features"]),
                    "--k",
                    str(shape["in_features"]),
                    "--bits",
                    str(shape["bits"]),
                    "--iterations",
                    str(args.iterations),
                ]
                result = subprocess.run(
                    command, check=True, capture_output=True, text=True
                )
                row.update(
                    {
                        "route": "nax-candidate",
                        "executed": True,
                        "result": json.loads(result.stdout),
                    }
                )
            rows.append(row)

    output = {
        "schema_version": 1,
        "model_revision": manifest["model_revision"],
        "execute_requested": args.execute,
        "rows": rows,
    }
    rendered = json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

