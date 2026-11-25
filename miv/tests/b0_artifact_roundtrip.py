#!/usr/bin/env python3
"""B0 container, adapter, decoder, determinism, and corruption tests."""

from __future__ import annotations

import argparse
import copy
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib
from pathlib import Path

import mlx.core as mx
import numpy as np

MIV_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MIV_ROOT / "tools"))

import m5_artifact as artifact


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def pack_codes(codes: list[int], bits: int) -> bytes:
    output = bytearray((len(codes) * bits + 7) // 8)
    for index, code in enumerate(codes):
        for bit in range(bits):
            if code & (1 << bit):
                absolute = index * bits + bit
                output[absolute // 8] |= 1 << (absolute % 8)
    return bytes(output)


def pack_nibbles(codes: list[int]) -> bytes:
    expect(len(codes) % 2 == 0, "nibble fixture must have an even code count")
    return bytes(
        (codes[index] & 0xF) | ((codes[index + 1] & 0xF) << 4)
        for index in range(0, len(codes), 2)
    )


def bf16(values: list[float]) -> bytes:
    words = np.asarray(values, dtype=np.float32).view(np.uint32) >> 16
    return words.astype("<u2").tobytes()


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], bytes]]
) -> int:
    header: dict[str, dict] = {}
    offset = 0
    for name in sorted(tensors):
        dtype, shape, payload = tensors[name]
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    encoded = json.dumps(header, separators=(",", ":"), sort_keys=True).encode()
    encoded += b" " * (-len(encoded) % 8)
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(encoded)))
        handle.write(encoded)
        for name in sorted(tensors):
            handle.write(tensors[name][2])
    return path.stat().st_size


def write_model(
    root: Path,
    revision: str,
    config: dict,
    tensors: dict[str, tuple[str, list[int], bytes]],
) -> Path:
    root.mkdir(parents=True)
    shard = "model.safetensors"
    size = write_safetensors(root / shard, tensors)
    index = {
        "metadata": {"total_size": size},
        "weight_map": {name: shard for name in sorted(tensors)},
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2) + "\n"
    )
    (root / "REVISION").write_text(revision + "\n")
    return root


def make_fixtures(root: Path) -> tuple[dict[str, Path], dict[str, list[int]]]:
    q4 = [index % 16 for index in range(64)]
    q5 = [(index * 3) % 32 for index in range(64)]
    q5_g128 = [(index * 7) % 32 for index in range(128)]
    q6 = [(index * 5) % 64 for index in range(64)]
    q6_g128 = [(index * 11) % 64 for index in range(128)]
    q8 = [(index * 13) % 256 for index in range(64)]
    q8_g128 = [(index * 17) % 256 for index in range(128)]
    mx_codes = [index % 16 for index in range(64)]
    mx_packed = pack_nibbles(mx_codes)
    fixtures: dict[str, Path] = {}
    fixtures["oq"] = write_model(
        root / "oq",
        "oq-fixture-revision",
        {
            "model_type": "fixture",
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                "q5": {"bits": 5, "group_size": 64, "mode": "affine"},
                "q5_g128": {"bits": 5, "group_size": 128, "mode": "affine"},
                "q6": {"bits": 6, "group_size": 64, "mode": "affine"},
                "q6_g128": {"bits": 6, "group_size": 128, "mode": "affine"},
                "q8": {"bits": 8, "group_size": 64, "mode": "affine"},
                "q8_g128": {"bits": 8, "group_size": 128, "mode": "affine"},
            },
        },
        {
            "experts.biases": ("BF16", [2, 3, 1], bf16([-1.0] * 6)),
            "experts.scales": ("BF16", [2, 3, 1], bf16([0.5] * 6)),
            "experts.weight": ("U32", [2, 3, 8], pack_codes(q4, 4) * 6),
            "norm.weight": ("BF16", [1, 2], bf16([1.0, 2.0])),
            "q4.biases": ("BF16", [1, 1], bf16([-1.0])),
            "q4.scales": ("BF16", [1, 1], bf16([0.5])),
            "q4.weight": ("U32", [1, 8], pack_codes(q4, 4)),
            "q5.biases": ("BF16", [1, 1], bf16([0.5])),
            "q5.scales": ("BF16", [1, 1], bf16([0.25])),
            "q5.weight": ("U32", [1, 10], pack_codes(q5, 5)),
            "q5_g128.biases": ("BF16", [1, 1], bf16([0.5])),
            "q5_g128.scales": ("BF16", [1, 1], bf16([0.25])),
            "q5_g128.weight": ("U32", [1, 20], pack_codes(q5_g128, 5)),
            "q6.biases": ("BF16", [1, 1], bf16([-2.0])),
            "q6.scales": ("BF16", [1, 1], bf16([0.125])),
            "q6.weight": ("U32", [1, 12], pack_codes(q6, 6)),
            "q6_g128.biases": ("BF16", [1, 1], bf16([-2.0])),
            "q6_g128.scales": ("BF16", [1, 1], bf16([0.125])),
            "q6_g128.weight": ("U32", [1, 24], pack_codes(q6_g128, 6)),
            "q8.biases": ("BF16", [1, 1], bf16([-4.0])),
            "q8.scales": ("BF16", [1, 1], bf16([0.0625])),
            "q8.weight": ("U32", [1, 16], pack_codes(q8, 8)),
            "q8_g128.biases": ("BF16", [1, 1], bf16([-4.0])),
            "q8_g128.scales": ("BF16", [1, 1], bf16([0.0625])),
            "q8_g128.weight": ("U32", [1, 32], pack_codes(q8_g128, 8)),
        },
    )
    mlx_tensors = {
        "experts.scales": ("U8", [2, 3, 2], bytes([127, 128]) * 6),
        "experts.weight": ("U32", [2, 3, 8], mx_packed * 6),
        "linear.scales": ("U8", [1, 2], bytes([127, 128])),
        "linear.weight": ("U32", [1, 8], mx_packed),
        "norm.weight": ("BF16", [1, 2], bf16([1.0, 2.0])),
        "router.biases": ("BF16", [2, 1], bf16([-4.0, -4.0])),
        "router.scales": ("BF16", [2, 1], bf16([0.0625, 0.0625])),
        "router.weight": ("U32", [2, 16], pack_codes(q8, 8) * 2),
    }
    fixtures["mlx"] = write_model(
        root / "mlx",
        "mlx-fixture-revision",
        {
            "model_type": "fixture",
            "quantization": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
                "router": {"bits": 8, "group_size": 64},
            },
        },
        mlx_tensors,
    )
    fixtures["redhat"] = write_model(
        root / "redhat",
        "redhat-fixture-revision",
        {
            "model_type": "fixture",
            "quantization_config": {"format": "mxfp4-pack-quantized"},
        },
        {
            "linear.weight_packed": ("U8", [1, 32], mx_packed),
            "linear.weight_scale": ("U8", [1, 2], bytes([127, 128])),
            "norm.weight": ("BF16", [1, 2], bf16([1.0, 2.0])),
        },
    )
    return fixtures, {
        "q4": q4,
        "q5": q5,
        "q5_g128": q5_g128,
        "q6": q6,
        "q6_g128": q6_g128,
        "q8": q8,
        "q8_g128": q8_g128,
        "mx": mx_codes,
    }


def convert(source: Path, output: Path, profile: str) -> tuple[dict, dict]:
    model = artifact.SafeTensorSource(source, (source / "REVISION").read_text().strip())
    try:
        report = artifact.write_artifact(model, output, "fixture/repo", profile)
    finally:
        model.close()
    summary, manifest = artifact.read_artifact(output, verify_payload=True)
    expect(
        report["artifact_sha256"] == artifact.sha256_file(output), "report SHA mismatch"
    )
    expect(summary["payload_verified"], "payload was not verified")
    return report, manifest


def plane_bytes(path: Path, manifest: dict, object_name: str, kind: str) -> bytes:
    record = next(item for item in manifest["objects"] if item["name"] == object_name)
    plane = next(item for item in record["planes"] if item["kind"] == kind)
    with path.open("rb") as handle:
        payload_offset = artifact.HEADER_STRUCT.unpack(
            handle.read(artifact.HEADER_BYTES)
        )[5]
        handle.seek(payload_offset + plane["offset"])
        return handle.read(plane["bytes"])


def test_roundtrip(
    root: Path,
    fixtures: dict[str, Path],
    codes: dict[str, list[int]],
    golden: dict,
    cpp_inspector: Path,
) -> dict[str, tuple[Path, dict]]:
    profiles = {
        "oq": "oq_affine",
        "mlx": "mlx_mxfp4",
        "redhat": "compressed_mxfp4",
    }
    outputs: dict[str, tuple[Path, dict]] = {}
    for name, profile in profiles.items():
        first = root / f"{name}-first.omlxm5"
        second = root / f"{name}-second.omlxm5"
        first_report, manifest = convert(fixtures[name], first, profile)
        second_report, _ = convert(fixtures[name], second, profile)
        expect(
            first_report["artifact_sha256"] == second_report["artifact_sha256"],
            f"{name} conversion is not deterministic",
        )
        expect(first.read_bytes() == second.read_bytes(), f"{name} bytes differ")
        outputs[name] = (first, manifest)

    oq_path, oq_manifest = outputs["oq"]
    q4 = artifact.decode_artifact_row(oq_path, oq_manifest, "q4", 0, 0, 16)
    q5 = artifact.decode_artifact_row(oq_path, oq_manifest, "q5", 0, 0, 32)
    q6 = artifact.decode_artifact_row(oq_path, oq_manifest, "q6", 0, 0, 16)
    q8 = artifact.decode_artifact_row(oq_path, oq_manifest, "q8", 0, 0, 16)
    expect(q4 == golden["oq_affine_q4_g64_bf16_v1"]["decoded"], "Q4 golden mismatch")
    expect(q5 == golden["oq_affine_q5_g64_bf16_v1"]["decoded"], "Q5 golden mismatch")
    expect(q6 == golden["oq_affine_q6_g64_bf16_v1"]["decoded"], "Q6 golden mismatch")
    expect(q8 == golden["oq_affine_q8_g64_bf16_v1"]["decoded"], "Q8 golden mismatch")
    expect(
        artifact.decode_artifact_row(oq_path, oq_manifest, "q5_g128", 0, 0, 128)
        == [0.25 * code + 0.5 for code in codes["q5_g128"]],
        "Q5/G128 decode mismatch",
    )
    expect(
        artifact.decode_artifact_row(oq_path, oq_manifest, "q6_g128", 0, 0, 128)
        == [0.125 * code - 2.0 for code in codes["q6_g128"]],
        "Q6/G128 decode mismatch",
    )
    expect(
        artifact.decode_artifact_row(oq_path, oq_manifest, "q8_g128", 0, 0, 128)
        == [0.0625 * code - 4.0 for code in codes["q8_g128"]],
        "Q8/G128 decode mismatch",
    )
    expert_record = next(
        item for item in oq_manifest["objects"] if item["name"] == "experts"
    )
    expect(
        expert_record["shape"] == [2, 3, 64]
        and expert_record["layout"] == "oq_expert_rowsplit_k128_v1",
        "affine expert-major layout mismatch",
    )
    expect(
        artifact.decode_artifact_row(oq_path, oq_manifest, "experts", 4, 0, 16)
        == golden["oq_affine_q4_g64_bf16_v1"]["decoded"],
        "affine expert row decode mismatch",
    )

    base = plane_bytes(oq_path, oq_manifest, "q5", "base_code")
    high = plane_bytes(oq_path, oq_manifest, "q5", "high_bit")
    reconstructed = []
    for column in range(128):
        code = (base[column // 2] >> ((column & 1) * 4)) & 0xF
        code |= ((high[column // 8] >> (column & 7)) & 1) << 4
        reconstructed.append(code)
    expect(reconstructed[:64] == codes["q5"], "Q5 base/high split is not bit-exact")
    expect(reconstructed[64:] == [0] * 64, "Q5 K padding is not zero")

    q6_base = plane_bytes(oq_path, oq_manifest, "q6", "base_code")
    q6_high = plane_bytes(oq_path, oq_manifest, "q6", "high_2bit")
    q6_reconstructed = []
    for column in range(128):
        code = (q6_base[column // 2] >> ((column & 1) * 4)) & 0xF
        code |= ((q6_high[column // 4] >> ((column & 3) * 2)) & 3) << 4
        q6_reconstructed.append(code)
    expect(q6_reconstructed[:64] == codes["q6"], "Q6 split is not bit-exact")
    expect(q6_reconstructed[64:] == [0] * 64, "Q6 K padding is not zero")

    for name in ("mlx", "redhat"):
        path, manifest = outputs[name]
        decoded = artifact.decode_artifact_row(path, manifest, "linear", 0, 0, 64)
        packed = np.frombuffer(pack_nibbles(codes["mx"]), dtype=np.uint32).reshape(1, 8)
        scales = np.asarray([[127, 128]], dtype=np.uint8)
        oracle = np.asarray(
            mx.dequantize(
                mx.array(packed),
                mx.array(scales),
                group_size=32,
                bits=4,
                mode="mxfp4",
                dtype=mx.float32,
            )
        )[0].tolist()
        expect(decoded == oracle, f"{name} decode disagrees with stock MLX")

        command = [
            str(cpp_inspector),
            str(path),
            "--verify-payload",
            "--materialization-plan",
            "--sample-object",
            "linear",
            "--sample-count",
            "64",
        ]
        native = json.loads(subprocess.check_output(command, text=True))
        expect(native["sample"]["values"] == oracle, f"{name} C++ decode mismatch")
        plan = native["materialization_plan"]
        expect(plan["upload_bytes"] <= plan["arena_bytes"], "upload ledger overflow")
        expect(
            plan["arena_bytes"] - plan["upload_bytes"] == plan["arena_padding_bytes"],
            "upload ledger padding mismatch",
        )

    mlx_path, mlx_manifest = outputs["mlx"]
    expert_mx = next(
        item for item in mlx_manifest["objects"] if item["name"] == "experts"
    )
    expect(
        expert_mx["shape"] == [2, 3, 64]
        and expert_mx["layout"] == "mx_expert_rowsplit_k128_v1",
        "MXFP4 expert-major layout mismatch",
    )
    expect(
        artifact.decode_artifact_row(mlx_path, mlx_manifest, "experts", 5, 0, 64)
        == oracle,
        "MXFP4 expert row decode mismatch",
    )
    router = artifact.decode_artifact_row(mlx_path, mlx_manifest, "router", 1, 0, 16)
    expect(
        router == golden["oq_affine_q8_g64_bf16_v1"]["decoded"],
        "MX Q8 override mismatch",
    )
    for object_name in ("q6", "q8", "experts"):
        native = json.loads(
            subprocess.check_output(
                [
                    str(cpp_inspector),
                    str(oq_path),
                    "--sample-object",
                    object_name,
                    "--sample-row",
                    "4" if object_name == "experts" else "0",
                    "--sample-count",
                    "16",
                ],
                text=True,
            )
        )
        python_values = artifact.decode_artifact_row(
            oq_path,
            oq_manifest,
            object_name,
            4 if object_name == "experts" else 0,
            0,
            16,
        )
        expect(
            native["sample"]["values"] == python_values, f"{object_name} C++ mismatch"
        )

    mlx_record = next(
        item for item in outputs["mlx"][1]["objects"] if item["name"] == "linear"
    )
    redhat_record = next(
        item for item in outputs["redhat"][1]["objects"] if item["name"] == "linear"
    )
    for key in ("format", "layout", "shape", "padded_k"):
        expect(mlx_record[key] == redhat_record[key], f"MX adapters differ in {key}")
    expect(
        [(p["kind"], p["bytes"]) for p in mlx_record["planes"]]
        == [(p["kind"], p["bytes"]) for p in redhat_record["planes"]],
        "MX adapters do not normalize to one plane contract",
    )
    return outputs


def rewrite_manifest(source: Path, target: Path, mutate) -> None:
    raw = source.read_bytes()
    header = artifact.HEADER_STRUCT.unpack(raw[: artifact.HEADER_BYTES])
    manifest_offset, manifest_bytes, old_payload_offset, payload_bytes = header[3:7]
    manifest = json.loads(raw[manifest_offset : manifest_offset + manifest_bytes])
    mutate(manifest)
    encoded = artifact.canonical_json(manifest)
    payload_offset = artifact.align_up(
        artifact.HEADER_BYTES + len(encoded), artifact.PAYLOAD_ALIGNMENT
    )
    manifest_crc = artifact.crc32(encoded)
    prefix = artifact.HEADER_STRUCT.pack(
        artifact.MAGIC,
        artifact.VERSION,
        artifact.HEADER_BYTES,
        artifact.HEADER_BYTES,
        len(encoded),
        payload_offset,
        payload_bytes,
        len(manifest["objects"]),
        manifest_crc,
        0,
        0,
    )
    header_bytes = artifact.HEADER_STRUCT.pack(
        artifact.MAGIC,
        artifact.VERSION,
        artifact.HEADER_BYTES,
        artifact.HEADER_BYTES,
        len(encoded),
        payload_offset,
        payload_bytes,
        len(manifest["objects"]),
        manifest_crc,
        artifact.crc32(prefix[:56]),
        0,
    )
    payload = raw[old_payload_offset : old_payload_offset + payload_bytes]
    target.write_bytes(
        header_bytes
        + encoded
        + bytes(payload_offset - artifact.HEADER_BYTES - len(encoded))
        + payload
    )


def expect_rejected(
    path: Path, cpp_inspector: Path, verify_payload: bool = False
) -> None:
    try:
        artifact.read_artifact(path, verify_payload=verify_payload)
    except (ValueError, OverflowError, KeyError, TypeError):
        pass
    else:
        raise AssertionError(f"Python reader accepted corruption: {path.name}")
    command = [str(cpp_inspector), str(path)]
    if verify_payload:
        command.append("--verify-payload")
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    expect(result.returncode != 0, f"C++ reader accepted corruption: {path.name}")


def test_corruption(
    root: Path, source: Path, manifest: dict, cpp_inspector: Path
) -> None:
    unsupported = root / "unsupported-version.omlxm5"
    raw = bytearray(source.read_bytes())
    struct.pack_into("<I", raw, 8, 2)
    struct.pack_into("<I", raw, 56, zlib.crc32(raw[:56]) & 0xFFFFFFFF)
    unsupported.write_bytes(raw)
    expect_rejected(unsupported, cpp_inspector)

    bad_manifest = root / "bad-manifest.omlxm5"
    raw = bytearray(source.read_bytes())
    raw[artifact.HEADER_BYTES] ^= 1
    bad_manifest.write_bytes(raw)
    expect_rejected(bad_manifest, cpp_inspector)

    bad_alignment = root / "bad-alignment.omlxm5"
    rewrite_manifest(
        source,
        bad_alignment,
        lambda value: value["objects"][0].__setitem__("object_offset", 1),
    )
    expect_rejected(bad_alignment, cpp_inspector)

    bad_range = root / "bad-range.omlxm5"
    rewrite_manifest(
        source,
        bad_range,
        lambda value: value["objects"][0].__setitem__("object_bytes", 2**64 - 1),
    )
    expect_rejected(bad_range, cpp_inspector)

    bad_hash = root / "bad-hash.omlxm5"
    rewrite_manifest(
        source,
        bad_hash,
        lambda value: value["objects"][0]["planes"][0].__setitem__("sha256", "0" * 64),
    )
    try:
        artifact.read_artifact(bad_hash, verify_payload=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Python reader accepted a bad SHA-256")
    expect_rejected(bad_hash, cpp_inspector, verify_payload=True)

    bad_payload = root / "bad-payload.omlxm5"
    shutil.copyfile(source, bad_payload)
    record = next(item for item in manifest["objects"] if item["name"] == "q4")
    plane = record["planes"][0]
    with bad_payload.open("r+b") as handle:
        header = artifact.HEADER_STRUCT.unpack(handle.read(artifact.HEADER_BYTES))
        handle.seek(header[5] + plane["offset"])
        byte = handle.read(1)
        handle.seek(-1, 1)
        handle.write(bytes([byte[0] ^ 1]))
    expect_rejected(bad_payload, cpp_inspector, verify_payload=True)

    bad_padding = root / "bad-padding.omlxm5"
    shutil.copyfile(source, bad_padding)
    with bad_padding.open("r+b") as handle:
        header = artifact.HEADER_STRUCT.unpack(handle.read(artifact.HEADER_BYTES))
        padding_offset = header[3] + header[4]
        expect(padding_offset < header[5], "fixture has no manifest padding")
        handle.seek(padding_offset)
        handle.write(b"\x01")
    expect_rejected(bad_padding, cpp_inspector)


def test_unsafe_source_index(root: Path) -> None:
    source = root / "unsafe-source"
    source.mkdir()
    (source / "config.json").write_text('{"quantization":{"mode":"mxfp4"}}\n')
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"linear.weight": "../escape.safetensors"}})
    )
    try:
        artifact.SafeTensorSource(source)
    except ValueError as error:
        expect("unsafe shard name" in str(error), "unsafe source failed ambiguously")
    else:
        raise AssertionError("source adapter accepted a path-traversing shard")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-inspector", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    golden = json.loads(args.golden.read_text())
    expect(golden["schema"] == "omlx-m5-b0-golden-v1", "golden schema mismatch")
    with tempfile.TemporaryDirectory(prefix="omlx-m5-b0-") as temporary:
        root = Path(temporary)
        fixtures, codes = make_fixtures(root / "sources")
        outputs = test_roundtrip(
            root, fixtures, codes, golden, args.cpp_inspector.resolve()
        )
        oq_path, oq_manifest = outputs["oq"]
        test_corruption(root, oq_path, copy.deepcopy(oq_manifest), args.cpp_inspector)
        test_unsafe_source_index(root)
    print("B0 artifact round-trip tests passed")


if __name__ == "__main__":
    main()
