#!/usr/bin/env python3
"""B0 container, adapter, decoder, determinism, and corruption tests."""

from __future__ import annotations

import argparse
import copy
import json
import os
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
    magic, version = header[0], header[1]
    manifest_offset, manifest_bytes, old_payload_offset, payload_bytes = header[3:7]
    flags = header[10]
    manifest = json.loads(raw[manifest_offset : manifest_offset + manifest_bytes])
    mutate(manifest)
    encoded = artifact.canonical_json(manifest)
    payload_offset = artifact.align_up(
        artifact.HEADER_BYTES + len(encoded), artifact.PAYLOAD_ALIGNMENT
    )
    manifest_crc = artifact.crc32(encoded)
    prefix = artifact.HEADER_STRUCT.pack(
        magic,
        version,
        artifact.HEADER_BYTES,
        artifact.HEADER_BYTES,
        len(encoded),
        payload_offset,
        payload_bytes,
        len(manifest["objects"]),
        manifest_crc,
        0,
        flags,
    )
    header_bytes = artifact.HEADER_STRUCT.pack(
        magic,
        version,
        artifact.HEADER_BYTES,
        artifact.HEADER_BYTES,
        len(encoded),
        payload_offset,
        payload_bytes,
        len(manifest["objects"]),
        manifest_crc,
        artifact.crc32(prefix[:56]),
        flags,
    )
    payload = raw[old_payload_offset : old_payload_offset + payload_bytes]
    target.write_bytes(
        header_bytes
        + encoded
        + bytes(payload_offset - artifact.HEADER_BYTES - len(encoded))
        + payload
    )


def expect_rejected(
    path: Path, cpp_inspector: Path | None, verify_payload: bool = False
) -> None:
    try:
        artifact.read_artifact(path, verify_payload=verify_payload)
    except (ValueError, OverflowError, KeyError, TypeError):
        pass
    else:
        raise AssertionError(f"Python reader accepted corruption: {path.name}")
    if cpp_inspector is not None:
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

    unsupported_v3 = root / "unsupported-version-v3.omlxm5"
    raw = bytearray(source.read_bytes())
    struct.pack_into("<I", raw, 8, 3)
    struct.pack_into("<I", raw, 56, zlib.crc32(raw[:56]) & 0xFFFFFFFF)
    unsupported_v3.write_bytes(raw)
    expect_rejected(unsupported_v3, cpp_inspector)

    bad_flags = root / "bad-flags.omlxm5"
    raw = bytearray(source.read_bytes())
    struct.pack_into("<I", raw, 60, 0x4)
    bad_flags.write_bytes(raw)
    expect_rejected(bad_flags, cpp_inspector)

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

    for label, bad_magic in (
        ("rejected-alt-magic", b"MIVART" + b"01"),
        ("random-magic", b"BADMAGIC"),
    ):
        bad_magic_path = root / f"{label}.miv"
        raw = bytearray(source.read_bytes())
        raw[:8] = bad_magic
        struct.pack_into("<I", raw, 56, zlib.crc32(raw[:56]) & 0xFFFFFFFF)
        bad_magic_path.write_bytes(raw)
        expect_rejected(bad_magic_path, cpp_inspector)

    for label, bad_schema in (
        ("v1-mismatched-new-v2-schema", "miv-artifact-v2"),
        ("v1-mismatched-legacy-v2-schema", "omlx-m5-artifact-v2"),
        ("v1-unknown-schema", "unknown-artifact-v1"),
    ):
        bad_schema_path = root / f"{label}.miv"
        rewrite_manifest(
            source,
            bad_schema_path,
            lambda value, s=bad_schema: value.__setitem__("schema", s),
        )
        expect_rejected(bad_schema_path, cpp_inspector)


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


def test_affine8_nax_pretile_roundtrip(
    root: Path, cpp_inspector: Path | None = None
) -> None:
    rng = np.random.default_rng(2026)
    # 1) Pure 8-bit Affine checkpoint with q8_g64 (unaligned N=20, non-128-multiple K=576)
    #    and q8_g128 (unaligned N=18, K=512).
    n_g64, k_g64 = 20, 576
    n_g128, k_g128 = 18, 512
    codes_g64 = rng.integers(0, 256, size=(n_g64, k_g64), dtype=np.uint8)
    scales_g64 = rng.uniform(0.01, 0.15, size=(n_g64, k_g64 // 64)).astype(np.float32)
    biases_g64 = rng.uniform(-2.0, 2.0, size=(n_g64, k_g64 // 64)).astype(np.float32)

    codes_g128 = rng.integers(0, 256, size=(n_g128, k_g128), dtype=np.uint8)
    scales_g128 = rng.uniform(0.01, 0.15, size=(n_g128, k_g128 // 128)).astype(
        np.float32
    )
    biases_g128 = rng.uniform(-2.0, 2.0, size=(n_g128, k_g128 // 128)).astype(
        np.float32
    )

    pure_src = write_model(
        root / "oq_affine8_pretile_src",
        "oq-affine8-pretile-rev",
        {
            "model_type": "fixture",
            "quantization": {
                "bits": 8,
                "group_size": 64,
                "mode": "affine",
                "down_proj": {"bits": 8, "group_size": 128, "mode": "affine"},
            },
        },
        {
            "q_proj.weight": (
                "U32",
                [n_g64, k_g64 * 8 // 32],
                np.ascontiguousarray(codes_g64).tobytes(),
            ),
            "q_proj.scales": (
                "BF16",
                [n_g64, k_g64 // 64],
                bf16(scales_g64.reshape(-1).tolist()),
            ),
            "q_proj.biases": (
                "BF16",
                [n_g64, k_g64 // 64],
                bf16(biases_g64.reshape(-1).tolist()),
            ),
            "down_proj.weight": (
                "U32",
                [n_g128, k_g128 * 8 // 32],
                np.ascontiguousarray(codes_g128).tobytes(),
            ),
            "down_proj.scales": (
                "BF16",
                [n_g128, k_g128 // 128],
                bf16(scales_g128.reshape(-1).tolist()),
            ),
            "down_proj.biases": (
                "BF16",
                [n_g128, k_g128 // 128],
                bf16(biases_g128.reshape(-1).tolist()),
            ),
            "norm.weight": ("BF16", [512], bf16([1.0] * 512)),
        },
    )

    pure_untiled = root / "oq-affine8-untiled.omlxm5"
    pure_pretiled = root / "oq-affine8-pretiled.omlxm5"
    pure_twostep = root / "oq-affine8-twostep.omlxm5"

    subprocess.run(
        [
            sys.executable,
            str(MIV_ROOT / "tools" / "m5_artifact.py"),
            "convert",
            "--source",
            str(pure_src),
            "--output",
            str(pure_untiled),
            "--source-repo",
            "fixture/pure-affine8",
            "--expected-revision",
            "oq-affine8-pretile-rev",
            "--artifact-version",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    conv_res = subprocess.run(
        [
            sys.executable,
            str(MIV_ROOT / "tools" / "m5_artifact.py"),
            "convert",
            "--source",
            str(pure_src),
            "--output",
            str(pure_pretiled),
            "--source-repo",
            "fixture/pure-affine8",
            "--expected-revision",
            "oq-affine8-pretile-rev",
            "--artifact-version",
            "1",
            "--nax-pretile",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    conv_report = json.loads(conv_res.stdout)
    expect(conv_report["pretiled_objects"] == 2, "expected 2 pretiled affine8 objects")

    pt_res = artifact.pretile_artifact(pure_untiled, pure_twostep)
    expect(pt_res["pretiled_objects"] == 2, "two-step pretile count mismatch")
    expect(
        pure_pretiled.read_bytes() == pure_twostep.read_bytes(),
        "convert --nax-pretile and pretile produced non-identical bytes",
    )

    _, u_manifest = artifact.read_artifact(pure_untiled, verify_payload=True)
    _, p_manifest = artifact.read_artifact(pure_pretiled, verify_payload=True)
    p_objs = {obj["name"]: obj for obj in p_manifest["objects"]}

    expect(
        p_objs["q_proj"]["layout"] == "oq_nax_tiled_g64_v1"
        and [(p["kind"], p["bytes"]) for p in p_objs["q_proj"]["planes"]]
        == [("nax_codes", 32 * k_g64), ("nax_sb", 32 * (k_g64 // 64) * 4)],
        "q_proj q8_g64 NAX plane layout/bytes mismatch",
    )
    expect(
        p_objs["down_proj"]["layout"] == "oq_nax_tiled_g64_v1"
        and [(p["kind"], p["bytes"]) for p in p_objs["down_proj"]["planes"]]
        == [("nax_codes", 32 * k_g128), ("nax_sb", 32 * (k_g128 // 64) * 4)],
        "down_proj q8_g128 NAX plane layout/bytes mismatch",
    )

    # Stock MLX oracle dequantize & quantized_matmul verification for both q8_g64 and q8_g128.
    for obj_name, codes_u8, sc_raw, bi_raw, gs in (
        ("q_proj", codes_g64, scales_g64, biases_g64, 64),
        ("down_proj", codes_g128, scales_g128, biases_g128, 128),
    ):
        n_rows, k_cols = codes_u8.shape
        w_u32 = mx.array(np.ascontiguousarray(codes_u8).view(np.uint32))
        s_bf16 = mx.array(
            np.frombuffer(bf16(sc_raw.reshape(-1).tolist()), dtype=np.uint16).reshape(
                sc_raw.shape
            )
        ).view(mx.bfloat16)
        b_bf16 = mx.array(
            np.frombuffer(bf16(bi_raw.reshape(-1).tolist()), dtype=np.uint16).reshape(
                bi_raw.shape
            )
        ).view(mx.bfloat16)
        oracle_w = np.asarray(
            mx.dequantize(
                w_u32,
                s_bf16.astype(mx.float32),
                b_bf16.astype(mx.float32),
                group_size=gs,
                bits=8,
                mode="affine",
                dtype=mx.float32,
            )
        )
        u_w = np.asarray(
            [
                artifact.decode_artifact_row(pure_untiled, u_manifest, obj_name, r, 0, k_cols)
                for r in range(n_rows)
            ],
            dtype=np.float32,
        )
        p_w = np.asarray(
            [
                artifact.decode_artifact_row(pure_pretiled, p_manifest, obj_name, r, 0, k_cols)
                for r in range(n_rows)
            ],
            dtype=np.float32,
        )
        expect(np.array_equal(u_w, oracle_w), f"{obj_name} untiled disagrees with MLX")
        expect(np.array_equal(p_w, oracle_w), f"{obj_name} pretiled disagrees with MLX")

        # Matmul consistency across T=1 and T=8
        for t_rows in (1, 8):
            x_np = rng.normal(0.0, 0.5, size=(t_rows, k_cols)).astype(np.float32)
            x_mx = mx.array(x_np).astype(mx.bfloat16)
            x_bf16_f32 = np.asarray(x_mx.astype(mx.float32))
            y_mlx_f32 = np.asarray(
                mx.quantized_matmul(
                    x_mx.astype(mx.float32),
                    w_u32,
                    s_bf16.astype(mx.float32),
                    b_bf16.astype(mx.float32),
                    transpose=True,
                    group_size=gs,
                    bits=8,
                    mode="affine",
                )
            )
            y_pretiled = x_bf16_f32 @ p_w.T
            expect(
                np.allclose(y_pretiled, y_mlx_f32, rtol=1e-4, atol=1e-3),
                f"{obj_name} T={t_rows} matmul disagrees with MLX quantized_matmul",
            )
            expect(
                np.array_equal(y_pretiled, x_bf16_f32 @ u_w.T),
                f"{obj_name} T={t_rows} pretiled vs untiled matmul mismatch",
            )

        if cpp_inspector is not None:
            for sample_row in (0, min(n_rows - 1, 17)):
                native = json.loads(
                    subprocess.check_output(
                        [
                            str(cpp_inspector),
                            str(pure_pretiled),
                            "--verify-payload",
                            "--materialization-plan",
                            "--sample-object",
                            obj_name,
                            "--sample-row",
                            str(sample_row),
                            "--sample-count",
                            "64",
                        ],
                        text=True,
                    )
                )
                expect(
                    native["sample"]["values"] == p_w[sample_row, :64].tolist(),
                    f"{obj_name} row {sample_row} C++ pretiled decode mismatch",
                )

    # 2) Mixed MXFP4 + affine8 (q8_g64 & q8_g128) checkpoint --nax-pretile roundtrip.
    k_mix = 512
    mx_codes = rng.integers(0, 16, size=(16, k_mix), dtype=np.uint8)
    mx_packed = (mx_codes[:, 0::2] & 0xF) | ((mx_codes[:, 1::2] & 0xF) << 4)
    mx_scales = rng.integers(125, 130, size=(16, k_mix // 32), dtype=np.uint8)

    mix_src = write_model(
        root / "mlx_mixed_mxfp4_affine8_src",
        "mlx-mixed-affine8-rev",
        {
            "model_type": "fixture",
            "quantization": {
                "bits": 4,
                "group_size": 32,
                "mode": "mxfp4",
                "o_proj": {"bits": 8, "group_size": 64},
                "down_proj": {"bits": 8, "group_size": 128, "mode": "affine"},
            },
        },
        {
            "q_proj.weight": (
                "U32",
                [16, k_mix // 8],
                np.ascontiguousarray(mx_packed).tobytes(),
            ),
            "q_proj.scales": ("U8", [16, k_mix // 32], mx_scales.tobytes()),
            "o_proj.weight": (
                "U32",
                [16, k_mix * 8 // 32],
                np.ascontiguousarray(codes_g128[:16]).tobytes(),
            ),
            "o_proj.scales": (
                "BF16",
                [16, k_mix // 64],
                bf16(np.repeat(scales_g128[:16], 2, axis=1).reshape(-1).tolist()),
            ),
            "o_proj.biases": (
                "BF16",
                [16, k_mix // 64],
                bf16(np.repeat(biases_g128[:16], 2, axis=1).reshape(-1).tolist()),
            ),
            "down_proj.weight": (
                "U32",
                [n_g128, k_mix * 8 // 32],
                np.ascontiguousarray(codes_g128).tobytes(),
            ),
            "down_proj.scales": (
                "BF16",
                [n_g128, k_mix // 128],
                bf16(scales_g128.reshape(-1).tolist()),
            ),
            "down_proj.biases": (
                "BF16",
                [n_g128, k_mix // 128],
                bf16(biases_g128.reshape(-1).tolist()),
            ),
        },
    )
    mix_untiled = root / "mixed-untiled.omlxm5"
    mix_pretiled = root / "mixed-pretiled.omlxm5"
    convert(mix_src, mix_untiled, "mlx_mxfp4")
    artifact.pretile_artifact(mix_untiled, mix_pretiled)
    _, mix_u_man = artifact.read_artifact(mix_untiled, verify_payload=True)
    _, mix_p_man = artifact.read_artifact(mix_pretiled, verify_payload=True)
    mix_objs = {obj["name"]: obj for obj in mix_p_man["objects"]}
    expect(
        mix_objs["q_proj"]["layout"] == "mx_nax_tiled_bk128_v1"
        and mix_objs["o_proj"]["layout"] == "oq_nax_tiled_g64_v1"
        and mix_objs["down_proj"]["layout"] == "oq_nax_tiled_g64_v1",
        "mixed MXFP4 + affine8 NAX layouts mismatch",
    )
    for obj_name in ("q_proj", "o_proj", "down_proj"):
        row_count = mix_objs[obj_name]["shape"][0]
        for r in range(row_count):
            expect(
                artifact.decode_artifact_row(mix_untiled, mix_u_man, obj_name, r, 0, k_mix)
                == artifact.decode_artifact_row(mix_pretiled, mix_p_man, obj_name, r, 0, k_mix),
                f"mixed {obj_name} row {r} untiled vs pretiled mismatch",
            )
        if cpp_inspector is not None:
            native_mix = json.loads(
                subprocess.check_output(
                    [
                        str(cpp_inspector),
                        str(mix_pretiled),
                        "--verify-payload",
                        "--sample-object",
                        obj_name,
                        "--sample-row",
                        "0",
                        "--sample-count",
                        "64",
                    ],
                    text=True,
                )
            )
            expect(
                native_mix["sample"]["values"]
                == artifact.decode_artifact_row(
                    mix_pretiled, mix_p_man, obj_name, 0, 0, 64
                ),
                f"mixed {obj_name} C++ pretiled decode mismatch",
            )

    # 3) Corruption checks for affine8 NAX tiled artifacts (rejected by Python & C++).
    bad_q8_high = root / "bad-q8-spurious-high2.omlxm5"
    rewrite_manifest(
        pure_pretiled,
        bad_q8_high,
        lambda m: next(o for o in m["objects"] if o["name"] == "q_proj")[
            "planes"
        ].insert(
            1,
            {
                "kind": "nax_high2",
                "offset": 0,
                "bytes": 256,
                "alignment": 256,
                "crc32": "00000000",
                "sha256": "0" * 64,
            },
        ),
    )
    expect_rejected(bad_q8_high, cpp_inspector)

    bad_q8_bytes = root / "bad-q8-half-codes.omlxm5"
    rewrite_manifest(
        pure_pretiled,
        bad_q8_bytes,
        lambda m: next(o for o in m["objects"] if o["name"] == "q_proj")["planes"][
            0
        ].__setitem__("bytes", 32 * k_g64 // 2),
    )
    expect_rejected(bad_q8_bytes, cpp_inspector)


def test_miv_v2_fwht64_pi64_roundtrip(
    root: Path, cpp_inspector: Path | None = None
) -> None:
    # 1. Exact Sylvester H_64 orthogonality & involutivity vs explicit matrix
    idx = np.arange(64, dtype=np.uint32)
    h64 = np.array(
        [
            [(-1.0 if bin(int(i & j)).count("1") & 1 else 1.0) / 8.0 for j in idx]
            for i in idx
        ],
        dtype=np.float32,
    )
    rng = np.random.default_rng(777)
    x_rand = rng.normal(0.0, 1.0, size=(7, 512)).astype(np.float32)
    y_fwht = x_rand.copy()
    artifact.fwht64_f32_inplace(y_fwht)
    y_ref = (x_rand.reshape(7, 8, 64) @ h64).reshape(7, 512)
    expect(
        float(np.max(np.abs(y_fwht - y_ref))) < 1e-6,
        "fwht64_f32_inplace disagrees with explicit Sylvester H_64",
    )
    y_back = y_fwht.copy()
    artifact.fwht64_f32_inplace(y_back)
    expect(
        float(np.max(np.abs(y_back - x_rand))) < 1e-6,
        "fwht64_f32_inplace is not involutive",
    )

    # 2. Mixed oQ4e (q3/q4/q5/q6/q8, g64/g128, 3D MoE experts, MTP) + MXFP4 v2 CLI convert
    k = 512
    n = 20
    tensors: dict[str, tuple[str, list[int], bytes]] = {}
    q_cfg: dict[str, object] = {"bits": 4, "group_size": 64, "mode": "affine"}
    specs = [
        ("model.layers.0.self_attn.q_proj", 4, 64, (n, k)),
        ("model.layers.0.self_attn.o_proj", 5, 64, (n, k)),
        ("model.layers.0.mlp.down_proj", 5, 64, (n, k)),
        ("model.layers.0.mlp.shared_expert.down_proj", 6, 128, (n, k)),
        ("model.layers.0.mlp.switch_mlp.down_proj", 4, 64, (4, 8, k)),
        ("model.layers.0.linear_attn.out_proj", 3, 64, (n, k)),
        ("language_model.mtp.layers.0.mlp.down_proj", 8, 128, (n, k)),
        ("language_model.mtp.layers.0.self_attn.o_proj", 6, 64, (n, k)),
    ]
    orig_w: dict[str, np.ndarray] = {}
    for mod, bits, gs, shp in specs:
        q_cfg[mod] = {"bits": bits, "group_size": gs, "mode": "affine"}
        w = rng.normal(0.0, 0.25, size=shp).astype(np.float32)
        codes, sc_u16, bi_u16, _, _ = artifact.quantize_oq_affine_f32(w, bits, gs)
        sc_f32 = artifact._bf16_u16_to_float32(sc_u16)
        bi_f32 = artifact._bf16_u16_to_float32(bi_u16)
        orig_w[mod] = (
            codes.astype(np.float32).reshape(*shp[:-1], k // gs, gs) * sc_f32[..., None]
            + bi_f32[..., None]
        ).reshape(shp)
        packed = artifact._pack_mlx_affine_u32(codes, bits)
        tensors[mod + ".weight"] = ("U32", list(packed.shape), packed.tobytes())
        tensors[mod + ".scales"] = ("BF16", list(sc_u16.shape), sc_u16.tobytes())
        tensors[mod + ".biases"] = ("BF16", list(bi_u16.shape), bi_u16.tobytes())
    tensors["model.norm.weight"] = ("BF16", [k], bf16([1.0] * k))

    oq_src = write_model(
        root / "v2_oq_src",
        "v2-oq-rev",
        {"model_type": "qwen3_8", "quantization": q_cfg},
        tensors,
    )
    out_j1 = root / "v2_oq_j1.miv"
    out_j4 = root / "v2_oq_j4.miv"
    out_untiled = root / "v2_oq_untiled.miv"
    out_twostep = root / "v2_oq_twostep.miv"
    out_no_oproj = root / "v2_oq_no_oproj.miv"
    out_no_fwht = root / "v2_oq_no_fwht.miv"

    for out_p, extra in (
        (out_j1, ["--nax-pretile", "--jobs", "1"]),
        (out_j4, ["--nax-pretile", "--jobs", "4"]),
        (out_untiled, ["--jobs", "2"]),
        (out_no_oproj, ["--nax-pretile", "--no-fwht64-o-proj", "--jobs", "2"]),
        (out_no_fwht, ["--nax-pretile", "--no-fwht64", "--jobs", "2"]),
    ):
        subprocess.run(
            [
                sys.executable,
                str(MIV_ROOT / "tools" / "m5_artifact.py"),
                "convert",
                "--source",
                str(oq_src),
                "--output",
                str(out_p),
                "--source-repo",
                "fixture/v2-oq",
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    subprocess.run(
        [
            sys.executable,
            str(MIV_ROOT / "tools" / "m5_artifact.py"),
            "pretile",
            "--input",
            str(out_untiled),
            "--output",
            str(out_twostep),
            "--jobs",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    expect(out_j1.read_bytes() == out_j4.read_bytes(), "--jobs 1 vs 4 bytes differ")
    expect(
        out_j1.read_bytes() == out_twostep.read_bytes(),
        "single-pass v2 convert --nax-pretile vs two-step pretile bytes differ",
    )

    summary, manifest = artifact.read_artifact(out_j4, verify_payload=True)
    expect(
        out_j4.read_bytes()[:8] == b"MIV00001"
        and summary["container_magic"] == "MIV00001"
        and summary["container_version"] == 2
        and summary["container_flags"] == 3
        and summary["schema"] == "miv-artifact-v2"
        and manifest["converter"]["name"] == "miv-artifact",
        "v2 summary header/magic/flags/schema/converter mismatch",
    )

    # Dual Magic/Schema interop: --artifact-version 1 (new branding) and --legacy-magic (v1 & v2)
    out_v1_new = root / "v1_oq_new.miv"
    out_v1_legacy = root / "v1_oq_legacy.miv"
    out_v2_legacy = root / "v2_oq_legacy.miv"
    for out_p, extra, exp_magic, exp_ver, exp_schema, exp_conv in (
        (
            out_v1_new,
            ["--artifact-version", "1", "--nax-pretile", "--jobs", "2"],
            b"MIV00001",
            1,
            "miv-artifact-v1",
            "miv-artifact",
        ),
        (
            out_v1_legacy,
            ["--artifact-version", "1", "--legacy-magic", "--nax-pretile", "--jobs", "2"],
            b"OMLXM5A1",
            1,
            "omlx-m5-artifact-v1",
            "omlx-m5-artifact",
        ),
        (
            out_v2_legacy,
            ["--legacy-magic", "--nax-pretile", "--jobs", "2"],
            b"OMLXM5A1",
            2,
            "omlx-m5-artifact-v2",
            "omlx-m5-artifact",
        ),
    ):
        subprocess.run(
            [
                sys.executable,
                str(MIV_ROOT / "tools" / "m5_artifact.py"),
                "convert",
                "--source",
                str(oq_src),
                "--output",
                str(out_p),
                "--source-repo",
                "fixture/v2-oq",
                *extra,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        expect(out_p.read_bytes()[:8] == exp_magic, f"{out_p.name} magic mismatch")
        s_b, m_b = artifact.read_artifact(out_p, verify_payload=True)
        expect(
            s_b["container_magic"] == exp_magic.decode("ascii")
            and s_b["container_version"] == exp_ver
            and s_b["schema"] == exp_schema
            and m_b["converter"]["name"] == exp_conv,
            f"{out_p.name} dual-branding summary/converter mismatch",
        )
        if cpp_inspector is not None:
            native_b = json.loads(
                subprocess.check_output(
                    [
                        str(cpp_inspector),
                        str(out_p),
                        "--verify-payload",
                        "--sample-object",
                        "model.layers.0.mlp.down_proj",
                        "--sample-row",
                        "0",
                        "--sample-start",
                        "0",
                        "--sample-count",
                        "64",
                    ],
                    text=True,
                )
            )
            expect(
                native_b["schema"] == "miv-artifact-inspection-v1"
                and native_b["container_magic"] == exp_magic.decode("ascii")
                and native_b["manifest_schema"] == exp_schema
                and native_b["container_version"] == exp_ver,
                f"{out_p.name} C++ inspector dual-branding metadata mismatch",
            )
            py_b = artifact.decode_artifact_row(
                out_p, m_b, "model.layers.0.mlp.down_proj", 0, 0, 64
            )
            expect(
                np.allclose(native_b["sample"]["values"], py_b, rtol=1e-6, atol=1e-6),
                f"{out_p.name} C++ vs Python dual-branding decode mismatch",
            )

    # Strict v2 version/schema mismatch rejection in both Python and C++
    for bad_name, bad_schema in (
        ("v2_bad_new_v1_schema.miv", "miv-artifact-v1"),
        ("v2_bad_legacy_v1_schema.miv", "omlx-m5-artifact-v1"),
    ):
        bad_v2_p = root / bad_name
        rewrite_manifest(
            out_j4,
            bad_v2_p,
            lambda value, s=bad_schema: value.__setitem__("schema", s),
        )
        expect_rejected(bad_v2_p, cpp_inspector)
    objs = {o["name"]: o for o in manifest["objects"]}
    expect(objs["model.layers.0.self_attn.q_proj"]["layout"] == "oq_nax_lane_g64_v2", "q_proj layout")
    expect(objs["model.layers.0.self_attn.q_proj"].get("input_transform", "none") == "none", "q_proj unrotated")
    expect(objs["model.layers.0.linear_attn.out_proj"]["layout"] == "oq_nax_lane_g64_v2", "gdn out layout")
    expect(objs["model.layers.0.linear_attn.out_proj"].get("input_transform", "none") == "none", "gdn out unrotated")
    expect(objs["model.layers.0.mlp.down_proj"]["layout"] == "oq_nax_lane_g64_v2", "down_proj layout")
    expect(objs["model.layers.0.mlp.down_proj"]["input_transform"] == "fwht64", "down_proj rotated")
    expect(objs["model.layers.0.self_attn.o_proj"]["input_transform"] == "fwht64", "o_proj rotated")
    expect(objs["model.layers.0.mlp.shared_expert.down_proj"]["layout"] == "oq_nax_lane_g128_v2", "shared g128 layout")
    expect(objs["model.layers.0.mlp.shared_expert.down_proj"]["input_transform"] == "fwht64", "shared down rotated")
    expect(objs["model.layers.0.mlp.switch_mlp.down_proj"]["layout"] == "oq_expert_rowsplit_k128_v1", "expert layout")
    expect(objs["model.layers.0.mlp.switch_mlp.down_proj"]["input_transform"] == "fwht64", "expert down rotated")
    expect(objs["language_model.mtp.layers.0.mlp.down_proj"]["layout"] == "oq_nax_lane_g128_v2", "mtp down layout")
    expect(objs["language_model.mtp.layers.0.mlp.down_proj"]["input_transform"] == "fwht64", "mtp down rotated")
    expect(objs["language_model.mtp.layers.0.self_attn.o_proj"]["input_transform"] == "fwht64", "mtp o_proj rotated")

    # Verify --no-fwht64-o-proj and --no-fwht64
    s_no_o, m_no_o = artifact.read_artifact(out_no_oproj, verify_payload=True)
    o_no_o = {o["name"]: o for o in m_no_o["objects"]}
    expect(s_no_o["container_flags"] == 3, "--no-fwht64-o-proj flags")
    expect(o_no_o["model.layers.0.self_attn.o_proj"].get("input_transform", "none") == "none", "o_proj should not rotate")
    expect(o_no_o["model.layers.0.mlp.down_proj"]["input_transform"] == "fwht64", "down_proj should rotate")

    s_nf, m_nf = artifact.read_artifact(out_no_fwht, verify_payload=True)
    expect(s_nf["container_flags"] == 2, "--no-fwht64 should only set NAX_LANE_V2 flag")
    expect(all(o.get("input_transform", "none") == "none" for o in m_nf["objects"]), "--no-fwht64 rotated something")

    # Verify SQNR on rotated objects & exact bit-match on unrotated objects
    for mod, bits, gs, shp in specs:
        flat_rows = int(np.prod(shp[:-1]))
        dec = np.asarray(
            [
                artifact.decode_artifact_row(out_j4, manifest, mod, r, 0, k)
                for r in range(flat_rows)
            ],
            dtype=np.float32,
        ).reshape(shp)
        ref = orig_w[mod]
        if objs[mod].get("input_transform") == "fwht64":
            err = float(np.sum((dec - ref) ** 2))
            sig = float(np.sum(ref ** 2))
            sqnr_db = 10.0 * np.log10(sig / max(err, 1e-30))
            min_sqnr = {3: 14.5, 4: 20.5, 5: 26.8, 6: 32.5, 8: 43.0}[bits]
            expect(sqnr_db > min_sqnr, f"{mod} (q{bits}) SQNR too low: {sqnr_db:.2f} dB")
        else:
            expect(np.array_equal(dec, ref), f"unrotated {mod} changed in v2")

        if cpp_inspector is not None:
            native = json.loads(
                subprocess.check_output(
                    [
                        str(cpp_inspector),
                        str(out_j4),
                        "--verify-payload",
                        "--sample-object",
                        mod,
                        "--sample-row",
                        "0",
                        "--sample-start",
                        "13",
                        "--sample-count",
                        "77",
                    ],
                    text=True,
                )
            )
            py_slice = artifact.decode_artifact_row(out_j4, manifest, mod, 0, 13, 77)
            expect(
                np.allclose(native["sample"]["values"], py_slice, rtol=1e-6, atol=1e-6),
                f"C++ vs Python v2 decode mismatch on {mod}",
            )

    # 3. MXFP4 v2 with FWHT-64 (2D down_proj + 3D MoE expert down)
    mx_down_w = rng.normal(0.0, 0.25, size=(16, k)).astype(np.float32)
    mx_exp_w = rng.normal(0.0, 0.25, size=(2, 8, k)).astype(np.float32)
    mx_d_p, mx_d_s, _, _ = artifact.quantize_mxfp4_e2m1_f32(mx_down_w)
    mx_e_p, mx_e_s, _, _ = artifact.quantize_mxfp4_e2m1_f32(mx_exp_w)
    mx_d_ref = artifact._dequantize_mxfp4_u8_f32(mx_d_p, mx_d_s)
    mx_src = write_model(
        root / "v2_mx_src",
        "v2-mx-rev",
        {"model_type": "qwen3_6_moe", "quantization": {"bits": 4, "group_size": 32, "mode": "mxfp4"}},
        {
            "model.layers.0.mlp.down_proj.weight": ("U32", [16, k // 8], mx_d_p.tobytes()),
            "model.layers.0.mlp.down_proj.scales": ("U8", [16, k // 32], mx_d_s.tobytes()),
            "model.layers.0.mlp.switch_mlp.down_proj.weight": ("U32", [2, 8, k // 8], mx_e_p.tobytes()),
            "model.layers.0.mlp.switch_mlp.down_proj.scales": ("U8", [2, 8, k // 32], mx_e_s.tobytes()),
        },
    )
    mx_v2 = root / "v2_mx.miv"
    subprocess.run(
        [
            sys.executable,
            str(MIV_ROOT / "tools" / "m5_artifact.py"),
            "convert",
            "--source",
            str(mx_src),
            "--output",
            str(mx_v2),
            "--source-repo",
            "fixture/v2-mx",
            "--nax-pretile",
            "--jobs",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    mx_sum, mx_man = artifact.read_artifact(mx_v2, verify_payload=True)
    expect(mx_sum["container_version"] == 2 and mx_sum["container_flags"] == 1, "MXFP4 v2 flags should be 0x1")
    mx_objs = {o["name"]: o for o in mx_man["objects"]}
    expect(
        mx_objs["model.layers.0.mlp.down_proj"]["layout"] == "mx_nax_tiled_bk128_v1"
        and mx_objs["model.layers.0.mlp.down_proj"]["input_transform"] == "fwht64",
        "MXFP4 2D down_proj layout/transform mismatch",
    )
    expect(
        mx_objs["model.layers.0.mlp.switch_mlp.down_proj"]["layout"] == "mx_expert_rowsplit_k128_v1"
        and mx_objs["model.layers.0.mlp.switch_mlp.down_proj"]["input_transform"] == "fwht64",
        "MXFP4 3D expert down layout/transform mismatch",
    )
    mx_dec = np.asarray(
        [artifact.decode_artifact_row(mx_v2, mx_man, "model.layers.0.mlp.down_proj", r, 0, k) for r in range(16)],
        dtype=np.float32,
    )
    mx_sqnr = 10.0 * np.log10(float(np.sum(mx_d_ref ** 2)) / max(float(np.sum((mx_dec - mx_d_ref) ** 2)), 1e-30))
    expect(mx_sqnr > 17.5, f"MXFP4 FWHT-64 3-exp SQNR too low: {mx_sqnr:.2f} dB")
    if cpp_inspector is not None:
        native = json.loads(
            subprocess.check_output(
                [
                    str(cpp_inspector),
                    str(mx_v2),
                    "--verify-payload",
                    "--sample-object",
                    "model.layers.0.mlp.down_proj",
                    "--sample-row",
                    "3",
                    "--sample-start",
                    "5",
                    "--sample-count",
                    "90",
                ],
                text=True,
            )
        )
        py_mx = artifact.decode_artifact_row(mx_v2, mx_man, "model.layers.0.mlp.down_proj", 3, 5, 90)
        expect(np.allclose(native["sample"]["values"], py_mx, rtol=1e-6, atol=1e-6), "MXFP4 v2 C++ decode mismatch")


# Mark helper functions taking positional non-fixture args so pytest collection
# runs the dedicated pytest entrypoints below while preserving CLI & import APIs.
test_roundtrip.__test__ = False  # type: ignore[attr-defined]
test_corruption.__test__ = False  # type: ignore[attr-defined]
test_unsafe_source_index.__test__ = False  # type: ignore[attr-defined]
test_affine8_nax_pretile_roundtrip.__test__ = False  # type: ignore[attr-defined]
test_miv_v2_fwht64_pi64_roundtrip.__test__ = False  # type: ignore[attr-defined]


def _discover_cpp_inspector() -> Path | None:
    env_path = os.environ.get("OMLX_M5_CPP_INSPECTOR")
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate.resolve()
    for rel in (
        "build/release/miv/omlx_m5_inspect_artifact",
        "build/release/omlx_m5_inspect_artifact",
    ):
        candidate = ROOT / rel
        if candidate.is_file():
            return candidate.resolve()
    return None


def test_pytest_b0_affine8_nax_pretile_roundtrip(tmp_path: Path) -> None:
    test_affine8_nax_pretile_roundtrip(tmp_path, cpp_inspector=None)
    test_miv_v2_fwht64_pi64_roundtrip(tmp_path / "v2", cpp_inspector=None)
    test_unsafe_source_index(tmp_path)


def test_pytest_b0_full_roundtrip_with_cpp_inspector(tmp_path: Path) -> None:
    import pytest

    inspector = _discover_cpp_inspector()
    if inspector is None:
        pytest.skip("omlx_m5_inspect_artifact not yet built in build/release")
    golden_path = MIV_ROOT / "tests" / "data" / "b0_golden_vectors.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    fixtures, codes = make_fixtures(tmp_path / "sources")
    outputs = test_roundtrip(tmp_path, fixtures, codes, golden, inspector)
    test_affine8_nax_pretile_roundtrip(tmp_path / "pretile", inspector)
    test_miv_v2_fwht64_pi64_roundtrip(tmp_path / "v2", inspector)
    oq_path, oq_manifest = outputs["oq"]
    test_corruption(tmp_path, oq_path, copy.deepcopy(oq_manifest), inspector)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-inspector", type=Path, required=True)
    parser.add_argument("--golden", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    golden = json.loads(args.golden.read_text())
    expect(golden["schema"] == "omlx-m5-b0-golden-v1", "golden schema mismatch")
    inspector = args.cpp_inspector.resolve()
    with tempfile.TemporaryDirectory(prefix="omlx-m5-b0-") as temporary:
        root = Path(temporary)
        fixtures, codes = make_fixtures(root / "sources")
        outputs = test_roundtrip(root, fixtures, codes, golden, inspector)
        test_affine8_nax_pretile_roundtrip(root / "pretile", inspector)
        test_miv_v2_fwht64_pi64_roundtrip(root / "v2", inspector)
        oq_path, oq_manifest = outputs["oq"]
        test_corruption(root, oq_path, copy.deepcopy(oq_manifest), inspector)
        test_unsafe_source_index(root)
    print("B0 artifact round-trip tests passed")


if __name__ == "__main__":
    main()
