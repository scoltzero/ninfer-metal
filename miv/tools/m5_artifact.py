#!/usr/bin/env python3
"""Build and inspect deterministic omlx-m5 backend artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import struct
import tempfile
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import BinaryIO

import numpy as np

MAGIC = b"OMLXM5A1"
VERSION = 1
HEADER_BYTES = 64
PAYLOAD_ALIGNMENT = 16384
LEGACY_PAYLOAD_ALIGNMENT = 4096
OBJECT_ALIGNMENT = 256
PLANE_ALIGNMENT = 256
HEADER_STRUCT = struct.Struct("<8sIIQQQQIIII")
SCHEMA = "omlx-m5-artifact-v1"

DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "U16": 2,
    "I16": 2,
    "F16": 2,
    "BF16": 2,
    "U32": 4,
    "I32": 4,
    "F32": 4,
    "U64": 8,
    "I64": 8,
    "F64": 8,
}


def align_up(value: int, alignment: int) -> int:
    if value < 0 or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError("alignment must be a positive power of two")
    return (value + alignment - 1) & -alignment


def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def crc32(data: bytes | bytearray | memoryview, initial: int = 0) -> int:
    return zlib.crc32(data, initial) & 0xFFFFFFFF


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_range(
    handle: BinaryIO,
    offset: int,
    size: int,
    chunk_bytes: int = 16 * 1024 * 1024,
) -> str:
    digest = hashlib.sha256()
    current = handle.tell()
    try:
        handle.seek(offset)
        remaining = size
        while remaining:
            chunk = handle.read(min(remaining, chunk_bytes))
            if not chunk:
                raise ValueError("temporary payload is truncated")
            digest.update(chunk)
            remaining -= len(chunk)
    finally:
        handle.seek(current)
    return digest.hexdigest()


def file_sha256_if_present(path: Path) -> str:
    return sha256_file(path) if path.is_file() else ""


def infer_weight_use(name: str) -> str:
    if name.endswith("embed_tokens"):
        return "embedding"
    if name.endswith("lm_head"):
        return "lm_head"
    if ".mlp.switch_mlp.gate_proj" in name:
        return "moe_expert_gate"
    if ".mlp.switch_mlp.up_proj" in name:
        return "moe_expert_up"
    if ".mlp.switch_mlp.down_proj" in name:
        return "moe_expert_down"
    if name.endswith(".mlp.gate"):
        return "moe_router"
    if name.endswith(".mlp.shared_expert_gate"):
        return "moe_shared_gate"
    if ".mlp.shared_expert.gate_proj" in name:
        return "moe_shared_expert_gate"
    if ".mlp.shared_expert.up_proj" in name:
        return "moe_shared_expert_up"
    if ".mlp.shared_expert.down_proj" in name:
        return "moe_shared_expert_down"
    if ".mlp.gate_proj" in name:
        return "mlp_gate"
    if ".mlp.up_proj" in name:
        return "mlp_up"
    if ".mlp.down_proj" in name:
        return "mlp_down"
    if ".linear_attn.in_proj_qkv" in name:
        return "gdn_qkv"
    if ".linear_attn.in_proj_z" in name:
        return "gdn_z"
    if ".linear_attn.in_proj_a" in name:
        return "gdn_a"
    if ".linear_attn.in_proj_b" in name:
        return "gdn_b"
    if ".linear_attn.out_proj" in name:
        return "gdn_out"
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if f".self_attn.{projection}" in name:
            return f"attention_{projection.removesuffix('_proj')}"
    return "linear"


def product(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        if value <= 0:
            raise ValueError("tensor dimensions must be positive")
        result *= value
    return result


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: str
    shape: tuple[int, ...]
    shard: str
    data_offset: int
    nbytes: int


class MappedShard:
    def __init__(self, path: Path):
        self.path = path
        self._file = path.open("rb")
        try:
            self.mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._file.close()
            raise
        if len(self.mapping) < 8:
            raise ValueError(f"{path}: safetensors prefix is missing")
        header_bytes = struct.unpack_from("<Q", self.mapping, 0)[0]
        if not 0 < header_bytes <= 128 * 1024 * 1024:
            raise ValueError(f"{path}: invalid safetensors header length")
        self.data_start = 8 + header_bytes
        if self.data_start > len(self.mapping):
            raise ValueError(f"{path}: truncated safetensors header")
        try:
            header = json.loads(self.mapping[8 : self.data_start])
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{path}: invalid safetensors JSON: {exc}") from exc
        self.tensors: dict[str, TensorInfo] = {}
        ranges: list[tuple[int, int]] = []
        for name, record in header.items():
            if name == "__metadata__":
                continue
            dtype = str(record["dtype"])
            shape = tuple(int(value) for value in record["shape"])
            offsets = tuple(int(value) for value in record["data_offsets"])
            if dtype not in DTYPE_BYTES or len(offsets) != 2 or offsets[1] < offsets[0]:
                raise ValueError(f"{path}: invalid tensor record {name}")
            nbytes = offsets[1] - offsets[0]
            elements = 1
            for dimension in shape:
                if dimension < 0:
                    raise ValueError(f"{path}: negative tensor dimension for {name}")
                elements *= dimension
            if elements * DTYPE_BYTES[dtype] != nbytes:
                raise ValueError(f"{path}: tensor byte count mismatch for {name}")
            begin = self.data_start + offsets[0]
            end = self.data_start + offsets[1]
            if end > len(self.mapping):
                raise ValueError(f"{path}: tensor extends beyond shard: {name}")
            ranges.append((begin, end))
            self.tensors[name] = TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                shard=path.name,
                data_offset=begin,
                nbytes=nbytes,
            )
        ranges.sort()
        for previous, current in pairwise(ranges):
            if current[0] < previous[1]:
                raise ValueError(f"{path}: overlapping tensor payloads")

    def view(self, info: TensorInfo) -> memoryview:
        return memoryview(self.mapping)[
            info.data_offset : info.data_offset + info.nbytes
        ]

    def close(self) -> None:
        self.mapping.close()
        self._file.close()


class SafeTensorSource:
    def __init__(self, model_dir: Path, expected_revision: str | None = None):
        self.model_dir = model_dir.resolve()
        if not self.model_dir.is_dir():
            raise ValueError(f"model directory does not exist: {model_dir}")
        revision_file = self.model_dir / "REVISION"
        self.revision = (
            revision_file.read_text().strip() if revision_file.exists() else ""
        )
        if expected_revision and self.revision != expected_revision:
            raise ValueError(
                f"revision mismatch: expected {expected_revision}, found {self.revision or '<none>'}"
            )
        self.config = json.loads((self.model_dir / "config.json").read_text())
        index_path = self.model_dir / "model.safetensors.index.json"
        if index_path.exists():
            index = json.loads(index_path.read_text())
            weight_map = {str(k): str(v) for k, v in index["weight_map"].items()}
            shard_names = sorted(set(weight_map.values()))
        else:
            single = self.model_dir / "model.safetensors"
            if not single.exists():
                raise ValueError("no safetensors index or model.safetensors found")
            shard_names = [single.name]
            weight_map = {}
        for shard_name in shard_names:
            if Path(shard_name).name != shard_name:
                raise ValueError(f"unsafe shard name: {shard_name}")
        self.shards: dict[str, MappedShard] = {}
        try:
            for shard_name in shard_names:
                path = (self.model_dir / shard_name).resolve()
                if path.parent != self.model_dir:
                    raise ValueError(f"unsafe shard path: {shard_name}")
                self.shards[shard_name] = MappedShard(path)
        except Exception:
            self.close()
            raise
        if not weight_map:
            for name, shard in self.shards.items():
                for tensor_name in shard.tensors:
                    if tensor_name in weight_map:
                        raise ValueError(f"duplicate tensor {tensor_name}")
                    weight_map[tensor_name] = name
        self.tensors: dict[str, TensorInfo] = {}
        for tensor_name, shard_name in weight_map.items():
            if Path(shard_name).name != shard_name:
                raise ValueError(f"unsafe shard name: {shard_name}")
            shard = self.shards.get(shard_name)
            if shard is None or tensor_name not in shard.tensors:
                raise ValueError(f"index points to missing tensor {tensor_name}")
            self.tensors[tensor_name] = shard.tensors[tensor_name]
        actual_names = {
            name for shard in self.shards.values() for name in shard.tensors
        }
        if actual_names != set(self.tensors):
            raise ValueError("safetensors index does not exactly cover shard tensors")

    def view(self, name: str) -> memoryview:
        info = self.tensors[name]
        return self.shards[info.shard].view(info)

    def close(self) -> None:
        for shard in self.shards.values():
            shard.close()
        self.shards.clear()


@dataclass
class PlaneRecipe:
    kind: str
    logical_bytes: int
    writer: Callable[[BinaryIO], tuple[int, int, str]]
    alignment: int = PLANE_ALIGNMENT


@dataclass
class ObjectRecipe:
    name: str
    format_id: str
    layout_id: str
    shape: tuple[int, ...]
    padded_k: int | None
    planes: list[PlaneRecipe]
    source_tensors: list[str]
    binding: dict | None = None


class PayloadWriter:
    def __init__(self, handle: BinaryIO):
        self.handle = handle

    @property
    def offset(self) -> int:
        return self.handle.tell()

    def align(self, alignment: int) -> None:
        target = align_up(self.offset, alignment)
        if target != self.offset:
            self.handle.write(b"\0" * (target - self.offset))

    def write_chunks(
        self, chunks: Iterable[bytes | memoryview]
    ) -> tuple[int, int, str]:
        total = 0
        checksum = 0
        digest = hashlib.sha256()
        for chunk in chunks:
            view = memoryview(chunk)
            if not view:
                continue
            self.handle.write(view)
            total += len(view)
            checksum = crc32(view, checksum)
            digest.update(view)
        return total, checksum, digest.hexdigest()


def bytes_chunks(
    data: memoryview, chunk_bytes: int = 16 * 1024 * 1024
) -> Iterator[memoryview]:
    for offset in range(0, len(data), chunk_bytes):
        yield data[offset : offset + chunk_bytes]


def padded_row_chunks(
    data: memoryview,
    rows: int,
    source_row_bytes: int,
    target_row_bytes: int,
) -> Iterator[bytes | memoryview]:
    if target_row_bytes < source_row_bytes or len(data) != rows * source_row_bytes:
        raise ValueError("invalid padded-row geometry")
    if source_row_bytes == target_row_bytes:
        yield from bytes_chunks(data)
        return
    padding = b"\0" * (target_row_bytes - source_row_bytes)
    for row in range(rows):
        begin = row * source_row_bytes
        yield data[begin : begin + source_row_bytes]
        yield padding


def affine_code_plane_chunks(
    data: memoryview,
    rows: int,
    logical_k: int,
    padded_k: int,
    bits_per_code: int,
    plane: str,
    rows_per_chunk: int = 16,
) -> Iterator[bytes]:
    if bits_per_code not in (3, 5, 6):
        raise ValueError("split affine planes require Q3, Q5 or Q6")
    source_row_bytes = logical_k * bits_per_code // 8
    if (
        len(data) != rows * source_row_bytes
        or (logical_k * bits_per_code) % 8
        or padded_k % 8
    ):
        raise ValueError("invalid split affine source geometry")
    raw = np.frombuffer(data, dtype=np.uint8).reshape(rows, source_row_bytes)
    weights = (1 << np.arange(bits_per_code, dtype=np.uint8)).astype(np.uint8)
    for begin in range(0, rows, rows_per_chunk):
        packed = raw[begin : begin + rows_per_chunk]
        bits = np.unpackbits(packed, axis=1, bitorder="little")
        bits = bits[:, : logical_k * bits_per_code].reshape(
            packed.shape[0], logical_k, bits_per_code
        )
        codes = np.sum(bits * weights, axis=2, dtype=np.uint16).astype(np.uint8)
        if padded_k != logical_k:
            physical = np.zeros((codes.shape[0], padded_k), dtype=np.uint8)
            physical[:, :logical_k] = codes
            codes = physical
        if plane == "base_code" and bits_per_code in (5, 6):
            out = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
        elif plane == "low_2bit" and bits_per_code == 3:
            low = codes & 0x03
            out = (
                low[:, 0::4]
                | (low[:, 1::4] << 2)
                | (low[:, 2::4] << 4)
                | (low[:, 3::4] << 6)
            )
        elif plane == "high_bit" and bits_per_code in (3, 5):
            shift = 2 if bits_per_code == 3 else 4
            out = np.packbits((codes >> shift) & 1, axis=1, bitorder="little")
        elif plane == "high_2bit" and bits_per_code == 6:
            high = (codes >> 4) & 0x03
            out = (
                high[:, 0::4]
                | (high[:, 1::4] << 2)
                | (high[:, 2::4] << 4)
                | (high[:, 3::4] << 6)
            )
        else:
            raise ValueError(f"unknown Q{bits_per_code} plane {plane}")
        yield out.tobytes(order="C")


def _quantization_table(config: dict) -> dict:
    return config.get("quantization") or config.get("quantization_config") or {}


def detect_profile(source: SafeTensorSource) -> str:
    architectures = tuple(source.config.get("architectures") or ())
    if architectures == ("DFlash2DraftModel",):
        return "dflash2_bf16"
    quant = _quantization_table(source.config)
    mode = quant.get("mode")
    if mode == "affine" and any(name.endswith(".biases") for name in source.tensors):
        return "oq_affine"
    if mode == "mxfp4" and any(name.endswith(".scales") for name in source.tensors):
        return "mlx_mxfp4"
    qconfig = source.config.get("quantization_config") or {}
    fmt = qconfig.get("format")
    if fmt == "mxfp4-pack-quantized" and any(
        name.endswith(".weight_packed") for name in source.tensors
    ):
        return "compressed_mxfp4"
    raise ValueError("unable to detect a supported B0 source profile")


def quant_spec(config: dict, module: str) -> dict:
    table = _quantization_table(config)
    spec = {
        "bits": int(table.get("bits", 0)),
        "group_size": int(table.get("group_size", 0)),
        "mode": str(table.get("mode", "")),
    }
    if module in table and isinstance(table[module], dict):
        spec.update(table[module])
    return spec


def _plane_from_tensor(
    source: SafeTensorSource,
    tensor_name: str,
    kind: str,
    rows: int,
    source_row_bytes: int,
    target_row_bytes: int,
) -> PlaneRecipe:
    def writer(handle: BinaryIO) -> tuple[int, int, str]:
        view = source.view(tensor_name)
        try:
            return PayloadWriter(handle).write_chunks(
                padded_row_chunks(view, rows, source_row_bytes, target_row_bytes)
            )
        finally:
            view.release()

    return PlaneRecipe(kind=kind, logical_bytes=rows * target_row_bytes, writer=writer)


def _build_affine_recipe(
    source: SafeTensorSource,
    module: str,
    *,
    require_affine_mode: bool,
) -> ObjectRecipe:
    weight_name = module + ".weight"
    scales_name = module + ".scales"
    biases_name = module + ".biases"
    weight = source.tensors[weight_name]
    scales = source.tensors[scales_name]
    biases = source.tensors[biases_name]
    spec = quant_spec(source.config, module)
    bits = int(spec["bits"])
    group_size = int(spec["group_size"])
    if (
        (require_affine_mode and spec.get("mode") != "affine")
        or bits not in (3, 4, 5, 6, 8)
        or group_size not in (64, 128)
    ):
        raise ValueError(f"{module}: unsupported affine spec {spec}")
    if weight.dtype != "U32" or scales.dtype != "BF16" or biases.dtype != "BF16":
        raise ValueError(f"{module}: affine tensors have unexpected dtypes")
    if len(weight.shape) < 2 or scales.shape != biases.shape or len(scales.shape) < 2:
        raise ValueError(f"{module}: affine tensors have unexpected shapes")
    leading_shape = scales.shape[:-1]
    rows = product(leading_shape)
    groups = scales.shape[-1]
    logical_k = groups * group_size
    padded_k = align_up(logical_k, 128)
    if weight.shape != (*leading_shape, logical_k * bits // 32):
        raise ValueError(f"{module}: packed weight shape disagrees with quantization")
    source_code_row = logical_k * bits // 8
    planes: list[PlaneRecipe] = []
    if bits == 4:
        planes.append(
            _plane_from_tensor(
                source,
                weight_name,
                "base_code",
                rows,
                source_code_row,
                padded_k // 2,
            )
        )
    elif bits in (3, 5, 6):
        if bits == 3:
            split_specs = (
                ("low_2bit", rows * padded_k // 4),
                ("high_bit", rows * padded_k // 8),
            )
        else:
            high_kind = "high_bit" if bits == 5 else "high_2bit"
            high_bytes = rows * padded_k // (8 if bits == 5 else 4)
            split_specs = (
                ("base_code", rows * padded_k // 2),
                (high_kind, high_bytes),
            )
        for plane_kind, plane_bytes in split_specs:

            def writer(
                handle: BinaryIO,
                kind: str = plane_kind,
                tensor_name: str = weight_name,
                output_rows: int = rows,
                input_k: int = logical_k,
                output_k: int = padded_k,
                code_bits: int = bits,
            ) -> tuple[int, int, str]:
                view = source.view(tensor_name)
                try:
                    return PayloadWriter(handle).write_chunks(
                        affine_code_plane_chunks(
                            view,
                            output_rows,
                            input_k,
                            output_k,
                            code_bits,
                            kind,
                        )
                    )
                finally:
                    view.release()

            planes.append(PlaneRecipe(plane_kind, plane_bytes, writer))
    else:
        planes.append(
            _plane_from_tensor(
                source,
                weight_name,
                "code",
                rows,
                source_code_row,
                padded_k,
            )
        )
    planes.extend(
        [
            _plane_from_tensor(
                source,
                scales_name,
                "scale",
                rows,
                groups * 2,
                padded_k // group_size * 2,
            ),
            _plane_from_tensor(
                source,
                biases_name,
                "bias",
                rows,
                groups * 2,
                padded_k // group_size * 2,
            ),
        ]
    )
    format_id = f"oq_affine_q{bits}_g{group_size}_bf16_v1"
    shape = (*leading_shape, logical_k)
    return ObjectRecipe(
        name=module,
        format_id=format_id,
        layout_id=(
            "oq_rowsplit_k128_v1" if len(shape) == 2 else "oq_expert_rowsplit_k128_v1"
        ),
        shape=shape,
        padded_k=padded_k,
        planes=planes,
        source_tensors=[weight_name, scales_name, biases_name],
        binding={
            "name": module,
            "format": format_id,
            "use": infer_weight_use(module),
            "activation_policy": "A16Only",
            "leading_shape": list(leading_shape[:-1]),
            "out_features": leading_shape[-1],
            "in_features": logical_k,
        },
    )


def build_oq_recipes(source: SafeTensorSource) -> tuple[list[ObjectRecipe], set[str]]:
    recipes: list[ObjectRecipe] = []
    consumed: set[str] = set()
    for weight_name in sorted(
        name for name in source.tensors if name.endswith(".weight")
    ):
        module = weight_name[: -len(".weight")]
        source_names = [module + suffix for suffix in (".weight", ".scales", ".biases")]
        if not all(name in source.tensors for name in source_names):
            continue
        recipe = _build_affine_recipe(source, module, require_affine_mode=True)
        recipes.append(recipe)
        consumed.update(source_names)
    if not recipes:
        raise ValueError("oQ source contains no affine modules")
    return recipes, consumed


def build_mlx_mxfp4_recipes(
    source: SafeTensorSource,
) -> tuple[list[ObjectRecipe], set[str]]:
    recipes: list[ObjectRecipe] = []
    consumed: set[str] = set()
    for weight_name in sorted(
        name for name in source.tensors if name.endswith(".weight")
    ):
        module = weight_name[: -len(".weight")]
        scales_name = module + ".scales"
        biases_name = module + ".biases"
        if scales_name not in source.tensors:
            continue
        if biases_name in source.tensors:
            recipe = _build_affine_recipe(source, module, require_affine_mode=False)
            recipes.append(recipe)
            consumed.update([weight_name, scales_name, biases_name])
            continue
        weight = source.tensors[weight_name]
        scales = source.tensors[scales_name]
        if weight.dtype != "U32" or scales.dtype not in ("U8", "F8_E8M0"):
            continue
        if len(weight.shape) < 2 or len(scales.shape) < 2:
            raise ValueError(f"{module}: MXFP4 tensors must have rank >= 2")
        leading_shape = scales.shape[:-1]
        rows = product(leading_shape)
        scale_groups = scales.shape[-1]
        logical_k = scale_groups * 32
        padded_k = align_up(logical_k, 128)
        if weight.shape != (*leading_shape, logical_k // 8):
            raise ValueError(f"{module}: MXFP4 weight shape mismatch")
        source_names = [weight_name, scales_name]
        consumed.update(source_names)
        format_id = "mxfp4_e2m1_g32_e8m0_v1"
        recipes.append(
            ObjectRecipe(
                name=module,
                format_id=format_id,
                layout_id=(
                    "mx_rowsplit_k128_v1"
                    if len(leading_shape) == 1
                    else "mx_expert_rowsplit_k128_v1"
                ),
                shape=(*leading_shape, logical_k),
                padded_k=padded_k,
                planes=[
                    _plane_from_tensor(
                        source,
                        weight_name,
                        "data",
                        rows,
                        logical_k // 2,
                        padded_k // 2,
                    ),
                    _plane_from_tensor(
                        source,
                        scales_name,
                        "scale",
                        rows,
                        scale_groups,
                        padded_k // 32,
                    ),
                ],
                source_tensors=source_names,
                binding={
                    "name": module,
                    "format": format_id,
                    "use": infer_weight_use(module),
                    "activation_policy": "AllowA4",
                    "leading_shape": list(leading_shape[:-1]),
                    "out_features": leading_shape[-1],
                    "in_features": logical_k,
                },
            )
        )
    if not recipes:
        raise ValueError("MLX source contains no MXFP4 modules")
    return recipes, consumed


def build_compressed_mxfp4_recipes(
    source: SafeTensorSource,
) -> tuple[list[ObjectRecipe], set[str]]:
    recipes: list[ObjectRecipe] = []
    consumed: set[str] = set()
    for packed_name in sorted(
        name for name in source.tensors if name.endswith(".weight_packed")
    ):
        module = packed_name[: -len(".weight_packed")]
        scale_name = module + ".weight_scale"
        if scale_name not in source.tensors:
            raise ValueError(f"{module}: compressed MXFP4 scale is missing")
        packed = source.tensors[packed_name]
        scales = source.tensors[scale_name]
        if packed.dtype not in ("U8", "I8") or scales.dtype not in ("U8", "F8_E8M0"):
            raise ValueError(f"{module}: compressed MXFP4 dtypes are unsupported")
        if len(packed.shape) < 2 or len(scales.shape) < 2:
            raise ValueError(f"{module}: compressed MXFP4 tensors must have rank >= 2")
        leading_shape = packed.shape[:-1]
        rows = product(leading_shape)
        packed_columns = packed.shape[-1]
        logical_k = packed_columns * 2
        if scales.shape != (*leading_shape, logical_k // 32):
            raise ValueError(f"{module}: compressed MXFP4 scale shape mismatch")
        padded_k = align_up(logical_k, 128)
        source_names = [packed_name, scale_name]
        shape_name = module + ".weight_shape"
        if shape_name in source.tensors:
            source_names.append(shape_name)
        consumed.update(source_names)
        format_id = "mxfp4_e2m1_g32_e8m0_v1"
        recipes.append(
            ObjectRecipe(
                name=module,
                format_id=format_id,
                layout_id=(
                    "mx_rowsplit_k128_v1"
                    if len(leading_shape) == 1
                    else "mx_expert_rowsplit_k128_v1"
                ),
                shape=(*leading_shape, logical_k),
                padded_k=padded_k,
                planes=[
                    _plane_from_tensor(
                        source,
                        packed_name,
                        "data",
                        rows,
                        logical_k // 2,
                        padded_k // 2,
                    ),
                    _plane_from_tensor(
                        source,
                        scale_name,
                        "scale",
                        rows,
                        logical_k // 32,
                        padded_k // 32,
                    ),
                ],
                source_tensors=source_names,
                binding={
                    "name": module,
                    "format": format_id,
                    "use": infer_weight_use(module),
                    "activation_policy": "AllowA4",
                    "leading_shape": list(leading_shape[:-1]),
                    "out_features": leading_shape[-1],
                    "in_features": logical_k,
                },
            )
        )
    if not recipes:
        raise ValueError("compressed-tensors source contains no MXFP4 modules")
    return recipes, consumed


def build_direct_recipes(
    source: SafeTensorSource, consumed: set[str]
) -> list[ObjectRecipe]:
    recipes: list[ObjectRecipe] = []
    for name in sorted(set(source.tensors) - consumed):
        info = source.tensors[name]

        def writer(handle: BinaryIO, tensor_name: str = name) -> tuple[int, int, str]:
            view = source.view(tensor_name)
            try:
                return PayloadWriter(handle).write_chunks(bytes_chunks(view))
            finally:
                view.release()

        recipes.append(
            ObjectRecipe(
                name=name,
                format_id=f"direct_{info.dtype.lower()}_v1",
                layout_id="contiguous_le_v1",
                shape=info.shape,
                padded_k=None,
                planes=[PlaneRecipe("data", info.nbytes, writer)],
                source_tensors=[name],
            )
        )
    return recipes


def build_dflash2_recipes(source: SafeTensorSource) -> list[ObjectRecipe]:
    if tuple(source.config.get("architectures") or ()) != ("DFlash2DraftModel",):
        raise ValueError("DFlash2 profile requires DFlash2DraftModel config")
    if not source.tensors or any(info.dtype != "BF16" for info in source.tensors.values()):
        raise ValueError("DFlash2 profile requires a nonempty all-BF16 checkpoint")
    recipes: list[ObjectRecipe] = []
    for source_name in sorted(source.tensors):
        info = source.tensors[source_name]
        object_name = f"dflash2.{source_name}"

        def writer(
            handle: BinaryIO, tensor_name: str = source_name
        ) -> tuple[int, int, str]:
            view = source.view(tensor_name)
            try:
                return PayloadWriter(handle).write_chunks(bytes_chunks(view))
            finally:
                view.release()

        recipes.append(
            ObjectRecipe(
                name=object_name,
                format_id="direct_bf16_v1",
                layout_id="contiguous_le_v1",
                shape=info.shape,
                padded_k=None,
                planes=[PlaneRecipe("data", info.nbytes, writer)],
                source_tensors=[source_name],
                binding=None,
            )
        )
    return recipes


def build_recipes(source: SafeTensorSource, profile: str) -> list[ObjectRecipe]:
    if profile == "auto":
        profile = detect_profile(source)
    if profile == "dflash2_bf16":
        return build_dflash2_recipes(source)
    if profile == "oq_affine":
        quantized, consumed = build_oq_recipes(source)
    elif profile == "mlx_mxfp4":
        quantized, consumed = build_mlx_mxfp4_recipes(source)
    elif profile == "compressed_mxfp4":
        quantized, consumed = build_compressed_mxfp4_recipes(source)
    else:
        raise ValueError(f"unsupported source profile {profile}")
    return sorted(
        quantized + build_direct_recipes(source, consumed), key=lambda item: item.name
    )


def _conversion_report_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".conversion.json")


def _write_conversion_report(output_path: Path, report: dict) -> Path:
    report_path = _conversion_report_path(output_path)
    report_raw = json.dumps(report, indent=2, sort_keys=True) + "\n"
    report_fd, report_name = tempfile.mkstemp(
        prefix=report_path.name + ".tmp.", dir=report_path.parent
    )
    try:
        with os.fdopen(report_fd, "w") as report_handle:
            report_handle.write(report_raw)
            report_handle.flush()
            os.fsync(report_handle.fileno())
        os.replace(report_name, report_path)
    finally:
        if os.path.exists(report_name):
            os.unlink(report_name)
    return report_path


def write_artifact(
    source: SafeTensorSource,
    output_path: Path,
    source_repo: str,
    profile: str = "auto",
    *,
    source_provenance: dict[str, Any] | None = None,
    write_report: bool = True,
) -> dict:
    detected = detect_profile(source) if profile == "auto" else profile
    recipes = build_recipes(source, detected)
    output_path = output_path.resolve()
    protected_paths = {shard.path.resolve() for shard in source.shards.values()} | {
        (source.model_dir / "config.json").resolve(),
        (source.model_dir / "model.safetensors.index.json").resolve(),
        (source.model_dir / "REVISION").resolve(),
    }
    if (
        output_path in protected_paths
        or _conversion_report_path(output_path) in protected_paths
    ):
        raise ValueError("artifact output would overwrite a source checkpoint file")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_fd, payload_name = tempfile.mkstemp(
        prefix=output_path.name + ".payload.", dir=output_path.parent
    )
    os.close(payload_fd)
    object_records: list[dict] = []
    bindings: list[dict] = []
    try:
        with open(payload_name, "w+b") as payload:
            writer = PayloadWriter(payload)
            for object_id, recipe in enumerate(recipes):
                writer.align(OBJECT_ALIGNMENT)
                object_begin = writer.offset
                plane_records = []
                for plane in recipe.planes:
                    writer.align(plane.alignment)
                    plane_offset = writer.offset
                    written, checksum, plane_sha256 = plane.writer(payload)
                    if written != plane.logical_bytes:
                        raise ValueError(
                            f"{recipe.name}/{plane.kind}: wrote {written}, expected {plane.logical_bytes}"
                        )
                    plane_records.append(
                        {
                            "kind": plane.kind,
                            "offset": plane_offset,
                            "bytes": written,
                            "alignment": plane.alignment,
                            "crc32": f"{checksum:08x}",
                            "sha256": plane_sha256,
                        }
                    )
                payload.flush()
                object_bytes = writer.offset - object_begin
                record = {
                    "id": object_id,
                    "name": recipe.name,
                    "format": recipe.format_id,
                    "layout": recipe.layout_id,
                    "shape": list(recipe.shape),
                    "object_offset": object_begin,
                    "object_bytes": object_bytes,
                    "object_sha256": sha256_range(payload, object_begin, object_bytes),
                    "planes": plane_records,
                    "source_tensors": recipe.source_tensors,
                }
                if recipe.padded_k is not None:
                    record["padded_k"] = recipe.padded_k
                object_records.append(record)
                if recipe.binding is not None:
                    bindings.append({**recipe.binding, "object_id": object_id})
            payload.flush()
            os.fsync(payload.fileno())
            payload_bytes = writer.offset

        format_registry: dict[str, dict] = {}
        for format_id in sorted(
            {
                record["format"]
                for record in object_records
                if not record["format"].startswith("direct_")
            }
        ):
            if format_id.startswith("oq_affine_q"):
                fields = format_id.split("_")
                format_registry[format_id] = {
                    "code_bits": int(fields[2].removeprefix("q")),
                    "group_size": int(fields[3].removeprefix("g")),
                    "scale": "BF16",
                    "bias": "BF16",
                }
            elif format_id == "mxfp4_e2m1_g32_e8m0_v1":
                format_registry[format_id] = {
                    "code": "E2M1",
                    "group_size": 32,
                    "scale": "E8M0",
                }
            else:
                raise ValueError(f"unsupported format registry entry {format_id}")

        provenance = source_provenance or {}
        source_revision = str(provenance.get("source_revision", source.revision))
        source_config_sha256 = str(
            provenance.get(
                "source_config_sha256",
                file_sha256_if_present(source.model_dir / "config.json"),
            )
        )
        source_index_sha256 = str(
            provenance.get(
                "source_index_sha256",
                file_sha256_if_present(
                    source.model_dir / "model.safetensors.index.json"
                ),
            )
        )

        manifest = {
            "schema": SCHEMA,
            "container_version": VERSION,
            "model": {
                "model_type": source.config.get("model_type", ""),
                "text_model_type": (source.config.get("text_config") or {}).get(
                    "model_type", ""
                ),
                "source_repo": source_repo,
                "source_revision": source_revision,
                "source_profile": detected,
                "source_config_sha256": source_config_sha256,
                "source_index_sha256": source_index_sha256,
                "source_tensor_count": len(source.tensors),
                "source_tensor_bytes": sum(
                    tensor.nbytes for tensor in source.tensors.values()
                ),
            },
            "converter": {
                "name": "omlx-m5-artifact",
                "version": 1,
                "endianness": "little",
            },
            "format_registry": format_registry,
            "objects": object_records,
            "views": [],
            "bindings": bindings,
            "payload_bytes": payload_bytes,
        }
        manifest_bytes = canonical_json(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_crc = crc32(manifest_bytes)
        payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), PAYLOAD_ALIGNMENT)
        flags = 0
        header_prefix = HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            HEADER_BYTES,
            HEADER_BYTES,
            len(manifest_bytes),
            payload_offset,
            payload_bytes,
            len(object_records),
            manifest_crc,
            0,
            flags,
        )
        header_crc = crc32(header_prefix[:56])
        header = HEADER_STRUCT.pack(
            MAGIC,
            VERSION,
            HEADER_BYTES,
            HEADER_BYTES,
            len(manifest_bytes),
            payload_offset,
            payload_bytes,
            len(object_records),
            manifest_crc,
            header_crc,
            flags,
        )
        final_fd, final_name = tempfile.mkstemp(
            prefix=output_path.name + ".tmp.", dir=output_path.parent
        )
        os.close(final_fd)
        try:
            with open(final_name, "w+b") as output, open(payload_name, "rb") as payload:
                output.write(header)
                output.write(manifest_bytes)
                output.write(b"\0" * (payload_offset - output.tell()))
                while chunk := payload.read(16 * 1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            os.replace(final_name, output_path)
        finally:
            if os.path.exists(final_name):
                os.unlink(final_name)
        artifact_sha256 = sha256_file(output_path)
        counts = Counter(record["format"] for record in object_records)
        report = {
            "schema": "omlx-m5-conversion-report-v1",
            "artifact": str(output_path),
            "artifact_bytes": output_path.stat().st_size,
            "artifact_sha256": artifact_sha256,
            "manifest_crc32": f"{manifest_crc:08x}",
            "manifest_sha256": manifest_sha256,
            "payload_bytes": payload_bytes,
            "object_count": len(object_records),
            "binding_count": len(bindings),
            "formats": dict(sorted(counts.items())),
            "source_repo": source_repo,
            "source_revision": source_revision,
            "source_profile": detected,
            "source_config_sha256": source_config_sha256,
            "source_index_sha256": source_index_sha256,
            "objects": [
                {
                    "id": record["id"],
                    "name": record["name"],
                    "format": record["format"],
                    "bytes": record["object_bytes"],
                    "sha256": record["object_sha256"],
                }
                for record in object_records
            ],
        }
        if write_report:
            _write_conversion_report(output_path, report)
        return report
    finally:
        if os.path.exists(payload_name):
            os.unlink(payload_name)


def _read_exact(handle: BinaryIO, size: int, label: str) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"{label} is truncated")
    return data


def _is_lower_hex(value: object, digits: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == digits
        and all(character in "0123456789abcdef" for character in value)
    )


def _verify_zero_range(handle: BinaryIO, offset: int, size: int, label: str) -> None:
    handle.seek(offset)
    remaining = size
    while remaining:
        chunk = _read_exact(handle, min(remaining, 1024 * 1024), label)
        if any(chunk):
            raise ValueError(f"{label} is not zero-filled")
        remaining -= len(chunk)


def _hash_range(handle: BinaryIO, offset: int, size: int) -> tuple[int, str]:
    handle.seek(offset)
    remaining = size
    checksum = 0
    digest = hashlib.sha256()
    while remaining:
        chunk = _read_exact(
            handle, min(remaining, 16 * 1024 * 1024), "artifact payload"
        )
        checksum = crc32(chunk, checksum)
        digest.update(chunk)
        remaining -= len(chunk)
    return checksum, digest.hexdigest()


def _parse_affine_format(format_id: str) -> tuple[int, int] | None:
    fields = format_id.split("_")
    if (
        len(fields) == 6
        and fields[0:2] == ["oq", "affine"]
        and fields[2].startswith("q")
        and fields[3].startswith("g")
        and fields[4:] == ["bf16", "v1"]
    ):
        bits = int(fields[2][1:])
        group_size = int(fields[3][1:])
        if bits in (3, 4, 5, 6, 8) and group_size in (64, 128):
            return bits, group_size
    return None


def _expected_plane_bytes(record: dict) -> dict[str, int] | None:
    if record.get("format", "").startswith("direct_"):
        return None
    if len(record.get("shape", [])) < 2:
        raise ValueError(f"invalid quantized rank: {record.get('name', '<unknown>')}")
    rows = product(int(value) for value in record["shape"][:-1])
    logical_k = int(record["shape"][-1])
    padded_k = int(record.get("padded_k", 0))
    if rows <= 0 or logical_k <= 0 or padded_k < logical_k or padded_k % 128:
        raise ValueError(
            f"invalid quantized geometry: {record.get('name', '<unknown>')}"
        )
    affine = _parse_affine_format(record["format"])
    if affine is not None:
        bits, group_size = affine
        if record.get("layout") == "oq_nax_tiled_g64_v1":
            if (
                len(record["shape"]) != 2
                or bits not in (3, 4, 5, 6)
                or group_size not in (64, 128)
                or logical_k < 512
                or (logical_k % group_size) != 0
            ):
                raise ValueError("artifact affine NAX tiled geometry mismatch")
            bn = 32 if rows >= 65536 else 16
            padded_n = align_up(rows, bn)
            if bits == 3:
                planes = {
                    "nax_codes": padded_n * logical_k // 4,
                    "nax_high1": padded_n * logical_k // 8,
                }
            else:
                planes = {"nax_codes": padded_n * logical_k // 2}
                if bits in (5, 6):
                    planes["nax_high2"] = padded_n * logical_k // 4
            planes["nax_sb"] = padded_n * (logical_k // 64) * 4
            return planes
        planes = {}
        if bits == 3:
            planes["low_2bit"] = rows * padded_k // 4
            planes["high_bit"] = rows * padded_k // 8
        elif bits == 4:
            planes["base_code"] = rows * padded_k // 2
        elif bits == 5:
            planes["base_code"] = rows * padded_k // 2
            planes["high_bit"] = rows * padded_k // 8
        elif bits == 6:
            planes["base_code"] = rows * padded_k // 2
            planes["high_2bit"] = rows * padded_k // 4
        else:
            planes["code"] = rows * padded_k
        planes["scale"] = rows * padded_k // group_size * 2
        planes["bias"] = rows * padded_k // group_size * 2
        return planes
    if record["format"] == "mxfp4_e2m1_g32_e8m0_v1":
        if record.get("layout") == "mx_nax_tiled_bk128_v1":
            if (
                len(record["shape"]) != 2
                or logical_k < 512
                or (logical_k % 128) != 0
            ):
                raise ValueError("artifact MXFP4 NAX tiled geometry mismatch")
            bn = 32 if rows >= 65536 else 16
            padded_n = align_up(rows, bn)
            return {
                "nax_data": padded_n * logical_k // 2,
                "nax_scale": padded_n * logical_k // 32,
            }
        return {
            "data": rows * padded_k // 2,
            "scale": rows * padded_k // 32,
        }
    return None


def _validate_object_contract(record: dict, expected_id: int) -> None:
    if record.get("id") != expected_id or not record.get("name"):
        raise ValueError("artifact object identity is invalid")
    if record["object_offset"] % OBJECT_ALIGNMENT:
        raise ValueError("artifact object alignment is invalid")
    if not record.get("source_tensors") or not record.get("planes"):
        raise ValueError("artifact object directory is incomplete")
    if not _is_lower_hex(record.get("object_sha256"), 64):
        raise ValueError("artifact object checksum metadata is invalid")
    expected = _expected_plane_bytes(record)
    planes = record["planes"]
    kinds = [plane.get("kind") for plane in planes]
    if len(kinds) != len(set(kinds)):
        raise ValueError("artifact object has duplicate plane kinds")
    if expected is not None:
        is_expert = len(record["shape"]) > 2
        allowed_layouts = (
            ("mx_expert_rowsplit_k128_v1",)
            if (is_expert and record["format"] == "mxfp4_e2m1_g32_e8m0_v1")
            else ("mx_rowsplit_k128_v1", "mx_nax_tiled_bk128_v1")
            if record["format"] == "mxfp4_e2m1_g32_e8m0_v1"
            else ("oq_expert_rowsplit_k128_v1",)
            if is_expert
            else ("oq_rowsplit_k128_v1", "oq_nax_tiled_g64_v1")
        )
        if record.get("layout") not in allowed_layouts or kinds != list(expected):
            raise ValueError("artifact quantized layout contract mismatch")
        if any(plane["bytes"] != expected[plane["kind"]] for plane in planes):
            raise ValueError("artifact quantized plane size mismatch")
    elif record["format"].startswith("direct_"):
        if record.get("layout") != "contiguous_le_v1" or kinds != ["data"]:
            raise ValueError("artifact direct layout contract mismatch")
        dtype = record["format"][len("direct_") : -len("_v1")].upper()
        if dtype not in DTYPE_BYTES:
            raise ValueError("artifact direct dtype is unsupported")
        elements = 1
        for dimension in record["shape"]:
            if dimension < 0:
                raise ValueError("artifact object has a negative dimension")
            elements *= dimension
        if planes[0]["bytes"] != elements * DTYPE_BYTES[dtype]:
            raise ValueError("artifact direct plane size mismatch")
    else:
        raise ValueError(f"unsupported artifact format: {record['format']}")


def read_artifact(path: Path, verify_payload: bool = False) -> tuple[dict, dict]:
    path = path.resolve()
    with path.open("rb") as handle:
        header = _read_exact(handle, HEADER_BYTES, "artifact header")
        (
            magic,
            version,
            header_bytes,
            manifest_offset,
            manifest_bytes,
            payload_offset,
            payload_bytes,
            object_count,
            manifest_crc,
            header_crc,
            flags,
        ) = HEADER_STRUCT.unpack(header)
        if (
            magic != MAGIC
            or version != VERSION
            or header_bytes != HEADER_BYTES
            or flags != 0
        ):
            raise ValueError("unsupported artifact header")
        if crc32(header[:56]) != header_crc:
            raise ValueError("artifact header checksum mismatch")
        size = path.stat().st_size
        if (
            manifest_offset != HEADER_BYTES
            or manifest_bytes <= 0
            or manifest_offset + manifest_bytes > payload_offset
            or payload_offset % LEGACY_PAYLOAD_ALIGNMENT
            or payload_offset + payload_bytes != size
        ):
            raise ValueError("artifact ranges are invalid")
        handle.seek(manifest_offset)
        manifest_raw = _read_exact(handle, manifest_bytes, "artifact manifest")
        if crc32(manifest_raw) != manifest_crc:
            raise ValueError("artifact manifest checksum mismatch")
        try:
            manifest = json.loads(manifest_raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"artifact manifest JSON is invalid: {exc}") from exc
        if canonical_json(manifest) != manifest_raw:
            raise ValueError("artifact manifest is not canonical JSON")
        if (
            manifest.get("schema") != SCHEMA
            or manifest.get("container_version") != VERSION
            or manifest.get("payload_bytes") != payload_bytes
            or not isinstance(manifest.get("model"), dict)
            or not isinstance(manifest.get("objects"), list)
            or not isinstance(manifest.get("views"), list)
            or not isinstance(manifest.get("bindings"), list)
            or len(manifest["objects"]) != object_count
        ):
            raise ValueError("artifact manifest contract mismatch")
        if manifest["views"]:
            raise ValueError("artifact v1 converter emitted unsupported explicit views")
        _verify_zero_range(
            handle,
            manifest_offset + manifest_bytes,
            payload_offset - manifest_offset - manifest_bytes,
            "artifact manifest padding",
        )
        previous_end = 0
        object_names: set[str] = set()
        for expected_id, record in enumerate(manifest["objects"]):
            _validate_object_contract(record, expected_id)
            if record["name"] in object_names:
                raise ValueError("artifact object name is duplicated")
            object_names.add(record["name"])
            if record["object_offset"] < previous_end:
                raise ValueError("artifact objects are unordered or overlapping")
            _verify_zero_range(
                handle,
                payload_offset + previous_end,
                record["object_offset"] - previous_end,
                "artifact object padding",
            )
            object_end = record["object_offset"] + record["object_bytes"]
            if object_end > payload_bytes or record["object_bytes"] <= 0:
                raise ValueError("artifact object exceeds payload")
            plane_end = record["object_offset"]
            single_exact = (
                len(record["planes"]) == 1
                and record["planes"][0].get("offset") == record["object_offset"]
                and record["planes"][0].get("bytes") == record["object_bytes"]
            )
            object_digest_ctx = (
                hashlib.sha256() if verify_payload and not single_exact else None
            )
            first_plane_digest = ""
            for plane in record["planes"]:
                if (
                    plane.get("alignment") != PLANE_ALIGNMENT
                    or plane["offset"] < plane_end
                    or plane["offset"] % plane["alignment"]
                    or plane["offset"] + plane["bytes"] > object_end
                    or plane["bytes"] <= 0
                ):
                    raise ValueError("artifact plane range is invalid")
                if not _is_lower_hex(plane.get("crc32"), 8) or not _is_lower_hex(
                    plane.get("sha256"), 64
                ):
                    raise ValueError("artifact plane checksum metadata is invalid")
                pad_bytes = plane["offset"] - plane_end
                _verify_zero_range(
                    handle,
                    payload_offset + plane_end,
                    pad_bytes,
                    "artifact plane padding",
                )
                if object_digest_ctx is not None and pad_bytes > 0:
                    object_digest_ctx.update(b"\0" * pad_bytes)
                if verify_payload:
                    handle.seek(payload_offset + plane["offset"])
                    remaining = plane["bytes"]
                    checksum = 0
                    plane_digest_ctx = hashlib.sha256()
                    while remaining:
                        chunk = _read_exact(
                            handle,
                            min(remaining, 16 * 1024 * 1024),
                            "artifact payload",
                        )
                        checksum = crc32(chunk, checksum)
                        plane_digest_ctx.update(chunk)
                        if object_digest_ctx is not None:
                            object_digest_ctx.update(chunk)
                        remaining -= len(chunk)
                    digest = plane_digest_ctx.hexdigest()
                    if f"{checksum:08x}" != plane["crc32"] or digest != plane["sha256"]:
                        raise ValueError(
                            "artifact plane checksum mismatch: "
                            f"{record['name']}/{plane['kind']}"
                        )
                    if single_exact:
                        first_plane_digest = digest
                plane_end = plane["offset"] + plane["bytes"]
            if plane_end != object_end:
                raise ValueError("artifact object has trailing bytes")
            if verify_payload:
                object_digest = (
                    first_plane_digest
                    if single_exact
                    else object_digest_ctx.hexdigest()  # type: ignore[union-attr]
                )
                if object_digest != record.get("object_sha256"):
                    raise ValueError(
                        f"artifact object checksum mismatch: {record['name']}"
                    )
            previous_end = object_end
        if previous_end != payload_bytes:
            raise ValueError("artifact payload has trailing bytes")
        bound_ids: set[int] = set()
        for binding in manifest["bindings"]:
            object_id = binding.get("object_id")
            if not isinstance(object_id, int) or not 0 <= object_id < object_count:
                raise ValueError("artifact binding object id is invalid")
            record = manifest["objects"][object_id]
            if (
                object_id in bound_ids
                or binding.get("name") != record["name"]
                or binding.get("format") != record["format"]
                or binding.get("leading_shape", []) != record["shape"][:-2]
                or binding.get("out_features") != record["shape"][-2]
                or binding.get("in_features") != record["shape"][-1]
                or not binding.get("use")
                or binding.get("activation_policy")
                != (
                    "AllowA4"
                    if record["format"] == "mxfp4_e2m1_g32_e8m0_v1"
                    else "A16Only"
                )
            ):
                raise ValueError("artifact binding is invalid")
            bound_ids.add(object_id)
        quantized_ids = {
            record["id"]
            for record in manifest["objects"]
            if not record["format"].startswith("direct_")
        }
        if bound_ids != quantized_ids:
            raise ValueError("artifact quantized binding coverage is incomplete")
        summary = {
            "schema": manifest["schema"],
            "container_version": version,
            "artifact": str(path),
            "artifact_bytes": size,
            "manifest_bytes": manifest_bytes,
            "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
            "payload_offset": payload_offset,
            "payload_bytes": payload_bytes,
            "object_count": object_count,
            "binding_count": len(manifest["bindings"]),
            "formats": dict(
                sorted(Counter(item["format"] for item in manifest["objects"]).items())
            ),
            "model": manifest["model"],
            "payload_verified": verify_payload,
        }
        return summary, manifest


def _bf16_to_float(raw: bytes) -> float:
    word = int.from_bytes(raw, "little")
    return struct.unpack("<f", struct.pack("<I", word << 16))[0]


def _e2m1_to_float(code: int) -> float:
    magnitude = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[code & 7]
    return -magnitude if code & 8 else magnitude


def _e8m0_to_float(code: int) -> float:
    if code == 0xFF:
        raise ValueError("MXFP4 E8M0 NaN scale is not accepted")
    return 2.0 ** (-127 if code == 0 else code - 127)


def decode_artifact_row(
    path: Path,
    manifest: dict,
    object_name: str,
    row: int,
    start: int,
    count: int,
) -> list[float]:
    records = [item for item in manifest["objects"] if item["name"] == object_name]
    if len(records) != 1:
        raise ValueError(f"artifact object is not unique: {object_name}")
    record = records[0]
    row_count = product(int(value) for value in record["shape"][:-1])
    logical_k = int(record["shape"][-1])
    if (
        len(record["shape"]) < 2
        or row < 0
        or row >= row_count
        or start < 0
        or count <= 0
        or start + count > logical_k
    ):
        raise ValueError("artifact decode range is invalid")
    planes = {plane["kind"]: plane for plane in record["planes"]}
    padded_k = record["padded_k"]
    with path.open("rb") as handle:
        header = _read_exact(handle, HEADER_BYTES, "artifact header")
        payload_offset = HEADER_STRUCT.unpack(header)[5]

        def read_plane(kind: str, offset: int, size: int) -> bytes:
            plane = planes[kind]
            if offset < 0 or offset + size > plane["bytes"]:
                raise ValueError("artifact decode plane range is invalid")
            handle.seek(payload_offset + plane["offset"] + offset)
            return _read_exact(handle, size, f"artifact {kind} plane")

        values: list[float] = []
        affine = _parse_affine_format(record["format"])
        if affine is not None:
            bits, group_size = affine
            if record.get("layout") == "oq_nax_tiled_g64_v1":
                if (
                    len(record["shape"]) != 2
                    or bits not in (3, 4, 5, 6)
                    or group_size not in (64, 128)
                    or logical_k < 512
                    or (logical_k % group_size) != 0
                ):
                    raise ValueError("artifact affine NAX tiled geometry mismatch")
                bn = 32 if row_count >= 65536 else 16
                padded_n = align_up(row_count, bn)
                groups = logical_k // 64
                rn = row % bn
                tile_row = row // bn
                for column in range(start, start + count):
                    g = column // 64
                    col_in_g = column % 64
                    tile_idx = tile_row * groups + g
                    if bits == 3:
                        c_byte = read_plane(
                            "nax_codes",
                            tile_idx * (bn * 16) + rn * 16 + col_in_g // 4,
                            1,
                        )[0]
                        h_byte = read_plane(
                            "nax_high1",
                            tile_idx * (bn * 8) + rn * 8 + col_in_g // 8,
                            1,
                        )[0]
                        code = ((c_byte >> ((col_in_g & 3) * 2)) & 0x3) | (
                            ((h_byte >> (col_in_g & 7)) & 0x1) << 2
                        )
                    else:
                        c_byte = read_plane(
                            "nax_codes",
                            tile_idx * (bn * 32) + rn * 32 + col_in_g // 2,
                            1,
                        )[0]
                        code = (c_byte >> ((col_in_g & 1) * 4)) & 0xF
                        if bits in (5, 6):
                            h_byte = read_plane(
                                "nax_high2",
                                tile_idx * (bn * 16) + rn * 16 + col_in_g // 4,
                                1,
                            )[0]
                            h2 = (h_byte >> ((col_in_g & 3) * 2)) & 0x3
                            code |= (h2 & 1 if bits == 5 else h2) << 4
                    sb_raw = read_plane("nax_sb", (g * padded_n + row) * 4, 4)
                    scale = _bf16_to_float(sb_raw[:2])
                    bias = _bf16_to_float(sb_raw[2:4])
                    values.append(scale * code + bias)
                return values
            code_row = (
                padded_k
                if bits == 8
                else padded_k // 4
                if bits == 3
                else padded_k // 2
            )
            code_kind = (
                "code"
                if bits == 8
                else "low_2bit"
                if bits == 3
                else "base_code"
            )
            high_row = padded_k // (8 if bits in (3, 5) else 4)
            group_row = padded_k // group_size * 2
            base = read_plane(code_kind, row * code_row, code_row)
            high_kind = "high_bit" if bits in (3, 5) else "high_2bit"
            high = (
                read_plane(high_kind, row * high_row, high_row)
                if bits in (3, 5, 6)
                else b""
            )
            scales = read_plane("scale", row * group_row, group_row)
            biases = read_plane("bias", row * group_row, group_row)
            for column in range(start, start + count):
                if bits == 8:
                    code = base[column]
                elif bits == 3:
                    code = (
                        ((base[column // 4] >> ((column & 3) * 2)) & 0x3)
                        | (((high[column // 8] >> (column & 7)) & 0x1) << 2)
                    )
                else:
                    code = (base[column // 2] >> ((column & 1) * 4)) & 0xF
                if bits == 5:
                    code |= ((high[column // 8] >> (column & 7)) & 1) << 4
                elif bits == 6:
                    code |= ((high[column // 4] >> ((column & 3) * 2)) & 3) << 4
                group = column // group_size
                scale = _bf16_to_float(scales[group * 2 : group * 2 + 2])
                bias = _bf16_to_float(biases[group * 2 : group * 2 + 2])
                values.append(scale * code + bias)
            return values
        if record["format"] == "mxfp4_e2m1_g32_e8m0_v1":
            if record.get("layout") == "mx_nax_tiled_bk128_v1":
                if (
                    len(record["shape"]) != 2
                    or logical_k < 512
                    or (logical_k % 128) != 0
                ):
                    raise ValueError("artifact MXFP4 NAX tiled geometry mismatch")
                bn = 32 if row_count >= 65536 else 16
                k_blocks = logical_k // 128
                rn = row % bn
                tile_row = row // bn
                for column in range(start, start + count):
                    blk = column // 128
                    col_in_blk = column % 128
                    tile_idx = tile_row * k_blocks + blk
                    packed = read_plane(
                        "nax_data",
                        tile_idx * (bn * 64) + rn * 64 + col_in_blk // 2,
                        1,
                    )[0]
                    code = (packed >> ((col_in_blk & 1) * 4)) & 0xF
                    sc = read_plane(
                        "nax_scale",
                        tile_idx * (bn * 4) + rn * 4 + col_in_blk // 32,
                        1,
                    )[0]
                    values.append(_e2m1_to_float(code) * _e8m0_to_float(sc))
                return values
            data_row = padded_k // 2
            scale_row = padded_k // 32
            data = read_plane("data", row * data_row, data_row)
            scales = read_plane("scale", row * scale_row, scale_row)
            for column in range(start, start + count):
                code = (data[column // 2] >> ((column & 1) * 4)) & 0xF
                values.append(
                    _e2m1_to_float(code) * _e8m0_to_float(scales[column // 32])
                )
            return values
    raise ValueError(f"artifact object is not decodable: {record['format']}")


def pretile_artifact(input_path: Path, output_path: Path) -> dict:
    import numpy as np

    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("pretile output must differ from input artifact path")
    _, src_manifest = read_artifact(input_path, verify_payload=False)
    embedding_obj_ids = {
        int(b["object_id"])
        for b in src_manifest.get("bindings", [])
        if b.get("role") == "embedding" or b.get("use") == "embedding"
    }
    lut_4x1_to_4x2 = np.array(
        [
            (nib & 1) | ((nib & 2) << 1) | ((nib & 4) << 2) | ((nib & 8) << 3)
            for nib in range(16)
        ],
        dtype=np.uint8,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_fd, payload_name = tempfile.mkstemp(
        prefix=output_path.name + ".payload.", dir=output_path.parent
    )
    os.close(payload_fd)
    new_objects: list[dict] = []
    pretiled_count = 0

    try:
        with input_path.open("rb") as src_f, open(payload_name, "w+b") as dst_payload:
            src_header = _read_exact(src_f, HEADER_BYTES, "artifact header")
            src_payload_offset = HEADER_STRUCT.unpack(src_header)[5]
            writer = PayloadWriter(dst_payload)

            def read_src_plane(plane_rec: dict) -> bytes:
                src_f.seek(src_payload_offset + int(plane_rec["offset"]))
                return _read_exact(src_f, int(plane_rec["bytes"]), plane_rec["kind"])

            def emit_raw_plane(kind: str, data: bytes | memoryview, alignment: int = PLANE_ALIGNMENT) -> dict:
                writer.align(alignment)
                p_off = writer.offset
                written, csum, p_sha = writer.write_chunks([memoryview(data)])
                return {
                    "kind": kind,
                    "offset": p_off,
                    "bytes": written,
                    "alignment": alignment,
                    "crc32": f"{csum:08x}",
                    "sha256": p_sha,
                }

            for obj in src_manifest["objects"]:
                obj_id = int(obj["id"])
                name = obj["name"]
                fmt = obj["format"]
                layout = obj["layout"]
                shape = list(obj["shape"])
                is_embed = (obj_id in embedding_obj_ids) or (infer_weight_use(name) == "embedding")
                planes_by_kind = {p["kind"]: p for p in obj["planes"]}

                writer.align(OBJECT_ALIGNMENT)
                object_begin = writer.offset
                new_planes: list[dict] = []
                new_layout = layout

                if (
                    not is_embed
                    and fmt == "mxfp4_e2m1_g32_e8m0_v1"
                    and layout == "mx_rowsplit_k128_v1"
                    and len(shape) == 2
                    and shape[0] > 0
                    and shape[1] >= 512
                    and (shape[1] % 128) == 0
                ):
                    n, k = int(shape[0]), int(shape[1])
                    padded_k = int(obj["padded_k"])
                    bn = 32 if n >= 65536 else 16
                    padded_n = align_up(n, bn)
                    k_blocks = k // 128

                    raw_c = np.frombuffer(read_src_plane(planes_by_kind["data"]), dtype=np.uint8).reshape(n, padded_k // 2)[:, : k // 2]
                    raw_s = np.frombuffer(read_src_plane(planes_by_kind["scale"]), dtype=np.uint8).reshape(n, padded_k // 32)[:, : k // 32]
                    if padded_n > n:
                        pad_c = np.zeros((padded_n, k // 2), dtype=np.uint8)
                        pad_c[:n] = raw_c
                        raw_c = pad_c
                        pad_s = np.full((padded_n, k // 32), 0x7F, dtype=np.uint8)
                        pad_s[:n] = raw_s
                        raw_s = pad_s
                    tiled_c = np.ascontiguousarray(
                        raw_c.reshape(padded_n // bn, bn, k_blocks, 64).transpose(0, 2, 1, 3)
                    )
                    tiled_s = np.ascontiguousarray(
                        raw_s.reshape(padded_n // bn, bn, k_blocks, 4).transpose(0, 2, 1, 3)
                    )
                    new_planes.append(emit_raw_plane("nax_data", tiled_c.tobytes()))
                    new_planes.append(emit_raw_plane("nax_scale", tiled_s.tobytes()))
                    new_layout = "mx_nax_tiled_bk128_v1"
                    pretiled_count += 1
                elif (
                    not is_embed
                    and _parse_affine_format(fmt)
                    in (
                        (3, 64),
                        (3, 128),
                        (4, 64),
                        (4, 128),
                        (5, 64),
                        (5, 128),
                        (6, 64),
                        (6, 128),
                    )
                    and layout == "oq_rowsplit_k128_v1"
                    and len(shape) == 2
                    and shape[0] > 0
                    and shape[1] >= 512
                    and (shape[1] % _parse_affine_format(fmt)[1]) == 0
                ):
                    bits, src_gs = _parse_affine_format(fmt)
                    n, k = int(shape[0]), int(shape[1])
                    padded_k = int(obj["padded_k"])
                    bn = 32 if n >= 65536 else 16
                    padded_n = align_up(n, bn)
                    groups = k // 64

                    if bits == 3:
                        raw_l2 = np.frombuffer(
                            read_src_plane(planes_by_kind["low_2bit"]), dtype=np.uint8
                        ).reshape(n, padded_k // 4)[:, : k // 4]
                        if padded_n > n:
                            pad_l2 = np.zeros((padded_n, k // 4), dtype=np.uint8)
                            pad_l2[:n] = raw_l2
                            raw_l2 = pad_l2
                        tiled_l2 = np.ascontiguousarray(
                            raw_l2.reshape(padded_n // bn, bn, groups, 16).transpose(
                                0, 2, 1, 3
                            )
                        )
                        new_planes.append(emit_raw_plane("nax_codes", tiled_l2.tobytes()))

                        raw_h1 = np.frombuffer(
                            read_src_plane(planes_by_kind["high_bit"]), dtype=np.uint8
                        ).reshape(n, padded_k // 8)[:, : k // 8]
                        if padded_n > n:
                            pad_h1 = np.zeros((padded_n, k // 8), dtype=np.uint8)
                            pad_h1[:n] = raw_h1
                            raw_h1 = pad_h1
                        tiled_h1 = np.ascontiguousarray(
                            raw_h1.reshape(padded_n // bn, bn, groups, 8).transpose(
                                0, 2, 1, 3
                            )
                        )
                        new_planes.append(emit_raw_plane("nax_high1", tiled_h1.tobytes()))
                    else:
                        raw_c = np.frombuffer(
                            read_src_plane(planes_by_kind["base_code"]), dtype=np.uint8
                        ).reshape(n, padded_k // 2)[:, : k // 2]
                        if padded_n > n:
                            pad_c = np.zeros((padded_n, k // 2), dtype=np.uint8)
                            pad_c[:n] = raw_c
                            raw_c = pad_c
                        tiled_c = np.ascontiguousarray(
                            raw_c.reshape(padded_n // bn, bn, groups, 32).transpose(
                                0, 2, 1, 3
                            )
                        )
                        new_planes.append(emit_raw_plane("nax_codes", tiled_c.tobytes()))

                        if bits == 5:
                            raw_h = np.frombuffer(
                                read_src_plane(planes_by_kind["high_bit"]), dtype=np.uint8
                            ).reshape(n, padded_k // 8)[:, : k // 8]
                            lo = lut_4x1_to_4x2[raw_h & 0x0F]
                            hi = lut_4x1_to_4x2[raw_h >> 4]
                            exp_h = np.empty((n, k // 4), dtype=np.uint8)
                            exp_h[:, 0::2] = lo
                            exp_h[:, 1::2] = hi
                            if padded_n > n:
                                pad_h = np.zeros((padded_n, k // 4), dtype=np.uint8)
                                pad_h[:n] = exp_h
                                exp_h = pad_h
                            tiled_h = np.ascontiguousarray(
                                exp_h.reshape(padded_n // bn, bn, groups, 16).transpose(
                                    0, 2, 1, 3
                                )
                            )
                            new_planes.append(
                                emit_raw_plane("nax_high2", tiled_h.tobytes())
                            )
                        elif bits == 6:
                            raw_h = np.frombuffer(
                                read_src_plane(planes_by_kind["high_2bit"]),
                                dtype=np.uint8,
                            ).reshape(n, padded_k // 4)[:, : k // 4]
                            if padded_n > n:
                                pad_h = np.zeros((padded_n, k // 4), dtype=np.uint8)
                                pad_h[:n] = raw_h
                                raw_h = pad_h
                            tiled_h = np.ascontiguousarray(
                                raw_h.reshape(padded_n // bn, bn, groups, 16).transpose(
                                    0, 2, 1, 3
                                )
                            )
                            new_planes.append(
                                emit_raw_plane("nax_high2", tiled_h.tobytes())
                            )

                    src_groups = k // src_gs
                    raw_sc = np.frombuffer(
                        read_src_plane(planes_by_kind["scale"]), dtype=np.uint16
                    ).reshape(n, padded_k // src_gs)[:, :src_groups]
                    raw_bi = np.frombuffer(
                        read_src_plane(planes_by_kind["bias"]), dtype=np.uint16
                    ).reshape(n, padded_k // src_gs)[:, :src_groups]
                    if src_gs == 128:
                        raw_sc = np.repeat(raw_sc, 2, axis=1)
                        raw_bi = np.repeat(raw_bi, 2, axis=1)
                    packed_sb = raw_sc.astype(np.uint32) | (
                        raw_bi.astype(np.uint32) << 16
                    )
                    if padded_n > n:
                        pad_sb = np.zeros((padded_n, groups), dtype=np.uint32)
                        pad_sb[:n] = packed_sb
                        packed_sb = pad_sb
                    tiled_sb = np.ascontiguousarray(packed_sb.T)
                    new_planes.append(emit_raw_plane("nax_sb", tiled_sb.tobytes()))
                    new_layout = "oq_nax_tiled_g64_v1"
                    pretiled_count += 1
                else:
                    for p in obj["planes"]:
                        new_planes.append(
                            emit_raw_plane(p["kind"], read_src_plane(p), int(p["alignment"]))
                        )

                dst_payload.flush()
                object_bytes = writer.offset - object_begin
                rec = {
                    **obj,
                    "layout": new_layout,
                    "object_offset": object_begin,
                    "object_bytes": object_bytes,
                    "object_sha256": sha256_range(dst_payload, object_begin, object_bytes),
                    "planes": new_planes,
                }
                new_objects.append(rec)

            dst_payload.flush()
            os.fsync(dst_payload.fileno())
            payload_bytes = writer.offset

        manifest = {
            **src_manifest,
            "objects": new_objects,
            "payload_bytes": payload_bytes,
        }
        manifest_bytes = canonical_json(manifest)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        manifest_crc = crc32(manifest_bytes)
        payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), PAYLOAD_ALIGNMENT)
        flags = 0
        header_prefix = HEADER_STRUCT.pack(
            MAGIC, VERSION, HEADER_BYTES, HEADER_BYTES, len(manifest_bytes),
            payload_offset, payload_bytes, len(new_objects), manifest_crc, 0, flags,
        )
        header_crc = crc32(header_prefix[:56])
        header = HEADER_STRUCT.pack(
            MAGIC, VERSION, HEADER_BYTES, HEADER_BYTES, len(manifest_bytes),
            payload_offset, payload_bytes, len(new_objects), manifest_crc, header_crc, flags,
        )
        final_fd, final_name = tempfile.mkstemp(
            prefix=output_path.name + ".tmp.", dir=output_path.parent
        )
        os.close(final_fd)
        file_digest = hashlib.sha256()
        try:
            with open(final_name, "w+b") as out_f, open(payload_name, "rb") as pay_f:
                out_f.write(header)
                file_digest.update(header)
                out_f.write(manifest_bytes)
                file_digest.update(manifest_bytes)
                pad_bytes = b"\0" * (payload_offset - out_f.tell())
                if pad_bytes:
                    out_f.write(pad_bytes)
                    file_digest.update(pad_bytes)
                while chunk := pay_f.read(16 * 1024 * 1024):
                    out_f.write(chunk)
                    file_digest.update(chunk)
                out_f.flush()
                os.fsync(out_f.fileno())
            os.replace(final_name, output_path)
        finally:
            if os.path.exists(final_name):
                os.unlink(final_name)
    finally:
        if os.path.exists(payload_name):
            os.unlink(payload_name)

    return {
        "artifact": str(output_path),
        "artifact_bytes": output_path.stat().st_size,
        "artifact_sha256": file_digest.hexdigest(),
        "manifest_crc32": f"{manifest_crc:08x}",
        "manifest_sha256": manifest_sha256,
        "payload_bytes": payload_bytes,
        "object_count": len(new_objects),
        "pretiled_objects": pretiled_count,
    }


def _finalize_pretiled_conversion_report(
    report: dict, output_path: Path, pt: dict
) -> dict:
    output_path = output_path.resolve()
    _, manifest = read_artifact(output_path, verify_payload=False)
    updated = {
        **report,
        "artifact": str(output_path),
        "artifact_bytes": pt["artifact_bytes"],
        "artifact_sha256": pt["artifact_sha256"],
        "manifest_crc32": pt["manifest_crc32"],
        "manifest_sha256": pt["manifest_sha256"],
        "payload_bytes": pt["payload_bytes"],
        "object_count": pt["object_count"],
        "pretiled_objects": pt["pretiled_objects"],
        "objects": [
            {
                "id": record["id"],
                "name": record["name"],
                "format": record["format"],
                "bytes": record["object_bytes"],
                "sha256": record["object_sha256"],
            }
            for record in manifest["objects"]
        ],
    }
    _write_conversion_report(output_path, updated)
    return updated


def command_convert(args: argparse.Namespace) -> None:
    source = SafeTensorSource(args.source, args.expected_revision)
    try:
        if getattr(args, "nax_pretile", False):
            output_path = args.output.resolve()
            tmp_out = output_path.with_suffix(output_path.suffix + ".untiled.tmp")
            tmp_report = _conversion_report_path(tmp_out)
            try:
                report = write_artifact(
                    source,
                    tmp_out,
                    args.source_repo,
                    args.profile,
                    write_report=False,
                )
                pt = pretile_artifact(tmp_out, output_path)
                report = _finalize_pretiled_conversion_report(report, output_path, pt)
            finally:
                if tmp_out.exists():
                    tmp_out.unlink()
                if tmp_report.exists():
                    tmp_report.unlink()
        else:
            report = write_artifact(source, args.output, args.source_repo, args.profile)
        print(
            json.dumps(
                {key: value for key, value in report.items() if key != "objects"},
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        source.close()


def command_pretile(args: argparse.Namespace) -> None:
    summary = pretile_artifact(args.input, args.output)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _float32_to_bf16_u16(arr: "np.ndarray") -> "np.ndarray":
    bits = arr.astype(np.float32).view(np.uint32)
    lsb = (bits >> 16) & 1
    rounded = bits + np.uint32(0x7FFF) + lsb
    return (rounded >> 16).astype(np.uint16)


def _bf16_u16_to_float32(arr: "np.ndarray") -> "np.ndarray":
    return (arr.astype(np.uint32) << 16).view(np.float32)


def quantize_affine_oqe(
    weight_f32: "np.ndarray",
    bits: int,
    group_size: int,
    hessian_diag: "np.ndarray | None" = None,
    wls_iters: int = 2,
    greedy_flip: bool = True,
) -> tuple["np.ndarray", "np.ndarray", "np.ndarray", float]:
    """#5292 oQe Hessian-diagonal WLS scale/bias refit + MACaQ greedy code flip.

    Returns (codes_u8 [rows, K], scales_bf16_u16 [rows, G], biases_bf16_u16 [rows, G], weighted_mse).
    """
    if bits not in (3, 4, 5, 6, 8) or group_size not in (64, 128):
        raise ValueError(f"unsupported oQe spec bits={bits}, group_size={group_size}")
    w2d = np.asarray(weight_f32, dtype=np.float32)
    leading = w2d.shape[:-1]
    k = int(w2d.shape[-1])
    if k == 0 or (k % group_size) != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    rows = product(leading)
    groups = k // group_size
    w_g = w2d.reshape(rows, groups, group_size)
    if hessian_diag is None:
        h_g = np.ones((1, groups, group_size), dtype=np.float32)
    else:
        h_vec = np.maximum(np.asarray(hessian_diag, dtype=np.float32).reshape(-1), 1e-8)
        if h_vec.size != k:
            raise ValueError(f"hessian_diag length {h_vec.size} != K={k}")
        h_g = h_vec.reshape(1, groups, group_size)

    qmax = float((1 << bits) - 1)
    w_min = np.min(w_g, axis=2)
    w_max = np.max(w_g, axis=2)
    span = np.maximum(w_max - w_min, 1e-12)
    scale = span / qmax
    bias = w_min
    scale = _bf16_u16_to_float32(_float32_to_bf16_u16(scale))
    bias = _bf16_u16_to_float32(_float32_to_bf16_u16(bias))

    inv_s = np.where(scale > 0.0, 1.0 / scale, 0.0)
    q = np.clip(np.rint((w_g - bias[:, :, None]) * inv_s[:, :, None]), 0.0, qmax)

    # #5292 Alternating Hessian-weighted 2x2 least-squares refit of (scale, bias).
    s_h = np.sum(h_g, axis=2)  # [1, G]
    for _ in range(max(0, wls_iters)):
        s_hq = np.sum(h_g * q, axis=2)
        s_hq2 = np.sum(h_g * (q * q), axis=2)
        s_hw = np.sum(h_g * w_g, axis=2)
        s_hqw = np.sum(h_g * q * w_g, axis=2)
        det = s_h * s_hq2 - s_hq * s_hq
        valid = det > 1e-10
        new_s = np.where(
            valid, (s_h * s_hqw - s_hq * s_hw) / np.where(valid, det, 1.0), scale
        )
        new_s = np.maximum(new_s, 0.0)
        new_b = np.where(valid, (s_hw - new_s * s_hq) / s_h, bias)
        scale = _bf16_u16_to_float32(_float32_to_bf16_u16(new_s))
        bias = _bf16_u16_to_float32(_float32_to_bf16_u16(new_b))
        inv_s = np.where(scale > 0.0, 1.0 / scale, 0.0)
        q = np.clip(np.rint((w_g - bias[:, :, None]) * inv_s[:, :, None]), 0.0, qmax)

    # MACaQ coordinate-wise +/-1 flip check under BF16-rounded (scale, bias).
    if greedy_flip:
        recon = scale[:, :, None] * q + bias[:, :, None]
        base_err = h_g * ((recon - w_g) ** 2)
        for delta in (-1.0, 1.0):
            cand_q = np.clip(q + delta, 0.0, qmax)
            cand_err = h_g * ((scale[:, :, None] * cand_q + bias[:, :, None] - w_g) ** 2)
            improve = cand_err < base_err
            q = np.where(improve, cand_q, q)
            base_err = np.where(improve, cand_err, base_err)

    final_recon = scale[:, :, None] * q + bias[:, :, None]
    wmse = float(np.mean(h_g * ((final_recon - w_g) ** 2)))
    return (
        q.astype(np.uint8).reshape(*leading, k),
        _float32_to_bf16_u16(scale).reshape(*leading, groups),
        _float32_to_bf16_u16(bias).reshape(*leading, groups),
        wmse,
    )


def _pack_mlx_affine_u32(codes_u8: "np.ndarray", bits: int) -> "np.ndarray":
    leading = codes_u8.shape[:-1]
    k = int(codes_u8.shape[-1])
    rows = product(leading)
    flat = codes_u8.reshape(rows, k)
    if bits == 8:
        packed_u8 = flat
    elif bits == 4:
        packed_u8 = (flat[:, 0::2] & 0x0F) | ((flat[:, 1::2] & 0x0F) << 4)
    else:
        bit_planes = (
            (flat[:, :, None] >> np.arange(bits, dtype=np.uint8)) & 1
        ).reshape(rows, k * bits)
        packed_u8 = np.packbits(bit_planes, axis=1, bitorder="little")
    return packed_u8.view(np.uint32).reshape(*leading, k * bits // 32)


def _qda_role_kl_multiplier(module_name: str) -> tuple[str, float]:
    use = infer_weight_use(module_name)
    # #4776 & ninfer QDA sensitivity priors:
    # lm_head / embedding execute 1x per token (over-allocating Q8 wastes GiBs),
    # while down_proj, o_proj/out_proj, and value/gate projections compound across L layers.
    if use in ("lm_head", "embedding"):
        return use, 0.35
    if any(tok in module_name for tok in ("down_proj", "o_proj", "out_proj")):
        return use, 2.40
    if any(tok in module_name for tok in ("v_proj", "in_proj_z", "in_proj_vz", "gate_proj")):
        return use, 1.65
    if "router" in module_name or use == "moe_router":
        return "moe_router", 6.00
    return use, 1.00


def qda_search_checkpoint(
    source_dir: Path,
    *,
    target_bpw: float | None = None,
    target_gib: float | None = None,
    head_max_bits: int = 6,
    min_bits: int = 3,
    hessian_json: Path | None = None,
) -> dict:
    """#4776 + #5292 + QDA top-k KL knapsack bit-allocation optimizer."""
    import heapq

    source = SafeTensorSource(source_dir, None)
    try:
        hessian_map: dict[str, np.ndarray] = {}
        if hessian_json is not None and hessian_json.exists():
            raw_h = json.loads(hessian_json.read_text(encoding="utf-8"))
            for k_name, vals in raw_h.items():
                hessian_map[k_name] = np.asarray(vals, dtype=np.float32)

        tiers_all = [
            (3, 128),
            (3, 64),
            (4, 128),
            (4, 64),
            (5, 64),
            (6, 64),
            (8, 64),
        ]
        modules: list[dict] = []
        fixed_bytes = 0
        total_quant_weights = 0

        for name in sorted(source.tensors):
            info = source.tensors[name]
            if (
                not name.endswith(".weight")
                or len(info.shape) < 2
                or ".conv1d." in name
            ):
                if not name.endswith((".scales", ".biases", ".weight_scale")):
                    fixed_bytes += info.nbytes
                continue
            module = name[: -len(".weight")]
            # Support both float/bf16 checkpoints and already-quantized checkpoints (via scales shape).
            if (module + ".scales") in source.tensors:
                sc_info = source.tensors[module + ".scales"]
                spec = quant_spec(source.config, module)
                gs0 = int(spec.get("group_size") or 64)
                leading = sc_info.shape[:-1]
                k = int(sc_info.shape[-1]) * gs0
                w_var = 1.0
            else:
                if info.dtype not in ("BF16", "F16", "F32"):
                    fixed_bytes += info.nbytes
                    continue
                leading = info.shape[:-1]
                k = int(info.shape[-1])
                if k < 128 or (k % 64) != 0:
                    fixed_bytes += info.nbytes
                    continue
                view = source.view(name)
                try:
                    if info.dtype == "BF16":
                        raw_slice = bytes(view[: min(len(view), 262144 * 2)])
                        u16 = np.frombuffer(raw_slice, dtype=np.uint16)
                        sample = _bf16_u16_to_float32(u16)
                    elif info.dtype == "F16":
                        raw_slice = bytes(view[: min(len(view), 262144 * 2)])
                        sample = np.frombuffer(raw_slice, dtype=np.float16).astype(
                            np.float32
                        )
                    else:
                        raw_slice = bytes(view[: min(len(view), 262144 * 4)])
                        sample = np.frombuffer(raw_slice, dtype=np.float32)
                    h_vec = hessian_map.get(module)
                    if h_vec is not None and h_vec.size == k and sample.size >= k:
                        rows_s = sample.size // k
                        w_var = float(
                            np.mean((sample[: rows_s * k].reshape(rows_s, k) ** 2) * h_vec[None, :])
                        )
                    else:
                        w_var = float(np.mean(sample * sample))
                finally:
                    view.release()
                w_var = max(w_var, 1e-8)

            rows = product(leading)
            num_w = rows * k
            use, role_mult = _qda_role_kl_multiplier(module)
            mod_min_bits = max(min_bits, 4 if use in ("lm_head", "embedding") else min_bits)
            if use == "moe_router":
                mod_min_bits = 8
            mod_max_bits = head_max_bits if use in ("lm_head", "embedding") else 8
            if use == "moe_router":
                mod_max_bits = 8

            allowed = [
                (b, g)
                for (b, g) in tiers_all
                if mod_min_bits <= b <= mod_max_bits and (k % g) == 0
            ]
            if not allowed:
                allowed = [(4, 64)]

            def tier_bytes(b: int, g: int, n_rows: int = rows, n_k: int = k) -> int:
                pk = align_up(n_k, 128)
                code_b = n_rows * pk * b // 8
                sb_b = n_rows * (pk // g) * 4
                return code_b + sb_b

            def tier_kl_cost(b: int, g: int, kappa: float = role_mult * w_var * num_w) -> float:
                return kappa * (4.0 ** (-b)) * ((g / 64.0) ** 0.35)

            modules.append(
                {
                    "module": module,
                    "use": use,
                    "shape": [*leading, k],
                    "weights": num_w,
                    "role_kl_multiplier": role_mult,
                    "tiers": allowed,
                    "tier_bytes": [tier_bytes(b, g) for (b, g) in allowed],
                    "tier_cost": [tier_kl_cost(b, g) for (b, g) in allowed],
                    "idx": 0,
                }
            )
            total_quant_weights += num_w

        if not modules or total_quant_weights == 0:
            raise ValueError("qda-search found no eligible linear/expert weight tensors")

        if target_gib is not None:
            budget_bytes = int(target_gib * (1 << 30)) - fixed_bytes
        else:
            bpw = target_bpw if target_bpw is not None else 4.25
            budget_bytes = int(round(total_quant_weights * bpw / 8.0))

        current_bytes = sum(m["tier_bytes"][0] for m in modules)
        heap: list[tuple[float, int, int]] = []
        for m_i, m in enumerate(modules):
            if len(m["tiers"]) > 1:
                db = max(1, m["tier_bytes"][1] - m["tier_bytes"][0])
                dc = m["tier_cost"][0] - m["tier_cost"][1]
                heapq.heappush(heap, (-dc / db, m_i, 0))

        while heap and current_bytes < budget_bytes:
            neg_slope, m_i, expected_idx = heapq.heappop(heap)
            m = modules[m_i]
            if m["idx"] != expected_idx:
                continue
            nxt = expected_idx + 1
            delta_b = m["tier_bytes"][nxt] - m["tier_bytes"][expected_idx]
            if current_bytes + delta_b > budget_bytes:
                continue
            m["idx"] = nxt
            current_bytes += delta_b
            if nxt + 1 < len(m["tiers"]):
                db2 = max(1, m["tier_bytes"][nxt + 1] - m["tier_bytes"][nxt])
                dc2 = m["tier_cost"][nxt] - m["tier_cost"][nxt + 1]
                heapq.heappush(heap, (-dc2 / db2, m_i, nxt))

        overrides: dict[str, dict[str, int | str]] = {}
        hist: dict[str, int] = {}
        for m in modules:
            b, g = m["tiers"][m["idx"]]
            key = f"q{b}_g{g}"
            hist[key] = hist.get(key, 0) + 1
            overrides[m["module"]] = {"bits": b, "group_size": g, "mode": "affine"}

        effective_bpw = (current_bytes * 8.0) / float(total_quant_weights)
        total_kl_proxy = sum(m["tier_cost"][m["idx"]] for m in modules)
        return {
            "schema": "omlx-qda-recipe-v1",
            "source_dir": str(source_dir.resolve()),
            "quantized_modules": len(modules),
            "quantized_weights": total_quant_weights,
            "quantized_bytes": current_bytes,
            "fixed_non_quant_bytes": fixed_bytes,
            "estimated_total_bytes": current_bytes + fixed_bytes,
            "effective_bpw": round(effective_bpw, 4),
            "head_max_bits": head_max_bits,
            "min_bits": min_bits,
            "kl_objective_proxy": total_kl_proxy,
            "tier_histogram": dict(sorted(hist.items())),
            "quantization": {
                "bits": 4,
                "group_size": 64,
                "mode": "affine",
                **overrides,
            },
        }
    finally:
        source.close()


def quantize_source_with_recipe_to_artifact(
    source_dir: Path,
    recipe: dict,
    output_artifact: Path,
    source_repo: str,
    *,
    nax_pretile: bool = False,
    hessian_json: Path | None = None,
) -> dict:
    """Apply #5292 oQe WLS+flip quantization using a QDA recipe and emit .omlxm5."""
    hessian_map: dict[str, np.ndarray] = {}
    if hessian_json is not None and hessian_json.exists():
        raw_h = json.loads(hessian_json.read_text(encoding="utf-8"))
        for k_name, vals in raw_h.items():
            hessian_map[k_name] = np.asarray(vals, dtype=np.float32)

    qtable = recipe["quantization"]
    src = SafeTensorSource(source_dir, None)
    try:
        provenance = {
            "source_revision": src.revision,
            "source_config_sha256": file_sha256_if_present(
                src.model_dir / "config.json"
            ),
            "source_index_sha256": file_sha256_if_present(
                src.model_dir / "model.safetensors.index.json"
            ),
        }
        with tempfile.TemporaryDirectory(prefix="omlx_oqe_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            effective_qtable = dict(qtable)
            has_passthrough_mxfp4 = False
            tensors_out: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
            for name, info in sorted(src.tensors.items()):
                view = src.view(name)
                raw_copy = bytes(view)
                view.release()
                if name.endswith(".weight"):
                    module = name[: -len(".weight")]
                    if (module + ".scales") in src.tensors:
                        if (module + ".biases") in src.tensors:
                            src_spec = quant_spec(src.config, module)
                            effective_qtable[module] = {
                                "bits": int(src_spec["bits"]),
                                "group_size": int(src_spec["group_size"]),
                                "mode": "affine",
                            }
                        elif src.tensors[module + ".scales"].dtype in (
                            "U8",
                            "F8_E8M0",
                        ):
                            has_passthrough_mxfp4 = True
                            effective_qtable.pop(module, None)
                        tensors_out[name] = (info.dtype, info.shape, raw_copy)
                        continue
                    if module in qtable and isinstance(qtable[module], dict):
                        mspec = qtable[module]
                        bits = int(mspec["bits"])
                        gs = int(mspec["group_size"])
                        if info.dtype == "BF16":
                            w_f32 = _bf16_u16_to_float32(
                                np.frombuffer(raw_copy, dtype=np.uint16)
                            ).reshape(info.shape)
                        elif info.dtype == "F16":
                            w_f32 = (
                                np.frombuffer(raw_copy, dtype=np.float16)
                                .astype(np.float32)
                                .reshape(info.shape)
                            )
                        elif info.dtype == "F32":
                            w_f32 = np.frombuffer(raw_copy, dtype=np.float32).reshape(
                                info.shape
                            )
                        else:
                            raise ValueError(
                                f"{name}: oQe quantization requires BF16/F16/F32 source"
                            )
                        codes, sc_u16, bi_u16, _ = quantize_affine_oqe(
                            w_f32, bits, gs, hessian_map.get(module)
                        )
                        packed_u32 = _pack_mlx_affine_u32(codes, bits)
                        tensors_out[module + ".weight"] = (
                            "U32",
                            packed_u32.shape,
                            packed_u32.tobytes(order="C"),
                        )
                        tensors_out[module + ".scales"] = (
                            "BF16",
                            sc_u16.shape,
                            sc_u16.tobytes(order="C"),
                        )
                        tensors_out[module + ".biases"] = (
                            "BF16",
                            bi_u16.shape,
                            bi_u16.tobytes(order="C"),
                        )
                        continue
                tensors_out[name] = (info.dtype, info.shape, raw_copy)

            if has_passthrough_mxfp4:
                effective_qtable["mode"] = "mxfp4"
                effective_qtable["bits"] = 4
                effective_qtable["group_size"] = 32
            cfg = dict(src.config)
            cfg["quantization"] = effective_qtable
            (tmp_dir / "config.json").write_text(
                json.dumps(cfg, indent=2, sort_keys=True), encoding="utf-8"
            )
            out_profile = "mlx_mxfp4" if has_passthrough_mxfp4 else "oq_affine"

            # Write single model.safetensors in tmp_dir and convert to .omlxm5.
            st_header: dict[str, Any] = {}
            cursor = 0
            for tname, (dt, shp, raw_b) in tensors_out.items():
                st_header[tname] = {
                    "dtype": dt,
                    "shape": list(shp),
                    "data_offsets": [cursor, cursor + len(raw_b)],
                }
                cursor += len(raw_b)
            hdr_bytes = json.dumps(st_header, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
            pad_hdr = (8 - (len(hdr_bytes) % 8)) % 8
            hdr_bytes += b" " * pad_hdr
            with (tmp_dir / "model.safetensors").open("wb") as st_f:
                st_f.write(struct.pack("<Q", len(hdr_bytes)))
                st_f.write(hdr_bytes)
                for _, (_, _, raw_b) in tensors_out.items():
                    st_f.write(raw_b)

            q_src = SafeTensorSource(tmp_dir, None)
            try:
                if nax_pretile:
                    final_output = output_artifact.resolve()
                    untiled = final_output.with_suffix(
                        final_output.suffix + ".untiled.tmp"
                    )
                    untiled_report = _conversion_report_path(untiled)
                    try:
                        rep = write_artifact(
                            q_src,
                            untiled,
                            source_repo,
                            out_profile,
                            source_provenance=provenance,
                            write_report=False,
                        )
                        pt = pretile_artifact(untiled, final_output)
                        rep = _finalize_pretiled_conversion_report(
                            rep, final_output, pt
                        )
                    finally:
                        if untiled.exists():
                            untiled.unlink()
                        if untiled_report.exists():
                            untiled_report.unlink()
                else:
                    rep = write_artifact(
                        q_src,
                        output_artifact,
                        source_repo,
                        out_profile,
                        source_provenance=provenance,
                    )
                return rep
            finally:
                q_src.close()
    finally:
        src.close()


def command_qda_search(args: argparse.Namespace) -> None:
    recipe = qda_search_checkpoint(
        args.source,
        target_bpw=args.target_bpw,
        target_gib=args.target_gib,
        head_max_bits=args.head_max_bits,
        min_bits=args.min_bits,
        hessian_json=args.hessian_stats,
    )
    if args.output_recipe is not None:
        args.output_recipe.parent.mkdir(parents=True, exist_ok=True)
        args.output_recipe.write_text(
            json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    summary = {k: v for k, v in recipe.items() if k != "quantization"}
    if args.output_artifact is not None:
        art_rep = quantize_source_with_recipe_to_artifact(
            args.source,
            recipe,
            args.output_artifact,
            args.source_repo,
            nax_pretile=args.nax_pretile,
            hessian_json=args.hessian_stats,
        )
        summary["artifact"] = str(args.output_artifact.resolve())
        summary["artifact_bytes"] = art_rep["artifact_bytes"]
        summary["pretiled_objects"] = art_rep.get("pretiled_objects", 0)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_inspect(args: argparse.Namespace) -> None:
    summary, manifest = read_artifact(args.artifact, args.verify_payload)
    if args.list_objects:
        summary["objects"] = [
            {
                "id": item["id"],
                "name": item["name"],
                "format": item["format"],
                "shape": item["shape"],
            }
            for item in manifest["objects"]
        ]
    if args.sample_object:
        summary["sample"] = {
            "object": args.sample_object,
            "row": args.sample_row,
            "start": args.sample_start,
            "values": decode_artifact_row(
                args.artifact,
                manifest,
                args.sample_object,
                args.sample_row,
                args.sample_start,
                args.sample_count,
            ),
        }
    print(json.dumps(summary, indent=2, sort_keys=True))



def _is_mtp_object_name(name: str) -> bool:
    return name.startswith("mtp.") or ".mtp." in name


def _format_registry_for_objects(object_records: list[dict]) -> dict[str, dict]:
    registry: dict[str, dict] = {}
    for format_id in sorted(
        {
            str(record["format"])
            for record in object_records
            if not str(record["format"]).startswith("direct_")
        }
    ):
        if format_id.startswith("oq_affine_q"):
            fields = format_id.split("_")
            registry[format_id] = {
                "code_bits": int(fields[2].removeprefix("q")),
                "group_size": int(fields[3].removeprefix("g")),
                "scale": "BF16",
                "bias": "BF16",
            }
        elif format_id == "mxfp4_e2m1_g32_e8m0_v1":
            registry[format_id] = {
                "code": "E2M1",
                "group_size": 32,
                "scale": "E8M0",
            }
        else:
            raise ValueError(f"unsupported format registry entry {format_id}")
    return registry


def _write_container_from_payload(
    output_path: Path, manifest: dict, payload_name: str
) -> tuple[str, int, str]:
    manifest_bytes = canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_crc = crc32(manifest_bytes)
    payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), PAYLOAD_ALIGNMENT)
    payload_bytes = int(manifest["payload_bytes"])
    object_count = len(manifest["objects"])
    flags = 0
    header_prefix = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        HEADER_BYTES,
        len(manifest_bytes),
        payload_offset,
        payload_bytes,
        object_count,
        manifest_crc,
        0,
        flags,
    )
    header_crc = crc32(header_prefix[:56])
    header = HEADER_STRUCT.pack(
        MAGIC,
        VERSION,
        HEADER_BYTES,
        HEADER_BYTES,
        len(manifest_bytes),
        payload_offset,
        payload_bytes,
        object_count,
        manifest_crc,
        header_crc,
        flags,
    )
    final_fd, final_name = tempfile.mkstemp(
        prefix=output_path.name + ".tmp.", dir=output_path.parent
    )
    os.close(final_fd)
    try:
        with open(final_name, "w+b") as out_f, open(payload_name, "rb") as pay_f:
            out_f.write(header)
            out_f.write(manifest_bytes)
            out_f.write(b"\0" * (payload_offset - out_f.tell()))
            while chunk := pay_f.read(16 * 1024 * 1024):
                out_f.write(chunk)
            out_f.flush()
            os.fsync(out_f.fileno())
        os.replace(final_name, output_path)
    finally:
        if os.path.exists(final_name):
            os.unlink(final_name)
    return sha256_file(output_path), manifest_crc, manifest_sha256


def _copy_selected_artifact_objects(
    src_path: Path,
    src_manifest: dict,
    predicate: Callable[[str], bool],
    dst_payload: BinaryIO,
    writer: PayloadWriter,
) -> list[tuple[dict, dict | None]]:
    bindings_by_obj = {
        int(b["object_id"]): dict(b) for b in src_manifest.get("bindings", [])
    }
    emitted: list[tuple[dict, dict | None]] = []
    with src_path.open("rb") as src_f:
        src_header = _read_exact(src_f, HEADER_BYTES, "artifact header")
        src_payload_offset = HEADER_STRUCT.unpack(src_header)[5]
        for obj in src_manifest["objects"]:
            if not predicate(str(obj["name"])):
                continue
            writer.align(OBJECT_ALIGNMENT)
            object_begin = writer.offset
            new_planes: list[dict] = []
            for p in obj["planes"]:
                alignment = int(p.get("alignment", PLANE_ALIGNMENT))
                writer.align(alignment)
                p_off = writer.offset
                src_f.seek(src_payload_offset + int(p["offset"]))
                raw = _read_exact(src_f, int(p["bytes"]), str(p["kind"]))
                written, csum, p_sha = writer.write_chunks([memoryview(raw)])
                new_planes.append(
                    {
                        "kind": p["kind"],
                        "offset": p_off,
                        "bytes": written,
                        "alignment": alignment,
                        "crc32": f"{csum:08x}",
                        "sha256": p_sha,
                    }
                )
            dst_payload.flush()
            object_bytes = writer.offset - object_begin
            rec = {
                **obj,
                "object_offset": object_begin,
                "object_bytes": object_bytes,
                "object_sha256": sha256_range(dst_payload, object_begin, object_bytes),
                "planes": new_planes,
            }
            emitted.append((rec, bindings_by_obj.get(int(obj["id"]))))
    return emitted


def extract_mtp_artifact(
    source_path: Path,
    output_path: Path,
    source_repo: str = "",
    *,
    expected_revision: str | None = None,
    profile: str = "auto",
    nax_pretile: bool = False,
) -> dict:
    """Extract only ModelComponent::mtp weights from a checkpoint dir or .omlxm5 into a standalone .omlxm5."""
    source_path = source_path.resolve()
    output_path = output_path.resolve()
    if nax_pretile:
        tmp_out = output_path.with_suffix(output_path.suffix + ".untiled.tmp")
        try:
            rep = extract_mtp_artifact(
                source_path,
                tmp_out,
                source_repo,
                expected_revision=expected_revision,
                profile=profile,
                nax_pretile=False,
            )
            pt = pretile_artifact(tmp_out, output_path)
            rep["artifact"] = pt["artifact"]
            rep["artifact_bytes"] = pt["artifact_bytes"]
            rep["artifact_sha256"] = pt["artifact_sha256"]
            rep["manifest_crc32"] = pt["manifest_crc32"]
            rep["manifest_sha256"] = pt["manifest_sha256"]
            rep["payload_bytes"] = pt["payload_bytes"]
            rep["object_count"] = pt["object_count"]
            rep["pretiled_objects"] = pt["pretiled_objects"]
            return rep
        finally:
            if tmp_out.exists():
                tmp_out.unlink()

    if source_path.is_file():
        if source_path == output_path:
            raise ValueError("extract-mtp output must differ from source artifact path")
        _, src_manifest = read_artifact(source_path, verify_payload=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload_fd, payload_name = tempfile.mkstemp(
            prefix=output_path.name + ".payload.", dir=output_path.parent
        )
        os.close(payload_fd)
        try:
            with open(payload_name, "w+b") as dst_payload:
                writer = PayloadWriter(dst_payload)
                selected = _copy_selected_artifact_objects(
                    source_path, src_manifest, _is_mtp_object_name, dst_payload, writer
                )
                if not selected:
                    raise ValueError(f"{source_path}: artifact contains no MTP objects")
                dst_payload.flush()
                os.fsync(dst_payload.fileno())
                payload_bytes = writer.offset

            selected.sort(key=lambda item: str(item[0]["name"]))
            objects: list[dict] = []
            bindings: list[dict] = []
            for new_id, (rec, bind) in enumerate(selected):
                rec_copy = {**rec, "id": new_id}
                objects.append(rec_copy)
                if bind is not None:
                    bindings.append({**bind, "object_id": new_id})

            model_meta = dict(src_manifest.get("model", {}))
            if source_repo:
                model_meta["source_repo"] = source_repo
            manifest = {
                "schema": SCHEMA,
                "container_version": VERSION,
                "model": model_meta,
                "converter": dict(src_manifest.get("converter", {"name": "omlx-m5-artifact", "version": 1, "endianness": "little"})),
                "format_registry": _format_registry_for_objects(objects),
                "objects": objects,
                "views": [],
                "bindings": bindings,
                "payload_bytes": payload_bytes,
            }
            art_sha, manifest_crc, manifest_sha = _write_container_from_payload(
                output_path, manifest, payload_name
            )
            counts = Counter(str(r["format"]) for r in objects)
            return {
                "schema": "omlx-m5-mtp-extract-report-v1",
                "artifact": str(output_path),
                "artifact_bytes": output_path.stat().st_size,
                "artifact_sha256": art_sha,
                "manifest_crc32": f"{manifest_crc:08x}",
                "manifest_sha256": manifest_sha,
                "payload_bytes": payload_bytes,
                "object_count": len(objects),
                "binding_count": len(bindings),
                "formats": dict(sorted(counts.items())),
            }
        finally:
            if os.path.exists(payload_name):
                os.unlink(payload_name)

    source = SafeTensorSource(source_path, expected_revision)
    try:
        mtp_tensor_names = [n for n in source.tensors if _is_mtp_object_name(n)]
        if not mtp_tensor_names:
            raise ValueError(f"{source_path}: checkpoint contains no MTP tensors")
        if profile == "auto":
            try:
                detected = detect_profile(source)
            except ValueError:
                detected = "direct_bf16"
        else:
            detected = profile

        if detected == "direct_bf16":
            all_recipes = build_direct_recipes(source, set())
        else:
            if detected == "oq_affine":
                try:
                    quantized, consumed = build_oq_recipes(source)
                except ValueError:
                    quantized, consumed = [], set()
            elif detected == "mlx_mxfp4":
                try:
                    quantized, consumed = build_mlx_mxfp4_recipes(source)
                except ValueError:
                    quantized, consumed = [], set()
            elif detected == "compressed_mxfp4":
                try:
                    quantized, consumed = build_compressed_mxfp4_recipes(source)
                except ValueError:
                    quantized, consumed = [], set()
            else:
                raise ValueError(f"unsupported MTP source profile {detected}")
            all_recipes = sorted(
                quantized + build_direct_recipes(source, consumed),
                key=lambda item: item.name,
            )

        recipes = [r for r in all_recipes if _is_mtp_object_name(r.name)]
        if not recipes:
            raise ValueError(f"{source_path}: no MTP object recipes produced")

        protected_paths = {shard.path.resolve() for shard in source.shards.values()} | {
            (source.model_dir / "config.json").resolve(),
            (source.model_dir / "model.safetensors.index.json").resolve(),
            (source.model_dir / "REVISION").resolve(),
        }
        if output_path in protected_paths:
            raise ValueError("artifact output would overwrite a source checkpoint file")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload_fd, payload_name = tempfile.mkstemp(
            prefix=output_path.name + ".payload.", dir=output_path.parent
        )
        os.close(payload_fd)
        object_records: list[dict] = []
        bindings = []
        try:
            with open(payload_name, "w+b") as payload:
                writer = PayloadWriter(payload)
                for object_id, recipe in enumerate(recipes):
                    writer.align(OBJECT_ALIGNMENT)
                    object_begin = writer.offset
                    plane_records = []
                    for plane in recipe.planes:
                        writer.align(plane.alignment)
                        plane_offset = writer.offset
                        written, checksum, plane_sha256 = plane.writer(payload)
                        if written != plane.logical_bytes:
                            raise ValueError(
                                f"{recipe.name}/{plane.kind}: wrote {written}, expected {plane.logical_bytes}"
                            )
                        plane_records.append(
                            {
                                "kind": plane.kind,
                                "offset": plane_offset,
                                "bytes": written,
                                "alignment": plane.alignment,
                                "crc32": f"{checksum:08x}",
                                "sha256": plane_sha256,
                            }
                        )
                    payload.flush()
                    object_bytes = writer.offset - object_begin
                    rec = {
                        "id": object_id,
                        "name": recipe.name,
                        "format": recipe.format_id,
                        "layout": recipe.layout_id,
                        "shape": list(recipe.shape),
                        "object_offset": object_begin,
                        "object_bytes": object_bytes,
                        "object_sha256": sha256_range(payload, object_begin, object_bytes),
                        "planes": plane_records,
                        "source_tensors": recipe.source_tensors,
                    }
                    if recipe.padded_k is not None:
                        rec["padded_k"] = recipe.padded_k
                    object_records.append(rec)
                    if recipe.binding is not None:
                        bindings.append({**recipe.binding, "object_id": object_id})
                payload.flush()
                os.fsync(payload.fileno())
                payload_bytes = writer.offset

            repo_tag = source_repo or source.model_dir.name
            manifest = {
                "schema": SCHEMA,
                "container_version": VERSION,
                "model": {
                    "model_type": source.config.get("model_type", ""),
                    "text_model_type": (source.config.get("text_config") or {}).get(
                        "model_type", ""
                    ),
                    "source_repo": repo_tag,
                    "source_revision": source.revision,
                    "source_profile": detected,
                    "source_config_sha256": file_sha256_if_present(
                        source.model_dir / "config.json"
                    ),
                    "source_index_sha256": file_sha256_if_present(
                        source.model_dir / "model.safetensors.index.json"
                    ),
                    "source_tensor_count": len(mtp_tensor_names),
                    "source_tensor_bytes": sum(
                        source.tensors[n].nbytes for n in mtp_tensor_names
                    ),
                },
                "converter": {
                    "name": "omlx-m5-artifact",
                    "version": 1,
                    "endianness": "little",
                },
                "format_registry": _format_registry_for_objects(object_records),
                "objects": object_records,
                "views": [],
                "bindings": bindings,
                "payload_bytes": payload_bytes,
            }
            art_sha, manifest_crc, manifest_sha = _write_container_from_payload(
                output_path, manifest, payload_name
            )
            counts = Counter(str(r["format"]) for r in object_records)
            return {
                "schema": "omlx-m5-mtp-extract-report-v1",
                "artifact": str(output_path),
                "artifact_bytes": output_path.stat().st_size,
                "artifact_sha256": art_sha,
                "manifest_crc32": f"{manifest_crc:08x}",
                "manifest_sha256": manifest_sha,
                "payload_bytes": payload_bytes,
                "object_count": len(object_records),
                "binding_count": len(bindings),
                "formats": dict(sorted(counts.items())),
            }
        finally:
            if os.path.exists(payload_name):
                os.unlink(payload_name)
    finally:
        source.close()


def bundle_mtp_artifact(
    base_artifact_path: Path,
    mtp_source_path: Path,
    output_path: Path,
    *,
    nax_pretile: bool = False,
) -> dict:
    """Bundle MTP objects from a checkpoint dir or .omlxm5 into an existing base .omlxm5 without mutating backbone planes."""
    base_artifact_path = base_artifact_path.resolve()
    mtp_source_path = mtp_source_path.resolve()
    output_path = output_path.resolve()
    if output_path in (base_artifact_path, mtp_source_path):
        raise ValueError("bundle-mtp output path must differ from base and mtp inputs")

    _, base_manifest = read_artifact(base_artifact_path, verify_payload=False)
    base_has_pretiled = any(
        str(obj.get("layout", "")) in ("mx_nax_tiled_bk128_v1", "oq_nax_tiled_g64_v1")
        for obj in base_manifest["objects"]
        if not _is_mtp_object_name(str(obj["name"]))
    )
    want_mtp_pretile = nax_pretile or base_has_pretiled

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_mtp_artifact: Path | None = None
    try:
        if mtp_source_path.is_dir():
            fd, tmp_name = tempfile.mkstemp(
                prefix=output_path.name + ".mtp_src.",
                suffix=".omlxm5",
                dir=output_path.parent,
            )
            os.close(fd)
            tmp_mtp_artifact = Path(tmp_name)
            extract_mtp_artifact(
                mtp_source_path,
                tmp_mtp_artifact,
                source_repo=str(base_manifest.get("model", {}).get("source_repo", "")),
                nax_pretile=want_mtp_pretile,
            )
            mtp_art_path = tmp_mtp_artifact
        elif want_mtp_pretile:
            _, raw_mtp_manifest = read_artifact(mtp_source_path, verify_payload=False)
            already_tiled = any(
                str(obj.get("layout", ""))
                in ("mx_nax_tiled_bk128_v1", "oq_nax_tiled_g64_v1")
                for obj in raw_mtp_manifest["objects"]
                if _is_mtp_object_name(str(obj["name"]))
            )
            if not already_tiled:
                fd, tmp_name = tempfile.mkstemp(
                    prefix=output_path.name + ".mtp_tiled.",
                    suffix=".omlxm5",
                    dir=output_path.parent,
                )
                os.close(fd)
                tmp_mtp_artifact = Path(tmp_name)
                extract_mtp_artifact(
                    mtp_source_path, tmp_mtp_artifact, nax_pretile=True
                )
                mtp_art_path = tmp_mtp_artifact
            else:
                mtp_art_path = mtp_source_path
        else:
            mtp_art_path = mtp_source_path

        _, mtp_manifest = read_artifact(mtp_art_path, verify_payload=False)
        base_objs = [
            obj
            for obj in base_manifest["objects"]
            if not _is_mtp_object_name(str(obj["name"]))
        ]
        mtp_objs = [
            obj
            for obj in mtp_manifest["objects"]
            if _is_mtp_object_name(str(obj["name"]))
        ]
        if not mtp_objs:
            raise ValueError(f"{mtp_source_path}: contains no MTP objects to bundle")

        base_binds = {
            int(b["object_id"]): dict(b) for b in base_manifest.get("bindings", [])
        }
        mtp_binds = {
            int(b["object_id"]): dict(b) for b in mtp_manifest.get("bindings", [])
        }
        plan_entries: list[tuple[str, Path, dict, dict | None]] = []
        for obj in base_objs:
            plan_entries.append(
                ("base", base_artifact_path, obj, base_binds.get(int(obj["id"])))
            )
        for obj in mtp_objs:
            plan_entries.append(
                ("mtp", mtp_art_path, obj, mtp_binds.get(int(obj["id"])))
            )
        plan_entries.sort(key=lambda entry: str(entry[2]["name"]))

        payload_fd, payload_name = tempfile.mkstemp(
            prefix=output_path.name + ".payload.", dir=output_path.parent
        )
        os.close(payload_fd)
        try:
            merged_objects: list[dict] = []
            merged_bindings: list[dict] = []
            with (
                base_artifact_path.open("rb") as base_f,
                mtp_art_path.open("rb") as mtp_f,
                open(payload_name, "w+b") as dst_payload,
            ):
                base_pay_off = HEADER_STRUCT.unpack(
                    _read_exact(base_f, HEADER_BYTES, "base header")
                )[5]
                mtp_pay_off = HEADER_STRUCT.unpack(
                    _read_exact(mtp_f, HEADER_BYTES, "mtp header")
                )[5]
                writer = PayloadWriter(dst_payload)
                for new_id, (tag, _, obj, bind) in enumerate(plan_entries):
                    src_f = base_f if tag == "base" else mtp_f
                    src_pay_off = base_pay_off if tag == "base" else mtp_pay_off
                    writer.align(OBJECT_ALIGNMENT)
                    object_begin = writer.offset
                    new_planes = []
                    for p in obj["planes"]:
                        alignment = int(p.get("alignment", PLANE_ALIGNMENT))
                        writer.align(alignment)
                        p_off = writer.offset
                        src_f.seek(src_pay_off + int(p["offset"]))
                        raw = _read_exact(src_f, int(p["bytes"]), str(p["kind"]))
                        written, csum, p_sha = writer.write_chunks([memoryview(raw)])
                        new_planes.append(
                            {
                                "kind": p["kind"],
                                "offset": p_off,
                                "bytes": written,
                                "alignment": alignment,
                                "crc32": f"{csum:08x}",
                                "sha256": p_sha,
                            }
                        )
                    dst_payload.flush()
                    object_bytes = writer.offset - object_begin
                    rec = {
                        **obj,
                        "id": new_id,
                        "object_offset": object_begin,
                        "object_bytes": object_bytes,
                        "object_sha256": sha256_range(
                            dst_payload, object_begin, object_bytes
                        ),
                        "planes": new_planes,
                    }
                    merged_objects.append(rec)
                    if bind is not None:
                        merged_bindings.append({**bind, "object_id": new_id})
                dst_payload.flush()
                os.fsync(dst_payload.fileno())
                payload_bytes = writer.offset

            manifest = {
                **base_manifest,
                "format_registry": _format_registry_for_objects(merged_objects),
                "objects": merged_objects,
                "views": [],
                "bindings": merged_bindings,
                "payload_bytes": payload_bytes,
            }
            art_sha, manifest_crc, manifest_sha = _write_container_from_payload(
                output_path, manifest, payload_name
            )
            return {
                "schema": "omlx-m5-mtp-bundle-report-v1",
                "artifact": str(output_path),
                "artifact_bytes": output_path.stat().st_size,
                "artifact_sha256": art_sha,
                "manifest_crc32": f"{manifest_crc:08x}",
                "manifest_sha256": manifest_sha,
                "base_objects": len(base_objs),
                "mtp_objects": len(mtp_objs),
                "object_count": len(merged_objects),
                "binding_count": len(merged_bindings),
            }
        finally:
            if os.path.exists(payload_name):
                os.unlink(payload_name)
    finally:
        if tmp_mtp_artifact is not None and tmp_mtp_artifact.exists():
            tmp_mtp_artifact.unlink()


def command_extract_mtp(args: argparse.Namespace) -> None:
    report = extract_mtp_artifact(
        args.source,
        args.output,
        args.source_repo,
        expected_revision=args.expected_revision,
        profile=args.profile,
        nax_pretile=args.nax_pretile,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def command_bundle_mtp(args: argparse.Namespace) -> None:
    report = bundle_mtp_artifact(
        args.base,
        args.mtp,
        args.output,
        nax_pretile=args.nax_pretile,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    convert = subparsers.add_parser("convert", help="convert a local checkpoint")
    convert.add_argument("--source", type=Path, required=True)
    convert.add_argument("--output", type=Path, required=True)
    convert.add_argument("--source-repo", required=True)
    convert.add_argument("--expected-revision")
    convert.add_argument(
        "--profile",
        choices=(
            "auto",
            "oq_affine",
            "mlx_mxfp4",
            "compressed_mxfp4",
            "dflash2_bf16",
        ),
        default="auto",
    )
    convert.add_argument(
        "--nax-pretile",
        action="store_true",
        help="offline pre-tile rank-2 MXFP4 and Q3/Q4/Q5/Q6 G64/G128 weights for zero-repack M5 NAX",
    )
    convert.set_defaults(func=command_convert)
    pretile = subparsers.add_parser(
        "pretile",
        help="offline pre-tile an existing .omlxm5 artifact for zero-repack M5 NAX",
    )
    pretile.add_argument("--input", type=Path, required=True)
    pretile.add_argument("--output", type=Path, required=True)
    pretile.set_defaults(func=command_pretile)
    extract_mtp = subparsers.add_parser(
        "extract-mtp",
        help="extract ModelComponent::mtp weights from a checkpoint directory or .omlxm5 into a standalone .omlxm5",
    )
    extract_mtp.add_argument("--source", type=Path, required=True)
    extract_mtp.add_argument("--output", type=Path, required=True)
    extract_mtp.add_argument("--source-repo", default="")
    extract_mtp.add_argument("--expected-revision", default=None)
    extract_mtp.add_argument(
        "--profile",
        choices=(
            "auto",
            "oq_affine",
            "mlx_mxfp4",
            "compressed_mxfp4",
            "direct_bf16",
        ),
        default="auto",
    )
    extract_mtp.add_argument("--nax-pretile", action="store_true")
    extract_mtp.set_defaults(func=command_extract_mtp)
    bundle_mtp = subparsers.add_parser(
        "bundle-mtp",
        help="bundle ModelComponent::mtp weights from a checkpoint directory or .omlxm5 into an existing base .omlxm5",
    )
    bundle_mtp.add_argument("--base", type=Path, required=True)
    bundle_mtp.add_argument("--mtp", type=Path, required=True)
    bundle_mtp.add_argument("--output", type=Path, required=True)
    bundle_mtp.add_argument("--nax-pretile", action="store_true")
    bundle_mtp.set_defaults(func=command_bundle_mtp)
    qda = subparsers.add_parser(
        "qda-search",
        help="#4776 + #5292 oQe + QDA top-k KL knapsack bit-allocation search & optional .omlxm5 export",
    )
    qda.add_argument("--source", type=Path, required=True)
    qda.add_argument("--target-bpw", type=float, default=None)
    qda.add_argument("--target-gib", type=float, default=None)
    qda.add_argument("--head-max-bits", type=int, default=6)
    qda.add_argument("--min-bits", type=int, default=3)
    qda.add_argument("--hessian-stats", type=Path, default=None)
    qda.add_argument("--output-recipe", type=Path, default=None)
    qda.add_argument("--output-artifact", type=Path, default=None)
    qda.add_argument("--source-repo", default="local/qda-oqe")
    qda.add_argument("--nax-pretile", action="store_true")
    qda.set_defaults(func=command_qda_search)
    inspect = subparsers.add_parser("inspect", help="inspect an artifact")
    inspect.add_argument("artifact", type=Path)
    inspect.add_argument("--verify-payload", action="store_true")
    inspect.add_argument("--list-objects", action="store_true")
    inspect.add_argument("--sample-object")
    inspect.add_argument("--sample-row", type=int, default=0)
    inspect.add_argument("--sample-start", type=int, default=0)
    inspect.add_argument("--sample-count", type=int, default=16)
    inspect.set_defaults(func=command_inspect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
