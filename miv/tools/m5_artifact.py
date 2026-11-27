#!/usr/bin/env python3
"""Build and inspect deterministic .miv (ninfer-metal / miv) backend artifacts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import mmap
import os
import struct
import tempfile
import time
import zlib
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

MAGIC = b"MIV00001"
LEGACY_MAGIC = b"OMLXM5A1"
VALID_MAGICS = (MAGIC, LEGACY_MAGIC)
VERSION = 1
VERSION_V1 = 1
VERSION_V2 = 2
DEFAULT_ARTIFACT_VERSION = VERSION_V2
FLAG_FWHT64 = 0x1
FLAG_NAX_LANE_V2 = 0x2
VALID_V2_FLAGS_MASK = FLAG_FWHT64 | FLAG_NAX_LANE_V2
HEADER_BYTES = 64
PAYLOAD_ALIGNMENT = 16384
LEGACY_PAYLOAD_ALIGNMENT = 4096
OBJECT_ALIGNMENT = 256
PLANE_ALIGNMENT = 256
HEADER_STRUCT = struct.Struct("<8sIIQQQQIIII")
SCHEMA = "miv-artifact-v1"
SCHEMA_V1 = "miv-artifact-v1"
SCHEMA_V2 = "miv-artifact-v2"
LEGACY_SCHEMA_V1 = "omlx-m5-artifact-v1"
LEGACY_SCHEMA_V2 = "omlx-m5-artifact-v2"
VALID_SCHEMAS_V1 = (SCHEMA_V1, LEGACY_SCHEMA_V1)
VALID_SCHEMAS_V2 = (SCHEMA_V2, LEGACY_SCHEMA_V2)
CONVERTER_NAME = "miv-artifact"
LEGACY_CONVERTER_NAME = "omlx-m5-artifact"
DEFAULT_JOBS = min(32, os.cpu_count() or 8)


def _branding_for_write(
    artifact_version: int, *, legacy_magic: bool = False
) -> tuple[bytes, str, str]:
    if legacy_magic:
        magic = LEGACY_MAGIC
        schema_id = LEGACY_SCHEMA_V2 if artifact_version == VERSION_V2 else LEGACY_SCHEMA_V1
        converter_name = LEGACY_CONVERTER_NAME
    else:
        magic = MAGIC
        schema_id = SCHEMA_V2 if artifact_version == VERSION_V2 else SCHEMA_V1
        converter_name = CONVERTER_NAME
    return magic, schema_id, converter_name

_E2M1_MAG_LUT = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
_E2M1_LUT = np.array(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=np.float32,
)
_E8M0_LUT = np.array([2.0 ** (i - 127) for i in range(255)] + [np.nan], dtype=np.float32)
_E8M0_INV_LUT = np.array(
    [2.0 ** (127 - i) for i in range(255)] + [np.nan], dtype=np.float32
)
_LUT_4X1_TO_4X2 = np.array(
    [
        (nib & 1) | ((nib & 2) << 1) | ((nib & 4) << 2) | ((nib & 8) << 3)
        for nib in range(16)
    ],
    dtype=np.uint8,
)
_AFFINE_NAX_V2_LAYOUTS = ("oq_nax_lane_g64_v2", "oq_nax_lane_g128_v2")
_AFFINE_NAX_ALL_LAYOUTS = ("oq_nax_tiled_g64_v1", *_AFFINE_NAX_V2_LAYOUTS)

KNOWN_MODEL_TYPE_HINTS = frozenset(
    {
        "qwen3_5",
        "qwen3_5_text",
        "qwen3_5_moe",
        "qwen3_5_moe_text",
        "qwen3_6",
        "qwen3_6_text",
        "qwen3_6_moe",
        "qwen3_6_moe_text",
        "qwen3_8",
        "qwen3_8_text",
        "qwen3_8_moe",
        "qwen3_8_moe_text",
    }
)

_NON_TEXT_TENSOR_PREFIXES = (
    "visual.",
    "model.visual.",
    "vision_tower.",
    "model.vision_tower.",
    "vision_model.",
    "model.vision_model.",
    "multi_modal_projector.",
    "model.multi_modal_projector.",
    "mm_projector.",
    "model.mm_projector.",
    "aligner.",
    "model.aligner.",
    "audio_tower.",
    "model.audio_tower.",
    "talker.",
    "model.talker.",
)

_NON_TEXT_TENSOR_INFIXES = (
    ".visual.",
    ".vision_tower.",
    ".vision_model.",
    ".multi_modal_projector.",
    ".mm_projector.",
    ".aligner.",
    ".audio_tower.",
    ".talker.",
)

_MLA_CONFIG_KEYS = (
    "kv_lora_rank",
    "q_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
)

_MLA_TENSOR_MARKERS = (
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "q_a_proj",
    "q_b_proj",
)

_MAMBA_CONFIG_KEYS = (
    "mamba_d_state",
    "ssm_state_size",
    "mamba_d_conv",
)

_FP8_DTYPES = frozenset(
    {"F8_E4M3", "F8_E4M3FN", "F8_E4M3FNUZ", "F8_E5M2", "F8_E5M2FNUZ"}
)


def _is_non_text_tensor_name(name: str) -> bool:
    if name.startswith(_NON_TEXT_TENSOR_PREFIXES):
        return True
    probe = "." + name if not name.startswith(".") else name
    return any(infix in probe for infix in _NON_TEXT_TENSOR_INFIXES)


DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
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
    probe = name if name.startswith(".") else "." + name
    if name.endswith("embed_tokens"):
        return "embedding"
    if name.endswith("lm_head"):
        return "lm_head"
    if ".mlp.switch_mlp.gate_proj" in probe or ".mlp.experts.gate_proj" in probe:
        return "moe_expert_gate"
    if ".mlp.switch_mlp.up_proj" in probe or ".mlp.experts.up_proj" in probe:
        return "moe_expert_up"
    if ".mlp.switch_mlp.down_proj" in probe or ".mlp.experts.down_proj" in probe:
        return "moe_expert_down"
    if probe.endswith(".mlp.gate"):
        return "moe_router"
    if probe.endswith(".mlp.shared_expert_gate"):
        return "moe_shared_gate"
    if ".mlp.shared_expert.gate_proj" in probe or ".mlp.shared_experts.gate_proj" in probe:
        return "moe_shared_expert_gate"
    if ".mlp.shared_expert.up_proj" in probe or ".mlp.shared_experts.up_proj" in probe:
        return "moe_shared_expert_up"
    if ".mlp.shared_expert.down_proj" in probe or ".mlp.shared_experts.down_proj" in probe:
        return "moe_shared_expert_down"
    if ".mlp.gate_proj" in probe:
        return "mlp_gate"
    if ".mlp.up_proj" in probe:
        return "mlp_up"
    if ".mlp.down_proj" in probe:
        return "mlp_down"
    if ".linear_attn.in_proj_qkv" in probe:
        return "gdn_qkv"
    if ".linear_attn.in_proj_z" in probe:
        return "gdn_z"
    if ".linear_attn.in_proj_a" in probe:
        return "gdn_a"
    if ".linear_attn.in_proj_b" in probe:
        return "gdn_b"
    if ".linear_attn.out_proj" in probe:
        return "gdn_out"
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if f".self_attn.{projection}" in probe:
            return f"attention_{projection.removesuffix('_proj')}"
    return "linear"


def pi64_slot(off: int) -> int:
    """Involutive intra-row NAX quartet permutation pi_64(off)."""
    return ((off & 12) << 2) | ((off >> 2) & 12) | (off & 3)


def resolve_fwht64_enabled(
    artifact_version: int, fwht64: bool | None, profile: str = ""
) -> bool:
    if artifact_version not in (VERSION_V1, VERSION_V2):
        raise ValueError(f"unsupported artifact_version {artifact_version}")
    if artifact_version == VERSION_V1:
        if fwht64 is True:
            raise ValueError("--fwht64 requires --artifact-version 2")
        return False
    if profile == "dflash2_bf16":
        return False
    return True if fwht64 is None else bool(fwht64)


def is_fwht64_eligible(
    name: str,
    shape: tuple[int, ...] | list[int],
    format_id: str,
    *,
    profile: str,
    fwht64: bool,
    fwht64_o_proj: bool = True,
) -> bool:
    if not fwht64 or profile == "dflash2_bf16" or format_id.startswith("direct_"):
        return False
    if len(shape) < 2:
        return False
    in_features = int(shape[-1])
    if in_features < 64 or (in_features % 64) != 0:
        return False
    use = infer_weight_use(name)
    if use in ("mlp_down", "moe_expert_down", "moe_shared_expert_down"):
        return True
    if fwht64_o_proj and use == "attention_o":
        return True
    return False


def fwht64_f32_inplace(x: np.ndarray) -> np.ndarray:
    """Exact normalized block-orthogonal Sylvester FWHT-64 (H_64 = (1/8) * H_tilde_64).

    Applies 6 vectorized float32 butterfly passes along the last axis (K % 64 == 0).
    """
    if x.dtype != np.float32:
        raise ValueError(f"fwht64_f32_inplace requires float32, got {x.dtype}")
    if x.ndim == 0 or x.shape[-1] == 0 or (x.shape[-1] % 64) != 0:
        raise ValueError(
            f"fwht64_f32_inplace requires last axis divisible by 64, got {x.shape}"
        )
    orig_shape = x.shape
    if x.flags["C_CONTIGUOUS"]:
        work = x
    else:
        work = np.ascontiguousarray(x, dtype=np.float32)
    y = work.reshape(-1, 64)
    tmp = np.empty((y.shape[0], 32), dtype=np.float32)
    for h in (1, 2, 4, 8, 16, 32):
        blocks = 64 // (2 * h)
        v = y.reshape(-1, blocks, 2, h)
        a = tmp.reshape(-1, blocks, h)
        b = v[:, :, 1, :]
        np.copyto(a, v[:, :, 0, :])
        np.add(a, b, out=v[:, :, 0, :])
        np.subtract(a, b, out=v[:, :, 1, :])
    y *= np.float32(0.125)
    if work is not x:
        np.copyto(x, work.reshape(orig_shape))
    return x


def _permute_pi64_q8(arr: np.ndarray) -> np.ndarray:
    """Apply involutive pi_64 to [..., 64] uint8 (1 element/byte)."""
    prefix = arr.shape[:-1]
    return np.ascontiguousarray(
        arr.reshape(*prefix, 4, 4, 4).swapaxes(-3, -2).reshape(*prefix, 64)
    )


def _permute_pi64_nibbles(arr: np.ndarray) -> np.ndarray:
    """Apply involutive pi_64 to [..., 32] uint8 (2 elements/byte; 4 elems = 2 bytes)."""
    prefix = arr.shape[:-1]
    return np.ascontiguousarray(
        arr.reshape(*prefix, 4, 4, 2).swapaxes(-3, -2).reshape(*prefix, 32)
    )


def _permute_pi64_2bit(arr: np.ndarray) -> np.ndarray:
    """Apply involutive pi_64 to [..., 16] uint8 (4 elements/byte; 4 elems = 1 byte)."""
    prefix = arr.shape[:-1]
    return np.ascontiguousarray(
        arr.reshape(*prefix, 4, 4).swapaxes(-2, -1).reshape(*prefix, 16)
    )


def _permute_pi64_1bit(arr: np.ndarray) -> np.ndarray:
    """Apply involutive pi_64 to [..., 8] uint8 (8 elements/byte; 4 elems = 1 nibble)."""
    prefix = arr.shape[:-1]
    nib = np.empty((*prefix, 16), dtype=np.uint8)
    nib[..., 0::2] = arr & 0x0F
    nib[..., 1::2] = arr >> 4
    nib = nib.reshape(*prefix, 4, 4).swapaxes(-2, -1).reshape(*prefix, 16)
    return np.ascontiguousarray((nib[..., 0::2] | (nib[..., 1::2] << 4)).astype(np.uint8))


def _unpack_packed_affine_codes_u8(
    raw_u8: np.ndarray, logical_k: int, bits: int
) -> np.ndarray:
    """Fast vectorized unpack of MLX little-endian packed affine codes -> uint8 [rows, logical_k]."""
    rows = raw_u8.shape[0]
    if bits == 8:
        return np.ascontiguousarray(raw_u8[:, :logical_k])
    if bits == 4:
        out = np.empty((rows, logical_k), dtype=np.uint8)
        out[:, 0::2] = raw_u8[:, : logical_k // 2] & 0x0F
        out[:, 1::2] = raw_u8[:, : logical_k // 2] >> 4
        return out
    if bits == 5:
        # 8 codes = 40 bits = 5 bytes
        grp = raw_u8[:, : logical_k * 5 // 8].reshape(rows, logical_k // 8, 5).astype(np.uint64)
        w = (
            grp[:, :, 0]
            | (grp[:, :, 1] << np.uint64(8))
            | (grp[:, :, 2] << np.uint64(16))
            | (grp[:, :, 3] << np.uint64(24))
            | (grp[:, :, 4] << np.uint64(32))
        )
        out = np.empty((rows, logical_k // 8, 8), dtype=np.uint8)
        for i in range(8):
            out[:, :, i] = ((w >> np.uint64(5 * i)) & np.uint64(0x1F)).astype(np.uint8)
        return out.reshape(rows, logical_k)
    if bits == 6:
        # 4 codes = 24 bits = 3 bytes
        grp = raw_u8[:, : logical_k * 6 // 8].reshape(rows, logical_k // 4, 3).astype(np.uint32)
        w = grp[:, :, 0] | (grp[:, :, 1] << np.uint32(8)) | (grp[:, :, 2] << np.uint32(16))
        out = np.empty((rows, logical_k // 4, 4), dtype=np.uint8)
        for i in range(4):
            out[:, :, i] = ((w >> np.uint32(6 * i)) & np.uint32(0x3F)).astype(np.uint8)
        return out.reshape(rows, logical_k)
    if bits == 3:
        # 8 codes = 24 bits = 3 bytes
        grp = raw_u8[:, : logical_k * 3 // 8].reshape(rows, logical_k // 8, 3).astype(np.uint32)
        w = grp[:, :, 0] | (grp[:, :, 1] << np.uint32(8)) | (grp[:, :, 2] << np.uint32(16))
        out = np.empty((rows, logical_k // 8, 8), dtype=np.uint8)
        for i in range(8):
            out[:, :, i] = ((w >> np.uint32(3 * i)) & np.uint32(0x07)).astype(np.uint8)
        return out.reshape(rows, logical_k)
    raise ValueError(f"unsupported affine bits={bits}")


def _unpack_rowsplit_affine_codes_u8(
    base_u8: np.ndarray, high_u8: np.ndarray | None, logical_k: int, bits: int
) -> np.ndarray:
    """Fast vectorized unpack of .miv rowsplit affine planes -> uint8 [rows, logical_k]."""
    rows = base_u8.shape[0]
    if bits == 8:
        return np.ascontiguousarray(base_u8[:, :logical_k])
    if bits == 4:
        out = np.empty((rows, logical_k), dtype=np.uint8)
        b = base_u8[:, : logical_k // 2]
        out[:, 0::2] = b & 0x0F
        out[:, 1::2] = b >> 4
        return out
    if bits == 3:
        assert high_u8 is not None
        out = np.empty((rows, logical_k), dtype=np.uint8)
        lo = base_u8[:, : logical_k // 4]
        out[:, 0::4] = lo & 0x03
        out[:, 1::4] = (lo >> 2) & 0x03
        out[:, 2::4] = (lo >> 4) & 0x03
        out[:, 3::4] = (lo >> 6) & 0x03
        hi = high_u8[:, : logical_k // 8]
        for bit_i in range(8):
            out[:, bit_i::8] |= ((hi >> bit_i) & 1) << 2
        return out
    if bits == 5:
        assert high_u8 is not None
        out = np.empty((rows, logical_k), dtype=np.uint8)
        b = base_u8[:, : logical_k // 2]
        out[:, 0::2] = b & 0x0F
        out[:, 1::2] = b >> 4
        hi = high_u8[:, : logical_k // 8]
        for bit_i in range(8):
            out[:, bit_i::8] |= ((hi >> bit_i) & 1) << 4
        return out
    if bits == 6:
        assert high_u8 is not None
        out = np.empty((rows, logical_k), dtype=np.uint8)
        b = base_u8[:, : logical_k // 2]
        out[:, 0::2] = b & 0x0F
        out[:, 1::2] = b >> 4
        hi = high_u8[:, : logical_k // 4]
        out[:, 0::4] |= (hi & 0x03) << 4
        out[:, 1::4] |= ((hi >> 2) & 0x03) << 4
        out[:, 2::4] |= ((hi >> 4) & 0x03) << 4
        out[:, 3::4] |= ((hi >> 6) & 0x03) << 4
        return out
    raise ValueError(f"unsupported affine bits={bits}")


def _pack_rowsplit_affine_planes_from_codes(
    codes_u8: np.ndarray, bits: int, padded_k: int
) -> list[tuple[str, np.ndarray]]:
    """Pack uint8 [rows, logical_k] codes into .miv rowsplit code planes (padded to padded_k)."""
    rows, logical_k = codes_u8.shape
    if padded_k != logical_k:
        phys = np.zeros((rows, padded_k), dtype=np.uint8)
        phys[:, :logical_k] = codes_u8
        codes = phys
    else:
        codes = codes_u8
    if bits == 8:
        return [("code", np.ascontiguousarray(codes))]
    if bits == 4:
        base = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
        return [("base_code", np.ascontiguousarray(base))]
    if bits == 3:
        lo = codes & 0x03
        low2 = (
            lo[:, 0::4]
            | (lo[:, 1::4] << 2)
            | (lo[:, 2::4] << 4)
            | (lo[:, 3::4] << 6)
        )
        h = (codes >> 2) & 1
        high1 = (
            h[:, 0::8]
            | (h[:, 1::8] << 1)
            | (h[:, 2::8] << 2)
            | (h[:, 3::8] << 3)
            | (h[:, 4::8] << 4)
            | (h[:, 5::8] << 5)
            | (h[:, 6::8] << 6)
            | (h[:, 7::8] << 7)
        )
        return [
            ("low_2bit", np.ascontiguousarray(low2)),
            ("high_bit", np.ascontiguousarray(high1)),
        ]
    if bits == 5:
        base = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
        h = (codes >> 4) & 1
        high1 = (
            h[:, 0::8]
            | (h[:, 1::8] << 1)
            | (h[:, 2::8] << 2)
            | (h[:, 3::8] << 3)
            | (h[:, 4::8] << 4)
            | (h[:, 5::8] << 5)
            | (h[:, 6::8] << 6)
            | (h[:, 7::8] << 7)
        )
        return [
            ("base_code", np.ascontiguousarray(base)),
            ("high_bit", np.ascontiguousarray(high1)),
        ]
    if bits == 6:
        base = (codes[:, 0::2] & 0x0F) | ((codes[:, 1::2] & 0x0F) << 4)
        h = (codes >> 4) & 0x03
        high2 = (
            h[:, 0::4]
            | (h[:, 1::4] << 2)
            | (h[:, 2::4] << 4)
            | (h[:, 3::4] << 6)
        )
        return [
            ("base_code", np.ascontiguousarray(base)),
            ("high_2bit", np.ascontiguousarray(high2)),
        ]
    raise ValueError(f"unsupported affine bits={bits}")


def _pack_nax_affine_code_tiles(
    codes_u8: np.ndarray, bits: int, bn: int, lane_v2: bool
) -> list[tuple[str, np.ndarray]]:
    """Pack uint8 [padded_rows_chunk, logical_k] (padded_rows_chunk % bn == 0) into NAX code tiles."""
    rows, k = codes_u8.shape
    assert (rows % bn) == 0 and (k % 64) == 0
    groups = k // 64
    n_tiles = rows // bn
    if bits == 8:
        tiled = np.ascontiguousarray(
            codes_u8.reshape(n_tiles, bn, groups, 64).transpose(0, 2, 1, 3)
        )
        if lane_v2:
            tiled = _permute_pi64_q8(tiled)
        return [("nax_codes", tiled)]
    if bits == 3:
        lo = codes_u8 & 0x03
        low2 = (
            lo[:, 0::4]
            | (lo[:, 1::4] << 2)
            | (lo[:, 2::4] << 4)
            | (lo[:, 3::4] << 6)
        )
        tiled_l2 = np.ascontiguousarray(
            low2.reshape(n_tiles, bn, groups, 16).transpose(0, 2, 1, 3)
        )
        h = (codes_u8 >> 2) & 1
        high1 = (
            h[:, 0::8]
            | (h[:, 1::8] << 1)
            | (h[:, 2::8] << 2)
            | (h[:, 3::8] << 3)
            | (h[:, 4::8] << 4)
            | (h[:, 5::8] << 5)
            | (h[:, 6::8] << 6)
            | (h[:, 7::8] << 7)
        )
        tiled_h1 = np.ascontiguousarray(
            high1.reshape(n_tiles, bn, groups, 8).transpose(0, 2, 1, 3)
        )
        if lane_v2:
            tiled_l2 = _permute_pi64_2bit(tiled_l2)
            tiled_h1 = _permute_pi64_1bit(tiled_h1)
        return [("nax_codes", tiled_l2), ("nax_high1", tiled_h1)]
    # bits in (4, 5, 6)
    base = (codes_u8[:, 0::2] & 0x0F) | ((codes_u8[:, 1::2] & 0x0F) << 4)
    tiled_c = np.ascontiguousarray(
        base.reshape(n_tiles, bn, groups, 32).transpose(0, 2, 1, 3)
    )
    if lane_v2:
        tiled_c = _permute_pi64_nibbles(tiled_c)
    out = [("nax_codes", tiled_c)]
    if bits in (5, 6):
        mask = 1 if bits == 5 else 3
        h = (codes_u8 >> 4) & mask
        high2 = (
            h[:, 0::4]
            | (h[:, 1::4] << 2)
            | (h[:, 2::4] << 4)
            | (h[:, 3::4] << 6)
        )
        tiled_h2 = np.ascontiguousarray(
            high2.reshape(n_tiles, bn, groups, 16).transpose(0, 2, 1, 3)
        )
        if lane_v2:
            tiled_h2 = _permute_pi64_2bit(tiled_h2)
        out.append(("nax_high2", tiled_h2))
    return out


def quantize_oq_affine_f32(
    w_f32: np.ndarray, bits: int, group_size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Vectorized MSE-optimal BF16 (scale, bias) affine refit (shrink grid + 2 WLS steps).

    Returns (codes_u8 [rows, K], scales_bf16_u16 [rows, G], biases_bf16_u16 [rows, G], sse, sig_sq).
    """
    if bits not in (3, 4, 5, 6, 8) or group_size not in (64, 128):
        raise ValueError(f"unsupported oQ affine spec bits={bits}, group_size={group_size}")
    w2d = np.asarray(w_f32, dtype=np.float32)
    leading = w2d.shape[:-1]
    k = int(w2d.shape[-1])
    if k == 0 or (k % group_size) != 0:
        raise ValueError(f"K={k} must be divisible by group_size={group_size}")
    rows = product(leading)
    groups = k // group_size
    w_g = w2d.reshape(rows, groups, group_size)

    qmax = np.float32((1 << bits) - 1)
    inv_qmax = np.float32(1.0 / float((1 << bits) - 1))
    g_f32 = np.float32(group_size)
    inv_g = np.float32(1.0 / float(group_size))

    w_min = np.min(w_g, axis=2)
    w_max = np.max(w_g, axis=2)
    zero_span = w_max <= w_min
    center = np.float32(0.5) * (w_min + w_max)
    half_span = np.float32(0.5) * (w_max - w_min)
    s_w = np.sum(w_g, axis=2, dtype=np.float32)
    sig_sq = float(np.sum(w_g * w_g, dtype=np.float64))

    best_err = np.full((rows, groups), np.inf, dtype=np.float32)
    best_s = np.zeros((rows, groups), dtype=np.float32)
    best_b = np.zeros((rows, groups), dtype=np.float32)
    best_q = np.zeros((rows, groups, group_size), dtype=np.float32)

    def _snap_bf16(arr: np.ndarray) -> np.ndarray:
        bits_u32 = arr.astype(np.float32, copy=False).view(np.uint32)
        rounded = bits_u32 + np.uint32(0x7FFF) + ((bits_u32 >> np.uint32(16)) & np.uint32(1))
        return (rounded & np.uint32(0xFFFF0000)).view(np.float32)

    # 5-point clipping grid search over center-preserving shrinking factors
    for alpha in (1.00, 0.99, 0.98, 0.965, 0.95):
        hs = half_span * np.float32(alpha)
        b_c = _snap_bf16(center - hs)
        s_c = _snap_bf16((np.float32(2.0) * hs) * inv_qmax)
        pos = s_c > np.float32(0.0)
        inv_s = np.where(pos, np.float32(1.0) / np.where(pos, s_c, np.float32(1.0)), np.float32(0.0))
        q_c = np.clip(np.rint((w_g - b_c[:, :, None]) * inv_s[:, :, None]), np.float32(0.0), qmax)
        diff = s_c[:, :, None] * q_c + b_c[:, :, None] - w_g
        err = np.sum(diff * diff, axis=2, dtype=np.float32)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_s = np.where(better, s_c, best_s)
        best_b = np.where(better, b_c, best_b)
        best_q = np.where(better[:, :, None], q_c, best_q)

    # 2 alternating closed-form 2x2 WLS updates snapped to hardware BF16 (scale, bias)
    cur_s = best_s
    cur_b = best_b
    cur_q = best_q
    for _ in range(2):
        s_q = np.sum(cur_q, axis=2, dtype=np.float32)
        s_q2 = np.sum(cur_q * cur_q, axis=2, dtype=np.float32)
        s_qw = np.sum(cur_q * w_g, axis=2, dtype=np.float32)
        det = g_f32 * s_q2 - s_q * s_q
        valid = det > np.float32(1e-10)
        safe_det = np.where(valid, det, np.float32(1.0))
        new_s = np.where(valid, (g_f32 * s_qw - s_q * s_w) / safe_det, cur_s)
        new_s = np.maximum(new_s, np.float32(0.0))
        new_b = np.where(valid, (s_w - new_s * s_q) * inv_g, cur_b)
        s_snap = _snap_bf16(new_s)
        b_snap = _snap_bf16(new_b)
        pos = s_snap > np.float32(0.0)
        inv_s = np.where(pos, np.float32(1.0) / np.where(pos, s_snap, np.float32(1.0)), np.float32(0.0))
        q_next = np.clip(np.rint((w_g - b_snap[:, :, None]) * inv_s[:, :, None]), np.float32(0.0), qmax)
        diff = s_snap[:, :, None] * q_next + b_snap[:, :, None] - w_g
        err = np.sum(diff * diff, axis=2, dtype=np.float32)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_s = np.where(better, s_snap, best_s)
        best_b = np.where(better, b_snap, best_b)
        best_q = np.where(better[:, :, None], q_next, best_q)
        cur_s = s_snap
        cur_b = b_snap
        cur_q = q_next

    if np.any(zero_span):
        b_zero = _snap_bf16(w_min)
        best_s = np.where(zero_span, np.float32(0.0), best_s)
        best_b = np.where(zero_span, b_zero, best_b)
        best_q = np.where(zero_span[:, :, None], np.float32(0.0), best_q)
        diff_z = best_b[:, :, None] - w_g
        best_err = np.where(zero_span, np.sum(diff_z * diff_z, axis=2, dtype=np.float32), best_err)

    sse = float(np.sum(best_err, dtype=np.float64))
    sc_u16 = (best_s.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    bi_u16 = (best_b.view(np.uint32) >> np.uint32(16)).astype(np.uint16)
    return (
        best_q.astype(np.uint8).reshape(*leading, k),
        sc_u16.reshape(*leading, groups),
        bi_u16.reshape(*leading, groups),
        sse,
        sig_sq,
    )


def quantize_mxfp4_e2m1_f32(
    w_f32: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Vectorized 3-exponent (e0-1, e0, e0+1) MSE-optimal UE8M0 + E2M1 MXFP4 quantization.

    Returns (packed_u8 [..., K//2], scales_u8 [..., K//32], sse, sig_sq).
    """
    w_arr = np.asarray(w_f32, dtype=np.float32)
    leading = w_arr.shape[:-1]
    k = int(w_arr.shape[-1])
    if k == 0 or (k % 32) != 0:
        raise ValueError(f"MXFP4 quantization requires K divisible by 32, got K={k}")
    rows = product(leading)
    g32 = k // 32
    u = w_arr.reshape(rows, g32, 32)
    abs_u = np.abs(u)
    sign_u = (u < np.float32(0.0)).astype(np.uint8) << np.uint8(3)
    m = np.max(abs_u, axis=2)
    zero_mask = m == np.float32(0.0)
    sig_sq = float(np.sum(u * u, dtype=np.float64))

    safe_m = np.where(zero_mask, np.float32(6.0), np.maximum(m, np.float32(1e-38)))
    e0 = np.clip(
        np.floor(np.log2(safe_m * np.float32(1.0 / 6.0))).astype(np.int32) + 127,
        0,
        254,
    )

    best_err = np.full((rows, g32), np.inf, dtype=np.float32)
    best_e = np.zeros((rows, g32), dtype=np.uint8)
    best_mag = np.zeros((rows, g32, 32), dtype=np.uint8)

    for delta in (-1, 0, 1):
        e_cand = np.clip(e0 + delta, 0, 254).astype(np.uint8)
        inv_sc = _E8M0_INV_LUT[e_cand]
        sc = _E8M0_LUT[e_cand]
        a = abs_u * inv_sc[:, :, None]
        # Exact OCP E2M1 ties-to-even quantization thresholds
        mag = (
            (a > np.float32(0.25)).astype(np.uint8)
            + (a >= np.float32(0.75)).astype(np.uint8)
            + (a > np.float32(1.25)).astype(np.uint8)
            + (a >= np.float32(1.75)).astype(np.uint8)
            + (a > np.float32(2.50)).astype(np.uint8)
            + (a >= np.float32(3.50)).astype(np.uint8)
            + (a > np.float32(5.00)).astype(np.uint8)
        )
        recon_abs = _E2M1_MAG_LUT[mag] * sc[:, :, None]
        diff = recon_abs - abs_u
        err = np.sum(diff * diff, axis=2, dtype=np.float32)
        better = err < best_err
        best_err = np.where(better, err, best_err)
        best_e = np.where(better, e_cand, best_e)
        best_mag = np.where(better[:, :, None], mag, best_mag)

    if np.any(zero_mask):
        best_e = np.where(zero_mask, np.uint8(0), best_e)
        best_mag = np.where(zero_mask[:, :, None], np.uint8(0), best_mag)
        best_err = np.where(zero_mask, np.float32(0.0), best_err)

    codes = np.where(best_mag == np.uint8(0), np.uint8(0), sign_u | best_mag).reshape(rows, k)
    packed = np.ascontiguousarray(
        (codes[:, 0::2] & np.uint8(0x0F)) | ((codes[:, 1::2] & np.uint8(0x0F)) << np.uint8(4))
    )
    sse = float(np.sum(best_err, dtype=np.float64))
    return packed.reshape(*leading, k // 2), best_e.reshape(*leading, g32), sse, sig_sq


def _pwrite_all(fd: int, data: memoryview | bytes | bytearray, offset: int) -> None:
    view = memoryview(data).cast("B")
    pos = 0
    total = len(view)
    while pos < total:
        n = os.pwrite(fd, view[pos : min(pos + 16 * 1024 * 1024, total)], offset + pos)
        if n <= 0:
            raise OSError("short pwrite to artifact payload")
        pos += n


def _pread_all(fd: int, size: int, offset: int) -> bytes:
    out = bytearray(size)
    pos = 0
    while pos < size:
        chunk = os.pread(fd, min(16 * 1024 * 1024, size - pos), offset + pos)
        if not chunk:
            raise OSError("unexpected EOF during pread")
        out[pos : pos + len(chunk)] = chunk
        pos += len(chunk)
    return bytes(out)


def _sha256_fd_range(fd: int, offset: int, size: int) -> str:
    digest = hashlib.sha256()
    pos = 0
    while pos < size:
        chunk = os.pread(fd, min(16 * 1024 * 1024, size - pos), offset + pos)
        if not chunk:
            raise OSError("unexpected EOF hashing artifact object")
        digest.update(chunk)
        pos += len(chunk)
    return digest.hexdigest()


def _compute_v2_flags_and_features(
    objects: list[dict],
    *,
    fwht64_o_proj: bool = True,
) -> tuple[int, dict[str, Any]]:
    has_fwht = any(obj.get("input_transform") == "fwht64" for obj in objects)
    has_lane_v2 = any(
        obj.get("layout") in _AFFINE_NAX_V2_LAYOUTS for obj in objects
    )
    flags = (FLAG_FWHT64 if has_fwht else 0) | (FLAG_NAX_LANE_V2 if has_lane_v2 else 0)
    targets: list[str] = []
    if has_fwht:
        targets = ["mlp_down", "moe_expert_down", "moe_shared_mlp_down"]
        if fwht64_o_proj or any(
            obj.get("input_transform") == "fwht64"
            and infer_weight_use(str(obj.get("name", ""))) == "attention_o"
            for obj in objects
        ):
            targets.append("attention_o")
    features = {
        "fwht64_prerotated": has_fwht,
        "fwht64_targets": targets,
        "nax_lane_layout": "pi64_quartet_v2" if has_lane_v2 else "none",
    }
    return flags, features


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


def _read_safetensors_header_only(path: Path) -> dict[str, TensorInfo]:
    """Read ONLY the 8-byte length prefix + JSON header of a .safetensors shard (zero mmap/payload I/O)."""
    with path.open("rb") as fh:
        prefix = fh.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: safetensors prefix is missing")
        header_len = struct.unpack("<Q", prefix)[0]
        if not 0 < header_len <= 128 * 1024 * 1024:
            raise ValueError(f"{path}: invalid safetensors header length {header_len}")
        raw_header = fh.read(header_len)
        if len(raw_header) != header_len:
            raise ValueError(f"{path}: truncated safetensors header")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: invalid safetensors JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise ValueError(f"{path}: safetensors header must be a JSON object")
    data_start = 8 + header_len
    out: dict[str, TensorInfo] = {}
    for name, record in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(record, dict):
            raise ValueError(f"{path}: invalid tensor record {name}")
        dtype = str(record.get("dtype", ""))
        raw_shape = record.get("shape")
        raw_offsets = record.get("data_offsets", (0, 0))
        if not isinstance(raw_shape, list) or not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
            raise ValueError(f"{path}: invalid tensor metadata for {name}")
        shape = tuple(int(v) for v in raw_shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(f"{path}: negative tensor dimension for {name}")
        off0, off1 = int(raw_offsets[0]), int(raw_offsets[1])
        if off1 < off0:
            raise ValueError(f"{path}: invalid data_offsets for {name}")
        elem_bytes = DTYPE_BYTES.get(dtype, 0)
        elements = 1
        for dim in shape:
            elements *= dim
        nbytes = off1 - off0 if off1 > off0 else elements * elem_bytes
        out[str(name)] = TensorInfo(
            name=str(name),
            dtype=dtype,
            shape=shape,
            shard=path.name,
            data_offset=data_start + off0,
            nbytes=nbytes,
        )
    return out


def _read_checkpoint_headers_fast(
    model_dir: Path,
) -> tuple[dict[str, Any], dict[str, TensorInfo]]:
    """O(1) metadata-only checkpoint reader: config.json + index + safetensors JSON headers."""
    resolved = model_dir.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    cfg_path = resolved / "config.json"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"missing config.json in {model_dir}")
    config = json.loads(cfg_path.read_bytes())
    if not isinstance(config, dict):
        raise ValueError(f"{cfg_path}: config.json must be a JSON object")

    index_path = resolved / "model.safetensors.index.json"
    weight_map: dict[str, str] = {}
    if index_path.is_file():
        index_doc = json.loads(index_path.read_bytes())
        raw_wm = index_doc.get("weight_map") if isinstance(index_doc, dict) else None
        if not isinstance(raw_wm, dict) or not raw_wm:
            raise ValueError(f"{index_path}: weight_map must be a non-empty object")
        weight_map = {str(k): str(v) for k, v in raw_wm.items()}
        shard_names = sorted(set(weight_map.values()))
    else:
        single = resolved / "model.safetensors"
        if single.is_file():
            shard_names = [single.name]
        else:
            shard_names = sorted(
                p.name
                for p in resolved.glob("*.safetensors")
                if p.is_file() and not p.name.startswith("adapter_model")
            )
            if not shard_names:
                raise FileNotFoundError(
                    f"no model.safetensors or model.safetensors.index.json in {model_dir}"
                )

    shard_headers: dict[str, dict[str, TensorInfo]] = {}
    for shard_name in shard_names:
        if Path(shard_name).name != shard_name:
            raise ValueError(f"unsafe shard name: {shard_name}")
        shard_path = (resolved / shard_name).resolve()
        if shard_path.parent != resolved or not shard_path.is_file():
            raise FileNotFoundError(f"missing or unsafe safetensors shard: {shard_name}")
        shard_headers[shard_name] = _read_safetensors_header_only(shard_path)

    tensors: dict[str, TensorInfo] = {}
    if weight_map:
        for tname, sname in weight_map.items():
            sh = shard_headers.get(sname)
            if sh is None or tname not in sh:
                raise ValueError(f"safetensors index points to missing tensor {tname} in {sname}")
            tensors[tname] = sh[tname]
    else:
        for sname in shard_names:
            for tname, tinfo in shard_headers[sname].items():
                if tname in tensors:
                    raise ValueError(f"duplicate tensor {tname} across safetensors shards")
                tensors[tname] = tinfo
    return config, tensors


def _is_strict_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _opt_int(cfg: dict[str, Any], *keys: str, default: int = 0) -> int:
    for key in keys:
        val = cfg.get(key)
        if _is_strict_int(val):
            return int(val)
    return default


def _is_dflash2_config(raw_config: dict[str, Any]) -> bool:
    archs = raw_config.get("architectures") or ()
    if isinstance(archs, list | tuple):
        for arch in archs:
            s = str(arch).strip()
            if s == "DFlash2DraftModel" or ("dflash" in s.lower() and "draft" in s.lower()):
                return True
    return isinstance(raw_config.get("dflash_config"), dict)


def _normalize_model_config(raw_config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Normalize nested text_config (VLM/hybrid) or flat root config.json into a unified text config."""
    if _is_dflash2_config(raw_config):
        return dict(raw_config), "dflash2"
    nested = raw_config.get("text_config")
    if isinstance(nested, dict) and nested:
        merged = dict(nested)
        for fallback_key in (
            "tie_word_embeddings",
            "quantization",
            "quantization_config",
            "architectures",
            "dtype",
            "torch_dtype",
            "eos_token_id",
            "bos_token_id",
            "rope_theta",
            "partial_rotary_factor",
        ):
            if fallback_key not in merged and fallback_key in raw_config:
                merged[fallback_key] = raw_config[fallback_key]
        return merged, "nested_text_config"
    return dict(raw_config), "flat_root"


def _has_full_model_arch_config(raw_config: dict[str, Any]) -> bool:
    if _is_dflash2_config(raw_config):
        return (
            isinstance(raw_config.get("dflash_config"), dict)
            or "num_hidden_layers" in raw_config
        )
    text_cfg, _ = _normalize_model_config(raw_config)
    return "num_hidden_layers" in text_cfg


def _expected_module_in_features(module: str, text_cfg: dict[str, Any]) -> int:
    """Return structurally expected input dimension K for known text modules, or 0 if unknown."""
    if _is_mtp_object_name(module):
        return 0
    h = _opt_int(text_cfg, "hidden_size")
    if h <= 0:
        return 0
    use = infer_weight_use(module)
    if use in (
        "embedding",
        "lm_head",
        "attention_q",
        "attention_k",
        "attention_v",
        "gdn_qkv",
        "gdn_z",
        "gdn_a",
        "gdn_b",
        "mlp_gate",
        "mlp_up",
        "moe_router",
        "moe_shared_gate",
        "moe_expert_gate",
        "moe_expert_up",
        "moe_shared_expert_gate",
        "moe_shared_expert_up",
    ):
        return h
    if use == "attention_o":
        q_heads = _opt_int(text_cfg, "num_attention_heads")
        head_dim = _opt_int(
            text_cfg, "head_dim", default=(h // q_heads if q_heads > 0 and h % q_heads == 0 else 0)
        )
        return q_heads * head_dim if q_heads > 0 and head_dim > 0 else 0
    if use == "gdn_out":
        v_heads = _opt_int(text_cfg, "linear_num_value_heads")
        v_dim = _opt_int(text_cfg, "linear_value_head_dim")
        return v_heads * v_dim if v_heads > 0 and v_dim > 0 else 0
    if use == "mlp_down":
        inter = _opt_int(text_cfg, "intermediate_size", "moe_intermediate_size")
        return inter if inter > 0 else 0
    if use == "moe_expert_down":
        moe_inter = _opt_int(text_cfg, "moe_intermediate_size", "intermediate_size")
        return moe_inter if moe_inter > 0 else 0
    if use == "moe_shared_expert_down":
        sh_inter = _opt_int(text_cfg, "shared_expert_intermediate_size")
        return sh_inter if sh_inter > 0 else 0
    return 0


def _lookup_explicit_module_quant(table: dict[str, Any], module: str) -> dict[str, Any] | None:
    candidates = [module]
    if module.startswith("language_model."):
        candidates.append(module.removeprefix("language_model."))
    else:
        candidates.append("language_model." + module)
    if module.startswith("model."):
        candidates.append(module.removeprefix("model."))
    for key in candidates:
        entry = table.get(key)
        if isinstance(entry, dict):
            return entry
    return None


def _infer_module_affine_spec(
    raw_config: dict[str, Any],
    text_cfg: dict[str, Any],
    module: str,
    weight: TensorInfo,
    scales: TensorInfo,
    biases: TensorInfo,
) -> tuple[int, int, int]:
    """Infer (bits, group_size, logical_k) per module from tensor shapes + config overrides.

    Does NOT validate whether (bits, group_size) are supported by miv so callers can report
    exact unsupported group_size (e.g. gs=32) or bits (e.g. bits=2) blockers per tensor.
    """
    if weight.dtype != "U32" or scales.dtype not in ("BF16", "F16") or biases.dtype not in ("BF16", "F16"):
        raise ValueError(
            f"{module}: affine tensors have unexpected dtypes "
            f"(weight={weight.dtype}, scales={scales.dtype}, biases={biases.dtype})"
        )
    if (
        len(weight.shape) < 2
        or scales.shape != biases.shape
        or len(scales.shape) != len(weight.shape)
        or weight.shape[:-1] != scales.shape[:-1]
    ):
        raise ValueError(
            f"{module}: affine tensor shapes disagree "
            f"(weight={list(weight.shape)}, scales={list(scales.shape)}, biases={list(biases.shape)})"
        )
    w_cols = int(weight.shape[-1])
    groups = int(scales.shape[-1])
    if w_cols <= 0 or groups <= 0:
        raise ValueError(f"{module}: affine tensor dimensions must be positive")
    total_bits = w_cols * 32
    expected_k = (
        _expected_module_in_features(module, text_cfg)
        if "num_hidden_layers" in text_cfg
        else 0
    )

    qtable = _quantization_table(raw_config)
    explicit_entry = _lookup_explicit_module_quant(qtable, module)
    if explicit_entry is not None:
        eb = _opt_int(explicit_entry, "bits", default=_opt_int(qtable, "bits"))
        egs = _opt_int(explicit_entry, "group_size", default=_opt_int(qtable, "group_size"))
        if eb > 0 and egs > 0 and groups * egs * eb == total_bits:
            if expected_k <= 0 or groups * egs == expected_k:
                return eb, egs, groups * egs

    if expected_k > 0 and (expected_k % groups) == 0 and (total_bits % expected_k) == 0:
        gs = expected_k // groups
        bits = total_bits // expected_k
        if bits > 0 and gs > 0:
            return bits, gs, expected_k

    spec = quant_spec(raw_config, module)
    gb = int(spec.get("bits") or 0)
    ggs = int(spec.get("group_size") or 0)
    if gb > 0 and ggs > 0 and groups * ggs * gb == total_bits:
        return gb, ggs, groups * ggs

    fallback_k = _expected_module_in_features(module, text_cfg)
    if fallback_k > 0 and (fallback_k % groups) == 0 and (total_bits % fallback_k) == 0:
        gs = fallback_k // groups
        bits = total_bits // fallback_k
        if bits > 0 and gs > 0:
            return bits, gs, fallback_k

    if (total_bits % groups) == 0:
        bg = total_bits // groups
        for cand_gs in (64, 128):
            if (bg % cand_gs) == 0 and (bg // cand_gs) in (3, 4, 5, 6, 8):
                bits = bg // cand_gs
                return bits, cand_gs, groups * cand_gs
        for cand_gs in (32, 16, 256):
            if (bg % cand_gs) == 0 and (bg // cand_gs) in (2, 3, 4, 5, 6, 8):
                bits = bg // cand_gs
                return bits, cand_gs, groups * cand_gs

    raise ValueError(
        f"{module}: packed weight shape {list(weight.shape)} disagrees with scales shape {list(scales.shape)}"
    )


def _plan_repack_layers(
    raw_config: dict[str, Any],
    text_cfg: dict[str, Any],
    tensors: dict[str, TensorInfo],
    *,
    is_dflash2: bool,
    nax_pretile: bool,
    artifact_version: int,
) -> dict[str, Any]:
    """Layer 2: Per-tensor quantization & layout repack planner."""
    skipped_tensors: list[str] = []
    tensor_cat_counts: dict[str, int] = {
        "nax_pretile": 0,
        "rowsplit_or_direct": 0,
        "companion_mtp": 0,
        "skip_non_text": 0,
        "incompatible_blocker": 0,
    }
    cat_counts: dict[str, int] = {
        "nax_pretile": 0,
        "rowsplit_or_direct": 0,
        "companion_mtp": 0,
        "skip_non_text": 0,
        "incompatible_blocker": 0,
    }
    fmt_hist: Counter[str] = Counter()
    layout_hist: Counter[str] = Counter()
    entries: list[dict[str, Any]] = []
    first_blocker: dict[str, Any] | None = None

    def _record_blocker(tname: str, shape: tuple[int, ...], code: str, reason: str, src_names: list[str]) -> None:
        nonlocal first_blocker
        cat_counts["incompatible_blocker"] += 1
        tensor_cat_counts["incompatible_blocker"] += len(src_names)
        if first_blocker is None:
            first_blocker = {
                "tensor": tname,
                "shape": list(shape),
                "blocker_code": code,
                "reason": reason,
            }
        entries.append(
            {
                "name": tname,
                "category": "incompatible_blocker",
                "blocker_code": code,
                "reason": reason,
                "shape": list(shape),
                "source_tensors": src_names,
            }
        )

    active_tensors: dict[str, TensorInfo] = {}
    for tname in sorted(tensors):
        if _is_non_text_tensor_name(tname):
            skipped_tensors.append(tname)
            tensor_cat_counts["skip_non_text"] += 1
            cat_counts["skip_non_text"] += 1
        else:
            active_tensors[tname] = tensors[tname]

    if is_dflash2:
        for tname in sorted(active_tensors):
            info = active_tensors[tname]
            if info.dtype != "BF16":
                _record_blocker(
                    tname,
                    info.shape,
                    "dflash2_non_bf16_tensor",
                    f"DFlash2 draft tensor '{tname}' (shape {list(info.shape)}) has dtype '{info.dtype}'; DFlash2 requires unquantized BF16 tensors",
                    [tname],
                )
                continue
            fmt = "direct_bf16_v1"
            lay = "contiguous_le_v1"
            cat_counts["rowsplit_or_direct"] += 1
            tensor_cat_counts["rowsplit_or_direct"] += 1
            fmt_hist[fmt] += 1
            layout_hist[lay] += 1
            entries.append(
                {
                    "name": f"dflash2.{tname}",
                    "category": "rowsplit_or_direct",
                    "format": fmt,
                    "layout": lay,
                    "shape": list(info.shape),
                    "source_tensors": [tname],
                }
            )
        eligible = first_blocker is None and len(active_tensors) > 0
        blocker_code = first_blocker["blocker_code"] if first_blocker else (None if eligible else "empty_checkpoint")
        reason = first_blocker["reason"] if first_blocker else ("ok" if eligible else "DFlash2 checkpoint contains no tensors")
        return {
            "eligible": eligible,
            "blocker_code": blocker_code,
            "reason": reason,
            "first_blocker": first_blocker,
            "nax_pretile": nax_pretile,
            "artifact_version": artifact_version,
            "total_source_tensors": len(tensors),
            "category_counts": cat_counts,
            "tensor_category_counts": tensor_cat_counts,
            "format_histogram": dict(sorted(fmt_hist.items())),
            "layout_histogram": dict(sorted(layout_hist.items())),
            "skipped_tensors": skipped_tensors,
            "entries": entries,
        }

    qtable = _quantization_table(raw_config)
    quant_method = str(
        qtable.get("quant_method") or qtable.get("format") or ""
    ).strip().lower()
    has_any_g_idx = any(n.endswith(".g_idx") for n in active_tensors)
    has_any_qzeros = any(n.endswith(".qzeros") for n in active_tensors)

    consumed: set[str] = set()

    # Pre-scan for explicit incompatible quantization tensors (AWQ, GPTQ, FP8, bitsandbytes, additive .bias)
    for tname in sorted(active_tensors):
        if tname in consumed:
            continue
        info = active_tensors[tname]
        if tname.endswith(".bias") and not tname.endswith(".dt_bias"):
            consumed.add(tname)
            _record_blocker(
                tname,
                info.shape,
                "unsupported_linear_bias",
                f"Unsupported additive linear/norm bias tensor '{tname}' (shape {list(info.shape)}); miv text runtime requires bias-free projections",
                [tname],
            )
            continue
        if tname.endswith((".qweight", ".qzeros", ".g_idx")):
            mod_base = (
                tname.removesuffix(".qweight")
                .removesuffix(".qzeros")
                .removesuffix(".g_idx")
            )
            related = [
                cand
                for suffix in (".qweight", ".qzeros", ".scales", ".g_idx")
                if (cand := mod_base + suffix) in active_tensors and cand not in consumed
            ]
            consumed.update(related or [tname])
            is_gptq = (
                tname.endswith(".g_idx")
                or (mod_base + ".g_idx") in active_tensors
                or quant_method == "gptq"
                or (has_any_g_idx and not has_any_qzeros)
            )
            code = "unsupported_quant_gptq" if is_gptq else "unsupported_quant_awq"
            label = "GPTQ" if is_gptq else "AWQ"
            primary_t = (mod_base + ".qweight") if (mod_base + ".qweight") in active_tensors else tname
            primary_info = active_tensors[primary_t]
            _record_blocker(
                primary_t,
                primary_info.shape,
                code,
                f"Unsupported {label} quantization on tensor '{primary_t}' (shape {list(primary_info.shape)}): {label} packed weights are not supported by miv",
                related or [tname],
            )
            continue
        if (
            tname.endswith((".weight_scale_inv", ".activation_scale", ".input_scale"))
            or info.dtype in _FP8_DTYPES
        ):
            mod_base = (
                tname.removesuffix(".weight_scale_inv")
                .removesuffix(".activation_scale")
                .removesuffix(".input_scale")
                .removesuffix(".weight")
            )
            related = [
                cand
                for suffix in (".weight", ".weight_scale_inv", ".activation_scale", ".input_scale")
                if (cand := mod_base + suffix) in active_tensors and cand not in consumed
            ]
            consumed.update(related or [tname])
            primary_t = (
                (mod_base + ".weight")
                if (mod_base + ".weight") in active_tensors
                and active_tensors[mod_base + ".weight"].dtype in _FP8_DTYPES
                else tname
            )
            primary_info = active_tensors[primary_t]
            _record_blocker(
                primary_t,
                primary_info.shape,
                "unsupported_quant_fp8",
                f"Unsupported FP8 quantization on tensor '{primary_t}' (shape {list(primary_info.shape)}, dtype {primary_info.dtype})",
                related or [tname],
            )
            continue
        if (
            tname.endswith((".absmax", ".quant_map", ".nested_absmax", ".nested_quant_map", ".SCB", ".weight_format"))
            or ".quant_state." in tname
        ):
            consumed.add(tname)
            _record_blocker(
                tname,
                info.shape,
                "unsupported_quant_bitsandbytes",
                f"Unsupported bitsandbytes quantization metadata tensor '{tname}' (shape {list(info.shape)})",
                [tname],
            )
            continue

    # Compressed MXFP4 modules (.weight_packed + .weight_scale [+ .weight_shape])
    for packed_name in sorted(
        n for n in active_tensors if n.endswith(".weight_packed") and n not in consumed
    ):
        module = packed_name[: -len(".weight_packed")]
        scale_name = module + ".weight_scale"
        packed = active_tensors[packed_name]
        src_names = [packed_name]
        if scale_name in active_tensors:
            src_names.append(scale_name)
        shape_name = module + ".weight_shape"
        if shape_name in active_tensors:
            src_names.append(shape_name)
        consumed.update(src_names)
        if scale_name not in active_tensors:
            _record_blocker(
                packed_name,
                packed.shape,
                "invalid_compressed_mxfp4",
                f"Compressed MXFP4 tensor '{packed_name}' (shape {list(packed.shape)}) is missing '{scale_name}'",
                src_names,
            )
            continue
        scales = active_tensors[scale_name]
        if packed.dtype not in ("U8", "I8") or scales.dtype not in ("U8", "F8_E8M0") or len(packed.shape) not in (2, 3):
            _record_blocker(
                packed_name,
                packed.shape,
                "invalid_compressed_mxfp4",
                f"Unsupported compressed MXFP4 dtype/rank on '{packed_name}' (shape {list(packed.shape)}, dtype {packed.dtype})",
                src_names,
            )
            continue
        logical_k = int(packed.shape[-1]) * 2
        if (logical_k % 32) != 0 or scales.shape != (*packed.shape[:-1], logical_k // 32):
            _record_blocker(
                packed_name,
                packed.shape,
                "invalid_compressed_mxfp4",
                f"Compressed MXFP4 scale shape {list(scales.shape)} disagrees with '{packed_name}' shape {list(packed.shape)}",
                src_names,
            )
            continue
        logical_shape = (*packed.shape[:-1], logical_k)
        fmt = "mxfp4_e2m1_g32_e8m0_v1"
        base_lay = "mx_rowsplit_k128_v1" if len(logical_shape) == 2 else "mx_expert_rowsplit_k128_v1"
        use = infer_weight_use(module)
        is_mtp = _is_mtp_object_name(module)
        target_lay, did_pt = _target_layout_for_object(
            module,
            fmt,
            base_lay,
            logical_shape,
            nax_pretile=nax_pretile,
            artifact_version=artifact_version,
            is_embedding=(use == "embedding"),
        )
        moe_nax_ok = len(logical_shape) == 3 and nax_pretile and logical_k >= 128 and (logical_k % 128) == 0
        cat = "companion_mtp" if is_mtp else ("nax_pretile" if (did_pt or moe_nax_ok) else "rowsplit_or_direct")
        cat_counts[cat] += 1
        tensor_cat_counts[cat] += len(src_names)
        fmt_hist[fmt] += 1
        layout_hist[target_lay] += 1
        entries.append(
            {
                "name": module,
                "category": cat,
                "format": fmt,
                "layout": target_lay,
                "shape": list(logical_shape),
                "source_tensors": src_names,
            }
        )

    # Quantized .weight + .scales [+ .biases] modules
    for weight_name in sorted(
        n for n in active_tensors if n.endswith(".weight") and n not in consumed
    ):
        module = weight_name[: -len(".weight")]
        scales_name = module + ".scales"
        biases_name = module + ".biases"
        if scales_name not in active_tensors and biases_name not in active_tensors:
            continue
        weight = active_tensors[weight_name]
        if scales_name not in active_tensors:
            src_names = [weight_name, biases_name]
            consumed.update(src_names)
            _record_blocker(
                weight_name,
                weight.shape,
                "orphaned_quant_scales",
                f"Quantized module '{module}' has '{biases_name}' without '{scales_name}'",
                src_names,
            )
            continue
        scales = active_tensors[scales_name]
        if biases_name in active_tensors:
            biases = active_tensors[biases_name]
            src_names = [weight_name, scales_name, biases_name]
            consumed.update(src_names)
            try:
                bits, gs, logical_k = _infer_module_affine_spec(
                    raw_config, text_cfg, module, weight, scales, biases
                )
            except ValueError as exc:
                _record_blocker(
                    weight_name,
                    weight.shape,
                    "invalid_affine_quant_geometry",
                    f"Invalid affine quantization on '{weight_name}' (shape {list(weight.shape)}): {exc}",
                    src_names,
                )
                continue
            if gs not in (64, 128):
                _record_blocker(
                    weight_name,
                    weight.shape,
                    "unsupported_quant_affine_group_size",
                    f"Unsupported affine quantization group_size={gs} (bits={bits}) on '{weight_name}' (shape {list(weight.shape)}); miv requires group_size in (64, 128)",
                    src_names,
                )
                continue
            if bits not in (3, 4, 5, 6, 8):
                _record_blocker(
                    weight_name,
                    weight.shape,
                    "unsupported_quant_affine_bits",
                    f"Unsupported affine quantization bits={bits} (group_size={gs}) on '{weight_name}' (shape {list(weight.shape)}); miv requires bits in (3, 4, 5, 6, 8)",
                    src_names,
                )
                continue
            if len(scales.shape) not in (2, 3):
                _record_blocker(
                    weight_name,
                    weight.shape,
                    "invalid_affine_quant_rank",
                    f"Unsupported affine rank={len(scales.shape)} on '{weight_name}' (shape {list(weight.shape)}); expected 2D or 3D",
                    src_names,
                )
                continue
            logical_shape = (*scales.shape[:-1], logical_k)
            fmt = f"oq_affine_q{bits}_g{gs}_bf16_v1"
            base_lay = "oq_rowsplit_k128_v1" if len(logical_shape) == 2 else "oq_expert_rowsplit_k128_v1"
            use = infer_weight_use(module)
            is_mtp = _is_mtp_object_name(module)
            target_lay, did_pt = _target_layout_for_object(
                module,
                fmt,
                base_lay,
                logical_shape,
                nax_pretile=nax_pretile,
                artifact_version=artifact_version,
                is_embedding=(use == "embedding"),
            )
            moe_nax_ok = (
                len(logical_shape) == 3
                and nax_pretile
                and logical_k >= 128
                and (logical_k % gs) == 0
            )
            cat = "companion_mtp" if is_mtp else ("nax_pretile" if (did_pt or moe_nax_ok) else "rowsplit_or_direct")
            cat_counts[cat] += 1
            tensor_cat_counts[cat] += len(src_names)
            fmt_hist[fmt] += 1
            layout_hist[target_lay] += 1
            entries.append(
                {
                    "name": module,
                    "category": cat,
                    "format": fmt,
                    "layout": target_lay,
                    "shape": list(logical_shape),
                    "bits": bits,
                    "group_size": gs,
                    "source_tensors": src_names,
                }
            )
        else:
            src_names = [weight_name, scales_name]
            consumed.update(src_names)
            if weight.dtype != "U32" or scales.dtype not in ("U8", "F8_E8M0"):
                code = (
                    "unsupported_quant_bitsandbytes"
                    if quant_method == "bitsandbytes"
                    else "unsupported_quant_encoding"
                )
                _record_blocker(
                    weight_name,
                    weight.shape,
                    code,
                    f"Unsupported quantized tensor pair '{weight_name}' (dtype={weight.dtype}, shape={list(weight.shape)}) + '{scales_name}' (dtype={scales.dtype}, shape={list(scales.shape)}) without affine .biases",
                    src_names,
                )
                continue
            if len(weight.shape) not in (2, 3) or len(scales.shape) != len(weight.shape):
                _record_blocker(
                    weight_name,
                    weight.shape,
                    "invalid_mxfp4_shape",
                    f"Unsupported MXFP4 tensor rank on '{weight_name}' (shape {list(weight.shape)})",
                    src_names,
                )
                continue
            logical_k = int(scales.shape[-1]) * 32
            if weight.shape != (*scales.shape[:-1], logical_k // 8):
                _record_blocker(
                    weight_name,
                    weight.shape,
                    "invalid_mxfp4_shape",
                    f"MXFP4 weight '{weight_name}' shape {list(weight.shape)} disagrees with scales shape {list(scales.shape)}",
                    src_names,
                )
                continue
            logical_shape = (*scales.shape[:-1], logical_k)
            fmt = "mxfp4_e2m1_g32_e8m0_v1"
            base_lay = "mx_rowsplit_k128_v1" if len(logical_shape) == 2 else "mx_expert_rowsplit_k128_v1"
            use = infer_weight_use(module)
            is_mtp = _is_mtp_object_name(module)
            target_lay, did_pt = _target_layout_for_object(
                module,
                fmt,
                base_lay,
                logical_shape,
                nax_pretile=nax_pretile,
                artifact_version=artifact_version,
                is_embedding=(use == "embedding"),
            )
            moe_nax_ok = len(logical_shape) == 3 and nax_pretile and logical_k >= 128 and (logical_k % 128) == 0
            cat = "companion_mtp" if is_mtp else ("nax_pretile" if (did_pt or moe_nax_ok) else "rowsplit_or_direct")
            cat_counts[cat] += 1
            tensor_cat_counts[cat] += len(src_names)
            fmt_hist[fmt] += 1
            layout_hist[target_lay] += 1
            entries.append(
                {
                    "name": module,
                    "category": cat,
                    "format": fmt,
                    "layout": target_lay,
                    "shape": list(logical_shape),
                    "bits": 4,
                    "group_size": 32,
                    "source_tensors": src_names,
                }
            )

    # Remaining direct / orphaned tensors
    for tname in sorted(set(active_tensors) - consumed):
        info = active_tensors[tname]
        if tname.endswith((".scales", ".biases", ".weight_scale", ".weight_shape")):
            _record_blocker(
                tname,
                info.shape,
                "orphaned_quant_tensor",
                f"Orphaned quantization metadata tensor '{tname}' (shape {list(info.shape)}) has no matching weight tensor",
                [tname],
            )
            continue
        mod_probe = tname[: -len(".weight")] if tname.endswith(".weight") else tname
        use = infer_weight_use(mod_probe)
        if len(info.shape) == 3 and use in ("moe_expert_gate", "moe_expert_up", "moe_expert_down"):
            _record_blocker(
                tname,
                info.shape,
                "unsupported_unquantized_moe_experts",
                f"Unsupported unquantized 3D MoE expert tensor '{tname}' (shape {list(info.shape)}, dtype {info.dtype}); miv requires quantized (oq_affine or mxfp4) MoE experts",
                [tname],
            )
            continue
        if ".mlp.experts." in ("." + tname) and any(
            seg.isdigit() for seg in tname.split(".mlp.experts.", 1)[1].split(".")
        ):
            _record_blocker(
                tname,
                info.shape,
                "unsupported_unstacked_moe_experts",
                f"Unsupported unstacked per-expert MoE tensor '{tname}' (shape {list(info.shape)}); miv requires stacked 3D switch_mlp/experts tensors",
                [tname],
            )
            continue
        if info.dtype not in ("BF16", "F16", "F32"):
            _record_blocker(
                tname,
                info.shape,
                "unsupported_tensor_dtype",
                f"Unsupported unquantized tensor '{tname}' (shape {list(info.shape)}, dtype '{info.dtype}'); expected BF16/F16/F32 or supported quantized format",
                [tname],
            )
            continue
        fmt = "direct_bf16_v1"
        lay = "contiguous_le_v1"
        cat = "companion_mtp" if _is_mtp_object_name(tname) else "rowsplit_or_direct"
        cat_counts[cat] += 1
        tensor_cat_counts[cat] += 1
        fmt_hist[fmt] += 1
        layout_hist[lay] += 1
        entries.append(
            {
                "name": tname,
                "category": cat,
                "format": fmt,
                "layout": lay,
                "shape": list(info.shape),
                "source_dtype": info.dtype,
                "source_tensors": [tname],
            }
        )

    if first_blocker is None and quant_method in (
        "awq",
        "gptq",
        "fp8",
        "bitsandbytes",
        "gguf",
        "aqlm",
        "quip",
        "hqq",
        "exl2",
        "marlin",
    ):
        sample_name = next(iter(sorted(active_tensors)), "<config>")
        sample_shape = list(active_tensors[sample_name].shape) if sample_name in active_tensors else []
        _record_blocker(
            sample_name,
            tuple(sample_shape),
            f"unsupported_quant_{quant_method}",
            f"Unsupported quantization method '{quant_method}' declared in config.json (first tensor '{sample_name}', shape {sample_shape})",
            [sample_name] if sample_name in active_tensors else [],
        )

    if first_blocker is None and not active_tensors:
        first_blocker = {
            "tensor": "<none>",
            "shape": [],
            "blocker_code": "empty_text_tensors",
            "reason": "Checkpoint contains no text model tensors",
        }

    entries.sort(key=lambda item: str(item["name"]))
    eligible = first_blocker is None
    return {
        "eligible": eligible,
        "blocker_code": first_blocker["blocker_code"] if first_blocker else None,
        "reason": first_blocker["reason"] if first_blocker else "ok",
        "first_blocker": first_blocker,
        "nax_pretile": nax_pretile,
        "artifact_version": artifact_version,
        "total_source_tensors": len(tensors),
        "category_counts": cat_counts,
        "tensor_category_counts": tensor_cat_counts,
        "format_histogram": dict(sorted(fmt_hist.items())),
        "layout_histogram": dict(sorted(layout_hist.items())),
        "skipped_tensors": skipped_tensors,
        "entries": entries,
    }


def _evaluate_graph_capability_gate(
    raw_config: dict[str, Any],
    text_cfg: dict[str, Any],
    config_layout: str,
    tensors: dict[str, TensorInfo],
    *,
    target_config: dict[str, Any] | None = None,
    require_complete_model: bool = True,
) -> dict[str, Any]:
    """Layer 1: Structural Graph Capability Gate (architecture, topology, geometry & completeness)."""
    root_mt = str(raw_config.get("model_type") or "").strip()
    text_mt = str((raw_config.get("text_config") or {}).get("model_type") or "").strip()
    is_dflash2 = config_layout == "dflash2"
    hint_matched = (
        True
        if is_dflash2 and root_mt.lower() == "qwen3"
        else (
            root_mt.lower() in KNOWN_MODEL_TYPE_HINTS
            or text_mt.lower() in KNOWN_MODEL_TYPE_HINTS
        )
    )
    active_names = [n for n in sorted(tensors) if not _is_non_text_tensor_name(n)]
    text_names = [n for n in active_names if not _is_mtp_object_name(n)]

    base_gate: dict[str, Any] = {
        "eligible": False,
        "blocker_code": None,
        "reason": "",
        "first_blocker": None,
        "config_layout": config_layout,
        "model_type": root_mt,
        "text_model_type": text_mt,
        "model_type_hint_matched": hint_matched,
        "topology": "dflash2_draft" if is_dflash2 else "unknown",
        "num_hidden_layers": _opt_int(text_cfg, "num_hidden_layers"),
        "full_attention_layers": 0,
        "linear_attention_layers": 0,
        "hidden_size": _opt_int(text_cfg, "hidden_size"),
        "intermediate_size": _opt_int(text_cfg, "intermediate_size", "moe_intermediate_size"),
        "vocab_size": _opt_int(text_cfg, "vocab_size"),
        "num_attention_heads": _opt_int(text_cfg, "num_attention_heads"),
        "num_key_value_heads": _opt_int(text_cfg, "num_key_value_heads"),
        "head_dim": _opt_int(text_cfg, "head_dim"),
        "has_attn_output_gate": None if is_dflash2 else False,
        "has_qk_norm": False,
        "num_experts": 0,
        "num_experts_per_tok": 0,
        "shared_expert_intermediate_size": 0,
        "has_shared_expert": False,
        "tie_word_embeddings": bool(
            text_cfg.get("tie_word_embeddings", raw_config.get("tie_word_embeddings", False))
        ),
        "target_compatible": None,
    }

    def _fail(
        code: str,
        reason: str,
        *,
        blocker_tensor: str | None = None,
        blocker_shape: tuple[int, ...] | list[int] | None = None,
    ) -> dict[str, Any]:
        base_gate["eligible"] = False
        base_gate["blocker_code"] = code
        base_gate["reason"] = reason
        if blocker_tensor is not None:
            base_gate["first_blocker"] = {
                "tensor": blocker_tensor,
                "shape": list(blocker_shape or ()),
                "blocker_code": code,
                "reason": reason,
            }
        return base_gate

    # --- DFlash2 Draft Mode ---
    if is_dflash2:
        if not require_complete_model:
            base_gate["eligible"] = True
            base_gate["reason"] = "ok"
            return base_gate
        is_causal = raw_config.get("is_causal")
        if not (isinstance(is_causal, bool) and is_causal is False):
            return _fail(
                "dflash2_invalid_is_causal",
                f"DFlash2 draft requires is_causal=False in config.json, got {is_causal!r}",
            )
        df_cfg = raw_config.get("dflash_config")
        if not isinstance(df_cfg, dict):
            return _fail(
                "dflash2_missing_config",
                "DFlash2 draft config.json is missing required 'dflash_config' object",
            )
        block_size = _opt_int(df_cfg, "block_size")
        conv_kernel = _opt_int(df_cfg, "conv_kernel_size")
        conv_group = _opt_int(df_cfg, "conv_group_size")
        sel_rank = _opt_int(df_cfg, "selector_rank")
        sel_topk = _opt_int(df_cfg, "selector_top_k")
        target_layer_ids = df_cfg.get("target_layer_ids")
        if not (2 <= block_size <= 16):
            return _fail(
                "dflash2_invalid_block_size",
                f"DFlash2 block_size={block_size} is out of supported range 2..16",
            )
        if conv_kernel != 2 or conv_group <= 0:
            return _fail(
                "dflash2_invalid_conv",
                f"DFlash2 requires conv_kernel_size == 2 and conv_group_size > 0, got kernel={conv_kernel}, group={conv_group}",
            )
        if sel_rank <= 0 or sel_topk <= 0:
            return _fail(
                "dflash2_invalid_selector",
                f"DFlash2 requires selector_rank > 0 and selector_top_k > 0, got rank={sel_rank}, top_k={sel_topk}",
            )
        if (
            not isinstance(target_layer_ids, list)
            or not target_layer_ids
            or not all(_is_strict_int(x) and x >= 0 for x in target_layer_ids)
        ):
            return _fail(
                "dflash2_invalid_target_layers",
                "DFlash2 dflash_config.target_layer_ids must be a non-empty list of non-negative layer indices",
            )
        h = _opt_int(raw_config, "hidden_size")
        vocab = _opt_int(raw_config, "vocab_size")
        inter = _opt_int(raw_config, "intermediate_size")
        n_layers = _opt_int(raw_config, "num_hidden_layers")
        q_heads = _opt_int(raw_config, "num_attention_heads")
        kv_heads = _opt_int(raw_config, "num_key_value_heads")
        head_dim = _opt_int(
            raw_config,
            "head_dim",
            default=(h // q_heads if q_heads > 0 and h % q_heads == 0 else 0),
        )
        base_gate["head_dim"] = head_dim
        if h <= 0 or (h % 64) != 0 or (h % conv_group) != 0:
            return _fail(
                "dflash2_invalid_hidden_size",
                f"DFlash2 hidden_size={h} must be a positive multiple of 64 and conv_group_size={conv_group}",
            )
        if vocab <= 0 or sel_topk > vocab:
            return _fail(
                "dflash2_invalid_vocab_size",
                f"DFlash2 vocab_size={vocab} must be positive and >= selector_top_k={sel_topk}",
            )
        mask_id = df_cfg.get("mask_token_id")
        if mask_id is not None and (not _is_strict_int(mask_id) or not (0 <= mask_id < vocab)):
            return _fail(
                "dflash2_invalid_mask_token_id",
                f"DFlash2 mask_token_id={mask_id!r} must be in 0..vocab_size-1 ({vocab - 1})",
            )
        if inter <= 0 or q_heads <= 0 or kv_heads <= 0 or (q_heads % kv_heads) != 0 or head_dim <= 0:
            return _fail(
                "dflash2_invalid_attention_or_mlp",
                f"DFlash2 invalid attention/MLP geometry: intermediate_size={inter}, heads={q_heads}/{kv_heads}, head_dim={head_dim}",
            )
        if not (1 <= n_layers <= 5):
            return _fail(
                "dflash2_invalid_num_layers",
                f"DFlash2 num_hidden_layers={n_layers} must be in 1..5",
            )
        ltypes = raw_config.get("layer_types")
        if (
            not isinstance(ltypes, list)
            or len(ltypes) != n_layers
            or any(str(lt) != "sliding_attention" for lt in ltypes)
        ):
            return _fail(
                "dflash2_invalid_layer_types",
                f"DFlash2 requires {n_layers} 'sliding_attention' layer_types, got {ltypes!r}",
            )
        for tname in active_names:
            tinfo = tensors[tname]
            if tinfo.dtype != "BF16":
                return _fail(
                    "dflash2_non_bf16_tensor",
                    f"DFlash2 draft tensor '{tname}' (shape {list(tinfo.shape)}) has dtype '{tinfo.dtype}'; DFlash2 requires BF16 unquantized tensors",
                    blocker_tensor=tname,
                    blocker_shape=tinfo.shape,
                )
        base_gate["has_qk_norm"] = all(
            f"layers.{l}.self_attn.q_norm.weight" in tensors
            and f"layers.{l}.self_attn.k_norm.weight" in tensors
            for l in range(n_layers)
        )
        if target_config is not None:
            tgt_text, _ = _normalize_model_config(target_config)
            tgt_vocab = _opt_int(tgt_text, "vocab_size")
            tgt_h = _opt_int(tgt_text, "hidden_size")
            tgt_layers = _opt_int(tgt_text, "num_hidden_layers")
            if tgt_vocab > 0 and vocab != tgt_vocab:
                base_gate["target_compatible"] = False
                return _fail(
                    "dflash2_target_vocab_mismatch",
                    f"DFlash2 draft vocab_size={vocab} does not match target vocab_size={tgt_vocab}",
                )
            if tgt_h > 0 and h != tgt_h:
                base_gate["target_compatible"] = False
                return _fail(
                    "dflash2_target_hidden_mismatch",
                    f"DFlash2 draft hidden_size={h} does not match target hidden_size={tgt_h}",
                )
            max_tap = max(int(x) for x in target_layer_ids)
            if tgt_layers <= 0 or max_tap >= tgt_layers:
                base_gate["target_compatible"] = False
                return _fail(
                    "dflash2_target_layer_out_of_range",
                    f"DFlash2 target_layer_ids max={max_tap} exceeds target num_hidden_layers={tgt_layers}",
                )
            decl_tgt_layers = raw_config.get(
                "num_target_layers", df_cfg.get("num_target_layers")
            )
            if decl_tgt_layers is not None and int(decl_tgt_layers) != tgt_layers:
                base_gate["target_compatible"] = False
                return _fail(
                    "dflash2_num_target_layers_mismatch",
                    f"DFlash2 num_target_layers={decl_tgt_layers} does not match target num_hidden_layers={tgt_layers}",
                )
            base_gate["target_compatible"] = True
        base_gate["eligible"] = True
        base_gate["reason"] = "ok"
        return base_gate

    # --- Immediate Architecture Blockers (checked on ALL non-DFlash2 checkpoints) ---
    # 1. MLA (DeepSeek)
    for mla_key in _MLA_CONFIG_KEYS:
        if text_cfg.get(mla_key) is not None or raw_config.get(mla_key) is not None:
            val = text_cfg.get(mla_key, raw_config.get(mla_key))
            return _fail(
                "unsupported_arch_mla",
                f"Unsupported MLA (DeepSeek Multi-head Latent Attention) architecture: config defines '{mla_key}={val}'",
            )
    for tname in active_names:
        probe = "." + tname
        for marker in _MLA_TENSOR_MARKERS:
            if f".{marker}." in probe or probe.endswith(f".{marker}"):
                return _fail(
                    "unsupported_arch_mla",
                    f"Unsupported MLA (DeepSeek Multi-head Latent Attention) tensor '{tname}' (shape {list(tensors[tname].shape)})",
                    blocker_tensor=tname,
                    blocker_shape=tensors[tname].shape,
                )

    # 2. Mamba / Mamba2 / SSM
    raw_ltypes = text_cfg.get("layer_types")
    if isinstance(raw_ltypes, list):
        for idx, lt in enumerate(raw_ltypes):
            lt_s = str(lt).strip().lower()
            if "mamba" in lt_s or "ssm" in lt_s:
                return _fail(
                    "unsupported_arch_mamba_ssm",
                    f"Unsupported Mamba/SSM state-space architecture: layer_types[{idx}]='{lt}'",
                )
    for mamba_key in _MAMBA_CONFIG_KEYS:
        if text_cfg.get(mamba_key) is not None or raw_config.get(mamba_key) is not None:
            val = text_cfg.get(mamba_key, raw_config.get(mamba_key))
            return _fail(
                "unsupported_arch_mamba_ssm",
                f"Unsupported Mamba/SSM state-space architecture: config defines '{mamba_key}={val}'",
            )
    for tname in active_names:
        probe = "." + tname
        if ".mamba." in probe or ".ssm." in probe or ".mamba2." in probe:
            return _fail(
                "unsupported_arch_mamba_ssm",
                f"Unsupported Mamba/SSM state-space tensor '{tname}' (shape {list(tensors[tname].shape)})",
                blocker_tensor=tname,
                blocker_shape=tensors[tname].shape,
            )

    # 3. Unknown layer_types / hidden_act / additive .bias
    if isinstance(raw_ltypes, list):
        for idx, lt in enumerate(raw_ltypes):
            if str(lt) not in ("full_attention", "linear_attention"):
                return _fail(
                    "unsupported_layer_type",
                    f"Unsupported layer_type '{lt}' at layer {idx}; supported text layer_types are ['full_attention', 'linear_attention']",
                )
    act = text_cfg.get("hidden_act", text_cfg.get("hidden_activation"))
    if act is not None and str(act).strip().lower() not in ("silu", "swiglu"):
        return _fail(
            "unsupported_activation",
            f"Unsupported MLP hidden_act '{act}'; miv requires SwiGLU ('silu' or 'swiglu')",
        )
    for tname in active_names:
        if tname.endswith(".bias") and not tname.endswith(".dt_bias"):
            return _fail(
                "unsupported_linear_bias",
                f"Unsupported additive bias tensor '{tname}' (shape {list(tensors[tname].shape)}); miv text runtime requires bias-free projections",
                blocker_tensor=tname,
                blocker_shape=tensors[tname].shape,
            )

    # Minimal low-level container fixture bypass (when write_artifact is called on synthetic
    # shard unit-test stubs that omit num_hidden_layers; inspect_source_checkpoint always sets
    # require_complete_model=True).
    if not require_complete_model and "num_hidden_layers" not in text_cfg:
        base_gate["has_qk_norm"] = any(
            n.endswith((".self_attn.q_norm.weight", ".self_attn.k_norm.weight"))
            for n in text_names
        )
        base_gate["has_shared_expert"] = any(
            ".mlp.shared_expert." in ("." + n) or ".mlp.shared_experts." in ("." + n)
            for n in text_names
        )
        base_gate["has_attn_output_gate"] = bool(
            text_cfg.get("has_attn_output_gate", text_cfg.get("attn_output_gate", False))
        )
        has_gdn = any(".linear_attn." in ("." + n) for n in text_names)
        base_gate["topology"] = "gdn_hybrid_gqa" if has_gdn else "pure_gqa_swiglu"
        base_gate["eligible"] = True
        base_gate["reason"] = "ok"
        return base_gate

    # --- Full Model Structural & Geometry Validation ---
    n_layers = _opt_int(text_cfg, "num_hidden_layers")
    if n_layers <= 0:
        return _fail(
            "invalid_num_hidden_layers",
            f"Missing or non-positive num_hidden_layers={text_cfg.get('num_hidden_layers')!r} in config.json",
        )
    h = _opt_int(text_cfg, "hidden_size")
    if h < 64 or (h % 64) != 0:
        return _fail(
            "invalid_hidden_size",
            f"Unsupported hidden_size={h}; miv requires hidden_size >= 64 and divisible by 64",
        )
    vocab = _opt_int(text_cfg, "vocab_size")
    if vocab <= 0:
        return _fail(
            "invalid_vocab_size",
            f"Missing or non-positive vocab_size={text_cfg.get('vocab_size')!r} in config.json",
        )

    num_experts = _opt_int(
        text_cfg, "num_experts", "n_routed_experts", "num_local_experts", default=0
    )
    raw_num_experts = text_cfg.get(
        "num_experts", text_cfg.get("n_routed_experts", text_cfg.get("num_local_experts"))
    )
    if raw_num_experts is not None and not _is_strict_int(raw_num_experts):
        return _fail("invalid_moe_geometry", f"Invalid num_experts={raw_num_experts!r}")
    base_gate["num_experts"] = num_experts

    inter = _opt_int(text_cfg, "intermediate_size", "moe_intermediate_size")
    moe_inter = _opt_int(text_cfg, "moe_intermediate_size")
    eff_inter = moe_inter if (num_experts > 0 and moe_inter > 0 and inter <= 0) else inter
    base_gate["intermediate_size"] = eff_inter
    if eff_inter < 64 or (eff_inter % 64) != 0:
        return _fail(
            "invalid_intermediate_size",
            f"Unsupported intermediate_size={eff_inter}; miv requires intermediate_size >= 64 and divisible by 64",
        )
    if moe_inter > 0 and (moe_inter < 64 or (moe_inter % 64) != 0):
        return _fail(
            "invalid_intermediate_size",
            f"Unsupported moe_intermediate_size={moe_inter}; miv requires moe_intermediate_size >= 64 and divisible by 64",
        )

    q_heads = _opt_int(text_cfg, "num_attention_heads")
    kv_heads = _opt_int(text_cfg, "num_key_value_heads")
    if q_heads <= 0 or kv_heads <= 0 or (q_heads % kv_heads) != 0:
        return _fail(
            "invalid_attention_heads",
            f"Invalid GQA head counts: num_attention_heads={q_heads}, num_key_value_heads={kv_heads} (must be positive with num_attention_heads % num_key_value_heads == 0)",
        )
    if text_cfg.get("head_dim") is not None:
        if not _is_strict_int(text_cfg["head_dim"]):
            return _fail("unsupported_head_dim", f"Invalid head_dim={text_cfg['head_dim']!r}")
        head_dim = int(text_cfg["head_dim"])
    else:
        if (h % q_heads) != 0:
            return _fail(
                "unsupported_head_dim",
                f"Cannot infer head_dim: hidden_size={h} is not divisible by num_attention_heads={q_heads}",
            )
        head_dim = h // q_heads
    base_gate["head_dim"] = head_dim
    if head_dim not in (64, 128, 256):
        return _fail(
            "unsupported_head_dim",
            f"Unsupported head_dim={head_dim}; miv NAX/exact attention supports head_dim in (64, 128, 256)",
        )

    # RoPE validation (both rope_parameters object and top-level rope_theta / partial_rotary_factor)
    rope_theta_raw: Any = text_cfg.get("rope_theta", raw_config.get("rope_theta", 10000.0))
    prf_raw: Any = text_cfg.get(
        "partial_rotary_factor", raw_config.get("partial_rotary_factor", 1.0)
    )
    rope_params = text_cfg.get("rope_parameters")
    if rope_params is not None:
        if not isinstance(rope_params, dict):
            return _fail("invalid_rope_parameters", "rope_parameters in config.json must be an object")
        if rope_params.get("rope_theta") is not None:
            rope_theta_raw = rope_params["rope_theta"]
        if rope_params.get("partial_rotary_factor") is not None:
            prf_raw = rope_params["partial_rotary_factor"]
    try:
        rope_theta = float(rope_theta_raw)
        prf = float(prf_raw)
    except (TypeError, ValueError):
        return _fail(
            "invalid_rope_parameters",
            f"Invalid RoPE parameters: rope_theta={rope_theta_raw!r}, partial_rotary_factor={prf_raw!r}",
        )
    rotary_dim_f = head_dim * prf
    rotary_dim = int(round(rotary_dim_f))
    if (
        not math.isfinite(rope_theta)
        or rope_theta <= 0.0
        or not math.isfinite(prf)
        or prf <= 0.0
        or prf > 1.0
        or abs(rotary_dim_f - rotary_dim) > 1e-4
        or rotary_dim < 2
        or (rotary_dim % 2) != 0
    ):
        return _fail(
            "invalid_rope_parameters",
            f"Invalid RoPE geometry: rope_theta={rope_theta}, partial_rotary_factor={prf}, rotary_dim={rotary_dim_f} (requires finite rope_theta > 0, 0 < partial_rotary_factor <= 1, even rotary_dim >= 2)",
        )

    # Layer types resolution
    if raw_ltypes is not None:
        if not isinstance(raw_ltypes, list) or len(raw_ltypes) != n_layers:
            return _fail(
                "invalid_layer_types",
                f"layer_types length ({len(raw_ltypes) if isinstance(raw_ltypes, list) else 'non-list'}) does not match num_hidden_layers={n_layers}",
            )
        layer_types = [str(lt) for lt in raw_ltypes]
    else:
        layer_types = [
            "linear_attention"
            if any(f"layers.{i}.linear_attn." in n for n in text_names)
            else "full_attention"
            for i in range(n_layers)
        ]

    full_attn_indices = [i for i, lt in enumerate(layer_types) if lt == "full_attention"]
    lin_attn_indices = [i for i, lt in enumerate(layer_types) if lt == "linear_attention"]
    base_gate["full_attention_layers"] = len(full_attn_indices)
    base_gate["linear_attention_layers"] = len(lin_attn_indices)
    base_gate["topology"] = (
        "gdn_hybrid_gqa" if len(lin_attn_indices) > 0 else "pure_gqa_swiglu"
    )

    # GDN geometry bounds (ONLY when linear_attention_layers > 0)
    if len(lin_attn_indices) > 0:
        lk_heads = _opt_int(text_cfg, "linear_num_key_heads")
        lv_heads = _opt_int(text_cfg, "linear_num_value_heads")
        lk_dim = _opt_int(text_cfg, "linear_key_head_dim")
        lv_dim = _opt_int(text_cfg, "linear_value_head_dim")
        l_conv = _opt_int(text_cfg, "linear_conv_kernel_dim")
        if (
            lk_heads <= 0
            or lv_heads <= 0
            or (lv_heads % lk_heads) != 0
            or lk_dim <= 0
            or lv_dim <= 0
            or l_conv < 2
        ):
            return _fail(
                "invalid_gdn_geometry",
                f"Invalid GDN linear_attention geometry: key_heads={lk_heads}, value_heads={lv_heads}, key_dim={lk_dim}, value_dim={lv_dim}, conv_kernel={l_conv}",
            )

    # MoE geometry bounds (when num_experts != 0)
    has_shared_expert = any(
        ".mlp.shared_expert." in ("." + n) or ".mlp.shared_experts." in ("." + n)
        for n in text_names
    )
    base_gate["has_shared_expert"] = has_shared_expert
    if num_experts < 0 or num_experts == 1 or num_experts > 256:
        return _fail(
            "invalid_moe_geometry",
            f"Unsupported MoE num_experts={num_experts}; miv supports dense (0) or 2..256 routed experts",
        )
    if num_experts > 0:
        top_k = _opt_int(text_cfg, "num_experts_per_tok", "num_experts_per_token", "top_k")
        base_gate["num_experts_per_tok"] = top_k
        if top_k < 1 or top_k > 8 or top_k > num_experts:
            return _fail(
                "invalid_moe_geometry",
                f"Unsupported MoE num_experts_per_tok={top_k} for num_experts={num_experts}; miv requires 1 <= num_experts_per_tok <= 8 and <= num_experts",
            )
        raw_shared = text_cfg.get("shared_expert_intermediate_size")
        if raw_shared is not None and (not _is_strict_int(raw_shared) or int(raw_shared) < 0):
            return _fail(
                "invalid_moe_geometry",
                f"Invalid shared_expert_intermediate_size={raw_shared!r}",
            )
        shared_inter = int(raw_shared) if raw_shared is not None else 0
        if shared_inter > 0 and (shared_inter < 64 or (shared_inter % 64) != 0):
            return _fail(
                "invalid_moe_geometry",
                f"Unsupported shared_expert_intermediate_size={shared_inter}; must be 0 or a multiple of 64",
            )
        if shared_inter > 0 and not has_shared_expert:
            return _fail(
                "missing_required_tensor",
                f"Config sets shared_expert_intermediate_size={shared_inter}, but checkpoint has no mlp.shared_expert tensors",
            )
        if has_shared_expert and shared_inter == 0:
            sh_gate_t = next(
                (
                    tensors[n]
                    for n in text_names
                    if n.endswith(
                        (
                            ".mlp.shared_expert.gate_proj.weight",
                            ".mlp.shared_expert.gate_proj.weight_packed",
                            ".mlp.shared_experts.gate_proj.weight",
                            ".mlp.shared_experts.gate_proj.weight_packed",
                        )
                    )
                ),
                None,
            )
            if sh_gate_t is not None and len(sh_gate_t.shape) >= 2:
                shared_inter = int(sh_gate_t.shape[0])
                if shared_inter < 64 or (shared_inter % 64) != 0:
                    return _fail(
                        "invalid_moe_geometry",
                        f"Inferred shared_expert_intermediate_size={shared_inter} from '{sh_gate_t.name}' must be >= 64 and divisible by 64",
                        blocker_tensor=sh_gate_t.name,
                        blocker_shape=sh_gate_t.shape,
                    )
        base_gate["shared_expert_intermediate_size"] = shared_inter

    # Build fast lookup sets for module / tensor suffixes among text tensors
    text_set = set(text_names)
    module_weight_info: dict[str, TensorInfo] = {}
    for n in text_names:
        for suf in (".weight", ".weight_packed"):
            if n.endswith(suf):
                module_weight_info[n[: -len(suf)]] = tensors[n]
                break

    def _find_module_by_suffix(suffix: str) -> tuple[str, TensorInfo] | None:
        if suffix in module_weight_info:
            return suffix, module_weight_info[suffix]
        dot_suf = "." + suffix
        for mod_name, tinfo in module_weight_info.items():
            if mod_name.endswith(dot_suf):
                return mod_name, tinfo
        return None

    def _has_tensor_suffix(suffix: str) -> bool:
        if suffix in text_set:
            return True
        dot_suf = "." + suffix
        return any(n.endswith(dot_suf) for n in text_names)

    # Required global text tensors
    if _find_module_by_suffix("embed_tokens") is None:
        return _fail(
            "missing_required_tensor",
            "Missing required text embedding tensor 'embed_tokens.weight'",
        )
    if not (_has_tensor_suffix("model.norm.weight") or _has_tensor_suffix("norm.weight")):
        return _fail(
            "missing_required_tensor",
            "Missing required final RMSNorm tensor 'model.norm.weight'",
        )
    if not base_gate["tie_word_embeddings"] and _find_module_by_suffix("lm_head") is None:
        return _fail(
            "missing_required_tensor",
            "Missing required 'lm_head' tensor while tie_word_embeddings is False",
        )

    # Attention Output Gate & QK-Norm auto-detection across full-attention layers
    q_dim = q_heads * head_dim
    gate_flags: set[bool] = set()
    qk_norm_flags: set[bool] = set()
    for l_idx in full_attn_indices:
        q_found = _find_module_by_suffix(f"layers.{l_idx}.self_attn.q_proj")
        if q_found is None:
            return _fail(
                "missing_required_tensor",
                f"Missing required full-attention projection 'layers.{l_idx}.self_attn.q_proj'",
            )
        q_mod, q_tinfo = q_found
        if len(q_tinfo.shape) != 2:
            return _fail(
                "invalid_q_proj_shape",
                f"Tensor '{q_tinfo.name}' must be rank-2, got shape {list(q_tinfo.shape)}",
                blocker_tensor=q_tinfo.name,
                blocker_shape=q_tinfo.shape,
            )
        n_q = int(q_tinfo.shape[0])
        if n_q == 2 * q_dim:
            gate_flags.add(True)
        elif n_q == q_dim:
            gate_flags.add(False)
        else:
            return _fail(
                "invalid_q_proj_shape",
                f"Layer {l_idx} '{q_tinfo.name}' output rows N_q={n_q} matches neither ungated ({q_dim}) nor gated ({2 * q_dim}) for num_attention_heads={q_heads}, head_dim={head_dim}",
                blocker_tensor=q_tinfo.name,
                blocker_shape=q_tinfo.shape,
            )
        for req_proj in ("k_proj", "v_proj", "o_proj"):
            if _find_module_by_suffix(f"layers.{l_idx}.self_attn.{req_proj}") is None:
                return _fail(
                    "missing_required_tensor",
                    f"Missing required full-attention projection 'layers.{l_idx}.self_attn.{req_proj}'",
                )
        has_qn = _has_tensor_suffix(f"layers.{l_idx}.self_attn.q_norm.weight")
        has_kn = _has_tensor_suffix(f"layers.{l_idx}.self_attn.k_norm.weight")
        if has_qn != has_kn:
            return _fail(
                "partial_qk_norm",
                f"Layer {l_idx} has mismatched QK-norm tensors (q_norm={has_qn}, k_norm={has_kn})",
            )
        qk_norm_flags.add(has_qn and has_kn)

    if len(gate_flags) > 1:
        return _fail(
            "inconsistent_attn_output_gate",
            "Cross-layer disagreement in self_attn.q_proj output gate dimension across full_attention layers",
        )
    if len(qk_norm_flags) > 1:
        return _fail(
            "inconsistent_qk_norm",
            "Cross-layer disagreement in self_attn.q_norm/k_norm presence across full_attention layers",
        )
    detected_gate = (
        next(iter(gate_flags))
        if gate_flags
        else bool(text_cfg.get("has_attn_output_gate", text_cfg.get("attn_output_gate", False)))
    )
    explicit_gate = text_cfg.get("has_attn_output_gate", text_cfg.get("attn_output_gate"))
    if isinstance(explicit_gate, bool) and gate_flags and explicit_gate != detected_gate:
        return _fail(
            "attn_output_gate_config_mismatch",
            f"Config declares has_attn_output_gate={explicit_gate}, but self_attn.q_proj shape indicates {detected_gate}",
        )
    base_gate["has_attn_output_gate"] = detected_gate
    base_gate["has_qk_norm"] = next(iter(qk_norm_flags)) if qk_norm_flags else False

    # Per-layer norms, GDN & MLP/MoE completeness check
    for l_idx, ltype in enumerate(layer_types):
        for norm_suf in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            if not _has_tensor_suffix(f"layers.{l_idx}.{norm_suf}"):
                return _fail(
                    "missing_required_tensor",
                    f"Missing required layer norm tensor 'layers.{l_idx}.{norm_suf}'",
                )
        if ltype == "linear_attention":
            for gdn_proj in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"):
                if _find_module_by_suffix(f"layers.{l_idx}.linear_attn.{gdn_proj}") is None:
                    return _fail(
                        "missing_required_tensor",
                        f"Missing required GDN projection 'layers.{l_idx}.linear_attn.{gdn_proj}'",
                    )
            for gdn_param in ("conv1d.weight", "A_log", "dt_bias", "norm.weight"):
                if not _has_tensor_suffix(f"layers.{l_idx}.linear_attn.{gdn_param}"):
                    return _fail(
                        "missing_required_tensor",
                        f"Missing required GDN parameter 'layers.{l_idx}.linear_attn.{gdn_param}'",
                    )
        if num_experts == 0:
            for mlp_proj in ("gate_proj", "up_proj", "down_proj"):
                if _find_module_by_suffix(f"layers.{l_idx}.mlp.{mlp_proj}") is None:
                    return _fail(
                        "missing_required_tensor",
                        f"Missing required MLP projection 'layers.{l_idx}.mlp.{mlp_proj}'",
                    )
        else:
            if _find_module_by_suffix(f"layers.{l_idx}.mlp.gate") is None:
                return _fail(
                    "missing_required_tensor",
                    f"Missing required MoE router 'layers.{l_idx}.mlp.gate'",
                )
            for exp_proj in ("gate_proj", "up_proj", "down_proj"):
                has_sw = _find_module_by_suffix(f"layers.{l_idx}.mlp.switch_mlp.{exp_proj}") is not None
                has_ex = _find_module_by_suffix(f"layers.{l_idx}.mlp.experts.{exp_proj}") is not None
                if not (has_sw or has_ex):
                    return _fail(
                        "missing_required_tensor",
                        f"Missing required 3D MoE expert tensor 'layers.{l_idx}.mlp.switch_mlp.{exp_proj}'",
                    )
            if has_shared_expert:
                for sh_proj in ("gate_proj", "up_proj", "down_proj"):
                    has_sh1 = _find_module_by_suffix(f"layers.{l_idx}.mlp.shared_expert.{sh_proj}") is not None
                    has_sh2 = _find_module_by_suffix(f"layers.{l_idx}.mlp.shared_experts.{sh_proj}") is not None
                    if not (has_sh1 or has_sh2):
                        return _fail(
                            "missing_required_tensor",
                            f"Missing required MoE shared_expert tensor 'layers.{l_idx}.mlp.shared_expert.{sh_proj}'",
                        )

    base_gate["eligible"] = True
    base_gate["reason"] = "ok"
    return base_gate


def _detect_profile_from_headers(
    raw_config: dict[str, Any], tensors: dict[str, TensorInfo]
) -> str:
    if _is_dflash2_config(raw_config):
        return "dflash2_bf16"
    active = {n: info for n, info in tensors.items() if not _is_non_text_tensor_name(n)}
    qtable = _quantization_table(raw_config)
    mode = str(qtable.get("mode") or "").strip().lower()
    fmt = str(qtable.get("format") or "").strip().lower()
    has_comp_mx = any(n.endswith(".weight_packed") for n in active)
    has_affine = any(n.endswith(".biases") for n in active)
    has_mlx_mx = any(
        n.endswith(".scales")
        and (n[: -len(".scales")] + ".biases") not in active
        and active[n].dtype in ("U8", "F8_E8M0")
        for n in active
    )
    if fmt == "mxfp4-pack-quantized" or (has_comp_mx and not has_affine and not has_mlx_mx):
        return "compressed_mxfp4"
    if mode == "mxfp4" and (has_mlx_mx or has_comp_mx):
        return "compressed_mxfp4" if has_comp_mx and not has_mlx_mx else "mlx_mxfp4"
    if mode == "affine" and has_affine:
        return "oq_affine"
    if has_mlx_mx:
        return "mlx_mxfp4"
    if has_comp_mx:
        return "compressed_mxfp4"
    if has_affine:
        return "oq_affine"
    if active and all(info.dtype in ("BF16", "F16", "F32") for info in active.values()):
        return "direct_bf16"
    return "unknown"


def _evaluate_two_layer_capability_and_plan(
    raw_config: dict[str, Any],
    tensors: dict[str, TensorInfo],
    *,
    model_dir: Path | None = None,
    target_config: dict[str, Any] | None = None,
    nax_pretile: bool = True,
    artifact_version: int = DEFAULT_ARTIFACT_VERSION,
    require_complete_model: bool = True,
) -> dict[str, Any]:
    text_cfg, config_layout = _normalize_model_config(raw_config)
    is_dflash2 = config_layout == "dflash2"
    repack_plan = _plan_repack_layers(
        raw_config,
        text_cfg,
        tensors,
        is_dflash2=is_dflash2,
        nax_pretile=nax_pretile,
        artifact_version=artifact_version,
    )
    graph_gate = _evaluate_graph_capability_gate(
        raw_config,
        text_cfg,
        config_layout,
        tensors,
        target_config=target_config,
        require_complete_model=require_complete_model,
    )

    # Prioritize architecture blockers (MLA, Mamba, bad head_dim/geometry) first; BUT if
    # Layer 1 only reported "missing_required_tensor" because an unsupported quant scheme
    # (e.g. AWQ/GPTQ .qweight) renamed .weight tensors, surface the exact Layer 2 quant blocker!
    if not graph_gate["eligible"] and not (
        graph_gate["blocker_code"] == "missing_required_tensor"
        and not repack_plan["eligible"]
        and repack_plan["blocker_code"] is not None
        and repack_plan["blocker_code"].startswith("unsupported_quant_")
    ):
        eligible = False
        blocker_code = graph_gate["blocker_code"]
        reason = graph_gate["reason"]
        first_blocker = graph_gate.get("first_blocker") or repack_plan.get("first_blocker")
    elif not repack_plan["eligible"]:
        eligible = False
        blocker_code = repack_plan["blocker_code"]
        reason = repack_plan["reason"]
        first_blocker = repack_plan["first_blocker"]
    else:
        eligible = True
        blocker_code = None
        reason = "ok"
        first_blocker = None

    cat_counts = repack_plan["category_counts"]
    repack_summary = {
        "nax_pretile": cat_counts["nax_pretile"],
        "rowsplit_or_direct": cat_counts["rowsplit_or_direct"],
        "companion_mtp": cat_counts["companion_mtp"],
        "skip_non_text": cat_counts["skip_non_text"],
        "category_counts": {
            "nax_pretile": cat_counts["nax_pretile"],
            "rowsplit_or_direct": cat_counts["rowsplit_or_direct"],
            "companion_mtp": cat_counts["companion_mtp"],
            "skip_non_text": cat_counts["skip_non_text"],
        },
        "format_histogram": dict(repack_plan["format_histogram"]),
        "layout_histogram": dict(repack_plan["layout_histogram"]),
        "skipped_non_text_count": len(repack_plan["skipped_tensors"]),
    }

    return {
        "schema": "miv-source-inspection-v1",
        "model_dir": str(model_dir.resolve()) if model_dir is not None else "",
        "eligible": eligible,
        "blocker_code": blocker_code,
        "reason": reason,
        "first_blocker": first_blocker,
        "model_type": graph_gate["model_type"],
        "text_model_type": graph_gate["text_model_type"],
        "model_type_hint_matched": graph_gate["model_type_hint_matched"],
        "topology": graph_gate["topology"],
        "detected_profile": _detect_profile_from_headers(raw_config, tensors),
        "graph_gate": graph_gate,
        "repack_plan": repack_plan,
        "skipped_tensors": repack_plan["skipped_tensors"],
        "repack_plan_summary": repack_summary,
    }


def inspect_source_checkpoint(
    model_dir: Path | str,
    *,
    target_model_dir: Path | str | None = None,
    target_config: dict[str, Any] | None = None,
    nax_pretile: bool = True,
    artifact_version: int = DEFAULT_ARTIFACT_VERSION,
) -> dict[str, Any]:
    """O(1) < 5 ms metadata-only two-layer capability gate & per-tensor repack planner."""
    t0 = time.perf_counter()
    model_path = Path(model_dir)
    if artifact_version not in (VERSION_V1, VERSION_V2):
        raise ValueError(f"unsupported artifact_version {artifact_version}")
    try:
        raw_config, tensors = _read_checkpoint_headers_fast(model_path)
        eff_target_cfg = target_config
        if eff_target_cfg is None and target_model_dir is not None:
            tgt_path = Path(target_model_dir).resolve()
            tgt_cfg_file = tgt_path / "config.json"
            if not tgt_cfg_file.is_file():
                raise FileNotFoundError(f"missing target config.json in {target_model_dir}")
            eff_target_cfg = json.loads(tgt_cfg_file.read_bytes())
        result = _evaluate_two_layer_capability_and_plan(
            raw_config,
            tensors,
            model_dir=model_path,
            target_config=eff_target_cfg,
            nax_pretile=nax_pretile,
            artifact_version=artifact_version,
            require_complete_model=True,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError) as exc:
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        code = "missing_checkpoint_files" if isinstance(exc, FileNotFoundError) else "invalid_checkpoint_metadata"
        return {
            "schema": "miv-source-inspection-v1",
            "model_dir": str(model_path),
            "eligible": False,
            "blocker_code": code,
            "reason": str(exc),
            "first_blocker": None,
            "elapsed_ms": elapsed_ms,
            "model_type": "",
            "text_model_type": "",
            "model_type_hint_matched": False,
            "topology": "unknown",
            "detected_profile": "unknown",
            "graph_gate": {
                "eligible": False,
                "blocker_code": code,
                "reason": str(exc),
                "model_type_hint_matched": False,
            },
            "repack_plan": {
                "eligible": False,
                "blocker_code": code,
                "reason": str(exc),
                "first_blocker": None,
                "nax_pretile": nax_pretile,
                "artifact_version": artifact_version,
                "total_source_tensors": 0,
                "category_counts": {
                    "nax_pretile": 0,
                    "rowsplit_or_direct": 0,
                    "companion_mtp": 0,
                    "skip_non_text": 0,
                    "incompatible_blocker": 0,
                },
                "tensor_category_counts": {
                    "nax_pretile": 0,
                    "rowsplit_or_direct": 0,
                    "companion_mtp": 0,
                    "skip_non_text": 0,
                    "incompatible_blocker": 0,
                },
                "format_histogram": {},
                "layout_histogram": {},
                "skipped_tensors": [],
                "entries": [],
            },
            "skipped_tensors": [],
            "repack_plan_summary": {
                "nax_pretile": 0,
                "rowsplit_or_direct": 0,
                "companion_mtp": 0,
                "skip_non_text": 0,
                "category_counts": {
                    "nax_pretile": 0,
                    "rowsplit_or_direct": 0,
                    "companion_mtp": 0,
                    "skip_non_text": 0,
                },
                "format_histogram": {},
                "layout_histogram": {},
                "skipped_non_text_count": 0,
            },
        }
    result["elapsed_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)
    return result


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
    input_transform: str = "none"


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
    merged: dict[str, Any] = {}
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    for candidate in (
        text_cfg.get("quantization_config"),
        config.get("quantization_config"),
        text_cfg.get("quantization"),
        config.get("quantization"),
    ):
        if isinstance(candidate, dict):
            merged.update(candidate)
    return merged


def detect_profile(source: SafeTensorSource) -> str:
    profile = _detect_profile_from_headers(source.config, source.tensors)
    if profile == "unknown":
        raise ValueError("unable to detect a supported B0 source profile")
    return profile


def quant_spec(config: dict, module: str) -> dict:
    table = _quantization_table(config)
    spec = {
        "bits": int(table.get("bits") or 0),
        "group_size": int(table.get("group_size") or 0),
        "mode": str(table.get("mode") or ""),
    }
    override = _lookup_explicit_module_quant(table, module)
    if override is not None:
        spec.update(override)
    return spec


def _f16_to_bf16_bytes(raw: memoryview | bytes | bytearray) -> bytes:
    f32 = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    return _float32_to_bf16_u16(f32).astype("<u2", copy=False).tobytes()


def _f32_to_bf16_bytes(raw: memoryview | bytes | bytearray) -> bytes:
    f32 = np.frombuffer(raw, dtype="<f4")
    return _float32_to_bf16_u16(f32).astype("<u2", copy=False).tobytes()


def _cast_bf16_chunks(
    data: memoryview, src_dtype: str, chunk_elems: int = 8 * 1024 * 1024
) -> Iterator[bytes | memoryview]:
    if src_dtype == "BF16":
        yield from bytes_chunks(data)
        return
    elem_b = 2 if src_dtype == "F16" else 4
    step = chunk_elems * elem_b
    for off in range(0, len(data), step):
        sl = data[off : off + step]
        if src_dtype == "F16":
            yield _f16_to_bf16_bytes(sl)
        elif src_dtype == "F32":
            yield _f32_to_bf16_bytes(sl)
        else:
            raise ValueError(f"cannot cast {src_dtype} to BF16")


def _plane_from_tensor(
    source: SafeTensorSource,
    tensor_name: str,
    kind: str,
    rows: int,
    source_row_bytes: int,
    target_row_bytes: int,
    *,
    cast_f16_to_bf16: bool = False,
) -> PlaneRecipe:
    def writer(handle: BinaryIO) -> tuple[int, int, str]:
        view = source.view(tensor_name)
        try:
            if cast_f16_to_bf16:
                bf16_raw = memoryview(_f16_to_bf16_bytes(view))
                return PayloadWriter(handle).write_chunks(
                    padded_row_chunks(bf16_raw, rows, source_row_bytes, target_row_bytes)
                )
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
    text_cfg, _ = _normalize_model_config(source.config)
    bits, group_size, logical_k = _infer_module_affine_spec(
        source.config, text_cfg, module, weight, scales, biases
    )
    spec = quant_spec(source.config, module)
    if (
        (
            require_affine_mode
            and spec.get("mode") not in ("", "affine")
            and _lookup_explicit_module_quant(_quantization_table(source.config), module) is None
            and "num_hidden_layers" not in text_cfg
        )
        or bits not in (3, 4, 5, 6, 8)
        or group_size not in (64, 128)
    ):
        raise ValueError(
            f"{module}: unsupported affine spec bits={bits}, group_size={group_size} (tensor '{weight_name}', shape {list(weight.shape)})"
        )
    leading_shape = scales.shape[:-1]
    rows = product(leading_shape)
    groups = scales.shape[-1]
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
                cast_f16_to_bf16=(scales.dtype == "F16"),
            ),
            _plane_from_tensor(
                source,
                biases_name,
                "bias",
                rows,
                groups * 2,
                padded_k // group_size * 2,
                cast_f16_to_bf16=(biases.dtype == "F16"),
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


def _build_mlx_mxfp4_module_recipe(
    source: SafeTensorSource, module: str
) -> tuple[ObjectRecipe, list[str]] | None:
    weight_name = module + ".weight"
    scales_name = module + ".scales"
    weight = source.tensors[weight_name]
    scales = source.tensors[scales_name]
    if weight.dtype != "U32" or scales.dtype not in ("U8", "F8_E8M0"):
        return None
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
    format_id = "mxfp4_e2m1_g32_e8m0_v1"
    recipe = ObjectRecipe(
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
    return recipe, source_names


def _build_compressed_mxfp4_module_recipe(
    source: SafeTensorSource, module: str
) -> tuple[ObjectRecipe, list[str]]:
    packed_name = module + ".weight_packed"
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
    format_id = "mxfp4_e2m1_g32_e8m0_v1"
    recipe = ObjectRecipe(
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
    return recipe, source_names


def _build_all_quantized_recipes(
    source: SafeTensorSource, *, require_affine_mode: bool = False
) -> tuple[list[ObjectRecipe], set[str]]:
    recipes: list[ObjectRecipe] = []
    consumed: set[str] = set()
    active_names = [n for n in sorted(source.tensors) if not _is_non_text_tensor_name(n)]
    for packed_name in (n for n in active_names if n.endswith(".weight_packed")):
        module = packed_name[: -len(".weight_packed")]
        recipe, src_names = _build_compressed_mxfp4_module_recipe(source, module)
        recipes.append(recipe)
        consumed.update(src_names)
    for weight_name in (n for n in active_names if n.endswith(".weight")):
        if weight_name in consumed:
            continue
        module = weight_name[: -len(".weight")]
        scales_name = module + ".scales"
        biases_name = module + ".biases"
        if scales_name in source.tensors and biases_name in source.tensors:
            recipe = _build_affine_recipe(
                source, module, require_affine_mode=require_affine_mode
            )
            recipes.append(recipe)
            consumed.update([weight_name, scales_name, biases_name])
        elif scales_name in source.tensors:
            mx_res = _build_mlx_mxfp4_module_recipe(source, module)
            if mx_res is not None:
                recipe, mx_names = mx_res
                recipes.append(recipe)
                consumed.update(mx_names)
    return recipes, consumed


def build_oq_recipes(source: SafeTensorSource) -> tuple[list[ObjectRecipe], set[str]]:
    recipes, consumed = _build_all_quantized_recipes(source, require_affine_mode=False)
    if not recipes:
        raise ValueError("oQ source contains no affine modules")
    return recipes, consumed


def build_mlx_mxfp4_recipes(
    source: SafeTensorSource,
) -> tuple[list[ObjectRecipe], set[str]]:
    recipes, consumed = _build_all_quantized_recipes(source, require_affine_mode=False)
    if not recipes:
        raise ValueError("MLX source contains no MXFP4 modules")
    return recipes, consumed


def build_compressed_mxfp4_recipes(
    source: SafeTensorSource,
) -> tuple[list[ObjectRecipe], set[str]]:
    recipes: list[ObjectRecipe] = []
    consumed: set[str] = set()
    for packed_name in sorted(
        name
        for name in source.tensors
        if name.endswith(".weight_packed") and not _is_non_text_tensor_name(name)
    ):
        module = packed_name[: -len(".weight_packed")]
        recipe, source_names = _build_compressed_mxfp4_module_recipe(source, module)
        recipes.append(recipe)
        consumed.update(source_names)
    # Also support mixed affine/mxfp4 layers if present alongside compressed mxfp4
    for weight_name in sorted(
        name
        for name in source.tensors
        if name.endswith(".weight")
        and name not in consumed
        and not _is_non_text_tensor_name(name)
    ):
        module = weight_name[: -len(".weight")]
        scales_name = module + ".scales"
        biases_name = module + ".biases"
        if scales_name in source.tensors and biases_name in source.tensors:
            recipe = _build_affine_recipe(source, module, require_affine_mode=False)
            recipes.append(recipe)
            consumed.update([weight_name, scales_name, biases_name])
        elif scales_name in source.tensors:
            mx_res = _build_mlx_mxfp4_module_recipe(source, module)
            if mx_res is not None:
                recipe, mx_names = mx_res
                recipes.append(recipe)
                consumed.update(mx_names)
    if not recipes:
        raise ValueError("compressed-tensors source contains no MXFP4 modules")
    return recipes, consumed


def build_direct_recipes(
    source: SafeTensorSource, consumed: set[str]
) -> list[ObjectRecipe]:
    recipes: list[ObjectRecipe] = []
    for name in sorted(set(source.tensors) - consumed):
        if _is_non_text_tensor_name(name):
            continue
        info = source.tensors[name]
        if info.dtype in ("F16", "F32"):
            fmt_id = "direct_bf16_v1"
            elem_count = 1
            for dim in info.shape:
                elem_count *= dim
            logical_bytes = elem_count * 2
        else:
            fmt_id = f"direct_{info.dtype.lower()}_v1"
            logical_bytes = info.nbytes

        def writer(
            handle: BinaryIO,
            tensor_name: str = name,
            src_dtype: str = info.dtype,
        ) -> tuple[int, int, str]:
            view = source.view(tensor_name)
            try:
                if src_dtype in ("F16", "F32"):
                    return PayloadWriter(handle).write_chunks(
                        _cast_bf16_chunks(view, src_dtype)
                    )
                return PayloadWriter(handle).write_chunks(bytes_chunks(view))
            finally:
                view.release()

        recipes.append(
            ObjectRecipe(
                name=name,
                format_id=fmt_id,
                layout_id="contiguous_le_v1",
                shape=info.shape,
                padded_k=None,
                planes=[PlaneRecipe("data", logical_bytes, writer)],
                source_tensors=[name],
            )
        )
    return recipes


def build_dflash2_recipes(source: SafeTensorSource) -> list[ObjectRecipe]:
    if not _is_dflash2_config(source.config):
        raise ValueError("DFlash2 profile requires DFlash2DraftModel config")
    active = {
        name: info
        for name, info in source.tensors.items()
        if not _is_non_text_tensor_name(name)
    }
    if not active or any(info.dtype != "BF16" for info in active.values()):
        raise ValueError("DFlash2 profile requires a nonempty all-BF16 checkpoint")
    recipes: list[ObjectRecipe] = []
    for source_name in sorted(active):
        info = active[source_name]
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
    if profile == "direct_bf16":
        quantized, consumed = _build_all_quantized_recipes(source, require_affine_mode=False)
    elif profile == "oq_affine":
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


def _target_layout_for_object(
    name: str,
    format_id: str,
    base_layout: str,
    shape: tuple[int, ...] | list[int],
    *,
    nax_pretile: bool,
    artifact_version: int,
    is_embedding: bool = False,
) -> tuple[str, bool]:
    """Return (target_layout, did_pretile) for an object."""
    use = infer_weight_use(name)
    if (
        not nax_pretile
        or is_embedding
        or use in ("embedding", "moe_router", "moe_shared_gate")
        or len(shape) != 2
    ):
        return base_layout, False
    rows, logical_k = int(shape[0]), int(shape[1])
    if rows <= 0 or logical_k < 512:
        return base_layout, False
    if format_id == "mxfp4_e2m1_g32_e8m0_v1" and (logical_k % 128) == 0:
        return "mx_nax_tiled_bk128_v1", True
    affine = _parse_affine_format(format_id)
    if affine is not None:
        bits, src_gs = affine
        if bits in (3, 4, 5, 6, 8) and src_gs in (64, 128) and (logical_k % src_gs) == 0:
            if artifact_version == VERSION_V2:
                return f"oq_nax_lane_g{src_gs}_v2", True
            return "oq_nax_tiled_g64_v1", True
    return base_layout, False


def _plan_plane_specs(
    format_id: str,
    layout_id: str,
    shape: tuple[int, ...] | list[int],
    padded_k: int | None,
    direct_nbytes: int | None = None,
) -> list[tuple[str, int]]:
    if format_id.startswith("direct_"):
        assert direct_nbytes is not None
        return [("data", int(direct_nbytes))]
    expected = _expected_plane_bytes(
        {
            "name": "<plan>",
            "format": format_id,
            "layout": layout_id,
            "shape": list(shape),
            "padded_k": padded_k,
        }
    )
    assert expected is not None
    return list(expected.items())


def _dequantize_mxfp4_u8_f32(packed_u8: np.ndarray, scales_u8: np.ndarray) -> np.ndarray:
    rows, half_k = packed_u8.shape
    k = half_k * 2
    lo = packed_u8 & np.uint8(0x0F)
    hi = packed_u8 >> np.uint8(4)
    codes = np.empty((rows, k), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    if np.any(scales_u8 == np.uint8(0xFF)):
        raise ValueError("MXFP4 E8M0 NaN scale is not accepted")
    vals = _E2M1_LUT[codes].reshape(rows, k // 32, 32)
    sc = _E8M0_LUT[scales_u8]
    return np.ascontiguousarray((vals * sc[:, :, None]).reshape(rows, k), dtype=np.float32)


def _write_plane_stream(
    fd: int,
    plane_offset: int,
    chunks: Iterable[bytes | bytearray | memoryview],
) -> tuple[int, int, str]:
    written = 0
    checksum = 0
    digest = hashlib.sha256()
    for chunk in chunks:
        view = memoryview(chunk).cast("B")
        if not view:
            continue
        _pwrite_all(fd, view, plane_offset + written)
        written += len(view)
        checksum = crc32(view, checksum)
        digest.update(view)
    return written, checksum, digest.hexdigest()


class _IncrementalPlaneWriter:
    def __init__(self, fd: int, kind: str, offset: int, expected_bytes: int, alignment: int = PLANE_ALIGNMENT):
        self.fd = fd
        self.kind = kind
        self.offset = offset
        self.expected_bytes = expected_bytes
        self.alignment = alignment
        self.written = 0
        self.crc = 0
        self.sha = hashlib.sha256()

    def write(self, data: bytes | bytearray | memoryview | np.ndarray) -> None:
        if isinstance(data, np.ndarray):
            view = memoryview(np.ascontiguousarray(data)).cast("B")
        else:
            view = memoryview(data).cast("B")
        if not view:
            return
        _pwrite_all(self.fd, view, self.offset + self.written)
        self.written += len(view)
        self.crc = crc32(view, self.crc)
        self.sha.update(view)

    def finish(self, obj_name: str) -> dict:
        if self.written != self.expected_bytes:
            raise ValueError(
                f"{obj_name}/{self.kind}: wrote {self.written}, expected {self.expected_bytes}"
            )
        return {
            "kind": self.kind,
            "offset": self.offset,
            "bytes": self.written,
            "alignment": self.alignment,
            "crc32": f"{self.crc:08x}",
            "sha256": self.sha.hexdigest(),
        }


def _emit_affine_from_source(
    source: SafeTensorSource,
    recipe: ObjectRecipe,
    fd: int,
    object_offset: int,
    object_bytes: int,
    plane_layouts: list[tuple[str, int, int, int]],
    target_layout: str,
    *,
    do_fwht64: bool,
) -> tuple[list[dict], str, float, float]:
    bits, src_gs = _parse_affine_format(recipe.format_id)  # type: ignore[misc]
    rows = product(recipe.shape[:-1])
    logical_k = int(recipe.shape[-1])
    padded_k = int(recipe.padded_k or align_up(logical_k, 128))
    weight_name, scales_name, biases_name = recipe.source_tensors[:3]
    is_tiled = target_layout in _AFFINE_NAX_ALL_LAYOUTS
    lane_v2 = target_layout in _AFFINE_NAX_V2_LAYOUTS

    w_view = source.view(weight_name)
    s_view = source.view(scales_name)
    b_view = source.view(biases_name)
    total_sse = 0.0
    total_sig_sq = 0.0
    try:
        src_code_row = logical_k * bits // 8
        src_groups = logical_k // src_gs
        w_raw = np.frombuffer(w_view, dtype=np.uint8).reshape(rows, src_code_row)
        if source.tensors[scales_name].dtype == "F16":
            s_raw = _float32_to_bf16_u16(
                np.frombuffer(s_view, dtype="<f2").astype(np.float32)
            ).reshape(rows, src_groups)
        else:
            s_raw = np.frombuffer(s_view, dtype=np.uint16).reshape(rows, src_groups)
        if source.tensors[biases_name].dtype == "F16":
            b_raw = _float32_to_bf16_u16(
                np.frombuffer(b_view, dtype="<f2").astype(np.float32)
            ).reshape(rows, src_groups)
        else:
            b_raw = np.frombuffer(b_view, dtype=np.uint16).reshape(rows, src_groups)

        writers = {
            kind: _IncrementalPlaneWriter(fd, kind, off, nbytes, align)
            for kind, off, nbytes, align in plane_layouts
        }

        if is_tiled:
            bn = 32 if rows >= 65536 else 16
            padded_n = align_up(rows, bn)
            groups64 = logical_k // 64
            sb_buf = np.zeros((groups64, padded_n), dtype=np.uint32)
            # Chunk rows by multiple of bn for bounded RAM and L2 locality
            chunk_rows = max(bn, (256 // bn) * bn) if do_fwht64 else max(bn, (1024 // bn) * bn)
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                n_chunk = r1 - r0
                padded_chunk = align_up(n_chunk, bn)
                if do_fwht64:
                    codes = _unpack_packed_affine_codes_u8(w_raw[r0:r1], logical_k, bits)
                    sc_f32 = _bf16_u16_to_float32(s_raw[r0:r1])
                    bi_f32 = _bf16_u16_to_float32(b_raw[r0:r1])
                    w_f32 = (
                        codes.astype(np.float32).reshape(n_chunk, src_groups, src_gs)
                        * sc_f32[:, :, None]
                        + bi_f32[:, :, None]
                    ).reshape(n_chunk, logical_k)
                    w_f32 = np.ascontiguousarray(w_f32, dtype=np.float32)
                    fwht64_f32_inplace(w_f32)
                    codes, sc_u16, bi_u16, sse, sig_sq = quantize_oq_affine_f32(
                        w_f32, bits, src_gs
                    )
                    total_sse += sse
                    total_sig_sq += sig_sq
                else:
                    codes = _unpack_packed_affine_codes_u8(w_raw[r0:r1], logical_k, bits)
                    sc_u16 = s_raw[r0:r1]
                    bi_u16 = b_raw[r0:r1]

                if padded_chunk > n_chunk:
                    pad_codes = np.zeros((padded_chunk, logical_k), dtype=np.uint8)
                    pad_codes[:n_chunk] = codes
                    codes = pad_codes
                for kind, arr in _pack_nax_affine_code_tiles(codes, bits, bn, lane_v2):
                    writers[kind].write(arr)

                if src_gs == 128:
                    sc_64 = np.repeat(sc_u16, 2, axis=1)
                    bi_64 = np.repeat(bi_u16, 2, axis=1)
                else:
                    sc_64 = sc_u16
                    bi_64 = bi_u16
                packed_sb = sc_64.astype(np.uint32) | (bi_64.astype(np.uint32) << np.uint32(16))
                sb_buf[:, r0:r1] = packed_sb.T
            writers["nax_sb"].write(sb_buf)
        else:
            # Rowsplit untiled (2D or 3D MoE experts)
            target_groups = padded_k // src_gs
            chunk_rows = 256 if do_fwht64 else 1024
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                n_chunk = r1 - r0
                if do_fwht64:
                    codes = _unpack_packed_affine_codes_u8(w_raw[r0:r1], logical_k, bits)
                    sc_f32 = _bf16_u16_to_float32(s_raw[r0:r1])
                    bi_f32 = _bf16_u16_to_float32(b_raw[r0:r1])
                    w_f32 = (
                        codes.astype(np.float32).reshape(n_chunk, src_groups, src_gs)
                        * sc_f32[:, :, None]
                        + bi_f32[:, :, None]
                    ).reshape(n_chunk, logical_k)
                    w_f32 = np.ascontiguousarray(w_f32, dtype=np.float32)
                    fwht64_f32_inplace(w_f32)
                    codes, sc_u16, bi_u16, sse, sig_sq = quantize_oq_affine_f32(
                        w_f32, bits, src_gs
                    )
                    total_sse += sse
                    total_sig_sq += sig_sq
                    for kind, arr in _pack_rowsplit_affine_planes_from_codes(
                        codes, bits, padded_k
                    ):
                        writers[kind].write(arr)
                else:
                    if bits in (4, 8):
                        kind = "code" if bits == 8 else "base_code"
                        target_row_b = padded_k if bits == 8 else padded_k // 2
                        chunk_b = w_raw[r0:r1]
                        if target_row_b != src_code_row:
                            padded_b = np.zeros((n_chunk, target_row_b), dtype=np.uint8)
                            padded_b[:, :src_code_row] = chunk_b
                            chunk_b = padded_b
                        writers[kind].write(chunk_b)
                    else:
                        codes = _unpack_packed_affine_codes_u8(w_raw[r0:r1], logical_k, bits)
                        for kind, arr in _pack_rowsplit_affine_planes_from_codes(
                            codes, bits, padded_k
                        ):
                            writers[kind].write(arr)
                    sc_u16 = s_raw[r0:r1]
                    bi_u16 = b_raw[r0:r1]

                if target_groups != src_groups:
                    sc_pad = np.zeros((n_chunk, target_groups), dtype=np.uint16)
                    bi_pad = np.zeros((n_chunk, target_groups), dtype=np.uint16)
                    sc_pad[:, :src_groups] = sc_u16
                    bi_pad[:, :src_groups] = bi_u16
                    sc_u16, bi_u16 = sc_pad, bi_pad
                writers["scale"].write(sc_u16)
                writers["bias"].write(bi_u16)

        plane_records = [writers[kind].finish(recipe.name) for kind, _, _, _ in plane_layouts]
        obj_sha = _sha256_fd_range(fd, object_offset, object_bytes)
        return plane_records, obj_sha, total_sse, total_sig_sq
    finally:
        w_view.release()
        s_view.release()
        b_view.release()


def _emit_mxfp4_from_source(
    source: SafeTensorSource,
    recipe: ObjectRecipe,
    fd: int,
    object_offset: int,
    object_bytes: int,
    plane_layouts: list[tuple[str, int, int, int]],
    target_layout: str,
    *,
    do_fwht64: bool,
) -> tuple[list[dict], str, float, float]:
    rows = product(recipe.shape[:-1])
    logical_k = int(recipe.shape[-1])
    padded_k = int(recipe.padded_k or align_up(logical_k, 128))
    weight_name, scales_name = recipe.source_tensors[:2]
    is_tiled = target_layout == "mx_nax_tiled_bk128_v1"

    w_view = source.view(weight_name)
    s_view = source.view(scales_name)
    total_sse = 0.0
    total_sig_sq = 0.0
    try:
        w_raw = np.frombuffer(w_view, dtype=np.uint8).reshape(rows, logical_k // 2)
        s_raw = np.frombuffer(s_view, dtype=np.uint8).reshape(rows, logical_k // 32)
        writers = {
            kind: _IncrementalPlaneWriter(fd, kind, off, nbytes, align)
            for kind, off, nbytes, align in plane_layouts
        }

        if is_tiled:
            bn = 32 if rows >= 65536 else 16
            k_blocks = logical_k // 128
            chunk_rows = max(bn, (256 // bn) * bn) if do_fwht64 else max(bn, (1024 // bn) * bn)
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                n_chunk = r1 - r0
                padded_chunk = align_up(n_chunk, bn)
                if do_fwht64:
                    w_f32 = _dequantize_mxfp4_u8_f32(w_raw[r0:r1], s_raw[r0:r1])
                    fwht64_f32_inplace(w_f32)
                    c_chunk, s_chunk, sse, sig_sq = quantize_mxfp4_e2m1_f32(w_f32)
                    total_sse += sse
                    total_sig_sq += sig_sq
                else:
                    c_chunk = w_raw[r0:r1]
                    s_chunk = s_raw[r0:r1]
                if padded_chunk > n_chunk:
                    pad_c = np.zeros((padded_chunk, logical_k // 2), dtype=np.uint8)
                    pad_c[:n_chunk] = c_chunk
                    c_chunk = pad_c
                    pad_s = np.full((padded_chunk, logical_k // 32), 0x7F, dtype=np.uint8)
                    pad_s[:n_chunk] = s_chunk
                    s_chunk = pad_s
                n_tiles = padded_chunk // bn
                tiled_c = np.ascontiguousarray(
                    c_chunk.reshape(n_tiles, bn, k_blocks, 64).transpose(0, 2, 1, 3)
                )
                tiled_s = np.ascontiguousarray(
                    s_chunk.reshape(n_tiles, bn, k_blocks, 4).transpose(0, 2, 1, 3)
                )
                writers["nax_data"].write(tiled_c)
                writers["nax_scale"].write(tiled_s)
        else:
            chunk_rows = 256 if do_fwht64 else 1024
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                n_chunk = r1 - r0
                if do_fwht64:
                    w_f32 = _dequantize_mxfp4_u8_f32(w_raw[r0:r1], s_raw[r0:r1])
                    fwht64_f32_inplace(w_f32)
                    c_chunk, s_chunk, sse, sig_sq = quantize_mxfp4_e2m1_f32(w_f32)
                    total_sse += sse
                    total_sig_sq += sig_sq
                else:
                    c_chunk = w_raw[r0:r1]
                    s_chunk = s_raw[r0:r1]
                if padded_k != logical_k:
                    c_pad = np.zeros((n_chunk, padded_k // 2), dtype=np.uint8)
                    c_pad[:, : logical_k // 2] = c_chunk
                    s_pad = np.zeros((n_chunk, padded_k // 32), dtype=np.uint8)
                    s_pad[:, : logical_k // 32] = s_chunk
                    c_chunk, s_chunk = c_pad, s_pad
                writers["data"].write(c_chunk)
                writers["scale"].write(s_chunk)

        plane_records = [writers[kind].finish(recipe.name) for kind, _, _, _ in plane_layouts]
        obj_sha = _sha256_fd_range(fd, object_offset, object_bytes)
        return plane_records, obj_sha, total_sse, total_sig_sq
    finally:
        w_view.release()
        s_view.release()


def _emit_direct_from_source(
    source: SafeTensorSource,
    recipe: ObjectRecipe,
    fd: int,
    plane_layouts: list[tuple[str, int, int, int]],
) -> tuple[list[dict], str, float, float]:
    kind, p_off, p_bytes, p_align = plane_layouts[0]
    tensor_name = recipe.source_tensors[0]
    src_dtype = source.tensors[tensor_name].dtype
    view = source.view(tensor_name)
    try:
        chunks = (
            _cast_bf16_chunks(view, src_dtype)
            if recipe.format_id == "direct_bf16_v1" and src_dtype in ("F16", "F32")
            else bytes_chunks(view)
        )
        written, checksum, p_sha = _write_plane_stream(fd, p_off, chunks)
    finally:
        view.release()
    if written != p_bytes:
        raise ValueError(f"{recipe.name}/{kind}: wrote {written}, expected {p_bytes}")
    rec = {
        "kind": kind,
        "offset": p_off,
        "bytes": written,
        "alignment": p_align,
        "crc32": f"{checksum:08x}",
        "sha256": p_sha,
    }
    return [rec], p_sha, 0.0, 0.0


def _write_artifact_container(
    output_path: Path,
    manifest: dict,
    payload_name: str,
    *,
    artifact_version: int,
    flags: int,
    magic: bytes = MAGIC,
) -> tuple[str, int, str]:
    if magic not in VALID_MAGICS:
        raise ValueError(f"unsupported artifact magic: {magic!r}")
    manifest_bytes = canonical_json(manifest)
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest_crc = crc32(manifest_bytes)
    payload_offset = align_up(HEADER_BYTES + len(manifest_bytes), PAYLOAD_ALIGNMENT)
    payload_bytes = int(manifest["payload_bytes"])
    object_count = len(manifest["objects"])
    header_prefix = HEADER_STRUCT.pack(
        magic,
        artifact_version,
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
        magic,
        artifact_version,
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
    return file_digest.hexdigest(), manifest_crc, manifest_sha256


def write_artifact(
    source: SafeTensorSource,
    output_path: Path,
    source_repo: str,
    profile: str = "auto",
    *,
    source_provenance: dict[str, Any] | None = None,
    write_report: bool = True,
    artifact_version: int = VERSION_V1,
    nax_pretile: bool = False,
    fwht64: bool | None = None,
    fwht64_o_proj: bool = True,
    jobs: int = DEFAULT_JOBS,
    prerotated_modules: set[str] | frozenset[str] | None = None,
    legacy_magic: bool = False,
) -> dict:
    t0 = time.monotonic()
    preflight = _evaluate_two_layer_capability_and_plan(
        source.config,
        source.tensors,
        model_dir=source.model_dir,
        nax_pretile=nax_pretile,
        artifact_version=artifact_version,
        require_complete_model=_has_full_model_arch_config(source.config),
    )
    if not preflight["eligible"]:
        fb = preflight.get("first_blocker")
        if isinstance(fb, dict) and fb.get("tensor"):
            raise ValueError(
                f"checkpoint failed capability/repack gate [{preflight['blocker_code']}] "
                f"on tensor '{fb['tensor']}' (shape {fb['shape']}): {fb['reason']}"
            )
        raise ValueError(
            f"checkpoint failed capability/repack gate [{preflight['blocker_code']}]: {preflight['reason']}"
        )
    detected = preflight["detected_profile"] if profile == "auto" else profile
    if detected == "unknown":
        detected = detect_profile(source)
    eff_fwht64 = resolve_fwht64_enabled(artifact_version, fwht64, detected)
    max_workers = max(1, int(jobs))
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

    # Pre-plan exact deterministic object & plane byte offsets upfront
    planned_tasks: list[dict[str, Any]] = []
    cursor = 0
    pretiled_count = 0
    prerot_set = set(prerotated_modules or ())
    for object_id, recipe in enumerate(recipes):
        is_embed = (
            infer_weight_use(recipe.name) == "embedding"
            or (recipe.binding is not None and recipe.binding.get("use") == "embedding")
        )
        target_layout, did_pretile = _target_layout_for_object(
            recipe.name,
            recipe.format_id,
            recipe.layout_id,
            recipe.shape,
            nax_pretile=nax_pretile,
            artifact_version=artifact_version,
            is_embedding=is_embed,
        )
        if did_pretile:
            pretiled_count += 1
        already_rot = recipe.name in prerot_set
        do_rot = (not already_rot) and is_fwht64_eligible(
            recipe.name,
            recipe.shape,
            recipe.format_id,
            profile=detected,
            fwht64=eff_fwht64,
            fwht64_o_proj=fwht64_o_proj,
        )
        input_transform = "fwht64" if (already_rot or do_rot) else "none"
        direct_nbytes = recipe.planes[0].logical_bytes if recipe.format_id.startswith("direct_") else None
        plane_sizes = _plan_plane_specs(
            recipe.format_id, target_layout, recipe.shape, recipe.padded_k, direct_nbytes
        )
        cursor = align_up(cursor, OBJECT_ALIGNMENT)
        object_begin = cursor
        plane_layouts: list[tuple[str, int, int, int]] = []
        for p_kind, p_bytes in plane_sizes:
            cursor = align_up(cursor, PLANE_ALIGNMENT)
            plane_layouts.append((p_kind, cursor, p_bytes, PLANE_ALIGNMENT))
            cursor += p_bytes
        object_bytes = cursor - object_begin
        planned_tasks.append(
            {
                "object_id": object_id,
                "recipe": recipe,
                "target_layout": target_layout,
                "do_rot": do_rot,
                "input_transform": input_transform,
                "object_begin": object_begin,
                "object_bytes": object_bytes,
                "plane_layouts": plane_layouts,
            }
        )
    payload_bytes = cursor

    try:
        os.ftruncate(payload_fd, payload_bytes)

        def _execute_one(task: dict[str, Any]) -> tuple[int, list[dict], str, float, float]:
            recipe: ObjectRecipe = task["recipe"]
            fmt = recipe.format_id
            if fmt.startswith("direct_"):
                planes, obj_sha, sse, sig_sq = _emit_direct_from_source(
                    source, recipe, payload_fd, task["plane_layouts"]
                )
            elif fmt == "mxfp4_e2m1_g32_e8m0_v1":
                planes, obj_sha, sse, sig_sq = _emit_mxfp4_from_source(
                    source,
                    recipe,
                    payload_fd,
                    task["object_begin"],
                    task["object_bytes"],
                    task["plane_layouts"],
                    task["target_layout"],
                    do_fwht64=task["do_rot"],
                )
            elif _parse_affine_format(fmt) is not None:
                planes, obj_sha, sse, sig_sq = _emit_affine_from_source(
                    source,
                    recipe,
                    payload_fd,
                    task["object_begin"],
                    task["object_bytes"],
                    task["plane_layouts"],
                    task["target_layout"],
                    do_fwht64=task["do_rot"],
                )
            else:
                raise ValueError(f"unsupported recipe format {fmt}")
            return task["object_id"], planes, obj_sha, sse, sig_sq

        results_by_id: dict[int, tuple[list[dict], str, float, float]] = {}
        effective_workers = min(max_workers, max(1, len(planned_tasks)))
        if effective_workers <= 1:
            for task in planned_tasks:
                oid, planes, obj_sha, sse, sig_sq = _execute_one(task)
                results_by_id[oid] = (planes, obj_sha, sse, sig_sq)
        else:
            # Submit heaviest tasks first (LPT scheduling) so tail stragglers don't dominate
            ordered_tasks = sorted(
                planned_tasks,
                key=lambda item: (item["do_rot"], item["object_bytes"]),
                reverse=True,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = [pool.submit(_execute_one, task) for task in ordered_tasks]
                for fut in concurrent.futures.as_completed(futures):
                    oid, planes, obj_sha, sse, sig_sq = fut.result()
                    results_by_id[oid] = (planes, obj_sha, sse, sig_sq)

        os.fsync(payload_fd)
        os.close(payload_fd)
        payload_fd = -1

        object_records: list[dict] = []
        bindings: list[dict] = []
        total_sse = 0.0
        total_sig_sq = 0.0
        fwht64_count = 0
        for task in planned_tasks:
            oid = task["object_id"]
            recipe = task["recipe"]
            plane_records, obj_sha, sse, sig_sq = results_by_id[oid]
            total_sse += sse
            total_sig_sq += sig_sq
            record: dict[str, Any] = {
                "id": oid,
                "name": recipe.name,
                "format": recipe.format_id,
                "layout": task["target_layout"],
                "shape": list(recipe.shape),
                "object_offset": task["object_begin"],
                "object_bytes": task["object_bytes"],
                "object_sha256": obj_sha,
                "planes": plane_records,
                "source_tensors": recipe.source_tensors,
            }
            if recipe.padded_k is not None:
                record["padded_k"] = recipe.padded_k
            if task["input_transform"] == "fwht64":
                record["input_transform"] = "fwht64"
                fwht64_count += 1
            object_records.append(record)
            if recipe.binding is not None:
                bindings.append({**recipe.binding, "object_id": oid})

        graph_gate = preflight["graph_gate"]
        skipped_tensors = list(preflight["skipped_tensors"])
        repack_plan_summary = dict(preflight["repack_plan_summary"])

        # Tied word embeddings: emit use="lm_head" binding pointing to embed_tokens object_id when lm_head is absent.
        if (
            detected != "dflash2_bf16"
            and bool(graph_gate.get("tie_word_embeddings"))
            and not any(
                not _is_mtp_object_name(str(rec["name"]))
                and infer_weight_use(str(rec["name"]).removesuffix(".weight")) == "lm_head"
                for rec in object_records
            )
        ):
            embed_rec = next(
                (
                    rec
                    for rec in object_records
                    if not _is_mtp_object_name(str(rec["name"]))
                    and infer_weight_use(str(rec["name"]).removesuffix(".weight")) == "embedding"
                    and len(rec["shape"]) >= 2
                ),
                None,
            )
            if embed_rec is not None:
                emb_base = str(embed_rec["name"]).removesuffix(".weight")
                if emb_base.endswith("model.embed_tokens"):
                    tied_head_name = emb_base[: -len("model.embed_tokens")] + "lm_head"
                elif emb_base.endswith(".embed_tokens"):
                    tied_head_name = emb_base[: -len(".embed_tokens")] + ".lm_head"
                else:
                    tied_head_name = "lm_head"
                bindings.append(
                    {
                        "name": tied_head_name,
                        "object_id": int(embed_rec["id"]),
                        "format": str(embed_rec["format"]),
                        "use": "lm_head",
                        "activation_policy": (
                            "AllowA4"
                            if embed_rec["format"] == "mxfp4_e2m1_g32_e8m0_v1"
                            else "A16Only"
                        ),
                        "leading_shape": list(embed_rec["shape"][:-2]),
                        "out_features": int(embed_rec["shape"][-2]),
                        "in_features": int(embed_rec["shape"][-1]),
                    }
                )

        format_registry = _format_registry_for_objects(object_records)
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

        magic, schema_id, converter_name = _branding_for_write(
            artifact_version, legacy_magic=legacy_magic
        )
        if artifact_version == VERSION_V2:
            flags, format_features = _compute_v2_flags_and_features(
                object_records, fwht64_o_proj=fwht64_o_proj
            )
        else:
            flags = 0
            format_features = None

        model_meta: dict[str, Any] = {
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
            "has_qk_norm": bool(graph_gate.get("has_qk_norm", False)),
            "has_shared_expert": bool(graph_gate.get("has_shared_expert", False)),
            "skipped_non_text_count": len(skipped_tensors),
        }
        if detected != "dflash2_bf16":
            model_meta["has_attn_output_gate"] = bool(
                graph_gate.get("has_attn_output_gate", False)
            )

        manifest: dict[str, Any] = {
            "schema": schema_id,
            "container_version": artifact_version,
            "model": model_meta,
            "converter": {
                "name": converter_name,
                "version": artifact_version,
                "endianness": "little",
            },
            "format_registry": format_registry,
            "objects": object_records,
            "views": [],
            "bindings": bindings,
            "payload_bytes": payload_bytes,
        }
        if format_features is not None:
            manifest["format_features"] = format_features

        artifact_sha256, manifest_crc, manifest_sha256 = _write_artifact_container(
            output_path,
            manifest,
            payload_name,
            artifact_version=artifact_version,
            flags=flags,
            magic=magic,
        )
        elapsed = time.monotonic() - t0
        counts = Counter(record["format"] for record in object_records)
        report: dict[str, Any] = {
            "schema": "omlx-m5-conversion-report-v1",
            "artifact": str(output_path),
            "artifact_version": artifact_version,
            "container_flags": flags,
            "artifact_bytes": output_path.stat().st_size,
            "artifact_sha256": artifact_sha256,
            "manifest_crc32": f"{manifest_crc:08x}",
            "manifest_sha256": manifest_sha256,
            "payload_bytes": payload_bytes,
            "object_count": len(object_records),
            "binding_count": len(bindings),
            "fwht64_objects": fwht64_count,
            "formats": dict(sorted(counts.items())),
            "skipped_tensors": skipped_tensors,
            "repack_plan_summary": repack_plan_summary,
            "source_repo": source_repo,
            "source_revision": source_revision,
            "source_profile": detected,
            "source_config_sha256": source_config_sha256,
            "source_index_sha256": source_index_sha256,
            "elapsed_seconds": round(elapsed, 3),
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
        if nax_pretile:
            report["pretiled_objects"] = pretiled_count
        if total_sig_sq > 0.0:
            sqnr_db = 10.0 * math.log10(total_sig_sq / max(total_sse, 1e-30))
            report["fwht64_sqnr_db"] = round(sqnr_db, 4)
        if write_report:
            _write_conversion_report(output_path, report)
        return report
    finally:
        if payload_fd >= 0:
            os.close(payload_fd)
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
        layout = record.get("layout")
        is_nax_v1 = layout == "oq_nax_tiled_g64_v1"
        is_nax_v2 = layout in _AFFINE_NAX_V2_LAYOUTS
        if is_nax_v1 or is_nax_v2:
            v2_gs_ok = (
                not is_nax_v2
                or (layout == "oq_nax_lane_g64_v2" and group_size == 64)
                or (layout == "oq_nax_lane_g128_v2" and group_size == 128)
            )
            if (
                not v2_gs_ok
                or len(record["shape"]) != 2
                or bits not in (3, 4, 5, 6, 8)
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
            elif bits == 8:
                planes = {"nax_codes": padded_n * logical_k}
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


def _validate_object_contract(
    record: dict, expected_id: int, *, container_version: int = VERSION_V2
) -> None:
    if record.get("id") != expected_id or not record.get("name"):
        raise ValueError("artifact object identity is invalid")
    if record["object_offset"] % OBJECT_ALIGNMENT:
        raise ValueError("artifact object alignment is invalid")
    if not record.get("source_tensors") or not record.get("planes"):
        raise ValueError("artifact object directory is incomplete")
    if not _is_lower_hex(record.get("object_sha256"), 64):
        raise ValueError("artifact object checksum metadata is invalid")
    input_transform = record.get("input_transform", "none")
    if not isinstance(input_transform, str) or input_transform not in ("none", "fwht64"):
        raise ValueError(f"unsupported artifact input_transform: {input_transform!r}")
    if input_transform == "fwht64":
        if container_version < VERSION_V2:
            raise ValueError("input_transform=fwht64 requires container version 2")
        if (
            record.get("format", "").startswith("direct_")
            or len(record.get("shape", [])) < 2
            or int(record["shape"][-1]) < 64
            or (int(record["shape"][-1]) % 64) != 0
        ):
            raise ValueError("invalid geometry for input_transform=fwht64")
    if (
        record.get("layout") in _AFFINE_NAX_V2_LAYOUTS
        and container_version < VERSION_V2
    ):
        raise ValueError("v2 NAX lane layout requires container version 2")
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
            else ("oq_rowsplit_k128_v1", *_AFFINE_NAX_ALL_LAYOUTS)
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
        valid_version = version in (VERSION_V1, VERSION_V2)
        valid_flags = (
            (flags == 0)
            if version == VERSION_V1
            else ((flags & ~VALID_V2_FLAGS_MASK) == 0)
        )
        if (
            magic not in VALID_MAGICS
            or not valid_version
            or header_bytes != HEADER_BYTES
            or not valid_flags
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
        valid_schemas = VALID_SCHEMAS_V2 if version == VERSION_V2 else VALID_SCHEMAS_V1
        if (
            manifest.get("schema") not in valid_schemas
            or manifest.get("container_version") != version
            or manifest.get("payload_bytes") != payload_bytes
            or not isinstance(manifest.get("model"), dict)
            or not isinstance(manifest.get("objects"), list)
            or not isinstance(manifest.get("views"), list)
            or not isinstance(manifest.get("bindings"), list)
            or len(manifest["objects"]) != object_count
            or (
                version == VERSION_V2
                and "format_features" in manifest
                and not isinstance(manifest["format_features"], dict)
            )
        ):
            raise ValueError("artifact manifest contract mismatch")
        if manifest["views"]:
            raise ValueError("artifact converter emitted unsupported explicit views")
        _verify_zero_range(
            handle,
            manifest_offset + manifest_bytes,
            payload_offset - manifest_offset - manifest_bytes,
            "artifact manifest padding",
        )
        previous_end = 0
        object_names: set[str] = set()
        for expected_id, record in enumerate(manifest["objects"]):
            _validate_object_contract(record, expected_id, container_version=version)
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
        primary_bound_ids: set[int] = set()
        bound_names: set[str] = set()
        for binding in manifest["bindings"]:
            object_id = binding.get("object_id")
            if not isinstance(object_id, int) or not 0 <= object_id < object_count:
                raise ValueError("artifact binding object id is invalid")
            record = manifest["objects"][object_id]
            bname = binding.get("name")
            if not isinstance(bname, str) or not bname or bname in bound_names:
                raise ValueError("artifact binding is invalid")
            is_tied_lm_head_alias = (
                binding.get("use") == "lm_head"
                and infer_weight_use(bname) == "lm_head"
                and infer_weight_use(str(record["name"]).removesuffix(".weight"))
                == "embedding"
            )
            if len(record["shape"]) < 2:
                raise ValueError("artifact binding is invalid")
            if (
                (not is_tied_lm_head_alias and (object_id in primary_bound_ids or bname != record["name"] or record["format"].startswith("direct_")))
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
            bound_names.add(bname)
            if not is_tied_lm_head_alias:
                primary_bound_ids.add(object_id)
        quantized_ids = {
            record["id"]
            for record in manifest["objects"]
            if not record["format"].startswith("direct_")
        }
        if primary_bound_ids != quantized_ids:
            raise ValueError("artifact quantized binding coverage is incomplete")
        summary = {
            "container_magic": magic.decode("ascii"),
            "schema": manifest["schema"],
            "container_version": version,
            "container_flags": flags,
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
        if "format_features" in manifest:
            summary["format_features"] = manifest["format_features"]
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
    *,
    logical_unrotated: bool = True,
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
    unrotate = logical_unrotated and record.get("input_transform") == "fwht64"
    if unrotate:
        if logical_k < 64 or (logical_k % 64) != 0:
            raise ValueError("artifact fwht64 decode requires K >= 64 and divisible by 64")
        decode_start = (start // 64) * 64
        decode_end = ((start + count + 63) // 64) * 64
        decode_count = decode_end - decode_start
    else:
        decode_start = start
        decode_count = count

    def _finalize(vals: list[float]) -> list[float]:
        if not unrotate:
            return vals
        arr = np.asarray(vals, dtype=np.float32)
        fwht64_f32_inplace(arr)
        off = start - decode_start
        return arr[off : off + count].tolist()

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
            layout = record.get("layout")
            is_nax_v2 = layout in _AFFINE_NAX_V2_LAYOUTS
            if layout == "oq_nax_tiled_g64_v1" or is_nax_v2:
                v2_gs_ok = (
                    not is_nax_v2
                    or (layout == "oq_nax_lane_g64_v2" and group_size == 64)
                    or (layout == "oq_nax_lane_g128_v2" and group_size == 128)
                )
                if (
                    not v2_gs_ok
                    or len(record["shape"]) != 2
                    or bits not in (3, 4, 5, 6, 8)
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
                for column in range(decode_start, decode_start + decode_count):
                    g = column // 64
                    col_in_g = column % 64
                    slot = pi64_slot(col_in_g) if is_nax_v2 else col_in_g
                    tile_idx = tile_row * groups + g
                    if bits == 3:
                        c_byte = read_plane(
                            "nax_codes",
                            tile_idx * (bn * 16) + rn * 16 + slot // 4,
                            1,
                        )[0]
                        h_byte = read_plane(
                            "nax_high1",
                            tile_idx * (bn * 8) + rn * 8 + slot // 8,
                            1,
                        )[0]
                        code = ((c_byte >> ((slot & 3) * 2)) & 0x3) | (
                            ((h_byte >> (slot & 7)) & 0x1) << 2
                        )
                    elif bits == 8:
                        code = read_plane(
                            "nax_codes",
                            tile_idx * (bn * 64) + rn * 64 + slot,
                            1,
                        )[0]
                    else:
                        c_byte = read_plane(
                            "nax_codes",
                            tile_idx * (bn * 32) + rn * 32 + slot // 2,
                            1,
                        )[0]
                        code = (c_byte >> ((slot & 1) * 4)) & 0xF
                        if bits in (5, 6):
                            h_byte = read_plane(
                                "nax_high2",
                                tile_idx * (bn * 16) + rn * 16 + slot // 4,
                                1,
                            )[0]
                            h2 = (h_byte >> ((slot & 3) * 2)) & 0x3
                            code |= (h2 & 1 if bits == 5 else h2) << 4
                    sb_raw = read_plane("nax_sb", (g * padded_n + row) * 4, 4)
                    scale = _bf16_to_float(sb_raw[:2])
                    bias = _bf16_to_float(sb_raw[2:4])
                    values.append(scale * code + bias)
                return _finalize(values)
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
            for column in range(decode_start, decode_start + decode_count):
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
            return _finalize(values)
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
                for column in range(decode_start, decode_start + decode_count):
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
                return _finalize(values)
            data_row = padded_k // 2
            scale_row = padded_k // 32
            data = read_plane("data", row * data_row, data_row)
            scales = read_plane("scale", row * scale_row, scale_row)
            for column in range(decode_start, decode_start + decode_count):
                code = (data[column // 2] >> ((column & 1) * 4)) & 0xF
                values.append(
                    _e2m1_to_float(code) * _e8m0_to_float(scales[column // 32])
                )
            return _finalize(values)
    raise ValueError(f"artifact object is not decodable: {record['format']}")


def _untile_affine_nax_chunk_codes_u8(
    src_fd: int,
    src_payload_offset: int,
    planes_by_kind: dict[str, dict],
    bits: int,
    bn: int,
    groups: int,
    tile_r0: int,
    tile_r1: int,
    lane_v2: bool,
) -> np.ndarray:
    """Read tile rows [tile_r0, tile_r1) from NAX tiled affine planes -> uint8 [(tile_r1-tile_r0)*bn, groups*64]."""
    n_tiles = tile_r1 - tile_r0
    rows_chunk = n_tiles * bn
    k = groups * 64
    c_off = src_payload_offset + int(planes_by_kind["nax_codes"]["offset"])
    if bits == 8:
        tile_b = bn * 64
        raw = np.frombuffer(
            _pread_all(src_fd, n_tiles * groups * tile_b, c_off + tile_r0 * groups * tile_b),
            dtype=np.uint8,
        ).reshape(n_tiles, groups, bn, 64)
        if lane_v2:
            raw = _permute_pi64_q8(raw)
        return np.ascontiguousarray(raw.transpose(0, 2, 1, 3).reshape(rows_chunk, k))
    if bits == 3:
        tile_b = bn * 16
        l2 = np.frombuffer(
            _pread_all(src_fd, n_tiles * groups * tile_b, c_off + tile_r0 * groups * tile_b),
            dtype=np.uint8,
        ).reshape(n_tiles, groups, bn, 16)
        h_off = src_payload_offset + int(planes_by_kind["nax_high1"]["offset"])
        h_tile_b = bn * 8
        h1 = np.frombuffer(
            _pread_all(src_fd, n_tiles * groups * h_tile_b, h_off + tile_r0 * groups * h_tile_b),
            dtype=np.uint8,
        ).reshape(n_tiles, groups, bn, 8)
        if lane_v2:
            l2 = _permute_pi64_2bit(l2)
            h1 = _permute_pi64_1bit(h1)
        l2_flat = np.ascontiguousarray(l2.transpose(0, 2, 1, 3).reshape(rows_chunk, k // 4))
        h1_flat = np.ascontiguousarray(h1.transpose(0, 2, 1, 3).reshape(rows_chunk, k // 8))
        return _unpack_rowsplit_affine_codes_u8(l2_flat, h1_flat, k, 3)
    # bits in (4, 5, 6)
    tile_b = bn * 32
    base = np.frombuffer(
        _pread_all(src_fd, n_tiles * groups * tile_b, c_off + tile_r0 * groups * tile_b),
        dtype=np.uint8,
    ).reshape(n_tiles, groups, bn, 32)
    if lane_v2:
        base = _permute_pi64_nibbles(base)
    base_flat = np.ascontiguousarray(base.transpose(0, 2, 1, 3).reshape(rows_chunk, k // 2))
    out = np.empty((rows_chunk, k), dtype=np.uint8)
    out[:, 0::2] = base_flat & np.uint8(0x0F)
    out[:, 1::2] = base_flat >> np.uint8(4)
    if bits in (5, 6):
        h_off = src_payload_offset + int(planes_by_kind["nax_high2"]["offset"])
        h_tile_b = bn * 16
        h2 = np.frombuffer(
            _pread_all(src_fd, n_tiles * groups * h_tile_b, h_off + tile_r0 * groups * h_tile_b),
            dtype=np.uint8,
        ).reshape(n_tiles, groups, bn, 16)
        if lane_v2:
            h2 = _permute_pi64_2bit(h2)
        h2_flat = np.ascontiguousarray(h2.transpose(0, 2, 1, 3).reshape(rows_chunk, k // 4))
        mask = np.uint8(1 if bits == 5 else 3)
        out[:, 0::4] |= (h2_flat & mask) << np.uint8(4)
        out[:, 1::4] |= ((h2_flat >> np.uint8(2)) & mask) << np.uint8(4)
        out[:, 2::4] |= ((h2_flat >> np.uint8(4)) & mask) << np.uint8(4)
        out[:, 3::4] |= ((h2_flat >> np.uint8(6)) & mask) << np.uint8(4)
    return out


def pretile_artifact(
    input_path: Path,
    output_path: Path,
    *,
    artifact_version: int | None = None,
    fwht64: bool | None = None,
    fwht64_o_proj: bool = True,
    jobs: int = DEFAULT_JOBS,
    legacy_magic: bool = False,
) -> dict:
    t0 = time.monotonic()
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("pretile output must differ from input artifact path")
    src_summary, src_manifest = read_artifact(input_path, verify_payload=False)
    src_version = int(src_summary["container_version"])
    target_version = src_version if artifact_version is None else int(artifact_version)
    src_profile = str(src_manifest.get("model", {}).get("source_profile", ""))
    eff_fwht64 = resolve_fwht64_enabled(target_version, fwht64, src_profile)
    max_workers = max(1, int(jobs))

    if target_version == VERSION_V1 and any(
        obj.get("input_transform") == "fwht64" for obj in src_manifest["objects"]
    ):
        raise ValueError("cannot downgrade fwht64-rotated v2 artifact to v1")

    embedding_obj_ids = {
        int(b["object_id"])
        for b in src_manifest.get("bindings", [])
        if b.get("role") == "embedding" or b.get("use") == "embedding"
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload_fd, payload_name = tempfile.mkstemp(
        prefix=output_path.name + ".payload.", dir=output_path.parent
    )

    planned_tasks: list[dict[str, Any]] = []
    cursor = 0
    pretiled_count = 0
    for obj in src_manifest["objects"]:
        obj_id = int(obj["id"])
        name = str(obj["name"])
        fmt = str(obj["format"])
        src_layout = str(obj["layout"])
        shape = list(obj["shape"])
        padded_k = obj.get("padded_k")
        is_embed = (obj_id in embedding_obj_ids) or (infer_weight_use(name) == "embedding")
        target_layout, can_pretile = _target_layout_for_object(
            name,
            fmt,
            src_layout,
            shape,
            nax_pretile=True,
            artifact_version=target_version,
            is_embedding=is_embed,
        )
        if can_pretile and (
            src_layout in ("mx_rowsplit_k128_v1", "oq_rowsplit_k128_v1")
            or src_layout != target_layout
        ):
            pretiled_count += 1
        already_rot = obj.get("input_transform", "none") == "fwht64"
        do_rot = (not already_rot) and is_fwht64_eligible(
            name,
            shape,
            fmt,
            profile=src_profile,
            fwht64=eff_fwht64,
            fwht64_o_proj=fwht64_o_proj,
        )
        input_transform = "fwht64" if (already_rot or do_rot) else "none"
        direct_nbytes = int(obj["planes"][0]["bytes"]) if fmt.startswith("direct_") else None
        plane_sizes = _plan_plane_specs(fmt, target_layout, shape, padded_k, direct_nbytes)
        cursor = align_up(cursor, OBJECT_ALIGNMENT)
        object_begin = cursor
        plane_layouts: list[tuple[str, int, int, int]] = []
        for p_kind, p_bytes in plane_sizes:
            cursor = align_up(cursor, PLANE_ALIGNMENT)
            plane_layouts.append((p_kind, cursor, p_bytes, PLANE_ALIGNMENT))
            cursor += p_bytes
        object_bytes = cursor - object_begin
        planned_tasks.append(
            {
                "obj": obj,
                "src_layout": src_layout,
                "target_layout": target_layout,
                "do_rot": do_rot,
                "input_transform": input_transform,
                "object_begin": object_begin,
                "object_bytes": object_bytes,
                "plane_layouts": plane_layouts,
            }
        )
    payload_bytes = cursor

    src_fd = os.open(input_path, os.O_RDONLY)
    try:
        src_header = _pread_all(src_fd, HEADER_BYTES, 0)
        src_payload_offset = HEADER_STRUCT.unpack(src_header)[5]
        os.ftruncate(payload_fd, payload_bytes)

        def _execute_pretile_one(task: dict[str, Any]) -> tuple[int, list[dict], str, float, float]:
            obj = task["obj"]
            oid = int(obj["id"])
            name = str(obj["name"])
            fmt = str(obj["format"])
            src_layout = task["src_layout"]
            target_layout = task["target_layout"]
            do_rot = bool(task["do_rot"])
            plane_layouts = task["plane_layouts"]
            planes_by_kind = {p["kind"]: p for p in obj["planes"]}

            # Fast passthrough copy when layout and rotation are unchanged
            if target_layout == src_layout and not do_rot:
                new_planes: list[dict] = []
                for kind, p_off, p_bytes, p_align in plane_layouts:
                    src_p = planes_by_kind[kind]
                    s_off = src_payload_offset + int(src_p["offset"])
                    w = _IncrementalPlaneWriter(payload_fd, kind, p_off, p_bytes, p_align)
                    rem = p_bytes
                    pos = 0
                    while rem:
                        chunk_sz = min(rem, 16 * 1024 * 1024)
                        w.write(_pread_all(src_fd, chunk_sz, s_off + pos))
                        pos += chunk_sz
                        rem -= chunk_sz
                    new_planes.append(w.finish(name))
                if len(new_planes) == 1 and new_planes[0]["offset"] == task["object_begin"] and new_planes[0]["bytes"] == task["object_bytes"]:
                    obj_sha = new_planes[0]["sha256"]
                else:
                    obj_sha = _sha256_fd_range(payload_fd, task["object_begin"], task["object_bytes"])
                return oid, new_planes, obj_sha, 0.0, 0.0

            shape = [int(v) for v in obj["shape"]]
            rows = product(shape[:-1])
            logical_k = shape[-1]
            padded_k = int(obj["padded_k"])
            writers = {
                kind: _IncrementalPlaneWriter(payload_fd, kind, off, nbytes, align)
                for kind, off, nbytes, align in plane_layouts
            }
            total_sse = 0.0
            total_sig_sq = 0.0

            if fmt == "mxfp4_e2m1_g32_e8m0_v1":
                dst_tiled = target_layout == "mx_nax_tiled_bk128_v1"
                src_tiled = src_layout == "mx_nax_tiled_bk128_v1"
                bn = 32 if rows >= 65536 else 16
                k_blocks = logical_k // 128
                chunk_rows = max(bn, (256 // bn) * bn) if do_rot else max(bn, (1024 // bn) * bn)
                if src_tiled:
                    d_off = src_payload_offset + int(planes_by_kind["nax_data"]["offset"])
                    s_off = src_payload_offset + int(planes_by_kind["nax_scale"]["offset"])
                else:
                    d_off = src_payload_offset + int(planes_by_kind["data"]["offset"])
                    s_off = src_payload_offset + int(planes_by_kind["scale"]["offset"])
                for r0 in range(0, rows, chunk_rows):
                    r1 = min(r0 + chunk_rows, rows)
                    n_chunk = r1 - r0
                    padded_chunk = align_up(n_chunk, bn) if dst_tiled else n_chunk
                    if src_tiled:
                        t0_i = r0 // bn
                        t1_i = align_up(r1, bn) // bn
                        nt = t1_i - t0_i
                        tc = np.frombuffer(
                            _pread_all(src_fd, nt * k_blocks * bn * 64, d_off + t0_i * k_blocks * bn * 64),
                            dtype=np.uint8,
                        ).reshape(nt, k_blocks, bn, 64)
                        ts = np.frombuffer(
                            _pread_all(src_fd, nt * k_blocks * bn * 4, s_off + t0_i * k_blocks * bn * 4),
                            dtype=np.uint8,
                        ).reshape(nt, k_blocks, bn, 4)
                        raw_c = np.ascontiguousarray(tc.transpose(0, 2, 1, 3).reshape(nt * bn, logical_k // 2))[:n_chunk]
                        raw_s = np.ascontiguousarray(ts.transpose(0, 2, 1, 3).reshape(nt * bn, logical_k // 32))[:n_chunk]
                    else:
                        raw_c = np.frombuffer(
                            _pread_all(src_fd, n_chunk * (padded_k // 2), d_off + r0 * (padded_k // 2)),
                            dtype=np.uint8,
                        ).reshape(n_chunk, padded_k // 2)[:, : logical_k // 2]
                        raw_s = np.frombuffer(
                            _pread_all(src_fd, n_chunk * (padded_k // 32), s_off + r0 * (padded_k // 32)),
                            dtype=np.uint8,
                        ).reshape(n_chunk, padded_k // 32)[:, : logical_k // 32]
                    if do_rot:
                        w_f32 = _dequantize_mxfp4_u8_f32(raw_c, raw_s)
                        fwht64_f32_inplace(w_f32)
                        raw_c, raw_s, sse, sig_sq = quantize_mxfp4_e2m1_f32(w_f32)
                        total_sse += sse
                        total_sig_sq += sig_sq
                    if dst_tiled:
                        if padded_chunk > n_chunk:
                            pad_c = np.zeros((padded_chunk, logical_k // 2), dtype=np.uint8)
                            pad_c[:n_chunk] = raw_c
                            raw_c = pad_c
                            pad_s = np.full((padded_chunk, logical_k // 32), 0x7F, dtype=np.uint8)
                            pad_s[:n_chunk] = raw_s
                            raw_s = pad_s
                        nt = padded_chunk // bn
                        writers["nax_data"].write(
                            np.ascontiguousarray(raw_c.reshape(nt, bn, k_blocks, 64).transpose(0, 2, 1, 3))
                        )
                        writers["nax_scale"].write(
                            np.ascontiguousarray(raw_s.reshape(nt, bn, k_blocks, 4).transpose(0, 2, 1, 3))
                        )
                    else:
                        if padded_k != logical_k:
                            c_pad = np.zeros((n_chunk, padded_k // 2), dtype=np.uint8)
                            c_pad[:, : logical_k // 2] = raw_c
                            s_pad = np.zeros((n_chunk, padded_k // 32), dtype=np.uint8)
                            s_pad[:, : logical_k // 32] = raw_s
                            raw_c, raw_s = c_pad, s_pad
                        writers["data"].write(raw_c)
                        writers["scale"].write(raw_s)
                plane_records = [writers[kind].finish(name) for kind, _, _, _ in plane_layouts]
                obj_sha = _sha256_fd_range(payload_fd, task["object_begin"], task["object_bytes"])
                return oid, plane_records, obj_sha, total_sse, total_sig_sq

            affine = _parse_affine_format(fmt)
            if affine is None:
                raise ValueError(f"unsupported pretile format {fmt}")
            bits, src_gs = affine
            dst_tiled = target_layout in _AFFINE_NAX_ALL_LAYOUTS
            dst_lane_v2 = target_layout in _AFFINE_NAX_V2_LAYOUTS
            src_tiled = src_layout in _AFFINE_NAX_ALL_LAYOUTS
            src_lane_v2 = src_layout in _AFFINE_NAX_V2_LAYOUTS
            bn = 32 if rows >= 65536 else 16
            padded_n = align_up(rows, bn)
            groups64 = logical_k // 64
            src_groups = logical_k // src_gs

            # Fast zero-dequant permutation when converting v1 tiled -> v2 lane without FWHT-64
            if src_tiled and dst_tiled and not do_rot and src_lane_v2 != dst_lane_v2:
                total_tiles = (padded_n // bn) * groups64
                if bits == 8:
                    raw = np.frombuffer(
                        _pread_all(src_fd, int(planes_by_kind["nax_codes"]["bytes"]), src_payload_offset + int(planes_by_kind["nax_codes"]["offset"])),
                        dtype=np.uint8,
                    ).reshape(total_tiles, bn, 64)
                    writers["nax_codes"].write(_permute_pi64_q8(raw))
                elif bits == 3:
                    l2 = np.frombuffer(
                        _pread_all(src_fd, int(planes_by_kind["nax_codes"]["bytes"]), src_payload_offset + int(planes_by_kind["nax_codes"]["offset"])),
                        dtype=np.uint8,
                    ).reshape(total_tiles, bn, 16)
                    h1 = np.frombuffer(
                        _pread_all(src_fd, int(planes_by_kind["nax_high1"]["bytes"]), src_payload_offset + int(planes_by_kind["nax_high1"]["offset"])),
                        dtype=np.uint8,
                    ).reshape(total_tiles, bn, 8)
                    writers["nax_codes"].write(_permute_pi64_2bit(l2))
                    writers["nax_high1"].write(_permute_pi64_1bit(h1))
                else:
                    c = np.frombuffer(
                        _pread_all(src_fd, int(planes_by_kind["nax_codes"]["bytes"]), src_payload_offset + int(planes_by_kind["nax_codes"]["offset"])),
                        dtype=np.uint8,
                    ).reshape(total_tiles, bn, 32)
                    writers["nax_codes"].write(_permute_pi64_nibbles(c))
                    if bits in (5, 6):
                        h2 = np.frombuffer(
                            _pread_all(src_fd, int(planes_by_kind["nax_high2"]["bytes"]), src_payload_offset + int(planes_by_kind["nax_high2"]["offset"])),
                            dtype=np.uint8,
                        ).reshape(total_tiles, bn, 16)
                        writers["nax_high2"].write(_permute_pi64_2bit(h2))
                sb_p = planes_by_kind["nax_sb"]
                writers["nax_sb"].write(
                    _pread_all(src_fd, int(sb_p["bytes"]), src_payload_offset + int(sb_p["offset"]))
                )
                plane_records = [writers[kind].finish(name) for kind, _, _, _ in plane_layouts]
                obj_sha = _sha256_fd_range(payload_fd, task["object_begin"], task["object_bytes"])
                return oid, plane_records, obj_sha, 0.0, 0.0

            sb_buf = np.zeros((groups64, padded_n), dtype=np.uint32) if dst_tiled else None
            src_sb_u32 = None
            if src_tiled:
                sb_p = planes_by_kind["nax_sb"]
                src_sb_u32 = np.frombuffer(
                    _pread_all(src_fd, int(sb_p["bytes"]), src_payload_offset + int(sb_p["offset"])),
                    dtype=np.uint32,
                ).reshape(groups64, padded_n)
            else:
                code_kind = "code" if bits == 8 else "low_2bit" if bits == 3 else "base_code"
                code_row_b = padded_k if bits == 8 else padded_k // 4 if bits == 3 else padded_k // 2
                high_kind = "high_bit" if bits in (3, 5) else "high_2bit" if bits == 6 else None
                high_row_b = padded_k // (8 if bits in (3, 5) else 4)
                sb_row_b = (padded_k // src_gs) * 2
                c_off = src_payload_offset + int(planes_by_kind[code_kind]["offset"])
                h_off = (
                    src_payload_offset + int(planes_by_kind[high_kind]["offset"])
                    if high_kind is not None
                    else 0
                )
                sc_off = src_payload_offset + int(planes_by_kind["scale"]["offset"])
                bi_off = src_payload_offset + int(planes_by_kind["bias"]["offset"])

            chunk_rows = max(bn, (256 // bn) * bn) if do_rot else max(bn, (1024 // bn) * bn)
            target_groups = padded_k // src_gs
            for r0 in range(0, rows, chunk_rows):
                r1 = min(r0 + chunk_rows, rows)
                n_chunk = r1 - r0
                padded_chunk = align_up(n_chunk, bn) if dst_tiled else n_chunk

                if src_tiled:
                    t0_i = r0 // bn
                    t1_i = align_up(r1, bn) // bn
                    codes = _untile_affine_nax_chunk_codes_u8(
                        src_fd, src_payload_offset, planes_by_kind, bits, bn, groups64, t0_i, t1_i, src_lane_v2
                    )[:n_chunk]
                    assert src_sb_u32 is not None
                    sb_slice = src_sb_u32[:, r0:r1].T  # [n_chunk, groups64]
                    if src_gs == 128:
                        sb_slice = sb_slice[:, 0::2]
                    sc_u16 = np.ascontiguousarray((sb_slice & np.uint32(0xFFFF)).astype(np.uint16))
                    bi_u16 = np.ascontiguousarray((sb_slice >> np.uint32(16)).astype(np.uint16))
                else:
                    base_u8 = np.frombuffer(
                        _pread_all(src_fd, n_chunk * code_row_b, c_off + r0 * code_row_b),
                        dtype=np.uint8,
                    ).reshape(n_chunk, code_row_b)
                    high_u8 = (
                        np.frombuffer(
                            _pread_all(src_fd, n_chunk * high_row_b, h_off + r0 * high_row_b),
                            dtype=np.uint8,
                        ).reshape(n_chunk, high_row_b)
                        if high_kind is not None
                        else None
                    )
                    codes = _unpack_rowsplit_affine_codes_u8(base_u8, high_u8, logical_k, bits)
                    sc_u16 = np.frombuffer(
                        _pread_all(src_fd, n_chunk * sb_row_b, sc_off + r0 * sb_row_b),
                        dtype=np.uint16,
                    ).reshape(n_chunk, padded_k // src_gs)[:, :src_groups]
                    bi_u16 = np.frombuffer(
                        _pread_all(src_fd, n_chunk * sb_row_b, bi_off + r0 * sb_row_b),
                        dtype=np.uint16,
                    ).reshape(n_chunk, padded_k // src_gs)[:, :src_groups]

                if do_rot:
                    sc_f32 = _bf16_u16_to_float32(sc_u16)
                    bi_f32 = _bf16_u16_to_float32(bi_u16)
                    w_f32 = (
                        codes.astype(np.float32).reshape(n_chunk, src_groups, src_gs)
                        * sc_f32[:, :, None]
                        + bi_f32[:, :, None]
                    ).reshape(n_chunk, logical_k)
                    w_f32 = np.ascontiguousarray(w_f32, dtype=np.float32)
                    fwht64_f32_inplace(w_f32)
                    codes, sc_u16, bi_u16, sse, sig_sq = quantize_oq_affine_f32(
                        w_f32, bits, src_gs
                    )
                    total_sse += sse
                    total_sig_sq += sig_sq

                if dst_tiled:
                    if padded_chunk > n_chunk:
                        pad_c = np.zeros((padded_chunk, logical_k), dtype=np.uint8)
                        pad_c[:n_chunk] = codes
                        codes = pad_c
                    for kind, arr in _pack_nax_affine_code_tiles(codes, bits, bn, dst_lane_v2):
                        writers[kind].write(arr)
                    sc_64 = np.repeat(sc_u16, 2, axis=1) if src_gs == 128 else sc_u16
                    bi_64 = np.repeat(bi_u16, 2, axis=1) if src_gs == 128 else bi_u16
                    packed_sb = sc_64.astype(np.uint32) | (bi_64.astype(np.uint32) << np.uint32(16))
                    assert sb_buf is not None
                    sb_buf[:, r0:r1] = packed_sb.T
                else:
                    for kind, arr in _pack_rowsplit_affine_planes_from_codes(codes, bits, padded_k):
                        writers[kind].write(arr)
                    if target_groups != src_groups:
                        sc_pad = np.zeros((n_chunk, target_groups), dtype=np.uint16)
                        bi_pad = np.zeros((n_chunk, target_groups), dtype=np.uint16)
                        sc_pad[:, :src_groups] = sc_u16
                        bi_pad[:, :src_groups] = bi_u16
                        sc_u16, bi_u16 = sc_pad, bi_pad
                    writers["scale"].write(sc_u16)
                    writers["bias"].write(bi_u16)

            if dst_tiled:
                assert sb_buf is not None
                writers["nax_sb"].write(sb_buf)
            plane_records = [writers[kind].finish(name) for kind, _, _, _ in plane_layouts]
            obj_sha = _sha256_fd_range(payload_fd, task["object_begin"], task["object_bytes"])
            return oid, plane_records, obj_sha, total_sse, total_sig_sq

        results_by_id: dict[int, tuple[list[dict], str, float, float]] = {}
        effective_workers = min(max_workers, max(1, len(planned_tasks)))
        if effective_workers <= 1:
            for task in planned_tasks:
                oid, planes, obj_sha, sse, sig_sq = _execute_pretile_one(task)
                results_by_id[oid] = (planes, obj_sha, sse, sig_sq)
        else:
            ordered = sorted(
                planned_tasks,
                key=lambda item: (item["do_rot"], item["object_bytes"]),
                reverse=True,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=effective_workers) as pool:
                futures = [pool.submit(_execute_pretile_one, task) for task in ordered]
                for fut in concurrent.futures.as_completed(futures):
                    oid, planes, obj_sha, sse, sig_sq = fut.result()
                    results_by_id[oid] = (planes, obj_sha, sse, sig_sq)

        os.fsync(payload_fd)
        os.close(payload_fd)
        payload_fd = -1

        new_objects: list[dict] = []
        total_sse = 0.0
        total_sig_sq = 0.0
        fwht64_count = 0
        for task in planned_tasks:
            obj = task["obj"]
            oid = int(obj["id"])
            plane_records, obj_sha, sse, sig_sq = results_by_id[oid]
            total_sse += sse
            total_sig_sq += sig_sq
            rec = {
                **obj,
                "layout": task["target_layout"],
                "object_offset": task["object_begin"],
                "object_bytes": task["object_bytes"],
                "object_sha256": obj_sha,
                "planes": plane_records,
            }
            if task["input_transform"] == "fwht64":
                rec["input_transform"] = "fwht64"
                fwht64_count += 1
            else:
                rec.pop("input_transform", None)
            new_objects.append(rec)

        magic, schema_id, converter_name = _branding_for_write(
            target_version, legacy_magic=legacy_magic
        )
        converter_meta = dict(
            src_manifest.get(
                "converter",
                {"name": converter_name, "version": target_version, "endianness": "little"},
            )
        )
        converter_meta["name"] = converter_name
        converter_meta["version"] = target_version
        converter_meta.setdefault("endianness", "little")
        manifest: dict[str, Any] = {
            **src_manifest,
            "schema": schema_id,
            "container_version": target_version,
            "converter": converter_meta,
            "objects": new_objects,
            "payload_bytes": payload_bytes,
        }
        if target_version == VERSION_V2:
            flags, format_features = _compute_v2_flags_and_features(
                new_objects, fwht64_o_proj=fwht64_o_proj
            )
            manifest["format_features"] = format_features
        else:
            flags = 0
            manifest.pop("format_features", None)

        art_sha, manifest_crc, manifest_sha256 = _write_artifact_container(
            output_path,
            manifest,
            payload_name,
            artifact_version=target_version,
            flags=flags,
            magic=magic,
        )
    finally:
        os.close(src_fd)
        if payload_fd >= 0:
            os.close(payload_fd)
        if os.path.exists(payload_name):
            os.unlink(payload_name)

    elapsed = time.monotonic() - t0
    out_summary: dict[str, Any] = {
        "artifact": str(output_path),
        "artifact_version": target_version,
        "container_flags": flags,
        "artifact_bytes": output_path.stat().st_size,
        "artifact_sha256": art_sha,
        "manifest_crc32": f"{manifest_crc:08x}",
        "manifest_sha256": manifest_sha256,
        "payload_bytes": payload_bytes,
        "object_count": len(new_objects),
        "pretiled_objects": pretiled_count,
        "fwht64_objects": fwht64_count,
        "elapsed_seconds": round(elapsed, 3),
    }
    if total_sig_sq > 0.0:
        out_summary["fwht64_sqnr_db"] = round(
            10.0 * math.log10(total_sig_sq / max(total_sse, 1e-30)), 4
        )
    return out_summary


def _finalize_pretiled_conversion_report(
    report: dict, output_path: Path, pt: dict
) -> dict:
    output_path = output_path.resolve()
    _, manifest = read_artifact(output_path, verify_payload=False)
    updated = {
        **report,
        "artifact": str(output_path),
        "artifact_version": pt.get("artifact_version", report.get("artifact_version", VERSION_V1)),
        "container_flags": pt.get("container_flags", report.get("container_flags", 0)),
        "artifact_bytes": pt["artifact_bytes"],
        "artifact_sha256": pt["artifact_sha256"],
        "manifest_crc32": pt["manifest_crc32"],
        "manifest_sha256": pt["manifest_sha256"],
        "payload_bytes": pt["payload_bytes"],
        "object_count": pt["object_count"],
        "pretiled_objects": pt["pretiled_objects"],
        "fwht64_objects": pt.get("fwht64_objects", 0),
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
    if "fwht64_sqnr_db" in pt:
        updated["fwht64_sqnr_db"] = pt["fwht64_sqnr_db"]
    _write_conversion_report(output_path, updated)
    return updated


def command_convert(args: argparse.Namespace) -> None:
    source = SafeTensorSource(args.source, args.expected_revision)
    try:
        report = write_artifact(
            source,
            args.output,
            args.source_repo,
            args.profile,
            artifact_version=int(getattr(args, "artifact_version", DEFAULT_ARTIFACT_VERSION)),
            nax_pretile=bool(getattr(args, "nax_pretile", False)),
            fwht64=getattr(args, "fwht64", None),
            fwht64_o_proj=bool(getattr(args, "fwht64_o_proj", True)),
            jobs=int(getattr(args, "jobs", DEFAULT_JOBS)),
            legacy_magic=bool(getattr(args, "legacy_magic", False)),
        )
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
    summary = pretile_artifact(
        args.input,
        args.output,
        artifact_version=int(getattr(args, "artifact_version", DEFAULT_ARTIFACT_VERSION)),
        fwht64=getattr(args, "fwht64", None),
        fwht64_o_proj=bool(getattr(args, "fwht64_o_proj", True)),
        jobs=int(getattr(args, "jobs", DEFAULT_JOBS)),
        legacy_magic=bool(getattr(args, "legacy_magic", False)),
    )
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
            if _is_non_text_tensor_name(name):
                continue
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
                if (module + ".biases") in source.tensors:
                    text_cfg_q, _ = _normalize_model_config(source.config)
                    _, gs0, k = _infer_module_affine_spec(
                        source.config,
                        text_cfg_q,
                        module,
                        info,
                        sc_info,
                        source.tensors[module + ".biases"],
                    )
                    leading = sc_info.shape[:-1]
                else:
                    spec = quant_spec(source.config, module)
                    gs0 = int(spec.get("group_size") or 32)
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
    artifact_version: int = VERSION_V1,
    fwht64: bool | None = None,
    fwht64_o_proj: bool = True,
    jobs: int = DEFAULT_JOBS,
    legacy_magic: bool = False,
) -> dict:
    """Apply #5292 oQe WLS+flip quantization using a QDA recipe and emit .miv (v1 or v2)."""
    eff_fwht64 = resolve_fwht64_enabled(artifact_version, fwht64, "oq_affine")
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
            prerotated_modules: set[str] = set()
            tensors_out: dict[str, tuple[str, tuple[int, ...], bytes]] = {}
            text_cfg_src, _ = _normalize_model_config(src.config)
            for name, info in sorted(src.tensors.items()):
                if _is_non_text_tensor_name(name):
                    continue
                view = src.view(name)
                raw_copy = bytes(view)
                view.release()
                if name.endswith(".weight"):
                    module = name[: -len(".weight")]
                    if (module + ".scales") in src.tensors:
                        if (module + ".biases") in src.tensors:
                            src_bits, src_gs, _ = _infer_module_affine_spec(
                                src.config,
                                text_cfg_src,
                                module,
                                info,
                                src.tensors[module + ".scales"],
                                src.tensors[module + ".biases"],
                            )
                            effective_qtable[module] = {
                                "bits": int(src_bits),
                                "group_size": int(src_gs),
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
                        w_f32 = np.ascontiguousarray(w_f32, dtype=np.float32)
                        h_diag = hessian_map.get(module)
                        fmt_id = f"oq_affine_q{bits}_g{gs}_bf16_v1"
                        if is_fwht64_eligible(
                            module,
                            info.shape,
                            fmt_id,
                            profile="oq_affine",
                            fwht64=eff_fwht64,
                            fwht64_o_proj=fwht64_o_proj,
                        ):
                            # Case 1: unquantized source -> rotate in FP32 BEFORE quantization (zero double-quant loss)
                            fwht64_f32_inplace(w_f32)
                            prerotated_modules.add(module)
                            if h_diag is not None and h_diag.size == int(info.shape[-1]):
                                h_diag = np.repeat(
                                    np.mean(h_diag.reshape(-1, 64), axis=1), 64
                                ).astype(np.float32)
                        codes, sc_u16, bi_u16, _ = quantize_affine_oqe(
                            w_f32, bits, gs, h_diag
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
                return write_artifact(
                    q_src,
                    output_artifact,
                    source_repo,
                    out_profile,
                    source_provenance=provenance,
                    artifact_version=artifact_version,
                    nax_pretile=nax_pretile,
                    fwht64=eff_fwht64,
                    fwht64_o_proj=fwht64_o_proj,
                    jobs=jobs,
                    prerotated_modules=prerotated_modules,
                    legacy_magic=legacy_magic,
                )
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
            artifact_version=int(getattr(args, "artifact_version", DEFAULT_ARTIFACT_VERSION)),
            fwht64=getattr(args, "fwht64", None),
            fwht64_o_proj=bool(getattr(args, "fwht64_o_proj", True)),
            jobs=int(getattr(args, "jobs", DEFAULT_JOBS)),
            legacy_magic=bool(getattr(args, "legacy_magic", False)),
        )
        summary["artifact"] = str(args.output_artifact.resolve())
        summary["artifact_bytes"] = art_rep["artifact_bytes"]
        summary["pretiled_objects"] = art_rep.get("pretiled_objects", 0)
        summary["fwht64_objects"] = art_rep.get("fwht64_objects", 0)
    print(json.dumps(summary, indent=2, sort_keys=True))


def command_inspect(args: argparse.Namespace) -> None:
    summary, manifest = read_artifact(args.artifact, args.verify_payload)
    if args.list_objects:
        summary["objects"] = [
            {
                "id": item["id"],
                "name": item["name"],
                "format": item["format"],
                "layout": item["layout"],
                "input_transform": item.get("input_transform", "none"),
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
                logical_unrotated=not bool(getattr(args, "raw_rotated", False)),
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
    output_path: Path,
    manifest: dict,
    payload_name: str,
    *,
    legacy_magic: bool = False,
) -> tuple[str, int, str]:
    artifact_version = int(manifest.get("container_version", VERSION_V1))
    magic, schema_id, converter_name = _branding_for_write(
        artifact_version, legacy_magic=legacy_magic
    )
    conv_meta = dict(
        manifest.get(
            "converter",
            {"name": converter_name, "version": artifact_version, "endianness": "little"},
        )
    )
    conv_meta["name"] = converter_name
    conv_meta["version"] = artifact_version
    conv_meta.setdefault("endianness", "little")
    if artifact_version == VERSION_V2:
        flags, format_features = _compute_v2_flags_and_features(manifest["objects"])
        manifest = {
            **manifest,
            "schema": schema_id,
            "container_version": VERSION_V2,
            "converter": conv_meta,
            "format_features": manifest.get("format_features", format_features),
        }
    else:
        flags = 0
        manifest = {
            **manifest,
            "schema": schema_id,
            "container_version": VERSION_V1,
            "converter": conv_meta,
        }
        manifest.pop("format_features", None)
    return _write_artifact_container(
        output_path,
        manifest,
        payload_name,
        artifact_version=artifact_version,
        flags=flags,
        magic=magic,
    )


def _copy_selected_artifact_objects(
    src_path: Path,
    src_manifest: dict,
    predicate: Callable[[str], bool],
    dst_payload: BinaryIO,
    writer: PayloadWriter,
) -> list[tuple[dict, list[dict]]]:
    bindings_by_obj: dict[int, list[dict]] = {}
    for b in src_manifest.get("bindings", []):
        bindings_by_obj.setdefault(int(b["object_id"]), []).append(dict(b))
    emitted: list[tuple[dict, list[dict]]] = []
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
            emitted.append((rec, bindings_by_obj.get(int(obj["id"]), [])))
    return emitted


def extract_mtp_artifact(
    source_path: Path,
    output_path: Path,
    source_repo: str = "",
    *,
    expected_revision: str | None = None,
    profile: str = "auto",
    nax_pretile: bool = False,
    artifact_version: int | None = VERSION_V1,
    fwht64: bool | None = None,
    fwht64_o_proj: bool = True,
    jobs: int = DEFAULT_JOBS,
    legacy_magic: bool = False,
) -> dict:
    """Extract only ModelComponent::mtp weights from a checkpoint dir or .miv into a standalone .miv."""
    source_path = source_path.resolve()
    output_path = output_path.resolve()

    if source_path.is_file():
        if source_path == output_path:
            raise ValueError("extract-mtp output must differ from source artifact path")
        src_summary, src_manifest = read_artifact(source_path, verify_payload=False)
        src_ver = int(src_summary["container_version"])
        target_ver = src_ver if artifact_version is None else int(artifact_version)
        eff_fwht64 = resolve_fwht64_enabled(
            target_ver, fwht64, str(src_manifest.get("model", {}).get("source_profile", ""))
        )
        if nax_pretile or (target_ver != src_ver) or (
            eff_fwht64
            and any(
                obj.get("input_transform", "none") != "fwht64"
                and _is_mtp_object_name(str(obj["name"]))
                and is_fwht64_eligible(
                    str(obj["name"]),
                    obj["shape"],
                    str(obj["format"]),
                    profile=str(src_manifest.get("model", {}).get("source_profile", "")),
                    fwht64=eff_fwht64,
                    fwht64_o_proj=fwht64_o_proj,
                )
                for obj in src_manifest["objects"]
            )
        ):
            tmp_out = output_path.with_suffix(output_path.suffix + ".untiled.tmp")
            try:
                rep = extract_mtp_artifact(
                    source_path,
                    tmp_out,
                    source_repo,
                    expected_revision=expected_revision,
                    profile=profile,
                    nax_pretile=False,
                    artifact_version=src_ver,
                    fwht64=False if src_ver == VERSION_V1 else None,
                    fwht64_o_proj=fwht64_o_proj,
                    jobs=jobs,
                )
                pt = pretile_artifact(
                    tmp_out,
                    output_path,
                    artifact_version=target_ver,
                    fwht64=eff_fwht64,
                    fwht64_o_proj=fwht64_o_proj,
                    jobs=jobs,
                    legacy_magic=legacy_magic,
                )
                rep["artifact"] = pt["artifact"]
                rep["artifact_version"] = pt["artifact_version"]
                rep["container_flags"] = pt["container_flags"]
                rep["artifact_bytes"] = pt["artifact_bytes"]
                rep["artifact_sha256"] = pt["artifact_sha256"]
                rep["manifest_crc32"] = pt["manifest_crc32"]
                rep["manifest_sha256"] = pt["manifest_sha256"]
                rep["payload_bytes"] = pt["payload_bytes"]
                rep["object_count"] = pt["object_count"]
                rep["pretiled_objects"] = pt["pretiled_objects"]
                rep["fwht64_objects"] = pt["fwht64_objects"]
                return rep
            finally:
                if tmp_out.exists():
                    tmp_out.unlink()
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
            for new_id, (rec, bind_list) in enumerate(selected):
                rec_copy = {**rec, "id": new_id}
                objects.append(rec_copy)
                for bind in bind_list:
                    bindings.append({**bind, "object_id": new_id})

            model_meta = dict(src_manifest.get("model", {}))
            if source_repo:
                model_meta["source_repo"] = source_repo
            _, schema_id, converter_name = _branding_for_write(
                target_ver, legacy_magic=legacy_magic
            )
            conv_meta = dict(
                src_manifest.get(
                    "converter",
                    {"name": converter_name, "version": target_ver, "endianness": "little"},
                )
            )
            conv_meta["name"] = converter_name
            conv_meta["version"] = target_ver
            conv_meta.setdefault("endianness", "little")
            manifest: dict[str, Any] = {
                "schema": schema_id,
                "container_version": target_ver,
                "model": model_meta,
                "converter": conv_meta,
                "format_registry": _format_registry_for_objects(objects),
                "objects": objects,
                "views": [],
                "bindings": bindings,
                "payload_bytes": payload_bytes,
            }
            if target_ver == VERSION_V2:
                _, features = _compute_v2_flags_and_features(objects, fwht64_o_proj=fwht64_o_proj)
                manifest["format_features"] = features
            art_sha, manifest_crc, manifest_sha = _write_container_from_payload(
                output_path, manifest, payload_name, legacy_magic=legacy_magic
            )
            counts = Counter(str(r["format"]) for r in objects)
            return {
                "schema": "omlx-m5-mtp-extract-report-v1",
                "artifact": str(output_path),
                "artifact_version": target_ver,
                "artifact_bytes": output_path.stat().st_size,
                "artifact_sha256": art_sha,
                "manifest_crc32": f"{manifest_crc:08x}",
                "manifest_sha256": manifest_sha,
                "payload_bytes": payload_bytes,
                "object_count": len(objects),
                "binding_count": len(bindings),
                "fwht64_objects": sum(1 for r in objects if r.get("input_transform") == "fwht64"),
                "formats": dict(sorted(counts.items())),
            }
        finally:
            if os.path.exists(payload_name):
                os.unlink(payload_name)

    target_ver = VERSION_V2 if artifact_version is None else int(artifact_version)
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
        eff_fwht64 = resolve_fwht64_enabled(target_ver, fwht64, detected)

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
        planned_tasks: list[dict[str, Any]] = []
        cursor = 0
        pretiled_count = 0
        for object_id, recipe in enumerate(recipes):
            is_embed = (
                infer_weight_use(recipe.name) == "embedding"
                or (recipe.binding is not None and recipe.binding.get("use") == "embedding")
            )
            target_layout, did_pretile = _target_layout_for_object(
                recipe.name,
                recipe.format_id,
                recipe.layout_id,
                recipe.shape,
                nax_pretile=nax_pretile,
                artifact_version=target_ver,
                is_embedding=is_embed,
            )
            if did_pretile:
                pretiled_count += 1
            do_rot = is_fwht64_eligible(
                recipe.name,
                recipe.shape,
                recipe.format_id,
                profile=detected,
                fwht64=eff_fwht64,
                fwht64_o_proj=fwht64_o_proj,
            )
            direct_nbytes = (
                recipe.planes[0].logical_bytes if recipe.format_id.startswith("direct_") else None
            )
            plane_sizes = _plan_plane_specs(
                recipe.format_id, target_layout, recipe.shape, recipe.padded_k, direct_nbytes
            )
            cursor = align_up(cursor, OBJECT_ALIGNMENT)
            object_begin = cursor
            plane_layouts: list[tuple[str, int, int, int]] = []
            for p_kind, p_bytes in plane_sizes:
                cursor = align_up(cursor, PLANE_ALIGNMENT)
                plane_layouts.append((p_kind, cursor, p_bytes, PLANE_ALIGNMENT))
                cursor += p_bytes
            object_bytes = cursor - object_begin
            planned_tasks.append(
                {
                    "object_id": object_id,
                    "recipe": recipe,
                    "target_layout": target_layout,
                    "do_rot": do_rot,
                    "input_transform": "fwht64" if do_rot else "none",
                    "object_begin": object_begin,
                    "object_bytes": object_bytes,
                    "plane_layouts": plane_layouts,
                }
            )
        payload_bytes = cursor

        try:
            os.ftruncate(payload_fd, payload_bytes)

            def _exec_mtp(task: dict[str, Any]) -> tuple[int, list[dict], str, float, float]:
                recipe: ObjectRecipe = task["recipe"]
                fmt = recipe.format_id
                if fmt.startswith("direct_"):
                    planes, obj_sha, sse, sig_sq = _emit_direct_from_source(
                        source, recipe, payload_fd, task["plane_layouts"]
                    )
                elif fmt == "mxfp4_e2m1_g32_e8m0_v1":
                    planes, obj_sha, sse, sig_sq = _emit_mxfp4_from_source(
                        source,
                        recipe,
                        payload_fd,
                        task["object_begin"],
                        task["object_bytes"],
                        task["plane_layouts"],
                        task["target_layout"],
                        do_fwht64=task["do_rot"],
                    )
                else:
                    planes, obj_sha, sse, sig_sq = _emit_affine_from_source(
                        source,
                        recipe,
                        payload_fd,
                        task["object_begin"],
                        task["object_bytes"],
                        task["plane_layouts"],
                        task["target_layout"],
                        do_fwht64=task["do_rot"],
                    )
                return task["object_id"], planes, obj_sha, sse, sig_sq

            results_by_id: dict[int, tuple[list[dict], str, float, float]] = {}
            workers = min(max(1, int(jobs)), max(1, len(planned_tasks)))
            if workers <= 1:
                for task in planned_tasks:
                    oid, planes, obj_sha, sse, sig_sq = _exec_mtp(task)
                    results_by_id[oid] = (planes, obj_sha, sse, sig_sq)
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    for fut in concurrent.futures.as_completed(
                        [pool.submit(_exec_mtp, t) for t in planned_tasks]
                    ):
                        oid, planes, obj_sha, sse, sig_sq = fut.result()
                        results_by_id[oid] = (planes, obj_sha, sse, sig_sq)

            os.fsync(payload_fd)
            os.close(payload_fd)
            payload_fd = -1

            object_records: list[dict] = []
            bindings = []
            fwht64_count = 0
            total_sse = 0.0
            total_sig_sq = 0.0
            for task in planned_tasks:
                oid = task["object_id"]
                recipe = task["recipe"]
                plane_records, obj_sha, sse, sig_sq = results_by_id[oid]
                total_sse += sse
                total_sig_sq += sig_sq
                rec = {
                    "id": oid,
                    "name": recipe.name,
                    "format": recipe.format_id,
                    "layout": task["target_layout"],
                    "shape": list(recipe.shape),
                    "object_offset": task["object_begin"],
                    "object_bytes": task["object_bytes"],
                    "object_sha256": obj_sha,
                    "planes": plane_records,
                    "source_tensors": recipe.source_tensors,
                }
                if recipe.padded_k is not None:
                    rec["padded_k"] = recipe.padded_k
                if task["input_transform"] == "fwht64":
                    rec["input_transform"] = "fwht64"
                    fwht64_count += 1
                object_records.append(rec)
                if recipe.binding is not None:
                    bindings.append({**recipe.binding, "object_id": oid})

            repo_tag = source_repo or source.model_dir.name
            magic, schema_id, converter_name = _branding_for_write(
                target_ver, legacy_magic=legacy_magic
            )
            manifest = {
                "schema": schema_id,
                "container_version": target_ver,
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
                    "name": converter_name,
                    "version": target_ver,
                    "endianness": "little",
                },
                "format_registry": _format_registry_for_objects(object_records),
                "objects": object_records,
                "views": [],
                "bindings": bindings,
                "payload_bytes": payload_bytes,
            }
            if target_ver == VERSION_V2:
                flags, features = _compute_v2_flags_and_features(
                    object_records, fwht64_o_proj=fwht64_o_proj
                )
                manifest["format_features"] = features
            else:
                flags = 0
            art_sha, manifest_crc, manifest_sha = _write_artifact_container(
                output_path,
                manifest,
                payload_name,
                artifact_version=target_ver,
                flags=flags,
                magic=magic,
            )
            counts = Counter(str(r["format"]) for r in object_records)
            rep = {
                "schema": "omlx-m5-mtp-extract-report-v1",
                "artifact": str(output_path),
                "artifact_version": target_ver,
                "container_flags": flags,
                "artifact_bytes": output_path.stat().st_size,
                "artifact_sha256": art_sha,
                "manifest_crc32": f"{manifest_crc:08x}",
                "manifest_sha256": manifest_sha,
                "payload_bytes": payload_bytes,
                "object_count": len(object_records),
                "binding_count": len(bindings),
                "fwht64_objects": fwht64_count,
                "formats": dict(sorted(counts.items())),
            }
            if nax_pretile:
                rep["pretiled_objects"] = pretiled_count
            if total_sig_sq > 0.0:
                rep["fwht64_sqnr_db"] = round(
                    10.0 * math.log10(total_sig_sq / max(total_sse, 1e-30)), 4
                )
            return rep
        finally:
            if payload_fd >= 0:
                os.close(payload_fd)
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
    legacy_magic: bool = False,
) -> dict:
    """Bundle MTP objects from a checkpoint dir or .miv into an existing base .miv without mutating backbone planes."""
    base_artifact_path = base_artifact_path.resolve()
    mtp_source_path = mtp_source_path.resolve()
    output_path = output_path.resolve()
    if output_path in (base_artifact_path, mtp_source_path):
        raise ValueError("bundle-mtp output path must differ from base and mtp inputs")

    base_summary, base_manifest = read_artifact(base_artifact_path, verify_payload=False)
    base_version = int(base_summary["container_version"])
    tiled_layouts = ("mx_nax_tiled_bk128_v1", *_AFFINE_NAX_ALL_LAYOUTS)
    base_has_pretiled = any(
        str(obj.get("layout", "")) in tiled_layouts
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
                suffix=".miv",
                dir=output_path.parent,
            )
            os.close(fd)
            tmp_mtp_artifact = Path(tmp_name)
            extract_mtp_artifact(
                mtp_source_path,
                tmp_mtp_artifact,
                source_repo=str(base_manifest.get("model", {}).get("source_repo", "")),
                nax_pretile=want_mtp_pretile,
                artifact_version=base_version,
            )
            mtp_art_path = tmp_mtp_artifact
        elif want_mtp_pretile:
            _, raw_mtp_manifest = read_artifact(mtp_source_path, verify_payload=False)
            already_tiled = any(
                str(obj.get("layout", "")) in tiled_layouts
                for obj in raw_mtp_manifest["objects"]
                if _is_mtp_object_name(str(obj["name"]))
            )
            if not already_tiled:
                fd, tmp_name = tempfile.mkstemp(
                    prefix=output_path.name + ".mtp_tiled.",
                    suffix=".miv",
                    dir=output_path.parent,
                )
                os.close(fd)
                tmp_mtp_artifact = Path(tmp_name)
                extract_mtp_artifact(
                    mtp_source_path,
                    tmp_mtp_artifact,
                    nax_pretile=True,
                    artifact_version=base_version,
                )
                mtp_art_path = tmp_mtp_artifact
            else:
                mtp_art_path = mtp_source_path
        else:
            mtp_art_path = mtp_source_path

        mtp_summary, mtp_manifest = read_artifact(mtp_art_path, verify_payload=False)
        target_version = max(base_version, int(mtp_summary["container_version"]))
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

        base_binds: dict[int, list[dict]] = {}
        for b in base_manifest.get("bindings", []):
            base_binds.setdefault(int(b["object_id"]), []).append(dict(b))
        mtp_binds: dict[int, list[dict]] = {}
        for b in mtp_manifest.get("bindings", []):
            mtp_binds.setdefault(int(b["object_id"]), []).append(dict(b))
        plan_entries: list[tuple[str, Path, dict, list[dict]]] = []
        for obj in base_objs:
            plan_entries.append(
                ("base", base_artifact_path, obj, base_binds.get(int(obj["id"]), []))
            )
        for obj in mtp_objs:
            plan_entries.append(
                ("mtp", mtp_art_path, obj, mtp_binds.get(int(obj["id"]), []))
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
                for new_id, (tag, _, obj, bind_list) in enumerate(plan_entries):
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
                    for bind in bind_list:
                        merged_bindings.append({**bind, "object_id": new_id})
                dst_payload.flush()
                os.fsync(dst_payload.fileno())
                payload_bytes = writer.offset

            _, schema_id, converter_name = _branding_for_write(
                target_version, legacy_magic=legacy_magic
            )
            conv_meta = dict(
                base_manifest.get(
                    "converter",
                    {"name": converter_name, "version": target_version, "endianness": "little"},
                )
            )
            conv_meta["name"] = converter_name
            conv_meta["version"] = target_version
            conv_meta.setdefault("endianness", "little")
            manifest = {
                **base_manifest,
                "schema": schema_id,
                "container_version": target_version,
                "converter": conv_meta,
                "format_registry": _format_registry_for_objects(merged_objects),
                "objects": merged_objects,
                "views": [],
                "bindings": merged_bindings,
                "payload_bytes": payload_bytes,
            }
            if target_version == VERSION_V2:
                _, features = _compute_v2_flags_and_features(merged_objects)
                manifest["format_features"] = features
            else:
                manifest.pop("format_features", None)
            art_sha, manifest_crc, manifest_sha = _write_container_from_payload(
                output_path, manifest, payload_name, legacy_magic=legacy_magic
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
        artifact_version=getattr(args, "artifact_version", None),
        fwht64=getattr(args, "fwht64", None),
        fwht64_o_proj=bool(getattr(args, "fwht64_o_proj", True)),
        jobs=int(getattr(args, "jobs", DEFAULT_JOBS)),
        legacy_magic=bool(getattr(args, "legacy_magic", False)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def command_bundle_mtp(args: argparse.Namespace) -> None:
    report = bundle_mtp_artifact(
        args.base,
        args.mtp,
        args.output,
        nax_pretile=args.nax_pretile,
        legacy_magic=bool(getattr(args, "legacy_magic", False)),
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _add_v2_artifact_args(
    subparser: argparse.ArgumentParser,
    *,
    default_version: int | None = DEFAULT_ARTIFACT_VERSION,
) -> None:
    subparser.add_argument(
        "--artifact-version",
        type=int,
        choices=(VERSION_V1, VERSION_V2),
        default=default_version,
        help="target .miv container version (2 = FWHT-64 + pi_64 NAX quartet layout; 1 = legacy v1)",
    )
    subparser.add_argument(
        "--fwht64",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="apply offline FWHT-64 pre-rotation on eligible down_proj/o_proj weights (default: True for v2 unless dflash2_bf16)",
    )
    subparser.add_argument(
        "--fwht64-o-proj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include self_attn.o_proj in offline FWHT-64 pre-rotation when --fwht64 is active (default: True)",
    )
    subparser.add_argument(
        "--jobs",
        type=int,
        default=DEFAULT_JOBS,
        help=f"parallel worker count for quantization/pre-tiling (default: {DEFAULT_JOBS})",
    )
    subparser.add_argument(
        "--legacy-magic",
        action="store_true",
        help="emit legacy OMLXM5A1 magic, omlx-m5-artifact-v{1,2} schema, and omlx-m5-artifact converter name for backward-compat regression tests",
    )


def command_inspect_source(args: argparse.Namespace) -> None:
    report = inspect_source_checkpoint(
        args.model_dir,
        target_model_dir=getattr(args, "target_model", None),
        nax_pretile=bool(getattr(args, "nax_pretile", True)),
        artifact_version=int(getattr(args, "artifact_version", DEFAULT_ARTIFACT_VERSION)),
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
            "direct_bf16",
            "dflash2_bf16",
        ),
        default="auto",
    )
    convert.add_argument(
        "--nax-pretile",
        action="store_true",
        help="offline pre-tile rank-2 MXFP4 and Q3/Q4/Q5/Q6/Q8 G64/G128 weights for zero-repack M5 NAX",
    )
    _add_v2_artifact_args(convert, default_version=DEFAULT_ARTIFACT_VERSION)
    convert.set_defaults(func=command_convert)
    pretile = subparsers.add_parser(
        "pretile",
        help="offline pre-tile an existing .miv artifact for zero-repack M5 NAX",
    )
    pretile.add_argument("--input", type=Path, required=True)
    pretile.add_argument("--output", type=Path, required=True)
    _add_v2_artifact_args(pretile, default_version=DEFAULT_ARTIFACT_VERSION)
    pretile.set_defaults(func=command_pretile)
    extract_mtp = subparsers.add_parser(
        "extract-mtp",
        help="extract ModelComponent::mtp weights from a checkpoint directory or .miv into a standalone .miv",
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
    _add_v2_artifact_args(extract_mtp, default_version=None)
    extract_mtp.set_defaults(func=command_extract_mtp)
    bundle_mtp = subparsers.add_parser(
        "bundle-mtp",
        help="bundle ModelComponent::mtp weights from a checkpoint directory or .miv into an existing base .miv",
    )
    bundle_mtp.add_argument("--base", type=Path, required=True)
    bundle_mtp.add_argument("--mtp", type=Path, required=True)
    bundle_mtp.add_argument("--output", type=Path, required=True)
    bundle_mtp.add_argument("--nax-pretile", action="store_true")
    bundle_mtp.add_argument(
        "--legacy-magic",
        action="store_true",
        help="emit legacy OMLXM5A1 magic, omlx-m5-artifact-v{1,2} schema, and omlx-m5-artifact converter name for backward-compat regression tests",
    )
    bundle_mtp.set_defaults(func=command_bundle_mtp)
    qda = subparsers.add_parser(
        "qda-search",
        help="#4776 + #5292 oQe + QDA top-k KL knapsack bit-allocation search & optional .miv export",
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
    _add_v2_artifact_args(qda, default_version=DEFAULT_ARTIFACT_VERSION)
    qda.set_defaults(func=command_qda_search)
    inspect = subparsers.add_parser("inspect", help="inspect an artifact")
    inspect.add_argument("artifact", type=Path)
    inspect.add_argument("--verify-payload", action="store_true")
    inspect.add_argument("--list-objects", action="store_true")
    inspect.add_argument("--sample-object")
    inspect.add_argument("--sample-row", type=int, default=0)
    inspect.add_argument("--sample-start", type=int, default=0)
    inspect.add_argument("--sample-count", type=int, default=16)
    inspect.add_argument(
        "--raw-rotated",
        action="store_true",
        help="return stored rotated weights instead of unrotating fwht64 objects",
    )
    inspect.set_defaults(func=command_inspect)
    inspect_source = subparsers.add_parser(
        "inspect-source",
        help="O(1) < 5 ms metadata-only capability gate & per-tensor repack planner for a source checkpoint",
    )
    inspect_source.add_argument("model_dir", type=Path)
    inspect_source.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help="emit JSON inspection report",
    )
    inspect_source.add_argument(
        "--target-model",
        type=Path,
        default=None,
        help="optional target model directory for DFlash2 draft compatibility validation",
    )
    inspect_source.add_argument(
        "--nax-pretile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="plan NAX pre-tile layouts where eligible (default: True)",
    )
    inspect_source.add_argument(
        "--artifact-version",
        type=int,
        choices=(VERSION_V1, VERSION_V2),
        default=DEFAULT_ARTIFACT_VERSION,
        help="target .miv container version (default: 2)",
    )
    inspect_source.set_defaults(func=command_inspect_source)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
