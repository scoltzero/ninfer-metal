#!/usr/bin/env python3
"""Download and verify the pinned B0 Dense/MoE checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelPin:
    key: str
    repo: str
    revision: str
    files: tuple[str, ...]
    lfs_sha256: dict[str, str]

    def __post_init__(self) -> None:
        for filename, digest in self.lfs_sha256.items():
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(
                    f"{self.key}: invalid SHA-256 for {filename}: "
                    f"expected 64 lowercase hex chars, got {len(digest)}"
                )


PINS = {
    "q36_27_oq": ModelPin(
        key="q36_27_oq",
        repo="Jundot/Qwen3.6-27B-oQ4e-mtp",
        revision="2523e7a5702d38a1a319c50d193cbac20f9ecb78",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ),
        lfs_sha256={
            "model-00001-of-00004.safetensors": "41405c90147c9e1d769533c98f3a6176c676d27a6eba13986de5f506feda700b",
            "model-00002-of-00004.safetensors": "c1f61a82c77b9af4bd19b892e0d35dc4ccd6069e70e6793d7b9c5b52babcffa2",
            "model-00003-of-00004.safetensors": "4304e8992a2d51a3382a2e651995b80fa7314675f7db591e009f6f0b43b1b876",
            "model-00004-of-00004.safetensors": "f143f9a1520907fc68faddbd5058584d54e09ed3c80f7b0e4cad1a232142ae44",
        },
    ),
    "q36_27_mx": ModelPin(
        key="q36_27_mx",
        repo="OsaurusAI/Qwen3.6-27B-MXFP4",
        revision="3258bea991decf16bf84fcd21cd73ae089bf3bbc",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ),
        lfs_sha256={
            "model-00001-of-00003.safetensors": "2033729d117e57b4393bdb6545aed2ea3d4e38a047f0dd22a1516828c7063155",
            "model-00002-of-00003.safetensors": "a4876ab5a89b9ca40e47509285f4b4f15087da7d18557d48681ca2d9a19ceb54",
            "model-00003-of-00003.safetensors": "68ea75a3fa4127761624db74eaa40441ed9b4852712902e11dca6985ad183ff1",
        },
    ),
    "q36_35_oq": ModelPin(
        key="q36_35_oq",
        repo="Jundot/Qwen3.6-35B-A3B-oQ4e-mtp",
        revision="14c285372cbdb1777adea5bb49087ced0bffc0b5",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ),
        lfs_sha256={
            "model-00001-of-00005.safetensors": "4c545f85c5308b74e02bdf20e5f648b6cd2f7e4383c826f1688def66f5d63fbd",
            "model-00002-of-00005.safetensors": "599c2d149705f4837b4280965301cba8610898adf37755ae575cef8a79c08739",
            "model-00003-of-00005.safetensors": "c16d7f4b2afe97d6c01e0d0ef68f3c460c13d9ccb0664364881d701d4b74dbcc",
            "model-00004-of-00005.safetensors": "e80f1425e70683ec0da8850c5908d6fc8bb65e3c15b7f802b53892fcf21e786d",
            "model-00005-of-00005.safetensors": "bbc72190da61a293643e4d1ee8914f0b115a3725f06cb29f92469722851b5014",
        },
    ),
    "q36_35_mx": ModelPin(
        key="q36_35_mx",
        repo="OsaurusAI/Qwen3.6-35B-A3B-mxfp4",
        revision="5a6aaf3d0db3590f186d743b99c9b9d270ec7857",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00005.safetensors",
            "model-00002-of-00005.safetensors",
            "model-00003-of-00005.safetensors",
            "model-00004-of-00005.safetensors",
            "model-00005-of-00005.safetensors",
        ),
        lfs_sha256={
            "model-00001-of-00005.safetensors": "1a90eaeb5128b4d2ffbfc6438971637450ae9ed9db87df4fe32aeef65e82ae7a",
            "model-00002-of-00005.safetensors": "570ca827e6435b07fe42a3d2606e2e7dbd7d4c8c4e0301f0d98901b29e982a20",
            "model-00003-of-00005.safetensors": "90b82a0f591c47dc04a938a0bdcba8b5d33558343bc9afced47a81f57393c3f0",
            "model-00004-of-00005.safetensors": "2c18a515919fbb45a9d6dd0d0f838a3071ca212c757aa0a5a8232c69ae19f62b",
            "model-00005-of-00005.safetensors": "3345555e47465a225bb780bbd091d481e9ae13c96f4863b3571eb09cd6a7bd31",
        },
    ),
    "q38_27_oq": ModelPin(
        key="q38_27_oq",
        repo="Jundot/Qwen3.8-27B-oQ4e-mtp",
        revision="04dc5509edd8670fc78cc8c2f74bf9b77b1f2acc",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ),
        lfs_sha256={
            "model-00001-of-00004.safetensors": "12d98dd2b44baf3745134095e3f7c103e136c36ab34ab4a944a2e6edfa0d1a45",
            "model-00002-of-00004.safetensors": "51ff1e62717b7a731cc584b149fe6e1fe645921dd6e4a3702dade338eccc0cbd",
            "model-00003-of-00004.safetensors": "d3555aff4d37c9b4d76ad42a654a62ef3054cded714922d011968b410fa9359f",
            "model-00004-of-00004.safetensors": "a253a34b222e911ff08ed7defa74349fcb3b5614373ae359ce588703356895cb",
        },
    ),
    "q38_27_oq_latest": ModelPin(
        key="q38_27_oq_latest",
        repo="Jundot/Qwen3.8-27B-oQ4e-mtp",
        revision="836759eefbef7ef588a5ffd44cc59a640808f944",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00004.safetensors",
            "model-00002-of-00004.safetensors",
            "model-00003-of-00004.safetensors",
            "model-00004-of-00004.safetensors",
        ),
        lfs_sha256={
            "model-00001-of-00004.safetensors": "bb1333e2f9e10e2970f661a4b70ebf019a019b01d3aefba2d106d407d011efe8",
            "model-00002-of-00004.safetensors": "f6ef37318fc2bab532e1fec6ce228bf7711e2285e9ff37ca6b43838d973adbbd",
            "model-00003-of-00004.safetensors": "184672ac40a38046220540316c14a7ce154b710f9f3eb4044d1baa013d9a8134",
            "model-00004-of-00004.safetensors": "b464d60538c4472f0aacdeda28d79fba5a0170649e82d7cf08f70902807bff68",
        },
    ),
    "dflash2": ModelPin(
        key="dflash2",
        repo="z-lab/Qwen3.8-27B-DFlash2",
        revision="50307d4c4cde6860d4eee73e2547cd786fe8e8a4",
        files=(
            "config.json",
            "model.safetensors",
        ),
        lfs_sha256={
            "model.safetensors": "67fc76d68dc5a9415511a4f394ef744d67510cd20e93b37cc2cc7d28e4bab65c",
        },
    ),
    "mlx": ModelPin(
        key="mlx",
        repo="mlx-community/Qwen3.8-27B-mxfp4",
        revision="97ab0819817ab1c61d7d39f9169fc71999915641",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "model-00001-of-00003.safetensors",
            "model-00002-of-00003.safetensors",
            "model-00003-of-00003.safetensors",
        ),
        lfs_sha256={
            "model-00001-of-00003.safetensors": "da6d7815bc72b8d19aa348c33b64ea55a711845cc66fc4b3ad2512fcd45c0d33",
            "model-00002-of-00003.safetensors": "79691fbaf6f93ac18214e0906900b9e1ee11b9122798ee9dcd22b35cca8805cc",
            "model-00003-of-00003.safetensors": "d0135d71838ec5979749f45417a544618f77321b527ec0df069fd5ba3cdb7478",
        },
    ),
    "redhat": ModelPin(
        key="redhat",
        repo="RedHatAI/Qwen3.8-27B-MXFP4",
        revision="651eed99386dae5645cd4949024e8ebd163a533b",
        files=(
            "config.json",
            "model.safetensors.index.json",
            "recipe.yaml",
            "model.safetensors",
            "model_mtp.safetensors",
        ),
        lfs_sha256={
            "model.safetensors": "1ccad7f8f264fb0aeaf1466ee73860b3aa752976a1e838c0dbff8c622983f4c5",
            "model_mtp.safetensors": "1d8268aa85ace093a561e3e7b63b9d390dac1cd55a90cd55b5ec509c3c9da9fe",
        },
    ),
}


BUNDLES: dict[str, tuple[str, ...]] = {
    "early_bringup": ("q38_27_oq", "mlx", "dflash2"),
    "early_bringup_moe": ("q38_27_oq", "mlx", "dflash2", "q36_35_oq", "q36_35_mx"),
}


def hf_endpoint() -> str:
    return os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")


def _curl_ip_flags() -> list[str]:
    return ["-4"] if os.environ.get("HF_IPV4", "0") in ("1", "true", "yes") else []


def api_metadata(pin: ModelPin) -> dict[str, dict]:
    url = f"{hf_endpoint()}/api/models/{pin.repo}/revision/{pin.revision}?blobs=true"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    command = [
        "curl",
        *_curl_ip_flags(),
        "--fail",
        "--location",
        "--retry",
        "5",
        "--silent",
        "--show-error",
        *(["-H", f"Authorization: Bearer {token}"] if token else []),
        url,
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    payload = json.loads(result.stdout)
    if payload.get("sha") != pin.revision:
        raise RuntimeError(f"{pin.repo}: revision response does not match pin")
    return {item["rfilename"]: item for item in payload["siblings"]}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(pin: ModelPin, name: str, size: int, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    if destination.is_file() and destination.stat().st_size == size:
        return
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite wrong-sized file: {destination}")
    url = f"{hf_endpoint()}/{pin.repo}/resolve/{pin.revision}/{name}"
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    for attempt in range(1, 31):
        command = [
            "curl",
            *_curl_ip_flags(),
            "--fail",
            "--location",
            "--retry",
            "5",
            "--continue-at",
            "-",
            "--silent",
            "--show-error",
            *(["-H", f"Authorization: Bearer {token}"] if token else []),
            "--output",
            str(partial),
            url,
        ]
        result = subprocess.run(command, check=False)
        if (
            result.returncode == 0
            and partial.is_file()
            and partial.stat().st_size == size
        ):
            os.replace(partial, destination)
            return
        if partial.is_file() and partial.stat().st_size == size:
            os.replace(partial, destination)
            return
        time.sleep(min(60, attempt * 3))
    raise RuntimeError(
        f"download did not complete after 30 attempts: {pin.repo}/{name}"
    )


def atomic_json(path: Path, value: dict) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".tmp.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def download_model(pin: ModelPin, root: Path) -> dict:
    metadata = api_metadata(pin)
    destination = root / pin.repo
    destination.mkdir(parents=True, exist_ok=True)
    report_files = []
    for name in pin.files:
        if name not in metadata or not isinstance(metadata[name].get("size"), int):
            raise RuntimeError(f"{pin.repo}: missing size metadata for {name}")
        size = int(metadata[name]["size"])
        path = destination / name
        print(f"[{pin.key}] {name}: {size / 1_000_000_000:.3f} GB", flush=True)
        download_file(pin, name, size, path)
        expected_hash = pin.lfs_sha256.get(name)
        actual_hash = sha256_file(path) if expected_hash else ""
        if expected_hash and actual_hash != expected_hash:
            raise RuntimeError(f"{pin.repo}: SHA-256 mismatch for {name}")
        report_files.append(
            {
                "name": name,
                "bytes": size,
                "sha256": actual_hash,
            }
        )
    (destination / "REVISION").write_text(pin.revision + "\n")
    report = {
        "schema": "omlx-m5-b0-download-v1",
        "repo": pin.repo,
        "revision": pin.revision,
        "files": report_files,
    }
    atomic_json(destination / "DOWNLOAD.json", report)
    print(f"[{pin.key}] verified at {destination}", flush=True)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=("all", *BUNDLES, *PINS),
        default="all",
        help="checkpoint or bundle to fetch (e.g. early_bringup, early_bringup_moe, all)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=(
            Path(__file__).resolve().parents[2]
            if Path(__file__).resolve().parents[1].name == "miv"
            else Path(__file__).resolve().parents[1]
        )
        / "models",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model == "all":
        keys = tuple(PINS)
    elif args.model in BUNDLES:
        keys = BUNDLES[args.model]
    else:
        keys = (args.model,)
    reports = [download_model(PINS[key], args.root.resolve()) for key in keys]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
