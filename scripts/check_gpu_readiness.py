#!/usr/bin/env python3
"""Read-only readiness gate for the one-patient GPU extraction pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


EXPECTED_OFFICIAL_COMMIT = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
EXPECTED_WSI_MD5 = "824785fee9387dcf46a7058a0722739b"
EXPECTED_COORDINATES_SHA256 = (
    "e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51"
)
EXPECTED_OMIC_SHA256 = "9fa2cd83906c00e1f50113ba8e806ea4537806aeb2aacbee60e86c80b53f6929"
EXPECTED_CHECKPOINT_SHA256_PREFIX = "11ad3fa6"


def _digest(path: Path, algorithm: str) -> str:
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _file_check(
    path: Path | None,
    *,
    algorithm: str,
    expected: str,
    prefix: bool = False,
) -> dict[str, Any]:
    if path is None:
        return {"path": None, "present": False, "digest": None, "matches": False}
    resolved = path.expanduser().resolve()
    present = resolved.is_file()
    digest = _digest(resolved, algorithm) if present else None
    matches = bool(digest and (digest.startswith(expected) if prefix else digest == expected))
    return {
        "path": str(resolved),
        "present": present,
        "size_bytes": resolved.stat().st_size if present else None,
        f"{algorithm}": digest,
        "expected": expected,
        "matches": matches,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check GPU, checkpoint, source, and optional data readiness without writing files."
    )
    parser.add_argument(
        "--official-repo",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "healnet",
        help="Sibling checkout of the official HEALNet repository.",
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--wsi", type=Path)
    parser.add_argument("--coordinates", type=Path)
    parser.add_argument("--omic", type=Path)
    parser.add_argument(
        "--hardware-only",
        action="store_true",
        help="Check compute/source prerequisites without requiring checkpoint or data files.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    try:
        import torch
        import torchvision
    except Exception as exc:  # pragma: no cover - depends on target environment
        print(json.dumps({"ready": False, "import_error": repr(exc)}, indent=2))
        return 1

    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "resnet50-11ad3fa6.pth"

    nvidia_smi_path = shutil.which("nvidia-smi")
    nvidia_smi_ok = False
    nvidia_smi_summary: str | None = None
    if nvidia_smi_path is not None:
        completed = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        nvidia_smi_ok = completed.returncode == 0
        nvidia_smi_summary = (completed.stdout or completed.stderr).strip() or None

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count()
    device_name = torch.cuda.get_device_name(0) if cuda_available else None
    official_model = args.official_repo / "healnet" / "models" / "healnet.py"

    data_checks: dict[str, dict[str, Any]] = {
        "wsi": _file_check(
            args.wsi,
            algorithm="md5",
            expected=EXPECTED_WSI_MD5,
        ),
        "coordinates": _file_check(
            args.coordinates,
            algorithm="sha256",
            expected=EXPECTED_COORDINATES_SHA256,
        ),
        "omic": _file_check(
            args.omic,
            algorithm="sha256",
            expected=EXPECTED_OMIC_SHA256,
        ),
    }
    data_ok = all(item["matches"] for item in data_checks.values())

    git_result = subprocess.run(
        ["git", "-C", str(args.official_repo), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    official_commit = git_result.stdout.strip() if git_result.returncode == 0 else None

    try:
        import openslide

        openslide_import_ok = True
        openslide_python_version = getattr(openslide, "__version__", None)
        openslide_library_version = getattr(openslide, "__library_version__", None)
    except Exception as exc:  # pragma: no cover - depends on target environment
        openslide_import_ok = False
        openslide_python_version = None
        openslide_library_version = None
        openslide_error = repr(exc)
    else:
        openslide_error = None

    checkpoint_check = _file_check(
        checkpoint,
        algorithm="sha256",
        expected=EXPECTED_CHECKPOINT_SHA256_PREFIX,
        prefix=True,
    )

    checks: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "device_name": device_name,
        "nvidia_smi_available": nvidia_smi_path is not None,
        "nvidia_smi_ok": nvidia_smi_ok,
        "nvidia_smi_summary": nvidia_smi_summary,
        "official_repo": str(args.official_repo.resolve()),
        "official_model_present": official_model.is_file(),
        "official_commit": official_commit,
        "official_commit_expected": EXPECTED_OFFICIAL_COMMIT,
        "official_commit_matches": official_commit == EXPECTED_OFFICIAL_COMMIT,
        "openslide_import_ok": openslide_import_ok,
        "openslide_python_version": openslide_python_version,
        "openslide_library_version": openslide_library_version,
        "openslide_error": openslide_error,
        "checkpoint": checkpoint_check,
        "weight_enum": "ResNet50_Weights.IMAGENET1K_V2",
        "data": data_checks,
        "mode": "hardware_only" if args.hardware_only else "full",
    }

    hardware_ready = bool(
        cuda_available
        and cuda_device_count > 0
        and nvidia_smi_ok
        and official_model.is_file()
        and official_commit == EXPECTED_OFFICIAL_COMMIT
        and openslide_import_ok
    )
    checks["hardware_ready"] = hardware_ready
    checks["ready"] = bool(
        hardware_ready
        if args.hardware_only
        else hardware_ready and checkpoint_check["matches"] and data_ok
    )
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 0 if checks["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
