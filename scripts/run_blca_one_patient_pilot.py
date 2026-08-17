#!/usr/bin/env python3
"""Run the locked real BLCA one-patient multiscale interface pilot once."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import h5py
import numpy as np
import openslide
import cv2
from PIL import __version__ as pillow_version
import torch
import torchvision


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiscale_feature_pilot.src.artifacts import (  # noqa: E402
    ArtifactRecord,
    atomic_save_tensor,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from multiscale_feature_pilot.src.feature_extraction import (  # noqa: E402
    PatchBranchSpec,
    StreamingOpenSlideDataset,
    build_resnet50_imagenet1k_v2,
    extract_feature_matrix,
)
from multiscale_feature_pilot.src.healnet_smoke import (  # noqa: E402
    OFFICIAL_HEALNET_COMMIT,
    run_one_patient_healnet_smoke,
)
from multiscale_feature_pilot.src.multiscale_bag import (  # noqa: E402
    build_multiscale_bag,
    validate_feature_matrix,
)
from multiscale_feature_pilot.src.omic import (  # noqa: E402
    BLCA_PILOT_CASE_ID,
    BLCA_PILOT_SLIDE_ID,
    load_blca_pilot_omics,
)
from multiscale_feature_pilot.src.provenance import (  # noqa: E402
    PROVENANCE_FIELDS,
    BranchProvenanceSpec,
    provenance_as_dicts,
    validate_provenance_alignment,
)
from multiscale_feature_pilot.src.scale_2x_policy import (  # noqa: E402
    CLAM_COMMIT,
    EFFECTIVE_MPP,
    GRID_LABEL,
    OUTPUT_PATCH_SIZE,
    POLICY_STATUS,
    SOURCE_FOOTPRINT,
    SOURCE_LEVEL,
    SOURCE_MPP,
)
from multiscale_feature_pilot.src.tissue_coordinates import (  # noqa: E402
    generate_locked_scale_2x_coordinates,
)


EXPECTED_PYTHON = Path("/home/zeus/miniconda3/envs/cloudspace/bin/python")
EXPECTED_WSI_SIZE = 2_658_499_382
EXPECTED_WSI_MD5 = "824785fee9387dcf46a7058a0722739b"
EXPECTED_HDF5_SIZE = 572_080
EXPECTED_HDF5_SHA256 = (
    "e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51"
)
EXPECTED_OMIC_SHA256 = "9fa2cd83906c00e1f50113ba8e806ea4537806aeb2aacbee60e86c80b53f6929"
EXPECTED_CHECKPOINT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
EXPECTED_4X_COUNT = 8911
EXPECTED_4X_LEVEL = 1
EXPECTED_PATCH_SIZE = 256
MANIFEST_SCHEMA_VERSION = 1
IMPLEMENTATION_FILES = (
    "scripts/run_blca_one_patient_pilot.py",
    "requirements-handoff.txt",
    "multiscale_feature_pilot/config/pilot_config.yaml",
    "multiscale_feature_pilot/provenance/scale_selection.yaml",
    "multiscale_feature_pilot/src/artifacts.py",
    "multiscale_feature_pilot/src/feature_extraction.py",
    "multiscale_feature_pilot/src/healnet_smoke.py",
    "multiscale_feature_pilot/src/multiscale_bag.py",
    "multiscale_feature_pilot/src/omic.py",
    "multiscale_feature_pilot/src/provenance.py",
    "multiscale_feature_pilot/src/scale_2x_policy.py",
    "multiscale_feature_pilot/src/tissue_coordinates.py",
)


class PilotContractError(RuntimeError):
    """Raised when a real-pilot prerequisite or output violates its contract."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--wsi", type=Path, required=True)
    parser.add_argument("--coordinates", type=Path, required=True)
    parser.add_argument("--omic", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def _md5_file(path: Path) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotContractError(message)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise PilotContractError(
            f"git {' '.join(arguments)} failed for {repo}: "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return result.stdout.strip()


def _pilot_implementation_identity() -> dict[str, Any]:
    file_hashes: dict[str, str] = {}
    for relative_name in IMPLEMENTATION_FILES:
        source = REPO_ROOT / relative_name
        _require(source.is_file(), f"missing execution-critical source: {source}")
        file_hashes[relative_name] = sha256_file(source)
    status_lines = _git(REPO_ROOT, "status", "--porcelain=v1").splitlines()
    return {
        "pilot_repo": str(REPO_ROOT.resolve()),
        "git_head": _git(REPO_ROOT, "rev-parse", "HEAD"),
        "git_branch": _git(REPO_ROOT, "branch", "--show-current"),
        "working_tree_dirty": bool(status_lines),
        "working_tree_status": status_lines,
        "execution_file_sha256": file_hashes,
    }


def _record(
    record: ArtifactRecord,
    *,
    published_path: Path | None = None,
) -> dict[str, object]:
    return {
        "path": str((published_path or record.path).resolve()),
        "size_bytes": record.size_bytes,
        "sha256": record.sha256,
        "reused": record.reused,
    }


def _emit(event: str, **values: object) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


def _normalise_json_value(value: Any) -> Any:
    """Round-trip values through strict JSON for persisted-value comparison."""

    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


def _transaction_paths(output_dir: Path) -> tuple[Path, Path]:
    staging = output_dir.parent / f".{output_dir.name}.staging"
    lock = output_dir.parent / f".{output_dir.name}.lock"
    return staging, lock


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise PilotContractError(f"could not resolve an existing parent for {path}")
        candidate = candidate.parent
    return candidate if candidate.is_dir() else candidate.parent


def _validate_output_destination(output_dir: Path, official_repo: Path) -> None:
    output_dir = output_dir.resolve()
    official_repo = official_repo.resolve()
    for label, repository in (("pilot", REPO_ROOT.resolve()), ("official", official_repo)):
        if output_dir == repository or output_dir.is_relative_to(repository):
            raise PilotContractError(
                f"artifact output directory must not be inside the {label} Git repository"
            )

    existing_parent = _existing_parent(output_dir.parent)
    result = subprocess.run(
        ["git", "-C", str(existing_parent), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        worktree = Path(result.stdout.strip()).resolve()
        if output_dir == worktree or output_dir.is_relative_to(worktree):
            raise PilotContractError(
                f"artifact output directory must be outside Git: {worktree}"
            )

    staging, lock = _transaction_paths(output_dir)
    _require(not os.path.lexists(output_dir), f"final output directory already exists: {output_dir}")
    _require(not os.path.lexists(staging), f"stale pilot staging directory exists: {staging}")
    _require(not os.path.lexists(lock), f"stale pilot transaction lock exists: {lock}")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _begin_output_transaction(output_dir: Path) -> tuple[Path, Path]:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging, lock = _transaction_paths(output_dir)
    try:
        descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise PilotContractError(f"pilot transaction is already locked: {lock}") from exc
    with os.fdopen(descriptor, "w", encoding="ascii") as stream:
        stream.write(f"pid={os.getpid()}\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        staging.mkdir()
    except Exception:
        lock.unlink(missing_ok=True)
        raise
    _fsync_directory(output_dir.parent)
    return staging, lock


def _publish_output_transaction(
    staging: Path,
    output_dir: Path,
    lock: Path,
) -> None:
    _require(staging.is_dir(), f"missing pilot staging directory: {staging}")
    _require(not os.path.lexists(output_dir), f"refusing to replace output directory: {output_dir}")
    _require(lock.is_file(), f"missing pilot transaction lock: {lock}")
    _fsync_directory(staging)
    lock.unlink()
    _fsync_directory(output_dir.parent)
    # Every fallible validation and cleanup operation occurs before this call.
    # A successful rename is the single run-level publication commit point.
    os.rename(staging, output_dir)


def _load_verified_4x_coordinates(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    with h5py.File(path, "r") as handle:
        _require(list(handle.keys()) == ["coords"], "HDF5 must contain only the coords dataset")
        dataset = handle["coords"]
        coordinates = np.asarray(dataset[...])
        attributes = dict(dataset.attrs)

    _require(coordinates.shape == (EXPECTED_4X_COUNT, 2), "HDF5 coords shape mismatch")
    _require(coordinates.dtype == np.int64, "HDF5 coords dtype must be int64")
    _require(int(attributes.get("patch_level", -1)) == EXPECTED_4X_LEVEL, "patch_level mismatch")
    _require(int(attributes.get("patch_size", -1)) == EXPECTED_PATCH_SIZE, "patch_size mismatch")
    unique_count = int(np.unique(coordinates, axis=0).shape[0])
    _require(unique_count == EXPECTED_4X_COUNT, "HDF5 coordinates contain duplicates")
    _require(Path(path).stem == Path(BLCA_PILOT_SLIDE_ID).stem, "HDF5 identity mismatch")
    dataset_name = str(attributes.get("name", ""))
    _require(dataset_name == Path(BLCA_PILOT_SLIDE_ID).stem, "HDF5 dataset name mismatch")
    tensor = torch.from_numpy(np.ascontiguousarray(coordinates)).to(dtype=torch.int64)
    return tensor, {
        "shape": list(coordinates.shape),
        "dtype": str(coordinates.dtype),
        "patch_level": EXPECTED_4X_LEVEL,
        "patch_size": EXPECTED_PATCH_SIZE,
        "unique_coordinates": unique_count,
        "duplicate_coordinates": EXPECTED_4X_COUNT - unique_count,
        "dataset_name": dataset_name,
    }


def _preflight(args: argparse.Namespace) -> tuple[dict[str, Any], torch.Tensor, object]:
    official_repo = args.official_repo.resolve()
    wsi = args.wsi.resolve()
    coordinates_path = args.coordinates.resolve()
    omic_path = args.omic.resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.resolve()

    _require(Path(sys.executable).resolve() == EXPECTED_PYTHON.resolve(), "wrong Python interpreter")
    for label, path in (
        ("official repository", official_repo),
        ("WSI", wsi),
        ("HDF5", coordinates_path),
        ("Omic CSV", omic_path),
        ("checkpoint", checkpoint),
    ):
        _require(path.is_dir() if label == "official repository" else path.is_file(), f"missing {label}: {path}")

    _validate_output_destination(output_dir, official_repo)

    _require(wsi.name == BLCA_PILOT_SLIDE_ID, "WSI identity mismatch")
    _require(wsi.stat().st_size == EXPECTED_WSI_SIZE, "WSI size mismatch")
    wsi_md5 = _md5_file(wsi)
    _require(wsi_md5 == EXPECTED_WSI_MD5, "WSI MD5 mismatch")

    _require(coordinates_path.stat().st_size == EXPECTED_HDF5_SIZE, "HDF5 size mismatch")
    hdf5_sha256 = sha256_file(coordinates_path)
    _require(hdf5_sha256 == EXPECTED_HDF5_SHA256, "HDF5 SHA256 mismatch")
    scale_4x_coordinates, hdf5_structure = _load_verified_4x_coordinates(coordinates_path)

    omic_sha256 = sha256_file(omic_path)
    _require(omic_sha256 == EXPECTED_OMIC_SHA256, "Omic SHA256 mismatch")
    omic = load_blca_pilot_omics(omic_path)
    _require(omic.case_id == BLCA_PILOT_CASE_ID, "Omic case identity mismatch")
    _require(omic.slide_id == BLCA_PILOT_SLIDE_ID, "Omic slide identity mismatch")

    checkpoint_sha256 = sha256_file(checkpoint)
    _require(checkpoint_sha256 == EXPECTED_CHECKPOINT_SHA256, "checkpoint SHA256 mismatch")

    head = _git(official_repo, "rev-parse", "HEAD")
    _require(head == OFFICIAL_HEALNET_COMMIT, "official HEALNet HEAD mismatch")
    _require(_git(official_repo, "status", "--porcelain") == "", "official HEALNet is modified")
    pilot_implementation = _pilot_implementation_identity()

    _require(torch.cuda.is_available(), "torch CUDA is unavailable")
    _require(torch.cuda.device_count() >= 1, "no CUDA device is exposed")
    device_name = torch.cuda.get_device_name(0)
    _require("Tesla T4" in device_name, f"expected Tesla T4, got {device_name}")
    nvidia_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(nvidia_result.returncode == 0, "nvidia-smi query failed")
    driver_version, memory_total_mib = (
        item.strip() for item in nvidia_result.stdout.strip().split(",", maxsplit=1)
    )

    slide = openslide.OpenSlide(str(wsi))
    try:
        mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
        level_dimensions = [list(map(int, dimensions)) for dimensions in slide.level_dimensions]
        level_downsamples = [float(value) for value in slide.level_downsamples]
    finally:
        slide.close()
    _require(level_downsamples == [1.0, 4.0, 16.0, 64.0], "WSI pyramid mismatch")
    _require(mpp_x * level_downsamples[2] == 3.6432, "level-2 MPP mismatch")
    _require(mpp_y * level_downsamples[2] == 3.6432, "level-2 MPP mismatch")

    preflight = {
        "python": sys.executable,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "opencv_version": cv2.__version__,
        "pillow_version": pillow_version,
        "openslide_python_version": openslide.__version__,
        "openslide_library_version": openslide.__library_version__,
        "cuda_device_count": torch.cuda.device_count(),
        "device_name": device_name,
        "nvidia_driver_version": driver_version,
        "gpu_memory_total_mib": int(memory_total_mib),
        "wsi": {
            "path": str(wsi),
            "size_bytes": wsi.stat().st_size,
            "md5": wsi_md5,
            "level_dimensions": level_dimensions,
            "level_downsamples": level_downsamples,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
        },
        "hdf5": {
            "path": str(coordinates_path),
            "size_bytes": coordinates_path.stat().st_size,
            "sha256": hdf5_sha256,
            **hdf5_structure,
        },
        "omic": {
            "path": str(omic_path),
            "size_bytes": omic_path.stat().st_size,
            "sha256": omic_sha256,
            "case_id": omic.case_id,
            "sample_id": omic.sample_id,
            "slide_id": omic.slide_id,
            "shapes": {
                "rna": list(omic.rna.shape),
                "mutation": list(omic.mutation.shape),
                "cnv": list(omic.cnv.shape),
            },
        },
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
            "weights": "ResNet50_Weights.IMAGENET1K_V2",
        },
        "official_healnet": {
            "path": str(official_repo),
            "head": head,
            "modified": False,
        },
        "pilot_implementation": pilot_implementation,
    }
    return preflight, scale_4x_coordinates, omic


def _validate_saved_tensor(path: Path, expected: torch.Tensor, name: str) -> None:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    _require(isinstance(loaded, torch.Tensor), f"{name} artifact is not tensor-only")
    _require(loaded.shape == expected.shape, f"{name} artifact shape mismatch")
    _require(loaded.dtype == expected.dtype, f"{name} artifact dtype mismatch")
    _require(torch.equal(loaded, expected), f"{name} artifact content mismatch")


def _run(args: argparse.Namespace, stage: dict[str, str]) -> dict[str, Any]:
    overall_start = time.perf_counter()
    stage["name"] = "preflight"
    _emit("stage", name=stage["name"])
    preflight, scale_4x_coordinates, omic = _preflight(args)
    _emit("preflight_passed", device=preflight["device_name"])
    output_dir = args.output_dir.resolve()
    staging_dir, transaction_lock = _begin_output_transaction(output_dir)
    _emit("output_transaction_started", staging=str(staging_dir))

    stage["name"] = "coordinates_2x"
    _emit("stage", name=stage["name"])
    coordinate_start = time.perf_counter()
    scale_2x_result = generate_locked_scale_2x_coordinates(args.wsi)
    coordinate_seconds = time.perf_counter() - coordinate_start
    scale_2x_coordinates = torch.from_numpy(
        np.ascontiguousarray(scale_2x_result.coordinates)
    ).to(dtype=torch.int64)
    _require(scale_2x_coordinates.ndim == 2 and scale_2x_coordinates.shape[1] == 2, "2x coordinate shape mismatch")
    _require(scale_2x_coordinates.shape[0] > 0, "2x coordinate generation returned no tissue")
    _require(
        torch.unique(scale_2x_coordinates, dim=0).shape[0] == scale_2x_coordinates.shape[0],
        "2x coordinates contain duplicates",
    )
    n1 = int(scale_2x_coordinates.shape[0])
    _emit(
        "coordinates_2x_ready",
        count=n1,
        seconds=coordinate_seconds,
        contours=scale_2x_result.contour_count,
        holes=scale_2x_result.retained_hole_count,
    )

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    model = build_resnet50_imagenet1k_v2(args.checkpoint)

    last_progress: dict[str, int] = {"scale_2x": 0, "scale_4x": 0}

    def progress(branch: str):
        def emit_progress(completed: int, total: int) -> None:
            interval = max(args.batch_size * 20, 1)
            if completed == total or completed - last_progress[branch] >= interval:
                last_progress[branch] = completed
                _emit("extraction_progress", branch=branch, completed=completed, total=total)

        return emit_progress

    stage["name"] = "extraction_2x"
    _emit("stage", name=stage["name"], rows=n1)
    scale_2x_dataset = StreamingOpenSlideDataset(
        args.wsi,
        PatchBranchSpec(
            "scale_2x",
            scale_2x_coordinates,
            patch_level=SOURCE_LEVEL,
            patch_size=OUTPUT_PATCH_SIZE,
        ),
    )
    scale_2x_extraction = extract_feature_matrix(
        scale_2x_dataset,
        model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress=progress("scale_2x"),
    )

    stage["name"] = "extraction_4x"
    _emit("stage", name=stage["name"], rows=EXPECTED_4X_COUNT)
    scale_4x_dataset = StreamingOpenSlideDataset(
        args.wsi,
        PatchBranchSpec(
            "scale_4x",
            scale_4x_coordinates,
            patch_level=EXPECTED_4X_LEVEL,
            patch_size=EXPECTED_PATCH_SIZE,
        ),
    )
    scale_4x_extraction = extract_feature_matrix(
        scale_4x_dataset,
        model,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        progress=progress("scale_4x"),
    )
    del model, scale_2x_dataset, scale_4x_dataset
    gc.collect()
    torch.cuda.empty_cache()

    stage["name"] = "feature_validation"
    _emit("stage", name=stage["name"])
    features_2x = validate_feature_matrix(scale_2x_extraction.features, name="scale_2x")
    features_4x = validate_feature_matrix(scale_4x_extraction.features, name="scale_4x")
    _require(features_2x.shape[0] == n1, "2x feature/coordinate row mismatch")
    _require(features_4x.shape[0] == EXPECTED_4X_COUNT, "4x feature/coordinate row mismatch")

    stage["name"] = "concatenation"
    _emit("stage", name=stage["name"])
    bag = build_multiscale_bag(
        features_2x,
        features_4x,
        scale_2x_provenance=BranchProvenanceSpec(
            branch="scale_2x",
            coordinates=scale_2x_coordinates,
            level=SOURCE_LEVEL,
            mpp_x=EFFECTIVE_MPP,
            mpp_y=EFFECTIVE_MPP,
        ),
        scale_4x_provenance=BranchProvenanceSpec(
            branch="scale_4x",
            coordinates=scale_4x_coordinates,
            level=EXPECTED_4X_LEVEL,
            mpp_x=0.9108,
            mpp_y=0.9108,
        ),
    )
    expected_combined_shape = (n1 + EXPECTED_4X_COUNT, 2048)
    _require(tuple(bag.features.shape) == expected_combined_shape, "combined shape mismatch")
    validate_provenance_alignment(bag.features.shape[0], bag.provenance)

    stage["name"] = "healnet_interface"
    _emit("stage", name=stage["name"], patches=expected_combined_shape[0])
    torch.cuda.reset_peak_memory_stats(device)
    healnet_start = time.perf_counter()
    wsi_input = bag.features.transpose(0, 1).unsqueeze(0).contiguous().to(device)
    rna = omic.rna.contiguous().to(device)
    mutation = omic.mutation.contiguous().to(device)
    cnv = omic.cnv.contiguous().to(device)
    smoke = run_one_patient_healnet_smoke(
        official_repo=args.official_repo,
        wsi=wsi_input,
        rna=rna,
        mutation=mutation,
        cnv=cnv,
    )
    torch.cuda.synchronize(device)
    healnet_interface_seconds = time.perf_counter() - healnet_start
    healnet_peak = int(torch.cuda.max_memory_allocated(device))
    del wsi_input, rna, mutation, cnv
    gc.collect()
    torch.cuda.empty_cache()
    _require(_git(args.official_repo.resolve(), "status", "--porcelain") == "", "official HEALNet was modified")

    stage["name"] = "artifacts"
    _emit("stage", name=stage["name"])
    filenames = {
        "scale_2x_coordinates": "scale_2x_coordinates.pt",
        "scale_2x_features": "scale_2x_resnet50_imagenet1k_v2_features.pt",
        "scale_4x_features": "scale_4x_resnet50_imagenet1k_v2_features.pt",
        "combined_features": "combined_multiscale_features.pt",
        "combined_provenance": "combined_provenance.csv",
        "manifest": "pilot_manifest.json",
        "manifest_sha256": "pilot_manifest.sha256",
    }
    paths = {name: staging_dir / filename for name, filename in filenames.items()}
    published_paths = {name: output_dir / filename for name, filename in filenames.items()}
    artifact_records = {
        "scale_2x_coordinates": atomic_save_tensor(scale_2x_coordinates, paths["scale_2x_coordinates"]),
        "scale_2x_features": atomic_save_tensor(features_2x, paths["scale_2x_features"]),
        "scale_4x_features": atomic_save_tensor(features_4x, paths["scale_4x_features"]),
        "combined_features": atomic_save_tensor(bag.features, paths["combined_features"]),
        "combined_provenance": atomic_write_csv(
            provenance_as_dicts(bag.provenance),
            paths["combined_provenance"],
            fieldnames=PROVENANCE_FIELDS,
        ),
    }
    _validate_saved_tensor(paths["scale_2x_coordinates"], scale_2x_coordinates, "2x coordinates")
    _validate_saved_tensor(paths["scale_2x_features"], features_2x, "2x features")
    _validate_saved_tensor(paths["scale_4x_features"], features_4x, "4x features")
    _validate_saved_tensor(paths["combined_features"], bag.features, "combined features")

    resnet_total_model_seconds = (
        scale_2x_extraction.model_forward_seconds
        + scale_4x_extraction.model_forward_seconds
    )
    streaming_total_seconds = (
        scale_2x_extraction.streaming_extraction_seconds
        + scale_4x_extraction.streaming_extraction_seconds
    )
    peak_gpu_memory = max(
        scale_2x_extraction.peak_gpu_memory_bytes,
        scale_4x_extraction.peak_gpu_memory_bytes,
        healnet_peak,
    )
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "BLCA_ONE_PATIENT_PILOT_SUCCESS",
        "scientific_interpretation": "REAL-PATIENT INTERFACE / NUMERICAL SMOKE TEST ONLY",
        "patient_id": BLCA_PILOT_CASE_ID,
        "preflight": preflight,
        "scale_2x_policy": {
            "status": POLICY_STATUS,
            "source_level": SOURCE_LEVEL,
            "source_mpp": SOURCE_MPP,
            "source_footprint": [SOURCE_FOOTPRINT, SOURCE_FOOTPRINT],
            "output_patch": [OUTPUT_PATCH_SIZE, OUTPUT_PATCH_SIZE],
            "effective_mpp": EFFECTIVE_MPP,
            "resampling": "PIL.Image.Resampling.LANCZOS",
            "segmentation_level": scale_2x_result.segmentation_level,
            "segmentation_downsample": list(scale_2x_result.segmentation_downsample),
            "segmentation_mpp": list(scale_2x_result.segmentation_mpp),
            "clam_commit": CLAM_COMMIT,
            "grid_label": GRID_LABEL,
            "grid_anchor": [0, 0],
            "grid_step": [512, 512],
            "ordering": "row_major_y_x",
            "approximation_label": "engineering 2x resampling approximation",
            "exact_0p5_mpp_claim": False,
            "coordinate_generation_seconds": coordinate_seconds,
            "contour_count": scale_2x_result.contour_count,
            "retained_hole_count": scale_2x_result.retained_hole_count,
            "tissue_parameters": {
                "sthresh": 8,
                "mthresh": 7,
                "close": 4,
                "use_otsu": False,
                "a_t": 100,
                "a_h": 16,
                "max_n_holes": 8,
            },
            "contour_rule": "pinned_four_pt_easy",
            "boundary_handling": "reject_incomplete_source_footprints",
            "coordinate_shape": list(scale_2x_coordinates.shape),
            "unique_coordinates": n1,
            "duplicate_coordinates": 0,
            "row_major_y_x": True,
        },
        "resnet_preprocessing": {
            "rgb_conversion": "explicit",
            "branch_input_patch": [256, 256],
            "model_resize": [224, 224],
            "interpolation": "bilinear",
            "antialias": True,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
            "input_dtype": "float32",
            "automatic_mixed_precision": False,
            "classifier_removed": True,
            "model_eval": True,
            "inference_mode": True,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
        },
        "branches": {
            "scale_2x": {
                "coordinate_count": n1,
                "feature_shape": list(features_2x.shape),
                "feature_dtype": str(features_2x.dtype).removeprefix("torch."),
                "all_finite": True,
                "batch_size": scale_2x_extraction.batch_size,
                "streaming_extraction_seconds": scale_2x_extraction.streaming_extraction_seconds,
                "model_forward_seconds": scale_2x_extraction.model_forward_seconds,
                "peak_gpu_memory_bytes": scale_2x_extraction.peak_gpu_memory_bytes,
            },
            "scale_4x": {
                "coordinate_count": EXPECTED_4X_COUNT,
                "feature_shape": list(features_4x.shape),
                "feature_dtype": str(features_4x.dtype).removeprefix("torch."),
                "all_finite": True,
                "batch_size": scale_4x_extraction.batch_size,
                "streaming_extraction_seconds": scale_4x_extraction.streaming_extraction_seconds,
                "model_forward_seconds": scale_4x_extraction.model_forward_seconds,
                "peak_gpu_memory_bytes": scale_4x_extraction.peak_gpu_memory_bytes,
            },
        },
        "concatenation": {
            "operation": "torch.cat",
            "dim": 0,
            "branch_order": ["scale_2x", "scale_4x"],
            "shape": list(bag.features.shape),
            "dtype": "float32",
            "all_finite": True,
            "provenance_rows": len(bag.provenance),
            "provenance_aligned": True,
        },
        "healnet_smoke": {
            **asdict(smoke),
            "interface_seconds": healnet_interface_seconds,
            "peak_gpu_memory_bytes": healnet_peak,
            "trained_prediction": False,
        },
        "timing_and_memory": {
            "resnet_total_model_forward_seconds": resnet_total_model_seconds,
            "streaming_extraction_total_seconds": streaming_total_seconds,
            "healnet_interface_seconds": healnet_interface_seconds,
            "peak_gpu_memory_bytes": peak_gpu_memory,
        },
        "artifacts": {
            name: _record(record, published_path=published_paths[name])
            for name, record in artifact_records.items()
        },
        "safety": {
            "official_repository_modified": False,
            "training_performed": False,
            "kirp_downloaded": False,
            "individual_patch_images_saved": False,
            "verified_4x_hdf5_modified": False,
        },
    }
    manifest["total_runtime_seconds"] = time.perf_counter() - overall_start
    manifest_record = atomic_write_json(manifest, paths["manifest"])
    manifest_sha256_record = atomic_write_text(
        f"{manifest_record.sha256}  {filenames['manifest']}\n",
        paths["manifest_sha256"],
    )
    _require(sha256_file(paths["manifest"]) == manifest_record.sha256, "staged manifest hash mismatch")
    persisted_manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    _require(
        persisted_manifest == _normalise_json_value(manifest),
        "persisted manifest differs from run result",
    )
    _require(
        paths["manifest_sha256"].read_text(encoding="utf-8")
        == f"{manifest_record.sha256}  {filenames['manifest']}\n",
        "staged manifest sidecar mismatch",
    )
    _require(
        _pilot_implementation_identity() == preflight["pilot_implementation"],
        "pilot implementation changed during execution",
    )
    _require(_git(args.official_repo.resolve(), "rev-parse", "HEAD") == OFFICIAL_HEALNET_COMMIT, "official HEALNet HEAD changed")
    _require(_git(args.official_repo.resolve(), "status", "--porcelain") == "", "official HEALNet was modified")
    _publish_output_transaction(staging_dir, output_dir, transaction_lock)
    return {
        "status": "BLCA_ONE_PATIENT_PILOT_SUCCESS",
        "manifest": manifest,
        "publication": {
            "output_directory": str(output_dir),
            "manifest": _record(
                manifest_record,
                published_path=published_paths["manifest"],
            ),
            "manifest_sha256_sidecar": _record(
                manifest_sha256_record,
                published_path=published_paths["manifest_sha256"],
            ),
            "atomic_directory_publish": True,
        },
    }


def main() -> int:
    args = _parse_args()
    stage = {"name": "startup"}
    status_by_stage = {
        "coordinates_2x": "BLOCKED_2X_EXTRACTION",
        "extraction_2x": "BLOCKED_2X_EXTRACTION",
        "extraction_4x": "BLOCKED_4X_EXTRACTION",
        "feature_validation": "BLOCKED_FEATURE_VALIDATION",
        "concatenation": "BLOCKED_CONCATENATION",
        "healnet_interface": "BLOCKED_HEALNET_INTERFACE",
    }
    try:
        result = _run(args, stage)
    except Exception as exc:
        traceback.print_exc()
        print(
            json.dumps(
                {
                    "status": status_by_stage.get(stage["name"], "BLOCKED_RUNTIME"),
                    "failed_stage": stage["name"],
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
