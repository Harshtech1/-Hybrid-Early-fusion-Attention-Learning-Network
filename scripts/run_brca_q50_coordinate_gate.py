#!/usr/bin/env python3
"""Execute the authorized BRCA Q50 one-mask-read CPU coordinate gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Callable, Final

import cv2
import h5py
import numpy as np
import openslide
import yaml

_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
    CoordinateArtifactSetRecord,
    CoordinateBranchMetadata,
    publish_brca_coordinate_artifacts,
)
from multiscale_feature_pilot.src.brca_q25_coordinates import (
    A_H,
    A_T,
    CLOSE,
    MAX_N_HOLES,
    M_THRESH,
    REF_PATCH_SIZE,
    S_THRESH,
    USE_OTSU,
)
from multiscale_feature_pilot.src.brca_q50_coordinates import (
    CLAM_COMMIT,
    EXPECTED_FILENAME,
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_MD5,
    EXPECTED_MPP,
    EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
    EXPECTED_PATIENT_ID,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SLIDE_ID,
    MASK_LEVEL,
    POLICY_STATUS,
    SCALE_2X_EFFECTIVE_MPP,
    SCALE_2X_INTERPOLATION,
    SCALE_2X_OUTPUT_PATCH_SIZE,
    SCALE_2X_SOURCE_FOOTPRINT,
    SCALE_2X_SOURCE_LEVEL,
    SCALE_2X_STEP,
    SCALE_4X_APPROVED_PHYSICAL_MPP,
    SCALE_4X_SOURCE_LEVEL,
    SCALE_4X_SOURCE_PATCH_SIZE,
    build_q50_coordinate_bags,
    expected_q50_observation_from_verified_values,
)


SCHEMA: Final = "BRCA_Q50_COORDINATE_GATE_RESULT_V1"
REPO_ROOT: Final = _SCRIPT_REPO_ROOT
OFFICIAL_REPO: Final = Path("/teamspace/studios/this_studio/healnet")
WSI_PATH: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q50.incoming/"
    "5c1216f3-19ec-4d3c-9bb0-9bd740b79f62/"
    "TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs"
)
OUTPUT_DIR: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q50.coordinates"
)
OFFICIAL_HEAD: Final = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
FROZEN_TAG: Final = "blca-one-patient-pilot-v1"
FROZEN_COMMIT: Final = "df7cf2bda783ab6cc09e95d6a1fa0914da05a433"
POLICY_COMMIT: Final = "bd2b73b4a38ee42e24fc6c596631888dbb09f137"
AUTH_RELATIVE_PATH: Final = Path(
    "multiscale_feature_pilot/config/"
    "brca_q50_coordinate_execution_authorization.yaml"
)
APPROVAL_RELATIVE_PATH: Final = Path(
    "multiscale_feature_pilot/provenance/"
    "brca_q50_coordinate_execution_approval.yaml"
)
AUTH_SHA256: Final = (
    "9e580e6051ec911ad4366b3bace5f0a0352eb498a6c55359b160c106086e74dd"
)
BOUND_FILES: Final = (
    (
        Path("multiscale_feature_pilot/provenance/brca_q50_metadata_gate.yaml"),
        "1d912d81416ad4cc41c128f82e6ac8a02ae4315cd99b1ba098ea1024383efb7a",
    ),
    (
        Path(
            "multiscale_feature_pilot/provenance/"
            "brca_q50_policy_preexecution.yaml"
        ),
        "e387430c028a8c6f791477b12a17027c92da3e0f2b498eb43d0312716baf0bce",
    ),
    (
        Path("multiscale_feature_pilot/config/brca_q50_scale_policy.yaml"),
        "979278fd97d79718464f5918e2250993addbf5638fa721390b09dbc0a74eae32",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q50_scale_policy.py"),
        "d54fccccaca2271552eaccf02fcbb7eaf1561a10a2c07da3d53e6ab4f1951a0c",
    ),
    (
        Path("multiscale_feature_pilot/config/brca_q50_coordinate_policy.yaml"),
        "e5cb83739d3d8fab04da8a63ae1560df04ccc547a79512c219eecb575c0c2114",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q50_coordinates.py"),
        "7dd739667cb6fe0887f3452127c9ff4d43659831d17096922a9f62685149f892",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q25_coordinates.py"),
        "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
        "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf",
    ),
)
CRITICAL_TRACKED_PATHS: Final = (
    Path("scripts/run_brca_q50_coordinate_gate.py"),
    AUTH_RELATIVE_PATH,
    APPROVAL_RELATIVE_PATH,
    *(item[0] for item in BOUND_FILES),
)
MASK_LOCATION: Final = (0, 0)
MASK_SIZE: Final = EXPECTED_LEVEL_DIMENSIONS[MASK_LEVEL]
COORDINATE_POLICY_SHA256: Final = dict(BOUND_FILES)[
    Path("multiscale_feature_pilot/config/brca_q50_coordinate_policy.yaml")
]
_CHUNK_SIZE: Final = 8 * 1024 * 1024
_MINIMUM_AVAILABLE_BYTES: Final = 1_000_000_000


class Q50CoordinateGateError(RuntimeError):
    """Fail-closed gate violation."""


@dataclass(frozen=True)
class GatePaths:
    repo_root: Path = REPO_ROOT
    official_repo: Path = OFFICIAL_REPO
    wsi: Path = WSI_PATH
    output: Path = OUTPUT_DIR
    auth: Path = REPO_ROOT / AUTH_RELATIVE_PATH


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q50CoordinateGateError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wsi_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_CHUNK_SIZE), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _regular_non_symlink(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Q50CoordinateGateError(f"missing {label}: {path}") from exc
    _require(
        stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
        f"{label} must be a regular non-symlink file: {path}",
    )
    return metadata


def _require_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    _require(path.is_absolute(), f"path must be absolute: {path}")
    parts = path.parts
    limit = len(parts) if include_leaf else len(parts) - 1
    current = Path(parts[0])
    for component in parts[1:limit]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise Q50CoordinateGateError(
                f"missing path component: {current}"
            ) from exc
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"symlink component prohibited: {current}",
        )


def _git(repo: Path, *arguments: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _repository_snapshot(paths: GatePaths) -> dict[str, object]:
    official_head = _git(paths.official_repo, "rev-parse", "HEAD")
    official_status = _git(
        paths.official_repo, "status", "--porcelain=v1", "-z"
    )
    _require(official_head == OFFICIAL_HEAD, "official HEALNet HEAD drift")
    _require(official_status == "", "official HEALNet worktree is modified")
    frozen_commit = _git(
        paths.repo_root, "rev-parse", f"{FROZEN_TAG}^{{commit}}"
    )
    _require(frozen_commit == FROZEN_COMMIT, "frozen BLCA tag drift")
    source_head = _git(paths.repo_root, "rev-parse", "HEAD")
    critical_hashes: dict[str, str] = {}
    for relative_path in CRITICAL_TRACKED_PATHS:
        path = paths.repo_root / relative_path
        _regular_non_symlink(path, f"critical source {relative_path}")
        try:
            committed = _git(
                paths.repo_root,
                "show",
                f"HEAD:{relative_path.as_posix()}",
                binary=True,
            )
        except subprocess.CalledProcessError as exc:
            raise Q50CoordinateGateError(
                f"critical source is not tracked at HEAD: {relative_path}"
            ) from exc
        _require(
            path.read_bytes() == committed,
            f"critical source differs from HEAD: {relative_path}",
        )
        critical_hashes[relative_path.as_posix()] = _sha256_file(path)
    return {
        "official_head": official_head,
        "official_status": official_status,
        "frozen_tag": FROZEN_TAG,
        "frozen_commit": frozen_commit,
        "source_commit_at_execution": source_head,
        "critical_execution_source_sha256": critical_hashes,
        "pilot_status_porcelain_v1_z": _git(
            paths.repo_root, "status", "--porcelain=v1", "-z"
        ),
    }


def _expected_bound_policy() -> dict[str, object]:
    names = (
        "metadata_evidence",
        "policy_preexecution_record",
        "scale_policy_config",
        "scale_policy_evaluator",
        "coordinate_policy_config",
        "coordinate_policy_core",
        "reviewed_shared_coordinate_algorithm",
        "coordinate_artifact_publisher",
    )
    return {
        "policy_commit": POLICY_COMMIT,
        **{
            name: {"path": path.as_posix(), "sha256": sha256}
            for name, (path, sha256) in zip(names, BOUND_FILES, strict=True)
        },
    }


def _validate_authorization_semantics(document: object, paths: GatePaths) -> None:
    _require(isinstance(document, dict), "authorization YAML root must be a mapping")
    _require(document.get("schema_version") == 1, "authorization schema drift")
    _require(document.get("cohort") == "TCGA-BRCA", "authorization cohort drift")
    _require(document.get("candidate_label") == "Q50", "authorization candidate drift")
    _require(document.get("phase") == "BRCA_Q50_COORDINATE_GATE", "authorization phase drift")
    _require(
        document.get("status") == "AUTHORIZED_Q50_MASK_READ_AND_COORDINATES_ONLY",
        "authorization status drift",
    )
    approval = document.get("approval_evidence")
    _require(isinstance(approval, dict), "approval evidence missing")
    _require(approval.get("source") == "direct_user_instruction", "approval source drift")
    _require(approval.get("user_statement") == "ok do it start it", "approval statement drift")
    _require(
        document.get("q50_identity")
        == {
            "patient_id": EXPECTED_PATIENT_ID,
            "slide_id": EXPECTED_SLIDE_ID,
            "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
            "filename": EXPECTED_FILENAME,
            "size_bytes": EXPECTED_SIZE_BYTES,
            "md5": EXPECTED_MD5,
            "sha256": EXPECTED_SHA256,
            "exact_path": str(paths.wsi),
        },
        "authorization Q50 identity drift",
    )
    _require(
        document.get("bound_policy_identity") == _expected_bound_policy(),
        "authorization bound-policy chain drift",
    )
    operations = document.get("authorized_operations")
    _require(isinstance(operations, dict), "authorization operations missing")
    _require(
        operations.get("openslide_open_exact_q50") is True,
        "OpenSlide authorization drift",
    )
    _require(operations.get("header_metadata_read") is True, "header authorization drift")
    _require(
        operations.get("mask_pixel_read")
        == {
            "maximum_calls": 1,
            "level": MASK_LEVEL,
            "level_0_location": list(MASK_LOCATION),
            "size_at_level": list(MASK_SIZE),
        },
        "mask-read authorization drift",
    )
    _require(
        operations.get("mask_processing")
        == "resolved_brca_q50_coordinate_policy_v1",
        "mask-processing authorization drift",
    )
    _require(
        operations.get("coordinate_generation")
        == {
            "branches": ["scale_2x", "scale_4x"],
            "coordinate_space": "level_0_x_y",
        },
        "coordinate-generation authorization drift",
    )
    _require(
        operations.get("artifact_publication")
        == {
            "exact_output_directory": str(paths.output),
            "atomic_directory_transaction": "required",
            "no_overwrite": True,
        },
        "artifact-publication authorization drift",
    )
    _require(
        document.get("execution_policy")
        == {
            "tissue_mask_level": 2,
            "tissue_mask_is_shared_by_branches": True,
            "scale_2x": {
                "level_0_footprint": [512, 512],
                "level_0_step": [512, 512],
                "later_output_patch": [256, 256],
                "later_interpolation": "PIL.Image.Resampling.LANCZOS",
                "later_effective_mpp": [0.4936, 0.4936],
            },
            "scale_4x": {
                "level_0_declared_footprint": [1024, 1024],
                "level_0_step": [1024, 1024],
                "later_read_level": 1,
                "later_read_size": [256, 256],
                "later_effective_mpp": [
                    0.9872151105124595,
                    0.9872151105124595,
                ],
            },
            "grid_anchor": [0, 0],
            "reject_incomplete_footprints": True,
            "coordinate_order": "row_major_y_then_x",
            "require_nonempty": True,
            "require_unique": True,
        },
        "execution policy drift",
    )
    prohibited = {
        "q25_rerun_or_modification",
        "level_0_or_level_1_patch_pixel_read",
        "patch_resampling_execution",
        "associated_image_or_thumbnail_read",
        "resnet50_feature_extraction",
        "pt_feature_generation",
        "healnet_execution",
        "model_training",
        "q75_download_open_or_processing",
        "full_cohort_download_or_processing",
        "google_drive_mount_upload_or_delete",
        "raw_wsi_deletion",
        "official_healnet_modification",
        "frozen_blca_modification",
        "git_tracking_of_wsi_hdf5_or_feature_artifacts",
    }
    _require(
        set(document.get("explicitly_prohibited", ())) == prohibited,
        "prohibition set drift",
    )
    _require(
        document.get("required_stop")
        == {
            "after": "Q50_COORDINATE_ARTIFACT_VALIDATION_AND_REPORT",
            "gpu_feature_extraction_authorized_by_this_record": False,
            "q75_authorized": False,
            "training_authorized": False,
        },
        "required-stop boundary drift",
    )


def _validate_bound_sources(paths: GatePaths) -> dict[str, str]:
    expected = ((AUTH_RELATIVE_PATH, AUTH_SHA256), *BOUND_FILES)
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in expected:
        path = paths.repo_root / relative_path
        _regular_non_symlink(path, f"bound source {relative_path}")
        actual_sha256 = _sha256_file(path)
        _require(
            actual_sha256 == expected_sha256,
            f"bound source SHA-256 drift: {relative_path}",
        )
        observed[relative_path.as_posix()] = actual_sha256
    _require(
        paths.auth == paths.repo_root / AUTH_RELATIVE_PATH,
        "authorization path drift",
    )
    try:
        authorization = yaml.safe_load(paths.auth.read_text(encoding="utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise Q50CoordinateGateError("authorization YAML is unreadable") from exc
    _validate_authorization_semantics(authorization, paths)
    return observed


def _validate_paths(paths: GatePaths) -> None:
    _require(MASK_LOCATION == (0, 0), "mask read location constant drift")
    _require(MASK_LEVEL == 2, "mask read level constant drift")
    _require(MASK_SIZE == (6_247, 4_083), "mask read size constant drift")
    _require(paths.repo_root == REPO_ROOT, "pilot repository path drift")
    _require(paths.official_repo == OFFICIAL_REPO, "official repository path drift")
    _require(paths.wsi == WSI_PATH, "Q50 WSI path drift")
    _require(paths.output == OUTPUT_DIR, "Q50 output path drift")
    _require_no_symlink_components(paths.wsi, include_leaf=True)
    _require_no_symlink_components(paths.output, include_leaf=False)
    for repo in (paths.repo_root.resolve(), paths.official_repo.resolve()):
        _require(
            not paths.output.is_relative_to(repo),
            f"coordinate output must be outside Git worktree: {repo}",
        )
    _require(not os.path.lexists(paths.output), "coordinate output already exists")
    lock = paths.output.parent / f".{paths.output.name}.lock"
    _require(not os.path.lexists(lock), "coordinate output lock already exists")
    staging_prefix = f".{paths.output.name}.staging."
    _require(
        not any(
            item.name.startswith(staging_prefix)
            for item in paths.output.parent.iterdir()
        ),
        "coordinate output has a stale/active staging sibling",
    )


def _df_snapshot(path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["df", "-B1", "--output=source,size,used,avail,pcent,target", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    _require(len(lines) == 2, "unexpected df output")
    fields = lines[1].split()
    _require(len(fields) >= 6, "unexpected df data row")
    filesystem, total, used, available, percent = fields[:5]
    mountpoint = " ".join(fields[5:])
    values = (int(total), int(used), int(available))
    _require(
        all(value >= 0 for value in values)
        and values[2] >= _MINIMUM_AVAILABLE_BYTES,
        "insufficient physical filesystem headroom for coordinate gate",
    )
    return {
        "measurement": "runtime_df_capacity_not_lightning_logical_quota",
        "filesystem": filesystem,
        "total_bytes": values[0],
        "used_bytes": values[1],
        "available_bytes": values[2],
        "use_percent": percent,
        "mountpoint": mountpoint,
        "minimum_required_available_bytes": _MINIMUM_AVAILABLE_BYTES,
        "lightning_dashboard_evidence": {
            "source": "user_supplied_dashboard_screenshot",
            "displayed_total": "200 GB",
            "displayed_used": "5.08 GB",
        },
    }


def _verify_wsi(path: Path) -> tuple[dict[str, object], tuple[int, int, int, int]]:
    metadata = _regular_non_symlink(path, "Q50 WSI")
    _require(path.name == EXPECTED_FILENAME, "Q50 WSI filename drift")
    _require(metadata.st_size == EXPECTED_SIZE_BYTES, "Q50 WSI size drift")
    partials = tuple(
        entry.name
        for entry in path.parent.iterdir()
        if entry.name.endswith((".partial", ".part"))
    )
    _require(not partials, "partial WSI transfer sibling exists")
    md5, sha256 = _wsi_hashes(path)
    _require(md5 == EXPECTED_MD5, "Q50 WSI MD5 drift")
    _require(sha256 == EXPECTED_SHA256, "Q50 WSI SHA-256 drift")
    identity_token = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "md5": md5,
        "sha256": sha256,
        "regular_non_symlink": True,
        "partial_siblings": [],
    }, identity_token


def _read_exact_mask(slide) -> tuple[np.ndarray, object, int, str]:
    properties = slide.properties
    try:
        mpp_x = float(properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(properties[openslide.PROPERTY_NAME_MPP_Y])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise Q50CoordinateGateError("Q50 MPP header is missing or invalid") from exc
    observation = expected_q50_observation_from_verified_values(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        filename=EXPECTED_FILENAME,
        size_bytes=EXPECTED_SIZE_BYTES,
        md5=EXPECTED_MD5,
        sha256=EXPECTED_SHA256,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        level_dimensions=tuple(tuple(item) for item in slide.level_dimensions),
        openslide_level_downsamples=tuple(
            float(item) for item in slide.level_downsamples
        ),
    )
    pil_image = slide.read_region(MASK_LOCATION, MASK_LEVEL, MASK_SIZE)
    try:
        array = np.ascontiguousarray(np.asarray(pil_image))
    finally:
        close = getattr(pil_image, "close", None)
        if callable(close):
            close()
    _require(array.dtype == np.uint8, "mask image dtype drift")
    _require(
        array.ndim == 3
        and array.shape[:2] == (MASK_SIZE[1], MASK_SIZE[0])
        and array.shape[2] in (3, 4),
        "mask image shape/channels drift",
    )
    mask_sha256 = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return array, observation, 1, mask_sha256


def _metadata_for_branch(
    branch: str,
    *,
    mask_image_channels: int,
    mask_image_sha256: str,
    contour_count: int,
    retained_hole_count: int,
) -> CoordinateBranchMetadata:
    shared = dict(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        wsi_filename=EXPECTED_FILENAME,
        wsi_size_bytes=EXPECTED_SIZE_BYTES,
        wsi_md5=EXPECTED_MD5,
        wsi_sha256=EXPECTED_SHA256,
        level_0_dimensions=EXPECTED_LEVEL_DIMENSIONS[0],
        mask_level=MASK_LEVEL,
        mask_level_dimensions=EXPECTED_LEVEL_DIMENSIONS[MASK_LEVEL],
        openslide_reported_mask_downsample=(
            EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES[MASK_LEVEL]
        ),
        mask_image_channels=mask_image_channels,
        mask_image_sha256=mask_image_sha256,
        mask_parameters={
            "color_space": "HSV_saturation",
            "sthresh": S_THRESH,
            "mthresh": M_THRESH,
            "close": CLOSE,
            "use_otsu": USE_OTSU,
            "a_t": A_T,
            "a_h": A_H,
            "max_n_holes": MAX_N_HOLES,
            "reference_patch_size": REF_PATCH_SIZE,
            "contour_rule": "pinned_four_pt_easy_any_probe_on_or_inside",
            "hole_rule": "reject_only_when_mapped_patch_center_is_strictly_inside",
        },
        contour_count=contour_count,
        retained_hole_count=retained_hole_count,
        clam_commit=CLAM_COMMIT,
        policy_sha256=COORDINATE_POLICY_SHA256,
    )
    if branch == "scale_2x":
        return CoordinateBranchMetadata(
            branch=branch,
            source_level=SCALE_2X_SOURCE_LEVEL,
            source_level_dimensions=EXPECTED_LEVEL_DIMENSIONS[
                SCALE_2X_SOURCE_LEVEL
            ],
            openslide_reported_source_downsample=(
                EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES[SCALE_2X_SOURCE_LEVEL]
            ),
            source_patch_size=(SCALE_2X_SOURCE_FOOTPRINT,) * 2,
            output_patch_size=(SCALE_2X_OUTPUT_PATCH_SIZE,) * 2,
            level_0_declared_footprint=(SCALE_2X_SOURCE_FOOTPRINT,) * 2,
            level_0_step=(SCALE_2X_STEP,) * 2,
            target_mpp=0.5,
            effective_mpp=SCALE_2X_EFFECTIVE_MPP,
            interpolation=SCALE_2X_INTERPOLATION,
            resampling="explicit_2x_spatial_downsample",
            geometry_compatibility="LEVEL0_IDENTITY_GEOMETRY",
            **shared,
        )
    _require(branch == "scale_4x", "unknown coordinate branch")
    return CoordinateBranchMetadata(
        branch=branch,
        source_level=SCALE_4X_SOURCE_LEVEL,
        source_level_dimensions=EXPECTED_LEVEL_DIMENSIONS[SCALE_4X_SOURCE_LEVEL],
        openslide_reported_source_downsample=(
            EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES[SCALE_4X_SOURCE_LEVEL]
        ),
        source_patch_size=(SCALE_4X_SOURCE_PATCH_SIZE,) * 2,
        output_patch_size=(SCALE_4X_SOURCE_PATCH_SIZE,) * 2,
        level_0_declared_footprint=(1024, 1024),
        level_0_step=(1024, 1024),
        target_mpp=1.0,
        effective_mpp=SCALE_4X_APPROVED_PHYSICAL_MPP,
        interpolation="none",
        resampling="none",
        geometry_compatibility="CLAM_INT_CAST_GEOMETRY_COMPATIBLE",
        **shared,
    )


def _record_for_branch(
    record: CoordinateArtifactSetRecord, branch: str
) -> dict[str, object]:
    item = record.branch_for(branch)
    return {
        "coordinate_count": item.coordinate_count,
        "h5_path": str(item.path),
        "h5_size_bytes": item.size_bytes,
        "h5_sha256": item.sha256,
        "coordinates_sha256": item.coordinates_sha256,
    }


def run_coordinate_gate(
    *,
    paths: GatePaths = GatePaths(),
    slide_factory: Callable[[str], object] = openslide.OpenSlide,
    mask_reader: Callable[[object], tuple[np.ndarray, object, int, str]] = (
        _read_exact_mask
    ),
    coordinate_builder: Callable[..., object] = build_q50_coordinate_bags,
    publisher: Callable[..., CoordinateArtifactSetRecord] = (
        publish_brca_coordinate_artifacts
    ),
) -> dict[str, object]:
    """Run the exact Q50 gate; seams exist only for fail-closed mocked tests."""

    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    timings: dict[str, float] = {}
    _validate_paths(paths)
    bound_sources = _validate_bound_sources(paths)
    repositories_before = _repository_snapshot(paths)
    storage = _df_snapshot(paths.output.parent)
    stage = time.perf_counter()
    wsi, identity_token = _verify_wsi(paths.wsi)
    timings["wsi_identity_and_hash_seconds"] = time.perf_counter() - stage

    slide = None
    stage = time.perf_counter()
    try:
        slide = slide_factory(str(paths.wsi))
        mask_image, observation, read_region_count, mask_image_sha256 = mask_reader(
            slide
        )
    finally:
        if slide is not None:
            slide.close()
    timings["openslide_header_and_mask_read_seconds"] = time.perf_counter() - stage
    _require(read_region_count == 1, "exactly one mask read is required")

    post_read = paths.wsi.lstat()
    _require(
        (
            post_read.st_dev,
            post_read.st_ino,
            post_read.st_size,
            post_read.st_mtime_ns,
        )
        == identity_token,
        "Q50 WSI changed during the mask read",
    )
    stage = time.perf_counter()
    bags = coordinate_builder(mask_image, observation=observation)
    timings["segmentation_and_coordinate_generation_seconds"] = (
        time.perf_counter() - stage
    )
    _require(bags.policy_status == POLICY_STATUS, "coordinate policy status drift")
    common_metadata = {
        "mask_image_channels": int(mask_image.shape[2]),
        "mask_image_sha256": mask_image_sha256,
        "contour_count": bags.contour_count,
        "retained_hole_count": bags.retained_hole_count,
    }
    scale_2x_metadata = _metadata_for_branch("scale_2x", **common_metadata)
    scale_4x_metadata = _metadata_for_branch("scale_4x", **common_metadata)

    # Every mutable-input and source gate runs before the terminal publisher.
    _validate_paths(paths)
    final_wsi, final_identity_token = _verify_wsi(paths.wsi)
    _require(final_wsi == wsi, "Q50 WSI content changed before publication")
    _require(final_identity_token == identity_token, "Q50 WSI identity changed")
    _require(
        _validate_bound_sources(paths) == bound_sources,
        "bound sources changed during coordinate generation",
    )
    repositories_prepublication = _repository_snapshot(paths)
    _require(
        repositories_prepublication == repositories_before,
        "repository state changed during coordinate generation",
    )

    stage = time.perf_counter()
    record = publisher(
        paths.output,
        scale_2x_coordinates=bags.scale_2x,
        scale_4x_coordinates=bags.scale_4x,
        scale_2x_metadata=scale_2x_metadata,
        scale_4x_metadata=scale_4x_metadata,
    )
    timings["atomic_publication_and_validation_seconds"] = time.perf_counter() - stage
    timings["total_seconds"] = time.perf_counter() - started

    return {
        "schema": SCHEMA,
        "status": "BRCA_Q50_COORDINATES_VERIFIED",
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id": EXPECTED_PATIENT_ID,
        "slide_id": EXPECTED_SLIDE_ID,
        "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
        "authorization_sha256": AUTH_SHA256,
        "bound_source_sha256": bound_sources,
        "source_commit_at_execution": repositories_before[
            "source_commit_at_execution"
        ],
        "critical_execution_source_sha256": repositories_before[
            "critical_execution_source_sha256"
        ],
        "runtime_versions": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "h5py": h5py.__version__,
            "openslide_python": openslide.__version__,
            "openslide_library": openslide.__library_version__,
        },
        "wsi": wsi,
        "header": {
            "mpp_x": observation.mpp_x,
            "mpp_y": observation.mpp_y,
            "level_dimensions": [list(item) for item in observation.level_dimensions],
            "openslide_level_downsamples": list(
                observation.openslide_level_downsamples
            ),
        },
        "mask": {
            "read_region_calls": read_region_count,
            "location_level_0": list(MASK_LOCATION),
            "level": MASK_LEVEL,
            "size_at_level": list(MASK_SIZE),
            "channels": int(mask_image.shape[2]),
            "hash_serialization": "contiguous_uint8_C_order_raw_bytes",
            "contiguous_uint8_bytes_sha256": mask_image_sha256,
            "contour_count": bags.contour_count,
            "retained_hole_count": bags.retained_hole_count,
        },
        "coordinate_artifacts": {
            "directory": str(record.directory),
            "manifest_path": str(record.manifest_path),
            "manifest_sha256": record.manifest_sha256,
            "scale_2x": _record_for_branch(record, "scale_2x"),
            "scale_4x": _record_for_branch(record, "scale_4x"),
        },
        "storage": storage,
        "repositories_unchanged_before_publication": True,
        "operations": {
            "openslide_open_count": 1,
            "header_reads": 1,
            "mask_read_region_calls": 1,
            "level_0_or_level_1_patch_pixel_reads": 0,
            "thumbnail_reads": 0,
            "associated_image_reads": 0,
            "patch_resampling": 0,
            "resnet50_feature_extraction": 0,
            "pt_generation": 0,
            "healnet_execution": 0,
            "training": 0,
            "q25_reruns": 0,
            "q75_operations": 0,
            "google_drive_operations": 0,
            "raw_wsi_deletions": 0,
        },
        "timings": timings,
        "required_stop_reached": True,
    }


def _strict_json(document: dict[str, object]) -> str:
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-exact-q50-coordinate-gate",
        action="store_true",
        help="Acknowledge the exact Q50 one-mask-read CPU boundary.",
    )
    args = parser.parse_args(argv)
    if not args.execute_exact_q50_coordinate_gate:
        parser.error("explicit --execute-exact-q50-coordinate-gate is required")
    try:
        result = run_coordinate_gate()
    except Exception as exc:
        print(
            _strict_json(
                {
                    "schema": SCHEMA,
                    "status": "BRCA_Q50_COORDINATE_GATE_BLOCKED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "required_stop_reached": True,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(_strict_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
