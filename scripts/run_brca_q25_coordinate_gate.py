#!/usr/bin/env python3
"""Execute the single, authorized BRCA Q25 CPU coordinate gate.

This entry point is intentionally narrower than a general WSI tool.  It binds
one WSI, one mask read and one external output directory.  It never reads
level-0/level-1 patch pixels and imports no feature/model/training code.
"""

from __future__ import annotations

import argparse
import cv2
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import h5py
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Callable, Final

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
    validate_brca_coordinate_artifacts,
)
from multiscale_feature_pilot.src.brca_q25_coordinates import (
    A_H,
    A_T,
    CLAM_COMMIT,
    CLOSE,
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
    MAX_N_HOLES,
    M_THRESH,
    POLICY_STATUS,
    REF_PATCH_SIZE,
    SCALE_2X_EFFECTIVE_MPP,
    SCALE_2X_INTERPOLATION,
    SCALE_2X_OUTPUT_PATCH_SIZE,
    SCALE_2X_SOURCE_FOOTPRINT,
    SCALE_2X_SOURCE_LEVEL,
    SCALE_2X_STEP,
    SCALE_4X_APPROVED_PHYSICAL_MPP,
    SCALE_4X_GEOMETRY_LABEL,
    SCALE_4X_SOURCE_FOOTPRINT,
    SCALE_4X_SOURCE_LEVEL,
    S_THRESH,
    USE_OTSU,
    build_q25_coordinate_bags,
    expected_q25_observation_from_verified_values,
)


SCHEMA: Final = "BRCA_Q25_COORDINATE_GATE_RESULT_V1"
REPO_ROOT: Final = _SCRIPT_REPO_ROOT
OFFICIAL_REPO: Final = Path("/teamspace/studios/this_studio/healnet")
WSI_PATH: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q25.incoming/"
    "dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6/"
    "TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs"
)
OUTPUT_DIR: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q25.coordinates"
)
OFFICIAL_HEAD: Final = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
FROZEN_TAG: Final = "blca-one-patient-pilot-v1"
FROZEN_COMMIT: Final = "df7cf2bda783ab6cc09e95d6a1fa0914da05a433"
AUTH_RELATIVE_PATH: Final = Path(
    "multiscale_feature_pilot/config/"
    "brca_q25_coordinate_execution_authorization.yaml"
)
AUTH_SHA256: Final = (
    "3797f428f5d1d49334fc2c0665325728318083d5deb8831deec5ad1f560ac617"
)
BOUND_FILES: Final = (
    (
        Path("multiscale_feature_pilot/config/brca_q25_scale_policy.yaml"),
        "fd54080543706d56cf6fe336b61630f3f8c09a6741e4fcf5ea7c42801d0ff816",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q25_scale_policy.py"),
        "6ede1db26638363f1ecca2fbd8db3f2ae222eacff291ba6de44634f26d148a89",
    ),
    (
        Path("multiscale_feature_pilot/config/brca_q25_coordinate_policy.yaml"),
        "85410751aec43b14997fa4c0e2a611ceb329178f788df04f336031104b697d43",
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
    Path("scripts/run_brca_q25_coordinate_gate.py"),
    AUTH_RELATIVE_PATH,
    Path("multiscale_feature_pilot/config/brca_q25_scale_policy.yaml"),
    Path("multiscale_feature_pilot/src/brca_q25_scale_policy.py"),
    Path("multiscale_feature_pilot/config/brca_q25_coordinate_policy.yaml"),
    Path("multiscale_feature_pilot/src/brca_q25_coordinates.py"),
    Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
    Path(
        "multiscale_feature_pilot/provenance/"
        "brca_q25_coordinate_execution_approval.yaml"
    ),
)
MASK_LOCATION: Final = (0, 0)
MASK_SIZE: Final = EXPECTED_LEVEL_DIMENSIONS[MASK_LEVEL]
_CHUNK_SIZE: Final = 8 * 1024 * 1024


class Q25CoordinateGateError(RuntimeError):
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
        raise Q25CoordinateGateError(message)


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
        raise Q25CoordinateGateError(f"missing {label}: {path}") from exc
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
            raise Q25CoordinateGateError(f"missing path component: {current}") from exc
        _require(not stat.S_ISLNK(metadata.st_mode), f"symlink component prohibited: {current}")


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
    official_status = _git(paths.official_repo, "status", "--porcelain=v1", "-z")
    _require(official_head == OFFICIAL_HEAD, "official HEALNet HEAD drift")
    _require(official_status == "", "official HEALNet worktree is modified")
    frozen_commit = _git(paths.repo_root, "rev-parse", f"{FROZEN_TAG}^{{commit}}")
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
            raise Q25CoordinateGateError(
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


def _validate_authorization_semantics(document: object, paths: GatePaths) -> None:
    _require(isinstance(document, dict), "authorization YAML root must be a mapping")
    _require(document.get("schema_version") == 1, "authorization schema drift")
    _require(document.get("cohort") == "TCGA-BRCA", "authorization cohort drift")
    _require(document.get("candidate_label") == "Q25", "authorization candidate drift")
    _require(document.get("phase") == "BRCA_Q25_COORDINATE_GATE", "authorization phase drift")
    _require(
        document.get("status") == "AUTHORIZED_Q25_MASK_READ_AND_COORDINATES_ONLY",
        "authorization status drift",
    )
    _require(
        document.get("q25_identity")
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
        "authorization Q25 identity drift",
    )
    bound = document.get("bound_policy_identity")
    _require(isinstance(bound, dict), "authorization bound-policy identity missing")
    observed_bound = {
        (Path(value["path"]), value["sha256"])
        for value in bound.values()
        if isinstance(value, dict) and set(value) == {"path", "sha256"}
    }
    _require(observed_bound == set(BOUND_FILES), "authorization bound-policy chain drift")
    operations = document.get("authorized_operations")
    _require(isinstance(operations, dict), "authorization operations missing")
    _require(operations.get("openslide_open_exact_q25") is True, "OpenSlide authorization drift")
    _require(operations.get("header_metadata_read") is True, "header authorization drift")
    _require(
        operations.get("mask_pixel_read")
        == {
            "maximum_calls": 1,
            "level": MASK_LEVEL,
            "level_0_location": list(MASK_LOCATION),
            "size_at_level": list(EXPECTED_LEVEL_DIMENSIONS[MASK_LEVEL]),
        },
        "mask-read authorization drift",
    )
    _require(
        operations.get("coordinate_generation")
        == {
            "branches": ["scale_2x", "scale_4x"],
            "coordinate_space": "level_0_x_y",
        },
        "coordinate-generation authorization drift",
    )
    publication = operations.get("artifact_publication")
    _require(isinstance(publication, dict), "artifact publication authorization missing")
    _require(
        publication.get("exact_output_directory") == str(paths.output),
        "authorized output path drift",
    )
    _require(publication.get("atomic_directory_transaction") == "required", "atomic publication authorization drift")
    _require(publication.get("no_overwrite") is True, "no-overwrite authorization drift")
    prohibited = {
        "level_0_or_level_1_patch_pixel_read",
        "patch_resampling_execution",
        "associated_image_or_thumbnail_read",
        "resnet50_feature_extraction",
        "pt_feature_generation",
        "healnet_execution",
        "model_training",
        "q50_or_q75_download_or_open",
        "full_cohort_download_or_processing",
        "google_drive_mount_upload_or_delete",
        "official_healnet_modification",
        "frozen_blca_modification",
        "git_tracking_of_wsi_hdf5_or_feature_artifacts",
    }
    _require(set(document.get("explicitly_prohibited", ())) == prohibited, "prohibition set drift")
    _require(
        document.get("required_stop")
        == {
            "after": "Q25_COORDINATE_ARTIFACT_VALIDATION_AND_REPORT",
            "gpu_switch_authorized_by_this_record": False,
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
    _require(paths.auth == paths.repo_root / AUTH_RELATIVE_PATH, "authorization path drift")
    try:
        authorization = yaml.safe_load(paths.auth.read_text(encoding="utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise Q25CoordinateGateError("authorization YAML is unreadable") from exc
    _validate_authorization_semantics(authorization, paths)
    return observed


def _validate_paths(paths: GatePaths) -> None:
    _require(MASK_LOCATION == (0, 0), "mask read location constant drift")
    _require(MASK_LEVEL == 2, "mask read level constant drift")
    _require(
        MASK_SIZE == EXPECTED_LEVEL_DIMENSIONS[2],
        "mask read size constant drift",
    )
    _require(paths.repo_root == REPO_ROOT, "pilot repository path drift")
    _require(paths.official_repo == OFFICIAL_REPO, "official repository path drift")
    _require(paths.wsi == WSI_PATH, "Q25 WSI path drift")
    _require(paths.output == OUTPUT_DIR, "Q25 output path drift")
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
        not any(item.name.startswith(staging_prefix) for item in paths.output.parent.iterdir()),
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
    _require(all(value >= 0 for value in values) and values[2] > 0, "df reports no available bytes")
    return {
        "measurement": "runtime_df_filesystem_capacity_not_lightning_logical_quota",
        "filesystem": filesystem,
        "total_bytes": values[0],
        "used_bytes": values[1],
        "available_bytes": values[2],
        "use_percent": percent,
        "mountpoint": mountpoint,
        "lightning_dashboard_evidence": {
            "source": "user_supplied_dashboard_screenshot",
            "displayed_total": "200 GB",
            "displayed_used": "3.41 GB",
        },
    }


def _verify_wsi(path: Path) -> tuple[dict[str, object], tuple[int, int, int, int]]:
    metadata = _regular_non_symlink(path, "Q25 WSI")
    _require(path.name == EXPECTED_FILENAME, "Q25 WSI filename drift")
    _require(metadata.st_size == EXPECTED_SIZE_BYTES, "Q25 WSI size drift")
    partials = tuple(
        entry.name
        for entry in path.parent.iterdir()
        if entry.name.endswith((".partial", ".part"))
    )
    _require(not partials, "partial WSI transfer sibling exists")
    md5, sha256 = _wsi_hashes(path)
    _require(md5 == EXPECTED_MD5, "Q25 WSI MD5 drift")
    _require(sha256 == EXPECTED_SHA256, "Q25 WSI SHA-256 drift")
    identity_token = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "md5": md5,
        "sha256": sha256,
        "regular_non_symlink": True,
        "partial_siblings": [],
    }, identity_token


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
        openslide_reported_mask_downsample=EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES[MASK_LEVEL],
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
        policy_sha256=dict(BOUND_FILES)[Path("multiscale_feature_pilot/config/brca_q25_coordinate_policy.yaml")],
    )
    if branch == "scale_2x":
        return CoordinateBranchMetadata(
            branch=branch,
            source_level=SCALE_2X_SOURCE_LEVEL,
            source_level_dimensions=EXPECTED_LEVEL_DIMENSIONS[SCALE_2X_SOURCE_LEVEL],
            openslide_reported_source_downsample=EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES[SCALE_2X_SOURCE_LEVEL],
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
        openslide_reported_source_downsample=EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES[SCALE_4X_SOURCE_LEVEL],
        source_patch_size=(SCALE_4X_SOURCE_FOOTPRINT,) * 2,
        output_patch_size=(SCALE_4X_SOURCE_FOOTPRINT,) * 2,
        level_0_declared_footprint=(1024, 1024),
        level_0_step=(1024, 1024),
        target_mpp=1.0,
        effective_mpp=SCALE_4X_APPROVED_PHYSICAL_MPP,
        interpolation="none",
        resampling="none",
        geometry_compatibility=SCALE_4X_GEOMETRY_LABEL,
        **shared,
    )


def _read_exact_mask(slide) -> tuple[np.ndarray, object, int]:
    properties = slide.properties
    observation = expected_q25_observation_from_verified_values(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        filename=EXPECTED_FILENAME,
        size_bytes=EXPECTED_SIZE_BYTES,
        md5=EXPECTED_MD5,
        sha256=EXPECTED_SHA256,
        mpp_x=float(properties["openslide.mpp-x"]),
        mpp_y=float(properties["openslide.mpp-y"]),
        level_dimensions=tuple(tuple(item) for item in slide.level_dimensions),
        openslide_level_downsamples=tuple(float(item) for item in slide.level_downsamples),
    )
    mask = slide.read_region(MASK_LOCATION, MASK_LEVEL, MASK_SIZE)
    array = np.ascontiguousarray(np.asarray(mask))
    _require(array.dtype == np.uint8, "mask image dtype drift")
    _require(
        array.ndim == 3
        and array.shape[:2] == (MASK_SIZE[1], MASK_SIZE[0])
        and array.shape[2] in (3, 4),
        "mask image shape/channels drift",
    )
    return array, observation, 1


def _record_for_branch(record: CoordinateArtifactSetRecord, branch: str) -> dict[str, object]:
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
    publisher: Callable[..., CoordinateArtifactSetRecord] = publish_brca_coordinate_artifacts,
) -> dict[str, object]:
    """Run the exact Q25 gate; injectable seams exist only for mocked tests."""

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
    read_region_count = 0
    stage = time.perf_counter()
    try:
        slide = slide_factory(str(paths.wsi))
        mask_image, observation, read_region_count = _read_exact_mask(slide)
    finally:
        if slide is not None:
            slide.close()
    timings["openslide_header_and_mask_read_seconds"] = time.perf_counter() - stage
    _require(read_region_count == 1, "exactly one mask read is required")
    post_read = paths.wsi.lstat()
    _require(
        (post_read.st_dev, post_read.st_ino, post_read.st_size, post_read.st_mtime_ns)
        == identity_token,
        "Q25 WSI changed during the gate",
    )

    mask_image_sha256 = hashlib.sha256(mask_image.tobytes(order="C")).hexdigest()
    stage = time.perf_counter()
    bags = build_q25_coordinate_bags(mask_image, observation=observation)
    timings["segmentation_and_coordinate_generation_seconds"] = time.perf_counter() - stage
    _require(bags.policy_status == POLICY_STATUS, "coordinate policy status drift")
    common_metadata = {
        "mask_image_channels": int(mask_image.shape[2]),
        "mask_image_sha256": mask_image_sha256,
        "contour_count": bags.contour_count,
        "retained_hole_count": bags.retained_hole_count,
    }
    scale_2x_metadata = _metadata_for_branch("scale_2x", **common_metadata)
    scale_4x_metadata = _metadata_for_branch("scale_4x", **common_metadata)

    # Recheck destination immediately before the sole external publication.
    _validate_paths(paths)
    stage = time.perf_counter()
    record = publisher(
        paths.output,
        scale_2x_coordinates=bags.scale_2x,
        scale_4x_coordinates=bags.scale_4x,
        scale_2x_metadata=scale_2x_metadata,
        scale_4x_metadata=scale_4x_metadata,
    )
    record = validate_brca_coordinate_artifacts(
        paths.output,
        expected_manifest_sha256=record.manifest_sha256,
    )
    timings["atomic_publication_and_validation_seconds"] = time.perf_counter() - stage
    repositories_after = _repository_snapshot(paths)
    _require(repositories_after == repositories_before, "repository state changed during gate")
    timings["total_seconds"] = time.perf_counter() - started

    return {
        "schema": SCHEMA,
        "status": "BRCA_Q25_COORDINATES_VERIFIED",
        "started_at_utc": started_utc,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id": EXPECTED_PATIENT_ID,
        "slide_id": EXPECTED_SLIDE_ID,
        "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
        "authorization_sha256": AUTH_SHA256,
        "bound_source_sha256": bound_sources,
        "source_commit_at_execution": repositories_before["source_commit_at_execution"],
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
            "openslide_level_downsamples": list(observation.openslide_level_downsamples),
        },
        "mask": {
            "read_region_calls": read_region_count,
            "location_level_0": list(MASK_LOCATION),
            "level": MASK_LEVEL,
            "size_at_level": list(MASK_SIZE),
            "channels": int(mask_image.shape[2]),
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
        "repositories_unchanged": True,
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
            "q50_q75_operations": 0,
            "google_drive_operations": 0,
        },
        "timings": timings,
        "required_stop_reached": True,
    }


def _strict_json(document: dict[str, object]) -> str:
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-exact-q25-coordinate-gate",
        action="store_true",
        help="Acknowledge the exact one-mask-read CPU execution boundary.",
    )
    args = parser.parse_args(argv)
    if not args.execute_exact_q25_coordinate_gate:
        parser.error("explicit --execute-exact-q25-coordinate-gate is required")
    try:
        result = run_coordinate_gate()
    except Exception as exc:  # CLI emits a strict machine-readable stop record.
        print(
            _strict_json(
                {
                    "schema": SCHEMA,
                    "status": "BRCA_Q25_COORDINATE_GATE_BLOCKED",
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
