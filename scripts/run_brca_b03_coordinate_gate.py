#!/usr/bin/env python3
"""Execute the exact authorized B03 single-mask coordinate gate."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import openslide
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiscale_feature_pilot.src.brca_coordinate_artifacts import (  # noqa: E402
    BRANCH_FILENAMES,
    MANIFEST_FILENAME,
    MANIFEST_SHA256_FILENAME,
    SCHEMA,
    CoordinateBranchMetadata,
    sha256_file,
    validate_brca_coordinate_artifacts,
)
from multiscale_feature_pilot.src.brca_omic import (  # noqa: E402
    BRCA_RELEASE_ARCHIVE_SHA256,
    load_brca_patient_omics,
)
from multiscale_feature_pilot.src.brca_q25_coordinates import (  # noqa: E402
    generate_level_0_lattice_coordinates,
    segment_tissue_contours,
)


WORKSPACE = ROOT.parent
DATA = WORKSPACE / "brca_pilot_data"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_b03_coordinate_execution_authorization.yaml"
POLICY = ROOT / "multiscale_feature_pilot/config/brca_b03_scale_coordinate_policy.yaml"
POLICY_REVIEW = ROOT / "multiscale_feature_pilot/provenance/brca_b03_scale_coordinate_policy_review.yaml"
HEADER_RESULT = ROOT / "multiscale_feature_pilot/provenance/brca_b03_header_metadata_result.yaml"
INCOMING = DATA / "BRCA_BATCH_B03.incoming"
WSI = INCOMING / "266c3852-f5d0-4815-94d9-dac5b0ff8276/TCGA-AR-A1AY-01Z-00-DX1.6AC0BE3B-FFC5-4EDA-9E40-B18CAAC52B81.svs"
PARCEL = INCOMING / "266c3852-f5d0-4815-94d9-dac5b0ff8276/logs/TCGA-AR-A1AY-01Z-00-DX1.6AC0BE3B-FFC5-4EDA-9E40-B18CAAC52B81.svs.parcel"
OMIC = WORKSPACE / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip"
DEST = DATA / "BRCA_BATCH_B03.coordinates"

PATIENT = "TCGA-AR-A1AY"
SLIDE = "TCGA-AR-A1AY-01Z-00-DX1.6AC0BE3B-FFC5-4EDA-9E40-B18CAAC52B81.svs"
UUID = "266c3852-f5d0-4815-94d9-dac5b0ff8276"
SIZE = 918454431
MD5 = "1980b183d5bb948c2fc263af62a4b1b4"
SHA256 = "4ef4ac79ce3cc0bfc5a4ea62985f080f0f877dca4c7e43191a04a35b2eba8228"
PARCEL_SIZE = 55285
PARCEL_SHA256 = "0b85bab6359b670616f9499089128a55a1aa2d5ac4577e6fbc495a701d136a4b"
AUTH_SHA256 = "102346d52105eb30463e2daf4c022fd3fdec1cef2ae5957eed44776b3ce9228a"
POLICY_SHA256 = "58de5cedb887a3d0faf12c68e8629d26cae4cf7ff8a867e20742d395890a522c"
POLICY_REVIEW_SHA256 = "067b3ef6bb7d77778af0813a9c093b19ac11e69a769835b9b50ca10dd4a071f9"
HEADER_RESULT_SHA256 = "f0d3ea106e49f24220ded29743b31459caed06d325a22cc123a6ea2addb77111"
DIMS = ((93296, 58121), (23324, 14530), (5831, 3632), (2915, 1816))
DOWNSAMPLES = (1.0, 4.000034411562285, 16.00123898678414, 32.00522239895422)
MPP = (0.2468, 0.2468)
MASK = (5831, 3632)

BOUND_FILES = {
    Path("scripts/run_brca_b03_coordinate_gate.py"): None,
    Path("multiscale_feature_pilot/config/brca_b03_coordinate_execution_authorization.yaml"): AUTH_SHA256,
    Path("multiscale_feature_pilot/config/brca_b03_scale_coordinate_policy.yaml"): POLICY_SHA256,
    Path("multiscale_feature_pilot/provenance/brca_b03_scale_coordinate_policy_review.yaml"): POLICY_REVIEW_SHA256,
    Path("multiscale_feature_pilot/provenance/brca_b03_header_metadata_result.yaml"): HEADER_RESULT_SHA256,
    Path("multiscale_feature_pilot/src/brca_q25_coordinates.py"): "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82",
    Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"): "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf",
    Path("multiscale_feature_pilot/src/brca_omic.py"): "5061039913cb0dd4e8e30958c8b4b76f088396f7841255dd94234c3b462d5a5d",
}
ALLOWED_STATUS = {
    "M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}
EXPECTED_TREE = {
    f"{UUID}/{SLIDE}",
    f"{UUID}/logs/{SLIDE}.parcel",
}


class GateError(RuntimeError):
    """Raised when the B03 coordinate gate fails closed."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise GateError(message)


def digest(path: Path, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def digest_fd(file_descriptor: int, algorithm: str = "sha256") -> str:
    result = hashlib.new(algorithm)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(file_descriptor, 8 * 1024 * 1024)
        if not chunk:
            break
        result.update(chunk)
    os.lseek(file_descriptor, 0, os.SEEK_SET)
    return result.hexdigest()


def git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def validate_repository(expected_commit: str) -> None:
    require(len(expected_commit) == 40, "expected source commit must be full length")
    require(git("rev-parse", "HEAD") == expected_commit, "HEAD drift")
    statuses = set(filter(None, git("status", "--short").splitlines()))
    require(statuses <= ALLOWED_STATUS, f"unexpected Git status {statuses - ALLOWED_STATUS}")
    for relative, expected_hash in BOUND_FILES.items():
        committed = subprocess.run(
            ("git", "show", f"HEAD:{relative.as_posix()}"),
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        actual = (ROOT / relative).read_bytes()
        require(actual == committed, f"critical worktree drift: {relative}")
        if expected_hash is not None:
            require(hashlib.sha256(actual).hexdigest() == expected_hash, f"critical hash drift: {relative}")


def validate_inputs() -> None:
    auth = yaml.safe_load(AUTH.read_text())
    require(auth["status"] == "AUTHORIZED_B03_SINGLE_MASK_READ_AND_COORDINATE_PUBLICATION", "authorization status")
    require(auth["authorized_read"] == {
        "openslide_open_count": 1,
        "read_region_count": 1,
        "level_0_location": [0, 0],
        "level": 2,
        "size_at_level": [5831, 3632],
    }, "authorization read scope")
    require(digest(AUTH) == AUTH_SHA256 and digest(POLICY) == POLICY_SHA256, "authorization or policy drift")
    require(digest(POLICY_REVIEW) == POLICY_REVIEW_SHA256 and digest(HEADER_RESULT) == HEADER_RESULT_SHA256, "review or header drift")
    require(not os.path.lexists(DEST), "coordinate destination already exists")
    require(not tuple(DATA.glob(".BRCA_BATCH_B03.coordinates.staging.*")), "coordinate staging collision")
    require(INCOMING.is_dir() and not INCOMING.is_symlink(), "incoming directory drift")
    files = {path.relative_to(INCOMING).as_posix() for path in INCOMING.rglob("*") if path.is_file()}
    require(files == EXPECTED_TREE, "incoming tree drift")
    for path in (WSI, PARCEL):
        require(path.is_file() and not path.is_symlink(), f"non-regular input: {path}")
    require(WSI.stat().st_size == SIZE and PARCEL.stat().st_size == PARCEL_SIZE, "input size drift")
    require(digest(WSI, "md5") == MD5 and digest(WSI) == SHA256, "WSI hash drift")
    require(digest(PARCEL) == PARCEL_SHA256, "parcel hash drift")
    omics = load_brca_patient_omics(
        OMIC,
        case_id=PATIENT,
        slide_id=SLIDE,
        expected_archive_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
    )
    require(
        omics.source_row_index == "372"
        and all(tensor.device.type == "cpu" and bool(tensor.isfinite().all()) for tensor in (omics.rna, omics.mutation, omics.cnv)),
        "Omic drift",
    )


def read_exact_mask() -> tuple[np.ndarray, str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(WSI, flags)
    calls = 0
    try:
        before = os.fstat(file_descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size == SIZE, "held WSI drift")
        token = (before.st_dev, before.st_ino, before.st_size)
        require(digest_fd(file_descriptor, "md5") == MD5 and digest_fd(file_descriptor) == SHA256, "held WSI hash drift")
        slide = openslide.OpenSlide(f"/proc/self/fd/{file_descriptor}")
        try:
            dimensions = tuple(tuple(map(int, item)) for item in slide.level_dimensions)
            downsamples = tuple(float(value) for value in slide.level_downsamples)
            mpp = (
                float(slide.properties[openslide.PROPERTY_NAME_MPP_X]),
                float(slide.properties[openslide.PROPERTY_NAME_MPP_Y]),
            )
            require(dimensions == DIMS and downsamples == DOWNSAMPLES and mpp == MPP, "header drift")
            image = slide.read_region((0, 0), 2, MASK)
            calls += 1
            mask = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            require(mask.shape == (3632, 5831, 4), "mask shape drift")
        finally:
            slide.close()
        after = os.fstat(file_descriptor)
        current = os.stat(WSI, follow_symlinks=False)
        require(token == (after.st_dev, after.st_ino, after.st_size), "held descriptor drift")
        require(token == (current.st_dev, current.st_ino, current.st_size), "WSI path identity drift")
        require(calls == 1, "mask read count drift")
        require(digest_fd(file_descriptor, "md5") == MD5 and digest_fd(file_descriptor) == SHA256, "final held WSI hash drift")
    finally:
        os.close(file_descriptor)
    return mask, hashlib.sha256(mask.tobytes(order="C")).hexdigest()


def branch_metadata(branch: str, mask_sha256: str, contours: int, holes: int) -> CoordinateBranchMetadata:
    is_scale_2x = branch == "scale_2x"
    level = 0 if is_scale_2x else 1
    return CoordinateBranchMetadata(
        branch=branch,
        patient_id=PATIENT,
        slide_id=SLIDE,
        gdc_file_uuid=UUID,
        wsi_filename=SLIDE,
        wsi_size_bytes=SIZE,
        wsi_md5=MD5,
        wsi_sha256=SHA256,
        level_0_dimensions=DIMS[0],
        source_level=level,
        source_level_dimensions=DIMS[level],
        openslide_reported_source_downsample=DOWNSAMPLES[level],
        source_patch_size=(512, 512) if is_scale_2x else (256, 256),
        output_patch_size=(256, 256),
        level_0_declared_footprint=(512, 512) if is_scale_2x else (1024, 1024),
        level_0_step=(512, 512) if is_scale_2x else (1024, 1024),
        target_mpp=0.5 if is_scale_2x else 1.0,
        effective_mpp=(0.4936, 0.4936) if is_scale_2x else (0.9872084927735719, 0.9872084927735719),
        interpolation="PIL.Image.Resampling.LANCZOS" if is_scale_2x else "none",
        resampling="linear_factor_2" if is_scale_2x else "none",
        mask_level=2,
        mask_level_dimensions=MASK,
        openslide_reported_mask_downsample=DOWNSAMPLES[2],
        mask_image_channels=4,
        mask_image_sha256=mask_sha256,
        mask_parameters={"sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False, "a_t": 100, "a_h": 16, "max_n_holes": 8, "reference_patch_size": 512},
        contour_count=contours,
        retained_hole_count=holes,
        clam_commit="26e0b6c4873e112f1ccd74cd834894c4ab7a2934",
        policy_sha256=POLICY_SHA256,
        geometry_compatibility="GLOBAL_LEVEL0_NATIVE" if is_scale_2x else "CLAM_INT_CAST_GEOMETRY_COMPATIBLE",
    )


def publish(coordinates_2x: np.ndarray, coordinates_4x: np.ndarray, metadata_2x: CoordinateBranchMetadata, metadata_4x: CoordinateBranchMetadata):
    stage = DATA / f".BRCA_BATCH_B03.coordinates.staging.{uuid.uuid4().hex}"
    require(not os.path.lexists(stage) and not os.path.lexists(DEST), "publication collision")
    stage.mkdir(mode=0o700)
    branches = {}
    for branch, coordinates, metadata in (
        ("scale_2x", coordinates_2x, metadata_2x),
        ("scale_4x", coordinates_4x, metadata_4x),
    ):
        path = stage / BRANCH_FILENAMES[branch]
        with h5py.File(path, "x") as output:
            dataset = output.create_dataset("coords", data=np.ascontiguousarray(coordinates, dtype=np.int64))
            for key, value in metadata.to_attributes().items():
                dataset.attrs[key] = value
            output.flush()
        branches[branch] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "coordinates_sha256": hashlib.sha256(np.ascontiguousarray(coordinates, dtype="<i8").tobytes()).hexdigest(),
            "coordinate_count": int(coordinates.shape[0]),
            "attributes": metadata.to_attributes(),
        }
    payload = (json.dumps({"schema": SCHEMA, "branches": branches}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    (stage / MANIFEST_FILENAME).write_bytes(payload)
    anchor = hashlib.sha256(payload).hexdigest()
    (stage / MANIFEST_SHA256_FILENAME).write_text(f"{anchor}  {MANIFEST_FILENAME}\n", encoding="ascii", newline="")
    validate_brca_coordinate_artifacts(stage, expected_manifest_sha256=anchor)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(stage), -100, os.fsencode(DEST), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise GateError("destination appeared before publication")
        raise OSError(error_number, os.strerror(error_number))
    return validate_brca_coordinate_artifacts(DEST, expected_manifest_sha256=anchor)


def run(expected_commit: str) -> dict:
    started = datetime.now(timezone.utc)
    timer = time.perf_counter()
    validate_repository(expected_commit)
    validate_inputs()
    mask, mask_sha256 = read_exact_mask()
    geometry = segment_tissue_contours(mask, level_0_dimensions=DIMS[0], mask_dimensions=MASK)
    coordinates_2x = generate_level_0_lattice_coordinates(level_0_dimensions=DIMS[0], level_0_patch_size=512, level_0_step=512, geometry=geometry)
    coordinates_4x = generate_level_0_lattice_coordinates(level_0_dimensions=DIMS[0], level_0_patch_size=1024, level_0_step=1024, geometry=geometry)
    require(coordinates_2x.shape[0] > 0 and coordinates_4x.shape[0] > 0, "empty coordinate bag")
    contour_count = len(geometry.contours)
    hole_count = sum(len(items) for items in geometry.holes)
    validate_repository(expected_commit)
    validate_inputs()
    record = publish(
        coordinates_2x,
        coordinates_4x,
        branch_metadata("scale_2x", mask_sha256, contour_count, hole_count),
        branch_metadata("scale_4x", mask_sha256, contour_count, hole_count),
    )
    return {
        "status": "BRCA_B03_COORDINATES_VERIFIED",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": time.perf_counter() - timer,
        "source_commit": expected_commit,
        "manifest_sha256": record.manifest_sha256,
        "scale_2x_count": int(coordinates_2x.shape[0]),
        "scale_4x_count": int(coordinates_4x.shape[0]),
        "contour_count": contour_count,
        "retained_hole_count": hole_count,
        "read_region_calls": 1,
        "mask_sha256": mask_sha256,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-commit", required=True)
    print(json.dumps(run(parser.parse_args().expected_source_commit), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
