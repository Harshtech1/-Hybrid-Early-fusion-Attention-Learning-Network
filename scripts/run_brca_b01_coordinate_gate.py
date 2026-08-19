#!/usr/bin/env python3
"""Execute the authorized B01 single-mask coordinate gate and stop."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import uuid

import h5py
import numpy as np
import openslide
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
    BRANCH_FILENAMES,
    CoordinateBranchMetadata,
    MANIFEST_FILENAME,
    MANIFEST_SHA256_FILENAME,
    SCHEMA,
    sha256_file,
    validate_brca_coordinate_artifacts,
)
from multiscale_feature_pilot.src.brca_omic import (
    BRCA_RELEASE_ARCHIVE_SHA256,
    load_brca_patient_omics,
)
from multiscale_feature_pilot.src.brca_q25_coordinates import (
    generate_level_0_lattice_coordinates,
    segment_tissue_contours,
)


WORKSPACE = ROOT.parent
DATA_ROOT = WORKSPACE / "brca_pilot_data"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_b01_coordinate_execution_authorization.yaml"
POLICY = ROOT / "multiscale_feature_pilot/config/brca_b01_scale_coordinate_policy.yaml"
WSI = DATA_ROOT / "BRCA_BATCH_B01.incoming/0a886f18-c44c-4b5e-b243-6df6e27f426a/TCGA-GI-A2C8-01Z-00-DX1.09BD8AC9-645A-4C8B-9B36-77D833BDBA09.svs"
OMIC = WORKSPACE / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip"
DESTINATION = DATA_ROOT / "BRCA_BATCH_B01.coordinates"

PATIENT = "TCGA-GI-A2C8"
SLIDE = "TCGA-GI-A2C8-01Z-00-DX1.09BD8AC9-645A-4C8B-9B36-77D833BDBA09.svs"
GDC_UUID = "0a886f18-c44c-4b5e-b243-6df6e27f426a"
SIZE = 408_704_377
MD5 = "a9f830d456b4a1fe0e9bb5b5b99f4b7e"
SHA256 = "b3c27e220c5c3961600782af42e91a52ea0b85a710d8d4aa722831ea00f9ad5f"
AUTH_SHA256 = "2701184101fd95c95dd8bebb7cca715ae5010793693bbe1a30e5ae7454926d49"
POLICY_SHA256 = "9a98a0edad90f1e82ad28025b9164c136c39041ad27fb5342251bdb8d062cb60"
LEVEL_DIMS = ((63_784, 39_311), (15_946, 9_827), (3_986, 2_456))
LEVEL_DS = (1.0, 4.00015264068383, 16.004057258221366)
MPP = (0.2468, 0.2468)
MASK_LEVEL = 2
MASK_SIZE = (3_986, 2_456)

CRITICAL = (
    Path("scripts/run_brca_b01_coordinate_gate.py"),
    Path("multiscale_feature_pilot/config/brca_b01_coordinate_execution_authorization.yaml"),
    Path("multiscale_feature_pilot/config/brca_b01_scale_coordinate_policy.yaml"),
    Path("multiscale_feature_pilot/src/brca_q25_coordinates.py"),
    Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
    Path("multiscale_feature_pilot/src/brca_omic.py"),
)
ALLOWED_STATUS = {
    " M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}


class GateError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise GateError(message)


def digest_path(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip()


def preflight(expected_commit: str) -> None:
    require(len(expected_commit) == 40, "full expected commit required")
    require(git("rev-parse", "HEAD") == expected_commit, "HEAD drift")
    statuses = set(filter(None, git("status", "--short").splitlines()))
    require(statuses <= ALLOWED_STATUS, f"unexpected Git status: {statuses - ALLOWED_STATUS}")
    for relative in CRITICAL:
        working = (ROOT / relative).read_bytes()
        committed = subprocess.run(
            ("git", "show", f"HEAD:{relative.as_posix()}"), cwd=ROOT, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).stdout
        require(working == committed, f"critical source drift: {relative}")
    require(digest_path(AUTH, "sha256") == AUTH_SHA256, "authorization drift")
    require(digest_path(POLICY, "sha256") == POLICY_SHA256, "policy drift")
    auth = yaml.safe_load(AUTH.read_text())
    require(auth["status"] == "AUTHORIZED_B01_SINGLE_MASK_READ_AND_COORDINATE_PUBLICATION", "authorization status drift")
    require(not os.path.lexists(DESTINATION), "coordinate destination already exists")
    require(not tuple(DATA_ROOT.glob(".BRCA_BATCH_B01.coordinates.staging.*")), "stale staging exists")
    require(not WSI.is_symlink() and WSI.is_file(), "WSI must be regular non-symlink")
    require(WSI.stat().st_size == SIZE, "WSI size drift")
    require(digest_path(WSI, "md5") == MD5, "WSI MD5 drift")
    require(digest_path(WSI, "sha256") == SHA256, "WSI SHA256 drift")
    omic = load_brca_patient_omics(
        OMIC, case_id=PATIENT, slide_id=SLIDE,
        expected_archive_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
    )
    require(omic.source_row_index == "924", "Omic row drift")
    require(all(t.device.type == "cpu" and bool(t.isfinite().all()) for t in (omic.rna, omic.mutation, omic.cnv)), "Omic tensor drift")


def read_single_mask() -> tuple[np.ndarray, dict[str, object]]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(WSI, flags)
    calls = 0
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode) and before.st_size == SIZE, "held WSI identity drift")
        slide = openslide.OpenSlide(f"/proc/self/fd/{descriptor}")
        try:
            dimensions = tuple(tuple(map(int, item)) for item in slide.level_dimensions)
            downsamples = tuple(float(item) for item in slide.level_downsamples)
            mpp = (
                float(slide.properties[openslide.PROPERTY_NAME_MPP_X]),
                float(slide.properties[openslide.PROPERTY_NAME_MPP_Y]),
            )
            require(dimensions == LEVEL_DIMS and downsamples == LEVEL_DS and mpp == MPP, "OpenSlide header drift")
            image = slide.read_region((0, 0), MASK_LEVEL, MASK_SIZE)
            calls += 1
            array = np.asarray(image, dtype=np.uint8)
            require(array.shape == (MASK_SIZE[1], MASK_SIZE[0], 4), "mask shape drift")
            mask = np.ascontiguousarray(array)
        finally:
            slide.close()
        require(calls == 1, "exactly one read_region call required")
        require(os.fstat(descriptor).st_size == SIZE, "held WSI changed")
    finally:
        os.close(descriptor)
    require(digest_path(WSI, "md5") == MD5 and digest_path(WSI, "sha256") == SHA256, "final WSI hash drift")
    return mask, {
        "read_region_calls": calls,
        "mask_sha256": hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
    }


def metadata(branch: str, mask_sha: str, contour_count: int, hole_count: int) -> CoordinateBranchMetadata:
    source_level = 0 if branch == "scale_2x" else 1
    source_patch = (512, 512) if branch == "scale_2x" else (256, 256)
    return CoordinateBranchMetadata(
        branch=branch, patient_id=PATIENT, slide_id=SLIDE, gdc_file_uuid=GDC_UUID,
        wsi_filename=SLIDE, wsi_size_bytes=SIZE, wsi_md5=MD5, wsi_sha256=SHA256,
        level_0_dimensions=LEVEL_DIMS[0], source_level=source_level,
        source_level_dimensions=LEVEL_DIMS[source_level],
        openslide_reported_source_downsample=LEVEL_DS[source_level],
        source_patch_size=source_patch, output_patch_size=(256, 256),
        level_0_declared_footprint=(512, 512) if branch == "scale_2x" else (1024, 1024),
        level_0_step=(512, 512) if branch == "scale_2x" else (1024, 1024),
        target_mpp=0.5 if branch == "scale_2x" else 1.0,
        effective_mpp=(0.4936, 0.4936) if branch == "scale_2x" else (0.9872376717207693, 0.9872376717207693),
        interpolation="PIL.Image.Resampling.LANCZOS" if branch == "scale_2x" else "none",
        resampling="linear_factor_2" if branch == "scale_2x" else "none",
        mask_level=2, mask_level_dimensions=MASK_SIZE,
        openslide_reported_mask_downsample=LEVEL_DS[2], mask_image_channels=4,
        mask_image_sha256=mask_sha,
        mask_parameters={"sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False, "a_t": 100, "a_h": 16, "max_n_holes": 8, "reference_patch_size": 512},
        contour_count=contour_count, retained_hole_count=hole_count,
        clam_commit="26e0b6c4873e112f1ccd74cd834894c4ab7a2934",
        policy_sha256=POLICY_SHA256,
        geometry_compatibility="GLOBAL_LEVEL0_NATIVE" if branch == "scale_2x" else "CLAM_INT_CAST_GEOMETRY_COMPATIBLE",
    )


def publish_zero_delete(scale_2x: np.ndarray, scale_4x: np.ndarray, meta2: CoordinateBranchMetadata, meta4: CoordinateBranchMetadata) -> object:
    staging = DATA_ROOT / f".BRCA_BATCH_B01.coordinates.staging.{uuid.uuid4().hex}"
    require(not os.path.lexists(staging) and not os.path.lexists(DESTINATION), "publication path collision")
    staging.mkdir(mode=0o700)
    branches: dict[str, object] = {}
    for branch, coords, meta in (("scale_2x", scale_2x, meta2), ("scale_4x", scale_4x, meta4)):
        path = staging / BRANCH_FILENAMES[branch]
        with h5py.File(path, "x") as handle:
            dataset = handle.create_dataset("coords", data=np.ascontiguousarray(coords, dtype=np.int64))
            for key, value in meta.to_attributes().items():
                dataset.attrs[key] = value
            handle.flush()
        branches[branch] = {
            "filename": path.name, "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "coordinates_sha256": hashlib.sha256(np.ascontiguousarray(coords, dtype="<i8").tobytes()).hexdigest(),
            "coordinate_count": int(coords.shape[0]), "attributes": meta.to_attributes(),
        }
    payload = (json.dumps({"schema": SCHEMA, "branches": branches}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode("utf-8")
    (staging / MANIFEST_FILENAME).write_bytes(payload)
    manifest_sha = hashlib.sha256(payload).hexdigest()
    (staging / MANIFEST_SHA256_FILENAME).write_text(f"{manifest_sha}  {MANIFEST_FILENAME}\n", encoding="ascii", newline="")
    validate_brca_coordinate_artifacts(staging, expected_manifest_sha256=manifest_sha)
    libc = ctypes.CDLL(None, use_errno=True)
    result = libc.renameat2(-100, os.fsencode(staging), -100, os.fsencode(DESTINATION), 1)
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise GateError("destination appeared during publication")
        raise OSError(number, os.strerror(number))
    return validate_brca_coordinate_artifacts(DESTINATION, expected_manifest_sha256=manifest_sha)


def run(expected_commit: str) -> dict[str, object]:
    preflight(expected_commit)
    mask, read_record = read_single_mask()
    geometry = segment_tissue_contours(mask, level_0_dimensions=LEVEL_DIMS[0], mask_dimensions=MASK_SIZE)
    scale_2x = generate_level_0_lattice_coordinates(level_0_dimensions=LEVEL_DIMS[0], level_0_patch_size=512, level_0_step=512, geometry=geometry)
    scale_4x = generate_level_0_lattice_coordinates(level_0_dimensions=LEVEL_DIMS[0], level_0_patch_size=1024, level_0_step=1024, geometry=geometry)
    require(scale_2x.shape[0] > 0 and scale_4x.shape[0] > 0, "coordinate bags must be nonempty")
    contour_count = len(geometry.contours)
    hole_count = sum(len(items) for items in geometry.holes)
    record = publish_zero_delete(scale_2x, scale_4x, metadata("scale_2x", str(read_record["mask_sha256"]), contour_count, hole_count), metadata("scale_4x", str(read_record["mask_sha256"]), contour_count, hole_count))
    return {
        "status": "BRCA_B01_COORDINATES_VERIFIED",
        "manifest_sha256": record.manifest_sha256,
        "scale_2x_count": int(scale_2x.shape[0]),
        "scale_4x_count": int(scale_4x.shape[0]),
        "contour_count": contour_count,
        "retained_hole_count": hole_count,
        **read_record,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.expected_source_commit), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
