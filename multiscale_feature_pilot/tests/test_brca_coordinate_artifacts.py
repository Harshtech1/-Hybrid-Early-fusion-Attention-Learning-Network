from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
    BRANCH_FILENAMES,
    COORDINATE_FRAME,
    LATTICE,
    MANIFEST_FILENAME,
    MANIFEST_SHA256_FILENAME,
    MAPPING,
    SELECTION_CLAIM,
    CoordinateArtifactExistsError,
    CoordinateBranchMetadata,
    CoordinateHashMismatchError,
    CoordinatePublicationInProgressError,
    CoordinateValidationError,
    publish_brca_coordinate_artifacts,
    validate_brca_coordinate_artifacts,
)


def _metadata(branch: str) -> CoordinateBranchMetadata:
    shared = dict(
        patient_id="TCGA-AA-0001",
        slide_id="TCGA-AA-0001-01Z-00-DX1",
        gdc_file_uuid="11111111-2222-4333-8444-555555555555",
        wsi_filename="TCGA-AA-0001-01Z-00-DX1.TEST.svs",
        wsi_size_bytes=123456,
        wsi_md5="1" * 32,
        wsi_sha256="2" * 64,
        level_0_dimensions=(4096, 4100),
        mask_level=2,
        mask_level_dimensions=(256, 256),
        openslide_reported_mask_downsample=16.001,
        mask_image_channels=4,
        mask_image_sha256="5" * 64,
        mask_parameters={
            "sthresh": 8,
            "mthresh": 7,
            "close": 4,
            "use_otsu": False,
            "a_t": 100,
            "a_h": 16,
            "max_n_holes": 8,
        },
        contour_count=2,
        retained_hole_count=1,
        clam_commit="3" * 40,
        policy_sha256="4" * 64,
    )
    if branch == "scale_2x":
        return CoordinateBranchMetadata(
            branch=branch,
            source_level=0,
            source_level_dimensions=(4096, 4100),
            openslide_reported_source_downsample=1.0,
            source_patch_size=(512, 512),
            output_patch_size=(256, 256),
            level_0_declared_footprint=(512, 512),
            level_0_step=(512, 512),
            target_mpp=0.5,
            effective_mpp=(0.505, 0.505),
            interpolation="PIL.Image.Resampling.LANCZOS",
            resampling="explicit_2x_spatial_downsample",
            geometry_compatibility="LEVEL0_IDENTITY_GEOMETRY",
            **shared,
        )
    return CoordinateBranchMetadata(
        branch=branch,
        source_level=1,
        source_level_dimensions=(1024, 1024),
        openslide_reported_source_downsample=4.00005934365913,
        source_patch_size=(256, 256),
        output_patch_size=(256, 256),
        level_0_declared_footprint=(1024, 1024),
        level_0_step=(1024, 1024),
        target_mpp=1.0,
        effective_mpp=(1.0100149842739303, 1.0100149842739303),
        interpolation="none",
        resampling="none",
        geometry_compatibility="CLAM_INT_CAST_GEOMETRY_COMPATIBLE",
        **shared,
    )


def _coordinates(step: int) -> np.ndarray:
    return np.asarray(
        [[0, 0], [step, 0], [0, step], [step, step]],
        dtype=np.int64,
    )


def _publish(tmp_path: Path):
    destination = tmp_path / "q25_coordinates"
    result = publish_brca_coordinate_artifacts(
        destination,
        scale_2x_coordinates=_coordinates(512),
        scale_4x_coordinates=_coordinates(1024),
        scale_2x_metadata=_metadata("scale_2x"),
        scale_4x_metadata=_metadata("scale_4x"),
    )
    return destination, result


def test_publishes_exact_atomic_artifact_set_and_validates_it(tmp_path: Path) -> None:
    destination, result = _publish(tmp_path)

    assert {entry.name for entry in destination.iterdir()} == {
        *BRANCH_FILENAMES.values(),
        MANIFEST_FILENAME,
        MANIFEST_SHA256_FILENAME,
    }
    assert result.directory == destination
    assert result.manifest_sha256_path.read_text(encoding="ascii") == (
        f"{result.manifest_sha256}  {MANIFEST_FILENAME}\n"
    )
    validated = validate_brca_coordinate_artifacts(
        destination,
        expected_manifest_sha256=result.manifest_sha256,
    )
    assert [item.branch for item in validated.branches] == ["scale_2x", "scale_4x"]
    assert validated.branch_for("scale_2x").coordinate_count == 4
    assert validated.branch_for("scale_4x").coordinate_count == 4
    assert not list(tmp_path.glob(".q25_coordinates.staging.*"))
    assert not (tmp_path / ".q25_coordinates.lock").exists()

    with h5py.File(destination / BRANCH_FILENAMES["scale_4x"], "r") as h5:
        coords = h5["coords"]
        assert coords.shape == (4, 2)
        assert coords.dtype == np.dtype(np.int64)
        assert coords.attrs["coordinate_frame"] == COORDINATE_FRAME
        assert coords.attrs["lattice"] == LATTICE
        assert coords.attrs["coordinate_mapping"] == MAPPING
        assert coords.attrs["coordinate_selection_claim"] == SELECTION_CLAIM
        # Reported scalar and dimension-derived axis geometry are distinct.
        assert coords.attrs["openslide_reported_source_downsample"] == pytest.approx(4.00005934365913)
        assert coords.attrs["source_to_level_0_scale_x"] == 4.0
        assert coords.attrs["source_to_level_0_scale_y"] == pytest.approx(4100 / 1024)
        assert coords.attrs["mask_contour_scale_x"] == 16.0
        assert coords.attrs["mask_contour_scale_y"] == pytest.approx(4100 / 256)
        assert coords.attrs["mask_image_channels"] == 4
        assert coords.attrs["mask_image_sha256"] == "5" * 64
        assert coords.attrs["mask_image_dtype"] == "uint8"
        assert coords.attrs["mask_image_channel_order"] == "RGBA"
        assert (
            coords.attrs["mask_image_hash_serialization"]
            == "contiguous_uint8_C_order_raw_bytes"
        )
        assert coords.attrs["contour_count"] == 2
        assert coords.attrs["retained_hole_count"] == 1


@pytest.mark.parametrize(
    ("coordinates", "message"),
    [
        (np.empty((0, 2), dtype=np.int64), "nonempty"),
        (np.zeros((2, 2), dtype=np.int32), "dtype must be int64"),
        (np.zeros((2, 3), dtype=np.int64), r"shape \[N,2\]"),
        (np.asarray([[0, 0], [0, 0]], dtype=np.int64), "unique row-major"),
        (np.asarray([[0, 512], [512, 0]], dtype=np.int64), "unique row-major"),
        (np.asarray([[1, 0]], dtype=np.int64), "off the global"),
        (np.asarray([[4096, 0]], dtype=np.int64), "incomplete level-0"),
        (
            np.asarray([[2**63 - 512, 2**63 - 512]], dtype=np.int64),
            "incomplete level-0",
        ),
    ],
)
def test_rejects_invalid_coordinates_without_leaving_output(
    tmp_path: Path,
    coordinates: np.ndarray,
    message: str,
) -> None:
    destination = tmp_path / "invalid"
    with pytest.raises(CoordinateValidationError, match=message):
        publish_brca_coordinate_artifacts(
            destination,
            scale_2x_coordinates=coordinates,
            scale_4x_coordinates=_coordinates(1024),
            scale_2x_metadata=_metadata("scale_2x"),
            scale_4x_metadata=_metadata("scale_4x"),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".invalid.staging.*"))
    assert not (tmp_path / ".invalid.lock").exists()


def test_never_overwrites_or_resumes_an_existing_destination(tmp_path: Path) -> None:
    destination, result = _publish(tmp_path)
    original_manifest = result.manifest_path.read_bytes()
    with pytest.raises(CoordinateArtifactExistsError, match="already exists"):
        publish_brca_coordinate_artifacts(
            destination,
            scale_2x_coordinates=_coordinates(512),
            scale_4x_coordinates=_coordinates(1024),
            scale_2x_metadata=_metadata("scale_2x"),
            scale_4x_metadata=_metadata("scale_4x"),
        )
    assert result.manifest_path.read_bytes() == original_manifest

    stale_destination = tmp_path / "stale"
    (tmp_path / ".stale.staging.abandoned").mkdir()
    with pytest.raises(CoordinatePublicationInProgressError, match="staging"):
        publish_brca_coordinate_artifacts(
            stale_destination,
            scale_2x_coordinates=_coordinates(512),
            scale_4x_coordinates=_coordinates(1024),
            scale_2x_metadata=_metadata("scale_2x"),
            scale_4x_metadata=_metadata("scale_4x"),
        )


def test_lock_blocks_second_cooperative_publisher(tmp_path: Path) -> None:
    lock = tmp_path / ".locked.lock"
    lock.write_text("pid=123\n", encoding="ascii")
    with pytest.raises(CoordinatePublicationInProgressError, match="locked"):
        publish_brca_coordinate_artifacts(
            tmp_path / "locked",
            scale_2x_coordinates=_coordinates(512),
            scale_4x_coordinates=_coordinates(1024),
            scale_2x_metadata=_metadata("scale_2x"),
            scale_4x_metadata=_metadata("scale_4x"),
        )
    assert lock.read_text(encoding="ascii") == "pid=123\n"


def test_exact_hash_anchor_and_h5_hash_detect_tampering(tmp_path: Path) -> None:
    destination, result = _publish(tmp_path)
    with pytest.raises(CoordinateHashMismatchError, match="manifest SHA-256"):
        validate_brca_coordinate_artifacts(
            destination,
            expected_manifest_sha256="0" * 64,
        )

    h5_path = destination / BRANCH_FILENAMES["scale_2x"]
    with h5_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(CoordinateHashMismatchError, match="HDF5 SHA-256"):
        validate_brca_coordinate_artifacts(
            destination,
            expected_manifest_sha256=result.manifest_sha256,
        )


def test_rejects_unexpected_files_and_sidecar_drift(tmp_path: Path) -> None:
    destination, result = _publish(tmp_path)
    (destination / "resume.partial").write_text("ambiguous", encoding="utf-8")
    with pytest.raises(CoordinateValidationError, match="file set is not exact"):
        validate_brca_coordinate_artifacts(
            destination,
            expected_manifest_sha256=result.manifest_sha256,
        )
    (destination / "resume.partial").unlink()
    result.manifest_sha256_path.write_text("0" * 64 + "  coordinate_manifest.json\n")
    with pytest.raises(CoordinateHashMismatchError, match="sidecar"):
        validate_brca_coordinate_artifacts(
            destination,
            expected_manifest_sha256=result.manifest_sha256,
        )


def test_branch_identity_policy_and_declared_geometry_must_match(tmp_path: Path) -> None:
    with pytest.raises(CoordinateValidationError, match="shared field patient_id"):
        publish_brca_coordinate_artifacts(
            tmp_path / "identity_drift",
            scale_2x_coordinates=_coordinates(512),
            scale_4x_coordinates=_coordinates(1024),
            scale_2x_metadata=_metadata("scale_2x"),
            scale_4x_metadata=replace(_metadata("scale_4x"), patient_id="TCGA-AA-0002"),
        )

    with pytest.raises(CoordinateValidationError, match=r"source_patch\*int"):
        replace(
            _metadata("scale_4x"),
            level_0_declared_footprint=(1024, 1025),
            level_0_step=(1024, 1025),
        )


def test_manifest_contains_exact_h5_hashes_and_strict_attributes(tmp_path: Path) -> None:
    destination, result = _publish(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {"schema", "branches"}
    for branch, filename in BRANCH_FILENAMES.items():
        entry = manifest["branches"][branch]
        assert entry["filename"] == filename
        assert len(entry["sha256"]) == 64
        assert len(entry["coordinates_sha256"]) == 64
        assert entry["coordinate_count"] == 4
        assert entry["attributes"]["policy_sha256"] == "4" * 64
        assert json.loads(entry["attributes"]["mask_parameters_json"])["sthresh"] == 8
