from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import yaml

from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
    SCHEMA,
    CoordinateBranchMetadata,
    publish_brca_coordinate_artifacts,
    validate_brca_coordinate_artifacts,
)
from multiscale_feature_pilot.src.brca_q50_coordinates import (
    CLAM_COMMIT,
    EXPECTED_FILENAME,
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_MD5,
    EXPECTED_PATIENT_ID,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SLIDE_ID,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "multiscale_feature_pilot/config/brca_q50_coordinate_policy.yaml"
ARTIFACT_SOURCE = ROOT / "multiscale_feature_pilot/src/brca_coordinate_artifacts.py"
MASK_PARAMETERS = {
    "color_space": "HSV_saturation",
    "sthresh": 8,
    "mthresh": 7,
    "close": 4,
    "use_otsu": False,
    "a_t": 100,
    "a_h": 16,
    "max_n_holes": 8,
    "reference_patch_size": 512,
    "contour_rule": "four_pt_easy_any_probe_on_or_inside",
    "hole_rule": "strict_center_inside_rejection",
}


def _metadata(branch: str) -> CoordinateBranchMetadata:
    shared = dict(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        wsi_filename=EXPECTED_FILENAME,
        wsi_size_bytes=EXPECTED_SIZE_BYTES,
        wsi_md5=EXPECTED_MD5,
        wsi_sha256=EXPECTED_SHA256,
        level_0_dimensions=EXPECTED_LEVEL_DIMENSIONS[0],
        mask_level=2,
        mask_level_dimensions=EXPECTED_LEVEL_DIMENSIONS[2],
        openslide_reported_mask_downsample=16.001375061204985,
        mask_image_channels=4,
        mask_image_sha256="5" * 64,
        mask_parameters=MASK_PARAMETERS,
        contour_count=1,
        retained_hole_count=0,
        clam_commit=CLAM_COMMIT,
        policy_sha256=hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest(),
    )
    if branch == "scale_2x":
        return CoordinateBranchMetadata(
            branch=branch,
            source_level=0,
            source_level_dimensions=EXPECTED_LEVEL_DIMENSIONS[0],
            openslide_reported_source_downsample=1.0,
            source_patch_size=(512, 512),
            output_patch_size=(256, 256),
            level_0_declared_footprint=(512, 512),
            level_0_step=(512, 512),
            target_mpp=0.5,
            effective_mpp=(0.4936, 0.4936),
            interpolation="PIL.Image.Resampling.LANCZOS",
            resampling="explicit_2x_spatial_downsample",
            geometry_compatibility="LEVEL0_IDENTITY_GEOMETRY",
            **shared,
        )
    return CoordinateBranchMetadata(
        branch=branch,
        source_level=1,
        source_level_dimensions=EXPECTED_LEVEL_DIMENSIONS[1],
        openslide_reported_source_downsample=4.000061225739301,
        source_patch_size=(256, 256),
        output_patch_size=(256, 256),
        level_0_declared_footprint=(1024, 1024),
        level_0_step=(1024, 1024),
        target_mpp=1.0,
        effective_mpp=(0.9872151105124595, 0.9872151105124595),
        interpolation="none",
        resampling="none",
        geometry_compatibility="CLAM_INT_CAST_GEOMETRY_COMPATIBLE",
        **shared,
    )


def test_q50_contract_publishes_and_revalidates_synthetic_artifacts(
    tmp_path: Path,
) -> None:
    scale_2x = np.asarray(
        [[0, 0], [512, 0], [0, 512], [512, 512]], dtype=np.int64
    )
    scale_4x = np.asarray(
        [[0, 0], [1024, 0], [0, 1024], [1024, 1024]], dtype=np.int64
    )
    destination = tmp_path / "Q50.coordinates.synthetic"
    published = publish_brca_coordinate_artifacts(
        destination,
        scale_2x_coordinates=scale_2x,
        scale_4x_coordinates=scale_4x,
        scale_2x_metadata=_metadata("scale_2x"),
        scale_4x_metadata=_metadata("scale_4x"),
    )
    validated = validate_brca_coordinate_artifacts(
        destination, expected_manifest_sha256=published.manifest_sha256
    )
    assert validated.manifest_sha256 == published.manifest_sha256
    assert validated.branch_for("scale_2x").coordinate_count == 4
    assert validated.branch_for("scale_4x").coordinate_count == 4
    assert validated.branch_for("scale_2x").metadata.patient_id == EXPECTED_PATIENT_ID
    assert validated.branch_for("scale_4x").metadata.wsi_sha256 == EXPECTED_SHA256


def test_q50_policy_pins_the_reviewed_generic_artifact_schema() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    artifact_policy = policy["coordinate_artifact_schema"]
    assert artifact_policy["schema"] == SCHEMA
    assert artifact_policy["sha256"] == hashlib.sha256(
        ARTIFACT_SOURCE.read_bytes()
    ).hexdigest()
    assert artifact_policy["current_validation_scope"] == "synthetic_only"
    assert artifact_policy["real_artifact_publication"] == (
        "NOT_AUTHORIZED_BY_THIS_POLICY"
    )
