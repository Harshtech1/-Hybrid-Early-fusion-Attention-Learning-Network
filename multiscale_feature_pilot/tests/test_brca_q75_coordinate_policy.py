from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src import brca_q75_coordinate_policy as module
from multiscale_feature_pilot.src.brca_q75_coordinate_policy import (
    EXPECTED_CNV_SHAPE,
    EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256,
    EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX,
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_HEADER_GATE_SOURCE_COMMIT,
    EXPECTED_HEADER_REPORT_SHA256,
    EXPECTED_HEADER_RESULT_COMMIT,
    EXPECTED_HEADER_RESULT_SHA256,
    EXPECTED_KNOWN_ISSUES_SHA256,
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_LEVEL_DOWNSAMPLES,
    EXPECTED_MD5,
    EXPECTED_MPP_X,
    EXPECTED_MPP_Y,
    EXPECTED_MUTATION_SHAPE,
    EXPECTED_PATIENT_ID,
    EXPECTED_Q25_COORDINATE_POLICY_SHA256,
    EXPECTED_Q25_COORDINATE_SOURCE_SHA256,
    EXPECTED_Q50_COORDINATE_POLICY_SHA256,
    EXPECTED_Q50_COORDINATE_SOURCE_SHA256,
    EXPECTED_RNA_SHAPE,
    EXPECTED_SCALE_APPROVAL_COMMIT,
    EXPECTED_SCALE_CONFIG_SHA256,
    EXPECTED_SCALE_PROVENANCE_SHA256,
    EXPECTED_SCALE_REPORT_SHA256,
    EXPECTED_SCALE_SOURCE_SHA256,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SLIDE_ID,
    EXPECTED_USER_STATEMENT,
    EXPECTED_USER_STATEMENT_SHA256,
    POLICY_STATUS,
    Q75CoordinateEvidence,
    Q75CoordinatePolicyError,
    review_q75_coordinate_policy,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "multiscale_feature_pilot/config/brca_q75_coordinate_policy.yaml"
SOURCE_PATH = ROOT / "multiscale_feature_pilot/src/brca_q75_coordinate_policy.py"
TEST_PATH = ROOT / "multiscale_feature_pilot/tests/test_brca_q75_coordinate_policy.py"
PROVENANCE_PATH = (
    ROOT
    / "multiscale_feature_pilot/provenance/brca_q75_coordinate_policy_review.yaml"
)
REPORT_PATH = ROOT / "reports/brca_q75_coordinate_policy_review.md"
HEADER_RESULT_PATH = (
    ROOT
    / "multiscale_feature_pilot/provenance/brca_q75_header_metadata_result/result.yaml"
)
HEADER_REPORT_PATH = (
    ROOT
    / "multiscale_feature_pilot/provenance/brca_q75_header_metadata_result/report.md"
)
SCALE_CONFIG_PATH = ROOT / "multiscale_feature_pilot/config/brca_q75_scale_policy.yaml"
SCALE_SOURCE_PATH = ROOT / "multiscale_feature_pilot/src/brca_q75_scale_policy.py"
SCALE_PROVENANCE_PATH = (
    ROOT / "multiscale_feature_pilot/provenance/brca_q75_scale_approval.yaml"
)
SCALE_REPORT_PATH = ROOT / "reports/brca_q75_scale_approval.md"
Q25_POLICY_PATH = ROOT / "multiscale_feature_pilot/config/brca_q25_coordinate_policy.yaml"
Q25_SOURCE_PATH = ROOT / "multiscale_feature_pilot/src/brca_q25_coordinates.py"
Q50_POLICY_PATH = ROOT / "multiscale_feature_pilot/config/brca_q50_coordinate_policy.yaml"
Q50_SOURCE_PATH = ROOT / "multiscale_feature_pilot/src/brca_q50_coordinates.py"
ARTIFACT_SCHEMA_PATH = ROOT / "multiscale_feature_pilot/src/brca_coordinate_artifacts.py"
KNOWN_ISSUES_PATH = ROOT / "shared/provenance/known_issues.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(**overrides: object) -> Q75CoordinateEvidence:
    values: dict[str, object] = {
        "user_statement": EXPECTED_USER_STATEMENT,
        "user_statement_sha256": EXPECTED_USER_STATEMENT_SHA256,
        "header_result_sha256": EXPECTED_HEADER_RESULT_SHA256,
        "header_report_sha256": EXPECTED_HEADER_REPORT_SHA256,
        "header_gate_source_commit": EXPECTED_HEADER_GATE_SOURCE_COMMIT,
        "header_result_commit": EXPECTED_HEADER_RESULT_COMMIT,
        "scale_approval_commit": EXPECTED_SCALE_APPROVAL_COMMIT,
        "scale_config_sha256": EXPECTED_SCALE_CONFIG_SHA256,
        "scale_source_sha256": EXPECTED_SCALE_SOURCE_SHA256,
        "scale_provenance_sha256": EXPECTED_SCALE_PROVENANCE_SHA256,
        "scale_report_sha256": EXPECTED_SCALE_REPORT_SHA256,
        "q25_coordinate_source_sha256": EXPECTED_Q25_COORDINATE_SOURCE_SHA256,
        "q25_coordinate_policy_sha256": EXPECTED_Q25_COORDINATE_POLICY_SHA256,
        "q50_coordinate_source_sha256": EXPECTED_Q50_COORDINATE_SOURCE_SHA256,
        "q50_coordinate_policy_sha256": EXPECTED_Q50_COORDINATE_POLICY_SHA256,
        "coordinate_artifact_schema_sha256": (
            EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256
        ),
        "known_issues_sha256": EXPECTED_KNOWN_ISSUES_SHA256,
        "patient_id": EXPECTED_PATIENT_ID,
        "slide_id": EXPECTED_SLIDE_ID,
        "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
        "size_bytes": EXPECTED_SIZE_BYTES,
        "md5": EXPECTED_MD5,
        "sha256": EXPECTED_SHA256,
        "exact_omic_source_row_index": EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX,
        "rna_shape": EXPECTED_RNA_SHAPE,
        "mutation_shape": EXPECTED_MUTATION_SHAPE,
        "cnv_shape": EXPECTED_CNV_SHAPE,
    }
    values.update(overrides)
    return Q75CoordinateEvidence(**values)  # type: ignore[arg-type]


def _review(**overrides: object):
    values: dict[str, object] = {
        "evidence": _evidence(),
        "mpp_x": EXPECTED_MPP_X,
        "mpp_y": EXPECTED_MPP_Y,
        "level_dimensions": EXPECTED_LEVEL_DIMENSIONS,
        "level_downsamples": EXPECTED_LEVEL_DOWNSAMPLES,
    }
    values.update(overrides)
    return review_q75_coordinate_policy(**values)


def test_exact_evidence_returns_scalar_only_reviewed_design() -> None:
    plan = _review()
    assert plan.policy_status == POLICY_STATUS
    assert plan.coordinate_policy_reviewed is True
    assert plan.design_only is True
    assert plan.real_coordinates_generated is False
    assert plan.level_dimensions == EXPECTED_LEVEL_DIMENSIONS
    assert plan.level_downsamples == EXPECTED_LEVEL_DOWNSAMPLES
    expected_scales = (
        (1.0, 1.0),
        (4.0, 4.000132643586682),
        (16.0, 16.001238061549344),
        (32.004718372161605, 32.00247612309869),
    )
    for actual, expected in zip(
        plan.coordinate_geometry_scale_xy, expected_scales, strict=True
    ):
        assert actual == pytest.approx(expected)


def test_mask_policy_pins_complete_reviewed_hsv_semantics() -> None:
    mask = _review().mask
    assert mask.level == 2
    assert mask.dimensions == (6783, 5654)
    assert mask.coordinate_geometry_scale_xy == pytest.approx(
        (16.0, 16.001238061549344)
    )
    assert mask.theoretical_pixel_count == 38_351_082
    assert mask.theoretical_rgba_uint8_bytes == 153_404_328
    assert mask.scaled_reference_area == 1023
    assert mask.area_threshold_multiplier == 100
    assert mask.tissue_net_area_threshold_mask_pixels == 102_300
    assert mask.hole_area_threshold_multiplier == 16
    assert mask.retained_hole_area_threshold_mask_pixels == 16_368
    assert mask.hole_ranking_and_limit == "sort_area_descending_then_take_first_8"
    assert mask.alpha_channel_policy == "ignore_alpha_use_first_three_RGB_channels"
    assert mask.saturation_filter.endswith("kernel_7")
    assert mask.binary_threshold == "fixed_saturation_threshold_8_no_otsu"
    assert mask.morphology == "MORPH_CLOSE_4x4_ones_kernel"
    assert mask.contour_retrieval == "RETR_CCOMP_direct_foreground_and_holes"
    assert mask.contour_approximation == "CHAIN_APPROX_NONE"
    assert mask.contour_scaling_to_level_0.endswith("cast_numpy_int32")


def test_scale_2x_geometry_and_theoretical_capacity_are_exact() -> None:
    branch = _review().branch_for("scale_2x")
    assert branch.source_level == 0
    assert branch.source_footprint == (512, 512)
    assert branch.output_patch == (256, 256)
    assert branch.effective_mpp == pytest.approx((0.4936, 0.4936))
    assert branch.level_0_declared_footprint == (512, 512)
    assert branch.level_0_step == (512, 512)
    assert branch.theoretical_full_slide_columns == 211
    assert branch.theoretical_full_slide_rows == 176
    assert branch.theoretical_full_slide_sites_before_tissue_filter == 37_136
    assert branch.last_complete_level_0_origin == (107_520, 89_600)
    assert branch.trailing_strip_right_bottom == (496, 359)
    assert branch.accepted_tissue_site_count_known is False


def test_scale_4x_geometry_and_theoretical_capacity_are_exact() -> None:
    branch = _review().branch_for("scale_4x")
    assert branch.source_level == 1
    assert branch.source_footprint == (256, 256)
    assert branch.output_patch == (256, 256)
    assert branch.effective_mpp == pytest.approx(
        (0.9872163682185965, 0.9872163682185965)
    )
    assert branch.coordinate_geometry_scale_xy == pytest.approx(
        (4.0, 4.000132643586682)
    )
    assert branch.level_0_declared_footprint == (1024, 1024)
    assert branch.level_0_step == (1024, 1024)
    assert branch.geometry_derivation == (
        "native_footprint*int(coordinate_geometry_scale_xy)"
    )
    assert branch.geometry_compatibility == "CLAM_INT_CAST_GEOMETRY_COMPATIBLE"
    assert branch.theoretical_full_slide_columns == 105
    assert branch.theoretical_full_slide_rows == 88
    assert branch.theoretical_full_slide_sites_before_tissue_filter == 9_240
    assert branch.last_complete_level_0_origin == (106_496, 89_088)
    assert branch.trailing_strip_right_bottom == (1008, 359)
    assert branch.accepted_tissue_site_count_known is False


def test_algorithm_and_future_artifact_contract_are_explicit() -> None:
    plan = _review()
    algorithm = plan.reviewed_algorithm
    assert algorithm.source_path == (
        "multiscale_feature_pilot/src/brca_q25_coordinates.py"
    )
    assert algorithm.source_sha256 == EXPECTED_Q25_COORDINATE_SOURCE_SHA256
    assert algorithm.clam_compatibility_commit == (
        "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"
    )
    assert algorithm.clam_attribution_status == (
        "historically_aligned_compatibility_pin_not_proven_author_runtime"
    )
    assert algorithm.numerical_q25_or_q50_slide_metadata_reused is False
    assert algorithm.executed_by_this_review is False

    contract = plan.coordinate_contract
    assert contract.branch_order == ("scale_2x", "scale_4x")
    assert contract.shape_each == "[N,2]"
    assert contract.dtype == "int64"
    assert contract.columns == ("x", "y")
    assert contract.contour_boundary_is_accepted is True
    assert contract.hole_boundary_is_rejected is False
    assert contract.artifact_schema_sha256 == (
        EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256
    )
    assert contract.artifacts_written_by_policy_core is False
    assert contract.overwrite_or_resume_allowed is False


def test_every_execution_surface_remains_locked() -> None:
    plan = _review()
    locks = {
        key: value
        for key, value in asdict(plan).items()
        if key.endswith("_authorized")
    }
    assert locks
    assert set(locks.values()) == {False}
    assert plan.real_coordinates_generated is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user_statement", EXPECTED_USER_STATEMENT + " ", "statement drift"),
        ("user_statement_sha256", "0" * 64, "statement SHA256 drift"),
        ("header_result_sha256", "0" * 64, "header result SHA256 drift"),
        ("header_report_sha256", "0" * 64, "header report SHA256 drift"),
        ("header_gate_source_commit", "0" * 40, "header-gate source commit drift"),
        ("header_result_commit", "0" * 40, "header-result commit drift"),
        ("scale_approval_commit", "0" * 40, "scale-approval commit drift"),
        ("scale_config_sha256", "0" * 64, "scale config SHA256 drift"),
        ("scale_source_sha256", "0" * 64, "scale source SHA256 drift"),
        ("scale_provenance_sha256", "0" * 64, "scale provenance SHA256 drift"),
        ("scale_report_sha256", "0" * 64, "scale report SHA256 drift"),
        ("q25_coordinate_source_sha256", "0" * 64, "Q25 coordinate source SHA256 drift"),
        ("q25_coordinate_policy_sha256", "0" * 64, "Q25 coordinate policy SHA256 drift"),
        ("q50_coordinate_source_sha256", "0" * 64, "Q50 coordinate source SHA256 drift"),
        ("q50_coordinate_policy_sha256", "0" * 64, "Q50 coordinate policy SHA256 drift"),
        ("coordinate_artifact_schema_sha256", "0" * 64, "artifact schema SHA256 drift"),
        ("known_issues_sha256", "0" * 64, "known-issues SHA256 drift"),
        ("patient_id", "TCGA-E2-WRONG", "patient ID drift"),
        ("slide_id", "wrong.svs", "slide ID drift"),
        ("gdc_file_uuid", "wrong", "GDC UUID drift"),
        ("size_bytes", EXPECTED_SIZE_BYTES + 1, "WSI size drift"),
        ("md5", "0" * 32, "WSI MD5 drift"),
        ("sha256", "0" * 64, "WSI SHA256 drift"),
        ("exact_omic_source_row_index", "772", "exact Omic source row drift"),
        ("rna_shape", (1, 1, 1557), "RNA shape drift"),
        ("mutation_shape", (1, 1, 22), "mutation shape drift"),
        ("cnv_shape", (1, 1, 1332), "CNV shape drift"),
    ],
)
def test_any_evidence_drift_fails_closed(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(Q75CoordinatePolicyError, match=message):
        _review(evidence=replace(_evidence(), **{field: value}))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mpp_x", 0.2469, "mpp_x drift"),
        ("mpp_y", 0.2467, "mpp_y drift"),
        (
            "level_dimensions",
            ((108_528, 90_471), (27_133, 22_617), (6_783, 5_654), (3_391, 2_827)),
            "level 1 dimensions drift",
        ),
        (
            "level_downsamples",
            (1.0, 4.1, 16.000619030774672, 32.00359724763015),
            "level 1 downsample drift",
        ),
    ],
)
def test_any_pinned_header_drift_fails_closed(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(Q75CoordinatePolicyError, match=message):
        _review(**{field: value})


def test_checked_in_predecessors_and_reviewed_dependencies_are_exact() -> None:
    expected = {
        HEADER_RESULT_PATH: EXPECTED_HEADER_RESULT_SHA256,
        HEADER_REPORT_PATH: EXPECTED_HEADER_REPORT_SHA256,
        SCALE_CONFIG_PATH: EXPECTED_SCALE_CONFIG_SHA256,
        SCALE_SOURCE_PATH: EXPECTED_SCALE_SOURCE_SHA256,
        SCALE_PROVENANCE_PATH: EXPECTED_SCALE_PROVENANCE_SHA256,
        SCALE_REPORT_PATH: EXPECTED_SCALE_REPORT_SHA256,
        Q25_POLICY_PATH: EXPECTED_Q25_COORDINATE_POLICY_SHA256,
        Q25_SOURCE_PATH: EXPECTED_Q25_COORDINATE_SOURCE_SHA256,
        Q50_POLICY_PATH: EXPECTED_Q50_COORDINATE_POLICY_SHA256,
        Q50_SOURCE_PATH: EXPECTED_Q50_COORDINATE_SOURCE_SHA256,
        ARTIFACT_SCHEMA_PATH: EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256,
        KNOWN_ISSUES_PATH: EXPECTED_KNOWN_ISSUES_SHA256,
    }
    assert {path: _sha256(path) for path in expected} == expected
    known_issues = KNOWN_ISSUES_PATH.read_text(encoding="utf-8")
    assert "exact CLAM commit used by the authors is not proven" in known_issues


def test_yaml_matches_review_and_keeps_actual_counts_unknown() -> None:
    policy = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert policy["policy_status"] == POLICY_STATUS
    statement = policy["authority"]["exact_user_statement"]
    assert statement == EXPECTED_USER_STATEMENT
    assert hashlib.sha256(statement.encode("utf-8")).hexdigest() == (
        policy["authority"]["exact_user_statement_sha256"]
    )
    assert policy["future_tissue_mask_policy"]["scaled_reference_area"] == 1023
    assert policy["future_tissue_mask_policy"][
        "tissue_net_area_threshold_mask_pixels_squared"
    ] == 102_300
    assert policy["future_tissue_mask_policy"][
        "hole_area_threshold_mask_pixels_squared"
    ] == 16_368
    assert policy["branches"]["scale_2x"][
        "accepted_tissue_site_count"
    ] == "unknown_until_separately_authorized_execution"
    assert policy["branches"]["scale_4x"][
        "accepted_tissue_site_count"
    ] == "unknown_until_separately_authorized_execution"
    assert policy["review_result"]["real_coordinates_generated"] is False
    boundary = policy["execution_boundary"]
    assert boundary["status"] == "EXECUTION_LOCKED"
    assert all(
        value is False
        for key, value in boundary.items()
        if key.endswith("_authorized")
    )


def test_public_api_has_no_path_image_array_or_write_surface() -> None:
    assert set(inspect.signature(review_q75_coordinate_policy).parameters) == {
        "evidence",
        "mpp_x",
        "mpp_y",
        "level_dimensions",
        "level_downsamples",
    }
    source = inspect.getsource(module)
    for prohibited in (
        "import openslide",
        "read_region(",
        "from pathlib",
        "import pathlib",
        "from PIL",
        "import PIL",
        "import cv2",
        "import numpy",
        "import torch",
        "import h5py",
        "requests",
        "subprocess",
        "urllib",
        "cuda",
        "open(",
        ".write(",
        ".save(",
    ):
        assert prohibited not in source
    plan_values = asdict(_review())
    assert "coordinates" not in plan_values


def test_provenance_and_report_record_only_the_locked_review() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["status"] == (
        "BRCA_Q75_COORDINATE_POLICY_REVIEWED_CPU_ONLY_EXECUTION_LOCKED"
    )
    assert provenance["authority"]["exact_user_statement"] == (
        EXPECTED_USER_STATEMENT
    )
    assert provenance["review_result"]["coordinates_generated"] == 0
    assert provenance["execution_boundary"]["status"] == "EXECUTION_LOCKED"
    assert set(provenance["operations_in_this_transition"].values()) == {0}
    implementation = provenance["implementation"]
    assert implementation["config"]["sha256"] == _sha256(CONFIG_PATH)
    assert implementation["pure_policy_core"]["sha256"] == _sha256(SOURCE_PATH)
    assert implementation["focused_tests"]["sha256"] == _sha256(TEST_PATH)

    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "BRCA_Q75_COORDINATE_POLICY_REVIEWED_CPU_ONLY_EXECUTION_LOCKED" in report
    assert "No coordinates were generated" in report
    assert "separate explicit authorization" in report
