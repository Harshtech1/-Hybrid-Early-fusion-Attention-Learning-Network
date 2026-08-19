from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src import brca_q75_scale_policy as module
from multiscale_feature_pilot.src.brca_q75_scale_policy import (
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_HEADER_GATE_SOURCE_COMMIT,
    EXPECTED_HEADER_REPORT_SHA256,
    EXPECTED_HEADER_RESULT_COMMIT,
    EXPECTED_HEADER_RESULT_SHA256,
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_LEVEL_DOWNSAMPLES,
    EXPECTED_MD5,
    EXPECTED_PATIENT_ID,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SLIDE_ID,
    EXPECTED_USER_STATEMENT,
    EXPECTED_USER_STATEMENT_SHA256,
    POLICY_STATUS,
    Q75ScaleEvidence,
    Q75ScalePolicyError,
    evaluate_approved_q75_scale_plan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/config/brca_q75_scale_policy.yaml"
)
HEADER_RESULT_PATH = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q75_header_metadata_result/result.yaml"
)
HEADER_REPORT_PATH = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q75_header_metadata_result/report.md"
)
PROVENANCE_PATH = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q75_scale_approval.yaml"
)
REPORT_PATH = REPOSITORY_ROOT / "reports/brca_q75_scale_approval.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(**overrides: object) -> Q75ScaleEvidence:
    values: dict[str, object] = {
        "user_statement": EXPECTED_USER_STATEMENT,
        "user_statement_sha256": EXPECTED_USER_STATEMENT_SHA256,
        "header_result_sha256": EXPECTED_HEADER_RESULT_SHA256,
        "header_report_sha256": EXPECTED_HEADER_REPORT_SHA256,
        "header_gate_source_commit": EXPECTED_HEADER_GATE_SOURCE_COMMIT,
        "header_result_commit": EXPECTED_HEADER_RESULT_COMMIT,
        "patient_id": EXPECTED_PATIENT_ID,
        "slide_id": EXPECTED_SLIDE_ID,
        "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
        "size_bytes": EXPECTED_SIZE_BYTES,
        "md5": EXPECTED_MD5,
        "sha256": EXPECTED_SHA256,
    }
    values.update(overrides)
    return Q75ScaleEvidence(**values)  # type: ignore[arg-type]


def _evaluate(**overrides: object):
    inputs: dict[str, object] = {
        "evidence": _evidence(),
        "mpp_x": 0.2468,
        "mpp_y": 0.2468,
        "level_dimensions": EXPECTED_LEVEL_DIMENSIONS,
        "level_downsamples": EXPECTED_LEVEL_DOWNSAMPLES,
    }
    inputs.update(overrides)
    return evaluate_approved_q75_scale_plan(**inputs)


def test_exact_evidence_and_metadata_produce_fixed_q75_mapping() -> None:
    plan = _evaluate()
    assert plan.policy_status == POLICY_STATUS
    assert plan.scale_mapping_approved is True

    scale_2x = plan.branch_for("scale_2x")
    assert scale_2x.source_level == 0
    assert scale_2x.source_footprint == (512, 512)
    assert scale_2x.output_patch == (256, 256)
    assert scale_2x.operation == "EXPLICIT_LINEAR_DOWNSAMPLE"
    assert scale_2x.interpolation == "PIL.Image.Resampling.LANCZOS"
    assert (scale_2x.effective_mpp_x, scale_2x.effective_mpp_y) == pytest.approx(
        (0.4936, 0.4936)
    )
    assert (scale_2x.relative_error_x, scale_2x.relative_error_y) == pytest.approx(
        (0.0128, 0.0128)
    )

    scale_4x = plan.branch_for("scale_4x")
    assert scale_4x.source_level == 1
    assert scale_4x.source_footprint == (256, 256)
    assert scale_4x.output_patch == (256, 256)
    assert scale_4x.operation == "NATIVE_LEVEL"
    assert scale_4x.interpolation == "none"
    assert (scale_4x.effective_mpp_x, scale_4x.effective_mpp_y) == pytest.approx(
        (0.9872163682185965, 0.9872163682185965)
    )
    assert (scale_4x.relative_error_x, scale_4x.relative_error_y) == pytest.approx(
        (0.012783631781403515, 0.012783631781403515)
    )


def test_mapping_approval_does_not_authorize_any_execution() -> None:
    plan = _evaluate()
    locked_flags = (
        plan.silent_level_substitution_allowed,
        plan.silent_resampling_allowed,
        plan.wsi_open_authorized,
        plan.pixel_or_region_access_authorized,
        plan.resampling_execution_authorized,
        plan.tissue_mask_generation_authorized,
        plan.coordinate_generation_authorized,
        plan.patch_extraction_authorized,
        plan.feature_extraction_authorized,
        plan.resnet50_inference_authorized,
        plan.healnet_execution_authorized,
        plan.gpu_work_authorized,
        plan.raw_wsi_deletion_authorized,
        plan.google_drive_operations_authorized,
        plan.full_cohort_processing_authorized,
        plan.training_authorized,
    )
    assert locked_flags == (False,) * len(locked_flags)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("user_statement", "OK CONTINUE PLEASE ", "user statement drift"),
        (
            "user_statement_sha256",
            "0" * 64,
            "user statement SHA256 drift",
        ),
        ("header_result_sha256", "0" * 64, "header result SHA256 drift"),
        ("header_report_sha256", "0" * 64, "header report SHA256 drift"),
        (
            "header_gate_source_commit",
            "0" * 40,
            "header-gate source commit drift",
        ),
        ("header_result_commit", "0" * 40, "header-result commit drift"),
        ("patient_id", "TCGA-E2-WRONG", "patient ID drift"),
        ("slide_id", "wrong.svs", "slide ID drift"),
        ("gdc_file_uuid", "wrong", "GDC UUID drift"),
        ("size_bytes", EXPECTED_SIZE_BYTES + 1, "WSI size drift"),
        ("md5", "0" * 32, "WSI MD5 drift"),
        ("sha256", "0" * 64, "WSI SHA256 drift"),
    ],
)
def test_any_evidence_drift_fails_closed(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(Q75ScalePolicyError, match=message):
        _evaluate(evidence=replace(_evidence(), **{field: value}))


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
    with pytest.raises(Q75ScalePolicyError, match=message):
        _evaluate(**{field: value})


def test_checked_in_header_result_and_report_are_exactly_bound() -> None:
    assert _sha256(HEADER_RESULT_PATH) == EXPECTED_HEADER_RESULT_SHA256
    assert _sha256(HEADER_REPORT_PATH) == EXPECTED_HEADER_REPORT_SHA256
    result = yaml.safe_load(HEADER_RESULT_PATH.read_text(encoding="utf-8"))
    assert result["status"] == "BRCA_Q75_FILE_AND_HEADER_METADATA_VERIFIED"
    assert result["source"]["source_commit"] == EXPECTED_HEADER_GATE_SOURCE_COMMIT
    assert result["required_stop_reached"] is True
    assert result["scale_policy_approved"] is False
    assert result["operations"]["pixel_or_region_reads"] == 0


def test_policy_yaml_matches_evaluator_and_keeps_execution_locked() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["policy_status"] == POLICY_STATUS
    assert policy["authority"]["exact_user_statement"] == EXPECTED_USER_STATEMENT
    assert hashlib.sha256(
        policy["authority"]["exact_user_statement"].encode("utf-8")
    ).hexdigest() == policy["authority"]["exact_user_statement_sha256"]
    assert (
        policy["header_result_binding"]["result_sha256"]
        == EXPECTED_HEADER_RESULT_SHA256
    )
    assert (
        policy["header_result_binding"]["report_sha256"]
        == EXPECTED_HEADER_REPORT_SHA256
    )
    assert (
        policy["header_result_binding"]["header_gate_source_commit"]
        == EXPECTED_HEADER_GATE_SOURCE_COMMIT
    )
    assert (
        policy["header_result_binding"]["header_result_commit"]
        == EXPECTED_HEADER_RESULT_COMMIT
    )
    assert policy["branches"]["scale_2x"]["effective_mpp_x"] == pytest.approx(
        0.4936
    )
    assert policy["branches"]["scale_4x"]["effective_mpp_x"] == pytest.approx(
        0.9872163682185965
    )
    boundary = policy["execution_boundary"]
    assert boundary["scale_mapping_resolved"] is True
    assert all(
        value is False
        for key, value in boundary.items()
        if key.endswith("_authorized")
    )


def test_public_api_is_metadata_only_and_source_has_no_execution_surface() -> None:
    assert set(inspect.signature(evaluate_approved_q75_scale_plan).parameters) == {
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
        "from PIL",
        "import torch",
        "subprocess",
        "requests",
        "urllib",
        "cuda",
    ):
        assert prohibited not in source


def test_provenance_and_report_record_only_the_narrow_locked_transition() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["status"] == (
        "BRCA_Q75_SCALE_POLICY_APPROVED_CPU_RECORDED_EXECUTION_LOCKED"
    )
    assert provenance["authority"]["exact_user_statement"] == (
        EXPECTED_USER_STATEMENT
    )
    assert provenance["header_result_evidence"]["result_sha256"] == (
        EXPECTED_HEADER_RESULT_SHA256
    )
    assert provenance["header_result_evidence"]["report_sha256"] == (
        EXPECTED_HEADER_REPORT_SHA256
    )
    assert provenance["execution_boundary"]["scale_mapping_approved"] is True
    assert provenance["execution_boundary"]["status"] == "EXECUTION_LOCKED"
    operations = provenance["operations_in_this_transition"]
    assert operations
    assert set(operations.values()) == {0}

    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "BRCA_Q75_SCALE_POLICY_APPROVED_CPU_RECORDED_EXECUTION_LOCKED" in report
    assert "22/22" in report
    assert "Pixel access and coordinate generation remain unauthorized" in report
