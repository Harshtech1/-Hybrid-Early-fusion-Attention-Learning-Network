import pytest

from multiscale_feature_pilot.src.brca_singleton_streaming_policy import (
    PatientStage,
    PilotObservation,
    StreamingPolicyError,
    advance_stage,
    estimate_from_pilots,
    validate_static_plan,
    validate_storage_headroom,
)


PILOTS = (
    PilotObservation("Q25", 9322, 155.79171268499977, 153208853),
    PilotObservation("Q50", 10793, 171.42871093199983, 177402321),
    PilotObservation("Q75", 16945, 245.15734978399996, 278525034),
)


def test_frozen_static_plan_passes() -> None:
    validate_static_plan(
        patient_count=894,
        concurrency=1,
        initial_batch_size=1,
        raw_wsi_retention_count=1,
        quota_bytes=200_000_000_000,
        safety_floor_bytes=20_000_000_000,
        compact_retention_only=True,
    )


@pytest.mark.parametrize(
    "override",
    [
        {"patient_count": 895},
        {"concurrency": 2},
        {"initial_batch_size": 2},
        {"raw_wsi_retention_count": 2},
        {"quota_bytes": 201_000_000_000},
        {"safety_floor_bytes": 19_999_999_999},
        {"compact_retention_only": False},
    ],
)
def test_static_plan_fails_closed(override: dict[str, object]) -> None:
    plan = dict(
        patient_count=894,
        concurrency=1,
        initial_batch_size=1,
        raw_wsi_retention_count=1,
        quota_bytes=200_000_000_000,
        safety_floor_bytes=20_000_000_000,
        compact_retention_only=True,
    )
    plan.update(override)
    with pytest.raises(StreamingPolicyError):
        validate_static_plan(**plan)


def test_transition_requires_exact_next_stage_and_fresh_authority() -> None:
    assert (
        advance_stage(
            PatientStage.PLANNED,
            PatientStage.ACQUISITION_AUTHORIZED,
            separately_authorized=True,
            evidence_verified=True,
        )
        is PatientStage.ACQUISITION_AUTHORIZED
    )
    with pytest.raises(StreamingPolicyError, match="lacks separate authorization"):
        advance_stage(
            PatientStage.PLANNED,
            PatientStage.ACQUISITION_AUTHORIZED,
            separately_authorized=False,
            evidence_verified=True,
        )
    with pytest.raises(StreamingPolicyError, match="advance exactly"):
        advance_stage(
            PatientStage.PLANNED,
            PatientStage.RAW_VERIFIED,
            separately_authorized=True,
            evidence_verified=True,
        )


def test_terminal_stage_cannot_advance() -> None:
    with pytest.raises(StreamingPolicyError, match="terminal"):
        advance_stage(
            PatientStage.TERMINAL_RECORDED,
            PatientStage.PLANNED,
            separately_authorized=True,
            evidence_verified=True,
        )


def test_three_pilot_estimate_is_exact_and_labelled_as_estimate() -> None:
    result = estimate_from_pilots(PILOTS)
    assert result.patients == 894
    assert result.observed_patch_count_min == 9322
    assert result.observed_patch_count_mean == pytest.approx(12353.333333333334)
    assert result.observed_patch_count_max == 16945
    assert result.projected_gpu_hours_at_observed_min == pytest.approx(38.68827531677495)
    assert result.projected_gpu_hours_at_observed_mean == pytest.approx(47.380160131527184)
    assert result.projected_gpu_hours_at_observed_max == pytest.approx(60.88074186302666)
    assert result.projected_complete_artifact_bytes_at_observed_min == 136968714582
    assert result.projected_complete_artifact_bytes_at_observed_mean == 181522589984
    assert result.projected_complete_artifact_bytes_at_observed_max == 249001380396
    assert result.estimates_not_capacity_guarantees


def test_estimate_rejects_missing_or_reordered_pilot() -> None:
    with pytest.raises(StreamingPolicyError, match="ordered Q25"):
        estimate_from_pilots(tuple(reversed(PILOTS)))


def test_storage_headroom_exact_boundary_and_low_space_failure() -> None:
    required = validate_storage_headroom(
        available_bytes=25_000_000_000,
        raw_wsi_bytes=3_000_000_000,
        staging_bytes=1_000_000_000,
        final_artifact_bytes=1_000_000_000,
    )
    assert required == 25_000_000_000
    with pytest.raises(StreamingPolicyError, match="insufficient storage"):
        validate_storage_headroom(
            available_bytes=24_999_999_999,
            raw_wsi_bytes=3_000_000_000,
            staging_bytes=1_000_000_000,
            final_artifact_bytes=1_000_000_000,
        )
