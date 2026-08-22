from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src import brca_q25_scale_policy
from multiscale_feature_pilot.src.brca_q25_scale_policy import (
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_LEVEL_DOWNSAMPLES,
    MAX_RELATIVE_ERROR_PER_AXIS,
    POLICY_STATUS,
    Q25ScalePolicyError,
    evaluate_approved_q25_scale_plan,
)


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "brca_q25_scale_policy.yaml"
)


def _evaluate(**overrides: object):
    inputs: dict[str, object] = {
        "mpp_x": 0.2525,
        "mpp_y": 0.2525,
        "level_dimensions": EXPECTED_LEVEL_DIMENSIONS,
        "level_downsamples": EXPECTED_LEVEL_DOWNSAMPLES,
    }
    inputs.update(overrides)
    return evaluate_approved_q25_scale_plan(**inputs)


def test_exact_q25_metadata_produces_the_approved_two_branch_plan() -> None:
    plan = _evaluate()

    assert plan.policy_status == "APPROVED_Q25_EXPLICIT_2X_SCALE_POLICY_V1"
    scale_2x = plan.branch_for("scale_2x")
    assert scale_2x.source_level == 0
    assert scale_2x.operation == "EXPLICIT_LINEAR_DOWNSAMPLE"
    assert scale_2x.linear_downsample_factor == pytest.approx(2.0)
    assert (scale_2x.effective_mpp_x, scale_2x.effective_mpp_y) == pytest.approx(
        (0.505, 0.505)
    )
    assert (scale_2x.relative_error_x, scale_2x.relative_error_y) == pytest.approx(
        (0.01, 0.01)
    )

    scale_4x = plan.branch_for("scale_4x")
    assert scale_4x.source_level == 1
    assert scale_4x.operation == "NATIVE_LEVEL"
    assert scale_4x.linear_downsample_factor == pytest.approx(1.0)
    assert (
        scale_4x.effective_mpp_x,
        scale_4x.effective_mpp_y,
    ) == pytest.approx((1.0100149842739303, 1.0100149842739303))
    assert scale_4x.relative_error_x == pytest.approx(0.010014984273930328)
    assert scale_4x.relative_error_y == pytest.approx(0.010014984273930328)


def test_plan_is_scale_approval_only_and_never_silently_substitutes() -> None:
    plan = _evaluate()

    assert plan.silent_level_substitution_allowed is False
    assert plan.silent_resampling_allowed is False
    assert plan.pixel_execution_authorized is False
    assert plan.coordinate_generation_authorized is False
    assert plan.feature_extraction_authorized is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mpp_x", 0.2526, "mpp_x drift"),
        ("mpp_y", 0.2524, "mpp_y drift"),
        (
            "level_dimensions",
            ((65736, 67406), (16435, 16851), (4108, 4212), (2054, 2106)),
            "level 1 dimensions",
        ),
        (
            "level_downsamples",
            (1.0, 4.0001, 16.002635628163056, 32.00527125632611),
            "level 1 downsample drift",
        ),
    ],
)
def test_metadata_drift_fails_closed(field: str, value: object, message: str) -> None:
    with pytest.raises(Q25ScalePolicyError, match=message):
        _evaluate(**{field: value})


def test_public_evaluator_cannot_accept_a_path_or_slide_object() -> None:
    assert set(inspect.signature(evaluate_approved_q25_scale_plan).parameters) == {
        "mpp_x",
        "mpp_y",
        "level_dimensions",
        "level_downsamples",
    }
    assert "openslide" not in brca_q25_scale_policy.__dict__
    assert "Image" not in brca_q25_scale_policy.__dict__


def test_yaml_records_exact_approval_and_keeps_execution_closed() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))

    assert policy["policy_status"] == POLICY_STATUS
    assert policy["authority"]["approved_scope"] == "Q25_SCALE_MAPPING_ONLY"
    assert policy["branches"]["scale_2x"]["source_level"] == 0
    assert policy["branches"]["scale_2x"][
        "linear_downsample_factor"
    ] == pytest.approx(2.0)
    assert policy["branches"]["scale_2x"]["effective_mpp_x"] == pytest.approx(
        0.505
    )
    assert policy["branches"]["scale_4x"]["source_level"] == 1
    assert policy["branches"]["scale_4x"]["effective_mpp_x"] == pytest.approx(
        1.0100149842739303
    )
    assert policy["acceptance"]["maximum_relative_error_per_axis"] == pytest.approx(
        MAX_RELATIVE_ERROR_PER_AXIS
    )
    assert policy["acceptance"]["silent_level_substitution"] == "prohibited"
    assert policy["acceptance"]["silent_resampling"] == "prohibited"
    assert policy["execution_boundary"]["real_pixel_execution_enabled"] is False
    assert policy["execution_boundary"]["q50_q75_status"] == "LOCKED"
