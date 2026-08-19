from __future__ import annotations

import inspect
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src import brca_q50_scale_policy as module
from multiscale_feature_pilot.src.brca_q50_scale_policy import (
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_LEVEL_DOWNSAMPLES,
    POLICY_STATUS,
    Q50ScalePolicyError,
    evaluate_q50_scale_plan,
)


POLICY_PATH = (
    Path(__file__).resolve().parents[1] / "config/brca_q50_scale_policy.yaml"
)


def _evaluate(**overrides: object):
    inputs: dict[str, object] = {
        "mpp_x": 0.2468,
        "mpp_y": 0.2468,
        "level_dimensions": EXPECTED_LEVEL_DIMENSIONS,
        "level_downsamples": EXPECTED_LEVEL_DOWNSAMPLES,
    }
    inputs.update(overrides)
    return evaluate_q50_scale_plan(**inputs)


def test_exact_q50_metadata_produces_fixed_two_branch_plan() -> None:
    plan = _evaluate()
    assert plan.policy_status == POLICY_STATUS
    scale_2x = plan.branch_for("scale_2x")
    assert scale_2x.source_level == 0
    assert scale_2x.source_footprint == (512, 512)
    assert scale_2x.output_patch == (256, 256)
    assert scale_2x.interpolation == "PIL.Image.Resampling.LANCZOS"
    assert (scale_2x.effective_mpp_x, scale_2x.effective_mpp_y) == pytest.approx(
        (0.4936, 0.4936)
    )
    assert (scale_2x.relative_error_x, scale_2x.relative_error_y) == pytest.approx(
        (0.0128, 0.0128)
    )

    scale_4x = plan.branch_for("scale_4x")
    assert scale_4x.source_level == 1
    assert scale_4x.operation == "NATIVE_LEVEL"
    assert scale_4x.source_footprint == (256, 256)
    assert scale_4x.output_patch == (256, 256)
    assert scale_4x.interpolation == "none"
    assert (scale_4x.effective_mpp_x, scale_4x.effective_mpp_y) == pytest.approx(
        (0.9872151105124595, 0.9872151105124595)
    )


def test_policy_core_never_authorizes_execution_or_q75() -> None:
    plan = _evaluate()
    assert plan.silent_level_substitution_allowed is False
    assert plan.silent_resampling_allowed is False
    assert plan.pixel_execution_authorized is False
    assert plan.coordinate_generation_authorized is False
    assert plan.feature_extraction_authorized is False
    assert plan.q75_authorized is False
    assert plan.training_authorized is False


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("mpp_x", 0.2469, "mpp_x drift"),
        ("mpp_y", 0.2467, "mpp_y drift"),
        (
            "level_dimensions",
            ((99_960, 65_334), (24_991, 16_333), (6_247, 4_083), (3_123, 2_041)),
            "level 1 dimensions drift",
        ),
        (
            "level_downsamples",
            (1.0, 4.1, 16.001375061204985, 32.009231974117526),
            "level 1 downsample drift",
        ),
    ],
)
def test_any_pinned_metadata_drift_fails_closed(
    field: str, value: object, message: str
) -> None:
    with pytest.raises(Q50ScalePolicyError, match=message):
        _evaluate(**{field: value})


def test_public_api_has_no_path_slide_or_image_input() -> None:
    assert set(inspect.signature(evaluate_q50_scale_plan).parameters) == {
        "mpp_x",
        "mpp_y",
        "level_dimensions",
        "level_downsamples",
    }
    source = inspect.getsource(module)
    assert "import openslide" not in source
    assert "read_region(" not in source
    assert "torch" not in source


def test_yaml_matches_executable_mapping_and_keeps_pixels_locked() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["policy_status"] == POLICY_STATUS
    assert policy["candidate_label"] == "Q50"
    assert policy["pinned_header_metadata"]["level_dimensions"] == [
        list(item) for item in EXPECTED_LEVEL_DIMENSIONS
    ]
    assert policy["branches"]["scale_2x"]["effective_mpp_x"] == pytest.approx(
        0.4936
    )
    assert policy["branches"]["scale_4x"]["effective_mpp_x"] == pytest.approx(
        0.9872151105124595
    )
    assert policy["execution_boundary"]["real_pixel_execution_enabled"] is False
    assert policy["authority"]["q75_processing"] == "NOT_AUTHORIZED"
    assert policy["authority"]["training"] == "NOT_AUTHORIZED"
