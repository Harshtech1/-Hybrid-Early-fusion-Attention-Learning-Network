from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src import wsi_metadata_policy
from multiscale_feature_pilot.src.wsi_metadata_policy import (
    APPROVED_PER_AXIS_TOLERANCE_FRACTION,
    POLICY_STATUS,
    PROPOSED_PER_AXIS_TOLERANCE_FRACTION,
    WsiMetadataPolicyError,
    preflight_wsi_metadata,
    validate_metadata_pyramid,
)


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "brca_phase2_metadata_policy.yaml"
)


def _preflight(
    *,
    mpp_x: object = 0.25,
    mpp_y: object = 0.25,
    level_dimensions: object = ((4000, 2000), (2000, 1000), (1000, 500)),
    level_downsamples: object = (1.0, 2.0, 4.0),
):
    return preflight_wsi_metadata(
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
        tolerance_fraction=PROPOSED_PER_AXIS_TOLERANCE_FRACTION,
    )


def test_valid_exact_native_levels_are_selected_without_pixel_authorization() -> None:
    result = _preflight()

    assert result.policy_status == "APPROVED_NATIVE_LEVEL_METADATA_GATE_V1"
    assert result.real_wsi_execution_authorized is False
    assert result.silent_resampling_allowed is False
    assert [item.level_index for item in result.selections] == [1, 2]
    assert [item.target_mpp for item in result.selections] == [0.5, 1.0]
    assert all(item.resampling_performed is False for item in result.selections)
    assert result.selection_for("scale_2x").native_mpp_x == pytest.approx(0.5)
    assert result.selection_for("scale_4x").native_mpp_y == pytest.approx(1.0)


def test_native_mpp_and_per_axis_errors_are_computed_independently() -> None:
    result = _preflight(mpp_x="0.26", mpp_y="0.24")

    assert [
        (item.native_mpp_x, item.native_mpp_y) for item in result.native_levels
    ] == pytest.approx(
        [(0.26, 0.24), (0.52, 0.48), (1.04, 0.96)]
    )
    scale_2x = result.selection_for("scale_2x")
    scale_4x = result.selection_for("scale_4x")
    assert (scale_2x.relative_error_x, scale_2x.relative_error_y) == pytest.approx(
        (0.04, 0.04)
    )
    assert (scale_4x.relative_error_x, scale_4x.relative_error_y) == pytest.approx(
        (0.04, 0.04)
    )


def test_exact_ten_percent_boundary_is_inclusive_on_both_axes() -> None:
    result = _preflight(mpp_x=0.275, mpp_y=0.225)

    assert result.selection_for("scale_2x").relative_error_x == pytest.approx(0.10)
    assert result.selection_for("scale_2x").relative_error_y == pytest.approx(0.10)
    assert result.selection_for("scale_4x").relative_error_x == pytest.approx(0.10)
    assert result.selection_for("scale_4x").relative_error_y == pytest.approx(0.10)


def test_nearest_level_tie_uses_lower_native_level_index() -> None:
    result = _preflight(
        mpp_x=0.45,
        mpp_y=0.45,
        level_dimensions=((11000, 11000), (9000, 9000), (4950, 4950)),
        level_downsamples=(1.0, 11.0 / 9.0, 20.0 / 9.0),
    )

    assert result.selection_for("scale_2x").level_index == 0
    assert result.selection_for("scale_2x").relative_error_x == pytest.approx(0.10)
    assert result.selection_for("scale_4x").level_index == 2


def test_axis_disagreement_is_rejected_as_ambiguous() -> None:
    with pytest.raises(
        WsiMetadataPolicyError,
        match="ambiguous.*x selects level 1.*y selects level 0",
    ):
        _preflight(mpp_x=0.25, mpp_y=0.50)


def test_one_axis_outside_tolerance_rejects_the_slide_metadata() -> None:
    with pytest.raises(
        WsiMetadataPolicyError,
        match="outside the 10.0% per-axis tolerance",
    ):
        _preflight(mpp_x=0.25, mpp_y=0.30)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mpp_x", None),
        ("mpp_x", 0),
        ("mpp_x", -0.25),
        ("mpp_x", float("nan")),
        ("mpp_y", float("inf")),
        ("mpp_y", True),
        ("mpp_y", "not-a-number"),
        ("level_dimensions", None),
        ("level_dimensions", ()),
        ("level_downsamples", None),
        ("level_downsamples", ()),
    ],
)
def test_missing_or_nonpositive_nonfinite_metadata_fails_closed(
    field: str, value: object
) -> None:
    kwargs = {field: value}
    with pytest.raises(WsiMetadataPolicyError):
        _preflight(**kwargs)


@pytest.mark.parametrize(
    ("dimensions", "downsamples", "message"),
    [
        (((4000, 2000),), (1.0, 2.0), "same length"),
        (((4000, 2000), (2000, 1000)), (2.0, 4.0), "must equal 1.0"),
        (
            ((4000, 2000), (2000, 1000), (1000, 500)),
            (1.0, 4.0, 2.0),
            "strictly increasing",
        ),
        (
            ((4000, 2000), (2000, 1000), (2000, 1000)),
            (1.0, 2.0, 4.0),
            "duplicate native dimensions",
        ),
        (
            ((4000, 2000), (2000, 1000), (2100, 500)),
            (1.0, 2.0, 4.0),
            "must not grow",
        ),
        (
            ((4000, 2000), (1900, 1000), (1000, 500)),
            (1.0, 2.0, 4.0),
            "inconsistent with its reported downsample",
        ),
        (
            ((4000, 2000), (2000.0, 1000)),
            (1.0, 2.0),
            "positive integers",
        ),
    ],
)
def test_inconsistent_pyramids_fail_closed(
    dimensions: object, downsamples: object, message: str
) -> None:
    with pytest.raises(WsiMetadataPolicyError, match=message):
        validate_metadata_pyramid(
            mpp_x=0.25,
            mpp_y=0.25,
            level_dimensions=dimensions,
            level_downsamples=downsamples,
        )


def test_floor_or_ceiling_dimension_rounding_is_accepted() -> None:
    levels = validate_metadata_pyramid(
        mpp_x=0.25,
        mpp_y=0.25,
        level_dimensions=((1001, 503), (501, 251), (250, 126)),
        level_downsamples=(1.0, 2.0, 4.0),
    )

    assert [item.dimensions for item in levels] == [
        (1001, 503),
        (501, 251),
        (250, 126),
    ]


def test_tolerance_is_required_explicitly_and_locked_to_approved_policy() -> None:
    parameters = inspect.signature(preflight_wsi_metadata).parameters
    assert parameters["tolerance_fraction"].default is inspect.Parameter.empty

    with pytest.raises(WsiMetadataPolicyError, match="approved 10% policy"):
        preflight_wsi_metadata(
            mpp_x=0.25,
            mpp_y=0.25,
            level_dimensions=((4000, 2000), (2000, 1000), (1000, 500)),
            level_downsamples=(1.0, 2.0, 4.0),
            tolerance_fraction=0.20,
        )


def test_public_contract_has_no_file_or_slide_object_input() -> None:
    parameter_names = set(inspect.signature(preflight_wsi_metadata).parameters)
    assert parameter_names == {
        "mpp_x",
        "mpp_y",
        "level_dimensions",
        "level_downsamples",
        "tolerance_fraction",
    }
    assert "openslide" not in wsi_metadata_policy.__dict__


def test_yaml_records_approved_metadata_policy_but_prohibits_pixels() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))

    assert policy["status"] == POLICY_STATUS
    assert policy["target_policy"]["status"] == "APPROVED"
    assert policy["resampling_policy"]["status"] == (
        "APPROVED_NATIVE_ONLY_NO_RESAMPLING"
    )
    assert policy["target_policy"][
        "approved_per_axis_relative_tolerance_fraction"
    ] == pytest.approx(0.10)
    assert policy["target_policy"]["tie_break_rule"] == "lower_native_level_index"
    assert policy["resampling_policy"]["silent_resampling"] == "prohibited"
    assert policy["result_contract"]["real_slide_authorized_by_success"] is False
    assert policy["result_contract"]["execution_enabled"] is True
    assert policy["result_contract"]["pixel_read_enabled"] is False
    assert policy["authority"]["pixel_or_region_read"] == "NOT_AUTHORIZED"


def test_policy_constants_are_approved_and_non_pixel_authorizing() -> None:
    assert POLICY_STATUS == "APPROVED_NATIVE_LEVEL_METADATA_GATE_V1"
    assert PROPOSED_PER_AXIS_TOLERANCE_FRACTION == pytest.approx(0.10)
    assert APPROVED_PER_AXIS_TOLERANCE_FRACTION == pytest.approx(0.10)
    assert math.isfinite(PROPOSED_PER_AXIS_TOLERANCE_FRACTION)
