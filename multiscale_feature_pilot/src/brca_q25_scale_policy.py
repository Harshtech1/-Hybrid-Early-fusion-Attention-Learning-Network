"""Pure, fail-closed evaluator for the approved BRCA Q25 scale mapping.

The function in this module accepts already-collected header values only. It
has no WSI path or slide-object input, imports neither OpenSlide nor an image
library, and cannot read or resample pixels. Its result records the approved
mapping: explicit 2x spatial downsampling from native level 0 for the 0.5-MPP
branch and native level 1 for the 1.0-MPP branch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from multiscale_feature_pilot.src.wsi_metadata_policy import (
    NativeLevelMetadata,
    WsiMetadataPolicyError,
    validate_metadata_pyramid,
)


POLICY_STATUS = "APPROVED_Q25_EXPLICIT_2X_SCALE_POLICY_V1"
MAX_RELATIVE_ERROR_PER_AXIS = 0.10
EXPECTED_MPP_X = 0.2525
EXPECTED_MPP_Y = 0.2525
EXPECTED_LEVEL_DIMENSIONS = (
    (65736, 67406),
    (16434, 16851),
    (4108, 4212),
    (2054, 2106),
)
EXPECTED_LEVEL_DOWNSAMPLES = (
    1.0,
    4.00005934365913,
    16.002635628163056,
    32.00527125632611,
)
SCALE_2X_TARGET_MPP = 0.5
SCALE_2X_SOURCE_LEVEL = 0
SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR = 2.0
SCALE_4X_TARGET_MPP = 1.0
SCALE_4X_SOURCE_LEVEL = 1
_EXACT_ABS_TOLERANCE = 1e-12


class Q25ScalePolicyError(ValueError):
    """Raised when supplied metadata differs from the approved Q25 evidence."""


@dataclass(frozen=True)
class ApprovedScaleBranch:
    """One approved physical-scale mapping, without a pixel implementation."""

    branch: str
    target_mpp: float
    source_level: int
    source_native_mpp_x: float
    source_native_mpp_y: float
    operation: str
    linear_downsample_factor: float
    effective_mpp_x: float
    effective_mpp_y: float
    relative_error_x: float
    relative_error_y: float


@dataclass(frozen=True)
class Q25ScalePlan:
    """Approved Q25 scale plan that remains non-executable by construction."""

    policy_status: str
    native_levels: tuple[NativeLevelMetadata, ...]
    branches: tuple[ApprovedScaleBranch, ...]
    silent_level_substitution_allowed: bool = field(default=False, init=False)
    silent_resampling_allowed: bool = field(default=False, init=False)
    pixel_execution_authorized: bool = field(default=False, init=False)
    coordinate_generation_authorized: bool = field(default=False, init=False)
    feature_extraction_authorized: bool = field(default=False, init=False)

    def branch_for(self, branch: str) -> ApprovedScaleBranch:
        """Return exactly one branch by its fixed label."""

        matches = tuple(item for item in self.branches if item.branch == branch)
        if len(matches) != 1:
            raise Q25ScalePolicyError(
                f"scale plan does not contain exactly one {branch!r} branch"
            )
        return matches[0]


def _matches_exact_float(actual: float, expected: float) -> bool:
    return math.isclose(
        actual,
        expected,
        rel_tol=0.0,
        abs_tol=_EXACT_ABS_TOLERANCE,
    )


def _relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / target


def _require_approved_metadata(
    levels: tuple[NativeLevelMetadata, ...],
) -> None:
    if len(levels) != len(EXPECTED_LEVEL_DIMENSIONS):
        raise Q25ScalePolicyError("Q25 native level count drift")

    for level, expected_dimensions, expected_downsample in zip(
        levels,
        EXPECTED_LEVEL_DIMENSIONS,
        EXPECTED_LEVEL_DOWNSAMPLES,
        strict=True,
    ):
        if level.dimensions != expected_dimensions:
            raise Q25ScalePolicyError(
                f"Q25 level {level.level_index} dimensions drift"
            )
        if not _matches_exact_float(level.downsample, expected_downsample):
            raise Q25ScalePolicyError(
                f"Q25 level {level.level_index} downsample drift"
            )

    level_0 = levels[0]
    if not _matches_exact_float(level_0.native_mpp_x, EXPECTED_MPP_X):
        raise Q25ScalePolicyError("Q25 mpp_x drift")
    if not _matches_exact_float(level_0.native_mpp_y, EXPECTED_MPP_Y):
        raise Q25ScalePolicyError("Q25 mpp_y drift")


def _require_within_tolerance(branch: ApprovedScaleBranch) -> None:
    if (
        branch.relative_error_x > MAX_RELATIVE_ERROR_PER_AXIS
        or branch.relative_error_y > MAX_RELATIVE_ERROR_PER_AXIS
    ):
        raise Q25ScalePolicyError(
            f"{branch.branch} effective MPP is outside the approved 10% tolerance"
        )


def evaluate_approved_q25_scale_plan(
    *,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
) -> Q25ScalePlan:
    """Validate pinned Q25 metadata and calculate its approved scale mappings.

    Metadata validation is delegated to the earlier header-only pyramid
    evaluator. Any drift from the exact observed Q25 values rejects the plan;
    a nearby level is never substituted. No interpolation kernel, patch
    geometry, coordinate lattice, or pixel operation is selected here.
    """

    try:
        levels = validate_metadata_pyramid(
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
        )
    except WsiMetadataPolicyError as exc:
        raise Q25ScalePolicyError(str(exc)) from exc

    _require_approved_metadata(levels)

    scale_2x_level = levels[SCALE_2X_SOURCE_LEVEL]
    scale_2x_effective_x = (
        scale_2x_level.native_mpp_x * SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR
    )
    scale_2x_effective_y = (
        scale_2x_level.native_mpp_y * SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR
    )
    scale_2x = ApprovedScaleBranch(
        branch="scale_2x",
        target_mpp=SCALE_2X_TARGET_MPP,
        source_level=SCALE_2X_SOURCE_LEVEL,
        source_native_mpp_x=scale_2x_level.native_mpp_x,
        source_native_mpp_y=scale_2x_level.native_mpp_y,
        operation="EXPLICIT_LINEAR_DOWNSAMPLE",
        linear_downsample_factor=SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR,
        effective_mpp_x=scale_2x_effective_x,
        effective_mpp_y=scale_2x_effective_y,
        relative_error_x=_relative_error(
            scale_2x_effective_x, SCALE_2X_TARGET_MPP
        ),
        relative_error_y=_relative_error(
            scale_2x_effective_y, SCALE_2X_TARGET_MPP
        ),
    )

    scale_4x_level = levels[SCALE_4X_SOURCE_LEVEL]
    scale_4x = ApprovedScaleBranch(
        branch="scale_4x",
        target_mpp=SCALE_4X_TARGET_MPP,
        source_level=SCALE_4X_SOURCE_LEVEL,
        source_native_mpp_x=scale_4x_level.native_mpp_x,
        source_native_mpp_y=scale_4x_level.native_mpp_y,
        operation="NATIVE_LEVEL",
        linear_downsample_factor=1.0,
        effective_mpp_x=scale_4x_level.native_mpp_x,
        effective_mpp_y=scale_4x_level.native_mpp_y,
        relative_error_x=_relative_error(
            scale_4x_level.native_mpp_x, SCALE_4X_TARGET_MPP
        ),
        relative_error_y=_relative_error(
            scale_4x_level.native_mpp_y, SCALE_4X_TARGET_MPP
        ),
    )

    _require_within_tolerance(scale_2x)
    _require_within_tolerance(scale_4x)
    return Q25ScalePlan(
        policy_status=POLICY_STATUS,
        native_levels=levels,
        branches=(scale_2x, scale_4x),
    )
