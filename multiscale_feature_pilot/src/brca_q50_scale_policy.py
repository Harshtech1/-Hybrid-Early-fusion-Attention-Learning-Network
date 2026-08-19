"""Pure fail-closed evaluator for the BRCA Q50 physical-scale mapping.

The public function accepts verified header values only.  It has no WSI path,
OpenSlide object, image input, or write API and cannot read or resample pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from multiscale_feature_pilot.src.wsi_metadata_policy import (
    NativeLevelMetadata,
    WsiMetadataPolicyError,
    validate_metadata_pyramid,
)


POLICY_STATUS = "RESOLVED_BRCA_Q50_SCALE_POLICY_V1"
MAX_RELATIVE_ERROR_PER_AXIS = 0.10
EXPECTED_MPP_X = 0.2468
EXPECTED_MPP_Y = 0.2468
EXPECTED_LEVEL_DIMENSIONS = (
    (99_960, 65_334),
    (24_990, 16_333),
    (6_247, 4_083),
    (3_123, 2_041),
)
EXPECTED_LEVEL_DOWNSAMPLES = (
    1.0,
    4.000061225739301,
    16.001375061204985,
    32.009231974117526,
)
SCALE_2X_TARGET_MPP = 0.5
SCALE_2X_SOURCE_LEVEL = 0
SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR = 2.0
SCALE_2X_SOURCE_FOOTPRINT = (512, 512)
SCALE_2X_OUTPUT_PATCH = (256, 256)
SCALE_2X_INTERPOLATION = "PIL.Image.Resampling.LANCZOS"
SCALE_4X_TARGET_MPP = 1.0
SCALE_4X_SOURCE_LEVEL = 1
SCALE_4X_SOURCE_FOOTPRINT = (256, 256)
SCALE_4X_OUTPUT_PATCH = (256, 256)
_EXACT_ABS_TOLERANCE = 1e-12


class Q50ScalePolicyError(ValueError):
    """Raised when metadata or mapping differs from the Q50 policy."""


@dataclass(frozen=True)
class Q50ScaleBranch:
    branch: str
    target_mpp: float
    source_level: int
    source_native_mpp_x: float
    source_native_mpp_y: float
    source_footprint: tuple[int, int]
    output_patch: tuple[int, int]
    operation: str
    linear_downsample_factor: float
    interpolation: str
    effective_mpp_x: float
    effective_mpp_y: float
    relative_error_x: float
    relative_error_y: float


@dataclass(frozen=True)
class Q50ScalePlan:
    policy_status: str
    native_levels: tuple[NativeLevelMetadata, ...]
    branches: tuple[Q50ScaleBranch, ...]
    silent_level_substitution_allowed: bool = field(default=False, init=False)
    silent_resampling_allowed: bool = field(default=False, init=False)
    pixel_execution_authorized: bool = field(default=False, init=False)
    coordinate_generation_authorized: bool = field(default=False, init=False)
    feature_extraction_authorized: bool = field(default=False, init=False)
    q75_authorized: bool = field(default=False, init=False)
    training_authorized: bool = field(default=False, init=False)

    def branch_for(self, branch: str) -> Q50ScaleBranch:
        matches = tuple(item for item in self.branches if item.branch == branch)
        if len(matches) != 1:
            raise Q50ScalePolicyError(
                f"scale plan does not contain exactly one {branch!r} branch"
            )
        return matches[0]


def _exact_float(actual: float, expected: float) -> bool:
    return math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=_EXACT_ABS_TOLERANCE
    )


def _relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / target


def _require_raw_pinned_metadata(
    *,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
) -> None:
    """Identify exact Q50 header drift before generic consistency checks.

    Structurally malformed values are deliberately left to the shared pyramid
    validator so callers still receive its precise type/range diagnostics.
    """

    for name, value, expected in (
        ("mpp_x", mpp_x, EXPECTED_MPP_X),
        ("mpp_y", mpp_y, EXPECTED_MPP_Y),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed) and not _exact_float(parsed, expected):
            raise Q50ScalePolicyError(f"Q50 {name} drift")

    try:
        dimensions = tuple(tuple(row) for row in level_dimensions)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        dimensions = ()
    if dimensions:
        if len(dimensions) != len(EXPECTED_LEVEL_DIMENSIONS):
            raise Q50ScalePolicyError("Q50 native level count drift")
        for level_index, (actual, expected) in enumerate(
            zip(dimensions, EXPECTED_LEVEL_DIMENSIONS, strict=True)
        ):
            if actual != expected:
                raise Q50ScalePolicyError(
                    f"Q50 level {level_index} dimensions drift"
                )

    try:
        downsamples = tuple(float(item) for item in level_downsamples)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        downsamples = ()
    if downsamples:
        if len(downsamples) != len(EXPECTED_LEVEL_DOWNSAMPLES):
            raise Q50ScalePolicyError("Q50 native level count drift")
        for level_index, (actual, expected) in enumerate(
            zip(downsamples, EXPECTED_LEVEL_DOWNSAMPLES, strict=True)
        ):
            if math.isfinite(actual) and not _exact_float(actual, expected):
                raise Q50ScalePolicyError(
                    f"Q50 level {level_index} downsample drift"
                )


def _require_pinned_metadata(levels: tuple[NativeLevelMetadata, ...]) -> None:
    if len(levels) != len(EXPECTED_LEVEL_DIMENSIONS):
        raise Q50ScalePolicyError("Q50 native level count drift")
    for level, dimensions, downsample in zip(
        levels,
        EXPECTED_LEVEL_DIMENSIONS,
        EXPECTED_LEVEL_DOWNSAMPLES,
        strict=True,
    ):
        if level.dimensions != dimensions:
            raise Q50ScalePolicyError(
                f"Q50 level {level.level_index} dimensions drift"
            )
        if not _exact_float(level.downsample, downsample):
            raise Q50ScalePolicyError(
                f"Q50 level {level.level_index} downsample drift"
            )
    if not _exact_float(levels[0].native_mpp_x, EXPECTED_MPP_X):
        raise Q50ScalePolicyError("Q50 mpp_x drift")
    if not _exact_float(levels[0].native_mpp_y, EXPECTED_MPP_Y):
        raise Q50ScalePolicyError("Q50 mpp_y drift")


def _require_tolerance(branch: Q50ScaleBranch) -> None:
    if (
        branch.relative_error_x > MAX_RELATIVE_ERROR_PER_AXIS
        or branch.relative_error_y > MAX_RELATIVE_ERROR_PER_AXIS
    ):
        raise Q50ScalePolicyError(
            f"{branch.branch} effective MPP exceeds the 10% tolerance"
        )


def evaluate_q50_scale_plan(
    *,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
) -> Q50ScalePlan:
    """Validate exact Q50 metadata and calculate its two fixed branches."""

    _require_raw_pinned_metadata(
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
    )
    try:
        levels = validate_metadata_pyramid(
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
        )
    except WsiMetadataPolicyError as exc:
        raise Q50ScalePolicyError(str(exc)) from exc
    _require_pinned_metadata(levels)

    level_0 = levels[SCALE_2X_SOURCE_LEVEL]
    effective_2x_x = level_0.native_mpp_x * SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR
    effective_2x_y = level_0.native_mpp_y * SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR
    scale_2x = Q50ScaleBranch(
        branch="scale_2x",
        target_mpp=SCALE_2X_TARGET_MPP,
        source_level=SCALE_2X_SOURCE_LEVEL,
        source_native_mpp_x=level_0.native_mpp_x,
        source_native_mpp_y=level_0.native_mpp_y,
        source_footprint=SCALE_2X_SOURCE_FOOTPRINT,
        output_patch=SCALE_2X_OUTPUT_PATCH,
        operation="EXPLICIT_LINEAR_DOWNSAMPLE",
        linear_downsample_factor=SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR,
        interpolation=SCALE_2X_INTERPOLATION,
        effective_mpp_x=effective_2x_x,
        effective_mpp_y=effective_2x_y,
        relative_error_x=_relative_error(effective_2x_x, SCALE_2X_TARGET_MPP),
        relative_error_y=_relative_error(effective_2x_y, SCALE_2X_TARGET_MPP),
    )

    level_1 = levels[SCALE_4X_SOURCE_LEVEL]
    scale_4x = Q50ScaleBranch(
        branch="scale_4x",
        target_mpp=SCALE_4X_TARGET_MPP,
        source_level=SCALE_4X_SOURCE_LEVEL,
        source_native_mpp_x=level_1.native_mpp_x,
        source_native_mpp_y=level_1.native_mpp_y,
        source_footprint=SCALE_4X_SOURCE_FOOTPRINT,
        output_patch=SCALE_4X_OUTPUT_PATCH,
        operation="NATIVE_LEVEL",
        linear_downsample_factor=1.0,
        interpolation="none",
        effective_mpp_x=level_1.native_mpp_x,
        effective_mpp_y=level_1.native_mpp_y,
        relative_error_x=_relative_error(level_1.native_mpp_x, SCALE_4X_TARGET_MPP),
        relative_error_y=_relative_error(level_1.native_mpp_y, SCALE_4X_TARGET_MPP),
    )
    _require_tolerance(scale_2x)
    _require_tolerance(scale_4x)
    return Q50ScalePlan(
        policy_status=POLICY_STATUS,
        native_levels=levels,
        branches=(scale_2x, scale_4x),
    )
