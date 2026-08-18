"""Fail-closed, metadata-only BRCA native-level preflight.

This module deliberately accepts values supplied by a caller.  It has no path
argument, does not open slides, and does not import OpenSlide.  A successful
result means only that synthetic or previously collected metadata satisfies
the proposed policy; it never authorizes WSI access or extraction.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from numbers import Integral


POLICY_STATUS = "PENDING_SUPERVISOR_APPROVAL"
PROPOSED_PER_AXIS_TOLERANCE_FRACTION = 0.10
TARGET_MPP_BY_BRANCH = (
    ("scale_2x", 0.5),
    ("scale_4x", 1.0),
)
SILENT_RESAMPLING_ALLOWED = False
REAL_WSI_EXECUTION_AUTHORIZED = False
_BOUNDARY_ABS_TOLERANCE = 1e-12


class WsiMetadataPolicyError(ValueError):
    """Raised when supplied WSI metadata cannot pass the proposed policy."""


@dataclass(frozen=True)
class NativeLevelMetadata:
    """Validated metadata and calculated physical scale for one native level."""

    level_index: int
    dimensions: tuple[int, int]
    downsample: float
    native_mpp_x: float
    native_mpp_y: float


@dataclass(frozen=True)
class NativeLevelSelection:
    """One target's unresampled native-level match under the proposal."""

    branch: str
    target_mpp: float
    level_index: int
    dimensions: tuple[int, int]
    downsample: float
    native_mpp_x: float
    native_mpp_y: float
    relative_error_x: float
    relative_error_y: float
    tolerance_fraction: float
    resampling_performed: bool = field(default=False, init=False)


@dataclass(frozen=True)
class WsiMetadataPreflight:
    """Validated proposal output that remains non-authorizing by construction."""

    policy_status: str
    proposed_tolerance_fraction: float
    level_0_mpp_x: float
    level_0_mpp_y: float
    native_levels: tuple[NativeLevelMetadata, ...]
    selections: tuple[NativeLevelSelection, ...]
    silent_resampling_allowed: bool = field(
        default=SILENT_RESAMPLING_ALLOWED,
        init=False,
    )
    real_wsi_execution_authorized: bool = field(
        default=REAL_WSI_EXECUTION_AUTHORIZED,
        init=False,
    )

    def selection_for(self, branch: str) -> NativeLevelSelection:
        """Return the selection for an exact branch label."""

        matches = tuple(item for item in self.selections if item.branch == branch)
        if len(matches) != 1:
            raise WsiMetadataPolicyError(
                f"preflight does not contain exactly one selection for {branch!r}"
            )
        return matches[0]


def _positive_finite_float(value: object, *, name: str) -> float:
    if value is None:
        raise WsiMetadataPolicyError(f"{name} is required")
    if isinstance(value, bool):
        raise WsiMetadataPolicyError(f"{name} must be numeric, not boolean")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise WsiMetadataPolicyError(
            f"{name} must be a finite positive number"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise WsiMetadataPolicyError(f"{name} must be finite and positive")
    return parsed


def _materialise_iterable(value: object, *, name: str) -> tuple[object, ...]:
    if value is None:
        raise WsiMetadataPolicyError(f"{name} is required")
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Iterable):
        raise WsiMetadataPolicyError(f"{name} must be a non-empty iterable")
    try:
        items = tuple(value)
    except (TypeError, ValueError) as exc:
        raise WsiMetadataPolicyError(f"{name} could not be read as metadata") from exc
    if not items:
        raise WsiMetadataPolicyError(f"{name} must not be empty")
    return items


def _normalise_dimensions(value: object) -> tuple[tuple[int, int], ...]:
    rows = _materialise_iterable(value, name="level_dimensions")
    dimensions: list[tuple[int, int]] = []
    for level_index, raw_row in enumerate(rows):
        if (
            isinstance(raw_row, (str, bytes, bytearray))
            or not isinstance(raw_row, Iterable)
        ):
            raise WsiMetadataPolicyError(
                f"level_dimensions[{level_index}] must contain width and height"
            )
        row = tuple(raw_row)
        if len(row) != 2:
            raise WsiMetadataPolicyError(
                f"level_dimensions[{level_index}] must contain exactly two values"
            )
        if any(
            isinstance(item, bool) or not isinstance(item, Integral) or item <= 0
            for item in row
        ):
            raise WsiMetadataPolicyError(
                f"level_dimensions[{level_index}] must contain positive integers"
            )
        dimensions.append((int(row[0]), int(row[1])))
    return tuple(dimensions)


def _normalise_downsamples(value: object) -> tuple[float, ...]:
    raw_values = _materialise_iterable(value, name="level_downsamples")
    return tuple(
        _positive_finite_float(item, name=f"level_downsamples[{level_index}]")
        for level_index, item in enumerate(raw_values)
    )


def _dimension_matches_downsample(
    *,
    level_0_dimension: int,
    level_dimension: int,
    downsample: float,
) -> bool:
    """Allow only the integer floor/ceiling implied by a scalar downsample."""

    try:
        expected = level_0_dimension / downsample
    except (OverflowError, ZeroDivisionError):
        return False
    if not math.isfinite(expected) or expected <= 0:
        return False
    lower = max(1, math.floor(expected))
    upper = max(1, math.ceil(expected))
    return level_dimension in {lower, upper}


def validate_metadata_pyramid(
    *,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
) -> tuple[NativeLevelMetadata, ...]:
    """Validate supplied pyramid metadata and calculate native per-axis MPP.

    Dimensions must be consistent with ``level_0_dimension / downsample`` up
    to unavoidable integer floor/ceiling rounding.  Native levels must be in
    increasing-downsample order and cannot repeat or grow in either axis.
    """

    base_mpp_x = _positive_finite_float(mpp_x, name="mpp_x")
    base_mpp_y = _positive_finite_float(mpp_y, name="mpp_y")
    dimensions = _normalise_dimensions(level_dimensions)
    downsamples = _normalise_downsamples(level_downsamples)

    if len(dimensions) != len(downsamples):
        raise WsiMetadataPolicyError(
            "level_dimensions and level_downsamples must have the same length"
        )
    if not math.isclose(
        downsamples[0], 1.0, rel_tol=0.0, abs_tol=_BOUNDARY_ABS_TOLERANCE
    ):
        raise WsiMetadataPolicyError("level_downsamples[0] must equal 1.0")
    if any(
        current <= previous
        for previous, current in zip(downsamples, downsamples[1:])
    ):
        raise WsiMetadataPolicyError(
            "level_downsamples must be strictly increasing by level index"
        )

    level_0_width, level_0_height = dimensions[0]
    native_levels: list[NativeLevelMetadata] = []
    for level_index, (level_size, downsample) in enumerate(
        zip(dimensions, downsamples, strict=True)
    ):
        width, height = level_size
        if level_index > 0:
            previous_width, previous_height = dimensions[level_index - 1]
            if width > previous_width or height > previous_height:
                raise WsiMetadataPolicyError(
                    "pyramid dimensions must not grow at a higher level index"
                )
            if (width, height) == (previous_width, previous_height):
                raise WsiMetadataPolicyError(
                    "pyramid contains ambiguous duplicate native dimensions"
                )

        if not _dimension_matches_downsample(
            level_0_dimension=level_0_width,
            level_dimension=width,
            downsample=downsample,
        ) or not _dimension_matches_downsample(
            level_0_dimension=level_0_height,
            level_dimension=height,
            downsample=downsample,
        ):
            raise WsiMetadataPolicyError(
                f"level {level_index} dimensions are inconsistent with its "
                "reported downsample"
            )

        native_mpp_x = base_mpp_x * downsample
        native_mpp_y = base_mpp_y * downsample
        if not math.isfinite(native_mpp_x) or not math.isfinite(native_mpp_y):
            raise WsiMetadataPolicyError(
                f"level {level_index} calculated native MPP is not finite"
            )
        native_levels.append(
            NativeLevelMetadata(
                level_index=level_index,
                dimensions=level_size,
                downsample=downsample,
                native_mpp_x=native_mpp_x,
                native_mpp_y=native_mpp_y,
            )
        )
    return tuple(native_levels)


def _relative_error(value: float, target: float) -> float:
    return abs(value - target) / target


def _nearest_level_index(errors: Sequence[float]) -> int:
    # The index is part of the sort key so numerically exact ties always choose
    # the lower native level index.
    return min(range(len(errors)), key=lambda index: (errors[index], index))


def _within_tolerance(error: float, tolerance: float) -> bool:
    return error <= tolerance or math.isclose(
        error,
        tolerance,
        rel_tol=0.0,
        abs_tol=_BOUNDARY_ABS_TOLERANCE,
    )


def _select_target(
    levels: Sequence[NativeLevelMetadata],
    *,
    branch: str,
    target_mpp: float,
    tolerance_fraction: float,
) -> NativeLevelSelection:
    errors_x = tuple(_relative_error(level.native_mpp_x, target_mpp) for level in levels)
    errors_y = tuple(_relative_error(level.native_mpp_y, target_mpp) for level in levels)
    nearest_x = _nearest_level_index(errors_x)
    nearest_y = _nearest_level_index(errors_y)
    if nearest_x != nearest_y:
        raise WsiMetadataPolicyError(
            f"{branch} target {target_mpp} MPP is ambiguous: x selects level "
            f"{nearest_x}, y selects level {nearest_y}"
        )

    level = levels[nearest_x]
    error_x = errors_x[nearest_x]
    error_y = errors_y[nearest_x]
    if not _within_tolerance(error_x, tolerance_fraction) or not _within_tolerance(
        error_y, tolerance_fraction
    ):
        raise WsiMetadataPolicyError(
            f"{branch} nearest native level {level.level_index} is outside the "
            f"{tolerance_fraction:.1%} per-axis tolerance: "
            f"x={error_x:.6%}, y={error_y:.6%}"
        )

    return NativeLevelSelection(
        branch=branch,
        target_mpp=target_mpp,
        level_index=level.level_index,
        dimensions=level.dimensions,
        downsample=level.downsample,
        native_mpp_x=level.native_mpp_x,
        native_mpp_y=level.native_mpp_y,
        relative_error_x=error_x,
        relative_error_y=error_y,
        tolerance_fraction=tolerance_fraction,
    )


def preflight_wsi_metadata(
    *,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
    tolerance_fraction: object,
) -> WsiMetadataPreflight:
    """Evaluate the two fixed native-MPP targets under the pending proposal.

    The tolerance is required explicitly and must equal the currently proposed
    10%.  Returning a result validates metadata only: real WSI access,
    extraction, and all resampling remain unauthorized.
    """

    tolerance = _positive_finite_float(
        tolerance_fraction, name="tolerance_fraction"
    )
    if not math.isclose(
        tolerance,
        PROPOSED_PER_AXIS_TOLERANCE_FRACTION,
        rel_tol=0.0,
        abs_tol=_BOUNDARY_ABS_TOLERANCE,
    ):
        raise WsiMetadataPolicyError(
            "tolerance_fraction must explicitly equal the pending 10% proposal"
        )

    levels = validate_metadata_pyramid(
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        level_dimensions=level_dimensions,
        level_downsamples=level_downsamples,
    )
    selections = tuple(
        _select_target(
            levels,
            branch=branch,
            target_mpp=target_mpp,
            tolerance_fraction=tolerance,
        )
        for branch, target_mpp in TARGET_MPP_BY_BRANCH
    )
    selected_indices = tuple(selection.level_index for selection in selections)
    if len(selected_indices) != len(set(selected_indices)):
        raise WsiMetadataPolicyError(
            "the two target branches do not have distinct native-level mappings"
        )

    return WsiMetadataPreflight(
        policy_status=POLICY_STATUS,
        proposed_tolerance_fraction=tolerance,
        level_0_mpp_x=levels[0].native_mpp_x,
        level_0_mpp_y=levels[0].native_mpp_y,
        native_levels=levels,
        selections=selections,
    )
