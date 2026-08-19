"""Pure, fail-closed evaluator for the approved BRCA Q75 scale mapping.

This module formalizes the user's narrow approval of the scale mapping proposed
after the completed Q75 header-only gate.  It accepts recorded scalar metadata
and evidence identities only.  It has no WSI path, OpenSlide object, image
input, or write API and cannot read, resample, or otherwise access pixels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

from multiscale_feature_pilot.src.wsi_metadata_policy import (
    NativeLevelMetadata,
    WsiMetadataPolicyError,
    validate_metadata_pyramid,
)


POLICY_STATUS = "APPROVED_BRCA_Q75_SCALE_POLICY_V1_EXECUTION_LOCKED"
MAX_RELATIVE_ERROR_PER_AXIS = 0.10

EXPECTED_USER_STATEMENT = "OK CONTINUE PLEASE"
EXPECTED_USER_STATEMENT_SHA256 = (
    "70e658b86380c6b3c86fd0f980937ecb081e38d51ed3b7056cc534f9cb95e3e3"
)
EXPECTED_HEADER_RESULT_SHA256 = (
    "08a7ed3e67ddf17513ee2dbda2adfd2398333787aaa75fe9eacf911f3c1a3898"
)
EXPECTED_HEADER_REPORT_SHA256 = (
    "9ed50ecc8464109e0a8ca121082462f8765b1c70663499180cc644cfe604d985"
)
EXPECTED_HEADER_GATE_SOURCE_COMMIT = (
    "ef6c8921d993567c8178fc932f7315378714ca5a"
)
EXPECTED_HEADER_RESULT_COMMIT = "c7e98f4ce2663556be9b487441fe36494364ff18"

EXPECTED_PATIENT_ID = "TCGA-E2-A154"
EXPECTED_SLIDE_ID = (
    "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
)
EXPECTED_GDC_FILE_UUID = "25aec062-60d1-446e-a1c6-0c79cc74a770"
EXPECTED_SIZE_BYTES = 1_360_743_825
EXPECTED_MD5 = "a8c4b68fb6e0ab3e862efe3ed1fe10d7"
EXPECTED_SHA256 = (
    "844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37"
)

EXPECTED_MPP_X = 0.2468
EXPECTED_MPP_Y = 0.2468
EXPECTED_LEVEL_DIMENSIONS = (
    (108_528, 90_471),
    (27_132, 22_617),
    (6_783, 5_654),
    (3_391, 2_827),
)
EXPECTED_LEVEL_DOWNSAMPLES = (
    1.0,
    4.000066321793341,
    16.000619030774672,
    32.00359724763015,
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


class Q75ScalePolicyError(ValueError):
    """Raised when Q75 identity, evidence, or header metadata drifts."""


@dataclass(frozen=True)
class Q75ScaleEvidence:
    """Exact non-pixel evidence required by the scale-policy transition."""

    user_statement: str
    user_statement_sha256: str
    header_result_sha256: str
    header_report_sha256: str
    header_gate_source_commit: str
    header_result_commit: str
    patient_id: str
    slide_id: str
    gdc_file_uuid: str
    size_bytes: int
    md5: str
    sha256: str


@dataclass(frozen=True)
class Q75ScaleBranch:
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
class Q75ScalePlan:
    policy_status: str
    evidence: Q75ScaleEvidence
    native_levels: tuple[NativeLevelMetadata, ...]
    branches: tuple[Q75ScaleBranch, ...]
    scale_mapping_approved: bool = field(default=True, init=False)
    silent_level_substitution_allowed: bool = field(default=False, init=False)
    silent_resampling_allowed: bool = field(default=False, init=False)
    wsi_open_authorized: bool = field(default=False, init=False)
    pixel_or_region_access_authorized: bool = field(default=False, init=False)
    resampling_execution_authorized: bool = field(default=False, init=False)
    tissue_mask_generation_authorized: bool = field(default=False, init=False)
    coordinate_generation_authorized: bool = field(default=False, init=False)
    patch_extraction_authorized: bool = field(default=False, init=False)
    feature_extraction_authorized: bool = field(default=False, init=False)
    resnet50_inference_authorized: bool = field(default=False, init=False)
    healnet_execution_authorized: bool = field(default=False, init=False)
    gpu_work_authorized: bool = field(default=False, init=False)
    raw_wsi_deletion_authorized: bool = field(default=False, init=False)
    google_drive_operations_authorized: bool = field(default=False, init=False)
    full_cohort_processing_authorized: bool = field(default=False, init=False)
    training_authorized: bool = field(default=False, init=False)

    def branch_for(self, branch: str) -> Q75ScaleBranch:
        matches = tuple(item for item in self.branches if item.branch == branch)
        if len(matches) != 1:
            raise Q75ScalePolicyError(
                f"scale plan does not contain exactly one {branch!r} branch"
            )
        return matches[0]


def _exact_float(actual: float, expected: float) -> bool:
    return math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=_EXACT_ABS_TOLERANCE
    )


def _relative_error(actual: float, target: float) -> float:
    return abs(actual - target) / target


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q75ScalePolicyError(message)


def _require_exact_evidence(evidence: Q75ScaleEvidence) -> None:
    try:
        statement_digest = hashlib.sha256(
            evidence.user_statement.encode("utf-8")
        ).hexdigest()
    except (AttributeError, UnicodeError) as exc:
        raise Q75ScalePolicyError("Q75 user statement must be UTF-8 text") from exc

    expected = {
        "user statement": EXPECTED_USER_STATEMENT,
        "user statement SHA256": EXPECTED_USER_STATEMENT_SHA256,
        "header result SHA256": EXPECTED_HEADER_RESULT_SHA256,
        "header report SHA256": EXPECTED_HEADER_REPORT_SHA256,
        "header-gate source commit": EXPECTED_HEADER_GATE_SOURCE_COMMIT,
        "header-result commit": EXPECTED_HEADER_RESULT_COMMIT,
        "patient ID": EXPECTED_PATIENT_ID,
        "slide ID": EXPECTED_SLIDE_ID,
        "GDC UUID": EXPECTED_GDC_FILE_UUID,
        "WSI size": EXPECTED_SIZE_BYTES,
        "WSI MD5": EXPECTED_MD5,
        "WSI SHA256": EXPECTED_SHA256,
    }
    actual = {
        "user statement": evidence.user_statement,
        "user statement SHA256": evidence.user_statement_sha256,
        "header result SHA256": evidence.header_result_sha256,
        "header report SHA256": evidence.header_report_sha256,
        "header-gate source commit": evidence.header_gate_source_commit,
        "header-result commit": evidence.header_result_commit,
        "patient ID": evidence.patient_id,
        "slide ID": evidence.slide_id,
        "GDC UUID": evidence.gdc_file_uuid,
        "WSI size": evidence.size_bytes,
        "WSI MD5": evidence.md5,
        "WSI SHA256": evidence.sha256,
    }
    for label, expected_value in expected.items():
        _require(actual[label] == expected_value, f"Q75 {label} drift")
    _require(
        statement_digest == evidence.user_statement_sha256,
        "Q75 user statement content/hash mismatch",
    )


def _require_raw_pinned_metadata(
    *,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
) -> None:
    """Identify exact Q75 header drift before shared consistency checks."""

    for name, value, expected in (
        ("mpp_x", mpp_x, EXPECTED_MPP_X),
        ("mpp_y", mpp_y, EXPECTED_MPP_Y),
    ):
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            continue
        if math.isfinite(parsed) and not _exact_float(parsed, expected):
            raise Q75ScalePolicyError(f"Q75 {name} drift")

    try:
        dimensions = tuple(tuple(row) for row in level_dimensions)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        dimensions = ()
    if dimensions:
        if len(dimensions) != len(EXPECTED_LEVEL_DIMENSIONS):
            raise Q75ScalePolicyError("Q75 native level count drift")
        for level_index, (actual, expected) in enumerate(
            zip(dimensions, EXPECTED_LEVEL_DIMENSIONS, strict=True)
        ):
            if actual != expected:
                raise Q75ScalePolicyError(
                    f"Q75 level {level_index} dimensions drift"
                )

    try:
        downsamples = tuple(float(item) for item in level_downsamples)  # type: ignore[arg-type]
    except (TypeError, ValueError, OverflowError):
        downsamples = ()
    if downsamples:
        if len(downsamples) != len(EXPECTED_LEVEL_DOWNSAMPLES):
            raise Q75ScalePolicyError("Q75 native level count drift")
        for level_index, (actual, expected) in enumerate(
            zip(downsamples, EXPECTED_LEVEL_DOWNSAMPLES, strict=True)
        ):
            if math.isfinite(actual) and not _exact_float(actual, expected):
                raise Q75ScalePolicyError(
                    f"Q75 level {level_index} downsample drift"
                )


def _require_pinned_metadata(levels: tuple[NativeLevelMetadata, ...]) -> None:
    _require(
        len(levels) == len(EXPECTED_LEVEL_DIMENSIONS),
        "Q75 native level count drift",
    )
    for level, dimensions, downsample in zip(
        levels,
        EXPECTED_LEVEL_DIMENSIONS,
        EXPECTED_LEVEL_DOWNSAMPLES,
        strict=True,
    ):
        _require(
            level.dimensions == dimensions,
            f"Q75 level {level.level_index} dimensions drift",
        )
        _require(
            _exact_float(level.downsample, downsample),
            f"Q75 level {level.level_index} downsample drift",
        )
    _require(
        _exact_float(levels[0].native_mpp_x, EXPECTED_MPP_X),
        "Q75 mpp_x drift",
    )
    _require(
        _exact_float(levels[0].native_mpp_y, EXPECTED_MPP_Y),
        "Q75 mpp_y drift",
    )


def _require_tolerance(branch: Q75ScaleBranch) -> None:
    _require(
        branch.relative_error_x <= MAX_RELATIVE_ERROR_PER_AXIS
        and branch.relative_error_y <= MAX_RELATIVE_ERROR_PER_AXIS,
        f"{branch.branch} effective MPP exceeds the 10% tolerance",
    )


def evaluate_approved_q75_scale_plan(
    *,
    evidence: Q75ScaleEvidence,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
) -> Q75ScalePlan:
    """Validate exact Q75 evidence and return its approved, non-executable plan."""

    if not isinstance(evidence, Q75ScaleEvidence):
        raise Q75ScalePolicyError("Q75 evidence must be Q75ScaleEvidence")
    _require_exact_evidence(evidence)
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
        raise Q75ScalePolicyError(str(exc)) from exc
    _require_pinned_metadata(levels)

    level_0 = levels[SCALE_2X_SOURCE_LEVEL]
    effective_2x_x = level_0.native_mpp_x * SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR
    effective_2x_y = level_0.native_mpp_y * SCALE_2X_LINEAR_DOWNSAMPLE_FACTOR
    scale_2x = Q75ScaleBranch(
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
    scale_4x = Q75ScaleBranch(
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
    return Q75ScalePlan(
        policy_status=POLICY_STATUS,
        evidence=evidence,
        native_levels=levels,
        branches=(scale_2x, scale_4x),
    )
