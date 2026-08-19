"""Pure, non-executable BRCA Q75 coordinate-policy review.

This module accepts exact scalar evidence and returns a geometry design.  It
has no filesystem path, OpenSlide object, pixel/image input, coordinate-array
output, or artifact-write API.  In particular, reviewing this policy cannot
open the Q75 WSI, create a tissue mask, or generate coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math

from multiscale_feature_pilot.src.brca_q75_scale_policy import (
    EXPECTED_HEADER_GATE_SOURCE_COMMIT,
    EXPECTED_HEADER_REPORT_SHA256,
    EXPECTED_HEADER_RESULT_COMMIT,
    EXPECTED_HEADER_RESULT_SHA256,
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_LEVEL_DOWNSAMPLES,
    EXPECTED_MD5,
    EXPECTED_MPP_X,
    EXPECTED_MPP_Y,
    EXPECTED_PATIENT_ID,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SLIDE_ID,
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_USER_STATEMENT as EXPECTED_SCALE_USER_STATEMENT,
    EXPECTED_USER_STATEMENT_SHA256 as EXPECTED_SCALE_USER_STATEMENT_SHA256,
    Q75ScaleEvidence,
    Q75ScalePolicyError,
    evaluate_approved_q75_scale_plan,
)


POLICY_STATUS = "REVIEWED_BRCA_Q75_COORDINATE_POLICY_V1_EXECUTION_LOCKED"

EXPECTED_USER_STATEMENT = (
    "Continue with CPU-only Q75 coordinate-policy design and review. "
    "Do not open the WSI or generate coordinates yet."
)
EXPECTED_USER_STATEMENT_SHA256 = (
    "1dd1a3f8d74e6ecb276748ba0db193811e88b8fde2dae387961fb5719cfbd109"
)

EXPECTED_SCALE_APPROVAL_COMMIT = (
    "42db4dd4b402b3b7c32970f96db2f8d0c5f46180"
)
EXPECTED_SCALE_CONFIG_SHA256 = (
    "d29be0892e0b0324ae9b4390a1db9a9ae4b5a60b4541ddb7a36c81b8d2bca6b5"
)
EXPECTED_SCALE_SOURCE_SHA256 = (
    "3aecb1f3818f9ae98708cdf61f6ccf4b938ffe5fe78bbbaff6e11896e5eb4482"
)
EXPECTED_SCALE_PROVENANCE_SHA256 = (
    "aae6547c7c23cfdad51f62e3587b33c7abe3c5fb7d6fbdcd15ffeed3737fdd8e"
)
EXPECTED_SCALE_REPORT_SHA256 = (
    "6ae91ce127e426e079c68e752be39912e31528687dd897d7ce7f1594cd9fd017"
)

EXPECTED_Q25_COORDINATE_SOURCE_SHA256 = (
    "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82"
)
EXPECTED_Q25_COORDINATE_POLICY_SHA256 = (
    "85410751aec43b14997fa4c0e2a611ceb329178f788df04f336031104b697d43"
)
EXPECTED_Q50_COORDINATE_SOURCE_SHA256 = (
    "7dd739667cb6fe0887f3452127c9ff4d43659831d17096922a9f62685149f892"
)
EXPECTED_Q50_COORDINATE_POLICY_SHA256 = (
    "e5cb83739d3d8fab04da8a63ae1560df04ccc547a79512c219eecb575c0c2114"
)
EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256 = (
    "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf"
)
EXPECTED_KNOWN_ISSUES_SHA256 = (
    "8dff689f8181f7e08215595252042185542d9970c5885693b9afdaa7aa32c3c4"
)

EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX = "771"
EXPECTED_RNA_SHAPE = (1, 1, 1558)
EXPECTED_MUTATION_SHAPE = (1, 1, 21)
EXPECTED_CNV_SHAPE = (1, 1, 1333)

MASK_LEVEL = 2
CLAM_COMMIT = "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"
MASK_STHRESH = 8
MASK_MTHRESH = 7
MASK_CLOSE = 4
MASK_USE_OTSU = False
MASK_AREA_THRESHOLD = 100
MASK_HOLE_AREA_THRESHOLD = 16
MASK_MAX_HOLES = 8
MASK_REFERENCE_PATCH_SIZE = 512

GRID_ANCHOR = (0, 0)
SCALE_2X_LEVEL_0_FOOTPRINT = (512, 512)
SCALE_2X_LEVEL_0_STEP = (512, 512)
SCALE_4X_LEVEL_0_FOOTPRINT = (1024, 1024)
SCALE_4X_LEVEL_0_STEP = (1024, 1024)

_EXACT_ABS_TOLERANCE = 1e-12


class Q75CoordinatePolicyError(ValueError):
    """Raised when Q75 policy evidence or scalar geometry drifts."""


@dataclass(frozen=True)
class Q75CoordinateEvidence:
    """Exact immutable evidence required to review the Q75 policy."""

    user_statement: str
    user_statement_sha256: str
    header_result_sha256: str
    header_report_sha256: str
    header_gate_source_commit: str
    header_result_commit: str
    scale_approval_commit: str
    scale_config_sha256: str
    scale_source_sha256: str
    scale_provenance_sha256: str
    scale_report_sha256: str
    q25_coordinate_source_sha256: str
    q25_coordinate_policy_sha256: str
    q50_coordinate_source_sha256: str
    q50_coordinate_policy_sha256: str
    coordinate_artifact_schema_sha256: str
    known_issues_sha256: str
    patient_id: str
    slide_id: str
    gdc_file_uuid: str
    size_bytes: int
    md5: str
    sha256: str
    exact_omic_source_row_index: str
    rna_shape: tuple[int, int, int]
    mutation_shape: tuple[int, int, int]
    cnv_shape: tuple[int, int, int]


@dataclass(frozen=True)
class Q75MaskPolicy:
    """Future mask design expressed only as scalar metadata."""

    level: int
    dimensions: tuple[int, int]
    coordinate_geometry_scale_xy: tuple[float, float]
    theoretical_pixel_count: int
    theoretical_rgba_uint8_bytes: int
    color_space: str
    sthresh: int
    mthresh: int
    close: int
    use_otsu: bool
    area_threshold_multiplier: int
    hole_area_threshold_multiplier: int
    max_holes: int
    reference_patch_size: int
    scaled_reference_area: int
    tissue_net_area_threshold_mask_pixels: int
    tissue_net_area_comparison: str
    retained_hole_area_threshold_mask_pixels: int
    hole_area_comparison: str
    hole_ranking_and_limit: str
    alpha_channel_policy: str
    saturation_filter: str
    binary_threshold: str
    morphology: str
    contour_retrieval: str
    contour_approximation: str
    contour_scaling_to_level_0: str
    shared_by_branches: bool
    policy_kind: str


@dataclass(frozen=True)
class Q75ReviewedAlgorithm:
    """Exact reviewed compatibility dependency and its provenance caveat."""

    source_path: str
    source_sha256: str
    clam_compatibility_commit: str
    clam_attribution_status: str
    known_issue_path: str
    known_issue_sha256: str
    numerical_q25_or_q50_slide_metadata_reused: bool
    executed_by_this_review: bool


@dataclass(frozen=True)
class Q75CoordinateBranchPlan:
    """Scalar-only future geometry for one Q75 scale branch."""

    branch: str
    target_mpp: float
    effective_mpp: tuple[float, float]
    source_level: int
    source_footprint: tuple[int, int]
    output_patch: tuple[int, int]
    later_operation: str
    later_interpolation: str
    coordinate_geometry_scale_xy: tuple[float, float]
    level_0_lattice_anchor: tuple[int, int]
    level_0_declared_footprint: tuple[int, int]
    level_0_step: tuple[int, int]
    theoretical_full_slide_columns: int
    theoretical_full_slide_rows: int
    theoretical_full_slide_sites_before_tissue_filter: int
    last_complete_level_0_origin: tuple[int, int]
    trailing_strip_right_bottom: tuple[int, int]
    geometry_derivation: str
    geometry_compatibility: str
    accepted_tissue_site_count_known: bool = field(default=False, init=False)
    released_clam_coordinate_reproduction: bool = field(default=False, init=False)
    exact_paper_coordinate_reproduction: bool = field(default=False, init=False)


@dataclass(frozen=True)
class Q75CoordinateContract:
    """Contract for hypothetical artifacts in a separately authorized run."""

    branch_order: tuple[str, str]
    shape_each: str
    dtype: str
    columns: tuple[str, str]
    coordinate_space: str
    ordering: str
    require_nonempty: bool
    require_unique: bool
    reject_incomplete_level_0_footprints: bool
    artifacts_written_by_policy_core: bool
    contour_rule: str
    contour_boundary_is_accepted: bool
    hole_rule: str
    hole_boundary_is_rejected: bool
    artifact_schema_path: str
    artifact_schema_sha256: str
    artifact_schema: str
    future_publication: str
    overwrite_or_resume_allowed: bool


@dataclass(frozen=True)
class Q75CoordinatePolicyPlan:
    """Reviewed design whose entire execution surface remains locked."""

    policy_status: str
    evidence: Q75CoordinateEvidence
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    coordinate_geometry_scale_xy: tuple[tuple[float, float], ...]
    reviewed_algorithm: Q75ReviewedAlgorithm
    mask: Q75MaskPolicy
    branches: tuple[Q75CoordinateBranchPlan, ...]
    coordinate_contract: Q75CoordinateContract
    coordinate_policy_reviewed: bool = field(default=True, init=False)
    design_only: bool = field(default=True, init=False)
    real_coordinates_generated: bool = field(default=False, init=False)
    wsi_open_authorized: bool = field(default=False, init=False)
    pixel_or_region_access_authorized: bool = field(default=False, init=False)
    tissue_mask_generation_authorized: bool = field(default=False, init=False)
    coordinate_generation_authorized: bool = field(default=False, init=False)
    coordinate_artifact_publication_authorized: bool = field(
        default=False, init=False
    )
    patch_extraction_authorized: bool = field(default=False, init=False)
    resampling_execution_authorized: bool = field(default=False, init=False)
    resnet50_inference_authorized: bool = field(default=False, init=False)
    feature_generation_authorized: bool = field(default=False, init=False)
    healnet_execution_authorized: bool = field(default=False, init=False)
    gpu_work_authorized: bool = field(default=False, init=False)
    raw_wsi_deletion_authorized: bool = field(default=False, init=False)
    google_drive_operations_authorized: bool = field(default=False, init=False)
    full_cohort_processing_authorized: bool = field(default=False, init=False)
    q25_q50_modification_or_rerun_authorized: bool = field(
        default=False, init=False
    )
    blca_modification_authorized: bool = field(default=False, init=False)
    official_healnet_modification_authorized: bool = field(
        default=False, init=False
    )
    training_authorized: bool = field(default=False, init=False)

    def branch_for(self, branch: str) -> Q75CoordinateBranchPlan:
        matches = tuple(item for item in self.branches if item.branch == branch)
        if len(matches) != 1:
            raise Q75CoordinatePolicyError(
                f"coordinate plan does not contain exactly one {branch!r} branch"
            )
        return matches[0]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q75CoordinatePolicyError(message)


def _exact_float(actual: float, expected: float) -> bool:
    return math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=_EXACT_ABS_TOLERANCE
    )


def _require_exact_evidence(evidence: Q75CoordinateEvidence) -> None:
    try:
        statement_digest = hashlib.sha256(
            evidence.user_statement.encode("utf-8")
        ).hexdigest()
    except (AttributeError, UnicodeError) as exc:
        raise Q75CoordinatePolicyError(
            "Q75 coordinate-review statement must be UTF-8 text"
        ) from exc

    expected: dict[str, object] = {
        "coordinate-review statement": EXPECTED_USER_STATEMENT,
        "coordinate-review statement SHA256": EXPECTED_USER_STATEMENT_SHA256,
        "header result SHA256": EXPECTED_HEADER_RESULT_SHA256,
        "header report SHA256": EXPECTED_HEADER_REPORT_SHA256,
        "header-gate source commit": EXPECTED_HEADER_GATE_SOURCE_COMMIT,
        "header-result commit": EXPECTED_HEADER_RESULT_COMMIT,
        "scale-approval commit": EXPECTED_SCALE_APPROVAL_COMMIT,
        "scale config SHA256": EXPECTED_SCALE_CONFIG_SHA256,
        "scale source SHA256": EXPECTED_SCALE_SOURCE_SHA256,
        "scale provenance SHA256": EXPECTED_SCALE_PROVENANCE_SHA256,
        "scale report SHA256": EXPECTED_SCALE_REPORT_SHA256,
        "Q25 coordinate source SHA256": EXPECTED_Q25_COORDINATE_SOURCE_SHA256,
        "Q25 coordinate policy SHA256": EXPECTED_Q25_COORDINATE_POLICY_SHA256,
        "Q50 coordinate source SHA256": EXPECTED_Q50_COORDINATE_SOURCE_SHA256,
        "Q50 coordinate policy SHA256": EXPECTED_Q50_COORDINATE_POLICY_SHA256,
        "coordinate artifact schema SHA256": (
            EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256
        ),
        "known-issues SHA256": EXPECTED_KNOWN_ISSUES_SHA256,
        "patient ID": EXPECTED_PATIENT_ID,
        "slide ID": EXPECTED_SLIDE_ID,
        "GDC UUID": EXPECTED_GDC_FILE_UUID,
        "WSI size": EXPECTED_SIZE_BYTES,
        "WSI MD5": EXPECTED_MD5,
        "WSI SHA256": EXPECTED_SHA256,
        "exact Omic source row": EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX,
        "RNA shape": EXPECTED_RNA_SHAPE,
        "mutation shape": EXPECTED_MUTATION_SHAPE,
        "CNV shape": EXPECTED_CNV_SHAPE,
    }
    actual: dict[str, object] = {
        "coordinate-review statement": evidence.user_statement,
        "coordinate-review statement SHA256": evidence.user_statement_sha256,
        "header result SHA256": evidence.header_result_sha256,
        "header report SHA256": evidence.header_report_sha256,
        "header-gate source commit": evidence.header_gate_source_commit,
        "header-result commit": evidence.header_result_commit,
        "scale-approval commit": evidence.scale_approval_commit,
        "scale config SHA256": evidence.scale_config_sha256,
        "scale source SHA256": evidence.scale_source_sha256,
        "scale provenance SHA256": evidence.scale_provenance_sha256,
        "scale report SHA256": evidence.scale_report_sha256,
        "Q25 coordinate source SHA256": evidence.q25_coordinate_source_sha256,
        "Q25 coordinate policy SHA256": evidence.q25_coordinate_policy_sha256,
        "Q50 coordinate source SHA256": evidence.q50_coordinate_source_sha256,
        "Q50 coordinate policy SHA256": evidence.q50_coordinate_policy_sha256,
        "coordinate artifact schema SHA256": (
            evidence.coordinate_artifact_schema_sha256
        ),
        "known-issues SHA256": evidence.known_issues_sha256,
        "patient ID": evidence.patient_id,
        "slide ID": evidence.slide_id,
        "GDC UUID": evidence.gdc_file_uuid,
        "WSI size": evidence.size_bytes,
        "WSI MD5": evidence.md5,
        "WSI SHA256": evidence.sha256,
        "exact Omic source row": evidence.exact_omic_source_row_index,
        "RNA shape": evidence.rna_shape,
        "mutation shape": evidence.mutation_shape,
        "CNV shape": evidence.cnv_shape,
    }
    for label, expected_value in expected.items():
        _require(actual[label] == expected_value, f"Q75 {label} drift")
    _require(
        statement_digest == evidence.user_statement_sha256,
        "Q75 coordinate-review statement content/hash mismatch",
    )


def _coordinate_scale_xy(
    level_dimensions: tuple[tuple[int, int], ...], level: int
) -> tuple[float, float]:
    level_0_width, level_0_height = level_dimensions[0]
    level_width, level_height = level_dimensions[level]
    return level_0_width / level_width, level_0_height / level_height


def _full_slide_axis_sites(length: int, footprint: int, step: int) -> int:
    if length < footprint:
        return 0
    return ((length - footprint) // step) + 1


def _branch_plan(
    *,
    branch: str,
    target_mpp: float,
    effective_mpp: tuple[float, float],
    source_level: int,
    source_footprint: tuple[int, int],
    output_patch: tuple[int, int],
    later_operation: str,
    later_interpolation: str,
    coordinate_scale_xy: tuple[float, float],
    level_0_footprint: tuple[int, int],
    level_0_step: tuple[int, int],
    level_0_dimensions: tuple[int, int],
) -> Q75CoordinateBranchPlan:
    columns = _full_slide_axis_sites(
        level_0_dimensions[0], level_0_footprint[0], level_0_step[0]
    )
    rows = _full_slide_axis_sites(
        level_0_dimensions[1], level_0_footprint[1], level_0_step[1]
    )
    last_origin = (
        (columns - 1) * level_0_step[0],
        (rows - 1) * level_0_step[1],
    )
    trailing_strip = (
        level_0_dimensions[0] - last_origin[0] - level_0_footprint[0],
        level_0_dimensions[1] - last_origin[1] - level_0_footprint[1],
    )
    return Q75CoordinateBranchPlan(
        branch=branch,
        target_mpp=target_mpp,
        effective_mpp=effective_mpp,
        source_level=source_level,
        source_footprint=source_footprint,
        output_patch=output_patch,
        later_operation=later_operation,
        later_interpolation=later_interpolation,
        coordinate_geometry_scale_xy=coordinate_scale_xy,
        level_0_lattice_anchor=GRID_ANCHOR,
        level_0_declared_footprint=level_0_footprint,
        level_0_step=level_0_step,
        theoretical_full_slide_columns=columns,
        theoretical_full_slide_rows=rows,
        theoretical_full_slide_sites_before_tissue_filter=columns * rows,
        last_complete_level_0_origin=last_origin,
        trailing_strip_right_bottom=trailing_strip,
        geometry_derivation=(
            "native_footprint*int(coordinate_geometry_scale_xy)"
            if source_level == 1
            else "direct_level_0_footprint"
        ),
        geometry_compatibility=(
            "CLAM_INT_CAST_GEOMETRY_COMPATIBLE"
            if source_level == 1
            else "DIRECT_LEVEL_0_GEOMETRY"
        ),
    )


def review_q75_coordinate_policy(
    *,
    evidence: Q75CoordinateEvidence,
    mpp_x: object,
    mpp_y: object,
    level_dimensions: object,
    level_downsamples: object,
) -> Q75CoordinatePolicyPlan:
    """Return the exact Q75 scalar geometry design without coordinates."""

    if not isinstance(evidence, Q75CoordinateEvidence):
        raise Q75CoordinatePolicyError(
            "Q75 coordinate evidence must be Q75CoordinateEvidence"
        )
    _require_exact_evidence(evidence)

    scale_evidence = Q75ScaleEvidence(
        user_statement=EXPECTED_SCALE_USER_STATEMENT,
        user_statement_sha256=EXPECTED_SCALE_USER_STATEMENT_SHA256,
        header_result_sha256=evidence.header_result_sha256,
        header_report_sha256=evidence.header_report_sha256,
        header_gate_source_commit=evidence.header_gate_source_commit,
        header_result_commit=evidence.header_result_commit,
        patient_id=evidence.patient_id,
        slide_id=evidence.slide_id,
        gdc_file_uuid=evidence.gdc_file_uuid,
        size_bytes=evidence.size_bytes,
        md5=evidence.md5,
        sha256=evidence.sha256,
    )
    try:
        scale_plan = evaluate_approved_q75_scale_plan(
            evidence=scale_evidence,
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            level_dimensions=level_dimensions,
            level_downsamples=level_downsamples,
        )
    except Q75ScalePolicyError as exc:
        raise Q75CoordinatePolicyError(str(exc)) from exc

    dimensions = tuple(level.dimensions for level in scale_plan.native_levels)
    downsamples = tuple(level.downsample for level in scale_plan.native_levels)
    coordinate_scales = tuple(
        _coordinate_scale_xy(dimensions, level)
        for level in range(len(dimensions))
    )

    scale_2x_mapping = scale_plan.branch_for("scale_2x")
    scale_4x_mapping = scale_plan.branch_for("scale_4x")
    scale_4x_int_cast_footprint = tuple(
        size * int(scale)
        for size, scale in zip(
            scale_4x_mapping.source_footprint,
            coordinate_scales[scale_4x_mapping.source_level],
            strict=True,
        )
    )
    _require(
        scale_4x_int_cast_footprint == SCALE_4X_LEVEL_0_FOOTPRINT,
        "Q75 scale_4x CLAM integer-cast geometry drift",
    )

    mask_width, mask_height = dimensions[MASK_LEVEL]
    mask_scale_x, mask_scale_y = coordinate_scales[MASK_LEVEL]
    scaled_reference_area = int(
        MASK_REFERENCE_PATCH_SIZE**2 / (mask_scale_x * mask_scale_y)
    )
    mask = Q75MaskPolicy(
        level=MASK_LEVEL,
        dimensions=(mask_width, mask_height),
        coordinate_geometry_scale_xy=coordinate_scales[MASK_LEVEL],
        theoretical_pixel_count=mask_width * mask_height,
        theoretical_rgba_uint8_bytes=mask_width * mask_height * 4,
        color_space="HSV_saturation",
        sthresh=MASK_STHRESH,
        mthresh=MASK_MTHRESH,
        close=MASK_CLOSE,
        use_otsu=MASK_USE_OTSU,
        area_threshold_multiplier=MASK_AREA_THRESHOLD,
        hole_area_threshold_multiplier=MASK_HOLE_AREA_THRESHOLD,
        max_holes=MASK_MAX_HOLES,
        reference_patch_size=MASK_REFERENCE_PATCH_SIZE,
        scaled_reference_area=scaled_reference_area,
        tissue_net_area_threshold_mask_pixels=(
            MASK_AREA_THRESHOLD * scaled_reference_area
        ),
        tissue_net_area_comparison=(
            "contour_area_minus_all_direct_hole_areas > threshold"
        ),
        retained_hole_area_threshold_mask_pixels=(
            MASK_HOLE_AREA_THRESHOLD * scaled_reference_area
        ),
        hole_area_comparison="direct_hole_area > threshold",
        hole_ranking_and_limit="sort_area_descending_then_take_first_8",
        alpha_channel_policy="ignore_alpha_use_first_three_RGB_channels",
        saturation_filter="RGB_to_HSV_then_median_blur_saturation_kernel_7",
        binary_threshold="fixed_saturation_threshold_8_no_otsu",
        morphology="MORPH_CLOSE_4x4_ones_kernel",
        contour_retrieval="RETR_CCOMP_direct_foreground_and_holes",
        contour_approximation="CHAIN_APPROX_NONE",
        contour_scaling_to_level_0=(
            "multiply_x_y_independently_then_cast_numpy_int32"
        ),
        shared_by_branches=True,
        policy_kind="custom_engineering_mask_selection",
    )

    scale_2x = _branch_plan(
        branch="scale_2x",
        target_mpp=scale_2x_mapping.target_mpp,
        effective_mpp=(
            scale_2x_mapping.effective_mpp_x,
            scale_2x_mapping.effective_mpp_y,
        ),
        source_level=scale_2x_mapping.source_level,
        source_footprint=scale_2x_mapping.source_footprint,
        output_patch=scale_2x_mapping.output_patch,
        later_operation=scale_2x_mapping.operation,
        later_interpolation=scale_2x_mapping.interpolation,
        coordinate_scale_xy=coordinate_scales[scale_2x_mapping.source_level],
        level_0_footprint=SCALE_2X_LEVEL_0_FOOTPRINT,
        level_0_step=SCALE_2X_LEVEL_0_STEP,
        level_0_dimensions=dimensions[0],
    )
    scale_4x = _branch_plan(
        branch="scale_4x",
        target_mpp=scale_4x_mapping.target_mpp,
        effective_mpp=(
            scale_4x_mapping.effective_mpp_x,
            scale_4x_mapping.effective_mpp_y,
        ),
        source_level=scale_4x_mapping.source_level,
        source_footprint=scale_4x_mapping.source_footprint,
        output_patch=scale_4x_mapping.output_patch,
        later_operation=scale_4x_mapping.operation,
        later_interpolation=scale_4x_mapping.interpolation,
        coordinate_scale_xy=coordinate_scales[scale_4x_mapping.source_level],
        level_0_footprint=scale_4x_int_cast_footprint,
        level_0_step=SCALE_4X_LEVEL_0_STEP,
        level_0_dimensions=dimensions[0],
    )

    return Q75CoordinatePolicyPlan(
        policy_status=POLICY_STATUS,
        evidence=evidence,
        level_dimensions=dimensions,
        level_downsamples=downsamples,
        coordinate_geometry_scale_xy=coordinate_scales,
        reviewed_algorithm=Q75ReviewedAlgorithm(
            source_path="multiscale_feature_pilot/src/brca_q25_coordinates.py",
            source_sha256=evidence.q25_coordinate_source_sha256,
            clam_compatibility_commit=CLAM_COMMIT,
            clam_attribution_status=(
                "historically_aligned_compatibility_pin_not_proven_author_runtime"
            ),
            known_issue_path="shared/provenance/known_issues.md",
            known_issue_sha256=evidence.known_issues_sha256,
            numerical_q25_or_q50_slide_metadata_reused=False,
            executed_by_this_review=False,
        ),
        mask=mask,
        branches=(scale_2x, scale_4x),
        coordinate_contract=Q75CoordinateContract(
            branch_order=("scale_2x", "scale_4x"),
            shape_each="[N,2]",
            dtype="int64",
            columns=("x", "y"),
            coordinate_space="level_0",
            ordering="row_major_y_then_x",
            require_nonempty=True,
            require_unique=True,
            reject_incomplete_level_0_footprints=True,
            artifacts_written_by_policy_core=False,
            contour_rule="any_four_point_easy_probe_pointPolygonTest>=0",
            contour_boundary_is_accepted=True,
            hole_rule="reject_only_if_patch_center_pointPolygonTest>0",
            hole_boundary_is_rejected=False,
            artifact_schema_path=(
                "multiscale_feature_pilot/src/brca_coordinate_artifacts.py"
            ),
            artifact_schema_sha256=(
                evidence.coordinate_artifact_schema_sha256
            ),
            artifact_schema="BRCA_COORDINATE_ARTIFACT_SET_V1",
            future_publication="sibling_staging_then_linux_RENAME_NOREPLACE",
            overwrite_or_resume_allowed=False,
        ),
    )
