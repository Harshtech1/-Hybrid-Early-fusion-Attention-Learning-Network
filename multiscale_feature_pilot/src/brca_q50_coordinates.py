"""Deterministic, metadata-bound BRCA Q50 coordinate-policy core.

The module accepts an already-read level-2 mask array.  It performs no path,
OpenSlide, pixel-read, or artifact-write operation.  Tissue segmentation and
global level-0 lattice predicates deliberately reuse the reviewed generic
algorithm implementation currently housed in ``brca_q25_coordinates``; all
slide identity, pyramid geometry, physical scale, and output contracts below
are independently pinned for Q50.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Final

import numpy as np
from numpy.typing import NDArray

from multiscale_feature_pilot.src.brca_q25_coordinates import (
    Q25CoordinatePolicyError as _SharedCoordinatePolicyError,
    clam_int_cast_level_0_geometry as _shared_clam_int_cast_geometry,
    generate_level_0_lattice_coordinates as _shared_generate_lattice,
    segment_tissue_contours as _shared_segment_tissue_contours,
)


POLICY_STATUS: Final = "RESOLVED_BRCA_Q50_COORDINATE_POLICY_V1"
SHARED_ALGORITHM_SOURCE_SHA256: Final = (
    "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82"
)
CLAM_COMMIT: Final = "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"

EXPECTED_PATIENT_ID: Final = "TCGA-AR-A1AW"
EXPECTED_SLIDE_ID: Final = (
    "TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs"
)
EXPECTED_GDC_FILE_UUID: Final = "5c1216f3-19ec-4d3c-9bb0-9bd740b79f62"
EXPECTED_FILENAME: Final = EXPECTED_SLIDE_ID
EXPECTED_SIZE_BYTES: Final = 975_626_387
EXPECTED_MD5: Final = "304509e03f26cbecc9aee4ea691c8e5a"
EXPECTED_SHA256: Final = (
    "6b960db7e6e5772f4f626daa7b023ffbb5e3b20f9f6ff5c7d52f4554cf33399b"
)
EXPECTED_MPP: Final = (0.2468, 0.2468)
EXPECTED_LEVEL_DIMENSIONS: Final = (
    (99_960, 65_334),
    (24_990, 16_333),
    (6_247, 4_083),
    (3_123, 2_041),
)
EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES: Final = (
    1.0,
    4.000061225739301,
    16.001375061204985,
    32.009231974117526,
)

MASK_LEVEL: Final = 2
SCALE_2X_SOURCE_LEVEL: Final = 0
SCALE_2X_SOURCE_FOOTPRINT: Final = 512
SCALE_2X_STEP: Final = 512
SCALE_2X_OUTPUT_PATCH_SIZE: Final = 256
SCALE_2X_INTERPOLATION: Final = "PIL.Image.Resampling.LANCZOS"
SCALE_2X_EFFECTIVE_MPP: Final = (0.4936, 0.4936)
SCALE_4X_SOURCE_LEVEL: Final = 1
SCALE_4X_SOURCE_PATCH_SIZE: Final = 256
SCALE_4X_APPROVED_PHYSICAL_MPP: Final = (
    0.9872151105124595,
    0.9872151105124595,
)
GRID_ANCHOR: Final = (0, 0)
_FLOAT_ABS_TOLERANCE: Final = 1e-12


class Q50CoordinatePolicyError(ValueError):
    """Raised when evidence or geometry violates the Q50 policy."""


def _exact_float(actual: float, expected: float) -> bool:
    return math.isclose(
        float(actual), float(expected), rel_tol=0.0, abs_tol=_FLOAT_ABS_TOLERANCE
    )


@dataclass(frozen=True)
class Q50SlideObservation:
    """Independently verified Q50 identity, hash, and header evidence."""

    patient_id: str
    slide_id: str
    gdc_file_uuid: str
    filename: str
    size_bytes: int
    md5: str
    sha256: str
    mpp_x: float
    mpp_y: float
    level_dimensions: tuple[tuple[int, int], ...]
    openslide_level_downsamples: tuple[float, ...]

    def __post_init__(self) -> None:
        expected_strings = {
            "patient_id": EXPECTED_PATIENT_ID,
            "slide_id": EXPECTED_SLIDE_ID,
            "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
            "filename": EXPECTED_FILENAME,
            "md5": EXPECTED_MD5,
            "sha256": EXPECTED_SHA256,
        }
        for field_name, expected in expected_strings.items():
            if getattr(self, field_name) != expected:
                raise Q50CoordinatePolicyError(f"Q50 {field_name} drift")
        if self.size_bytes != EXPECTED_SIZE_BYTES:
            raise Q50CoordinatePolicyError("Q50 size_bytes drift")
        if not _exact_float(self.mpp_x, EXPECTED_MPP[0]):
            raise Q50CoordinatePolicyError("Q50 mpp_x drift")
        if not _exact_float(self.mpp_y, EXPECTED_MPP[1]):
            raise Q50CoordinatePolicyError("Q50 mpp_y drift")
        if tuple(self.level_dimensions) != EXPECTED_LEVEL_DIMENSIONS:
            raise Q50CoordinatePolicyError("Q50 level_dimensions drift")
        if len(self.openslide_level_downsamples) != len(
            EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES
        ) or not all(
            _exact_float(actual, expected)
            for actual, expected in zip(
                self.openslide_level_downsamples,
                EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
                strict=True,
            )
        ):
            raise Q50CoordinatePolicyError("Q50 level_downsamples drift")

    def coordinate_geometry_scale_xy(self, level: int) -> tuple[float, float]:
        """Return per-axis dimension ratios for geometry, never physical MPP."""

        if level < 0 or level >= len(self.level_dimensions):
            raise Q50CoordinatePolicyError(f"invalid Q50 level {level}")
        level_0_width, level_0_height = self.level_dimensions[0]
        level_width, level_height = self.level_dimensions[level]
        return level_0_width / level_width, level_0_height / level_height


def _positive_dimensions(value: object, *, label: str) -> tuple[int, int]:
    try:
        width, height = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise Q50CoordinatePolicyError(f"{label} must be a width/height pair") from exc
    if (
        isinstance(width, bool)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or not isinstance(height, int)
    ):
        raise Q50CoordinatePolicyError(f"{label} must contain integer dimensions")
    if width <= 0 or height <= 0:
        raise Q50CoordinatePolicyError(f"{label} dimensions must be positive")
    return width, height


def _require_polygon(polygon: object, *, label: str) -> None:
    if not isinstance(polygon, np.ndarray):
        raise Q50CoordinatePolicyError(f"{label} must be an ndarray")
    if polygon.dtype != np.int32:
        raise Q50CoordinatePolicyError(f"{label} must have dtype int32")
    if polygon.ndim == 3:
        valid_shape = polygon.shape[1:] == (1, 2)
        points = polygon.shape[0]
    elif polygon.ndim == 2:
        valid_shape = polygon.shape[1:] == (2,)
        points = polygon.shape[0]
    else:
        valid_shape = False
        points = 0
    if not valid_shape or points < 3:
        raise Q50CoordinatePolicyError(
            f"{label} must contain at least three int32 x/y points"
        )


@dataclass(frozen=True)
class Q50TissueGeometry:
    """Q50 contours and holes expressed in level-0 coordinates."""

    contours: tuple[NDArray[np.int32], ...]
    holes: tuple[tuple[NDArray[np.int32], ...], ...]
    level_0_dimensions: tuple[int, int]
    mask_dimensions: tuple[int, int]
    mask_downsample_xy: tuple[float, float]
    mask_level: int = MASK_LEVEL
    clam_commit: str = CLAM_COMMIT

    def __post_init__(self) -> None:
        if len(self.contours) != len(self.holes):
            raise Q50CoordinatePolicyError(
                "each foreground contour must have one hole collection"
            )
        _positive_dimensions(self.level_0_dimensions, label="level_0_dimensions")
        _positive_dimensions(self.mask_dimensions, label="mask_dimensions")
        if self.mask_level != MASK_LEVEL:
            raise Q50CoordinatePolicyError("Q50 tissue mask level drift")
        if self.clam_commit != CLAM_COMMIT:
            raise Q50CoordinatePolicyError("Q50 CLAM commit drift")
        if len(self.mask_downsample_xy) != 2 or not all(
            math.isfinite(float(item)) and float(item) > 0
            for item in self.mask_downsample_xy
        ):
            raise Q50CoordinatePolicyError(
                "mask_downsample_xy must contain finite positive x/y values"
            )
        for contour_index, (contour, holes) in enumerate(
            zip(self.contours, self.holes, strict=True)
        ):
            _require_polygon(contour, label=f"contours[{contour_index}]")
            if not isinstance(holes, tuple):
                raise Q50CoordinatePolicyError(
                    f"holes[{contour_index}] must be a tuple"
                )
            for hole_index, hole in enumerate(holes):
                _require_polygon(
                    hole, label=f"holes[{contour_index}][{hole_index}]"
                )


@dataclass(frozen=True)
class Q50CoordinateBags:
    """Two nonempty Q50 bags; every row is one level-0 ``(x,y)`` origin."""

    scale_2x: NDArray[np.int64]
    scale_4x: NDArray[np.int64]
    mask_downsample_xy: tuple[float, float]
    scale_4x_coordinate_geometry_scale_xy: tuple[float, float]
    contour_count: int
    retained_hole_count: int
    policy_status: str = POLICY_STATUS
    pixel_reads_authorized: bool = field(default=False, init=False)
    artifact_writes_authorized: bool = field(default=False, init=False)
    feature_extraction_authorized: bool = field(default=False, init=False)
    q75_authorized: bool = field(default=False, init=False)
    training_authorized: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _require_coordinate_contract(self.scale_2x, branch="scale_2x")
        _require_coordinate_contract(self.scale_4x, branch="scale_4x")
        if self.policy_status != POLICY_STATUS:
            raise Q50CoordinatePolicyError("Q50 coordinate policy status drift")
        if any(
            (
                self.pixel_reads_authorized,
                self.artifact_writes_authorized,
                self.feature_extraction_authorized,
                self.q75_authorized,
                self.training_authorized,
            )
        ):
            raise Q50CoordinatePolicyError("policy core cannot authorize execution")


def expected_q50_observation_from_verified_values(
    *,
    patient_id: str,
    slide_id: str,
    gdc_file_uuid: str,
    filename: str,
    size_bytes: int,
    md5: str,
    sha256: str,
    mpp_x: float,
    mpp_y: float,
    level_dimensions: tuple[tuple[int, int], ...],
    openslide_level_downsamples: tuple[float, ...],
) -> Q50SlideObservation:
    """Validate caller-supplied evidence and return the locked observation."""

    return Q50SlideObservation(
        patient_id=patient_id,
        slide_id=slide_id,
        gdc_file_uuid=gdc_file_uuid,
        filename=filename,
        size_bytes=size_bytes,
        md5=md5,
        sha256=sha256,
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        level_dimensions=level_dimensions,
        openslide_level_downsamples=openslide_level_downsamples,
    )


def segment_q50_tissue(
    mask_level_2_image: NDArray[np.generic],
    *,
    observation: Q50SlideObservation,
) -> Q50TissueGeometry:
    """Segment an already-supplied exact Q50 level-2 image."""

    try:
        shared = _shared_segment_tissue_contours(
            mask_level_2_image,
            level_0_dimensions=observation.level_dimensions[0],
            mask_dimensions=observation.level_dimensions[MASK_LEVEL],
        )
    except _SharedCoordinatePolicyError as exc:
        raise Q50CoordinatePolicyError(str(exc)) from exc
    return Q50TissueGeometry(
        contours=shared.contours,
        holes=shared.holes,
        level_0_dimensions=shared.level_0_dimensions,
        mask_dimensions=shared.mask_dimensions,
        mask_downsample_xy=shared.mask_downsample_xy,
        mask_level=shared.mask_level,
        clam_commit=shared.clam_commit,
    )


def _row_major(coordinates: NDArray[np.int64]) -> bool:
    rows = [tuple(map(int, row)) for row in coordinates.tolist()]
    return rows == sorted(rows, key=lambda point: (point[1], point[0]))


def _require_coordinate_contract(
    coordinates: NDArray[np.int64], *, branch: str
) -> None:
    if not isinstance(coordinates, np.ndarray):
        raise Q50CoordinatePolicyError(f"{branch} coordinates must be an ndarray")
    if coordinates.dtype != np.int64:
        raise Q50CoordinatePolicyError(f"{branch} coordinates must have dtype int64")
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise Q50CoordinatePolicyError(f"{branch} coordinates must have shape [N,2]")
    if coordinates.shape[0] == 0:
        raise Q50CoordinatePolicyError(f"{branch} coordinates must be non-empty")
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise Q50CoordinatePolicyError(f"{branch} coordinates contain duplicates")
    if not _row_major(coordinates):
        raise Q50CoordinatePolicyError(
            f"{branch} coordinates must be ordered row-major by (y,x)"
        )


def generate_q50_coordinate_bags(
    geometry: Q50TissueGeometry,
    *,
    observation: Q50SlideObservation,
) -> Q50CoordinateBags:
    """Generate both Q50 branches from one shared level-2 tissue geometry."""

    expected_mask_downsample = observation.coordinate_geometry_scale_xy(MASK_LEVEL)
    if geometry.level_0_dimensions != observation.level_dimensions[0]:
        raise Q50CoordinatePolicyError("Q50 geometry level-0 dimensions drift")
    if geometry.mask_dimensions != observation.level_dimensions[MASK_LEVEL]:
        raise Q50CoordinatePolicyError("Q50 geometry mask dimensions drift")
    if not all(
        _exact_float(actual, expected)
        for actual, expected in zip(
            geometry.mask_downsample_xy, expected_mask_downsample, strict=True
        )
    ):
        raise Q50CoordinatePolicyError("Q50 geometry mask downsample drift")

    scale_4x_coordinate_scale = observation.coordinate_geometry_scale_xy(
        SCALE_4X_SOURCE_LEVEL
    )
    try:
        scale_4x_footprint_xy = _shared_clam_int_cast_geometry(
            native_patch_size=SCALE_4X_SOURCE_PATCH_SIZE,
            coordinate_geometry_scale_xy=scale_4x_coordinate_scale,
        )
    except _SharedCoordinatePolicyError as exc:
        raise Q50CoordinatePolicyError(str(exc)) from exc
    if scale_4x_footprint_xy != (1024, 1024):
        raise Q50CoordinatePolicyError("Q50 scale_4x int-cast geometry drift")

    try:
        scale_2x = _shared_generate_lattice(
            level_0_dimensions=observation.level_dimensions[0],
            level_0_patch_size=SCALE_2X_SOURCE_FOOTPRINT,
            level_0_step=SCALE_2X_STEP,
            geometry=geometry,
        )
        scale_4x = _shared_generate_lattice(
            level_0_dimensions=observation.level_dimensions[0],
            level_0_patch_size=scale_4x_footprint_xy[0],
            level_0_step=scale_4x_footprint_xy[0],
            geometry=geometry,
        )
    except _SharedCoordinatePolicyError as exc:
        raise Q50CoordinatePolicyError(str(exc)) from exc
    return Q50CoordinateBags(
        scale_2x=scale_2x,
        scale_4x=scale_4x,
        mask_downsample_xy=expected_mask_downsample,
        scale_4x_coordinate_geometry_scale_xy=scale_4x_coordinate_scale,
        contour_count=len(geometry.contours),
        retained_hole_count=sum(len(items) for items in geometry.holes),
    )


def build_q50_coordinate_bags(
    mask_level_2_image: NDArray[np.generic],
    *,
    observation: Q50SlideObservation,
) -> Q50CoordinateBags:
    """Segment one supplied mask array, then build both coordinate bags."""

    geometry = segment_q50_tissue(mask_level_2_image, observation=observation)
    return generate_q50_coordinate_bags(geometry, observation=observation)
