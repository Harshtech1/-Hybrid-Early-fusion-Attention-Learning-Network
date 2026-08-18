"""Deterministic, Q25-only BRCA tissue and coordinate geometry.

This module deliberately has no path API and performs no artifact writes.  A
caller must supply the already-read level-2 mask image together with the
independently verified Q25 identity and header metadata.  The implementation
then reproduces the pinned CLAM tissue predicates and constructs two custom
global level-0 lattices.

Pixel extraction is outside this module.  In particular, the scale-2x branch
records 512x512 level-0 source footprints that are to be resized to 256x256
with Lanczos later; this module neither reads those footprints nor resizes
them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray


POLICY_STATUS: Final = "APPROVED_BRCA_Q25_COORDINATE_POLICY_V1"
CLAM_COMMIT: Final = "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"

EXPECTED_PATIENT_ID: Final = "TCGA-LL-A6FP"
EXPECTED_SLIDE_ID: Final = "TCGA-LL-A6FP-01Z-00-DX1"
EXPECTED_GDC_FILE_UUID: Final = "dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6"
EXPECTED_FILENAME: Final = (
    "TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs"
)
EXPECTED_SIZE_BYTES: Final = 648_046_947
EXPECTED_MD5: Final = "75536393096ffd928bc35ec9503c3655"
EXPECTED_SHA256: Final = (
    "ac852be806eb0d91214145983319b604919a8e1d16bd59378a3dba69a600979c"
)
EXPECTED_MPP: Final = (0.2525, 0.2525)
EXPECTED_LEVEL_DIMENSIONS: Final = (
    (65_736, 67_406),
    (16_434, 16_851),
    (4_108, 4_212),
    (2_054, 2_106),
)
EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES: Final = (
    1.0,
    4.00005934365913,
    16.002635628163056,
    32.00527125632611,
)

MASK_LEVEL: Final = 2
S_THRESH: Final = 8
M_THRESH: Final = 7
CLOSE: Final = 4
USE_OTSU: Final = False
A_T: Final = 100
A_H: Final = 16
MAX_N_HOLES: Final = 8
REF_PATCH_SIZE: Final = 512

SCALE_2X_SOURCE_LEVEL: Final = 0
SCALE_2X_SOURCE_FOOTPRINT: Final = 512
SCALE_2X_STEP: Final = 512
SCALE_2X_OUTPUT_PATCH_SIZE: Final = 256
SCALE_2X_INTERPOLATION: Final = "PIL.Image.Resampling.LANCZOS"
SCALE_2X_EFFECTIVE_MPP: Final = (0.505, 0.505)

SCALE_4X_SOURCE_LEVEL: Final = 1
SCALE_4X_SOURCE_FOOTPRINT: Final = 256
SCALE_4X_GRID_LABEL: Final = "custom_global_level0_lattice_v1"
SCALE_4X_GEOMETRY_LABEL: Final = "CLAM_INT_CAST_GEOMETRY_COMPATIBLE"
SCALE_4X_APPROVED_PHYSICAL_MPP: Final = (
    1.0100149842739303,
    1.0100149842739303,
)

GRID_ANCHOR: Final = (0, 0)
_FLOAT_ABS_TOLERANCE: Final = 1e-12


class Q25CoordinatePolicyError(ValueError):
    """Raised when evidence or geometry violates the approved Q25 policy."""


def _exact_float(actual: float, expected: float) -> bool:
    return math.isclose(
        float(actual),
        float(expected),
        rel_tol=0.0,
        abs_tol=_FLOAT_ABS_TOLERANCE,
    )


def _positive_dimensions(value: object, *, label: str) -> tuple[int, int]:
    try:
        width, height = value  # type: ignore[misc]
    except (TypeError, ValueError) as exc:
        raise Q25CoordinatePolicyError(f"{label} must be a width/height pair") from exc
    if isinstance(width, bool) or isinstance(height, bool):
        raise Q25CoordinatePolicyError(f"{label} must contain integer dimensions")
    if not isinstance(width, int) or not isinstance(height, int):
        raise Q25CoordinatePolicyError(f"{label} must contain integer dimensions")
    if width <= 0 or height <= 0:
        raise Q25CoordinatePolicyError(f"{label} dimensions must be positive")
    return width, height


@dataclass(frozen=True)
class Q25SlideObservation:
    """Independently collected identity, hash, and header evidence for Q25."""

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
                raise Q25CoordinatePolicyError(f"Q25 {field_name} drift")
        if self.size_bytes != EXPECTED_SIZE_BYTES:
            raise Q25CoordinatePolicyError("Q25 size_bytes drift")
        if not _exact_float(self.mpp_x, EXPECTED_MPP[0]):
            raise Q25CoordinatePolicyError("Q25 mpp_x drift")
        if not _exact_float(self.mpp_y, EXPECTED_MPP[1]):
            raise Q25CoordinatePolicyError("Q25 mpp_y drift")

        if tuple(self.level_dimensions) != EXPECTED_LEVEL_DIMENSIONS:
            raise Q25CoordinatePolicyError("Q25 level_dimensions drift")
        if len(self.openslide_level_downsamples) != len(
            EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES
        ):
            raise Q25CoordinatePolicyError("Q25 level_downsamples drift")
        if not all(
            _exact_float(actual, expected)
            for actual, expected in zip(
                self.openslide_level_downsamples,
                EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
                strict=True,
            )
        ):
            raise Q25CoordinatePolicyError("Q25 level_downsamples drift")

    def coordinate_geometry_scale_xy(self, level: int) -> tuple[float, float]:
        """Return dimension-derived x/y coordinate scale, not physical MPP."""

        if level < 0 or level >= len(self.level_dimensions):
            raise Q25CoordinatePolicyError(f"invalid Q25 level {level}")
        level_0_width, level_0_height = self.level_dimensions[0]
        level_width, level_height = self.level_dimensions[level]
        return level_0_width / level_width, level_0_height / level_height


@dataclass(frozen=True)
class Q25TissueGeometry:
    """Pinned tissue contours and holes expressed in level-0 coordinates."""

    contours: tuple[NDArray[np.int32], ...]
    holes: tuple[tuple[NDArray[np.int32], ...], ...]
    level_0_dimensions: tuple[int, int]
    mask_dimensions: tuple[int, int]
    mask_downsample_xy: tuple[float, float]
    mask_level: int = MASK_LEVEL
    clam_commit: str = CLAM_COMMIT

    def __post_init__(self) -> None:
        if len(self.contours) != len(self.holes):
            raise Q25CoordinatePolicyError(
                "each foreground contour must have one hole collection"
            )
        _positive_dimensions(self.level_0_dimensions, label="level_0_dimensions")
        _positive_dimensions(self.mask_dimensions, label="mask_dimensions")
        if self.mask_level != MASK_LEVEL:
            raise Q25CoordinatePolicyError("Q25 tissue mask level drift")
        if self.clam_commit != CLAM_COMMIT:
            raise Q25CoordinatePolicyError("Q25 CLAM commit drift")
        if len(self.mask_downsample_xy) != 2 or not all(
            math.isfinite(float(item)) and float(item) > 0
            for item in self.mask_downsample_xy
        ):
            raise Q25CoordinatePolicyError(
                "mask_downsample_xy must contain finite positive x/y values"
            )


@dataclass(frozen=True)
class Q25CoordinateBags:
    """The two approved non-empty coordinate bags, both in level-0 x/y."""

    scale_2x: NDArray[np.int64]
    scale_4x: NDArray[np.int64]
    mask_downsample_xy: tuple[float, float]
    scale_4x_coordinate_geometry_scale_xy: tuple[float, float]
    contour_count: int
    retained_hole_count: int
    policy_status: str = POLICY_STATUS

    def __post_init__(self) -> None:
        _require_coordinate_contract(self.scale_2x, branch="scale_2x")
        _require_coordinate_contract(self.scale_4x, branch="scale_4x")
        if self.policy_status != POLICY_STATUS:
            raise Q25CoordinatePolicyError("Q25 coordinate policy status drift")


def expected_q25_observation_from_verified_values(
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
) -> Q25SlideObservation:
    """Validate caller-supplied evidence and return the locked observation."""

    return Q25SlideObservation(
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


def _approved_mask_image(
    image: NDArray[np.generic],
    *,
    expected_dimensions: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise Q25CoordinatePolicyError(
            f"mask image must have dtype uint8, got {array.dtype}"
        )
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise Q25CoordinatePolicyError(
            "mask image must have shape [height,width,3|4]"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise Q25CoordinatePolicyError("mask image dimensions must be positive")
    if expected_dimensions is not None:
        expected_width, expected_height = expected_dimensions
        if (array.shape[1], array.shape[0]) != (expected_width, expected_height):
            raise Q25CoordinatePolicyError("Q25 level-2 mask dimensions drift")
    return array


def segment_tissue_contours(
    mask_image: NDArray[np.generic],
    *,
    level_0_dimensions: tuple[int, int],
    mask_dimensions: tuple[int, int],
) -> Q25TissueGeometry:
    """Apply the pinned CLAM mask algorithm to an already-read RGB(A) image.

    The x and y scale factors are measured independently from level dimensions.
    OpenSlide's scalar downsample estimate is intentionally not used for
    contour scaling.
    """

    level_0_width, level_0_height = _positive_dimensions(
        level_0_dimensions,
        label="level_0_dimensions",
    )
    mask_width, mask_height = _positive_dimensions(
        mask_dimensions,
        label="mask_dimensions",
    )
    image = _approved_mask_image(
        mask_image,
        expected_dimensions=(mask_width, mask_height),
    )
    mask_downsample_xy = (
        level_0_width / mask_width,
        level_0_height / mask_height,
    )

    # The alpha channel is not part of CLAM's HSV saturation mask.
    rgb = np.ascontiguousarray(image[:, :, :3])
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = cv2.medianBlur(hsv[:, :, 1], M_THRESH)
    threshold_type = cv2.THRESH_BINARY | (
        cv2.THRESH_OTSU if USE_OTSU else 0
    )
    _, binary = cv2.threshold(saturation, S_THRESH, 255, threshold_type)
    kernel = np.ones((CLOSE, CLOSE), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    found = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None or not contours:
        return Q25TissueGeometry(
            contours=(),
            holes=(),
            level_0_dimensions=(level_0_width, level_0_height),
            mask_dimensions=(mask_width, mask_height),
            mask_downsample_xy=mask_downsample_xy,
        )

    hierarchy_rows = np.squeeze(hierarchy, axis=0)
    foreground_ids = np.flatnonzero(hierarchy_rows[:, 3] == -1)
    scaled_reference_area = int(
        REF_PATCH_SIZE**2
        / (mask_downsample_xy[0] * mask_downsample_xy[1])
    )
    tissue_area_threshold = A_T * scaled_reference_area
    hole_area_threshold = A_H * scaled_reference_area
    axis_scale = np.asarray(mask_downsample_xy, dtype=np.float64)

    retained_contours: list[NDArray[np.int32]] = []
    retained_holes: list[tuple[NDArray[np.int32], ...]] = []
    for contour_id in foreground_ids:
        contour = contours[int(contour_id)]
        hole_ids = np.flatnonzero(hierarchy_rows[:, 3] == contour_id)
        direct_holes = [contours[int(hole_id)] for hole_id in hole_ids]
        foreground_area = cv2.contourArea(contour) - sum(
            cv2.contourArea(hole) for hole in direct_holes
        )
        if foreground_area == 0 or foreground_area <= tissue_area_threshold:
            continue

        largest_holes = sorted(
            direct_holes,
            key=cv2.contourArea,
            reverse=True,
        )[:MAX_N_HOLES]
        filtered_holes = tuple(
            np.asarray(hole * axis_scale, dtype=np.int32)
            for hole in largest_holes
            if cv2.contourArea(hole) > hole_area_threshold
        )
        retained_contours.append(
            np.asarray(contour * axis_scale, dtype=np.int32)
        )
        retained_holes.append(filtered_holes)

    return Q25TissueGeometry(
        contours=tuple(retained_contours),
        holes=tuple(retained_holes),
        level_0_dimensions=(level_0_width, level_0_height),
        mask_dimensions=(mask_width, mask_height),
        mask_downsample_xy=mask_downsample_xy,
    )


def segment_q25_tissue(
    mask_level_2_image: NDArray[np.generic],
    *,
    observation: Q25SlideObservation,
) -> Q25TissueGeometry:
    """Segment the exact Q25 level-2 image after identity/header validation."""

    return segment_tissue_contours(
        mask_level_2_image,
        level_0_dimensions=observation.level_dimensions[0],
        mask_dimensions=observation.level_dimensions[MASK_LEVEL],
    )


def _four_pt_easy_and_center(
    origin: tuple[int, int],
    *,
    patch_size: int,
) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    """Return CLAM's four probes and center for a level-0 square footprint."""

    if patch_size <= 0:
        raise Q25CoordinatePolicyError("patch_size must be positive")
    half = patch_size // 2
    shift = int(half * 0.5)
    center = (origin[0] + half, origin[1] + half)
    probes = (
        (center[0] - shift, center[1] - shift),
        (center[0] + shift, center[1] + shift),
        (center[0] + shift, center[1] - shift),
        (center[0] - shift, center[1] + shift),
    )
    return probes, center


def level_0_patch_is_tissue(
    level_0_origin: tuple[int, int],
    geometry: Q25TissueGeometry,
    *,
    level_0_patch_size: int,
) -> bool:
    """Apply pinned four-point-easy and strict center-in-hole exclusion."""

    probes, center = _four_pt_easy_and_center(
        level_0_origin,
        patch_size=level_0_patch_size,
    )
    for contour, holes in zip(geometry.contours, geometry.holes, strict=True):
        if not any(
            cv2.pointPolygonTest(
                contour,
                (float(probe[0]), float(probe[1])),
                False,
            )
            >= 0
            for probe in probes
        ):
            continue
        # This strict comparison is CLAM's center-hole behavior: a center on
        # the boundary is retained, while a center inside a hole is rejected.
        if not any(
            cv2.pointPolygonTest(
                hole,
                (float(center[0]), float(center[1])),
                False,
            )
            > 0
            for hole in holes
        ):
            return True
    return False


def _row_major(coordinates: NDArray[np.int64]) -> bool:
    rows = [tuple(map(int, row)) for row in coordinates.tolist()]
    return rows == sorted(rows, key=lambda point: (point[1], point[0]))


def _require_coordinate_contract(
    coordinates: NDArray[np.int64],
    *,
    branch: str,
) -> None:
    if not isinstance(coordinates, np.ndarray):
        raise Q25CoordinatePolicyError(f"{branch} coordinates must be an ndarray")
    if coordinates.dtype != np.int64:
        raise Q25CoordinatePolicyError(f"{branch} coordinates must have dtype int64")
    if coordinates.ndim != 2 or coordinates.shape[1:] != (2,):
        raise Q25CoordinatePolicyError(f"{branch} coordinates must have shape [N,2]")
    if coordinates.shape[0] == 0:
        raise Q25CoordinatePolicyError(f"{branch} coordinates must be non-empty")
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise Q25CoordinatePolicyError(f"{branch} coordinates contain duplicates")
    if not _row_major(coordinates):
        raise Q25CoordinatePolicyError(
            f"{branch} coordinates must be ordered row-major by (y,x)"
        )


def generate_level_0_lattice_coordinates(
    *,
    level_0_dimensions: tuple[int, int],
    level_0_patch_size: int,
    level_0_step: int,
    geometry: Q25TissueGeometry,
) -> NDArray[np.int64]:
    """Generate complete global level-0 footprints and accepted origins."""

    level_0_width, level_0_height = _positive_dimensions(
        level_0_dimensions,
        label="level_0_dimensions",
    )
    if level_0_patch_size <= 0 or level_0_step <= 0:
        raise Q25CoordinatePolicyError("level-0 patch size and step must be positive")
    if level_0_width < level_0_patch_size or level_0_height < level_0_patch_size:
        return np.empty((0, 2), dtype=np.int64)

    accepted: list[tuple[int, int]] = []
    for level_0_y in range(
        GRID_ANCHOR[1],
        level_0_height - level_0_patch_size + 1,
        level_0_step,
    ):
        for level_0_x in range(
            GRID_ANCHOR[0],
            level_0_width - level_0_patch_size + 1,
            level_0_step,
        ):
            level_0_origin = (level_0_x, level_0_y)
            if level_0_patch_is_tissue(
                level_0_origin,
                geometry,
                level_0_patch_size=level_0_patch_size,
            ):
                accepted.append(level_0_origin)

    if not accepted:
        return np.empty((0, 2), dtype=np.int64)
    coordinates = np.asarray(accepted, dtype=np.int64)
    if not _row_major(coordinates):
        raise Q25CoordinatePolicyError(
            "level-0 lattice coordinates are not row-major by (y,x)"
        )
    if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
        raise Q25CoordinatePolicyError("level-0 lattice coordinates contain duplicates")
    return coordinates


def clam_int_cast_level_0_geometry(
    *,
    native_patch_size: int,
    coordinate_geometry_scale_xy: tuple[float, float],
) -> tuple[int, int]:
    """Apply CLAM's cast-before-multiply reference-footprint rule.

    This intentionally computes ``patch_size * int(scale)`` per axis, not
    ``int(patch_size * scale)``.  The distinction is part of the frozen
    coordinate policy even though both expressions happen to yield 1024 for
    Q25's near-4x ratios.
    """

    if native_patch_size <= 0:
        raise Q25CoordinatePolicyError("native_patch_size must be positive")
    if len(coordinate_geometry_scale_xy) != 2 or not all(
        math.isfinite(float(item)) and float(item) >= 1.0
        for item in coordinate_geometry_scale_xy
    ):
        raise Q25CoordinatePolicyError(
            "coordinate_geometry_scale_xy must contain finite values >= 1"
        )
    return tuple(
        native_patch_size * int(value)
        for value in coordinate_geometry_scale_xy
    )


def generate_q25_coordinate_bags(
    geometry: Q25TissueGeometry,
    *,
    observation: Q25SlideObservation,
) -> Q25CoordinateBags:
    """Generate both Q25 bags from the same pinned level-0 contour geometry."""

    expected_mask_downsample = observation.coordinate_geometry_scale_xy(MASK_LEVEL)
    if geometry.level_0_dimensions != observation.level_dimensions[0]:
        raise Q25CoordinatePolicyError("Q25 geometry level-0 dimensions drift")
    if geometry.mask_dimensions != observation.level_dimensions[MASK_LEVEL]:
        raise Q25CoordinatePolicyError("Q25 geometry mask dimensions drift")
    if not all(
        _exact_float(actual, expected)
        for actual, expected in zip(
            geometry.mask_downsample_xy,
            expected_mask_downsample,
            strict=True,
        )
    ):
        raise Q25CoordinatePolicyError("Q25 geometry mask downsample drift")

    scale_4x_coordinate_scale = observation.coordinate_geometry_scale_xy(
        SCALE_4X_SOURCE_LEVEL
    )
    # Pinned CLAM first casts each reported/geometry axis scale to int and only
    # then multiplies by patch_size.  For Q25 this deliberately yields a
    # constant 1024x1024 level-0 footprint and step.  It does not reproduce an
    # exact native level-1 grid, whose non-integer y ratio could accumulate
    # occasional extra level-0 pixels.
    scale_4x_level_0_footprint_xy = clam_int_cast_level_0_geometry(
        native_patch_size=SCALE_4X_SOURCE_FOOTPRINT,
        coordinate_geometry_scale_xy=scale_4x_coordinate_scale,
    )
    if scale_4x_level_0_footprint_xy != (1024, 1024):
        raise Q25CoordinatePolicyError("Q25 scale_4x CLAM int-cast geometry drift")

    scale_2x = generate_level_0_lattice_coordinates(
        level_0_dimensions=observation.level_dimensions[SCALE_2X_SOURCE_LEVEL],
        level_0_patch_size=SCALE_2X_SOURCE_FOOTPRINT,
        level_0_step=SCALE_2X_STEP,
        geometry=geometry,
    )
    scale_4x = generate_level_0_lattice_coordinates(
        level_0_dimensions=observation.level_dimensions[0],
        level_0_patch_size=scale_4x_level_0_footprint_xy[0],
        level_0_step=scale_4x_level_0_footprint_xy[0],
        geometry=geometry,
    )
    return Q25CoordinateBags(
        scale_2x=scale_2x,
        scale_4x=scale_4x,
        mask_downsample_xy=expected_mask_downsample,
        scale_4x_coordinate_geometry_scale_xy=scale_4x_coordinate_scale,
        contour_count=len(geometry.contours),
        retained_hole_count=sum(len(items) for items in geometry.holes),
    )


def build_q25_coordinate_bags(
    mask_level_2_image: NDArray[np.generic],
    *,
    observation: Q25SlideObservation,
) -> Q25CoordinateBags:
    """Segment one supplied mask image, then build both coordinate bags."""

    geometry = segment_q25_tissue(
        mask_level_2_image,
        observation=observation,
    )
    return generate_q25_coordinate_bags(geometry, observation=observation)
