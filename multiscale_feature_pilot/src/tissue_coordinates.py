"""Pinned-CLAM tissue segmentation for the approved scale-2x pilot lattice.

The geometry layer accepts an already-read OpenSlide level-2 RGB/RGBA image,
which keeps every pixel operation deterministic and independently testable.
The path-level entry point performs the one required read-only mask-level read
and never writes patches or coordinates itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy.typing import NDArray

from .scale_2x_policy import (
    CLAM_COMMIT,
    SOURCE_FOOTPRINT,
    Level0Dimensions,
    four_pt_easy_points,
    generate_global_lattice,
    is_row_major,
)


SEGMENTATION_LEVEL: Final = 2
SEGMENTATION_DOWNSAMPLE: Final = (16.0, 16.0)
SEGMENTATION_MPP: Final = (3.6432, 3.6432)
S_THRESH: Final = 8
M_THRESH: Final = 7
CLOSE: Final = 4
A_T: Final = 100
A_H: Final = 16
MAX_N_HOLES: Final = 8
REF_PATCH_SIZE: Final = 512


class TissueCoordinateError(ValueError):
    """Raised when input geometry would violate the approved 2x policy."""


@dataclass(frozen=True)
class TissueGeometry:
    """Pinned-CLAM tissue contours and their associated holes in level-0 pixels."""

    contours: tuple[NDArray[np.int32], ...]
    holes: tuple[tuple[NDArray[np.int32], ...], ...]
    segmentation_level: int = SEGMENTATION_LEVEL
    level_downsample: tuple[float, float] = SEGMENTATION_DOWNSAMPLE
    clam_commit: str = CLAM_COMMIT

    def __post_init__(self) -> None:
        if len(self.contours) != len(self.holes):
            raise TissueCoordinateError(
                "each foreground contour must have one associated hole collection"
            )


@dataclass(frozen=True)
class LockedScale2xCoordinates:
    """Coordinates plus the measured mask geometry needed for provenance."""

    coordinates: NDArray[np.int64]
    level_0_dimensions: tuple[int, int]
    segmentation_dimensions: tuple[int, int]
    segmentation_level: int
    segmentation_downsample: tuple[float, float]
    segmentation_mpp: tuple[float, float]
    contour_count: int
    retained_hole_count: int
    clam_commit: str = CLAM_COMMIT


def _approved_image(image: NDArray[np.generic]) -> NDArray[np.uint8]:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise TissueCoordinateError(
            f"segmentation image must have dtype uint8, got {array.dtype}"
        )
    if array.ndim != 3 or array.shape[2] not in (3, 4):
        raise TissueCoordinateError(
            "segmentation image must have shape [height,width,3|4]"
        )
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise TissueCoordinateError("segmentation image dimensions must be positive")
    return array


def _approved_downsample(
    level_downsample: float | tuple[float, float],
) -> tuple[float, float]:
    if isinstance(level_downsample, (int, float)):
        value = (float(level_downsample), float(level_downsample))
    else:
        if len(level_downsample) != 2:
            raise TissueCoordinateError("level downsample must contain x and y values")
        value = (float(level_downsample[0]), float(level_downsample[1]))
    if not np.allclose(value, SEGMENTATION_DOWNSAMPLE, rtol=0.0, atol=1e-6):
        raise TissueCoordinateError(
            "approved tissue segmentation requires OpenSlide level 2 at 16x downsample"
        )
    return value


def segment_approved_tissue(
    level_2_image: NDArray[np.generic],
    *,
    level_downsample: float | tuple[float, float] = SEGMENTATION_DOWNSAMPLE,
) -> TissueGeometry:
    """Reproduce pinned CLAM ``segmentTissue`` for the locked pilot settings.

    Behavior is pinned to CLAM commit ``26e0b6c...``: HSV saturation,
    median-blur kernel 7, fixed threshold 8, 4x4 closing, CCOMP contours,
    scaled area thresholds 100/16, and at most eight retained holes.  Returned
    contours are scaled into level-0 coordinates exactly as CLAM does.
    """

    image = _approved_image(level_2_image)
    scale_x, scale_y = _approved_downsample(level_downsample)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation = cv2.medianBlur(hsv[:, :, 1], M_THRESH)
    _, binary = cv2.threshold(
        saturation,
        S_THRESH,
        255,
        cv2.THRESH_BINARY,
    )
    kernel = np.ones((CLOSE, CLOSE), dtype=np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    found = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = found[-2], found[-1]
    if hierarchy is None or not contours:
        return TissueGeometry(contours=(), holes=())

    # CLAM slices the OpenCV hierarchy to [first_child, parent] and then uses
    # its second column for both top-level and direct-hole selection.
    child_parent = np.squeeze(hierarchy, axis=0)[:, 2:]
    foreground_ids = np.flatnonzero(child_parent[:, 1] == -1)

    scaled_ref_patch_area = int(
        REF_PATCH_SIZE**2 / (scale_x * scale_y)
    )
    tissue_area_threshold = A_T * scaled_ref_patch_area
    hole_area_threshold = A_H * scaled_ref_patch_area

    retained_contours: list[NDArray[np.int32]] = []
    retained_holes: list[tuple[NDArray[np.int32], ...]] = []
    scale = np.asarray((scale_x, scale_y), dtype=np.float64)

    for contour_id in foreground_ids:
        contour = contours[int(contour_id)]
        hole_ids = np.flatnonzero(child_parent[:, 1] == contour_id)
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
            np.asarray(hole * scale, dtype=np.int32)
            for hole in largest_holes
            if cv2.contourArea(hole) > hole_area_threshold
        )
        retained_contours.append(np.asarray(contour * scale, dtype=np.int32))
        retained_holes.append(filtered_holes)

    return TissueGeometry(
        contours=tuple(retained_contours),
        holes=tuple(retained_holes),
    )


def origin_is_tissue(
    origin: tuple[int, int],
    geometry: TissueGeometry,
) -> bool:
    """Apply pinned four-point-easy inclusion and center-in-hole exclusion."""

    center = (
        origin[0] + SOURCE_FOOTPRINT / 2,
        origin[1] + SOURCE_FOOTPRINT / 2,
    )
    probes = four_pt_easy_points(origin, patch_size=SOURCE_FOOTPRINT)

    for contour, holes in zip(geometry.contours, geometry.holes):
        probe_inside = any(
            cv2.pointPolygonTest(
                contour,
                (float(probe[0]), float(probe[1])),
                False,
            )
            >= 0
            for probe in probes
        )
        if not probe_inside:
            continue

        # Pinned CLAM excludes only a point strictly inside a hole.  A center
        # lying exactly on a hole boundary remains accepted.
        center_in_hole = any(
            cv2.pointPolygonTest(hole, center, False) > 0 for hole in holes
        )
        if not center_in_hole:
            return True
    return False


def generate_approved_tissue_coordinates(
    dimensions: Level0Dimensions,
    geometry: TissueGeometry,
) -> NDArray[np.int64]:
    """Return the approved complete-footprint lattice as unique ``[N,2]`` int64."""

    rows = generate_global_lattice(
        dimensions,
        tissue_accepts=lambda origin: origin_is_tissue(origin, geometry),
    )
    if not is_row_major(rows):  # defensive guard against policy drift
        raise RuntimeError("approved coordinates are not ordered row-major by (y,x)")
    if len(rows) != len(set(rows)):
        raise RuntimeError("approved coordinates contain duplicates")
    if not rows:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def build_approved_2x_coordinates(
    level_2_image: NDArray[np.generic],
    dimensions: Level0Dimensions,
    *,
    level_downsample: float | tuple[float, float] = SEGMENTATION_DOWNSAMPLE,
) -> tuple[NDArray[np.int64], TissueGeometry]:
    """Segment one level-2 image and generate the locked 2x coordinates."""

    geometry = segment_approved_tissue(
        level_2_image,
        level_downsample=level_downsample,
    )
    return generate_approved_tissue_coordinates(dimensions, geometry), geometry


def generate_locked_scale_2x_coordinates(
    wsi_path: str | Path,
) -> LockedScale2xCoordinates:
    """Read only the locked mask level and return approved 2x coordinates.

    The slide is always closed, no patch images or coordinate artifacts are
    written, and no CLAM command is executed.  The implementation reproduces
    only the pinned segmentation and contour predicates needed by the custom
    global lattice.
    """

    path = Path(wsi_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    import openslide  # imported lazily so pure geometry tests need no WSI runtime

    slide = openslide.OpenSlide(str(path))
    try:
        if slide.level_count <= SEGMENTATION_LEVEL:
            raise TissueCoordinateError(
                f"slide does not contain required level {SEGMENTATION_LEVEL}"
            )

        level_0_width, level_0_height = map(int, slide.level_dimensions[0])
        segmentation_width, segmentation_height = map(
            int,
            slide.level_dimensions[SEGMENTATION_LEVEL],
        )
        measured_downsample = (
            level_0_width / segmentation_width,
            level_0_height / segmentation_height,
        )
        measured_downsample = _approved_downsample(measured_downsample)

        try:
            base_mpp = (
                float(slide.properties["openslide.mpp-x"]),
                float(slide.properties["openslide.mpp-y"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TissueCoordinateError(
                "slide must expose finite positive openslide.mpp-x and openslide.mpp-y"
            ) from exc
        if not all(math.isfinite(value) and value > 0 for value in base_mpp):
            raise TissueCoordinateError(
                "slide must expose finite positive openslide.mpp-x and openslide.mpp-y"
            )
        measured_mpp = (
            base_mpp[0] * measured_downsample[0],
            base_mpp[1] * measured_downsample[1],
        )
        if not np.allclose(measured_mpp, SEGMENTATION_MPP, rtol=0.0, atol=1e-6):
            raise TissueCoordinateError(
                f"required level-2 mask MPP is {SEGMENTATION_MPP}, got {measured_mpp}"
            )

        level_2_image = np.asarray(
            slide.read_region(
                (0, 0),
                SEGMENTATION_LEVEL,
                (segmentation_width, segmentation_height),
            )
        )
        coordinates, geometry = build_approved_2x_coordinates(
            level_2_image,
            Level0Dimensions(width=level_0_width, height=level_0_height),
            level_downsample=measured_downsample,
        )
        return LockedScale2xCoordinates(
            coordinates=coordinates,
            level_0_dimensions=(level_0_width, level_0_height),
            segmentation_dimensions=(segmentation_width, segmentation_height),
            segmentation_level=SEGMENTATION_LEVEL,
            segmentation_downsample=measured_downsample,
            segmentation_mpp=measured_mpp,
            contour_count=len(geometry.contours),
            retained_hole_count=sum(len(holes) for holes in geometry.holes),
        )
    finally:
        slide.close()
