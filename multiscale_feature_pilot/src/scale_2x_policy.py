"""Approved geometry for the controlled scale-2x engineering pilot."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

from PIL import Image


POLICY_STATUS = "APPROVED_2X_ENGINEERING_POLICY_V1"
CLAM_COMMIT = "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"
GRID_LABEL = "custom_global_lattice_v1"
SOURCE_LEVEL = 0
SOURCE_MPP = 0.2277
SOURCE_FOOTPRINT = 512
OUTPUT_PATCH_SIZE = 256
GRID_STEP = 512
EFFECTIVE_MPP = 0.4554


@dataclass(frozen=True)
class Level0Dimensions:
    """Integer level-0 slide dimensions without opening a WSI."""

    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("level-0 dimensions must be positive")


def four_pt_easy_points(
    origin: tuple[int, int],
    *,
    patch_size: int = SOURCE_FOOTPRINT,
) -> tuple[tuple[int, int], ...]:
    """Return the exact four probes used by pinned CLAM ``isInContourV3_Easy``.

    Pinned provenance: CLAM commit ``26e0b6c4873e112f1ccd74cd834894c4ab7a2934``.
    A boundary point is considered inside by the caller, matching OpenCV's
    ``pointPolygonTest(...) >= 0`` condition in that implementation.
    """

    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    x, y = origin
    half = patch_size // 2
    shift = int(half * 0.5)
    center_x, center_y = x + half, y + half
    return (
        (center_x - shift, center_y - shift),
        (center_x + shift, center_y + shift),
        (center_x + shift, center_y - shift),
        (center_x - shift, center_y + shift),
    )


def four_pt_easy_accepts(
    origin: tuple[int, int],
    point_in_contour: Callable[[tuple[int, int]], bool],
    *,
    patch_size: int = SOURCE_FOOTPRINT,
) -> bool:
    """Apply pinned CLAM's easy rule: accept when any one probe is inside."""

    return any(
        point_in_contour(point)
        for point in four_pt_easy_points(origin, patch_size=patch_size)
    )


def generate_global_lattice(
    dimensions: Level0Dimensions,
    *,
    tissue_accepts: Callable[[tuple[int, int]], bool] | None = None,
) -> tuple[tuple[int, int], ...]:
    """Generate complete 512-pixel footprints in deterministic row-major order.

    This is the approved custom global lattice, not released-CLAM coordinate
    reproduction. Origins are level-0 ``(x, y)`` coordinates anchored at zero.
    Incomplete right and bottom footprints are rejected by construction.
    """

    accepts = tissue_accepts or (lambda _origin: True)
    if dimensions.width < SOURCE_FOOTPRINT or dimensions.height < SOURCE_FOOTPRINT:
        return ()
    return tuple(
        (x, y)
        for y in range(0, dimensions.height - SOURCE_FOOTPRINT + 1, GRID_STEP)
        for x in range(0, dimensions.width - SOURCE_FOOTPRINT + 1, GRID_STEP)
        if accepts((x, y))
    )


def downsample_source_patch(image: Image.Image) -> Image.Image:
    """Convert a complete source footprint to RGB and Lanczos-downsample it."""

    if image.size != (SOURCE_FOOTPRINT, SOURCE_FOOTPRINT):
        raise ValueError(
            f"source patch must be {SOURCE_FOOTPRINT}x{SOURCE_FOOTPRINT}, got {image.size}"
        )
    return image.convert("RGB").resize(
        (OUTPUT_PATCH_SIZE, OUTPUT_PATCH_SIZE),
        resample=Image.Resampling.LANCZOS,
    )


def is_row_major(coordinates: Iterable[tuple[int, int]]) -> bool:
    """Return whether coordinates are ordered by ``(y, x)``."""

    rows = list(coordinates)
    return rows == sorted(rows, key=lambda point: (point[1], point[0]))
