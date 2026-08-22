from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image

from multiscale_feature_pilot.src.scale_2x_policy import CLAM_COMMIT, Level0Dimensions
from multiscale_feature_pilot.src.tissue_coordinates import (
    A_H,
    A_T,
    MAX_N_HOLES,
    SEGMENTATION_LEVEL,
    TissueCoordinateError,
    TissueGeometry,
    build_approved_2x_coordinates,
    generate_locked_scale_2x_coordinates,
    generate_approved_tissue_coordinates,
    origin_is_tissue,
    segment_approved_tissue,
)


def _rectangle(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.asarray(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
        dtype=np.int32,
    )


def test_locked_level_and_pinned_commit() -> None:
    assert SEGMENTATION_LEVEL == 2
    assert CLAM_COMMIT == "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"
    assert (A_T, A_H, MAX_N_HOLES) == (100, 16, 8)


def test_rejects_wrong_segmentation_downsample_and_image_contract() -> None:
    image = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(TissueCoordinateError, match="level 2 at 16x"):
        segment_approved_tissue(image, level_downsample=64.0)
    with pytest.raises(TissueCoordinateError, match="dtype uint8"):
        segment_approved_tissue(image.astype(np.float32))
    with pytest.raises(TissueCoordinateError, match=r"\[height,width,3\|4\]"):
        segment_approved_tissue(image[:, :, 0])


def test_segment_approved_tissue_uses_scaled_area_threshold() -> None:
    # At 16x, CLAM scales the reference area to 1024 pixels, so a_t=100
    # requires strictly more than 102,400 foreground pixels.
    below_threshold = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(below_threshold, (40, 40), (339, 339), (255, 0, 0), -1)
    assert segment_approved_tissue(below_threshold).contours == ()

    above_threshold = np.full((400, 400, 4), 255, dtype=np.uint8)
    cv2.rectangle(above_threshold, (20, 20), (379, 379), (255, 0, 0, 255), -1)
    geometry = segment_approved_tissue(above_threshold)
    assert len(geometry.contours) == 1
    assert geometry.contours[0].dtype == np.int32
    # The pinned even-sized 4x4 closing kernel advances this synthetic edge
    # by one mask pixel before level-0 scaling.
    assert geometry.contours[0].reshape(-1, 2).max(axis=0).tolist() == [6080, 6080]


def test_large_hole_is_retained_and_center_excluded() -> None:
    image = np.full((512, 512, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (16, 16), (495, 495), (255, 0, 0), -1)
    cv2.rectangle(image, (176, 176), (335, 335), (255, 255, 255), -1)

    geometry = segment_approved_tissue(image)
    assert len(geometry.contours) == 1
    assert len(geometry.holes[0]) == 1

    # Both origins pass the outer four-point test.  Only the second patch
    # center lies strictly inside the retained hole.
    assert origin_is_tissue((512, 512), geometry)
    assert not origin_is_tissue((3072, 3072), geometry)


def test_four_point_boundary_is_inside_and_hole_boundary_is_not_excluded() -> None:
    outer = _rectangle(128, 128, 896, 896)
    hole = _rectangle(256, 256, 768, 768)
    geometry = TissueGeometry(contours=(outer,), holes=((hole,),))

    # For origin (0,0), probe (128,128) is exactly on the outer boundary.
    # Its center (256,256) is exactly on the hole boundary. Pinned CLAM uses
    # >= 0 for the former and > 0 for the latter, so this remains accepted.
    assert origin_is_tissue((0, 0), geometry)


def test_global_lattice_is_complete_row_major_unique_and_deterministic() -> None:
    geometry = TissueGeometry(
        contours=(_rectangle(0, 0, 1536, 1024),),
        holes=((),),
    )
    dimensions = Level0Dimensions(width=1600, height=1100)

    first = generate_approved_tissue_coordinates(dimensions, geometry)
    second = generate_approved_tissue_coordinates(dimensions, geometry)

    expected = np.asarray(
        [
            [0, 0],
            [512, 0],
            [1024, 0],
            [0, 512],
            [512, 512],
            [1024, 512],
        ],
        dtype=np.int64,
    )
    assert np.array_equal(first, expected)
    assert np.array_equal(second, expected)
    assert first.dtype == np.int64
    assert np.unique(first, axis=0).shape[0] == first.shape[0]
    assert np.all(first[:, 0] + 512 <= dimensions.width)
    assert np.all(first[:, 1] + 512 <= dimensions.height)


def test_end_to_end_synthetic_coordinate_build_without_wsi_io() -> None:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (379, 379), (255, 0, 0), -1)
    coordinates, geometry = build_approved_2x_coordinates(
        image,
        Level0Dimensions(width=6400, height=6400),
    )

    assert len(geometry.contours) == 1
    assert coordinates.ndim == 2
    assert coordinates.shape[1] == 2
    assert coordinates.dtype == np.int64
    assert coordinates.shape[0] > 0


def test_path_api_reads_only_locked_level_and_returns_provenance(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    image = np.full((400, 400, 4), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (379, 379), (255, 0, 0, 255), -1)

    class FakeSlide:
        level_count = 3
        level_dimensions = ((6400, 6400), (1600, 1600), (400, 400))
        properties = {"openslide.mpp-x": "0.2277", "openslide.mpp-y": "0.2277"}

        def __init__(self) -> None:
            self.read_calls = []
            self.closed = False

        def read_region(self, location, level, size):
            self.read_calls.append((location, level, size))
            return Image.fromarray(image, mode="RGBA")

        def close(self) -> None:
            self.closed = True

    fake_slide = FakeSlide()
    import openslide

    monkeypatch.setattr(openslide, "OpenSlide", lambda _path: fake_slide)
    path = tmp_path / "synthetic.svs"
    path.touch()

    result = generate_locked_scale_2x_coordinates(path)

    assert fake_slide.read_calls == [((0, 0), 2, (400, 400))]
    assert fake_slide.closed
    assert result.coordinates.dtype == np.int64
    assert result.coordinates.shape[1] == 2
    assert result.level_0_dimensions == (6400, 6400)
    assert result.segmentation_dimensions == (400, 400)
    assert result.segmentation_level == 2
    assert result.segmentation_downsample == (16.0, 16.0)
    assert result.segmentation_mpp == pytest.approx((3.6432, 3.6432))
    assert result.contour_count == 1
    assert result.retained_hole_count == 0
