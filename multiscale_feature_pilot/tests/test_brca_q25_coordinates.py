from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from multiscale_feature_pilot.src import brca_q25_coordinates as module
from multiscale_feature_pilot.src.brca_q25_coordinates import (
    A_H,
    A_T,
    CLAM_COMMIT,
    EXPECTED_FILENAME,
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_MD5,
    EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
    EXPECTED_PATIENT_ID,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SLIDE_ID,
    MAX_N_HOLES,
    POLICY_STATUS,
    Q25CoordinatePolicyError,
    Q25SlideObservation,
    Q25TissueGeometry,
    SCALE_2X_INTERPOLATION,
    SCALE_2X_OUTPUT_PATCH_SIZE,
    build_q25_coordinate_bags,
    clam_int_cast_level_0_geometry,
    generate_level_0_lattice_coordinates,
    generate_q25_coordinate_bags,
    level_0_patch_is_tissue,
    segment_tissue_contours,
)


POLICY_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "brca_q25_coordinate_policy.yaml"
)


def _observation() -> Q25SlideObservation:
    return Q25SlideObservation(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        filename=EXPECTED_FILENAME,
        size_bytes=EXPECTED_SIZE_BYTES,
        md5=EXPECTED_MD5,
        sha256=EXPECTED_SHA256,
        mpp_x=0.2525,
        mpp_y=0.2525,
        level_dimensions=EXPECTED_LEVEL_DIMENSIONS,
        openslide_level_downsamples=EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
    )


def _rectangle(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.asarray(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]],
        dtype=np.int32,
    )


def _full_q25_geometry(*, holes=()) -> Q25TissueGeometry:
    observation = _observation()
    return Q25TissueGeometry(
        contours=(_rectangle(0, 0, 65_735, 67_405),),
        holes=(tuple(holes),),
        level_0_dimensions=EXPECTED_LEVEL_DIMENSIONS[0],
        mask_dimensions=EXPECTED_LEVEL_DIMENSIONS[2],
        mask_downsample_xy=observation.coordinate_geometry_scale_xy(2),
    )


def test_identity_hash_and_header_evidence_are_exact_and_fail_closed() -> None:
    observation = _observation()
    assert observation.coordinate_geometry_scale_xy(1) == pytest.approx(
        (4.0, 4.00011868731826)
    )
    assert observation.coordinate_geometry_scale_xy(2) == pytest.approx(
        (16.001947419668937, 16.00332383665717)
    )

    for field_name, value, message in (
        ("md5", "0" * 32, "md5 drift"),
        ("sha256", "0" * 64, "sha256 drift"),
        ("size_bytes", EXPECTED_SIZE_BYTES + 1, "size_bytes drift"),
        ("mpp_x", 0.2526, "mpp_x drift"),
        (
            "level_dimensions",
            ((65_736, 67_406), (16_435, 16_851), (4_108, 4_212), (2_054, 2_106)),
            "level_dimensions drift",
        ),
    ):
        with pytest.raises(Q25CoordinatePolicyError, match=message):
            replace(observation, **{field_name: value})


def test_segmentation_uses_measured_per_axis_mask_scale_and_pinned_clam() -> None:
    image = np.full((400, 400, 4), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (379, 379), (255, 0, 0, 255), -1)
    geometry = segment_tissue_contours(
        image,
        level_0_dimensions=(6_400, 6_404),
        mask_dimensions=(400, 400),
    )

    assert geometry.mask_downsample_xy == pytest.approx((16.0, 16.01))
    assert len(geometry.contours) == 1
    assert geometry.contours[0].dtype == np.int32
    assert geometry.contours[0].reshape(-1, 2).max(axis=0).tolist() == [6080, 6083]
    assert CLAM_COMMIT == "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"
    assert (A_T, A_H, MAX_N_HOLES) == (100, 16, 8)


def test_four_point_boundary_passes_but_center_strictly_inside_hole_fails() -> None:
    outer = _rectangle(128, 128, 896, 896)
    hole = _rectangle(256, 256, 768, 768)
    geometry = Q25TissueGeometry(
        contours=(outer,),
        holes=((hole,),),
        level_0_dimensions=(1_024, 1_024),
        mask_dimensions=(64, 64),
        mask_downsample_xy=(16.0, 16.0),
    )

    # Probe (128,128) and center (256,256) lie on boundaries, so CLAM keeps it.
    assert level_0_patch_is_tissue(
        (0, 0),
        geometry,
        level_0_patch_size=512,
    )
    assert not level_0_patch_is_tissue(
        (256, 256),
        geometry,
        level_0_patch_size=512,
    )


def test_scale_2x_lattice_is_anchor_zero_complete_unique_and_row_major() -> None:
    geometry = Q25TissueGeometry(
        contours=(_rectangle(0, 0, 1_535, 1_023),),
        holes=((),),
        level_0_dimensions=(1_600, 1_100),
        mask_dimensions=(100, 100),
        mask_downsample_xy=(16.0, 11.0),
    )
    coordinates = generate_level_0_lattice_coordinates(
        level_0_dimensions=(1_600, 1_100),
        level_0_patch_size=512,
        level_0_step=512,
        geometry=geometry,
    )

    assert coordinates.tolist() == [
        [0, 0],
        [512, 0],
        [1024, 0],
        [0, 512],
        [512, 512],
        [1024, 512],
    ]
    assert coordinates.dtype == np.int64
    assert np.unique(coordinates, axis=0).shape[0] == coordinates.shape[0]
    assert np.all(coordinates[:, 0] + 512 <= 1_600)
    assert np.all(coordinates[:, 1] + 512 <= 1_100)
    assert SCALE_2X_OUTPUT_PATCH_SIZE == 256
    assert SCALE_2X_INTERPOLATION == "PIL.Image.Resampling.LANCZOS"


def test_scale_4x_uses_constant_complete_clam_int_cast_level_0_grid() -> None:
    assert clam_int_cast_level_0_geometry(
        native_patch_size=256,
        coordinate_geometry_scale_xy=(4.0, 4.9),
    ) == (1024, 1024)
    assert int(256 * 4.9) == 1254  # proves cast-before-multiply is intentional

    level_0_dimensions = (2_501, 3_100)
    geometry = Q25TissueGeometry(
        contours=(_rectangle(0, 0, 2_500, 3_099),),
        holes=((),),
        level_0_dimensions=level_0_dimensions,
        mask_dimensions=(100, 100),
        mask_downsample_xy=(25.01, 31.0),
    )
    coordinates = generate_level_0_lattice_coordinates(
        level_0_dimensions=level_0_dimensions,
        level_0_patch_size=1024,
        level_0_step=1024,
        geometry=geometry,
    )

    assert coordinates.tolist() == [
        [0, 0],
        [1024, 0],
        [0, 1024],
        [1024, 1024],
        [0, 2048],
        [1024, 2048],
    ]
    assert np.all(coordinates[:, 0] + 1024 <= level_0_dimensions[0])
    assert np.all(coordinates[:, 1] + 1024 <= level_0_dimensions[1])
    assert coordinates.dtype == np.int64


def test_q25_both_branches_share_geometry_and_satisfy_output_contract() -> None:
    observation = _observation()
    result = generate_q25_coordinate_bags(
        _full_q25_geometry(),
        observation=observation,
    )

    assert result.policy_status == POLICY_STATUS
    assert result.scale_2x.shape == (16_768, 2)
    assert result.scale_4x.shape == (4_160, 2)
    for coordinates in (result.scale_2x, result.scale_4x):
        assert coordinates.dtype == np.int64
        assert coordinates.shape[1] == 2
        assert coordinates.shape[0] > 0
        assert np.unique(coordinates, axis=0).shape[0] == coordinates.shape[0]
        rows = coordinates.tolist()
        assert rows == sorted(rows, key=lambda point: (point[1], point[0]))
    assert result.scale_4x_coordinate_geometry_scale_xy == pytest.approx(
        (4.0, 4.00011868731826)
    )


def test_empty_tissue_fails_closed_for_both_bag_contract() -> None:
    observation = _observation()
    geometry = Q25TissueGeometry(
        contours=(),
        holes=(),
        level_0_dimensions=EXPECTED_LEVEL_DIMENSIONS[0],
        mask_dimensions=EXPECTED_LEVEL_DIMENSIONS[2],
        mask_downsample_xy=observation.coordinate_geometry_scale_xy(2),
    )
    with pytest.raises(Q25CoordinatePolicyError, match="scale_2x.*non-empty"):
        generate_q25_coordinate_bags(geometry, observation=observation)


def test_integrated_core_has_no_path_or_wsi_api_and_writes_no_artifact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation()
    synthetic_view = np.broadcast_to(
        np.zeros((1, 1, 3), dtype=np.uint8),
        (4_212, 4_108, 3),
    )
    calls = []

    def fake_segment(_image, *, observation):
        calls.append(observation)
        return _full_q25_geometry()

    monkeypatch.setattr(module, "segment_q25_tissue", fake_segment)
    result = build_q25_coordinate_bags(synthetic_view, observation=observation)

    assert calls == [observation]
    assert result.scale_2x.shape[0] > 0
    assert set(inspect.signature(build_q25_coordinate_bags).parameters) == {
        "mask_level_2_image",
        "observation",
    }
    source = inspect.getsource(module)
    assert "import openslide" not in source
    assert "read_region(" not in source
    assert ".write(" not in source
    assert "torch.save" not in source


def test_yaml_matches_the_executable_coordinate_policy() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))

    assert policy["policy_status"] == POLICY_STATUS
    assert policy["q25_identity"]["sha256"] == EXPECTED_SHA256
    assert policy["pinned_header_metadata"]["level_dimensions"] == [
        list(item) for item in EXPECTED_LEVEL_DIMENSIONS
    ]
    assert policy["tissue_mask"]["level"] == 2
    assert policy["tissue_mask"]["shared_by_branches"] is True
    assert policy["tissue_mask"]["clam_commit"] == CLAM_COMMIT
    scale_2x = policy["branches"]["scale_2x"]
    assert scale_2x["grid_label"] == "custom_global_level0_lattice_v1"
    assert scale_2x["released_clam_coordinate_reproduction"] is False
    assert scale_2x["exact_paper_coordinate_reproduction"] is False
    assert scale_2x["native_source_footprint"] == [512, 512]
    assert scale_2x["native_step"] == [512, 512]
    assert scale_2x["output_patch_later"] == [256, 256]
    assert policy["branches"]["scale_4x"]["native_source_footprint"] == [256, 256]
    assert policy["branches"]["scale_4x"]["level_0_declared_footprint"] == [1024, 1024]
    assert policy["branches"]["scale_4x"]["level_0_step"] == [1024, 1024]
    scale_4x = policy["branches"]["scale_4x"]
    assert scale_4x["grid_label"] == "custom_global_level0_lattice_v1"
    assert scale_4x["geometry_compatibility"] == (
        "CLAM_INT_CAST_GEOMETRY_COMPATIBLE"
    )
    assert scale_4x["released_clam_coordinate_reproduction"] is False
    assert policy["branches"]["scale_4x"]["exact_native_level_grid_tiling"] is False
    assert scale_4x["geometry_derivation"] == (
        "256*int(ds_x),256*int(ds_y)"
    )
    assert scale_4x[
        "approved_physical_mpp_from_openslide_scalar"
    ] == pytest.approx(
        [1.0100149842739303, 1.0100149842739303]
    )
    assert policy["tissue_mask"]["exact_released_clam_or_paper_reproduction"] is False
    assert policy["branches"]["scale_4x"]["resampling"] == "none"
    assert policy["coordinate_contract"]["dtype"] == "int64"
    assert policy["coordinate_contract"]["require_nonempty"] is True
    assert policy["coordinate_contract"]["require_unique"] is True
    assert policy["coordinate_contract"]["artifacts_written_by_core"] is False
