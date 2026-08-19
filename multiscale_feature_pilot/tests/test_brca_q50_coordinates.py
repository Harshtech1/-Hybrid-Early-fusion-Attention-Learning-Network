from __future__ import annotations

from dataclasses import replace
import hashlib
import inspect
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from multiscale_feature_pilot.src import brca_q50_coordinates as module
from multiscale_feature_pilot.src.brca_q50_coordinates import (
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
    POLICY_STATUS,
    Q50CoordinatePolicyError,
    Q50SlideObservation,
    Q50TissueGeometry,
    SHARED_ALGORITHM_SOURCE_SHA256,
    build_q50_coordinate_bags,
    generate_q50_coordinate_bags,
    segment_q50_tissue,
)


ROOT = Path(__file__).resolve().parents[2]
POLICY_PATH = ROOT / "multiscale_feature_pilot/config/brca_q50_coordinate_policy.yaml"
SHARED_SOURCE_PATH = ROOT / "multiscale_feature_pilot/src/brca_q25_coordinates.py"


def _observation() -> Q50SlideObservation:
    return Q50SlideObservation(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        filename=EXPECTED_FILENAME,
        size_bytes=EXPECTED_SIZE_BYTES,
        md5=EXPECTED_MD5,
        sha256=EXPECTED_SHA256,
        mpp_x=0.2468,
        mpp_y=0.2468,
        level_dimensions=EXPECTED_LEVEL_DIMENSIONS,
        openslide_level_downsamples=EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
    )


def _rectangle(x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    return np.asarray(
        [[[x0, y0]], [[x1, y0]], [[x1, y1]], [[x0, y1]]], dtype=np.int32
    )


def _full_geometry() -> Q50TissueGeometry:
    observation = _observation()
    return Q50TissueGeometry(
        contours=(_rectangle(0, 0, 99_959, 65_333),),
        holes=((),),
        level_0_dimensions=EXPECTED_LEVEL_DIMENSIONS[0],
        mask_dimensions=EXPECTED_LEVEL_DIMENSIONS[2],
        mask_downsample_xy=observation.coordinate_geometry_scale_xy(2),
    )


def test_q50_identity_header_and_per_axis_geometry_are_exact() -> None:
    observation = _observation()
    assert observation.coordinate_geometry_scale_xy(1) == pytest.approx(
        (4.0, 4.000122451478602)
    )
    assert observation.coordinate_geometry_scale_xy(2) == pytest.approx(
        (16.001280614695054, 16.001469507714916)
    )
    for field_name, value, message in (
        ("md5", "0" * 32, "md5 drift"),
        ("sha256", "0" * 64, "sha256 drift"),
        ("size_bytes", EXPECTED_SIZE_BYTES + 1, "size_bytes drift"),
        ("mpp_x", 0.2469, "mpp_x drift"),
        (
            "level_dimensions",
            ((99_960, 65_334), (24_991, 16_333), (6_247, 4_083), (3_123, 2_041)),
            "level_dimensions drift",
        ),
    ):
        with pytest.raises(Q50CoordinatePolicyError, match=message):
            replace(observation, **{field_name: value})


def test_reviewed_algorithm_dependency_is_exact_and_explicit() -> None:
    assert hashlib.sha256(SHARED_SOURCE_PATH.read_bytes()).hexdigest() == (
        SHARED_ALGORITHM_SOURCE_SHA256
    )
    assert CLAM_COMMIT == "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"


def test_q50_segmentation_uses_its_own_level_dimensions() -> None:
    image = np.full((4_083, 6_247, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (200, 200), (6_000, 3_800), (255, 0, 0), -1)
    geometry = segment_q50_tissue(image, observation=_observation())
    assert geometry.mask_dimensions == (6_247, 4_083)
    assert geometry.level_0_dimensions == (99_960, 65_334)
    assert geometry.mask_downsample_xy == pytest.approx(
        (16.001280614695054, 16.001469507714916)
    )
    assert len(geometry.contours) == 1


def test_both_q50_branches_share_geometry_and_are_deterministic() -> None:
    result = generate_q50_coordinate_bags(_full_geometry(), observation=_observation())
    assert result.policy_status == POLICY_STATUS
    assert result.scale_2x.shape == (24_765, 2)
    assert result.scale_4x.shape == (6_111, 2)
    assert result.mask_downsample_xy == pytest.approx(
        (16.001280614695054, 16.001469507714916)
    )
    assert result.scale_4x_coordinate_geometry_scale_xy == pytest.approx(
        (4.0, 4.000122451478602)
    )
    for coordinates, step in ((result.scale_2x, 512), (result.scale_4x, 1024)):
        assert coordinates.dtype == np.int64
        assert np.unique(coordinates, axis=0).shape[0] == coordinates.shape[0]
        assert coordinates.tolist() == sorted(
            coordinates.tolist(), key=lambda point: (point[1], point[0])
        )
        assert np.all(coordinates[:, 0] % step == 0)
        assert np.all(coordinates[:, 1] % step == 0)
        assert np.all(coordinates[:, 0] <= EXPECTED_LEVEL_DIMENSIONS[0][0] - step)
        assert np.all(coordinates[:, 1] <= EXPECTED_LEVEL_DIMENSIONS[0][1] - step)
    assert result.pixel_reads_authorized is False
    assert result.artifact_writes_authorized is False
    assert result.feature_extraction_authorized is False
    assert result.q75_authorized is False
    assert result.training_authorized is False


def test_empty_tissue_fails_closed() -> None:
    observation = _observation()
    geometry = Q50TissueGeometry(
        contours=(),
        holes=(),
        level_0_dimensions=EXPECTED_LEVEL_DIMENSIONS[0],
        mask_dimensions=EXPECTED_LEVEL_DIMENSIONS[2],
        mask_downsample_xy=observation.coordinate_geometry_scale_xy(2),
    )
    with pytest.raises(Q50CoordinatePolicyError, match="scale_2x.*non-empty"):
        generate_q50_coordinate_bags(geometry, observation=observation)


def test_integrated_core_has_no_path_openslide_read_or_write_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observation = _observation()
    synthetic_view = np.broadcast_to(
        np.zeros((1, 1, 3), dtype=np.uint8), (4_083, 6_247, 3)
    )
    calls: list[Q50SlideObservation] = []

    def fake_segment(_image, *, observation):
        calls.append(observation)
        return _full_geometry()

    monkeypatch.setattr(module, "segment_q50_tissue", fake_segment)
    result = build_q50_coordinate_bags(synthetic_view, observation=observation)
    assert calls == [observation]
    assert result.scale_2x.shape[0] > 0
    assert set(inspect.signature(build_q50_coordinate_bags).parameters) == {
        "mask_level_2_image",
        "observation",
    }
    source = inspect.getsource(module)
    assert "import openslide" not in source
    assert "read_region(" not in source
    assert ".write(" not in source
    assert "torch.save" not in source


def test_yaml_matches_q50_executable_policy_and_locks_execution() -> None:
    policy = yaml.safe_load(POLICY_PATH.read_text(encoding="utf-8"))
    assert policy["policy_status"] == POLICY_STATUS
    assert policy["q50_identity"]["sha256"] == EXPECTED_SHA256
    assert policy["pinned_header_metadata"]["level_dimensions"] == [
        list(item) for item in EXPECTED_LEVEL_DIMENSIONS
    ]
    assert policy["reviewed_algorithm_dependency"]["sha256"] == (
        SHARED_ALGORITHM_SOURCE_SHA256
    )
    assert policy["reviewed_algorithm_dependency"]["numerical_q25_metadata_reused"] is False
    assert policy["tissue_mask"]["level"] == 2
    assert policy["tissue_mask"]["shared_by_branches"] is True
    assert policy["branches"]["scale_2x"]["effective_mpp"] == [0.4936, 0.4936]
    assert policy["branches"]["scale_4x"]["level_0_declared_footprint"] == [1024, 1024]
    assert policy["branches"]["scale_4x"]["geometry_derivation"] == (
        "256*int(ds_x),256*int(ds_y)"
    )
    assert policy["execution_boundary"]["real_mask_read_performed"] is False
    assert policy["execution_boundary"]["separate_reviewed_execution_authorization_required"] is True
    assert policy["scope"]["q75_processing"] == "NOT_AUTHORIZED"
    assert policy["scope"]["training"] == "NOT_AUTHORIZED"
