from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "multiscale_feature_pilot/config/brca_b03_scale_coordinate_policy.yaml"


def _load() -> dict:
    return yaml.safe_load(POLICY.read_text())


def test_b03_policy_is_metadata_only_and_execution_locked() -> None:
    policy = _load()
    assert policy["status"].endswith("EXECUTION_LOCKED")
    assert not any(
        value
        for key, value in policy["execution_boundary"].items()
        if key != "status"
    )
    statement = policy["authority"]["exact_user_statement"]
    assert hashlib.sha256(statement.encode()).hexdigest() == policy["authority"][
        "exact_user_statement_sha256"
    ]


def test_b03_frozen_header_binding_and_coordinate_ratios() -> None:
    policy = _load()
    evidence = policy["frozen_evidence"]
    header_path = ROOT / evidence["header_result_path"]
    assert hashlib.sha256(header_path.read_bytes()).hexdigest() == evidence[
        "header_result_sha256"
    ]
    dims = policy["pinned_header"]["level_dimensions"]
    ratios = policy["pinned_header"]["coordinate_geometry_scale_xy"]
    for level in range(4):
        assert ratios[f"level_{level}"] == [
            dims[0][0] / dims[level][0],
            dims[0][1] / dims[level][1],
        ]


def test_b03_scale_mapping_and_tolerance() -> None:
    policy = _load()
    scale = policy["scale_policy"]
    downsamples = policy["pinned_header"]["openslide_scalar_level_downsamples"]
    mpp = policy["pinned_header"]["mpp_x"]
    assert math.isclose(scale["scale_2x"]["effective_mpp"][0], mpp * 2.0)
    assert math.isclose(scale["scale_4x"]["effective_mpp"][0], mpp * downsamples[1])
    for branch in ("scale_2x", "scale_4x"):
        assert all(
            error <= scale["tolerance_fraction"]
            for error in scale[branch]["relative_target_error"]
        )


def test_b03_mask_arithmetic_is_scalar_only() -> None:
    policy = _load()
    mask = policy["future_mask_policy"]
    width, height = mask["dimensions"]
    assert mask["theoretical_pixel_count"] == width * height == 21_178_192
    assert mask["theoretical_rgba_uint8_bytes"] == width * height * 4 == 84_712_768
    scale_x, scale_y = mask["coordinate_geometry_scale_xy"]
    reference_area = int(512 * 512 / (scale_x * scale_y))
    assert mask["scaled_reference_area"] == reference_area == 1023
    assert mask["tissue_net_area_threshold_mask_pixels"] == reference_area * 100
    assert mask["retained_hole_area_threshold_mask_pixels"] == reference_area * 16


def test_b03_lattice_capacities_last_origins_and_trailing_strips() -> None:
    policy = _load()
    width, height = policy["pinned_header"]["level_dimensions"][0]
    for branch, footprint in (("scale_2x", 512), ("scale_4x", 1024)):
        spec = policy["branches"][branch]
        columns = (width - footprint) // footprint + 1
        rows = (height - footprint) // footprint + 1
        last = [(columns - 1) * footprint, (rows - 1) * footprint]
        trailing = [width - last[0] - footprint, height - last[1] - footprint]
        assert spec["theoretical_columns"] == columns
        assert spec["theoretical_rows"] == rows
        assert spec["theoretical_sites_before_tissue_filter"] == columns * rows
        assert spec["last_complete_origin"] == last
        assert spec["trailing_strip_right_bottom"] == trailing
