from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "P0002": {"index": 2, "dims": [[73704, 77550], [18426, 19387], [4606, 4846], [2303, 2423]], "downsamples": [1.0, 4.0000515809563115, 16.002312922780654, 32.00462584556131], "tuple": {"level_0_location": [0, 0], "level": 2, "size_at_level": [4606, 4846]}, "sites": [21593, 5325, 26918], "header_sha": "b778b1eef8e1116023c6473e179fcbbc1f29f2d87bbf7db0cf930c47f33cb9a4"},
    "P0003": {"index": 3, "dims": [[61752, 59840], [15438, 14960], [3859, 3740], [1929, 1870]], "downsamples": [1.0, 4.0, 16.001036537963202, 32.00622083981337], "tuple": {"level_0_location": [0, 0], "level": 2, "size_at_level": [3859, 3740]}, "sites": [13920, 3480, 17400], "header_sha": "b3944520e2dafee3c8dc0faf18c72b433e4e4e46e40c495c87aab186c98bc37a"},
    "P0004": {"index": 4, "dims": [[89640, 79673], [22410, 19918], [5602, 4979], [2801, 2489]], "downsamples": [1.0, 4.00002510292198, 16.00161782664628, 32.00645015863444], "tuple": {"level_0_location": [0, 0], "level": 2, "size_at_level": [5602, 4979]}, "sites": [27125, 6699, 33824], "header_sha": "22fdf4b1e4a2c3496a0af1d6c213286356643527bad535a5e1937164b1464a5a"},
}


def load(label: str) -> dict:
    return yaml.safe_load((ROOT / f"multiscale_feature_pilot/config/brca_{label.lower()}_scale_coordinate_policy.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("label", EXPECTED)
def test_frozen_header_binding_and_execution_lock(label: str) -> None:
    policy = load(label)
    expected = EXPECTED[label]
    assert policy["patient_label"] == label and policy["cohort_index"] == expected["index"]
    assert policy["status"].endswith("EXECUTION_LOCKED")
    assert policy["pinned_header"]["level_dimensions"] == expected["dims"]
    assert policy["pinned_header"]["openslide_scalar_level_downsamples"] == expected["downsamples"]
    header_path = ROOT / policy["frozen_evidence"]["header_result_path"]
    assert hashlib.sha256(header_path.read_bytes()).hexdigest() == expected["header_sha"]
    assert not any(value for key, value in policy["execution_boundary"].items() if key != "status")


@pytest.mark.parametrize("label", EXPECTED)
def test_coordinate_geometry_is_dimension_derived(label: str) -> None:
    policy = load(label)
    dims = policy["pinned_header"]["level_dimensions"]
    ratios = policy["pinned_header"]["coordinate_geometry_scale_xy"]
    for level in range(4):
        assert ratios[f"level_{level}"] == [dims[0][0] / dims[level][0], dims[0][1] / dims[level][1]]
    assert policy["future_mask_policy"]["coordinate_geometry_scale_xy"] == ratios["level_2"]


@pytest.mark.parametrize("label", EXPECTED)
def test_scale_mapping_and_tolerance(label: str) -> None:
    policy = load(label)
    mpp = policy["pinned_header"]["mpp_x"]
    downsample = policy["pinned_header"]["openslide_scalar_level_downsamples"][1]
    assert policy["scale_policy"]["scale_2x"]["effective_mpp"] == [mpp * 2] * 2
    assert policy["scale_policy"]["scale_4x"]["effective_mpp"] == [mpp * downsample] * 2
    for branch in ("scale_2x", "scale_4x"):
        assert all(error <= 0.10 for error in policy["scale_policy"][branch]["relative_target_error"])


@pytest.mark.parametrize("label", EXPECTED)
def test_exact_proposed_mask_tuple_and_area_arithmetic(label: str) -> None:
    policy = load(label)
    mask = policy["future_mask_policy"]
    assert mask["read_region_tuple"] == EXPECTED[label]["tuple"]
    width, height = mask["dimensions"]
    assert mask["theoretical_pixel_count"] == width * height
    assert mask["theoretical_rgba_uint8_bytes"] == width * height * 4
    sx, sy = mask["coordinate_geometry_scale_xy"]
    assert mask["scaled_reference_area"] == int(512 * 512 / (sx * sy)) == 1023
    assert mask["tissue_net_area_threshold_mask_pixels"] == 102300
    assert mask["retained_hole_area_threshold_mask_pixels"] == 16368


@pytest.mark.parametrize("label", EXPECTED)
def test_exact_lattice_capacity(label: str) -> None:
    policy = load(label)
    width, height = policy["pinned_header"]["level_dimensions"][0]
    expected_sites = EXPECTED[label]["sites"]
    total = 0
    for index, (branch, footprint) in enumerate((("scale_2x", 512), ("scale_4x", 1024))):
        spec = policy["branches"][branch]
        columns = (width - footprint) // footprint + 1
        rows = (height - footprint) // footprint + 1
        assert spec["theoretical_columns"] == columns
        assert spec["theoretical_rows"] == rows
        assert spec["theoretical_sites_before_tissue_filter"] == expected_sites[index] == columns * rows
        total += columns * rows
    assert policy["branches"]["theoretical_total_sites_before_tissue_filter"] == expected_sites[2] == total


def test_no_policy_claims_execution_or_selected_counts() -> None:
    for label in EXPECTED:
        policy = load(label)
        assert policy["next_gate"].startswith("COMBINED_EXPLICIT")
        assert "actual" not in policy["future_mask_policy"]


@pytest.mark.parametrize("label", EXPECTED)
def test_review_record_binds_policy_report_and_test(label: str) -> None:
    review = yaml.safe_load((ROOT / f"multiscale_feature_pilot/provenance/brca_{label.lower()}_scale_coordinate_policy_review.yaml").read_text(encoding="utf-8"))
    assert review["required_stop_reached"] is True
    assert review["review"]["real_wsi_opened"] is False
    assert review["review"]["pixels_read"] == 0
    for path_key, hash_key in (("policy_path", "policy_sha256"), ("report_path", "report_sha256")):
        assert hashlib.sha256((ROOT / review["outputs"][path_key]).read_bytes()).hexdigest() == review["outputs"][hash_key]
