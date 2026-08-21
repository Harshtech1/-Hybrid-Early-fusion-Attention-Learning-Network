from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "P0005": {"index": 5, "tuple": [[0, 0], 2, [6474, 5594]], "sites": [35148, 8787], "total": 43935},
    "P0006": {"index": 6, "tuple": [[0, 0], 2, [7019, 5184]], "sites": [35478, 8829], "total": 44307},
    "P0007": {"index": 7, "tuple": [[0, 0], 2, [5879, 5017]], "sites": [28548, 7098], "total": 35646},
    "P0008": {"index": 8, "tuple": [[0, 0], 2, [7679, 4583]], "sites": [34177, 8449], "total": 42626},
}


def _policy(label: str) -> dict:
    return yaml.safe_load((ROOT / f"multiscale_feature_pilot/config/brca_{label.lower()}_scale_coordinate_policy.yaml").read_text(encoding="utf-8"))


@pytest.mark.parametrize("label", EXPECTED)
def test_policy_is_exact_header_bound_metadata_only_and_locked(label: str) -> None:
    policy = _policy(label)
    expected = EXPECTED[label]
    assert policy["patient_label"] == label and policy["cohort_index"] == expected["index"]
    assert policy["status"].endswith("EXECUTION_LOCKED")
    assert policy["authority"]["exact_user_statement_sha256"] == "3ada92434dfc6f78bbf29e7d127fe2dd199e8fa11a7ea9356bbcea92e85f88fc"
    evidence = policy["frozen_evidence"]
    for path_key, sha_key in (("header_result_path", "header_result_sha256"), ("header_report_path", "header_report_sha256")):
        assert hashlib.sha256((ROOT / evidence[path_key]).read_bytes()).hexdigest() == evidence[sha_key]
    assert not any(value for key, value in policy["execution_boundary"].items() if key != "status")


@pytest.mark.parametrize("label", EXPECTED)
def test_coordinate_ratios_scale_mapping_and_mask_tuple_are_exact(label: str) -> None:
    policy = _policy(label)
    dims = policy["pinned_header"]["level_dimensions"]
    for level, level_dims in enumerate(dims):
        expected_ratio = [dims[0][0] / level_dims[0], dims[0][1] / level_dims[1]]
        assert policy["pinned_header"]["coordinate_geometry_scale_xy"][f"level_{level}"] == expected_ratio
    mpp = policy["pinned_header"]["mpp_x"]
    scalar = policy["pinned_header"]["openslide_scalar_level_downsamples"]
    assert policy["scale_policy"]["scale_2x"]["effective_mpp"] == [mpp * 2] * 2
    assert policy["scale_policy"]["scale_4x"]["effective_mpp"] == [mpp * scalar[1]] * 2
    assert all(error <= 0.10 for branch in ("scale_2x", "scale_4x") for error in policy["scale_policy"][branch]["relative_target_error"])
    read = policy["future_mask_policy"]["proposed_read_region"]
    expected = EXPECTED[label]["tuple"]
    assert [read["level_0_location"], read["level"], read["size"]] == expected
    assert [policy["future_mask_policy"]["level_0_location"], policy["future_mask_policy"]["level"], policy["future_mask_policy"]["dimensions"]] == expected
    assert policy["future_mask_policy"]["shared_by_branches"] is True
    assert read["size"] == dims[2]
    assert policy["future_mask_policy"]["theoretical_pixel_count"] == dims[2][0] * dims[2][1]
    assert policy["future_mask_policy"]["theoretical_rgba_uint8_bytes"] == dims[2][0] * dims[2][1] * 4


@pytest.mark.parametrize("label", EXPECTED)
def test_lattice_capacities_and_threshold_scaling_are_exact(label: str) -> None:
    policy = _policy(label)
    width, height = policy["pinned_header"]["level_dimensions"][0]
    for branch, footprint, expected_sites in zip(("scale_2x", "scale_4x"), (512, 1024), EXPECTED[label]["sites"], strict=True):
        spec = policy["branches"][branch]
        columns = (width - footprint) // footprint + 1
        rows = (height - footprint) // footprint + 1
        last = [(columns - 1) * footprint, (rows - 1) * footprint]
        trailing = [width - last[0] - footprint, height - last[1] - footprint]
        assert (spec["theoretical_columns"], spec["theoretical_rows"], spec["theoretical_sites_before_tissue_filter"]) == (columns, rows, expected_sites)
        assert spec["last_complete_origin"] == last and spec["trailing_strip_right_bottom"] == trailing
    assert policy["branches"]["theoretical_total_sites_before_tissue_filter"] == EXPECTED[label]["total"]
    scale_x, scale_y = policy["future_mask_policy"]["coordinate_geometry_scale_xy"]
    reference = int(512 * 512 / (scale_x * scale_y))
    assert reference == policy["future_mask_policy"]["scaled_reference_area"] == 1023
    assert policy["future_mask_policy"]["tissue_net_area_threshold_mask_pixels"] == reference * 100
    assert policy["future_mask_policy"]["retained_hole_area_threshold_mask_pixels"] == reference * 16


def test_review_records_bind_policy_report_and_test_hashes() -> None:
    for label in EXPECTED:
        review = yaml.safe_load((ROOT / f"multiscale_feature_pilot/provenance/brca_{label.lower()}_scale_coordinate_policy_review.yaml").read_text(encoding="utf-8"))
        assert review["required_stop_reached"] is True
        assert review["review"]["pixels_read"] == review["review"]["coordinates_generated"] == 0
        for path_key, sha_key in (("policy_path", "policy_sha256"), ("report_path", "report_sha256"), ("test_path", "test_sha256")):
            assert hashlib.sha256((ROOT / review["outputs"][path_key]).read_bytes()).hexdigest() == review["outputs"][sha_key]
