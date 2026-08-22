from __future__ import annotations

import hashlib
import math
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "multiscale_feature_pilot/config/brca_p0001_scale_coordinate_policy.yaml"
PROVENANCE = ROOT / "multiscale_feature_pilot/provenance/brca_p0001_scale_coordinate_policy_review.yaml"


def _load() -> dict:
    return yaml.safe_load(POLICY.read_text(encoding="utf-8"))


def test_p0001_policy_is_metadata_only_and_execution_locked() -> None:
    policy = _load()
    assert policy["status"].endswith("EXECUTION_LOCKED")
    assert policy["cohort_index"] == 1
    assert policy["patient_label"] == "P0001"
    assert not any(
        value
        for key, value in policy["execution_boundary"].items()
        if key != "status"
    )
    statement = policy["authority"]["exact_user_statement"]
    assert hashlib.sha256(statement.encode()).hexdigest() == policy["authority"][
        "exact_user_statement_sha256"
    ]


def test_p0001_frozen_header_and_reviewed_policy_bindings() -> None:
    policy = _load()
    evidence = policy["frozen_evidence"]
    for path_key, hash_key in (
        ("header_result_path", "header_result_sha256"),
        ("header_report_path", "header_report_sha256"),
    ):
        path = ROOT / evidence[path_key]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence[hash_key]
    frozen = {
        "b01_reviewed_policy_sha256": "multiscale_feature_pilot/config/brca_b01_scale_coordinate_policy.yaml",
        "b02_reviewed_policy_sha256": "multiscale_feature_pilot/config/brca_b02_scale_coordinate_policy.yaml",
        "b03_reviewed_policy_sha256": "multiscale_feature_pilot/config/brca_b03_scale_coordinate_policy.yaml",
        "q75_coordinate_policy_sha256": "multiscale_feature_pilot/config/brca_q75_coordinate_policy.yaml",
    }
    for hash_key, relative in frozen.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == evidence[hash_key]


def test_p0001_exact_coordinate_ratios_and_scalar_downsamples() -> None:
    policy = _load()
    dims = policy["pinned_header"]["level_dimensions"]
    ratios = policy["pinned_header"]["coordinate_geometry_scale_xy"]
    scalar = policy["pinned_header"]["openslide_scalar_level_downsamples"]
    for level in range(4):
        expected = [dims[0][0] / dims[level][0], dims[0][1] / dims[level][1]]
        assert ratios[f"level_{level}"] == expected
        assert expected == [scalar[level], scalar[level]]


def test_p0001_scale_mapping_has_zero_target_error() -> None:
    policy = _load()
    scale = policy["scale_policy"]
    mpp = policy["pinned_header"]["mpp_x"]
    downsamples = policy["pinned_header"]["openslide_scalar_level_downsamples"]
    assert scale["scale_2x"]["effective_mpp"] == [mpp * 2.0, mpp * 2.0] == [0.5, 0.5]
    assert scale["scale_4x"]["effective_mpp"] == [mpp * downsamples[1]] * 2 == [1.0, 1.0]
    for branch in ("scale_2x", "scale_4x"):
        assert scale[branch]["relative_target_error"] == [0.0, 0.0]
        assert all(error <= scale["tolerance_fraction"] for error in scale[branch]["relative_target_error"])


def test_p0001_mask_arithmetic_is_theoretical_and_exact() -> None:
    policy = _load()
    mask = policy["future_mask_policy"]
    width, height = mask["dimensions"]
    assert mask["level_0_location"] == [0, 0]
    assert mask["theoretical_pixel_count"] == width * height == 30_556_160
    assert mask["theoretical_rgba_uint8_bytes"] == width * height * 4 == 122_224_640
    scale_x, scale_y = mask["coordinate_geometry_scale_xy"]
    reference_area = int(512 * 512 / (scale_x * scale_y))
    assert mask["scaled_reference_area"] == reference_area == 1024
    assert mask["tissue_net_area_threshold_mask_pixels"] == reference_area * 100 == 102_400
    assert mask["retained_hole_area_threshold_mask_pixels"] == reference_area * 16 == 16_384


def test_p0001_lattice_capacities_origins_and_trailing_strips() -> None:
    policy = _load()
    width, height = policy["pinned_header"]["level_dimensions"][0]
    total = 0
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
        total += columns * rows
    assert policy["branches"]["theoretical_total_sites_before_tissue_filter"] == total == 37_200


def test_p0001_policy_does_not_claim_coordinate_execution() -> None:
    policy = _load()
    assert policy["future_artifact_contract"]["shape_each"] == "[N,2]"
    assert policy["future_artifact_contract"]["dtype"] == "int64"
    assert policy["next_gate"].startswith("SEPARATE_EXPLICIT_P0001")
    assert math.prod(policy["future_mask_policy"]["dimensions"]) == 30_556_160


def test_p0001_review_provenance_binds_all_outputs() -> None:
    review = yaml.safe_load(PROVENANCE.read_text(encoding="utf-8"))
    assert review["required_stop_reached"] is True
    assert review["resolved_policy"]["actual_tissue_selected_counts"].startswith("UNKNOWN")
    outputs = review["outputs"]
    for path_key, hash_key in (
        ("policy_path", "policy_sha256"),
        ("report_path", "report_sha256"),
        ("test_path", "test_sha256"),
    ):
        assert hashlib.sha256((ROOT / outputs[path_key]).read_bytes()).hexdigest() == outputs[hash_key]
