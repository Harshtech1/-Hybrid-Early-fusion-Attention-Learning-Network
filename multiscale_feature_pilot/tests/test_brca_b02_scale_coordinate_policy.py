from pathlib import Path
import math
import yaml

ROOT = Path(__file__).resolve().parents[2]
POLICY = ROOT / "multiscale_feature_pilot/config/brca_b02_scale_coordinate_policy.yaml"

def test_b02_policy_geometry_and_execution_lock() -> None:
    p = yaml.safe_load(POLICY.read_text())
    assert p["status"].endswith("EXECUTION_LOCKED")
    assert not any(value for key, value in p["execution_boundary"].items() if key != "status")
    dims = p["pinned_header"]["level_dimensions"]
    assert p["pinned_header"]["coordinate_geometry_scale_xy"]["level_2"] == [dims[0][0]/dims[2][0], dims[0][1]/dims[2][1]]
    assert p["future_mask_policy"]["theoretical_pixel_count"] == dims[2][0] * dims[2][1]

def test_b02_scale_errors_and_lattices() -> None:
    p = yaml.safe_load(POLICY.read_text())
    assert math.isclose(p["scale_policy"]["scale_2x"]["effective_mpp"][0], 0.2505 * 2)
    assert math.isclose(p["scale_policy"]["scale_4x"]["effective_mpp"][0], 0.2505 * 4.0001494261056205)
    assert p["branches"]["scale_2x"]["theoretical_sites_before_tissue_filter"] == 174 * 142
    assert p["branches"]["scale_4x"]["theoretical_sites_before_tissue_filter"] == 87 * 71
