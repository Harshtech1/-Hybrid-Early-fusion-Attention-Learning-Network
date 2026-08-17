from __future__ import annotations

from PIL import Image

from multiscale_feature_pilot.src.multiscale_bag import concatenate_feature_matrices
from multiscale_feature_pilot.src.scale_2x_policy import (
    CLAM_COMMIT,
    EFFECTIVE_MPP,
    GRID_LABEL,
    GRID_STEP,
    Level0Dimensions,
    downsample_source_patch,
    four_pt_easy_accepts,
    four_pt_easy_points,
    generate_global_lattice,
    is_row_major,
)


def test_512_source_patch_becomes_256_rgb_with_lanczos() -> None:
    source = Image.new("RGBA", (512, 512), (12, 34, 56, 78))
    output = downsample_source_patch(source)
    assert output.size == (256, 256)
    assert output.mode == "RGB"


def test_global_lattice_uses_level_0_step_512_and_row_major_order() -> None:
    coordinates = generate_global_lattice(Level0Dimensions(width=1536, height=1024))
    assert coordinates == (
        (0, 0),
        (512, 0),
        (1024, 0),
        (0, 512),
        (512, 512),
        (1024, 512),
    )
    assert GRID_STEP == 512
    assert is_row_major(coordinates)
    assert GRID_LABEL == "custom_global_lattice_v1"


def test_incomplete_right_and_bottom_footprints_are_rejected() -> None:
    coordinates = generate_global_lattice(Level0Dimensions(width=1100, height=700))
    assert coordinates == ((0, 0), (512, 0))
    assert all(x + 512 <= 1100 and y + 512 <= 700 for x, y in coordinates)


def test_coordinate_generation_is_deterministic_and_filterable() -> None:
    dimensions = Level0Dimensions(width=1024, height=1024)
    accepts = lambda origin: origin != (512, 0)
    first = generate_global_lattice(dimensions, tissue_accepts=accepts)
    second = generate_global_lattice(dimensions, tissue_accepts=accepts)
    assert first == second == ((0, 0), (0, 512), (512, 512))


def test_effective_mpp_is_approved_engineering_value() -> None:
    assert EFFECTIVE_MPP == 0.4554


def test_four_pt_easy_matches_pinned_clam_probe_geometry_and_any_rule() -> None:
    assert CLAM_COMMIT == "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"
    probes = four_pt_easy_points((0, 0), patch_size=512)
    assert probes == ((128, 128), (384, 384), (384, 128), (128, 384))
    assert four_pt_easy_accepts((0, 0), lambda point: point == (384, 128))
    assert not four_pt_easy_accepts((0, 0), lambda _point: False)


def test_scale_2_rows_still_precede_scale_4_rows() -> None:
    import torch

    scale_2x = torch.full((2, 2048), 2.0, dtype=torch.float32)
    scale_4x = torch.full((1, 2048), 4.0, dtype=torch.float32)
    combined = concatenate_feature_matrices(scale_2x, scale_4x)
    assert torch.equal(combined[:2], scale_2x)
    assert torch.equal(combined[2:], scale_4x)
