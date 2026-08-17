from __future__ import annotations

import pytest
import torch

from multiscale_feature_pilot.src.multiscale_bag import (
    FeatureMatrixError,
    build_multiscale_bag,
    concatenate_feature_matrices,
)
from multiscale_feature_pilot.src.provenance import (
    BranchProvenanceSpec,
    ProvenanceError,
    provenance_as_dicts,
)


def _coordinates(count: int, *, offset: int = 0) -> torch.Tensor:
    indices = torch.arange(count, dtype=torch.int64)
    return torch.stack((indices + offset, (indices + offset) * 10), dim=1)


def test_concatenates_100_and_80_rows_along_patch_axis() -> None:
    scale_2x = torch.full((100, 2048), 2.0, dtype=torch.float32)
    scale_4x = torch.full((80, 2048), 4.0, dtype=torch.float32)

    combined = concatenate_feature_matrices(scale_2x, scale_4x)

    assert combined.shape == (180, 2048)
    assert torch.equal(combined[:100], scale_2x)
    assert torch.equal(combined[100:], scale_4x)


def test_concatenates_one_row_per_branch() -> None:
    combined = concatenate_feature_matrices(
        torch.zeros((1, 2048), dtype=torch.float32),
        torch.ones((1, 2048), dtype=torch.float32),
    )

    assert combined.shape == (2, 2048)


def test_rejects_wrong_feature_width() -> None:
    with pytest.raises(FeatureMatrixError, match="feature width 2048"):
        concatenate_feature_matrices(
            torch.zeros((2, 1024), dtype=torch.float32),
            torch.zeros((2, 2048), dtype=torch.float32),
        )

@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nan_and_inf(bad_value: float) -> None:
    scale_2x = torch.zeros((2, 2048), dtype=torch.float32)
    scale_2x[0, 0] = bad_value

    with pytest.raises(FeatureMatrixError, match="NaN or Inf"):
        concatenate_feature_matrices(
            scale_2x,
            torch.zeros((2, 2048), dtype=torch.float32),
        )


def test_rejects_feature_axis_concat_and_stack() -> None:
    scale_2x = torch.zeros((2, 2048), dtype=torch.float32)
    scale_4x = torch.zeros((3, 2048), dtype=torch.float32)

    with pytest.raises(FeatureMatrixError, match="dim=0"):
        concatenate_feature_matrices(scale_2x, scale_4x, dim=1)
    with pytest.raises(FeatureMatrixError, match="stack is explicitly rejected"):
        concatenate_feature_matrices(scale_2x, scale_4x, operation="stack")


def test_rejects_non_float32() -> None:
    with pytest.raises(FeatureMatrixError, match="torch.float32"):
        concatenate_feature_matrices(
            torch.zeros((2, 2048), dtype=torch.float64),
            torch.zeros((2, 2048), dtype=torch.float32),
        )


def test_provenance_matches_scale_order_and_feature_rows() -> None:
    scale_2x = torch.full((3, 2048), 2.0, dtype=torch.float32)
    scale_4x = torch.full((2, 2048), 4.0, dtype=torch.float32)
    bag = build_multiscale_bag(
        scale_2x,
        scale_4x,
        scale_2x_provenance=BranchProvenanceSpec(
            branch="scale_2x",
            coordinates=_coordinates(3),
            level=0,
            mpp_x=0.5,
            mpp_y=0.5,
        ),
        scale_4x_provenance=BranchProvenanceSpec(
            branch="scale_4x",
            coordinates=_coordinates(2, offset=100),
            level=1,
            mpp_x=1.0,
            mpp_y=1.0,
        ),
    )

    assert bag.features.shape == (5, 2048)
    rows = provenance_as_dicts(bag.provenance)
    assert [row["global_row_index"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["branch"] for row in rows] == [
        "scale_2x",
        "scale_2x",
        "scale_2x",
        "scale_4x",
        "scale_4x",
    ]
    assert [row["local_patch_index"] for row in rows] == [0, 1, 2, 0, 1]
    assert rows[3] == {
        "global_row_index": 3,
        "branch": "scale_4x",
        "local_patch_index": 0,
        "x": 100,
        "y": 1000,
        "level": 1,
        "mpp_x": 1.0,
        "mpp_y": 1.0,
    }


def test_rejects_provenance_count_mismatch() -> None:
    with pytest.raises(ProvenanceError, match="does not match feature-row count"):
        build_multiscale_bag(
            torch.zeros((2, 2048), dtype=torch.float32),
            torch.zeros((1, 2048), dtype=torch.float32),
            scale_2x_provenance=BranchProvenanceSpec(
                branch="scale_2x",
                coordinates=_coordinates(1),
                level=0,
                mpp_x=0.5,
                mpp_y=0.5,
            ),
            scale_4x_provenance=BranchProvenanceSpec(
                branch="scale_4x",
                coordinates=_coordinates(1),
                level=1,
                mpp_x=1.0,
                mpp_y=1.0,
            ),
        )
