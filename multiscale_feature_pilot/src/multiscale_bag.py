"""Strict construction of a two-scale, 2048-D WSI feature bag."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch

from .provenance import (
    BranchProvenanceSpec,
    PatchProvenance,
    build_two_scale_provenance,
    validate_provenance_alignment,
)


FEATURE_DIM = 2048


class FeatureMatrixError(ValueError):
    """Raised when a feature matrix violates the pilot contract."""


@dataclass(frozen=True)
class MultiscaleBag:
    """Validated combined features and their immutable row provenance."""

    features: torch.Tensor
    provenance: tuple[PatchProvenance, ...]


def validate_feature_matrix(
    matrix: torch.Tensor,
    *,
    name: str,
    feature_dim: int = FEATURE_DIM,
) -> torch.Tensor:
    """Validate shape, dtype, finiteness, and non-empty rows without mutation."""

    if not isinstance(matrix, torch.Tensor):
        raise FeatureMatrixError(f"{name}: expected torch.Tensor, got {type(matrix).__name__}")
    if matrix.ndim != 2:
        raise FeatureMatrixError(f"{name}: expected ndim=2, got shape {tuple(matrix.shape)}")
    if matrix.shape[0] == 0:
        raise FeatureMatrixError(f"{name}: feature matrix must contain at least one patch row")
    if matrix.shape[1] != feature_dim:
        raise FeatureMatrixError(
            f"{name}: expected feature width {feature_dim}, got {matrix.shape[1]}"
        )
    if matrix.dtype is not torch.float32:
        raise FeatureMatrixError(f"{name}: expected dtype torch.float32, got {matrix.dtype}")
    if not bool(torch.isfinite(matrix).all().item()):
        raise FeatureMatrixError(f"{name}: feature matrix contains NaN or Inf")
    return matrix


def load_feature_matrix(
    path: str | Path,
    *,
    name: str,
    map_location: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load one tensor-only `.pt` artifact and apply the strict matrix contract."""

    feature_path = Path(path)
    if not feature_path.is_file():
        raise FileNotFoundError(feature_path)
    matrix = torch.load(feature_path, map_location=map_location, weights_only=True)
    return validate_feature_matrix(matrix, name=name)


def concatenate_feature_matrices(
    scale_2x: torch.Tensor,
    scale_4x: torch.Tensor,
    *,
    operation: Literal["cat"] | str = "cat",
    dim: int = 0,
) -> torch.Tensor:
    """Concatenate scale-2x rows before scale-4x rows; reject other operations."""

    if operation != "cat":
        raise FeatureMatrixError("only torch.cat is supported; stack is explicitly rejected")
    if dim != 0:
        raise FeatureMatrixError("multiscale features must be concatenated along dim=0")

    validate_feature_matrix(scale_2x, name="scale_2x")
    validate_feature_matrix(scale_4x, name="scale_4x")
    if scale_2x.device != scale_4x.device:
        raise FeatureMatrixError(
            f"branch devices must match, got {scale_2x.device} and {scale_4x.device}"
        )

    combined = torch.cat((scale_2x, scale_4x), dim=0)
    validate_feature_matrix(combined, name="combined")
    expected_rows = scale_2x.shape[0] + scale_4x.shape[0]
    if combined.shape != (expected_rows, FEATURE_DIM):
        raise FeatureMatrixError(
            f"combined shape must be {(expected_rows, FEATURE_DIM)}, got {tuple(combined.shape)}"
        )
    return combined


def build_multiscale_bag(
    scale_2x_features: torch.Tensor,
    scale_4x_features: torch.Tensor,
    *,
    scale_2x_provenance: BranchProvenanceSpec,
    scale_4x_provenance: BranchProvenanceSpec,
) -> MultiscaleBag:
    """Combine features and build a row-aligned provenance table atomically."""

    combined = concatenate_feature_matrices(scale_2x_features, scale_4x_features)
    records = build_two_scale_provenance(
        scale_2x=scale_2x_provenance,
        scale_4x=scale_4x_provenance,
        scale_2x_count=scale_2x_features.shape[0],
        scale_4x_count=scale_4x_features.shape[0],
    )
    validate_provenance_alignment(combined.shape[0], records)
    return MultiscaleBag(features=combined, provenance=records)
