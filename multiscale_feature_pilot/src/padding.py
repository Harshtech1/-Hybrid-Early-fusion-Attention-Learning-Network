"""Patient-local orientation and padding for the released HEALNet WSI path."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .multiscale_bag import FEATURE_DIM, FeatureMatrixError, validate_feature_matrix


@dataclass(frozen=True)
class PaddedWSIBatch:
    """A fixed-width released-orientation batch plus patch-slot validity."""

    features: torch.Tensor
    valid_mask: torch.Tensor
    lengths: torch.Tensor
    pad_value: float


def to_released_healnet_orientation(patient_features: torch.Tensor) -> torch.Tensor:
    """Convert one patient from `[P,2048]` to contiguous `[2048,P]`."""

    validate_feature_matrix(patient_features, name="patient_features")
    return patient_features.transpose(0, 1).contiguous()


def pad_patient_bags(
    patient_feature_bags: Sequence[torch.Tensor],
    *,
    pad_value: float = 0.0,
) -> PaddedWSIBatch:
    """Pad each patient independently and return `[B,2048,P_max]` plus `[B,P_max]`."""

    if not patient_feature_bags:
        raise FeatureMatrixError("at least one patient feature bag is required")

    validated = [
        validate_feature_matrix(bag, name=f"patient_{index}")
        for index, bag in enumerate(patient_feature_bags)
    ]
    first_device = validated[0].device
    if any(bag.device != first_device for bag in validated[1:]):
        raise FeatureMatrixError("all patient feature bags must be on the same device")

    lengths = torch.tensor(
        [bag.shape[0] for bag in validated],
        dtype=torch.int64,
        device=first_device,
    )
    batch_size = len(validated)
    max_patches = int(lengths.max().item())
    batch = torch.full(
        (batch_size, FEATURE_DIM, max_patches),
        fill_value=float(pad_value),
        dtype=torch.float32,
        device=first_device,
    )
    valid_mask = torch.zeros(
        (batch_size, max_patches),
        dtype=torch.bool,
        device=first_device,
    )

    for patient_index, bag in enumerate(validated):
        patch_count = bag.shape[0]
        batch[patient_index, :, :patch_count] = to_released_healnet_orientation(bag)
        valid_mask[patient_index, :patch_count] = True

    result = PaddedWSIBatch(
        features=batch,
        valid_mask=valid_mask,
        lengths=lengths,
        pad_value=float(pad_value),
    )
    validate_padded_batch(result)
    return result


def validate_padded_batch(batch: PaddedWSIBatch) -> None:
    """Validate dimensions and ensure every invalid slot still contains only padding."""

    if batch.features.ndim != 3 or batch.features.shape[1] != FEATURE_DIM:
        raise FeatureMatrixError(
            f"padded features must have shape [B,{FEATURE_DIM},P_max], "
            f"got {tuple(batch.features.shape)}"
        )
    expected_mask_shape = (batch.features.shape[0], batch.features.shape[2])
    if tuple(batch.valid_mask.shape) != expected_mask_shape:
        raise FeatureMatrixError(
            f"valid_mask must have shape {expected_mask_shape}, got {tuple(batch.valid_mask.shape)}"
        )
    if batch.valid_mask.dtype is not torch.bool:
        raise FeatureMatrixError(f"valid_mask must be bool, got {batch.valid_mask.dtype}")
    if tuple(batch.lengths.shape) != (batch.features.shape[0],):
        raise FeatureMatrixError("lengths must have shape [B]")
    if not torch.equal(batch.valid_mask.sum(dim=1).to(batch.lengths.dtype), batch.lengths):
        raise FeatureMatrixError("mask-valid counts do not equal patient lengths")

    invalid = (~batch.valid_mask).unsqueeze(1).expand_as(batch.features)
    invalid_values = batch.features.masked_select(invalid)
    if invalid_values.numel() and not bool(
        torch.all(invalid_values == batch.pad_value).item()
    ):
        raise FeatureMatrixError("invalid patch slots contain non-padding values")
