"""Supervisor-aligned, patient-preserving WSI tensor contract.

This module intentionally implements only the Phase-1, batch-size-one path.
One WSI remains one patch bag with shape ``[P_i, D]`` where ``D=2048``;
the natural HEALNet input adds only a batch axis, yielding
``[1, P_i, 2048]``.  It does not pool patches, pad bags, create a mask, or
concatenate tensors belonging to different patients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch


FEATURE_DIM = 2048


class SupervisorTensorError(ValueError):
    """Raised when a tensor or identity violates the supervisor contract."""


class DuplicateIdentityError(SupervisorTensorError):
    """Raised when a cohort description repeats a WSI identity."""


class CrossPatientCombinationError(SupervisorTensorError):
    """Raised when different patients are offered to one tensor operation."""


@dataclass(frozen=True, slots=True)
class WSIIdentity:
    """Stable patient and slide identifiers for exactly one WSI patch bag."""

    patient_id: str
    wsi_id: str

    def __post_init__(self) -> None:
        _validate_identifier(self.patient_id, name="patient_id")
        _validate_identifier(self.wsi_id, name="wsi_id")


@dataclass(frozen=True, slots=True)
class TensorValidation:
    """Recorded facts established while validating the source feature bag."""

    shape: tuple[int, int]
    patch_count: int
    feature_dim: int
    dtype: torch.dtype
    device: torch.device
    contiguous: bool
    finite: bool


@dataclass(frozen=True, slots=True)
class WSITensorProvenance:
    """Identity and shape lineage for one natural-orientation model input."""

    identity: WSIIdentity
    source_shape: tuple[int, int]
    model_input_shape: tuple[int, int, int]
    source_layout: str
    model_input_layout: str
    batch_size: int
    padding_applied: bool
    mask_present: bool


@dataclass(frozen=True, slots=True)
class NaturalHealNetWSIInput:
    """Validated batch-size-one WSI input in natural patch-token orientation."""

    tensor: torch.Tensor
    validation: TensorValidation
    provenance: WSITensorProvenance
    valid_mask: None = None


@dataclass(frozen=True, slots=True)
class RaggedWSIEntry:
    """Metadata-only description of one member of a ragged cohort."""

    identity: WSIIdentity
    patch_count: int
    feature_dim: int
    source_shape: tuple[int, int]


@dataclass(frozen=True, slots=True)
class RaggedCohortDescription:
    """Metadata-only cohort summary; no patient tensors are joined or stored."""

    s: int
    feature_dim: int
    entries: tuple[RaggedWSIEntry, ...]
    patch_counts: tuple[int, ...]
    total_patch_count: int
    tensors_concatenated: bool = False


def _validate_identifier(value: object, *, name: str) -> None:
    if not isinstance(value, str):
        raise SupervisorTensorError(f"{name} must be a string")
    if not value:
        raise SupervisorTensorError(f"{name} must not be empty")
    if value != value.strip():
        raise SupervisorTensorError(f"{name} must not contain surrounding whitespace")
    if not value.strip():
        raise SupervisorTensorError(f"{name} must not be blank")


def validate_patient_feature_tensor(
    features: torch.Tensor,
    *,
    name: str = "patient_features",
) -> TensorValidation:
    """Validate one finite, dense, contiguous CPU ``[P_i,2048]`` tensor.

    Validation is deliberately non-coercing: no dtype conversion, device
    transfer, reshape, or contiguous copy is performed.  This keeps failures
    visible and makes the provenance describe the exact supplied artifact.
    """

    if not isinstance(features, torch.Tensor):
        raise SupervisorTensorError(
            f"{name} must be a torch.Tensor, got {type(features).__name__}"
        )
    if features.layout is not torch.strided:
        raise SupervisorTensorError(f"{name} must be a dense strided tensor")
    if features.ndim != 2:
        raise SupervisorTensorError(
            f"{name} must have shape [P_i,{FEATURE_DIM}], got {tuple(features.shape)}"
        )
    if features.shape[0] == 0:
        raise SupervisorTensorError(f"{name} must contain at least one patch row")
    if features.shape[1] != FEATURE_DIM:
        raise SupervisorTensorError(
            f"{name} must have feature width D={FEATURE_DIM}, got {features.shape[1]}"
        )
    if features.dtype is not torch.float32:
        raise SupervisorTensorError(
            f"{name} must have dtype torch.float32, got {features.dtype}"
        )
    if features.device.type != "cpu":
        raise SupervisorTensorError(
            f"{name} must be on CPU during Phase 1, got device {features.device}"
        )
    if not features.is_contiguous():
        raise SupervisorTensorError(f"{name} must be contiguous")
    if not bool(torch.isfinite(features).all().item()):
        raise SupervisorTensorError(f"{name} must contain only finite values")

    patch_count = int(features.shape[0])
    return TensorValidation(
        shape=(patch_count, FEATURE_DIM),
        patch_count=patch_count,
        feature_dim=FEATURE_DIM,
        dtype=features.dtype,
        device=features.device,
        contiguous=True,
        finite=True,
    )


def build_natural_healnet_wsi_input(
    features: torch.Tensor,
    *,
    patient_id: str,
    wsi_id: str,
) -> NaturalHealNetWSIInput:
    """Add only ``B=1`` to one patient's bag: ``[P_i,D] -> [1,P_i,D]``."""

    identity = WSIIdentity(patient_id=patient_id, wsi_id=wsi_id)
    validation = validate_patient_feature_tensor(features)
    model_input = features.unsqueeze(0)
    expected_shape = (1, validation.patch_count, FEATURE_DIM)
    if tuple(model_input.shape) != expected_shape or not model_input.is_contiguous():
        raise SupervisorTensorError(
            f"natural HEALNet input must be contiguous with shape {expected_shape}"
        )

    provenance = WSITensorProvenance(
        identity=identity,
        source_shape=validation.shape,
        model_input_shape=expected_shape,
        source_layout="[P_i,D]",
        model_input_layout="[B,P_i,D]",
        batch_size=1,
        padding_applied=False,
        mask_present=False,
    )
    return NaturalHealNetWSIInput(
        tensor=model_input,
        validation=validation,
        provenance=provenance,
    )


def build_batch_size_one(
    samples: Sequence[tuple[torch.Tensor, WSIIdentity]],
) -> NaturalHealNetWSIInput:
    """Build the only supported batch and reject all multi-WSI combination.

    This explicit sequence entry point makes an accidental call with two
    patient bags fail before any stacking or concatenation can occur.
    """

    sample_tuple = tuple(samples)
    if not sample_tuple:
        raise SupervisorTensorError("batch-size-one input requires exactly one WSI")

    identities: list[WSIIdentity] = []
    for index, sample in enumerate(sample_tuple):
        if not isinstance(sample, tuple) or len(sample) != 2:
            raise SupervisorTensorError(
                f"sample {index} must be a (features, WSIIdentity) tuple"
            )
        _, identity = sample
        if not isinstance(identity, WSIIdentity):
            raise SupervisorTensorError(f"sample {index} identity must be WSIIdentity")
        identities.append(identity)

    identity_keys = [(item.patient_id, item.wsi_id) for item in identities]
    if len(set(identity_keys)) != len(identity_keys):
        raise DuplicateIdentityError("duplicate patient/WSI identity is not allowed")

    patient_ids = {item.patient_id for item in identities}
    if len(patient_ids) > 1:
        raise CrossPatientCombinationError(
            "cannot combine WSIs from different patients into one tensor"
        )
    if len(sample_tuple) != 1:
        raise SupervisorTensorError(
            "batch size is fixed at 1; combining multiple WSIs is not supported"
        )

    features, identity = sample_tuple[0]
    return build_natural_healnet_wsi_input(
        features,
        patient_id=identity.patient_id,
        wsi_id=identity.wsi_id,
    )


def describe_ragged_cohort(
    inputs: Sequence[NaturalHealNetWSIInput],
) -> RaggedCohortDescription:
    """Describe ``S`` independent WSI bags without stacking or concatenation."""

    input_tuple = tuple(inputs)
    if not input_tuple:
        raise SupervisorTensorError("ragged cohort must contain at least one WSI")

    entries: list[RaggedWSIEntry] = []
    seen_identities: set[tuple[str, str]] = set()
    seen_patient_ids: set[str] = set()
    seen_wsi_ids: dict[str, str] = {}

    for index, item in enumerate(input_tuple):
        if not isinstance(item, NaturalHealNetWSIInput):
            raise SupervisorTensorError(
                f"cohort item {index} must be NaturalHealNetWSIInput"
            )
        _validate_natural_input(item, index=index)
        identity = item.provenance.identity
        identity_key = (identity.patient_id, identity.wsi_id)
        if identity_key in seen_identities:
            raise DuplicateIdentityError(
                f"duplicate patient/WSI identity: {identity.patient_id}/{identity.wsi_id}"
            )
        if identity.patient_id in seen_patient_ids:
            raise DuplicateIdentityError(
                f"multiple WSI bags for patient {identity.patient_id} are excluded"
            )
        if identity.wsi_id in seen_wsi_ids:
            first_patient = seen_wsi_ids[identity.wsi_id]
            raise DuplicateIdentityError(
                f"WSI identity {identity.wsi_id} is assigned to both "
                f"{first_patient} and {identity.patient_id}"
            )
        seen_identities.add(identity_key)
        seen_patient_ids.add(identity.patient_id)
        seen_wsi_ids[identity.wsi_id] = identity.patient_id
        entries.append(
            RaggedWSIEntry(
                identity=identity,
                patch_count=item.validation.patch_count,
                feature_dim=FEATURE_DIM,
                source_shape=item.validation.shape,
            )
        )

    patch_counts = tuple(entry.patch_count for entry in entries)
    return RaggedCohortDescription(
        s=len(entries),
        feature_dim=FEATURE_DIM,
        entries=tuple(entries),
        patch_counts=patch_counts,
        total_patch_count=sum(patch_counts),
    )


def _validate_natural_input(item: NaturalHealNetWSIInput, *, index: int) -> None:
    """Defend the cohort helper against manually forged result dataclasses."""

    if not isinstance(item.tensor, torch.Tensor):
        raise SupervisorTensorError(f"cohort item {index} tensor must be a torch.Tensor")
    source = (
        item.tensor.squeeze(0)
        if item.tensor.ndim == 3 and item.tensor.shape[0] == 1
        else item.tensor
    )
    validation = validate_patient_feature_tensor(
        source,
        name=f"cohort item {index}",
    )
    expected_shape = (1, validation.patch_count, FEATURE_DIM)
    if tuple(item.tensor.shape) != expected_shape:
        raise SupervisorTensorError(
            f"cohort item {index} must have shape {expected_shape}, got {tuple(item.tensor.shape)}"
        )
    if item.validation != validation:
        raise SupervisorTensorError(f"cohort item {index} validation metadata is inconsistent")
    provenance = item.provenance
    if provenance.source_shape != validation.shape:
        raise SupervisorTensorError(f"cohort item {index} source provenance is inconsistent")
    if provenance.model_input_shape != expected_shape:
        raise SupervisorTensorError(f"cohort item {index} input provenance is inconsistent")
    if provenance.source_layout != "[P_i,D]":
        raise SupervisorTensorError(f"cohort item {index} source layout is inconsistent")
    if provenance.model_input_layout != "[B,P_i,D]":
        raise SupervisorTensorError(f"cohort item {index} model layout is inconsistent")
    if provenance.batch_size != 1 or provenance.padding_applied or provenance.mask_present:
        raise SupervisorTensorError(f"cohort item {index} violates the unpadded B=1 contract")
    if item.valid_mask is not None:
        raise SupervisorTensorError(f"cohort item {index} must not contain a padding mask")


__all__ = [
    "FEATURE_DIM",
    "CrossPatientCombinationError",
    "DuplicateIdentityError",
    "NaturalHealNetWSIInput",
    "RaggedCohortDescription",
    "RaggedWSIEntry",
    "SupervisorTensorError",
    "TensorValidation",
    "WSIIdentity",
    "WSITensorProvenance",
    "build_batch_size_one",
    "build_natural_healnet_wsi_input",
    "describe_ragged_cohort",
    "validate_patient_feature_tensor",
]
