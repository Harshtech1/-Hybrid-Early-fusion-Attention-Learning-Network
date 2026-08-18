from __future__ import annotations

from dataclasses import fields, replace

import pytest
import torch

from multiscale_feature_pilot.src.supervisor_tensor import (
    CrossPatientCombinationError,
    DuplicateIdentityError,
    SupervisorTensorError,
    WSIIdentity,
    build_batch_size_one,
    build_natural_healnet_wsi_input,
    describe_ragged_cohort,
    validate_patient_feature_tensor,
)


def _features(patch_count: int, *, value: float = 1.0) -> torch.Tensor:
    return torch.full((patch_count, 2048), value, dtype=torch.float32)


def _input(patient_id: str, wsi_id: str, patch_count: int):
    return build_natural_healnet_wsi_input(
        _features(patch_count),
        patient_id=patient_id,
        wsi_id=wsi_id,
    )


def test_natural_contract_adds_only_batch_axis() -> None:
    source = torch.arange(3 * 2048, dtype=torch.float32).reshape(3, 2048)

    result = build_natural_healnet_wsi_input(
        source,
        patient_id="TCGA-XX-0001",
        wsi_id="slide-0001",
    )

    assert result.tensor.shape == (1, 3, 2048)
    assert result.tensor.is_contiguous()
    assert result.tensor.data_ptr() == source.data_ptr()
    assert torch.equal(result.tensor[0], source)
    assert result.valid_mask is None
    assert result.validation.shape == (3, 2048)
    assert result.validation.patch_count == 3
    assert result.validation.feature_dim == 2048
    assert result.validation.dtype is torch.float32
    assert result.validation.device.type == "cpu"
    assert result.validation.contiguous
    assert result.validation.finite
    assert result.provenance.identity == WSIIdentity("TCGA-XX-0001", "slide-0001")
    assert result.provenance.source_shape == (3, 2048)
    assert result.provenance.model_input_shape == (1, 3, 2048)
    assert result.provenance.source_layout == "[P_i,D]"
    assert result.provenance.model_input_layout == "[B,P_i,D]"
    assert result.provenance.batch_size == 1
    assert not result.provenance.padding_applied
    assert not result.provenance.mask_present


def test_batch_size_one_accepts_exactly_one_identity_bound_bag() -> None:
    source = _features(2)
    identity = WSIIdentity(patient_id="patient-a", wsi_id="wsi-a")

    result = build_batch_size_one([(source, identity)])

    assert result.tensor.shape == (1, 2, 2048)
    assert result.provenance.identity == identity


@pytest.mark.parametrize(
    ("features", "message"),
    [
        (torch.empty((0, 2048), dtype=torch.float32), "at least one patch"),
        (torch.empty((2, 1024), dtype=torch.float32), "feature width D=2048"),
        (torch.empty((2, 2048, 1), dtype=torch.float32), r"shape \[P_i,2048\]"),
        (torch.empty((2, 2048), dtype=torch.float64), "dtype torch.float32"),
    ],
)
def test_rejects_empty_wrong_shape_and_wrong_dtype(
    features: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(SupervisorTensorError, match=message):
        validate_patient_feature_tensor(features)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_rejects_nonfinite_features(bad_value: float) -> None:
    features = _features(2)
    features[0, 0] = bad_value

    with pytest.raises(SupervisorTensorError, match="only finite"):
        build_natural_healnet_wsi_input(
            features,
            patient_id="patient-a",
            wsi_id="wsi-a",
        )


def test_rejects_noncontiguous_features_without_copying() -> None:
    features = torch.zeros((2048, 3), dtype=torch.float32).transpose(0, 1)
    assert features.shape == (3, 2048)
    assert not features.is_contiguous()

    with pytest.raises(SupervisorTensorError, match="must be contiguous"):
        build_natural_healnet_wsi_input(
            features,
            patient_id="patient-a",
            wsi_id="wsi-a",
        )


def test_rejects_non_cpu_tensor_during_phase_one() -> None:
    features = torch.empty((2, 2048), dtype=torch.float32, device="meta")

    with pytest.raises(SupervisorTensorError, match="must be on CPU during Phase 1"):
        validate_patient_feature_tensor(features)


@pytest.mark.parametrize(
    ("patient_id", "wsi_id", "message"),
    [
        ("", "wsi-a", "patient_id must not be empty"),
        ("   ", "wsi-a", "surrounding whitespace"),
        ("patient-a", "", "wsi_id must not be empty"),
        ("patient-a", " wsi-a", "surrounding whitespace"),
    ],
)
def test_rejects_invalid_identifiers(
    patient_id: str,
    wsi_id: str,
    message: str,
) -> None:
    with pytest.raises(SupervisorTensorError, match=message):
        build_natural_healnet_wsi_input(
            _features(1),
            patient_id=patient_id,
            wsi_id=wsi_id,
        )


def test_rejects_attempt_to_combine_different_patients() -> None:
    samples = [
        (_features(2), WSIIdentity("patient-a", "wsi-a")),
        (_features(3), WSIIdentity("patient-b", "wsi-b")),
    ]

    with pytest.raises(CrossPatientCombinationError, match="different patients"):
        build_batch_size_one(samples)


def test_rejects_duplicate_identity_before_tensor_combination() -> None:
    identity = WSIIdentity("patient-a", "wsi-a")

    with pytest.raises(DuplicateIdentityError, match="duplicate patient/WSI"):
        build_batch_size_one([(_features(2), identity), (_features(2), identity)])


def test_rejects_multiple_ws_is_even_for_one_patient() -> None:
    samples = [
        (_features(2), WSIIdentity("patient-a", "wsi-a")),
        (_features(3), WSIIdentity("patient-a", "wsi-b")),
    ]

    with pytest.raises(SupervisorTensorError, match="batch size is fixed at 1"):
        build_batch_size_one(samples)


def test_ragged_cohort_is_metadata_only_and_preserves_boundaries() -> None:
    first = _input("patient-a", "wsi-a", 3)
    second = _input("patient-b", "wsi-b", 5)

    cohort = describe_ragged_cohort([first, second])

    assert cohort.s == 2
    assert cohort.feature_dim == 2048
    assert cohort.patch_counts == (3, 5)
    assert cohort.total_patch_count == 8
    assert not cohort.tensors_concatenated
    assert [entry.identity.patient_id for entry in cohort.entries] == [
        "patient-a",
        "patient-b",
    ]
    assert [entry.source_shape for entry in cohort.entries] == [
        (3, 2048),
        (5, 2048),
    ]
    assert not any(
        isinstance(getattr(cohort, field.name), torch.Tensor)
        for field in fields(cohort)
    )


def test_ragged_cohort_rejects_duplicate_patient_wsi_identity() -> None:
    first = _input("patient-a", "wsi-a", 2)
    duplicate = _input("patient-a", "wsi-a", 4)

    with pytest.raises(DuplicateIdentityError, match="duplicate patient/WSI"):
        describe_ragged_cohort([first, duplicate])


def test_ragged_cohort_rejects_one_wsi_assigned_to_two_patients() -> None:
    first = _input("patient-a", "wsi-a", 2)
    collision = _input("patient-b", "wsi-a", 4)

    with pytest.raises(DuplicateIdentityError, match="assigned to both"):
        describe_ragged_cohort([first, collision])


def test_ragged_cohort_rejects_multiple_ws_is_for_one_patient() -> None:
    first = _input("patient-a", "wsi-a", 2)
    second = _input("patient-a", "wsi-b", 4)

    with pytest.raises(DuplicateIdentityError, match="multiple WSI bags"):
        describe_ragged_cohort([first, second])


def test_ragged_cohort_rejects_forged_inconsistent_provenance() -> None:
    valid = _input("patient-a", "wsi-a", 2)
    forged = replace(
        valid,
        provenance=replace(valid.provenance, model_input_shape=(1, 99, 2048)),
    )

    with pytest.raises(SupervisorTensorError, match="input provenance is inconsistent"):
        describe_ragged_cohort([forged])
