"""Strict patient-matched Omic loading for the one-patient pilot."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


BLCA_PILOT_CASE_ID = "TCGA-2F-A9KT"
BLCA_PILOT_SLIDE_ID = (
    "TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs"
)
BLCA_PILOT_DIMS = {"rna": 1523, "mutation": 1125, "cnv": 193}


class OmicContractError(ValueError):
    """Raised when patient identity or Omic tensor validation fails."""


@dataclass(frozen=True)
class PatientOmicModalities:
    case_id: str
    sample_id: str
    slide_id: str
    rna: torch.Tensor
    mutation: torch.Tensor
    cnv: torch.Tensor


def _unique_column_index(header: Sequence[str], column: str) -> int:
    matches = [index for index, name in enumerate(header) if name == column]
    if len(matches) != 1:
        raise OmicContractError(
            f"expected exactly one {column!r} column, found {len(matches)}"
        )
    return matches[0]


def _feature_indices(header: Sequence[str], suffix: str) -> list[int]:
    indices = [index for index, name in enumerate(header) if name.endswith(suffix)]
    if not indices:
        raise OmicContractError(f"no feature columns found with suffix {suffix!r}")
    return indices


def _tensor_from_row(
    row: Sequence[str],
    indices: Sequence[int],
    *,
    name: str,
) -> torch.Tensor:
    try:
        values = [float(row[index]) for index in indices]
    except (IndexError, TypeError, ValueError) as exc:
        raise OmicContractError(f"{name} contains missing or non-numeric values") from exc

    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1)
    if not bool(torch.isfinite(tensor).all()):
        raise OmicContractError(f"{name} contains NaN or Inf")
    return tensor


def load_patient_omic_modalities(
    csv_path: str | Path,
    *,
    case_id: str,
    slide_id: str,
    expected_dims: Mapping[str, int] | None = None,
) -> PatientOmicModalities:
    """Load exactly one case/slide row and keep RNA, mutation, and CNV separate.

    Feature order is the original CSV header order. Matching requires both
    ``_PATIENT`` and ``slide_id``; row order is never used for identity.
    """

    path = Path(csv_path)
    if not path.is_file():
        raise OmicContractError(f"Omic CSV does not exist: {path}")

    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.reader(stream)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise OmicContractError("Omic CSV is empty") from exc

        patient_index = _unique_column_index(header, "_PATIENT")
        sample_index = _unique_column_index(header, "sample")
        slide_index = _unique_column_index(header, "slide_id")
        group_indices = {
            "rna": _feature_indices(header, "_rnaseq"),
            "mutation": _feature_indices(header, "_mut"),
            "cnv": _feature_indices(header, "_cnv"),
        }

        matches = [
            row
            for row in reader
            if len(row) == len(header)
            and row[patient_index] == case_id
            and row[slide_index] == slide_id
        ]

    if len(matches) != 1:
        raise OmicContractError(
            "expected exactly one exact case/slide Omic row, "
            f"found {len(matches)} for case={case_id!r}, slide={slide_id!r}"
        )

    row = matches[0]
    tensors = {
        name: _tensor_from_row(row, indices, name=name)
        for name, indices in group_indices.items()
    }

    if expected_dims is not None:
        for name, expected in expected_dims.items():
            if name not in tensors:
                raise OmicContractError(f"unknown expected modality {name!r}")
            actual = tensors[name].shape[-1]
            if actual != expected:
                raise OmicContractError(
                    f"{name} width must be {expected}, received {actual}"
                )

    return PatientOmicModalities(
        case_id=row[patient_index],
        sample_id=row[sample_index],
        slide_id=row[slide_index],
        rna=tensors["rna"],
        mutation=tensors["mutation"],
        cnv=tensors["cnv"],
    )


def load_blca_pilot_omics(csv_path: str | Path) -> PatientOmicModalities:
    """Load the fixed, checksum-gated BLCA one-patient pilot row."""

    return load_patient_omic_modalities(
        csv_path,
        case_id=BLCA_PILOT_CASE_ID,
        slide_id=BLCA_PILOT_SLIDE_ID,
        expected_dims=BLCA_PILOT_DIMS,
    )
