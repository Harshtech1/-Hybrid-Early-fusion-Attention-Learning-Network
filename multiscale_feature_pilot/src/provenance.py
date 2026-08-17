"""Row-level provenance for deterministic multiscale feature bags."""

from __future__ import annotations

import csv
import math
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterable, Sequence

import torch


class ProvenanceError(ValueError):
    """Raised when patch provenance cannot match feature rows exactly."""


@dataclass(frozen=True)
class BranchProvenanceSpec:
    """Metadata and ordered coordinates for one WSI scale branch."""

    branch: str
    coordinates: Sequence[Sequence[int]] | torch.Tensor
    level: int
    mpp_x: float
    mpp_y: float


@dataclass(frozen=True)
class PatchProvenance:
    """The required one-to-one mapping from a combined row to its patch."""

    global_row_index: int
    branch: str
    local_patch_index: int
    x: int
    y: int
    level: int
    mpp_x: float
    mpp_y: float


PROVENANCE_FIELDS = tuple(PatchProvenance.__dataclass_fields__)
_INTEGER_DTYPES = {
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint8,
}


def _validate_branch_spec(spec: BranchProvenanceSpec) -> None:
    if not spec.branch or not spec.branch.strip():
        raise ProvenanceError("branch must be a non-empty string")
    if isinstance(spec.level, bool) or not isinstance(spec.level, Integral) or spec.level < 0:
        raise ProvenanceError(f"{spec.branch}: level must be a non-negative integer")
    for name, value in (("mpp_x", spec.mpp_x), ("mpp_y", spec.mpp_y)):
        if not math.isfinite(float(value)) or float(value) <= 0:
            raise ProvenanceError(f"{spec.branch}: {name} must be finite and positive")


def _normalise_coordinates(
    coordinates: Sequence[Sequence[int]] | torch.Tensor,
    *,
    branch: str,
) -> list[tuple[int, int]]:
    if isinstance(coordinates, torch.Tensor):
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ProvenanceError(
                f"{branch}: coordinates must have shape [N,2], got {tuple(coordinates.shape)}"
            )
        if coordinates.dtype not in _INTEGER_DTYPES:
            raise ProvenanceError(
                f"{branch}: coordinate dtype must be integral, got {coordinates.dtype}"
            )
        coordinate_rows = coordinates.detach().cpu().tolist()
    else:
        coordinate_rows = list(coordinates)

    result: list[tuple[int, int]] = []
    for index, row in enumerate(coordinate_rows):
        if len(row) != 2:
            raise ProvenanceError(f"{branch}: coordinate row {index} does not contain x and y")
        x, y = row
        if any(isinstance(value, bool) or not isinstance(value, Integral) for value in (x, y)):
            raise ProvenanceError(f"{branch}: coordinate row {index} must contain integers")
        result.append((int(x), int(y)))
    return result


def _build_branch_records(
    spec: BranchProvenanceSpec,
    *,
    expected_count: int,
    global_offset: int,
) -> tuple[PatchProvenance, ...]:
    _validate_branch_spec(spec)
    coordinates = _normalise_coordinates(spec.coordinates, branch=spec.branch)
    if len(coordinates) != expected_count:
        raise ProvenanceError(
            f"{spec.branch}: coordinate count {len(coordinates)} does not match "
            f"feature-row count {expected_count}"
        )

    return tuple(
        PatchProvenance(
            global_row_index=global_offset + local_index,
            branch=spec.branch,
            local_patch_index=local_index,
            x=x,
            y=y,
            level=int(spec.level),
            mpp_x=float(spec.mpp_x),
            mpp_y=float(spec.mpp_y),
        )
        for local_index, (x, y) in enumerate(coordinates)
    )


def build_two_scale_provenance(
    *,
    scale_2x: BranchProvenanceSpec,
    scale_4x: BranchProvenanceSpec,
    scale_2x_count: int,
    scale_4x_count: int,
) -> tuple[PatchProvenance, ...]:
    """Build records in the fixed order: all scale-2x rows, then scale-4x rows."""

    if scale_2x_count < 0 or scale_4x_count < 0:
        raise ProvenanceError("feature-row counts must be non-negative")
    if scale_2x.branch != "scale_2x" or scale_4x.branch != "scale_4x":
        raise ProvenanceError(
            "two-scale provenance requires branch labels 'scale_2x' then 'scale_4x'"
        )

    first = _build_branch_records(
        scale_2x,
        expected_count=scale_2x_count,
        global_offset=0,
    )
    second = _build_branch_records(
        scale_4x,
        expected_count=scale_4x_count,
        global_offset=scale_2x_count,
    )
    return first + second


def validate_provenance_alignment(
    feature_row_count: int,
    records: Sequence[PatchProvenance],
) -> None:
    """Require exactly one contiguous provenance record for every feature row."""

    if len(records) != feature_row_count:
        raise ProvenanceError(
            f"provenance rows {len(records)} do not match feature rows {feature_row_count}"
        )
    global_indices = [record.global_row_index for record in records]
    expected = list(range(feature_row_count))
    if global_indices != expected:
        raise ProvenanceError("global_row_index must be contiguous and match combined row order")

    branch_local_indices: dict[str, list[int]] = {}
    for record in records:
        branch_local_indices.setdefault(record.branch, []).append(record.local_patch_index)
    for branch, local_indices in branch_local_indices.items():
        if local_indices != list(range(len(local_indices))):
            raise ProvenanceError(f"{branch}: local_patch_index must start at zero and be contiguous")


def provenance_as_dicts(records: Iterable[PatchProvenance]) -> list[dict[str, object]]:
    """Return serialization-ready table rows without changing their order."""

    return [asdict(record) for record in records]


def write_provenance_csv(records: Sequence[PatchProvenance], path: str | Path) -> Path:
    """Write the exact row-to-patch mapping as a deterministic CSV table."""

    validate_provenance_alignment(len(records), records)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROVENANCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(provenance_as_dicts(records))
    return output_path
