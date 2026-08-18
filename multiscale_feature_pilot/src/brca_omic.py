"""CPU-only loader for the released clean TCGA-BRCA Omic archive.

The released file is a ZIP archive containing one CSV payload (despite the
member's ``.zip`` suffix).  This module streams that member without extracting
it and never opens or otherwise touches WSI data.
"""

from __future__ import annotations

import csv
import hashlib
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch


BRCA_ARCHIVE_MEMBER = "./tcga_brca_all_clean.csv.zip"
BRCA_RELEASE_ARCHIVE_SHA256 = (
    "4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8"
)
BRCA_METADATA_COLUMNS = (
    "",
    "case_id",
    "slide_id",
    "age",
    "site",
    "survival_months",
    "censorship",
    "is_female",
    "oncotree_code",
    "train",
)
BRCA_FEATURE_SUFFIXES = {
    "rna": "_rnaseq",
    "mutation": "_mut",
    "cnv": "_cnv",
}
BRCA_EXPECTED_DIMS = {"rna": 1558, "mutation": 21, "cnv": 1333}


class BrcaOmicContractError(ValueError):
    """Raised when the BRCA archive, identity, or tensors violate contract."""


@dataclass(frozen=True)
class BrcaPatientOmics:
    """One exactly matched BRCA row with modalities kept separate."""

    source_row_index: str
    case_id: str
    slide_id: str
    rna: torch.Tensor
    mutation: torch.Tensor
    cnv: torch.Tensor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_requested_identity(case_id: str, slide_id: str) -> None:
    for name, value in (("case_id", case_id), ("slide_id", slide_id)):
        if not isinstance(value, str) or not value:
            raise BrcaOmicContractError(f"{name} must be a non-empty string")
        if value != value.strip():
            raise BrcaOmicContractError(
                f"{name} must not contain leading or trailing whitespace"
            )


def _validate_header(header: Sequence[str]) -> Mapping[str, list[int]]:
    metadata_width = len(BRCA_METADATA_COLUMNS)
    if tuple(header[:metadata_width]) != BRCA_METADATA_COLUMNS:
        raise BrcaOmicContractError(
            "BRCA metadata schema mismatch: expected leading columns "
            f"{BRCA_METADATA_COLUMNS!r}"
        )
    if len(header) != len(set(header)):
        raise BrcaOmicContractError("BRCA CSV header contains duplicate columns")

    groups = {
        name: [
            index
            for index, column in enumerate(header)
            if column.endswith(suffix)
        ]
        for name, suffix in BRCA_FEATURE_SUFFIXES.items()
    }
    for name, expected_width in BRCA_EXPECTED_DIMS.items():
        actual_width = len(groups[name])
        if actual_width != expected_width:
            raise BrcaOmicContractError(
                f"{name} width must be {expected_width}, received {actual_width}"
            )

    # This is the released CSV block order.  The tensors are returned in the
    # HEALNet input order independently; the source archive itself stores CNV
    # before mutation.
    ordered_feature_indices = [
        index
        for name in ("rna", "cnv", "mutation")
        for index in groups[name]
    ]
    expected_feature_indices = list(range(metadata_width, len(header)))
    if ordered_feature_indices != expected_feature_indices:
        raise BrcaOmicContractError(
            "BRCA feature columns must be complete, non-overlapping, and ordered "
            "as RNA, CNV, then mutation"
        )

    expected_total = metadata_width + sum(BRCA_EXPECTED_DIMS.values())
    if len(header) != expected_total:
        raise BrcaOmicContractError(
            f"BRCA CSV must contain {expected_total} columns, received {len(header)}"
        )
    return groups


def _tensor_from_row(
    row: Sequence[str], indices: Sequence[int], *, name: str
) -> torch.Tensor:
    try:
        values = [float(row[index]) for index in indices]
    except (IndexError, TypeError, ValueError) as exc:
        raise BrcaOmicContractError(
            f"{name} contains missing or non-numeric values"
        ) from exc

    tensor = torch.tensor(values, dtype=torch.float32).reshape(1, 1, -1).contiguous()
    if tensor.dtype is not torch.float32 or not tensor.is_contiguous():
        raise BrcaOmicContractError(f"{name} could not be represented as contiguous float32")
    if not bool(torch.isfinite(tensor).all().item()):
        raise BrcaOmicContractError(f"{name} contains NaN or Inf")
    return tensor


def load_brca_patient_omics(
    archive_path: str | Path,
    *,
    case_id: str,
    slide_id: str,
    expected_archive_sha256: str | None = None,
) -> BrcaPatientOmics:
    """Stream and load exactly one ``case_id`` + ``slide_id`` BRCA row.

    The clean-release schema and all three fixed feature widths are mandatory.
    Matching is literal and never falls back to case-only matching or row
    position.  Pass the provenance SHA256 as ``expected_archive_sha256`` when
    loading the official source-of-truth artifact.
    """

    _validate_requested_identity(case_id, slide_id)
    path = Path(archive_path)
    if not path.is_file():
        raise BrcaOmicContractError(f"BRCA Omic archive does not exist: {path}")

    if expected_archive_sha256 is not None:
        if not isinstance(expected_archive_sha256, str):
            raise BrcaOmicContractError(
                "expected_archive_sha256 must be a 64-character hexadecimal digest"
            )
        expected_digest = expected_archive_sha256.lower()
        if len(expected_digest) != 64 or any(
            character not in "0123456789abcdef" for character in expected_digest
        ):
            raise BrcaOmicContractError(
                "expected_archive_sha256 must be a 64-character hexadecimal digest"
            )
        actual_digest = _sha256(path)
        if actual_digest != expected_digest:
            raise BrcaOmicContractError(
                "BRCA Omic archive SHA256 mismatch: "
                f"expected {expected_digest}, received {actual_digest}"
            )

    try:
        with zipfile.ZipFile(path, mode="r") as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].is_dir():
                raise BrcaOmicContractError(
                    "BRCA Omic archive must contain exactly one file member"
                )
            member = members[0]
            if member.filename != BRCA_ARCHIVE_MEMBER:
                raise BrcaOmicContractError(
                    "unexpected BRCA Omic archive member: "
                    f"expected {BRCA_ARCHIVE_MEMBER!r}, received {member.filename!r}"
                )
            if member.flag_bits & 0x1:
                raise BrcaOmicContractError("encrypted BRCA Omic archives are unsupported")

            with archive.open(member, mode="r") as raw_stream:
                with io.TextIOWrapper(
                    raw_stream, encoding="utf-8-sig", newline=""
                ) as text_stream:
                    reader = csv.reader(text_stream, strict=True)
                    try:
                        header = next(reader)
                    except StopIteration as exc:
                        raise BrcaOmicContractError("BRCA Omic CSV is empty") from exc

                    groups = _validate_header(header)
                    case_index = header.index("case_id")
                    slide_index = header.index("slide_id")
                    matched_row: list[str] | None = None
                    match_count = 0

                    for line_number, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise BrcaOmicContractError(
                                f"BRCA CSV row {line_number} has {len(row)} columns; "
                                f"expected {len(header)}"
                            )
                        if row[case_index] == case_id and row[slide_index] == slide_id:
                            match_count += 1
                            if matched_row is None:
                                matched_row = row

    except BrcaOmicContractError:
        raise
    except (OSError, UnicodeError, csv.Error, zipfile.BadZipFile) as exc:
        raise BrcaOmicContractError(
            f"could not read BRCA Omic archive {path}: {exc}"
        ) from exc

    if match_count != 1 or matched_row is None:
        raise BrcaOmicContractError(
            "expected exactly one exact case_id/slide_id BRCA row, "
            f"found {match_count} for case_id={case_id!r}, slide_id={slide_id!r}"
        )

    tensors = {
        name: _tensor_from_row(matched_row, indices, name=name)
        for name, indices in groups.items()
    }
    for name, expected_width in BRCA_EXPECTED_DIMS.items():
        expected_shape = (1, 1, expected_width)
        if tuple(tensors[name].shape) != expected_shape:
            raise BrcaOmicContractError(
                f"{name} must have shape {expected_shape}, "
                f"received {tuple(tensors[name].shape)}"
            )

    return BrcaPatientOmics(
        source_row_index=matched_row[0],
        case_id=matched_row[case_index],
        slide_id=matched_row[slide_index],
        rna=tensors["rna"],
        mutation=tensors["mutation"],
        cnv=tensors["cnv"],
    )


def load_official_brca_patient_omics(
    archive_path: str | Path, *, case_id: str, slide_id: str
) -> BrcaPatientOmics:
    """Load a patient only when the archive matches the frozen release SHA256."""

    return load_brca_patient_omics(
        archive_path,
        case_id=case_id,
        slide_id=slide_id,
        expected_archive_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
    )
