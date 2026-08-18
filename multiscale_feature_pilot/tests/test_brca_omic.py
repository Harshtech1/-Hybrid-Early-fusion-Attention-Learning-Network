from __future__ import annotations

import hashlib
import io
from pathlib import Path
import zipfile

import pytest
import torch

from multiscale_feature_pilot.src.brca_omic import (
    BRCA_ARCHIVE_MEMBER,
    BRCA_EXPECTED_DIMS,
    BRCA_METADATA_COLUMNS,
    BrcaOmicContractError,
    load_brca_patient_omics,
)


CASE_ID = "TCGA-AA-0001"
SLIDE_ID = "TCGA-AA-0001-01Z-00-DX1.TEST.svs"


def _header(*, rna_width: int = BRCA_EXPECTED_DIMS["rna"]) -> list[str]:
    return [
        *BRCA_METADATA_COLUMNS,
        *(f"RNA{i:04d}_rnaseq" for i in range(rna_width)),
        *(f"CNV{i:04d}_cnv" for i in range(BRCA_EXPECTED_DIMS["cnv"])),
        *(f"MUT{i:04d}_mut" for i in range(BRCA_EXPECTED_DIMS["mutation"])),
    ]


def _row(
    *,
    row_index: str = "7",
    case_id: str = CASE_ID,
    slide_id: str = SLIDE_ID,
    rna_width: int = BRCA_EXPECTED_DIMS["rna"],
) -> list[str]:
    rna = ["1.25"] * rna_width
    rna[0] = "-2.5"
    rna[-1] = "3.75"
    mutation = ["0"] * BRCA_EXPECTED_DIMS["mutation"]
    cnv = ["-1"] * BRCA_EXPECTED_DIMS["cnv"]
    return [
        row_index,
        case_id,
        slide_id,
        "50.0",
        "AA",
        "24.0",
        "1.0",
        "1.0",
        "IDC",
        "1.0",
        *rna,
        *cnv,
        *mutation,
    ]


def _csv_bytes(header: list[str], rows: list[list[str]]) -> bytes:
    stream = io.StringIO(newline="")
    # csv.writer is intentionally imported locally so the fixture mirrors the
    # parser's quoting rules without hand-assembling a 2,922-column line.
    import csv

    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_archive(
    path: Path,
    *,
    header: list[str] | None = None,
    rows: list[list[str]] | None = None,
    member_name: str = BRCA_ARCHIVE_MEMBER,
    extra_member: bool = False,
) -> None:
    payload = _csv_bytes(header or _header(), rows or [_row()])
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member_name, payload)
        if extra_member:
            archive.writestr("unexpected.txt", "not allowed")


def test_load_exact_case_and_slide_returns_separate_contiguous_tensors(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "tcga_brca_all_clean.csv.zip"
    other_slide = "TCGA-AA-0001-01Z-00-DX2.OTHER.svs"
    _write_archive(
        archive_path,
        rows=[
            _row(row_index="6", slide_id=other_slide),
            _row(),
            _row(row_index="8", case_id="TCGA-BB-0002"),
        ],
    )

    result = load_brca_patient_omics(
        archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
    )

    assert result.source_row_index == "7"
    assert result.case_id == CASE_ID
    assert result.slide_id == SLIDE_ID
    for name, expected_width in BRCA_EXPECTED_DIMS.items():
        tensor = getattr(result, name)
        assert tensor.shape == (1, 1, expected_width)
        assert tensor.dtype is torch.float32
        assert tensor.is_contiguous()
        assert bool(torch.isfinite(tensor).all().item())
    assert result.rna[0, 0, [0, -1]].tolist() == [-2.5, 3.75]
    assert torch.count_nonzero(result.mutation).item() == 0
    assert torch.unique(result.cnv).tolist() == [-1.0]


def test_match_never_falls_back_to_case_only_or_row_position(tmp_path: Path) -> None:
    archive_path = tmp_path / "omic.zip"
    _write_archive(archive_path, rows=[_row(slide_id="other.svs")])

    with pytest.raises(BrcaOmicContractError, match="found 0"):
        load_brca_patient_omics(
            archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
        )


def test_duplicate_exact_case_and_slide_rows_are_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "omic.zip"
    _write_archive(archive_path, rows=[_row(row_index="1"), _row(row_index="2")])

    with pytest.raises(BrcaOmicContractError, match="found 2"):
        load_brca_patient_omics(
            archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
        )


def test_schema_requires_released_suffix_widths(tmp_path: Path) -> None:
    archive_path = tmp_path / "omic.zip"
    wrong_rna_width = BRCA_EXPECTED_DIMS["rna"] - 1
    _write_archive(
        archive_path,
        header=_header(rna_width=wrong_rna_width),
        rows=[_row(rna_width=wrong_rna_width)],
    )

    with pytest.raises(BrcaOmicContractError, match="rna width must be 1558"):
        load_brca_patient_omics(
            archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
        )


def test_schema_requires_released_rna_cnv_mutation_block_order(tmp_path: Path) -> None:
    archive_path = tmp_path / "omic.zip"
    header = _header()
    rna_end = len(BRCA_METADATA_COLUMNS) + BRCA_EXPECTED_DIMS["rna"]
    cnv_end = rna_end + BRCA_EXPECTED_DIMS["cnv"]
    reordered_header = [
        *header[:rna_end],
        *header[cnv_end:],
        *header[rna_end:cnv_end],
    ]
    row = _row()
    reordered_row = [
        *row[:rna_end],
        *row[cnv_end:],
        *row[rna_end:cnv_end],
    ]
    _write_archive(archive_path, header=reordered_header, rows=[reordered_row])

    with pytest.raises(BrcaOmicContractError, match="RNA, CNV, then mutation"):
        load_brca_patient_omics(
            archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
        )


@pytest.mark.parametrize("bad_value", ["", "nan", "inf", "-inf", "not-a-number"])
def test_missing_non_numeric_or_nonfinite_values_are_rejected(
    tmp_path: Path, bad_value: str
) -> None:
    archive_path = tmp_path / "omic.zip"
    row = _row()
    row[len(BRCA_METADATA_COLUMNS)] = bad_value
    _write_archive(archive_path, rows=[row])

    with pytest.raises(BrcaOmicContractError, match="rna contains"):
        load_brca_patient_omics(
            archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
        )


def test_every_csv_row_must_have_exact_schema_width(tmp_path: Path) -> None:
    archive_path = tmp_path / "omic.zip"
    malformed_unmatched_row = _row(case_id="TCGA-ZZ-9999")[:-1]
    _write_archive(archive_path, rows=[_row(), malformed_unmatched_row])

    with pytest.raises(BrcaOmicContractError, match="row 3 has 2921 columns"):
        load_brca_patient_omics(
            archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
        )


@pytest.mark.parametrize(
    ("member_name", "extra_member", "message"),
    [
        ("wrong.csv", False, "unexpected BRCA Omic archive member"),
        (BRCA_ARCHIVE_MEMBER, True, "exactly one file member"),
    ],
)
def test_archive_container_shape_is_fail_closed(
    tmp_path: Path, member_name: str, extra_member: bool, message: str
) -> None:
    archive_path = tmp_path / "omic.zip"
    _write_archive(
        archive_path, member_name=member_name, extra_member=extra_member
    )

    with pytest.raises(BrcaOmicContractError, match=message):
        load_brca_patient_omics(
            archive_path, case_id=CASE_ID, slide_id=SLIDE_ID
        )


def test_optional_archive_sha256_gate(tmp_path: Path) -> None:
    archive_path = tmp_path / "omic.zip"
    _write_archive(archive_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()

    result = load_brca_patient_omics(
        archive_path,
        case_id=CASE_ID,
        slide_id=SLIDE_ID,
        expected_archive_sha256=digest,
    )
    assert result.case_id == CASE_ID

    with pytest.raises(BrcaOmicContractError, match="SHA256 mismatch"):
        load_brca_patient_omics(
            archive_path,
            case_id=CASE_ID,
            slide_id=SLIDE_ID,
            expected_archive_sha256="0" * 64,
        )

    with pytest.raises(BrcaOmicContractError, match="64-character hexadecimal"):
        load_brca_patient_omics(
            archive_path,
            case_id=CASE_ID,
            slide_id=SLIDE_ID,
            expected_archive_sha256="not-a-sha256",
        )
