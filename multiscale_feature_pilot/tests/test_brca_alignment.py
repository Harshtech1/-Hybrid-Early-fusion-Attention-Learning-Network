from __future__ import annotations

import csv
import hashlib
import io
import uuid
import zipfile
from pathlib import Path

import pytest

from scripts.build_brca_alignment import (
    AlignmentError,
    OmicSource,
    OmicRow,
    SourceIdentity,
    WsiRow,
    align_exact_keys,
    read_omic_archive,
    read_wsi_manifest,
    render_alignment_csv,
    render_download_plan_tsv,
    summarize_alignment,
)


def _slide(case_id: str, marker: str) -> str:
    return f"{case_id}-01Z-00-DX1.{marker}.svs"


def _uuid(number: int) -> str:
    return str(uuid.UUID(int=number))


def _wsi(case_id: str, marker: str, number: int) -> WsiRow:
    return WsiRow(
        case_id=case_id,
        slide_id=_slide(case_id, marker),
        gdc_file_uuid=_uuid(number),
        md5=f"{number:032x}",
        size_bytes=number * 100,
        state="released",
    )


def _omic_source() -> OmicSource:
    return OmicSource(
        identity=SourceIdentity("/source.zip", 1, "0" * 64),
        member_name="cohort.csv",
        member_size_bytes=1,
        member_sha256="1" * 64,
        rna_features=2,
        mutation_features=1,
        cnv_features=1,
    )


def test_exact_full_outer_join_is_sorted_and_never_uses_row_order() -> None:
    case_a = "TCGA-AA-0001"
    case_b = "TCGA-BB-0002"
    case_c = "TCGA-CC-0003"
    case_d = "TCGA-DD-0004"
    a1, a2 = _slide(case_a, "A1"), _slide(case_a, "A2")
    b1 = _slide(case_b, "B1")
    c1 = _slide(case_c, "C1")

    # Deliberately use different, non-key orders and unrelated source ordinals.
    omic_rows = [
        OmicRow(case_b, b1, "90"),
        OmicRow(case_a, a2, "3"),
        OmicRow(case_c, c1, "7"),
        OmicRow(case_a, a1, "42"),
    ]
    wsi_a1 = _wsi(case_a, "A1", 1)
    wsi_a2 = _wsi(case_a, "A2", 2)
    wsi_b1 = _wsi(case_b, "B1", 3)
    wsi_d1 = _wsi(case_d, "D1", 4)
    wsi_rows = [wsi_a2, wsi_d1, wsi_b1, wsi_a1]

    rows = align_exact_keys(omic_rows, wsi_rows)

    assert [(row.case_id, row.slide_id) for row in rows] == sorted(
        (row.case_id, row.slide_id) for row in rows
    )
    assert [row.alignment_status for row in rows] == [
        "AMBIGUOUS",
        "AMBIGUOUS",
        "KEEP",
        "OMIC_ONLY",
        "WSI_ONLY",
    ]
    by_slide = {row.slide_id: row for row in rows}
    assert by_slide[a1].gdc_file_uuid == wsi_a1.gdc_file_uuid
    assert by_slide[a1].omic_source_row_id == "42"
    assert by_slide[a2].gdc_file_uuid == wsi_a2.gdc_file_uuid
    assert by_slide[c1].gdc_file_uuid == ""

    summary = summarize_alignment(omic_rows, wsi_rows, rows, _omic_source())
    assert summary.keep_patients == 1
    assert summary.ambiguous_patients == 1
    assert summary.wsi_only_rows == 1
    assert summary.wsi_only_patients == 1
    assert summary.omic_only_rows == 1
    assert summary.omic_only_patients == 1


def test_rendering_is_deterministic_and_downloads_are_not_authorized() -> None:
    case_a, case_b = "TCGA-AA-0001", "TCGA-BB-0002"
    omic_rows = [
        OmicRow(case_b, _slide(case_b, "B"), "0"),
        OmicRow(case_a, _slide(case_a, "A"), "1"),
    ]
    wsi_rows = [_wsi(case_a, "A", 1), _wsi(case_b, "B", 2)]
    forward = align_exact_keys(omic_rows, wsi_rows)
    reverse = align_exact_keys(list(reversed(omic_rows)), list(reversed(wsi_rows)))

    alignment_csv = render_alignment_csv(forward)
    assert alignment_csv == render_alignment_csv(reverse)
    alignment_header = next(csv.reader(io.StringIO(alignment_csv)))
    assert alignment_header == [
        "case_id",
        "slide_id",
        "alignment_status",
        "alignment_reason",
        "patient_wsi_count",
        "patient_omic_count",
        "omic_source_row_id",
        "id",
        "filename",
        "md5",
        "size",
        "state",
    ]
    plan = render_download_plan_tsv(reverse)
    assert plan == render_download_plan_tsv(forward)
    parsed = list(csv.DictReader(io.StringIO(plan), delimiter="\t"))
    assert list(parsed[0]) == [
        "case_id",
        "slide_id",
        "id",
        "filename",
        "md5",
        "size",
        "state",
        "alignment_status",
        "candidate_disposition",
        "download_status",
        "authorization_basis",
    ]
    assert [row["case_id"] for row in parsed] == [case_a, case_b]
    assert {row["download_status"] for row in parsed} == {"NOT_AUTHORIZED"}
    assert {row["authorization_basis"] for row in parsed} == {
        "AWAIT_EXACT_THREE_PATIENT_CONFIRMATION"
    }


def test_source_readers_preserve_exact_identity_and_feature_counts(tmp_path: Path) -> None:
    case_id = "TCGA-AA-0001"
    slide_id = _slide(case_id, "ONE")
    csv_payload = (
        ",case_id,slide_id,gene_a_rnaseq,gene_b_rnaseq,gene_c_mut,gene_d_cnv\n"
        f"17,{case_id},{slide_id},1,2,0,-1\n"
    ).encode()
    archive_path = tmp_path / "omic.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("cohort.csv", csv_payload)

    manifest_path = tmp_path / "manifest.tsv"
    manifest_path.write_text(
        "id\tfilename\tmd5\tsize\tstate\n"
        f"{_uuid(1)}\t{slide_id}\t{'a' * 32}\t123\treleased\n",
        encoding="utf-8",
    )

    omic_rows, omic_source = read_omic_archive(archive_path)
    wsi_rows, manifest_source = read_wsi_manifest(manifest_path)

    assert omic_rows == [OmicRow(case_id, slide_id, "17")]
    assert (omic_source.rna_features, omic_source.mutation_features, omic_source.cnv_features) == (
        2,
        1,
        1,
    )
    assert omic_source.member_sha256 == hashlib.sha256(csv_payload).hexdigest()
    assert manifest_source.sha256 == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert wsi_rows[0].case_id == case_id
    assert wsi_rows[0].slide_id == slide_id


def test_omic_case_slide_mismatch_is_rejected(tmp_path: Path) -> None:
    archive_path = tmp_path / "omic.zip"
    payload = (
        ",case_id,slide_id,gene_rnaseq,gene_mut,gene_cnv\n"
        f"0,TCGA-AA-0001,{_slide('TCGA-BB-0002', 'WRONG')},1,0,1\n"
    )
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("cohort.csv", payload)

    with pytest.raises(AlignmentError, match="case/slide mismatch"):
        read_omic_archive(archive_path)


def test_duplicate_exact_key_is_rejected() -> None:
    case_id = "TCGA-AA-0001"
    slide_id = _slide(case_id, "DUP")
    rows = [OmicRow(case_id, slide_id, "0"), OmicRow(case_id, slide_id, "1")]

    with pytest.raises(AlignmentError, match="duplicate Omic exact key"):
        align_exact_keys(rows, [])
