from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from multiscale_feature_pilot.src.omic import (
    OmicContractError,
    load_patient_omic_modalities,
)


CASE = "TCGA-AA-0001"
SLIDE = "TCGA-AA-0001-01Z-00-DX1.TEST.svs"
HEADER = [
    "_PATIENT",
    "sample",
    "slide_id",
    "gene_b_rnaseq",
    "gene_a_rnaseq",
    "gene_c_mut",
    "gene_d_cnv",
]


def _write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(rows)


def _matching_row(*, rna_b: str = "2.0") -> list[str]:
    return [CASE, f"{CASE}-01", SLIDE, rna_b, "1.0", "0", "-1.0"]


def test_exact_patient_and_slide_match_preserves_group_order(tmp_path: Path) -> None:
    csv_path = tmp_path / "omic.csv"
    _write_csv(
        csv_path,
        [
            ["TCGA-BB-0002", "TCGA-BB-0002-01", "other.svs", "9", "9", "1", "1"],
            _matching_row(),
        ],
    )

    result = load_patient_omic_modalities(
        csv_path,
        case_id=CASE,
        slide_id=SLIDE,
        expected_dims={"rna": 2, "mutation": 1, "cnv": 1},
    )

    assert result.case_id == CASE
    assert result.slide_id == SLIDE
    assert result.rna.shape == (1, 1, 2)
    assert result.mutation.shape == (1, 1, 1)
    assert result.cnv.shape == (1, 1, 1)
    assert result.rna.dtype is torch.float32
    assert result.rna.flatten().tolist() == [2.0, 1.0]


def test_cross_patient_or_slide_mismatch_is_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "omic.csv"
    _write_csv(csv_path, [_matching_row()])

    with pytest.raises(OmicContractError, match="found 0"):
        load_patient_omic_modalities(
            csv_path,
            case_id="TCGA-WRONG-0000",
            slide_id=SLIDE,
        )


def test_duplicate_exact_rows_are_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "omic.csv"
    _write_csv(csv_path, [_matching_row(), _matching_row()])

    with pytest.raises(OmicContractError, match="found 2"):
        load_patient_omic_modalities(csv_path, case_id=CASE, slide_id=SLIDE)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", ""])
def test_nonfinite_or_missing_values_are_rejected(tmp_path: Path, value: str) -> None:
    csv_path = tmp_path / "omic.csv"
    _write_csv(csv_path, [_matching_row(rna_b=value)])

    with pytest.raises(OmicContractError):
        load_patient_omic_modalities(csv_path, case_id=CASE, slide_id=SLIDE)


def test_wrong_expected_group_width_is_rejected(tmp_path: Path) -> None:
    csv_path = tmp_path / "omic.csv"
    _write_csv(csv_path, [_matching_row()])

    with pytest.raises(OmicContractError, match="rna width"):
        load_patient_omic_modalities(
            csv_path,
            case_id=CASE,
            slide_id=SLIDE,
            expected_dims={"rna": 3},
        )
