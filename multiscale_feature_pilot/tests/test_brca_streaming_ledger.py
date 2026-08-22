from dataclasses import replace
from pathlib import Path

import pytest

from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
from multiscale_feature_pilot.src.brca_streaming_ledger import (
    LedgerError,
    append_record,
    build_record,
    load_ledger,
    validate_records,
)


PATIENT = "TCGA-AA-0001"
SLIDE = "TCGA-AA-0001-01Z-00-DX1.TEST.svs"


def _next(existing, stage: PatientStage, number: int):
    return build_record(
        existing,
        patient_id=PATIENT,
        slide_id=SLIDE,
        stage=stage,
        evidence_sha256=f"{number:x}" * 64,
        authorization_sha256=f"{number + 1:x}" * 64,
        recorded_at_utc=f"2026-08-19T00:00:{number:02d}Z",
    )


def test_full_append_only_chain_round_trip(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    stages = list(PatientStage)
    for number, stage in enumerate(stages, start=1):
        existing = load_ledger(directory)
        record = _next(existing, stage, number)
        append_record(directory, record)
    loaded = load_ledger(directory)
    assert [record.stage for record in loaded] == stages
    assert sorted(path.name for path in directory.iterdir()) == [
        f"{index:08d}.json" for index in range(1, 9)
    ]


def test_stage_skip_and_identity_drift_are_rejected() -> None:
    planned = _next((), PatientStage.PLANNED, 1)
    with pytest.raises(LedgerError, match="advance exactly"):
        _next((planned,), PatientStage.RAW_VERIFIED, 2)
    with pytest.raises(LedgerError, match="one patient"):
        build_record(
            (planned,),
            patient_id="TCGA-BB-0002",
            slide_id=SLIDE,
            stage=PatientStage.ACQUISITION_AUTHORIZED,
            evidence_sha256="2" * 64,
            authorization_sha256="3" * 64,
            recorded_at_utc="2026-08-19T00:00:02Z",
        )


def test_tampered_hash_chain_is_rejected() -> None:
    planned = _next((), PatientStage.PLANNED, 1)
    authorized = _next((planned,), PatientStage.ACQUISITION_AUTHORIZED, 2)
    with pytest.raises(LedgerError, match="hash chain"):
        validate_records((planned, replace(authorized, previous_record_sha256="f" * 64)))


def test_existing_sequence_file_is_never_overwritten(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    planned = _next((), PatientStage.PLANNED, 1)
    append_record(directory, planned)
    before = (directory / "00000001.json").read_bytes()
    with pytest.raises(LedgerError):
        append_record(directory, replace(planned, record_sha256="f" * 64))
    assert (directory / "00000001.json").read_bytes() == before


def test_partial_or_unexpected_filename_stops_resume(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    directory.mkdir()
    (directory / "00000001.json.partial").write_text("incomplete")
    with pytest.raises(LedgerError, match="filenames"):
        load_ledger(directory)
