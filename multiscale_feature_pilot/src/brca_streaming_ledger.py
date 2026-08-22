"""Crash-safe append-only patient-stage ledger for BRCA streaming design."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .brca_singleton_streaming_policy import PatientStage, advance_stage


GENESIS_HASH = "0" * 64


class LedgerError(RuntimeError):
    pass


@dataclass(frozen=True)
class LedgerRecord:
    sequence: int
    patient_id: str
    slide_id: str
    stage: PatientStage
    previous_record_sha256: str
    evidence_sha256: str
    authorization_sha256: str
    recorded_at_utc: str
    record_sha256: str


def _canonical(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value: str, label: str) -> None:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise LedgerError(f"{label} must be a lowercase SHA256")


def _payload(record: LedgerRecord) -> dict[str, object]:
    return {
        "sequence": record.sequence,
        "patient_id": record.patient_id,
        "slide_id": record.slide_id,
        "stage": record.stage.value,
        "previous_record_sha256": record.previous_record_sha256,
        "evidence_sha256": record.evidence_sha256,
        "authorization_sha256": record.authorization_sha256,
        "recorded_at_utc": record.recorded_at_utc,
    }


def build_record(
    existing: Sequence[LedgerRecord],
    *,
    patient_id: str,
    slide_id: str,
    stage: PatientStage,
    evidence_sha256: str,
    authorization_sha256: str,
    recorded_at_utc: str,
) -> LedgerRecord:
    """Build the next record after validating identity and stage continuity."""

    _digest(evidence_sha256, "evidence_sha256")
    _digest(authorization_sha256, "authorization_sha256")
    if not patient_id or not slide_id or not recorded_at_utc:
        raise LedgerError("patient, slide, and timestamp are required")
    if existing:
        prior = existing[-1]
        if prior.patient_id != patient_id or prior.slide_id != slide_id:
            raise LedgerError("one ledger directory may contain only one patient/slide")
        try:
            advance_stage(
                prior.stage,
                stage,
                separately_authorized=True,
                evidence_verified=True,
            )
        except ValueError as error:
            raise LedgerError(str(error)) from error
        previous = prior.record_sha256
    else:
        if stage is not PatientStage.PLANNED:
            raise LedgerError("first ledger record must be PLANNED")
        previous = GENESIS_HASH
    provisional = LedgerRecord(
        sequence=len(existing) + 1,
        patient_id=patient_id,
        slide_id=slide_id,
        stage=stage,
        previous_record_sha256=previous,
        evidence_sha256=evidence_sha256,
        authorization_sha256=authorization_sha256,
        recorded_at_utc=recorded_at_utc,
        record_sha256="",
    )
    digest = hashlib.sha256(_canonical(_payload(provisional))).hexdigest()
    return LedgerRecord(**{**provisional.__dict__, "record_sha256": digest})


def validate_records(records: Sequence[LedgerRecord]) -> None:
    for index, record in enumerate(records):
        if record.sequence != index + 1:
            raise LedgerError("ledger sequence is not contiguous")
        expected_previous = GENESIS_HASH if index == 0 else records[index - 1].record_sha256
        if record.previous_record_sha256 != expected_previous:
            raise LedgerError("ledger hash chain is broken")
        expected_hash = hashlib.sha256(_canonical(_payload(record))).hexdigest()
        if record.record_sha256 != expected_hash:
            raise LedgerError("ledger record hash mismatch")
        _digest(record.evidence_sha256, "evidence_sha256")
        _digest(record.authorization_sha256, "authorization_sha256")
        if index == 0 and record.stage is not PatientStage.PLANNED:
            raise LedgerError("ledger must start at PLANNED")
        if index:
            try:
                advance_stage(
                    records[index - 1].stage,
                    record.stage,
                    separately_authorized=True,
                    evidence_verified=True,
                )
            except ValueError as error:
                raise LedgerError(str(error)) from error


def load_ledger(directory: str | Path) -> tuple[LedgerRecord, ...]:
    directory = Path(directory)
    if not directory.exists():
        return ()
    if not directory.is_dir() or directory.is_symlink():
        raise LedgerError("ledger path must be a non-symlink directory")
    paths = sorted(directory.iterdir())
    expected_names = [f"{index:08d}.json" for index in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected_names:
        raise LedgerError("ledger filenames must be contiguous immutable sequence numbers")
    result = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise LedgerError("ledger records must be regular non-symlink files")
        document = json.loads(path.read_text(encoding="utf-8"))
        result.append(LedgerRecord(stage=PatientStage(document.pop("stage")), **document))
    validate_records(result)
    return tuple(result)


def append_record(directory: str | Path, record: LedgerRecord) -> Path:
    """Publish one immutable record with exclusive creation and fsync."""

    directory = Path(directory)
    directory.mkdir(mode=0o750, parents=False, exist_ok=True)
    current = load_ledger(directory)
    expected = build_record(
        current,
        patient_id=record.patient_id,
        slide_id=record.slide_id,
        stage=record.stage,
        evidence_sha256=record.evidence_sha256,
        authorization_sha256=record.authorization_sha256,
        recorded_at_utc=record.recorded_at_utc,
    )
    if record != expected:
        raise LedgerError("record does not match the current ledger tip")
    path = directory / f"{record.sequence:08d}.json"
    document = {**_payload(record), "record_sha256": record.record_sha256}
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o640)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(_canonical(document))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    return path


__all__ = [
    "GENESIS_HASH",
    "LedgerError",
    "LedgerRecord",
    "append_record",
    "build_record",
    "load_ledger",
    "validate_records",
]
