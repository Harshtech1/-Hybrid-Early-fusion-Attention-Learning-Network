"""Deterministic, leakage-safe BRCA survival split and discretization.

This module reads only the frozen alignment CSV and Omic archive metadata.  It
does not import Torch, open feature tensors or WSIs, execute a model, access a
network, or expose a training operation.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Iterable, Mapping, Sequence
import zipfile


ALIGNMENT_SHA256 = "13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe"
OMIC_ARCHIVE_SHA256 = "4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8"
OMIC_MEMBER = "./tcga_brca_all_clean.csv.zip"
OMIC_MEMBER_SIZE = 15_021_018
COHORT_ORDER_SHA256 = "1c97fa4f8305185f2da191f5ebaed603db7d2bdd11c89a580e784ef46655af5a"
SPLIT_SEED = 20_260_820
TRAIN_PERCENT = 85
MAX_ALIGNMENT_BYTES = 16_000_000
MAX_OMIC_ARCHIVE_BYTES = 5_000_000
MAX_OMIC_MEMBER_BYTES = 20_000_000
MAX_SURVIVAL_KS = Decimal("0.15")
PARTITIONS = ("training", "validation", "locked_test")
SPLIT_COLUMNS = (
    "cohort_index",
    "patient_id",
    "slide_id",
    "gdc_uuid",
    "omic_source_row_id",
    "source_train_value",
    "partition",
    "survival_months",
    "censorship",
    "event_observed",
    "discrete_time_bin",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class SurvivalProtocolError(RuntimeError):
    """Raised when endpoint, split or cutpoint evidence drifts."""


@dataclass(frozen=True)
class EndpointRecord:
    cohort_index: int
    patient_id: str
    slide_id: str
    gdc_uuid: str
    omic_source_row_id: str
    source_train_value: int
    survival_months_text: str
    survival_months: Decimal
    censorship: int
    partition: str = "UNASSIGNED"
    discrete_time_bin: int = -1

    @property
    def event_observed(self) -> int:
        return 1 - self.censorship


@dataclass(frozen=True)
class FrozenSurvivalSplit:
    records: tuple[EndpointRecord, ...]
    cutpoints_months: tuple[Decimal, Decimal, Decimal]
    training_event_patients: int
    train_validation_ks: Decimal
    manifest_sha256: str
    cutpoints_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SurvivalProtocolError(message)


def _strict_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise SurvivalProtocolError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for parent in reversed(absolute.parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode), f"symlink ancestor forbidden: {parent}")


def _read_bounded_nofollow(path: Path, maximum_bytes: int) -> bytes:
    _no_symlink_ancestors(path)
    _require(os.path.lexists(path), f"required source is absent: {path}")
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"regular non-symlink required: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        token = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        _require(token == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns), "source changed before secure open")
        _require(0 < opened.st_size <= maximum_bytes, "source size violates bound")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            _require(bool(chunk), "unexpected EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", "source grew during bounded read")
        final = os.fstat(descriptor)
        after = path.lstat()
        final_token = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
        path_token = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        _require(final_token == token == path_token, "source identity changed during secure read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _parse_decimal(text: str, label: str) -> Decimal:
    try:
        value = Decimal(text)
    except (InvalidOperation, ValueError) as error:
        raise SurvivalProtocolError(f"{label} is not a decimal") from error
    _require(value.is_finite(), f"{label} must be finite")
    return value


def _load_alignment(payload: bytes) -> tuple[dict[str, str], ...]:
    _require(_sha256(payload) == ALIGNMENT_SHA256, "alignment SHA256 mismatch")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SurvivalProtocolError("alignment is not strict UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    required = {
        "case_id", "slide_id", "alignment_status", "alignment_reason",
        "patient_wsi_count", "patient_omic_count", "omic_source_row_id",
        "id", "filename", "md5", "size", "state",
    }
    _require(required.issubset(set(reader.fieldnames or ())), "alignment columns drift")
    rows = [
        row for row in reader
        if row["alignment_status"] == "KEEP"
        and row["alignment_reason"] == "EXACT_SINGLETON_CASE_AND_SLIDE_MATCH"
    ]
    rows.sort(key=lambda row: (row["case_id"], row["slide_id"], row["id"]))
    _require(len(rows) == 894, "alignment must retain exactly 894 singleton patients")
    _require(len({row["case_id"] for row in rows}) == 894, "duplicate retained patient")
    _require(len({row["slide_id"] for row in rows}) == 894, "duplicate retained slide")
    _require(len({row["id"] for row in rows}) == 894, "duplicate retained UUID")
    return tuple(rows)


def _load_omic_metadata(payload: bytes) -> dict[tuple[str, str], dict[str, str]]:
    _require(_sha256(payload) == OMIC_ARCHIVE_SHA256, "Omic archive SHA256 mismatch")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            members = archive.infolist()
            _require(len(members) == 1 and not members[0].is_dir(), "Omic archive must contain exactly one file")
            member = members[0]
            _require(member.filename == OMIC_MEMBER, "Omic archive member drift")
            _require(member.file_size == OMIC_MEMBER_SIZE and member.file_size <= MAX_OMIC_MEMBER_BYTES, "Omic member size drift")
            _require(not (member.flag_bits & 0x1), "encrypted Omic archive unsupported")
            with archive.open(member) as raw:
                with io.TextIOWrapper(raw, encoding="utf-8-sig", newline="") as text:
                    reader = csv.DictReader(text, strict=True)
                    header = tuple(reader.fieldnames or ())
                    _require(len(header) == len(set(header)), "duplicate Omic column")
                    required = {"", "case_id", "slide_id", "survival_months", "censorship", "train"}
                    _require(required.issubset(set(header)), "Omic metadata columns drift")
                    result: dict[tuple[str, str], dict[str, str]] = {}
                    total = 0
                    for row in reader:
                        total += 1
                        _require(None not in row, f"Omic row {total + 1} has extra columns")
                        key = (row["case_id"], row["slide_id"])
                        _require(key not in result, f"duplicate Omic case/slide row: {key}")
                        result[key] = {
                            "omic_source_row_id": row[""],
                            "survival_months": row["survival_months"],
                            "censorship": row["censorship"],
                            "train": row["train"],
                        }
                    _require(total == 1022, "Omic source row count drift")
                    return result
    except SurvivalProtocolError:
        raise
    except (OSError, UnicodeError, csv.Error, zipfile.BadZipFile) as error:
        raise SurvivalProtocolError(f"could not parse Omic archive: {error}") from error


def load_endpoint_records(alignment_path: str | Path, omic_archive_path: str | Path) -> tuple[EndpointRecord, ...]:
    """Bind exact endpoint metadata to the frozen 894-patient order."""

    alignment = _load_alignment(_read_bounded_nofollow(Path(alignment_path), MAX_ALIGNMENT_BYTES))
    omic = _load_omic_metadata(_read_bounded_nofollow(Path(omic_archive_path), MAX_OMIC_ARCHIVE_BYTES))
    records: list[EndpointRecord] = []
    for cohort_index, row in enumerate(alignment, start=1):
        key = (row["case_id"], row["slide_id"])
        _require(key in omic, f"retained patient is absent from Omic source: {key}")
        metadata = omic[key]
        _require(metadata["omic_source_row_id"] == row["omic_source_row_id"], f"Omic source row mismatch: {key}")
        survival = _parse_decimal(metadata["survival_months"], "survival_months")
        _require(survival > 0, "survival_months must be strictly positive")
        _require(metadata["censorship"] in {"0.0", "1.0"}, "censorship must be 0.0 or 1.0")
        _require(metadata["train"] in {"0.0", "1.0"}, "source train value must be 0.0 or 1.0")
        records.append(
            EndpointRecord(
                cohort_index=cohort_index,
                patient_id=row["case_id"],
                slide_id=row["slide_id"],
                gdc_uuid=row["id"],
                omic_source_row_id=row["omic_source_row_id"],
                source_train_value=int(Decimal(metadata["train"])),
                survival_months_text=metadata["survival_months"],
                survival_months=survival,
                censorship=int(Decimal(metadata["censorship"])),
            )
        )
    _require(len(records) == 894, "endpoint record count drift")
    _require(sum(record.event_observed for record in records) == 126, "observed-event count drift")
    _require(sum(record.censorship for record in records) == 768, "censored count drift")
    return tuple(records)


def _ranking_digest(record: EndpointRecord) -> str:
    payload = f"{SPLIT_SEED}\0{record.censorship}.0\0{record.patient_id}\0{record.slide_id}".encode("utf-8")
    return _sha256(payload)


def _quantile_type7(values: Sequence[Decimal], numerator: int, denominator: int) -> Decimal:
    _require(bool(values), "cannot fit a quantile to an empty population")
    ordered = sorted(values)
    position = Decimal(len(ordered) - 1) * Decimal(numerator) / Decimal(denominator)
    lower = int(position)
    fraction = position - Decimal(lower)
    if not fraction:
        return ordered[lower]
    return ordered[lower] + fraction * (ordered[lower + 1] - ordered[lower])


def _bin_index(value: Decimal, cutpoints: Sequence[Decimal]) -> int:
    return sum(value >= cutpoint for cutpoint in cutpoints)


def _ks_distance(left: Sequence[Decimal], right: Sequence[Decimal]) -> Decimal:
    _require(bool(left) and bool(right), "KS populations cannot be empty")
    x = sorted(left)
    y = sorted(right)
    values = sorted(set(x + y))
    i = 0
    j = 0
    maximum = Decimal(0)
    for value in values:
        while i < len(x) and x[i] <= value:
            i += 1
        while j < len(y) and y[j] <= value:
            j += 1
        difference = abs(Decimal(i) / Decimal(len(x)) - Decimal(j) / Decimal(len(y)))
        maximum = max(maximum, difference)
    return maximum


def split_manifest_bytes(records: Sequence[EndpointRecord]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(SPLIT_COLUMNS)
    for record in records:
        writer.writerow(
            (
                record.cohort_index,
                record.patient_id,
                record.slide_id,
                record.gdc_uuid,
                record.omic_source_row_id,
                record.source_train_value,
                record.partition,
                record.survival_months_text,
                record.censorship,
                record.event_observed,
                record.discrete_time_bin,
            )
        )
    return buffer.getvalue().encode("utf-8")


def cutpoints_document(split: FrozenSurvivalSplit) -> dict[str, object]:
    counts: dict[str, dict[str, object]] = {}
    for partition in PARTITIONS:
        subset = [record for record in split.records if record.partition == partition]
        counts[partition] = {
            "patients": len(subset),
            "observed_events": sum(record.event_observed for record in subset),
            "right_censored": sum(record.censorship for record in subset),
            "bin_counts": [sum(record.discrete_time_bin == index for record in subset) for index in range(4)],
            "event_bin_counts": [
                sum(record.discrete_time_bin == index and record.event_observed == 1 for record in subset)
                for index in range(4)
            ],
        }
    return {
        "schema": "BRCA_SURVIVAL_CUTPOINTS_V1",
        "endpoint": "source_defined_time_to_event",
        "time_column": "survival_months",
        "time_unit": "months",
        "event_indicator": "1-censorship",
        "fit_partition": "training",
        "fit_population": "observed_event_patients_only",
        "quantile_method": "Hyndman_Fan_type_7_linear",
        "internal_quantiles": ["0.25", "0.50", "0.75"],
        "internal_cutpoints_months": [str(value) for value in split.cutpoints_months],
        "intervals": ["[-inf,b1)", "[b1,b2)", "[b2,b3)", "[b3,+inf]"],
        "split_seed": SPLIT_SEED,
        "split_manifest_sha256": split.manifest_sha256,
        "training_event_patients": split.training_event_patients,
        "train_validation_survival_ks": str(split.train_validation_ks),
        "maximum_allowed_train_validation_survival_ks": str(MAX_SURVIVAL_KS),
        "partition_counts": counts,
    }


def cutpoints_bytes(split: FrozenSurvivalSplit) -> bytes:
    document = cutpoints_document(split)
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")


def build_frozen_split(records: Sequence[EndpointRecord]) -> FrozenSurvivalSplit:
    """Create the deterministic outer and inner patient partitions."""

    _require(len(records) == 894, "split requires exactly 894 endpoint records")
    development = [record for record in records if record.source_train_value == 1]
    locked_test = [record for record in records if record.source_train_value == 0]
    _require(len(development) == 776 and len(locked_test) == 118, "outer partition count drift")
    training: list[EndpointRecord] = []
    validation: list[EndpointRecord] = []
    for censorship in (0, 1):
        stratum = [record for record in development if record.censorship == censorship]
        stratum.sort(key=lambda record: (_ranking_digest(record), record.patient_id, record.slide_id))
        training_count = (len(stratum) * TRAIN_PERCENT + 50) // 100
        training.extend(stratum[:training_count])
        validation.extend(stratum[training_count:])
    _require((len(training), len(validation), len(locked_test)) == (660, 116, 118), "inner partition count drift")
    _require(
        tuple(sum(record.event_observed for record in group) for group in (training, validation, locked_test))
        == (79, 14, 33),
        "partition event-count drift",
    )
    assignments = {record.cohort_index: "training" for record in training}
    assignments.update({record.cohort_index: "validation" for record in validation})
    assignments.update({record.cohort_index: "locked_test" for record in locked_test})
    _require(len(assignments) == 894, "partition assignments are incomplete or overlapping")
    event_times = [record.survival_months for record in training if record.event_observed == 1]
    _require(len(event_times) == 79, "training event population drift")
    cutpoints = (
        _quantile_type7(event_times, 1, 4),
        _quantile_type7(event_times, 1, 2),
        _quantile_type7(event_times, 3, 4),
    )
    _require(all(value.is_finite() for value in cutpoints), "nonfinite cutpoint")
    _require(cutpoints[0] < cutpoints[1] < cutpoints[2], "cutpoints must be strictly increasing")
    frozen = tuple(
        replace(
            record,
            partition=assignments[record.cohort_index],
            discrete_time_bin=_bin_index(record.survival_months, cutpoints),
        )
        for record in sorted(records, key=lambda item: item.cohort_index)
    )
    train_values = [record.survival_months for record in frozen if record.partition == "training"]
    validation_values = [record.survival_months for record in frozen if record.partition == "validation"]
    distance = _ks_distance(train_values, validation_values)
    _require(distance <= MAX_SURVIVAL_KS, "train/validation survival distribution imbalance exceeds frozen limit")
    manifest_sha = _sha256(split_manifest_bytes(frozen))
    provisional = FrozenSurvivalSplit(
        records=frozen,
        cutpoints_months=cutpoints,
        training_event_patients=len(event_times),
        train_validation_ks=distance,
        manifest_sha256=manifest_sha,
        cutpoints_sha256="0" * 64,
    )
    return replace(provisional, cutpoints_sha256=_sha256(cutpoints_bytes(provisional)))


def parse_and_validate_split_artifacts(manifest_payload: bytes, cutpoints_payload: bytes) -> FrozenSurvivalSplit:
    """Strictly validate serialized split/cutpoint artifacts without source data."""

    try:
        reader = csv.DictReader(io.StringIO(manifest_payload.decode("utf-8"), newline=""), delimiter="\t", strict=True)
    except (UnicodeDecodeError, csv.Error) as error:
        raise SurvivalProtocolError("split manifest is not strict UTF-8 TSV") from error
    _require(tuple(reader.fieldnames or ()) == SPLIT_COLUMNS, "split manifest columns drift")
    records: list[EndpointRecord] = []
    for expected_index, row in enumerate(reader, start=1):
        _require(int(row["cohort_index"]) == expected_index, "split manifest cohort index is not contiguous")
        survival = _parse_decimal(row["survival_months"], "survival_months")
        censorship = int(row["censorship"])
        event = int(row["event_observed"])
        _require(censorship in {0, 1} and event == 1 - censorship, "split event/censorship mismatch")
        partition = row["partition"]
        _require(partition in PARTITIONS, "unknown split partition")
        bin_index = int(row["discrete_time_bin"])
        _require(bin_index in {0, 1, 2, 3}, "invalid discrete time bin")
        records.append(
            EndpointRecord(
                cohort_index=expected_index,
                patient_id=row["patient_id"],
                slide_id=row["slide_id"],
                gdc_uuid=row["gdc_uuid"],
                omic_source_row_id=row["omic_source_row_id"],
                source_train_value=int(row["source_train_value"]),
                survival_months_text=row["survival_months"],
                survival_months=survival,
                censorship=censorship,
                partition=partition,
                discrete_time_bin=bin_index,
            )
        )
    _require(len(records) == 894, "serialized split must contain exactly 894 rows")
    try:
        document = json.loads(cutpoints_payload.decode("utf-8"), object_pairs_hook=_strict_json_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SurvivalProtocolError("cutpoints document is not strict JSON") from error
    _require(document.get("schema") == "BRCA_SURVIVAL_CUTPOINTS_V1", "cutpoints schema drift")
    _require(document.get("split_manifest_sha256") == _sha256(manifest_payload), "cutpoints do not bind split manifest")
    cutpoints = tuple(_parse_decimal(value, "cutpoint") for value in document["internal_cutpoints_months"])
    _require(len(cutpoints) == 3 and cutpoints[0] < cutpoints[1] < cutpoints[2], "serialized cutpoints are invalid")
    for record in records:
        _require(record.discrete_time_bin == _bin_index(record.survival_months, cutpoints), "serialized bin assignment drift")
    distance = _parse_decimal(document["train_validation_survival_ks"], "KS distance")
    result = FrozenSurvivalSplit(
        records=tuple(records),
        cutpoints_months=(cutpoints[0], cutpoints[1], cutpoints[2]),
        training_event_patients=int(document["training_event_patients"]),
        train_validation_ks=distance,
        manifest_sha256=_sha256(manifest_payload),
        cutpoints_sha256=_sha256(cutpoints_payload),
    )
    _require(cutpoints_bytes(result) == cutpoints_payload, "cutpoints document is not canonical or internally consistent")
    return result


def write_split_artifacts(split: FrozenSurvivalSplit, manifest_path: str | Path, cutpoints_path: str | Path) -> None:
    """Publish derived CPU metadata once, without overwrite or deletion."""

    outputs = (
        (Path(manifest_path), split_manifest_bytes(split.records)),
        (Path(cutpoints_path), cutpoints_bytes(split)),
    )
    _require(outputs[0][0].parent == outputs[1][0].parent, "split artifacts must share a parent")
    parent = outputs[0][0].parent
    _require(parent.is_dir() and not parent.is_symlink(), "split artifact parent must be a regular directory")
    for path, _ in outputs:
        _require(not os.path.lexists(path), f"refusing to overwrite split artifact: {path}")
    for path, payload in outputs:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o644)
        try:
            written = 0
            while written < len(payload):
                written += os.write(descriptor, payload[written:])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


__all__ = [
    "ALIGNMENT_SHA256",
    "COHORT_ORDER_SHA256",
    "OMIC_ARCHIVE_SHA256",
    "SPLIT_COLUMNS",
    "SPLIT_SEED",
    "EndpointRecord",
    "FrozenSurvivalSplit",
    "SurvivalProtocolError",
    "build_frozen_split",
    "cutpoints_bytes",
    "cutpoints_document",
    "load_endpoint_records",
    "parse_and_validate_split_artifacts",
    "split_manifest_bytes",
    "write_split_artifacts",
]
