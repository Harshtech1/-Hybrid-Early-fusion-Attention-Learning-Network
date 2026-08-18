"""Fail-closed one-row GDC manifests for the BRCA metadata preflight.

This module deliberately creates metadata artifacts only.  The three reviewed
BRCA objects remain ``NOT_AUTHORIZED`` and no combined manifest or download
command is produced.  Each GDC manifest retains the exact standard five-column
schema so it can be compared byte-for-byte with the released source manifest.
"""

from __future__ import annotations

import csv
import hashlib
import io
import os
import re
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent

DEFAULT_PROPOSAL = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_three_patient_proposal.yaml"
)
DEFAULT_ALIGNMENT = REPOSITORY_ROOT / "reports/brca_row_level_alignment.csv"
DEFAULT_SOURCE_MANIFEST = (
    WORKSPACE_ROOT
    / "healnet/data/tcga/gdc_manifests/filtered/brca_wsi_manifest_filtered.txt"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_phase2_manifests"
)

EXPECTED_PROPOSAL_SHA256 = (
    "b1bbeb06bb200813122e0b1a88a3d0258660eaff067d35f1d8de1dd6d79badb2"
)
EXPECTED_ALIGNMENT_SHA256 = (
    "13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = (
    "ac1b4d591ce255a7d4e6bde3778a041b107576693657ee296fe6eb36c4e79a92"
)
GDC_FIELDS = ("id", "filename", "md5", "size", "state")
AUTHORIZATION_STATUS = "NOT_AUTHORIZED"
POLICY_LABEL = "BRCA_PHASE2_METADATA_PREFLIGHT_V1"
AUTHORIZATION_RECORD_BASENAME = "MANIFEST_SET.NOT_AUTHORIZED.yaml"

UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
PATIENT_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")

EXPECTED_SELECTIONS = (
    (
        "Q25",
        "TCGA-LL-A6FP",
        "dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6",
        "TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs",
        "75536393096ffd928bc35ec9503c3655",
        648_046_947,
        "released",
    ),
    (
        "Q50",
        "TCGA-AR-A1AW",
        "5c1216f3-19ec-4d3c-9bb0-9bd740b79f62",
        "TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs",
        "304509e03f26cbecc9aee4ea691c8e5a",
        975_626_387,
        "released",
    ),
    (
        "Q75",
        "TCGA-E2-A154",
        "25aec062-60d1-446e-a1c6-0c79cc74a770",
        "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs",
        "a8c4b68fb6e0ab3e862efe3ed1fe10d7",
        1_360_743_825,
        "released",
    ),
)


class OneRowManifestError(ValueError):
    """Raised when a source or generated manifest violates the frozen policy."""


@dataclass(frozen=True)
class OneRowSelection:
    """One reviewed BRCA patient and its exact GDC file metadata."""

    label: str
    patient_id: str
    gdc_file_uuid: str
    filename: str
    md5: str
    size_bytes: int
    state: str

    @property
    def guarded_basename(self) -> str:
        return (
            f"{self.label}_{self.patient_id}_{self.gdc_file_uuid}."
            f"{AUTHORIZATION_STATUS}.gdc.tsv"
        )

    def as_gdc_row(self) -> dict[str, str]:
        return {
            "id": self.gdc_file_uuid,
            "filename": self.filename,
            "md5": self.md5,
            "size": str(self.size_bytes),
            "state": self.state,
        }


@dataclass(frozen=True)
class OneRowManifestArtifact:
    """Validated path and content identity for one generated manifest."""

    selection: OneRowSelection
    path: Path
    sha256: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OneRowManifestError(message)


def _require_source_hash(path: Path, expected: str, label: str) -> None:
    _require(path.is_file(), f"{label} is missing: {path}")
    actual = sha256_file(path)
    _require(
        actual == expected,
        f"{label} SHA256 drift: expected {expected}, got {actual}",
    )


def _load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise OneRowManifestError(f"cannot read YAML {path}: {exc}") from exc
    _require(isinstance(payload, Mapping), f"YAML root is not a mapping: {path}")
    return payload


def _selection_from_mapping(item: Mapping[str, Any]) -> OneRowSelection:
    required = {
        "label",
        "patient_id",
        "wsi_uuid",
        "slide_id",
        "wsi_md5",
        "wsi_size_bytes",
        "gdc_state",
    }
    _require(required.issubset(item), "proposal selection is missing required fields")
    try:
        selection = OneRowSelection(
            label=str(item["label"]),
            patient_id=str(item["patient_id"]),
            gdc_file_uuid=str(item["wsi_uuid"]),
            filename=str(item["slide_id"]),
            md5=str(item["wsi_md5"]),
            size_bytes=int(item["wsi_size_bytes"]),
            state=str(item["gdc_state"]),
        )
    except (TypeError, ValueError) as exc:
        raise OneRowManifestError("proposal selection has invalid field types") from exc
    _validate_selection(selection)
    return selection


def _validate_selection(item: OneRowSelection) -> None:
    _require(item.label in {"Q25", "Q50", "Q75"}, "invalid selection label")
    _require(PATIENT_RE.fullmatch(item.patient_id) is not None, "invalid patient_id")
    _require(UUID_RE.fullmatch(item.gdc_file_uuid) is not None, "invalid GDC UUID")
    _require(MD5_RE.fullmatch(item.md5) is not None, "invalid GDC MD5")
    _require(item.filename.startswith(item.patient_id + "-"), "patient/slide mismatch")
    _require(item.filename.endswith(".svs"), "selected filename is not an SVS")
    _require(item.size_bytes > 0, "selected WSI size is not positive")
    _require(item.state == "released", "selected GDC object is not released")
    _require(
        AUTHORIZATION_STATUS in item.guarded_basename,
        "generated manifest basename lacks NOT_AUTHORIZED guard",
    )


def load_reviewed_selections(
    proposal_path: Path = DEFAULT_PROPOSAL,
    *,
    verify_hash: bool = True,
) -> tuple[OneRowSelection, ...]:
    """Load exactly the three frozen proposal rows and reject any drift."""

    if verify_hash:
        _require_source_hash(
            proposal_path, EXPECTED_PROPOSAL_SHA256, "BRCA three-patient proposal"
        )
    payload = _load_yaml_mapping(proposal_path)
    _require(
        payload.get("status") == "PROPOSED_BRCA_PILOT_NOT_AUTHORIZED",
        "proposal is not explicitly NOT_AUTHORIZED",
    )
    decision_scope = payload.get("decision_scope")
    _require(isinstance(decision_scope, Mapping), "proposal decision_scope is missing")
    _require(
        decision_scope.get("wsi_download") == "not_authorized",
        "proposal does not prohibit WSI download",
    )
    raw_selections = payload.get("selected_patients")
    _require(isinstance(raw_selections, list), "proposal selections are missing")
    _require(
        len(raw_selections) == 3,
        f"proposal must contain exactly 3 rows, found {len(raw_selections)}",
    )
    _require(
        all(isinstance(item, Mapping) for item in raw_selections),
        "proposal selections must be mappings",
    )
    selections = tuple(_selection_from_mapping(item) for item in raw_selections)
    for attribute in ("label", "patient_id", "gdc_file_uuid", "filename"):
        values = [getattr(item, attribute) for item in selections]
        _require(len(set(values)) == 3, f"duplicate proposal {attribute}")
    actual = tuple(
        (
            item.label,
            item.patient_id,
            item.gdc_file_uuid,
            item.filename,
            item.md5,
            item.size_bytes,
            item.state,
        )
        for item in selections
    )
    _require(actual == EXPECTED_SELECTIONS, "reviewed BRCA selection identity drift")
    return selections


def _read_delimited_rows(
    path: Path, *, delimiter: str, expected_fields: Sequence[str], label: str
) -> list[dict[str, str]]:
    _require(path.is_file(), f"{label} is missing: {path}")
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=delimiter)
            _require(
                tuple(reader.fieldnames or ()) == tuple(expected_fields),
                f"unexpected {label} header: {reader.fieldnames!r}",
            )
            rows = list(reader)
    except OSError as exc:
        raise OneRowManifestError(f"cannot read {label}: {exc}") from exc
    _require(all(None not in row for row in rows), f"malformed row in {label}")
    return rows


def _cross_validate_sources(
    selections: Sequence[OneRowSelection],
    alignment_path: Path,
    source_manifest_path: Path,
    *,
    verify_hashes: bool,
) -> None:
    if verify_hashes:
        _require_source_hash(
            alignment_path, EXPECTED_ALIGNMENT_SHA256, "BRCA row-level alignment"
        )
        _require_source_hash(
            source_manifest_path,
            EXPECTED_SOURCE_MANIFEST_SHA256,
            "BRCA filtered GDC manifest",
        )

    alignment_fields = (
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
    )
    alignment = _read_delimited_rows(
        alignment_path,
        delimiter=",",
        expected_fields=alignment_fields,
        label="BRCA row-level alignment",
    )
    source = _read_delimited_rows(
        source_manifest_path,
        delimiter="\t",
        expected_fields=GDC_FIELDS,
        label="BRCA filtered GDC manifest",
    )

    for key, rows in (("alignment UUID", alignment), ("source UUID", source)):
        ids = [row["id"] for row in rows]
        _require(len(ids) == len(set(ids)), f"duplicate {key}")

    alignment_by_id = {row["id"]: row for row in alignment}
    source_by_id = {row["id"]: row for row in source}
    for item in selections:
        _require(item.gdc_file_uuid in alignment_by_id, "selected UUID missing from alignment")
        _require(item.gdc_file_uuid in source_by_id, "selected UUID missing from source manifest")
        alignment_row = alignment_by_id[item.gdc_file_uuid]
        source_row = source_by_id[item.gdc_file_uuid]
        _require(alignment_row["alignment_status"] == "KEEP", "selected row is not KEEP")
        _require(alignment_row["case_id"] == item.patient_id, "alignment patient drift")
        _require(alignment_row["slide_id"] == item.filename, "alignment slide drift")
        expected = item.as_gdc_row()
        for field in GDC_FIELDS:
            _require(
                alignment_row[field] == expected[field],
                f"alignment {field} drift for {item.label}",
            )
            _require(
                source_row[field] == expected[field],
                f"source manifest {field} drift for {item.label}",
            )


def _manifest_bytes(selection: OneRowSelection) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=GDC_FIELDS,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="raise",
    )
    writer.writeheader()
    writer.writerow(selection.as_gdc_row())
    return stream.getvalue().encode("utf-8")


def _write_new_or_require_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        _require(path.is_file(), f"output path is not a regular file: {path}")
        _require(path.read_bytes() == payload, f"existing output drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _authorization_record_bytes(
    artifacts: Sequence[OneRowManifestArtifact],
) -> bytes:
    payload = {
        "schema_version": 1,
        "policy_label": POLICY_LABEL,
        "status": AUTHORIZATION_STATUS,
        "download_authorized": False,
        "metadata_only": True,
        "source_hashes": {
            "proposal_sha256": EXPECTED_PROPOSAL_SHA256,
            "alignment_sha256": EXPECTED_ALIGNMENT_SHA256,
            "filtered_gdc_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        },
        "manifest_count": 3,
        "combined_manifest_present": False,
        "entries": [
            {
                "label": artifact.selection.label,
                "patient_id": artifact.selection.patient_id,
                "basename": artifact.path.name,
                "status": AUTHORIZATION_STATUS,
                "rows": 1,
                "sha256": artifact.sha256,
                **artifact.selection.as_gdc_row(),
            }
            for artifact in artifacts
        ],
    }
    return yaml.safe_dump(payload, sort_keys=False).encode("utf-8")


def _validate_guarded_output_set(
    output_directory: Path, expected_basenames: set[str]
) -> None:
    _require(
        not output_directory.is_symlink(),
        f"output directory must not be a symlink: {output_directory}",
    )
    _require(output_directory.is_dir(), f"output directory is missing: {output_directory}")
    entries = tuple(output_directory.iterdir())
    actual = {path.name for path in entries}
    expected = expected_basenames | {AUTHORIZATION_RECORD_BASENAME}
    _require(actual == expected, f"output file set drift: expected {sorted(expected)}, got {sorted(actual)}")
    for path in entries:
        _require(not path.is_symlink(), f"output entry must not be a symlink: {path}")
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise OneRowManifestError(f"cannot stat output entry: {path}") from exc
        _require(
            stat.S_ISREG(mode),
            f"output entry must be a regular file: {path}",
        )
    _require(
        all(AUTHORIZATION_STATUS in basename for basename in actual),
        "an output basename lacks the NOT_AUTHORIZED guard",
    )


def validate_phase2_manifest_set(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    proposal_path: Path = DEFAULT_PROPOSAL,
    alignment_path: Path = DEFAULT_ALIGNMENT,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    *,
    verify_source_hashes: bool = True,
) -> tuple[OneRowManifestArtifact, ...]:
    """Validate the complete guarded metadata set without writing anything."""

    selections = load_reviewed_selections(
        proposal_path, verify_hash=verify_source_hashes
    )
    _cross_validate_sources(
        selections,
        alignment_path,
        source_manifest_path,
        verify_hashes=verify_source_hashes,
    )
    expected_basenames = {item.guarded_basename for item in selections}
    _validate_guarded_output_set(output_directory, expected_basenames)

    artifacts: list[OneRowManifestArtifact] = []
    for selection in selections:
        path = output_directory / selection.guarded_basename
        rows = _read_delimited_rows(
            path,
            delimiter="\t",
            expected_fields=GDC_FIELDS,
            label=f"one-row manifest {selection.label}",
        )
        _require(
            len(rows) == 1,
            f"{selection.label} manifest must have exactly one data row, got {len(rows)}",
        )
        _require(rows[0] == selection.as_gdc_row(), f"{selection.label} output drift")
        artifacts.append(
            OneRowManifestArtifact(selection, path, sha256_file(path))
        )

    record_path = output_directory / AUTHORIZATION_RECORD_BASENAME
    record = _load_yaml_mapping(record_path)
    _require(record.get("status") == AUTHORIZATION_STATUS, "authorization record status drift")
    _require(record.get("download_authorized") is False, "download was marked authorized")
    _require(record.get("metadata_only") is True, "metadata-only guard is absent")
    _require(record.get("manifest_count") == 3, "authorization manifest count drift")
    _require(record.get("combined_manifest_present") is False, "combined manifest is prohibited")
    entries = record.get("entries")
    _require(isinstance(entries, list) and len(entries) == 3, "authorization entries drift")
    expected_record = yaml.safe_load(_authorization_record_bytes(artifacts))
    _require(record == expected_record, "authorization record content drift")
    return tuple(artifacts)


def build_phase2_manifest_set(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    proposal_path: Path = DEFAULT_PROPOSAL,
    alignment_path: Path = DEFAULT_ALIGNMENT,
    source_manifest_path: Path = DEFAULT_SOURCE_MANIFEST,
    *,
    verify_source_hashes: bool = True,
) -> tuple[OneRowManifestArtifact, ...]:
    """Create, then independently validate, the guarded one-row manifest set."""

    selections = load_reviewed_selections(
        proposal_path, verify_hash=verify_source_hashes
    )
    _cross_validate_sources(
        selections,
        alignment_path,
        source_manifest_path,
        verify_hashes=verify_source_hashes,
    )
    _require(
        not output_directory.is_symlink(),
        f"output directory must not be a symlink: {output_directory}",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    expected_basenames = {item.guarded_basename for item in selections}
    existing_entries = tuple(output_directory.iterdir())
    existing = {path.name for path in existing_entries}
    allowed = expected_basenames | {AUTHORIZATION_RECORD_BASENAME}
    _require(
        existing.issubset(allowed),
        f"unexpected output entries: {sorted(existing - allowed)}",
    )
    for path in existing_entries:
        _require(
            not path.is_symlink(),
            f"existing output entry must not be a symlink: {path}",
        )
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            raise OneRowManifestError(f"cannot stat existing output entry: {path}") from exc
        _require(
            stat.S_ISREG(mode),
            f"existing output entry must be a regular file: {path}",
        )

    artifacts: list[OneRowManifestArtifact] = []
    for selection in selections:
        path = output_directory / selection.guarded_basename
        payload = _manifest_bytes(selection)
        _write_new_or_require_equal(path, payload)
        artifacts.append(
            OneRowManifestArtifact(
                selection=selection,
                path=path,
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    record_path = output_directory / AUTHORIZATION_RECORD_BASENAME
    _write_new_or_require_equal(record_path, _authorization_record_bytes(artifacts))
    return validate_phase2_manifest_set(
        output_directory,
        proposal_path,
        alignment_path,
        source_manifest_path,
        verify_source_hashes=verify_source_hashes,
    )
