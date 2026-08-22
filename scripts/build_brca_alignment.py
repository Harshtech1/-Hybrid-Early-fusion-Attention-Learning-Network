#!/usr/bin/env python3
"""Build the deterministic BRCA Omic/WSI alignment without opening WSI files.

The only WSI input to this script is the small, tab-delimited GDC manifest.
Matching is a full outer join on the exact ``(case_id, slide_id)`` pair; the
source CSV ordinal is retained for provenance but is never used for matching.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import re
import tempfile
import uuid
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent

DEFAULT_OMIC_ARCHIVE = (
    WORKSPACE_ROOT
    / "Author_Official_Repo_directery"
    / "healnet"
    / "data"
    / "tcga"
    / "omic"
    / "tcga_brca_all_clean.csv.zip"
)
DEFAULT_WSI_MANIFEST = (
    WORKSPACE_ROOT
    / "healnet"
    / "data"
    / "tcga"
    / "gdc_manifests"
    / "filtered"
    / "brca_wsi_manifest_filtered.txt"
)
DEFAULT_REPORTS_DIRECTORY = REPOSITORY_ROOT / "reports"

EXPECTED_OMIC_ARCHIVE_SHA256 = (
    "4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8"
)
EXPECTED_OMIC_ARCHIVE_BYTES = 4_081_277
EXPECTED_OMIC_MEMBER_NAME = "./tcga_brca_all_clean.csv.zip"
EXPECTED_OMIC_MEMBER_SHA256 = (
    "052637f2a69c515812796d9638566cb75299b6a3571dbdc5363496f12665027d"
)
EXPECTED_OMIC_MEMBER_BYTES = 15_021_018
EXPECTED_WSI_MANIFEST_SHA256 = (
    "ac1b4d591ce255a7d4e6bde3778a041b107576693657ee296fe6eb36c4e79a92"
)
EXPECTED_WSI_MANIFEST_BYTES = 157_900

# Git identities are recorded alongside content hashes. The archive is an LFS
# object in the current official checkout; the manifest is the same Git blob in
# both official checkouts used for this audit.
OMIC_SOURCE_COMMIT = "90459b65a3d4a4ef9fd405671c457b5a1163cc7d"
OMIC_POINTER_GIT_BLOB = "eed96279b655023feb891970b327d9f2328023f6"
OMIC_LFS_OID = EXPECTED_OMIC_ARCHIVE_SHA256
MANIFEST_SOURCE_COMMIT = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
MANIFEST_SOURCE_TAG = "v0.1.0"
MANIFEST_GIT_BLOB = "03a7bc71ff1cf072759b2af05d6e405891f486ec"

EXPECTED_COUNTS = {
    "omic_rows": 1_022,
    "omic_patients": 956,
    "wsi_rows": 1_022,
    "wsi_patients": 956,
    "exact_match_rows": 1_022,
    "keep_rows": 894,
    "keep_patients": 894,
    "ambiguous_rows": 128,
    "ambiguous_patients": 62,
    "wsi_only_rows": 0,
    "wsi_only_patients": 0,
    "omic_only_rows": 0,
    "omic_only_patients": 0,
    "slide_key_mismatch_patients": 0,
    "rna_features": 1_558,
    "mutation_features": 21,
    "cnv_features": 1_333,
}

ALIGNMENT_FIELDNAMES = [
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

DOWNLOAD_FIELDNAMES = [
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

CASE_FROM_SLIDE_RE = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-")
MD5_RE = re.compile(r"^[0-9a-f]{32}$")


class AlignmentError(ValueError):
    """Raised when a source or alignment violates the frozen contract."""


@dataclass(frozen=True)
class SourceIdentity:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class OmicSource:
    identity: SourceIdentity
    member_name: str
    member_size_bytes: int
    member_sha256: str
    rna_features: int
    mutation_features: int
    cnv_features: int


@dataclass(frozen=True)
class OmicRow:
    case_id: str
    slide_id: str
    source_row_id: str


@dataclass(frozen=True)
class WsiRow:
    case_id: str
    slide_id: str
    gdc_file_uuid: str
    md5: str
    size_bytes: int
    state: str


@dataclass(frozen=True)
class AlignmentRow:
    case_id: str
    slide_id: str
    alignment_status: str
    alignment_reason: str
    patient_wsi_count: int
    patient_omic_count: int
    omic_source_row_id: str
    gdc_file_uuid: str
    gdc_md5: str
    gdc_size_bytes: str
    gdc_state: str


@dataclass(frozen=True)
class AlignmentSummary:
    omic_rows: int
    omic_patients: int
    wsi_rows: int
    wsi_patients: int
    exact_match_rows: int
    keep_rows: int
    keep_patients: int
    ambiguous_rows: int
    ambiguous_patients: int
    wsi_only_rows: int
    wsi_only_patients: int
    omic_only_rows: int
    omic_only_patients: int
    slide_key_mismatch_patients: int
    rna_features: int
    mutation_features: int
    cnv_features: int
    total_wsi_bytes: int
    keep_wsi_bytes: int
    ambiguous_wsi_bytes: int


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _source_identity(path: Path) -> SourceIdentity:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise AlignmentError(f"source is not a file: {resolved}")
    return SourceIdentity(
        path=str(resolved),
        size_bytes=resolved.stat().st_size,
        sha256=sha256_file(resolved),
    )


def case_id_from_slide(slide_id: str) -> str:
    match = CASE_FROM_SLIDE_RE.match(slide_id)
    if match is None:
        raise AlignmentError(f"cannot derive TCGA case_id from slide_id: {slide_id!r}")
    return match.group(1)


def _require_columns(actual: Sequence[str] | None, required: set[str], source: str) -> None:
    if actual is None:
        raise AlignmentError(f"{source} has no header")
    missing = sorted(required.difference(actual))
    if missing:
        raise AlignmentError(f"{source} missing required columns: {missing}")


def read_omic_archive(path: Path) -> tuple[list[OmicRow], OmicSource]:
    identity = _source_identity(path)
    try:
        with zipfile.ZipFile(path) as archive:
            members = [item for item in archive.infolist() if not item.is_dir()]
            if len(members) != 1:
                raise AlignmentError(
                    f"Omic archive must contain exactly one file; found {len(members)}"
                )
            member = members[0]
            payload = archive.read(member)
    except (OSError, zipfile.BadZipFile) as exc:
        raise AlignmentError(f"cannot read Omic archive {path}: {exc}") from exc

    member_sha256 = hashlib.sha256(payload).hexdigest()
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise AlignmentError("Omic archive member is not UTF-8 CSV") from exc

    reader = csv.DictReader(io.StringIO(text, newline=""))
    _require_columns(reader.fieldnames, {"", "case_id", "slide_id"}, "Omic CSV")
    fieldnames = list(reader.fieldnames or [])
    rows: list[OmicRow] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_source_ids: set[str] = set()

    for line_number, raw in enumerate(reader, start=2):
        case_id = (raw.get("case_id") or "").strip()
        slide_id = (raw.get("slide_id") or "").strip()
        source_row_id = (raw.get("") or "").strip()
        if not case_id or not slide_id or not source_row_id:
            raise AlignmentError(
                f"Omic CSV line {line_number} has an empty identity field"
            )
        derived_case_id = case_id_from_slide(slide_id)
        if derived_case_id != case_id:
            raise AlignmentError(
                f"Omic CSV line {line_number} case/slide mismatch: "
                f"{case_id!r} != {derived_case_id!r}"
            )
        key = (case_id, slide_id)
        if key in seen_keys:
            raise AlignmentError(f"duplicate Omic exact key: {key!r}")
        if source_row_id in seen_source_ids:
            raise AlignmentError(f"duplicate Omic source row id: {source_row_id!r}")
        seen_keys.add(key)
        seen_source_ids.add(source_row_id)
        rows.append(OmicRow(case_id, slide_id, source_row_id))

    rows.sort(key=lambda row: (row.case_id, row.slide_id, row.source_row_id))
    return rows, OmicSource(
        identity=identity,
        member_name=member.filename,
        member_size_bytes=len(payload),
        member_sha256=member_sha256,
        rna_features=sum(name.endswith("_rnaseq") for name in fieldnames),
        mutation_features=sum(name.endswith("_mut") for name in fieldnames),
        cnv_features=sum(name.endswith("_cnv") for name in fieldnames),
    )


def read_wsi_manifest(path: Path) -> tuple[list[WsiRow], SourceIdentity]:
    identity = _source_identity(path)
    rows: list[WsiRow] = []
    seen_keys: set[tuple[str, str]] = set()
    seen_uuids: set[str] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require_columns(
            reader.fieldnames,
            {"id", "filename", "md5", "size", "state"},
            "WSI manifest",
        )
        for line_number, raw in enumerate(reader, start=2):
            slide_id = (raw.get("filename") or "").strip()
            case_id = case_id_from_slide(slide_id)
            gdc_file_uuid = (raw.get("id") or "").strip().lower()
            md5 = (raw.get("md5") or "").strip().lower()
            state = (raw.get("state") or "").strip()
            try:
                parsed_uuid = str(uuid.UUID(gdc_file_uuid))
            except ValueError as exc:
                raise AlignmentError(
                    f"WSI manifest line {line_number} has invalid UUID: {gdc_file_uuid!r}"
                ) from exc
            if parsed_uuid != gdc_file_uuid:
                raise AlignmentError(
                    f"WSI manifest line {line_number} has non-canonical UUID"
                )
            if MD5_RE.fullmatch(md5) is None:
                raise AlignmentError(
                    f"WSI manifest line {line_number} has invalid MD5: {md5!r}"
                )
            try:
                size_bytes = int((raw.get("size") or "").strip())
            except ValueError as exc:
                raise AlignmentError(
                    f"WSI manifest line {line_number} has invalid size"
                ) from exc
            if size_bytes <= 0:
                raise AlignmentError(
                    f"WSI manifest line {line_number} has non-positive size"
                )
            key = (case_id, slide_id)
            if key in seen_keys:
                raise AlignmentError(f"duplicate WSI exact key: {key!r}")
            if gdc_file_uuid in seen_uuids:
                raise AlignmentError(f"duplicate GDC file UUID: {gdc_file_uuid!r}")
            seen_keys.add(key)
            seen_uuids.add(gdc_file_uuid)
            rows.append(
                WsiRow(
                    case_id=case_id,
                    slide_id=slide_id,
                    gdc_file_uuid=gdc_file_uuid,
                    md5=md5,
                    size_bytes=size_bytes,
                    state=state,
                )
            )

    rows.sort(key=lambda row: (row.case_id, row.slide_id, row.gdc_file_uuid))
    return rows, identity


def align_exact_keys(
    omic_rows: Iterable[OmicRow],
    wsi_rows: Iterable[WsiRow],
) -> list[AlignmentRow]:
    omic_by_key: dict[tuple[str, str], OmicRow] = {}
    wsi_by_key: dict[tuple[str, str], WsiRow] = {}
    omic_case_counts: Counter[str] = Counter()
    wsi_case_counts: Counter[str] = Counter()

    for row in omic_rows:
        key = (row.case_id, row.slide_id)
        if key in omic_by_key:
            raise AlignmentError(f"duplicate Omic exact key: {key!r}")
        omic_by_key[key] = row
        omic_case_counts[row.case_id] += 1
    for row in wsi_rows:
        key = (row.case_id, row.slide_id)
        if key in wsi_by_key:
            raise AlignmentError(f"duplicate WSI exact key: {key!r}")
        wsi_by_key[key] = row
        wsi_case_counts[row.case_id] += 1

    result: list[AlignmentRow] = []
    for case_id, slide_id in sorted(set(omic_by_key).union(wsi_by_key)):
        key = (case_id, slide_id)
        omic = omic_by_key.get(key)
        wsi = wsi_by_key.get(key)
        omic_count = omic_case_counts[case_id]
        wsi_count = wsi_case_counts[case_id]
        if omic is not None and wsi is not None:
            if omic_count == 1 and wsi_count == 1:
                status = "KEEP"
                reason = "EXACT_SINGLETON_CASE_AND_SLIDE_MATCH"
            else:
                status = "AMBIGUOUS"
                reason = "EXACT_SLIDE_MATCH_BUT_MULTIPLE_ROWS_FOR_CASE"
        elif wsi is not None:
            status = "WSI_ONLY"
            reason = "NO_EXACT_OMIC_CASE_AND_SLIDE_MATCH"
        else:
            status = "OMIC_ONLY"
            reason = "NO_EXACT_WSI_CASE_AND_SLIDE_MATCH"

        result.append(
            AlignmentRow(
                case_id=case_id,
                slide_id=slide_id,
                alignment_status=status,
                alignment_reason=reason,
                patient_wsi_count=wsi_count,
                patient_omic_count=omic_count,
                omic_source_row_id=omic.source_row_id if omic is not None else "",
                gdc_file_uuid=wsi.gdc_file_uuid if wsi is not None else "",
                gdc_md5=wsi.md5 if wsi is not None else "",
                gdc_size_bytes=str(wsi.size_bytes) if wsi is not None else "",
                gdc_state=wsi.state if wsi is not None else "",
            )
        )
    return result


def summarize_alignment(
    omic_rows: Sequence[OmicRow],
    wsi_rows: Sequence[WsiRow],
    alignment_rows: Sequence[AlignmentRow],
    omic_source: OmicSource,
) -> AlignmentSummary:
    omic_cases = {row.case_id for row in omic_rows}
    wsi_cases = {row.case_id for row in wsi_rows}
    omic_slides_by_case: dict[str, set[str]] = {
        case_id: {row.slide_id for row in omic_rows if row.case_id == case_id}
        for case_id in omic_cases
    }
    wsi_slides_by_case: dict[str, set[str]] = {
        case_id: {row.slide_id for row in wsi_rows if row.case_id == case_id}
        for case_id in wsi_cases
    }
    status_cases = {
        status: {
            row.case_id for row in alignment_rows if row.alignment_status == status
        }
        for status in ("KEEP", "AMBIGUOUS", "WSI_ONLY", "OMIC_ONLY")
    }
    status_rows = Counter(row.alignment_status for row in alignment_rows)
    exact_match_rows = status_rows["KEEP"] + status_rows["AMBIGUOUS"]
    size_by_status = Counter(
        {
            status: sum(
                int(row.gdc_size_bytes)
                for row in alignment_rows
                if row.alignment_status == status and row.gdc_size_bytes
            )
            for status in ("KEEP", "AMBIGUOUS", "WSI_ONLY")
        }
    )
    return AlignmentSummary(
        omic_rows=len(omic_rows),
        omic_patients=len(omic_cases),
        wsi_rows=len(wsi_rows),
        wsi_patients=len(wsi_cases),
        exact_match_rows=exact_match_rows,
        keep_rows=status_rows["KEEP"],
        keep_patients=len(status_cases["KEEP"]),
        ambiguous_rows=status_rows["AMBIGUOUS"],
        ambiguous_patients=len(status_cases["AMBIGUOUS"]),
        wsi_only_rows=status_rows["WSI_ONLY"],
        wsi_only_patients=len(wsi_cases.difference(omic_cases)),
        omic_only_rows=status_rows["OMIC_ONLY"],
        omic_only_patients=len(omic_cases.difference(wsi_cases)),
        slide_key_mismatch_patients=sum(
            omic_slides_by_case[case_id] != wsi_slides_by_case[case_id]
            for case_id in omic_cases.intersection(wsi_cases)
        ),
        rna_features=omic_source.rna_features,
        mutation_features=omic_source.mutation_features,
        cnv_features=omic_source.cnv_features,
        total_wsi_bytes=sum(row.size_bytes for row in wsi_rows),
        keep_wsi_bytes=size_by_status["KEEP"],
        ambiguous_wsi_bytes=size_by_status["AMBIGUOUS"],
    )


def validate_official_sources(
    omic_source: OmicSource,
    manifest_source: SourceIdentity,
) -> None:
    expected_actual = [
        ("Omic archive SHA256", EXPECTED_OMIC_ARCHIVE_SHA256, omic_source.identity.sha256),
        ("Omic archive bytes", EXPECTED_OMIC_ARCHIVE_BYTES, omic_source.identity.size_bytes),
        ("Omic member name", EXPECTED_OMIC_MEMBER_NAME, omic_source.member_name),
        ("Omic member SHA256", EXPECTED_OMIC_MEMBER_SHA256, omic_source.member_sha256),
        ("Omic member bytes", EXPECTED_OMIC_MEMBER_BYTES, omic_source.member_size_bytes),
        ("WSI manifest SHA256", EXPECTED_WSI_MANIFEST_SHA256, manifest_source.sha256),
        ("WSI manifest bytes", EXPECTED_WSI_MANIFEST_BYTES, manifest_source.size_bytes),
    ]
    failures = [
        f"{label}: expected {expected!r}, got {actual!r}"
        for label, expected, actual in expected_actual
        if actual != expected
    ]
    if failures:
        raise AlignmentError("official source identity mismatch:\n" + "\n".join(failures))


def validate_official_counts(summary: AlignmentSummary) -> None:
    values = asdict(summary)
    failures = [
        f"{name}: expected {expected}, got {values[name]}"
        for name, expected in EXPECTED_COUNTS.items()
        if values[name] != expected
    ]
    if failures:
        raise AlignmentError("official BRCA cohort count mismatch:\n" + "\n".join(failures))


def render_alignment_csv(rows: Sequence[AlignmentRow]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=ALIGNMENT_FIELDNAMES,
        lineterminator="\n",
    )
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (item.case_id, item.slide_id)):
        writer.writerow(
            {
                "case_id": row.case_id,
                "slide_id": row.slide_id,
                "alignment_status": row.alignment_status,
                "alignment_reason": row.alignment_reason,
                "patient_wsi_count": row.patient_wsi_count,
                "patient_omic_count": row.patient_omic_count,
                "omic_source_row_id": row.omic_source_row_id,
                "id": row.gdc_file_uuid,
                "filename": row.slide_id if row.gdc_file_uuid else "",
                "md5": row.gdc_md5,
                "size": row.gdc_size_bytes,
                "state": row.gdc_state,
            }
        )
    return output.getvalue()


def _candidate_disposition(status: str) -> str:
    return {
        "KEEP": "ELIGIBLE_SINGLETON_NOT_SELECTED",
        "AMBIGUOUS": "INELIGIBLE_MULTI_WSI_AMBIGUOUS",
        "WSI_ONLY": "INELIGIBLE_NO_EXACT_OMIC_MATCH",
    }[status]


def render_download_plan_tsv(rows: Sequence[AlignmentRow]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output,
        fieldnames=DOWNLOAD_FIELDNAMES,
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in sorted(rows, key=lambda item: (item.case_id, item.slide_id)):
        if not row.gdc_file_uuid:
            continue
        writer.writerow(
            {
                "case_id": row.case_id,
                "slide_id": row.slide_id,
                "id": row.gdc_file_uuid,
                "filename": row.slide_id,
                "md5": row.gdc_md5,
                "size": row.gdc_size_bytes,
                "state": row.gdc_state,
                "alignment_status": row.alignment_status,
                "candidate_disposition": _candidate_disposition(row.alignment_status),
                "download_status": "NOT_AUTHORIZED",
                "authorization_basis": "AWAIT_EXACT_THREE_PATIENT_CONFIRMATION",
            }
        )
    return output.getvalue()


def _format_bytes_decimal(value: int) -> str:
    return f"{value / 1_000_000_000:.3f} GB"


def render_source_report(
    *,
    omic_source: OmicSource,
    manifest_source: SourceIdentity,
    summary: AlignmentSummary,
    alignment_identity: SourceIdentity,
    download_identity: SourceIdentity,
) -> str:
    return f"""# BRCA source of truth and alignment gate

BRCA is the approved next cohort; the frozen BLCA pilot remains the engineering
reference. This approval does **not** authorize WSI acquisition. This audit is
CPU-only: no WSI payload was downloaded, opened, or inspected, and no feature
extraction or training was performed.

## Authoritative source identities

| Input | Exact local path | Bytes | SHA256 | Repository identity |
|---|---|---:|---|---|
| Materialized BRCA Omic archive | `{omic_source.identity.path}` | {omic_source.identity.size_bytes:,} | `{omic_source.identity.sha256}` | official checkout commit `{OMIC_SOURCE_COMMIT}`; Git LFS pointer blob `{OMIC_POINTER_GIT_BLOB}`; LFS OID `sha256:{OMIC_LFS_OID}` |
| Filtered BRCA WSI manifest | `{manifest_source.path}` | {manifest_source.size_bytes:,} | `{manifest_source.sha256}` | official `{MANIFEST_SOURCE_TAG}` commit `{MANIFEST_SOURCE_COMMIT}`; Git blob `{MANIFEST_GIT_BLOB}` |

The Omic ZIP contains exactly one CSV member:

| Member | Uncompressed bytes | SHA256 |
|---|---:|---|
| `{omic_source.member_name}` | {omic_source.member_size_bytes:,} | `{omic_source.member_sha256}` |

All source sizes and hashes are enforced by
`scripts/build_brca_alignment.py`; generation stops on any mismatch.

## Deterministic identity policy

- The only join key is the exact pair `(case_id, slide_id)`.
- Manifest `case_id` is derived from the first three TCGA barcode fields in
  the complete manifest filename; it must equal the Omic `case_id`.
- The unnamed Omic CSV ordinal is retained as `omic_source_row_id` only for
  traceability. It is never a join key and CSV/manifest row order is ignored.
- A case with exactly one WSI row and one exact Omic row is `KEEP`.
- Every exactly matched row belonging to a multi-WSI case is `AMBIGUOUS`; no
  slide is selected implicitly.
- A full outer join preserves `WSI_ONLY` and `OMIC_ONLY` records if present.
- Rows are sorted lexicographically by `case_id`, then complete `slide_id`.

## Verified cohort

| Measure | Count |
|---|---:|
| Omic rows | {summary.omic_rows:,} |
| Omic patients | {summary.omic_patients:,} |
| Filtered-manifest WSI rows | {summary.wsi_rows:,} |
| Filtered-manifest WSI patients | {summary.wsi_patients:,} |
| Exact `(case_id, slide_id)` matches | {summary.exact_match_rows:,} |
| `KEEP` singleton patients / rows | {summary.keep_patients:,} / {summary.keep_rows:,} |
| `AMBIGUOUS` multi-WSI patients / rows | {summary.ambiguous_patients:,} / {summary.ambiguous_rows:,} |
| WSI-only patients / rows | {summary.wsi_only_patients:,} / {summary.wsi_only_rows:,} |
| Omic-only patients / rows | {summary.omic_only_patients:,} / {summary.omic_only_rows:,} |
| Shared cases with different slide-key sets | {summary.slide_key_mismatch_patients:,} |

The 62 ambiguous patients have multiple exact WSI/Omic rows and remain
excluded from the singleton pilot candidate pool. There are 894 unambiguous
singleton candidates. No patient or slide is chosen by source row position.

## Omic modality contract

| Modality | Feature columns |
|---|---:|
| RNA (`*_rnaseq`) | {summary.rna_features:,} |
| Mutation (`*_mut`) | {summary.mutation_features:,} |
| CNV (`*_cnv`) | {summary.cnv_features:,} |

The archive has {summary.omic_rows:,} rows across {summary.omic_patients:,}
patients. CNV is embedded in the Omic table; it is not a separate file.

## Storage and authorization

| WSI set | Manifest-declared bytes | Decimal size |
|---|---:|---:|
| All filtered rows | {summary.total_wsi_bytes:,} | {_format_bytes_decimal(summary.total_wsi_bytes)} |
| `KEEP` singleton rows | {summary.keep_wsi_bytes:,} | {_format_bytes_decimal(summary.keep_wsi_bytes)} |
| `AMBIGUOUS` rows | {summary.ambiguous_wsi_bytes:,} | {_format_bytes_decimal(summary.ambiguous_wsi_bytes)} |

`reports/brca_download_plan.tsv` records every manifest row, including its
alignment disposition, but every `download_status` is `NOT_AUTHORIZED`.
Acquisition remains blocked until the exact three-patient pilot is confirmed.
Bulk download is not authorized.

## Generated artifact identities

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `reports/brca_row_level_alignment.csv` | {alignment_identity.size_bytes:,} | `{alignment_identity.sha256}` |
| `reports/brca_download_plan.tsv` | {download_identity.size_bytes:,} | `{download_identity.sha256}` |

Final status: `BRCA_ALIGNMENT_READY__WSI_DOWNLOAD_NOT_AUTHORIZED`
"""


def _atomic_write_text(path: Path, content: str) -> None:
    if not path.parent.is_dir():
        raise AlignmentError(f"output directory does not exist: {path.parent}")
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as stream:
        temporary_path = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_path, path)


def generate_outputs(
    *,
    omic_archive: Path,
    wsi_manifest: Path,
    reports_directory: Path,
) -> AlignmentSummary:
    if not reports_directory.is_dir():
        raise AlignmentError(f"reports directory does not exist: {reports_directory}")

    omic_rows, omic_source = read_omic_archive(omic_archive)
    wsi_rows, manifest_source = read_wsi_manifest(wsi_manifest)
    validate_official_sources(omic_source, manifest_source)

    alignment_rows = align_exact_keys(omic_rows, wsi_rows)
    summary = summarize_alignment(omic_rows, wsi_rows, alignment_rows, omic_source)
    validate_official_counts(summary)

    alignment_path = reports_directory / "brca_row_level_alignment.csv"
    download_path = reports_directory / "brca_download_plan.tsv"
    source_report_path = reports_directory / "brca_source_of_truth.md"
    _atomic_write_text(alignment_path, render_alignment_csv(alignment_rows))
    _atomic_write_text(download_path, render_download_plan_tsv(alignment_rows))

    alignment_identity = _source_identity(alignment_path)
    download_identity = _source_identity(download_path)
    _atomic_write_text(
        source_report_path,
        render_source_report(
            omic_source=omic_source,
            manifest_source=manifest_source,
            summary=summary,
            alignment_identity=alignment_identity,
            download_identity=download_identity,
        ),
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pinned, deterministic BRCA Omic/filtered-manifest alignment "
            "artifacts without reading WSI payloads."
        )
    )
    parser.add_argument("--omic-archive", type=Path, default=DEFAULT_OMIC_ARCHIVE)
    parser.add_argument("--wsi-manifest", type=Path, default=DEFAULT_WSI_MANIFEST)
    parser.add_argument(
        "--reports-directory",
        type=Path,
        default=DEFAULT_REPORTS_DIRECTORY,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = generate_outputs(
        omic_archive=args.omic_archive,
        wsi_manifest=args.wsi_manifest,
        reports_directory=args.reports_directory,
    )
    print(
        "BRCA alignment ready: "
        f"{summary.keep_patients} KEEP singleton patients, "
        f"{summary.ambiguous_patients} AMBIGUOUS multi-WSI patients; "
        "all downloads NOT_AUTHORIZED"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
