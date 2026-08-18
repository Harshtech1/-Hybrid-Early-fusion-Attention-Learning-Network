#!/usr/bin/env python3
"""Derive a metadata-only BRCA three-patient pilot proposal.

This script reads only the verified clean Omic ZIP and filtered GDC manifest.
It never downloads or opens an SVS. Multi-WSI patients are excluded rather
than resolved by an arbitrary tie-break.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence


OFFICIAL_RELEASE_COMMIT = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
EXPECTED_OMIC_SHA256 = (
    "4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8"
)
EXPECTED_MANIFEST_SHA256 = (
    "ac1b4d591ce255a7d4e6bde3778a041b107576693657ee296fe6eb36c4e79a92"
)
EXPECTED_OMIC_ROWS = 1022
EXPECTED_OMIC_PATIENTS = 956
EXPECTED_MANIFEST_ROWS = 1022
EXPECTED_SINGLETONS = 894
EXPECTED_MULTI_WSI_PATIENTS = 62
EXPECTED_FEATURE_DIMS = {"rna": 1558, "mutation": 21, "cnv": 1333}
EXPECTED_PROPOSAL = (
    (
        "Q25",
        224,
        "TCGA-LL-A6FP",
        "dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6",
        648_046_947,
        "75536393096ffd928bc35ec9503c3655",
    ),
    (
        "Q50",
        448,
        "TCGA-AR-A1AW",
        "5c1216f3-19ec-4d3c-9bb0-9bd740b79f62",
        975_626_387,
        "304509e03f26cbecc9aee4ea691c8e5a",
    ),
    (
        "Q75",
        671,
        "TCGA-E2-A154",
        "25aec062-60d1-446e-a1c6-0c79cc74a770",
        1_360_743_825,
        "a8c4b68fb6e0ab3e862efe3ed1fe10d7",
    ),
)
EXPECTED_PROPOSAL_TOTAL_BYTES = 2_984_417_159

DEFAULT_AUTHOR_REPO = Path(
    "/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet"
)
DEFAULT_OMIC_ZIP = DEFAULT_AUTHOR_REPO / (
    "data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
DEFAULT_WSI_MANIFEST = DEFAULT_AUTHOR_REPO / (
    "data/tcga/gdc_manifests/filtered/brca_wsi_manifest_filtered.txt"
)

CASE_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
SLIDE_RE = re.compile(
    r"^(?P<case>TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-"
    r"[0-9]{2}[A-Z]-[0-9]{2}-DX[1-9][0-9]*\.[A-Za-z0-9-]+\.svs$"
)
UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
    r"[0-9a-f]{4}-[0-9a-f]{12}$"
)
MD5_RE = re.compile(r"^[0-9a-f]{32}$")


class ProposalValidationError(RuntimeError):
    """Raised when a source or deterministic selection invariant fails."""


@dataclass(frozen=True)
class SingletonCandidate:
    patient_id: str
    omic_source_index: int
    omic_csv_line: int
    slide_id: str
    wsi_uuid: str
    wsi_size_bytes: int
    wsi_md5: str
    gdc_state: str


@dataclass(frozen=True)
class QuantileSelection:
    label: str
    quantile: float
    one_based_rank: int
    candidate: SingletonCandidate


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProposalValidationError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _case_from_slide(slide_id: str) -> str:
    match = SLIDE_RE.fullmatch(slide_id)
    _require(match is not None, f"invalid diagnostic-slide filename: {slide_id!r}")
    assert match is not None
    return match.group("case")


def _load_omic_rows(path: Path, *, verify_hash: bool = True) -> list[dict[str, str]]:
    _require(path.is_file(), f"BRCA Omic archive is missing: {path}")
    if verify_hash:
        actual = _sha256_file(path)
        _require(
            actual == EXPECTED_OMIC_SHA256,
            f"BRCA Omic SHA256 mismatch: expected {EXPECTED_OMIC_SHA256}, got {actual}",
        )

    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        _require(len(members) == 1, "BRCA Omic ZIP must contain exactly one file")
        with archive.open(members[0]) as binary_stream:
            text_stream = io.TextIOWrapper(
                binary_stream, encoding="utf-8-sig", newline=""
            )
            reader = csv.DictReader(text_stream)
            _require(reader.fieldnames is not None, "BRCA Omic CSV has no header")
            header = list(reader.fieldnames)
            for required in ("", "case_id", "slide_id"):
                _require(
                    header.count(required) == 1,
                    f"expected exactly one Omic column {required!r}",
                )
            dimensions = {
                "rna": sum(name.endswith("_rnaseq") for name in header),
                "mutation": sum(name.endswith("_mut") for name in header),
                "cnv": sum(name.endswith("_cnv") for name in header),
            }
            _require(
                dimensions == EXPECTED_FEATURE_DIMS,
                f"unexpected BRCA Omic dimensions: {dimensions!r}",
            )

            rows: list[dict[str, str]] = []
            for position, row in enumerate(reader):
                _require(None not in row, f"malformed Omic CSV row {position + 2}")
                patient_id = row["case_id"]
                _require(
                    CASE_RE.fullmatch(patient_id) is not None,
                    f"invalid case_id at Omic CSV row {position + 2}: {patient_id!r}",
                )
                _require(
                    _case_from_slide(row["slide_id"]) == patient_id,
                    f"case/slide mismatch at Omic CSV row {position + 2}",
                )
                enriched = dict(row)
                enriched["__source_position"] = str(position)
                enriched["__csv_line"] = str(position + 2)
                rows.append(enriched)

    if verify_hash:
        _require(len(rows) == EXPECTED_OMIC_ROWS, "unexpected BRCA Omic row count")
        _require(
            len({row["case_id"] for row in rows}) == EXPECTED_OMIC_PATIENTS,
            "unexpected BRCA Omic patient count",
        )
        _require(
            len({row["slide_id"] for row in rows}) == len(rows),
            "BRCA Omic slide_id values are not unique",
        )
    return rows


def _load_wsi_rows(path: Path, *, verify_hash: bool = True) -> list[dict[str, str]]:
    _require(path.is_file(), f"BRCA filtered WSI manifest is missing: {path}")
    if verify_hash:
        actual = _sha256_file(path)
        _require(
            actual == EXPECTED_MANIFEST_SHA256,
            "BRCA filtered WSI manifest SHA256 mismatch: "
            f"expected {EXPECTED_MANIFEST_SHA256}, got {actual}",
        )

    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        _require(
            reader.fieldnames == ["id", "filename", "md5", "size", "state"],
            f"unexpected GDC manifest schema: {reader.fieldnames!r}",
        )
        rows: list[dict[str, str]] = []
        for position, row in enumerate(reader):
            _require(None not in row, f"malformed GDC manifest row {position + 2}")
            _case_from_slide(row["filename"])
            _require(UUID_RE.fullmatch(row["id"]) is not None, "invalid GDC UUID")
            _require(MD5_RE.fullmatch(row["md5"]) is not None, "invalid GDC MD5")
            try:
                size = int(row["size"])
            except ValueError as exc:
                raise ProposalValidationError("invalid GDC size") from exc
            _require(size > 0, "GDC WSI size must be positive")
            _require(row["state"] == "released", "GDC object is not released")
            enriched = dict(row)
            enriched["__case_id"] = _case_from_slide(row["filename"])
            rows.append(enriched)

    if verify_hash:
        _require(len(rows) == EXPECTED_MANIFEST_ROWS, "unexpected BRCA WSI row count")
    for key in ("id", "filename"):
        _require(
            len({row[key] for row in rows}) == len(rows),
            f"duplicate GDC {key} values",
        )
    return rows


def build_singleton_candidates(
    omic_zip: Path,
    wsi_manifest: Path,
    *,
    verify_hashes: bool = True,
) -> tuple[list[SingletonCandidate], int]:
    """Return exact singleton matches and the number of deferred multi cases."""

    omic_rows = _load_omic_rows(omic_zip, verify_hash=verify_hashes)
    wsi_rows = _load_wsi_rows(wsi_manifest, verify_hash=verify_hashes)
    omic_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    wsi_by_patient: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in omic_rows:
        omic_by_patient[row["case_id"]].append(row)
    for row in wsi_rows:
        wsi_by_patient[row["__case_id"]].append(row)

    common = set(omic_by_patient) & set(wsi_by_patient)
    singleton: list[SingletonCandidate] = []
    multi_count = 0
    for patient_id in sorted(common):
        omic_group = omic_by_patient[patient_id]
        wsi_group = wsi_by_patient[patient_id]
        if len(omic_group) == len(wsi_group) == 1:
            omic = omic_group[0]
            wsi = wsi_group[0]
            _require(
                omic["slide_id"] == wsi["filename"],
                f"singleton exact slide mismatch for {patient_id}",
            )
            singleton.append(
                SingletonCandidate(
                    patient_id=patient_id,
                    omic_source_index=int(omic["__source_position"]),
                    omic_csv_line=int(omic["__csv_line"]),
                    slide_id=omic["slide_id"],
                    wsi_uuid=wsi["id"],
                    wsi_size_bytes=int(wsi["size"]),
                    wsi_md5=wsi["md5"],
                    gdc_state=wsi["state"],
                )
            )
        elif len(wsi_group) > 1:
            multi_count += 1

    singleton.sort(
        key=lambda item: (item.wsi_size_bytes, item.patient_id, item.wsi_uuid)
    )
    if verify_hashes:
        _require(len(singleton) == EXPECTED_SINGLETONS, "unexpected singleton count")
        _require(
            multi_count == EXPECTED_MULTI_WSI_PATIENTS,
            "unexpected multi-WSI patient count",
        )
    return singleton, multi_count


def nearest_observed_index(length: int, quantile: float) -> int:
    """Map an inclusive quantile to the nearest observed item, ties upward."""

    _require(length > 0, "cannot select from an empty cohort")
    _require(0.0 <= quantile <= 1.0, "quantile must be in [0, 1]")
    return math.floor((length - 1) * quantile + 0.5)


def select_quantile_representatives(
    candidates: Sequence[SingletonCandidate],
    quantiles: Iterable[tuple[str, float]] = (
        ("Q25", 0.25),
        ("Q50", 0.50),
        ("Q75", 0.75),
    ),
) -> list[QuantileSelection]:
    ordered = sorted(
        candidates,
        key=lambda item: (item.wsi_size_bytes, item.patient_id, item.wsi_uuid),
    )
    selections = [
        QuantileSelection(
            label=label,
            quantile=quantile,
            one_based_rank=nearest_observed_index(len(ordered), quantile) + 1,
            candidate=ordered[nearest_observed_index(len(ordered), quantile)],
        )
        for label, quantile in quantiles
    ]
    _require(
        len({item.candidate.patient_id for item in selections}) == len(selections),
        "quantile policy selected a duplicate patient",
    )
    return selections


def validate_official_proposal(selections: Sequence[QuantileSelection]) -> None:
    """Require the generated proposal to match the reviewed frozen identities."""

    actual = tuple(
        (
            item.label,
            item.one_based_rank,
            item.candidate.patient_id,
            item.candidate.wsi_uuid,
            item.candidate.wsi_size_bytes,
            item.candidate.wsi_md5,
        )
        for item in selections
    )
    _require(
        actual == EXPECTED_PROPOSAL,
        f"official BRCA three-patient proposal changed: {actual!r}",
    )
    _require(
        sum(item.candidate.wsi_size_bytes for item in selections)
        == EXPECTED_PROPOSAL_TOTAL_BYTES,
        "official BRCA three-patient proposal byte total changed",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--omic", type=Path, default=DEFAULT_OMIC_ZIP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_WSI_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    candidates, multi_count = build_singleton_candidates(args.omic, args.manifest)
    selections = select_quantile_representatives(candidates)
    validate_official_proposal(selections)
    payload = {
        "status": "PROPOSED_BRCA_PILOT_NOT_AUTHORIZED",
        "official_healnet_commit": OFFICIAL_RELEASE_COMMIT,
        "omic_sha256": EXPECTED_OMIC_SHA256,
        "filtered_wsi_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "singleton_patient_count": len(candidates),
        "multi_wsi_patient_count": multi_count,
        "selection_policy": {
            "sort": ["wsi_size_bytes", "patient_id", "wsi_uuid"],
            "quantiles": [0.25, 0.50, 0.75],
            "index": "floor((n - 1) * q + 0.5)",
            "tie_policy": "nearest observed rank, half upward",
        },
        "selected": [
            {
                "label": item.label,
                "quantile": item.quantile,
                "one_based_rank": item.one_based_rank,
                **asdict(item.candidate),
            }
            for item in selections
        ],
        "total_declared_wsi_bytes": sum(
            item.candidate.wsi_size_bytes for item in selections
        ),
        "download_authorized": False,
        "wsi_opened": False,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
