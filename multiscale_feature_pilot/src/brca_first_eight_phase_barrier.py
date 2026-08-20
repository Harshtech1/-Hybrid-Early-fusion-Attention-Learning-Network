"""Validator for the CPU-only first-eight phase-barrier scheduling policy."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import yaml


class PhaseBarrierError(ValueError):
    """Raised when block identity, scheduling, or an operation lock drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PhaseBarrierError(message)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_policy(root: Path, policy_path: Path | None = None) -> Mapping[str, object]:
    path = policy_path or root / "multiscale_feature_pilot/config/brca_first_eight_phase_barrier_policy.yaml"
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(document["status"] == "CPU_PHASE_BARRIER_AUTHORIZED_GPU_AND_PIXEL_STAGES_LOCKED", "policy status drift")
    scheduling = document["scheduling"]
    _require(scheduling["download_concurrency"] == 1, "download concurrency must remain one")
    _require(scheduling["cpu_patient_preparation_workers"] == 2, "CPU worker limit must be exactly two")
    _require(scheduling["gpu_concurrency"] == 0, "GPU must remain locked")
    _require(scheduling["transaction_size_patients"] == 1, "patient transactions must remain singleton")
    _require(scheduling["download_order"] == [f"P{index:04d}" for index in range(2, 9)], "download order drift")
    _require(scheduling["stop_entire_block_on_first_identity_hash_authorization_or_stage_drift"] is True, "stop-on-drift must be enabled")
    locks = document["operation_locks"]
    _require(set(locks) == {"read_region_or_pixel_access", "mask_or_coordinate_generation", "patch_reads", "gpu_or_cuda", "feature_extraction", "healnet_execution", "compact_feature_publication", "drive_operations", "raw_or_existing_file_deletion", "cohort_patients_outside_p0001_p0008", "training"}, "operation-lock surface drift")
    _require(all(locks.values()), "every prohibited operation must remain locked")

    binding = document["cohort_binding"]
    proposal = root / binding["source_tsv"]
    _require(_sha(proposal) == binding["source_tsv_sha256"], "first-eight source TSV drift")
    with proposal.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    _require(len(rows) == 8, "first-eight source must contain exactly eight rows")
    _require([int(row["cohort_index"]) for row in rows] == list(range(1, 9)), "cohort-order drift")
    _require(sum(int(row["size_bytes"]) for row in rows) == binding["exact_raw_wsi_bytes"], "raw-byte total drift")
    _require(sum(int(row["size_bytes"]) for row in rows[1:]) == binding["p0002_p0008_raw_wsi_bytes"], "P0002-P0008 byte total drift")
    manifests = document["manifests"]
    _require(list(manifests) == [f"P{index:04d}" for index in range(1, 9)], "manifest labels drift")
    directory = root / binding["manifests_directory"]
    files = sorted(directory.glob("*.REQUEST_ONLY.gdc.tsv"))
    _require(len(files) == 8, "exactly eight request-only manifests required")
    for label, row, manifest in zip(manifests, rows, files, strict=True):
        record = manifests[label]
        _require(record["cohort_index"] == int(row["cohort_index"]), f"{label} cohort index drift")
        _require(record["patient_id"] == row["patient_id"] and record["gdc_uuid"] == row["gdc_uuid"], f"{label} identity drift")
        _require(row["gdc_uuid"] in manifest.name, f"{label} manifest filename drift")
        _require(_sha(manifest) == record["sha256"], f"{label} manifest SHA256 drift")
    return document


def validate_active_work(active: Sequence[Mapping[str, object]]) -> None:
    """Fail closed on an attempted runtime schedule outside the CPU barrier."""

    _require(len(active) <= 2, "more than two CPU patient workers are active")
    labels = [item.get("patient") for item in active]
    _require(len(labels) == len(set(labels)), "one patient cannot occupy two active workers")
    _require(all(label in {f"P{index:04d}" for index in range(1, 9)} for label in labels), "patient outside first block")
    downloads = sum(item.get("operation") == "DOWNLOAD" for item in active)
    _require(downloads <= 1, "download concurrency exceeds one")
    permitted = {"DOWNLOAD", "RAW_VERIFY", "OMIC_REMATCH", "HEADER_ONLY", "POLICY_DESIGN"}
    _require(all(item.get("operation") in permitted for item in active), "operation is outside CPU phase")


__all__ = ["PhaseBarrierError", "validate_active_work", "validate_policy"]
