"""CPU-only feature-package validation for production rows P0002--P0008.

This control-plane module reads only committed provenance and published
coordinate artifacts.  It has no WSI, OpenSlide, pixel, Torch/CUDA, model,
network, publication, deletion, Drive, or training surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import uuid

import h5py
import numpy as np
import yaml

from .brca_singleton_streaming_policy import PatientStage
from .brca_streaming_executor_v2 import start_event_from_plan
from .brca_streaming_production_adapter import (
    ALIGNMENT_SHA256,
    COMPACT_FILES,
    CompactArtifactValidationEvidence,
    SOURCE_POLICY_HASHES,
    ValidatedStageOutcome,
    load_frozen_cohort_order,
    plan_bound_stage,
    transaction_identity,
    validated_success_event,
)
from .brca_streaming_recovery_v2 import RecoveryEvent, ReplayAction, replay_events


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/teamspace/studios/this_studio/brca_pilot_data")
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
COORDINATE_RESULT = (
    ROOT / "multiscale_feature_pilot/provenance/"
    "brca_p0002_p0008_coordinate_execution_result.yaml"
)
COORDINATE_RESULT_SHA256 = "bdb8f9e03164006b8420230e392c18df073df37519d52aa89ae7d5a05baa665b"
COORDINATE_REQUEST = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_request.yaml"
COORDINATE_REQUEST_SHA256 = "47805f82a8a727f6dc646860a109b8cd5d8c19a3213a8c18fba5c31fcd96ca6e"
CHECKPOINT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
FEATURE_DIM = 2_048
PACKAGE_LABELS = tuple(f"P{index:04d}" for index in range(2, 9))


class FeaturePackageError(RuntimeError):
    """Raised when frozen feature inputs or the fail-closed contract drift."""


@dataclass(frozen=True)
class FeaturePackage:
    label: str
    cohort_index: int
    patient_id: str
    slide_id: str
    gdc_uuid: str
    omic_source_row_id: str
    coordinate_manifest_sha256: str
    coordinate_result_sha256: str
    scale_policy_sha256: str
    scale_2x_rows: int
    scale_4x_rows: int
    total_rows: int
    raw_feature_bytes: int
    compact_artifact_bytes_range: tuple[int, int]
    feature_directory: Path
    recovery_directory: Path

    @property
    def combined_shape(self) -> tuple[int, int]:
        return self.total_rows, FEATURE_DIM

    @property
    def natural_healnet_wsi_shape(self) -> tuple[int, int, int]:
        return 1, self.total_rows, FEATURE_DIM

    @property
    def row_ranges(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (0, self.scale_2x_rows), (self.scale_2x_rows, self.total_rows)


@dataclass(frozen=True)
class ConsolidatedGPUPlan:
    patient_labels: tuple[str, ...]
    total_patch_reads: int
    total_raw_feature_bytes: int
    total_compact_artifact_bytes_range: tuple[int, int]
    estimated_gpu_wall_seconds_range: tuple[int, int]
    execution_order: tuple[str, ...]
    maximum_concurrent_patients: int
    checkpoint_sha256: str


def _require(value: bool, message: str) -> None:
    if not value:
        raise FeaturePackageError(message)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path) -> None:
    _require(os.path.lexists(path), f"required path missing: {path}")
    details = path.lstat()
    _require(stat.S_ISREG(details.st_mode) and not stat.S_ISLNK(details.st_mode), f"regular non-symlink file required: {path}")


def _directory(path: Path) -> None:
    _require(os.path.lexists(path), f"required directory missing: {path}")
    details = path.lstat()
    _require(stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode), f"directory non-symlink required: {path}")


def _hash_label(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _coordinate_directory(label: str) -> Path:
    return DATA_ROOT / f"BRCA_PRODUCTION_{label}.coordinates"


def _feature_directory(label: str) -> Path:
    return DATA_ROOT / f"BRCA_PRODUCTION_{label}.features"


def _recovery_directory(label: str) -> Path:
    return DATA_ROOT / f"BRCA_PRODUCTION_{label}.recovery_v2"


def prepare_p0002_p0008_feature_packages() -> tuple[FeaturePackage, ...]:
    """Validate frozen coordinate evidence and return seven non-executable packages."""

    _regular(COORDINATE_RESULT)
    _require(_sha256_path(COORDINATE_RESULT) == COORDINATE_RESULT_SHA256, "coordinate result SHA256 drift")
    result = yaml.safe_load(COORDINATE_RESULT.read_text(encoding="utf-8"))
    _regular(COORDINATE_REQUEST)
    _require(_sha256_path(COORDINATE_REQUEST) == COORDINATE_REQUEST_SHA256, "coordinate request SHA256 drift")
    request = yaml.safe_load(COORDINATE_REQUEST.read_text(encoding="utf-8"))
    _require(result["status"] == "BRCA_PRODUCTION_P0002_P0008_COORDINATES_VERIFIED", "coordinate result not verified")
    _require(tuple(result["patients"]) == PACKAGE_LABELS, "coordinate patient order drift")
    cohort = load_frozen_cohort_order(ALIGNMENT)
    packages: list[FeaturePackage] = []
    for cohort_index, label in enumerate(PACKAGE_LABELS, start=2):
        binding = cohort[cohort_index - 1]
        record = result["patients"][label]
        directory = _coordinate_directory(label)
        _directory(directory)
        expected_files = {
            "coordinate_manifest.json",
            "coordinate_manifest.json.sha256",
            "scale_2x_coordinates.h5",
            "scale_4x_coordinates.h5",
        }
        _require({path.name for path in directory.iterdir()} == expected_files, f"coordinate file set drift: {label}")
        manifest_path = directory / "coordinate_manifest.json"
        sidecar_path = directory / "coordinate_manifest.json.sha256"
        _regular(manifest_path)
        _regular(sidecar_path)
        manifest_sha256 = _sha256_path(manifest_path)
        _require(manifest_sha256 == record["manifest_sha256"], f"manifest hash drift: {label}")
        _require(_sha256_path(sidecar_path) == record["manifest_sidecar_sha256"], f"manifest sidecar hash drift: {label}")
        _require(sidecar_path.read_text(encoding="ascii").strip() == f"{manifest_sha256}  coordinate_manifest.json", f"manifest sidecar content drift: {label}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _require(manifest["schema"] == "BRCA_COORDINATE_ARTIFACT_SET_V1", f"coordinate schema drift: {label}")
        branch_2x = manifest["branches"]["scale_2x"]
        branch_4x = manifest["branches"]["scale_4x"]
        policy_sha256 = branch_2x["attributes"]["policy_sha256"]
        _require(policy_sha256 == branch_4x["attributes"]["policy_sha256"], f"branch policy mismatch: {label}")
        _require(policy_sha256 == request["patients"][label]["policy_sha256"], f"policy not bound by frozen request: {label}")
        for branch_name, branch, result_key in (("scale_2x", branch_2x, "scale_2x"), ("scale_4x", branch_4x, "scale_4x")):
            attributes = branch["attributes"]
            _require(attributes["patient_id"] == binding.patient_id, f"patient binding drift: {label}/{branch_name}")
            _require(attributes["slide_id"] == binding.slide_id, f"slide binding drift: {label}/{branch_name}")
            _require(attributes["gdc_file_uuid"] == binding.gdc_uuid, f"UUID binding drift: {label}/{branch_name}")
            path = directory / branch["filename"]
            _regular(path)
            _require(_sha256_path(path) == branch["sha256"] == record[result_key]["file_sha256"], f"HDF5 file hash drift: {label}/{branch_name}")
            with h5py.File(path, "r") as handle:
                coordinates = handle["coords"][...]
            _require(coordinates.dtype == np.int64 and coordinates.ndim == 2 and coordinates.shape[1] == 2, f"coordinate tensor schema drift: {label}/{branch_name}")
            _require(coordinates.shape[0] == branch["coordinate_count"] == record[result_key]["count"], f"coordinate count drift: {label}/{branch_name}")
            content_hash = hashlib.sha256(np.ascontiguousarray(coordinates).tobytes()).hexdigest()
            _require(content_hash == branch["coordinates_sha256"] == record[result_key]["coordinates_sha256"], f"coordinate content hash drift: {label}/{branch_name}")
            _require(np.unique(coordinates, axis=0).shape[0] == coordinates.shape[0], f"coordinate duplicate drift: {label}/{branch_name}")
            order = np.lexsort((coordinates[:, 0], coordinates[:, 1]))
            _require(np.array_equal(coordinates, coordinates[order]), f"coordinate order drift: {label}/{branch_name}")
        rows_2x = int(branch_2x["coordinate_count"])
        rows_4x = int(branch_4x["coordinate_count"])
        total = rows_2x + rows_4x
        raw_bytes = total * FEATURE_DIM * 4
        packages.append(
            FeaturePackage(
                label=label,
                cohort_index=cohort_index,
                patient_id=binding.patient_id,
                slide_id=binding.slide_id,
                gdc_uuid=binding.gdc_uuid,
                omic_source_row_id=binding.omic_source_row_id,
                coordinate_manifest_sha256=manifest_sha256,
                coordinate_result_sha256=COORDINATE_RESULT_SHA256,
                scale_policy_sha256=policy_sha256,
                scale_2x_rows=rows_2x,
                scale_4x_rows=rows_4x,
                total_rows=total,
                raw_feature_bytes=raw_bytes,
                compact_artifact_bytes_range=(raw_bytes + 700_000, raw_bytes + 1_300_000),
                feature_directory=_feature_directory(label),
                recovery_directory=_recovery_directory(label),
            )
        )
    _require(sum(item.total_rows for item in packages) == result["totals"]["all_coordinates"], "combined coordinate total drift")
    return tuple(packages)


def consolidated_first_eight_gpu_plan(packages: tuple[FeaturePackage, ...]) -> ConsolidatedGPUPlan:
    """Return a bounded serial GPU plan; it does not authorize execution."""

    _require(tuple(item.label for item in packages) == PACKAGE_LABELS, "package order drift")
    p0001_rows = 16_816
    p0001_raw_bytes = p0001_rows * FEATURE_DIM * 4
    rows = p0001_rows + sum(item.total_rows for item in packages)
    raw_bytes = p0001_raw_bytes + sum(item.raw_feature_bytes for item in packages)
    compact_low = 138_500_000 + sum(item.compact_artifact_bytes_range[0] for item in packages)
    compact_high = 139_000_000 + sum(item.compact_artifact_bytes_range[1] for item in packages)
    _require(rows == 112_754 and raw_bytes == 923_680_768, "first-eight resource total drift")
    return ConsolidatedGPUPlan(
        patient_labels=("P0001",) + PACKAGE_LABELS,
        total_patch_reads=rows,
        total_raw_feature_bytes=raw_bytes,
        total_compact_artifact_bytes_range=(compact_low, compact_high),
        estimated_gpu_wall_seconds_range=(1_210, 1_800),
        execution_order=("P0001",) + PACKAGE_LABELS,
        maximum_concurrent_patients=1,
        checkpoint_sha256=CHECKPOINT_SHA256,
    )


def rehearse_feature_recovery(package: FeaturePackage) -> tuple[RecoveryEvent, ...]:
    """Replay synthetic P0002--P0008 events through FEATURES_VERIFIED only."""

    cohort = load_frozen_cohort_order(ALIGNMENT)
    binding = cohort[package.cohort_index - 1]
    identity = transaction_identity(
        binding,
        str(uuid.uuid5(uuid.NAMESPACE_URL, f"brca-first-eight-feature/{package.label}")),
    )
    events: list[RecoveryEvent] = []
    for index, stage in enumerate(PatientStage):
        if stage is PatientStage.TERMINAL_RECORDED:
            break
        authorization = _hash_label(f"synthetic {package.label} {stage.value} authorization")
        inputs = _stage_inputs(stage, binding, package, authorization)
        plan = plan_bound_stage(
            events,
            binding=binding,
            identity=identity,
            run_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{package.label}/{stage.value}/run")),
            authorization_sha256=authorization,
            retry_authorization_sha256=None,
            stage_input_hashes=inputs,
            source_policy_hashes=SOURCE_POLICY_HASHES,
        )
        started = start_event_from_plan(events, plan, recorded_at_utc=f"2000-01-01T00:{index * 2:02d}:00Z")
        events.append(started)
        output_label, output_hash = _stage_output(stage, package, authorization)
        evidence = ValidatedStageOutcome(
            stage=stage,
            authorization_sha256=authorization,
            output_hashes=((output_label, output_hash),),
            validation_record_sha256=output_hash,
            compact_artifact=(
                _synthetic_compact_evidence(package, plan, output_hash)
                if stage is PatientStage.FEATURES_VERIFIED
                else None
            ),
        )
        events.append(validated_success_event(events, plan, evidence, existing_output_hashes=None, recorded_at_utc=f"2000-01-01T00:{index * 2 + 1:02d}:00Z"))
    replay = replay_events(events)
    _require(replay.action is ReplayAction.ADVANCE_STAGE and replay.target_stage is PatientStage.TERMINAL_RECORDED, f"synthetic recovery replay drift: {package.label}")
    return tuple(events)


def _stage_inputs(stage: PatientStage, binding: object, package: FeaturePackage, authorization: str) -> dict[str, str]:
    if stage is PatientStage.PLANNED:
        return {"cohort_alignment": ALIGNMENT_SHA256, "singleton_identity": binding.identity_sha256}
    if stage is PatientStage.ACQUISITION_AUTHORIZED:
        return {"one_row_manifest": binding.manifest_sha256}
    if stage is PatientStage.RAW_VERIFIED:
        return {"raw_identity_declaration": binding.raw_identity_declaration_sha256}
    if stage is PatientStage.HEADER_POLICY_VERIFIED:
        return {"raw_verification_report": _hash_label(f"{package.label} frozen header result")}
    if stage is PatientStage.COORDINATES_VERIFIED:
        return {"header_policy_report": package.scale_policy_sha256}
    if stage is PatientStage.GPU_AUTHORIZED:
        return {"checkpoint_identity": CHECKPOINT_SHA256, "coordinate_manifest": package.coordinate_manifest_sha256}
    if stage is PatientStage.FEATURES_VERIFIED:
        return {
            "checkpoint_identity": CHECKPOINT_SHA256,
            "coordinate_manifest": package.coordinate_manifest_sha256,
            "gpu_authorization_record": authorization,
        }
    raise AssertionError(f"unhandled rehearsal stage: {stage}")


def _stage_output(stage: PatientStage, package: FeaturePackage, authorization: str) -> tuple[str, str]:
    if stage is PatientStage.PLANNED:
        return "patient_plan", _hash_label(f"{package.label} synthetic patient plan")
    if stage is PatientStage.ACQUISITION_AUTHORIZED:
        return "acquisition_authorization_record", _hash_label(f"{package.label} synthetic acquisition authorization")
    if stage is PatientStage.RAW_VERIFIED:
        return "raw_verification_report", _hash_label(f"{package.label} frozen header result")
    if stage is PatientStage.HEADER_POLICY_VERIFIED:
        return "header_policy_report", package.scale_policy_sha256
    if stage is PatientStage.COORDINATES_VERIFIED:
        return "coordinate_manifest", package.coordinate_manifest_sha256
    if stage is PatientStage.GPU_AUTHORIZED:
        return "gpu_authorization_record", authorization
    if stage is PatientStage.FEATURES_VERIFIED:
        return "compact_feature_manifest", _hash_label(f"{package.label} synthetic compact manifest")
    raise AssertionError(f"unhandled rehearsal stage: {stage}")


def _synthetic_compact_evidence(
    package: FeaturePackage, plan: object, compact_manifest_sha256: str
) -> CompactArtifactValidationEvidence:
    return CompactArtifactValidationEvidence(
        patient_id=package.patient_id,
        slide_id=package.slide_id,
        gdc_uuid=package.gdc_uuid,
        omic_source_row_id=package.omic_source_row_id,
        bound_source_policy_hashes=plan.source_policy_hashes,
        bound_input_hashes=plan.input_hashes,
        exact_files=COMPACT_FILES,
        manifest_sha256=compact_manifest_sha256,
        sidecar_manifest_sha256=compact_manifest_sha256,
        file_hashes=tuple(
            (name, compact_manifest_sha256 if name == "compact_manifest.json" else _hash_label(f"{package.label} synthetic {name}"))
            for name in COMPACT_FILES
        ),
        tensor_shape=package.combined_shape,
        tensor_dtype="float32",
        tensor_device="cpu",
        tensor_contiguous=True,
        tensor_finite=True,
        tensor_requires_grad=False,
        scale_2x_row_range=package.row_ranges[0],
        scale_4x_row_range=package.row_ranges[1],
        row_provenance_count=package.total_rows,
    )


__all__ = [
    "CHECKPOINT_SHA256",
    "ConsolidatedGPUPlan",
    "FEATURE_DIM",
    "FeaturePackage",
    "FeaturePackageError",
    "PACKAGE_LABELS",
    "consolidated_first_eight_gpu_plan",
    "prepare_p0002_p0008_feature_packages",
    "rehearse_feature_recovery",
]
