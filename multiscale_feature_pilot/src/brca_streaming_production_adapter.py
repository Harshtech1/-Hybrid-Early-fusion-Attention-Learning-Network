"""CPU-only control-plane adapter for the frozen BRCA streaming cohort.

The adapter securely loads and binds the exact 894-row singleton inventory,
constructs exact one-row GDC manifest bytes, validates Omic identity evidence,
checks compact-artifact validation evidence, and submits metadata-only stage
plans and outcomes to recovery-v2.  It deliberately contains no downloader,
OpenSlide, pixel, coordinate, Torch/CUDA, feature, model, Drive, raw-deletion,
or training operation surface.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
import stat
import uuid
from typing import Mapping, Sequence

from .brca_singleton_streaming_policy import PatientStage
from .brca_streaming_executor_v2 import (
    STAGE_CONTRACTS,
    StagePlan,
    outcome_event_from_plan,
    plan_next_stage,
)
from .brca_streaming_recovery_v2 import (
    FailureClassification,
    PatientTransactionIdentity,
    RecoveryEvent,
    append_event,
)


ALIGNMENT_SHA256 = "13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe"
COHORT_ORDER_SHA256 = "1c97fa4f8305185f2da191f5ebaed603db7d2bdd11c89a580e784ef46655af5a"
COHORT_PATIENTS = 894
COHORT_RAW_BYTES = 918_532_189_383
MAX_ALIGNMENT_BYTES = 16_000_000
COMPACT_FILES = (
    "combined_features.pt",
    "row_provenance.csv",
    "compact_manifest.json",
    "compact_manifest.json.sha256",
)
SOURCE_POLICY_HASHES = {
    "cohort_order_manifest": COHORT_ORDER_SHA256,
    "compact_artifact_policy": "59d4b1168108052ba5b18d8c4b5f5aa6de83c36ab234907892bf374ee9f712b4",
    "executor_v2_policy": "7558dcd296b1c5822ec698ac2458e86f38595748648aba54ca1db567b5c2a7cc",
    "singleton_streaming_policy": "9172449122f11cb17ef196d79911d1e5914125544531dd7b1eeeca6a11ae7b5f",
}
ALIGNMENT_COLUMNS = (
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
_PATIENT = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
_MD5 = re.compile(r"^[0-9a-f]{32}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ProductionAdapterError(RuntimeError):
    """Raised when live evidence exceeds the frozen control-plane contract."""


@dataclass(frozen=True)
class CohortPatientBinding:
    cohort_index: int
    patient_id: str
    slide_id: str
    gdc_uuid: str
    omic_source_row_id: str
    filename: str
    md5: str
    size_bytes: int
    state: str

    @property
    def identity_sha256(self) -> str:
        return _sha256_json(_canonical_row(self))

    @property
    def manifest_sha256(self) -> str:
        return hashlib.sha256(exact_manifest_bytes(self)).hexdigest()

    @property
    def omic_identity_sha256(self) -> str:
        return _sha256_json(
            {
                "patient_id": self.patient_id,
                "slide_id": self.slide_id,
                "omic_source_row_id": self.omic_source_row_id,
            }
        )

    @property
    def raw_identity_declaration_sha256(self) -> str:
        return _sha256_json(
            {
                "gdc_uuid": self.gdc_uuid,
                "filename": self.filename,
                "md5": self.md5,
                "size_bytes": self.size_bytes,
                "state": self.state,
                "omic_identity_sha256": self.omic_identity_sha256,
            }
        )


@dataclass(frozen=True)
class CompactArtifactValidationEvidence:
    patient_id: str
    slide_id: str
    gdc_uuid: str
    omic_source_row_id: str
    bound_source_policy_hashes: tuple[tuple[str, str], ...]
    bound_input_hashes: tuple[tuple[str, str], ...]
    exact_files: tuple[str, ...]
    manifest_sha256: str
    sidecar_manifest_sha256: str
    file_hashes: tuple[tuple[str, str], ...]
    tensor_shape: tuple[int, int]
    tensor_dtype: str
    tensor_device: str
    tensor_contiguous: bool
    tensor_finite: bool
    tensor_requires_grad: bool
    scale_2x_row_range: tuple[int, int]
    scale_4x_row_range: tuple[int, int]
    row_provenance_count: int


@dataclass(frozen=True)
class ValidatedStageOutcome:
    stage: PatientStage
    authorization_sha256: str
    output_hashes: tuple[tuple[str, str], ...]
    validation_record_sha256: str
    compact_artifact: CompactArtifactValidationEvidence | None = None


FORBIDDEN_OPERATION_SURFACES = {
    "download": False,
    "network": False,
    "openslide": False,
    "pixel_read": False,
    "coordinate_generation": False,
    "torch_or_cuda": False,
    "feature_extraction": False,
    "healnet": False,
    "google_drive": False,
    "raw_wsi_deletion": False,
    "training": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ProductionAdapterError(message)


def _validate_sha256(value: str, label: str) -> str:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{label} must be a lowercase SHA256")
    return value


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _sha256_json(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _canonical_row(binding: CohortPatientBinding) -> dict[str, object]:
    return {
        "cohort_index": binding.cohort_index,
        "patient_id": binding.patient_id,
        "slide_id": binding.slide_id,
        "gdc_uuid": binding.gdc_uuid,
        "omic_source_row_id": binding.omic_source_row_id,
        "md5": binding.md5,
        "size_bytes": binding.size_bytes,
    }


def _validate_no_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for parent in reversed(absolute.parents):
        try:
            details = parent.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(details.st_mode), f"symlink ancestor forbidden: {parent}")


def _read_bounded_nofollow(path: Path, *, maximum_bytes: int) -> bytes:
    _validate_no_symlink_ancestors(path)
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
            _require(bool(chunk), "unexpected EOF while reading source")
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


def _binding_from_row(row: Mapping[str, str], cohort_index: int) -> CohortPatientBinding:
    _require(row["alignment_status"] == "KEEP", "retained row status drift")
    _require(row["alignment_reason"] == "EXACT_SINGLETON_CASE_AND_SLIDE_MATCH", "retained row reason drift")
    _require(row["patient_wsi_count"] == row["patient_omic_count"] == "1", "singleton count drift")
    patient = row["case_id"]
    slide = row["slide_id"]
    _require(_PATIENT.fullmatch(patient) is not None, "invalid TCGA patient ID")
    _require(slide == row["filename"] and slide.startswith(f"{patient}-") and slide.endswith(".svs"), "patient/slide/filename mismatch")
    _require("/" not in slide and "\\" not in slide, "slide filename contains a path separator")
    try:
        parsed_uuid = uuid.UUID(row["id"])
    except ValueError as error:
        raise ProductionAdapterError("invalid GDC UUID") from error
    _require(str(parsed_uuid) == row["id"], "GDC UUID must be canonical lowercase")
    _require(row["omic_source_row_id"].isdigit(), "Omic row ID must remain a decimal string")
    _require(_MD5.fullmatch(row["md5"]) is not None, "invalid manifest MD5")
    _require(row["state"] == "released", "GDC file is not released")
    try:
        size = int(row["size"])
    except ValueError as error:
        raise ProductionAdapterError("invalid WSI size") from error
    _require(size > 0 and str(size) == row["size"], "WSI size must be a canonical positive integer")
    return CohortPatientBinding(
        cohort_index=cohort_index,
        patient_id=patient,
        slide_id=slide,
        gdc_uuid=row["id"],
        omic_source_row_id=row["omic_source_row_id"],
        filename=row["filename"],
        md5=row["md5"],
        size_bytes=size,
        state=row["state"],
    )


def load_frozen_cohort_order(
    alignment_path: str | Path,
    *,
    expected_alignment_sha256: str = ALIGNMENT_SHA256,
    expected_order_sha256: str = COHORT_ORDER_SHA256,
) -> tuple[CohortPatientBinding, ...]:
    """Securely load, validate, sort, and bind the exact singleton cohort."""

    payload = _read_bounded_nofollow(Path(alignment_path), maximum_bytes=MAX_ALIGNMENT_BYTES)
    _require(hashlib.sha256(payload).hexdigest() == expected_alignment_sha256, "alignment SHA256 mismatch")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ProductionAdapterError("alignment is not strict UTF-8") from error
    reader = csv.DictReader(io.StringIO(text, newline=""))
    _require(tuple(reader.fieldnames or ()) == ALIGNMENT_COLUMNS, "alignment column contract drift")
    retained = [
        row
        for row in reader
        if row["alignment_status"] == "KEEP"
        and row["alignment_reason"] == "EXACT_SINGLETON_CASE_AND_SLIDE_MATCH"
    ]
    retained.sort(key=lambda row: (row["case_id"], row["slide_id"], row["id"]))
    bindings = tuple(_binding_from_row(row, index) for index, row in enumerate(retained, start=1))
    _require(len(bindings) == COHORT_PATIENTS, "cohort must contain exactly 894 singleton patients")
    _require(len({item.patient_id for item in bindings}) == COHORT_PATIENTS, "duplicate patient in cohort")
    _require(len({item.slide_id for item in bindings}) == COHORT_PATIENTS, "duplicate slide in cohort")
    _require(len({item.gdc_uuid for item in bindings}) == COHORT_PATIENTS, "duplicate GDC UUID in cohort")
    _require(sum(item.size_bytes for item in bindings) == COHORT_RAW_BYTES, "raw inventory byte total drift")
    digest = hashlib.sha256(b"".join(_canonical_json(_canonical_row(item)) for item in bindings)).hexdigest()
    _require(digest == expected_order_sha256, "canonical 894-patient order digest mismatch")
    return bindings


def exact_manifest_bytes(binding: CohortPatientBinding) -> bytes:
    header = "id\tfilename\tmd5\tsize\tstate\n"
    row = f"{binding.gdc_uuid}\t{binding.filename}\t{binding.md5}\t{binding.size_bytes}\t{binding.state}\n"
    return (header + row).encode("utf-8")


def validate_exact_manifest_bytes(binding: CohortPatientBinding, payload: bytes) -> str:
    _require(payload == exact_manifest_bytes(binding), "one-row GDC manifest bytes do not match frozen patient")
    return hashlib.sha256(payload).hexdigest()


def validate_exact_omic_rematch(
    binding: CohortPatientBinding,
    *,
    patient_id: str,
    slide_id: str,
    source_row_id: str,
) -> str:
    _require(patient_id == binding.patient_id, "Omic patient mismatch")
    _require(slide_id == binding.slide_id, "Omic full-slide mismatch")
    _require(source_row_id == binding.omic_source_row_id, "Omic source-row mismatch")
    return binding.omic_identity_sha256


def transaction_identity(binding: CohortPatientBinding, transaction_id: str) -> PatientTransactionIdentity:
    return PatientTransactionIdentity(
        cohort_index=binding.cohort_index,
        patient_id=binding.patient_id,
        slide_id=binding.slide_id,
        gdc_uuid=binding.gdc_uuid,
        transaction_id=transaction_id,
    )


def validate_compact_artifact_evidence(evidence: CompactArtifactValidationEvidence) -> str:
    _require(_PATIENT.fullmatch(evidence.patient_id) is not None, "compact evidence patient ID is invalid")
    _require(
        evidence.slide_id.startswith(f"{evidence.patient_id}-")
        and evidence.slide_id.endswith(".svs"),
        "compact evidence slide identity is invalid",
    )
    try:
        parsed_uuid = uuid.UUID(evidence.gdc_uuid)
    except ValueError as error:
        raise ProductionAdapterError("compact evidence GDC UUID is invalid") from error
    _require(str(parsed_uuid) == evidence.gdc_uuid, "compact evidence GDC UUID must be canonical lowercase")
    _require(evidence.omic_source_row_id.isdigit(), "compact evidence Omic row must be decimal")
    _require(
        dict(evidence.bound_source_policy_hashes) == SOURCE_POLICY_HASHES,
        "compact evidence source-policy bindings drift",
    )
    _require(
        len(dict(evidence.bound_source_policy_hashes))
        == len(evidence.bound_source_policy_hashes),
        "duplicate compact evidence source-policy label",
    )
    bound_inputs = dict(evidence.bound_input_hashes)
    _require(len(bound_inputs) == len(evidence.bound_input_hashes), "duplicate compact evidence input label")
    for label, digest in evidence.bound_input_hashes:
        _validate_sha256(digest, f"compact evidence input {label}")
    _validate_sha256(evidence.manifest_sha256, "compact manifest")
    _require(evidence.sidecar_manifest_sha256 == evidence.manifest_sha256, "compact sidecar does not reference manifest")
    _require(evidence.exact_files == COMPACT_FILES, "compact artifact exact file set drift")
    hashes = dict(evidence.file_hashes)
    _require(len(hashes) == len(evidence.file_hashes), "duplicate compact file hash label")
    _require(tuple(hashes) == COMPACT_FILES, "compact file hash order or labels drift")
    for label, digest in evidence.file_hashes:
        _validate_sha256(digest, f"compact file {label}")
    _require(hashes["compact_manifest.json"] == evidence.manifest_sha256, "compact manifest file hash mismatch")
    rows, width = evidence.tensor_shape
    _require(rows > 1 and width == 2048, "compact tensor shape must be [P_i,2048]")
    _require(evidence.tensor_dtype == "float32" and evidence.tensor_device == "cpu", "compact tensor dtype/device drift")
    _require(evidence.tensor_contiguous and evidence.tensor_finite and not evidence.tensor_requires_grad, "compact tensor validity flags failed")
    split = evidence.scale_2x_row_range[1]
    _require(evidence.scale_2x_row_range == (0, split) and 0 < split < rows, "invalid 2x row range")
    _require(evidence.scale_4x_row_range == (split, rows), "2x/4x row ranges are not contiguous")
    _require(evidence.row_provenance_count == rows, "row provenance count does not match tensor rows")
    return evidence.manifest_sha256


def _validate_binding_identity(binding: CohortPatientBinding, identity: PatientTransactionIdentity) -> None:
    _require(identity.cohort_index == binding.cohort_index, "cohort index mismatch")
    _require(identity.patient_id == binding.patient_id, "transaction patient mismatch")
    _require(identity.slide_id == binding.slide_id, "transaction slide mismatch")
    _require(identity.gdc_uuid == binding.gdc_uuid, "transaction UUID mismatch")


def plan_bound_stage(
    existing: Sequence[RecoveryEvent],
    *,
    binding: CohortPatientBinding,
    identity: PatientTransactionIdentity,
    run_id: str,
    authorization_sha256: str,
    retry_authorization_sha256: str | None,
    stage_input_hashes: Mapping[str, str],
    source_policy_hashes: Mapping[str, str] = SOURCE_POLICY_HASHES,
) -> StagePlan:
    """Validate exact patient-specific bindings and create a metadata-only plan."""

    _validate_binding_identity(binding, identity)
    _require(dict(source_policy_hashes) == SOURCE_POLICY_HASHES, "source policy hash set drift")
    # The v2 planner authoritatively derives the next stage; special patient
    # bindings below are checked whenever their label is present.
    inputs = dict(stage_input_hashes)
    if "cohort_alignment" in inputs:
        _require(inputs["cohort_alignment"] == ALIGNMENT_SHA256, "planned stage alignment binding drift")
    if "singleton_identity" in inputs:
        _require(inputs["singleton_identity"] == binding.identity_sha256, "singleton identity binding drift")
    if "one_row_manifest" in inputs:
        _require(inputs["one_row_manifest"] == binding.manifest_sha256, "one-row manifest binding drift")
    if "raw_identity_declaration" in inputs:
        _require(inputs["raw_identity_declaration"] == binding.raw_identity_declaration_sha256, "raw identity declaration binding drift")
    return plan_next_stage(
        existing,
        identity=identity,
        run_id=run_id,
        authorization_sha256=authorization_sha256,
        retry_authorization_sha256=retry_authorization_sha256,
        source_policy_hashes=source_policy_hashes,
        stage_input_hashes=inputs,
    )


def validated_success_event(
    existing: Sequence[RecoveryEvent],
    plan: StagePlan,
    evidence: ValidatedStageOutcome,
    *,
    existing_output_hashes: Mapping[str, str] | None,
    recorded_at_utc: str,
) -> RecoveryEvent:
    _require(evidence.stage is plan.stage, "validated outcome stage differs from plan")
    _require(evidence.authorization_sha256 == plan.authorization_sha256, "validated outcome authorization differs from plan")
    _validate_sha256(evidence.validation_record_sha256, "validation record")
    output_hashes = dict(evidence.output_hashes)
    _require(len(output_hashes) == len(evidence.output_hashes), "duplicate validated output label")
    _require(
        evidence.validation_record_sha256 in output_hashes.values(),
        "validation record must be the ledger-recorded stage output",
    )
    if plan.stage is PatientStage.FEATURES_VERIFIED:
        _require(evidence.compact_artifact is not None, "feature success requires compact validation evidence")
        compact = evidence.compact_artifact
        _require(compact.patient_id == plan.identity.patient_id, "compact evidence patient differs from stage plan")
        _require(compact.slide_id == plan.identity.slide_id, "compact evidence slide differs from stage plan")
        _require(compact.gdc_uuid == plan.identity.gdc_uuid, "compact evidence UUID differs from stage plan")
        _require(
            compact.bound_source_policy_hashes == plan.source_policy_hashes,
            "compact evidence source policies differ from stage plan",
        )
        _require(compact.bound_input_hashes == plan.input_hashes, "compact evidence inputs differ from stage plan")
        manifest = validate_compact_artifact_evidence(evidence.compact_artifact)
        _require(output_hashes.get("compact_feature_manifest") == manifest, "feature output does not bind compact manifest")
    else:
        _require(evidence.compact_artifact is None, "compact evidence is only valid for FEATURES_VERIFIED")
    return outcome_event_from_plan(
        existing,
        plan,
        succeeded=True,
        output_hashes=output_hashes,
        existing_output_hashes=existing_output_hashes,
        outputs_validated=True,
        failure_classification=None,
        error_code=None,
        error_class=None,
        error_message=None,
        recorded_at_utc=recorded_at_utc,
    )


def validated_failure_event(
    existing: Sequence[RecoveryEvent],
    plan: StagePlan,
    *,
    classification: FailureClassification,
    error_code: str,
    error_class: str,
    error_message: str,
    recorded_at_utc: str,
) -> RecoveryEvent:
    return outcome_event_from_plan(
        existing,
        plan,
        succeeded=False,
        output_hashes={},
        existing_output_hashes=None,
        outputs_validated=False,
        failure_classification=classification,
        error_code=error_code,
        error_class=error_class,
        error_message=error_message,
        recorded_at_utc=recorded_at_utc,
    )


def append_control_event(directory: str | Path, event: RecoveryEvent) -> Path:
    """Append only a pre-built metadata event to recovery-v2."""

    return append_event(directory, event)


__all__ = [
    "ALIGNMENT_SHA256",
    "COHORT_ORDER_SHA256",
    "COHORT_PATIENTS",
    "COHORT_RAW_BYTES",
    "COMPACT_FILES",
    "SOURCE_POLICY_HASHES",
    "CompactArtifactValidationEvidence",
    "CohortPatientBinding",
    "FORBIDDEN_OPERATION_SURFACES",
    "ProductionAdapterError",
    "ValidatedStageOutcome",
    "append_control_event",
    "exact_manifest_bytes",
    "load_frozen_cohort_order",
    "plan_bound_stage",
    "transaction_identity",
    "validate_compact_artifact_evidence",
    "validate_exact_manifest_bytes",
    "validate_exact_omic_rematch",
    "validated_failure_event",
    "validated_success_event",
]
