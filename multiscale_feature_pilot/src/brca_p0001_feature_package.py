"""CPU-only P0001 production feature-package and recovery-v2 rehearsal.

This module binds existing, verified P0001 evidence to the frozen 894-patient
production adapter and constructs an in-memory recovery-v2 transaction.  It
has no WSI, OpenSlide, pixel, Torch/CUDA, model, publication, deletion,
network, Drive, or training surface.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

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


PATIENT_ID = "TCGA-3C-AALK"
SLIDE_ID = "TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53.svs"
GDC_UUID = "93b26333-5723-4fa4-a4de-6124c04ab243"
OMIC_SOURCE_ROW_ID = "4"
COHORT_INDEX = 1
SCALE_2X_ROWS = 13_372
SCALE_4X_ROWS = 3_444
TOTAL_ROWS = 16_816
FEATURE_DIM = 2_048
CHECKPOINT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
COORDINATE_MANIFEST_SHA256 = "f1825acf7d0b96c92bfb66038d60af6e1572ea945490e2245d5e3ec222677d6e"
FUTURE_GPU_AUTHORIZATION_STATEMENT_SHA256 = (
    "86db1c4ddb2deead9ce8ca0d6282a890136c3e63451685aa862909a184867f48"
)
TRANSACTION_ID = "386116c1-6696-5020-a495-446d6e37b829"
RUN_IDS = (
    "040da6b0-6832-5015-acc6-f560ceea24d3",
    "f3e03424-7cd7-52dc-870b-da1b070596a6",
    "b0c11772-4f47-517c-aa8c-1d116bda7b02",
    "97a7ba7f-6a87-5e25-8f10-66876ad92dfb",
    "81627690-8c0f-5123-8a6d-504c5fc4c360",
    "4d4a6ed7-d1b5-502e-9db3-7e6e3cadfa3a",
    "02fb9531-b15c-5df5-bf23-12323a529731",
    "7cba2785-a7c4-554d-a89d-27cbaaaebd0c",
)

PRODUCTION_REQUEST_SHA256 = "04da2f211bfae390ec94ec2dd082ed032204d9c3b81424c6273e33679b14134b"
HEADER_AUTHORIZATION_SHA256 = "691a8d536c9a614a326d26f971f586b7e5ab441a5efea189e347f2c4d664d61a"
HEADER_RESULT_SHA256 = "6c7faa0c4e80f4649d784140b907280c1f5b889f5153b4550da9e3e2f198efb3"
POLICY_REVIEW_SHA256 = "aefd0e5de9b41ce726e880b14d5002d4012acbffaed7196ba301b28c323da77c"
COORDINATE_AUTHORIZATION_SHA256 = (
    "dcf916daf81a25c4a412f4a6aa43fe22bd616ec5755c68f94a7d2f2f2f6f5baa"
)
RAW_LIFECYCLE_DECISION_SHA256 = (
    "04da2f211bfae390ec94ec2dd082ed032204d9c3b81424c6273e33679b14134b"
)


@dataclass(frozen=True)
class P0001RecoveryRehearsal:
    events: tuple[RecoveryEvent, ...]
    final_action: ReplayAction
    compact_manifest_sha256: str
    transaction_id: str


def _sha256_label(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _stage_inputs(stage: PatientStage, binding: object, gpu_authorization: str, compact: str) -> dict[str, str]:
    if stage is PatientStage.PLANNED:
        return {
            "cohort_alignment": ALIGNMENT_SHA256,
            "singleton_identity": binding.identity_sha256,
        }
    if stage is PatientStage.ACQUISITION_AUTHORIZED:
        return {"one_row_manifest": binding.manifest_sha256}
    if stage is PatientStage.RAW_VERIFIED:
        return {"raw_identity_declaration": binding.raw_identity_declaration_sha256}
    if stage is PatientStage.HEADER_POLICY_VERIFIED:
        return {"raw_verification_report": HEADER_RESULT_SHA256}
    if stage is PatientStage.COORDINATES_VERIFIED:
        return {"header_policy_report": POLICY_REVIEW_SHA256}
    if stage is PatientStage.GPU_AUTHORIZED:
        return {
            "checkpoint_identity": CHECKPOINT_SHA256,
            "coordinate_manifest": COORDINATE_MANIFEST_SHA256,
        }
    if stage is PatientStage.FEATURES_VERIFIED:
        return {
            "checkpoint_identity": CHECKPOINT_SHA256,
            "coordinate_manifest": COORDINATE_MANIFEST_SHA256,
            "gpu_authorization_record": gpu_authorization,
        }
    if stage is PatientStage.TERMINAL_RECORDED:
        return {
            "compact_feature_manifest": compact,
            "raw_lifecycle_decision": RAW_LIFECYCLE_DECISION_SHA256,
        }
    raise AssertionError(f"unhandled stage {stage}")


def _stage_authorization(stage: PatientStage, gpu_authorization: str) -> str:
    return {
        PatientStage.PLANNED: PRODUCTION_REQUEST_SHA256,
        PatientStage.ACQUISITION_AUTHORIZED: HEADER_AUTHORIZATION_SHA256,
        PatientStage.RAW_VERIFIED: HEADER_AUTHORIZATION_SHA256,
        PatientStage.HEADER_POLICY_VERIFIED: POLICY_REVIEW_SHA256,
        PatientStage.COORDINATES_VERIFIED: COORDINATE_AUTHORIZATION_SHA256,
        PatientStage.GPU_AUTHORIZED: gpu_authorization,
        PatientStage.FEATURES_VERIFIED: gpu_authorization,
        PatientStage.TERMINAL_RECORDED: RAW_LIFECYCLE_DECISION_SHA256,
    }[stage]


def _stage_output(stage: PatientStage, gpu_authorization: str, compact: str) -> tuple[str, str]:
    return {
        PatientStage.PLANNED: ("patient_plan", PRODUCTION_REQUEST_SHA256),
        PatientStage.ACQUISITION_AUTHORIZED: (
            "acquisition_authorization_record",
            HEADER_AUTHORIZATION_SHA256,
        ),
        PatientStage.RAW_VERIFIED: ("raw_verification_report", HEADER_RESULT_SHA256),
        PatientStage.HEADER_POLICY_VERIFIED: (
            "header_policy_report",
            POLICY_REVIEW_SHA256,
        ),
        PatientStage.COORDINATES_VERIFIED: (
            "coordinate_manifest",
            COORDINATE_MANIFEST_SHA256,
        ),
        PatientStage.GPU_AUTHORIZED: ("gpu_authorization_record", gpu_authorization),
        PatientStage.FEATURES_VERIFIED: ("compact_feature_manifest", compact),
        PatientStage.TERMINAL_RECORDED: (
            "terminal_record",
            _sha256_label("P0001 synthetic terminal retention record"),
        ),
    }[stage]


def _synthetic_compact_evidence(
    plan: object,
    compact_manifest_sha256: str,
    compact_file_hashes: tuple[tuple[str, str], ...] | None,
) -> CompactArtifactValidationEvidence:
    file_hashes = compact_file_hashes or tuple(
        (
            name,
            compact_manifest_sha256
            if name == "compact_manifest.json"
            else _sha256_label(f"P0001 synthetic {name}"),
        )
        for name in COMPACT_FILES
    )
    return CompactArtifactValidationEvidence(
        patient_id=PATIENT_ID,
        slide_id=SLIDE_ID,
        gdc_uuid=GDC_UUID,
        omic_source_row_id=OMIC_SOURCE_ROW_ID,
        bound_source_policy_hashes=plan.source_policy_hashes,
        bound_input_hashes=plan.input_hashes,
        exact_files=COMPACT_FILES,
        manifest_sha256=compact_manifest_sha256,
        sidecar_manifest_sha256=compact_manifest_sha256,
        file_hashes=file_hashes,
        tensor_shape=(TOTAL_ROWS, FEATURE_DIM),
        tensor_dtype="float32",
        tensor_device="cpu",
        tensor_contiguous=True,
        tensor_finite=True,
        tensor_requires_grad=False,
        scale_2x_row_range=(0, SCALE_2X_ROWS),
        scale_4x_row_range=(SCALE_2X_ROWS, TOTAL_ROWS),
        row_provenance_count=TOTAL_ROWS,
    )


def rehearse_p0001_feature_transaction(
    alignment_path: str | Path,
    *,
    gpu_authorization_sha256: str = FUTURE_GPU_AUTHORIZATION_STATEMENT_SHA256,
    compact_manifest_sha256: str | None = None,
    compact_file_hashes: tuple[tuple[str, str], ...] | None = None,
    through_stage: PatientStage = PatientStage.TERMINAL_RECORDED,
    event_timestamps: tuple[str, ...] | None = None,
) -> P0001RecoveryRehearsal:
    """Build and replay all eight P0001 stages in memory using synthetic outputs."""

    compact_manifest_sha256 = compact_manifest_sha256 or _sha256_label(
        "P0001 synthetic compact manifest"
    )
    timestamps = event_timestamps or tuple(
        f"2000-01-01T00:{index:02d}:00Z" for index in range(16)
    )
    if len(timestamps) != 16:
        raise ValueError("P0001 recovery rehearsal requires 16 event timestamps")
    cohort = load_frozen_cohort_order(alignment_path)
    binding = cohort[0]
    if (
        binding.cohort_index,
        binding.patient_id,
        binding.slide_id,
        binding.gdc_uuid,
        binding.omic_source_row_id,
    ) != (COHORT_INDEX, PATIENT_ID, SLIDE_ID, GDC_UUID, OMIC_SOURCE_ROW_ID):
        raise ValueError("P0001 frozen cohort binding drift")
    identity = transaction_identity(binding, TRANSACTION_ID)
    events: list[RecoveryEvent] = []
    for index, stage in enumerate(PatientStage):
        authorization = _stage_authorization(stage, gpu_authorization_sha256)
        plan = plan_bound_stage(
            events,
            binding=binding,
            identity=identity,
            run_id=RUN_IDS[index],
            authorization_sha256=authorization,
            retry_authorization_sha256=None,
            stage_input_hashes=_stage_inputs(
                stage, binding, gpu_authorization_sha256, compact_manifest_sha256
            ),
            source_policy_hashes=SOURCE_POLICY_HASHES,
        )
        if plan.stage is not stage:
            raise ValueError("P0001 stage-order drift")
        started = start_event_from_plan(
            events, plan, recorded_at_utc=timestamps[index * 2]
        )
        events.append(started)
        output = _stage_output(stage, gpu_authorization_sha256, compact_manifest_sha256)
        evidence = ValidatedStageOutcome(
            stage=stage,
            authorization_sha256=authorization,
            output_hashes=(output,),
            validation_record_sha256=output[1],
            compact_artifact=(
                _synthetic_compact_evidence(
                    plan, compact_manifest_sha256, compact_file_hashes
                )
                if stage is PatientStage.FEATURES_VERIFIED
                else None
            ),
        )
        succeeded = validated_success_event(
            events,
            plan,
            evidence,
            existing_output_hashes=None,
            recorded_at_utc=timestamps[index * 2 + 1],
        )
        events.append(succeeded)
        if stage is through_stage:
            break
    replay = replay_events(events)
    expected_action = (
        ReplayAction.STOP_TERMINAL
        if through_stage is PatientStage.TERMINAL_RECORDED
        else ReplayAction.ADVANCE_STAGE
    )
    if replay.action is not expected_action:
        raise ValueError("P0001 synthetic recovery transaction reached an invalid tip")
    return P0001RecoveryRehearsal(
        events=tuple(events),
        final_action=replay.action,
        compact_manifest_sha256=compact_manifest_sha256,
        transaction_id=TRANSACTION_ID,
    )


__all__ = [
    "CHECKPOINT_SHA256",
    "COHORT_INDEX",
    "COORDINATE_MANIFEST_SHA256",
    "FEATURE_DIM",
    "FUTURE_GPU_AUTHORIZATION_STATEMENT_SHA256",
    "GDC_UUID",
    "OMIC_SOURCE_ROW_ID",
    "P0001RecoveryRehearsal",
    "PATIENT_ID",
    "SCALE_2X_ROWS",
    "SCALE_4X_ROWS",
    "SLIDE_ID",
    "TOTAL_ROWS",
    "rehearse_p0001_feature_transaction",
]
