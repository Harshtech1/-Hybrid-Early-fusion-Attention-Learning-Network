"""Non-executable planner for one-patient BRCA streaming transactions.

"Executor" here means a typed stage contract and deterministic recovery plan.
There is deliberately no command, callback, plugin, network, WSI, pixel,
coordinate, CUDA, feature, Drive, deletion, or training operation hook.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from .brca_singleton_streaming_policy import PatientStage
from .brca_streaming_recovery_v2 import (
    EventType,
    FailureClassification,
    PatientTransactionIdentity,
    RecoveryError,
    RecoveryEvent,
    ReplayAction,
    build_outcome_event,
    build_start_event,
    canonical_json,
    compute_idempotency_key,
    freeze_hashes,
    replay_events,
    validate_identity,
)


class ExecutorContractError(ValueError):
    """Raised when a plan exceeds the CPU-only contract or recovery tip."""


class PlanningSurface(str, Enum):
    METADATA_HASH_AND_LEDGER_ONLY = "METADATA_HASH_AND_LEDGER_ONLY"


@dataclass(frozen=True)
class ExecutorPolicyV2:
    cohort_patient_count: int = 894
    patient_transaction_size: int = 1
    patient_concurrency: int = 1
    download_concurrency: int = 1
    gpu_concurrency: int = 1
    next_patient_requires_terminal_record: bool = True
    atomic_no_overwrite_ledger: bool = True
    planning_surface: PlanningSurface = PlanningSurface.METADATA_HASH_AND_LEDGER_ONLY


@dataclass(frozen=True)
class StageContract:
    stage: PatientStage
    required_input_labels: tuple[str, ...]
    required_success_output_labels: tuple[str, ...]
    optional_input_labels: tuple[str, ...] = ()
    separate_authorization_required: bool = True


@dataclass(frozen=True)
class StagePlan:
    identity: PatientTransactionIdentity
    run_id: str
    stage: PatientStage
    attempt_number: int
    replay_action: ReplayAction
    authorization_sha256: str
    retry_authorization_sha256: str | None
    source_policy_hashes: tuple[tuple[str, str], ...]
    input_hashes: tuple[tuple[str, str], ...]
    required_success_output_labels: tuple[str, ...]
    idempotency_key: str
    plan_sha256: str
    planning_surface: PlanningSurface = PlanningSurface.METADATA_HASH_AND_LEDGER_ONLY


FORBIDDEN_EXECUTION_SURFACES = {
    "download": False,
    "network": False,
    "wsi_or_openslide": False,
    "pixel_or_read_region": False,
    "mask_or_coordinate_generation": False,
    "torch_or_cuda": False,
    "feature_extraction": False,
    "healnet": False,
    "google_drive": False,
    "raw_wsi_deletion": False,
    "training": False,
}

REQUIRED_SOURCE_POLICY_LABELS = (
    "cohort_order_manifest",
    "compact_artifact_policy",
    "executor_v2_policy",
    "singleton_streaming_policy",
)

STAGE_CONTRACTS = {
    PatientStage.PLANNED: StageContract(
        PatientStage.PLANNED,
        ("cohort_alignment", "singleton_identity"),
        ("patient_plan",),
        ("supersede_authorization", "superseded_failure"),
    ),
    PatientStage.ACQUISITION_AUTHORIZED: StageContract(
        PatientStage.ACQUISITION_AUTHORIZED,
        ("one_row_manifest",),
        ("acquisition_authorization_record",),
    ),
    PatientStage.RAW_VERIFIED: StageContract(
        PatientStage.RAW_VERIFIED,
        ("raw_identity_declaration",),
        ("raw_verification_report",),
    ),
    PatientStage.HEADER_POLICY_VERIFIED: StageContract(
        PatientStage.HEADER_POLICY_VERIFIED,
        ("raw_verification_report",),
        ("header_policy_report",),
    ),
    PatientStage.COORDINATES_VERIFIED: StageContract(
        PatientStage.COORDINATES_VERIFIED,
        ("header_policy_report",),
        ("coordinate_manifest",),
    ),
    PatientStage.GPU_AUTHORIZED: StageContract(
        PatientStage.GPU_AUTHORIZED,
        ("checkpoint_identity", "coordinate_manifest"),
        ("gpu_authorization_record",),
    ),
    PatientStage.FEATURES_VERIFIED: StageContract(
        PatientStage.FEATURES_VERIFIED,
        ("checkpoint_identity", "coordinate_manifest", "gpu_authorization_record"),
        ("compact_feature_manifest",),
    ),
    PatientStage.TERMINAL_RECORDED: StageContract(
        PatientStage.TERMINAL_RECORDED,
        ("compact_feature_manifest", "raw_lifecycle_decision"),
        ("terminal_record",),
    ),
}


def validate_executor_policy(policy: ExecutorPolicyV2) -> None:
    if policy.cohort_patient_count != 894:
        raise ExecutorContractError("executor is bound to exactly 894 singleton patients")
    if policy.patient_transaction_size != 1:
        raise ExecutorContractError("atomic transaction size must be one patient")
    if (
        policy.patient_concurrency != 1
        or policy.download_concurrency != 1
        or policy.gpu_concurrency != 1
    ):
        raise ExecutorContractError("patient, download, and GPU concurrency must be one")
    if not policy.next_patient_requires_terminal_record:
        raise ExecutorContractError("next patient must require a terminal record")
    if not policy.atomic_no_overwrite_ledger:
        raise ExecutorContractError("ledger must remain atomic no-overwrite")
    if policy.planning_surface is not PlanningSurface.METADATA_HASH_AND_LEDGER_ONLY:
        raise ExecutorContractError("planner cannot expose a real operation surface")


def _plan_payload(plan: StagePlan) -> dict[str, object]:
    return {
        "cohort_index": plan.identity.cohort_index,
        "patient_id": plan.identity.patient_id,
        "slide_id": plan.identity.slide_id,
        "gdc_uuid": plan.identity.gdc_uuid,
        "transaction_id": plan.identity.transaction_id,
        "run_id": plan.run_id,
        "stage": plan.stage.value,
        "attempt_number": plan.attempt_number,
        "replay_action": plan.replay_action.value,
        "authorization_sha256": plan.authorization_sha256,
        "retry_authorization_sha256": plan.retry_authorization_sha256,
        "source_policy_hashes": dict(plan.source_policy_hashes),
        "input_hashes": dict(plan.input_hashes),
        "required_success_output_labels": list(plan.required_success_output_labels),
        "idempotency_key": plan.idempotency_key,
        "planning_surface": plan.planning_surface.value,
    }


def _finalize_plan(provisional: StagePlan) -> StagePlan:
    digest = hashlib.sha256(canonical_json(_plan_payload(provisional))).hexdigest()
    return StagePlan(**{**provisional.__dict__, "plan_sha256": digest})


def _validate_plan(plan: StagePlan) -> None:
    if plan.planning_surface is not PlanningSurface.METADATA_HASH_AND_LEDGER_ONLY:
        raise ExecutorContractError("plan exposes an unauthorized operation surface")
    try:
        validate_identity(plan.identity)
        sources = freeze_hashes(plan.source_policy_hashes, label="source_policy_hashes")
        inputs = freeze_hashes(plan.input_hashes, label="input_hashes")
        freeze_hashes(
            {"authorization": plan.authorization_sha256}, label="authorization"
        )
        if plan.retry_authorization_sha256 is not None:
            freeze_hashes(
                {"retry_authorization": plan.retry_authorization_sha256},
                label="retry_authorization",
            )
        expected_idempotency = compute_idempotency_key(
            identity=plan.identity,
            stage=plan.stage,
            authorization_sha256=plan.authorization_sha256,
            source_policy_hashes=sources,
            input_hashes=inputs,
        )
    except RecoveryError as error:
        raise ExecutorContractError(str(error)) from error
    if tuple(key for key, _ in sources) != tuple(sorted(REQUIRED_SOURCE_POLICY_LABELS)):
        raise ExecutorContractError("plan source policy labels violate the frozen contract")
    contract = STAGE_CONTRACTS[plan.stage]
    if plan.required_success_output_labels != contract.required_success_output_labels:
        raise ExecutorContractError("plan success output labels violate the frozen stage contract")
    input_labels = {key for key, _ in inputs}
    if not set(contract.required_input_labels) <= input_labels:
        raise ExecutorContractError("plan is missing frozen stage input bindings")
    allowed_inputs = (
        set(contract.required_input_labels)
        | set(contract.optional_input_labels)
        | {"previous_record", "previous_success_record", "prior_failure"}
    )
    if not input_labels <= allowed_inputs:
        raise ExecutorContractError("plan contains unexpected stage input bindings")
    optional_present = input_labels & set(contract.optional_input_labels)
    if optional_present and optional_present != set(contract.optional_input_labels):
        raise ExecutorContractError("supersede input bindings must be supplied together")
    if plan.attempt_number < 1:
        raise ExecutorContractError("plan attempt number must be positive")
    if expected_idempotency != plan.idempotency_key:
        raise ExecutorContractError("plan idempotency key violates semantic bindings")
    expected = hashlib.sha256(canonical_json(_plan_payload(plan))).hexdigest()
    if plan.plan_sha256 != expected:
        raise ExecutorContractError("stage plan SHA256 mismatch")


def plan_next_stage(
    existing: Sequence[RecoveryEvent],
    *,
    identity: PatientTransactionIdentity,
    run_id: str,
    authorization_sha256: str,
    retry_authorization_sha256: str | None,
    source_policy_hashes: Mapping[str, str],
    stage_input_hashes: Mapping[str, str],
) -> StagePlan:
    """Plan only the next start allowed by replay; perform no project operation."""

    replay = replay_events(existing)
    if replay.action in {
        ReplayAction.COMPLETE_STARTED_STAGE,
        ReplayAction.BLOCKED_NON_RETRYABLE,
        ReplayAction.BLOCKED_RETRY_LIMIT,
        ReplayAction.STOP_TERMINAL,
    } or replay.target_stage is None:
        raise ExecutorContractError(f"cannot plan stage while action is {replay.action.value}")
    if replay.identity is not None and identity != replay.identity:
        raise ExecutorContractError("patient transaction identity does not match ledger")
    contract = STAGE_CONTRACTS[replay.target_stage]
    supplied = dict(stage_input_hashes)
    missing = sorted(set(contract.required_input_labels) - supplied.keys())
    allowed = set(contract.required_input_labels) | set(contract.optional_input_labels)
    unexpected = sorted(set(supplied) - allowed)
    if missing or unexpected:
        raise ExecutorContractError(
            f"stage input labels mismatch: missing={missing}, unexpected={unexpected}"
        )
    optional_present = set(supplied) & set(contract.optional_input_labels)
    if optional_present and optional_present != set(contract.optional_input_labels):
        raise ExecutorContractError("supersede input bindings must be supplied together")
    source_missing = sorted(set(REQUIRED_SOURCE_POLICY_LABELS) - source_policy_hashes.keys())
    source_unexpected = sorted(
        set(source_policy_hashes) - set(REQUIRED_SOURCE_POLICY_LABELS)
    )
    if source_missing or source_unexpected:
        raise ExecutorContractError(
            "source policy labels mismatch: "
            f"missing={source_missing}, unexpected={source_unexpected}"
        )
    if existing:
        supplied["previous_record"] = replay.previous_record_sha256
        supplied["previous_success_record"] = replay.previous_success_record_sha256
        if replay.action is ReplayAction.RETRY_FAILED_STAGE:
            if replay.failed_event_sha256 is None:
                raise ExecutorContractError("retry plan lacks prior failure binding")
            supplied["prior_failure"] = replay.failed_event_sha256
    try:
        sources = freeze_hashes(source_policy_hashes, label="source_policy_hashes")
        inputs = freeze_hashes(supplied, label="input_hashes")
        idempotency_key = compute_idempotency_key(
            identity=identity,
            stage=replay.target_stage,
            authorization_sha256=authorization_sha256,
            source_policy_hashes=sources,
            input_hashes=inputs,
        )
        # Authoritative validation occurs by constructing, but not publishing,
        # the synthetic start event represented by this plan.
        build_start_event(
            existing,
            identity=identity,
            run_id=run_id,
            authorization_sha256=authorization_sha256,
            retry_authorization_sha256=retry_authorization_sha256,
            source_policy_hashes=dict(sources),
            input_hashes=dict(inputs),
            recorded_at_utc=(
                existing[-1].recorded_at_utc
                if existing
                else "2000-01-01T00:00:00Z"
            ),
        )
    except RecoveryError as error:
        raise ExecutorContractError(str(error)) from error
    return _finalize_plan(
        StagePlan(
            identity=identity,
            run_id=run_id,
            stage=replay.target_stage,
            attempt_number=replay.attempt_number or 0,
            replay_action=replay.action,
            authorization_sha256=authorization_sha256,
            retry_authorization_sha256=retry_authorization_sha256,
            source_policy_hashes=sources,
            input_hashes=inputs,
            required_success_output_labels=contract.required_success_output_labels,
            idempotency_key=idempotency_key,
            plan_sha256="",
        )
    )


def start_event_from_plan(
    existing: Sequence[RecoveryEvent], plan: StagePlan, *, recorded_at_utc: str
) -> RecoveryEvent:
    _validate_plan(plan)
    try:
        event = build_start_event(
            existing,
            identity=plan.identity,
            run_id=plan.run_id,
            authorization_sha256=plan.authorization_sha256,
            retry_authorization_sha256=plan.retry_authorization_sha256,
            source_policy_hashes=dict(plan.source_policy_hashes),
            input_hashes=dict(plan.input_hashes),
            recorded_at_utc=recorded_at_utc,
        )
    except RecoveryError as error:
        raise ExecutorContractError(str(error)) from error
    if (
        event.stage is not plan.stage
        or event.attempt_number != plan.attempt_number
        or event.idempotency_key != plan.idempotency_key
    ):
        raise ExecutorContractError("stage start does not match its plan")
    return event


def resume_pending_plan(existing: Sequence[RecoveryEvent]) -> StagePlan:
    """Deterministically reconstruct the plan for a crash after STAGE_STARTED."""

    replay = replay_events(existing)
    if replay.action is not ReplayAction.COMPLETE_STARTED_STAGE or not existing:
        raise ExecutorContractError("ledger has no pending started stage")
    start = existing[-1]
    contract = STAGE_CONTRACTS[start.stage]
    return _finalize_plan(
        StagePlan(
            identity=start.identity,
            run_id=start.run_id,
            stage=start.stage,
            attempt_number=start.attempt_number,
            replay_action=ReplayAction.COMPLETE_STARTED_STAGE,
            authorization_sha256=start.authorization_sha256,
            retry_authorization_sha256=start.retry_authorization_sha256,
            source_policy_hashes=start.source_policy_hashes,
            input_hashes=start.input_hashes,
            required_success_output_labels=contract.required_success_output_labels,
            idempotency_key=start.idempotency_key,
            plan_sha256="",
        )
    )


def outcome_event_from_plan(
    existing: Sequence[RecoveryEvent],
    plan: StagePlan,
    *,
    succeeded: bool,
    output_hashes: Mapping[str, str],
    existing_output_hashes: Mapping[str, str] | None,
    outputs_validated: bool,
    failure_classification: FailureClassification | None,
    error_code: str | None,
    error_class: str | None,
    error_message: str | None,
    recorded_at_utc: str,
) -> RecoveryEvent:
    _validate_plan(plan)
    replay = replay_events(existing)
    if (
        replay.action is not ReplayAction.COMPLETE_STARTED_STAGE
        or replay.pending_run_id != plan.run_id
        or replay.target_stage is not plan.stage
        or replay.attempt_number != plan.attempt_number
    ):
        raise ExecutorContractError("outcome plan does not match the pending start")
    start = existing[-1]
    if (
        start.authorization_sha256 != plan.authorization_sha256
        or start.retry_authorization_sha256 != plan.retry_authorization_sha256
        or start.source_policy_hashes != plan.source_policy_hashes
        or start.input_hashes != plan.input_hashes
        or start.idempotency_key != plan.idempotency_key
    ):
        raise ExecutorContractError("outcome plan semantic bindings differ from stage start")
    labels = set(output_hashes)
    required = set(plan.required_success_output_labels)
    if succeeded and labels != required:
        raise ExecutorContractError("success outputs do not match the frozen stage contract")
    if succeeded and not outputs_validated:
        raise ExecutorContractError("success requires explicit artifact/hash validation")
    if not succeeded and labels:
        raise ExecutorContractError("failure cannot claim output hashes")
    try:
        return build_outcome_event(
            existing,
            event_type=(EventType.STAGE_SUCCEEDED if succeeded else EventType.STAGE_FAILED),
            output_hashes=output_hashes,
            existing_output_hashes=existing_output_hashes,
            outputs_validated=outputs_validated,
            failure_classification=failure_classification,
            error_code=error_code,
            error_class=error_class,
            error_message=error_message,
            recorded_at_utc=recorded_at_utc,
        )
    except RecoveryError as error:
        raise ExecutorContractError(str(error)) from error


def validate_next_patient_gate(
    current_patient_events: Sequence[RecoveryEvent],
    *,
    next_identity: PatientTransactionIdentity,
) -> None:
    try:
        validate_identity(next_identity)
    except RecoveryError as error:
        raise ExecutorContractError(str(error)) from error
    if not current_patient_events:
        if next_identity.cohort_index != 1:
            raise ExecutorContractError("first cohort transaction must have cohort_index 1")
        return
    replay = replay_events(current_patient_events)
    if replay.action is not ReplayAction.STOP_TERMINAL:
        raise ExecutorContractError("next patient blocked until terminal success")
    if replay.identity == next_identity:
        raise ExecutorContractError("next transaction must be a different patient")
    if replay.identity is None:
        raise ExecutorContractError("terminal replay lacks patient identity")
    if next_identity.cohort_index != replay.identity.cohort_index + 1:
        raise ExecutorContractError("next patient must be the next cohort manifest index")
    if next_identity.patient_id == replay.identity.patient_id:
        raise ExecutorContractError("patient ID cannot repeat in the cohort queue")


def validate_superseding_transaction_gate(
    failed_events: Sequence[RecoveryEvent],
    *,
    superseding_identity: PatientTransactionIdentity,
    failure_sha256: str,
    supersede_authorization_sha256: str,
) -> dict[str, str]:
    """Validate bindings for a separately approved redo ledger; run nothing."""

    replay = replay_events(failed_events)
    if replay.action not in {
        ReplayAction.BLOCKED_NON_RETRYABLE,
        ReplayAction.BLOCKED_RETRY_LIMIT,
    }:
        raise ExecutorContractError("superseding transaction requires a terminal failure block")
    if replay.identity is None or replay.failed_event_sha256 != failure_sha256:
        raise ExecutorContractError("supersede binding does not match the failed transaction")
    try:
        validate_identity(superseding_identity)
        freeze_hashes(
            {"supersede_authorization": supersede_authorization_sha256},
            label="supersede_authorization",
        )
    except RecoveryError as error:
        raise ExecutorContractError(str(error)) from error
    previous = replay.identity
    if (
        superseding_identity.cohort_index != previous.cohort_index
        or superseding_identity.patient_id != previous.patient_id
        or superseding_identity.slide_id != previous.slide_id
        or superseding_identity.gdc_uuid != previous.gdc_uuid
        or superseding_identity.transaction_id == previous.transaction_id
    ):
        raise ExecutorContractError(
            "superseding ledger must preserve cohort/patient/slide/UUID and use a new transaction"
        )
    return {
        "superseded_failure": failure_sha256,
        "supersede_authorization": supersede_authorization_sha256,
    }


__all__ = [
    "ExecutorContractError",
    "ExecutorPolicyV2",
    "FORBIDDEN_EXECUTION_SURFACES",
    "PlanningSurface",
    "REQUIRED_SOURCE_POLICY_LABELS",
    "STAGE_CONTRACTS",
    "StageContract",
    "StagePlan",
    "outcome_event_from_plan",
    "plan_next_stage",
    "resume_pending_plan",
    "start_event_from_plan",
    "validate_executor_policy",
    "validate_next_patient_gate",
    "validate_superseding_transaction_gate",
]
