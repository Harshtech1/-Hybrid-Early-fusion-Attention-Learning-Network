"""Recovery-only state machine and immutable event ledger for BRCA v2.

This module records declared stage starts and outcomes and deterministically
replays them.  It contains no downloader, network client, OpenSlide, pixel,
coordinate, Torch, CUDA, feature, Drive, raw-file deletion, or training
surface.  Its only side effect is atomic publication of JSON events into a
caller-supplied ledger directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Mapping, Sequence

from .brca_singleton_streaming_policy import PatientStage


GENESIS_HASH = "0" * 64
MAX_ATTEMPTS_PER_STAGE = 3
MAX_EVENT_BYTES = 1_000_000
RETRYABLE_ERROR_CODES = frozenset(
    {"RESOURCE_TEMPORARILY_UNAVAILABLE", "SESSION_INTERRUPTED", "TRANSIENT_IO"}
)
_PATIENT = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")
_HASH_LABEL = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_NEXT_STAGE = {
    PatientStage.PLANNED: PatientStage.ACQUISITION_AUTHORIZED,
    PatientStage.ACQUISITION_AUTHORIZED: PatientStage.RAW_VERIFIED,
    PatientStage.RAW_VERIFIED: PatientStage.HEADER_POLICY_VERIFIED,
    PatientStage.HEADER_POLICY_VERIFIED: PatientStage.COORDINATES_VERIFIED,
    PatientStage.COORDINATES_VERIFIED: PatientStage.GPU_AUTHORIZED,
    PatientStage.GPU_AUTHORIZED: PatientStage.FEATURES_VERIFIED,
    PatientStage.FEATURES_VERIFIED: PatientStage.TERMINAL_RECORDED,
}


class RecoveryError(RuntimeError):
    """Raised when construction, publication, or replay must stop."""


class EventType(str, Enum):
    STAGE_STARTED = "STAGE_STARTED"
    STAGE_SUCCEEDED = "STAGE_SUCCEEDED"
    STAGE_FAILED = "STAGE_FAILED"


class AttemptDecision(str, Enum):
    INITIAL = "INITIAL"
    ADVANCE = "ADVANCE"
    RETRY = "RETRY"


class FailureClassification(str, Enum):
    RETRYABLE = "RETRYABLE"
    NON_RETRYABLE = "NON_RETRYABLE"


class ReplayAction(str, Enum):
    START_PLANNING = "START_PLANNING"
    ADVANCE_STAGE = "ADVANCE_STAGE"
    COMPLETE_STARTED_STAGE = "COMPLETE_STARTED_STAGE"
    RETRY_FAILED_STAGE = "RETRY_FAILED_STAGE"
    BLOCKED_NON_RETRYABLE = "BLOCKED_NON_RETRYABLE"
    BLOCKED_RETRY_LIMIT = "BLOCKED_RETRY_LIMIT"
    STOP_TERMINAL = "STOP_TERMINAL"


class OutputDisposition(str, Enum):
    PUBLISH_NEW = "PUBLISH_NEW"
    VALIDATE_AND_REUSE = "VALIDATE_AND_REUSE"


class CrashDisposition(str, Enum):
    CLEAN = "CLEAN"
    UNPUBLISHED_STAGING_MANUAL_REVIEW_REQUIRED = (
        "UNPUBLISHED_STAGING_MANUAL_REVIEW_REQUIRED"
    )
    REDUNDANT_STAGING_MANUAL_CLEANUP_REQUIRED = (
        "REDUNDANT_STAGING_MANUAL_CLEANUP_REQUIRED"
    )
    AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED = "AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED"


@dataclass(frozen=True)
class PatientTransactionIdentity:
    cohort_index: int
    patient_id: str
    slide_id: str
    gdc_uuid: str
    transaction_id: str


@dataclass(frozen=True)
class RecoveryEvent:
    sequence: int
    identity: PatientTransactionIdentity
    run_id: str
    attempt_number: int
    decision: AttemptDecision
    stage: PatientStage
    event_type: EventType
    previous_record_sha256: str
    authorization_sha256: str
    retry_authorization_sha256: str | None
    source_policy_hashes: tuple[tuple[str, str], ...]
    input_hashes: tuple[tuple[str, str], ...]
    output_hashes: tuple[tuple[str, str], ...]
    output_disposition: OutputDisposition | None
    idempotency_key: str
    failure_classification: FailureClassification | None
    error_code: str | None
    error_class: str | None
    error_message: str | None
    recorded_at_utc: str
    record_sha256: str


@dataclass(frozen=True)
class ReplayPlan:
    identity: PatientTransactionIdentity | None
    action: ReplayAction
    last_durable_stage: PatientStage | None
    target_stage: PatientStage | None
    attempt_number: int | None
    decision: AttemptDecision | None
    previous_record_sha256: str
    previous_success_record_sha256: str
    failed_event_sha256: str | None
    pending_start_event_sha256: str | None
    pending_run_id: str | None


def canonical_json(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )


def _validate_sha256(value: str, label: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise RecoveryError(f"{label} must be a lowercase SHA256")


def _validate_uuid(value: str, label: str) -> None:
    try:
        parsed = uuid.UUID(value)
    except (AttributeError, ValueError) as error:
        raise RecoveryError(f"{label} must be a canonical UUID") from error
    if str(parsed) != value:
        raise RecoveryError(f"{label} must be a lowercase canonical UUID")


def validate_identity(identity: PatientTransactionIdentity) -> None:
    if isinstance(identity.cohort_index, bool) or not isinstance(identity.cohort_index, int):
        raise RecoveryError("cohort_index must be an integer")
    if not 1 <= identity.cohort_index <= 894:
        raise RecoveryError("cohort_index must be within the 894-patient cohort")
    if not _PATIENT.fullmatch(identity.patient_id):
        raise RecoveryError("patient_id is not an exact TCGA case identifier")
    if (
        not identity.slide_id.startswith(f"{identity.patient_id}-")
        or not identity.slide_id.endswith(".svs")
        or "/" in identity.slide_id
        or "\\" in identity.slide_id
    ):
        raise RecoveryError("slide_id is not an exact patient-bound SVS filename")
    _validate_uuid(identity.gdc_uuid, "gdc_uuid")
    _validate_uuid(identity.transaction_id, "transaction_id")


def _validate_timestamp(value: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise RecoveryError("recorded_at_utc must use UTC Z notation")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as error:
        raise RecoveryError("recorded_at_utc is not ISO-8601") from error
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise RecoveryError("recorded_at_utc must be UTC")
    return parsed


def freeze_hashes(
    values: Mapping[str, str] | Sequence[tuple[str, str]], *, label: str
) -> tuple[tuple[str, str], ...]:
    if isinstance(values, Mapping):
        items = tuple(values.items())
    else:
        items = tuple(values)
    if len({key for key, _ in items}) != len(items):
        raise RecoveryError(f"{label} contains a duplicate label")
    for key, value in items:
        if not isinstance(key, str) or not _HASH_LABEL.fullmatch(key):
            raise RecoveryError(f"{label} labels must be lowercase snake-case tokens")
        _validate_sha256(value, f"{label}.{key}")
    return tuple(sorted(items))


def compute_idempotency_key(
    *,
    identity: PatientTransactionIdentity,
    stage: PatientStage,
    authorization_sha256: str,
    source_policy_hashes: Mapping[str, str] | Sequence[tuple[str, str]],
    input_hashes: Mapping[str, str] | Sequence[tuple[str, str]],
) -> str:
    """Bind durable identity and stage inputs; retry number is deliberately excluded."""

    validate_identity(identity)
    _validate_sha256(authorization_sha256, "authorization_sha256")
    sources = freeze_hashes(source_policy_hashes, label="source_policy_hashes")
    inputs = freeze_hashes(input_hashes, label="input_hashes")
    stage_inputs = tuple(
        item
        for item in inputs
        if item[0] not in {"previous_record", "previous_success_record", "prior_failure"}
    )
    payload = {
        "cohort_index": identity.cohort_index,
        "patient_id": identity.patient_id,
        "slide_id": identity.slide_id,
        "gdc_uuid": identity.gdc_uuid,
        "transaction_id": identity.transaction_id,
        "stage": stage.value,
        "authorization_sha256": authorization_sha256,
        "source_policy_hashes": dict(sources),
        "stage_input_hashes": dict(stage_inputs),
    }
    return hashlib.sha256(canonical_json(payload)).hexdigest()


def _payload(event: RecoveryEvent) -> dict[str, object]:
    return {
        "sequence": event.sequence,
        "cohort_index": event.identity.cohort_index,
        "patient_id": event.identity.patient_id,
        "slide_id": event.identity.slide_id,
        "gdc_uuid": event.identity.gdc_uuid,
        "transaction_id": event.identity.transaction_id,
        "run_id": event.run_id,
        "attempt_number": event.attempt_number,
        "decision": event.decision.value,
        "stage": event.stage.value,
        "event_type": event.event_type.value,
        "previous_record_sha256": event.previous_record_sha256,
        "authorization_sha256": event.authorization_sha256,
        "retry_authorization_sha256": event.retry_authorization_sha256,
        "source_policy_hashes": dict(event.source_policy_hashes),
        "input_hashes": dict(event.input_hashes),
        "output_hashes": dict(event.output_hashes),
        "output_disposition": (
            None if event.output_disposition is None else event.output_disposition.value
        ),
        "idempotency_key": event.idempotency_key,
        "failure_classification": (
            None
            if event.failure_classification is None
            else event.failure_classification.value
        ),
        "error_code": event.error_code,
        "error_class": event.error_class,
        "error_message": event.error_message,
        "recorded_at_utc": event.recorded_at_utc,
    }


def _expected_start(
    *,
    durable_stage: PatientStage | None,
    failure: RecoveryEvent | None,
) -> tuple[PatientStage, int, AttemptDecision, ReplayAction]:
    if failure is not None:
        if failure.failure_classification is FailureClassification.NON_RETRYABLE:
            raise RecoveryError("non-retryable failure blocks automatic replay")
        if failure.attempt_number >= MAX_ATTEMPTS_PER_STAGE:
            raise RecoveryError("maximum retry attempts reached")
        return (
            failure.stage,
            failure.attempt_number + 1,
            AttemptDecision.RETRY,
            ReplayAction.RETRY_FAILED_STAGE,
        )
    if durable_stage is None:
        return (
            PatientStage.PLANNED,
            1,
            AttemptDecision.INITIAL,
            ReplayAction.START_PLANNING,
        )
    if durable_stage is PatientStage.TERMINAL_RECORDED:
        raise RecoveryError("terminal patient transaction cannot reopen")
    return (
        _NEXT_STAGE[durable_stage],
        1,
        AttemptDecision.ADVANCE,
        ReplayAction.ADVANCE_STAGE,
    )


def replay_events(events: Sequence[RecoveryEvent]) -> ReplayPlan:
    """Validate the full chain and derive its one deterministic next action."""

    identity: PatientTransactionIdentity | None = None
    durable_stage: PatientStage | None = None
    durable_sha = GENESIS_HASH
    failure: RecoveryEvent | None = None
    pending: RecoveryEvent | None = None
    start_run_ids: set[str] = set()
    last_timestamp: datetime | None = None

    for index, event in enumerate(events):
        if event.sequence != index + 1:
            raise RecoveryError("event sequence is not contiguous")
        validate_identity(event.identity)
        _validate_uuid(event.run_id, "run_id")
        timestamp = _validate_timestamp(event.recorded_at_utc)
        if last_timestamp is not None and timestamp < last_timestamp:
            raise RecoveryError("event timestamps must be nondecreasing")
        last_timestamp = timestamp
        _validate_sha256(event.previous_record_sha256, "previous_record_sha256")
        _validate_sha256(event.authorization_sha256, "authorization_sha256")
        if event.retry_authorization_sha256 is not None:
            _validate_sha256(
                event.retry_authorization_sha256, "retry_authorization_sha256"
            )
        _validate_sha256(event.idempotency_key, "idempotency_key")
        _validate_sha256(event.record_sha256, "record_sha256")
        sources = freeze_hashes(event.source_policy_hashes, label="source_policy_hashes")
        inputs = freeze_hashes(event.input_hashes, label="input_hashes")
        outputs = freeze_hashes(event.output_hashes, label="output_hashes")
        if sources != event.source_policy_hashes or inputs != event.input_hashes:
            raise RecoveryError("event hash maps are not in canonical order")
        if outputs != event.output_hashes:
            raise RecoveryError("event output hashes are not in canonical order")
        if not sources or not inputs:
            raise RecoveryError("source policy and input hash maps cannot be empty")
        if identity is None:
            identity = event.identity
        elif event.identity != identity:
            raise RecoveryError("cohort index, patient, slide, UUID, or transaction drift")
        expected_previous = GENESIS_HASH if index == 0 else events[index - 1].record_sha256
        if event.previous_record_sha256 != expected_previous:
            raise RecoveryError("event hash chain is broken")
        expected_key = compute_idempotency_key(
            identity=event.identity,
            stage=event.stage,
            authorization_sha256=event.authorization_sha256,
            source_policy_hashes=event.source_policy_hashes,
            input_hashes=event.input_hashes,
        )
        if event.idempotency_key != expected_key:
            raise RecoveryError("idempotency key does not bind identity and stage inputs")
        expected_record = hashlib.sha256(canonical_json(_payload(event))).hexdigest()
        if event.record_sha256 != expected_record:
            raise RecoveryError("event record SHA256 mismatch")

        if event.event_type is EventType.STAGE_STARTED:
            if pending is not None:
                raise RecoveryError("a stage outcome is required before another start")
            stage, attempt, decision, _ = _expected_start(
                durable_stage=durable_stage, failure=failure
            )
            if event.stage is not stage:
                raise RecoveryError(f"stage must be exactly {stage.value}; no skipping")
            if event.attempt_number != attempt or event.decision is not decision:
                raise RecoveryError("attempt number or replay decision is not deterministic")
            if decision is AttemptDecision.RETRY:
                if event.retry_authorization_sha256 is None:
                    raise RecoveryError("retry start requires explicit retry authorization")
                prior_start = events[index - 2]
                if (
                    event.authorization_sha256 != prior_start.authorization_sha256
                    or event.source_policy_hashes != prior_start.source_policy_hashes
                    or event.idempotency_key != prior_start.idempotency_key
                ):
                    raise RecoveryError(
                        "retry must preserve stage authorization, policy, inputs, and idempotency"
                    )
            elif event.retry_authorization_sha256 is not None:
                raise RecoveryError("non-retry start cannot contain retry authorization")
            if event.run_id in start_run_ids:
                raise RecoveryError("run_id cannot be reused for another attempt")
            start_run_ids.add(event.run_id)
            if index:
                bindings = dict(event.input_hashes)
                if bindings.get("previous_record") != expected_previous:
                    raise RecoveryError("stage start does not bind previous_record")
                if bindings.get("previous_success_record") != durable_sha:
                    raise RecoveryError("stage start does not bind previous_success_record")
                if failure is not None and bindings.get("prior_failure") != failure.record_sha256:
                    raise RecoveryError("retry does not bind the prior failure")
                if failure is None and "prior_failure" in bindings:
                    raise RecoveryError("advance stage cannot claim a prior failure")
            if event.output_hashes:
                raise RecoveryError("stage start cannot claim outputs")
            if event.output_disposition is not None:
                raise RecoveryError("stage start cannot claim an output disposition")
            if any(
                value is not None
                for value in (
                    event.failure_classification,
                    event.error_code,
                    event.error_class,
                    event.error_message,
                )
            ):
                raise RecoveryError("stage start cannot contain failure details")
            pending = event
            continue

        if pending is None:
            raise RecoveryError("stage outcome requires an immediately preceding start")
        if (
            event.stage is not pending.stage
            or event.run_id != pending.run_id
            or event.attempt_number != pending.attempt_number
            or event.decision is not pending.decision
            or event.authorization_sha256 != pending.authorization_sha256
            or event.retry_authorization_sha256 != pending.retry_authorization_sha256
            or event.source_policy_hashes != pending.source_policy_hashes
            or event.input_hashes != pending.input_hashes
            or event.idempotency_key != pending.idempotency_key
        ):
            raise RecoveryError("stage outcome does not match its immutable start record")
        if event.event_type is EventType.STAGE_SUCCEEDED:
            if not event.output_hashes:
                raise RecoveryError("successful stage requires output hashes")
            if event.output_disposition is None:
                raise RecoveryError("successful stage requires validated output disposition")
            if any(
                value is not None
                for value in (
                    event.failure_classification,
                    event.error_code,
                    event.error_class,
                    event.error_message,
                )
            ):
                raise RecoveryError("successful stage cannot contain failure details")
            durable_stage = event.stage
            durable_sha = event.record_sha256
            failure = None
        else:
            if event.output_hashes:
                raise RecoveryError("failed stage cannot claim outputs")
            if event.output_disposition is not None:
                raise RecoveryError("failed stage cannot contain output disposition")
            if (
                event.failure_classification is None
                or not event.error_code
                or not event.error_class
                or not event.error_message
            ):
                raise RecoveryError("failed stage requires classification, class, and message")
            retryable_code = event.error_code in RETRYABLE_ERROR_CODES
            if retryable_code != (
                event.failure_classification is FailureClassification.RETRYABLE
            ):
                raise RecoveryError("failure classification violates frozen error-code policy")
            failure = event
        pending = None

    previous = GENESIS_HASH if not events else events[-1].record_sha256
    if pending is not None:
        return ReplayPlan(
            identity=identity,
            action=ReplayAction.COMPLETE_STARTED_STAGE,
            last_durable_stage=durable_stage,
            target_stage=pending.stage,
            attempt_number=pending.attempt_number,
            decision=pending.decision,
            previous_record_sha256=previous,
            previous_success_record_sha256=durable_sha,
            failed_event_sha256=None if failure is None else failure.record_sha256,
            pending_start_event_sha256=pending.record_sha256,
            pending_run_id=pending.run_id,
        )
    if failure is not None:
        if failure.failure_classification is FailureClassification.NON_RETRYABLE:
            action = ReplayAction.BLOCKED_NON_RETRYABLE
        elif failure.attempt_number >= MAX_ATTEMPTS_PER_STAGE:
            action = ReplayAction.BLOCKED_RETRY_LIMIT
        else:
            action = ReplayAction.RETRY_FAILED_STAGE
        return ReplayPlan(
            identity=identity,
            action=action,
            last_durable_stage=durable_stage,
            target_stage=(failure.stage if action is ReplayAction.RETRY_FAILED_STAGE else None),
            attempt_number=(
                failure.attempt_number + 1
                if action is ReplayAction.RETRY_FAILED_STAGE
                else None
            ),
            decision=(AttemptDecision.RETRY if action is ReplayAction.RETRY_FAILED_STAGE else None),
            previous_record_sha256=previous,
            previous_success_record_sha256=durable_sha,
            failed_event_sha256=failure.record_sha256,
            pending_start_event_sha256=None,
            pending_run_id=None,
        )
    if durable_stage is PatientStage.TERMINAL_RECORDED:
        return ReplayPlan(
            identity=identity,
            action=ReplayAction.STOP_TERMINAL,
            last_durable_stage=durable_stage,
            target_stage=None,
            attempt_number=None,
            decision=None,
            previous_record_sha256=previous,
            previous_success_record_sha256=durable_sha,
            failed_event_sha256=None,
            pending_start_event_sha256=None,
            pending_run_id=None,
        )
    stage, attempt, decision, action = _expected_start(
        durable_stage=durable_stage, failure=None
    )
    return ReplayPlan(
        identity=identity,
        action=action,
        last_durable_stage=durable_stage,
        target_stage=stage,
        attempt_number=attempt,
        decision=decision,
        previous_record_sha256=previous,
        previous_success_record_sha256=durable_sha,
        failed_event_sha256=None,
        pending_start_event_sha256=None,
        pending_run_id=None,
    )


def _finalize(provisional: RecoveryEvent) -> RecoveryEvent:
    digest = hashlib.sha256(canonical_json(_payload(provisional))).hexdigest()
    event = RecoveryEvent(**{**provisional.__dict__, "record_sha256": digest})
    serialized = canonical_json({**_payload(event), "record_sha256": event.record_sha256})
    if len(serialized) > MAX_EVENT_BYTES:
        raise RecoveryError("ledger event exceeds the maximum serialized size")
    return event


def build_start_event(
    existing: Sequence[RecoveryEvent],
    *,
    identity: PatientTransactionIdentity,
    run_id: str,
    authorization_sha256: str,
    retry_authorization_sha256: str | None,
    source_policy_hashes: Mapping[str, str],
    input_hashes: Mapping[str, str],
    recorded_at_utc: str,
) -> RecoveryEvent:
    """Build the sole stage start allowed by replay; no operation is run."""

    plan = replay_events(existing)
    if plan.action in {
        ReplayAction.COMPLETE_STARTED_STAGE,
        ReplayAction.BLOCKED_NON_RETRYABLE,
        ReplayAction.BLOCKED_RETRY_LIMIT,
        ReplayAction.STOP_TERMINAL,
    }:
        raise RecoveryError(f"cannot start stage while replay action is {plan.action.value}")
    validate_identity(identity)
    if plan.identity is not None and identity != plan.identity:
        raise RecoveryError("patient transaction identity drift")
    _validate_uuid(run_id, "run_id")
    _validate_sha256(authorization_sha256, "authorization_sha256")
    if plan.action is ReplayAction.RETRY_FAILED_STAGE:
        if retry_authorization_sha256 is None:
            raise RecoveryError("retry start requires explicit retry authorization")
        _validate_sha256(retry_authorization_sha256, "retry_authorization_sha256")
    elif retry_authorization_sha256 is not None:
        raise RecoveryError("non-retry start cannot contain retry authorization")
    _validate_timestamp(recorded_at_utc)
    sources = freeze_hashes(source_policy_hashes, label="source_policy_hashes")
    inputs = freeze_hashes(input_hashes, label="input_hashes")
    if not sources or not inputs:
        raise RecoveryError("source policy and input hash maps cannot be empty")
    if not existing and {
        "previous_record",
        "previous_success_record",
        "prior_failure",
    } & set(dict(inputs)):
        raise RecoveryError("genesis stage cannot contain chain or failure bindings")
    if existing:
        bindings = dict(inputs)
        if bindings.get("previous_record") != plan.previous_record_sha256:
            raise RecoveryError("input hashes must bind previous_record")
        if bindings.get("previous_success_record") != plan.previous_success_record_sha256:
            raise RecoveryError("input hashes must bind previous_success_record")
        if plan.action is ReplayAction.RETRY_FAILED_STAGE:
            if bindings.get("prior_failure") != plan.failed_event_sha256:
                raise RecoveryError("retry input hashes must bind prior_failure")
        elif "prior_failure" in bindings:
            raise RecoveryError("non-retry start cannot bind prior_failure")
    provisional = RecoveryEvent(
        sequence=len(existing) + 1,
        identity=identity,
        run_id=run_id,
        attempt_number=plan.attempt_number or 0,
        decision=plan.decision or AttemptDecision.INITIAL,
        stage=plan.target_stage or PatientStage.PLANNED,
        event_type=EventType.STAGE_STARTED,
        previous_record_sha256=plan.previous_record_sha256,
        authorization_sha256=authorization_sha256,
        retry_authorization_sha256=retry_authorization_sha256,
        source_policy_hashes=sources,
        input_hashes=inputs,
        output_hashes=(),
        output_disposition=None,
        idempotency_key=compute_idempotency_key(
            identity=identity,
            stage=plan.target_stage or PatientStage.PLANNED,
            authorization_sha256=authorization_sha256,
            source_policy_hashes=sources,
            input_hashes=inputs,
        ),
        failure_classification=None,
        error_code=None,
        error_class=None,
        error_message=None,
        recorded_at_utc=recorded_at_utc,
        record_sha256="",
    )
    event = _finalize(provisional)
    replay_events((*existing, event))
    return event


def build_outcome_event(
    existing: Sequence[RecoveryEvent],
    *,
    event_type: EventType,
    output_hashes: Mapping[str, str],
    existing_output_hashes: Mapping[str, str] | None,
    outputs_validated: bool,
    failure_classification: FailureClassification | None,
    error_code: str | None,
    error_class: str | None,
    error_message: str | None,
    recorded_at_utc: str,
) -> RecoveryEvent:
    """Build success/failure only for the pending immutable stage start."""

    if event_type not in {EventType.STAGE_SUCCEEDED, EventType.STAGE_FAILED}:
        raise RecoveryError("outcome event must be STAGE_SUCCEEDED or STAGE_FAILED")
    plan = replay_events(existing)
    if plan.action is not ReplayAction.COMPLETE_STARTED_STAGE or not existing:
        raise RecoveryError("no pending stage start to complete")
    start = existing[-1]
    outputs = freeze_hashes(output_hashes, label="output_hashes")
    _validate_timestamp(recorded_at_utc)
    if event_type is EventType.STAGE_SUCCEEDED:
        if not outputs:
            raise RecoveryError("successful stage requires output hashes")
        if not outputs_validated:
            raise RecoveryError("successful stage requires explicit artifact/hash validation")
        output_disposition = resolve_output_collision(
            dict(outputs), existing_output_hashes
        )
        if any(
            value is not None
            for value in (failure_classification, error_code, error_class, error_message)
        ):
            raise RecoveryError("successful stage cannot contain failure details")
    else:
        if outputs:
            raise RecoveryError("failed stage cannot claim output hashes")
        if existing_output_hashes is not None or outputs_validated:
            raise RecoveryError("failed stage cannot claim output validation or collision reuse")
        output_disposition = None
        if failure_classification is None or not error_code or not error_class or not error_message:
            raise RecoveryError("failed stage requires classification, class, and message")
        retryable_code = error_code in RETRYABLE_ERROR_CODES
        if retryable_code != (failure_classification is FailureClassification.RETRYABLE):
            raise RecoveryError("failure classification violates frozen error-code policy")
    provisional = RecoveryEvent(
        sequence=len(existing) + 1,
        identity=start.identity,
        run_id=start.run_id,
        attempt_number=start.attempt_number,
        decision=start.decision,
        stage=start.stage,
        event_type=event_type,
        previous_record_sha256=start.record_sha256,
        authorization_sha256=start.authorization_sha256,
        retry_authorization_sha256=start.retry_authorization_sha256,
        source_policy_hashes=start.source_policy_hashes,
        input_hashes=start.input_hashes,
        output_hashes=outputs,
        output_disposition=output_disposition,
        idempotency_key=start.idempotency_key,
        failure_classification=failure_classification,
        error_code=error_code,
        error_class=error_class,
        error_message=error_message,
        recorded_at_utc=recorded_at_utc,
        record_sha256="",
    )
    event = _finalize(provisional)
    replay_events((*existing, event))
    return event


def resolve_output_collision(
    expected_hashes: Mapping[str, str],
    existing_hashes: Mapping[str, str] | None,
) -> OutputDisposition:
    """Permit reuse only after exact hash equality; never overwrite a collision."""

    expected = freeze_hashes(expected_hashes, label="expected_output_hashes")
    if not expected:
        raise RecoveryError("expected output hashes cannot be empty")
    if existing_hashes is None:
        return OutputDisposition.PUBLISH_NEW
    existing = freeze_hashes(existing_hashes, label="existing_output_hashes")
    if existing != expected:
        raise RecoveryError("existing output collision has different hashes")
    return OutputDisposition.VALIDATE_AND_REUSE


def _document(event: RecoveryEvent) -> dict[str, object]:
    return {**_payload(event), "record_sha256": event.record_sha256}


def _validate_no_symlink_ancestors(path: Path) -> None:
    current = path.absolute()
    for candidate in (current, *current.parents):
        if os.path.lexists(candidate) and stat.S_ISLNK(os.lstat(candidate).st_mode):
            raise RecoveryError("ledger path ancestry cannot contain a symlink")


def _read_bounded_held_path(path: Path) -> bytes:
    try:
        before = os.lstat(path)
    except FileNotFoundError as error:
        raise RecoveryError("ledger event disappeared before secure open") from error
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise RecoveryError("ledger event must be a regular non-symlink file")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise RecoveryError("ledger event must be a regular file")
        if (before.st_dev, before.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RecoveryError("ledger event pathname changed during secure open")
        if metadata.st_size <= 0 or metadata.st_size > MAX_EVENT_BYTES:
            raise RecoveryError("ledger event size is outside the accepted bound")
        chunks: list[bytes] = []
        remaining = metadata.st_size
        while remaining:
            chunk = os.read(descriptor, min(remaining, 65_536))
            if not chunk:
                raise RecoveryError("ledger event was truncated during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        if os.read(descriptor, 1):
            raise RecoveryError("ledger event grew during read")
        try:
            after = os.lstat(path)
        except FileNotFoundError as error:
            raise RecoveryError("ledger event pathname disappeared during read") from error
        if (after.st_dev, after.st_ino) != (metadata.st_dev, metadata.st_ino):
            raise RecoveryError("ledger event pathname changed during read")
    finally:
        os.close(descriptor)
    return b"".join(chunks)


def inspect_crash_state(directory: str | Path) -> CrashDisposition:
    """Classify stranded staging without deleting or publishing anything."""

    directory = Path(directory)
    if not os.path.lexists(directory):
        return CrashDisposition.CLEAN
    if directory.is_symlink() or not directory.is_dir():
        return CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    staging = sorted(directory.glob(".staging-*.json"))
    if not staging:
        return CrashDisposition.CLEAN
    if len(staging) != 1 or staging[0].is_symlink() or not staging[0].is_file():
        return CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    try:
        staging_bytes = _read_bounded_held_path(staging[0])
        document = json.loads(staging_bytes.decode("utf-8"))
        sequence = int(document["sequence"])
    except (
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        RecoveryError,
    ):
        return CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    final = directory / f"event-{sequence:08d}.json"
    if not os.path.lexists(final):
        return CrashDisposition.UNPUBLISHED_STAGING_MANUAL_REVIEW_REQUIRED
    if final.is_symlink() or not final.is_file():
        return CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    try:
        final_bytes = _read_bounded_held_path(final)
    except RecoveryError:
        return CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    if final_bytes == staging_bytes:
        return CrashDisposition.REDUNDANT_STAGING_MANUAL_CLEANUP_REQUIRED
    return CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED


def _strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise RecoveryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _read_event_document(path: Path) -> dict[str, object]:
    try:
        document = json.loads(
            _read_bounded_held_path(path).decode("utf-8"),
            object_pairs_hook=_strict_object,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RecoveryError("ledger event is not strict UTF-8 JSON") from error
    if not isinstance(document, dict):
        raise RecoveryError("ledger event JSON root must be an object")
    return document


def load_events(directory: str | Path) -> tuple[RecoveryEvent, ...]:
    directory = Path(directory)
    _validate_no_symlink_ancestors(directory)
    if not os.path.lexists(directory):
        return ()
    if directory.is_symlink() or not directory.is_dir():
        raise RecoveryError("ledger path must be a regular non-symlink directory")
    crash = inspect_crash_state(directory)
    if crash is not CrashDisposition.CLEAN:
        raise RecoveryError(f"ledger staging requires manual resolution: {crash.value}")
    paths = sorted(directory.iterdir())
    expected = [f"event-{index:08d}.json" for index in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected:
        raise RecoveryError("ledger filenames must be contiguous immutable events")
    events: list[RecoveryEvent] = []
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise RecoveryError("ledger events must be regular non-symlink files")
        try:
            document = _read_event_document(path)
            identity = PatientTransactionIdentity(
                cohort_index=document.pop("cohort_index"),
                patient_id=document.pop("patient_id"),
                slide_id=document.pop("slide_id"),
                gdc_uuid=document.pop("gdc_uuid"),
                transaction_id=document.pop("transaction_id"),
            )
            classification = document.pop("failure_classification")
            disposition = document.pop("output_disposition")
            event = RecoveryEvent(
                identity=identity,
                decision=AttemptDecision(document.pop("decision")),
                stage=PatientStage(document.pop("stage")),
                event_type=EventType(document.pop("event_type")),
                source_policy_hashes=freeze_hashes(
                    document.pop("source_policy_hashes"), label="source_policy_hashes"
                ),
                input_hashes=freeze_hashes(document.pop("input_hashes"), label="input_hashes"),
                output_hashes=freeze_hashes(
                    document.pop("output_hashes"), label="output_hashes"
                ),
                failure_classification=(
                    None if classification is None else FailureClassification(classification)
                ),
                output_disposition=(
                    None if disposition is None else OutputDisposition(disposition)
                ),
                **document,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise RecoveryError(f"invalid ledger event: {path.name}") from error
        events.append(event)
    replay_events(events)
    return tuple(events)


def append_event(directory: str | Path, event: RecoveryEvent) -> Path:
    """Atomically publish one immutable event; stale/concurrent tips fail closed."""

    directory = Path(directory)
    _validate_no_symlink_ancestors(directory.parent)
    serialized = canonical_json(_document(event))
    if len(serialized) > MAX_EVENT_BYTES:
        raise RecoveryError("ledger event exceeds the maximum serialized size")
    if os.path.lexists(directory):
        if directory.is_symlink() or not directory.is_dir():
            raise RecoveryError("ledger path must be a regular non-symlink directory")
    else:
        directory.mkdir(mode=0o750, parents=False)
        parent_fd = os.open(
            directory.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC
        )
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    current = load_events(directory)
    if event.sequence != len(current) + 1:
        raise RecoveryError("event was built from a stale ledger sequence")
    replay_events((*current, event))
    final = directory / f"event-{event.sequence:08d}.json"
    staging = directory / f".staging-{event.record_sha256}.json"
    descriptor = os.open(
        staging,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC,
        0o640,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    try:
        os.link(staging, final, follow_symlinks=False)
    except FileExistsError as error:
        raise RecoveryError("event destination exists; overwrite refused") from error
    finally:
        staging.unlink(missing_ok=True)
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    return final


__all__ = [
    "AttemptDecision",
    "CrashDisposition",
    "EventType",
    "FailureClassification",
    "GENESIS_HASH",
    "MAX_ATTEMPTS_PER_STAGE",
    "MAX_EVENT_BYTES",
    "OutputDisposition",
    "PatientTransactionIdentity",
    "RecoveryError",
    "RecoveryEvent",
    "ReplayAction",
    "ReplayPlan",
    "RETRYABLE_ERROR_CODES",
    "append_event",
    "build_outcome_event",
    "build_start_event",
    "canonical_json",
    "compute_idempotency_key",
    "freeze_hashes",
    "inspect_crash_state",
    "load_events",
    "replay_events",
    "resolve_output_collision",
    "validate_identity",
]
