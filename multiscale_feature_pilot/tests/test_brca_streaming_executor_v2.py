from dataclasses import replace
from uuid import UUID

import pytest

from multiscale_feature_pilot.src import brca_streaming_executor_v2 as executor_v2
from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
from multiscale_feature_pilot.src.brca_streaming_executor_v2 import (
    ExecutorContractError,
    ExecutorPolicyV2,
    FORBIDDEN_EXECUTION_SURFACES,
    STAGE_CONTRACTS,
    outcome_event_from_plan,
    plan_next_stage,
    resume_pending_plan,
    start_event_from_plan,
    validate_executor_policy,
    validate_next_patient_gate,
    validate_superseding_transaction_gate,
)
from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
    FailureClassification,
    OutputDisposition,
    PatientTransactionIdentity,
    ReplayAction,
    replay_events,
)


IDENTITY = PatientTransactionIdentity(
    cohort_index=1,
    patient_id="TCGA-AA-0001",
    slide_id="TCGA-AA-0001-01Z-00-DX1.TEST.svs",
    gdc_uuid="00000000-0000-0000-0000-000000000001",
    transaction_id="00000000-0000-0000-0000-000000000002",
)
SOURCES = {
    "cohort_order_manifest": "d" * 64,
    "compact_artifact_policy": "a" * 64,
    "executor_v2_policy": "b" * 64,
    "singleton_streaming_policy": "c" * 64,
}


def _run(number: int) -> str:
    return str(UUID(int=number + 300))


def _inputs(stage: PatientStage) -> dict[str, str]:
    return {
        label: f"{(index + 5) % 16:x}" * 64
        for index, label in enumerate(STAGE_CONTRACTS[stage].required_input_labels)
    }


def _plan(events, number: int, *, identity=IDENTITY):
    replay = replay_events(events)
    return plan_next_stage(
        events,
        identity=identity,
        run_id=_run(number),
        authorization_sha256="9" * 64,
        retry_authorization_sha256=(
            "e" * 64 if replay.action is ReplayAction.RETRY_FAILED_STAGE else None
        ),
        source_policy_hashes=SOURCES,
        stage_input_hashes=_inputs(replay.target_stage),
    )


def _start(events, plan):
    return start_event_from_plan(
        events,
        plan,
        recorded_at_utc=f"2026-08-20T01:{len(events):02d}:00Z",
    )


def _success(events, plan, *, existing=None, validated=True):
    outputs = {plan.required_success_output_labels[0]: "f" * 64}
    return outcome_event_from_plan(
        events,
        plan,
        succeeded=True,
        output_hashes=outputs,
        existing_output_hashes=existing,
        outputs_validated=validated,
        failure_classification=None,
        error_code=None,
        error_class=None,
        error_message=None,
        recorded_at_utc=f"2026-08-20T01:{len(events):02d}:00Z",
    )


def _failure(events, plan, *, retryable=True):
    return outcome_event_from_plan(
        events,
        plan,
        succeeded=False,
        output_hashes={},
        existing_output_hashes=None,
        outputs_validated=False,
        failure_classification=(
            FailureClassification.RETRYABLE
            if retryable
            else FailureClassification.NON_RETRYABLE
        ),
        error_code="TRANSIENT_IO" if retryable else "IDENTITY_MISMATCH",
        error_class="SyntheticFailure",
        error_message="synthetic fixture only",
        recorded_at_utc=f"2026-08-20T01:{len(events):02d}:00Z",
    )


def _advance_success(events: list, number: int) -> None:
    plan = _plan(events, number)
    events.append(_start(events, plan))
    events.append(_success(events, resume_pending_plan(events)))


def test_policy_is_exactly_894_serial_one_patient_and_has_no_operation_surface() -> None:
    validate_executor_policy(ExecutorPolicyV2())
    assert FORBIDDEN_EXECUTION_SURFACES
    assert not any(FORBIDDEN_EXECUTION_SURFACES.values())
    for override in (
        {"cohort_patient_count": 895},
        {"patient_transaction_size": 2},
        {"patient_concurrency": 2},
        {"download_concurrency": 2},
        {"gpu_concurrency": 2},
        {"next_patient_requires_terminal_record": False},
        {"atomic_no_overwrite_ledger": False},
    ):
        with pytest.raises(ExecutorContractError):
            validate_executor_policy(replace(ExecutorPolicyV2(), **override))


def test_plan_is_deterministic_and_binds_exact_identity_uuid_policies_and_inputs() -> None:
    first = _plan([], 1)
    second = _plan([], 1)
    assert first == second
    assert first.stage is PatientStage.PLANNED
    assert first.attempt_number == 1
    assert dict(first.source_policy_hashes) == SOURCES
    assert first.identity.gdc_uuid == IDENTITY.gdc_uuid


def test_source_and_stage_input_labels_fail_closed() -> None:
    with pytest.raises(ExecutorContractError, match="source policy labels"):
        plan_next_stage(
            [],
            identity=IDENTITY,
            run_id=_run(1),
            authorization_sha256="9" * 64,
            retry_authorization_sha256=None,
            source_policy_hashes={"compact_artifact_policy": "a" * 64},
            stage_input_hashes=_inputs(PatientStage.PLANNED),
        )
    with pytest.raises(ExecutorContractError, match="input labels"):
        plan_next_stage(
            [],
            identity=IDENTITY,
            run_id=_run(1),
            authorization_sha256="9" * 64,
            retry_authorization_sha256=None,
            source_policy_hashes=SOURCES,
            stage_input_hashes={"cohort_alignment": "a" * 64},
        )


def test_full_synthetic_contract_reaches_terminal_without_running_operations() -> None:
    events = []
    for number, stage in enumerate(PatientStage, start=1):
        assert replay_events(events).target_stage is stage
        _advance_success(events, number)
    assert replay_events(events).action is ReplayAction.STOP_TERMINAL
    assert len(events) == 16


def test_pending_start_is_reconstructed_after_synthetic_crash() -> None:
    plan = _plan([], 1)
    start = _start([], plan)
    resumed = resume_pending_plan((start,))
    assert resumed.run_id == plan.run_id
    assert resumed.idempotency_key == plan.idempotency_key
    assert resumed.replay_action is ReplayAction.COMPLETE_STARTED_STAGE


def test_success_requires_validation_and_collision_hash_equality() -> None:
    plan = _plan([], 1)
    start = _start([], plan)
    pending = resume_pending_plan((start,))
    with pytest.raises(ExecutorContractError, match="explicit artifact"):
        _success((start,), pending, validated=False)
    with pytest.raises(ExecutorContractError, match="different hashes"):
        _success((start,), pending, existing={"patient_plan": "0" * 64})
    outcome = _success((start,), pending, existing={"patient_plan": "f" * 64})
    assert outcome.output_disposition is OutputDisposition.VALIDATE_AND_REUSE


def test_retry_preserves_stage_contract_and_requires_separate_retry_authority() -> None:
    events = []
    _advance_success(events, 1)
    plan = _plan(events, 2)
    events.append(_start(events, plan))
    events.append(_failure(events, resume_pending_plan(events)))
    retry = _plan(events, 3)
    assert retry.stage is PatientStage.ACQUISITION_AUTHORIZED
    assert retry.attempt_number == 2
    assert retry.retry_authorization_sha256 == "e" * 64
    assert retry.idempotency_key == plan.idempotency_key
    with pytest.raises(ExecutorContractError, match="retry authorization"):
        plan_next_stage(
            events,
            identity=IDENTITY,
            run_id=_run(4),
            authorization_sha256="9" * 64,
            retry_authorization_sha256=None,
            source_policy_hashes=SOURCES,
            stage_input_hashes=_inputs(PatientStage.ACQUISITION_AUTHORIZED),
        )
    with pytest.raises(ExecutorContractError, match="preserve"):
        plan_next_stage(
            events,
            identity=IDENTITY,
            run_id=_run(4),
            authorization_sha256="8" * 64,
            retry_authorization_sha256="e" * 64,
            source_policy_hashes=SOURCES,
            stage_input_hashes=_inputs(PatientStage.ACQUISITION_AUTHORIZED),
        )
    changed_sources = dict(SOURCES)
    changed_sources["compact_artifact_policy"] = "0" * 64
    with pytest.raises(ExecutorContractError, match="preserve"):
        plan_next_stage(
            events,
            identity=IDENTITY,
            run_id=_run(5),
            authorization_sha256="9" * 64,
            retry_authorization_sha256="e" * 64,
            source_policy_hashes=changed_sources,
            stage_input_hashes=_inputs(PatientStage.ACQUISITION_AUTHORIZED),
        )
    changed_inputs = _inputs(PatientStage.ACQUISITION_AUTHORIZED)
    changed_inputs["one_row_manifest"] = "0" * 64
    with pytest.raises(ExecutorContractError, match="preserve"):
        plan_next_stage(
            events,
            identity=IDENTITY,
            run_id=_run(6),
            authorization_sha256="9" * 64,
            retry_authorization_sha256="e" * 64,
            source_policy_hashes=SOURCES,
            stage_input_hashes=changed_inputs,
        )


def test_plan_hash_tamper_and_identity_drift_are_rejected() -> None:
    plan = _plan([], 1)
    with pytest.raises(ExecutorContractError, match="plan SHA256"):
        start_event_from_plan([], replace(plan, plan_sha256="0" * 64), recorded_at_utc="2026-08-20T01:00:00Z")
    start = _start([], plan)
    success = _success((start,), resume_pending_plan((start,)))
    with pytest.raises(ExecutorContractError, match="identity"):
        _plan((start, success), 2, identity=replace(IDENTITY, cohort_index=2))


def test_rehashed_plan_cannot_forge_stage_output_contract() -> None:
    plan = _plan([], 1)
    forged = executor_v2._finalize_plan(
        replace(
            plan,
            required_success_output_labels=("forged_output",),
            plan_sha256="",
        )
    )
    with pytest.raises(ExecutorContractError, match="output labels"):
        start_event_from_plan(
            [], forged, recorded_at_utc="2026-08-20T01:00:00Z"
        )


def test_rehashed_plan_cannot_supply_partial_supersede_binding() -> None:
    plan = _plan([], 1)
    forged_inputs = dict(plan.input_hashes)
    forged_inputs["supersede_authorization"] = "7" * 64
    frozen_inputs = tuple(sorted(forged_inputs.items()))
    forged_idempotency = executor_v2.compute_idempotency_key(
        identity=plan.identity,
        stage=plan.stage,
        authorization_sha256=plan.authorization_sha256,
        source_policy_hashes=plan.source_policy_hashes,
        input_hashes=frozen_inputs,
    )
    forged = executor_v2._finalize_plan(
        replace(
            plan,
            input_hashes=frozen_inputs,
            idempotency_key=forged_idempotency,
            plan_sha256="",
        )
    )
    with pytest.raises(ExecutorContractError, match="supersede input bindings"):
        start_event_from_plan(
            [], forged, recorded_at_utc="2026-08-20T01:00:00Z"
        )


def test_next_patient_requires_terminal_success_and_exact_manifest_order() -> None:
    second = PatientTransactionIdentity(
        cohort_index=2,
        patient_id="TCGA-BB-0002",
        slide_id="TCGA-BB-0002-01Z-00-DX1.TEST.svs",
        gdc_uuid="00000000-0000-0000-0000-000000000011",
        transaction_id="00000000-0000-0000-0000-000000000012",
    )
    validate_next_patient_gate([], next_identity=IDENTITY)
    plan = _plan([], 1)
    start = _start([], plan)
    with pytest.raises(ExecutorContractError, match="terminal"):
        validate_next_patient_gate((start,), next_identity=second)
    events = []
    for number in range(1, 9):
        _advance_success(events, number)
    validate_next_patient_gate(events, next_identity=second)
    with pytest.raises(ExecutorContractError, match="next cohort manifest index"):
        validate_next_patient_gate(events, next_identity=replace(second, cohort_index=3))


def test_nonretryable_failure_has_explicit_separately_authorized_supersede_route() -> None:
    plan = _plan([], 1)
    start = _start([], plan)
    failure = _failure((start,), resume_pending_plan((start,)), retryable=False)
    events = (start, failure)
    superseding = replace(
        IDENTITY, transaction_id="00000000-0000-0000-0000-000000000099"
    )
    bindings = validate_superseding_transaction_gate(
        events,
        superseding_identity=superseding,
        failure_sha256=failure.record_sha256,
        supersede_authorization_sha256="7" * 64,
    )
    assert bindings["superseded_failure"] == failure.record_sha256
    with pytest.raises(ExecutorContractError, match="new transaction"):
        validate_superseding_transaction_gate(
            events,
            superseding_identity=IDENTITY,
            failure_sha256=failure.record_sha256,
            supersede_authorization_sha256="7" * 64,
        )
