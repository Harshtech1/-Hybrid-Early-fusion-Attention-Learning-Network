from dataclasses import replace
from pathlib import Path
from uuid import UUID

import pytest

from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
    CrashDisposition,
    EventType,
    FailureClassification,
    OutputDisposition,
    PatientTransactionIdentity,
    RecoveryError,
    ReplayAction,
    append_event,
    build_outcome_event,
    build_start_event,
    inspect_crash_state,
    load_events,
    replay_events,
    resolve_output_collision,
)


IDENTITY = PatientTransactionIdentity(
    cohort_index=1,
    patient_id="TCGA-AA-0001",
    slide_id="TCGA-AA-0001-01Z-00-DX1.TEST.svs",
    gdc_uuid="00000000-0000-0000-0000-000000000001",
    transaction_id="00000000-0000-0000-0000-000000000002",
)
SOURCES = {
    "compact_artifact_policy": "a" * 64,
    "cohort_order_manifest": "d" * 64,
    "executor_v2_policy": "b" * 64,
    "singleton_streaming_policy": "c" * 64,
}


def _run(number: int) -> str:
    return str(UUID(int=number + 100))


def _hash(number: int) -> str:
    return f"{number % 16:x}" * 64


def _start(events, number: int, *, identity=IDENTITY, run_id=None):
    replay = replay_events(events)
    inputs = {"stage_evidence": _hash(8)}
    if events:
        inputs.update(
            previous_record=replay.previous_record_sha256,
            previous_success_record=replay.previous_success_record_sha256,
        )
        if replay.action is ReplayAction.RETRY_FAILED_STAGE:
            inputs["prior_failure"] = replay.failed_event_sha256
    return build_start_event(
        events,
        identity=identity,
        run_id=_run(number) if run_id is None else run_id,
        authorization_sha256=_hash(9),
        retry_authorization_sha256=(
            _hash(10) if replay.action is ReplayAction.RETRY_FAILED_STAGE else None
        ),
        source_policy_hashes=SOURCES,
        input_hashes=inputs,
        recorded_at_utc=f"2026-08-20T00:{number * 2:02d}:00Z",
    )


def _success(events, number: int):
    return build_outcome_event(
        events,
        event_type=EventType.STAGE_SUCCEEDED,
        output_hashes={"stage_output": _hash(number)},
        existing_output_hashes=None,
        outputs_validated=True,
        failure_classification=None,
        error_code=None,
        error_class=None,
        error_message=None,
        recorded_at_utc=f"2026-08-20T00:{number * 2 + 1:02d}:00Z",
    )


def _failure(events, number: int, classification=FailureClassification.RETRYABLE):
    return build_outcome_event(
        events,
        event_type=EventType.STAGE_FAILED,
        output_hashes={},
        existing_output_hashes=None,
        outputs_validated=False,
        failure_classification=classification,
        error_code=(
            "TRANSIENT_IO"
            if classification is FailureClassification.RETRYABLE
            else "IDENTITY_MISMATCH"
        ),
        error_class="SyntheticFailure",
        error_message="synthetic fixture only",
        recorded_at_utc=f"2026-08-20T00:{number * 2 + 1:02d}:00Z",
    )


def test_full_two_event_per_stage_chain_reaches_terminal() -> None:
    events = []
    for number, stage in enumerate(PatientStage, start=1):
        started = _start(events, number)
        assert started.stage is stage
        events.append(started)
        events.append(_success(events, number))
    replay = replay_events(events)
    assert replay.action is ReplayAction.STOP_TERMINAL
    assert replay.last_durable_stage is PatientStage.TERMINAL_RECORDED
    assert len(events) == 16


def test_stage_must_start_before_outcome_and_crash_resumes_same_attempt() -> None:
    started = _start([], 1)
    replay = replay_events((started,))
    assert replay.action is ReplayAction.COMPLETE_STARTED_STAGE
    assert replay.pending_run_id == started.run_id
    with pytest.raises(RecoveryError, match="cannot start"):
        _start((started,), 2)


def test_failure_is_immutable_and_retry_is_monotonic_and_bound() -> None:
    started = _start([], 1)
    success = _success((started,), 1)
    raw_start = _start((started, success), 2)
    failed = _failure((started, success, raw_start), 2)
    before_retry = (started, success, raw_start, failed)
    replay = replay_events(before_retry)
    assert replay.action is ReplayAction.RETRY_FAILED_STAGE
    retried = _start(before_retry, 3)
    assert retried.stage is PatientStage.ACQUISITION_AUTHORIZED
    assert retried.attempt_number == 2
    assert dict(retried.input_hashes)["prior_failure"] == failed.record_sha256
    assert retried.idempotency_key == raw_start.idempotency_key
    retry_success = _success((*before_retry, retried), 3)
    assert replay_events((*before_retry, retried, retry_success)).action is ReplayAction.ADVANCE_STAGE
    assert failed in (*before_retry, retried, retry_success)


def test_non_retryable_failure_blocks_replay() -> None:
    start = _start([], 1)
    failed = _failure((start,), 1, FailureClassification.NON_RETRYABLE)
    events = (start, failed)
    assert replay_events(events).action is ReplayAction.BLOCKED_NON_RETRYABLE
    with pytest.raises(RecoveryError, match="cannot start"):
        _start(events, 2)


def test_identity_uuid_and_cohort_index_drift_fail_closed() -> None:
    start = _start([], 1)
    success = _success((start,), 1)
    drift = replace(IDENTITY, cohort_index=2)
    with pytest.raises(RecoveryError, match="identity drift"):
        _start((start, success), 2, identity=drift)
    with pytest.raises(RecoveryError, match="canonical UUID"):
        _start([], 1, identity=replace(IDENTITY, gdc_uuid="not-a-uuid"))


def test_run_id_cannot_be_reused_between_attempts() -> None:
    start = _start([], 1)
    success = _success((start,), 1)
    with pytest.raises(RecoveryError, match="run_id cannot be reused"):
        _start((start, success), 2, run_id=start.run_id)


def test_tampered_hash_chain_input_binding_and_record_hash_are_rejected() -> None:
    start = _start([], 1)
    success = _success((start,), 1)
    next_start = _start((start, success), 2)
    with pytest.raises(RecoveryError, match="hash chain"):
        replay_events((start, success, replace(next_start, previous_record_sha256="f" * 64)))
    altered_inputs = tuple(
        (key, "e" * 64 if key == "previous_record" else value)
        for key, value in next_start.input_hashes
    )
    with pytest.raises(RecoveryError, match="idempotency key|previous_record|record SHA256"):
        replay_events((start, success, replace(next_start, input_hashes=altered_inputs)))


def test_atomic_append_round_trip_and_existing_event_never_overwritten(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    start = _start([], 1)
    path = append_event(directory, start)
    before = path.read_bytes()
    with pytest.raises(RecoveryError, match="stale|sequence|hash chain"):
        append_event(directory, start)
    assert path.read_bytes() == before
    assert load_events(directory) == (start,)


def test_stale_concurrent_writer_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "ledger"
    start = _start([], 1)
    append_event(directory, start)
    winner = _success((start,), 1)
    loser = build_outcome_event(
        (start,),
        event_type=EventType.STAGE_SUCCEEDED,
        output_hashes={"different": "d" * 64},
        existing_output_hashes=None,
        outputs_validated=True,
        failure_classification=None,
        error_code=None,
        error_class=None,
        error_message=None,
        recorded_at_utc="2026-08-20T00:03:01Z",
    )
    append_event(directory, winner)
    with pytest.raises(RecoveryError, match="stale|hash chain|outcome"):
        append_event(directory, loser)


@pytest.mark.parametrize("unexpected", ["event-00000001.json.partial", ".staging-deadbeef.json"])
def test_truncated_or_staging_crash_file_stops_replay(tmp_path: Path, unexpected: str) -> None:
    directory = tmp_path / "ledger"
    directory.mkdir()
    (directory / unexpected).write_text("synthetic incomplete file", encoding="utf-8")
    with pytest.raises(RecoveryError, match="filenames|manual resolution"):
        load_events(directory)


def test_symlink_ledger_and_symlink_record_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(target, target_is_directory=True)
    with pytest.raises(RecoveryError, match="symlink"):
        load_events(linked)
    (target / "real.json").write_text("{}", encoding="utf-8")
    (target / "event-00000001.json").symlink_to(target / "real.json")
    with pytest.raises(RecoveryError, match="symlink"):
        load_events(target / "event-00000001.json")


def test_output_collision_is_validate_and_reuse_only() -> None:
    expected = {"compact_manifest": "a" * 64}
    assert resolve_output_collision(expected, None) is OutputDisposition.PUBLISH_NEW
    assert (
        resolve_output_collision(expected, dict(expected))
        is OutputDisposition.VALIDATE_AND_REUSE
    )
    with pytest.raises(RecoveryError, match="different hashes"):
        resolve_output_collision(expected, {"compact_manifest": "b" * 64})


def test_retry_limit_blocks_after_three_failed_attempts() -> None:
    events = []
    start = _start(events, 1)
    events.extend((start, _success((start,), 1)))
    for number in (2, 3, 4):
        started = _start(events, number)
        events.append(started)
        events.append(_failure(events, number))
    replay = replay_events(events)
    assert replay.action is ReplayAction.BLOCKED_RETRY_LIMIT
    with pytest.raises(RecoveryError, match="cannot start"):
        _start(events, 5)


def test_timestamp_regression_and_retryability_policy_are_rejected() -> None:
    start = _start([], 1)
    with pytest.raises(RecoveryError, match="nondecreasing"):
        build_outcome_event(
            (start,),
            event_type=EventType.STAGE_SUCCEEDED,
            output_hashes={"stage_output": "1" * 64},
            existing_output_hashes=None,
            outputs_validated=True,
            failure_classification=None,
            error_code=None,
            error_class=None,
            error_message=None,
            recorded_at_utc="2026-08-20T00:00:00Z",
        )
    with pytest.raises(RecoveryError, match="error-code policy"):
        build_outcome_event(
            (start,),
            event_type=EventType.STAGE_FAILED,
            output_hashes={},
            existing_output_hashes=None,
            outputs_validated=False,
            failure_classification=FailureClassification.RETRYABLE,
            error_code="IDENTITY_MISMATCH",
            error_class="SyntheticFailure",
            error_message="synthetic fixture only",
            recorded_at_utc="2026-08-20T00:03:00Z",
        )


def test_crash_before_and_after_atomic_link_are_explicit_manual_blocks(tmp_path: Path) -> None:
    source = tmp_path / "source"
    start = _start([], 1)
    final = append_event(source, start)
    document = final.read_bytes()

    before_link = tmp_path / "before-link"
    before_link.mkdir()
    (before_link / f".staging-{start.record_sha256}.json").write_bytes(document)
    assert (
        inspect_crash_state(before_link)
        is CrashDisposition.UNPUBLISHED_STAGING_MANUAL_REVIEW_REQUIRED
    )
    with pytest.raises(RecoveryError, match="manual resolution"):
        load_events(before_link)

    after_link = tmp_path / "after-link"
    after_link.mkdir()
    (after_link / "event-00000001.json").write_bytes(document)
    (after_link / f".staging-{start.record_sha256}.json").write_bytes(document)
    assert (
        inspect_crash_state(after_link)
        is CrashDisposition.REDUNDANT_STAGING_MANUAL_CLEANUP_REQUIRED
    )
    with pytest.raises(RecoveryError, match="manual resolution"):
        load_events(after_link)


def test_duplicate_json_keys_and_oversized_event_fail_closed(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate"
    duplicate.mkdir()
    (duplicate / "event-00000001.json").write_text(
        '{"sequence":1,"sequence":1}\n', encoding="utf-8"
    )
    with pytest.raises(RecoveryError, match="duplicate JSON key"):
        load_events(duplicate)

    oversized = tmp_path / "oversized"
    oversized.mkdir()
    (oversized / "event-00000001.json").write_bytes(b"x" * 1_000_001)
    with pytest.raises(RecoveryError, match="size"):
        load_events(oversized)


def test_crash_inspection_rejects_symlink_and_oversized_staging(tmp_path: Path) -> None:
    broken_directory_link = tmp_path / "broken-ledger-link"
    broken_directory_link.symlink_to(tmp_path / "missing-ledger", target_is_directory=True)
    assert (
        inspect_crash_state(broken_directory_link)
        is CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    )

    target = tmp_path / "target.json"
    target.write_text('{"sequence":1}\n', encoding="utf-8")
    symlink_dir = tmp_path / "symlink-staging"
    symlink_dir.mkdir()
    (symlink_dir / ".staging-test.json").symlink_to(target)
    assert (
        inspect_crash_state(symlink_dir)
        is CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    )

    oversized_dir = tmp_path / "oversized-staging"
    oversized_dir.mkdir()
    (oversized_dir / ".staging-test.json").write_bytes(b"x" * 1_000_001)
    assert (
        inspect_crash_state(oversized_dir)
        is CrashDisposition.AMBIGUOUS_LEDGER_MANUAL_REVIEW_REQUIRED
    )


def test_oversized_canonical_event_is_rejected_before_publication(tmp_path: Path) -> None:
    huge_sources = {f"policy_{index}": "a" * 64 for index in range(15_000)}
    with pytest.raises(RecoveryError, match="maximum serialized size"):
        build_start_event(
            (),
            identity=IDENTITY,
            run_id=_run(1),
            authorization_sha256="9" * 64,
            retry_authorization_sha256=None,
            source_policy_hashes=huge_sources,
            input_hashes={"stage_evidence": "8" * 64},
            recorded_at_utc="2026-08-20T00:02:00Z",
        )
    assert not (tmp_path / "ledger").exists()


def test_genesis_cannot_forge_previous_chain_or_failure_bindings() -> None:
    for reserved in ("previous_record", "previous_success_record", "prior_failure"):
        with pytest.raises(RecoveryError, match="genesis"):
            build_start_event(
                (),
                identity=IDENTITY,
                run_id=_run(1),
                authorization_sha256="9" * 64,
                retry_authorization_sha256=None,
                source_policy_hashes=SOURCES,
                input_hashes={"stage_evidence": "8" * 64, reserved: "0" * 64},
                recorded_at_utc="2026-08-20T00:02:00Z",
            )
