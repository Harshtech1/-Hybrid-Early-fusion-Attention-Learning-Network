from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src.brca_first_eight_phase_barrier import (
    PhaseBarrierError,
    validate_active_work,
    validate_policy,
)
from multiscale_feature_pilot.src.brca_p0001_recovery_prefix_bootstrap import (
    AUTHORIZATION_STATEMENT_SHA256,
    BOOTSTRAP_RECORDED_AT_UTC,
    EXPECTED_STAGES,
    PrefixBootstrapError,
    derive_prefix,
    publish_prefix,
)
from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
    EventType,
    ReplayAction,
    load_events,
    replay_events,
)


ROOT = Path(__file__).resolve().parents[2]
AUTH = ROOT / "multiscale_feature_pilot/config/brca_p0001_recovery_prefix_bootstrap_authorization.yaml"
POLICY = ROOT / "multiscale_feature_pilot/config/brca_first_eight_phase_barrier_policy.yaml"
RUNNER = ROOT / "scripts/run_brca_p0001_recovery_prefix_bootstrap.py"


def test_direct_authorization_is_exact_and_narrow() -> None:
    document = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    statement = document["authorization"]["exact_statement"]
    assert hashlib.sha256(statement.encode()).hexdigest() == AUTHORIZATION_STATEMENT_SHA256
    assert document["authorization"]["exact_statement_sha256"] == AUTHORIZATION_STATEMENT_SHA256
    assert document["scope"]["exact_events"] == 10
    assert document["scope"]["bootstrap_recorded_at_utc"] == BOOTSTRAP_RECORDED_AT_UTC
    assert document["scope"]["exact_stage_successes"] == [stage.value for stage in EXPECTED_STAGES]
    assert document["scope"]["terminal_tip"] == {
        "last_durable_stage": "COORDINATES_VERIFIED",
        "replay_action": "ADVANCE_STAGE",
        "target_stage": "GPU_AUTHORIZED",
    }
    assert all(document["prohibited"].values())
    assert document["publication"]["delete_or_overwrite_any_path"] is False


def test_prefix_is_exact_deterministic_and_ready_for_gpu_authorization() -> None:
    first = derive_prefix(ROOT)
    second = derive_prefix(ROOT)
    assert first == second
    assert len(first) == 10
    assert [event.stage for event in first[::2]] == list(EXPECTED_STAGES)
    assert all(event.event_type is EventType.STAGE_STARTED for event in first[::2])
    assert all(event.event_type is EventType.STAGE_SUCCEEDED for event in first[1::2])
    assert all(event.attempt_number == 1 for event in first)
    assert {event.recorded_at_utc for event in first} == {BOOTSTRAP_RECORDED_AT_UTC}
    replay = replay_events(first)
    assert replay.action is ReplayAction.ADVANCE_STAGE
    assert replay.last_durable_stage is PatientStage.COORDINATES_VERIFIED
    assert replay.target_stage is PatientStage.GPU_AUTHORIZED


def test_atomic_complete_prefix_publication_and_no_overwrite(tmp_path: Path) -> None:
    destination = tmp_path / "ledger"
    events = publish_prefix(ROOT, destination)
    assert load_events(destination) == events
    assert sorted(path.name for path in destination.iterdir()) == [f"event-{index:08d}.json" for index in range(1, 11)]
    before = {path.name: path.read_bytes() for path in destination.iterdir()}
    with pytest.raises(PrefixBootstrapError, match="must be absent"):
        publish_prefix(ROOT, destination)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == before


def test_publication_rejects_symlink_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PrefixBootstrapError, match="symlink"):
        publish_prefix(ROOT, linked_parent / "ledger")
    assert not (real_parent / "ledger").exists()


def test_phase_policy_binds_exact_eight_and_locks_gpu_pixels() -> None:
    policy = validate_policy(ROOT, POLICY)
    assert policy["scheduling"]["download_concurrency"] == 1
    assert policy["scheduling"]["cpu_patient_preparation_workers"] == 2
    assert policy["scheduling"]["gpu_concurrency"] == 0
    assert policy["cohort_binding"]["p0002_p0008_raw_wsi_bytes"] == 6_527_281_524
    assert all(policy["operation_locks"].values())


def test_phase_scheduler_accepts_only_two_cpu_workers_and_one_download() -> None:
    validate_active_work((
        {"patient": "P0002", "operation": "DOWNLOAD"},
        {"patient": "P0003", "operation": "POLICY_DESIGN"},
    ))
    with pytest.raises(PhaseBarrierError, match="two CPU"):
        validate_active_work((
            {"patient": "P0002", "operation": "DOWNLOAD"},
            {"patient": "P0003", "operation": "HEADER_ONLY"},
            {"patient": "P0004", "operation": "POLICY_DESIGN"},
        ))
    with pytest.raises(PhaseBarrierError, match="download concurrency"):
        validate_active_work((
            {"patient": "P0002", "operation": "DOWNLOAD"},
            {"patient": "P0003", "operation": "DOWNLOAD"},
        ))
    with pytest.raises(PhaseBarrierError, match="outside CPU"):
        validate_active_work(({"patient": "P0002", "operation": "READ_REGION"},))


def test_runner_has_no_forbidden_import_or_process_surface() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not ({"openslide", "torch", "torchvision", "requests"} & imports)
    text = RUNNER.read_text(encoding="utf-8")
    assert "read_region(" not in text
    assert "shutil.rmtree" not in text
    assert "os.remove" not in text
    assert "unlink(" not in text
