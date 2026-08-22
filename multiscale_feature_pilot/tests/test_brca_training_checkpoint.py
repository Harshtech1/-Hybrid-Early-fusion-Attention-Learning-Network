from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from multiscale_feature_pilot.src.brca_training_checkpoint import (
    PAYLOAD_FILES,
    CheckpointError,
    EpochState,
    TrainingRunIdentity,
    discover_checkpoints,
    plan_recovery,
    publish_checkpoint,
    validate_checkpoint,
)


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _identity() -> TrainingRunIdentity:
    return TrainingRunIdentity(
        training_run_id="00000000-0000-4000-8000-000000000001",
        source_commit="1" * 40,
        authorization_sha256=_hash("authorization"),
        split_manifest_sha256=_hash("split"),
        cutpoints_sha256=_hash("cutpoints"),
        feature_registry_sha256=_hash("registry"),
        training_policy_sha256=_hash("policy"),
        official_healnet_commit="2" * 40,
    )


def _state(epoch: int, *, stopped: bool = False) -> EpochState:
    return EpochState(
        epoch=epoch,
        optimizer_steps=epoch * 42,
        best_epoch=epoch,
        best_validation_nll=1.0 / epoch,
        epochs_without_improvement=0,
        early_stop_reached=stopped,
    )


def _payloads(epoch: int) -> dict[str, bytes]:
    return {name: f"{name}:epoch:{epoch}\n".encode() for name in PAYLOAD_FILES}


def test_append_validate_and_resume_two_epochs(tmp_path: Path) -> None:
    identity = _identity()
    assert plan_recovery(tmp_path, identity).action == "START_NEW"
    first = publish_checkpoint(tmp_path, identity=identity, state=_state(1), payloads=_payloads(1))
    assert validate_checkpoint(first.directory, identity) == first
    assert plan_recovery(tmp_path, identity).action == "RESUME_LATEST"
    second = publish_checkpoint(tmp_path, identity=identity, state=_state(2), payloads=_payloads(2))
    assert [item.state.epoch for item in discover_checkpoints(tmp_path, identity)] == [1, 2]
    assert plan_recovery(tmp_path, identity).latest == second


def test_early_stop_checkpoint_closes_epoch_training(tmp_path: Path) -> None:
    identity = _identity()
    publish_checkpoint(tmp_path, identity=identity, state=_state(1, stopped=True), payloads=_payloads(1))
    assert plan_recovery(tmp_path, identity).action == "TRAINING_EPOCHS_COMPLETE"


def test_overwrite_gap_and_identity_drift_fail_closed(tmp_path: Path) -> None:
    identity = _identity()
    publish_checkpoint(tmp_path, identity=identity, state=_state(1), payloads=_payloads(1))
    with pytest.raises(CheckpointError, match="next contiguous"):
        publish_checkpoint(tmp_path, identity=identity, state=_state(1), payloads=_payloads(1))
    with pytest.raises(CheckpointError, match="next contiguous"):
        publish_checkpoint(tmp_path, identity=identity, state=_state(3), payloads=_payloads(3))
    with pytest.raises(CheckpointError, match="identity drift"):
        discover_checkpoints(tmp_path, replace(identity, authorization_sha256=_hash("other")))


def test_payload_corruption_and_symlink_fail_closed(tmp_path: Path) -> None:
    identity = _identity()
    checkpoint = publish_checkpoint(tmp_path, identity=identity, state=_state(1), payloads=_payloads(1))
    (checkpoint.directory / "model_state.pt").write_bytes(b"changed")
    with pytest.raises(CheckpointError, match="hash/size"):
        validate_checkpoint(checkpoint.directory, identity)
    other = tmp_path / "other"
    other.mkdir()
    linked = other / "checkpoint-epoch-0001"
    linked.symlink_to(checkpoint.directory, target_is_directory=True)
    with pytest.raises(CheckpointError, match="symlink"):
        validate_checkpoint(linked, identity)


def test_stranded_staging_and_unexpected_entries_block_recovery(tmp_path: Path) -> None:
    identity = _identity()
    (tmp_path / f".checkpoint-epoch-0001.staging-{identity.training_run_id}").mkdir()
    with pytest.raises(CheckpointError, match="stranded"):
        discover_checkpoints(tmp_path, identity)
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "notes.txt").write_text("unexpected")
    with pytest.raises(CheckpointError, match="unexpected"):
        discover_checkpoints(clean, identity)


def test_payload_order_and_epoch_history_are_mandatory(tmp_path: Path) -> None:
    identity = _identity()
    payloads = _payloads(1)
    reordered = {name: payloads[name] for name in reversed(PAYLOAD_FILES)}
    with pytest.raises(CheckpointError, match="labels/order"):
        publish_checkpoint(tmp_path, identity=identity, state=_state(1), payloads=reordered)
    missing = dict(payloads)
    missing.pop("epoch_history.json")
    with pytest.raises(CheckpointError, match="labels/order"):
        publish_checkpoint(tmp_path, identity=identity, state=_state(1), payloads=missing)
