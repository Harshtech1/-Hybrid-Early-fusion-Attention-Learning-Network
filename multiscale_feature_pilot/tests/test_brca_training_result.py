from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import pytest

from multiscale_feature_pilot.src.brca_training_checkpoint import TrainingRunIdentity
from multiscale_feature_pilot.src.brca_training_result import (
    TrainingResultError, begin_locked_test, publish_training_result,
    validate_start_marker, validate_training_result,
)


def _identity() -> TrainingRunIdentity:
    return TrainingRunIdentity(
        training_run_id=str(uuid4()), source_commit="1" * 40,
        authorization_sha256="2" * 64, split_manifest_sha256="3" * 64,
        cutpoints_sha256="4" * 64, feature_registry_sha256="5" * 64,
        training_policy_sha256="6" * 64, official_healnet_commit="7" * 40,
    )


def _summary():
    return {
        "protocol_id": "BRCA_HEALNET_IMAGENET1K_V2_SINGLE_SPLIT_V1",
        "best_epoch": 7, "best_validation_nll": 0.8,
        "locked_test": {
            "patients": 118, "mean_nll": 0.9, "harrell_c_index": 0.61,
            "comparable_pairs": 1234,
            "patient_bootstrap_95_percent_ci": [0.52, 0.70],
            "bootstrap_valid_replicates": 2000,
            "bootstrap_requested_replicates": 2000,
        },
        "locked_test_evaluations_this_run": 1,
        "training_complete": True,
    }


def test_marker_and_atomic_result_round_trip(tmp_path: Path) -> None:
    identity = _identity()
    marker = begin_locked_test(tmp_path, identity)
    assert marker == validate_start_marker(tmp_path, identity)
    document = publish_training_result(tmp_path, identity, _summary())
    assert document == validate_training_result(tmp_path, identity)


def test_second_marker_or_result_fails_without_overwrite(tmp_path: Path) -> None:
    identity = _identity()
    begin_locked_test(tmp_path, identity)
    with pytest.raises(FileExistsError):
        begin_locked_test(tmp_path, identity)
    publish_training_result(tmp_path, identity, _summary())
    with pytest.raises(TrainingResultError, match="already exists"):
        publish_training_result(tmp_path, identity, _summary())


def test_identity_and_metric_drift_fail_closed(tmp_path: Path) -> None:
    identity = _identity()
    begin_locked_test(tmp_path, identity)
    with pytest.raises(TrainingResultError, match="marker identity"):
        validate_start_marker(tmp_path, replace(identity, authorization_sha256="8" * 64))
    invalid = _summary()
    invalid["locked_test"] = dict(invalid["locked_test"], harrell_c_index=1.2)
    with pytest.raises(TrainingResultError, match="C-index"):
        publish_training_result(tmp_path, identity, invalid)


def test_marker_and_result_symlinks_fail_closed(tmp_path: Path) -> None:
    identity = _identity()
    target = tmp_path / "target"
    target.write_text("{}\n", encoding="utf-8")
    (tmp_path / "locked_test_started.json").symlink_to(target)
    with pytest.raises(TrainingResultError, match="marker"):
        validate_start_marker(tmp_path, identity)
