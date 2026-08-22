from __future__ import annotations

import json
from pathlib import Path

import pytest

from multiscale_feature_pilot.src.artifacts import (
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)
from scripts import run_blca_one_patient_pilot as runner


def test_output_destination_rejects_both_git_checkouts(tmp_path: Path) -> None:
    official = tmp_path / "official"
    official.mkdir()

    with pytest.raises(runner.PilotContractError, match="pilot Git repository"):
        runner._validate_output_destination(runner.REPO_ROOT / "artifacts", official)
    with pytest.raises(runner.PilotContractError, match="official Git repository"):
        runner._validate_output_destination(official / "artifacts", official)


def test_output_destination_accepts_fresh_external_directory(tmp_path: Path) -> None:
    official = tmp_path / "official"
    official.mkdir()

    runner._validate_output_destination(tmp_path / "external" / "pilot", official)


def test_json_normalisation_matches_persisted_tuple_shapes() -> None:
    assert runner._normalise_json_value(
        {"output_shape": (1, 4), "finite": True}
    ) == {"output_shape": [1, 4], "finite": True}


def test_transaction_publishes_complete_directory_and_persisted_manifest(
    tmp_path: Path,
) -> None:
    final = tmp_path / "pilot_outputs"
    staging, lock = runner._begin_output_transaction(final)
    manifest = {"status": "BLCA_ONE_PATIENT_PILOT_SUCCESS", "rows": 17}
    manifest_record = atomic_write_json(manifest, staging / "pilot_manifest.json")
    atomic_write_text(
        f"{manifest_record.sha256}  pilot_manifest.json\n",
        staging / "pilot_manifest.sha256",
    )
    (staging / "features.pt").write_bytes(b"validated-feature-placeholder")

    runner._publish_output_transaction(staging, final, lock)

    assert final.is_dir()
    assert not staging.exists()
    assert not lock.exists()
    assert json.loads((final / "pilot_manifest.json").read_text()) == manifest
    assert sha256_file(final / "pilot_manifest.json") == manifest_record.sha256
    assert (final / "pilot_manifest.sha256").read_text().startswith(
        manifest_record.sha256
    )
    assert (final / "features.pt").read_bytes() == b"validated-feature-placeholder"


def test_transaction_refuses_existing_final_directory(tmp_path: Path) -> None:
    final = tmp_path / "pilot_outputs"
    staging, lock = runner._begin_output_transaction(final)
    final.mkdir()

    with pytest.raises(runner.PilotContractError, match="refusing to replace"):
        runner._publish_output_transaction(staging, final, lock)


def test_output_destination_refuses_stale_incomplete_transaction(
    tmp_path: Path,
) -> None:
    official = tmp_path / "official"
    official.mkdir()
    final = tmp_path / "pilot_outputs"
    staging, lock = runner._transaction_paths(final)
    staging.mkdir()
    lock.write_text("pid=123\n", encoding="ascii")

    with pytest.raises(runner.PilotContractError, match="stale pilot staging"):
        runner._validate_output_destination(final, official)
