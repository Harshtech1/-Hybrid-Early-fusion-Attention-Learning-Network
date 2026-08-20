from __future__ import annotations

import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from multiscale_feature_pilot.src.brca_p0001_feature_package import (
    CHECKPOINT_SHA256,
    COORDINATE_MANIFEST_SHA256,
    FUTURE_GPU_AUTHORIZATION_STATEMENT_SHA256,
    SCALE_2X_ROWS,
    SCALE_4X_ROWS,
    TOTAL_ROWS,
    rehearse_p0001_feature_transaction,
)
from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
    EventType,
    ReplayAction,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_brca_p0001_gpu_production.py"
PREEXECUTION = ROOT / "multiscale_feature_pilot/config/brca_p0001_gpu_preexecution.yaml"
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
CHECKPOINT = Path("/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")
COORDINATES = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/BRCA_PRODUCTION_P0001.coordinates"
)
OUTPUT = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/BRCA_PRODUCTION_P0001.features"
)
LEDGER = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/BRCA_PRODUCTION_P0001.recovery_v2"
)


def _load_runner():
    spec = importlib.util.spec_from_file_location("p0001_locked_gpu_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_preparation_authority_is_locked() -> None:
    record = yaml.safe_load(PREEXECUTION.read_text(encoding="utf-8"))
    assert record["status"] == (
        "P0001_PRODUCTION_GPU_PREEXECUTION_PREPARATION_AUTHORIZED_EXECUTION_LOCKED"
    )
    assert record["executable"] is False
    assert not any(record["execution_lock"].values())
    statement = record["preparation_authority"]["exact_statement"]
    assert len(statement) == 660
    assert hashlib.sha256(statement.encode()).hexdigest() == (
        record["preparation_authority"]["exact_statement_sha256"]
    )
    assert record["preparation_authority"]["exact_statement_sha256"] == (
        "0988b40c965de4a6dfbffde6a29dce11070931340d5ce6cab6c409ccee114976"
    )


def test_exact_p0001_natural_layout_checkpoint_and_estimates() -> None:
    record = yaml.safe_load(PREEXECUTION.read_text(encoding="utf-8"))
    inputs = record["verified_inputs"]
    assert (SCALE_2X_ROWS, SCALE_4X_ROWS, TOTAL_ROWS) == (13372, 3444, 16816)
    assert inputs["scale_2x_rows"] == SCALE_2X_ROWS
    assert inputs["scale_4x_rows"] == SCALE_4X_ROWS
    assert inputs["combined_shape"] == [TOTAL_ROWS, 2048]
    assert inputs["natural_healnet_wsi_shape"] == [1, TOTAL_ROWS, 2048]
    assert inputs["row_ranges"] == {
        "scale_2x": [0, SCALE_2X_ROWS],
        "scale_4x": [SCALE_2X_ROWS, TOTAL_ROWS],
    }
    assert inputs["pooling_performed"] is False
    assert inputs["transpose_performed"] is False
    assert inputs["checkpoint"]["sha256"] == CHECKPOINT_SHA256
    estimates = record["estimates_not_measurements"]
    assert estimates["raw_combined_tensor_bytes"] == TOTAL_ROWS * 2048 * 4
    assert estimates["gpu_wall_seconds_range"] == [180, 260]
    assert estimates["compact_artifact_bytes_range"] == [138500000, 139000000]


def test_future_authorization_text_is_exact_and_scoped() -> None:
    record = yaml.safe_load(PREEXECUTION.read_text(encoding="utf-8"))
    statement = record["future_exact_authorization_text"]
    assert hashlib.sha256(statement.encode()).hexdigest() == (
        record["future_exact_authorization_text_sha256"]
    )
    assert record["future_exact_authorization_text_sha256"] == (
        FUTURE_GPU_AUTHORIZATION_STATEMENT_SHA256
    )
    assert "13,372 scale-2x and 3,444 scale-4x patch reads" in statement
    assert "[16816,2048]" in statement
    assert "only cleanup of runner-created ephemeral recovery staging is permitted" in statement
    assert "No training" in statement


def test_synthetic_recovery_rehearsal_reaches_terminal_in_exact_order() -> None:
    result = rehearse_p0001_feature_transaction(ALIGNMENT)
    assert len(result.events) == 16
    assert result.final_action is ReplayAction.STOP_TERMINAL
    assert [event.stage for event in result.events[::2]] == list(PatientStage)
    assert all(event.event_type is EventType.STAGE_STARTED for event in result.events[::2])
    assert all(event.event_type is EventType.STAGE_SUCCEEDED for event in result.events[1::2])
    assert [event.sequence for event in result.events] == list(range(1, 17))
    assert len({event.record_sha256 for event in result.events}) == 16
    feature_success = result.events[13]
    assert feature_success.stage is PatientStage.FEATURES_VERIFIED
    assert dict(feature_success.output_hashes)["compact_feature_manifest"] == (
        result.compact_manifest_sha256
    )


def test_synthetic_rehearsal_can_stop_at_features_for_future_real_runner() -> None:
    result = rehearse_p0001_feature_transaction(
        ALIGNMENT,
        compact_manifest_sha256="a" * 64,
        through_stage=PatientStage.FEATURES_VERIFIED,
        event_timestamps=("2026-08-20T23:00:00Z",) * 16,
    )
    assert len(result.events) == 14
    assert result.final_action is ReplayAction.ADVANCE_STAGE
    assert result.events[-1].stage is PatientStage.FEATURES_VERIFIED
    assert dict(result.events[-1].output_hashes) == {
        "compact_feature_manifest": "a" * 64
    }


def test_runner_first_operation_is_lock_and_import_does_not_load_torch() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    run_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    first = run_function.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name)
    assert first.value.func.id == "_require_authorized"
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util,sys;"
                f"p={str(RUNNER)!r};"
                "s=importlib.util.spec_from_file_location('p0001_import',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_runner_is_locked_before_dependency_or_operational_work(monkeypatch) -> None:
    runner = _load_runner()
    assert runner.EXECUTION_AUTHORIZED is False
    assert not (ROOT / "multiscale_feature_pilot/config/brca_p0001_gpu_execution_authorization.yaml").exists()
    monkeypatch.setattr(
        runner,
        "_execute",
        lambda commit: pytest.fail("locked runner reached operational function"),
    )
    with pytest.raises(runner.ExecutionLocked, match="locked pending"):
        runner.run("0" * 40)


def test_real_cli_attempt_fails_before_output_ledger_or_torch_import() -> None:
    assert not os.path.lexists(OUTPUT)
    assert not os.path.lexists(LEDGER)
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--expected-source-commit", "0" * 40],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 1
    payload = yaml.safe_load(completed.stdout)
    assert payload["status"] == "BLOCKED"
    assert "ExecutionLocked" in payload["error"]
    assert not os.path.lexists(OUTPUT)
    assert not os.path.lexists(LEDGER)


def test_runner_binds_production_adapter_compact_publisher_and_recovery_v2() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    for token in (
        "BRCA_PRODUCTION_P0001.features",
        "BRCA_PRODUCTION_P0001.recovery_v2",
        "publish_compact_feature_artifacts",
        "validate_compact_feature_artifacts",
        "rehearse_p0001_feature_transaction",
        "append_control_event",
        "torch.cat((result_2x.features, result_4x.features), dim=0)",
        "preserve_failed_staging=True",
        'torch.device("cuda:0")',
        "CUBLAS_WORKSPACE_CONFIG",
        "authorization_statement_sha256",
        "not all(authorization[\"prohibited\"].values())",
    ):
        assert token in source
    assert "COUNTS = (13_372, 3_444)" in source
    assert "TOTAL = 16_816" in source
    runner = _load_runner()
    assert {
        Path("multiscale_feature_pilot/src/brca_singleton_streaming_policy.py"),
        Path("multiscale_feature_pilot/src/scale_2x_policy.py"),
    }.issubset(set(runner.BOUND))


def test_runner_has_no_training_calls_or_other_patient_paths() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"backward", "step", "train"}.intersection(calls)
    for label in (
        "BRCA_PRODUCTION_P0002",
        "BRCA_PRODUCTION_P0003",
        "BRCA_BATCH_B01",
        "BRCA_BATCH_B02",
        "BRCA_BATCH_B03",
        "BRCA_BATCH_B06",
        "Q25.features",
        "Q50.features",
        "Q75.features",
    ):
        assert label not in source


def test_local_checkpoint_and_coordinate_anchor_are_present_without_execution() -> None:
    assert CHECKPOINT.is_file() and not CHECKPOINT.is_symlink()
    assert CHECKPOINT.stat().st_size == 102_540_417
    hasher = hashlib.sha256()
    with CHECKPOINT.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    assert hasher.hexdigest() == CHECKPOINT_SHA256
    manifest = COORDINATES / "coordinate_manifest.json"
    assert manifest.is_file() and not manifest.is_symlink()
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == (
        COORDINATE_MANIFEST_SHA256
    )
