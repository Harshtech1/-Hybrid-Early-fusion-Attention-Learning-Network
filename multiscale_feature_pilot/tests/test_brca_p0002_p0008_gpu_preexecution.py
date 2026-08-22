from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml

from multiscale_feature_pilot.src.brca_p0002_p0008_feature_package import (
    CHECKPOINT_SHA256,
    PACKAGE_LABELS,
    consolidated_first_eight_gpu_plan,
    prepare_p0002_p0008_feature_packages,
    rehearse_feature_recovery,
)
from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import EventType, ReplayAction, replay_events


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_gpu_preexecution.yaml"
MODULE = ROOT / "multiscale_feature_pilot/src/brca_p0002_p0008_feature_package.py"


def test_cpu_only_module_has_no_forbidden_runtime_imports() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = {
        node.names[0].name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import) and node.names
    }
    imports |= {
        node.module.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    assert not {"torch", "openslide", "torchvision"} & imports


def test_exact_coordinate_packages_and_natural_layouts_validate() -> None:
    packages = prepare_p0002_p0008_feature_packages()
    assert tuple(item.label for item in packages) == PACKAGE_LABELS
    assert [(item.scale_2x_rows, item.scale_4x_rows, item.total_rows) for item in packages] == [
        (9785, 2486, 12271), (3461, 921, 4382), (3933, 1034, 4967),
        (23559, 5971, 29530), (7505, 1962, 9467), (9238, 2407, 11645),
        (18877, 4799, 23676),
    ]
    assert all(item.natural_healnet_wsi_shape == (1, item.total_rows, 2048) for item in packages)
    assert sum(item.total_rows for item in packages) == 95938


def test_each_synthetic_recovery_reaches_feature_tip() -> None:
    for package in prepare_p0002_p0008_feature_packages():
        events = rehearse_feature_recovery(package)
        assert len(events) == 14
        assert [event.stage for event in events[::2]] == list(PatientStage)[:-1]
        assert all(event.event_type is EventType.STAGE_STARTED for event in events[::2])
        assert all(event.event_type is EventType.STAGE_SUCCEEDED for event in events[1::2])
        replay = replay_events(events)
        assert replay.action is ReplayAction.ADVANCE_STAGE
        assert replay.target_stage is PatientStage.TERMINAL_RECORDED


def test_consolidated_plan_is_serial_and_resource_bound() -> None:
    plan = consolidated_first_eight_gpu_plan(prepare_p0002_p0008_feature_packages())
    assert plan.patient_labels == ("P0001",) + PACKAGE_LABELS
    assert plan.execution_order == plan.patient_labels
    assert plan.maximum_concurrent_patients == 1
    assert plan.total_patch_reads == 112754
    assert plan.total_raw_feature_bytes == 923680768
    assert plan.estimated_gpu_wall_seconds_range == (1210, 1800)
    assert plan.checkpoint_sha256 == CHECKPOINT_SHA256


def test_config_is_execution_locked_and_matches_preparation_authority() -> None:
    record = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert record["status"] == "P0002_P0008_GPU_PREEXECUTION_PREPARED_EXECUTION_LOCKED"
    assert record["executable"] is False
    assert not any(record["execution_lock"].values())
    statement = record["preparation_authority"]["exact_statement"]
    assert hashlib.sha256(statement.encode()).hexdigest() == record["preparation_authority"]["exact_statement_sha256"]
    assert hashlib.sha256(record["future_exact_authorization_text"].encode()).hexdigest() == record["future_exact_authorization_text_sha256"]
    assert record["consolidated_future_gpu_plan"]["total_patch_reads"] == 112754
