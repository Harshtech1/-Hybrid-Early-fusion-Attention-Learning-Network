from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import yaml

from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
    exact_manifest_bytes,
    load_frozen_cohort_order,
)


ROOT = Path(__file__).resolve().parents[2]


def _yaml(relative: str):
    return yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))


def test_final_survival_protocol_binds_exact_split_cutpoints_and_no_execution() -> None:
    protocol = _yaml("multiscale_feature_pilot/config/brca_survival_evaluation_protocol.yaml")
    assert protocol["status"] == "FINALIZED_EXECUTION_NOT_AUTHORIZED"
    assert protocol["split"]["counts"] == {
        "training": {"patients": 660, "observed_events": 79, "right_censored": 581},
        "validation": {"patients": 116, "observed_events": 14, "right_censored": 102},
        "locked_test": {"patients": 118, "observed_events": 33, "right_censored": 85},
    }
    assert protocol["discretization"]["internal_cutpoints_months"] == [22.505, 44.84, 67.940]
    assert protocol["training"]["true_gradient_accumulation_patients"] == 16
    assert not any(protocol["authorization"].values())


def test_first_eight_are_exact_production_rows_and_manifests() -> None:
    request = _yaml("multiscale_feature_pilot/config/brca_first_eight_production_execution_request.yaml")
    assert request["classification"] == "FIRST_PRODUCTION_BLOCK_NOT_SCIENTIFIC_PILOT"
    assert request["binding"]["exact_cohort_indices"] == list(range(1, 9))
    assert request["binding"]["exact_raw_wsi_bytes"] == 8_297_129_620
    assert not any(request["authorization"].values())
    cohort = load_frozen_cohort_order(ROOT / "reports/brca_row_level_alignment.csv")
    directory = ROOT / "multiscale_feature_pilot/provenance/brca_first_eight_production_manifests"
    files = sorted(directory.iterdir())
    assert len(files) == 8
    for index, (binding, path) in enumerate(zip(cohort[:8], files), start=1):
        assert path.name == f"P{index:04d}_{binding.gdc_uuid}.REQUEST_ONLY.gdc.tsv"
        assert path.read_bytes() == exact_manifest_bytes(binding)


def test_generated_split_and_cutpoints_are_cross_bound() -> None:
    split_path = ROOT / "multiscale_feature_pilot/provenance/brca_894_survival_split.tsv"
    cutpoint_path = ROOT / "multiscale_feature_pilot/provenance/brca_894_survival_cutpoints.json"
    rows = list(csv.DictReader(split_path.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
    cutpoints = json.loads(cutpoint_path.read_text(encoding="utf-8"))
    assert len(rows) == 894
    assert cutpoints["split_manifest_sha256"] == hashlib.sha256(split_path.read_bytes()).hexdigest()
    assert cutpoints["internal_cutpoints_months"] == ["22.505", "44.84", "67.940"]
    assert {partition: sum(row["partition"] == partition for row in rows) for partition in ("training", "validation", "locked_test")} == {
        "training": 660, "validation": 116, "locked_test": 118,
    }


def test_training_request_and_raw_lifecycle_remain_locked() -> None:
    training = _yaml("multiscale_feature_pilot/config/brca_894_healnet_training_execution_request.yaml")
    assert training["status"] == "TRAINING_PACKAGE_READY_FOR_EXECUTION_REVIEW_NOT_AUTHORIZED"
    assert training["prerequisites"]["compact_feature_registry"]["status"] == "MISSING_UNTIL_894_PRODUCTION_FEATURES_COMPLETE"
    assert training["runner_contract"]["execution_authorized_constant"] is False
    assert not any(training["authorization"].values())
    lifecycle = _yaml("multiscale_feature_pilot/config/brca_raw_wsi_lifecycle_proposal.yaml")
    assert lifecycle["status"] == "PROPOSAL_ONLY_NO_DELETION_AUTHORIZED"
    assert lifecycle["retention"]["b06"] == "RETAIN_INDEFINITELY_PENDING_SEPARATE_DECISION"
    assert lifecycle["safe_to_release_local_raw_gate"]["deletion_of_b06"] == "prohibited"
    assert not any(lifecycle["authorization"].values())
