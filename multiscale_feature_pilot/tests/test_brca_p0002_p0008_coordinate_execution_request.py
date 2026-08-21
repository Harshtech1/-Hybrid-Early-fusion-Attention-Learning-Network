from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
REQUEST = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_request.yaml"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_request_binds_exact_locked_policies_and_headers() -> None:
    request = yaml.safe_load(REQUEST.read_text(encoding="utf-8"))
    assert request["status"] == "REQUEST_PREPARED_EXECUTION_NOT_AUTHORIZED"
    assert request["executable"] is False
    assert request["proposed_execution"]["maximum_cpu_patient_workers"] == 2
    assert request["proposed_execution"]["exact_total_read_region_calls"] == 7
    assert request["proposed_execution"]["patient_labels"] == [f"P{i:04d}" for i in range(2, 9)]

    for label, binding in request["patients"].items():
        lower = label.lower()
        policy = ROOT / f"multiscale_feature_pilot/config/brca_{lower}_scale_coordinate_policy.yaml"
        header = ROOT / f"multiscale_feature_pilot/provenance/brca_{lower}_header_metadata_result/result.yaml"
        assert _sha256(policy) == binding["policy_sha256"]
        assert _sha256(header) == binding["header_result_sha256"]
        policy_document = yaml.safe_load(policy.read_text(encoding="utf-8"))
        assert policy_document["execution_boundary"]["status"] == "EXECUTION_LOCKED"
        assert not any(
            value
            for key, value in policy_document["execution_boundary"].items()
            if key != "status"
        )
        proposed = policy_document["future_mask_policy"]
        tuple_from_policy = proposed.get("read_region_tuple") or proposed.get("proposed_read_region")
        size = tuple_from_policy.get("size_at_level", tuple_from_policy.get("size"))
        assert binding["read_region"] == {
            "level_0_location": tuple_from_policy["level_0_location"],
            "level": tuple_from_policy["level"],
            "size_at_level": size,
        }


def test_request_keeps_every_execution_surface_locked() -> None:
    request = yaml.safe_load(REQUEST.read_text(encoding="utf-8"))
    assert request["authority"]
    assert all(value is False for value in request["authority"].values())
    assert request["required_stop"] == "COMBINED_COORDINATE_EXECUTION_REVIEW"
