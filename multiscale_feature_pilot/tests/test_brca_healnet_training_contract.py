from __future__ import annotations

from dataclasses import replace
import hashlib
import math
from pathlib import Path

import pytest

from multiscale_feature_pilot.src.brca_healnet_training_contract import (
    FeatureRegistryEntry,
    TrainingContractError,
    TrainingPolicy,
    deterministic_epoch_order,
    discrete_time_nll,
    feature_registry_bytes,
    harrell_concordance_index,
    patient_bootstrap_c_index,
    parse_feature_registry,
    validate_training_policy,
)
from multiscale_feature_pilot.src.brca_survival_protocol import parse_and_validate_split_artifacts


ROOT = Path(__file__).resolve().parents[2]
SPLIT = ROOT / "multiscale_feature_pilot/provenance/brca_894_survival_split.tsv"
CUTPOINTS = ROOT / "multiscale_feature_pilot/provenance/brca_894_survival_cutpoints.json"


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _split():
    return parse_and_validate_split_artifacts(SPLIT.read_bytes(), CUTPOINTS.read_bytes())


def _registry():
    return tuple(
        FeatureRegistryEntry(
            cohort_index=record.cohort_index,
            patient_id=record.patient_id,
            slide_id=record.slide_id,
            gdc_uuid=record.gdc_uuid,
            partition=record.partition,
            compact_directory=f"/future/brca_pilot_data/patient-{record.cohort_index:04d}.features",
            compact_manifest_sha256=_hash(f"manifest:{record.cohort_index}"),
            combined_tensor_file_sha256=_hash(f"file:{record.cohort_index}"),
            combined_tensor_content_sha256=_hash(f"content:{record.cohort_index}"),
            row_provenance_file_sha256=_hash(f"provenance:{record.cohort_index}"),
            rows=1000 + record.cohort_index,
        )
        for record in _split().records
    )


def test_frozen_training_policy() -> None:
    policy = TrainingPolicy()
    validate_training_policy(policy)
    with pytest.raises(TrainingContractError, match="AMP|precision"):
        validate_training_policy(replace(policy, amp=True))
    with pytest.raises(TrainingContractError, match="accumulation"):
        validate_training_policy(replace(policy, gradient_accumulation_patients=1))


def test_exact_894_feature_registry_round_trip() -> None:
    split = _split()
    entries = _registry()
    payload = feature_registry_bytes(entries)
    assert parse_feature_registry(payload, split) == entries
    assert len(entries) == 894


def test_feature_registry_identity_partition_and_hash_drift_fail_closed() -> None:
    split = _split()
    entries = list(_registry())
    entries[0] = replace(entries[0], patient_id="TCGA-00-0000")
    with pytest.raises(TrainingContractError, match="patient mismatch"):
        parse_feature_registry(feature_registry_bytes(entries), split)
    entries = list(_registry())
    entries[0] = replace(entries[0], partition="locked_test")
    with pytest.raises(TrainingContractError, match="partition mismatch"):
        parse_feature_registry(feature_registry_bytes(entries), split)
    entries = list(_registry())
    entries[0] = replace(entries[0], compact_manifest_sha256="not-a-hash")
    with pytest.raises(TrainingContractError, match="invalid"):
        parse_feature_registry(feature_registry_bytes(entries), split)


def test_epoch_order_is_deterministic_epoch_specific_and_training_only() -> None:
    training = tuple(entry for entry in _registry() if entry.partition == "training")
    assert deterministic_epoch_order(training, epoch=1) == deterministic_epoch_order(training, epoch=1)
    assert deterministic_epoch_order(training, epoch=1) != deterministic_epoch_order(training, epoch=2)
    with pytest.raises(TrainingContractError, match="training entries"):
        deterministic_epoch_order(_registry(), epoch=1)


def test_harrell_c_index_counts_concordance_discordance_and_ties() -> None:
    perfect = harrell_concordance_index([1, 1, 0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
    assert perfect.concordance_index == 1.0
    assert perfect.comparable_pairs == 3
    inverse = harrell_concordance_index([1, 1, 0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert inverse.concordance_index == 0.0
    tied = harrell_concordance_index([1, 0], [1.0, 2.0], [1.0, 1.0])
    assert tied.concordance_index == 0.5 and tied.tied_risk_pairs == 1
    same_time = harrell_concordance_index([1, 0], [1.0, 1.0], [2.0, 1.0])
    assert same_time.concordance_index == 1.0


def test_discrete_nll_is_finite_and_rejects_invalid_hazards() -> None:
    event_loss = discrete_time_nll([0.2, 0.3, 0.4, 0.5], discrete_time_bin=2, censorship=0)
    censored_loss = discrete_time_nll([0.2, 0.3, 0.4, 0.5], discrete_time_bin=2, censorship=1)
    # Exact scalar equivalents of the pinned HEALNet nll_loss masks:
    # uncensored = -(1-c)[log(S_{Y-1}) + log(h_Y)]
    # censored = -c log(S_Y), loss = (1-alpha)(uncensored+censored)+alpha*uncensored.
    assert event_loss == pytest.approx(-(math.log(0.8 * 0.7) + math.log(0.4)))
    assert censored_loss == pytest.approx(0.6 * -math.log(0.8 * 0.7 * 0.6))
    assert event_loss > censored_loss > 0
    with pytest.raises(TrainingContractError, match="probabilities"):
        discrete_time_nll([0.2, 0.0, 0.4, 0.5], discrete_time_bin=2, censorship=0)


def test_patient_bootstrap_c_index_is_deterministic_and_bounded() -> None:
    events = [1, 1, 1, 0, 0, 1]
    times = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    risk = [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
    first = patient_bootstrap_c_index(events, times, risk, replicates=1000, seed=7)
    second = patient_bootstrap_c_index(events, times, risk, replicates=1000, seed=7)
    assert first == second
    assert 0 <= first.lower <= first.upper <= 1
    assert first.valid_replicates >= 950
