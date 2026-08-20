"""Fail-closed metadata contract for the BRCA HEALNet training run.

The functions in this module validate frozen protocol and feature-registry
metadata, compute deterministic patient order, and provide framework-neutral
survival metric/loss cross-checks.  They do not load tensors or execute a model.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import math
from pathlib import Path, PurePosixPath
import random
import re
from typing import Sequence

from .brca_survival_protocol import FrozenSurvivalSplit


SPLIT_MANIFEST_SHA256 = "3e519a26eaa24852862bf368a48cceffaf26783c73e80007737134ae6ed626ad"
CUTPOINTS_SHA256 = "77b514098387f883a7bc5205ef191ff321aff94e6c874442f7ccc65ff9059d6d"
OFFICIAL_HEALNET_COMMIT = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
OFFICIAL_HEALNET_MODEL_SHA256 = "78e9c2b10455f6ad43a88ed4e0a646cbf6304bf52cac06a5140f49dbc4796820"
OFFICIAL_HEALNET_LOSS_SHA256 = "63dcd4c83169dcddea68a3c623ba3486d4263956e14d1a3db9480304470fdd2f"
ENCODER_CHECKPOINT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
OMIC_ARCHIVE_SHA256 = "4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8"
FEATURE_REGISTRY_COLUMNS = (
    "cohort_index",
    "patient_id",
    "slide_id",
    "gdc_uuid",
    "partition",
    "compact_directory",
    "compact_manifest_sha256",
    "combined_tensor_file_sha256",
    "combined_tensor_content_sha256",
    "row_provenance_file_sha256",
    "rows",
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TrainingContractError(RuntimeError):
    """Raised when the frozen training contract or registry drifts."""


@dataclass(frozen=True)
class TrainingPolicy:
    protocol_id: str = "BRCA_HEALNET_IMAGENET1K_V2_SINGLE_SPLIT_V1"
    seed: int = 20_260_820
    modalities: tuple[str, ...] = ("wsi", "rna", "mutation", "cnv")
    channel_dims: tuple[int, ...] = (2048, 1558, 21, 1333)
    output_bins: int = 4
    patient_batch_size: int = 1
    gradient_accumulation_patients: int = 16
    maximum_epochs: int = 50
    early_stopping_patience: int = 5
    optimizer: str = "Adam"
    learning_rate: float = 0.007765016508403882
    scheduler: str = "OneCycleLR"
    maximum_learning_rate: float = 0.008
    l1_coefficient: float = 0.00000682386175773137
    bin_weighting: str = "none"
    precision: str = "float32"
    amp: bool = False
    tf32: bool = False
    cpu_fallback: bool = False
    primary_metric: str = "Harrell_concordance_index"
    early_stopping_monitor: str = "validation_nll"


@dataclass(frozen=True)
class FeatureRegistryEntry:
    cohort_index: int
    patient_id: str
    slide_id: str
    gdc_uuid: str
    partition: str
    compact_directory: str
    compact_manifest_sha256: str
    combined_tensor_file_sha256: str
    combined_tensor_content_sha256: str
    row_provenance_file_sha256: str
    rows: int


@dataclass(frozen=True)
class ConcordanceResult:
    concordance_index: float
    concordant_pairs: int
    discordant_pairs: int
    tied_risk_pairs: int
    comparable_pairs: int


@dataclass(frozen=True)
class BootstrapInterval:
    lower: float
    upper: float
    valid_replicates: int
    requested_replicates: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingContractError(message)


def validate_training_policy(policy: TrainingPolicy) -> None:
    _require(policy.protocol_id == "BRCA_HEALNET_IMAGENET1K_V2_SINGLE_SPLIT_V1", "training protocol ID drift")
    _require(policy.seed == 20_260_820, "training seed drift")
    _require(policy.modalities == ("wsi", "rna", "mutation", "cnv"), "modality order drift")
    _require(policy.channel_dims == (2048, 1558, 21, 1333), "channel dimensions drift")
    _require(policy.output_bins == 4 and policy.patient_batch_size == 1, "output bins or patient batch size drift")
    _require(policy.gradient_accumulation_patients == 16, "gradient accumulation drift")
    _require(policy.maximum_epochs == 50 and policy.early_stopping_patience == 5, "epoch/early-stop contract drift")
    _require(policy.optimizer == "Adam" and policy.scheduler == "OneCycleLR", "optimizer/scheduler drift")
    _require(math.isclose(policy.learning_rate, 0.007765016508403882, rel_tol=0, abs_tol=0), "learning rate drift")
    _require(math.isclose(policy.maximum_learning_rate, 0.008, rel_tol=0, abs_tol=0), "maximum learning rate drift")
    _require(math.isclose(policy.l1_coefficient, 0.00000682386175773137, rel_tol=0, abs_tol=0), "L1 coefficient drift")
    _require(policy.bin_weighting == "none", "survival-bin weighting must remain disabled")
    _require(policy.precision == "float32" and not policy.amp and not policy.tf32, "precision contract drift")
    _require(not policy.cpu_fallback, "CPU fallback is prohibited for authorized training")
    _require(policy.primary_metric == "Harrell_concordance_index", "primary metric drift")
    _require(policy.early_stopping_monitor == "validation_nll", "early-stopping monitor drift")


def feature_registry_bytes(entries: Sequence[FeatureRegistryEntry]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, delimiter="\t", lineterminator="\n")
    writer.writerow(FEATURE_REGISTRY_COLUMNS)
    for entry in entries:
        writer.writerow(tuple(getattr(entry, field) for field in FEATURE_REGISTRY_COLUMNS))
    return buffer.getvalue().encode("utf-8")


def parse_feature_registry(payload: bytes, split: FrozenSurvivalSplit) -> tuple[FeatureRegistryEntry, ...]:
    try:
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""), delimiter="\t", strict=True)
    except (UnicodeDecodeError, csv.Error) as error:
        raise TrainingContractError("feature registry is not strict UTF-8 TSV") from error
    _require(tuple(reader.fieldnames or ()) == FEATURE_REGISTRY_COLUMNS, "feature registry columns drift")
    split_by_index = {record.cohort_index: record for record in split.records}
    entries: list[FeatureRegistryEntry] = []
    for expected_index, row in enumerate(reader, start=1):
        try:
            cohort_index = int(row["cohort_index"])
            rows = int(row["rows"])
        except ValueError as error:
            raise TrainingContractError("feature registry integer field is invalid") from error
        _require(cohort_index == expected_index, "feature registry cohort order is not contiguous")
        _require(cohort_index in split_by_index, "feature registry cohort index is unknown")
        expected = split_by_index[cohort_index]
        _require(row["patient_id"] == expected.patient_id, "feature registry patient mismatch")
        _require(row["slide_id"] == expected.slide_id, "feature registry slide mismatch")
        _require(row["gdc_uuid"] == expected.gdc_uuid, "feature registry UUID mismatch")
        _require(row["partition"] == expected.partition, "feature registry partition mismatch")
        directory = PurePosixPath(row["compact_directory"])
        _require(directory.is_absolute() and ".." not in directory.parts, "compact directory must be an absolute normalized path")
        for label in (
            "compact_manifest_sha256", "combined_tensor_file_sha256",
            "combined_tensor_content_sha256", "row_provenance_file_sha256",
        ):
            _require(_SHA256.fullmatch(row[label]) is not None, f"feature registry {label} is invalid")
        _require(rows > 1, "feature registry row count must exceed one")
        entries.append(
            FeatureRegistryEntry(
                cohort_index=cohort_index,
                patient_id=row["patient_id"],
                slide_id=row["slide_id"],
                gdc_uuid=row["gdc_uuid"],
                partition=row["partition"],
                compact_directory=row["compact_directory"],
                compact_manifest_sha256=row["compact_manifest_sha256"],
                combined_tensor_file_sha256=row["combined_tensor_file_sha256"],
                combined_tensor_content_sha256=row["combined_tensor_content_sha256"],
                row_provenance_file_sha256=row["row_provenance_file_sha256"],
                rows=rows,
            )
        )
    _require(len(entries) == 894, "feature registry must contain exactly 894 patients")
    _require(len({entry.compact_directory for entry in entries}) == 894, "feature registry reuses a compact directory")
    _require(
        {partition: sum(entry.partition == partition for entry in entries) for partition in ("training", "validation", "locked_test")}
        == {"training": 660, "validation": 116, "locked_test": 118},
        "feature registry partition counts drift",
    )
    return tuple(entries)


def deterministic_epoch_order(entries: Sequence[FeatureRegistryEntry], *, epoch: int, seed: int = 20_260_820) -> tuple[int, ...]:
    _require(epoch >= 1, "epoch must be positive")
    _require(all(entry.partition == "training" for entry in entries), "epoch ordering accepts training entries only")
    generator = random.Random(f"{seed}:{epoch}:BRCA_HEALNET_V1")
    indices = [entry.cohort_index for entry in entries]
    generator.shuffle(indices)
    return tuple(indices)


def harrell_concordance_index(
    event_observed: Sequence[int],
    event_time: Sequence[float],
    risk: Sequence[float],
    *,
    tied_tolerance: float = 1.0e-8,
) -> ConcordanceResult:
    """Compute Harrell C using sksurv-compatible comparable-pair semantics."""

    _require(len(event_observed) == len(event_time) == len(risk) and len(risk) >= 2, "metric inputs must have equal length >=2")
    rows = []
    for event, time, estimate in zip(event_observed, event_time, risk):
        _require(event in {0, 1}, "event indicator must be binary")
        _require(math.isfinite(time) and time > 0, "event time must be finite and positive")
        _require(math.isfinite(estimate), "risk estimate must be finite")
        rows.append((float(time), int(event), float(estimate)))
    rows.sort(key=lambda item: item[0])
    concordant = 0
    discordant = 0
    tied = 0
    start = 0
    while start < len(rows):
        end = start + 1
        while end < len(rows) and rows[end][0] == rows[start][0]:
            end += 1
        censored_same_time = [row for row in rows[start:end] if row[1] == 0]
        later = rows[end:]
        for _, event, estimate in rows[start:end]:
            if event != 1:
                continue
            for _, _, comparison in (*censored_same_time, *later):
                difference = estimate - comparison
                if abs(difference) <= tied_tolerance:
                    tied += 1
                elif difference > 0:
                    concordant += 1
                else:
                    discordant += 1
        start = end
    comparable = concordant + discordant + tied
    _require(comparable > 0, "Harrell C is not estimable: no comparable pairs")
    value = (concordant + 0.5 * tied) / comparable
    return ConcordanceResult(value, concordant, discordant, tied, comparable)


def patient_bootstrap_c_index(
    event_observed: Sequence[int],
    event_time: Sequence[float],
    risk: Sequence[float],
    *,
    replicates: int = 2000,
    seed: int = 20_260_820,
) -> BootstrapInterval:
    """Deterministic patient bootstrap percentile interval for Harrell C."""

    _require(len(event_observed) == len(event_time) == len(risk) >= 2, "bootstrap inputs must align")
    _require(replicates >= 1000, "at least 1000 bootstrap replicates are required")
    generator = random.Random(seed)
    values: list[float] = []
    population = len(risk)
    for _ in range(replicates):
        indices = [generator.randrange(population) for _ in range(population)]
        try:
            result = harrell_concordance_index(
                [event_observed[index] for index in indices],
                [event_time[index] for index in indices],
                [risk[index] for index in indices],
            )
        except TrainingContractError as error:
            if "no comparable pairs" not in str(error):
                raise
            continue
        values.append(result.concordance_index)
    _require(len(values) >= math.ceil(replicates * 0.95), "too many bootstrap replicates lack comparable pairs")
    values.sort()

    def percentile(probability: float) -> float:
        position = (len(values) - 1) * probability
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return values[lower]
        return values[lower] + (position - lower) * (values[upper] - values[lower])

    return BootstrapInterval(percentile(0.025), percentile(0.975), len(values), replicates)


def discrete_time_nll(
    hazards: Sequence[float],
    *,
    discrete_time_bin: int,
    censorship: int,
    alpha: float = 0.4,
    epsilon: float = 1.0e-7,
) -> float:
    """Framework-neutral cross-check of the released discrete NLL semantics."""

    _require(len(hazards) == 4 and discrete_time_bin in {0, 1, 2, 3}, "hazard/bin shape contract drift")
    _require(censorship in {0, 1}, "censorship must be binary")
    _require(0 <= alpha <= 1 and 0 < epsilon < 1, "NLL alpha/epsilon is invalid")
    survival = []
    cumulative = 1.0
    for hazard in hazards:
        _require(math.isfinite(hazard) and 0 < hazard < 1, "hazards must be finite probabilities in (0,1)")
        cumulative *= 1.0 - hazard
        survival.append(cumulative)
    previous_survival = 1.0 if discrete_time_bin == 0 else survival[discrete_time_bin - 1]
    current_survival = survival[discrete_time_bin]
    current_hazard = hazards[discrete_time_bin]
    uncensored_loss = -(1 - censorship) * (
        math.log(max(previous_survival, epsilon))
        + math.log(max(current_hazard, epsilon))
    )
    censored_loss = -censorship * math.log(max(current_survival, epsilon))
    negative_log_likelihood = uncensored_loss + censored_loss
    return (1 - alpha) * negative_log_likelihood + alpha * uncensored_loss


__all__ = [
    "CUTPOINTS_SHA256",
    "ENCODER_CHECKPOINT_SHA256",
    "FEATURE_REGISTRY_COLUMNS",
    "OFFICIAL_HEALNET_COMMIT",
    "OFFICIAL_HEALNET_LOSS_SHA256",
    "OFFICIAL_HEALNET_MODEL_SHA256",
    "OMIC_ARCHIVE_SHA256",
    "SPLIT_MANIFEST_SHA256",
    "ConcordanceResult",
    "BootstrapInterval",
    "FeatureRegistryEntry",
    "TrainingContractError",
    "TrainingPolicy",
    "deterministic_epoch_order",
    "discrete_time_nll",
    "feature_registry_bytes",
    "harrell_concordance_index",
    "patient_bootstrap_c_index",
    "parse_feature_registry",
    "validate_training_policy",
]
