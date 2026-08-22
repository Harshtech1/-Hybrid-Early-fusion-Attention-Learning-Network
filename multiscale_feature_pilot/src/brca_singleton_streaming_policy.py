"""Pure planning contracts for the BRCA singleton streaming pipeline.

This module deliberately has no filesystem, network, OpenSlide, Torch, CUDA,
or training surface. It describes the order in which separately authorized
execution gates must be satisfied; it does not execute those gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from statistics import mean
from typing import Iterable


class StreamingPolicyError(ValueError):
    """Raised when a proposed singleton-streaming plan violates policy."""


class PatientStage(str, Enum):
    PLANNED = "PLANNED"
    ACQUISITION_AUTHORIZED = "ACQUISITION_AUTHORIZED"
    RAW_VERIFIED = "RAW_VERIFIED"
    HEADER_POLICY_VERIFIED = "HEADER_POLICY_VERIFIED"
    COORDINATES_VERIFIED = "COORDINATES_VERIFIED"
    GPU_AUTHORIZED = "GPU_AUTHORIZED"
    FEATURES_VERIFIED = "FEATURES_VERIFIED"
    TERMINAL_RECORDED = "TERMINAL_RECORDED"


_NEXT_STAGE = {
    PatientStage.PLANNED: PatientStage.ACQUISITION_AUTHORIZED,
    PatientStage.ACQUISITION_AUTHORIZED: PatientStage.RAW_VERIFIED,
    PatientStage.RAW_VERIFIED: PatientStage.HEADER_POLICY_VERIFIED,
    PatientStage.HEADER_POLICY_VERIFIED: PatientStage.COORDINATES_VERIFIED,
    PatientStage.COORDINATES_VERIFIED: PatientStage.GPU_AUTHORIZED,
    PatientStage.GPU_AUTHORIZED: PatientStage.FEATURES_VERIFIED,
    PatientStage.FEATURES_VERIFIED: PatientStage.TERMINAL_RECORDED,
}


@dataclass(frozen=True)
class PilotObservation:
    candidate: str
    patch_count: int
    gpu_runtime_seconds: float
    retained_artifact_bytes: int


@dataclass(frozen=True)
class CohortEstimate:
    patients: int
    observed_patch_count_min: int
    observed_patch_count_mean: float
    observed_patch_count_max: int
    projected_gpu_hours_at_observed_min: float
    projected_gpu_hours_at_observed_mean: float
    projected_gpu_hours_at_observed_max: float
    projected_complete_artifact_bytes_at_observed_min: int
    projected_complete_artifact_bytes_at_observed_mean: int
    projected_complete_artifact_bytes_at_observed_max: int
    estimates_not_capacity_guarantees: bool = True


def advance_stage(
    current: PatientStage,
    requested: PatientStage,
    *,
    separately_authorized: bool,
    evidence_verified: bool,
) -> PatientStage:
    """Validate one state transition without performing an operation."""

    expected = _NEXT_STAGE.get(current)
    if expected is None:
        raise StreamingPolicyError("terminal patient record cannot advance")
    if requested is not expected:
        raise StreamingPolicyError(
            f"stage must advance exactly from {current.value} to {expected.value}"
        )
    if not separately_authorized:
        raise StreamingPolicyError("next patient stage lacks separate authorization")
    if not evidence_verified:
        raise StreamingPolicyError("next patient stage lacks verified evidence")
    return requested


def validate_static_plan(
    *,
    patient_count: int,
    concurrency: int,
    initial_batch_size: int,
    raw_wsi_retention_count: int,
    quota_bytes: int,
    safety_floor_bytes: int,
    compact_retention_only: bool,
) -> None:
    """Validate the non-executable cohort policy constants."""

    if patient_count != 894:
        raise StreamingPolicyError("singleton cohort must contain exactly 894 patients")
    if concurrency != 1 or initial_batch_size != 1:
        raise StreamingPolicyError("download, extraction, and model batch concurrency must be one")
    if raw_wsi_retention_count != 1:
        raise StreamingPolicyError("streaming design may retain at most one raw WSI")
    if quota_bytes != 200_000_000_000:
        raise StreamingPolicyError("Lightning organization quota must remain 200 GB")
    if safety_floor_bytes < 20_000_000_000:
        raise StreamingPolicyError("storage safety floor must be at least 20 GB")
    if not compact_retention_only:
        raise StreamingPolicyError("complete duplicate feature retention exceeds the quota")


def validate_storage_headroom(
    *,
    available_bytes: int,
    raw_wsi_bytes: int,
    staging_bytes: int,
    final_artifact_bytes: int,
    safety_floor_bytes: int = 20_000_000_000,
) -> int:
    """Fail closed unless one transaction and the safety floor fit."""

    values = {
        "available_bytes": available_bytes,
        "raw_wsi_bytes": raw_wsi_bytes,
        "staging_bytes": staging_bytes,
        "final_artifact_bytes": final_artifact_bytes,
        "safety_floor_bytes": safety_floor_bytes,
    }
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values.values()):
        raise StreamingPolicyError("storage values must be non-negative integers")
    if safety_floor_bytes < 20_000_000_000:
        raise StreamingPolicyError("storage safety floor must remain at least 20 GB")
    required = raw_wsi_bytes + staging_bytes + final_artifact_bytes + safety_floor_bytes
    if available_bytes < required:
        raise StreamingPolicyError(
            f"insufficient storage headroom: require {required}, have {available_bytes}"
        )
    return required


def estimate_from_pilots(
    observations: Iterable[PilotObservation], *, patient_count: int = 894
) -> CohortEstimate:
    """Scale three measured pilots; returned values remain explicit estimates."""

    values = tuple(observations)
    if tuple(item.candidate for item in values) != ("Q25", "Q50", "Q75"):
        raise StreamingPolicyError("estimates require ordered Q25, Q50, and Q75 observations")
    if patient_count != 894:
        raise StreamingPolicyError("estimate is bound to the 894 singleton cohort")
    if any(
        item.patch_count <= 0
        or item.gpu_runtime_seconds <= 0
        or item.retained_artifact_bytes <= 0
        for item in values
    ):
        raise StreamingPolicyError("pilot measurements must be positive")

    patches = [item.patch_count for item in values]
    runtimes = [item.gpu_runtime_seconds for item in values]
    artifacts = [item.retained_artifact_bytes for item in values]
    return CohortEstimate(
        patients=patient_count,
        observed_patch_count_min=min(patches),
        observed_patch_count_mean=mean(patches),
        observed_patch_count_max=max(patches),
        projected_gpu_hours_at_observed_min=min(runtimes) * patient_count / 3600,
        projected_gpu_hours_at_observed_mean=mean(runtimes) * patient_count / 3600,
        projected_gpu_hours_at_observed_max=max(runtimes) * patient_count / 3600,
        projected_complete_artifact_bytes_at_observed_min=min(artifacts) * patient_count,
        projected_complete_artifact_bytes_at_observed_mean=round(mean(artifacts) * patient_count),
        projected_complete_artifact_bytes_at_observed_max=max(artifacts) * patient_count,
    )


__all__ = [
    "CohortEstimate",
    "PatientStage",
    "PilotObservation",
    "StreamingPolicyError",
    "advance_stage",
    "estimate_from_pilots",
    "validate_static_plan",
    "validate_storage_headroom",
]
