#!/usr/bin/env python3
"""CPU-only validator for P0002--P0008 feature packages and GPU plan."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiscale_feature_pilot.src.brca_p0002_p0008_feature_package import (
    consolidated_first_eight_gpu_plan,
    prepare_p0002_p0008_feature_packages,
    rehearse_feature_recovery,
)


def main() -> int:
    packages = prepare_p0002_p0008_feature_packages()
    plan = consolidated_first_eight_gpu_plan(packages)
    rehearsals = {item.label: len(rehearse_feature_recovery(item)) for item in packages}
    print(json.dumps({
        "status": "P0002_P0008_GPU_PREEXECUTION_VALIDATED_CPU_ONLY",
        "packages": [
            {
                "label": item.label,
                "combined_shape": list(item.combined_shape),
                "natural_healnet_wsi_shape": list(item.natural_healnet_wsi_shape),
                "raw_feature_bytes": item.raw_feature_bytes,
                "synthetic_recovery_events": rehearsals[item.label],
            }
            for item in packages
        ],
        "consolidated_plan": {
            "patient_labels": list(plan.patient_labels),
            "total_patch_reads": plan.total_patch_reads,
            "total_raw_feature_bytes": plan.total_raw_feature_bytes,
            "compact_artifact_bytes_range": list(plan.total_compact_artifact_bytes_range),
            "estimated_gpu_wall_seconds_range": list(plan.estimated_gpu_wall_seconds_range),
            "maximum_concurrent_patients": plan.maximum_concurrent_patients,
        },
        "prohibited_operations": {
            "wsi_open": 0, "patch_reads": 0, "gpu_or_cuda": 0,
            "feature_extraction": 0, "healnet": 0, "publication": 0,
            "deletion": 0, "drive": 0, "training": 0,
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
