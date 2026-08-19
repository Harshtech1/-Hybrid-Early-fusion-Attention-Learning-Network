#!/usr/bin/env python3
"""Build the authorized Q75-only GDC manifest without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from multiscale_feature_pilot.src.brca_q75_authorized_manifest import (  # noqa: E402
    DEFAULT_AUTHORIZATION_CONFIG,
    DEFAULT_GUARDED_DIRECTORY,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_Q25_RESULT,
    DEFAULT_Q50_RESULT,
    DEFAULT_REVIEW_PROVENANCE,
    build_q75_authorized_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guarded-directory", type=Path, default=DEFAULT_GUARDED_DIRECTORY
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument(
        "--authorization", type=Path, default=DEFAULT_AUTHORIZATION_CONFIG
    )
    parser.add_argument(
        "--review-provenance", type=Path, default=DEFAULT_REVIEW_PROVENANCE
    )
    parser.add_argument("--q25-result", type=Path, default=DEFAULT_Q25_RESULT)
    parser.add_argument("--q50-result", type=Path, default=DEFAULT_Q50_RESULT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact = build_q75_authorized_manifest(
        output_directory=args.output_directory,
        guarded_directory=args.guarded_directory,
        authorization_path=args.authorization,
        review_provenance_path=args.review_provenance,
        q25_result_path=args.q25_result,
        q50_result_path=args.q50_result,
    )
    print(
        json.dumps(
            {
                "status": "AUTHORIZED_Q75_ONLY",
                "network_request_performed": False,
                "download_invoked": False,
                "wsi_opened": False,
                "pixel_or_region_reads": 0,
                "use_cuda": False,
                "google_drive_operations": 0,
                "label": artifact.selection.label,
                "patient_id": artifact.selection.patient_id,
                "uuid": artifact.selection.gdc_file_uuid,
                "manifest": str(artifact.path),
                "manifest_sha256": artifact.sha256,
                "current_scope": "Q75_DOWNLOAD_AND_HEADER_ONLY",
                "scale_policy_status": "NOT_AUTHORIZED",
                "training_status": "NOT_AUTHORIZED",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
