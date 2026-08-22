#!/usr/bin/env python3
"""Build the authorized Q25-only GDC manifest without downloading data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from multiscale_feature_pilot.src.brca_q25_authorized_manifest import (  # noqa: E402
    DEFAULT_AUTHORIZATION_CONFIG,
    DEFAULT_GUARDED_DIRECTORY,
    DEFAULT_METADATA_POLICY,
    DEFAULT_OUTPUT_DIRECTORY,
    build_q25_authorized_manifest,
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
        "--metadata-policy", type=Path, default=DEFAULT_METADATA_POLICY
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifact = build_q25_authorized_manifest(
        output_directory=args.output_directory,
        guarded_directory=args.guarded_directory,
        authorization_path=args.authorization,
        metadata_policy_path=args.metadata_policy,
    )
    print(
        json.dumps(
            {
                "status": "AUTHORIZED_Q25_ONLY",
                "network_request_performed": False,
                "download_invoked": False,
                "label": artifact.selection.label,
                "patient_id": artifact.selection.patient_id,
                "uuid": artifact.selection.gdc_file_uuid,
                "manifest": str(artifact.path),
                "manifest_sha256": artifact.sha256,
                "q50_q75_status": "LOCKED_PENDING_Q25_REPORT",
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
