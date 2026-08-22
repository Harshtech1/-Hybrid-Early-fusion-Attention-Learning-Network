#!/usr/bin/env python3
"""Build and validate three NOT_AUTHORIZED one-row BRCA GDC manifests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from multiscale_feature_pilot.src.brca_one_row_manifest import (
    AUTHORIZATION_STATUS,
    DEFAULT_ALIGNMENT,
    DEFAULT_OUTPUT_DIRECTORY,
    DEFAULT_PROPOSAL,
    DEFAULT_SOURCE_MANIFEST,
    build_phase2_manifest_set,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--alignment", type=Path, default=DEFAULT_ALIGNMENT)
    parser.add_argument(
        "--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST
    )
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    artifacts = build_phase2_manifest_set(
        output_directory=args.output_directory,
        proposal_path=args.proposal,
        alignment_path=args.alignment,
        source_manifest_path=args.source_manifest,
    )
    payload = {
        "status": AUTHORIZATION_STATUS,
        "metadata_only": True,
        "manifest_count": len(artifacts),
        "manifests": [
            {
                "label": item.selection.label,
                "patient_id": item.selection.patient_id,
                "path": str(item.path),
                "rows": 1,
                "sha256": item.sha256,
            }
            for item in artifacts
        ],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
