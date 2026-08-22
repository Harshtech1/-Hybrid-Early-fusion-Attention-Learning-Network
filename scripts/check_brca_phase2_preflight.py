#!/usr/bin/env python3
"""CLI for the local, read-only BRCA Phase-2 CPU preflight."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from multiscale_feature_pilot.src.brca_phase2_preflight import (  # noqa: E402
    PreflightPaths,
    default_paths,
    run_preflight,
    to_strict_json,
)


def _parse_args() -> argparse.Namespace:
    defaults = default_paths(REPOSITORY_ROOT)
    parser = argparse.ArgumentParser(
        description=(
            "Validate the BRCA Phase-2 CPU preflight without network access, "
            "WSI opening, extraction, or training."
        )
    )
    parser.add_argument("--authorization", type=Path, default=defaults.authorization)
    parser.add_argument("--metadata-policy", type=Path, default=defaults.metadata_policy)
    parser.add_argument(
        "--manifests-directory", type=Path, default=defaults.manifests_directory
    )
    parser.add_argument(
        "--authorized-q25-directory",
        type=Path,
        default=defaults.authorized_q25_directory,
    )
    parser.add_argument("--gdc-client", type=Path, default=defaults.gdc_client)
    parser.add_argument("--official-repo", type=Path, default=defaults.official_repo)
    parser.add_argument("--pilot-repo", type=Path, default=defaults.pilot_repo)
    parser.add_argument("--staging-root", type=Path, default=defaults.staging_root)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    result = run_preflight(
        PreflightPaths(
            authorization=args.authorization,
            metadata_policy=args.metadata_policy,
            manifests_directory=args.manifests_directory,
            authorized_q25_directory=args.authorized_q25_directory,
            gdc_client=args.gdc_client,
            official_repo=args.official_repo,
            pilot_repo=args.pilot_repo,
            staging_root=args.staging_root,
        )
    )
    print(to_strict_json(result))
    return 0 if result["cpu_preflight_ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
