#!/usr/bin/env python3
"""Run the exact authorized BRCA Q75 GDC acquisition and stop."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from multiscale_feature_pilot.src.brca_q75_download_runner import (
    DEFAULT_RESULT_FILE,
    run_q75_download,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download and exactly verify only the authorized BRCA Q75 GDC SVS, "
            "publish one append-only result, and stop before OpenSlide."
        )
    )
    parser.add_argument(
        "--expected-source-commit",
        required=True,
        help=(
            "Explicit full commit that must equal HEAD and contain byte-identical "
            "execution sources. Never derive this argument inside the runner."
        ),
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    result = run_q75_download(expected_source_commit=arguments.expected_source_commit)
    print(f"{result['status']}: {result['wsi']['path']}")
    print(f"Size: {result['wsi']['size_bytes']}")
    print(f"MD5: {result['wsi']['md5']}")
    print(f"SHA256: {result['wsi']['sha256']}")
    print(f"Result: {DEFAULT_RESULT_FILE}")
    print("STOP: OpenSlide and every Q75 pixel/scale/feature operation remain outside this runner.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
