#!/usr/bin/env python3
"""Run the authorized CPU-only BRCA Q75 exact-file/header gate."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import openslide

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from multiscale_feature_pilot.src.brca_q75_header_gate import (
    DEFAULT_OUTPUT_DIRECTORY,
    GatePaths,
    run_header_gate,
)


def _parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=(
            "Verify the exact BRCA Q75 SVS and Omic row, collect only OpenSlide "
            "header metadata using fixed production paths, publish one atomic "
            "no-replace result directory, and stop."
        )
    )


def main() -> int:
    _parser().parse_args()
    paths = GatePaths()
    record = run_header_gate(paths=paths, slide_factory=openslide.OpenSlide)
    print(f"{record['status']}: {record['wsi']['path']}")
    print(f"Atomic result directory: {DEFAULT_OUTPUT_DIRECTORY}")
    print("STOP: no Q75 pixel access or scale policy is authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
