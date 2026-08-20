#!/usr/bin/env python3
"""Run the authorized CPU-only P0002--P0008 acquisition/header phase."""

import argparse
import json

from multiscale_feature_pilot.src.brca_p0002_p0008_header_package import run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    results = run(expected_source_commit=args.expected_source_commit)
    print(json.dumps({"status": "P0002_P0008_HEADERS_VERIFIED", "patients": len(results)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
