#!/usr/bin/env python3
"""Execute the authorized P0002--P0008 CPU coordinate phase."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiscale_feature_pilot.src.brca_p0002_p0008_coordinate_phase import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-commit", required=True)
    result = run(parser.parse_args().expected_source_commit)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        raise SystemExit(1)
