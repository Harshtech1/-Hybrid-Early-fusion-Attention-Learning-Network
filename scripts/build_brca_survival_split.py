#!/usr/bin/env python3
"""Build only the frozen BRCA survival split and cutpoint metadata."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "multiscale_feature_pilot/src/brca_survival_protocol.py"
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
OMIC = ROOT.parent / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip"
MANIFEST = ROOT / "multiscale_feature_pilot/provenance/brca_894_survival_split.tsv"
CUTPOINTS = ROOT / "multiscale_feature_pilot/provenance/brca_894_survival_cutpoints.json"


def _module():
    spec = importlib.util.spec_from_file_location("brca_survival_protocol_standalone", SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen survival protocol module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alignment", type=Path, default=ALIGNMENT)
    parser.add_argument("--omic-archive", type=Path, default=OMIC)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--cutpoints", type=Path, default=CUTPOINTS)
    args = parser.parse_args()
    protocol = _module()
    records = protocol.load_endpoint_records(args.alignment, args.omic_archive)
    split = protocol.build_frozen_split(records)
    protocol.write_split_artifacts(split, args.manifest, args.cutpoints)
    print(
        json.dumps(
            {
                "status": "BRCA_SURVIVAL_SPLIT_FROZEN",
                "patients": len(split.records),
                "cutpoints_months": [str(value) for value in split.cutpoints_months],
                "manifest_sha256": split.manifest_sha256,
                "cutpoints_sha256": split.cutpoints_sha256,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
