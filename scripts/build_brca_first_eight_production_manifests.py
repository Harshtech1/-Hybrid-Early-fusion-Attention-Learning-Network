#!/usr/bin/env python3
"""Build exact request-only one-row GDC manifests for production rows 1-8."""

from __future__ import annotations

import csv
import hashlib
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "reports/brca_first_eight_canary_proposal.tsv"
SOURCE_SHA256 = "940b8fd1f7d194c2c9b7c69ddae58ffff3c55196b6841ac859c60cbc01095dfd"
DESTINATION = ROOT / "multiscale_feature_pilot/provenance/brca_first_eight_production_manifests"


def main() -> None:
    payload = SOURCE.read_bytes()
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise RuntimeError("first-eight source TSV drift")
    rows = list(csv.DictReader(payload.decode("utf-8").splitlines(), delimiter="\t"))
    if len(rows) != 8 or [int(row["cohort_index"]) for row in rows] != list(range(1, 9)):
        raise RuntimeError("first production block must be exact rows 1-8")
    DESTINATION.mkdir(mode=0o700, parents=False, exist_ok=False)
    for row in rows:
        index = int(row["cohort_index"])
        name = f"P{index:04d}_{row['gdc_uuid']}.REQUEST_ONLY.gdc.tsv"
        manifest = (
            "id\tfilename\tmd5\tsize\tstate\n"
            f"{row['gdc_uuid']}\t{row['filename']}\t{row['md5']}\t{row['size_bytes']}\t{row['state']}\n"
        ).encode("utf-8")
        descriptor = os.open(DESTINATION / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            os.write(descriptor, manifest)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


if __name__ == "__main__":
    main()
