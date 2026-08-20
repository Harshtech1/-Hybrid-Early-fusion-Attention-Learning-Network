#!/usr/bin/env python3
"""Publish only the authorized P0001 ten-event recovery prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiscale_feature_pilot.src.brca_p0001_recovery_prefix_bootstrap import (  # noqa: E402
    AUTHORIZATION_PATH,
    AUTHORIZATION_STATEMENT_SHA256,
    EVIDENCE_HASHES,
    LEDGER_PATH,
    publish_prefix,
)


BOUND = (
    Path("scripts/run_brca_p0001_recovery_prefix_bootstrap.py"),
    AUTHORIZATION_PATH,
    Path("multiscale_feature_pilot/config/brca_first_eight_phase_barrier_policy.yaml"),
    Path("multiscale_feature_pilot/src/brca_p0001_recovery_prefix_bootstrap.py"),
    Path("multiscale_feature_pilot/src/brca_first_eight_phase_barrier.py"),
    Path("multiscale_feature_pilot/src/brca_p0001_feature_package.py"),
    Path("multiscale_feature_pilot/src/brca_singleton_streaming_policy.py"),
    Path("multiscale_feature_pilot/src/brca_streaming_executor_v2.py"),
    Path("multiscale_feature_pilot/src/brca_streaming_recovery_v2.py"),
    Path("multiscale_feature_pilot/src/brca_streaming_production_adapter.py"),
    Path("multiscale_feature_pilot/__init__.py"),
    Path("multiscale_feature_pilot/src/__init__.py"),
    Path("multiscale_feature_pilot/src/multiscale_bag.py"),
    Path("multiscale_feature_pilot/src/omic.py"),
    Path("multiscale_feature_pilot/src/padding.py"),
    Path("multiscale_feature_pilot/src/provenance.py"),
    *(Path(relative) for relative in EVIDENCE_HASHES),
)
ALLOWED_DIRTY = {
    " M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}


def _git(*arguments: str) -> str:
    return subprocess.run(("git", *arguments), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def _validate_repository(expected_commit: str) -> None:
    if len(expected_commit) != 40 or _git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("source commit drift")
    status = set(filter(None, _git("status", "--short").splitlines()))
    if status - ALLOWED_DIRTY:
        raise RuntimeError(f"unexpected Git status: {status - ALLOWED_DIRTY}")
    for relative in BOUND:
        committed = subprocess.run(("git", "show", f"HEAD:{relative.as_posix()}"), cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout
        if (ROOT / relative).read_bytes() != committed:
            raise RuntimeError(f"bound source drift: {relative}")


def _validate_authorization() -> str:
    path = ROOT / AUTHORIZATION_PATH
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if document["status"] != "P0001_RECOVERY_V2_PREFIX_BOOTSTRAP_AUTHORIZED_NOT_EXECUTED" or document["executable"] is not True:
        raise RuntimeError("bootstrap authorization is not executable")
    statement = document["authorization"]["exact_statement"]
    digest = hashlib.sha256(statement.encode()).hexdigest()
    if digest != AUTHORIZATION_STATEMENT_SHA256 or document["authorization"]["exact_statement_sha256"] != digest:
        raise RuntimeError("bootstrap authorization statement drift")
    if not all(document["prohibited"].values()):
        raise RuntimeError("bootstrap prohibited-operation lock drift")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-commit", required=True)
    arguments = parser.parse_args()
    _validate_repository(arguments.expected_source_commit)
    authorization_sha256 = _validate_authorization()
    events = publish_prefix(ROOT, LEDGER_PATH)
    result = {
        "status": "P0001_RECOVERY_PREFIX_VERIFIED",
        "source_commit": arguments.expected_source_commit,
        "authorization_sha256": authorization_sha256,
        "ledger": str(LEDGER_PATH),
        "exact_events": len(events),
        "first_record_sha256": events[0].record_sha256,
        "tip_record_sha256": events[-1].record_sha256,
        "last_durable_stage": events[-1].stage.value,
        "next_stage": "GPU_AUTHORIZED",
        "wsi_or_pixel_operations": 0,
        "gpu_or_cuda_operations": 0,
        "deletion_or_overwrite_operations": 0,
    }
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}, sort_keys=True), file=sys.stderr)
        raise SystemExit(1)
