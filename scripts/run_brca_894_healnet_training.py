#!/usr/bin/env python3
"""Fail-closed launcher for the frozen BRCA HEALNet training protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import sys


EXECUTION_AUTHORIZED = False
AUTHORIZATION_SHA256 = "PENDING_SEPARATE_TRAINING_AUTHORIZATION"
AUTHORIZATION_PATH = "multiscale_feature_pilot/config/brca_894_healnet_training_execution_authorization.yaml"
PROTOCOL_SHA256 = "4bdbc0e92f3db35b115d030a6b2341f5c1c5c391b505a2a50339b1d855b687a8"


class TrainingGateBlocked(RuntimeError):
    pass


def _require_execution_authorized() -> None:
    if not EXECUTION_AUTHORIZED:
        raise TrainingGateBlocked(
            "BLOCKED: BRCA HEALNet training is not authorized; no paths, Torch, CUDA, features or model were accessed"
        )
    if len(AUTHORIZATION_SHA256) != 64 or any(character not in "0123456789abcdef" for character in AUTHORIZATION_SHA256):
        raise TrainingGateBlocked("BLOCKED: pinned training authorization SHA256 is absent")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    _require_execution_authorized()  # Must remain the first operational statement.
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--feature-registry", required=True)
    parser.add_argument("--checkpoint-root", required=True)
    parser.add_argument("--result-root", required=True)
    arguments = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    authorization = root / AUTHORIZATION_PATH
    if _sha256(authorization) != AUTHORIZATION_SHA256:
        raise TrainingGateBlocked("training authorization bytes drift")
    protocol = root / "multiscale_feature_pilot/config/brca_survival_evaluation_protocol.yaml"
    if _sha256(protocol) != PROTOCOL_SHA256:
        raise TrainingGateBlocked("final training protocol bytes drift")

    # Lazy-load only after the authorization and policy byte gates pass.
    sys.path.insert(0, str(root))
    runtime = importlib.import_module("multiscale_feature_pilot.src.brca_healnet_training_runtime")

    from multiscale_feature_pilot.src.brca_training_checkpoint import TrainingRunIdentity
    registry = Path(arguments.feature_registry).resolve(strict=True)
    identity = TrainingRunIdentity(
        training_run_id=arguments.training_run_id,
        source_commit=arguments.expected_source_commit,
        authorization_sha256=AUTHORIZATION_SHA256,
        split_manifest_sha256="3e519a26eaa24852862bf368a48cceffaf26783c73e80007737134ae6ed626ad",
        cutpoints_sha256="77b514098387f883a7bc5205ef191ff321aff94e6c874442f7ccc65ff9059d6d",
        feature_registry_sha256=_sha256(registry),
        training_policy_sha256=PROTOCOL_SHA256,
        official_healnet_commit="28ba5da6ab99fd8069972c22e986d83edb658dd4",
    )
    paths = runtime.TrainingRuntimePaths(
        project_root=root,
        official_healnet_root=root.parent / "Author_Official_Repo_directery/healnet",
        omic_archive=root.parent / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip",
        split_manifest=root / "multiscale_feature_pilot/provenance/brca_894_survival_split.tsv",
        cutpoints=root / "multiscale_feature_pilot/provenance/brca_894_survival_cutpoints.json",
        feature_registry=registry,
        checkpoint_root=Path(arguments.checkpoint_root).resolve(strict=True),
        result_root=Path(arguments.result_root).resolve(strict=True),
    )
    result = runtime.execute_authorized_training(
        paths, identity=identity, expected_project_commit=arguments.expected_source_commit,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TrainingGateBlocked as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(2)
