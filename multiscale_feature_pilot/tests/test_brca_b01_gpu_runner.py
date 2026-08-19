from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_brca_b01_gpu_pilot.py"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_b01_gpu_execution_authorization.yaml"

def test_authorization_exact_scope() -> None:
    payload = yaml.safe_load(AUTH.read_text())
    assert payload["status"] == "B01_GPU_FEATURE_PILOT_AUTHORIZED"
    assert payload["scope"]["scale_2x_patch_reads"] == 3773
    assert payload["scope"]["scale_4x_patch_reads"] == 969
    assert payload["scope"]["combined_shape"] == [4742, 2048]
    assert all(payload["prohibited"].values())
    assert hashlib.sha256(AUTH.read_bytes()).hexdigest() == "4afc085250cb969d9d63c6db60ace06ac834769f2c64c239e017cf6ff861f902"

def test_runner_has_one_patient_and_no_training_calls() -> None:
    source = RUNNER.read_text(); tree = ast.parse(source)
    assert source.count("StreamingOpenSlideDataset(WSI") == 2
    assert "TCGA-GI-A2C8" in source
    assert "B02" not in source and "B06" not in source
    calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    assert not {"backward", "step", "train"}.intersection(calls)

def test_runner_requires_deterministic_gpu_contract() -> None:
    source = RUNNER.read_text()
    for token in ('CUBLAS_WORKSPACE_CONFIG") != ":4096:8"', "torch.use_deterministic_algorithms(True)", "allow_tf32 = False", 'torch.device("cuda:0")', '"Tesla T4"', "preserve_failed_staging=True"):
        assert token in source

def test_protected_first_status_line_survives_trimmed_git_output() -> None:
    source = RUNNER.read_text()
    assert '"M reports/blca_one_patient_multiscale_pilot.md"' in source

def test_both_healnet_smokes_use_keyword_only_interface() -> None:
    source = RUNNER.read_text()
    assert source.count("official_repo=OFFICIAL") == 2
    assert source.count("wsi=") == 2
