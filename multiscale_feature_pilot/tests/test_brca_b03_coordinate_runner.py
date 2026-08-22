from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_brca_b03_coordinate_gate.py"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_b03_coordinate_execution_authorization.yaml"


def test_exact_authorization_and_read_tuple() -> None:
    authorization = yaml.safe_load(AUTH.read_text())
    assert authorization["status"] == "AUTHORIZED_B03_SINGLE_MASK_READ_AND_COORDINATE_PUBLICATION"
    assert authorization["authorized_read"] == {
        "openslide_open_count": 1,
        "read_region_count": 1,
        "level_0_location": [0, 0],
        "level": 2,
        "size_at_level": [5831, 3632],
    }
    statement = authorization["approval"]["exact_statement"]
    assert hashlib.sha256(statement.encode()).hexdigest() == authorization["approval"]["exact_statement_sha256"]


def test_runner_contains_one_read_call_and_no_model_surface() -> None:
    source = RUNNER.read_text()
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_region"
    ]
    assert len(calls) == 1
    for token in ("torch", "cuda", "ResNet", "HealNet", "DataLoader"):
        assert token not in source


def test_runner_is_b03_only_and_uses_zero_delete_publication() -> None:
    source = RUNNER.read_text()
    assert "BRCA_BATCH_B03" in source
    for label in ("BRCA_BATCH_B01", "BRCA_BATCH_B02", "BRCA_BATCH_B04", "BRCA_BATCH_B05", "BRCA_BATCH_B06"):
        assert label not in source
    assert "renameat2" in source
    assert "rmtree" not in source
    assert ".unlink(" not in source
    assert "os.remove" not in source


def test_runner_binds_exact_policy_header_and_input_hashes() -> None:
    source = RUNNER.read_text()
    assert "58de5cedb887a3d0faf12c68e8629d26cae4cf7ff8a867e20742d395890a522c" in source
    assert "f0d3ea106e49f24220ded29743b31459caed06d325a22cc123a6ea2addb77111" in source
    assert "4ef4ac79ce3cc0bfc5a4ea62985f080f0f877dca4c7e43191a04a35b2eba8228" in source
    assert "(5831, 3632)" in source
