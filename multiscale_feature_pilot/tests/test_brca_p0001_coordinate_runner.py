from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_brca_p0001_coordinate_gate.py"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_p0001_coordinate_execution_authorization.yaml"
POLICY = ROOT / "multiscale_feature_pilot/config/brca_p0001_scale_coordinate_policy.yaml"


def _authorization() -> dict:
    return yaml.safe_load(AUTH.read_text())


def test_exact_authorization_statement_and_read_tuple() -> None:
    authorization = _authorization()
    assert authorization["status"] == "AUTHORIZED_P0001_SINGLE_MASK_READ_AND_COORDINATE_PUBLICATION"
    assert authorization["authorized_read"] == {
        "openslide_open_count": 1,
        "read_region_count": 1,
        "level_0_location": [0, 0],
        "level": 2,
        "size_at_level": [5968, 5120],
    }
    statement = authorization["approval"]["exact_statement"]
    assert len(statement) == 515
    assert hashlib.sha256(statement.encode()).hexdigest() == "e04c4424a75f88327592d083029880cdfeab4f2cd2f82a54439f82cbfd1aa43f"
    assert authorization["approval"]["exact_statement_sha256"] == hashlib.sha256(statement.encode()).hexdigest()


def test_authorization_is_p0001_only_and_all_prohibitions_remain_false() -> None:
    authorization = _authorization()
    assert authorization["identity"]["patient_id"] == "TCGA-3C-AALK"
    assert authorization["identity"]["gdc_uuid"] == "93b26333-5723-4fa4-a4de-6124c04ab243"
    assert authorization["authorized_outputs"]["destination"].endswith("/BRCA_PRODUCTION_P0001.coordinates")
    assert authorization["authorized_outputs"]["branches"] == ["scale_2x", "scale_4x"]
    assert authorization["authority"]
    assert not any(authorization["authority"].values())


def test_runner_has_exactly_one_read_region_source_call() -> None:
    tree = ast.parse(RUNNER.read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_region"
    ]
    assert len(calls) == 1
    call = calls[0]
    assert ast.unparse(call.args[0]) == "(0, 0)"
    assert ast.literal_eval(call.args[1]) == 2


def test_runner_has_no_gpu_model_patch_or_training_surface() -> None:
    tree = ast.parse(RUNNER.read_text())
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert "torch" not in imports
    forbidden_attributes = {
        "cuda",
        "backward",
        "step",
        "read_associated_image",
        "get_thumbnail",
    }
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert not (attributes & forbidden_attributes)
    source = RUNNER.read_text()
    for token in ("ResNet50", "HealNet", "DataLoader", "optimizer", "read patches"):
        assert token not in source


def test_runner_uses_zero_delete_atomic_no_overwrite_publication() -> None:
    source = RUNNER.read_text()
    assert "renameat2" in source
    assert "RENAME_NOREPLACE" not in source  # numeric flag 1 is passed directly
    for token in ("rmtree", ".unlink(", "os.remove", "shutil", "rm -"):
        assert token not in source


def test_runner_is_production_p0001_only() -> None:
    source = RUNNER.read_text()
    assert "BRCA_PRODUCTION_P0001" in source
    for label in (
        "BRCA_BATCH_P0001",
        "BRCA_PRODUCTION_P0002",
        "BRCA_PRODUCTION_P0003",
        "BRCA_BATCH_B01",
        "BRCA_BATCH_B02",
        "BRCA_BATCH_B03",
        "BRCA_BATCH_B06",
        "Q25.coordinates",
        "Q50.coordinates",
        "Q75.coordinates",
    ):
        assert label not in source


def test_runner_binds_exact_policy_header_input_and_omic_hashes() -> None:
    source = RUNNER.read_text()
    for expected in (
        "d5ddfdf38f98a921a0876e71ed57fbf196407cf8bbf234433f3c0c0a46513cd4",
        "aefd0e5de9b41ce726e880b14d5002d4012acbffaed7196ba301b28c323da77c",
        "6c7faa0c4e80f4649d784140b907280c1f5b889f5153b4550da9e3e2f198efb3",
        "f43597a87463d8d15007918dd5174ff966aa28dcb0de71cdc5752576cd7c2b5b",
        "1894e15a5dbba2559c61e8521394599153a0ada90cf482fd9eb0c45347f5082a",
        "33767ff31d3c7c11a69ba46c746125f679492e03f5dec7c48f8117aa2a6b3c52",
        "a0ef410e624c698475b78dc0270bf2613e2e320ba4935f8580cb0867d41bfd50",
    ):
        assert expected in source
    assert "MASK = (5968, 5120)" in source
    assert "DIMS = ((95488, 81920), (23872, 20480), (5968, 5120), (2984, 2560))" in source


def test_frozen_policy_matches_runner_geometry_and_theoretical_capacity() -> None:
    policy = yaml.safe_load(POLICY.read_text())
    assert policy["pinned_header"]["level_dimensions"] == [
        [95488, 81920],
        [23872, 20480],
        [5968, 5120],
        [2984, 2560],
    ]
    assert policy["future_mask_policy"]["dimensions"] == [5968, 5120]
    assert policy["branches"]["scale_2x"]["theoretical_sites_before_tissue_filter"] == 29760
    assert policy["branches"]["scale_4x"]["theoretical_sites_before_tissue_filter"] == 7440
    assert policy["branches"]["theoretical_total_sites_before_tissue_filter"] == 37200


def test_runner_rechecks_held_descriptor_and_path_after_final_hash() -> None:
    source = RUNNER.read_text()
    final_hash_position = source.index('"final held WSI hash drift"')
    final_fstat_position = source.index("final = os.fstat(file_descriptor)")
    final_path_position = source.index("current_after_hash = os.stat(WSI, follow_symlinks=False)")
    assert final_hash_position < final_fstat_position < final_path_position
