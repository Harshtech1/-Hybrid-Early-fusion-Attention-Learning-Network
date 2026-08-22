from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_brca_894_healnet_training.py"
RUNTIME = ROOT / "multiscale_feature_pilot/src/brca_healnet_training_runtime.py"


def test_runner_is_double_locked_and_first_statement_is_gate() -> None:
    tree = ast.parse(RUNNER.read_text(encoding="utf-8"))
    assignments = {
        node.targets[0].id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert ast.literal_eval(assignments["EXECUTION_AUTHORIZED"]) is False
    assert "PENDING" in ast.literal_eval(assignments["AUTHORIZATION_SHA256"])
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    first = main.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name) and first.value.func.id == "_require_execution_authorized"


def test_real_cli_fails_before_argument_path_torch_or_cuda_access() -> None:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, str(RUNNER)], cwd=ROOT, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment, timeout=30,
    )
    assert result.returncode == 2
    assert "no paths, Torch, CUDA, features or model were accessed" in result.stderr
    assert "usage:" not in result.stderr


def test_runtime_import_surface_is_stdlib_only_and_training_is_lazy() -> None:
    tree = ast.parse(RUNTIME.read_text(encoding="utf-8"))
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            top_imports.append((node.module or "").split(".")[0])
    assert "torch" not in top_imports
    source = RUNTIME.read_text(encoding="utf-8")
    assert "def execute_authorized_training" in source
    assert "torch.cuda.is_available()" in source
    assert "CPU fallback is prohibited" in source
    assert "gradient_accumulation_patients" in source
    assert "locked_test_evaluations_this_run\": 1" in source
    assert "begin_locked_test(paths.result_root, identity)" in source
    assert "publish_training_result(paths.result_root, identity, summary)" in source


def test_runtime_binds_four_modalities_and_pinned_architecture() -> None:
    source = RUNTIME.read_text(encoding="utf-8")
    for fragment in (
        "n_modalities=4", "channel_dims=[2048, 1558, 21, 1333]",
        "num_spatial_axes=[1, 1, 1, 1]", "out_dims=4", "depth=2",
        "l_c=17", "l_d=126", "x_heads=1", "l_heads=8",
        "cross_dim_head=63", "latent_dim_head=20",
        "weights=None, alpha=0.4, eps=1e-7",
    ):
        assert fragment in source
