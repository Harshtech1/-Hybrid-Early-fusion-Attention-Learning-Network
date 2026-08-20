from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "scripts/run_brca_b03_gpu_pilot.py"
PREPARATION = ROOT / "multiscale_feature_pilot/config/brca_b03_gpu_preexecution.yaml"
AUTHORIZATION = (
    ROOT / "multiscale_feature_pilot/config/brca_b03_gpu_execution_authorization.yaml"
)
OUTPUT = Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B03.features")


def load_runner():
    spec = importlib.util.spec_from_file_location("b03_locked_gpu_runner", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_preexecution_record_is_locked_and_bound() -> None:
    record = yaml.safe_load(PREPARATION.read_text(encoding="utf-8"))
    assert record["status"] == "B03_GPU_PREEXECUTION_PREPARATION_AUTHORIZED_EXECUTION_LOCKED"
    assert record["executable"] is False
    assert not any(record["execution_lock"].values())
    statement = record["preparation_authority"]["exact_statement"]
    assert hashlib.sha256(statement.encode()).hexdigest() == record["preparation_authority"][
        "exact_statement_sha256"
    ]
    future = record["future_exact_authorization_text"]
    assert hashlib.sha256(future.encode()).hexdigest() == record[
        "future_exact_authorization_text_sha256"
    ]


def test_exact_b03_tensor_checkpoint_and_estimate_contract() -> None:
    record = yaml.safe_load(PREPARATION.read_text(encoding="utf-8"))
    inputs = record["inputs"]
    assert inputs["scale_2x_rows"] == 8875
    assert inputs["scale_4x_rows"] == 2257
    assert inputs["combined_shape"] == [11132, 2048]
    assert inputs["natural_healnet_wsi_shape"] == [1, 11132, 2048]
    assert inputs["checkpoint_sha256"] == (
        "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
    )
    assert record["estimates_not_measurements"]["raw_combined_tensor_bytes"] == (
        11132 * 2048 * 4
    )


def test_runner_authorization_gate_allows_only_the_pinned_package(monkeypatch) -> None:
    runner = load_runner()
    monkeypatch.setattr(runner, "_execute", lambda commit: {"commit": commit})
    assert runner.run("0" * 40) == {"commit": "0" * 40}
    assert runner.EXECUTION_AUTHORIZED is True
    assert runner.EXECUTION_AUTH_SHA256 == hashlib.sha256(
        AUTHORIZATION.read_bytes()
    ).hexdigest()


def test_importing_authorized_runner_does_not_initialize_cuda() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util,sys;"
                f"p={str(RUNNER)!r};"
                "s=importlib.util.spec_from_file_location('b03_import',p);"
                "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
                "assert 'torch' not in sys.modules"
            ),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_runner_source_closure_and_no_training_surface() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "COUNTS = (8875, 2257)" in source
    assert "TOTAL = 11132" in source
    assert "BRCA_BATCH_B03.features" in source
    assert "torch.cat((result_2x.features, result_4x.features), dim=0)" in source
    assert "preserve_failed_staging=True" in source
    assert 'torch.device("cuda:0")' in source
    assert "CUBLAS_WORKSPACE_CONFIG" in source
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not {"backward", "step", "train"}.intersection(calls)
    run_function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run"
    )
    first = run_function.body[0]
    assert isinstance(first, ast.Expr) and isinstance(first.value, ast.Call)
    assert isinstance(first.value.func, ast.Name) and first.value.func.id == "_require_authorized"
    runner = load_runner()
    required = {
        Path("multiscale_feature_pilot/__init__.py"),
        Path("multiscale_feature_pilot/src/__init__.py"),
        Path("multiscale_feature_pilot/src/omic.py"),
        Path("multiscale_feature_pilot/src/padding.py"),
    }
    assert required.issubset(set(runner.BOUND))


def test_executable_authorization_is_exact_and_feature_output_is_absent() -> None:
    assert AUTHORIZATION.is_file()
    assert not AUTHORIZATION.is_symlink()
    authorization = yaml.safe_load(AUTHORIZATION.read_text(encoding="utf-8"))
    assert authorization["status"] == "B03_GPU_FEATURE_PILOT_AUTHORIZED"
    assert authorization["scope"]["combined_shape"] == [11132, 2048]
    assert authorization["scope"]["scale_2x_patch_reads"] == 8875
    assert authorization["scope"]["scale_4x_patch_reads"] == 2257
    assert all(authorization["prohibited"].values())
    assert not OUTPUT.exists()
    assert not OUTPUT.is_symlink()
