from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scripts import run_brca_q25_gpu_pilot as runner
from multiscale_feature_pilot.src import brca_q25_feature_artifacts as feature_artifacts


def _auth_document(paths: runner.PilotPaths) -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "BRCA_Q25_GPU_FEATURE_AND_INTERFACE_PILOT",
        "status": "AUTHORIZED_Q25_FEATURE_EXTRACTION_AND_HEALNET_SMOKE_ONLY",
        "cohort": "TCGA-BRCA",
        "candidate": "Q25",
        "approval_evidence": {
            "user_action": "switched to GPU Now !",
            "supervisor_cohort": "Go with brca, blca",
        },
        "corrective_transition": {
            "prior_attempt_status": (
                "BRCA_Q25_GPU_ATTEMPT_1_BLOCKED_RECOVERABLE_CONFIGURATION"
            ),
            "incident_record_path": (
                "multiscale_feature_pilot/provenance/"
                "brca_q25_gpu_attempt_1.yaml"
            ),
            "incident_record_sha256": (
                "2fe5fd4343082bec7bd421c56d039b1bcb50c5097e300d836ab7dd5101579e46"
            ),
            "root_cause": "missing_pre_cuda_CUBLAS_WORKSPACE_CONFIG",
            "scope_expanded": False,
        },
        "approval_scope": {
            "allowed": sorted(runner._ALLOWED_OPERATIONS),
            "prohibited": sorted(runner._PROHIBITED_OPERATIONS),
        },
        "execution_contract": {
            "batch_size": 32,
            "num_workers": 2,
            "seed": 0,
            "device": "cuda:0",
            "dtype": "float32",
            "automatic_mixed_precision": False,
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": ":4096:8",
            "pre_extraction_synthetic_interface_smoke": True,
            "branch_order": ["scale_2x", "scale_4x"],
            "combined_operation": "torch.cat",
            "combined_dim": 0,
            "natural_model_shape": [1, 9322, 2048],
            "no_transpose": True,
        },
        "bound_inputs": runner._expected_bound_inputs(paths),
        "bound_sources": {
            f"source_{index}": {"path": path.as_posix(), "sha256": digest}
            for index, (path, digest) in enumerate(runner.BOUND_FILES)
        },
        "required_stop": {
            "after": "VALIDATED_Q25_FEATURE_ARTIFACTS_AND_HEALNET_SMOKE",
            "training_authorized": False,
        },
    }


def test_authorization_semantics_are_exact_and_keep_training_locked() -> None:
    paths = runner.PilotPaths()
    document = _auth_document(paths)
    runner._validate_authorization_semantics(document, paths)

    wrong_orientation = yaml.safe_load(yaml.safe_dump(document))
    wrong_orientation["execution_contract"]["no_transpose"] = False
    with pytest.raises(runner.Q25GpuPilotError, match="execution contract"):
        runner._validate_authorization_semantics(wrong_orientation, paths)

    training = yaml.safe_load(yaml.safe_dump(document))
    training["approval_scope"]["prohibited"].remove("model_training")
    with pytest.raises(runner.Q25GpuPilotError, match="prohibited operation"):
        runner._validate_authorization_semantics(training, paths)

    q50 = yaml.safe_load(yaml.safe_dump(document))
    q50["approval_scope"]["allowed"].append("q50_feature_extraction")
    with pytest.raises(runner.Q25GpuPilotError, match="authorized operation"):
        runner._validate_authorization_semantics(q50, paths)


def test_finalized_authorization_and_source_hashes_match_worktree() -> None:
    paths = runner.PilotPaths()
    auth_bytes = paths.auth.read_bytes()
    assert hashlib.sha256(auth_bytes).hexdigest() == runner.AUTH_SHA256
    observed = runner._validate_bound_sources(paths)
    assert observed[runner.AUTH_RELATIVE_PATH.as_posix()] == runner.AUTH_SHA256
    assert set(observed) == {
        runner.AUTH_RELATIVE_PATH.as_posix(),
        *(path.as_posix() for path, _digest in runner.BOUND_FILES),
    }


def _empty_dependencies(pixel_events: list[str]) -> runner.PilotDependencies:
    def pixel_capable_factory(_path: Path, _spec: object, _identity: object) -> object:
        pixel_events.append("dataset_constructed")
        return object()

    return runner.PilotDependencies(
        coordinate_validator=lambda *args, **kwargs: object(),
        branch_spec_loader=lambda _record: object(),
        wsi_identity_capture=lambda _path: object(),
        dataset_factory=pixel_capable_factory,
        model_builder=lambda _path: object(),
        feature_extractor=lambda *args, **kwargs: object(),
        omic_loader=lambda *args, **kwargs: object(),
        smoke_runner=lambda **kwargs: object(),
        slide_factory=lambda _path: object(),
        artifact_publisher=lambda *args, **kwargs: object(),
    )


def test_authorization_failure_occurs_before_any_pixel_capable_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixel_events: list[str] = []
    monkeypatch.setattr(runner, "_validate_paths", lambda _paths: None)
    monkeypatch.setattr(
        runner,
        "_validate_bound_sources",
        lambda _paths: (_ for _ in ()).throw(runner.Q25GpuPilotError("auth denied")),
    )

    with pytest.raises(runner.Q25GpuPilotError, match="auth denied"):
        runner.run_q25_gpu_pilot(dependencies=_empty_dependencies(pixel_events))
    assert pixel_events == []


def test_wsi_hash_failure_occurs_before_any_pixel_capable_dataset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixel_events: list[str] = []
    monkeypatch.setattr(runner, "_validate_paths", lambda _paths: None)
    monkeypatch.setattr(runner, "_validate_bound_sources", lambda _paths: {})
    monkeypatch.setattr(
        runner,
        "_repository_snapshot",
        lambda _paths: {"source_head": "a" * 40},
    )
    monkeypatch.setattr(
        runner,
        "_verify_wsi",
        lambda _path: (_ for _ in ()).throw(runner.Q25GpuPilotError("WSI SHA-256 drift")),
    )

    with pytest.raises(runner.Q25GpuPilotError, match="WSI SHA-256 drift"):
        runner.run_q25_gpu_pilot(dependencies=_empty_dependencies(pixel_events))
    assert pixel_events == []


class _FakeDataset:
    def __init__(self, branch: str, events: list[str]) -> None:
        self.branch = branch
        self.events = events
        self.closed = 0

    def close(self) -> None:
        self.closed += 1
        self.events.append(f"close:{self.branch}")


class _FakeCoordinateSet:
    def __init__(self, manifest: Path, records: dict[str, object]) -> None:
        self.manifest_path = manifest
        self.manifest_sha256 = runner.COORDINATE_MANIFEST_SHA256
        self.directory = manifest.parent
        self._records = records

    def branch_for(self, branch: str) -> object:
        return self._records[branch]


def _coordinate_digest(coordinates: torch.Tensor) -> str:
    return hashlib.sha256(
        coordinates.numpy().astype("<i8", copy=False).tobytes(order="C")
    ).hexdigest()


def _small_run_setup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    fail_first_extraction: bool = False,
) -> tuple[
    runner.PilotPaths,
    runner.PilotDependencies,
    list[str],
    list[_FakeDataset],
    dict[str, torch.Tensor],
]:
    # Shrink only the synthetic test contract.  The production constants remain
    # 7,404 + 1,918 rows and width 2,048.
    coordinates_2x = torch.tensor([[0, 0], [512, 0]], dtype=torch.int64)
    coordinates_4x = torch.tensor([[0, 0]], dtype=torch.int64)
    digest_2x = _coordinate_digest(coordinates_2x)
    digest_4x = _coordinate_digest(coordinates_4x)
    expected = {
        "scale_2x": dict(runner.EXPECTED_COORDINATE_BRANCHES["scale_2x"]),
        "scale_4x": dict(runner.EXPECTED_COORDINATE_BRANCHES["scale_4x"]),
    }
    expected["scale_2x"].update(count=2, coordinates_sha256=digest_2x)
    expected["scale_4x"].update(count=1, coordinates_sha256=digest_4x)
    monkeypatch.setattr(runner, "EXPECTED_COORDINATE_BRANCHES", expected)
    monkeypatch.setattr(runner, "EXPECTED_TOTAL_PATCHES", 3)
    monkeypatch.setattr(runner, "FEATURE_DIM", 4)
    monkeypatch.setattr(runner, "DEVICE", "cpu")
    monkeypatch.setattr(runner, "AUTH_SHA256", "a" * 64)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(feature_artifacts, "SCALE_2X_ROWS", 2)
    monkeypatch.setattr(feature_artifacts, "SCALE_4X_ROWS", 1)
    monkeypatch.setattr(feature_artifacts, "COMBINED_ROWS", 3)
    monkeypatch.setattr(feature_artifacts, "FEATURE_DIM", 4)

    data = tmp_path / "data"
    data.mkdir()
    wsi = data / runner.EXPECTED_FILENAME
    wsi.write_bytes(b"mock-wsi-never-opened")
    manifest = data / "coordinate_manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    checkpoint = data / runner.CHECKPOINT_PATH.name
    checkpoint.write_bytes(b"mock-checkpoint")
    omic_path = data / "mock-omic.zip"
    omic_path.write_bytes(b"mock-omic")
    auth = data / "auth.yaml"
    auth.write_text("mock: true\n", encoding="utf-8")
    paths = runner.PilotPaths(
        repo_root=tmp_path,
        official_repo=tmp_path / "official",
        wsi=wsi,
        coordinates=data,
        omic=omic_path,
        checkpoint=checkpoint,
        output=data / "Q25.features",
        auth=auth,
    )

    def branch_record(branch: str, digest: str, count: int) -> object:
        branch_expected = expected[branch]
        return SimpleNamespace(
            branch=branch,
            path=data / str(branch_expected["filename"]),
            size_bytes=int(branch_expected["size_bytes"]),
            sha256=str(branch_expected["sha256"]),
            coordinates_sha256=digest,
            coordinate_count=count,
            metadata=SimpleNamespace(
                source_level=int(branch_expected["source_level"]),
                effective_mpp=branch_expected["effective_mpp"],
            ),
        )

    records = {
        "scale_2x": branch_record("scale_2x", digest_2x, 2),
        "scale_4x": branch_record("scale_4x", digest_4x, 1),
    }
    coordinate_set = _FakeCoordinateSet(manifest, records)
    specs = {
        "scale_2x": SimpleNamespace(
            branch="scale_2x",
            coordinates=coordinates_2x,
            source_level=0,
            effective_mpp=(0.505, 0.505),
        ),
        "scale_4x": SimpleNamespace(
            branch="scale_4x",
            coordinates=coordinates_4x,
            source_level=1,
            effective_mpp=(1.0100149842739303, 1.0100149842739303),
        ),
    }
    events: list[str] = []
    datasets: list[_FakeDataset] = []
    observed: dict[str, torch.Tensor] = {}

    def dataset_factory(_path: Path, spec: object, _identity: object) -> _FakeDataset:
        events.append(f"dataset:{spec.branch}")
        dataset = _FakeDataset(spec.branch, events)
        datasets.append(dataset)
        return dataset

    feature_values = {
        "scale_2x": torch.arange(8, dtype=torch.float32).reshape(2, 4),
        "scale_4x": torch.tensor([[100.0, 101.0, 102.0, 103.0]], dtype=torch.float32),
    }

    def extractor(dataset: _FakeDataset, _model: object, **kwargs: object) -> object:
        events.append(f"extract:{dataset.branch}")
        assert kwargs["batch_size"] == 32
        assert kwargs["num_workers"] == 2
        assert str(kwargs["device"]) == "cpu"
        if fail_first_extraction and dataset.branch == "scale_2x":
            raise RuntimeError("synthetic first-branch failure")
        return SimpleNamespace(
            features=feature_values[dataset.branch],
            streaming_extraction_seconds=0.01,
            model_forward_seconds=0.005,
            peak_gpu_memory_bytes=123,
            batch_size=32,
        )

    omic = SimpleNamespace(
        source_row_index="956",
        case_id=runner.EXPECTED_PATIENT_ID,
        slide_id=runner.EXPECTED_FILENAME,
        rna=torch.zeros((1, 1, 2), dtype=torch.float32),
        mutation=torch.zeros((1, 1, 1), dtype=torch.float32),
        cnv=torch.zeros((1, 1, 3), dtype=torch.float32),
    )

    smoke_calls = 0

    def smoke(**kwargs: object) -> dict[str, object]:
        nonlocal smoke_calls
        smoke_calls += 1
        event = "pre_extraction_smoke" if smoke_calls == 1 else "smoke"
        events.append(event)
        wsi_input = kwargs["wsi"]
        assert isinstance(wsi_input, torch.Tensor)
        if smoke_calls == 2:
            observed["wsi"] = wsi_input.detach().cpu().clone()
        return {
            "training": False,
            "patch_count": 3,
            "wsi_layout": "[1,P,2048]",
            "output_shape": (1, 4),
            "input_shapes": (
                (1, 3, 4),
                (1, 1, 2),
                (1, 1, 1),
                (1, 1, 3),
            ),
            "attention_shapes": (
                (1, 2, 3),
                (1, 2, 1),
                (1, 2, 1),
                (1, 2, 1),
            ),
            "attention_finite": True,
            "output_finite": True,
        }

    dependencies = runner.PilotDependencies(
        coordinate_validator=lambda *args, **kwargs: coordinate_set,
        branch_spec_loader=lambda record: specs[record.branch],
        wsi_identity_capture=lambda _path: SimpleNamespace(
            st_dev=wsi_stat.st_dev,
            st_ino=wsi_stat.st_ino,
            st_size=wsi_stat.st_size,
            st_mtime_ns=wsi_stat.st_mtime_ns,
        ),
        dataset_factory=dataset_factory,
        model_builder=lambda _path: events.append("model") or object(),
        feature_extractor=extractor,
        omic_loader=lambda *args, **kwargs: omic,
        smoke_runner=smoke,
        slide_factory=lambda _path: (_ for _ in ()).throw(AssertionError("pixels/header mocked")),
        artifact_publisher=feature_artifacts.publish_brca_q25_feature_artifacts,
    )

    repository_snapshot = {
        "source_head": "b" * 40,
        "source_branch": "test",
        "source_status_porcelain_v1_z": "",
        "official_head": runner.OFFICIAL_HEAD,
        "official_status_porcelain_v1_z": "",
        "frozen_tag": runner.FROZEN_TAG,
        "frozen_commit": runner.FROZEN_COMMIT,
        "critical_execution_source_sha256": {
            "scripts/run_brca_q25_gpu_pilot.py": "c" * 64
        },
    }
    wsi_stat = wsi.lstat()
    token = (wsi_stat.st_dev, wsi_stat.st_ino, wsi_stat.st_size, wsi_stat.st_mtime_ns)
    monkeypatch.setattr(runner, "_validate_paths", lambda _paths: None)
    monkeypatch.setattr(runner, "_validate_bound_sources", lambda _paths: {"auth": "a" * 64})
    monkeypatch.setattr(runner, "_repository_snapshot", lambda _paths: dict(repository_snapshot))
    monkeypatch.setattr(
        runner,
        "_verify_wsi",
        lambda _path: (
            {
                "path": str(wsi),
                "size_bytes": runner.EXPECTED_SIZE_BYTES,
                "md5": runner.EXPECTED_MD5,
                "sha256": runner.EXPECTED_SHA256,
            },
            token,
        ),
    )
    def verified_regular(path: Path, **kwargs: object):
        metadata = path.lstat()
        return (
            {
                "path": str(path),
                "size_bytes": kwargs["size_bytes"],
                "sha256": kwargs["sha256"],
                "regular_non_symlink": True,
            },
            (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns),
        )

    monkeypatch.setattr(runner, "_verify_regular_sha256", verified_regular)
    monkeypatch.setattr(
        runner,
        "_validate_coordinate_record",
        lambda _record: {
            "directory": str(data),
            "manifest_sha256": runner.COORDINATE_MANIFEST_SHA256,
        },
    )
    monkeypatch.setattr(
        runner,
        "_verify_omic",
        lambda _paths, _loader: (
            omic,
            {
                "path": str(omic_path),
                "size_bytes": runner.OMIC_SIZE_BYTES,
                "sha256": runner.BRCA_RELEASE_ARCHIVE_SHA256,
                "regular_non_symlink": True,
                "source_row_index": "956",
                "shapes": {"rna": [1, 1, 2], "mutation": [1, 1, 1], "cnv": [1, 1, 3]},
            },
            (
                omic_path.lstat().st_dev,
                omic_path.lstat().st_ino,
                omic_path.lstat().st_size,
                omic_path.lstat().st_mtime_ns,
            ),
        ),
    )
    monkeypatch.setattr(
        runner,
        "_verify_slide_header",
        lambda _path, _factory: {"header_only_open_count": 1, "patch_pixel_reads": 0},
    )
    monkeypatch.setattr(runner, "_configure_determinism", lambda _device: {"seed": 0})
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda _device: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda _device=None: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device=None: 456)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: events.append("empty_cache"))

    return paths, dependencies, events, datasets, observed


def test_mocked_gpu_run_preserves_branch_order_no_transpose_and_real_artifact_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, dependencies, events, datasets, observed = _small_run_setup(monkeypatch, tmp_path)

    result = runner.run_q25_gpu_pilot(
        paths=paths,
        dependencies=dependencies,
        gpu_preflight=lambda: (
            torch.device("cpu"),
            {"device": "cpu", "name": "mocked Tesla T4", "cpu_fallback": False},
        ),
    )

    assert result["status"] == runner.SUCCESS_STATUS
    assert events.index("pre_extraction_smoke") < events.index("model")
    assert events.index("pre_extraction_smoke") < events.index("dataset:scale_2x")
    assert events.index("extract:scale_2x") < events.index("dataset:scale_4x")
    assert events.index("extract:scale_4x") < events.index("smoke")
    assert all(dataset.closed >= 1 for dataset in datasets)
    expected_combined = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [100.0, 101.0, 102.0, 103.0],
        ]
    )
    assert tuple(observed["wsi"].shape) == (1, 3, 4)
    assert torch.equal(observed["wsi"][0], expected_combined)
    assert result["features"]["combined"]["branch_order"] == ["scale_2x", "scale_4x"]
    assert result["features"]["combined"]["transpose_performed"] is False
    assert result["healnet_smoke"]["output_shape"] == [1, 4]
    assert result["pre_extraction_healnet_contract"]["output_shape"] == [1, 4]
    assert result["operations"]["healnet_interface_smokes"] == 2
    assert result["operations"]["training"] == 0
    assert result["operations"]["q50_q75_operations"] == 0
    assert paths.output.is_dir()
    assert {path.name for path in paths.output.iterdir()} == {
        "scale_2x_features.pt",
        "scale_4x_features.pt",
        "combined_features.pt",
        "row_provenance.csv",
        "feature_manifest.json",
        "feature_manifest.json.sha256",
    }
    assert feature_artifacts.validate_brca_q25_feature_artifacts(
        paths.output,
        expected_manifest_sha256=result["feature_artifacts"]["manifest_sha256"],
    ).manifest_sha256 == result["feature_artifacts"]["manifest_sha256"]


def test_first_branch_failure_closes_dataset_clears_cache_and_never_builds_4x_or_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, dependencies, events, datasets, _observed = _small_run_setup(
        monkeypatch,
        tmp_path,
        fail_first_extraction=True,
    )

    with pytest.raises(RuntimeError, match="synthetic first-branch failure"):
        runner.run_q25_gpu_pilot(
            paths=paths,
            dependencies=dependencies,
            gpu_preflight=lambda: (
                torch.device("cpu"),
                {"device": "cpu", "name": "mocked Tesla T4", "cpu_fallback": False},
            ),
        )

    assert "dataset:scale_2x" in events
    assert "extract:scale_2x" in events
    assert "dataset:scale_4x" not in events
    assert "extract:scale_4x" not in events
    assert "empty_cache" in events
    assert datasets[0].closed >= 1
    assert not paths.output.exists()


def test_geometry_drift_is_rejected_before_provenance_can_reach_artifacts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "EXPECTED_COORDINATE_BRANCHES",
        {
            "scale_2x": {**runner.EXPECTED_COORDINATE_BRANCHES["scale_2x"], "count": 1},
            "scale_4x": {**runner.EXPECTED_COORDINATE_BRANCHES["scale_4x"], "count": 1},
        },
    )
    monkeypatch.setattr(runner, "EXPECTED_TOTAL_PATCHES", 2)
    bad_2x = SimpleNamespace(
        coordinates=torch.tensor([[0, 0]], dtype=torch.int64),
        source_level=1,
        effective_mpp=(0.505, 0.505),
    )
    good_4x = SimpleNamespace(
        coordinates=torch.tensor([[0, 0]], dtype=torch.int64),
        source_level=1,
        effective_mpp=(1.0100149842739303, 1.0100149842739303),
    )
    with pytest.raises(runner.Q25GpuPilotError, match="source level drift"):
        runner._build_row_provenance(bad_2x, good_4x)


def test_cli_has_one_execution_switch_and_strict_json_rejects_nan() -> None:
    parser_source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "--execute-authorized-q25-gpu-pilot" in parser_source
    assert "--wsi" not in parser_source
    assert "--output-dir" not in parser_source
    with pytest.raises(ValueError):
        runner._strict_json({"bad": float("nan")})
    assert json.loads(runner._strict_json({"ok": True})) == {"ok": True}


def test_missing_cublas_workspace_config_fails_before_gpu_preflight_or_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, dependencies, events, _datasets, _observed = _small_run_setup(
        monkeypatch, tmp_path
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    gpu_calls: list[str] = []

    with pytest.raises(runner.Q25GpuPilotError, match="CUBLAS_WORKSPACE_CONFIG"):
        runner.run_q25_gpu_pilot(
            paths=paths,
            dependencies=dependencies,
            gpu_preflight=lambda: (
                gpu_calls.append("gpu") or torch.device("cpu"),
                {"device": "cpu"},
            ),
        )

    assert gpu_calls == []
    assert not any(event.startswith("dataset:") for event in events)
    assert not paths.output.exists()


def test_pre_extraction_interface_failure_occurs_before_model_or_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths, dependencies, events, _datasets, _observed = _small_run_setup(
        monkeypatch, tmp_path
    )

    def blocked_smoke(**_kwargs: object) -> object:
        events.append("pre_extraction_smoke_failed")
        raise RuntimeError("synthetic deterministic attention failure")

    dependencies = replace(dependencies, smoke_runner=blocked_smoke)
    with pytest.raises(RuntimeError, match="deterministic attention failure"):
        runner.run_q25_gpu_pilot(
            paths=paths,
            dependencies=dependencies,
            gpu_preflight=lambda: (
                torch.device("cpu"),
                {"device": "cpu", "name": "mocked Tesla T4", "cpu_fallback": False},
            ),
        )

    assert "pre_extraction_smoke_failed" in events
    assert "model" not in events
    assert not any(event.startswith("dataset:") for event in events)
    assert not paths.output.exists()


def test_preinitialized_cuda_is_rejected_at_process_environment_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    with pytest.raises(runner.Q25GpuPilotError, match="CUDA was initialized"):
        runner._validate_process_environment()
