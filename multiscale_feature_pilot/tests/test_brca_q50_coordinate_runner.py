from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts import run_brca_q50_coordinate_gate as runner
from multiscale_feature_pilot.src.brca_q50_coordinates import (
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
    Q50SlideObservation,
)


class _FakeSlide:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _HeaderDriftSlide(_FakeSlide):
    def __init__(self) -> None:
        super().__init__()
        self.properties = {
            "openslide.mpp-x": "0.3",
            "openslide.mpp-y": "0.2468",
        }
        self.level_dimensions = EXPECTED_LEVEL_DIMENSIONS
        self.level_downsamples = EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES
        self.calls: list[object] = []

    def read_region(self, *args):
        self.calls.append(args)
        raise AssertionError("pixel read must not occur after header drift")


def _observation() -> Q50SlideObservation:
    return Q50SlideObservation(
        patient_id=runner.EXPECTED_PATIENT_ID,
        slide_id=runner.EXPECTED_SLIDE_ID,
        gdc_file_uuid=runner.EXPECTED_GDC_FILE_UUID,
        filename=runner.EXPECTED_FILENAME,
        size_bytes=runner.EXPECTED_SIZE_BYTES,
        md5=runner.EXPECTED_MD5,
        sha256=runner.EXPECTED_SHA256,
        mpp_x=runner.EXPECTED_MPP[0],
        mpp_y=runner.EXPECTED_MPP[1],
        level_dimensions=runner.EXPECTED_LEVEL_DIMENSIONS,
        openslide_level_downsamples=runner.EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
    )


def _paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> runner.GatePaths:
    repo = tmp_path / "pilot"
    official = tmp_path / "official"
    data = tmp_path / "data"
    wsi_parent = data / "incoming"
    for directory in (repo, official, wsi_parent):
        directory.mkdir(parents=True)
    wsi = wsi_parent / runner.EXPECTED_FILENAME
    wsi.write_bytes(b"synthetic-not-a-real-wsi")
    output = data / "Q50.coordinates"
    auth = repo / "auth.yaml"
    auth.write_text("synthetic: true\n", encoding="utf-8")
    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(runner, "OFFICIAL_REPO", official)
    monkeypatch.setattr(runner, "WSI_PATH", wsi)
    monkeypatch.setattr(runner, "OUTPUT_DIR", output)
    return runner.GatePaths(
        repo_root=repo,
        official_repo=official,
        wsi=wsi,
        output=output,
        auth=auth,
    )


def _install_safe_preflight(
    monkeypatch: pytest.MonkeyPatch, paths: runner.GatePaths
) -> dict[str, object]:
    snapshot = {
        "official_head": runner.OFFICIAL_HEAD,
        "official_status": "",
        "frozen_tag": runner.FROZEN_TAG,
        "frozen_commit": runner.FROZEN_COMMIT,
        "source_commit_at_execution": "a" * 40,
        "critical_execution_source_sha256": {"runner": "c" * 64},
        "pilot_status_porcelain_v1_z": "",
    }
    monkeypatch.setattr(
        runner, "_validate_bound_sources", lambda _paths: {"auth": "b" * 64}
    )
    monkeypatch.setattr(
        runner, "_repository_snapshot", lambda _paths: dict(snapshot)
    )
    monkeypatch.setattr(
        runner,
        "_df_snapshot",
        lambda _path: {
            "measurement": "runtime_df_capacity_not_lightning_logical_quota",
            "available_bytes": 2_000_000_000,
        },
    )
    metadata = paths.wsi.lstat()
    token = (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )
    monkeypatch.setattr(
        runner,
        "_verify_wsi",
        lambda _path: (
            {
                "path": str(paths.wsi),
                "size_bytes": runner.EXPECTED_SIZE_BYTES,
                "md5": runner.EXPECTED_MD5,
                "sha256": runner.EXPECTED_SHA256,
                "regular_non_symlink": True,
                "partial_siblings": [],
            },
            token,
        ),
    )
    return snapshot


def _mask_reader(_slide):
    mask = np.zeros((2, 2, 4), dtype=np.uint8)
    return mask, _observation(), 1, hashlib.sha256(mask.tobytes()).hexdigest()


def _coordinate_builder(_mask, *, observation):
    assert observation == _observation()
    return SimpleNamespace(
        scale_2x=np.asarray([[0, 0], [512, 0]], dtype=np.int64),
        scale_4x=np.asarray([[0, 0], [1024, 0]], dtype=np.int64),
        contour_count=2,
        retained_hole_count=1,
        policy_status=runner.POLICY_STATUS,
    )


def test_mocked_gate_publishes_only_q50_coordinates_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch, paths)
    slide = _FakeSlide()
    result = runner.run_coordinate_gate(
        paths=paths,
        slide_factory=lambda _path: slide,
        mask_reader=_mask_reader,
        coordinate_builder=_coordinate_builder,
    )

    assert slide.closed
    assert result["status"] == "BRCA_Q50_COORDINATES_VERIFIED"
    assert result["source_commit_at_execution"] == "a" * 40
    assert result["mask"]["read_region_calls"] == 1
    assert result["coordinate_artifacts"]["scale_2x"]["coordinate_count"] == 2
    assert result["coordinate_artifacts"]["scale_4x"]["coordinate_count"] == 2
    assert result["operations"]["level_0_or_level_1_patch_pixel_reads"] == 0
    assert result["operations"]["resnet50_feature_extraction"] == 0
    assert result["operations"]["healnet_execution"] == 0
    assert result["operations"]["q25_reruns"] == 0
    assert result["operations"]["q75_operations"] == 0
    assert result["operations"]["training"] == 0
    assert result["required_stop_reached"] is True
    assert paths.output.is_dir()


def test_header_drift_stops_before_pixel_read() -> None:
    slide = _HeaderDriftSlide()
    with pytest.raises(Exception, match="mpp_x drift"):
        runner._read_exact_mask(slide)
    assert slide.calls == []


@pytest.mark.parametrize("stale_kind", ["destination", "lock", "staging"])
def test_existing_output_state_fails_before_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stale_kind: str
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    if stale_kind == "destination":
        paths.output.mkdir()
    elif stale_kind == "lock":
        (paths.output.parent / f".{paths.output.name}.lock").write_text("pid=1\n")
    else:
        (paths.output.parent / f".{paths.output.name}.staging.old").mkdir()
    opens: list[str] = []
    with pytest.raises(runner.Q50CoordinateGateError):
        runner.run_coordinate_gate(
            paths=paths, slide_factory=lambda path: opens.append(path)
        )
    assert opens == []


def test_wrong_read_constant_fails_before_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "MASK_LOCATION", (1, 0))
    opens: list[str] = []
    with pytest.raises(runner.Q50CoordinateGateError, match="location constant"):
        runner.run_coordinate_gate(
            paths=paths, slide_factory=lambda path: opens.append(path)
        )
    assert opens == []


def test_mask_reader_failure_closes_slide_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch, paths)
    slide = _FakeSlide()
    publishes: list[object] = []

    def fail_reader(_slide):
        raise RuntimeError("synthetic mask failure")

    with pytest.raises(RuntimeError, match="synthetic mask failure"):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda _path: slide,
            mask_reader=fail_reader,
            coordinate_builder=_coordinate_builder,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )
    assert slide.closed
    assert publishes == []
    assert not paths.output.exists()


def test_repository_drift_after_generation_blocks_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    snapshot = _install_safe_preflight(monkeypatch, paths)
    snapshots = [dict(snapshot), {**snapshot, "source_commit_at_execution": "d" * 40}]
    monkeypatch.setattr(runner, "_repository_snapshot", lambda _paths: snapshots.pop(0))
    publishes: list[object] = []
    with pytest.raises(runner.Q50CoordinateGateError, match="repository state changed"):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda _path: _FakeSlide(),
            mask_reader=_mask_reader,
            coordinate_builder=_coordinate_builder,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )
    assert publishes == []
    assert not paths.output.exists()


def test_authorization_semantics_are_exact_and_q75_training_stay_locked() -> None:
    document = yaml.safe_load(
        (runner.REPO_ROOT / runner.AUTH_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    paths = runner.GatePaths()
    runner._validate_authorization_semantics(document, paths)

    wrong_read = yaml.safe_load(yaml.safe_dump(document))
    wrong_read["authorized_operations"]["mask_pixel_read"]["maximum_calls"] = 2
    with pytest.raises(runner.Q50CoordinateGateError, match="mask-read"):
        runner._validate_authorization_semantics(wrong_read, paths)

    missing_q75_lock = yaml.safe_load(yaml.safe_dump(document))
    missing_q75_lock["explicitly_prohibited"].remove(
        "q75_download_open_or_processing"
    )
    with pytest.raises(runner.Q50CoordinateGateError, match="prohibition"):
        runner._validate_authorization_semantics(missing_q75_lock, paths)

    training_unlock = yaml.safe_load(yaml.safe_dump(document))
    training_unlock["required_stop"]["training_authorized"] = True
    with pytest.raises(runner.Q50CoordinateGateError, match="required-stop"):
        runner._validate_authorization_semantics(training_unlock, paths)


def test_bound_source_hash_drift_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    auth = repo / "auth.yaml"
    policy = repo / "policy.py"
    auth.write_bytes(b"approved\n")
    policy.write_bytes(b"original\n")
    monkeypatch.setattr(runner, "AUTH_RELATIVE_PATH", Path("auth.yaml"))
    monkeypatch.setattr(
        runner, "AUTH_SHA256", hashlib.sha256(auth.read_bytes()).hexdigest()
    )
    monkeypatch.setattr(
        runner,
        "BOUND_FILES",
        ((Path("policy.py"), hashlib.sha256(policy.read_bytes()).hexdigest()),),
    )
    policy.write_bytes(b"drift\n")
    paths = runner.GatePaths(
        repo_root=repo,
        official_repo=tmp_path,
        wsi=tmp_path / "unused.svs",
        output=tmp_path / "unused-output",
        auth=auth,
    )
    with pytest.raises(runner.Q50CoordinateGateError, match="SHA-256 drift"):
        runner._validate_bound_sources(paths)


def test_runner_binds_q50_only_and_imports_no_model_stack() -> None:
    assert runner.AUTH_SHA256 == (
        "9e580e6051ec911ad4366b3bace5f0a0352eb498a6c55359b160c106086e74dd"
    )
    expected = dict(runner.BOUND_FILES)
    assert expected[
        Path("multiscale_feature_pilot/config/brca_q50_coordinate_policy.yaml")
    ] == "e5cb83739d3d8fab04da8a63ae1560df04ccc547a79512c219eecb575c0c2114"
    assert expected[
        Path("multiscale_feature_pilot/src/brca_q50_coordinates.py")
    ] == "7dd739667cb6fe0887f3452127c9ff4d43659831d17096922a9f62685149f892"

    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported_roots.isdisjoint({"torch", "torchvision", "healnet"})
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert source.count(".read_region(") == 1
    assert "read_thumbnail" not in source
    assert "associated_images" not in source


def test_execution_approval_binds_authorization_runner_and_stop_boundary() -> None:
    approval_path = (
        runner.REPO_ROOT
        / "multiscale_feature_pilot/provenance/"
        "brca_q50_coordinate_execution_approval.yaml"
    )
    approval = yaml.safe_load(approval_path.read_text(encoding="utf-8"))
    assert approval["status"] == "Q50_COORDINATE_EXECUTION_AUTHORIZED_CPU_ONLY"
    assert approval["approval"]["user_statement"] == "ok do it start it"
    assert approval["approval"]["authorization_config"]["sha256"] == (
        hashlib.sha256(
            (runner.REPO_ROOT / runner.AUTH_RELATIVE_PATH).read_bytes()
        ).hexdigest()
    )
    assert approval["execution_implementation"]["runner"]["sha256"] == (
        hashlib.sha256(Path(runner.__file__).read_bytes()).hexdigest()
    )
    boundary = approval["execution_boundary"]
    assert boundary["maximum_mask_reads"] == 1
    assert boundary["feature_extraction"] == "prohibited"
    assert boundary["q75"] == "prohibited"
    assert boundary["training"] == "prohibited"
