from __future__ import annotations

import ast
import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from scripts import run_brca_q25_coordinate_gate as runner
from multiscale_feature_pilot.src.brca_q25_coordinates import (
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
)


class _FakeSlide:
    def __init__(self, *, mpp: str = "0.2525", fail_read: bool = False) -> None:
        self.properties = {
            "openslide.mpp-x": mpp,
            "openslide.mpp-y": "0.2525",
        }
        self.level_dimensions = EXPECTED_LEVEL_DIMENSIONS
        self.level_downsamples = EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES
        self.fail_read = fail_read
        self.calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
        self.closed = False

    def read_region(self, location, level, size):
        self.calls.append((location, level, size))
        if self.fail_read:
            raise RuntimeError("synthetic read failure")
        if (location, level, size) != ((0, 0), 2, EXPECTED_LEVEL_DIMENSIONS[2]):
            raise RuntimeError("wrong mask read")
        width, height = EXPECTED_LEVEL_DIMENSIONS[2]
        return np.zeros((height, width, 4), dtype=np.uint8)

    def close(self) -> None:
        self.closed = True


def _paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> runner.GatePaths:
    repo = tmp_path / "pilot"
    official = tmp_path / "official"
    data = tmp_path / "data"
    wsi_parent = data / "incoming"
    for directory in (repo, official, wsi_parent):
        directory.mkdir(parents=True)
    wsi = wsi_parent / runner.EXPECTED_FILENAME
    wsi.write_bytes(b"synthetic-not-a-real-wsi")
    output = data / "Q25.coordinates"
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
    monkeypatch: pytest.MonkeyPatch,
    paths: runner.GatePaths,
) -> None:
    snapshot = {
        "official_head": runner.OFFICIAL_HEAD,
        "official_status": "",
        "frozen_tag": runner.FROZEN_TAG,
        "frozen_commit": runner.FROZEN_COMMIT,
        "source_commit_at_execution": "a" * 40,
        "critical_execution_source_sha256": {"runner": "c" * 64},
        "pilot_status_porcelain_v1_z": "",
    }
    monkeypatch.setattr(runner, "_validate_bound_sources", lambda _paths: {"auth": "b" * 64})
    monkeypatch.setattr(runner, "_repository_snapshot", lambda _paths: dict(snapshot))
    monkeypatch.setattr(
        runner,
        "_df_snapshot",
        lambda _path: {
            "measurement": "runtime_df_filesystem_capacity_not_lightning_logical_quota",
            "available_bytes": 1_000_000,
        },
    )
    metadata = paths.wsi.lstat()
    token = (metadata.st_dev, metadata.st_ino, metadata.st_size, metadata.st_mtime_ns)
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
    monkeypatch.setattr(
        runner,
        "build_q25_coordinate_bags",
        lambda _mask, *, observation: SimpleNamespace(
            scale_2x=np.asarray([[0, 0], [512, 0]], dtype=np.int64),
            scale_4x=np.asarray([[0, 0], [1024, 0]], dtype=np.int64),
            contour_count=2,
            retained_hole_count=1,
            policy_status=runner.POLICY_STATUS,
        ),
    )


def test_mocked_gate_reads_only_exact_mask_and_publishes_strict_result(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch, paths)
    slide = _FakeSlide()

    result = runner.run_coordinate_gate(paths=paths, slide_factory=lambda _path: slide)

    assert slide.calls == [((0, 0), 2, EXPECTED_LEVEL_DIMENSIONS[2])]
    assert slide.closed
    assert result["status"] == "BRCA_Q25_COORDINATES_VERIFIED"
    assert result["source_commit_at_execution"] == "a" * 40
    assert result["critical_execution_source_sha256"] == {"runner": "c" * 64}
    assert set(result["runtime_versions"]) == {
        "python_executable",
        "python",
        "numpy",
        "opencv",
        "h5py",
        "openslide_python",
        "openslide_library",
    }
    assert result["mask"]["channels"] == 4
    assert result["mask"]["contour_count"] == 2
    assert result["mask"]["retained_hole_count"] == 1
    assert result["coordinate_artifacts"]["scale_2x"]["coordinate_count"] == 2
    assert result["coordinate_artifacts"]["scale_4x"]["coordinate_count"] == 2
    assert result["operations"]["mask_read_region_calls"] == 1
    assert result["operations"]["level_0_or_level_1_patch_pixel_reads"] == 0
    assert result["operations"]["resnet50_feature_extraction"] == 0
    assert result["operations"]["training"] == 0
    assert result["required_stop_reached"] is True
    assert paths.output.is_dir()


def test_wrong_pixel_read_fails_closed_closes_slide_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch, paths)
    monkeypatch.setattr(runner, "MASK_LOCATION", (1, 0))
    slide = _FakeSlide()
    publishes: list[object] = []

    with pytest.raises(runner.Q25CoordinateGateError, match="location constant drift"):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda _path: slide,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )

    assert slide.calls == []
    assert not slide.closed  # The slide was never opened.
    assert publishes == []
    assert not paths.output.exists()


def test_header_drift_stops_before_pixel_read_or_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch, paths)
    slide = _FakeSlide(mpp="0.3")
    publishes: list[object] = []

    with pytest.raises(Exception, match="mpp_x drift"):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda _path: slide,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )

    assert slide.calls == []
    assert slide.closed
    assert publishes == []
    assert not paths.output.exists()


@pytest.mark.parametrize("stale_kind", ["destination", "lock", "staging"])
def test_existing_or_stale_output_state_fails_before_open_or_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stale_kind: str,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    if stale_kind == "destination":
        paths.output.mkdir()
    elif stale_kind == "lock":
        (paths.output.parent / f".{paths.output.name}.lock").write_text("pid=1\n")
    else:
        (paths.output.parent / f".{paths.output.name}.staging.old").mkdir()
    opens: list[str] = []
    publishes: list[object] = []

    with pytest.raises(runner.Q25CoordinateGateError):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda path: opens.append(path),
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )

    assert opens == []
    assert publishes == []


def test_forbidden_git_output_fails_before_open_or_publish(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    forbidden = paths.repo_root / "coordinates"
    monkeypatch.setattr(runner, "OUTPUT_DIR", forbidden)
    paths = runner.GatePaths(
        repo_root=paths.repo_root,
        official_repo=paths.official_repo,
        wsi=paths.wsi,
        output=forbidden,
        auth=paths.auth,
    )
    opens: list[str] = []
    publishes: list[object] = []

    with pytest.raises(runner.Q25CoordinateGateError, match="outside Git"):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda path: opens.append(path),
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )

    assert opens == []
    assert publishes == []


def test_bound_source_hash_drift_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    auth = repo / "auth.yaml"
    policy = repo / "policy.py"
    auth.write_bytes(b"approved\n")
    policy.write_bytes(b"original\n")
    auth_sha = hashlib.sha256(auth.read_bytes()).hexdigest()
    expected_policy_sha = hashlib.sha256(policy.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "AUTH_RELATIVE_PATH", Path("auth.yaml"))
    monkeypatch.setattr(runner, "AUTH_SHA256", auth_sha)
    monkeypatch.setattr(runner, "BOUND_FILES", ((Path("policy.py"), expected_policy_sha),))
    policy.write_bytes(b"drift\n")
    paths = runner.GatePaths(
        repo_root=repo,
        official_repo=tmp_path,
        wsi=tmp_path / "unused.svs",
        output=tmp_path / "unused-output",
        auth=auth,
    )

    with pytest.raises(runner.Q25CoordinateGateError, match="SHA-256 drift"):
        runner._validate_bound_sources(paths)


def test_authorization_semantics_reject_wrong_read_scope_and_q50_unlock() -> None:
    document = yaml.safe_load(
        (runner.REPO_ROOT / runner.AUTH_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    paths = runner.GatePaths()
    runner._validate_authorization_semantics(document, paths)

    wrong_read = yaml.safe_load(yaml.safe_dump(document))
    wrong_read["authorized_operations"]["mask_pixel_read"]["maximum_calls"] = 2
    with pytest.raises(runner.Q25CoordinateGateError, match="mask-read"):
        runner._validate_authorization_semantics(wrong_read, paths)

    missing_q50_lock = yaml.safe_load(yaml.safe_dump(document))
    missing_q50_lock["explicitly_prohibited"].remove("q50_or_q75_download_or_open")
    with pytest.raises(runner.Q25CoordinateGateError, match="prohibition"):
        runner._validate_authorization_semantics(missing_q50_lock, paths)


def test_read_failure_always_closes_and_never_publishes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch, paths)
    slide = _FakeSlide(fail_read=True)
    publishes: list[object] = []

    with pytest.raises(RuntimeError, match="synthetic read failure"):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda _path: slide,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )
    assert slide.closed
    assert publishes == []
    assert not paths.output.exists()


def test_runner_binds_final_authorization_and_has_no_model_stack_imports() -> None:
    assert runner.AUTH_SHA256 == (
        "3797f428f5d1d49334fc2c0665325728318083d5deb8831deec5ad1f560ac617"
    )
    expected = dict(runner.BOUND_FILES)
    assert expected[Path("multiscale_feature_pilot/config/brca_q25_coordinate_policy.yaml")] == (
        "85410751aec43b14997fa4c0e2a611ceb329178f788df04f336031104b697d43"
    )
    assert expected[Path("multiscale_feature_pilot/src/brca_q25_coordinates.py")] == (
        "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82"
    )
    assert expected[Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py")] == (
        "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf"
    )

    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported_roots.isdisjoint({"torch", "torchvision", "healnet"})
    source = Path(runner.__file__).read_text(encoding="utf-8")
    assert "read_thumbnail" not in source
    assert "associated_images" not in source
