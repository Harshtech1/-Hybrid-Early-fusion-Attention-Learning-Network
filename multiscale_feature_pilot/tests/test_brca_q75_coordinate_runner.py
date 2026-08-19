from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from scripts import run_brca_q75_coordinate_gate as runner


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
            "openslide.mpp-y": str(runner.EXPECTED_MPP_Y),
        }
        self.level_dimensions = runner.EXPECTED_LEVEL_DIMENSIONS
        self.level_downsamples = runner.EXPECTED_LEVEL_DOWNSAMPLES
        self.calls: list[object] = []

    def read_region(self, *args):
        self.calls.append(args)
        raise AssertionError("header drift must block before the pixel read")


def _plan() -> runner.Q75CoordinatePolicyPlan:
    slide = SimpleNamespace(
        properties={
            "openslide.mpp-x": str(runner.EXPECTED_MPP_X),
            "openslide.mpp-y": str(runner.EXPECTED_MPP_Y),
        },
        level_dimensions=runner.EXPECTED_LEVEL_DIMENSIONS,
        level_downsamples=runner.EXPECTED_LEVEL_DOWNSAMPLES,
    )
    return runner._locked_policy_plan(slide)


def _paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> runner.GatePaths:
    repo = tmp_path / "pilot"
    official = tmp_path / "official"
    data = tmp_path / "data"
    incoming = data / "Q75.incoming"
    uuid_dir = incoming / runner.EXPECTED_GDC_FILE_UUID
    logs = uuid_dir / "logs"
    for directory in (repo, official, logs):
        directory.mkdir(parents=True)
    wsi = uuid_dir / runner.EXPECTED_FILENAME
    wsi.write_bytes(b"synthetic-not-a-real-wsi")
    (logs / f"{runner.EXPECTED_FILENAME}.parcel").write_bytes(b"parcel")
    omic = data / "omic.zip"
    omic.write_bytes(b"synthetic-not-a-real-archive")
    output = data / "Q75.coordinates"
    auth = repo / "auth.yaml"
    auth.write_text("synthetic: true\n", encoding="utf-8")

    monkeypatch.setattr(runner, "REPO_ROOT", repo)
    monkeypatch.setattr(runner, "OFFICIAL_REPO", official)
    monkeypatch.setattr(runner, "INCOMING_DIR", incoming)
    monkeypatch.setattr(runner, "WSI_PATH", wsi)
    monkeypatch.setattr(runner, "OMIC_PATH", omic)
    monkeypatch.setattr(runner, "OUTPUT_DIR", output)
    return runner.GatePaths(
        repo_root=repo,
        official_repo=official,
        incoming=incoming,
        wsi=wsi,
        omic=omic,
        output=output,
        auth=auth,
    )


def _snapshot() -> dict[str, object]:
    return {
        "official_head": runner.OFFICIAL_HEAD,
        "official_status": "",
        "frozen_tag": runner.FROZEN_TAG,
        "frozen_commit": runner.FROZEN_COMMIT,
        "source_commit_at_execution": "a" * 40,
        "critical_execution_source_sha256": {"runner": "c" * 64},
        "pilot_status_porcelain_v1_z": "",
    }


def _install_safe_preflight(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    snapshot = _snapshot()
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
    return snapshot


def _held_wsi(paths: runner.GatePaths) -> runner.HeldWsi:
    descriptor = os.open(paths.wsi, os.O_RDONLY)
    return runner.HeldWsi(
        path=paths.wsi,
        descriptor=descriptor,
        token=runner.FileToken.from_stat(os.fstat(descriptor)),
        md5=runner.EXPECTED_MD5,
        sha256=runner.EXPECTED_SHA256,
    )


def _held_omic(paths: runner.GatePaths) -> runner.HeldOmic:
    descriptor = os.open(paths.omic, os.O_RDONLY)
    return runner.HeldOmic(
        path=paths.omic,
        descriptor=descriptor,
        token=runner.FileToken.from_stat(os.fstat(descriptor)),
        sha256=runner.BRCA_RELEASE_ARCHIVE_SHA256,
    )


def _omics(*_args, **_kwargs):
    return SimpleNamespace(
        source_row_index=runner.EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX,
        case_id=runner.EXPECTED_PATIENT_ID,
        slide_id=runner.EXPECTED_SLIDE_ID,
        rna=torch.zeros((1, 1, 1558), dtype=torch.float32),
        mutation=torch.zeros((1, 1, 21), dtype=torch.float32),
        cnv=torch.zeros((1, 1, 1333), dtype=torch.float32),
    )


def _mask_reader(_slide):
    mask = np.zeros((2, 2, 4), dtype=np.uint8)
    return (
        mask,
        _plan(),
        1,
        hashlib.sha256(mask.tobytes(order="C")).hexdigest(),
    )


def _coordinate_builder(_mask, *, plan):
    assert plan == _plan()
    return runner.Q75CoordinateBags(
        scale_2x=np.asarray([[0, 0], [512, 0]], dtype=np.int64),
        scale_4x=np.asarray([[0, 0], [1024, 0]], dtype=np.int64),
        contour_count=2,
        retained_hole_count=1,
        mask_downsample_xy=plan.mask.coordinate_geometry_scale_xy,
    )


def _run_with_mocks(
    paths: runner.GatePaths,
    *,
    slide_factory=None,
    mask_reader=_mask_reader,
    coordinate_builder=_coordinate_builder,
    publisher=None,
):
    slide = _FakeSlide()
    opened_paths: list[str] = []

    def open_slide(path: str):
        opened_paths.append(path)
        return slide

    result = runner.run_coordinate_gate(
        paths=paths,
        slide_factory=open_slide if slide_factory is None else slide_factory,
        wsi_opener=lambda _path, **_kwargs: _held_wsi(paths),
        wsi_reverifier=lambda *_args, **_kwargs: None,
        omic_opener=lambda _path: _held_omic(paths),
        omic_loader=_omics,
        omic_reverifier=lambda _held: None,
        mask_reader=mask_reader,
        coordinate_builder=coordinate_builder,
        publisher=publisher,
    )
    return result, slide, opened_paths


def test_default_dependency_integration_reaches_real_atomic_publisher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch)
    slide = _FakeSlide()
    opened_paths: list[str] = []
    monkeypatch.setattr(
        runner.openslide,
        "OpenSlide",
        lambda path: (opened_paths.append(path), slide)[1],
    )
    monkeypatch.setattr(
        runner, "_open_verified_wsi", lambda _path, **_kwargs: _held_wsi(paths)
    )
    monkeypatch.setattr(runner, "_reverify_held_wsi", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_open_verified_omic", lambda _path: _held_omic(paths))
    monkeypatch.setattr(runner, "load_brca_patient_omics", _omics)
    monkeypatch.setattr(runner, "_reverify_held_omic", lambda _held: None)
    monkeypatch.setattr(runner, "_read_exact_mask", _mask_reader)
    monkeypatch.setattr(runner, "_build_q75_coordinate_bags", _coordinate_builder)

    result = runner.run_coordinate_gate(paths=paths)

    assert slide.closed
    assert len(opened_paths) == 1
    assert opened_paths[0].startswith("/proc/self/fd/")
    assert result["status"] == "BRCA_Q75_COORDINATES_VERIFIED"
    assert result["mask"]["read_region_calls"] == 1
    assert result["omic"]["source_row_index"] == "771"
    assert result["coordinate_artifacts"]["scale_2x"]["coordinate_count"] == 2
    assert result["coordinate_artifacts"]["scale_4x"]["coordinate_count"] == 2
    assert result["operations"]["gpu_operations"] == 0
    assert result["operations"]["training"] == 0
    assert result["operations"][
        "preexisting_raw_user_project_or_final_artifact_deletions"
    ] == 0
    cleanup = result["coordinate_artifacts"]["transaction_cleanup"]
    assert cleanup["runner_created_lock_present_after_return"] is False
    assert cleanup["runner_created_staging_present_after_return"] is False
    assert cleanup["preexisting_or_final_artifact_deletion_permitted"] is False
    assert paths.output.is_dir()
    assert (paths.output / "coordinate_manifest.json").is_file()


def test_mocked_gate_uses_held_proc_fd_and_stops_at_q75_coordinates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch)
    result, slide, opened_paths = _run_with_mocks(paths)

    assert slide.closed
    assert len(opened_paths) == 1
    assert opened_paths[0].startswith("/proc/self/fd/")
    assert result["operations"]["mask_read_region_calls"] == 1
    assert result["operations"]["level_0_or_level_1_patch_pixel_reads"] == 0
    assert result["operations"]["resnet50_feature_extraction"] == 0
    assert result["operations"]["healnet_execution"] == 0
    assert result["operations"]["q25_q50_reruns_or_modifications"] == 0
    assert result["operations"]["full_cohort_operations"] == 0
    assert result["required_stop_reached"] is True


def test_header_drift_stops_before_only_pixel_call() -> None:
    slide = _HeaderDriftSlide()
    with pytest.raises(Exception, match="MPP|mpp"):
        runner._read_exact_mask(slide)
    assert slide.calls == []


def test_success_requires_exactly_one_mask_read(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch)
    publishes: list[object] = []

    def two_reads(_slide):
        mask, plan, _, digest = _mask_reader(_slide)
        return mask, plan, 2, digest

    with pytest.raises(runner.Q75CoordinateGateError, match="exactly one"):
        _run_with_mocks(
            paths,
            mask_reader=two_reads,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )
    assert publishes == []
    assert not paths.output.exists()


def test_omic_mismatch_blocks_before_wsi_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch)
    wsi_opens: list[object] = []

    def wrong_omics(*_args, **_kwargs):
        value = _omics()
        value.source_row_index = "0"
        return value

    with pytest.raises(runner.Q75CoordinateGateError, match="source-row"):
        runner.run_coordinate_gate(
            paths=paths,
            omic_opener=lambda _path: _held_omic(paths),
            omic_loader=wrong_omics,
            wsi_opener=lambda *args, **kwargs: wsi_opens.append((args, kwargs)),
        )
    assert wsi_opens == []
    assert not paths.output.exists()


def test_injected_non_cpu_omic_tensor_is_rejected_before_conversion() -> None:
    class _CudaClaim:
        shape = (1, 1, 1558)
        dtype = torch.float32
        device = SimpleNamespace(type="cuda")

    value = _omics()
    value.rna = _CudaClaim()
    with pytest.raises(runner.Q75CoordinateGateError, match="CPU-only"):
        runner._load_and_validate_exact_omics(
            Path("/proc/self/fd/123"), loader=lambda *_args, **_kwargs: value
        )


@pytest.mark.parametrize("stale_kind", ["destination", "lock", "staging"])
def test_preexisting_publication_state_fails_before_any_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, stale_kind: str
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    if stale_kind == "destination":
        paths.output.mkdir()
    elif stale_kind == "lock":
        (paths.output.parent / f".{paths.output.name}.lock").write_text("preexisting\n")
    else:
        (paths.output.parent / f".{paths.output.name}.staging.preexisting").mkdir()
    opens: list[object] = []
    with pytest.raises(runner.Q75CoordinateGateError):
        runner.run_coordinate_gate(
            paths=paths,
            omic_opener=lambda path: opens.append(path),
            wsi_opener=lambda *args, **kwargs: opens.append((args, kwargs)),
        )
    assert opens == []


def test_wrong_mask_constant_fails_before_any_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "MASK_LOCATION", (1, 0))
    opens: list[object] = []
    with pytest.raises(runner.Q75CoordinateGateError, match="location constant"):
        runner.run_coordinate_gate(
            paths=paths,
            omic_opener=lambda path: opens.append(path),
        )
    assert opens == []


def test_wsi_reverification_failure_blocks_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch)
    publishes: list[object] = []

    def drift(*_args, **_kwargs):
        raise runner.Q75CoordinateGateError("held Q75 WSI SHA256 changed")

    with pytest.raises(runner.Q75CoordinateGateError, match="SHA256 changed"):
        runner.run_coordinate_gate(
            paths=paths,
            slide_factory=lambda _path: _FakeSlide(),
            wsi_opener=lambda _path, **_kwargs: _held_wsi(paths),
            wsi_reverifier=drift,
            omic_opener=lambda _path: _held_omic(paths),
            omic_loader=_omics,
            omic_reverifier=lambda _held: None,
            mask_reader=_mask_reader,
            coordinate_builder=_coordinate_builder,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )
    assert publishes == []
    assert not paths.output.exists()


def test_repository_drift_after_generation_blocks_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch)
    before = _snapshot()
    snapshots = [before, {**before, "source_commit_at_execution": "d" * 40}]
    monkeypatch.setattr(
        runner, "_repository_snapshot", lambda _paths: snapshots.pop(0)
    )
    publishes: list[object] = []
    with pytest.raises(runner.Q75CoordinateGateError, match="repository state changed"):
        _run_with_mocks(
            paths,
            publisher=lambda *args, **kwargs: publishes.append((args, kwargs)),
        )
    assert publishes == []
    assert not paths.output.exists()


def test_authorization_semantics_use_frozen_pure_validator() -> None:
    paths = runner.GatePaths()
    document = yaml.safe_load(paths.auth.read_text(encoding="utf-8"))
    runner._validate_authorization_semantics(document, paths)

    expanded_cleanup = yaml.safe_load(yaml.safe_dump(document))
    expanded_cleanup["authorized_operations"]["artifact_publication"][
        "ephemeral_transaction_cleanup"
    ] = "any_file"
    with pytest.raises(runner.Q75CoordinateGateError, match="authorized operations"):
        runner._validate_authorization_semantics(expanded_cleanup, paths)

    unlocked_gpu = yaml.safe_load(yaml.safe_dump(document))
    unlocked_gpu["required_stop"]["gpu_authorized"] = True
    with pytest.raises(runner.Q75CoordinateGateError, match="required stop"):
        runner._validate_authorization_semantics(unlocked_gpu, paths)


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
        incoming=tmp_path,
        wsi=tmp_path / "unused.svs",
        omic=tmp_path / "unused.zip",
        output=tmp_path / "unused-output",
        auth=auth,
    )
    with pytest.raises(runner.Q75CoordinateGateError, match="SHA-256 drift"):
        runner._validate_bound_sources(paths)


def test_final_publisher_is_not_followed_by_a_failing_semantic_gate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _paths(monkeypatch, tmp_path)
    _install_safe_preflight(monkeypatch)
    calls = 0

    def df(_path):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"available_bytes": 2_000_000_000}
        raise OSError("synthetic postpublication df failure")

    monkeypatch.setattr(runner, "_df_snapshot", df)
    result, _, _ = _run_with_mocks(paths)
    assert result["status"] == "BRCA_Q75_COORDINATES_VERIFIED"
    assert result["storage"]["after"]["measurement"] == (
        "postpublication_df_unavailable_nonfatal"
    )
    assert paths.output.is_dir()


def test_coordinate_builder_reuses_exact_reviewed_segmentation_and_lattices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    contour = np.asarray([[[0, 0]], [[10, 0]], [[10, 10]]], dtype=np.int32)
    geometry = SimpleNamespace(
        contours=(contour,),
        holes=((),),
        mask_downsample_xy=plan.mask.coordinate_geometry_scale_xy,
    )
    calls: list[tuple[int, int]] = []
    monkeypatch.setattr(
        runner,
        "segment_tissue_contours",
        lambda *_args, **_kwargs: geometry,
    )

    def lattice(*, level_0_patch_size, level_0_step, **_kwargs):
        calls.append((level_0_patch_size, level_0_step))
        return np.asarray([[0, 0]], dtype=np.int64)

    monkeypatch.setattr(runner, "generate_level_0_lattice_coordinates", lattice)
    mask = np.zeros((1, 1, 4), dtype=np.uint8)
    bags = runner._build_q75_coordinate_bags(mask, plan=plan)
    assert calls == [(512, 512), (1024, 1024)]
    assert bags.contour_count == 1
    assert bags.retained_hole_count == 0


def test_coordinate_bounds_check_is_overflow_safe() -> None:
    coordinates = np.asarray([[np.iinfo(np.int64).max, 0]], dtype=np.int64)
    with pytest.raises(runner.Q75CoordinateGateError, match="x bounds"):
        runner._require_coordinate_array(
            coordinates, branch="scale_2x", plan=_plan()
        )


def test_secure_open_rejects_final_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"payload")
    link = tmp_path / "link"
    link.symlink_to(target)
    with pytest.raises(runner.Q75CoordinateGateError, match="securely open"):
        runner._open_no_follow(link, label="test symlink")


def test_runner_binds_exact_q75_auth_sources_and_no_model_stack() -> None:
    assert runner.AUTH_RELATIVE_PATH == Path(
        "multiscale_feature_pilot/config/"
        "brca_q75_coordinate_execution_authorization.yaml"
    )
    assert runner.AUTH_SHA256 == (
        "4510cf2849edf3b0478030453b77faa1e0348f245b7e6703232d661c062f4539"
    )
    expected = dict(runner.BOUND_FILES)
    assert expected[
        Path("multiscale_feature_pilot/src/brca_q75_coordinate_authorization.py")
    ] == "794b759df886eaefdad017b468f381081a1328111064034a996782d6361e458f"
    assert expected[
        Path("multiscale_feature_pilot/config/brca_q75_coordinate_policy.yaml")
    ] == "58f15a9e39fcd3469ec656ef98c72ad6e42b8a3eab16fcbc24c4345cc4337d88"
    assert expected[
        Path("multiscale_feature_pilot/src/brca_omic.py")
    ] == "5061039913cb0dd4e8e30958c8b4b76f088396f7841255dd94234c3b462d5a5d"
    authorization = yaml.safe_load(
        (runner.REPO_ROOT / runner.AUTH_RELATIVE_PATH).read_text(encoding="utf-8")
    )
    bound_paths = {
        Path(value["path"])
        for value in authorization["bound_policy_identity"].values()
        if isinstance(value, dict) and "path" in value
    }
    assert bound_paths.issubset(expected)

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
    assert "unlink(" not in source and "rmtree(" not in source


def test_critical_sources_include_cycle_free_execution_chain() -> None:
    critical = set(runner.CRITICAL_TRACKED_PATHS)
    assert Path("scripts/run_brca_q75_coordinate_gate.py") in critical
    assert runner.AUTH_RELATIVE_PATH in critical
    assert runner.APPROVAL_RELATIVE_PATH in critical
    assert {path for path, _ in runner.BOUND_FILES}.issubset(critical)
