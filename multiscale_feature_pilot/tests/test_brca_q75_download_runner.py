from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

import pytest
import yaml

from multiscale_feature_pilot.src import brca_q75_download_runner as runner


class _SyntheticProcessRunner:
    def __init__(
        self,
        paths: runner.DownloadPaths,
        *,
        version: str,
        download_returncode: int = 0,
        create_tree: bool = True,
    ) -> None:
        self.paths = paths
        self.version = version
        self.download_returncode = download_returncode
        self.create_tree = create_tree
        self.calls: list[tuple[str, ...]] = []

    def __call__(self, command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(tuple(command))
        assert kwargs["shell"] is False if "shell" in kwargs else True
        if command == [str(self.paths.gdc_client), "--version"]:
            return subprocess.CompletedProcess(command, 0, f"{self.version}\n", "")
        assert command == runner._exact_download_command(self.paths)
        assert kwargs["cwd"] == str(self.paths.data_root)
        assert kwargs["check"] is False
        assert kwargs["capture_output"] is True
        assert kwargs["text"] is True
        if self.create_tree:
            uuid_directory = self.paths.incoming_directory / runner.EXPECTED_GDC_UUID
            logs = uuid_directory / "logs"
            logs.mkdir(parents=True)
            self.paths.wsi.write_bytes(b"synthetic-q75-wsi-fixture")
            (logs / f"{runner.EXPECTED_FILENAME}.parcel").write_bytes(
                b"synthetic-completed-parcel"
            )
        return subprocess.CompletedProcess(
            command,
            self.download_returncode,
            "Successfully downloaded: 1\n" if self.download_returncode == 0 else "",
            "" if self.download_returncode == 0 else "synthetic failure\n",
        )


def _copy_authorized_inputs(paths: runner.DownloadPaths) -> None:
    inputs = (
        (runner.DEFAULT_AUTHORIZATION, paths.authorization),
        (runner.DEFAULT_AUTHORIZATION_RECORD, paths.authorization_record),
        (runner.DEFAULT_MANIFEST, paths.manifest),
    )
    for source, destination in inputs:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())


def _paths(tmp_path: Path) -> tuple[runner.DownloadPaths, runner.ClientExpectation]:
    repo = tmp_path / "repo"
    data = tmp_path / "data"
    repo.mkdir()
    data.mkdir()
    authorization = repo / "config" / "authorization.yaml"
    authorization_record = repo / "provenance" / "authorization-record.yaml"
    manifest = repo / "provenance" / "q75.gdc.tsv"
    client = tmp_path / "tools" / "gdc-client"
    client.parent.mkdir()
    client_payload = b"synthetic-gdc-client-binary"
    client.write_bytes(client_payload)
    client.chmod(0o700)
    incoming = data / "Q75.incoming"
    result_directory = data / "Q75.download_result"
    paths = runner.DownloadPaths(
        repo_root=repo,
        authorization=authorization,
        authorization_record=authorization_record,
        manifest=manifest,
        gdc_client=client,
        data_root=data,
        incoming_directory=incoming,
        wsi=incoming / runner.EXPECTED_GDC_UUID / runner.EXPECTED_FILENAME,
        result_directory=result_directory,
        result_file=result_directory / "download_result.yaml",
        lock_file=data / ".Q75.download.lock",
        staging_directory=data / ".Q75.download_result.staging",
    )
    _copy_authorized_inputs(paths)
    expectation = runner.ClientExpectation(
        version="2.3",
        size_bytes=len(client_payload),
        sha256=hashlib.sha256(client_payload).hexdigest(),
    )
    return paths, expectation


def _source_binding() -> runner.SourceBinding:
    return runner.SourceBinding(
        commit="a" * 40,
        head_equal=True,
        files_head_equal=True,
        file_sha256={
            path.as_posix(): hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()
            for path in runner.SOURCE_RELATIVE_PATHS
        },
    )


def _disk_snapshots() -> runner.DiskSnapshotter:
    snapshots = iter(
        (
            {
                "measurement_path": "/synthetic",
                "total_bytes": 100_000_000_000,
                "used_bytes": 10_000_000_000,
                "free_bytes": 90_000_000_000,
            },
            {
                "measurement_path": "/synthetic",
                "total_bytes": 100_000_000_000,
                "used_bytes": 11_360_743_825,
                "free_bytes": 88_639_256_175,
            },
        )
    )

    def snapshot(_path: Path) -> Mapping[str, int]:
        return next(snapshots)

    return snapshot


def _synthetic_wsi_verifier(paths: runner.DownloadPaths) -> Mapping[str, Any]:
    details = runner._hash_regular_file(
        paths.wsi, include_md5=True, independent_sha256=True
    )
    return {
        **details,
        "gdc_uuid": runner.EXPECTED_GDC_UUID,
        "filename": runner.EXPECTED_FILENAME,
        "expected_size_bytes": details["size_bytes"],
        "expected_md5": details["md5"],
        "exact_uuid_and_filename": True,
        "exact_size_matches": True,
        "md5_matches": True,
        "exact_svs_suffix": True,
    }


def _run_success(
    paths: runner.DownloadPaths,
    expectation: runner.ClientExpectation,
    process: _SyntheticProcessRunner,
    **overrides: Any,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "paths": paths,
        "source_binding": _source_binding(),
        "client_expectation": expectation,
        "process_runner": process,
        "disk_snapshotter": _disk_snapshots(),
        "process_scanner": lambda _client: (),
        "rename_noreplace": runner._rename_noreplace_linux,
        "wsi_verifier": _synthetic_wsi_verifier,
    }
    arguments.update(overrides)
    return runner._run_download_gate(**arguments)


def test_synthetic_exact_download_publishes_atomic_sole_result_and_stops(
    tmp_path: Path,
) -> None:
    paths, expectation = _paths(tmp_path)
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    returned = _run_success(paths, expectation, process)

    assert returned["status"] == runner.RESULT_STATUS
    assert returned["execution"]["mode"] == "CPU_LOGIC_ONLY_NO_OPENSLIDE"
    assert returned["execution"]["source_commit"] == "a" * 40
    assert returned["command"]["returncode"] == 0
    assert returned["command"]["gdc_client_processes_started"] == 1
    assert returned["command"]["active_gdc_processes_before"] == 0
    assert returned["command"]["active_gdc_processes_after"] == 0
    assert returned["command"]["n_processes_1_passed"] is False
    assert returned["command"]["weakening_flags_passed"] is False
    assert process.calls == [
        (str(paths.gdc_client), "--version"),
        tuple(runner._exact_download_command(paths)),
    ]
    assert not paths.lock_file.exists()
    assert not paths.staging_directory.exists()
    assert paths.incoming_directory.is_dir()
    assert paths.result_directory.is_dir()
    assert [entry.name for entry in paths.result_directory.iterdir()] == [
        "download_result.yaml"
    ]
    published = yaml.safe_load(paths.result_file.read_text(encoding="utf-8"))
    assert published == returned
    assert published["download_tree"]["completed_tree_exact"] is True
    assert published["download_tree"]["exact_entry_count_excluding_root"] == 4
    assert published["download_tree"]["relative_entries"] == [
        runner.EXPECTED_GDC_UUID,
        f"{runner.EXPECTED_GDC_UUID}/{runner.EXPECTED_FILENAME}",
        f"{runner.EXPECTED_GDC_UUID}/logs",
        (
            f"{runner.EXPECTED_GDC_UUID}/logs/"
            f"{runner.EXPECTED_FILENAME}.parcel"
        ),
    ]
    assert published["wsi"]["sha256_independent_recheck_matches"] is True
    assert published["storage"]["filesystem_before_download"]["free_bytes"] == (
        90_000_000_000
    )
    assert published["storage"]["filesystem_after_download"]["free_bytes"] == (
        88_639_256_175
    )
    assert published["required_stop_reached"] is True
    assert published["next_gate"].endswith("AUTHORIZED_NEXT")
    assert published["operations"] == {
        "gdc_downloads": 1,
        "wsi_opens": 0,
        "pixel_or_region_reads": 0,
        "tissue_masks": 0,
        "coordinate_generation": 0,
        "patch_extraction": 0,
        "resnet50_inference": 0,
        "healnet_execution": 0,
        "q75_feature_generation": 0,
        "q75_raw_file_deletions": 0,
        "google_drive_operations": 0,
        "full_cohort_processing": 0,
        "q25_q50_modifications": 0,
        "blca_modifications": 0,
        "official_healnet_modifications": 0,
        "training_runs": 0,
        "backward_passes": 0,
        "optimizer_steps": 0,
    }


def test_exact_command_has_one_manifest_one_destination_and_no_weakening_flags(
    tmp_path: Path,
) -> None:
    paths, _ = _paths(tmp_path)
    command = runner._exact_download_command(paths)

    assert command == [
        str(paths.gdc_client),
        "download",
        "-m",
        str(paths.manifest),
        "--dir",
        str(paths.incoming_directory),
        "--no-related-files",
        "--no-annotations",
    ]
    for forbidden in ("-n", "--n-processes", "--no-verify", "--token-file"):
        assert forbidden not in command


def test_production_shaped_runner_record_is_accepted_by_header_gate_schema(
    tmp_path: Path,
) -> None:
    from multiscale_feature_pilot.src import brca_q75_header_gate as header

    paths, expectation = _paths(tmp_path)
    process = _SyntheticProcessRunner(paths, version=expectation.version)
    document = _run_success(paths, expectation, process)
    production_paths = runner.DownloadPaths()
    document["gdc_client"] = {
        "path": str(runner.DEFAULT_GDC_CLIENT),
        "version_output": runner.EXPECTED_GDC_CLIENT_VERSION,
        "size_bytes": runner.EXPECTED_GDC_CLIENT_SIZE_BYTES,
        "sha256": runner.EXPECTED_GDC_CLIENT_SHA256,
        "regular_non_symlink": True,
        "executable": True,
    }
    document["command"]["argv"] = runner._exact_download_command(production_paths)
    document["command"]["cwd"] = str(production_paths.data_root)
    document["download_tree"]["incoming_directory"] = str(
        production_paths.incoming_directory
    )
    document["download_tree"]["uuid_directory"] = str(
        production_paths.incoming_directory / runner.EXPECTED_GDC_UUID
    )
    document["download_tree"]["parcel"]["absolute_path"] = str(
        production_paths.incoming_directory
        / runner.EXPECTED_GDC_UUID
        / "logs"
        / f"{runner.EXPECTED_FILENAME}.parcel"
    )
    document["wsi"].update(
        {
            "path": str(production_paths.wsi),
            "size_bytes": runner.EXPECTED_SIZE_BYTES,
            "md5": runner.EXPECTED_MD5,
            "expected_size_bytes": runner.EXPECTED_SIZE_BYTES,
            "expected_md5": runner.EXPECTED_MD5,
        }
    )
    document["publication"]["result_directory"] = str(
        production_paths.result_directory
    )
    document["publication"]["result_file"] = str(production_paths.result_file)
    contract_directory = tmp_path / "contract-result"
    contract_directory.mkdir()
    contract_path = contract_directory / "download_result.yaml"
    contract_path.write_text(
        yaml.safe_dump(document, sort_keys=False), encoding="utf-8"
    )

    validated = header.validate_download_result(
        contract_directory,
        contract_path,
        current_source_commit=document["execution"]["source_commit"],
        current_source_hashes=document["execution"]["source_file_sha256"],
    )

    assert validated["wsi_sha256"] == document["wsi"]["sha256"]
    assert validated["gdc_created_tree"] == document["download_tree"][
        "relative_entries"
    ]


@pytest.mark.parametrize("state", ["destination", "lock", "staging", "result"])
@pytest.mark.parametrize("kind", ["regular", "symlink"])
def test_stale_or_symlink_state_fails_before_any_subprocess(
    tmp_path: Path, state: str, kind: str
) -> None:
    paths, expectation = _paths(tmp_path)
    selected = {
        "destination": paths.incoming_directory,
        "lock": paths.lock_file,
        "staging": paths.staging_directory,
        "result": paths.result_directory,
    }[state]
    if kind == "regular":
        if selected.suffix == ".lock":
            selected.write_text("stale\n", encoding="utf-8")
        else:
            selected.mkdir()
    else:
        selected.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    with pytest.raises(runner.Q75DownloadError, match="must be absent"):
        _run_success(paths, expectation, process)

    assert process.calls == []


def test_nonzero_gdc_exit_fails_closed_and_never_publishes_result(tmp_path: Path) -> None:
    paths, expectation = _paths(tmp_path)
    process = _SyntheticProcessRunner(
        paths,
        version=expectation.version,
        download_returncode=17,
        create_tree=True,
    )

    with pytest.raises(runner.Q75DownloadError, match="returned nonzero: 17"):
        _run_success(paths, expectation, process)

    assert paths.incoming_directory.exists()  # partial raw state is never deleted
    assert not paths.result_directory.exists()
    assert not paths.staging_directory.exists()
    assert not paths.lock_file.exists()


def test_unexpected_download_tree_entry_fails_before_result_publication(
    tmp_path: Path,
) -> None:
    paths, expectation = _paths(tmp_path)
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    def tree_with_extra(download_paths: runner.DownloadPaths) -> Mapping[str, Any]:
        extra = download_paths.incoming_directory / "unexpected.partial"
        extra.write_bytes(b"partial")
        return runner._validate_exact_download_tree(download_paths)

    with pytest.raises(runner.Q75DownloadError, match="partial or unexpected"):
        _run_success(paths, expectation, process, tree_validator=tree_with_extra)

    assert not paths.result_directory.exists()


def test_symlink_wsi_is_rejected_by_secure_file_verifier(tmp_path: Path) -> None:
    path = tmp_path / "slide.svs"
    target = tmp_path / "target.svs"
    target.write_bytes(b"payload")
    path.symlink_to(target)

    with pytest.raises(runner.Q75DownloadError, match="must not be a symlink"):
        runner._hash_regular_file(path, include_md5=True, independent_sha256=True)


def test_atomic_publication_interruption_cleans_private_stage_but_not_raw_wsi(
    tmp_path: Path,
) -> None:
    paths, expectation = _paths(tmp_path)
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    def interrupted(_source: Path, _destination: Path) -> None:
        raise InterruptedError("synthetic EINTR")

    with pytest.raises(InterruptedError, match="synthetic EINTR"):
        _run_success(paths, expectation, process, rename_noreplace=interrupted)

    assert paths.incoming_directory.exists()
    assert paths.wsi.exists()
    assert not paths.result_directory.exists()
    assert not paths.staging_directory.exists()
    assert not paths.lock_file.exists()


@pytest.mark.parametrize("destination_kind", ["directory", "symlink"])
def test_renameat2_noreplace_refuses_existing_destination(
    tmp_path: Path, destination_kind: str
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "download_result.yaml").write_text("status: fixture\n", encoding="utf-8")
    destination = tmp_path / "destination"
    if destination_kind == "directory":
        destination.mkdir()
        marker = destination / "keep"
        marker.write_text("do not overwrite\n", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        destination.symlink_to(target, target_is_directory=True)

    with pytest.raises(runner.Q75DownloadError, match="already exists"):
        runner._rename_noreplace_linux(source, destination)

    assert source.is_dir()
    if destination_kind == "directory":
        assert marker.read_text(encoding="utf-8") == "do not overwrite\n"
    else:
        assert destination.is_symlink()


def test_tampered_manifest_fails_before_version_or_download(tmp_path: Path) -> None:
    paths, expectation = _paths(tmp_path)
    paths.manifest.write_bytes(paths.manifest.read_bytes() + b"\n")
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    with pytest.raises(runner.Q75DownloadError, match="manifest is not the exact"):
        _run_success(paths, expectation, process)

    assert process.calls == []
    assert not paths.incoming_directory.exists()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("authorization", "authorization config SHA256 mismatch"),
        ("authorization_record", "authorization-record SHA256 mismatch"),
    ],
)
def test_tampered_authority_fails_before_client_execution(
    tmp_path: Path, field: str, message: str
) -> None:
    paths, expectation = _paths(tmp_path)
    selected = getattr(paths, field)
    selected.write_bytes(selected.read_bytes() + b"\n")
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    with pytest.raises(runner.Q75DownloadError, match=message):
        _run_success(paths, expectation, process)

    assert process.calls == []
    assert not paths.incoming_directory.exists()


def test_tampered_gdc_client_fails_before_download(tmp_path: Path) -> None:
    paths, expectation = _paths(tmp_path)
    paths.gdc_client.write_bytes(paths.gdc_client.read_bytes() + b"tamper")
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    with pytest.raises(runner.Q75DownloadError, match="GDC client size mismatch"):
        _run_success(paths, expectation, process)

    assert process.calls == []
    assert not paths.incoming_directory.exists()


def test_insufficient_storage_fails_before_download(tmp_path: Path) -> None:
    paths, expectation = _paths(tmp_path)
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    def insufficient(_path: Path) -> Mapping[str, int]:
        return {
            "total_bytes": runner.MINIMUM_FREE_BYTES,
            "used_bytes": 1,
            "free_bytes": runner.MINIMUM_FREE_BYTES - 1,
        }

    with pytest.raises(runner.Q75DownloadError, match="insufficient free storage"):
        _run_success(
            paths,
            expectation,
            process,
            disk_snapshotter=insufficient,
        )

    assert process.calls == [(str(paths.gdc_client), "--version")]
    assert not paths.incoming_directory.exists()


def test_existing_active_gdc_process_fails_before_download(tmp_path: Path) -> None:
    paths, expectation = _paths(tmp_path)
    process = _SyntheticProcessRunner(paths, version=expectation.version)

    with pytest.raises(runner.Q75DownloadError, match="another GDC client"):
        _run_success(
            paths,
            expectation,
            process,
            process_scanner=lambda _client: (4321,),
        )

    assert process.calls == [(str(paths.gdc_client), "--version")]
    assert not paths.incoming_directory.exists()


def test_exact_wsi_verifier_fails_on_md5_mismatch_after_exact_size(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths, _ = _paths(tmp_path)
    uuid_directory = paths.incoming_directory / runner.EXPECTED_GDC_UUID
    logs = uuid_directory / "logs"
    logs.mkdir(parents=True)
    payload = b"small-synthetic-wsi"
    paths.wsi.write_bytes(payload)
    (logs / f"{runner.EXPECTED_FILENAME}.parcel").write_bytes(b"complete")
    monkeypatch.setattr(runner, "EXPECTED_SIZE_BYTES", len(payload))
    monkeypatch.setattr(runner, "EXPECTED_MD5", "0" * 32)

    with pytest.raises(runner.Q75DownloadError, match="MD5 mismatch"):
        runner._verify_exact_wsi(paths)


def test_private_stage_cleanup_never_deletes_a_swapped_directory(tmp_path: Path) -> None:
    stage = tmp_path / ".Q75.download_result.staging"
    stage.mkdir()
    original = stage.lstat()
    moved = tmp_path / "original-stage"
    stage.rename(moved)
    stage.mkdir()
    replacement = stage / "download_result.yaml"
    replacement.write_text("replacement\n", encoding="utf-8")
    replacement_details = replacement.lstat()

    runner._cleanup_private_staging(
        stage,
        "download_result.yaml",
        expected_stage_identity=(original.st_dev, original.st_ino),
        expected_file_identity=(replacement_details.st_dev, replacement_details.st_ino),
    )

    assert stage.is_dir()
    assert replacement.read_text(encoding="utf-8") == "replacement\n"
    assert moved.is_dir()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_source_binding_requires_explicit_head_and_byte_equal_tracked_sources(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.test")
    _git(repo, "config", "user.name", "Fixture")
    source = repo / "src" / "runner.py"
    script = repo / "scripts" / "run.py"
    source.parent.mkdir()
    script.parent.mkdir()
    source.write_text("VALUE = 1\n", encoding="utf-8")
    script.write_text("print('fixture')\n", encoding="utf-8")
    _git(repo, "add", "src/runner.py", "scripts/run.py")
    _git(repo, "commit", "-m", "fixture")
    commit = _git(repo, "rev-parse", "HEAD")

    binding = runner.verify_source_binding(
        repo, commit, relative_paths=(Path("src/runner.py"), Path("scripts/run.py"))
    )
    assert binding.commit == commit
    assert binding.head_equal is True
    assert binding.files_head_equal is True
    assert set(binding.file_sha256) == {"src/runner.py", "scripts/run.py"}

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(runner.Q75DownloadError, match="not HEAD-equal"):
        runner.verify_source_binding(
            repo,
            commit,
            relative_paths=(Path("src/runner.py"), Path("scripts/run.py")),
        )


def test_source_binding_rejects_wrong_head_before_comparing_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "fixture@example.test")
    _git(repo, "config", "user.name", "Fixture")
    source = repo / "source.py"
    source.write_text("pass\n", encoding="utf-8")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "fixture")

    with pytest.raises(runner.Q75DownloadError, match="HEAD does not equal"):
        runner.verify_source_binding(
            repo, "0" * 40, relative_paths=(Path("source.py"),)
        )


def test_runner_has_no_openslide_cuda_pixel_or_model_implementation() -> None:
    source = Path(runner.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "openslide" not in imports
    assert "torch" not in imports
    assert "torchvision" not in imports
    assert "read_region" not in calls
    assert "backward" not in calls
    assert "step" not in calls


def test_production_path_object_is_exact_and_not_overrideable_by_cli() -> None:
    paths = runner.DownloadPaths()
    runner._validate_production_paths(paths)

    with pytest.raises(runner.Q75DownloadError, match="production incoming_directory drift"):
        runner._validate_production_paths(
            replace(paths, incoming_directory=paths.data_root / "different")
        )
