from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from multiscale_feature_pilot.src.brca_omic import BrcaPatientOmics
from multiscale_feature_pilot.src import brca_q75_header_gate as gate


class _HeaderOnlySlide:
    def __init__(self, *, mpp_x: object = "0.25") -> None:
        self.properties = {
            "openslide.mpp-x": mpp_x,
            "openslide.mpp-y": "0.25",
        }
        self.level_count = 3
        self.level_dimensions = ((4000, 2000), (2000, 1000), (1000, 500))
        self.level_downsamples = (1.0, 2.0, 4.0)
        self.forbidden_calls: list[object] = []
        self.closed = False

    def read_region(self, *arguments: object) -> None:
        self.forbidden_calls.append(arguments)
        raise AssertionError("pixel access is forbidden")

    def close(self) -> None:
        self.closed = True


def _create_tree(tmp_path: Path) -> gate.GatePaths:
    incoming = tmp_path / "brca_pilot_data/Q75.incoming"
    uuid_directory = incoming / gate.EXPECTED_GDC_UUID
    logs = uuid_directory / "logs"
    logs.mkdir(parents=True)
    wsi = uuid_directory / gate.EXPECTED_FILENAME
    wsi.write_bytes(b"synthetic fixture; never decoded")
    (logs / f"{gate.EXPECTED_FILENAME}.parcel").write_text(
        "completed synthetic parcel\n", encoding="utf-8"
    )
    omic = tmp_path / "omic.zip"
    omic.write_bytes(b"synthetic held Omic fixture")
    output_parent = tmp_path / "repo/provenance"
    output_parent.mkdir(parents=True)
    return gate.GatePaths(
        repo_root=tmp_path / "repo",
        authorization=gate.DEFAULT_AUTHORIZATION,
        authorization_record=gate.DEFAULT_AUTHORIZATION_RECORD,
        manifest=gate.DEFAULT_AUTHORIZED_MANIFEST,
        download_result_directory=tmp_path / "unused-download-result",
        download_result=tmp_path / "unused-download-result/download_result.yaml",
        incoming_directory=incoming,
        wsi=wsi,
        omic_archive=omic,
        output_directory=output_parent / "q75_header_result",
    )


def _omics(*, slide_id: str = gate.EXPECTED_SLIDE_ID) -> BrcaPatientOmics:
    return BrcaPatientOmics(
        source_row_index=gate.EXPECTED_OMIC_SOURCE_INDEX,
        case_id=gate.EXPECTED_PATIENT_ID,
        slide_id=slide_id,
        rna=torch.zeros((1, 1, 1558), dtype=torch.float32),
        mutation=torch.zeros((1, 1, 21), dtype=torch.float32),
        cnv=torch.zeros((1, 1, 1333), dtype=torch.float32),
    )


def _held(path: Path, *, sha256: str) -> gate.HeldRegularFile:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    return gate.HeldRegularFile(
        path=path,
        descriptor=descriptor,
        token=gate.FileToken.from_stat(os.fstat(descriptor)),
        sha256=sha256,
    )


def _verified(paths: gate.GatePaths, *, sha256: str = "f" * 64) -> gate.VerifiedWsi:
    descriptor = os.open(
        paths.wsi,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    return gate.VerifiedWsi(
        path=str(paths.wsi),
        descriptor=descriptor,
        token=gate.FileToken.from_stat(os.fstat(descriptor)),
        size_bytes=gate.EXPECTED_SIZE_BYTES,
        md5=gate.EXPECTED_MD5,
        sha256=sha256,
    )


def _source() -> dict[str, object]:
    return {
        "source_commit": "a" * 40,
        "allowed_dirty_status": [gate.PROTECTED_DIRTY_STATUS],
        "critical_file_sha256": {
            **_download_source_hashes(),
            "multiscale_feature_pilot/src/brca_q75_header_gate.py": "e" * 64,
        },
        "critical_files_head_equal": True,
    }


def _download_source_hashes() -> dict[str, str]:
    return {
        relative.as_posix(): "b" * 64
        for relative in gate.DOWNLOAD_SOURCE_RELATIVE_PATHS
    }


def _download(*, sha256: str = "f" * 64) -> dict[str, object]:
    return {
        "path": str(gate.DEFAULT_DOWNLOAD_RESULT),
        "sha256": "c" * 64,
        "source_commit": "a" * 40,
        "wsi_sha256": sha256,
        "filesystem_before_download": {
            "total_bytes": 1000,
            "used_bytes": 300,
            "free_bytes": 700,
        },
        "filesystem_after_download": {
            "total_bytes": 1000,
            "used_bytes": 500,
            "free_bytes": 500,
        },
        "gdc_created_tree": gate._expected_gdc_tree(),
        "parcel": {
            "path": gate._expected_gdc_tree()[-1],
            "size_bytes": 10,
            "sha256": "d" * 64,
            "regular_non_symlink": True,
        },
    }


def _install_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate,
        "_disk_snapshot",
        lambda _path: {"total_bytes": 1000, "used_bytes": 500, "free_bytes": 500},
    )


def test_header_gate_uses_stable_fds_atomic_directory_and_stops(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _create_tree(tmp_path)
    _install_disk(monkeypatch)
    slide = _HeaderOnlySlide()
    verified_holder: list[gate.VerifiedWsi] = []
    omic_holder: list[gate.HeldRegularFile] = []
    opened_paths: list[str] = []
    repository_calls = 0

    def repository(_repo: Path) -> dict[str, object]:
        nonlocal repository_calls
        repository_calls += 1
        return _source()

    def verifier(_path: Path, **_kwargs: object) -> gate.VerifiedWsi:
        value = _verified(paths)
        verified_holder.append(value)
        return value

    def omic_opener(_path: Path) -> gate.HeldRegularFile:
        value = _held(paths.omic_archive, sha256=gate.BRCA_RELEASE_ARCHIVE_SHA256)
        omic_holder.append(value)
        return value

    def slide_factory(path: str) -> _HeaderOnlySlide:
        opened_paths.append(path)
        assert path == f"/proc/self/fd/{verified_holder[0].descriptor}"
        assert gate.FileToken.from_stat(os.stat(path)) == verified_holder[0].token
        return slide

    record = gate.run_header_gate(
        paths=paths,
        slide_factory=slide_factory,
        omic_loader=lambda *_args, **_kwargs: _omics(),
        repository_validator=repository,
        download_validator=lambda *_args, **_kwargs: _download(),
        wsi_verifier=verifier,
        omic_opener=omic_opener,
        wsi_reverifier=lambda *_args, **_kwargs: None,
        omic_reverifier=lambda *_args, **_kwargs: None,
    )

    assert repository_calls == 2
    assert opened_paths == [f"/proc/self/fd/{verified_holder[0].descriptor}"]
    assert slide.closed and slide.forbidden_calls == []
    with pytest.raises(OSError):
        os.fstat(verified_holder[0].descriptor)
    with pytest.raises(OSError):
        os.fstat(omic_holder[0].descriptor)
    assert record["source"]["source_commit"] == "a" * 40
    assert record["download_result"]["sha256"] == "c" * 64
    assert record["wsi"]["held_o_nofollow_descriptor_used"] is True
    assert record["omic"]["held_o_nofollow_descriptor_used"] is True
    assert record["scale_policy_approved"] is False
    assert record["required_stop_reached"] is True
    assert {item.name for item in paths.output_directory.iterdir()} == {
        gate.RESULT_BASENAME,
        gate.REPORT_BASENAME,
    }
    result = yaml.safe_load(
        (paths.output_directory / gate.RESULT_BASENAME).read_text(encoding="utf-8")
    )
    assert result["status"] == gate.RESULT_STATUS
    assert "does not select or approve" in (
        paths.output_directory / gate.REPORT_BASENAME
    ).read_text(encoding="utf-8")


def test_invalid_header_closes_fds_and_publishes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _create_tree(tmp_path)
    _install_disk(monkeypatch)
    slide = _HeaderOnlySlide(mpp_x="invalid")
    verified = _verified(paths)
    held = _held(paths.omic_archive, sha256=gate.BRCA_RELEASE_ARCHIVE_SHA256)

    with pytest.raises(gate.Q75HeaderGateError, match="openslide.mpp-x"):
        gate.run_header_gate(
            paths=paths,
            slide_factory=lambda _path: slide,
            omic_loader=lambda *_args, **_kwargs: _omics(),
            repository_validator=lambda _repo: _source(),
            download_validator=lambda *_args, **_kwargs: _download(),
            wsi_verifier=lambda *_args, **_kwargs: verified,
            omic_opener=lambda _path: held,
            wsi_reverifier=lambda *_args, **_kwargs: None,
            omic_reverifier=lambda *_args, **_kwargs: None,
        )

    assert slide.closed and slide.forbidden_calls == []
    with pytest.raises(OSError):
        os.fstat(verified.descriptor)
    with pytest.raises(OSError):
        os.fstat(held.descriptor)
    assert not paths.output_directory.exists()


def test_omic_drift_stops_before_openslide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _create_tree(tmp_path)
    _install_disk(monkeypatch)
    opens: list[str] = []

    with pytest.raises(gate.Q75HeaderGateError, match="full slide mismatch"):
        gate.run_header_gate(
            paths=paths,
            slide_factory=lambda path: opens.append(path),
            omic_loader=lambda *_args, **_kwargs: _omics(slide_id="wrong.svs"),
            repository_validator=lambda _repo: _source(),
            download_validator=lambda *_args, **_kwargs: _download(),
            wsi_verifier=lambda *_args, **_kwargs: _verified(paths),
            omic_opener=lambda _path: _held(
                paths.omic_archive, sha256=gate.BRCA_RELEASE_ARCHIVE_SHA256
            ),
            wsi_reverifier=lambda *_args, **_kwargs: None,
            omic_reverifier=lambda *_args, **_kwargs: None,
        )

    assert opens == []
    assert not paths.output_directory.exists()


def test_source_change_before_publication_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _create_tree(tmp_path)
    _install_disk(monkeypatch)
    calls = 0

    def repository(_repo: Path) -> dict[str, object]:
        nonlocal calls
        calls += 1
        value = _source()
        if calls == 2:
            value = {**value, "source_commit": "e" * 40}
        return value

    with pytest.raises(gate.Q75HeaderGateError, match="source binding changed"):
        gate.run_header_gate(
            paths=paths,
            slide_factory=lambda _path: _HeaderOnlySlide(),
            omic_loader=lambda *_args, **_kwargs: _omics(),
            repository_validator=repository,
            download_validator=lambda *_args, **_kwargs: _download(),
            wsi_verifier=lambda *_args, **_kwargs: _verified(paths),
            omic_opener=lambda _path: _held(
                paths.omic_archive, sha256=gate.BRCA_RELEASE_ARCHIVE_SHA256
            ),
            wsi_reverifier=lambda *_args, **_kwargs: None,
            omic_reverifier=lambda *_args, **_kwargs: None,
        )
    assert not paths.output_directory.exists()


def test_download_sha_mismatch_stops_before_omic_and_openslide(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _create_tree(tmp_path)
    _install_disk(monkeypatch)
    omic_opens: list[str] = []
    slide_opens: list[str] = []

    with pytest.raises(gate.Q75HeaderGateError, match="WSI/download SHA256 mismatch"):
        gate.run_header_gate(
            paths=paths,
            slide_factory=lambda path: slide_opens.append(path),
            repository_validator=lambda _repo: _source(),
            download_validator=lambda *_args, **_kwargs: _download(sha256="0" * 64),
            wsi_verifier=lambda *_args, **_kwargs: _verified(paths),
            omic_opener=lambda path: omic_opens.append(str(path)),
        )
    assert omic_opens == [] and slide_opens == []


def test_exact_wsi_descriptor_survives_path_swap_but_reverification_rejects_it(
    tmp_path: Path,
) -> None:
    paths = _create_tree(tmp_path)
    payload = paths.wsi.read_bytes()
    verified = gate.verify_exact_wsi(
        paths.wsi,
        incoming_directory=paths.incoming_directory,
        expected_size=len(payload),
        expected_md5=hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    )
    replacement = paths.wsi.with_suffix(".replacement")
    replacement.write_bytes(payload)
    os.replace(replacement, paths.wsi)
    try:
        assert os.read(verified.descriptor, 0) == b""
        with pytest.raises(gate.Q75HeaderGateError, match="pathname changed"):
            gate.reverify_held_wsi(
                paths.wsi,
                incoming_directory=paths.incoming_directory,
                verified=verified,
            )
    finally:
        os.close(verified.descriptor)


def test_exact_wsi_rejects_partial_and_symlink(tmp_path: Path) -> None:
    paths = _create_tree(tmp_path)
    payload = paths.wsi.read_bytes()
    kwargs = {
        "incoming_directory": paths.incoming_directory,
        "expected_size": len(payload),
        "expected_md5": hashlib.md5(payload, usedforsecurity=False).hexdigest(),
    }
    partial = paths.wsi.with_suffix(".svs.partial")
    partial.write_bytes(b"partial")
    with pytest.raises(gate.Q75HeaderGateError, match="partial or unexpected"):
        gate.verify_exact_wsi(paths.wsi, **kwargs)
    partial.unlink()
    target = tmp_path / "target.svs"
    target.write_bytes(payload)
    paths.wsi.unlink()
    paths.wsi.symlink_to(target)
    with pytest.raises(gate.Q75HeaderGateError):
        gate.verify_exact_wsi(paths.wsi, **kwargs)


def test_secure_metadata_read_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.yaml"
    target.write_text("status: safe\n", encoding="utf-8")
    link = tmp_path / "link.yaml"
    link.symlink_to(target)
    with pytest.raises(gate.Q75HeaderGateError):
        gate._secure_regular_bytes(link, label="fixture", maximum=100)


def test_atomic_result_directory_is_no_replace_and_exact(tmp_path: Path) -> None:
    parent = tmp_path / "provenance"
    parent.mkdir()
    output = parent / "result"
    hashes = gate.publish_result_directory(
        output,
        result_payload=b"status: success\n",
        report_payload=b"# report\n",
    )
    assert set(hashes) == {gate.RESULT_BASENAME, gate.REPORT_BASENAME}
    assert {item.name for item in output.iterdir()} == {
        gate.RESULT_BASENAME,
        gate.REPORT_BASENAME,
    }
    with pytest.raises(gate.Q75HeaderGateError, match="already exists"):
        gate.publish_result_directory(
            output,
            result_payload=b"different\n",
            report_payload=b"different\n",
        )
    assert (output / gate.RESULT_BASENAME).read_bytes() == b"status: success\n"


def test_atomic_result_staging_is_removed_on_rename_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    parent = tmp_path / "provenance"
    parent.mkdir()
    output = parent / "result"
    monkeypatch.setattr(
        gate,
        "_rename_directory_no_replace",
        lambda *_args: (_ for _ in ()).throw(gate.Q75HeaderGateError("synthetic")),
    )
    with pytest.raises(gate.Q75HeaderGateError, match="synthetic"):
        gate.publish_result_directory(
            output,
            result_payload=b"result\n",
            report_payload=b"report\n",
        )
    assert not output.exists()
    assert not tuple(parent.glob(".result.staging.*"))


def test_repository_binding_requires_head_equal_files_and_only_protected_edit(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    critical = repo / "critical.py"
    protected = repo / "reports/blca_one_patient_multiscale_pilot.md"
    protected.parent.mkdir(parents=True)
    critical.write_text("VALUE = 1\n", encoding="utf-8")
    protected.write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture"], check=True)
    protected.write_text("user modification\n", encoding="utf-8")

    result = gate.validate_repository_binding(repo, critical_paths=[Path("critical.py")])
    assert result["allowed_dirty_status"] == [gate.PROTECTED_DIRTY_STATUS]
    assert result["critical_files_head_equal"] is True

    critical.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(gate.Q75HeaderGateError, match="Git state"):
        gate.validate_repository_binding(repo, critical_paths=[Path("critical.py")])


def _download_document() -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": gate.EXPECTED_DOWNLOAD_STATUS,
        "cohort": "TCGA-BRCA",
        "candidate": "Q75",
        "patient_id": gate.EXPECTED_PATIENT_ID,
        "slide_id": gate.EXPECTED_SLIDE_ID,
        "gdc_file_uuid": gate.EXPECTED_GDC_UUID,
        "execution": {
            "source_commit": "a" * 40,
            "source_head_equal": True,
            "source_files_head_equal": True,
            "source_file_sha256": _download_source_hashes(),
        },
        "bindings": {
            "authorization_config_sha256": gate.EXPECTED_AUTHORIZATION_SHA256,
            "authorization_record_sha256": gate.EXPECTED_AUTHORIZATION_RECORD_SHA256,
            "manifest_sha256": gate.EXPECTED_MANIFEST_SHA256,
        },
        "gdc_client": {
            "path": str(gate.DEFAULT_GDC_CLIENT),
            "version_output": gate.EXPECTED_GDC_CLIENT_VERSION,
            "size_bytes": gate.EXPECTED_GDC_CLIENT_SIZE_BYTES,
            "sha256": gate.EXPECTED_GDC_CLIENT_SHA256,
            "regular_non_symlink": True,
            "executable": True,
        },
        "command": {
            "argv": [
                str(gate.DEFAULT_GDC_CLIENT),
                "download",
                "-m",
                str(gate.DEFAULT_AUTHORIZED_MANIFEST),
                "--dir",
                str(gate.DEFAULT_INCOMING_DIRECTORY),
                "--no-related-files",
                "--no-annotations",
            ],
            "returncode": 0,
            "gdc_client_processes_started": 1,
            "active_gdc_processes_before": 0,
            "active_gdc_processes_after": 0,
        },
        "storage": {
            "filesystem_before_download": {
                "total_bytes": 1000,
                "used_bytes": 300,
                "free_bytes": 700,
            },
            "filesystem_after_download": {
                "total_bytes": 1000,
                "used_bytes": 500,
                "free_bytes": 500,
            },
        },
        "download_tree": {
            "incoming_directory": str(gate.DEFAULT_INCOMING_DIRECTORY),
            "uuid_directory": str(
                gate.DEFAULT_INCOMING_DIRECTORY / gate.EXPECTED_GDC_UUID
            ),
            "relative_entries": gate._expected_gdc_tree(),
            "entry_types": {},
            "exact_entry_count_excluding_root": 4,
            "partial_or_unexpected_entries": [],
            "completed_tree_exact": True,
            "parcel": {
                "path": gate._expected_gdc_tree()[-1],
                "absolute_path": str(
                    gate.DEFAULT_INCOMING_DIRECTORY / gate._expected_gdc_tree()[-1]
                ),
                "size_bytes": 10,
                "sha256": "1" * 64,
                "regular_non_symlink": True,
            },
        },
        "wsi": {
            "path": str(gate.DEFAULT_WSI),
            "size_bytes": gate.EXPECTED_SIZE_BYTES,
            "md5": gate.EXPECTED_MD5,
            "sha256": "2" * 64,
            "sha256_independent_recheck_matches": True,
            "regular_non_symlink": True,
            "exact_svs_suffix": True,
            "gdc_uuid": gate.EXPECTED_GDC_UUID,
            "filename": gate.EXPECTED_FILENAME,
            "expected_size_bytes": gate.EXPECTED_SIZE_BYTES,
            "expected_md5": gate.EXPECTED_MD5,
            "exact_uuid_and_filename": True,
            "exact_size_matches": True,
            "md5_matches": True,
        },
        "validations": {"all": "PASS"},
        "operations": dict(gate.EXPECTED_DOWNLOAD_OPERATIONS),
        "publication": {
            "result_directory": str(gate.DEFAULT_DOWNLOAD_RESULT_DIRECTORY),
            "result_file": str(gate.DEFAULT_DOWNLOAD_RESULT),
            "atomic_directory_rename_noreplace": True,
            "sole_file": True,
        },
        "required_stop": gate.EXPECTED_DOWNLOAD_STOP,
        "required_stop_reached": True,
        "next_gate": "HEADER_REVIEW",
    }


def test_download_result_exact_schema_and_sha_are_bound(tmp_path: Path) -> None:
    directory = tmp_path / "Q75.download_result"
    directory.mkdir()
    path = directory / "download_result.yaml"
    path.write_text(yaml.safe_dump(_download_document(), sort_keys=False), encoding="utf-8")
    result = gate.validate_download_result(
        directory,
        path,
        current_source_commit="a" * 40,
        current_source_hashes=_download_source_hashes(),
    )
    assert result["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["gdc_created_tree"] == gate._expected_gdc_tree()
    assert result["filesystem_before_download"]["used_bytes"] == 300


@pytest.mark.parametrize("drift", ["old_tree_schema", "download_counter", "source_hash"])
def test_download_result_rejects_cross_schema_and_binding_drift(
    tmp_path: Path, drift: str
) -> None:
    document = copy.deepcopy(_download_document())
    if drift == "old_tree_schema":
        tree = document["download_tree"]
        assert isinstance(tree, dict)
        tree["exact_relative_entries"] = tree.pop("relative_entries")
    elif drift == "download_counter":
        operations = document["operations"]
        assert isinstance(operations, dict)
        operations["gdc_downloads"] = 0
    else:
        execution = document["execution"]
        assert isinstance(execution, dict)
        source_hashes = execution["source_file_sha256"]
        assert isinstance(source_hashes, dict)
        source_hashes[next(iter(source_hashes))] = "0" * 64

    directory = tmp_path / "Q75.download_result"
    directory.mkdir()
    path = directory / "download_result.yaml"
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    with pytest.raises(gate.Q75HeaderGateError):
        gate.validate_download_result(
            directory,
            path,
            current_source_commit="a" * 40,
            current_source_hashes=_download_source_hashes(),
        )


def test_source_has_no_pixel_call_and_cli_has_no_path_overrides() -> None:
    source_paths = [
        Path(gate.__file__),
        Path(__file__).resolve().parents[2] / "scripts/run_brca_q75_header_gate.py",
    ]
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "read_region"
        ]
    cli_source = source_paths[1].read_text(encoding="utf-8")
    for forbidden in (
        "--wsi",
        "--incoming-directory",
        "--omic-archive",
        "--result",
        "--report",
        "--manifest",
    ):
        assert forbidden not in cli_source


def test_checked_in_authorization_artifacts_bind_exactly_and_reject_symlink(
    tmp_path: Path,
) -> None:
    authorization_sha256 = gate.validate_authorization(
        gate.DEFAULT_AUTHORIZATION,
        expected_sha256=gate.EXPECTED_AUTHORIZATION_SHA256,
    )
    assert gate.validate_authorization_record(
        gate.DEFAULT_AUTHORIZATION_RECORD,
        expected_sha256=gate.EXPECTED_AUTHORIZATION_RECORD_SHA256,
        authorization_sha256=authorization_sha256,
    ) == gate.EXPECTED_AUTHORIZATION_RECORD_SHA256
    assert gate.validate_authorized_manifest(
        gate.DEFAULT_AUTHORIZED_MANIFEST
    ) == gate.EXPECTED_MANIFEST_SHA256

    link = tmp_path / "authorization.yaml"
    link.symlink_to(gate.DEFAULT_AUTHORIZATION)
    with pytest.raises(gate.Q75HeaderGateError):
        gate.validate_authorization(
            link,
            expected_sha256=gate.EXPECTED_AUTHORIZATION_SHA256,
        )
