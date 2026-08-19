from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src.brca_one_row_manifest import OneRowManifestError
from multiscale_feature_pilot.src.brca_q75_authorized_manifest import (
    AUTHORIZATION_RECORD_BASENAME,
    AUTHORIZED_MANIFEST_BASENAME,
    DEFAULT_AUTHORIZATION_CONFIG,
    DEFAULT_GUARDED_DIRECTORY,
    DEFAULT_Q25_RESULT,
    DEFAULT_Q50_RESULT,
    DEFAULT_REVIEW_PROVENANCE,
    EXPECTED_APPROVAL_STATEMENT_SHA256,
    EXPECTED_AUTHORIZATION_CONFIG_SHA256,
    EXPECTED_Q25_RESULT_SHA256,
    EXPECTED_Q50_RESULT_SHA256,
    EXPECTED_REVIEW_PROVENANCE_SHA256,
    GUARDED_MANIFEST_BASENAME,
    _read_regular_bytes,
    build_q75_authorized_manifest,
    validate_q75_approval_binding,
    validate_q75_authorized_manifest,
)


def test_builds_exactly_one_q75_manifest_and_keeps_later_work_locked(
    tmp_path: Path,
) -> None:
    output = tmp_path / "authorized"
    artifact = build_q75_authorized_manifest(output_directory=output)

    assert artifact.selection.label == "Q75"
    assert artifact.selection.patient_id == "TCGA-E2-A154"
    assert artifact.selection.gdc_file_uuid == "25aec062-60d1-446e-a1c6-0c79cc74a770"
    assert artifact.selection.size_bytes == 1_360_743_825
    assert artifact.selection.md5 == "a8c4b68fb6e0ab3e862efe3ed1fe10d7"
    assert artifact.sha256 == (
        "8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a"
    )
    assert {path.name for path in output.iterdir()} == {
        AUTHORIZED_MANIFEST_BASENAME,
        AUTHORIZATION_RECORD_BASENAME,
    }
    record = yaml.safe_load(
        (output / AUTHORIZATION_RECORD_BASENAME).read_text(encoding="utf-8")
    )
    assert record["download_authorized"] is True
    assert record["current_executable_scope"] == "Q75_DOWNLOAD_AND_HEADER_ONLY"
    assert record["patient_concurrency"] == 1
    assert record["active_gdc_client_processes"] == 1
    assert record["pass_n_processes_1"] is False
    assert record["q25_status"] == "SUCCESS_FROZEN_NO_RERUN"
    assert record["q50_status"] == "SUCCESS_FROZEN_NO_RERUN"
    assert record["openslide_construction_authorized"] is True
    assert record["pixel_or_region_reads_authorized"] is False
    assert record["coordinate_generation_authorized"] is False
    assert record["feature_generation_authorized"] is False
    assert record["healnet_execution_authorized"] is False
    assert record["training_authorized"] is False
    assert record["automatic_scale_policy_authorized"] is False
    assert record["use_cuda"] is False


def test_approval_binding_is_exact_and_both_predecessors_succeeded() -> None:
    binding = validate_q75_approval_binding()

    assert binding.authorization_config_sha256 == EXPECTED_AUTHORIZATION_CONFIG_SHA256
    assert binding.approval_statement_sha256 == EXPECTED_APPROVAL_STATEMENT_SHA256
    assert binding.review_provenance_sha256 == EXPECTED_REVIEW_PROVENANCE_SHA256
    assert binding.q25_result_sha256 == EXPECTED_Q25_RESULT_SHA256
    assert binding.q50_result_sha256 == EXPECTED_Q50_RESULT_SHA256


def test_exact_authorization_text_hash_and_cpu_directive_are_pinned() -> None:
    config = yaml.safe_load(DEFAULT_AUTHORIZATION_CONFIG.read_text(encoding="utf-8"))
    statement = config["approval_record"]["exact_user_statement"]
    assert hashlib.sha256(statement.encode("utf-8")).hexdigest() == (
        EXPECTED_APPROVAL_STATEMENT_SHA256
    )
    assert config["execution_mode"] == "CPU_LOGIC_ONLY_CUDA_NOT_REQUIRED"
    assert config["execution_contract"]["use_cuda"] is False


def test_checked_in_authorized_manifest_is_reproducible() -> None:
    artifact = validate_q75_authorized_manifest()
    assert artifact.path.name == AUTHORIZED_MANIFEST_BASENAME
    assert artifact.sha256 == (
        "8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a"
    )


def test_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    artifact = build_q75_authorized_manifest(output_directory=output)
    artifact.path.write_text(
        artifact.path.read_text(encoding="utf-8").replace("1360743825", "1360743826"),
        encoding="utf-8",
    )

    with pytest.raises(OneRowManifestError, match="manifest content drift"):
        validate_q75_authorized_manifest(output_directory=output)


def test_record_cannot_unlock_pixels_scale_policy_drive_or_training(
    tmp_path: Path,
) -> None:
    output = tmp_path / "authorized"
    build_q75_authorized_manifest(output_directory=output)
    record_path = output / AUTHORIZATION_RECORD_BASENAME
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    record["pixel_or_region_reads_authorized"] = True
    record["automatic_scale_policy_authorized"] = True
    record["google_drive_operations_authorized"] = True
    record["training_authorized"] = True
    record_path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="record content drift"):
        validate_q75_authorized_manifest(output_directory=output)


def test_extra_entry_or_combined_manifest_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    build_q75_authorized_manifest(output_directory=output)
    (output / "COMBINED.AUTHORIZED.gdc.tsv").write_text("forbidden\n", encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="only the Q75 manifest"):
        validate_q75_authorized_manifest(output_directory=output)


def test_symlink_manifest_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    artifact = build_q75_authorized_manifest(output_directory=output)
    referent = tmp_path / "manifest.tsv"
    referent.write_bytes(artifact.path.read_bytes())
    artifact.path.unlink()
    artifact.path.symlink_to(referent)

    with pytest.raises(OneRowManifestError, match="must not be a symlink"):
        validate_q75_authorized_manifest(output_directory=output)


@pytest.mark.parametrize(
    ("source", "argument"),
    [
        (DEFAULT_AUTHORIZATION_CONFIG, "authorization_path"),
        (DEFAULT_REVIEW_PROVENANCE, "review_provenance_path"),
        (DEFAULT_Q25_RESULT, "q25_result_path"),
        (DEFAULT_Q50_RESULT, "q50_result_path"),
    ],
)
def test_symlink_approval_source_is_rejected_by_nofollow_open(
    tmp_path: Path, source: Path, argument: str
) -> None:
    link = tmp_path / source.name
    link.symlink_to(source)

    with pytest.raises(OneRowManifestError, match="must not be a symlink"):
        validate_q75_approval_binding(**{argument: link})


def test_symlink_guarded_manifest_source_is_rejected(tmp_path: Path) -> None:
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    source = DEFAULT_GUARDED_DIRECTORY / GUARDED_MANIFEST_BASENAME
    (guarded / GUARDED_MANIFEST_BASENAME).symlink_to(source)

    with pytest.raises(OneRowManifestError, match="must not be a symlink"):
        build_q75_authorized_manifest(
            output_directory=tmp_path / "output", guarded_directory=guarded
        )
    assert not (tmp_path / "output").exists()


def test_path_swap_after_descriptor_open_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.yaml"
    displaced = tmp_path / "displaced.yaml"
    replacement = tmp_path / "replacement.yaml"
    payload = b"stable-content\n"
    source.write_bytes(payload)
    replacement.write_bytes(payload)
    original_read = os.read
    swapped = False

    def swap_after_first_read(descriptor: int, count: int) -> bytes:
        nonlocal swapped
        chunk = original_read(descriptor, count)
        if not swapped:
            swapped = True
            source.rename(displaced)
            replacement.rename(source)
        return chunk

    monkeypatch.setattr(os, "read", swap_after_first_read)
    with pytest.raises(
        OneRowManifestError, match="(changed|path identity changed) during read"
    ):
        _read_regular_bytes(source, "swap test source", maximum=1024)

    assert displaced.read_bytes() == payload
    assert source.read_bytes() == payload


def test_symlink_output_directory_fails_before_creation(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "authorized"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(OneRowManifestError, match="directory must not be a symlink"):
        build_q75_authorized_manifest(output_directory=output)
    assert not tuple(target.iterdir())


def test_authorization_byte_drift_fails_before_output(tmp_path: Path) -> None:
    authorization = tmp_path / "authorization.yaml"
    authorization.write_bytes(DEFAULT_AUTHORIZATION_CONFIG.read_bytes() + b"\n")

    with pytest.raises(OneRowManifestError, match="authorization config SHA256 drift"):
        build_q75_authorized_manifest(
            output_directory=tmp_path / "output", authorization_path=authorization
        )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("source", "keyword", "argument", "message"),
    [
        (DEFAULT_REVIEW_PROVENANCE, "review", "review_provenance_path", "CPU review"),
        (DEFAULT_Q25_RESULT, "q25", "q25_result_path", "Q25 result"),
        (DEFAULT_Q50_RESULT, "q50", "q50_result_path", "Q50 result"),
    ],
)
def test_frozen_predecessor_byte_drift_fails_before_output(
    tmp_path: Path,
    source: Path,
    keyword: str,
    argument: str,
    message: str,
) -> None:
    changed = tmp_path / f"{keyword}.yaml"
    changed.write_bytes(source.read_bytes() + b"\n")
    kwargs = {argument: changed}

    with pytest.raises(OneRowManifestError, match=rf"{message} SHA256 drift"):
        build_q75_authorized_manifest(output_directory=tmp_path / "output", **kwargs)
    assert not (tmp_path / "output").exists()


def test_gdc_concurrency_is_one_object_not_invalid_n_processes_one() -> None:
    config = yaml.safe_load(DEFAULT_AUTHORIZATION_CONFIG.read_text(encoding="utf-8"))
    download = config["authority"]["exact_wsi_download"]
    execution = config["execution_contract"]

    assert download["manifest_rows"] == 1
    assert download["patient_concurrency"] == 1
    assert download["active_gdc_client_processes"] == 1
    assert download["pass_n_processes_1"] is False
    assert execution["pass_n_processes_1"] is False
    assert execution["gdc_internal_transfer_processes"] == (
        "client_supported_default_minimum_3"
    )


def test_config_requires_all_file_checks_before_openslide_and_omic_rematch() -> None:
    config = yaml.safe_load(DEFAULT_AUTHORIZATION_CONFIG.read_text(encoding="utf-8"))
    checks = config["authority"]["post_download_verification"]["required_checks"]

    assert checks == [
        "exact_uuid",
        "exact_filename",
        "exact_byte_size",
        "expected_md5",
        "independent_sha256",
        "partial_or_incomplete_download_absent",
        "regular_file",
        "non_symlink",
        "svs_filename_suffix",
    ]
    assert config["execution_contract"]["require_exact_omic_rematch_before_recording"]
    assert config["execution_contract"]["prohibit_read_region_and_pixel_access"]
    assert config["execution_contract"]["infer_or_approve_scale_policy"] is False
