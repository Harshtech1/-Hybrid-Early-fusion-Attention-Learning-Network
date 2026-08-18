from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src.brca_one_row_manifest import OneRowManifestError
from multiscale_feature_pilot.src.brca_q50_authorized_manifest import (
    AUTHORIZATION_RECORD_BASENAME,
    AUTHORIZED_MANIFEST_BASENAME,
    DEFAULT_AUTHORIZATION_CONFIG,
    DEFAULT_Q25_FEATURE_MANIFEST,
    DEFAULT_Q25_RESULT,
    EXPECTED_APPROVAL_IDENTITY_SHA256,
    EXPECTED_AUTHORIZATION_CONFIG_SHA256,
    EXPECTED_Q25_FEATURE_MANIFEST_SHA256,
    EXPECTED_Q25_RESULT_SHA256,
    build_q50_authorized_manifest,
    validate_q50_approval_binding,
    validate_q50_authorized_manifest,
)


def test_builds_exactly_one_q50_manifest_and_keeps_later_work_locked(
    tmp_path: Path,
) -> None:
    output = tmp_path / "authorized"
    artifact = build_q50_authorized_manifest(output_directory=output)

    assert artifact.selection.label == "Q50"
    assert artifact.selection.patient_id == "TCGA-AR-A1AW"
    assert artifact.selection.gdc_file_uuid == "5c1216f3-19ec-4d3c-9bb0-9bd740b79f62"
    assert artifact.selection.size_bytes == 975_626_387
    assert artifact.selection.md5 == "304509e03f26cbecc9aee4ea691c8e5a"
    assert {path.name for path in output.iterdir()} == {
        AUTHORIZED_MANIFEST_BASENAME,
        AUTHORIZATION_RECORD_BASENAME,
    }
    record = yaml.safe_load(
        (output / AUTHORIZATION_RECORD_BASENAME).read_text(encoding="utf-8")
    )
    assert record["download_authorized"] is True
    assert record["current_executable_scope"] == "Q50_ONLY"
    assert record["q25_status"] == "SUCCESS_FROZEN_NO_RERUN"
    assert record["q75_status"] == "NOT_AUTHORIZED"
    assert record["pixel_reads_authorized"] is False
    assert record["coordinate_generation_authorized"] is False
    assert record["feature_extraction_authorized"] is False
    assert record["healnet_execution_authorized"] is False
    assert record["training_authorized"] is False
    assert record["google_drive_required"] is False
    assert record["google_drive_operations_authorized"] is False


def test_approval_binding_is_exact_and_predecessor_is_complete() -> None:
    binding = validate_q50_approval_binding()

    assert binding.authorization_config_sha256 == EXPECTED_AUTHORIZATION_CONFIG_SHA256
    assert binding.approval_identity_sha256 == EXPECTED_APPROVAL_IDENTITY_SHA256
    assert binding.q25_result_sha256 == EXPECTED_Q25_RESULT_SHA256
    assert (
        binding.q25_feature_manifest_sha256
        == EXPECTED_Q25_FEATURE_MANIFEST_SHA256
    )


def test_checked_in_authorized_manifest_is_reproducible() -> None:
    artifact = validate_q50_authorized_manifest()
    assert artifact.path.name == AUTHORIZED_MANIFEST_BASENAME
    assert artifact.sha256 == (
        "1028665e49bab6895a47947d069e225c2d6b4b90f420f7a9dc916ca9313a6062"
    )


def test_manifest_tamper_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    artifact = build_q50_authorized_manifest(output_directory=output)
    artifact.path.write_text(
        artifact.path.read_text(encoding="utf-8").replace("975626387", "975626388"),
        encoding="utf-8",
    )

    with pytest.raises(OneRowManifestError, match="manifest content drift"):
        validate_q50_authorized_manifest(output_directory=output)


def test_record_cannot_unlock_q75_pixels_or_training(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    build_q50_authorized_manifest(output_directory=output)
    record_path = output / AUTHORIZATION_RECORD_BASENAME
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    record["q75_status"] = "AUTHORIZED"
    record["pixel_reads_authorized"] = True
    record["training_authorized"] = True
    record_path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="record content drift"):
        validate_q50_authorized_manifest(output_directory=output)


def test_extra_entry_or_q75_manifest_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    build_q50_authorized_manifest(output_directory=output)
    (output / "Q75.AUTHORIZED.gdc.tsv").write_text("forbidden\n", encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="only the Q50 manifest"):
        validate_q50_authorized_manifest(output_directory=output)


def test_symlink_manifest_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    artifact = build_q50_authorized_manifest(output_directory=output)
    referent = tmp_path / "manifest.tsv"
    referent.write_bytes(artifact.path.read_bytes())
    artifact.path.unlink()
    artifact.path.symlink_to(referent)

    with pytest.raises(OneRowManifestError, match="must not be a symlink"):
        validate_q50_authorized_manifest(output_directory=output)


def test_symlink_output_directory_fails_before_creation(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    output = tmp_path / "authorized"
    output.symlink_to(target, target_is_directory=True)

    with pytest.raises(OneRowManifestError, match="directory must not be a symlink"):
        build_q50_authorized_manifest(output_directory=output)
    assert not tuple(target.iterdir())


def test_authorization_byte_drift_fails_closed(tmp_path: Path) -> None:
    authorization = tmp_path / "authorization.yaml"
    authorization.write_bytes(DEFAULT_AUTHORIZATION_CONFIG.read_bytes() + b"\n")

    with pytest.raises(OneRowManifestError, match="authorization config SHA256 drift"):
        build_q50_authorized_manifest(
            output_directory=tmp_path / "output",
            authorization_path=authorization,
        )
    assert not (tmp_path / "output").exists()


def test_q25_result_drift_fails_before_output(tmp_path: Path) -> None:
    result = tmp_path / "q25_result.yaml"
    result.write_bytes(DEFAULT_Q25_RESULT.read_bytes() + b"\n")

    with pytest.raises(OneRowManifestError, match="Q25 result SHA256 drift"):
        build_q50_authorized_manifest(
            output_directory=tmp_path / "output", q25_result_path=result
        )
    assert not (tmp_path / "output").exists()


def test_q25_feature_manifest_drift_fails_before_output(tmp_path: Path) -> None:
    manifest = json.loads(DEFAULT_Q25_FEATURE_MANIFEST.read_text(encoding="utf-8"))
    manifest["contract"]["combined_rows"] = 9323
    path = tmp_path / "feature_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="feature manifest SHA256 drift"):
        build_q50_authorized_manifest(
            output_directory=tmp_path / "output", q25_feature_manifest_path=path
        )
    assert not (tmp_path / "output").exists()


def test_config_keeps_drive_optional_and_q25_scale_mapping_forbidden() -> None:
    config = yaml.safe_load(DEFAULT_AUTHORIZATION_CONFIG.read_text(encoding="utf-8"))
    assert config["storage_contract"]["google_drive_required"] is False
    assert config["storage_contract"]["google_drive_operations_authorized"] is False
    assert config["execution_contract"]["reuse_q25_level_indices_or_scale_mapping"] is False
    assert config["authority"]["q75_processing"]["status"] == "NOT_AUTHORIZED"
    assert config["authority"]["training"]["status"] == "NOT_AUTHORIZED"
