from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src.brca_one_row_manifest import OneRowManifestError
from multiscale_feature_pilot.src.brca_q25_authorized_manifest import (
    AUTHORIZATION_RECORD_BASENAME,
    AUTHORIZED_MANIFEST_BASENAME,
    DEFAULT_AUTHORIZATION_CONFIG,
    DEFAULT_METADATA_POLICY,
    EXPECTED_APPROVAL_IDENTITY_SHA256,
    EXPECTED_AUTHORIZATION_CONFIG_SHA256,
    EXPECTED_METADATA_POLICY_SHA256,
    build_q25_authorized_manifest,
    validate_approval_binding,
    validate_q25_authorized_manifest,
)


def test_builds_only_q25_and_locks_q50_q75(tmp_path: Path) -> None:
    output = tmp_path / "authorized"

    artifact = build_q25_authorized_manifest(output_directory=output)

    assert artifact.selection.label == "Q25"
    assert artifact.path.name == AUTHORIZED_MANIFEST_BASENAME
    assert {path.name for path in output.iterdir()} == {
        AUTHORIZED_MANIFEST_BASENAME,
        AUTHORIZATION_RECORD_BASENAME,
    }
    record = yaml.safe_load(
        (output / AUTHORIZATION_RECORD_BASENAME).read_text(encoding="utf-8")
    )
    assert record["download_authorized"] is True
    assert record["current_executable_scope"] == "Q25_ONLY"
    assert record["concurrency"] == 1
    assert record["q50_q75_status"] == "LOCKED_PENDING_Q25_REPORT"
    assert record["pixel_reads_authorized"] is False
    assert record["coordinate_generation_authorized"] is False
    assert record["feature_extraction_authorized"] is False
    assert record["approval_binding"] == {
        "authorization_config_sha256": EXPECTED_AUTHORIZATION_CONFIG_SHA256,
        "metadata_policy_sha256": EXPECTED_METADATA_POLICY_SHA256,
        "approval_identity_sha256": EXPECTED_APPROVAL_IDENTITY_SHA256,
    }


def test_production_authorized_manifest_is_reproducible_and_valid() -> None:
    artifact = validate_q25_authorized_manifest()

    assert artifact.selection.gdc_file_uuid == (
        "dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6"
    )
    assert artifact.selection.size_bytes == 648_046_947
    assert artifact.selection.md5 == "75536393096ffd928bc35ec9503c3655"


def test_extra_or_q50_manifest_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    build_q25_authorized_manifest(output_directory=output)
    (output / "Q50.AUTHORIZED.gdc.tsv").write_text("forbidden\n", encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="only the Q25 manifest"):
        validate_q25_authorized_manifest(output_directory=output)


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    artifact = build_q25_authorized_manifest(output_directory=output)
    artifact.path.write_text(
        artifact.path.read_text(encoding="utf-8").replace("648046947", "648046948"),
        encoding="utf-8",
    )

    with pytest.raises(OneRowManifestError, match="manifest content drift"):
        validate_q25_authorized_manifest(output_directory=output)


def test_record_cannot_authorize_pixels_or_progression(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    build_q25_authorized_manifest(output_directory=output)
    path = output / AUTHORIZATION_RECORD_BASENAME
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    record["pixel_reads_authorized"] = True
    record["q50_q75_status"] = "AUTHORIZED"
    path.write_text(yaml.safe_dump(record, sort_keys=False), encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="record content drift"):
        validate_q25_authorized_manifest(output_directory=output)


def test_symlink_manifest_fails_closed(tmp_path: Path) -> None:
    output = tmp_path / "authorized"
    artifact = build_q25_authorized_manifest(output_directory=output)
    referent = tmp_path / "manifest.tsv"
    referent.write_bytes(artifact.path.read_bytes())
    artifact.path.unlink()
    artifact.path.symlink_to(referent)

    with pytest.raises(OneRowManifestError, match="must not be a symlink"):
        validate_q25_authorized_manifest(output_directory=output)


def test_builder_denies_reverted_pending_authorization(tmp_path: Path) -> None:
    authorization = yaml.safe_load(
        DEFAULT_AUTHORIZATION_CONFIG.read_text(encoding="utf-8")
    )
    authorization["phase"] = "BRCA_PHASE_2_CPU_PREFLIGHT"
    authorization["status"] = "CPU_PREFLIGHT_AUTHORIZED_ACQUISITION_NOT_AUTHORIZED"
    path = tmp_path / "authorization.yaml"
    path.write_text(yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="approved Q25 acquisition gate"):
        build_q25_authorized_manifest(
            output_directory=tmp_path / "output",
            authorization_path=path,
        )

    assert not (tmp_path / "output").exists()


def test_builder_denies_pending_metadata_policy(tmp_path: Path) -> None:
    policy = yaml.safe_load(DEFAULT_METADATA_POLICY.read_text(encoding="utf-8"))
    policy["status"] = "PENDING_SUPERVISOR_APPROVAL"
    path = tmp_path / "policy.yaml"
    path.write_text(yaml.safe_dump(policy, sort_keys=False), encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="metadata policy status is not approved"):
        build_q25_authorized_manifest(
            output_directory=tmp_path / "output",
            metadata_policy_path=path,
        )

    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("authorization", "authorization config SHA256 drift"),
        ("policy", "metadata policy SHA256 drift"),
    ],
)
def test_semantically_equivalent_config_byte_tamper_fails_hash_binding(
    tmp_path: Path, source: str, expected: str
) -> None:
    authorization = tmp_path / "authorization.yaml"
    policy = tmp_path / "policy.yaml"
    authorization.write_bytes(DEFAULT_AUTHORIZATION_CONFIG.read_bytes())
    policy.write_bytes(DEFAULT_METADATA_POLICY.read_bytes())
    target = authorization if source == "authorization" else policy
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(OneRowManifestError, match=expected):
        validate_approval_binding(
            authorization_path=authorization,
            metadata_policy_path=policy,
        )


def test_approval_identity_tamper_fails_before_manifest_creation(tmp_path: Path) -> None:
    authorization = yaml.safe_load(
        DEFAULT_AUTHORIZATION_CONFIG.read_text(encoding="utf-8")
    )
    authorization["approval_record"]["approved_request"] += " Extra scope."
    path = tmp_path / "authorization.yaml"
    path.write_text(yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8")

    with pytest.raises(OneRowManifestError, match="approved request drift"):
        build_q25_authorized_manifest(
            output_directory=tmp_path / "output",
            authorization_path=path,
        )

    assert not (tmp_path / "output").exists()
