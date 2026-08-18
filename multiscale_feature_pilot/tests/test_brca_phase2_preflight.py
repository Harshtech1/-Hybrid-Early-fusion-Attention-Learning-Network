from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from multiscale_feature_pilot.src.brca_phase2_preflight import (
    EXPECTED_BLCA_TAG,
    EXPECTED_BLCA_TAG_COMMIT,
    EXPECTED_OFFICIAL_COMMIT,
    DisallowedSubprocessCommand,
    ExactCommandRunner,
    PreflightExpectations,
    PreflightPaths,
    inspect_staging_root,
    run_preflight,
    to_strict_json,
    validate_authorization_document,
    validate_gdc_client,
    validate_frozen_manifest_sources,
    validate_manifest_directory,
    validate_metadata_policy_document,
)


def _candidate_rows() -> list[dict[str, Any]]:
    return [
        {
            "label": "Q25",
            "patient_id": "TCGA-AA-0001",
            "wsi_uuid": "11111111-1111-4111-8111-111111111111",
            "filename": "TCGA-AA-0001-01Z-00-DX1.ONE.svs",
            "declared_bytes": 101,
            "md5": "1" * 32,
            "state": "released",
        },
        {
            "label": "Q50",
            "patient_id": "TCGA-BB-0002",
            "wsi_uuid": "22222222-2222-4222-8222-222222222222",
            "filename": "TCGA-BB-0002-01Z-00-DX1.TWO.svs",
            "declared_bytes": 202,
            "md5": "2" * 32,
            "state": "released",
        },
        {
            "label": "Q75",
            "patient_id": "TCGA-CC-0003",
            "wsi_uuid": "33333333-3333-4333-8333-333333333333",
            "filename": "TCGA-CC-0003-01Z-00-DX1.THREE.svs",
            "declared_bytes": 303,
            "md5": "3" * 32,
            "state": "released",
        },
    ]


def _authorization_document() -> dict[str, Any]:
    rows = _candidate_rows()
    return {
        "schema_version": 1,
        "phase": "BRCA_PHASE_2_CPU_PREFLIGHT",
        "status": "CPU_PREFLIGHT_AUTHORIZED_ACQUISITION_NOT_AUTHORIZED",
        "authority": {
            "supervisor_cohort_choice": {"status": "APPROVED"},
            "user_phase_2_cpu_preflight": {"status": "APPROVED"},
            "exact_three_wsi_download": {"status": "NOT_RECORDED"},
            "wsi_open_or_metadata_read": {"status": "NOT_AUTHORIZED"},
            "coordinate_generation": {"status": "NOT_AUTHORIZED"},
            "feature_extraction": {"status": "NOT_AUTHORIZED"},
            "healnet_real_input_execution": {"status": "NOT_AUTHORIZED"},
            "training": {"status": "NOT_AUTHORIZED"},
            "full_cohort_acquisition": {"status": "NOT_AUTHORIZED"},
        },
        "prohibited_during_cpu_preflight": [
            "invoke_gdc_download",
            "download_or_open_any_wsi",
            "import_openslide_in_metadata_policy_module",
            "inspect_real_wsi_pixels_or_metadata",
            "generate_coordinates",
            "run_resnet50",
            "run_real_input_healnet",
            "train",
        ],
        "proposed_candidates": {
            "selection_status": "PROPOSED_NOT_AUTHORIZED",
            "concurrency_after_future_approval": 1,
            "total_declared_bytes": sum(row["declared_bytes"] for row in rows),
            "rows": rows,
        },
    }


def _metadata_policy_document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cohort": "TCGA-BRCA",
        "phase": 2,
        "status": "PENDING_SUPERVISOR_APPROVAL",
        "authority": {
            "real_wsi_access": "NOT_AUTHORIZED",
            "slide_opening": "NOT_AUTHORIZED",
            "feature_extraction": "NOT_AUTHORIZED",
            "training": "NOT_AUTHORIZED",
        },
        "input_contract": {
            "supplied_metadata_values_only": True,
            "accepted_fields": [
                "mpp_x",
                "mpp_y",
                "level_dimensions",
                "level_downsamples",
            ],
            "file_path_input": "prohibited",
            "file_opening": "prohibited",
            "openslide_import": "prohibited",
        },
        "pyramid_validation": {
            "mpp_axes": ["x", "y"],
            "require_positive_finite_mpp": True,
            "require_positive_integer_dimensions": True,
            "require_positive_finite_downsamples": True,
            "require_matching_level_counts": True,
            "require_level_0_downsample": 1.0,
            "require_strictly_increasing_downsamples": True,
            "prohibit_growing_dimensions": True,
            "prohibit_duplicate_native_dimensions": True,
            "dimension_consistency_rule": (
                "Each native width and height must equal the floor or ceiling of the "
                "corresponding level-0 dimension divided by the reported scalar downsample."
            ),
            "native_mpp_formula": {
                "x": "mpp_x * level_downsample",
                "y": "mpp_y * level_downsample",
            },
        },
        "target_policy": {
            "status": "PENDING_SUPERVISOR_APPROVAL",
            "targets_micrometers_per_pixel": {"scale_2x": 0.5, "scale_4x": 1.0},
            "proposed_per_axis_relative_tolerance_fraction": 0.10,
            "tolerance_must_be_explicit_at_call": True,
            "nearest_level_rule": "independently_minimize_relative_error_on_each_axis",
            "tie_break_rule": "lower_native_level_index",
            "reject_if_axis_nearest_levels_disagree": True,
            "require_distinct_native_level_per_target": True,
            "missing_metadata_action": "reject",
            "invalid_metadata_action": "reject",
            "ambiguous_mapping_action": "reject",
            "out_of_tolerance_action": "reject",
        },
        "resampling_policy": {
            "status": "PENDING_SUPERVISOR_APPROVAL",
            "silent_resampling": "prohibited",
            "resampling_performed_by_preflight": False,
            "native_level_only": True,
        },
        "result_contract": {
            "validates_and_reports_metadata_only": True,
            "real_slide_authorized_by_success": False,
            "execution_enabled": False,
        },
    }


def _manifest_row(candidate: dict[str, Any]) -> dict[str, str]:
    return {
        "id": candidate["wsi_uuid"],
        "filename": candidate["filename"],
        "md5": candidate["md5"],
        "size": str(candidate["declared_bytes"]),
        "state": candidate["state"],
    }


def _manifest_basename(candidate: dict[str, Any]) -> str:
    return (
        f"{candidate['label']}_{candidate['patient_id']}_{candidate['wsi_uuid']}"
        ".NOT_AUTHORIZED.gdc.tsv"
    )


def _write_manifest_set(directory: Path, rows: list[dict[str, Any]]) -> None:
    directory.mkdir()
    entries: list[dict[str, Any]] = []
    for candidate in rows:
        basename = _manifest_basename(candidate)
        row = _manifest_row(candidate)
        text = "id\tfilename\tmd5\tsize\tstate\n" + "\t".join(
            row[field] for field in ("id", "filename", "md5", "size", "state")
        ) + "\n"
        path = directory / basename
        path.write_text(text, encoding="utf-8")
        entries.append(
            {
                "label": candidate["label"],
                "patient_id": candidate["patient_id"],
                "basename": basename,
                "status": "NOT_AUTHORIZED",
                "rows": 1,
                "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                **row,
            }
        )
    record = {
        "schema_version": 1,
        "policy_label": "BRCA_PHASE_2_EXACT_THREE_WSI_NOT_AUTHORIZED",
        "status": "NOT_AUTHORIZED",
        "download_authorized": False,
        "metadata_only": True,
        "source_hashes": {"authorization": "fixture"},
        "manifest_count": 3,
        "combined_manifest_present": False,
        "entries": entries,
    }
    (directory / "MANIFEST_SET.NOT_AUTHORIZED.yaml").write_text(
        yaml.safe_dump(record, sort_keys=False), encoding="utf-8"
    )


def _fake_elf(path: Path) -> str:
    header = bytearray(64)
    header[:4] = b"\x7fELF"
    header[4] = 2
    header[5] = 1
    header[18:20] = (62).to_bytes(2, "little")
    path.write_bytes(bytes(header) + b"fixture-gdc-client")
    path.chmod(0o755)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FixtureRunner:
    def __init__(self, official_repo: Path, pilot_repo: Path, client: Path) -> None:
        self.official_repo = official_repo
        self.pilot_repo = pilot_repo
        self.client = client
        self.commands: list[tuple[str, ...]] = []

    def __call__(self, command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        self.commands.append(tuple(command))
        if command[:2] == ["file", "--brief"]:
            return subprocess.CompletedProcess(
                command, 0, "ELF 64-bit LSB executable, x86-64\n", ""
            )
        if command == [str(self.client), "--version"]:
            return subprocess.CompletedProcess(command, 0, "2.3\n", "")
        if command[:2] == ["git", "-C"]:
            repository = Path(command[2])
            arguments = command[3:]
            if repository == self.official_repo and arguments == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(
                    command, 0, EXPECTED_OFFICIAL_COMMIT + "\n", ""
                )
            if repository == self.official_repo and arguments == [
                "status",
                "--porcelain",
            ]:
                return subprocess.CompletedProcess(command, 0, "", "")
            if repository == self.pilot_repo and arguments == [
                "rev-parse",
                f"{EXPECTED_BLCA_TAG}^{{}}",
            ]:
                return subprocess.CompletedProcess(
                    command, 0, EXPECTED_BLCA_TAG_COMMIT + "\n", ""
                )
        return subprocess.CompletedProcess(command, 1, "", "unexpected fixture command")


class _FixtureManifestSourceValidator:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.calls: list[tuple[Path, bool]] = []

    def __call__(
        self,
        *,
        output_directory: Path,
        verify_source_hashes: bool,
    ):
        self.calls.append((output_directory, verify_source_hashes))
        artifacts = []
        for row in self.rows:
            path = output_directory / _manifest_basename(row)
            artifacts.append(
                SimpleNamespace(
                    selection=SimpleNamespace(
                        label=row["label"],
                        patient_id=row["patient_id"],
                        gdc_file_uuid=row["wsi_uuid"],
                        filename=row["filename"],
                        md5=row["md5"],
                        size_bytes=row["declared_bytes"],
                        state=row["state"],
                    ),
                    path=path,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                )
            )
        return tuple(artifacts)


def _fixture_preflight(tmp_path: Path):
    authorization = tmp_path / "authorization.yaml"
    policy = tmp_path / "policy.yaml"
    manifests = tmp_path / "manifests"
    client = tmp_path / "gdc-client"
    official_repo = tmp_path / "official"
    pilot_repo = tmp_path / "pilot"
    staging = tmp_path / "staging"
    authorization.write_text(
        yaml.safe_dump(_authorization_document(), sort_keys=False), encoding="utf-8"
    )
    policy.write_text(
        yaml.safe_dump(_metadata_policy_document(), sort_keys=False), encoding="utf-8"
    )
    _write_manifest_set(manifests, _candidate_rows())
    client_sha256 = _fake_elf(client)
    official_repo.mkdir()
    pilot_repo.mkdir()
    paths = PreflightPaths(
        authorization=authorization,
        metadata_policy=policy,
        manifests_directory=manifests,
        gdc_client=client,
        official_repo=official_repo,
        pilot_repo=pilot_repo,
        staging_root=staging,
    )
    runner = _FixtureRunner(official_repo, pilot_repo, client)
    manifest_validator = _FixtureManifestSourceValidator(_candidate_rows())
    expectations = PreflightExpectations(gdc_client_sha256=client_sha256)
    return paths, runner, expectations, manifest_validator


def test_authorization_document_is_technically_valid_but_not_authorized() -> None:
    result = validate_authorization_document(_authorization_document())

    assert result["ok"]
    assert result["candidate_count"] == 3
    assert result["candidate_labels"] == ["Q25", "Q50", "Q75"]
    assert not result["acquisition_authorized"]


def test_authorization_rejects_duplicate_uuid_and_wrong_total() -> None:
    document = _authorization_document()
    document["proposed_candidates"]["rows"][1]["wsi_uuid"] = document[
        "proposed_candidates"
    ]["rows"][0]["wsi_uuid"]
    document["proposed_candidates"]["total_declared_bytes"] = 1

    result = validate_authorization_document(document)

    assert not result["ok"]
    assert any("UUIDs must be unique" in error for error in result["errors"])
    assert any("total_declared_bytes" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("top_status", "status must equal"),
        ("download_status", "exact_three_wsi_download.status"),
        ("selection_status", "selection_status"),
    ],
)
def test_authorization_phase_state_is_immutable(
    mutation: str,
    expected_error: str,
) -> None:
    document = _authorization_document()
    if mutation == "top_status":
        document["status"] = "APPROVED"
    elif mutation == "download_status":
        document["authority"]["exact_three_wsi_download"]["status"] = "APPROVED"
    else:
        document["proposed_candidates"]["selection_status"] = "APPROVED"

    result = validate_authorization_document(document)

    assert not result["ok"]
    assert not result["acquisition_authorized"]
    assert any(expected_error in error for error in result["errors"])


@pytest.mark.parametrize(
    "field",
    [
        "wsi_open_or_metadata_read",
        "coordinate_generation",
        "feature_extraction",
        "healnet_real_input_execution",
        "training",
        "full_cohort_acquisition",
    ],
)
def test_every_real_authority_must_remain_not_authorized(field: str) -> None:
    document = _authorization_document()
    document["authority"][field]["status"] = "APPROVED"

    result = validate_authorization_document(document)

    assert not result["ok"]
    assert any(f"authority.{field}.status" in error for error in result["errors"])


def test_metadata_policy_accepts_pending_as_valid_but_not_approved() -> None:
    result = validate_metadata_policy_document(_metadata_policy_document())

    assert result["ok"]
    assert not result["decision_approved"]


def test_metadata_policy_rejects_file_opening_and_silent_resampling() -> None:
    document = _metadata_policy_document()
    document["input_contract"]["file_opening"] = "allowed"
    document["resampling_policy"]["silent_resampling"] = "allowed"

    result = validate_metadata_policy_document(document)

    assert not result["ok"]
    assert any("file_opening" in error for error in result["errors"])
    assert any("silent_resampling" in error for error in result["errors"])


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("root", "status", "APPROVED"),
        ("authority", "real_wsi_access", "APPROVED"),
        ("target_policy", "status", "APPROVED"),
        ("target_policy", "proposed_per_axis_relative_tolerance_fraction", 0.11),
        ("target_policy", "nearest_level_rule", "choose_any_level"),
        ("target_policy", "tie_break_rule", "higher_native_level_index"),
        ("target_policy", "reject_if_axis_nearest_levels_disagree", False),
        ("target_policy", "require_distinct_native_level_per_target", False),
        ("resampling_policy", "status", "APPROVED"),
        ("resampling_policy", "native_level_only", False),
        ("resampling_policy", "resampling_performed_by_preflight", True),
        ("pyramid_validation", "dimension_consistency_rule", "approximately close"),
        ("pyramid_validation", "native_mpp_formula", {"x": "wrong", "y": "wrong"}),
    ],
)
def test_metadata_policy_rule_mutations_fail_closed(
    section: str,
    field: str,
    value: object,
) -> None:
    document = _metadata_policy_document()
    if section == "root":
        document[field] = value
    else:
        document[section][field] = value

    result = validate_metadata_policy_document(document)

    assert not result["ok"]
    assert not result["decision_approved"]
    assert any(field in error for error in result["errors"])


def test_manifest_set_matches_authorization_exactly(tmp_path: Path) -> None:
    rows = _candidate_rows()
    directory = tmp_path / "manifests"
    _write_manifest_set(directory, rows)

    result = validate_manifest_directory(
        directory,
        [_manifest_row(row) for row in rows],
        [_manifest_basename(row) for row in rows],
    )

    assert result["ok"]
    assert result["manifest_count"] == 3
    assert result["rows_match_authorization"]
    assert result["manifest_set_record_ok"]
    assert result["authorization_lock"] == "NOT_AUTHORIZED"


def test_manifest_tampering_fails_closed(tmp_path: Path) -> None:
    rows = _candidate_rows()
    directory = tmp_path / "manifests"
    _write_manifest_set(directory, rows)
    first = directory / _manifest_basename(rows[0])
    first.write_text(first.read_text(encoding="utf-8").replace("\t101\t", "\t102\t"))

    result = validate_manifest_directory(
        directory,
        [_manifest_row(row) for row in rows],
        [_manifest_basename(row) for row in rows],
    )

    assert not result["ok"]
    assert not result["rows_match_authorization"]
    assert any("sha256" in error for error in result["errors"])


def test_manifest_directory_rejects_subdirectories(tmp_path: Path) -> None:
    rows = _candidate_rows()
    directory = tmp_path / "manifests"
    _write_manifest_set(directory, rows)
    (directory / "unexpected_directory").mkdir()

    result = validate_manifest_directory(
        directory,
        [_manifest_row(row) for row in rows],
        [_manifest_basename(row) for row in rows],
    )

    assert not result["ok"]
    assert any("unexpected entries" in error for error in result["errors"])


def test_manifest_directory_rejects_expected_name_fifo_before_read(
    tmp_path: Path,
) -> None:
    rows = _candidate_rows()
    directory = tmp_path / "manifests"
    _write_manifest_set(directory, rows)
    fifo = directory / _manifest_basename(rows[0])
    fifo.unlink()
    fifo_path = str(fifo)
    os.mkfifo(fifo_path)

    result = validate_manifest_directory(
        directory,
        [_manifest_row(row) for row in rows],
        [_manifest_basename(row) for row in rows],
    )

    assert not result["ok"]
    assert any("regular file" in error for error in result["errors"])


def test_production_manifest_validator_pins_all_frozen_sources() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    authorization_document = yaml.safe_load(
        (
            repository_root
            / "multiscale_feature_pilot/config/brca_phase2_authorization.yaml"
        ).read_text(encoding="utf-8")
    )
    authorization = validate_authorization_document(authorization_document)

    result = validate_frozen_manifest_sources(
        repository_root
        / "multiscale_feature_pilot/provenance/brca_phase2_manifests",
        authorization,
    )

    assert result["ok"]
    assert result["frozen_sources_verified"]
    assert result["artifact_count"] == 3
    assert result["validator"] == "validate_phase2_manifest_set"
    assert result["proposal_sha256"] == (
        "b1bbeb06bb200813122e0b1a88a3d0258660eaff067d35f1d8de1dd6d79badb2"
    )
    assert result["alignment_sha256"] == (
        "13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe"
    )
    assert result["official_filtered_manifest_sha256"] == (
        "ac1b4d591ce255a7d4e6bde3778a041b107576693657ee296fe6eb36c4e79a92"
    )


def test_wrong_client_hash_prevents_any_execution(tmp_path: Path) -> None:
    client = tmp_path / "gdc-client"
    _fake_elf(client)
    called = False

    def forbidden_runner(*_: Any, **__: Any):
        nonlocal called
        called = True
        raise AssertionError("a mismatching executable must never be run")

    result = validate_gdc_client(
        client,
        expected_sha256="0" * 64,
        expected_version="2.3",
        runner=forbidden_runner,
    )

    assert not result["ok"]
    assert not called


def test_staging_scan_rejects_every_entry_without_opening(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "ordinary.txt").write_bytes(b"")
    (staging / "subdirectory").mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"")
    (staging / "link").symlink_to(target)

    result = inspect_staging_root(staging)

    assert not result["ok"]
    assert result["entries"] == ["link", "ordinary.txt", "subdirectory"]
    assert not result["wsi_content_opened"]


def test_staging_scan_accepts_only_absent_or_completely_empty(tmp_path: Path) -> None:
    absent = inspect_staging_root(tmp_path / "absent")
    empty_path = tmp_path / "empty"
    empty_path.mkdir()
    empty = inspect_staging_root(empty_path)

    assert absent["ok"] and absent["empty"]
    assert empty["ok"] and empty["empty"]


def test_exact_command_runner_rejects_every_unlisted_subprocess() -> None:
    delegate_called = False

    def delegate(*_: Any, **__: Any):
        nonlocal delegate_called
        delegate_called = True
        return subprocess.CompletedProcess([], 0, "", "")

    runner = ExactCommandRunner(delegate, [("git", "--version")])

    with pytest.raises(DisallowedSubprocessCommand):
        runner(["gdc-client", "download", "forbidden"])

    assert not delegate_called
    assert runner.executed_commands == []
    assert runner.rejected_commands == [("gdc-client", "download", "forbidden")]


def test_current_expected_state_is_preflight_ready_but_not_download_ready(
    tmp_path: Path,
) -> None:
    paths, runner, expectations, manifest_validator = _fixture_preflight(tmp_path)

    result = run_preflight(
        paths,
        expectations=expectations,
        runner=runner,
        manifest_source_validator=manifest_validator,
    )

    assert result["cpu_preflight_ready"]
    assert result["frozen_sources_verified"]
    assert not result["acquisition_authorized"]
    assert not result["ready_to_download"]
    assert result["checks"]["metadata_policy"]["status"] == (
        "PENDING_SUPERVISOR_APPROVAL"
    )
    assert result["checks"]["manifests"]["authorization_lock"] == "NOT_AUTHORIZED"
    assert manifest_validator.calls == [(paths.manifests_directory, True)]
    assert result["checks"]["subprocess_allowlist"][
        "exact_expected_sequence_executed"
    ]
    assert "storage" not in result["checks"]
    assert result["safety"] == {
        "network_requests_performed": False,
        "download_invocation_constructed": False,
        "wsi_opened": False,
        "openslide_imported": False,
        "coordinates_generated": False,
        "features_extracted": False,
        "training_performed": False,
    }
    serialized = to_strict_json(result)
    assert json.loads(serialized)["cpu_preflight_ready"] is True


def test_yaml_approval_alone_cannot_release_safety_locked_manifests(
    tmp_path: Path,
) -> None:
    paths, runner, expectations, manifest_validator = _fixture_preflight(tmp_path)
    authorization = yaml.safe_load(paths.authorization.read_text(encoding="utf-8"))
    authorization["authority"]["exact_three_wsi_download"]["status"] = "APPROVED"
    authorization["proposed_candidates"]["selection_status"] = "APPROVED"
    paths.authorization.write_text(
        yaml.safe_dump(authorization, sort_keys=False), encoding="utf-8"
    )

    result = run_preflight(
        paths,
        expectations=expectations,
        runner=runner,
        manifest_source_validator=manifest_validator,
    )

    assert not result["cpu_preflight_ready"]
    assert not result["acquisition_authorized"]
    assert not result["ready_to_download"]
    assert any("exact_three_wsi_download" in blocker for blocker in result["blockers"])


def test_any_staged_slide_makes_cpu_preflight_fail(tmp_path: Path) -> None:
    paths, runner, expectations, manifest_validator = _fixture_preflight(tmp_path)
    paths.staging_root.mkdir()
    (paths.staging_root / "unexpected.svs").write_bytes(b"")

    result = run_preflight(
        paths,
        expectations=expectations,
        runner=runner,
        manifest_source_validator=manifest_validator,
    )

    assert not result["cpu_preflight_ready"]
    assert not result["ready_to_download"]
    assert result["checks"]["staging"]["entries"] == ["unexpected.svs"]
