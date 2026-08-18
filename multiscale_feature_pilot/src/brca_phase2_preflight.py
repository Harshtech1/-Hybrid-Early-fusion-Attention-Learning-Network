"""Read-only CPU preflight for the guarded BRCA Phase-2 transition.

This module validates configuration, one-row manifest candidates, source
identities, the external GDC client, and an empty staging boundary.  It never
performs a network request, constructs a download invocation, imports
OpenSlide, or opens a WSI.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from multiscale_feature_pilot.src.brca_one_row_manifest import (
    EXPECTED_ALIGNMENT_SHA256,
    EXPECTED_PROPOSAL_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    OneRowManifestError,
    validate_phase2_manifest_set,
)


EXPECTED_GDC_CLIENT_SHA256 = (
    "1df9281cbacbb53413607a4b9b682947dcb48b6ac7fade6527748bc025ae8c96"
)
EXPECTED_GDC_CLIENT_VERSION = "2.3"
EXPECTED_OFFICIAL_COMMIT = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
EXPECTED_BLCA_TAG = "blca-one-patient-pilot-v1"
EXPECTED_BLCA_TAG_COMMIT = "df7cf2bda783ab6cc09e95d6a1fa0914da05a433"
EXPECTED_LABELS = ("Q25", "Q50", "Q75")
MANIFEST_COLUMNS = ("id", "filename", "md5", "size", "state")

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_PATIENT_RE = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
ManifestSourceValidator = Callable[..., Sequence[Any]]


class DisallowedSubprocessCommand(OSError):
    """Raised before any subprocess outside the exact local probe set runs."""


class ExactCommandRunner:
    """Permit only an exact, precomputed set of read-only local commands."""

    def __init__(
        self,
        delegate: CommandRunner,
        allowed_commands: Sequence[Sequence[str]],
    ) -> None:
        self._delegate = delegate
        self.allowed_commands = tuple(tuple(command) for command in allowed_commands)
        self.executed_commands: list[tuple[str, ...]] = []
        self.rejected_commands: list[tuple[str, ...]] = []

    def __call__(
        self,
        command: Sequence[str],
        **kwargs: Any,
    ) -> subprocess.CompletedProcess[str]:
        normalized = tuple(command)
        if normalized not in self.allowed_commands:
            self.rejected_commands.append(normalized)
            raise DisallowedSubprocessCommand(
                "subprocess command is outside the BRCA Phase-2 preflight allowlist"
            )
        self.executed_commands.append(normalized)
        return self._delegate(list(command), **kwargs)


@dataclass(frozen=True, slots=True)
class PreflightPaths:
    """All filesystem inputs inspected by the preflight."""

    authorization: Path
    metadata_policy: Path
    manifests_directory: Path
    gdc_client: Path
    official_repo: Path
    pilot_repo: Path
    staging_root: Path


@dataclass(frozen=True, slots=True)
class PreflightExpectations:
    """Pinned identities, injectable only to support isolated fixture tests."""

    gdc_client_sha256: str = EXPECTED_GDC_CLIENT_SHA256
    gdc_client_version: str = EXPECTED_GDC_CLIENT_VERSION
    official_commit: str = EXPECTED_OFFICIAL_COMMIT
    blca_tag: str = EXPECTED_BLCA_TAG
    blca_tag_commit: str = EXPECTED_BLCA_TAG_COMMIT


def default_paths(repository_root: Path) -> PreflightPaths:
    """Return the pinned project layout without creating any paths."""

    root = repository_root.resolve()
    workspace = root.parent
    return PreflightPaths(
        authorization=root
        / "multiscale_feature_pilot"
        / "config"
        / "brca_phase2_authorization.yaml",
        metadata_policy=root
        / "multiscale_feature_pilot"
        / "config"
        / "brca_phase2_metadata_policy.yaml",
        manifests_directory=root
        / "multiscale_feature_pilot"
        / "provenance"
        / "brca_phase2_manifests",
        gdc_client=workspace / "tools" / "gdc-client" / "2.3.0" / "gdc-client",
        official_repo=workspace / "healnet",
        pilot_repo=root,
        staging_root=workspace / "brca_pilot_data",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_yaml_mapping(path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        if path.is_symlink():
            return None, ["configuration path must not be a symlink"]
        if not path.is_file():
            return None, ["configuration file is missing"]
        if path.stat().st_size > 1024 * 1024:
            return None, ["configuration file exceeds the 1 MiB safety limit"]
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return None, [f"configuration could not be read: {type(exc).__name__}"]
    if not isinstance(loaded, dict):
        errors.append("configuration root must be a mapping")
        return None, errors
    return loaded, errors


def _mapping(value: object, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping")
        return {}
    return value


def _require_equal(
    actual: object,
    expected: object,
    name: str,
    errors: list[str],
) -> None:
    if actual != expected:
        errors.append(f"{name} must equal {expected!r}")


def validate_authorization_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the exact three-candidate authorization boundary."""

    errors: list[str] = []
    _require_equal(document.get("schema_version"), 1, "schema_version", errors)
    _require_equal(
        document.get("phase"),
        "BRCA_PHASE_2_CPU_PREFLIGHT",
        "phase",
        errors,
    )
    _require_equal(
        document.get("status"),
        "CPU_PREFLIGHT_AUTHORIZED_ACQUISITION_NOT_AUTHORIZED",
        "status",
        errors,
    )

    authority = _mapping(document.get("authority"), "authority", errors)
    supervisor = _mapping(
        authority.get("supervisor_cohort_choice"),
        "authority.supervisor_cohort_choice",
        errors,
    )
    cpu_approval = _mapping(
        authority.get("user_phase_2_cpu_preflight"),
        "authority.user_phase_2_cpu_preflight",
        errors,
    )
    _require_equal(
        supervisor.get("status"),
        "APPROVED",
        "authority.supervisor_cohort_choice.status",
        errors,
    )
    _require_equal(
        cpu_approval.get("status"),
        "APPROVED",
        "authority.user_phase_2_cpu_preflight.status",
        errors,
    )

    exact_download = _mapping(
        authority.get("exact_three_wsi_download"),
        "authority.exact_three_wsi_download",
        errors,
    )
    exact_download_status = exact_download.get("status")
    _require_equal(
        exact_download_status,
        "NOT_RECORDED",
        "authority.exact_three_wsi_download.status",
        errors,
    )

    for field in (
        "wsi_open_or_metadata_read",
        "coordinate_generation",
        "feature_extraction",
        "healnet_real_input_execution",
        "training",
        "full_cohort_acquisition",
    ):
        entry = _mapping(authority.get(field), f"authority.{field}", errors)
        status_value = entry.get("status")
        _require_equal(
            status_value,
            "NOT_AUTHORIZED",
            f"authority.{field}.status",
            errors,
        )

    prohibited = document.get("prohibited_during_cpu_preflight")
    required_prohibitions = {
        "invoke_gdc_download",
        "download_or_open_any_wsi",
        "import_openslide_in_metadata_policy_module",
        "inspect_real_wsi_pixels_or_metadata",
        "generate_coordinates",
        "run_resnet50",
        "run_real_input_healnet",
        "train",
    }
    if not isinstance(prohibited, list) or not all(
        isinstance(item, str) for item in prohibited
    ):
        errors.append("prohibited_during_cpu_preflight must be a list of strings")
    elif not required_prohibitions.issubset(set(prohibited)):
        errors.append("prohibited_during_cpu_preflight omits a required safety boundary")

    proposed = _mapping(document.get("proposed_candidates"), "proposed_candidates", errors)
    selection_status = proposed.get("selection_status")
    _require_equal(
        selection_status,
        "PROPOSED_NOT_AUTHORIZED",
        "proposed_candidates.selection_status",
        errors,
    )
    _require_equal(
        proposed.get("concurrency_after_future_approval"),
        1,
        "proposed_candidates.concurrency_after_future_approval",
        errors,
    )

    raw_rows = proposed.get("rows")
    rows: list[dict[str, Any]] = []
    if not isinstance(raw_rows, list) or len(raw_rows) != 3:
        errors.append("proposed_candidates.rows must contain exactly three rows")
    else:
        for index, raw_row in enumerate(raw_rows):
            row_mapping = _mapping(raw_row, f"proposed_candidates.rows[{index}]", errors)
            if row_mapping:
                rows.append(dict(row_mapping))

    labels: list[object] = []
    patient_ids: list[object] = []
    uuids: list[object] = []
    filenames: list[object] = []
    declared_sizes: list[int] = []
    for index, row in enumerate(rows):
        prefix = f"proposed_candidates.rows[{index}]"
        label = row.get("label")
        patient_id = row.get("patient_id")
        file_uuid = row.get("wsi_uuid")
        filename = row.get("filename")
        declared_bytes = row.get("declared_bytes")
        md5 = row.get("md5")
        state = row.get("state")

        labels.append(label)
        patient_ids.append(patient_id)
        uuids.append(file_uuid)
        filenames.append(filename)
        if not isinstance(patient_id, str) or not _PATIENT_RE.fullmatch(patient_id):
            errors.append(f"{prefix}.patient_id is not a canonical TCGA patient ID")
        if not isinstance(file_uuid, str) or not _UUID_RE.fullmatch(file_uuid):
            errors.append(f"{prefix}.wsi_uuid is not a canonical lowercase UUID")
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".svs")
            or not isinstance(patient_id, str)
            or not filename.startswith(patient_id + "-")
        ):
            errors.append(f"{prefix}.filename is not the expected patient-bound SVS basename")
        if isinstance(declared_bytes, bool) or not isinstance(declared_bytes, int):
            errors.append(f"{prefix}.declared_bytes must be a positive integer")
        elif declared_bytes <= 0:
            errors.append(f"{prefix}.declared_bytes must be positive")
        else:
            declared_sizes.append(declared_bytes)
        if not isinstance(md5, str) or not _MD5_RE.fullmatch(md5):
            errors.append(f"{prefix}.md5 must be 32 lowercase hexadecimal characters")
        if state != "released":
            errors.append(f"{prefix}.state must equal 'released'")

    if tuple(labels) != EXPECTED_LABELS:
        errors.append("candidate labels/order must be exactly Q25, Q50, Q75")
    for name, values in (
        ("patient IDs", patient_ids),
        ("WSI UUIDs", uuids),
        ("filenames", filenames),
    ):
        if len(values) != len(set(values)):
            errors.append(f"candidate {name} must be unique")
    if len(declared_sizes) == 3:
        _require_equal(
            proposed.get("total_declared_bytes"),
            sum(declared_sizes),
            "proposed_candidates.total_declared_bytes",
            errors,
        )

    # This checker freezes the pre-approval phase.  A future authorization
    # requires a separate reviewed transition, not mutation of these records.
    acquisition_authorized = False
    return {
        "ok": not errors,
        "errors": errors,
        "candidate_count": len(rows),
        "candidate_labels": labels,
        "total_declared_bytes": sum(declared_sizes) if len(declared_sizes) == 3 else None,
        "exact_three_wsi_download_status": exact_download_status,
        "selection_status": selection_status,
        "acquisition_authorized": acquisition_authorized,
        "rows": rows,
    }


def validate_metadata_policy_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a deterministic, metadata-only policy while allowing PENDING."""

    errors: list[str] = []
    _require_equal(document.get("schema_version"), 1, "schema_version", errors)
    _require_equal(document.get("cohort"), "TCGA-BRCA", "cohort", errors)
    _require_equal(document.get("phase"), 2, "phase", errors)
    policy_status = document.get("status")
    _require_equal(
        policy_status,
        "PENDING_SUPERVISOR_APPROVAL",
        "status",
        errors,
    )

    authority = _mapping(document.get("authority"), "authority", errors)
    for field in ("real_wsi_access", "slide_opening", "feature_extraction", "training"):
        _require_equal(
            authority.get(field),
            "NOT_AUTHORIZED",
            f"authority.{field}",
            errors,
        )

    input_contract = _mapping(document.get("input_contract"), "input_contract", errors)
    _require_equal(
        input_contract.get("supplied_metadata_values_only"),
        True,
        "input_contract.supplied_metadata_values_only",
        errors,
    )
    _require_equal(
        input_contract.get("accepted_fields"),
        ["mpp_x", "mpp_y", "level_dimensions", "level_downsamples"],
        "input_contract.accepted_fields",
        errors,
    )
    for field in ("file_path_input", "file_opening", "openslide_import"):
        _require_equal(
            input_contract.get(field),
            "prohibited",
            f"input_contract.{field}",
            errors,
        )

    pyramid = _mapping(document.get("pyramid_validation"), "pyramid_validation", errors)
    _require_equal(pyramid.get("mpp_axes"), ["x", "y"], "pyramid_validation.mpp_axes", errors)
    for field in (
        "require_positive_finite_mpp",
        "require_positive_integer_dimensions",
        "require_positive_finite_downsamples",
        "require_matching_level_counts",
        "require_strictly_increasing_downsamples",
        "prohibit_growing_dimensions",
        "prohibit_duplicate_native_dimensions",
    ):
        _require_equal(pyramid.get(field), True, f"pyramid_validation.{field}", errors)
    _require_equal(
        pyramid.get("require_level_0_downsample"),
        1.0,
        "pyramid_validation.require_level_0_downsample",
        errors,
    )
    _require_equal(
        pyramid.get("dimension_consistency_rule"),
        (
            "Each native width and height must equal the floor or ceiling of the "
            "corresponding level-0 dimension divided by the reported scalar downsample."
        ),
        "pyramid_validation.dimension_consistency_rule",
        errors,
    )
    _require_equal(
        pyramid.get("native_mpp_formula"),
        {
            "x": "mpp_x * level_downsample",
            "y": "mpp_y * level_downsample",
        },
        "pyramid_validation.native_mpp_formula",
        errors,
    )

    targets = _mapping(document.get("target_policy"), "target_policy", errors)
    target_status = targets.get("status")
    _require_equal(
        target_status,
        "PENDING_SUPERVISOR_APPROVAL",
        "target_policy.status",
        errors,
    )
    target_mpp = _mapping(
        targets.get("targets_micrometers_per_pixel"),
        "target_policy.targets_micrometers_per_pixel",
        errors,
    )
    _require_equal(target_mpp.get("scale_2x"), 0.5, "target_policy scale_2x", errors)
    _require_equal(target_mpp.get("scale_4x"), 1.0, "target_policy scale_4x", errors)
    tolerance = targets.get("proposed_per_axis_relative_tolerance_fraction")
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
        errors.append(
            "target_policy.proposed_per_axis_relative_tolerance_fraction "
            "must equal 0.10"
        )
    elif float(tolerance) != 0.10:
        errors.append(
            "target_policy.proposed_per_axis_relative_tolerance_fraction "
            "must equal 0.10"
        )
    _require_equal(
        targets.get("tolerance_must_be_explicit_at_call"),
        True,
        "target_policy.tolerance_must_be_explicit_at_call",
        errors,
    )
    _require_equal(
        targets.get("nearest_level_rule"),
        "independently_minimize_relative_error_on_each_axis",
        "target_policy.nearest_level_rule",
        errors,
    )
    _require_equal(
        targets.get("tie_break_rule"),
        "lower_native_level_index",
        "target_policy.tie_break_rule",
        errors,
    )
    _require_equal(
        targets.get("reject_if_axis_nearest_levels_disagree"),
        True,
        "target_policy.reject_if_axis_nearest_levels_disagree",
        errors,
    )
    _require_equal(
        targets.get("require_distinct_native_level_per_target"),
        True,
        "target_policy.require_distinct_native_level_per_target",
        errors,
    )
    for field in (
        "missing_metadata_action",
        "invalid_metadata_action",
        "ambiguous_mapping_action",
        "out_of_tolerance_action",
    ):
        _require_equal(targets.get(field), "reject", f"target_policy.{field}", errors)

    resampling = _mapping(document.get("resampling_policy"), "resampling_policy", errors)
    resampling_status = resampling.get("status")
    _require_equal(
        resampling_status,
        "PENDING_SUPERVISOR_APPROVAL",
        "resampling_policy.status",
        errors,
    )
    _require_equal(
        resampling.get("silent_resampling"),
        "prohibited",
        "resampling_policy.silent_resampling",
        errors,
    )
    _require_equal(
        resampling.get("resampling_performed_by_preflight"),
        False,
        "resampling_policy.resampling_performed_by_preflight",
        errors,
    )
    _require_equal(
        resampling.get("native_level_only"),
        True,
        "resampling_policy.native_level_only",
        errors,
    )

    result_contract = _mapping(document.get("result_contract"), "result_contract", errors)
    _require_equal(
        result_contract.get("validates_and_reports_metadata_only"),
        True,
        "result_contract.validates_and_reports_metadata_only",
        errors,
    )
    _require_equal(
        result_contract.get("real_slide_authorized_by_success"),
        False,
        "result_contract.real_slide_authorized_by_success",
        errors,
    )
    _require_equal(
        result_contract.get("execution_enabled"),
        False,
        "result_contract.execution_enabled",
        errors,
    )

    decision_approved = False
    return {
        "ok": not errors,
        "errors": errors,
        "status": policy_status,
        "target_policy_status": target_status,
        "resampling_policy_status": resampling_status,
        "decision_approved": decision_approved,
    }


def _authorization_manifest_rows(
    authorization_result: Mapping[str, Any],
) -> tuple[dict[str, str], ...]:
    expected: list[dict[str, str]] = []
    rows = authorization_result.get("rows")
    if not isinstance(rows, list):
        return ()
    for row in rows:
        if not isinstance(row, Mapping):
            return ()
        try:
            expected.append(
                {
                    "id": str(row["wsi_uuid"]),
                    "filename": str(row["filename"]),
                    "md5": str(row["md5"]),
                    "size": str(row["declared_bytes"]),
                    "state": str(row["state"]),
                }
            )
        except KeyError:
            return ()
    return tuple(expected)


def _authorization_manifest_basenames(
    authorization_result: Mapping[str, Any],
) -> tuple[str, ...]:
    rows = authorization_result.get("rows")
    if not isinstance(rows, list):
        return ()
    basenames: list[str] = []
    for row in rows:
        if not isinstance(row, Mapping):
            return ()
        try:
            basenames.append(
                f"{row['label']}_{row['patient_id']}_{row['wsi_uuid']}"
                ".NOT_AUTHORIZED.gdc.tsv"
            )
        except KeyError:
            return ()
    return tuple(basenames)


def validate_manifest_directory(
    directory: Path,
    expected_rows: Sequence[Mapping[str, str]],
    expected_basenames: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validate exactly three safety-locked, standard one-row GDC manifests."""

    errors: list[str] = []
    parsed_rows: list[dict[str, str]] = []
    parsed_by_basename: dict[str, dict[str, str]] = {}
    manifest_sha256: dict[str, str] = {}
    filenames: list[str] = []
    record_path = directory / "MANIFEST_SET.NOT_AUTHORIZED.yaml"
    if directory.is_symlink():
        errors.append("manifest directory must not be a symlink")
        candidates: list[Path] = []
    elif not directory.is_dir():
        errors.append("manifest directory is missing")
        candidates = []
    else:
        candidates = sorted(directory.glob("*.NOT_AUTHORIZED.gdc.tsv"))
        all_entries = sorted(directory.iterdir())
        allowed_entries = set(candidates) | {record_path}
        if set(all_entries) != allowed_entries:
            errors.append("manifest directory contains unexpected entries")
    if len(candidates) != 3:
        errors.append("manifest directory must contain exactly three NOT_AUTHORIZED manifests")

    for path in candidates:
        filenames.append(path.name)
        if path.is_symlink():
            errors.append(f"{path.name}: manifest must not be a symlink")
            continue
        try:
            path_stat = path.lstat()
            if not stat.S_ISREG(path_stat.st_mode):
                errors.append(f"{path.name}: manifest must be a regular file")
                continue
            if path_stat.st_size > 64 * 1024:
                errors.append(f"{path.name}: manifest exceeds the 64 KiB safety limit")
                continue
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"{path.name}: could not read manifest ({type(exc).__name__})")
            continue
        manifest_sha256[path.name] = _sha256(path)
        if len(text.splitlines()) != 2:
            errors.append(f"{path.name}: manifest must contain one header and one data row")
            continue
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            errors.append(f"{path.name}: columns must be {MANIFEST_COLUMNS}")
            continue
        rows = list(reader)
        if len(rows) != 1:
            errors.append(f"{path.name}: manifest must contain exactly one row")
            continue
        row = rows[0]
        if None in row or any(value is None or value == "" for value in row.values()):
            errors.append(f"{path.name}: manifest row contains blank or surplus fields")
            continue
        if not _UUID_RE.fullmatch(row["id"]):
            errors.append(f"{path.name}: id is not a canonical lowercase UUID")
        if Path(row["filename"]).name != row["filename"] or not row["filename"].endswith(
            ".svs"
        ):
            errors.append(f"{path.name}: filename is not an SVS basename")
        if not _MD5_RE.fullmatch(row["md5"]):
            errors.append(f"{path.name}: md5 is invalid")
        try:
            parsed_size = int(row["size"])
        except ValueError:
            errors.append(f"{path.name}: size is not an integer")
        else:
            if parsed_size <= 0 or str(parsed_size) != row["size"]:
                errors.append(f"{path.name}: size must be a canonical positive integer")
        if row["state"] != "released":
            errors.append(f"{path.name}: state must equal 'released'")
        parsed_row = dict(row)
        parsed_rows.append(parsed_row)
        parsed_by_basename[path.name] = parsed_row

    if expected_basenames is not None and tuple(sorted(filenames)) != tuple(
        sorted(expected_basenames)
    ):
        errors.append("manifest basenames do not exactly match the authorization candidates")

    normalized_expected = tuple(
        sorted((dict(row) for row in expected_rows), key=lambda row: row.get("id", ""))
    )
    normalized_actual = tuple(sorted(parsed_rows, key=lambda row: row.get("id", "")))
    if normalized_actual != normalized_expected:
        errors.append("manifest rows do not exactly match the authorization candidates")
    ids = [row.get("id") for row in parsed_rows]
    if len(ids) != len(set(ids)):
        errors.append("manifest UUIDs must be unique")

    record_document, record_read_errors = _read_yaml_mapping(record_path)
    errors.extend(f"manifest-set record: {error}" for error in record_read_errors)
    record_ok = record_document is not None
    if record_document is not None:
        if record_document.get("schema_version") != 1:
            errors.append("manifest-set record: schema_version must equal 1")
        record_status = record_document.get("status")
        if not isinstance(record_status, str) or "NOT_AUTHORIZED" not in record_status:
            errors.append("manifest-set record: status must remain NOT_AUTHORIZED")
        if record_document.get("download_authorized") is not False:
            errors.append("manifest-set record: download_authorized must be false")
        if record_document.get("metadata_only") is not True:
            errors.append("manifest-set record: metadata_only must be true")
        if record_document.get("manifest_count") != 3:
            errors.append("manifest-set record: manifest_count must equal 3")
        if record_document.get("combined_manifest_present") is not False:
            errors.append("manifest-set record: combined_manifest_present must be false")
        record_entries = record_document.get("entries")
        if not isinstance(record_entries, list) or len(record_entries) != 3:
            errors.append("manifest-set record: entries must contain exactly three rows")
        else:
            record_basenames: list[object] = []
            for index, record_entry in enumerate(record_entries):
                entry = _mapping(
                    record_entry,
                    f"manifest-set record entries[{index}]",
                    errors,
                )
                basename = entry.get("basename")
                record_basenames.append(basename)
                if entry.get("status") != "NOT_AUTHORIZED":
                    errors.append(
                        f"manifest-set record entries[{index}].status must be NOT_AUTHORIZED"
                    )
                if entry.get("rows") != 1:
                    errors.append(
                        f"manifest-set record entries[{index}].rows must equal 1"
                    )
                if not isinstance(basename, str) or basename not in parsed_by_basename:
                    errors.append(
                        f"manifest-set record entries[{index}].basename is not a manifest"
                    )
                    continue
                expected_row = parsed_by_basename[basename]
                for field in MANIFEST_COLUMNS:
                    if str(entry.get(field)) != expected_row[field]:
                        errors.append(
                            f"manifest-set record entries[{index}].{field} "
                            "does not match its TSV"
                        )
                if entry.get("sha256") != manifest_sha256.get(basename):
                    errors.append(
                        f"manifest-set record entries[{index}].sha256 does not match its TSV"
                    )
            if tuple(sorted(str(value) for value in record_basenames)) != tuple(
                sorted(filenames)
            ):
                errors.append("manifest-set record basenames do not match the TSV set")
        record_ok = not any(error.startswith("manifest-set record") for error in errors)

    return {
        "ok": not errors,
        "errors": errors,
        "directory": str(directory.resolve(strict=False)),
        "manifest_count": len(candidates),
        "manifest_filenames": filenames,
        "rows_match_authorization": normalized_actual == normalized_expected,
        "manifest_set_record": str(record_path.resolve(strict=False)),
        "manifest_set_record_ok": record_ok,
        "authorization_lock": "NOT_AUTHORIZED",
    }


def validate_frozen_manifest_sources(
    directory: Path,
    authorization_result: Mapping[str, Any],
    *,
    validator: ManifestSourceValidator = validate_phase2_manifest_set,
) -> dict[str, Any]:
    """Independently pin proposal, alignment, official manifest, and outputs."""

    errors: list[str] = []
    artifacts: Sequence[Any] = ()
    try:
        artifacts = validator(
            output_directory=directory,
            verify_source_hashes=True,
        )
    except (OneRowManifestError, OSError, TypeError, ValueError) as exc:
        errors.append(f"frozen manifest-source validation failed: {type(exc).__name__}")

    artifact_rows: list[dict[str, str]] = []
    artifact_summaries: list[dict[str, Any]] = []
    if not errors:
        if not isinstance(artifacts, Sequence) or len(artifacts) != 3:
            errors.append("frozen manifest-source validator must return exactly three artifacts")
        else:
            for index, artifact in enumerate(artifacts):
                try:
                    selection = artifact.selection
                    artifact_row = {
                        "id": str(selection.gdc_file_uuid),
                        "filename": str(selection.filename),
                        "md5": str(selection.md5),
                        "size": str(selection.size_bytes),
                        "state": str(selection.state),
                    }
                    artifact_path = Path(artifact.path)
                    artifact_sha256 = str(artifact.sha256)
                except (AttributeError, TypeError, ValueError):
                    errors.append(
                        f"frozen manifest-source artifact {index} has an invalid contract"
                    )
                    continue
                artifact_rows.append(artifact_row)
                artifact_summaries.append(
                    {
                        "label": str(selection.label),
                        "patient_id": str(selection.patient_id),
                        "wsi_uuid": str(selection.gdc_file_uuid),
                        "basename": artifact_path.name,
                        "sha256": artifact_sha256,
                    }
                )
            expected_rows = _authorization_manifest_rows(authorization_result)
            normalized_artifacts = tuple(
                sorted(artifact_rows, key=lambda row: row.get("id", ""))
            )
            normalized_expected = tuple(
                sorted((dict(row) for row in expected_rows), key=lambda row: row["id"])
            )
            if normalized_artifacts != normalized_expected:
                errors.append(
                    "frozen proposal/alignment/source-manifest candidates do not match authorization"
                )

    return {
        "ok": not errors,
        "errors": errors,
        "frozen_sources_verified": not errors,
        "proposal_sha256": EXPECTED_PROPOSAL_SHA256,
        "alignment_sha256": EXPECTED_ALIGNMENT_SHA256,
        "official_filtered_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        "artifact_count": len(artifact_summaries),
        "artifacts": artifact_summaries,
        "validator": getattr(validator, "__name__", type(validator).__name__),
    }


def validate_gdc_client(
    path: Path,
    *,
    expected_sha256: str,
    expected_version: str,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Validate the pinned client before executing only its local version probe."""

    errors: list[str] = []
    resolved = path.resolve(strict=False)
    sha256: str | None = None
    header: bytes | None = None
    is_executable = False
    if path.is_symlink():
        errors.append("GDC client must not be a symlink")
    elif not path.is_file():
        errors.append("GDC client is missing")
    else:
        mode = path.stat().st_mode
        is_executable = bool(mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH))
        if not is_executable or not os.access(path, os.X_OK):
            errors.append("GDC client is not executable")
        sha256 = _sha256(path)
        if sha256 != expected_sha256:
            errors.append("GDC client SHA256 does not match the pinned executable")
        with path.open("rb") as stream:
            header = stream.read(20)

    elf64_x86_64 = bool(
        header
        and len(header) >= 20
        and header[:4] == b"\x7fELF"
        and header[4] == 2
        and header[5] == 1
        and int.from_bytes(header[18:20], "little") == 62
    )
    if path.is_file() and not elf64_x86_64:
        errors.append("GDC client is not a little-endian ELF64 x86-64 executable")

    file_description: str | None = None
    version: str | None = None
    # Run probes only after content, executable mode, and ELF identity pass.
    if not errors:
        try:
            file_result = runner(
                ["file", "--brief", str(path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"file probe failed: {type(exc).__name__}")
        else:
            file_description = (file_result.stdout or "").strip() or None
            if (
                file_result.returncode != 0
                or file_description is None
                or "ELF 64-bit" not in file_description
                or "x86-64" not in file_description
            ):
                errors.append("file probe did not identify a 64-bit x86-64 ELF")

    if not errors:
        try:
            version_result = runner(
                [str(path), "--version"],
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f"version probe failed: {type(exc).__name__}")
        else:
            version = (version_result.stdout or "").strip() or None
            if version_result.returncode != 0 or version != expected_version:
                errors.append(f"GDC client version must equal {expected_version!r}")

    return {
        "ok": not errors,
        "errors": errors,
        "path": str(resolved),
        "sha256": sha256,
        "expected_sha256": expected_sha256,
        "executable": is_executable,
        "elf64_x86_64": elf64_x86_64,
        "file_description": file_description,
        "version": version,
        "expected_version": expected_version,
    }


def _run_git(
    runner: CommandRunner,
    repository: Path,
    arguments: Sequence[str],
) -> tuple[str | None, str | None]:
    try:
        result = runner(
            ["git", "-C", str(repository), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, type(exc).__name__
    if result.returncode != 0:
        return None, (result.stderr or "git command failed").strip()
    return result.stdout.strip(), None


def validate_source_identities(
    official_repo: Path,
    pilot_repo: Path,
    *,
    expected_official_commit: str,
    blca_tag: str,
    expected_blca_tag_commit: str,
    runner: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Verify the pinned official checkout and frozen BLCA tag read-only."""

    errors: list[str] = []
    official_head, head_error = _run_git(runner, official_repo, ["rev-parse", "HEAD"])
    if head_error:
        errors.append("official HEALNet HEAD could not be resolved")
    elif official_head != expected_official_commit:
        errors.append("official HEALNet HEAD does not match v0.1.0")

    official_status, status_error = _run_git(
        runner, official_repo, ["status", "--porcelain"]
    )
    if status_error:
        errors.append("official HEALNet worktree status could not be read")
    elif official_status:
        errors.append("official HEALNet worktree is not clean")

    peeled_ref = f"{blca_tag}^{{}}"
    tag_commit, tag_error = _run_git(runner, pilot_repo, ["rev-parse", peeled_ref])
    if tag_error:
        errors.append("frozen BLCA tag could not be resolved")
    elif tag_commit != expected_blca_tag_commit:
        errors.append("frozen BLCA tag does not resolve to its pinned commit")

    return {
        "ok": not errors,
        "errors": errors,
        "official_repo": str(official_repo.resolve(strict=False)),
        "official_head": official_head,
        "expected_official_head": expected_official_commit,
        "official_worktree_clean": official_status == "" if not status_error else False,
        "pilot_repo": str(pilot_repo.resolve(strict=False)),
        "frozen_blca_tag": blca_tag,
        "frozen_blca_tag_commit": tag_commit,
        "expected_frozen_blca_tag_commit": expected_blca_tag_commit,
    }


def inspect_staging_root(staging_root: Path) -> dict[str, Any]:
    """Require the designated root to be absent or completely empty."""

    errors: list[str] = []
    entries: list[str] = []
    exists = os.path.lexists(staging_root)
    if staging_root.is_symlink():
        errors.append("staging root must not be a symlink")
    elif exists and not staging_root.is_dir():
        errors.append("staging root exists but is not a directory")
    elif exists:
        try:
            with os.scandir(staging_root) as iterator:
                entries = sorted(entry.name for entry in iterator)
        except OSError as exc:
            errors.append(f"staging root could not be inspected: {type(exc).__name__}")
    if entries:
        errors.append("staging root must be absent or completely empty")
    return {
        "ok": not errors,
        "errors": errors,
        "path": str(staging_root.resolve(strict=False)),
        "exists": exists,
        "entries": entries,
        "empty": bool(not entries and not errors),
        "wsi_content_opened": False,
    }


def expected_subprocess_commands(
    paths: PreflightPaths,
    expectations: PreflightExpectations,
) -> tuple[tuple[str, ...], ...]:
    """Return the complete local read-only subprocess allowlist in run order."""

    return (
        ("file", "--brief", str(paths.gdc_client)),
        (str(paths.gdc_client), "--version"),
        ("git", "-C", str(paths.official_repo), "rev-parse", "HEAD"),
        ("git", "-C", str(paths.official_repo), "status", "--porcelain"),
        (
            "git",
            "-C",
            str(paths.pilot_repo),
            "rev-parse",
            f"{expectations.blca_tag}^{{}}",
        ),
    )


def run_preflight(
    paths: PreflightPaths,
    *,
    expectations: PreflightExpectations = PreflightExpectations(),
    runner: CommandRunner = subprocess.run,
    manifest_source_validator: ManifestSourceValidator = validate_phase2_manifest_set,
) -> dict[str, Any]:
    """Run every local, read-only check and return a JSON-safe result."""

    auth_document, auth_read_errors = _read_yaml_mapping(paths.authorization)
    if auth_document is None:
        authorization: dict[str, Any] = {
            "ok": False,
            "errors": auth_read_errors,
            "acquisition_authorized": False,
            "rows": [],
        }
    else:
        authorization = validate_authorization_document(auth_document)
        authorization["errors"] = auth_read_errors + list(authorization["errors"])
        authorization["ok"] = not authorization["errors"]
    authorization["path"] = str(paths.authorization.resolve(strict=False))

    policy_document, policy_read_errors = _read_yaml_mapping(paths.metadata_policy)
    if policy_document is None:
        metadata_policy: dict[str, Any] = {
            "ok": False,
            "errors": policy_read_errors,
            "decision_approved": False,
        }
    else:
        metadata_policy = validate_metadata_policy_document(policy_document)
        metadata_policy["errors"] = policy_read_errors + list(metadata_policy["errors"])
        metadata_policy["ok"] = not metadata_policy["errors"]
    metadata_policy["path"] = str(paths.metadata_policy.resolve(strict=False))

    expected_manifest_rows = _authorization_manifest_rows(authorization)
    expected_manifest_basenames = _authorization_manifest_basenames(authorization)
    manifests = validate_manifest_directory(
        paths.manifests_directory,
        expected_manifest_rows,
        expected_manifest_basenames,
    )
    frozen_manifest_sources = validate_frozen_manifest_sources(
        paths.manifests_directory,
        authorization,
        validator=manifest_source_validator,
    )
    allowed_commands = expected_subprocess_commands(paths, expectations)
    audited_runner = ExactCommandRunner(runner, allowed_commands)
    gdc_client = validate_gdc_client(
        paths.gdc_client,
        expected_sha256=expectations.gdc_client_sha256,
        expected_version=expectations.gdc_client_version,
        runner=audited_runner,
    )
    source_identities = validate_source_identities(
        paths.official_repo,
        paths.pilot_repo,
        expected_official_commit=expectations.official_commit,
        blca_tag=expectations.blca_tag,
        expected_blca_tag_commit=expectations.blca_tag_commit,
        runner=audited_runner,
    )
    staging = inspect_staging_root(paths.staging_root)

    executed_commands = tuple(audited_runner.executed_commands)
    subprocess_allowlist = {
        "ok": bool(
            not audited_runner.rejected_commands
            and executed_commands == allowed_commands
        ),
        "errors": [],
        "expected_commands": [list(command) for command in allowed_commands],
        "executed_commands": [list(command) for command in executed_commands],
        "rejected_commands": [
            list(command) for command in audited_runner.rejected_commands
        ],
        "exact_expected_sequence_executed": executed_commands == allowed_commands,
    }
    if audited_runner.rejected_commands:
        subprocess_allowlist["errors"].append(
            "a subprocess outside the exact local read-only allowlist was rejected"
        )
    if executed_commands != allowed_commands:
        subprocess_allowlist["errors"].append(
            "the exact expected local probe sequence was not executed"
        )

    checks = {
        "authorization": authorization,
        "metadata_policy": metadata_policy,
        "manifests": manifests,
        "frozen_manifest_sources": frozen_manifest_sources,
        "gdc_client": gdc_client,
        "source_identities": source_identities,
        "staging": staging,
        "subprocess_allowlist": subprocess_allowlist,
    }
    cpu_preflight_ready = all(bool(check.get("ok")) for check in checks.values())
    acquisition_authorized = bool(
        authorization.get("ok") and authorization.get("acquisition_authorized")
    )
    policy_approved = bool(
        metadata_policy.get("ok") and metadata_policy.get("decision_approved")
    )
    manifest_lock_released = manifests.get("authorization_lock") == "AUTHORIZED"
    ready_to_download = bool(
        cpu_preflight_ready
        and acquisition_authorized
        and policy_approved
        and manifest_lock_released
    )

    blockers: list[str] = []
    for check_name, check in checks.items():
        for error in check.get("errors", []):
            blockers.append(f"{check_name}: {error}")
    if not acquisition_authorized:
        blockers.append(
            "acquisition authorization is not recorded for the exact three candidates"
        )
    if not policy_approved:
        blockers.append("MPP/resampling policy remains pending or unapproved")
    if not manifest_lock_released:
        blockers.append("candidate manifests remain explicitly NOT_AUTHORIZED")

    return {
        "schema_version": 1,
        "phase": "BRCA_PHASE_2_CPU_PREFLIGHT",
        "mode": "LOCAL_READ_ONLY_NO_NETWORK_NO_WSI",
        "checks": checks,
        "frozen_sources_verified": bool(
            frozen_manifest_sources.get("frozen_sources_verified")
        ),
        "cpu_preflight_ready": cpu_preflight_ready,
        "acquisition_authorized": acquisition_authorized,
        "ready_to_download": ready_to_download,
        "blockers": blockers,
        "safety": {
            "network_requests_performed": False,
            "download_invocation_constructed": False,
            "wsi_opened": False,
            "openslide_imported": False,
            "coordinates_generated": False,
            "features_extracted": False,
            "training_performed": False,
        },
    }


def to_strict_json(result: Mapping[str, Any]) -> str:
    """Serialize without NaN/Infinity and with deterministic key ordering."""

    return json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
