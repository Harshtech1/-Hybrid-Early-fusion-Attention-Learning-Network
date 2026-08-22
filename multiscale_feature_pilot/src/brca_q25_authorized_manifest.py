"""Build and validate the single currently authorized BRCA Q25 manifest.

The supervisor approved the exact Q25/Q50/Q75 candidates sequentially, but
the executable acquisition boundary advances only one slide at a time.  This
module therefore emits exactly one standard GDC manifest for Q25.  Q50 and
Q75 remain absent and locked until the Q25 size, MD5, and pyramid report has
been reviewed.

This module performs no network request and never opens a WSI.
"""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from multiscale_feature_pilot.src.brca_one_row_manifest import (
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_GUARDED_DIRECTORY,
    EXPECTED_ALIGNMENT_SHA256,
    EXPECTED_PROPOSAL_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    OneRowManifestArtifact,
    OneRowManifestError,
    OneRowSelection,
    validate_phase2_manifest_set,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHORIZATION_CONFIG = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/config/brca_phase2_authorization.yaml"
)
DEFAULT_METADATA_POLICY = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/config/brca_phase2_metadata_policy.yaml"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_phase2_q25_authorized"
)
POLICY_LABEL = "BRCA_PHASE2_Q25_SEQUENTIAL_ACQUISITION_V1"
AUTHORIZATION_STATUS = "AUTHORIZED_Q25_ONLY"
AUTHORIZATION_RECORD_BASENAME = "MANIFEST_SET.AUTHORIZED_Q25.yaml"
AUTHORIZED_MANIFEST_BASENAME = (
    "Q25_TCGA-LL-A6FP_dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6."
    "AUTHORIZED_Q25_ONLY.gdc.tsv"
)
EXPECTED_AUTHORIZATION_CONFIG_SHA256 = (
    "a3a0704b8cc56acc6b082fd969e137107af3f64f27217fd808b9e9f4a54c016b"
)
EXPECTED_METADATA_POLICY_SHA256 = (
    "daffdd77d271f28cb9440061b675d0991529f03ab99fb77bcdf5a3fb17275c43"
)
APPROVED_REQUEST_STATEMENT = (
    "Download only the exact Q25/Q50/Q75 candidates one at a time, beginning "
    "with Q25; use native pyramid levels within 10% per axis of approximately "
    "0.5 and 1.0 micrometers per pixel; reject failures without silent "
    "resampling; stop after Q25 size, MD5, and MPP/pyramid metadata."
)
EXPECTED_APPROVAL_IDENTITY_SHA256 = (
    "cf2ba45a8df674a9e83310ec26ffa9dff90d403446c71371c8cd63ae15011cee"
)


@dataclass(frozen=True)
class Q25AuthorizedManifestArtifact:
    """Validated Q25-only GDC manifest and its content identity."""

    selection: OneRowSelection
    path: Path
    sha256: str


@dataclass(frozen=True)
class ApprovalBinding:
    """Cryptographic identity of the exact authorization and policy decision."""

    authorization_config_sha256: str
    metadata_policy_sha256: str
    approval_identity_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OneRowManifestError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_yaml_mapping(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_file(), f"{label} is missing: {path}")
    _require(path.stat().st_size <= 1024 * 1024, f"{label} exceeds 1 MiB")
    payload = path.read_bytes()
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise OneRowManifestError(f"cannot parse {label}") from exc
    _require(isinstance(document, Mapping), f"{label} root must be a mapping")
    return document, payload


def _approval_identity(document: Mapping[str, Any]) -> str:
    approval = document.get("approval_record")
    _require(isinstance(approval, Mapping), "authorization approval_record is missing")
    source = approval.get("source")
    user_statement = approval.get("user_statement")
    request = approval.get("approved_request")
    _require(
        source == "USER_REPORTED_SUPERVISOR_APPROVAL",
        "authorization approval source is not the recorded supervisor decision",
    )
    _require(user_statement == "sir approved", "authorization user statement drift")
    _require(request == APPROVED_REQUEST_STATEMENT, "authorization approved request drift")
    canonical = f"{source}\n{user_statement}\n{request}".encode("utf-8")
    return _sha256_bytes(canonical)


def validate_approval_binding(
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    metadata_policy_path: Path = DEFAULT_METADATA_POLICY,
) -> ApprovalBinding:
    """Require the exact approved decision files before manifest creation."""

    authorization, authorization_bytes = _read_yaml_mapping(
        authorization_path, "BRCA Phase-2 authorization config"
    )
    _require(
        authorization.get("phase") == "BRCA_PHASE_2_Q25_ACQUISITION_GATE",
        "authorization phase is not the approved Q25 acquisition gate",
    )
    _require(
        authorization.get("status") == "Q25_ACQUISITION_AND_METADATA_AUTHORIZED",
        "authorization status is not approved",
    )
    authority = authorization.get("authority")
    _require(isinstance(authority, Mapping), "authorization authority is missing")
    exact_download = authority.get("exact_three_wsi_download")
    _require(isinstance(exact_download, Mapping), "exact download authority is missing")
    _require(
        exact_download.get("status") == "APPROVED_SEQUENTIAL",
        "exact three-slide sequential download is not approved",
    )
    _require(exact_download.get("concurrency") == 1, "approved concurrency is not one")
    _require(exact_download.get("begin_with") == "Q25", "approved first slide is not Q25")
    _require(
        exact_download.get("current_executable_scope") == "Q25_ONLY",
        "current executable scope is not Q25-only",
    )
    _require(
        exact_download.get("q50_q75_status") == "LOCKED_PENDING_Q25_REPORT",
        "Q50/Q75 progression is not locked",
    )
    metadata_authority = authority.get("wsi_open_or_metadata_read")
    _require(isinstance(metadata_authority, Mapping), "metadata authority is missing")
    _require(
        metadata_authority.get("status")
        == "APPROVED_Q25_METADATA_ONLY_AFTER_SIZE_MD5",
        "Q25 metadata-only opening is not approved",
    )
    _require(
        metadata_authority.get("pixel_or_region_read") == "NOT_AUTHORIZED",
        "pixel or region reads are not prohibited",
    )
    for field in (
        "coordinate_generation",
        "feature_extraction",
        "healnet_real_input_execution",
        "training",
        "full_cohort_acquisition",
    ):
        value = authority.get(field)
        _require(
            isinstance(value, Mapping) and value.get("status") == "NOT_AUTHORIZED",
            f"{field} must remain not authorized",
        )
    approval_identity_sha256 = _approval_identity(authorization)
    _require(
        approval_identity_sha256 == EXPECTED_APPROVAL_IDENTITY_SHA256,
        "approval identity SHA256 drift",
    )
    authorization_sha256 = _sha256_bytes(authorization_bytes)
    _require(
        authorization_sha256 == EXPECTED_AUTHORIZATION_CONFIG_SHA256,
        "authorization config SHA256 drift",
    )

    policy, policy_bytes = _read_yaml_mapping(
        metadata_policy_path, "BRCA Phase-2 metadata policy"
    )
    _require(
        policy.get("status") == "APPROVED_NATIVE_LEVEL_METADATA_GATE_V1",
        "metadata policy status is not approved",
    )
    target_policy = policy.get("target_policy")
    _require(isinstance(target_policy, Mapping), "target policy is missing")
    _require(target_policy.get("status") == "APPROVED", "target policy is not approved")
    _require(
        target_policy.get("targets_micrometers_per_pixel")
        == {"scale_2x": 0.5, "scale_4x": 1.0},
        "approved MPP targets drift",
    )
    _require(
        target_policy.get("approved_per_axis_relative_tolerance_fraction") == 0.10,
        "approved per-axis tolerance is not 10%",
    )
    resampling = policy.get("resampling_policy")
    _require(isinstance(resampling, Mapping), "resampling policy is missing")
    _require(
        resampling.get("status") == "APPROVED_NATIVE_ONLY_NO_RESAMPLING",
        "native-only no-resampling policy is not approved",
    )
    _require(
        resampling.get("silent_resampling") == "prohibited"
        and resampling.get("native_level_only") is True,
        "silent resampling is not prohibited",
    )
    policy_authority = policy.get("authority")
    _require(isinstance(policy_authority, Mapping), "metadata policy authority is missing")
    _require(
        policy_authority.get("pixel_or_region_read") == "NOT_AUTHORIZED",
        "metadata policy does not prohibit pixel reads",
    )
    metadata_policy_sha256 = _sha256_bytes(policy_bytes)
    _require(
        metadata_policy_sha256 == EXPECTED_METADATA_POLICY_SHA256,
        "metadata policy SHA256 drift",
    )
    return ApprovalBinding(
        authorization_config_sha256=authorization_sha256,
        metadata_policy_sha256=metadata_policy_sha256,
        approval_identity_sha256=approval_identity_sha256,
    )


def _write_new_or_require_equal(path: Path, payload: bytes) -> None:
    if path.exists():
        _require(not path.is_symlink(), f"output must not be a symlink: {path}")
        _require(path.is_file(), f"output must be a regular file: {path}")
        _require(path.read_bytes() == payload, f"existing output drift: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _q25_from_guarded_artifacts(
    artifacts: Sequence[OneRowManifestArtifact],
) -> OneRowManifestArtifact:
    matches = tuple(item for item in artifacts if item.selection.label == "Q25")
    _require(len(matches) == 1, "guarded source must contain exactly one Q25 row")
    return matches[0]


def _record_payload(
    selection: OneRowSelection,
    manifest_sha256: str,
    approval_binding: ApprovalBinding,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_label": POLICY_LABEL,
        "status": AUTHORIZATION_STATUS,
        "download_authorized": True,
        "authorized_label": "Q25",
        "current_executable_scope": "Q25_ONLY",
        "concurrency": 1,
        "q50_q75_status": "LOCKED_PENDING_Q25_REPORT",
        "metadata_opening": "Q25_ONLY_AFTER_EXACT_SIZE_AND_MD5",
        "pixel_reads_authorized": False,
        "coordinate_generation_authorized": False,
        "feature_extraction_authorized": False,
        "stop_after": "Q25_SIZE_MD5_AND_MPP_PYRAMID_REPORT",
        "approval_binding": {
            "authorization_config_sha256": (
                approval_binding.authorization_config_sha256
            ),
            "metadata_policy_sha256": approval_binding.metadata_policy_sha256,
            "approval_identity_sha256": approval_binding.approval_identity_sha256,
        },
        "source_hashes": {
            "proposal_sha256": EXPECTED_PROPOSAL_SHA256,
            "alignment_sha256": EXPECTED_ALIGNMENT_SHA256,
            "filtered_gdc_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        },
        "manifest_count": 1,
        "combined_manifest_present": False,
        "entry": {
            "label": selection.label,
            "patient_id": selection.patient_id,
            "basename": AUTHORIZED_MANIFEST_BASENAME,
            "status": AUTHORIZATION_STATUS,
            "rows": 1,
            "sha256": manifest_sha256,
            **selection.as_gdc_row(),
        },
    }


def _record_bytes(
    selection: OneRowSelection,
    manifest_sha256: str,
    approval_binding: ApprovalBinding,
) -> bytes:
    return yaml.safe_dump(
        _record_payload(selection, manifest_sha256, approval_binding), sort_keys=False
    ).encode("utf-8")


def validate_q25_authorized_manifest_against_selection(
    output_directory: Path,
    selection: OneRowSelection,
    *,
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    metadata_policy_path: Path = DEFAULT_METADATA_POLICY,
) -> Q25AuthorizedManifestArtifact:
    """Validate exactly one Q25 manifest against an already pinned selection."""

    approval_binding = validate_approval_binding(
        authorization_path=authorization_path,
        metadata_policy_path=metadata_policy_path,
    )
    _require(selection.label == "Q25", "authorized selection must be Q25")
    _require(
        not output_directory.is_symlink(),
        f"authorized directory must not be a symlink: {output_directory}",
    )
    _require(
        output_directory.is_dir(),
        f"authorized directory is missing: {output_directory}",
    )
    manifest_path = output_directory / AUTHORIZED_MANIFEST_BASENAME
    record_path = output_directory / AUTHORIZATION_RECORD_BASENAME
    expected_paths = {manifest_path, record_path}
    actual_paths = set(output_directory.iterdir())
    _require(
        actual_paths == expected_paths,
        "authorized directory must contain only the Q25 manifest and its record",
    )
    for path in actual_paths:
        _require(not path.is_symlink(), f"authorized output must not be a symlink: {path}")
        mode = path.stat(follow_symlinks=False).st_mode
        _require(stat.S_ISREG(mode), f"authorized output must be a regular file: {path}")

    expected_manifest = (
        "id\tfilename\tmd5\tsize\tstate\n"
        + "\t".join(selection.as_gdc_row()[field] for field in (
            "id", "filename", "md5", "size", "state"
        ))
        + "\n"
    ).encode("utf-8")
    actual_manifest = manifest_path.read_bytes()
    _require(actual_manifest == expected_manifest, "authorized Q25 manifest content drift")
    manifest_sha256 = _sha256_bytes(actual_manifest)

    try:
        record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise OneRowManifestError("cannot read authorized Q25 record") from exc
    expected_record = _record_payload(selection, manifest_sha256, approval_binding)
    _require(record == expected_record, "authorized Q25 record content drift")
    return Q25AuthorizedManifestArtifact(selection, manifest_path, manifest_sha256)


def validate_q25_authorized_manifest(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    guarded_directory: Path = DEFAULT_GUARDED_DIRECTORY,
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    metadata_policy_path: Path = DEFAULT_METADATA_POLICY,
) -> Q25AuthorizedManifestArtifact:
    """Pin all frozen sources, then validate the Q25-only authorized output."""

    guarded = validate_phase2_manifest_set(
        output_directory=guarded_directory,
        verify_source_hashes=True,
    )
    q25 = _q25_from_guarded_artifacts(guarded)
    return validate_q25_authorized_manifest_against_selection(
        output_directory,
        q25.selection,
        authorization_path=authorization_path,
        metadata_policy_path=metadata_policy_path,
    )


def build_q25_authorized_manifest(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    guarded_directory: Path = DEFAULT_GUARDED_DIRECTORY,
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    metadata_policy_path: Path = DEFAULT_METADATA_POLICY,
) -> Q25AuthorizedManifestArtifact:
    """Create the deterministic Q25-only manifest; never invoke the GDC client."""

    approval_binding = validate_approval_binding(
        authorization_path=authorization_path,
        metadata_policy_path=metadata_policy_path,
    )
    guarded = validate_phase2_manifest_set(
        output_directory=guarded_directory,
        verify_source_hashes=True,
    )
    q25 = _q25_from_guarded_artifacts(guarded)
    _require(
        not output_directory.is_symlink(),
        f"authorized directory must not be a symlink: {output_directory}",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    allowed_names = {
        AUTHORIZED_MANIFEST_BASENAME,
        AUTHORIZATION_RECORD_BASENAME,
    }
    actual_names = {path.name for path in output_directory.iterdir()}
    _require(
        actual_names.issubset(allowed_names),
        f"unexpected authorized output entries: {sorted(actual_names - allowed_names)}",
    )
    manifest_payload = q25.path.read_bytes()
    manifest_sha256 = _sha256_bytes(manifest_payload)
    _write_new_or_require_equal(
        output_directory / AUTHORIZED_MANIFEST_BASENAME,
        manifest_payload,
    )
    _write_new_or_require_equal(
        output_directory / AUTHORIZATION_RECORD_BASENAME,
        _record_bytes(q25.selection, manifest_sha256, approval_binding),
    )
    return validate_q25_authorized_manifest(
        output_directory=output_directory,
        guarded_directory=guarded_directory,
        authorization_path=authorization_path,
        metadata_policy_path=metadata_policy_path,
    )
