"""Build and validate the exact BRCA Q50-only GDC manifest.

This is an append-only transition after the successful Q25 pilot.  It permits
one Q50 GDC download and, only after exact size and MD5 verification, a
header-only OpenSlide metadata inspection.  It performs no network request,
does not open a WSI, and keeps Q75, pixels, coordinates, features, HEALNet,
and training prohibited.
"""

from __future__ import annotations

import hashlib
import json
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
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DEFAULT_AUTHORIZATION_CONFIG = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/config/brca_q50_acquisition_authorization.yaml"
)
DEFAULT_Q25_RESULT = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q25_gpu_pilot_result.yaml"
)
DEFAULT_Q25_FEATURE_MANIFEST = (
    WORKSPACE_ROOT / "brca_pilot_data/Q25.features/feature_manifest.json"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_phase2_q50_authorized"
)

POLICY_LABEL = "BRCA_Q50_SEQUENTIAL_ACQUISITION_V1"
AUTHORIZATION_STATUS = "AUTHORIZED_Q50_ONLY"
AUTHORIZATION_RECORD_BASENAME = "MANIFEST_SET.AUTHORIZED_Q50.yaml"
AUTHORIZED_MANIFEST_BASENAME = (
    "Q50_TCGA-AR-A1AW_5c1216f3-19ec-4d3c-9bb0-9bd740b79f62."
    "AUTHORIZED_Q50_ONLY.gdc.tsv"
)

EXPECTED_AUTHORIZATION_CONFIG_SHA256 = (
    "00e2d41323f44a8deec97d3dad3d83bb87ef6700b6a2408253976257e52fec40"
)
EXPECTED_APPROVAL_IDENTITY_SHA256 = (
    "36d82521bd3dada2219452a19844cdf033bf73149571251e0d884fd3fedb0599"
)
EXPECTED_Q25_RESULT_SHA256 = (
    "76df862c438f4d94e34342b54c80c0866a440af0574be9003444075d7073c6e4"
)
EXPECTED_Q25_FEATURE_MANIFEST_SHA256 = (
    "88a3146d34cf907604cecf109ff3879b26afcf6458ce86a9e2e46a7e127dcc0f"
)
APPROVAL_STATEMENT = (
    "Authorize and start the Q50-only BRCA pilot. Do not train or process "
    "Q75 yet."
)
STORAGE_STATEMENT = "can we do without Google drive if possible for now"

EXPECTED_Q50 = {
    "label": "Q50",
    "patient_id": "TCGA-AR-A1AW",
    "id": "5c1216f3-19ec-4d3c-9bb0-9bd740b79f62",
    "filename": (
        "TCGA-AR-A1AW-01Z-00-DX1."
        "E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs"
    ),
    "md5": "304509e03f26cbecc9aee4ea691c8e5a",
    "size": "975626387",
    "state": "released",
}
EXPECTED_PROHIBITED_ACTIONS = [
    "q75_download_open_or_processing",
    "model_training",
    "optimizer_or_backward_execution",
    "coordinate_generation",
    "resnet50_feature_extraction",
    "healnet_real_input_execution",
    "full_cohort_download_or_processing",
    "google_drive_mount_upload_or_delete",
    "raw_wsi_deletion",
    "official_healnet_modification",
    "frozen_blca_commit_or_tag_modification",
]


@dataclass(frozen=True)
class Q50ApprovalBinding:
    authorization_config_sha256: str
    approval_identity_sha256: str
    q25_result_sha256: str
    q25_feature_manifest_sha256: str


@dataclass(frozen=True)
class Q50AuthorizedManifestArtifact:
    selection: OneRowSelection
    path: Path
    sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OneRowManifestError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_bytes(path: Path, label: str, *, maximum: int) -> bytes:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    try:
        mode = path.stat(follow_symlinks=False).st_mode
    except OSError as exc:
        raise OneRowManifestError(f"{label} is missing: {path}") from exc
    _require(stat.S_ISREG(mode), f"{label} must be a regular file")
    _require(path.stat().st_size <= maximum, f"{label} exceeds size limit")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise OneRowManifestError(f"cannot read {label}: {path}") from exc


def _yaml_mapping(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise OneRowManifestError(f"cannot parse {label}") from exc
    _require(isinstance(document, Mapping), f"{label} root must be a mapping")
    return document


def _json_mapping(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise OneRowManifestError(f"cannot parse {label}") from exc
    _require(isinstance(document, Mapping), f"{label} root must be a mapping")
    return document


def _approval_identity(approval: Mapping[str, Any]) -> str:
    source = approval.get("source")
    statement = approval.get("user_statement")
    storage_source = approval.get("storage_decision_source")
    storage_statement = approval.get("storage_user_statement")
    _require(source == "DIRECT_USER_AUTHORIZATION", "Q50 approval source drift")
    _require(statement == APPROVAL_STATEMENT, "Q50 approval statement drift")
    _require(
        storage_source == "DIRECT_USER_AUTHORIZATION",
        "Q50 storage decision source drift",
    )
    _require(storage_statement == STORAGE_STATEMENT, "Q50 storage statement drift")
    canonical = "\n".join(
        (str(source), str(statement), str(storage_source), str(storage_statement))
    ).encode("utf-8")
    return _sha256_bytes(canonical)


def _validate_q25_predecessor(result: Mapping[str, Any]) -> None:
    _require(
        result.get("status") == "BRCA_Q25_GPU_FEATURE_PILOT_SUCCESS",
        "Q25 predecessor did not succeed",
    )
    _require(result.get("candidate") == "Q25", "Q25 predecessor candidate drift")
    artifacts = result.get("artifacts")
    _require(isinstance(artifacts, Mapping), "Q25 predecessor artifacts are missing")
    manifest = artifacts.get("manifest")
    _require(isinstance(manifest, Mapping), "Q25 feature manifest record is missing")
    _require(
        manifest.get("sha256") == EXPECTED_Q25_FEATURE_MANIFEST_SHA256,
        "Q25 feature manifest identity drift",
    )
    postconditions = result.get("postconditions")
    _require(isinstance(postconditions, Mapping), "Q25 postconditions are missing")
    _require(postconditions.get("required_stop_reached") is True, "Q25 stop not reached")
    for field in ("training_runs", "backward_passes", "optimizer_steps"):
        _require(postconditions.get(field) == 0, f"Q25 {field} must be zero")
    _require(
        postconditions.get("q50_q75_operations") == 0,
        "Q25 result already contains Q50/Q75 operations",
    )


def _validate_q25_feature_manifest(manifest: Mapping[str, Any]) -> None:
    metadata = manifest.get("metadata")
    _require(isinstance(metadata, Mapping), "Q25 feature metadata is missing")
    identity = metadata.get("identity")
    _require(isinstance(identity, Mapping), "Q25 feature identity is missing")
    _require(identity.get("candidate_label") == "Q25", "Q25 feature candidate drift")
    contract = manifest.get("contract")
    _require(isinstance(contract, Mapping), "Q25 feature contract is missing")
    _require(contract.get("combined_rows") == 9322, "Q25 combined rows drift")
    _require(contract.get("feature_dim") == 2048, "Q25 feature dimension drift")
    _require(contract.get("finite") is True, "Q25 feature contract is not finite")


def validate_q50_approval_binding(
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    q25_result_path: Path = DEFAULT_Q25_RESULT,
    q25_feature_manifest_path: Path = DEFAULT_Q25_FEATURE_MANIFEST,
) -> Q50ApprovalBinding:
    """Validate the direct approval and completed Q25 predecessor."""

    authorization_bytes = _read_regular_bytes(
        authorization_path, "Q50 authorization config", maximum=1024 * 1024
    )
    authorization_sha256 = _sha256_bytes(authorization_bytes)
    _require(
        authorization_sha256 == EXPECTED_AUTHORIZATION_CONFIG_SHA256,
        "Q50 authorization config SHA256 drift",
    )
    authorization = _yaml_mapping(authorization_bytes, "Q50 authorization config")
    _require(authorization.get("schema_version") == 1, "Q50 schema version drift")
    _require(
        authorization.get("phase") == "BRCA_Q50_SEQUENTIAL_ACQUISITION_GATE",
        "Q50 authorization phase drift",
    )
    _require(
        authorization.get("status")
        == "AUTHORIZED_Q50_ACQUISITION_AND_METADATA_ONLY",
        "Q50 authorization is not active",
    )
    _require(authorization.get("cohort") == "TCGA-BRCA", "Q50 cohort drift")
    _require(authorization.get("candidate") == "Q50", "Q50 candidate drift")
    approval = authorization.get("approval_record")
    _require(isinstance(approval, Mapping), "Q50 approval record is missing")
    approval_sha256 = _approval_identity(approval)
    _require(
        approval_sha256 == EXPECTED_APPROVAL_IDENTITY_SHA256,
        "Q50 approval identity SHA256 drift",
    )

    predecessor = authorization.get("predecessor")
    _require(isinstance(predecessor, Mapping), "Q50 predecessor binding is missing")
    _require(
        predecessor.get("q25_result_sha256") == EXPECTED_Q25_RESULT_SHA256,
        "Q25 result binding drift",
    )
    _require(
        predecessor.get("q25_feature_manifest_sha256")
        == EXPECTED_Q25_FEATURE_MANIFEST_SHA256,
        "Q25 feature manifest binding drift",
    )

    authority = authorization.get("authority")
    _require(isinstance(authority, Mapping), "Q50 authority is missing")
    download = authority.get("exact_wsi_download")
    _require(isinstance(download, Mapping), "Q50 download authority is missing")
    _require(download.get("status") == "APPROVED_Q50_ONLY", "Q50 download not approved")
    _require(download.get("source") == "GDC", "Q50 source must be GDC")
    _require(download.get("manifest_rows") == 1, "Q50 manifest must have one row")
    _require(download.get("concurrency") == 1, "Q50 patient concurrency must be one")
    _require(
        download.get("current_executable_scope") == "Q50_ONLY",
        "Q50 executable scope drift",
    )
    metadata = authority.get("wsi_open_or_metadata_read")
    _require(isinstance(metadata, Mapping), "Q50 metadata authority is missing")
    _require(
        metadata.get("status")
        == "APPROVED_Q50_HEADER_METADATA_ONLY_AFTER_EXACT_SIZE_AND_MD5",
        "Q50 header metadata scope drift",
    )
    _require(
        metadata.get("allowed_fields")
        == ["mpp_x", "mpp_y", "level_dimensions", "level_downsamples"],
        "Q50 metadata field allowlist drift",
    )
    _require(metadata.get("pixel_or_region_read") == "NOT_AUTHORIZED", "pixels unlocked")
    for field in (
        "q75_processing",
        "coordinate_generation",
        "feature_extraction",
        "healnet_real_input_execution",
        "training",
        "full_cohort_acquisition",
    ):
        item = authority.get(field)
        _require(
            isinstance(item, Mapping) and item.get("status") == "NOT_AUTHORIZED",
            f"{field} must remain not authorized",
        )

    selected = authorization.get("authorized_q50")
    _require(isinstance(selected, Mapping), "authorized Q50 identity is missing")
    expected_selected = {
        "patient_id": EXPECTED_Q50["patient_id"],
        "wsi_uuid": EXPECTED_Q50["id"],
        "filename": EXPECTED_Q50["filename"],
        "declared_bytes": int(EXPECTED_Q50["size"]),
        "md5": EXPECTED_Q50["md5"],
        "state": EXPECTED_Q50["state"],
        "omic_source_index": "370",
        "exact_case_and_full_slide_match": True,
        "omic_shapes": {
            "rna": [1, 1, 1558],
            "mutation": [1, 1, 21],
            "cnv": [1, 1, 1333],
        },
    }
    _require(dict(selected) == expected_selected, "authorized Q50 identity drift")

    storage = authorization.get("storage_contract")
    _require(isinstance(storage, Mapping), "Q50 storage contract is missing")
    _require(
        storage.get("download_destination")
        == "/teamspace/studios/this_studio/brca_pilot_data/Q50.incoming",
        "Q50 local download destination drift",
    )
    _require(storage.get("google_drive_required") is False, "Drive became required")
    _require(
        storage.get("google_drive_operations_authorized") is False,
        "Drive operations became authorized",
    )
    _require(
        storage.get("process_directly_from_rclone_mount") is False,
        "direct mount processing became allowed",
    )
    _require(
        storage.get("local_persistent_staging_required") is True,
        "local persistent staging is not required",
    )
    _require(
        storage.get("raw_wsi_deletion_authorized") is False,
        "raw WSI deletion became authorized",
    )

    execution = authorization.get("execution_contract")
    _require(isinstance(execution, Mapping), "Q50 execution contract is missing")
    _require(execution.get("one_row_manifest_only") is True, "bulk manifest allowed")
    _require(execution.get("one_patient_at_a_time") is True, "patient concurrency drift")
    _require(
        execution.get("gdc_internal_transfer_threads") == "client_default",
        "GDC client thread policy drift",
    )
    _require(
        execution.get("require_exact_size_and_md5_before_openslide") is True,
        "pre-OpenSlide identity gate is missing",
    )
    _require(execution.get("require_sha256_after_download") is True, "SHA256 gate missing")
    _require(
        execution.get("reuse_q25_level_indices_or_scale_mapping") is False,
        "Q25 slide-specific scale mapping was reused",
    )
    _require(
        execution.get("stop_after")
        == "Q50_EXACT_FILE_VERIFICATION_AND_HEADER_METADATA_REPORT",
        "Q50 stop boundary drift",
    )
    _require(
        authorization.get("prohibited_actions") == EXPECTED_PROHIBITED_ACTIONS,
        "Q50 prohibited-action boundary drift",
    )

    q25_result_bytes = _read_regular_bytes(
        q25_result_path, "Q25 result provenance", maximum=1024 * 1024
    )
    q25_result_sha256 = _sha256_bytes(q25_result_bytes)
    _require(q25_result_sha256 == EXPECTED_Q25_RESULT_SHA256, "Q25 result SHA256 drift")
    _validate_q25_predecessor(_yaml_mapping(q25_result_bytes, "Q25 result provenance"))

    feature_bytes = _read_regular_bytes(
        q25_feature_manifest_path, "Q25 feature manifest", maximum=1024 * 1024
    )
    feature_sha256 = _sha256_bytes(feature_bytes)
    _require(
        feature_sha256 == EXPECTED_Q25_FEATURE_MANIFEST_SHA256,
        "Q25 feature manifest SHA256 drift",
    )
    _validate_q25_feature_manifest(_json_mapping(feature_bytes, "Q25 feature manifest"))
    return Q50ApprovalBinding(
        authorization_config_sha256=authorization_sha256,
        approval_identity_sha256=approval_sha256,
        q25_result_sha256=q25_result_sha256,
        q25_feature_manifest_sha256=feature_sha256,
    )


def _q50_from_guarded_artifacts(
    artifacts: Sequence[OneRowManifestArtifact],
) -> OneRowManifestArtifact:
    matches = tuple(item for item in artifacts if item.selection.label == "Q50")
    _require(len(matches) == 1, "guarded source must contain exactly one Q50 row")
    item = matches[0]
    actual = {
        "label": item.selection.label,
        "patient_id": item.selection.patient_id,
        **item.selection.as_gdc_row(),
    }
    _require(actual == EXPECTED_Q50, "guarded Q50 identity drift")
    return item


def _manifest_bytes(selection: OneRowSelection) -> bytes:
    return (
        "id\tfilename\tmd5\tsize\tstate\n"
        + "\t".join(
            selection.as_gdc_row()[field]
            for field in ("id", "filename", "md5", "size", "state")
        )
        + "\n"
    ).encode("utf-8")


def _record_payload(
    selection: OneRowSelection,
    manifest_sha256: str,
    binding: Q50ApprovalBinding,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_label": POLICY_LABEL,
        "status": AUTHORIZATION_STATUS,
        "download_authorized": True,
        "authorized_label": "Q50",
        "current_executable_scope": "Q50_ONLY",
        "concurrency": 1,
        "q25_status": "SUCCESS_FROZEN_NO_RERUN",
        "q75_status": "NOT_AUTHORIZED",
        "metadata_opening": "Q50_HEADER_ONLY_AFTER_EXACT_SIZE_AND_MD5",
        "pixel_reads_authorized": False,
        "coordinate_generation_authorized": False,
        "feature_extraction_authorized": False,
        "healnet_execution_authorized": False,
        "training_authorized": False,
        "google_drive_required": False,
        "google_drive_operations_authorized": False,
        "stop_after": "Q50_EXACT_FILE_VERIFICATION_AND_HEADER_METADATA_REPORT",
        "approval_binding": {
            "authorization_config_sha256": binding.authorization_config_sha256,
            "approval_identity_sha256": binding.approval_identity_sha256,
            "q25_result_sha256": binding.q25_result_sha256,
            "q25_feature_manifest_sha256": binding.q25_feature_manifest_sha256,
        },
        "source_hashes": {
            "proposal_sha256": EXPECTED_PROPOSAL_SHA256,
            "alignment_sha256": EXPECTED_ALIGNMENT_SHA256,
            "filtered_gdc_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "guarded_q50_manifest_sha256": manifest_sha256,
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
    binding: Q50ApprovalBinding,
) -> bytes:
    return yaml.safe_dump(
        _record_payload(selection, manifest_sha256, binding), sort_keys=False
    ).encode("utf-8")


def _write_new_or_require_equal(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        _require(not path.is_symlink(), f"output must not be a symlink: {path}")
        mode = path.stat(follow_symlinks=False).st_mode
        _require(stat.S_ISREG(mode), f"output must be a regular file: {path}")
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
        try:
            os.link(temporary_name, path)
        except FileExistsError:
            _require(
                not path.is_symlink() and path.is_file() and path.read_bytes() == payload,
                f"concurrent output drift: {path}",
            )
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _validate_output_directory(output_directory: Path) -> tuple[Path, Path]:
    _require(
        not output_directory.is_symlink(),
        "authorized directory must not be a symlink",
    )
    _require(
        output_directory.is_dir(),
        f"authorized directory is missing: {output_directory}",
    )
    manifest_path = output_directory / AUTHORIZED_MANIFEST_BASENAME
    record_path = output_directory / AUTHORIZATION_RECORD_BASENAME
    expected = {manifest_path, record_path}
    actual = set(output_directory.iterdir())
    _require(
        actual == expected,
        "authorized directory must contain only the Q50 manifest and record",
    )
    for path in actual:
        _require(not path.is_symlink(), f"authorized output must not be a symlink: {path}")
        _require(
            stat.S_ISREG(path.stat(follow_symlinks=False).st_mode),
            f"authorized output must be a regular file: {path}",
        )
    return manifest_path, record_path


def validate_q50_authorized_manifest(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    guarded_directory: Path = DEFAULT_GUARDED_DIRECTORY,
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    q25_result_path: Path = DEFAULT_Q25_RESULT,
    q25_feature_manifest_path: Path = DEFAULT_Q25_FEATURE_MANIFEST,
) -> Q50AuthorizedManifestArtifact:
    binding = validate_q50_approval_binding(
        authorization_path=authorization_path,
        q25_result_path=q25_result_path,
        q25_feature_manifest_path=q25_feature_manifest_path,
    )
    guarded = validate_phase2_manifest_set(
        output_directory=guarded_directory, verify_source_hashes=True
    )
    q50 = _q50_from_guarded_artifacts(guarded)
    manifest_path, record_path = _validate_output_directory(output_directory)
    expected_manifest = _manifest_bytes(q50.selection)
    actual_manifest = manifest_path.read_bytes()
    _require(actual_manifest == expected_manifest, "authorized Q50 manifest content drift")
    manifest_sha256 = _sha256_bytes(actual_manifest)
    record = _yaml_mapping(
        _read_regular_bytes(record_path, "authorized Q50 record", maximum=1024 * 1024),
        "authorized Q50 record",
    )
    _require(
        record == _record_payload(q50.selection, manifest_sha256, binding),
        "authorized Q50 record content drift",
    )
    return Q50AuthorizedManifestArtifact(q50.selection, manifest_path, manifest_sha256)


def build_q50_authorized_manifest(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    guarded_directory: Path = DEFAULT_GUARDED_DIRECTORY,
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    q25_result_path: Path = DEFAULT_Q25_RESULT,
    q25_feature_manifest_path: Path = DEFAULT_Q25_FEATURE_MANIFEST,
) -> Q50AuthorizedManifestArtifact:
    """Create the Q50-only metadata artifacts without invoking GDC."""

    binding = validate_q50_approval_binding(
        authorization_path=authorization_path,
        q25_result_path=q25_result_path,
        q25_feature_manifest_path=q25_feature_manifest_path,
    )
    guarded = validate_phase2_manifest_set(
        output_directory=guarded_directory, verify_source_hashes=True
    )
    q50 = _q50_from_guarded_artifacts(guarded)
    _require(
        not output_directory.is_symlink(),
        "authorized directory must not be a symlink",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    allowed = {AUTHORIZED_MANIFEST_BASENAME, AUTHORIZATION_RECORD_BASENAME}
    entries = tuple(output_directory.iterdir())
    _require(
        {path.name for path in entries}.issubset(allowed),
        "unexpected Q50 authorized output entries",
    )
    for path in entries:
        _require(not path.is_symlink(), f"existing output must not be a symlink: {path}")
        _require(
            stat.S_ISREG(path.stat(follow_symlinks=False).st_mode),
            f"existing output must be a regular file: {path}",
        )
    manifest_payload = _manifest_bytes(q50.selection)
    manifest_sha256 = _sha256_bytes(manifest_payload)
    _write_new_or_require_equal(
        output_directory / AUTHORIZED_MANIFEST_BASENAME, manifest_payload
    )
    _write_new_or_require_equal(
        output_directory / AUTHORIZATION_RECORD_BASENAME,
        _record_bytes(q50.selection, manifest_sha256, binding),
    )
    return validate_q50_authorized_manifest(
        output_directory=output_directory,
        guarded_directory=guarded_directory,
        authorization_path=authorization_path,
        q25_result_path=q25_result_path,
        q25_feature_manifest_path=q25_feature_manifest_path,
    )
