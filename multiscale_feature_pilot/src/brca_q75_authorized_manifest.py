"""Build and validate the exact BRCA Q75-only GDC manifest.

This append-only transition binds the user's exact acquisition and header-only
authorization to the frozen Q25/Q50 review and successful pilot records.  It
creates metadata artifacts only: it performs no network request, does not open
a WSI, and cannot authorize pixels, coordinates, features, HEALNet, Drive,
deletion, cohort processing, or training.
"""

from __future__ import annotations

import hashlib
import errno
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from multiscale_feature_pilot.src.brca_one_row_manifest import (
    DEFAULT_OUTPUT_DIRECTORY as DEFAULT_GUARDED_DIRECTORY,
    EXPECTED_ALIGNMENT_SHA256,
    EXPECTED_PROPOSAL_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    OneRowManifestError,
    OneRowSelection,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUTHORIZATION_CONFIG = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/config/brca_q75_acquisition_authorization.yaml"
)
DEFAULT_REVIEW_PROVENANCE = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q25_q50_cpu_review.yaml"
)
DEFAULT_Q25_RESULT = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q25_gpu_pilot_result.yaml"
)
DEFAULT_Q50_RESULT = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q50_gpu_pilot_result.yaml"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_phase2_q75_authorized"
)

POLICY_LABEL = "BRCA_Q75_SEQUENTIAL_ACQUISITION_V1"
AUTHORIZATION_STATUS = "AUTHORIZED_Q75_ONLY"
AUTHORIZATION_RECORD_BASENAME = "MANIFEST_SET.AUTHORIZED_Q75.yaml"
AUTHORIZED_MANIFEST_BASENAME = (
    "Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770."
    "AUTHORIZED_Q75_ONLY.gdc.tsv"
)
GUARDED_MANIFEST_BASENAME = (
    "Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770."
    "NOT_AUTHORIZED.gdc.tsv"
)

EXPECTED_AUTHORIZATION_CONFIG_SHA256 = (
    "335e6d36aac1c21cc1cd52f8a14e5d2ecfde1f3a6f398d796bf842baaca35979"
)
EXPECTED_APPROVAL_STATEMENT_SHA256 = (
    "b43031fe0a9df8e5c44f35f9666567b202e4eb80ecb816386679a856a013bf01"
)
EXPECTED_REVIEW_COMMIT = "69d89a5df986b129fd0b9dc4dcc1e8534627093e"
EXPECTED_REQUEST_SHA256 = (
    "ee2c5147c0900c524daf8a7d8af468f5c71a536d1fc6d4f31e8e32ae6bb66c3c"
)
EXPECTED_REVIEW_REPORT_SHA256 = (
    "b9ad7c8cd271e769f1365a53c6b5e5ba3db951748e2e14b0d36f01e802a3a927"
)
EXPECTED_REVIEW_PROVENANCE_SHA256 = (
    "fc6784c891938c8213ec6e6468ca24d949fc0de4adcde62da6a5fbf48afe7a45"
)
EXPECTED_Q25_RESULT_SHA256 = (
    "76df862c438f4d94e34342b54c80c0866a440af0574be9003444075d7073c6e4"
)
EXPECTED_Q50_RESULT_SHA256 = (
    "904ad3c7bef859f89fef6151014f9eb4b37b5af5d6668972feb19b0e656d263b"
)
EXPECTED_GUARDED_Q75_SHA256 = (
    "8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a"
)
EXPECTED_CLEAN_OMIC_SHA256 = (
    "4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8"
)

EXPECTED_Q75 = {
    "label": "Q75",
    "patient_id": "TCGA-E2-A154",
    "id": "25aec062-60d1-446e-a1c6-0c79cc74a770",
    "filename": (
        "TCGA-E2-A154-01Z-00-DX1."
        "01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
    ),
    "md5": "a8c4b68fb6e0ab3e862efe3ed1fe10d7",
    "size": "1360743825",
    "state": "released",
}

EXPECTED_PROHIBITED_ACTIONS = [
    "read_region_or_any_pixel_region_access",
    "tissue_mask_generation",
    "coordinate_generation",
    "patch_extraction",
    "resnet50_inference",
    "healnet_execution",
    "q75_feature_generation",
    "q75_raw_file_deletion",
    "google_drive_operations",
    "full_cohort_processing",
    "q25_or_q50_modification_or_rerun",
    "blca_modification",
    "official_healnet_modification",
    "training_backward_or_optimizer_execution",
    "automatic_q75_scale_policy_inference_or_approval",
]


@dataclass(frozen=True)
class Q75ApprovalBinding:
    authorization_config_sha256: str
    approval_statement_sha256: str
    review_provenance_sha256: str
    q25_result_sha256: str
    q50_result_sha256: str


@dataclass(frozen=True)
class Q75AuthorizedManifestArtifact:
    selection: OneRowSelection
    path: Path
    sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise OneRowManifestError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_regular_bytes(path: Path, label: str, *, maximum: int) -> bytes:
    """Read a stable regular-file snapshot without following a final symlink.

    The opened descriptor, rather than a prior path stat, is authoritative.
    Initial/final descriptor tokens detect in-place changes; the final path
    token also detects rename/symlink swaps that occurred after ``os.open``.
    """

    _require(maximum >= 0, f"{label} size limit must be non-negative")
    _require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW")
    _require(hasattr(os, "O_CLOEXEC"), "platform lacks O_CLOEXEC")
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(os.fspath(path), flags)
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OneRowManifestError(f"{label} must not be a symlink") from exc
        if exc.errno == errno.ENOENT:
            raise OneRowManifestError(f"{label} is missing: {path}") from exc
        raise OneRowManifestError(f"cannot open {label}: {path}") from exc

    def token(details: os.stat_result) -> tuple[int, ...]:
        return (
            details.st_dev,
            details.st_ino,
            details.st_mode,
            details.st_nlink,
            details.st_uid,
            details.st_gid,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
        )

    try:
        initial = os.fstat(descriptor)
        _require(stat.S_ISREG(initial.st_mode), f"{label} must be a regular file")
        _require(initial.st_size <= maximum, f"{label} exceeds size limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            try:
                chunk = os.read(descriptor, 1024 * 1024)
            except OSError as exc:
                raise OneRowManifestError(f"cannot read {label}: {path}") from exc
            if not chunk:
                break
            total += len(chunk)
            _require(total <= maximum, f"{label} exceeds size limit")
            chunks.append(chunk)
        final = os.fstat(descriptor)
        _require(token(initial) == token(final), f"{label} changed during read")
        _require(total == initial.st_size, f"{label} size changed during read")
        try:
            current = os.stat(os.fspath(path), follow_symlinks=False)
        except OSError as exc:
            raise OneRowManifestError(f"{label} path changed during read") from exc
        _require(
            token(current) == token(final),
            f"{label} path identity changed during read",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _yaml_mapping(payload: bytes, label: str) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise OneRowManifestError(f"cannot parse {label}") from exc
    _require(isinstance(document, Mapping), f"{label} root must be a mapping")
    return document


def _validate_successful_pilot(
    result: Mapping[str, Any], *, candidate: str, status: str
) -> None:
    _require(result.get("candidate") == candidate, f"{candidate} candidate drift")
    _require(result.get("status") == status, f"{candidate} predecessor did not succeed")
    postconditions = result.get("postconditions")
    _require(isinstance(postconditions, Mapping), f"{candidate} postconditions missing")
    _require(
        postconditions.get("required_stop_reached") is True,
        f"{candidate} required stop was not reached",
    )
    for field in ("training_runs", "backward_passes", "optimizer_steps"):
        _require(postconditions.get(field) == 0, f"{candidate} {field} must be zero")
    _require(
        postconditions.get("google_drive_operations") == 0,
        f"{candidate} Drive operations must be zero",
    )
    _require(
        postconditions.get("raw_wsi_deletions") == 0,
        f"{candidate} raw WSI deletions must be zero",
    )


def _validate_review(review: Mapping[str, Any]) -> None:
    _require(
        review.get("status") == "BRCA_Q25_Q50_CPU_REVIEW_COMPLETE",
        "CPU review status drift",
    )
    _require(
        review.get("recommendation") == "RUN_Q75_PILOT_RECOMMENDED",
        "CPU review recommendation drift",
    )
    scope = review.get("scope")
    _require(isinstance(scope, Mapping), "CPU review scope missing")
    for field in (
        "q75_wsi_downloaded",
        "q75_wsi_opened",
        "q75_pixels_or_regions_read",
        "gpu_work",
        "full_cohort_processing",
        "google_drive_operations",
        "raw_wsi_deletion",
        "training",
    ):
        _require(scope.get(field) is False, f"CPU review {field} must be false")
    q75 = review.get("q75_verified_metadata")
    _require(isinstance(q75, Mapping), "reviewed Q75 metadata missing")
    expected = {
        "patient_id": EXPECTED_Q75["patient_id"],
        "slide_id": EXPECTED_Q75["filename"],
        "gdc_uuid": EXPECTED_Q75["id"],
        "declared_size_bytes": int(EXPECTED_Q75["size"]),
        "expected_md5_from_manifest": EXPECTED_Q75["md5"],
        "exact_case_and_full_slide_omic_match": True,
        "patient_wsi_count": 1,
        "patient_omic_count": 1,
        "omic_source_index": 771,
        "omic_physical_csv_line": 773,
    }
    for field, value in expected.items():
        _require(q75.get(field) == value, f"reviewed Q75 {field} drift")
    guarded = q75.get("guarded_manifest")
    _require(isinstance(guarded, Mapping), "reviewed guarded Q75 manifest missing")
    _require(
        guarded.get("sha256") == EXPECTED_GUARDED_Q75_SHA256,
        "reviewed guarded Q75 manifest identity drift",
    )


def _validate_authority(config: Mapping[str, Any]) -> None:
    authority = config.get("authority")
    _require(isinstance(authority, Mapping), "Q75 authority missing")
    download = authority.get("exact_wsi_download")
    _require(isinstance(download, Mapping), "Q75 download authority missing")
    expected_download = {
        "status": "APPROVED_Q75_ONLY",
        "source": "NCI_GDC",
        "manifest_rows": 1,
        "patient_concurrency": 1,
        "active_gdc_client_processes": 1,
        "gdc_internal_transfer_processes": "client_supported_default_minimum_3",
        "pass_n_processes_1": False,
        "current_executable_scope": "Q75_DOWNLOAD_AND_HEADER_ONLY",
    }
    _require(dict(download) == expected_download, "Q75 download authority drift")

    verification = authority.get("post_download_verification")
    _require(isinstance(verification, Mapping), "verification authority missing")
    _require(
        verification.get("status") == "REQUIRED_BEFORE_OPENSLIDE_CONSTRUCTION",
        "pre-OpenSlide verification gate drift",
    )
    _require(
        verification.get("required_checks")
        == [
            "exact_uuid",
            "exact_filename",
            "exact_byte_size",
            "expected_md5",
            "independent_sha256",
            "partial_or_incomplete_download_absent",
            "regular_file",
            "non_symlink",
            "svs_filename_suffix",
        ],
        "Q75 verification checklist drift",
    )
    header = authority.get("wsi_header_read")
    _require(isinstance(header, Mapping), "header authority missing")
    _require(
        header.get("status") == "APPROVED_Q75_HEADER_ONLY_AFTER_ALL_FILE_CHECKS",
        "Q75 header status drift",
    )
    _require(
        header.get("allowed_fields")
        == ["mpp_x", "mpp_y", "level_count", "level_dimensions", "level_downsamples"],
        "Q75 header allowlist drift",
    )
    _require(header.get("openslide_construction_allowed") is True, "OpenSlide locked")
    _require(
        header.get("read_region_or_pixel_access") == "NOT_AUTHORIZED",
        "pixel or region access unlocked",
    )
    _require(
        authority.get("omic_identity_reverification", {}).get("status")
        == "REQUIRED_BEFORE_METADATA_PUBLICATION",
        "Omic identity recheck not required",
    )
    for field in (
        "scale_policy_inference_or_approval",
        "tissue_mask_generation",
        "coordinate_generation",
        "patch_extraction",
        "resnet50_inference",
        "healnet_execution",
        "q75_feature_generation",
        "full_cohort_processing",
        "training",
    ):
        item = authority.get(field)
        _require(
            isinstance(item, Mapping) and item.get("status") == "NOT_AUTHORIZED",
            f"{field} must remain not authorized",
        )


def _validate_config(config: Mapping[str, Any]) -> str:
    _require(config.get("schema_version") == 1, "Q75 schema version drift")
    _require(
        config.get("phase") == "BRCA_Q75_ACQUISITION_AND_HEADER_METADATA_GATE",
        "Q75 authorization phase drift",
    )
    _require(
        config.get("status") == "AUTHORIZED_Q75_ACQUISITION_AND_HEADER_METADATA_ONLY",
        "Q75 authorization is not active",
    )
    _require(config.get("cohort") == "TCGA-BRCA", "Q75 cohort drift")
    _require(config.get("candidate") == "Q75", "Q75 candidate drift")
    _require(
        config.get("execution_mode") == "CPU_LOGIC_ONLY_CUDA_NOT_REQUIRED",
        "Q75 CPU-only execution directive drift",
    )
    approval = config.get("approval_record")
    _require(isinstance(approval, Mapping), "Q75 approval record missing")
    _require(approval.get("source") == "DIRECT_USER_AUTHORIZATION", "source drift")
    statement = approval.get("exact_user_statement")
    _require(isinstance(statement, str), "exact Q75 authorization text missing")
    statement_sha256 = _sha256_bytes(statement.encode("utf-8"))
    _require(
        statement_sha256 == EXPECTED_APPROVAL_STATEMENT_SHA256,
        "exact Q75 authorization statement drift",
    )
    _require(
        approval.get("exact_user_statement_sha256") == statement_sha256,
        "embedded Q75 authorization statement hash drift",
    )
    _require(
        approval.get("cpu_directive")
        == (
            "Stay on CPU logic even though the Studio may physically have a GPU "
            "attached; this Q75 stage does not need CUDA."
        ),
        "Q75 CPU directive drift",
    )

    predecessor = config.get("frozen_review_predecessor")
    _require(isinstance(predecessor, Mapping), "frozen review predecessor missing")
    expected_predecessor = {
        "commit": EXPECTED_REVIEW_COMMIT,
        "commit_subject": "docs: review BRCA pilots and prepare Q75 gate",
        "request_path": "multiscale_feature_pilot/config/brca_q75_acquisition_request.yaml",
        "request_sha256": EXPECTED_REQUEST_SHA256,
        "review_report_path": "reports/brca_q25_q50_cpu_review.md",
        "review_report_sha256": EXPECTED_REVIEW_REPORT_SHA256,
        "review_provenance_path": (
            "multiscale_feature_pilot/provenance/brca_q25_q50_cpu_review.yaml"
        ),
        "review_provenance_sha256": EXPECTED_REVIEW_PROVENANCE_SHA256,
        "recommendation": "RUN_Q75_PILOT_RECOMMENDED",
    }
    _require(dict(predecessor) == expected_predecessor, "review predecessor drift")

    successful = config.get("successful_predecessors")
    _require(isinstance(successful, Mapping), "successful predecessors missing")
    for label, expected_hash, expected_status in (
        ("q25", EXPECTED_Q25_RESULT_SHA256, "BRCA_Q25_GPU_FEATURE_PILOT_SUCCESS"),
        ("q50", EXPECTED_Q50_RESULT_SHA256, "BRCA_Q50_GPU_FEATURE_PILOT_SUCCESS"),
    ):
        item = successful.get(label)
        _require(isinstance(item, Mapping), f"{label} predecessor binding missing")
        _require(item.get("result_sha256") == expected_hash, f"{label} hash drift")
        _require(item.get("status") == expected_status, f"{label} status drift")
        _require(item.get("required_stop_reached") is True, f"{label} stop drift")
        _require(item.get("frozen_no_rerun") is True, f"{label} is not frozen")

    _validate_authority(config)
    selected = config.get("authorized_q75")
    _require(isinstance(selected, Mapping), "authorized Q75 identity missing")
    expected_selected = {
        "patient_id": EXPECTED_Q75["patient_id"],
        "wsi_uuid": EXPECTED_Q75["id"],
        "filename": EXPECTED_Q75["filename"],
        "declared_bytes": int(EXPECTED_Q75["size"]),
        "md5": EXPECTED_Q75["md5"],
        "state": EXPECTED_Q75["state"],
        "selection_quantile": 0.75,
        "singleton_rank": 671,
        "singleton_population": 894,
        "omic_source_index": 771,
        "omic_physical_csv_line": 773,
        "exact_case_and_full_slide_match": True,
        "patient_wsi_count": 1,
        "patient_omic_count": 1,
        "omic_shapes": {
            "rna": [1, 1, 1558],
            "mutation": [1, 1, 21],
            "cnv": [1, 1, 1333],
        },
        "omic_dtype": "float32",
        "omic_all_finite": True,
    }
    _require(dict(selected) == expected_selected, "authorized Q75 identity drift")

    sources = config.get("source_bindings")
    _require(isinstance(sources, Mapping), "Q75 source bindings missing")
    _require(
        sources.get("guarded_q75_manifest_sha256") == EXPECTED_GUARDED_Q75_SHA256,
        "guarded Q75 binding drift",
    )
    _require(
        sources.get("proposal_sha256") == EXPECTED_PROPOSAL_SHA256,
        "proposal binding drift",
    )
    _require(
        sources.get("alignment_sha256") == EXPECTED_ALIGNMENT_SHA256,
        "alignment binding drift",
    )
    _require(
        sources.get("filtered_gdc_manifest_sha256") == EXPECTED_SOURCE_MANIFEST_SHA256,
        "source manifest binding drift",
    )
    _require(
        sources.get("clean_omic_archive_sha256") == EXPECTED_CLEAN_OMIC_SHA256,
        "clean Omic binding drift",
    )

    storage = config.get("storage_contract")
    _require(isinstance(storage, Mapping), "Q75 storage contract missing")
    _require(
        storage.get("download_destination")
        == "/teamspace/studios/this_studio/brca_pilot_data/Q75.incoming",
        "Q75 staging path drift",
    )
    for field, value in (
        ("local_persistent_staging_required", True),
        ("process_directly_from_rclone_mount", False),
        ("google_drive_required", False),
        ("google_drive_operations_authorized", False),
        ("raw_wsi_deletion_authorized", False),
        ("retain_raw_wsi_after_metadata_gate", True),
    ):
        _require(storage.get(field) is value, f"Q75 storage {field} drift")

    execution = config.get("execution_contract")
    _require(isinstance(execution, Mapping), "Q75 execution contract missing")
    required_true = (
        "one_row_manifest_only",
        "one_patient_at_a_time",
        "one_active_gdc_client_process",
        "require_no_partial_or_incomplete_download_artifacts",
        "require_regular_non_symlink_svs",
        "require_exact_uuid_filename_size_and_md5_before_openslide",
        "require_independent_sha256_before_openslide",
        "prohibit_read_region_and_pixel_access",
        "require_exact_omic_rematch_before_recording",
        "allow_only_header_property_access",
    )
    for field in required_true:
        _require(execution.get(field) is True, f"Q75 execution {field} drift")
    _require(execution.get("pass_n_processes_1") is False, "invalid -n 1 enabled")
    _require(
        execution.get("gdc_internal_transfer_processes")
        == "client_supported_default_minimum_3",
        "GDC internal transfer policy drift",
    )
    _require(execution.get("infer_or_approve_scale_policy") is False, "scale unlocked")
    _require(execution.get("use_cuda") is False, "CUDA became authorized")
    _require(
        execution.get("stop_after")
        == "Q75_EXACT_FILE_VERIFICATION_AND_HEADER_METADATA_REPORT",
        "Q75 stop boundary drift",
    )
    _require(
        config.get("prohibited_actions") == EXPECTED_PROHIBITED_ACTIONS,
        "Q75 prohibited-action boundary drift",
    )
    required_stop = config.get("required_stop")
    _require(isinstance(required_stop, Mapping), "Q75 required stop missing")
    _require(
        dict(required_stop)
        == {
            "state": "WAIT_FOR_USER_REVIEW_AFTER_METADATA_REPORT",
            "later_work_authorized": False,
        },
        "Q75 required stop drift",
    )
    return statement_sha256


def validate_q75_approval_binding(
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    review_provenance_path: Path = DEFAULT_REVIEW_PROVENANCE,
    q25_result_path: Path = DEFAULT_Q25_RESULT,
    q50_result_path: Path = DEFAULT_Q50_RESULT,
) -> Q75ApprovalBinding:
    """Validate exact approval, frozen review, and both successful pilots."""

    authorization_bytes = _read_regular_bytes(
        authorization_path, "Q75 authorization config", maximum=2 * 1024 * 1024
    )
    authorization_sha256 = _sha256_bytes(authorization_bytes)
    _require(
        authorization_sha256 == EXPECTED_AUTHORIZATION_CONFIG_SHA256,
        "Q75 authorization config SHA256 drift",
    )
    approval_sha256 = _validate_config(
        _yaml_mapping(authorization_bytes, "Q75 authorization config")
    )

    review_bytes = _read_regular_bytes(
        review_provenance_path, "Q25/Q50 CPU review", maximum=2 * 1024 * 1024
    )
    review_sha256 = _sha256_bytes(review_bytes)
    _require(
        review_sha256 == EXPECTED_REVIEW_PROVENANCE_SHA256,
        "Q25/Q50 CPU review SHA256 drift",
    )
    _validate_review(_yaml_mapping(review_bytes, "Q25/Q50 CPU review"))

    q25_bytes = _read_regular_bytes(
        q25_result_path, "Q25 result provenance", maximum=2 * 1024 * 1024
    )
    q25_sha256 = _sha256_bytes(q25_bytes)
    _require(q25_sha256 == EXPECTED_Q25_RESULT_SHA256, "Q25 result SHA256 drift")
    _validate_successful_pilot(
        _yaml_mapping(q25_bytes, "Q25 result provenance"),
        candidate="Q25",
        status="BRCA_Q25_GPU_FEATURE_PILOT_SUCCESS",
    )

    q50_bytes = _read_regular_bytes(
        q50_result_path, "Q50 result provenance", maximum=2 * 1024 * 1024
    )
    q50_sha256 = _sha256_bytes(q50_bytes)
    _require(q50_sha256 == EXPECTED_Q50_RESULT_SHA256, "Q50 result SHA256 drift")
    _validate_successful_pilot(
        _yaml_mapping(q50_bytes, "Q50 result provenance"),
        candidate="Q50",
        status="BRCA_Q50_GPU_FEATURE_PILOT_SUCCESS",
    )
    return Q75ApprovalBinding(
        authorization_config_sha256=authorization_sha256,
        approval_statement_sha256=approval_sha256,
        review_provenance_sha256=review_sha256,
        q25_result_sha256=q25_sha256,
        q50_result_sha256=q50_sha256,
    )


def _q75_from_guarded_directory(guarded_directory: Path) -> OneRowSelection:
    """Securely bind the exact reviewed Q75 row without a path-based reread."""

    selection = OneRowSelection(
        label=EXPECTED_Q75["label"],
        patient_id=EXPECTED_Q75["patient_id"],
        gdc_file_uuid=EXPECTED_Q75["id"],
        filename=EXPECTED_Q75["filename"],
        md5=EXPECTED_Q75["md5"],
        size_bytes=int(EXPECTED_Q75["size"]),
        state=EXPECTED_Q75["state"],
    )
    path = guarded_directory / GUARDED_MANIFEST_BASENAME
    payload = _read_regular_bytes(
        path, "guarded Q75 manifest", maximum=1024 * 1024
    )
    _require(payload == _manifest_bytes(selection), "guarded Q75 manifest content drift")
    _require(
        _sha256_bytes(payload) == EXPECTED_GUARDED_Q75_SHA256,
        "guarded Q75 SHA drift",
    )
    return selection


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
    binding: Q75ApprovalBinding,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_label": POLICY_LABEL,
        "status": AUTHORIZATION_STATUS,
        "download_authorized": True,
        "authorized_label": "Q75",
        "current_executable_scope": "Q75_DOWNLOAD_AND_HEADER_ONLY",
        "patient_concurrency": 1,
        "active_gdc_client_processes": 1,
        "gdc_internal_transfer_processes": "CLIENT_SUPPORTED_DEFAULT_MINIMUM_3",
        "pass_n_processes_1": False,
        "q25_status": "SUCCESS_FROZEN_NO_RERUN",
        "q50_status": "SUCCESS_FROZEN_NO_RERUN",
        "metadata_opening": "Q75_HEADER_ONLY_AFTER_ALL_EXACT_FILE_CHECKS",
        "openslide_construction_authorized": True,
        "pixel_or_region_reads_authorized": False,
        "tissue_mask_generation_authorized": False,
        "coordinate_generation_authorized": False,
        "patch_extraction_authorized": False,
        "resnet50_inference_authorized": False,
        "healnet_execution_authorized": False,
        "feature_generation_authorized": False,
        "training_authorized": False,
        "google_drive_required": False,
        "google_drive_operations_authorized": False,
        "raw_wsi_deletion_authorized": False,
        "full_cohort_processing_authorized": False,
        "automatic_scale_policy_authorized": False,
        "use_cuda": False,
        "stop_after": "Q75_EXACT_FILE_VERIFICATION_AND_HEADER_METADATA_REPORT",
        "approval_binding": {
            "authorization_config_sha256": binding.authorization_config_sha256,
            "approval_statement_sha256": binding.approval_statement_sha256,
            "review_commit": EXPECTED_REVIEW_COMMIT,
            "review_provenance_sha256": binding.review_provenance_sha256,
            "q25_result_sha256": binding.q25_result_sha256,
            "q50_result_sha256": binding.q50_result_sha256,
        },
        "source_hashes": {
            "proposal_sha256": EXPECTED_PROPOSAL_SHA256,
            "alignment_sha256": EXPECTED_ALIGNMENT_SHA256,
            "filtered_gdc_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
            "guarded_q75_manifest_sha256": manifest_sha256,
            "clean_omic_archive_sha256": EXPECTED_CLEAN_OMIC_SHA256,
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
    binding: Q75ApprovalBinding,
) -> bytes:
    return yaml.safe_dump(
        _record_payload(selection, manifest_sha256, binding), sort_keys=False
    ).encode("utf-8")


def _write_new_or_require_equal(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        existing = _read_regular_bytes(
            path, "existing output", maximum=len(payload)
        )
        _require(existing == payload, f"existing output drift: {path}")
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
            existing = _read_regular_bytes(
                path, "concurrent output", maximum=len(payload)
            )
            _require(
                existing == payload,
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
        "authorized directory must contain only the Q75 manifest and record",
    )
    for path in actual:
        _require(not path.is_symlink(), f"authorized output must not be a symlink: {path}")
        _require(
            stat.S_ISREG(path.stat(follow_symlinks=False).st_mode),
            f"authorized output must be a regular file: {path}",
        )
    return manifest_path, record_path


def validate_q75_authorized_manifest(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    guarded_directory: Path = DEFAULT_GUARDED_DIRECTORY,
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    review_provenance_path: Path = DEFAULT_REVIEW_PROVENANCE,
    q25_result_path: Path = DEFAULT_Q25_RESULT,
    q50_result_path: Path = DEFAULT_Q50_RESULT,
) -> Q75AuthorizedManifestArtifact:
    binding = validate_q75_approval_binding(
        authorization_path=authorization_path,
        review_provenance_path=review_provenance_path,
        q25_result_path=q25_result_path,
        q50_result_path=q50_result_path,
    )
    q75 = _q75_from_guarded_directory(guarded_directory)
    manifest_path, record_path = _validate_output_directory(output_directory)
    expected_manifest = _manifest_bytes(q75)
    actual_manifest = _read_regular_bytes(
        manifest_path, "authorized Q75 manifest", maximum=1024 * 1024
    )
    _require(actual_manifest == expected_manifest, "authorized Q75 manifest content drift")
    manifest_sha256 = _sha256_bytes(actual_manifest)
    _require(
        manifest_sha256 == EXPECTED_GUARDED_Q75_SHA256,
        "authorized Q75 manifest SHA256 drift",
    )
    record = _yaml_mapping(
        _read_regular_bytes(record_path, "authorized Q75 record", maximum=1024 * 1024),
        "authorized Q75 record",
    )
    _require(
        record == _record_payload(q75, manifest_sha256, binding),
        "authorized Q75 record content drift",
    )
    return Q75AuthorizedManifestArtifact(q75, manifest_path, manifest_sha256)


def build_q75_authorized_manifest(
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY,
    guarded_directory: Path = DEFAULT_GUARDED_DIRECTORY,
    authorization_path: Path = DEFAULT_AUTHORIZATION_CONFIG,
    review_provenance_path: Path = DEFAULT_REVIEW_PROVENANCE,
    q25_result_path: Path = DEFAULT_Q25_RESULT,
    q50_result_path: Path = DEFAULT_Q50_RESULT,
) -> Q75AuthorizedManifestArtifact:
    """Create the Q75-only authorization metadata without invoking GDC."""

    binding = validate_q75_approval_binding(
        authorization_path=authorization_path,
        review_provenance_path=review_provenance_path,
        q25_result_path=q25_result_path,
        q50_result_path=q50_result_path,
    )
    q75 = _q75_from_guarded_directory(guarded_directory)
    _require(
        not output_directory.is_symlink(),
        "authorized directory must not be a symlink",
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    allowed = {AUTHORIZED_MANIFEST_BASENAME, AUTHORIZATION_RECORD_BASENAME}
    entries = tuple(output_directory.iterdir())
    _require(
        {path.name for path in entries}.issubset(allowed),
        "unexpected Q75 authorized output entries",
    )
    for path in entries:
        _require(not path.is_symlink(), f"existing output must not be a symlink: {path}")
        _require(
            stat.S_ISREG(path.stat(follow_symlinks=False).st_mode),
            f"existing output must be a regular file: {path}",
        )
    manifest_payload = _manifest_bytes(q75)
    manifest_sha256 = _sha256_bytes(manifest_payload)
    _require(
        manifest_sha256 == EXPECTED_GUARDED_Q75_SHA256,
        "Q75 manifest bytes do not match guarded source identity",
    )
    _write_new_or_require_equal(
        output_directory / AUTHORIZED_MANIFEST_BASENAME, manifest_payload
    )
    _write_new_or_require_equal(
        output_directory / AUTHORIZATION_RECORD_BASENAME,
        _record_bytes(q75, manifest_sha256, binding),
    )
    return validate_q75_authorized_manifest(
        output_directory=output_directory,
        guarded_directory=guarded_directory,
        authorization_path=authorization_path,
        review_provenance_path=review_provenance_path,
        q25_result_path=q25_result_path,
        q50_result_path=q50_result_path,
    )
