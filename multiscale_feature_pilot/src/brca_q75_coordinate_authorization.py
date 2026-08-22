"""Pure validation for the bounded BRCA Q75 coordinate authorization.

This module contains only immutable scalar/string policy data.  It has no
filesystem, OpenSlide, image, array, artifact-publication, CUDA, or network
API.  The separately implemented runner must SHA-pin the authorization file
and this validator before it performs the authorized operation.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
from typing import Any, Final


AUTHORIZATION_STATUS: Final = "AUTHORIZED_Q75_MASK_READ_AND_COORDINATES_ONLY"
APPROVAL_STATEMENT: Final = (
    "I authorize one Q75-only CPU coordinate execution: exactly one OpenSlide "
    "level-2 mask read at (0,0), size (6783,5654), using the frozen Q75 "
    "coordinate policy, followed by tissue segmentation and atomic publication "
    "of the two coordinate bags. No patch reads, ResNet50, HEALNet, GPU, "
    "Q25/Q50/BLCA changes, Drive operations, deletion, full-cohort work, or "
    "training"
)
APPROVAL_STATEMENT_SHA256: Final = (
    "4caef338b722aecad6810870221707f312986af0cd72c07c8091a732d3fea86c"
)
POLICY_COMMIT: Final = "1582d5b1d5eb5fac7d44e5d0e5d4fb2feebf87f9"
AUTHORIZATION_RELATIVE_PATH: Final = (
    "multiscale_feature_pilot/config/"
    "brca_q75_coordinate_execution_authorization.yaml"
)

EXPECTED_PATIENT_ID: Final = "TCGA-E2-A154"
EXPECTED_SLIDE_ID: Final = (
    "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
)
EXPECTED_GDC_FILE_UUID: Final = "25aec062-60d1-446e-a1c6-0c79cc74a770"
EXPECTED_SIZE_BYTES: Final = 1_360_743_825
EXPECTED_MD5: Final = "a8c4b68fb6e0ab3e862efe3ed1fe10d7"
EXPECTED_SHA256: Final = (
    "844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37"
)
EXPECTED_WSI_PATH: Final = (
    "/teamspace/studios/this_studio/brca_pilot_data/Q75.incoming/"
    "25aec062-60d1-446e-a1c6-0c79cc74a770/"
    "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
)
EXPECTED_OUTPUT_PATH: Final = (
    "/teamspace/studios/this_studio/brca_pilot_data/Q75.coordinates"
)
MASK_LEVEL: Final = 2
MASK_LOCATION: Final = (0, 0)
MASK_SIZE: Final = (6783, 5654)


class Q75CoordinateAuthorizationError(ValueError):
    """Raised when the authorization record expands or drifts."""


_EXPECTED_TOP_LEVEL_KEYS: Final = {
    "schema_version",
    "cohort",
    "candidate_label",
    "phase",
    "status",
    "approval_evidence",
    "supersession",
    "q75_identity",
    "bound_policy_identity",
    "prerequisites",
    "authorized_operations",
    "execution_policy",
    "explicitly_prohibited",
    "required_stop",
}

_EXPECTED_SUPERSESSION: Final = {
    "predecessor_policy_status": (
        "REVIEWED_BRCA_Q75_COORDINATE_POLICY_V1_EXECUTION_LOCKED"
    ),
    "predecessor_records_remain_immutable": True,
    "execution_lock_superseded_only_for": [
        "exact_q75_wsi_secure_open_and_header_reverification",
        "exactly_one_q75_level_2_full_mask_read",
        "frozen_q75_tissue_segmentation",
        "frozen_q75_scale_2x_and_scale_4x_coordinate_generation",
        "q75_coordinate_artifact_atomic_no_overwrite_publication",
        "q75_coordinate_artifact_validation_and_reporting",
    ],
}

_EXPECTED_IDENTITY: Final = {
    "patient_id": EXPECTED_PATIENT_ID,
    "slide_id": EXPECTED_SLIDE_ID,
    "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
    "filename": EXPECTED_SLIDE_ID,
    "size_bytes": EXPECTED_SIZE_BYTES,
    "md5": EXPECTED_MD5,
    "sha256": EXPECTED_SHA256,
    "exact_path": EXPECTED_WSI_PATH,
    "exact_omic_source_row_index": "771",
    "exact_case_and_full_slide_omic_match": True,
    "omic_shapes": {
        "rna": [1, 1, 1558],
        "mutation": [1, 1, 21],
        "cnv": [1, 1, 1333],
    },
}

_EXPECTED_POLICY_BINDINGS: Final = {
    "policy_commit": POLICY_COMMIT,
    "header_result": {
        "path": (
            "multiscale_feature_pilot/provenance/"
            "brca_q75_header_metadata_result/result.yaml"
        ),
        "sha256": (
            "08a7ed3e67ddf17513ee2dbda2adfd2398333787aaa75fe9eacf911f3c1a3898"
        ),
        "result_commit": "c7e98f4ce2663556be9b487441fe36494364ff18",
    },
    "header_report": {
        "path": (
            "multiscale_feature_pilot/provenance/"
            "brca_q75_header_metadata_result/report.md"
        ),
        "sha256": (
            "9ed50ecc8464109e0a8ca121082462f8765b1c70663499180cc644cfe604d985"
        ),
    },
    "scale_policy_config": {
        "path": "multiscale_feature_pilot/config/brca_q75_scale_policy.yaml",
        "sha256": (
            "d29be0892e0b0324ae9b4390a1db9a9ae4b5a60b4541ddb7a36c81b8d2bca6b5"
        ),
    },
    "scale_policy_core": {
        "path": "multiscale_feature_pilot/src/brca_q75_scale_policy.py",
        "sha256": (
            "3aecb1f3818f9ae98708cdf61f6ccf4b938ffe5fe78bbbaff6e11896e5eb4482"
        ),
    },
    "scale_policy_provenance": {
        "path": (
            "multiscale_feature_pilot/provenance/brca_q75_scale_approval.yaml"
        ),
        "sha256": (
            "aae6547c7c23cfdad51f62e3587b33c7abe3c5fb7d6fbdcd15ffeed3737fdd8e"
        ),
    },
    "coordinate_policy_config": {
        "path": (
            "multiscale_feature_pilot/config/brca_q75_coordinate_policy.yaml"
        ),
        "sha256": (
            "58f15a9e39fcd3469ec656ef98c72ad6e42b8a3eab16fcbc24c4345cc4337d88"
        ),
    },
    "coordinate_policy_core": {
        "path": "multiscale_feature_pilot/src/brca_q75_coordinate_policy.py",
        "sha256": (
            "6f58c3b5f23f91d16c60e041933418ddf176bad70defdc7428176ae7505c35d0"
        ),
    },
    "coordinate_policy_review": {
        "path": (
            "multiscale_feature_pilot/provenance/"
            "brca_q75_coordinate_policy_review.yaml"
        ),
        "sha256": (
            "3edd1a2000eb808c140972af4b4b9bf3b6e25d3602b4a2076b99c5fcc0046197"
        ),
    },
    "coordinate_policy_report": {
        "path": "reports/brca_q75_coordinate_policy_review.md",
        "sha256": (
            "e4df78300e6e6607a9301f50858ab27152390e05485707f66867f2d95838dbeb"
        ),
    },
    "reviewed_shared_coordinate_algorithm": {
        "path": "multiscale_feature_pilot/src/brca_q25_coordinates.py",
        "sha256": (
            "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82"
        ),
    },
    "coordinate_artifact_publisher": {
        "path": "multiscale_feature_pilot/src/brca_coordinate_artifacts.py",
        "sha256": (
            "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf"
        ),
    },
    "known_issues": {
        "path": "shared/provenance/known_issues.md",
        "sha256": (
            "8dff689f8181f7e08215595252042185542d9970c5885693b9afdaa7aa32c3c4"
        ),
    },
}

_EXPECTED_PREREQUISITES: Final = {
    "cpu_only": "required",
    "exact_regular_non_symlink_svs": "required",
    "secure_o_nofollow_held_descriptor": "required",
    "openslide_access_through_held_proc_fd": "required",
    "same_descriptor_identity_and_hash_recheck_before_publication": "required",
    "no_partial_or_incomplete_download_siblings": "required",
    "exact_uuid_filename_size_md5_sha256": "required",
    "exact_header_and_pyramid_metadata": "required",
    "exact_patient_slide_and_omic_row_match": "required",
    "critical_sources_byte_equal_to_execution_commit": "required",
    "official_healnet_clean_and_pinned": "required",
    "frozen_blca_tag_unchanged": "required",
    "storage_recheck": "required",
    "final_output_directory_absent": "required",
    "sibling_staging_and_lock_absent": "required",
    "output_outside_git": "required",
}

_EXPECTED_AUTHORIZED_OPERATIONS: Final = {
    "openslide_open_exact_q75": True,
    "header_metadata_reverification": True,
    "mask_pixel_read": {
        "api": "OpenSlide.read_region",
        "required_calls_for_success": 1,
        "maximum_calls": 1,
        "level": MASK_LEVEL,
        "level_0_location": list(MASK_LOCATION),
        "size_at_level": list(MASK_SIZE),
    },
    "mask_processing": "REVIEWED_BRCA_Q75_COORDINATE_POLICY_V1_EXECUTION_LOCKED",
    "coordinate_generation": {
        "branches": ["scale_2x", "scale_4x"],
        "coordinate_space": "level_0_x_y",
    },
    "artifact_publication": {
        "schema": "BRCA_COORDINATE_ARTIFACT_SET_V1",
        "exact_output_directory": EXPECTED_OUTPUT_PATH,
        "atomic_directory_transaction": (
            "sibling_staging_then_linux_RENAME_NOREPLACE"
        ),
        "no_overwrite": True,
        "no_resume": True,
        "ephemeral_transaction_cleanup": (
            "runner_created_q75_coordinate_lock_and_staging_paths_only"
        ),
        "preexisting_or_final_artifact_deletion": "prohibited",
    },
}

_EXPECTED_EXECUTION_POLICY: Final = {
    "device": "cpu",
    "cuda_calls": "prohibited",
    "tissue_mask_level": MASK_LEVEL,
    "tissue_mask_dimensions": list(MASK_SIZE),
    "tissue_mask_is_shared_by_branches": True,
    "coordinate_geometry_scale_xy": [16.0, 16.001238061549344],
    "scale_2x": {
        "level_0_footprint": [512, 512],
        "level_0_step": [512, 512],
        "later_output_patch": [256, 256],
        "later_interpolation": "PIL.Image.Resampling.LANCZOS",
        "later_effective_mpp": [0.4936, 0.4936],
    },
    "scale_4x": {
        "level_0_declared_footprint": [1024, 1024],
        "level_0_step": [1024, 1024],
        "later_read_level": 1,
        "later_read_size": [256, 256],
        "later_effective_mpp": [0.9872163682185965, 0.9872163682185965],
    },
    "grid_anchor": [0, 0],
    "reject_incomplete_footprints": True,
    "coordinate_order": "row_major_y_then_x",
    "require_nonempty": True,
    "require_unique": True,
}

EXPECTED_PROHIBITIONS: Final = (
    "patch_or_tile_pixel_reads",
    "level_0_or_level_1_pixel_reads",
    "associated_image_or_thumbnail_reads",
    "patch_resampling_execution",
    "resnet50_inference_or_feature_extraction",
    "q75_feature_or_pt_generation",
    "healnet_execution",
    "gpu_or_cuda_operations",
    "q25_or_q50_modification_or_rerun",
    "blca_modification",
    "google_drive_operations",
    "raw_wsi_deletion",
    "preexisting_raw_user_project_or_final_artifact_deletion",
    "full_cohort_processing",
    "official_healnet_modification",
    "model_training_or_optimizer_operations",
    "git_tracking_of_wsi_hdf5_or_feature_artifacts",
)

_EXPECTED_REQUIRED_STOP: Final = {
    "after": "Q75_COORDINATE_ARTIFACT_VALIDATION_AND_REPORT",
    "patch_extraction_authorized": False,
    "resnet50_authorized": False,
    "healnet_authorized": False,
    "gpu_authorized": False,
    "q75_feature_generation_authorized": False,
    "q25_q50_blca_changes_authorized": False,
    "google_drive_authorized": False,
    "preexisting_raw_user_project_or_final_data_deletion_authorized": False,
    "runner_owned_ephemeral_transaction_cleanup_authorized": True,
    "full_cohort_authorized": False,
    "training_authorized": False,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q75CoordinateAuthorizationError(message)


def _exact_mapping(actual: object, expected: Mapping[str, Any], label: str) -> None:
    _require(isinstance(actual, Mapping), f"{label} must be a mapping")
    _require(dict(actual) == dict(expected), f"{label} drifted from authorization")


def validate_q75_coordinate_execution_authorization(
    document: Mapping[str, Any],
) -> None:
    """Fail closed unless *document* is the exact narrow Q75 authorization."""

    _require(isinstance(document, Mapping), "authorization must be a mapping")
    _require(
        set(document) == _EXPECTED_TOP_LEVEL_KEYS,
        "authorization top-level schema drifted",
    )
    _require(document.get("schema_version") == 1, "schema version drifted")
    _require(document.get("cohort") == "TCGA-BRCA", "cohort drifted")
    _require(document.get("candidate_label") == "Q75", "candidate drifted")
    _require(document.get("phase") == "BRCA_Q75_COORDINATE_GATE", "phase drifted")
    _require(document.get("status") == AUTHORIZATION_STATUS, "status drifted")

    approval = document.get("approval_evidence")
    _require(isinstance(approval, Mapping), "approval evidence must be a mapping")
    _require(
        set(approval)
        == {
            "source",
            "exact_user_statement",
            "exact_user_statement_sha256",
            "bounded_interpretation",
            "authorized_scope",
        },
        "approval evidence schema drifted",
    )
    _require(approval.get("source") == "DIRECT_USER_INSTRUCTION", "source drifted")
    statement = approval.get("exact_user_statement")
    _require(statement == APPROVAL_STATEMENT, "exact user statement drifted")
    _require(isinstance(statement, str), "exact user statement must be text")
    digest = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    _require(digest == APPROVAL_STATEMENT_SHA256, "computed statement SHA256 drifted")
    _require(
        approval.get("exact_user_statement_sha256") == APPROVAL_STATEMENT_SHA256,
        "recorded statement SHA256 drifted",
    )
    _require(
        approval.get("authorized_scope")
        == "Q75_SINGLE_MASK_READ_COORDINATE_GENERATION_AND_PUBLICATION_ONLY",
        "authorized scope drifted",
    )
    _require(
        approval.get("bounded_interpretation")
        == (
            "Supersede the frozen Q75 coordinate policy's execution lock only "
            "for one CPU execution against the exact Q75 WSI: securely "
            "reverify identity and frozen dependencies, perform exactly one "
            "OpenSlide read_region call for the complete native level-2 mask "
            "at level-0 location (0,0), segment tissue, generate the scale_2x "
            "and scale_4x coordinate bags, atomically publish the validated "
            "artifact set outside Git without overwrite or resume, and stop."
        ),
        "bounded interpretation drifted",
    )

    _exact_mapping(document.get("supersession"), _EXPECTED_SUPERSESSION, "supersession")
    _exact_mapping(document.get("q75_identity"), _EXPECTED_IDENTITY, "Q75 identity")
    _exact_mapping(
        document.get("bound_policy_identity"),
        _EXPECTED_POLICY_BINDINGS,
        "bound policy identity",
    )
    _exact_mapping(
        document.get("prerequisites"), _EXPECTED_PREREQUISITES, "prerequisites"
    )
    _exact_mapping(
        document.get("authorized_operations"),
        _EXPECTED_AUTHORIZED_OPERATIONS,
        "authorized operations",
    )
    _exact_mapping(
        document.get("execution_policy"),
        _EXPECTED_EXECUTION_POLICY,
        "execution policy",
    )

    prohibitions = document.get("explicitly_prohibited")
    _require(isinstance(prohibitions, list), "prohibitions must be a list")
    _require(
        prohibitions == list(EXPECTED_PROHIBITIONS),
        "explicit prohibition set or order drifted",
    )
    _require(len(prohibitions) == len(set(prohibitions)), "duplicate prohibition")
    _exact_mapping(
        document.get("required_stop"), _EXPECTED_REQUIRED_STOP, "required stop"
    )


__all__ = [
    "APPROVAL_STATEMENT",
    "APPROVAL_STATEMENT_SHA256",
    "AUTHORIZATION_RELATIVE_PATH",
    "AUTHORIZATION_STATUS",
    "EXPECTED_GDC_FILE_UUID",
    "EXPECTED_MD5",
    "EXPECTED_OUTPUT_PATH",
    "EXPECTED_PATIENT_ID",
    "EXPECTED_PROHIBITIONS",
    "EXPECTED_SHA256",
    "EXPECTED_SIZE_BYTES",
    "EXPECTED_SLIDE_ID",
    "EXPECTED_WSI_PATH",
    "MASK_LEVEL",
    "MASK_LOCATION",
    "MASK_SIZE",
    "POLICY_COMMIT",
    "Q75CoordinateAuthorizationError",
    "validate_q75_coordinate_execution_authorization",
]
