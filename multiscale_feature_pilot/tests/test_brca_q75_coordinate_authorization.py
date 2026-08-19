from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import pytest
import yaml

from multiscale_feature_pilot.src import brca_q75_coordinate_authorization as auth


REPO_ROOT = Path(__file__).resolve().parents[2]
AUTH_PATH = REPO_ROOT / auth.AUTHORIZATION_RELATIVE_PATH
PROVENANCE_PATH = (
    REPO_ROOT
    / "multiscale_feature_pilot/provenance/"
    "brca_q75_coordinate_execution_approval.yaml"
)
REPORT_PATH = REPO_ROOT / "reports/brca_q75_coordinate_execution_approval.md"


def _document() -> dict[str, object]:
    value = yaml.safe_load(AUTH_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_exact_authorization_record_passes_pure_validator() -> None:
    auth.validate_q75_coordinate_execution_authorization(_document())


def test_exact_user_statement_and_hash_are_verbatim() -> None:
    approval = _document()["approval_evidence"]
    assert approval["exact_user_statement"] == auth.APPROVAL_STATEMENT
    assert approval["exact_user_statement_sha256"] == (
        auth.APPROVAL_STATEMENT_SHA256
    )
    assert hashlib.sha256(auth.APPROVAL_STATEMENT.encode("utf-8")).hexdigest() == (
        "4caef338b722aecad6810870221707f312986af0cd72c07c8091a732d3fea86c"
    )


def test_identity_and_stable_descriptor_prerequisites_are_exact() -> None:
    document = _document()
    identity = document["q75_identity"]
    assert identity["patient_id"] == auth.EXPECTED_PATIENT_ID
    assert identity["slide_id"] == auth.EXPECTED_SLIDE_ID
    assert identity["gdc_file_uuid"] == auth.EXPECTED_GDC_FILE_UUID
    assert identity["size_bytes"] == auth.EXPECTED_SIZE_BYTES
    assert identity["md5"] == auth.EXPECTED_MD5
    assert identity["sha256"] == auth.EXPECTED_SHA256
    assert identity["exact_path"] == auth.EXPECTED_WSI_PATH
    prerequisites = document["prerequisites"]
    assert prerequisites["secure_o_nofollow_held_descriptor"] == "required"
    assert prerequisites["openslide_access_through_held_proc_fd"] == "required"
    assert (
        prerequisites[
            "same_descriptor_identity_and_hash_recheck_before_publication"
        ]
        == "required"
    )


def test_only_one_exact_level_2_mask_read_is_authorized() -> None:
    operations = _document()["authorized_operations"]
    read = operations["mask_pixel_read"]
    assert read == {
        "api": "OpenSlide.read_region",
        "required_calls_for_success": 1,
        "maximum_calls": 1,
        "level": auth.MASK_LEVEL,
        "level_0_location": list(auth.MASK_LOCATION),
        "size_at_level": list(auth.MASK_SIZE),
    }
    assert operations["mask_processing"] == (
        "REVIEWED_BRCA_Q75_COORDINATE_POLICY_V1_EXECUTION_LOCKED"
    )
    assert operations["coordinate_generation"]["branches"] == [
        "scale_2x",
        "scale_4x",
    ]


def test_atomic_publication_and_deletion_boundary_are_not_contradictory() -> None:
    document = _document()
    publication = document["authorized_operations"]["artifact_publication"]
    assert publication["exact_output_directory"] == auth.EXPECTED_OUTPUT_PATH
    assert publication["atomic_directory_transaction"] == (
        "sibling_staging_then_linux_RENAME_NOREPLACE"
    )
    assert publication["no_overwrite"] is True
    assert publication["no_resume"] is True
    assert publication["ephemeral_transaction_cleanup"] == (
        "runner_created_q75_coordinate_lock_and_staging_paths_only"
    )
    assert publication["preexisting_or_final_artifact_deletion"] == "prohibited"
    assert "raw_wsi_deletion" in document["explicitly_prohibited"]
    assert (
        "preexisting_raw_user_project_or_final_artifact_deletion"
        in document["explicitly_prohibited"]
    )
    stop = document["required_stop"]
    assert (
        stop["preexisting_raw_user_project_or_final_data_deletion_authorized"]
        is False
    )
    assert stop["runner_owned_ephemeral_transaction_cleanup_authorized"] is True


@pytest.mark.parametrize(
    ("mutator", "match"),
    [
        (
            lambda doc: doc["approval_evidence"].__setitem__(
                "exact_user_statement", auth.APPROVAL_STATEMENT + "."
            ),
            "statement",
        ),
        (
            lambda doc: doc["q75_identity"].__setitem__("patient_id", "TCGA-X"),
            "identity",
        ),
        (
            lambda doc: doc["authorized_operations"]["mask_pixel_read"].__setitem__(
                "maximum_calls", 2
            ),
            "authorized operations",
        ),
        (
            lambda doc: doc["authorized_operations"]["mask_pixel_read"].__setitem__(
                "level", 1
            ),
            "authorized operations",
        ),
        (
            lambda doc: doc["authorized_operations"]["mask_pixel_read"].__setitem__(
                "size_at_level", [6782, 5654]
            ),
            "authorized operations",
        ),
        (
            lambda doc: doc["execution_policy"].__setitem__("device", "cuda"),
            "execution policy",
        ),
        (
            lambda doc: doc["explicitly_prohibited"].remove(
                "resnet50_inference_or_feature_extraction"
            ),
            "prohibition",
        ),
        (
            lambda doc: doc["required_stop"].__setitem__(
                "training_authorized", True
            ),
            "required stop",
        ),
        (
            lambda doc: doc["bound_policy_identity"][
                "coordinate_policy_core"
            ].__setitem__("sha256", "0" * 64),
            "bound policy identity",
        ),
    ],
)
def test_authorization_drift_fails_closed(mutator, match: str) -> None:
    document = deepcopy(_document())
    mutator(document)
    with pytest.raises(auth.Q75CoordinateAuthorizationError, match=match):
        auth.validate_q75_coordinate_execution_authorization(document)


def test_every_bound_policy_file_matches_its_recorded_sha256() -> None:
    bindings = _document()["bound_policy_identity"]
    assert bindings["policy_commit"] == auth.POLICY_COMMIT
    for label, binding in bindings.items():
        if label == "policy_commit":
            continue
        path = REPO_ROOT / binding["path"]
        assert path.is_file(), label
        assert _sha256(path) == binding["sha256"], label


def test_validator_exposes_no_execution_or_io_surface() -> None:
    source_path = Path(auth.__file__)
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported_roots = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported_roots.isdisjoint(
        {
            "cv2",
            "h5py",
            "numpy",
            "openslide",
            "os",
            "pathlib",
            "PIL",
            "shutil",
            "subprocess",
            "torch",
            "torchvision",
        }
    )
    forbidden_calls = {
        "open",
        "read_region",
        "unlink",
        "rmtree",
        "rename",
        "replace",
        "write_bytes",
        "write_text",
    }
    called_names = {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, (ast.Attribute, ast.Name))
    }
    assert called_names.isdisjoint(forbidden_calls)


def test_prior_execution_lock_is_superseded_only_by_exact_named_operations() -> None:
    policy = yaml.safe_load(
        (
            REPO_ROOT
            / "multiscale_feature_pilot/config/brca_q75_coordinate_policy.yaml"
        ).read_text(encoding="utf-8")
    )
    assert policy["execution_boundary"]["status"] == "EXECUTION_LOCKED"
    assert policy["execution_boundary"]["pixel_or_region_access_authorized"] is False
    supersession = _document()["supersession"]
    assert supersession["predecessor_records_remain_immutable"] is True
    assert supersession["execution_lock_superseded_only_for"] == [
        "exact_q75_wsi_secure_open_and_header_reverification",
        "exactly_one_q75_level_2_full_mask_read",
        "frozen_q75_tissue_segmentation",
        "frozen_q75_scale_2x_and_scale_4x_coordinate_generation",
        "q75_coordinate_artifact_atomic_no_overwrite_publication",
        "q75_coordinate_artifact_validation_and_reporting",
    ]


def test_approval_provenance_and_report_preserve_stop_boundary() -> None:
    provenance = yaml.safe_load(PROVENANCE_PATH.read_text(encoding="utf-8"))
    assert provenance["status"] == "Q75_COORDINATE_EXECUTION_AUTHORIZED_CPU_ONLY"
    assert provenance["approval"]["exact_user_statement"] == auth.APPROVAL_STATEMENT
    assert provenance["approval"]["exact_user_statement_sha256"] == (
        auth.APPROVAL_STATEMENT_SHA256
    )
    assert provenance["implementation"]["authorization_config"]["sha256"] == (
        _sha256(AUTH_PATH)
    )
    assert provenance["implementation"]["pure_authorization_validator"][
        "sha256"
    ] == _sha256(Path(auth.__file__))
    boundary = provenance["execution_boundary"]
    assert boundary["maximum_mask_reads"] == 1
    assert boundary["patch_reads"] == "prohibited"
    assert boundary["gpu"] == "prohibited"
    assert boundary["training"] == "prohibited"
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "exactly one" in report
    assert "runner-created ephemeral" in report
    assert "No Q75 pixel was read" in report
