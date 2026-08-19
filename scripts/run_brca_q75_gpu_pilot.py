#!/usr/bin/env python3
"""Execution-locked runner for the proposed BRCA Q75 GPU feature pilot.

This is deliberately not a general feature-extraction command.  Every input,
output, execution setting, and stop boundary is fixed to the already selected
TCGA-BRCA Q75 patient.  The command fails before constructing a pixel-reading
dataset unless the authorization, source tree, immutable inputs, coordinates,
Omic row, checkpoint, WSI header, and Tesla T4 runtime all pass their gates.

The future successful boundary is Q75 feature extraction plus a random-weight
HEALNet interface/numerical smoke.  Until a separate exact authorization is
recorded, this module fails before WSI hashing, OpenSlide, CUDA, model creation,
or construction of a pixel-capable dataset.  It never trains, downloads,
deletes, processes another candidate, processes a cohort, or uses Google Drive.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Any, Callable, Final, Mapping
import zipfile

import h5py
import numpy as np
import openslide
from PIL import __version__ as pillow_version
import torch
import torchvision
import yaml


REPO_ROOT: Final = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from multiscale_feature_pilot.src.brca_omic import (  # noqa: E402
    BRCA_EXPECTED_DIMS,
    BRCA_RELEASE_ARCHIVE_SHA256,
    load_official_brca_patient_omics,
)
from multiscale_feature_pilot.src.brca_coordinate_artifacts import (  # noqa: E402
    CoordinateArtifactSetRecord,
    validate_brca_coordinate_artifacts,
)
from multiscale_feature_pilot.src.brca_q75_feature_artifacts import (  # noqa: E402
    CoordinateFeatureBinding,
    EXPECTED_ENCODER_IDENTITY,
    EXPECTED_OMIC_CONTENT_SHA256,
    FeatureArtifactMetadata,
    publish_brca_q75_feature_artifacts,
)
from multiscale_feature_pilot.src.brca_q75_feature_extraction import (  # noqa: E402
    Q75_GDC_FILE_UUID as EXPECTED_GDC_FILE_UUID,
    Q75_PATIENT_ID as EXPECTED_PATIENT_ID,
    Q75_SLIDE_ID as EXPECTED_SLIDE_ID,
    Q75_WSI_FILENAME as EXPECTED_FILENAME,
    Q75_WSI_MD5 as EXPECTED_MD5,
    Q75_WSI_SHA256 as EXPECTED_SHA256,
    Q75_WSI_SIZE_BYTES as EXPECTED_SIZE_BYTES,
    StreamingQ75OpenSlideDataset,
    capture_q75_wsi_file_identity,
    load_q75_branch_read_spec,
)
from multiscale_feature_pilot.src.brca_q75_scale_policy import (  # noqa: E402
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_LEVEL_DOWNSAMPLES as EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES,
    EXPECTED_MPP_X,
    EXPECTED_MPP_Y,
)
from multiscale_feature_pilot.src.feature_extraction import (  # noqa: E402
    build_resnet50_imagenet1k_v2,
    extract_feature_matrix,
)
from multiscale_feature_pilot.src.provenance import (  # noqa: E402
    BranchProvenanceSpec,
    PatchProvenance,
    build_two_scale_provenance,
    validate_provenance_alignment,
)
from multiscale_feature_pilot.src.supervisor_healnet_smoke import (  # noqa: E402
    run_one_patient_supervisor_healnet_smoke,
)


SCHEMA: Final = "BRCA_Q75_GPU_PILOT_RESULT_V1"
SUCCESS_STATUS: Final = "BRCA_Q75_GPU_FEATURE_PILOT_SUCCESS"
BLOCKED_STATUS: Final = "BRCA_Q75_GPU_FEATURE_PILOT_BLOCKED"
OFFICIAL_REPO: Final = Path("/teamspace/studios/this_studio/healnet")
OFFICIAL_HEAD: Final = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
FROZEN_TAG: Final = "blca-one-patient-pilot-v1"
FROZEN_COMMIT: Final = "df7cf2bda783ab6cc09e95d6a1fa0914da05a433"

WSI_PATH: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q75.incoming/"
    "25aec062-60d1-446e-a1c6-0c79cc74a770/"
    "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
)
COORDINATE_DIR: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q75.coordinates"
)
OMIC_PATH: Final = Path(
    "/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/"
    "data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
CHECKPOINT_PATH: Final = Path(
    "/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth"
)
OUTPUT_DIR: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q75.features"
)
AUTH_RELATIVE_PATH: Final = Path(
    "multiscale_feature_pilot/config/brca_q75_gpu_execution_authorization.yaml"
)

OMIC_SIZE_BYTES: Final = 4_081_277
OMIC_MEMBER_FILENAME: Final = "./tcga_brca_all_clean.csv.zip"
OMIC_MEMBER_SIZE_BYTES: Final = 15_021_018
OMIC_MEMBER_SHA256: Final = (
    "052637f2a69c515812796d9638566cb75299b6a3571dbdc5363496f12665027d"
)
CHECKPOINT_SIZE_BYTES: Final = 102_540_417
CHECKPOINT_SHA256: Final = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
CHECKPOINT_IDENTITY: Final = EXPECTED_ENCODER_IDENTITY
COORDINATE_MANIFEST_SHA256: Final = (
    "438165ce6b3be9d26d66c65cd70793e29cc92208cfb6a78bf68043bc4b4a4e90"
)
COORDINATE_RESULT_SHA256: Final = (
    "1b583b9366a5308c26ee74de3686f080992f9381dc93b8b2d8f6729854e4be17"
)
EXPECTED_COORDINATE_BRANCHES: Final = {
    "scale_2x": {
        "count": 13_487,
        "filename": "scale_2x_coordinates.h5",
        "size_bytes": 225_464,
        "sha256": "d88c201d379a5954cdfa4d785760f6c8f9d4b8bec498f7f31d040b1fdf7440ec",
        "coordinates_sha256": "88e1ac8e00d4f05da7f83e542bfe7c933e9849a29ae60deac68adabe79e748b0",
        "source_level": 0,
        "source_patch_size": (512, 512),
        "output_patch_size": (256, 256),
        "level_0_footprint": (512, 512),
        "effective_mpp": (0.4936, 0.4936),
        "interpolation": "PIL.Image.Resampling.LANCZOS",
        "resampling": "explicit_2x_spatial_downsample",
    },
    "scale_4x": {
        "count": 3_458,
        "filename": "scale_4x_coordinates.h5",
        "size_bytes": 65_000,
        "sha256": "0b0cfdaa26493dd24c3bbcba9f57c6b10d6060ba0f5f8b0a59bc0938ff655d94",
        "coordinates_sha256": "63f58c687943509a55314ced55c7afe1610a26873a751d3e1d0f2d06cff3fb5d",
        "source_level": 1,
        "source_patch_size": (256, 256),
        "output_patch_size": (256, 256),
        "level_0_footprint": (1024, 1024),
        "effective_mpp": (0.9872163682185965, 0.9872163682185965),
        "interpolation": "none",
        "resampling": "none",
    },
}
EXPECTED_TOTAL_PATCHES: Final = 16_945
FEATURE_DIM: Final = 2_048
BATCH_SIZE: Final = 32
NUM_WORKERS: Final = 2
SEED: Final = 0
DEVICE: Final = "cuda:0"
EXPECTED_GPU_NAME_TOKEN: Final = "Tesla T4"
CUBLAS_WORKSPACE_CONFIG: Final = ":4096:8"

# CPU preparation is not authority to touch Q75 pixels or CUDA.  A later exact
# user statement must be recorded, hashed, and committed before maintainers may
# deliberately flip this gate and finalize AUTH_SHA256.
EXECUTION_AUTHORIZED: Final = False
AUTH_SHA256: Final = "PENDING_SEPARATE_EXACT_Q75_GPU_AUTHORIZATION"
BOUND_FILES: Final = (
    (
        Path("multiscale_feature_pilot/__init__.py"),
        "f05f42721ce2380799c1e655e85d267b73bdf54eec02bcd6a7afaaf42faf93d3",
    ),
    (
        Path("multiscale_feature_pilot/src/__init__.py"),
        "1f56e6eecfd95b2d6fdc69114a616edf4cf87d6174958bfd2cc58cb79b6605dd",
    ),
    (
        Path("multiscale_feature_pilot/src/omic.py"),
        "cd9c80bd9ab3a049beca682131f2553f526a38a0dee09608eb016ae1f79607ed",
    ),
    (
        Path("multiscale_feature_pilot/src/padding.py"),
        "27498695f5aa717411adb44583c4044cbae96e47e0662586c3f4d4eb8cd64174",
    ),
    (
        Path(
            "multiscale_feature_pilot/provenance/"
            "brca_q75_coordinate_execution_result.yaml"
        ),
        COORDINATE_RESULT_SHA256,
    ),
    (
        Path("multiscale_feature_pilot/config/brca_q75_scale_policy.yaml"),
        "d29be0892e0b0324ae9b4390a1db9a9ae4b5a60b4541ddb7a36c81b8d2bca6b5",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q75_scale_policy.py"),
        "3aecb1f3818f9ae98708cdf61f6ccf4b938ffe5fe78bbbaff6e11896e5eb4482",
    ),
    (
        Path("multiscale_feature_pilot/config/brca_q75_coordinate_policy.yaml"),
        "58f15a9e39fcd3469ec656ef98c72ad6e42b8a3eab16fcbc24c4345cc4337d88",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q75_coordinate_policy.py"),
        "6f58c3b5f23f91d16c60e041933418ddf176bad70defdc7428176ae7505c35d0",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
        "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q75_feature_extraction.py"),
        "5b2c3c426eab17b9312cdd692e81c35c4dac552d47e0f9cddd07a05df1e2abd1",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q75_feature_artifacts.py"),
        "df927ba75f8cfa1865f69b31073b1fab36f7480ad652c1ecc7c2e81488e8a280",
    ),
    (
        Path("multiscale_feature_pilot/src/feature_extraction.py"),
        "9731f9773238800268673ec299acbcdf8caaa4ac676c96f4559cb10604c707e7",
    ),
    (
        Path("multiscale_feature_pilot/src/scale_2x_policy.py"),
        "a3672f730f8be026edfab0ea25c4460836aad227099152e5cc7d290c03d1f7aa",
    ),
    (
        Path("multiscale_feature_pilot/src/multiscale_bag.py"),
        "26a85db8738b80c7f7f2f75d1379a54203ee870519ee1ee63f78f809ed17d914",
    ),
    (
        Path("multiscale_feature_pilot/src/provenance.py"),
        "76f8c9eac1ba0c32679a1d7f8d34c07c79fc27cdc28c6762dd13a169e5db5917",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_omic.py"),
        "5061039913cb0dd4e8e30958c8b4b76f088396f7841255dd94234c3b462d5a5d",
    ),
    (
        Path("multiscale_feature_pilot/src/supervisor_healnet_smoke.py"),
        "df2c0010347ddc6cfc49200bf35802337752c01a78b96727a5ad538fb847eec4",
    ),
    (
        Path("multiscale_feature_pilot/src/healnet_smoke.py"),
        "34a7ad1204dfc84ef8e63a2ac4cd8b932dbccf39e9f424b424dc6eb37a4d6009",
    ),
)
CRITICAL_TRACKED_PATHS: Final = (
    Path("scripts/run_brca_q75_gpu_pilot.py"),
    AUTH_RELATIVE_PATH,
    *(path for path, _digest in BOUND_FILES),
)

_HASH_CHUNK_SIZE: Final = 8 * 1024 * 1024
_ALLOWED_OPERATIONS: Final = {
    "q75_exact_input_identity_hash_header_omic_checkpoint_preflights",
    "q75_exact_13487_level_0_scale_2x_patch_reads",
    "q75_exact_3458_level_1_scale_4x_patch_reads",
    "q75_scale_2x_lanczos_resampling",
    "q75_resnet50_imagenet1k_v2_float32_feature_extraction",
    "q75_external_atomic_feature_artifact_publication",
    "q75_full_size_synthetic_natural_layout_healnet_contract_smoke",
    "q75_real_feature_natural_layout_four_modality_healnet_smoke",
}
_PROHIBITED_OPERATIONS: Final = {
    "model_training",
    "optimizer_or_backward_execution",
    "q75_operations_outside_bound_patient",
    "q75_tissue_mask_or_coordinate_regeneration",
    "q75_thumbnail_or_associated_image_reads",
    "q75_patch_image_persistence",
    "q75_feature_output_overwrite_or_resume",
    "q25_q50_rerun_or_modification",
    "blca_modification",
    "full_cohort_download_or_processing",
    "any_network_or_wsi_download_operation",
    "google_drive_mount_upload_or_delete",
    "raw_wsi_or_existing_artifact_deletion",
    "official_healnet_modification",
    "git_tracking_of_wsi_hdf5_pt_pth_or_checkpoint_artifacts",
    "cpu_feature_extraction_fallback",
    "automatic_mixed_precision",
    "tf32",
    "feature_pooling",
    "cross_patient_concatenation",
    "wsi_feature_transpose",
}

PROPOSED_EXACT_USER_AUTHORIZATION: Final = (
    "I authorize the exact Q75-only GPU pilot for patient TCGA-E2-A154 using "
    "the frozen verified Q75 coordinates: read-only input/hash/header/Omic/"
    "checkpoint preflights; 13,487 scale-2x and 3,458 scale-4x patch reads; "
    "float32 ResNet50 ImageNet1K V2 feature extraction; "
    "torch.cat([features_2x, features_4x], dim=0) with no transpose or "
    "pooling; atomic publication to Q75.features; and synthetic plus "
    "real-feature four-modality HEALNet numerical smoke tests. No training, "
    "backward pass, optimizer step, AMP, TF32, CPU fallback, coordinate "
    "regeneration, Q25/Q50/BLCA changes, full-cohort processing, network or "
    "Google Drive operations, raw-file deletion, or official HEALNet "
    "modification. Stop after validated Q75 feature artifacts and smoke-test "
    "reporting."
)


class Q75GpuPilotError(RuntimeError):
    """Raised whenever the exact authorized GPU contract is not satisfied."""


@dataclass(frozen=True)
class PilotPaths:
    """The only paths accepted by the production Q75 entry point."""

    repo_root: Path = REPO_ROOT
    official_repo: Path = OFFICIAL_REPO
    wsi: Path = WSI_PATH
    coordinates: Path = COORDINATE_DIR
    omic: Path = OMIC_PATH
    checkpoint: Path = CHECKPOINT_PATH
    output: Path = OUTPUT_DIR
    auth: Path = REPO_ROOT / AUTH_RELATIVE_PATH


@dataclass(frozen=True)
class PilotDependencies:
    """Injectable seams used only by focused tests; production uses defaults."""

    coordinate_validator: Callable[..., object]
    branch_spec_loader: Callable[[object], object]
    wsi_identity_capture: Callable[[Path], object]
    dataset_factory: Callable[[Path, object, object], object]
    model_builder: Callable[[Path], object]
    feature_extractor: Callable[..., object]
    omic_loader: Callable[..., object]
    smoke_runner: Callable[..., object]
    slide_factory: Callable[[str], object]
    artifact_publisher: Callable[..., object]


def _default_dependencies() -> PilotDependencies:
    """Return the fixed BRCA-only implementation stack."""

    return PilotDependencies(
        coordinate_validator=validate_brca_coordinate_artifacts,
        branch_spec_loader=load_q75_branch_read_spec,
        wsi_identity_capture=capture_q75_wsi_file_identity,
        dataset_factory=lambda path, spec, identity: StreamingQ75OpenSlideDataset(
            path,
            spec,
            expected_file_identity=identity,
        ),
        model_builder=build_resnet50_imagenet1k_v2,
        feature_extractor=extract_feature_matrix,
        omic_loader=load_official_brca_patient_omics,
        smoke_runner=run_one_patient_supervisor_healnet_smoke,
        slide_factory=openslide.OpenSlide,
        artifact_publisher=publish_brca_q75_feature_artifacts,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q75GpuPilotError(message)


def _require_execution_authorized() -> None:
    """Keep the prepared runner inert until a later exact user authorization."""

    _require(
        EXECUTION_AUTHORIZED is True,
        "Q75 GPU execution is locked pending separate exact user authorization",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _wsi_hashes(path: Path) -> tuple[str, str]:
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    """Hash contiguous CPU float32 values in the frozen artifact encoding."""

    array = tensor.detach().cpu().numpy().astype(np.dtype("<f4"), copy=False)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _regular_non_symlink(path: Path, label: str) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Q75GpuPilotError(f"missing {label}: {path}") from exc
    _require(
        stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
        f"{label} must be a regular non-symlink file: {path}",
    )
    return metadata


def _stat_identity(metadata: os.stat_result) -> tuple[int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
    )


def _adapter_identity_tuple(identity: object) -> tuple[int, int, int, int]:
    values = tuple(
        getattr(identity, name, None)
        for name in ("st_dev", "st_ino", "st_size", "st_mtime_ns")
    )
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in values),
        "Q75 adapter WSI identity token is invalid",
    )
    return values  # type: ignore[return-value]


def _require_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    _require(path.is_absolute(), f"path must be absolute: {path}")
    parts = path.parts
    limit = len(parts) if include_leaf else len(parts) - 1
    current = Path(parts[0])
    for component in parts[1:limit]:
        current /= component
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise Q75GpuPilotError(f"missing path component: {current}") from exc
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"symlink component prohibited: {current}",
        )


def _git(repo: Path, *arguments: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=not binary,
    )
    if result.returncode != 0:
        stderr = result.stderr if isinstance(result.stderr, str) else b""
        stdout = result.stdout if isinstance(result.stdout, str) else b""
        detail = (stderr or stdout).strip()
        raise Q75GpuPilotError(
            f"git {' '.join(arguments)} failed for {repo}: {detail}"
        )
    return result.stdout if binary else result.stdout.strip()


def _repository_snapshot(paths: PilotPaths) -> dict[str, object]:
    official_head = _git(paths.official_repo, "rev-parse", "HEAD")
    official_status = _git(paths.official_repo, "status", "--porcelain=v1", "-z")
    _require(official_head == OFFICIAL_HEAD, "official HEALNet HEAD drift")
    _require(official_status == "", "official HEALNet worktree is modified")
    frozen_commit = _git(paths.repo_root, "rev-parse", f"{FROZEN_TAG}^{{commit}}")
    _require(frozen_commit == FROZEN_COMMIT, "frozen BLCA tag drift")

    critical_hashes: dict[str, str] = {}
    for relative_path in CRITICAL_TRACKED_PATHS:
        path = paths.repo_root / relative_path
        _regular_non_symlink(path, f"critical source {relative_path}")
        try:
            committed = _git(
                paths.repo_root,
                "show",
                f"HEAD:{relative_path.as_posix()}",
                binary=True,
            )
        except Q75GpuPilotError as exc:
            raise Q75GpuPilotError(
                f"critical source is not tracked at HEAD: {relative_path}"
            ) from exc
        _require(
            path.read_bytes() == committed,
            f"critical source differs from HEAD: {relative_path}",
        )
        critical_hashes[relative_path.as_posix()] = _sha256_file(path)

    return {
        "official_head": official_head,
        "official_status_porcelain_v1_z": official_status,
        "frozen_tag": FROZEN_TAG,
        "frozen_commit": frozen_commit,
        "source_head": _git(paths.repo_root, "rev-parse", "HEAD"),
        "source_branch": _git(paths.repo_root, "branch", "--show-current"),
        "source_status_porcelain_v1_z": _git(
            paths.repo_root, "status", "--porcelain=v1", "-z"
        ),
        "critical_execution_source_sha256": critical_hashes,
    }


def _expected_bound_inputs(paths: PilotPaths) -> dict[str, object]:
    return {
        "wsi": {
            "path": str(paths.wsi),
            "patient_id": EXPECTED_PATIENT_ID,
            "slide_id": EXPECTED_SLIDE_ID,
            "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
            "filename": EXPECTED_FILENAME,
            "size_bytes": EXPECTED_SIZE_BYTES,
            "md5": EXPECTED_MD5,
            "sha256": EXPECTED_SHA256,
        },
        "coordinates": {
            "directory": str(paths.coordinates),
            "coordinate_execution_result_sha256": COORDINATE_RESULT_SHA256,
            "manifest_sha256": COORDINATE_MANIFEST_SHA256,
            "branches": {
                branch: {
                    "filename": values["filename"],
                    "size_bytes": values["size_bytes"],
                    "sha256": values["sha256"],
                    "coordinates_sha256": values["coordinates_sha256"],
                    "coordinate_count": values["count"],
                }
                for branch, values in EXPECTED_COORDINATE_BRANCHES.items()
            },
        },
        "omic": {
            "path": str(paths.omic),
            "size_bytes": OMIC_SIZE_BYTES,
            "sha256": BRCA_RELEASE_ARCHIVE_SHA256,
            "member_filename": OMIC_MEMBER_FILENAME,
            "member_size_bytes": OMIC_MEMBER_SIZE_BYTES,
            "member_sha256": OMIC_MEMBER_SHA256,
            "exact_case_id": EXPECTED_PATIENT_ID,
            "exact_slide_id": EXPECTED_FILENAME,
            "exact_source_row_index": "771",
            "rna_shape": [1, 1, BRCA_EXPECTED_DIMS["rna"]],
            "mutation_shape": [1, 1, BRCA_EXPECTED_DIMS["mutation"]],
            "cnv_shape": [1, 1, BRCA_EXPECTED_DIMS["cnv"]],
            "modality_content_sha256": dict(EXPECTED_OMIC_CONTENT_SHA256),
        },
        "checkpoint": {
            "path": str(paths.checkpoint),
            "size_bytes": CHECKPOINT_SIZE_BYTES,
            "sha256": CHECKPOINT_SHA256,
            "identity": CHECKPOINT_IDENTITY,
        },
        "output": {
            "directory": str(paths.output),
            "must_be_outside_git": True,
            "atomic_directory_publication": True,
            "overwrite_or_resume": False,
        },
    }


def _validate_authorization_semantics(document: object, paths: PilotPaths) -> None:
    _require(isinstance(document, dict), "authorization YAML root must be a mapping")
    _require(document.get("schema_version") == 1, "authorization schema drift")
    _require(
        document.get("phase") == "BRCA_Q75_GPU_FEATURE_AND_INTERFACE_PILOT",
        "authorization phase drift",
    )
    _require(
        document.get("status")
        == "AUTHORIZED_Q75_FEATURE_EXTRACTION_AND_HEALNET_SMOKE_ONLY",
        "authorization status drift",
    )
    _require(document.get("cohort") == "TCGA-BRCA", "authorization cohort drift")
    _require(document.get("candidate") == "Q75", "authorization candidate drift")
    _require(
        document.get("execution_authorized") is True,
        "authorization execution flag is not true",
    )
    _require(
        document.get("approval_evidence")
        == {
            "user_authorization": PROPOSED_EXACT_USER_AUTHORIZATION,
            "supervisor_cohort": "Go with brca, blca",
        },
        "authorization approval evidence drift",
    )

    approval_scope = document.get("approval_scope")
    _require(isinstance(approval_scope, dict), "authorization approval scope missing")
    _require(
        set(approval_scope.get("allowed", ())) == _ALLOWED_OPERATIONS,
        "authorized operation set drift",
    )
    _require(
        set(approval_scope.get("prohibited", ())) == _PROHIBITED_OPERATIONS,
        "prohibited operation set drift",
    )

    _require(
        document.get("execution_contract")
        == {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "seed": SEED,
            "device": DEVICE,
            "dtype": "float32",
            "automatic_mixed_precision": False,
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "torch_deterministic_algorithms": True,
            "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
            "pre_extraction_synthetic_interface_smoke": True,
            "branch_order": ["scale_2x", "scale_4x"],
            "combined_operation": "torch.cat",
            "combined_dim": 0,
            "natural_model_shape": [1, EXPECTED_TOTAL_PATCHES, FEATURE_DIM],
            "no_transpose": True,
            "no_pooling": True,
            "sequential_branch_extraction": True,
            "cpu_concatenation": True,
        },
        "authorization execution contract drift",
    )
    _require(
        document.get("bound_inputs") == _expected_bound_inputs(paths),
        "authorization input binding drift",
    )
    bound_sources = document.get("bound_sources")
    _require(isinstance(bound_sources, dict), "authorization source bindings missing")
    observed_bound = {
        (Path(value["path"]), value["sha256"])
        for value in bound_sources.values()
        if isinstance(value, dict) and set(value) == {"path", "sha256"}
    }
    _require(
        observed_bound == set(BOUND_FILES),
        "authorization source binding chain drift",
    )
    _require(
        document.get("required_stop")
        == {
            "after": "VALIDATED_Q75_FEATURE_ARTIFACTS_AND_HEALNET_SMOKE",
            "training_authorized": False,
        },
        "authorization stop boundary drift",
    )


def _validate_bound_sources(paths: PilotPaths) -> dict[str, str]:
    _require_execution_authorized()
    _require(
        len(AUTH_SHA256) == 64
        and all(character in "0123456789abcdef" for character in AUTH_SHA256),
        "execution authorization SHA-256 has not been finalized",
    )
    _require(
        all(
            len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            for _path, digest in BOUND_FILES
        ),
        "bound source SHA-256 values have not been finalized",
    )
    _require(
        paths.auth == paths.repo_root / AUTH_RELATIVE_PATH,
        "authorization path drift",
    )
    expected = ((AUTH_RELATIVE_PATH, AUTH_SHA256), *BOUND_FILES)
    observed: dict[str, str] = {}
    for relative_path, expected_sha256 in expected:
        path = paths.repo_root / relative_path
        _regular_non_symlink(path, f"bound source {relative_path}")
        actual_sha256 = _sha256_file(path)
        _require(
            actual_sha256 == expected_sha256,
            f"bound source SHA-256 drift: {relative_path}",
        )
        observed[relative_path.as_posix()] = actual_sha256
    try:
        authorization = yaml.safe_load(paths.auth.read_text(encoding="utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise Q75GpuPilotError("execution authorization YAML is unreadable") from exc
    _validate_authorization_semantics(authorization, paths)
    return observed


def _validate_paths(paths: PilotPaths) -> None:
    _require(paths.repo_root == REPO_ROOT, "pilot repository path drift")
    _require(paths.official_repo == OFFICIAL_REPO, "official repository path drift")
    _require(paths.wsi == WSI_PATH, "Q75 WSI path drift")
    _require(paths.coordinates == COORDINATE_DIR, "Q75 coordinate path drift")
    _require(paths.omic == OMIC_PATH, "BRCA Omic path drift")
    _require(paths.checkpoint == CHECKPOINT_PATH, "checkpoint path drift")
    _require(paths.output == OUTPUT_DIR, "Q75 feature output path drift")
    _require(paths.auth == REPO_ROOT / AUTH_RELATIVE_PATH, "authorization path drift")

    for path in (paths.wsi, paths.omic, paths.checkpoint, paths.auth):
        _require_no_symlink_components(path, include_leaf=True)
    _require_no_symlink_components(paths.coordinates, include_leaf=True)
    _require_no_symlink_components(paths.output, include_leaf=False)
    coordinate_metadata = paths.coordinates.lstat()
    _require(
        stat.S_ISDIR(coordinate_metadata.st_mode) and not paths.coordinates.is_symlink(),
        "coordinate input must be a non-symlink directory",
    )

    resolved_output = paths.output.resolve()
    for label, repository in (
        ("pilot", paths.repo_root.resolve()),
        ("official", paths.official_repo.resolve()),
    ):
        _require(
            resolved_output != repository
            and not resolved_output.is_relative_to(repository),
            f"feature output must be outside the {label} Git worktree",
        )
    _require(paths.output.parent.is_dir(), "feature output parent must already exist")
    _require(not os.path.lexists(paths.output), "feature output already exists")
    lock = paths.output.parent / f".{paths.output.name}.lock"
    _require(not os.path.lexists(lock), "feature output lock already exists")
    staging_prefix = f".{paths.output.name}.staging."
    _require(
        not any(item.name.startswith(staging_prefix) for item in paths.output.parent.iterdir()),
        "feature output has a stale or active staging sibling",
    )


def _verify_wsi(path: Path) -> tuple[dict[str, object], tuple[int, int, int, int]]:
    metadata = _regular_non_symlink(path, "Q75 WSI")
    _require(path.name == EXPECTED_FILENAME, "Q75 WSI filename drift")
    _require(metadata.st_size == EXPECTED_SIZE_BYTES, "Q75 WSI size drift")
    partials = tuple(
        entry.name
        for entry in path.parent.iterdir()
        if entry.name.endswith((".partial", ".part"))
    )
    _require(not partials, "partial WSI transfer sibling exists")
    md5, sha256 = _wsi_hashes(path)
    _require(md5 == EXPECTED_MD5, "Q75 WSI MD5 drift")
    _require(sha256 == EXPECTED_SHA256, "Q75 WSI SHA-256 drift")
    token = _stat_identity(metadata)
    return {
        "path": str(path),
        "size_bytes": metadata.st_size,
        "md5": md5,
        "sha256": sha256,
        "regular_non_symlink": True,
        "partial_siblings": [],
    }, token


def _verify_regular_sha256(
    path: Path,
    *,
    label: str,
    size_bytes: int,
    sha256: str,
) -> tuple[dict[str, object], tuple[int, int, int, int]]:
    metadata = _regular_non_symlink(path, label)
    identity_token = _stat_identity(metadata)
    _require(metadata.st_size == size_bytes, f"{label} size drift")
    actual_sha256 = _sha256_file(path)
    _require(actual_sha256 == sha256, f"{label} SHA-256 drift")
    _require(
        _stat_identity(_regular_non_symlink(path, label)) == identity_token,
        f"{label} changed while it was hashed",
    )
    return (
        {
            "path": str(path),
            "size_bytes": metadata.st_size,
            "sha256": actual_sha256,
            "regular_non_symlink": True,
        },
        identity_token,
    )


def _validate_coordinate_record(record: object) -> dict[str, object]:
    _require(
        isinstance(record, CoordinateArtifactSetRecord),
        "coordinate validator returned an unexpected record type",
    )
    _require(
        record.manifest_sha256 == COORDINATE_MANIFEST_SHA256,
        "coordinate manifest hash drift",
    )
    result: dict[str, object] = {
        "directory": str(record.directory),
        "manifest_sha256": record.manifest_sha256,
        "branches": {},
    }
    branch_results = result["branches"]
    assert isinstance(branch_results, dict)
    for branch in ("scale_2x", "scale_4x"):
        expected = EXPECTED_COORDINATE_BRANCHES[branch]
        item = record.branch_for(branch)
        metadata = item.metadata
        _require(item.path.name == expected["filename"], f"{branch} filename drift")
        _require(item.size_bytes == expected["size_bytes"], f"{branch} size drift")
        _require(item.sha256 == expected["sha256"], f"{branch} HDF5 hash drift")
        _require(
            item.coordinates_sha256 == expected["coordinates_sha256"],
            f"{branch} coordinate-content hash drift",
        )
        _require(item.coordinate_count == expected["count"], f"{branch} count drift")
        _require(metadata.patient_id == EXPECTED_PATIENT_ID, f"{branch} patient drift")
        _require(metadata.slide_id == EXPECTED_SLIDE_ID, f"{branch} slide drift")
        _require(
            metadata.gdc_file_uuid == EXPECTED_GDC_FILE_UUID,
            f"{branch} GDC UUID drift",
        )
        _require(metadata.wsi_filename == EXPECTED_FILENAME, f"{branch} WSI drift")
        _require(metadata.wsi_sha256 == EXPECTED_SHA256, f"{branch} WSI hash drift")
        _require(
            metadata.source_level == expected["source_level"],
            f"{branch} source level drift",
        )
        _require(
            metadata.source_patch_size == expected["source_patch_size"],
            f"{branch} source patch geometry drift",
        )
        _require(
            metadata.output_patch_size == expected["output_patch_size"],
            f"{branch} output patch geometry drift",
        )
        _require(
            metadata.level_0_declared_footprint == expected["level_0_footprint"],
            f"{branch} level-0 footprint drift",
        )
        _require(
            metadata.effective_mpp == expected["effective_mpp"],
            f"{branch} effective MPP drift",
        )
        _require(metadata.interpolation == expected["interpolation"], f"{branch} interpolation drift")
        _require(metadata.resampling == expected["resampling"], f"{branch} resampling drift")
        branch_results[branch] = {
            "coordinate_count": item.coordinate_count,
            "h5_path": str(item.path),
            "h5_size_bytes": item.size_bytes,
            "h5_sha256": item.sha256,
            "coordinates_sha256": item.coordinates_sha256,
            "source_level": metadata.source_level,
            "source_patch_size": list(metadata.source_patch_size),
            "output_patch_size": list(metadata.output_patch_size),
            "level_0_footprint": list(metadata.level_0_declared_footprint),
            "effective_mpp": list(metadata.effective_mpp),
            "interpolation": metadata.interpolation,
            "resampling": metadata.resampling,
        }
    return result


def _verify_omic(
    paths: PilotPaths,
    loader: Callable[..., object],
) -> tuple[object, dict[str, object], tuple[int, int, int, int]]:
    source, identity_token = _verify_regular_sha256(
        paths.omic,
        label="official BRCA Omic archive",
        size_bytes=OMIC_SIZE_BYTES,
        sha256=BRCA_RELEASE_ARCHIVE_SHA256,
    )
    try:
        with zipfile.ZipFile(paths.omic, mode="r") as archive:
            members = archive.infolist()
            _require(
                len(members) == 1 and not members[0].is_dir(),
                "official BRCA Omic archive must contain exactly one file member",
            )
            member = members[0]
            _require(
                member.filename == OMIC_MEMBER_FILENAME,
                "official BRCA Omic member filename drift",
            )
            _require(
                member.file_size == OMIC_MEMBER_SIZE_BYTES,
                "official BRCA Omic member size drift",
            )
            _require(
                not (member.flag_bits & 0x1),
                "encrypted BRCA Omic member is prohibited",
            )
            member_digest = hashlib.sha256()
            member_bytes = 0
            with archive.open(member, mode="r") as stream:
                for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
                    member_digest.update(chunk)
                    member_bytes += len(chunk)
    except (OSError, zipfile.BadZipFile) as exc:
        raise Q75GpuPilotError("official BRCA Omic ZIP member is unreadable") from exc
    _require(member_bytes == OMIC_MEMBER_SIZE_BYTES, "BRCA Omic member byte count drift")
    _require(
        member_digest.hexdigest() == OMIC_MEMBER_SHA256,
        "official BRCA Omic member SHA-256 drift",
    )
    omic = loader(
        paths.omic,
        case_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_FILENAME,
    )
    _require(getattr(omic, "case_id", None) == EXPECTED_PATIENT_ID, "Omic patient drift")
    _require(getattr(omic, "slide_id", None) == EXPECTED_FILENAME, "Omic slide drift")
    _require(getattr(omic, "source_row_index", None) == "771", "Omic row index drift")
    shapes: dict[str, list[int]] = {}
    content_sha256: dict[str, str] = {}
    for name in ("rna", "mutation", "cnv"):
        tensor = getattr(omic, name, None)
        expected_shape = (1, 1, BRCA_EXPECTED_DIMS[name])
        _require(isinstance(tensor, torch.Tensor), f"Omic {name} is not a tensor")
        _require(tuple(tensor.shape) == expected_shape, f"Omic {name} shape drift")
        _require(tensor.dtype is torch.float32, f"Omic {name} dtype drift")
        _require(tensor.device.type == "cpu", f"Omic {name} must load on CPU")
        _require(tensor.is_contiguous(), f"Omic {name} must be contiguous")
        _require(bool(torch.isfinite(tensor).all().item()), f"Omic {name} is non-finite")
        digest = _tensor_content_sha256(tensor)
        _require(
            digest == EXPECTED_OMIC_CONTENT_SHA256[name],
            f"Omic {name} content SHA-256 drift",
        )
        shapes[name] = list(expected_shape)
        content_sha256[name] = digest
    _require(
        _stat_identity(_regular_non_symlink(paths.omic, "official BRCA Omic archive"))
        == identity_token,
        "official BRCA Omic archive changed while its row was loaded",
    )
    return (
        omic,
        {
            **source,
            "member_filename": OMIC_MEMBER_FILENAME,
            "member_size_bytes": member_bytes,
            "member_sha256": member_digest.hexdigest(),
            "source_row_index": "771",
            "case_id": EXPECTED_PATIENT_ID,
            "slide_id": EXPECTED_FILENAME,
            "shapes": shapes,
            "modality_content_sha256": content_sha256,
            "separate_modalities": True,
        },
        identity_token,
    )


def _verify_slide_header(path: Path, slide_factory: Callable[[str], object]) -> dict[str, object]:
    slide = None
    try:
        slide = slide_factory(str(path))
        properties = slide.properties
        mpp_x = float(properties["openslide.mpp-x"])
        mpp_y = float(properties["openslide.mpp-y"])
        dimensions = tuple(tuple(int(value) for value in pair) for pair in slide.level_dimensions)
        downsamples = tuple(float(value) for value in slide.level_downsamples)
    finally:
        if slide is not None:
            close = getattr(slide, "close", None)
            if callable(close):
                close()
    _require(math.isclose(mpp_x, EXPECTED_MPP_X, rel_tol=0.0, abs_tol=1e-12), "Q75 mpp_x drift")
    _require(math.isclose(mpp_y, EXPECTED_MPP_Y, rel_tol=0.0, abs_tol=1e-12), "Q75 mpp_y drift")
    _require(dimensions == EXPECTED_LEVEL_DIMENSIONS, "Q75 level dimensions drift")
    _require(downsamples == EXPECTED_OPENSLIDE_LEVEL_DOWNSAMPLES, "Q75 level downsamples drift")
    return {
        "header_only_open_count": 1,
        "patch_pixel_reads": 0,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "level_dimensions": [list(pair) for pair in dimensions],
        "level_downsamples": list(downsamples),
    }


def _gpu_preflight() -> tuple[torch.device, dict[str, object]]:
    _require(torch.cuda.is_available(), "CUDA is unavailable; CPU fallback is prohibited")
    _require(torch.cuda.device_count() >= 1, "no CUDA device is exposed")
    device = torch.device(DEVICE)
    name = torch.cuda.get_device_name(device)
    _require(EXPECTED_GPU_NAME_TOKEN in name, f"expected Tesla T4, received {name}")
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(device))
    _require(capability == (7, 5), f"Tesla T4 capability drift: {capability}")
    properties = torch.cuda.get_device_properties(device)
    probe = torch.tensor([1.0], dtype=torch.float32, device=device) + 1.0
    _require(float(probe.item()) == 2.0, "CUDA arithmetic probe failed")
    del probe
    torch.cuda.synchronize(device)

    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    _require(query.returncode == 0, "nvidia-smi query failed")
    rows = [line for line in query.stdout.splitlines() if line.strip()]
    _require(len(rows) >= 1, "nvidia-smi returned no GPU rows")
    fields = [field.strip() for field in rows[0].split(",")]
    _require(len(fields) == 5, "unexpected nvidia-smi result")
    uuid, smi_name, driver, total_mib, free_mib = fields
    _require(EXPECTED_GPU_NAME_TOKEN in smi_name, "nvidia-smi GPU identity drift")
    return device, {
        "device": str(device),
        "name": name,
        "uuid": uuid,
        "compute_capability": list(capability),
        "torch_total_memory_bytes": int(properties.total_memory),
        "nvidia_driver": driver,
        "nvidia_memory_total_mib": int(total_mib),
        "nvidia_memory_free_mib_at_preflight": int(free_mib),
        "cuda_arithmetic_probe": 2.0,
        "cpu_fallback": False,
    }


def _configure_determinism(device: torch.device) -> dict[str, object]:
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == CUBLAS_WORKSPACE_CONFIG,
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before any CUDA operation",
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    _require(torch.backends.cudnn.benchmark is False, "cuDNN benchmark must be disabled")
    _require(torch.backends.cudnn.deterministic is True, "cuDNN deterministic mode drift")
    _require(torch.backends.cudnn.allow_tf32 is False, "cuDNN TF32 must be disabled")
    _require(torch.backends.cuda.matmul.allow_tf32 is False, "matmul TF32 must be disabled")
    _require(
        torch.are_deterministic_algorithms_enabled(),
        "PyTorch deterministic algorithms must be enabled",
    )
    torch.cuda.synchronize(device)
    return {
        "seed": SEED,
        "dtype": "float32",
        "automatic_mixed_precision": False,
        "tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "torch_deterministic_algorithms": True,
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
    }


def _validate_process_environment() -> dict[str, str]:
    """Require the deterministic cuBLAS contract before CUDA is touched."""

    _require(
        not torch.cuda.is_initialized(),
        "CUDA was initialized before deterministic cuBLAS environment validation",
    )
    observed = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    _require(
        observed == CUBLAS_WORKSPACE_CONFIG,
        "CUBLAS_WORKSPACE_CONFIG must be :4096:8 before any CUDA operation",
    )
    return {"CUBLAS_WORKSPACE_CONFIG": observed}


def _validate_smoke_summary(
    smoke: object,
    *,
    label: str,
    expected_input_shapes: list[list[int]],
) -> dict[str, object]:
    summary = _as_strict_mapping(smoke, label)
    expected_attention_shapes = [
        [1, 2, EXPECTED_TOTAL_PATCHES],
        [1, 2, 1],
        [1, 2, 1],
        [1, 2, 1],
    ]
    _require(summary.get("training") is False, f"{label} unexpectedly trained")
    _require(
        summary.get("patch_count") == EXPECTED_TOTAL_PATCHES,
        f"{label} patch count drift",
    )
    _require(summary.get("wsi_layout") == "[1,P,2048]", f"{label} WSI layout drift")
    _require(summary.get("output_shape") == [1, 4], f"{label} output shape drift")
    _require(
        summary.get("input_shapes") == expected_input_shapes,
        f"{label} input shape drift",
    )
    _require(
        summary.get("attention_shapes") == expected_attention_shapes,
        f"{label} attention shape drift",
    )
    _require(summary.get("attention_finite") is True, f"{label} attention is non-finite")
    _require(summary.get("output_finite") is True, f"{label} output is non-finite")
    return summary


def _run_pre_extraction_healnet_contract(
    *,
    deps: PilotDependencies,
    paths: PilotPaths,
    device: torch.device,
    omic: object,
) -> tuple[dict[str, object], int]:
    """Exercise the exact full-size interface before any WSI patch read."""

    rna_source = getattr(omic, "rna", None)
    mutation_source = getattr(omic, "mutation", None)
    cnv_source = getattr(omic, "cnv", None)
    _require(
        all(isinstance(value, torch.Tensor) for value in (rna_source, mutation_source, cnv_source)),
        "pre-extraction HEALNet contract requires three Omic tensors",
    )
    expected_input_shapes = [
        [1, EXPECTED_TOTAL_PATCHES, FEATURE_DIM],
        list(rna_source.shape),
        list(mutation_source.shape),
        list(cnv_source.shape),
    ]
    wsi = torch.zeros(
        (1, EXPECTED_TOTAL_PATCHES, FEATURE_DIM),
        dtype=torch.float32,
        device=device,
    )
    rna = torch.zeros(tuple(rna_source.shape), dtype=torch.float32, device=device)
    mutation = torch.zeros(
        tuple(mutation_source.shape), dtype=torch.float32, device=device
    )
    cnv = torch.zeros(tuple(cnv_source.shape), dtype=torch.float32, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        smoke = deps.smoke_runner(
            official_repo=paths.official_repo,
            wsi=wsi,
            rna=rna,
            mutation=mutation,
            cnv=cnv,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory_bytes = 0
        summary = _validate_smoke_summary(
            smoke,
            label="pre-extraction synthetic HEALNet contract",
            expected_input_shapes=expected_input_shapes,
        )
    finally:
        del wsi, rna, mutation, cnv
        gc.collect()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    return summary, peak_memory_bytes


def _feature_tensor(value: object, *, branch: str, rows: int) -> torch.Tensor:
    tensor = getattr(value, "features", None)
    _require(isinstance(tensor, torch.Tensor), f"{branch} extractor returned no tensor")
    _require(tuple(tensor.shape) == (rows, FEATURE_DIM), f"{branch} feature shape drift")
    _require(tensor.dtype is torch.float32, f"{branch} features must be float32")
    _require(tensor.device.type == "cpu", f"{branch} features must be returned on CPU")
    _require(tensor.is_contiguous(), f"{branch} features must be contiguous")
    _require(not tensor.requires_grad, f"{branch} features must not require gradients")
    _require(tensor.storage_offset() == 0, f"{branch} features must have zero storage offset")
    _require(
        tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(),
        f"{branch} features must use compact storage",
    )
    _require(bool(torch.isfinite(tensor).all().item()), f"{branch} features are non-finite")
    return tensor


def _close_dataset(dataset: object | None) -> None:
    if dataset is None:
        return
    close = getattr(dataset, "close", None)
    if callable(close):
        close()


def _close_datasets(datasets: list[object]) -> None:
    for dataset in datasets:
        _close_dataset(dataset)
    datasets.clear()


def _as_strict_mapping(value: object, label: str) -> dict[str, object]:
    if is_dataclass(value) and not isinstance(value, type):
        document = asdict(value)
    elif isinstance(value, Mapping):
        document = dict(value)
    else:
        raise Q75GpuPilotError(f"{label} did not return a dataclass or mapping")
    try:
        return json.loads(json.dumps(document, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise Q75GpuPilotError(f"{label} result is not strict JSON") from exc


def _build_row_provenance(
    scale_2x_spec: object,
    scale_4x_spec: object,
) -> tuple[PatchProvenance, ...]:
    specs: dict[str, object] = {
        "scale_2x": scale_2x_spec,
        "scale_4x": scale_4x_spec,
    }
    for branch, spec in (("scale_2x", scale_2x_spec), ("scale_4x", scale_4x_spec)):
        coordinates = getattr(spec, "coordinates", None)
        expected_count = int(EXPECTED_COORDINATE_BRANCHES[branch]["count"])
        _require(isinstance(coordinates, torch.Tensor), f"{branch} spec lacks coordinates")
        _require(tuple(coordinates.shape) == (expected_count, 2), f"{branch} spec coordinate shape drift")
        _require(coordinates.dtype is torch.int64, f"{branch} spec coordinates must be int64")
        _require(coordinates.device.type == "cpu", f"{branch} coordinates must remain on CPU")
        _require(
            getattr(spec, "source_level", None)
            == EXPECTED_COORDINATE_BRANCHES[branch]["source_level"],
            f"{branch} spec source level drift",
        )
        _require(
            getattr(spec, "effective_mpp", None)
            == EXPECTED_COORDINATE_BRANCHES[branch]["effective_mpp"],
            f"{branch} spec effective MPP drift",
        )

    provenance = build_two_scale_provenance(
        scale_2x=BranchProvenanceSpec(
            branch="scale_2x",
            coordinates=getattr(specs["scale_2x"], "coordinates"),
            level=int(getattr(specs["scale_2x"], "source_level")),
            mpp_x=float(getattr(specs["scale_2x"], "effective_mpp")[0]),
            mpp_y=float(getattr(specs["scale_2x"], "effective_mpp")[1]),
        ),
        scale_4x=BranchProvenanceSpec(
            branch="scale_4x",
            coordinates=getattr(specs["scale_4x"], "coordinates"),
            level=int(getattr(specs["scale_4x"], "source_level")),
            mpp_x=float(getattr(specs["scale_4x"], "effective_mpp")[0]),
            mpp_y=float(getattr(specs["scale_4x"], "effective_mpp")[1]),
        ),
        scale_2x_count=int(EXPECTED_COORDINATE_BRANCHES["scale_2x"]["count"]),
        scale_4x_count=int(EXPECTED_COORDINATE_BRANCHES["scale_4x"]["count"]),
    )
    validate_provenance_alignment(EXPECTED_TOTAL_PATCHES, provenance)
    return provenance


def _coordinate_binding(branch: str, record: object) -> CoordinateFeatureBinding:
    metadata = getattr(record, "metadata", None)
    return CoordinateFeatureBinding(
        branch=branch,
        artifact_filename=getattr(record, "path").name,
        artifact_size_bytes=int(getattr(record, "size_bytes")),
        artifact_sha256=str(getattr(record, "sha256")),
        coordinates_sha256=str(getattr(record, "coordinates_sha256")),
        coordinate_count=int(getattr(record, "coordinate_count")),
        source_level=int(getattr(metadata, "source_level")),
        effective_mpp_x=float(getattr(metadata, "effective_mpp")[0]),
        effective_mpp_y=float(getattr(metadata, "effective_mpp")[1]),
    )


def _feature_artifact_metadata(
    *,
    coordinate_record: object,
    repositories: Mapping[str, object],
) -> FeatureArtifactMetadata:
    manifest_path = getattr(coordinate_record, "manifest_path", None)
    _require(isinstance(manifest_path, Path), "coordinate manifest path missing")
    manifest_metadata = _regular_non_symlink(manifest_path, "coordinate manifest")
    critical_hashes = repositories.get("critical_execution_source_sha256")
    _require(isinstance(critical_hashes, dict) and critical_hashes, "critical source hashes missing")
    source_head = repositories.get("source_head")
    _require(isinstance(source_head, str), "implementation Git commit missing")
    return FeatureArtifactMetadata(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        wsi_filename=EXPECTED_FILENAME,
        wsi_size_bytes=EXPECTED_SIZE_BYTES,
        wsi_md5=EXPECTED_MD5,
        wsi_sha256=EXPECTED_SHA256,
        coordinate_manifest_filename=manifest_path.name,
        coordinate_manifest_size_bytes=manifest_metadata.st_size,
        coordinate_manifest_sha256=COORDINATE_MANIFEST_SHA256,
        scale_2x_coordinates=_coordinate_binding(
            "scale_2x", coordinate_record.branch_for("scale_2x")
        ),
        scale_4x_coordinates=_coordinate_binding(
            "scale_4x", coordinate_record.branch_for("scale_4x")
        ),
        omic_archive_filename=OMIC_PATH.name,
        omic_archive_size_bytes=OMIC_SIZE_BYTES,
        omic_archive_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
        omic_member_filename=OMIC_MEMBER_FILENAME,
        omic_member_size_bytes=OMIC_MEMBER_SIZE_BYTES,
        omic_member_sha256=OMIC_MEMBER_SHA256,
        omic_source_row_index="771",
        rna_feature_count=BRCA_EXPECTED_DIMS["rna"],
        mutation_feature_count=BRCA_EXPECTED_DIMS["mutation"],
        cnv_feature_count=BRCA_EXPECTED_DIMS["cnv"],
        rna_content_sha256=EXPECTED_OMIC_CONTENT_SHA256["rna"],
        mutation_content_sha256=EXPECTED_OMIC_CONTENT_SHA256["mutation"],
        cnv_content_sha256=EXPECTED_OMIC_CONTENT_SHA256["cnv"],
        encoder_identity=EXPECTED_ENCODER_IDENTITY,
        checkpoint_filename=CHECKPOINT_PATH.name,
        checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        checkpoint_sha256=CHECKPOINT_SHA256,
        source_policy_name=AUTH_RELATIVE_PATH.name,
        source_policy_sha256=AUTH_SHA256,
        implementation_git_commit=source_head,
        implementation_source_sha256=critical_hashes,
    )


def _runtime_versions() -> dict[str, object]:
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "h5py": h5py.__version__,
        "openslide_python": openslide.__version__,
        "openslide_library": openslide.__library_version__,
        "pillow": pillow_version,
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def run_q75_gpu_pilot(
    *,
    paths: PilotPaths = PilotPaths(),
    dependencies: PilotDependencies | None = None,
    gpu_preflight: Callable[[], tuple[torch.device, dict[str, object]]] = _gpu_preflight,
) -> dict[str, object]:
    """Execute the one exact Q75 pilot; all injected seams are for mocks only."""

    # This is intentionally the first operational gate.  CPU-only preparation
    # authority cannot trigger filesystem preflight, OpenSlide, CUDA, models,
    # or a pixel-capable dataset.
    _require_execution_authorized()
    deps = dependencies if dependencies is not None else _default_dependencies()
    started = time.perf_counter()
    started_at = datetime.now(timezone.utc).isoformat()
    timings: dict[str, float] = {}

    # Every authorization and immutable-input check precedes construction of a
    # dataset capable of reading level-0/level-1 pixels.
    _validate_paths(paths)
    bound_sources = _validate_bound_sources(paths)
    repositories_before = _repository_snapshot(paths)

    stage = time.perf_counter()
    wsi, wsi_identity_token = _verify_wsi(paths.wsi)
    adapter_wsi_identity = deps.wsi_identity_capture(paths.wsi)
    _require(
        _adapter_identity_tuple(adapter_wsi_identity) == wsi_identity_token,
        "Q75 adapter WSI identity differs from the hash-gated payload",
    )
    timings["wsi_identity_and_hash_seconds"] = time.perf_counter() - stage
    checkpoint, checkpoint_identity_token = _verify_regular_sha256(
        paths.checkpoint,
        label="ResNet50 ImageNet1K V2 checkpoint",
        size_bytes=CHECKPOINT_SIZE_BYTES,
        sha256=CHECKPOINT_SHA256,
    )
    checkpoint["identity"] = CHECKPOINT_IDENTITY

    stage = time.perf_counter()
    coordinate_record = deps.coordinate_validator(
        paths.coordinates,
        expected_manifest_sha256=COORDINATE_MANIFEST_SHA256,
    )
    coordinate_summary = _validate_coordinate_record(coordinate_record)
    scale_2x_record = coordinate_record.branch_for("scale_2x")
    scale_4x_record = coordinate_record.branch_for("scale_4x")
    scale_2x_spec = deps.branch_spec_loader(scale_2x_record)
    scale_4x_spec = deps.branch_spec_loader(scale_4x_record)
    row_provenance = _build_row_provenance(scale_2x_spec, scale_4x_spec)
    timings["coordinate_validation_seconds"] = time.perf_counter() - stage

    stage = time.perf_counter()
    omic, omic_summary, omic_identity_token = _verify_omic(paths, deps.omic_loader)
    timings["omic_validation_seconds"] = time.perf_counter() - stage

    header = _verify_slide_header(paths.wsi, deps.slide_factory)
    _require(
        _stat_identity(_regular_non_symlink(paths.wsi, "Q75 WSI"))
        == wsi_identity_token,
        "Q75 WSI changed during header preflight",
    )
    # This environment variable must already exist before _gpu_preflight makes
    # the first CUDA call.  Setting it after CUDA initialization is too late to
    # guarantee deterministic cuBLAS behavior.
    process_environment = _validate_process_environment()
    device, gpu = gpu_preflight()
    _require(str(device) == DEVICE, "GPU preflight selected a non-authorized device")
    determinism = _configure_determinism(device)

    stage = time.perf_counter()
    pre_extraction_smoke, pre_extraction_smoke_peak = (
        _run_pre_extraction_healnet_contract(
            deps=deps,
            paths=paths,
            device=device,
            omic=omic,
        )
    )
    timings["pre_extraction_synthetic_healnet_seconds"] = (
        time.perf_counter() - stage
    )

    # Re-evaluate all cheap, mutable gates immediately before the first object
    # capable of a patch read is created.
    _validate_paths(paths)
    _require(
        _repository_snapshot(paths) == repositories_before,
        "repository state changed during preflight",
    )
    current_wsi = paths.wsi.lstat()
    _require(
        (
            current_wsi.st_dev,
            current_wsi.st_ino,
            current_wsi.st_size,
            current_wsi.st_mtime_ns,
        )
        == wsi_identity_token,
        "Q75 WSI changed during preflight",
    )

    model: object | None = None
    datasets: list[object] = []
    scale_2x_result: object | None = None
    scale_4x_result: object | None = None
    stage = time.perf_counter()
    try:
        model = deps.model_builder(paths.checkpoint)
        checkpoint_after_load, checkpoint_after_load_token = _verify_regular_sha256(
            paths.checkpoint,
            label="ResNet50 ImageNet1K V2 checkpoint",
            size_bytes=CHECKPOINT_SIZE_BYTES,
            sha256=CHECKPOINT_SHA256,
        )
        _require(
            checkpoint_after_load_token == checkpoint_identity_token
            and checkpoint_after_load == {
                key: value for key, value in checkpoint.items() if key != "identity"
            },
            "checkpoint changed during strict ResNet50 load",
        )
        scale_2x_dataset = deps.dataset_factory(
            paths.wsi,
            scale_2x_spec,
            adapter_wsi_identity,
        )
        datasets.append(scale_2x_dataset)
        scale_2x_result = deps.feature_extractor(
            scale_2x_dataset,
            model,
            device=device,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        _close_dataset(scale_2x_dataset)

        # The second branch is never constructed until 2x extraction succeeds.
        scale_4x_dataset = deps.dataset_factory(
            paths.wsi,
            scale_4x_spec,
            adapter_wsi_identity,
        )
        datasets.append(scale_4x_dataset)
        scale_4x_result = deps.feature_extractor(
            scale_4x_dataset,
            model,
            device=device,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )
        _close_dataset(scale_4x_dataset)
    finally:
        _close_datasets(datasets)
        # Drop the caller's final ResNet reference before clearing the CUDA
        # allocator; the HEALNet smoke starts only after this completes.
        model = None
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
    timings["sequential_resnet_extraction_seconds"] = time.perf_counter() - stage

    _require(scale_2x_result is not None, "2x extraction did not complete")
    _require(scale_4x_result is not None, "4x extraction did not complete")
    features_2x = _feature_tensor(
        scale_2x_result,
        branch="scale_2x",
        rows=int(EXPECTED_COORDINATE_BRANCHES["scale_2x"]["count"]),
    )
    features_4x = _feature_tensor(
        scale_4x_result,
        branch="scale_4x",
        rows=int(EXPECTED_COORDINATE_BRANCHES["scale_4x"]["count"]),
    )
    combined = torch.cat([features_2x, features_4x], dim=0).contiguous()
    _require(
        tuple(combined.shape) == (EXPECTED_TOTAL_PATCHES, FEATURE_DIM),
        "combined feature shape drift",
    )
    _require(combined.dtype is torch.float32 and combined.device.type == "cpu", "combined feature dtype/device drift")
    _require(combined.is_contiguous(), "combined features must be contiguous")
    _require(bool(torch.isfinite(combined).all().item()), "combined features are non-finite")
    split = int(EXPECTED_COORDINATE_BRANCHES["scale_2x"]["count"])
    _require(torch.equal(combined[:split], features_2x), "combined 2x row order drift")
    _require(torch.equal(combined[split:], features_4x), "combined 4x row order drift")

    # Natural HEALNet layout is unsqueeze-only.  A transpose here would turn
    # patches into channels and is explicitly prohibited by the authorization.
    wsi_input = combined.unsqueeze(0).contiguous().to(device=device, dtype=torch.float32)
    _require(
        tuple(wsi_input.shape) == (1, EXPECTED_TOTAL_PATCHES, FEATURE_DIM),
        "natural WSI model shape drift; transpose is prohibited",
    )
    _require(torch.equal(wsi_input[0].cpu(), combined), "natural WSI layout changed row order")
    rna = omic.rna.to(device=device, dtype=torch.float32).contiguous()
    mutation = omic.mutation.to(device=device, dtype=torch.float32).contiguous()
    cnv = omic.cnv.to(device=device, dtype=torch.float32).contiguous()
    torch.cuda.reset_peak_memory_stats(device)
    smoke_started = time.perf_counter()
    try:
        smoke = deps.smoke_runner(
            official_repo=paths.official_repo,
            wsi=wsi_input,
            rna=rna,
            mutation=mutation,
            cnv=cnv,
        )
        torch.cuda.synchronize(device)
        healnet_peak = int(torch.cuda.max_memory_allocated(device))
    finally:
        del wsi_input, rna, mutation, cnv
        gc.collect()
        torch.cuda.empty_cache()
    timings["healnet_interface_seconds"] = time.perf_counter() - smoke_started
    smoke_summary = _validate_smoke_summary(
        smoke,
        label="real-feature HEALNet smoke",
        expected_input_shapes=[
            [1, EXPECTED_TOTAL_PATCHES, FEATURE_DIM],
            list(omic.rna.shape),
            list(omic.mutation.shape),
            list(omic.cnv.shape),
        ],
    )

    extraction_summary = {
        "scale_2x": {
            "shape": list(features_2x.shape),
            "dtype": "float32",
            "all_finite": True,
            "streaming_extraction_seconds": float(
                getattr(scale_2x_result, "streaming_extraction_seconds")
            ),
            "model_forward_seconds": float(
                getattr(scale_2x_result, "model_forward_seconds")
            ),
            "peak_gpu_memory_bytes": int(
                getattr(scale_2x_result, "peak_gpu_memory_bytes")
            ),
        },
        "scale_4x": {
            "shape": list(features_4x.shape),
            "dtype": "float32",
            "all_finite": True,
            "streaming_extraction_seconds": float(
                getattr(scale_4x_result, "streaming_extraction_seconds")
            ),
            "model_forward_seconds": float(
                getattr(scale_4x_result, "model_forward_seconds")
            ),
            "peak_gpu_memory_bytes": int(
                getattr(scale_4x_result, "peak_gpu_memory_bytes")
            ),
        },
        "combined": {
            "operation": "torch.cat",
            "dim": 0,
            "branch_order": ["scale_2x", "scale_4x"],
            "shape": list(combined.shape),
            "dtype": "float32",
            "device": "cpu",
            "all_finite": True,
            "row_provenance_count": len(row_provenance),
            "natural_model_shape": [1, EXPECTED_TOTAL_PATCHES, FEATURE_DIM],
            "transpose_performed": False,
            "pooling_performed": False,
        },
    }
    artifact_metadata = _feature_artifact_metadata(
        coordinate_record=coordinate_record,
        repositories=repositories_before,
    )

    # Complete every fallible external-input and repository gate before the
    # artifact module reaches its irreversible RENAME_NOREPLACE commit point.
    _validate_paths(paths)
    final_wsi, final_wsi_identity_token = _verify_wsi(paths.wsi)
    _require(
        final_wsi == wsi and final_wsi_identity_token == wsi_identity_token,
        "Q75 WSI changed during extraction",
    )
    final_checkpoint, final_checkpoint_identity_token = _verify_regular_sha256(
        paths.checkpoint,
        label="ResNet50 ImageNet1K V2 checkpoint",
        size_bytes=CHECKPOINT_SIZE_BYTES,
        sha256=CHECKPOINT_SHA256,
    )
    _require(
        final_checkpoint_identity_token == checkpoint_identity_token
        and final_checkpoint
        == {key: value for key, value in checkpoint.items() if key != "identity"},
        "checkpoint changed during extraction",
    )
    final_omic, final_omic_summary, final_omic_identity_token = _verify_omic(
        paths,
        deps.omic_loader,
    )
    _require(
        final_omic_identity_token == omic_identity_token
        and final_omic_summary == omic_summary
        and all(
            torch.equal(getattr(final_omic, name), getattr(omic, name))
            for name in ("rna", "mutation", "cnv")
        ),
        "official BRCA Omic archive changed during extraction",
    )
    final_coordinate_record = deps.coordinate_validator(
        paths.coordinates,
        expected_manifest_sha256=COORDINATE_MANIFEST_SHA256,
    )
    _require(
        _validate_coordinate_record(final_coordinate_record) == coordinate_summary,
        "coordinate artifacts changed during extraction",
    )
    repositories_prepublication = _repository_snapshot(paths)
    _require(
        repositories_prepublication == repositories_before,
        "repository state changed during pilot",
    )
    runtime_versions = _runtime_versions()
    prepublication_checks = {
        "output_absent_and_outside_git": True,
        "wsi_identity_size_md5_sha256_unchanged": True,
        "checkpoint_identity_size_sha256_unchanged": True,
        "omic_identity_size_sha256_unchanged": True,
        "coordinate_manifest_and_branches_unchanged": True,
        "official_repository_unchanged": True,
        "frozen_blca_tag_unchanged": True,
        "pilot_repository_unchanged": True,
    }

    # The feature-artifact module owns a no-overwrite sibling-staging
    # transaction.  It strictly validates the staging directory, commits with
    # RENAME_NOREPLACE, and revalidates the published directory before return.
    # No external-input semantic gate follows this terminal publication call.
    artifact_record = deps.artifact_publisher(
        paths.output,
        scale_2x_features=features_2x,
        scale_4x_features=features_4x,
        combined_features=combined,
        row_provenance=row_provenance,
        metadata=artifact_metadata,
    )
    manifest_sha256 = artifact_record.manifest_sha256
    timings["total_seconds"] = time.perf_counter() - started

    return {
        "schema": SCHEMA,
        "status": SUCCESS_STATUS,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id": EXPECTED_PATIENT_ID,
        "slide_id": EXPECTED_SLIDE_ID,
        "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
        "authorization_sha256": AUTH_SHA256,
        "bound_source_sha256": bound_sources,
        "source_commit_at_execution": repositories_before["source_head"],
        "critical_execution_source_sha256": repositories_before[
            "critical_execution_source_sha256"
        ],
        "repositories_unchanged": True,
        "runtime_versions": runtime_versions,
        "process_environment": process_environment,
        "gpu": gpu,
        "determinism": determinism,
        "wsi": wsi,
        "wsi_header": header,
        "coordinates": coordinate_summary,
        "omic": omic_summary,
        "checkpoint": checkpoint,
        "features": extraction_summary,
        "pre_extraction_healnet_contract": {
            **pre_extraction_smoke,
            "synthetic_zero_inputs": True,
            "ran_before_pixel_capable_dataset": True,
            "interface_seconds": timings[
                "pre_extraction_synthetic_healnet_seconds"
            ],
            "peak_gpu_memory_bytes": pre_extraction_smoke_peak,
            "trained_prediction": False,
        },
        "healnet_smoke": {
            **smoke_summary,
            "interface_seconds": timings["healnet_interface_seconds"],
            "peak_gpu_memory_bytes": healnet_peak,
            "trained_prediction": False,
        },
        "feature_artifacts": {
            "directory": str(paths.output),
            "manifest_sha256": manifest_sha256,
            "atomic_directory_publication": True,
            "validated_after_publication": True,
        },
        "prepublication_checks": prepublication_checks,
        "operations": {
            "patients_processed": 1,
            "scale_2x_patch_reads": int(EXPECTED_COORDINATE_BRANCHES["scale_2x"]["count"]),
            "scale_4x_patch_reads": int(EXPECTED_COORDINATE_BRANCHES["scale_4x"]["count"]),
            "resnet50_feature_rows": EXPECTED_TOTAL_PATCHES,
            "healnet_interface_smokes": 2,
            "synthetic_pre_extraction_interface_smokes": 1,
            "real_feature_interface_smokes": 1,
            "training": 0,
            "backward_passes": 0,
            "optimizer_steps": 0,
            "q25_q50_operations": 0,
            "q75_coordinate_regeneration": 0,
            "q75_patch_images_persisted": 0,
            "full_cohort_operations": 0,
            "google_drive_operations": 0,
            "raw_wsi_deletions": 0,
        },
        "scientific_interpretation": "NOT A TRAINED SURVIVAL PREDICTION",
        "required_stop_reached": True,
        "timings": timings,
    }


def _strict_json(document: Mapping[str, object]) -> str:
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-authorized-q75-gpu-pilot",
        action="store_true",
        help="Acknowledge the exact Q75-only GPU feature/interface boundary.",
    )
    args = parser.parse_args(argv)
    if not args.execute_authorized_q75_gpu_pilot:
        parser.error("explicit --execute-authorized-q75-gpu-pilot is required")
    try:
        result = run_q75_gpu_pilot()
    except Exception as exc:
        print(
            _strict_json(
                {
                    "schema": SCHEMA,
                    "status": BLOCKED_STATUS,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "required_stop_reached": True,
                }
            ),
            file=sys.stderr,
        )
        return 1
    print(_strict_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
