"""Fail-closed, CPU-only BRCA Q75 exact-file and OpenSlide-header gate.

The gate holds secure file descriptors for the WSI and released Omic archive,
gives OpenSlide a stable ``/proc/self/fd`` pathname, reads only five header
attributes, and atomically publishes one no-replace directory containing the
YAML result and Markdown report. Success never approves pixel access or a Q75
scale policy.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

import torch
import yaml

from multiscale_feature_pilot.src.brca_omic import (
    BRCA_EXPECTED_DIMS,
    BRCA_RELEASE_ARCHIVE_SHA256,
    BrcaPatientOmics,
    load_brca_patient_omics,
)
from multiscale_feature_pilot.src.brca_q75_authorized_manifest import (
    EXPECTED_AUTHORIZATION_CONFIG_SHA256 as AUTHORIZED_CONFIG_SHA256,
    EXPECTED_PROHIBITED_ACTIONS,
)
from multiscale_feature_pilot.src.brca_q75_download_runner import (
    DEFAULT_GDC_CLIENT,
    EXPECTED_GDC_CLIENT_SHA256,
    EXPECTED_GDC_CLIENT_SIZE_BYTES,
    EXPECTED_GDC_CLIENT_VERSION,
    SOURCE_RELATIVE_PATHS as DOWNLOAD_SOURCE_RELATIVE_PATHS,
)
from multiscale_feature_pilot.src.wsi_metadata_policy import (
    APPROVED_PER_AXIS_TOLERANCE_FRACTION,
    NativeLevelMetadata,
    WsiMetadataPolicyError,
    validate_metadata_pyramid,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent

EXPECTED_PATIENT_ID = "TCGA-E2-A154"
EXPECTED_SLIDE_ID = (
    "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
)
EXPECTED_FILENAME = EXPECTED_SLIDE_ID
EXPECTED_GDC_UUID = "25aec062-60d1-446e-a1c6-0c79cc74a770"
EXPECTED_SIZE_BYTES = 1_360_743_825
EXPECTED_MD5 = "a8c4b68fb6e0ab3e862efe3ed1fe10d7"
EXPECTED_OMIC_SOURCE_INDEX = "771"
EXPECTED_MANIFEST_SHA256 = (
    "8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a"
)
EXPECTED_AUTHORIZATION_SHA256 = AUTHORIZED_CONFIG_SHA256
EXPECTED_AUTHORIZATION_RECORD_SHA256 = (
    "2330c4bc66c73c8e150be2d028aefb2a84916b18e4b5076d95fc28cf869d7050"
)

DEFAULT_AUTHORIZATION = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/config/brca_q75_acquisition_authorization.yaml"
)
DEFAULT_AUTHORIZATION_RECORD = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/"
    "MANIFEST_SET.AUTHORIZED_Q75.yaml"
)
DEFAULT_AUTHORIZED_MANIFEST = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/"
    "Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770."
    "AUTHORIZED_Q75_ONLY.gdc.tsv"
)
DEFAULT_DOWNLOAD_RESULT_DIRECTORY = WORKSPACE_ROOT / "brca_pilot_data/Q75.download_result"
DEFAULT_DOWNLOAD_RESULT = DEFAULT_DOWNLOAD_RESULT_DIRECTORY / "download_result.yaml"
DEFAULT_INCOMING_DIRECTORY = WORKSPACE_ROOT / "brca_pilot_data/Q75.incoming"
DEFAULT_WSI = DEFAULT_INCOMING_DIRECTORY / EXPECTED_GDC_UUID / EXPECTED_FILENAME
DEFAULT_OMIC_ARCHIVE = (
    WORKSPACE_ROOT
    / "Author_Official_Repo_directery/healnet/data/tcga/omic/"
    "tcga_brca_all_clean.csv.zip"
)
DEFAULT_OUTPUT_DIRECTORY = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_q75_header_metadata_result"
)
RESULT_BASENAME = "result.yaml"
REPORT_BASENAME = "report.md"

PROTECTED_DIRTY_STATUS = " M reports/blca_one_patient_multiscale_pilot.md"
CRITICAL_RELATIVE_PATHS = (
    Path("multiscale_feature_pilot/config/brca_q75_acquisition_authorization.yaml"),
    Path(
        "multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/"
        "MANIFEST_SET.AUTHORIZED_Q75.yaml"
    ),
    Path(
        "multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/"
        "Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770."
        "AUTHORIZED_Q75_ONLY.gdc.tsv"
    ),
    Path("multiscale_feature_pilot/src/brca_omic.py"),
    Path("multiscale_feature_pilot/src/brca_q75_authorized_manifest.py"),
    Path("multiscale_feature_pilot/src/brca_q75_download_runner.py"),
    Path("multiscale_feature_pilot/src/brca_q75_header_gate.py"),
    Path("multiscale_feature_pilot/src/wsi_metadata_policy.py"),
    Path("scripts/run_brca_q75_download.py"),
    Path("scripts/run_brca_q75_header_gate.py"),
)

RESULT_STATUS = "BRCA_Q75_FILE_AND_HEADER_METADATA_VERIFIED"
REQUIRED_STOP = "Q75_EXACT_FILE_OMIC_AND_OPENSLIDE_HEADER_METADATA_ONLY"
EXPECTED_DOWNLOAD_STATUS = "BRCA_Q75_GDC_DOWNLOAD_VERIFIED"
EXPECTED_DOWNLOAD_STOP = "Q75_GDC_DOWNLOAD_AND_EXACT_FILE_VERIFICATION_ONLY"
EXPECTED_DOWNLOAD_OPERATIONS = {
    "gdc_downloads": 1,
    "wsi_opens": 0,
    "pixel_or_region_reads": 0,
    "tissue_masks": 0,
    "coordinate_generation": 0,
    "patch_extraction": 0,
    "resnet50_inference": 0,
    "healnet_execution": 0,
    "q75_feature_generation": 0,
    "q75_raw_file_deletions": 0,
    "google_drive_operations": 0,
    "full_cohort_processing": 0,
    "q25_q50_modifications": 0,
    "blca_modifications": 0,
    "official_healnet_modifications": 0,
    "training_runs": 0,
    "backward_passes": 0,
    "optimizer_steps": 0,
}
_CHUNK_SIZE = 8 * 1024 * 1024
_TARGETS = (
    ("approximately_0_5_um_per_px", 0.5),
    ("approximately_1_0_um_per_px", 1.0),
)


class Q75HeaderGateError(RuntimeError):
    """Raised on any authorization, identity, integrity, or scope violation."""


@dataclass(frozen=True)
class GatePaths:
    """Internal/test bundle; the production CLI always uses exact defaults."""

    repo_root: Path = REPOSITORY_ROOT
    authorization: Path = DEFAULT_AUTHORIZATION
    authorization_record: Path = DEFAULT_AUTHORIZATION_RECORD
    manifest: Path = DEFAULT_AUTHORIZED_MANIFEST
    download_result_directory: Path = DEFAULT_DOWNLOAD_RESULT_DIRECTORY
    download_result: Path = DEFAULT_DOWNLOAD_RESULT
    incoming_directory: Path = DEFAULT_INCOMING_DIRECTORY
    wsi: Path = DEFAULT_WSI
    omic_archive: Path = DEFAULT_OMIC_ARCHIVE
    output_directory: Path = DEFAULT_OUTPUT_DIRECTORY


@dataclass(frozen=True)
class FileToken:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, metadata: os.stat_result) -> "FileToken":
        return cls(
            device=metadata.st_dev,
            inode=metadata.st_ino,
            size=metadata.st_size,
            mtime_ns=metadata.st_mtime_ns,
            ctime_ns=metadata.st_ctime_ns,
        )


@dataclass(frozen=True)
class HeldRegularFile:
    path: Path
    descriptor: int
    token: FileToken
    sha256: str

    @property
    def stable_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.descriptor}")


@dataclass(frozen=True)
class VerifiedWsi:
    path: str
    descriptor: int
    token: FileToken
    size_bytes: int
    md5: str
    sha256: str

    @property
    def stable_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.descriptor}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q75HeaderGateError(message)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    _require(path.is_absolute(), f"path must be absolute: {path}")
    stop = len(path.parts) if include_leaf else len(path.parts) - 1
    current = Path(path.parts[0])
    for part in path.parts[1:stop]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise Q75HeaderGateError(f"missing path component: {current}") from exc
        _require(not stat.S_ISLNK(metadata.st_mode), f"symlink component prohibited: {current}")


def _path_token(path: Path, *, label: str) -> FileToken:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Q75HeaderGateError(f"missing {label}: {path}") from exc
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} must not be a symlink")
    _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
    return FileToken.from_stat(metadata)


def _open_no_follow(path: Path, *, label: str) -> tuple[int, FileToken]:
    _require_no_symlink_components(path, include_leaf=False)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Q75HeaderGateError(f"cannot securely open {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
        token = FileToken.from_stat(metadata)
        _require(_path_token(path, label=label) == token, f"{label} pathname/open mismatch")
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, token


def _read_descriptor(descriptor: int, *, maximum: int) -> bytes:
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    total = 0
    while True:
        remaining = maximum + 1 - total
        chunk = os.read(descriptor, min(_CHUNK_SIZE, remaining))
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        _require(total <= maximum, "secure file read exceeds size limit")
    return b"".join(chunks)


def _secure_regular_bytes(path: Path, *, label: str, maximum: int) -> bytes:
    descriptor, token = _open_no_follow(path, label=label)
    try:
        _require(token.size <= maximum, f"{label} exceeds its size limit")
        payload = _read_descriptor(descriptor, maximum=maximum)
        _require(FileToken.from_stat(os.fstat(descriptor)) == token, f"{label} changed")
        _require(_path_token(path, label=label) == token, f"{label} pathname changed")
        return payload
    finally:
        os.close(descriptor)


def _hash_descriptor(
    descriptor: int, *, include_md5: bool
) -> tuple[str | None, str]:
    os.lseek(descriptor, 0, os.SEEK_SET)
    md5 = hashlib.md5(usedforsecurity=False) if include_md5 else None
    sha256 = hashlib.sha256()
    while True:
        chunk = os.read(descriptor, _CHUNK_SIZE)
        if not chunk:
            break
        if md5 is not None:
            md5.update(chunk)
        sha256.update(chunk)
    return (md5.hexdigest() if md5 is not None else None, sha256.hexdigest())


def _yaml_mapping(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise Q75HeaderGateError(f"cannot parse {label}") from exc
    _require(isinstance(value, Mapping), f"{label} root must be a mapping")
    return value


def _validate_authorized_identity(value: object, *, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    expected = {
        "patient_id": EXPECTED_PATIENT_ID,
        "wsi_uuid": EXPECTED_GDC_UUID,
        "filename": EXPECTED_FILENAME,
        "declared_bytes": EXPECTED_SIZE_BYTES,
        "md5": EXPECTED_MD5,
        "state": "released",
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
    _require(dict(value) == expected, f"{label} identity drift")


def validate_authorization(path: Path, *, expected_sha256: str) -> str:
    _require(_valid_sha256(expected_sha256), "authorization SHA256 is not pinned")
    payload = _secure_regular_bytes(
        path, label="Q75 authorization", maximum=2 * 1024 * 1024
    )
    digest = _sha256_bytes(payload)
    _require(digest == expected_sha256, "Q75 authorization SHA256 mismatch")
    document = _yaml_mapping(payload, label="Q75 authorization")
    _require(document.get("schema_version") == 1, "authorization schema drift")
    _require(
        document.get("phase") == "BRCA_Q75_ACQUISITION_AND_HEADER_METADATA_GATE",
        "authorization phase drift",
    )
    _require(
        document.get("status")
        == "AUTHORIZED_Q75_ACQUISITION_AND_HEADER_METADATA_ONLY",
        "Q75 header gate is not authorized",
    )
    _require(document.get("cohort") == "TCGA-BRCA", "authorization cohort drift")
    _require(document.get("candidate") == "Q75", "authorization candidate drift")
    _require(
        document.get("execution_mode") == "CPU_LOGIC_ONLY_CUDA_NOT_REQUIRED",
        "authorization CPU directive drift",
    )
    _validate_authorized_identity(document.get("authorized_q75"), label="authorized_q75")
    authority = document.get("authority")
    _require(isinstance(authority, Mapping), "authorization authority missing")
    download = authority.get("exact_wsi_download")
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
    _require(isinstance(download, Mapping) and dict(download) == expected_download, "download authority drift")
    header = authority.get("wsi_header_read")
    _require(isinstance(header, Mapping), "Q75 header authority missing")
    _require(
        header.get("status") == "APPROVED_Q75_HEADER_ONLY_AFTER_ALL_FILE_CHECKS",
        "Q75 header scope drift",
    )
    _require(
        header.get("allowed_fields")
        == ["mpp_x", "mpp_y", "level_count", "level_dimensions", "level_downsamples"],
        "Q75 header allowlist drift",
    )
    _require(header.get("openslide_construction_allowed") is True, "OpenSlide locked")
    _require(header.get("read_region_or_pixel_access") == "NOT_AUTHORIZED", "pixels unlocked")
    execution = document.get("execution_contract")
    _require(isinstance(execution, Mapping), "execution contract missing")
    for key in (
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
    ):
        _require(execution.get(key) is True, f"execution contract missing {key}")
    _require(execution.get("pass_n_processes_1") is False, "invalid -n 1 enabled")
    _require(execution.get("infer_or_approve_scale_policy") is False, "scale unlocked")
    _require(execution.get("use_cuda") is False, "CUDA unlocked")
    _require(
        execution.get("stop_after")
        == "Q75_EXACT_FILE_VERIFICATION_AND_HEADER_METADATA_REPORT",
        "Q75 stop boundary drift",
    )
    _require(document.get("prohibited_actions") == EXPECTED_PROHIBITED_ACTIONS, "prohibited action drift")
    return digest


def validate_authorization_record(
    path: Path, *, expected_sha256: str, authorization_sha256: str
) -> str:
    _require(_valid_sha256(expected_sha256), "authorization-record SHA256 is not pinned")
    payload = _secure_regular_bytes(
        path, label="Q75 authorization record", maximum=1024 * 1024
    )
    digest = _sha256_bytes(payload)
    _require(digest == expected_sha256, "Q75 authorization-record SHA256 mismatch")
    record = _yaml_mapping(payload, label="Q75 authorization record")
    expected_scalars = {
        "schema_version": 1,
        "policy_label": "BRCA_Q75_SEQUENTIAL_ACQUISITION_V1",
        "status": "AUTHORIZED_Q75_ONLY",
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
    }
    for key, expected in expected_scalars.items():
        _require(record.get(key) == expected, f"authorization record {key} drift")
    binding = record.get("approval_binding")
    _require(
        isinstance(binding, Mapping)
        and binding.get("authorization_config_sha256") == authorization_sha256,
        "authorization-record config binding mismatch",
    )
    entry = record.get("entry")
    expected_entry = {
        "label": "Q75",
        "patient_id": EXPECTED_PATIENT_ID,
        "basename": DEFAULT_AUTHORIZED_MANIFEST.name,
        "status": "AUTHORIZED_Q75_ONLY",
        "rows": 1,
        "sha256": EXPECTED_MANIFEST_SHA256,
        "id": EXPECTED_GDC_UUID,
        "filename": EXPECTED_FILENAME,
        "md5": EXPECTED_MD5,
        "size": str(EXPECTED_SIZE_BYTES),
        "state": "released",
    }
    _require(isinstance(entry, Mapping) and dict(entry) == expected_entry, "authorization entry drift")
    return digest


def validate_authorized_manifest(path: Path) -> str:
    payload = _secure_regular_bytes(
        path, label="authorized Q75 manifest", maximum=16 * 1024
    )
    expected = (
        "id\tfilename\tmd5\tsize\tstate\n"
        f"{EXPECTED_GDC_UUID}\t{EXPECTED_FILENAME}\t{EXPECTED_MD5}\t"
        f"{EXPECTED_SIZE_BYTES}\treleased\n"
    ).encode("utf-8")
    _require(payload == expected, "authorized manifest content mismatch")
    digest = _sha256_bytes(payload)
    _require(digest == EXPECTED_MANIFEST_SHA256, "authorized manifest SHA256 mismatch")
    return digest


def _git(repo: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        timeout=30,
    )
    _require(result.returncode == 0, f"git {' '.join(arguments)} failed")
    return result.stdout


def validate_repository_binding(
    repo: Path,
    *,
    critical_paths: Sequence[Path] = CRITICAL_RELATIVE_PATHS,
) -> dict[str, Any]:
    """Require HEAD-equal sources and only the protected BLCA worktree edit."""

    _require(repo.is_absolute(), "repository path must be absolute")
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").decode().strip()
    _require(len(head) == 40, "source HEAD is not a full commit hash")
    raw_status = _git(
        repo, "status", "--porcelain=v1", "-z", "--untracked-files=all"
    )
    status = [item.decode("utf-8") for item in raw_status.split(b"\0") if item]
    _require(status == [PROTECTED_DIRTY_STATUS], "Git state exceeds protected BLCA edit")
    hashes: dict[str, str] = {}
    for relative in critical_paths:
        _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe critical path")
        current = _secure_regular_bytes(
            repo / relative,
            label=f"critical source {relative}",
            maximum=4 * 1024 * 1024,
        )
        head_blob = _git(repo, "cat-file", "blob", f"HEAD:{relative.as_posix()}")
        _require(current == head_blob, f"critical source is not HEAD-equal: {relative}")
        hashes[relative.as_posix()] = _sha256_bytes(current)
    return {
        "source_commit": head,
        "allowed_dirty_status": status,
        "critical_file_sha256": hashes,
        "critical_files_head_equal": True,
    }


def _expected_gdc_tree() -> list[str]:
    parcel = f"{EXPECTED_FILENAME}.parcel"
    return [
        EXPECTED_GDC_UUID,
        f"{EXPECTED_GDC_UUID}/{EXPECTED_FILENAME}",
        f"{EXPECTED_GDC_UUID}/logs",
        f"{EXPECTED_GDC_UUID}/logs/{parcel}",
    ]


def _validate_exact_incoming_tree(incoming: Path, wsi: Path) -> list[str]:
    _require_no_symlink_components(incoming, include_leaf=True)
    _require(stat.S_ISDIR(incoming.lstat().st_mode), "Q75 incoming path must be a directory")
    uuid_directory = incoming / EXPECTED_GDC_UUID
    _require(wsi == uuid_directory / EXPECTED_FILENAME, "Q75 WSI path identity mismatch")
    _require_no_symlink_components(uuid_directory, include_leaf=True)
    _require(stat.S_ISDIR(uuid_directory.lstat().st_mode), "Q75 UUID path must be a directory")
    _require(
        tuple(sorted(item.name for item in incoming.iterdir())) == (EXPECTED_GDC_UUID,),
        "partial or unexpected Q75 incoming entry",
    )
    _require(
        tuple(sorted(item.name for item in uuid_directory.iterdir()))
        == (EXPECTED_FILENAME, "logs"),
        "partial or unexpected Q75 UUID entry",
    )
    _path_token(wsi, label="Q75 WSI")
    logs = uuid_directory / "logs"
    _require_no_symlink_components(logs, include_leaf=True)
    _require(stat.S_ISDIR(logs.lstat().st_mode), "Q75 logs path must be a directory")
    parcel_name = f"{EXPECTED_FILENAME}.parcel"
    _require(
        tuple(sorted(item.name for item in logs.iterdir())) == (parcel_name,),
        "partial or unexpected Q75 log entry",
    )
    _path_token(logs / parcel_name, label="Q75 GDC parcel")
    return _expected_gdc_tree()


def verify_exact_wsi(
    path: Path,
    *,
    incoming_directory: Path,
    expected_size: int = EXPECTED_SIZE_BYTES,
    expected_md5: str = EXPECTED_MD5,
) -> VerifiedWsi:
    """Open once with O_NOFOLLOW, hash twice, and retain that descriptor."""

    _validate_exact_incoming_tree(incoming_directory, path)
    _require(path.name == EXPECTED_FILENAME, "Q75 filename mismatch")
    _require(path.suffix.lower() == ".svs", "Q75 SVS suffix mismatch")
    descriptor, token = _open_no_follow(path, label="Q75 WSI")
    try:
        _require(token.size == expected_size, "Q75 WSI exact byte-size mismatch")
        md5, first_sha256 = _hash_descriptor(descriptor, include_md5=True)
        _, second_sha256 = _hash_descriptor(descriptor, include_md5=False)
        _require(FileToken.from_stat(os.fstat(descriptor)) == token, "Q75 WSI changed while hashing")
        _require(_path_token(path, label="Q75 WSI") == token, "Q75 WSI pathname changed")
        _require(md5 == expected_md5, "Q75 WSI MD5 mismatch")
        _require(first_sha256 == second_sha256, "Q75 independent SHA256 mismatch")
        _require(
            FileToken.from_stat(os.stat(f"/proc/self/fd/{descriptor}")) == token,
            "Q75 stable descriptor mismatch",
        )
        _validate_exact_incoming_tree(incoming_directory, path)
        _require(
            _path_token(path, label="Q75 WSI") == token,
            "Q75 WSI pathname changed during tree validation",
        )
        return VerifiedWsi(
            path=str(path),
            descriptor=descriptor,
            token=token,
            size_bytes=expected_size,
            md5=expected_md5,
            sha256=first_sha256,
        )
    except Exception:
        os.close(descriptor)
        raise


def reverify_held_wsi(
    path: Path, *, incoming_directory: Path, verified: VerifiedWsi
) -> None:
    _require(
        _path_token(path, label="Q75 WSI") == verified.token,
        "Q75 WSI pathname changed",
    )
    _require(
        FileToken.from_stat(os.fstat(verified.descriptor)) == verified.token,
        "held Q75 WSI identity changed",
    )
    md5, sha256 = _hash_descriptor(verified.descriptor, include_md5=True)
    _require(md5 == verified.md5, "held Q75 WSI MD5 changed")
    _require(sha256 == verified.sha256, "held Q75 WSI SHA256 changed")
    _require(
        FileToken.from_stat(os.fstat(verified.descriptor)) == verified.token,
        "held Q75 WSI changed during final hash",
    )
    _require(
        _path_token(path, label="Q75 WSI") == verified.token,
        "Q75 WSI pathname changed during final hash",
    )
    _validate_exact_incoming_tree(incoming_directory, path)
    _require(
        _path_token(path, label="Q75 WSI") == verified.token,
        "Q75 WSI pathname changed during tree revalidation",
    )


def open_verified_omic(path: Path) -> HeldRegularFile:
    descriptor, token = _open_no_follow(path, label="released BRCA Omic archive")
    try:
        _require(token.size <= 64 * 1024 * 1024, "Omic archive exceeds size limit")
        _, sha256 = _hash_descriptor(descriptor, include_md5=False)
        _require(sha256 == BRCA_RELEASE_ARCHIVE_SHA256, "BRCA Omic SHA256 mismatch")
        _require(FileToken.from_stat(os.fstat(descriptor)) == token, "Omic archive changed")
        _require(_path_token(path, label="released BRCA Omic archive") == token, "Omic pathname changed")
        return HeldRegularFile(path=path, descriptor=descriptor, token=token, sha256=sha256)
    except Exception:
        os.close(descriptor)
        raise


def reverify_held_omic(held: HeldRegularFile) -> None:
    _require(FileToken.from_stat(os.fstat(held.descriptor)) == held.token, "held Omic changed")
    _, sha256 = _hash_descriptor(held.descriptor, include_md5=False)
    _require(sha256 == held.sha256, "held Omic SHA256 changed")
    _require(
        FileToken.from_stat(os.fstat(held.descriptor)) == held.token,
        "held Omic changed during final hash",
    )
    _require(_path_token(held.path, label="released BRCA Omic archive") == held.token, "Omic path changed")


def _validate_omics(omics: BrcaPatientOmics) -> dict[str, Any]:
    _require(omics.case_id == EXPECTED_PATIENT_ID, "Q75 Omic patient mismatch")
    _require(omics.slide_id == EXPECTED_SLIDE_ID, "Q75 Omic full slide mismatch")
    _require(omics.source_row_index == EXPECTED_OMIC_SOURCE_INDEX, "Q75 Omic row mismatch")
    shapes: dict[str, list[int]] = {}
    for name, width in BRCA_EXPECTED_DIMS.items():
        tensor = getattr(omics, name)
        expected_shape = (1, 1, width)
        _require(tuple(tensor.shape) == expected_shape, f"Q75 {name} shape mismatch")
        _require(tensor.dtype is torch.float32, f"Q75 {name} must be float32")
        _require(tensor.device.type == "cpu", f"Q75 {name} must remain on CPU")
        _require(tensor.is_contiguous(), f"Q75 {name} must be contiguous")
        _require(bool(torch.isfinite(tensor).all().item()), f"Q75 {name} is nonfinite")
        shapes[name] = list(expected_shape)
    return {
        "patient_id": omics.case_id,
        "slide_id": omics.slide_id,
        "source_row_index": omics.source_row_index,
        "exact_case_and_full_slide_match": True,
        "shapes": shapes,
        "dtype": "float32",
        "device": "cpu",
        "all_finite": True,
    }


def load_and_validate_exact_omics(
    stable_archive_path: Path,
    *,
    loader: Callable[..., BrcaPatientOmics] = load_brca_patient_omics,
) -> dict[str, Any]:
    return _validate_omics(
        loader(
            stable_archive_path,
            case_id=EXPECTED_PATIENT_ID,
            slide_id=EXPECTED_SLIDE_ID,
            expected_archive_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
        )
    )


def _header_value(properties: Mapping[str, Any], key: str) -> float:
    try:
        value = float(properties[key])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise Q75HeaderGateError(f"missing or invalid OpenSlide header {key}") from exc
    _require(math.isfinite(value) and value > 0, f"OpenSlide {key} must be finite and positive")
    return value


def collect_header_only(slide: object) -> dict[str, Any]:
    """Read only MPP, level count, dimensions, and downsample attributes."""

    try:
        properties = slide.properties
        mpp_x = _header_value(properties, "openslide.mpp-x")
        mpp_y = _header_value(properties, "openslide.mpp-y")
        level_count = slide.level_count
        dimensions = tuple(tuple(item) for item in slide.level_dimensions)
        downsamples = tuple(slide.level_downsamples)
    except Q75HeaderGateError:
        raise
    except Exception as exc:
        raise Q75HeaderGateError("could not collect authorized OpenSlide headers") from exc
    _require(
        isinstance(level_count, int) and not isinstance(level_count, bool) and level_count > 0,
        "OpenSlide level_count must be a positive integer",
    )
    _require(len(dimensions) == level_count, "OpenSlide level_count/dimensions mismatch")
    _require(len(downsamples) == level_count, "OpenSlide level_count/downsamples mismatch")
    try:
        levels = validate_metadata_pyramid(
            mpp_x=mpp_x,
            mpp_y=mpp_y,
            level_dimensions=dimensions,
            level_downsamples=downsamples,
        )
    except WsiMetadataPolicyError as exc:
        raise Q75HeaderGateError(f"invalid OpenSlide pyramid header: {exc}") from exc
    return {
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "level_count": level_count,
        "level_dimensions": [list(level.dimensions) for level in levels],
        "level_downsamples": [level.downsample for level in levels],
        "native_levels": [
            {
                "level_index": level.level_index,
                "dimensions": list(level.dimensions),
                "downsample": level.downsample,
                "native_mpp_x": level.native_mpp_x,
                "native_mpp_y": level.native_mpp_y,
            }
            for level in levels
        ],
        "_validated_levels": levels,
    }


def assess_targets_without_approving_policy(
    levels: Sequence[NativeLevelMetadata],
) -> list[dict[str, Any]]:
    assessments: list[dict[str, Any]] = []
    tolerance = APPROVED_PER_AXIS_TOLERANCE_FRACTION
    for label, target in _TARGETS:
        closest = min(
            levels,
            key=lambda level: (
                max(abs(level.native_mpp_x - target), abs(level.native_mpp_y - target)),
                level.level_index,
            ),
        )
        error_x = abs(closest.native_mpp_x - target) / target
        error_y = abs(closest.native_mpp_y - target) / target
        within = error_x <= tolerance and error_y <= tolerance
        assessments.append(
            {
                "target_label": label,
                "target_mpp": target,
                "closest_native_level": closest.level_index,
                "closest_native_mpp_x": closest.native_mpp_x,
                "closest_native_mpp_y": closest.native_mpp_y,
                "relative_error_x": error_x,
                "relative_error_y": error_y,
                "comparison_tolerance_fraction": tolerance,
                "appears_natively_achievable": within,
                "controlled_resampling_appears_required": not within,
                "scale_policy_approved": False,
            }
        )
    return assessments


def _disk_snapshot(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "free_bytes": usage.free}


def _validate_storage_snapshot(value: object, *, label: str) -> dict[str, int]:
    _require(isinstance(value, Mapping), f"{label} must be a mapping")
    result: dict[str, int] = {}
    for key in ("total_bytes", "used_bytes", "free_bytes"):
        item = value.get(key)
        _require(
            isinstance(item, int) and not isinstance(item, bool) and item >= 0,
            f"{label}.{key} invalid",
        )
        result[key] = item
    _require(result["used_bytes"] + result["free_bytes"] <= result["total_bytes"], f"{label} inconsistent")
    return result


def validate_download_result(
    directory: Path,
    path: Path,
    *,
    current_source_commit: str,
    current_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Securely validate and bind the sole finalized download record."""

    _require_no_symlink_components(directory, include_leaf=True)
    _require(stat.S_ISDIR(directory.lstat().st_mode), "download result must be a directory")
    _require(path == directory / "download_result.yaml", "download-result path drift")
    _require(
        tuple(item.name for item in directory.iterdir()) == ("download_result.yaml",),
        "download-result directory must contain one file",
    )
    payload = _secure_regular_bytes(path, label="Q75 download result", maximum=2 * 1024 * 1024)
    digest = _sha256_bytes(payload)
    document = _yaml_mapping(payload, label="Q75 download result")
    expected_root = {
        "schema_version",
        "status",
        "cohort",
        "candidate",
        "patient_id",
        "slide_id",
        "gdc_file_uuid",
        "execution",
        "bindings",
        "gdc_client",
        "command",
        "storage",
        "download_tree",
        "wsi",
        "validations",
        "operations",
        "publication",
        "required_stop",
        "required_stop_reached",
        "next_gate",
    }
    _require(set(document) == expected_root, "download-result root schema drift")
    _require(document.get("status") == EXPECTED_DOWNLOAD_STATUS, "Q75 download did not verify")
    _require(document.get("cohort") == "TCGA-BRCA", "download cohort drift")
    _require(document.get("candidate") == "Q75", "download candidate drift")
    _require(document.get("patient_id") == EXPECTED_PATIENT_ID, "download patient drift")
    _require(document.get("slide_id") == EXPECTED_SLIDE_ID, "download slide drift")
    _require(document.get("gdc_file_uuid") == EXPECTED_GDC_UUID, "download UUID drift")
    execution = document.get("execution")
    _require(isinstance(execution, Mapping), "download execution binding missing")
    _require(execution.get("source_commit") == current_source_commit, "download/header commit mismatch")
    _require(execution.get("source_head_equal") is True, "download source was not HEAD-equal")
    _require(execution.get("source_files_head_equal") is True, "download critical files were not HEAD-equal")
    expected_download_source_hashes = {
        relative.as_posix(): current_source_hashes.get(relative.as_posix())
        for relative in DOWNLOAD_SOURCE_RELATIVE_PATHS
    }
    _require(
        all(_valid_sha256(value) for value in expected_download_source_hashes.values()),
        "current download-source hash binding is incomplete",
    )
    _require(
        execution.get("source_file_sha256") == expected_download_source_hashes,
        "download source-file hash binding drift",
    )
    bindings = document.get("bindings")
    _require(isinstance(bindings, Mapping), "download bindings missing")
    _require(bindings.get("authorization_config_sha256") == EXPECTED_AUTHORIZATION_SHA256, "download auth drift")
    _require(bindings.get("authorization_record_sha256") == EXPECTED_AUTHORIZATION_RECORD_SHA256, "download auth-record drift")
    _require(bindings.get("manifest_sha256") == EXPECTED_MANIFEST_SHA256, "download manifest drift")
    client = document.get("gdc_client")
    expected_client = {
        "path": str(DEFAULT_GDC_CLIENT),
        "version_output": EXPECTED_GDC_CLIENT_VERSION,
        "size_bytes": EXPECTED_GDC_CLIENT_SIZE_BYTES,
        "sha256": EXPECTED_GDC_CLIENT_SHA256,
        "regular_non_symlink": True,
        "executable": True,
    }
    _require(isinstance(client, Mapping), "download GDC client record missing")
    _require(dict(client) == expected_client, "download GDC client binding drift")
    command = document.get("command")
    _require(isinstance(command, Mapping), "download command record missing")
    argv = command.get("argv")
    expected_argv = [
        str(DEFAULT_GDC_CLIENT),
        "download",
        "-m",
        str(DEFAULT_AUTHORIZED_MANIFEST),
        "--dir",
        str(DEFAULT_INCOMING_DIRECTORY),
        "--no-related-files",
        "--no-annotations",
    ]
    _require(argv == expected_argv, "download argv is not the exact authorized command")
    _require(command.get("returncode") == 0, "GDC client did not exit successfully")
    _require(command.get("gdc_client_processes_started") == 1, "download client process count drift")
    _require(command.get("active_gdc_processes_before") == 0, "preexisting GDC process detected")
    _require(command.get("active_gdc_processes_after") == 0, "GDC process remained active")
    storage = document.get("storage")
    _require(isinstance(storage, Mapping), "download storage missing")
    before = _validate_storage_snapshot(storage.get("filesystem_before_download"), label="before download")
    after = _validate_storage_snapshot(storage.get("filesystem_after_download"), label="after download")
    tree = document.get("download_tree")
    _require(isinstance(tree, Mapping), "download tree missing")
    relative_entries = tree.get("relative_entries")
    _require(relative_entries == _expected_gdc_tree(), "download tree entries drift")
    _require(
        tree.get("incoming_directory") == str(DEFAULT_INCOMING_DIRECTORY),
        "download incoming-directory drift",
    )
    _require(
        tree.get("uuid_directory")
        == str(DEFAULT_INCOMING_DIRECTORY / EXPECTED_GDC_UUID),
        "download UUID-directory drift",
    )
    _require(tree.get("exact_entry_count_excluding_root") == 4, "download tree count drift")
    _require(tree.get("partial_or_unexpected_entries") == [], "download tree contains extras")
    _require(tree.get("completed_tree_exact") is True, "download tree was not exact")
    parcel = tree.get("parcel")
    expected_parcel_path = f"{EXPECTED_GDC_UUID}/logs/{EXPECTED_FILENAME}.parcel"
    _require(isinstance(parcel, Mapping), "download parcel record missing")
    _require(parcel.get("path") == expected_parcel_path, "download parcel path drift")
    _require(
        parcel.get("absolute_path")
        == str(DEFAULT_INCOMING_DIRECTORY / expected_parcel_path),
        "download parcel absolute path drift",
    )
    _require(isinstance(parcel.get("size_bytes"), int) and parcel.get("size_bytes") > 0, "download parcel size invalid")
    _require(_valid_sha256(parcel.get("sha256")), "download parcel SHA256 invalid")
    _require(parcel.get("regular_non_symlink") is True, "download parcel file type drift")
    wsi = document.get("wsi")
    _require(isinstance(wsi, Mapping), "download WSI record missing")
    _require(wsi.get("path") == str(DEFAULT_WSI), "download WSI path drift")
    _require(wsi.get("size_bytes") == EXPECTED_SIZE_BYTES, "download WSI size drift")
    _require(wsi.get("md5") == EXPECTED_MD5, "download WSI MD5 drift")
    _require(_valid_sha256(wsi.get("sha256")), "download WSI SHA256 invalid")
    _require(wsi.get("sha256_independent_recheck_matches") is True, "download SHA recheck missing")
    _require(wsi.get("regular_non_symlink") is True, "download WSI file type drift")
    _require(wsi.get("exact_svs_suffix") is True, "download WSI suffix drift")
    _require(wsi.get("gdc_uuid") == EXPECTED_GDC_UUID, "download WSI UUID binding drift")
    _require(wsi.get("filename") == EXPECTED_FILENAME, "download WSI filename binding drift")
    _require(wsi.get("expected_size_bytes") == EXPECTED_SIZE_BYTES, "download expected-size drift")
    _require(wsi.get("expected_md5") == EXPECTED_MD5, "download expected-MD5 drift")
    _require(wsi.get("exact_uuid_and_filename") is True, "download exact identity check missing")
    _require(wsi.get("exact_size_matches") is True, "download exact size check missing")
    _require(wsi.get("md5_matches") is True, "download MD5 check missing")
    operations = document.get("operations")
    _require(
        isinstance(operations, Mapping) and dict(operations) == EXPECTED_DOWNLOAD_OPERATIONS,
        "download operation counters drift",
    )
    publication = document.get("publication")
    _require(isinstance(publication, Mapping), "download publication missing")
    _require(publication.get("result_directory") == str(DEFAULT_DOWNLOAD_RESULT_DIRECTORY), "download result directory drift")
    _require(publication.get("result_file") == str(DEFAULT_DOWNLOAD_RESULT), "download result file drift")
    _require(publication.get("atomic_directory_rename_noreplace") is True, "download result was not atomic")
    _require(publication.get("sole_file") is True, "download result directory is not sole-file")
    _require(document.get("required_stop") == EXPECTED_DOWNLOAD_STOP, "download stop boundary drift")
    _require(document.get("required_stop_reached") is True, "download required stop missing")
    return {
        "path": str(path),
        "sha256": digest,
        "source_commit": execution["source_commit"],
        "wsi_sha256": wsi["sha256"],
        "filesystem_before_download": before,
        "filesystem_after_download": after,
        "gdc_created_tree": list(relative_entries),
        "parcel": dict(parcel),
    }


def _render_report(record: Mapping[str, Any]) -> str:
    wsi = record["wsi"]
    header = record["openslide_header"]
    lines = [
        "# BRCA Q75 exact-file and OpenSlide header-only result",
        "",
        f"Status: `{record['status']}`",
        "",
        "CPU-only metadata result. No pixel access or scale-policy approval occurred.",
        "",
        "## Identity",
        "",
        f"- Patient: `{EXPECTED_PATIENT_ID}`",
        f"- UUID: `{EXPECTED_GDC_UUID}`",
        f"- Slide: `{EXPECTED_FILENAME}`",
        f"- Local path: `{wsi['path']}`",
        f"- Size: `{wsi['size_bytes']}` bytes",
        f"- MD5: `{wsi['md5']}`",
        f"- SHA256: `{wsi['sha256']}`",
        "",
        "## OpenSlide header",
        "",
        f"- mpp-x: `{header['mpp_x']}`",
        f"- mpp-y: `{header['mpp_y']}`",
        f"- Level count: `{header['level_count']}`",
        "",
        "| Level | Dimensions | Downsample | Native MPP x | Native MPP y |",
        "|---:|---:|---:|---:|---:|",
    ]
    for level in header["native_levels"]:
        lines.append(
            f"| {level['level_index']} | {level['dimensions'][0]} × {level['dimensions'][1]} | "
            f"{level['downsample']} | {level['native_mpp_x']} | {level['native_mpp_y']} |"
        )
    lines.extend(["", "## Descriptive target comparison", ""])
    for item in record["target_assessment"]:
        appearance = (
            "appears natively achievable"
            if item["appears_natively_achievable"]
            else "appears to require controlled resampling"
        )
        lines.append(
            f"- `{item['target_mpp']} µm/px`: closest level {item['closest_native_level']}; {appearance}."
        )
    lines.extend(
        [
            "",
            "This comparison does not select or approve a Q75 scale policy.",
            "",
            "## Download and storage binding",
            "",
            f"- Download result SHA256: `{record['download_result']['sha256']}`",
            f"- Used bytes before download: `{record['storage']['filesystem_before_download']['used_bytes']}`",
            f"- Used bytes after download: `{record['storage']['filesystem_after_download']['used_bytes']}`",
            f"- Used bytes before header gate: `{record['storage']['filesystem_before_header_gate']['used_bytes']}`",
            f"- Used bytes after header gate: `{record['storage']['filesystem_after_header_gate']['used_bytes']}`",
            "- Exact GDC-created tree:",
            *[f"  - `{item}`" for item in record["download_result"]["gdc_created_tree"]],
            "",
            "## Source binding",
            "",
            f"- Source commit: `{record['source']['source_commit']}`",
            *[
                f"- `{path}`: `{digest}`"
                for path, digest in record["source"]["critical_file_sha256"].items()
            ],
            "",
            "## Validation and stop",
            "",
            *[f"- {name}: `{value}`" for name, value in record["validations"].items()],
            f"- Atomic result directory: `{record['publication']['output_directory']}`",
            f"- Required stop reached: `{record['required_stop_reached']}`",
            "",
            "Further Q75 work requires separate user review and authorization.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_new_file(path: Path, payload: bytes) -> None:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        written = 0
        while written < len(payload):
            written += os.write(descriptor, payload[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_result_directory(
    directory: Path,
    *,
    expected_result: bytes,
    expected_report: bytes,
) -> dict[str, str]:
    _require_no_symlink_components(directory, include_leaf=True)
    _require(stat.S_ISDIR(directory.lstat().st_mode), "result output must be a directory")
    _require(
        {item.name for item in directory.iterdir()} == {RESULT_BASENAME, REPORT_BASENAME},
        "result directory content drift",
    )
    result = _secure_regular_bytes(
        directory / RESULT_BASENAME,
        label="Q75 result YAML",
        maximum=4 * 1024 * 1024,
    )
    report = _secure_regular_bytes(
        directory / REPORT_BASENAME,
        label="Q75 result report",
        maximum=4 * 1024 * 1024,
    )
    _require(result == expected_result, "published Q75 result mismatch")
    _require(report == expected_report, "published Q75 report mismatch")
    return {RESULT_BASENAME: _sha256_bytes(result), REPORT_BASENAME: _sha256_bytes(report)}


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    _require(function is not None, "renameat2 is required for no-replace publication")
    function.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    function.restype = ctypes.c_int
    result = function(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise Q75HeaderGateError(f"append-only output already exists: {destination}")
        raise Q75HeaderGateError(f"atomic no-replace publication failed: errno={error}")


def _remove_owned_staging(directory: Path) -> None:
    if not directory.exists() and not directory.is_symlink():
        return
    _require(not directory.is_symlink() and directory.is_dir(), "unsafe staging directory")
    for name in (RESULT_BASENAME, REPORT_BASENAME):
        path = directory / name
        if path.exists() or path.is_symlink():
            _require(not path.is_symlink() and path.is_file(), "unsafe staged output")
            path.unlink()
    _require(not tuple(directory.iterdir()), "unexpected owned-staging entry")
    directory.rmdir()


def publish_result_directory(
    output_directory: Path,
    *,
    result_payload: bytes,
    report_payload: bytes,
) -> dict[str, str]:
    """Atomically rename one validated directory with RENAME_NOREPLACE."""

    _require(output_directory.is_absolute(), "result directory must be absolute")
    _require_no_symlink_components(output_directory.parent, include_leaf=True)
    _require(
        not output_directory.exists() and not output_directory.is_symlink(),
        f"append-only output already exists: {output_directory}",
    )
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.staging.",
            dir=output_directory.parent,
        )
    )
    published = False
    try:
        _write_new_file(staging / RESULT_BASENAME, result_payload)
        _write_new_file(staging / REPORT_BASENAME, report_payload)
        _validate_result_directory(
            staging,
            expected_result=result_payload,
            expected_report=report_payload,
        )
        _fsync_directory(staging)
        _rename_directory_no_replace(staging, output_directory)
        published = True
        _fsync_directory(output_directory.parent)
        return _validate_result_directory(
            output_directory,
            expected_result=result_payload,
            expected_report=report_payload,
        )
    finally:
        if not published:
            _remove_owned_staging(staging)


def run_header_gate(
    *,
    slide_factory: Callable[[str], object],
    paths: GatePaths = GatePaths(),
    omic_loader: Callable[..., BrcaPatientOmics] = load_brca_patient_omics,
    repository_validator: Callable[[Path], dict[str, Any]] = validate_repository_binding,
    download_validator: Callable[..., dict[str, Any]] = validate_download_result,
    wsi_verifier: Callable[..., VerifiedWsi] = verify_exact_wsi,
    omic_opener: Callable[[Path], HeldRegularFile] = open_verified_omic,
    wsi_reverifier: Callable[..., None] = reverify_held_wsi,
    omic_reverifier: Callable[[HeldRegularFile], None] = reverify_held_omic,
    publish: bool = True,
) -> dict[str, Any]:
    """Execute exact identity/header checks and stop before pixel access."""

    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    _require(
        not paths.output_directory.exists() and not paths.output_directory.is_symlink(),
        f"append-only output already exists: {paths.output_directory}",
    )
    source_before = repository_validator(paths.repo_root)
    auth_digest = validate_authorization(
        paths.authorization,
        expected_sha256=EXPECTED_AUTHORIZATION_SHA256,
    )
    auth_record_digest = validate_authorization_record(
        paths.authorization_record,
        expected_sha256=EXPECTED_AUTHORIZATION_RECORD_SHA256,
        authorization_sha256=auth_digest,
    )
    manifest_digest = validate_authorized_manifest(paths.manifest)
    download = download_validator(
        paths.download_result_directory,
        paths.download_result,
        current_source_commit=source_before["source_commit"],
        current_source_hashes=source_before["critical_file_sha256"],
    )
    disk_before_header = _disk_snapshot(paths.incoming_directory.parent)

    verified: VerifiedWsi | None = None
    held_omic: HeldRegularFile | None = None
    try:
        verified = wsi_verifier(paths.wsi, incoming_directory=paths.incoming_directory)
        _require(verified.sha256 == download["wsi_sha256"], "WSI/download SHA256 mismatch")
        held_omic = omic_opener(paths.omic_archive)
        omic_first = load_and_validate_exact_omics(
            held_omic.stable_path,
            loader=omic_loader,
        )
        _require(
            _path_token(paths.wsi, label="Q75 WSI") == verified.token,
            "Q75 WSI pathname changed before OpenSlide",
        )
        slide = None
        try:
            slide = slide_factory(str(verified.stable_path))
            close = getattr(slide, "close", None)
            _require(callable(close), "OpenSlide object lacks close")
            header = collect_header_only(slide)
        except Q75HeaderGateError:
            raise
        except Exception as exc:
            raise Q75HeaderGateError("OpenSlide could not open held Q75 descriptor") from exc
        finally:
            if slide is not None:
                close = getattr(slide, "close", None)
                if callable(close):
                    close()
        levels = header.pop("_validated_levels")
        wsi_reverifier(
            paths.wsi,
            incoming_directory=paths.incoming_directory,
            verified=verified,
        )
        omic_second = load_and_validate_exact_omics(
            held_omic.stable_path,
            loader=omic_loader,
        )
        _require(omic_second == omic_first, "Q75 Omic rematch changed")
        omic_reverifier(held_omic)
        live_tree = _validate_exact_incoming_tree(paths.incoming_directory, paths.wsi)
        _require(download["gdc_created_tree"] == live_tree, "download/live tree mismatch")
    finally:
        if held_omic is not None:
            os.close(held_omic.descriptor)
        if verified is not None:
            os.close(verified.descriptor)

    disk_after_header = _disk_snapshot(paths.incoming_directory.parent)
    source_before_publication = repository_validator(paths.repo_root)
    _require(source_before_publication == source_before, "source binding changed during gate")
    finished = datetime.now(timezone.utc)
    record: dict[str, Any] = {
        "schema_version": 2,
        "status": RESULT_STATUS,
        "cohort": "TCGA-BRCA",
        "candidate": "Q75",
        "patient_id": EXPECTED_PATIENT_ID,
        "slide_id": EXPECTED_SLIDE_ID,
        "gdc_file_uuid": EXPECTED_GDC_UUID,
        "execution": {
            "mode": "CPU_ONLY_HEADER_METADATA",
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "total_seconds": time.perf_counter() - start_clock,
            "cuda_used": False,
            "gpu_work_performed": False,
        },
        "source": source_before,
        "authorization": {
            "config_sha256": auth_digest,
            "record_sha256": auth_record_digest,
            "manifest_sha256": manifest_digest,
        },
        "download_result": download,
        "wsi": {
            "path": verified.path,
            "size_bytes": verified.size_bytes,
            "md5": verified.md5,
            "sha256": verified.sha256,
            "sha256_status": "ESTABLISHED_AND_BOUND_TO_DOWNLOAD_RESULT",
            "held_o_nofollow_descriptor_used": True,
            "openslide_stable_proc_fd_path_used": True,
            "same_descriptor_final_hash_recheck": True,
            "regular_non_symlink": True,
            "exact_svs_suffix": True,
        },
        "omic": {
            "archive_path": str(paths.omic_archive),
            "archive_sha256": held_omic.sha256,
            "held_o_nofollow_descriptor_used": True,
            **omic_second,
            "reverified_before_recording": True,
        },
        "openslide_header": header,
        "target_assessment": assess_targets_without_approving_policy(levels),
        "storage": {
            "filesystem_before_download": download["filesystem_before_download"],
            "filesystem_after_download": download["filesystem_after_download"],
            "filesystem_before_header_gate": disk_before_header,
            "filesystem_after_header_gate": disk_after_header,
            "raw_wsi_retained": True,
        },
        "validations": {
            "source_commit_and_critical_files_head_equal": "PASS",
            "only_protected_blca_worktree_edit": "PASS",
            "secure_authorization_record_manifest_reads": "PASS",
            "download_result_binding": "PASS",
            "same_held_wsi_descriptor_before_and_after_header": "PASS",
            "exact_uuid_filename_size_md5_sha256": "PASS",
            "partial_or_incomplete_download_check": "PASS",
            "regular_non_symlink_svs_and_parcel": "PASS",
            "held_omic_archive_and_exact_row_rematch": "PASS",
            "finite_positive_consistent_openslide_header": "PASS",
            "zero_pixel_or_region_reads": "PASS",
        },
        "operations": {
            "openslide_header_opens": 1,
            "pixel_or_region_reads": 0,
            "tissue_masks": 0,
            "coordinates": 0,
            "patches": 0,
            "resnet50_inference": 0,
            "healnet_execution": 0,
            "q75_features": 0,
            "raw_wsi_deletions": 0,
            "google_drive_operations": 0,
            "full_cohort_operations": 0,
            "q25_q50_modifications": 0,
            "blca_modifications": 0,
            "training_runs": 0,
        },
        "scale_policy_approved": False,
        "publication": {
            "atomic_directory_rename_noreplace": True,
            "output_directory": str(paths.output_directory),
            "files": [RESULT_BASENAME, REPORT_BASENAME],
            "other_files_modified_by_runner": [],
        },
        "required_stop": REQUIRED_STOP,
        "required_stop_reached": True,
        "next_gate": "USER_REVIEW_REQUIRED_BEFORE_ANY_Q75_PIXEL_ACCESS_OR_SCALE_POLICY",
    }
    if publish:
        result_payload = yaml.safe_dump(record, sort_keys=False).encode("utf-8")
        report_payload = _render_report(record).encode("utf-8")
        publish_result_directory(
            paths.output_directory,
            result_payload=result_payload,
            report_payload=report_payload,
        )
    return record


__all__ = [
    "CRITICAL_RELATIVE_PATHS",
    "DEFAULT_AUTHORIZATION",
    "DEFAULT_AUTHORIZATION_RECORD",
    "DEFAULT_AUTHORIZED_MANIFEST",
    "DEFAULT_DOWNLOAD_RESULT",
    "DEFAULT_DOWNLOAD_RESULT_DIRECTORY",
    "DEFAULT_INCOMING_DIRECTORY",
    "DEFAULT_OMIC_ARCHIVE",
    "DEFAULT_OUTPUT_DIRECTORY",
    "DEFAULT_WSI",
    "EXPECTED_AUTHORIZATION_RECORD_SHA256",
    "EXPECTED_AUTHORIZATION_SHA256",
    "EXPECTED_FILENAME",
    "EXPECTED_GDC_UUID",
    "EXPECTED_MANIFEST_SHA256",
    "EXPECTED_MD5",
    "EXPECTED_PATIENT_ID",
    "EXPECTED_SIZE_BYTES",
    "EXPECTED_SLIDE_ID",
    "FileToken",
    "GatePaths",
    "HeldRegularFile",
    "Q75HeaderGateError",
    "REPORT_BASENAME",
    "RESULT_BASENAME",
    "VerifiedWsi",
    "assess_targets_without_approving_policy",
    "collect_header_only",
    "load_and_validate_exact_omics",
    "open_verified_omic",
    "publish_result_directory",
    "reverify_held_omic",
    "reverify_held_wsi",
    "run_header_gate",
    "validate_authorization",
    "validate_authorization_record",
    "validate_authorized_manifest",
    "validate_download_result",
    "validate_repository_binding",
    "verify_exact_wsi",
]
