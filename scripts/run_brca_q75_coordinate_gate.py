#!/usr/bin/env python3
"""Execute the authorized BRCA Q75 one-mask-read CPU coordinate gate.

This entry point is deliberately bound to one exact WSI, one held
``O_NOFOLLOW`` descriptor, one level-2 ``read_region`` call, and one external
append-only coordinate directory.  It imports no model or training stack and
cannot read level-0/level-1 patch pixels.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import stat
import subprocess
import sys
import time
from typing import Callable, Final

import cv2
import h5py
import numpy as np
import openslide
import yaml

_SCRIPT_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_REPO_ROOT))

from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
    CoordinateArtifactSetRecord,
    CoordinateBranchMetadata,
    publish_brca_coordinate_artifacts,
)
from multiscale_feature_pilot.src.brca_omic import (
    BRCA_EXPECTED_DIMS,
    BRCA_RELEASE_ARCHIVE_SHA256,
    BrcaPatientOmics,
    load_brca_patient_omics,
)
from multiscale_feature_pilot.src.brca_q75_coordinate_authorization import (
    APPROVAL_STATEMENT,
    APPROVAL_STATEMENT_SHA256,
    AUTHORIZATION_RELATIVE_PATH,
    Q75CoordinateAuthorizationError,
    validate_q75_coordinate_execution_authorization,
)
from multiscale_feature_pilot.src.brca_q25_coordinates import (
    A_H,
    A_T,
    CLOSE,
    MAX_N_HOLES,
    M_THRESH,
    REF_PATCH_SIZE,
    S_THRESH,
    USE_OTSU,
    generate_level_0_lattice_coordinates,
    segment_tissue_contours,
)
from multiscale_feature_pilot.src.brca_q75_coordinate_policy import (
    CLAM_COMMIT,
    EXPECTED_CNV_SHAPE,
    EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256,
    EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX,
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_HEADER_GATE_SOURCE_COMMIT,
    EXPECTED_HEADER_REPORT_SHA256,
    EXPECTED_HEADER_RESULT_COMMIT,
    EXPECTED_HEADER_RESULT_SHA256,
    EXPECTED_KNOWN_ISSUES_SHA256,
    EXPECTED_LEVEL_DIMENSIONS,
    EXPECTED_LEVEL_DOWNSAMPLES,
    EXPECTED_MD5,
    EXPECTED_MPP_X,
    EXPECTED_MPP_Y,
    EXPECTED_MUTATION_SHAPE,
    EXPECTED_PATIENT_ID,
    EXPECTED_Q25_COORDINATE_POLICY_SHA256,
    EXPECTED_Q25_COORDINATE_SOURCE_SHA256,
    EXPECTED_Q50_COORDINATE_POLICY_SHA256,
    EXPECTED_Q50_COORDINATE_SOURCE_SHA256,
    EXPECTED_RNA_SHAPE,
    EXPECTED_SCALE_APPROVAL_COMMIT,
    EXPECTED_SCALE_CONFIG_SHA256,
    EXPECTED_SCALE_PROVENANCE_SHA256,
    EXPECTED_SCALE_REPORT_SHA256,
    EXPECTED_SCALE_SOURCE_SHA256,
    EXPECTED_SHA256,
    EXPECTED_SIZE_BYTES,
    EXPECTED_SLIDE_ID,
    EXPECTED_USER_STATEMENT,
    EXPECTED_USER_STATEMENT_SHA256,
    GRID_ANCHOR,
    MASK_AREA_THRESHOLD,
    MASK_CLOSE,
    MASK_HOLE_AREA_THRESHOLD,
    MASK_LEVEL,
    MASK_MAX_HOLES,
    MASK_MTHRESH,
    MASK_REFERENCE_PATCH_SIZE,
    MASK_STHRESH,
    MASK_USE_OTSU,
    POLICY_STATUS,
    Q75CoordinateEvidence,
    Q75CoordinatePolicyPlan,
    SCALE_2X_LEVEL_0_FOOTPRINT,
    SCALE_2X_LEVEL_0_STEP,
    SCALE_4X_LEVEL_0_FOOTPRINT,
    SCALE_4X_LEVEL_0_STEP,
    review_q75_coordinate_policy,
)


SCHEMA: Final = "BRCA_Q75_COORDINATE_GATE_RESULT_V1"
EXPECTED_FILENAME: Final = EXPECTED_SLIDE_ID
REPO_ROOT: Final = _SCRIPT_REPO_ROOT
OFFICIAL_REPO: Final = Path("/teamspace/studios/this_studio/healnet")
INCOMING_DIR: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q75.incoming"
)
WSI_PATH: Final = (
    INCOMING_DIR / EXPECTED_GDC_FILE_UUID / EXPECTED_FILENAME
)
OUTPUT_DIR: Final = Path(
    "/teamspace/studios/this_studio/brca_pilot_data/Q75.coordinates"
)
OMIC_PATH: Final = Path(
    "/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/"
    "data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
OMIC_SIZE_BYTES: Final = 4_081_277
OFFICIAL_HEAD: Final = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
FROZEN_TAG: Final = "blca-one-patient-pilot-v1"
FROZEN_COMMIT: Final = "df7cf2bda783ab6cc09e95d6a1fa0914da05a433"
POLICY_COMMIT: Final = "1582d5b1d5eb5fac7d44e5d0e5d4fb2feebf87f9"

AUTH_RELATIVE_PATH: Final = Path(AUTHORIZATION_RELATIVE_PATH)
AUTH_SHA256: Final = (
    "4510cf2849edf3b0478030453b77faa1e0348f245b7e6703232d661c062f4539"
)
APPROVAL_RELATIVE_PATH: Final = Path(
    "multiscale_feature_pilot/provenance/"
    "brca_q75_coordinate_execution_approval.yaml"
)

BOUND_FILES: Final = (
    (
        Path("multiscale_feature_pilot/src/brca_q75_coordinate_authorization.py"),
        "794b759df886eaefdad017b468f381081a1328111064034a996782d6361e458f",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_omic.py"),
        "5061039913cb0dd4e8e30958c8b4b76f088396f7841255dd94234c3b462d5a5d",
    ),
    (
        Path("multiscale_feature_pilot/provenance/brca_q75_header_metadata_result/result.yaml"),
        "08a7ed3e67ddf17513ee2dbda2adfd2398333787aaa75fe9eacf911f3c1a3898",
    ),
    (
        Path("multiscale_feature_pilot/provenance/brca_q75_header_metadata_result/report.md"),
        "9ed50ecc8464109e0a8ca121082462f8765b1c70663499180cc644cfe604d985",
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
        Path("multiscale_feature_pilot/provenance/brca_q75_scale_approval.yaml"),
        "aae6547c7c23cfdad51f62e3587b33c7abe3c5fb7d6fbdcd15ffeed3737fdd8e",
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
        Path("multiscale_feature_pilot/provenance/brca_q75_coordinate_policy_review.yaml"),
        "3edd1a2000eb808c140972af4b4b9bf3b6e25d3602b4a2076b99c5fcc0046197",
    ),
    (
        Path("reports/brca_q75_coordinate_policy_review.md"),
        "e4df78300e6e6607a9301f50858ab27152390e05485707f66867f2d95838dbeb",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_q25_coordinates.py"),
        "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82",
    ),
    (
        Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
        "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf",
    ),
    (
        Path("shared/provenance/known_issues.md"),
        "8dff689f8181f7e08215595252042185542d9970c5885693b9afdaa7aa32c3c4",
    ),
)
CRITICAL_TRACKED_PATHS: Final = (
    Path("scripts/run_brca_q75_coordinate_gate.py"),
    AUTH_RELATIVE_PATH,
    APPROVAL_RELATIVE_PATH,
    *(item[0] for item in BOUND_FILES),
)
MASK_LOCATION: Final = (0, 0)
MASK_SIZE: Final = EXPECTED_LEVEL_DIMENSIONS[MASK_LEVEL]
COORDINATE_POLICY_SHA256: Final = dict(BOUND_FILES)[
    Path("multiscale_feature_pilot/config/brca_q75_coordinate_policy.yaml")
]
EXPECTED_AUTHORIZATION_STATEMENT: Final = APPROVAL_STATEMENT
EXPECTED_AUTHORIZATION_STATEMENT_SHA256: Final = APPROVAL_STATEMENT_SHA256
_CHUNK_SIZE: Final = 8 * 1024 * 1024
_MINIMUM_AVAILABLE_BYTES: Final = 1_000_000_000
_MAXIMUM_BOUND_SOURCE_BYTES: Final = 64 * 1024 * 1024
_GDC_PARCEL_SIZE_BYTES: Final = 82_071
_GDC_PARCEL_SHA256: Final = (
    "680777fd62dee07311118ac84560e7e64d35a80e8317c95adeb5777fb3b49bbe"
)


class Q75CoordinateGateError(RuntimeError):
    """Fail-closed Q75 coordinate-gate violation."""


@dataclass(frozen=True)
class GatePaths:
    repo_root: Path = REPO_ROOT
    official_repo: Path = OFFICIAL_REPO
    incoming: Path = INCOMING_DIR
    wsi: Path = WSI_PATH
    omic: Path = OMIC_PATH
    output: Path = OUTPUT_DIR
    auth: Path = REPO_ROOT / AUTH_RELATIVE_PATH


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
class HeldWsi:
    path: Path
    descriptor: int
    token: FileToken
    md5: str
    sha256: str

    @property
    def stable_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.descriptor}")


@dataclass(frozen=True)
class HeldOmic:
    path: Path
    descriptor: int
    token: FileToken
    sha256: str

    @property
    def stable_path(self) -> Path:
        return Path(f"/proc/self/fd/{self.descriptor}")


@dataclass(frozen=True)
class Q75CoordinateBags:
    scale_2x: np.ndarray
    scale_4x: np.ndarray
    contour_count: int
    retained_hole_count: int
    mask_downsample_xy: tuple[float, float]
    policy_status: str = POLICY_STATUS


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q75CoordinateGateError(message)


def _git(repo: Path, *arguments: str, binary: bool = False):
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        capture_output=True,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _require_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    _require(path.is_absolute(), f"path must be absolute: {path}")
    stop = len(path.parts) if include_leaf else len(path.parts) - 1
    current = Path(path.parts[0])
    for part in path.parts[1:stop]:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError as exc:
            raise Q75CoordinateGateError(
                f"missing path component: {current}"
            ) from exc
        _require(
            not stat.S_ISLNK(metadata.st_mode),
            f"symlink component prohibited: {current}",
        )


def _path_token(path: Path, *, label: str) -> FileToken:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Q75CoordinateGateError(f"missing {label}: {path}") from exc
    _require(not stat.S_ISLNK(metadata.st_mode), f"{label} must not be a symlink")
    _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
    return FileToken.from_stat(metadata)


def _open_no_follow(path: Path, *, label: str) -> tuple[int, FileToken]:
    _require_no_symlink_components(path, include_leaf=False)
    _require(hasattr(os, "O_NOFOLLOW"), "platform lacks O_NOFOLLOW")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Q75CoordinateGateError(f"cannot securely open {label}: {path}") from exc
    try:
        metadata = os.fstat(descriptor)
        _require(stat.S_ISREG(metadata.st_mode), f"{label} must be a regular file")
        token = FileToken.from_stat(metadata)
        _require(token == _path_token(path, label=label), f"{label} path/fd mismatch")
        return descriptor, token
    except Exception:
        os.close(descriptor)
        raise


def _secure_regular_bytes(path: Path, *, label: str) -> bytes:
    descriptor, token = _open_no_follow(path, label=label)
    try:
        _require(
            token.size <= _MAXIMUM_BOUND_SOURCE_BYTES,
            f"{label} exceeds secure source size limit",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            _require(total <= token.size, f"{label} grew during read")
        payload = b"".join(chunks)
        _require(len(payload) == token.size, f"{label} changed size during read")
        _require(
            FileToken.from_stat(os.fstat(descriptor)) == token,
            f"{label} changed during read",
        )
        _require(_path_token(path, label=label) == token, f"{label} path changed")
        return payload
    finally:
        os.close(descriptor)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(
        _secure_regular_bytes(path, label=f"SHA256 source {path}")
    ).hexdigest()


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


def _validate_exact_incoming_tree(incoming: Path, wsi: Path) -> dict[str, object]:
    _require(incoming == INCOMING_DIR, "Q75 incoming path drift")
    uuid_directory = incoming / EXPECTED_GDC_FILE_UUID
    _require(wsi == uuid_directory / EXPECTED_FILENAME, "Q75 WSI path drift")
    _require_no_symlink_components(uuid_directory, include_leaf=True)
    _require(
        stat.S_ISDIR(uuid_directory.lstat().st_mode),
        "Q75 UUID path must be a directory",
    )
    _require(
        tuple(sorted(item.name for item in incoming.iterdir()))
        == (EXPECTED_GDC_FILE_UUID,),
        "partial or unexpected Q75 incoming entry",
    )
    _require(
        tuple(sorted(item.name for item in uuid_directory.iterdir()))
        == (EXPECTED_FILENAME, "logs"),
        "partial or unexpected Q75 UUID entry",
    )
    logs = uuid_directory / "logs"
    _require_no_symlink_components(logs, include_leaf=True)
    _require(stat.S_ISDIR(logs.lstat().st_mode), "Q75 logs path must be a directory")
    parcel = logs / f"{EXPECTED_FILENAME}.parcel"
    _require(
        tuple(sorted(item.name for item in logs.iterdir())) == (parcel.name,),
        "partial or unexpected Q75 log entry",
    )
    parcel_token = _path_token(parcel, label="Q75 GDC parcel")
    _require(
        parcel_token.size == _GDC_PARCEL_SIZE_BYTES,
        "Q75 GDC parcel size drift",
    )
    _require(_sha256_file(parcel) == _GDC_PARCEL_SHA256, "Q75 GDC parcel SHA256 drift")
    return {
        "incoming_entries": [EXPECTED_GDC_FILE_UUID],
        "uuid_entries": [EXPECTED_FILENAME, "logs"],
        "log_entries": [parcel.name],
    }


def _open_verified_wsi(path: Path, *, incoming: Path) -> HeldWsi:
    """Hold one no-follow fd after two independent pre-open hash passes."""

    _validate_exact_incoming_tree(incoming, path)
    _require(path.name == EXPECTED_FILENAME, "Q75 filename drift")
    _require(path.suffix.lower() == ".svs", "Q75 SVS suffix drift")
    descriptor, token = _open_no_follow(path, label="Q75 WSI")
    try:
        _require(token.size == EXPECTED_SIZE_BYTES, "Q75 exact byte-size drift")
        md5, first_sha256 = _hash_descriptor(descriptor, include_md5=True)
        _, second_sha256 = _hash_descriptor(descriptor, include_md5=False)
        _require(
            FileToken.from_stat(os.fstat(descriptor)) == token,
            "Q75 WSI changed during prehash",
        )
        _require(_path_token(path, label="Q75 WSI") == token, "Q75 path changed")
        _require(md5 == EXPECTED_MD5, "Q75 MD5 drift")
        _require(first_sha256 == EXPECTED_SHA256, "Q75 SHA256 drift")
        _require(first_sha256 == second_sha256, "Q75 independent SHA256 mismatch")
        _require(
            FileToken.from_stat(os.stat(f"/proc/self/fd/{descriptor}")) == token,
            "Q75 stable descriptor mismatch",
        )
        _validate_exact_incoming_tree(incoming, path)
        _require(
            _path_token(path, label="Q75 WSI") == token,
            "Q75 WSI path changed during tree validation",
        )
        return HeldWsi(
            path=path,
            descriptor=descriptor,
            token=token,
            md5=md5,
            sha256=first_sha256,
        )
    except Exception:
        os.close(descriptor)
        raise


def _reverify_held_wsi(held: HeldWsi, *, incoming: Path) -> None:
    _require(
        FileToken.from_stat(os.fstat(held.descriptor)) == held.token,
        "held Q75 WSI identity changed",
    )
    _require(_path_token(held.path, label="Q75 WSI") == held.token, "Q75 path changed")
    md5, sha256 = _hash_descriptor(held.descriptor, include_md5=True)
    _require(md5 == held.md5 == EXPECTED_MD5, "held Q75 WSI MD5 changed")
    _require(sha256 == held.sha256 == EXPECTED_SHA256, "held Q75 WSI SHA256 changed")
    _require(
        FileToken.from_stat(os.fstat(held.descriptor)) == held.token,
        "held Q75 WSI changed during final hash",
    )
    _require(_path_token(held.path, label="Q75 WSI") == held.token, "Q75 path changed")
    _validate_exact_incoming_tree(incoming, held.path)
    _require(
        _path_token(held.path, label="Q75 WSI") == held.token,
        "Q75 WSI path changed during final tree validation",
    )


def _open_verified_omic(path: Path) -> HeldOmic:
    descriptor, token = _open_no_follow(path, label="BRCA Omic archive")
    try:
        _require(token.size == OMIC_SIZE_BYTES, "BRCA Omic archive size drift")
        _, sha256 = _hash_descriptor(descriptor, include_md5=False)
        _require(
            sha256 == BRCA_RELEASE_ARCHIVE_SHA256,
            "BRCA Omic archive SHA256 drift",
        )
        _require(
            FileToken.from_stat(os.fstat(descriptor)) == token,
            "BRCA Omic archive changed during hash",
        )
        _require(
            _path_token(path, label="BRCA Omic archive") == token,
            "BRCA Omic archive path changed",
        )
        return HeldOmic(
            path=path,
            descriptor=descriptor,
            token=token,
            sha256=sha256,
        )
    except Exception:
        os.close(descriptor)
        raise


def _reverify_held_omic(held: HeldOmic) -> None:
    _require(
        FileToken.from_stat(os.fstat(held.descriptor)) == held.token,
        "held BRCA Omic archive identity changed",
    )
    _, sha256 = _hash_descriptor(held.descriptor, include_md5=False)
    _require(sha256 == held.sha256, "held BRCA Omic archive SHA256 changed")
    _require(
        FileToken.from_stat(os.fstat(held.descriptor)) == held.token,
        "held BRCA Omic archive changed during final hash",
    )
    _require(
        _path_token(held.path, label="BRCA Omic archive") == held.token,
        "BRCA Omic archive path changed",
    )


def _load_and_validate_exact_omics(
    stable_archive_path: Path,
    *,
    loader: Callable[..., BrcaPatientOmics] = load_brca_patient_omics,
) -> dict[str, object]:
    omics = loader(
        stable_archive_path,
        case_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        expected_archive_sha256=None,
    )
    _require(omics.case_id == EXPECTED_PATIENT_ID, "Q75 Omic patient drift")
    _require(omics.slide_id == EXPECTED_SLIDE_ID, "Q75 Omic full-slide drift")
    _require(
        omics.source_row_index == EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX,
        "Q75 Omic source-row drift",
    )
    modality_records: dict[str, object] = {}
    for name, expected_width in BRCA_EXPECTED_DIMS.items():
        tensor = getattr(omics, name)
        _require(tuple(tensor.shape) == (1, 1, expected_width), f"Q75 {name} shape drift")
        _require(str(tensor.dtype) == "torch.float32", f"Q75 {name} dtype drift")
        _require(tensor.device.type == "cpu", f"Q75 {name} must remain CPU-only")
        _require(bool(tensor.is_contiguous()), f"Q75 {name} contiguity drift")
        _require(bool(tensor.isfinite().all().item()), f"Q75 {name} contains nonfinite values")
        payload = tensor.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C")
        modality_records[name] = {
            "shape": list(tensor.shape),
            "dtype": "float32",
            "all_finite": True,
            "content_sha256": hashlib.sha256(payload).hexdigest(),
        }
    return {
        "archive_path": str(OMIC_PATH),
        "archive_size_bytes": OMIC_SIZE_BYTES,
        "archive_sha256": BRCA_RELEASE_ARCHIVE_SHA256,
        "source_row_index": omics.source_row_index,
        "case_id": omics.case_id,
        "slide_id": omics.slide_id,
        "exact_case_and_full_slide_match": True,
        "modalities": modality_records,
    }


def _repository_snapshot(paths: GatePaths) -> dict[str, object]:
    official_head = _git(paths.official_repo, "rev-parse", "HEAD")
    official_status = _git(paths.official_repo, "status", "--porcelain=v1", "-z")
    _require(official_head == OFFICIAL_HEAD, "official HEALNet HEAD drift")
    _require(official_status == "", "official HEALNet worktree is modified")
    frozen_commit = _git(paths.repo_root, "rev-parse", f"{FROZEN_TAG}^{{commit}}")
    _require(frozen_commit == FROZEN_COMMIT, "frozen BLCA tag drift")
    source_head = _git(paths.repo_root, "rev-parse", "HEAD")
    critical_hashes: dict[str, str] = {}
    for relative_path in CRITICAL_TRACKED_PATHS:
        path = paths.repo_root / relative_path
        payload = _secure_regular_bytes(
            path, label=f"critical source {relative_path}"
        )
        try:
            committed = _git(
                paths.repo_root,
                "show",
                f"HEAD:{relative_path.as_posix()}",
                binary=True,
            )
        except subprocess.CalledProcessError as exc:
            raise Q75CoordinateGateError(
                f"critical source is not tracked at HEAD: {relative_path}"
            ) from exc
        _require(
            payload == committed,
            f"critical source differs from HEAD: {relative_path}",
        )
        critical_hashes[relative_path.as_posix()] = hashlib.sha256(payload).hexdigest()
    return {
        "official_head": official_head,
        "official_status": official_status,
        "frozen_tag": FROZEN_TAG,
        "frozen_commit": frozen_commit,
        "source_commit_at_execution": source_head,
        "critical_execution_source_sha256": critical_hashes,
        "pilot_status_porcelain_v1_z": _git(
            paths.repo_root, "status", "--porcelain=v1", "-z"
        ),
    }


def _validate_authorization_semantics(document: object, paths: GatePaths) -> None:
    _require(paths.wsi == WSI_PATH, "authorization validation WSI path drift")
    _require(paths.output == OUTPUT_DIR, "authorization validation output path drift")
    try:
        validate_q75_coordinate_execution_authorization(document)  # type: ignore[arg-type]
    except Q75CoordinateAuthorizationError as exc:
        raise Q75CoordinateGateError(str(exc)) from exc


def _validate_bound_sources(paths: GatePaths) -> dict[str, str]:
    expected = ((AUTH_RELATIVE_PATH, AUTH_SHA256), *BOUND_FILES)
    observed: dict[str, str] = {}
    authorization_payload: bytes | None = None
    for relative_path, expected_sha256 in expected:
        path = paths.repo_root / relative_path
        payload = _secure_regular_bytes(path, label=f"bound source {relative_path}")
        actual_sha256 = hashlib.sha256(payload).hexdigest()
        _require(
            actual_sha256 == expected_sha256,
            f"bound source SHA-256 drift: {relative_path}",
        )
        observed[relative_path.as_posix()] = actual_sha256
        if relative_path == AUTH_RELATIVE_PATH:
            authorization_payload = payload
    _require(
        paths.auth == paths.repo_root / AUTH_RELATIVE_PATH,
        "authorization path drift",
    )
    _require(authorization_payload is not None, "authorization payload was not bound")
    try:
        authorization = yaml.safe_load(authorization_payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise Q75CoordinateGateError("authorization YAML is unreadable") from exc
    _validate_authorization_semantics(authorization, paths)
    return observed


def _validate_static_policy_bindings() -> None:
    _require((S_THRESH, M_THRESH, CLOSE, USE_OTSU) == (
        MASK_STHRESH, MASK_MTHRESH, MASK_CLOSE, MASK_USE_OTSU
    ), "shared segmentation threshold binding drift")
    _require((A_T, A_H, MAX_N_HOLES, REF_PATCH_SIZE) == (
        MASK_AREA_THRESHOLD,
        MASK_HOLE_AREA_THRESHOLD,
        MASK_MAX_HOLES,
        MASK_REFERENCE_PATCH_SIZE,
    ), "shared segmentation area binding drift")
    _require(GRID_ANCHOR == (0, 0), "Q75 grid anchor drift")
    _require(SCALE_2X_LEVEL_0_FOOTPRINT == SCALE_2X_LEVEL_0_STEP == (512, 512), "Q75 2x lattice drift")
    _require(SCALE_4X_LEVEL_0_FOOTPRINT == SCALE_4X_LEVEL_0_STEP == (1024, 1024), "Q75 4x lattice drift")


def _validate_paths(paths: GatePaths) -> None:
    _validate_static_policy_bindings()
    _require(MASK_LOCATION == (0, 0), "mask read location constant drift")
    _require(MASK_LEVEL == 2, "mask read level constant drift")
    _require(MASK_SIZE == (6_783, 5_654), "mask read size constant drift")
    _require(paths.repo_root == REPO_ROOT, "pilot repository path drift")
    _require(paths.official_repo == OFFICIAL_REPO, "official repository path drift")
    _require(paths.incoming == INCOMING_DIR, "Q75 incoming directory drift")
    _require(paths.wsi == WSI_PATH, "Q75 WSI path drift")
    _require(paths.omic == OMIC_PATH, "BRCA Omic archive path drift")
    _require(paths.output == OUTPUT_DIR, "Q75 output path drift")
    _require_no_symlink_components(paths.wsi, include_leaf=True)
    _require_no_symlink_components(paths.omic, include_leaf=True)
    _require_no_symlink_components(paths.output, include_leaf=False)
    for repo in (paths.repo_root.resolve(), paths.official_repo.resolve()):
        _require(
            not paths.output.is_relative_to(repo),
            f"coordinate output must be outside Git worktree: {repo}",
        )
    _require(not os.path.lexists(paths.output), "coordinate output already exists")
    lock = paths.output.parent / f".{paths.output.name}.lock"
    _require(not os.path.lexists(lock), "coordinate output lock already exists")
    staging_prefix = f".{paths.output.name}.staging."
    _require(
        not any(
            item.name.startswith(staging_prefix)
            for item in paths.output.parent.iterdir()
        ),
        "coordinate output has a stale/active staging sibling",
    )


def _df_snapshot(path: Path) -> dict[str, object]:
    result = subprocess.run(
        ["df", "-B1", "--output=source,size,used,avail,pcent,target", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    _require(len(lines) == 2, "unexpected df output")
    fields = lines[1].split()
    _require(len(fields) >= 6, "unexpected df data row")
    filesystem, total, used, available, percent = fields[:5]
    values = (int(total), int(used), int(available))
    _require(
        all(value >= 0 for value in values)
        and values[2] >= _MINIMUM_AVAILABLE_BYTES,
        "insufficient filesystem headroom for Q75 coordinate gate",
    )
    return {
        "measurement": "runtime_df_capacity_not_lightning_logical_quota",
        "filesystem": filesystem,
        "total_bytes": values[0],
        "used_bytes": values[1],
        "available_bytes": values[2],
        "use_percent": percent,
        "mountpoint": " ".join(fields[5:]),
        "minimum_required_available_bytes": _MINIMUM_AVAILABLE_BYTES,
    }


def _policy_evidence() -> Q75CoordinateEvidence:
    return Q75CoordinateEvidence(
        user_statement=EXPECTED_USER_STATEMENT,
        user_statement_sha256=EXPECTED_USER_STATEMENT_SHA256,
        header_result_sha256=EXPECTED_HEADER_RESULT_SHA256,
        header_report_sha256=EXPECTED_HEADER_REPORT_SHA256,
        header_gate_source_commit=EXPECTED_HEADER_GATE_SOURCE_COMMIT,
        header_result_commit=EXPECTED_HEADER_RESULT_COMMIT,
        scale_approval_commit=EXPECTED_SCALE_APPROVAL_COMMIT,
        scale_config_sha256=EXPECTED_SCALE_CONFIG_SHA256,
        scale_source_sha256=EXPECTED_SCALE_SOURCE_SHA256,
        scale_provenance_sha256=EXPECTED_SCALE_PROVENANCE_SHA256,
        scale_report_sha256=EXPECTED_SCALE_REPORT_SHA256,
        q25_coordinate_source_sha256=EXPECTED_Q25_COORDINATE_SOURCE_SHA256,
        q25_coordinate_policy_sha256=EXPECTED_Q25_COORDINATE_POLICY_SHA256,
        q50_coordinate_source_sha256=EXPECTED_Q50_COORDINATE_SOURCE_SHA256,
        q50_coordinate_policy_sha256=EXPECTED_Q50_COORDINATE_POLICY_SHA256,
        coordinate_artifact_schema_sha256=EXPECTED_COORDINATE_ARTIFACT_SCHEMA_SHA256,
        known_issues_sha256=EXPECTED_KNOWN_ISSUES_SHA256,
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        size_bytes=EXPECTED_SIZE_BYTES,
        md5=EXPECTED_MD5,
        sha256=EXPECTED_SHA256,
        exact_omic_source_row_index=EXPECTED_EXACT_OMIC_SOURCE_ROW_INDEX,
        rna_shape=EXPECTED_RNA_SHAPE,
        mutation_shape=EXPECTED_MUTATION_SHAPE,
        cnv_shape=EXPECTED_CNV_SHAPE,
    )


def _locked_policy_plan(slide: object) -> Q75CoordinatePolicyPlan:
    properties = slide.properties
    try:
        mpp_x = float(properties[openslide.PROPERTY_NAME_MPP_X])
        mpp_y = float(properties[openslide.PROPERTY_NAME_MPP_Y])
        dimensions = tuple(tuple(item) for item in slide.level_dimensions)
        downsamples = tuple(float(item) for item in slide.level_downsamples)
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise Q75CoordinateGateError("Q75 OpenSlide header is invalid") from exc
    return review_q75_coordinate_policy(
        evidence=_policy_evidence(),
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        level_dimensions=dimensions,
        level_downsamples=downsamples,
    )


def _read_exact_mask(
    slide: object,
) -> tuple[np.ndarray, Q75CoordinatePolicyPlan, int, str]:
    plan = _locked_policy_plan(slide)
    image = slide.read_region(MASK_LOCATION, MASK_LEVEL, MASK_SIZE)
    try:
        array = np.ascontiguousarray(np.asarray(image))
    finally:
        close = getattr(image, "close", None)
        if callable(close):
            close()
    _require(array.dtype == np.uint8, "mask image dtype drift")
    _require(
        array.ndim == 3
        and array.shape[:2] == (MASK_SIZE[1], MASK_SIZE[0])
        and array.shape[2] in (3, 4),
        "mask image shape/channels drift",
    )
    digest = hashlib.sha256(array.tobytes(order="C")).hexdigest()
    return array, plan, 1, digest


def _require_coordinate_array(
    coordinates: np.ndarray, *, branch: str, plan: Q75CoordinatePolicyPlan
) -> None:
    _require(isinstance(coordinates, np.ndarray), f"{branch} must be ndarray")
    _require(coordinates.dtype == np.int64, f"{branch} dtype drift")
    _require(coordinates.ndim == 2 and coordinates.shape[1:] == (2,), f"{branch} shape drift")
    _require(coordinates.shape[0] > 0, f"{branch} coordinates are empty")
    _require(np.unique(coordinates, axis=0).shape[0] == len(coordinates), f"{branch} coordinates duplicate")
    rows = [tuple(map(int, row)) for row in coordinates.tolist()]
    _require(rows == sorted(rows, key=lambda point: (point[1], point[0])), f"{branch} ordering drift")
    branch_plan = plan.branch_for(branch)
    width, height = plan.level_dimensions[0]
    footprint_x, footprint_y = branch_plan.level_0_declared_footprint
    _require(np.all(coordinates[:, 0] >= 0) and np.all(coordinates[:, 1] >= 0), f"{branch} negative origin")
    _require(np.all(coordinates[:, 0] <= width - footprint_x), f"{branch} x bounds drift")
    _require(np.all(coordinates[:, 1] <= height - footprint_y), f"{branch} y bounds drift")
    _require(np.all(coordinates[:, 0] % branch_plan.level_0_step[0] == 0), f"{branch} x lattice drift")
    _require(np.all(coordinates[:, 1] % branch_plan.level_0_step[1] == 0), f"{branch} y lattice drift")
    _require(len(coordinates) <= branch_plan.theoretical_full_slide_sites_before_tissue_filter, f"{branch} count exceeds theoretical grid")


def _build_q75_coordinate_bags(
    mask: np.ndarray, *, plan: Q75CoordinatePolicyPlan
) -> Q75CoordinateBags:
    _require(plan.policy_status == POLICY_STATUS, "Q75 policy status drift")
    geometry = segment_tissue_contours(
        mask,
        level_0_dimensions=plan.level_dimensions[0],
        mask_dimensions=plan.mask.dimensions,
    )
    expected_mask_scale = plan.mask.coordinate_geometry_scale_xy
    _require(
        all(abs(actual - expected) <= 1e-12 for actual, expected in zip(
            geometry.mask_downsample_xy, expected_mask_scale, strict=True
        )),
        "Q75 mask coordinate scale drift",
    )
    scale_2x_plan = plan.branch_for("scale_2x")
    scale_4x_plan = plan.branch_for("scale_4x")
    scale_2x = generate_level_0_lattice_coordinates(
        level_0_dimensions=plan.level_dimensions[0],
        level_0_patch_size=scale_2x_plan.level_0_declared_footprint[0],
        level_0_step=scale_2x_plan.level_0_step[0],
        geometry=geometry,
    )
    scale_4x = generate_level_0_lattice_coordinates(
        level_0_dimensions=plan.level_dimensions[0],
        level_0_patch_size=scale_4x_plan.level_0_declared_footprint[0],
        level_0_step=scale_4x_plan.level_0_step[0],
        geometry=geometry,
    )
    _require_coordinate_array(scale_2x, branch="scale_2x", plan=plan)
    _require_coordinate_array(scale_4x, branch="scale_4x", plan=plan)
    return Q75CoordinateBags(
        scale_2x=scale_2x,
        scale_4x=scale_4x,
        contour_count=len(geometry.contours),
        retained_hole_count=sum(len(items) for items in geometry.holes),
        mask_downsample_xy=geometry.mask_downsample_xy,
    )


def _metadata_for_branch(
    branch: str,
    *,
    plan: Q75CoordinatePolicyPlan,
    mask_image_channels: int,
    mask_image_sha256: str,
    contour_count: int,
    retained_hole_count: int,
) -> CoordinateBranchMetadata:
    branch_plan = plan.branch_for(branch)
    shared = dict(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        wsi_filename=EXPECTED_FILENAME,
        wsi_size_bytes=EXPECTED_SIZE_BYTES,
        wsi_md5=EXPECTED_MD5,
        wsi_sha256=EXPECTED_SHA256,
        level_0_dimensions=plan.level_dimensions[0],
        mask_level=MASK_LEVEL,
        mask_level_dimensions=plan.mask.dimensions,
        openslide_reported_mask_downsample=plan.level_downsamples[MASK_LEVEL],
        mask_image_channels=mask_image_channels,
        mask_image_sha256=mask_image_sha256,
        mask_parameters={
            "color_space": "HSV_saturation",
            "sthresh": MASK_STHRESH,
            "mthresh": MASK_MTHRESH,
            "close": MASK_CLOSE,
            "use_otsu": MASK_USE_OTSU,
            "a_t": MASK_AREA_THRESHOLD,
            "a_h": MASK_HOLE_AREA_THRESHOLD,
            "max_n_holes": MASK_MAX_HOLES,
            "reference_patch_size": MASK_REFERENCE_PATCH_SIZE,
            "contour_rule": "pinned_four_pt_easy_any_probe_on_or_inside",
            "hole_rule": "reject_only_when_mapped_patch_center_is_strictly_inside",
        },
        contour_count=contour_count,
        retained_hole_count=retained_hole_count,
        clam_commit=CLAM_COMMIT,
        policy_sha256=COORDINATE_POLICY_SHA256,
    )
    return CoordinateBranchMetadata(
        branch=branch,
        source_level=branch_plan.source_level,
        source_level_dimensions=plan.level_dimensions[branch_plan.source_level],
        openslide_reported_source_downsample=plan.level_downsamples[branch_plan.source_level],
        source_patch_size=branch_plan.source_footprint,
        output_patch_size=branch_plan.output_patch,
        level_0_declared_footprint=branch_plan.level_0_declared_footprint,
        level_0_step=branch_plan.level_0_step,
        target_mpp=branch_plan.target_mpp,
        effective_mpp=branch_plan.effective_mpp,
        interpolation=branch_plan.later_interpolation,
        resampling=(
            "explicit_2x_spatial_downsample"
            if branch == "scale_2x"
            else "none"
        ),
        geometry_compatibility=branch_plan.geometry_compatibility,
        **shared,
    )


def _record_for_branch(
    record: CoordinateArtifactSetRecord, branch: str
) -> dict[str, object]:
    item = record.branch_for(branch)
    return {
        "coordinate_count": item.coordinate_count,
        "h5_path": str(item.path),
        "h5_size_bytes": item.size_bytes,
        "h5_sha256": item.sha256,
        "coordinates_sha256": item.coordinates_sha256,
    }


def _postpublication_transaction_observation(output: Path) -> dict[str, object]:
    """Best-effort observation; never turns a published result into failure."""

    try:
        lock = output.parent / f".{output.name}.lock"
        staging_prefix = f".{output.name}.staging."
        staging = sorted(
            item.name
            for item in output.parent.iterdir()
            if item.name.startswith(staging_prefix)
        )
        files = sorted(item.name for item in output.iterdir())
        return {
            "observation_available": True,
            "runner_created_lock_present_after_return": os.path.lexists(lock),
            "runner_created_staging_present_after_return": bool(staging),
            "staging_entries": staging,
            "final_directory_present": output.is_dir() and not output.is_symlink(),
            "final_exact_files": files,
            "permitted_cleanup_scope": (
                "runner_created_q75_coordinate_lock_and_staging_paths_only"
            ),
            "preexisting_or_final_artifact_deletion_permitted": False,
        }
    except Exception as exc:
        return {
            "observation_available": False,
            "error_type": type(exc).__name__,
            "permitted_cleanup_scope": (
                "runner_created_q75_coordinate_lock_and_staging_paths_only"
            ),
            "preexisting_or_final_artifact_deletion_permitted": False,
        }


def run_coordinate_gate(
    *,
    paths: GatePaths = GatePaths(),
    slide_factory: Callable[[str], object] | None = None,
    wsi_opener: Callable[..., HeldWsi] | None = None,
    wsi_reverifier: Callable[..., None] | None = None,
    omic_opener: Callable[[Path], HeldOmic] | None = None,
    omic_loader: Callable[..., BrcaPatientOmics] | None = None,
    omic_reverifier: Callable[[HeldOmic], None] | None = None,
    mask_reader: Callable[
        [object], tuple[np.ndarray, Q75CoordinatePolicyPlan, int, str]
    ]
    | None = None,
    coordinate_builder: Callable[..., Q75CoordinateBags] | None = None,
    publisher: Callable[..., CoordinateArtifactSetRecord] | None = None,
) -> dict[str, object]:
    """Execute the exact Q75 gate; injected seams exist for fail-closed tests."""

    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    timings: dict[str, float] = {}
    slide_factory = openslide.OpenSlide if slide_factory is None else slide_factory
    wsi_opener = _open_verified_wsi if wsi_opener is None else wsi_opener
    wsi_reverifier = (
        _reverify_held_wsi if wsi_reverifier is None else wsi_reverifier
    )
    omic_opener = _open_verified_omic if omic_opener is None else omic_opener
    omic_loader = load_brca_patient_omics if omic_loader is None else omic_loader
    omic_reverifier = (
        _reverify_held_omic if omic_reverifier is None else omic_reverifier
    )
    mask_reader = _read_exact_mask if mask_reader is None else mask_reader
    coordinate_builder = (
        _build_q75_coordinate_bags
        if coordinate_builder is None
        else coordinate_builder
    )
    publisher = publish_brca_coordinate_artifacts if publisher is None else publisher
    _validate_paths(paths)
    bound_sources = _validate_bound_sources(paths)
    repositories_before = _repository_snapshot(paths)
    storage_before = _df_snapshot(paths.output.parent)

    held: HeldWsi | None = None
    held_omic: HeldOmic | None = None
    held_open = False
    held_omic_open = False
    slide = None
    try:
        stage = time.perf_counter()
        held_omic = omic_opener(paths.omic)
        held_omic_open = True
        omic_first = _load_and_validate_exact_omics(
            held_omic.stable_path, loader=omic_loader
        )
        timings["omic_secure_open_hash_and_exact_row_match_seconds"] = (
            time.perf_counter() - stage
        )

        stage = time.perf_counter()
        held = wsi_opener(paths.wsi, incoming=paths.incoming)
        held_open = True
        timings["wsi_identity_and_two_pass_prehash_seconds"] = time.perf_counter() - stage

        stage = time.perf_counter()
        slide = slide_factory(str(held.stable_path))
        close = getattr(slide, "close", None)
        _require(callable(close), "OpenSlide object lacks close")
        mask_image, plan, read_count, mask_sha256 = mask_reader(slide)
        timings["openslide_header_and_single_mask_read_seconds"] = time.perf_counter() - stage
        _require(read_count == 1, "exactly one Q75 mask read is required")
        close()
        slide = None

        stage = time.perf_counter()
        bags = coordinate_builder(mask_image, plan=plan)
        timings["segmentation_and_coordinate_generation_seconds"] = time.perf_counter() - stage
        _require(bags.policy_status == POLICY_STATUS, "Q75 coordinate policy status drift")
        common_metadata = {
            "plan": plan,
            "mask_image_channels": int(mask_image.shape[2]),
            "mask_image_sha256": mask_sha256,
            "contour_count": bags.contour_count,
            "retained_hole_count": bags.retained_hole_count,
        }
        scale_2x_metadata = _metadata_for_branch("scale_2x", **common_metadata)
        scale_4x_metadata = _metadata_for_branch("scale_4x", **common_metadata)

        # All mutable-input and semantic gates precede the terminal publisher.
        stage = time.perf_counter()
        wsi_reverifier(held, incoming=paths.incoming)
        timings["same_descriptor_final_hash_seconds"] = time.perf_counter() - stage
        stage = time.perf_counter()
        omic_second = _load_and_validate_exact_omics(
            held_omic.stable_path, loader=omic_loader
        )
        _require(omic_second == omic_first, "Q75 Omic exact row changed")
        omic_reverifier(held_omic)
        timings["omic_exact_row_and_same_descriptor_recheck_seconds"] = (
            time.perf_counter() - stage
        )
        _validate_paths(paths)
        _require(
            _validate_bound_sources(paths) == bound_sources,
            "bound sources changed during Q75 coordinate generation",
        )
        repositories_prepublication = _repository_snapshot(paths)
        _require(
            repositories_prepublication == repositories_before,
            "repository state changed during Q75 coordinate generation",
        )

        # Input descriptors are no longer needed. Close them before the
        # terminal no-replace publisher so no close failure can occur after a
        # successfully published final directory.
        os.close(held.descriptor)
        held_open = False
        os.close(held_omic.descriptor)
        held_omic_open = False

        stage = time.perf_counter()
        record = publisher(
            paths.output,
            scale_2x_coordinates=bags.scale_2x,
            scale_4x_coordinates=bags.scale_4x,
            scale_2x_metadata=scale_2x_metadata,
            scale_4x_metadata=scale_4x_metadata,
        )
        timings["atomic_publication_and_validation_seconds"] = time.perf_counter() - stage
        transaction_observation = _postpublication_transaction_observation(
            paths.output
        )
    finally:
        if slide is not None:
            close = getattr(slide, "close", None)
            if callable(close):
                close()
        if held is not None and held_open:
            os.close(held.descriptor)
        if held_omic is not None and held_omic_open:
            os.close(held_omic.descriptor)

    timings["total_seconds"] = time.perf_counter() - started
    try:
        storage_after = _df_snapshot(paths.output.parent)
    except Exception as exc:  # Publication is terminal; observation is non-gating.
        storage_after = {
            "measurement": "postpublication_df_unavailable_nonfatal",
            "error_type": type(exc).__name__,
        }
    return {
        "schema": SCHEMA,
        "status": "BRCA_Q75_COORDINATES_VERIFIED",
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "patient_id": EXPECTED_PATIENT_ID,
        "slide_id": EXPECTED_SLIDE_ID,
        "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
        "authorization_sha256": AUTH_SHA256,
        "bound_source_sha256": bound_sources,
        "source_commit_at_execution": repositories_before["source_commit_at_execution"],
        "critical_execution_source_sha256": repositories_before["critical_execution_source_sha256"],
        "runtime_versions": {
            "python_executable": sys.executable,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "h5py": h5py.__version__,
            "openslide_python": openslide.__version__,
            "openslide_library": openslide.__library_version__,
        },
        "wsi": {
            "path": str(held.path),
            "size_bytes": held.token.size,
            "md5": held.md5,
            "sha256": held.sha256,
            "regular_non_symlink": True,
            "held_o_nofollow_descriptor_used": True,
            "openslide_stable_proc_fd_path_used": True,
            "two_independent_sha256_passes_before_openslide": True,
            "same_descriptor_final_md5_sha256_recheck": True,
        },
        "header": {
            "mpp_x": EXPECTED_MPP_X,
            "mpp_y": EXPECTED_MPP_Y,
            "level_dimensions": [list(item) for item in plan.level_dimensions],
            "openslide_level_downsamples": list(plan.level_downsamples),
        },
        "omic": {
            **omic_second,
            "held_o_nofollow_descriptor_used": True,
            "exact_row_rematched_before_publication": True,
        },
        "mask": {
            "read_region_calls": read_count,
            "location_level_0": list(MASK_LOCATION),
            "level": MASK_LEVEL,
            "size_at_level": list(MASK_SIZE),
            "channels": int(mask_image.shape[2]),
            "hash_serialization": "contiguous_uint8_C_order_raw_bytes",
            "contiguous_uint8_bytes_sha256": mask_sha256,
            "contour_count": bags.contour_count,
            "retained_hole_count": bags.retained_hole_count,
            "coordinate_geometry_scale_xy": list(bags.mask_downsample_xy),
        },
        "coordinate_artifacts": {
            "directory": str(record.directory),
            "manifest_path": str(record.manifest_path),
            "manifest_sha256": record.manifest_sha256,
            "scale_2x": _record_for_branch(record, "scale_2x"),
            "scale_4x": _record_for_branch(record, "scale_4x"),
            "transaction_cleanup": transaction_observation,
        },
        "storage": {"before": storage_before, "after": storage_after},
        "repositories_unchanged_before_publication": True,
        "operations": {
            "openslide_open_count": 1,
            "header_reads": 1,
            "mask_read_region_calls": 1,
            "level_0_or_level_1_patch_pixel_reads": 0,
            "thumbnail_reads": 0,
            "associated_image_reads": 0,
            "patch_resampling": 0,
            "resnet50_feature_extraction": 0,
            "pt_generation": 0,
            "healnet_execution": 0,
            "gpu_operations": 0,
            "training": 0,
            "q25_q50_reruns_or_modifications": 0,
            "full_cohort_operations": 0,
            "google_drive_operations": 0,
            "raw_wsi_deletions": 0,
            "preexisting_raw_user_project_or_final_artifact_deletions": 0,
            "runner_owned_ephemeral_transaction_cleanup_scope": (
                "Q75_coordinate_lock_and_staging_created_by_this_run_only"
            ),
        },
        "timings": timings,
        "required_stop_reached": True,
    }


def _strict_json(document: dict[str, object]) -> str:
    return json.dumps(document, allow_nan=False, indent=2, sort_keys=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--execute-exact-q75-coordinate-gate",
        action="store_true",
        help="Acknowledge the exact Q75 one-mask-read CPU boundary.",
    )
    args = parser.parse_args(argv)
    if not args.execute_exact_q75_coordinate_gate:
        parser.error("explicit --execute-exact-q75-coordinate-gate is required")
    try:
        result = run_coordinate_gate()
    except Exception as exc:
        print(
            _strict_json(
                {
                    "schema": SCHEMA,
                    "status": "BRCA_Q75_COORDINATE_GATE_BLOCKED",
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
