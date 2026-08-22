#!/usr/bin/env python3
"""Acquire and inspect only the authorized BRCA P0001 OpenSlide header.

This fail-closed runner performs one single-client GDC transfer, verifies
the exact regular non-symlink payload through a held ``O_NOFOLLOW`` descriptor,
rematches the exact released Omic row, reads only OpenSlide header properties,
and publishes append-only result files.  It contains no pixel-access call.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import openslide
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiscale_feature_pilot.src.brca_omic import (  # noqa: E402
    BRCA_RELEASE_ARCHIVE_SHA256,
    load_official_brca_patient_omics,
)
from multiscale_feature_pilot.src.brca_streaming_production_adapter import (  # noqa: E402
    load_frozen_cohort_order,
)


class P0001HeaderGateError(RuntimeError):
    """Raised whenever the authorized P0001 boundary cannot be proven."""


PATIENT = "TCGA-3C-AALK"
SLIDE = "TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53.svs"
UUID = "93b26333-5723-4fa4-a4de-6124c04ab243"
SIZE = 1_769_848_096
MD5 = "3d63b3311612d763525b6edb0848b986"
OMIC_ROW = "4"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_p0001_acquisition_header_authorization.yaml"
MANIFEST = ROOT / (
    "multiscale_feature_pilot/provenance/brca_first_eight_production_manifests/"
    "P0001_93b26333-5723-4fa4-a4de-6124c04ab243.REQUEST_ONLY.gdc.tsv"
)
REQUEST = ROOT / "multiscale_feature_pilot/config/brca_first_eight_production_execution_request.yaml"
FIRST_EIGHT = ROOT / "reports/brca_first_eight_canary_proposal.tsv"
ADAPTER_POLICY = ROOT / "multiscale_feature_pilot/config/brca_894_production_adapter_policy.yaml"
SURVIVAL_SPLIT = ROOT / "multiscale_feature_pilot/provenance/brca_894_survival_split.tsv"
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
INCOMING = Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_PRODUCTION_P0001.incoming")
WSI = INCOMING / UUID / SLIDE
PARCEL = INCOMING / UUID / "logs" / f"{SLIDE}.parcel"
LOCK = INCOMING.parent / ".BRCA_PRODUCTION_P0001.header.lock"
OMIC = Path(
    "/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/"
    "data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
CLIENT = Path("/teamspace/studios/this_studio/tools/gdc-client/2.3.0/gdc-client")
CLIENT_SHA256 = "1df9281cbacbb53413607a4b9b682947dcb48b6ac7fade6527748bc025ae8c96"
AUTH_STATUS = "AUTHORIZED_P0001_ACQUISITION_AND_HEADER_ONLY"
RESULT_STATUS = "BRCA_PRODUCTION_P0001_HEADER_METADATA_VERIFIED"
RESULT_BUNDLE = ROOT / "multiscale_feature_pilot/provenance/brca_p0001_header_metadata_result"
RESULT = RESULT_BUNDLE / "result.yaml"
REPORT = RESULT_BUNDLE / "report.md"
REQUIRED_AVAILABLE = 2 * SIZE + 20_000_000_000
STATEMENT_SHA256 = "1d755c3432c29450ba40520287f28b1c6273c6969af1962b9345f6aededf824b"
AUTH_SHA256 = "691a8d536c9a614a326d26f971f586b7e5ab441a5efea189e347f2c4d664d61a"
MANIFEST_SHA256 = "1c05dc64b6af54604648b52e688e02f611f8e42aaf82cf911c81035b3fc385f2"
REQUEST_SHA256 = "04da2f211bfae390ec94ec2dd082ed032204d9c3b81424c6273e33679b14134b"
DOWNLOAD_TIMEOUT_SECONDS = 10_800
SOURCE_BINDING_HASHES = {
    FIRST_EIGHT: "940b8fd1f7d194c2c9b7c69ddae58ffff3c55196b6841ac859c60cbc01095dfd",
    ADAPTER_POLICY: "78c24ea714814c9775b1f6c1268179d3fe47c89c55391485bdb65691f51dddd8",
    SURVIVAL_SPLIT: "3e519a26eaa24852862bf368a48cceffaf26783c73e80007737134ae6ed626ad",
    ALIGNMENT: "13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe",
}
EXPECTED_OMIC_HASHES = {
    "rna": "1894e15a5dbba2559c61e8521394599153a0ada90cf482fd9eb0c45347f5082a",
    "mutation": "33767ff31d3c7c11a69ba46c746125f679492e03f5dec7c48f8117aa2a6b3c52",
    "cnv": "a0ef410e624c698475b78dc0270bf2613e2e320ba4935f8580cb0867d41bfd50",
}
LATER_PATIENT_LABELS = tuple(f"P{index:04d}" for index in range(2, 9))
FROZEN_COHORT_ORDER_SHA256 = "1c97fa4f8305185f2da191f5ebaed603db7d2bdd11c89a580e784ef46655af5a"
MANIFEST_DIRECTORY = MANIFEST.parent
BLOCK_MANIFESTS = tuple(sorted(MANIFEST_DIRECTORY.glob("P*.REQUEST_ONLY.gdc.tsv")))
CRITICAL = (
    AUTH,
    MANIFEST,
    REQUEST,
    FIRST_EIGHT,
    ADAPTER_POLICY,
    SURVIVAL_SPLIT,
    *BLOCK_MANIFESTS,
    Path(__file__).resolve(),
    ROOT / "multiscale_feature_pilot/__init__.py",
    ROOT / "multiscale_feature_pilot/src/__init__.py",
    ROOT / "multiscale_feature_pilot/src/brca_omic.py",
    ROOT / "multiscale_feature_pilot/src/brca_streaming_production_adapter.py",
    ROOT / "multiscale_feature_pilot/src/brca_singleton_streaming_policy.py",
    ROOT / "multiscale_feature_pilot/src/brca_streaming_executor_v2.py",
    ROOT / "multiscale_feature_pilot/src/brca_streaming_recovery_v2.py",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise P0001HeaderGateError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_fd(descriptor: int) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 - required external integrity binding
    sha256 = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while True:
        chunk = os.read(descriptor, 4 * 1024 * 1024)
        if not chunk:
            break
        md5.update(chunk)
        sha256.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return md5.hexdigest(), sha256.hexdigest()


def _regular_non_symlink(path: Path) -> os.stat_result:
    value = path.lstat()
    require(not stat.S_ISLNK(value.st_mode), f"symlink forbidden: {path}")
    require(stat.S_ISREG(value.st_mode), f"regular file required: {path}")
    return value


def _token(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_size,
        value.st_mtime_ns, value.st_ctime_ns,
    )


def _directory_non_symlink(path: Path) -> None:
    value = path.lstat()
    require(not stat.S_ISLNK(value.st_mode), f"directory symlink forbidden: {path}")
    require(stat.S_ISDIR(value.st_mode), f"directory required: {path}")


def _disk(path: Path) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": usage.total, "used_bytes": usage.used, "available_bytes": usage.free}


def _git(*arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def validate_source(expected_commit: str) -> dict[str, Any]:
    require(len(expected_commit) == 40, "expected source commit must be a full SHA")
    head = _git("rev-parse", "HEAD")
    require(head == expected_commit, f"HEAD mismatch: expected {expected_commit}, received {head}")
    hashes: dict[str, str] = {}
    for path in CRITICAL:
        relative = path.relative_to(ROOT).as_posix()
        require(path.is_file() and not path.is_symlink(), f"critical file unsafe: {relative}")
        working = path.read_bytes()
        head_bytes = subprocess.run(
            ["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        require(working == head_bytes, f"critical file differs from HEAD: {relative}")
        hashes[relative] = hashlib.sha256(working).hexdigest()
    return {"source_commit": head, "critical_file_sha256": hashes}


def validate_authorization() -> dict[str, Any]:
    _regular_non_symlink(AUTH)
    require(sha256_path(AUTH) == AUTH_SHA256, "authorization config SHA256 drift")
    value = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    require(value["status"] == AUTH_STATUS and value["executable"] is True, "authorization locked")
    identity = value["identity"]
    require(identity["patient_id"] == PATIENT, "authorization patient drift")
    require(identity["slide_id"] == SLIDE and identity["filename"] == SLIDE, "slide drift")
    require(identity["gdc_uuid"] == UUID, "UUID drift")
    require(identity["declared_size_bytes"] == SIZE and identity["expected_md5"] == MD5, "file drift")
    require(identity["omic_source_row_index"] == OMIC_ROW, "Omic row drift")
    execution = value["execution"]
    require(execution["active_patient_concurrency"] == 1, "patient concurrency must be one")
    require(execution["active_gdc_clients"] == 1, "GDC concurrency must be one")
    require(Path(execution["request_manifest"]) == MANIFEST.relative_to(ROOT), "manifest drift")
    require(execution["request_manifest_sha256"] == MANIFEST_SHA256, "manifest binding drift")
    require(execution["strict_serial_order"] is True, "serial order must be locked")
    require(execution["stop_after_patient"] == "P0001", "P0001 stop drift")
    authority = value["authority"]
    require(authority["download_p0001"] is True and authority["openslide_header_only"] is True, "scope locked")
    for key in (
        "read_region_or_pixel_access", "masks_or_coordinates", "features_or_models",
        "gpu_or_cuda", "raw_file_deletion", "google_drive", "later_patient_processing",
        "cohort_expansion", "training",
    ):
        require(authority[key] is False, f"prohibited authority unlocked: {key}")
    statement = value["approval"]["exact_statement"]
    require("Begin with P0001" in statement, "P0001 approval absent")
    require("brca_first_eight_production_execution_request.yaml" in statement, "block request absent")
    require("Do not call `read_region`, access pixels" in statement, "pixel prohibition absent")
    statement_sha256 = hashlib.sha256(statement.encode("utf-8")).hexdigest()
    require(statement_sha256 == STATEMENT_SHA256, "exact authorization statement drift")
    require(value["approval"]["exact_statement_sha256"] == STATEMENT_SHA256, "statement binding drift")
    require(execution["conservative_required_available_bytes"] == REQUIRED_AVAILABLE, "storage binding drift")
    require(execution["fresh_preflight_required"] is True, "fresh preflight not required")
    bindings = value["source_bindings"]
    require(bindings["preparation_commit"] == "63c4e040072ff9c962a85e94b814eec1a52b901a", "preparation commit drift")
    require(bindings["production_request_sha256"] == REQUEST_SHA256, "request binding drift")
    require(bindings["first_eight_source_sha256"] == SOURCE_BINDING_HASHES[FIRST_EIGHT], "first-eight binding drift")
    require(bindings["row_level_alignment_sha256"] == SOURCE_BINDING_HASHES[ALIGNMENT], "alignment binding drift")
    require(bindings["frozen_cohort_order_sha256"] == FROZEN_COHORT_ORDER_SHA256, "cohort order binding drift")
    require(bindings["production_adapter_policy_sha256"] == SOURCE_BINDING_HASHES[ADAPTER_POLICY], "adapter policy binding drift")
    require(bindings["survival_split_sha256"] == SOURCE_BINDING_HASHES[SURVIVAL_SPLIT], "survival split binding drift")
    require(bindings["omic_archive_sha256"] == BRCA_RELEASE_ARCHIVE_SHA256, "Omic binding drift")
    require(sha256_path(REQUEST) == REQUEST_SHA256, "request file drift")
    request = yaml.safe_load(REQUEST.read_text(encoding="utf-8"))
    require(request["block_id"] == "BRCA_PRODUCTION_BLOCK_0001_0008", "block ID drift")
    require(request["binding"]["frozen_cohort_order_sha256"] == FROZEN_COHORT_ORDER_SHA256, "request order drift")
    require(request["binding"]["exact_cohort_indices"] == list(range(1, 9)), "request indices drift")
    contract = request["execution_contract"]
    require(contract["strict_order"] is True and contract["patient_concurrency"] == 1, "request serial contract drift")
    require(contract["download_concurrency"] == 1, "request download concurrency drift")
    for path, digest in SOURCE_BINDING_HASHES.items():
        require(sha256_path(path) == digest, f"source binding drift: {path.name}")
    return {
        "path": AUTH.relative_to(ROOT).as_posix(),
        "sha256": AUTH_SHA256,
        "statement_sha256": statement_sha256,
    }


def validate_manifest() -> str:
    cohort = load_frozen_cohort_order(ALIGNMENT)
    require(len(cohort) == 894, "frozen cohort cardinality drift")
    frozen_first = cohort[0]
    require(frozen_first.cohort_index == 1 and frozen_first.patient_id == PATIENT, "frozen P0001 order drift")
    require(frozen_first.slide_id == SLIDE and frozen_first.gdc_uuid == UUID, "frozen P0001 identity drift")
    require(frozen_first.omic_source_row_id == OMIC_ROW, "frozen P0001 Omic row drift")
    require(frozen_first.md5 == MD5 and frozen_first.size_bytes == SIZE, "frozen P0001 file drift")
    require(frozen_first.manifest_sha256 == MANIFEST_SHA256, "frozen P0001 manifest digest drift")
    require(len(BLOCK_MANIFESTS) == 8, "first production block must contain exactly eight manifests")
    with FIRST_EIGHT.open("r", encoding="utf-8", newline="") as stream:
        block_rows = list(csv.DictReader(stream, delimiter="\t"))
    require(len(block_rows) == 8, "first production block must contain exactly eight rows")
    require([int(row["cohort_index"]) for row in block_rows] == list(range(1, 9)), "production order drift")
    for index, row in enumerate(block_rows, start=1):
        expected_name = f"P{index:04d}_{row['gdc_uuid']}.REQUEST_ONLY.gdc.tsv"
        candidate = MANIFEST_DIRECTORY / expected_name
        require(candidate in BLOCK_MANIFESTS, f"block manifest absent: {expected_name}")
        _regular_non_symlink(candidate)
        with candidate.open("r", encoding="utf-8", newline="") as stream:
            candidate_rows = list(csv.DictReader(stream, delimiter="\t"))
        require(len(candidate_rows) == 1, f"manifest row-count drift: {expected_name}")
        expected = {
            "id": row["gdc_uuid"], "filename": row["filename"], "md5": row["md5"],
            "size": row["size_bytes"], "state": row["state"],
        }
        require(candidate_rows[0] == expected, f"manifest identity drift: {expected_name}")
    first = block_rows[0]
    require(first["patient_id"] == PATIENT and first["slide_id"] == SLIDE, "P0001 cohort identity drift")
    require(first["gdc_uuid"] == UUID and first["omic_source_row_id"] == OMIC_ROW, "P0001 row drift")
    require(first["md5"] == MD5 and first["size_bytes"] == str(SIZE), "P0001 file drift")
    _regular_non_symlink(MANIFEST)
    with MANIFEST.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    require(len(rows) == 1, "authorized manifest must contain exactly one row")
    require(list(rows[0]) == ["id", "filename", "md5", "size", "state"], "manifest schema drift")
    require(rows[0] == {
        "id": UUID, "filename": SLIDE, "md5": MD5, "size": str(SIZE), "state": "released"
    }, "manifest identity drift")
    digest = sha256_path(MANIFEST)
    require(digest == MANIFEST_SHA256, "authorized manifest SHA256 drift")
    return digest


def _active_gdc_clients() -> list[int]:
    active: list[int] = []
    proc = Path("/proc")
    require(proc.is_dir(), "/proc unavailable for process isolation")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise P0001HeaderGateError(f"cannot audit process {entry.name}") from exc
        argv0 = command.split(b" ", 1)[0]
        if os.fsdecode(argv0) == str(CLIENT):
            active.append(int(entry.name))
    return active


def preflight() -> dict[str, Any]:
    require(not INCOMING.exists() and not INCOMING.is_symlink(), "P0001 incoming destination must be absent")
    require(not LOCK.exists() and not LOCK.is_symlink(), "P0001 execution lock already exists")
    require(not RESULT_BUNDLE.exists() and not RESULT_BUNDLE.is_symlink(), "append-only P0001 result bundle exists")
    for label in LATER_PATIENT_LABELS:
        later_incoming = INCOMING.parent / f"BRCA_PRODUCTION_{label}.incoming"
        later_lock = INCOMING.parent / f".BRCA_PRODUCTION_{label}.header.lock"
        later_result = ROOT / f"multiscale_feature_pilot/provenance/brca_{label.lower()}_header_metadata_result"
        require(not later_incoming.exists() and not later_incoming.is_symlink(), f"later patient already started: {label}")
        require(not later_lock.exists() and not later_lock.is_symlink(), f"later patient lock exists: {label}")
        require(not later_result.exists() and not later_result.is_symlink(), f"later patient result exists: {label}")
    parent = INCOMING.parent
    _directory_non_symlink(parent)
    available = _disk(parent)
    require(available["available_bytes"] >= REQUIRED_AVAILABLE, "conservative P0001 storage gate failed")
    require(not _active_gdc_clients(), "another gdc-client process is active")
    _regular_non_symlink(CLIENT)
    require(sha256_path(CLIENT) == CLIENT_SHA256, "gdc-client SHA256 mismatch")
    version = subprocess.run(
        [str(CLIENT), "--version"], check=True, capture_output=True, text=True, timeout=30
    ).stdout.strip()
    require(version == "2.3", f"gdc-client version mismatch: {version}")
    return {"filesystem_before_download": available, "gdc_client_version": version}


def _create_lock() -> tuple[int, tuple[int, int, int, int, int, int]]:
    descriptor = os.open(
        LOCK, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600
    )
    return descriptor, _token(os.fstat(descriptor))


def _create_owned_incoming() -> tuple[int, int]:
    require(not INCOMING.exists() and not INCOMING.is_symlink(), "P0001 incoming appeared after preflight")
    _directory_non_symlink(INCOMING.parent)
    os.mkdir(INCOMING, 0o700)
    details = INCOMING.lstat()
    require(stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode), "owned P0001 incoming is unsafe")
    require(stat.S_IMODE(details.st_mode) == 0o700, "owned P0001 incoming mode must be 0700")
    identity = (details.st_dev, details.st_ino)
    _fsync_directory(INCOMING.parent)
    return identity


def _require_owned_incoming(identity: tuple[int, int]) -> None:
    try:
        details = INCOMING.lstat()
    except FileNotFoundError as exc:
        raise P0001HeaderGateError("owned P0001 incoming disappeared") from exc
    require(stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode), "owned P0001 incoming is unsafe")
    require((details.st_dev, details.st_ino) == identity, "owned P0001 incoming identity changed")
    require(stat.S_IMODE(details.st_mode) == 0o700, "owned P0001 incoming mode changed")


def _diagnostic_tail(value: str, *, limit: int = 2000) -> str:
    sanitized = "".join(
        character if character in "\n\t" or ord(character) >= 32 else "?"
        for character in value
    )
    return sanitized[-limit:]


def _release_owned_lock(descriptor: int, token: tuple[int, int, int, int, int, int]) -> None:
    os.close(descriptor)
    try:
        current = LOCK.lstat()
    except FileNotFoundError as exc:
        raise P0001HeaderGateError("owned execution lock disappeared") from exc
    require(
        stat.S_ISREG(current.st_mode) and not stat.S_ISLNK(current.st_mode) and _token(current) == token,
        "owned execution lock identity changed",
    )
    LOCK.unlink()


def download() -> dict[str, Any]:
    started = time.perf_counter()
    command = [
        str(CLIENT), "download", "--no-related-files", "--no-annotations",
        "-m", str(MANIFEST), "-d", str(INCOMING),
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=DOWNLOAD_TIMEOUT_SECONDS,
    )
    elapsed = time.perf_counter() - started
    require(tuple(completed.args) == tuple(command), "executed GDC argv drift")
    stdout_tail = _diagnostic_tail(completed.stdout)
    stderr_tail = _diagnostic_tail(completed.stderr)
    require(
        completed.returncode == 0,
        f"GDC download failed ({completed.returncode}); stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}",
    )
    require(not _active_gdc_clients(), "gdc-client remained active after transfer")
    _regular_non_symlink(CLIENT)
    require(sha256_path(CLIENT) == CLIENT_SHA256, "gdc-client changed during transfer")
    return {
        "seconds": elapsed,
        "returncode": completed.returncode,
        "stdout_tail": stdout_tail,
        "stderr_tail": stderr_tail,
        "argv": command,
        "timeout_seconds": DOWNLOAD_TIMEOUT_SECONDS,
    }


def validate_tree() -> dict[str, Any]:
    _directory_non_symlink(INCOMING)
    _directory_non_symlink(INCOMING / UUID)
    _directory_non_symlink(INCOMING / UUID / "logs")
    directories = sorted(path for path in INCOMING.rglob("*") if path.is_dir())
    require(
        directories == sorted([INCOMING / UUID, INCOMING / UUID / "logs"]),
        f"unexpected P0001 download directory set: {directories}",
    )
    files = sorted(path for path in INCOMING.rglob("*") if path.is_file() or path.is_symlink())
    require(files == sorted([WSI, PARCEL]), f"unexpected P0001 download file set: {files}")
    wsi_stat = _regular_non_symlink(WSI)
    parcel_stat = _regular_non_symlink(PARCEL)
    require(WSI.suffix.lower() == ".svs", "payload is not SVS")
    require(wsi_stat.st_size == SIZE, f"WSI size mismatch: {wsi_stat.st_size}")
    forbidden = (".part", ".partial", ".tmp", ".incomplete", ".download")
    require(not any(any(token in path.name.lower() for token in forbidden) for path in files), "partial file found")
    return {
        "entries": [path.relative_to(INCOMING).as_posix() for path in files],
        "directories": [path.relative_to(INCOMING).as_posix() for path in directories],
        "parcel_size_bytes": parcel_stat.st_size,
        "parcel_sha256": sha256_path(PARCEL),
        "total_regular_file_bytes": sum(path.stat().st_size for path in files),
    }


def _tensor_hash(tensor: torch.Tensor) -> str:
    require(tensor.device.type == "cpu", "Omic tensor must remain on CPU")
    require(tensor.dtype is torch.float32 and tensor.is_contiguous(), "Omic tensor contract drift")
    require(bool(torch.isfinite(tensor).all().item()), "non-finite Omic tensor")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def load_omic() -> dict[str, Any]:
    _regular_non_symlink(OMIC)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(OMIC, flags)
    try:
        token = _token(os.fstat(descriptor))
        stable = Path(f"/proc/self/fd/{descriptor}")
        require(sha256_path(stable) == BRCA_RELEASE_ARCHIVE_SHA256, "Omic archive SHA256 mismatch")
        omics = load_official_brca_patient_omics(stable, case_id=PATIENT, slide_id=SLIDE)
        require(omics.source_row_index == OMIC_ROW, "exact Omic source row mismatch")
        require(_token(os.fstat(descriptor)) == token, "held Omic descriptor changed")
        result = {
            "source_row_index": omics.source_row_index,
            "exact_case_and_full_slide_match": True,
            "archive_sha256": BRCA_RELEASE_ARCHIVE_SHA256,
            "rna": {"shape": list(omics.rna.shape), "dtype": "float32", "content_sha256": _tensor_hash(omics.rna)},
            "mutation": {"shape": list(omics.mutation.shape), "dtype": "float32", "content_sha256": _tensor_hash(omics.mutation)},
            "cnv": {"shape": list(omics.cnv.shape), "dtype": "float32", "content_sha256": _tensor_hash(omics.cnv)},
        }
        for name, digest in EXPECTED_OMIC_HASHES.items():
            require(result[name]["content_sha256"] == digest, f"{name} content hash mismatch")
        return result
    finally:
        os.close(descriptor)


def collect_header(slide_factory: Callable[[str], Any] = openslide.OpenSlide) -> tuple[dict[str, Any], str]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(WSI, flags)
    try:
        before_stat = os.fstat(descriptor)
        before_token = _token(before_stat)
        path_token = _token(WSI.lstat())
        require(path_token == before_token, "WSI pathname does not bind held descriptor")
        require(before_stat.st_size == SIZE and stat.S_ISREG(before_stat.st_mode), "held WSI identity drift")
        first_md5, first_sha256 = _hash_fd(descriptor)
        require(first_md5 == MD5, f"WSI MD5 mismatch: {first_md5}")
        stable = f"/proc/self/fd/{descriptor}"
        slide = slide_factory(stable)
        try:
            mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            level_count = int(slide.level_count)
            dimensions = [list(map(int, pair)) for pair in slide.level_dimensions]
            downsamples = [float(value) for value in slide.level_downsamples]
        finally:
            slide.close()
        require(math.isfinite(mpp_x) and math.isfinite(mpp_y) and mpp_x > 0 and mpp_y > 0, "MPP must be finite and positive")
        require(level_count > 0 and len(dimensions) == level_count == len(downsamples), "pyramid length mismatch")
        require(all(w > 0 and h > 0 for w, h in dimensions), "invalid pyramid dimensions")
        require(downsamples[0] == 1.0 and all(math.isfinite(value) and value > 0 for value in downsamples), "invalid downsample values")
        require(all(downsamples[i] < downsamples[i + 1] for i in range(level_count - 1)), "downsamples not increasing")
        require(all(dimensions[i][0] >= dimensions[i + 1][0] and dimensions[i][1] >= dimensions[i + 1][1] for i in range(level_count - 1)), "dimensions not decreasing")
        require(_token(os.fstat(descriptor)) == before_token, "held WSI descriptor changed during header inspection")
        require(_token(WSI.lstat()) == before_token, "WSI pathname changed during header inspection")
        second_md5, second_sha256 = _hash_fd(descriptor)
        require((second_md5, second_sha256) == (first_md5, first_sha256), "same-descriptor hash changed")
        require(_token(os.fstat(descriptor)) == before_token, "held WSI descriptor changed during final hash")
        require(_token(WSI.lstat()) == before_token, "WSI pathname changed during final hash")
        levels = [
            {"level": index, "dimensions": pair, "downsample": downsamples[index]}
            for index, pair in enumerate(dimensions)
        ]
        return ({
            "inspection": "OPENSLIDE_HEADER_ONLY_NO_PIXEL_ACCESS",
            "held_o_nofollow_descriptor": True,
            "stable_proc_fd_path": True,
            "mpp_x": mpp_x,
            "mpp_y": mpp_y,
            "level_count": level_count,
            "levels": levels,
            "read_region_calls": 0,
            "thumbnail_calls": 0,
            "associated_image_accesses": 0,
        }, first_sha256)
    finally:
        os.close(descriptor)


def _write_new(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            require(written > 0, "short write during result publication")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable; atomic publication refused")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise P0001HeaderGateError(f"atomic no-replace result publication failed: {os.strerror(error)}")


def publish_result_set(yaml_payload: bytes, report_payload: bytes) -> None:
    require(not RESULT_BUNDLE.exists() and not RESULT_BUNDLE.is_symlink(), "append-only result bundle exists")
    stage = Path(os.path.join(RESULT_BUNDLE.parent, f".{RESULT_BUNDLE.name}.staging.{os.getpid()}"))
    require(not stage.exists() and not stage.is_symlink(), "result staging collision")
    os.mkdir(stage, 0o700)
    stage_details = stage.lstat()
    stage_identity = (stage_details.st_dev, stage_details.st_ino)
    published = False
    try:
        _write_new(stage / "result.yaml", yaml_payload)
        _write_new(stage / "report.md", report_payload)
        require((stage / "result.yaml").read_bytes() == yaml_payload, "staged YAML drift")
        require((stage / "report.md").read_bytes() == report_payload, "staged report drift")
        _fsync_directory(stage)
        _rename_directory_noreplace(stage, RESULT_BUNDLE)
        published = True
        _fsync_directory(RESULT_BUNDLE.parent)
    finally:
        if not published and stage.exists() and not stage.is_symlink():
            current_stage = stage.lstat()
            if (current_stage.st_dev, current_stage.st_ino) == stage_identity:
                for name in ("result.yaml", "report.md"):
                    candidate = stage / name
                    if candidate.exists() and candidate.is_file() and not candidate.is_symlink():
                        candidate.unlink()
                if not tuple(stage.iterdir()):
                    stage.rmdir()


def _report(record: dict[str, Any]) -> str:
    header = record["header"]
    rows = "\n".join(
        f"| {item['level']} | {item['dimensions'][0]:,} × {item['dimensions'][1]:,} | {item['downsample']} |"
        for item in header["levels"]
    )
    return f"""# P0001 production acquisition and header-only metadata result

P0001, frozen cohort row 1, passed the authorized CPU-only acquisition, exact
file integrity, exact Omic rematch, and OpenSlide header-only inspection.
No pixel API was called, and no later production patient was started.

| Field | Verified value |
|---|---|
| Production block | `BRCA_PRODUCTION_BLOCK_0001_0008` |
| Cohort row | `1` (`P0001`) |
| Patient | `{PATIENT}` |
| Slide | `{SLIDE}` |
| GDC UUID | `{UUID}` |
| Local path | `{WSI}` |
| Size | {SIZE:,} bytes |
| MD5 | `{MD5}` |
| SHA256 | `{record['identity']['sha256']}` |
| Exact Omic row | {OMIC_ROW} |
| Native MPP | {header['mpp_x']} × {header['mpp_y']} µm/px |

## Pyramid

| Level | Dimensions | OpenSlide downsample |
|---:|---:|---:|
{rows}

This report records geometry only and does not infer or approve a scale or
coordinate policy. The raw SVS remains retained. P0002 through P0008 remain
unstarted. No masks, coordinates, patches, features, HEALNet, CUDA, deletion,
Drive, cohort expansion, or training occurred. Separate review is required
before any P0001 pixel access.
"""


def run(*, expected_source_commit: str, slide_factory: Callable[[str], Any] = openslide.OpenSlide) -> dict[str, Any]:
    started = datetime.now(timezone.utc)
    clock = time.perf_counter()
    source = validate_source(expected_source_commit)
    authorization = validate_authorization()
    manifest_sha256 = validate_manifest()
    before = preflight()
    lock_descriptor, lock_token = _create_lock()
    incoming_identity: tuple[int, int] | None = None
    try:
        incoming_identity = _create_owned_incoming()
        _require_owned_incoming(incoming_identity)
        transfer = download()
        _require_owned_incoming(incoming_identity)
        tree = validate_tree()
        _require_owned_incoming(incoming_identity)
        omic_first = load_omic()
        header, wsi_sha256 = collect_header(slide_factory)
        _require_owned_incoming(incoming_identity)
        omic_second = load_omic()
        require(omic_second == omic_first, "exact Omic rematch changed after header")
        tree_second = validate_tree()
        require(tree_second == tree, "download tree changed during inspection")
        _require_owned_incoming(incoming_identity)
    finally:
        _release_owned_lock(lock_descriptor, lock_token)
    after = _disk(INCOMING.parent)
    require(torch.cuda.is_initialized() is False, "CUDA was initialized")
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": RESULT_STATUS,
        "cohort": "TCGA-BRCA",
        "production_block": "BRCA_PRODUCTION_BLOCK_0001_0008",
        "cohort_index": 1,
        "patient_label": "P0001",
        "frozen_cohort_order_sha256": FROZEN_COHORT_ORDER_SHA256,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": {
            "mode": "CPU_ONLY_ACQUISITION_INTEGRITY_AND_HEADER_ONLY",
            "source_commit": source["source_commit"],
            "total_seconds": time.perf_counter() - clock,
            "authorization_path": authorization["path"],
            "authorization_sha256": authorization["sha256"],
            "authorization_statement_sha256": authorization["statement_sha256"],
            "authorized_manifest_sha256": manifest_sha256,
            "gdc_client_version": before["gdc_client_version"],
            "gdc_client_sha256": CLIENT_SHA256,
            "concurrency": 1,
            "strict_serial_order": True,
            "later_patient_started": False,
            "successful_download_count": 1,
            "download_seconds": transfer["seconds"],
            "download_argv": transfer["argv"],
            "download_timeout_seconds": transfer["timeout_seconds"],
        },
        "identity": {
            "patient_id": PATIENT, "slide_id": SLIDE, "filename": SLIDE,
            "gdc_file_uuid": UUID, "path": str(WSI), "size_bytes": SIZE,
            "md5": MD5, "sha256": wsi_sha256, "regular_non_symlink": True,
            "partial_or_incomplete_files_absent": True,
            "held_o_nofollow_descriptor_used": True,
            "same_descriptor_md5_sha256_rechecked_after_header": True,
        },
        "download_tree": tree,
        "header": header,
        "omic": omic_second,
        "storage": {
            **before,
            "filesystem_after_header": after,
            "available_bytes_delta": after["available_bytes"] - before["filesystem_before_download"]["available_bytes"],
            "raw_wsi_retained": True,
        },
        "source": source,
        "validations": {
            "committed_critical_source_files_head_equal": "PASS",
            "exact_authorization_and_one_row_manifest": "PASS",
            "exact_eight_manifest_block_and_p0001_order": "PASS",
            "fresh_conservative_storage_gate": "PASS",
            "single_gdc_client_concurrency": "PASS",
            "exact_uuid_filename_size_md5_sha256": "PASS",
            "partial_or_incomplete_files_absent": "PASS",
            "regular_non_symlink_svs_and_parcel": "PASS",
            "same_held_descriptor_hash_before_after_header": "PASS",
            "exact_omic_row_rematched": "PASS",
            "finite_positive_complete_openslide_header": "PASS",
        },
        "operations": {
            "read_region_or_pixel_calls": 0, "mask_generations": 0,
            "coordinate_generations": 0, "patch_reads": 0,
            "feature_extractions": 0, "healnet_executions": 0,
            "gpu_or_cuda_operations": 0, "later_patient_operations": 0,
            "deletions_of_user_or_raw_files": 0, "drive_operations": 0,
            "cohort_expansions": 0, "training_runs": 0,
        },
        "required_stop": "P0001_EXACT_FILE_OMIC_AND_HEADER_METADATA_REPORT",
        "required_stop_reached": True,
        "scientific_interpretation": "EXACT_FILE_OMIC_AND_PYRAMID_METADATA_ONLY_NO_PIXELS_OR_MODEL_OUTPUT",
        "next_gate": "USER_REVIEW_REQUIRED_BEFORE_P0001_SCALE_OR_PIXEL_POLICY; P0002_NOT_STARTED",
    }
    yaml_payload = yaml.safe_dump(record, sort_keys=False).encode("utf-8")
    report_payload = _report(record).encode("utf-8")
    require(incoming_identity is not None, "owned P0001 incoming identity was not established")
    _require_owned_incoming(incoming_identity)
    publish_result_set(yaml_payload, report_payload)
    _require_owned_incoming(incoming_identity)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-source-commit", required=True)
    arguments = parser.parse_args()
    record = run(expected_source_commit=arguments.expected_source_commit)
    print(json.dumps({
        "status": record["status"],
        "path": record["identity"]["path"],
        "sha256": record["identity"]["sha256"],
        "header": record["header"],
    }, indent=2))
    print("STOP: P0001 pixels and all downstream operations remain unauthorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
