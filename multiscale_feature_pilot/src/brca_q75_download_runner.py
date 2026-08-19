"""Fail-closed BRCA Q75-only GDC acquisition runner.

The production entry point in this module can download exactly one authorized
public GDC object.  It has no OpenSlide, pixel, patch, feature, HEALNet, CUDA,
Drive, deletion, cohort, or training implementation.  A successful run leaves
the raw SVS in persistent staging and atomically publishes one verification
record before stopping.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import shutil
import stat
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = REPOSITORY_ROOT.parent
DATA_ROOT = WORKSPACE_ROOT / "brca_pilot_data"

EXPECTED_PATIENT_ID = "TCGA-E2-A154"
EXPECTED_SLIDE_ID = (
    "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
)
EXPECTED_FILENAME = EXPECTED_SLIDE_ID
EXPECTED_GDC_UUID = "25aec062-60d1-446e-a1c6-0c79cc74a770"
EXPECTED_SIZE_BYTES = 1_360_743_825
EXPECTED_MD5 = "a8c4b68fb6e0ab3e862efe3ed1fe10d7"

EXPECTED_AUTHORIZATION_CONFIG_SHA256 = (
    "335e6d36aac1c21cc1cd52f8a14e5d2ecfde1f3a6f398d796bf842baaca35979"
)
EXPECTED_AUTHORIZATION_RECORD_SHA256 = (
    "2330c4bc66c73c8e150be2d028aefb2a84916b18e4b5076d95fc28cf869d7050"
)
EXPECTED_MANIFEST_SHA256 = (
    "8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a"
)

EXPECTED_GDC_CLIENT_VERSION = "2.3"
EXPECTED_GDC_CLIENT_SIZE_BYTES = 22_168_960
EXPECTED_GDC_CLIENT_SHA256 = (
    "1df9281cbacbb53413607a4b9b682947dcb48b6ac7fade6527748bc025ae8c96"
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
DEFAULT_MANIFEST = (
    REPOSITORY_ROOT
    / "multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/"
    "Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770."
    "AUTHORIZED_Q75_ONLY.gdc.tsv"
)
DEFAULT_GDC_CLIENT = WORKSPACE_ROOT / "tools/gdc-client/2.3.0/gdc-client"
DEFAULT_INCOMING_DIRECTORY = DATA_ROOT / "Q75.incoming"
DEFAULT_WSI = DEFAULT_INCOMING_DIRECTORY / EXPECTED_GDC_UUID / EXPECTED_FILENAME
DEFAULT_RESULT_DIRECTORY = DATA_ROOT / "Q75.download_result"
DEFAULT_RESULT_FILE = DEFAULT_RESULT_DIRECTORY / "download_result.yaml"
DEFAULT_LOCK_FILE = DATA_ROOT / ".Q75.download.lock"
DEFAULT_STAGING_DIRECTORY = DATA_ROOT / ".Q75.download_result.staging"

SOURCE_RELATIVE_PATHS = (
    Path("multiscale_feature_pilot/src/brca_q75_download_runner.py"),
    Path("scripts/run_brca_q75_download.py"),
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
)

RESULT_STATUS = "BRCA_Q75_GDC_DOWNLOAD_VERIFIED"
REQUIRED_STOP = "Q75_GDC_DOWNLOAD_AND_EXACT_FILE_VERIFICATION_ONLY"
MINIMUM_FREE_BYTES = 2 * EXPECTED_SIZE_BYTES + 512 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 6 * 60 * 60
_CHUNK_SIZE = 8 * 1024 * 1024
_AT_FDCWD = -100
_RENAME_NOREPLACE = 1


class Q75DownloadError(RuntimeError):
    """Raised on every scope, source, transfer, or integrity violation."""


@dataclass(frozen=True)
class DownloadPaths:
    repo_root: Path = REPOSITORY_ROOT
    authorization: Path = DEFAULT_AUTHORIZATION
    authorization_record: Path = DEFAULT_AUTHORIZATION_RECORD
    manifest: Path = DEFAULT_MANIFEST
    gdc_client: Path = DEFAULT_GDC_CLIENT
    data_root: Path = DATA_ROOT
    incoming_directory: Path = DEFAULT_INCOMING_DIRECTORY
    wsi: Path = DEFAULT_WSI
    result_directory: Path = DEFAULT_RESULT_DIRECTORY
    result_file: Path = DEFAULT_RESULT_FILE
    lock_file: Path = DEFAULT_LOCK_FILE
    staging_directory: Path = DEFAULT_STAGING_DIRECTORY


@dataclass(frozen=True)
class SourceBinding:
    commit: str
    head_equal: bool
    files_head_equal: bool
    file_sha256: Mapping[str, str]


@dataclass(frozen=True)
class ClientExpectation:
    version: str
    size_bytes: int
    sha256: str


PRODUCTION_CLIENT_EXPECTATION = ClientExpectation(
    version=EXPECTED_GDC_CLIENT_VERSION,
    size_bytes=EXPECTED_GDC_CLIENT_SIZE_BYTES,
    sha256=EXPECTED_GDC_CLIENT_SHA256,
)


@dataclass(frozen=True)
class FileToken:
    device: int
    inode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, details: os.stat_result) -> "FileToken":
        return cls(
            device=details.st_dev,
            inode=details.st_ino,
            size=details.st_size,
            mtime_ns=details.st_mtime_ns,
            ctime_ns=details.st_ctime_ns,
        )


ProcessRunner = Callable[..., subprocess.CompletedProcess[str]]
DiskSnapshotter = Callable[[Path], Mapping[str, Any]]
ProcessScanner = Callable[[Path], Sequence[int]]
RenameNoReplace = Callable[[Path, Path], None]
TreeValidator = Callable[[DownloadPaths], Mapping[str, Any]]
WsiVerifier = Callable[[DownloadPaths], Mapping[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q75DownloadError(message)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _lexists(path: Path) -> bool:
    return os.path.lexists(os.fspath(path))


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_no_symlink_components(path: Path, *, include_leaf: bool) -> None:
    _require(path.is_absolute(), f"path must be absolute: {path}")
    parts = path.parts
    stop = len(parts) if include_leaf else len(parts) - 1
    current = Path(parts[0])
    for part in parts[1:stop]:
        current /= part
        try:
            details = current.lstat()
        except FileNotFoundError as exc:
            raise Q75DownloadError(f"missing path component: {current}") from exc
        _require(not stat.S_ISLNK(details.st_mode), f"symlink component prohibited: {current}")


def _read_regular_bytes(path: Path, *, label: str, maximum: int) -> bytes:
    _require_no_symlink_components(path, include_leaf=False)
    try:
        before = path.lstat()
    except FileNotFoundError as exc:
        raise Q75DownloadError(f"missing {label}: {path}") from exc
    _require(not stat.S_ISLNK(before.st_mode), f"{label} must not be a symlink")
    _require(stat.S_ISREG(before.st_mode), f"{label} must be a regular file")
    _require(before.st_size <= maximum, f"{label} exceeds its size limit")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Q75DownloadError(f"cannot securely open {label}: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        _require(FileToken.from_stat(opened) == FileToken.from_stat(before), f"{label} identity changed")
        chunks: list[bytes] = []
        remaining = maximum + 1
        while remaining:
            chunk = os.read(descriptor, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(remaining > 0, f"{label} exceeds its size limit")
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(FileToken.from_stat(after) == FileToken.from_stat(before), f"{label} changed while read")
    _require(FileToken.from_stat(path.lstat()) == FileToken.from_stat(before), f"{label} pathname changed")
    return b"".join(chunks)


def _yaml_mapping(payload: bytes, *, label: str) -> Mapping[str, Any]:
    try:
        document = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeError, yaml.YAMLError) as exc:
        raise Q75DownloadError(f"cannot parse {label}") from exc
    _require(isinstance(document, Mapping), f"{label} root must be a mapping")
    return document


def _validate_authorized_inputs(paths: DownloadPaths) -> dict[str, str]:
    authorization = _read_regular_bytes(
        paths.authorization, label="Q75 authorization", maximum=2 * 1024 * 1024
    )
    authorization_sha = _sha256_bytes(authorization)
    _require(
        authorization_sha == EXPECTED_AUTHORIZATION_CONFIG_SHA256,
        "Q75 authorization config SHA256 mismatch",
    )
    authorization_document = _yaml_mapping(authorization, label="Q75 authorization")
    _require(
        authorization_document.get("status")
        == "AUTHORIZED_Q75_ACQUISITION_AND_HEADER_METADATA_ONLY",
        "Q75 acquisition is not authorized",
    )
    _require(authorization_document.get("candidate") == "Q75", "authorization candidate drift")
    _require(
        authorization_document.get("execution_mode") == "CPU_LOGIC_ONLY_CUDA_NOT_REQUIRED",
        "authorization CPU-only directive drift",
    )
    selected = authorization_document.get("authorized_q75")
    _require(isinstance(selected, Mapping), "authorized Q75 identity is missing")
    for key, expected in (
        ("patient_id", EXPECTED_PATIENT_ID),
        ("wsi_uuid", EXPECTED_GDC_UUID),
        ("filename", EXPECTED_FILENAME),
        ("declared_bytes", EXPECTED_SIZE_BYTES),
        ("md5", EXPECTED_MD5),
    ):
        _require(selected.get(key) == expected, f"authorized Q75 {key} drift")

    record = _read_regular_bytes(
        paths.authorization_record,
        label="Q75 authorization record",
        maximum=1024 * 1024,
    )
    record_sha = _sha256_bytes(record)
    _require(
        record_sha == EXPECTED_AUTHORIZATION_RECORD_SHA256,
        "Q75 authorization-record SHA256 mismatch",
    )
    record_document = _yaml_mapping(record, label="Q75 authorization record")
    _require(record_document.get("status") == "AUTHORIZED_Q75_ONLY", "Q75 record is not authorized")
    _require(record_document.get("download_authorized") is True, "Q75 download is locked")
    _require(record_document.get("manifest_count") == 1, "Q75 authorization record is not one-row only")
    _require(record_document.get("pass_n_processes_1") is False, "invalid GDC -n 1 was enabled")
    binding = record_document.get("approval_binding")
    _require(isinstance(binding, Mapping), "authorization-record binding is missing")
    _require(
        binding.get("authorization_config_sha256") == authorization_sha,
        "authorization-record config binding mismatch",
    )

    manifest = _read_regular_bytes(paths.manifest, label="Q75 manifest", maximum=16 * 1024)
    expected_manifest = (
        "id\tfilename\tmd5\tsize\tstate\n"
        f"{EXPECTED_GDC_UUID}\t{EXPECTED_FILENAME}\t{EXPECTED_MD5}\t"
        f"{EXPECTED_SIZE_BYTES}\treleased\n"
    ).encode("utf-8")
    _require(manifest == expected_manifest, "Q75 manifest is not the exact authorized one-row payload")
    manifest_sha = _sha256_bytes(manifest)
    _require(manifest_sha == EXPECTED_MANIFEST_SHA256, "Q75 manifest SHA256 mismatch")
    return {
        "authorization_config_path": str(paths.authorization),
        "authorization_config_sha256": authorization_sha,
        "authorization_record_path": str(paths.authorization_record),
        "authorization_record_sha256": record_sha,
        "manifest_path": str(paths.manifest),
        "manifest_sha256": manifest_sha,
    }


def _git_run(repo: Path, arguments: Sequence[str], *, binary: bool = False) -> subprocess.CompletedProcess[Any]:
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=False,
        capture_output=True,
        text=not binary,
        timeout=30,
    )


def verify_source_binding(
    repo: Path,
    expected_commit: str,
    *,
    relative_paths: Sequence[Path] = SOURCE_RELATIVE_PATHS,
) -> SourceBinding:
    """Require the executing sources and approvals to equal the explicit HEAD."""

    _require(
        isinstance(expected_commit, str)
        and len(expected_commit) == 40
        and all(character in "0123456789abcdef" for character in expected_commit),
        "expected source commit must be a full lowercase Git commit",
    )
    _require_no_symlink_components(repo, include_leaf=True)
    _require(repo.is_dir() and not repo.is_symlink(), "repository root is unsafe")
    head_result = _git_run(repo, ["rev-parse", "--verify", "HEAD"])
    _require(head_result.returncode == 0, "cannot resolve repository HEAD")
    head = head_result.stdout.strip()
    _require(head == expected_commit, "repository HEAD does not equal expected source commit")

    digests: dict[str, str] = {}
    for relative in relative_paths:
        _require(not relative.is_absolute() and ".." not in relative.parts, "unsafe source path")
        path = repo / relative
        worktree = _read_regular_bytes(path, label=f"execution source {relative}", maximum=4 * 1024 * 1024)
        committed = _git_run(repo, ["show", f"{expected_commit}:{relative.as_posix()}"], binary=True)
        _require(committed.returncode == 0, f"execution source is not tracked at HEAD: {relative}")
        _require(committed.stdout == worktree, f"execution source is not HEAD-equal: {relative}")
        digests[relative.as_posix()] = _sha256_bytes(worktree)
    return SourceBinding(
        commit=head,
        head_equal=True,
        files_head_equal=True,
        file_sha256=digests,
    )


def _hash_regular_file(path: Path, *, include_md5: bool, independent_sha256: bool) -> dict[str, Any]:
    _require_no_symlink_components(path, include_leaf=False)
    before_path = path.lstat()
    _require(not stat.S_ISLNK(before_path.st_mode), f"file must not be a symlink: {path}")
    _require(stat.S_ISREG(before_path.st_mode), f"file must be regular: {path}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise Q75DownloadError(f"cannot securely open file: {path}") from exc
    token = FileToken.from_stat(before_path)
    try:
        _require(FileToken.from_stat(os.fstat(descriptor)) == token, f"file identity changed: {path}")
        md5 = hashlib.md5(usedforsecurity=False) if include_md5 else None
        sha_first = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, _CHUNK_SIZE)
            if not chunk:
                break
            if md5 is not None:
                md5.update(chunk)
            sha_first.update(chunk)
        sha_second: Any | None = None
        if independent_sha256:
            os.lseek(descriptor, 0, os.SEEK_SET)
            sha_second = hashlib.sha256()
            while True:
                chunk = os.read(descriptor, _CHUNK_SIZE)
                if not chunk:
                    break
                sha_second.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    _require(FileToken.from_stat(after) == token, f"file changed while hashing: {path}")
    _require(FileToken.from_stat(path.lstat()) == token, f"file pathname changed while hashing: {path}")
    first = sha_first.hexdigest()
    second = sha_second.hexdigest() if sha_second is not None else None
    if independent_sha256:
        _require(first == second, f"independent SHA256 mismatch: {path}")
    return {
        "path": str(path),
        "size_bytes": token.size,
        "md5": md5.hexdigest() if md5 is not None else None,
        "sha256": first,
        "sha256_independent_recheck": second,
        "sha256_independent_recheck_matches": second == first if independent_sha256 else None,
        "regular_non_symlink": True,
    }


def _verify_gdc_client(
    path: Path,
    expectation: ClientExpectation,
    *,
    process_runner: ProcessRunner,
) -> dict[str, Any]:
    _require(_valid_sha256(expectation.sha256), "GDC client SHA256 expectation is invalid")
    details = _hash_regular_file(path, include_md5=False, independent_sha256=False)
    _require(details["size_bytes"] == expectation.size_bytes, "GDC client size mismatch")
    _require(details["sha256"] == expectation.sha256, "GDC client SHA256 mismatch")
    _require(os.access(path, os.X_OK), "GDC client is not executable")
    command = [str(path), "--version"]
    try:
        result = process_runner(
            command,
            check=False,
            capture_output=True,
            text=True,
            shell=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise Q75DownloadError("GDC client version probe failed") from exc
    _require(tuple(result.args) == tuple(command), "GDC client version command drift")
    _require(result.returncode == 0, "GDC client version probe returned nonzero")
    version = result.stdout.strip()
    _require(version == expectation.version, "GDC client version mismatch")
    return {
        "path": str(path),
        "version_output": version,
        "size_bytes": details["size_bytes"],
        "sha256": details["sha256"],
        "regular_non_symlink": True,
        "executable": True,
    }


def _reverify_gdc_client_file(
    path: Path,
    expected_record: Mapping[str, Any],
) -> None:
    """Rebind the client pathname without starting another client process."""

    details = _hash_regular_file(path, include_md5=False, independent_sha256=False)
    _require(details["size_bytes"] == expected_record.get("size_bytes"), "GDC client size changed")
    _require(details["sha256"] == expected_record.get("sha256"), "GDC client SHA256 changed")
    _require(os.access(path, os.X_OK), "GDC client executable permission changed")


def _disk_snapshot(path: Path) -> Mapping[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "measurement_path": str(path),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def _scan_active_gdc_processes(client: Path) -> Sequence[int]:
    """Best-effort exact-argv /proc scan; unreadable process entries fail closed."""

    matches: list[int] = []
    proc = Path("/proc")
    _require(proc.is_dir(), "/proc is unavailable for GDC process isolation")
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        command_line = entry / "cmdline"
        try:
            payload = command_line.read_bytes()
        except FileNotFoundError:
            continue
        except PermissionError as exc:
            raise Q75DownloadError(f"cannot audit process {entry.name}") from exc
        if not payload:
            continue
        argv0 = payload.split(b"\0", 1)[0]
        try:
            decoded = os.fsdecode(argv0)
        except Exception as exc:
            raise Q75DownloadError(f"cannot decode process {entry.name} argv") from exc
        if decoded == str(client):
            matches.append(int(entry.name))
    return tuple(sorted(matches))


def _validate_production_paths(paths: DownloadPaths) -> None:
    expected = DownloadPaths()
    for field in (
        "repo_root",
        "authorization",
        "authorization_record",
        "manifest",
        "gdc_client",
        "data_root",
        "incoming_directory",
        "wsi",
        "result_directory",
        "result_file",
        "lock_file",
        "staging_directory",
    ):
        _require(getattr(paths, field) == getattr(expected, field), f"production {field} drift")


def _validate_absent_preconditions(paths: DownloadPaths) -> None:
    _require_no_symlink_components(paths.data_root, include_leaf=True)
    root = paths.data_root.lstat()
    _require(stat.S_ISDIR(root.st_mode) and not stat.S_ISLNK(root.st_mode), "Q75 data root is unsafe")
    _require(paths.incoming_directory.parent == paths.data_root, "incoming parent drift")
    _require(paths.result_directory.parent == paths.data_root, "result parent drift")
    _require(paths.result_file == paths.result_directory / "download_result.yaml", "result filename drift")
    _require(paths.lock_file == paths.data_root / ".Q75.download.lock", "download lock path drift")
    _require(
        paths.staging_directory == paths.data_root / ".Q75.download_result.staging",
        "result staging path drift",
    )
    for path, label in (
        (paths.incoming_directory, "Q75 incoming destination"),
        (paths.result_directory, "Q75 download result"),
        (paths.staging_directory, "Q75 result staging"),
        (paths.lock_file, "Q75 download lock"),
    ):
        _require(not _lexists(path), f"{label} must be absent before execution: {path}")


def _acquire_lock(path: Path) -> tuple[int, FileToken]:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError as exc:
        raise Q75DownloadError(f"Q75 download lock already exists: {path}") from exc
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        token = FileToken.from_stat(os.fstat(descriptor))
        _fsync_directory(path.parent)
    except Exception:
        os.close(descriptor)
        raise
    return descriptor, token


def _release_lock(path: Path, descriptor: int, token: FileToken) -> None:
    try:
        os.close(descriptor)
    finally:
        try:
            current = path.lstat()
        except FileNotFoundError:
            return
        if FileToken.from_stat(current) == token and stat.S_ISREG(current.st_mode):
            path.unlink()
            _fsync_directory(path.parent)


def _validate_exact_download_tree(paths: DownloadPaths) -> dict[str, Any]:
    incoming = paths.incoming_directory
    _require_no_symlink_components(incoming, include_leaf=True)
    _require(stat.S_ISDIR(incoming.lstat().st_mode), "Q75 incoming must be a directory")
    uuid_directory = incoming / EXPECTED_GDC_UUID
    logs = uuid_directory / "logs"
    parcel = logs / f"{EXPECTED_FILENAME}.parcel"
    _require(paths.wsi == uuid_directory / EXPECTED_FILENAME, "Q75 WSI path drift")

    for directory, label in (
        (uuid_directory, "Q75 UUID directory"),
        (logs, "Q75 logs directory"),
    ):
        _require_no_symlink_components(directory, include_leaf=True)
        details = directory.lstat()
        _require(stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode), f"{label} is unsafe")

    incoming_entries = tuple(sorted(item.name for item in incoming.iterdir()))
    uuid_entries = tuple(sorted(item.name for item in uuid_directory.iterdir()))
    log_entries = tuple(sorted(item.name for item in logs.iterdir()))
    _require(incoming_entries == (EXPECTED_GDC_UUID,), "partial or unexpected Q75 incoming entry")
    _require(uuid_entries == (EXPECTED_FILENAME, "logs"), "partial or unexpected Q75 UUID entry")
    _require(log_entries == (parcel.name,), "partial or unexpected Q75 GDC log entry")

    parcel_hash = _hash_regular_file(parcel, include_md5=False, independent_sha256=False)
    _require(parcel_hash["size_bytes"] > 0, "Q75 GDC parcel is empty")
    relative_entries = [
        EXPECTED_GDC_UUID,
        f"{EXPECTED_GDC_UUID}/{EXPECTED_FILENAME}",
        f"{EXPECTED_GDC_UUID}/logs",
        f"{EXPECTED_GDC_UUID}/logs/{parcel.name}",
    ]
    return {
        "incoming_directory": str(incoming),
        "uuid_directory": str(uuid_directory),
        "relative_entries": relative_entries,
        "entry_types": {
            relative_entries[0]: "directory",
            relative_entries[1]: "regular_file",
            relative_entries[2]: "directory",
            relative_entries[3]: "regular_file",
        },
        "exact_entry_count_excluding_root": len(relative_entries),
        "partial_or_unexpected_entries": [],
        "completed_tree_exact": True,
        "parcel": {
            "path": relative_entries[3],
            "absolute_path": parcel_hash["path"],
            "size_bytes": parcel_hash["size_bytes"],
            "sha256": parcel_hash["sha256"],
            "regular_non_symlink": True,
        },
    }


def _verify_exact_wsi(paths: DownloadPaths) -> dict[str, Any]:
    verified = _hash_regular_file(paths.wsi, include_md5=True, independent_sha256=True)
    _require(verified["size_bytes"] == EXPECTED_SIZE_BYTES, "Q75 WSI exact byte-size mismatch")
    _require(verified["md5"] == EXPECTED_MD5, "Q75 WSI MD5 mismatch")
    _require(paths.wsi.name == EXPECTED_FILENAME, "Q75 WSI filename mismatch")
    _require(paths.wsi.suffix == ".svs", "Q75 WSI suffix mismatch")
    verified.update(
        {
            "gdc_uuid": EXPECTED_GDC_UUID,
            "filename": EXPECTED_FILENAME,
            "expected_size_bytes": EXPECTED_SIZE_BYTES,
            "expected_md5": EXPECTED_MD5,
            "exact_uuid_and_filename": True,
            "exact_size_matches": True,
            "md5_matches": True,
            "exact_svs_suffix": True,
        }
    )
    return verified


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace_linux(source: Path, destination: Path) -> None:
    """Atomically rename a directory and refuse an existing destination."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    _require(renameat2 is not None, "renameat2 is unavailable; atomic no-overwrite publication refused")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise Q75DownloadError(f"append-only result already exists: {destination}")
        raise Q75DownloadError(
            f"atomic result-directory publication failed: {os.strerror(error_number)}"
        )


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        _require(written > 0, "short write while staging Q75 result")
        offset += written


def _cleanup_private_staging(
    stage: Path,
    result_name: str,
    *,
    expected_stage_identity: tuple[int, int],
    expected_file_identity: tuple[int, int] | None,
) -> None:
    if not _lexists(stage):
        return
    details = stage.lstat()
    if (
        not stat.S_ISDIR(details.st_mode)
        or stat.S_ISLNK(details.st_mode)
        or (details.st_dev, details.st_ino) != expected_stage_identity
    ):
        return
    entries = tuple(stage.iterdir())
    if len(entries) == 1 and entries[0].name == result_name:
        entry = entries[0].lstat()
        if (
            expected_file_identity is not None
            and stat.S_ISREG(entry.st_mode)
            and not stat.S_ISLNK(entry.st_mode)
            and (entry.st_dev, entry.st_ino) == expected_file_identity
        ):
            entries[0].unlink()
            stage.rmdir()
    elif not entries:
        stage.rmdir()


def _publish_result_directory(
    paths: DownloadPaths,
    record: Mapping[str, Any],
    *,
    rename_noreplace: RenameNoReplace,
) -> str:
    _require(not _lexists(paths.result_directory), "Q75 result directory already exists")
    _require(not _lexists(paths.staging_directory), "Q75 result staging already exists")
    payload = yaml.safe_dump(dict(record), sort_keys=False).encode("utf-8")
    digest = _sha256_bytes(payload)
    try:
        os.mkdir(paths.staging_directory, 0o700)
    except FileExistsError as exc:
        raise Q75DownloadError(
            f"Q75 result staging appeared before publication: {paths.staging_directory}"
        ) from exc
    stage_details = paths.staging_directory.lstat()
    stage_identity = (stage_details.st_dev, stage_details.st_ino)
    staged_file = paths.staging_directory / paths.result_file.name
    renamed = False
    staged_file_identity: tuple[int, int] | None = None
    try:
        _fsync_directory(paths.data_root)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(staged_file, flags, 0o600)
        try:
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        staged_file_details = staged_file.lstat()
        staged_file_identity = (
            staged_file_details.st_dev,
            staged_file_details.st_ino,
        )
        _fsync_directory(paths.staging_directory)
        staged_payload = _read_regular_bytes(
            staged_file, label="staged Q75 download result", maximum=4 * 1024 * 1024
        )
        _require(staged_payload == payload, "staged Q75 result payload mismatch")
        staged_document = _yaml_mapping(staged_payload, label="staged Q75 download result")
        _require(staged_document.get("status") == RESULT_STATUS, "staged Q75 result status mismatch")
        _require(tuple(path.name for path in paths.staging_directory.iterdir()) == (paths.result_file.name,), "staged Q75 result is not a sole file")
        stage_details = paths.staging_directory.lstat()
        _require(
            (stage_details.st_dev, stage_details.st_ino) == stage_identity,
            "Q75 result staging directory identity changed",
        )
        rename_noreplace(paths.staging_directory, paths.result_directory)
        renamed = True
        _fsync_directory(paths.data_root)
        published = paths.result_directory.lstat()
        _require(
            (published.st_dev, published.st_ino) == stage_identity,
            "published Q75 result directory identity mismatch",
        )
        _require(stat.S_ISDIR(published.st_mode) and not stat.S_ISLNK(published.st_mode), "published Q75 result directory is unsafe")
        _require(tuple(path.name for path in paths.result_directory.iterdir()) == (paths.result_file.name,), "published Q75 result must contain one file")
        published_file_details = paths.result_file.lstat()
        _require(
            (published_file_details.st_dev, published_file_details.st_ino)
            == staged_file_identity,
            "published Q75 result file identity mismatch",
        )
        published_payload = _read_regular_bytes(
            paths.result_file, label="published Q75 download result", maximum=4 * 1024 * 1024
        )
        _require(published_payload == payload, "published Q75 result payload mismatch")
        _require(_sha256_bytes(published_payload) == digest, "published Q75 result SHA256 mismatch")
        return digest
    finally:
        if not renamed:
            _cleanup_private_staging(
                paths.staging_directory,
                paths.result_file.name,
                expected_stage_identity=stage_identity,
                expected_file_identity=staged_file_identity,
            )


def _exact_download_command(paths: DownloadPaths) -> list[str]:
    command = [
        str(paths.gdc_client),
        "download",
        "-m",
        str(paths.manifest),
        "--dir",
        str(paths.incoming_directory),
        "--no-related-files",
        "--no-annotations",
    ]
    forbidden = {"-n", "--n-processes", "--no-verify", "--token-file"}
    _require(not forbidden.intersection(command), "GDC command contains a forbidden or weakening flag")
    _require(command.count(str(paths.manifest)) == 1, "GDC command manifest count drift")
    _require(command.count(str(paths.incoming_directory)) == 1, "GDC destination count drift")
    return command


def _run_download_gate(
    *,
    paths: DownloadPaths,
    source_binding: SourceBinding,
    client_expectation: ClientExpectation,
    process_runner: ProcessRunner,
    disk_snapshotter: DiskSnapshotter,
    process_scanner: ProcessScanner,
    rename_noreplace: RenameNoReplace,
    tree_validator: TreeValidator = _validate_exact_download_tree,
    wsi_verifier: WsiVerifier = _verify_exact_wsi,
) -> dict[str, Any]:
    """Testable core. Production callers must use :func:`run_q75_download`."""

    started = datetime.now(timezone.utc)
    start_clock = time.perf_counter()
    _validate_absent_preconditions(paths)
    lock_descriptor, lock_token = _acquire_lock(paths.lock_file)
    try:
        for path, label in (
            (paths.incoming_directory, "Q75 incoming destination"),
            (paths.result_directory, "Q75 download result"),
            (paths.staging_directory, "Q75 result staging"),
        ):
            _require(not _lexists(path), f"{label} appeared after lock acquisition")

        bindings = _validate_authorized_inputs(paths)
        _require(
            len(source_binding.commit) == 40
            and all(character in "0123456789abcdef" for character in source_binding.commit),
            "execution source commit is not a full lowercase Git commit",
        )
        _require(source_binding.head_equal, "execution source commit is not HEAD-equal")
        _require(source_binding.files_head_equal, "execution source files are not HEAD-equal")
        _require(len(source_binding.file_sha256) >= 2, "execution source-file binding is incomplete")
        client = _verify_gdc_client(
            paths.gdc_client, client_expectation, process_runner=process_runner
        )
        before = dict(disk_snapshotter(paths.data_root))
        free_before = before.get("free_bytes")
        _require(isinstance(free_before, int) and free_before >= MINIMUM_FREE_BYTES, "insufficient free storage for exact Q75 acquisition")

        active_before = tuple(process_scanner(paths.gdc_client))
        _require(not active_before, "another GDC client process is active")
        _require(
            _validate_authorized_inputs(paths) == bindings,
            "authorized inputs changed immediately before GDC execution",
        )
        _reverify_gdc_client_file(paths.gdc_client, client)
        command = _exact_download_command(paths)
        command_started = datetime.now(timezone.utc)
        command_clock = time.perf_counter()
        try:
            completed = process_runner(
                command,
                check=False,
                capture_output=True,
                text=True,
                shell=False,
                timeout=DOWNLOAD_TIMEOUT_SECONDS,
                cwd=str(paths.data_root),
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise Q75DownloadError("exact Q75 GDC subprocess failed") from exc
        command_finished = datetime.now(timezone.utc)
        command_seconds = time.perf_counter() - command_clock
        _require(tuple(completed.args) == tuple(command), "executed GDC argv drift")
        _require(isinstance(completed.stdout, str), "GDC stdout was not captured as text")
        _require(isinstance(completed.stderr, str), "GDC stderr was not captured as text")
        _require(completed.returncode == 0, f"GDC client returned nonzero: {completed.returncode}")
        active_after = tuple(process_scanner(paths.gdc_client))
        _require(not active_after, "GDC client process remained active after subprocess return")
        _require(
            _validate_authorized_inputs(paths) == bindings,
            "authorized inputs changed during GDC execution",
        )
        _reverify_gdc_client_file(paths.gdc_client, client)

        tree = dict(tree_validator(paths))
        wsi = dict(wsi_verifier(paths))
        tree_recheck = dict(tree_validator(paths))
        _require(tree_recheck == tree, "Q75 completed tree changed during verification")
        after = dict(disk_snapshotter(paths.data_root))
        used_before = before.get("used_bytes")
        used_after = after.get("used_bytes")
        free_after = after.get("free_bytes")
        for value, label in (
            (used_before, "used bytes before"),
            (used_after, "used bytes after"),
            (free_after, "free bytes after"),
        ):
            _require(isinstance(value, int) and value >= 0, f"invalid storage snapshot: {label}")
        _require(
            _validate_authorized_inputs(paths) == bindings,
            "authorized inputs changed before result publication",
        )
        _reverify_gdc_client_file(paths.gdc_client, client)
        final_wsi = dict(wsi_verifier(paths))
        _require(final_wsi == wsi, "Q75 WSI changed before result publication")
        final_tree = dict(tree_validator(paths))
        _require(final_tree == tree, "Q75 completed tree changed before result publication")

        finished = datetime.now(timezone.utc)
        record: dict[str, Any] = {
            "schema_version": 1,
            "status": RESULT_STATUS,
            "cohort": "TCGA-BRCA",
            "candidate": "Q75",
            "patient_id": EXPECTED_PATIENT_ID,
            "slide_id": EXPECTED_SLIDE_ID,
            "gdc_file_uuid": EXPECTED_GDC_UUID,
            "execution": {
                "mode": "CPU_LOGIC_ONLY_NO_OPENSLIDE",
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "total_seconds": time.perf_counter() - start_clock,
                "source_commit": source_binding.commit,
                "source_head_equal": True,
                "source_files_head_equal": True,
                "source_file_sha256": dict(source_binding.file_sha256),
                "cuda_used": False,
                "gpu_work_performed": False,
            },
            "bindings": bindings,
            "gdc_client": client,
            "command": {
                "argv": command,
                "shell": False,
                "cwd": str(paths.data_root),
                "timeout_seconds": DOWNLOAD_TIMEOUT_SECONDS,
                "started_at_utc": command_started.isoformat(),
                "finished_at_utc": command_finished.isoformat(),
                "total_seconds": command_seconds,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "manifest_rows": 1,
                "client_version_probe_processes": 1,
                "gdc_client_processes_started": 1,
                "download_client_processes_started": 1,
                "one_download_subprocess_invoked": True,
                "cooperating_runner_lock_held": True,
                "continuous_systemwide_process_monitoring_performed": False,
                "process_scan_scope": "EXACT_APPROVED_CLIENT_ARGV0_BEFORE_AND_AFTER",
                "active_gdc_processes_before": len(active_before),
                "active_gdc_processes_after": len(active_after),
                "internal_transfer_processes": "CLIENT_SUPPORTED_DEFAULT_MINIMUM_3",
                "n_processes_1_passed": False,
                "weakening_flags_passed": False,
            },
            "storage": {
                "minimum_free_bytes_required_before_download": MINIMUM_FREE_BYTES,
                "filesystem_before_download": before,
                "filesystem_after_download": after,
                "used_bytes_delta": used_after - used_before,
                "free_bytes_delta": free_after - free_before,
                "raw_wsi_retained": True,
            },
            "download_tree": tree,
            "wsi": wsi,
            "validations": {
                "authorization_config_exact_sha256": "PASS",
                "authorization_record_exact_sha256": "PASS",
                "one_row_manifest_exact_sha256_and_content": "PASS",
                "source_commit_head_equal": "PASS",
                "execution_sources_head_equal": "PASS",
                "gdc_client_path_version_size_sha256": "PASS",
                "authorized_inputs_revalidated_before_after_and_before_publication": "PASS",
                "gdc_client_file_revalidated_before_after_and_before_publication": "PASS",
                "one_active_gdc_client_process": "PASS",
                "exact_argv_no_weakening_flags": "PASS",
                "subprocess_exit_zero": "PASS",
                "completed_tree_exact_no_partial_or_extras": "PASS",
                "regular_non_symlink_svs": "PASS",
                "exact_uuid_filename_size_md5": "PASS",
                "independent_sha256_recheck": "PASS",
                "final_wsi_and_tree_reverification_before_publication": "PASS",
            },
            "operations": {
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
            },
            "publication": {
                "result_directory": str(paths.result_directory),
                "result_file": str(paths.result_file),
                "atomic_directory_rename_noreplace": True,
                "staging_directory": str(paths.staging_directory),
                "append_only_no_overwrite": True,
                "sole_file": True,
                "other_files_created_or_modified_outside_download_tree_and_result": [],
            },
            "required_stop": REQUIRED_STOP,
            "required_stop_reached": True,
            "next_gate": "Q75_EXACT_OMIC_REMATCH_AND_OPENSLIDE_HEADER_ONLY_AUTHORIZED_NEXT",
        }
        _publish_result_directory(
            paths, record, rename_noreplace=rename_noreplace
        )
        return record
    finally:
        _release_lock(paths.lock_file, lock_descriptor, lock_token)


def run_q75_download(
    *,
    expected_source_commit: str,
    paths: DownloadPaths = DownloadPaths(),
) -> dict[str, Any]:
    """Run the exact production Q75 acquisition and stop before OpenSlide."""

    _validate_production_paths(paths)
    _validate_absent_preconditions(paths)
    source_binding = verify_source_binding(paths.repo_root, expected_source_commit)
    return _run_download_gate(
        paths=paths,
        source_binding=source_binding,
        client_expectation=PRODUCTION_CLIENT_EXPECTATION,
        process_runner=subprocess.run,
        disk_snapshotter=_disk_snapshot,
        process_scanner=_scan_active_gdc_processes,
        rename_noreplace=_rename_noreplace_linux,
    )


__all__ = [
    "DEFAULT_INCOMING_DIRECTORY",
    "DEFAULT_RESULT_DIRECTORY",
    "DEFAULT_RESULT_FILE",
    "DownloadPaths",
    "EXPECTED_FILENAME",
    "EXPECTED_GDC_UUID",
    "EXPECTED_MD5",
    "EXPECTED_PATIENT_ID",
    "EXPECTED_SIZE_BYTES",
    "EXPECTED_SLIDE_ID",
    "Q75DownloadError",
    "RESULT_STATUS",
    "run_q75_download",
    "verify_source_binding",
]
