"""Append-only locked-test start marker and atomic BRCA training result."""

from __future__ import annotations

import ctypes
from dataclasses import asdict
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from typing import Mapping

from .brca_training_checkpoint import TrainingRunIdentity


START_MARKER = "locked_test_started.json"
RESULT_DIRECTORY = "training_result"
RESULT_FILE = "result.json"
RESULT_SIDECAR = "result.json.sha256"
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


class TrainingResultError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrainingResultError(message)


def _canonical(value: Mapping[str, object]) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(payload):
            count = os.write(descriptor, payload[offset:])
            _require(count > 0, "result write made no progress")
            offset += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    function = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    _require(function is not None, "atomic RENAME_NOREPLACE is unavailable")
    result = function(_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(destination), _RENAME_NOREPLACE)
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise TrainingResultError("training result already exists")
        raise OSError(error, os.strerror(error), str(destination))


def _validate_root(root: Path) -> None:
    info = root.lstat()
    _require(stat.S_ISDIR(info.st_mode) and not root.is_symlink(), "result root must be a regular non-symlink directory")


def _read_bounded_regular(path: Path, maximum_bytes: int = 1_000_000) -> bytes:
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not path.is_symlink(), f"regular non-symlink file required: {path.name}")
    _require(0 < before.st_size <= maximum_bytes, f"result file size invalid: {path.name}")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        token = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        _require(token == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns), "result file changed before open")
        chunks: list[bytes] = []
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            _require(bool(chunk), "unexpected result EOF")
            chunks.append(chunk)
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", "result file grew during read")
        final = os.fstat(descriptor)
        after = path.lstat()
        final_token = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
        path_token = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        _require(token == final_token == path_token, "result file identity changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def begin_locked_test(root: str | Path, identity: TrainingRunIdentity) -> str:
    """Record the irreversible decision to consume the locked test exactly once."""

    root = Path(root)
    _validate_root(root)
    _require(not os.path.lexists(root / RESULT_DIRECTORY), "training result already exists")
    document = {
        "schema": "BRCA_LOCKED_TEST_START_V1",
        "identity": asdict(identity),
        "locked_test_evaluation_budget": 1,
        "status": "LOCKED_TEST_EVALUATION_STARTED",
    }
    payload = _canonical(document)
    _write_exclusive(root / START_MARKER, payload)
    _fsync(root)
    return hashlib.sha256(payload).hexdigest()


def validate_start_marker(root: str | Path, identity: TrainingRunIdentity) -> str:
    root = Path(root)
    _validate_root(root)
    path = root / START_MARKER
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and not path.is_symlink() and 0 < info.st_size <= 100_000, "invalid locked-test marker")
    payload = _read_bounded_regular(path, 100_000)
    document = json.loads(payload)
    _require(document == {
        "schema": "BRCA_LOCKED_TEST_START_V1",
        "identity": asdict(identity),
        "locked_test_evaluation_budget": 1,
        "status": "LOCKED_TEST_EVALUATION_STARTED",
    }, "locked-test marker identity or schema drift")
    return hashlib.sha256(payload).hexdigest()


def _validate_summary(summary: Mapping[str, object]) -> None:
    required = {
        "protocol_id", "best_epoch", "best_validation_nll", "locked_test",
        "locked_test_evaluations_this_run", "training_complete",
    }
    _require(set(summary) == required, "training summary keys drift")
    _require(summary["protocol_id"] == "BRCA_HEALNET_IMAGENET1K_V2_SINGLE_SPLIT_V1", "protocol drift")
    _require(1 <= int(summary["best_epoch"]) <= 50, "best epoch drift")
    _require(math.isfinite(float(summary["best_validation_nll"])) and float(summary["best_validation_nll"]) >= 0, "best validation NLL invalid")
    _require(summary["locked_test_evaluations_this_run"] == 1 and summary["training_complete"] is True, "completion/test budget drift")
    locked = summary["locked_test"]
    _require(isinstance(locked, dict) and locked.get("patients") == 118, "locked test count drift")
    for key in ("mean_nll", "harrell_c_index"):
        _require(math.isfinite(float(locked[key])), f"locked test {key} invalid")
    _require(0 <= float(locked["harrell_c_index"]) <= 1, "locked test C-index invalid")
    interval = locked.get("patient_bootstrap_95_percent_ci")
    _require(isinstance(interval, list) and len(interval) == 2 and 0 <= float(interval[0]) <= float(interval[1]) <= 1, "bootstrap interval invalid")
    _require(int(locked.get("bootstrap_valid_replicates", 0)) >= 1900, "insufficient bootstrap replicates")
    _require(locked.get("bootstrap_requested_replicates") == 2000, "bootstrap request count drift")


def publish_training_result(root: str | Path, identity: TrainingRunIdentity, summary: Mapping[str, object]) -> dict[str, object]:
    root = Path(root)
    _validate_root(root)
    marker_sha = validate_start_marker(root, identity)
    _validate_summary(summary)
    final = root / RESULT_DIRECTORY
    stage = root / f".{RESULT_DIRECTORY}.staging-{identity.training_run_id}"
    _require(not os.path.lexists(final) and not os.path.lexists(stage), "result final or staging path already exists")
    os.mkdir(stage, 0o700)
    document = {
        "schema": "BRCA_HEALNET_TRAINING_RESULT_V1",
        "identity": asdict(identity),
        "locked_test_start_marker_sha256": marker_sha,
        "summary": dict(summary),
    }
    payload = _canonical(document)
    digest = hashlib.sha256(payload).hexdigest()
    _write_exclusive(stage / RESULT_FILE, payload)
    _write_exclusive(stage / RESULT_SIDECAR, f"{digest}  {RESULT_FILE}\n".encode("ascii"))
    _fsync(stage)
    _rename_noreplace(stage, final)
    _fsync(root)
    return validate_training_result(root, identity)


def validate_training_result(root: str | Path, identity: TrainingRunIdentity) -> dict[str, object]:
    root = Path(root)
    final = root / RESULT_DIRECTORY
    info = final.lstat()
    _require(stat.S_ISDIR(info.st_mode) and not final.is_symlink(), "training result directory invalid")
    _require({item.name for item in final.iterdir()} == {RESULT_FILE, RESULT_SIDECAR}, "training result exact files drift")
    for name in (RESULT_FILE, RESULT_SIDECAR):
        item = final / name
        item_info = item.lstat()
        _require(stat.S_ISREG(item_info.st_mode) and not item.is_symlink(), "training result file is not regular")
    payload = _read_bounded_regular(final / RESULT_FILE)
    digest = hashlib.sha256(payload).hexdigest()
    _require(_read_bounded_regular(final / RESULT_SIDECAR, 1_000) == f"{digest}  {RESULT_FILE}\n".encode("ascii"), "training result sidecar drift")
    document = json.loads(payload)
    _require(document.get("schema") == "BRCA_HEALNET_TRAINING_RESULT_V1", "training result schema drift")
    _require(document.get("identity") == asdict(identity), "training result identity drift")
    _require(document.get("locked_test_start_marker_sha256") == validate_start_marker(root, identity), "locked-test marker binding drift")
    _validate_summary(document["summary"])
    return document


__all__ = [
    "RESULT_DIRECTORY", "START_MARKER", "TrainingResultError",
    "begin_locked_test", "publish_training_result", "validate_start_marker",
    "validate_training_result",
]
