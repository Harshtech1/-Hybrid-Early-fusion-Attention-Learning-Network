"""Append-only, hash-bound checkpoint publication and recovery for BRCA.

The module treats serialized model/optimizer/scheduler/RNG states as opaque
bytes.  It imports no ML framework and performs no model or GPU operation.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Mapping
import uuid


SCHEMA = "BRCA_HEALNET_CHECKPOINT_V1"
PAYLOAD_FILES = (
    "model_state.pt",
    "optimizer_state.pt",
    "scheduler_state.pt",
    "rng_state.pt",
    "epoch_history.json",
)
MANIFEST = "checkpoint_manifest.json"
SIDECAR = "checkpoint_manifest.json.sha256"
EXACT_FILES = frozenset((*PAYLOAD_FILES, MANIFEST, SIDECAR))
MAX_SMALL_FILE_BYTES = 16_000_000
MAX_STATE_FILE_BYTES = 2_000_000_000
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_FINAL = re.compile(r"^checkpoint-epoch-(\d{4})$")
_STAGING = re.compile(r"^\.checkpoint-epoch-(\d{4})\.staging-[0-9a-f-]{36}$")
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


class CheckpointError(RuntimeError):
    """Raised for checkpoint drift, collision or ambiguous recovery state."""


@dataclass(frozen=True)
class TrainingRunIdentity:
    training_run_id: str
    source_commit: str
    authorization_sha256: str
    split_manifest_sha256: str
    cutpoints_sha256: str
    feature_registry_sha256: str
    training_policy_sha256: str
    official_healnet_commit: str


@dataclass(frozen=True)
class EpochState:
    epoch: int
    optimizer_steps: int
    best_epoch: int
    best_validation_nll: float
    epochs_without_improvement: int
    early_stop_reached: bool


@dataclass(frozen=True)
class ValidatedCheckpoint:
    directory: Path
    identity: TrainingRunIdentity
    state: EpochState
    manifest_sha256: str
    payload_hashes: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class RecoveryPlan:
    action: str
    latest: ValidatedCheckpoint | None
    next_epoch: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CheckpointError(message)


def _validate_sha(value: str, label: str) -> None:
    _require(isinstance(value, str) and _SHA256.fullmatch(value) is not None, f"{label} must be lowercase SHA256")


def _validate_identity(identity: TrainingRunIdentity) -> None:
    try:
        parsed = uuid.UUID(identity.training_run_id)
    except ValueError as error:
        raise CheckpointError("training_run_id must be a canonical UUID") from error
    _require(str(parsed) == identity.training_run_id, "training_run_id must be a canonical lowercase UUID")
    _require(_COMMIT.fullmatch(identity.source_commit) is not None, "source_commit must be a full lowercase Git SHA")
    _require(_COMMIT.fullmatch(identity.official_healnet_commit) is not None, "official_healnet_commit must be a full lowercase Git SHA")
    for label in (
        "authorization_sha256", "split_manifest_sha256", "cutpoints_sha256",
        "feature_registry_sha256", "training_policy_sha256",
    ):
        _validate_sha(getattr(identity, label), label)


def _validate_state(state: EpochState) -> None:
    _require(1 <= state.epoch <= 50, "checkpoint epoch must be in [1,50]")
    _require(state.optimizer_steps >= 1, "optimizer_steps must be positive")
    _require(1 <= state.best_epoch <= state.epoch, "best_epoch must be within completed epochs")
    _require(math.isfinite(state.best_validation_nll) and state.best_validation_nll >= 0, "best validation NLL must be finite and nonnegative")
    _require(0 <= state.epochs_without_improvement <= state.epoch, "invalid early-stopping counter")


def _canonical_json(document: Mapping[str, object]) -> bytes:
    return (json.dumps(document, sort_keys=True, indent=2) + "\n").encode("utf-8")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _no_symlink_ancestors(path: Path) -> None:
    for parent in reversed(path.absolute().parents):
        try:
            info = parent.lstat()
        except FileNotFoundError:
            continue
        _require(not stat.S_ISLNK(info.st_mode), f"symlink ancestor forbidden: {parent}")


def _regular_directory(path: Path) -> None:
    _no_symlink_ancestors(path)
    _require(os.path.lexists(path), f"directory is absent: {path}")
    info = path.lstat()
    _require(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode), f"regular non-symlink directory required: {path}")


def _read_small(path: Path) -> bytes:
    _no_symlink_ancestors(path)
    _require(os.path.lexists(path), f"file is absent: {path}")
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"regular non-symlink file required: {path}")
    _require(0 < before.st_size <= MAX_SMALL_FILE_BYTES, f"small file size invalid: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        token = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        _require(token == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns), "file changed before secure open")
        payload = b""
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(1_048_576, remaining))
            _require(bool(chunk), "unexpected EOF")
            payload += chunk
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", "file grew during read")
        final = os.fstat(descriptor)
        after = path.lstat()
        final_token = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
        path_token = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        _require(final_token == token == path_token, "file identity changed during read")
        return payload
    finally:
        os.close(descriptor)


def _hash_regular_file(path: Path, maximum_bytes: int) -> tuple[str, int]:
    _no_symlink_ancestors(path)
    before = path.lstat()
    _require(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode), f"regular non-symlink file required: {path}")
    _require(0 < before.st_size <= maximum_bytes, f"checkpoint payload size invalid: {path.name}")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    digest = hashlib.sha256()
    try:
        opened = os.fstat(descriptor)
        token = (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns)
        _require(token == (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns), "payload changed before secure open")
        remaining = opened.st_size
        while remaining:
            chunk = os.read(descriptor, min(8 * 1024 * 1024, remaining))
            _require(bool(chunk), "unexpected checkpoint EOF")
            digest.update(chunk)
            remaining -= len(chunk)
        _require(os.read(descriptor, 1) == b"", "checkpoint payload grew during hash")
        final = os.fstat(descriptor)
        after = path.lstat()
        final_token = (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns, final.st_ctime_ns)
        path_token = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
        _require(final_token == token == path_token, "checkpoint payload identity changed during hash")
        return digest.hexdigest(), opened.st_size
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes, mode: int = 0o600) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), mode)
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            _require(count > 0, "checkpoint write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    _require(function is not None, "atomic RENAME_NOREPLACE is unavailable")
    result = function(_AT_FDCWD, os.fsencode(source), _AT_FDCWD, os.fsencode(destination), _RENAME_NOREPLACE)
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise CheckpointError(f"checkpoint destination already exists: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def _identity_from(document: Mapping[str, object]) -> TrainingRunIdentity:
    return TrainingRunIdentity(**{key: str(value) for key, value in document.items()})


def _state_from(document: Mapping[str, object]) -> EpochState:
    return EpochState(
        epoch=int(document["epoch"]),
        optimizer_steps=int(document["optimizer_steps"]),
        best_epoch=int(document["best_epoch"]),
        best_validation_nll=float(document["best_validation_nll"]),
        epochs_without_improvement=int(document["epochs_without_improvement"]),
        early_stop_reached=bool(document["early_stop_reached"]),
    )


def validate_checkpoint(directory: str | Path, expected_identity: TrainingRunIdentity | None = None) -> ValidatedCheckpoint:
    directory = Path(directory)
    _regular_directory(directory)
    match = _FINAL.fullmatch(directory.name)
    _require(match is not None, "checkpoint directory name is invalid")
    _require({item.name for item in directory.iterdir()} == EXACT_FILES, "checkpoint exact file set drift")
    manifest_payload = _read_small(directory / MANIFEST)
    sidecar_payload = _read_small(directory / SIDECAR)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    _require(sidecar_payload == f"{manifest_sha}  {MANIFEST}\n".encode("ascii"), "checkpoint sidecar mismatch")
    try:
        document = json.loads(manifest_payload.decode("utf-8"), object_pairs_hook=_strict_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CheckpointError("checkpoint manifest is not strict JSON") from error
    _require(document.get("schema") == SCHEMA, "checkpoint schema drift")
    identity = _identity_from(document["identity"])
    state = _state_from(document["state"])
    _validate_identity(identity)
    _validate_state(state)
    _require(state.epoch == int(match.group(1)), "checkpoint epoch/name mismatch")
    if expected_identity is not None:
        _require(identity == expected_identity, "checkpoint training identity drift")
    files = document.get("files")
    _require(isinstance(files, dict) and set(files) == set(PAYLOAD_FILES), "checkpoint payload labels drift")
    hashes: list[tuple[str, str]] = []
    for name in PAYLOAD_FILES:
        entry = files[name]
        _require(isinstance(entry, dict) and set(entry) == {"sha256", "size_bytes"}, f"checkpoint file metadata drift: {name}")
        maximum = MAX_SMALL_FILE_BYTES if name == "epoch_history.json" else MAX_STATE_FILE_BYTES
        digest, size = _hash_regular_file(directory / name, maximum)
        _require(entry["sha256"] == digest and entry["size_bytes"] == size, f"checkpoint payload hash/size mismatch: {name}")
        hashes.append((name, digest))
    _require(_canonical_json(document) == manifest_payload, "checkpoint manifest is not canonical")
    return ValidatedCheckpoint(directory, identity, state, manifest_sha, tuple(hashes))


def discover_checkpoints(root: str | Path, expected_identity: TrainingRunIdentity | None = None) -> tuple[ValidatedCheckpoint, ...]:
    root = Path(root)
    _regular_directory(root)
    children = list(root.iterdir())
    staging = [child.name for child in children if _STAGING.fullmatch(child.name)]
    _require(not staging, f"stranded checkpoint staging requires review: {staging}")
    unexpected = [child.name for child in children if _FINAL.fullmatch(child.name) is None]
    _require(not unexpected, f"unexpected checkpoint-root entries: {unexpected}")
    checkpoints = sorted((validate_checkpoint(child, expected_identity) for child in children), key=lambda item: item.state.epoch)
    _require([item.state.epoch for item in checkpoints] == list(range(1, len(checkpoints) + 1)), "checkpoint epochs must be contiguous from one")
    return tuple(checkpoints)


def plan_recovery(root: str | Path, expected_identity: TrainingRunIdentity) -> RecoveryPlan:
    _validate_identity(expected_identity)
    checkpoints = discover_checkpoints(root, expected_identity)
    if not checkpoints:
        return RecoveryPlan("START_NEW", None, 1)
    latest = checkpoints[-1]
    if latest.state.early_stop_reached or latest.state.epoch == 50:
        return RecoveryPlan("TRAINING_EPOCHS_COMPLETE", latest, latest.state.epoch + 1)
    return RecoveryPlan("RESUME_LATEST", latest, latest.state.epoch + 1)


def publish_checkpoint(
    root: str | Path,
    *,
    identity: TrainingRunIdentity,
    state: EpochState,
    payloads: Mapping[str, bytes],
) -> ValidatedCheckpoint:
    """Publish the next complete checkpoint with no overwrite or cleanup."""

    root = Path(root)
    _regular_directory(root)
    _validate_identity(identity)
    _validate_state(state)
    _require(tuple(payloads) == PAYLOAD_FILES, "checkpoint payload labels/order drift")
    for name, payload in payloads.items():
        _require(isinstance(payload, bytes) and payload, f"checkpoint payload must be nonempty bytes: {name}")
        maximum = MAX_SMALL_FILE_BYTES if name == "epoch_history.json" else MAX_STATE_FILE_BYTES
        _require(len(payload) <= maximum, f"checkpoint payload exceeds bound: {name}")
    existing = discover_checkpoints(root, identity)
    _require(state.epoch == len(existing) + 1, "checkpoint publication must append the next contiguous epoch")
    final = root / f"checkpoint-epoch-{state.epoch:04d}"
    stage = root / f".checkpoint-epoch-{state.epoch:04d}.staging-{identity.training_run_id}"
    _require(not os.path.lexists(final) and not os.path.lexists(stage), "checkpoint final or staging path already exists")
    os.mkdir(stage, 0o700)
    _fsync_directory(root)
    file_metadata: dict[str, dict[str, object]] = {}
    for name, payload in payloads.items():
        _write_exclusive(stage / name, payload)
        file_metadata[name] = {"sha256": hashlib.sha256(payload).hexdigest(), "size_bytes": len(payload)}
    document = {
        "schema": SCHEMA,
        "identity": asdict(identity),
        "state": asdict(state),
        "files": file_metadata,
    }
    manifest_payload = _canonical_json(document)
    manifest_sha = hashlib.sha256(manifest_payload).hexdigest()
    _write_exclusive(stage / MANIFEST, manifest_payload)
    _write_exclusive(stage / SIDECAR, f"{manifest_sha}  {MANIFEST}\n".encode("ascii"))
    _fsync_directory(stage)
    # The staging name is intentionally not accepted by the public validator.
    # Hashes were constructed from exact bytes above and are revalidated after
    # the atomic no-overwrite rename.
    _rename_noreplace(stage, final)
    _fsync_directory(root)
    return validate_checkpoint(final, identity)


__all__ = [
    "EXACT_FILES",
    "MANIFEST",
    "PAYLOAD_FILES",
    "SIDECAR",
    "CheckpointError",
    "EpochState",
    "RecoveryPlan",
    "TrainingRunIdentity",
    "ValidatedCheckpoint",
    "discover_checkpoints",
    "plan_recovery",
    "publish_checkpoint",
    "validate_checkpoint",
]
