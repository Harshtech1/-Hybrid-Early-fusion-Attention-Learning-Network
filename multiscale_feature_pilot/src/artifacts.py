"""Atomic, checksum-addressed writes for external pilot artifacts.

These helpers deliberately refuse to replace an existing destination.  A
previous artifact may only be reused when its caller supplies the expected
SHA-256 digest and the on-disk file matches it exactly.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


_HASH_CHUNK_SIZE = 8 * 1024 * 1024


class ArtifactError(RuntimeError):
    """Base class for safe artifact-write failures."""


class ArtifactExistsError(ArtifactError):
    """Raised when a destination exists but was not explicitly reusable."""


class ArtifactHashMismatchError(ArtifactError):
    """Raised when an existing or newly serialized artifact has the wrong hash."""


class ArtifactWriteInProgressError(ArtifactError):
    """Raised when another cooperative writer owns the destination lock."""


@dataclass(frozen=True)
class ArtifactRecord:
    """Identity returned after a new atomic write or verified reuse."""

    path: Path
    size_bytes: int
    sha256: str
    reused: bool


def sha256_file(path: str | Path) -> str:
    """Return a lowercase SHA-256 digest while reading the file in chunks."""

    source = Path(path)
    hasher = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _normalise_expected_sha256(expected_sha256: str | None) -> str | None:
    if expected_sha256 is None:
        return None
    digest = expected_sha256.strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("expected_sha256 must contain exactly 64 hexadecimal characters")
    return digest


def _is_regular_file_without_symlink(path: Path) -> bool:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return False
    return stat.S_ISREG(metadata.st_mode) and not path.is_symlink()


def _record_for_existing(path: Path, expected_sha256: str | None) -> ArtifactRecord | None:
    """Return a verified reuse record, or ``None`` when no entry exists."""

    if not os.path.lexists(path):
        return None
    if not _is_regular_file_without_symlink(path):
        raise ArtifactExistsError(f"destination exists and is not a regular file: {path}")
    if expected_sha256 is None:
        raise ArtifactExistsError(
            f"destination already exists; expected_sha256 is required for reuse: {path}"
        )
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ArtifactHashMismatchError(
            f"existing artifact SHA-256 mismatch for {path}: "
            f"expected {expected_sha256}, got {actual}"
        )
    return ArtifactRecord(
        path=path,
        size_bytes=path.stat().st_size,
        sha256=actual,
        reused=True,
    )


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(
    path: str | Path,
    *,
    writer: Callable[[Path], None],
    expected_sha256: str | None,
) -> ArtifactRecord:
    destination = Path(path)
    expected = _normalise_expected_sha256(expected_sha256)
    destination.parent.mkdir(parents=True, exist_ok=True)

    lock_path = destination.parent / f".{destination.name}.lock"
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ArtifactWriteInProgressError(
            f"artifact destination is locked by another writer: {destination}"
        ) from exc

    temporary_path: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "w", encoding="ascii") as lock_stream:
            lock_stream.write(f"pid={os.getpid()}\n")
            lock_stream.flush()
            os.fsync(lock_stream.fileno())

        existing = _record_for_existing(destination, expected)
        if existing is not None:
            return existing

        temporary_descriptor, temporary_name = tempfile.mkstemp(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".partial",
        )
        os.close(temporary_descriptor)
        temporary_path = Path(temporary_name)

        writer(temporary_path)
        _fsync_file(temporary_path)
        actual = sha256_file(temporary_path)
        if expected is not None and actual != expected:
            raise ArtifactHashMismatchError(
                f"new artifact SHA-256 mismatch for {destination}: "
                f"expected {expected}, got {actual}"
            )

        # The lock serializes cooperative writers.  Recheck immediately before
        # os.replace so an unexpected external writer is never silently clobbered.
        if os.path.lexists(destination):
            raise ArtifactExistsError(
                f"destination appeared during serialization; refusing to replace it: {destination}"
            )
        os.replace(temporary_path, destination)
        temporary_path = None
        _fsync_directory(destination.parent)
        return ArtifactRecord(
            path=destination,
            size_bytes=destination.stat().st_size,
            sha256=actual,
            reused=False,
        )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        lock_path.unlink(missing_ok=True)


def atomic_save_tensor(
    tensor: torch.Tensor,
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ArtifactRecord:
    """Atomically save one tensor-only ``.pt`` artifact outside Git.

    CUDA tensors are detached and copied to contiguous CPU storage before
    serialization, so the resulting artifact loads portably with
    ``torch.load(..., map_location="cpu", weights_only=True)``.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be torch.Tensor, got {type(tensor).__name__}")
    destination = Path(path)
    if destination.suffix != ".pt":
        raise ValueError(f"tensor artifact must use a .pt suffix: {destination}")
    payload = tensor.detach().cpu().contiguous()

    def write_tensor(temporary_path: Path) -> None:
        torch.save(payload, temporary_path)

    return _atomic_write(
        destination,
        writer=write_tensor,
        expected_sha256=expected_sha256,
    )


def atomic_write_text(
    text: str,
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ArtifactRecord:
    """Atomically write UTF-8 text with the same no-overwrite contract."""

    if not isinstance(text, str):
        raise TypeError(f"text must be str, got {type(text).__name__}")

    def write_text(temporary_path: Path) -> None:
        temporary_path.write_text(text, encoding="utf-8", newline="")

    return _atomic_write(path, writer=write_text, expected_sha256=expected_sha256)


def atomic_write_json(
    value: Any,
    path: str | Path,
    *,
    expected_sha256: str | None = None,
) -> ArtifactRecord:
    """Atomically write deterministic, strict JSON terminated by one newline."""

    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    return atomic_write_text(serialized, path, expected_sha256=expected_sha256)


def atomic_write_csv(
    rows: Iterable[Mapping[str, object]],
    path: str | Path,
    *,
    fieldnames: Sequence[str],
    expected_sha256: str | None = None,
) -> ArtifactRecord:
    """Atomically write a deterministic CSV table with ``\n`` line endings."""

    if not fieldnames:
        raise ValueError("fieldnames must not be empty")
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return atomic_write_text(
        stream.getvalue(),
        path,
        expected_sha256=expected_sha256,
    )
