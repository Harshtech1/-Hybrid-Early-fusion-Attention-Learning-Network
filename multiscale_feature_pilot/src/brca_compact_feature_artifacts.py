"""Compact, patient-generic BRCA feature artifact persistence.

The compact layout retains one canonical combined tensor and row provenance.
It accepts already-produced CPU features only; it has no WSI, OpenSlide, CUDA,
model, network, deletion, or training interface.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .provenance import (
    PROVENANCE_FIELDS,
    PatchProvenance,
    ProvenanceError,
    validate_provenance_alignment,
)


SCHEMA = "BRCA_COMPACT_FEATURE_ARTIFACT_SET_V1"
FEATURE_DIM = 2048
COMBINED_FILENAME = "combined_features.pt"
PROVENANCE_FILENAME = "row_provenance.csv"
MANIFEST_FILENAME = "compact_manifest.json"
SIDECAR_FILENAME = "compact_manifest.json.sha256"
EXACT_FILES = frozenset(
    {COMBINED_FILENAME, PROVENANCE_FILENAME, MANIFEST_FILENAME, SIDECAR_FILENAME}
)
_RENAME_NOREPLACE = 1
_AT_FDCWD = -100


class CompactArtifactError(RuntimeError):
    pass


class CompactArtifactExistsError(CompactArtifactError):
    pass


@dataclass(frozen=True)
class CompactFeatureMetadata:
    patient_id: str
    slide_id: str
    gdc_file_uuid: str
    wsi_sha256: str
    coordinate_manifest_sha256: str
    omic_archive_sha256: str
    checkpoint_sha256: str
    source_policy_sha256: str
    implementation_commit: str
    scale_2x_rows: int
    scale_4x_rows: int


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hex(value: str, length: int, label: str) -> None:
    if len(value) != length or any(c not in "0123456789abcdef" for c in value):
        raise CompactArtifactError(f"{label} must be {length} lowercase hexadecimal characters")


def _validate_metadata(metadata: CompactFeatureMetadata) -> None:
    for label in ("patient_id", "slide_id", "gdc_file_uuid"):
        value = getattr(metadata, label)
        if not isinstance(value, str) or not value or Path(value).name != value:
            raise CompactArtifactError(f"{label} must be a non-empty basename-safe string")
    for label in (
        "wsi_sha256",
        "coordinate_manifest_sha256",
        "omic_archive_sha256",
        "checkpoint_sha256",
        "source_policy_sha256",
    ):
        _require_hex(getattr(metadata, label), 64, label)
    _require_hex(metadata.implementation_commit, 40, "implementation_commit")
    if metadata.scale_2x_rows <= 0 or metadata.scale_4x_rows <= 0:
        raise CompactArtifactError("both branch row counts must be positive")


def _validate_tensor(features: torch.Tensor, expected_rows: int) -> torch.Tensor:
    if not isinstance(features, torch.Tensor):
        raise CompactArtifactError("combined features must be a tensor")
    if features.device.type != "cpu":
        raise CompactArtifactError("combined features must be on CPU before publication")
    if features.dtype != torch.float32 or tuple(features.shape) != (expected_rows, FEATURE_DIM):
        raise CompactArtifactError(
            f"combined features must be float32 [{expected_rows},{FEATURE_DIM}]"
        )
    if features.requires_grad or not features.is_contiguous() or not bool(torch.isfinite(features).all()):
        raise CompactArtifactError("combined features must be detached, contiguous, and finite")
    return features


def _validate_provenance(records: Sequence[PatchProvenance], metadata: CompactFeatureMetadata) -> None:
    total = metadata.scale_2x_rows + metadata.scale_4x_rows
    try:
        validate_provenance_alignment(total, records)
    except ProvenanceError as error:
        raise CompactArtifactError(f"invalid row provenance: {error}") from error
    expected = (
        ["scale_2x"] * metadata.scale_2x_rows
        + ["scale_4x"] * metadata.scale_4x_rows
    )
    if [record.branch for record in records] != expected:
        raise CompactArtifactError("provenance must contain the complete 2x prefix then 4x suffix")


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().numpy().astype(np.dtype("<f4"), copy=False)
    return _sha256_bytes(array.tobytes(order="C"))


def _write_csv(path: Path, records: Sequence[PatchProvenance]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=PROVENANCE_FIELDS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
        stream.flush()
        os.fsync(stream.fileno())


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, "renameat2", None)
    if function is None:
        raise CompactArtifactError("atomic RENAME_NOREPLACE is unavailable")
    result = function(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result != 0:
        error = ctypes.get_errno()
        if error in (errno.EEXIST, errno.ENOTEMPTY):
            raise CompactArtifactExistsError(f"destination already exists: {destination}")
        raise OSError(error, os.strerror(error), str(destination))


def publish_compact_feature_artifacts(
    destination: str | Path,
    *,
    combined_features: torch.Tensor,
    row_provenance: Sequence[PatchProvenance],
    metadata: CompactFeatureMetadata,
) -> dict[str, object]:
    """Atomically publish the exact four-file compact layout."""

    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise CompactArtifactExistsError(f"destination already exists: {destination}")
    if not destination.parent.is_dir() or destination.parent.is_symlink():
        raise CompactArtifactError("destination parent must be an existing regular directory")
    _validate_metadata(metadata)
    rows = metadata.scale_2x_rows + metadata.scale_4x_rows
    tensor = _validate_tensor(combined_features, rows)
    _validate_provenance(row_provenance, metadata)

    stage = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging.", dir=destination.parent))
    try:
        tensor_path = stage / COMBINED_FILENAME
        with tensor_path.open("xb") as stream:
            torch.save(tensor, stream)
            stream.flush()
            os.fsync(stream.fileno())
        provenance_path = stage / PROVENANCE_FILENAME
        _write_csv(provenance_path, row_provenance)
        manifest = {
            "schema": SCHEMA,
            "identity": {
                "patient_id": metadata.patient_id,
                "slide_id": metadata.slide_id,
                "gdc_file_uuid": metadata.gdc_file_uuid,
            },
            "source_hashes": {
                "wsi_sha256": metadata.wsi_sha256,
                "coordinate_manifest_sha256": metadata.coordinate_manifest_sha256,
                "omic_archive_sha256": metadata.omic_archive_sha256,
                "checkpoint_sha256": metadata.checkpoint_sha256,
                "source_policy_sha256": metadata.source_policy_sha256,
                "implementation_commit": metadata.implementation_commit,
            },
            "tensor": {
                "filename": COMBINED_FILENAME,
                "shape": [rows, FEATURE_DIM],
                "dtype": "float32",
                "file_sha256": _sha256_file(tensor_path),
                "content_sha256": _tensor_content_sha256(tensor),
                "scale_2x_row_range": [0, metadata.scale_2x_rows],
                "scale_4x_row_range": [metadata.scale_2x_rows, rows],
                "pooling_performed": False,
                "transpose_performed": False,
                "natural_model_shape": [1, rows, FEATURE_DIM],
            },
            "provenance": {
                "filename": PROVENANCE_FILENAME,
                "rows": rows,
                "file_sha256": _sha256_file(provenance_path),
                "branch_order": ["scale_2x", "scale_4x"],
            },
            "retention": {
                "separate_branch_tensor_files": False,
                "patch_images": False,
                "canonical_tensor_count": 1,
            },
        }
        manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
        manifest_path = stage / MANIFEST_FILENAME
        with manifest_path.open("xb") as stream:
            stream.write(manifest_bytes)
            stream.flush()
            os.fsync(stream.fileno())
        manifest_sha = _sha256_bytes(manifest_bytes)
        sidecar = f"{manifest_sha}  {MANIFEST_FILENAME}\n".encode()
        with (stage / SIDECAR_FILENAME).open("xb") as stream:
            stream.write(sidecar)
            stream.flush()
            os.fsync(stream.fileno())
        validate_compact_feature_artifacts(stage, expected_manifest_sha256=manifest_sha)
        _rename_noreplace(stage, destination)
        return validate_compact_feature_artifacts(
            destination, expected_manifest_sha256=manifest_sha
        )
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _regular_file(path: Path) -> None:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise CompactArtifactError(f"artifact must be a regular non-symlink file: {path.name}")


def validate_compact_feature_artifacts(
    directory: str | Path, *, expected_manifest_sha256: str
) -> dict[str, object]:
    """Strictly validate one compact artifact set and return its manifest."""

    directory = Path(directory)
    if not directory.is_dir() or directory.is_symlink():
        raise CompactArtifactError("compact artifact path must be a non-symlink directory")
    if {item.name for item in directory.iterdir()} != EXACT_FILES:
        raise CompactArtifactError("compact artifact directory must contain exactly four files")
    for name in EXACT_FILES:
        _regular_file(directory / name)
    _require_hex(expected_manifest_sha256, 64, "expected_manifest_sha256")
    manifest_path = directory / MANIFEST_FILENAME
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise CompactArtifactError("compact manifest SHA256 mismatch")
    sidecar = (directory / SIDECAR_FILENAME).read_text(encoding="ascii")
    if sidecar != f"{expected_manifest_sha256}  {MANIFEST_FILENAME}\n":
        raise CompactArtifactError("compact manifest sidecar mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise CompactArtifactError("compact schema mismatch")
    tensor_info = manifest["tensor"]
    rows, width = tensor_info["shape"]
    if width != FEATURE_DIM or rows <= 1:
        raise CompactArtifactError("compact tensor shape is invalid")
    if tensor_info["scale_2x_row_range"][0] != 0:
        raise CompactArtifactError("2x row range must start at zero")
    split = tensor_info["scale_2x_row_range"][1]
    if tensor_info["scale_4x_row_range"] != [split, rows] or not 0 < split < rows:
        raise CompactArtifactError("branch row ranges are not contiguous")
    tensor_path = directory / COMBINED_FILENAME
    if _sha256_file(tensor_path) != tensor_info["file_sha256"]:
        raise CompactArtifactError("combined tensor file SHA256 mismatch")
    tensor = torch.load(tensor_path, map_location="cpu", weights_only=True)
    _validate_tensor(tensor, rows)
    if _tensor_content_sha256(tensor) != tensor_info["content_sha256"]:
        raise CompactArtifactError("combined tensor content SHA256 mismatch")
    provenance = manifest["provenance"]
    if provenance["rows"] != rows or provenance["branch_order"] != ["scale_2x", "scale_4x"]:
        raise CompactArtifactError("compact provenance contract mismatch")
    if _sha256_file(directory / PROVENANCE_FILENAME) != provenance["file_sha256"]:
        raise CompactArtifactError("provenance file SHA256 mismatch")
    with (directory / PROVENANCE_FILENAME).open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if len(records) != rows:
        raise CompactArtifactError("provenance row count mismatch")
    if [int(row["global_row_index"]) for row in records] != list(range(rows)):
        raise CompactArtifactError("provenance global indices are not contiguous")
    expected_branches = ["scale_2x"] * split + ["scale_4x"] * (rows - split)
    if [row["branch"] for row in records] != expected_branches:
        raise CompactArtifactError("provenance branch order mismatch")
    if manifest["retention"] != {
        "canonical_tensor_count": 1,
        "patch_images": False,
        "separate_branch_tensor_files": False,
    }:
        raise CompactArtifactError("compact retention contract mismatch")
    return manifest


__all__ = [
    "CompactArtifactError",
    "CompactArtifactExistsError",
    "CompactFeatureMetadata",
    "publish_compact_feature_artifacts",
    "validate_compact_feature_artifacts",
]
