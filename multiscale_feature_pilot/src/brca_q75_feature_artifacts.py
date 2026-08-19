"""Atomic publication of the locked BRCA Q75 feature artifact set.

This module owns persistence only.  It never opens a WSI, constructs a model,
uses CUDA, or trains.  Already-extracted CPU tensors and their row provenance
are checked against the exact Q75 contract, written to a sibling staging
directory, validated byte-for-byte, and committed with Linux
``RENAME_NOREPLACE``.  Existing destinations are never resumed or replaced.
"""

from __future__ import annotations

import csv
import ctypes
import errno
import hashlib
import io
import json
import math
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Final

import numpy as np
import torch

from .provenance import PROVENANCE_FIELDS, PatchProvenance


SCHEMA: Final = "BRCA_Q75_FEATURE_ARTIFACT_SET_V1"
EXPECTED_OUTPUT_BASENAME: Final = "Q75.features"
SCALE_2X_ROWS: Final = 13_487
SCALE_4X_ROWS: Final = 3_458
FEATURE_DIM: Final = 2_048
COMBINED_ROWS: Final = SCALE_2X_ROWS + SCALE_4X_ROWS
BRANCH_ORDER: Final = ("scale_2x", "scale_4x")

FEATURE_FILENAMES: Final = {
    "scale_2x_features": "scale_2x_features.pt",
    "scale_4x_features": "scale_4x_features.pt",
    "combined_features": "combined_features.pt",
}
PROVENANCE_FILENAME: Final = "row_provenance.csv"
MANIFEST_FILENAME: Final = "feature_manifest.json"
MANIFEST_SHA256_FILENAME: Final = "feature_manifest.json.sha256"
TENSOR_SERIALIZATION: Final = "torch.save_file_object_zip_v1"
TENSOR_CONTENT_SERIALIZATION: Final = "little_endian_float32_C_order_raw_bytes"
COORDINATE_CONTENT_SERIALIZATION: Final = "little_endian_int64_xy_C_order_raw_bytes"

EXPECTED_PATIENT_ID: Final = "TCGA-E2-A154"
EXPECTED_SLIDE_ID: Final = (
    "TCGA-E2-A154-01Z-00-DX1."
    "01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
)
EXPECTED_GDC_FILE_UUID: Final = "25aec062-60d1-446e-a1c6-0c79cc74a770"
EXPECTED_WSI_FILENAME: Final = EXPECTED_SLIDE_ID
EXPECTED_WSI_SIZE_BYTES: Final = 1_360_743_825
EXPECTED_WSI_MD5: Final = "a8c4b68fb6e0ab3e862efe3ed1fe10d7"
EXPECTED_WSI_SHA256: Final = (
    "844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37"
)

EXPECTED_COORDINATE_MANIFEST_FILENAME: Final = "coordinate_manifest.json"
EXPECTED_COORDINATE_MANIFEST_SIZE_BYTES: Final = 6_537
EXPECTED_COORDINATE_MANIFEST_SHA256: Final = (
    "438165ce6b3be9d26d66c65cd70793e29cc92208cfb6a78bf68043bc4b4a4e90"
)
EXPECTED_COORDINATE_BRANCHES: Final = MappingProxyType(
    {
        "scale_2x": MappingProxyType(
            {
                "artifact_filename": "scale_2x_coordinates.h5",
                "artifact_size_bytes": 225_464,
                "artifact_sha256": (
                    "d88c201d379a5954cdfa4d785760f6c8f9d4b8bec498f7f31d040b1fdf7440ec"
                ),
                "coordinates_sha256": (
                    "88e1ac8e00d4f05da7f83e542bfe7c933e9849a29ae60deac68adabe79e748b0"
                ),
                "coordinate_count": 13_487,
                "source_level": 0,
                "effective_mpp_x": 0.4936,
                "effective_mpp_y": 0.4936,
            }
        ),
        "scale_4x": MappingProxyType(
            {
                "artifact_filename": "scale_4x_coordinates.h5",
                "artifact_size_bytes": 65_000,
                "artifact_sha256": (
                    "0b0cfdaa26493dd24c3bbcba9f57c6b10d6060ba0f5f8b0a59bc0938ff655d94"
                ),
                "coordinates_sha256": (
                    "63f58c687943509a55314ced55c7afe1610a26873a751d3e1d0f2d06cff3fb5d"
                ),
                "coordinate_count": 3_458,
                "source_level": 1,
                "effective_mpp_x": 0.9872163682185965,
                "effective_mpp_y": 0.9872163682185965,
            }
        ),
    }
)

EXPECTED_OMIC_ARCHIVE_FILENAME: Final = "tcga_brca_all_clean.csv.zip"
EXPECTED_OMIC_ARCHIVE_SIZE_BYTES: Final = 4_081_277
EXPECTED_OMIC_ARCHIVE_SHA256: Final = (
    "4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8"
)
EXPECTED_OMIC_MEMBER_FILENAME: Final = "./tcga_brca_all_clean.csv.zip"
EXPECTED_OMIC_MEMBER_SIZE_BYTES: Final = 15_021_018
EXPECTED_OMIC_MEMBER_SHA256: Final = (
    "052637f2a69c515812796d9638566cb75299b6a3571dbdc5363496f12665027d"
)
EXPECTED_OMIC_SOURCE_ROW_INDEX: Final = "771"
EXPECTED_OMIC_DIMS: Final = MappingProxyType(
    {"rna": 1_558, "mutation": 21, "cnv": 1_333}
)
EXPECTED_OMIC_CONTENT_SHA256: Final = MappingProxyType(
    {
        "rna": "7c8ec0aa7c77227ad1ad5347b3612fe2c38d210295d6d1914862c7638960f916",
        "mutation": "81adcc3d1d6cfdde773ea46a27284afcf586598b5bd5cfecb675d98d70eabedf",
        "cnv": "78bf5e3d18c12923687506d873800f5c01a83a5737e488296acde26814ed6803",
    }
)

EXPECTED_ENCODER_IDENTITY: Final = (
    "torchvision.models.resnet50+ResNet50_Weights.IMAGENET1K_V2+fc_identity"
)
EXPECTED_CHECKPOINT_FILENAME: Final = "resnet50-11ad3fa6.pth"
EXPECTED_CHECKPOINT_SIZE_BYTES: Final = 102_540_417
EXPECTED_CHECKPOINT_SHA256: Final = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
EXPECTED_SOURCE_POLICY_NAME: Final = "brca_q75_gpu_execution_authorization.yaml"

_HASH_CHUNK_SIZE: Final = 8 * 1024 * 1024
_RENAME_NOREPLACE: Final = 1
_AT_FDCWD: Final = -100
_COMMIT_RE: Final = re.compile(r"^[0-9a-f]{40}$")


class FeatureArtifactError(RuntimeError):
    """Base class for Q75 feature artifact failures."""


class FeatureValidationError(FeatureArtifactError):
    """Raised when inputs or persisted artifacts violate the locked schema."""


class FeatureArtifactExistsError(FeatureArtifactError):
    """Raised instead of replacing or resuming an existing destination."""


class FeaturePublicationInProgressError(FeatureArtifactError):
    """Raised when a lock or sibling staging directory already exists."""


class FeatureHashMismatchError(FeatureArtifactError):
    """Raised when an artifact differs from a recorded or external digest."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FeatureValidationError(message)


def _digest(value: object, length: int, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a hexadecimal string")
    normalized = value.lower()
    _require(
        len(normalized) == length
        and all(character in "0123456789abcdef" for character in normalized),
        f"{label} must contain exactly {length} hexadecimal characters",
    )
    return normalized


def _positive_int(value: object, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{label} must be a positive integer",
    )
    return int(value)


def _positive_float(value: object, label: str) -> float:
    _require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0,
        f"{label} must be finite and positive",
    )
    return float(value)


def _basename(value: object, suffix: str, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} is required")
    _require(
        Path(value).name == value and value.lower().endswith(suffix),
        f"{label} must be a basename ending in {suffix}",
    )
    return value


def sha256_file(path: str | Path) -> str:
    """Hash one file without loading the whole artifact into memory."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


@dataclass(frozen=True)
class CoordinateFeatureBinding:
    """Immutable identity of one exact Q75 coordinate branch."""

    branch: str
    artifact_filename: str
    artifact_size_bytes: int
    artifact_sha256: str
    coordinates_sha256: str
    coordinate_count: int
    source_level: int
    effective_mpp_x: float
    effective_mpp_y: float

    def __post_init__(self) -> None:
        _require(self.branch in BRANCH_ORDER, "coordinate branch is not locked")
        expected = EXPECTED_COORDINATE_BRANCHES[self.branch]
        actual = {
            "artifact_filename": self.artifact_filename,
            "artifact_size_bytes": self.artifact_size_bytes,
            "artifact_sha256": self.artifact_sha256,
            "coordinates_sha256": self.coordinates_sha256,
            "coordinate_count": self.coordinate_count,
            "source_level": self.source_level,
            "effective_mpp_x": self.effective_mpp_x,
            "effective_mpp_y": self.effective_mpp_y,
        }
        _basename(self.artifact_filename, ".h5", "coordinate artifact filename")
        _positive_int(self.artifact_size_bytes, "coordinate artifact size")
        _digest(self.artifact_sha256, 64, "coordinate artifact SHA-256")
        _digest(self.coordinates_sha256, 64, "coordinate-content SHA-256")
        _require(
            isinstance(self.coordinate_count, int)
            and not isinstance(self.coordinate_count, bool),
            "coordinate_count must be an integer",
        )
        _require(
            isinstance(self.source_level, int) and not isinstance(self.source_level, bool),
            "source_level must be an integer",
        )
        _positive_float(self.effective_mpp_x, f"{self.branch} effective_mpp_x")
        _positive_float(self.effective_mpp_y, f"{self.branch} effective_mpp_y")
        _require(actual == dict(expected), f"{self.branch} coordinate identity drift")

    def to_manifest(self) -> dict[str, object]:
        return {
            "artifact_filename": self.artifact_filename,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "branch": self.branch,
            "coordinate_count": self.coordinate_count,
            "coordinates_sha256": self.coordinates_sha256,
            "effective_mpp_x": float(self.effective_mpp_x),
            "effective_mpp_y": float(self.effective_mpp_y),
            "source_level": self.source_level,
        }


@dataclass(frozen=True)
class FeatureArtifactMetadata:
    """Exact external and implementation identities signed by the manifest."""

    patient_id: str
    slide_id: str
    gdc_file_uuid: str
    wsi_filename: str
    wsi_size_bytes: int
    wsi_md5: str
    wsi_sha256: str
    coordinate_manifest_filename: str
    coordinate_manifest_size_bytes: int
    coordinate_manifest_sha256: str
    scale_2x_coordinates: CoordinateFeatureBinding
    scale_4x_coordinates: CoordinateFeatureBinding
    omic_archive_filename: str
    omic_archive_size_bytes: int
    omic_archive_sha256: str
    omic_member_filename: str
    omic_member_size_bytes: int
    omic_member_sha256: str
    omic_source_row_index: str
    rna_feature_count: int
    mutation_feature_count: int
    cnv_feature_count: int
    rna_content_sha256: str
    mutation_content_sha256: str
    cnv_content_sha256: str
    encoder_identity: str
    checkpoint_filename: str
    checkpoint_size_bytes: int
    checkpoint_sha256: str
    source_policy_name: str
    source_policy_sha256: str
    implementation_git_commit: str
    implementation_source_sha256: Mapping[str, str]

    def __post_init__(self) -> None:
        exact_values = {
            "patient_id": EXPECTED_PATIENT_ID,
            "slide_id": EXPECTED_SLIDE_ID,
            "gdc_file_uuid": EXPECTED_GDC_FILE_UUID,
            "wsi_filename": EXPECTED_WSI_FILENAME,
            "wsi_size_bytes": EXPECTED_WSI_SIZE_BYTES,
            "wsi_md5": EXPECTED_WSI_MD5,
            "wsi_sha256": EXPECTED_WSI_SHA256,
            "coordinate_manifest_filename": EXPECTED_COORDINATE_MANIFEST_FILENAME,
            "coordinate_manifest_size_bytes": EXPECTED_COORDINATE_MANIFEST_SIZE_BYTES,
            "coordinate_manifest_sha256": EXPECTED_COORDINATE_MANIFEST_SHA256,
            "omic_archive_filename": EXPECTED_OMIC_ARCHIVE_FILENAME,
            "omic_archive_size_bytes": EXPECTED_OMIC_ARCHIVE_SIZE_BYTES,
            "omic_archive_sha256": EXPECTED_OMIC_ARCHIVE_SHA256,
            "omic_member_filename": EXPECTED_OMIC_MEMBER_FILENAME,
            "omic_member_size_bytes": EXPECTED_OMIC_MEMBER_SIZE_BYTES,
            "omic_member_sha256": EXPECTED_OMIC_MEMBER_SHA256,
            "omic_source_row_index": EXPECTED_OMIC_SOURCE_ROW_INDEX,
            "rna_feature_count": EXPECTED_OMIC_DIMS["rna"],
            "mutation_feature_count": EXPECTED_OMIC_DIMS["mutation"],
            "cnv_feature_count": EXPECTED_OMIC_DIMS["cnv"],
            "rna_content_sha256": EXPECTED_OMIC_CONTENT_SHA256["rna"],
            "mutation_content_sha256": EXPECTED_OMIC_CONTENT_SHA256["mutation"],
            "cnv_content_sha256": EXPECTED_OMIC_CONTENT_SHA256["cnv"],
            "encoder_identity": EXPECTED_ENCODER_IDENTITY,
            "checkpoint_filename": EXPECTED_CHECKPOINT_FILENAME,
            "checkpoint_size_bytes": EXPECTED_CHECKPOINT_SIZE_BYTES,
            "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        }
        for field_name, expected in exact_values.items():
            _require(getattr(self, field_name) == expected, f"Q75 {field_name} drift")
        _require(
            isinstance(self.scale_2x_coordinates, CoordinateFeatureBinding)
            and self.scale_2x_coordinates.branch == "scale_2x",
            "scale_2x coordinate binding drift",
        )
        _require(
            isinstance(self.scale_4x_coordinates, CoordinateFeatureBinding)
            and self.scale_4x_coordinates.branch == "scale_4x",
            "scale_4x coordinate binding drift",
        )
        _require(
            self.source_policy_name == EXPECTED_SOURCE_POLICY_NAME,
            f"Q75 source_policy_name must be {EXPECTED_SOURCE_POLICY_NAME}",
        )
        _digest(self.source_policy_sha256, 64, "source policy SHA-256")
        _require(
            isinstance(self.implementation_git_commit, str)
            and bool(_COMMIT_RE.fullmatch(self.implementation_git_commit)),
            "implementation_git_commit must be a lowercase 40-character Git object ID",
        )
        _require(
            isinstance(self.implementation_source_sha256, Mapping)
            and bool(self.implementation_source_sha256),
            "implementation_source_sha256 must be a nonempty mapping",
        )
        normalized_sources: dict[str, str] = {}
        for source_name, source_digest in self.implementation_source_sha256.items():
            _require(
                isinstance(source_name, str) and bool(source_name),
                "implementation source path must be a nonempty string",
            )
            parsed = PurePosixPath(source_name)
            _require(
                not parsed.is_absolute()
                and str(parsed) == source_name
                and ".." not in parsed.parts
                and "." not in parsed.parts,
                f"implementation source path must be normalized and relative: {source_name}",
            )
            normalized_sources[source_name] = _digest(
                source_digest, 64, f"implementation source SHA-256 for {source_name}"
            )
        object.__setattr__(
            self,
            "implementation_source_sha256",
            MappingProxyType(dict(sorted(normalized_sources.items()))),
        )

    def to_manifest(self) -> dict[str, object]:
        return {
            "checkpoint": {
                "encoder_identity": self.encoder_identity,
                "filename": self.checkpoint_filename,
                "sha256": self.checkpoint_sha256,
                "size_bytes": self.checkpoint_size_bytes,
            },
            "coordinate_manifest": {
                "filename": self.coordinate_manifest_filename,
                "sha256": self.coordinate_manifest_sha256,
                "size_bytes": self.coordinate_manifest_size_bytes,
            },
            "coordinates": {
                "scale_2x": self.scale_2x_coordinates.to_manifest(),
                "scale_4x": self.scale_4x_coordinates.to_manifest(),
            },
            "identity": {
                "candidate_label": "Q75",
                "cohort": "TCGA-BRCA",
                "gdc_file_uuid": self.gdc_file_uuid,
                "patient_id": self.patient_id,
                "slide_id": self.slide_id,
                "wsi_filename": self.wsi_filename,
                "wsi_md5": self.wsi_md5,
                "wsi_sha256": self.wsi_sha256,
                "wsi_size_bytes": self.wsi_size_bytes,
            },
            "omic": {
                "archive_filename": self.omic_archive_filename,
                "archive_sha256": self.omic_archive_sha256,
                "archive_size_bytes": self.omic_archive_size_bytes,
                "feature_counts": {
                    "cnv": self.cnv_feature_count,
                    "mutation": self.mutation_feature_count,
                    "rna": self.rna_feature_count,
                },
                "member_filename": self.omic_member_filename,
                "member_sha256": self.omic_member_sha256,
                "member_size_bytes": self.omic_member_size_bytes,
                "modality_content_sha256": {
                    "cnv": self.cnv_content_sha256,
                    "mutation": self.mutation_content_sha256,
                    "rna": self.rna_content_sha256,
                },
                "source_row_index": self.omic_source_row_index,
            },
            "source_policy": {
                "implementation_git_commit": self.implementation_git_commit,
                "implementation_source_sha256": dict(self.implementation_source_sha256),
                "name": self.source_policy_name,
                "sha256": self.source_policy_sha256,
            },
        }


@dataclass(frozen=True)
class FeatureFileRecord:
    name: str
    path: Path
    shape: tuple[int, int]
    size_bytes: int
    sha256: str
    tensor_content_sha256: str


@dataclass(frozen=True)
class ProvenanceFileRecord:
    path: Path
    row_count: int
    size_bytes: int
    sha256: str
    coordinate_sha256_by_branch: Mapping[str, str]


@dataclass(frozen=True)
class FeatureArtifactSetRecord:
    directory: Path
    manifest_path: Path
    manifest_sha256_path: Path
    manifest_sha256: str
    features: tuple[FeatureFileRecord, FeatureFileRecord, FeatureFileRecord]
    provenance: ProvenanceFileRecord
    metadata: FeatureArtifactMetadata

    def feature_for(self, name: str) -> FeatureFileRecord:
        matches = tuple(record for record in self.features if record.name == name)
        if len(matches) != 1:
            raise FeatureValidationError(f"artifact set lacks exactly one {name}")
        return matches[0]


def _expected_feature_shapes() -> dict[str, tuple[int, int]]:
    return {
        "scale_2x_features": (SCALE_2X_ROWS, FEATURE_DIM),
        "scale_4x_features": (SCALE_4X_ROWS, FEATURE_DIM),
        "combined_features": (SCALE_2X_ROWS + SCALE_4X_ROWS, FEATURE_DIM),
    }


def _validate_feature_tensor(value: object, name: str) -> torch.Tensor:
    _require(isinstance(value, torch.Tensor), f"{name} must be a torch.Tensor")
    tensor = value
    expected_shape = _expected_feature_shapes()[name]
    _require(tuple(tensor.shape) == expected_shape, f"{name} shape must be {expected_shape}")
    _require(tensor.dtype is torch.float32, f"{name} dtype must be torch.float32")
    _require(tensor.device.type == "cpu", f"{name} must be on CPU")
    _require(tensor.is_contiguous(), f"{name} must be contiguous")
    _require(not tensor.requires_grad, f"{name} must not require gradients")
    _require(tensor.storage_offset() == 0, f"{name} must have zero storage offset")
    _require(
        tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size(),
        f"{name} must use compact tensor storage",
    )
    _require(bool(torch.isfinite(tensor).all().item()), f"{name} contains NaN or Inf")
    return tensor


def _hash_memoryview(view: memoryview) -> str:
    hasher = hashlib.sha256()
    byte_view = view.cast("B")
    for offset in range(0, len(byte_view), _HASH_CHUNK_SIZE):
        hasher.update(byte_view[offset : offset + _HASH_CHUNK_SIZE])
    return hasher.hexdigest()


def _tensor_content_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().numpy().astype(np.dtype("<f4"), copy=False)
    return _hash_memoryview(memoryview(array))


def _validate_exact_concatenation(
    scale_2x: torch.Tensor,
    scale_4x: torch.Tensor,
    combined: torch.Tensor,
) -> None:
    split = SCALE_2X_ROWS
    _require(
        _tensor_content_sha256(combined[:split]) == _tensor_content_sha256(scale_2x),
        "combined_features does not contain scale_2x rows first",
    )
    _require(
        _tensor_content_sha256(combined[split:]) == _tensor_content_sha256(scale_4x),
        "combined_features does not contain scale_4x rows second",
    )


def _coordinate_sha256(rows: Sequence[PatchProvenance]) -> str:
    coordinates = np.empty((len(rows), 2), dtype=np.dtype("<i8"))
    for index, row in enumerate(rows):
        for column, value in enumerate((row.x, row.y)):
            _require(
                isinstance(value, int)
                and not isinstance(value, bool)
                and 0 <= value <= np.iinfo(np.int64).max,
                "provenance coordinates must be nonnegative int64 values",
            )
            coordinates[index, column] = value
    return _hash_memoryview(memoryview(coordinates))


def _validate_provenance(
    records: object,
    metadata: FeatureArtifactMetadata,
) -> tuple[PatchProvenance, ...]:
    _require(
        isinstance(records, Sequence) and not isinstance(records, (str, bytes)),
        "row_provenance must be a sequence",
    )
    rows = tuple(records)
    expected_total = SCALE_2X_ROWS + SCALE_4X_ROWS
    _require(len(rows) == expected_total, f"row_provenance must contain {expected_total} rows")
    bindings = {
        "scale_2x": metadata.scale_2x_coordinates,
        "scale_4x": metadata.scale_4x_coordinates,
    }
    by_branch: dict[str, list[PatchProvenance]] = {branch: [] for branch in BRANCH_ORDER}
    for global_index, row in enumerate(rows):
        _require(isinstance(row, PatchProvenance), "every provenance row must be PatchProvenance")
        expected_branch = "scale_2x" if global_index < SCALE_2X_ROWS else "scale_4x"
        expected_local = global_index if expected_branch == "scale_2x" else global_index - SCALE_2X_ROWS
        binding = bindings[expected_branch]
        _require(row.global_row_index == global_index, "global_row_index must be contiguous")
        _require(row.branch == expected_branch, "provenance branch order must be scale_2x then scale_4x")
        _require(row.local_patch_index == expected_local, "local_patch_index must be contiguous per branch")
        _require(row.level == binding.source_level, f"{expected_branch} provenance level drift")
        _require(
            math.isfinite(float(row.mpp_x))
            and math.isfinite(float(row.mpp_y))
            and float(row.mpp_x) == float(binding.effective_mpp_x)
            and float(row.mpp_y) == float(binding.effective_mpp_y),
            f"{expected_branch} provenance MPP drift",
        )
        by_branch[expected_branch].append(row)
    for branch, binding in bindings.items():
        if _coordinate_sha256(by_branch[branch]) != binding.coordinates_sha256:
            raise FeatureHashMismatchError(
                f"{branch} provenance coordinates do not match coordinate-content SHA-256"
            )
    return rows


def _provenance_bytes(rows: Sequence[PatchProvenance]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=PROVENANCE_FIELDS, lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(
            {
                "global_row_index": row.global_row_index,
                "branch": row.branch,
                "local_patch_index": row.local_patch_index,
                "x": row.x,
                "y": row.y,
                "level": row.level,
                "mpp_x": row.mpp_x,
                "mpp_y": row.mpp_y,
            }
        )
    return stream.getvalue().encode("utf-8")


def _parse_provenance(path: Path) -> tuple[PatchProvenance, ...]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _require(reader.fieldnames == list(PROVENANCE_FIELDS), "provenance CSV header drift")
            parsed: list[PatchProvenance] = []
            for row in reader:
                _require(set(row) == set(PROVENANCE_FIELDS), "provenance CSV column drift")
                try:
                    parsed.append(
                        PatchProvenance(
                            global_row_index=int(row["global_row_index"]),
                            branch=row["branch"],
                            local_patch_index=int(row["local_patch_index"]),
                            x=int(row["x"]),
                            y=int(row["y"]),
                            level=int(row["level"]),
                            mpp_x=float(row["mpp_x"]),
                            mpp_y=float(row["mpp_y"]),
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise FeatureValidationError("provenance CSV contains an invalid value") from exc
    except UnicodeError as exc:
        raise FeatureValidationError("provenance CSV is not valid UTF-8") from exc
    return tuple(parsed)


def _manifest_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(document, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _write_tensor(path: Path, tensor: torch.Tensor) -> None:
    # A file object fixes the internal ZIP root to ``archive`` so bytes do not
    # depend on the destination basename.
    with path.open("xb") as stream:
        torch.save(tensor, stream, _use_new_zipfile_serialization=True)
        stream.flush()
        os.fsync(stream.fileno())


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise FeatureValidationError(f"missing {label}: {path}") from exc
    _require(
        stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
        f"{label} must be a regular non-symlink file",
    )


def _existing_parent(path: Path) -> Path:
    candidate = path
    while not os.path.lexists(candidate):
        if candidate.parent == candidate:
            raise FeatureValidationError(f"cannot locate an existing parent for {path}")
        candidate = candidate.parent
    return candidate


def _absolute_without_resolving(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _require_no_symlink_components(path: Path) -> None:
    existing = _existing_parent(path)
    for component in (existing, *existing.parents):
        metadata = component.lstat()
        _require(not stat.S_ISLNK(metadata.st_mode), f"path contains a symlink component: {component}")


def _require_outside_git(path: Path) -> None:
    existing = _existing_parent(path)
    directory = existing if existing.is_dir() else existing.parent
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        worktree = Path(result.stdout.strip()).resolve()
        resolved = path.resolve(strict=False)
        if resolved == worktree or resolved.is_relative_to(worktree):
            raise FeatureValidationError(f"feature artifacts must be published outside Git: {worktree}")


def _binding_from_manifest(branch: str, value: object) -> CoordinateFeatureBinding:
    expected_keys = {
        "artifact_filename",
        "artifact_sha256",
        "artifact_size_bytes",
        "branch",
        "coordinate_count",
        "coordinates_sha256",
        "effective_mpp_x",
        "effective_mpp_y",
        "source_level",
    }
    _require(isinstance(value, dict) and set(value) == expected_keys, f"{branch} binding keys drift")
    return CoordinateFeatureBinding(
        branch=value["branch"],
        artifact_filename=value["artifact_filename"],
        artifact_size_bytes=value["artifact_size_bytes"],
        artifact_sha256=value["artifact_sha256"],
        coordinates_sha256=value["coordinates_sha256"],
        coordinate_count=value["coordinate_count"],
        source_level=value["source_level"],
        effective_mpp_x=value["effective_mpp_x"],
        effective_mpp_y=value["effective_mpp_y"],
    )


def _metadata_from_manifest(value: object) -> FeatureArtifactMetadata:
    _require(isinstance(value, dict), "manifest metadata must be an object")
    _require(
        set(value) == {"checkpoint", "coordinate_manifest", "coordinates", "identity", "omic", "source_policy"},
        "manifest metadata keys drift",
    )
    checkpoint = value["checkpoint"]
    coordinate_manifest = value["coordinate_manifest"]
    coordinates = value["coordinates"]
    identity = value["identity"]
    omic = value["omic"]
    source_policy = value["source_policy"]
    required = (
        (checkpoint, {"encoder_identity", "filename", "sha256", "size_bytes"}, "checkpoint"),
        (coordinate_manifest, {"filename", "sha256", "size_bytes"}, "coordinate_manifest"),
        (coordinates, set(BRANCH_ORDER), "coordinates"),
        (
            identity,
            {
                "candidate_label",
                "cohort",
                "gdc_file_uuid",
                "patient_id",
                "slide_id",
                "wsi_filename",
                "wsi_md5",
                "wsi_sha256",
                "wsi_size_bytes",
            },
            "identity",
        ),
        (
            omic,
            {
                "archive_filename",
                "archive_sha256",
                "archive_size_bytes",
                "feature_counts",
                "member_filename",
                "member_sha256",
                "member_size_bytes",
                "modality_content_sha256",
                "source_row_index",
            },
            "omic",
        ),
        (
            source_policy,
            {"implementation_git_commit", "implementation_source_sha256", "name", "sha256"},
            "source_policy",
        ),
    )
    for item, keys, label in required:
        _require(isinstance(item, dict) and set(item) == keys, f"{label} metadata keys drift")
    _require(identity["candidate_label"] == "Q75", "candidate label drift")
    _require(identity["cohort"] == "TCGA-BRCA", "cohort drift")
    feature_counts = omic["feature_counts"]
    _require(
        isinstance(feature_counts, dict) and set(feature_counts) == set(EXPECTED_OMIC_DIMS),
        "Omic feature-count keys drift",
    )
    modality_content_sha256 = omic["modality_content_sha256"]
    _require(
        isinstance(modality_content_sha256, dict)
        and set(modality_content_sha256) == set(EXPECTED_OMIC_CONTENT_SHA256),
        "Omic modality-content SHA-256 keys drift",
    )
    metadata = FeatureArtifactMetadata(
        patient_id=identity["patient_id"],
        slide_id=identity["slide_id"],
        gdc_file_uuid=identity["gdc_file_uuid"],
        wsi_filename=identity["wsi_filename"],
        wsi_size_bytes=identity["wsi_size_bytes"],
        wsi_md5=identity["wsi_md5"],
        wsi_sha256=identity["wsi_sha256"],
        coordinate_manifest_filename=coordinate_manifest["filename"],
        coordinate_manifest_size_bytes=coordinate_manifest["size_bytes"],
        coordinate_manifest_sha256=coordinate_manifest["sha256"],
        scale_2x_coordinates=_binding_from_manifest("scale_2x", coordinates["scale_2x"]),
        scale_4x_coordinates=_binding_from_manifest("scale_4x", coordinates["scale_4x"]),
        omic_archive_filename=omic["archive_filename"],
        omic_archive_size_bytes=omic["archive_size_bytes"],
        omic_archive_sha256=omic["archive_sha256"],
        omic_member_filename=omic["member_filename"],
        omic_member_size_bytes=omic["member_size_bytes"],
        omic_member_sha256=omic["member_sha256"],
        omic_source_row_index=omic["source_row_index"],
        rna_feature_count=feature_counts["rna"],
        mutation_feature_count=feature_counts["mutation"],
        cnv_feature_count=feature_counts["cnv"],
        rna_content_sha256=modality_content_sha256["rna"],
        mutation_content_sha256=modality_content_sha256["mutation"],
        cnv_content_sha256=modality_content_sha256["cnv"],
        encoder_identity=checkpoint["encoder_identity"],
        checkpoint_filename=checkpoint["filename"],
        checkpoint_size_bytes=checkpoint["size_bytes"],
        checkpoint_sha256=checkpoint["sha256"],
        source_policy_name=source_policy["name"],
        source_policy_sha256=source_policy["sha256"],
        implementation_git_commit=source_policy["implementation_git_commit"],
        implementation_source_sha256=source_policy["implementation_source_sha256"],
    )
    _require(metadata.to_manifest() == value, "manifest metadata is not canonical")
    return metadata


def _load_tensor(path: Path, name: str) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise FeatureValidationError(f"{name} is not a valid tensor-only artifact") from exc
    return _validate_feature_tensor(tensor, name)


def _validate_directory(directory: Path, expected_manifest_sha256: str) -> FeatureArtifactSetRecord:
    expected_manifest_sha256 = _digest(expected_manifest_sha256, 64, "expected_manifest_sha256")
    _require_no_symlink_components(directory)
    _require_outside_git(directory)
    try:
        directory_metadata = directory.lstat()
    except FileNotFoundError as exc:
        raise FeatureValidationError(f"feature artifact directory is missing: {directory}") from exc
    _require(
        stat.S_ISDIR(directory_metadata.st_mode) and not directory.is_symlink(),
        "feature artifact destination must be a non-symlink directory",
    )
    expected_names = set(FEATURE_FILENAMES.values()) | {
        PROVENANCE_FILENAME,
        MANIFEST_FILENAME,
        MANIFEST_SHA256_FILENAME,
    }
    actual_names = {entry.name for entry in directory.iterdir()}
    _require(actual_names == expected_names, "feature artifact directory file set is not exact")
    for filename in expected_names:
        _regular_file(directory / filename, filename)

    manifest_path = directory / MANIFEST_FILENAME
    sidecar_path = directory / MANIFEST_SHA256_FILENAME
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise FeatureHashMismatchError(
            f"manifest SHA-256 mismatch: expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except UnicodeError as exc:
        raise FeatureValidationError("manifest SHA-256 sidecar is not ASCII") from exc
    if sidecar != f"{expected_manifest_sha256}  {MANIFEST_FILENAME}\n":
        raise FeatureHashMismatchError("manifest SHA-256 sidecar mismatch")
    manifest_bytes = manifest_path.read_bytes()
    try:
        document = json.loads(manifest_bytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise FeatureValidationError("feature manifest is not valid UTF-8 JSON") from exc
    _require(isinstance(document, dict), "feature manifest root must be an object")
    _require(set(document) == {"artifacts", "contract", "metadata", "schema"}, "feature manifest keys drift")
    _require(document["schema"] == SCHEMA, "feature manifest schema drift")
    _require(_manifest_bytes(document) == manifest_bytes, "feature manifest serialization is not canonical")
    metadata = _metadata_from_manifest(document["metadata"])

    expected_contract = {
        "branch_order": list(BRANCH_ORDER),
        "combined_rows": SCALE_2X_ROWS + SCALE_4X_ROWS,
        "concatenation": "torch.cat([scale_2x,scale_4x],dim=0)",
        "device": "cpu",
        "dtype": "float32",
        "feature_dim": FEATURE_DIM,
        "finite": True,
        "memory_layout": "contiguous_C_order",
        "natural_model_shape": [1, SCALE_2X_ROWS + SCALE_4X_ROWS, FEATURE_DIM],
        "pooling_performed": False,
        "scale_2x_rows": SCALE_2X_ROWS,
        "scale_4x_rows": SCALE_4X_ROWS,
        "transpose_performed": False,
    }
    _require(document["contract"] == expected_contract, "feature contract drift")
    artifact_document = document["artifacts"]
    _require(
        isinstance(artifact_document, dict)
        and set(artifact_document) == {*FEATURE_FILENAMES, "row_provenance"},
        "feature manifest artifact set drift",
    )

    feature_records: list[FeatureFileRecord] = []
    loaded: dict[str, torch.Tensor] = {}
    for name in ("scale_2x_features", "scale_4x_features", "combined_features"):
        entry = artifact_document[name]
        expected_keys = {
            "content_serialization",
            "device",
            "dtype",
            "filename",
            "memory_layout",
            "serialization",
            "sha256",
            "shape",
            "size_bytes",
            "tensor_content_sha256",
        }
        _require(isinstance(entry, dict) and set(entry) == expected_keys, f"{name} manifest keys drift")
        _require(entry["filename"] == FEATURE_FILENAMES[name], f"{name} filename drift")
        _require(entry["serialization"] == TENSOR_SERIALIZATION, f"{name} serialization drift")
        _require(entry["content_serialization"] == TENSOR_CONTENT_SERIALIZATION, f"{name} content serialization drift")
        _require(entry["device"] == "cpu", f"{name} device drift")
        _require(entry["dtype"] == "float32", f"{name} dtype drift")
        _require(entry["memory_layout"] == "contiguous_C_order", f"{name} memory layout drift")
        expected_shape = _expected_feature_shapes()[name]
        _require(entry["shape"] == list(expected_shape), f"{name} shape metadata drift")
        artifact_path = directory / FEATURE_FILENAMES[name]
        actual_sha256 = sha256_file(artifact_path)
        expected_sha256 = _digest(entry["sha256"], 64, f"{name} SHA-256")
        if actual_sha256 != expected_sha256:
            raise FeatureHashMismatchError(f"{name} file SHA-256 mismatch")
        _require(
            isinstance(entry["size_bytes"], int)
            and not isinstance(entry["size_bytes"], bool)
            and entry["size_bytes"] == artifact_path.stat().st_size,
            f"{name} size mismatch",
        )
        tensor = _load_tensor(artifact_path, name)
        actual_content_sha256 = _tensor_content_sha256(tensor)
        expected_content_sha256 = _digest(entry["tensor_content_sha256"], 64, f"{name} tensor-content SHA-256")
        if actual_content_sha256 != expected_content_sha256:
            raise FeatureHashMismatchError(f"{name} tensor-content SHA-256 mismatch")
        loaded[name] = tensor
        feature_records.append(
            FeatureFileRecord(
                name=name,
                path=artifact_path,
                shape=expected_shape,
                size_bytes=artifact_path.stat().st_size,
                sha256=actual_sha256,
                tensor_content_sha256=actual_content_sha256,
            )
        )
    _validate_exact_concatenation(
        loaded["scale_2x_features"],
        loaded["scale_4x_features"],
        loaded["combined_features"],
    )

    provenance_entry = artifact_document["row_provenance"]
    expected_provenance_keys = {
        "columns",
        "coordinate_content_serialization",
        "coordinate_sha256_by_branch",
        "filename",
        "row_count",
        "sha256",
        "size_bytes",
    }
    _require(
        isinstance(provenance_entry, dict) and set(provenance_entry) == expected_provenance_keys,
        "row_provenance manifest keys drift",
    )
    _require(provenance_entry["filename"] == PROVENANCE_FILENAME, "provenance filename drift")
    _require(provenance_entry["columns"] == list(PROVENANCE_FIELDS), "provenance columns drift")
    _require(
        provenance_entry["coordinate_content_serialization"] == COORDINATE_CONTENT_SERIALIZATION,
        "coordinate content serialization drift",
    )
    expected_total = SCALE_2X_ROWS + SCALE_4X_ROWS
    _require(provenance_entry["row_count"] == expected_total, "provenance row count drift")
    provenance_path = directory / PROVENANCE_FILENAME
    actual_provenance_sha256 = sha256_file(provenance_path)
    expected_provenance_sha256 = _digest(provenance_entry["sha256"], 64, "row_provenance SHA-256")
    if actual_provenance_sha256 != expected_provenance_sha256:
        raise FeatureHashMismatchError("row_provenance file SHA-256 mismatch")
    _require(
        isinstance(provenance_entry["size_bytes"], int)
        and not isinstance(provenance_entry["size_bytes"], bool)
        and provenance_entry["size_bytes"] == provenance_path.stat().st_size,
        "row_provenance size mismatch",
    )
    rows = _validate_provenance(_parse_provenance(provenance_path), metadata)
    _require(_provenance_bytes(rows) == provenance_path.read_bytes(), "provenance CSV is not canonical")
    coordinate_digests = {
        "scale_2x": _coordinate_sha256(rows[:SCALE_2X_ROWS]),
        "scale_4x": _coordinate_sha256(rows[SCALE_2X_ROWS:]),
    }
    _require(
        provenance_entry["coordinate_sha256_by_branch"] == coordinate_digests,
        "provenance coordinate hashes drift",
    )
    provenance_record = ProvenanceFileRecord(
        path=provenance_path,
        row_count=expected_total,
        size_bytes=provenance_path.stat().st_size,
        sha256=actual_provenance_sha256,
        coordinate_sha256_by_branch=MappingProxyType(coordinate_digests),
    )
    return FeatureArtifactSetRecord(
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256_path=sidecar_path,
        manifest_sha256=actual_manifest_sha256,
        features=(feature_records[0], feature_records[1], feature_records[2]),
        provenance=provenance_record,
        metadata=metadata,
    )


def validate_brca_q75_feature_artifacts(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> FeatureArtifactSetRecord:
    """Validate one exact externally anchored Q75 feature artifact set."""

    artifact_directory = _absolute_without_resolving(directory)
    _require(
        artifact_directory.name == EXPECTED_OUTPUT_BASENAME,
        f"Q75 destination basename must be {EXPECTED_OUTPUT_BASENAME}",
    )
    return _validate_directory(artifact_directory, expected_manifest_sha256)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise FeatureArtifactError("atomic RENAME_NOREPLACE is unavailable; refusing non-atomic publication")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        _AT_FDCWD,
        os.fsencode(source),
        _AT_FDCWD,
        os.fsencode(destination),
        _RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FeatureArtifactExistsError(f"destination appeared during publication: {destination}")
    raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_brca_q75_feature_artifacts(
    directory: str | Path,
    *,
    scale_2x_features: torch.Tensor,
    scale_4x_features: torch.Tensor,
    combined_features: torch.Tensor,
    row_provenance: Sequence[PatchProvenance],
    metadata: FeatureArtifactMetadata,
) -> FeatureArtifactSetRecord:
    """Atomically publish the exact Q75 feature set into an absent directory."""

    destination = _absolute_without_resolving(directory)
    _require(
        destination.name == EXPECTED_OUTPUT_BASENAME,
        f"Q75 destination basename must be {EXPECTED_OUTPUT_BASENAME}",
    )
    _require_no_symlink_components(destination.parent)
    _require_outside_git(destination)
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    parent_metadata = parent.lstat()
    _require(
        stat.S_ISDIR(parent_metadata.st_mode) and not parent.is_symlink(),
        "feature artifact parent must be a non-symlink directory",
    )
    if os.path.lexists(destination):
        raise FeatureArtifactExistsError(f"feature artifact destination already exists: {destination}")
    lock_path = parent / f".{destination.name}.lock"
    staging_prefix = f".{destination.name}.staging."
    if any(entry.name.startswith(staging_prefix) for entry in parent.iterdir()):
        raise FeaturePublicationInProgressError(f"stale or active feature staging path exists for {destination}")
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise FeaturePublicationInProgressError(f"feature publication is locked: {destination}") from exc

    staging: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "w", encoding="ascii") as lock_stream:
            lock_stream.write(f"pid={os.getpid()}\n")
            lock_stream.flush()
            os.fsync(lock_stream.fileno())
        if os.path.lexists(destination):
            raise FeatureArtifactExistsError(f"feature artifact destination appeared before staging: {destination}")
        _require(isinstance(metadata, FeatureArtifactMetadata), "metadata must be FeatureArtifactMetadata")
        tensors = {
            "scale_2x_features": _validate_feature_tensor(scale_2x_features, "scale_2x_features"),
            "scale_4x_features": _validate_feature_tensor(scale_4x_features, "scale_4x_features"),
            "combined_features": _validate_feature_tensor(combined_features, "combined_features"),
        }
        _validate_exact_concatenation(
            tensors["scale_2x_features"],
            tensors["scale_4x_features"],
            tensors["combined_features"],
        )
        provenance = _validate_provenance(row_provenance, metadata)
        staging = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=parent))
        artifact_manifest: dict[str, object] = {}
        for name in ("scale_2x_features", "scale_4x_features", "combined_features"):
            tensor = tensors[name]
            filename = FEATURE_FILENAMES[name]
            artifact_path = staging / filename
            _write_tensor(artifact_path, tensor)
            artifact_manifest[name] = {
                "content_serialization": TENSOR_CONTENT_SERIALIZATION,
                "device": "cpu",
                "dtype": "float32",
                "filename": filename,
                "memory_layout": "contiguous_C_order",
                "serialization": TENSOR_SERIALIZATION,
                "sha256": sha256_file(artifact_path),
                "shape": list(tensor.shape),
                "size_bytes": artifact_path.stat().st_size,
                "tensor_content_sha256": _tensor_content_sha256(tensor),
            }
        provenance_path = staging / PROVENANCE_FILENAME
        provenance_payload = _provenance_bytes(provenance)
        _write_exclusive(provenance_path, provenance_payload)
        coordinate_digests = {
            "scale_2x": _coordinate_sha256(provenance[:SCALE_2X_ROWS]),
            "scale_4x": _coordinate_sha256(provenance[SCALE_2X_ROWS:]),
        }
        artifact_manifest["row_provenance"] = {
            "columns": list(PROVENANCE_FIELDS),
            "coordinate_content_serialization": COORDINATE_CONTENT_SERIALIZATION,
            "coordinate_sha256_by_branch": coordinate_digests,
            "filename": PROVENANCE_FILENAME,
            "row_count": len(provenance),
            "sha256": hashlib.sha256(provenance_payload).hexdigest(),
            "size_bytes": len(provenance_payload),
        }
        document = {
            "artifacts": artifact_manifest,
            "contract": {
                "branch_order": list(BRANCH_ORDER),
                "combined_rows": SCALE_2X_ROWS + SCALE_4X_ROWS,
                "concatenation": "torch.cat([scale_2x,scale_4x],dim=0)",
                "device": "cpu",
                "dtype": "float32",
                "feature_dim": FEATURE_DIM,
                "finite": True,
                "memory_layout": "contiguous_C_order",
                "natural_model_shape": [1, SCALE_2X_ROWS + SCALE_4X_ROWS, FEATURE_DIM],
                "pooling_performed": False,
                "scale_2x_rows": SCALE_2X_ROWS,
                "scale_4x_rows": SCALE_4X_ROWS,
                "transpose_performed": False,
            },
            "metadata": metadata.to_manifest(),
            "schema": SCHEMA,
        }
        manifest_payload = _manifest_bytes(document)
        manifest_path = staging / MANIFEST_FILENAME
        _write_exclusive(manifest_path, manifest_payload)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        _write_exclusive(
            staging / MANIFEST_SHA256_FILENAME,
            f"{manifest_sha256}  {MANIFEST_FILENAME}\n".encode("ascii"),
        )
        _fsync_directory(staging)
        _validate_directory(staging, manifest_sha256)
        if os.path.lexists(destination):
            raise FeatureArtifactExistsError(f"feature artifact destination appeared during staging: {destination}")
        _rename_noreplace(staging, destination)
        staging = None
        _fsync_directory(parent)
        return _validate_directory(destination, manifest_sha256)
    finally:
        if staging is not None and os.path.lexists(staging):
            shutil.rmtree(staging)
        lock_path.unlink(missing_ok=True)
