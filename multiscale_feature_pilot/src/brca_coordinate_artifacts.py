"""Atomic publication and strict validation of BRCA coordinate artifacts.

This module performs no WSI I/O and no coordinate generation.  It accepts two
already-generated synthetic or real coordinate arrays, validates their locked
level-0 lattice contract, writes them into a sibling staging directory, and
publishes the complete directory with Linux ``RENAME_NOREPLACE``.  Existing
destinations and stale staging directories are never resumed or replaced.
"""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import h5py
import numpy as np
from numpy.typing import NDArray


SCHEMA: Final = "BRCA_COORDINATE_ARTIFACT_SET_V1"
COORDINATE_FRAME: Final = "level0_x_y"
COORDINATE_ORDERING: Final = "row_major_y_then_x"
LATTICE: Final = "custom_global_level0_lattice_v1"
MAPPING: Final = "GLOBAL_LEVEL0_CONSTANT_STEP"
SELECTION_CLAIM: Final = "CUSTOM_ENGINEERING_POLICY_NOT_PAPER_EXACT"
BRANCH_FILENAMES: Final = {
    "scale_2x": "scale_2x_coordinates.h5",
    "scale_4x": "scale_4x_coordinates.h5",
}
MANIFEST_FILENAME: Final = "coordinate_manifest.json"
MANIFEST_SHA256_FILENAME: Final = "coordinate_manifest.json.sha256"
_HASH_CHUNK_SIZE: Final = 8 * 1024 * 1024
_RENAME_NOREPLACE: Final = 1
_AT_FDCWD: Final = -100
_PATIENT_RE: Final = re.compile(r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}$")


class CoordinateArtifactError(RuntimeError):
    """Base class for coordinate publication and validation failures."""


class CoordinateValidationError(CoordinateArtifactError):
    """Raised when coordinates or provenance violate the locked schema."""


class CoordinateArtifactExistsError(CoordinateArtifactError):
    """Raised rather than overwriting or resuming an existing destination."""


class CoordinatePublicationInProgressError(CoordinateArtifactError):
    """Raised when a lock or stale sibling staging directory exists."""


class CoordinateHashMismatchError(CoordinateArtifactError):
    """Raised when a file or coordinate-content digest differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CoordinateValidationError(message)


def _positive_int_pair(value: object, label: str) -> tuple[int, int]:
    _require(isinstance(value, tuple) and len(value) == 2, f"{label} must be a pair")
    left, right = value
    _require(
        not isinstance(left, bool)
        and not isinstance(right, bool)
        and isinstance(left, int)
        and isinstance(right, int),
        f"{label} must contain integers",
    )
    _require(left > 0 and right > 0, f"{label} must be positive")
    return left, right


def _positive_float(value: object, label: str) -> float:
    _require(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0,
        f"{label} must be finite and positive",
    )
    return float(value)


def _digest(value: object, length: int, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a hexadecimal string")
    normalized = value.lower()
    _require(
        len(normalized) == length
        and all(character in "0123456789abcdef" for character in normalized),
        f"{label} must contain exactly {length} hexadecimal characters",
    )
    return normalized


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without loading it wholly into memory."""

    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(_HASH_CHUNK_SIZE), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _coordinates_sha256(coordinates: NDArray[np.int64]) -> str:
    canonical = np.ascontiguousarray(coordinates, dtype=np.dtype("<i8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class CoordinateBranchMetadata:
    """Complete, immutable provenance expected on one ``coords`` dataset."""

    branch: str
    patient_id: str
    slide_id: str
    gdc_file_uuid: str
    wsi_filename: str
    wsi_size_bytes: int
    wsi_md5: str
    wsi_sha256: str
    level_0_dimensions: tuple[int, int]
    source_level: int
    source_level_dimensions: tuple[int, int]
    openslide_reported_source_downsample: float
    source_patch_size: tuple[int, int]
    output_patch_size: tuple[int, int]
    level_0_declared_footprint: tuple[int, int]
    level_0_step: tuple[int, int]
    target_mpp: float
    effective_mpp: tuple[float, float]
    interpolation: str
    resampling: str
    mask_level: int
    mask_level_dimensions: tuple[int, int]
    openslide_reported_mask_downsample: float
    mask_image_channels: int
    mask_image_sha256: str
    mask_parameters: Mapping[str, object]
    contour_count: int
    retained_hole_count: int
    clam_commit: str
    policy_sha256: str
    geometry_compatibility: str

    def __post_init__(self) -> None:
        _require(self.branch in BRANCH_FILENAMES, "branch must be scale_2x or scale_4x")
        _require(
            isinstance(self.patient_id, str) and _PATIENT_RE.fullmatch(self.patient_id),
            "patient_id must be a canonical TCGA patient ID",
        )
        _require(isinstance(self.slide_id, str) and self.slide_id, "slide_id is required")
        try:
            parsed_uuid = str(uuid.UUID(self.gdc_file_uuid))
        except (ValueError, AttributeError) as exc:
            raise CoordinateValidationError("gdc_file_uuid must be canonical UUID") from exc
        _require(parsed_uuid == self.gdc_file_uuid, "gdc_file_uuid must be lowercase canonical UUID")
        _require(
            isinstance(self.wsi_filename, str)
            and Path(self.wsi_filename).name == self.wsi_filename
            and self.wsi_filename.lower().endswith(".svs"),
            "wsi_filename must be a basename ending in .svs",
        )
        _require(
            isinstance(self.wsi_size_bytes, int)
            and not isinstance(self.wsi_size_bytes, bool)
            and self.wsi_size_bytes > 0,
            "wsi_size_bytes must be positive",
        )
        _digest(self.wsi_md5, 32, "wsi_md5")
        _digest(self.wsi_sha256, 64, "wsi_sha256")
        level_0 = _positive_int_pair(self.level_0_dimensions, "level_0_dimensions")
        source_dimensions = _positive_int_pair(
            self.source_level_dimensions, "source_level_dimensions"
        )
        _require(
            isinstance(self.source_level, int)
            and not isinstance(self.source_level, bool)
            and self.source_level >= 0,
            "source_level must be a nonnegative integer",
        )
        _positive_float(
            self.openslide_reported_source_downsample,
            "openslide_reported_source_downsample",
        )
        source_patch = _positive_int_pair(self.source_patch_size, "source_patch_size")
        output_patch = _positive_int_pair(self.output_patch_size, "output_patch_size")
        declared = _positive_int_pair(
            self.level_0_declared_footprint, "level_0_declared_footprint"
        )
        step = _positive_int_pair(self.level_0_step, "level_0_step")
        source_scale = (
            level_0[0] / source_dimensions[0],
            level_0[1] / source_dimensions[1],
        )
        expected_declared = (
            source_patch[0] * int(source_scale[0]),
            source_patch[1] * int(source_scale[1]),
        )
        _require(
            declared == expected_declared,
            "level_0_declared_footprint must use source_patch*int(per-axis dimension ratio)",
        )
        _require(step == declared, "locked lattice must be non-overlapping (step=footprint)")
        _require(
            source_patch[0] <= source_dimensions[0]
            and source_patch[1] <= source_dimensions[1],
            "source patch exceeds its source level",
        )
        _positive_float(self.target_mpp, "target_mpp")
        _require(
            isinstance(self.effective_mpp, tuple) and len(self.effective_mpp) == 2,
            "effective_mpp must be an x/y pair",
        )
        _positive_float(self.effective_mpp[0], "effective_mpp_x")
        _positive_float(self.effective_mpp[1], "effective_mpp_y")
        for label, value in (
            ("interpolation", self.interpolation),
            ("resampling", self.resampling),
            ("geometry_compatibility", self.geometry_compatibility),
        ):
            _require(isinstance(value, str) and value.strip() == value and value, f"{label} is required")
        _require(
            isinstance(self.mask_level, int)
            and not isinstance(self.mask_level, bool)
            and self.mask_level >= 0,
            "mask_level must be nonnegative",
        )
        _positive_int_pair(self.mask_level_dimensions, "mask_level_dimensions")
        _positive_float(
            self.openslide_reported_mask_downsample,
            "openslide_reported_mask_downsample",
        )
        _require(
            isinstance(self.mask_image_channels, int)
            and not isinstance(self.mask_image_channels, bool)
            and self.mask_image_channels in (3, 4),
            "mask_image_channels must be 3 or 4",
        )
        _digest(self.mask_image_sha256, 64, "mask_image_sha256")
        _require(isinstance(self.mask_parameters, Mapping), "mask_parameters must be a mapping")
        try:
            mask_json = json.dumps(
                self.mask_parameters,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        except (TypeError, ValueError) as exc:
            raise CoordinateValidationError("mask_parameters must be strict JSON") from exc
        _require(json.loads(mask_json) != {}, "mask_parameters must not be empty")
        _require(
            isinstance(self.contour_count, int)
            and not isinstance(self.contour_count, bool)
            and self.contour_count > 0,
            "contour_count must be a positive integer",
        )
        _require(
            isinstance(self.retained_hole_count, int)
            and not isinstance(self.retained_hole_count, bool)
            and self.retained_hole_count >= 0,
            "retained_hole_count must be a nonnegative integer",
        )
        _digest(self.clam_commit, 40, "clam_commit")
        _digest(self.policy_sha256, 64, "policy_sha256")
        _require(output_patch[0] > 0 and output_patch[1] > 0, "output patch must be positive")

    @property
    def source_to_level_0_scale_xy(self) -> tuple[float, float]:
        return (
            self.level_0_dimensions[0] / self.source_level_dimensions[0],
            self.level_0_dimensions[1] / self.source_level_dimensions[1],
        )

    @property
    def mask_contour_scale_xy(self) -> tuple[float, float]:
        return (
            self.level_0_dimensions[0] / self.mask_level_dimensions[0],
            self.level_0_dimensions[1] / self.mask_level_dimensions[1],
        )

    def to_attributes(self) -> dict[str, object]:
        source_scale = self.source_to_level_0_scale_xy
        mask_scale = self.mask_contour_scale_xy
        mask_json = json.dumps(
            self.mask_parameters,
            allow_nan=False,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return {
            "schema": SCHEMA,
            "branch": self.branch,
            "patient_id": self.patient_id,
            "slide_id": self.slide_id,
            "gdc_file_uuid": self.gdc_file_uuid,
            "wsi_filename": self.wsi_filename,
            "wsi_size_bytes": self.wsi_size_bytes,
            "wsi_md5": self.wsi_md5,
            "wsi_sha256": self.wsi_sha256,
            "coordinate_frame": COORDINATE_FRAME,
            "coordinate_ordering": COORDINATE_ORDERING,
            "source_level": self.source_level,
            "level_0_width": self.level_0_dimensions[0],
            "level_0_height": self.level_0_dimensions[1],
            "source_level_width": self.source_level_dimensions[0],
            "source_level_height": self.source_level_dimensions[1],
            "openslide_reported_source_downsample": float(
                self.openslide_reported_source_downsample
            ),
            "source_to_level_0_scale_x": source_scale[0],
            "source_to_level_0_scale_y": source_scale[1],
            "source_patch_width": self.source_patch_size[0],
            "source_patch_height": self.source_patch_size[1],
            "output_patch_width": self.output_patch_size[0],
            "output_patch_height": self.output_patch_size[1],
            "level_0_declared_footprint_width": self.level_0_declared_footprint[0],
            "level_0_declared_footprint_height": self.level_0_declared_footprint[1],
            "level_0_step_x": self.level_0_step[0],
            "level_0_step_y": self.level_0_step[1],
            "lattice_anchor_x": 0,
            "lattice_anchor_y": 0,
            "lattice": LATTICE,
            "coordinate_mapping": MAPPING,
            "target_mpp": float(self.target_mpp),
            "effective_mpp_x": float(self.effective_mpp[0]),
            "effective_mpp_y": float(self.effective_mpp[1]),
            "interpolation": self.interpolation,
            "resampling": self.resampling,
            "mask_level": self.mask_level,
            "mask_level_width": self.mask_level_dimensions[0],
            "mask_level_height": self.mask_level_dimensions[1],
            "openslide_reported_mask_downsample": float(
                self.openslide_reported_mask_downsample
            ),
            "mask_image_channels": self.mask_image_channels,
            "mask_image_sha256": self.mask_image_sha256,
            "mask_image_dtype": "uint8",
            "mask_image_channel_order": (
                "RGBA" if self.mask_image_channels == 4 else "RGB"
            ),
            "mask_image_hash_serialization": "contiguous_uint8_C_order_raw_bytes",
            "mask_contour_scale_x": mask_scale[0],
            "mask_contour_scale_y": mask_scale[1],
            "mask_parameters_json": mask_json,
            "contour_count": self.contour_count,
            "retained_hole_count": self.retained_hole_count,
            "clam_commit": self.clam_commit,
            "policy_sha256": self.policy_sha256,
            "geometry_compatibility": self.geometry_compatibility,
            "coordinate_selection_claim": SELECTION_CLAIM,
        }


@dataclass(frozen=True)
class CoordinateBranchRecord:
    branch: str
    path: Path
    coordinate_count: int
    size_bytes: int
    sha256: str
    coordinates_sha256: str
    metadata: CoordinateBranchMetadata


@dataclass(frozen=True)
class CoordinateArtifactSetRecord:
    directory: Path
    manifest_path: Path
    manifest_sha256_path: Path
    manifest_sha256: str
    branches: tuple[CoordinateBranchRecord, CoordinateBranchRecord]

    def branch_for(self, branch: str) -> CoordinateBranchRecord:
        matches = tuple(record for record in self.branches if record.branch == branch)
        if len(matches) != 1:
            raise CoordinateValidationError(f"artifact set lacks exactly one {branch} branch")
        return matches[0]


def _validate_coordinates(
    value: object,
    metadata: CoordinateBranchMetadata,
) -> NDArray[np.int64]:
    _require(isinstance(value, np.ndarray), f"{metadata.branch} coordinates must be ndarray")
    coordinates = value
    _require(coordinates.dtype == np.dtype(np.int64), f"{metadata.branch} dtype must be int64")
    _require(
        coordinates.ndim == 2 and coordinates.shape[1] == 2,
        f"{metadata.branch} coordinates must have shape [N,2]",
    )
    _require(coordinates.shape[0] > 0, f"{metadata.branch} coordinates must be nonempty")
    x = coordinates[:, 0]
    y = coordinates[:, 1]
    _require(np.all(x >= 0) and np.all(y >= 0), f"{metadata.branch} coordinates must be nonnegative")
    if coordinates.shape[0] > 1:
        strictly_row_major = (y[1:] > y[:-1]) | (
            (y[1:] == y[:-1]) & (x[1:] > x[:-1])
        )
        _require(
            bool(np.all(strictly_row_major)),
            f"{metadata.branch} coordinates must be unique row-major (y,x)",
        )
    footprint_x, footprint_y = metadata.level_0_declared_footprint
    width, height = metadata.level_0_dimensions
    _require(
        bool(np.all(x <= width - footprint_x))
        and bool(np.all(y <= height - footprint_y)),
        f"{metadata.branch} contains an incomplete level-0 footprint",
    )
    step_x, step_y = metadata.level_0_step
    _require(
        bool(np.all(x % step_x == 0)) and bool(np.all(y % step_y == 0)),
        f"{metadata.branch} coordinate is off the global level-0 lattice",
    )
    return np.ascontiguousarray(coordinates)


def _require_same_slide(
    scale_2x: CoordinateBranchMetadata,
    scale_4x: CoordinateBranchMetadata,
) -> None:
    common = (
        "patient_id",
        "slide_id",
        "gdc_file_uuid",
        "wsi_filename",
        "wsi_size_bytes",
        "wsi_md5",
        "wsi_sha256",
        "level_0_dimensions",
        "mask_level",
        "mask_level_dimensions",
        "openslide_reported_mask_downsample",
        "mask_image_channels",
        "mask_image_sha256",
        "contour_count",
        "retained_hole_count",
        "clam_commit",
        "policy_sha256",
    )
    for field_name in common:
        _require(
            getattr(scale_2x, field_name) == getattr(scale_4x, field_name),
            f"branch metadata differs for shared field {field_name}",
        )
    _require(
        scale_2x.to_attributes()["mask_parameters_json"]
        == scale_4x.to_attributes()["mask_parameters_json"],
        "branch mask parameters differ",
    )


def _write_h5(
    path: Path,
    coordinates: NDArray[np.int64],
    metadata: CoordinateBranchMetadata,
) -> None:
    with h5py.File(path, "x") as h5:
        dataset = h5.create_dataset(
            "coords",
            data=coordinates,
            dtype=np.int64,
            track_times=False,
        )
        for key, value in metadata.to_attributes().items():
            dataset.attrs[key] = value
        h5.flush()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
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
        raise CoordinateValidationError(f"missing {label}: {path}") from exc
    _require(stat.S_ISREG(metadata.st_mode) and not path.is_symlink(), f"{label} must be a regular non-symlink file")


def _normalise_h5_attribute(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _metadata_from_attributes(attributes: Mapping[str, object]) -> CoordinateBranchMetadata:
    def pair(left: str, right: str) -> tuple[int, int]:
        return (int(attributes[left]), int(attributes[right]))

    metadata = CoordinateBranchMetadata(
        branch=str(attributes["branch"]),
        patient_id=str(attributes["patient_id"]),
        slide_id=str(attributes["slide_id"]),
        gdc_file_uuid=str(attributes["gdc_file_uuid"]),
        wsi_filename=str(attributes["wsi_filename"]),
        wsi_size_bytes=int(attributes["wsi_size_bytes"]),
        wsi_md5=str(attributes["wsi_md5"]),
        wsi_sha256=str(attributes["wsi_sha256"]),
        level_0_dimensions=pair("level_0_width", "level_0_height"),
        source_level=int(attributes["source_level"]),
        source_level_dimensions=pair("source_level_width", "source_level_height"),
        openslide_reported_source_downsample=float(
            attributes["openslide_reported_source_downsample"]
        ),
        source_patch_size=pair("source_patch_width", "source_patch_height"),
        output_patch_size=pair("output_patch_width", "output_patch_height"),
        level_0_declared_footprint=pair(
            "level_0_declared_footprint_width",
            "level_0_declared_footprint_height",
        ),
        level_0_step=pair("level_0_step_x", "level_0_step_y"),
        target_mpp=float(attributes["target_mpp"]),
        effective_mpp=(
            float(attributes["effective_mpp_x"]),
            float(attributes["effective_mpp_y"]),
        ),
        interpolation=str(attributes["interpolation"]),
        resampling=str(attributes["resampling"]),
        mask_level=int(attributes["mask_level"]),
        mask_level_dimensions=pair("mask_level_width", "mask_level_height"),
        openslide_reported_mask_downsample=float(
            attributes["openslide_reported_mask_downsample"]
        ),
        mask_image_channels=int(attributes["mask_image_channels"]),
        mask_image_sha256=str(attributes["mask_image_sha256"]),
        mask_parameters=json.loads(str(attributes["mask_parameters_json"])),
        contour_count=int(attributes["contour_count"]),
        retained_hole_count=int(attributes["retained_hole_count"]),
        clam_commit=str(attributes["clam_commit"]),
        policy_sha256=str(attributes["policy_sha256"]),
        geometry_compatibility=str(attributes["geometry_compatibility"]),
    )
    expected = metadata.to_attributes()
    _require(dict(attributes) == expected, "coordinate dataset attributes drift from strict schema")
    return metadata


def _manifest_bytes(document: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            document,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _validate_directory(
    directory: Path,
    expected_manifest_sha256: str,
) -> CoordinateArtifactSetRecord:
    expected_manifest_sha256 = _digest(
        expected_manifest_sha256, 64, "expected_manifest_sha256"
    )
    try:
        directory_metadata = directory.lstat()
    except FileNotFoundError as exc:
        raise CoordinateValidationError(f"artifact directory is missing: {directory}") from exc
    _require(
        stat.S_ISDIR(directory_metadata.st_mode) and not directory.is_symlink(),
        "artifact destination must be a non-symlink directory",
    )
    expected_names = set(BRANCH_FILENAMES.values()) | {
        MANIFEST_FILENAME,
        MANIFEST_SHA256_FILENAME,
    }
    actual_names = {entry.name for entry in directory.iterdir()}
    _require(actual_names == expected_names, "artifact directory file set is not exact")

    manifest_path = directory / MANIFEST_FILENAME
    sidecar_path = directory / MANIFEST_SHA256_FILENAME
    _regular_file(manifest_path, "coordinate manifest")
    _regular_file(sidecar_path, "manifest SHA-256 sidecar")
    actual_manifest_sha256 = sha256_file(manifest_path)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise CoordinateHashMismatchError(
            f"manifest SHA-256 mismatch: expected {expected_manifest_sha256}, got {actual_manifest_sha256}"
        )
    expected_sidecar = f"{expected_manifest_sha256}  {MANIFEST_FILENAME}\n"
    try:
        sidecar = sidecar_path.read_text(encoding="ascii")
    except UnicodeError as exc:
        raise CoordinateValidationError("manifest sidecar is not ASCII") from exc
    if sidecar != expected_sidecar:
        raise CoordinateHashMismatchError("manifest SHA-256 sidecar mismatch")
    try:
        document = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CoordinateValidationError("coordinate manifest is not valid JSON") from exc
    _require(isinstance(document, dict), "coordinate manifest root must be an object")
    _require(set(document) == {"schema", "branches"}, "coordinate manifest keys drift")
    _require(document["schema"] == SCHEMA, "coordinate manifest schema drift")
    branches = document["branches"]
    _require(isinstance(branches, dict), "coordinate manifest branches must be an object")
    _require(set(branches) == set(BRANCH_FILENAMES), "coordinate manifest branch set drift")

    records: list[CoordinateBranchRecord] = []
    for branch in ("scale_2x", "scale_4x"):
        entry = branches[branch]
        _require(isinstance(entry, dict), f"{branch} manifest entry must be an object")
        _require(
            set(entry)
            == {
                "filename",
                "size_bytes",
                "sha256",
                "coordinates_sha256",
                "coordinate_count",
                "attributes",
            },
            f"{branch} manifest entry keys drift",
        )
        filename = BRANCH_FILENAMES[branch]
        _require(entry["filename"] == filename, f"{branch} filename drift")
        artifact_path = directory / filename
        _regular_file(artifact_path, f"{branch} artifact")
        actual_sha256 = sha256_file(artifact_path)
        expected_h5_sha256 = _digest(entry["sha256"], 64, f"{branch} sha256")
        if actual_sha256 != expected_h5_sha256:
            raise CoordinateHashMismatchError(f"{branch} HDF5 SHA-256 mismatch")
        _require(
            isinstance(entry["size_bytes"], int)
            and not isinstance(entry["size_bytes"], bool)
            and entry["size_bytes"] == artifact_path.stat().st_size,
            f"{branch} size mismatch",
        )
        _require(isinstance(entry["attributes"], dict), f"{branch} attributes must be an object")
        manifest_attributes = dict(entry["attributes"])
        with h5py.File(artifact_path, "r") as h5:
            _require(list(h5.keys()) == ["coords"], f"{branch} HDF5 must contain only coords")
            _require(len(h5.attrs) == 0, f"{branch} HDF5 root attributes are prohibited")
            dataset = h5["coords"]
            _require(isinstance(dataset, h5py.Dataset), f"{branch} coords must be a dataset")
            _require(dataset.dtype == np.dtype(np.int64), f"{branch} HDF5 dtype must be int64")
            attributes = {
                key: _normalise_h5_attribute(dataset.attrs[key])
                for key in dataset.attrs.keys()
            }
            _require(attributes == manifest_attributes, f"{branch} HDF5/manifest attributes differ")
            metadata = _metadata_from_attributes(attributes)
            _require(metadata.branch == branch, f"{branch} metadata branch drift")
            coordinates = np.asarray(dataset[...])
        coordinates = _validate_coordinates(coordinates, metadata)
        _require(
            isinstance(entry["coordinate_count"], int)
            and not isinstance(entry["coordinate_count"], bool)
            and entry["coordinate_count"] == coordinates.shape[0],
            f"{branch} coordinate count mismatch",
        )
        actual_coordinates_sha256 = _coordinates_sha256(coordinates)
        expected_coordinates_sha256 = _digest(
            entry["coordinates_sha256"], 64, f"{branch} coordinates_sha256"
        )
        if actual_coordinates_sha256 != expected_coordinates_sha256:
            raise CoordinateHashMismatchError(f"{branch} coordinate-content SHA-256 mismatch")
        records.append(
            CoordinateBranchRecord(
                branch=branch,
                path=artifact_path,
                coordinate_count=coordinates.shape[0],
                size_bytes=artifact_path.stat().st_size,
                sha256=actual_sha256,
                coordinates_sha256=actual_coordinates_sha256,
                metadata=metadata,
            )
        )
    _require_same_slide(records[0].metadata, records[1].metadata)
    return CoordinateArtifactSetRecord(
        directory=directory,
        manifest_path=manifest_path,
        manifest_sha256_path=sidecar_path,
        manifest_sha256=actual_manifest_sha256,
        branches=(records[0], records[1]),
    )


def validate_brca_coordinate_artifacts(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
) -> CoordinateArtifactSetRecord:
    """Validate an exact, externally anchored artifact set read-only."""

    return _validate_directory(Path(directory), expected_manifest_sha256)


def _rename_noreplace(source: Path, destination: Path) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise CoordinateArtifactError(
            "atomic RENAME_NOREPLACE is unavailable; refusing non-atomic publication"
        )
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
        raise CoordinateArtifactExistsError(
            f"destination appeared during publication: {destination}"
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def publish_brca_coordinate_artifacts(
    directory: str | Path,
    *,
    scale_2x_coordinates: NDArray[np.int64],
    scale_4x_coordinates: NDArray[np.int64],
    scale_2x_metadata: CoordinateBranchMetadata,
    scale_4x_metadata: CoordinateBranchMetadata,
) -> CoordinateArtifactSetRecord:
    """Publish one complete two-branch set into an absent final directory.

    The destination cannot be reused, resumed, or overwritten.  A caller must
    retain the returned ``manifest_sha256`` as the external validation anchor.
    """

    destination = Path(directory)
    if not destination.name:
        raise CoordinateValidationError("artifact destination must have a basename")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(destination):
        raise CoordinateArtifactExistsError(f"artifact destination already exists: {destination}")
    lock_path = parent / f".{destination.name}.lock"
    staging_prefix = f".{destination.name}.staging."
    stale = tuple(entry for entry in parent.iterdir() if entry.name.startswith(staging_prefix))
    if stale:
        raise CoordinatePublicationInProgressError(
            f"stale or active staging directory exists for {destination}"
        )
    try:
        lock_descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise CoordinatePublicationInProgressError(
            f"coordinate publication is locked: {destination}"
        ) from exc

    staging: Path | None = None
    try:
        with os.fdopen(lock_descriptor, "w", encoding="ascii") as lock_stream:
            lock_stream.write(f"pid={os.getpid()}\n")
            lock_stream.flush()
            os.fsync(lock_stream.fileno())
        if os.path.lexists(destination):
            raise CoordinateArtifactExistsError(
                f"artifact destination appeared before staging: {destination}"
            )
        _require(scale_2x_metadata.branch == "scale_2x", "scale_2x metadata branch drift")
        _require(scale_4x_metadata.branch == "scale_4x", "scale_4x metadata branch drift")
        _require_same_slide(scale_2x_metadata, scale_4x_metadata)
        coordinate_inputs = {
            "scale_2x": _validate_coordinates(scale_2x_coordinates, scale_2x_metadata),
            "scale_4x": _validate_coordinates(scale_4x_coordinates, scale_4x_metadata),
        }
        metadata_inputs = {
            "scale_2x": scale_2x_metadata,
            "scale_4x": scale_4x_metadata,
        }
        staging = Path(tempfile.mkdtemp(prefix=staging_prefix, dir=parent))
        manifest_branches: dict[str, object] = {}
        for branch in ("scale_2x", "scale_4x"):
            filename = BRANCH_FILENAMES[branch]
            artifact_path = staging / filename
            coordinates = coordinate_inputs[branch]
            metadata = metadata_inputs[branch]
            _write_h5(artifact_path, coordinates, metadata)
            _fsync_file(artifact_path)
            manifest_branches[branch] = {
                "filename": filename,
                "size_bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
                "coordinates_sha256": _coordinates_sha256(coordinates),
                "coordinate_count": int(coordinates.shape[0]),
                "attributes": metadata.to_attributes(),
            }
        manifest_path = staging / MANIFEST_FILENAME
        manifest_payload = _manifest_bytes(
            {"schema": SCHEMA, "branches": manifest_branches}
        )
        manifest_path.write_bytes(manifest_payload)
        manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
        sidecar_path = staging / MANIFEST_SHA256_FILENAME
        sidecar_path.write_text(
            f"{manifest_sha256}  {MANIFEST_FILENAME}\n",
            encoding="ascii",
            newline="",
        )
        _fsync_file(manifest_path)
        _fsync_file(sidecar_path)
        _fsync_directory(staging)
        _validate_directory(staging, manifest_sha256)
        if os.path.lexists(destination):
            raise CoordinateArtifactExistsError(
                f"artifact destination appeared during staging: {destination}"
            )
        _rename_noreplace(staging, destination)
        staging = None
        _fsync_directory(parent)
        return _validate_directory(destination, manifest_sha256)
    finally:
        if staging is not None and os.path.lexists(staging):
            shutil.rmtree(staging)
        lock_path.unlink(missing_ok=True)
