"""Fail-closed BRCA Q50 patch reads for the authorized GPU pilot.

The caller must first validate the complete coordinate directory against
``Q50_COORDINATE_MANIFEST_SHA256``.  This module then independently validates
each returned branch record, loads immutable level-0 origins, and exposes a
streaming dataset.  Constructing the dataset performs no pixel read; a worker
opens the exact hash-gated Q50 pathname only when an item is requested.

The reviewed Q50 pixel policy is explicit here.  ``scale_2x`` reads a 512x512
region from level 0 and resizes it to 256x256 with PIL Lanczos.  ``scale_4x``
reads a native 256x256 region from level 1.  Both RGB patches are passed to the
shared ImageNet preprocessing helper without local normalization drift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import hashlib
import os
from pathlib import Path
import stat
from typing import Literal

import h5py
import numpy as np
import openslide
from PIL import Image
import torch
from torch.utils.data import Dataset

from .brca_coordinate_artifacts import (
    BRANCH_FILENAMES,
    CoordinateBranchRecord,
    sha256_file,
)
from .feature_extraction import prepare_patch_for_resnet


Q50_PATIENT_ID = "TCGA-AR-A1AW"
Q50_SLIDE_ID = (
    "TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs"
)
Q50_GDC_FILE_UUID = "5c1216f3-19ec-4d3c-9bb0-9bd740b79f62"
Q50_WSI_FILENAME = Q50_SLIDE_ID
Q50_WSI_SIZE_BYTES = 975_626_387
Q50_WSI_MD5 = "304509e03f26cbecc9aee4ea691c8e5a"
Q50_WSI_SHA256 = (
    "6b960db7e6e5772f4f626daa7b023ffbb5e3b20f9f6ff5c7d52f4554cf33399b"
)
Q50_COORDINATE_MANIFEST_SHA256 = (
    "a79179e7fe3a0cf6bdda5089da11910c0db6b1f4bcab9a15c6d5b8e7ec43e485"
)
Q50_COORDINATE_POLICY_SHA256 = (
    "e5cb83739d3d8fab04da8a63ae1560df04ccc547a79512c219eecb575c0c2114"
)
Q50_MASK_IMAGE_SHA256 = (
    "86434ebab952209699798c3129d3ee8fbd8126dffd34705ead0d478c29e77601"
)
Q50_CLAM_COMMIT = "26e0b6c4873e112f1ccd74cd834894c4ab7a2934"
Q50_LEVEL_0_DIMENSIONS = (99_960, 65_334)
Q50_MASK_PARAMETERS: Mapping[str, object] = {
    "a_h": 16,
    "a_t": 100,
    "close": 4,
    "color_space": "HSV_saturation",
    "contour_rule": "pinned_four_pt_easy_any_probe_on_or_inside",
    "hole_rule": "reject_only_when_mapped_patch_center_is_strictly_inside",
    "max_n_holes": 8,
    "mthresh": 7,
    "reference_patch_size": 512,
    "sthresh": 8,
    "use_otsu": False,
}


class Q50FeatureExtractionContractError(ValueError):
    """Raised before a Q50 coordinate can drive a WSI pixel read."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Q50FeatureExtractionContractError(message)


@dataclass(frozen=True, slots=True)
class Q50WSIFileIdentity:
    """Stable leaf identity captured immediately after exact hash preflight."""

    st_dev: int
    st_ino: int
    st_size: int
    st_mtime_ns: int

    def __post_init__(self) -> None:
        for label, value in (
            ("st_dev", self.st_dev),
            ("st_ino", self.st_ino),
            ("st_size", self.st_size),
            ("st_mtime_ns", self.st_mtime_ns),
        ):
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"Q50 WSI {label} must be a nonnegative integer",
            )
        _require(
            self.st_size == Q50_WSI_SIZE_BYTES,
            "Q50 WSI identity size differs from the hash-gated payload",
        )


@dataclass(frozen=True, slots=True)
class _ExpectedBranch:
    coordinate_count: int
    source_level: int
    source_level_dimensions: tuple[int, int]
    openslide_reported_source_downsample: float
    read_size: tuple[int, int]
    branch_output_size: tuple[int, int]
    level_0_footprint: tuple[int, int]
    level_0_step: tuple[int, int]
    target_mpp: float
    effective_mpp: tuple[float, float]
    interpolation: str
    resampling: str
    geometry_compatibility: str


_EXPECTED_BRANCHES: Mapping[str, _ExpectedBranch] = {
    "scale_2x": _ExpectedBranch(
        coordinate_count=8_580,
        source_level=0,
        source_level_dimensions=(99_960, 65_334),
        openslide_reported_source_downsample=1.0,
        read_size=(512, 512),
        branch_output_size=(256, 256),
        level_0_footprint=(512, 512),
        level_0_step=(512, 512),
        target_mpp=0.5,
        effective_mpp=(0.4936, 0.4936),
        interpolation="PIL.Image.Resampling.LANCZOS",
        resampling="explicit_2x_spatial_downsample",
        geometry_compatibility="LEVEL0_IDENTITY_GEOMETRY",
    ),
    "scale_4x": _ExpectedBranch(
        coordinate_count=2_213,
        source_level=1,
        source_level_dimensions=(24_990, 16_333),
        openslide_reported_source_downsample=4.000061225739301,
        read_size=(256, 256),
        branch_output_size=(256, 256),
        level_0_footprint=(1_024, 1_024),
        level_0_step=(1_024, 1_024),
        target_mpp=1.0,
        effective_mpp=(0.9872151105124595, 0.9872151105124595),
        interpolation="none",
        resampling="none",
        geometry_compatibility="CLAM_INT_CAST_GEOMETRY_COMPATIBLE",
    ),
}


def _lexical_absolute_path(path: str | Path) -> Path:
    """Return an absolute path without resolving or following a symlink."""

    return Path(os.path.abspath(os.fspath(path)))


def _lstat_regular_non_symlink(path: Path) -> os.stat_result:
    try:
        metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Q50FeatureExtractionContractError(f"Q50 WSI is missing: {path}") from exc
    _require(
        stat.S_ISREG(metadata.st_mode) and not path.is_symlink(),
        "Q50 WSI must be a regular non-symlink file",
    )
    return metadata


def _identity_from_stat(metadata: os.stat_result) -> Q50WSIFileIdentity:
    return Q50WSIFileIdentity(
        st_dev=int(metadata.st_dev),
        st_ino=int(metadata.st_ino),
        st_size=int(metadata.st_size),
        st_mtime_ns=int(metadata.st_mtime_ns),
    )


def capture_q50_wsi_file_identity(wsi_path: str | Path) -> Q50WSIFileIdentity:
    """Capture the Q50 leaf token after the runner's exact hash preflight.

    This helper deliberately does not hash the nearly 1 GB payload again.  The
    runner must verify the pinned size, MD5, and SHA-256 first, then capture
    this token before permitting any mutation or constructing a dataset.
    """

    path = _lexical_absolute_path(wsi_path)
    _require(path.name == Q50_WSI_FILENAME, "Q50 WSI filename mismatch")
    return _identity_from_stat(_lstat_regular_non_symlink(path))


def _normalise_hdf5_attribute(value: object) -> object:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _coordinates_sha256(coordinates: np.ndarray) -> str:
    canonical = np.ascontiguousarray(coordinates, dtype=np.dtype("<i8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _validate_digest(value: object, *, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a SHA-256 string")
    digest = value.lower()
    _require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} must contain exactly 64 hexadecimal characters",
    )
    return digest


def _validate_coordinates(
    coordinates: torch.Tensor,
    *,
    branch: str,
    expected: _ExpectedBranch,
    expected_content_sha256: str,
) -> None:
    _require(isinstance(coordinates, torch.Tensor), f"{branch}: coordinates must be a tensor")
    _require(coordinates.layout is torch.strided, f"{branch}: coordinates must be dense")
    _require(coordinates.device.type == "cpu", f"{branch}: coordinates must remain on CPU")
    _require(coordinates.dtype is torch.int64, f"{branch}: coordinates must be int64")
    _require(coordinates.is_contiguous(), f"{branch}: coordinates must be contiguous")
    _require(
        tuple(coordinates.shape) == (expected.coordinate_count, 2),
        f"{branch}: coordinates must have shape [{expected.coordinate_count},2]",
    )
    _require(bool(torch.isfinite(coordinates).all().item()), f"{branch}: coordinates must be finite")

    x = coordinates[:, 0]
    y = coordinates[:, 1]
    _require(
        bool((x >= 0).all().item()) and bool((y >= 0).all().item()),
        f"{branch}: coordinates must be nonnegative",
    )
    if coordinates.shape[0] > 1:
        ordered = (y[1:] > y[:-1]) | ((y[1:] == y[:-1]) & (x[1:] > x[:-1]))
        _require(bool(ordered.all().item()), f"{branch}: coordinates must be unique row-major (y,x)")

    width, height = Q50_LEVEL_0_DIMENSIONS
    footprint_x, footprint_y = expected.level_0_footprint
    _require(
        bool((x <= width - footprint_x).all().item())
        and bool((y <= height - footprint_y).all().item()),
        f"{branch}: coordinate contains an incomplete level-0 footprint",
    )
    step_x, step_y = expected.level_0_step
    _require(
        bool((x.remainder(step_x) == 0).all().item())
        and bool((y.remainder(step_y) == 0).all().item()),
        f"{branch}: coordinate is off the global level-0 lattice",
    )
    _require(
        _coordinates_sha256(coordinates.numpy()) == expected_content_sha256,
        f"{branch}: coordinate-content SHA-256 mismatch",
    )


@dataclass(frozen=True, slots=True)
class Q50BranchReadSpec:
    """One exact, validated Q50 branch read plan with level-0 origins."""

    branch: Literal["scale_2x", "scale_4x"]
    coordinates: torch.Tensor
    coordinate_hdf5_path: Path
    coordinate_hdf5_size_bytes: int
    coordinate_hdf5_sha256: str
    coordinate_content_sha256: str
    source_level: int
    read_size: tuple[int, int]
    branch_output_size: tuple[int, int]
    level_0_dimensions: tuple[int, int]
    level_0_footprint: tuple[int, int]
    level_0_step: tuple[int, int]
    target_mpp: float
    effective_mpp: tuple[float, float]
    interpolation: str
    resampling: str

    def __post_init__(self) -> None:
        _require(
            isinstance(self.branch, str) and self.branch in _EXPECTED_BRANCHES,
            "branch must be scale_2x or scale_4x",
        )
        expected = _EXPECTED_BRANCHES[self.branch]
        checks = {
            "source level": (self.source_level, expected.source_level),
            "read size": (self.read_size, expected.read_size),
            "output size": (self.branch_output_size, expected.branch_output_size),
            "level-0 dimensions": (self.level_0_dimensions, Q50_LEVEL_0_DIMENSIONS),
            "footprint": (self.level_0_footprint, expected.level_0_footprint),
            "lattice step": (self.level_0_step, expected.level_0_step),
            "target MPP": (self.target_mpp, expected.target_mpp),
            "effective MPP": (self.effective_mpp, expected.effective_mpp),
            "interpolation": (self.interpolation, expected.interpolation),
            "resampling": (self.resampling, expected.resampling),
        }
        for label, (actual, wanted) in checks.items():
            _require(actual == wanted, f"{self.branch}: {label} drift")
        _require(isinstance(self.coordinate_hdf5_path, Path), f"{self.branch}: coordinate path must be pathlib.Path")
        _require(
            isinstance(self.coordinate_hdf5_size_bytes, int)
            and not isinstance(self.coordinate_hdf5_size_bytes, bool)
            and self.coordinate_hdf5_size_bytes > 0,
            f"{self.branch}: coordinate HDF5 size must be positive",
        )
        hdf5_sha256 = _validate_digest(
            self.coordinate_hdf5_sha256,
            label=f"{self.branch} coordinate HDF5 SHA-256",
        )
        content_sha256 = _validate_digest(
            self.coordinate_content_sha256,
            label=f"{self.branch} coordinate-content SHA-256",
        )
        _validate_coordinates(
            self.coordinates,
            branch=self.branch,
            expected=expected,
            expected_content_sha256=content_sha256,
        )
        object.__setattr__(self, "coordinate_hdf5_sha256", hdf5_sha256)
        object.__setattr__(self, "coordinate_content_sha256", content_sha256)

    @property
    def coordinate_count(self) -> int:
        return int(self.coordinates.shape[0])


def _validate_q50_metadata(record: CoordinateBranchRecord) -> _ExpectedBranch:
    branch = record.branch
    _require(branch in _EXPECTED_BRANCHES, "coordinate record branch is unsupported")
    expected = _EXPECTED_BRANCHES[branch]
    metadata = record.metadata
    checks = {
        "branch": (metadata.branch, branch),
        "patient_id": (metadata.patient_id, Q50_PATIENT_ID),
        "slide_id": (metadata.slide_id, Q50_SLIDE_ID),
        "gdc_file_uuid": (metadata.gdc_file_uuid, Q50_GDC_FILE_UUID),
        "wsi_filename": (metadata.wsi_filename, Q50_WSI_FILENAME),
        "wsi_size_bytes": (metadata.wsi_size_bytes, Q50_WSI_SIZE_BYTES),
        "wsi_md5": (metadata.wsi_md5, Q50_WSI_MD5),
        "wsi_sha256": (metadata.wsi_sha256, Q50_WSI_SHA256),
        "level_0_dimensions": (metadata.level_0_dimensions, Q50_LEVEL_0_DIMENSIONS),
        "source_level": (metadata.source_level, expected.source_level),
        "source_level_dimensions": (metadata.source_level_dimensions, expected.source_level_dimensions),
        "openslide_reported_source_downsample": (
            metadata.openslide_reported_source_downsample,
            expected.openslide_reported_source_downsample,
        ),
        "source_patch_size": (metadata.source_patch_size, expected.read_size),
        "output_patch_size": (metadata.output_patch_size, expected.branch_output_size),
        "level_0_declared_footprint": (
            metadata.level_0_declared_footprint,
            expected.level_0_footprint,
        ),
        "level_0_step": (metadata.level_0_step, expected.level_0_step),
        "target_mpp": (metadata.target_mpp, expected.target_mpp),
        "effective_mpp": (metadata.effective_mpp, expected.effective_mpp),
        "interpolation": (metadata.interpolation, expected.interpolation),
        "resampling": (metadata.resampling, expected.resampling),
        "mask_level": (metadata.mask_level, 2),
        "mask_level_dimensions": (metadata.mask_level_dimensions, (6_247, 4_083)),
        "openslide_reported_mask_downsample": (
            metadata.openslide_reported_mask_downsample,
            16.001375061204985,
        ),
        "mask_image_channels": (metadata.mask_image_channels, 4),
        "mask_image_sha256": (metadata.mask_image_sha256, Q50_MASK_IMAGE_SHA256),
        "mask_parameters": (dict(metadata.mask_parameters), dict(Q50_MASK_PARAMETERS)),
        "contour_count": (metadata.contour_count, 4),
        "retained_hole_count": (metadata.retained_hole_count, 2),
        "clam_commit": (metadata.clam_commit, Q50_CLAM_COMMIT),
        "policy_sha256": (metadata.policy_sha256, Q50_COORDINATE_POLICY_SHA256),
        "geometry_compatibility": (metadata.geometry_compatibility, expected.geometry_compatibility),
    }
    for label, (actual, wanted) in checks.items():
        _require(actual == wanted, f"{branch}: {label} drift")
    _require(record.coordinate_count == expected.coordinate_count, f"{branch}: coordinate count drift")
    return expected


def load_q50_branch_read_spec(record: CoordinateBranchRecord) -> Q50BranchReadSpec:
    """Load one branch returned by the externally anchored strict validator."""

    _require(
        isinstance(record, CoordinateBranchRecord),
        "record must be CoordinateBranchRecord from the strict validator",
    )
    expected = _validate_q50_metadata(record)
    path = record.path
    _require(isinstance(path, Path), f"{record.branch}: coordinate path must be Path")
    try:
        file_metadata = path.lstat()
    except FileNotFoundError as exc:
        raise Q50FeatureExtractionContractError(
            f"{record.branch}: coordinate HDF5 is missing: {path}"
        ) from exc
    _require(
        stat.S_ISREG(file_metadata.st_mode) and not path.is_symlink(),
        f"{record.branch}: coordinate HDF5 must be a regular non-symlink file",
    )
    _require(path.name == BRANCH_FILENAMES[record.branch], f"{record.branch}: coordinate filename drift")
    _require(file_metadata.st_size == record.size_bytes, f"{record.branch}: coordinate HDF5 size mismatch")
    expected_file_sha256 = _validate_digest(
        record.sha256, label=f"{record.branch} record SHA-256"
    )
    _require(
        sha256_file(path) == expected_file_sha256,
        f"{record.branch}: coordinate HDF5 SHA-256 mismatch",
    )
    expected_content_sha256 = _validate_digest(
        record.coordinates_sha256,
        label=f"{record.branch} coordinate-content SHA-256",
    )

    try:
        with h5py.File(path, "r") as h5:
            _require(list(h5.keys()) == ["coords"], f"{record.branch}: HDF5 must contain only coords")
            _require(len(h5.attrs) == 0, f"{record.branch}: HDF5 root attributes are prohibited")
            dataset = h5["coords"]
            _require(isinstance(dataset, h5py.Dataset), f"{record.branch}: coords must be a dataset")
            _require(dataset.dtype == np.dtype(np.int64), f"{record.branch}: HDF5 coords must be int64")
            attributes = {
                key: _normalise_hdf5_attribute(dataset.attrs[key])
                for key in dataset.attrs.keys()
            }
            _require(
                attributes == record.metadata.to_attributes(),
                f"{record.branch}: HDF5 attributes differ from validated metadata",
            )
            coordinate_array = np.asarray(dataset[...])
    except Q50FeatureExtractionContractError:
        raise
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise Q50FeatureExtractionContractError(
            f"{record.branch}: could not read coordinate HDF5: {exc}"
        ) from exc

    _require(coordinate_array.dtype == np.dtype(np.int64), f"{record.branch}: loaded coordinates must be int64")
    _require(
        _coordinates_sha256(coordinate_array) == expected_content_sha256,
        f"{record.branch}: coordinate-content SHA-256 mismatch",
    )
    coordinates = torch.from_numpy(np.ascontiguousarray(coordinate_array)).clone()
    return Q50BranchReadSpec(
        branch=record.branch,
        coordinates=coordinates,
        coordinate_hdf5_path=path,
        coordinate_hdf5_size_bytes=record.size_bytes,
        coordinate_hdf5_sha256=expected_file_sha256,
        coordinate_content_sha256=expected_content_sha256,
        source_level=expected.source_level,
        read_size=expected.read_size,
        branch_output_size=expected.branch_output_size,
        level_0_dimensions=Q50_LEVEL_0_DIMENSIONS,
        level_0_footprint=expected.level_0_footprint,
        level_0_step=expected.level_0_step,
        target_mpp=expected.target_mpp,
        effective_mpp=expected.effective_mpp,
        interpolation=expected.interpolation,
        resampling=expected.resampling,
    )


class StreamingQ50OpenSlideDataset(Dataset[torch.Tensor]):
    """Stream one exact Q50 branch without saving or caching patch images."""

    def __init__(
        self,
        wsi_path: str | Path,
        spec: Q50BranchReadSpec,
        *,
        expected_file_identity: Q50WSIFileIdentity,
        slide_factory: Callable[[str], object] = openslide.OpenSlide,
    ) -> None:
        _require(isinstance(spec, Q50BranchReadSpec), "spec must be Q50BranchReadSpec")
        _require(
            isinstance(expected_file_identity, Q50WSIFileIdentity),
            "expected_file_identity must be Q50WSIFileIdentity from hash preflight",
        )
        path = _lexical_absolute_path(wsi_path)
        _require(path.name == Q50_WSI_FILENAME, "Q50 WSI filename mismatch")
        observed_identity = _identity_from_stat(_lstat_regular_non_symlink(path))
        _require(
            observed_identity == expected_file_identity,
            "Q50 WSI filesystem identity differs from hash preflight",
        )
        _require(callable(slide_factory), "slide_factory must be callable")
        self.wsi_path = str(path)
        self.expected_file_identity = expected_file_identity
        self.spec = replace(spec, coordinates=spec.coordinates.clone().contiguous())
        self.slide_factory = slide_factory
        self._slide: object | None = None

    def __len__(self) -> int:
        return self.spec.coordinate_count

    def _get_slide(self) -> object:
        if self._slide is None:
            path = Path(self.wsi_path)
            before_open = _identity_from_stat(_lstat_regular_non_symlink(path))
            _require(
                before_open == self.expected_file_identity,
                "Q50 WSI filesystem identity changed before OpenSlide open",
            )
            slide = self.slide_factory(self.wsi_path)
            try:
                after_open = _identity_from_stat(_lstat_regular_non_symlink(path))
                _require(
                    after_open == self.expected_file_identity,
                    "Q50 WSI filesystem identity changed during OpenSlide open",
                )
            except Exception:
                close = getattr(slide, "close", None)
                if callable(close):
                    close()
                raise
            self._slide = slide
        return self._slide

    def __getitem__(self, index: int) -> torch.Tensor:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("Q50 patch index must be an integer")
        if index < 0 or index >= len(self):
            raise IndexError(index)
        x, y = (int(value) for value in self.spec.coordinates[index].tolist())
        slide = self._get_slide()
        read_region = getattr(slide, "read_region", None)
        if not callable(read_region):
            raise Q50FeatureExtractionContractError("OpenSlide object lacks read_region")
        image = read_region((x, y), self.spec.source_level, self.spec.read_size)
        _require(isinstance(image, Image.Image), "OpenSlide read_region must return PIL.Image")
        _require(image.size == self.spec.read_size, f"{self.spec.branch}: OpenSlide read size mismatch")

        rgb = image.convert("RGB")
        if self.spec.branch == "scale_2x":
            rgb = rgb.resize(
                self.spec.branch_output_size,
                resample=Image.Resampling.LANCZOS,
            )
        _require(
            rgb.mode == "RGB" and rgb.size == self.spec.branch_output_size,
            f"{self.spec.branch}: branch output must be RGB 256x256",
        )
        return prepare_patch_for_resnet(rgb)

    def close(self) -> None:
        slide = getattr(self, "_slide", None)
        if slide is not None:
            close = getattr(slide, "close", None)
            if callable(close):
                close()
            self._slide = None

    def __enter__(self) -> "StreamingQ50OpenSlideDataset":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_slide"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup
        self.close()


__all__ = [
    "Q50BranchReadSpec",
    "Q50FeatureExtractionContractError",
    "Q50WSIFileIdentity",
    "StreamingQ50OpenSlideDataset",
    "capture_q50_wsi_file_identity",
    "load_q50_branch_read_spec",
]
