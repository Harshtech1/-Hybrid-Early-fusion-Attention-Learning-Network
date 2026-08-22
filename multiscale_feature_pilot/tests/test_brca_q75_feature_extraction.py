from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import pickle

import h5py
import numpy as np
from PIL import Image, ImageChops
import pytest
import torch

from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
    BRANCH_FILENAMES,
    CoordinateBranchMetadata,
    CoordinateBranchRecord,
    sha256_file,
)
from multiscale_feature_pilot.src import brca_q75_feature_extraction as q75
from multiscale_feature_pilot.src.brca_q75_feature_extraction import (
    Q75BranchReadSpec,
    Q75FeatureExtractionContractError,
    Q75WSIFileIdentity,
    StreamingQ75OpenSlideDataset,
    capture_q75_wsi_file_identity,
    load_q75_branch_read_spec,
)


def test_public_constants_lock_verified_q75_identity_and_coordinate_manifest() -> None:
    assert q75.Q75_PATIENT_ID == "TCGA-E2-A154"
    assert q75.Q75_SLIDE_ID == (
        "TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs"
    )
    assert q75.Q75_GDC_FILE_UUID == "25aec062-60d1-446e-a1c6-0c79cc74a770"
    assert q75.Q75_WSI_SIZE_BYTES == 1_360_743_825
    assert q75.Q75_WSI_MD5 == "a8c4b68fb6e0ab3e862efe3ed1fe10d7"
    assert q75.Q75_WSI_SHA256 == (
        "844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37"
    )
    assert q75.Q75_COORDINATE_MANIFEST_SHA256 == (
        "438165ce6b3be9d26d66c65cd70793e29cc92208cfb6a78bf68043bc4b4a4e90"
    )
    assert q75.Q75_COORDINATE_POLICY_SHA256 == (
        "58f15a9e39fcd3469ec656ef98c72ad6e42b8a3eab16fcbc24c4345cc4337d88"
    )
    assert q75.Q75_LEVEL_0_DIMENSIONS == (108_528, 90_471)


def _coordinates(branch: str) -> np.ndarray:
    if branch == "scale_2x":
        count, footprint, step = 13_487, 512, 512
    else:
        count, footprint, step = 3_458, 1_024, 1_024
    rows = (
        (x, y)
        for y in range(0, q75.Q75_LEVEL_0_DIMENSIONS[1] - footprint + 1, step)
        for x in range(0, q75.Q75_LEVEL_0_DIMENSIONS[0] - footprint + 1, step)
    )
    return np.asarray([next(rows) for _ in range(count)], dtype=np.int64)


def _coordinate_digest(coordinates: np.ndarray) -> str:
    canonical = np.ascontiguousarray(coordinates, dtype=np.dtype("<i8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _metadata(branch: str) -> CoordinateBranchMetadata:
    if branch == "scale_2x":
        source_level = 0
        source_dimensions = (108_528, 90_471)
        source_downsample = 1.0
        source_patch = (512, 512)
        footprint = (512, 512)
        target_mpp = 0.5
        effective_mpp = (0.4936, 0.4936)
        interpolation = "PIL.Image.Resampling.LANCZOS"
        resampling = "explicit_2x_spatial_downsample"
        geometry = "DIRECT_LEVEL_0_GEOMETRY"
    else:
        source_level = 1
        source_dimensions = (27_132, 22_617)
        source_downsample = 4.000066321793341
        source_patch = (256, 256)
        footprint = (1_024, 1_024)
        target_mpp = 1.0
        effective_mpp = (0.9872163682185965, 0.9872163682185965)
        interpolation = "none"
        resampling = "none"
        geometry = "CLAM_INT_CAST_GEOMETRY_COMPATIBLE"
    return CoordinateBranchMetadata(
        branch=branch,
        patient_id=q75.Q75_PATIENT_ID,
        slide_id=q75.Q75_SLIDE_ID,
        gdc_file_uuid=q75.Q75_GDC_FILE_UUID,
        wsi_filename=q75.Q75_WSI_FILENAME,
        wsi_size_bytes=q75.Q75_WSI_SIZE_BYTES,
        wsi_md5=q75.Q75_WSI_MD5,
        wsi_sha256=q75.Q75_WSI_SHA256,
        level_0_dimensions=q75.Q75_LEVEL_0_DIMENSIONS,
        source_level=source_level,
        source_level_dimensions=source_dimensions,
        openslide_reported_source_downsample=source_downsample,
        source_patch_size=source_patch,
        output_patch_size=(256, 256),
        level_0_declared_footprint=footprint,
        level_0_step=footprint,
        target_mpp=target_mpp,
        effective_mpp=effective_mpp,
        interpolation=interpolation,
        resampling=resampling,
        mask_level=2,
        mask_level_dimensions=(6_783, 5_654),
        openslide_reported_mask_downsample=16.000619030774672,
        mask_image_channels=4,
        mask_image_sha256=q75.Q75_MASK_IMAGE_SHA256,
        mask_parameters=q75.Q75_MASK_PARAMETERS,
        contour_count=4,
        retained_hole_count=17,
        clam_commit=q75.Q75_CLAM_COMMIT,
        policy_sha256=q75.Q75_COORDINATE_POLICY_SHA256,
        geometry_compatibility=geometry,
    )


def _record(tmp_path: Path, branch: str) -> CoordinateBranchRecord:
    coordinates = _coordinates(branch)
    metadata = _metadata(branch)
    path = tmp_path / BRANCH_FILENAMES[branch]
    with h5py.File(path, "x") as h5:
        dataset = h5.create_dataset(
            "coords", data=coordinates, dtype=np.int64, track_times=False
        )
        for key, value in metadata.to_attributes().items():
            dataset.attrs[key] = value
    return CoordinateBranchRecord(
        branch=branch,
        path=path,
        coordinate_count=coordinates.shape[0],
        size_bytes=path.stat().st_size,
        sha256=sha256_file(path),
        coordinates_sha256=_coordinate_digest(coordinates),
        metadata=metadata,
    )


@pytest.fixture(scope="module")
def records(tmp_path_factory: pytest.TempPathFactory) -> dict[str, CoordinateBranchRecord]:
    root = tmp_path_factory.mktemp("q75-coordinates")
    return {branch: _record(root, branch) for branch in ("scale_2x", "scale_4x")}


@pytest.fixture(scope="module")
def loaded_specs(
    records: dict[str, CoordinateBranchRecord],
) -> dict[str, Q75BranchReadSpec]:
    return {branch: load_q75_branch_read_spec(record) for branch, record in records.items()}


def test_strict_records_load_exact_q75_branch_specs(
    loaded_specs: dict[str, Q75BranchReadSpec],
) -> None:
    scale_2x = loaded_specs["scale_2x"]
    assert scale_2x.coordinates.shape == (13_487, 2)
    assert scale_2x.coordinates.dtype is torch.int64
    assert scale_2x.coordinates.device.type == "cpu"
    assert scale_2x.coordinates.is_contiguous()
    assert scale_2x.source_level == 0
    assert scale_2x.read_size == (512, 512)
    assert scale_2x.branch_output_size == (256, 256)
    assert scale_2x.level_0_footprint == scale_2x.level_0_step == (512, 512)
    assert scale_2x.effective_mpp == (0.4936, 0.4936)

    scale_4x = loaded_specs["scale_4x"]
    assert scale_4x.coordinates.shape == (3_458, 2)
    assert scale_4x.source_level == 1
    assert scale_4x.read_size == scale_4x.branch_output_size == (256, 256)
    assert scale_4x.level_0_footprint == scale_4x.level_0_step == (1_024, 1_024)
    assert scale_4x.effective_mpp == (
        0.9872163682185965,
        0.9872163682185965,
    )


class _FakeSlide:
    def __init__(self, _path: str) -> None:
        self.calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
        self.returned_images: list[Image.Image] = []
        self.closed = False

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        self.calls.append((location, level, size))
        width, height = size
        x_gradient = np.tile(np.arange(width, dtype=np.uint16) % 256, (height, 1))
        y_gradient = np.tile(
            (np.arange(height, dtype=np.uint16) % 256)[:, None], (1, width)
        )
        rgba = np.stack(
            (
                x_gradient.astype(np.uint8),
                y_gradient.astype(np.uint8),
                np.full((height, width), 73, dtype=np.uint8),
                np.full((height, width), 255, dtype=np.uint8),
            ),
            axis=2,
        )
        image = Image.fromarray(rgba, mode="RGBA")
        self.returned_images.append(image.copy())
        return image

    def close(self) -> None:
        self.closed = True


class _SlideFactory:
    def __init__(self) -> None:
        self.slides: list[_FakeSlide] = []

    def __call__(self, path: str) -> _FakeSlide:
        slide = _FakeSlide(path)
        self.slides.append(slide)
        return slide


def _fake_wsi(tmp_path: Path) -> Path:
    path = tmp_path / q75.Q75_WSI_FILENAME
    path.touch()
    os.truncate(path, q75.Q75_WSI_SIZE_BYTES)
    return path


@pytest.mark.parametrize(
    ("branch", "index", "expected_level", "expected_size"),
    [
        ("scale_2x", 1, 0, (512, 512)),
        ("scale_4x", 1, 1, (256, 256)),
    ],
)
def test_dataset_uses_level0_origins_and_exact_q75_pixel_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded_specs: dict[str, Q75BranchReadSpec],
    branch: str,
    index: int,
    expected_level: int,
    expected_size: tuple[int, int],
) -> None:
    prepared: list[Image.Image] = []

    def capture(image: Image.Image) -> torch.Tensor:
        prepared.append(image.copy())
        return torch.zeros((3, 224, 224), dtype=torch.float32)

    monkeypatch.setattr(q75, "prepare_patch_for_resnet", capture)
    factory = _SlideFactory()
    spec = loaded_specs[branch]
    wsi_path = _fake_wsi(tmp_path)
    dataset = StreamingQ75OpenSlideDataset(
        wsi_path,
        spec,
        expected_file_identity=capture_q75_wsi_file_identity(wsi_path),
        slide_factory=factory,
    )
    result = dataset[index]
    slide = factory.slides[0]
    expected_origin = tuple(int(value) for value in spec.coordinates[index].tolist())

    assert result.shape == (3, 224, 224)
    assert slide.calls == [(expected_origin, expected_level, expected_size)]
    source_rgb = slide.returned_images[0].convert("RGB")
    expected_rgb = (
        source_rgb.resize((256, 256), resample=Image.Resampling.LANCZOS)
        if branch == "scale_2x"
        else source_rgb
    )
    assert len(prepared) == 1
    assert prepared[0].mode == "RGB" and prepared[0].size == (256, 256)
    assert ImageChops.difference(prepared[0], expected_rgb).getbbox() is None
    dataset.close()
    assert slide.closed and dataset._slide is None


def test_dataset_delegates_to_shared_imagenet_preprocessing(
    tmp_path: Path,
    loaded_specs: dict[str, Q75BranchReadSpec],
) -> None:
    factory = _SlideFactory()
    wsi_path = _fake_wsi(tmp_path)
    with StreamingQ75OpenSlideDataset(
        wsi_path,
        loaded_specs["scale_4x"],
        expected_file_identity=capture_q75_wsi_file_identity(wsi_path),
        slide_factory=factory,
    ) as dataset:
        patch = dataset[0]
        assert patch.shape == (3, 224, 224)
        assert patch.dtype is torch.float32
        assert bool(torch.isfinite(patch).all().item())
    assert factory.slides[0].closed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("policy_sha256", "0" * 64, "policy_sha256 drift"),
        ("wsi_sha256", "0" * 64, "wsi_sha256 drift"),
        ("target_mpp", 0.6, "target_mpp drift"),
        ("geometry_compatibility", "DRIFT", "geometry_compatibility drift"),
        ("mask_image_sha256", "0" * 64, "mask_image_sha256 drift"),
    ],
)
def test_loader_rejects_q75_identity_policy_hash_and_geometry_drift(
    records: dict[str, CoordinateBranchRecord],
    field: str,
    value: object,
    message: str,
) -> None:
    record = records["scale_2x"]
    drifted_metadata = replace(record.metadata, **{field: value})
    with pytest.raises(Q75FeatureExtractionContractError, match=message):
        load_q75_branch_read_spec(replace(record, metadata=drifted_metadata))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source_level_dimensions", (27_132, 22_616), "source_level_dimensions drift"),
        (
            "openslide_reported_source_downsample",
            4.0,
            "openslide_reported_source_downsample drift",
        ),
        ("mask_level_dimensions", (6_783, 5_653), "mask_level_dimensions drift"),
        (
            "openslide_reported_mask_downsample",
            16.0,
            "openslide_reported_mask_downsample drift",
        ),
    ],
)
def test_loader_rejects_q75_pyramid_and_mask_geometry_drift(
    records: dict[str, CoordinateBranchRecord],
    field: str,
    value: object,
    message: str,
) -> None:
    record = records["scale_4x"]
    drifted_metadata = replace(record.metadata, **{field: value})
    with pytest.raises(Q75FeatureExtractionContractError, match=message):
        load_q75_branch_read_spec(replace(record, metadata=drifted_metadata))


def test_loader_rejects_file_and_coordinate_hash_drift(
    records: dict[str, CoordinateBranchRecord],
) -> None:
    record = records["scale_2x"]
    with pytest.raises(Q75FeatureExtractionContractError, match="HDF5 SHA-256 mismatch"):
        load_q75_branch_read_spec(replace(record, sha256="0" * 64))
    with pytest.raises(
        Q75FeatureExtractionContractError,
        match="coordinate-content SHA-256 mismatch",
    ):
        load_q75_branch_read_spec(replace(record, coordinates_sha256="0" * 64))


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda spec: replace(spec, coordinates=spec.coordinates.to(torch.int32)),
            "coordinates must be int64",
        ),
        (
            lambda spec: replace(spec, coordinates=spec.coordinates[:-1].clone()),
            r"shape \[13487,2\]",
        ),
        (
            lambda spec: replace(
                spec,
                coordinates=torch.cat(
                    (spec.coordinates[1:2], spec.coordinates[0:1], spec.coordinates[2:])
                ).contiguous(),
            ),
            "unique row-major",
        ),
        (
            lambda spec: replace(
                spec,
                coordinates=torch.cat(
                    (spec.coordinates[:1] + torch.tensor([[1, 0]]), spec.coordinates[1:])
                ).contiguous(),
            ),
            "off the global level-0 lattice",
        ),
        (
            lambda spec: replace(
                spec,
                coordinates=torch.cat(
                    (
                        spec.coordinates[:-1],
                        torch.tensor([[108_032, 89_600]], dtype=torch.int64),
                    )
                ).contiguous(),
            ),
            "incomplete level-0 footprint",
        ),
        (
            lambda spec: replace(spec, coordinate_content_sha256="0" * 64),
            "coordinate-content SHA-256 mismatch",
        ),
    ],
)
def test_spec_rejects_dtype_count_order_lattice_bounds_and_hash_drift(
    loaded_specs: dict[str, Q75BranchReadSpec],
    mutator: object,
    message: str,
) -> None:
    with pytest.raises(Q75FeatureExtractionContractError, match=message):
        mutator(loaded_specs["scale_2x"])


def test_dataset_is_lazy_and_rejects_invalid_indices_before_open(
    tmp_path: Path,
    loaded_specs: dict[str, Q75BranchReadSpec],
) -> None:
    factory = _SlideFactory()
    wsi_path = _fake_wsi(tmp_path)
    dataset = StreamingQ75OpenSlideDataset(
        wsi_path,
        loaded_specs["scale_2x"],
        expected_file_identity=capture_q75_wsi_file_identity(wsi_path),
        slide_factory=factory,
    )
    assert factory.slides == []
    with pytest.raises(IndexError):
        _ = dataset[-1]
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
    with pytest.raises(TypeError):
        _ = dataset[True]
    assert factory.slides == []


def test_identity_token_captures_exact_hash_preflight_leaf(tmp_path: Path) -> None:
    path = _fake_wsi(tmp_path)
    metadata = path.lstat()
    assert capture_q75_wsi_file_identity(path) == Q75WSIFileIdentity(
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_size=metadata.st_size,
        st_mtime_ns=metadata.st_mtime_ns,
    )


def test_dataset_rejects_regular_leaf_swap_before_worker_open(
    tmp_path: Path,
    loaded_specs: dict[str, Q75BranchReadSpec],
) -> None:
    path = _fake_wsi(tmp_path)
    identity = capture_q75_wsi_file_identity(path)
    factory = _SlideFactory()
    dataset = StreamingQ75OpenSlideDataset(
        path,
        loaded_specs["scale_2x"],
        expected_file_identity=identity,
        slide_factory=factory,
    )
    path.rename(tmp_path / "verified-original.svs")
    path.touch()
    os.truncate(path, q75.Q75_WSI_SIZE_BYTES)
    with pytest.raises(
        Q75FeatureExtractionContractError,
        match="identity changed before OpenSlide open",
    ):
        _ = dataset[0]
    assert factory.slides == []


def test_dataset_rejects_symlink_leaf_swap_before_worker_open(
    tmp_path: Path,
    loaded_specs: dict[str, Q75BranchReadSpec],
) -> None:
    path = _fake_wsi(tmp_path)
    identity = capture_q75_wsi_file_identity(path)
    factory = _SlideFactory()
    dataset = StreamingQ75OpenSlideDataset(
        path,
        loaded_specs["scale_4x"],
        expected_file_identity=identity,
        slide_factory=factory,
    )
    original = tmp_path / "verified-original.svs"
    path.rename(original)
    path.symlink_to(original)
    with pytest.raises(Q75FeatureExtractionContractError, match="regular non-symlink"):
        _ = dataset[0]
    assert factory.slides == []


def test_dataset_closes_handle_when_leaf_swaps_during_open(
    tmp_path: Path,
    loaded_specs: dict[str, Q75BranchReadSpec],
) -> None:
    path = _fake_wsi(tmp_path)
    identity = capture_q75_wsi_file_identity(path)

    class SwapDuringOpenFactory:
        def __init__(self) -> None:
            self.slide: _FakeSlide | None = None

        def __call__(self, supplied_path: str) -> _FakeSlide:
            lexical_path = Path(supplied_path)
            lexical_path.rename(tmp_path / "verified-original.svs")
            lexical_path.touch()
            os.truncate(lexical_path, q75.Q75_WSI_SIZE_BYTES)
            self.slide = _FakeSlide(supplied_path)
            return self.slide

    factory = SwapDuringOpenFactory()
    dataset = StreamingQ75OpenSlideDataset(
        path,
        loaded_specs["scale_2x"],
        expected_file_identity=identity,
        slide_factory=factory,
    )
    with pytest.raises(
        Q75FeatureExtractionContractError,
        match="identity changed during OpenSlide open",
    ):
        _ = dataset[0]
    assert factory.slide is not None and factory.slide.closed
    assert dataset._slide is None


def test_worker_serialization_drops_process_local_slide_handle(
    tmp_path: Path,
    loaded_specs: dict[str, Q75BranchReadSpec],
) -> None:
    path = _fake_wsi(tmp_path)
    factory = _SlideFactory()
    dataset = StreamingQ75OpenSlideDataset(
        path,
        loaded_specs["scale_4x"],
        expected_file_identity=capture_q75_wsi_file_identity(path),
        slide_factory=factory,
    )
    _ = dataset[0]
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._slide is None
    assert torch.equal(restored.spec.coordinates, dataset.spec.coordinates)
    dataset.close()
