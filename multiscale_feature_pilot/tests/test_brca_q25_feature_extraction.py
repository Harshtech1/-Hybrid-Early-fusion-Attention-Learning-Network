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
from multiscale_feature_pilot.src import brca_q25_feature_extraction as q25
from multiscale_feature_pilot.src.brca_q25_feature_extraction import (
    Q25BranchReadSpec,
    Q25FeatureExtractionContractError,
    Q25WSIFileIdentity,
    StreamingQ25OpenSlideDataset,
    capture_q25_wsi_file_identity,
    load_q25_branch_read_spec,
)


def _coordinates(branch: str) -> np.ndarray:
    if branch == "scale_2x":
        count, footprint, step = 7_404, 512, 512
    else:
        count, footprint, step = 1_918, 1_024, 1_024
    rows = (
        (x, y)
        for y in range(0, q25.Q25_LEVEL_0_DIMENSIONS[1] - footprint + 1, step)
        for x in range(0, q25.Q25_LEVEL_0_DIMENSIONS[0] - footprint + 1, step)
    )
    return np.asarray([next(rows) for _ in range(count)], dtype=np.int64)


def _coordinate_digest(coordinates: np.ndarray) -> str:
    canonical = np.ascontiguousarray(coordinates, dtype=np.dtype("<i8"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _metadata(branch: str) -> CoordinateBranchMetadata:
    if branch == "scale_2x":
        source_level = 0
        source_dimensions = (65_736, 67_406)
        source_downsample = 1.0
        source_patch = (512, 512)
        footprint = (512, 512)
        target_mpp = 0.5
        effective_mpp = (0.505, 0.505)
        interpolation = "PIL.Image.Resampling.LANCZOS"
        resampling = "explicit_2x_spatial_downsample"
        geometry = "LEVEL0_IDENTITY_GEOMETRY"
    else:
        source_level = 1
        source_dimensions = (16_434, 16_851)
        source_downsample = 4.00005934365913
        source_patch = (256, 256)
        footprint = (1_024, 1_024)
        target_mpp = 1.0
        effective_mpp = (1.0100149842739303, 1.0100149842739303)
        interpolation = "none"
        resampling = "none"
        geometry = "CLAM_INT_CAST_GEOMETRY_COMPATIBLE"
    return CoordinateBranchMetadata(
        branch=branch,
        patient_id=q25.Q25_PATIENT_ID,
        slide_id=q25.Q25_SLIDE_ID,
        gdc_file_uuid=q25.Q25_GDC_FILE_UUID,
        wsi_filename=q25.Q25_WSI_FILENAME,
        wsi_size_bytes=q25.Q25_WSI_SIZE_BYTES,
        wsi_md5=q25.Q25_WSI_MD5,
        wsi_sha256=q25.Q25_WSI_SHA256,
        level_0_dimensions=q25.Q25_LEVEL_0_DIMENSIONS,
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
        mask_level_dimensions=(4_108, 4_212),
        openslide_reported_mask_downsample=16.002635628163056,
        mask_image_channels=4,
        mask_image_sha256=q25.Q25_MASK_IMAGE_SHA256,
        mask_parameters=q25.Q25_MASK_PARAMETERS,
        contour_count=3,
        retained_hole_count=4,
        clam_commit=q25.Q25_CLAM_COMMIT,
        policy_sha256=q25.Q25_COORDINATE_POLICY_SHA256,
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
def loaded_specs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Q25BranchReadSpec]:
    root = tmp_path_factory.mktemp("q25-coordinates")
    return {
        branch: load_q25_branch_read_spec(_record(root, branch))
        for branch in ("scale_2x", "scale_4x")
    }


def test_strict_records_load_explicit_q25_branch_specs(
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    scale_2x = loaded_specs["scale_2x"]
    assert scale_2x.coordinate_count == 7_404
    assert scale_2x.coordinates.shape == (7_404, 2)
    assert scale_2x.coordinates.dtype is torch.int64
    assert scale_2x.coordinates.device.type == "cpu"
    assert scale_2x.coordinates.is_contiguous()
    assert scale_2x.source_level == 0
    assert scale_2x.read_size == (512, 512)
    assert scale_2x.branch_output_size == (256, 256)
    assert scale_2x.level_0_footprint == scale_2x.level_0_step == (512, 512)
    assert scale_2x.effective_mpp == (0.505, 0.505)

    scale_4x = loaded_specs["scale_4x"]
    assert scale_4x.coordinate_count == 1_918
    assert scale_4x.coordinates.shape == (1_918, 2)
    assert scale_4x.source_level == 1
    assert scale_4x.read_size == scale_4x.branch_output_size == (256, 256)
    assert scale_4x.level_0_footprint == scale_4x.level_0_step == (1_024, 1_024)
    assert scale_4x.effective_mpp == (
        1.0100149842739303,
        1.0100149842739303,
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
    path = tmp_path / q25.Q25_WSI_FILENAME
    path.touch()
    # Sparse file: exact identity size without materializing or reading pixels.
    os.truncate(path, q25.Q25_WSI_SIZE_BYTES)
    return path


@pytest.mark.parametrize(
    ("branch", "index", "expected_level", "expected_size"),
    [
        ("scale_2x", 1, 0, (512, 512)),
        ("scale_4x", 1, 1, (256, 256)),
    ],
)
def test_dataset_uses_level0_origins_and_exact_branch_pixel_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    loaded_specs: dict[str, Q25BranchReadSpec],
    branch: str,
    index: int,
    expected_level: int,
    expected_size: tuple[int, int],
) -> None:
    prepared: list[Image.Image] = []

    def capture(image: Image.Image) -> torch.Tensor:
        prepared.append(image.copy())
        return torch.zeros((3, 224, 224), dtype=torch.float32)

    monkeypatch.setattr(q25, "prepare_patch_for_resnet", capture)
    factory = _SlideFactory()
    spec = loaded_specs[branch]
    wsi_path = _fake_wsi(tmp_path)
    dataset = StreamingQ25OpenSlideDataset(
        wsi_path,
        spec,
        expected_file_identity=capture_q25_wsi_file_identity(wsi_path),
        slide_factory=factory,
    )
    result = dataset[index]
    slide = factory.slides[0]
    expected_origin = tuple(int(value) for value in spec.coordinates[index].tolist())

    assert result.shape == (3, 224, 224)
    assert slide.calls == [(expected_origin, expected_level, expected_size)]
    assert len(prepared) == 1
    source_rgb = slide.returned_images[0].convert("RGB")
    expected_rgb = (
        source_rgb.resize((256, 256), resample=Image.Resampling.LANCZOS)
        if branch == "scale_2x"
        else source_rgb
    )
    assert prepared[0].mode == "RGB"
    assert prepared[0].size == (256, 256)
    assert ImageChops.difference(prepared[0], expected_rgb).getbbox() is None
    dataset.close()
    assert slide.closed
    assert dataset._slide is None


def test_dataset_uses_shared_resnet_preprocessing_and_context_cleanup(
    tmp_path: Path,
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    factory = _SlideFactory()
    wsi_path = _fake_wsi(tmp_path)
    with StreamingQ25OpenSlideDataset(
        wsi_path,
        loaded_specs["scale_4x"],
        expected_file_identity=capture_q25_wsi_file_identity(wsi_path),
        slide_factory=factory,
    ) as dataset:
        patch = dataset[0]
        slide = factory.slides[0]
        assert patch.shape == (3, 224, 224)
        assert patch.dtype is torch.float32
        assert bool(torch.isfinite(patch).all().item())
    assert slide.closed
    assert dataset._slide is None


def test_worker_serialization_drops_process_local_slide_handle(
    tmp_path: Path,
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    factory = _SlideFactory()
    wsi_path = _fake_wsi(tmp_path)
    dataset = StreamingQ25OpenSlideDataset(
        wsi_path,
        loaded_specs["scale_4x"],
        expected_file_identity=capture_q25_wsi_file_identity(wsi_path),
        slide_factory=factory,
    )
    _ = dataset[0]
    assert dataset._slide is not None

    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._slide is None
    assert torch.equal(restored.spec.coordinates, dataset.spec.coordinates)
    dataset.close()


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda spec: replace(
                spec, coordinates=spec.coordinates.to(dtype=torch.int32)
            ),
            "coordinates must be int64",
        ),
        (
            lambda spec: replace(spec, coordinates=spec.coordinates[:-1].clone()),
            r"shape \[7404,2\]",
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
                    (
                        spec.coordinates[:1] + torch.tensor([[1, 0]]),
                        spec.coordinates[1:],
                    )
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
                        torch.tensor(
                            [[65_536, int(spec.coordinates[-1, 1])]],
                            dtype=torch.int64,
                        ),
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
    loaded_specs: dict[str, Q25BranchReadSpec],
    mutator: object,
    message: str,
) -> None:
    with pytest.raises(Q25FeatureExtractionContractError, match=message):
        mutator(loaded_specs["scale_2x"])


def test_record_loader_rejects_metadata_and_file_hash_drift(tmp_path: Path) -> None:
    record = _record(tmp_path, "scale_2x")
    drifted_metadata = replace(record.metadata, target_mpp=0.6)
    with pytest.raises(Q25FeatureExtractionContractError, match="target_mpp drift"):
        load_q25_branch_read_spec(replace(record, metadata=drifted_metadata))

    with pytest.raises(Q25FeatureExtractionContractError, match="HDF5 SHA-256 mismatch"):
        load_q25_branch_read_spec(replace(record, sha256="0" * 64))


def test_dataset_rejects_wrong_wsi_identity_before_opening_slide(
    tmp_path: Path,
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    wrong = tmp_path / "wrong.svs"
    wrong.touch()
    os.truncate(wrong, q25.Q25_WSI_SIZE_BYTES)
    factory = _SlideFactory()
    expected_identity = Q25WSIFileIdentity(
        st_dev=wrong.stat().st_dev,
        st_ino=wrong.stat().st_ino,
        st_size=wrong.stat().st_size,
        st_mtime_ns=wrong.stat().st_mtime_ns,
    )
    with pytest.raises(Q25FeatureExtractionContractError, match="filename mismatch"):
        StreamingQ25OpenSlideDataset(
            wrong,
            loaded_specs["scale_2x"],
            expected_file_identity=expected_identity,
            slide_factory=factory,
        )
    assert factory.slides == []


def test_dataset_rejects_invalid_indices_without_opening_slide(
    tmp_path: Path,
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    factory = _SlideFactory()
    wsi_path = _fake_wsi(tmp_path)
    dataset = StreamingQ25OpenSlideDataset(
        wsi_path,
        loaded_specs["scale_2x"],
        expected_file_identity=capture_q25_wsi_file_identity(wsi_path),
        slide_factory=factory,
    )
    with pytest.raises(IndexError):
        _ = dataset[-1]
    with pytest.raises(IndexError):
        _ = dataset[len(dataset)]
    with pytest.raises(TypeError):
        _ = dataset[True]
    assert factory.slides == []


def test_identity_helper_captures_exact_hash_preflight_token(tmp_path: Path) -> None:
    path = _fake_wsi(tmp_path)
    metadata = path.lstat()

    identity = capture_q25_wsi_file_identity(path)

    assert identity == Q25WSIFileIdentity(
        st_dev=metadata.st_dev,
        st_ino=metadata.st_ino,
        st_size=metadata.st_size,
        st_mtime_ns=metadata.st_mtime_ns,
    )


def test_dataset_rejects_regular_leaf_swap_before_first_worker_open(
    tmp_path: Path,
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    path = _fake_wsi(tmp_path)
    identity = capture_q25_wsi_file_identity(path)
    factory = _SlideFactory()
    dataset = StreamingQ25OpenSlideDataset(
        path,
        loaded_specs["scale_2x"],
        expected_file_identity=identity,
        slide_factory=factory,
    )

    original = tmp_path / "verified-original.svs"
    path.rename(original)
    path.touch()
    os.truncate(path, q25.Q25_WSI_SIZE_BYTES)

    with pytest.raises(
        Q25FeatureExtractionContractError,
        match="identity changed before OpenSlide open",
    ):
        _ = dataset[0]
    assert factory.slides == []


def test_dataset_rejects_symlink_leaf_swap_before_first_worker_open(
    tmp_path: Path,
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    path = _fake_wsi(tmp_path)
    identity = capture_q25_wsi_file_identity(path)
    factory = _SlideFactory()
    dataset = StreamingQ25OpenSlideDataset(
        path,
        loaded_specs["scale_4x"],
        expected_file_identity=identity,
        slide_factory=factory,
    )

    original = tmp_path / "verified-original.svs"
    path.rename(original)
    path.symlink_to(original)

    with pytest.raises(
        Q25FeatureExtractionContractError,
        match="regular non-symlink",
    ):
        _ = dataset[0]
    assert factory.slides == []


def test_dataset_rejects_leaf_swap_during_first_openslide_open(
    tmp_path: Path,
    loaded_specs: dict[str, Q25BranchReadSpec],
) -> None:
    path = _fake_wsi(tmp_path)
    identity = capture_q25_wsi_file_identity(path)

    class SwapDuringOpenFactory:
        def __init__(self) -> None:
            self.slide: _FakeSlide | None = None

        def __call__(self, supplied_path: str) -> _FakeSlide:
            lexical_path = Path(supplied_path)
            lexical_path.rename(tmp_path / "verified-original.svs")
            lexical_path.touch()
            os.truncate(lexical_path, q25.Q25_WSI_SIZE_BYTES)
            self.slide = _FakeSlide(supplied_path)
            return self.slide

    factory = SwapDuringOpenFactory()
    dataset = StreamingQ25OpenSlideDataset(
        path,
        loaded_specs["scale_2x"],
        expected_file_identity=identity,
        slide_factory=factory,
    )

    with pytest.raises(
        Q25FeatureExtractionContractError,
        match="identity changed during OpenSlide open",
    ):
        _ = dataset[0]
    assert factory.slide is not None
    assert factory.slide.closed
    assert dataset._slide is None
