from __future__ import annotations

from PIL import Image
import pytest
import torch

from multiscale_feature_pilot.src.feature_extraction import (
    PatchBranchSpec,
    StreamingOpenSlideDataset,
    prepare_patch_for_resnet,
)


class _FakeSlide:
    def __init__(self, _path: str) -> None:
        self.calls: list[tuple[tuple[int, int], int, tuple[int, int]]] = []
        self.closed = False

    def read_region(
        self,
        location: tuple[int, int],
        level: int,
        size: tuple[int, int],
    ) -> Image.Image:
        self.calls.append((location, level, size))
        return Image.new("RGBA", size, (location[0] % 255, location[1] % 255, 64, 255))

    def close(self) -> None:
        self.closed = True


def test_preprocess_is_locked_finite_float32_shape() -> None:
    result = prepare_patch_for_resnet(Image.new("RGB", (256, 256), (128, 64, 32)))

    assert result.shape == (3, 224, 224)
    assert result.dtype is torch.float32
    assert torch.isfinite(result).all()


def test_preprocess_rejects_wrong_patch_size() -> None:
    with pytest.raises(ValueError, match="256x256"):
        prepare_patch_for_resnet(Image.new("RGB", (224, 224)))


def test_2x_dataset_reads_512_level0_then_downsamples() -> None:
    coordinates = torch.tensor([[0, 0], [512, 1024]], dtype=torch.int64)
    spec = PatchBranchSpec("scale_2x", coordinates, patch_level=0, patch_size=256)
    dataset = StreamingOpenSlideDataset("fake.svs", spec, slide_factory=_FakeSlide)

    first = dataset[0]
    second = dataset[1]
    slide = dataset._slide

    assert first.shape == second.shape == (3, 224, 224)
    assert slide.calls == [((0, 0), 0, (512, 512)), ((512, 1024), 0, (512, 512))]
    dataset.close()
    assert slide.closed


def test_4x_dataset_preserves_coordinate_order_at_level1() -> None:
    coordinates = torch.tensor([[8, 4], [16, 12]], dtype=torch.int64)
    spec = PatchBranchSpec("scale_4x", coordinates, patch_level=1, patch_size=256)
    dataset = StreamingOpenSlideDataset("fake.svs", spec, slide_factory=_FakeSlide)

    _ = dataset[1]
    _ = dataset[0]

    assert dataset._slide.calls == [((16, 12), 1, (256, 256)), ((8, 4), 1, (256, 256))]


def test_branch_spec_rejects_non_int64_coordinates() -> None:
    with pytest.raises(ValueError, match="torch.int64"):
        PatchBranchSpec(
            "scale_4x",
            torch.tensor([[0.0, 0.0]], dtype=torch.float32),
            patch_level=1,
            patch_size=256,
        )
