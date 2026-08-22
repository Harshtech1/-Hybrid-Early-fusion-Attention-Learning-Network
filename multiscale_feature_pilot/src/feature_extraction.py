"""Streaming, GPU-backed ResNet50 feature extraction for the BLCA pilot."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import openslide
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet50
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from .multiscale_bag import FEATURE_DIM, validate_feature_matrix
from .scale_2x_policy import (
    OUTPUT_PATCH_SIZE,
    SOURCE_FOOTPRINT,
    SOURCE_LEVEL,
    downsample_source_patch,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET_INPUT_SIZE = 224


@dataclass(frozen=True)
class PatchBranchSpec:
    """How ordered level-0 origins are read for one WSI branch."""

    name: Literal["scale_2x", "scale_4x"]
    coordinates: torch.Tensor
    patch_level: int
    patch_size: int

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 2 or self.coordinates.shape[1] != 2:
            raise ValueError(
                f"{self.name}: coordinates must have shape [N,2], "
                f"got {tuple(self.coordinates.shape)}"
            )
        if self.coordinates.dtype is not torch.int64:
            raise ValueError(f"{self.name}: coordinates must use torch.int64")
        if self.coordinates.shape[0] == 0:
            raise ValueError(f"{self.name}: coordinates must not be empty")
        if bool((self.coordinates < 0).any().item()):
            raise ValueError(f"{self.name}: coordinates must be non-negative")
        if self.patch_level < 0 or self.patch_size <= 0:
            raise ValueError(f"{self.name}: invalid patch level or patch size")
        expected = (
            (SOURCE_LEVEL, OUTPUT_PATCH_SIZE)
            if self.name == "scale_2x"
            else (1, OUTPUT_PATCH_SIZE)
        )
        if (self.patch_level, self.patch_size) != expected:
            raise ValueError(
                f"{self.name}: expected patch_level/patch_size {expected}, "
                f"got {(self.patch_level, self.patch_size)}"
            )


@dataclass(frozen=True)
class ExtractionResult:
    """One validated feature matrix and measured CUDA execution metadata."""

    features: torch.Tensor
    streaming_extraction_seconds: float
    model_forward_seconds: float
    peak_gpu_memory_bytes: int
    batch_size: int


def prepare_patch_for_resnet(image: Image.Image) -> torch.Tensor:
    """Apply the locked RGB, 256->224, and ImageNet normalization policy."""

    if image.size != (OUTPUT_PATCH_SIZE, OUTPUT_PATCH_SIZE):
        raise ValueError(
            f"patch must be {OUTPUT_PATCH_SIZE}x{OUTPUT_PATCH_SIZE}, got {image.size}"
        )
    rgb = image.convert("RGB")
    tensor = vision_functional.pil_to_tensor(rgb).to(dtype=torch.float32).div_(255.0)
    tensor = vision_functional.resize(
        tensor,
        [RESNET_INPUT_SIZE, RESNET_INPUT_SIZE],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )
    tensor = vision_functional.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
    if tensor.shape != (3, RESNET_INPUT_SIZE, RESNET_INPUT_SIZE):
        raise RuntimeError(f"unexpected preprocessed patch shape: {tuple(tensor.shape)}")
    if tensor.dtype is not torch.float32 or not bool(torch.isfinite(tensor).all().item()):
        raise RuntimeError("preprocessed patch must be finite float32")
    return tensor


class StreamingOpenSlideDataset(Dataset[torch.Tensor]):
    """Read and transform patches lazily without materializing patch images."""

    def __init__(
        self,
        wsi_path: str | Path,
        branch: PatchBranchSpec,
        *,
        slide_factory: Callable[[str], object] = openslide.OpenSlide,
    ) -> None:
        self.wsi_path = str(Path(wsi_path).resolve())
        self.branch = branch
        self.slide_factory = slide_factory
        self._slide: object | None = None

    def __len__(self) -> int:
        return int(self.branch.coordinates.shape[0])

    def _get_slide(self) -> object:
        if self._slide is None:
            self._slide = self.slide_factory(self.wsi_path)
        return self._slide

    def __getitem__(self, index: int) -> torch.Tensor:
        x, y = (int(value) for value in self.branch.coordinates[index].tolist())
        slide = self._get_slide()
        if self.branch.name == "scale_2x":
            patch = slide.read_region(
                (x, y), SOURCE_LEVEL, (SOURCE_FOOTPRINT, SOURCE_FOOTPRINT)
            )
            patch = downsample_source_patch(patch)
        else:
            patch = slide.read_region(
                (x, y), self.branch.patch_level, (self.branch.patch_size,) * 2
            ).convert("RGB")
        return prepare_patch_for_resnet(patch)

    def close(self) -> None:
        if self._slide is not None:
            close = getattr(self._slide, "close", None)
            if callable(close):
                close()
            self._slide = None

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_slide"] = None
        return state

    def __del__(self) -> None:  # pragma: no cover - defensive interpreter cleanup
        self.close()


def build_resnet50_imagenet1k_v2(checkpoint: str | Path) -> nn.Module:
    """Load the already checksum-gated V2 state dict without network access."""

    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError("ResNet50 checkpoint must contain a state dictionary")
    model = resnet50(weights=None)
    model.load_state_dict(state, strict=True)
    model.fc = nn.Identity()
    return model


def extract_feature_matrix(
    dataset: Dataset[torch.Tensor],
    model: nn.Module,
    *,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
    progress: Callable[[int, int], None] | None = None,
) -> ExtractionResult:
    """Stream one branch through a classifier-free model in fixed row order."""

    if len(dataset) <= 0:
        raise ValueError("extraction dataset must not be empty")
    if batch_size <= 0 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers non-negative")
    if device.type != "cuda":
        raise ValueError("the real pilot requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the extraction context")

    loader_kwargs: dict[str, object] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": True,
        "drop_last": False,
    }
    if num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **loader_kwargs)

    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    features = torch.empty((len(dataset), FEATURE_DIM), dtype=torch.float32)
    completed_rows = 0
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    import time

    start = time.perf_counter()
    model_forward_milliseconds = 0.0
    with torch.inference_mode():
        for batch in loader:
            if batch.dtype is not torch.float32:
                raise RuntimeError(f"input batch must be float32, got {batch.dtype}")
            batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)
            model_start = torch.cuda.Event(enable_timing=True)
            model_end = torch.cuda.Event(enable_timing=True)
            model_start.record()
            output = model(batch)
            model_end.record()
            if output.ndim != 2 or output.shape[1] != FEATURE_DIM:
                raise RuntimeError(
                    f"ResNet output must have shape [B,{FEATURE_DIM}], got {tuple(output.shape)}"
                )
            if output.dtype is not torch.float32 or not bool(torch.isfinite(output).all().item()):
                raise RuntimeError("ResNet output contains invalid dtype or non-finite values")
            model_forward_milliseconds += model_start.elapsed_time(model_end)
            next_row = completed_rows + output.shape[0]
            features[completed_rows:next_row].copy_(output.to(device="cpu"))
            completed_rows = next_row
            if progress is not None:
                progress(completed_rows, len(dataset))

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    peak = int(torch.cuda.max_memory_allocated(device))
    if completed_rows != len(dataset):
        raise RuntimeError(
            f"feature rows {completed_rows} do not match coordinates {len(dataset)}"
        )
    validate_feature_matrix(features, name="extracted_features")
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
    return ExtractionResult(
        features=features,
        streaming_extraction_seconds=elapsed,
        model_forward_seconds=model_forward_milliseconds / 1000.0,
        peak_gpu_memory_bytes=peak,
        batch_size=batch_size,
    )
