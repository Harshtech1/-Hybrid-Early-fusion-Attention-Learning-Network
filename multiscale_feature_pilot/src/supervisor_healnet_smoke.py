"""Read-only HEALNet v0.1.0 smoke for the natural patch-attention layout.

This helper validates the supervisor-policy tensor interpretation in which one
patient's WSI bag is ``[1, P, 2048]``: patches are the single spatial/attention
axis and each patch has 2,048 channels.  It is intentionally separate from
``healnet_smoke.py``, which preserves the frozen released-loader orientation
``[1, 2048, P]`` for historical reproducibility.

The smoke uses random model weights, batch size one, no padding, and no mask.
It neither reads WSI pixels nor trains a model.  The official model class is
loaded read-only from the pinned v0.1.0 Git object by the existing loader.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .healnet_smoke import (
    FEATURE_DIM,
    HEALNET_SMOKE_SEED,
    OFFICIAL_HEALNET_COMMIT,
    OFFICIAL_HEALNET_TAG,
    OUT_DIM,
    HealNetSmokeError,
    load_pinned_official_healnet,
)


NATURAL_WSI_LAYOUT = "[1,P,2048]"
FROZEN_RELEASED_WSI_LAYOUT = "[1,2048,P]"
NATURAL_WSI_ATTENTION_AXIS = "patches"


class SupervisorHealNetSmokeError(RuntimeError):
    """Raised when the natural one-patient HEALNet contract is violated."""


@dataclass(frozen=True)
class SupervisorHealNetSmokeResult:
    """Auditable facts from a successful natural-orientation forward pass."""

    official_tag: str
    official_commit: str
    seed: int
    interface: str
    wsi_layout: str
    frozen_released_wsi_layout: str
    wsi_attention_axis: str
    batch_size: int
    uses_padding: bool
    uses_mask: bool
    training: bool
    device: str
    patch_count: int
    omic_dims: tuple[int, int, int]
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtype: str
    model_channel_dims: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_dtype: str
    output_finite: bool
    attention_shapes: tuple[tuple[int, ...], ...]
    attention_dtype: str
    attention_finite: bool


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _validate_inputs(
    tensors: Sequence[torch.Tensor],
) -> tuple[torch.device, int, tuple[int, int, int]]:
    names = ("wsi", "rna", "mutation", "cnv")
    if len(tensors) != len(names):
        raise SupervisorHealNetSmokeError("exactly four input modalities are required")
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise SupervisorHealNetSmokeError("all four modalities must be torch tensors")

    wsi, *omics = tensors
    if wsi.ndim != 3:
        raise SupervisorHealNetSmokeError(
            f"wsi must have natural shape [1,P,{FEATURE_DIM}], got {_shape(wsi)}"
        )
    if wsi.shape[0] != 1:
        raise SupervisorHealNetSmokeError(
            f"this no-padding smoke requires batch size B=1, got B={wsi.shape[0]}"
        )
    if wsi.shape[1] <= 0 or wsi.shape[2] != FEATURE_DIM:
        raise SupervisorHealNetSmokeError(
            f"wsi must have natural shape [1,P,{FEATURE_DIM}], got {_shape(wsi)}; "
            f"do not pass the frozen released orientation {FROZEN_RELEASED_WSI_LAYOUT}"
        )

    omic_dims: list[int] = []
    for name, tensor in zip(names[1:], omics):
        if tensor.ndim != 3:
            raise SupervisorHealNetSmokeError(
                f"{name} must have shape [1,1,D], got {_shape(tensor)}"
            )
        if tensor.shape[0] != 1:
            raise SupervisorHealNetSmokeError(
                f"this no-padding smoke requires batch size B=1; "
                f"{name} has B={tensor.shape[0]}"
            )
        if tensor.shape[1] != 1 or tensor.shape[2] <= 0:
            raise SupervisorHealNetSmokeError(
                f"{name} must have shape [1,1,D] with D>0, got {_shape(tensor)}"
            )
        omic_dims.append(int(tensor.shape[2]))

    for name, tensor in zip(names, tensors):
        if tensor.dtype is not torch.float32:
            raise SupervisorHealNetSmokeError(
                f"{name} must have dtype float32, got {_dtype_name(tensor.dtype)}"
            )
        if not tensor.is_contiguous():
            raise SupervisorHealNetSmokeError(f"{name} must be contiguous")
        if not bool(torch.isfinite(tensor).all().item()):
            raise SupervisorHealNetSmokeError(f"{name} contains NaN or Inf")

    device = wsi.device
    for name, tensor in zip(names[1:], omics):
        if tensor.device != device:
            raise SupervisorHealNetSmokeError(
                f"all modalities must share one device; {name} is on "
                f"{tensor.device}, expected {device}"
            )

    validated_omic_dims = (omic_dims[0], omic_dims[1], omic_dims[2])
    return device, int(wsi.shape[1]), validated_omic_dims


def _model_kwargs(omic_dims: tuple[int, int, int]) -> dict[str, object]:
    """Return the exact tiny random-weight four-modality configuration."""

    return {
        "n_modalities": 4,
        "channel_dims": [FEATURE_DIM, *omic_dims],
        "num_spatial_axes": [1, 1, 1, 1],
        "out_dims": OUT_DIM,
        "depth": 1,
        "num_freq_bands": 2,
        "max_freq": 2.0,
        "l_c": 2,
        "l_d": 8,
        "x_heads": 1,
        "l_heads": 1,
        "cross_dim_head": 4,
        "latent_dim_head": 4,
        "attn_dropout": 0.0,
        "ff_dropout": 0.0,
        "weight_tie_layers": False,
        "fourier_encode_data": True,
        "self_per_cross_attn": 0,
        "final_classifier_head": True,
        "snn": False,
    }


def run_one_patient_supervisor_healnet_smoke(
    *,
    official_repo: str | Path,
    wsi: torch.Tensor,
    rna: torch.Tensor,
    mutation: torch.Tensor,
    cnv: torch.Tensor,
) -> SupervisorHealNetSmokeResult:
    """Run a deterministic ``[1,4]`` interface smoke over natural WSI input.

    Inputs are used exactly as supplied: this function never transposes,
    pads, masks, moves, or coerces them.  The forward is evaluation-only with
    random weights and is therefore an interface test, not inference or
    training.  RNA, mutation, and CNV widths may differ between cancer cohorts.
    """

    tensors = (wsi, rna, mutation, cnv)
    device, patch_count, omic_dims = _validate_inputs(tensors)
    try:
        healnet_class, peeled_commit = load_pinned_official_healnet(official_repo)
    except HealNetSmokeError as exc:
        raise SupervisorHealNetSmokeError(str(exc)) from exc

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            int(device.index)
            if device.index is not None
            else torch.cuda.current_device()
        )

    # Keep the caller's RNG state unchanged while fixing model initialization.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(HEALNET_SMOKE_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(HEALNET_SMOKE_SEED)

        model_kwargs = _model_kwargs(omic_dims)
        model = healnet_class(**model_kwargs).float().to(device)
        model.eval()
        # Official v0.1.0 mutates the list container during preprocessing.
        forward_inputs = [wsi, rna, mutation, cnv]
        with torch.inference_mode():
            output = model(forward_inputs)

    expected_output_shape = (1, OUT_DIM)
    if _shape(output) != expected_output_shape:
        raise SupervisorHealNetSmokeError(
            f"HEALNet output must have shape {expected_output_shape}, got {_shape(output)}"
        )
    if output.dtype is not torch.float32:
        raise SupervisorHealNetSmokeError(
            f"HEALNet output must have dtype float32, got {_dtype_name(output.dtype)}"
        )
    if output.device != device:
        raise SupervisorHealNetSmokeError(
            f"HEALNet output is on {output.device}, expected {device}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise SupervisorHealNetSmokeError("HEALNet output contains NaN or Inf")

    # v0.1.0 swallows cross-attention exceptions.  Require evidence that all
    # four cross-attention modules ran and attended over the intended axes.
    attention_weights = model.get_attention_weights()
    expected_attention_shapes = (
        (1, 2, patch_count),
        (1, 2, 1),
        (1, 2, 1),
        (1, 2, 1),
    )
    if len(attention_weights) != len(expected_attention_shapes):
        raise SupervisorHealNetSmokeError(
            "expected exactly four cross-attention tensors, "
            f"found {len(attention_weights)}"
        )

    observed_attention_shapes: list[tuple[int, ...]] = []
    for index, (weights, expected_shape) in enumerate(
        zip(attention_weights, expected_attention_shapes)
    ):
        if not isinstance(weights, torch.Tensor):
            raise SupervisorHealNetSmokeError(
                f"cross-attention modality {index} did not run"
            )
        observed_shape = _shape(weights)
        if observed_shape != expected_shape:
            axis_description = "P patches" if index == 0 else "one Omic token"
            raise SupervisorHealNetSmokeError(
                f"cross-attention modality {index} must attend over "
                f"{axis_description} with shape {expected_shape}, got {observed_shape}"
            )
        if weights.dtype is not torch.float32:
            raise SupervisorHealNetSmokeError(
                f"cross-attention modality {index} must have dtype float32, "
                f"got {_dtype_name(weights.dtype)}"
            )
        if weights.device != device:
            raise SupervisorHealNetSmokeError(
                f"cross-attention modality {index} is on {weights.device}, "
                f"expected {device}"
            )
        if not bool(torch.isfinite(weights).all().item()):
            raise SupervisorHealNetSmokeError(
                f"cross-attention modality {index} contains NaN or Inf"
            )
        observed_attention_shapes.append(observed_shape)

    return SupervisorHealNetSmokeResult(
        official_tag=OFFICIAL_HEALNET_TAG,
        official_commit=peeled_commit,
        seed=HEALNET_SMOKE_SEED,
        interface="supervisor-policy natural patch attention",
        wsi_layout=NATURAL_WSI_LAYOUT,
        frozen_released_wsi_layout=FROZEN_RELEASED_WSI_LAYOUT,
        wsi_attention_axis=NATURAL_WSI_ATTENTION_AXIS,
        batch_size=1,
        uses_padding=False,
        uses_mask=False,
        training=False,
        device=str(device),
        patch_count=patch_count,
        omic_dims=omic_dims,
        input_shapes=tuple(_shape(tensor) for tensor in tensors),
        input_dtype="float32",
        model_channel_dims=(FEATURE_DIM, *omic_dims),
        output_shape=_shape(output),
        output_dtype=_dtype_name(output.dtype),
        output_finite=True,
        attention_shapes=tuple(observed_attention_shapes),
        attention_dtype="float32",
        attention_finite=True,
    )
