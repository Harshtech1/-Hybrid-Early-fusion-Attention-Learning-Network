"""Pinned, random-weight HEALNet interface smoke test for one real patient.

This module deliberately loads the released model source from the immutable
``v0.1.0`` Git object.  It does not import or modify the official checkout's
working tree, load trained HEALNet weights, or perform scientific inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import types
from typing import Sequence

import torch


OFFICIAL_HEALNET_TAG = "v0.1.0"
OFFICIAL_HEALNET_COMMIT = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
OFFICIAL_HEALNET_SOURCE = "healnet/models/healnet.py"
HEALNET_SMOKE_SEED = 1

FEATURE_DIM = 2048
RNA_DIM = 1523
MUTATION_DIM = 1125
CNV_DIM = 193
OUT_DIM = 4


class HealNetSmokeError(RuntimeError):
    """Raised when the pinned interface or one-patient smoke contract fails."""


@dataclass(frozen=True)
class HealNetSmokeResult:
    """Serializable facts from a successful random-weight interface forward."""

    official_tag: str
    official_commit: str
    seed: int
    device: str
    patch_count: int
    input_shapes: tuple[tuple[int, ...], ...]
    input_dtype: str
    model_channel_dims: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_dtype: str
    output_finite: bool
    attention_shapes: tuple[tuple[int, ...], ...]
    attention_dtype: str
    attention_finite: bool


def _run_git(official_repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(official_repo), *args],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        detail = stderr.strip() or str(exc)
        raise HealNetSmokeError(
            f"could not read pinned official HEALNet source: {detail}"
        ) from exc
    return result.stdout


def load_pinned_official_healnet(
    official_repo: str | Path,
) -> tuple[type[torch.nn.Module], str]:
    """Load ``HealNet`` from the verified ``v0.1.0`` Git object, read-only."""

    repo = Path(official_repo).resolve()
    if not repo.is_dir():
        raise HealNetSmokeError(f"official HEALNet repository does not exist: {repo}")

    peeled_commit = _run_git(
        repo,
        "rev-parse",
        f"{OFFICIAL_HEALNET_TAG}^{{commit}}",
    ).strip()
    if peeled_commit != OFFICIAL_HEALNET_COMMIT:
        raise HealNetSmokeError(
            "official HEALNet tag mismatch: "
            f"expected {OFFICIAL_HEALNET_COMMIT}, got {peeled_commit}"
        )

    source = _run_git(
        repo,
        "show",
        f"{OFFICIAL_HEALNET_TAG}:{OFFICIAL_HEALNET_SOURCE}",
    )
    module = types.ModuleType("pilot_read_only_official_healnet_v0_1_0")
    module.__file__ = (
        f"{repo}@{OFFICIAL_HEALNET_TAG}:{OFFICIAL_HEALNET_SOURCE}"
    )
    try:
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        healnet_class = module.HealNet
    except (AttributeError, ImportError, SyntaxError) as exc:
        raise HealNetSmokeError("could not load HealNet from pinned source") from exc

    return healnet_class, peeled_commit


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(int(dimension) for dimension in tensor.shape)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _validate_inputs(tensors: Sequence[torch.Tensor]) -> torch.device:
    names = ("wsi", "rna", "mutation", "cnv")
    if len(tensors) != len(names):
        raise HealNetSmokeError("exactly four input modalities are required")
    if any(not isinstance(tensor, torch.Tensor) for tensor in tensors):
        raise HealNetSmokeError("all four modalities must be torch tensors")

    wsi, rna, mutation, cnv = tensors
    if wsi.ndim != 3 or wsi.shape[0] != 1 or wsi.shape[1] != FEATURE_DIM:
        raise HealNetSmokeError(
            f"wsi must have shape [1,{FEATURE_DIM},P], got {_shape(wsi)}"
        )
    if wsi.shape[2] <= 0:
        raise HealNetSmokeError("wsi patch count P must be positive")

    expected_shapes = (
        (1, FEATURE_DIM, int(wsi.shape[2])),
        (1, 1, RNA_DIM),
        (1, 1, MUTATION_DIM),
        (1, 1, CNV_DIM),
    )
    for name, tensor, expected_shape in zip(names, tensors, expected_shapes):
        if _shape(tensor) != expected_shape:
            raise HealNetSmokeError(
                f"{name} must have shape {expected_shape}, got {_shape(tensor)}"
            )
        if tensor.dtype is not torch.float32:
            raise HealNetSmokeError(
                f"{name} must have dtype float32, got {_dtype_name(tensor.dtype)}"
            )
        if not tensor.is_contiguous():
            raise HealNetSmokeError(f"{name} must be contiguous")
        if not bool(torch.isfinite(tensor).all().item()):
            raise HealNetSmokeError(f"{name} contains NaN or Inf")

    device = wsi.device
    for name, tensor in zip(names[1:], tensors[1:]):
        if tensor.device != device:
            raise HealNetSmokeError(
                f"all modalities must share one device; {name} is on "
                f"{tensor.device}, expected {device}"
            )
    return device


def _model_kwargs(patch_count: int) -> dict[str, object]:
    """Return the committed tiny four-modality compatibility configuration."""

    return {
        "n_modalities": 4,
        "channel_dims": [patch_count, RNA_DIM, MUTATION_DIM, CNV_DIM],
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


def run_one_patient_healnet_smoke(
    *,
    official_repo: str | Path,
    wsi: torch.Tensor,
    rna: torch.Tensor,
    mutation: torch.Tensor,
    cnv: torch.Tensor,
) -> HealNetSmokeResult:
    """Run one finite ``[1,4]`` random-weight interface forward.

    The caller supplies already validated real tensors.  This helper enforces
    their released orientation and does not move or coerce them implicitly.
    ``channel_dims[0]`` is the runtime patch count because the released
    ``[1,2048,P]`` layout treats 2048 as its single attention axis.
    """

    tensors = (wsi, rna, mutation, cnv)
    device = _validate_inputs(tensors)
    patch_count = int(wsi.shape[2])
    healnet_class, peeled_commit = load_pinned_official_healnet(official_repo)

    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices.append(
            int(device.index) if device.index is not None else torch.cuda.current_device()
        )

    # Isolate the smoke model's fixed random initialization from caller RNG state.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(HEALNET_SMOKE_SEED)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(HEALNET_SMOKE_SEED)

        model = healnet_class(**_model_kwargs(patch_count)).float().to(device)
        model.eval()
        # The official forward mutates the list container, so always pass a new one.
        forward_inputs = [wsi, rna, mutation, cnv]
        with torch.inference_mode():
            output = model(forward_inputs)

    expected_output_shape = (1, OUT_DIM)
    if _shape(output) != expected_output_shape:
        raise HealNetSmokeError(
            f"HEALNet output must have shape {expected_output_shape}, got {_shape(output)}"
        )
    if output.dtype is not torch.float32:
        raise HealNetSmokeError(
            f"HEALNet output must have dtype float32, got {_dtype_name(output.dtype)}"
        )
    if output.device != device:
        raise HealNetSmokeError(
            f"HEALNet output is on {output.device}, expected {device}"
        )
    if not bool(torch.isfinite(output).all().item()):
        raise HealNetSmokeError("HEALNet output contains NaN or Inf")

    # The released forward swallows cross-attention exceptions.  A finite
    # classifier output alone is therefore insufficient evidence that any
    # modality participated; require every expected attention tensor.
    attention_weights = model.get_attention_weights()
    expected_attention_shapes = (
        (1, 2, FEATURE_DIM),
        (1, 2, 1),
        (1, 2, 1),
        (1, 2, 1),
    )
    if len(attention_weights) != len(expected_attention_shapes):
        raise HealNetSmokeError(
            "expected four cross-attention tensors, "
            f"found {len(attention_weights)}"
        )

    observed_attention_shapes: list[tuple[int, ...]] = []
    for index, (weights, expected_shape) in enumerate(
        zip(attention_weights, expected_attention_shapes)
    ):
        if not isinstance(weights, torch.Tensor):
            raise HealNetSmokeError(
                f"cross-attention modality {index} did not run"
            )
        observed_shape = _shape(weights)
        if observed_shape != expected_shape:
            raise HealNetSmokeError(
                f"cross-attention modality {index} must have shape "
                f"{expected_shape}, got {observed_shape}"
            )
        if weights.dtype is not torch.float32:
            raise HealNetSmokeError(
                f"cross-attention modality {index} must be float32"
            )
        if weights.device != device:
            raise HealNetSmokeError(
                f"cross-attention modality {index} is on {weights.device}, "
                f"expected {device}"
            )
        if not bool(torch.isfinite(weights).all().item()):
            raise HealNetSmokeError(
                f"cross-attention modality {index} contains NaN or Inf"
            )
        observed_attention_shapes.append(observed_shape)

    return HealNetSmokeResult(
        official_tag=OFFICIAL_HEALNET_TAG,
        official_commit=peeled_commit,
        seed=HEALNET_SMOKE_SEED,
        device=str(device),
        patch_count=patch_count,
        input_shapes=tuple(_shape(tensor) for tensor in tensors),
        input_dtype="float32",
        model_channel_dims=(patch_count, RNA_DIM, MUTATION_DIM, CNV_DIM),
        output_shape=_shape(output),
        output_dtype=_dtype_name(output.dtype),
        output_finite=True,
        attention_shapes=tuple(observed_attention_shapes),
        attention_dtype="float32",
        attention_finite=True,
    )
