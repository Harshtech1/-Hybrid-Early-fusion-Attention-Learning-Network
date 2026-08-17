from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multiscale_feature_pilot.src import healnet_smoke
from multiscale_feature_pilot.src.healnet_smoke import (
    CNV_DIM,
    FEATURE_DIM,
    HEALNET_SMOKE_SEED,
    MUTATION_DIM,
    OFFICIAL_HEALNET_COMMIT,
    OFFICIAL_HEALNET_TAG,
    RNA_DIM,
    HealNetSmokeError,
    HealNetSmokeResult,
    run_one_patient_healnet_smoke,
)


OFFICIAL_REPO = Path(__file__).resolve().parents[3] / "healnet"


def _valid_inputs(patch_count: int = 3) -> dict[str, torch.Tensor]:
    return {
        "wsi": torch.zeros((1, FEATURE_DIM, patch_count), dtype=torch.float32),
        "rna": torch.zeros((1, 1, RNA_DIM), dtype=torch.float32),
        "mutation": torch.zeros((1, 1, MUTATION_DIM), dtype=torch.float32),
        "cnv": torch.zeros((1, 1, CNV_DIM), dtype=torch.float32),
    }


def test_dynamic_model_config_matches_committed_tiny_interface() -> None:
    assert healnet_smoke._model_kwargs(7) == {
        "n_modalities": 4,
        "channel_dims": [7, RNA_DIM, MUTATION_DIM, CNV_DIM],
        "num_spatial_axes": [1, 1, 1, 1],
        "out_dims": 4,
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


def test_real_input_contract_runs_pinned_four_modality_smoke() -> None:
    inputs = _valid_inputs(patch_count=3)

    result = run_one_patient_healnet_smoke(
        official_repo=OFFICIAL_REPO,
        **inputs,
    )

    assert isinstance(result, HealNetSmokeResult)
    assert result.official_tag == OFFICIAL_HEALNET_TAG
    assert result.official_commit == OFFICIAL_HEALNET_COMMIT
    assert result.seed == HEALNET_SMOKE_SEED
    assert result.device == "cpu"
    assert result.patch_count == 3
    assert result.input_shapes == (
        (1, FEATURE_DIM, 3),
        (1, 1, RNA_DIM),
        (1, 1, MUTATION_DIM),
        (1, 1, CNV_DIM),
    )
    assert result.input_dtype == "float32"
    assert result.model_channel_dims == (3, RNA_DIM, MUTATION_DIM, CNV_DIM)
    assert result.output_shape == (1, 4)
    assert result.output_dtype == "float32"
    assert result.output_finite
    assert result.attention_shapes == (
        (1, 2, FEATURE_DIM),
        (1, 2, 1),
        (1, 2, 1),
        (1, 2, 1),
    )
    assert result.attention_dtype == "float32"
    assert result.attention_finite


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        (
            "wsi",
            torch.zeros((1, FEATURE_DIM, 2), dtype=torch.float64),
            "wsi must have dtype float32",
        ),
        (
            "rna",
            torch.zeros((1, 1, RNA_DIM + 1), dtype=torch.float32),
            "rna must have shape",
        ),
    ],
)
def test_rejects_wrong_dtype_or_shape(
    field: str,
    replacement: torch.Tensor,
    message: str,
) -> None:
    inputs = _valid_inputs(patch_count=2)
    inputs[field] = replacement

    with pytest.raises(HealNetSmokeError, match=message):
        run_one_patient_healnet_smoke(
            official_repo=OFFICIAL_REPO,
            **inputs,
        )


def test_rejects_noncontiguous_released_wsi_layout() -> None:
    inputs = _valid_inputs(patch_count=2)
    inputs["wsi"] = torch.zeros(
        (1, 2, FEATURE_DIM), dtype=torch.float32
    ).transpose(1, 2)
    assert inputs["wsi"].shape == (1, FEATURE_DIM, 2)
    assert not inputs["wsi"].is_contiguous()

    with pytest.raises(HealNetSmokeError, match="wsi must be contiguous"):
        run_one_patient_healnet_smoke(
            official_repo=OFFICIAL_REPO,
            **inputs,
        )


def test_finite_logits_do_not_hide_swallowed_attention_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class SilentCrossAttentionFailure(torch.nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
            return torch.zeros((1, 4), dtype=torch.float32, device=tensors[0].device)

        def get_attention_weights(self) -> list[None]:
            return [None, None, None, None]

    monkeypatch.setattr(
        healnet_smoke,
        "load_pinned_official_healnet",
        lambda _: (SilentCrossAttentionFailure, OFFICIAL_HEALNET_COMMIT),
    )

    with pytest.raises(
        HealNetSmokeError,
        match="cross-attention modality 0 did not run",
    ):
        run_one_patient_healnet_smoke(
            official_repo=OFFICIAL_REPO,
            **_valid_inputs(patch_count=2),
        )
