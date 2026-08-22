from __future__ import annotations

from pathlib import Path

import pytest
import torch

from multiscale_feature_pilot.src import supervisor_healnet_smoke
from multiscale_feature_pilot.src.healnet_smoke import (
    CNV_DIM,
    FEATURE_DIM,
    HEALNET_SMOKE_SEED,
    MUTATION_DIM,
    OFFICIAL_HEALNET_COMMIT,
    OFFICIAL_HEALNET_TAG,
    RNA_DIM,
)
from multiscale_feature_pilot.src.supervisor_healnet_smoke import (
    FROZEN_RELEASED_WSI_LAYOUT,
    NATURAL_WSI_ATTENTION_AXIS,
    NATURAL_WSI_LAYOUT,
    SupervisorHealNetSmokeError,
    SupervisorHealNetSmokeResult,
    run_one_patient_supervisor_healnet_smoke,
)


OFFICIAL_REPO = Path(__file__).resolve().parents[3] / "healnet"


def _valid_inputs(
    patch_count: int = 3,
    *,
    batch_size: int = 1,
) -> dict[str, torch.Tensor]:
    return {
        "wsi": torch.zeros(
            (batch_size, patch_count, FEATURE_DIM), dtype=torch.float32
        ),
        "rna": torch.zeros((batch_size, 1, RNA_DIM), dtype=torch.float32),
        "mutation": torch.zeros(
            (batch_size, 1, MUTATION_DIM), dtype=torch.float32
        ),
        "cnv": torch.zeros((batch_size, 1, CNV_DIM), dtype=torch.float32),
    }


def test_model_config_fixes_2048_as_wsi_channels() -> None:
    assert supervisor_healnet_smoke._model_kwargs(
        (RNA_DIM, MUTATION_DIM, CNV_DIM)
    ) == {
        "n_modalities": 4,
        "channel_dims": [FEATURE_DIM, RNA_DIM, MUTATION_DIM, CNV_DIM],
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


def test_natural_interface_runs_pinned_four_modality_cpu_smoke() -> None:
    result = run_one_patient_supervisor_healnet_smoke(
        official_repo=OFFICIAL_REPO,
        **_valid_inputs(patch_count=3),
    )

    assert isinstance(result, SupervisorHealNetSmokeResult)
    assert result.official_tag == OFFICIAL_HEALNET_TAG
    assert result.official_commit == OFFICIAL_HEALNET_COMMIT
    assert result.seed == HEALNET_SMOKE_SEED
    assert result.interface == "supervisor-policy natural patch attention"
    assert result.wsi_layout == NATURAL_WSI_LAYOUT
    assert result.frozen_released_wsi_layout == FROZEN_RELEASED_WSI_LAYOUT
    assert result.wsi_layout != result.frozen_released_wsi_layout
    assert result.wsi_attention_axis == NATURAL_WSI_ATTENTION_AXIS
    assert result.batch_size == 1
    assert not result.uses_padding
    assert not result.uses_mask
    assert not result.training
    assert result.device == "cpu"
    assert result.patch_count == 3
    assert result.omic_dims == (RNA_DIM, MUTATION_DIM, CNV_DIM)
    assert result.input_shapes == (
        (1, 3, FEATURE_DIM),
        (1, 1, RNA_DIM),
        (1, 1, MUTATION_DIM),
        (1, 1, CNV_DIM),
    )
    assert result.input_dtype == "float32"
    assert result.model_channel_dims == (
        FEATURE_DIM,
        RNA_DIM,
        MUTATION_DIM,
        CNV_DIM,
    )
    assert result.output_shape == (1, 4)
    assert result.output_dtype == "float32"
    assert result.output_finite
    assert result.attention_shapes == (
        (1, 2, 3),
        (1, 2, 1),
        (1, 2, 1),
        (1, 2, 1),
    )
    assert result.attention_shapes[0][-1] == result.patch_count
    assert result.attention_shapes[0][-1] != FEATURE_DIM
    assert result.attention_dtype == "float32"
    assert result.attention_finite


def test_rejects_frozen_released_orientation_instead_of_transposing_it() -> None:
    inputs = _valid_inputs(patch_count=3)
    inputs["wsi"] = torch.zeros((1, FEATURE_DIM, 3), dtype=torch.float32)

    with pytest.raises(
        SupervisorHealNetSmokeError,
        match=r"natural shape \[1,P,2048\].*frozen released orientation",
    ):
        run_one_patient_supervisor_healnet_smoke(
            official_repo=OFFICIAL_REPO,
            **inputs,
        )


def test_rejects_batch_greater_than_one_before_forward() -> None:
    with pytest.raises(
        SupervisorHealNetSmokeError,
        match="requires batch size B=1, got B=2",
    ):
        run_one_patient_supervisor_healnet_smoke(
            official_repo=OFFICIAL_REPO,
            **_valid_inputs(patch_count=3, batch_size=2),
        )


def test_accepts_cohort_specific_omic_widths() -> None:
    inputs = {
        "wsi": torch.zeros((1, 2, FEATURE_DIM), dtype=torch.float32),
        "rna": torch.zeros((1, 1, 5), dtype=torch.float32),
        "mutation": torch.zeros((1, 1, 7), dtype=torch.float32),
        "cnv": torch.zeros((1, 1, 11), dtype=torch.float32),
    }

    result = run_one_patient_supervisor_healnet_smoke(
        official_repo=OFFICIAL_REPO,
        **inputs,
    )

    assert result.omic_dims == (5, 7, 11)
    assert result.model_channel_dims == (FEATURE_DIM, 5, 7, 11)
    assert result.attention_shapes[0] == (1, 2, 2)


def test_finite_logits_do_not_hide_wrong_wsi_attention_axis(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ReleasedAxisPretender(torch.nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
            return torch.zeros((1, 4), dtype=torch.float32, device=tensors[0].device)

        def get_attention_weights(self) -> list[torch.Tensor]:
            return [
                torch.zeros((1, 2, FEATURE_DIM), dtype=torch.float32),
                torch.zeros((1, 2, 1), dtype=torch.float32),
                torch.zeros((1, 2, 1), dtype=torch.float32),
                torch.zeros((1, 2, 1), dtype=torch.float32),
            ]

    monkeypatch.setattr(
        supervisor_healnet_smoke,
        "load_pinned_official_healnet",
        lambda _: (ReleasedAxisPretender, OFFICIAL_HEALNET_COMMIT),
    )

    with pytest.raises(
        SupervisorHealNetSmokeError,
        match="must attend over P patches",
    ):
        run_one_patient_supervisor_healnet_smoke(
            official_repo=OFFICIAL_REPO,
            **_valid_inputs(patch_count=3),
        )


def test_finite_logits_do_not_hide_missing_attention_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MissingCnvAttention(torch.nn.Module):
        def __init__(self, **_: object) -> None:
            super().__init__()

        def forward(self, tensors: list[torch.Tensor]) -> torch.Tensor:
            return torch.zeros((1, 4), dtype=torch.float32, device=tensors[0].device)

        def get_attention_weights(self) -> list[torch.Tensor | None]:
            return [
                torch.zeros((1, 2, tensors_patch_count), dtype=torch.float32),
                torch.zeros((1, 2, 1), dtype=torch.float32),
                torch.zeros((1, 2, 1), dtype=torch.float32),
                None,
            ]

    tensors_patch_count = 3
    monkeypatch.setattr(
        supervisor_healnet_smoke,
        "load_pinned_official_healnet",
        lambda _: (MissingCnvAttention, OFFICIAL_HEALNET_COMMIT),
    )

    with pytest.raises(
        SupervisorHealNetSmokeError,
        match="cross-attention modality 3 did not run",
    ):
        run_one_patient_supervisor_healnet_smoke(
            official_repo=OFFICIAL_REPO,
            **_valid_inputs(patch_count=tensors_patch_count),
        )
