from __future__ import annotations

import os
from pathlib import Path
import subprocess
import types

import torch

from multiscale_feature_pilot.src.padding import (
    pad_patient_bags,
    to_released_healnet_orientation,
    validate_padded_batch,
)


def test_patient_orientation_is_2048_by_patch_count() -> None:
    patient = torch.arange(3 * 2048, dtype=torch.float32).reshape(3, 2048)

    oriented = to_released_healnet_orientation(patient)

    assert oriented.shape == (2048, 3)
    assert oriented.is_contiguous()
    assert torch.equal(oriented[:, 0], patient[0])


def test_variable_length_batch_and_mask_contract() -> None:
    patient_a = torch.full((180, 2048), 1.0, dtype=torch.float32)
    patient_b = torch.full((250, 2048), 2.0, dtype=torch.float32)

    result = pad_patient_bags([patient_a, patient_b])

    assert result.features.shape == (2, 2048, 250)
    assert result.valid_mask.shape == (2, 250)
    assert result.valid_mask.dtype is torch.bool
    assert result.lengths.tolist() == [180, 250]
    assert bool(result.valid_mask[0, :180].all())
    assert not bool(result.valid_mask[0, 180:].any())
    assert bool(result.valid_mask[1].all())
    assert torch.equal(result.features[0, :, :180], patient_a.transpose(0, 1))
    assert torch.equal(result.features[1], patient_b.transpose(0, 1))
    validate_padded_batch(result)


def test_no_padded_values_are_marked_valid() -> None:
    patient_a = torch.ones((2, 2048), dtype=torch.float32)
    patient_b = torch.ones((4, 2048), dtype=torch.float32)

    result = pad_patient_bags([patient_a, patient_b], pad_value=-7.0)

    invalid = ~result.valid_mask[0]
    assert invalid.tolist() == [False, False, True, True]
    assert torch.all(result.features[0, :, invalid] == -7.0)
    assert not bool(result.valid_mask[0, invalid].any())


def _load_official_healnet_class():
    official_repo = Path(
        os.environ.get(
            "HEALNET_OFFICIAL_REPO",
            str(Path(__file__).resolve().parents[3] / "healnet"),
        )
    )
    expected_commit = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
    resolved_commit = subprocess.run(
        ["git", "-C", str(official_repo), "rev-parse", "v0.1.0^{commit}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert resolved_commit == expected_commit

    source = subprocess.run(
        [
            "git",
            "-C",
            str(official_repo),
            "show",
            "v0.1.0:healnet/models/healnet.py",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("pilot_read_only_official_healnet_v0_1_0")
    module.__file__ = f"{official_repo}@v0.1.0:healnet/models/healnet.py"
    exec(
        compile(source, module.__file__, "exec"),
        module.__dict__,
    )
    return module.HealNet


def test_official_healnet_accepts_synthetic_released_wsi_shape() -> None:
    """Read-only, random-weight compatibility check; no checkpoint or training."""

    torch.manual_seed(0)
    combined = torch.randn((180, 2048), dtype=torch.float32)
    model_input = to_released_healnet_orientation(combined).unsqueeze(0)
    assert model_input.shape == (1, 2048, 180)

    HealNet = _load_official_healnet_class()
    model = HealNet(
        n_modalities=1,
        channel_dims=[180],
        num_spatial_axes=[1],
        out_dims=4,
        depth=1,
        num_freq_bands=2,
        max_freq=2.0,
        l_c=2,
        l_d=8,
        x_heads=1,
        l_heads=1,
        cross_dim_head=4,
        latent_dim_head=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        fourier_encode_data=True,
        self_per_cross_attn=0,
        final_classifier_head=True,
        snn=False,
    )
    model.eval()
    with torch.no_grad():
        output = model([model_input.clone()])

    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()


def test_official_healnet_accepts_separate_wsi_rna_mutation_cnv_inputs() -> None:
    """Verify the planned four-modality interface with synthetic values only."""

    torch.manual_seed(1)
    combined_wsi = torch.randn((180, 2048), dtype=torch.float32)
    wsi = to_released_healnet_orientation(combined_wsi).unsqueeze(0)
    rna = torch.randn((1, 1, 1523), dtype=torch.float32)
    mutation = torch.randn((1, 1, 1125), dtype=torch.float32)
    cnv = torch.randn((1, 1, 193), dtype=torch.float32)

    HealNet = _load_official_healnet_class()
    model = HealNet(
        n_modalities=4,
        channel_dims=[180, 1523, 1125, 193],
        num_spatial_axes=[1, 1, 1, 1],
        out_dims=4,
        depth=1,
        num_freq_bands=2,
        max_freq=2.0,
        l_c=2,
        l_d=8,
        x_heads=1,
        l_heads=1,
        cross_dim_head=4,
        latent_dim_head=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        fourier_encode_data=True,
        self_per_cross_attn=0,
        final_classifier_head=True,
        snn=False,
    )
    model.eval()
    with torch.no_grad():
        output = model([wsi, rna, mutation, cnv])

    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()


def test_patch_validity_mask_is_not_the_official_attention_axis() -> None:
    result = pad_patient_bags(
        [
            torch.zeros((180, 2048), dtype=torch.float32),
            torch.zeros((250, 2048), dtype=torch.float32),
        ]
    )

    # Released input is [B, 2048, P_max], so HEALNet's flattened attention
    # sequence has length 2048. The adapter mask tracks P_max patch slots.
    assert result.features.shape[1] == 2048
    assert result.valid_mask.shape[1] == 250
    assert result.features.shape[1] != result.valid_mask.shape[1]
