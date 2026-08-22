from __future__ import annotations

from pathlib import Path

import torch
import yaml

from multiscale_feature_pilot.src.supervisor_healnet_smoke import (
    run_one_patient_supervisor_healnet_smoke,
)
from multiscale_feature_pilot.src.supervisor_tensor import (
    build_natural_healnet_wsi_input,
)


PILOT_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_REPO = PILOT_ROOT.parent / "healnet"
CONFIG_PATH = PILOT_ROOT / "multiscale_feature_pilot/config/brca_pilot_config.yaml"
PROVENANCE_PATH = (
    PILOT_ROOT
    / "multiscale_feature_pilot/provenance/supervisor_tensor_policy.yaml"
)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict)
    return value


def test_config_and_provenance_encode_the_same_natural_wsi_contract() -> None:
    config = _load_yaml(CONFIG_PATH)
    provenance = _load_yaml(PROVENANCE_PATH)

    contract = config["four_input_contract"]
    per_wsi = provenance["per_wsi_contract"]
    cohort = provenance["cohort_contract"]

    assert contract["stored_wsi_shape"] == ["P_i", 2048]
    assert contract["model_wsi_shape"] == [1, "P_i", 2048]
    assert contract["wsi_channel_dimension"] == 2048
    assert contract["wsi_attention_length"] == "P_i"
    assert contract["cohort_representation"] == (
        "ragged_collection_of_per_wsi_bags"
    )
    assert contract["cross_patient_patch_concatenation"] == "prohibited"
    assert contract["per_wsi_global_pooling"] == "prohibited"
    assert contract["initial_execution_batch_size"] == 1
    assert contract["initial_padding"] == "prohibited"
    assert contract["initial_attention_mask"] == "none"

    assert per_wsi["stored_shape"] == contract["stored_wsi_shape"]
    assert per_wsi["model_shape"] == contract["model_wsi_shape"]
    assert per_wsi["healnet_channel_dimension"] == 2048
    assert per_wsi["healnet_attention_length"] == "P_i"
    assert cohort["representation"] == contract["cohort_representation"]
    assert cohort["initial_batch_size"] == 1
    assert cohort["initial_attention_mask"] == "none"


def test_actual_brca_widths_run_through_adapter_and_natural_cpu_smoke() -> None:
    patient_features = torch.zeros((2, 2048), dtype=torch.float32)
    wsi = build_natural_healnet_wsi_input(
        patient_features,
        patient_id="TCGA-AA-0001",
        wsi_id="TCGA-AA-0001-01Z-00-DX1.TEST.svs",
    )

    result = run_one_patient_supervisor_healnet_smoke(
        official_repo=OFFICIAL_REPO,
        wsi=wsi.tensor,
        rna=torch.zeros((1, 1, 1558), dtype=torch.float32),
        mutation=torch.zeros((1, 1, 21), dtype=torch.float32),
        cnv=torch.zeros((1, 1, 1333), dtype=torch.float32),
    )

    assert result.input_shapes == (
        (1, 2, 2048),
        (1, 1, 1558),
        (1, 1, 21),
        (1, 1, 1333),
    )
    assert result.model_channel_dims == (2048, 1558, 21, 1333)
    assert result.attention_shapes[0] == (1, 2, 2)
    assert result.output_shape == (1, 4)
