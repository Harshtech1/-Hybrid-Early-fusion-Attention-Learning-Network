from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from multiscale_feature_pilot.src.brca_compact_feature_artifacts import (
    COMBINED_FILENAME,
    MANIFEST_FILENAME,
    CompactArtifactError,
    CompactArtifactExistsError,
    CompactFeatureMetadata,
    publish_compact_feature_artifacts,
    validate_compact_feature_artifacts,
)
from multiscale_feature_pilot.src.provenance import PatchProvenance


def _metadata() -> CompactFeatureMetadata:
    return CompactFeatureMetadata(
        patient_id="TCGA-AA-0001",
        slide_id="TCGA-AA-0001-01Z-00-DX1.TEST.svs",
        gdc_file_uuid="11111111-2222-3333-4444-555555555555",
        wsi_sha256="1" * 64,
        coordinate_manifest_sha256="2" * 64,
        omic_archive_sha256="3" * 64,
        checkpoint_sha256="4" * 64,
        source_policy_sha256="5" * 64,
        implementation_commit="6" * 40,
        scale_2x_rows=4,
        scale_4x_rows=2,
    )


def _records() -> tuple[PatchProvenance, ...]:
    rows = []
    for index in range(4):
        rows.append(PatchProvenance(index, "scale_2x", index, index * 512, 0, 0, 0.5, 0.5))
    for index in range(2):
        rows.append(PatchProvenance(4 + index, "scale_4x", index, index * 1024, 0, 1, 1.0, 1.0))
    return tuple(rows)


def _tensor() -> torch.Tensor:
    return torch.arange(6 * 2048, dtype=torch.float32).reshape(6, 2048).contiguous()


def test_publish_and_validate_exact_compact_set(tmp_path: Path) -> None:
    destination = tmp_path / "TCGA-AA-0001.features.compact"
    result = publish_compact_feature_artifacts(
        destination,
        combined_features=_tensor(),
        row_provenance=_records(),
        metadata=_metadata(),
    )
    assert result["tensor"]["shape"] == [6, 2048]
    assert result["tensor"]["scale_2x_row_range"] == [0, 4]
    assert result["tensor"]["scale_4x_row_range"] == [4, 6]
    assert result["retention"]["canonical_tensor_count"] == 1
    assert {item.name for item in destination.iterdir()} == {
        "combined_features.pt",
        "row_provenance.csv",
        "compact_manifest.json",
        "compact_manifest.json.sha256",
    }
    tensor = torch.load(destination / COMBINED_FILENAME, weights_only=True)
    assert torch.equal(tensor, _tensor())


def test_existing_destination_is_never_replaced(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    marker = destination / "owned.txt"
    marker.write_text("preserve")
    with pytest.raises(CompactArtifactExistsError):
        publish_compact_feature_artifacts(
            destination,
            combined_features=_tensor(),
            row_provenance=_records(),
            metadata=_metadata(),
        )
    assert marker.read_text() == "preserve"


def test_rejects_wrong_branch_boundary_before_write(tmp_path: Path) -> None:
    records = list(_records())
    records[3] = replace(records[3], branch="scale_4x")
    with pytest.raises(CompactArtifactError, match="invalid row provenance|2x prefix"):
        publish_compact_feature_artifacts(
            tmp_path / "bad",
            combined_features=_tensor(),
            row_provenance=records,
            metadata=_metadata(),
        )
    assert not (tmp_path / "bad").exists()


def test_rejects_nonfinite_or_wrong_layout(tmp_path: Path) -> None:
    tensor = _tensor()
    tensor[0, 0] = float("nan")
    with pytest.raises(CompactArtifactError, match="finite"):
        publish_compact_feature_artifacts(
            tmp_path / "nan",
            combined_features=tensor,
            row_provenance=_records(),
            metadata=_metadata(),
        )
    with pytest.raises(CompactArtifactError, match="float32"):
        publish_compact_feature_artifacts(
            tmp_path / "wrong",
            combined_features=_tensor().double(),
            row_provenance=_records(),
            metadata=_metadata(),
        )


def test_postpublication_tensor_corruption_is_detected(tmp_path: Path) -> None:
    destination = tmp_path / "compact"
    manifest = publish_compact_feature_artifacts(
        destination,
        combined_features=_tensor(),
        row_provenance=_records(),
        metadata=_metadata(),
    )
    manifest_sha = (destination / "compact_manifest.json.sha256").read_text().split()[0]
    with (destination / COMBINED_FILENAME).open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(CompactArtifactError, match="file SHA256"):
        validate_compact_feature_artifacts(
            destination, expected_manifest_sha256=manifest_sha
        )
    assert manifest["schema"] == "BRCA_COMPACT_FEATURE_ARTIFACT_SET_V1"


def test_manifest_branch_range_drift_is_anchored_by_sidecar(tmp_path: Path) -> None:
    destination = tmp_path / "compact"
    publish_compact_feature_artifacts(
        destination,
        combined_features=_tensor(),
        row_provenance=_records(),
        metadata=_metadata(),
    )
    sidecar_sha = (destination / "compact_manifest.json.sha256").read_text().split()[0]
    path = destination / MANIFEST_FILENAME
    document = json.loads(path.read_text())
    document["tensor"]["scale_2x_row_range"] = [0, 3]
    path.write_text(json.dumps(document))
    with pytest.raises(CompactArtifactError, match="manifest SHA256"):
        validate_compact_feature_artifacts(destination, expected_manifest_sha256=sidecar_sha)
