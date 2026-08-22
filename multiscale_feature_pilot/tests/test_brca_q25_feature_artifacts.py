from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch

import multiscale_feature_pilot.src.brca_q25_feature_artifacts as artifacts
from multiscale_feature_pilot.src.brca_q25_feature_artifacts import (
    EXPECTED_CHECKPOINT_FILENAME,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE_BYTES,
    EXPECTED_ENCODER_IDENTITY,
    FEATURE_FILENAMES,
    MANIFEST_FILENAME,
    MANIFEST_SHA256_FILENAME,
    PROVENANCE_FILENAME,
    CoordinateFeatureBinding,
    FeatureArtifactExistsError,
    FeatureArtifactMetadata,
    FeatureHashMismatchError,
    FeaturePublicationInProgressError,
    FeatureValidationError,
    publish_brca_q25_feature_artifacts,
    validate_brca_q25_feature_artifacts,
)
from multiscale_feature_pilot.src.provenance import PatchProvenance


@pytest.fixture(autouse=True)
def _small_synthetic_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise the real code paths without allocating the 153-MB Q25 set."""

    monkeypatch.setattr(artifacts, "SCALE_2X_ROWS", 4)
    monkeypatch.setattr(artifacts, "SCALE_4X_ROWS", 2)
    monkeypatch.setattr(artifacts, "FEATURE_DIM", 3)


def _coordinate_hash(coordinates: list[tuple[int, int]]) -> str:
    array = np.asarray(coordinates, dtype=np.dtype("<i8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _provenance() -> tuple[PatchProvenance, ...]:
    scale_2x_coordinates = [(0, 0), (512, 0), (0, 512), (512, 512)]
    scale_4x_coordinates = [(0, 0), (1024, 0)]
    result: list[PatchProvenance] = []
    for local_index, (x, y) in enumerate(scale_2x_coordinates):
        result.append(
            PatchProvenance(
                global_row_index=local_index,
                branch="scale_2x",
                local_patch_index=local_index,
                x=x,
                y=y,
                level=0,
                mpp_x=0.505,
                mpp_y=0.505,
            )
        )
    for local_index, (x, y) in enumerate(scale_4x_coordinates):
        result.append(
            PatchProvenance(
                global_row_index=4 + local_index,
                branch="scale_4x",
                local_patch_index=local_index,
                x=x,
                y=y,
                level=1,
                mpp_x=1.0100149842739303,
                mpp_y=1.0100149842739303,
            )
        )
    return tuple(result)


def _metadata() -> FeatureArtifactMetadata:
    rows = _provenance()
    return FeatureArtifactMetadata(
        patient_id="TCGA-LL-A6FP",
        slide_id="TCGA-LL-A6FP-01Z-00-DX1",
        gdc_file_uuid="dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6",
        wsi_filename="TCGA-LL-A6FP-01Z-00-DX1.TEST.svs",
        wsi_size_bytes=648_046_947,
        wsi_md5="1" * 32,
        wsi_sha256="2" * 64,
        coordinate_manifest_filename="coordinate_manifest.json",
        coordinate_manifest_size_bytes=6_470,
        coordinate_manifest_sha256="3" * 64,
        scale_2x_coordinates=CoordinateFeatureBinding(
            branch="scale_2x",
            artifact_filename="scale_2x_coordinates.h5",
            artifact_size_bytes=128_136,
            artifact_sha256="4" * 64,
            coordinates_sha256=_coordinate_hash([(row.x, row.y) for row in rows[:4]]),
            coordinate_count=4,
            source_level=0,
            effective_mpp_x=0.505,
            effective_mpp_y=0.505,
        ),
        scale_4x_coordinates=CoordinateFeatureBinding(
            branch="scale_4x",
            artifact_filename="scale_4x_coordinates.h5",
            artifact_size_bytes=40_360,
            artifact_sha256="5" * 64,
            coordinates_sha256=_coordinate_hash([(row.x, row.y) for row in rows[4:]]),
            coordinate_count=2,
            source_level=1,
            effective_mpp_x=1.0100149842739303,
            effective_mpp_y=1.0100149842739303,
        ),
        encoder_identity=EXPECTED_ENCODER_IDENTITY,
        checkpoint_filename=EXPECTED_CHECKPOINT_FILENAME,
        checkpoint_size_bytes=EXPECTED_CHECKPOINT_SIZE_BYTES,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        source_policy_name="BRCA_Q25_GPU_FEATURE_EXTRACTION_POLICY_V1",
        source_policy_sha256="6" * 64,
        implementation_git_commit="7" * 40,
        implementation_source_sha256={
            "multiscale_feature_pilot/src/feature_extraction.py": "8" * 64,
            "scripts/run_brca_q25_gpu_pilot.py": "9" * 64,
        },
    )


def _features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_2x = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
    scale_4x = (torch.arange(6, dtype=torch.float32) + 100).reshape(2, 3).contiguous()
    return scale_2x, scale_4x, torch.cat((scale_2x, scale_4x), dim=0)


def _publish(destination: Path):
    scale_2x, scale_4x, combined = _features()
    return publish_brca_q25_feature_artifacts(
        destination,
        scale_2x_features=scale_2x,
        scale_4x_features=scale_4x,
        combined_features=combined,
        row_provenance=_provenance(),
        metadata=_metadata(),
    )


def test_public_contract_constants_are_the_real_q25_shapes() -> None:
    # The fixture patches runtime globals only after collection; these source
    # constants are checked by reading the module file so the test remains fast.
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    assert "SCALE_2X_ROWS: Final = 7_404" in source
    assert "SCALE_4X_ROWS: Final = 1_918" in source
    assert "FEATURE_DIM: Final = 2_048" in source


def test_publishes_exact_set_validates_readback_and_is_byte_deterministic(
    tmp_path: Path,
) -> None:
    first = _publish(tmp_path / "q25_features_a")
    second = _publish(tmp_path / "q25_features_b")

    expected_names = {
        *FEATURE_FILENAMES.values(),
        PROVENANCE_FILENAME,
        MANIFEST_FILENAME,
        MANIFEST_SHA256_FILENAME,
    }
    assert {entry.name for entry in first.directory.iterdir()} == expected_names
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest_path.read_bytes() == second.manifest_path.read_bytes()
    for name in FEATURE_FILENAMES:
        assert first.feature_for(name).sha256 == second.feature_for(name).sha256
        assert first.feature_for(name).tensor_content_sha256 == second.feature_for(name).tensor_content_sha256
    assert first.provenance.sha256 == second.provenance.sha256
    assert first.manifest_sha256_path.read_text(encoding="ascii") == (
        f"{first.manifest_sha256}  {MANIFEST_FILENAME}\n"
    )

    validated = validate_brca_q25_feature_artifacts(
        first.directory,
        expected_manifest_sha256=first.manifest_sha256,
    )
    assert validated.feature_for("scale_2x_features").shape == (4, 3)
    assert validated.feature_for("scale_4x_features").shape == (2, 3)
    assert validated.feature_for("combined_features").shape == (6, 3)
    assert validated.provenance.row_count == 6
    assert validated.metadata.patient_id == "TCGA-LL-A6FP"
    manifest = json.loads(validated.manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract"]["branch_order"] == ["scale_2x", "scale_4x"]
    assert manifest["contract"]["concatenation"] == "torch.cat([scale_2x,scale_4x],dim=0)"
    assert manifest["metadata"]["checkpoint"]["sha256"] == EXPECTED_CHECKPOINT_SHA256
    assert not list(tmp_path.glob(".q25_features_a.staging.*"))
    assert not (tmp_path / ".q25_features_a.lock").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_shape", "shape must be"),
        ("wrong_dtype", "dtype must be torch.float32"),
        ("nonfinite", "contains NaN or Inf"),
        ("noncontiguous", "must be contiguous"),
        ("wrong_combination", "scale_2x rows first"),
    ],
)
def test_rejects_invalid_feature_inputs_without_partial_output(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    scale_2x, scale_4x, combined = _features()
    if mutation == "wrong_shape":
        scale_2x = torch.zeros((3, 3), dtype=torch.float32)
    elif mutation == "wrong_dtype":
        scale_2x = scale_2x.to(dtype=torch.float64)
    elif mutation == "nonfinite":
        scale_2x[0, 0] = torch.nan
    elif mutation == "noncontiguous":
        scale_2x = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    else:
        combined = combined.clone()
        combined[0, 0] = -1
    destination = tmp_path / "invalid"
    with pytest.raises(FeatureValidationError, match=message):
        publish_brca_q25_feature_artifacts(
            destination,
            scale_2x_features=scale_2x,
            scale_4x_features=scale_4x,
            combined_features=combined,
            row_provenance=_provenance(),
            metadata=_metadata(),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".invalid.staging.*"))
    assert not (tmp_path / ".invalid.lock").exists()


def test_provenance_must_bind_coordinate_hash_order_level_and_mpp(tmp_path: Path) -> None:
    scale_2x, scale_4x, combined = _features()
    metadata = _metadata()
    metadata = replace(
        metadata,
        scale_2x_coordinates=replace(
            metadata.scale_2x_coordinates,
            coordinates_sha256="0" * 64,
        ),
    )
    with pytest.raises(FeatureHashMismatchError, match="coordinate-content SHA-256"):
        publish_brca_q25_feature_artifacts(
            tmp_path / "hash_drift",
            scale_2x_features=scale_2x,
            scale_4x_features=scale_4x,
            combined_features=combined,
            row_provenance=_provenance(),
            metadata=metadata,
        )

    rows = list(_provenance())
    rows[4] = replace(rows[4], branch="scale_2x")
    with pytest.raises(FeatureValidationError, match="branch order"):
        publish_brca_q25_feature_artifacts(
            tmp_path / "order_drift",
            scale_2x_features=scale_2x,
            scale_4x_features=scale_4x,
            combined_features=combined,
            row_provenance=rows,
            metadata=_metadata(),
        )


def test_no_clobber_stale_staging_and_lock_are_fail_closed(tmp_path: Path) -> None:
    destination = tmp_path / "q25"
    result = _publish(destination)
    manifest_before = result.manifest_path.read_bytes()
    with pytest.raises(FeatureArtifactExistsError, match="already exists"):
        _publish(destination)
    assert result.manifest_path.read_bytes() == manifest_before

    (tmp_path / ".stale.staging.abandoned").mkdir()
    with pytest.raises(FeaturePublicationInProgressError, match="staging"):
        _publish(tmp_path / "stale")

    lock = tmp_path / ".locked.lock"
    lock.write_text("pid=123\n", encoding="ascii")
    with pytest.raises(FeaturePublicationInProgressError, match="locked"):
        _publish(tmp_path / "locked")
    assert lock.read_text(encoding="ascii") == "pid=123\n"


def test_external_hashes_exact_file_set_and_regular_file_checks_detect_tampering(
    tmp_path: Path,
) -> None:
    result = _publish(tmp_path / "q25")
    with pytest.raises(FeatureHashMismatchError, match="manifest SHA-256"):
        validate_brca_q25_feature_artifacts(
            result.directory,
            expected_manifest_sha256="0" * 64,
        )

    extra = result.directory / "resume.partial"
    extra.write_text("not allowed", encoding="utf-8")
    with pytest.raises(FeatureValidationError, match="file set is not exact"):
        validate_brca_q25_feature_artifacts(
            result.directory,
            expected_manifest_sha256=result.manifest_sha256,
        )
    extra.unlink()

    tensor_path = result.directory / FEATURE_FILENAMES["scale_2x_features"]
    with tensor_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(FeatureHashMismatchError, match="file SHA-256"):
        validate_brca_q25_feature_artifacts(
            result.directory,
            expected_manifest_sha256=result.manifest_sha256,
        )


def test_symlink_artifacts_and_destinations_inside_git_are_rejected(tmp_path: Path) -> None:
    result = _publish(tmp_path / "q25")
    provenance = result.directory / PROVENANCE_FILENAME
    external_copy = tmp_path / "provenance-copy.csv"
    external_copy.write_bytes(provenance.read_bytes())
    provenance.unlink()
    provenance.symlink_to(external_copy)
    with pytest.raises(FeatureValidationError, match="regular non-symlink"):
        validate_brca_q25_feature_artifacts(
            result.directory,
            expected_manifest_sha256=result.manifest_sha256,
        )

    repository = tmp_path / "temporary-repository"
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(FeatureValidationError, match="outside Git"):
        _publish(repository / "generated-features")
    assert not (repository / "generated-features").exists()


def test_symlink_parent_is_rejected_without_following_it(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    os.symlink(real_parent, alias)
    with pytest.raises(FeatureValidationError, match="symlink component"):
        _publish(alias / "q25")
    assert not (real_parent / "q25").exists()
