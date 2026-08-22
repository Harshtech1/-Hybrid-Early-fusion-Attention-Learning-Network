from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
from types import MappingProxyType

import numpy as np
import pytest
import torch

import multiscale_feature_pilot.src.brca_q50_feature_artifacts as artifacts
from multiscale_feature_pilot.src.brca_q50_feature_artifacts import (
    EXPECTED_CHECKPOINT_FILENAME,
    EXPECTED_CHECKPOINT_SHA256,
    EXPECTED_CHECKPOINT_SIZE_BYTES,
    EXPECTED_COORDINATE_MANIFEST_FILENAME,
    EXPECTED_COORDINATE_MANIFEST_SHA256,
    EXPECTED_COORDINATE_MANIFEST_SIZE_BYTES,
    EXPECTED_ENCODER_IDENTITY,
    EXPECTED_GDC_FILE_UUID,
    EXPECTED_OMIC_ARCHIVE_FILENAME,
    EXPECTED_OMIC_ARCHIVE_SHA256,
    EXPECTED_OMIC_ARCHIVE_SIZE_BYTES,
    EXPECTED_OMIC_DIMS,
    EXPECTED_OMIC_MEMBER_FILENAME,
    EXPECTED_OMIC_MEMBER_SHA256,
    EXPECTED_OMIC_MEMBER_SIZE_BYTES,
    EXPECTED_OMIC_SOURCE_ROW_INDEX,
    EXPECTED_PATIENT_ID,
    EXPECTED_SLIDE_ID,
    EXPECTED_WSI_FILENAME,
    EXPECTED_WSI_MD5,
    EXPECTED_WSI_SHA256,
    EXPECTED_WSI_SIZE_BYTES,
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
    publish_brca_q50_feature_artifacts,
    validate_brca_q50_feature_artifacts,
)
from multiscale_feature_pilot.src.provenance import PatchProvenance


SYNTHETIC_2X = [(0, 0), (512, 0), (0, 512), (512, 512)]
SYNTHETIC_4X = [(0, 0), (1024, 0)]


def _coordinate_hash(coordinates: list[tuple[int, int]]) -> str:
    array = np.asarray(coordinates, dtype=np.dtype("<i8"))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


@pytest.fixture(autouse=True)
def _small_synthetic_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """Run persistence paths without allocating the 88-MB Q50 tensor set."""

    monkeypatch.setattr(artifacts, "SCALE_2X_ROWS", 4)
    monkeypatch.setattr(artifacts, "SCALE_4X_ROWS", 2)
    monkeypatch.setattr(artifacts, "FEATURE_DIM", 3)
    monkeypatch.setattr(
        artifacts,
        "EXPECTED_COORDINATE_BRANCHES",
        MappingProxyType(
            {
                "scale_2x": MappingProxyType(
                    {
                        "artifact_filename": "scale_2x_coordinates.h5",
                        "artifact_size_bytes": 146_952,
                        "artifact_sha256": "4" * 64,
                        "coordinates_sha256": _coordinate_hash(SYNTHETIC_2X),
                        "coordinate_count": 4,
                        "source_level": 0,
                        "effective_mpp_x": 0.4936,
                        "effective_mpp_y": 0.4936,
                    }
                ),
                "scale_4x": MappingProxyType(
                    {
                        "artifact_filename": "scale_4x_coordinates.h5",
                        "artifact_size_bytes": 45_080,
                        "artifact_sha256": "5" * 64,
                        "coordinates_sha256": _coordinate_hash(SYNTHETIC_4X),
                        "coordinate_count": 2,
                        "source_level": 1,
                        "effective_mpp_x": 0.9872151105124595,
                        "effective_mpp_y": 0.9872151105124595,
                    }
                ),
            }
        ),
    )


def _provenance() -> tuple[PatchProvenance, ...]:
    result: list[PatchProvenance] = []
    for local_index, (x, y) in enumerate(SYNTHETIC_2X):
        result.append(
            PatchProvenance(
                global_row_index=local_index,
                branch="scale_2x",
                local_patch_index=local_index,
                x=x,
                y=y,
                level=0,
                mpp_x=0.4936,
                mpp_y=0.4936,
            )
        )
    for local_index, (x, y) in enumerate(SYNTHETIC_4X):
        result.append(
            PatchProvenance(
                global_row_index=4 + local_index,
                branch="scale_4x",
                local_patch_index=local_index,
                x=x,
                y=y,
                level=1,
                mpp_x=0.9872151105124595,
                mpp_y=0.9872151105124595,
            )
        )
    return tuple(result)


def _metadata() -> FeatureArtifactMetadata:
    expected_2x = artifacts.EXPECTED_COORDINATE_BRANCHES["scale_2x"]
    expected_4x = artifacts.EXPECTED_COORDINATE_BRANCHES["scale_4x"]

    def binding(branch: str, expected: MappingProxyType) -> CoordinateFeatureBinding:
        return CoordinateFeatureBinding(branch=branch, **dict(expected))

    return FeatureArtifactMetadata(
        patient_id=EXPECTED_PATIENT_ID,
        slide_id=EXPECTED_SLIDE_ID,
        gdc_file_uuid=EXPECTED_GDC_FILE_UUID,
        wsi_filename=EXPECTED_WSI_FILENAME,
        wsi_size_bytes=EXPECTED_WSI_SIZE_BYTES,
        wsi_md5=EXPECTED_WSI_MD5,
        wsi_sha256=EXPECTED_WSI_SHA256,
        coordinate_manifest_filename=EXPECTED_COORDINATE_MANIFEST_FILENAME,
        coordinate_manifest_size_bytes=EXPECTED_COORDINATE_MANIFEST_SIZE_BYTES,
        coordinate_manifest_sha256=EXPECTED_COORDINATE_MANIFEST_SHA256,
        scale_2x_coordinates=binding("scale_2x", expected_2x),
        scale_4x_coordinates=binding("scale_4x", expected_4x),
        omic_archive_filename=EXPECTED_OMIC_ARCHIVE_FILENAME,
        omic_archive_size_bytes=EXPECTED_OMIC_ARCHIVE_SIZE_BYTES,
        omic_archive_sha256=EXPECTED_OMIC_ARCHIVE_SHA256,
        omic_member_filename=EXPECTED_OMIC_MEMBER_FILENAME,
        omic_member_size_bytes=EXPECTED_OMIC_MEMBER_SIZE_BYTES,
        omic_member_sha256=EXPECTED_OMIC_MEMBER_SHA256,
        omic_source_row_index=EXPECTED_OMIC_SOURCE_ROW_INDEX,
        rna_feature_count=EXPECTED_OMIC_DIMS["rna"],
        mutation_feature_count=EXPECTED_OMIC_DIMS["mutation"],
        cnv_feature_count=EXPECTED_OMIC_DIMS["cnv"],
        encoder_identity=EXPECTED_ENCODER_IDENTITY,
        checkpoint_filename=EXPECTED_CHECKPOINT_FILENAME,
        checkpoint_size_bytes=EXPECTED_CHECKPOINT_SIZE_BYTES,
        checkpoint_sha256=EXPECTED_CHECKPOINT_SHA256,
        source_policy_name="brca_q50_gpu_execution_authorization.yaml",
        source_policy_sha256="6" * 64,
        implementation_git_commit="7" * 40,
        implementation_source_sha256={
            "multiscale_feature_pilot/src/brca_q50_feature_artifacts.py": "8" * 64,
            "scripts/run_brca_q50_gpu_pilot.py": "9" * 64,
        },
    )


def _features() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale_2x = torch.arange(12, dtype=torch.float32).reshape(4, 3).contiguous()
    scale_4x = (torch.arange(6, dtype=torch.float32) + 100).reshape(2, 3).contiguous()
    return scale_2x, scale_4x, torch.cat((scale_2x, scale_4x), dim=0)


def _destination(parent: Path) -> Path:
    return parent / "Q50.features"


def _publish(parent: Path):
    scale_2x, scale_4x, combined = _features()
    return publish_brca_q50_feature_artifacts(
        _destination(parent),
        scale_2x_features=scale_2x,
        scale_4x_features=scale_4x,
        combined_features=combined,
        row_provenance=_provenance(),
        metadata=_metadata(),
    )


def test_public_constants_lock_real_q50_contract() -> None:
    source = Path(artifacts.__file__).read_text(encoding="utf-8")
    assert "SCALE_2X_ROWS: Final = 8_580" in source
    assert "SCALE_4X_ROWS: Final = 2_213" in source
    assert "FEATURE_DIM: Final = 2_048" in source
    assert "EXPECTED_OUTPUT_BASENAME: Final = \"Q50.features\"" in source
    assert "EXPECTED_OMIC_SOURCE_ROW_INDEX: Final = \"370\"" in source
    assert "EXPECTED_COORDINATE_MANIFEST_SIZE_BYTES: Final = 6_558" in source


def test_publishes_exact_set_validates_and_is_byte_deterministic(tmp_path: Path) -> None:
    first = _publish(tmp_path / "first")
    second = _publish(tmp_path / "second")

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
        assert (
            first.feature_for(name).tensor_content_sha256
            == second.feature_for(name).tensor_content_sha256
        )
    assert first.provenance.sha256 == second.provenance.sha256
    assert first.manifest_sha256_path.read_text(encoding="ascii") == (
        f"{first.manifest_sha256}  {MANIFEST_FILENAME}\n"
    )

    validated = validate_brca_q50_feature_artifacts(
        first.directory,
        expected_manifest_sha256=first.manifest_sha256,
    )
    assert validated.feature_for("scale_2x_features").shape == (4, 3)
    assert validated.feature_for("scale_4x_features").shape == (2, 3)
    assert validated.feature_for("combined_features").shape == (6, 3)
    assert validated.provenance.row_count == 6
    assert validated.metadata.patient_id == EXPECTED_PATIENT_ID
    manifest = json.loads(validated.manifest_path.read_text(encoding="utf-8"))
    assert manifest["contract"]["branch_order"] == ["scale_2x", "scale_4x"]
    assert manifest["contract"]["concatenation"] == "torch.cat([scale_2x,scale_4x],dim=0)"
    assert manifest["metadata"]["omic"] == {
        "archive_filename": EXPECTED_OMIC_ARCHIVE_FILENAME,
        "archive_sha256": EXPECTED_OMIC_ARCHIVE_SHA256,
        "archive_size_bytes": EXPECTED_OMIC_ARCHIVE_SIZE_BYTES,
        "feature_counts": {"cnv": 1333, "mutation": 21, "rna": 1558},
        "member_filename": EXPECTED_OMIC_MEMBER_FILENAME,
        "member_sha256": EXPECTED_OMIC_MEMBER_SHA256,
        "member_size_bytes": EXPECTED_OMIC_MEMBER_SIZE_BYTES,
        "source_row_index": "370",
    }
    assert not list((tmp_path / "first").glob(".Q50.features.staging.*"))
    assert not (tmp_path / "first" / ".Q50.features.lock").exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_shape", "shape must be"),
        ("wrong_dtype", "dtype must be torch.float32"),
        ("nonfinite", "contains NaN or Inf"),
        ("noncontiguous", "must be contiguous"),
        ("noncompact", "compact tensor storage"),
        ("requires_grad", "must not require gradients"),
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
    elif mutation == "noncompact":
        scale_2x = torch.zeros((5, 3), dtype=torch.float32)[:4]
    elif mutation == "requires_grad":
        scale_2x.requires_grad_(True)
    else:
        combined = combined.clone()
        combined[0, 0] = -1
    destination = _destination(tmp_path)
    with pytest.raises(FeatureValidationError, match=message):
        publish_brca_q50_feature_artifacts(
            destination,
            scale_2x_features=scale_2x,
            scale_4x_features=scale_4x,
            combined_features=combined,
            row_provenance=_provenance(),
            metadata=_metadata(),
        )
    assert not destination.exists()
    assert not list(tmp_path.glob(".Q50.features.staging.*"))
    assert not (tmp_path / ".Q50.features.lock").exists()


def test_metadata_is_locked_to_q50_wsi_coordinate_omic_and_checkpoint() -> None:
    metadata = _metadata()
    with pytest.raises(FeatureValidationError, match="omic_source_row_index drift"):
        replace(metadata, omic_source_row_index="371")
    with pytest.raises(FeatureValidationError, match="wsi_sha256 drift"):
        replace(metadata, wsi_sha256="0" * 64)
    with pytest.raises(FeatureValidationError, match="checkpoint_sha256 drift"):
        replace(metadata, checkpoint_sha256="0" * 64)
    with pytest.raises(FeatureValidationError, match="coordinate identity drift"):
        replace(
            metadata.scale_2x_coordinates,
            artifact_sha256="0" * 64,
        )


def test_provenance_binds_coordinate_hash_order_level_and_mpp(tmp_path: Path) -> None:
    scale_2x, scale_4x, combined = _features()
    rows = list(_provenance())
    rows[4] = replace(rows[4], branch="scale_2x")
    with pytest.raises(FeatureValidationError, match="branch order"):
        publish_brca_q50_feature_artifacts(
            _destination(tmp_path),
            scale_2x_features=scale_2x,
            scale_4x_features=scale_4x,
            combined_features=combined,
            row_provenance=rows,
            metadata=_metadata(),
        )

    rows = list(_provenance())
    rows[0] = replace(rows[0], x=1)
    with pytest.raises(FeatureHashMismatchError, match="coordinate-content SHA-256"):
        publish_brca_q50_feature_artifacts(
            _destination(tmp_path),
            scale_2x_features=scale_2x,
            scale_4x_features=scale_4x,
            combined_features=combined,
            row_provenance=rows,
            metadata=_metadata(),
        )


def test_no_clobber_stale_staging_and_lock_are_fail_closed(tmp_path: Path) -> None:
    parent = tmp_path / "existing"
    result = _publish(parent)
    manifest_before = result.manifest_path.read_bytes()
    with pytest.raises(FeatureArtifactExistsError, match="already exists"):
        _publish(parent)
    assert result.manifest_path.read_bytes() == manifest_before

    stale_parent = tmp_path / "stale"
    stale_parent.mkdir()
    (stale_parent / ".Q50.features.staging.abandoned").mkdir()
    with pytest.raises(FeaturePublicationInProgressError, match="staging"):
        _publish(stale_parent)

    locked_parent = tmp_path / "locked"
    locked_parent.mkdir()
    lock = locked_parent / ".Q50.features.lock"
    lock.write_text("pid=123\n", encoding="ascii")
    with pytest.raises(FeaturePublicationInProgressError, match="locked"):
        _publish(locked_parent)
    assert lock.read_text(encoding="ascii") == "pid=123\n"


def test_existing_destination_symlink_and_special_file_are_never_replaced(
    tmp_path: Path,
) -> None:
    scale_2x, scale_4x, combined = _features()

    def attempt(destination: Path) -> None:
        with pytest.raises(FeatureArtifactExistsError, match="already exists"):
            publish_brca_q50_feature_artifacts(
                destination,
                scale_2x_features=scale_2x,
                scale_4x_features=scale_4x,
                combined_features=combined,
                row_provenance=_provenance(),
                metadata=_metadata(),
            )

    symlink_parent = tmp_path / "destination-symlink"
    symlink_parent.mkdir()
    symlink_destination = _destination(symlink_parent)
    symlink_target = tmp_path / "must-remain"
    symlink_target.write_text("untouched\n", encoding="utf-8")
    symlink_destination.symlink_to(symlink_target)
    attempt(symlink_destination)
    assert symlink_destination.is_symlink()
    assert symlink_target.read_text(encoding="utf-8") == "untouched\n"

    fifo_parent = tmp_path / "destination-fifo"
    fifo_parent.mkdir()
    fifo_destination = _destination(fifo_parent)
    os.mkfifo(fifo_destination)
    attempt(fifo_destination)
    assert fifo_destination.exists()


def test_external_anchor_extras_and_tampering_are_rejected(tmp_path: Path) -> None:
    result = _publish(tmp_path)
    with pytest.raises(FeatureHashMismatchError, match="manifest SHA-256"):
        validate_brca_q50_feature_artifacts(
            result.directory,
            expected_manifest_sha256="0" * 64,
        )

    extra = result.directory / "resume.partial"
    extra.write_text("not allowed", encoding="utf-8")
    with pytest.raises(FeatureValidationError, match="file set is not exact"):
        validate_brca_q50_feature_artifacts(
            result.directory,
            expected_manifest_sha256=result.manifest_sha256,
        )
    extra.unlink()

    tensor_path = result.directory / FEATURE_FILENAMES["scale_2x_features"]
    with tensor_path.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(FeatureHashMismatchError, match="file SHA-256"):
        validate_brca_q50_feature_artifacts(
            result.directory,
            expected_manifest_sha256=result.manifest_sha256,
        )


def test_symlink_and_special_artifacts_are_rejected(tmp_path: Path) -> None:
    symlink_parent = tmp_path / "symlink-case"
    result = _publish(symlink_parent)
    provenance = result.directory / PROVENANCE_FILENAME
    external_copy = tmp_path / "provenance-copy.csv"
    external_copy.write_bytes(provenance.read_bytes())
    provenance.unlink()
    provenance.symlink_to(external_copy)
    with pytest.raises(FeatureValidationError, match="regular non-symlink"):
        validate_brca_q50_feature_artifacts(
            result.directory,
            expected_manifest_sha256=result.manifest_sha256,
        )

    special_parent = tmp_path / "special-case"
    result = _publish(special_parent)
    provenance = result.directory / PROVENANCE_FILENAME
    provenance.unlink()
    os.mkfifo(provenance)
    with pytest.raises(FeatureValidationError, match="regular non-symlink"):
        validate_brca_q50_feature_artifacts(
            result.directory,
            expected_manifest_sha256=result.manifest_sha256,
        )


def test_symlink_parent_wrong_basename_and_inside_git_are_rejected(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias = tmp_path / "alias"
    os.symlink(real_parent, alias)
    with pytest.raises(FeatureValidationError, match="symlink component"):
        _publish(alias)
    assert not _destination(real_parent).exists()

    scale_2x, scale_4x, combined = _features()
    with pytest.raises(FeatureValidationError, match="destination basename"):
        publish_brca_q50_feature_artifacts(
            tmp_path / "wrong-name",
            scale_2x_features=scale_2x,
            scale_4x_features=scale_4x,
            combined_features=combined,
            row_provenance=_provenance(),
            metadata=_metadata(),
        )

    repository = tmp_path / "temporary-repository"
    subprocess.run(
        ["git", "init", "--quiet", str(repository)],
        check=True,
        capture_output=True,
        text=True,
    )
    with pytest.raises(FeatureValidationError, match="outside Git"):
        _publish(repository)
    assert not _destination(repository).exists()
