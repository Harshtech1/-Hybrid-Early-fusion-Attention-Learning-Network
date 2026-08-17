from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from multiscale_feature_pilot.src.artifacts import (
    ArtifactExistsError,
    ArtifactHashMismatchError,
    ArtifactWriteInProgressError,
    atomic_save_tensor,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    sha256_file,
)


def test_sha256_file_streams_exact_bytes(tmp_path: Path) -> None:
    artifact = tmp_path / "payload.bin"
    payload = b"abc" * (3 * 1024 * 1024)
    artifact.write_bytes(payload)

    assert sha256_file(artifact) == hashlib.sha256(payload).hexdigest()


def test_tensor_is_saved_atomically_as_tensor_only_cpu_payload(tmp_path: Path) -> None:
    artifact = tmp_path / "features.pt"
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).transpose(0, 1)
    assert not source.is_contiguous()

    record = atomic_save_tensor(source, artifact)
    loaded = torch.load(artifact, map_location="cpu", weights_only=True)

    assert record.path == artifact
    assert record.size_bytes == artifact.stat().st_size
    assert record.sha256 == sha256_file(artifact)
    assert not record.reused
    assert isinstance(loaded, torch.Tensor)
    assert loaded.device.type == "cpu"
    assert loaded.is_contiguous()
    assert torch.equal(loaded, source.contiguous())
    assert not list(tmp_path.glob("*.partial"))
    assert not list(tmp_path.glob("*.lock"))


def test_existing_destination_requires_explicit_matching_hash(tmp_path: Path) -> None:
    artifact = tmp_path / "features.pt"
    first = atomic_save_tensor(torch.zeros((2, 3), dtype=torch.float32), artifact)
    original_bytes = artifact.read_bytes()

    with pytest.raises(ArtifactExistsError, match="expected_sha256 is required"):
        atomic_save_tensor(torch.ones((2, 3), dtype=torch.float32), artifact)
    with pytest.raises(ArtifactHashMismatchError, match="existing artifact"):
        atomic_save_tensor(
            torch.ones((2, 3), dtype=torch.float32),
            artifact,
            expected_sha256="0" * 64,
        )

    reused = atomic_save_tensor(
        torch.ones((2, 3), dtype=torch.float32),
        artifact,
        expected_sha256=first.sha256,
    )
    assert reused.reused
    assert reused.sha256 == first.sha256
    assert artifact.read_bytes() == original_bytes


def test_new_artifact_hash_mismatch_is_never_published(tmp_path: Path) -> None:
    artifact = tmp_path / "features.pt"

    with pytest.raises(ArtifactHashMismatchError, match="new artifact"):
        atomic_save_tensor(
            torch.ones((2, 3), dtype=torch.float32),
            artifact,
            expected_sha256="0" * 64,
        )

    assert not artifact.exists()
    assert not any(tmp_path.iterdir())


def test_lock_prevents_a_second_cooperative_writer(tmp_path: Path) -> None:
    artifact = tmp_path / "features.pt"
    lock = tmp_path / ".features.pt.lock"
    lock.write_text("pid=123\n", encoding="ascii")

    with pytest.raises(ArtifactWriteInProgressError, match="locked"):
        atomic_save_tensor(torch.ones((1,), dtype=torch.float32), artifact)

    assert lock.exists()
    assert not artifact.exists()


def test_text_json_and_csv_are_deterministic_and_reusable(tmp_path: Path) -> None:
    text_record = atomic_write_text("alpha\nbeta\n", tmp_path / "note.txt")
    assert (tmp_path / "note.txt").read_bytes() == b"alpha\nbeta\n"
    assert atomic_write_text(
        "ignored replacement",
        tmp_path / "note.txt",
        expected_sha256=text_record.sha256,
    ).reused

    json_record = atomic_write_json({"z": 1, "a": [2, 3]}, tmp_path / "manifest.json")
    assert json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8")) == {
        "a": [2, 3],
        "z": 1,
    }
    assert (tmp_path / "manifest.json").read_text(encoding="utf-8").startswith('{\n  "a"')
    assert json_record.sha256 == sha256_file(tmp_path / "manifest.json")

    csv_record = atomic_write_csv(
        [{"row": 0, "branch": "scale_2x"}, {"row": 1, "branch": "scale_4x"}],
        tmp_path / "rows.csv",
        fieldnames=("row", "branch"),
    )
    assert (tmp_path / "rows.csv").read_bytes() == (
        b"row,branch\n0,scale_2x\n1,scale_4x\n"
    )
    assert csv_record.sha256 == sha256_file(tmp_path / "rows.csv")


@pytest.mark.parametrize("bad_hash", ["", "abc", "g" * 64, "0" * 63])
def test_expected_sha256_requires_a_full_hex_digest(tmp_path: Path, bad_hash: str) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        atomic_write_text("payload", tmp_path / "note.txt", expected_sha256=bad_hash)
