from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import subprocess

import pytest
import torch
import yaml

from scripts import run_brca_b06_header_gate as gate


class HeaderOnlySlide:
    def __init__(self) -> None:
        self.properties = {
            "openslide.mpp-x": "0.2468",
            "openslide.mpp-y": "0.2468",
        }
        self.level_count = 3
        self.level_dimensions = ((4096, 2048), (1024, 512), (256, 128))
        self.level_downsamples = (1.0, 4.0, 16.0)
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_authorization_and_manifest_bind_exact_user_boundary() -> None:
    authorization = gate.validate_authorization()
    assert authorization["statement_sha256"] == gate.STATEMENT_SHA256
    assert gate.validate_manifest() == gate.MANIFEST_SHA256
    payload = yaml.safe_load(gate.AUTH.read_text(encoding="utf-8"))
    assert payload["authority"]["read_region_or_pixel_access"] is False
    assert payload["execution"]["active_gdc_clients"] == 1
    assert payload["execution"]["conservative_required_available_bytes"] == 23_382_989_506


def test_source_has_no_pixel_thumbnail_or_associated_image_call() -> None:
    tree = ast.parse(Path(gate.__file__).read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "read_region" not in called_attributes
    assert "get_thumbnail" not in called_attributes
    assert "associated_images" not in called_attributes


def test_collect_header_uses_held_descriptor_without_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = tmp_path / "slide.svs"
    content = b"header-only synthetic fixture"
    payload.write_bytes(content)
    monkeypatch.setattr(gate, "WSI", payload)
    monkeypatch.setattr(gate, "SIZE", len(content))
    monkeypatch.setattr(gate, "MD5", hashlib.md5(content).hexdigest())  # noqa: S324
    slides: list[HeaderOnlySlide] = []
    opened: list[str] = []

    def factory(path: str) -> HeaderOnlySlide:
        assert path.startswith("/proc/self/fd/")
        opened.append(path)
        slide = HeaderOnlySlide()
        slides.append(slide)
        return slide

    header, digest = gate.collect_header(factory)
    assert digest == hashlib.sha256(content).hexdigest()
    assert opened and slides[0].closed
    assert header["level_count"] == 3
    assert header["read_region_calls"] == 0
    assert header["levels"][1]["dimensions"] == [1024, 512]


def test_collect_header_rejects_path_swap_during_final_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = tmp_path / "slide.svs"
    content = b"same-size-old"
    payload.write_bytes(content)
    replacement = tmp_path / "replacement.svs"
    replacement.write_bytes(b"same-size-new")
    monkeypatch.setattr(gate, "WSI", payload)
    monkeypatch.setattr(gate, "SIZE", len(content))
    monkeypatch.setattr(gate, "MD5", hashlib.md5(content).hexdigest())  # noqa: S324
    original_hash = gate._hash_fd
    calls = 0

    def swapping_hash(descriptor: int) -> tuple[str, str]:
        nonlocal calls
        calls += 1
        value = original_hash(descriptor)
        if calls == 2:
            os.replace(replacement, payload)
        return value

    monkeypatch.setattr(gate, "_hash_fd", swapping_hash)
    with pytest.raises(gate.B06HeaderGateError, match="during final hash"):
        gate.collect_header(lambda _path: HeaderOnlySlide())


def test_exact_official_omic_row_and_hashes() -> None:
    result = gate.load_omic()
    assert result["source_row_index"] == "897"
    assert result["exact_case_and_full_slide_match"] is True
    for name, digest in gate.EXPECTED_OMIC_HASHES.items():
        assert result[name]["content_sha256"] == digest
        assert result[name]["dtype"] == "float32"


def test_preflight_fails_closed_if_destination_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    incoming = tmp_path / "B06.incoming"
    incoming.mkdir()
    monkeypatch.setattr(gate, "INCOMING", incoming)
    with pytest.raises(gate.B06HeaderGateError, match="must be absent"):
        gate.preflight()


def test_download_failure_reports_sanitized_capped_client_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failed(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="prefix" + "x" * 2100 + "\x00PUBLIC_STDOUT",
            stderr="PUBLIC_STDERR\x01",
        )

    monkeypatch.setattr(subprocess, "run", failed)
    with pytest.raises(gate.B06HeaderGateError) as captured:
        gate.download()
    message = str(captured.value)
    assert "PUBLIC_STDOUT" in message and "PUBLIC_STDERR?" in message
    assert "prefix" not in message
    assert "\\x00" not in message and "\\x01" not in message


def test_atomic_result_set_refuses_overwrite(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    bundle = tmp_path / "bundle"
    monkeypatch.setattr(gate, "RESULT_BUNDLE", bundle)
    gate.publish_result_set(b"first yaml\n", b"first report\n")
    with pytest.raises(gate.B06HeaderGateError, match="append-only"):
        gate.publish_result_set(b"second yaml\n", b"second report\n")
    assert (bundle / "result.yaml").read_bytes() == b"first yaml\n"
    assert (bundle / "report.md").read_bytes() == b"first report\n"


def test_orchestrator_publishes_only_after_all_checks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_path = tmp_path / "result.yaml"
    report_path = tmp_path / "report.md"
    bundle_path = tmp_path / "bundle"
    lock_path = tmp_path / "lock"
    incoming = tmp_path / "incoming"
    monkeypatch.setattr(gate, "RESULT", result_path)
    monkeypatch.setattr(gate, "REPORT", report_path)
    monkeypatch.setattr(gate, "RESULT_BUNDLE", bundle_path)
    monkeypatch.setattr(gate, "LOCK", lock_path)
    monkeypatch.setattr(gate, "INCOMING", incoming)
    monkeypatch.setattr(gate, "validate_source", lambda commit: {"source_commit": commit, "critical_file_sha256": {}})
    monkeypatch.setattr(gate, "validate_authorization", lambda: {"path": "auth.yaml", "sha256": "a" * 64, "statement_sha256": "b" * 64})
    monkeypatch.setattr(gate, "validate_manifest", lambda: "c" * 64)
    monkeypatch.setattr(gate, "preflight", lambda: {
        "filesystem_before_download": {"total_bytes": 10, "used_bytes": 1, "available_bytes": 9},
        "gdc_client_version": "2.3",
    })
    monkeypatch.setattr(gate, "download", lambda: {
        "seconds": 1.0, "returncode": 0,
        "argv": ["gdc-client", "download"], "timeout_seconds": 10_800,
    })
    tree = {"entries": ["x"], "parcel_size_bytes": 1, "parcel_sha256": "d" * 64, "total_regular_file_bytes": 2}
    monkeypatch.setattr(gate, "validate_tree", lambda: tree)
    omic = {
        "source_row_index": "897", "exact_case_and_full_slide_match": True,
        "archive_sha256": gate.BRCA_RELEASE_ARCHIVE_SHA256,
        "rna": {}, "mutation": {}, "cnv": {},
    }
    monkeypatch.setattr(gate, "load_omic", lambda: omic)
    header = {
        "inspection": "OPENSLIDE_HEADER_ONLY_NO_PIXEL_ACCESS", "mpp_x": 0.25,
        "mpp_y": 0.25, "level_count": 1,
        "levels": [{"level": 0, "dimensions": [10, 10], "downsample": 1.0}],
        "read_region_calls": 0, "thumbnail_calls": 0, "associated_image_accesses": 0,
    }
    monkeypatch.setattr(gate, "collect_header", lambda factory: (header, "e" * 64))
    monkeypatch.setattr(gate, "_disk", lambda path: {"total_bytes": 10, "used_bytes": 2, "available_bytes": 8})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)

    record = gate.run(expected_source_commit="f" * 40)
    assert record["status"] == gate.RESULT_STATUS
    assert not lock_path.exists()
    assert yaml.safe_load((bundle_path / "result.yaml").read_text(encoding="utf-8"))["required_stop_reached"] is True
    assert "No pixel API was called" in (bundle_path / "report.md").read_text(encoding="utf-8")
