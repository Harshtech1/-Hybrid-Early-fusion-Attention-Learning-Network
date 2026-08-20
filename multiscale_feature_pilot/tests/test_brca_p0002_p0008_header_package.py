from __future__ import annotations

import ast
import hashlib
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest
import torch

from multiscale_feature_pilot.src import brca_p0002_p0008_header_package as package


class HeaderOnlySlide:
    properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}
    level_count = 3
    level_dimensions = ((4096, 2048), (1024, 512), (256, 128))
    level_downsamples = (1.0, 4.0, 16.0)

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    def read_region(self, *_args: object) -> None:
        raise AssertionError("pixel access")


def test_exact_authorization_and_manifests() -> None:
    result = package.validate_package()
    assert result == {
        "authorization_sha256": package.AUTH_SHA256,
        "patient_count": 7,
        "total_bytes": 6_527_281_524,
    }
    assert [patient.label for patient in package.PATIENTS] == [f"P{i:04d}" for i in range(2, 9)]
    assert sum(patient.size for patient in package.PATIENTS) == package.TOTAL_BYTES


def test_implementation_has_no_pixel_api_call() -> None:
    tree = ast.parse(Path(package.__file__).read_text(encoding="utf-8"))
    calls = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert calls.isdisjoint({"read_region", "get_thumbnail", "associated_images", "get_associated_image"})


def test_header_proxy_runtime_blocks_every_pixel_surface() -> None:
    proxy = package._HeaderOnlyProxy(HeaderOnlySlide())
    for name in ("read_region", "get_thumbnail", "associated_images", "get_associated_image"):
        with pytest.raises(package.HeaderPackageError, match="pixel API prohibited"):
            getattr(proxy, name)


def test_collect_header_uses_held_descriptor_and_rehashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content = b"synthetic header-only svs"
    path = tmp_path / "slide.svs"
    path.write_bytes(content)
    source = package.PATIENTS[0]
    patient = package.Patient(
        source.label, source.cohort_index, source.patient_id, source.slide,
        source.uuid, source.omic_row, hashlib.md5(content).hexdigest(), len(content),
        source.manifest_sha256, source.omic_hashes,
    )
    monkeypatch.setattr(package.Patient, "wsi", property(lambda _self: path))
    opened: list[str] = []
    slides: list[HeaderOnlySlide] = []

    def factory(stable_path: str) -> HeaderOnlySlide:
        assert stable_path.startswith("/proc/self/fd/")
        opened.append(stable_path)
        slide = HeaderOnlySlide()
        slides.append(slide)
        return slide

    header, digest = package.collect_header(patient, factory)
    assert digest == hashlib.sha256(content).hexdigest()
    assert header["read_region_calls"] == 0
    assert header["levels"][2]["dimensions"] == [256, 128]
    assert opened and slides[0].closed


def test_phase_barrier_never_exceeds_one_download_and_two_patients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active_downloads = 0
    active_headers = 0
    peak_downloads = 0
    peak_patients = 0
    lock = threading.Lock()

    monkeypatch.setattr(package, "validate_source", lambda _commit: "f" * 40)
    monkeypatch.setattr(package, "validate_package", lambda: {})
    monkeypatch.setattr(package, "_active_gdc_clients", lambda: [])
    monkeypatch.setattr(package, "preflight_patient", lambda _patient: None)

    monkeypatch.setattr(package, "preflight_block", lambda: {})
    monkeypatch.setattr(package, "_create_incoming", lambda _patient: (1, 2, 0o40700))

    def download(patient: package.Patient, _identity: tuple[int, int, int]) -> dict[str, object]:
        nonlocal active_downloads, peak_downloads, peak_patients
        with lock:
            active_downloads += 1
            peak_downloads = max(peak_downloads, active_downloads)
            peak_patients = max(peak_patients, active_downloads + active_headers)
        time.sleep(0.005)
        with lock:
            active_downloads -= 1
        return {"label": patient.label}

    def inspect(patient: package.Patient, _transfer: dict[str, object], _identity: tuple[int, int, int]) -> dict[str, object]:
        nonlocal active_headers, peak_patients
        with lock:
            active_headers += 1
            peak_patients = max(peak_patients, active_downloads + active_headers)
        time.sleep(0.01)
        with lock:
            active_headers -= 1
        return {"patient_label": patient.label}

    monkeypatch.setattr(package, "download_patient", download)
    monkeypatch.setattr(package, "inspect_and_publish", inspect)
    results = package.run(expected_source_commit="f" * 40)
    assert [result["patient_label"] for result in results] == [f"P{i:04d}" for i in range(2, 9)]
    assert peak_downloads == 1
    assert peak_patients <= 2


def test_preflight_refuses_existing_raw_or_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = package.PATIENTS[0]
    monkeypatch.setattr(package.Patient, "incoming", property(lambda _self: tmp_path / "incoming"))
    monkeypatch.setattr(package.Patient, "result_bundle", property(lambda _self: tmp_path / "result"))
    source.incoming.mkdir()
    with pytest.raises(package.HeaderPackageError, match="incoming exists"):
        package.preflight_patient(source)


def test_process_audit_detects_executable_comm_and_argv_basename(tmp_path: Path) -> None:
    executable = tmp_path / "gdc-client"
    executable.write_bytes(b"binary")
    for pid, argv0, comm, link in (
        ("101", "/renamed/client", "gdc-client", executable),
        ("102", "/opt/gdc-client", "python", executable),
        ("103", "/opt/gdc-client", "python", tmp_path / "different"),
    ):
        entry = tmp_path / pid
        entry.mkdir()
        (entry / "cmdline").write_bytes(argv0.encode() + b"\0download\0")
        (entry / "comm").write_text(comm + "\n", encoding="utf-8")
        if not link.exists():
            link.write_bytes(b"binary")
        (entry / "exe").symlink_to(link)
    assert package._active_gdc_clients(tmp_path) == [101, 102, 103]


def test_exclusive_incoming_identity_rejects_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = package.PATIENTS[0]
    incoming = tmp_path / "incoming"
    monkeypatch.setattr(package, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(package.Patient, "incoming", property(lambda _self: incoming))
    identity = package._create_incoming(source)
    incoming.rename(tmp_path / "original")
    incoming.mkdir(mode=0o700)
    with pytest.raises(package.HeaderPackageError, match="identity drift"):
        package._require_incoming(source, identity)


def test_tree_rejects_symlink_ancestor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = package.PATIENTS[0]
    incoming = tmp_path / "incoming"
    incoming.mkdir(mode=0o700)
    identity_details = incoming.lstat()
    identity = (identity_details.st_dev, identity_details.st_ino, identity_details.st_mode)
    outside = tmp_path / "outside"
    outside.mkdir()
    (incoming / source.uuid).symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(package.Patient, "incoming", property(lambda _self: incoming))
    with pytest.raises(package.HeaderPackageError, match="symlink found"):
        package.validate_tree(source, identity)


def test_download_failure_sanitizes_both_streams_and_preserves_incoming(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source = package.PATIENTS[0]
    incoming = tmp_path / "incoming"
    incoming.mkdir(mode=0o700)
    details = incoming.lstat()
    identity = (details.st_dev, details.st_ino, details.st_mode)
    client = tmp_path / "gdc-client"
    client.write_bytes(b"client")
    monkeypatch.setattr(package.Patient, "incoming", property(lambda _self: incoming))
    monkeypatch.setattr(package, "CLIENT", client)
    monkeypatch.setattr(package, "CLIENT_SHA256", hashlib.sha256(b"client").hexdigest())
    monkeypatch.setattr(package, "_active_gdc_clients", lambda: [])

    def failure(command: list[str], **_kwargs: object):
        import subprocess
        return subprocess.CompletedProcess(command, 7, "old" + "x" * 2100 + "\x00OUT", "ERR\x01")

    monkeypatch.setattr(package.subprocess, "run", failure)
    with pytest.raises(package.HeaderPackageError) as captured:
        package.download_patient(source, identity)
    message = str(captured.value)
    assert "OUT" in message and "ERR?" in message and "old" not in message
    package._require_incoming(source, identity)


def test_whole_block_storage_floor_is_checked_before_patient_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    usage = type("Usage", (), {"free": package.WHOLE_BLOCK_REQUIRED_AVAILABLE_BYTES - 1})()
    monkeypatch.setattr(package, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(package, "RESULT_PARENT", tmp_path)
    monkeypatch.setattr(package.shutil, "disk_usage", lambda _path: usage)
    with pytest.raises(package.HeaderPackageError, match="whole-block storage"):
        package.preflight_block()


def test_omic_held_descriptor_rejects_path_swap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    archive = tmp_path / "omic.zip"
    archive.write_bytes(b"original")
    replacement = tmp_path / "replacement.zip"
    replacement.write_bytes(b"replaced")
    tensors = (
        torch.zeros((1, 1, 2), dtype=torch.float32),
        torch.zeros((1, 1, 1), dtype=torch.float32),
        torch.ones((1, 1, 1), dtype=torch.float32),
    )
    hashes = tuple(package._tensor_hash(tensor) for tensor in tensors)
    source = package.PATIENTS[0]
    patient = package.Patient(
        source.label, source.cohort_index, source.patient_id, source.slide,
        source.uuid, source.omic_row, source.md5, source.size,
        source.manifest_sha256, hashes,
    )
    monkeypatch.setattr(package, "OMIC", archive)
    monkeypatch.setattr(package, "OMIC_SIZE", len(b"original"))
    monkeypatch.setattr(package, "BRCA_RELEASE_ARCHIVE_SHA256", hashlib.sha256(b"original").hexdigest())

    def loader(_path: Path, **_kwargs: str) -> SimpleNamespace:
        replacement.replace(archive)
        return SimpleNamespace(
            source_row_index=patient.omic_row,
            rna=tensors[0], mutation=tensors[1], cnv=tensors[2],
        )

    monkeypatch.setattr(package, "load_official_brca_patient_omics", loader)
    with pytest.raises(package.HeaderPackageError, match="descriptor drift|pathname changed"):
        package.load_omic(patient)
