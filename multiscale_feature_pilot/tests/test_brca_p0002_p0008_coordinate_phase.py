from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from multiscale_feature_pilot.src import brca_p0002_p0008_coordinate_phase as phase


ROOT = Path(__file__).resolve().parents[2]
AUTH = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_authorization.yaml"
SOURCE = Path(phase.__file__)


def test_exact_authorization_and_request_binding() -> None:
    authorization = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    statement = authorization["approval"]["exact_statement"]
    assert authorization["status"] == "AUTHORIZED_P0002_P0008_EXACT_CPU_COORDINATE_PHASE"
    assert authorization["executable"] is True
    assert hashlib.sha256(statement.encode()).hexdigest() == phase.STATEMENT_SHA256
    assert authorization["request"]["sha256"] == phase.REQUEST_SHA256
    assert authorization["authorized_execution"]["exact_patient_labels"] == list(phase.LABELS)
    assert authorization["authorized_execution"]["maximum_cpu_patient_workers"] == 2
    assert authorization["authorized_execution"]["exact_total_read_region_calls"] == 7
    assert not any(authorization["authority"].values())


def test_all_seven_specs_bind_exact_tuples_and_frozen_inputs() -> None:
    specs = phase.load_specs()
    assert [spec.label for spec in specs] == [f"P{i:04d}" for i in range(2, 9)]
    assert [spec.mask_size for spec in specs] == [
        (4606, 4846),
        (3859, 3740),
        (5602, 4979),
        (6474, 5594),
        (7019, 5184),
        (5879, 5017),
        (7679, 4583),
    ]
    assert all(spec.dimensions[2] == spec.mask_size for spec in specs)
    assert all(spec.maximum_coordinates_2x > 0 and spec.maximum_coordinates_4x > 0 for spec in specs)


def test_source_has_one_pixel_call_site_and_no_forbidden_surface() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    read_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "read_region"
    ]
    assert len(read_calls) == 1
    source = SOURCE.read_text(encoding="utf-8")
    for token in (
        "get_thumbnail",
        "associated_images",
        "ResNet50",
        "HEALNet",
        "torch.cuda",
        ".unlink(",
        "rmtree",
        "os.remove",
    ):
        assert token not in source
    assert "ThreadPoolExecutor(max_workers=MAXIMUM_WORKERS" in source
    assert "RENAME_NOREPLACE" in source


def _synthetic_spec(tmp_path: Path) -> phase.PatientSpec:
    wsi = tmp_path / "slide.svs"
    wsi.write_bytes(b"abc")
    parcel = tmp_path / "slide.parcel"
    parcel.write_bytes(b"parcel")
    return phase.PatientSpec(
        label="P0002",
        patient_id="TCGA-X",
        slide_id="slide.svs",
        gdc_uuid="uuid",
        omic_row="1",
        wsi=wsi,
        incoming=tmp_path,
        parcel=parcel,
        size=3,
        md5="md5",
        sha256="sha",
        parcel_sha256="parcel-sha",
        dimensions=((64, 48), (16, 12), (4, 3), (2, 1)),
        downsamples=(1.0, 4.0, 16.0, 32.0),
        mpp=(0.25, 0.25),
        mask_size=(4, 3),
        coordinate_geometry_scale_xy=(16.0, 16.0),
        policy_sha256="policy",
        effective_mpp_2x=(0.5, 0.5),
        effective_mpp_4x=(1.0, 1.0),
        geometry_compatibility_4x="EXACT_INTEGER_LEVEL_1_TO_LEVEL_0_MAPPING",
        omic_hashes={"rna": "", "mutation": "", "cnv": ""},
        maximum_coordinates_2x=1,
        maximum_coordinates_4x=1,
    )


def test_runtime_mask_reader_opens_and_reads_exactly_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _synthetic_spec(tmp_path)
    counters = {"opens": 0, "reads": 0, "closes": 0}

    class Slide:
        level_dimensions = spec.dimensions
        level_downsamples = spec.downsamples
        properties = {
            "openslide.mpp-x": "0.25",
            "openslide.mpp-y": "0.25",
        }

        def read_region(self, location, level, size):
            counters["reads"] += 1
            assert location == (0, 0) and level == 2 and size == (4, 3)
            return np.zeros((3, 4, 4), dtype=np.uint8)

        def close(self):
            counters["closes"] += 1

    def factory(path: str):
        counters["opens"] += 1
        assert path.startswith("/proc/self/fd/")
        return Slide()

    monkeypatch.setattr(phase.openslide, "OpenSlide", factory)
    monkeypatch.setattr(phase, "_hash_fd", lambda _descriptor: (spec.md5, spec.sha256))
    mask, digest, descriptor, _token = phase.read_exact_mask(spec)
    try:
        assert mask.shape == (3, 4, 4)
        assert digest == hashlib.sha256(mask.tobytes(order="C")).hexdigest()
        assert counters == {"opens": 1, "reads": 1, "closes": 1}
    finally:
        phase.os.close(descriptor)


def _result(label: str) -> dict:
    return {
        "patient_label": label,
        "read_region_calls": 1,
        "operations": {
            "patch_reads": 0,
            "feature_extractions": 0,
            "resnet50": 0,
            "healnet": 0,
            "gpu_or_cuda": 0,
            "deletions": 0,
            "drive": 0,
            "training": 0,
        },
    }


def test_scheduler_never_exceeds_two_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = tuple(SimpleNamespace(label=label) for label in phase.LABELS)
    active = peak = 0
    lock = threading.Lock()

    monkeypatch.setattr(phase, "validate_repository", lambda _commit: None)
    monkeypatch.setattr(phase, "load_specs", lambda: specs)
    monkeypatch.setattr(phase, "validate_patient_inputs", lambda _spec: None)

    def execute(spec, _commit, _data_descriptor, _data_token):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.005)
        with lock:
            active -= 1
        return _result(spec.label)

    monkeypatch.setattr(phase, "execute_patient", execute)
    result = phase.run("f" * 40)
    assert [item["patient_label"] for item in result["patients"]] == list(phase.LABELS)
    assert peak == 2
    assert result["exact_total_read_region_calls"] == 7


def test_scheduler_starts_no_later_patient_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    specs = tuple(SimpleNamespace(label=label) for label in phase.LABELS)
    started: list[str] = []
    monkeypatch.setattr(phase, "validate_repository", lambda _commit: None)
    monkeypatch.setattr(phase, "load_specs", lambda: specs)
    monkeypatch.setattr(phase, "validate_patient_inputs", lambda _spec: None)

    def execute(spec, _commit, _data_descriptor, _data_token):
        started.append(spec.label)
        if spec.label == "P0002":
            raise phase.CoordinatePhaseError("synthetic failure")
        time.sleep(0.02)
        return _result(spec.label)

    monkeypatch.setattr(phase, "execute_patient", execute)
    with pytest.raises(phase.CoordinatePhaseError, match="synthetic failure"):
        phase.run("f" * 40)
    assert set(started) <= {"P0002", "P0003"}


def test_atomic_publisher_validates_and_never_overwrites(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = phase.load_specs()[0]
    monkeypatch.setattr(phase, "DATA", tmp_path)
    data_descriptor, data_token = phase._open_data_parent()
    coordinates = np.asarray([[0, 0]], dtype=np.int64)
    metadata_2x = phase._branch_metadata(spec, "scale_2x", "a" * 64, 1, 0)
    metadata_4x = phase._branch_metadata(spec, "scale_4x", "a" * 64, 1, 0)
    monkeypatch.setattr(phase, "_require_held_wsi", lambda *_args, **_kwargs: None)
    try:
        validation = phase.publish(
            spec, coordinates, coordinates, metadata_2x, metadata_4x,
            -1, (0, 0, 0), data_descriptor, data_token,
        )
        assert {
            branch.branch: branch.coordinate_count for branch in validation.branches
        } == {"scale_2x": 1, "scale_4x": 1}
        assert {path.name for path in spec.destination.iterdir()} == {
            "coordinate_manifest.json",
            "coordinate_manifest.json.sha256",
            "scale_2x_coordinates.h5",
            "scale_4x_coordinates.h5",
        }
        with pytest.raises(phase.CoordinatePhaseError, match="publication collision"):
            phase.publish(
                spec, coordinates, coordinates, metadata_2x, metadata_4x,
                -1, (0, 0, 0), data_descriptor, data_token,
            )
        assert not tuple(tmp_path.glob(f".{spec.destination.name}.staging.*"))
    finally:
        phase.os.close(data_descriptor)


def test_same_size_wsi_path_swap_blocks_before_publication(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    spec = _synthetic_spec(tmp_path)
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setattr(phase, "DATA", data)

    class Slide:
        level_dimensions = spec.dimensions
        level_downsamples = spec.downsamples
        properties = {"openslide.mpp-x": "0.25", "openslide.mpp-y": "0.25"}

        def read_region(self, *_args):
            return np.zeros((3, 4, 4), dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(phase.openslide, "OpenSlide", lambda _path: Slide())
    monkeypatch.setattr(phase, "_hash_fd", lambda _descriptor: (spec.md5, spec.sha256))
    _mask, _digest, wsi_descriptor, wsi_token = phase.read_exact_mask(spec)
    data_descriptor, data_token = phase._open_data_parent()
    replacement = tmp_path / "replacement.svs"
    replacement.write_bytes(b"xyz")
    replacement.replace(spec.wsi)
    coordinates = np.asarray([[0, 0]], dtype=np.int64)
    try:
        with pytest.raises(phase.CoordinatePhaseError, match="pathname identity drift"):
            phase.publish(
                spec,
                coordinates,
                coordinates,
                None,  # type: ignore[arg-type] -- rejected before metadata use
                None,  # type: ignore[arg-type] -- rejected before metadata use
                wsi_descriptor,
                wsi_token,
                data_descriptor,
                data_token,
            )
        assert not os.path.lexists(data / "BRCA_PRODUCTION_P0002.coordinates")
    finally:
        phase.os.close(wsi_descriptor)
        phase.os.close(data_descriptor)


def test_symlinked_data_parent_is_rejected_before_staging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "data"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(phase, "DATA", link)
    with pytest.raises(phase.CoordinatePhaseError, match="unsafe coordinate data parent"):
        phase._open_data_parent()
    assert not tuple(target.iterdir())


def test_porcelain_parser_preserves_leading_status_space() -> None:
    source = SOURCE.read_text(encoding="utf-8")
    assert "raw_status.splitlines()" in source
    assert 'raw_status = subprocess.run(' in source
    assert '.stdout.strip().splitlines()' not in source
