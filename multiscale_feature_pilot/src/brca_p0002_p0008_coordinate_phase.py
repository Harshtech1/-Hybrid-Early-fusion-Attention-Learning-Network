"""Fail-closed CPU coordinate phase for production rows P0002--P0008."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import time
from typing import Any
import uuid

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import h5py
import numpy as np
import openslide
import yaml

from .brca_coordinate_artifacts import (
    BRANCH_FILENAMES,
    MANIFEST_FILENAME,
    MANIFEST_SHA256_FILENAME,
    SCHEMA,
    CoordinateBranchMetadata,
    sha256_file,
    validate_brca_coordinate_artifacts,
)
from .brca_omic import BRCA_RELEASE_ARCHIVE_SHA256, load_brca_patient_omics
from .brca_q25_coordinates import (
    generate_level_0_lattice_coordinates,
    segment_tissue_contours,
)


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
DATA = WORKSPACE / "brca_pilot_data"
OMIC = (
    WORKSPACE
    / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
AUTH = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_authorization.yaml"
REQUEST = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_request.yaml"
APPROVAL = ROOT / "multiscale_feature_pilot/provenance/brca_p0002_p0008_coordinate_execution_approval.yaml"
LABELS = tuple(f"P{i:04d}" for i in range(2, 9))
REQUEST_SHA256 = "47805f82a8a727f6dc646860a109b8cd5d8c19a3213a8c18fba5c31fcd96ca6e"
STATEMENT_SHA256 = "057e9c42d68c200768696643edf53d0efeb41a4904b5661dfe2e61063366e01a"
MAXIMUM_WORKERS = 2
EXPECTED_SOURCE_HASHES = {
    Path("multiscale_feature_pilot/src/brca_q25_coordinates.py"): "da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82",
    Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"): "a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf",
    Path("multiscale_feature_pilot/src/brca_omic.py"): "5061039913cb0dd4e8e30958c8b4b76f088396f7841255dd94234c3b462d5a5d",
}
ALLOWED_STATUS = {
    " M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}


class CoordinatePhaseError(RuntimeError):
    """Raised on any authorization, input, scope, or publication drift."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise CoordinatePhaseError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_fd(descriptor: int) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 -- exact GDC identity contract
    sha256 = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 8 * 1024 * 1024):
        md5.update(chunk)
        sha256.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return md5.hexdigest(), sha256.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def _bound_paths() -> tuple[Path, ...]:
    paths = [
        Path("scripts/run_brca_p0002_p0008_coordinate_phase.py"),
        Path("multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_authorization.yaml"),
        Path("multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_request.yaml"),
        Path("multiscale_feature_pilot/provenance/brca_p0002_p0008_coordinate_execution_approval.yaml"),
        Path("multiscale_feature_pilot/src/brca_p0002_p0008_coordinate_phase.py"),
        Path("multiscale_feature_pilot/tests/test_brca_p0002_p0008_coordinate_phase.py"),
        Path("multiscale_feature_pilot/__init__.py"),
        Path("multiscale_feature_pilot/src/__init__.py"),
        Path("multiscale_feature_pilot/src/brca_q25_coordinates.py"),
        Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
        Path("multiscale_feature_pilot/src/brca_omic.py"),
        Path("multiscale_feature_pilot/src/omic.py"),
        Path("multiscale_feature_pilot/src/padding.py"),
        Path("multiscale_feature_pilot/src/multiscale_bag.py"),
        Path("multiscale_feature_pilot/src/provenance.py"),
    ]
    for label in LABELS:
        lower = label.lower()
        paths.extend(
            (
                Path(f"multiscale_feature_pilot/config/brca_{lower}_scale_coordinate_policy.yaml"),
                Path(f"multiscale_feature_pilot/provenance/brca_{lower}_scale_coordinate_policy_review.yaml"),
                Path(f"multiscale_feature_pilot/provenance/brca_{lower}_header_metadata_result/result.yaml"),
            )
        )
    return tuple(paths)


def validate_repository(expected_commit: str) -> None:
    require(len(expected_commit) == 40, "full expected source commit required")
    require(_git("rev-parse", "HEAD") == expected_commit, "source commit drift")
    raw_status = subprocess.run(
        ("git", "status", "--short"), cwd=ROOT, check=True, text=True, capture_output=True
    ).stdout
    statuses = set(filter(None, raw_status.splitlines()))
    require(statuses <= ALLOWED_STATUS, f"unexpected Git status: {statuses - ALLOWED_STATUS}")
    for relative in _bound_paths():
        path = ROOT / relative
        require(path.is_file() and not path.is_symlink(), f"unsafe bound file: {relative}")
        committed = subprocess.run(
            ("git", "show", f"HEAD:{relative.as_posix()}"),
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        require(path.read_bytes() == committed, f"bound source drift: {relative}")
    for relative, expected_hash in EXPECTED_SOURCE_HASHES.items():
        require(sha256_path(ROOT / relative) == expected_hash, f"shared source hash drift: {relative}")


@dataclass(frozen=True)
class PatientSpec:
    label: str
    patient_id: str
    slide_id: str
    gdc_uuid: str
    omic_row: str
    wsi: Path
    incoming: Path
    parcel: Path
    size: int
    md5: str
    sha256: str
    parcel_sha256: str
    dimensions: tuple[tuple[int, int], ...]
    downsamples: tuple[float, ...]
    mpp: tuple[float, float]
    mask_size: tuple[int, int]
    coordinate_geometry_scale_xy: tuple[float, float]
    policy_sha256: str
    effective_mpp_2x: tuple[float, float]
    effective_mpp_4x: tuple[float, float]
    geometry_compatibility_4x: str
    omic_hashes: dict[str, str]
    maximum_coordinates_2x: int
    maximum_coordinates_4x: int

    @property
    def destination(self) -> Path:
        return DATA / f"BRCA_PRODUCTION_{self.label}.coordinates"


def _load_documents() -> tuple[dict[str, Any], dict[str, Any]]:
    require(sha256_path(REQUEST) == REQUEST_SHA256, "coordinate request hash drift")
    request = yaml.safe_load(REQUEST.read_text(encoding="utf-8"))
    authorization = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    require(request["status"] == "REQUEST_PREPARED_EXECUTION_NOT_AUTHORIZED", "request status drift")
    require(request["executable"] is False, "request must remain an immutable proposal")
    require(authorization["status"] == "AUTHORIZED_P0002_P0008_EXACT_CPU_COORDINATE_PHASE", "authorization status drift")
    require(authorization["executable"] is True, "authorization is not executable")
    statement = authorization["approval"]["exact_statement"]
    require(hashlib.sha256(statement.encode()).hexdigest() == STATEMENT_SHA256, "authorization statement drift")
    require(authorization["approval"]["exact_statement_sha256"] == STATEMENT_SHA256, "statement hash drift")
    require(authorization["request"]["sha256"] == REQUEST_SHA256, "authorization request binding drift")
    execution = authorization["authorized_execution"]
    require(execution["exact_patient_labels"] == list(LABELS), "patient authorization drift")
    require(execution["maximum_cpu_patient_workers"] == MAXIMUM_WORKERS, "worker limit drift")
    require(execution["exact_total_read_region_calls"] == len(LABELS), "read count drift")
    require(not any(authorization["authority"].values()), "prohibited authorization enabled")
    require(not any(request["authority"].values()), "request execution lock drift")
    return request, authorization


def load_specs() -> tuple[PatientSpec, ...]:
    request, _authorization = _load_documents()
    specs: list[PatientSpec] = []
    for label in LABELS:
        lower = label.lower()
        policy_path = ROOT / f"multiscale_feature_pilot/config/brca_{lower}_scale_coordinate_policy.yaml"
        review_path = ROOT / f"multiscale_feature_pilot/provenance/brca_{lower}_scale_coordinate_policy_review.yaml"
        header_path = ROOT / f"multiscale_feature_pilot/provenance/brca_{lower}_header_metadata_result/result.yaml"
        policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
        review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
        header = yaml.safe_load(header_path.read_text(encoding="utf-8"))
        binding = request["patients"][label]
        policy_hash = sha256_path(policy_path)
        require(policy_hash == binding["policy_sha256"], f"{label} policy hash drift")
        require(sha256_path(header_path) == binding["header_result_sha256"], f"{label} header hash drift")
        require(review["outputs"]["policy_sha256"] == policy_hash, f"{label} review binding drift")
        require(policy["execution_boundary"]["status"] == "EXECUTION_LOCKED", f"{label} policy status drift")
        require(not any(value for key, value in policy["execution_boundary"].items() if key != "status"), f"{label} policy execution surface unlocked")
        require(header["operations"]["read_region_or_pixel_calls"] == 0, f"{label} prior pixel count drift")
        require(all(value == 0 for value in header["operations"].values()), f"{label} prior operation drift")
        identity = header["identity"]
        pinned = policy["pinned_header"]
        levels = header["header"]["levels"]
        dimensions = tuple(tuple(int(value) for value in item["dimensions"]) for item in levels)
        downsamples = tuple(float(item["downsample"]) for item in levels)
        require(dimensions == tuple(tuple(item) for item in pinned["level_dimensions"]), f"{label} dimension binding drift")
        require(downsamples == tuple(pinned["openslide_scalar_level_downsamples"]), f"{label} downsample binding drift")
        read = binding["read_region"]
        require(read["level_0_location"] == [0, 0] and read["level"] == 2, f"{label} read tuple drift")
        require(tuple(read["size_at_level"]) == dimensions[2], f"{label} mask size drift")
        require(policy["identity"]["sha256"] == identity["sha256"], f"{label} identity drift")
        incoming = DATA / f"BRCA_PRODUCTION_{label}.incoming"
        wsi = Path(identity["path"])
        parcel = incoming / identity["gdc_uuid"] / "logs" / f"{identity['slide_id']}.parcel"
        specs.append(
            PatientSpec(
                label=label,
                patient_id=identity["patient_id"],
                slide_id=identity["slide_id"],
                gdc_uuid=identity["gdc_uuid"],
                omic_row=str(header["omic"]["source_row_index"]),
                wsi=wsi,
                incoming=incoming,
                parcel=parcel,
                size=int(identity["size_bytes"]),
                md5=identity["md5"],
                sha256=identity["sha256"],
                parcel_sha256=header["download_tree"]["parcel_sha256"],
                dimensions=dimensions,
                downsamples=downsamples,
                mpp=(float(header["header"]["mpp_x"]), float(header["header"]["mpp_y"])),
                mask_size=tuple(read["size_at_level"]),
                coordinate_geometry_scale_xy=tuple(policy["future_mask_policy"]["coordinate_geometry_scale_xy"]),
                policy_sha256=policy_hash,
                effective_mpp_2x=tuple(policy["scale_policy"]["scale_2x"]["effective_mpp"]),
                effective_mpp_4x=tuple(policy["scale_policy"]["scale_4x"]["effective_mpp"]),
                geometry_compatibility_4x=policy["branches"]["scale_4x"]["geometry_compatibility"],
                omic_hashes=dict(header["omic"]["content_sha256"]),
                maximum_coordinates_2x=int(policy["branches"]["scale_2x"]["theoretical_sites_before_tissue_filter"]),
                maximum_coordinates_4x=int(policy["branches"]["scale_4x"]["theoretical_sites_before_tissue_filter"]),
            )
        )
    return tuple(specs)


def _validate_no_symlinks(root: Path) -> set[str]:
    require(root.is_dir() and not root.is_symlink(), f"unsafe incoming directory: {root}")
    files: set[str] = set()
    for path in root.rglob("*"):
        details = path.lstat()
        require(not stat.S_ISLNK(details.st_mode), f"symlink input forbidden: {path}")
        if stat.S_ISREG(details.st_mode):
            files.add(path.relative_to(root).as_posix())
        else:
            require(stat.S_ISDIR(details.st_mode), f"unsafe input type: {path}")
    return files


def _validate_omic(spec: PatientSpec) -> None:
    omics = load_brca_patient_omics(
        OMIC,
        case_id=spec.patient_id,
        slide_id=spec.slide_id,
        expected_archive_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
    )
    require(str(omics.source_row_index) == spec.omic_row, f"{spec.label} Omic row drift")
    for name in ("rna", "mutation", "cnv"):
        tensor = getattr(omics, name)
        require(
            tensor.device.type == "cpu"
            and str(tensor.dtype) == "torch.float32"
            and tensor.is_contiguous()
            and bool(tensor.isfinite().all()),
            f"{spec.label} {name} tensor drift",
        )
        digest = hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()
        require(digest == spec.omic_hashes[name], f"{spec.label} {name} hash drift")


def validate_patient_inputs(spec: PatientSpec, *, require_destination_absent: bool = True) -> None:
    expected_files = {
        f"{spec.gdc_uuid}/{spec.slide_id}",
        f"{spec.gdc_uuid}/logs/{spec.slide_id}.parcel",
    }
    require(_validate_no_symlinks(spec.incoming) == expected_files, f"{spec.label} incoming tree drift")
    wsi_stat = spec.wsi.lstat()
    parcel_stat = spec.parcel.lstat()
    require(stat.S_ISREG(wsi_stat.st_mode) and not stat.S_ISLNK(wsi_stat.st_mode), f"{spec.label} unsafe WSI")
    require(stat.S_ISREG(parcel_stat.st_mode) and not stat.S_ISLNK(parcel_stat.st_mode), f"{spec.label} unsafe parcel")
    require(wsi_stat.st_size == spec.size, f"{spec.label} WSI size drift")
    require(sha256_path(spec.parcel) == spec.parcel_sha256, f"{spec.label} parcel hash drift")
    _validate_omic(spec)
    if require_destination_absent:
        require(not os.path.lexists(spec.destination), f"{spec.label} coordinate destination exists")
        staging = tuple(DATA.glob(f".{spec.destination.name}.staging.*"))
        require(not staging, f"{spec.label} coordinate staging collision")


def _require_held_wsi(
    spec: PatientSpec,
    descriptor: int,
    token: tuple[int, int, int],
    *,
    rehash: bool,
) -> None:
    held = os.fstat(descriptor)
    current = os.stat(spec.wsi, follow_symlinks=False)
    require(stat.S_ISREG(held.st_mode), f"{spec.label} held WSI type drift")
    require(token == (held.st_dev, held.st_ino, held.st_size), f"{spec.label} held WSI identity drift")
    require(token == (current.st_dev, current.st_ino, current.st_size), f"{spec.label} WSI pathname identity drift")
    if rehash:
        require(_hash_fd(descriptor) == (spec.md5, spec.sha256), f"{spec.label} held WSI content drift")


def read_exact_mask(
    spec: PatientSpec,
) -> tuple[np.ndarray, str, int, tuple[int, int, int]]:
    descriptor = os.open(spec.wsi, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    calls = 0
    completed = False
    try:
        before = os.fstat(descriptor)
        token = (before.st_dev, before.st_ino, before.st_size)
        require(stat.S_ISREG(before.st_mode) and before.st_size == spec.size, f"{spec.label} held WSI drift")
        require(_hash_fd(descriptor) == (spec.md5, spec.sha256), f"{spec.label} held WSI hash drift")
        slide = openslide.OpenSlide(f"/proc/self/fd/{descriptor}")
        try:
            dimensions = tuple(tuple(map(int, item)) for item in slide.level_dimensions)
            downsamples = tuple(float(value) for value in slide.level_downsamples)
            mpp = (
                float(slide.properties[openslide.PROPERTY_NAME_MPP_X]),
                float(slide.properties[openslide.PROPERTY_NAME_MPP_Y]),
            )
            require(dimensions == spec.dimensions, f"{spec.label} header dimensions drift")
            require(downsamples == spec.downsamples and mpp == spec.mpp, f"{spec.label} header scalar drift")
            image = slide.read_region((0, 0), 2, spec.mask_size)
            calls += 1
            mask = np.ascontiguousarray(np.asarray(image, dtype=np.uint8))
            require(mask.shape == (spec.mask_size[1], spec.mask_size[0], 4), f"{spec.label} mask shape drift")
        finally:
            slide.close()
        require(calls == 1, f"{spec.label} mask read count drift")
        _require_held_wsi(spec, descriptor, token, rehash=True)
        completed = True
    finally:
        if not completed:
            os.close(descriptor)
    return mask, hashlib.sha256(mask.tobytes(order="C")).hexdigest(), descriptor, token


def _branch_metadata(
    spec: PatientSpec,
    branch: str,
    mask_sha256: str,
    contour_count: int,
    hole_count: int,
) -> CoordinateBranchMetadata:
    scale_2x = branch == "scale_2x"
    source_level = 0 if scale_2x else 1
    return CoordinateBranchMetadata(
        branch=branch,
        patient_id=spec.patient_id,
        slide_id=spec.slide_id,
        gdc_file_uuid=spec.gdc_uuid,
        wsi_filename=spec.slide_id,
        wsi_size_bytes=spec.size,
        wsi_md5=spec.md5,
        wsi_sha256=spec.sha256,
        level_0_dimensions=spec.dimensions[0],
        source_level=source_level,
        source_level_dimensions=spec.dimensions[source_level],
        openslide_reported_source_downsample=spec.downsamples[source_level],
        source_patch_size=(512, 512) if scale_2x else (256, 256),
        output_patch_size=(256, 256),
        level_0_declared_footprint=(512, 512) if scale_2x else (1024, 1024),
        level_0_step=(512, 512) if scale_2x else (1024, 1024),
        target_mpp=0.5 if scale_2x else 1.0,
        effective_mpp=spec.effective_mpp_2x if scale_2x else spec.effective_mpp_4x,
        interpolation="PIL.Image.Resampling.LANCZOS" if scale_2x else "none",
        resampling="linear_factor_2" if scale_2x else "none",
        mask_level=2,
        mask_level_dimensions=spec.mask_size,
        openslide_reported_mask_downsample=spec.downsamples[2],
        mask_image_channels=4,
        mask_image_sha256=mask_sha256,
        mask_parameters={"sthresh": 8, "mthresh": 7, "close": 4, "use_otsu": False, "a_t": 100, "a_h": 16, "max_n_holes": 8, "reference_patch_size": 512},
        contour_count=contour_count,
        retained_hole_count=hole_count,
        clam_commit="26e0b6c4873e112f1ccd74cd834894c4ab7a2934",
        policy_sha256=spec.policy_sha256,
        geometry_compatibility="GLOBAL_LEVEL0_NATIVE" if scale_2x else spec.geometry_compatibility_4x,
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_data_parent() -> tuple[int, tuple[int, int, int]]:
    details = DATA.lstat()
    require(stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode), "unsafe coordinate data parent")
    descriptor = os.open(DATA, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW)
    opened = os.fstat(descriptor)
    token = (opened.st_dev, opened.st_ino, opened.st_mode)
    require(token == (details.st_dev, details.st_ino, details.st_mode), "coordinate data parent identity drift")
    return descriptor, token


def _require_data_parent(descriptor: int, token: tuple[int, int, int]) -> None:
    held = os.fstat(descriptor)
    current = DATA.lstat()
    require(stat.S_ISDIR(held.st_mode) and not stat.S_ISLNK(current.st_mode), "unsafe coordinate data parent")
    require(token == (held.st_dev, held.st_ino, held.st_mode), "held coordinate parent drift")
    require(token == (current.st_dev, current.st_ino, current.st_mode), "coordinate parent pathname drift")


def publish(
    spec: PatientSpec,
    coordinates_2x: np.ndarray,
    coordinates_4x: np.ndarray,
    metadata_2x: CoordinateBranchMetadata,
    metadata_4x: CoordinateBranchMetadata,
    held_wsi_descriptor: int,
    held_wsi_token: tuple[int, int, int],
    data_descriptor: int,
    data_token: tuple[int, int, int],
):
    _require_held_wsi(spec, held_wsi_descriptor, held_wsi_token, rehash=True)
    _require_data_parent(data_descriptor, data_token)
    staging_name = f".{spec.destination.name}.staging.{uuid.uuid4().hex}"
    stable_parent = Path(f"/proc/self/fd/{data_descriptor}")
    staging = stable_parent / staging_name
    final = stable_parent / spec.destination.name
    require(not os.path.lexists(staging) and not os.path.lexists(final), f"{spec.label} publication collision")
    os.mkdir(staging_name, 0o700, dir_fd=data_descriptor)
    branches: dict[str, Any] = {}
    for branch, coordinates, metadata in (
        ("scale_2x", coordinates_2x, metadata_2x),
        ("scale_4x", coordinates_4x, metadata_4x),
    ):
        path = staging / BRANCH_FILENAMES[branch]
        with h5py.File(path, "x") as output:
            dataset = output.create_dataset("coords", data=np.ascontiguousarray(coordinates, dtype=np.int64))
            for key, value in metadata.to_attributes().items():
                dataset.attrs[key] = value
            output.flush()
        _fsync_file(path)
        branches[branch] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "coordinates_sha256": hashlib.sha256(np.ascontiguousarray(coordinates, dtype="<i8").tobytes()).hexdigest(),
            "coordinate_count": int(coordinates.shape[0]),
            "attributes": metadata.to_attributes(),
        }
    payload = (json.dumps({"schema": SCHEMA, "branches": branches}, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()
    manifest = staging / MANIFEST_FILENAME
    sidecar = staging / MANIFEST_SHA256_FILENAME
    manifest.write_bytes(payload)
    anchor = hashlib.sha256(payload).hexdigest()
    sidecar.write_text(f"{anchor}  {MANIFEST_FILENAME}\n", encoding="ascii", newline="")
    _fsync_file(manifest)
    _fsync_file(sidecar)
    _fsync_directory(staging)
    validate_brca_coordinate_artifacts(staging, expected_manifest_sha256=anchor)
    _require_held_wsi(spec, held_wsi_descriptor, held_wsi_token, rehash=False)
    _require_data_parent(data_descriptor, data_token)
    renameat2 = getattr(ctypes.CDLL(None, use_errno=True), "renameat2", None)
    require(renameat2 is not None, "atomic RENAME_NOREPLACE unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(data_descriptor, os.fsencode(staging_name), data_descriptor, os.fsencode(spec.destination.name), 1)
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise CoordinatePhaseError(f"{spec.label} destination appeared before publication")
        raise OSError(error_number, os.strerror(error_number))
    os.fsync(data_descriptor)
    _require_held_wsi(spec, held_wsi_descriptor, held_wsi_token, rehash=False)
    _require_data_parent(data_descriptor, data_token)
    return validate_brca_coordinate_artifacts(final, expected_manifest_sha256=anchor)


def execute_patient(
    spec: PatientSpec,
    expected_commit: str,
    data_descriptor: int,
    data_token: tuple[int, int, int],
) -> dict[str, Any]:
    started_at = datetime.now(timezone.utc)
    started = time.perf_counter()
    validate_repository(expected_commit)
    validate_patient_inputs(spec)
    mask, mask_sha256, held_wsi_descriptor, held_wsi_token = read_exact_mask(spec)
    try:
        geometry = segment_tissue_contours(
            mask,
            level_0_dimensions=spec.dimensions[0],
            mask_dimensions=spec.mask_size,
        )
        coordinates_2x = generate_level_0_lattice_coordinates(
            level_0_dimensions=spec.dimensions[0],
            level_0_patch_size=512,
            level_0_step=512,
            geometry=geometry,
        )
        coordinates_4x = generate_level_0_lattice_coordinates(
            level_0_dimensions=spec.dimensions[0],
            level_0_patch_size=1024,
            level_0_step=1024,
            geometry=geometry,
        )
        del mask
        require(coordinates_2x.shape[0] > 0 and coordinates_4x.shape[0] > 0, f"{spec.label} empty coordinate bag")
        require(coordinates_2x.shape[0] <= spec.maximum_coordinates_2x, f"{spec.label} 2x capacity exceeded")
        require(coordinates_4x.shape[0] <= spec.maximum_coordinates_4x, f"{spec.label} 4x capacity exceeded")
        contour_count = len(geometry.contours)
        hole_count = sum(len(items) for items in geometry.holes)
        validate_repository(expected_commit)
        validate_patient_inputs(spec)
        record = publish(
            spec,
            coordinates_2x,
            coordinates_4x,
            _branch_metadata(spec, "scale_2x", mask_sha256, contour_count, hole_count),
            _branch_metadata(spec, "scale_4x", mask_sha256, contour_count, hole_count),
            held_wsi_descriptor,
            held_wsi_token,
            data_descriptor,
            data_token,
        )
    finally:
        os.close(held_wsi_descriptor)
    return {
        "status": f"BRCA_PRODUCTION_{spec.label}_COORDINATES_VERIFIED",
        "patient_label": spec.label,
        "started_at_utc": started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": time.perf_counter() - started,
        "source_commit": expected_commit,
        "destination": str(spec.destination),
        "manifest_sha256": record.manifest_sha256,
        "scale_2x_count": int(coordinates_2x.shape[0]),
        "scale_4x_count": int(coordinates_4x.shape[0]),
        "contour_count": contour_count,
        "retained_hole_count": hole_count,
        "openslide_open_count": 1,
        "header_read_count": 1,
        "read_region_calls": 1,
        "read_region": {"level_0_location": [0, 0], "level": 2, "size_at_level": list(spec.mask_size)},
        "mask_sha256": mask_sha256,
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


def run(expected_commit: str) -> dict[str, Any]:
    phase_started_at = datetime.now(timezone.utc)
    phase_started = time.perf_counter()
    lock_descriptor = os.open(AUTH, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    data_descriptor = -1
    try:
        try:
            fcntl.flock(lock_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise CoordinatePhaseError("another coordinate phase invocation is active") from error
        validate_repository(expected_commit)
        data_descriptor, data_token = _open_data_parent()
        specs = load_specs()
        require(len(specs) == len(LABELS), "seven patient specifications required")
        for spec in specs:
            validate_patient_inputs(spec)

        results: dict[str, dict[str, Any]] = {}
        pending_specs = iter(specs)
        active: dict[Future[dict[str, Any]], PatientSpec] = {}
        with ThreadPoolExecutor(max_workers=MAXIMUM_WORKERS, thread_name_prefix="brca-coordinate") as pool:
            for _ in range(MAXIMUM_WORKERS):
                spec = next(pending_specs, None)
                if spec is not None:
                    active[pool.submit(execute_patient, spec, expected_commit, data_descriptor, data_token)] = spec
            while active:
                completed, _ = wait(active, return_when=FIRST_COMPLETED)
                completed_results: list[tuple[PatientSpec, dict[str, Any]]] = []
                for future in completed:
                    spec = active.pop(future)
                    # Resolve the entire completed set before admitting any
                    # later patient. If one failed, no replacement is started.
                    result = future.result()
                    completed_results.append((spec, result))
                for spec, result in completed_results:
                    results[spec.label] = result
                    next_spec = next(pending_specs, None)
                    if next_spec is not None:
                        active[pool.submit(execute_patient, next_spec, expected_commit, data_descriptor, data_token)] = next_spec
    finally:
        if data_descriptor >= 0:
            os.close(data_descriptor)
        os.close(lock_descriptor)
    require(set(results) == set(LABELS), "all seven coordinate results required")
    ordered = [results[label] for label in LABELS]
    require(sum(result["read_region_calls"] for result in ordered) == 7, "total mask read count drift")
    return {
        "status": "BRCA_PRODUCTION_P0002_P0008_COORDINATES_VERIFIED",
        "started_at_utc": phase_started_at.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_wall_seconds": time.perf_counter() - phase_started,
        "source_commit": expected_commit,
        "maximum_cpu_patient_workers": MAXIMUM_WORKERS,
        "exact_total_read_region_calls": 7,
        "patients": ordered,
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
        "required_stop_reached": True,
    }


__all__ = [
    "AUTH",
    "APPROVAL",
    "CoordinatePhaseError",
    "LABELS",
    "MAXIMUM_WORKERS",
    "PatientSpec",
    "REQUEST",
    "execute_patient",
    "load_specs",
    "publish",
    "read_exact_mask",
    "run",
    "validate_patient_inputs",
    "validate_repository",
]
