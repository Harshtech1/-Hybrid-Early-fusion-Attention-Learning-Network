"""Fail-closed P0002--P0008 serial acquisition and header-only inspection."""

from __future__ import annotations

import csv
from concurrent.futures import Future, ThreadPoolExecutor
import ctypes
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import shutil
import stat
import subprocess
import time
from typing import Any, Callable

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import openslide
import torch
import yaml

from multiscale_feature_pilot.src.brca_omic import (
    BRCA_RELEASE_ARCHIVE_SHA256,
    load_official_brca_patient_omics,
)


class HeaderPackageError(RuntimeError):
    """Raised on any identity, scope, concurrency, or publication drift."""


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/teamspace/studios/this_studio/brca_pilot_data")
OMIC = Path(
    "/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/"
    "data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
CLIENT = Path("/teamspace/studios/this_studio/tools/gdc-client/2.3.0/gdc-client")
CLIENT_SHA256 = "1df9281cbacbb53413607a4b9b682947dcb48b6ac7fade6527748bc025ae8c96"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_header_authorization.yaml"
AUTH_SHA256 = "ec70da619537db9c395fa33ed16594a8e363a4981132154fa70d5100cb2732dd"
STATEMENT_SHA256 = "3ada92434dfc6f78bbf29e7d127fe2dd199e8fa11a7ea9356bbcea92e85f88fc"
MANIFEST_DIR = ROOT / "multiscale_feature_pilot/provenance/brca_p0002_p0008_authorized_manifests"
RESULT_PARENT = ROOT / "multiscale_feature_pilot/provenance"
FIRST_EIGHT = ROOT / "reports/brca_first_eight_canary_proposal.tsv"
FIRST_EIGHT_SHA256 = "940b8fd1f7d194c2c9b7c69ddae58ffff3c55196b6841ac859c60cbc01095dfd"
TOTAL_BYTES = 6_527_281_524
WHOLE_BLOCK_REQUIRED_AVAILABLE_BYTES = 2 * TOTAL_BYTES + 20_000_000_000
OMIC_SIZE = 4_081_277
DOWNLOAD_TIMEOUT_SECONDS = 10_800


@dataclass(frozen=True)
class Patient:
    label: str
    cohort_index: int
    patient_id: str
    slide: str
    uuid: str
    omic_row: str
    md5: str
    size: int
    manifest_sha256: str
    omic_hashes: tuple[str, str, str]

    @property
    def manifest(self) -> Path:
        return MANIFEST_DIR / f"{self.label}_{self.uuid}.AUTHORIZED.gdc.tsv"

    @property
    def incoming(self) -> Path:
        return DATA_ROOT / f"BRCA_PRODUCTION_{self.label}.incoming"

    @property
    def wsi(self) -> Path:
        return self.incoming / self.uuid / self.slide

    @property
    def parcel(self) -> Path:
        return self.incoming / self.uuid / "logs" / f"{self.slide}.parcel"

    @property
    def result_bundle(self) -> Path:
        return RESULT_PARENT / f"brca_{self.label.lower()}_header_metadata_result"


PATIENTS = (
    Patient("P0002", 2, "TCGA-4H-AAAK", "TCGA-4H-AAAK-01Z-00-DX1.ABF1B042-1970-4E28-8671-43AAD393D2F9.svs", "fd6a8fe0-50d8-4f69-b678-1a884e1c5d3d", "5", "cedd01a8cdc0854887f75aef21049cfb", 1016839805, "b15b39701d3365503e9b2f340c6e451b04cf97bc03f171c62f7b0e21b386a4f0", ("3017c49e4aef65d10f5acc3cc9fb29838d1836d533261a701ffc91964ae96fe8", "6c4085881e14537c86b2ca93f75799268e8f49852782041c7ccdad5e3a595834", "fc00f4c654f0f6797ade7bcd60715d7c1a3fba0501b6aeec7845506d382f5d49")),
    Patient("P0003", 3, "TCGA-5L-AAT0", "TCGA-5L-AAT0-01Z-00-DX1.5E171263-30BF-4C6B-88A1-E8EA0522A861.svs", "b3780253-78fe-4907-9f5a-230b2bb4b24e", "6", "b03df56d20c11c51b8ad6c3e132a78de", 349684623, "339b2d52fb8948c088cd4ddc79ebb2cbaadeabdee98d03ac44b5702ff90bf982", ("e18d67f8c2b18539e14c03d33aaa0f68fa8772464f2c7ad555c93bf48f54b33b", "c12dfaa5c18591f07ece9ea8de2bbc374d65ebb77d683a075279b663d6a2e6ee", "dc25c9765f3117dc7d0b13a5fb7d5628e32a082aab3058bba58a0776e2cad9c1")),
    Patient("P0004", 4, "TCGA-5L-AAT1", "TCGA-5L-AAT1-01Z-00-DX1.F3449A5B-2AC4-4ED7-BF44-4C8946CDB47D.svs", "4eec69ca-381b-4c17-b3e9-49492d71560e", "7", "12416cc41421ba836c7e11aad25594f4", 592769341, "3692ecd473064cf13850beb4b1ad7a1131669419687a852cb793c65f30ed955c", ("f3958a422eb1a998356eacd55b9d44bbb3d45c42e4e6610a27d443d88e83f0a5", "6340b53289615cfbae6c28744dd2988230dbc2479b5d452a27a4b95a70c75bcb", "e6da6806b069184d59abfd795b9c8a9ade05ca357176b1518d3885b21958b959")),
    Patient("P0005", 5, "TCGA-5T-A9QA", "TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs", "cfe3c99d-0c00-4360-b768-2cb4fbd1040a", "8", "27fd9e24f2ac9701d108461e11db9e90", 2057150279, "0287e08038054bedfac19065016161267a0ade08123ea91b5e3866b52c33384a", ("30c8b12c9e1c2fadda897779fe87f673a6098074c1ef8127d47300359aacb8a5", "4fea5e6a3ec5f5474a26d858bc77b6d7bd3ab864ea02d988683fdc648602b248", "2dd2b13be3ad040104d46d3d3f92ccdb0aec9703335b87c6052d034e70d77f1f")),
    Patient("P0006", 6, "TCGA-A1-A0SB", "TCGA-A1-A0SB-01Z-00-DX1.B34C267B-CAAA-4AB6-AD5C-276C26F997A1.svs", "cea82b7d-135a-49d5-b4f6-3fb0215f7188", "9", "c8251741fc092506901596b30ee27cae", 741184858, "ce31dd16674f4ec1928bec81aba0f621baba9925c9cbf8038a1e38a81f1afec3", ("8b0b3f68f660aa217fd7c445833a909c21119723416e1903f1f2521a8d569ffe", "4fea5e6a3ec5f5474a26d858bc77b6d7bd3ab864ea02d988683fdc648602b248", "2a9628be65556afb0f26659d7b6b97c2f480439b6ee8c15c44fc771512ed863d")),
    Patient("P0007", 7, "TCGA-A1-A0SD", "TCGA-A1-A0SD-01Z-00-DX1.DB17BFA9-D951-42A8-91D2-F4C2EBC6EB9F.svs", "0a9ea7ac-9d51-4ff7-b40b-659a57e64945", "10", "85079bc9f671b29dfebc3d63fdf6b685", 655624470, "c28311d8b4a459167e3f9d1b552af61b8a01135a1f731b5297d7447214c31b06", ("24aa5a590ed4f37b8b7c7f3d10178ab77cb7fa165e8a16bdfa7e556e1e2344a4", "2a25c889832353363c50b08cfc9a43c533a952ce603f489d6001d63430a0c045", "614af2334a4baf7269b2c18fca95e64d15744dee181aed1f24bcea4d76b774da")),
    Patient("P0008", 8, "TCGA-A1-A0SE", "TCGA-A1-A0SE-01Z-00-DX1.04B09232-C6C4-46EF-AA2C-41D078D0A80A.svs", "53f1310d-cea9-4179-ad5c-3a257fcc7ed3", "11", "401a42c3dac58c61c857a148a0a96ad0", 1114028148, "c33cfc39a9c708cc2ddca6ab42447923cd959b6a847a22634fc9d970a26ce931", ("6d8e6b7ff772b8fd88f89ba1fa581f261af6f5853af630dbef1d3f7949f9f06a", "761bc2e512abf34ec489d2ac93a731f922fb9b866e53123023b46b4bfb1732e1", "682920a68076a8d3a53a441d9aa2afaf08206ba866ec01a1d3a6d26a62897fa6")),
)


def require(value: bool, message: str) -> None:
    if not value:
        raise HeaderPackageError(message)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _regular(path: Path) -> os.stat_result:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not stat.S_ISLNK(value.st_mode), f"unsafe file: {path}")
    return value


def _directory(path: Path) -> os.stat_result:
    value = path.lstat()
    require(stat.S_ISDIR(value.st_mode) and not stat.S_ISLNK(value.st_mode), f"unsafe directory: {path}")
    return value


def _token(value: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_mode, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _hash_fd(descriptor: int) -> tuple[str, str]:
    md5 = hashlib.md5()  # noqa: S324 -- GDC identity contract
    sha = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    while chunk := os.read(descriptor, 4 * 1024 * 1024):
        md5.update(chunk)
        sha.update(chunk)
    os.lseek(descriptor, 0, os.SEEK_SET)
    return md5.hexdigest(), sha.hexdigest()


def validate_package() -> dict[str, Any]:
    require(sha256_path(AUTH) == AUTH_SHA256, "authorization SHA256 drift")
    auth = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    require(auth["status"] == "AUTHORIZED_P0002_P0008_ACQUISITION_AND_HEADER_ONLY", "authorization locked")
    require(auth["executable"] is True, "authorization is not executable")
    statement = auth["approval"]["exact_statement"]
    require(hashlib.sha256(statement.encode()).hexdigest() == STATEMENT_SHA256, "statement drift")
    execution = auth["execution"]
    require(execution["download_concurrency"] == 1, "download concurrency must be one")
    require(execution["maximum_cpu_patient_workers"] == 2, "CPU patient workers must be two")
    require(execution["maximum_simultaneous_gdc_clients"] == 1, "GDC clients must be one")
    require(execution["exact_total_declared_bytes"] == TOTAL_BYTES, "total bytes drift")
    require(execution["whole_block_required_available_bytes"] == WHOLE_BLOCK_REQUIRED_AVAILABLE_BYTES, "whole-block storage floor drift")
    for key in ("read_region_or_pixel_access", "mask_or_coordinate_generation", "patch_or_feature_extraction", "healnet_execution", "gpu_or_cuda", "patients_outside_p0001_p0008", "deletion", "google_drive", "cohort_expansion", "training"):
        require(auth["authority"][key] is False, f"prohibited scope unlocked: {key}")
    require(sha256_path(FIRST_EIGHT) == FIRST_EIGHT_SHA256, "first-eight identity source drift")
    with FIRST_EIGHT.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))[1:8]
    require(len(rows) == len(PATIENTS) == 7, "patient cardinality drift")
    for patient, row in zip(PATIENTS, rows, strict=True):
        require((patient.cohort_index, patient.patient_id, patient.slide, patient.uuid, patient.omic_row, patient.md5, patient.size) == (int(row["cohort_index"]), row["patient_id"], row["slide_id"], row["gdc_uuid"], row["omic_source_row_id"], row["md5"], int(row["size_bytes"])), f"frozen identity drift: {patient.label}")
        require(sha256_path(patient.manifest) == patient.manifest_sha256, f"manifest hash drift: {patient.label}")
        with patient.manifest.open(newline="", encoding="utf-8") as stream:
            manifest_rows = list(csv.DictReader(stream, delimiter="\t"))
        require(manifest_rows == [{"id": patient.uuid, "filename": patient.slide, "md5": patient.md5, "size": str(patient.size), "state": "released"}], f"manifest content drift: {patient.label}")
    require(sum(patient.size for patient in PATIENTS) == TOTAL_BYTES, "declared byte sum drift")
    return {"authorization_sha256": AUTH_SHA256, "patient_count": 7, "total_bytes": TOTAL_BYTES}


def validate_source(expected_commit: str) -> str:
    require(len(expected_commit) == 40, "full expected source commit required")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    require(head == expected_commit, "source commit drift")
    critical = (
        AUTH,
        FIRST_EIGHT,
        Path(__file__).resolve(),
        ROOT / "scripts/run_brca_p0002_p0008_header_phase.py",
        ROOT / "multiscale_feature_pilot/__init__.py",
        ROOT / "multiscale_feature_pilot/src/__init__.py",
        ROOT / "multiscale_feature_pilot/src/brca_omic.py",
        ROOT / "multiscale_feature_pilot/src/multiscale_bag.py",
        ROOT / "multiscale_feature_pilot/src/omic.py",
        ROOT / "multiscale_feature_pilot/src/padding.py",
        ROOT / "multiscale_feature_pilot/src/provenance.py",
        *tuple(patient.manifest for patient in PATIENTS),
    )
    for path in critical:
        relative = path.relative_to(ROOT).as_posix()
        require(path.is_file() and not path.is_symlink(), f"unsafe critical source: {relative}")
        committed = subprocess.run(["git", "show", f"HEAD:{relative}"], cwd=ROOT, check=True, capture_output=True).stdout
        require(path.read_bytes() == committed, f"critical source differs from HEAD: {relative}")
    return head


def _active_gdc_clients(proc_root: Path = Path("/proc")) -> list[int]:
    active: list[int] = []
    require(proc_root.is_dir(), "process audit root unavailable")
    for entry in proc_root.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes().split(b"\0")
            comm = (entry / "comm").read_text(encoding="utf-8").strip()
            try:
                executable = (entry / "exe").resolve(strict=True)
            except FileNotFoundError:
                executable = None
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError as exc:
            raise HeaderPackageError(f"cannot audit process {entry.name}") from exc
        argv0 = os.fsdecode(cmdline[0]) if cmdline and cmdline[0] else ""
        names = {Path(argv0).name, comm, executable.name if executable is not None else ""}
        if "gdc-client" in names:
            active.append(int(entry.name))
    return active


def preflight_patient(patient: Patient) -> None:
    require(not patient.incoming.exists() and not patient.incoming.is_symlink(), f"incoming exists: {patient.label}")
    require(not patient.result_bundle.exists() and not patient.result_bundle.is_symlink(), f"result exists: {patient.label}")
    _directory(DATA_ROOT)


def preflight_block() -> dict[str, int]:
    _directory(DATA_ROOT)
    _directory(RESULT_PARENT)
    usage = shutil.disk_usage(DATA_ROOT)
    require(usage.free >= WHOLE_BLOCK_REQUIRED_AVAILABLE_BYTES, "whole-block storage floor failed")
    for patient in PATIENTS:
        preflight_patient(patient)
    return {"available_bytes": usage.free, "required_available_bytes": WHOLE_BLOCK_REQUIRED_AVAILABLE_BYTES}


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_incoming(patient: Patient) -> tuple[int, int, int]:
    require(not patient.incoming.exists() and not patient.incoming.is_symlink(), f"incoming appeared: {patient.label}")
    _directory(DATA_ROOT)
    os.mkdir(patient.incoming, 0o700)
    details = _directory(patient.incoming)
    require(stat.S_IMODE(details.st_mode) == 0o700, f"incoming mode drift: {patient.label}")
    _fsync_directory(DATA_ROOT)
    return details.st_dev, details.st_ino, details.st_mode


def _require_incoming(patient: Patient, identity: tuple[int, int, int]) -> None:
    details = _directory(patient.incoming)
    require((details.st_dev, details.st_ino, details.st_mode) == identity, f"incoming identity drift: {patient.label}")
    require(stat.S_IMODE(details.st_mode) == 0o700, f"incoming mode drift: {patient.label}")


def _diagnostic_tail(value: str, limit: int = 2000) -> str:
    sanitized = "".join(character if character in "\n\t" or ord(character) >= 32 else "?" for character in value)
    return sanitized[-limit:]


def _diagnostic_value(value: str | bytes | None) -> str:
    if value is None:
        return ""
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value


def download_patient(patient: Patient, incoming_identity: tuple[int, int, int]) -> dict[str, Any]:
    require(not _active_gdc_clients(), "another gdc-client is active")
    _regular(CLIENT)
    require(sha256_path(CLIENT) == CLIENT_SHA256, "GDC client drift")
    _require_incoming(patient, incoming_identity)
    command = [str(CLIENT), "download", "--no-related-files", "--no-annotations", "-m", str(patient.manifest), "-d", str(patient.incoming)]
    started = time.perf_counter()
    try:
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT_SECONDS, check=False)
    except subprocess.TimeoutExpired as exc:
        stdout_tail = _diagnostic_tail(_diagnostic_value(exc.stdout))
        stderr_tail = _diagnostic_tail(_diagnostic_value(exc.stderr))
        raise HeaderPackageError(f"GDC timed out for {patient.label}; stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}") from exc
    stdout_tail = _diagnostic_tail(result.stdout)
    stderr_tail = _diagnostic_tail(result.stderr)
    require(result.returncode == 0, f"GDC failed for {patient.label} ({result.returncode}); stdout_tail={stdout_tail!r}; stderr_tail={stderr_tail!r}")
    require(not _active_gdc_clients(), "GDC client remained active")
    _require_incoming(patient, incoming_identity)
    _regular(CLIENT)
    require(sha256_path(CLIENT) == CLIENT_SHA256, "GDC client changed during transfer")
    return {"seconds": time.perf_counter() - started, "argv": command, "returncode": 0, "stdout_tail": stdout_tail, "stderr_tail": stderr_tail}


def validate_tree(patient: Patient, incoming_identity: tuple[int, int, int]) -> dict[str, Any]:
    _require_incoming(patient, incoming_identity)
    expected_directories = sorted((patient.incoming / patient.uuid, patient.incoming / patient.uuid / "logs"))
    all_entries = sorted(patient.incoming.rglob("*"))
    require(not any(path.is_symlink() for path in all_entries), f"symlink found in tree: {patient.label}")
    directories = sorted(path for path in all_entries if path.is_dir())
    require(directories == expected_directories, f"unexpected download directories: {patient.label}")
    for path in (patient.incoming, *directories):
        details = path.lstat()
        require(stat.S_ISDIR(details.st_mode) and not stat.S_ISLNK(details.st_mode), f"unsafe directory: {path}")
    expected_files = sorted((patient.wsi, patient.parcel))
    files = sorted(path for path in all_entries if path.is_file() or path.is_symlink())
    require(files == expected_files, f"unexpected download tree: {patient.label}")
    for path in files:
        _regular(path)
        require(not any(token in path.name.lower() for token in (".part", ".partial", ".tmp", ".incomplete", ".download")), f"partial file: {path}")
    require(_regular(patient.wsi).st_size == patient.size, f"size mismatch: {patient.label}")
    _require_incoming(patient, incoming_identity)
    return {"directories": [path.relative_to(patient.incoming).as_posix() for path in directories], "files": [path.relative_to(patient.incoming).as_posix() for path in files], "partial_or_incomplete_absent": True, "parcel_sha256": sha256_path(patient.parcel)}


def _tensor_hash(tensor: torch.Tensor) -> str:
    require(tensor.device.type == "cpu" and tensor.dtype is torch.float32 and tensor.is_contiguous(), "Omic tensor contract drift")
    require(bool(torch.isfinite(tensor).all()), "non-finite Omic tensor")
    return hashlib.sha256(tensor.numpy().tobytes(order="C")).hexdigest()


def load_omic(patient: Patient) -> dict[str, Any]:
    original = _regular(OMIC)
    require(original.st_size == OMIC_SIZE, "Omic archive size drift")
    path_token = _token(original)
    descriptor = os.open(OMIC, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        token = _token(os.fstat(descriptor))
        require(token == path_token, "Omic pathname/descriptor mismatch")
        stable = Path(f"/proc/self/fd/{descriptor}")
        require(sha256_path(stable) == BRCA_RELEASE_ARCHIVE_SHA256, "Omic archive drift")
        omics = load_official_brca_patient_omics(stable, case_id=patient.patient_id, slide_id=patient.slide)
        require(omics.source_row_index == patient.omic_row, "Omic row drift")
        hashes = tuple(_tensor_hash(value) for value in (omics.rna, omics.mutation, omics.cnv))
        require(hashes == patient.omic_hashes, "Omic content drift")
        require(_token(os.fstat(descriptor)) == token, "held Omic descriptor drift")
        require(_token(_regular(OMIC)) == token, "Omic pathname changed")
        require(sha256_path(stable) == BRCA_RELEASE_ARCHIVE_SHA256, "held Omic archive changed")
        require(_token(os.fstat(descriptor)) == token == _token(_regular(OMIC)), "Omic changed during final hash")
        return {"source_row_index": patient.omic_row, "archive_size_bytes": OMIC_SIZE, "archive_sha256": BRCA_RELEASE_ARCHIVE_SHA256, "content_sha256": dict(zip(("rna", "mutation", "cnv"), hashes, strict=True)), "held_o_nofollow_descriptor": True, "pathname_rechecked": True}
    finally:
        os.close(descriptor)


class _HeaderOnlyProxy:
    def __init__(self, slide: Any) -> None:
        self._slide = slide

    def __getattr__(self, name: str) -> Any:
        if name in {"read_region", "get_thumbnail", "associated_images", "get_associated_image"}:
            raise HeaderPackageError(f"pixel API prohibited: {name}")
        return getattr(self._slide, name)


def collect_header(patient: Patient, slide_factory: Callable[[str], Any] = openslide.OpenSlide) -> tuple[dict[str, Any], str]:
    descriptor = os.open(patient.wsi, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    try:
        token = _token(os.fstat(descriptor))
        require(token == _token(patient.wsi.lstat()), "WSI path/descriptor mismatch")
        first = _hash_fd(descriptor)
        require(first[0] == patient.md5, "WSI MD5 mismatch")
        raw_slide = slide_factory(f"/proc/self/fd/{descriptor}")
        slide = _HeaderOnlyProxy(raw_slide)
        try:
            mpp_x = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
            mpp_y = float(slide.properties[openslide.PROPERTY_NAME_MPP_Y])
            count = int(slide.level_count)
            dimensions = [list(map(int, value)) for value in slide.level_dimensions]
            downsamples = [float(value) for value in slide.level_downsamples]
        finally:
            raw_slide.close()
        require(math.isfinite(mpp_x) and math.isfinite(mpp_y) and mpp_x > 0 and mpp_y > 0, "invalid MPP")
        require(count > 0 and count == len(dimensions) == len(downsamples), "pyramid length drift")
        require(all(w > 0 and h > 0 for w, h in dimensions), "invalid dimensions")
        require(downsamples[0] == 1.0 and all(math.isfinite(x) and x > 0 for x in downsamples), "invalid downsamples")
        require(_token(os.fstat(descriptor)) == token == _token(patient.wsi.lstat()), "WSI changed during header")
        second = _hash_fd(descriptor)
        require(second == first, "same-descriptor hash drift")
        require(_token(os.fstat(descriptor)) == token == _token(patient.wsi.lstat()), "WSI changed during final hash")
        return {"inspection": "OPENSLIDE_HEADER_ONLY_NO_PIXEL_ACCESS", "mpp_x": mpp_x, "mpp_y": mpp_y, "level_count": count, "levels": [{"level": i, "dimensions": dimensions[i], "downsample": downsamples[i]} for i in range(count)], "read_region_calls": 0, "thumbnail_calls": 0, "associated_image_accesses": 0}, first[1]
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0), 0o600)
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            require(written > 0, "short result write")
            offset += written
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish(patient: Patient, record: dict[str, Any], incoming_identity: tuple[int, int, int]) -> None:
    _require_incoming(patient, incoming_identity)
    require(not patient.result_bundle.exists() and not patient.result_bundle.is_symlink(), "append-only result exists")
    parent_details = _directory(RESULT_PARENT)
    parent_identity = (parent_details.st_dev, parent_details.st_ino, parent_details.st_mode)
    stage = RESULT_PARENT / f".{patient.result_bundle.name}.staging.{os.getpid()}"
    require(not stage.exists() and not stage.is_symlink(), "result staging collision")
    os.mkdir(stage, 0o700)
    stage_details = _directory(stage)
    stage_identity = (stage_details.st_dev, stage_details.st_ino, stage_details.st_mode)
    yaml_bytes = yaml.safe_dump(record, sort_keys=False).encode()
    report = f"# {patient.label} production header metadata result\n\n{patient.label} passed exact file, Omic-row, and OpenSlide header-only verification. No pixel API was called.\n\n| Field | Value |\n|---|---|\n| Patient | `{patient.patient_id}` |\n| Slide | `{patient.slide}` |\n| UUID | `{patient.uuid}` |\n| Size | {patient.size:,} bytes |\n| MD5 | `{patient.md5}` |\n| SHA256 | `{record['identity']['sha256']}` |\n| Omic row | `{patient.omic_row}` |\n| MPP | {record['header']['mpp_x']} × {record['header']['mpp_y']} µm/px |\n\nRaw WSI retained. No masks, coordinates, patches, features, GPU/CUDA, HEALNet, deletion, Drive, cohort expansion, or training occurred.\n".encode()
    _write_exclusive(stage / "result.yaml", yaml_bytes)
    _write_exclusive(stage / "report.md", report)
    require((stage / "result.yaml").read_bytes() == yaml_bytes, "staged YAML drift")
    require((stage / "report.md").read_bytes() == report, "staged report drift")
    stage_current = _directory(stage)
    require((stage_current.st_dev, stage_current.st_ino, stage_current.st_mode) == stage_identity, "result staging identity drift")
    _fsync_directory(stage)
    _require_incoming(patient, incoming_identity)
    parent_current = _directory(RESULT_PARENT)
    require((parent_current.st_dev, parent_current.st_ino, parent_current.st_mode) == parent_identity, "result parent identity drift")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 unavailable")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(-100, os.fsencode(stage), -100, os.fsencode(patient.result_bundle), 1)
    if result != 0:
        error = ctypes.get_errno()
        raise HeaderPackageError(f"atomic no-overwrite publication failed: {os.strerror(error)}")
    _fsync_directory(RESULT_PARENT)
    parent_final = _directory(RESULT_PARENT)
    require((parent_final.st_dev, parent_final.st_ino, parent_final.st_mode) == parent_identity, "result parent changed after publication")
    _require_incoming(patient, incoming_identity)


def inspect_and_publish(patient: Patient, transfer: dict[str, Any], incoming_identity: tuple[int, int, int], slide_factory: Callable[[str], Any] = openslide.OpenSlide) -> dict[str, Any]:
    _require_incoming(patient, incoming_identity)
    tree = validate_tree(patient, incoming_identity)
    omic_first = load_omic(patient)
    header, sha256 = collect_header(patient, slide_factory)
    require(load_omic(patient) == omic_first, "Omic rematch drift")
    require(validate_tree(patient, incoming_identity) == tree, "download tree changed")
    require(not torch.cuda.is_initialized(), "CUDA initialized")
    source_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()
    record = {"schema_version": 1, "status": f"BRCA_PRODUCTION_{patient.label}_HEADER_METADATA_VERIFIED", "cohort": "TCGA-BRCA", "production_block": "BRCA_PRODUCTION_BLOCK_0001_0008", "cohort_index": patient.cohort_index, "patient_label": patient.label, "recorded_at_utc": datetime.now(timezone.utc).isoformat(), "execution": {"mode": "CPU_ONLY_ACQUISITION_INTEGRITY_AND_HEADER_ONLY", "source_commit": source_commit, "authorization_sha256": AUTH_SHA256, "authorization_statement_sha256": STATEMENT_SHA256, "authorized_manifest_sha256": patient.manifest_sha256, "download_concurrency": 1, "maximum_cpu_patient_workers": 2, "download": transfer}, "identity": {"patient_id": patient.patient_id, "slide_id": patient.slide, "gdc_uuid": patient.uuid, "path": str(patient.wsi), "size_bytes": patient.size, "md5": patient.md5, "sha256": sha256, "regular_non_symlink": True, "partial_or_incomplete_absent": True}, "download_tree": tree, "omic": omic_first, "header": header, "operations": {"read_region_or_pixel_calls": 0, "mask_or_coordinate_generations": 0, "patch_or_feature_extractions": 0, "healnet_executions": 0, "gpu_or_cuda_operations": 0, "deletions": 0, "drive_operations": 0, "training_runs": 0}, "required_stop_reached": True, "next_gate": f"PREPARE_{patient.label}_SCALE_AND_COORDINATE_POLICY_FROM_FROZEN_METADATA"}
    _require_incoming(patient, incoming_identity)
    _publish(patient, record, incoming_identity)
    _require_incoming(patient, incoming_identity)
    return record


def run(*, expected_source_commit: str) -> list[dict[str, Any]]:
    validate_source(expected_source_commit)
    validate_package()
    require(not _active_gdc_clients(), "another gdc-client is active")
    preflight_block()
    results: list[dict[str, Any]] = []
    pending: Future[dict[str, Any]] | None = None
    # Exactly one main-thread transfer may overlap one preceding patient's
    # CPU-only held-descriptor hash/header worker: at most two active patients.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="brca-header") as pool:
        for patient in PATIENTS:
            preflight_patient(patient)
            incoming_identity = _create_incoming(patient)
            transfer = download_patient(patient, incoming_identity)
            if pending is not None:
                results.append(pending.result())
            pending = pool.submit(inspect_and_publish, patient, transfer, incoming_identity)
        if pending is not None:
            results.append(pending.result())
    require(len(results) == 7, "seven terminal header reports required")
    return results
