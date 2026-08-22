#!/usr/bin/env python3
"""Consolidated P0001--P0008 production GPU feature execution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
DATA_ROOT = WORKSPACE / "brca_pilot_data"
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
OFFICIAL = WORKSPACE / "healnet"
OMIC = WORKSPACE / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip"
CHECKPOINT = Path("/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")
PREEXECUTION = ROOT / "multiscale_feature_pilot/config/brca_p0002_p0008_gpu_preexecution.yaml"
COMBINED_REPORT = ROOT / "multiscale_feature_pilot/provenance/brca_p0001_p0008_gpu_execution_result.json"

EXECUTION_AUTHORIZED = True
EXECUTION_AUTH_SHA256 = "1c73211482bef0df20b9f6379557b7f5dc7f6430d3d5028d09694be66c393511"
CHECKPOINT_SHA256 = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
CHECKPOINT_SIZE = 102_540_417
OFFICIAL_HEAD = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
BRCA_ARCHIVE_SIZE = 4_081_277
FEATURE_DIM = 2_048
BATCH_SIZE = 32

PATIENT_LABELS = tuple(f"P{index:04d}" for index in range(1, 9))
P0001_COUNTS = {"scale_2x": 13_372, "scale_4x": 3_444}
EXPECTED_COUNTS = {
    "P0001": P0001_COUNTS,
    "P0002": {"scale_2x": 9_785, "scale_4x": 2_486},
    "P0003": {"scale_2x": 3_461, "scale_4x": 921},
    "P0004": {"scale_2x": 3_933, "scale_4x": 1_034},
    "P0005": {"scale_2x": 23_559, "scale_4x": 5_971},
    "P0006": {"scale_2x": 7_505, "scale_4x": 1_962},
    "P0007": {"scale_2x": 9_238, "scale_4x": 2_407},
    "P0008": {"scale_2x": 18_877, "scale_4x": 4_799},
}


class ExecutionLocked(RuntimeError):
    """Raised before any prohibited or unbound operation can proceed."""


class ManualRecoveryRequired(RuntimeError):
    """Raised when durable state exists but cannot be safely advanced."""


@dataclass(frozen=True)
class PatientRuntime:
    label: str
    patient_id: str
    slide_id: str
    gdc_uuid: str
    omic_source_row_id: str
    wsi_path: Path
    wsi_size: int
    wsi_md5: str
    wsi_sha256: str
    level_dimensions: tuple[tuple[int, int], ...]
    level_downsamples: tuple[float, ...]
    coordinate_manifest_sha256: str
    policy_sha256: str
    scale_2x_rows: int
    scale_4x_rows: int
    feature_directory: Path
    coordinate_directory: Path
    recovery_directory: Path

    @property
    def total_rows(self) -> int:
        return self.scale_2x_rows + self.scale_4x_rows


class HeldFileDescriptor:
    """Hold an immutable no-follow file descriptor for identity-sensitive inputs."""

    def __init__(
        self,
        path: Path,
        *,
        expected_size: int,
        expected_sha256: str,
        label: str,
        expected_md5: str | None = None,
    ) -> None:
        self.path = path
        self.label = label
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.expected_md5 = expected_md5
        if not os.path.lexists(path):
            raise RuntimeError(f"{label} is absent: {path}")
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise RuntimeError(f"{label} must be a regular non-symlink file")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        try:
            opened = os.fstat(descriptor)
            self._token = self._stat_token(opened)
            if self._token != self._stat_token(before):
                raise RuntimeError(f"{label} identity changed during secure open")
            if opened.st_size != expected_size:
                raise RuntimeError(f"{label} size drift")
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        self.stable_path = f"/proc/self/fd/{descriptor}"
        self.closed = False

    @staticmethod
    def _stat_token(details: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            details.st_dev,
            details.st_ino,
            details.st_size,
            details.st_mtime_ns,
            details.st_ctime_ns,
        )

    def digest(self, algorithm: str) -> str:
        hasher = hashlib.new(algorithm)
        offset = 0
        while offset < self.expected_size:
            chunk = os.pread(
                self.descriptor,
                min(8 * 1024 * 1024, self.expected_size - offset),
                offset,
            )
            if not chunk:
                raise RuntimeError(f"unexpected EOF from {self.label}")
            hasher.update(chunk)
            offset += len(chunk)
        if os.pread(self.descriptor, 1, self.expected_size) != b"":
            raise RuntimeError(f"{self.label} grew during hashing")
        return hasher.hexdigest()

    def validate(self) -> None:
        if self.closed:
            raise RuntimeError(f"{self.label} descriptor is closed")
        current = self._stat_token(os.fstat(self.descriptor))
        if current != self._token:
            raise RuntimeError(f"{self.label} descriptor identity drift")
        if self.digest("sha256") != self.expected_sha256:
            raise RuntimeError(f"{self.label} SHA256 drift")
        if self.expected_md5 is not None and self.digest("md5") != self.expected_md5:
            raise RuntimeError(f"{self.label} MD5 drift")

    def close(self) -> None:
        if not self.closed:
            os.close(self.descriptor)
            self.closed = True


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*arguments: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def tensor_sha256(tensor: object) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_environment(expected_commit: str) -> None:
    if not EXECUTION_AUTHORIZED:
        raise ExecutionLocked("consolidated GPU execution is not authorized")
    if git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("source commit drift")
    if git("rev-parse", "HEAD", cwd=OFFICIAL) != OFFICIAL_HEAD:
        raise RuntimeError("official HEALNet commit drift")
    if git("status", "--porcelain", cwd=OFFICIAL):
        raise RuntimeError("official HEALNet worktree is not clean")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG=:4096:8 is required")
    if not PREEXECUTION.is_file():
        raise RuntimeError("preexecution contract missing")
    import yaml

    preexecution = yaml.safe_load(PREEXECUTION.read_text(encoding="utf-8"))
    if (
        preexecution["future_exact_authorization_text_sha256"] != EXECUTION_AUTH_SHA256
        or preexecution["consolidated_future_gpu_plan"]["total_patch_reads"] != 112_754
        or preexecution["consolidated_future_gpu_plan"]["maximum_concurrent_patients"] != 1
    ):
        raise RuntimeError("consolidated authorization/preexecution drift")


def load_patient_runtimes() -> tuple[PatientRuntime, ...]:
    import yaml

    sys.path.insert(0, str(ROOT))
    from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
        validate_brca_coordinate_artifacts,
    )
    from multiscale_feature_pilot.src.brca_p0002_p0008_feature_package import (
        prepare_p0002_p0008_feature_packages,
    )
    from multiscale_feature_pilot.src.brca_p0001_feature_package import (
        COORDINATE_MANIFEST_SHA256 as P0001_COORD_SHA,
        GDC_UUID as P0001_UUID,
        OMIC_SOURCE_ROW_ID as P0001_OMIC_ROW,
        PATIENT_ID as P0001_PATIENT,
        SLIDE_ID as P0001_SLIDE,
    )
    from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
        load_frozen_cohort_order,
    )

    cohort = load_frozen_cohort_order(ALIGNMENT)
    p2_p8 = {item.label: item for item in prepare_p0002_p0008_feature_packages()}
    runtimes: list[PatientRuntime] = []
    for label in PATIENT_LABELS:
        binding = cohort[int(label[1:]) - 1]
        header_path = ROOT / f"multiscale_feature_pilot/provenance/brca_{label.lower()}_header_metadata_result/result.yaml"
        if not header_path.is_file():
            raise RuntimeError(f"missing header result for {label}")
        header = yaml.safe_load(header_path.read_text(encoding="utf-8"))
        identity = header["identity"]
        levels = header["header"]["levels"]
        gdc_uuid = identity.get("gdc_uuid", identity.get("gdc_file_uuid"))
        if (
            identity["patient_id"] != binding.patient_id
            or identity["slide_id"] != binding.slide_id
            or gdc_uuid != binding.gdc_uuid
        ):
            raise RuntimeError(f"header identity drift: {label}")
        coordinate_dir = DATA_ROOT / f"BRCA_PRODUCTION_{label}.coordinates"
        if label == "P0001":
            coordinate_manifest_sha256 = P0001_COORD_SHA
            if (binding.patient_id, binding.slide_id, binding.gdc_uuid, binding.omic_source_row_id) != (
                P0001_PATIENT,
                P0001_SLIDE,
                P0001_UUID,
                P0001_OMIC_ROW,
            ):
                raise RuntimeError("P0001 binding drift")
        else:
            coordinate_manifest_sha256 = p2_p8[label].coordinate_manifest_sha256
        record = validate_brca_coordinate_artifacts(
            coordinate_dir, expected_manifest_sha256=coordinate_manifest_sha256
        )
        counts = EXPECTED_COUNTS[label]
        if (
            record.branch_for("scale_2x").coordinate_count != counts["scale_2x"]
            or record.branch_for("scale_4x").coordinate_count != counts["scale_4x"]
        ):
            raise RuntimeError(f"coordinate count drift: {label}")
        manifest = json.loads((coordinate_dir / "coordinate_manifest.json").read_text(encoding="utf-8"))
        policy_sha = manifest["branches"]["scale_2x"]["attributes"]["policy_sha256"]
        runtimes.append(
            PatientRuntime(
                label=label,
                patient_id=binding.patient_id,
                slide_id=binding.slide_id,
                gdc_uuid=binding.gdc_uuid,
                omic_source_row_id=binding.omic_source_row_id,
                wsi_path=Path(identity["path"]),
                wsi_size=int(identity["size_bytes"]),
                wsi_md5=identity["md5"],
                wsi_sha256=identity["sha256"],
                level_dimensions=tuple(tuple(item["dimensions"]) for item in levels),
                level_downsamples=tuple(float(item["downsample"]) for item in levels),
                coordinate_manifest_sha256=coordinate_manifest_sha256,
                policy_sha256=policy_sha,
                scale_2x_rows=counts["scale_2x"],
                scale_4x_rows=counts["scale_4x"],
                feature_directory=DATA_ROOT / f"BRCA_PRODUCTION_{label}.features",
                coordinate_directory=coordinate_dir,
                recovery_directory=DATA_ROOT / f"BRCA_PRODUCTION_{label}.recovery_v2",
            )
        )
    if sum(item.total_rows for item in runtimes) != 112_754:
        raise RuntimeError("first-eight total patch count drift")
    return tuple(runtimes)


def ensure_prefix(runtime: PatientRuntime, authorization_sha256: str) -> None:
    sys.path.insert(0, str(ROOT))
    from multiscale_feature_pilot.src.brca_p0001_feature_package import (
        rehearse_p0001_feature_transaction,
    )
    from multiscale_feature_pilot.src.brca_p0002_p0008_feature_package import (
        prepare_p0002_p0008_feature_packages,
        rehearse_feature_recovery,
    )
    from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
    from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
        ReplayAction,
        load_events,
        replay_events,
    )
    from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
        append_control_event,
    )

    events = list(load_events(runtime.recovery_directory))
    if events:
        replay = replay_events(events)
        if (
            replay.action is ReplayAction.ADVANCE_STAGE
            and replay.target_stage is PatientStage.GPU_AUTHORIZED
        ):
            return
        if (
            replay.action is ReplayAction.ADVANCE_STAGE
            and replay.target_stage is PatientStage.FEATURES_VERIFIED
        ):
            return
        raise ManualRecoveryRequired(f"{runtime.label} recovery ledger is not at a GPU-ready tip")
    if runtime.label == "P0001":
        prefix = rehearse_p0001_feature_transaction(
            ALIGNMENT,
            gpu_authorization_sha256=authorization_sha256,
            through_stage=PatientStage.COORDINATES_VERIFIED,
        ).events
    else:
        package = {
            item.label: item for item in prepare_p0002_p0008_feature_packages()
        }[runtime.label]
        prefix = rehearse_feature_recovery(package)[:10]
    for event in prefix:
        append_control_event(runtime.recovery_directory, event)
    replay = replay_events(load_events(runtime.recovery_directory))
    if replay.action is not ReplayAction.ADVANCE_STAGE or replay.target_stage is not PatientStage.GPU_AUTHORIZED:
        raise RuntimeError(f"{runtime.label} prefix did not reach GPU_AUTHORIZED gate")


def append_stage_success(
    runtime: PatientRuntime,
    *,
    stage_name: str,
    authorization_sha256: str,
    input_hashes: dict[str, str],
    output_hashes: tuple[tuple[str, str], ...],
    compact_artifact: object | None,
) -> None:
    sys.path.insert(0, str(ROOT))
    from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
    from multiscale_feature_pilot.src.brca_streaming_executor_v2 import start_event_from_plan
    from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
        ValidatedStageOutcome,
        append_control_event,
        load_frozen_cohort_order,
        plan_bound_stage,
        transaction_identity,
        validated_success_event,
    )
    from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import load_events
    from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import replay_events

    stage = PatientStage(stage_name)
    events = list(load_events(runtime.recovery_directory))
    binding = load_frozen_cohort_order(ALIGNMENT)[int(runtime.label[1:]) - 1]
    replay = replay_events(events)
    transaction_id = (
        replay.identity.transaction_id
        if replay.identity is not None
        else str(uuid.uuid5(uuid.NAMESPACE_URL, f"brca-first-eight-production/{runtime.label}"))
    )
    identity = transaction_identity(
        binding,
        transaction_id,
    )
    plan = plan_bound_stage(
        events,
        binding=binding,
        identity=identity,
        run_id=str(uuid.uuid4()),
        authorization_sha256=authorization_sha256,
        retry_authorization_sha256=None,
        stage_input_hashes=input_hashes,
    )
    if plan.stage is not stage:
        raise RuntimeError(f"{runtime.label} recovery stage drift: expected {stage.value}, got {plan.stage.value}")
    started = start_event_from_plan(events, plan, recorded_at_utc=utc_now())
    append_control_event(runtime.recovery_directory, started)
    events.append(started)
    success = validated_success_event(
        events,
        plan,
        ValidatedStageOutcome(
            stage=stage,
            authorization_sha256=authorization_sha256,
            output_hashes=output_hashes,
            validation_record_sha256=output_hashes[0][1],
            compact_artifact=compact_artifact,
        ),
        existing_output_hashes=None,
        recorded_at_utc=utc_now(),
    )
    append_control_event(runtime.recovery_directory, success)


def execute_patient(runtime: PatientRuntime, model: object, device: object, commit: str) -> dict[str, object]:
    import h5py
    import numpy as np
    import openslide
    import torch

    sys.path.insert(0, str(ROOT))
    from multiscale_feature_pilot.src.brca_compact_feature_artifacts import (
        COMBINED_FILENAME,
        MANIFEST_FILENAME,
        PROVENANCE_FILENAME,
        SIDECAR_FILENAME,
        CompactFeatureMetadata,
        publish_compact_feature_artifacts,
        validate_compact_feature_artifacts,
    )
    from multiscale_feature_pilot.src.brca_omic import (
        BRCA_RELEASE_ARCHIVE_SHA256,
        load_official_brca_patient_omics,
    )
    from multiscale_feature_pilot.src.brca_singleton_streaming_policy import PatientStage
    from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
        COMPACT_FILES,
        CompactArtifactValidationEvidence,
        load_frozen_cohort_order,
        plan_bound_stage,
        transaction_identity,
    )
    from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
        ReplayAction,
        load_events,
        replay_events,
    )
    from multiscale_feature_pilot.src.feature_extraction import (
        PatchBranchSpec,
        StreamingOpenSlideDataset,
        extract_feature_matrix,
    )
    from multiscale_feature_pilot.src.provenance import (
        BranchProvenanceSpec,
        build_two_scale_provenance,
    )
    from multiscale_feature_pilot.src.supervisor_healnet_smoke import (
        run_one_patient_supervisor_healnet_smoke,
    )

    class CountedSlide:
        def __init__(self, stable_path: str, coordinates: object, level: int, read_size: int) -> None:
            self.stable_path = stable_path
            self.coordinates = coordinates
            self.level = level
            self.read_size = read_size
            self.slide = None
            self.open_count = 0
            self.read_count = 0
            self.closed = False

        def __call__(self, path: str) -> "CountedSlide":
            if path != self.stable_path or self.slide is not None:
                raise RuntimeError("OpenSlide path/open contract drift")
            self.slide = openslide.OpenSlide(path)
            self.open_count += 1
            return self

        def read_region(self, location: tuple[int, int], level: int, size: tuple[int, int]) -> object:
            if self.slide is None or self.closed:
                raise RuntimeError("OpenSlide is not open")
            if self.read_count >= len(self.coordinates):
                raise RuntimeError("patch-read count exceeded")
            expected = tuple(int(value) for value in self.coordinates[self.read_count].tolist())
            if location != expected or level != self.level or size != (self.read_size, self.read_size):
                raise RuntimeError("patch read tuple/order drift")
            self.read_count += 1
            return self.slide.read_region(location, level, size)

        def close(self) -> None:
            if self.slide is not None and not self.closed:
                self.slide.close()
                self.closed = True

    def extract_branch(branch: str, coordinates: object, held_wsi: HeldFileDescriptor):
        if branch == "scale_2x":
            patch_level = 0
            patch_size = 256
            read_size = 512
        else:
            patch_level = 1
            patch_size = 256
            read_size = 256
        counted = CountedSlide(held_wsi.stable_path, coordinates, patch_level, read_size)
        dataset = StreamingOpenSlideDataset(
            runtime.wsi_path,
            PatchBranchSpec(branch, coordinates, patch_level, patch_size),
            slide_factory=counted,
        )
        dataset.wsi_path = held_wsi.stable_path
        try:
            result = extract_feature_matrix(
                dataset,
                model,
                device=device,
                batch_size=BATCH_SIZE,
                num_workers=0,
            )
        finally:
            dataset.close()
        if counted.open_count != 1 or counted.read_count != len(coordinates) or not counted.closed:
            raise RuntimeError(f"{runtime.label}/{branch} OpenSlide lifecycle drift")
        return result, counted.open_count, counted.read_count

    if runtime.feature_directory.exists() or runtime.feature_directory.is_symlink():
        raise ManualRecoveryRequired(f"{runtime.label} feature directory already exists")
    if list(runtime.feature_directory.parent.glob(f".{runtime.feature_directory.name}.staging.*")):
        raise ManualRecoveryRequired(f"{runtime.label} feature staging exists")

    ensure_prefix(runtime, EXECUTION_AUTH_SHA256)
    replay = replay_events(load_events(runtime.recovery_directory))
    if replay.action is not ReplayAction.ADVANCE_STAGE or replay.target_stage is not PatientStage.GPU_AUTHORIZED:
        raise RuntimeError(f"{runtime.label} is not ready for GPU_AUTHORIZED")
    append_stage_success(
        runtime,
        stage_name=PatientStage.GPU_AUTHORIZED.value,
        authorization_sha256=EXECUTION_AUTH_SHA256,
        input_hashes={
            "checkpoint_identity": CHECKPOINT_SHA256,
            "coordinate_manifest": runtime.coordinate_manifest_sha256,
        },
        output_hashes=(("gpu_authorization_record", EXECUTION_AUTH_SHA256),),
        compact_artifact=None,
    )

    coords = {}
    for branch in ("scale_2x", "scale_4x"):
        with h5py.File(runtime.coordinate_directory / f"{branch}_coordinates.h5", "r") as source:
            coords[branch] = torch.from_numpy(np.asarray(source["coords"], dtype=np.int64)).contiguous()
    if (coords["scale_2x"].shape[0], coords["scale_4x"].shape[0]) != (
        runtime.scale_2x_rows,
        runtime.scale_4x_rows,
    ):
        raise RuntimeError(f"{runtime.label} coordinate tensor count drift")

    held_omic = HeldFileDescriptor(
        OMIC,
        expected_size=BRCA_ARCHIVE_SIZE,
        expected_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
        label="BRCA Omic archive",
    )
    try:
        held_omic.validate()
        omic = load_official_brca_patient_omics(
            held_omic.stable_path,
            case_id=runtime.patient_id,
            slide_id=runtime.slide_id,
        )
        held_omic.validate()
    finally:
        held_omic.close()
    if omic.source_row_index != runtime.omic_source_row_id:
        raise RuntimeError(f"{runtime.label} Omic row drift")

    held_wsi = HeldFileDescriptor(
        runtime.wsi_path,
        expected_size=runtime.wsi_size,
        expected_sha256=runtime.wsi_sha256,
        expected_md5=runtime.wsi_md5,
        label=f"{runtime.label} WSI",
    )
    try:
        held_wsi.validate()
        with openslide.OpenSlide(held_wsi.stable_path) as slide:
            if tuple(slide.level_dimensions) != runtime.level_dimensions or tuple(float(value) for value in slide.level_downsamples) != runtime.level_downsamples:
                raise RuntimeError(f"{runtime.label} WSI header drift")
        feature_replay = replay_events(load_events(runtime.recovery_directory))
        if feature_replay.action is not ReplayAction.ADVANCE_STAGE or feature_replay.target_stage is not PatientStage.FEATURES_VERIFIED:
            raise RuntimeError(f"{runtime.label} is not ready for FEATURES_VERIFIED")
        binding = load_frozen_cohort_order(ALIGNMENT)[int(runtime.label[1:]) - 1]
        events = list(load_events(runtime.recovery_directory))
        existing_replay = replay_events(events)
        if existing_replay.identity is None:
            raise RuntimeError(f"{runtime.label} recovery identity missing before feature plan")
        identity = transaction_identity(binding, existing_replay.identity.transaction_id)
        feature_plan = plan_bound_stage(
            events,
            binding=binding,
            identity=identity,
            run_id=str(uuid.uuid4()),
            authorization_sha256=EXECUTION_AUTH_SHA256,
            retry_authorization_sha256=None,
            stage_input_hashes={
                "checkpoint_identity": CHECKPOINT_SHA256,
                "coordinate_manifest": runtime.coordinate_manifest_sha256,
                "gpu_authorization_record": EXECUTION_AUTH_SHA256,
            },
        )
        synthetic = run_one_patient_supervisor_healnet_smoke(
            official_repo=OFFICIAL,
            wsi=torch.zeros((1, runtime.total_rows, FEATURE_DIM), device=device),
            rna=torch.zeros((1, 1, 1558), device=device),
            mutation=torch.zeros((1, 1, 21), device=device),
            cnv=torch.zeros((1, 1, 1333), device=device),
        )
        started = utc_now()
        start = time.perf_counter()
        result_2x, opens_2x, reads_2x = extract_branch("scale_2x", coords["scale_2x"], held_wsi)
        result_4x, opens_4x, reads_4x = extract_branch("scale_4x", coords["scale_4x"], held_wsi)
        held_wsi.validate()
    finally:
        held_wsi.close()

    combined = torch.cat((result_2x.features, result_4x.features), dim=0).contiguous()
    if tuple(combined.shape) != (runtime.total_rows, FEATURE_DIM):
        raise RuntimeError(f"{runtime.label} combined tensor shape drift")
    if combined.dtype is not torch.float32 or combined.requires_grad or not bool(torch.isfinite(combined).all()):
        raise RuntimeError(f"{runtime.label} combined tensor numerical drift")
    real = run_one_patient_supervisor_healnet_smoke(
        official_repo=OFFICIAL,
        wsi=combined.unsqueeze(0).to(device),
        rna=omic.rna.to(device),
        mutation=omic.mutation.to(device),
        cnv=omic.cnv.to(device),
    )
    rows = build_two_scale_provenance(
        scale_2x=BranchProvenanceSpec("scale_2x", coords["scale_2x"], 0, 0.5, 0.5),
        scale_4x=BranchProvenanceSpec("scale_4x", coords["scale_4x"], 1, 1.0, 1.0),
        scale_2x_count=runtime.scale_2x_rows,
        scale_4x_count=runtime.scale_4x_rows,
    )
    metadata = CompactFeatureMetadata(
        runtime.patient_id,
        runtime.slide_id,
        runtime.gdc_uuid,
        runtime.wsi_sha256,
        runtime.coordinate_manifest_sha256,
        BRCA_RELEASE_ARCHIVE_SHA256,
        CHECKPOINT_SHA256,
        runtime.policy_sha256,
        commit,
        runtime.scale_2x_rows,
        runtime.scale_4x_rows,
    )
    manifest = publish_compact_feature_artifacts(
        runtime.feature_directory,
        combined_features=combined,
        row_provenance=rows,
        metadata=metadata,
        preserve_failed_staging=False,
    )
    anchor = sha256_path(runtime.feature_directory / MANIFEST_FILENAME)
    validate_compact_feature_artifacts(runtime.feature_directory, expected_manifest_sha256=anchor)
    exact_names = (COMBINED_FILENAME, PROVENANCE_FILENAME, MANIFEST_FILENAME, SIDECAR_FILENAME)
    file_hashes = tuple((name, sha256_path(runtime.feature_directory / name)) for name in exact_names)
    compact_evidence = CompactArtifactValidationEvidence(
        patient_id=runtime.patient_id,
        slide_id=runtime.slide_id,
        gdc_uuid=runtime.gdc_uuid,
        omic_source_row_id=runtime.omic_source_row_id,
        bound_source_policy_hashes=feature_plan.source_policy_hashes,
        bound_input_hashes=feature_plan.input_hashes,
        exact_files=COMPACT_FILES,
        manifest_sha256=anchor,
        sidecar_manifest_sha256=anchor,
        file_hashes=file_hashes,
        tensor_shape=(runtime.total_rows, FEATURE_DIM),
        tensor_dtype="float32",
        tensor_device="cpu",
        tensor_contiguous=True,
        tensor_finite=True,
        tensor_requires_grad=False,
        scale_2x_row_range=(0, runtime.scale_2x_rows),
        scale_4x_row_range=(runtime.scale_2x_rows, runtime.total_rows),
        row_provenance_count=runtime.total_rows,
    )
    append_stage_success(
        runtime,
        stage_name=PatientStage.FEATURES_VERIFIED.value,
        authorization_sha256=EXECUTION_AUTH_SHA256,
        input_hashes={
            "checkpoint_identity": CHECKPOINT_SHA256,
            "coordinate_manifest": runtime.coordinate_manifest_sha256,
            "gpu_authorization_record": EXECUTION_AUTH_SHA256,
        },
        output_hashes=(("compact_feature_manifest", anchor),),
        compact_artifact=compact_evidence,
    )
    final_replay = replay_events(load_events(runtime.recovery_directory))
    if final_replay.action is not ReplayAction.ADVANCE_STAGE:
        raise RuntimeError(f"{runtime.label} recovery tip did not advance after features")
    return {
        "label": runtime.label,
        "patient_id": runtime.patient_id,
        "slide_id": runtime.slide_id,
        "counts": {
            "scale_2x": runtime.scale_2x_rows,
            "scale_4x": runtime.scale_4x_rows,
            "total": runtime.total_rows,
        },
        "feature_shapes": {
            "scale_2x": list(result_2x.features.shape),
            "scale_4x": list(result_4x.features.shape),
            "combined": list(combined.shape),
            "natural_healnet_wsi": [1, runtime.total_rows, FEATURE_DIM],
        },
        "timings_seconds": {
            "scale_2x": result_2x.streaming_extraction_seconds,
            "scale_4x": result_4x.streaming_extraction_seconds,
            "patient_total": time.perf_counter() - start,
        },
        "patch_io": {
            "openslide_patch_opens": opens_2x + opens_4x,
            "scale_2x_read_region_calls": reads_2x,
            "scale_4x_read_region_calls": reads_4x,
            "total_read_region_calls": reads_2x + reads_4x,
        },
        "peak_gpu_memory_bytes": max(result_2x.peak_gpu_memory_bytes, result_4x.peak_gpu_memory_bytes),
        "synthetic_healnet": asdict(synthetic),
        "real_healnet": asdict(real),
        "artifact": {
            "directory": str(runtime.feature_directory),
            "manifest_sha256": anchor,
            "total_file_bytes": sum(path.stat().st_size for path in runtime.feature_directory.iterdir()),
            "manifest": manifest,
        },
        "feature_content_sha256": tensor_sha256(combined),
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
    }


def run(expected_commit: str) -> dict[str, object]:
    validate_environment(expected_commit)
    runtimes = load_patient_runtimes()

    import torch
    import torchvision

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU required; CPU fallback forbidden")
    if "Tesla T4" not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("Tesla T4 capability 7.5 required")
    device = torch.device("cuda:0")
    held_checkpoint = HeldFileDescriptor(
        CHECKPOINT,
        expected_size=CHECKPOINT_SIZE,
        expected_sha256=CHECKPOINT_SHA256,
        label="ResNet50 ImageNet1K V2 checkpoint",
    )
    try:
        held_checkpoint.validate()
        state = torch.load(held_checkpoint.stable_path, map_location="cpu", weights_only=True)
        model = torchvision.models.resnet50(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()
        held_checkpoint.validate()
        started = utc_now()
        wall_start = time.perf_counter()
        patients: list[dict[str, object]] = []
        for runtime in runtimes:
            patients.append(execute_patient(runtime, model, device, expected_commit))
        result = {
            "schema_version": 1,
            "status": "BRCA_PRODUCTION_P0001_P0008_GPU_FEATURES_VERIFIED",
            "recorded_at_utc": utc_now(),
            "execution": {
                "mode": "SERIAL_CONSOLIDATED_P0001_P0008_GPU_FEATURE_EXECUTION",
                "started_at_utc": started,
                "finished_at_utc": utc_now(),
                "total_seconds": time.perf_counter() - wall_start,
                "source_commit": expected_commit,
                "authorization_sha256": EXECUTION_AUTH_SHA256,
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "torchvision": torchvision.__version__,
                "batch_size": BATCH_SIZE,
                "amp": False,
                "tf32": False,
                "cpu_fallback": False,
            },
            "patients": patients,
            "totals": {
                "patients": len(patients),
                "scale_2x_patch_reads": sum(item["counts"]["scale_2x"] for item in patients),
                "scale_4x_patch_reads": sum(item["counts"]["scale_4x"] for item in patients),
                "total_patch_reads": sum(item["counts"]["total"] for item in patients),
                "artifact_bytes": sum(item["artifact"]["total_file_bytes"] for item in patients),
                "peak_gpu_memory_bytes": max(item["peak_gpu_memory_bytes"] for item in patients),
            },
            "operations": {
                "training": 0,
                "backward": 0,
                "optimizer_steps": 0,
                "amp": 0,
                "tf32": 0,
                "coordinate_regeneration": 0,
                "patients_outside_p0001_p0008": 0,
                "drive": 0,
                "raw_or_final_deletions": 0,
            },
            "required_stop_reached": True,
        }
        if result["totals"]["total_patch_reads"] != 112_754:
            raise RuntimeError("consolidated patch total drift")
        tmp = COMBINED_REPORT.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, COMBINED_REPORT)
        return result
    finally:
        held_checkpoint.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-commit", required=True)
    args = parser.parse_args()
    try:
        result = run(args.expected_source_commit)
    except Exception as error:
        print(json.dumps({"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"}, sort_keys=True))
        return 1
    print(json.dumps({"status": result["status"], "report": str(COMBINED_REPORT), "totals": result["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
