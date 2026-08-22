#!/usr/bin/env python3
"""Execution-locked P0001 production GPU feature package."""

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
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EXECUTION_AUTHORIZED = True
EXECUTION_AUTH_SHA256 = "f69561be5c822d7f1ecfb0b54a41789855376de31a90d942260c426900a4b753"
AUTHORIZATION_STATEMENT_SHA256 = (
    "86db1c4ddb2deead9ce8ca0d6282a890136c3e63451685aa862909a184867f48"
)
AUTH = ROOT / "multiscale_feature_pilot/config/brca_p0001_gpu_execution_authorization.yaml"
PREEXECUTION = ROOT / "multiscale_feature_pilot/config/brca_p0001_gpu_preexecution.yaml"
ALIGNMENT = ROOT / "reports/brca_row_level_alignment.csv"
PATIENT = "TCGA-3C-AALK"
SLIDE = "TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53.svs"
UUID = "93b26333-5723-4fa4-a4de-6124c04ab243"
WSI = (
    WORKSPACE
    / "brca_pilot_data/BRCA_PRODUCTION_P0001.incoming"
    / UUID
    / SLIDE
)
COORD = WORKSPACE / "brca_pilot_data/BRCA_PRODUCTION_P0001.coordinates"
OUTPUT = WORKSPACE / "brca_pilot_data/BRCA_PRODUCTION_P0001.features"
LEDGER = WORKSPACE / "brca_pilot_data/BRCA_PRODUCTION_P0001.recovery_v2"
OMIC = (
    WORKSPACE
    / "Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
CHECKPOINT = Path("/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")
OFFICIAL = WORKSPACE / "healnet"

WSI_SIZE = 1_769_848_096
OMIC_SIZE = 4_081_277
CHECKPOINT_SIZE = 102_540_417
WSI_MD5 = "3d63b3311612d763525b6edb0848b986"
WSI_SHA256 = "f43597a87463d8d15007918dd5174ff966aa28dcb0de71cdc5752576cd7c2b5b"
COORDINATE_MANIFEST_SHA256 = (
    "f1825acf7d0b96c92bfb66038d60af6e1572ea945490e2245d5e3ec222677d6e"
)
POLICY_SHA256 = "d5ddfdf38f98a921a0876e71ed57fbf196407cf8bbf234433f3c0c0a46513cd4"
COORDINATE_RESULT_SHA256 = (
    "687cc3d7515f102666846310ef00fbad4e1df6b77ee3fd1f4689e659fa0f7ef2"
)
CHECKPOINT_SHA256 = (
    "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
)
OFFICIAL_HEAD = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
DIMS = ((95488, 81920), (23872, 20480), (5968, 5120), (2984, 2560))
DOWNSAMPLES = (1.0, 4.0, 16.0, 32.0)
COUNTS = (13_372, 3_444)
TOTAL = 16_816
OMIC_CONTENT_SHA256 = {
    "rna": "1894e15a5dbba2559c61e8521394599153a0ada90cf482fd9eb0c45347f5082a",
    "mutation": "33767ff31d3c7c11a69ba46c746125f679492e03f5dec7c48f8117aa2a6b3c52",
    "cnv": "a0ef410e624c698475b78dc0270bf2613e2e320ba4935f8580cb0867d41bfd50",
}
BOUND = (
    Path("scripts/run_brca_p0001_gpu_production.py"),
    Path("multiscale_feature_pilot/config/brca_p0001_gpu_execution_authorization.yaml"),
    Path("multiscale_feature_pilot/config/brca_p0001_gpu_preexecution.yaml"),
    Path("multiscale_feature_pilot/provenance/brca_p0001_coordinate_execution_result.yaml"),
    Path("multiscale_feature_pilot/config/brca_p0001_scale_coordinate_policy.yaml"),
    Path("multiscale_feature_pilot/config/brca_compact_artifact_policy.yaml"),
    Path("multiscale_feature_pilot/config/brca_894_streaming_executor_v2_policy.yaml"),
    Path("multiscale_feature_pilot/config/brca_894_singleton_streaming_policy.yaml"),
    Path("multiscale_feature_pilot/src/brca_p0001_feature_package.py"),
    Path("multiscale_feature_pilot/src/brca_compact_feature_artifacts.py"),
    Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
    Path("multiscale_feature_pilot/src/brca_omic.py"),
    Path("multiscale_feature_pilot/src/brca_streaming_production_adapter.py"),
    Path("multiscale_feature_pilot/src/brca_streaming_executor_v2.py"),
    Path("multiscale_feature_pilot/src/brca_streaming_recovery_v2.py"),
    Path("multiscale_feature_pilot/src/brca_singleton_streaming_policy.py"),
    Path("multiscale_feature_pilot/src/feature_extraction.py"),
    Path("multiscale_feature_pilot/src/scale_2x_policy.py"),
    Path("multiscale_feature_pilot/src/provenance.py"),
    Path("multiscale_feature_pilot/src/supervisor_healnet_smoke.py"),
    Path("multiscale_feature_pilot/__init__.py"),
    Path("multiscale_feature_pilot/src/__init__.py"),
    Path("multiscale_feature_pilot/src/omic.py"),
    Path("multiscale_feature_pilot/src/padding.py"),
    Path("multiscale_feature_pilot/src/multiscale_bag.py"),
)
ALLOWED_STATUS = {
    " M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}


class ExecutionLocked(RuntimeError):
    """Raised before imports, path checks, pixels, CUDA, or publication."""


class ManualRecoveryRequired(RuntimeError):
    """Raised when durable output and recovery-v2 cannot be reconciled automatically."""


class _HeldFileDescriptor:
    """Hold one immutable no-follow input identity across every consumer."""

    def __init__(
        self,
        path: Path,
        *,
        expected_size: int,
        expected_sha256: str,
        label: str,
        expected_md5: str | None = None,
    ) -> None:
        self.label = label
        self.expected_size = expected_size
        self.expected_sha256 = expected_sha256
        self.expected_md5 = expected_md5
        if not os.path.lexists(path):
            raise RuntimeError(f"{label} is absent")
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
        self.path = path
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
        if self.closed:
            raise RuntimeError(f"held {self.label} descriptor is closed")
        hasher = hashlib.new(algorithm)
        offset = 0
        while offset < self.expected_size:
            chunk = os.pread(
                self.descriptor,
                min(8 * 1024 * 1024, self.expected_size - offset),
                offset,
            )
            if not chunk:
                raise RuntimeError(f"unexpected EOF from held {self.label} descriptor")
            hasher.update(chunk)
            offset += len(chunk)
        if os.pread(self.descriptor, 1, self.expected_size) != b"":
            raise RuntimeError(f"held {self.label} descriptor grew during hashing")
        return hasher.hexdigest()

    def validate_identity_and_hashes(self) -> None:
        if self.closed:
            raise RuntimeError(f"held {self.label} descriptor is closed")
        descriptor_token = self._stat_token(os.fstat(self.descriptor))
        try:
            path_details = self.path.lstat()
        except FileNotFoundError as error:
            raise RuntimeError(
                f"{self.label} pathname disappeared while descriptor was held"
            ) from error
        if (
            stat.S_ISLNK(path_details.st_mode)
            or not stat.S_ISREG(path_details.st_mode)
            or descriptor_token != self._token
            or self._stat_token(path_details) != self._token
        ):
            raise RuntimeError(f"{self.label} descriptor/path identity changed")
        if self.digest("sha256") != self.expected_sha256:
            raise RuntimeError(f"held {self.label} SHA256 drift")
        if self.expected_md5 is not None and self.digest("md5") != self.expected_md5:
            raise RuntimeError(f"held {self.label} MD5 drift")

    def close(self) -> None:
        if not self.closed:
            os.close(self.descriptor)
            self.closed = True

    def __del__(self) -> None:  # pragma: no cover - defensive process teardown
        self.close()


def _require_authorized() -> None:
    if not EXECUTION_AUTHORIZED:
        raise ExecutionLocked(
            "P0001 production GPU execution is locked pending separate exact authorization"
        )
    if len(EXECUTION_AUTH_SHA256) != 64 or any(
        character not in "0123456789abcdef" for character in EXECUTION_AUTH_SHA256
    ):
        raise ExecutionLocked("P0001 execution authorization SHA256 is not pinned")


def digest(path: str | Path, algorithm: str = "sha256") -> str:
    hasher = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git(*arguments: str, cwd: Path = ROOT) -> str:
    return subprocess.run(
        ("git", *arguments), cwd=cwd, check=True, text=True, capture_output=True
    ).stdout.strip()


def _tensor_sha256(tensor: object) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _validate_repository(commit: str) -> None:
    if len(commit) != 40 or git("rev-parse", "HEAD") != commit:
        raise RuntimeError("source commit drift")
    status_output = subprocess.run(
        ("git", "status", "--short"),
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    status = set(filter(None, status_output.splitlines()))
    if status - ALLOWED_STATUS:
        raise RuntimeError(f"unexpected Git status {status - ALLOWED_STATUS}")
    for relative in BOUND:
        committed = subprocess.run(
            ("git", "show", f"HEAD:{relative.as_posix()}"),
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if (ROOT / relative).read_bytes() != committed:
            raise RuntimeError(f"bound source drift: {relative}")


def _stage_inputs_without_chain(event: object) -> dict[str, str]:
    return {
        label: value
        for label, value in event.input_hashes
        if label
        not in {"previous_record", "previous_success_record", "prior_failure"}
    }


def _validate_prior_recovery_events(
    events: tuple[object, ...], expected_prefix: tuple[object, ...], patient_stage: object
) -> str:
    """Accept only five pre-authorized stages, optionally plus this GPU authorization."""

    if len(events) not in {10, 12}:
        raise RuntimeError(
            "exact 10-event prior-stage recovery ledger is required; supplemental "
            "append-only bootstrap authorization has not been recorded"
        )
    for actual, expected in zip(events[:10], expected_prefix, strict=True):
        if (
            actual.sequence != expected.sequence
            or actual.identity != expected.identity
            or actual.stage is not expected.stage
            or actual.event_type is not expected.event_type
            or actual.attempt_number != 1
            or actual.decision is not expected.decision
            or actual.authorization_sha256 != expected.authorization_sha256
            or actual.retry_authorization_sha256 is not None
            or actual.source_policy_hashes != expected.source_policy_hashes
            or actual.output_hashes != expected.output_hashes
            or _stage_inputs_without_chain(actual)
            != _stage_inputs_without_chain(expected)
        ):
            raise RuntimeError("prior-stage recovery ledger semantic drift")
    if len(events) == 12:
        started, succeeded = events[10:]
        if (
            started.stage is not patient_stage.GPU_AUTHORIZED
            or succeeded.stage is not patient_stage.GPU_AUTHORIZED
            or started.event_type.value != "STAGE_STARTED"
            or succeeded.event_type.value != "STAGE_SUCCEEDED"
            or started.authorization_sha256 != EXECUTION_AUTH_SHA256
            or succeeded.authorization_sha256 != EXECUTION_AUTH_SHA256
            or dict(succeeded.output_hashes)
            != {"gpu_authorization_record": EXECUTION_AUTH_SHA256}
        ):
            raise RuntimeError("existing GPU_AUTHORIZED recovery events drift")
        return "GPU_AUTHORIZATION_ALREADY_RECORDED"
    return "GPU_AUTHORIZATION_APPEND_REQUIRED"


def _execute(commit: str) -> dict[str, object]:
    import h5py
    import numpy as np
    import openslide
    import torch
    import torchvision
    import yaml

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
    from multiscale_feature_pilot.src.brca_coordinate_artifacts import (
        validate_brca_coordinate_artifacts,
    )
    from multiscale_feature_pilot.src.brca_omic import (
        BRCA_RELEASE_ARCHIVE_SHA256,
        load_official_brca_patient_omics,
    )
    from multiscale_feature_pilot.src.brca_p0001_feature_package import (
        TRANSACTION_ID,
        rehearse_p0001_feature_transaction,
    )
    from multiscale_feature_pilot.src.brca_singleton_streaming_policy import (
        PatientStage,
    )
    from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
        COMPACT_FILES,
        CompactArtifactValidationEvidence,
        ValidatedStageOutcome,
        append_control_event,
        load_frozen_cohort_order,
        plan_bound_stage,
        transaction_identity,
        validated_success_event,
    )
    from multiscale_feature_pilot.src.brca_streaming_executor_v2 import (
        start_event_from_plan,
    )
    from multiscale_feature_pilot.src.brca_streaming_recovery_v2 import (
        EventType,
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
        """Count and validate every authorized patch read from one held WSI."""

        def __init__(
            self,
            stable_path: str,
            expected_coordinates: object,
            expected_level: int,
            expected_size: int,
        ) -> None:
            self._stable_path = stable_path
            self._coordinates = expected_coordinates
            self._expected_level = expected_level
            self._expected_size = expected_size
            self._slide: object | None = None
            self.open_count = 0
            self.read_count = 0
            self.closed = False

        def __call__(self, path: str) -> "CountedSlide":
            if path != self._stable_path or self._slide is not None:
                raise RuntimeError("counted OpenSlide path/open contract")
            self._slide = openslide.OpenSlide(path)
            self.open_count += 1
            return self

        def read_region(
            self, location: tuple[int, int], level: int, size: tuple[int, int]
        ) -> object:
            if self._slide is None or self.closed:
                raise RuntimeError("counted OpenSlide is not open")
            if self.read_count >= len(self._coordinates):
                raise RuntimeError("authorized patch-read count exceeded")
            expected_location = tuple(
                int(value) for value in self._coordinates[self.read_count].tolist()
            )
            if (
                location != expected_location
                or level != self._expected_level
                or size != (self._expected_size, self._expected_size)
            ):
                raise RuntimeError("patch read tuple/order drift")
            self.read_count += 1
            return self._slide.read_region(location, level, size)

        def close(self) -> None:
            if self._slide is not None and not self.closed:
                self._slide.close()
                self.closed = True

    def extract_exact_branch(
        *,
        held_wsi: _HeldFileDescriptor,
        branch_name: str,
        branch_coordinates: object,
        patch_level: int,
        patch_size: int,
        expected_source_size: int,
        model: object,
        device: object,
    ) -> tuple[object, int, int]:
        counted = CountedSlide(
            held_wsi.stable_path,
            branch_coordinates,
            patch_level,
            expected_source_size,
        )
        dataset = StreamingOpenSlideDataset(
            WSI,
            PatchBranchSpec(
                branch_name, branch_coordinates, patch_level, patch_size
            ),
            slide_factory=counted,
        )
        # StreamingOpenSlideDataset resolves paths; replace that value only with
        # the already-held descriptor path so pixels cannot be reopened by name.
        dataset.wsi_path = held_wsi.stable_path
        try:
            result = extract_feature_matrix(
                dataset,
                model,
                device=device,
                batch_size=32,
                num_workers=0,
            )
        finally:
            dataset.close()
        if (
            counted.open_count != 1
            or counted.read_count != len(branch_coordinates)
            or not counted.closed
        ):
            raise RuntimeError(f"{branch_name} OpenSlide lifecycle/count drift")
        return result, counted.open_count, counted.read_count

    _validate_repository(commit)
    if digest(AUTH) != EXECUTION_AUTH_SHA256:
        raise RuntimeError("execution authorization drift")
    authorization = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    if (
        authorization.get("status") != "P0001_PRODUCTION_GPU_FEATURE_EXECUTION_AUTHORIZED"
        or authorization.get("executable") is not True
        or authorization["authorization"]["source"] != "DIRECT_USER_AUTHORIZATION"
        or hashlib.sha256(
            authorization["authorization"]["exact_statement"].encode()
        ).hexdigest()
        != AUTHORIZATION_STATEMENT_SHA256
        or authorization["authorization"]["exact_statement_sha256"]
        != AUTHORIZATION_STATEMENT_SHA256
        or authorization["scope"]["combined_shape"] != [TOTAL, 2048]
        or authorization["scope"]["natural_healnet_wsi_shape"] != [1, TOTAL, 2048]
        or authorization["scope"]["scale_2x_patch_reads"] != COUNTS[0]
        or authorization["scope"]["scale_4x_patch_reads"] != COUNTS[1]
        or authorization["scope"]["total_patch_reads"] != TOTAL
        or authorization["scope"]["coordinate_manifest_sha256"]
        != COORDINATE_MANIFEST_SHA256
        or authorization["scope"]["checkpoint_sha256"] != CHECKPOINT_SHA256
        or authorization["scope"]["concatenation_order"] != ["scale_2x", "scale_4x"]
        or authorization["scope"]["row_ranges"]
        != {"scale_2x": [0, COUNTS[0]], "scale_4x": [COUNTS[0], TOTAL]}
        or authorization["scope"]["dtype"] != "float32"
        or authorization["scope"]["pooling_performed"] is not False
        or authorization["scope"]["transpose_performed"] is not False
        or authorization["scope"]["compact_publication"]
        != {"required": True, "atomic_no_overwrite": True, "exact_file_count": 4}
        or authorization["scope"]["recovery_v2"]
        != {
            "required": True,
            "authorized_stage_successes": ["GPU_AUTHORIZED", "FEATURES_VERIFIED"],
            "required_preexisting_prefix": {
                "exact_events": 10,
                "terminal_stage": "COORDINATES_VERIFIED",
                "bootstrap_authorized_by_this_statement": False,
            },
        }
        or authorization["permitted_cleanup"]
        != {
            "runner_created_ephemeral_recovery_staging_only": True,
            "preexisting_or_final_paths": False,
        }
        or not all(authorization["prohibited"].values())
    ):
        raise RuntimeError("execution authorization semantics")
    preexecution = yaml.safe_load(PREEXECUTION.read_text(encoding="utf-8"))
    if (
        preexecution["verified_inputs"]["coordinate_manifest_sha256"]
        != COORDINATE_MANIFEST_SHA256
        or preexecution["verified_inputs"]["combined_shape"] != [TOTAL, 2048]
    ):
        raise RuntimeError("preexecution contract drift")
    if os.path.lexists(OUTPUT):
        raise ManualRecoveryRequired(
            "P0001 feature output already exists; validate the published compact "
            "artifact and ledger tip manually without overwrite or deletion"
        )
    if list(OUTPUT.parent.glob(".BRCA_PRODUCTION_P0001.features.staging.*")):
        raise ManualRecoveryRequired(
            "P0001 feature staging exists and requires manual no-delete review"
        )
    if shutil.disk_usage(OUTPUT.parent).free < 278_000_000:
        raise RuntimeError("insufficient free disk for P0001 compact publication")
    if git("rev-parse", "HEAD", cwd=OFFICIAL) != OFFICIAL_HEAD or git(
        "status", "--porcelain", cwd=OFFICIAL
    ):
        raise RuntimeError("official HEALNet drift")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG required before CUDA")
    expected_prior = rehearse_p0001_feature_transaction(
        ALIGNMENT, through_stage=PatientStage.COORDINATES_VERIFIED
    ).events
    prior_events = load_events(LEDGER)
    prior_state = _validate_prior_recovery_events(
        prior_events, expected_prior, PatientStage
    )
    if replay_events(prior_events).action is not ReplayAction.ADVANCE_STAGE:
        raise RuntimeError("prior recovery ledger is not ready to advance")
    coordinate_result = (
        ROOT
        / "multiscale_feature_pilot/provenance/brca_p0001_coordinate_execution_result.yaml"
    )
    if digest(coordinate_result) != COORDINATE_RESULT_SHA256:
        raise RuntimeError("coordinate result drift")
    coordinate_record = validate_brca_coordinate_artifacts(
        COORD, expected_manifest_sha256=COORDINATE_MANIFEST_SHA256
    )
    coordinates = []
    for name, count in zip(("scale_2x", "scale_4x"), COUNTS, strict=True):
        if coordinate_record.branch_for(name).coordinate_count != count:
            raise RuntimeError("coordinate count drift")
        with h5py.File(COORD / f"{name}_coordinates.h5", "r") as source:
            coordinates.append(
                torch.from_numpy(np.asarray(source["coords"], dtype=np.int64)).contiguous()
            )
    held_omic = _HeldFileDescriptor(
        OMIC,
        expected_size=OMIC_SIZE,
        expected_sha256=BRCA_RELEASE_ARCHIVE_SHA256,
        label="BRCA Omic archive",
    )
    try:
        held_omic.validate_identity_and_hashes()
        omic = load_official_brca_patient_omics(
            held_omic.stable_path, case_id=PATIENT, slide_id=SLIDE
        )
        held_omic.validate_identity_and_hashes()
    finally:
        held_omic.close()
    if (
        omic.source_row_index != "4"
        or omic.case_id != PATIENT
        or omic.slide_id != SLIDE
        or any(
            _tensor_sha256(getattr(omic, name)) != expected
            for name, expected in OMIC_CONTENT_SHA256.items()
        )
    ):
        raise RuntimeError("Omic identity or content drift")

    recovery_events = list(prior_events)
    binding = load_frozen_cohort_order(ALIGNMENT)[0]
    identity = transaction_identity(binding, TRANSACTION_ID)
    if prior_state == "GPU_AUTHORIZATION_APPEND_REQUIRED":
        gpu_plan = plan_bound_stage(
            recovery_events,
            binding=binding,
            identity=identity,
            run_id=str(uuid.uuid4()),
            authorization_sha256=EXECUTION_AUTH_SHA256,
            retry_authorization_sha256=None,
            stage_input_hashes={
                "checkpoint_identity": CHECKPOINT_SHA256,
                "coordinate_manifest": COORDINATE_MANIFEST_SHA256,
            },
        )
        gpu_started = start_event_from_plan(
            recovery_events,
            gpu_plan,
            recorded_at_utc=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        recovery_events.append(gpu_started)
        gpu_succeeded = validated_success_event(
            recovery_events,
            gpu_plan,
            ValidatedStageOutcome(
                stage=PatientStage.GPU_AUTHORIZED,
                authorization_sha256=EXECUTION_AUTH_SHA256,
                output_hashes=(("gpu_authorization_record", EXECUTION_AUTH_SHA256),),
                validation_record_sha256=EXECUTION_AUTH_SHA256,
                compact_artifact=None,
            ),
            existing_output_hashes=None,
            recorded_at_utc=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        recovery_events.append(gpu_succeeded)
        try:
            append_control_event(LEDGER, gpu_started)
            append_control_event(LEDGER, gpu_succeeded)
        except Exception as error:
            raise ManualRecoveryRequired(
                "GPU_AUTHORIZED append was interrupted; inspect the append-only "
                "ledger tip and runner-created recovery staging before retry"
            ) from error

    held_wsi = _HeldFileDescriptor(
        WSI,
        expected_size=WSI_SIZE,
        expected_sha256=WSI_SHA256,
        expected_md5=WSI_MD5,
        label="P0001 WSI",
    )
    held_checkpoint = _HeldFileDescriptor(
        CHECKPOINT,
        expected_size=CHECKPOINT_SIZE,
        expected_sha256=CHECKPOINT_SHA256,
        label="ResNet50 checkpoint",
    )
    try:
        held_wsi.validate_identity_and_hashes()
        held_checkpoint.validate_identity_and_hashes()
        with openslide.OpenSlide(held_wsi.stable_path) as slide:
            if tuple(slide.level_dimensions) != DIMS or tuple(
                float(value) for value in slide.level_downsamples
            ) != DOWNSAMPLES:
                raise RuntimeError("WSI header drift")

        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("exactly one CUDA GPU required; CPU fallback forbidden")
        if "Tesla T4" not in torch.cuda.get_device_name(
            0
        ) or torch.cuda.get_device_capability(0) != (7, 5):
            raise RuntimeError("Tesla T4 capability 7.5 required")
        device = torch.device("cuda:0")
        persisted_before_features = list(load_events(LEDGER))
        if (
            len(persisted_before_features) != 12
            or replay_events(persisted_before_features).action
            is not ReplayAction.ADVANCE_STAGE
        ):
            raise ManualRecoveryRequired(
                "GPU_AUTHORIZED ledger tip drift before feature execution"
            )
        feature_plan = plan_bound_stage(
            persisted_before_features,
            binding=binding,
            identity=identity,
            run_id=str(uuid.uuid4()),
            authorization_sha256=EXECUTION_AUTH_SHA256,
            retry_authorization_sha256=None,
            stage_input_hashes={
                "checkpoint_identity": CHECKPOINT_SHA256,
                "coordinate_manifest": COORDINATE_MANIFEST_SHA256,
                "gpu_authorization_record": EXECUTION_AUTH_SHA256,
            },
        )
        feature_started = start_event_from_plan(
            persisted_before_features,
            feature_plan,
            recorded_at_utc=datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
        )
        try:
            append_control_event(LEDGER, feature_started)
        except Exception as error:
            raise ManualRecoveryRequired(
                "FEATURES_VERIFIED start append was interrupted; inspect the "
                "append-only ledger tip and runner-created recovery staging before retry"
            ) from error
        started = datetime.now(timezone.utc)
        start = time.perf_counter()
        synthetic = run_one_patient_supervisor_healnet_smoke(
            official_repo=OFFICIAL,
            wsi=torch.zeros((1, TOTAL, 2048), device=device),
            rna=torch.zeros((1, 1, 1558), device=device),
            mutation=torch.zeros((1, 1, 21), device=device),
            cnv=torch.zeros((1, 1, 1333), device=device),
        )
        state = torch.load(
            held_checkpoint.stable_path, map_location="cpu", weights_only=True
        )
        if not isinstance(state, dict):
            raise RuntimeError("ResNet50 checkpoint state-dict contract")
        model = torchvision.models.resnet50(weights=None)
        model.load_state_dict(state, strict=True)
        model.fc = torch.nn.Identity()
        model = model.to(device).eval()
        held_checkpoint.validate_identity_and_hashes()
        result_2x, opens_2x, reads_2x = extract_exact_branch(
            held_wsi=held_wsi,
            branch_name="scale_2x",
            branch_coordinates=coordinates[0],
            patch_level=0,
            patch_size=256,
            expected_source_size=512,
            model=model,
            device=device,
        )
        result_4x, opens_4x, reads_4x = extract_exact_branch(
            held_wsi=held_wsi,
            branch_name="scale_4x",
            branch_coordinates=coordinates[1],
            patch_level=1,
            patch_size=256,
            expected_source_size=256,
            model=model,
            device=device,
        )
        held_wsi.validate_identity_and_hashes()
    finally:
        held_checkpoint.close()
        held_wsi.close()
    combined = torch.cat((result_2x.features, result_4x.features), dim=0).contiguous()
    if tuple(combined.shape) != (TOTAL, 2048):
        raise RuntimeError("combined shape contract")
    if (
        combined.dtype != torch.float32
        or combined.requires_grad
        or not bool(torch.isfinite(combined).all())
    ):
        raise RuntimeError("combined numerical contract")
    real = run_one_patient_supervisor_healnet_smoke(
        official_repo=OFFICIAL,
        wsi=combined.unsqueeze(0).to(device),
        rna=omic.rna.to(device),
        mutation=omic.mutation.to(device),
        cnv=omic.cnv.to(device),
    )
    _validate_repository(commit)
    rows = build_two_scale_provenance(
        scale_2x=BranchProvenanceSpec("scale_2x", coordinates[0], 0, 0.5, 0.5),
        scale_4x=BranchProvenanceSpec("scale_4x", coordinates[1], 1, 1.0, 1.0),
        scale_2x_count=COUNTS[0],
        scale_4x_count=COUNTS[1],
    )
    metadata = CompactFeatureMetadata(
        PATIENT,
        SLIDE,
        UUID,
        WSI_SHA256,
        COORDINATE_MANIFEST_SHA256,
        BRCA_RELEASE_ARCHIVE_SHA256,
        CHECKPOINT_SHA256,
        POLICY_SHA256,
        commit,
        COUNTS[0],
        COUNTS[1],
    )
    manifest = publish_compact_feature_artifacts(
        OUTPUT,
        combined_features=combined,
        row_provenance=rows,
        metadata=metadata,
        preserve_failed_staging=True,
    )
    anchor = digest(OUTPUT / MANIFEST_FILENAME)
    validate_compact_feature_artifacts(OUTPUT, expected_manifest_sha256=anchor)
    exact_names = (
        COMBINED_FILENAME,
        PROVENANCE_FILENAME,
        MANIFEST_FILENAME,
        SIDECAR_FILENAME,
    )
    file_hashes = tuple((name, digest(OUTPUT / name)) for name in exact_names)
    persisted = list(load_events(LEDGER))
    if (
        len(persisted) != 13
        or persisted[-1].record_sha256 != feature_started.record_sha256
    ):
        raise ManualRecoveryRequired(
            "FEATURES_VERIFIED started ledger record drift before success append"
        )
    compact_evidence = CompactArtifactValidationEvidence(
        patient_id=PATIENT,
        slide_id=SLIDE,
        gdc_uuid=UUID,
        omic_source_row_id="4",
        bound_source_policy_hashes=feature_plan.source_policy_hashes,
        bound_input_hashes=feature_plan.input_hashes,
        exact_files=COMPACT_FILES,
        manifest_sha256=anchor,
        sidecar_manifest_sha256=anchor,
        file_hashes=file_hashes,
        tensor_shape=(TOTAL, 2048),
        tensor_dtype="float32",
        tensor_device="cpu",
        tensor_contiguous=True,
        tensor_finite=True,
        tensor_requires_grad=False,
        scale_2x_row_range=(0, COUNTS[0]),
        scale_4x_row_range=(COUNTS[0], TOTAL),
        row_provenance_count=TOTAL,
    )
    feature_succeeded = validated_success_event(
        persisted,
        feature_plan,
        ValidatedStageOutcome(
            stage=PatientStage.FEATURES_VERIFIED,
            authorization_sha256=EXECUTION_AUTH_SHA256,
            output_hashes=(("compact_feature_manifest", anchor),),
            validation_record_sha256=anchor,
            compact_artifact=compact_evidence,
        ),
        existing_output_hashes=None,
        recorded_at_utc=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )
    try:
        append_control_event(LEDGER, feature_succeeded)
    except Exception as error:
        raise ManualRecoveryRequired(
            "verified feature output exists but FEATURES_VERIFIED ledger append was "
            "interrupted; validate output and ledger manually without deletion"
        ) from error
    final_recovery = replay_events(load_events(LEDGER))
    if final_recovery.action is not ReplayAction.ADVANCE_STAGE:
        raise RuntimeError("FEATURES_VERIFIED recovery tip validation failed")
    return {
        "status": "BRCA_PRODUCTION_P0001_GPU_FEATURE_SUCCESS",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": time.perf_counter() - start,
        "source_commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "counts": {"scale_2x": COUNTS[0], "scale_4x": COUNTS[1], "total": TOTAL},
        "feature_shapes": {
            "scale_2x": list(result_2x.features.shape),
            "scale_4x": list(result_4x.features.shape),
            "combined": list(combined.shape),
            "natural": [1, TOTAL, 2048],
        },
        "timings": {
            "scale_2x": result_2x.streaming_extraction_seconds,
            "scale_4x": result_4x.streaming_extraction_seconds,
        },
        "patch_io": {
            "openslide_patch_opens": opens_2x + opens_4x,
            "scale_2x_read_region_calls": reads_2x,
            "scale_4x_read_region_calls": reads_4x,
            "total_read_region_calls": reads_2x + reads_4x,
            "num_workers": 0,
            "held_o_nofollow_descriptor": True,
            "stable_proc_fd_path": True,
        },
        "peak_gpu_memory_bytes": max(
            result_2x.peak_gpu_memory_bytes, result_4x.peak_gpu_memory_bytes
        ),
        "synthetic_healnet": asdict(synthetic),
        "real_healnet": asdict(real),
        "artifact": {
            "directory": str(OUTPUT),
            "manifest_sha256": anchor,
            "manifest": manifest,
            "total_file_bytes": sum(path.stat().st_size for path in OUTPUT.iterdir()),
        },
        "recovery": {
            "ledger": str(LEDGER),
            "events": len(load_events(LEDGER)),
            "final_action": final_recovery.action.value,
            "last_stage": PatientStage.FEATURES_VERIFIED.value,
        },
        "operations": {
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "amp": 0,
            "tf32": 0,
            "coordinate_generation": 0,
            "other_patients": 0,
            "raw_or_final_deletions": 0,
        },
    }


def run(commit: str) -> dict[str, object]:
    _require_authorized()
    return _execute(commit)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-source-commit", required=True)
    arguments = parser.parse_args()
    try:
        result = run(arguments.expected_source_commit)
    except Exception as error:
        print(
            json.dumps(
                {"status": "BLOCKED", "error": f"{type(error).__name__}: {error}"},
                sort_keys=True,
            )
        )
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
