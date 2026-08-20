#!/usr/bin/env python3
"""Execution-locked P0001 production GPU feature package."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE = ROOT.parent
EXECUTION_AUTHORIZED = False
EXECUTION_AUTH_SHA256 = "PENDING_SEPARATE_EXACT_P0001_GPU_AUTHORIZATION"
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
    "M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}


class ExecutionLocked(RuntimeError):
    """Raised before imports, path checks, pixels, CUDA, or publication."""


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
    status = set(filter(None, git("status", "--short").splitlines()))
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
        rehearse_p0001_feature_transaction,
    )
    from multiscale_feature_pilot.src.brca_singleton_streaming_policy import (
        PatientStage,
    )
    from multiscale_feature_pilot.src.brca_streaming_production_adapter import (
        append_control_event,
    )
    from multiscale_feature_pilot.src.feature_extraction import (
        PatchBranchSpec,
        StreamingOpenSlideDataset,
        build_resnet50_imagenet1k_v2,
        extract_feature_matrix,
    )
    from multiscale_feature_pilot.src.provenance import (
        BranchProvenanceSpec,
        build_two_scale_provenance,
    )
    from multiscale_feature_pilot.src.supervisor_healnet_smoke import (
        run_one_patient_supervisor_healnet_smoke,
    )

    _validate_repository(commit)
    if digest(AUTH) != EXECUTION_AUTH_SHA256:
        raise RuntimeError("execution authorization drift")
    authorization = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    if (
        authorization.get("status") != "P0001_PRODUCTION_GPU_FEATURE_EXECUTION_AUTHORIZED"
        or hashlib.sha256(authorization["authorization_statement"].encode()).hexdigest()
        != AUTHORIZATION_STATEMENT_SHA256
        or authorization["authorization_statement_sha256"]
        != AUTHORIZATION_STATEMENT_SHA256
        or authorization["scope"]["combined_shape"] != [TOTAL, 2048]
        or authorization["scope"]["scale_2x_patch_reads"] != COUNTS[0]
        or authorization["scope"]["scale_4x_patch_reads"] != COUNTS[1]
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
    if any(
        os.path.lexists(path)
        for path in (OUTPUT, LEDGER)
    ) or list(OUTPUT.parent.glob(".BRCA_PRODUCTION_P0001.features.staging.*")):
        raise RuntimeError("P0001 feature output, ledger, or staging collision")
    if git("rev-parse", "HEAD", cwd=OFFICIAL) != OFFICIAL_HEAD or git(
        "status", "--porcelain", cwd=OFFICIAL
    ):
        raise RuntimeError("official HEALNet drift")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG required before CUDA")
    if (
        WSI.is_symlink()
        or not WSI.is_file()
        or WSI.stat().st_size != WSI_SIZE
        or digest(WSI, "md5") != WSI_MD5
        or digest(WSI) != WSI_SHA256
    ):
        raise RuntimeError("WSI identity drift")
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
    omic = load_official_brca_patient_omics(OMIC, case_id=PATIENT, slide_id=SLIDE)
    if (
        omic.source_row_index != "4"
        or omic.case_id != PATIENT
        or omic.slide_id != SLIDE
        or digest(OMIC) != BRCA_RELEASE_ARCHIVE_SHA256
        or any(
            _tensor_sha256(getattr(omic, name)) != expected
            for name, expected in OMIC_CONTENT_SHA256.items()
        )
    ):
        raise RuntimeError("Omic identity or content drift")
    if (
        CHECKPOINT.is_symlink()
        or not CHECKPOINT.is_file()
        or CHECKPOINT.stat().st_size != 102_540_417
        or digest(CHECKPOINT) != CHECKPOINT_SHA256
    ):
        raise RuntimeError("checkpoint drift")
    with openslide.OpenSlide(str(WSI)) as slide:
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
    if "Tesla T4" not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (
        7,
        5,
    ):
        raise RuntimeError("Tesla T4 capability 7.5 required")
    device = torch.device("cuda:0")
    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    synthetic = run_one_patient_supervisor_healnet_smoke(
        official_repo=OFFICIAL,
        wsi=torch.zeros((1, TOTAL, 2048), device=device),
        rna=torch.zeros((1, 1, 1558), device=device),
        mutation=torch.zeros((1, 1, 21), device=device),
        cnv=torch.zeros((1, 1, 1333), device=device),
    )
    model = build_resnet50_imagenet1k_v2(CHECKPOINT).to(device).eval()
    result_2x = extract_feature_matrix(
        StreamingOpenSlideDataset(
            WSI, PatchBranchSpec("scale_2x", coordinates[0], 0, 256)
        ),
        model,
        device=device,
        batch_size=32,
        num_workers=2,
    )
    result_4x = extract_feature_matrix(
        StreamingOpenSlideDataset(
            WSI, PatchBranchSpec("scale_4x", coordinates[1], 1, 256)
        ),
        model,
        device=device,
        batch_size=32,
        num_workers=2,
    )
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
    if digest(WSI, "md5") != WSI_MD5 or digest(WSI) != WSI_SHA256:
        raise RuntimeError("post-extraction WSI identity drift")
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
    ledger_recorded_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    recovery = rehearse_p0001_feature_transaction(
        ALIGNMENT,
        gpu_authorization_sha256=EXECUTION_AUTH_SHA256,
        compact_manifest_sha256=anchor,
        compact_file_hashes=file_hashes,
        through_stage=PatientStage.FEATURES_VERIFIED,
        event_timestamps=(ledger_recorded_at,) * 16,
    )
    for event in recovery.events:
        append_control_event(LEDGER, event)
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
            "events": len(recovery.events),
            "final_action": recovery.final_action.value,
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
