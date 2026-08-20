#!/usr/bin/env python3
"""Execution-locked, one-patient B03 GPU feature pilot."""

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
EXECUTION_AUTHORIZED = True
EXECUTION_AUTH_SHA256 = "30e440b93c202ec1d445312ef27dd21b2bc8a6678913f9ecc264581d83be2fac"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_b03_gpu_execution_authorization.yaml"
PATIENT = "TCGA-AR-A1AY"
SLIDE = "TCGA-AR-A1AY-01Z-00-DX1.6AC0BE3B-FFC5-4EDA-9E40-B18CAAC52B81.svs"
UUID = "266c3852-f5d0-4815-94d9-dac5b0ff8276"
WSI = (
    Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B03.incoming")
    / UUID
    / SLIDE
)
COORD = Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B03.coordinates")
OUTPUT = Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B03.features")
OMIC = Path(
    "/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/"
    "data/tcga/omic/tcga_brca_all_clean.csv.zip"
)
CHECKPOINT = Path("/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")
OFFICIAL = Path("/teamspace/studios/this_studio/healnet")
WSI_SIZE = 918454431
WSI_MD5 = "1980b183d5bb948c2fc263af62a4b1b4"
WSI_SHA = "4ef4ac79ce3cc0bfc5a4ea62985f080f0f877dca4c7e43191a04a35b2eba8228"
COORD_SHA = "cced78f863415ee12d57b905131e60758ba849645196c0a14309e9ab8d4e1ae5"
POLICY_SHA = "58de5cedb887a3d0faf12c68e8629d26cae4cf7ff8a867e20742d395890a522c"
CHECKPOINT_SHA = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
OFFICIAL_HEAD = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
DIMS = ((93296, 58121), (23324, 14530), (5831, 3632), (2915, 1816))
DOWNSAMPLES = (1.0, 4.000034411562285, 16.00123898678414, 32.00522239895422)
COUNTS = (8875, 2257)
TOTAL = 11132
OMIC_CONTENT_SHA256 = {
    "rna": "463b056072b280c951cf2626f6ca80e5d30014133c486f8d51423695f9869dfa",
    "mutation": "7a67b8d094d93c2f5998b5384fdc07f07b46acc587e059aea33c60b7e9bd3afd",
    "cnv": "21fdf23ebbe236f295337f497620bd522df309a81b071395a09b22800c7285b3",
}
BOUND = (
    Path("scripts/run_brca_b03_gpu_pilot.py"),
    Path("multiscale_feature_pilot/config/brca_b03_gpu_preexecution.yaml"),
    Path("multiscale_feature_pilot/provenance/brca_b03_coordinate_execution_result.yaml"),
    Path("multiscale_feature_pilot/config/brca_b03_scale_coordinate_policy.yaml"),
    Path("multiscale_feature_pilot/src/brca_compact_feature_artifacts.py"),
    Path("multiscale_feature_pilot/src/brca_coordinate_artifacts.py"),
    Path("multiscale_feature_pilot/src/brca_omic.py"),
    Path("multiscale_feature_pilot/src/feature_extraction.py"),
    Path("multiscale_feature_pilot/src/provenance.py"),
    Path("multiscale_feature_pilot/src/supervisor_healnet_smoke.py"),
    Path("multiscale_feature_pilot/__init__.py"),
    Path("multiscale_feature_pilot/src/__init__.py"),
    Path("multiscale_feature_pilot/src/omic.py"),
    Path("multiscale_feature_pilot/src/padding.py"),
)
ALLOWED_STATUS = {
    "M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}


class ExecutionLocked(RuntimeError):
    """Raised before any operational work while GPU execution is unauthorized."""


def _require_authorized() -> None:
    if not EXECUTION_AUTHORIZED:
        raise ExecutionLocked("B03 GPU execution is locked pending separate exact authorization")
    if len(EXECUTION_AUTH_SHA256) != 64 or any(
        character not in "0123456789abcdef" for character in EXECUTION_AUTH_SHA256
    ):
        raise ExecutionLocked("B03 execution authorization SHA256 is not pinned")


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


def _execute(commit: str) -> dict[str, object]:
    import h5py
    import numpy as np
    import openslide
    import torch
    import torchvision
    import yaml

    sys.path.insert(0, str(ROOT))
    from multiscale_feature_pilot.src.brca_compact_feature_artifacts import (
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

    if git("rev-parse", "HEAD") != commit:
        raise RuntimeError("HEAD drift")
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
            raise RuntimeError(f"bound source drift {relative}")
    if digest(AUTH) != EXECUTION_AUTH_SHA256:
        raise RuntimeError("authorization drift")
    authorization = yaml.safe_load(AUTH.read_text(encoding="utf-8"))
    if (
        authorization.get("status") != "B03_GPU_FEATURE_PILOT_AUTHORIZED"
        or authorization["scope"]["combined_shape"] != [11132, 2048]
        or authorization["scope"]["scale_2x_patch_reads"] != 8875
        or authorization["scope"]["scale_4x_patch_reads"] != 2257
    ):
        raise RuntimeError("authorization semantics")
    if (
        OUTPUT.exists()
        or OUTPUT.is_symlink()
        or list(OUTPUT.parent.glob(".BRCA_BATCH_B03.features.staging.*"))
    ):
        raise RuntimeError("output collision")
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
        or digest(WSI) != WSI_SHA
    ):
        raise RuntimeError("WSI identity drift")
    coordinate_record = validate_brca_coordinate_artifacts(
        COORD, expected_manifest_sha256=COORD_SHA
    )
    coordinates = []
    for name, count in zip(("scale_2x", "scale_4x"), COUNTS, strict=True):
        if coordinate_record.branch_for(name).coordinate_count != count:
            raise RuntimeError("coordinate count drift")
        with h5py.File(COORD / f"{name}_coordinates.h5", "r") as h5:
            coordinates.append(
                torch.from_numpy(np.asarray(h5["coords"], dtype=np.int64)).contiguous()
            )
    omic = load_official_brca_patient_omics(OMIC, case_id=PATIENT, slide_id=SLIDE)
    if (
        omic.source_row_index != "372"
        or omic.case_id != PATIENT
        or omic.slide_id != SLIDE
        or digest(OMIC) != BRCA_RELEASE_ARCHIVE_SHA256
        or any(
            _tensor_sha256(getattr(omic, name)) != expected
            for name, expected in OMIC_CONTENT_SHA256.items()
        )
    ):
        raise RuntimeError("Omic drift")
    if (
        CHECKPOINT.is_symlink()
        or not CHECKPOINT.is_file()
        or CHECKPOINT.stat().st_size != 102540417
        or digest(CHECKPOINT) != CHECKPOINT_SHA
    ):
        raise RuntimeError("checkpoint drift")
    with openslide.OpenSlide(str(WSI)) as slide:
        if tuple(slide.level_dimensions) != DIMS or any(
            abs(actual - expected) > 1e-10
            for actual, expected in zip(slide.level_downsamples, DOWNSAMPLES, strict=True)
        ):
            raise RuntimeError("header drift")

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
    if tuple(combined.shape) != (11132, 2048):
        raise RuntimeError("combined shape contract")
    if combined.dtype != torch.float32 or not bool(torch.isfinite(combined).all()):
        raise RuntimeError("combined numerical contract")
    real = run_one_patient_supervisor_healnet_smoke(
        official_repo=OFFICIAL,
        wsi=combined.unsqueeze(0).to(device),
        rna=omic.rna.to(device),
        mutation=omic.mutation.to(device),
        cnv=omic.cnv.to(device),
    )
    rows = build_two_scale_provenance(
        scale_2x=BranchProvenanceSpec("scale_2x", coordinates[0], 0, 0.4936, 0.4936),
        scale_4x=BranchProvenanceSpec(
            "scale_4x", coordinates[1], 1, 0.9872084927735719, 0.9872084927735719
        ),
        scale_2x_count=8875,
        scale_4x_count=2257,
    )
    metadata = CompactFeatureMetadata(
        PATIENT,
        SLIDE,
        UUID,
        WSI_SHA,
        COORD_SHA,
        BRCA_RELEASE_ARCHIVE_SHA256,
        CHECKPOINT_SHA,
        POLICY_SHA,
        commit,
        8875,
        2257,
    )
    manifest = publish_compact_feature_artifacts(
        OUTPUT,
        combined_features=combined,
        row_provenance=rows,
        metadata=metadata,
        preserve_failed_staging=True,
    )
    anchor = digest(OUTPUT / "compact_manifest.json")
    validate_compact_feature_artifacts(OUTPUT, expected_manifest_sha256=anchor)
    return {
        "status": "BRCA_BATCH_B03_GPU_FEATURE_PILOT_SUCCESS",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_seconds": time.perf_counter() - start,
        "source_commit": commit,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "counts": {"scale_2x": 8875, "scale_4x": 2257, "total": TOTAL},
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
        "operations": {
            "training": 0,
            "backward": 0,
            "optimizer_steps": 0,
            "amp": 0,
            "tf32": 0,
            "coordinate_generation": 0,
            "deletions": 0,
            "other_patients": 0,
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
