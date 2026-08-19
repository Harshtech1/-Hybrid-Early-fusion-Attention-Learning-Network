#!/usr/bin/env python3
"""Fail-closed, one-shot B01 GPU feature pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import h5py
import numpy as np
import openslide
import torch
import torchvision
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from multiscale_feature_pilot.src.brca_compact_feature_artifacts import (  # noqa: E402
    CompactFeatureMetadata, publish_compact_feature_artifacts,
    validate_compact_feature_artifacts,
)
from multiscale_feature_pilot.src.brca_coordinate_artifacts import validate_brca_coordinate_artifacts  # noqa: E402
from multiscale_feature_pilot.src.brca_omic import BRCA_RELEASE_ARCHIVE_SHA256, load_official_brca_patient_omics  # noqa: E402
from multiscale_feature_pilot.src.feature_extraction import (  # noqa: E402
    PatchBranchSpec, StreamingOpenSlideDataset,
    build_resnet50_imagenet1k_v2, extract_feature_matrix,
)
from multiscale_feature_pilot.src.provenance import BranchProvenanceSpec, build_two_scale_provenance  # noqa: E402
from multiscale_feature_pilot.src.supervisor_healnet_smoke import run_one_patient_supervisor_healnet_smoke  # noqa: E402

PATIENT = "TCGA-GI-A2C8"
SLIDE = "TCGA-GI-A2C8-01Z-00-DX1.09BD8AC9-645A-4C8B-9B36-77D833BDBA09.svs"
UUID = "0a886f18-c44c-4b5e-b243-6df6e27f426a"
WSI = Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B01.incoming") / UUID / SLIDE
WSI_SIZE = 408_704_377
WSI_MD5 = "a9f830d456b4a1fe0e9bb5b5b99f4b7e"
WSI_SHA = "b3c27e220c5c3961600782af42e91a52ea0b85a710d8d4aa722831ea00f9ad5f"
COORD = Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B01.coordinates")
COORD_MANIFEST_SHA = "ff1eebc9fa51128cdda6294fa48cf34008b203db236acf59e8570725b3fc4c8b"
OUTPUT = Path("/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B01.features")
OMIC = Path("/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip")
CHECKPOINT = Path("/home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth")
CHECKPOINT_SHA = "11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca"
POLICY_SHA = "9a98a0edad90f1e82ad28025b9164c136c39041ad27fb5342251bdb8d062cb60"
AUTH = ROOT / "multiscale_feature_pilot/config/brca_b01_gpu_execution_authorization.yaml"
AUTH_SHA = "4afc085250cb969d9d63c6db60ace06ac834769f2c64c239e017cf6ff861f902"
OFFICIAL = Path("/teamspace/studios/this_studio/healnet")
OFFICIAL_HEAD = "28ba5da6ab99fd8069972c22e986d83edb658dd4"
EXPECTED_DIMS = ((63784, 39311), (15946, 9827), (3986, 2456))
EXPECTED_DS = (1.0, 4.00015264068383, 16.004057258221366)
COUNTS = (3773, 969)
FEATURE_DIM = 2048
ALLOWED_DIRTY = {
    " M reports/blca_one_patient_multiscale_pilot.md",
    " M reports/brca_compact_artifact_and_recovery_design.md",
    "?? reports/brca_supervisor_progress_report.html",
}

def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def git(*args: str, cwd: Path = ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, text=True, capture_output=True).stdout.strip()

def preflight(expected_commit: str) -> None:
    if git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("source HEAD mismatch")
    status = set(filter(None, git("status", "--short").splitlines()))
    if status - ALLOWED_DIRTY:
        raise RuntimeError(f"unexpected worktree changes: {sorted(status - ALLOWED_DIRTY)}")
    if digest(AUTH) != AUTH_SHA:
        raise RuntimeError("authorization hash mismatch")
    auth = yaml.safe_load(AUTH.read_text())
    if auth.get("status") != "B01_GPU_FEATURE_PILOT_AUTHORIZED" or auth["scope"]["combined_shape"] != [4742, 2048]:
        raise RuntimeError("authorization semantics mismatch")
    if OUTPUT.exists() or OUTPUT.is_symlink() or list(OUTPUT.parent.glob(".BRCA_BATCH_B01.features.staging.*")):
        raise RuntimeError("output or staging already exists")
    if git("rev-parse", "HEAD", cwd=OFFICIAL) != OFFICIAL_HEAD or git("status", "--porcelain", cwd=OFFICIAL):
        raise RuntimeError("official HEALNet is not exact and clean")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be :4096:8 before launch")

def validate_inputs() -> tuple[torch.Tensor, torch.Tensor, object]:
    if WSI.is_symlink() or not WSI.is_file() or WSI.stat().st_size != WSI_SIZE:
        raise RuntimeError("WSI file identity mismatch")
    if digest(WSI, "md5") != WSI_MD5 or digest(WSI) != WSI_SHA:
        raise RuntimeError("WSI digest mismatch")
    record = validate_brca_coordinate_artifacts(COORD, expected_manifest_sha256=COORD_MANIFEST_SHA)
    for name, count in zip(("scale_2x", "scale_4x"), COUNTS):
        if record.branch_for(name).coordinate_count != count:
            raise RuntimeError("coordinate count mismatch")
    coords = []
    for name in ("scale_2x", "scale_4x"):
        with h5py.File(COORD / f"{name}_coordinates.h5", "r") as h:
            coords.append(torch.from_numpy(np.asarray(h["coords"], dtype=np.int64)).contiguous())
    omic = load_official_brca_patient_omics(OMIC, case_id=PATIENT, slide_id=SLIDE)
    if omic.source_row_index != "924" or digest(OMIC) != BRCA_RELEASE_ARCHIVE_SHA256:
        raise RuntimeError("Omic identity mismatch")
    if CHECKPOINT.stat().st_size != 102_540_417 or digest(CHECKPOINT) != CHECKPOINT_SHA:
        raise RuntimeError("checkpoint mismatch")
    with openslide.OpenSlide(str(WSI)) as slide:
        if tuple(slide.level_dimensions) != EXPECTED_DIMS:
            raise RuntimeError("WSI level dimensions mismatch")
        if any(abs(a-b) > 1e-10 for a,b in zip(slide.level_downsamples, EXPECTED_DS)):
            raise RuntimeError("WSI downsample mismatch")
    return coords[0], coords[1], omic

def configure_cuda() -> torch.device:
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA GPU is required; CPU fallback forbidden")
    if "Tesla T4" not in torch.cuda.get_device_name(0) or torch.cuda.get_device_capability(0) != (7, 5):
        raise RuntimeError("Tesla T4 compute capability 7.5 required")
    return torch.device("cuda:0")

def run(expected_commit: str) -> dict[str, object]:
    started = datetime.now(timezone.utc); start = time.perf_counter()
    preflight(expected_commit)
    c2, c4, omic = validate_inputs()
    device = configure_cuda()
    synthetic = run_one_patient_supervisor_healnet_smoke(
        OFFICIAL, torch.zeros((1,4742,2048),device=device),
        torch.zeros((1,1,1558),device=device), torch.zeros((1,1,21),device=device),
        torch.zeros((1,1,1333),device=device))
    model = build_resnet50_imagenet1k_v2(CHECKPOINT).to(device).eval()
    branch2 = PatchBranchSpec("scale_2x", c2, 0, 256)
    branch4 = PatchBranchSpec("scale_4x", c4, 1, 256)
    r2 = extract_feature_matrix(StreamingOpenSlideDataset(WSI, branch2), model, device=device, batch_size=32, num_workers=2)
    r4 = extract_feature_matrix(StreamingOpenSlideDataset(WSI, branch4), model, device=device, batch_size=32, num_workers=2)
    combined = torch.cat((r2.features, r4.features), dim=0).contiguous()
    if tuple(combined.shape) != (4742,2048) or combined.dtype != torch.float32 or not torch.isfinite(combined).all():
        raise RuntimeError("combined feature contract failed")
    real = run_one_patient_supervisor_healnet_smoke(
        OFFICIAL, combined.unsqueeze(0).to(device), omic.rna.to(device),
        omic.mutation.to(device), omic.cnv.to(device))
    provenance = build_two_scale_provenance(
        scale_2x=BranchProvenanceSpec("scale_2x", c2, 0, .4936, .4936),
        scale_4x=BranchProvenanceSpec("scale_4x", c4, 1, .9872376717207693, .9872376717207693),
        scale_2x_count=3773, scale_4x_count=969)
    metadata = CompactFeatureMetadata(PATIENT, SLIDE, UUID, WSI_SHA, COORD_MANIFEST_SHA,
        BRCA_RELEASE_ARCHIVE_SHA256, CHECKPOINT_SHA, POLICY_SHA, expected_commit, 3773, 969)
    manifest = publish_compact_feature_artifacts(OUTPUT, combined_features=combined,
        row_provenance=provenance, metadata=metadata, preserve_failed_staging=True)
    manifest_sha = digest(OUTPUT / "compact_manifest.json")
    validate_compact_feature_artifacts(OUTPUT, expected_manifest_sha256=manifest_sha)
    finished = datetime.now(timezone.utc)
    return {
      "status":"BRCA_BATCH_B01_GPU_FEATURE_PILOT_SUCCESS",
      "started_at_utc":started.isoformat(), "finished_at_utc":finished.isoformat(),
      "total_seconds":time.perf_counter()-start, "source_commit":expected_commit,
      "device":str(device), "gpu":torch.cuda.get_device_name(0),
      "torch":torch.__version__, "torchvision":torchvision.__version__,
      "counts":{"scale_2x":3773,"scale_4x":969,"total":4742},
      "feature_shapes":{"scale_2x":list(r2.features.shape),"scale_4x":list(r4.features.shape),"combined":list(combined.shape),"natural":[1,4742,2048]},
      "timings":{"scale_2x":r2.streaming_extraction_seconds,"scale_4x":r4.streaming_extraction_seconds},
      "peak_gpu_memory_bytes":max(r2.peak_gpu_memory_bytes,r4.peak_gpu_memory_bytes),
      "synthetic_healnet":asdict(synthetic), "real_healnet":asdict(real),
      "artifact":{"directory":str(OUTPUT),"manifest_sha256":manifest_sha,"manifest":manifest,
        "total_file_bytes":sum(p.stat().st_size for p in OUTPUT.iterdir())},
      "operations":{"training":0,"backward":0,"optimizer_steps":0,"amp":0,"tf32":0,"coordinate_generation":0,"deletions":0,"other_patients":0},
    }

def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--expected-source-commit",required=True)
    args=parser.parse_args()
    try: result=run(args.expected_source_commit)
    except Exception as exc:
        print(json.dumps({"status":"BLOCKED","error":f"{type(exc).__name__}: {exc}"},sort_keys=True)); return 1
    print(json.dumps(result,sort_keys=True)); return 0

if __name__ == "__main__": raise SystemExit(main())
