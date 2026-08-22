# GPU-machine handoff

> Historical pre-extraction handoff. This procedure records the migration state before the successful GPU pilot and is superseded for current status by [reports/validated_pilot_baseline.md](reports/validated_pilot_baseline.md).

This is the canonical restart guide for moving the supervisor-aligned HEALNet pilot to a machine with an NVIDIA GPU.

## Current stopping point

```text
patient identity + architecture + adapter + synthetic interface: VERIFIED
real ResNet50 extraction:                                      NOT STARTED
runtime gate measured 2026-08-17:                              BLOCKED_NO_GPU
```

The source machine had no accessible NVIDIA GPU and did not contain `resnet50-11ad3fa6.pth`. It generated no real `.pt` features and performed no training.

## Authority and fixed architecture

Supervisor instructions take priority. The paper and released repository are provenance references.

```text
SAME PATIENT
├── WSI
│   ├── 2x engineering branch, 0.4554 micrometres/pixel
│   │   └── patches -> ImageNet1K V2 ResNet50 -> [N1,2048]
│   └── 4x approximate native branch, 0.9108 micrometres/pixel
│       └── patches -> same ResNet50 -> [N2,2048]
│
│   torch.cat([scale_2x, scale_4x], dim=0) -> [N1+N2,2048]
│   transpose + batch -> [1,2048,N1+N2]
│
├── RNA      -> [1,1,1523]
├── Mutation -> [1,1,1125]
└── CNV      -> [1,1,193]

[WSI, RNA, Mutation, CNV] -> HEALNet -> [1,4]
```

Never concatenate WSI directly with RNA, mutation, or CNV. The only explicit pre-HEALNet concatenation is between the two WSI feature bags along patch rows.

## Clone layout

Keep the pilot and official checkout as siblings because the compatibility tests locate the official model read-only at `../healnet`:

```text
workspace/
├── healnet/        # official repository, tag v0.1.0 for reference
└── healnet_pilot/  # this repository and handoff branch
```

```bash
git clone --branch gpu-handoff-multiscale-pilot \
  https://github.com/Harshtech1/-Hybrid-Early-fusion-Attention-Learning-Network.git \
  healnet_pilot

GIT_LFS_SKIP_SMUDGE=1 git clone --branch v0.1.0 \
  https://github.com/konst-int-i/healnet.git \
  healnet

git -C healnet lfs pull \
  --include='data/tcga/omic_xena/blca_master.csv' \
  --exclude=''
```

Do not copy the source machine's modified official HEALNet worktree. Use a clean official checkout and keep it read-only.

## Environment

Create an isolated environment using Python 3.12 if available. Install the PyTorch build appropriate for the target NVIDIA driver/CUDA environment, then install the small pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install the target-machine CUDA build of torch and torchvision first.
# Verified source pair: torch 2.8.0 / torchvision 0.23.0.

python -m pip install -r healnet_pilot/requirements-handoff.txt
```

The operating system also needs the OpenSlide shared library. `openslide-python` alone does not supply that system library on every platform. For example, use the target distribution's `openslide-tools`/`libopenslide` package and verify that `import openslide` succeeds.

## Data are deliberately not in GitHub

Securely transfer or remount these artifacts outside Git. Preserve filenames and verify hashes before use:

| Artifact | Required identity |
|---|---|
| BLCA WSI | `TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs` |
| WSI GDC UUID | `bc9e3954-59d0-4f25-9022-42c97db7aea2` |
| WSI MD5 | `824785fee9387dcf46a7058a0722739b` |
| Level-1 coordinate HDF5 | same slide stem, `.h5` |
| HDF5 SHA256 | `e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51` |
| BLCA Omic source | local `blca_master.csv` containing case `TCGA-2F-A9KT` |
| BLCA Omic SHA256 | `9fa2cd83906c00e1f50113ba8e806ea4537806aeb2aacbee60e86c80b53f6929` |

The matching BLCA Omic CSV is a Git LFS object in the official repository. A clone without `git lfs pull` may contain only a small pointer file; the readiness gate rejects that pointer through its SHA256 check. Alternatively, securely transfer the verified 9,318,450-byte CSV. The matched row has one exact patient/slide match and complete finite groups: RNA 1,523, mutation 1,125, and CNV 193. Revalidate this after transfer; never match by row order.

The implemented loader `load_blca_pilot_omics` requires both `_PATIENT == "TCGA-2F-A9KT"` and the exact full `slide_id`. It then selects `*_rnaseq`, `*_mut`, and `*_cnv` columns in original CSV header order, requires exactly 1,523/1,125/193 values, converts them to float32 `[1,1,F]` tensors, and rejects zero/multiple matches, missing values, NaN, and Inf.

Never commit `.svs`, `.h5`, `.pt`, `.pth`, `.ckpt`, credentials, OAuth/rclone configuration, or the nested CLAM checkout.

## Validate the clone before real work

From `healnet_pilot`:

```bash
python -m pytest multiscale_feature_pilot/tests -q
```

Expected result at handoff: `24 passed`.

Run the read-only gate with explicit data paths:

```bash
python scripts/check_gpu_readiness.py \
  --official-repo ../healnet \
  --wsi /secure/path/to/TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs \
  --coordinates /secure/path/to/TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.h5 \
  --omic /secure/path/to/blca_master.csv
```

The gate exits nonzero until CUDA, `nvidia-smi`, the official model, checkpoint, and every supplied data path are available.

For an initial hardware/source-only check before transferring data or downloading the checkpoint:

```bash
python scripts/check_gpu_readiness.py --official-repo ../healnet --hardware-only
```

The full gate verifies the official peeled commit `28ba5da6ab99fd8069972c22e986d83edb658dd4`, imports OpenSlide, checks the WSI/HDF5/Omic hashes, and verifies that the checkpoint SHA256 begins with its official filename hash `11ad3fa6`.

## Checkpoint gate

Use only:

```text
torchvision.models.ResNet50_Weights.IMAGENET1K_V2
expected file: resnet50-11ad3fa6.pth
```

Only after the NVIDIA gate passes, allow torchvision to download this official checkpoint into its external cache. Record its path, byte size, and SHA256. Do not place it in either repository.

## Real one-patient sequence

1. Record GPU, driver, CUDA, torch, torchvision, OpenSlide, and checkpoint provenance.
2. Finalize and record 2x interpolation, tissue filtering/threshold, and incomplete-boundary handling.
3. Generate independent 2x coordinates from level 0. A 256x256 output patch covers a 512x512 level-0 footprint and uses a 512-pixel non-overlapping level-0 step.
4. Use the already validated 8,911 level-1 coordinates for the 4x approximate branch.
5. Stream patches without saving patch images.
6. Apply RGB conversion, the documented resize/normalization, classifier-removed ImageNet1K V2 ResNet50, `eval()`, and inference mode on GPU.
7. Validate each branch as rank-2 float32 `[N,2048]` with finite values and matching provenance rows.
8. Combine only with `torch.cat([features_2x, features_4x], dim=0)`.
9. Verify row order and hashes; then save artifacts atomically outside Git.
10. Load RNA, mutation, and CNV for the same patient as separate inputs.
11. Run one random-initialized HEALNet interface forward with real patient inputs and require a finite `[1,4]` output. This is a shape/numerical smoke test, not scientific inference or a survival prediction.
12. Stop. Do not train or acquire the 249-patient KIRP cohort yet.

## Canonical pilot records

- `multiscale_feature_pilot/config/pilot_config.yaml`
- `multiscale_feature_pilot/provenance/gate_status.yaml`
- `multiscale_feature_pilot/provenance/scale_selection.yaml`
- `multiscale_feature_pilot/reports/adapter_design.md`
- `multiscale_feature_pilot/reports/next_real_multiscale_extraction.md`

The transfer-context source used for this handoff was `healnet/Reference/HEALNet_TRANSFER_CONTEXT.md` in the source workspace. This document captures the execution-critical subset needed after cloning.

## Deliberate implementation boundary

This branch contains the tested multiscale bag/provenance/padding adapters, synthetic HEALNet interface tests, and the strict readiness gate. It does **not** yet contain a push-button real extraction CLI, a finalized 2x coordinate generator, or a real-patient HEALNet runner. Those are the first implementation tasks on the GPU machine after the three remaining 2x policy details are frozen; the handoff must not be described as completed real extraction software.
