# HEALNet: start here tomorrow

## Current stopping point

```text
verified HDF5 coordinates (8,911)
    ↓
ImageNet1K_V2 ResNet50
    ↓
released-code pilot .pt
```

Feature extraction has **not started**. No pilot `.pt` exists.

## First action: attach a GPU

The current blocker is that Lightning Studio has no accessible NVIDIA GPU. Do not run the feature pilot on CPU.

Run:

```bash
nvidia-smi
python --version
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"
```

Continue only when `nvidia-smi` succeeds and CUDA availability is `True`.

## Pilot identifiers

- Track: `released_code`
- WSI UUID: `bc9e3954-59d0-4f25-9022-42c97db7aea2`
- HDF5: `/teamspace/studios/this_studio/healnet_pilot/blca_preprocessed/patches/TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.h5`
- Coordinates: `8911`
- CLAM SHA: `26e0b6c4873e112f1ccd74cd834894c4ab7a2934`
- Expected released-code weight: `resnet50-11ad3fa6.pth` (`IMAGENET1K_V2`), not yet downloaded
- Planned tensor: `(8911, 2048)` float32

## Resume location

Follow [PROJECT_STATE.md §19](PROJECT_STATE.md#19-tomorrows-exact-next-steps) and [released-code provenance](tracks/released_code/provenance/status.yaml).

The last local execution prompt is:

```text
/teamspace/studios/this_studio/.codex/attachments/25d17b7a-2988-4cfd-a7a0-fd4475fcca05/pasted-text.txt
```

## Do not confuse the tracks

**Released-code reproduction ≠ paper-faithful reproduction.**

- Released-code encoder: ImageNet1K V2 ResNet50
- Paper encoder claim: Kather100K ResNet50, checkpoint unresolved

Author email is pending. The author is out of office until August 17, 2026. Do not send another email now.

Stop after the one-slide feature and loader-shape smoke test. Do not download a cohort or run training.
