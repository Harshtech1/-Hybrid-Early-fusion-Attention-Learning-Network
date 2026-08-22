# GPU-machine restart checklist

> Historical pre-extraction checklist. The one-patient pilot subsequently passed; use [reports/validated_pilot_baseline.md](reports/validated_pilot_baseline.md) for the frozen reference state.

Start with [GPU_HANDOFF.md](GPU_HANDOFF.md). It is the canonical operational handoff.

## Current stopping point

```text
same-patient BLCA identity                         VERIFIED
two-scale tensor/provenance/padding adapter        VERIFIED
four-modality synthetic HEALNet forward -> [1,4]  VERIFIED
unit tests                                         24 PASSED
real GPU ResNet50 extraction                       NOT STARTED
```

The source runtime was blocked because `nvidia-smi` was unavailable, CUDA device count was zero, and `resnet50-11ad3fa6.pth` was not cached.

## First actions on the GPU machine

1. Clone this branch and a clean official HEALNet `v0.1.0` checkout as sibling directories.
2. Restore the BLCA WSI, level-1 coordinate HDF5, and BLCA Omic table outside Git.
3. Verify their hashes and the exact `TCGA-2F-A9KT` patient/slide match.
4. Install a CUDA-compatible torch/torchvision pair and `requirements-handoff.txt`.
5. Run:

   ```bash
   python -m pytest multiscale_feature_pilot/tests -q
   python scripts/check_gpu_readiness.py --official-repo ../healnet \
     --wsi /secure/path/pilot.svs \
     --coordinates /secure/path/pilot.h5 \
     --omic /secure/path/blca_master.csv
   ```

6. Continue only after the gate reports `"ready": true`.

## Fixed decisions

- Same-patient matching is mandatory; mismatches are discarded.
- Encoder: classifier-removed `ResNet50_Weights.IMAGENET1K_V2`, 2048-D per patch.
- WSI 2x branch: level-0 factor-2 engineering approximation, `0.4554 µm/px`.
- WSI 4x branch: native level 1 approximation, `0.9108 µm/px`.
- WSI branch combination: `torch.cat([scale_2x, scale_4x], dim=0)`.
- WSI, RNA, mutation, and CNV remain separate HEALNet modalities.
- Expected one-patient output: finite `[1,4]`.

## Still to record before real extraction

- 2x interpolation;
- tissue-filtering resolution/threshold;
- incomplete-boundary handling;
- target-machine GPU/software/checkpoint provenance.

The repository contains the tested adapter and readiness gate, not a completed real extraction CLI. Implement or validate the extraction runner on the GPU machine only after the remaining 2x policy details are frozen.

Stop after one real BLCA forward. Do not train, download the KIRP cohort, modify the official repository, or commit data/model artifacts.
