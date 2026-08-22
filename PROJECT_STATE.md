# HEALNet pilot project state

Last updated: 2026-08-17 UTC

Operational continuation: [GPU_HANDOFF.md](GPU_HANDOFF.md)

## Authority

The PhD supervisor's instructions are the primary implementation target. The official paper and immutable released code are evidence/provenance sources. Differences are recorded rather than silently reconciled.

## Project definition

Build a patient-matched WSI + multi-omic survival pipeline in which patches from two WSI scales are encoded by an ImageNet-pretrained ResNet50, the two WSI feature bags are concatenated along their patch-row axis, and combined WSI, RNA, mutation, and CNV remain separate inputs to HEALNet's shared-latent fusion model.

## Fixed patient and tensor contracts

```text
WSI patient A + Omic patient A -> keep
WSI patient A + Omic patient B -> discard
missing WSI or Omic            -> discard
```

```text
scale_2x [N1,2048]
scale_4x [N2,2048]
torch.cat(..., dim=0) -> [N1+N2,2048]
loader form           -> [2048,N1+N2]
one-patient batch     -> [1,2048,N1+N2]
```

Do not use `cat(dim=1)`, `stack`, or direct WSI/Omic concatenation.

## Current measured status

| Item | Status |
|---|---|
| Supervisor architecture | Confirmed and documented |
| KIRP alignment summary | 260 matched: 249 single-candidate, 11 multi-WSI deferred; deterministic row-level selection manifest still required |
| Verified BLCA WSI | Completed |
| BLCA OpenSlide metadata and MD5 | Completed |
| Level-1 CLAM coordinates | 8,911 rows, HDF5 verified |
| BLCA WSI/Omic patient + slide identity | Exact match |
| BLCA RNA/mutation/CNV availability | 1,523 / 1,125 / 193 finite values |
| Multiscale adapter | Implemented |
| Unit tests | 66 passed at freeze validation |
| Four-modality synthetic HEALNet interface | Finite `[1,4]` |
| GPU / checkpoint / OpenSlide gates | Verified on Tesla T4 |
| Real ResNet50 features | `[35534,2048]` and `[8911,2048]`, verified |
| Combined feature bag | `[44445,2048]`, verified |
| Real-input HEALNet interface smoke | Finite `[1,4]`; random initialization |
| BLCA one-patient pilot | `BLCA_ONE_PATIENT_PILOT_SUCCESS` |
| Training/evaluation | Not started |

The synthetic HEALNet test uses random model weights and proves only shape/numerical interface compatibility. It is not scientific inference or a survival prediction.

## BLCA engineering pilot

```text
case:       TCGA-2F-A9KT
GDC UUID:   bc9e3954-59d0-4f25-9022-42c97db7aea2
slide:      TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs
size:       2,658,499,382 bytes
WSI MD5:    824785fee9387dcf46a7058a0722739b
HDF5 SHA:   e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51
```

OpenSlide levels:

| Level | Downsample | MPP |
|---:|---:|---:|
| 0 | 1 | 0.2277 |
| 1 | 4 | 0.9108 |
| 2 | 16 | 3.6432 |
| 3 | 64 | 14.5728 |

Scale policy:

- 2x branch: level 0 downsampled by factor 2, effective `0.4554 µm/px`, explicitly labelled an engineering approximation.
- 4x branch: native level 1, actual `0.9108 µm/px`, explicitly labelled approximate.
- Remaining 2x implementation details: interpolation, tissue-filtering/threshold, and incomplete-boundary handling.

## Encoder provenance

| Track | Encoder |
|---|---|
| Paper claim | Kather100K-pretrained ResNet50 |
| Released code | torchvision `ResNet50_Weights.IMAGENET1K_V2` |
| Supervisor implementation | torchvision `ResNet50_Weights.IMAGENET1K_V2` |

The supervisor track is not an exact reproduction of the paper's encoder choice.

Expected checkpoint filename: `resnet50-11ad3fa6.pth`. Keep it in the external torch cache, never Git.

## Validated GPU pilot runtime

```text
torch:                     2.8.0+cu128
torchvision:               0.23.0+cu128
torch.cuda.is_available(): true
torch CUDA device count:   1
GPU:                       Tesla T4, 15,360 MiB
NVIDIA driver:             580.173.02
checkpoint cached:         true
checkpoint SHA256:         11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
```

## Repository boundary

Pilot repository: this directory.

Official reference repository: a separate `healnet` checkout. The official source remains read-only. The immutable camera-ready tag `v0.1.0` peels to commit `28ba5da6ab99fd8069972c22e986d83edb658dd4`.

GitHub contains only small source, tests, configuration, manifests, and provenance. It excludes raw WSI, HDF5, `.pt` features, checkpoints, credentials, local caches, generated images, and the nested CLAM checkout.

## Next phase

The BLCA one-patient engineering pilot has passed and is frozen. Before any KIRP acquisition, restore the real KIRP Omic LFS object and materialize a deterministic row-level alignment/selection manifest. Resolve the 11 multi-WSI cases explicitly, compute exact selected storage, and recheck disk headroom. Do not train or download the cohort until that evidence is complete.

The authoritative result record is [reports/validated_pilot_baseline.md](reports/validated_pilot_baseline.md). The HEALNet interface smoke was random-initialized and is not a trained survival prediction.
