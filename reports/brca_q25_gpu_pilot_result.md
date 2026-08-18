# BRCA Q25 GPU feature pilot result

Status: `BRCA_Q25_GPU_FEATURE_PILOT_SUCCESS`

The corrected Q25-only GPU pilot completed successfully on the Tesla T4. It
produced and atomically published two ImageNet1K V2 ResNet50 feature bags, an
exact row-wise concatenation, and row-level provenance. It then passed the
natural-layout four-modality HEALNet numerical interface check and stopped.
No training occurred.

[Attempt 1](brca_q25_gpu_attempt_1.md) had stopped before publication because
the pre-CUDA cuBLAS environment requirement was missing. It retained zero
feature artifacts and performed zero training. Attempt 2 used the corrected,
reviewed deterministic process contract recorded here.

## Same-patient inputs

- Case: `TCGA-LL-A6FP`
- GDC UUID: `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6`
- WSI: exact Omic `case_id` and full `slide_id` match
- Level-0 dimensions: `65,736 × 67,406` = `4,431,000,816` pixels
- RNA: `[1,1,1558]`
- mutation: `[1,1,21]`
- CNV: `[1,1,1333]`
- Encoder: classifier-removed torchvision ResNet50 with
  `ResNet50_Weights.IMAGENET1K_V2`

The WSI, coordinate bags, Omic archive/row, and checkpoint were hash-checked
before use and rechecked before publication.

RNA, mutation, and CNV remained three separate HEALNet modalities. None was
concatenated with the WSI tensor.

## Feature result

| Branch | Effective MPP | Feature shape | Finite |
|---|---:|---:|---:|
| scale-2x | `0.505` | `[7404,2048]` | yes |
| scale-4x | `1.0100149842739303` | `[1918,2048]` | yes |
| concatenated | — | `[9322,2048]` | yes |

The combined tensor is exactly
`torch.cat([scale_2x, scale_4x], dim=0)`: its first 7,404 rows equal the 2x
tensor and its remaining 1,918 rows equal the 4x tensor. No pooling or
transpose was performed. The HEALNet WSI input was therefore the natural
layout `[1,9322,2048]`.

## HEALNet gates

The corrected process required `CUBLAS_WORKSPACE_CONFIG=:4096:8` while keeping
deterministic algorithms enabled. Before any WSI patch read, a full-size
synthetic interface check passed. After extraction, the real-feature check
also passed.

Both checks produced:

- finite output `[1,4]`
- WSI attention `[1,2,9322]`
- RNA, mutation, and CNV attention `[1,2,1]` each

HEALNet was randomly initialized, in evaluation/inference mode, with no
backward pass or optimizer. The `[1,4]` value is an interface/numerical smoke
result—not a survival prediction.

## Retained artifacts

Directory:
`/teamspace/studios/this_studio/brca_pilot_data/Q25.features`

This directory is external to the Git repository and none of its data files is
Git-tracked.

| File | Bytes | SHA256 |
|---|---:|---|
| `scale_2x_features.pt` | 60,655,145 | `f1dcdfc93510700f72dffcb264d01aded1f2927b35850d7da28ee3ac7f14b9d3` |
| `scale_4x_features.pt` | 15,713,833 | `bdf5622c52e753a1032d6a1747f6ce2c3c9c961910f2757ddbb6e2797695e19a` |
| `combined_features.pt` | 76,367,401 | `5dbd65789458d6239aac2190b5bce5d38a29769a7d589fdeada26f9d60ae1b9d` |
| `row_provenance.csv` | 465,018 | `40bb70fdf87b5bf0633aa44935cfd836ea3aa7b06ac726c3f4f37ffae3153395` |
| `feature_manifest.json` | 7,368 | `88a3146d34cf907604cecf109ff3879b26afcf6458ce86a9e2e46a7e127dcc0f` |
| `feature_manifest.json.sha256` | 88 | `d8f9e0f28e544d9588f2c4eb4c184789359401dabdac96d7a220c80d964a7e55` |

Total retained file bytes: `153,208,853` (about 153.2 MB decimal).

The project validator and an independent tensor reload both passed. All six
paths are regular files; the manifest binds the execution commit
`61caefed25852c070282bb294f063b783a3c35ba`, authorization, WSI, coordinates,
checkpoint, and implementation hashes.

## Runtime and stop boundary

- Total elapsed: `155.79` seconds
- Peak extraction allocation: `481,334,784` bytes
- Source commit: `61caefed25852c070282bb294f063b783a3c35ba`
- Official HEALNet: unchanged at `28ba5da6ab99fd8069972c22e986d83edb658dd4`
- Frozen BLCA tag: unchanged at `df7cf2bda783ab6cc09e95d6a1fa0914da05a433`
- Q50/Q75 and full-cohort operations: zero
- Drive operations and raw WSI deletion: zero
- Training and optimizer steps: zero
- Backward passes and KIRP operations: zero

The required stop was reached. A separate reviewed decision is required before
training or processing any additional patient.
