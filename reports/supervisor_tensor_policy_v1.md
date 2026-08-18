# Supervisor tensor policy v1

Status: `PHASE_1_TENSOR_POLICY_CPU_VERIFIED`

This Phase 1 policy resolves the meaning of `N` without downloading or opening
any WSI. It preserves one 2,048-dimensional ResNet50 feature vector per
accepted patch, while keeping each WSI as a separate variable-length sample.
It does not average a WSI into one vector and does not merge patches from
different patients into one unlabeled tensor.

## Notation and shapes

| Symbol | Meaning |
|---|---|
| `S` | number of WSI-aligned cohort samples |
| `P_2x_i` | accepted 2x-branch patches for WSI `i` |
| `P_4x_i` | accepted 4x-branch patches for WSI `i` |
| `P_i` | `P_2x_i + P_4x_i` |
| `D` | ResNet50 feature width, fixed at 2,048 |

For each WSI `i`:

```text
F_2x_i [P_2x_i, 2048]
F_4x_i [P_4x_i, 2048]
F_i = cat([F_2x_i, F_4x_i], dim=0) -> [P_i, 2048]
HEALNet batch-size-one input                -> [1, P_i, 2048]
```

The complete cohort is the ragged collection `{F_i}_{i=1..S}`. There is no
single rectangular cohort tensor until an explicit batching and masking policy
is implemented. Cross-patient concatenation and per-WSI global pooling are both
prohibited.

## Why this orientation is used

The paper states that a 2,048-dimensional feature vector is generated for each
patch. The released code also computes one ResNet50 output row per patch and
saves one feature file per slide. In the generic HEALNet interface, inputs are
`[batch, spatial axes..., channels]`; therefore patches are the attention axis
and 2,048 is the WSI channel dimension. The natural one-WSI input is
`[1,P_i,2048]`, with `channel_dims[0]=2048`.

The historical released loader transposes slide features to `[2048,P]`. That
orientation is retained only as historical behavior and is not used by this
new supervisor-aligned adapter. The frozen BLCA artifacts remain valid because
their stored combined feature matrix is already `[44445,2048]`; no pixel
reprocessing or feature re-extraction is needed to test the corrected
orientation later.

The interface smoke loads the official model read-only from Git tag `v0.1.0`,
peeled commit `28ba5da6ab99fd8069972c22e986d83edb658dd4`; the tagged model-source
SHA256 is
`78e9c2b10455f6ad43a88ed4e0a646cbf6304bf52cac06a5140f49dbc4796820`.
The BLCA reference tag remains `blca-one-patient-pilot-v1` at
`df7cf2bda783ab6cc09e95d6a1fa0914da05a433`.

## Initial batching rule

Phase 1 fixes batch size to one WSI, with no padding and no attention mask.
Multi-WSI patients remain excluded. Any later multi-patient batch must add and
test modality-specific padding masks; the official model's single shared mask
cannot safely represent a variable WSI axis alongside one-token Omic axes.
Patient-grouped train/validation/test splits are mandatory to prevent leakage.

## Encoder identity

The engineering reference uses torchvision ResNet50 with
`IMAGENET1K_V2`, classifier removed, and one finite float32 2,048-vector per
patch. The checkpoint is 102,540,417 bytes and has SHA256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
This matches the frozen BLCA engineering pilot and released feature extractor.
The paper describes Kather100K pretraining, so this pipeline is not an exact
paper reproduction.

## Phase boundary

This phase is CPU-only and synthetic/non-pixel. It authorizes source,
configuration, provenance, and tests only. It does not authorize BRCA WSI
download, WSI opening, coordinate generation, ResNet inference, HEALNet
training, or full-cohort processing. The GPU switch remains unnecessary until
an exact three-slide download is authorized, hashes pass, and the physical
MPP/coordinate/storage policies are approved.

Verification used
`/home/zeus/miniconda3/envs/cloudspace/bin/python`. The 32 focused Phase 1
tests passed, including a synthetic forward with the actual BRCA modality
widths `[1558,21,1333]`; the complete non-pixel pilot suite passed 121 tests.
The smoke produced finite `[1,4]` output and verified WSI cross-attention length
equal to `P_i`, not 2,048. No real WSI was read and no feature extraction or
training occurred.

The execution-critical Phase 1 source hashes are:

| File | SHA256 |
|---|---|
| `src/supervisor_tensor.py` | `9456f92e578bacc0f5f639b0846fb3d953acecdd6e51ff80896f38650627589d` |
| `src/supervisor_healnet_smoke.py` | `df2c0010347ddc6cfc49200bf35802337752c01a78b96727a5ad538fb847eec4` |
| frozen helper `src/healnet_smoke.py` | `34a7ad1204dfc84ef8e63a2ac4cd8b932dbccf39e9f424b424dc6eb37a4d6009` |

The full path-to-hash map, including tests, is stored in
`multiscale_feature_pilot/provenance/supervisor_tensor_policy.yaml`.
