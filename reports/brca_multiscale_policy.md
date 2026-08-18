# BRCA multiscale policy gate

Status: `BRCA_PHASE_1_TENSOR_POLICY_CPU_VERIFIED_WSI_NOT_AUTHORIZED`

The supervisor has approved BRCA as the next cohort and BLCA remains the frozen
engineering reference. This approval does **not** yet authorize downloading,
opening, tiling, or extracting features from any BRCA WSI. A
three-patient proposal may be prepared from existing metadata, but its exact
rows require confirmation before acquisition. The current work is limited to
CPU-safe inspection of already-local metadata and the clean Omic archive, code,
configuration, and tests.

## Current verified Omic contract

The released archive at
`Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip`
was inspected without extraction. Its outer SHA256 is
`4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`.
It contains one CSV payload named `./tcga_brca_all_clean.csv.zip` with 1,022
data rows and 2,922 columns.

The identity key is the literal pair `(case_id, slide_id)`. A case-only match,
filename normalization, or row-position match is prohibited. Exactly one row
must match. The CSV feature order is retained within each modality:

| Input | Header suffix | Width | Tensor contract |
|---|---|---:|---|
| RNA | `_rnaseq` | 1,558 | contiguous finite `float32 [1,1,1558]` |
| Mutation | `_mut` | 21 | contiguous finite `float32 [1,1,21]` |
| CNV | `_cnv` | 1,333 | contiguous finite `float32 [1,1,1333]` |

The loader streams the sole archive member and fails on a changed metadata
prefix, duplicate header, missing/reordered/unclassified features, wrong group
width, malformed row, duplicate identity match, nonnumeric value, NaN/Inf,
unexpected archive member, or optional source checksum mismatch. It does not
import OpenSlide or access a WSI.

## Proposed physical-scale policy — not yet approved

`scale_2x` and `scale_4x` are pipeline branch labels. They must not be mapped to
OpenSlide level numbers. The proposed physical targets are approximately
`0.5 µm/px` and `1.0 µm/px`, respectively; neither target may be claimed as the
actual scale until slide metadata and any approved resampling are recorded.

For each subsequently authorized slide, the policy will be fail-closed:

1. Verify the exact approved GDC UUID, filename, byte size, and MD5 before the
   slide is opened.
2. Read native `mpp-x`, `mpp-y`, pyramid dimensions, and level downsample values.
   Missing, nonnumeric, nonfinite, zero, or negative physical metadata rejects
   the slide.
3. Compute each level's physical scale independently on both axes from the base
   MPP and reported downsample. Never assume that level 0, 1, or 2 represents a
   requested branch.
4. Compare candidate levels against the target in physical units on both axes.
   A numeric acceptance tolerance and tie rule must be approved before this
   comparison can authorize a level.
5. Reject the slide when either axis is outside the approved tolerance or when
   no unique acceptable mapping exists. There is no silent nearest-level
   fallback.
6. Do not resize/resample to manufacture a target scale unless the supervisor
   explicitly approves the resampling rule, interpolation, source footprint,
   and provenance label.
7. Record native MPP, effective MPP, chosen level/downsample, error on each
   axis, coordinate policy, and any approved resampling for every retained
   feature row.

No OpenSlide level is hardcoded by this policy. The frozen BLCA level choices
are slide-specific evidence and must not be copied to BRCA.

## Phase 1 WSI tensor contract

Phase 1 resolves the distinction between a patch count and a cohort count.
Let `S` be the number of WSI-aligned samples, `P_i` the accepted patch count
for WSI `i`, and `D=2048` the ResNet50 feature width. Each WSI remains its own
variable-length bag:

```text
scale_2x_i [P_2x_i, 2048]
scale_4x_i [P_4x_i, 2048]
torch.cat([scale_2x_i, scale_4x_i], dim=0) -> F_i [P_i, 2048]
one-WSI model batch                          -> [1, P_i, 2048]
P_i = P_2x_i + P_4x_i
cohort                                      -> {F_i}_{i=1..S}
```

The natural HEALNet WSI axis order is `[batch, patches, channels]`. Therefore
the WSI `channel_dims` entry is `2048`, while the attention length is the
runtime patch count `P_i`. `P_i` is not 44,445 by default, and the BLCA patch
count must never be hardcoded for BRCA.

The initial supervisor-aligned execution is deliberately batch size one. It
uses no padding and no attention mask. The released model exposes one shared
mask argument across modalities of different lengths; silently applying a WSI
padding mask to the one-token Omic modalities would be unsafe. Any future
multi-patient batching requires a separately implemented and tested
modality-specific masking extension.

Patches from different patients must not be concatenated into a single
unlabelled matrix, and a WSI must not be globally averaged to `[1,2048]`.
Multi-WSI patients remain excluded initially, and later train/validation/test
splits must be grouped by patient identity to prevent leakage. WSI remains a
separate HEALNet input alongside the three Omic tensors; WSI and Omic values
are never directly concatenated. This corrected interface passed synthetic CPU
validation in Phase 1 and has not been run on a BRCA WSI.

## Encoder contract

The supervisor requested ImageNet-pretrained ResNet50 features. For direct
engineering comparability with the frozen BLCA pilot and the released feature
code, Phase 1 freezes torchvision `ResNet50_Weights.IMAGENET1K_V2`, removes the
classifier, and retains a 2,048-dimensional vector for every patch. The
checkpoint is 102,540,417 bytes with SHA256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
The paper described Kather100K pretraining, so this engineering pipeline must
not be described as an exact paper reproduction.

## Pending decisions before any WSI access

- Confirm the exact three pilot rows: case ID, slide ID, GDC UUID, filename,
  declared size, and MD5.
- Explicitly authorize those three WSI downloads.
- Approve the physical-MPP tolerance and deterministic tie rule.
- Approve native-level use versus a defined resampling procedure.
- Approve incomplete-boundary patch handling and the tissue-coordinate policy.
- Approve raw-WSI retention, remote-upload, and verified deletion policy.

Training is separately unauthorized.

## When to switch to GPU

Remain on the CPU machine for Omic validation, alignment reports, manifest
preparation, and unit tests. Switch to a GPU machine only after the exact
three-patient plan and physical-scale policy are approved, immediately before
ResNet50 feature extraction and the four-input HEALNet smoke run. WSI access
must still have its own explicit authorization; a GPU switch is not acquisition
approval.
