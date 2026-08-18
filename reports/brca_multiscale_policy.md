# BRCA multiscale policy gate

Status: `BRCA_CPU_PREPARATION_READY_WSI_NOT_AUTHORIZED`

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

## Dynamic WSI input contract

Once authorized, the two feature matrices will remain separate until validated:

```text
scale_2x [N2, 2048]
scale_4x [N4, 2048]
torch.cat([scale_2x, scale_4x], dim=0) -> [P, 2048]
transpose + one-patient batch          -> [1, 2048, P]
P = N2 + N4
```

`P` is a positive runtime value for each patient. It is not 44,445 by default,
and the BLCA patch count must never be hardcoded for BRCA. WSI is a separate
HEALNet input alongside the three Omic tensors; WSI and Omic values are never
directly concatenated. This interface is documented only and has not been run
for BRCA.

## Pending decisions before any WSI access

- Confirm the exact three pilot rows: case ID, slide ID, GDC UUID, filename,
  declared size, and MD5.
- Explicitly authorize those three WSI downloads.
- Approve the physical-MPP tolerance and deterministic tie rule.
- Approve native-level use versus a defined resampling procedure.
- Approve the treatment of multi-WSI patients.
- Approve incomplete-boundary patch handling and the tissue-coordinate policy.
- Choose ImageNet1K V2 or Kather100K encoder weights and freeze the checksum.
- Approve raw-WSI retention, remote-upload, and verified deletion policy.

Training is separately unauthorized.

## When to switch to GPU

Remain on the CPU machine for Omic validation, alignment reports, manifest
preparation, and unit tests. Switch to a GPU machine only after the exact
three-patient plan and physical-scale policy are approved, immediately before
ResNet50 feature extraction and the four-input HEALNet smoke run. WSI access
must still have its own explicit authorization; a GPU switch is not acquisition
approval.
