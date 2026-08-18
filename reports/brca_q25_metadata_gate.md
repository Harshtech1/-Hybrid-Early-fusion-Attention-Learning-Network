# BRCA Q25 metadata gate

Audit time: 2026-08-18T21:11:16Z

## Outcome

`BRCA_Q25_METADATA_BLOCKED_NO_NATIVE_0_5_MPP_LEVEL`

The approved Q25 WSI was downloaded and passed its exact file-identity gate. A
header-only OpenSlide inspection then found no native pyramid level within the
approved 10% per-axis tolerance of the 0.5 micrometers-per-pixel target. The
metadata policy therefore failed closed before any pixel read, coordinate
generation, feature extraction, HEALNet execution, or training.

This is a scale-policy mismatch for this slide, not a corrupt download, GPU
problem, ResNet problem, or HEALNet problem.

## Live storage gate

The Lightning dashboard supplied by the user immediately before acquisition
reported:

- total persistent storage: 200 GB
- used persistent storage: 3.41 GB
- derived available persistent storage: approximately 196.59 GB

This passed the storage prerequisite for the single 648 MB Q25 acquisition.
Google Drive was not mounted or used.

## Exact WSI identity

- cohort: `TCGA-BRCA`
- candidate label: `Q25`
- patient: `TCGA-LL-A6FP`
- GDC UUID: `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6`
- filename: `TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs`
- local path: `/teamspace/studios/this_studio/brca_pilot_data/Q25.incoming/dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6/TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs`
- expected and actual size: `648046947` bytes
- expected and actual MD5: `75536393096ffd928bc35ec9503c3655`
- independently calculated SHA256: `ac852be806eb0d91214145983319b604919a8e1d16bd59378a3dba69a600979c`
- regular file: yes
- symbolic link: no
- `.partial` files remaining: none
- number of `.svs` files in the Q25 acquisition: one

## Authorization and implementation identity

- pilot HEAD: `853bfeadd642cf575c258e220f55ce2e15cce7ad`
- authorization config SHA256: `a3a0704b8cc56acc6b082fd969e137107af3f64f27217fd808b9e9f4a54c016b`
- metadata policy SHA256: `daffdd77d271f28cb9440061b675d0991529f03ab99fb77bcdf5a3fb17275c43`
- metadata evaluator SHA256: `b443ec59b96fc5bf117891389e8c5521a2dda028215b270811b3fca4a80fce48`
- authorized one-row GDC manifest SHA256: `95bb1c4491497c265b868e96698cef3c3dd501458f801dae3c1dd702f3efa297`
- bound authorization-record SHA256: `8c6971df846a1dcbebfd83f689cd6496c8a85dcbf3b8eb74781cb7a85849aaea`
- OpenSlide Python: `1.4.6`
- native OpenSlide library: `3.4.1`

## Permitted metadata operations performed

Only the following operations were performed after the exact size and MD5
gate passed:

1. open the slide with OpenSlide;
2. read `openslide.mpp-x`;
3. read `openslide.mpp-y`;
4. read `level_dimensions`;
5. read `level_downsamples` and `level_count`;
6. close the slide.

`OpenSlide.read_region` was replaced with a fail-fast guard during the check.
It was not called. No associated-image pixels, thumbnails, or regions were
read.

## Raw OpenSlide metadata

- vendor: `aperio`
- MPP X: `0.2525`
- MPP Y: `0.2525`
- native level count: `4`

| Level | Dimensions (width x height) | Downsample | Calculated MPP X | Calculated MPP Y |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 65736 x 67406 | 1.0 | 0.2525 | 0.2525 |
| 1 | 16434 x 16851 | 4.00005934365913 | 1.0100149842739303 | 1.0100149842739303 |
| 2 | 4108 x 4212 | 16.002635628163056 | 4.040665496111171 | 4.040665496111171 |
| 3 | 2054 x 2106 | 32.00527125632611 | 8.081330992222343 | 8.081330992222343 |

The pyramid structure itself passed validation: metadata were finite and
positive, downsampling was strictly increasing, dimensions were non-growing,
and reported dimensions were consistent with the downsample values.

## Target evaluation

The approved policy requires native levels, separate native mappings for the
two targets, no resampling, and no more than 10% relative error on either MPP
axis.

| Branch | Target MPP | Diagnostic nearest native level | Native MPP | Per-axis error | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| scale_2x | 0.5 | 0 | 0.2525 | 49.5% | **FAIL**: outside 10% |
| scale_4x | 1.0 | 1 | 1.0100149842739303 | approximately 1.0015% | individually compatible |

The exact fail-closed error was:

```text
scale_2x nearest native level 0 is outside the 10.0% per-axis tolerance: x=49.500000%, y=49.500000%
```

The second row is a diagnostic calculation only. The combined policy did not
pass because both required branches must pass.

## Required decision before continuing

No coordinate or extraction work may begin under the current policy. A
supervisor-approved decision is required between at least these two choices:

1. retain native-only processing and select a different BRCA WSI whose
   pyramid genuinely has separate native levels near both 0.5 and 1.0 MPP; or
2. explicitly authorize a documented resampling policy for the 0.5-MPP branch
   (for example, resampling level 0), including interpolation, patch footprint,
   coordinate, boundary, and provenance rules.

This report does not authorize either choice. Silent resampling is prohibited.

## Explicit stop confirmation

- Q50/Q75 download: not performed and still locked
- pixel/region reads: not performed
- coordinate generation or CLAM: not performed
- ResNet50 feature extraction: not performed
- `.pt` generation: not performed
- HEALNet execution: not performed
- training: not performed
- GPU use: not required and not performed
- Google Drive mount/upload: not performed
- frozen BLCA tag: not modified
- official HEALNet checkout: not modified
