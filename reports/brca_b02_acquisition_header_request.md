# B02 acquisition and header-only authorization request

This package is prepared but **not executable**. B02 has not been downloaded or opened.

| Field | Frozen local metadata |
|---|---|
| Patient | `TCGA-BH-A0BG` |
| Slide | `TCGA-BH-A0BG-01Z-00-DX1.0838FB7F-8C85-4687-9F70-D136A1063383.svs` |
| GDC UUID | `c5331e5e-10b4-4979-958b-d4592a2805de` |
| Expected size | 724,114,911 bytes |
| Expected MD5 | `a8c6e730df401ff67e1a1e52a6cb6307` |
| Exact Omic row | 472; RNA 1558, mutation 21, CNV 1333 |
| One-row manifest SHA256 | `8c398bab8159885645ca45006a78872db64234eb37be566631f3ebf1b593bf89` |

The existing guarded one-row manifest remains explicitly `NOT_AUTHORIZED`. Current filesystem availability is 345,399,492,608 bytes; the conservative preflight requirement is 21,448,229,822 bytes, but storage must be checked again immediately before any transfer.

## Exact next authorization

> I authorize downloading only BRCA validation-batch patient B02, UUID `c5331e5e-10b4-4979-958b-d4592a2805de` (`724,114,911` bytes), from GDC to local persistent staging at concurrency one using its exact one-row manifest. After download, perform only exact UUID/filename/size/MD5 verification, independent SHA256 calculation, partial/incomplete checks, regular non-symlink SVS confirmation, exact Omic rematch, and OpenSlide header-only inspection of MPP and every pyramid level. Do not call `read_region` or access pixels; do not generate masks, coordinates, or features; do not use GPU/CUDA, Drive, deletion, B03–B06 processing, cohort expansion, or training. Stop after the B02 metadata report and wait for my review.

Until that statement is separately authorized, no B02 download or OpenSlide operation may occur.
