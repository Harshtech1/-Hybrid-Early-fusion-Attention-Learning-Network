# B03 acquisition and header-only authorization request

This package is prepared but **not executable**. B03 has not been downloaded or opened.

| Field | Frozen local metadata |
|---|---|
| Patient | `TCGA-AR-A1AY` |
| Slide | `TCGA-AR-A1AY-01Z-00-DX1.6AC0BE3B-FFC5-4EDA-9E40-B18CAAC52B81.svs` |
| GDC UUID | `266c3852-f5d0-4815-94d9-dac5b0ff8276` |
| Expected size | 918,454,431 bytes |
| Expected MD5 | `1980b183d5bb948c2fc263af62a4b1b4` |
| Exact Omic row | 372; RNA 1558, mutation 21, CNV 1333 |
| One-row manifest SHA256 | `847ff7458fafafbba4596178531d7caf3e66a01d47a775e1879c5db4526ac73b` |

The existing guarded one-row manifest remains explicitly `NOT_AUTHORIZED`. Current filesystem availability is 338,797,510,656 bytes; the conservative preflight requirement is 21,836,908,862 bytes, but storage must be checked again immediately before any transfer.

## Exact next authorization

> I authorize downloading only BRCA validation-batch patient B03, UUID `266c3852-f5d0-4815-94d9-dac5b0ff8276` (`918,454,431` bytes), from GDC to local persistent staging at concurrency one using its exact one-row manifest. After download, perform only exact UUID/filename/size/MD5 verification, independent SHA256 calculation, partial/incomplete checks, regular non-symlink SVS confirmation, exact Omic rematch, and OpenSlide header-only inspection of MPP and every pyramid level. Do not call `read_region` or access pixels; do not generate masks, coordinates, or features; do not use GPU/CUDA, Drive, deletion, B04–B06 processing, cohort expansion, or training. Stop after the B03 metadata report and wait for my review.

Until that statement is separately authorized, no B03 download or OpenSlide operation may occur.
