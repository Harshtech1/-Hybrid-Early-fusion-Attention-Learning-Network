# B06 acquisition and header-only authorization request

This is a non-executable request package. It uses only frozen local planning,
alignment, and guarded-manifest metadata. No B06 WSI was downloaded or opened,
and B04–B05 remain skipped.

| Field | Frozen local metadata |
|---|---|
| Patient | `TCGA-EW-A1OY` |
| Slide | `TCGA-EW-A1OY-01Z-00-DX1.42AF5C9A-A90F-4A58-B5D4-B615F8CD4333.svs` |
| GDC UUID | `00e926c5-d65d-42e5-8ba4-afaab66db47c` |
| Expected size | 1,691,494,753 bytes |
| Expected MD5 | `dbb9edd8528f25bbcaaee907b1b1ab68` |
| Exact Omic row | 897 |
| Alignment | `KEEP`; exact singleton case and full-slide match |
| Guarded one-row manifest SHA256 | `2616a09f26ee75460ca19dd1529c12b37981943256edbcf1f175dd50dfb3b3bb` |

The existing one-row manifest remains explicitly `NOT_AUTHORIZED`. The
current filesystem snapshot shows 344,027,402,240 bytes available. Applying
the frozen conservative gate,

\[
2(1{,}691{,}494{,}753) + 20{,}000{,}000{,}000
= 23{,}382{,}989{,}506\ \text{bytes},
\]

so the current estimated headroom is 320,644,412,734 bytes. This snapshot is
advisory; the storage gate must be checked again immediately before any future
transfer.

## Exact proposed authorization

> I authorize downloading only BRCA validation-batch patient B06, UUID `00e926c5-d65d-42e5-8ba4-afaab66db47c` (`1,691,494,753` bytes), from GDC to local persistent staging at concurrency one using its exact one-row manifest. After download, perform only exact UUID/filename/size/MD5 verification, independent SHA256 calculation, partial/incomplete checks, regular non-symlink SVS confirmation, exact Omic rematch, and OpenSlide header-only inspection of MPP and every pyramid level. Do not call `read_region` or access pixels; do not generate masks, coordinates, or features; do not use GPU/CUDA, Drive, deletion, additional-patient processing, cohort expansion, or training. Stop after the B06 metadata report and wait for my review.

If separately approved, that future stage would remain CPU-only and
single-patient/single-client. It would verify the exact payload and Omic match,
read only OpenSlide header properties (`mpp-x`, `mpp-y`, level count,
dimensions, and downsample values), and then stop. No pixel API, mask,
coordinate generation, feature extraction, model execution, deletion, Drive
operation, additional-patient processing, cohort execution, or training would
be permitted.

Until the exact statement above is authorized separately, B06 download,
OpenSlide access, and all downstream operations remain unauthorized.
