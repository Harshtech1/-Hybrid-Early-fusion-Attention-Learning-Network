# B06 acquisition and header-only metadata result

B06 passed the authorized CPU-only acquisition, exact file integrity, exact
Omic rematch, and OpenSlide header-only inspection. No pixel API was called.

| Field | Verified value |
|---|---|
| Patient | `TCGA-EW-A1OY` |
| Slide | `TCGA-EW-A1OY-01Z-00-DX1.42AF5C9A-A90F-4A58-B5D4-B615F8CD4333.svs` |
| GDC UUID | `00e926c5-d65d-42e5-8ba4-afaab66db47c` |
| Local path | `/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B06.incoming/00e926c5-d65d-42e5-8ba4-afaab66db47c/TCGA-EW-A1OY-01Z-00-DX1.42AF5C9A-A90F-4A58-B5D4-B615F8CD4333.svs` |
| Size | 1,691,494,753 bytes |
| MD5 | `dbb9edd8528f25bbcaaee907b1b1ab68` |
| SHA256 | `505e237c82a266abffe451f07bbd75911d7a34263c9f980cfb5e6a2dc3509924` |
| Exact Omic row | 897 |
| Native MPP | 0.2485 × 0.2485 µm/px |

## Pyramid

| Level | Dimensions | OpenSlide downsample |
|---:|---:|---:|
| 0 | 114,910 × 80,587 | 1.0 |
| 1 | 28,727 × 20,146 | 4.000109266924847 |
| 2 | 7,181 × 5,036 | 16.002066931213214 |
| 3 | 3,590 × 2,518 | 32.006362546213225 |

This report records geometry only and does not infer or approve a scale or
coordinate policy. The raw SVS remains retained. No masks, coordinates,
patches, features, HEALNet, CUDA, other-patient processing, deletion, Drive,
cohort expansion, or training occurred. Separate review is required before
any B06 pixel access.
