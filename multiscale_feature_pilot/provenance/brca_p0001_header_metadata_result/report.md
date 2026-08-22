# P0001 production acquisition and header-only metadata result

P0001, frozen cohort row 1, passed the authorized CPU-only acquisition, exact
file integrity, exact Omic rematch, and OpenSlide header-only inspection.
No pixel API was called, and no later production patient was started.

| Field | Verified value |
|---|---|
| Production block | `BRCA_PRODUCTION_BLOCK_0001_0008` |
| Cohort row | `1` (`P0001`) |
| Patient | `TCGA-3C-AALK` |
| Slide | `TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53.svs` |
| GDC UUID | `93b26333-5723-4fa4-a4de-6124c04ab243` |
| Local path | `/teamspace/studios/this_studio/brca_pilot_data/BRCA_PRODUCTION_P0001.incoming/93b26333-5723-4fa4-a4de-6124c04ab243/TCGA-3C-AALK-01Z-00-DX1.4E6EB156-BB19-410F-878F-FC0EA7BD0B53.svs` |
| Size | 1,769,848,096 bytes |
| MD5 | `3d63b3311612d763525b6edb0848b986` |
| SHA256 | `f43597a87463d8d15007918dd5174ff966aa28dcb0de71cdc5752576cd7c2b5b` |
| Exact Omic row | 4 |
| Native MPP | 0.25 × 0.25 µm/px |

## Pyramid

| Level | Dimensions | OpenSlide downsample |
|---:|---:|---:|
| 0 | 95,488 × 81,920 | 1.0 |
| 1 | 23,872 × 20,480 | 4.0 |
| 2 | 5,968 × 5,120 | 16.0 |
| 3 | 2,984 × 2,560 | 32.0 |

This report records geometry only and does not infer or approve a scale or
coordinate policy. The raw SVS remains retained. P0002 through P0008 remain
unstarted. No masks, coordinates, patches, features, HEALNet, CUDA, deletion,
Drive, cohort expansion, or training occurred. Separate review is required
before any P0001 pixel access.
