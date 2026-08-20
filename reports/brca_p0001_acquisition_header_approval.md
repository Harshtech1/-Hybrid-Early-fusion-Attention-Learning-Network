# P0001 production acquisition and header-only approval

P0001 is the first patient in the frozen eight-patient production block. The
direct authorization permits one CPU-only GDC transfer using the exact
request-only manifest, exact integrity verification, exact Omic row rematch,
and OpenSlide header inspection only.

## Frozen identity

| Field | Value |
|---|---|
| Cohort index | `1` (`P0001`) |
| Patient | `TCGA-3C-AALK` |
| UUID | `93b26333-5723-4fa4-a4de-6124c04ab243` |
| Size | `1,769,848,096` bytes |
| MD5 | `3d63b3311612d763525b6edb0848b986` |
| Omic row | `4` |
| Manifest SHA256 | `1c05dc64b6af54604648b52e688e02f611f8e42aaf82cf911c81035b3fc385f2` |

The gate must use one GDC client, retain the raw WSI, prove the absence of
partial files, hash the WSI through a held non-following descriptor, and open
OpenSlide only through the stable descriptor path. It contains no pixel API.

P0002 through P0008 remain unstarted. Coordinate generation, patch reads,
features, CUDA, HEALNet, Drive, deletion, and training remain prohibited. The
required stop is the atomic P0001 metadata result for user review.
