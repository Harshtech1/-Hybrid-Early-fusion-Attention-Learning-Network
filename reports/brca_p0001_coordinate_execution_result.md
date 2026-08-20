# BRCA production P0001 coordinate execution result

Status: **successfully verified**. The authorized P0001 CPU coordinate gate
completed in 33.55 seconds and stopped before patch, feature, model, GPU, or
training work.

Exactly one OpenSlide call read the level-2 RGBA mask at level-0 location
`(0,0)`, size `5968×5120`. The in-memory mask SHA-256 was
`95a9b6b09cf5815208af6caa1b941f97efce8aed6bd3025d12474566e6e55064`.
Segmentation retained four tissue contours and five holes.

| Branch | Coordinates | Source | Level-0 footprint and step | Effective MPP |
|---|---:|---|---:|---:|
| 2× | 13,372 | level 0 | 512×512 | 0.5 |
| 4× | 3,444 | native level 1 | 1024×1024 | 1.0 |
| **Total** | **16,816** | — | — | — |

The exact four-file coordinate set occupies 293,322 logical bytes (307,200
allocated bytes). Its manifest SHA-256 is
`f1825acf7d0b96c92bfb66038d60af6e1572ea945490e2245d5e3ec222677d6e`.

Read-only post-publication validation confirmed:

- both bags are nonempty `int64 [N,2]` arrays;
- every coordinate is unique, strictly ordered by `y` then `x`, on the frozen
  512/1024-pixel level-0 lattices, and within the complete-footprint bounds;
- all HDF5 attributes exactly match the manifest;
- coordinate-content, HDF5-file, manifest, and sidecar hashes all match; and
- the output directory contains exactly the four authorized regular files,
  with no lock or staging path left behind.

The P0001 WSI remains intact at 1,769,848,096 bytes. Its identity was checked
before the read and again using the same held `O_NOFOLLOW` descriptor after the
read: MD5 `3d63b3311612d763525b6edb0848b986`, SHA-256
`f43597a87463d8d15007918dd5174ff966aa28dcb0de71cdc5752576cd7c2b5b`.
The exact Omic row 4 and all three CPU modality tensor hashes were also
rematched before the WSI open and before publication.

No patch read, feature extraction, ResNet50, HEALNet, CUDA/GPU, P0002–P0008
processing, prior-pilot or BLCA change, Drive operation, raw-file deletion,
cohort expansion, or training occurred.

These are coordinate artifacts only—not features or scientific model output.
The required stop has been reached. Any P0001 patch reads or GPU feature
extraction require a separate reviewed package and explicit authorization.
