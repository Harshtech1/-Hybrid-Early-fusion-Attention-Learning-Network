# BRCA B03 coordinate execution result

Status: **successfully verified**. The authorized B03 CPU coordinate gate completed in 18.13 seconds and stopped before patch, feature, model, or GPU work.

Exactly one OpenSlide call read the level-2 RGBA mask at level-0 location `(0,0)`, size `5831×3632`. The in-memory mask SHA256 was `3bee296abdee520137bc4817f2e35bc8faba80c4b04e174cd130d417002a9a51`. Segmentation found two retained tissue contours and six retained holes.

| Branch | Coordinates | Source | Footprint and step | Effective MPP |
|---|---:|---|---:|---:|
| 2× | 8,875 | level 0 | 512×512 | 0.4936 |
| 4× | 2,257 | native level 1 | 1024×1024 in level-0 coordinates; native 256×256 later | 0.9872085 |
| **Total** | **11,132** | — | — | — |

The exact four-file coordinate set occupies 202,485 bytes (221,184 allocated bytes). Its manifest SHA256 is `cced78f863415ee12d57b905131e60758ba849645196c0a14309e9ab8d4e1ae5`.

Independent read-only validation reopened every file and confirmed:

- both bags are nonempty `int64 [N,2]` arrays;
- all rows are unique and strictly ordered by `y`, then `x`;
- coordinates follow the frozen 512/1024-pixel level-0 lattices and remain in bounds;
- HDF5 attributes exactly match the manifest;
- file, coordinate-content, manifest, and sidecar hashes all match;
- the output directory contains exactly the four authorized regular, non-symlink files, with no remaining lock or staging path.

The B03 raw WSI remains intact at 918,454,431 bytes and independently revalidated as MD5 `1980b183d5bb948c2fc263af62a4b1b4` and SHA256 `4ef4ac79ce3cc0bfc5a4ea62985f080f0f877dca4c7e43191a04a35b2eba8228`. Omic row 372 and all three CPU modality tensors were rematched before the WSI open and again before publication.

No patch read, feature extraction, ResNet50, HEALNet, CUDA/GPU, B04–B06 processing, Q25/Q50/Q75/B01/B02 or BLCA change, Drive operation, deletion, cohort expansion, or training occurred.

These are coordinate artifacts only, not features or scientific model output. The next gate requires separate review and authorization before any B03 patch reads or GPU feature extraction.
