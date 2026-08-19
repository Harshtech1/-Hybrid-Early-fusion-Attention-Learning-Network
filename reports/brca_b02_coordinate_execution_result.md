# BRCA B02 coordinate execution result

Status: **successfully verified**. The authorized B02 CPU coordinate gate completed in 18.97 seconds and stopped before patch or model work.

Exactly one OpenSlide call read the level-2 mask at `(0,0)`, size `5580×4560`. The in-memory RGBA mask SHA256 was `900382a4c56fe89894846bcd88004a7a2be8f8ea1ae8ee609bb5f6945a16d658`. Segmentation found one retained tissue contour and four retained holes.

| Branch | Coordinates | Source | Footprint/step | Effective MPP |
|---|---:|---|---:|---:|
| 2× | 7,158 | level 0 | 512×512 | 0.501 |
| 4× | 1,862 | level 1 | 1024×1024 level-0 declaration; native 256 read later | 1.0020374 |
| Total | 9,020 | — | — | — |

The exact four-file coordinate set occupies 168,736 bytes. Its manifest SHA256 is `2b3e5dd754ebb4ca4ec26f3e017e21548b0115dc2d0517ae83146d8f7ec52ba2`. Independent validation confirmed both bags are nonempty, unique, int64 `[N,2]`, row-major, lattice-aligned, in bounds, and consistent with every manifest attribute.

The raw WSI remains intact. No patch read, feature extraction, ResNet50, HEALNet, CUDA, B03–B06, Q25/Q50/Q75/B01 or BLCA change, Drive operation, deletion, cohort expansion, or training occurred.

The next gate requires separate review and authorization before any B02 patch reads or GPU feature extraction.
