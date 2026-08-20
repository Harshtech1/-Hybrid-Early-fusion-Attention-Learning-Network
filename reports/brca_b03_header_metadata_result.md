# B03 acquisition and header-only metadata result

## Result

B03 passed the authorized CPU-only acquisition, integrity, exact Omic-match, and OpenSlide header inspection. The SVS was not accessed through `read_region`, thumbnail, associated-image, mask, coordinate, patch, feature, CUDA, or model APIs.

| Field | Verified value |
|---|---|
| Patient | `TCGA-AR-A1AY` |
| Slide | `TCGA-AR-A1AY-01Z-00-DX1.6AC0BE3B-FFC5-4EDA-9E40-B18CAAC52B81.svs` |
| GDC UUID | `266c3852-f5d0-4815-94d9-dac5b0ff8276` |
| Local path | `/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B03.incoming/266c3852-f5d0-4815-94d9-dac5b0ff8276/TCGA-AR-A1AY-01Z-00-DX1.6AC0BE3B-FFC5-4EDA-9E40-B18CAAC52B81.svs` |
| Size | 918,454,431 bytes |
| MD5 | `1980b183d5bb948c2fc263af62a4b1b4` |
| SHA256 | `4ef4ac79ce3cc0bfc5a4ea62985f080f0f877dca4c7e43191a04a35b2eba8228` |
| Exact Omic row | 372 |
| Native MPP | 0.2468 × 0.2468 µm/px |

## Pyramid

| Level | Dimensions | OpenSlide downsample |
|---:|---:|---:|
| 0 | 93,296 × 58,121 | 1.0 |
| 1 | 23,324 × 14,530 | 4.000034411562285 |
| 2 | 5,831 × 3,632 | 16.00123898678414 |
| 3 | 2,915 × 1,816 | 32.00522239895422 |

Level 1 corresponds to approximately 0.9872085 µm/px and is therefore a native candidate for the approximately 1.0 µm/px branch. No native approximately 0.5 µm/px level exists; reaching that target would require separately reviewed controlled resampling from level 0. This is a geometry observation, not an approved scale policy.

## Boundary and storage

The exact completed tree contains only the SVS and its GDC parcel log. Every payload is regular and non-symlink, no partial/incomplete file exists, and the same held file descriptor produced the same SHA256 before and after header inspection. Available filesystem capacity changed from 338,799,800,320 to 337,877,794,816 bytes; the raw SVS remains retained.

No B04, B05, B06, cohort, Drive, deletion, GPU, feature, HEALNet, or training operation occurred. The required stop has been reached. Separate review is required before any B03 pixel access or scale-policy approval.
