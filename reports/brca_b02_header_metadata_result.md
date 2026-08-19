# BRCA B02 acquisition and header-only result

Status: **verified successfully**. Exactly one B02 GDC object was downloaded at concurrency one and inspected using header properties only. No pixel API was called.

## Identity and integrity

| Field | Verified value |
|---|---|
| Patient | `TCGA-BH-A0BG` |
| Slide | `TCGA-BH-A0BG-01Z-00-DX1.0838FB7F-8C85-4687-9F70-D136A1063383.svs` |
| UUID | `c5331e5e-10b4-4979-958b-d4592a2805de` |
| Size | 724,114,911 bytes |
| MD5 | `a8c6e730df401ff67e1a1e52a6cb6307` |
| SHA256 | `df85b3c048b18ae0a5b9414e7e220110d98891f73f28189849e6e602d1743741` |
| Omic match | exact case and full slide, row 472 |

The SVS is a regular non-symlink file. No partial, temporary, or incomplete payload exists. The completed GDC tree contains the SVS and its expected parcel log only. The raw WSI remains retained.

## OpenSlide header

Native MPP is `0.2505 × 0.2505 µm/px`.

| Level | Dimensions | OpenSlide downsample |
|---:|---:|---:|
| 0 | 89,291 × 72,971 | 1.0 |
| 1 | 22,322 × 18,242 | 4.0001494261 |
| 2 | 5,580 × 4,560 | 16.0021918034 |
| 3 | 2,790 × 2,280 | 32.0043836069 |

The metadata suggests that approximately 0.5 µm/px can be obtained by a controlled 2× resampling of level 0, while level 1 provides approximately 1.002 µm/px natively. This is an observation, not an approved scale policy.

One initial client invocation exited before transfer because the authorized destination directory did not yet exist. The exact empty directory was created and the same one-row command then completed successfully; no partial payload resulted from the first invocation.

No `read_region`, mask, coordinate, feature, CUDA, B03–B06, deletion, Drive, cohort-expansion, or training operation occurred. The required stop has been reached. A separate review is required before any B02 pixel access or scale-policy approval.
