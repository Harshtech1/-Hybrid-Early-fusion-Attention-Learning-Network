# BRCA Validation Batch B01 — Acquisition and Header-Only Result

## Result

`BRCA_B01_FILE_OMIC_AND_HEADER_METADATA_VERIFIED`

B01 was downloaded and verified within the authorized CPU-only scope. The raw SVS remains in persistent staging. No pixel or region API was called, and B02–B06 remain untouched.

## Exact identity and integrity

| Field | Verified value |
|---|---|
| Patient | `TCGA-GI-A2C8` |
| Slide | `TCGA-GI-A2C8-01Z-00-DX1.09BD8AC9-645A-4C8B-9B36-77D833BDBA09.svs` |
| GDC UUID | `0a886f18-c44c-4b5e-b243-6df6e27f426a` |
| Size | 408,704,377 bytes |
| MD5 | `a9f830d456b4a1fe0e9bb5b5b99f4b7e` |
| Independent SHA256 | `b3c27e220c5c3961600782af42e91a52ea0b85a710d8d4aa722831ea00f9ad5f` |
| File type | Regular, non-symlink SVS |
| Partial/incomplete files | None |
| Local path | `/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B01.incoming/0a886f18-c44c-4b5e-b243-6df6e27f426a/TCGA-GI-A2C8-01Z-00-DX1.09BD8AC9-645A-4C8B-9B36-77D833BDBA09.svs` |

The GDC client reported one successful download. Its first invocation exited before transfer because the planned destination directory did not yet exist; the empty directory was then created and the identical one-row command succeeded. Nothing was deleted.

## Exact Omic match

The clean BRCA archive retained its frozen SHA256, and row `924` exactly matched both the patient and full slide identifier. RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]` tensors were finite CPU `float32` values.

## OpenSlide header only

| Level | Dimensions | OpenSlide downsample |
|---:|---:|---:|
| 0 | 63,784 × 39,311 | 1.0 |
| 1 | 15,946 × 9,827 | 4.00015264068383 |
| 2 | 3,986 × 2,456 | 16.004057258221366 |

- Native MPP: `0.2468 × 0.2468 µm/px`
- Pyramid levels: `3`
- `read_region` calls: `0`

OpenSlide received a stable `/proc/self/fd` path backed by an `O_NOFOLLOW` descriptor. `read_region` was temporarily replaced with a blocking function during inspection, and its call counter remained zero. The file's MD5 and SHA256 were recomputed after the header inspection and remained unchanged.

No scale policy is inferred or approved by this result.

## Storage and boundary

Filesystem usage increased by 409,268,224 bytes. The raw WSI and the GDC parcel log are retained. No GPU/CUDA, mask, coordinates, features, model execution, deletion, Drive operation, additional-patient processing, cohort expansion, or training occurred.

The required stop has been reached. The next decision must separately address either B01 pixel/coordinate policy work or B02 acquisition; neither is currently authorized.
