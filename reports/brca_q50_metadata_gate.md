# BRCA Q50 exact-file and metadata gate

Status: `BRCA_Q50_EXACT_FILE_AND_HEADER_METADATA_VERIFIED`

The exact Q50-only GDC transfer completed and passed independent file-identity
checks. A header-only OpenSlide inspection then collected MPP and pyramid
metadata. No region or pixel was read, and the slide was closed immediately.
Google Drive was not used.

## Exact matched patient

| Field | Exact value |
|---|---|
| Patient | `TCGA-AR-A1AW` |
| GDC UUID | `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62` |
| Filename | `TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs` |
| Bytes | `975,626,387` |
| MD5 | `304509e03f26cbecc9aee4ea691c8e5a` |
| SHA256 | `6b960db7e6e5772f4f626daa7b023ffbb5e3b20f9f6ff5c7d52f4554cf33399b` |
| Omic source row | `370` |
| Omic shapes | RNA `[1,1,1558]`; mutation `[1,1,21]`; CNV `[1,1,1333]` |

The SVS is a regular non-symlink file, its size and MD5 exactly match the
committed one-row GDC manifest, and no `.partial` or `.part` file is present.

## Header metadata

- MPP-X: `0.2468`
- MPP-Y: `0.2468`
- Level-0 dimensions: `99,960 × 65,334`
- Level-0 pixel count: `6,530,786,640` (about 6.53 gigapixels)
- OpenSlide levels: `4`

| Level | Dimensions | Reported downsample | Scalar-derived native MPP |
|---:|---:|---:|---:|
| 0 | `99,960 × 65,334` | `1.0` | `0.2468` |
| 1 | `24,990 × 16,333` | `4.000061225739301` | `0.9872151105124595` |
| 2 | `6,247 × 4,083` | `16.001375061204985` | `3.9491393651053903` |
| 3 | `3,123 × 2,041` | `32.009231974117526` | `7.899878451212205` |

The scalar OpenSlide downsample and the exact per-axis dimension ratios are
recorded separately in the provenance YAML. This avoids pretending rounded
pyramid dimensions define identical x/y geometry.

## Candidate scale mapping—not yet an execution policy

The actual Q50 metadata supports the supervisor's approximate physical scales:

- approximately 0.5 MPP: level-0 `512 × 512` source footprint resampled to
  `256 × 256`, yielding `0.4936` MPP (1.28% target error);
- approximately 1.0 MPP: native level-1 `256 × 256`, yielding
  `0.9872151105124595` MPP (about 1.2785% target error).

This is a Q50-specific metadata observation, not a silent reuse of Q25's
numerical scale mapping. Interpolation, lattice, mask level and algorithm,
complete-boundary handling, coordinate ordering, and artifact publication
must be frozen and reviewed before the first Q50 pixel read.

## Stop boundary

- `read_region` calls: `0`
- pixel or associated-image reads: `0`
- Q75 operations: `0`
- Google Drive operations: `0`
- coordinate and feature operations: `0`
- HEALNet and training operations: `0`
- raw WSI deletions: `0`

The authorized stop was reached. The next milestone is a reviewed Q50-specific
scale and coordinate policy; no coordinate or GPU operation is authorized by
this record.
