# BRCA Q75 acquisition and header-only transition

Status: `BRCA_Q75_LOCAL_GDC_ACQUISITION_READY`

The user's exact authorization has been bound to the frozen CPU review at
commit `69d89a5df986b129fd0b9dc4dcc1e8534627093e`, the successful Q25 and
Q50 result records, and the previously guarded Q75 row. This transition only
creates the executable one-row GDC manifest and fail-closed policy metadata.
It did not contact GDC, create local Q75 staging, open an SVS, access pixels,
or use CUDA.

## Exact authorized object

| Field | Value |
|---|---|
| Patient | `TCGA-E2-A154` |
| GDC UUID | `25aec062-60d1-446e-a1c6-0c79cc74a770` |
| Filename | `TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs` |
| Declared bytes | `1,360,743,825` |
| Expected MD5 | `a8c4b68fb6e0ab3e862efe3ed1fe10d7` |
| GDC state | `released` |
| Omic source index | `771` (physical CSV line `773`) |
| RNA | `[1,1,1558]` |
| Mutation | `[1,1,21]` |
| CNV | `[1,1,1333]` |

The reviewed alignment records exactly one released WSI and one clean Omic
row for this patient, with an exact full slide match. That identity must be
reverified again before any header result is published.

## Executable manifest and binding

The only newly executable data manifest is:

`multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770.AUTHORIZED_Q75_ONLY.gdc.tsv`

It contains the standard GDC header and exactly one data row. Its SHA256 is
`8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a`,
identical to the reviewed guarded row's content hash. No combined manifest was
created. The authorization config SHA256 is
`335e6d36aac1c21cc1cd52f8a14e5d2ecfde1f3a6f398d796bf842baaca35979`;
the decoded exact user statement has SHA256
`b43031fe0a9df8e5c44f35f9666567b202e4eb80ecb816386679a856a013bf01`.

The manifest-set record SHA256 is
`2330c4bc66c73c8e150be2d028aefb2a84916b18e4b5076d95fc28cf869d7050`.
It freezes Q25/Q50, prohibits CUDA and later stages, and requires the metadata
stop.

## Transfer concurrency interpretation

"Concurrency one" is encoded as one manifest row, one WSI, one patient, and
one active GDC client process. It is not encoded as `--n-processes 1`: GDC
documents that the client's internal `-n/--n-processes` value cannot be below
three. Internal client transfer processes must remain at the supported client
default/minimum. This does not authorize another patient or another WSI.

## Required post-download gate

Before OpenSlide construction, the eventual runner must fail closed unless:

- the sole completed object resolves to the exact UUID directory and exact
  filename;
- byte size is exactly `1,360,743,825` and MD5 is exactly
  `a8c4b68fb6e0ab3e862efe3ed1fe10d7`;
- an independent SHA256 is calculated and bound to the result;
- no partial or incomplete-download artifact remains; and
- the SVS is a regular, non-symlink file.

Only then may OpenSlide be constructed for header-property access. The
allowlist is native `mpp-x`, `mpp-y`, level count, every level dimension, and
every level downsample. `read_region` and all other pixel or region paths are
forbidden. The result may assess whether approximately `0.5` and `1.0 µm/px`
are natively achievable or appear to require controlled resampling, but it
must not infer or approve a Q75 scale policy.

## Safety boundary

- WSI download: `AUTHORIZED_Q75_ONLY`, not performed by this transition
- OpenSlide construction: header-only after every exact-file check
- Pixel/region access, masks, coordinates and patches: `NOT_AUTHORIZED`
- ResNet50, feature generation and HEALNet: `NOT_AUTHORIZED`
- Scale-policy approval: `NOT_AUTHORIZED`
- Q25/Q50 or BLCA modification: `NOT_AUTHORIZED`
- Google Drive and raw-file deletion: `NOT_AUTHORIZED`
- Full-cohort processing and training: `NOT_AUTHORIZED`
- CUDA: not required and not authorized for this gate

At transition time, `Q75.incoming`, `Q75.coordinates`, and `Q75.features` were
absent. The existing BRCA pilot tree used `1,954,756,342` apparent bytes and
the filesystem reported `349,292,040,192` available bytes. These are a CPU
preflight observation, not proof of future logical-quota availability.

## Validation and required stop

The focused Q75 authorization suite passed `21/21`. It covers exact approval
and source hashes, exact one-row reproduction, predecessor success, symlink
and tamper rejection, later-scope lockout, pre-OpenSlide checks, and the GDC
concurrency interpretation. Building the manifest reported zero network
requests, zero downloads, zero WSI opens, zero pixel reads, and no CUDA use.

All authorization, CPU-review, Q25/Q50-result, guarded-manifest, authorized-
manifest, and authorization-record reads use an opened descriptor with
`O_RDONLY|O_NOFOLLOW|O_CLOEXEC`. The reader verifies a regular file and size
limit with `fstat`, reads only from that descriptor, then compares initial and
final descriptor tokens plus the final path token. Symlink substitution,
in-place mutation, and rename/replacement during a read therefore fail closed.
Dedicated source-symlink and path-swap regressions passed. The finalized
authorization config, authorized TSV, and manifest-set record were unchanged
by this hardening.

Stop after the separately executed exact-file and header-only report. The next
decision must come from the user; this transition does not authorize tissue
masking, coordinates, scale policy, GPU extraction, HEALNet, or training.
