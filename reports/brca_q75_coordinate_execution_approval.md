# BRCA Q75 coordinate-execution approval

Status: `Q75_COORDINATE_EXECUTION_AUTHORIZED_CPU_ONLY`

The user explicitly authorized one narrowly bounded Q75 coordinate execution.
The authorized operation is exactly one OpenSlide level-2 mask read at
level-0 location `(0,0)` with size `6783×5654`, followed by the frozen tissue
segmentation policy, generation of the two Q75 coordinate bags, atomic
external publication, validation, and reporting.

No Q75 pixel was read while this authorization package was recorded.

## Exact authorized object

- Patient: `TCGA-E2-A154`
- Slide: `TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`
- GDC UUID: `25aec062-60d1-446e-a1c6-0c79cc74a770`
- Size: `1,360,743,825` bytes
- MD5: `a8c4b68fb6e0ab3e862efe3ed1fe10d7`
- SHA256: `844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37`
- Exact Omic row: `771`, with exact case and full-slide match
- Output: `/teamspace/studios/this_studio/brca_pilot_data/Q75.coordinates`

The WSI must be opened through a held `O_NOFOLLOW` descriptor. Identity and
hashes must be checked on that descriptor, OpenSlide must use its stable
`/proc/self/fd` path, and the same descriptor must be rechecked before
publication. Any identity, header, policy, source, or repository drift fails
closed before publication.

## Frozen execution policy

The shared mask uses the already reviewed
`REVIEWED_BRCA_Q75_COORDINATE_POLICY_V1_EXECUTION_LOCKED` parameters. Both
branches use the same level-2 tissue contours:

- `scale_2x`: global level-0 `512×512` footprint and step;
- `scale_4x`: global level-0 `1024×1024` footprint and step;
- coordinates: unique, nonempty `int64 [N,2]`, columns `(x,y)`, row-major by
  `(y,x)`;
- publication: `BRCA_COORDINATE_ARTIFACT_SET_V1`, sibling staging followed by
  Linux `RENAME_NOREPLACE`, with neither overwrite nor resume.

## Deletion boundary

The user's deletion prohibition applies to the raw Q75 WSI, every pre-existing
user or project file, prior coordinate artifacts, and the final published
output. Atomic publication narrowly permits cleanup of runner-created ephemeral
Q75 coordinate lock or staging paths from this transaction only. It does not
permit deletion or replacement of the final output.

## Required stop

The authorization does not permit patch or tile reads, level-0/level-1 pixel
reads, patch resampling, ResNet50 or feature generation, HEALNet, GPU/CUDA,
Q25/Q50/BLCA changes, Google Drive, raw-file deletion, full-cohort processing,
official HEALNet changes, or training. Execution must stop immediately after
the coordinate artifact is validated and reported.

The authorization file and pure fail-closed validator are cycle-free: they
bind the frozen predecessor evidence, while the separately committed runner
must pin their byte hashes and all execution-critical source files before any
WSI open.

The authorization-focused suite passed `18/18` tests without opening the WSI
or exposing an image, coordinate, publication, CUDA, or network API through the
pure authorization validator.
