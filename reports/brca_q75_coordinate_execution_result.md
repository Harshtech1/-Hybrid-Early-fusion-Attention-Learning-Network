# BRCA Q75 coordinate execution result

Execution time: 2026-08-19 16:51 UTC

Status: `BRCA_Q75_COORDINATES_VERIFIED`

The authorized CPU-only Q75 coordinate milestone passed. The runner securely
reverified the exact WSI and matching Omic row, made exactly one OpenSlide
level-2 mask read, generated the two deterministic coordinate bags, and
published an exact four-file artifact set atomically outside Git. Independent
read-only validation reproduced every file hash, coordinate-content hash,
shape, dtype, count, attribute, ordering, lattice, and complete-footprint
bounds check.

No level-0 or level-1 patch pixels were read. No patch extraction,
resampling, ResNet50, HEALNet, GPU, training, Drive, deletion, or cohort work
occurred.

## Exact matched input

- Patient: `TCGA-E2-A154`
- Slide: `TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`
- GDC UUID: `25aec062-60d1-446e-a1c6-0c79cc74a770`
- WSI size: `1,360,743,825` bytes
- WSI MD5: `a8c4b68fb6e0ab3e862efe3ed1fe10d7`
- WSI SHA256: `844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37`
- Native MPP: `(0.2468, 0.2468)`
- Pyramid dimensions: `[(108528,90471), (27132,22617), (6783,5654), (3391,2827)]`
- OpenSlide downsamples: `[1.0, 4.000066321793341, 16.000619030774672, 32.00359724763015]`
- Exact matching Omic row: `771`
- Omic archive SHA256: `4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`

The WSI was held through an `O_NOFOLLOW` descriptor and accessed by its stable
descriptor path. It passed two independent SHA256 passes before OpenSlide and
a final same-descriptor MD5/SHA256 recheck. The Omic archive was likewise held
securely, and row 771 was rematched before publication. All three CPU float32
modalities were contiguous and finite:

| Modality | Shape | Content SHA256 |
| --- | --- | --- |
| RNA | `[1,1,1558]` | `7c8ec0aa7c77227ad1ad5347b3612fe2c38d210295d6d1914862c7638960f916` |
| Mutation | `[1,1,21]` | `81adcc3d1d6cfdde773ea46a27284afcf586598b5bd5cfecb675d98d70eabedf` |
| CNV | `[1,1,1333]` | `78bf5e3d18c12923687506d873800f5c01a83a5737e488296acde26814ed6803` |

## The one authorized mask read

- OpenSlide opens: `1`
- `read_region` calls: exactly `1`
- Location in level-0 coordinates: `(0,0)`
- Read level: `2`
- Read size at level 2: `6783 × 5654`
- Returned representation: contiguous RGBA `uint8`
- Raw C-order mask SHA256: `1dba486b3238f96dea5fdb430702986f1ee4d6e32254ef73d1ca9d92e0bbdde6`
- Retained foreground contours: `4`
- Retained holes: `17`
- Per-axis mask-to-level-0 geometry: `(16.0, 16.001238061549344)`

The mask existed only in memory for the authorized computation and was not
retained as an artifact.

## Coordinate artifacts

Artifact directory:
`/teamspace/studios/this_studio/brca_pilot_data/Q75.coordinates`

| Branch | Effective MPP | Coordinates | Range `(x,y)` | HDF5 bytes | HDF5 SHA256 | Coordinate-content SHA256 |
| --- | --- | ---: | --- | ---: | --- | --- |
| 2× | `(0.4936,0.4936)` | `[13487,2] int64` | `(15360,0)` to `(100864,88576)` | 225,464 | `d88c201d379a5954cdfa4d785760f6c8f9d4b8bec498f7f31d040b1fdf7440ec` | `88e1ac8e00d4f05da7f83e542bfe7c933e9849a29ae60deac68adabe79e748b0` |
| 4× | `(0.9872163682185965,0.9872163682185965)` | `[3458,2] int64` | `(15360,0)` to `(100352,88064)` | 65,000 | `0b0cfdaa26493dd24c3bbcba9f57c6b10d6060ba0f5f8b0a59bc0938ff655d94` | `63f58c687943509a55314ced55c7afe1610a26873a751d3e1d0f2d06cff3fb5d` |

The retained total is `16,945` coordinate rows. Both arrays are non-empty,
duplicate-free, strictly row-major by `(y,x)`, aligned to their global
level-0 lattices, and limited to complete footprints. The 2× step/footprint is
`512 × 512`; the 4× step/footprint is `1024 × 1024`.

- Manifest: `6,537` bytes; SHA256 `438165ce6b3be9d26d66c65cd70793e29cc92208cfb6a78bf68043bc4b4a4e90`
- Sidecar: `91` bytes; SHA256 `28885db3c297b1604a4a76a2e61c1e84df413f94b1146c819a7ba2cc51fc2fbc`
- Exact artifact file bytes: `297,092`
- Filesystem used-space change: `438,272` bytes

The output has exactly the two HDF5 files, JSON manifest, and SHA256 sidecar.
No transaction lock or staging directory remained after return. Publication
used the frozen no-overwrite atomic transaction.

## Execution identity and timing

- Source commit: `e22099eae2893fa3bb6123f1ff53872238f7c5e4`
- Authorization SHA256: `4510cf2849edf3b0478030453b77faa1e0348f245b7e6703232d661c062f4539`
- Runner SHA256: `77a3e69f8454502e0a56e742b27cfda50e5046b961f5d5ad3636d19c1e7410af`
- Coordinate-policy config SHA256: `58f15a9e39fcd3469ec656ef98c72ad6e42b8a3eab16fcbc24c4345cc4337d88`
- Coordinate artifact publisher SHA256: `a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf`
- Total runtime: `44.46511643800113` seconds
- Segmentation and coordinate generation: `31.544678085998385` seconds
- OpenSlide header and one mask read: `4.200526618000367` seconds
- WSI prehash and final hash verification: `7.663288233001367` seconds total
- Atomic publication and validation: `0.056765004999761004` seconds

The official HEALNet checkout remained clean at
`28ba5da6ab99fd8069972c22e986d83edb658dd4`; the frozen BLCA tag remained at
`df7cf2bda783ab6cc09e95d6a1fa0914da05a433`. The pre-existing, unrelated user
edit in `reports/blca_one_patient_multiscale_pilot.md` was preserved and not
touched by execution or this result recording.

## Required stop and next decision

The Q75 coordinate gate reached its required stop. Confirmed zero:

- level-0 or level-1 patch reads, thumbnails, and associated-image reads
- patch extraction or resampling
- ResNet50 inference, feature tensors, or `.pt` generation
- HEALNet execution, backward passes, optimizer steps, or training
- GPU or CUDA work
- Q25, Q50, BLCA, or official HEALNet modification
- full-cohort and Google Drive operations
- raw WSI, pre-existing, or final-artifact deletion

These files are coordinate artifacts only; they are not features or a model
prediction. The next decision is a separate review and explicit authorization
before any Q75 patch reads or GPU ResNet50 feature extraction.
