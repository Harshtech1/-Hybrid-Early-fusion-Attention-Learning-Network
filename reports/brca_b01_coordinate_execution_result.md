# BRCA Validation Batch B01 — Coordinate Execution Result

Status: `BRCA_B01_COORDINATES_VERIFIED`

The authorized CPU-only B01 coordinate gate completed successfully. It performed exactly one OpenSlide level-2 read at `(0,0)` with size `3,986×2,456`, generated both coordinate bags using the frozen policy, published them atomically, and passed an independent artifact-only validation.

## Result matrix

| Metric | 2× branch | 4× branch |
|---|---:|---:|
| Coordinate count | 3,773 | 969 |
| Shape | `[3773,2]` | `[969,2]` |
| Dtype | `int64` | `int64` |
| Unique rows | 3,773 | 969 |
| Minimum `(x,y)` | `(6144,3584)` | `(6144,3072)` |
| Maximum `(x,y)` | `(58368,33792)` | `(57344,33792)` |
| HDF5 bytes | 70,040 | 25,176 |

The total future WSI token count for B01 is therefore:

\[
P_{B01}=3{,}773+969=4{,}742.
\]

The mask SHA256 is `093b00e4c2aa271b01fae4754631638f63758ff8084ba51b74d6adecd6cf7427`. Segmentation retained one foreground contour and no qualifying holes.

## Artifact integrity

- Directory: `/teamspace/studios/this_studio/brca_pilot_data/BRCA_BATCH_B01.coordinates`
- Manifest SHA256: `ff1eebc9fa51128cdda6294fa48cf34008b203db236acf59e8570725b3fc4c8b`
- 2× coordinate-content SHA256: `19bacb5413e57a328d4c1a3a7c3f2830786b0e494e20947e582cc09bec95cf26`
- 4× coordinate-content SHA256: `ee10e5ea1e4dfa9b2af8ebcd794cc8b5c118f077ce7b7b881062ea184065451f`
- Exact files: manifest, sidecar, and two HDF5 coordinate bags
- Total retained bytes: 100,270

Publication used a unique same-filesystem staging directory followed by Linux `RENAME_NOREPLACE`. To honor the no-deletion boundary, no lock-file unlink or staging cleanup operation was used. The successful rename moved the staging directory to its final name, leaving no staging path.

The initial invocation failed closed during Git preflight because of status-string normalization. It occurred before any WSI open or pixel access and created no artifact. The corrected, committed gate then completed once.

## Boundary

Patch reads, ResNet50, HEALNet, feature generation, GPU/CUDA, B02–B06, Q25/Q50/Q75 or BLCA modification, Drive, deletion, cohort expansion, and training were not performed.

The required stop has been reached. B01 patch reading or feature extraction requires a new, separate authorization.
