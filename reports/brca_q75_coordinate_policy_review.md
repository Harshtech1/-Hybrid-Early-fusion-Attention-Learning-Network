# BRCA Q75 coordinate-policy design and review

Status: `BRCA_Q75_COORDINATE_POLICY_REVIEWED_CPU_ONLY_EXECUTION_LOCKED`

The Q75 coordinate policy is now resolved as a scalar, CPU-only design. No coordinates were generated, the WSI was not opened, and no image, mask, or artifact was created. Actual tissue-selected patch counts therefore remain unknown.

## Exact evidence boundary

The review is bound to patient `TCGA-E2-A154`, slide
`TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`, GDC UUID
`25aec062-60d1-446e-a1c6-0c79cc74a770`, and the verified Q75 WSI SHA256
`844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37`.
It also binds the exact Omic row `771`, the frozen header result at commit
`c7e98f4ce2663556be9b487441fe36494364ff18`, and the frozen scale approval at
commit `42db4dd4b402b3b7c32970f96db2f8d0c5f46180`.

The user's exact instruction has SHA256
`1dd1a3f8d74e6ecb276748ba0db193811e88b8fde2dae387961fb5719cfbd109`.
Any evidence, identity, pyramid, Omic-shape, predecessor-file, or reviewed
dependency drift is rejected.

## Reviewed mask design

If a later, separately authorized coordinate execution occurs, it will read
one shared level-2 mask at `6783×5654`. The level-2-to-level-0 geometry ratios
are derived independently per axis as `(16.0, 16.001238061549344)`; the
OpenSlide scalar downsample is retained only as physical-MPP evidence.

The future mask algorithm is the reviewed Q25/Q50 custom engineering policy:

- ignore alpha, convert RGB to HSV, and use the saturation channel;
- apply median blur with kernel `7`, fixed threshold `8`, no Otsu, then a
  `4×4` morphological close;
- use `RETR_CCOMP` and `CHAIN_APPROX_NONE`;
- calculate `scaled_reference_area = 1023`;
- retain foreground only when contour area minus all direct-hole areas is
  greater than `102,300` mask pixels²;
- sort direct holes by descending area, retain at most eight, and keep only
  holes with area greater than `16,368` mask pixels²;
- scale contour x/y independently to level 0, then cast to `int32`;
- accept a patch if any four-point-easy probe is on or inside a contour
  (`pointPolygonTest >= 0`), and reject it only when its center is strictly
  inside a retained hole (`> 0`). A hole boundary is retained.

The level-2 image would contain `38,351,082` pixels and exactly `153,404,328`
bytes as contiguous RGBA `uint8`; these are arithmetic design values, not an
allocation or observed runtime measurement.

The compatibility pin is CLAM commit
`26e0b6c4873e112f1ccd74cd834894c4ab7a2934`, via the reviewed source
`multiscale_feature_pilot/src/brca_q25_coordinates.py` with SHA256
`da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82`.
As already recorded in `shared/provenance/known_issues.md`, this is a
historically aligned compatibility pin; it is not proven to be the exact CLAM
commit used by the paper authors.

## Reviewed branch geometry

Both future branches use a custom global level-0 lattice anchored at `(0,0)`,
reject incomplete footprints, use the same tissue geometry, and would return
unique row-major `(x,y)` origins in level-0 coordinates.

| Branch | Future source and operation | Level-0 footprint / step | Full-slide sites before tissue filtering | Last origin | Right/bottom remainder |
| --- | --- | --- | ---: | --- | --- |
| scale 2x | level 0 `512×512`, later Lanczos to `256×256` | `512×512` / `512×512` | `37,136` (`211×176`) | `(107520,89600)` | `(496,359)` |
| scale 4x | native level 1 `256×256`, no resampling | `1024×1024` / `1024×1024` | `9,240` (`105×88`) | `(106496,89088)` | `(1008,359)` |

The scale-4x footprint uses CLAM-compatible cast-before-multiply geometry:
`256*int(ds_x), 256*int(ds_y)`. Because Q75's level-1 coordinate ratios are
`(4.0, 4.000132643586682)`, this yields `1024×1024`. It is a custom
constant-step engineering lattice, not exact native-level grid inversion,
released-CLAM coordinate reproduction, or exact paper preprocessing.

The counts in the table are only theoretical full-slide lattice capacities
before tissue filtering. They are not expected or measured patch counts.

## Future artifact contract

A separately authorized execution would have two `int64 [N,2]` bags in order
`scale_2x`, then `scale_4x`, with columns `(x,y)`, nonempty/unique checks, and
row-major `(y,x)` ordering. Publication would use the pinned
`BRCA_COORDINATE_ARTIFACT_SET_V1` schema and sibling staging followed by Linux
`RENAME_NOREPLACE`; overwrite and resume remain prohibited.

No artifact API is exposed by the policy core, and no artifact was written.

## Verification and stop boundary

The pure reviewer accepts only evidence identities and scalar metadata. It has
no path, OpenSlide, pixel/image, coordinate-array, CUDA, network, or write API.
The focused policy suite passed `41/41` tests, and the complete pilot suite
passed `526/526` tests. Exact results are also recorded in
`multiscale_feature_pilot/provenance/brca_q75_coordinate_policy_review.yaml`.

This transition performed zero WSI opens, pixel/region reads, masks,
coordinates, coordinate publications, resampling operations, patches,
ResNet50 calls, HEALNet calls, GPU work, Drive operations, raw-file deletions,
cohort operations, Q25/Q50 modifications, BLCA modifications, or training.

The next possible gate is Q75 coordinate execution. It requires separate explicit authorization before one level-2 mask read, tissue segmentation, coordinate generation, or artifact publication can occur.
