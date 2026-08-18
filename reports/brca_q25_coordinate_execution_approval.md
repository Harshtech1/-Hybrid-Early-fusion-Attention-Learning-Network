# BRCA Q25 coordinate-execution approval

Recorded at: 2026-08-18T21:32:09Z

Status: `Q25_COORDINATE_EXECUTION_AUTHORIZED_CPU_ONLY`

The user's relayed supervisor approval and subsequent instruction to continue
are bound to a narrow Q25 execution boundary. They authorize the exact Q25
level-2 tissue-mask read, deterministic two-branch coordinate generation,
atomic publication outside Git, validation, and reporting.

This record does not authorize Q50 or Q75, level-0 or level-1 patch-image
reads, ResNet50 feature extraction, HEALNet execution, training, full-cohort
processing, or Google Drive operations.

## Exact execution boundary

- WSI: `TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs`
- UUID: `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6`
- verified size: `648046947` bytes
- verified MD5: `75536393096ffd928bc35ec9503c3655`
- verified SHA256: `ac852be806eb0d91214145983319b604919a8e1d16bd59378a3dba69a600979c`
- maximum real mask reads: one
- mask read: OpenSlide level 2, location `(0,0)`, size `4108×4212`
- coordinate output: `/teamspace/studios/this_studio/brca_pilot_data/Q25.coordinates`
- required stop: after coordinate validation and report

The execution authorization is stored in
`multiscale_feature_pilot/config/brca_q25_coordinate_execution_authorization.yaml`
with SHA256
`3797f428f5d1d49334fc2c0665325728318083d5deb8831deec5ad1f560ac617`.

## Frozen coordinate rules

- Both branches use the same Q25 level-2 tissue contours.
- The mask uses the pinned CLAM thresholds and contour predicates recorded in
  `APPROVED_BRCA_Q25_COORDINATE_POLICY_V1`.
- The approximately 0.5-MPP branch uses a global level-0 `512×512` footprint
  and step. Its later patch path will resize to `256×256` using Lanczos,
  producing an effective `0.505 MPP`.
- The approximately 1.0-MPP branch uses a custom global level-0 `1024×1024`
  footprint and step, followed later by a native level-1 `256×256` read. Its
  approved physical scale is approximately `1.010015 MPP`.
- Both lattices use anchor `(0,0)`, reject incomplete footprints, return
  level-0 `(x,y)` coordinates, require `[N,2] int64`, require non-empty unique
  rows, and use row-major `(y,x)` ordering.

The second branch is CLAM integer-cast geometry compatible, but neither branch
is claimed to reproduce released CLAM coordinates or an exact paper
implementation. This is a supervisor-aligned, explicitly documented
engineering policy.

At the time this approval record was written, no WSI was opened, no pixels
were read, and no coordinates or artifacts had been generated.
