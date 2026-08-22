# BRCA Q50 scale and coordinate policy pre-execution gate

Status: `BRCA_Q50_SCALE_AND_COORDINATE_POLICY_CPU_VERIFIED_EXECUTION_LOCKED`

Q50 now has a slide-specific, deterministic scale and coordinate policy. The
policy and its artifact contract passed focused and full CPU test suites. This
stage did not reopen the WSI, read a pixel, create a real coordinate file,
extract a feature, run HEALNet, process Q75, use Google Drive, or train.

## Where Q50 is now

The exact matched case is `TCGA-AR-A1AW`, backed by GDC UUID
`5c1216f3-19ec-4d3c-9bb0-9bd740b79f62`. Its downloaded SVS previously passed
size, MD5, SHA-256, exact slide-ID, and header-only OpenSlide checks. Its Omic
row is exactly matched and contains RNA `[1,1,1558]`, mutation `[1,1,21]`, and
CNV `[1,1,1333]`.

The Q50 physical branches are now fixed as follows:

| Branch | Source | Later patch operation | Effective MPP |
|---|---|---|---:|
| scale 2x | level 0, `512×512` footprint | Lanczos to `256×256` | `0.4936` |
| scale 4x | native level 1, `256×256` | no scale resampling | `0.9872151105124595` |

Both are within about 1.28% of the supervisor's approximate 0.5 and 1.0 MPP
targets. These are Q50-specific values; Q25's numerical metadata was not
copied.

## Coordinate rules now fixed

- One shared tissue mask at native level 2 (`6,247×4,083`).
- Pinned HSV-saturation/CLAM-compatible tissue thresholds and the reviewed
  CLAM commit `26e0b6c4873e112f1ccd74cd834894c4ab7a2934`.
- Contours are mapped to level 0 with independent x/y dimension ratios, not a
  single scalar approximation.
- Scale 2x uses a global `(0,0)` level-0 lattice with 512-pixel footprints and
  512-pixel steps.
- Scale 4x uses level-0 origins with declared 1,024-pixel footprints and steps;
  later reads are native level-1 `256×256` regions.
- Incomplete footprints are rejected; coordinates are unique `int64 [N,2]`
  rows ordered by y then x.
- The policy is a documented engineering extension. It is not claimed to be
  exact paper coordinate reproduction or exact released-CLAM execution.

The reviewed segmentation/lattice implementation is reused by source hash,
while Q50 supplies its own identity, dimensions, ratios, physical scales, and
contracts. A change to that dependency must fail the later real-execution
gate.

## Verification

- Q50 focused tests: `17/17` passed.
- Full pilot suite: `335/335` passed.
- Python compilation and Git diff checks passed.
- The generic atomic coordinate format was published and revalidated only
  with synthetic arrays in a temporary test directory. No test artifact was
  retained.
- The frozen BLCA tag and official HEALNet checkout remain unchanged.
- The unrelated user edit in `reports/blca_one_patient_multiscale_pilot.md`
  was not touched or staged.

## Current stop boundary

Real Q50 coordinate generation is still locked. The next transition, if
separately authorized, is deliberately narrow:

1. reverify the exact Q50 SVS and all policy/source hashes;
2. make one level-2 mask read;
3. generate the two coordinate bags on CPU;
4. atomically publish and independently validate `Q50.coordinates`;
5. stop before level-0/level-1 patch reads, ResNet50, HEALNet, Q75, or training.

The real coordinate counts are intentionally `NOT_YET_KNOWN`; synthetic test
counts are not scientific results.
