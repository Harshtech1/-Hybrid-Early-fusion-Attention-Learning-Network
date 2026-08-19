# BRCA Q75 scale-policy approval transition

Status: `BRCA_Q75_SCALE_POLICY_APPROVED_CPU_RECORDED_EXECUTION_LOCKED`

The user's exact response, **“OK CONTINUE PLEASE”**, is recorded as approval
only of the scale mapping proposed immediately before that reply. This is a
CPU-only policy transition. It neither accesses pixels nor authorizes the next
execution stage.

## Exact evidence binding

The transition is bound to the completed Q75 header-only result for patient
`TCGA-E2-A154`, slide
`TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`, and GDC
UUID `25aec062-60d1-446e-a1c6-0c79cc74a770`.

- exact user statement SHA256:
  `70e658b86380c6b3c86fd0f980937ecb081e38d51ed3b7056cc534f9cb95e3e3`
- header result SHA256:
  `08a7ed3e67ddf17513ee2dbda2adfd2398333787aaa75fe9eacf911f3c1a3898`
- header report SHA256:
  `9ed50ecc8464109e0a8ca121082462f8765b1c70663499180cc644cfe604d985`
- header-gate source commit:
  `ef6c8921d993567c8178fc932f7315378714ca5a`
- commit containing the frozen header result:
  `c7e98f4ce2663556be9b487441fe36494364ff18`

The pure evaluator also pins the exact WSI size, MD5, SHA256, patient, slide,
UUID, native MPP, all pyramid dimensions, and every level-downsample value.
Any drift is rejected.

## Approved mapping

| Branch | Source | Declared operation if separately authorized later | Effective MPP | Per-axis error |
| --- | --- | --- | ---: | ---: |
| scale 2x | level 0, `512×512` source footprint | resize to `256×256` with `PIL.Image.Resampling.LANCZOS` | `0.4936` | `1.28%` |
| scale 4x | native level 1, `256×256` | no resampling | `0.9872163682185965` | `1.2783631781403515%` |

Both mappings remain within the established 10% per-axis tolerance. The
effective values are reported as measured engineering scales; neither is
mislabeled as exactly 0.5 or exactly 1.0 MPP.

## Fail-closed implementation

The new evaluator accepts evidence identities and already-recorded scalar
header metadata only. It has no WSI path, slide object, image input, network
client, or artifact-write API. It cannot open a WSI, read a pixel, resize an
image, generate coordinates, or run a model. Focused verification passed
`22/22` tests, including evidence drift, identity drift, pyramid drift, and
execution-boundary checks. The complete CPU pilot suite passed `485/485`
tests.

## Current stop boundary

This transition performed zero WSI opens, pixel/region reads, resampling
executions, masks, coordinates, patches, ResNet50 calls, HEALNet calls, GPU
operations, Drive operations, raw-file deletions, cohort operations, or
training runs. Q25, Q50, BLCA, and the official HEALNet checkout were not
modified by this transition.

The next possible stage is a separate review of a Q75 coordinate-generation
policy. Pixel access and coordinate generation remain unauthorized until that
stage receives explicit approval.

## New implementation files

- `multiscale_feature_pilot/config/brca_q75_scale_policy.yaml`
  (`d29be0892e0b0324ae9b4390a1db9a9ae4b5a60b4541ddb7a36c81b8d2bca6b5`)
- `multiscale_feature_pilot/src/brca_q75_scale_policy.py`
  (`3aecb1f3818f9ae98708cdf61f6ccf4b938ffe5fe78bbbaff6e11896e5eb4482`)
- `multiscale_feature_pilot/tests/test_brca_q75_scale_policy.py`
  (`77d3d6ccc718a498baa29cea4c621cffaea79e96fdc905663daef1cc506d1431`)
- `multiscale_feature_pilot/provenance/brca_q75_scale_approval.yaml`
- `reports/brca_q75_scale_approval.md`
