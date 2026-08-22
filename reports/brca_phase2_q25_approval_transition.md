# BRCA Phase 2 Q25 approval transition

Status: `BRCA_Q25_ACQUISITION_READY`

The user reported the supervisor's approval of the exact previously presented
request. The approved three candidates remain sequential, with concurrency
one. Only Q25 is executable now; Q50 and Q75 remain locked until the Q25
size/MD5 and MPP/pyramid report has been reviewed.

The local gate now reports:

```text
cpu_preflight_ready: true
frozen_sources_verified: true
acquisition_authorized: true
ready_to_download: true
current_download_scope: Q25_ONLY
q50_q75_locked: true
blockers: []
```

No network request, GDC dataset download, WSI opening, pixel read, coordinate
generation, feature extraction, HEALNet execution, or training occurred while
recording this transition.

The complete pilot test suite passed `222/222`, and `git diff --check` passed.

## Current authorized object

| Field | Exact value |
|---|---|
| Label | `Q25` |
| Patient | `TCGA-LL-A6FP` |
| GDC UUID | `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6` |
| Filename | `TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs` |
| Declared bytes | `648,046,947` |
| Expected MD5 | `75536393096ffd928bc35ec9503c3655` |
| Authorized manifest SHA256 | `95bb1c4491497c265b868e96698cef3c3dd501458f801dae3c1dd702f3efa297` |
| Authorization-record SHA256 | `8c6971df846a1dcbebfd83f689cd6496c8a85dcbf3b8eb74781cb7a85849aaea` |

The authorization record cryptographically binds the exact authorization
config (`a3a0704b...`), approved metadata policy (`daffdd77...`), and canonical
approval identity (`cf2ba45a...`). The builder fails closed if any decision is
pending, reverted, or byte-tampered.

The authorized manifest directory contains exactly one standard GDC TSV and
one authorization record. The original three `NOT_AUTHORIZED` manifests are
retained unchanged as frozen candidate provenance; they are not download
inputs.

Concurrency one means one patient/one WSI manifest is processed at a time. It
does not mean forcing the GDC client's internal transfer-thread flag to one;
the client default remains unchanged.

## Approved metadata boundary

After Q25 has matched both the exact declared size and MD5, it may be opened
only to collect `mpp_x`, `mpp_y`, `level_dimensions`, and
`level_downsamples`. The approved evaluator selects distinct native pyramid
levels nearest approximately 0.5 and 1.0 micrometers per pixel, requires no
more than 10% relative error on each axis, and rejects invalid, ambiguous, or
out-of-tolerance metadata. Silent resampling is prohibited.

`read_region`, associated-image pixel access, coordinate generation,
ResNet50 extraction, HEALNet execution, training, and automatic progression
to Q50/Q75 remain prohibited. The required stop point is the Q25 size, MD5,
and MPP/pyramid report.

Immediately before the future network operation, the live Lightning quota,
local free space, and empty staging boundary must be checked again. Google
Drive mounting is not required for this Q25 local staging step.
