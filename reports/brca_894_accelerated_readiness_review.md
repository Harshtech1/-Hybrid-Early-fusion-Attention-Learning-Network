# BRCA 894-patient accelerated readiness review

Status: `CPU_REVIEW_COMPLETE__FULL_COHORT_NOT_YET_READY_OR_AUTHORIZED`

## Executive outcome

The engineering feature path is mature enough to stop repeating middle-range
pilots. Q25, Q50, Q75 and compact B01, B02, B03 all support the same two-scale,
2,048-wide ImageNet1K V2 feature contract, and B01–B03 prove the four-file
compact publisher on real feature tensors. Skipping B04 and B05 is therefore a
reasonable schedule decision. B06 should still receive a header-only review
because its declared 1.69 GB WSI is the planned 90th-percentile tail sample.

The 894-patient loop should **not** start immediately after B06. Two production
gaps remain: the current ledger only records successful ordered milestones and
does not encode failure/retry events, and no raw-WSI lifecycle has been approved
that can reconcile 918.53 GB of raw data with a 200 GB quota.

## Completed and ready

| Area | Evidence-backed state |
|---|---|
| Cohort identity | 894 exact singleton case-plus-slide matches; 62 ambiguous multi-WSI cases excluded |
| WSI feature policy | Approximately 0.5 and 1.0 µm/px branches; 2× rows followed by 4× rows |
| Encoder | ResNet50 ImageNet1K V2, classifier removed, 2,048 features per patch |
| HEALNet interface | Per-patient WSI `[1,P_i,2048]`; RNA `[1,1,1558]`, mutation `[1,1,21]`, CNV `[1,1,1333]`; no padding/mask |
| Compact representation | One `[P_i,2048]` float32 tensor, exact row provenance, manifest and sidecar |
| Compatibility | Q25/Q50/Q75 are losslessly compact-representable; calculated duplicate-tensor saving is 49.84% |
| Real compact pilots | B01, B02 and B03 atomically published and independently validated |
| Image encoder decision | ImageNet1K V2 selected; the paper's Kather100K difference remains an explicit limitation |

### Real compact-pilot measurements

| Pilot | Patch rows | Complete GPU pilot | Compact artifact | HEALNet numerical smokes |
|---|---:|---:|---:|---|
| B01 | 4,742 | 73.46 s | 39,092,076 bytes | PASS |
| B02 | 9,020 | 124.94 s | 74,342,370 bytes | PASS |
| B03 | 11,132 | 118.41 s | 91,770,745 bytes | PASS |

These are engineering/interface tests with randomly initialized HEALNet, not
trained survival results.

## Accelerated validation disposition

- B01–B03: complete through compact GPU artifact validation.
- B04–B05: skip under the accelerated schedule; this is a planning decision,
  not evidence that those slides were inspected.
- B06: perform only acquisition verification and header-only MPP/pyramid
  inspection first. Decide from those facts whether it adds new scale-policy
  coverage before authorizing a mask read, coordinates, or GPU work.

B06 is `TCGA-EW-A1OY`, UUID
`00e926c5-d65d-42e5-8ba4-afaab66db47c`, with declared size
1,691,494,753 bytes. Its patch load and runtime remain unknown.

## Restart and recovery audit

The implemented ledger already has valuable fail-closed properties: immutable
numbered JSON records, a SHA256 hash chain, exclusive creation, exact stage
ordering, patient/slide identity continuity, rejection of partial or unexpected
files, and a terminal state that cannot advance.

Its current record model is nevertheless a successful-stage ledger, not a full
operational event ledger. It has no explicit failure event, retry attempt,
`run_id`, direct GDC UUID field, error class/message, input/output hash maps, or
replay-to-retry action. There is also no generic executor that connects download,
header, coordinate, GPU, compact publication and ledger replay for arbitrary
rows of the 894-patient cohort. These must be implemented and synthetically
tested before unattended cohort work.

The authoritative evidence for this conclusion is the frozen config,
provenance and tested source. The current working copy of
`reports/brca_compact_artifact_and_recovery_design.md` was already modified and
contains pasted instruction text; this audit did not edit or treat it as frozen
evidence.

## Recommended restartable batch

The restart-safe transaction size remains **one patient**:

- one active download;
- one active raw WSI;
- one coordinate transaction;
- one GPU extraction;
- one compact atomic publication; and
- one terminal ledger record before the next patient.

To reduce administrative effort, authorize and report in blocks of **eight
serial patients**, while retaining one-patient concurrency and validation after
every patient. At Q-pilot rates, an eight-patient block is estimated at
20.8–32.7 T4 minutes and 0.61–1.12 GB of compact features. The first mismatch
stops the block. An eight-patient block is not an eight-patient tensor batch and
does not permit concurrent WSIs.

## Time and storage estimates—not guarantees

The conservative Q25/Q50/Q75 planning scenarios project 38.69–60.88 T4
GPU-hours for 894 patients, with a mean-rate scenario of 47.38 hours. Naively
scaling all six completed feature pilots gives 18.24–60.88 hours, but the low
end comes from lower-quantile B01–B03 slides and is not a defensible capacity
floor. Downloads, CPU coordinates, failures, retries and interactive-session
interruptions are not included.

The frozen compact projection is 68,694,848,250–124,899,345,360 bytes, with a
mean-Q scenario of 91,048,305,348 bytes. These are workload estimates. Every
patient must still pass a live storage check covering raw input, staging, final
artifact and the mandatory 20 GB free-space floor.

The raw singleton inventory is 918,532,189,383 bytes. Consequently, all raw
WSIs cannot remain locally alongside compact outputs. Full-cohort completion
requires an explicit, separately approved retention/deletion or recoverable
external-source policy. Feature success alone must never trigger deletion.

## Blocking decisions

1. Complete B06 acquisition/header review and decide whether B06 pixel work is
   scientifically useful.
2. Extend the ledger and generic executor with explicit failure, retry, attempt,
   replay and conflict handling; validate entirely with CPU synthetic tests.
3. Approve the raw-WSI lifecycle and exact `SAFE_TO_DELETE` prerequisites.
   That approval should define policy only; each deletion remains separately
   controlled unless explicitly authorized.
4. Keep at least 20 GB free and run an exact storage preflight before every
   patient transition.
5. Record ImageNet1K V2 as the chosen engineering encoder and disclose the
   Kather100K difference. Freeze survival splits, loss and evaluation metrics
   separately before training.

## Exact next gates

1. CPU-only B03 consolidation and B06 header-package preparation; stop before
   downloading or opening B06.
2. Separately authorize B06-only acquisition verification and OpenSlide
   header-only inspection; stop without `read_region`.
3. Review B06 geometry and choose either `SKIP_B06_PIXEL_WORK` or prepare a
   separate coordinate gate.
4. CPU-only implementation and synthetic testing of the generic executor and
   failure/retry ledger extension; stop before any cohort download or GPU use.
5. Approve raw-WSI lifecycle and storage policy.
6. Authorize the first operational release as eight serial, independently
   restartable one-patient transactions, with a stop after every patient and a
   formal review after patient eight.

No download, WSI/OpenSlide access, pixel read, coordinate generation, CUDA/GPU
operation, feature extraction, Drive operation, deletion, cohort processing or
training occurred in this review.
