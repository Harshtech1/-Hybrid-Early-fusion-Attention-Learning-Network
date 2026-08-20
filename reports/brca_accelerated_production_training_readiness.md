# Accelerated BRCA production and training readiness

Status: **CPU packages ready for execution review; no production operation or
training is authorized yet**.

## Executive conclusion

The project has moved beyond scientific pilot selection. The exact first eight
rows of the frozen 894-patient cohort are now the first production block. No
additional B04/B05/B06 pixel pilot is required. B06 remains a retained
header-only reference and its raw WSI is not eligible for deletion.

The production control plane, recovery-v2 ledger, exact first-eight request
manifests, survival endpoint, patient split, training-only bin cutpoints,
HEALNet training contract, checkpoint/recovery layer, and locked-test result
gate are ready for execution review. The next useful work is production
preprocessing—not another architecture rehearsal.

## What is complete

| Component | Exact result | State |
|---|---|---|
| Cohort | 894 exact singleton case/slide matches | Frozen |
| Raw inventory | 918,532,189,383 bytes | Frozen |
| Production order | SHA256 `1c97fa…af5a` | Frozen |
| First production block | Cohort rows 1–8; 8,297,129,620 bytes | Packaged, locked |
| Endpoint | `survival_months`; event = `1-censorship` | Finalized |
| Training split | 660 patients; 79 events | Finalized |
| Validation split | 116 patients; 14 events | Finalized |
| Locked test | 118 patients; 33 events | Finalized |
| Training-only cutpoints | 22.505, 44.84, 67.940 months | Finalized |
| Model inputs | WSI 2048, RNA 1558, mutation 21, CNV 1333 | Frozen |
| Checkpointing | Model/optimizer/scheduler/RNG/history each epoch | Validated synthetically |
| Locked test | One-use start marker + atomic result | Validated synthetically |
| CPU regression | 765 tests passed | Pass |

The split manifest contains all 894 patients exactly once. It preserves the
source outer partition (`train=1` development, `train=0` locked test), creates a
deterministic 85:15 development split within censoring strata, and keeps all
preprocessing and checkpoint selection independent of the locked test.

## Production path from here

```mermaid
flowchart LR
  A[Authorize exact rows 1-8 acquisition] --> B[Serial patient production]
  B --> C[Header + scale policy]
  C --> D[One bounded CPU mask read + coordinates]
  D --> E[T4 ResNet50 ImageNet1K V2 features]
  E --> F[Compact artifact validation + ledger terminal]
  F --> G[Formal block review]
  G --> H[Continue exact rows 9-894 serially]
  H --> I[894-row compact feature registry]
  I --> J[Authorize full HEALNet training]
  J --> K[660 train / 116 validation]
  K --> L[One locked-test evaluation: 118]
```

The first block is operational production and supplies a natural throughput and
storage checkpoint. It is not a new scientific pilot and should not delay the
cohort with a new architecture decision. Patient concurrency, download
concurrency, and GPU concurrency remain one; recovery happens at patient/stage
boundaries.

## Frozen training contract

For patient \(i\), the four separate HEALNet inputs are

\[
X_i^{WSI}\in\mathbb{R}^{1\times P_i\times2048},\quad
X_i^{RNA}\in\mathbb{R}^{1\times1\times1558},
\]

\[
X_i^{MUT}\in\mathbb{R}^{1\times1\times21},\quad
X_i^{CNV}\in\mathbb{R}^{1\times1\times1333}.
\]

The pinned BRCA HEALNet configuration has four outputs, depth 2, 17 latents,
latent dimension 126, cross-head dimension 63, and latent-head dimension 20.
For logit \(z_{ik}\),

\[
h_{ik}=\sigma(z_{ik}),\qquad
S_{ik}=\prod_{j=0}^{k}(1-h_{ij}),\qquad
r_i=-\sum_{k=0}^{3}S_{ik}.
\]

Optimization is Adam (`lr=0.007765016508403882`) with OneCycleLR
(`max_lr=0.008`), the released L1 coefficient, true gradient accumulation over
16 patients, float32 only, and at most 50 epochs. Early stopping uses validation
NLL only with patience 5. AMP, TF32, CPU fallback, test-based early stopping,
and test-based checkpoint selection are prohibited.

Each epoch is an append-only atomic checkpoint. Recovery binds the run UUID,
committed source, authorization, split, cutpoints, future 894-row feature
registry, protocol, and official HEALNet commit. After best-checkpoint
selection, an irreversible marker consumes the one allowed locked-test
evaluation; a crash after that marker stops for review rather than silently
rerunning the test.

## Paper and supervisor alignment

The architecture and tensor direction follow the supervisor's instruction:
all cohort WSIs are processed into patient-local multiscale bags, and HEALNet
receives WSI, RNA, mutation, and CNV as separate modalities. The accelerated
route therefore remains on the agreed engineering path.

It is not exact paper reproduction. The engineering encoder is ImageNet1K V2,
whereas the paper describes Kather100K pretraining, and this implementation
uses three separately supplied Omic blocks. Results must be labelled
**supervisor-aligned engineering implementation**, not exact reproduction.
This disclosure does not block the requested production route.

## Time and GPU position

No GPU is needed for the current review. GPU becomes necessary during each
patient's production feature stage and later for HEALNet training. Existing
Q25/Q50/Q75 measurements project approximately 39–61 T4 hours for feature
extraction across 894 patients, plus downloads and CPU coordinate work. The
first production block is expected to consume roughly 21–33 T4 minutes based
on those measured rates, but its actual patch counts are not known before
coordinate generation.

A defensible full-training wall time cannot be stated before the complete
feature registry exists because \(P_i\) controls HEALNet attention work. No
extra timing pilot is recommended. Record production epoch 1 as the timing
benchmark, then continue the same checkpointed run; this avoids another delay.

## Remaining authorizations and decisions

1. **First production block acquisition:** authorize exact serial download and
   header processing for frozen rows 1–8 using the eight request-only manifests.
2. **Patient pixel stages:** after each header is known, authorize its exact one
   mask read and coordinate publication; after counts are known, authorize its
   exact T4 feature extraction and compact publication.
3. **Rows 9–894 release:** after formal row-8 review, authorize continuation in
   frozen order with the same one-patient recovery contract.
4. **Raw lifecycle:** choose a verified recovery source. First-eight raw WSIs
   and B06 remain retained. Any future release needs a separately authorized,
   identity-bound single-patient action after strict compact validation.
5. **Training:** once all 894 compact artifacts form a validated hashed feature
   registry, authorize one exact training run UUID, committed source, output
   roots, one CUDA device, and the frozen protocol.

No download, WSI open, pixel read, coordinate generation, CUDA operation,
feature extraction, HEALNet forward pass, checkpoint write, deletion, Drive
operation, or training occurred while preparing this package.
