# BRCA supervisor cohort decision

Decision date: 2026-08-18 UTC

## Recorded direction

The supervisor replied:

> Go with brca, blca

This resolves the cohort-selection question as follows:

- **BLCA** remains the frozen, validated one-patient reference at tag
  `blca-one-patient-pilot-v1`.
- **BRCA** is the approved next cohort for the supervisor-aligned
  WSI/RNA/mutation/CNV extension.
- **KIRP** is outside this phase because the released KIRP Omic artifacts have
  no CNV columns.

## Scope of this approval

The recorded reply authorizes CPU-only BRCA preparation: source verification,
exact row-level alignment, Omic-group validation, multiscale policy design,
tests, storage planning, and a proposed three-patient pilot list.

It does **not** by itself authorize:

- downloading any BRCA WSI;
- opening or processing WSI pixels;
- feature extraction;
- HEALNet training;
- processing any multi-WSI patient; or
- full-cohort acquisition.

The exact three-patient UUID list and the proposed physical-scale policy must
be reviewed before a separate, explicit pilot-execution approval is requested.
Initial acquisition and extraction concurrency will be one.

Status: `BRCA_CPU_PREPARATION_AUTHORIZED`

## Phase 1 tensor interpretation

The subsequent supervisor exchange clarified that `N` refers to the set of
WSI samples, not to one fixed patch count shared by every slide. Consistent
with the paper's per-patch 2,048-dimensional features and the official code's
one-file-per-slide behavior, Phase 1 represents the cohort as a ragged set of
per-WSI bags:

```text
WSI i: [P_i,2048]
one-WSI HEALNet batch: [1,P_i,2048]
cohort: {F_i}_{i=1..S}
```

There is no per-WSI averaging to `[1,2048]`, and patches from separate patients
are not concatenated into one unlabeled tensor. Initial execution uses batch
size one, excludes multi-WSI patients, and requires patient-grouped splits.
This interpretation is frozen as `SUPERVISOR_TENSOR_POLICY_V1`; it does not
expand the earlier approval to WSI download, extraction, or training.
