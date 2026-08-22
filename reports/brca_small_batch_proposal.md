# BRCA six-patient validation-batch proposal

Status: `METADATA_ONLY__NOT_AUTHORIZED_FOR_DOWNLOAD_OR_EXECUTION`

## Purpose

This batch is intended to validate the compact artifact publisher and recovery
ledger under more varied WSI sizes before any request to process all 894
singleton patients. It excludes the completed Q25, Q50, and Q75 pilots.

## Deterministic selection

The 894 exact singleton rows are ordered by ascending declared WSI size, then
case ID and slide ID. Six target quantiles—10%, 30%, 45%, 55%, 70%, and 90%—
are mapped using `floor(q × (n−1) + 0.5)`. The selected one-based ranks are
90, 269, 403, 492, 626, and 805.

| Quantile | Rank | Patient | GDC UUID | Declared bytes | Omic row |
|---:|---:|---|---|---:|---:|
| 10% | 90 | `TCGA-GI-A2C8` | `0a886f18-c44c-4b5e-b243-6df6e27f426a` | 408,704,377 | 924 |
| 30% | 269 | `TCGA-BH-A0BG` | `c5331e5e-10b4-4979-958b-d4592a2805de` | 724,114,911 | 472 |
| 45% | 403 | `TCGA-AR-A1AY` | `266c3852-f5d0-4815-94d9-dac5b0ff8276` | 918,454,431 | 372 |
| 55% | 492 | `TCGA-D8-A4Z1` | `d39d5d1d-2999-4b82-b2d0-8fba2ccc20cc` | 1,038,326,117 | 742 |
| 70% | 626 | `TCGA-LL-A5YL` | `1e3d7e89-ee2c-46fd-8697-2e62886657fb` | 1,281,197,011 | 951 |
| 90% | 805 | `TCGA-EW-A1OY` | `00e926c5-d65d-42e5-8ba4-afaab66db47c` | 1,691,494,753 | 897 |

All six are released, exact singleton case-plus-full-slide matches. Their Omic
rows were independently loaded from the local archive and yielded finite CPU
float32 RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]` tensors.

The complete exact identities, filenames, MD5 values, and proposal locks are
in `reports/brca_small_batch_proposal.tsv`. The TSV is deliberately not a GDC
manifest and every row is marked `NOT_AUTHORIZED`.

## Planning estimates

- Declared raw total: 6,062,291,600 bytes (6.06 GB).
- Largest single raw WSI: 1,691,494,753 bytes.
- Extraction concurrency if later approved: one patient.
- Pilot-rate GPU scenario: approximately 15.58–24.52 T4 minutes total; mean
  scenario 19.08 minutes.
- Pilot-rate compact artifact scenario: approximately 461–838 MB total; mean
  scenario 611 MB.

Runtime and feature storage cannot be known from WSI file size alone. These are
planning scenarios and must not be treated as resource guarantees.

## Proposed future stopping points

If the batch is accepted, authorization should remain staged:

1. one-row acquisition and exact raw verification;
2. header-only geometry review;
3. separately approved CPU mask and coordinates;
4. separately approved GPU features into the compact schema;
5. independent artifact and ledger audit; and
6. stop before the next patient on any mismatch.

No raw deletion, Drive operation, cohort expansion, or training should be part
of the batch authorization.

## Decision requested

Accept, reject, or revise this six-patient selection. Acceptance of the
proposal does not itself authorize downloads or processing; exact staged
execution authorizations would be prepared afterward.
