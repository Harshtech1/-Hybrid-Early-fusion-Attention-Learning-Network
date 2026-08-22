# P0002–P0008 GPU feature-package preparation

Status: **CPU-only preparation and synthetic validation complete; GPU execution remains locked.**

The seven verified coordinate artifact sets were read and validated without
opening a WSI or importing a model runtime. Each package binds the frozen
cohort identity, Omic row, coordinate manifest, coordinate counts, ImageNet1K
V2 ResNet50 checkpoint, natural feature layout, compact artifact contract, and
recovery-v2 stage contract.

| Patient | 2× rows | 4× rows | Feature tensor | Float32 payload |
|---|---:|---:|---:|---:|
| P0002 | 9,785 | 2,486 | `[12271,2048]` | 100.52 MB |
| P0003 | 3,461 | 921 | `[4382,2048]` | 35.90 MB |
| P0004 | 3,933 | 1,034 | `[4967,2048]` | 40.69 MB |
| P0005 | 23,559 | 5,971 | `[29530,2048]` | 241.91 MB |
| P0006 | 7,505 | 1,962 | `[9467,2048]` | 77.55 MB |
| P0007 | 9,238 | 2,407 | `[11645,2048]` | 95.40 MB |
| P0008 | 18,877 | 4,799 | `[23676,2048]` | 193.95 MB |

Each future HEALNet WSI input is simply `[1,P_i,2048]`; no pooling or
transpose is permitted. Every package completed a synthetic recovery-v2 replay
through `FEATURES_VERIFIED` with 14 immutable events and a valid compact
artifact evidence contract. No real ledger, feature artifact, or output path
was written.

## Consolidated P0001–P0008 GPU session

The plan uses a single serial GPU worker and preserves each patient as an
independent atomic/recoverable transaction. It covers **112,754 patch reads**,
including P0001, with a raw float32 payload of **923,680,768 bytes**. The
estimated final compact storage is **929.3–934.0 MB** and the estimated T4 GPU
wall time is **1,210–1,800 seconds** (about **20–30 minutes**), excluding
machine startup and any external interruption.

This is an estimate derived from the measured engineering pilots and exact row
counts, not a completed GPU measurement. GPU remains locked until the separate
consolidated authorization is received and recorded.

The next GPU run must use ImageNet1K V2 ResNet50 as selected for the
engineering track. It must not claim exact paper reproduction because the
paper describes Kather100K pretraining.
