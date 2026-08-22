# Supervisor-aligned HEALNet pilot

This independent repository contains the code, tests, provenance, and documentation for a patient-matched WSI + multi-omic HEALNet survival pilot. Supervisor instructions are the implementation priority; paper and released-code behavior are kept as explicit provenance rather than silently substituted.

This repository does not redistribute TCGA WSI data, generated coordinate/feature files, model checkpoints, third-party source checkouts, or credentials.

## Start here

- [Validated BLCA pilot baseline](reports/validated_pilot_baseline.md)
- [KIRP cohort-readiness audit](reports/kirp_cohort_readiness.md)
- [Supervisor-aligned architecture](SUPERVISOR_PIPELINE.md)
- [Current project state](PROJECT_STATE.md)
- [Multiscale adapter design](multiscale_feature_pilot/reports/adapter_design.md)
- [Historical GPU-machine handoff](GPU_HANDOFF.md)
- [Historical extraction plan](multiscale_feature_pilot/reports/next_real_multiscale_extraction.md)
- [GitHub inclusion/exclusion boundary](GITHUB_CONTENTS.md)

## Fixed architecture

```text
same patient
├── WSI 2x -> patches -> ImageNet1K V2 ResNet50 -> [N1,2048]
├── WSI 4x -> patches -> ImageNet1K V2 ResNet50 -> [N2,2048]
│
│   torch.cat([scale_2x, scale_4x], dim=0)
│   -> [N1+N2,2048] -> [1,2048,N1+N2]
│
├── RNA      -> separate HEALNet input
├── Mutation -> separate HEALNet input
└── CNV      -> separate HEALNet input
                         ↓
                      HEALNet
                         ↓
                       [1,4]
```

WSI is never directly concatenated with Omic values. Only the two WSI feature bags are concatenated, along patch rows.

## Current status

| Component | Status |
|---|---|
| Patient identity rule | Verified |
| BLCA WSI/Omic exact match | Verified for `TCGA-2F-A9KT` |
| 2x / 4x ResNet50 features | `[35534,2048]` / `[8911,2048]`, verified |
| Combined WSI feature bag | `[44445,2048]`, verified |
| Real-input random-weight HEALNet interface smoke | Finite `[1,4]` |
| Tests | 66 passed at freeze validation |
| BLCA one-patient pilot | `BLCA_ONE_PATIENT_PILOT_SUCCESS` |
| HEALNet training | Not started |
| KIRP acquisition | Not started; deterministic row-level selection manifest required |

The one-patient implementation is frozen as a reproducible engineering reference. The HEALNet smoke used random initialization, so its output is not a trained survival prediction or scientific result. See the validated baseline for immutable input, feature, provenance, and implementation hashes.

## Repository layout

```text
healnet_pilot/
├── multiscale_feature_pilot/
│   ├── config/
│   ├── provenance/
│   ├── reports/
│   ├── src/
│   └── tests/
├── scripts/
│   ├── check_gpu_readiness.py
│   └── run_blca_one_patient_pilot.py
├── reports/                  # audit/history documents
├── shared/provenance/
├── tracks/                   # paper/released-code provenance
├── GPU_HANDOFF.md
├── SUPERVISOR_PIPELINE.md
├── PROJECT_STATE.md
└── TOMORROW_START.md
```

## Test

Keep a clean official HEALNet `v0.1.0` checkout in a sibling directory named `healnet`, then run:

```bash
python -m pytest multiscale_feature_pilot/tests -q
```

See [reports/validated_pilot_baseline.md](reports/validated_pilot_baseline.md) for the frozen reference contract and [GPU_HANDOFF.md](GPU_HANDOFF.md) for the historical machine-migration procedure.

## Reproducibility rules

- Match WSI and Omic data by stable patient/case ID, never row order.
- Keep only patients with both sources.
- Preserve paper, released-code, and supervisor-track differences.
- Do not label `0.4554` or `0.9108 µm/px` as exact `0.5` or `1.0`.
- Do not modify the official HEALNet checkout for this pilot.
- Do not commit `.svs`, `.h5`, `.pt`, `.pth`, `.ckpt`, credentials, OAuth/rclone files, caches, or nested third-party repositories.
- Do not scale or train until the KIRP row-level cohort selection is independently reproducible and approved.

## Official sources

- HEALNet repository: <https://github.com/konst-int-i/healnet>
- HEALNet paper: <https://arxiv.org/abs/2311.09115>
- CLAM: <https://github.com/mahmoodlab/CLAM>
