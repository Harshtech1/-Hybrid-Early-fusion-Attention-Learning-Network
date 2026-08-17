# Supervisor-aligned HEALNet pilot

This independent repository contains the code, tests, provenance, and documentation for a patient-matched WSI + multi-omic HEALNet survival pilot. Supervisor instructions are the implementation priority; paper and released-code behavior are kept as explicit provenance rather than silently substituted.

This repository does not redistribute TCGA WSI data, generated coordinate/feature files, model checkpoints, third-party source checkouts, or credentials.

## Start here

- [GPU-machine handoff and exact restart procedure](GPU_HANDOFF.md)
- [Short restart checklist](TOMORROW_START.md)
- [Supervisor-aligned architecture](SUPERVISOR_PIPELINE.md)
- [Current project state](PROJECT_STATE.md)
- [Multiscale adapter design](multiscale_feature_pilot/reports/adapter_design.md)
- [Next real extraction plan](multiscale_feature_pilot/reports/next_real_multiscale_extraction.md)
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
| Two-scale tensor/provenance/padding adapter | Implemented |
| Synthetic WSI + RNA + mutation + CNV HEALNet forward | Finite `[1,4]` |
| Tests | 24 passed |
| Real ResNet50 feature extraction | Not started |
| Runtime gate | `BLOCKED_NO_GPU` on source machine |
| HEALNet training | Not started |

The code handoff is ready for a GPU machine, but it is not a completed push-button extraction pipeline. It includes the strict adapter, synthetic interface tests, readiness checker, and runbook. The real extraction runner and final 2x coordinate-generation details remain the next implementation step.

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
│   └── check_gpu_readiness.py
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

See [GPU_HANDOFF.md](GPU_HANDOFF.md) for environment, data-transfer, hash-verification, and GPU-gate instructions.

## Reproducibility rules

- Match WSI and Omic data by stable patient/case ID, never row order.
- Keep only patients with both sources.
- Preserve paper, released-code, and supervisor-track differences.
- Do not label `0.4554` or `0.9108 µm/px` as exact `0.5` or `1.0`.
- Do not modify the official HEALNet checkout for this pilot.
- Do not commit `.svs`, `.h5`, `.pt`, `.pth`, `.ckpt`, credentials, OAuth/rclone files, caches, or nested third-party repositories.
- Stop after the first real one-patient forward; do not scale or train until it passes.

## Official sources

- HEALNet repository: <https://github.com/konst-int-i/healnet>
- HEALNet paper: <https://arxiv.org/abs/2311.09115>
- CLAM: <https://github.com/mahmoodlab/CLAM>
