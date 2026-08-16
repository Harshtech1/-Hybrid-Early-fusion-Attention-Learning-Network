# HEALNet reproducibility pilot

This repository documents a reproduction and engineering study of **HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data** (NeurIPS 2024). It separates claims from the paper from behavior in the camera-ready code and preserves metadata for one checksum-verified BLCA WSI engineering pilot.

This is an independent reproducibility project, not the official authors' repository. This repository does not redistribute TCGA WSI data, trained checkpoints, or authentication credentials.

Start with:

- [Complete project state and recovery guide](PROJECT_STATE.md)
- [Tomorrow's 30-second restart checklist](TOMORROW_START.md)
- [GitHub inclusion/exclusion plan](GITHUB_CONTENTS.md)
- [Third-party data notice](THIRD_PARTY_DATA_NOTICE.md)
- [Track comparison](reports/track_comparison.md)

## Workspace layout

```text
healnet_pilot/
├── CLAM/                         # pinned external checkout; shared read-only input
├── blca/                         # verified pilot WSI; do not commit
├── blca_preprocessed/            # CLAM HDF5 and QC artifacts; shared read-only input
├── shared/
│   ├── provenance/
│   │   ├── pilot_manifest.yaml
│   │   └── known_issues.md
│   └── qc/
├── tracks/
│   ├── paper_faithful/
│   │   ├── config/
│   │   ├── models/
│   │   ├── features/
│   │   ├── logs/
│   │   └── provenance/
│   └── released_code/
│       ├── config/
│       ├── models/
│       ├── features/
│       ├── logs/
│       └── provenance/
├── reports/
├── PROJECT_STATE.md
├── TOMORROW_START.md
├── GITHUB_CONTENTS.md
└── .gitignore
```

The existing WSI, CLAM checkout, and CLAM outputs remain in their original locations. They were not moved or duplicated during workspace organization.

## Experimental tracks

### Paper-faithful

- Intended encoder: Kather100K-pretrained ResNet50
- Intended WSI resolution: approximately 0.5/1.0 micrometres per pixel
- Status: `BLOCKED_PENDING_AUTHOR`
- Reason: exact checkpoint and physical-resolution procedure are unresolved

### Released-code

- Encoder: torchvision ResNet50 `IMAGENET1K_V2`
- Camera-ready transform: `256×256` OpenSlide region, RGB, tensor conversion, resize to `224×224`, ImageNet normalization
- Status: `READY_FOR_CONTROLLED_PILOT_PENDING_GPU`
- Label: released-code engineering reproduction, not paper-faithful

## Current status

The pipeline is validated through one-slide CLAM coordinate generation and QC:

```text
GDC WSI → MD5 → OpenSlide → pinned CLAM → 8,911 HDF5 coordinates
```

Feature extraction is not started. The current Lightning Studio has no accessible GPU, and the exact released-code ImageNet checkpoint has not been downloaded. HEALNet training and evaluation are not started.

## Official sources

- HEALNet repository: <https://github.com/konst-int-i/healnet>
- HEALNet paper: <https://arxiv.org/abs/2311.09115>
- GDC Data Portal: <https://portal.gdc.cancer.gov/>
- GDC Data Transfer Tool: <https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/>
- OpenSlide: <https://openslide.org/download/>
- CLAM: <https://github.com/mahmoodlab/CLAM>

## Reproducibility rules

- Never silently replace Kather100K with ImageNet.
- Never silently replace missing molecular modalities.
- Never mix paper-faithful and released-code results.
- Do not download complete WSI cohorts until the pilot passes.
- Do not destructively modify or reset the official HEALNet checkout.
- Never commit credentials, OAuth material, raw `.svs`, generated `.h5`/`.pt`, or model checkpoints.
- Record immutable SHAs, hashes, versions, tensor shapes, and data lineage for every generated artifact.

This directory is not yet a Git repository. Do not initialize, commit, create GitHub resources, or push until the separate repository-creation phase is explicitly authorized.
