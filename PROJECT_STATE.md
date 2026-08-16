# HEALNet Reproduction Project State

Last updated: 2026-08-15 UTC

## 1. Project objective

This project is a forensic reproducibility and implementation study of **HEALNet: Multimodal Fusion for Heterogeneous Biomedical Data** (NeurIPS 2024). It aims to reconstruct the data lineage, preprocessing, feature extraction, model input contract, training procedure, and reported TCGA survival experiments without silently resolving discrepancies between the paper and released code.

Two experimental claims must remain separate:

- **Paper-faithful reproduction:** reproduces the methodology stated in the paper, including Kather100K-pretrained ResNet50 features and approximately 0.5/1.0 micrometres-per-pixel WSI processing. This track is blocked until authoritative provenance is available.
- **Released-code reproduction:** reproduces the behavior of the camera-ready `v0.1.0` implementation, including torchvision ResNet50 `IMAGENET1K_V2`. This is an engineering reproduction and must not be labelled paper-faithful.

Primary paper cohorts are BLCA, BRCA, KIRP, and UCEC. The repository also supports LUAD, LUSC, PAAD, and HNSC, but those four were not identified as primary paper TCGA experiments.

## 2. Official sources

- HEALNet GitHub: <https://github.com/konst-int-i/healnet>
- HEALNet paper: <https://arxiv.org/abs/2311.09115>
- OpenSlide: <https://openslide.org/download/>
- GDC Data Portal: <https://portal.gdc.cancer.gov/>
- GDC Data Transfer Tool: <https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/>
- CLAM: <https://github.com/mahmoodlab/CLAM>

Author-linked TCGA/TCIA/reference pages:

- BLCA: <https://www.cancerimagingarchive.net/collection/tcga-blca/>
- BRCA: <https://www.cancerimagingarchive.net/collection/tcga-brca/>
- UCEC: <https://www.cancerimagingarchive.net/collection/tcga-ucec/>
- KIRP: <https://www.cancerimagingarchive.net/collection/tcga-kirp/>
- LUAD: <https://www.cancerimagingarchive.net/collection/tcga-luad/>
- LUSC: <https://www.cancerimagingarchive.net/collection/tcga-lusc/>
- HNSC: <https://www.cancerimagingarchive.net/collection/tcga-hnsc/>
- PAAD reference: <https://gdc.cancer.gov/resources-tcga-users/tcga-code-tables/tcga-study-abbreviations>

The TCIA links are author-provided cohort/reference pages. HEALNet pathology `.svs` acquisition uses the GDC Data Portal/manifests and the official `gdc-client`; TCIA is not the download path used by the repository task.

## 3. Current high-level pipeline

```text
MULTIOMICS
Git LFS
  ↓
RNA-seq / Mutation / CNV / outcomes

WSI
GDC
  ↓
.svs
  ↓
OpenSlide / CLAM
  ↓
patch coordinates .h5
  ↓
ResNet50
  ↓
.pt features
  ↓
HEALNet
```

## 4. Current project status

| Component | Status | Details |
|---|---|---|
| Repository audit | COMPLETED | Camera-ready tag `v0.1.0` and relevant history, paths, tasks, loaders, and configs inspected. |
| Paper audit | COMPLETED | Main methodology, cohorts, WSI resolutions, feature dimension, and encoder claim inspected. |
| Google Drive isolation | COMPLETED | Dedicated read-only remote points to the empty `HEALNet_Data` scaffold. |
| rclone setup | COMPLETED | `healnet-gdrive` uses read-only scope, fixed root, and shortcut skipping. |
| Omics audit | COMPLETED / UNRESOLVED PROVENANCE | Local clean and Xena files inventoried; KIRP CNV and UCEC RNA absence remains unresolved. |
| GDC manifests | COMPLETED / WAITING FOR AUTHOR | Counts and candidate mismatches audited; exact final paper exclusion lists were not recovered. |
| GDC client | COMPLETED | Official client `2.3` installed at `/home/zeus/.local/bin/gdc-client`. |
| One-slide WSI | COMPLETED | Exactly one BLCA pilot WSI downloaded through GDC for this pilot. |
| WSI MD5 | COMPLETED | Exact checksum match. |
| OpenSlide | COMPLETED | Slide opened and small `read_region` tests passed. |
| MPP/pyramid inspection | COMPLETED | Four native levels and effective MPP values measured. |
| CLAM provenance | COMPLETED / UNCERTAIN | Historical compatibility pin selected; exact author runtime SHA is unknown. |
| CLAM installation | COMPLETED | Isolated checkout at pinned commit under `healnet_pilot/CLAM`. |
| CLAM preprocessing | COMPLETED | One slide produced mask, coordinate HDF5, stitch, and process metadata. |
| CLAM QC | COMPLETED | Engineering PASS; lower-left debris/artifact remains a pathology-QC consideration. |
| `.pt` contract tracing | COMPLETED | Raw shape, dtype, loader permutation, and camera-ready buffer defect identified. |
| Released-code feature extraction | WAITING FOR GPU | Exact checkpoint is absent locally; CUDA is unavailable. No feature file exists. |
| Paper-faithful feature extraction | WAITING FOR AUTHOR | Exact Kather100K checkpoint and physical-resolution procedure unresolved. |
| HEALNet loader test | NOT STARTED | Must follow a validated one-slide `.pt`. |
| Training | NOT STARTED | No HEALNet training has been run. |
| Evaluation | NOT STARTED | No reproduction metrics have been generated. |

## 5. Verified Google Drive architecture

- Remote: `healnet-gdrive`
- Config: `/teamspace/studios/this_studio/rclone-config/healnet.conf`
- Scope: `drive.readonly`
- Root folder ID: `1yHt2wBVDB0rAiSabOCm2eJSduBlTr05W`
- `skip_shortcuts`: `true`

Logical Drive scaffold:

```text
HEALNet_Data/
├── experiments/
├── metadata/
├── omics/
├── patch_features/
├── processed/
├── raw/
└── wsi/
```

The scaffold was manually created and is currently empty. No HEALNet dataset has been uploaded there, and Google Drive has not been used for pilot data. Never expose the OAuth token or copy credentials into project documentation or Git.

## 6. Local repository

Official local checkout:

```text
/teamspace/studios/this_studio/healnet
```

The checkout contains pre-existing user modifications and untracked files. They must not be reset, discarded, overwritten, or presented as pilot-generated changes. The pilot work intentionally operates under `/teamspace/studios/this_studio/healnet_pilot` and did not modify those existing repository changes.

The immutable camera-ready tag audited was:

```text
v0.1.0
beb8b000121e311a219ff5c1837cf05ff8c44c1b
```

## 7. Local data already available

Repository data locations:

```text
/teamspace/studios/this_studio/healnet/data/tcga/omic/
/teamspace/studios/this_studio/healnet/data/tcga/omic_xena/
/teamspace/studios/this_studio/healnet/data/tcga/gdc_manifests/full/
/teamspace/studios/this_studio/healnet/data/tcga/gdc_manifests/filtered/
```

Four primary clean omic archives:

```text
tcga_blca_all_clean.csv.zip
tcga_brca_all_clean.csv.zip
tcga_kirp_all_clean.csv.zip
tcga_ucec_all_clean.csv.zip
```

Observed modality-column counts:

| Cohort | RNA | CNV | Mutation |
|---|---:|---:|---:|
| BLCA | 1554 | 256 | 371 |
| BRCA | 1558 | 1333 | 21 |
| KIRP | 1557 | 0 | 20 |
| UCEC | 0 | 5 | 1406 |

Do not repair these values or merge `omic_xena/*_master.csv` into the clean tables without a separate, documented provenance decision.

## 8. WSI manifest status

Filtered repository manifest counts:

| Cohort | Slides |
|---|---:|
| BLCA | 437 |
| BRCA | 1022 |
| KIRP | 300 |
| UCEC | 566 |
| **Total** | **2325** |

Estimated storage is approximately 2.906 TB decimal / 2.644 TiB.

Paper-reported slide counts differ:

- BLCA: 436
- BRCA: 1019 or 1021 depending on paper section/context
- KIRP: 297
- UCEC: 566

The exact final paper slide-exclusion artifacts have not been recovered. Do not describe the filtered manifests as the exact paper inclusion lists.

## 9. Verified pilot WSI

- Cohort: BLCA
- GDC UUID: `bc9e3954-59d0-4f25-9022-42c97db7aea2`
- Filename: `TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs`
- Directory: `/teamspace/studios/this_studio/healnet_pilot/blca/bc9e3954-59d0-4f25-9022-42c97db7aea2/`
- Size: `2,658,499,382` bytes
- MD5: `824785fee9387dcf46a7058a0722739b`
- Clean-table match: YES
- Case: `TCGA-2F-A9KT`
- Survival: `108.87` months
- Censoring: `1`

The pinned CLAM repository itself contains two upstream demo `.svs` files. They were not downloaded as pilot TCGA data and must not be confused with the verified BLCA WSI.

## 10. OpenSlide pilot results

- OpenSlide Python: `1.4.6`
- Native OpenSlide library: `3.4.1`
- Vendor: `aperio`
- Level-0 dimensions: `131072 × 104192`
- Base MPP: `0.2277 × 0.2277` micrometres/pixel

| Level | Downsample | Effective MPP |
|---:|---:|---:|
| 0 | 1× | 0.2277 |
| 1 | 4× | 0.9108 |
| 2 | 16× | 3.6432 |
| 3 | 64× | 14.5728 |

Critical lesson: an OpenSlide integer level does **not** equal the paper's physical-resolution wording of “2×/4×” or approximately 0.5/1.0 micrometres per pixel. Level 1 is only an engineering approximation to 1.0 micrometre/pixel for this particular slide.

## 11. CLAM provenance

- Checkout: `/teamspace/studios/this_studio/healnet_pilot/CLAM`
- Pinned pilot commit: `26e0b6c4873e112f1ccd74cd834894c4ab7a2934`

This is a historically aligned reproducibility pin, **not** a proven exact author runtime commit. HEALNet's task clones CLAM without an immutable revision. The exact author runtime CLAM revision remains uncertain.

## 12. CLAM pilot

- Patch level: `1`
- Patch size: `256 × 256`
- Effective MPP: `0.9108` micrometres/pixel
- Coordinate count: `8911`
- Duplicate coordinates: `0`
- Out-of-bounds coordinates: `0`
- HDF5: `/teamspace/studios/this_studio/healnet_pilot/blca_preprocessed/patches/TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.h5`
- HDF5 SHA256: `e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51`
- QC status: engineering PASS

Generated and verified:

- `masks/<slide_id>.jpg`
- `stitches/<slide_id>.jpg`
- `patches/<slide_id>.h5`
- `process_list_autogen.csv`
- `valid_prep_ids.csv`

No pilot `.pt` feature file was generated. The lower-left debris/artifact region is included by the segmentation, so pathology review is advisable before scientific use.

## 13. Exact `.pt` contract

Camera-ready feature generation saves a raw tensor with:

```text
shape: (max_patches, 2048)
dtype: float32
```

`TCGADataset.load_patch_features()` applies:

```python
patch_features = torch.load(file, weights_only=True)
patch_features = patch_features.permute(1, 0)
```

The resulting batched model input is:

```text
(batch, 2048, max_patches)
```

Coordinates are stored separately in HDF5. Training/model inference reads `.pt`, not `.h5`; the HDF5 is used during feature extraction and for coordinate-based explanation utilities.

For the isolated one-slide pilot, the intended released-code-compatible raw shape is exactly:

```text
(8911, 2048)
```

## 14. Feature-extractor discrepancy

This contradiction must remain explicit:

| Source | Encoder |
|---|---|
| Paper | ResNet50 pretrained on Kather100K |
| Camera-ready code | torchvision ResNet50 `ResNet50_Weights.IMAGENET1K_V2` |

The exact Kather100K checkpoint is **UNRESOLVED**. Never silently substitute ImageNet features into the paper-faithful track.

## 15. Feature extraction status

Status: **NOT STARTED**

Blockers:

1. Paper-faithful: exact Kather100K checkpoint and preprocessing provenance unresolved.
2. Released-code: NVIDIA GPU unavailable.
3. Released-code: exact `resnet50-11ad3fa6.pth` checkpoint is not cached locally and has not been downloaded.

Current runtime:

- Python: `3.12.11`
- PyTorch: `2.8.0+cu128`
- torchvision: `0.23.0+cu128`
- CUDA available: `False`
- GPU: unavailable
- `nvidia-smi`: unavailable

Next planned hardware state: attach/enable an NVIDIA GPU in Lightning Studio, then re-run the GPU gate before downloading or loading the released-code checkpoint.

## 16. Known camera-ready bug

The feature task allocates `patch_tensors = torch.zeros(max_patches, 2048)` once outside the slide loop and reuses it for all slides. When a shorter slide follows a longer slide, trailing rows can retain feature values from the previous slide.

Do not fix the official repository yet. The problem does not apply to the planned one-slide pilot because it will allocate exactly `(8911, 2048)`. Future cohort work must explicitly choose and label either:

- bug-for-bug released-code reproduction; or
- corrected implementation.

## 17. Two-track architecture

```text
tracks/
├── paper_faithful/
│   ├── config/
│   ├── models/
│   ├── features/
│   ├── logs/
│   └── provenance/
└── released_code/
    ├── config/
    ├── models/
    ├── features/
    ├── logs/
    └── provenance/
```

### Track A: `paper_faithful`

- Status: `BLOCKED_PENDING_AUTHOR`
- Encoder: Kather100K ResNet50
- Checkpoint: unresolved
- Resolution: approximately 0.5/1.0 micrometres per pixel
- Exact resolution procedure: unresolved

### Track B: `released_code`

- Operational status: `READY_FOR_CONTROLLED_PILOT_PENDING_GPU`
- Encoder: torchvision ResNet50 `IMAGENET1K_V2`
- Planned pilot output: `(8911, 2048)` float32
- Label: released-code engineering reproduction, **not** paper-faithful reproduction

## 18. Current exact stopping point

```text
verified .h5 patch coordinates
        ↓
ImageNet1K_V2 ResNet50
        ↓
pilot .pt feature tensor
```

Work is stopped before ResNet50 because no GPU is attached. The paper-faithful track is additionally blocked on author provenance.

No released-code pilot `.pt` or feature provenance YAML exists yet.

## 19. Tomorrow's exact next steps

1. Attach/enable an NVIDIA GPU in Lightning Studio.
2. Verify hardware without starting inference:

   ```bash
   nvidia-smi
   python --version
   python -c "import torch; print(torch.__version__)"
   python -c "import torch; print(torch.cuda.is_available())"
   ```

3. Continue only if `torch.cuda.is_available()` is `True` and `nvidia-smi` succeeds.
4. Download only the official released-code checkpoint:

   ```text
   URL: https://download.pytorch.org/models/resnet50-11ad3fa6.pth
   cache: /home/zeus/.cache/torch/hub/checkpoints/resnet50-11ad3fa6.pth
   ```

5. Record checkpoint size and SHA256; verify torchvision identifies it as `ResNet50_Weights.IMAGENET1K_V2`.
6. Extract features for exactly the existing 8,911 HDF5 coordinates, preserving order, using level 1, `256×256` RGB reads, resize to `224×224`, and ImageNet normalization.
7. Use `model.eval()` and `torch.inference_mode()` with batched GPU inference. Allocate exactly `(8911, 2048)` float32; do not use the cohort-wide buffer.
8. Save only to `tracks/released_code/features/<slide_id>.pt` with provenance YAML beside it.
9. Verify shape, dtype, row count, all-finite values, tensor/file SHA256, timing, throughput, GPU, versions, and checkpoint identity.
10. Perform only the loader permutation smoke test: `(8911, 2048) → (2048, 8911)`.
11. Stop before any full-cohort download, HEALNet training, or paper-faithful claim.

Canonical continuation requirements are in this section and in `tracks/released_code/provenance/status.yaml`. The last execution prompt is preserved locally at:

```text
/teamspace/studios/this_studio/.codex/attachments/25d17b7a-2988-4cfd-a7a0-fd4475fcca05/pasted-text.txt
```

The project must not rely on that attachment; this document contains the recovery-critical requirements.

## 20. Author communication

- Email sent to a HEALNet author.
- Automatic reply received.
- Author is out of office until August 17, 2026.
- Status: waiting for clarification; do not send another email now.

Questions requested:

- exact Kather100K checkpoint;
- final WSI inclusion lists;
- physical-resolution procedure;
- KIRP CNV and UCEC RNA provenance;
- four modalities versus combined-omics implementation;
- split seeds/folds; and
- exact preprocessing details.

## 21. Research conclusions so far

Proven locally or directly from official sources:

- TCGA/GDC is the WSI acquisition source used by the repository workflow.
- The official GDC client can download the selected slide.
- The selected WSI checksum, OpenSlide access, pyramid, and effective MPP are valid.
- The pinned CLAM revision can segment the slide and generate 8,911 valid coordinates.
- The CLAM coordinate HDF5 and QC outputs are structurally valid.
- The camera-ready `.pt` serialization and HEALNet loader transformation are identified.

Not proven:

- exact author Kather100K checkpoint;
- exact final paper WSI subset;
- exact physical-resolution selection/resampling method;
- exact molecular-table provenance for missing KIRP CNV and UCEC RNA modalities;
- exact fold seeds/split artifacts; and
- exact CLAM author runtime SHA.

## 22. Important rules for future work

> **NEVER silently replace Kather100K with ImageNet.**
>
> **NEVER silently replace missing molecular modalities.**
>
> **NEVER mix paper-faithful and released-code results.**
>
> **NEVER download all WSI before a pilot succeeds.**
>
> **NEVER modify the official repository destructively.**
>
> **NEVER commit credentials.**
>
> **NEVER commit raw `.svs` files to GitHub.**
>
> **NEVER commit large `.h5`/`.pt` datasets unless explicitly intended.**
>
> **NEVER upload proprietary or authentication files.**

No Git repository has been initialized for `healnet_pilot`, and nothing has been committed, pushed, uploaded, or mounted as part of this handoff.
