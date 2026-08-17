# Validated BLCA one-patient pilot baseline

Baseline result: `BLCA_ONE_PATIENT_PILOT_SUCCESS`

This document freezes the evidence from the successful real-input, one-patient BLCA multiscale pilot. It records a feature-extraction and interface/numerical smoke-test baseline only. The HEALNet instance used random initialization; its finite `[1,4]` output is **not** a trained survival prediction, clinical result, or scientific performance result.

## Patient and output contract

- Patient/case: `TCGA-2F-A9KT`
- 2× engineering branch patch count: `35,534`
- 4× engineering branch patch count: `8,911`
- Concatenation: `torch.cat([features_2x, features_4x], dim=0)`
- Combined features: `[44445,2048]`, `float32`, finite
- HEALNet WSI input: `[1,2048,44445]`
- RNA input: `[1,1,1523]`
- Mutation input: `[1,1,1125]`
- CNV input: `[1,1,193]`
- HEALNet interface output: `[1,4]`, `float32`, finite
- Cross-attention validation: all four expected paths produced finite attention tensors

The Omic row was selected by exact patient/slide identity, not by row order. The matched Omic sample was `TCGA-2F-A9KT-01`.

## Verified inputs and model identity

| Input | Size | Verified identity |
|---|---:|---|
| WSI `TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs` | `2,658,499,382` bytes | MD5 `824785fee9387dcf46a7058a0722739b` |
| Coordinate HDF5 `TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.h5` | `572,080` bytes | SHA256 `e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51` |
| Omic `blca_master.csv` | `9,318,450` bytes | SHA256 `9fa2cd83906c00e1f50113ba8e806ea4537806aeb2aacbee60e86c80b53f6929` |
| ResNet50 `resnet50-11ad3fa6.pth` | `102,540,417` bytes | SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca` |

The feature backbone identity was torchvision `ResNet50_Weights.IMAGENET1K_V2`, with its classifier removed. Extraction used explicit RGB conversion, tensor-domain bilinear resizing from `256×256` to `224×224` with antialiasing, ImageNet normalization, `float32`, `model.eval()`, and `torch.inference_mode()` on the Tesla T4.

The HDF5 was consumed read-only. Its `coords` dataset was `[8911,2]`, `int64`, with `patch_level=1`, `patch_size=256`, 8,911 unique coordinates, and zero duplicates.

## Preserved pilot artifact identities

These are the exact hashes written into the successful external pilot manifest. The artifacts remain outside Git.

| Artifact | Shape or rows | Size | SHA256 |
|---|---:|---:|---|
| 2× coordinates | `[35534,2]` | `570,335` bytes | `03dd61138d0d9cf968e2e01ac64e40a888b8d13ccb1408fa02fa3bd1b8d92493` |
| 2× ResNet50 features | `[35534,2048]` | `291,096,555` bytes | `e65f47f480cf81cc63d1198066b8f5ac28775e3877c64ff6a9a4f402ca46b774` |
| 4× ResNet50 features | `[8911,2048]` | `73,000,939` bytes | `67deb6ce5a797c59b9077bbe67197d3d30d565f295e2b458cb67c89e66cb3863` |
| Combined multiscale features | `[44445,2048]` | `364,095,383` bytes | `efbee6a86cab43b63ed75064b29cc87b9cabce16443ffa37c7689679fbe3c7ca` |
| Combined provenance CSV | `44,445` rows | `2,151,419` bytes | `581dbef73b3a35951ba77d99380726233266b4dc89ff92f7d33d0720f3f9fe01` |

- Successful external manifest SHA256: `e761161b9a8f0fd0d0510c2f52ea8559164b8683800ad4d2b0dbcec131dfbb03`
- External artifact directory: `/teamspace/studios/this_studio/healnet_pilot_data/blca/bc9e3954-59d0-4f25-9022-42c97db7aea2/blca_one_patient_multiscale_pilot`

## Approved 2× engineering policy

- Policy: `APPROVED_2X_ENGINEERING_POLICY_V1`
- Source: OpenSlide level 0 at `0.2277` µm/px
- Source footprint: complete `512×512` level-0 pixels
- Output: `256×256` pixels via `PIL.Image.Resampling.LANCZOS`
- Effective resolution: `0.4554` µm/px
- Tissue mask: OpenSlide level 2 at `3.6432` µm/px (16× level-0 downsample)
- Pinned CLAM behavior commit: `26e0b6c4873e112f1ccd74cd834894c4ab7a2934`
- Tissue parameters: `sthresh=8`, `mthresh=7`, `close=4`, `use_otsu=false`, `a_t=100`, `a_h=16`, `max_n_holes=8`
- Contour rule: `pinned_four_pt_easy`
- Grid: global `(0,0)` anchor, `(512,512)` step, row-major `(y,x)`, `custom_global_lattice_v1`
- Boundary behavior: reject incomplete source footprints
- Retained mask geometry: 2 tissue contours and 4 holes
- Result: `[35534,2]`, `int64`, 35,534 unique coordinates, zero duplicates

The effective `0.4554` µm/px is **not exact 0.5 MPP** and must not be presented as exact paper reproduction. It is the approved engineering approximation for this pilot.

## 4× branch caveat

The verified HDF5 coordinates belong only to OpenSlide level 1, whose actual resolution is `0.9108` µm/px and whose downsample is 4× relative to level 0. The branch was accepted as the pilot's approximate 1.0 MPP / 4× branch; it is not an exact native 1.0 MPP level. The HDF5 was not assigned to the 2× branch, modified, or regenerated.

## Official HEALNet source identity

- Release tag: `v0.1.0`
- Peeled Git commit (official source identity): `28ba5da6ab99fd8069972c22e986d83edb658dd4`
- Executed Git-object path: `healnet/models/healnet.py`
- Executed Git-object content SHA256: `78e9c2b10455f6ad43a88ed4e0a646cbf6304bf52cac06a5140f49dbc4796820`
- Official checkout was clean and was not modified.

The smoke helper loaded `healnet/models/healnet.py` read-only from the pinned `v0.1.0` Git object, rather than trusting mutable working-tree source.

## Exact pilot execution source identity

The successful manifest records pilot repository HEAD `79f0dd0195cf5dc03257d105d2b93b7d6302fe7d` on branch `gpu-handoff-multiscale-pilot`, with a dirty working tree. Therefore, that Git HEAD alone is **not** a complete identity for the code that executed. The `Executed SHA256` values below are the authoritative byte identities captured by the successful manifest.

The `Freeze-candidate SHA256` column records the source/config bytes present while this baseline report was authored. Three files differ from their executed bytes because the successful transaction was subsequently accepted into provenance/config and the runner received a serialization-only strict-JSON self-check correction. No patches were re-read, no ResNet or HEALNet computation was rerun, and none of the successful external artifacts was changed.

| Repository-relative file | Executed SHA256 (manifest-recorded) | Freeze-candidate SHA256 |
|---|---|---|
| `multiscale_feature_pilot/config/pilot_config.yaml` | `524230d47e110a15f30c43fa6b054bb470756f58b3e7ad532860831125513976` | `cbb6ef808504823e094d24f3b6313d39ece9952a43019c3b0f799a320e8c58ac` |
| `multiscale_feature_pilot/provenance/scale_selection.yaml` | `745d914f023cd7a976cd8ab78db340389d917b19b8af07c23a282b902b681fc2` | `41500b61a4330d9591fe37aeb341c9ad0b6cbaf718b9665b744b2733adf2200e` |
| `multiscale_feature_pilot/src/artifacts.py` | `b9d3670dcc114610e412d86fd55aa9b4e9d7ff0d53234296e7dbfcb7e9b63c03` | `b9d3670dcc114610e412d86fd55aa9b4e9d7ff0d53234296e7dbfcb7e9b63c03` |
| `multiscale_feature_pilot/src/feature_extraction.py` | `9731f9773238800268673ec299acbcdf8caaa4ac676c96f4559cb10604c707e7` | `9731f9773238800268673ec299acbcdf8caaa4ac676c96f4559cb10604c707e7` |
| `multiscale_feature_pilot/src/healnet_smoke.py` | `34a7ad1204dfc84ef8e63a2ac4cd8b932dbccf39e9f424b424dc6eb37a4d6009` | `34a7ad1204dfc84ef8e63a2ac4cd8b932dbccf39e9f424b424dc6eb37a4d6009` |
| `multiscale_feature_pilot/src/multiscale_bag.py` | `26a85db8738b80c7f7f2f75d1379a54203ee870519ee1ee63f78f809ed17d914` | `26a85db8738b80c7f7f2f75d1379a54203ee870519ee1ee63f78f809ed17d914` |
| `multiscale_feature_pilot/src/omic.py` | `cd9c80bd9ab3a049beca682131f2553f526a38a0dee09608eb016ae1f79607ed` | `cd9c80bd9ab3a049beca682131f2553f526a38a0dee09608eb016ae1f79607ed` |
| `multiscale_feature_pilot/src/provenance.py` | `76f8c9eac1ba0c32679a1d7f8d34c07c79fc27cdc28c6762dd13a169e5db5917` | `76f8c9eac1ba0c32679a1d7f8d34c07c79fc27cdc28c6762dd13a169e5db5917` |
| `multiscale_feature_pilot/src/scale_2x_policy.py` | `a3672f730f8be026edfab0ea25c4460836aad227099152e5cc7d290c03d1f7aa` | `a3672f730f8be026edfab0ea25c4460836aad227099152e5cc7d290c03d1f7aa` |
| `multiscale_feature_pilot/src/tissue_coordinates.py` | `159f4f5b0e49eb7ef92c213fca62be696351bee573417be09e11d18d11c3e75a` | `159f4f5b0e49eb7ef92c213fca62be696351bee573417be09e11d18d11c3e75a` |
| `requirements-handoff.txt` | `2bf6686e619e97a5d7b9c746e51966d9649994e3116b52af71084150fa619023` | `2bf6686e619e97a5d7b9c746e51966d9649994e3116b52af71084150fa619023` |
| `scripts/run_blca_one_patient_pilot.py` | `d43a5dab944116cff72ec048f84de5b5facb956e6ccf4669c534ebbf97e2c05f` | `dd2b14111696e0c9053721833a1c9c5bd6a7764fa0e4dc8e8b508c9c0f8bf621` |

The freeze commit and tag are intentionally not predicted in this file: a commit cannot embed its own object ID without changing that ID. The annotated freeze tag must point to the reviewed commit containing this report, and the staged source hashes must be rechecked against the freeze-candidate column immediately before that commit is created.

## Scope and safety boundary

- Training performed: **NO**
- Trained HEALNet weights loaded: **NO**
- Scientific survival prediction produced: **NO**
- KIRP downloaded or processed: **NO**
- Additional patients processed: **NO**
- Official HEALNet modified: **NO**
- WSI, HDF5, feature tensors, and checkpoint placed in Git: **NO**

This baseline authorizes neither HEALNet training nor full-cohort KIRP acquisition. Those are separate future milestones.
