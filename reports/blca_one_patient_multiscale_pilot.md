# BLCA one-patient multiscale pilot

Final status: `BLCA_ONE_PATIENT_PILOT_SUCCESS`

This was one real-patient interface and numerical smoke test. The HEALNet model was randomly initialized; its finite output is not a trained survival prediction or scientific result.

## Patient and verified inputs

- Patient/case: `TCGA-2F-A9KT`
- WSI: `TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs`
  - Size: `2,658,499,382` bytes
  - MD5: `824785fee9387dcf46a7058a0722739b`
- Coordinate HDF5: `TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C![![alt text](image.png)](image.png)9EB3970F.h5`
  - Size: `572,080` bytes
  - SHA256: `e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51`
  - `coords`: `[8911,2]`, `int64`, `patch_level=1`, `patch_size=256`, 8,911 unique, zero duplicates
- Omic CSV: `blca_master.csv`
  - SHA256: `9fa2cd83906c00e1f50113ba8e806ea4537806aeb2aacbee60e86c80b53f6929`
  - Exact case/slide match: one row, sample `TCGA-2F-A9KT-01`
- ResNet50 checkpoint: `resnet50-11ad3fa6.pth`
  - SHA256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`
- GPU: Tesla T4, 15,360 MiB; driver `580.173.02`; PyTorch CUDA build `12.8`

## Locked 2× engineering branch

- Policy: `APPROVED_2X_ENGINEERING_POLICY_V1`
- Source: OpenSlide level 0, `0.2277` MPP, complete `512×512` footprints
- Output patch: `256×256` using `PIL.Image.Resampling.LANCZOS`
- Effective resolution: `0.4554` MPP, an engineering approximation—not exact 0.5 MPP or exact paper reproduction
- Tissue mask: OpenSlide level 2, `3.6432` MPP
- Pinned CLAM behavior commit: `26e0b6c4873e112f1ccd74cd834894c4ab7a2934`
- Grid: global `(0,0)` anchor, `(512,512)` step, row-major `(y,x)`, `custom_global_lattice_v1`
- Retained mask geometry: 2 tissue contours and 4 holes
- Coordinates: `N1=35,534`, `[35534,2]`, `int64`, 35,534 unique, zero duplicates
- Coordinate generation time: `26.5893 s`

## ResNet50 extraction and artifacts

The classifier was removed from ImageNet1K_V2 ResNet50. Both branches used explicit RGB, tensor-domain bilinear resize from `256×256` to `224×224` with antialiasing, ImageNet normalization, float32, `model.eval()`, and `torch.inference_mode()` on CUDA. Batch size was 32 with two ordered read workers. No individual patch images were saved.

| Branch | Coordinates | Feature shape | Finite | Model-forward time | Streaming extraction time | Artifact SHA256 |
|---|---:|---:|---|---:|---:|---|
| 2× | 35,534 | `[35534,2048]` | yes | `91.5196 s` | `830.8398 s` | `e65f47f480cf81cc63d1198066b8f5ac28775e3877c64ff6a9a4f402ca46b774` |
| 4× | 8,911 | `[8911,2048]` | yes | `22.9008 s` | `77.7428 s` | `67deb6ce5a797c59b9077bbe67197d3d30d565f295e2b458cb67c89e66cb3863` |

- Total ResNet model-forward time: `114.4204 s`
- Total end-to-end streaming extraction time: `908.5826 s`
- ResNet branch peak allocated GPU memory: `447,780,352` bytes (`427.04 MiB`)

## Concatenation and provenance

- Operation: `torch.cat([features_2x, features_4x], dim=0)`
- Branch order: all 2× rows, then all 4× rows
- Combined shape: `[44445,2048]`, float32, all finite
- Combined feature SHA256: `efbee6a86cab43b63ed75064b29cc87b9cabce16443ffa37c7689679fbe3c7ca`
- Provenance: 44,445 contiguous, row-aligned records
- Provenance CSV SHA256: `581dbef73b3a35951ba77d99380726233266b4dc89ff92f7d33d0720f3f9fe01`

Each provenance row contains `global_row_index`, `branch`, `local_patch_index`, `x`, `y`, `level`, `mpp_x`, and `mpp_y`.

## Official HEALNet real-input interface smoke

- Official version: `v0.1.0`
- Official commit: `28ba5da6ab99fd8069972c22e986d83edb658dd4`
- WSI input: `[1,2048,44445]`
- RNA input: `[1,1,1523]`
- Mutation input: `[1,1,1125]`
- CNV input: `[1,1,193]`
- Output: `[1,4]`, float32, finite
- All four cross-attention paths ran and produced finite attention tensors
- Interface duration, including input staging, pinned-source loading, model setup, forward, and validation: `1.2071 s`
- Overall peak allocated GPU memory: `1,106,335,232` bytes (`1055.08 MiB`)

## Publication and recovery note

Artifacts were written outside Git into a sibling staging directory, read back, hashed, and published by one atomic directory rename. The initial runner invocation stopped before publication because its final self-check compared in-memory tuple shapes with their JSON array representation. No extraction or model computation failed. Every staged tensor, hash, provenance row, HEALNet attention result, official-repository check, and execution-source hash was revalidated before the preserved transaction was atomically published. The runner comparison was then corrected to normalize values through strict JSON.

- Output directory: `/teamspace/studios/this_studio/healnet_pilot_data/blca/bc9e3954-59d0-4f25-9022-42c97db7aea2/blca_one_patient_multiscale_pilot`
- Manifest SHA256: `e761161b9a8f0fd0d0510c2f52ea8559164b8683800ad4d2b0dbcec131dfbb03`
- Total recorded run time: `953.1476 s`

## Safety confirmation

- Official HEALNet modified: **NO**
- Training performed: **NO**
- KIRP downloaded: **NO**
- Additional patients processed: **NO**
- Verified 4× HDF5 modified or regenerated: **NO**
- Individual patch images saved: **NO**
- Git commit or push performed: **NO**
