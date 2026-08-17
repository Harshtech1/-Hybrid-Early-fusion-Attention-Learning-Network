# Next real multiscale extraction step

> Historical execution plan. The plan was completed successfully; current results and hashes are frozen in [../../reports/validated_pilot_baseline.md](../../reports/validated_pilot_baseline.md).

This document defines the next GPU-enabled action. GPU, OpenSlide, code, and the official ImageNet1K V2 checkpoint are ready. Real extraction remains closed until the external WSI, coordinate HDF5, and Omic object pass the full gate.

## Fixed adapter contract

```text
branch scale_2x: [N1,2048]
branch scale_4x: [N2,2048]
combined:        torch.cat([scale_2x, scale_4x], dim=0)
result:          [N1+N2,2048]
loader form:     [2048,N1+N2]
batch form:      [1,2048,N1+N2] for the one-slide pilot
```

The combined provenance table must list every scale-2x row before every scale-4x row.

This is a supervisor-aligned preprocessing track: it uses ImageNet1K V2 ResNet50 rather than the paper's Kather100K-pretrained encoder. The paper supplies the HEALNet fusion design; the supervisor-specified WSI construction takes implementation priority.

The instruction to use a WSI "around four gigapixels" is interpreted as using the original high-resolution slide rather than a thumbnail. It is not treated as a requirement to resize every slide to exactly four billion pixels; that wording remains unresolved.

## Existing BLCA pilot slide

```text
GDC UUID:   bc9e3954-59d0-4f25-9022-42c97db7aea2
base MPP:   0.2277 x 0.2277
level 0:    downsample 1.0, actual MPP 0.2277
level 1:    downsample 4.0, actual MPP 0.9108
level 2:    downsample 16.0, actual MPP 3.6432
level 3:    downsample 64.0, actual MPP 14.5728
```

Neither 0.5 nor 1.0 micrometres per pixel is present exactly as a native level.

## Same-patient identity gate

The one-slide pilot uses case `TCGA-2F-A9KT`. A read-only lookup found exactly one matching row in the local BLCA Omic master table:

```text
WSI case:       TCGA-2F-A9KT
Omic _PATIENT: TCGA-2F-A9KT
Omic sample:   TCGA-2F-A9KT-01
Slide ID:      exact filename match
Decision:      KEEP
```

This establishes metadata identity only. No Omic transformation was performed. A WSI and Omic row with different patient identifiers must be discarded, never paired by row order.

The matching row contains complete finite values for the three planned Omic inputs:

```text
RNA:       [1,1,1523]
Mutation:  [1,1,1125]
CNV:       [1,1,193]
```

These tensors remain separate from each other and from WSI. They are not concatenated into the multiscale WSI bag.

The pilot Omic loader now enforces this identity and grouping contract directly: it requires the exact `_PATIENT` and full `slide_id`, preserves CSV header order within `*_rnaseq`, `*_mut`, and `*_cnv`, requires one match, and rejects missing/non-finite values.

## Branch scale_2x — target approximately 0.5 MPP

- Target: approximately 0.5 micrometres per pixel.
- Closest native level: level 0 at 0.2277 MPP.
- Native relative error: 54.46%; it must not be labeled exact or adequate merely because it is closest.
- The selected one-patient pilot policy is a paper-factor 2x virtual downsample from level 0, yielding 0.4554 MPP with an 8.92% target error.
- Exact target 0.5 would require an effective downsample of approximately 2.195872 from level 0.
- The existing HDF5 has `patch_level=1` and must not be used for this branch.

This is a supervisor-aligned engineering approximation, not an exact native 0.5 MPP reproduction. The policy is now frozen as `APPROVED_2X_ENGINEERING_POLICY_V1`: read complete 512 x 512 footprints from level 0, convert explicitly to RGB, downsample to 256 x 256 with `PIL.Image.Resampling.LANCZOS`, and reject incomplete footprints. Lanczos itself supplies the PIL downsampling antialiasing; there is no separate PIL antialias flag.

Tissue segmentation is fixed at OpenSlide level 2 (3.6432 MPP) using the pinned CLAM commit `26e0b6c4873e112f1ccd74cd834894c4ab7a2934` parameters (`sthresh=8`, `mthresh=7`, `close=4`, `use_otsu=false`, `a_t=100`, `a_h=16`, `max_n_holes=8`) and the exact pinned `four_pt` easy rule.

Coordinates use a level-0 global lattice anchored at `(0,0)`, step `(512,512)`, sorted row-major by `(y,x)`. This policy is labelled `custom_global_lattice_v1`; it is an explicit engineering choice and is **not** released-CLAM coordinate reproduction.

The 256 x 256 RGB patches are resized to 224 x 224 with bilinear interpolation and `antialias=True`, then normalized with ImageNet mean `[0.485, 0.456, 0.406]` and standard deviation `[0.229, 0.224, 0.225]`.

Every artifact must report `effective_mpp=0.4554`, the 8.92% target error, and the label `engineering 2x resampling approximation`.

## Branch scale_4x — target approximately 1.0 MPP

- Target: approximately 1.0 micrometre per pixel.
- Selected native test level: level 1.
- Native downsample: 4.0.
- Actual MPP: 0.9108 x 0.9108.
- Relative target error: 8.92%.
- Existing validated HDF5: `patch_level=1`, `patch_size=256`, 8,911 unique coordinates.

Level 1 is appropriate for the first approximate native 4x branch test. It must still be reported as 0.9108 MPP rather than exact 1.0 MPP.

## GPU/checkpoint gate

Latest measured state (`2026-08-17T16:59:52Z`):

```text
nvidia-smi:                         Tesla T4, driver 580.173.02, 15360 MiB
torch.cuda.is_available():         true
torch.cuda.device_count():         1
torch / torchvision:               2.8.0+cu128 / 0.23.0+cu128
resnet50-11ad3fa6.pth cached:      true
checkpoint SHA256:                 11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca
synthetic ResNet CUDA smoke test:  [1,2048], float32, finite
real extraction or inference run:  no
```

Before touching the slide or coordinates in the resumed run:

1. require `torch.cuda.is_available() == True`;
2. record `torch.cuda.get_device_name(0)`;
3. require a successful `nvidia-smi` result;
4. confirm PyTorch/torchvision compatibility;
5. check the external torch cache for `resnet50-11ad3fa6.pth`;
6. only if GPU is available and the checkpoint is absent, download the official torchvision `IMAGENET1K_V2` checkpoint outside the repository and record SHA256.

Do not switch to CPU inference.

## First real extraction sequence

After the GPU gate passes and the remaining Branch scale-2x implementation details are recorded:

1. generate or load independently validated coordinates for each branch;
2. stream patches directly from the one existing WSI without saving patch images;
3. apply RGB conversion, resize to 224 x 224, and ImageNet normalization;
4. run classifier-removed ImageNet V2 ResNet50 in GPU batches;
5. verify each branch independently: row count, `[N,2048]`, float32, finite values;
6. build row-level provenance for each branch;
7. concatenate with `dim=0` only;
8. verify combined feature/provenance alignment;
9. save validated artifacts atomically and record SHA256;
10. convert the WSI bag to `[1,2048,P]`;
11. load and validate RNA `[1,1,1523]`, mutation `[1,1,1125]`, and CNV `[1,1,193]` from the same patient;
12. pass `[WSI, RNA, mutation, CNV]` as four separate modalities to HEALNet;
13. require a finite output with shape `[1,4]`;
14. stop without training.

A synthetic, random-weight, read-only interface test already verifies these four shapes together and produces a finite `[1,4]` output. It does not substitute for the blocked real-patient demonstration.

The existing official HEALNet repository remains read-only throughout.
