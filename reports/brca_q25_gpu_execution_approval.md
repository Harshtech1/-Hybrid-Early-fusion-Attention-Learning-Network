# BRCA Q25 GPU execution approval

Status: `BRCA_Q25_GPU_EXECUTION_AUTHORIZED_NOT_YET_RUN`

The separate Q25 GPU gate is now implemented, reviewed, and fail-closed. The
Tesla T4 is available in the approved execution context, every immutable input
has passed its exact identity/hash check, and the complete synthetic suite
passes. No Q25 patch pixels or models had been executed when this approval
record was written.

## Exact bounded operation

The next command may do only the following for `TCGA-LL-A6FP`:

1. Revalidate the exact WSI, coordinate artifacts, BRCA Omic row, ImageNet1K
   V2 checkpoint, official HEALNet commit, frozen BLCA tag, GPU, storage path,
   authorization, and tracked implementation bytes.
2. Read 7,404 approved level-0 `512×512` footprints, convert to RGB, and resize
   each to `256×256` with PIL Lanczos for the effective `0.505` MPP branch.
3. Read 1,918 approved native level-1 `256×256` footprints for the effective
   `1.0100149842739303` MPP branch.
4. Apply the locked RGB → bilinear-antialiased `224×224` → ImageNet
   normalization preprocessing and classifier-removed ImageNet1K V2 ResNet50.
5. Preserve branch rows as `[7404,2048]` and `[1918,2048]`, then perform only
   `torch.cat([scale_2x, scale_4x], dim=0)` to obtain `[9322,2048]`.
6. Publish the exact external six-file feature set through an atomic,
   no-overwrite directory transaction.
7. Release ResNet50, add only the batch axis to create `[1,9322,2048]`, pair it
   with the same patient's RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV
   `[1,1,1333]`, and run the random-initialized four-input HEALNet interface
   smoke.
8. Require finite `[1,4]` output and all four attention tensors, then stop.

This approval does not permit a transpose to `[1,2048,9322]`, pooling, mixed
precision, TF32, CPU fallback, training, backward/optimizer steps, Q50/Q75,
full-cohort work, KIRP, Drive operations, WSI deletion, or modification of the
official HEALNet checkout or frozen BLCA tag.

## Fixed identities

- WSI SHA256: `ac852be806eb0d91214145983319b604919a8e1d16bd59378a3dba69a600979c`
- Coordinate manifest SHA256: `7d64ec37595792994e61ab3bf60461498e805bbff0429599f0328b49c93d2ad2`
- BRCA Omic archive SHA256: `4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`
- ResNet50 checkpoint SHA256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`
- Execution authorization SHA256: `0195f1f2d631a3a29c01077aa455b8a0f20a6b2c885988fcb162ba7b1bfd8805`
- Official HEALNet commit: `28ba5da6ab99fd8069972c22e986d83edb658dd4`
- Frozen BLCA tag commit: `df7cf2bda783ab6cc09e95d6a1fa0914da05a433`

The gate uses batch size 32, two ordered data workers, float32 only, deterministic
PyTorch/cuDNN algorithms, and TF32 disabled. The reviewed full synthetic suite
passed `303/303`. Real execution requires a source-only commit first; the
runner refuses any critical file whose worktree bytes differ from that commit.
