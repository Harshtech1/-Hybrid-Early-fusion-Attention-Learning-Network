# BRCA Q50 GPU execution approval

Status: `BRCA_Q50_GPU_EXECUTION_AUTHORIZED_NOT_YET_RUN`

The user explicitly authorized one bounded Q50 GPU operation:

> I authorize Q50-only GPU ResNet50 ImageNet1K V2 feature extraction and the
> four-modality HEALNet numerical smoke test. No Q75, full-cohort processing,
> or training.

This record does not itself execute the operation. The Q50 extraction and
artifact modules are complete, their source hashes are finalized, all 90 Q50
tests pass, and the full 396-test synthetic suite passes. The runner remains
fail-closed until the complete source-only gate is committed; real execution
must use that exact commit.

## Exact bounded operation

For only `TCGA-AR-A1AW`, the authorized runner may:

1. Revalidate the exact Q50 WSI, coordinate set, same-patient BRCA Omic row,
   ImageNet1K V2 checkpoint, official HEALNet checkout, frozen BLCA tag, GPU,
   output path, authorization, and committed implementation bytes.
2. Before constructing a pixel-capable dataset, require the launch-time
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` environment and run a full-size synthetic
   four-modality HEALNet contract smoke using WSI shape `[1,10793,2048]`.
3. Sequentially read the 8,580 approved scale-2x level-0 `512×512` footprints,
   resize each to `256×256` with PIL Lanczos, then read the 2,213 approved
   native scale-4x level-1 `256×256` footprints.
4. Apply classifier-removed torchvision ResNet50 with
   `ResNet50_Weights.IMAGENET1K_V2`, float32 only, batch size 32, and two
   ordered workers.
5. Preserve feature rows as `[8580,2048]` followed by `[2213,2048]`, and use
   only `torch.cat([scale_2x, scale_4x], dim=0)` to produce `[10793,2048]`.
6. Add only a batch axis, producing natural HEALNet layout `[1,10793,2048]`.
   Pair it with separate RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV
   `[1,1,1333]` tensors for the same patient.
7. Require a finite random-initialized HEALNet `[1,4]` output and finite
   attention tensors before atomically publishing and validating the exact
   external six-file Q50 feature artifact set.
8. Stop.

The authorization prohibits WSI-feature transposition, pooling, mixed
precision, TF32, CPU fallback, backward/optimizer steps, model training, Q75,
full-cohort work, additional downloads or network operations, Google Drive,
raw-WSI deletion, and modification of the official HEALNet checkout or frozen
BLCA reference.

## Fixed identities

- Q50 patient: `TCGA-AR-A1AW`
- Q50 GDC UUID: `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62`
- WSI SHA256: `6b960db7e6e5772f4f626daa7b023ffbb5e3b20f9f6ff5c7d52f4554cf33399b`
- Coordinate manifest SHA256: `a79179e7fe3a0cf6bdda5089da11910c0db6b1f4bcab9a15c6d5b8e7ec43e485`
- BRCA Omic archive SHA256: `4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`
- ResNet50 checkpoint SHA256: `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`
- Execution authorization SHA256: `2bfe61f7e17668ec50aeb6f3d2b214e0a3dd91e4f47692610880bec8afb2b080`
- Q50 extraction adapter SHA256: `7e2b92c672ca7dc0b7b683a66ea4ecf5aa2fdf7ea14811ea7960da08470d7181`
- Q50 artifact publisher SHA256: `369830a2ba4f0bdbf8646599f70cf430e7a439cf87988f80014c54695176b9ac`
- Q50 execution runner SHA256: `ada9cf9077573a7d02077182ae1e3e42e0c8fc742aa185a231ad74a8bc0d9a6b`
- Official HEALNet commit: `28ba5da6ab99fd8069972c22e986d83edb658dd4`
- Frozen BLCA tag commit: `df7cf2bda783ab6cc09e95d6a1fa0914da05a433`

The HEALNet smoke is an interface and numerical test with randomly initialized
HEALNet weights. It is not training, inference from a trained model, or a
scientific survival prediction.
