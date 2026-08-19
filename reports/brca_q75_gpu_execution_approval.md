# BRCA Q75 GPU execution approval

Status: `BRCA_Q75_GPU_EXECUTION_AUTHORIZED_NOT_YET_RUN`

## Exact user authorization

> I authorize the exact Q75 GPU pilot described in
> reports/brca_q75_gpu_preexecution.md. No training, full-cohort processing,
> Drive operations, deletion, or Q25/Q50/BLCA changes.

The exact 176-character statement has SHA256
`23aae783e0064e4629662584533f5191a1435ea3411ba0a39733c2bcab3dc880`.
It incorporates the complete execution contract in the frozen pre-execution
report rather than widening the project boundary.

## Authorized operation

The execution is restricted to patient `TCGA-E2-A154` and the verified Q75
coordinate set. It may read exactly 13,487 scale-2x patches and 3,458
scale-4x patches, extract float32 2048-dimensional ImageNet1K V2 ResNet50
features, concatenate the branch matrices on CPU in 2x-then-4x order, run one
full-size synthetic and one real-feature four-modality HEALNet numerical smoke
test, and atomically publish the validated six-file `Q75.features` artifact
set outside Git.

The combined feature matrix must be `[16945,2048]`. The HEALNet WSI input is
formed by adding only the batch axis, producing `[1,16945,2048]`; pooling and
transposition are prohibited. RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV
`[1,1,1333]` remain three separate modalities.

## Runtime contract

The future process must start with `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
any CUDA initialization. It must use one Tesla T4, `cuda:0`, batch size 32,
two workers, float32, deterministic algorithms, cuDNN deterministic mode,
and disabled AMP, TF32, and CPU fallback. The full-size synthetic interface
smoke must pass before any pixel-capable dataset is constructed.

## Still prohibited

Training, backward passes, optimizer steps, coordinate regeneration,
operations outside this Q75 patient, Q25/Q50/BLCA changes, full-cohort work,
network or Drive operations, raw or existing-artifact deletion, official
HEALNet modification, pooling, transposition, overwrite, and resume remain
unauthorized.

## Authorization identity

The executable authorization is
`multiscale_feature_pilot/config/brca_q75_gpu_execution_authorization.yaml`,
SHA256
`4594bf3a165cf6c276e355ada0dfe434ee5f0474f32f0aff5390671fb00e17c7`.
The finalized runner SHA256 is
`151ad4f5ba9b04119d6113eb4e26162854e05be4670e89207da2fd6876838a78`.
The gate binds all 19 local runtime sources, the exact WSI and coordinate
hashes, Omic row 771 and modality hashes, the ImageNet1K V2 checkpoint, the
natural tensor layout, and the atomic external output contract.

The finalized CPU verification passed all 59 focused Q75 tests and all 623
repository tests.

## Record-time boundary

This approval transition itself performed no WSI access, patch reads, CUDA
operation, ResNet50 inference, HEALNet execution, artifact publication, or
training. `Q75.features` remained absent. A source-only commit and final CPU
test/review pass are required before asking the user to switch to GPU.
