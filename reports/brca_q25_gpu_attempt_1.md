# BRCA Q25 GPU attempt 1 — blocked safely

Status: `BRCA_Q25_GPU_ATTEMPT_1_BLOCKED_RECOVERABLE_CONFIGURATION`

The first Q25 GPU attempt completed both ResNet50 feature-extraction branches
in memory, then stopped at the mandatory HEALNet attention check. It did not
publish feature artifacts and did not train a model.

## What failed

The visible exception was:

```text
SupervisorHealNetSmokeError: cross-attention modality 0 did not run
```

The underlying CUDA exception was deterministic cuBLAS configuration. The
runner enabled `torch.use_deterministic_algorithms(True)`, but the process had
not been launched with `CUBLAS_WORKSPACE_CONFIG`. Official HEALNet v0.1.0
catches cross-attention exceptions internally, so it produced a finite
`[1,4]` value even though all four attention tensors were absent. Our stricter
attention gate correctly rejected that value.

This was not a WSI, coordinate, Omic, tensor-orientation, memory-capacity, or
ResNet50 extraction failure.

## Safe postcondition

- Q25 feature output directory: absent
- feature staging directory and lock: absent
- persisted `.pt` files: zero
- active Q25 GPU processes: zero
- training, backward passes, and optimizer steps: zero
- Q50/Q75, full-cohort, Drive, and deletion operations: zero
- official HEALNet checkout and frozen BLCA tag: unchanged

The 7,404 scale-2x and 1,918 scale-4x embeddings existed only in process
memory and were released when the blocked process exited.

## Corrective verification

The deterministic contract is now fixed to exactly:

```text
CUBLAS_WORKSPACE_CONFIG=:4096:8
torch.use_deterministic_algorithms(True)
```

With that process environment, an exact full-size synthetic BRCA contract
check passed on the Tesla T4:

- WSI input: `[1,9322,2048]`
- RNA: `[1,1,1558]`
- mutation: `[1,1,21]`
- CNV: `[1,1,1333]`
- output: finite `[1,4]`
- attention: `[1,2,9322]`, then three `[1,2,1]` tensors
- peak allocated GPU memory: `266,871,808` bytes

The corrected runner requires the environment value before its first CUDA
operation and repeats this exact synthetic attention check before constructing
ResNet50 or any pixel-reading dataset. It will therefore fail before WSI patch
reads if this CUDA contract regresses.

The original pre-execution approval record is retained as historical evidence.
The corrected authorization supersedes its runtime configuration only; all
scope limits remain unchanged. This remains an interface pilot, not a trained
survival prediction.
