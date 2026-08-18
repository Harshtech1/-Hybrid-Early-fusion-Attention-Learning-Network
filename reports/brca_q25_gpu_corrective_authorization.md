# BRCA Q25 GPU corrective rerun authorization

Status: `BRCA_Q25_GPU_CORRECTIVE_RERUN_AUTHORIZED`

The Q25-only GPU pilot remains authorized within its original narrow scope.
The first attempt exposed a deterministic CUDA configuration omission and
stopped safely before artifact publication. This corrective transition does
not expand the experiment.

## Required launch contract

The runner must be launched with the environment value already present:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  /home/zeus/miniconda3/envs/cloudspace/bin/python \
  scripts/run_brca_q25_gpu_pilot.py \
  --execute-authorized-q25-gpu-pilot
```

The runner refuses any other value or an absent value before its GPU preflight.
It keeps `torch.use_deterministic_algorithms(True)`; determinism is not disabled
or downgraded to warning-only behavior.

## New pre-extraction stop gate

Before constructing ResNet50 or a WSI patch-reading dataset, the runner must
execute the pinned official HEALNet interface on full-size synthetic zeros:

- WSI `[1,9322,2048]`
- RNA `[1,1,1558]`
- mutation `[1,1,21]`
- CNV `[1,1,1333]`

It requires finite `[1,4]` output and all four exact attention shapes:
`[1,2,9322]` plus three `[1,2,1]` tensors. Any exception, absent attention,
shape drift, non-finite value, or CUDA configuration drift stops before WSI
patch reads.

The exact contract passed on the Tesla T4 with a peak allocation of
`266,871,808` bytes. The full synthetic repository suite passes `306/306`.

## Unchanged limits

Only the exact Q25 patient may be processed. Q50/Q75, full-cohort execution,
KIRP, model training, backward or optimizer operations, CPU fallback, AMP,
TF32, WSI transposition, Drive operations, and raw-data deletion remain
prohibited. Official HEALNet and the frozen BLCA tag remain read-only.

The corrective runner must be source-only committed and independently reviewed
before the real rerun. The result must publish atomically only after the real
feature HEALNet attention smoke and every immutable-input/source gate pass.
