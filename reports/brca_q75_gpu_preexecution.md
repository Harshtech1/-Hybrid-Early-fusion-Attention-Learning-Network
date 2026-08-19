# BRCA Q75 GPU pre-execution package

Status: `Q75_GPU_PREEXECUTION_PREPARED_EXECUTION_LOCKED`

Authorization status: `AWAITING_SEPARATE_USER_AUTHORIZATION`

This is a CPU-only design and review package. It prepares a fail-closed Q75
feature adapter, external artifact publisher, and exact one-patient runner. It
does **not** authorize or perform WSI hashing, OpenSlide access, patch reads,
CUDA initialization, ResNet50 or HEALNet execution, feature publication, or
training. No executable GPU authorization configuration has been created.

## Exact frozen input boundary

The future pilot is bound to only patient `TCGA-E2-A154`, full slide ID
`TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`, and GDC
UUID `25aec062-60d1-446e-a1c6-0c79cc74a770`. The WSI identity remains
`1,360,743,825` bytes, MD5 `a8c4b68fb6e0ab3e862efe3ed1fe10d7`, and SHA256
`844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37`.
These values are inherited from the committed coordinate result; this CPU
preparation stage did not reopen or rehash the WSI.

The completed coordinate result is bound by SHA256
`1b583b9366a5308c26ee74de3686f080992f9381dc93b8b2d8f6729854e4be17`.
Its atomic coordinate manifest SHA256 is
`438165ce6b3be9d26d66c65cd70793e29cc92208cfb6a78bf68043bc4b4a4e90`.
The only future patch plan is:

| Branch | Verified coordinates | Future pixel operation | Effective MPP | Expected features |
|---|---:|---|---:|---:|
| 2x | 13,487 | level 0 `512×512`, Lanczos to `256×256` | `0.4936` | `[13487,2048]` |
| 4x | 3,458 | native level 1 `256×256`, no resampling | `0.9872163682185965` | `[3458,2048]` |

The expected combined tensor is exactly
`torch.cat([features_2x, features_4x], dim=0)` with shape `[16945,2048]`.
The 2x rows must be the exact prefix and the 4x rows the exact suffix. Pooling
and transposition are prohibited. Adding only a batch dimension produces the
natural HEALNet WSI shape `[1,16945,2048]`.

The same-patient Omic binding remains exact row `771` from the clean archive
with SHA256
`4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`:
RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]`. They remain
three separate modalities and are never concatenated with WSI features. The
encoder is classifier-removed torchvision ResNet50 with
`ResNet50_Weights.IMAGENET1K_V2`; checkpoint SHA256 is
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.

## Prepared future runtime contract

Only after a new exact authorization, the runner would require the Tesla T4
and `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA. It fixes seed 0, float32,
batch size 32, two workers, deterministic PyTorch/cuDNN behavior, sequential
2x-then-4x extraction, and CPU concatenation. AMP, TF32, cuDNN benchmarking,
CPU feature fallback, backward passes, and optimizer steps remain disabled.

Before constructing a pixel-capable dataset, it would run a full-size
synthetic four-modality HEALNet check. After feature extraction it would repeat
the numerical check with real features. Each must produce finite float32
output `[1,4]` and attention shapes `[1,2,16945]`, `[1,2,1]`, `[1,2,1]`, and
`[1,2,1]`. HEALNet would be randomly initialized in evaluation/inference mode;
this is a shape and finiteness check, not training or scientific prediction.

Successful future artifacts would be an exact six-file no-overwrite set at
`/teamspace/studios/this_studio/brca_pilot_data/Q75.features`, outside Git,
published atomically only after every validation succeeds. Existing output,
lock, or staging paths make the runner fail closed; resume and replacement are
not allowed.

## Resource estimate, not a Q75 result

The exact raw float32 payloads implied by the frozen shapes are 110,485,504
bytes for 2x, 28,327,936 bytes for 4x, and 138,813,440 bytes for their combined
copy. Retaining all three tensors therefore requires exactly 277,626,880 raw
payload bytes before serialization and provenance overhead.

Extrapolating from the verified Q25 and Q50 six-file results gives about
278.49–278.52 MB decimal retained, so 279 MB is the planning value. Scaling
the two prior total T4 runtimes by Q75's 16,945 coordinate rows gives about
269–284 seconds (roughly 4.5–4.8 minutes); six minutes is a sensible planning
allowance. Fixed-batch ResNet extraction previously peaked at 481,334,784
allocated bytes, while a linear HEALNet token estimate is about 444 MB. A
0.55 GB allocated-memory planning figure leaves modest model-side headroom,
but CUDA context/driver reservation is separate. Slide I/O can move the runtime,
and only an authorized Q75 run can measure the actual values.

## CPU verification

The Q75 adapter, publisher, and runner focused suite passed `59/59`; the full
pilot suite passed `623/623`. Python compilation, YAML parsing, source-hash
checks, and diff checks passed. The finalized runner SHA256 is
`93436c9f164b74c3e8e2fc513c1ede99fd090c2b4c5601de5a7be5b4a4d949ea`;
the adapter and publisher SHA256 values are
`5b2c3c426eab17b9312cdd692e81c35c4dac552d47e0f9cddd07a05df1e2abd1`
and `df927ba75f8cfa1865f69b31073b1fab36f7480ad652c1ecc7c2e81488e8a280`.

The runner has `EXECUTION_AUTHORIZED=False` and a deliberately non-final
authorization hash. Its first executed gate rejects before WSI hashing,
OpenSlide, CUDA, model construction, dependency use, path probing, or a
pixel-capable dataset. Its committed-source closure also binds the local
package initialization chain imported by the CLI before that runtime gate.
The executable Q75 GPU authorization file and
`Q75.features` are both absent. This preparation stage performed zero WSI
accesses, patch reads, CUDA operations, model executions, feature writes, and
training runs.

The official HEALNet checkout remains pinned read-only at
`28ba5da6ab99fd8069972c22e986d83edb658dd4`; the frozen BLCA commit remains
`df7cf2bda783ab6cc09e95d6a1fa0914da05a433`. Q25/Q50, the protected BLCA
report, full-cohort data, Google Drive, and raw data are outside this stage.

## Exact next authorization, not yet granted

The following is proposed future authorization text. Its presence here is not
authorization and must not be interpreted as one:

> I authorize the exact Q75-only GPU pilot for patient TCGA-E2-A154 using the
> frozen verified Q75 coordinates: read-only input/hash/header/Omic/checkpoint
> preflights; 13,487 scale-2x and 3,458 scale-4x patch reads; float32 ResNet50
> ImageNet1K V2 feature extraction;
> torch.cat([features_2x, features_4x], dim=0) with no transpose or pooling;
> atomic publication to Q75.features; and synthetic plus real-feature
> four-modality HEALNet numerical smoke tests. No training, backward pass,
> optimizer step, AMP, TF32, CPU fallback, coordinate regeneration,
> Q25/Q50/BLCA changes, full-cohort processing, network or Google Drive
> operations, raw-file deletion, or official HEALNet modification. Stop after
> validated Q75 feature artifacts and smoke-test reporting.

Until the user supplies that exact authorization and a source-only gate is
committed, remain on CPU and do not run the Q75 pilot. The next physical step
after authorization is switching to a Tesla T4; training remains a later,
separate project gate.
