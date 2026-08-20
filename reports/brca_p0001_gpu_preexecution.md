# BRCA production P0001 GPU feature pre-execution package

Status: **CPU preparation complete; GPU execution remains locked**.

P0001 is cohort index 1 in the frozen 894-patient order and the first member
of the first production block. This is not an additional scientific pilot.
The package binds the verified patient, WSI, Omic row 4, coordinate artifacts,
ImageNet1K V2 checkpoint, compact feature schema, and recovery-v2 transaction.

## Exact future tensor contract

| Branch | Coordinate/patch rows | Source | Feature shape |
|---|---:|---|---:|
| 2× | 13,372 | level 0, 512→256 px | `[13372,2048]` |
| 4× | 3,444 | native level 1, 256 px | `[3444,2048]` |
| **Natural concatenation** | **16,816** | 2× prefix, then 4× suffix | **`[16816,2048]`** |

The HEALNet WSI input is created only by adding the patient batch axis:
`[1,16816,2048]`. No pooling or transpose is permitted. RNA, mutation, and
CNV remain separate inputs with shapes `[1,1,1558]`, `[1,1,21]`, and
`[1,1,1333]`.

The exact uncompressed float32 feature payload is

\[
16{,}816 \times 2{,}048 \times 4 = 137{,}756{,}672\ \text{bytes}.
\]

The compact publisher will retain one canonical CPU float32 tensor and exact
row provenance in four files: `combined_features.pt`, `row_provenance.csv`,
`compact_manifest.json`, and its SHA-256 sidecar. Publication is validated in
a sibling staging directory and committed with Linux `RENAME_NOREPLACE`.

## Encoder and execution contract

The frozen encoder is torchvision ResNet50 with
`ResNet50_Weights.IMAGENET1K_V2`, classifier replaced by identity. The local
checkpoint is 102,540,417 bytes with SHA-256
`11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`.
This is the supervisor-aligned engineering track; it is not described as an
exact paper encoder reproduction because the paper states Kather100K
pretraining.

Future execution is constrained to one Tesla T4, float32, batch size 32, two
workers, deterministic algorithms, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, no AMP,
no TF32, and no CPU fallback. The feature matrix will be validated as finite,
contiguous, detached, and exactly `[16816,2048]` before publication.

## Recovery-v2 binding

The CPU-only rehearsal bound P0001 to transaction
`386116c1-6696-5020-a495-446d6e37b829` and reproduced all eight frozen stages:

```text
PLANNED → ACQUISITION_AUTHORIZED → RAW_VERIFIED
→ HEADER_POLICY_VERIFIED → COORDINATES_VERIFIED
→ GPU_AUTHORIZED → FEATURES_VERIFIED → TERMINAL_RECORDED
```

It produced 16 immutable in-memory events—one start and one validated outcome
per stage—and ended at `STOP_TERMINAL`. A separate rehearsal stopped after
`FEATURES_VERIFIED` with 14 events, which is the future GPU runner's required
durable tip before result review and terminal retention recording. No real
ledger file was written during preparation.

## Runtime and storage estimates

These are planning estimates, not measurements for P0001:

- T4 wall time: **180–260 seconds** (approximately 3.0–4.3 minutes);
- final compact artifact: **138.5–139.0 MB**;
- maximum incremental disk allowance including staging: **278 MB**; and
- expected peak GPU allocation: **481–485 MB**.

The estimates use P0001's exact row count and completed B01, B02, B03, and Q75
T4 measurements. Patch decoding and preprocessing, rather than GPU memory,
are expected to dominate.

## Fail-closed boundary

The runner's first operational statement checks `EXECUTION_AUTHORIZED`, which
is `False`. The authorization SHA is deliberately non-hexadecimal, the future
execution authorization file is absent, and both the P0001 feature output and
recovery ledger paths are absent. A real CLI attempt stops before Torch import,
dependency/path validation, WSI access, CUDA, checkpoint loading, model
construction, HEALNet, publication, or ledger writes.

During preparation there were zero WSI opens, patch reads, CUDA operations,
feature extractions, HEALNet calls, artifact publications, recovery-ledger
writes, P0002–P0008 operations, Drive operations, deletions, cohort expansion,
or training runs.

## Exact next authorization

> I authorize the exact P0001-only production GPU feature execution using the frozen verified P0001 coordinates: 13,372 scale-2x and 3,444 scale-4x patch reads; float32 ResNet50 ImageNet1K V2 feature extraction; natural row concatenation into [16816,2048]; atomic compact artifact publication; recovery-v2 GPU_AUTHORIZED and FEATURES_VERIFIED ledger recording; and synthetic plus real-feature four-modality HEALNet numerical smoke tests. No training, backward pass, optimizer step, AMP, TF32, CPU fallback, coordinate regeneration, P0002–P0008 processing, Q25/Q50/Q75/B01/B02/B03/B06 or BLCA changes, Drive operations, cohort expansion, raw/pre-existing/user/project/final-artifact deletion, or official HEALNet modification; only cleanup of runner-created ephemeral recovery staging is permitted. Stop after artifact validation, recovery recording, and reporting.

Until that separate authorization is received and recorded, no GPU switch is
needed and the package remains non-executable.
