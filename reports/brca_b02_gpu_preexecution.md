# B02 GPU feature-pilot pre-execution package

Status: **CPU preparation complete; GPU execution locked**.

The package binds B02 patient `TCGA-BH-A0BG`, the verified coordinate manifest `2b3e5dd7…52ba2`, 7,158 scale-2× rows, 1,862 scale-4× rows, Omic row 472, the ImageNet1K V2 ResNet50 checkpoint, and natural combined layout `[9020,2048]` / HEALNet input `[1,9020,2048]`.

The future process is deterministic float32 on one Tesla T4 with batch size 32, two workers, no AMP, no TF32, no CPU fallback, sequential 2× then 4× extraction, one canonical combined tensor, row provenance, and synthetic plus real-feature four-modality HEALNet numerical smokes.

The frozen completed B02 result already records **124.94 seconds** on a Tesla T4, a **74,342,370-byte** compact artifact, and **483,956,224 bytes** peak GPU allocation. A future controlled rerun would therefore be budgeted at approximately **120–140 seconds** and 74.4 MB. The raw combined tensor is 73,891,840 bytes.

The runner's first operational gate is `EXECUTION_AUTHORIZED=False`. A historical authorization and validated B02 artifact exist from the completed run, but the current CPU-only instruction does not authorize a rerun. Accidental invocation therefore stops before input checks, WSI access, OpenSlide, CUDA, model construction, or output publication; the existing artifact is also protected by no-overwrite output checks.

## Exact authorization required later

> I authorize the exact B02-only GPU feature pilot using the frozen verified B02 coordinates: 7,158 scale-2x and 1,862 scale-4x patch reads; float32 ResNet50 ImageNet1K V2 feature extraction; natural row concatenation into `[9020,2048]`; atomic compact artifact publication; and synthetic plus real-feature four-modality HEALNet numerical smoke tests. No training, backward pass, optimizer step, AMP, TF32, CPU fallback, coordinate regeneration, B03–B06 processing, Q25/Q50/Q75/B01 or BLCA changes, Drive operations, deletion, cohort expansion, or official HEALNet modification. Stop after artifact validation and reporting.
