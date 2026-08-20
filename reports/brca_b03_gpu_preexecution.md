# B03 GPU feature-pilot pre-execution package

Status: **CPU preparation complete; GPU execution locked**.

The package binds B03 patient `TCGA-AR-A1AY`, verified coordinate manifest `cced78f…1ae5`, 8,875 scale-2× rows, 2,257 scale-4× rows, exact Omic row 372, and the local ResNet50 ImageNet1K V2 checkpoint. The future canonical feature matrix is `[11132,2048]`; the natural HEALNet WSI input is `[1,11132,2048]`. RNA, mutation, and CNV remain separate inputs with shapes `[1,1,1558]`, `[1,1,21]`, and `[1,1,1333]`.

The user has frozen ImageNet1K V2 as the engineering checkpoint. Its local file is 102,540,417 bytes with SHA256 `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. This preserves comparability with Q25, Q50, Q75, B01, and B02. It must still be described as an engineering implementation difference because the paper states Kather100K pretraining.

The future process is deterministic float32 on one Tesla T4: batch size 32, two workers, no AMP, no TF32, no CPU fallback, sequential 2× then 4× extraction, natural `torch.cat(..., dim=0)`, one compact canonical tensor, exact row provenance, and synthetic plus real-feature four-modality HEALNet numerical smokes.

## Estimates, not measurements

The raw combined float32 tensor is exactly:

\[
11132\times 2048\times 4 = 91{,}193{,}344\ \text{bytes}.
\]

Based on completed B01/B02 compact artifacts and the five completed T4 pilots, the B03 run is budgeted at **160–200 seconds** and approximately **91.6–91.8 MB** of compact files. Expected peak GPU allocation remains approximately **481–485 MB**. These values are estimates until the separately authorized B03 GPU execution completes.

## Fail-closed boundary

The runner's first operational statement checks `EXECUTION_AUTHORIZED`, which is currently `False`. Its authorization SHA is deliberately pending, the executable authorization record does not exist, and `BRCA_BATCH_B03.features` is absent. Accidental CLI invocation therefore stops before input validation, WSI access, OpenSlide, CUDA, checkpoint loading, model construction, HEALNet, or publication.

No B03 WSI open, patch read, CUDA operation, feature extraction, HEALNet call, feature publication, B04–B06 processing, deletion, Drive access, cohort expansion, or training occurred during preparation.

## Exact authorization required later

> I authorize the exact B03-only GPU feature pilot using the frozen verified B03 coordinates: 8,875 scale-2x and 2,257 scale-4x patch reads; float32 ResNet50 ImageNet1K V2 feature extraction; natural row concatenation into [11132,2048]; atomic compact artifact publication; and synthetic plus real-feature four-modality HEALNet numerical smoke tests. No training, backward pass, optimizer step, AMP, TF32, CPU fallback, coordinate regeneration, B04–B06 processing, Q25/Q50/Q75/B01/B02 or BLCA changes, Drive operations, deletion, cohort expansion, or official HEALNet modification. Stop after artifact validation and reporting.
