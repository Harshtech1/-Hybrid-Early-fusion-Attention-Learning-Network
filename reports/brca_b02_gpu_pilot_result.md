# BRCA B02 compact GPU feature pilot result

Status: **SUCCESS**. The authorized B02-only Tesla T4 run completed and stopped after compact-artifact validation. This is an engineering feature/interface pilot, not a trained survival prediction.

## Measured result

| Metric | B02 result |
|---|---:|
| 2× patches | 7,158 |
| 4× patches | 1,862 |
| Total WSI tokens | 9,020 |
| 2× extraction time | 94.57 s |
| 4× extraction time | 18.59 s |
| Complete run | 124.94 s (2.08 min) |
| Peak GPU allocation | 483,956,224 B (~462 MiB) |
| Combined tensor | `[9020, 2048]`, float32 |
| Natural HEALNet WSI input | `[1, 9020, 2048]` |
| Compact artifact size | 74,342,370 B (74.34 MB) |

The 2× branch read 512×512 level-0 footprints and resampled them with Lanczos to 256×256, giving 0.501 µm/px. The 4× branch used native level-1 256×256 reads without resampling, giving 1.0020374312 µm/px. Rows were concatenated in fixed `2×` then `4×` order; no pooling or transpose was performed.

Both synthetic and real-feature four-modality HEALNet numerical smokes passed. Inputs were WSI `[1,9020,2048]`, RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]`. Both produced finite output `[1,4]`; WSI attention was `[1,2,9020]`. The HEALNet instance was random-initialized and evaluation-only.

## Compact artifact

The external `BRCA_BATCH_B02.features` directory contains exactly four files. Its manifest SHA256 is `07bc593acf5cad0712154b7eb05771661439412047cc3a98a48c4c1ffb116dec`. The combined tensor file SHA256 is `ef314a798519ab61fc36c27bbbeed942e0db8003626d48a15a0e0f1b4342fcc6`, with tensor-content SHA256 `2cbdc7ed988024ab6d52795c7a5d41b3a1a675039ddb6f9b5e628bec003aed8c`. Atomic no-overwrite publication and an independent strict CPU validation passed.

No training, backward pass, optimizer step, AMP, TF32, CPU fallback, coordinate regeneration, B03–B06 processing, protected-pilot modification, Drive operation, deletion, cohort expansion, or official HEALNet modification occurred.

Next gate: switch back to CPU for B02 comparison/review and B03 planning. Further GPU work requires a separate authorization.
