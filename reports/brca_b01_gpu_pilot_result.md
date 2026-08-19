# BRCA B01 compact GPU feature pilot result

Status: **SUCCESS**. The authorized B01-only Tesla T4 run completed and stopped at artifact validation. This is an engineering feature/interface pilot, not a trained survival prediction.

## Measured result

| Metric | B01 result |
|---|---:|
| 2× patches | 3,773 |
| 4× patches | 969 |
| Total WSI tokens | 4,742 |
| 2× extraction time | 56.24 s |
| 4× extraction time | 10.01 s |
| Complete successful run | 73.46 s |
| Peak GPU allocation | 481,334,784 B (~459 MiB) |
| Combined tensor | `[4742, 2048]`, float32 |
| Natural HEALNet WSI input | `[1, 4742, 2048]` |
| Compact artifact size | 39,092,076 B (39.09 MB) |

The 2× level-0 branch used 512×512 source footprints resampled with Lanczos to 256×256 (effective 0.4936 µm/px). The 4× branch used native level-1 256×256 reads without resampling (effective 0.9872376717 µm/px). Rows were concatenated in the fixed order `2×` then `4×`; no pooling or transpose was performed.

Both the pre-extraction synthetic and real-feature four-modality HEALNet smokes passed. Inputs were WSI `[1,4742,2048]`, RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]`. Both produced finite output `[1,4]`; WSI attention was `[1,2,4742]`. The HEALNet instance was random-initialized and evaluation-only.

## Compact artifact

The external directory `BRCA_BATCH_B01.features` contains exactly four files. Its manifest SHA256 is `f20556d3e64185879d6bc9a5f837dafc72ebb88835506fbc90fd4e5c61c2a90f`. The combined tensor file SHA256 is `b23b2c1a028103223bc509a889adabcdd2d4bcb89e54a3aed800e6be842a1e34`, with tensor-content SHA256 `db4ca27eac04293bb878b4255044479bf8c3965fb250e4b1ae826cfd3ef12d01`. Atomic publication and independent strict validation passed.

Two earlier invocations failed closed before patch reads or artifact creation: one exposed a protected-Git-status parser edge case, and one exposed a keyword-only HEALNet helper call mismatch. Both were corrected, regression-tested, and committed before the successful run.

No training, backward pass, optimizer step, AMP, TF32, CPU fallback, coordinate regeneration, other-patient processing, Drive operation, deletion, cohort expansion, or official HEALNet modification occurred.

Next decision: supervisor review is required before any B02–B06 processing or training.
