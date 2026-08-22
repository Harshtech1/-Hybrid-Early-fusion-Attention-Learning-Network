# Draft email to supervisor

**Subject:** TCGA-BRCA HEALNet progress, GPU timing, and guidance requested

Dear Professor,

I am writing to share a concise update on the TCGA-BRCA HEALNet implementation
and request your guidance on the next execution phase.

We have completed three deterministic engineering pilots representing the
25th, 50th, and 75th percentiles of BRCA WSI file size. For all three patients,
we verified exact WSI–Omic identity, generated tissue coordinates at two
physical scales, extracted ImageNet1K V2 ResNet50 features on a Tesla T4, and
passed the four-modality HEALNet numerical interface test using WSI, RNA,
mutation, and CNV as separate inputs.

The principal results are:

| Pilot | Total WSI tokens | CPU coordinate time | Complete GPU pilot time | Feature artifacts |
|---|---:|---:|---:|---:|
| Q25 | 9,322 | 5.92 s | 155.79 s | 153.21 MB |
| Q50 | 10,793 | 9.50 s | 171.43 s | 177.40 MB |
| Q75 | 16,945 | 31.54 s | 245.16 s | 278.53 MB |

All HEALNet smoke outputs were finite with shape `[1,4]`, and the attention
length correctly matched each patient's WSI token count. These were
randomly initialized numerical/interface tests rather than trained survival
predictions. Training has not started.

The GPU runtime increased with patch count, as expected. The dominant time is
WSI reading and patch preprocessing rather than GPU memory; peak ResNet
allocation remained approximately 482 MB on the 15 GB T4. The Q75 run also had
one prepublication interruption caused by the interactive session. It left no
partial artifacts or active process, and the controlled retry completed
successfully in 245.16 seconds. For the full 894-patient singleton cohort, the
three pilots imply approximately 39–61 T4 GPU-hours, with an estimated 2–4
days of sequential wall time after including downloads, CPU coordinate work,
validation, and possible interruptions.

We have aligned 894 unambiguous singleton BRCA patients. Their raw WSI inventory
is approximately 918.53 GB, so it cannot be retained simultaneously under the
200 GB Lightning quota. We have implemented and synthetically tested a compact
artifact layout that projects to approximately 69–125 GB, together with an
append-only recovery ledger. No cohort extraction, raw deletion, or model
training has been performed.

Since the previous update, we accepted a six-patient validation batch spanning
the WSI-size distribution. The first patient, B01 (`TCGA-GI-A2C8`), has now
passed exact download, checksum, WSI–Omic identity, and header-only validation.
Its native resolution is 0.2468 µm/px with a three-level pyramid. No pixels
were read during that inspection. We have completed the CPU-only scale and
coordinate-policy review, but coordinate execution and GPU feature extraction
remain separately gated.

I would appreciate your guidance on the following points:

1. Do you approve completing the accepted six-patient validation batch before
   proceeding to the full 894 patients?
2. Should we reserve one continuous GPU window or use restartable GPU batches?
3. Is retaining one canonical combined WSI tensor, with 2x/4x branch ranges and
   complete row provenance, acceptable for the cohort?
4. What should be the raw WSI retention/deletion policy after verified feature
   publication?
5. Should we continue with ImageNet1K V2 ResNet50 for the engineering study, or
   must we switch to Kather100K weights for closer paper reproduction?
6. Which train/validation/test split and survival-evaluation protocol would you
   like us to freeze before training?

The immediate next step is one bounded CPU coordinate run for B01, subject to
separate authorization. We will need the GPU again only after B01 coordinates
are independently validated and feature extraction is explicitly approved.

I have attached the updated progress report with the architecture, measured
matrix, timing, storage projections, and remaining milestones.

Kind regards,

[Your Name]
