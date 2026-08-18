# BRCA pilot execution approval request

Status: `PENDING_SUPERVISOR_EXECUTION_APPROVAL`

The cohort decision is resolved: BLCA is the frozen reference and BRCA is the
next supervisor-aligned cohort. CPU validation found 894 exact singleton
BRCA patients with genuine RNA, mutation, and CNV inputs. All 62 multi-WSI
patients remain excluded.

Before any WSI is downloaded, request explicit approval for:

1. Only the three UUIDs listed in `reports/brca_three_patient_plan.md`
   (Q25/Q50/Q75 by declared WSI size; total 2,984,417,159 bytes).
2. Initial concurrency of one and one-row GDC manifests.
3. Metadata-first inspection: verify size/MD5, then read only MPP and pyramid
   metadata before any coordinate generation.
4. Fail closed when a slide cannot support the approved approximately
   0.5/1.0 µm/px scale policy; no silent level assumption or resampling.
5. Keep all multi-WSI patients excluded from this pilot.
6. Use the frozen BLCA ImageNet1K_V2 ResNet50 checkpoint for direct engineering
   comparability, while explicitly recording that the paper described
   Kather100K weights.
7. No full-cohort acquisition and no training after this pilot without a new
   approval.

Suggested WhatsApp message:

> Thank you, sir. I’ll keep BLCA as the frozen reference and proceed with
> BRCA. The official BRCA data has all four inputs and gives 894 unambiguous
> single-slide patients; 62 multi-slide patients will be excluded. Before any
> full-cohort work, I propose a three-patient BRCA pilot (lower-quartile,
> median, and upper-quartile WSI sizes; 2.98 GB total), downloaded one at a
> time from GDC.
> I’ll verify each hash and MPP/pyramid first, then use the same frozen BLCA
> ImageNet1K_V2 pipeline for comparability. May I proceed with only these three
> WSI downloads and the pilot extraction? No training or full-cohort download
> will start.

If the supervisor approves only the downloads but not extraction, remain on
CPU and stop after hash plus MPP/pyramid inspection. A GPU switch is required
only immediately before approved ResNet50 extraction and the real-input
HEALNet interface smoke test.
