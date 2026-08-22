# BRCA pilot execution approval request

Historical status: `PENDING_SUPERVISOR_EXECUTION_APPROVAL`

This request has been answered and is superseded by
`reports/brca_phase2_q25_approval_transition.md`. Only the Q25 sequential
acquisition/metadata scope recorded there is currently executable; this
historical request is not an active gate.

Phase 2 CPU preflight is now technically ready. The verified Linux GDC client,
three separate one-row manifests, storage boundary, and metadata-only policy
all pass their local checks. The executable gate still reports
`acquisition_authorized: false` and `ready_to_download: false`; preparation did
not broaden the supervisor's earlier cohort choice into slide-download
permission.

The cohort decision is resolved: BLCA is the frozen reference and BRCA is the
next supervisor-aligned cohort. CPU validation found 894 exact singleton
BRCA patients with genuine RNA, mutation, and CNV inputs. All 62 multi-WSI
patients remain excluded.

Phase 1 also fixes the tensor semantics: every WSI remains a separate
variable-length patch bag `[P_i,2048]`, supplied to HEALNet at batch size one
as `[1,P_i,2048]`. There is no cross-patient patch concatenation, no per-WSI
global pooling, no padding, and no mask in the initial pilot.

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

The scale-policy approval should be recorded explicitly as well:

> For the metadata gate, may I use a native-level-only rule with a maximum 10%
> MPP error on both x and y axes for the approximately 0.5 and 1.0 µm/px
> branches, rejecting any slide that cannot provide two distinct acceptable
> levels? I will not resample silently; if resampling is preferred, I will stop
> and document a separate deterministic rule first.

If the supervisor approves only the downloads but not extraction, remain on
CPU and stop after hash plus MPP/pyramid inspection. A GPU switch is required
only immediately before approved ResNet50 extraction and the real-input
HEALNet interface smoke test.
