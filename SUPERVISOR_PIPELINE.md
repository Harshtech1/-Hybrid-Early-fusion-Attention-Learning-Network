# Supervisor-aligned HEALNet pipeline

This is the primary implementation track. Supervisor instructions take precedence; the paper and released code remain explicit provenance references.

## Patient-level architecture

```text
                         SAME PATIENT
                              │
                ┌─────────────┴─────────────┐
                │                           │
               WSI                        OMICS
                │                    ┌──────┼──────┐
         ┌──────┴──────┐            RNA  Mutation CNV
         │             │
       2x WSI         4x WSI
     0.4554 MPP      0.9108 MPP
         │             │
   256x256 patches  256x256 patches
         │             │
      ResNet50      ResNet50
  ImageNet1K V2, shared encoder policy
         │             │
    [N1,2048]     [N2,2048]
         └──────┬──────┘
                │
       torch.cat(..., dim=0)
                │
        [N1+N2,2048]
                │
             transpose
                │
        [1,2048,N1+N2]
                │
                ├──────── RNA      [1,1,1523]
                ├──────── Mutation [1,1,1125]
                └──────── CNV      [1,1,193]
                │
             HEALNet
                │
       shared latent attention
                │
              [1,4]
```

## Fixed rules

1. Match using the stable TCGA patient/case identifier. Never pair by row or directory order.
2. Keep only patients with matching WSI and Omic data; discard mismatches and missing modalities.
3. Use the original high-resolution WSI, not a thumbnail. “Around four gigapixels” is descriptive unless the supervisor later makes it prescriptive.
4. Extract non-overlapping 256x256 patches at two scale branches.
5. Use classifier-removed torchvision `ResNet50_Weights.IMAGENET1K_V2`, producing float32 2048-D patch features.
6. Combine WSI branches only with `torch.cat([features_2x, features_4x], dim=0)`.
7. Preserve all scale-2x rows first, then all scale-4x rows, with row-level provenance.
8. Do not concatenate WSI with RNA, mutation, or CNV.
9. Pass combined WSI, RNA, mutation, and CNV as separate HEALNet modalities.
10. Demonstrate one real matched BLCA patient and stop before training or KIRP acquisition.

## Paper/code deviations

| Topic | Paper | Released code | Supervisor track |
|---|---|---|---|
| Encoder weights | Kather100K ResNet50 | ImageNet1K V2 | ImageNet1K V2 |
| WSI scales | 2x and 4x | one integer level per run | two branches, then row concatenation |
| Scale concatenation | not explicit | not implemented | required by supervisor |
| Main modality fusion | separate modalities via shared latent | merged Omic + WSI in released loader | WSI + separate RNA/mutation/CNV |

These differences must remain visible in provenance. This track is supervisor-aligned, not an exact reproduction of every paper preprocessing choice.

## Verified one-patient metadata

```text
case:       TCGA-2F-A9KT
WSI/Omic:   exact patient and slide match
RNA:        1,523 finite values
Mutation:   1,125 finite values
CNV:        193 finite values
```

Synthetic and real-input four-modality interface testing produced finite `[1,4]` outputs. The real WSI input was `[1,2048,44445]`, paired with the same patient's RNA, mutation, and CNV tensors. Because HEALNet was randomly initialized and untrained, this is a shape/numerical smoke test, not scientific inference or a survival prediction.

## Current gate

```text
adapter/interface: VERIFIED
runtime:           TESLA_T4_VERIFIED
checkpoint:        IMAGENET1K_V2_VERIFIED
real extraction:   BLCA_ONE_PATIENT_PILOT_SUCCESS
training:          NOT STARTED
```

See [reports/validated_pilot_baseline.md](reports/validated_pilot_baseline.md) for the frozen result and exact hashes.
