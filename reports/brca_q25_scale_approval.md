# BRCA Q25 scale-policy approval transition

Recorded at: 2026-08-18T21:16:39Z

## Outcome

`BRCA_Q25_SCALE_POLICY_APPROVED_CPU_RECORDED`

The user relayed the supervisor response as **“he approved everything”** in
response to the exact Q25 scale decision. That decision is now recorded with a
narrow, fail-closed interpretation:

- the approximately 0.5-MPP branch uses native level 0 at `0.2525 MPP`, with
  an explicit spatial downsample factor of `2.0`, producing `0.505 MPP`;
- the approximately 1.0-MPP branch uses native level 1 at
  `1.0100149842739303 MPP` without resampling;
- neither a different native level nor any other resampling may be substituted
  silently.

This resolves the physical-scale mapping that blocked the Q25 metadata gate.
It does not itself execute or authorize pixel reads, coordinate generation,
feature extraction, HEALNet execution, or training.

## Exact Q25 identity

- patient: `TCGA-LL-A6FP`
- slide ID: `TCGA-LL-A6FP-01Z-00-DX1`
- GDC UUID: `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6`
- filename: `TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs`
- verified size: `648046947` bytes
- verified MD5: `75536393096ffd928bc35ec9503c3655`
- verified SHA256: `ac852be806eb0d91214145983319b604919a8e1d16bd59378a3dba69a600979c`

## Approved scale calculation

| Branch | Target MPP | Source | Operation | Effective MPP | Per-axis target error | Result |
| --- | ---: | --- | --- | ---: | ---: | --- |
| scale_2x | 0.5 | native level 0, 0.2525 MPP | explicit 2x linear spatial downsample | 0.505 | 1.0% | pass |
| scale_4x | 1.0 | native level 1 | native, no resampling | 1.0100149842739303 | approximately 1.0015% | pass |

For the first branch:

```text
0.2525 MPP * 2.0 = 0.505 MPP
abs(0.505 - 0.5) / 0.5 = 0.01
```

For the second branch:

```text
abs(1.0100149842739303 - 1.0) / 1.0
  = 0.010014984273930328
```

Both effective scales are within the already approved 10% per-axis target
tolerance. `0.505 MPP` must be reported as the effective engineering scale; it
must not be mislabeled as exact `0.5 MPP`.

## Fail-closed implementation

The transition adds a configuration record and a pure evaluator that accepts
only already-collected MPP, dimensions, and downsample values. The evaluator:

- has no WSI path or slide-object input;
- imports neither OpenSlide nor an image library;
- rejects any drift from the exact recorded Q25 pyramid;
- pins source levels 0 and 1 rather than searching for substitutes;
- records that pixel execution, coordinate generation, and feature extraction
  remain unauthorized by this transition.

Focused verification result: `8 passed`. Full pilot test-suite result:
`230 passed`.

## Remaining execution-policy boundary

The exact relayed decision did not name an interpolation kernel, patch
footprint/output geometry, coordinate lattice, tissue-mask method, or boundary
rule. Those details are deliberately not invented here and remain pending a
separate, reviewable Q25 coordinate-and-extraction execution policy.

Q50 and Q75 remain locked. The historical native-only metadata policy and its
failed Q25 report are preserved unchanged as audit evidence; this transition
supersedes only their native-only scale decision for this exact WSI.

## Recorded file identities

- policy config SHA256: `fd54080543706d56cf6fe336b61630f3f8c09a6741e4fcf5ea7c42801d0ff816`
- pure evaluator SHA256: `6ede1db26638363f1ecca2fbd8db3f2ae222eacff291ba6de44634f26d148a89`
- focused tests SHA256: `7711991c9a4ec9f46e50a8f5455d3341ad3b7f8cb4537daace9831fce51caa8a`
- prior metadata YAML SHA256: `6c342b5be92844e07aa9b57551537432d7bb261849ed3cc3d661582763e6bb26`
- prior metadata report SHA256: `795cd14f0d6507b2bcc48f51a8a5bfc6f2bda02f2083597af253a575e5d1af97`

## Stop confirmation

During this CPU-only transition, the WSI was not opened, no pixels were read,
no coordinates were generated, no image was resampled, and no ResNet50,
HEALNet, training, download, Drive, Git staging, commit, or push operation was
performed. The official HEALNet checkout and frozen BLCA baseline were not
modified.
