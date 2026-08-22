# KIRP Cohort Readiness Audit

Audit time: 2026-08-17 (UTC)

The cohort inputs and official repository were inspected read-only; only this
pilot-repository report was created. No KIRP WSI or Omic payload was downloaded,
and no extraction, inference, training, or Git-history mutation was performed.

## Authoritative inputs

The audit used the clean official HEALNet `v0.1.0` checkout at
`/teamspace/studios/this_studio/healnet`, peeled commit
`28ba5da6ab99fd8069972c22e986d83edb658dd4`.

| Input | Local path | Local file bytes | SHA256 | Contents represented |
|---|---|---:|---|---|
| Full GDC WSI manifest | `/teamspace/studios/this_studio/healnet/data/tcga/gdc_manifests/full/kirp_wsi_manifest_full.txt` | 119,078 | `d84c687742a4f90c3a8cb5c063b62b799c5c6724da95e616c84a07913f8856c6` | 773 WSI records across 291 patient barcodes; 411,956,997,636 declared payload bytes |
| Filtered GDC WSI manifest | `/teamspace/studios/this_studio/healnet/data/tcga/gdc_manifests/filtered/kirp_wsi_manifest_filtered.txt` | 46,363 | `dab0d2cf5393628f6be67d9b0438fa9d7148b5a3dec0f22fdeaf3a051f394ff6` | 300 WSI records; 310,068,314,540 declared payload bytes |
| KIRP Omic CSV pointer | `/teamspace/studios/this_studio/healnet/data/tcga/omic_xena/kirp_master.csv` | 132 | `c2fd43d8a56fbf447b6a3fe9381ea440495885546aa2bb45b930f5b94a25b275` | Git LFS pointer, not the CSV payload |

The Omic pointer declares object OID
`sha256:520f911b110b28635047b6267662080e6350a9888c697919ab906721bb4d6751`
and payload size 6,361,784 bytes. That LFS object is not materialized in the
checkout, so patient-level Omic rows cannot be independently enumerated from
this machine yet.

## Cohort alignment

Patient IDs in the filtered WSI manifest were computed from the first three
TCGA barcode fields in each filename (`TCGA-XX-XXXX`). The manifest contains
276 distinct WSI patient cases: 265 cases with one filtered WSI record and 11
cases with multiple filtered WSI records. The 11 multi-WSI cases account for
35 records, with case multiplicities `{2: 8, 3: 1, 7: 1, 9: 1}`.

| Category | Patients | Provenance |
|---|---:|---|
| Omic patients | 284 | Supplied HEALNet paper/reference cohort cardinality; not re-enumerable locally while the CSV remains an LFS pointer |
| Filtered-manifest WSI cases | 276 | Computed directly from the 300 filtered manifest rows |
| Matched WSI/Omic patients | 260 | Inferred from the prior validated split: 249 single-candidate plus 11 multi-WSI patients |
| Unambiguous matched patients | 249 | Existing `PROJECT_STATE.md` audit summary: “249 single-candidate cases” |
| Ambiguous matched patients | 11 | Existing `PROJECT_STATE.md` audit summary: “11 multi-WSI cases deferred” |
| WSI-only discards | 16 | Inferred as `276 - 260` |
| Omic-only discards | 24 | Inferred as `284 - 260` |

The aggregate counts are internally consistent, but they are not a substitute
for a row-level alignment artifact. In particular, `PROJECT_STATE.md` preserves
only the 249/11 summary, not the exact 249 patient IDs, Omic row identifiers,
and corresponding GDC file UUIDs needed for deterministic acquisition.

## Storage facts

- The full manifest declares 411,956,997,636 WSI bytes (411.957 GB;
  383.665 GiB).
- The filtered manifest declares 310,068,314,540 WSI bytes (310.068 GB;
  288.774 GiB).
- Its 265 singleton-case WSI files total 270,111,741,046 bytes
  (270.112 GB; 251.561 GiB).
- The exact byte total for the selected 249 unambiguous matched files cannot be
  computed until the row-level alignment/selection manifest is restored.
- A mathematically safe bound, not an estimate, can be computed from the 265
  singleton file sizes: any 249-file subset must occupy at least
  226,279,440,372 bytes (226.279 GB; 210.739 GiB) and at most
  268,680,116,297 bytes (268.680 GB; 250.228 GiB). These bounds do not identify
  the cohort and must not be used as a replacement selection rule.
- The HEALNet reference paper reports 275 GB for KIRP disk space. This is a
  published descriptive figure, not an exact acquisition total for the local
  249-patient subset.
- The Omic LFS payload declares 6,361,784 bytes in addition to WSI storage.
- At 2026-08-17T18:46:32Z, `df -B1` reported 338,823,127,040 bytes available
  (338.823 GB; 315.554 GiB) on `/teamspace/studios/this_studio`. This is only a
  point-in-time capacity measurement and does not reserve space for partial
  transfers, extracted features, provenance, or other working artifacts.

## Readiness decision

The cohort definition is supported at aggregate-count level, but acquisition
is blocked because the exact selection cannot yet be reproduced safely.

Before any KIRP download:

1. Restore and verify the real KIRP Omic Git LFS object with OID
   `520f911b110b28635047b6267662080e6350a9888c697919ab906721bb4d6751`.
2. Materialize a deterministic row-level alignment manifest containing the 249
   accepted patient IDs and their exact GDC file UUIDs, plus explicit records
   for the 11 ambiguous, 16 WSI-only, and 24 Omic-only patients.
3. Recompute the exact selected-file byte total and review the download plan
   against available storage before acquisition.

**Blocker: KIRP Omic payload and deterministic row-level 249-patient selection
manifest are not present locally.**

KIRP downloaded: **NO**

Final status: `KIRP_OMIC_LFS_AND_ROW_LEVEL_SELECTION_MANIFEST_REQUIRED`
