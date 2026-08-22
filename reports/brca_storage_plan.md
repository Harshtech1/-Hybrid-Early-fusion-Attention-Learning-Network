# BRCA storage-aware pilot and cohort plan

Status: `PLANNING_ONLY — NO WSI DOWNLOAD AUTHORIZED`

## Hard limits

- Lightning organization persistent quota: `200,000,000,000` bytes (200 GB).
- User-reported organization usage at planning time: approximately 107.57 MB.
- Mandatory unused safety floor: `20,000,000,000` bytes.
- Provisional future-training hold: `5,000,000,000` bytes. This is a reserve,
  not authorization to train.
- Initial download and processing concurrency: exactly one.

The local filesystem currently reports more physical free space than the
organization quota. That does not increase the permitted 200 GB budget. No
independent additive ephemeral disk has been established.

## Exact raw WSI inventory

The 894 deterministic singleton BRCA WSIs total:

```text
918,532,189,383 bytes
918.532 GB decimal
855.450 GiB
```

The full singleton cohort cannot be retained on Lightning. The proposed
three-patient pilot totals `2,984,417,159` bytes; its largest WSI is
`1,360,743,825` bytes.

## Feature-storage scenarios

The following values scale measured BLCA artifacts and are capacity scenarios,
not predictions of BRCA patch counts:

| Retention policy | BLCA measured bytes/patient | 3-patient reference | 894-patient reference |
|---|---:|---:|---:|
| Complete frozen output directory | 730,926,685 | 2,192,780,055 | 653,448,456,390 |
| Compact combined features + coordinates/provenance/HDF5 | 367,401,271 | 1,102,203,813 | 328,456,736,274 |

Both cohort-scale scenarios exceed the 200 GB Lightning quota. Actual BRCA
patch counts must be measured during the approved pilot before any cohort
retention forecast is accepted.

At the largest proposed WSI size, this conservative pre-pilot reference sum is
`28,666,606,029` bytes: two raw copies, one BLCA-sized complete output,
ImageNet1K_V2 checkpoint, BRCA Omic archive, reported current usage, the 5 GB
training hold, and the 20 GB safety floor. It leaves ample nominal headroom for
one patient, but an adaptive gate must be rerun after real coordinate counts
are known.

## Storage-aware state machine

For each separately approved patient:

1. Recheck the Lightning quota and local ledger.
2. Derive exactly one GDC TSV row from the frozen approved alignment.
3. Download one WSI to an external-to-Git staging directory with resumable
   `.partial` behavior; never submit the 1,022-row official manifest.
4. Verify the exact filename, byte size, and MD5 before opening the WSI.
5. Read only slide metadata first and apply the approved MPP/pyramid gate.
6. Generate and validate coordinates on CPU; recompute the projected feature
   peak from the actual 2× and 4× counts.
7. Switch to a GPU machine only after these gates pass.
8. Extract at concurrency one without saving patch images.
9. Validate, hash, and atomically publish retained artifacts.
10. Upload results only to a separately configured write-limited Drive remote,
    verify the remote artifact hashes, then delete only the exact local raw WSI
    if the approved retention policy permits deletion.
11. Mark the patient complete before proceeding to the next UUID.

On any failure, stop. Do not download the next patient, auto-delete an unknown
path, select a multi-WSI patient, or reduce the safety floor.

## Google Drive boundary

The existing `MY_GDRIVE` remote is read-only and therefore cannot receive
outputs. It has previously reported approximately 4.837 TiB free, but that
does not authorize upload. Before execution, configure a separate
`drive.file`-scoped results remote under user-controlled OAuth. Do not expose
tokens or client secrets. Copy one verified artifact transaction at a time;
do not run OpenSlide directly against a network mount.

No WSI was downloaded, opened, or processed while preparing this plan.
