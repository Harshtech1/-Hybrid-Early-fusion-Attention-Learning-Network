# BRCA Q50-only acquisition transition

Status: `BRCA_Q50_LOCAL_GDC_ACQUISITION_READY`

The user directly authorized the Q50-only BRCA pilot and explicitly prohibited
training or Q75 processing. Google Drive has been removed from the critical
path: the exact Q50 WSI will be acquired from its authoritative GDC UUID into
local persistent storage. Drive may be configured later as optional backup,
but it is not required and no Drive operation is authorized by this record.

Q25 remains frozen as a successful predecessor and will not be rerun.

## Exact authorized object

| Field | Value |
|---|---|
| Patient | `TCGA-AR-A1AW` |
| GDC UUID | `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62` |
| Filename | `TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs` |
| Declared bytes | `975,626,387` |
| Expected MD5 | `304509e03f26cbecc9aee4ea691c8e5a` |
| GDC state | `released` |
| Omic row | `370` |
| RNA | `[1,1,1558]` |
| Mutation | `[1,1,21]` |
| CNV | `[1,1,1333]` |

The patient has exactly one released filtered-manifest WSI and exactly one
clean-archive Omic row. The full slide filename matches exactly; no row-order
or ambiguous patient-only pairing is used.

## Executable acquisition boundary

The only executable data manifest is:

`multiscale_feature_pilot/provenance/brca_phase2_q50_authorized/Q50_TCGA-AR-A1AW_5c1216f3-19ec-4d3c-9bb0-9bd740b79f62.AUTHORIZED_Q50_ONLY.gdc.tsv`

It contains one standard GDC data row and has SHA256
`1028665e49bab6895a47947d069e225c2d6b4b90f420f7a9dc916ca9313a6062`.
The historical three-candidate `NOT_AUTHORIZED` files remain unchanged and
must never be passed to the GDC client.

The configured Linux GDC client is version `2.3`, SHA256
`1df9281cbacbb53413607a4b9b682947dcb48b6ac7fade6527748bc025ae8c96`.
Patient concurrency remains one. The client may retain its internal default
transfer threads; this is not multi-patient processing.

After download, a regular non-symlink SVS must match both `975,626,387` bytes
and MD5 `304509e03f26cbecc9aee4ea691c8e5a` before OpenSlide construction. A
SHA256 will then be calculated. Only `mpp_x`, `mpp_y`, `level_dimensions`,
and `level_downsamples` may be collected from the header. No region or pixel
read is authorized.

Q25's level indices and scale mapping are slide-specific and cannot be copied
to Q50. The required stop is a Q50-specific file-identity and header-metadata
report, after which a distinct scale-policy review is required.

## Safety and verification

- Q75 download, opening, or processing: `NOT_AUTHORIZED`
- Pixel reads and coordinate generation: `NOT_AUTHORIZED`
- ResNet50 or HEALNet execution: `NOT_AUTHORIZED`
- Training, backward passes, and optimizer steps: `NOT_AUTHORIZED`
- Raw WSI deletion: `NOT_AUTHORIZED`
- Google Drive operations: `NOT_AUTHORIZED` and not required
- Existing Q50 local staging/coordinates/features at gate time: absent
- Existing BRCA pilot data: `801,470,077` bytes
- Observed physical filesystem headroom: `350,888,206,336` bytes
- Full test suite: `318/318` passed
- Focused Q50 authorization tests: `12/12` passed

The authorization was built without network access, WSI download, WSI
opening, pixel access, coordinate generation, feature extraction, or training.
