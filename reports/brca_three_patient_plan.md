# Proposed BRCA three-patient pilot

Status: `PROPOSED BRCA PILOT — NOT AUTHORIZED`

The supervisor has selected BLCA and BRCA as the project cohorts. This report
proposes the first BRCA engineering pilot; it does not authorize WSI download,
pixel access, extraction, or training.

## Source and population

The proposal is derived read-only from official HEALNet v0.1.0 at commit
`28ba5da6ab99fd8069972c22e986d83edb658dd4`:

- clean Omic archive `data/tcga/omic/tcga_brca_all_clean.csv.zip`, SHA256
  `4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`;
- filtered GDC manifest
  `data/tcga/gdc_manifests/filtered/brca_wsi_manifest_filtered.txt`, SHA256
  `ac1b4d591ce255a7d4e6bde3778a041b107576693657ee296fe6eb36c4e79a92`.

The exact case-plus-slide join yields 894 singleton patients and 62 patients
with multiple WSIs. All 62 multi-WSI patients remain deferred. There are no
WSI-only or Omic-only patients in the released BRCA clean cohort.

The generated row-level alignment SHA256 is
`13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe`;
the all-`NOT_AUTHORIZED` download-plan SHA256 is
`ef32418e6551ded13bab3e7d1f5c2760d628b348c59c394ebdd6865e60372a82`.

## Deterministic selection

Sort the 894 singleton rows by
`(declared WSI byte size, patient_id, GDC UUID)`. For each inclusive quantile
`q` in `0.25, 0.50, 0.75`, select zero-based index
`floor((n - 1) * q + 0.5)`. This is the nearest observed rank with half ties
rounded upward. For `n=894`, the one-based ranks are 224, 448, and 671.

| Label | Rank | Patient | GDC UUID | Declared bytes |
|---|---:|---|---|---:|
| Q25 | 224 | `TCGA-LL-A6FP` | `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6` | 648,046,947 |
| Q50 | 448 | `TCGA-AR-A1AW` | `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62` | 975,626,387 |
| Q75 | 671 | `TCGA-E2-A154` | `25aec062-60d1-446e-a1c6-0c79cc74a770` | 1,360,743,825 |

Total proposed raw size: `2,984,417,159` bytes (`2.984 GB`, `2.779 GiB`).

### Q25 — TCGA-LL-A6FP

- slide: `TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs`
- UUID: `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6`
- expected MD5: `75536393096ffd928bc35ec9503c3655`
- expected size: `648,046,947` bytes
- Omic source index: `956`; physical CSV line: `958`

### Q50 — TCGA-AR-A1AW

- slide: `TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs`
- UUID: `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62`
- expected MD5: `304509e03f26cbecc9aee4ea691c8e5a`
- expected size: `975,626,387` bytes
- Omic source index: `370`; physical CSV line: `372`

### Q75 — TCGA-E2-A154

- slide: `TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`
- UUID: `25aec062-60d1-446e-a1c6-0c79cc74a770`
- expected MD5: `a8c4b68fb6e0ab3e862efe3ed1fe10d7`
- expected size: `1,360,743,825` bytes
- Omic source index: `771`; physical CSV line: `773`

For every proposed patient, the clean Omic `case_id` matches the TCGA
participant encoded in the slide filename, and the complete Omic `slide_id`
equals the filtered-manifest filename. The metadata does not expose MPP or
pyramid structure; those properties must be inspected only after a separately
authorized download and before any coordinate generation.

## Required approval before execution

The supervisor must approve this exact UUID list and the BRCA physical-scale
policy. After approval, the controller will create one one-row GDC manifest at
a time, download at concurrency one, verify size and MD5, and stop for each
slide whose MPP/pyramid does not satisfy the approved policy.

No BRCA WSI was downloaded or opened while preparing this proposal.
