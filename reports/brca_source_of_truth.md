# BRCA source of truth and alignment gate

BRCA is the approved next cohort; the frozen BLCA pilot remains the engineering
reference. This approval does **not** authorize WSI acquisition. This audit is
CPU-only: no WSI payload was downloaded, opened, or inspected, and no feature
extraction or training was performed.

## Authoritative source identities

| Input | Exact local path | Bytes | SHA256 | Repository identity |
|---|---|---:|---|---|
| Materialized BRCA Omic archive | `/teamspace/studios/this_studio/Author_Official_Repo_directery/healnet/data/tcga/omic/tcga_brca_all_clean.csv.zip` | 4,081,277 | `4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8` | official checkout commit `90459b65a3d4a4ef9fd405671c457b5a1163cc7d`; Git LFS pointer blob `eed96279b655023feb891970b327d9f2328023f6`; LFS OID `sha256:4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8` |
| Filtered BRCA WSI manifest | `/teamspace/studios/this_studio/healnet/data/tcga/gdc_manifests/filtered/brca_wsi_manifest_filtered.txt` | 157,900 | `ac1b4d591ce255a7d4e6bde3778a041b107576693657ee296fe6eb36c4e79a92` | official `v0.1.0` commit `28ba5da6ab99fd8069972c22e986d83edb658dd4`; Git blob `03a7bc71ff1cf072759b2af05d6e405891f486ec` |

The Omic ZIP contains exactly one CSV member:

| Member | Uncompressed bytes | SHA256 |
|---|---:|---|
| `./tcga_brca_all_clean.csv.zip` | 15,021,018 | `052637f2a69c515812796d9638566cb75299b6a3571dbdc5363496f12665027d` |

All source sizes and hashes are enforced by
`scripts/build_brca_alignment.py`; generation stops on any mismatch.

## Deterministic identity policy

- The only join key is the exact pair `(case_id, slide_id)`.
- Manifest `case_id` is derived from the first three TCGA barcode fields in
  the complete manifest filename; it must equal the Omic `case_id`.
- The unnamed Omic CSV ordinal is retained as `omic_source_row_id` only for
  traceability. It is never a join key and CSV/manifest row order is ignored.
- A case with exactly one WSI row and one exact Omic row is `KEEP`.
- Every exactly matched row belonging to a multi-WSI case is `AMBIGUOUS`; no
  slide is selected implicitly.
- A full outer join preserves `WSI_ONLY` and `OMIC_ONLY` records if present.
- Rows are sorted lexicographically by `case_id`, then complete `slide_id`.

## Verified cohort

| Measure | Count |
|---|---:|
| Omic rows | 1,022 |
| Omic patients | 956 |
| Filtered-manifest WSI rows | 1,022 |
| Filtered-manifest WSI patients | 956 |
| Exact `(case_id, slide_id)` matches | 1,022 |
| `KEEP` singleton patients / rows | 894 / 894 |
| `AMBIGUOUS` multi-WSI patients / rows | 62 / 128 |
| WSI-only patients / rows | 0 / 0 |
| Omic-only patients / rows | 0 / 0 |
| Shared cases with different slide-key sets | 0 |

The 62 ambiguous patients have multiple exact WSI/Omic rows and remain
excluded from the singleton pilot candidate pool. There are 894 unambiguous
singleton candidates. No patient or slide is chosen by source row position.

## Omic modality contract

| Modality | Feature columns |
|---|---:|
| RNA (`*_rnaseq`) | 1,558 |
| Mutation (`*_mut`) | 21 |
| CNV (`*_cnv`) | 1,333 |

The archive has 1,022 rows across 956
patients. CNV is embedded in the Omic table; it is not a separate file.

## Storage and authorization

| WSI set | Manifest-declared bytes | Decimal size |
|---|---:|---:|
| All filtered rows | 1,054,468,732,178 | 1054.469 GB |
| `KEEP` singleton rows | 918,532,189,383 | 918.532 GB |
| `AMBIGUOUS` rows | 135,936,542,795 | 135.937 GB |

`reports/brca_download_plan.tsv` records every manifest row, including its
alignment disposition, but every `download_status` is `NOT_AUTHORIZED`.
Acquisition remains blocked until the exact three-patient pilot is confirmed.
Bulk download is not authorized.

## Generated artifact identities

| Artifact | Bytes | SHA256 |
|---|---:|---|
| `reports/brca_row_level_alignment.csv` | 290,399 | `13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe` |
| `reports/brca_download_plan.tsv` | 331,222 | `ef32418e6551ded13bab3e7d1f5c2760d628b348c59c394ebdd6865e60372a82` |

Final status: `BRCA_ALIGNMENT_READY__WSI_DOWNLOAD_NOT_AUTHORIZED`
