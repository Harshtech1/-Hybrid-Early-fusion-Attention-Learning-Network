# BRCA Q75 exact-file and OpenSlide header-only result

Status: `BRCA_Q75_FILE_AND_HEADER_METADATA_VERIFIED`

CPU-only metadata result. No pixel access or scale-policy approval occurred.

## Identity

- Patient: `TCGA-E2-A154`
- UUID: `25aec062-60d1-446e-a1c6-0c79cc74a770`
- Slide: `TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`
- Local path: `/teamspace/studios/this_studio/brca_pilot_data/Q75.incoming/25aec062-60d1-446e-a1c6-0c79cc74a770/TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`
- Size: `1360743825` bytes
- MD5: `a8c4b68fb6e0ab3e862efe3ed1fe10d7`
- SHA256: `844eb17d1bf66136b61f0c0e69ae1609e7dc9eca096e8c206e60ddd34332ab37`

## OpenSlide header

- mpp-x: `0.2468`
- mpp-y: `0.2468`
- Level count: `4`

| Level | Dimensions | Downsample | Native MPP x | Native MPP y |
|---:|---:|---:|---:|---:|
| 0 | 108528 × 90471 | 1.0 | 0.2468 | 0.2468 |
| 1 | 27132 × 22617 | 4.000066321793341 | 0.9872163682185965 | 0.9872163682185965 |
| 2 | 6783 × 5654 | 16.000619030774672 | 3.948952776795189 | 3.948952776795189 |
| 3 | 3391 × 2827 | 32.00359724763015 | 7.89848780071512 | 7.89848780071512 |

## Descriptive target comparison

- `0.5 µm/px`: closest level 0; appears to require controlled resampling.
- `1.0 µm/px`: closest level 1; appears natively achievable.

This comparison does not select or approve a Q75 scale policy.

## Download and storage binding

- Download result SHA256: `510d51d6cb81b4c603b2fd653aab615cef230743135145719972f4b8616eba00`
- Used bytes before download: `25645146112`
- Used bytes after download: `27006050304`
- Used bytes before header gate: `27006148608`
- Used bytes after header gate: `27006148608`
- Exact GDC-created tree:
  - `25aec062-60d1-446e-a1c6-0c79cc74a770`
  - `25aec062-60d1-446e-a1c6-0c79cc74a770/TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`
  - `25aec062-60d1-446e-a1c6-0c79cc74a770/logs`
  - `25aec062-60d1-446e-a1c6-0c79cc74a770/logs/TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs.parcel`

## Source binding

- Source commit: `ef6c8921d993567c8178fc932f7315378714ca5a`
- `multiscale_feature_pilot/config/brca_q75_acquisition_authorization.yaml`: `335e6d36aac1c21cc1cd52f8a14e5d2ecfde1f3a6f398d796bf842baaca35979`
- `multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/MANIFEST_SET.AUTHORIZED_Q75.yaml`: `2330c4bc66c73c8e150be2d028aefb2a84916b18e4b5076d95fc28cf869d7050`
- `multiscale_feature_pilot/provenance/brca_phase2_q75_authorized/Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770.AUTHORIZED_Q75_ONLY.gdc.tsv`: `8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a`
- `multiscale_feature_pilot/src/brca_omic.py`: `5061039913cb0dd4e8e30958c8b4b76f088396f7841255dd94234c3b462d5a5d`
- `multiscale_feature_pilot/src/brca_q75_authorized_manifest.py`: `6caedfc164617fcf99bb17b99da975ebd8a89d5daec4827826cad8f016ff0afa`
- `multiscale_feature_pilot/src/brca_q75_download_runner.py`: `53d178dd5a230c06be7dd80e96b92567b8622d4e0849cb3aa12a8f72bffdf070`
- `multiscale_feature_pilot/src/brca_q75_header_gate.py`: `7c07a215edf077d041f9279eb854a4459c5adeb28fafc0fef92fdfa47ba4f996`
- `multiscale_feature_pilot/src/wsi_metadata_policy.py`: `b443ec59b96fc5bf117891389e8c5521a2dda028215b270811b3fca4a80fce48`
- `scripts/run_brca_q75_download.py`: `dfb28a7984ee08f2c7a571105cebe1f302cf546fe9895a85fa1865d5a12c0b79`
- `scripts/run_brca_q75_header_gate.py`: `93ed8d3727e2682503397cdf1328b06b5c6b2fb1e79c6126a3474297d0e643a9`

## Validation and stop

- source_commit_and_critical_files_head_equal: `PASS`
- only_protected_blca_worktree_edit: `PASS`
- secure_authorization_record_manifest_reads: `PASS`
- download_result_binding: `PASS`
- same_held_wsi_descriptor_before_and_after_header: `PASS`
- exact_uuid_filename_size_md5_sha256: `PASS`
- partial_or_incomplete_download_check: `PASS`
- regular_non_symlink_svs_and_parcel: `PASS`
- held_omic_archive_and_exact_row_rematch: `PASS`
- finite_positive_consistent_openslide_header: `PASS`
- zero_pixel_or_region_reads: `PASS`
- Atomic result directory: `/teamspace/studios/this_studio/healnet_pilot/multiscale_feature_pilot/provenance/brca_q75_header_metadata_result`
- Required stop reached: `True`

Further Q75 work requires separate user review and authorization.
