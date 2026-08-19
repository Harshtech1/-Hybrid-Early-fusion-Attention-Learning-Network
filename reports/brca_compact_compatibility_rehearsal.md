# BRCA frozen-pilot compact compatibility rehearsal

Status: `CPU_READ_ONLY_COMPATIBILITY_PASS__NO_MIGRATION_PERFORMED`

## Authorized scope

The frozen Q25, Q50, and Q75 feature directories were read and validated on
CPU against the information required by `BRCA_COMPACT_FEATURE_ARTIFACT_SET_V1`.
No file in any pilot directory was created, modified, migrated, overwritten,
or deleted. No WSI was opened and no GPU, extraction, Drive, or training
operation occurred.

## Compatibility result

| Check | Q25 | Q50 | Q75 |
|---|---|---|---|
| Original strict validator | PASS | PASS | PASS |
| Exact six regular non-symlink files | PASS | PASS | PASS |
| Combined equals `torch.cat((2x,4x), dim=0)` | PASS | PASS | PASS |
| CPU contiguous finite float32 | PASS | PASS | PASS |
| Complete contiguous row provenance | PASS | PASS | PASS |
| 2x prefix followed by 4x suffix | PASS | PASS | PASS |
| Lossless compact representability | **PASS** | **PASS** | **PASS** |

## Exact tensors and semantic hashes

| Pilot | 2x shape | 4x shape | Combined shape | Combined semantic SHA256 |
|---|---:|---:|---:|---|
| Q25 | `[7404,2048]` | `[1918,2048]` | `[9322,2048]` | `300196a20822060236f81a9158a1c54ea8193d6b8408ad4271f306b997ccf327` |
| Q50 | `[8580,2048]` | `[2213,2048]` | `[10793,2048]` | `5c3f37b406451eb10de4bf1b294c8fd018bb21e2685752182e8220dd0beee511` |
| Q75 | `[13487,2048]` | `[3458,2048]` | `[16945,2048]` | `6e01631d8a0a3ecb0dceefb29cd32866d49199a796ec78aaf00f2124c54a4637` |

The semantic digest is calculated from contiguous little-endian float32 raw
bytes. Since the existing combined tensors are exactly equal to the ordered
branch concatenations, removing permanent branch tensor copies would not
remove a feature value. Branch recovery remains exact through manifest row
ranges and row provenance.

## Exact read-only storage calculation

| Pilot | Current six-file bytes | Duplicate branch tensor bytes removable | Canonical tensor + provenance bytes |
|---|---:|---:|---:|
| Q25 | 153,208,853 | 76,368,978 | 76,832,419 |
| Q50 | 177,402,321 | 88,419,410 | 88,974,882 |
| Q75 | 278,525,034 | 138,816,594 | 139,699,484 |
| **Total** | **609,136,208** | **303,604,982** | **305,506,785** |

The exact duplicate-tensor saving across the three pilots would be 303,604,982
bytes, or 49.84% of their current complete directories. The compact manifest
and sidecar would add a small amount to the canonical tensor-plus-provenance
column; their exact bytes remain unknown until an authorized migration writes
them. No such write was performed.

At the observed mean canonical-tensor-plus-provenance rate, naive scaling to
894 patients is 91,041,021,930 bytes before compact manifests. This remains an
estimate, not a storage guarantee.

## Conclusion

All three frozen pilots are losslessly representable under the compact schema.
The compatibility gate passes. Actual migration of pilot or future cohort
artifacts remains separately unauthorized.
