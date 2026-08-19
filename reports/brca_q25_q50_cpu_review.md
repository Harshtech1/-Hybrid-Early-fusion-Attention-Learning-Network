# BRCA Q25/Q50 CPU review and Q75 readiness

Status: `BRCA_Q25_Q50_CPU_REVIEW_COMPLETE`

Recommendation: `RUN_Q75_PILOT_RECOMMENDED`

This is a CPU-only, read-only review of the committed Q25/Q50 reports and
provenance plus locally available Q75 manifest/alignment metadata. It did not
open or download the Q75 WSI, read any WSI pixels, run a GPU, rerun either
completed pilot, execute HEALNet, process the full cohort, use Google Drive,
delete a raw WSI, or train a model.

## Frozen source records

The comparison uses the following committed result records as its primary
sources:

| Record | SHA256 |
|---|---|
| `reports/brca_q25_gpu_pilot_result.md` | `4adfe5673b2e69267738756173f6266d0c7895412950715667893f5e94fc8a46` |
| `multiscale_feature_pilot/provenance/brca_q25_gpu_pilot_result.yaml` | `76df862c438f4d94e34342b54c80c0866a440af0574be9003444075d7073c6e4` |
| `reports/brca_q50_gpu_pilot_result.md` | `ff21444147734234b3d4d6fbf9a6fd7fce1ee81ce58d7508a1d5dc848efe5ae3` |
| `multiscale_feature_pilot/provenance/brca_q50_gpu_pilot_result.yaml` | `904ad3c7bef859f89fef6151014f9eb4b37b5af5d6668972feb19b0e656d263b` |
| `multiscale_feature_pilot/provenance/brca_q25_coordinate_gate.yaml` | `7d757fe6973123b3f8b90f3276245fde392e513f784d6c50764f42493218828c` |
| `multiscale_feature_pilot/provenance/brca_q50_coordinate_gate.yaml` | `98f142a291f674a0c35b6af823c6f252ced4b9ed36e04fd85458bf1de3153f54` |
| `multiscale_feature_pilot/provenance/brca_q25_scale_approval.yaml` | `6be21ecf25cd2eb9589109feef9657537f99c8968500c0fdf68adc5ac86435e1` |
| `multiscale_feature_pilot/provenance/brca_q50_policy_preexecution.yaml` | `e387430c028a8c6f791477b12a17027c92da3e0f2b498eb43d0312716baf0bce` |

The recorded WSI and external-artifact hashes were not recomputed in this
review, because doing so would no longer be a comparison using only the frozen
reports/provenance. Both guarded executions had already independently checked
their bound inputs and published artifacts.

## Side-by-side verified results

| Metric | Q25 | Q50 | Observation |
|---|---:|---:|---|
| Patient | `TCGA-LL-A6FP` | `TCGA-AR-A1AW` | Exact singleton patients |
| Raw WSI size | 648,046,947 bytes | 975,626,387 bytes | Q50 is 50.55% larger |
| Level-0 dimensions | `65,736 × 67,406` | `99,960 × 65,334` | Different aspect and pyramid rounding |
| Level-0 pixels | 4,431,000,816 | 6,530,786,640 | Q50 has 47.39% more pixels |
| Native MPP (x/y) | `0.2525 / 0.2525` | `0.2468 / 0.2468` | Q50 is 2.26% finer |
| Chosen 2x effective MPP | `0.505 / 0.505` | `0.4936 / 0.4936` | Both use level 0 plus 2:1 resampling |
| Chosen 4x effective MPP | `1.0100149843 / 1.0100149843` | `0.9872151105 / 0.9872151105` | Both use native level 1 |
| 2x accepted patches | 7,404 | 8,580 | Q50 has 15.88% more |
| 4x accepted patches | 1,918 | 2,213 | Q50 has 15.38% more |
| Total feature rows | 9,322 | 10,793 | Q50 has 15.78% more |
| Segmentation + coordinate generation | 5.9154 s | 9.5045 s | Like-for-like recorded substage |
| Complete coordinate gate | 8.6470 s | 17.1419 s | Includes identity, mask read, publication, and validation |
| Recorded branch extraction wall-time sum | 134.6872 s | 146.8887 s | Streaming branch times; includes model-forward work |
| Top-level sequential ResNet extraction | Not retained | 148.3946 s | Provenance completeness difference; do not equate with the branch sum |
| Total GPU pilot runtime | 155.7917 s | 171.4287 s | Q50 took 10.04% longer |
| Peak ResNet allocation | 481,334,784 bytes | 481,334,784 bytes | Constant at batch size 32 |
| Peak HEALNet allocation | 266,871,808 bytes | 301,068,288 bytes | Q50 is 12.81% higher |
| Retained feature artifacts | 153,208,853 bytes | 177,402,321 bytes | Includes branch bags plus duplicated combined bag |

The comparable branch extraction details were:

| Branch timing | Q25 | Q50 |
|---|---:|---:|
| 2x streaming wall time | 114.4298 s | 124.7218 s |
| 2x forward sub-time | 20.2641 s | 24.0248 s |
| 4x streaming wall time | 20.2573 s | 22.1669 s |
| 4x forward sub-time | 5.1543 s | 6.3304 s |

The model-forward figures are contained within the branch streaming wall
times and must not be added to them. Q25 did not retain the Q50 runner's newer
top-level sequential-extraction timing field; its exact branch times remain
available and are the fair cross-pilot comparison.

## Pyramid geometry and resampling

| Pyramid level | Q25 dimensions / scalar-derived MPP | Q50 dimensions / scalar-derived MPP |
|---:|---|---|
| 0 | `65,736 × 67,406` / `0.2525` | `99,960 × 65,334` / `0.2468` |
| 1 | `16,434 × 16,851` / `1.0100149843` | `24,990 × 16,333` / `0.9872151105` |
| 2 (mask) | `4,108 × 4,212` / approximately `4.04067` | `6,247 × 4,083` / approximately `3.94914` |
| 3 | `2,054 × 2,106` / approximately `8.08133` | `3,123 × 2,041` / approximately `7.89988` |

Both slides have an approximately 4x native level-1 downsample and neither has
a native approximately 0.5-MPP level. Their reviewed final mappings are
therefore the same in method but slide-specific in physical resolution:

- 2x branch: read a `512 × 512` level-0 footprint and LANCZOS-resample it to
  `256 × 256`; effective MPP is `0.505` for Q25 and `0.4936` for Q50.
- 4x branch: read a native `256 × 256` level-1 region without resampling;
  effective MPP is `1.0100149843` for Q25 and `0.9872151105` for Q50.
- Both use level-0 lattice steps of 512 pixels for 2x and 1,024 pixels for 4x,
  a level-2 tissue mask, complete-boundary patches, and 2x rows before 4x rows.

Rounded pyramid dimensions create small, slide-specific x/y geometry
differences even where the OpenSlide scalar downsample is nearly 4, 16, or 32.
The tissue geometry also differs: Q25 recorded 3 contours and 4 retained
holes, whereas Q50 recorded 4 contours and 2 retained holes. These differences
help explain why raw file size and pixel count did not translate linearly into
accepted-patch count. Q25 initially exposed the missing native 0.5-MPP-level
case and proceeded only after explicit resampling review; Q50 independently
confirmed the same mapping against its own header before execution.

## HEALNet interface results

Both pilots used the same classifier-removed
`ResNet50_Weights.IMAGENET1K_V2` checkpoint and preserved finite float32,
unpooled feature rows:

| Tensor | Q25 | Q50 |
|---|---|---|
| 2x WSI features | `[7404,2048]` | `[8580,2048]` |
| 4x WSI features | `[1918,2048]` | `[2213,2048]` |
| Concatenated WSI | `[9322,2048]` | `[10793,2048]` |
| Natural batched WSI | `[1,9322,2048]` | `[1,10793,2048]` |
| RNA | `[1,1,1558]` | `[1,1,1558]` |
| Mutation | `[1,1,21]` | `[1,1,21]` |
| CNV | `[1,1,1333]` | `[1,1,1333]` |

For both Q25 and Q50, the full-size synthetic pre-extraction check and the
real-feature check returned a finite float32 output `[1,4]`. WSI attention was
`[1,2,9322]` and `[1,2,10793]`, respectively; each Omic attention tensor was
`[1,2,1]`. RNA, mutation, CNV, and WSI remained four separate modalities.
There was no WSI pooling or transpose. HEALNet was randomly initialized in
evaluation/inference mode, so these are numerical/interface smoke tests—not
trained predictions or scientific validation.

## Q75 local metadata audit

### Verified facts

- Deterministic selection: Q75, rank 671 of the 894 exact singleton matches.
- Patient: `TCGA-E2-A154`.
- Exact slide/filename:
  `TCGA-E2-A154-01Z-00-DX1.01FC9B1A-8ECD-4467-9EDD-0B02E4AEEF72.svs`.
- GDC UUID: `25aec062-60d1-446e-a1c6-0c79cc74a770`; state: `released`.
- Manifest-declared size: `1,360,743,825` bytes (`1.361 GB`, `1.267 GiB`).
- Manifest-declared expected MD5: `a8c4b68fb6e0ab3e862efe3ed1fe10d7`.
- Exact case and full-slide Omic match: yes; `patient_wsi_count=1` and
  `patient_omic_count=1`.
- Omic source index `771` (physical CSV line `773`). Locally loaded tensors
  were finite float32: RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV
  `[1,1,1333]`.
- The official filtered manifest SHA256 is
  `ac1b4d591ce255a7d4e6bde3778a041b107576693657ee296fe6eb36c4e79a92`;
  the row-level alignment SHA256 is
  `13b1e8e58b28d4669d8015f759e7d6df3f3296a16f77920b6a83a099999c19fe`;
  and the materialized clean Omic archive SHA256 is
  `4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`.
- The Q75 WSI, staging directory, coordinates, features, and authorized Q75
  manifest are absent. No local WSI SHA256 exists. The expected size and MD5
  above are source-manifest metadata, not locally verified file digests.
- The existing pilot-data tree occupies `1,954,756,342` apparent bytes.

The prepared one-row GDC manifest is the existing guarded file:

`multiscale_feature_pilot/provenance/brca_phase2_manifests/Q75_TCGA-E2-A154_25aec062-60d1-446e-a1c6-0c79cc74a770.NOT_AUTHORIZED.gdc.tsv`

It contains exactly one data row and has SHA256
`8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a`.
Its `NOT_AUTHORIZED` name and policy remain operative. No executable or
`AUTHORIZED` copy was created.

### Planning estimates—not verified Q75 properties

The following ranges extrapolate only from Q25/Q50 ratios. Compression,
tissue fraction, scanner/pyramid construction, segmentation geometry, and I/O
behavior can move Q75 outside them.

| Quantity | Planning range | Basis and limitation |
|---|---:|---|
| Level-0 pixels | 9.11–9.30 Gpx | Q25/Q50 pixel-to-file-byte ratios; actual dimensions unknown |
| Total accepted patches | 15,053–19,574 | Q25/Q50 accepted-patch-to-file-byte ratios; tissue yield dominates |
| 2x patches | 11,962–15,554 | Observed approximately 79.5% share of total |
| 4x patches | 3,092–4,020 | Observed approximately 20.5% share of total |
| Retained feature artifacts | 247–322 MB | Observed bytes per row, including duplicated combined bag |
| Coordinate artifacts | 0.28–0.37 MB | Observed bytes per coordinate row |
| Total T4 GPU pilot runtime | 239–327 s | Observed total-runtime-per-row; plan operationally for 4–7 min |
| Peak ResNet allocation | approximately 481 MB | Expected if batch size remains 32; not a guarantee |
| Peak HEALNet allocation | approximately 400–505 MB | Linear token-count extrapolation from two points only |

Adding the declared raw Q75 WSI and estimated coordinate/feature artifacts to
the current pilot tree gives approximately `3.563–3.638 GB` retained. A
conservative transfer/publication scenario with one additional full raw-file
temporary copy gives approximately `4.924–4.999 GB`. These are local-tree
planning estimates, not a statement of current logical Lightning quota usage;
the 200-GB quota should be rechecked immediately before any authorized
transfer.

Q75 native MPP, level dimensions, downsample factors, mask geometry, accepted
coordinates, exact runtime, and exact output size are all **unknown** until a
future authorized acquisition/header gate and separately authorized
coordinate gate. They must not be inferred as facts from file size.

## Engineering-value assessment

Q75 is 39.47% larger by declared bytes than Q50 and 109.98% larger than Q25.
Running it would complete the predetermined lower/median/upper-size pilot and
would test upper-quartile transfer, temporary storage, another pyramid/MPP
geometry, a likely longer variable-size WSI bag, artifact publication, and
HEALNet attention memory.

The marginal value is engineering coverage, not biological or predictive
evidence. Q50 was 50.55% larger than Q25 by raw bytes but produced only 15.78%
more patches, so selecting Q75 by SVS byte quantile does not guarantee it is
the cohort's worst compute or attention case. Two successful patients already
validate the same extraction/interface contract; Q75 adds an upper-tail stress
test and a third opportunity to expose scanner/pyramid variation before the
policy is frozen.

For that bounded reason, the recommendation is:

`RUN_Q75_PILOT_RECOMMENDED`

## Exact next authorization gate

The non-executable request package is
`multiscale_feature_pilot/config/brca_q75_acquisition_request.yaml`. If the
user chooses to proceed, the exact next authorization statement is:

> I authorize downloading only BRCA Q75 UUID 25aec062-60d1-446e-a1c6-0c79cc74a770 (1,360,743,825 bytes) from GDC to local persistent staging at concurrency one, followed only by exact filename/size/MD5/SHA256 verification and OpenSlide header metadata collection. No pixel or region reads, coordinates, feature extraction, HEALNet execution, Drive, raw deletion, full-cohort processing, or training.

That statement would authorize only acquisition, exact-file verification, and
header metadata. It would not authorize a mask read or coordinates. If that
gate succeeds, Q75 scale policy and CPU coordinate access must be reviewed and
authorized separately; GPU extraction/smoke would require another later
authorization. Training and full-cohort work remain outside all of these
gates.

## Review stop and repository scope

This review creates only:

- `reports/brca_q25_q50_cpu_review.md`
- `multiscale_feature_pilot/provenance/brca_q25_q50_cpu_review.yaml`
- `multiscale_feature_pilot/config/brca_q75_acquisition_request.yaml`

The pre-existing user modification to
`reports/blca_one_patient_multiscale_pilot.md` was not read for this review and
was not modified. Official HEALNet and the frozen BLCA pilot remain unchanged.
The package is intentionally uncommitted pending parent review. YAML parsing
and whitespace/diff validation are the only tests appropriate to these
documentation/provenance additions.

At the completion snapshot, both YAML files parsed with `yaml.safe_load`; the
recommendation, non-executable status, prohibition fields, guarded-manifest
content, and guarded-manifest SHA assertions passed. `git diff --check` passed,
and a no-index whitespace check of each new file returned no warnings. The full
CPU synthetic suite also passed: `396 passed`. No GPU test or real-data
execution was performed.

Exact Git status at that snapshot on branch
`brca-phase2-metadata-preflight`:

```text
 M reports/blca_one_patient_multiscale_pilot.md
?? multiscale_feature_pilot/config/brca_q75_acquisition_request.yaml
?? multiscale_feature_pilot/provenance/brca_q25_q50_cpu_review.yaml
?? reports/brca_q25_q50_cpu_review.md
```

The modified BLCA report is the protected pre-existing user change; this
review neither authored nor touched it. The three untracked paths are this
review package. The next decision required is whether to accept or decline the
exact Q75 acquisition-and-header-only authorization statement above.
