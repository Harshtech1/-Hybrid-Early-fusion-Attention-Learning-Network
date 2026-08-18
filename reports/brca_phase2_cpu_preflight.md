# BRCA Phase 2 CPU preflight

Status: `BRCA_PHASE_2_CPU_PREFLIGHT_READY_ACQUISITION_BLOCKED`

Phase 2 preparation is complete on the CPU machine. The software, manifests,
source identities, and synthetic metadata policy are ready. The storage values
below are a static plan, not a live Lightning quota measurement. The gate
correctly reports:

```text
cpu_preflight_ready: true
acquisition_authorized: false
ready_to_download: false
```

No GDC download request was made, no BRCA WSI was downloaded or opened,
OpenSlide was not imported by this preflight, and no coordinates, features, or
training outputs were produced.

The hardened Phase 2 checks passed `82/82` focused tests, and the complete
pilot suite passed `203/203` tests. The central checker independently verified
the frozen proposal, row-level alignment, official BRCA manifest, and exact
three one-row artifacts; it did not derive trust solely from editable Phase 2
configuration.

## Verified GDC client

The tracked `gdc-client` files in the author repositories are macOS Mach-O
binaries and remain untouched. Phase 2 instead uses the external Linux client:

| Property | Verified value |
|---|---|
| Official package | NCI GDC Data Transfer Tool 2.3.0, Ubuntu x64 |
| Executable | `/teamspace/studios/this_studio/tools/gdc-client/2.3.0/gdc-client` |
| Runtime version | `2.3` |
| Binary type | ELF 64-bit x86-64 |
| Binary SHA256 | `1df9281cbacbb53413607a4b9b682947dcb48b6ac7fade6527748bc025ae8c96` |
| Outer archive bytes | `21,987,739` |
| Outer archive MD5 | `18591d74de07cdcd396dab71c52663da` |
| Outer archive SHA256 | `5805635e5551962280589f5a59d047bde93b380c653de2b9b62460e7336c8189` |

The archive MD5 matches the checksum published on the official
[GDC Data Transfer Tool page](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool).
No dataset command was executed while validating the client.

## Guarded one-row candidates

Exactly three standard GDC manifests were generated. Each contains the normal
five-column header and exactly one data row. There is deliberately no combined
manifest and no generated download command. Every basename and the manifest-set
record contain `NOT_AUTHORIZED`.

| Label | Patient | UUID | Declared bytes | Manifest SHA256 |
|---|---|---|---:|---|
| Q25 | `TCGA-LL-A6FP` | `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6` | 648,046,947 | `95bb1c4491497c265b868e96698cef3c3dd501458f801dae3c1dd702f3efa297` |
| Q50 | `TCGA-AR-A1AW` | `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62` | 975,626,387 | `1028665e49bab6895a47947d069e225c2d6b4b90f420f7a9dc916ca9313a6062` |
| Q75 | `TCGA-E2-A154` | `25aec062-60d1-446e-a1c6-0c79cc74a770` | 1,360,743,825 | `8e2a17e21c6dafbae384ea729ed24b7c769df7510df0b318ee6ce15e2169553a` |

Their total declared raw size is `2,984,417,159` bytes. Every row was checked
against the frozen proposal, exact row-level alignment, and official filtered
GDC manifest for UUID, complete filename, MD5, size, and released state.

The manifest-set record SHA256 is
`bcb5c45961556fceb4e0575d3ea3fa35867d5825c85164175287964be9ce1957`.
The complete implementation hash map is recorded in
`multiscale_feature_pilot/provenance/brca_phase2_cpu_preflight.yaml`.

## Proposed metadata policy

The metadata evaluator takes supplied values only; it has no path parameter,
does not import OpenSlide, and cannot open a real slide. It validates positive
finite `mpp-x/mpp-y`, dimensions, and downsamples before calculating native
MPP independently on each axis.

The current proposal is:

- targets: approximately `0.5 µm/px` and `1.0 µm/px`;
- proposed maximum relative error: 10% on each axis;
- nearest native level selected independently on x and y;
- exact ties choose the lower level index;
- the two targets must select distinct native levels;
- axis disagreement, invalid metadata, ambiguity, or excess error rejects the
  slide;
- no silent fallback or resampling.

This policy remains `PENDING_SUPERVISOR_APPROVAL`. Passing it with synthetic or
later collected metadata cannot itself authorize a download, slide opening, or
extraction. In particular, a BRCA slide without two acceptable distinct native
levels would be rejected under the current native-only proposal; resampling
would require a separately approved deterministic policy.

## Storage gate

The user-reported organization limit remains 200 GB, regardless of the larger
local `df` capacity. Planning retains a 20 GB safety floor and provisional 5 GB future
training hold. The prior conservative reference peak for the largest proposed
slide is `28,666,606,029` bytes, so a concurrency-one pilot is technically
plausible. Actual feature storage remains unknown until an authorized slide's
coordinates are counted; this estimate is not permission to acquire data.
The Lightning quota/dashboard and local staging ledger must be checked again
immediately before any future download.

## Remaining blockers

1. Explicit supervisor approval for these exact three UUIDs is not recorded.
2. The 10% native-level MPP policy and no-resampling rule are still pending.
3. All three candidate manifests remain explicitly `NOT_AUTHORIZED`.

After those decisions are recorded, begin with Q25 only, concurrency one.
Verify its exact size and MD5 before opening it, read metadata before any
coordinate generation, and stop to report the MPP/pyramid result. Do not move
to Q50 automatically.

The GPU is not required yet. Switch only after one authorized WSI passes its
size, MD5, MPP, pyramid, coordinate, and storage gates, immediately before
ResNet50 feature extraction.
