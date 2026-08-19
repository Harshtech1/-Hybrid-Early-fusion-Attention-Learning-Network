# BRCA Six-Patient Staged Acquisition and Header-Only Plan

## Decision and boundary

The proposed six-patient validation batch is accepted **for planning only**. This package does not authorize a download, an OpenSlide open, or any other execution. All six manifests are explicitly marked `NOT_AUTHORIZED`, contain exactly one GDC row, and cannot be combined or advanced automatically.

No WSI was downloaded or opened while preparing this package. No GPU/CUDA, pixel access, coordinate generation, feature extraction, deletion, Google Drive operation, cohort expansion, or training occurred.

## Fixed validation sequence

| Stage | Patient | Quantile | GDC UUID | Declared bytes | MD5 | Omic row |
|---|---|---:|---|---:|---|---:|
| B01 | TCGA-GI-A2C8 | 0.10 | `0a886f18-c44c-4b5e-b243-6df6e27f426a` | 408,704,377 | `a9f830d456b4a1fe0e9bb5b5b99f4b7e` | 924 |
| B02 | TCGA-BH-A0BG | 0.30 | `c5331e5e-10b4-4979-958b-d4592a2805de` | 724,114,911 | `a8c6e730df401ff67e1a1e52a6cb6307` | 472 |
| B03 | TCGA-AR-A1AY | 0.45 | `266c3852-f5d0-4815-94d9-dac5b0ff8276` | 918,454,431 | `1980b183d5bb948c2fc263af62a4b1b4` | 372 |
| B04 | TCGA-D8-A4Z1 | 0.55 | `d39d5d1d-2999-4b82-b2d0-8fba2ccc20cc` | 1,038,326,117 | `597bdbd393212083833deb80358f2301` | 742 |
| B05 | TCGA-LL-A5YL | 0.70 | `1e3d7e89-ee2c-46fd-8697-2e62886657fb` | 1,281,197,011 | `b0394b138fefd48d2a2f43b7900fa284` | 951 |
| B06 | TCGA-EW-A1OY | 0.90 | `00e926c5-d65d-42e5-8ba4-afaab66db47c` | 1,691,494,753 | `dbb9edd8528f25bbcaaee907b1b1ab68` | 897 |

The total declared raw size is 6,062,291,600 bytes (about 6.06 GB decimal). The order intentionally moves from a smaller slide toward the largest slide. Only one patient may be active. After each header report, the process stops; the next patient requires a new explicit authorization.

## Future per-patient gate

```text
separate user authorization
          |
          v
recheck sources + free space + absent destination
          |
          v
download exactly one UUID with one active client
          |
          v
verify UUID / filename / bytes / MD5 / SHA256 / file type / completeness
          |
          v
rematch exact patient + full slide to the frozen Omic row
          |
          v
OpenSlide header only: MPP + level count + dimensions + downsamples
          |
          v
STOP, publish metadata report, request review
```

Before each future download, available storage must be rechecked. The conservative gate is:

\[
A_{required} = 2S_{raw} + 20{,}000{,}000{,}000\ \text{bytes},
\]

where \(S_{raw}\) is that patient's declared WSI size. This reserves space for the retained file, transfer or verification overhead of equal size, and a 20 GB safety floor. The planning-time filesystem snapshot showed 339,486,973,952 bytes available, but this is advisory and must not be reused as an execution-time check.

OpenSlide, if separately authorized later, is restricted to header properties: native `mpp-x`, `mpp-y`, level count, every level's dimensions, and every level's downsample. `read_region`, thumbnails, associated images, masks, coordinates, and all feature/model operations remain forbidden.

## Failure and stopping rules

Any authorization, hash, identity, storage, file-type, completeness, Omic, or OpenSlide-header mismatch fails closed. A pre-existing destination also fails closed. No cleanup or deletion is authorized. A successful B01 metadata result does not authorize B02.

## Exact next authorization requested

The following wording is a template for the next decision and is **not** current authority:

> I authorize downloading only BRCA validation-batch patient B01, UUID `0a886f18-c44c-4b5e-b243-6df6e27f426a` (`408,704,377` bytes), from GDC to local persistent staging at concurrency one using its exact one-row manifest. After download, perform only exact UUID/filename/size/MD5 verification, independent SHA256 calculation, partial/incomplete checks, regular non-symlink SVS confirmation, exact Omic rematch, and OpenSlide header-only inspection of MPP and every pyramid level. Do not call `read_region` or access pixels; do not generate masks, coordinates, or features; do not use GPU/CUDA, Drive, deletion, additional-patient processing, cohort expansion, or training. Stop after the B01 metadata report and wait for my review.

The next required decision is to approve or decline that exact B01-only acquisition and header-only scope.
