# BRCA Validation Batch B01 — Scale and Coordinate Policy Review

Status: `REVIEWED_BRCA_B01_SCALE_AND_COORDINATE_POLICY_V1_EXECUTION_LOCKED`

This CPU-only review used the frozen B01 header record. It did not open the WSI, read pixels, generate a mask or coordinates, publish artifacts, use CUDA, or process B02.

## Scale mapping

| Branch | Target | Source | Read | Output | Effective MPP | Operation |
|---|---:|---|---:|---:|---:|---|
| 2× | 0.5 µm/px | level 0 | 512×512 | 256×256 | 0.4936 | Lanczos downsample |
| 4× | 1.0 µm/px | level 1 | 256×256 | 256×256 | 0.9872377 | native, no resampling |

Both branches are within the frozen 10% physical-MPP tolerance. The level-1 OpenSlide scalar downsample is `4.00015264068383`; it is used for physical MPP, while coordinate geometry uses the independently derived dimension ratios `(4.0, 4.000305281367661)`.

## Future coordinate geometry

The proposed shared tissue mask is one level-2 image of `3,986×2,456` pixels. Its level-0 coordinate ratios are `(16.00200702458605, 16.006107491856678)`. An in-memory RGBA mask would contain 9,789,616 pixels and 39,158,464 raw bytes.

The reviewed HSV segmentation semantics remain the same as the successful pilots: saturation threshold 8, median kernel 7, 4×4 morphological closing, area multiplier 100, hole multiplier 16, and at most eight retained holes. This is a documented engineering policy compatible with the internal reviewed implementation; it is not claimed to reproduce the paper authors' exact CLAM runtime.

| Branch | Level-0 footprint/step | Maximum sites before tissue filtering | Last complete origin | Trailing right/bottom |
|---|---:|---:|---:|---:|
| 2× | 512×512 | 9,424 (`124×76`) | `(62976,38400)` | `(296,399)` |
| 4× | 1024×1024 | 2,356 (`62×38`) | `(62464,37888)` | `(296,399)` |

Actual accepted patch counts remain unknown until a separately authorized mask read and coordinate execution.

## Required next authorization

The next bounded operation would be exactly one OpenSlide call:

```text
read_region((0, 0), level=2, size=(3986, 2456))
```

followed by the frozen in-memory tissue segmentation, generation of the two coordinate bags, atomic no-overwrite publication, validation, and an immediate stop. Patch reads, feature extraction, GPU/CUDA, B02 acquisition, deletion, Drive, cohort processing, and training would remain prohibited.

No such execution is currently authorized.
