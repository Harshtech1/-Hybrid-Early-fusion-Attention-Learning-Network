# BRCA B03 scale and coordinate-policy review

Status: **reviewed and execution-locked**. This CPU-only design used only the frozen B03 metadata result and existing reviewed policies. The B03 WSI was not opened, and no pixels, masks, coordinates, models, or CUDA operations were used.

## Scale mapping

| Branch | Target | Source | Read | Output | Effective MPP | Operation |
|---|---:|---|---:|---:|---:|---|
| 2× | 0.5 µm/px | level 0 | 512×512 | 256×256 | 0.4936 | Lanczos downsample |
| 4× | 1.0 µm/px | level 1 | 256×256 | 256×256 | 0.9872085 | native, no resampling |

The relative target errors are 1.28% and 1.279%, respectively, inside the frozen 10% tolerance. Physical MPP uses the OpenSlide scalar downsample. Coordinate mapping instead uses independent dimension ratios; at level 1 these are `(4.0, 4.0000688231)`.

## Proposed coordinate geometry

The shared tissue mask would be exactly one level-2 image of `5,831×3,632` pixels. Level-0 coordinate ratios are `(16.0, 16.0024779736)`. Its theoretical RGBA allocation is 84,712,768 bytes.

The validated segmentation settings remain saturation threshold 8, median kernel 7, 4×4 closing, area multiplier 100, hole multiplier 16, and at most eight holes. This is the reviewed internal engineering policy, not a claim of exact reproduction of the paper authors' historical CLAM runtime.

| Branch | Level-0 footprint/step | Maximum sites before tissue filtering | Last origin | Trailing right/bottom |
|---|---:|---:|---:|---:|
| 2× | 512×512 | 20,566 (`182×113`) | `(92672,57344)` | `(112,265)` |
| 4× | 1024×1024 | 5,096 (`91×56`) | `(92160,56320)` | `(112,777)` |

Actual tissue-selected counts remain unknown. No coordinate array was generated.

## Next separately authorized operation

The next bounded execution would permit exactly one call:

```text
read_region((0, 0), level=2, size=(5831, 3632))
```

followed by the frozen segmentation, creation of both coordinate bags, atomic no-overwrite publication, strict validation, and an immediate stop. Patch reads, features, ResNet50, HEALNet, CUDA, B04–B06, deletion, Drive, cohort expansion, and training would remain prohibited.
