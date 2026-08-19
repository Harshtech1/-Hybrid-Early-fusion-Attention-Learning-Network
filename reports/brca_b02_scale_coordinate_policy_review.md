# BRCA B02 scale and coordinate-policy review

Status: **reviewed and execution-locked**. This CPU-only design used only frozen B02 metadata. The WSI was not opened, and no pixels, masks, coordinates, models, or CUDA operations were used.

## Scale mapping

| Branch | Target | Source | Read | Output | Effective MPP | Operation |
|---|---:|---|---:|---:|---:|---|
| 2× | 0.5 µm/px | level 0 | 512×512 | 256×256 | 0.501 | Lanczos downsample |
| 4× | 1.0 µm/px | level 1 | 256×256 | 256×256 | 1.0020374 | native, no resampling |

Both errors are approximately 0.2%, well inside the frozen 10% tolerance. Physical MPP uses the OpenSlide scalar downsample. Coordinate mapping instead uses independent dimension ratios; at level 1 these are `(4.0001343966, 4.0001644557)`.

## Proposed coordinate geometry

The shared tissue mask would be exactly one level-2 image of `5,580×4,560` pixels. Level-0 coordinate ratios are `(16.0019713262, 16.0024122807)`. Its theoretical RGBA allocation is 101,779,200 bytes.

The validated segmentation settings remain saturation threshold 8, median kernel 7, 4×4 closing, area multiplier 100, hole multiplier 16, and at most eight holes. This is the reviewed internal engineering policy, not a claim of exact reproduction of the paper authors' historical CLAM runtime.

| Branch | Level-0 footprint/step | Maximum sites before tissue filtering | Last origin | Trailing right/bottom |
|---|---:|---:|---:|---:|
| 2× | 512×512 | 24,708 (`174×142`) | `(88576,72192)` | `(203,267)` |
| 4× | 1024×1024 | 6,177 (`87×71`) | `(88064,71680)` | `(203,267)` |

Actual tissue-selected counts remain unknown. No coordinate array was generated.

## Next separately authorized operation

The next bounded execution would permit exactly one call:

```text
read_region((0, 0), level=2, size=(5580, 4560))
```

followed by the frozen segmentation, creation of both coordinate bags, atomic no-overwrite publication, strict validation, and an immediate stop. Patch reads, features, ResNet50, HEALNet, CUDA, B03–B06, deletion, Drive, cohort expansion, and training would remain prohibited.
