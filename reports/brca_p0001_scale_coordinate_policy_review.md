# BRCA production P0001 scale and coordinate-policy review

Status: **reviewed and execution-locked**. This CPU-only design used the frozen
P0001 metadata result and existing validated policies. The P0001 WSI was not
opened, and no pixels, masks, coordinates, models, or CUDA operations were
used. P0002 through P0008 remain unstarted.

## Scale mapping

| Branch | Target | Future source | Future read | Output | Effective MPP | Relative error |
|---|---:|---|---:|---:|---:|---:|
| 2× | 0.5 µm/px | level 0 | 512×512 | 256×256 | 0.5 | 0% |
| 4× | 1.0 µm/px | level 1 | 256×256 | 256×256 | 1.0 | 0% |

For the 2× branch, downsampling a 512-pixel level-0 footprint to 256 pixels
gives

```text
effective MPP = 0.25 × (512 / 256) = 0.5 µm/px.
```

For the 4× branch, OpenSlide level 1 has an exact scalar downsample of 4:

```text
effective MPP = 0.25 × 4 = 1.0 µm/px.
```

Both branches therefore have zero target error, well inside the frozen 10%
tolerance. P0001 is also geometrically exact: the dimension-derived x/y ratios
are `(4,4)`, `(16,16)`, and `(32,32)` for levels 1–3. Physical MPP and
coordinate geometry remain separate policy concepts even though their values
coincide for this slide.

## Proposed coordinate geometry

The shared tissue mask would be exactly one future level-2 image of
`5,968×5,120` pixels, read from level-0 location `(0,0)`. Its theoretical size
is 30,556,160 pixels or 122,224,640 RGBA uint8 bytes. No such image was read or
allocated in this review.

The segmentation policy remains saturation threshold 8, median kernel 7,
4×4 closing, area multiplier 100, hole multiplier 16, and at most eight holes.
Because the level-2 coordinate ratio is exactly `16×16`, the scaled reference
area is exactly 1,024 mask pixels, giving tissue and retained-hole thresholds
of 102,400 and 16,384 mask pixels.

| Branch | Level-0 footprint/step | Maximum sites before tissue filtering | Last origin | Trailing right/bottom |
|---|---:|---:|---:|---:|
| 2× | 512×512 | 29,760 (`186×160`) | `(94720,81408)` | `(256,0)` |
| 4× | 1024×1024 | 7,440 (`93×80`) | `(94208,80896)` | `(256,0)` |

The theoretical combined ceiling is 37,200 sites. Actual tissue-selected
counts remain unknown because no mask or coordinate array was generated.

## Next separately authorized operation

The next bounded execution would permit exactly one pixel call:

```text
read_region((0, 0), level=2, size=(5968, 5120))
```

It would then apply the frozen segmentation, create the two coordinate bags,
publish them atomically without overwrite, validate them, and stop. Patch
reads, feature extraction, ResNet50, HEALNet, CUDA, P0002–P0008 processing,
deletion, Drive, cohort expansion, and training remain prohibited.
