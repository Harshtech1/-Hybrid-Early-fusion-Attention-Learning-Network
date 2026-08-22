# P0003 scale and coordinate policy review

P0003 was reviewed from its frozen header metadata only. The WSI was not
opened and no pixels, masks, or coordinates were produced.

| Item | Frozen policy |
|---|---|
| Native MPP | 0.2525 × 0.2525 µm/px |
| 2× branch | level 0, 512×512 source to 256×256, effective 0.505 µm/px |
| 4× branch | level 1, 256×256 native patch, effective 1.01 µm/px |
| Proposed mask read | `read_region((0,0), 2, (3859,3740))` |
| Mask geometry ratio | 16.002073075926404 × 16.0 |
| Maximum lattice sites | 13,920 (2×) + 3,480 (4×) = 17,400 |

The proposed tuple is not authorized for execution. A combined explicit
coordinate-execution authorization is required before any WSI pixel access.
