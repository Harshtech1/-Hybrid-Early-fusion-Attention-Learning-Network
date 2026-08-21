# P0004 scale and coordinate policy review

P0004 was reviewed from its frozen header metadata only. The WSI was not
opened and no pixels, masks, or coordinates were produced.

| Item | Frozen policy |
|---|---|
| Native MPP | 0.2525 × 0.2525 µm/px |
| 2× branch | level 0, 512×512 source to 256×256, effective 0.505 µm/px |
| 4× branch | level 1, 256×256 native patch, effective 1.0100063384878 µm/px |
| Proposed mask read | `read_region((0,0), 2, (5602,4979))` |
| Mask geometry ratio | 16.001428061406642 × 16.00180759188592 |
| Maximum lattice sites | 27,125 (2×) + 6,699 (4×) = 33,824 |

The proposed tuple is not authorized for execution. A combined explicit
coordinate-execution authorization is required before any WSI pixel access.
