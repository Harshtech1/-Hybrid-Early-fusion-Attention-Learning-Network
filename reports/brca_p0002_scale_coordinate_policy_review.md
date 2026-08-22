# P0002 scale and coordinate policy review

P0002 was reviewed from its frozen header metadata only. The WSI was not
opened and no pixels, masks, or coordinates were produced.

| Item | Frozen policy |
|---|---|
| Native MPP | 0.2525 × 0.2525 µm/px |
| 2× branch | level 0, 512×512 source to 256×256, effective 0.505 µm/px |
| 4× branch | level 1, 256×256 native patch, effective 1.0100130241914687 µm/px |
| Proposed mask read | `read_region((0,0), 2, (4606,4846))` |
| Mask geometry ratio | 16.00173686495875 × 16.00288898060256 |
| Maximum lattice sites | 21,593 (2×) + 5,325 (4×) = 26,918 |

The proposed tuple is not authorized for execution. A combined explicit
coordinate-execution authorization is required before any WSI pixel access.
