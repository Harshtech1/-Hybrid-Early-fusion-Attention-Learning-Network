# P0005–P0008 scale and coordinate policy review

Status: **metadata policies reviewed; all pixel and execution stages locked**.

The four policies were derived exclusively from the frozen P0005–P0008
header-result bundles. No WSI was opened and no pixel, mask, coordinate,
feature, CUDA, HEALNet, deletion, Drive, expansion, or training operation was
performed.

| Patient | Base MPP | Proposed only `read_region(location, level, size)` | 2× effective MPP | 4× effective MPP | Theoretical 2× / 4× lattice sites |
|---|---:|---|---:|---:|---:|
| P0005 | 0.2525 | `([0,0], 2, [6474,5594])` | 0.5050 | 1.0100169 | 35,148 / 8,787 |
| P0006 | 0.2521 | `([0,0], 2, [7019,5184])` | 0.5042 | 1.0084151 | 35,478 / 8,829 |
| P0007 | 0.2521 | `([0,0], 2, [5879,5017])` | 0.5042 | 1.0084349 | 28,548 / 7,098 |
| P0008 | 0.2521 | `([0,0], 2, [7679,4583])` | 0.5042 | 1.0084041 | 34,177 / 8,449 |

All physical-scale errors are below the frozen 10% tolerance. Coordinate
geometry uses independent level-0/level width and height ratios; scalar
OpenSlide downsample values are used only for physical MPP selection. The 2×
branch uses a 512×512 level-0 footprint resampled to 256×256. The 4× branch
uses native 256×256 level-1 patches with the existing CLAM-compatible integer
level-0 lattice contract.

The mask tuples above are proposals for a future authorization request only.
They do not authorize `read_region`. Each future coordinate execution must use
exactly one listed level-2 read, shared by both branches, followed by the
frozen HSV segmentation and atomic coordinate artifact contract. Actual tissue
counts remain unknown until that separate execution is approved.

The next gate is a combined exact P0002–P0008 coordinate authorization after
all seven policy reviews have been assembled and independently validated.
