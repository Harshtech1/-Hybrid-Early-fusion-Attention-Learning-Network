# P0002–P0008 combined CPU coordinate-execution request

Status: **prepared for review; execution is not authorized**.

All seven raw WSIs passed exact manifest identity, size, MD5, independently
calculated SHA256, partial-file, regular non-symlink, exact Omic-row, MPP, and
complete pyramid-header checks. Their scale and coordinate policies were then
derived from the committed metadata reports without reopening a WSI.

| Patient | Proposed exact OpenSlide call | Maximum lattice sites before tissue filtering |
|---|---|---:|
| P0002 | `read_region((0,0), 2, (4606,4846))` | 26,918 |
| P0003 | `read_region((0,0), 2, (3859,3740))` | 17,400 |
| P0004 | `read_region((0,0), 2, (5602,4979))` | 33,824 |
| P0005 | `read_region((0,0), 2, (6474,5594))` | 43,935 |
| P0006 | `read_region((0,0), 2, (7019,5184))` | 44,307 |
| P0007 | `read_region((0,0), 2, (5879,5017))` | 35,646 |
| P0008 | `read_region((0,0), 2, (7679,4583))` | 42,626 |

The proposed phase permits exactly seven mask reads in total, one per patient,
with at most two CPU patient workers. Each completed patient would receive two
validated coordinate bags and an atomic no-overwrite result. The combined raw
RGBA mask volume is 807,742,184 bytes if all masks were retained concurrently;
the implementation must release each in-memory mask after its patient result.

Patch reads, feature extraction, ResNet50, HEALNet, CUDA, deletion, Drive,
cohort expansion, and training remain locked. Actual tissue-selected coordinate
counts remain unknown until this separate execution is authorized.

## Exact reply needed to proceed

> I authorize the exact P0002-P0008 CPU coordinate phase defined in
> `multiscale_feature_pilot/config/brca_p0002_p0008_coordinate_execution_request.yaml`:
> exactly one OpenSlide level-2 RGBA mask read per patient at its frozen tuple,
> with at most two CPU patient workers, followed by the frozen tissue segmentation
> and atomic no-overwrite publication and validation of each patient's 2x and 4x
> coordinate bags. No patch reads, feature extraction, ResNet50, HEALNet,
> GPU/CUDA, patients outside P0002-P0008, deletion, Drive operations, cohort
> expansion, or training. Stop after all seven coordinate results are validated
> and reported.
