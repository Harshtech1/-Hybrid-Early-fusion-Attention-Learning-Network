# P0002–P0008 production coordinate result

Status: **all seven CPU coordinate results verified**.

The authorized phase finished in 101.69 seconds with at most two patient
workers. It performed exactly one level-2 RGBA mask read for each of P0002
through P0008, followed by the frozen tissue segmentation and atomic
publication of two coordinate bags per patient.

| Patient | Level-2 mask | 2× coordinates | 4× coordinates | Wall time |
|---|---:|---:|---:|---:|
| P0002 | 4,606 × 4,846 | 9,785 | 2,486 | 17.64 s |
| P0003 | 3,859 × 3,740 | 3,461 | 921 | 10.99 s |
| P0004 | 5,602 × 4,979 | 3,933 | 1,034 | 25.59 s |
| P0005 | 6,474 × 5,594 | 23,559 | 5,971 | 39.19 s |
| P0006 | 7,019 × 5,184 | 7,505 | 1,962 | 31.36 s |
| P0007 | 5,879 × 5,017 | 9,238 | 2,407 | 42.36 s |
| P0008 | 7,679 × 4,583 | 18,877 | 4,799 | 24.47 s |
| **Total** | **7 reads** | **76,358** | **19,580** | **101.69 s phase wall time** |

The seven artifact directories contain exactly 28 files and 95,938 coordinate
rows in total. Their combined size is 1,705,741 bytes. Independent read-only
validation recomputed every manifest, sidecar, HDF5 file, and coordinate-content
hash and verified nonempty unique row-major lattices, integer dtype, attributes,
bounds, and policy capacities. No successful-run staging directory remained.

The WSI descriptor remained held through each patient's publication, while the
coordinate parent remained held through a no-follow directory descriptor and
dirfd-relative `RENAME_NOREPLACE`. No patch read, feature extraction, ResNet50,
HEALNet, GPU/CUDA, deletion, Drive, cohort expansion, or training operation was
performed.

## Pipeline position

P0001–P0008 now all have verified 2× and 4× coordinate artifacts. The next
phase is CPU-only preparation of P0002–P0008 compact GPU feature packages and a
consolidated first-eight GPU execution plan. GPU switching is not required
until those packages have passed review.
