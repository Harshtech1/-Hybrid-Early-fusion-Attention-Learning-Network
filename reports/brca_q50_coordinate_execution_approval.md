# BRCA Q50 coordinate execution approval

Status: `Q50_COORDINATE_EXECUTION_AUTHORIZED_CPU_ONLY`

The user's instruction “ok do it start it” authorizes the exact next Q50
milestone described immediately beforehand. The approved operation is one
native level-2 mask read followed by deterministic CPU coordinate generation,
atomic external publication, validation, and reporting.

## Exact authorized object

- Patient: `TCGA-AR-A1AW`
- GDC UUID: `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62`
- Size: `975,626,387` bytes
- MD5: `304509e03f26cbecc9aee4ea691c8e5a`
- SHA-256: `6b960db7e6e5772f4f626daa7b023ffbb5e3b20f9f6ff5c7d52f4554cf33399b`
- Mask read: level 2, location `(0,0)`, size `6,247×4,083`, maximum one call
- Output: `/teamspace/studios/this_studio/brca_pilot_data/Q50.coordinates`

## Required stop

This record does **not** authorize level-0/level-1 patch reads, Lanczos
resampling execution, ResNet50, `.pt` features, HEALNet, Q25 reprocessing,
Q75, full-cohort work, Google Drive, raw deletion, or training.

The runner binds the exact Q50 WSI, policy commit, source hashes, official
HEALNet commit and frozen BLCA tag. It rehashes the WSI before publication,
requires all critical execution files to be byte-identical to the execution
commit, refuses existing/stale/symlink output paths, and publishes through the
reviewed atomic no-replace artifact writer.

Verification before execution:

- runner boundary tests: `12/12` passed;
- complete pilot suite: `347/347` passed;
- diff and compilation checks: passed;
- real Q50 pixel reads at approval-record time: `0`.
