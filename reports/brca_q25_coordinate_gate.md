# BRCA Q25 coordinate gate

Execution time: 2026-08-18 21:54 UTC

Status: `BRCA_Q25_COORDINATES_VERIFIED`

The authorized CPU-only coordinate milestone passed. Exactly one OpenSlide
level-2 tissue-mask region was read from the verified Q25 WSI. Two
deterministic coordinate HDF5 files were then generated, atomically published
outside Git, and independently revalidated. No level-0 or level-1 patch pixels
were read and no feature extraction or model execution occurred.

## Exact input identity

- Patient: `TCGA-LL-A6FP`
- Slide ID: `TCGA-LL-A6FP-01Z-00-DX1`
- GDC UUID: `dd3158fb-e1bc-4aac-a742-ca3fc86ed9f6`
- WSI: `TCGA-LL-A6FP-01Z-00-DX1.6261398A-7288-4924-BBE2-FC1949256E40.svs`
- WSI size: `648046947` bytes
- WSI MD5: `75536393096ffd928bc35ec9503c3655`
- WSI SHA256: `ac852be806eb0d91214145983319b604919a8e1d16bd59378a3dba69a600979c`
- Level-0 MPP: `(0.2525, 0.2525)`
- Pyramid dimensions: `[(65736,67406), (16434,16851), (4108,4212), (2054,2106)]`
- OpenSlide downsamples: `[1.0, 4.00005934365913, 16.002635628163056, 32.00527125632611]`

## Authorized mask operation

- OpenSlide calls: one open and one close
- `read_region` calls: exactly one
- Read location in level-0 coordinates: `(0,0)`
- Read level: `2`
- Read size at level 2: `4108×4212`
- Returned array: contiguous `uint8`, RGBA
- Mask-array SHA256: `815f2a2ecf78eb666bbe409f5b278a5432a7bbf97971e37f84127f3beeaf4eea`
- Retained foreground contours: `3`
- Retained holes: `4`

The mask hash is over the exact contiguous RGBA `uint8` array in C order.
The pinned tissue parameters and CLAM behavioral commit are recorded in each
HDF5 dataset's attributes.

## Coordinate artifacts

Artifact directory:
`/teamspace/studios/this_studio/brca_pilot_data/Q25.coordinates`

| Branch | Effective MPP | `coords` | Unique | Duplicates | HDF5 bytes | HDF5 SHA256 | Coordinate-content SHA256 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| scale_2x | `(0.505,0.505)` | `[7404,2] int64` | 7,404 | 0 | 128,136 | `4458a8d1c79eac1b7a0bec68871ffc7a71ec2dc21b4e3b9206ae2601410ff0cc` | `84fbe621186a4c0945f7baf3a7c82f39d54f34b37eb4c39a6498b85c709befdd` |
| scale_4x | `(1.0100149842739303,1.0100149842739303)` | `[1918,2] int64` | 1,918 | 0 | 40,360 | `f055a37018c9b978da60c3aca6258b75853778974e1c4f9ddd32e2aaa2b492fa` | `b6181b69a80b5ba83344fcf7f9de9b53eccdc31a4a09e86548576a8bbba64946` |

Both arrays are non-empty, unique, strictly row-major by `(y,x)`, aligned to
their global level-0 lattices, and contain only complete footprints.

- scale_2x footprint/step: `512×512`; coordinate range x `3584..62464`, y `4096..66560`
- scale_4x footprint/step: `1024×1024`; coordinate range x `3072..62464`, y `4096..65536`
- Manifest size: `6470` bytes
- Manifest SHA256: `7d64ec37595792994e61ab3bf60461498e805bbff0429599f0328b49c93d2ad2`
- Manifest sidecar size: `91` bytes
- Manifest sidecar SHA256: `ecb95686c21979ee349b01a7d3b67ce251cce40efd8c93592577d42db5132230`

The output directory contains exactly the two HDF5 files, the JSON manifest,
and its SHA256 sidecar. Independent read-only validation reproduced every
shape, dtype, count, content hash, file hash, ordering, lattice, and bounds
check.

## Execution identity

- Pilot source commit: `e8b6cd54dcbc94889770dea873204871df32dd6b`
- Runner SHA256: `9b56770f5978d3d507244fa259d024b4570630a3512c51938a3d289e4d0a5a2a`
- Authorization SHA256: `3797f428f5d1d49334fc2c0665325728318083d5deb8831deec5ad1f560ac617`
- Coordinate-policy config SHA256: `85410751aec43b14997fa4c0e2a611ceb329178f788df04f336031104b697d43`
- Coordinate-policy core SHA256: `da4c5e97b6685c4801b73704bfe239ce716eab7017efdb7c1b7db7b70905ca82`
- Artifact publisher SHA256: `a8abb62fae7ca429f57c8de403aa96ab6481a5f514e7229cf994778aa6574ebf`
- Official HEALNet commit: `28ba5da6ab99fd8069972c22e986d83edb658dd4` (clean)
- Frozen BLCA tag commit: `df7cf2bda783ab6cc09e95d6a1fa0914da05a433` (unchanged)
- Python: `3.12.11`
- NumPy: `1.26.4`
- OpenCV: `4.10.0`
- h5py: `3.16.0`
- openslide-python: `1.4.6`
- native OpenSlide: `3.4.1`

The pre-execution synthetic suite passed `265/265`. The runner verified that
every execution-critical file was tracked and byte-identical to the recorded
commit. The unrelated user edit in
`reports/blca_one_patient_multiscale_pilot.md` remained unstaged and unchanged
throughout execution.

## Stop and next milestone

Confirmed not performed:

- level-0 or level-1 patch reads
- patch resampling
- ResNet50 feature extraction
- `.pt` generation
- HEALNet execution
- training
- Q50 or Q75 operations
- Google Drive operations

The coordinate milestone is complete. The next milestone is GPU-only Q25
feature extraction using ImageNet1K V2 ResNet50:

- scale_2x expected feature shape: `[7404,2048]`
- scale_4x expected feature shape: `[1918,2048]`
- concatenated Q25 bag: `[9322,2048]`

It is now safe to switch the Studio to a GPU machine for the next preparation
milestone. This coordinate record does not itself authorize feature
extraction: a separate reviewed GPU extraction gate must pass before any
level-0 or level-1 patch pixels are read.

No claim is made yet that those feature tensors exist or that any prediction
has been produced.
