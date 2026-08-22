# BLCA pilot data restoration

> Historical intermediate checkpoint. This report captured the moment before the verified HDF5 arrived. It is superseded by [validated_pilot_baseline.md](validated_pilot_baseline.md) and must not be read as current readiness state.

Checked: 2026-08-17 UTC

## Scope

This milestone restores and validates existing one-patient BLCA inputs only. It does not perform patch extraction, coordinate generation, ResNet inference on the WSI, HEALNet execution, training, or KIRP acquisition.

## WSI

- GDC UUID: `bc9e3954-59d0-4f25-9022-42c97db7aea2`
- Path: `/teamspace/studios/this_studio/healnet_pilot_data/blca/bc9e3954-59d0-4f25-9022-42c97db7aea2/TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.svs`
- Acquisition: direct download of the single UUID with official NCI GDC Data Transfer Tool 2.3 for Ubuntu
- Expected size: `2,658,499,382` bytes
- Measured size: `2,658,499,382` bytes
- Expected MD5: `824785fee9387dcf46a7058a0722739b`
- Measured MD5: `824785fee9387dcf46a7058a0722739b`
- Result: `VERIFIED`
- Existing WSI duplicated: no; the exact WSI was absent before download
- Source file modified: no

## Coordinate HDF5

- Expected filename: `TCGA-2F-A9KT-01Z-00-DX1.ADD6D87C-0CC2-4B1F-A75F-108C9EB3970F.h5`
- Expected size: `572,080` bytes
- Expected SHA256: `e22bc149a2b192bf917559fb69154143867e8e4e6eb3f28a7b5b5a87a07e0b51`
- Search result: absent from bounded workspace/content roots
- Structural validation: not possible because the artifact is absent
- HDF5 regenerated: no
- Result: `HDF5_TRANSFER_REQUIRED`

The previously validated HDF5 must be transferred. It must not be silently regenerated or replaced.

## Omic

- Path: `/teamspace/studios/this_studio/healnet/data/tcga/omic_xena/blca_master.csv`
- Expected size: `9,318,450` bytes
- Measured size: `9,318,450` bytes
- Expected SHA256: `9fa2cd83906c00e1f50113ba8e806ea4537806aeb2aacbee60e86c80b53f6929`
- Measured SHA256: `9fa2cd83906c00e1f50113ba8e806ea4537806aeb2aacbee60e86c80b53f6929`
- Patient: `TCGA-2F-A9KT`
- Exact slide match: previously revalidated on this machine with the strict loader
- Result: `VERIFIED`

## Readiness

- WSI: verified
- HDF5: absent
- Omic: verified
- Full readiness checker: not run because all three required inputs are not present
- Real WSI opened for extraction: no
- Real feature extraction: no
- `.pt` generated: no
- HEALNet run: no
- KIRP downloaded: no

Final status: `HDF5_TRANSFER_REQUIRED`
