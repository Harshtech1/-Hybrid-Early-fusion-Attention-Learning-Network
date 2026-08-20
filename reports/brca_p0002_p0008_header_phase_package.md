# P0002–P0008 acquisition and header-only execution package

This package binds the seven exact frozen first-block identities and their
byte-identical authorized one-row GDC manifests. It is ready for a committed
source review but has not been executed.

## Execution boundary

- Downloads use exactly one GDC client.
- One download may overlap one preceding CPU hash/header worker; therefore no
  more than two patients can be active.
- Each WSI is verified by UUID, filename, size, MD5, independent SHA256,
  regular non-symlink status, and absence of partial or unexpected files.
- The exact Omic row and RNA/mutation/CNV content hashes are verified through
  a held `O_NOFOLLOW` descriptor.
- OpenSlide uses a stable held `/proc/self/fd` path and records only MPP and
  every pyramid level. Static and runtime guards prohibit pixel APIs.
- Each result is atomically published without overwrite; raw WSIs are retained.

The exact authorized raw total is **6,527,281,524 bytes**. This package does
not generate scale policies, masks, coordinates, patches, features, or model
outputs. Those policy artifacts are produced only after the seven measured
headers exist.

The runner applies a whole-block preflight floor of **33,054,563,048 bytes**
before creating P0002 and revalidates every exclusively created `0700`
incoming directory by device, inode, and mode around transfer, hashing, header
inspection, and result publication. The frozen authorization SHA256 is
`ec70da619537db9c395fa33ed16594a8e363a4981132154fa70d5100cb2732dd`.

## Stop

After the seven header bundles are published, the CPU policy workstream may
derive exact scale policies and proposed level-2 mask-read tuples. Pixel access,
GPU/CUDA, HEALNet, deletion, Drive operations, cohort expansion, and training
remain prohibited.
