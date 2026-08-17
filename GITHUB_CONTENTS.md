# GitHub publication boundary

This repository publishes only portable source code, tests, small configuration/provenance records, and reviewed documentation.

## Included

- multiscale tensor, provenance, padding, and patient-matched Omic adapters;
- synthetic unit/interface tests;
- read-only GPU/source/data/checkpoint readiness checker;
- GPU restart documentation and dependency snapshot;
- small YAML/Markdown provenance;
- explicitly reviewed public TCGA identifier/checksum metadata; and
- immutable third-party source URLs and commits.

## Excluded

- raw or derived `.svs`, `.h5`, `.pt`, `.pth`, `.ckpt`, `.tif`, and similar artifacts;
- complete TCGA datasets or archives;
- generated masks, stitches, patch images, logs, caches, and temporary files;
- the nested CLAM or official HEALNet checkout;
- OAuth/rclone configuration, tokens, credentials, private keys, and `.env` files; and
- any proprietary or controlled-access material.

## Pre-push checks

1. Stage only explicit paths; never `git add .` or `git add -A`.
2. Inspect `git diff --cached --stat` and `git diff --cached`.
3. Run `git diff --cached --check`.
4. Confirm no forbidden extensions, ignored data, symlinks, credentials, or large files are staged.
5. Run the full synthetic test suite and YAML validation.
6. Record the branch, commit, and pushed remote without claiming that excluded data were published.

The GPU handoff intentionally requires secure out-of-band transfer or targeted Git LFS retrieval of the verified BLCA inputs. See [GPU_HANDOFF.md](GPU_HANDOFF.md).
