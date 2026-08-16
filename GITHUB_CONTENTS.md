# Planned GitHub repository contents

No Git repository or GitHub repository has been created yet. This document defines the intended boundary for the future repository.

## Include

The future repository should contain reproducibility code, small metadata, and documentation:

- `README.md`
- `PROJECT_STATE.md`
- `TOMORROW_START.md`
- `GITHUB_CONTENTS.md`
- `.gitignore`
- `reports/`
- `shared/provenance/`
- `tracks/paper_faithful/provenance/`
- `tracks/released_code/provenance/`
- experiment configuration files
- small CSV manifests and inclusion/exclusion lists
- checksum metadata
- feature-extraction and validation scripts created for this project
- loader smoke tests
- environment and dependency specifications
- documentation describing every deviation from the paper or released code

The pinned CLAM commit should be recorded as metadata or reproduced through an installation script. Do not vendor the current nested `CLAM/` checkout into the future repository.

## Keep outside GitHub

- raw `.svs` whole-slide images
- large/generated `.h5` coordinate or patch datasets
- `.pt` feature tensors
- `.pth`/`.ckpt` model checkpoints
- full TCGA datasets
- large archives (`.tar`, `.zip`, `.7z`, and similar)
- generated masks and stitches unless a deliberately small, reviewed example is separately approved
- the nested external CLAM checkout
- OAuth tokens, refresh tokens, client secrets, or OAuth JSON
- `rclone.conf`, `healnet.conf`, or any Google Drive configuration
- `.env` files and credentials
- proprietary, controlled-access, or authentication material

## Before the first commit

1. Run an explicit credential and large-file audit.
2. Confirm `.gitignore` excludes the existing CLAM demo checkpoint and demo slides.
3. Confirm no `.svs`, `.h5`, `.pt`, `.pth`, `.ckpt`, Drive config, or token is staged.
4. Review all small manifests for identifiers that are acceptable to publish.
5. Record external source URLs and immutable commits rather than copying third-party repositories.
6. Create the new Git repository only after separate authorization.

Markdown, YAML provenance, checksums, source code, experiment configuration, reports, and small non-sensitive CSV manifests are intentionally not globally ignored.
