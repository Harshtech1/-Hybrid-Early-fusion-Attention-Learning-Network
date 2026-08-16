# Third-party data notice

This repository documents a reproducibility and engineering study of HEALNet. It does not redistribute The Cancer Genome Atlas (TCGA) datasets or other large biomedical data.

## Data sources

- TCGA data are externally sourced and remain subject to the terms, policies, and access controls of their original providers.
- Whole-slide image (`.svs`) files used by this project are obtained from the National Cancer Institute Genomic Data Commons (GDC) using GDC manifests and the official GDC Data Transfer Tool.
- The HEALNet authors' repository provides or refers to data-acquisition mechanisms and cohort references; it is not mirrored in this repository.
- TCIA cohort pages are retained as scientific reference links, while the repository workflow acquires the pathology slides through GDC.

## Repository boundary

Large biomedical datasets, raw WSI files, derived HDF5 coordinate files, feature tensors, model checkpoints, and authentication material must remain outside this repository. They are excluded by `.gitignore` and must also be excluded through explicit staging review.

Small reproduction metadata may be retained here, including:

- public dataset identifiers;
- filenames and cohort labels;
- file sizes and cryptographic checksums;
- preprocessing parameters;
- software versions and immutable source commits; and
- validation results that do not contain the underlying biomedical data.

Storing metadata or checksums here does not grant access to, transfer, sublicense, or redistribute the corresponding third-party data.

Official sources:

- GDC Data Portal: <https://portal.gdc.cancer.gov/>
- GDC Data Transfer Tool: <https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/>
- Official HEALNet repository: <https://github.com/konst-int-i/healnet>
- HEALNet paper: <https://arxiv.org/abs/2311.09115>
