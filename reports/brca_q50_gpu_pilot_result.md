# BRCA Q50 GPU feature pilot result

Status: `BRCA_Q50_GPU_FEATURE_PILOT_SUCCESS`

The authorized Q50-only GPU pilot completed successfully on the Tesla T4. It
produced and atomically published two float32 ImageNet1K V2 ResNet50 feature
bags, their exact row-wise concatenation, and row-level provenance. It then
passed both the pre-pixel synthetic and real-feature four-modality HEALNet
numerical interface checks and reached the required stop. No training occurred.

## Same-patient bound inputs

- Case: `TCGA-AR-A1AW`
- GDC UUID: `5c1216f3-19ec-4d3c-9bb0-9bd740b79f62`
- WSI/slide ID:
  `TCGA-AR-A1AW-01Z-00-DX1.E527CA46-D83F-4055-8C7E-AEFEF13C1E29.svs`
- WSI: `975,626,387` bytes; MD5
  `304509e03f26cbecc9aee4ea691c8e5a`; SHA256
  `6b960db7e6e5772f4f626daa7b023ffbb5e3b20f9f6ff5c7d52f4554cf33399b`
- Level-0 dimensions: `99,960 × 65,334` = `6,530,786,640` pixels;
  native MPP `0.2468 × 0.2468`
- Coordinate manifest SHA256:
  `a79179e7fe3a0cf6bdda5089da11910c0db6b1f4bcab9a15c6d5b8e7ec43e485`
- 2x coordinates: 8,580; HDF5 SHA256
  `f412b3e04e11c7e2dbfd2db3da70c5eda45a1634e7a48aa322796f007c5b1050`;
  coordinate-content SHA256
  `9775ccdb9aec661021b26e2a2e005ccd5d9bbf528578bbebd1fc5b8aa303084f`
- 4x coordinates: 2,213; HDF5 SHA256
  `f4032dd245359694e59bff8356cb4f0b673f2e71c44fac7f19f1873da9685479`;
  coordinate-content SHA256
  `8bb20dbd7bb876f339b1ba85beed95922d732c36c037273f22f68f46069b0b7b`
- Official clean Omic archive SHA256:
  `4bc9de58ef6e8f8f7566fef6512c6c7db62cb827e4117b6f356d679e26e3e5f8`;
  exact source row `370`
- RNA: `[1,1,1558]`; mutation: `[1,1,21]`; CNV: `[1,1,1333]`
- Encoder: classifier-removed torchvision ResNet50 with
  `ResNet50_Weights.IMAGENET1K_V2`; checkpoint SHA256
  `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`

The WSI, coordinate bags, Omic archive/row, and checkpoint were identity- and
hash-gated before use and rechecked before publication. The exact case and
full slide ID matched the Omic row. RNA, mutation, and CNV remained three
separate HEALNet modalities and were never concatenated with the WSI tensor.

## Feature result

| Branch | Source | Effective MPP | Feature shape | Finite |
|---|---|---:|---:|---:|
| scale-2x | level 0, `512×512` to `256×256` LANCZOS | `0.4936` | `[8580,2048]` | yes |
| scale-4x | native level 1, `256×256` | `0.9872151105124595` | `[2213,2048]` | yes |
| concatenated | 2x rows followed by 4x rows | — | `[10793,2048]` | yes |

The combined tensor is exactly
`torch.cat([scale_2x, scale_4x], dim=0)`. Its first 8,580 rows are the 2x bag
and its remaining 2,213 rows are the 4x bag. No pooling or transpose was
performed. Adding only a batch axis produced the natural HEALNet WSI input
`[1,10793,2048]`.

## Four-modality HEALNet checks

The launch environment contained `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA
initialization. Deterministic algorithms and deterministic cuDNN were enabled;
TF32, automatic mixed precision, cuDNN benchmarking, CPU fallback, backward
passes, and optimizer execution were disabled.

Before any pixel-capable dataset was constructed, a full-size synthetic check
passed. After sequential ResNet extraction, the real-feature check also
passed. Both used channel dimensions `[2048,1558,21,1333]`, no mask or padding,
and produced:

- finite float32 output `[1,4]`
- WSI attention `[1,2,10793]`
- RNA, mutation, and CNV attention `[1,2,1]` each

HEALNet was randomly initialized and in evaluation/inference mode. These
outputs establish tensor compatibility and numerical finiteness only. They are
**not a trained survival prediction or scientific inference**.

## Retained artifacts

Directory:
`/teamspace/studios/this_studio/brca_pilot_data/Q50.features`

The six-file directory is external to the Git repository and was published by
an atomic no-overwrite directory operation only after all gates passed. The
runner validated it after publication; this result-recording audit also
independently recomputed every file SHA256 and checked the manifest sidecar.

| File | Bytes | SHA256 | Tensor-content SHA256 |
|---|---:|---|---|
| `scale_2x_features.pt` | 70,288,937 | `b654895cfedec9099c893ef7a25f8646bf2656454937dfd006c038315cd37e3b` | `607f1e67c76f2ea44f899b518fbb9da28ec1bf18a8f6ed5bb368cb23b65c94ce` |
| `scale_4x_features.pt` | 18,130,473 | `969b1857be95dab78055e45cf4073e9cef4b824b5795279dfc71297f5ae006f7` | `4053a416f787a95435775d6015eb3d497de7d9adb083460d0158bd78f52559a5` |
| `combined_features.pt` | 88,417,833 | `52a2213de662385569d27a32ff82d9edc2de028f0cfa483a56abe1687cdb2999` | `5c3f37b406451eb10de4bf1b294c8fd018bb21e2685752182e8220dd0beee511` |
| `row_provenance.csv` | 557,049 | `fb69eb75ed18dbab43746b1eae2a832d511f3b455cceb335734e19a6da515e87` | — |
| `feature_manifest.json` | 7,941 | `1111f4bfd4ad1183896321c0121d4c92b830c5e327a58c2408ae19267292dae0` | — |
| `feature_manifest.json.sha256` | 88 | `54bfc5b6092aa0a6451c745336933cfbe9e70c1ba54df6a0302d5f9f13e1d0b2` | — |

Total retained file bytes: `177,402,321` (about 177.4 MB decimal). The
manifest binds the exact patient/slide, source commit, authorization, WSI,
coordinate artifacts, clean Omic source, ImageNet1K V2 checkpoint, and
implementation hashes.

## Runtime and timing

- Execution: `2026-08-19T01:12:52.254613+00:00` to
  `2026-08-19T01:15:43.683340+00:00`
- Total elapsed: `171.42871093199983` seconds
- Coordinate validation: `0.120990284` seconds
- Omic validation: `0.656650842` seconds
- WSI identity/hash validation: `5.46209107` seconds
- Pre-extraction synthetic HEALNet check: `0.641421918` seconds; peak allocated
  GPU memory `301,068,288` bytes
- Sequential ResNet extraction: `148.39458222` seconds
- 2x streaming/forward: `124.72181307` / `24.0247815323` seconds; peak
  `481,334,784` bytes
- 4x streaming/forward: `22.166922364` / `6.33044137955` seconds; peak
  `481,334,784` bytes
- Real-feature HEALNet check: `0.19592683` seconds; peak `301,068,288` bytes
- GPU: Tesla T4, compute capability 7.5, driver `580.173.02`
- Python `3.12.11`; PyTorch `2.8.0+cu128`; torchvision `0.23.0+cu128`;
  CUDA build `12.8`; cuDNN `91002`

## Provenance and stop boundary

- Source commit:
  `6e2d5511eab60aa0ab515ca6bb875d7c9fcc2190`
- Authorization SHA256:
  `2bfe61f7e17668ec50aeb6f3d2b214e0a3dd91e4f47692610880bec8afb2b080`
- Official HEALNet remained read-only at
  `28ba5da6ab99fd8069972c22e986d83edb658dd4`
- Frozen BLCA commit/tag remained
  `df7cf2bda783ab6cc09e95d6a1fa0914da05a433`
- The protected, user-modified
  `reports/blca_one_patient_multiscale_pilot.md` was left untouched
- Processed: one Q50 patient, 8,580 2x reads, 2,213 4x reads, and two HEALNet
  interface checks
- Training, backward passes, optimizer steps, Q75 operations, full-cohort
  operations, Google Drive operations, raw WSI deletions, and KIRP operations:
  zero

The required Q50 stop was reached. Separate review and explicit authorization
are required before training, Q75, or any full-cohort processing.
