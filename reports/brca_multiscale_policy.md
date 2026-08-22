# BRCA multiscale extraction policy

Status: `Q25_Q50_Q75_VERIFIED__EXTRACTION_POLICY_READY_TO_FREEZE`

The Q25, Q50, and Q75 BRCA pilots have now completed coordinate generation,
ResNet50 feature extraction, and four-modality HEALNet numerical smoke tests.
This replaces the earlier pre-pilot policy gate. It does not authorize cohort
processing or training.

## Frozen extraction contract

- Match one exact singleton `(case_id, slide_id)` pair; never join by row order.
- Validate UUID, filename, size, MD5/SHA256, OpenSlide header, coordinates,
  Omic row, and checkpoint before processing.
- Target approximately `0.5 µm/px` for `scale_2x` and `1.0 µm/px` for
  `scale_4x`, evaluating physical MPP independently from coordinate geometry.
- For the verified approximately `0.25 µm/px` Aperio slides, `scale_2x` reads
  a level-0 `512×512` footprint and resizes to `256×256` with PIL Lanczos.
- `scale_4x` uses the native level near `1.0 µm/px` as a `256×256` read with
  no resampling. Silent level substitution remains prohibited.
- Encode every patch with classifier-removed torchvision
  `ResNet50_Weights.IMAGENET1K_V2`, producing finite float32 `[P,2048]` rows.
- Preserve 2x rows first and 4x rows second, then use
  `torch.cat([features_2x, features_4x], dim=0)` with no pooling or transpose.
- Supply WSI `[1,P_i,2048]`, RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV
  `[1,1,1333]` as four separate HEALNet modalities.
- Initial batch size is one, with no padding and no attention mask.

## Three-pilot evidence

| Pilot | WSI bytes | 2x patches | 4x patches | Total | GPU runtime | Artifacts |
|---|---:|---:|---:|---:|---:|---:|
| Q25 | 648,046,947 | 7,404 | 1,918 | 9,322 | 155.79 s | 153,208,853 B |
| Q50 | 975,626,387 | 8,580 | 2,213 | 10,793 | 171.43 s | 177,402,321 B |
| Q75 | 1,360,743,825 | 13,487 | 3,458 | 16,945 | 245.16 s | 278,525,034 B |

All three pilots produced finite `[1,4]` smoke outputs with WSI attention
length equal to the slide-specific patch count. These were random, untrained
interface checks and are not survival predictions.

The pilot results support freezing the engineering extraction interface across
the observed geometry and workload range. They do not prove that every BRCA
slide has compatible native metadata. Every future slide must still pass its
own header and physical-scale gate.

## Cohort representation and boundary

The 894 singleton patients form a ragged collection `{F_i}`. Patient bags must
not be concatenated across identities. Multi-WSI cases remain excluded, and
future data splits must be grouped by patient.

The next authorized work is CPU-only design and synthetic validation of a
one-patient-at-a-time streaming state machine. Downloads, WSI access,
coordinate generation, CUDA, feature extraction, deletion, Drive operations,
full-cohort processing, and training require later explicit gates.
