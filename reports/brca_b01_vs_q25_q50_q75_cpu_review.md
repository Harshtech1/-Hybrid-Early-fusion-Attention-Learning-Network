# CPU review: compact B01 versus Q25, Q50, and Q75 pilots

## Conclusion

The B01 pilot validates the intended compact artifact policy without changing the scientific tensor contract. All four pilots use two physical WSI scales, float32 ResNet50 ImageNet1K V2 features, natural WSI layout `[1,P,2048]`, three separate Omic modalities, and finite random-weight HEALNet interface smokes. B01 additionally demonstrates that retaining one canonical combined tensor and row provenance avoids the duplicated branch tensors in the earlier six-file pilot layout.

## Frozen measured comparison

| Pilot | Raw WSI | Level-0 dimensions | Tokens 2× / 4× / total | ResNet extraction | Complete GPU run | Peak GPU memory | Retained features | Layout |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B01 | 408.70 MB | 63,784 × 39,311 | 3,773 / 969 / 4,742 | 66.25 s | 73.46 s | 481.33 MB | 39.09 MB | compact, 4 files |
| Q25 | 648.05 MB | 65,736 × 67,406 | 7,404 / 1,918 / 9,322 | 134.69 s | 155.79 s | 481.33 MB | 153.21 MB | legacy pilot, 6 files |
| Q50 | 975.63 MB | 99,960 × 65,334 | 8,580 / 2,213 / 10,793 | 146.89 s | 171.43 s | 481.33 MB | 177.40 MB | legacy pilot, 6 files |
| Q75 | 1,360.74 MB | 108,528 × 90,471 | 13,487 / 3,458 / 16,945 | 214.40 s | 245.16 s | 481.86 MB | 278.53 MB | legacy pilot, 6 files |

Times and storage are measured values from frozen provenance. B01's extraction time is the sum of its measured 2× and 4× branch times. Q25's comparable extraction value is the sum of its branch streaming times; Q50 and Q75 also record a near-equivalent sequential extraction timer.

The earlier layout stored `scale_2x_features.pt`, `scale_4x_features.pt`, and `combined_features.pt`; therefore its feature rows were effectively stored twice. B01 stores only `combined_features.pt`, `row_provenance.csv`, the manifest, and its sidecar. B01's raw float32 tensor payload is `4742 × 2048 × 4 = 38,846,464` bytes, closely matching the measured 38,848,041-byte tensor file and 39,092,076-byte complete artifact set.

## Tensor and model contract

For each patient, let `P = P_2x + P_4x`. The retained WSI matrix is

`X_WSI = concat(X_2x, X_4x, dim=0) ∈ R^(P×2048)`.

HEALNet receives `[1,P,2048]`, RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]`. Every pilot produced finite output `[1,4]`; WSI attention was `[1,2,P]`, and each Omic attention tensor was `[1,2,1]`. These are numerical/interface checks using random initialized weights, not trained survival predictions.

## Engineering interpretation

- Runtime increases with tissue-selected patch count; B01 is faster because it has the smallest token bag.
- Peak T4 allocation remains approximately 481–482 MB across all four pilots, so streaming I/O and preprocessing—not GPU capacity—remain the dominant engineering constraint.
- B01 confirms the compact artifact is compatible with the established natural row order and HEALNet interface.
- One patient is not sufficient to approve cohort execution. B02 should proceed only through acquisition and header metadata next, followed by a separate geometry/pixel decision.

No WSI was downloaded or opened during this review. No GPU, coordinate, feature, deletion, Drive, cohort, or training operation occurred.
