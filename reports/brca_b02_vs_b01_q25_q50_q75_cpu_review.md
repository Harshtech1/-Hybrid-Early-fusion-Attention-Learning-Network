# CPU review: compact B02 versus B01, Q25, Q50, and Q75

## Conclusion

The completed B02 pilot independently validates the compact four-file artifact policy at a larger token load than B01 without changing the scientific tensor or HEALNet interface contract. B01 and B02 retain one canonical combined feature tensor plus row provenance; Q25, Q50, and Q75 used the earlier six-file layout that retained both branch tensors and the combined tensor.

## Frozen measured comparison

| Pilot | Raw WSI | Level-0 dimensions | Tokens 2× / 4× / total | ResNet extraction | Complete GPU run | Peak GPU memory | Retained features | Layout |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| B02 | 724.11 MB | 89,291 × 72,971 | 7,158 / 1,862 / 9,020 | 113.16 s | 124.94 s | 483.96 MB | 74.34 MB | compact, 4 files |
| B01 | 408.70 MB | 63,784 × 39,311 | 3,773 / 969 / 4,742 | 66.25 s | 73.46 s | 481.33 MB | 39.09 MB | compact, 4 files |
| Q25 | 648.05 MB | 65,736 × 67,406 | 7,404 / 1,918 / 9,322 | 134.69 s | 155.79 s | 481.33 MB | 153.21 MB | legacy pilot, 6 files |
| Q50 | 975.63 MB | 99,960 × 65,334 | 8,580 / 2,213 / 10,793 | 146.89 s | 171.43 s | 481.33 MB | 177.40 MB | legacy pilot, 6 files |
| Q75 | 1,360.74 MB | 108,528 × 90,471 | 13,487 / 3,458 / 16,945 | 214.40 s | 245.16 s | 481.86 MB | 278.53 MB | legacy pilot, 6 files |

All values are measurements from frozen provenance. B02's extraction time is the sum of its measured 2× and 4× branch times.

## Compact-storage result

B02's canonical float32 tensor contains `9020 × 2048 × 4 = 73,891,840` raw bytes. Its complete four-file artifact occupies 74,342,370 bytes, or 8,241.95 bytes per feature row. B01 gives essentially the same compact rate at 8,243.80 bytes per row. The three legacy pilots occupy 16,435.19–16,437.00 bytes per row because their separate branch tensors duplicate rows already present in the combined tensor. At a matched row count, B02 therefore demonstrates a measured 49.85–49.86% storage reduction relative to the legacy layout.

Read-only validation reloaded both compact artifacts on CPU and verified B02 `[9020,2048]` with natural model shape `[1,9020,2048]`, B01 `[4742,2048]`, branch order, row provenance, and content hashes. CUDA remained uninitialized.

## Model contract and engineering interpretation

For each patient, `P = P_2x + P_4x` and

`X_WSI = concat(X_2x, X_4x, dim=0) ∈ R^(P×2048)`.

HEALNet receives WSI `[1,P,2048]`, RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]`. All five pilots produced finite output `[1,4]` with matching attention lengths. These remain random-initialized numerical/interface tests, not trained survival predictions.

B02 confirms that compact publication is stable beyond the smaller B01 bag. Runtime continues to track tissue-selected token load and slide-specific I/O, while peak T4 allocation remains about 482–484 MB. This supports continuing the staged validation batch, but does not authorize B03 acquisition or any later pixel/model operation.

No B03 WSI was downloaded or opened during this review. No pixel, coordinate, GPU, feature, deletion, Drive, cohort-expansion, or training operation occurred.
