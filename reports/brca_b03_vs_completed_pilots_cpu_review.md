# B03 versus completed BRCA engineering pilots

Status: **CPU-only comparison complete**. All values below are measured values from frozen provenance; no WSI, pixel, GPU, model, or artifact mutation occurred during this review.

| Pilot | WSI patches | GPU pilot time | Seconds/1,000 patches | Retained artifacts | Bytes/patch | Peak GPU memory |
|---|---:|---:|---:|---:|---:|---:|
| Q25 | 9,322 | 155.79 s | 16.71 | 153.21 MB | 16,435 | 481.3 MB |
| Q50 | 10,793 | 171.43 s | 15.88 | 177.40 MB | 16,437 | 481.3 MB |
| Q75 | 16,945 | 245.16 s | 14.47 | 278.53 MB | 16,437 | 481.9 MB |
| B01 | 4,742 | 73.46 s | 15.49 | 39.09 MB | 8,244 | 481.3 MB |
| B02 | 9,020 | 124.94 s | 13.85 | 74.34 MB | 8,242 | 484.0 MB |
| **B03** | **11,132** | **118.41 s** | **10.64** | **91.77 MB** | **8,244** | **484.0 MB** |

B03 exactly produced the expected `[11132,2048]` float32 feature matrix and natural HEALNet WSI input `[1,11132,2048]`. Synthetic and real-feature numerical smokes passed with finite `[1,4]` output. As with every pilot, these random-initialization interface tests are not trained survival predictions.

The compact B01–B03 layout retains about 8.24 KB per patch, compared with about 16.44 KB per patch in the older Q25–Q75 duplicate layout. The observed reduction is approximately one half because the compact schema stores one canonical combined tensor instead of permanent 2×, 4×, and combined copies.

B03 completed faster than its conservative 160–200 second estimate and slightly faster than B02 despite more patches. Patch count is therefore an important capacity signal but not a sufficient wall-time predictor; slide encoding, storage I/O, worker scheduling, and tissue distribution also matter. Peak allocation remained approximately 484 MB, confirming that streaming I/O and preprocessing—not T4 memory capacity—remain the principal engineering constraint.

All six pilots use the user-selected ImageNet1K V2 ResNet50 engineering checkpoint. This gives consistent internal comparisons, but comparative reporting must continue to disclose that the paper states Kather100K pretraining.

The B01, B02, and B03 compact artifact sets were independently reopened and validated during this review. The recommended next step is the guarded B06 acquisition and header-only inspection. B06 should reach GPU work only if its high-size-tail pyramid/MPP geometry or expected patch burden adds material coverage.
