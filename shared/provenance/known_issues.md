# Known provenance and implementation issues

1. The paper states that patch features use a ResNet50 pretrained on Kather100K.
2. The camera-ready released code uses torchvision ResNet50 with `IMAGENET1K_V2` weights.
3. The paper describes patch extraction at approximately 0.5 and 1.0 micrometres per pixel.
4. The released code selects integer OpenSlide pyramid levels and does not implement an MPP-based level-selection or resampling policy.
5. Camera-ready feature extraction allocates one cohort-wide tensor outside the slide loop and reuses it. For a shorter slide, trailing rows may retain features from a previously processed longer slide.
6. The exact CLAM commit used by the authors is not proven. Commit `26e0b6c4873e112f1ccd74cd834894c4ab7a2934` is a historically aligned compatibility pin for this pilot.

These issues must not be silently reconciled, and no fix should be applied directly to the official HEALNet repository.
