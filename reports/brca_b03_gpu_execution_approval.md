# B03 GPU feature-pilot execution approval

Direct user authorization is recorded for the exact B03-only deterministic Tesla T4 feature pilot.

The authorized execution is restricted to 8,875 scale-2× and 2,257 scale-4× patch reads from the already verified B03 coordinate bags, float32 ResNet50 ImageNet1K V2 inference, natural concatenation into `[11132,2048]`, compact atomic publication, and synthetic plus real-feature four-modality HEALNet numerical smoke tests.

The runner pins the authorization file SHA256 `30e440b93c202ec1d445312ef27dd21b2bc8a6678913f9ecc264581d83be2fac`, coordinate manifest `cced78f863415ee12d57b905131e60758ba849645196c0a14309e9ab8d4e1ae5`, and checkpoint `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`. Execution must use a committed source snapshot and pass exact identity, repository, T4, deterministic-environment, Omic, coordinate, header, checkpoint, and no-overwrite checks before extraction.

Training, backward passes, optimizer steps, AMP, TF32, CPU fallback, coordinate regeneration, B04–B06 processing, Q25/Q50/Q75/B01/B02 or BLCA changes, Drive access, deletion, cohort expansion, and official HEALNet modification remain prohibited.

At approval recording time no WSI was opened, CUDA work was run, or feature artifact was created. The next action is to commit this exact executable package, repeat the read-only preflight against that commit, and run the authorized B03 pilot once.
