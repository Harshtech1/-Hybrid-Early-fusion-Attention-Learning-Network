# B02 GPU pilot CPU preflight

Status: **ready; waiting for a GPU machine**.

All CPU-side requirements have passed. The runner is committed and bound to the exact B02 WSI, verified coordinate manifest, 7,158 2× rows, 1,862 4× rows, Omic row 472, ImageNet1K V2 ResNet50 checkpoint, and compact artifact schema. The output destination is absent.

The expected feature matrix is `[9020,2048]`, exposed to HEALNet only as `[1,9020,2048]` with RNA `[1,1,1558]`, mutation `[1,1,21]`, and CNV `[1,1,1333]`. Publication will retain one canonical combined tensor plus row provenance, manifest, and sidecar. Expected storage is approximately 74.5 MB.

Focused checks passed 14/14 and the complete repository suite passed 660/660. The real CPU preflight revalidated the WSI hashes/header, coordinate artifacts, Omic identity, checkpoint, source commit, official HEALNet checkout, protected Git status, and absent output without reading a patch or initializing CUDA.

The current CPU machine cannot execute the authorized feature extraction. Switch the Studio to a Tesla T4, then run the already-authorized committed gate. Estimated GPU wall time is approximately 2–3 minutes. No additional scientific authorization is needed unless the execution scope changes.
