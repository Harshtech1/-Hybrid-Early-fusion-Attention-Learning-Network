# P0001 production GPU execution approval

The exact P0001 GPU authorization has been recorded and bound to the frozen
16,816-coordinate feature contract. The Tesla T4 environment was detected,
but no GPU process, WSI open, patch read, feature extraction, HEALNet run,
artifact publication, or training operation was started.

The executable runner was hardened before use. It now holds the WSI, Omic
archive, and ResNet50 checkpoint with no-follow file descriptors; supplies
the WSI to OpenSlide through a stable `/proc/self/fd` path; uses zero loader
workers; counts and verifies exactly 13,372 level-0 and 3,444 level-1 patch
reads; closes each OpenSlide instance; rehashes the held WSI after extraction;
and publishes the compact four-file artifact without overwrite.

## Current stop

Recovery-v2 requires a valid append-only chain beginning with ten events for
the five already completed stages through `COORDINATES_VERIFIED`. That ledger
does not yet exist. The user's GPU authorization explicitly names only the
new `GPU_AUTHORIZED` and `FEATURES_VERIFIED` stages, so the runner correctly
refuses to invent the earlier records.

One short CPU-only authorization is therefore required to create and validate
that exact prior-stage prefix from frozen committed evidence. No GPU is needed
for that step, no raw file is deleted, and no new patient is processed. After
the prefix is committed and validated, the already authorized P0001 GPU run
can proceed in one continuous Tesla T4 window.

Validation completed: 14 focused tests and 809 full CPU tests passed. The
feature and recovery output paths remain absent.
