# BRCA raw-WSI lifecycle proposal

Status: **proposal only; no deletion is authorized**.

The first eight frozen cohort rows are the first production block, not another
scientific pilot. Their raw WSIs should remain local until the block-level
review is complete. The already downloaded B06 WSI remains retained exactly as
directed and is outside any proposed release action.

Keeping all 894 raw slides locally is infeasible: the frozen inventory is
918,532,189,383 bytes, well above the approximately 200 GB workspace quota.
The production route therefore needs a one-patient-at-a-time lifecycle, but a
raw slide must never be released merely because a GPU call returned.

## Required release gate

A future patient-specific release may be considered only after all of the
following pass:

1. The exact cohort index, patient, full slide, UUID, byte size, MD5, and SHA256
   are revalidated.
2. The exact Omic row is rematched.
3. Coordinate and compact feature artifacts are independently reopened and
   strictly validated, including tensor and row-provenance hashes.
4. `FEATURES_VERIFIED` is terminally recorded in the recovery-v2 ledger with no
   pending, failed, ambiguous, or stranded transaction.
5. A verified recovery source or separately approved retained copy exists.
6. This lifecycle policy hash is bound to the terminal record.
7. A new authorization names that one exact raw WSI and permits its release.

Any mismatch means **retain and stop**. Glob deletion, batch deletion,
automatic post-feature cleanup, B06 deletion, and cleanup of pre-existing data
remain prohibited. The future release tool itself still needs a separate
implementation review and authorization.
