# BRCA 894-patient singleton streaming design

Status: `CPU_DESIGN_VERIFIED__EXECUTION_NOT_AUTHORIZED`

## Outcome

The three pilots are sufficient to freeze the extraction interface, but not to
retain the full duplicate artifact layout for all 894 patients under the
Lightning 200 GB quota. The cohort must be processed at concurrency one with a
compact, restartable, patient-level transaction.

## Verified facts

- Exactly 894 patients have one exact WSI and one exact Omic row.
- Their declared raw WSI inventory is 918,532,189,383 bytes; individual files
  range from 25,565,706 to 3,401,266,790 bytes (median 974,485,171 bytes).
- The three pilots span 9,322–16,945 patches, 155.79–245.16 GPU seconds, and
  153,208,853–278,525,034 retained bytes in the six-file duplicate layout.
- Per-patient HEALNet input remains ragged `[1,P_i,2048]` plus three one-token
  Omic modalities. No cross-patient tensor concatenation is allowed.

## Estimates, not guarantees

Scaling the three observations to 894 patients gives approximately 38.69,
47.38, or 60.88 T4 GPU-hours at the observed minimum, mean, or maximum rates.
The same naive scaling predicts 136.97, 181.52, or 249.00 GB for the complete
duplicate artifact layout. These values are planning scenarios; actual patch
counts and I/O performance remain slide-specific.

The mean layout is too close to the 200 GB quota after the mandatory 20 GB
safety floor, while the upper scenario exceeds it. Therefore the executable
cohort design must retain a single canonical combined tensor plus provenance
and validation metadata, rather than permanent duplicate 2x, 4x, and combined
tensors. The exact compact schema and migration validator require a separate
review before cohort work.

## Patient transaction

Each patient must advance through exactly these states:

1. `PLANNED`
2. `ACQUISITION_AUTHORIZED`
3. `RAW_VERIFIED`
4. `HEADER_POLICY_VERIFIED`
5. `COORDINATES_VERIFIED`
6. `GPU_AUTHORIZED`
7. `FEATURES_VERIFIED`
8. `TERMINAL_RECORDED`

No state may be skipped. Each transition requires separately recorded authority
and verified evidence. The next patient cannot begin until the current patient
has a terminal record.

The eventual executor must use an exact one-row manifest, concurrency one,
atomic no-overwrite publication, content hashes, stable patient identity, and
an append-only ledger. Patch images are never retained. A failed transaction
stops the queue and preserves diagnostically useful state; it must not trigger
the next download or delete pre-existing data.

## Storage and recovery rules

- Enforce the 200,000,000,000-byte organization quota and preserve at least
  20,000,000,000 bytes unused.
- Retain at most one raw WSI in the active transaction.
- Recalculate projected feature bytes from actual coordinate counts before GPU
  authorization.
- Require enough space for raw input, staging, final compact artifacts, and the
  safety floor before every state transition.
- Never overwrite a completed patient directory. Resume only from a validated
  ledger state, not from inferred filenames.
- Raw deletion and any remote archival remain separately unauthorized. A
  future deletion policy must name the exact verified file and recovery source.

## Validation completed now

The accompanying pure-Python policy has no filesystem, OpenSlide, Torch, CUDA,
network, or training imports. Synthetic tests verify exact cohort constants,
concurrency-one behavior, ordered state transitions, fresh-authorization
requirements, terminal stopping, and the three-pilot estimates.

## Next decision

Review and approve or reject a compact patient artifact schema and its failure-
recovery ledger. Even after that design review, actual cohort acquisition,
pixel access, GPU extraction, deletion, archival, and training remain separate
execution authorizations.
