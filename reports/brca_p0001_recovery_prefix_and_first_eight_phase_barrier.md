# P0001 recovery prefix and first-eight CPU phase barrier

Status: **P0001 ledger prefix published and independently verified; first-eight CPU phase barrier active**.

## P0001 recovery prefix

The package derives the historical recovery-v2 chain from frozen committed
P0001 evidence. It does not infer or re-run any scientific operation. The
derived chain contains exactly two immutable records for each completed stage:

| Sequence | Completed stage | Records |
|---:|---|---:|
| 1–2 | `PLANNED` | start + success |
| 3–4 | `ACQUISITION_AUTHORIZED` | start + success |
| 5–6 | `RAW_VERIFIED` | start + success |
| 7–8 | `HEADER_POLICY_VERIFIED` | start + success |
| 9–10 | `COORDINATES_VERIFIED` | start + success |

Every ledger record uses the explicit backfill recording timestamp
`2026-08-20T23:40:00Z`. This timestamp documents when the historical chain was
recorded; it does not represent when acquisition, header inspection, policy
review, or coordinate execution occurred. Their original operational times
remain authoritative in the frozen evidence.

The deterministic backfill tip is
`45077a18d10205a04c9ef3a80479be7af4079501a11891ca1b9cc8701a8347b6`.
Replay reports `ADVANCE_STAGE`, with `GPU_AUTHORIZED` as the only next stage.

The runner requires the final committed Git identity, binds its own committed
bytes plus the complete P0001 recovery construction import closure, re-hashes every frozen
evidence file, requires the external destination to be absent, constructs all
ten events in a sibling staging directory using the recovery-v2 append-only
primitive, validates replay, and publishes the complete directory with Linux
`RENAME_NOREPLACE`, and fsyncs the publication parent. All path ancestors and
the destination parent must be non-symlink directories. Existing paths are
never overwritten or deleted. An
interrupted staging directory is retained for manual review.

## First-eight CPU phase barrier

The scheduling amendment keeps the exact frozen cohort rows, manifests,
identity bindings, singleton transactions, and stage authorization rules. It
changes only the CPU preparation schedule:

- downloads remain serial (`concurrency = 1`);
- at most two distinct patients may occupy CPU acquisition/header/policy work;
- one download may overlap one other patient's header verification or policy
  design;
- each patient's own stages remain ordered;
- any identity, hash, authorization, or stage drift stops the block;
- pixel reads, coordinate generation, CUDA, features, HEALNet, deletion,
  Drive, cohort expansion, and training remain locked.

The policy binds all eight request-only manifests and exactly
8,297,129,620 raw bytes, of which P0002–P0008 contribute 6,527,281,524 bytes.

## Validation

Seven focused tests passed, and 41 combined bootstrap, recovery-v2, and P0001
GPU-preexecution tests passed. They covered deterministic ten-event derivation,
replay-tip validation, atomic complete-directory publication in a temporary
fixture, no-overwrite rejection, exact eight-manifest binding, CPU worker
limits, serial download enforcement, and rejection of pixel-stage scheduling.
Symlink-parent publication was also rejected without creating a ledger.
Python compilation passed. The package-preparation tests did not create the
real external ledger.

## P0001 execution result

The committed bootstrap runner executed once on CPU at source commit
`228aaa62fee2f04a9afe09b519bb4ebc7662402b`. It atomically published exactly
10 event files to the external recovery-v2 directory. Independent read-only
validation matched the complete frozen derived prefix, verified the hash chain,
and reproduced replay action `ADVANCE_STAGE`, durable stage
`COORDINATES_VERIFIED`, next stage `GPU_AUTHORIZED`, and tip
`45077a18d10205a04c9ef3a80479be7af4079501a11891ca1b9cc8701a8347b6`.
No bootstrap staging path remained. No WSI, pixel, coordinate, CUDA, feature,
HEALNet, deletion, Drive, or training operation was performed.

## Required execution order

1. Validate the seven P0002–P0008 serial acquisition/header results.
2. Freeze their exact scale and coordinate policies from metadata only.
3. Stop before any coordinate mask read and request the combined exact
   coordinate-execution authorization.
