# BRCA compact feature artifact and recovery ledger

Status: `CPU_IMPLEMENTED_AND_SYNTHETICALLY_VERIFIED__EXECUTION_NOT_AUTHORIZED`

## Decision

The six-file pilot format is retained as immutable pilot evidence, but it is
not the cohort retention format. Each pilot stores the 2x tensor, 4x tensor,
and their exact concatenation, which permanently duplicates every feature row.
For the 894-patient singleton pipeline, the approved design candidate retains
one canonical combined tensor and preserves branch boundaries in the manifest
and row provenance.

No existing Q25, Q50, or Q75 artifact was migrated or modified while preparing
this design.

## Exact compact set

Each completed patient directory contains exactly:

| File | Purpose |
|---|---|
| `combined_features.pt` | canonical CPU float32 `[P_i,2048]` tensor |
| `row_provenance.csv` | one ordered coordinate/branch record per tensor row |
| `compact_manifest.json` | identity, source hashes, branch ranges, tensor hashes, layout |
| `compact_manifest.json.sha256` | external canonical manifest anchor |

The manifest records the half-open row ranges

\[
R_{2x}=[0,P_i^{(2)}), \qquad
R_{4x}=[P_i^{(2)},P_i^{(2)}+P_i^{(4)}).
\]

Therefore the original branch tensors can be viewed without duplication:

\[
F_i^{(2)} = F_i[R_{2x}], \qquad F_i^{(4)} = F_i[R_{4x}].
\]

The validator requires a contiguous finite CPU float32 tensor, exact 2x-prefix
and 4x-suffix provenance, contiguous global/local indices, file SHA256,
semantic tensor-content SHA256, source hashes, and no unexpected files.

## Atomic publication

Publication writes only to a newly created sibling staging directory. The
complete staged set is validated before a Linux `RENAME_NOREPLACE` operation
commits the directory. An existing destination is never overwritten, merged,
or inferred as resumable. On failure, only the publisher-created staging
directory may be removed; pre-existing data are never deleted.

## Recovery ledger

Each patient has an independent ledger directory containing immutable files:

```text
00000001.json
00000002.json
...
00000008.json
```

Every record contains the exact patient and slide, stage, evidence SHA256,
authorization SHA256, previous-record SHA256, timestamp, and its own canonical
record SHA256. The first record points to a 64-zero genesis hash. Subsequent
records form a hash chain and must follow the frozen state order from `PLANNED`
through `TERMINAL_RECORDED` without skipping.

Records use exclusive file creation and filesystem synchronization. A partial,
unexpected, missing-sequence, altered, or identity-drifted record blocks
resume. The next patient cannot be considered until the current patient's
ledger validates through its terminal record.

## Storage projection

Using the actual combined tensor, provenance, manifest, and sidecar sizes from
the three pilots gives:

| Scenario | Compact bytes per pilot/patient | Projected 894-patient bytes |
|---|---:|---:|
| Q25 observed rate | 76,839,875 | 68,694,848,250 |
| Mean observed rate | 101,843,742 | 91,048,305,348 |
| Q75 observed rate | 139,708,440 | 124,899,345,360 |

These are workload scenarios, not capacity guarantees. Before every real
patient transition, the policy requires available bytes to cover the active
raw WSI, transaction staging, final artifact, and the 20 GB safety floor.
Low-space conditions fail before the transition.

## CPU validation performed

Synthetic tests cover:

- exact four-file publication and reload;
- combined-tensor equality and branch ranges;
- nonfinite and wrong-dtype rejection;
- provenance branch-boundary rejection;
- existing-destination preservation;
- tensor and manifest corruption detection;
- complete eight-stage ledger round-trip;
- stage skipping and patient-identity drift;
- hash-chain tampering;
- partial/unexpected ledger files; and
- exact-boundary and insufficient-storage behavior.

The implementation contains no WSI, OpenSlide, CUDA, model, network, Drive, or
training interface. It accepts already-produced CPU features only.

## Next gate

The next safe task is a CPU-only migration rehearsal using synthetic artifacts
and, only if separately authorized, read-only validation of the already
published Q25/Q50/Q75 pilot feature sets. No cohort acquisition or extraction
is authorized by this design. After migration compatibility is reviewed, an
exact small-batch cohort proposal should be approved before any 894-patient
execution.
