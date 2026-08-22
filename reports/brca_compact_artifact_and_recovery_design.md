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
- partial/unexpected ledger files; andkYou are continuing the HEALNet BRCA project from the validated three-patient pilot.

## Current authoritative status

The engineering pilot is complete:

* BRCA alignment: 894 exact singleton patients.
* Q25, Q50, and Q75 pilot WSIs: downloaded, identity-verified, coordinate-validated, ResNet50-feature-extracted, and four-modality HEALNet numerical smoke-tested.
* Natural HEALNet WSI representation is frozen as:

  * per-patient combined WSI features: `[P_i, 2048]`
  * model input: `[1, P_i, 2048]`
* Two WSI branches remain:

  * `scale_2x`
  * `scale_4x`
* Row order is fixed:

  1. all `scale_2x` rows;
  2. all `scale_4x` rows.
* RNA, mutation, and CNV remain separate modalities.
* Multi-WSI BRCA patients remain excluded initially.
* ImageNet1K V2 ResNet50 is frozen.
* No training has started.
* Current repository suite: 635 tests passing.
* The CPU-only 894-patient streaming state machine has already been designed.
* Do not modify official HEALNet.
* Preserve the existing unrelated BLCA report modification.
* Do not start any real cohort download or feature extraction in this task.
* GPU is not required for this task.

## Objective

Design and implement the **generic compact retained-artifact schema and append-only restart/recovery ledger** required before cohort-scale extraction.

This must generalize the validated Q25/Q50/Q75 artifacts. Do not create more patient-specific source files.

The final design must support all 894 BRCA singleton patients and one-patient-at-a-time streaming.

## Required retained patient artifact

For each successfully processed patient, retain one canonical compact package whose essential WSI content is:

```text
combined_features: [P_i, 2048] float32
```

Do not require permanent storage of separate duplicate 2× and 4× feature tensors if the same information is already represented losslessly in the canonical combined tensor.

The package must preserve enough metadata to reconstruct branch identity exactly.

At minimum preserve:

```text
case_id
slide_id
gdc_uuid

p2_count
p4_count
total_count

scale_2x row range: [0, p2_count)
scale_4x row range: [p2_count, p2_count + p4_count)

branch coordinate artifact hashes
combined feature tensor hash
ResNet50 checkpoint identity
preprocessing/source commit identity
scale-policy version
coordinate-policy version
omic identity
artifact schema version
```

The combined tensor must satisfy:

```text
shape == [p2_count + p4_count, 2048]
dtype == float32
all values finite
```

## Compact storage design

Implement a generic schema that minimizes duplicate retained data.

Prefer a patient artifact directory conceptually like:

```text
patient_artifact/
├── combined_features.pt
├── row_provenance.csv or a more compact equivalent
├── artifact_manifest.json
└── artifact_manifest.json.sha256
```

If row-level provenance can be represented more compactly without losing reproducibility, propose and test that alternative.

Do not remove information required to map any feature row back to:

```text
patient
slide
branch
branch-local patch index
coordinate row
physical-scale policy
```

Evaluate whether storing all repeated branch labels per row is necessary, or whether branch boundaries plus coordinate artifacts are sufficient.

## Atomic publication

Implement a no-overwrite publication transaction for the complete patient artifact package.

Requirements:

1. Final destination must not exist.
2. Write to a unique staging location on the same filesystem.
3. Validate every staged file before publication.
4. Verify tensor shape/dtype/finiteness.
5. Verify row counts and branch boundaries.
6. Verify hashes.
7. Publish atomically/fail-closed.
8. Never replace an existing final patient artifact.
9. Leave no stale transaction lock/staging directory after success.
10. Preserve failed staging information only when required for diagnosis; define the exact rule.
11. Never delete unrelated or pre-existing data.

## Append-only recovery ledger

Design a generic append-only cohort ledger.

The ledger must support safe restart after:

```text
download failure
hash mismatch
metadata rejection
coordinate failure
GPU extraction failure
artifact validation failure
unexpected process interruption
successful completion
```

Do not use one mutable status row that gets overwritten repeatedly.

Use an append-only event model.

Each event should include at minimum:

```text
timestamp_utc
run_id
case_id
slide_id
gdc_uuid
stage
event
status
attempt_number
source_commit
policy_version
input_hashes
output_hashes if available
error_class if applicable
error_message if applicable
host/runtime identity where useful
```

Define valid stages, for example:

```text
AUTHORIZED
DOWNLOAD_STARTED
DOWNLOAD_VERIFIED
METADATA_VERIFIED
COORDINATES_VERIFIED
GPU_EXTRACTION_STARTED
FEATURES_VERIFIED
PATIENT_COMPLETE
FAILED
```

But design the actual state machine carefully rather than blindly using these names.

The system must be able to derive the latest safe patient state purely by replaying the append-only ledger.

## Restart semantics

Specify and implement deterministic restart behavior.

Examples:

* `PATIENT_COMPLETE`:

  * never reprocess automatically.
* verified download but no coordinates:

  * resume from metadata/coordinate stage without re-downloading if identity still verifies.
* verified coordinates but no features:

  * resume at GPU extraction.
* failed feature artifact validation:

  * never treat the patient as complete.
* an existing final artifact whose manifest/hash does not match the ledger:

  * fail closed and require manual review.
* stale temporary staging:

  * detect and classify; do not silently overwrite.

Implement these rules in reusable code.

## Storage lifecycle model

Do not perform deletion in this task.

Design the future lifecycle states for:

```text
raw WSI
coordinates
combined features
manifests/provenance
```

Assume:

* Lightning persistent storage is limited;
* only one raw WSI should normally be active;
* canonical combined features are retained;
* coordinate artifacts are retained at least until cohort verification is frozen;
* raw WSI deletion must never happen merely because feature extraction returned successfully.

Define the exact future condition under which a raw WSI could be marked:

```text
SAFE_TO_DELETE
```

That condition should require at minimum:

* WSI identity verified;
* coordinates verified;
* features verified;
* final artifact package hash verified;
* ledger contains successful completion;
* any required backup/retention policy satisfied.

Do not actually delete anything.

## Generic cohort code

Create reusable generic modules. Do not add:

```text
brca_patient001_...
brca_patient002_...
```

or more Q-specific implementations.

Prefer a structure similar to:

```text
multiscale_feature_pilot/src/
    brca_patient_artifact.py
    brca_recovery_ledger.py
    brca_cohort_state.py
```

and generic tests.

Exact names may differ if a cleaner design is justified.

## Tests

Add comprehensive synthetic tests for:

### Artifact schema

* valid `[P,2048]`
* wrong width rejection
* wrong dtype rejection
* NaN/Inf rejection
* p2+p4 mismatch
* incorrect row boundary
* corrupted manifest
* corrupted feature hash
* pre-existing final destination
* interrupted staging
* no-overwrite semantics
* deterministic manifest serialization

### Ledger

* append-only behavior
* monotonic attempt numbers where appropriate
* invalid state transition rejection
* replay from empty state
* replay after interrupted run
* replay after failure and retry
* completed patient is terminal for automatic processing
* duplicate event handling
* malformed/corrupted ledger entry
* hash inconsistency detection
* manual-review state

### Restart state machine

* download verified → coordinates resume
* coordinates verified → GPU resume
* feature failure → retry GPU only
* completed → skip
* conflicting final artifact → block
* stale staging → block or explicitly recover according to policy

## Review constraints

Before committing:

* run the full repository suite;
* run focused new tests;
* run `git diff --check`;
* independently review the state-transition logic;
* verify no `.svs`, `.h5`, `.pt`, `.pth`, `.npy`, feature payload, coordinate payload, checkpoint, credential, or runtime artifact enters Git;
* verify official HEALNet is unchanged;
* verify Q25/Q50/Q75 retained artifacts are unchanged;
* verify the protected BLCA report modification remains untouched.

## Deliverables

Produce:

1. Generic compact artifact implementation.
2. Generic append-only ledger implementation.
3. Restart/state-replay implementation.
4. Synthetic tests.
5. A report:
   `reports/brca_cohort_artifact_and_recovery_design.md`
6. A provenance/config record describing:

   * schema version;
   * state-machine version;
   * artifact-retention policy;
   * restart semantics;
   * storage assumptions.
7. One selective source-only commit.

## Required final report

Return:

```text
STATUS
source commit
files added/modified
focused test result
full test-suite result
artifact schema summary
estimated retained bytes per patient from Q25/Q50/Q75 evidence
estimated 894-patient retained storage range
ledger event schema
state-transition summary
restart examples
future SAFE_TO_DELETE conditions
Git status
confirmation that no real cohort work occurred
confirmation that GPU was not used
confirmation that training did not occur
```

## Hard stop

Do not:

* download another WSI;
* process a real WSI;
* generate real coordinates;
* run ResNet50;
* run HEALNet;
* use GPU;
* delete any WSI;
* configure Google Drive;
* start the 894-patient loop;
* create train/validation/test splits;
* train any model.

Stop after the CPU-only generic compact-artifact/recovery-ledger implementation, tests, review, and source-only commit.

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
