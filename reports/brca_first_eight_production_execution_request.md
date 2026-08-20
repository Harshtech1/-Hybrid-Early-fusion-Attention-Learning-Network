# First BRCA production block — execution request

Status: **ready for execution review; no live stage is authorized**.

Rows 1–8 of the frozen 894-patient order are now classified as the first
production block, not an additional scientific pilot. The block contains eight
exact singleton patients and 8,297,129,620 declared raw WSI bytes. Each patient
has a byte-exact request-only one-row GDC manifest.

Execution remains strictly serial: one patient, one download, and one GPU job
at a time. A patient must traverse the eight recovery-v2 stages and reach a
validated terminal record before the next cohort index may start. Any identity,
hash, authorization, artifact, or ledger mismatch stops the entire block.

The production control plane is reusable, but header dimensions and patch
counts do not exist before each real slide reaches its corresponding stage.
Therefore acquisition/header access, the one bounded mask read, and GPU feature
extraction still require stage-specific execution authorization. Raw deletion
is prohibited for this block; all eight raw slides are retained through formal
block review.

This package performs no download, WSI open, pixel read, coordinate generation,
CUDA work, feature extraction, HEALNet execution, deletion, Drive operation, or
training.
