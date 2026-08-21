# P0002–P0008 coordinate phase pre-execution review

Status: **authorized package validated; execution waits for its committed source identity**.

The phase binds the exact seven frozen header results, policies, policy reviews,
and level-2 mask tuples. The request remains immutable and non-executable; the
separate direct-user authorization activates only this coordinate phase.

The scheduler admits at most two patients. It initially admits P0002 and P0003,
then admits a later row only after an active row succeeds. If any observed
future fails, no replacement patient is admitted. Already published patient
artifacts are never rolled back or overwritten.

Each patient uses one held `O_NOFOLLOW` descriptor, verifies WSI MD5 and SHA256,
opens one OpenSlide instance through `/proc/self/fd`, rechecks the complete
header, and performs exactly one authorized level-2 RGBA read. It hashes the
mask, rechecks the held WSI hashes and path identity, segments tissue in memory,
and generates the frozen 512-step 2× and 1024-step 4× lattices. Coordinate
counts must be nonempty and cannot exceed the policy's theoretical capacity.

Each exact four-file artifact set is written to a unique sibling staging
directory, fsynced, independently validated, published with Linux
`RENAME_NOREPLACE`, and validated again. The publisher has no delete or
overwrite path; interrupted staging is retained for review.

Twelve focused package/request tests and 875 full-project tests passed. Synthetic
tests covered exact authorization bindings, seven tuples, runtime one-open and
one-read counting, maximum concurrency two, failure admission behavior, atomic
publication, independent validation, no-overwrite rejection, same-size WSI path
replacement rejection, and symlinked parent rejection. No WSI was
opened and no pixel, coordinate, CUDA, feature, HEALNet, deletion, Drive, or
training operation occurred during package preparation.
