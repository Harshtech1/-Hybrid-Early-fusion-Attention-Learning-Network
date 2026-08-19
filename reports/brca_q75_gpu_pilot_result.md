# BRCA Q75 GPU pilot result

Status: `BRCA_Q75_GPU_FEATURE_PILOT_SUCCESS`

The exact authorized Q75 pilot for patient `TCGA-E2-A154` completed on one
Tesla T4. The successful managed process ran from
`2026-08-19T18:51:23.274684+00:00` to
`2026-08-19T18:55:28.431701+00:00`, taking 245.16 seconds. It used source
commit `72a86eb8c3669c24f28b246ae6971d8e76122b45` and authorization SHA256
`4594bf3a165cf6c276e355ada0dfe434ee5f0474f32f0aff5390671fb00e17c7`.

## Feature result

The two branches executed sequentially with one shared classifier-removed
ResNet50 and ImageNet1K V2 checkpoint:

- Scale 2x: `[13487,2048]` float32, finite, 181.63 seconds streaming,
  37.58 seconds inside model forward calls.
- Scale 4x: `[3458,2048]` float32, finite, 31.24 seconds streaming,
  9.86 seconds inside model forward calls.
- Combined: exact `torch.cat([scale_2x, scale_4x], dim=0)` result
  `[16945,2048]`, with all 2x rows first and all 4x rows second.

No pooling or transposition occurred. The natural HEALNet WSI shape was
`[1,16945,2048]`. ResNet peak allocated GPU memory was 481,859,072 bytes.

## HEALNet numerical checks

Both the pre-extraction synthetic check and the real-feature numerical smoke
returned finite float32 `[1,4]` outputs. Each produced the required attention
shapes: WSI `[1,2,16945]` and RNA, mutation, and CNV `[1,2,1]` each. The three
Omic tensors remained separate modalities with shapes `[1,1,1558]`,
`[1,1,21]`, and `[1,1,1333]`.

The HEALNet instance was randomly initialized and untrained. These results
verify the numerical interface and attention path; they are not predictions or
scientific survival results.

## Published artifacts

The exact six-file artifact set was atomically published outside Git at
`/teamspace/studios/this_studio/brca_pilot_data/Q75.features`. Its total size
is 278,525,034 bytes. The manifest SHA256 is
`589ce49baf2df0f41f345377229b661d05e6cc42399b8aa6545b1b3e30b09175`.

Independent post-publication validation confirmed:

- exact regular, non-symlink six-file set;
- CPU contiguous finite float32 tensors;
- `[13487,2048]`, `[3458,2048]`, and `[16945,2048]` shapes;
- bit-exact branch prefix/suffix concatenation;
- 16,945 row-provenance records in branch order;
- matching manifest, sidecar, file, semantic tensor, and coordinate hashes;
- no output lock or staging residue.

## Interruption note

An earlier managed process was interrupted by conversation-session steering
before publication. Its stdout and exact in-memory progress were not
recoverable. A read-only audit immediately afterward found no active process,
final directory, lock, staging directory, or published artifact. All immutable
source bindings were revalidated before the successful retry. A detached
launcher probe subsequently failed before Python started and is not counted as
a process attempt. Neither event represents a data, geometry, model, or
scientific failure.

## Boundary and stop

The successful attempt performed exactly 13,487 scale-2x and 3,458 scale-4x
patch reads for the one authorized patient. Training, backward passes,
optimizer steps, coordinate regeneration, Q25/Q50 operations, full-cohort
processing, Drive operations, and raw-file deletion were zero. The WSI remains
retained, official HEALNet and the frozen BLCA reference remain unchanged, and
the GPU returned to 0 MiB used with no compute process.

The required stop was reached. A separate reviewed design and authorization is
required before training or full-cohort processing.
