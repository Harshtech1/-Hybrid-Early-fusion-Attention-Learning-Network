# BRCA B03 GPU feature-pilot result

Status: **successfully completed and independently validated**.

The exact authorized B03 pilot processed 8,875 scale-2× patches and 2,257 scale-4× patches on one Tesla T4. End-to-end execution took **118.41 seconds**, including deterministic input checks, synthetic and real-feature HEALNet numerical smokes, compact atomic publication, and immediate validation.

| Branch | Feature shape | Streaming extraction time | Effective MPP |
|---|---:|---:|---:|
| 2× | `[8875,2048]` | 95.81 s | 0.4936 |
| 4× | `[2257,2048]` | 17.23 s | 0.9872085 |
| Combined | `[11132,2048]` | natural `torch.cat(..., dim=0)` | — |

The natural HEALNet WSI input was `[1,11132,2048]`; RNA, mutation, and CNV remained separate `[1,1,D]` modalities. Both the synthetic and real-feature interface tests produced finite float32 output `[1,4]` and WSI attention `[1,2,11132]`. These tests used random HEALNet initialization and are not trained survival predictions.

The compact artifact contains one canonical combined tensor and exact row provenance:

- Total files: 4
- Total logical size: **91,770,745 bytes**
- Manifest SHA256: `3e1e3651604f7fe51540446a3a7836ded3ba74d5c6bf17b3a2a152cfebab004e`
- Tensor SHA256: `f8c99940669135ea3f24074908288acb820f5dd214755b2a8e98c6869d5e961e`
- Tensor-content SHA256: `d7029ce42f1ca3bdd0eb629707274665e59a814fc5548241b7ad99157b0ad7a7`
- Provenance rows: 11,132, with 2× range `[0,8875]` and 4× range `[8875,11132]`

Independent read-only validation confirmed a finite, contiguous CPU float32 tensor, exact manifest and sidecar hashes, contiguous global/local provenance indices, correct branch ordering, and absence of staging or partial output. The raw WSI and coordinate manifest still match their frozen hashes. The T4 returned to 0 MiB allocated and 0% utilization after execution.

Peak GPU allocation was 483,956,224 bytes, confirming that patch streaming and preprocessing—not GPU capacity—remain the dominant cost.

The encoder was the user-selected ImageNet1K V2 ResNet50 checkpoint. This maintains engineering comparability with Q25, Q50, Q75, B01, and B02, but differs from the paper's stated Kather100K pretraining and must remain disclosed in comparative reporting.

No training, backward pass, optimizer step, AMP, TF32, CPU fallback, coordinate regeneration, B04–B06 processing, prior-pilot or BLCA change, Drive operation, deletion, cohort expansion, or official HEALNet modification occurred.

The required stop has been reached. The next stage is a CPU-only comparison of B03 with the completed pilots and a decision on B06 header-only inspection before any further GPU work.
