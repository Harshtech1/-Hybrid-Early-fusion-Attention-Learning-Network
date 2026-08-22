# Pilot-side multiscale WSI adapter design

> Historical adapter milestone. It records the pre-extraction 24-test state and is superseded for current status by [../../reports/validated_pilot_baseline.md](../../reports/validated_pilot_baseline.md).

**Final status: `READY_FOR_GPU_MULTISCALE_EXTRACTION`**

## Result

The pilot-side adapter now enforces the verified multiscale contract without modifying the official HEALNet repository:

```text
scale_2x [N1,2048] ─┐
                     ├─ torch.cat(dim=0) -> [N1+N2,2048]
scale_4x [N2,2048] ─┘
                                  |
                                  ├─ provenance rows [0..N1+N2-1]
                                  |
                                  └─ transpose -> [2048,N1+N2]
```

For multiple patients:

```text
patient feature bags [P_i,2048]
              -> independently initialized padding
              -> features [B,2048,P_max]
              -> valid_mask [B,P_max] (bool)
              -> lengths [B] (int64)
```

## Files created

```text
multiscale_feature_pilot/
├── __init__.py
├── config/
│   └── pilot_config.yaml
├── reports/
│   ├── adapter_design.md
│   └── next_real_multiscale_extraction.md
├── src/
│   ├── __init__.py
│   ├── multiscale_bag.py
│   ├── padding.py
│   └── provenance.py
└── tests/
    ├── test_multiscale_bag.py
    └── test_padding_adapter.py
```

Pre-existing gate and scale-provenance files under this pilot directory were left unchanged.

## Feature contract

Implemented in `src/multiscale_bag.py`:

- input type must be `torch.Tensor`;
- each branch must be rank 2;
- each branch must contain at least one row;
- feature width must equal 2048;
- dtype must be exactly `torch.float32`;
- every value must be finite;
- branch devices must match;
- operation must be `cat`;
- concatenation dimension must be `0`;
- `stack` and feature-axis concatenation are explicitly rejected;
- returned shape is asserted to equal `[N1+N2,2048]`.

`load_feature_matrix` uses `torch.load(..., weights_only=True)` and applies the same validation before returning a tensor.

Combined row order is fixed:

1. all `scale_2x` feature rows;
2. all `scale_4x` feature rows.

## Provenance contract

Implemented in `src/provenance.py`.

Every combined feature row has exactly one immutable `PatchProvenance` record:

```text
global_row_index
branch
local_patch_index
x
y
level
mpp_x
mpp_y
```

Validation requires:

- coordinate shape `[N,2]` and integral coordinates;
- coordinate count equal to the branch feature-row count;
- branch order and names exactly `scale_2x`, then `scale_4x`;
- contiguous global row indices beginning at zero;
- local patch indices beginning at zero independently within each branch;
- non-negative level and finite positive MPP values.

`write_provenance_csv` emits the table in the same deterministic row order. No real coordinate file was created in this task.

## Padding and mask contract

Implemented in `src/padding.py`.

One patient:

```text
[P_patient,2048] -> transpose + contiguous -> [2048,P_patient]
```

Multiple patients:

```text
input:       sequence of [P_i,2048] float32 tensors
features:    [B,2048,P_max] float32
valid_mask:  [B,P_max] bool
lengths:     [B] int64
```

For patient `i`, slots `[0:P_i]` contain that patient's features and are marked valid. Slots `[P_i:P_max]` are initialized to the explicit `pad_value` and remain invalid. Each batch is newly allocated; there is no cross-slide feature-buffer reuse.

The validator checks that mask sums equal recorded lengths and that all invalid positions still contain only the pad value.

### Important mask boundary

The `[B,P_max]` mask truthfully describes **patch-slot validity**. It must not be passed directly as the existing `HealNet.forward(mask=...)` argument under the released transposed layout:

```text
released WSI tensor:       [B,2048,P_max]
HEALNet attention length:  2048
adapter validity length:   P_max
```

The official attention mask is applied to the flattened 2048 axis, whereas padding occurs on the last `P_max` axis. The adapter therefore returns the mask as explicit bookkeeping metadata. No official code was patched, and no false claim of direct mask compatibility is made.

This does not block single-patient GPU feature extraction. It does require a later, explicit design decision before safe padded multi-patient training.

## Unit-test result

Command:

```text
python -m pytest multiscale_feature_pilot/tests -q
```

Measured result:

```text
24 passed
```

Covered cases include:

- `[100,2048] + [80,2048] -> [180,2048]`;
- `[1,2048] + [1,2048] -> [2,2048]`;
- wrong feature width rejection;
- non-float32 rejection;
- NaN, positive Inf, and negative Inf rejection;
- explicit `dim=1` and `stack` rejection;
- feature/provenance count mismatch rejection;
- scale-2-first/scale-4-second row and provenance ordering;
- per-patient transpose;
- separate synthetic WSI, RNA, mutation, and CNV inputs producing `[1,4]`;
- exact patient/slide Omic matching with separate ordered RNA, mutation, and CNV tensors;
- `[180,2048]` and `[250,2048]` padded to `[2,2048,250]`;
- exact `[2,250]` boolean mask semantics;
- no padded value marked valid;
- read-only synthetic HEALNet interface acceptance.

## HEALNet compatibility result

The test loaded the existing class directly from:

```text
/teamspace/studios/this_studio/healnet/healnet/models/healnet.py
```

It did not edit that file, load a checkpoint, or train. A tiny random-weight, WSI-only `HealNet` instance received:

```text
combined synthetic bag:  [180,2048]
released orientation:    [2048,180]
batched input:           [1,2048,180]
model output:            [1,4]
```

All output values were finite. Therefore the combined and transposed WSI shape is accepted by the actual existing model interface.

## Remaining blockers and deferred decisions

1. CUDA remains unavailable in the current runtime; real extraction remains gated.
2. The ImageNet V2 checkpoint is not cached and must not be downloaded until a GPU runtime is available.
3. The existing BLCA slide has no native 0.5-MPP level; the scale-2 branch requires an explicit resampling and coordinate-generation policy.
4. The `[B,P_max]` patch-validity mask does not align with the released HEALNet attention axis after transpose. Resolve this before multi-patient training; do not silently ignore padding.
5. Real feature hashes and real coordinate-provenance CSVs can only be produced after extraction.
6. Omic loading/fusion remains outside this adapter task.

## Safety confirmation

| Question | Result |
|---|---|
| Official HEALNet modified by this task? | **NO** |
| Google Drive touched? | **NO** |
| Real WSI opened or modified by this task? | **NO** |
| New WSI downloaded? | **NO** |
| CLAM run or real coordinates created? | **NO** |
| ImageNet checkpoint downloaded? | **NO** |
| ResNet50 run? | **NO** |
| Real ResNet features generated? | **NO** |
| HEALNet trained? | **NO** |

**Final status: `READY_FOR_GPU_MULTISCALE_EXTRACTION`**
