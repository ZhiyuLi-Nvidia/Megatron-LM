---
orphan: true
---

# Determinism Operation Catalog

> This content is for developers reviewing or extending deterministic-mode
> coverage. Terms are defined in the [glossary](./glossary.md).

The catalog has two buckets:

- Operations with a deterministic code path
- Operations that deterministic mode cannot support yet

The project goal is to shrink the second bucket and make the first bucket
faster.

Most operations in a training step need no entry here. The following are
deterministic as-is:

- Elementwise ops
- GEMMs under the pinned cuBLAS workspace
- Rank-indexed collectives (all-gather, all-to-all, broadcast)
- Stable sorts
- Unique-index writes

The tables list only the operations where a choice is made.

## Deterministic Code Path Operations

Selected by `torch.are_deterministic_algorithms_enabled()` or
`config.deterministic_mode`. The default-mode path stays in the other branch.

| Operation | Where | Deterministic Path | Default Path |
| --- | --- | --- | --- |
| MoE token unpermute (combine) | `megatron/core/transformer/moe/moe_utils.py` | `index_add_` — deterministic under torch deterministic algorithms and CUDA-graph safe | `scatter_add_` (atomic accumulation) |
| MoE routing map and probabilities | `megatron/core/transformer/moe/moe_utils.py` | `index_put_(accumulate=False)` row-wise writes | out-of-place `scatter` |
| Vocab-parallel embedding | `megatron/core/tensor_parallel/layers.py` | direct indexing `weight[idx]` (deterministic backward) | `F.embedding` (non-deterministic atomic backward) |
| Gated-delta-net kernel | `megatron/core/ssm/gated_delta_net.py` | torch `chunk_gated_delta_rule` | FLA fused kernel |
| Gated-delta-net causal conv1d | `megatron/core/ssm/gated_delta_net/` | `F.conv1d` (plus transposes) | FLA `causal_conv1d` |
| Mamba/SSM Triton ops | `megatron/core/ssm/ops/common/determinism.py` | one fixed autotune config plus a zero-initialized tiled workspace reduced with an ordered `sum` | timing-based autotune, uninitialized workspace |
| Mamba/GDP causal conv1d | `megatron/core/ssm/causal_conv1d.py` (checked at mixer construction) | causal_conv1d >= 1.6.0 reduces the conv weight and bias gradients through a per-block workspace and an ordered `sum` | `atomicAdd` accumulation, whose order varies from launch to launch |
| Transformer Engine attention | `megatron/core/extensions/transformer_engine.py` | requires `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`, under which TE selects only backends that support deterministic execution (including deterministic FlashAttention backward) | TE picks freely, including atomic-accumulation attention backward |
| Inference DP scheduling and RL rollout order | `megatron/core/inference/engines/dynamic_engine.py`, `megatron/rl/rl_utils.py` | sort by stable key | completion order |

The two conv rows resolve the same problem differently on purpose. Mamba and GDP
call Dao-AILab's `causal_conv1d`, which has a deterministic reduction of its own;
gated-delta-net binds FLA's `causal_conv1d`, a different Triton kernel that
`CAUSAL_CONV1D_DETERMINISTIC` does not control, so it still falls back to
`F.conv1d` — and pays roughly ten times the deterministic kernel's device time
for it. Giving FLA's kernel the same treatment would let that row converge on
the first one.

The following environment controls make the rest of the step deterministic.
The flag `--deterministic-mode` validates and defaults these settings. For
details, refer to `megatron/training/determinism.py`.

- `NCCL_ALGO=Ring`. The tree is rejected because its reduction order is not
  user-controllable.
- `CUBLAS_WORKSPACE_CONFIG=:4096:8` (or `:16:8`).
- `NVTE_ALLOW_NONDETERMINISTIC_ALGO=0`.
- `MAMBA_DETERMINISTIC` must not be disabled.
- `CAUSAL_CONV1D_DETERMINISTIC` must not be disabled. Unset is correct: the
  kernel then follows `torch.use_deterministic_algorithms`. The Mamba and GDP
  mixers refuse to build if a deterministic run would get a conv that cannot
  deliver it — an install older than causal_conv1d 1.6.0, or an explicit `0`.

## Operations Without Determinism Support

Deterministic mode either rejects these at validation (fails closed) or they
are known open gaps.

| Operation or Feature | Where Enforced or Observed | Status |
| --- | --- | --- |
| Fused cross-entropy loss (`--cross-entropy-loss-fusion`) | rejected by `--deterministic-mode` (`megatron/training/determinism.py`) | The fused kernel is non-deterministic. Whether a deterministic variant is feasible remains an open question. Until then, the framework uses the native vocab-parallel path. |
| TP communication overlap (`--tp-comm-overlap`) | rejected by `--deterministic-mode` | Overlapped collective ordering is not reproducible. |
| Packed sequence (`thd`) in gated-delta-net | assertion in `megatron/core/ssm/gated_delta_net.py` | No deterministic packed-sequence SSM path exists yet. |
| Cross-allocation floating-point collectives (TP all-reduce, DP grad reduce-scatter) | open gap | `NCCL_ALGO=Ring` pins the algorithm but not the physical ring an allocation receives. The environment variable alone does not guarantee bit-exactness across *different* allocations for these reductions. Runs repeated within one allocation, or on allocations with identical topology, remain bit-exact. |

## Performance Notes

The deterministic paths above cost roughly 15% of step time compared to
default mode, which varies by model. Models that rely heavily on Mixture of
Experts (MoE) pay more. Measured examples range from approximately four percent
on a large dense model to approximately 17% on a hybrid MoE model. The measured
hotspots are:

- The deterministic MoE scatter and unpermute path
- The sorted router top-k
- Attention backward
- Grouped-GEMM weight gradient (wgrad)

The SSM causal conv1d is not one of them. It trades `atomicAdd` contention on
`dweight` for a workspace and one `sum`, so it costs a fixed few microseconds at
low occupancy and *saves* time once enough blocks are contending — between +24%
and -25% of the convolution's device time across the shapes measured, against a
channel-last layout that is already several times cheaper than the
channels-first alternative it replaced. Measurements, the mechanism and the
repro command are in
[`causal-conv1d-overhead.md`](./causal-conv1d-overhead.md), which is the single
source for those numbers.

Reducing this cost is a tracked workstream in
[issue #5785](https://github.com/NVIDIA/Megatron-LM/issues/5785). A change to
any row above needs a bit-exact test and a comparison of deterministic and
default performance (`tests/performance_tests/shell_test_utils/determinism/`).
