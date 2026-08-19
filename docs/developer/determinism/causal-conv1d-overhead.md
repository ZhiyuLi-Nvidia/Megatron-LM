---
orphan: true
---

# SSM Causal Conv1d: Determinism Mechanism and Overhead

> This page explains how the SSM convolution's deterministic backward works and
> what it costs. Terms are defined in the [glossary](./glossary.md); the summary
> row lives in the [operation catalog](./op-catalog.md).

The Mamba and gated-delta-product (GDP) mixers call Dao-AILab's `causal_conv1d`.
Its backward reduces the convolution's weight and bias gradients across thread
blocks, and by default that reduction is not reproducible. Since 1.6.0 the
extension carries a deterministic reduction, which Megatron requires whenever
`--deterministic-mode` is on and refuses to run without.

## The general pattern: private buffer, then ordered reduce

The convolution is one instance of the shape almost every non-reproducible
kernel in this codebase has, so it is worth stating in the general form before
the specifics.

**The problem.** A kernel splits an output element's sum across many thread
blocks and combines the partials with `atomicAdd`. The atomic guarantees no
lost update, but it does not fix the *order*, and floating-point addition is not
associative — so the last places of the result follow whichever block finished
first, which varies from launch to launch.

**The remedy.** Give every block a private slot instead of a shared cell, then
reduce those slots in a fixed order afterwards:

1. Each block writes its partial to an address derived from its own block
   index. The addresses are disjoint, so there are no atomics at all.
2. A second launch reduces that workspace along the tile axes. Its order is a
   property of the tensor layout, not of the scheduler.

The kernel boundary between the two phases is what supplies the ordering.

**Three things this framing gets right that "minimize synchronization" gets
wrong**, and they matter for predicting the cost:

- `atomicAdd` is not a barrier and buys no synchronization. What it costs is
  *ordering information*. There is no race to fix and no sync to remove.
- The deterministic path **adds** work: a zeroed workspace, a reduction kernel,
  and a final accumulate. It is not a simplification.
- It nevertheless often runs *faster*, because the atomics were serialising on a
  small output. That gain scales with the number of contending blocks while the
  added reduction scales only with the workspace, so the two cross over — which
  is exactly the sign change in the [speed table](#speed) at batch 4.

**Why the result is reproducible**, link by link: the tiling is a pure function
of the shape, so the same blocks exist every run; each block's partial is
computed in a fixed unrolled order, so its bits are the same; each partial has a
fixed address, so the filled workspace is bit-identical; and the final reduction
has a fixed order, so the same bits come out.

**The cost, in general terms.** Memory grows by one fp32 slot per (tile, output
element) — proportional to the tiling, so it tracks the activation size rather
than the output size. Time grows by a zero-fill plus a reduction over that
workspace, and shrinks by whatever atomic contention is removed. The pattern is
cheapest exactly where it matters most: many contending blocks means both more
non-determinism and more contention to reclaim.

Megatron applies the same pattern to its in-repo Mamba Triton kernels through
`alloc_tile_workspace` and `finalize_tile_workspace` in
`megatron/core/ssm/ops/common/determinism.py` — a zero-initialized tiled
workspace reduced with an ordered `sum`. `causal_conv1d` implements it upstream;
Megatron's job for this op is only to require it and to fail closed when it is
not available.

## Why the default is not reproducible

Each thread block computes a partial `dweight` over its own tile of the input,
reduces it within the block, and then has to combine it with every other block's
partial. The default does that with `atomicAdd` straight into the output:

```c++
// csrc/causal_conv1d_bwd.cu, channel-last backward
dweight_vals[w] = Allreduce<kNThreadsPerRow>::run(dweight_vals[w], sum_op);
if (col_idx == 0 && ...) {
    atomicAdd(&dweight[row_idx * params.dweight_c_stride
                     + w * params.dweight_width_stride], dweight_vals[w]);
}
```

Every block adds into the same small `dim x width` fp32 array. `atomicAdd`
guarantees no lost update, but it does **not** fix the order, and fp32 addition
is not associative, so the last places of the result follow whatever order the
blocks happened to finish in — which varies from launch to launch.

How many partials land on one element is what decides whether that is visible:

| layout | grid | contributions per `dweight` element |
| --- | --- | --- |
| channels-first (`causal_conv1d_bwd_kernel`) | `(batch, dim)`, one block walks all of L | `batch` |
| channel-last (`causal_conv1d_channellast_bwd_kernel`) | `(batch, ceil(L/T), ceil(dim/C))` | `batch * ceil(L/T)` |

`T` is the sequence tile: `kChunkSizeL = seqlen <= 128 ? 64 : 128`, so 128 at
any training length. `C` is the channel tile, `128 / sizeof(dtype)` elements.

With a single contribution there is nothing to order, which is why the
channels-first layout is bit-exact at micro-batch 1 — by accident of the grid
shape, not by design. Every other case drifts. Measured, 20 replays of one
backward from identical inputs (GB300, bf16, dim 4096):

| batch | seqlen | layout | contributions | replays differing in `dweight` |
| ---: | ---: | --- | ---: | ---: |
| 1 | 4096 | channels-first | 1 | 0 / 19 |
| 4 | 4096 | channels-first | 4 | 19 / 19 |
| 1 | 4096 | channel-last | 32 | 19 / 19 |
| 1 | 8192 | channel-last | 64 | 19 / 19 |
| 4 | 8192 | channel-last | 256 | 19 / 19 |

`dx` is reproducible in every configuration, because each element is written by
exactly one block rather than accumulated. The forward is reproducible too.

## What the deterministic path does

It removes the cross-block accumulation instead of trying to order it. The
kernel branches only on *where* a partial is written — the arithmetic producing
`dweight_vals[w]` above is identical in both modes:

```c++
if constexpr (kDeterministic) {
    dweight_ws[batch_id * ... + chunk_l_id * ... + channel * ... + w] = dweight_vals[w];
} else {
    atomicAdd(&dweight[...], dweight_vals[w]);
}
```

Each block writes to a slot indexed by its own `(batch_id, chunk_l_id, channel,
w)`. Those addresses are disjoint, so there is no atomic, no contention, and no
ordering question inside the kernel at all. The accumulation then happens in a
second launch, whose order is fixed by the workspace layout rather than by the
scheduler:

```c++
dweight_workspace = at::zeros({batch_size, n_chunks_L, dim, width}, kFloat);
...                                    // kernel fills disjoint slots
dweight.add_(dweight_workspace.sum(at::IntArrayRef({0, 1})));
```

This is the general pattern above, instantiated: private buffer, then ordered
reduce. The workspace is
`batch x ceil(L/T) x dim x (width + 1)` fp32 values (`width` for `dweight`,
one for `dbias`), allocated and freed inside the backward.

Megatron does not select this path per call. `causal_conv1d` reads
`CAUSAL_CONV1D_DETERMINISTIC` and otherwise follows
`torch.are_deterministic_algorithms_enabled()`, which `--deterministic-mode`
sets. `assert_causal_conv1d_deterministic` in
`megatron/core/ssm/causal_conv1d.py` runs once per mixer construction and fails
the run if a deterministic build would silently get the atomic path.

## Overhead

Summary of everything the deterministic path changes, measured below:

| Axis | Deterministic mode off | Deterministic mode on |
| --- | --- | --- |
| Megatron-side check | one early return per mixer, at construction | one env read plus a memoized version lookup, at construction |
| Kernels in the backward | conv backward only | conv backward, workspace zero-fill, reduction, `add_` |
| Device time | baseline | **+24% worst case, -25% best case** (see [speed](#speed)) |
| Device memory | baseline | **+~8% of one activation tensor**, transient (see [memory](#memory)) |
| `dweight` / `dbias` | drift run to run | bit-reproducible |
| `dx`, forward | bit-reproducible already | unchanged |
| Numerical agreement | — | within 5.8e-07 relative (see [equivalence](#equivalence)) |

Nothing changes on the default path: the guard returns before doing any work
when `deterministic_mode` is off, and no kernel selection differs.

All figures GB300, bf16, width 4, `silu`, from
`tests/performance_tests/shell_test_utils/determinism/benchmark_causal_conv1d.py`.
Device time is the torch profiler's summed self-CUDA time, not wall clock: at
these sizes a python timing loop is CPU-launch-bound and reports the same number
for shapes whose work differs 4x.

**What these numbers are not.** They are single-operator microbenchmarks on one
GPU. They bound what the convolution contributes to a step; they are not a
step-time measurement, and no end-to-end A/A run backs them. The memory column
is the delta attributable to the backward with the forward outside the
measurement window — in a real step, peak memory is usually set elsewhere in the
activation graph, which is why the upstream benchmark reports no memory
difference at all at the whole-model level.

### Speed

Time is per forward + backward, including the transposes each layout needs, so
the channels-first rows carry the two full-activation copies that the
channel-last layout avoids.

| batch | dim | seqlen | channel-last | + deterministic | channels-first | `F.conv1d` |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 57.1 us | 70.9 us (+24.1%) | 229.0 us | 753.5 us |
| 1 | 4096 | 8192 | 104.7 us | 113.5 us (+8.4%) | 440.2 us | 1559.0 us |
| 4 | 4096 | 4096 | 215.6 us | 196.0 us (-9.1%) | 859.1 us | 2930.8 us |
| 4 | 4096 | 8192 | 479.2 us | 361.1 us (-24.6%) | 1693.7 us | 5816.1 us |

The cost is close to a fixed number of microseconds, so its share falls as the
op grows, and past a certain occupancy it goes negative — the deterministic path
becomes *faster* than the default. Run-to-run variation on these totals is about
1%, so treat the last digit as noise. The per-kernel split at the worst row
(batch 1, dim 4096, seqlen 4096, from the same command with `--breakdown`) shows
why:

| kernel | default | deterministic | delta |
| --- | ---: | ---: | ---: |
| `causal_conv1d_channellast_bwd_kernel` | 37.7 us | 32.2 us | **-5.5 us** |
| forward (unchanged) | 16.9 us | 16.9 us | 0 |
| `FillFunctor` zero-init | 2.5 us | 5.4 us | +2.9 us |
| `reduce_kernel` (the ordered sum) | — | 12.6 us | +12.6 us |
| `add_` into `dweight` / `dbias` | — | 3.6 us | +3.6 us |
| **total** | **57.1 us** | **70.7 us** | **+13.6 us** |

The zero-init row is present in both modes: the extension always zeroes the fp32
`dweight` and `dbias` buffers it accumulates into. Determinism adds the
workspace to what has to be zeroed, which is the +2.9 us; it does not introduce
the kernel.

The backward kernel itself gets *faster* by removing atomic contention on a
`dim x width` array that `batch * ceil(L/T) * ceil(dim/C)` blocks were all
hammering. That saving grows with the contention, while the reduction cost grows
only with the workspace, so the two cross over.

For scale: in that same breakdown, 191 us of the 229 us channels-first row is
`direct_copy_kernel`, i.e. the `.contiguous()` copies. Determinism costs a
fraction of what the channel-last layout saves.

### Memory

Extra peak allocated across the backward, against the analytic workspace size:

| batch | dim | seqlen | one activation | extra | workspace |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4096 | 4096 | 32.0 MB | 2.6 MB | 2.5 MB |
| 1 | 4096 | 8192 | 64.0 MB | 5.1 MB | 5.0 MB |
| 4 | 4096 | 4096 | 128.0 MB | 10.1 MB | 10.0 MB |
| 4 | 4096 | 8192 | 256.0 MB | 20.1 MB | 20.0 MB |

The workspace scales with `batch * seqlen * dim` exactly as activations do, so
it is a constant ~8% of one activation tensor at every shape. It is transient —
freed at the end of the backward — so it does not accumulate across layers,
though it does raise the peak if the conv backward is where peak occurs.

### Equivalence

The two paths sum the same partials in different orders, so they agree to fp32
accumulation error rather than bitwise. Two relative measures, because they say
different things:

| gradient | `max\|diff\| / max\|ref\|` | `max(\|diff\| / \|ref\|)` per element |
| --- | ---: | ---: |
| `dx` | 0 (bitwise; never goes through the reduction) | 0 |
| `dweight` | 5.6e-07 | 8.2e-03 |
| `dbias` | 4.3e-07 | 9.6e-03 |

Relative to each tensor's own magnitude the gap is a few fp32 ulps (eps is
1.2e-07), which is what reordering a sum should cost. Per element it reaches
~1e-2, and that is expected rather than alarming: a `dweight` entry near zero is
a near-total cancellation of much larger products, so its *relative* error is
large while its absolute error stays at the same few-ulp level. There is no
per-element relative bound to assert here.

`tests/unit_tests/ssm/test_causal_conv1d.py` therefore asserts the first
measure, `max|diff| <= 1e-5 * max|ref|`, roughly 18x the observed gap. Note what
that does and does not cover: it would catch a deterministic path that changed
the arithmetic or mis-indexed the workspace at scale, but not one that corrupted
only entries far below the tensor maximum.

## Reproducing

Inside a container with `causal_conv1d >= 1.6.0` and a GPU:

```bash
python tests/performance_tests/shell_test_utils/determinism/benchmark_causal_conv1d.py \
    --mode all --batch 1 4 --dim 4096 --seqlen 4096 8192
```

`--mode` selects `replay` (the determinism claim), `equivalence` (deterministic
versus default), `speed`, or `memory`; `--breakdown` adds the per-kernel split
to `speed`. Numbers above came from that command on one GB300.

The bit-exactness claims are also asserted as unit tests:

```bash
pytest tests/unit_tests/ssm/test_causal_conv1d.py
```
