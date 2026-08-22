# MXFP8 block scales are re-encoded across a checkpoint round trip (weights are preserved)

**Read the retraction first.** Earlier revisions of this document claimed MXFP8 *weights* are
corrupted by checkpoint save/load. **That claim is false and has been withdrawn** — see
"What was wrong" at the bottom. Checkpoint state round-trips correctly.

---

## What is actually true

**1. Model state survives a checkpoint round trip exactly.** Measured elementwise on the real
`save_checkpoint`/`load_checkpoint` path (4 GPUs, Nemotron-4-shaped hybrid, trained weights),
comparing every tensor at the same logical point in a continuous run and a resumed run:

| component | result |
|---|---|
| parameters | **numerically identical** — `max\|Δ\| = 0.0` over 404,002,560 elements |
| buffers | **identical**, 0/48 differ (incl. `expert_bias` and the `persistent=False` `local_tokens_per_expert`) |
| optimizer fp32 master, `exp_avg`, `exp_avg_sq` | **identical** |
| LR schedule, `consumed_train_samples` | **identical** |

The only bit-level difference anywhere is **267 signed zeros** (`+0.0` vs `−0.0`, i.e. `0x00000000`
vs `0x80000000`) out of 404M elements, all in GTP pad rows of `mixer.in_proj.weight`. Numerically
meaningless: `-0.0 == 0.0`.

**2. MXFP8 block scales ARE re-encoded, losslessly.** Every block whose scale changes moves by
exactly one E8M0 code, always finer, never coarser — 136/136 at model scale, 56/56 in the
standalone script, zero counterexamples. Concretely:

```
before:  data 224 @ scale 2⁻¹²      (E8M0 code 115)
after:   data 448 @ scale 2⁻¹³      (E8M0 code 114)
both represent 0.0546875 = 448 × 2⁻¹³ exactly
```

Why it is always finer: the quantizer must satisfy `amax / scale ≤ 448`. The original block's true
amax sat slightly above `0.0546875`, so 2⁻¹³ was too small and it used 2⁻¹². Quantizing rounded that
element *down* to exactly `224 × 2⁻¹² = 0.0546875` — precisely the ceiling of the finer scale. Save
stores the dequantized value; load recomputes the scale from it, and now 2⁻¹³ fits. Rounding can
only lower amax, so the recomputed scale can never be coarser.

This is a **representation** change with **zero** numerical effect: `repro_mxfp8_ckpt_scale.py`
reports `value gap: max|abs| = 0.0  max|rel| = 0.0`.

Why the scale must be recomputed at all: the checkpoint stores the weight dequantized to BF16
(`dist_checkpointing/strategies/filesystem_async.py:146-161`, a documented workaround for the async
writer not supporting quantized tensors) and stores no scale
(`grep -r scale_inv megatron/core/dist_checkpointing/` → nothing).

## Run it

```bash
python repro_mxfp8_ckpt_scale.py     # 1 Blackwell GPU, ~10 s, no Megatron, no distributed init
```

```
weight values differing after save->load : 0/32768
block scales differing after save->load  : 56/1024
  scale code delta min=-1 max=-1  finer=56 coarser=0
```

## The open question this does NOT answer

A 64-node 120B run resumed from a checkpoint still **does not reproduce** the continuous run
(iteration 51 loss `6.817687` vs `6.817666`, amplifying to `6.219394` vs `6.663007` by iteration
100), and two resumes from one checkpoint are byte-identical to each other (256/256 ranks).

Since the state going into the resumed step is now shown to be identical, **the checkpoint is not
the cause.** The remaining difference between the arms is that one process has run 50 steps and the
other has just started — kernel/algo selection caches, JIT state, workspace allocation, autotuning.
`--deterministic-mode` pins *which* algorithm is used; it does not make a fresh process's first-call
path identical to a warm one's. That hypothesis is untested.

## What was wrong

Earlier revisions claimed MXFP8 weights are corrupted, with a mechanism (a finer scale lowers the
`448 × scale` ceiling, so top-magnitude elements clamp). Both the claim and the mechanism are false:

- **Clamping**: refuted directly — dumping the 16 largest-magnitude values of every parameter and
  differencing the arms, **0 top-magnitude elements moved** in 231 tensors reported as "differing".
- **The "differing" counts themselves**: an artifact. The instrument compared SHA-1 of the raw
  bytes, which is bit-exact by design, so it counted `+0.0` vs `−0.0` as a difference. Elementwise
  comparison shows `max|Δ| = 0.0`. Every "value difference" reported in earlier revisions
  (782/782 at 64 GPUs, 63/136 at 4 GPUs, the lr sweep) was signed zeros.

The lesson worth passing on: SHA-1 over float buffers answers "are these bytes identical", not "are
these numbers different", and the two diverge on signed zeros, NaN payloads, and padding.
