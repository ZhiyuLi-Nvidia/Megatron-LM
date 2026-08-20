# MXFP8 weights are not recoverable from a checkpoint

**Symptom.** A run resumed from a checkpoint does not reproduce the run that wrote it, even with
training fully deterministic. Two resumes from the same checkpoint agree with each other
bit-for-bit; neither agrees with the continuous run.

**Scope.** Any run with `--fp8-param-gather` and `--fp8-recipe mxfp8`. Not model-specific:
reproduced on a plain GPT and on a Nemotron-4-shaped hybrid, and equally present in GDP mixer, MoE
routed/shared experts, latent projections and attention.

**Not affected.** fp32 master params, `exp_avg`, `exp_avg_sq`, and all non-quantized weights
round-trip exactly in every configuration measured.

---

# ROOT CAUSE

## In one sentence

An MXFP8 weight's **block scale is never stored in the checkpoint**, so it is **recomputed at load
from the dequantized BF16 copy** — whereas a running job computes it from the **fp32 master** — and
those two computations do not agree, so the resumed weights are not the weights that were saved.

## The two scale computations

This is the whole defect. The same weight's scale is derived from two different sources:

| | source of the scale | code |
|---|---|---|
| **running job** | **fp32 master** | `distrib_optimizer.py` `_copy_main_params_to_model_params` → `fp8_utils.py:674` `quantize_param_shard` → `fp8_utils.py:433,455` `cast_master_weights_to_fp8(model_params, `**`main_params`**`, …)` |
| **after resume** | **BF16 copy from disk** | the stored BF16 is written into the fp8 parameter; `copy_` on a `QuantizedTensor` requantizes, recomputing the scale |

## Step by step, with the exact code

1. **Save dequantizes the weight to BF16.**
   `dist_checkpointing/strategies/filesystem_async.py:146-161`, `_clone_or_dequantize_if_needed`
   inside `prepare_write_data`:
   ```python
   if ten.device.type == "cuda" and "dequantize" in type(ten).__dict__:
       ten = ten.dequantize()
   ```
   Its own comment calls this *"a workaround to avoid the issue of quantized tensors not being
   supported by the async writer."* So the BF16 store is a **writer-compatibility workaround**, not
   a considered storage format. (For GTP tensors the same effect is stated in the docstring of
   `generalized_tensor_parallelism.py:2542` — *"FP8 shards dequantized to BF16 for save"* — but that
   path is GTP-only and the bug reproduces with GTP entirely absent.)

2. **No scale is stored.** `grep -r scale_inv megatron/core/dist_checkpointing/` returns nothing.
   The checkpoint holds values, never the E8M0 block scales.

3. **Therefore load MUST recompute the scale**, and it has only the BF16 copy to compute it from.

4. **Load never re-derives the weight from the master**, even though the master was just restored
   *exactly*. `grep -c 'quantize_param_shard\|cast_master_weights_to_fp8\|_copy_main_params_to_model_params' megatron/training/checkpointing.py`
   → **0**.

5. **The recomputed scale is systematically finer** — measured, never once coarser:
   136/136 MXFP8 params on the model-scale run, 56/56 in the standalone repro, every one by exactly
   **one E8M0 exponent**.

6. **Weights do change** — 63/136 MXFP8 tensors at model scale, 782/782 at 64 GPUs. **How** the
   re-encoding turns into a value change is NOT established; see "What is not known" below.

7. **Different weights at the resumed step → different forward → the trajectory diverges.**

The transform is **lossy but deterministic**, which is why two resumes are byte-identical to each
other while neither matches the continuous run — a flaky bug could not do that.

---

## Run the repro

```bash
python repro_mxfp8_ckpt_scale.py     # 1 Blackwell GPU, ~10 s, no Megatron, no distributed init
```

Observed (GB200, sm_103):

```
weight values differing after save->load : 0/32768
block scales differing after save->load  : 56/1024
  scale code delta min=-1 max=-1  finer=56 coarser=0
```

Every differing scale moves by exactly one exponent, always finer, never coarser — a systematic
bias between two computations, not tie-breaking noise. Values survive at this size because both
scales represent these particular values exactly. Value changes do appear at model scale
(63/136 tensors), by a mechanism that is not yet established -- see "What is not known".

## Evidence

| measurement | result |
|---|---|
| 64 GPUs, 120B recipe: resume vs continuous | every MXFP8 param differs (782/782); optimizer state exact (0/1350) |
| 64 GPUs: two resumes from one checkpoint | byte-identical, 256/256 ranks — lossy but **deterministic** |
| 4 GPUs, Nemotron-4-shaped hybrid | 136/136 scales differ, 63/136 values differ, optimizer exact |
| scale direction, all 136 params | **finer 136, coarser 0** |
| value change without scale change | **never observed** |
| lr sweep 1.6e-4 / 1.6e-3 / 1.6e-2 | 65 / 71 / 43 differing — does **not** scale with lr, so not a step offset |
| this script (isolated, unsharded) | 56/1024 scales differ, all by exactly −1 exponent |

## Why CI is green

There **is** MXFP8 checkpoint coverage — it just cannot see an effect this small.

`tests/unit_tests/transformer/moe/test_moe_single_grouped_weight_numerics.py:506`
`run_mxfp8_checkpoint_save_load_next_loss` trains 2 steps, `force_param_sync`, `save_checkpoint`,
reloads, runs one more step — then asserts on the **loss** via `assert_loss_parity`, which for
mxfp8 uses `atol = rtol = 2e-2`. The effect here is ~3e-6 relative on the first resumed step, about
four orders of magnitude under that tolerance. It compares a downstream scalar, never the weights.

`tests/unit_tests/dist_checkpointing/test_fp8.py` cannot catch it either: it builds a tensorwise
`Float8Tensor` (never an `MXFP8Tensor`, so block scaling is never exercised), on a 3-element tensor
(smaller than one 32-element block), constant-filled with `4` (uniform and a power of two, so any
recomputed scale reproduces it), and asserts `loaded == 4` against a literal rather than against the
saved codes and scale.

**The gap: nothing compares an MXFP8 parameter's values and block scale against what was saved.**

---

## Suggested fixes (for the owner to choose)

**A — re-derive from master on load (smaller change, recommended first).**
After optimizer state is restored, regenerate fp8 weights from the fp32 master via
`quantize_param_shard` — the same call every training step makes. Both paths then compute the scale
from the same source. No format change; keeps the BF16 copy for resharding and `--no-load-optim`.

**B — store fp8 data + scale (format change).**
Store `_data` and `_rowwise_scale_inv` and skip requantization entirely. Exact and smaller on disk,
but fp8 shards cannot be resharded arithmetically (a different TP/EP/GTP topology must recompute
block scales over regrouped elements), every weight consumer must learn the format, and it would
require the async writer to handle quantized tensors — the very thing step 1's workaround exists to
avoid.

---

## What is not known

**How the re-encoding becomes a value change is unexplained.** An earlier draft claimed clamping:
a finer scale lowers the block ceiling `448 * scale`, so top-magnitude elements exceed it and clamp.
**That was measured and refuted.** On the real save/load path at model scale, dumping the 16
largest-magnitude values of every parameter at full precision:

```
MXFP8 tensors compared: 544, value-hash differing: 231
top-16 elements that actually moved: 0
```

Not one top-magnitude element moved in any of the 231 tensors whose values differ. Clamping can only
affect top-magnitude elements, so nothing clamped. This is corroborated by the aggregates: `absmax`,
`mean` and `abs-sum` are identical to **17 significant digits** on every value-differing tensor,
which a clamp of that kind could not leave untouched.

So the value differences are real but live in the smaller magnitudes, and **no maximum relative gap
has been measured** — the probe sampled the top of each tensor, which is where the loss turned out
not to be.

Also unverified: nobody has stepped inside `cast_master_weights_to_fp8` to watch the exponent being
selected. That is a TransformerEngine read.

**What is solid:** the scale is not stored, is recomputed on load from a different source than a
running job uses, and comes back uniformly one E8M0 exponent finer — 136/136 and 56/56, never once
coarser. And the weights differ after a resume. The link between those two facts is not yet proven.
- An earlier draft of this document cited `dist_checkpointing/utils.py:236`
  `force_all_tensors_to_non_fp8` as the save-side dequantize. That was **wrong**: it is called from
  `serialization.py:135` inside `def load(...)` and dequantizes the *destination* state dict on
  load. The save-side dequantize is `filesystem_async.py:146-161`, cited above.
