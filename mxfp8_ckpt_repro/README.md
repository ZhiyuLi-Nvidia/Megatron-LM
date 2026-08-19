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

## Run the repro

```bash
python repro_mxfp8_ckpt_scale.py     # 1 Blackwell GPU, ~10 s, no Megatron, no distributed init
```

Observed output (TE in `nvcr.io`-derived container, GB200, sm_103):

```
weight values differing after save->load : 0/32768
block scales differing after save->load  : 56/1024
  scale code delta: min=-1 max=-1 (negative = finer scale on load)
  finer: 56   coarser: 0
```

Every differing scale moves by **exactly one exponent, always finer, never coarser.** That is a
systematic bias between two scale computations, not tie-breaking noise.

In this minimal case the weight *values* survive — both scales represent these particular values
exactly. Value loss appears at model scale, where a block's `amax` can exceed what the finer scale
can represent (`448 * scale`) and the top elements clamp.

---

## Root cause

The same weight is quantized from **two different sources**, and only one of them is the source of
truth.

| path | code | scale derived from |
|---|---|---|
| running job | `distrib_optimizer.py` `_copy_main_params_to_model_params` → `fp8_utils.py:674` `quantize_param_shard` → `fp8_utils.py:423` `_quantize_param_shard_impl` → `cast_master_weights_to_fp8(model_params, main_params, …)` | **fp32 master** |
| after resume | dist-ckpt writes the stored BF16 into the parameter; `copy_` on a `QuantizedTensor` requantizes | **BF16 checkpoint copy** |

Save deliberately stores the weight dequantized to BF16, and stores **no scale**:

* `dist_checkpointing/utils.py:236` `force_all_tensors_to_non_fp8`
* `dist_checkpointing/strategies/filesystem_async.py:160` `ten.dequantize()`
* `generalized_tensor_parallel.py` — *"FP8 shards dequantized to BF16 for save"*
* `tests/unit_tests/dist_checkpointing/test_fp8.py` asserts this is correct behaviour

On load, **nothing re-derives the weight from the master.** `grep` of `checkpointing.py` finds no
call to `quantize_param_shard`, `cast_master_weights_to_fp8`, or
`_copy_main_params_to_model_params`. So the scale must be recomputed from the lossy BF16 copy —
while the exact fp32 master sits in the same checkpoint, already restored.

Chain: different amax source → uniformly finer scale → lower block ceiling (`448 * scale`) →
top-magnitude elements clamp → weights change → trajectory diverges.

---

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
reloads, and runs one more step. But it asserts on the **loss**, via `assert_loss_parity`
(same file), which for mxfp8 uses:

```python
atol = rtol = 2e-2      # 2% tolerance, on the loss
```

The effect measured here is ~3e-6 relative on the first resumed step — about four orders of
magnitude under that tolerance. The test compares a downstream scalar, never the weights.

`tests/unit_tests/dist_checkpointing/test_fp8.py` also cannot catch it, for four independent
reasons:

1. builds a tensorwise `Float8Tensor` (`Float8Quantizer`, `scale = 1.0`), never an `MXFP8Tensor`,
   so block scaling is never exercised;
2. tensor is `torch.full((3,), …)` — 3 elements, smaller than one 32-element block;
3. constant fill of `4` — uniform and a power of two, so any recomputed scale reproduces it;
4. asserts `loaded == 4` against a literal, not against the saved tensor's codes and scale.

So the gap is specific: **nothing compares an MXFP8 parameter's values and block scale against
what was saved.** Existing coverage checks that the loss stays within 2%, which it does.

---

## Suggested fixes (for the owner to choose)

**A — re-derive from master on load (smaller change, recommended first).**
After optimizer state is restored, regenerate fp8 weights from the fp32 master via
`quantize_param_shard` — the same call every training step makes. Both paths then compute the scale
from the same source. No format change; keeps the BF16 copy for resharding and `--no-load-optim`.
Gives exactness at the same topology, which is the case that matters for resume.

**B — store fp8 data + scale (format change).**
Store `_data` and `_rowwise_scale_inv` and skip requantization entirely. Exact and smaller on disk,
but fp8 shards cannot be resharded arithmetically (a different TP/EP/GTP topology must recompute
block scales over regrouped elements), every weight consumer must learn the format, and it inverts
the contract `test_fp8.py` currently asserts.

Worth noting `force_all_tensors_to_non_fp8` was introduced for `--no-load-optim`
(commit `1e42279a9`), where dequantized BF16 is genuinely the right thing because there is no master
to re-derive from. The defect is reusing that artifact on the normal path.

---

## Caveat

The chain "fp32-vs-BF16 amax → finer scale → clamping → value change" is inferred from the code path
plus the measurements above, which all agree with it. Nobody has stepped inside
`cast_master_weights_to_fp8` to watch the exponent being selected — that is a TransformerEngine-side
read and is the one unverified link.
