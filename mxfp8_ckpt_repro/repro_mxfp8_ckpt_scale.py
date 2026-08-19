#!/usr/bin/env python3
"""Minimal repro: an MXFP8 weight's block scale is not reproduced across a checkpoint round trip.

One Blackwell (SM100+) GPU, ~10 s. Deliberately standalone -- no Megatron import, no distributed
init, no checkpoint files -- so it can be run against any checkout. See README.md for the evidence
at model and cluster scale.

The same weight is quantized from two different sources, which is the whole bug:

  running job   distrib_optimizer.py  _copy_main_params_to_model_params
                -> fp8_utils.py:674   quantize_param_shard
                -> fp8_utils.py:423   _quantize_param_shard_impl
                -> transformer_engine cast_master_weights_to_fp8(model_params, main_params, ...)
                                                                              ^^^ fp32 MASTER

  save          dist_checkpointing/strategies/filesystem_async.py:146-161
                _clone_or_dequantize_if_needed, inside prepare_write_data, calls .dequantize()
                on quantized CUDA tensors ("a workaround to avoid the issue of quantized
                tensors not being supported by the async writer"). No scale is stored:
                `grep -r scale_inv megatron/core/dist_checkpointing/` returns nothing.

  load          the stored BF16 is written into the fp8 parameter, which REQUANTIZES --
                recomputing the block scale from the BF16 copy, not from the master.
                (grep checkpointing.py: no call to quantize_param_shard /
                cast_master_weights_to_fp8 / _copy_main_params_to_model_params)
"""

import sys

import torch

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import fp8_model_init
except ImportError as e:
    sys.exit(f"needs transformer_engine: {e}")

OUT_F, IN_F = 128, 256  # multiples of 32: MXFP8 scales in 32-element blocks


def make_mxfp8_linear():
    with fp8_model_init(recipe=MXFP8BlockScaling()):
        return te.Linear(IN_F, OUT_F, bias=False, params_dtype=torch.bfloat16).cuda()


def values(t):
    """Dequantized fp32 values on CPU. Not ``t.dequantize()`` -- on an MXFP8 *Parameter* that
    recurses through ``__torch_dispatch__`` until the stack overflows."""
    with torch.no_grad():
        return t.detach().float().cpu()


def scale(t):
    """The MXFP8 block-scale tensor (E8M0 codes: scale = 2**(code-127))."""
    for attr in ("_rowwise_scale_inv", "_scale_inv", "_columnwise_scale_inv"):
        s = getattr(t, attr, None)
        if torch.is_tensor(s):
            with torch.no_grad():
                return s.detach().float().cpu()
    return None


def main():
    if not torch.cuda.is_available():
        sys.exit("needs a GPU")
    major, _ = torch.cuda.get_device_capability()
    if major < 10:
        sys.exit(f"MXFP8 needs SM100+ (Blackwell); this GPU is sm_{major}x")

    torch.manual_seed(1234)
    master_fp32 = (torch.randn(OUT_F, IN_F, device="cuda") * 0.02).float()

    # A running job: quantize from the fp32 master.
    live = make_mxfp8_linear()
    with torch.no_grad():
        live.weight.copy_(master_fp32)
    live_vals, live_scale = values(live.weight), scale(live.weight)
    assert live_scale is not None, "no scale tensor found; this build has nothing to compare"

    # SAVE stores the weight dequantized to bf16, and no scale.
    stored_bf16 = live_vals.to(torch.bfloat16)

    # LOAD writes that bf16 into an fp8 parameter, which requantizes and recomputes the scale.
    resumed = make_mxfp8_linear()
    with torch.no_grad():
        resumed.weight.copy_(stored_bf16)
    res_vals, res_scale = values(resumed.weight), scale(resumed.weight)

    n_val = int((live_vals != res_vals).sum())
    n_sc = int((live_scale != res_scale).sum())
    print(f"weight values differing after save->load : {n_val}/{live_vals.numel()}")
    print(f"block scales differing after save->load  : {n_sc}/{live_scale.numel()}")
    if n_sc:
        d = (res_scale - live_scale)[live_scale != res_scale]
        print(f"  scale code delta min={d.min():.0f} max={d.max():.0f}  "
              f"finer={int((d < 0).sum())} coarser={int((d > 0).sum())}")
        print("  (E8M0: a LOWER code is a FINER scale, which lowers the block ceiling")
        print("   448*scale and can force top-magnitude elements to clamp)")

    print()
    if n_val:
        print(f"REPRODUCED (values): max |delta| = {(live_vals - res_vals).abs().max():.3e}")
    elif n_sc:
        print("REPRODUCED (scale): the block scale is recomputed differently at load.")
        print("Values survive at this size because both scales represent them exactly;")
        print("value loss appears once a block's amax exceeds 448*scale. See README.md.")
    else:
        print("not reproduced on this build")
    return 1 if (n_val or n_sc) else 0


if __name__ == "__main__":
    sys.exit(main())
