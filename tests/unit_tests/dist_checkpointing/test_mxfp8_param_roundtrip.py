# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""An MXFP8 parameter's block scale is not reproduced across a checkpoint round trip.

Save stores the weight dequantized to BF16 and stores no scale, so load must recompute the scale --
from the BF16 copy, while a running job computes it from the fp32 master. Full mechanism, evidence
at model and cluster scale, and suggested fixes: `mxfp8_ckpt_repro/README.md`.

These tests need no distributed launcher: they build one `te.Linear` and requantize it in place.
"""

import pytest
import torch

from megatron.core.fp8_utils import HAVE_TE_MXFP8TENSOR
from megatron.training.utils import get_device_arch_version

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from transformer_engine.pytorch import fp8_model_init

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False

# Mirrors tests/unit_tests/test_fp8_utils.py:22 -- MXFP8 block scaling needs Blackwell.
mxfp8_available = HAVE_TE and HAVE_TE_MXFP8TENSOR and get_device_arch_version() >= 10

# launch_on_gb200 is required or these SM100-only tests are never scheduled on Blackwell in CI and
# skip forever (marker declared in pyproject.toml, consumed by tests/unit_tests/find_test_cases.py).
pytestmark = [
    pytest.mark.launch_on_gb200,
    pytest.mark.skipif(not mxfp8_available, reason="MXFP8 requires TE and an SM100+ GPU"),
]

OUT_F, IN_F = 128, 256  # multiples of 32: MXFP8 scales in 32-element blocks


def _make_mxfp8_linear(seed):
    torch.manual_seed(seed)
    with fp8_model_init(recipe=MXFP8BlockScaling()):
        return te.Linear(IN_F, OUT_F, bias=False, params_dtype=torch.bfloat16).cuda()


def _values(t):
    """Dequantized fp32 values on CPU. Not ``t.dequantize()`` -- on an MXFP8 *Parameter* that
    recurses through ``__torch_dispatch__`` until the stack overflows."""
    with torch.no_grad():
        return t.detach().float().cpu()


def _scale(t):
    """The MXFP8 block-scale tensor (E8M0 codes: scale = 2**(code-127)), or None.

    Compared separately from the values because they move independently: the checkpoint stores no
    scale, so a values-only check reports "exact" even when the stored representation changed.
    """
    for attr in ("_rowwise_scale_inv", "_scale_inv", "_columnwise_scale_inv"):
        s = getattr(t, attr, None)
        if torch.is_tensor(s):
            with torch.no_grad():
                return s.detach().float().cpu()
    return None


class TestMXFP8ParamRoundTrip:
    def test_dequantize_requantize_preserves_values_and_scale(self):
        """In-place ``Q(D(x))`` should preserve both the values and the block scale.

        SCOPE: one unsharded tensor requantized in place, so the 32-element block partition is
        identical on both sides. It says nothing about save/load reassembling a tensor on a
        DIFFERENT partition -- for that, see the model-scale numbers in the README.
        """
        lin = _make_mxfp8_linear(seed=1234)
        w = lin.weight
        before, before_scale = _values(w), _scale(w)
        assert before_scale is not None, "no scale tensor; this build has nothing to compare"

        # SAVE stores the weight dequantized to bf16 (and no scale); LOAD writes it back into the
        # fp8 parameter, which requantizes and recomputes the scale.
        with torch.no_grad():
            w.copy_(before.to(torch.bfloat16))
        after, after_scale = _values(w), _scale(w)

        n_diff = int((before != after).sum())
        smask = before_scale != after_scale
        n_sdiff = int(smask.sum())
        print(f"\n[mxfp8-roundtrip] values differing: {n_diff}/{before.numel()}, "
              f"scales differing: {n_sdiff}/{before_scale.numel()}")
        if n_sdiff:
            b, a = before_scale.flatten(), after_scale.flatten()
            idx = (b != a).nonzero().flatten()[:8]
            print(f"[mxfp8-roundtrip] first differing scales (before -> after): "
                  f"{list(zip(b[idx].tolist(), a[idx].tolist()))}")

        assert n_sdiff == 0, (
            f"the block scale was not reproduced: {n_sdiff}/{before_scale.numel()} entries moved."
        )
        assert n_diff == 0, (
            f"weight values changed under dequantize->requantize: {n_diff}/{before.numel()}."
        )

    def test_roundtrip_is_deterministic(self):
        """Two loads of one saved copy agree -- the loss is real but repeatable, so this is a
        correctness bug rather than flakiness. Matches the cluster observation that two resumes are
        byte-identical to each other while neither matches the continuous run."""
        saved_bf16 = _values(_make_mxfp8_linear(seed=1234).weight).to(torch.bfloat16)

        outs = []
        for _ in range(2):
            fresh = _make_mxfp8_linear(seed=4321)  # different init, same load target
            with torch.no_grad():
                fresh.weight.copy_(saved_bf16)
            outs.append(_values(fresh.weight))

        assert torch.equal(outs[0], outs[1]), "two loads of one saved copy disagree"

    def test_second_roundtrip_is_idempotent(self):
        """Is the damage paid once, or on every restart? Matters for a long run that resumes many
        times."""
        lin = _make_mxfp8_linear(seed=1234)
        with torch.no_grad():
            lin.weight.copy_(_values(lin.weight).to(torch.bfloat16))
        once = _values(lin.weight)
        with torch.no_grad():
            lin.weight.copy_(once.to(torch.bfloat16))
        twice = _values(lin.weight)

        n_diff = int((once != twice).sum())
        print(f"\n[mxfp8-roundtrip] values moving on the SECOND cycle: {n_diff}/{once.numel()}")
        assert n_diff == 0, (
            f"not idempotent: {n_diff} values moved on the second cycle, so each resume degrades "
            "the weights further rather than paying a one-time cost."
        )
