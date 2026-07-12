# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Determinism coverage for hybrid Mamba + MoE under expert parallelism.

``test_hybrid_model.py`` deliberately excludes EP cells (see its note: "MoE in the MLP
slot of a hybrid pattern is marginal"). But the a55b HybridEP determinism break (see the
SSD-scan FINDINGS) happens in exactly the **Mamba + MoE + EP** config, so this file adds
that cell to the determinism matrix — the parallelism dimension of the repro.

It reuses the shared inputs + ``BitExactRunner``; MoE is auto-enabled by the runner when
``EP > 1`` (it merges ``configs.moe_overrides(tp, ep)`` into the config, turning the
dense-MLP ``-`` slot into a MoE layer). Note the runner's same-process, restored-RNG
two-run check can *mask* the fused-scan divergence (Triton's cached autotune config
carries over); the faithful kernel-level reproducer lives in
``test_ssd_scan_determinism.py``. This file's job is to keep the production Mamba+MoE+EP
build under the determinism contract at all.
"""

import pytest

from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec
from megatron.core.models.hybrid.hybrid_model import HybridModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner
from tests.unit_tests.determinism.configs import hybrid_base
from tests.unit_tests.determinism.correctness.test_hybrid_model import (
    _MICRO_BATCH,
    _SEQ_LEN,
    _VOCAB_SIZE,
    _hybrid_inputs,
)

# Mamba + attention + MLP. The '-' (dense MLP) slot becomes MoE when the runner applies
# moe_overrides for EP > 1 — giving a Mamba layer coexisting with an EP-sharded MoE layer.
_LAYER_PATTERN = "M*-"

_EP_CELLS = [
    pytest.param({"EP": 2}, id="ep2"),
    pytest.param({"TP": 2, "EP": 2}, id="tp2-ep2"),
]

# Module-level lifecycle helper — only setup/teardown are used (build lambda unused).
_LIFECYCLE = BitExactRunner(
    build_model=lambda *a, **k: None,
    make_inputs=_hybrid_inputs,
    base_config=hybrid_base,
    supports_pp=False,
)


class TestHybridMoEScanDeterminism:

    def setup_method(self, method):
        _LIFECYCLE.setup()

    def teardown_method(self, method):
        _LIFECYCLE.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", _EP_CELLS)
    def test_bit_exact_mamba_moe_ep(self, parallelism):
        def build(overrides, pre_process=True, post_process=True, vp_stage=None, **_):
            cfg = TransformerConfig(**(hybrid_base() | overrides))
            return HybridModel(
                config=cfg,
                hybrid_stack_spec=hybrid_stack_spec,
                vocab_size=_VOCAB_SIZE,
                max_sequence_length=_SEQ_LEN,
                hybrid_layer_pattern=_LAYER_PATTERN,
                pre_process=pre_process,
                post_process=post_process,
                vp_stage=vp_stage,
            ).cuda()

        runner = BitExactRunner(
            build_model=build,
            make_inputs=_hybrid_inputs,
            base_config=hybrid_base,
            supports_pp=False,
            seq_len=_SEQ_LEN,
            micro_batch=_MICRO_BATCH,
        )
        runner.run({}, parallelism)
