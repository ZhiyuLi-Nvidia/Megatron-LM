# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Model-level determinism check for GPTModel.

Adding a new parallelism cell is a one-line append to
``configs.PARALLELISM_CONFIGS`` — this file does not need to change.
"""

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.configs import GPT_CONFIGS, PARALLELISM_CONFIGS, gpt_base
from tests.unit_tests.determinism.bit_exact_runner import BitExactRunner

_SEQ_LEN = 32
_MICRO_BATCH = 4
_VOCAB_SIZE = 128


def _build_gpt(
    overrides: dict,
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage=None,
    **_,
):
    cfg = TransformerConfig(**(gpt_base() | overrides))
    model = GPTModel(
        config=cfg,
        transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(
            num_experts=(gpt_base() | overrides).get("num_moe_experts"),
        ),
        vocab_size=_VOCAB_SIZE,
        max_sequence_length=_SEQ_LEN,
        pre_process=pre_process,
        post_process=post_process,
        vp_stage=vp_stage,
    )
    return model.cuda()


def _gpt_inputs() -> dict:
    return {
        "input_ids": torch.randint(
            0, _VOCAB_SIZE, (_MICRO_BATCH, _SEQ_LEN), device="cuda", dtype=torch.long
        ),
        "position_ids": (
            torch.arange(_SEQ_LEN, device="cuda", dtype=torch.long)
            .unsqueeze(0)
            .repeat(_MICRO_BATCH, 1)
        ),
        "attention_mask": torch.ones(
            _MICRO_BATCH, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool, device="cuda"
        ),
    }


RUNNER = BitExactRunner(
    build_model=_build_gpt,
    make_inputs=_gpt_inputs,
    base_config=gpt_base,
    supports_pp=True,
    seq_len=_SEQ_LEN,
    micro_batch=_MICRO_BATCH,
)


class TestGPTModelDeterminism:

    def setup_method(self, method):
        RUNNER.setup()

    def teardown_method(self, method):
        RUNNER.teardown()

    @pytest.mark.internal
    @pytest.mark.parametrize("parallelism", PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("cfg_overrides", GPT_CONFIGS)
    def test_bit_exact_under_parallelism(self, cfg_overrides, parallelism):
        RUNNER.run(cfg_overrides, parallelism)
