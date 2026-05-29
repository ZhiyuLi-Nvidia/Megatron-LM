# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Determinism perf regression test — ``TransformerBlock`` (GPT-family).

Mirrors ``tests/functional_tests/test_cases/common/moe_perf/__main__.py``:

* ``cuda.Event``-based ``forward_ms`` / ``backward_ms`` / ``max_allocated_bytes``
* Asserted against ``baseline.json`` with a 1.02× regression bound.
* No per-module / per-event capture in CI — for per-module attribution,
  run nsys offline (recipe at the bottom of this file).

To re-record the baseline after an intentional perf change:

    MEGATRON_UPDATE_PERF_BASELINES=1 \
      DETERMINISM_PERF_MODE=det \
      pytest tests/unit_tests/determinism/perf/

The ``nondet`` mode does not assert against a baseline; it exists as the
comparison half — ``conftest.py``'s ``pytest_terminal_summary`` prints a
side-by-side ``TOTAL.fwd`` / ``TOTAL.bwd`` table when both modes have run.
"""

from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import pytest
import torch

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.transformer_block import TransformerBlock
from megatron.core.transformer.transformer_config import TransformerConfig
from tests.unit_tests.determinism.configs import (
    PERF_GPT_CONFIGS,
    PERF_PARALLELISM_CONFIGS,
    apply_parallelism,
    gpt_base,
    required_world_size,
)
from tests.unit_tests.determinism.perf.perf_runner import (
    UPDATE_BASELINES_ENV,
    PerfRunner,
    assert_within_baseline,
    current_perf_mode,
    load_baselines,
    maybe_update_baseline,
    read_perf_knobs,
    record_to_leaderboard,
)
from tests.unit_tests.test_utilities import Utils

_BASELINES_PATH = Path(__file__).resolve().parent / "baseline.json"

# Compute-bound knobs — see configs._PERF_SCALE.
_SEQ_LEN = int(os.environ.get("DETERMINISM_PERF_SEQ", "2048"))
_MICRO_BATCH = int(os.environ.get("DETERMINISM_PERF_BATCH", "4"))
_DTYPE = torch.bfloat16


def _build_gpt_block(overrides: dict) -> TransformerBlock:
    cfg_kwargs = gpt_base() | overrides
    cfg_kwargs.setdefault("deterministic_mode", True)
    cfg = TransformerConfig(**cfg_kwargs)
    spec = get_gpt_layer_with_transformer_engine_spec(num_experts=cfg_kwargs.get("num_moe_experts"))
    block = TransformerBlock(config=cfg, spec=spec, pre_process=True, post_process=True)
    return block.cuda().to(_DTYPE)


def _gpt_inputs(hidden_size: int) -> dict:
    hidden = torch.randn(
        _SEQ_LEN, _MICRO_BATCH, hidden_size, dtype=_DTYPE, device="cuda", requires_grad=True
    )
    mask = torch.ones(1, 1, _SEQ_LEN, _SEQ_LEN, dtype=torch.bool, device="cuda")
    return {"hidden_states": hidden, "attention_mask": mask}


class TestGptPerf:

    def setup_method(self, method):
        # CUDA_DEVICE_MAX_CONNECTIONS=1 is set in the package __init__.py
        # so it lands before CUDA context creation.
        if current_perf_mode() == "det":
            torch.use_deterministic_algorithms(True, warn_only=True)
        else:
            torch.use_deterministic_algorithms(False)

    def teardown_method(self, method):
        torch.use_deterministic_algorithms(False)
        Utils.destroy_model_parallel()
        torch.cuda.empty_cache()

    @pytest.mark.internal
    @pytest.mark.flaky_in_dev
    @pytest.mark.parametrize("parallelism", PERF_PARALLELISM_CONFIGS)
    @pytest.mark.parametrize("cfg_overrides", PERF_GPT_CONFIGS)
    def test_perf_regression(self, cfg_overrides, parallelism, determinism_leaderboard, request):
        is_moe = "num_moe_experts" in cfg_overrides
        if parallelism.get("EP", 1) > 1 and not is_moe:
            pytest.skip("EP cell requires an MoE preset")
        required = required_world_size(parallelism)
        if Utils.world_size < required:
            pytest.skip(f"Requires {required} GPUs for {parallelism}")

        init_kwargs, _, _ = apply_parallelism(parallelism)
        cell_overrides = dict(cfg_overrides)
        if is_moe:
            tp = init_kwargs.get("tensor_model_parallel_size", 1)
            ep = init_kwargs.get("expert_model_parallel_size", 1)
            if tp > 1:
                cell_overrides["sequence_parallel"] = True
                cell_overrides["tensor_model_parallel_size"] = tp
            # ColumnParallelLinear / RowParallelLinear read this from the
            # config to decide expert vs dense tp_group routing — must be
            # propagated even when parallel_state already has EP > 1.
            if ep > 1:
                cell_overrides["expert_model_parallel_size"] = ep

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(**init_kwargs)

        warmup, active = read_perf_knobs()
        hidden_size = (gpt_base() | cell_overrides).get("hidden_size", 64)
        runner = PerfRunner(build_model=_build_gpt_block, make_inputs=partial(_gpt_inputs, hidden_size))

        mode = current_perf_mode()
        metrics = runner.measure(
            cell_overrides, deterministic=(mode == "det"), warmup=warmup, active=active,
        )

        cell_id = request.node.callspec.id
        record_to_leaderboard(determinism_leaderboard, cell_id, mode, metrics)

        # Baseline assertion gates ``det`` mode only — ``nondet`` is the
        # informational comparison half. Rank 0 owns the baseline file.
        if mode == "det" and int(os.environ.get("RANK", "0")) == 0:
            baselines = load_baselines(_BASELINES_PATH)
            if os.environ.get(UPDATE_BASELINES_ENV) == "1":
                maybe_update_baseline(cell_id, metrics, baselines, _BASELINES_PATH)
            else:
                assert_within_baseline(cell_id, metrics, baselines)


# Offline per-leaf-kernel deep dive — same recipe as moe_perf/__main__.py:404-411.
#
# nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
#     -f true -x true \
#     --cuda-graph-trace=node \
#     --capture-range=cudaProfilerApi \
#     --capture-range-end=stop \
#     -o det-perf \
#     uv run python -m torch.distributed.run --nproc-per-node=8 \
#         -m pytest tests/unit_tests/determinism/perf/test_gpt_perf.py
# nsys stats --report nvtx_sum --format csv det-perf.nsys-rep
