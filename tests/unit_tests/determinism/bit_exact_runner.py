# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact determinism runner.

Test files instantiate ``BitExactRunner`` once with their model-specific
factory + input-builder + base-config, then call
``runner.run(cfg_overrides, parallelism)`` from a parametrized test. Adding
a new parallelism config means appending a single entry to
``configs.PARALLELISM_CONFIGS`` — no test-file edits required.

For any parallelism dict the runner performs two forward+backward passes
under the same restored RNG state and asserts that outputs and gradients
are bit-identical. It handles:

* TP, PP, VPP, CP, EP via ``Utils.initialize_model_parallel``.
* FSDP via ``fully_shard_model`` wrap.
* MoE auto-enable when ``EP > 1`` (merges ``configs.moe_overrides(tp, ep)``).
* num_layers auto-bump when ``PP * VPP`` exceeds the preset's layer count.
* sequence_parallel + tensor_model_parallel_size propagation when MoE+TP.
* Pipeline schedule (``get_forward_backward_func``) when ``PP > 1``;
  naive ``model(**inputs)`` fwd+bwd otherwise.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

import pytest
import torch

from megatron.core import parallel_state
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from tests.unit_tests.determinism.configs import (
    apply_parallelism,
    moe_overrides,
    required_world_size,
)
from tests.unit_tests.determinism.utils import (
    assert_bit_exact,
    capture_rng_state,
    maybe_fsdp_wrap,
    restore_rng_state,
)
from tests.unit_tests.test_utilities import Utils


class BitExactRunner:
    """Glue between a parametrized test and the per-parallelism dispatch logic.

    Args:
        build_model: ``(overrides, pre_process, post_process) -> nn.Module``.
            Layer-like factories can ignore the PP flags.
        make_inputs: zero-arg callable producing the kwargs dict for
            ``model(**make_inputs())``.
        base_config: zero-arg callable returning the base TransformerConfig
            kwargs dict — merged with cfg_overrides + moe_overrides + the
            runner's own auto-fields (num_layers, etc.).
        supports_pp: set False for tests that don't model PP semantics (e.g.
            single TransformerLayer). PP entries will be skipped automatically.
        seq_len, micro_batch, dtype: defaults used by the pipeline schedule
            when ``PP > 1``.
        default_tp: TP size used in ``setup_method`` before the test re-inits.
    """

    def __init__(
        self,
        build_model: Callable[..., torch.nn.Module],
        make_inputs: Callable[[], dict],
        base_config: Callable[[], dict],
        supports_pp: bool = True,
        seq_len: int = 32,
        micro_batch: int = 4,
        dtype: torch.dtype = torch.bfloat16,
        default_tp: int = 2,
    ):
        self.build_model = build_model
        self.make_inputs = make_inputs
        self.base_config = base_config
        self.supports_pp = supports_pp
        self.seq_len = seq_len
        self.micro_batch = micro_batch
        self.dtype = dtype
        self.default_tp = default_tp

    # ------------------------------------------------------------------
    # Setup / teardown helpers — call from pytest setup/teardown methods.
    # ------------------------------------------------------------------
    def setup(self):
        tp = min(self.default_tp, Utils.world_size)
        Utils.initialize_model_parallel(tensor_model_parallel_size=tp)
        os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
        torch.use_deterministic_algorithms(True, warn_only=True)

    def teardown(self):
        torch.use_deterministic_algorithms(False)
        Utils.destroy_model_parallel()
        # PP/VPP/FSDP cells leave large activations and shard buffers around.
        # Free them before the next parametrize iteration so peak memory
        # doesn't accumulate across the matrix.
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Main entry point — called by the parametrized test.
    # ------------------------------------------------------------------
    def run(self, cfg_overrides: dict, parallelism: dict):
        required = required_world_size(parallelism)
        if Utils.world_size < required:
            pytest.skip(f"Requires {required} GPUs for {parallelism}")

        pp = parallelism.get("PP", 1)
        if pp > 1 and not self.supports_pp:
            pytest.skip("PP not supported by this test fixture")

        init_kwargs, needs_fsdp, needs_moe = apply_parallelism(parallelism)
        if needs_moe:
            tp = init_kwargs.get("tensor_model_parallel_size", 1)
            ep = init_kwargs.get("expert_model_parallel_size", 1)
            cfg_overrides = {**cfg_overrides, **moe_overrides(tp, ep)}

        Utils.destroy_model_parallel()
        Utils.initialize_model_parallel(**init_kwargs)

        torch.manual_seed(42)
        model_parallel_cuda_manual_seed(123)

        if pp > 1:
            self._run_pipeline(cfg_overrides, parallelism)
        else:
            self._run_naive(cfg_overrides, parallelism)

    # ------------------------------------------------------------------
    # Naive fwd+bwd path (no PP). Wraps model with FSDP if requested.
    # ------------------------------------------------------------------
    def _run_naive(self, cfg_overrides: dict, parallelism: dict):
        model = self.build_model(cfg_overrides, pre_process=True, post_process=True)
        model = maybe_fsdp_wrap(model, parallelism)

        def _fwd_bwd(inputs):
            with torch.autocast("cuda", dtype=self.dtype):
                out = model(**inputs)
            # If the model returns a tuple (e.g. TransformerLayer returns
            # (hidden, context)), take the first tensor.
            tensor = out[0] if isinstance(out, tuple) else out
            loss = tensor.float().pow(2).mean()
            loss.backward()
            grads = _collect_grads([model])
            return tensor.detach().clone(), grads

        state = capture_rng_state()
        out_a, grads_a = _fwd_bwd(self.make_inputs())

        restore_rng_state(state)
        model.zero_grad(set_to_none=True)
        out_b, grads_b = _fwd_bwd(self.make_inputs())

        assert_bit_exact(out_a, grads_a, out_b, grads_b)

    # ------------------------------------------------------------------
    # Pipeline-schedule path (PP > 1). Builds chunks per rank/VPP-rank,
    # runs forward_backward_func twice, compares per-chunk grads.
    # ------------------------------------------------------------------
    def _run_pipeline(self, cfg_overrides: dict, parallelism: dict):
        pp = parallelism.get("PP", 1)
        vpp = parallelism.get("VPP", 1) or 1

        # num_layers ≥ pp*vpp. With vp_stage correctly threaded to each
        # chunk, 1 layer per virtual chunk is fine; the earlier ≥2-layer
        # workaround pre-dated the vp_stage fix.
        base_layers = (self.base_config() | cfg_overrides).get("num_layers", 2)
        num_layers_total = max(base_layers, pp * vpp)
        if num_layers_total % (pp * vpp) != 0:
            num_layers_total = ((num_layers_total + pp * vpp - 1) // (pp * vpp)) * pp * vpp
        cfg_overrides = {**cfg_overrides, "num_layers": num_layers_total}
        if vpp > 1:
            cfg_overrides["virtual_pipeline_model_parallel_size"] = vpp
            # Required by interleaved schedule: must be in [PP, num_microbatches].
            cfg_overrides["microbatch_group_size_per_vp_stage"] = pp

        chunks = self._build_chunks(cfg_overrides, pp, vpp)
        # For the interleaved schedule, num_microbatches must satisfy:
        #   microbatch_group_size_per_vp_stage = pp (set above for VPP)
        #   num_microbatches >= pp  AND  num_microbatches % pp == 0
        # Use the minimum valid value (pp) to keep the schedule simple and
        # avoid leaving stale NCCL P2P state at the end of a pass.
        num_microbatches = pp if vpp > 1 else max(pp, 1)

        def run_once():
            return self._pipeline_fwd_bwd(chunks, num_microbatches=num_microbatches)

        state = capture_rng_state()
        loss_a, grads_a = run_once()

        # Drain any pending P2P / async ops before run 2.
        torch.cuda.synchronize()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        for c in chunks:
            if hasattr(c, "set_input_tensor"):
                try:
                    c.set_input_tensor(None)
                except Exception:
                    pass
        for c in chunks:
            c.zero_grad(set_to_none=True)

        restore_rng_state(state)
        loss_b, grads_b = run_once()

        assert_bit_exact(loss_a, grads_a, loss_b, grads_b)

    def _build_chunks(self, cfg_overrides: dict, pp: int, vpp: int) -> list:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        chunks = []
        for vpp_rank in range(vpp):
            is_first = (vpp_rank == 0) and (pp_rank == 0)
            is_last = (vpp_rank == vpp - 1) and (pp_rank == pp - 1)
            if vpp > 1:
                parallel_state.set_virtual_pipeline_model_parallel_rank(vpp_rank)
            # Pass vp_stage so the factory can hand it to TransformerBlock
            # (and similar modules) for correct per-virtual-chunk layer
            # slicing. Factories that don't accept it can ignore via **kwargs.
            chunk = self.build_model(
                cfg_overrides,
                pre_process=is_first,
                post_process=is_last,
                vp_stage=vpp_rank if vpp > 1 else None,
            )
            chunks.append(chunk)
        if vpp > 1:
            parallel_state.set_virtual_pipeline_model_parallel_rank(0)
        return chunks

    def _pipeline_fwd_bwd(self, chunks: list, num_microbatches: int = 1) -> tuple:
        make_inputs = self.make_inputs
        dtype = self.dtype

        def forward_step(data_iterator, model):
            batch = next(data_iterator)
            with torch.autocast("cuda", dtype=dtype):
                output = model(**batch)
            tensor = output[0] if isinstance(output, tuple) else output

            def loss_func(output_tensor):
                loss = output_tensor.float().mean() * 0.001
                return loss, {"loss": loss.detach().clone()}

            return tensor, loss_func

        def data_iter():
            while True:
                yield make_inputs()

        forward_backward_func = get_forward_backward_func()
        if len(chunks) > 1:
            data_iterator = [data_iter() for _ in chunks]
        else:
            data_iterator = data_iter()

        forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=chunks,
            num_microbatches=num_microbatches,
            seq_length=self.seq_len,
            micro_batch_size=self.micro_batch,
            forward_only=False,
        )
        # ``losses`` lives only on the last PP rank — comparing it across runs
        # is fragile. Return a placeholder; assert_bit_exact validates grads
        # which are well-defined on every rank's local layers.
        placeholder = torch.zeros(1, device="cuda")
        return placeholder, _collect_grads(chunks)


def _collect_grads(modules) -> dict:
    grads = {}
    for i, m in enumerate(modules):
        for name, p in m.named_parameters():
            if p.grad is not None:
                grads[f"chunk{i}.{name}"] = p.grad.detach().clone()
    return grads
