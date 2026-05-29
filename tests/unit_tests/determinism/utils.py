# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Shared helpers for per-module determinism tests.

The env vars required for bit-exact reproducibility are set in each
subpackage's ``__init__.py`` (``correctness/`` always; ``perf/`` only when
``DETERMINISM_PERF_MODE != 'nondet'``) so they take effect on package
import, before any cuBLAS / TE call inside a test module.
"""

import random
from contextlib import contextmanager

import numpy as np
import torch

from megatron.core.timers import Timer


class _Elapsed:
    """Simple holder so ``with cuda_timer() as t:`` users can read
    ``t.elapsed_ms`` after the block."""

    def __init__(self) -> None:
        self.elapsed_ms: float = 0.0


@contextmanager
def cuda_timer(name: str = "cuda_timer"):
    """CUDA-synchronised wall-clock timer as a Python context manager.

    Wraps ``megatron.core.timers.Timer`` (the same primitive Megatron training
    uses for step timing) — ``.start()`` and ``.stop()`` both call
    ``torch.cuda.synchronize()`` so the measured window is properly bracketed
    even when GPU work is queued asynchronously.

    Usage:
        with cuda_timer("my-step") as t:
            ... gpu work ...
        # t.elapsed_ms is the elapsed wall time in milliseconds.
    """
    # NOTE: do NOT call ``Timer.elapsed(reset=False)`` — its restart-if-running
    # branch (megatron/core/timers.py:208-209) issues an extra
    # ``cuda.synchronize`` and leaves the discarded Timer in ``_started=True``,
    # inflating measurement asymmetrically across the fwd/bwd/opt blocks.
    timer = Timer(name)
    box = _Elapsed()
    timer.start()
    try:
        yield box
    finally:
        timer.stop()
        box.elapsed_ms = timer.elapsed(reset=False) * 1000.0

try:
    # Public-by-import helper used by PyTorch's own test_cuda.py to convert
    # milliseconds to device-cycle counts for torch.cuda._sleep.
    from torch.testing._internal.common_utils import get_cycles_per_ms
except ImportError:  # pragma: no cover — fallback only if PyTorch internals move
    def get_cycles_per_ms() -> float:
        # Rough lower bound: H100 boosts to ~1.8 GHz → ~1.8M cycles/ms. Picking
        # 1M is conservative — the jitter will be a bit shorter than requested,
        # not longer, which keeps test runtime bounded.
        return 1_000_000.0


def capture_rng_state() -> dict:
    """Snapshot every RNG that the framework consumes during a fwd+bwd pass.

    Mirrors ``RerunStateMachine._save_state`` in
    ``megatron/core/rerun_state_machine.py``. Also captures Megatron's own
    ``CudaRNGStatesTracker`` (used for TP-aware dropout), which advances
    independently of ``torch.cuda``'s RNG when any layer calls
    ``get_cuda_rng_tracker().fork()``.
    """
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {
        "random": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(),
        "mpu_tracker": get_cuda_rng_tracker().get_states(),
    }


def restore_rng_state(state: dict) -> None:
    """Inverse of ``capture_rng_state``."""
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    random.setstate(state["random"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state(state["torch_cuda"])
    if "mpu_tracker" in state:
        get_cuda_rng_tracker().set_states(state["mpu_tracker"])


def _strict_equal_with_nan(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Element-wise equality where NaN at the same position counts as equal.

    Plain ``torch.equal`` returns False for any NaN-vs-NaN comparison, which
    is the correct semantics for value equality but wrong for *determinism*
    where we only care that two runs produced bit-identical outputs — same
    NaN pattern included.
    """
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    eq = (a == b) | (a.isnan() & b.isnan())
    return bool(eq.all().item())


def assert_bit_exact(out_a, grads_a, out_b, grads_b) -> None:
    """Assert two (output, grad-dict) pairs are bit-exact equal."""
    assert _strict_equal_with_nan(out_a, out_b), "Outputs differ between deterministic runs"
    assert grads_a.keys() == grads_b.keys(), "Grad keys differ between runs"
    for name in grads_a:
        assert _strict_equal_with_nan(
            grads_a[name], grads_b[name]
        ), f"Grad mismatch for {name}"


class RacingStreams:
    """Context manager that launches sustained side-stream GEMMs to perturb
    default-stream scheduling.

    Sizing rationale: ``gemm_size=2048`` in bf16 → ~17 GFlops per matmul; the
    inner loop chains 200 of them so the side work runs in the millisecond
    range and overlaps a small layer's fwd+bwd. ``num_streams`` defaults to
    4 so multiple side streams compete with the default stream at once. The
    side GEMMs feed only into ``_noise`` and never into the module's output,
    so a correctly deterministic implementation cannot be affected — only
    incorrect ones will flake.
    """

    def __init__(self, num_streams: int = 4, gemm_size: int = 2048, num_iters: int = 200):
        self.num_streams = num_streams
        self.gemm_size = gemm_size
        self.num_iters = num_iters
        self.streams: list[torch.cuda.Stream] = []
        self._noise: list[torch.Tensor] = []

    def __enter__(self):
        for _ in range(self.num_streams):
            self.streams.append(torch.cuda.Stream())
        for s in self.streams:
            with torch.cuda.stream(s):
                a = torch.randn(
                    self.gemm_size, self.gemm_size, device="cuda", dtype=torch.bfloat16
                )
                b = torch.randn(
                    self.gemm_size, self.gemm_size, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(self.num_iters):
                    a = a @ b
                self._noise.append(a)
        return self

    def __exit__(self, *args):
        for s in self.streams:
            s.synchronize()
        self.streams.clear()
        self._noise.clear()


class CudaSleepJitter:
    """Inject rank-asymmetric ``torch.cuda._sleep`` calls on every submodule
    forward.

    Same pattern PyTorch's own ``test/test_cuda.py`` uses to stress stream
    ordering: ``_sleep`` is a no-op kernel that spins for a fixed device-cycle
    count, so it perturbs scheduling without touching memory. Pairing this
    with ``NCCL_LAUNCH_RACE_FATAL=1`` turns any latent collective-ordering bug
    into a hard test failure.

    Determinism semantics: a fresh ``torch.Generator`` is seeded per-rank in
    ``__enter__``, so two consecutive ``with`` blocks see identical jitter
    sequences per rank (different across ranks). Re-create the context
    manager around each run rather than reusing one across runs.
    """

    def __init__(
        self,
        module: torch.nn.Module,
        max_us_per_hook: int = 200,
        seed: int = 0xCAFE,
    ):
        self._module = module
        self._max_us = max_us_per_hook
        self._seed = seed
        self._handles: list = []

    def __enter__(self):
        rank = (
            torch.distributed.get_rank()
            if torch.distributed.is_initialized()
            else 0
        )
        gen = torch.Generator()
        gen.manual_seed(self._seed + rank)

        # microseconds → ms → cycles
        max_cycles = int((self._max_us / 1000.0) * get_cycles_per_ms())

        def _hook(_mod, _args, _max=max_cycles, _gen=gen):
            if _max <= 0:
                return
            cycles = int(torch.randint(0, _max + 1, (1,), generator=_gen).item())
            if cycles > 0:
                torch.cuda._sleep(cycles)

        for sub in self._module.modules():
            self._handles.append(sub.register_forward_pre_hook(_hook))
        return self

    def __exit__(self, *args):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def maybe_fsdp_wrap(model: torch.nn.Module, parallelism: dict) -> torch.nn.Module:
    """If ``parallelism["FSDP"] > 1``, wrap ``model`` with Megatron-FSDP.

    Uses the production path from ``megatron/training/training.py``: the
    ``FullyShardedDataParallel`` adapter from ``mcore_fsdp_adapter``, with the
    ``ProcessGroupCollection`` derived from the current ``parallel_state``.
    This means TP/PP/CP/EP groups already initialised by
    ``Utils.initialize_model_parallel`` are honoured automatically — FSDP
    just shards along the DP dimension that ``parallel_state`` exposes.
    """
    if parallelism.get("FSDP", 1) <= 1:
        return model

    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.distributed.fsdp.mcore_fsdp_adapter import FullyShardedDataParallel
    from megatron.core.process_groups_config import ProcessGroupCollection

    pg_collection = ProcessGroupCollection.use_mpu_process_groups()
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=False,  # determinism — disable async overlap
        overlap_param_gather=False,
        use_distributed_optimizer=True,
        bucket_size=40_000_000,
    )
    config = getattr(model, "config", None)
    return FullyShardedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=model,
        pg_collection=pg_collection,
    )


def make_forward_step_func(make_inputs_dict, autocast_dtype: torch.dtype = torch.bfloat16):
    """Return a ``forward_step_func`` suitable for ``get_forward_backward_func``.

    ``make_inputs_dict`` is a no-arg callable producing the kwargs dict for
    ``model(...)`` for the *current* microbatch. The loss is ``logits.pow(2).mean()``
    so the determinism check has a non-trivial gradient signal.
    """

    def forward_step(data_iterator, model):
        batch = next(data_iterator)
        with torch.autocast("cuda", dtype=autocast_dtype):
            output = model(**batch)

        def loss_func(output_tensor):
            loss = output_tensor.float().pow(2).mean()
            return loss, {"loss": loss.detach().clone()}

        return output, loss_func

    return forward_step


def infinite_iter(make_inputs_dict):
    """Yield ``make_inputs_dict()`` forever — used as ``data_iterator`` for
    ``get_forward_backward_func``."""
    while True:
        yield make_inputs_dict()
