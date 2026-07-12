# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Repro: the Mamba2 SSD selective-scan fused kernel is non-deterministic (at a55b's TP).

The a55b HybridEP determinism break was localized (op-level A/A trace) to
``mamba_chunk_scan_combined_varlen`` in ``megatron/core/ssm/ops/ssd_combined.py``:
bit-identical inputs, different SSM-state output across two independent runs, which then
compounds into the divergent loss. The existing determinism suite never catches this
because (1) no SSM model is in its matrix and (2) ``BitExactRunner._two_runs`` reruns
*in the same process under a restored RNG state*, so Triton's cached autotune config
carries over between the two runs and masks the divergence.

a55b runs **TP=2**, so each rank's Mamba mixer runs the scan on a head-shard
(``nheads / TP``). These tests reproduce at that same geometry — the scan is TP-local
(no cross-rank comm inside it), so "TP=2" means each rank runs the kernel on
``_NHEADS_TOTAL / 2`` heads:

* ``test_scan_deterministic_across_tp2_processes`` — two *independent*
  ``torchrun --nproc_per_node=2`` jobs (TP=2). Each job autotunes the scan on its own;
  comparing their all-reduced output fingerprint is the production A/A condition the
  same-process harness cannot reproduce. **This is the faithful repro.** (Needs 2 GPUs.)
* ``test_scan_deterministic_same_process`` — cheap 1-GPU guard: relaunch on the TP=2
  per-rank shard in one process (catches the atomic-accumulation component).
* ``test_autotune_configs_deterministic_selection`` — unit-covers
  ``megatron/core/ssm/ops/determinism.py`` (previously referenced by no test).

The two kernel tests are ``xfail(strict=False)``: they document a live bug and flip to
XPASS once the deterministic scan path (the currently-dead ``alloc_tile_workspace`` /
``finalize_tile_workspace`` atomic-free reduction in ``determinism.py``) is wired in.

Run on a GPU node::

    pytest tests/unit_tests/determinism/correctness/test_ssd_scan_determinism.py
"""

import os
import subprocess
import sys

import pytest
import torch

# determinism.py is lightweight (guarded triton import) and testable CPU-only.
try:
    from megatron.core.ssm.ops import determinism as ssd_det

    HAVE_SSD_DET = True
except Exception:
    HAVE_SSD_DET = False

# The fused kernel itself needs Triton 3+.
try:
    from megatron.core.ssm.ops.ssd_combined import mamba_chunk_scan_combined_varlen

    HAVE_SSD_KERNEL = True
except Exception:
    HAVE_SSD_KERNEL = False


# a55b parallelism: TP=2. The Mamba mixer shards heads across TP, so one rank runs the
# scan on _NHEADS_TOTAL / _TP heads. Geometry mirrors the a55b Mamba2 mixer at toy scale;
# a multi-chunk (4-chunk) scan so the chunk-state / state-passing reductions actually run.
_TP = 2
_NHEADS_TOTAL = 8
_NHEADS_LOCAL = _NHEADS_TOTAL // _TP  # per-rank head-shard under TP=2
_CHUNK = 16
_SEQLEN = 64  # 4 chunks
_HEADDIM = 64
_NGROUPS = 1
_DSTATE = 128
_SEED = 1234


def _scan_output_fingerprint(nheads: int, seed: int) -> tuple[float, float]:
    """Run the varlen SSD scan once on fixed seeded inputs of ``nheads`` heads; return
    fp64 sums of the token output and the final SSM states.

    Inputs are seeded, so any change in the returned pair between runs is the kernel's
    own non-determinism (the exact two tensors that diverged in the a55b trace).
    """
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    nchunks = _SEQLEN // _CHUNK
    x = torch.randn(_SEQLEN, nheads, _HEADDIM, device=dev, dtype=torch.float32)
    dt = torch.randn(_SEQLEN, nheads, device=dev, dtype=torch.float32)
    A = torch.randn(nheads, device=dev, dtype=torch.float32)
    B = torch.randn(_SEQLEN, _NGROUPS, _DSTATE, device=dev, dtype=torch.float32)
    C = torch.randn(_SEQLEN, _NGROUPS, _DSTATE, device=dev, dtype=torch.float32)
    out = torch.empty(_SEQLEN, nheads, _HEADDIM, device=dev, dtype=torch.float32)
    cu_chunk_seqlens = torch.arange(0, _SEQLEN + 1, _CHUNK, dtype=torch.int32, device=dev)
    last_chunk_indices = torch.tensor([nchunks - 1], dtype=torch.int64, device=dev)
    seq_idx = torch.zeros(nchunks, dtype=torch.int32, device=dev)

    states = mamba_chunk_scan_combined_varlen(
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        chunk_size=_CHUNK,
        cu_chunk_seqlens=cu_chunk_seqlens,
        last_chunk_indices=last_chunk_indices,
        seq_idx=seq_idx,
        out=out,
    )
    torch.cuda.synchronize()
    return out.double().sum().item(), states.double().sum().item()


@pytest.mark.internal
@pytest.mark.xfail(
    strict=False,
    reason="SSD scan diverges across independent TP=2 runs (independent autotune / atomic "
    "reductions) — a55b FINDINGS; XPASS once the deterministic scan path is wired.",
)
@pytest.mark.skipif(not HAVE_SSD_KERNEL, reason="SSD ops (Triton 3+) unavailable")
@pytest.mark.skipif(torch.cuda.device_count() < _TP, reason=f"needs {_TP} GPUs for TP={_TP}")
def test_scan_deterministic_across_tp2_processes():
    """Two independent TP=2 jobs must produce the same all-reduced scan fingerprint — the
    production A/A condition the same-process harness can't reproduce (its cached autotune
    config carries over)."""

    def run_once() -> str:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={_TP}",
                __file__,
                "--worker",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "MAMBA_DETERMINISTIC": "1"},
        )
        assert proc.returncode == 0, f"torchrun worker failed:\n{proc.stderr}"
        lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("FINGERPRINT ")]
        assert lines, f"no fingerprint printed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        return lines[-1]

    run_a = run_once()
    run_b = run_once()
    assert run_a == run_b, (
        "SSD scan fingerprint differs across two independent TP=2 runs:\n"
        f"  run1={run_a}\n  run2={run_b}"
    )


@pytest.mark.internal
@pytest.mark.xfail(
    strict=False,
    reason="SSD scan is non-deterministic on relaunch until the atomic-free tile reduction "
    "in ssm/ops/determinism.py is wired in (a55b FINDINGS); XPASS once fixed.",
)
@pytest.mark.skipif(not HAVE_SSD_KERNEL, reason="SSD ops (Triton 3+) unavailable")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_scan_deterministic_same_process():
    """Cheap 1-GPU guard: the scan on the TP=2 per-rank shard must be bit-identical when
    relaunched on identical inputs."""
    ssd_det.set_deterministic_mode(True)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        ref = _scan_output_fingerprint(nheads=_NHEADS_LOCAL, seed=_SEED)
        for i in range(8):
            got = _scan_output_fingerprint(nheads=_NHEADS_LOCAL, seed=_SEED)
            assert got == ref, (
                f"SSD scan output changed on relaunch {i} within one process: "
                f"ref={ref} got={got}"
            )
    finally:
        ssd_det.set_deterministic_mode(None)


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_SSD_DET, reason="ssm.ops.determinism unavailable")
def test_autotune_configs_deterministic_selection():
    """Deterministic mode must collapse Triton autotuning to a single, cheapest config
    (else per-run autotune is itself a non-determinism source). Pure-Python; no GPU."""

    class _Cfg:
        def __init__(self, block, warps, stages):
            self.kwargs = {"BLOCK_SIZE_M": block}
            self.num_warps = warps
            self.num_stages = stages

    configs = [_Cfg(64, 4, 2), _Cfg(32, 2, 1), _Cfg(128, 8, 3)]
    # Neutralize the env knobs so we exercise the default cheapest-config path.
    saved = {k: os.environ.pop(k) for k in list(os.environ) if k.startswith("TRITON_AUTOTUNE_")}
    cache = os.environ.pop("TRITON_CACHE_AUTOTUNING", None)
    try:
        ssd_det.set_deterministic_mode(False)
        assert len(ssd_det.autotune_configs(list(configs))) == len(configs), (
            "non-deterministic mode must leave the autotune config list untouched"
        )

        ssd_det.set_deterministic_mode(True)
        picked = ssd_det.autotune_configs(list(configs))
        assert len(picked) == 1, "deterministic mode must collapse autotune to one config"
        # cheapest = min(block_product * stages, then warps) → _Cfg(32, 2, 1).
        assert picked[0].kwargs["BLOCK_SIZE_M"] == 32
    finally:
        ssd_det.set_deterministic_mode(None)
        os.environ.update(saved)
        if cache is not None:
            os.environ["TRITON_CACHE_AUTOTUNING"] = cache


def _worker() -> None:
    """torchrun child (TP=_TP): run the scan on this rank's head-shard, all-reduce the
    fingerprint across the TP group, and print it once from rank 0."""
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        ssd_det.set_deterministic_mode(True)
        torch.use_deterministic_algorithms(True, warn_only=True)
        # Per-rank seed → each TP shard gets distinct but reproducible data (same across
        # the two jobs being compared). all-reduce combines the shards into one number.
        a, b = _scan_output_fingerprint(nheads=_NHEADS_LOCAL, seed=_SEED + rank)
        fp = torch.tensor([a, b], dtype=torch.float64, device="cuda")
        dist.all_reduce(fp, op=dist.ReduceOp.SUM)
        if rank == 0:
            print(f"FINGERPRINT {fp[0].item()!r} {fp[1].item()!r}")
        dist.barrier()
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
