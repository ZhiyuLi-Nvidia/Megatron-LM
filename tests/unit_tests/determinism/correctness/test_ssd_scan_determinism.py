# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""The Mamba2 SSD selective-scan kernel is deterministic ONLY under the deterministic-mode
autotune pin; without it two independent processes can diverge (the a55b break).

Mechanism: the scan is Triton-autotuned; independent processes can benchmark and pick
different configs -> different reduction order -> ~1e-5 divergence. ``--deterministic-mode``
makes ``megatron/core/ssm/ops/determinism.py::autotune_configs`` collapse to one fixed
config, so every process matches. Full forensics + a55b context: ``det_boot/FINDINGS.md``.

What this file asserts (via two independent ``torchrun`` jobs compared by fp64 fingerprint;
the suite's same-process ``BitExactRunner`` masks this because run 2 reuses run 1's warm
autotune cache):
  * det ON  -> bit-identical across runs (hard guard).
  * det OFF -> may diverge (xfail demo; XPASS when the break reproduces, autotune-cache-dependent).

Run: ``uv run --no-sync python -m pytest <this file> -s -v`` on a GPU node.
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


# a55b parallelism: TP=2. The Mamba mixer shards heads across TP, so one rank runs the scan
# on _NHEADS_TOTAL / TP heads. Geometry is a numerically-stable multi-chunk scan large
# enough to exercise the chunk-state / state-passing reductions where cross-process autotune
# divergence appears (an 8-chunk, 16-head scan reproduces it; a 4-chunk toy scan does not).
_TP = 2
_NHEADS_TOTAL = 16
_CHUNK = 128
_SEQLEN = 1024  # 8 chunks
_HEADDIM = 64
_NGROUPS = 1
_DSTATE = 128
_SEED = 1234


def _scan_output_fingerprint(nheads: int, seed: int) -> tuple[float, float]:
    """Run the varlen SSD scan once on fixed seeded, numerically-stable inputs; return fp64
    sums of the token output and the final SSM states.

    Inputs are seeded, so any change in the returned pair between runs is the kernel's own
    non-determinism. ``A`` is a negative decay and ``dt`` is small (matching real Mamba) so
    the scan stays finite instead of overflowing to NaN.
    """
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    nchunks = _SEQLEN // _CHUNK
    x = torch.randn(_SEQLEN, nheads, _HEADDIM, device=dev, dtype=torch.float32)
    dt = torch.rand(_SEQLEN, nheads, device=dev, dtype=torch.float32) * 0.05
    A = -torch.rand(nheads, device=dev, dtype=torch.float32) - 0.5
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


def _run_independent_job(tp: int, deterministic: bool) -> str:
    """Launch one independent ``torchrun`` job at the given TP and determinism setting;
    return its printed scan fingerprint. A fresh process => fresh Triton autotune, so this
    is the production A/A condition (two of these compared)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={tp}",
            __file__,
            "--worker",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "SSD_DET": "1" if deterministic else "0"},
    )
    assert proc.returncode == 0, f"torchrun worker failed:\n{proc.stderr}"
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("FINGERPRINT ")]
    assert lines, f"no fingerprint printed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    return lines[-1]


def _two_runs(tp: int, deterministic: bool) -> tuple[str, str]:
    """Fingerprints of two *independent* torchrun jobs (skips if too few GPUs)."""
    if torch.cuda.device_count() < tp:
        pytest.skip(f"needs {tp} GPU(s) for TP={tp}")
    return _run_independent_job(tp, deterministic), _run_independent_job(tp, deterministic)


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_SSD_KERNEL, reason="SSD ops (Triton 3+) unavailable")
@pytest.mark.parametrize("tp", [1, 2])
def test_scan_deterministic_with_deterministic_mode(tp):
    """With deterministic mode ON, two independent TP={tp} runs are BIT-IDENTICAL — the fix.
    ``autotune_configs`` pins the scan to one Triton config, so every process matches."""
    run_a, run_b = _two_runs(tp, deterministic=True)
    print(f"\n[SSD-scan, det ON] TP={tp}\n  run1={run_a}\n  run2={run_b}")
    assert run_a == run_b, (
        f"deterministic mode must make the SSD scan bit-reproducible across independent "
        f"TP={tp} runs, but they differed:\n  run1={run_a}\n  run2={run_b}"
    )


@pytest.mark.internal
@pytest.mark.xfail(
    strict=False,
    reason="Without the autotune pin the scan's cross-run reproducibility is Triton-autotune-"
    "cache-dependent: it diverges with cold/independent autotune (the a55b A/A condition) and "
    "agrees once the config is cached/shared. Demonstration, not a hard guarantee — XPASS when "
    "the break reproduces, XFAIL when the cache masks it.",
)
@pytest.mark.skipif(not HAVE_SSD_KERNEL, reason="SSD ops (Triton 3+) unavailable")
def test_scan_may_diverge_without_deterministic_mode():
    """Demonstration of the break at a55b's TP: with deterministic mode OFF, two independent
    runs of the SSD scan CAN diverge (different autotune config -> different reduction)."""
    run_a, run_b = _two_runs(_TP, deterministic=False)
    print(f"\n[SSD-scan, det OFF] TP={_TP}\n  run1={run_a}\n  run2={run_b}")
    assert run_a != run_b, (
        f"scan agreed across independent TP={_TP} runs with deterministic mode OFF "
        f"({run_a}) — autotune cache masked the divergence here (see xfail reason)."
    )


@pytest.mark.internal
@pytest.mark.skipif(not HAVE_SSD_DET, reason="ssm.ops.determinism unavailable")
def test_autotune_configs_deterministic_selection():
    """Deterministic mode must collapse Triton autotuning to a single, cheapest config —
    the mechanism that makes the scan reproducible. Pure-Python; no GPU."""

    class _Cfg:
        def __init__(self, block, warps, stages):
            self.kwargs = {"BLOCK_SIZE_M": block}
            self.num_warps = warps
            self.num_stages = stages

    configs = [_Cfg(64, 4, 2), _Cfg(32, 2, 1), _Cfg(128, 8, 3)]
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
        # cheapest = min(block_product * stages, then warps) -> _Cfg(32, 2, 1).
        assert picked[0].kwargs["BLOCK_SIZE_M"] == 32
    finally:
        ssd_det.set_deterministic_mode(None)
        os.environ.update(saved)
        if cache is not None:
            os.environ["TRITON_CACHE_AUTOTUNING"] = cache


def _worker() -> None:
    """torchrun child: run the scan on this rank's head-shard at the SSD_DET setting,
    all-reduce the fingerprint across the TP group, and print it once from rank 0."""
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    try:
        deterministic = os.environ.get("SSD_DET", "1") == "1"
        ssd_det.set_deterministic_mode(deterministic)
        torch.use_deterministic_algorithms(deterministic, warn_only=True)
        nheads_local = _NHEADS_TOTAL // world  # a55b: heads sharded across TP
        a, b = _scan_output_fingerprint(nheads=nheads_local, seed=_SEED + rank)
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
