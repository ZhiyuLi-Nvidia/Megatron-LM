# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Determinism perf runner — mirror of ``moe_perf/__main__.py``.

Measures ``forward_ms`` / ``backward_ms`` / ``max_allocated_bytes`` via
``cuda.Event`` and asserts against ``baseline.json`` with a 1.02×
regression bound. The top ``_OUTLIER_TRIM`` samples are trimmed.

The active-iter loop is bracketed by ``cudaProfilerStart`` /
``cudaProfilerStop`` so that an enclosing
``nsys profile --capture-range=cudaProfilerApi`` only captures the
measured window (warmup is excluded). For per-mcore-module breakdown,
the SLURM driver wraps pytest in nsys and post-processes the
``nsys stats --report nvtx_sum`` CSV into the leaderboard — see the
recipe comment at the bottom of ``test_gpt_perf.py``.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import pytest
import torch

from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.utils import configure_nvtx_profiling

DEFAULT_MAX_REGRESSION_RATIO = 1.02
# We trim top-N outliers before mean/std so single-iter scheduler hiccups
# (~10 ms on a 125 ms backward) don't move the gate. No variance check —
# the goal is regression detection, not measurement-quality QA.
_OUTLIER_TRIM = 3
UPDATE_BASELINES_ENV = "MEGATRON_UPDATE_PERF_BASELINES"


def current_perf_mode() -> str:
    """Return ``"det"`` or ``"nondet"``. The suite runs twice — once per
    mode — because the env vars pinning cuBLAS / TE to deterministic
    algorithms are sticky after the first kernel call."""
    m = os.environ.get("DETERMINISM_PERF_MODE", "det")
    if m not in ("det", "nondet"):
        raise ValueError(f"DETERMINISM_PERF_MODE must be 'det' or 'nondet', got {m!r}")
    return m


def read_perf_knobs() -> tuple[int, int]:
    """Return ``(warmup_iters, measure_iters)`` from env vars."""
    return (
        int(os.environ.get("DETERMINISM_PERF_WARMUP", "5")),
        int(os.environ.get("DETERMINISM_PERF_ITERS", "20")),
    )


def load_baselines(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def persist_baselines(path: Path, data: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")


def assert_within_baseline(
    case_name: str, metrics: Mapping[str, Any], baselines: Dict[str, Dict[str, float]]
) -> None:
    baseline = baselines.get(case_name)
    if baseline is None:
        pytest.fail(
            f"Missing baseline for {case_name!r}. "
            f"Set {UPDATE_BASELINES_ENV}=1 to record it."
        )

    max_ratio = baseline.get("max_regression_ratio", DEFAULT_MAX_REGRESSION_RATIO)
    fwd, bwd, mem = metrics["forward_ms"], metrics["backward_ms"], metrics["max_allocated_bytes"]

    assert fwd <= baseline["forward_ms"] * max_ratio, (
        f"Forward regressed for {case_name}: {fwd:.3f} ms "
        f"(limit {baseline['forward_ms'] * max_ratio:.3f} ms)."
    )
    assert bwd <= baseline["backward_ms"] * max_ratio, (
        f"Backward regressed for {case_name}: {bwd:.3f} ms "
        f"(limit {baseline['backward_ms'] * max_ratio:.3f} ms)."
    )
    assert mem <= baseline["max_allocated_bytes"] * max_ratio, (
        f"Peak memory regressed for {case_name}: "
        f"{mem / (1024 ** 2):.3f} MiB "
        f"(limit {baseline['max_allocated_bytes'] * max_ratio / (1024 ** 2):.3f} MiB)."
    )


def maybe_update_baseline(
    case_name: str,
    metrics: Dict[str, float],
    baselines: Dict[str, Dict[str, float]],
    baselines_path: Path,
) -> None:
    baselines[case_name] = {
        "forward_ms": metrics["forward_ms"],
        "backward_ms": metrics["backward_ms"],
        "max_allocated_bytes": metrics["max_allocated_bytes"],
        "max_regression_ratio": DEFAULT_MAX_REGRESSION_RATIO,
    }
    persist_baselines(baselines_path, baselines)


class PerfRunner:
    """Build a model, run warmup + measured iters, return metrics.

    Args:
        build_model: ``(overrides) -> nn.Module``. Receives the merged
            cell-overrides dict including ``deterministic_mode``.
        make_inputs: zero-arg callable returning the ``model(**inputs)`` kwargs.
    """

    def __init__(
        self,
        build_model: Callable[[dict], torch.nn.Module],
        make_inputs: Callable[[], dict],
    ):
        self.build_model = build_model
        self.make_inputs = make_inputs

    def measure(
        self,
        cell_overrides: dict,
        deterministic: bool,
        *,
        warmup: int,
        active: int,
    ) -> Dict[str, Any]:
        torch.manual_seed(7)
        model_parallel_cuda_manual_seed(123)
        model = self.build_model({**cell_overrides, "deterministic_mode": deterministic})
        optimizer = torch.optim.SGD(model.parameters(), lr=0.0)

        for _ in range(warmup):
            self._step(model, optimizer, record=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        # cudaProfilerStart/Stop bracket the measured window so an
        # enclosing ``nsys profile --capture-range=cudaProfilerApi`` only
        # collects events for these iters. No-op when not running under
        # nsys. mcore's ``nvtx_range_push/pop`` are off by default (gated
        # by ``_nvtx_enabled``); enable them so the NVTX domain nsys
        # captures includes mcore's ``self_attention`` / ``mlp`` / etc.
        # boundaries, not just TE / NCCL / CCCL internal ranges.
        #
        # The try block starts BEFORE cudaProfilerStart so an exception
        # in either setup call still triggers cleanup of whichever
        # setup did complete. NVTX flag is unconditionally disabled in
        # finally; current callers all enter measure() with NVTX off,
        # so a save/restore would be over-engineering — if that changes,
        # snapshot ``utils._nvtx_enabled`` here and restore it instead.
        fwd_ts: list[float] = []
        bwd_ts: list[float] = []
        peak_bytes: list[float] = []
        configure_nvtx_profiling(True)
        try:
            torch.cuda.cudart().cudaProfilerStart()
            try:
                for _ in range(active):
                    fwd_ms, bwd_ms = self._step(model, optimizer, record=True)
                    fwd_ts.append(fwd_ms)
                    bwd_ts.append(bwd_ms)
                    peak_bytes.append(torch.cuda.max_memory_allocated())
                    torch.cuda.reset_peak_memory_stats()
            finally:
                torch.cuda.cudart().cudaProfilerStop()
        finally:
            configure_nvtx_profiling(False)

        # Drop top-N outliers before mean — a single scheduler hiccup
        # (~10 ms on a 125 ms backward) would otherwise dominate.
        fwd_trimmed = sorted(fwd_ts)[:-_OUTLIER_TRIM] if len(fwd_ts) > _OUTLIER_TRIM else fwd_ts
        bwd_trimmed = sorted(bwd_ts)[:-_OUTLIER_TRIM] if len(bwd_ts) > _OUTLIER_TRIM else bwd_ts
        return {
            "forward_ms": statistics.mean(fwd_trimmed),
            "backward_ms": statistics.mean(bwd_trimmed),
            "max_allocated_bytes": statistics.mean(peak_bytes),
            "forward_timings": fwd_ts,
            "backward_timings": bwd_ts,
        }

    def _step(self, model, optimizer, *, record):
        if record:
            fwd_start = torch.cuda.Event(enable_timing=True)
            fwd_end = torch.cuda.Event(enable_timing=True)
            bwd_start = torch.cuda.Event(enable_timing=True)
            bwd_end = torch.cuda.Event(enable_timing=True)

        inputs = self.make_inputs()
        if record:
            fwd_start.record()
        out = model(**inputs)
        out = out[0] if isinstance(out, tuple) else out
        loss = (out * out).mean()
        if record:
            fwd_end.record()
            bwd_start.record()
        loss.backward()
        if record:
            bwd_end.record()
        optimizer.step()
        model.zero_grad(set_to_none=True)

        if record:
            torch.cuda.synchronize()
            return (fwd_start.elapsed_time(fwd_end), bwd_start.elapsed_time(bwd_end))


def record_to_leaderboard(
    leaderboard: list, cell_id: str, mode: str, metrics: Dict[str, Any]
) -> None:
    """Append TOTAL.fwd / TOTAL.bwd rows for the stdout printout (rank 0 only)."""
    if int(os.environ.get("RANK", "0")) != 0:
        return
    mem_mb = metrics["max_allocated_bytes"] / (1024 * 1024)
    common = {"kind": f"perf-{mode}", "mem_mb": mem_mb}
    leaderboard.append({**common, "cell": f"{cell_id}::TOTAL.fwd",
                        "ms_per_iter": metrics["forward_ms"]})
    leaderboard.append({**common, "cell": f"{cell_id}::TOTAL.bwd",
                        "ms_per_iter": metrics["backward_ms"]})
