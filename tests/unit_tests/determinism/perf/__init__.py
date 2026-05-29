# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Perf-leaderboard tests for ``--deterministic-mode``.

Measures speed + peak-memory cost of determinism. Parallelism IS in the
parametrize axis (``configs.PERF_PARALLELISM_CONFIGS``) because the
TP × EP composite is where the most interesting determinism cost lives
(grouped-GEMM on TP-sharded MoE experts).

The perf bucket needs to measure BOTH deterministic and non-deterministic
kernel paths. Because ``CUBLAS_*`` / ``NVTE_*`` env vars are read at first
kernel call and cached, the only clean way to compare modes is to run
this subpackage twice — once with env vars set, once without. The
orchestration env var is ``DETERMINISM_PERF_MODE``:

    DETERMINISM_PERF_MODE=det     → env vars set
    DETERMINISM_PERF_MODE=nondet  → env vars NOT set
    (unset)                       → defaults to ``det``
"""

import os

if os.environ.get("DETERMINISM_PERF_MODE", "det") != "nondet":
    from megatron.training.determinism import set_determinism_env_vars

    # ``strict=True`` also pins ``NCCL_LAUNCH_RACE_FATAL=1`` so any latent
    # collective-ordering bug surfaces as a hard test failure rather than a
    # silent warning.
    set_determinism_env_vars(strict=True)
