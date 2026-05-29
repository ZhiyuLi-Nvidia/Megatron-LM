# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable helpers for enabling bit-exact-reproducible execution.

Mirrors the split that ``megatron-bridge`` uses for the same purpose:

* :func:`set_determinism_env_vars` — env-var setdefaults that must happen
  BEFORE the first cuBLAS / Transformer Engine kernel invocation in the
  process. Equivalent to bridge's
  ``PerfEnvPlugin._set_determinism_env_vars`` (``scripts/performance/perf_plugins.py``).
* :func:`apply_determinism_to_args` — config-level overrides applied to a
  parsed ``args`` Namespace. Equivalent to bridge's
  ``apply_determinism_overrides`` (``recipes/utils/determinism_utils.py``)
  but works on the ``args`` produced by Megatron-LM's argparser.

Callers:

* CLI training (``pretrain_*.py``) goes through ``validate_args`` which
  calls :func:`apply_determinism_to_args` when ``--deterministic-mode`` is
  passed.
* Test suite (``tests/unit_tests/determinism/``) and standalone profiling
  scripts call :func:`set_determinism_env_vars` directly at import time —
  they don't have an ``args`` Namespace.
"""

from __future__ import annotations

import os

import torch


def set_determinism_env_vars(*, strict: bool = False) -> None:
    """Populate env vars required for bit-exact reproducibility.

    Must be called BEFORE the first cuBLAS / Transformer Engine call in the
    process — TE captures ``NVTE_ALLOW_NONDETERMINISTIC_ALGO`` on first use,
    and cuBLAS reads ``CUBLAS_WORKSPACE_CONFIG`` at handle creation. Uses
    ``setdefault`` so any value the launcher has already set survives
    (e.g. ``NCCL_ALGO=Tree`` if the user picked a different deterministic
    algorithm).

    Args:
        strict: If True, also set ``NCCL_LAUNCH_RACE_FATAL=1`` — turns NCCL
            collective-launch races into fatal errors instead of silent
            warnings. Useful for test environments that want any
            non-determinism bug to surface as a hard failure; production
            training typically leaves this off.
    """
    os.environ.setdefault("NCCL_ALGO", "Ring")
    os.environ.setdefault("NVTE_ALLOW_NONDETERMINISTIC_ALGO", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if strict:
        os.environ.setdefault("NCCL_LAUNCH_RACE_FATAL", "1")


def apply_determinism_to_args(args) -> None:
    """Apply deterministic-mode overrides to a parsed-args Namespace.

    Idempotent. Performs (in this order):

    1. Validates the args Namespace against deterministic-mode constraints.
    2. Sets env vars via :func:`set_determinism_env_vars`.
    3. Forces ``tp_comm_overlap=False`` (non-deterministic NCCL collectives).
    4. Calls ``torch.use_deterministic_algorithms(True)``.

    Validation runs FIRST so a malformed args Namespace fails fast without
    leaving the process in a half-deterministic state (env vars set but
    torch global state untouched).
    """
    # 1. Validate args first — direct attribute access so a malformed
    #    Namespace (missing the field entirely) fails loudly rather than
    #    silently passing the check.
    assert (
        not args.cross_entropy_loss_fusion
    ), "Cross Entropy Fusion is currently not deterministic."

    # NCCL_ALGO sanity. Accepted values:
    #   * positive tokens from ``{Ring, Tree, CollnetDirect, CollnetChain}`` —
    #     deterministic AllReduce algos. Comma-separated lists are valid
    #     NCCL syntax (try first, fall back) and accepted as subsets.
    #   * ``^NVLS`` — NCCL exclusion syntax. With NVLS excluded, NCCL falls
    #     back to one of the remaining deterministic algos. This is a
    #     legitimate way to enforce determinism without pinning a specific
    #     algo. Other ``^XXX`` forms (e.g. ``^Tree``) are NOT accepted
    #     because they could still let NCCL pick NVLS.
    accepted_tokens = {"Ring", "Tree", "CollnetDirect", "CollnetChain", "^NVLS"}
    nccl_algo = os.environ.get("NCCL_ALGO")
    if nccl_algo is not None:
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        assert tokens and all(t in accepted_tokens for t in tokens), (
            f"NCCL_ALGO={nccl_algo!r} must be a comma-separated subset of "
            f"{sorted(accepted_tokens)}."
        )

    # 2. Apply env vars after validation. ``set_determinism_env_vars`` uses
    #    ``setdefault`` so the NCCL_ALGO we just validated survives.
    set_determinism_env_vars()

    # 3. Override tp_comm_overlap. ``warn_rank_0`` (not print_rank_0) so the
    #    override is capturable via ``pytest.warns`` / ``-W error``.
    if args.tp_comm_overlap:
        # Lazy import — warn_rank_0 lives in training.utils which has heavier
        # dependencies than this module.
        from megatron.training.utils import warn_rank_0

        warn_rank_0("Disabling tp_comm_overlap for deterministic mode.")
        args.tp_comm_overlap = False

    # 4. Torch global state last — all assertions have already passed.
    torch.use_deterministic_algorithms(True)
