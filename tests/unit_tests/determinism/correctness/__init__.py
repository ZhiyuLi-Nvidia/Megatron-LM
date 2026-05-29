# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Bit-exact correctness tests for ``--deterministic-mode``.

Two-run comparison + parametrize over preset × parallelism. Separate from
the perf leaderboard tests (``tests/unit_tests/determinism/perf/``) so
correctness gating and perf measurement don't share a noise budget.

Always sets the determinism env vars on package import — correctness tests
have only one mode (``set``).
"""

from megatron.training.determinism import set_determinism_env_vars

# ``strict=True`` also pins ``NCCL_LAUNCH_RACE_FATAL=1`` so any latent
# collective-ordering bug surfaces as a hard test failure rather than a
# silent warning.
set_determinism_env_vars(strict=True)
