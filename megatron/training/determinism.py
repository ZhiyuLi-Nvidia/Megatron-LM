# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Reusable helpers for enabling bit-exact-reproducible execution.

Two entry points:

* :func:`apply_determinism_env` — validate env-var settings and setdefault
  the canonical values. Must run BEFORE the first cuBLAS / Transformer
  Engine kernel invocation.
* :func:`apply_determinism_to_args` — validate a parsed ``args`` Namespace,
  call :func:`apply_determinism_env` on ``os.environ``, and flip
  ``torch.use_deterministic_algorithms(True)``.
"""

from __future__ import annotations

import os
from typing import MutableMapping

import torch

# Maps each arg name to the value it must hold for bit-exact execution;
# verified by :func:`apply_determinism_to_args`.
ARG_VALUES_REQUIRED_FOR_DETERMINISM = {"cross_entropy_loss_fusion": False, "tp_comm_overlap": False}

# Env-var defaults required for bit-exact reproducibility.
DETERMINISM_ENV_VAR_DEFAULTS: dict[str, str] = {
    "NCCL_ALGO": "Ring",
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
}

# Accepted NCCL_ALGO tokens under --deterministic-mode. Comma-separated lists
# are valid NCCL syntax; every token in a launcher-supplied ``NCCL_ALGO`` must
# be in this set.
#   - ``Ring``          the default; bit-exact by construction, fully verified.
#   - ``CollnetDirect``,
#     ``CollnetChain``  verified bit-exact at smaller scale with SHARP
#                       in-network reduction (AllReduce and an end-to-end run).
#   - ``^NVLS``         excludes NVLS rather than selecting an algo, so NCCL
#                       falls back to whichever algo fits the hardware.
#                       Verified bit-exact in our setup; some risk remains
#                       because determinism then depends on that fallback algo.
#
# ``Tree`` is intentionally NOT accepted: its intra-node chain reduction
# order is not user-controllable and its multi-node inter-tree topology can
# vary across runs without a pinned topology file, so we cannot vouch for it.
ACCEPTED_NCCL_ALGO_TOKENS: frozenset[str] = frozenset({"Ring", "CollnetDirect", "CollnetChain", "^NVLS"})

# Env vars whose valid deterministic values are a small fixed exact-match set
# (unlike NCCL_ALGO which accepts comma-separated subsets of tokens). An unset
# value is fine -- apply_determinism_env() fills the canonical default. A
# set-but-invalid value fails hard.
#   - ``NVTE_ALLOW_NONDETERMINISTIC_ALGO``: TE reads it as ``int(value)``; only
#     ``"0"`` means deterministic (any nonzero int enables non-deterministic
#     algos). See ``megatron/core/extensions/transformer_engine.py``.
#   - ``CUBLAS_WORKSPACE_CONFIG``: NVIDIA docs list ``:4096:8`` (4x4MiB) and
#     ``:16:8`` (8x16KiB) as the two deterministic workspace configurations;
#     any other value breaks reproducibility.
ACCEPTED_ENV_VAR_VALUES: dict[str, frozenset[str]] = {
    "NVTE_ALLOW_NONDETERMINISTIC_ALGO": frozenset({"0"}),
    "CUBLAS_WORKSPACE_CONFIG": frozenset({":4096:8", ":16:8"}),
}

# Env vars whose library already auto-follows ``torch.are_deterministic_algorithms_enabled()``
# when they are unset, so leaving them alone is correct and only an explicit "off" needs
# rejecting. Value maps to what a non-'1' setting would silently turn off. Requiring a leading
# '1' is what makes an unrecognized value fail here rather than coast: the two libraries
# disagree on what to do with one. Mamba's reader treats anything not starting with '1' as OFF,
# while causal_conv1d only honours a leading '1' or '0' and otherwise falls through to torch.
# So `=true` would silently disable Mamba determinism and silently keep the conv's.
#   - ``MAMBA_DETERMINISTIC``: ``megatron/core/ssm/ops/common/determinism.py``.
#   - ``CAUSAL_CONV1D_DETERMINISTIC``: ``megatron/core/ssm/causal_conv1d.py``. causal_conv1d
#     >= 1.6.0 reduces the conv weight and bias gradients through a per-block workspace and an
#     ordered sum instead of atomicAdd. Needed by both SSM conv layouts, and needed at every
#     micro-batch size for the channel-last one.
AUTO_FOLLOWING_DETERMINISM_ENV_VARS: dict[str, str] = {
    "MAMBA_DETERMINISTIC": "Mamba SSM Triton determinism",
    "CAUSAL_CONV1D_DETERMINISTIC": "causal_conv1d convolution determinism",
}


def apply_determinism_env(env: MutableMapping[str, str]) -> None:
    """Validate every determinism env var in ``env``, then setdefault the canonical values.

    Semantics per key:

    * ``NCCL_ALGO`` — if set, each comma-separated token must be in
      :data:`ACCEPTED_NCCL_ALGO_TOKENS`.
    * ``NVTE_ALLOW_NONDETERMINISTIC_ALGO`` / ``CUBLAS_WORKSPACE_CONFIG`` —
      if set, must be in :data:`ACCEPTED_ENV_VAR_VALUES`.
    * every key of :data:`AUTO_FOLLOWING_DETERMINISM_ENV_VARS` — if set
      (non-empty), must start with ``'1'``; unset auto-follows
      :func:`torch.are_deterministic_algorithms_enabled`.

    After validation, ``setdefault`` fills every key in
    :data:`DETERMINISM_ENV_VAR_DEFAULTS` that has not been set — a value the
    caller has already set wins.

    These env vars are captured by their respective libraries at first use
    (NCCL at communicator init, cuBLAS at handle creation, TE at first
    attention forward), so the call must happen BEFORE any of those events.
    """
    # NCCL_ALGO subset check.
    nccl_algo = env.get("NCCL_ALGO")
    if nccl_algo is not None:
        tokens = [t.strip() for t in nccl_algo.split(",") if t.strip()]
        assert tokens and all(t in ACCEPTED_NCCL_ALGO_TOKENS for t in tokens), (
            f"NCCL_ALGO={nccl_algo!r}: each token must be in "
            f"{sorted(ACCEPTED_NCCL_ALGO_TOKENS)}."
        )

    # Exact-match env vars: reject only if the caller supplied a value we
    # haven't validated as deterministic; unset is fine.
    for name, accepted in ACCEPTED_ENV_VAR_VALUES.items():
        val = env.get(name)
        assert val is None or val in accepted, (
            f"{name}={val!r} is not a deterministic setting. Accepted: {sorted(accepted)}."
        )

    # These auto-follow torch when unset; only reject an explicit non-deterministic override.
    for name, what in AUTO_FOLLOWING_DETERMINISM_ENV_VARS.items():
        val = env.get(name)
        if val:
            assert val[0] == "1", (
                f"{name}={val!r} disables {what} under --deterministic-mode. "
                "Unset it or set to '1'."
            )

    # setdefault preserves any launcher-set value that just passed validation.
    for k, v in DETERMINISM_ENV_VAR_DEFAULTS.items():
        env.setdefault(k, v)


def apply_determinism_to_args(args) -> None:
    """Apply deterministic-mode requirements to a parsed-args Namespace.

    Idempotent. Performs (in this order):

    1. Asserts every option in ``ARG_VALUES_REQUIRED_FOR_DETERMINISM`` holds
       its required value. This is a verification-only check — it never
       mutates ``args``.
    2. Calls :func:`apply_determinism_env` on ``os.environ`` — validates
       every determinism-relevant env var (``NCCL_ALGO``, the keys of
       :data:`ACCEPTED_ENV_VAR_VALUES` and of
       :data:`AUTO_FOLLOWING_DETERMINISM_ENV_VARS`) and setdefaults the
       canonical values.
    3. Calls ``torch.use_deterministic_algorithms(True)``.

    Incompatible options are rejected with an explicit error rather than
    silently overridden: the user must turn them off themselves so the
    deterministic run matches the config they asked for.
    """
    # Verification only — read each option's effective value and never flip it,
    # so a default that drifts to a bad value breaks the run instead of silently
    # running non-deterministically.
    mismatched = [
        f"{name}={required!r} (got {actual!r})"
        for name, required in ARG_VALUES_REQUIRED_FOR_DETERMINISM.items()
        if (actual := getattr(args, name)) != required
    ]
    assert (
        not mismatched
    ), f"--deterministic-mode requires: {', '.join(mismatched)}. Adjust these options to continue."

    # --use-flash-attn is intentionally NOT rejected: TE's FlashAttention is
    # deterministic on supported configs and is covered by the bit-exact
    # correctness suite.

    # Delegate env-var validation + setdefault to the single-source helper.
    apply_determinism_env(os.environ)

    # Torch global state last — all assertions have already passed.
    torch.use_deterministic_algorithms(True)
