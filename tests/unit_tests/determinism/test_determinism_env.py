# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Env-var validation in ``megatron/training/determinism.py``.

Deliberately outside ``correctness/``, whose package import applies the determinism env to the
real ``os.environ``; these tests validate a throwaway mapping instead.
"""

import pytest

from megatron.training.determinism import (
    AUTO_FOLLOWING_DETERMINISM_ENV_VARS,
    DETERMINISM_ENV_VAR_DEFAULTS,
    apply_determinism_env,
)


@pytest.mark.parametrize("name", sorted(AUTO_FOLLOWING_DETERMINISM_ENV_VARS))
def test_auto_following_env_var_unset_is_accepted_and_left_alone(name):
    """Unset means "follow torch", which is what deterministic mode wants -- do not fill it in."""
    env = {}
    apply_determinism_env(env)
    assert name not in env


@pytest.mark.parametrize("name", sorted(AUTO_FOLLOWING_DETERMINISM_ENV_VARS))
def test_auto_following_env_var_rejects_an_explicit_off(name):
    """A launcher that switches one of these off would silently lose bit-exactness."""
    with pytest.raises(AssertionError, match=name):
        apply_determinism_env({name: "0"})


@pytest.mark.parametrize("name", sorted(AUTO_FOLLOWING_DETERMINISM_ENV_VARS))
def test_auto_following_env_var_accepts_an_explicit_on(name):
    apply_determinism_env({name: "1"})


def test_canonical_defaults_are_filled_but_do_not_override():
    env = {"NCCL_ALGO": "^NVLS"}
    apply_determinism_env(env)
    assert env["NCCL_ALGO"] == "^NVLS", "a validated launcher value must win over the default"
    for key, value in DETERMINISM_ENV_VAR_DEFAULTS.items():
        if key != "NCCL_ALGO":
            assert env[key] == value
