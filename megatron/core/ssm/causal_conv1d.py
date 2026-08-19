# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Megatron's use of the causal_conv1d extension.

Two things: the convolution over contiguous context-parallel sequence shards, and the
determinism guard the SSM mixers apply before they rely on that extension's backward.
"""

import os

import torch

from megatron.core.tensor_parallel.mappings import all_to_all
from megatron.core.utils import is_causal_conv1d_min_version

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None


# causal_conv1d accumulates dweight/dbias across thread blocks with atomicAdd, so their
# last-place rounding follows block completion order. 1.6.0 added a deterministic reduction
# (Dao-AILab/causal-conv1d#88): a zero-initialized per-block workspace plus an ordered sum.
CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION = "1.6.0"


def _use_causal_conv1d_deterministic_mode():
    """Whether causal_conv1d's backward will take its deterministic reduction.

    Mirrors the kernel's own ``use_deterministic_mode()`` (``csrc/causal_conv1d.cpp``)::

        const char* env = std::getenv("CAUSAL_CONV1D_DETERMINISTIC");
        if (env) {
            if (*env == '1') return true;
            if (*env == '0') return false;
        }
        return at::globalContext().deterministicAlgorithms();

    So only a leading ``'1'`` or ``'0'`` decides; anything else falls through to torch rather
    than counting as off. ``getenv`` runs on every backward call, so this is toggleable at
    runtime -- do not cache it.
    """
    env = os.environ.get('CAUSAL_CONV1D_DETERMINISTIC')
    if env:
        if env[0] == '1':
            return True
        if env[0] == '0':
            return False
    return torch.are_deterministic_algorithms_enabled()


def assert_causal_conv1d_deterministic(deterministic_mode):
    """Fail closed rather than let a deterministic run get a non-reproducible convolution.

    Two ways that can happen, and this rejects both: the deterministic path was never enabled
    (``--deterministic-mode`` calls ``torch.use_deterministic_algorithms(True)``, so this
    catches a hand-built config or a launcher that set ``CAUSAL_CONV1D_DETERMINISTIC=0``), and
    the installed causal_conv1d predates that path.

    It matters most for the channel-last layout, which is what GDP and Mamba's fused
    ``mamba_split_conv1d_scan_combined`` both feed the conv: the channel-last backward tiles
    the sequence as well as the batch, so each ``dweight`` element takes
    ``batch * ceil(seqlen / 128)`` atomicAdds -- the tile is 64 below seqlen 128, 128 at any
    training length -- instead of the channels-first kernel's ``batch``,
    and the drift grows with the contributor count. Channels-first -- Mamba's ``_static_prefill``
    and the inference paths -- is reproducible at micro-batch 1 and not beyond it.

    Scoped to ``deterministic_mode`` on purpose. Torch's global flag gets flipped for unrelated
    reasons -- several unit tests set it and never restore it -- and taking that as a licence to
    reject an older causal_conv1d would fail runs that never asked Megatron for bit-exactness.
    Call once at mixer construction, not per step.
    """
    if not deterministic_mode:
        return

    assert _use_causal_conv1d_deterministic_mode(), (
        "deterministic_mode requires a deterministic causal_conv1d backward. Enable it with "
        "torch.use_deterministic_algorithms(True) (which --deterministic-mode does) or "
        "CAUSAL_CONV1D_DETERMINISTIC=1."
    )
    assert is_causal_conv1d_min_version(CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION), (
        f"causal_conv1d < {CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION} has no deterministic "
        "backward: it reduces the conv weight and bias gradients with atomicAdd, which is "
        "not bit-reproducible. Upgrade causal_conv1d."
    )


def _exchange_initial_states(
    x: torch.Tensor, state_len: int, cp_group: torch.distributed.ProcessGroup
) -> torch.Tensor | None:
    """Exchange the preceding rank's tail as the local convolution state.

    All ranks participate in a differentiable ring exchange. Rank 0 zeros the
    wrapped tail to preserve the global causal boundary.
    """
    if state_len < 0:
        raise ValueError(f"state_len must be non-negative, got {state_len}")
    if state_len == 0 or cp_group.size() == 1:
        return None
    if x.shape[1] < state_len:
        raise ValueError(
            "Each local sequence shard must contain at least "
            f"{state_len} tokens for causal convolution, got {x.shape[1]}"
        )

    cp_size = cp_group.size()
    cp_rank = cp_group.rank()
    batch_size, _, channels = x.shape
    split_size = batch_size * state_len

    # Pack only the boundary tokens; x remains a strided sequence shard.
    tail = x[:, -state_len:, :].reshape(split_size, channels)
    input_splits = [0] * cp_size
    output_splits = [0] * cp_size
    input_splits[(cp_rank + 1) % cp_size] = split_size
    output_splits[(cp_rank - 1) % cp_size] = split_size
    previous_tail = all_to_all(
        cp_group, tail, output_split_sizes_=output_splits, input_split_sizes=input_splits
    ).view(batch_size, state_len, channels)

    if cp_rank == 0:
        # Preserve the autograd path while enforcing the global left boundary.
        previous_tail = previous_tail.clone()
        previous_tail.zero_()
    return previous_tail.transpose(1, 2)


def causal_conv1d_cp(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    activation: str | None,
    cp_group: torch.distributed.ProcessGroup,
) -> torch.Tensor:
    """Apply causal Conv1d to a contiguous context-parallel shard.

    Args:
        x: Input tensor of shape ``[B, T, D]``.
        weight: Depthwise weights of shape ``[D, W]``.
        bias: Optional channel-wise bias.
        activation: Optional activation passed to ``causal_conv1d_fn``.
        cp_group: Context-parallel process group ordered by sequence shard.

    Returns:
        Output tensor of shape ``[B, T, D]``.

    Raises:
        ImportError: If the optional ``causal-conv1d`` dependency is unavailable.
    """
    if causal_conv1d_fn is None:
        raise ImportError("causal_conv1d_cp requires the optional causal-conv1d dependency")

    initial_states = _exchange_initial_states(
        x=x, state_len=weight.shape[-1] - 1, cp_group=cp_group
    )
    output = causal_conv1d_fn(
        x=x.transpose(1, 2),
        weight=weight,
        bias=bias,
        initial_states=initial_states,
        activation=activation,
    )
    return output.transpose(1, 2)
