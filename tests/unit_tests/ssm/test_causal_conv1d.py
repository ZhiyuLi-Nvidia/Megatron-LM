# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

import pytest
import torch
import torch.distributed as dist

from megatron.core import parallel_state
from megatron.core.ssm import causal_conv1d as causal_conv1d_module
from megatron.core.ssm.causal_conv1d import (
    CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION,
    assert_causal_conv1d_deterministic,
)
from tests.unit_tests.test_utilities import Utils

try:
    from causal_conv1d import causal_conv1d_fn

    HAVE_CAUSAL_CONV1D = True
except ImportError:
    HAVE_CAUSAL_CONV1D = False

HAVE_DETERMINISTIC_CAUSAL_CONV1D = (
    HAVE_CAUSAL_CONV1D
    and causal_conv1d_module.is_causal_conv1d_min_version(CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION)
)


def _contiguous_slice(tensor, cp_rank, local_seq_len):
    return tensor[:, cp_rank * local_seq_len : (cp_rank + 1) * local_seq_len].contiguous()


@pytest.mark.internal
@pytest.mark.skipif(
    not HAVE_CAUSAL_CONV1D or not torch.cuda.is_available() or Utils.world_size < 2,
    reason="CP causal convolution parity requires causal-conv1d and at least two GPUs",
)
def test_causal_conv1d_cp_matches_full_sequence():
    Utils.initialize_model_parallel(context_parallel_size=Utils.world_size)
    try:
        cp_group = parallel_state.get_context_parallel_group()
        cp_size = dist.get_world_size(group=cp_group)
        cp_rank = dist.get_rank(group=cp_group)
        device = torch.device("cuda", torch.cuda.current_device())
        dtype = torch.float32
        rtol, atol = 3e-4, 1e-3
        local_seq_len = 64
        global_seq_len = cp_size * local_seq_len

        torch.manual_seed(1234)

        channels, width = 16, 4
        x_global = torch.randn(1, global_seq_len, channels, device=device, dtype=dtype)
        weight_global = torch.randn(channels, width, device=device, dtype=dtype)
        bias_global = torch.randn(channels, device=device, dtype=dtype)
        dy_global = torch.randn_like(x_global)

        x_ref = x_global.detach().clone().requires_grad_(True)
        weight_ref = weight_global.detach().clone().requires_grad_(True)
        bias_ref = bias_global.detach().clone().requires_grad_(True)
        output_ref = causal_conv1d_fn(
            x=x_ref.transpose(1, 2), weight=weight_ref, bias=bias_ref, activation="silu"
        ).transpose(1, 2)
        output_ref.backward(dy_global)

        x_local = _contiguous_slice(x_global, cp_rank, local_seq_len).detach().requires_grad_(True)
        weight_local = weight_global.detach().clone().requires_grad_(True)
        bias_local = bias_global.detach().clone().requires_grad_(True)
        output_local = causal_conv1d_module.causal_conv1d_cp(
            x=x_local, weight=weight_local, bias=bias_local, activation="silu", cp_group=cp_group
        )
        output_local.backward(_contiguous_slice(dy_global, cp_rank, local_seq_len))
        dist.all_reduce(weight_local.grad, group=cp_group)
        dist.all_reduce(bias_local.grad, group=cp_group)

        expected_output = _contiguous_slice(output_ref, cp_rank, local_seq_len)
        expected_dx = _contiguous_slice(x_ref.grad, cp_rank, local_seq_len)
        torch.testing.assert_close(output_local, expected_output, rtol=rtol, atol=atol)
        torch.testing.assert_close(x_local.grad, expected_dx, rtol=rtol, atol=atol)
        torch.testing.assert_close(weight_local.grad, weight_ref.grad, rtol=rtol, atol=atol)
        torch.testing.assert_close(bias_local.grad, bias_ref.grad, rtol=rtol, atol=atol)
    finally:
        Utils.destroy_model_parallel()


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal_conv1d is not installed")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_causal_conv1d_channel_contiguous_matches_sequence_contiguous(dtype):
    """Verify forward and backward equivalence across causal_conv1d layouts.

    Both inputs have shape (batch, dim, seqlen) and identical values. Only
    their strides differ: a unit dim stride selects the channel-last kernel,
    while a unit seqlen stride selects the standard kernel.
    """
    torch.manual_seed(42)
    batch, seq_len, dim, width = 2, 128, 64, 4
    padding = 16
    storage_dim = dim + 2 * padding

    # Slice a larger allocation so the channel-last input is non-dense and
    # has gaps between consecutive sequence elements.
    channel_last_storage = torch.randn(batch, seq_len, storage_dim, device="cuda", dtype=dtype)
    x_channel_contiguous = (
        channel_last_storage[:, :, padding : padding + dim]
        .transpose(1, 2)
        .detach()
        .requires_grad_()
    )
    x_sequence_contiguous = x_channel_contiguous.detach().contiguous().requires_grad_()

    assert x_channel_contiguous.shape == x_sequence_contiguous.shape == (batch, dim, seq_len)
    assert torch.equal(x_channel_contiguous, x_sequence_contiguous)
    assert x_channel_contiguous.stride() == (seq_len * storage_dim, 1, storage_dim)
    assert x_sequence_contiguous.stride() == (dim * seq_len, seq_len, 1)

    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(dim, device="cuda", dtype=torch.float32, requires_grad=True)

    out_channel_contiguous = causal_conv1d_fn(x_channel_contiguous, weight, bias, activation="silu")
    out_sequence_contiguous = causal_conv1d_fn(
        x_sequence_contiguous, weight, bias, activation="silu"
    )

    assert out_channel_contiguous.stride() == (seq_len * dim, 1, dim)
    assert out_sequence_contiguous.stride() == (dim * seq_len, seq_len, 1)

    # The forward kernels should be bitwise identical.
    assert torch.equal(out_channel_contiguous, out_sequence_contiguous)

    grad_channel_contiguous = torch.randn_like(out_channel_contiguous)
    grad_sequence_contiguous = grad_channel_contiguous.contiguous()
    assert torch.equal(grad_channel_contiguous, grad_sequence_contiguous)
    assert grad_channel_contiguous.stride() == (seq_len * dim, 1, dim)
    assert grad_sequence_contiguous.stride() == (dim * seq_len, seq_len, 1)

    dx_channel, dweight_channel, dbias_channel = torch.autograd.grad(
        out_channel_contiguous,
        (x_channel_contiguous, weight, bias),
        grad_outputs=grad_channel_contiguous,
    )
    dx_sequence, dweight_sequence, dbias_sequence = torch.autograd.grad(
        out_sequence_contiguous,
        (x_sequence_contiguous, weight, bias),
        grad_outputs=grad_sequence_contiguous,
    )

    assert dx_channel.stride() == (seq_len * dim, 1, dim)
    assert dx_sequence.stride() == (dim * seq_len, seq_len, 1)

    # dx is returned in the input dtype, so BF16 needs looser tolerances. Weight
    # and bias gradients are accumulated in FP32; their small differences come
    # from the kernels using different parallel reduction orders.
    input_grad_rtol, input_grad_atol = (3e-4, 1e-3) if dtype == torch.float32 else (1e-2, 5e-2)
    param_grad_rtol, param_grad_atol = 1e-3, 1e-3

    torch.testing.assert_close(dx_channel, dx_sequence, rtol=input_grad_rtol, atol=input_grad_atol)
    torch.testing.assert_close(
        dweight_channel, dweight_sequence, rtol=param_grad_rtol, atol=param_grad_atol
    )
    torch.testing.assert_close(
        dbias_channel, dbias_sequence, rtol=param_grad_rtol, atol=param_grad_atol
    )


GRAD_NAMES = ("dx", "dweight", "dbias")


def _channel_last_conv_inputs(batch, dim, seq_len, width):
    """Build a channel-last [B, D, L] conv input plus weight, bias and an output gradient."""
    torch.manual_seed(7)
    x = (
        torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.bfloat16)
        .transpose(1, 2)
        .detach()
        .requires_grad_()
    )
    assert x.stride(1) == 1, "these tests are about the channel-last kernel"
    weight = torch.randn(dim, width, device="cuda", dtype=torch.float32, requires_grad=True)
    bias = torch.randn(dim, device="cuda", dtype=torch.float32, requires_grad=True)
    grad = torch.randn(batch, seq_len, dim, device="cuda", dtype=torch.bfloat16).transpose(1, 2)
    return x, weight, bias, grad


def _conv_backward(x, weight, bias, grad):
    out = causal_conv1d_fn(x=x, weight=weight, bias=bias, activation="silu")
    return torch.autograd.grad(out, (x, weight, bias), grad_outputs=grad)


def _replay_channel_last_backward(batch, dim, seq_len, width, replays):
    """Run one channel-last backward `replays` times from identical inputs.

    Returns, per gradient, how many replays differed from the first bitwise.
    """
    inputs = _channel_last_conv_inputs(batch, dim, seq_len, width)
    first, differing = None, dict.fromkeys(GRAD_NAMES, 0)
    for _ in range(replays):
        got = _conv_backward(*inputs)
        if first is None:
            first = got
            continue
        for name, ref, cur in zip(GRAD_NAMES, first, got):
            differing[name] += not torch.equal(ref, cur)
    return differing


@pytest.mark.skipif(
    not HAVE_DETERMINISTIC_CAUSAL_CONV1D,
    reason=f"needs causal_conv1d >= {CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION}",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("deterministic", [False, True])
def test_channel_last_conv1d_backward_replays_bitwise(monkeypatch, deterministic):
    """The channel-last backward replays bit-for-bit only under the deterministic reduction.

    Both halves matter. ``deterministic=True`` is the property GDP's channel-last conv relies
    on; ``deterministic=False`` is its control, without which a build that quietly ignored
    ``CAUSAL_CONV1D_DETERMINISTIC`` would still pass. The control is an assertion about the
    *hardware's* scheduling, so it is sized with margin: the default path reduces each
    ``dweight`` element with ``atomicAdd`` over ``batch * ceil(seq_len / 128)`` blocks -- 64 of
    them here -- and 19 of 19 replays differed on GB300 at a shape with only 16. ``dx`` is
    written by disjoint stores, so it is reproducible either way.

    The env var is read by ``getenv`` on each backward call, so setting it here is enough; it
    does not have to be in place before the extension loads.
    """
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1" if deterministic else "0")
    replays = 8
    differing = _replay_channel_last_backward(
        batch=2, dim=1024, seq_len=4096, width=4, replays=replays
    )

    assert differing["dx"] == 0
    if deterministic:
        assert differing["dweight"] == 0 and differing["dbias"] == 0
    else:
        assert differing["dweight"] > 0, (
            f"the default channel-last backward was bit-reproducible over {replays} replays. "
            "Either upstream made it deterministic -- in which case drop this control -- or "
            "this GPU serialized the contending blocks, which the shape above is meant to "
            "prevent."
        )


@pytest.mark.skipif(
    not HAVE_DETERMINISTIC_CAUSAL_CONV1D,
    reason=f"needs causal_conv1d >= {CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION}",
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_deterministic_reduction_agrees_with_the_default_one(monkeypatch):
    """The deterministic path must reorder the same sum, not compute a different one.

    Both paths derive identical per-block partials -- the kernel branches only on where a
    partial is written, an `atomicAdd` into `dweight` versus a slot of its own in a workspace
    that is summed afterwards. So the two results may differ, but only by fp32 accumulation
    error: anything larger would mean the deterministic branch changed the arithmetic, which
    no amount of bitwise self-consistency would catch.

    Bitwise equality is the wrong bar here, since fp32 addition is not associative. The bar is
    the gap relative to each tensor's own magnitude: measured at 5.6e-07 for `dweight` and
    4.3e-07 for `dbias` on GB300 over batch 1-4 and seqlen 4096-8192, a few fp32 ulps (eps is
    1.2e-07). The bound below is 1e-5, roughly 18x that.

    Deliberately `rtol=0` with the whole tensor's magnitude in `atol`, not a per-element
    relative bound. Per element the gap reaches ~1e-2, because a `dweight` entry near zero is
    a near-total cancellation of much larger products, so its relative error is large while
    its absolute error stays at the same few-ulp level. The cost of that choice is that this
    would not catch a path corrupting only entries far below the tensor maximum; see
    `docs/developer/determinism/causal-conv1d-overhead.md`.

    `dx` never goes through the reduction and is expected to match bitwise.
    """
    inputs = _channel_last_conv_inputs(batch=2, dim=1024, seq_len=4096, width=4)

    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "0")
    default = _conv_backward(*inputs)
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    deterministic = _conv_backward(*inputs)

    for name, ref, got in zip(GRAD_NAMES, default, deterministic):
        if name == "dx":
            assert torch.equal(ref, got), "dx does not go through the reduction"
            continue
        torch.testing.assert_close(
            got,
            ref,
            rtol=0,
            atol=float(ref.abs().max()) * 1e-5,
            msg=f"{name} moved by more than fp32 accumulation error",
        )


@pytest.mark.skipif(not HAVE_CAUSAL_CONV1D, reason="causal_conv1d is not installed")
def test_assert_causal_conv1d_deterministic_rejects_a_disabled_kernel(monkeypatch):
    """deterministic_mode must not run against a conv that was explicitly switched off."""
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "0")
    with pytest.raises(AssertionError, match="deterministic causal_conv1d backward"):
        assert_causal_conv1d_deterministic(deterministic_mode=True)
    # Nothing to enforce when the run never asked for determinism.
    assert_causal_conv1d_deterministic(deterministic_mode=False)


@pytest.mark.skipif(
    not HAVE_DETERMINISTIC_CAUSAL_CONV1D,
    reason=f"needs causal_conv1d >= {CAUSAL_CONV1D_DETERMINISTIC_MIN_VERSION}",
)
def test_assert_causal_conv1d_deterministic_accepts_an_enabled_kernel(monkeypatch):
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    assert_causal_conv1d_deterministic(deterministic_mode=True)


def test_assert_causal_conv1d_deterministic_ignores_torch_flag_alone(monkeypatch):
    """Torch's global flag is not a request for Megatron determinism, so it must not gate here.

    Several unit tests call ``torch.use_deterministic_algorithms(True)`` and never restore it.
    Treating that as a licence to reject an older causal_conv1d would fail model construction
    for runs that never asked for bit-exactness.

    The two fakes are what give this teeth: they are exactly the state that makes the version
    check fire, so the paired ``deterministic_mode=True`` call below must raise. Without the
    early return the first call would raise too.
    """
    monkeypatch.delenv("CAUSAL_CONV1D_DETERMINISTIC", raising=False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        causal_conv1d_module, "is_causal_conv1d_min_version", lambda *_args, **_kwargs: False
    )

    assert_causal_conv1d_deterministic(deterministic_mode=False)

    with pytest.raises(AssertionError, match="no deterministic backward"):
        assert_causal_conv1d_deterministic(deterministic_mode=True)


def test_assert_causal_conv1d_deterministic_accepts_the_torch_flag_path(monkeypatch):
    """The configuration ``--deterministic-mode`` actually produces must pass.

    It leaves ``CAUSAL_CONV1D_DETERMINISTIC`` unset and calls
    ``torch.use_deterministic_algorithms(True)``, so the guard has to resolve the kernel's mode
    through the torch fallback rather than the env var. Every other test here sets the env var
    explicitly, which would leave that fallback -- the production path -- unexercised: a
    regression making it return False would abort every deterministic run at mixer
    construction.
    """
    monkeypatch.delenv("CAUSAL_CONV1D_DETERMINISTIC", raising=False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        causal_conv1d_module, "is_causal_conv1d_min_version", lambda *_args, **_kwargs: True
    )
    assert_causal_conv1d_deterministic(deterministic_mode=True)


def test_assert_causal_conv1d_deterministic_rejects_the_torch_flag_being_off(monkeypatch):
    """deterministic_mode without torch's flag and without the env var is not deterministic."""
    monkeypatch.delenv("CAUSAL_CONV1D_DETERMINISTIC", raising=False)
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: False)
    with pytest.raises(AssertionError, match="deterministic causal_conv1d backward"):
        assert_causal_conv1d_deterministic(deterministic_mode=True)


def test_assert_causal_conv1d_deterministic_rejects_an_old_kernel(monkeypatch):
    """An install predating the deterministic reduction fails closed rather than silently.

    Faked rather than installed: the point is the branch, and the version is the only input.
    """
    monkeypatch.setenv("CAUSAL_CONV1D_DETERMINISTIC", "1")
    monkeypatch.setattr(
        causal_conv1d_module, "is_causal_conv1d_min_version", lambda *_args, **_kwargs: False
    )
    with pytest.raises(AssertionError, match="no deterministic backward"):
        assert_causal_conv1d_deterministic(deterministic_mode=True)
